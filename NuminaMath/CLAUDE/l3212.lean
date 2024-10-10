import Mathlib

namespace joaozinho_meeting_day_l3212_321285

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Theorem statement
theorem joaozinho_meeting_day :
  ∀ (meeting_day : Day),
    (lies_on_day meeting_day →
      (meeting_day ≠ Day.Saturday ∧
       next_day meeting_day ≠ Day.Wednesday)) →
    meeting_day = Day.Thursday :=
by
  sorry


end joaozinho_meeting_day_l3212_321285


namespace divisibility_condition_l3212_321207

theorem divisibility_condition (m : ℕ+) :
  (∀ k : ℕ, k ≥ 3 → Odd k → (k^(m : ℕ) - 1) % 2^(m : ℕ) = 0) ↔ m = 1 ∨ m = 2 ∨ m = 4 := by
  sorry

end divisibility_condition_l3212_321207


namespace sine_increasing_omega_range_l3212_321250

/-- Given that y = sin(ωx) is increasing on the interval [-π/3, π/3], 
    the range of values for ω is (0, 3/2]. -/
theorem sine_increasing_omega_range (ω : ℝ) : 
  (∀ x ∈ Set.Icc (-π/3) (π/3), 
    Monotone (fun x => Real.sin (ω * x))) → 
  ω ∈ Set.Ioo 0 (3/2) :=
sorry

end sine_increasing_omega_range_l3212_321250


namespace choose_service_providers_and_accessories_l3212_321212

def total_individuals : ℕ := 4
def total_service_providers : ℕ := 25
def total_accessories : ℕ := 5

def ways_to_choose : ℕ := (total_service_providers - 0) *
                           (total_service_providers - 1) *
                           (total_service_providers - 2) *
                           (total_service_providers - 3) *
                           (total_accessories - 0) *
                           (total_accessories - 1) *
                           (total_accessories - 2) *
                           (total_accessories - 3)

theorem choose_service_providers_and_accessories :
  ways_to_choose = 36432000 :=
sorry

end choose_service_providers_and_accessories_l3212_321212


namespace student_number_problem_l3212_321263

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end student_number_problem_l3212_321263


namespace remainder_31_pow_31_plus_31_mod_32_l3212_321246

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end remainder_31_pow_31_plus_31_mod_32_l3212_321246


namespace apples_to_pears_ratio_l3212_321258

/-- Represents the contents of a shopping cart --/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Defines the relationships between fruit quantities in the shopping cart --/
def validCart (cart : ShoppingCart) : Prop :=
  cart.oranges = 2 * cart.apples ∧
  cart.pears = 5 * cart.oranges ∧
  cart.bananas = 3 * cart.pears ∧
  cart.peaches = cart.bananas / 2

/-- Theorem stating that apples are 1/10 of pears in a valid shopping cart --/
theorem apples_to_pears_ratio (cart : ShoppingCart) (h : validCart cart) :
  cart.apples = cart.pears / 10 := by
  sorry


end apples_to_pears_ratio_l3212_321258


namespace school_students_l3212_321297

def total_students (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  (n * (2 * largest_class - (n - 1) * diff)) / 2

theorem school_students :
  total_students 5 24 2 = 100 := by
  sorry

end school_students_l3212_321297


namespace exponent_multiplication_l3212_321275

theorem exponent_multiplication (x : ℝ) (a b : ℕ) : x^a * x^b = x^(a + b) := by
  sorry

end exponent_multiplication_l3212_321275


namespace remaining_water_l3212_321253

/-- 
Given an initial amount of water and an amount used, 
calculate the remaining amount of water.
-/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) :
  initial = 4 →
  used = 9/4 →
  remaining = initial - used →
  remaining = 7/4 := by
  sorry

end remaining_water_l3212_321253


namespace bobby_candy_chocolate_difference_l3212_321262

/-- Given the number of candy pieces Bobby ate initially and additionally,
    as well as the number of chocolate pieces, prove that Bobby ate 58 more
    pieces of candy than chocolate. -/
theorem bobby_candy_chocolate_difference
  (initial_candy : ℕ)
  (additional_candy : ℕ)
  (chocolate : ℕ)
  (h1 : initial_candy = 38)
  (h2 : additional_candy = 36)
  (h3 : chocolate = 16) :
  initial_candy + additional_candy - chocolate = 58 := by
  sorry

end bobby_candy_chocolate_difference_l3212_321262


namespace unit_digit_of_2_power_2024_l3212_321264

theorem unit_digit_of_2_power_2024 (unit_digit : ℕ → ℕ) (h : ∀ n : ℕ, unit_digit (2^n) = unit_digit (2^(n % 4))) : unit_digit (2^2024) = 6 := by
  sorry

end unit_digit_of_2_power_2024_l3212_321264


namespace set_operation_result_l3212_321266

def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {0, 1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 3, 4} := by sorry

end set_operation_result_l3212_321266


namespace paint_left_calculation_paint_problem_solution_l3212_321265

/-- Given the total amount of paint needed and the amount of paint to buy,
    calculate the amount of paint left from the previous project. -/
theorem paint_left_calculation (total_paint : ℕ) (paint_to_buy : ℕ) :
  total_paint ≥ paint_to_buy →
  total_paint - paint_to_buy = total_paint - paint_to_buy :=
by
  sorry

/-- The specific problem instance -/
def paint_problem : ℕ × ℕ := (333, 176)

/-- The solution to the specific problem instance -/
theorem paint_problem_solution :
  let (total_paint, paint_to_buy) := paint_problem
  total_paint - paint_to_buy = 157 :=
by
  sorry

end paint_left_calculation_paint_problem_solution_l3212_321265


namespace solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l3212_321274

-- Define the inequality
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Theorem for k = 3
theorem solution_for_k_3 :
  ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < 2/3 :=
sorry

-- Theorems for k < 0
theorem solution_for_k_neg_2 :
  ∀ x : ℝ, inequality (-2) x ↔ x ≠ -1 :=
sorry

theorem solution_for_k_lt_neg_2 :
  ∀ k x : ℝ, k < -2 → (inequality k x ↔ x < -1 ∨ x > 2/k) :=
sorry

theorem solution_for_k_between_neg_2_and_0 :
  ∀ k x : ℝ, -2 < k ∧ k < 0 → (inequality k x ↔ x > -1 ∨ x < 2/k) :=
sorry

end solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l3212_321274


namespace min_value_expression_l3212_321283

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2) / c^2 ≥ 2 ∧
  ∃ a' b' c', c' > b' ∧ b' > a' ∧ c' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' - a')^2) / c'^2 = 2 :=
by sorry

end min_value_expression_l3212_321283


namespace least_number_of_candles_l3212_321229

theorem least_number_of_candles (b : ℕ) : 
  b > 0 ∧ 
  b % 6 = 5 ∧ 
  b % 8 = 7 ∧ 
  b % 9 = 3 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 6 = 5 ∧ c % 8 = 7 ∧ c % 9 = 3 → c ≥ b) → 
  b = 119 :=
sorry

end least_number_of_candles_l3212_321229


namespace roots_quadratic_equation_l3212_321222

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 2*α - 1 = 0) → 
  (β^2 - 2*β - 1 = 0) → 
  (4 * α^3 + 5 * β^4 = -40*α + 153) := by
  sorry

end roots_quadratic_equation_l3212_321222


namespace fixed_point_on_line_l3212_321214

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a + 2) * x + (1 - a) * y - 3 = 0

-- Theorem statement
theorem fixed_point_on_line (a : ℝ) (h : a ≠ 0) : 
  line_equation a 1 1 := by sorry

end fixed_point_on_line_l3212_321214


namespace square_perimeter_l3212_321220

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 900) (h2 : side * side = area) : 
  4 * side = 120 := by
  sorry

end square_perimeter_l3212_321220


namespace intersection_points_count_l3212_321203

/-- The number of distinct intersection points for the given equations -/
def num_intersection_points : ℕ :=
  let eq1 := fun (x y : ℝ) => (x - y + 2) * (2 * x + 3 * y - 6) = 0
  let eq2 := fun (x y : ℝ) => (3 * x - 2 * y - 1) * (x + 2 * y - 4) = 0
  2

/-- Theorem stating that the number of distinct intersection points is 2 -/
theorem intersection_points_count :
  num_intersection_points = 2 := by sorry

end intersection_points_count_l3212_321203


namespace even_function_derivative_is_odd_l3212_321298

theorem even_function_derivative_is_odd 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_deriv : ∀ x, HasDerivAt f (g x) x) : 
  ∀ x, g (-x) = -g x := by sorry

end even_function_derivative_is_odd_l3212_321298


namespace expression_value_at_x_2_l3212_321221

theorem expression_value_at_x_2 :
  let x : ℝ := 2
  (3 * x + 4)^2 - 10 * x = 80 := by
  sorry

end expression_value_at_x_2_l3212_321221


namespace brooklyn_annual_donation_l3212_321256

/-- Brooklyn's monthly donation in dollars -/
def monthly_donation : ℕ := 1453

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Brooklyn's total donation in a year -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem brooklyn_annual_donation : annual_donation = 17436 := by
  sorry

end brooklyn_annual_donation_l3212_321256


namespace sum_of_digits_theorem_l3212_321272

def decimal_digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_theorem : 
  decimal_digit_sum (2^2007 * 5^2005 * 7) = 10 := by sorry

end sum_of_digits_theorem_l3212_321272


namespace expression_value_l3212_321219

theorem expression_value : 105^3 - 3 * 105^2 + 3 * 105 - 1 = 1124864 := by
  sorry

end expression_value_l3212_321219


namespace plant_sales_net_profit_l3212_321290

-- Define the costs
def basil_seed_cost : ℚ := 2
def mint_seed_cost : ℚ := 3
def zinnia_seed_cost : ℚ := 7
def potting_soil_cost : ℚ := 15

-- Define the number of plants per packet
def basil_plants_per_packet : ℕ := 20
def mint_plants_per_packet : ℕ := 15
def zinnia_plants_per_packet : ℕ := 10

-- Define the germination rates
def basil_germination_rate : ℚ := 4/5
def mint_germination_rate : ℚ := 3/4
def zinnia_germination_rate : ℚ := 7/10

-- Define the selling prices
def healthy_basil_price : ℚ := 5
def small_basil_price : ℚ := 3
def healthy_mint_price : ℚ := 6
def small_mint_price : ℚ := 4
def healthy_zinnia_price : ℚ := 10
def small_zinnia_price : ℚ := 7

-- Define the number of plants sold
def healthy_basil_sold : ℕ := 12
def small_basil_sold : ℕ := 8
def healthy_mint_sold : ℕ := 10
def small_mint_sold : ℕ := 4
def healthy_zinnia_sold : ℕ := 5
def small_zinnia_sold : ℕ := 2

-- Define the total cost
def total_cost : ℚ := basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost

-- Define the total revenue
def total_revenue : ℚ := 
  healthy_basil_price * healthy_basil_sold +
  small_basil_price * small_basil_sold +
  healthy_mint_price * healthy_mint_sold +
  small_mint_price * small_mint_sold +
  healthy_zinnia_price * healthy_zinnia_sold +
  small_zinnia_price * small_zinnia_sold

-- Define the net profit
def net_profit : ℚ := total_revenue - total_cost

-- Theorem to prove
theorem plant_sales_net_profit : net_profit = 197 := by sorry

end plant_sales_net_profit_l3212_321290


namespace sperner_theorem_l3212_321206

/-- The largest number of subsets of an n-element set such that no subset is contained in any other -/
def largestSperner (n : ℕ) : ℕ :=
  Nat.choose n (n / 2)

/-- Sperner's theorem -/
theorem sperner_theorem (n : ℕ) :
  largestSperner n = Nat.choose n (n / 2) :=
sorry

end sperner_theorem_l3212_321206


namespace quadratic_rewrite_ratio_l3212_321251

theorem quadratic_rewrite_ratio (b c : ℝ) :
  (∀ x, x^2 + 1300*x + 1300 = (x + b)^2 + c) →
  c / b = -648 := by
sorry

end quadratic_rewrite_ratio_l3212_321251


namespace canoe_trip_average_speed_l3212_321284

/-- Proves that the average distance per day for the remaining days of a canoe trip is 32 km/day -/
theorem canoe_trip_average_speed
  (total_distance : ℝ)
  (total_days : ℕ)
  (completed_days : ℕ)
  (completed_fraction : ℚ)
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_days = 3)
  (h4 : completed_fraction = 3/7)
  : (total_distance - completed_fraction * total_distance) / (total_days - completed_days : ℝ) = 32 := by
  sorry

#check canoe_trip_average_speed

end canoe_trip_average_speed_l3212_321284


namespace scaled_triangle_area_is_32_l3212_321281

/-- The area of a triangle with vertices at (0,0), (-3, 7), and (-7, 3), scaled by a factor of 2 -/
def scaledTriangleArea : ℝ := 32

/-- The scaling factor -/
def scalingFactor : ℝ := 2

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℝ × ℝ) := [(0, 0), (-3, 7), (-7, 3)]

/-- Theorem: The area of the scaled triangle is 32 square units -/
theorem scaled_triangle_area_is_32 :
  scaledTriangleArea = 32 :=
by sorry

end scaled_triangle_area_is_32_l3212_321281


namespace prime_sum_gcd_ratio_composite_sum_gcd_ratio_l3212_321236

-- Part 1
theorem prime_sum_gcd_ratio (n : ℕ) (hn : Nat.Prime (2 * n - 1)) :
  ∀ (a : Fin n → ℕ), Function.Injective a →
  ∃ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) ≥ 2 * n - 1 := by sorry

-- Part 2
theorem composite_sum_gcd_ratio (n : ℕ) (hn : ¬Nat.Prime (2 * n - 1)) (hn2 : 2 * n - 1 > 1) :
  ∃ (a : Fin n → ℕ), Function.Injective a ∧
  ∀ i j : Fin n, (a i + a j : ℚ) / Nat.gcd (a i) (a j) < 2 * n - 1 := by sorry

end prime_sum_gcd_ratio_composite_sum_gcd_ratio_l3212_321236


namespace zoo_animal_count_l3212_321267

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements --/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures := 4
  let tigers_per_enclosure := 4
  let zebra_enclosures := (tiger_enclosures / 2) * 3
  let zebras_per_enclosure := 10
  let elephant_giraffe_pattern_repetitions := 4
  let elephants_per_enclosure := 3
  let giraffes_per_enclosure := 2
  let rhino_enclosures := 5
  let rhinos_per_enclosure := 1
  let chimpanzee_enclosures := rhino_enclosures * 2
  let chimpanzees_per_enclosure := 8

  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := zebra_enclosures * zebras_per_enclosure
  let total_elephants := elephant_giraffe_pattern_repetitions * elephants_per_enclosure
  let total_giraffes := elephant_giraffe_pattern_repetitions * 2 * giraffes_per_enclosure
  let total_rhinos := rhino_enclosures * rhinos_per_enclosure
  let total_chimpanzees := chimpanzee_enclosures * chimpanzees_per_enclosure

  total_tigers + total_zebras + total_elephants + total_giraffes + total_rhinos + total_chimpanzees

theorem zoo_animal_count : total_animals_in_zoo = 189 := by
  sorry

end zoo_animal_count_l3212_321267


namespace square_of_integer_l3212_321295

theorem square_of_integer (x y : ℤ) (h : x + y = 10^18) :
  (x^2 * y^2) + ((x^2 + y^2) * (x + y)^2) = (x*y + x^2 + y^2)^2 := by
  sorry

end square_of_integer_l3212_321295


namespace circular_seating_arrangement_l3212_321252

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people who must sit together -/
def must_sit_together : ℕ := 2

/-- The number of people available to sit next to the fixed person -/
def available_neighbors : ℕ := total_people - must_sit_together - 1

/-- The number of neighbors to choose -/
def neighbors_to_choose : ℕ := 2

theorem circular_seating_arrangement :
  Nat.choose available_neighbors neighbors_to_choose = 10 := by
  sorry

end circular_seating_arrangement_l3212_321252


namespace age_ratio_problem_l3212_321241

/-- Given that:
    1. X's current age is 45
    2. Three years ago, X's age was some multiple of Y's age
    3. Seven years from now, the sum of their ages will be 83 years
    Prove that the ratio of X's age to Y's age three years ago is 2:1 -/
theorem age_ratio_problem (x_current y_current : ℕ) : 
  x_current = 45 →
  ∃ k : ℕ, k > 0 ∧ (x_current - 3) = k * (y_current - 3) →
  x_current + y_current + 14 = 83 →
  (x_current - 3) / (y_current - 3) = 2 := by
  sorry

end age_ratio_problem_l3212_321241


namespace triple_composition_even_l3212_321200

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

-- State the theorem
theorem triple_composition_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
by
  sorry

end triple_composition_even_l3212_321200


namespace product_and_sum_of_roots_l3212_321226

theorem product_and_sum_of_roots : 
  (16 : ℝ) ^ (1/4 : ℝ) * (32 : ℝ) ^ (1/5 : ℝ) + (64 : ℝ) ^ (1/6 : ℝ) = 6 := by
  sorry

end product_and_sum_of_roots_l3212_321226


namespace largest_prime_factor_of_9879_l3212_321210

theorem largest_prime_factor_of_9879 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9879 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9879 → q ≤ p :=
by
  -- The proof goes here
  sorry

end largest_prime_factor_of_9879_l3212_321210


namespace function_and_range_theorem_l3212_321215

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Define the function g
def g (m : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - m * x

-- State the theorem
theorem function_and_range_theorem (a b m : ℝ) :
  a ≠ 0 ∧
  (∀ x, f a b (x + 1) - f a b x = 2 * x - 1) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g m (f a b) x₁ - g m (f a b) x₂| ≤ 2) →
  (∀ x, f a b x = x^2 - 2*x + 3) ∧
  m ∈ Set.Icc (-1) 3 :=
by sorry

end function_and_range_theorem_l3212_321215


namespace big_dig_mining_theorem_l3212_321204

/-- Represents a mine with its daily production and ore percentages -/
structure Mine where
  dailyProduction : ℝ
  copperPercentage : ℝ
  ironPercentage : ℝ
  nickelPercentage : ℝ
  zincPercentage : ℝ

/-- Calculates the daily copper production for a given mine -/
def dailyCopperProduction (m : Mine) : ℝ :=
  m.dailyProduction * m.copperPercentage

/-- The Big Dig Mining Company problem -/
theorem big_dig_mining_theorem (mineA mineB mineC : Mine)
  (hA : mineA = { dailyProduction := 3000
                , copperPercentage := 0.05
                , ironPercentage := 0.60
                , nickelPercentage := 0.10
                , zincPercentage := 0.25 })
  (hB : mineB = { dailyProduction := 4000
                , copperPercentage := 0.10
                , ironPercentage := 0.50
                , nickelPercentage := 0.30
                , zincPercentage := 0.10 })
  (hC : mineC = { dailyProduction := 3500
                , copperPercentage := 0.15
                , ironPercentage := 0.45
                , nickelPercentage := 0.20
                , zincPercentage := 0.20 }) :
  dailyCopperProduction mineA + dailyCopperProduction mineB + dailyCopperProduction mineC = 1075 := by
  sorry

end big_dig_mining_theorem_l3212_321204


namespace absolute_value_equals_sqrt_of_square_l3212_321208

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end absolute_value_equals_sqrt_of_square_l3212_321208


namespace farm_animals_l3212_321248

theorem farm_animals (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 38 → chicken_count = 5 → ∃ (sheep_count : ℕ), 
    chicken_count + sheep_count = 12 ∧ 
    2 * chicken_count + 4 * sheep_count = total_legs :=
by
  sorry

end farm_animals_l3212_321248


namespace programming_contest_grouping_l3212_321259

/-- The number of programmers in the contest -/
def num_programmers : ℕ := 2008

/-- The number of rounds needed -/
def num_rounds : ℕ := 11

/-- A function that represents the grouping of programmers in each round -/
def grouping (round : ℕ) (programmer : ℕ) : Bool :=
  sorry

theorem programming_contest_grouping :
  (∀ (p1 p2 : ℕ), p1 < num_programmers → p2 < num_programmers → p1 ≠ p2 →
    ∃ (r : ℕ), r < num_rounds ∧ grouping r p1 ≠ grouping r p2) ∧
  (∀ (n : ℕ), n < num_rounds →
    ∃ (p1 p2 : ℕ), p1 < num_programmers ∧ p2 < num_programmers ∧ p1 ≠ p2 ∧
      ∀ (r : ℕ), r < n → grouping r p1 = grouping r p2) :=
sorry

end programming_contest_grouping_l3212_321259


namespace lcm_of_primes_l3212_321228

theorem lcm_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) 
  (hxy : x > y) (heq : 2 * x + y = 12) : 
  Nat.lcm x y = 10 := by
  sorry

end lcm_of_primes_l3212_321228


namespace sean_whistle_count_l3212_321231

/-- Given that Charles has 128 whistles and Sean has 95 more whistles than Charles,
    prove that Sean has 223 whistles. -/
theorem sean_whistle_count :
  let charles_whistles : ℕ := 128
  let sean_extra_whistles : ℕ := 95
  let sean_whistles : ℕ := charles_whistles + sean_extra_whistles
  sean_whistles = 223 := by
  sorry

end sean_whistle_count_l3212_321231


namespace distance_between_points_on_lines_l3212_321235

/-- The distance between two points on different lines with a given midpoint. -/
theorem distance_between_points_on_lines (xP yP xQ yQ : ℝ) :
  -- P is on the line 6y = 17x
  6 * yP = 17 * xP →
  -- Q is on the line 8y = 5x
  8 * yQ = 5 * xQ →
  -- (10, 5) is the midpoint of PQ
  (xP + xQ) / 2 = 10 →
  (yP + yQ) / 2 = 5 →
  -- The distance formula
  let distance := Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2)
  -- The distance is equal to some real value (which we don't specify)
  ∃ (d : ℝ), distance = d :=
by sorry

end distance_between_points_on_lines_l3212_321235


namespace expand_product_l3212_321287

theorem expand_product (x : ℝ) : 2 * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 := by
  sorry

end expand_product_l3212_321287


namespace distance_to_reflection_over_y_axis_l3212_321230

/-- Given a point F with coordinates (-4, 3), prove that the distance between F
    and its reflection over the y-axis is 8. -/
theorem distance_to_reflection_over_y_axis :
  let F : ℝ × ℝ := (-4, 3)
  let F' : ℝ × ℝ := (4, 3)  -- Reflection of F over y-axis
  dist F F' = 8 := by
  sorry

#check distance_to_reflection_over_y_axis

end distance_to_reflection_over_y_axis_l3212_321230


namespace square_triangle_equal_area_l3212_321237

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 40 →
  triangle_height = 60 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * x →
  x = 10/3 := by
sorry

end square_triangle_equal_area_l3212_321237


namespace domain_of_composite_function_l3212_321286

-- Define the function f with domain (-1, 0)
def f : ℝ → ℝ := sorry

-- Define the composite function g(x) = f(2x+1)
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Theorem statement
theorem domain_of_composite_function :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 0) →
  (∀ x, g x ≠ 0 → -1 < x ∧ x < -1/2) :=
sorry

end domain_of_composite_function_l3212_321286


namespace sunshine_orchard_pumpkins_l3212_321255

def moonglow_pumpkins : ℕ := 14

def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end sunshine_orchard_pumpkins_l3212_321255


namespace shortest_tangent_length_l3212_321278

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + (y - 3)^2 = 25
def C2 (x y : ℝ) : Prop := (x + 9)^2 + (y + 4)^2 = 49

-- Define the centers and radii
def center1 : ℝ × ℝ := (12, 3)
def center2 : ℝ × ℝ := (-9, -4)
def radius1 : ℝ := 5
def radius2 : ℝ := 7

-- Theorem statement
theorem shortest_tangent_length :
  ∃ (R S : ℝ × ℝ),
    C1 R.1 R.2 ∧ C2 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ), C1 P.1 P.2 → C2 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 70 :=
by sorry


end shortest_tangent_length_l3212_321278


namespace family_gathering_handshakes_count_l3212_321282

/-- Represents the number of unique handshakes at a family gathering with twins and triplets -/
def familyGatheringHandshakes : ℕ :=
  let twin_sets := 12
  let triplet_sets := 5
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let first_twin_sets := 4
  let first_twins := first_twin_sets * 2

  -- Handshakes among twins
  let twin_handshakes := (twins * (twins - 2)) / 2

  -- Handshakes among triplets
  let triplet_handshakes := (triplets * (triplets - 3)) / 2

  -- Handshakes between first 4 sets of twins and triplets
  let first_twin_triplet_handshakes := first_twins * (triplets / 3)

  twin_handshakes + triplet_handshakes + first_twin_triplet_handshakes

/-- The total number of unique handshakes at the family gathering is 394 -/
theorem family_gathering_handshakes_count :
  familyGatheringHandshakes = 394 := by
  sorry

end family_gathering_handshakes_count_l3212_321282


namespace mice_breeding_experiment_l3212_321225

/-- Calculates the number of mice after two generations of breeding and some pups being eaten --/
def final_mice_count (initial_mice : ℕ) (pups_per_mouse : ℕ) (pups_eaten_per_adult : ℕ) : ℕ :=
  let first_gen_total := initial_mice + initial_mice * pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pups_per_mouse
  second_gen_total - (first_gen_total * pups_eaten_per_adult)

/-- Theorem stating that under the given conditions, the final number of mice is 280 --/
theorem mice_breeding_experiment :
  final_mice_count 8 6 2 = 280 := by
  sorry

end mice_breeding_experiment_l3212_321225


namespace eggs_equal_to_rice_l3212_321232

/-- The cost of a pound of rice in dollars -/
def rice_cost : ℚ := 33/100

/-- The cost of a liter of kerosene in dollars -/
def kerosene_cost : ℚ := 22/100

/-- The number of eggs that cost as much as a half-liter of kerosene -/
def eggs_per_half_liter : ℕ := 4

/-- Theorem stating that 12 eggs cost as much as a pound of rice -/
theorem eggs_equal_to_rice : ℕ := by
  sorry

end eggs_equal_to_rice_l3212_321232


namespace set_equality_l3212_321209

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem set_equality : (Set.compl A) ∪ B = Set.Iic (-1) ∪ Set.Ioi 0 := by sorry

end set_equality_l3212_321209


namespace range_of_a_l3212_321242

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x else a * x^2 + 2 * x

/-- The range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  -1 ≤ a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l3212_321242


namespace ratio_equation_solution_l3212_321238

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 15 - 6 * a) : 
  a = 3 / 2 := by sorry

end ratio_equation_solution_l3212_321238


namespace coin_triangle_proof_l3212_321277

def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_triangle_proof (N : ℕ) (h : triangle_sum N = 2016) :
  sum_of_digits N = 9 := by
  sorry

end coin_triangle_proof_l3212_321277


namespace room_pave_cost_l3212_321216

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the cost to pave a rectangle given the cost per square meter -/
def pave_cost (r : Rectangle) (cost_per_sqm : ℝ) : ℝ := area r * cost_per_sqm

/-- The total cost to pave two rectangles -/
def total_pave_cost (r1 r2 : Rectangle) (cost1 cost2 : ℝ) : ℝ :=
  pave_cost r1 cost1 + pave_cost r2 cost2

theorem room_pave_cost :
  let rect1 : Rectangle := { length := 6, width := 4.75 }
  let rect2 : Rectangle := { length := 3, width := 2 }
  let cost1 : ℝ := 900
  let cost2 : ℝ := 750
  total_pave_cost rect1 rect2 cost1 cost2 = 30150 := by
  sorry

end room_pave_cost_l3212_321216


namespace quadratic_roots_sine_cosine_l3212_321217

theorem quadratic_roots_sine_cosine (α : Real) (c : Real) :
  (∃ (x y : Real), x = Real.sin α ∧ y = Real.cos α ∧ 
   10 * x^2 - 7 * x - c = 0 ∧ 10 * y^2 - 7 * y - c = 0) →
  c = 2.55 := by sorry

end quadratic_roots_sine_cosine_l3212_321217


namespace seventh_root_product_l3212_321223

theorem seventh_root_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 := by
  sorry

end seventh_root_product_l3212_321223


namespace nancy_pots_proof_l3212_321224

/-- Represents the number of clay pots Nancy created on Monday -/
def monday_pots : ℕ := sorry

/-- The total number of pots Nancy created over three days -/
def total_pots : ℕ := 50

/-- The number of pots Nancy created on Wednesday -/
def wednesday_pots : ℕ := 14

theorem nancy_pots_proof :
  monday_pots = 12 ∧
  monday_pots + 2 * monday_pots + wednesday_pots = total_pots :=
sorry

end nancy_pots_proof_l3212_321224


namespace prank_combinations_count_l3212_321270

/-- The number of choices for each day of the week-long prank --/
def prank_choices : List Nat := [1, 2, 3, 4, 2]

/-- The total number of combinations for the week-long prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 48 --/
theorem prank_combinations_count :
  total_combinations = 48 := by sorry

end prank_combinations_count_l3212_321270


namespace proportion_solution_l3212_321294

theorem proportion_solution (x : ℚ) : (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end proportion_solution_l3212_321294


namespace lcm_gcd_product_8_16_l3212_321288

theorem lcm_gcd_product_8_16 : Nat.lcm 8 16 * Nat.gcd 8 16 = 128 := by
  sorry

end lcm_gcd_product_8_16_l3212_321288


namespace money_sum_l3212_321291

theorem money_sum (a b : ℕ) (h1 : (1 : ℚ) / 3 * a = (1 : ℚ) / 4 * b) (h2 : b = 484) : a + b = 847 := by
  sorry

end money_sum_l3212_321291


namespace shaded_area_is_ten_l3212_321205

/-- A rectangle composed of twelve 1x1 squares -/
structure Rectangle where
  width : ℕ
  height : ℕ
  area : ℕ
  h1 : width = 3
  h2 : height = 4
  h3 : area = width * height
  h4 : area = 12

/-- The unshaded triangular region in the rectangle -/
structure UnshadedTriangle where
  base : ℕ
  height : ℕ
  area : ℝ
  h1 : base = 1
  h2 : height = 4
  h3 : area = (base * height : ℝ) / 2

/-- The total shaded area in the rectangle -/
def shadedArea (r : Rectangle) (ut : UnshadedTriangle) : ℝ :=
  (r.area : ℝ) - ut.area

theorem shaded_area_is_ten (r : Rectangle) (ut : UnshadedTriangle) :
  shadedArea r ut = 10 := by
  sorry

end shaded_area_is_ten_l3212_321205


namespace university_size_l3212_321244

/-- Represents the total number of students in a university --/
def total_students (sample_size : ℕ) (other_grades_sample : ℕ) (other_grades_total : ℕ) : ℕ :=
  (other_grades_total * sample_size) / other_grades_sample

/-- Theorem stating the total number of students in the university --/
theorem university_size :
  let sample_size : ℕ := 500
  let freshmen_sample : ℕ := 200
  let sophomore_sample : ℕ := 100
  let other_grades_sample : ℕ := sample_size - freshmen_sample - sophomore_sample
  let other_grades_total : ℕ := 3000
  total_students sample_size other_grades_sample other_grades_total = 7500 := by
  sorry

end university_size_l3212_321244


namespace fractional_equation_solution_l3212_321257

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (x - 2) - 1 = 2 / (2 - x) ∧ x = 5 / 2 := by
  sorry

end fractional_equation_solution_l3212_321257


namespace balloons_bought_at_park_l3212_321254

theorem balloons_bought_at_park (allan_initial : ℕ) (jake_initial : ℕ) (jake_bought : ℕ) :
  allan_initial = 6 →
  jake_initial = 2 →
  allan_initial = jake_initial + jake_bought + 1 →
  jake_bought = 3 := by
sorry

end balloons_bought_at_park_l3212_321254


namespace ordering_of_exponentials_l3212_321260

theorem ordering_of_exponentials :
  let a : ℝ := 2^(2/3)
  let b : ℝ := 2^(2/5)
  let c : ℝ := 3^(2/3)
  b < a ∧ a < c := by sorry

end ordering_of_exponentials_l3212_321260


namespace binomial_12_choose_6_l3212_321233

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end binomial_12_choose_6_l3212_321233


namespace kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l3212_321269

/-- Represents the number of nuts Kolya gets in each method -/
structure KolyaNuts (n : ℕ) where
  method1 : ℕ
  method2 : ℕ
  method3 : ℕ

/-- The theorem stating the most and least advantageous methods for Kolya -/
theorem kolya_best_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n),
    k.method1 ≥ n + 1 ∧
    k.method2 ≤ n ∧
    k.method3 ≤ n :=
by sorry

/-- Helper function to determine the most advantageous method -/
def most_advantageous (k : KolyaNuts n) : ℕ :=
  max k.method1 (max k.method2 k.method3)

/-- Helper function to determine the least advantageous method -/
def least_advantageous (k : KolyaNuts n) : ℕ :=
  min k.method1 (min k.method2 k.method3)

/-- Theorem stating that method 1 is the most advantageous for Kolya -/
theorem method1_most_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), most_advantageous k = k.method1 :=
by sorry

/-- Theorem stating that methods 2 and 3 are the least advantageous for Kolya -/
theorem method2_3_least_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), least_advantageous k = k.method2 ∧ least_advantageous k = k.method3 :=
by sorry

end kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l3212_321269


namespace parameter_a_condition_l3212_321202

theorem parameter_a_condition (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * x^2 + 2 * a * y^2 + 4 * a * x * y - 2 * x * y - y^2 - 2 * x + 1 ≥ 0) → 
  a ≥ 1/2 := by
  sorry

end parameter_a_condition_l3212_321202


namespace range_of_a_l3212_321289

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) → 
  a ∈ Set.Icc (-5/4 : ℝ) 1 := by
sorry

end range_of_a_l3212_321289


namespace solve_for_t_l3212_321243

theorem solve_for_t (s t : ℚ) (eq1 : 11 * s + 7 * t = 170) (eq2 : s = 2 * t - 3) : t = 203 / 29 := by
  sorry

end solve_for_t_l3212_321243


namespace negation_of_implication_l3212_321245

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) :=
by sorry

end negation_of_implication_l3212_321245


namespace parabola_directrix_l3212_321268

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = 4 * x^2 + 4

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = 63 / 16

/-- Theorem: The directrix of the parabola y = 4x^2 + 4 is y = 63/16 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p q : ℝ × ℝ, p.2 = 4 * p.1^2 + 4 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - d)^2) :=
sorry

end parabola_directrix_l3212_321268


namespace circle_radius_decrease_l3212_321249

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let original_area := π * r^2
  let new_area := 0.58 * original_area
  let new_radius := Real.sqrt (new_area / π)
  (r - new_radius) / r = 1 - Real.sqrt 0.58 := by
  sorry

end circle_radius_decrease_l3212_321249


namespace sum_of_distinct_prime_factors_462_l3212_321240

theorem sum_of_distinct_prime_factors_462 : 
  (Finset.sum (Nat.factors 462).toFinset id) = 23 := by
  sorry

end sum_of_distinct_prime_factors_462_l3212_321240


namespace marble_collection_problem_l3212_321261

/-- Represents the number of marbles collected by a person -/
structure MarbleCount where
  red : ℕ
  blue : ℕ

/-- The marble collection problem -/
theorem marble_collection_problem 
  (mary jenny anie tom : MarbleCount)
  (h1 : mary.red = 2 * jenny.red)
  (h2 : mary.blue = anie.blue / 2)
  (h3 : anie.red = mary.red + 20)
  (h4 : anie.blue = 2 * jenny.blue)
  (h5 : tom.red = anie.red + 10)
  (h6 : tom.blue = mary.blue)
  (h7 : jenny.red = 30)
  (h8 : jenny.blue = 25) :
  mary.blue + jenny.blue + anie.blue + tom.blue = 125 := by
  sorry

end marble_collection_problem_l3212_321261


namespace older_brother_stamps_l3212_321239

theorem older_brother_stamps (total : ℕ) (younger : ℕ) (older : ℕ) : 
  total = 25 →
  older = 2 * younger + 1 →
  total = older + younger →
  older = 17 :=
by
  sorry

end older_brother_stamps_l3212_321239


namespace parallel_lines_slope_l3212_321299

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), a * x - 5 * y - 9 = 0 ∧ 2 * x - 3 * y - 10 = 0) →
  a = 10/3 := by
  sorry

end parallel_lines_slope_l3212_321299


namespace larger_cross_section_distance_l3212_321280

/-- Represents a right triangular pyramid -/
structure RightTriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The area of the base of the pyramid -/
  baseArea : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right triangular pyramid, if two cross sections parallel to the base 
have areas of 144√3 and 324√3 square cm, and are 6 cm apart, 
then the larger cross section is 18 cm from the apex.
-/
theorem larger_cross_section_distance (pyramid : RightTriangularPyramid) 
  (section1 section2 : CrossSection) : 
  section1.area = 144 * Real.sqrt 3 →
  section2.area = 324 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 18 := by
  sorry

#check larger_cross_section_distance

end larger_cross_section_distance_l3212_321280


namespace ratio_as_percent_l3212_321201

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 4) (h2 : second_part = 20) :
  (first_part : ℚ) / second_part * 100 = 20 := by
  sorry

end ratio_as_percent_l3212_321201


namespace unique_n_mod_10_l3212_321218

theorem unique_n_mod_10 : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4000 [ZMOD 10] ∧ n = 0 := by
  sorry

end unique_n_mod_10_l3212_321218


namespace tangent_circle_equation_l3212_321247

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_A : (center.1 - 1)^2 + (center.2 - 2)^2 = radius^2
  passes_through_B : (center.1 - 1)^2 + (center.2 - 10)^2 = radius^2
  tangent_to_line : |center.1 - 2*center.2 - 1| / Real.sqrt 5 = radius

/-- The theorem stating that a circle passing through (1, 2) and (1, 10) and 
    tangent to x - 2y - 1 = 0 must have one of two specific equations -/
theorem tangent_circle_equation : 
  ∀ c : TangentCircle, 
    ((c.center.1 = 3 ∧ c.center.2 = 6 ∧ c.radius^2 = 20) ∨
     (c.center.1 = -7 ∧ c.center.2 = 6 ∧ c.radius^2 = 80)) :=
by sorry

end tangent_circle_equation_l3212_321247


namespace total_students_is_3700_l3212_321227

/-- Represents a high school with three grades -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The conditions of the problem -/
def problem_conditions (school : HighSchool) : Prop :=
  school.senior_students = 1000 ∧
  school.sample_size = 185 ∧
  school.freshman_sample = 75 ∧
  school.sophomore_sample = 60 ∧
  (school.senior_students : ℚ) / school.total_students = 
    (school.sample_size - school.freshman_sample - school.sophomore_sample : ℚ) / school.sample_size

/-- The theorem stating that under the given conditions, the total number of students is 3700 -/
theorem total_students_is_3700 (school : HighSchool) 
  (h : problem_conditions school) : school.total_students = 3700 := by
  sorry

end total_students_is_3700_l3212_321227


namespace prism_surface_area_l3212_321273

theorem prism_surface_area (R : ℝ) (h : ℝ) :
  R > 0 →
  (R / 2)^2 + 3 = R^2 →
  2 + h^2 = 4 * R^2 →
  2 + 4 * h = 4 * Real.sqrt 14 + 2 :=
by sorry

end prism_surface_area_l3212_321273


namespace probability_of_sum_26_l3212_321213

-- Define the faces of the dice
def die1_faces : Finset ℕ := Finset.range 20 \ {0, 19}
def die2_faces : Finset ℕ := (Finset.range 22 \ {0, 8, 21}) ∪ {0}

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20 * 20

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 13

-- Theorem statement
theorem probability_of_sum_26 :
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 400 :=
sorry

end probability_of_sum_26_l3212_321213


namespace hyperbola_asymptotes_l3212_321211

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eccentricity : Real.sqrt (1 + b^2 / a^2) = Real.sqrt 6 / 2) :
  let asymptote (x : ℝ) := Real.sqrt 2 / 2 * x
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → 
    (y = asymptote x ∨ y = -asymptote x) := by
  sorry

end hyperbola_asymptotes_l3212_321211


namespace largest_pot_cost_is_correct_l3212_321292

/-- Represents the cost of flower pots and their properties -/
structure FlowerPots where
  num_pots : ℕ
  total_cost_after_discount : ℚ
  discount_per_pot : ℚ
  price_difference : ℚ

/-- Calculates the cost of the largest pot before discount -/
def largest_pot_cost (fp : FlowerPots) : ℚ :=
  let total_discount := fp.num_pots * fp.discount_per_pot
  let total_cost_before_discount := fp.total_cost_after_discount + total_discount
  let smallest_pot_cost := (total_cost_before_discount - (fp.num_pots - 1) * fp.num_pots / 2 * fp.price_difference) / fp.num_pots
  smallest_pot_cost + (fp.num_pots - 1) * fp.price_difference

/-- Theorem stating that the cost of the largest pot before discount is $1.85 -/
theorem largest_pot_cost_is_correct (fp : FlowerPots) 
  (h1 : fp.num_pots = 6)
  (h2 : fp.total_cost_after_discount = 33/4)  -- $8.25 as a fraction
  (h3 : fp.discount_per_pot = 1/10)           -- $0.10 as a fraction
  (h4 : fp.price_difference = 3/20)           -- $0.15 as a fraction
  : largest_pot_cost fp = 37/20 := by         -- $1.85 as a fraction
  sorry

#eval largest_pot_cost {num_pots := 6, total_cost_after_discount := 33/4, discount_per_pot := 1/10, price_difference := 3/20}

end largest_pot_cost_is_correct_l3212_321292


namespace exactly_one_approve_probability_l3212_321276

def p_approve : ℝ := 0.7

def p_exactly_one_approve : ℝ :=
  3 * p_approve * (1 - p_approve) * (1 - p_approve)

theorem exactly_one_approve_probability :
  p_exactly_one_approve = 0.189 := by
  sorry

end exactly_one_approve_probability_l3212_321276


namespace parallelogram_vertex_product_l3212_321279

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by its four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two points are diagonally opposite in a parallelogram -/
def diagonallyOpposite (p : Parallelogram) (p1 p2 : Point) : Prop :=
  (p1 = p.A ∧ p2 = p.C) ∨ (p1 = p.B ∧ p2 = p.D) ∨ (p1 = p.C ∧ p2 = p.A) ∨ (p1 = p.D ∧ p2 = p.B)

/-- The main theorem -/
theorem parallelogram_vertex_product (p : Parallelogram) :
  p.A = Point.mk (-1) 3 →
  p.B = Point.mk 2 (-1) →
  p.D = Point.mk 7 6 →
  diagonallyOpposite p p.A p.D →
  p.C.x * p.C.y = 40 := by
  sorry

end parallelogram_vertex_product_l3212_321279


namespace hyperbola_line_intersection_l3212_321296

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = (1/3)*(x + 1)

/-- The number of intersection points -/
def intersection_count : ℕ := 1

/-- Theorem stating that the number of intersection points between the hyperbola and the line is 1 -/
theorem hyperbola_line_intersection :
  ∃! n : ℕ, n = intersection_count ∧ 
  (∃ (x y : ℝ), hyperbola x y ∧ line x y) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), hyperbola x₁ y₁ ∧ line x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end hyperbola_line_intersection_l3212_321296


namespace triangle_area_problem_l3212_321271

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 96 → x = 8 := by
  sorry

end triangle_area_problem_l3212_321271


namespace problem_statement_l3212_321293

theorem problem_statement :
  (∃ m : ℝ, |m| + m = 0 ∧ m ≥ 0) ∧
  (∃ a b : ℝ, |a - b| = b - a ∧ b ≤ a) ∧
  (∀ a b : ℝ, a^5 + b^5 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a / b ≠ -1) ∧
  (∀ a b c : ℚ, |a| / a + |b| / b + |c| / c = 1 → |a * b * c| / (a * b * c) = -1) :=
by sorry

end problem_statement_l3212_321293


namespace expression_evaluation_l3212_321234

theorem expression_evaluation : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end expression_evaluation_l3212_321234
