import Mathlib

namespace vector_magnitude_l3922_392238

theorem vector_magnitude (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2)
  (h3 : ‖a - b‖ = Real.sqrt 3) :
  ‖a + b‖ = Real.sqrt 7 := by
sorry

end vector_magnitude_l3922_392238


namespace simplify_expression_l3922_392268

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^8 = 0 := by
  sorry

end simplify_expression_l3922_392268


namespace hostel_expenditure_increase_l3922_392277

theorem hostel_expenditure_increase 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (budget_decrease : ℕ) 
  (new_total_expenditure : ℕ) 
  (h1 : initial_students = 100)
  (h2 : new_students = 132)
  (h3 : budget_decrease = 10)
  (h4 : new_total_expenditure = 5400) :
  ∃ (original_avg_budget : ℕ),
    new_total_expenditure - initial_students * original_avg_budget = 300 := by
  sorry

end hostel_expenditure_increase_l3922_392277


namespace rectangle_cannot_fit_in_square_l3922_392237

/-- Proves that a rectangle with area 90 cm² and length-to-width ratio 5:3 cannot fit in a 100 cm² square -/
theorem rectangle_cannot_fit_in_square : ¬ ∃ (length width : ℝ),
  (length * width = 90) ∧ 
  (length / width = 5 / 3) ∧
  (length ≤ 10) ∧
  (width ≤ 10) := by
  sorry

end rectangle_cannot_fit_in_square_l3922_392237


namespace fixed_point_of_linear_function_l3922_392283

theorem fixed_point_of_linear_function (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x - k + 2
  f 1 = 2 := by
  sorry

end fixed_point_of_linear_function_l3922_392283


namespace expression_equality_l3922_392292

theorem expression_equality : 
  (84 + 4 / 19 : ℚ) * (1375 / 1000 : ℚ) + (105 + 5 / 19 : ℚ) * (9 / 10 : ℚ) = 210 + 10 / 19 := by
  sorry

end expression_equality_l3922_392292


namespace triangle_angle_proof_l3922_392227

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b = 2 * a * Real.sin B →
  A = π / 6 ∨ A = 5 * π / 6 :=
by sorry

end triangle_angle_proof_l3922_392227


namespace mean_of_two_numbers_l3922_392214

def numbers : List ℕ := [1871, 1997, 2023, 2029, 2113, 2125, 2137]

def sum_of_all : ℕ := numbers.sum

def mean_of_five : ℕ := 2100

def sum_of_five : ℕ := 5 * mean_of_five

def sum_of_two : ℕ := sum_of_all - sum_of_five

theorem mean_of_two_numbers : (sum_of_two : ℚ) / 2 = 1397.5 := by
  sorry

end mean_of_two_numbers_l3922_392214


namespace point_distance_sum_l3922_392281

theorem point_distance_sum (a : ℝ) : 
  (2 * a < 0) →  -- P is in the second quadrant (x-coordinate negative)
  (1 - 3 * a > 0) →  -- P is in the second quadrant (y-coordinate positive)
  (abs (2 * a) + abs (1 - 3 * a) = 6) →  -- Sum of distances to axes is 6
  a = -1 := by
sorry

end point_distance_sum_l3922_392281


namespace max_notebooks_purchase_l3922_392253

theorem max_notebooks_purchase (available : ℚ) (cost : ℚ) : 
  available = 12 → cost = 1.25 → 
  ⌊available / cost⌋ = 9 := by sorry

end max_notebooks_purchase_l3922_392253


namespace triangle_inequality_l3922_392213

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → a + c > b → 
  ¬(a = 5 ∧ b = 9 ∧ c = 4) :=
by
  sorry

#check triangle_inequality

end triangle_inequality_l3922_392213


namespace circle_tangent_line_radius_l3922_392258

/-- Given a circle and a line that are tangent, prove that the radius of the circle is 4. -/
theorem circle_tangent_line_radius (r : ℝ) (h1 : r > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  (∀ x y : ℝ, x^2 + y^2 ≤ r^2 → 3*x - 4*y + 20 ≥ 0) →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ 3*x - 4*y + 20 = 0) →
  r = 4 :=
by sorry

end circle_tangent_line_radius_l3922_392258


namespace contrapositive_equivalence_l3922_392282

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a + b ≠ 3) → ¬(a ≠ 1 ∨ b ≠ 2)) ↔ (a = 1 ∧ b = 2 → a + b = 3) :=
by sorry

end contrapositive_equivalence_l3922_392282


namespace exponential_inequality_range_l3922_392244

theorem exponential_inequality_range (x : ℝ) : 
  (2 : ℝ) ^ (2 * x - 7) < (2 : ℝ) ^ (x - 3) → x < 4 := by
  sorry

end exponential_inequality_range_l3922_392244


namespace initial_bananas_count_l3922_392241

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 70

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left_on_tree : ℕ := 100

/-- The number of bananas in Raj's basket -/
def bananas_in_basket : ℕ := 2 * bananas_eaten

/-- The total number of bananas Raj cut from the tree -/
def bananas_cut : ℕ := bananas_eaten + bananas_in_basket

/-- The initial number of bananas on the tree -/
def initial_bananas : ℕ := bananas_cut + bananas_left_on_tree

theorem initial_bananas_count : initial_bananas = 310 := by
  sorry

end initial_bananas_count_l3922_392241


namespace exactly_two_out_of_four_l3922_392250

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def number_of_successes : ℕ := 2

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

theorem exactly_two_out_of_four :
  binomial_probability number_of_trials number_of_successes probability_of_success = 96/625 := by
  sorry

end exactly_two_out_of_four_l3922_392250


namespace paint_for_solar_system_l3922_392265

/-- Amount of paint available for the solar system given the usage by Mary, Mike, and Lucy --/
theorem paint_for_solar_system 
  (total_paint : ℝ) 
  (mary_paint : ℝ) 
  (mike_extra_paint : ℝ) 
  (lucy_paint : ℝ) 
  (h1 : total_paint = 25) 
  (h2 : mary_paint = 3) 
  (h3 : mike_extra_paint = 2) 
  (h4 : lucy_paint = 4) : 
  total_paint - (mary_paint + (mary_paint + mike_extra_paint) + lucy_paint) = 13 :=
by sorry

end paint_for_solar_system_l3922_392265


namespace gcf_of_60_and_75_l3922_392204

theorem gcf_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_of_60_and_75_l3922_392204


namespace x_plus_y_equals_two_l3922_392289

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x = 1) 
  (hy : y^3 - 3*y^2 + 5*y = 5) : 
  x + y = 2 := by
sorry

end x_plus_y_equals_two_l3922_392289


namespace passengers_ratio_l3922_392242

/-- Proves that the ratio of first class to second class passengers is 1:50 given the problem conditions -/
theorem passengers_ratio (fare_ratio : ℚ) (total_amount : ℕ) (second_class_amount : ℕ) :
  fare_ratio = 3 / 1 →
  total_amount = 1325 →
  second_class_amount = 1250 →
  ∃ (x y : ℕ), x ≠ 0 ∧ y ≠ 0 ∧ (x : ℚ) / y = 1 / 50 ∧
    fare_ratio * x * (second_class_amount : ℚ) / y = (total_amount - second_class_amount : ℚ) := by
  sorry

#check passengers_ratio

end passengers_ratio_l3922_392242


namespace x_squared_divides_x_plus_y_l3922_392203

theorem x_squared_divides_x_plus_y (x y : ℕ) :
  x^2 ∣ (x^2 + x*y + x + y) → x^2 ∣ (x + y) := by
sorry

end x_squared_divides_x_plus_y_l3922_392203


namespace police_emergency_number_prime_divisor_l3922_392286

theorem police_emergency_number_prime_divisor (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = 1000 * k + 133) : 
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
sorry

end police_emergency_number_prime_divisor_l3922_392286


namespace perpendicular_bisector_implies_m_equals_three_l3922_392219

/-- Given two points A and B, if the equation of the perpendicular bisector 
    of segment AB is x + 2y - 2 = 0, then the x-coordinate of B is 3. -/
theorem perpendicular_bisector_implies_m_equals_three 
  (A B : ℝ × ℝ) 
  (h1 : A = (1, -2))
  (h2 : B.2 = 2)
  (h3 : ∀ x y : ℝ, (x + 2*y - 2 = 0) ↔ 
    (x = (A.1 + B.1)/2 ∧ y = (A.2 + B.2)/2)) : 
  B.1 = 3 := by
sorry

end perpendicular_bisector_implies_m_equals_three_l3922_392219


namespace division_and_addition_l3922_392212

theorem division_and_addition : (12 / (1/6)) + 3 = 75 := by
  sorry

end division_and_addition_l3922_392212


namespace map_age_conversion_l3922_392274

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8^1 + (n % 10) * 8^0

theorem map_age_conversion :
  octal_to_decimal 7324 = 2004 := by
  sorry

end map_age_conversion_l3922_392274


namespace y₁_less_than_y₂_l3922_392294

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- y₁ is the y-value when x = -3 -/
def y₁ : ℝ := f (-3)

/-- y₂ is the y-value when x = 4 -/
def y₂ : ℝ := f 4

/-- Theorem: For the linear function f(x) = 2x + 1, y₁ < y₂ -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l3922_392294


namespace optimal_station_location_l3922_392225

/-- Represents the optimal station location problem for Factory A --/
theorem optimal_station_location :
  let num_buildings : ℕ := 5
  let building_distances : List ℝ := [0, 50, 100, 150, 200]
  let worker_counts : List ℕ := [1, 2, 3, 4, 5]
  let total_workers : ℕ := worker_counts.sum
  
  -- Function to calculate total walking distance for a given station location
  let total_distance (station_location : ℝ) : ℝ :=
    List.sum (List.zipWith (fun d w => w * |station_location - d|) building_distances worker_counts)
  
  -- The optimal location minimizes the total walking distance
  ∃ (optimal_location : ℝ),
    (∀ (x : ℝ), total_distance optimal_location ≤ total_distance x) ∧
    optimal_location = 150
  := by sorry

end optimal_station_location_l3922_392225


namespace bills_trips_l3922_392232

theorem bills_trips (total_trips : ℕ) (jeans_trips : ℕ) (h1 : total_trips = 40) (h2 : jeans_trips = 23) :
  total_trips - jeans_trips = 17 :=
by sorry

end bills_trips_l3922_392232


namespace henrikhs_distance_l3922_392248

/-- The number of blocks Henrikh lives from his office. -/
def blocks : ℕ :=
  sorry

/-- The time in minutes it takes Henrikh to walk to work. -/
def walkTime : ℚ :=
  blocks

/-- The time in minutes it takes Henrikh to cycle to work. -/
def cycleTime : ℚ :=
  blocks * (20 / 60)

theorem henrikhs_distance :
  blocks = 12 ∧ walkTime = cycleTime + 8 :=
by sorry

end henrikhs_distance_l3922_392248


namespace unstable_products_selection_l3922_392270

theorem unstable_products_selection (n : ℕ) (d : ℕ) (k : ℕ) (h1 : n = 10) (h2 : d = 2) (h3 : k = 3) :
  (Nat.choose (n - d) 1 * d * Nat.choose (d - 1) 1) = 32 :=
sorry

end unstable_products_selection_l3922_392270


namespace sum_of_digits_cd_l3922_392276

/-- c is an integer made up of a sequence of 2023 sixes -/
def c : ℕ := (6 : ℕ) * ((10 ^ 2023 - 1) / 9)

/-- d is an integer made up of a sequence of 2023 ones -/
def d : ℕ := (10 ^ 2023 - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of digits in cd is 12133 -/
theorem sum_of_digits_cd : sum_of_digits (c * d) = 12133 := by sorry

end sum_of_digits_cd_l3922_392276


namespace smallest_z_l3922_392240

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  (2 * y = x + z) →  -- arithmetic progression
  (z * z = x * y) →  -- geometric progression
  (∀ w : ℤ, (∃ a b c : ℤ, a < b ∧ b < w ∧ 2 * b = a + w ∧ w * w = a * b) → w ≥ z) →
  z = 2 :=
sorry

end smallest_z_l3922_392240


namespace graph_shift_l3922_392288

-- Define the functions f and g
def f (x : ℝ) : ℝ := (x + 3)^2 - 1
def g (x : ℝ) : ℝ := (x - 2)^2 + 3

-- State the theorem
theorem graph_shift : ∀ x : ℝ, f x = g (x - 5) + 4 := by sorry

end graph_shift_l3922_392288


namespace quadratic_equation_unique_solution_l3922_392207

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 6 * x + c = 0) →  -- exactly one solution
  (a + c = 7) →                      -- sum condition
  (a < c) →                          -- order condition
  (a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2) := by
  sorry

end quadratic_equation_unique_solution_l3922_392207


namespace dogwood_tree_count_l3922_392249

/-- The total number of dogwood trees after planting operations -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of trees after planting is 16 -/
theorem dogwood_tree_count :
  total_trees 7 5 4 = 16 := by
  sorry

end dogwood_tree_count_l3922_392249


namespace triangle_radius_inequality_l3922_392228

theorem triangle_radius_inequality (a b c R r : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hR : 0 < R) (hr : 0 < r)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circum : 4 * R * (a * b * c) = (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))
  (h_inradius : r * (a + b + c) = 2 * (a * b * c) / (a + b + c)) :
  1 / R^2 ≤ 1 / a^2 + 1 / b^2 + 1 / c^2 ∧ 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≤ 1 / (2 * r)^2 := by
sorry

end triangle_radius_inequality_l3922_392228


namespace three_digit_numbers_theorem_l3922_392246

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def digits (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def satisfies_condition (n : ℕ) : Prop :=
  let (a, b, c) := digits n
  a ≠ 0 ∧ (26 ∣ (a^2 + b^2 + c^2))

def valid_numbers : Finset ℕ :=
  {100, 110, 101, 320, 302, 230, 203, 510, 501, 150, 105}

theorem three_digit_numbers_theorem :
  ∀ n : ℕ, is_three_digit n → (satisfies_condition n ↔ n ∈ valid_numbers) :=
by sorry

end three_digit_numbers_theorem_l3922_392246


namespace jameson_medals_l3922_392257

theorem jameson_medals (total_medals track_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : track_medals = 5)
  (h3 : ∃ swimming_medals : ℕ, swimming_medals = 2 * track_medals) :
  ∃ badminton_medals : ℕ, badminton_medals = total_medals - (track_medals + 2 * track_medals) ∧ badminton_medals = 5 := by
  sorry

end jameson_medals_l3922_392257


namespace car_rental_cost_l3922_392295

theorem car_rental_cost (gas_gallons : ℕ) (gas_price : ℚ) (mile_cost : ℚ) (miles_driven : ℕ) (total_cost : ℚ) :
  gas_gallons = 8 →
  gas_price = 7/2 →
  mile_cost = 1/2 →
  miles_driven = 320 →
  total_cost = 338 →
  (total_cost - (↑gas_gallons * gas_price + ↑miles_driven * mile_cost) : ℚ) = 150 := by
sorry

end car_rental_cost_l3922_392295


namespace sunset_colors_l3922_392209

/-- Represents the number of colors in a quick shift -/
def quick_colors : ℕ := 5

/-- Represents the number of colors in a slow shift -/
def slow_colors : ℕ := 2

/-- Represents the duration of each shift in minutes -/
def shift_duration : ℕ := 10

/-- Represents the duration of a complete cycle (quick + slow) in minutes -/
def cycle_duration : ℕ := 2 * shift_duration

/-- Represents the duration of the sunset in minutes -/
def sunset_duration : ℕ := 2 * 60

/-- Represents the number of cycles in the sunset -/
def num_cycles : ℕ := sunset_duration / cycle_duration

/-- Represents the total number of colors in one cycle -/
def colors_per_cycle : ℕ := quick_colors + slow_colors

/-- Theorem stating that the total number of colors seen during the sunset is 42 -/
theorem sunset_colors : num_cycles * colors_per_cycle = 42 := by
  sorry

end sunset_colors_l3922_392209


namespace min_value_theorem_l3922_392224

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^2 * y^3 * z^2 ≥ 1/2268 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    1/x₀ + 1/y₀ + 1/z₀ = 9 ∧ 
    x₀^2 * y₀^3 * z₀^2 = 1/2268 :=
by sorry

end min_value_theorem_l3922_392224


namespace pen_cost_l3922_392216

theorem pen_cost (pen ink_refill pencil : ℝ) 
  (total_cost : pen + ink_refill + pencil = 2.35)
  (pen_ink_relation : pen = ink_refill + 1.50)
  (pencil_cost : pencil = 0.45) : 
  pen = 1.70 := by
  sorry

end pen_cost_l3922_392216


namespace chocolate_box_weight_l3922_392296

/-- The weight of a box of chocolate bars in kilograms -/
def box_weight (bar_weight : ℕ) (num_bars : ℕ) : ℚ :=
  (bar_weight * num_bars : ℚ) / 1000

/-- Theorem: The weight of a box containing 16 chocolate bars, each weighing 125 grams, is 2 kilograms -/
theorem chocolate_box_weight :
  box_weight 125 16 = 2 := by
  sorry

end chocolate_box_weight_l3922_392296


namespace smallest_valid_n_l3922_392229

def doubling_sum (a : ℕ) (n : ℕ) : ℕ := a * (2^n - 1)

def is_valid_n (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range 6, ∃ a : ℕ, a > 0 ∧ doubling_sum a (i + 1) = n

theorem smallest_valid_n :
  is_valid_n 9765 ∧ ∀ m < 9765, ¬is_valid_n m :=
sorry

end smallest_valid_n_l3922_392229


namespace october_birthdays_percentage_l3922_392275

theorem october_birthdays_percentage (total : ℕ) (october_births : ℕ) : 
  total = 120 → october_births = 18 → (october_births : ℚ) / total * 100 = 15 := by
  sorry

end october_birthdays_percentage_l3922_392275


namespace jellybean_problem_l3922_392200

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  (initial_bags * initial_average + (initial_bags + 1) * average_increase) = 362 :=
by sorry

end jellybean_problem_l3922_392200


namespace pure_imaginary_complex_l3922_392298

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (17 : ℂ) / (4 - Complex.I)).im = 0 → a = 4 := by
  sorry

end pure_imaginary_complex_l3922_392298


namespace no_integer_solutions_l3922_392261

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + x + y^2 = 3*y - 1 := by
  sorry

end no_integer_solutions_l3922_392261


namespace expand_and_simplify_l3922_392221

theorem expand_and_simplify (x : ℝ) : 
  (1 + x^2 + x^4) * (1 - x^3 + x^5) = 1 - x^3 + x^2 + x^4 + x^9 := by
  sorry

end expand_and_simplify_l3922_392221


namespace puddle_depth_calculation_l3922_392293

/-- Represents the rainfall rate in centimeters per hour -/
def rainfall_rate : ℝ := 10

/-- Represents the duration of rainfall in hours -/
def rainfall_duration : ℝ := 3

/-- Represents the base area of the puddle in square centimeters -/
def puddle_base_area : ℝ := 300

/-- Calculates the depth of the puddle given the rainfall rate and duration -/
def puddle_depth : ℝ := rainfall_rate * rainfall_duration

theorem puddle_depth_calculation :
  puddle_depth = 30 := by sorry

end puddle_depth_calculation_l3922_392293


namespace coconut_grove_yield_l3922_392262

/-- Proves that given the conditions of the coconut grove problem, 
    when x = 8, the yield Y of each (x - 4) tree is 180 nuts per year. -/
theorem coconut_grove_yield (x : ℕ) (Y : ℕ) : 
  x = 8 →
  ((x + 4) * 60 + x * 120 + (x - 4) * Y) / (3 * x) = 100 →
  Y = 180 := by
  sorry

#check coconut_grove_yield

end coconut_grove_yield_l3922_392262


namespace area_of_triangle_def_l3922_392247

/-- Triangle DEF with vertices D, E, and F -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The line on which point F lies -/
def line_equation (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 9

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (t : Triangle) : ℝ :=
  sorry

/-- The main theorem stating that the area of triangle DEF is 10 -/
theorem area_of_triangle_def :
  ∀ (t : Triangle),
    t.D = (4, 0) →
    t.E = (0, 4) →
    line_equation t.F →
    triangle_area t = 10 :=
by sorry

end area_of_triangle_def_l3922_392247


namespace bag_probability_l3922_392272

/-- Given a bag of 5 balls where the probability of picking a red ball is 0.4,
    prove that the probability of picking exactly one red ball and one white ball
    when two balls are picked is 3/5 -/
theorem bag_probability (total_balls : ℕ) (prob_red : ℝ) :
  total_balls = 5 →
  prob_red = 0.4 →
  (2 : ℝ) * prob_red * (1 - prob_red) = 3/5 := by
  sorry

end bag_probability_l3922_392272


namespace non_binary_listeners_l3922_392271

/-- Represents the survey data from StreamNow -/
structure StreamNowSurvey where
  total_listeners : ℕ
  male_listeners : ℕ
  female_non_listeners : ℕ
  non_binary_non_listeners : ℕ
  total_non_listeners : ℕ

/-- Theorem stating the number of non-binary listeners based on the survey data -/
theorem non_binary_listeners (survey : StreamNowSurvey) 
  (h1 : survey.total_listeners = 250)
  (h2 : survey.male_listeners = 85)
  (h3 : survey.female_non_listeners = 95)
  (h4 : survey.non_binary_non_listeners = 45)
  (h5 : survey.total_non_listeners = 230) :
  survey.total_listeners - survey.male_listeners - survey.female_non_listeners = 70 :=
by sorry

end non_binary_listeners_l3922_392271


namespace range_of_a_range_of_x_min_value_ratio_l3922_392297

-- Define the quadratic function
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Part 1
theorem range_of_a (a b : ℝ) :
  (∀ x ∈ Set.Ioo 2 5, quadratic a b x > 0) ∧ quadratic a b 1 = 1 →
  a ∈ Set.Ioi (3 - 2 * Real.sqrt 2) :=
sorry

-- Part 2
theorem range_of_x (x : ℝ) :
  (∀ a ∈ Set.Icc (-2) (-1), quadratic a (-a-1) x > 0) ∧ quadratic 0 (-1) 1 = 1 →
  x ∈ Set.Ioo ((1 - Real.sqrt 17) / 4) ((1 + Real.sqrt 17) / 4) :=
sorry

-- Part 3
theorem min_value_ratio (a b : ℝ) :
  b > 0 ∧ (∀ x : ℝ, quadratic a b x ≥ 0) →
  (a + 2) / b ≥ 1 :=
sorry

end range_of_a_range_of_x_min_value_ratio_l3922_392297


namespace sqrt_equation_solution_l3922_392267

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt ((3 / x) + 5) = 5/2 → x = 12/5 := by
  sorry

end sqrt_equation_solution_l3922_392267


namespace arithmetic_sequence_sum_remainder_l3922_392269

/-- The remainder when the sum of an arithmetic sequence is divided by 8 -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) : 
  a₁ = 3 → d = 6 → aₙ = 309 → n * (a₁ + aₙ) % 16 = 8 → 
  (n * (a₁ + aₙ) / 2) % 8 = 4 :=
by sorry

end arithmetic_sequence_sum_remainder_l3922_392269


namespace num_technicians_correct_l3922_392239

/-- Represents the number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop. -/
def total_workers : ℕ := 49

/-- Represents the average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 20000

/-- Represents the average salary of non-technician workers in the workshop. -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians satisfies the given conditions. -/
theorem num_technicians_correct :
  num_technicians * avg_salary_technicians +
  (total_workers - num_technicians) * avg_salary_rest =
  total_workers * avg_salary_all :=
by sorry

end num_technicians_correct_l3922_392239


namespace x_4_sufficient_not_necessary_l3922_392210

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)

def magnitude_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem x_4_sufficient_not_necessary :
  (∀ x : ℝ, x = 4 → magnitude_squared (vector_a x) = 25) ∧
  (∃ x : ℝ, x ≠ 4 ∧ magnitude_squared (vector_a x) = 25) := by
  sorry

end x_4_sufficient_not_necessary_l3922_392210


namespace target_row_sum_equals_2011_squared_l3922_392208

/-- The row number where the sum of all numbers equals 2011² -/
def target_row : ℕ := 1006

/-- The number of elements in the nth row -/
def num_elements (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of elements in the nth row -/
def row_sum (n : ℕ) : ℕ := (2 * n - 1)^2

/-- Theorem stating that the target_row is the row where the sum equals 2011² -/
theorem target_row_sum_equals_2011_squared :
  row_sum target_row = 2011^2 :=
sorry

end target_row_sum_equals_2011_squared_l3922_392208


namespace infinitely_many_factorizable_numbers_l3922_392245

theorem infinitely_many_factorizable_numbers :
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧
    ∃ a b : ℕ, 
      (n^3 + 4*n + 505 : ℤ) = (a * b : ℤ) ∧
      a > n.sqrt ∧
      b > n.sqrt :=
by sorry

end infinitely_many_factorizable_numbers_l3922_392245


namespace parametric_to_cartesian_equivalence_l3922_392202

/-- A line in 2D space defined by parametric equations. -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given parametric line. -/
def givenLine : ParametricLine where
  x := λ t => 5 + 3 * t
  y := λ t => 10 - 4 * t

/-- The Cartesian form of a line: ax + by + c = 0 -/
structure CartesianLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The Cartesian line we want to prove is equivalent. -/
def targetLine : CartesianLine where
  a := 4
  b := 3
  c := -50

/-- 
Theorem: The given parametric line is equivalent to the target Cartesian line.
-/
theorem parametric_to_cartesian_equivalence :
  ∀ t : ℝ, 
  4 * (givenLine.x t) + 3 * (givenLine.y t) - 50 = 0 :=
by
  sorry

#check parametric_to_cartesian_equivalence

end parametric_to_cartesian_equivalence_l3922_392202


namespace solve_for_D_l3922_392266

theorem solve_for_D : ∃ D : ℤ, 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89 ∧ D = -5 := by sorry

end solve_for_D_l3922_392266


namespace red_faced_cubes_l3922_392260

theorem red_faced_cubes (n : ℕ) (h : n = 4) : 
  (n ^ 3) - (8 + 12 * (n - 2) + (n - 2) ^ 3) = 24 := by
  sorry

end red_faced_cubes_l3922_392260


namespace reflected_ray_equation_l3922_392218

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The incident point of the light ray -/
def P : Point := ⟨5, 3⟩

/-- The point where the light ray intersects the x-axis -/
def Q : Point := ⟨2, 0⟩

/-- Function to calculate the reflected point across the x-axis -/
def reflect_across_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The reflected point of P across the x-axis -/
def P' : Point := reflect_across_x_axis P

/-- Function to create a line from two points -/
def line_from_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The reflected ray line -/
def reflected_ray : Line := line_from_points Q P'

/-- Theorem stating that the reflected ray line has the equation x + y - 2 = 0 -/
theorem reflected_ray_equation :
  reflected_ray = ⟨1, 1, -2⟩ := by sorry

end reflected_ray_equation_l3922_392218


namespace product_squares_relation_l3922_392251

theorem product_squares_relation (a b : ℝ) (h : a * b = 2 * (a^2 + b^2)) :
  2 * a * b - (a^2 + b^2) = a * b := by
  sorry

end product_squares_relation_l3922_392251


namespace board_number_theorem_l3922_392279

/-- Represents the state of the numbers on the board -/
structure BoardState where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The operation described in the problem -/
def applyOperation (state : BoardState) : BoardState :=
  ⟨state.a, state.b, state.a + state.b - state.c⟩

/-- Checks if the numbers form an arithmetic sequence with difference 6 -/
def isArithmeticSequence (state : BoardState) : Prop :=
  state.b - state.a = 6 ∧ state.c - state.b = 6

/-- The main theorem to be proved -/
theorem board_number_theorem :
  ∃ (n : ℕ) (finalState : BoardState),
    finalState = (applyOperation^[n] ⟨3, 9, 15⟩) ∧
    isArithmeticSequence finalState ∧
    finalState.a = 2013 ∧
    finalState.b = 2019 ∧
    finalState.c = 2025 := by
  sorry

end board_number_theorem_l3922_392279


namespace problem_l3922_392273

theorem problem (a b : ℝ) (h : a - |b| > 0) : a^2 - b^2 > 0 := by
  sorry

end problem_l3922_392273


namespace perimeter_of_PQRS_l3922_392226

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the perimeter function
def perimeter (quad : Quadrilateral) : ℝ := sorry

-- Define the properties of the quadrilateral
def is_right_angle_at_Q (quad : Quadrilateral) : Prop := sorry
def PR_perpendicular_to_RS (quad : Quadrilateral) : Prop := sorry
def PQ_length (quad : Quadrilateral) : ℝ := sorry
def QR_length (quad : Quadrilateral) : ℝ := sorry
def RS_length (quad : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem perimeter_of_PQRS (quad : Quadrilateral) :
  is_right_angle_at_Q quad →
  PR_perpendicular_to_RS quad →
  PQ_length quad = 24 →
  QR_length quad = 28 →
  RS_length quad = 16 →
  perimeter quad = 68 + Real.sqrt 1616 :=
by sorry

end perimeter_of_PQRS_l3922_392226


namespace diophantine_equation_solution_l3922_392230

theorem diophantine_equation_solution (x y z : ℕ) (h : x^2 + 3*y^2 = 2^z) :
  ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2*n + 2 := by
  sorry

end diophantine_equation_solution_l3922_392230


namespace exactly_one_root_l3922_392256

-- Define the function f(x) = -x^3 - x
def f (x : ℝ) : ℝ := -x^3 - x

-- State the theorem
theorem exactly_one_root (m n : ℝ) (h_interval : m ≤ n) (h_product : f m * f n < 0) :
  ∃! x, m ≤ x ∧ x ≤ n ∧ f x = 0 :=
sorry

end exactly_one_root_l3922_392256


namespace steves_return_speed_l3922_392206

def one_way_distance : ℝ := 40
def total_travel_time : ℝ := 6

theorem steves_return_speed (v : ℝ) (h1 : v > 0) :
  (one_way_distance / v + one_way_distance / (2 * v) = total_travel_time) →
  2 * v = 20 := by
  sorry

end steves_return_speed_l3922_392206


namespace arithmetic_sequence_proof_l3922_392259

theorem arithmetic_sequence_proof :
  ∃ (a b : ℕ), 
    a = 1477 ∧ 
    b = 2089 ∧ 
    a ≤ 2000 ∧ 
    2000 ≤ b ∧ 
    ∃ (d : ℕ), a * (a + 1) - 2 = d ∧ b * (b + 1) - a * (a + 1) = d :=
by
  sorry

end arithmetic_sequence_proof_l3922_392259


namespace toys_per_day_l3922_392263

def toys_per_week : ℕ := 5500
def work_days_per_week : ℕ := 4

theorem toys_per_day (equal_daily_production : True) : 
  toys_per_week / work_days_per_week = 1375 := by
  sorry

end toys_per_day_l3922_392263


namespace cos_alpha_value_l3922_392235

theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 / tanα) = 8/3) →  -- slope of the line 2x + (tanα)y + 1 = 0 is 8/3
  cosα = -4/5 := by
sorry

end cos_alpha_value_l3922_392235


namespace muffin_combinations_l3922_392217

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of muffin types -/
def muffin_types : ℕ := 4

/-- The number of additional muffins to distribute -/
def additional_muffins : ℕ := 4

theorem muffin_combinations :
  distribute additional_muffins muffin_types = 35 := by
  sorry

end muffin_combinations_l3922_392217


namespace dividend_calculation_l3922_392234

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 38)
  (h2 : quotient = 19)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 729 := by
sorry

end dividend_calculation_l3922_392234


namespace problem_solution_l3922_392284

theorem problem_solution (x y z : ℕ+) 
  (h1 : x^2 + y^2 + z^2 = 2*(y*z + 1)) 
  (h2 : x + y + z = 4032) : 
  x^2 * y + z = 4031 := by
sorry

end problem_solution_l3922_392284


namespace third_group_frequency_l3922_392285

/-- Given a sample of data distributed into groups, calculate the frequency of the unspecified group --/
theorem third_group_frequency 
  (total : ℕ) 
  (num_groups : ℕ) 
  (group1 : ℕ) 
  (group2 : ℕ) 
  (group4 : ℕ) 
  (h1 : total = 40) 
  (h2 : num_groups = 4) 
  (h3 : group1 = 5) 
  (h4 : group2 = 12) 
  (h5 : group4 = 8) : 
  total - (group1 + group2 + group4) = 15 := by
  sorry

#check third_group_frequency

end third_group_frequency_l3922_392285


namespace sum_of_a_and_b_l3922_392233

theorem sum_of_a_and_b (a b : ℝ) (h1 : a + 2*b = 8) (h2 : 2*a + b = 4) : a + b = 4 := by
  sorry

end sum_of_a_and_b_l3922_392233


namespace intersection_property_l3922_392231

-- Define the parabola and line
def parabola (x : ℝ) : ℝ := 2 * x^2
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the intersection points A and B
def A (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := sorry

-- Define the midpoint M
def M (k : ℝ) : ℝ × ℝ := sorry

-- Define point N on x-axis
def N (k : ℝ) : ℝ × ℝ := sorry

-- Define vectors NA and NB
def NA (k : ℝ) : ℝ × ℝ := sorry
def NB (k : ℝ) : ℝ × ℝ := sorry

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := sorry

theorem intersection_property (k : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ parabola x₁ = line k x₁ ∧ parabola x₂ = line k x₂) →
  (dot_product (NA k) (NB k) = 0) →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 :=
sorry

end intersection_property_l3922_392231


namespace probability_A_not_lose_l3922_392236

-- Define the probabilities
def prob_A_win : ℝ := 0.3
def prob_draw : ℝ := 0.5

-- Define the probability of A not losing
def prob_A_not_lose : ℝ := prob_A_win + prob_draw

-- Theorem statement
theorem probability_A_not_lose : prob_A_not_lose = 0.8 := by
  sorry

end probability_A_not_lose_l3922_392236


namespace eleventh_term_of_arithmetic_sequence_l3922_392243

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem eleventh_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_1 : a 1 = 100) 
  (h_10 : a 10 = 10) : 
  a 11 = 0 := by
  sorry

end eleventh_term_of_arithmetic_sequence_l3922_392243


namespace probability_open_path_correct_l3922_392254

/-- The probability of being able to go from the first to the last floor using only open doors -/
def probability_open_path (n : ℕ) : ℚ :=
  (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1))

/-- Theorem stating the probability of an open path in a building with n floors -/
theorem probability_open_path_correct (n : ℕ) (h : n > 1) :
  probability_open_path n = (2 ^ (n - 1 : ℕ)) / (Nat.choose (2 * (n - 1)) (n - 1)) :=
by sorry

end probability_open_path_correct_l3922_392254


namespace sum_of_coefficients_l3922_392290

def polynomial (x : ℝ) : ℝ := -3 * (x^8 - x^5 + 2*x^3 - 6) + 5 * (x^4 + 3*x^2) - 4 * (x^6 - 5)

theorem sum_of_coefficients : 
  (polynomial 1) = 48 := by sorry

end sum_of_coefficients_l3922_392290


namespace plane_line_relations_l3922_392201

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (within : Line → Plane → Prop)
variable (ne : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h1 : intersects α β l)
  (h2 : within m α)
  (h3 : ne m l) :
  (parallel m β → parallel_lines m l) ∧
  (parallel_lines m l → parallel m β) ∧
  (perpendicular m β → perpendicular_lines m l) ∧
  ¬(perpendicular_lines m l → perpendicular m β) :=
by sorry

end plane_line_relations_l3922_392201


namespace equation_solution_l3922_392223

theorem equation_solution :
  let f := fun x : ℝ => 2 / x - (3 / x) * (5 / x) + 1 / 2
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧
    x₁ = -2 + Real.sqrt 34 ∧ 
    x₂ = -2 - Real.sqrt 34 :=
by sorry

end equation_solution_l3922_392223


namespace batsman_average_l3922_392280

/-- Proves that given a batsman's average of 45 runs in 25 matches and an overall average of 38.4375 in 32 matches, the average runs scored in the last 7 matches is 15. -/
theorem batsman_average (first_25_avg : ℝ) (total_32_avg : ℝ) (first_25_matches : ℕ) (total_matches : ℕ) :
  first_25_avg = 45 →
  total_32_avg = 38.4375 →
  first_25_matches = 25 →
  total_matches = 32 →
  let last_7_matches := total_matches - first_25_matches
  let total_runs := total_32_avg * total_matches
  let first_25_runs := first_25_avg * first_25_matches
  let last_7_runs := total_runs - first_25_runs
  last_7_runs / last_7_matches = 15 := by
sorry

end batsman_average_l3922_392280


namespace flour_weight_range_l3922_392278

/-- Given a bag of flour labeled as 25 ± 0.02kg, prove that its weight m is within the range 24.98kg ≤ m ≤ 25.02kg -/
theorem flour_weight_range (m : ℝ) (h : |m - 25| ≤ 0.02) : 24.98 ≤ m ∧ m ≤ 25.02 := by
  sorry

end flour_weight_range_l3922_392278


namespace even_function_inequality_l3922_392252

/-- Given a function f(x) = a^(|x+b|) where a > 0, a ≠ 1, b ∈ ℝ, and f is even, prove f(b-3) < f(a+2) -/
theorem even_function_inequality (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(|x + b|)
  (∀ x, f x = f (-x)) →
  f (b - 3) < f (a + 2) := by
sorry

end even_function_inequality_l3922_392252


namespace sudoku_unique_solution_l3922_392264

def Sudoku := Fin 4 → Fin 4 → Fin 4

def valid_sudoku (s : Sudoku) : Prop :=
  (∀ i j₁ j₂, j₁ ≠ j₂ → s i j₁ ≠ s i j₂) ∧  -- rows
  (∀ i₁ i₂ j, i₁ ≠ i₂ → s i₁ j ≠ s i₂ j) ∧  -- columns
  (∀ b₁ b₂ c₁ c₂, (b₁ ≠ c₁ ∨ b₂ ≠ c₂) →     -- 2x2 subgrids
    s (2*b₁) (2*b₂) ≠ s (2*b₁+c₁) (2*b₂+c₂))

def initial_constraints (s : Sudoku) : Prop :=
  s 0 0 = 0 ∧  -- 3 in top-left (0-indexed)
  s 3 0 = 0 ∧  -- 1 in bottom-left
  s 2 2 = 1 ∧  -- 2 in third row, third column
  s 1 3 = 0    -- 1 in second row, fourth column

theorem sudoku_unique_solution (s : Sudoku) :
  valid_sudoku s ∧ initial_constraints s → s 0 1 = 1 := by sorry

end sudoku_unique_solution_l3922_392264


namespace stock_price_change_l3922_392205

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.05)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 4.5 := by
sorry

end stock_price_change_l3922_392205


namespace triangle_has_two_acute_angles_l3922_392222

-- Define a triangle as a structure with three angles
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

-- Define the property that the sum of angles in a triangle is 180°
def validTriangle (t : Triangle) : Prop :=
  t.angle1 + t.angle2 + t.angle3 = 180

-- Define an acute angle
def isAcute (angle : Real) : Prop := angle < 90

-- Theorem statement
theorem triangle_has_two_acute_angles (t : Triangle) (h : validTriangle t) :
  ∃ (a b : Real), (a = t.angle1 ∨ a = t.angle2 ∨ a = t.angle3) ∧
                  (b = t.angle1 ∨ b = t.angle2 ∨ b = t.angle3) ∧
                  (a ≠ b) ∧
                  isAcute a ∧ isAcute b :=
sorry

end triangle_has_two_acute_angles_l3922_392222


namespace max_point_inequality_l3922_392287

noncomputable section

variables (a : ℝ) (x₁ : ℝ)

def f (x : ℝ) : ℝ := Real.log x - 2 * a * x

def g (x : ℝ) : ℝ := f a x + (1/2) * x^2

theorem max_point_inequality (h1 : x₁ > 0) (h2 : IsLocalMax (g a) x₁) :
  (Real.log x₁) / x₁ + 1 / x₁^2 > a :=
sorry

end

end max_point_inequality_l3922_392287


namespace linear_function_point_relation_l3922_392291

/-- Given two points P₁(x₁, y₁) and P₂(x₂, y₂) on the line y = -3x + 4,
    if x₁ < x₂, then y₁ > y₂ -/
theorem linear_function_point_relation (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ + 4 →
  y₂ = -3 * x₂ + 4 →
  x₁ < x₂ →
  y₁ > y₂ := by
sorry

end linear_function_point_relation_l3922_392291


namespace no_consecutive_sum_32_l3922_392299

theorem no_consecutive_sum_32 : ¬∃ (n k : ℕ), n > 0 ∧ (n * (2 * k + n - 1)) / 2 = 32 := by
  sorry

end no_consecutive_sum_32_l3922_392299


namespace initial_worksheets_count_l3922_392220

/-- Given that a teacher would have 20 worksheets to grade after grading 4 and receiving 18 more,
    prove that she initially had 6 worksheets to grade. -/
theorem initial_worksheets_count : ∀ x : ℕ, x - 4 + 18 = 20 → x = 6 := by
  sorry

end initial_worksheets_count_l3922_392220


namespace tank_filling_time_l3922_392211

theorem tank_filling_time (p q r s : ℚ) 
  (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  p + q + r + s = 1 := by
  sorry

end tank_filling_time_l3922_392211


namespace six_digit_numbers_with_zero_l3922_392255

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible 6-digit numbers -/
def total_numbers : ℕ := 9 * 10^(num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := 9^num_digits

/-- Theorem: The number of 6-digit numbers with at least one zero is 368,559 -/
theorem six_digit_numbers_with_zero : 
  total_numbers - numbers_without_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_l3922_392255


namespace minimize_expression_l3922_392215

theorem minimize_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end minimize_expression_l3922_392215
