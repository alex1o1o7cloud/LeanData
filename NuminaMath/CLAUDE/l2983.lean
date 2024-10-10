import Mathlib

namespace complementary_angles_l2983_298350

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- A and B are complementary
  A = 7 * B →   -- A is 7 times B
  A = 78.75 :=  -- A is 78.75°
by sorry

end complementary_angles_l2983_298350


namespace unique_integer_pair_l2983_298375

theorem unique_integer_pair : ∃! (x y : ℕ+), 
  (x.val : ℝ) ^ (y.val : ℝ) + 1 = (y.val : ℝ) ^ (x.val : ℝ) ∧ 
  2 * (x.val : ℝ) ^ (y.val : ℝ) = (y.val : ℝ) ^ (x.val : ℝ) + 13 ∧ 
  x.val = 2 ∧ y.val = 4 := by
sorry

end unique_integer_pair_l2983_298375


namespace student_multiplication_problem_l2983_298337

theorem student_multiplication_problem (x : ℝ) : 40 * x - 138 = 102 → x = 6 := by
  sorry

end student_multiplication_problem_l2983_298337


namespace simultaneous_equations_solution_l2983_298369

theorem simultaneous_equations_solution (n : ℝ) :
  n ≠ (1/2 : ℝ) ↔ ∃ (x y : ℝ), y = (3*n + 1)*x + 2 ∧ y = (5*n - 2)*x + 5 := by
  sorry

end simultaneous_equations_solution_l2983_298369


namespace solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l2983_298389

-- Statement 1
theorem solution_set_inequality (x : ℝ) : 
  (x + 2) / (2 * x + 1) > 1 ↔ -1/2 < x ∧ x < 1 := by sorry

-- Statement 2
theorem not_sufficient_condition : 
  ∃ a b : ℝ, a * b > 1 ∧ ¬(a > 1 ∧ b > 1) := by sorry

-- Statement 3
theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 := by sorry

-- Statement 4
theorem not_necessary_condition : 
  ∃ a : ℝ, a < 6 ∧ a ≥ 2 := by sorry

end solution_set_inequality_not_sufficient_condition_negation_of_proposition_not_necessary_condition_l2983_298389


namespace pizza_toppings_combinations_l2983_298339

theorem pizza_toppings_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 3) :
  Nat.choose n k = 84 := by
  sorry

end pizza_toppings_combinations_l2983_298339


namespace curve_through_center_l2983_298365

-- Define the square
structure Square where
  center : ℝ × ℝ

-- Define the curve
structure Curve where
  -- A function that takes a real number parameter and returns a point on the curve
  pointAt : ℝ → ℝ × ℝ

-- Define the property that the curve divides the square into two equal areas
def divides_equally (s : Square) (γ : Curve) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- Define the property that a line segment passes through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the actual condition
  sorry

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally s γ) : 
  ∃ (a b : ℝ × ℝ), (∃ (t₁ t₂ : ℝ), γ.pointAt t₁ = a ∧ γ.pointAt t₂ = b) ∧ 
    passes_through a b s.center := by
  sorry

end curve_through_center_l2983_298365


namespace nested_expression_equals_one_l2983_298324

theorem nested_expression_equals_one :
  (3 * (3 * (3 * (3 * (3 * (3 - 2) - 2) - 2) - 2) - 2) - 2) = 1 := by
  sorry

end nested_expression_equals_one_l2983_298324


namespace cubic_sum_simplification_l2983_298390

theorem cubic_sum_simplification (a b : ℝ) : 
  a^2 = 9/25 → 
  b^2 = (3 + Real.sqrt 3)^2 / 15 → 
  a < 0 → 
  b > 0 → 
  (a + b)^3 = (-5670 * Real.sqrt 3 + 1620 * Real.sqrt 5 + 15 * Real.sqrt 15) / 50625 := by
sorry

end cubic_sum_simplification_l2983_298390


namespace g_16_value_l2983_298304

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 - x + 1

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, g x = a*x^4 + b*x^3 + c*x^2 + d*x + (-1)) ∧
  (g 0 = -1) ∧
  (∀ r : ℝ, f r = 0 → ∃ s : ℝ, g (r^2) = 0)

-- Theorem statement
theorem g_16_value (g : ℝ → ℝ) (h : is_valid_g g) : g 16 = -69905 := by
  sorry

end g_16_value_l2983_298304


namespace arithmetic_sequence_problem_l2983_298314

/-- Given an arithmetic sequence of 5 terms, prove that the first term is 1/6 under specific conditions --/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  (a 0 + a 1 + a 2 + a 3 + a 4 = 10) →  -- sum of all terms is 10
  (a 2 + a 3 + a 4 = (1 / 7) * (a 0 + a 1)) →  -- sum of larger three is 1/7 of sum of smaller two
  a 0 = 1 / 6 := by
sorry


end arithmetic_sequence_problem_l2983_298314


namespace newberg_airport_passengers_l2983_298382

theorem newberg_airport_passengers (on_time late : ℕ) 
  (h1 : on_time = 14507) 
  (h2 : late = 213) : 
  on_time + late = 14720 := by sorry

end newberg_airport_passengers_l2983_298382


namespace ratio_arithmetic_property_l2983_298340

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Our specific sequence -/
def our_sequence (a : ℕ → ℚ) : Prop :=
  is_ratio_arithmetic a 2 ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3

theorem ratio_arithmetic_property (a : ℕ → ℚ) (h : our_sequence a) :
  a 2019 / a 2017 = 4 * 2017^2 - 1 := by
  sorry

end ratio_arithmetic_property_l2983_298340


namespace monotonic_shift_l2983_298323

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of being monotonic on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- State the theorem
theorem monotonic_shift (a b : ℝ) (h : MonotonicOn f a b) :
  MonotonicOn (fun x => f (x + 3)) (a - 3) (b - 3) :=
sorry

end monotonic_shift_l2983_298323


namespace discount_percentage_l2983_298362

theorem discount_percentage
  (CP : ℝ) -- Cost Price
  (MP : ℝ) -- Marked Price
  (SP : ℝ) -- Selling Price
  (MP_condition : MP = CP * 1.5) -- Marked Price is 50% above Cost Price
  (SP_condition : SP = CP * 0.99) -- Selling Price results in 1% loss on Cost Price
  : (MP - SP) / MP * 100 = 34 := by
sorry

end discount_percentage_l2983_298362


namespace count_squarish_numbers_l2983_298368

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_squarish (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  is_perfect_square n ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_perfect_square (n / 100) ∧
  is_perfect_square (n % 100) ∧
  is_two_digit (n / 100) ∧
  is_two_digit (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 := by sorry

end count_squarish_numbers_l2983_298368


namespace num_broadcasting_methods_is_36_l2983_298333

/-- The number of different commercial ads -/
def num_commercial_ads : ℕ := 3

/-- The number of different Olympic promotional ads -/
def num_olympic_ads : ℕ := 2

/-- The total number of ads to be broadcast -/
def total_ads : ℕ := 5

/-- A function to calculate the number of broadcasting methods -/
def num_broadcasting_methods : ℕ := 
  let last_olympic_ad_choices := num_olympic_ads
  let second_olympic_ad_positions := total_ads - 2
  let remaining_ad_permutations := Nat.factorial num_commercial_ads
  last_olympic_ad_choices * second_olympic_ad_positions * remaining_ad_permutations

/-- Theorem stating that the number of broadcasting methods is 36 -/
theorem num_broadcasting_methods_is_36 : num_broadcasting_methods = 36 := by
  sorry


end num_broadcasting_methods_is_36_l2983_298333


namespace remaining_cooking_time_l2983_298353

def total_potatoes : ℕ := 15
def cooked_potatoes : ℕ := 8
def cooking_time_per_potato : ℕ := 9

theorem remaining_cooking_time :
  (total_potatoes - cooked_potatoes) * cooking_time_per_potato = 63 := by
  sorry

end remaining_cooking_time_l2983_298353


namespace partial_fraction_decomposition_product_l2983_298371

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x ≠ 3 → (45 * x - 36) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = -222.75 := by
sorry

end partial_fraction_decomposition_product_l2983_298371


namespace triangle_area_theorem_l2983_298381

def triangle_area (a b : ℝ) (cos_C : ℝ) : ℝ :=
  6

theorem triangle_area_theorem (a b cos_C : ℝ) :
  a = 3 →
  b = 5 →
  5 * cos_C^2 - 7 * cos_C - 6 = 0 →
  triangle_area a b cos_C = 6 := by
sorry

end triangle_area_theorem_l2983_298381


namespace distance_between_polar_points_l2983_298332

/-- Given two points in polar coordinates, prove their distance -/
theorem distance_between_polar_points (θ₁ θ₂ : ℝ) :
  let A : ℝ × ℝ := (4 * Real.cos θ₁, 4 * Real.sin θ₁)
  let B : ℝ × ℝ := (6 * Real.cos θ₂, 6 * Real.sin θ₂)
  θ₁ - θ₂ = π / 3 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 7 := by
  sorry

end distance_between_polar_points_l2983_298332


namespace min_bottles_to_fill_l2983_298398

def small_bottle_capacity : ℝ := 35
def large_bottle_capacity : ℝ := 500

theorem min_bottles_to_fill (small_cap large_cap : ℝ) (h1 : small_cap = small_bottle_capacity) (h2 : large_cap = large_bottle_capacity) :
  ⌈large_cap / small_cap⌉ = 15 := by
  sorry

end min_bottles_to_fill_l2983_298398


namespace unique_solution_l2983_298303

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y) = f (x * y^2) - 2 * x^2 * f y - f x - 1

theorem unique_solution (f : ℝ → ℝ) (h : functional_equation f) :
  ∀ y : ℝ, f y = y^2 - 1 := by sorry

end unique_solution_l2983_298303


namespace intersection_points_theorem_l2983_298307

def original_equation (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

def pair_A (x y : ℝ) : Prop := (y = x^2 - x) ∧ (y = 2*x - 2)
def pair_B (x y : ℝ) : Prop := (y = x^2 - 3*x + 2) ∧ (y = 0)
def pair_C (x y : ℝ) : Prop := (y = x - 1) ∧ (y = x + 1)
def pair_D (x y : ℝ) : Prop := (y = x^2 - 3*x + 3) ∧ (y = 1)

theorem intersection_points_theorem :
  (∃ x y : ℝ, pair_A x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_B x y ∧ original_equation x) ∧
  (∃ x y : ℝ, pair_D x y ∧ original_equation x) ∧
  ¬(∃ x y : ℝ, pair_C x y ∧ original_equation x) :=
sorry

end intersection_points_theorem_l2983_298307


namespace l₂_passes_through_fixed_point_l2983_298327

/-- A line in 2D space defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The point of symmetry -/
def symmetryPoint : ℝ × ℝ := (2, 1)

/-- Line l₁ defined as y = k(x - 4) -/
def l₁ (k : ℝ) : Line :=
  { slope := k, point := (4, 0) }

/-- Reflect a point about the symmetry point -/
def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * symmetryPoint.1 - p.1, 2 * symmetryPoint.2 - p.2)

/-- Line l₂ symmetric to l₁ about the symmetry point -/
def l₂ (k : ℝ) : Line :=
  { slope := -k, point := reflect (l₁ k).point }

theorem l₂_passes_through_fixed_point :
  ∀ k : ℝ, (l₂ k).point = (0, 2) := by sorry

end l₂_passes_through_fixed_point_l2983_298327


namespace annual_turbans_is_one_l2983_298364

/-- Represents the salary structure and conditions of Gopi's servant --/
structure SalaryStructure where
  annual_cash : ℕ  -- Annual cash salary in Rs.
  months_worked : ℕ  -- Number of months the servant worked
  cash_received : ℕ  -- Cash received by the servant
  turbans_received : ℕ  -- Number of turbans received by the servant
  turban_price : ℕ  -- Price of one turban in Rs.

/-- Calculates the number of turbans given as part of the annual salary --/
def calculate_annual_turbans (s : SalaryStructure) : ℕ :=
  -- Implementation not provided, use 'sorry'
  sorry

/-- Theorem stating that the number of turbans given annually is 1 --/
theorem annual_turbans_is_one (s : SalaryStructure) 
  (h1 : s.annual_cash = 90)
  (h2 : s.months_worked = 9)
  (h3 : s.cash_received = 45)
  (h4 : s.turbans_received = 1)
  (h5 : s.turban_price = 90) : 
  calculate_annual_turbans s = 1 := by
  sorry

end annual_turbans_is_one_l2983_298364


namespace dogs_not_liking_any_food_l2983_298302

theorem dogs_not_liking_any_food (total : ℕ) (watermelon salmon chicken : Finset ℕ) :
  total = 100 →
  watermelon.card = 20 →
  salmon.card = 70 →
  (watermelon ∩ salmon).card = 10 →
  chicken.card = 15 →
  (watermelon ∩ chicken).card = 5 →
  (salmon ∩ chicken).card = 8 →
  (watermelon ∩ salmon ∩ chicken).card = 3 →
  (total : ℤ) - (watermelon ∪ salmon ∪ chicken).card = 15 := by
  sorry

end dogs_not_liking_any_food_l2983_298302


namespace closest_multiple_of_15_to_2021_l2983_298392

theorem closest_multiple_of_15_to_2021 : ∃ (n : ℤ), 
  15 * n = 2025 ∧ 
  ∀ (m : ℤ), m ≠ n → 15 * m ≠ 2025 → |2021 - 15 * n| ≤ |2021 - 15 * m| := by
  sorry

end closest_multiple_of_15_to_2021_l2983_298392


namespace problem_1_problem_2_problem_3_problem_4_l2983_298344

-- 1. Prove that (-10) - (-22) + (-8) - 13 = -9
theorem problem_1 : (-10) - (-22) + (-8) - 13 = -9 := by sorry

-- 2. Prove that (-7/9 + 5/6 - 3/4) * (-36) = 25
theorem problem_2 : (-7/9 + 5/6 - 3/4) * (-36) = 25 := by sorry

-- 3. Prove that the solution to 6x - 7 = 4x - 5 is x = 1
theorem problem_3 : ∃ x : ℝ, 6*x - 7 = 4*x - 5 ∧ x = 1 := by sorry

-- 4. Prove that the solution to (x-3)/2 - (2x)/3 = 1 is x = -15
theorem problem_4 : ∃ x : ℝ, (x-3)/2 - (2*x)/3 = 1 ∧ x = -15 := by sorry

end problem_1_problem_2_problem_3_problem_4_l2983_298344


namespace aeroplane_transaction_loss_l2983_298338

theorem aeroplane_transaction_loss : 
  let selling_price : ℝ := 600
  let profit_percentage : ℝ := 0.2
  let loss_percentage : ℝ := 0.2
  let cost_price_profit : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_loss : ℝ := selling_price / (1 - loss_percentage)
  let total_cost : ℝ := cost_price_profit + cost_price_loss
  let total_revenue : ℝ := 2 * selling_price
  total_cost - total_revenue = 50 := by
sorry


end aeroplane_transaction_loss_l2983_298338


namespace student_line_arrangements_l2983_298357

theorem student_line_arrangements (n : ℕ) (h : n = 5) :
  (n.factorial : ℕ) - (((n - 1).factorial : ℕ) * 2) = 72 := by
  sorry

end student_line_arrangements_l2983_298357


namespace jolene_babysitting_l2983_298386

theorem jolene_babysitting (babysitting_rate : ℕ) (car_wash_rate : ℕ) (num_cars : ℕ) (total_raised : ℕ) :
  babysitting_rate = 30 →
  car_wash_rate = 12 →
  num_cars = 5 →
  total_raised = 180 →
  ∃ (num_families : ℕ), num_families * babysitting_rate + num_cars * car_wash_rate = total_raised ∧ num_families = 4 :=
by
  sorry

end jolene_babysitting_l2983_298386


namespace f_properties_l2983_298359

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (¬ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end f_properties_l2983_298359


namespace theater_revenue_calculation_l2983_298356

/-- Calculates the total revenue of a movie theater for a day --/
def theater_revenue (
  matinee_ticket_price evening_ticket_price opening_night_ticket_price : ℕ)
  (matinee_popcorn_price evening_popcorn_price opening_night_popcorn_price : ℕ)
  (matinee_drink_price evening_drink_price opening_night_drink_price : ℕ)
  (matinee_customers evening_customers opening_night_customers : ℕ)
  (popcorn_ratio drink_ratio : ℚ)
  (discount_groups : ℕ)
  (discount_group_size : ℕ)
  (discount_percentage : ℚ) : ℕ :=
  sorry

theorem theater_revenue_calculation :
  theater_revenue 5 7 10 8 10 12 3 4 5 32 40 58 (1/2) (1/4) 4 5 (1/10) = 1778 := by
  sorry

end theater_revenue_calculation_l2983_298356


namespace function_property_l2983_298312

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) + x = x * f y + f x)
  (h2 : f (-1) = 9) : 
  f (-500) = 1007 := by
  sorry

end function_property_l2983_298312


namespace cyclic_inequality_l2983_298348

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 / (a^2 + a*b + b^2) + b^3 / (b^2 + b*c + c^2) + c^3 / (c^2 + c*a + a^2) ≥ (a + b + c) / 3 := by
  sorry

end cyclic_inequality_l2983_298348


namespace solution_to_exponential_equation_l2983_298351

theorem solution_to_exponential_equation :
  ∃ y : ℝ, (3 : ℝ)^(y - 2) = (9 : ℝ)^(y + 2) ∧ y = -6 := by
  sorry

end solution_to_exponential_equation_l2983_298351


namespace power_inequality_l2983_298397

theorem power_inequality : 22^55 > 33^44 ∧ 33^44 > 55^33 ∧ 55^33 > 66^22 := by
  sorry

end power_inequality_l2983_298397


namespace sphere_surface_area_from_circumscribing_cube_l2983_298336

theorem sphere_surface_area_from_circumscribing_cube (cube_volume : ℝ) (sphere_surface_area : ℝ) : 
  cube_volume = 8 → sphere_surface_area = 4 * Real.pi := by
  sorry

end sphere_surface_area_from_circumscribing_cube_l2983_298336


namespace remainder_theorem_l2983_298378

theorem remainder_theorem (N : ℤ) (h : N % 35 = 25) : N % 15 = 10 := by
  sorry

end remainder_theorem_l2983_298378


namespace kite_AC_length_l2983_298311

-- Define the kite ABCD
structure Kite :=
  (A B C D : ℝ × ℝ)
  (diagonals_perpendicular : (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0)
  (BD_length : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 10)
  (AB_equals_BC : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 
                  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2))
  (AB_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (AD_equals_DC : Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 
                  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2))

-- Theorem statement
theorem kite_AC_length (k : Kite) : 
  Real.sqrt ((k.A.1 - k.C.1)^2 + (k.A.2 - k.C.2)^2) = 24 := by
  sorry

end kite_AC_length_l2983_298311


namespace dave_final_tickets_l2983_298330

/-- Calculates the final number of tickets Dave has after a series of events at the arcade. -/
def dave_tickets : ℕ :=
  let initial_tickets := 11
  let candy_bar_cost := 3
  let beanie_cost := 5
  let racing_game_win := 10
  let claw_machine_win := 7
  let after_spending := initial_tickets - candy_bar_cost - beanie_cost
  let after_winning := after_spending + racing_game_win + claw_machine_win
  2 * after_winning

/-- Theorem stating that Dave ends up with 40 tickets after all events at the arcade. -/
theorem dave_final_tickets : dave_tickets = 40 := by
  sorry

end dave_final_tickets_l2983_298330


namespace quadratic_inequalities_l2983_298325

/-- Given that the solution set of x^2 - px - q < 0 is {x | 2 < x < 3}, 
    prove the values of p and q, and the solution set of qx^2 - px - 1 > 0 -/
theorem quadratic_inequalities (p q : ℝ) : 
  (∀ x, x^2 - p*x - q < 0 ↔ 2 < x ∧ x < 3) →
  p = 5 ∧ q = -6 ∧ 
  (∀ x, q*x^2 - p*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end quadratic_inequalities_l2983_298325


namespace other_number_proof_l2983_298355

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 8820)
  (h2 : Nat.gcd a b = 36)
  (h3 : a = 360) :
  b = 882 := by
  sorry

end other_number_proof_l2983_298355


namespace sequence_properties_l2983_298334

def a (n : ℕ) : ℤ := 2^n - (-1)^n

theorem sequence_properties :
  (∀ k : ℕ, k > 0 →
    (a k + a (k + 2) = 2 * a (k + 1)) ↔ k = 2) ∧
  (∀ r s : ℕ, r > 1 ∧ s > r →
    (a 1 + a s = 2 * a r) → s = r + 1) ∧
  (∀ q r s t : ℕ, 0 < q ∧ q < r ∧ r < s ∧ s < t →
    ¬(a q + a t = a r + a s)) :=
sorry

end sequence_properties_l2983_298334


namespace number_problem_l2983_298366

theorem number_problem (x : ℚ) : 4 * x + 7 * x = 55 → x = 5 := by
  sorry

end number_problem_l2983_298366


namespace geometric_squares_existence_and_uniqueness_l2983_298315

theorem geometric_squares_existence_and_uniqueness :
  ∃! k : ℤ,
    (∃ a b c : ℤ,
      (49 + k = a^2) ∧
      (441 + k = b^2) ∧
      (961 + k = c^2) ∧
      (∃ r : ℚ, b = r * a ∧ c = r * b)) ∧
    k = 1152 := by
  sorry

end geometric_squares_existence_and_uniqueness_l2983_298315


namespace greatest_power_of_two_l2983_298376

theorem greatest_power_of_two (n : ℕ) : 
  ∃ k : ℕ, 2^k ∣ (10^1503 - 4^752) ∧ ∀ m : ℕ, 2^m ∣ (10^1503 - 4^752) → m ≤ k := by
  sorry

end greatest_power_of_two_l2983_298376


namespace partitions_6_3_l2983_298347

def partitions (n : ℕ) (k : ℕ) : ℕ := sorry

theorem partitions_6_3 : partitions 6 3 = 7 := by sorry

end partitions_6_3_l2983_298347


namespace only_paint_worthy_is_204_l2983_298399

/-- Represents a painting configuration for the fence. -/
structure PaintConfig where
  h : ℕ+  -- Harold's interval
  t : ℕ+  -- Tanya's interval
  u : ℕ+  -- Ulysses' interval

/-- Checks if a painting configuration is valid (covers all pickets exactly once). -/
def isValidConfig (config : PaintConfig) : Prop :=
  -- Harold starts from second picket
  -- Tanya starts from first picket
  -- Ulysses starts from fourth picket
  -- Each picket is painted exactly once
  sorry

/-- Calculates the paint-worthy number for a given configuration. -/
def paintWorthy (config : PaintConfig) : ℕ :=
  100 * config.h.val + 10 * config.t.val + config.u.val

/-- The main theorem stating that 204 is the only paint-worthy number. -/
theorem only_paint_worthy_is_204 :
  ∀ config : PaintConfig, isValidConfig config → paintWorthy config = 204 := by
  sorry

end only_paint_worthy_is_204_l2983_298399


namespace w_expression_l2983_298374

theorem w_expression (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y + 1/z = 1/w) : 
  w = x*y*z / (y*z + x*z + x*y) := by
sorry

end w_expression_l2983_298374


namespace min_value_of_f_l2983_298341

open Real

-- Define the function f
def f (a b c d e x : ℝ) : ℝ := |x - a| + |x - b| + |x - c| + |x - d| + |x - e|

-- State the theorem
theorem min_value_of_f (a b c d e : ℝ) (h : a < b ∧ b < c ∧ c < d ∧ d < e) :
  ∃ (m : ℝ), (∀ x, f a b c d e x ≥ m) ∧ m = e + d - b - a := by sorry

end min_value_of_f_l2983_298341


namespace sin_cos_value_l2983_298384

theorem sin_cos_value (x : Real) (h : 2 * Real.sin x = 5 * Real.cos x) : 
  Real.sin x * Real.cos x = 10 / 29 := by
  sorry

end sin_cos_value_l2983_298384


namespace quadratic_common_roots_l2983_298328

theorem quadratic_common_roots (p : ℚ) (x : ℚ) : 
  (x^2 - (p+1)*x + (p+1) = 0 ∧ 2*x^2 + (p-2)*x - p - 7 = 0) ↔ 
  ((p = 3 ∧ x = 2) ∨ (p = -3/2 ∧ x = -1)) := by
sorry

end quadratic_common_roots_l2983_298328


namespace ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l2983_298313

/-- The focal coordinates of an ellipse with equation x²/a² + y²/b² = 1 are (±c, 0) where c² = a² - b² -/
theorem ellipse_focal_coordinates (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let c := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (∃ x : ℝ, x = c ∨ x = -c) ∧ (∀ x : ℝ, x^2 = c^2 → x = c ∨ x = -c) :=
by sorry

/-- The focal coordinates of the ellipse x²/5 + y²/4 = 1 are (±1, 0) -/
theorem specific_ellipse_focal_coordinates :
  let c := Real.sqrt (5 - 4)
  (∀ x y : ℝ, x^2 / 5 + y^2 / 4 = 1) →
  (∃ x : ℝ, x = 1 ∨ x = -1) ∧ (∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) :=
by sorry

end ellipse_focal_coordinates_specific_ellipse_focal_coordinates_l2983_298313


namespace coefficient_x_squared_in_expansion_l2983_298326

theorem coefficient_x_squared_in_expansion : ∃ (a b c d e : ℤ), 
  (2 * X + 1)^2 * (X - 2)^3 = a * X^5 + b * X^4 + c * X^3 + 10 * X^2 + d * X + e :=
sorry

end coefficient_x_squared_in_expansion_l2983_298326


namespace product_equals_four_l2983_298373

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ x y z : ℝ, x * y * z = (Real.sqrt ((x + 2) * (y + 3))) / (z + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end product_equals_four_l2983_298373


namespace complement_M_l2983_298363

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

theorem complement_M : Set.compl M = {x : ℝ | x > 2 ∨ x < -2} := by sorry

end complement_M_l2983_298363


namespace cos_x_plus_3y_eq_one_l2983_298358

/-- Given x and y in [-π/6, π/6] and a ∈ ℝ satisfying the system of equations,
    prove that cos(x + 3y) = 1 -/
theorem cos_x_plus_3y_eq_one 
  (x y : ℝ) 
  (hx : x ∈ Set.Icc (-π/6) (π/6))
  (hy : y ∈ Set.Icc (-π/6) (π/6))
  (a : ℝ)
  (eq1 : x^3 + Real.sin x - 3*a = 0)
  (eq2 : 9*y^3 + (1/3) * Real.sin (3*y) + a = 0) :
  Real.cos (x + 3*y) = 1 := by sorry

end cos_x_plus_3y_eq_one_l2983_298358


namespace polygon_sides_l2983_298370

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ sum_interior_angles = (n - 2) * 180 :=
by
  sorry

end polygon_sides_l2983_298370


namespace min_value_range_l2983_298318

def f (x : ℝ) := x^2 - 6*x + 8

theorem min_value_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) →
  a ∈ Set.Ioo 1 3 ∪ {3} :=
by sorry

end min_value_range_l2983_298318


namespace specific_meal_cost_l2983_298380

/-- Calculates the total amount spent on a meal including tip -/
def totalSpent (lunchCost drinkCost tipPercentage : ℚ) : ℚ :=
  let subtotal := lunchCost + drinkCost
  let tipAmount := (tipPercentage / 100) * subtotal
  subtotal + tipAmount

/-- Theorem: Given the specific costs and tip percentage, the total spent is $68.13 -/
theorem specific_meal_cost :
  totalSpent 50.20 4.30 25 = 68.13 := by sorry

end specific_meal_cost_l2983_298380


namespace sufficient_conditions_for_positive_product_l2983_298367

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1)) → a * b > 0 := by
  sorry

end sufficient_conditions_for_positive_product_l2983_298367


namespace namjoon_position_l2983_298372

theorem namjoon_position (total_students : ℕ) (position_from_left : ℕ) :
  total_students = 15 →
  position_from_left = 7 →
  total_students - position_from_left + 1 = 9 :=
by sorry

end namjoon_position_l2983_298372


namespace new_person_weight_l2983_298395

/-- The weight of the new person given the initial conditions -/
def weightOfNewPerson (initialCount : ℕ) (averageIncrease : ℚ) (replacedWeight : ℚ) : ℚ :=
  replacedWeight + (initialCount : ℚ) * averageIncrease

/-- Theorem stating the weight of the new person under the given conditions -/
theorem new_person_weight :
  weightOfNewPerson 8 (5/2) 75 = 95 := by sorry

end new_person_weight_l2983_298395


namespace five_by_five_grid_squares_l2983_298394

/-- The number of squares of a given size in a 5x5 grid -/
def num_squares (size : Nat) : Nat :=
  (6 - size) ^ 2

/-- The total number of squares in a 5x5 grid -/
def total_squares : Nat :=
  (List.range 5).map (λ i => num_squares (i + 1)) |>.sum

theorem five_by_five_grid_squares :
  total_squares = 55 := by
  sorry

end five_by_five_grid_squares_l2983_298394


namespace quadratic_is_square_of_binomial_l2983_298308

theorem quadratic_is_square_of_binomial (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 - 25 * x + 9 = (r * x + s)^2) →
  a = 625 / 36 := by
  sorry

end quadratic_is_square_of_binomial_l2983_298308


namespace more_mashed_potatoes_than_bacon_l2983_298309

theorem more_mashed_potatoes_than_bacon (mashed_potatoes bacon : ℕ) 
  (h1 : mashed_potatoes = 457) 
  (h2 : bacon = 394) : 
  mashed_potatoes - bacon = 63 := by
sorry

end more_mashed_potatoes_than_bacon_l2983_298309


namespace f_increasing_l2983_298319

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end f_increasing_l2983_298319


namespace remaining_segments_length_l2983_298383

/-- Represents the dimensions of the initial polygon --/
structure PolygonDimensions where
  vertical1 : ℝ
  horizontal1 : ℝ
  vertical2 : ℝ
  horizontal2 : ℝ
  vertical3 : ℝ
  horizontal3 : ℝ

/-- Calculates the total length of segments in the polygon --/
def totalLength (d : PolygonDimensions) : ℝ :=
  d.vertical1 + d.horizontal1 + d.vertical2 + d.horizontal2 + d.vertical3 + d.horizontal3

/-- Theorem: The length of remaining segments after removal is 21 units --/
theorem remaining_segments_length
  (d : PolygonDimensions)
  (h1 : d.vertical1 = 10)
  (h2 : d.horizontal1 = 5)
  (h3 : d.vertical2 = 4)
  (h4 : d.horizontal2 = 3)
  (h5 : d.vertical3 = 4)
  (h6 : d.horizontal3 = 2)
  (h7 : totalLength d = 28)
  (h8 : ∃ (removed : ℝ), removed = 7) :
  totalLength d - 7 = 21 := by
  sorry

end remaining_segments_length_l2983_298383


namespace percentage_of_450_is_172_8_l2983_298316

theorem percentage_of_450_is_172_8 : 
  ∃ p : ℝ, (p / 100) * 450 = 172.8 ∧ p = 38.4 := by sorry

end percentage_of_450_is_172_8_l2983_298316


namespace correct_balloons_popped_l2983_298305

/-- The number of blue balloons Sally popped -/
def balloons_popped (joan_initial : ℕ) (jessica : ℕ) (total_now : ℕ) : ℕ :=
  joan_initial - total_now

theorem correct_balloons_popped (joan_initial jessica total_now : ℕ) 
  (h1 : joan_initial = 9)
  (h2 : jessica = 2)
  (h3 : total_now = 6) :
  balloons_popped joan_initial jessica total_now = 3 := by
  sorry

end correct_balloons_popped_l2983_298305


namespace nested_fourth_root_equation_solution_l2983_298393

/-- Defines the nested fourth root function for the left-hand side of the equation -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - Real.sqrt (x - Real.sqrt (x - Real.sqrt x)))

/-- Defines the nested fourth root function for the right-hand side of the equation -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

/-- There exists a positive real number x that satisfies the equation -/
theorem nested_fourth_root_equation_solution :
  ∃ x : ℝ, x > 0 ∧ f x = g x := by sorry

end nested_fourth_root_equation_solution_l2983_298393


namespace smallest_number_with_given_remainders_l2983_298387

theorem smallest_number_with_given_remainders : ∃ b : ℕ, 
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ n : ℕ, n < b → (n % 4 ≠ 2 ∨ n % 3 ≠ 2 ∨ n % 5 ≠ 3) :=
by sorry

end smallest_number_with_given_remainders_l2983_298387


namespace lg_ratio_theorem_l2983_298301

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_ratio_theorem (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (lg 12) / (lg 15) = (2 * a + b) / (1 - a + b) := by
  sorry

end lg_ratio_theorem_l2983_298301


namespace lynne_book_purchase_total_cost_lynne_spent_75_dollars_l2983_298331

theorem lynne_book_purchase_total_cost : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | cat_books, solar_books, magazines, book_cost, magazine_cost =>
    let total_books := cat_books + solar_books
    let book_total_cost := total_books * book_cost
    let magazine_total_cost := magazines * magazine_cost
    book_total_cost + magazine_total_cost

theorem lynne_spent_75_dollars : 
  lynne_book_purchase_total_cost 7 2 3 7 4 = 75 := by
  sorry

end lynne_book_purchase_total_cost_lynne_spent_75_dollars_l2983_298331


namespace decreasing_function_a_range_l2983_298396

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 2 else Real.log x / Real.log a

-- Define the property of f being decreasing on the entire real line
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- State the theorem
theorem decreasing_function_a_range (a : ℝ) :
  (is_decreasing (f a)) → (1/2 ≤ a ∧ a ≤ 3/4) :=
by sorry

end decreasing_function_a_range_l2983_298396


namespace no_real_solutions_quadratic_l2983_298377

theorem no_real_solutions_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 4*x + k ≠ 0) ↔ k > 4 := by
sorry

end no_real_solutions_quadratic_l2983_298377


namespace rosas_phone_calls_l2983_298346

/-- Rosa's phone calls over two weeks -/
theorem rosas_phone_calls (last_week : ℝ) (this_week : ℝ) 
  (h1 : last_week = 10.2)
  (h2 : this_week = 8.6) :
  last_week + this_week = 18.8 := by
  sorry

end rosas_phone_calls_l2983_298346


namespace xy_equals_nine_x_div_y_equals_thirtysix_l2983_298342

theorem xy_equals_nine_x_div_y_equals_thirtysix (x y : ℝ) 
  (h1 : x * y = 9)
  (h2 : x / y = 36)
  (hx : x > 0)
  (hy : y > 0) :
  y = 1/2 := by
sorry

end xy_equals_nine_x_div_y_equals_thirtysix_l2983_298342


namespace parabola_symmetry_range_l2983_298354

theorem parabola_symmetry_range (a : ℝ) : 
  a > 0 → 
  (∀ x y : ℝ, y = a * x^2 - 1 → 
    ∃ x1 y1 x2 y2 : ℝ, 
      y1 = a * x1^2 - 1 ∧ 
      y2 = a * x2^2 - 1 ∧ 
      x1 + y1 = -(x2 + y2) ∧ 
      (x1 ≠ x2 ∨ y1 ≠ y2)) → 
  a > 3/4 :=
sorry

end parabola_symmetry_range_l2983_298354


namespace parabola_line_intersection_right_angle_l2983_298391

/-- Parabola represented by the equation y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  isParabola : equation = fun x y => y^2 = 4*x

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line represented by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Angle between two vectors -/
def angle (v1 v2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_line_intersection_right_angle 
  (E : Parabola) 
  (M N : Point)
  (MN : Line)
  (A B : Point)
  (h1 : M.x = 1 ∧ M.y = -3)
  (h2 : N.x = 5 ∧ N.y = 1)
  (h3 : MN.p1 = M ∧ MN.p2 = N)
  (h4 : E.equation A.x A.y ∧ E.equation B.x B.y)
  (h5 : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
        A.x = M.x + t * (N.x - M.x) ∧ 
        A.y = M.y + t * (N.y - M.y))
  (h6 : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ 
        B.x = M.x + s * (N.x - M.x) ∧ 
        B.y = M.y + s * (N.y - M.y))
  : angle (A.x, A.y) (B.x, B.y) = π / 2 := by sorry

end parabola_line_intersection_right_angle_l2983_298391


namespace ralphs_purchase_cost_l2983_298310

/-- Calculates the final cost of Ralph's purchase given the initial conditions --/
theorem ralphs_purchase_cost
  (initial_total : ℝ)
  (discounted_item_price : ℝ)
  (item_discount_rate : ℝ)
  (total_discount_rate : ℝ)
  (h1 : initial_total = 54)
  (h2 : discounted_item_price = 20)
  (h3 : item_discount_rate = 0.2)
  (h4 : total_discount_rate = 0.1)
  : ∃ (final_cost : ℝ), final_cost = 45 :=
by
  sorry

end ralphs_purchase_cost_l2983_298310


namespace a_must_be_negative_l2983_298335

theorem a_must_be_negative (a b : ℝ) (hb : b > 0) (h : a / b < -2/3) : a < 0 := by
  sorry

end a_must_be_negative_l2983_298335


namespace polynomial_factorization_l2983_298329

theorem polynomial_factorization (a b : ℝ) : a^2 + 2*b - b^2 - 1 = (a-b+1)*(a+b-1) := by
  sorry

end polynomial_factorization_l2983_298329


namespace min_value_theorem_l2983_298352

-- Define the optimization problem
def optimization_problem (x y : ℝ) : Prop :=
  x - y ≥ 0 ∧ x + y - 2 ≥ 0 ∧ x ≤ 2

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  x^2 + y^2 - 2*x

-- Theorem statement
theorem min_value_theorem :
  ∃ (min_val : ℝ), min_val = -1/2 ∧
  (∀ (x y : ℝ), optimization_problem x y → objective_function x y ≥ min_val) ∧
  (∃ (x y : ℝ), optimization_problem x y ∧ objective_function x y = min_val) :=
sorry

end min_value_theorem_l2983_298352


namespace multiple_subtraction_problem_l2983_298349

theorem multiple_subtraction_problem (n : ℝ) (m : ℝ) : 
  n = 6 → m * n - 6 = 2 * n → m * n = 18 := by
  sorry

end multiple_subtraction_problem_l2983_298349


namespace lucy_deposit_l2983_298321

def initial_balance : ℕ := 65
def withdrawal : ℕ := 4
def final_balance : ℕ := 76

theorem lucy_deposit :
  ∃ (deposit : ℕ), initial_balance + deposit - withdrawal = final_balance :=
by
  sorry

end lucy_deposit_l2983_298321


namespace binomial_product_l2983_298300

theorem binomial_product (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 := by
  sorry

end binomial_product_l2983_298300


namespace f_even_and_decreasing_l2983_298360

def f (x : ℝ) : ℝ := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
  sorry

end f_even_and_decreasing_l2983_298360


namespace store_profit_l2983_298385

theorem store_profit (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 64 ∧ 
  profit_percent = 60 ∧ 
  loss_percent = 20 →
  let cost1 := price / (1 + profit_percent / 100)
  let cost2 := price / (1 - loss_percent / 100)
  price * 2 - (cost1 + cost2) = 8 := by
  sorry

end store_profit_l2983_298385


namespace rectangle_width_l2983_298322

/-- Given a rectangle with perimeter 50 cm and length 13 cm, its width is 12 cm. -/
theorem rectangle_width (perimeter length width : ℝ) : 
  perimeter = 50 ∧ length = 13 ∧ perimeter = 2 * length + 2 * width → width = 12 := by
  sorry

end rectangle_width_l2983_298322


namespace lcm_10_14_20_l2983_298345

theorem lcm_10_14_20 : Nat.lcm 10 (Nat.lcm 14 20) = 140 := by sorry

end lcm_10_14_20_l2983_298345


namespace modular_inverse_17_mod_23_l2983_298306

theorem modular_inverse_17_mod_23 :
  (∃ x : ℤ, (11 * x) % 23 = 1) →
  (∃ y : ℤ, (17 * y) % 23 = 1 ∧ 0 ≤ y ∧ y ≤ 22) ∧
  (∀ z : ℤ, (17 * z) % 23 = 1 → z % 23 = 19) :=
by sorry

end modular_inverse_17_mod_23_l2983_298306


namespace weight_loss_duration_l2983_298361

/-- Calculates the number of months required to reach a target weight given initial weight, weight loss per month, and target weight. -/
def months_to_reach_weight (initial_weight : ℕ) (weight_loss_per_month : ℕ) (target_weight : ℕ) : ℕ :=
  (initial_weight - target_weight) / weight_loss_per_month

/-- Proves that it takes 12 months to reduce weight from 250 pounds to 154 pounds, losing 8 pounds per month. -/
theorem weight_loss_duration :
  months_to_reach_weight 250 8 154 = 12 := by
  sorry

end weight_loss_duration_l2983_298361


namespace bracelets_given_to_school_is_three_l2983_298320

/-- The number of bracelets Chantel gave away to her friends at school -/
def bracelets_given_to_school : ℕ :=
  let days_first_period := 5
  let bracelets_per_day_first_period := 2
  let days_second_period := 4
  let bracelets_per_day_second_period := 3
  let bracelets_given_at_soccer := 6
  let bracelets_remaining := 13
  let total_bracelets_made := days_first_period * bracelets_per_day_first_period + 
                              days_second_period * bracelets_per_day_second_period
  let bracelets_after_soccer := total_bracelets_made - bracelets_given_at_soccer
  bracelets_after_soccer - bracelets_remaining

theorem bracelets_given_to_school_is_three : 
  bracelets_given_to_school = 3 := by sorry

end bracelets_given_to_school_is_three_l2983_298320


namespace correct_vote_distribution_l2983_298343

/-- Represents the number of votes for each candidate -/
structure Votes where
  eliot : ℕ
  shaun : ℕ
  randy : ℕ
  lisa : ℕ

/-- Checks if the vote distribution satisfies the given conditions -/
def is_valid_vote_distribution (v : Votes) : Prop :=
  v.eliot = 2 * v.shaun ∧
  v.eliot = 4 * v.randy ∧
  v.shaun = 5 * v.randy ∧
  v.shaun = 3 * v.lisa ∧
  v.randy = 16

/-- The theorem stating that the given vote distribution is correct -/
theorem correct_vote_distribution :
  ∃ (v : Votes), is_valid_vote_distribution v ∧
    v.eliot = 64 ∧ v.shaun = 80 ∧ v.randy = 16 ∧ v.lisa = 27 :=
by
  sorry


end correct_vote_distribution_l2983_298343


namespace base9_432_equals_base10_353_l2983_298379

/-- Converts a base 9 number to base 10 --/
def base9_to_base10 (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 9^2 + d₁ * 9^1 + d₀ * 9^0

/-- The base 9 number 432₉ is equal to 353 in base 10 --/
theorem base9_432_equals_base10_353 :
  base9_to_base10 4 3 2 = 353 := by sorry

end base9_432_equals_base10_353_l2983_298379


namespace third_median_length_l2983_298388

/-- Given a triangle with two medians of lengths 5 and 7 inches, and an area of 4√21 square inches,
    the length of the third median is 2√14 inches. -/
theorem third_median_length (m₁ m₂ : ℝ) (area : ℝ) (h₁ : m₁ = 5) (h₂ : m₂ = 7) (h_area : area = 4 * Real.sqrt 21) :
  ∃ (m₃ : ℝ), m₃ = 2 * Real.sqrt 14 ∧ 
  (∃ (a b c : ℝ), a^2 + b^2 + c^2 = 3 * (m₁^2 + m₂^2 + m₃^2) ∧
                   area = (4 / 3) * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :=
by sorry

end third_median_length_l2983_298388


namespace apples_per_case_l2983_298317

theorem apples_per_case (total_apples : ℕ) (num_cases : ℕ) (h1 : total_apples = 1080) (h2 : num_cases = 90) :
  total_apples / num_cases = 12 := by
  sorry

end apples_per_case_l2983_298317
