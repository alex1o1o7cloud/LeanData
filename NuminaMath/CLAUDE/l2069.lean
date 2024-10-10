import Mathlib

namespace special_number_l2069_206984

def is_consecutive (a b c d e : ℕ) : Prop :=
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ (d + 1 = e)

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    is_consecutive a b c d e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (a * 10 + b) * c = d * 10 + e

theorem special_number :
  satisfies_condition 13452 :=
sorry

end special_number_l2069_206984


namespace system_solution_l2069_206906

theorem system_solution :
  ∀ x y z : ℝ,
  x = Real.sqrt (2 * y + 3) →
  y = Real.sqrt (2 * z + 3) →
  z = Real.sqrt (2 * x + 3) →
  x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

#check system_solution

end system_solution_l2069_206906


namespace min_product_a_purchase_l2069_206926

theorem min_product_a_purchase (cost_a cost_b total_items max_cost : ℕ) 
  (h1 : cost_a = 20)
  (h2 : cost_b = 50)
  (h3 : total_items = 10)
  (h4 : max_cost = 350) : 
  ∃ min_a : ℕ, min_a = 5 ∧ 
  ∀ x : ℕ, (x ≤ total_items ∧ x * cost_a + (total_items - x) * cost_b ≤ max_cost) → x ≥ min_a := by
  sorry

end min_product_a_purchase_l2069_206926


namespace largest_trifecta_sum_l2069_206976

/-- A trifecta is an ordered triple of positive integers (a, b, c) with a < b < c
    such that a divides b, b divides c, and c divides ab. --/
def is_trifecta (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ b % a = 0 ∧ c % b = 0 ∧ (a * b) % c = 0

/-- The sum of a trifecta (a, b, c) --/
def trifecta_sum (a b c : ℕ) : ℕ := a + b + c

/-- The largest possible sum of a trifecta of three-digit integers is 700 --/
theorem largest_trifecta_sum :
  (∃ a b c : ℕ, 100 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 999 ∧ is_trifecta a b c ∧
    trifecta_sum a b c = 700) ∧
  (∀ a b c : ℕ, 100 ≤ a → a < b → b < c → c ≤ 999 → is_trifecta a b c →
    trifecta_sum a b c ≤ 700) :=
by sorry

end largest_trifecta_sum_l2069_206976


namespace arcsin_one_half_equals_pi_sixth_l2069_206925

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_equals_pi_sixth_l2069_206925


namespace min_value_A_l2069_206935

theorem min_value_A (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_abc : a + b + c = 3) :
  let A := (a^3 + b^3)/(8*a*b + 9 - c^2) + (b^3 + c^3)/(8*b*c + 9 - a^2) + (c^3 + a^3)/(8*c*a + 9 - b^2)
  A ≥ 3/8 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 3 ∧
    (a₀^3 + b₀^3)/(8*a₀*b₀ + 9 - c₀^2) + (b₀^3 + c₀^3)/(8*b₀*c₀ + 9 - a₀^2) + (c₀^3 + a₀^3)/(8*c₀*a₀ + 9 - b₀^2) = 3/8 :=
by sorry

end min_value_A_l2069_206935


namespace paper_goods_cost_l2069_206945

/-- Given that 100 paper plates and 200 paper cups cost $6.00, 
    prove that 20 paper plates and 40 paper cups cost $1.20 -/
theorem paper_goods_cost (plate_cost cup_cost : ℝ) 
    (h : 100 * plate_cost + 200 * cup_cost = 6) :
  20 * plate_cost + 40 * cup_cost = 1.2 := by
  sorry

end paper_goods_cost_l2069_206945


namespace ring_toss_revenue_l2069_206936

/-- The daily revenue of a ring toss game at a carnival -/
def daily_revenue (total_revenue : ℕ) (num_days : ℕ) : ℚ :=
  total_revenue / num_days

/-- Theorem stating that the daily revenue is 140 given the conditions -/
theorem ring_toss_revenue :
  daily_revenue 420 3 = 140 := by
  sorry

end ring_toss_revenue_l2069_206936


namespace total_interest_calculation_l2069_206951

/-- Calculate the total interest for two principal amounts -/
def totalInterest (principal1 principal2 : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal1 + principal2) * rate * time

theorem total_interest_calculation :
  let principal1 : ℝ := 1000
  let principal2 : ℝ := 1400
  let rate : ℝ := 0.03
  let time : ℝ := 4.861111111111111
  abs (totalInterest principal1 principal2 rate time - 350) < 0.01 := by
  sorry

end total_interest_calculation_l2069_206951


namespace even_quadratic_function_l2069_206959

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ [min (-b) (-a), max a b] → f (-x) = f x

theorem even_quadratic_function (a : ℝ) :
  let f := fun x ↦ a * x^2 + 1
  IsEvenOn f (3 - a) 5 → a = 8 := by
sorry

end even_quadratic_function_l2069_206959


namespace intersection_of_A_and_B_l2069_206969

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l2069_206969


namespace arithmetic_sequence_length_l2069_206986

/-- An arithmetic sequence starting at -58 with common difference 7 -/
def arithmeticSequence (n : ℕ) : ℤ := -58 + (n - 1) * 7

/-- The property that the sequence ends at or before 44 -/
def sequenceEndsBeforeOrAt44 (n : ℕ) : Prop := arithmeticSequence n ≤ 44

theorem arithmetic_sequence_length :
  ∃ (n : ℕ), n = 15 ∧ sequenceEndsBeforeOrAt44 n ∧ ¬sequenceEndsBeforeOrAt44 (n + 1) :=
sorry

end arithmetic_sequence_length_l2069_206986


namespace crushing_load_calculation_l2069_206989

theorem crushing_load_calculation (T H D : ℝ) (hT : T = 5) (hH : H = 15) (hD : D = 10) :
  let L := (30 * T^3) / (H * D)
  L = 25 := by
sorry

end crushing_load_calculation_l2069_206989


namespace circle_product_values_l2069_206938

noncomputable section

open Real Set

def circle_product (α β : ℝ × ℝ) : ℝ := 
  (α.1 * β.1 + α.2 * β.2) / (β.1 * β.1 + β.2 * β.2)

def angle (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem circle_product_values (a b : ℝ × ℝ) 
  (h1 : a ≠ (0, 0)) 
  (h2 : b ≠ (0, 0)) 
  (h3 : π/6 < angle a b ∧ angle a b < π/2) 
  (h4 : ∃ n : ℤ, circle_product a b = n/2) 
  (h5 : ∃ m : ℤ, circle_product b a = m/2) : 
  circle_product a b = 1 ∨ circle_product a b = 1/2 := by
sorry

end

end circle_product_values_l2069_206938


namespace m_fourth_plus_n_fourth_l2069_206998

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end m_fourth_plus_n_fourth_l2069_206998


namespace complement_union_sets_l2069_206929

def I : Finset ℕ := {0,1,2,3,4,5,6,7,8}
def M : Finset ℕ := {1,2,4,5}
def N : Finset ℕ := {0,3,5,7}

theorem complement_union_sets : (I \ (M ∪ N)) = {6,8} := by sorry

end complement_union_sets_l2069_206929


namespace largest_after_removal_l2069_206924

/-- Represents the initial number as a string -/
def initial_number : String := "123456789101112131415...99100"

/-- Represents the final number after digit removal as a string -/
def final_number : String := "9999978596061...99100"

/-- Function to remove digits from a string -/
def remove_digits (s : String) (n : Nat) : String := sorry

/-- Function to compare two strings as numbers -/
def compare_as_numbers (s1 s2 : String) : Bool := sorry

/-- Theorem stating that the final_number is the largest possible after removing 100 digits -/
theorem largest_after_removal :
  ∀ (s : String),
    s.length = initial_number.length - 100 →
    s = remove_digits initial_number 100 →
    compare_as_numbers final_number s = true :=
sorry

end largest_after_removal_l2069_206924


namespace arbor_day_saplings_l2069_206950

theorem arbor_day_saplings 
  (rate_A rate_B : ℚ) 
  (saplings_A saplings_B : ℕ) : 
  rate_A = (3 : ℚ) / 4 * rate_B → 
  saplings_B = saplings_A + 36 → 
  saplings_A + saplings_B = 252 := by
sorry

end arbor_day_saplings_l2069_206950


namespace curve_C_extrema_l2069_206973

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the function we want to maximize/minimize
def f (x y : ℝ) : ℝ := x + 2*y

-- State the theorem
theorem curve_C_extrema :
  (∀ x y : ℝ, C x y → 10 - Real.sqrt 6 ≤ f x y) ∧
  (∀ x y : ℝ, C x y → f x y ≤ 10 + Real.sqrt 6) ∧
  (∃ x₁ y₁ : ℝ, C x₁ y₁ ∧ f x₁ y₁ = 10 - Real.sqrt 6) ∧
  (∃ x₂ y₂ : ℝ, C x₂ y₂ ∧ f x₂ y₂ = 10 + Real.sqrt 6) :=
by sorry

end curve_C_extrema_l2069_206973


namespace shirt_cost_problem_l2069_206905

/-- Proves that the original cost of one of the remaining shirts is $12.50 -/
theorem shirt_cost_problem (total_original_cost : ℝ) (discounted_shirt_price : ℝ) 
  (discount_rate : ℝ) (current_total_cost : ℝ) :
  total_original_cost = 100 →
  discounted_shirt_price = 25 →
  discount_rate = 0.4 →
  current_total_cost = 85 →
  ∃ (remaining_shirt_cost : ℝ),
    remaining_shirt_cost = 12.5 ∧
    3 * discounted_shirt_price * (1 - discount_rate) + 2 * remaining_shirt_cost = current_total_cost ∧
    3 * discounted_shirt_price + 2 * remaining_shirt_cost = total_original_cost :=
by
  sorry


end shirt_cost_problem_l2069_206905


namespace sales_tax_difference_example_l2069_206943

/-- The difference between two sales tax amounts on a given price -/
def salesTaxDifference (price : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  price * rate2 - price * rate1

/-- Theorem stating that the difference between 8% and 7.5% sales tax on $50 is $0.25 -/
theorem sales_tax_difference_example : salesTaxDifference 50 0.075 0.08 = 0.25 := by
  sorry

end sales_tax_difference_example_l2069_206943


namespace xiao_li_commute_l2069_206942

/-- Xiao Li's commute problem -/
theorem xiao_li_commute 
  (distance : ℝ) 
  (walk_late : ℝ) 
  (bike_early : ℝ) 
  (bike_speed_factor : ℝ) 
  (breakdown_distance : ℝ) 
  (early_arrival : ℝ)
  (h1 : distance = 4.5)
  (h2 : walk_late = 5 / 60)
  (h3 : bike_early = 10 / 60)
  (h4 : bike_speed_factor = 1.5)
  (h5 : breakdown_distance = 1.5)
  (h6 : early_arrival = 5 / 60) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧ 
    bike_speed = 9 ∧ 
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_factor * walk_speed ∧
    breakdown_distance + (distance / bike_speed + bike_early - breakdown_distance / bike_speed - early_arrival) * min_run_speed ≥ distance :=
by
  sorry

end xiao_li_commute_l2069_206942


namespace fingernail_growth_rate_l2069_206923

/-- Proves that the rate of fingernail growth is 0.1 inch per month given the specified conditions. -/
theorem fingernail_growth_rate 
  (current_age : ℕ) 
  (record_age : ℕ) 
  (current_length : ℚ) 
  (record_length : ℚ) 
  (h1 : current_age = 12) 
  (h2 : record_age = 32) 
  (h3 : current_length = 2) 
  (h4 : record_length = 26) : 
  (record_length - current_length) / ((record_age - current_age) * 12 : ℚ) = 1/10 := by
  sorry

#eval (26 - 2 : ℚ) / ((32 - 12) * 12 : ℚ)

end fingernail_growth_rate_l2069_206923


namespace table_runner_coverage_l2069_206962

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 20) :
  (((total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
    two_layer_area + three_layer_area) / table_area) * 100 = 80 := by
  sorry

end table_runner_coverage_l2069_206962


namespace softball_team_savings_l2069_206954

/-- Calculates the total savings for a softball team's uniform purchase with group discount --/
theorem softball_team_savings
  (team_size : ℕ)
  (brand_a_shirt_cost brand_a_pants_cost brand_a_socks_cost : ℚ)
  (brand_b_shirt_cost brand_b_pants_cost brand_b_socks_cost : ℚ)
  (brand_a_customization_cost brand_b_customization_cost : ℚ)
  (brand_a_group_shirt_cost brand_a_group_pants_cost brand_a_group_socks_cost : ℚ)
  (brand_b_group_shirt_cost brand_b_group_pants_cost brand_b_group_socks_cost : ℚ)
  (individual_socks_players non_customized_shirts_players brand_b_socks_players : ℕ)
  (h1 : team_size = 12)
  (h2 : brand_a_shirt_cost = 7.5)
  (h3 : brand_a_pants_cost = 15)
  (h4 : brand_a_socks_cost = 4.5)
  (h5 : brand_b_shirt_cost = 10)
  (h6 : brand_b_pants_cost = 20)
  (h7 : brand_b_socks_cost = 6)
  (h8 : brand_a_customization_cost = 6)
  (h9 : brand_b_customization_cost = 8)
  (h10 : brand_a_group_shirt_cost = 6.5)
  (h11 : brand_a_group_pants_cost = 13)
  (h12 : brand_a_group_socks_cost = 4)
  (h13 : brand_b_group_shirt_cost = 8.5)
  (h14 : brand_b_group_pants_cost = 17)
  (h15 : brand_b_group_socks_cost = 5)
  (h16 : individual_socks_players = 3)
  (h17 : non_customized_shirts_players = 2)
  (h18 : brand_b_socks_players = 1) :
  (team_size * (brand_a_shirt_cost + brand_a_customization_cost + brand_b_pants_cost + brand_a_socks_cost)) -
  (team_size * (brand_a_group_shirt_cost + brand_a_customization_cost + brand_b_group_pants_cost + brand_a_group_socks_cost) +
   individual_socks_players * (brand_a_socks_cost - brand_a_group_socks_cost) -
   non_customized_shirts_players * brand_a_customization_cost +
   brand_b_socks_players * (brand_b_socks_cost - brand_a_group_socks_cost)) = 46.5 := by
  sorry


end softball_team_savings_l2069_206954


namespace abs_x_plus_one_gt_three_l2069_206974

theorem abs_x_plus_one_gt_three (x : ℝ) :
  |x + 1| > 3 ↔ x < -4 ∨ x > 2 :=
sorry

end abs_x_plus_one_gt_three_l2069_206974


namespace math_team_combinations_l2069_206968

theorem math_team_combinations (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 4) (h2 : boys = 6) : 
  (Nat.choose girls 3) * (Nat.choose boys 2) = 60 := by
  sorry

end math_team_combinations_l2069_206968


namespace min_value_2x_plus_y_l2069_206975

theorem min_value_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 1) :
  2 * x + y ≥ 2 * Real.sqrt 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x * y = 1 ∧ 2 * x + y = 2 * Real.sqrt 2 :=
by sorry

end min_value_2x_plus_y_l2069_206975


namespace water_molecule_radius_scientific_notation_l2069_206963

theorem water_molecule_radius_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00000000192 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ n = -9 :=
by sorry

end water_molecule_radius_scientific_notation_l2069_206963


namespace oliver_shelf_capacity_l2069_206961

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Oliver can fit 4 books on each shelf given the problem conditions. -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end oliver_shelf_capacity_l2069_206961


namespace no_two_digit_primes_with_digit_sum_9_and_tens_greater_l2069_206972

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ :=
  n / 10

/-- A function that returns the units digit of a two-digit number -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

/-- The main theorem -/
theorem no_two_digit_primes_with_digit_sum_9_and_tens_greater : 
  ∀ n : ℕ, 10 ≤ n → n < 100 → 
    (tens_digit n + units_digit n = 9 ∧ tens_digit n > units_digit n) → 
    ¬(is_prime n) :=
sorry

end no_two_digit_primes_with_digit_sum_9_and_tens_greater_l2069_206972


namespace stratified_sampling_medium_stores_l2069_206967

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300) 
  (h2 : medium_stores = 75) 
  (h3 : sample_size = 20) :
  ⌊(medium_stores : ℚ) / total_stores * sample_size⌋ = 5 := by
sorry

end stratified_sampling_medium_stores_l2069_206967


namespace sqrt_sum_inequality_l2069_206911

theorem sqrt_sum_inequality (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 19/3) :
  Real.sqrt (x - 1) + Real.sqrt (2 * x + 9) + Real.sqrt (19 - 3 * x) < 9 := by
  sorry

end sqrt_sum_inequality_l2069_206911


namespace pure_imaginary_product_l2069_206993

theorem pure_imaginary_product (x : ℝ) : 
  (∃ k : ℝ, (x + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 4) + 2*Complex.I) = k * Complex.I) ↔ 
  (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) :=
sorry

end pure_imaginary_product_l2069_206993


namespace solve_system_of_equations_l2069_206948

theorem solve_system_of_equations (x y z : ℤ) 
  (eq1 : 4 * x + y + z = 80)
  (eq2 : 2 * x - y - z = 40)
  (eq3 : 3 * x + y - z = 20) :
  x = 20 := by
sorry

end solve_system_of_equations_l2069_206948


namespace ana_dress_count_l2069_206947

/-- The number of dresses Ana has -/
def ana_dresses : ℕ := 15

/-- The number of dresses Lisa has -/
def lisa_dresses : ℕ := ana_dresses + 18

/-- The total number of dresses Ana and Lisa have combined -/
def total_dresses : ℕ := 48

theorem ana_dress_count : ana_dresses = 15 := by sorry

end ana_dress_count_l2069_206947


namespace inequality_solution_range_l2069_206919

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) →
  a ∈ Set.Ioi 3 ∪ Set.Iic 1 :=
by sorry

end inequality_solution_range_l2069_206919


namespace problem_solution_l2069_206931

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 9) 
  (h3 : x < y) : 
  (Real.sqrt x - Real.sqrt y) / (Real.sqrt x + Real.sqrt y) = -Real.sqrt 3 / 3 := by
  sorry

end problem_solution_l2069_206931


namespace symmetric_series_sum_sqrt_l2069_206987

def symmetric_series (n : ℕ) : ℕ := 
  2 * (n * (n + 1) / 2) + (n + 1)

theorem symmetric_series_sum_sqrt (n : ℕ) : 
  Real.sqrt (symmetric_series n) = (n : ℝ) + 0.5 :=
sorry

end symmetric_series_sum_sqrt_l2069_206987


namespace fraction_equality_l2069_206912

theorem fraction_equality : (3 * 4 * 5) / (2 * 3) = 10 := by
  sorry

end fraction_equality_l2069_206912


namespace max_k_value_l2069_206980

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 17) / 4 := by
  sorry

end max_k_value_l2069_206980


namespace parabola_line_intersection_chord_length_l2069_206920

/-- Represents a parabola with equation y² = 6x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  eq_def : equation = fun y x => y^2 = 6*x

/-- Represents a line with a 45° inclination passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ
  eq_def : slope = 1

/-- The length of the chord formed by the intersection of a parabola and a line -/
def chord_length (p : Parabola) (l : Line) : ℝ := 12

/-- Theorem: The length of the chord formed by the intersection of the parabola y² = 6x
    and a line passing through its focus with a 45° inclination is 12 -/
theorem parabola_line_intersection_chord_length (p : Parabola) (l : Line) :
  p.equation = fun y x => y^2 = 6*x →
  l.point = (3/2, 0) →
  l.slope = 1 →
  chord_length p l = 12 := by sorry

end parabola_line_intersection_chord_length_l2069_206920


namespace circle_sector_area_equality_l2069_206913

theorem circle_sector_area_equality (r : ℝ) (φ : ℝ) 
  (h1 : 0 < r) (h2 : 0 < φ) (h3 : φ < π / 4) :
  (φ * r^2 / 2 + r^2 * Real.sin φ / 2 = φ * r^2 + r^2 * Real.sin (2 * φ) / 2) ↔ φ = 0 :=
by sorry

end circle_sector_area_equality_l2069_206913


namespace probability_red_second_draw_three_five_l2069_206930

/-- Represents the color of a ball -/
inductive Color
  | Red
  | Blue

/-- Represents a box of balls -/
structure Box where
  total : Nat
  red : Nat
  blue : Nat
  h_total : total = red + blue

/-- Calculates the probability of drawing a red ball on the second draw -/
def probability_red_second_draw (box : Box) : Rat :=
  (box.red * (box.total - 1) + box.blue * box.red) / (box.total * (box.total - 1))

/-- Theorem stating the probability of drawing a red ball on the second draw -/
theorem probability_red_second_draw_three_five :
  let box : Box := ⟨5, 3, 2, rfl⟩
  probability_red_second_draw box = 3 / 5 := by
  sorry

end probability_red_second_draw_three_five_l2069_206930


namespace pure_imaginary_sum_l2069_206940

theorem pure_imaginary_sum (a b c d : ℝ) : 
  let z₁ : ℂ := a + b * Complex.I
  let z₂ : ℂ := c + d * Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a + c = 0 ∧ b + d ≠ 0 :=
by sorry

end pure_imaginary_sum_l2069_206940


namespace repeating_decimal_subtraction_l2069_206979

-- Define the repeating decimals
def repeating_246 : ℚ := 246 / 999
def repeating_135 : ℚ := 135 / 999
def repeating_579 : ℚ := 579 / 999

-- State the theorem
theorem repeating_decimal_subtraction :
  repeating_246 - repeating_135 - repeating_579 = -156 / 333 := by
  sorry

end repeating_decimal_subtraction_l2069_206979


namespace circle_area_diameter_increase_l2069_206985

theorem circle_area_diameter_increase : 
  ∀ (A D A' D' : ℝ), 
  A > 0 → D > 0 →
  A = π * (D / 2)^2 →
  A' = 4 * A →
  A' = π * (D' / 2)^2 →
  D' / D - 1 = 1 := by
sorry

end circle_area_diameter_increase_l2069_206985


namespace sum_of_cubes_divisibility_l2069_206982

theorem sum_of_cubes_divisibility (k n : ℤ) : 
  (∃ m : ℤ, k + n = 3 * m) → (∃ l : ℤ, k^3 + n^3 = 9 * l) := by
  sorry

end sum_of_cubes_divisibility_l2069_206982


namespace solution_set_when_a_eq_one_max_value_implies_a_l2069_206981

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

-- Part I
theorem solution_set_when_a_eq_one :
  {x : ℝ | f 1 x < 1} = {x : ℝ | x < (1/2 : ℝ)} := by sorry

-- Part II
theorem max_value_implies_a :
  (∃ (x : ℝ), f a x = 6) ∧ (∀ (x : ℝ), f a x ≤ 6) → a = 5 := by sorry

end solution_set_when_a_eq_one_max_value_implies_a_l2069_206981


namespace toy_store_problem_l2069_206934

/-- Toy store problem -/
theorem toy_store_problem 
  (purchase_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (max_price : ℝ) 
  (max_cost : ℝ) 
  (profit : ℝ) :
  purchase_price = 49 →
  base_price = 50 →
  base_sales = 50 →
  price_increment = 0.5 →
  sales_decrement = 3 →
  max_price = 60 →
  max_cost = 686 →
  profit = 147 →
  ∃ (x : ℝ) (a : ℝ),
    -- Part 1: Price range
    56 ≤ x ∧ x ≤ 60 ∧
    x ≤ max_price ∧
    purchase_price * (base_sales - sales_decrement * ((x - base_price) / price_increment)) ≤ max_cost ∧
    -- Part 2: Value of a
    a = 25 ∧
    (x * (1 + a / 100) - purchase_price) * (base_sales - sales_decrement * ((x - base_price) / price_increment)) * (1 - 2 * a / 100) = profit :=
by sorry

end toy_store_problem_l2069_206934


namespace a_fourth_plus_b_fourth_l2069_206902

theorem a_fourth_plus_b_fourth (a b : ℝ) 
  (h1 : a^2 - b^2 = 5) 
  (h2 : a * b = 2) : 
  a^4 + b^4 = 833 := by sorry

end a_fourth_plus_b_fourth_l2069_206902


namespace tan_seven_pi_sixths_l2069_206977

theorem tan_seven_pi_sixths : Real.tan (7 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_seven_pi_sixths_l2069_206977


namespace square_perimeter_sum_l2069_206921

theorem square_perimeter_sum (y : ℝ) (h1 : y^2 + (2*y)^2 = 145) (h2 : (2*y)^2 - y^2 = 105) :
  4*y + 8*y = 12 * Real.sqrt 35 := by
  sorry

end square_perimeter_sum_l2069_206921


namespace sperners_lemma_l2069_206983

theorem sperners_lemma (n : ℕ) (A : Finset (Finset ℕ)) :
  (∀ (i j : Finset ℕ), i ∈ A → j ∈ A → i ≠ j → (¬ i ⊆ j ∧ ¬ j ⊆ i)) →
  (∀ i ∈ A, i ⊆ Finset.range n) →
  A.card ≤ Nat.choose n (n / 2) := by
  sorry

end sperners_lemma_l2069_206983


namespace given_equation_is_quadratic_l2069_206922

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation x² = 2x - 3x² -/
def given_equation (x : ℝ) : ℝ := x^2 - 2*x + 3*x^2

theorem given_equation_is_quadratic :
  is_quadratic_equation given_equation :=
sorry

end given_equation_is_quadratic_l2069_206922


namespace function_upper_bound_l2069_206907

open Real

theorem function_upper_bound 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : a ∈ Set.Icc (-1 / Real.exp 1) 0) 
  (h2 : ∀ x > 0, f x = (x + 1) / Real.exp x - a * log x) :
  ∀ x ∈ Set.Ioo 0 2, f x < (1 - a - a^2) / Real.exp (-a) := by
sorry

end function_upper_bound_l2069_206907


namespace joe_tax_fraction_l2069_206941

/-- The fraction of income that goes to taxes -/
def tax_fraction (tax_payment : ℚ) (income : ℚ) : ℚ :=
  tax_payment / income

theorem joe_tax_fraction :
  let monthly_tax : ℚ := 848
  let monthly_income : ℚ := 2120
  tax_fraction monthly_tax monthly_income = 106 / 265 := by
sorry

end joe_tax_fraction_l2069_206941


namespace cos_theta_value_l2069_206988

theorem cos_theta_value (θ : Real) 
  (h1 : 0 ≤ θ ∧ θ ≤ π/2) 
  (h2 : Real.sin (θ - π/6) = 1/3) : 
  Real.cos θ = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end cos_theta_value_l2069_206988


namespace gcd_with_35_is_7_l2069_206957

theorem gcd_with_35_is_7 : 
  ∃ (s : Finset Nat), s = {n : Nat | 70 < n ∧ n < 90 ∧ Nat.gcd 35 n = 7} ∧ s = {77, 84} := by
  sorry

end gcd_with_35_is_7_l2069_206957


namespace hyperbola_equation_l2069_206916

/-- Hyperbola with center at origin, focus at (3,0), and intersection points with midpoint (-12,-15) -/
def Hyperbola (E : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)),
    (0, 0) ∈ E ∧  -- Center at origin
    (3, 0) ∈ E ∧  -- Focus at (3,0)
    (A ∈ E ∧ B ∈ E) ∧  -- A and B are on the hyperbola
    (A ∈ l ∧ B ∈ l ∧ (3, 0) ∈ l) ∧  -- A, B, and focus are on line l
    ((A.1 + B.1) / 2 = -12 ∧ (A.2 + B.2) / 2 = -15)  -- Midpoint of A and B is (-12,-15)

/-- The equation of the hyperbola -/
def HyperbolaEquation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

theorem hyperbola_equation (E : Set (ℝ × ℝ)) (h : Hyperbola E) :
  ∀ (x y : ℝ), (x, y) ∈ E ↔ HyperbolaEquation x y := by
  sorry

end hyperbola_equation_l2069_206916


namespace fixed_point_linear_function_l2069_206908

theorem fixed_point_linear_function (k : ℝ) : 
  (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end fixed_point_linear_function_l2069_206908


namespace son_work_time_l2069_206939

/-- Given a man and his son working on a job, this theorem proves how long it takes the son to complete the job alone. -/
theorem son_work_time (man_time son_time combined_time : ℚ)
  (hman : man_time = 5)
  (hcombined : combined_time = 4)
  (hwork : (1 / man_time) + (1 / son_time) = 1 / combined_time) :
  son_time = 20 := by
  sorry

#check son_work_time

end son_work_time_l2069_206939


namespace divisibility_implies_power_l2069_206965

theorem divisibility_implies_power (m n : ℕ+) 
  (h : (m * n) ∣ (m ^ 2010 + n ^ 2010 + n)) :
  ∃ k : ℕ+, n = k ^ 2010 := by sorry

end divisibility_implies_power_l2069_206965


namespace unclaimed_candy_l2069_206997

/-- Represents the order of arrival of the winners -/
inductive Winner : Type
  | Al | Bert | Carl | Dana

/-- The ratio of candy each winner should receive -/
def candy_ratio (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 3 / 10
  | Winner.Carl => 2 / 10
  | Winner.Dana => 1 / 10

/-- The amount of candy each winner actually takes -/
def candy_taken (w : Winner) : ℚ :=
  match w with
  | Winner.Al => 4 / 10
  | Winner.Bert => 9 / 50
  | Winner.Carl => 21 / 250
  | Winner.Dana => 19 / 250

theorem unclaimed_candy :
  1 - (candy_taken Winner.Al + candy_taken Winner.Bert + candy_taken Winner.Carl + candy_taken Winner.Dana) = 46 / 125 := by
  sorry

end unclaimed_candy_l2069_206997


namespace certain_number_sum_l2069_206910

theorem certain_number_sum (n : ℕ) : 
  (n % 423 = 0) → 
  (n / 423 = 423 - 421) → 
  (n + 421 = 1267) := by
sorry

end certain_number_sum_l2069_206910


namespace original_holes_additional_holes_l2069_206999

-- Define the circumference of the circular road
def circumference : ℕ := 400

-- Define the original interval between streetlamps
def original_interval : ℕ := 50

-- Define the new interval between streetlamps
def new_interval : ℕ := 40

-- Theorem for the number of holes in the original plan
theorem original_holes : circumference / original_interval = 8 := by sorry

-- Theorem for the number of additional holes in the new plan
theorem additional_holes : 
  circumference / new_interval - (circumference / (Nat.lcm original_interval new_interval)) = 8 := by sorry

end original_holes_additional_holes_l2069_206999


namespace geometric_sequence_a7_l2069_206966

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 = 4 →
  a 7 - 2 * a 5 = 32 →
  a 7 = 64 := by
sorry

end geometric_sequence_a7_l2069_206966


namespace largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l2069_206991

theorem largest_integer_in_fraction_inequality :
  ∀ x : ℤ, (2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11 → x ≤ 5 :=
by sorry

theorem five_satisfies_inequality :
  (2 : ℚ) / 5 < (5 : ℚ) / 7 ∧ (5 : ℚ) / 7 < 8 / 11 :=
by sorry

theorem largest_integer_is_five :
  ∃! x : ℤ, x = 5 ∧
    ((2 : ℚ) / 5 < (x : ℚ) / 7 ∧ (x : ℚ) / 7 < 8 / 11) ∧
    (∀ y : ℤ, (2 : ℚ) / 5 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 8 / 11 → y ≤ x) :=
by sorry

end largest_integer_in_fraction_inequality_five_satisfies_inequality_largest_integer_is_five_l2069_206991


namespace regular_polygon_exterior_angle_18_deg_has_20_sides_l2069_206900

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_exterior_angle_18_deg_has_20_sides :
  ∀ n : ℕ,
  n > 0 →
  (360 : ℝ) / n = 18 →
  n = 20 :=
by
  sorry

end regular_polygon_exterior_angle_18_deg_has_20_sides_l2069_206900


namespace bus_driver_compensation_l2069_206990

def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate_increase : ℝ := 0.75
def total_hours : ℝ := 54

def overtime_rate : ℝ := regular_rate * (1 + overtime_rate_increase)
def overtime_hours : ℝ := total_hours - regular_hours

def regular_pay : ℝ := regular_rate * regular_hours
def overtime_pay : ℝ := overtime_rate * overtime_hours
def total_compensation : ℝ := regular_pay + overtime_pay

theorem bus_driver_compensation :
  total_compensation = 1032 := by sorry

end bus_driver_compensation_l2069_206990


namespace train_crossing_poles_time_l2069_206927

/-- Calculates the total time for a train to cross multiple poles -/
theorem train_crossing_poles_time
  (train_speed : ℝ)
  (first_pole_crossing_time : ℝ)
  (pole_distances : List ℝ)
  (h1 : train_speed = 75)  -- 75 kmph
  (h2 : first_pole_crossing_time = 3)  -- 3 seconds
  (h3 : pole_distances = [500, 800, 1500, 2200]) :  -- distances in meters
  ∃ (total_time : ℝ),
    total_time = 243 ∧  -- 243 seconds
    total_time = first_pole_crossing_time +
      (pole_distances.map (λ d => d / (train_speed * 1000 / 3600))).sum :=
by sorry

end train_crossing_poles_time_l2069_206927


namespace arrangements_theorem_l2069_206994

/-- The number of arrangements of 5 people in a row with exactly 1 person between A and B -/
def arrangements_count : ℕ := 36

/-- The number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangements_theorem :
  ∀ (n : ℕ) (k : ℕ),
  n = total_people →
  k = people_between →
  arrangements_count = 36 :=
sorry

end arrangements_theorem_l2069_206994


namespace factor_million_three_ways_l2069_206949

/-- The number of ways to factor 1,000,000 into three factors, ignoring order -/
def factor_ways : ℕ := 139

/-- The prime factorization of 1,000,000 -/
def million_factorization : ℕ × ℕ := (6, 6)

theorem factor_million_three_ways :
  let (a, b) := million_factorization
  (2^a * 5^b = 1000000) →
  (factor_ways = 
    (1 : ℕ) + -- case where all factors are equal
    15 + -- case where exactly two factors are equal
    ((28 * 28 - 15 * 3 - 1) / 6 : ℕ) -- case where all factors are different
  ) := by sorry

end factor_million_three_ways_l2069_206949


namespace unique_quartic_polynomial_l2069_206917

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a*(x^3 : ℂ) + b*(x^2 : ℂ) + c*(x : ℂ) + d

theorem unique_quartic_polynomial 
  (q : ℝ → ℂ) 
  (monic : q = QuarticPolynomial (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ)) 
  (root_complex : q (1 - 3*I) = 0) 
  (root_zero : q 0 = -48) 
  (root_one : q 1 = 0) : 
  q = QuarticPolynomial (-7.8) 25.4 (-23.8) 48 := by
  sorry

end unique_quartic_polynomial_l2069_206917


namespace crayons_theorem_l2069_206952

/-- The number of crayons in the drawer at the end of Thursday. -/
def crayons_at_end_of_thursday (initial : ℕ) (mary_adds : ℕ) (john_removes : ℕ) (lisa_adds : ℕ) (jeremy_adds : ℕ) (sarah_removes : ℕ) : ℕ :=
  initial + mary_adds - john_removes + lisa_adds + jeremy_adds - sarah_removes

/-- Theorem stating that the number of crayons at the end of Thursday is 13. -/
theorem crayons_theorem : crayons_at_end_of_thursday 7 3 5 4 6 2 = 13 := by
  sorry

end crayons_theorem_l2069_206952


namespace polynomial_divisibility_l2069_206955

theorem polynomial_divisibility (n : ℤ) : 
  (∀ x : ℝ, (3 * x^2 + 5 * x + n) = (x - 2) * (3 * x + 11)) ↔ n = -22 :=
by sorry

end polynomial_divisibility_l2069_206955


namespace problem_solution_l2069_206964

theorem problem_solution (x y : ℝ) : (x - 1)^2 + Real.sqrt (y + 2) = 0 → x + y = -1 := by
  sorry

end problem_solution_l2069_206964


namespace quadratic_solution_existence_l2069_206928

theorem quadratic_solution_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - 2*x + a = 0) ↔ a ≤ 1 := by sorry

end quadratic_solution_existence_l2069_206928


namespace trig_identity_l2069_206932

theorem trig_identity : 
  1 / Real.cos (70 * π / 180) - 2 / Real.sin (70 * π / 180) = 
  2 * Real.tan (50 * π / 180) - 2 * Real.sqrt 3 := by
  sorry

end trig_identity_l2069_206932


namespace original_number_proof_l2069_206915

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 5) = 129 → x = 19 := by
  sorry

end original_number_proof_l2069_206915


namespace composite_sum_product_l2069_206996

theorem composite_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a^2 + a*c - c^2 = b^2 + b*d - d^2) : 
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ a*b + c*d = x*y :=
sorry

end composite_sum_product_l2069_206996


namespace three_rug_overlap_l2069_206933

theorem three_rug_overlap (total_area floor_area double_layer_area : ℝ) 
  (h1 : total_area = 200)
  (h2 : floor_area = 140)
  (h3 : double_layer_area = 24) : 
  ∃ (triple_layer_area : ℝ), 
    triple_layer_area = 18 ∧ 
    total_area = floor_area + double_layer_area + 2 * triple_layer_area :=
by
  sorry

end three_rug_overlap_l2069_206933


namespace max_value_inequality_l2069_206953

theorem max_value_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + 2*b + 3*c = 6) : 
  Real.sqrt (a + 1) + Real.sqrt (2*b + 1) + Real.sqrt (3*c + 1) ≤ 3 * Real.sqrt 3 := by
sorry

end max_value_inequality_l2069_206953


namespace log_expression_equality_l2069_206944

theorem log_expression_equality (a b c d e x y : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b) + Real.log (b^3 / c^2) + Real.log (c / d) + Real.log (d^2 / e) - Real.log (a^3 * y / (e^2 * x)) = Real.log ((b^2 * e * x) / (c * a * y)) :=
by sorry

end log_expression_equality_l2069_206944


namespace trig_identity_l2069_206970

theorem trig_identity : 4 * Real.cos (15 * π / 180) * Real.cos (75 * π / 180) - Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 3 / 4 := by
  sorry

end trig_identity_l2069_206970


namespace least_square_tiles_l2069_206914

theorem least_square_tiles (length width : ℕ) (h1 : length = 544) (h2 : width = 374) :
  let tile_size := Nat.gcd length width
  let num_tiles := (length * width) / (tile_size * tile_size)
  num_tiles = 50864 := by
sorry

end least_square_tiles_l2069_206914


namespace repair_cost_calculation_l2069_206992

def purchase_price : ℚ := 14000
def transportation_charges : ℚ := 1000
def selling_price : ℚ := 30000
def profit_percentage : ℚ := 50

theorem repair_cost_calculation (repair_cost : ℚ) :
  (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage / 100) = selling_price →
  repair_cost = 5000 := by
  sorry

end repair_cost_calculation_l2069_206992


namespace warehouse_optimization_l2069_206901

/-- Represents the warehouse dimensions and costs -/
structure Warehouse where
  x : ℝ  -- length of the iron fence (front)
  y : ℝ  -- length of one brick wall (side)
  iron_cost : ℝ := 40  -- cost per meter of iron fence
  brick_cost : ℝ := 45  -- cost per meter of brick wall
  top_cost : ℝ := 20  -- cost per square meter of the top
  budget : ℝ := 3200  -- total budget

/-- The total cost of the warehouse -/
def total_cost (w : Warehouse) : ℝ :=
  w.iron_cost * w.x + 2 * w.brick_cost * w.y + w.top_cost * w.x * w.y

/-- The area of the warehouse -/
def area (w : Warehouse) : ℝ :=
  w.x * w.y

/-- Theorem stating the maximum area and optimal dimensions -/
theorem warehouse_optimization (w : Warehouse) :
  (∀ w' : Warehouse, total_cost w' ≤ w.budget → area w' ≤ 100) ∧
  (∃ w' : Warehouse, total_cost w' ≤ w.budget ∧ area w' = 100) ∧
  (area w = 100 → total_cost w ≤ w.budget → w.x = 15) :=
sorry

end warehouse_optimization_l2069_206901


namespace area_of_constrained_region_l2069_206995

/-- The area of the region defined by specific constraints in a coordinate plane --/
theorem area_of_constrained_region : 
  let S := {p : ℝ × ℝ | p.1 ≤ 0 ∧ p.2 + p.1 - 1 ≥ 0 ∧ p.2 ≤ 4}
  MeasureTheory.volume S = 9/2 := by sorry

end area_of_constrained_region_l2069_206995


namespace dorothy_age_problem_l2069_206960

theorem dorothy_age_problem :
  let dorothy_age : ℕ := 15
  let sister_age : ℕ := dorothy_age / 3
  let years_later : ℕ := 5
  (dorothy_age + years_later) = 2 * (sister_age + years_later) :=
by sorry

end dorothy_age_problem_l2069_206960


namespace line_y_axis_intersection_l2069_206971

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * x - 7 * y = 35

/-- The y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ := (0, -5)

/-- Theorem: The point (0, -5) is the intersection of the line 5x - 7y = 35 with the y-axis -/
theorem line_y_axis_intersection :
  line_equation intersection_point.1 intersection_point.2 ∧
  y_axis intersection_point.1 := by
  sorry

end line_y_axis_intersection_l2069_206971


namespace min_sum_at_6_l2069_206903

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  first_term : a 1 = -14
  sum_5_6 : a 5 + a 6 = -4
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The value of n for which the sum reaches its minimum -/
def min_sum_index (seq : ArithmeticSequence) : ℕ :=
  6

/-- Theorem: The sum of the arithmetic sequence reaches its minimum when n = 6 -/
theorem min_sum_at_6 (seq : ArithmeticSequence) :
  ∀ n : ℕ, sum_n seq (min_sum_index seq) ≤ sum_n seq n :=
sorry

end min_sum_at_6_l2069_206903


namespace smallest_number_for_2_and_4_l2069_206958

def smallest_number (a b : ℕ) : ℕ := 
  if a ≤ b then 10 * a + b else 10 * b + a

theorem smallest_number_for_2_and_4 : 
  smallest_number 2 4 = 24 := by sorry

end smallest_number_for_2_and_4_l2069_206958


namespace lines_symmetric_about_y_axis_l2069_206946

/-- Two lines are symmetric about the y-axis if and only if their coefficients satisfy certain conditions -/
theorem lines_symmetric_about_y_axis 
  (m n p : ℝ) : 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ x + n * y + p = 0) ∧ 
  (∀ x y : ℝ, x + m * y + 5 = 0 ↔ -x + n * y + p = 0) ↔ 
  m = -n ∧ p = -5 := by sorry

end lines_symmetric_about_y_axis_l2069_206946


namespace modulo_congruence_solution_l2069_206918

theorem modulo_congruence_solution :
  ∃! k : ℤ, 0 ≤ k ∧ k < 17 ∧ -175 ≡ k [ZMOD 17] := by
  sorry

end modulo_congruence_solution_l2069_206918


namespace congruence_solution_l2069_206956

theorem congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 14 ∧ n ≡ 14589 [ZMOD 15] ∧ n = 9 := by
  sorry

end congruence_solution_l2069_206956


namespace opposite_solutions_k_value_l2069_206978

theorem opposite_solutions_k_value (x y k : ℝ) : 
  (2 * x + 5 * y = k) → 
  (x - 4 * y = 15) → 
  (x + y = 0) → 
  k = -9 := by
sorry

end opposite_solutions_k_value_l2069_206978


namespace vertex_C_coordinates_l2069_206937

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the median CM and altitude BH
def median_CM (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - p.2 - 5 = 0

def altitude_BH (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - 2 * p.2 - 5 = 0

-- Theorem statement
theorem vertex_C_coordinates (t : Triangle) :
  t.A = (5, 1) →
  median_CM t t.C →
  altitude_BH t t.B →
  (t.C.1 - t.A.1) * (t.B.1 - t.C.1) + (t.C.2 - t.A.2) * (t.B.2 - t.C.2) = 0 →
  t.C = (4, 3) := by
  sorry

end vertex_C_coordinates_l2069_206937


namespace matinee_children_count_l2069_206904

/-- Proves the number of children at a movie theater matinee --/
theorem matinee_children_count :
  let child_price : ℚ := 9/2
  let adult_price : ℚ := 27/4
  let total_receipts : ℚ := 405
  ∀ (num_adults : ℕ),
    (child_price * (num_adults + 20 : ℚ) + adult_price * num_adults = total_receipts) →
    (num_adults + 20 = 48) :=
by
  sorry

#check matinee_children_count

end matinee_children_count_l2069_206904


namespace chinese_digit_mapping_l2069_206909

/-- A function that maps Chinese characters to unique digits 1-9 -/
def ChineseToDigit : Type := Char → Fin 9

/-- The condition that the function maps different characters to different digits -/
def isInjective (f : ChineseToDigit) : Prop :=
  ∀ (c1 c2 : Char), f c1 = f c2 → c1 = c2

/-- The theorem statement -/
theorem chinese_digit_mapping (f : ChineseToDigit) 
  (h_injective : isInjective f)
  (h_zhu : f '祝' = 4)
  (h_he : f '贺' = 8) :
  (f '华') * 100 + (f '杯') * 10 + (f '赛') = 7632 := by
  sorry


end chinese_digit_mapping_l2069_206909
