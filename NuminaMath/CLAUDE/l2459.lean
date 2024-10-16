import Mathlib

namespace NUMINAMATH_CALUDE_salad_dressing_total_l2459_245905

/-- Given a ratio of ingredients and the amount of one ingredient, 
    calculate the total amount of all ingredients. -/
def total_ingredients (ratio : List Nat) (water_amount : Nat) : Nat :=
  let total_parts := ratio.sum
  let part_size := water_amount / ratio.head!
  part_size * total_parts

/-- Theorem: For a salad dressing with a water:olive oil:salt ratio of 3:2:1 
    and 15 cups of water, the total amount of ingredients is 30 cups. -/
theorem salad_dressing_total : 
  total_ingredients [3, 2, 1] 15 = 30 := by
  sorry

#eval total_ingredients [3, 2, 1] 15

end NUMINAMATH_CALUDE_salad_dressing_total_l2459_245905


namespace NUMINAMATH_CALUDE_nearest_integer_to_two_plus_sqrt_three_fourth_l2459_245988

theorem nearest_integer_to_two_plus_sqrt_three_fourth (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℤ), n = 194 ∧ |((2 : ℝ) + Real.sqrt 3)^4 - (n : ℝ)| < (1/2 : ℝ) + ε :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_two_plus_sqrt_three_fourth_l2459_245988


namespace NUMINAMATH_CALUDE_eightieth_digit_of_one_seventh_l2459_245950

def decimal_representation_of_one_seventh : List Nat := [1, 4, 2, 8, 5, 7]

theorem eightieth_digit_of_one_seventh : 
  (decimal_representation_of_one_seventh[(80 - 1) % decimal_representation_of_one_seventh.length] = 4) := by
  sorry

end NUMINAMATH_CALUDE_eightieth_digit_of_one_seventh_l2459_245950


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2459_245964

theorem logarithmic_equation_solution :
  ∀ x : ℝ, x > 0 → x ≠ 1 →
  (6 - (1 + 4 * 9^(4 - 2 * (Real.log 3 / Real.log (Real.sqrt 3)))) * (Real.log x / Real.log 7) = Real.log 7 / Real.log x) →
  (x = 7 ∨ x = Real.rpow 7 (1/5)) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2459_245964


namespace NUMINAMATH_CALUDE_seq1_infinitely_many_composites_seq2_infinitely_many_composites_l2459_245994

-- Define the first sequence
def seq1 (n : ℕ) : ℕ :=
  3^n * 10^n + 7

-- Define the second sequence
def seq2 (n : ℕ) : ℕ :=
  3^n * 10^n + 31

-- Statement for the first sequence
theorem seq1_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq1 n) :=
sorry

-- Statement for the second sequence
theorem seq2_infinitely_many_composites :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ (n : ℕ), n ∈ S → ¬ Nat.Prime (seq2 n) :=
sorry

end NUMINAMATH_CALUDE_seq1_infinitely_many_composites_seq2_infinitely_many_composites_l2459_245994


namespace NUMINAMATH_CALUDE_correlated_normal_distributions_l2459_245969

/-- Given two correlated normal distributions with specified parameters,
    this theorem proves the relationship between a value in the first distribution
    and its corresponding value in the second distribution. -/
theorem correlated_normal_distributions
  (μ₁ μ₂ σ₁ σ₂ ρ : ℝ)
  (h_μ₁ : μ₁ = 14.0)
  (h_μ₂ : μ₂ = 21.0)
  (h_σ₁ : σ₁ = 1.5)
  (h_σ₂ : σ₂ = 3.0)
  (h_ρ : ρ = 0.7)
  (x₁ : ℝ)
  (h_x₁ : x₁ = μ₁ - 2 * σ₁) :
  ∃ x₂ : ℝ, x₂ = μ₂ + ρ * (σ₂ / σ₁) * (x₁ - μ₁) :=
by
  sorry


end NUMINAMATH_CALUDE_correlated_normal_distributions_l2459_245969


namespace NUMINAMATH_CALUDE_cylinder_and_cone_properties_l2459_245960

/-- Properties of a cylinder and a cone --/
theorem cylinder_and_cone_properties 
  (base_area : ℝ) 
  (height : ℝ) 
  (cylinder_volume : ℝ) 
  (cone_volume : ℝ) 
  (h1 : base_area = 72) 
  (h2 : height = 6) 
  (h3 : cylinder_volume = base_area * height) 
  (h4 : cone_volume = (1/3) * cylinder_volume) : 
  cylinder_volume = 432 ∧ cone_volume = 144 := by
  sorry

#check cylinder_and_cone_properties

end NUMINAMATH_CALUDE_cylinder_and_cone_properties_l2459_245960


namespace NUMINAMATH_CALUDE_cookie_distribution_l2459_245978

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l2459_245978


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2459_245941

theorem inequality_solution_set (x : ℝ) :
  (((x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4) ↔ 
   (x ∈ Set.Ioc 0 (1/2) ∪ Set.Ioo (3/2) 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2459_245941


namespace NUMINAMATH_CALUDE_alex_remaining_money_l2459_245953

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def water_bill : ℝ := 55

theorem alex_remaining_money :
  weekly_income - (weekly_income * tax_rate + weekly_income * tithe_rate + water_bill) = 345 :=
by sorry

end NUMINAMATH_CALUDE_alex_remaining_money_l2459_245953


namespace NUMINAMATH_CALUDE_points_opposite_sides_iff_a_in_range_l2459_245956

/-- The coordinates of point A satisfy the given equation. -/
def point_A_equation (a x y : ℝ) : Prop :=
  5 * a^2 - 6 * a * x - 4 * a * y + 2 * x^2 + 2 * x * y + y^2 = 0

/-- The equation of the circle centered at point B. -/
def circle_B_equation (a x y : ℝ) : Prop :=
  a^2 * x^2 + a^2 * y^2 - 6 * a^2 * x - 2 * a^3 * y + 4 * a * y + a^4 + 4 = 0

/-- Points A and B lie on opposite sides of the line y = 1. -/
def opposite_sides (ya yb : ℝ) : Prop :=
  (ya - 1) * (yb - 1) < 0

/-- The main theorem statement. -/
theorem points_opposite_sides_iff_a_in_range (a : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    point_A_equation a xa ya ∧
    circle_B_equation a xb yb ∧
    opposite_sides ya yb) ↔
  (a > -1 ∧ a < 0) ∨ (a > 1 ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_points_opposite_sides_iff_a_in_range_l2459_245956


namespace NUMINAMATH_CALUDE_largest_number_with_sum_17_l2459_245943

/-- The largest number with all different digits whose sum is 17 -/
def largest_number : ℕ := 763210

/-- Function to get the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
  aux n []

/-- Theorem stating that 763210 is the largest number with all different digits whose sum is 17 -/
theorem largest_number_with_sum_17 :
  (∀ n : ℕ, n ≤ largest_number ∨
    (digits n).sum ≠ 17 ∨
    (digits n).length ≠ (digits n).toFinset.card) ∧
  (digits largest_number).sum = 17 ∧
  (digits largest_number).length = (digits largest_number).toFinset.card :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_17_l2459_245943


namespace NUMINAMATH_CALUDE_two_numbers_lcm_90_gcd_6_l2459_245922

theorem two_numbers_lcm_90_gcd_6 : ∃ (a b : ℕ+), 
  (¬(a ∣ b) ∧ ¬(b ∣ a)) ∧ 
  Nat.lcm a b = 90 ∧ 
  Nat.gcd a b = 6 ∧ 
  ({a, b} : Set ℕ+) = {18, 30} := by
sorry

end NUMINAMATH_CALUDE_two_numbers_lcm_90_gcd_6_l2459_245922


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equation_l2459_245957

/-- The canonical equation of the line of intersection of two planes -/
theorem line_intersection_canonical_equation 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ x y z, plane1 x y z ↔ x - y + z - 2 = 0)
  (h2 : ∀ x y z, plane2 x y z ↔ x - 2*y - z + 4 = 0) :
  ∃ t : ℝ, ∀ x y z, 
    (plane1 x y z ∧ plane2 x y z) ↔ 
    ((x - 8) / 3 = t ∧ (y - 6) / 2 = t ∧ z / (-1) = t) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equation_l2459_245957


namespace NUMINAMATH_CALUDE_power_of_power_l2459_245917

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2459_245917


namespace NUMINAMATH_CALUDE_money_difference_is_13_96_l2459_245923

def derek_initial : ℚ := 40
def derek_expenses : List ℚ := [14, 11, 5, 8]
def derek_discount_rate : ℚ := 0.1

def dave_initial : ℚ := 50
def dave_expenses : List ℚ := [7, 12, 9]
def dave_tax_rate : ℚ := 0.08

def calculate_remaining_money (initial : ℚ) (expenses : List ℚ) (rate : ℚ) (is_discount : Bool) : ℚ :=
  let total_expenses := expenses.sum
  let adjustment := total_expenses * rate
  if is_discount then
    initial - (total_expenses - adjustment)
  else
    initial - (total_expenses + adjustment)

theorem money_difference_is_13_96 :
  let derek_remaining := calculate_remaining_money derek_initial derek_expenses derek_discount_rate true
  let dave_remaining := calculate_remaining_money dave_initial dave_expenses dave_tax_rate false
  dave_remaining - derek_remaining = 13.96 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_is_13_96_l2459_245923


namespace NUMINAMATH_CALUDE_range_of_f_l2459_245926

noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2459_245926


namespace NUMINAMATH_CALUDE_f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l2459_245976

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * abs (x - a)

-- Statement 1: When a = 0, f is an even function
theorem f_even_when_a_zero :
  ∀ x : ℝ, f 0 x = f 0 (-x) := by sorry

-- Statement 2: When -1 < a < 1, f achieves a minimum value of a^2
theorem f_minimum_when_a_between_neg_one_and_one :
  ∀ a : ℝ, -1 < a → a < 1 → ∀ x : ℝ, f a x ≥ a^2 := by sorry

end NUMINAMATH_CALUDE_f_even_when_a_zero_f_minimum_when_a_between_neg_one_and_one_l2459_245976


namespace NUMINAMATH_CALUDE_carols_age_ratio_l2459_245984

theorem carols_age_ratio (carol alice betty : ℕ) : 
  carol = 5 * alice →
  alice = carol - 12 →
  betty = 6 →
  (carol : ℚ) / betty = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_carols_age_ratio_l2459_245984


namespace NUMINAMATH_CALUDE_ellipse_ratio_l2459_245983

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-focal length c,
    if a² + b² - 3c² = 0, then (a+c)/(a-c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_ratio_l2459_245983


namespace NUMINAMATH_CALUDE_dinner_pizzas_count_l2459_245947

-- Define the variables
def lunch_pizzas : ℕ := 9
def total_pizzas : ℕ := 15

-- Define the theorem
theorem dinner_pizzas_count : total_pizzas - lunch_pizzas = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_pizzas_count_l2459_245947


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2459_245930

theorem sqrt_simplification :
  Real.sqrt 32 - Real.sqrt 18 + Real.sqrt 8 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2459_245930


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l2459_245932

/-- Calculates the increase in wheel radius given original and new odometer readings -/
theorem wheel_radius_increase
  (original_radius : ℝ)
  (original_reading : ℝ)
  (new_reading : ℝ)
  (inches_per_mile : ℝ)
  (h1 : original_radius = 16)
  (h2 : original_reading = 1000)
  (h3 : new_reading = 980)
  (h4 : inches_per_mile = 62560) :
  ∃ (increase : ℝ), abs (increase - 0.33) < 0.005 :=
by sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_l2459_245932


namespace NUMINAMATH_CALUDE_total_interest_earned_l2459_245937

def initial_investment : ℝ := 2000
def interest_rate : ℝ := 0.12
def time_period : ℕ := 4

theorem total_interest_earned :
  let final_amount := initial_investment * (1 + interest_rate) ^ time_period
  final_amount - initial_investment = 1147.04 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_earned_l2459_245937


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2459_245966

theorem arithmetic_calculations :
  ((-20) + (-14) - (-18) - 13 = -29) ∧
  (-24 * ((-1/2) + (3/4) - (1/3)) = 2) ∧
  (-49 * (24/25) * 10 = -499.6) ∧
  (-(3^2) + (((-1/3) * (-3)) - ((8/5) / (2^2))) = -(8 + 2/5)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2459_245966


namespace NUMINAMATH_CALUDE_circle_distance_to_line_l2459_245924

/-- Given a circle (x-a)^2 + (y-a)^2 = 8 and the shortest distance from any point on the circle
    to the line y = -x is √2, prove that a = ±3 -/
theorem circle_distance_to_line (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - a)^2 = 8 →
    (∃ d : ℝ, d = Real.sqrt 2 ∧
      ∀ d' : ℝ, d' ≥ 0 → (x + y) / Real.sqrt 2 ≤ d')) →
  a = 3 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_distance_to_line_l2459_245924


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l2459_245938

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 218)
  (h2 : fuel_a_volume = 122)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  fuel_a_volume * fuel_a_ethanol_percent + (tank_capacity - fuel_a_volume) * fuel_b_ethanol_percent = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l2459_245938


namespace NUMINAMATH_CALUDE_log_27_3_l2459_245911

theorem log_27_3 : Real.log 3 / Real.log 27 = 1 / 3 := by
  have h : 27 = 3^3 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_27_3_l2459_245911


namespace NUMINAMATH_CALUDE_no_rational_cos_sqrt2_l2459_245967

theorem no_rational_cos_sqrt2 : ¬∃ (x : ℝ), (∃ (a b : ℚ), (Real.cos x + Real.sqrt 2 = a) ∧ (Real.cos (2 * x) + Real.sqrt 2 = b)) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_cos_sqrt2_l2459_245967


namespace NUMINAMATH_CALUDE_unique_solution_l2459_245999

def A : Nat := 89252525 -- ... (200 digits total)

def B (x y : Nat) : Nat := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

def digit_at (n : Nat) (pos : Nat) : Nat :=
  (n / (10 ^ (pos - 1))) % 10

theorem unique_solution :
  ∃! (x y : Nat),
    x < 10 ∧ y < 10 ∧
    digit_at (A * B x y) 53 = 1 ∧
    digit_at (A * B x y) 54 = 0 ∧
    x = 4 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2459_245999


namespace NUMINAMATH_CALUDE_f_difference_at_five_l2459_245908

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^4 + x^3 + 5*x

-- State the theorem
theorem f_difference_at_five : f 5 - f (-5) = 6550 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_five_l2459_245908


namespace NUMINAMATH_CALUDE_median_equation_l2459_245929

/-- The equation of median BD in triangle ABC -/
theorem median_equation (A B C D : ℝ × ℝ) : 
  A = (4, 1) → B = (0, 3) → C = (2, 4) → D = ((A.1 + C.1)/2, (A.2 + C.2)/2) →
  (fun (x y : ℝ) => x + 6*y - 18) = (fun (x y : ℝ) => 0) := by sorry

end NUMINAMATH_CALUDE_median_equation_l2459_245929


namespace NUMINAMATH_CALUDE_max_pons_is_nine_nine_pons_achievable_l2459_245997

/-- Represents the number of items Bill can buy -/
structure ItemCounts where
  pan : ℕ
  pin : ℕ
  pon : ℕ

/-- Calculates the total cost of the items -/
def totalCost (items : ItemCounts) : ℕ :=
  3 * items.pan + 5 * items.pin + 10 * items.pon

/-- Checks if the item counts satisfy the conditions -/
def isValid (items : ItemCounts) : Prop :=
  items.pan ≥ 1 ∧ items.pin ≥ 1 ∧ items.pon ≥ 1 ∧ totalCost items = 100

/-- The maximum number of pons that can be purchased -/
def maxPons : ℕ := 9

theorem max_pons_is_nine :
  ∀ items : ItemCounts, isValid items → items.pon ≤ maxPons :=
by sorry

theorem nine_pons_achievable :
  ∃ items : ItemCounts, isValid items ∧ items.pon = maxPons :=
by sorry

end NUMINAMATH_CALUDE_max_pons_is_nine_nine_pons_achievable_l2459_245997


namespace NUMINAMATH_CALUDE_problem_statement_l2459_245919

theorem problem_statement (a b : ℚ) (h1 : a = 1/2) (h2 : b = 1/3) : 
  (a - b) / (1/78) = 13 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2459_245919


namespace NUMINAMATH_CALUDE_prime_power_fraction_implies_prime_l2459_245986

theorem prime_power_fraction_implies_prime (n : ℕ) (h1 : n ≥ 2) :
  (∃ b : ℕ+, ∃ p : ℕ, ∃ k : ℕ, Prime p ∧ (b^n - 1) / (b - 1) = p^k) →
  Prime n :=
by sorry

end NUMINAMATH_CALUDE_prime_power_fraction_implies_prime_l2459_245986


namespace NUMINAMATH_CALUDE_eight_pow_plus_six_div_seven_l2459_245996

theorem eight_pow_plus_six_div_seven (n : ℕ) : 
  7 ∣ (8^n + 6) := by sorry

end NUMINAMATH_CALUDE_eight_pow_plus_six_div_seven_l2459_245996


namespace NUMINAMATH_CALUDE_integral_one_plus_sin_x_l2459_245982

theorem integral_one_plus_sin_x : ∫ x in (0)..(π/2), (1 + Real.sin x) = π/2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_one_plus_sin_x_l2459_245982


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l2459_245928

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ :=
  {x : ℝ | P x = c}

/-- Theorem: For any polynomial P of degree 8, if 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P (P 8)) :
  ∃ (x : ℝ), x ∈ A_P P (P 8) ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l2459_245928


namespace NUMINAMATH_CALUDE_divisibility_problem_l2459_245946

theorem divisibility_problem (n : ℤ) : 
  n > 101 →
  n % 101 = 0 →
  (∀ d : ℤ, d ∣ n → 1 < d → d < n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) →
  n % 100 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2459_245946


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2459_245979

/-- Given a cubic equation with distinct real roots between 0 and 1, 
    prove that the sum of the reciprocals of one minus each root equals 2/3 -/
theorem cubic_root_sum (a b c : ℝ) : 
  (24 * a^3 - 38 * a^2 + 18 * a - 1 = 0) →
  (24 * b^3 - 38 * b^2 + 18 * b - 1 = 0) →
  (24 * c^3 - 38 * c^2 + 18 * c - 1 = 0) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2459_245979


namespace NUMINAMATH_CALUDE_continued_fraction_simplification_l2459_245992

theorem continued_fraction_simplification :
  1 + (3 / (4 + (5 / 6))) = 47 / 29 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_simplification_l2459_245992


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l2459_245965

/-- Given polynomials f, g, and h, prove their sum is equal to the specified polynomial -/
theorem sum_of_polynomials (x : ℝ) : 
  let f := fun (x : ℝ) => -3*x^2 + x - 4
  let g := fun (x : ℝ) => -5*x^2 + 3*x - 8
  let h := fun (x : ℝ) => 5*x^2 + 5*x + 1
  f x + g x + h x = -3*x^2 + 9*x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l2459_245965


namespace NUMINAMATH_CALUDE_license_plate_increase_l2459_245934

/-- The number of possible letters in a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in a license plate. -/
def num_digits : ℕ := 10

/-- The number of letters in an old license plate. -/
def old_letters : ℕ := 3

/-- The number of digits in an old license plate. -/
def old_digits : ℕ := 2

/-- The number of letters in a new license plate. -/
def new_letters : ℕ := 2

/-- The number of digits in a new license plate. -/
def new_digits : ℕ := 4

/-- The theorem stating the increase in the number of possible license plates. -/
theorem license_plate_increase :
  (num_letters ^ new_letters * num_digits ^ new_digits) /
  (num_letters ^ old_letters * num_digits ^ old_digits) = 50 / 13 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l2459_245934


namespace NUMINAMATH_CALUDE_balance_implies_20g_difference_l2459_245952

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem balance_implies_20g_difference 
  (weights : Finset ℕ) 
  (h_weights : weights = Finset.range 40)
  (left_pan right_pan : Finset ℕ)
  (h_left : left_pan ⊆ weights ∧ left_pan.card = 10 ∧ ∀ n ∈ left_pan, is_even n)
  (h_right : right_pan ⊆ weights ∧ right_pan.card = 10 ∧ ∀ n ∈ right_pan, is_odd n)
  (h_balance : left_pan.sum id = right_pan.sum id) :
  ∃ (a b : ℕ), (a ∈ left_pan ∧ b ∈ left_pan ∧ a - b = 20) ∨ 
               (a ∈ right_pan ∧ b ∈ right_pan ∧ a - b = 20) :=
sorry

end NUMINAMATH_CALUDE_balance_implies_20g_difference_l2459_245952


namespace NUMINAMATH_CALUDE_positive_implies_increasing_exists_increasing_not_always_positive_l2459_245948

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Part 1: Sufficiency
theorem positive_implies_increasing :
  (∀ x, f x > 0) → MonotoneOn f Set.univ := by sorry

-- Part 2: Not Necessary
theorem exists_increasing_not_always_positive :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotoneOn f Set.univ ∧ ∃ x, f x ≤ 0 := by sorry

end NUMINAMATH_CALUDE_positive_implies_increasing_exists_increasing_not_always_positive_l2459_245948


namespace NUMINAMATH_CALUDE_josh_bracelets_l2459_245944

-- Define the parameters
def cost_per_bracelet : ℚ := 1
def selling_price : ℚ := 1.5
def cookie_cost : ℚ := 3
def money_left : ℚ := 3

-- Define the function to calculate the number of bracelets
def num_bracelets : ℚ := (cookie_cost + money_left) / (selling_price - cost_per_bracelet)

-- Theorem statement
theorem josh_bracelets : num_bracelets = 12 := by
  sorry

end NUMINAMATH_CALUDE_josh_bracelets_l2459_245944


namespace NUMINAMATH_CALUDE_polynomial_sum_simplification_l2459_245958

theorem polynomial_sum_simplification :
  let p₁ : Polynomial ℝ := 2 * X^3 - 3 * X^2 + 5 * X - 6
  let p₂ : Polynomial ℝ := 5 * X^4 - 2 * X^3 - 4 * X^2 - X + 8
  p₁ + p₂ = 5 * X^4 - 7 * X^2 + 4 * X + 2 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_sum_simplification_l2459_245958


namespace NUMINAMATH_CALUDE_quadratic_inequality_result_l2459_245945

theorem quadratic_inequality_result (x : ℝ) :
  x^2 - 5*x + 4 < 0 → 10 < x^2 + 4*x + 5 ∧ x^2 + 4*x + 5 < 37 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_result_l2459_245945


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2459_245931

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k * a.1 = b.1 ∧ k * a.2 = b.2)

/-- Theorem: If (1, 2) is parallel to (x, -4), then x = -2 -/
theorem parallel_vectors_imply_x_value :
  ∀ x : ℝ, parallel (1, 2) (x, -4) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2459_245931


namespace NUMINAMATH_CALUDE_safe_locks_theorem_l2459_245972

/-- The number of people in the commission -/
def n : ℕ := 9

/-- The minimum number of people required to access the safe -/
def k : ℕ := 6

/-- The number of keys per lock -/
def keys_per_lock : ℕ := n - k + 1

/-- The number of locks required -/
def num_locks : ℕ := Nat.choose n keys_per_lock

theorem safe_locks_theorem : 
  num_locks = Nat.choose n keys_per_lock :=
by sorry

end NUMINAMATH_CALUDE_safe_locks_theorem_l2459_245972


namespace NUMINAMATH_CALUDE_inequality_proof_l2459_245995

theorem inequality_proof (n : ℕ) (h : n > 1) : 1 + n * 2^((n-1)/2) < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2459_245995


namespace NUMINAMATH_CALUDE_museum_trip_cost_l2459_245903

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_trip_cost : total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_cost_l2459_245903


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2459_245907

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℝ
  dozen : ℝ
  two_dozen : ℝ

/-- Calculates the maximum number of roses that can be purchased given a budget and price structure -/
def max_roses (budget : ℝ) (prices : RosePrices) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased with $680 -/
theorem max_roses_for_680 :
  let prices : RosePrices := {
    individual := 4.5,
    dozen := 36,
    two_dozen := 50
  }
  max_roses 680 prices = 318 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2459_245907


namespace NUMINAMATH_CALUDE_container_capacity_increase_l2459_245902

/-- Proves that quadrupling all dimensions of a container increases its capacity by a factor of 64 -/
theorem container_capacity_increase (original_capacity new_capacity : ℝ) : 
  original_capacity = 5 → new_capacity = 320 → new_capacity = original_capacity * 64 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_increase_l2459_245902


namespace NUMINAMATH_CALUDE_stairs_theorem_l2459_245935

def stairs_problem (samir veronica ravi : ℕ) : Prop :=
  samir = 318 ∧
  veronica = (samir / 2) + 18 ∧
  ravi = 2 * veronica ∧
  samir + veronica + ravi = 849

theorem stairs_theorem : ∃ samir veronica ravi : ℕ, stairs_problem samir veronica ravi :=
  sorry

end NUMINAMATH_CALUDE_stairs_theorem_l2459_245935


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l2459_245970

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

theorem eighth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) (h₃ : a₃ = 4/3) :
  geometric_sequence a₁ (a₂ / a₁) 8 = 4/729 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l2459_245970


namespace NUMINAMATH_CALUDE_log_equation_solution_l2459_245939

theorem log_equation_solution (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q ≠ 2) :
  Real.log p + Real.log q = Real.log (2 * (p + q)) → p = (2 * (q - 1)) / (q - 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2459_245939


namespace NUMINAMATH_CALUDE_max_k_value_l2459_245980

theorem max_k_value (a b k : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → a^2 + b^2 ≥ k → k ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l2459_245980


namespace NUMINAMATH_CALUDE_petya_stickers_l2459_245906

/-- Calculates the final number of stickers after a series of trades -/
def final_stickers (initial : ℕ) (trade_in : ℕ) (trade_out : ℕ) (num_trades : ℕ) : ℕ :=
  initial + num_trades * (trade_out - trade_in)

/-- Theorem: Petya will have 121 stickers after 30 trades -/
theorem petya_stickers :
  final_stickers 1 1 5 30 = 121 := by
  sorry

end NUMINAMATH_CALUDE_petya_stickers_l2459_245906


namespace NUMINAMATH_CALUDE_quadratic_solution_set_l2459_245913

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3 * x + 2

-- Define the solution set
def solution_set (a b : ℝ) : Set ℝ := {x | b < x ∧ x < 1}

-- Theorem statement
theorem quadratic_solution_set (a b : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a b) →
  a = -5 ∧ b = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_set_l2459_245913


namespace NUMINAMATH_CALUDE_business_profit_l2459_245901

/-- Represents a business with spending and income -/
structure Business where
  spending : ℕ
  income : ℕ

/-- Calculates the profit of a business -/
def profit (b : Business) : ℕ := b.income - b.spending

/-- Theorem stating the profit for a business with given conditions -/
theorem business_profit :
  ∀ (b : Business),
  (b.spending : ℚ) / b.income = 5 / 9 →
  b.income = 108000 →
  profit b = 48000 := by
  sorry

end NUMINAMATH_CALUDE_business_profit_l2459_245901


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l2459_245993

theorem divisibility_by_seven (n : ℕ) : 
  (3^(12*n^2 + 1) + 2^(6*n + 2)) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l2459_245993


namespace NUMINAMATH_CALUDE_dealer_profit_approx_89_99_l2459_245916

/-- Calculates the dealer's profit percentage given the number of articles purchased,
    total purchase price, number of articles sold, and total selling price. -/
def dealer_profit_percentage (articles_purchased : ℕ) (purchase_price : ℚ) 
                             (articles_sold : ℕ) (selling_price : ℚ) : ℚ :=
  let cp_per_article := purchase_price / articles_purchased
  let sp_per_article := selling_price / articles_sold
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

/-- Theorem stating that the dealer's profit percentage is approximately 89.99%
    given the specific conditions of the problem. -/
theorem dealer_profit_approx_89_99 :
  ∃ ε > 0, abs (dealer_profit_percentage 15 25 12 38 - 89.99) < ε :=
sorry

end NUMINAMATH_CALUDE_dealer_profit_approx_89_99_l2459_245916


namespace NUMINAMATH_CALUDE_set_equivalences_l2459_245949

/-- The set of non-negative even numbers not greater than 10 -/
def nonNegEvenSet : Set ℕ :=
  {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

/-- The set of prime numbers not greater than 10 -/
def primeSet : Set ℕ :=
  {n : ℕ | Nat.Prime n ∧ n ≤ 10}

/-- The equation x^2 + 2x - 15 = 0 -/
def equation (x : ℝ) : Prop :=
  x^2 + 2*x - 15 = 0

theorem set_equivalences :
  (nonNegEvenSet = {0, 2, 4, 6, 8, 10}) ∧
  (primeSet = {2, 3, 5, 7}) ∧
  ({x : ℝ | equation x} = {-5, 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_equivalences_l2459_245949


namespace NUMINAMATH_CALUDE_coefficient_x3y5_l2459_245974

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expression
def expression (x y : ℚ) : ℚ := (2/3 * x - 4/5 * y)^8

-- State the theorem
theorem coefficient_x3y5 :
  (binomial 8 3) * (2/3)^3 * (-4/5)^5 = -458752/84375 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_l2459_245974


namespace NUMINAMATH_CALUDE_seven_ninths_rounded_l2459_245968

/-- Rounds a rational number to a specified number of decimal places -/
noncomputable def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (↑(⌊q * 10^places + 1/2⌋) : ℚ) / 10^places

/-- The fraction 7/9 rounded to 2 decimal places equals 0.78 -/
theorem seven_ninths_rounded : round_to_decimal_places (7/9) 2 = 78/100 := by
  sorry

end NUMINAMATH_CALUDE_seven_ninths_rounded_l2459_245968


namespace NUMINAMATH_CALUDE_cells_after_3_hours_l2459_245915

/-- Represents the number of cells after a given number of half-hour intervals -/
def cellCount (n : ℕ) : ℕ := 2^n

/-- The number of half-hour intervals in 3 hours -/
def intervalsIn3Hours : ℕ := 6

theorem cells_after_3_hours : 
  cellCount intervalsIn3Hours = 64 := by sorry

end NUMINAMATH_CALUDE_cells_after_3_hours_l2459_245915


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_l2459_245912

theorem cube_root_of_negative_one : ∃ x : ℝ, x^3 = -1 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_l2459_245912


namespace NUMINAMATH_CALUDE_equation_solution_l2459_245936

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (7 * x - 2) - Real.sqrt (3 * x - 1) = 2) ∧ (x = 0.515625) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2459_245936


namespace NUMINAMATH_CALUDE_min_angle_BFE_l2459_245991

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Main theorem
theorem min_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∀ θ : ℕ, (θ : ℝ) = angle B F E → θ ≥ 113 :=
sorry

end NUMINAMATH_CALUDE_min_angle_BFE_l2459_245991


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l2459_245998

/-- Represents the meeting point of two people walking towards each other along a line of lamps -/
def meeting_point (total_lamps : ℕ) (p_start p_pos : ℕ) (v_start v_pos : ℕ) : ℕ :=
  let p_traveled := p_pos - p_start
  let v_traveled := v_start - v_pos
  let remaining_distance := v_pos - p_pos
  let total_intervals := total_lamps - 1
  let p_speed := p_traveled
  let v_speed := v_traveled
  p_pos + (remaining_distance * p_speed) / (p_speed + v_speed)

/-- Theorem stating that Petya and Vasya meet at lamp 64 -/
theorem petya_vasya_meeting :
  let total_lamps : ℕ := 100
  let petya_start : ℕ := 1
  let vasya_start : ℕ := 100
  let petya_position : ℕ := 22
  let vasya_position : ℕ := 88
  meeting_point total_lamps petya_start petya_position vasya_start vasya_position = 64 := by
  sorry

#eval meeting_point 100 1 22 100 88

end NUMINAMATH_CALUDE_petya_vasya_meeting_l2459_245998


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_all_nines_l2459_245942

def is_all_nines (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^k - 1

theorem smallest_multiple_of_seven_all_nines :
  ∃ N : ℕ, (N = 142857 ∧
            is_all_nines (7 * N) ∧
            ∀ m : ℕ, m < N → ¬is_all_nines (7 * m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_all_nines_l2459_245942


namespace NUMINAMATH_CALUDE_parallel_condition_l2459_245963

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define the parallel relation between two lines
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ a x y ↔ l₂ a x y

-- Theorem statement
theorem parallel_condition (a : ℝ) : parallel a ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2459_245963


namespace NUMINAMATH_CALUDE_rose_bush_cost_l2459_245955

def total_cost : ℕ := 4100
def num_rose_bushes : ℕ := 20
def gardener_hourly_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def soil_volume : ℕ := 100
def soil_cost_per_unit : ℕ := 5

theorem rose_bush_cost : 
  (total_cost - (gardener_hourly_rate * gardener_hours_per_day * gardener_days) - 
   (soil_volume * soil_cost_per_unit)) / num_rose_bushes = 150 := by
  sorry

end NUMINAMATH_CALUDE_rose_bush_cost_l2459_245955


namespace NUMINAMATH_CALUDE_planes_distance_zero_l2459_245925

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x + 2*y - z = 3
def plane2 (x y z : ℝ) : Prop := 2*x + 4*y - 2*z = 6

-- Define the distance function between two planes
noncomputable def distance_between_planes (p1 p2 : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem planes_distance_zero :
  distance_between_planes plane1 plane2 = 0 := by sorry

end NUMINAMATH_CALUDE_planes_distance_zero_l2459_245925


namespace NUMINAMATH_CALUDE_handshake_count_l2459_245990

/-- Represents a company at the convention -/
inductive Company
| A | B | C | D | E

/-- The number of companies at the convention -/
def num_companies : Nat := 5

/-- The number of representatives per company -/
def reps_per_company : Nat := 4

/-- The total number of attendees at the convention -/
def total_attendees : Nat := num_companies * reps_per_company

/-- Determines if two companies are the same -/
def same_company (c1 c2 : Company) : Bool :=
  match c1, c2 with
  | Company.A, Company.A => true
  | Company.B, Company.B => true
  | Company.C, Company.C => true
  | Company.D, Company.D => true
  | Company.E, Company.E => true
  | _, _ => false

/-- Determines if a company is Company A -/
def is_company_a (c : Company) : Bool :=
  match c with
  | Company.A => true
  | _ => false

/-- Calculates the number of handshakes for a given company -/
def handshakes_for_company (c : Company) : Nat :=
  if is_company_a c then
    reps_per_company * (total_attendees - reps_per_company)
  else
    reps_per_company * (total_attendees - 2 * reps_per_company)

/-- The total number of handshakes at the convention -/
def total_handshakes : Nat :=
  (handshakes_for_company Company.A +
   handshakes_for_company Company.B +
   handshakes_for_company Company.C +
   handshakes_for_company Company.D +
   handshakes_for_company Company.E) / 2

/-- The main theorem stating that the total number of handshakes is 128 -/
theorem handshake_count : total_handshakes = 128 := by
  sorry


end NUMINAMATH_CALUDE_handshake_count_l2459_245990


namespace NUMINAMATH_CALUDE_intersection_circle_line_symmetry_l2459_245914

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y - 1

-- Define the circle equation
def circle_equation (m n p : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + m*x + n*y + p = 0

-- Define symmetry about y = x
def symmetric_about_y_eq_x (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = y₂ ∧ y₁ = x₂

-- Main theorem
theorem intersection_circle_line_symmetry 
  (m n p : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  line_equation m x₁ y₁ →
  line_equation m x₂ y₂ →
  circle_equation m n p x₁ y₁ →
  circle_equation m n p x₂ y₂ →
  symmetric_about_y_eq_x x₁ y₁ x₂ y₂ →
  p < -3/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_circle_line_symmetry_l2459_245914


namespace NUMINAMATH_CALUDE_green_square_area_percentage_l2459_245951

/-- Represents a square flag with a symmetrical cross -/
structure FlagWithCross where
  side : ℝ
  crossWidth : ℝ
  crossArea : ℝ
  greenSquareArea : ℝ

/-- Properties of the flag with cross -/
def FlagWithCross.properties (flag : FlagWithCross) : Prop :=
  flag.side > 0 ∧
  flag.crossWidth > 0 ∧
  flag.crossWidth < flag.side / 2 ∧
  flag.crossArea = 0.49 * flag.side * flag.side ∧
  flag.greenSquareArea = (flag.side - 2 * flag.crossWidth) * (flag.side - 2 * flag.crossWidth)

/-- Theorem: If the cross occupies 49% of the flag area, the green square occupies 25% -/
theorem green_square_area_percentage (flag : FlagWithCross) 
  (h : flag.properties) : flag.greenSquareArea = 0.25 * flag.side * flag.side := by
  sorry


end NUMINAMATH_CALUDE_green_square_area_percentage_l2459_245951


namespace NUMINAMATH_CALUDE_chapter_page_difference_l2459_245959

theorem chapter_page_difference (chapter1 chapter2 chapter3 : ℕ) 
  (h1 : chapter1 = 35)
  (h2 : chapter2 = 18)
  (h3 : chapter3 = 3) :
  chapter2 - chapter3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_chapter_page_difference_l2459_245959


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l2459_245962

theorem arithmetic_progression_sum (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ 
  4 < a ∧ a < b ∧ b < 16 ∧
  (b - a = a - 4) ∧ 
  (16 - b = b - a) ∧
  (b - a ≠ a - 4) →
  a + b = 20 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l2459_245962


namespace NUMINAMATH_CALUDE_solve_for_b_l2459_245971

theorem solve_for_b (a b : ℝ) 
  (eq1 : a * (a - 4) = 21)
  (eq2 : b * (b - 4) = 21)
  (neq : a ≠ b)
  (sum : a + b = 4) :
  b = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_b_l2459_245971


namespace NUMINAMATH_CALUDE_percentage_problem_l2459_245921

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2459_245921


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2459_245927

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2459_245927


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l2459_245900

theorem arithmetic_expression_equals_one :
  2016 * 2014 - 2013 * 2015 + 2012 * 2015 - 2013 * 2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l2459_245900


namespace NUMINAMATH_CALUDE_polynomial_factor_l2459_245977

-- Define the polynomial
def P (a b d x : ℝ) : ℝ := a * x^4 + b * x^3 + 27 * x^2 + d * x + 10

-- Define the factor
def F (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 2

-- Theorem statement
theorem polynomial_factor (a b d : ℝ) :
  (∃ (e f : ℝ), ∀ x, P a b d x = F x * (e * x^2 + f * x + 5)) →
  a = 2 ∧ b = -13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l2459_245977


namespace NUMINAMATH_CALUDE_sin_double_pi_minus_theta_l2459_245975

theorem sin_double_pi_minus_theta (θ : ℝ) 
  (h1 : 3 * (Real.cos θ)^2 = Real.tan θ + 3) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.sin (2 * (Real.pi - θ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_pi_minus_theta_l2459_245975


namespace NUMINAMATH_CALUDE_custom_op_example_l2459_245904

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a - 2 * b

-- State the theorem
theorem custom_op_example : custom_op 2 (-3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2459_245904


namespace NUMINAMATH_CALUDE_revenue_in_scientific_notation_l2459_245981

/-- Represents 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions -/
def revenue : ℝ := 2.93

theorem revenue_in_scientific_notation : 
  revenue * billion = 2.93 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_revenue_in_scientific_notation_l2459_245981


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l2459_245909

theorem polygon_sides_when_interior_thrice_exterior : ℕ → Prop :=
  fun n =>
    (180 * (n - 2) = 3 * 360) →
    n = 8

-- The proof is omitted
theorem polygon_sides_when_interior_thrice_exterior_proof :
  polygon_sides_when_interior_thrice_exterior 8 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_polygon_sides_when_interior_thrice_exterior_proof_l2459_245909


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2459_245961

theorem painted_cube_theorem (n : ℕ) (h1 : n > 2) :
  (12 * (n - 2) : ℝ) = (n - 2)^3 → n = 2 * Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l2459_245961


namespace NUMINAMATH_CALUDE_cos_75_minus_cos_15_l2459_245920

theorem cos_75_minus_cos_15 : Real.cos (75 * π / 180) - Real.cos (15 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_minus_cos_15_l2459_245920


namespace NUMINAMATH_CALUDE_james_bed_purchase_l2459_245918

/-- The price James pays for a bed and bed frame with a discount -/
theorem james_bed_purchase (bed_frame_price : ℝ) (bed_price_multiplier : ℕ) (discount_percentage : ℝ) : 
  bed_frame_price = 75 →
  bed_price_multiplier = 10 →
  discount_percentage = 0.2 →
  (bed_frame_price + bed_frame_price * bed_price_multiplier) * (1 - discount_percentage) = 660 :=
by sorry

end NUMINAMATH_CALUDE_james_bed_purchase_l2459_245918


namespace NUMINAMATH_CALUDE_orange_cost_l2459_245987

/-- Given Alexander's shopping scenario, prove the cost of each orange. -/
theorem orange_cost (apple_price : ℝ) (apple_count : ℕ) (orange_count : ℕ) (total_spent : ℝ) :
  apple_price = 1 →
  apple_count = 5 →
  orange_count = 2 →
  total_spent = 9 →
  (total_spent - apple_price * apple_count) / orange_count = 2 := by
sorry

end NUMINAMATH_CALUDE_orange_cost_l2459_245987


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2459_245973

theorem absolute_value_inequality (x : ℝ) : |x - 2| > 2 - x ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2459_245973


namespace NUMINAMATH_CALUDE_group_b_sample_size_l2459_245954

/-- Calculates the number of cities to be sampled from a group in stratified sampling -/
def stratified_sample_size (total_cities : ℕ) (group_cities : ℕ) (sample_size : ℕ) : ℕ :=
  (group_cities * sample_size) / total_cities

/-- Proves that the number of cities to be sampled from Group B is 4 -/
theorem group_b_sample_size :
  let total_cities : ℕ := 36
  let group_b_cities : ℕ := 12
  let sample_size : ℕ := 12
  stratified_sample_size total_cities group_b_cities sample_size = 4 := by
  sorry

#eval stratified_sample_size 36 12 12

end NUMINAMATH_CALUDE_group_b_sample_size_l2459_245954


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l2459_245910

/-- Given a rectangle A with perimeter 32 cm and length twice its width,
    and a square B with area one-third of rectangle A's area,
    prove that the perimeter of square B is 64√3/9 cm. -/
theorem rectangle_perimeter_problem (width_A : ℝ) (length_A : ℝ) (side_B : ℝ) :
  width_A > 0 →
  length_A = 2 * width_A →
  2 * (length_A + width_A) = 32 →
  side_B^2 = (1/3) * (length_A * width_A) →
  4 * side_B = (64 * Real.sqrt 3) / 9 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l2459_245910


namespace NUMINAMATH_CALUDE_third_year_planting_l2459_245985

def initial_planting : ℝ := 10000
def annual_increase : ℝ := 0.2

def acres_planted (year : ℕ) : ℝ :=
  initial_planting * (1 + annual_increase) ^ (year - 1)

theorem third_year_planting :
  acres_planted 3 = 14400 := by sorry

end NUMINAMATH_CALUDE_third_year_planting_l2459_245985


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2459_245933

/-- Given a hyperbola with equation x²-y²/b²=1 and b > 0, 
    prove that if one of its asymptote lines is 2x-y=0, then b = 2 -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 - y^2/b^2 = 1 → (2*x - y = 0 → b = 2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2459_245933


namespace NUMINAMATH_CALUDE_exactly_three_special_triangles_l2459_245989

/-- A right-angled triangle with integer sides where the area is twice the perimeter -/
structure SpecialTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  right_angled : a.val^2 + b.val^2 = c.val^2
  area_perimeter : (a.val * b.val : ℕ) = 4 * (a.val + b.val + c.val)

/-- There are exactly three special triangles -/
theorem exactly_three_special_triangles : 
  ∃! (list : List SpecialTriangle), list.length = 3 ∧ 
  (∀ t : SpecialTriangle, t ∈ list) ∧
  (∀ t ∈ list, t ∈ [⟨9, 40, 41, sorry, sorry⟩, ⟨10, 24, 26, sorry, sorry⟩, ⟨12, 16, 20, sorry, sorry⟩]) :=
sorry

end NUMINAMATH_CALUDE_exactly_three_special_triangles_l2459_245989


namespace NUMINAMATH_CALUDE_gcd_15893_35542_l2459_245940

theorem gcd_15893_35542 : Nat.gcd 15893 35542 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15893_35542_l2459_245940
