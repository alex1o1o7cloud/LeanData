import Mathlib

namespace NUMINAMATH_CALUDE_jeff_storage_used_l4170_417022

/-- Calculates the storage Jeff is already using on his phone. -/
def storage_used (total_capacity : ℕ) (song_size : ℕ) (num_songs : ℕ) (mb_per_gb : ℕ) : ℕ :=
  total_capacity - (song_size * num_songs) / mb_per_gb

/-- Proves that Jeff is already using 4 GB of storage on his phone. -/
theorem jeff_storage_used :
  storage_used 16 30 400 1000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jeff_storage_used_l4170_417022


namespace NUMINAMATH_CALUDE_max_distance_for_given_car_l4170_417031

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

end NUMINAMATH_CALUDE_max_distance_for_given_car_l4170_417031


namespace NUMINAMATH_CALUDE_inequality_holds_for_all_x_l4170_417016

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, x^2 - x - m + 1 > 0) ↔ m < 3/4 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_for_all_x_l4170_417016


namespace NUMINAMATH_CALUDE_square_roots_problem_l4170_417067

theorem square_roots_problem (x a b : ℝ) (hx : x > 0) 
  (h_roots : x = a^2 ∧ x = (a + b)^2) (h_sum : 2*a + b = 0) :
  (a = -2 → b = 4 ∧ x = 4) ∧
  (b = 6 → a = -3 ∧ x = 9) ∧
  (a^2*x + (a + b)^2*x = 8 → x = 2) := by
sorry

end NUMINAMATH_CALUDE_square_roots_problem_l4170_417067


namespace NUMINAMATH_CALUDE_abc_remainder_mod_seven_l4170_417015

theorem abc_remainder_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 →
  (a + 2*b + 3*c) % 7 = 1 →
  (2*a + 3*b + c) % 7 = 2 →
  (3*a + b + 2*c) % 7 = 1 →
  (a*b*c) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_seven_l4170_417015


namespace NUMINAMATH_CALUDE_total_toys_cost_l4170_417065

def toy_cars_cost : ℚ := 14.88
def skateboard_cost : ℚ := 4.88
def toy_trucks_cost : ℚ := 5.86
def pants_cost : ℚ := 14.55

theorem total_toys_cost :
  toy_cars_cost + skateboard_cost + toy_trucks_cost = 25.62 := by sorry

end NUMINAMATH_CALUDE_total_toys_cost_l4170_417065


namespace NUMINAMATH_CALUDE_greatest_integer_x_prime_l4170_417050

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def f (x : ℤ) : ℤ := |6 * x^2 - 47 * x + 15|

theorem greatest_integer_x_prime :
  ∀ x : ℤ, (is_prime (f x).toNat → x ≤ 8) ∧
  (is_prime (f 8).toNat) :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_prime_l4170_417050


namespace NUMINAMATH_CALUDE_distance_for_50L_800cc_l4170_417013

/-- The distance traveled using a given amount of diesel and engine capacity -/
def distance_traveled (diesel : ℝ) (engine_capacity : ℝ) : ℝ := sorry

/-- The volume of diesel required varies directly as the capacity of the engine -/
axiom diesel_engine_relation (d1 d2 e1 e2 dist : ℝ) :
  d1 / e1 = d2 / e2 → distance_traveled d1 e1 = distance_traveled d2 e2

theorem distance_for_50L_800cc :
  distance_traveled 50 800 = 6 :=
by
  have h1 : distance_traveled 100 1200 = 800 := sorry
  sorry

end NUMINAMATH_CALUDE_distance_for_50L_800cc_l4170_417013


namespace NUMINAMATH_CALUDE_elijah_card_decks_l4170_417068

theorem elijah_card_decks (total_cards : ℕ) (cards_per_deck : ℕ) (h1 : total_cards = 312) (h2 : cards_per_deck = 52) :
  total_cards / cards_per_deck = 6 :=
by sorry

end NUMINAMATH_CALUDE_elijah_card_decks_l4170_417068


namespace NUMINAMATH_CALUDE_survey_C_most_suitable_for_census_l4170_417099

-- Define a structure for a survey
structure Survey where
  description : String
  population_size : ℕ
  resource_requirement : ℕ

-- Define the suitability for census method
def suitable_for_census (s : Survey) : Prop :=
  s.population_size ≤ 100 ∧ s.resource_requirement ≤ 50

-- Define the four survey options
def survey_A : Survey :=
  { description := "Quality and safety of local grain processing",
    population_size := 1000,
    resource_requirement := 200 }

def survey_B : Survey :=
  { description := "Viewership ratings of the 2023 CCTV Spring Festival Gala",
    population_size := 1000000,
    resource_requirement := 500000 }

def survey_C : Survey :=
  { description := "Weekly duration of physical exercise for a ninth-grade class",
    population_size := 50,
    resource_requirement := 30 }

def survey_D : Survey :=
  { description := "Household chores participation of junior high school students in the entire city",
    population_size := 100000,
    resource_requirement := 10000 }

-- Theorem stating that survey C is the most suitable for census method
theorem survey_C_most_suitable_for_census :
  suitable_for_census survey_C ∧
  (¬ suitable_for_census survey_A ∧
   ¬ suitable_for_census survey_B ∧
   ¬ suitable_for_census survey_D) :=
by sorry


end NUMINAMATH_CALUDE_survey_C_most_suitable_for_census_l4170_417099


namespace NUMINAMATH_CALUDE_x_times_x_minus_one_eq_six_is_quadratic_l4170_417056

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x(x-1) = 6 -/
def f (x : ℝ) : ℝ := x * (x - 1) - 6

theorem x_times_x_minus_one_eq_six_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_times_x_minus_one_eq_six_is_quadratic_l4170_417056


namespace NUMINAMATH_CALUDE_third_month_sale_l4170_417032

def sales_problem (m1 m2 m4 m5 m6 average : ℕ) : Prop :=
  ∃ m3 : ℕ,
    m3 = 6 * average - (m1 + m2 + m4 + m5 + m6) ∧
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = average

theorem third_month_sale :
  sales_problem 5420 5660 6350 6500 8270 6400 →
  ∃ m3 : ℕ, m3 = 6200
:= by sorry

end NUMINAMATH_CALUDE_third_month_sale_l4170_417032


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_23m_and_33n_l4170_417063

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_23m_and_33n :
  (∀ k < 24, ¬(factorial k % (23 * k) = 0)) ∧
  (factorial 24 % (23 * 24) = 0) ∧
  (∀ k < 12, ¬(factorial k % (33 * k) = 0)) ∧
  (factorial 12 % (33 * 12) = 0) := by
  sorry

#check smallest_factorial_divisible_by_23m_and_33n

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_23m_and_33n_l4170_417063


namespace NUMINAMATH_CALUDE_f_of_5_equals_0_l4170_417098

theorem f_of_5_equals_0 (f : ℝ → ℝ) (h : ∀ x, f (2 * x + 1) = x^2 - 2*x) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_0_l4170_417098


namespace NUMINAMATH_CALUDE_fruit_salad_mixture_weight_l4170_417064

theorem fruit_salad_mixture_weight 
  (apple peach grape : ℝ) 
  (h1 : apple / grape = 12 / 7)
  (h2 : peach / grape = 8 / 7)
  (h3 : apple = grape + 10) :
  apple + peach + grape = 54 := by
sorry

end NUMINAMATH_CALUDE_fruit_salad_mixture_weight_l4170_417064


namespace NUMINAMATH_CALUDE_gcd_factorial_8_and_cube_factorial_6_l4170_417095

theorem gcd_factorial_8_and_cube_factorial_6 :
  Nat.gcd (Nat.factorial 8) (Nat.factorial 6 ^ 3) = 11520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_and_cube_factorial_6_l4170_417095


namespace NUMINAMATH_CALUDE_savings_calculation_l4170_417000

theorem savings_calculation (income expenditure savings : ℕ) : 
  income = 18000 →
  9 * expenditure = 8 * income →
  savings = income - expenditure →
  savings = 2000 := by sorry

end NUMINAMATH_CALUDE_savings_calculation_l4170_417000


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l4170_417046

theorem quadratic_root_difference (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ > 2 ∧ r₂ ≤ 2 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂)) → 
  ∃ r₁ r₂ : ℝ, r₁ - r₂ = 1 ∧ 
   x^2 - 5*x + 6 = (x - r₁) * (x - r₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l4170_417046


namespace NUMINAMATH_CALUDE_paco_cookies_eaten_l4170_417033

/-- Given that Paco had 17 cookies, gave 13 to his friend, and ate 1 more than he gave away,
    prove that Paco ate 14 cookies. -/
theorem paco_cookies_eaten (initial : ℕ) (given : ℕ) (eaten : ℕ) 
    (h1 : initial = 17)
    (h2 : given = 13)
    (h3 : eaten = given + 1) : 
  eaten = 14 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_eaten_l4170_417033


namespace NUMINAMATH_CALUDE_james_out_of_pocket_cost_l4170_417025

theorem james_out_of_pocket_cost (doctor_charge : ℝ) (insurance_coverage_percent : ℝ) 
  (h1 : doctor_charge = 300)
  (h2 : insurance_coverage_percent = 80) :
  doctor_charge * (1 - insurance_coverage_percent / 100) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_out_of_pocket_cost_l4170_417025


namespace NUMINAMATH_CALUDE_simple_interest_rate_l4170_417073

/-- Calculate the rate of simple interest given principal, time, and interest amount -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (interest : ℝ) 
  (h1 : principal = 20000)
  (h2 : time = 3)
  (h3 : interest = 7200) : 
  (interest * 100) / (principal * time) = 12 := by
  sorry

#check simple_interest_rate

end NUMINAMATH_CALUDE_simple_interest_rate_l4170_417073


namespace NUMINAMATH_CALUDE_hotel_profit_maximization_l4170_417012

/-- Represents the hotel profit maximization problem -/
theorem hotel_profit_maximization
  (total_rooms : ℕ)
  (base_price : ℝ)
  (price_increment : ℝ)
  (vacancy_increment : ℕ)
  (expense_per_room : ℝ)
  (max_profit_price : ℝ) :
  total_rooms = 50 →
  base_price = 180 →
  price_increment = 10 →
  vacancy_increment = 1 →
  expense_per_room = 20 →
  max_profit_price = 350 →
  ∀ price : ℝ,
    price ≥ base_price →
    let occupied_rooms := total_rooms - (price - base_price) / price_increment * vacancy_increment
    let profit := (price - expense_per_room) * occupied_rooms
    profit ≤ (max_profit_price - expense_per_room) * (total_rooms - (max_profit_price - base_price) / price_increment * vacancy_increment) :=
by
  sorry

#check hotel_profit_maximization

end NUMINAMATH_CALUDE_hotel_profit_maximization_l4170_417012


namespace NUMINAMATH_CALUDE_distance_range_l4170_417080

/-- Hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Focal length of the hyperbola -/
def focal_length : ℝ := 4

/-- Right triangle ABD formed by intersection with perpendicular line through right focus -/
def right_triangle_ABD : Prop := sorry

/-- Slopes of lines AM and AN -/
def slope_product (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -2

/-- Distance from A to line MN -/
def distance_A_to_MN (d : ℝ) : Prop := sorry

/-- Theorem stating the range of distance d -/
theorem distance_range :
  ∀ (d : ℝ), hyperbola_C 1 0 →
  focal_length = 4 →
  right_triangle_ABD →
  (∃ k₁ k₂, slope_product k₁ k₂) →
  distance_A_to_MN d →
  3 * Real.sqrt 3 < d ∧ d ≤ 6 := by sorry

end NUMINAMATH_CALUDE_distance_range_l4170_417080


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_l4170_417042

theorem inhabitable_earth_surface (total_surface area_land area_inhabitable : ℝ) :
  area_land = (1 / 3 : ℝ) * total_surface →
  area_inhabitable = (3 / 4 : ℝ) * area_land →
  area_inhabitable = (1 / 4 : ℝ) * total_surface :=
by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_l4170_417042


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l4170_417092

theorem rectangle_area_perimeter_sum (a b : ℕ+) : 
  let A := (a : ℕ) * b
  let P := 2 * ((a : ℕ) + b)
  A + P ≠ 102 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_sum_l4170_417092


namespace NUMINAMATH_CALUDE_quadratic_functions_problem_l4170_417085

/-- Given two quadratic functions y₁ and y₂ satisfying certain conditions, 
    prove that α = 1, y₁ = -2x² + 4x + 3, and y₂ = 3x² + 12x + 10 -/
theorem quadratic_functions_problem 
  (y₁ y₂ : ℝ → ℝ) 
  (α : ℝ) 
  (h_α_pos : α > 0)
  (h_y₁_max : ∀ x, y₁ x ≤ y₁ α)
  (h_y₁_max_val : y₁ α = 5)
  (h_y₂_α : y₂ α = 25)
  (h_y₂_min : ∀ x, y₂ x ≥ -2)
  (h_sum : ∀ x, y₁ x + y₂ x = x^2 + 16*x + 13) :
  α = 1 ∧ 
  (∀ x, y₁ x = -2*x^2 + 4*x + 3) ∧ 
  (∀ x, y₂ x = 3*x^2 + 12*x + 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_functions_problem_l4170_417085


namespace NUMINAMATH_CALUDE_total_amount_raised_l4170_417096

/-- Represents the sizes of rubber ducks --/
inductive DuckSize
  | Small
  | Medium
  | Large

/-- Calculates the price of a duck given its size --/
def price (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 2
  | DuckSize.Medium => 3
  | DuckSize.Large => 5

/-- Calculates the bulk discount rate for a given size and quantity --/
def bulkDiscountRate (s : DuckSize) (quantity : ℕ) : ℚ :=
  match s with
  | DuckSize.Small => if quantity ≥ 10 then 0.1 else 0
  | DuckSize.Medium => if quantity ≥ 15 then 0.15 else 0
  | DuckSize.Large => if quantity ≥ 20 then 0.2 else 0

/-- Returns the sales tax rate for a given duck size --/
def salesTaxRate (s : DuckSize) : ℚ :=
  match s with
  | DuckSize.Small => 0.05
  | DuckSize.Medium => 0.07
  | DuckSize.Large => 0.09

/-- Calculates the total amount raised for a given duck size and quantity --/
def amountRaised (s : DuckSize) (quantity : ℕ) : ℚ :=
  let basePrice := price s * quantity
  let discountedPrice := basePrice * (1 - bulkDiscountRate s quantity)
  discountedPrice * (1 + salesTaxRate s)

/-- Theorem stating the total amount raised for charity --/
theorem total_amount_raised :
  amountRaised DuckSize.Small 150 +
  amountRaised DuckSize.Medium 221 +
  amountRaised DuckSize.Large 185 = 1693.1 := by
  sorry


end NUMINAMATH_CALUDE_total_amount_raised_l4170_417096


namespace NUMINAMATH_CALUDE_arithmetic_seq_sum_l4170_417014

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Given an arithmetic sequence with S_5 = 20, prove that a_2 + a_3 + a_4 = 12 -/
theorem arithmetic_seq_sum (seq : ArithmeticSequence) (h : seq.S 5 = 20) :
  seq.a 2 + seq.a 3 + seq.a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_sum_l4170_417014


namespace NUMINAMATH_CALUDE_katrina_cookies_l4170_417071

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies Katrina sold in the morning -/
def morning_sales : ℕ := 3 * dozen

/-- The number of cookies Katrina sold during lunch rush -/
def lunch_sales : ℕ := 57

/-- The number of cookies Katrina sold in the afternoon -/
def afternoon_sales : ℕ := 16

/-- The number of cookies Katrina has left to take home -/
def cookies_left : ℕ := 11

/-- The total number of cookies Katrina had initially -/
def initial_cookies : ℕ := morning_sales + lunch_sales + afternoon_sales + cookies_left

theorem katrina_cookies : initial_cookies = 120 := by sorry

end NUMINAMATH_CALUDE_katrina_cookies_l4170_417071


namespace NUMINAMATH_CALUDE_larger_number_problem_l4170_417079

theorem larger_number_problem (x y : ℝ) : 
  y = x + 10 →  -- One number exceeds another by 10
  x = y / 2 →   -- The smaller number is half the larger number
  x + y = 34 →  -- Their sum is 34
  y = 20        -- The larger number is 20
:= by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l4170_417079


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4170_417038

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 40*x + 350 ≤ 0} = Set.Icc 10 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4170_417038


namespace NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l4170_417036

/-- The gain percentage of a shopkeeper using false weights -/
theorem shopkeeper_gain_percentage 
  (claimed_weight : ℝ) 
  (actual_weight : ℝ) 
  (claimed_weight_is_kg : claimed_weight = 1000) 
  (actual_weight_used : actual_weight = 980) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 980) / 980 * 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_gain_percentage_l4170_417036


namespace NUMINAMATH_CALUDE_max_pencils_buyable_l4170_417074

/-- Represents the number of pencils in a set -/
inductive PencilSet
| Large : PencilSet  -- 20 pencils
| Small : PencilSet  -- 5 pencils

/-- Represents the rebate percentage for a given set -/
def rebate_percentage (s : PencilSet) : ℚ :=
  match s with
  | PencilSet.Large => 25 / 100
  | PencilSet.Small => 10 / 100

/-- Represents the number of pencils in a given set -/
def pencils_in_set (s : PencilSet) : ℕ :=
  match s with
  | PencilSet.Large => 20
  | PencilSet.Small => 5

/-- The initial number of pencils Vasya can afford -/
def initial_pencils : ℕ := 30

/-- Theorem stating the maximum number of pencils Vasya can buy -/
theorem max_pencils_buyable :
  ∃ (large_sets small_sets : ℕ),
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small +
    ⌊large_sets * (rebate_percentage PencilSet.Large * pencils_in_set PencilSet.Large : ℚ)⌋ +
    ⌊small_sets * (rebate_percentage PencilSet.Small * pencils_in_set PencilSet.Small : ℚ)⌋ = 41 ∧
    large_sets * pencils_in_set PencilSet.Large +
    small_sets * pencils_in_set PencilSet.Small ≤ initial_pencils :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_buyable_l4170_417074


namespace NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l4170_417040

theorem sqrt_product_equals_sqrt_of_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_sqrt_of_product_l4170_417040


namespace NUMINAMATH_CALUDE_line_intercept_product_l4170_417001

/-- Given a line with equation y + 3 = -2(x + 5), 
    the product of its x-intercept and y-intercept is 84.5 -/
theorem line_intercept_product : 
  ∀ (x y : ℝ), y + 3 = -2 * (x + 5) → 
  ∃ (x_int y_int : ℝ), 
    (x_int + 5 = -13/2) ∧ 
    (y_int + 3 = -2 * 5) ∧ 
    (x_int * y_int = 84.5) := by
  sorry


end NUMINAMATH_CALUDE_line_intercept_product_l4170_417001


namespace NUMINAMATH_CALUDE_seed_germination_problem_l4170_417083

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot1 overall_germination_rate : ℚ)
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 1/5)
  (h4 : overall_germination_rate = 13/50) :
  (overall_germination_rate * (seeds_plot1 + seeds_plot2) - germination_rate_plot1 * seeds_plot1) / seeds_plot2 = 7/20 :=
sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l4170_417083


namespace NUMINAMATH_CALUDE_pears_for_apples_l4170_417075

/-- The cost of fruits in a common unit -/
structure FruitCost where
  apple : ℕ
  orange : ℕ
  pear : ℕ

/-- The relationship between apple and orange costs -/
def apple_orange_relation (fc : FruitCost) : Prop :=
  10 * fc.apple = 5 * fc.orange

/-- The relationship between orange and pear costs -/
def orange_pear_relation (fc : FruitCost) : Prop :=
  4 * fc.orange = 6 * fc.pear

/-- The main theorem: Nancy can buy 15 pears for the price of 20 apples -/
theorem pears_for_apples (fc : FruitCost) 
  (h1 : apple_orange_relation fc) 
  (h2 : orange_pear_relation fc) : 
  20 * fc.apple = 15 * fc.pear :=
by sorry

end NUMINAMATH_CALUDE_pears_for_apples_l4170_417075


namespace NUMINAMATH_CALUDE_count_non_multiples_eq_546_l4170_417023

/-- The count of three-digit numbers that are not multiples of 3 or 11 -/
def count_non_multiples : ℕ :=
  let total_three_digit := 999 - 100 + 1
  let multiples_of_3 := (999 / 3) - (99 / 3)
  let multiples_of_11 := (990 / 11) - (99 / 11)
  let multiples_of_33 := (990 / 33) - (99 / 33)
  total_three_digit - (multiples_of_3 + multiples_of_11 - multiples_of_33)

theorem count_non_multiples_eq_546 : count_non_multiples = 546 := by
  sorry

end NUMINAMATH_CALUDE_count_non_multiples_eq_546_l4170_417023


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l4170_417072

theorem complex_arithmetic_expression : 
  ((520 * 0.43) / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l4170_417072


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4170_417059

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^5 + 1 = (x^2 - 3*x + 5) * q + (11*x - 14) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4170_417059


namespace NUMINAMATH_CALUDE_sqrt_2023_bounds_l4170_417049

theorem sqrt_2023_bounds : 40 < Real.sqrt 2023 ∧ Real.sqrt 2023 < 45 := by
  have h1 : 1600 < 2023 := by sorry
  have h2 : 2023 < 2025 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_bounds_l4170_417049


namespace NUMINAMATH_CALUDE_debby_zoo_pictures_l4170_417087

theorem debby_zoo_pictures : ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
  museum_pics = 12 →
  deleted_pics = 14 →
  remaining_pics = 22 →
  zoo_pics + museum_pics - deleted_pics = remaining_pics →
  zoo_pics = 24 := by
sorry

end NUMINAMATH_CALUDE_debby_zoo_pictures_l4170_417087


namespace NUMINAMATH_CALUDE_union_of_sets_l4170_417003

/-- Given sets A and B, prove that their union is [-1, +∞) -/
theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | -3 ≤ 2*x - 1 ∧ 2*x - 1 ≤ 3}) →
  (B = {x : ℝ | x > 1}) →
  A ∪ B = Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l4170_417003


namespace NUMINAMATH_CALUDE_evaluate_g_l4170_417070

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

-- State the theorem
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l4170_417070


namespace NUMINAMATH_CALUDE_probability_under_20_l4170_417055

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 130) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 4 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_under_20_l4170_417055


namespace NUMINAMATH_CALUDE_gravelingCostIs3600_l4170_417018

/-- Represents the dimensions and cost parameters of a rectangular lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ

/-- Calculates the total cost of graveling two intersecting roads in a rectangular lawn -/
def totalGravelingCost (lawn : LawnWithRoads) : ℝ :=
  let lengthRoadArea := lawn.length * lawn.roadWidth
  let widthRoadArea := (lawn.width - lawn.roadWidth) * lawn.roadWidth
  let totalArea := lengthRoadArea + widthRoadArea
  totalArea * lawn.costPerSqm

/-- Theorem stating that the total cost of graveling for the given lawn is 3600 -/
theorem gravelingCostIs3600 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 80)
  (h2 : lawn.width = 50)
  (h3 : lawn.roadWidth = 10)
  (h4 : lawn.costPerSqm = 3) :
  totalGravelingCost lawn = 3600 := by
  sorry

#eval totalGravelingCost { length := 80, width := 50, roadWidth := 10, costPerSqm := 3 }

end NUMINAMATH_CALUDE_gravelingCostIs3600_l4170_417018


namespace NUMINAMATH_CALUDE_octagon_perimeter_l4170_417069

/-- The perimeter of a regular octagon with side length 2 units is 16 units. -/
theorem octagon_perimeter : ℕ → ℕ → ℕ
  | 8, 2 => 16
  | _, _ => 0

#check octagon_perimeter

end NUMINAMATH_CALUDE_octagon_perimeter_l4170_417069


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_k_values_l4170_417047

-- Define the quadratic equation
def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + k*(x - 1)

-- Theorem 1: The quadratic equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) : 
  ∃ x : ℝ, quadratic_equation k x = 0 :=
sorry

-- Theorem 2: If the roots satisfy the given condition, k is either 4 or -1
theorem k_values (k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    quadratic_equation k x₁ = 0 ∧ 
    quadratic_equation k x₂ = 0 ∧ 
    x₁^2 + x₂^2 = 7 - x₁*x₂) →
  (k = 4 ∨ k = -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_k_values_l4170_417047


namespace NUMINAMATH_CALUDE_work_completion_time_l4170_417010

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

end NUMINAMATH_CALUDE_work_completion_time_l4170_417010


namespace NUMINAMATH_CALUDE_intersection_intercept_sum_l4170_417076

/-- Given two lines that intersect at (3, 6), prove their y-intercepts sum to -9 -/
theorem intersection_intercept_sum (c d : ℝ) : 
  (3 = 2 * 6 + c) →  -- First line passes through (3, 6)
  (6 = 2 * 3 + d) →  -- Second line passes through (3, 6)
  c + d = -9 := by
sorry

end NUMINAMATH_CALUDE_intersection_intercept_sum_l4170_417076


namespace NUMINAMATH_CALUDE_lizard_eyes_count_l4170_417027

theorem lizard_eyes_count :
  ∀ (E W S : ℕ),
  W = 3 * E →
  S = 7 * W →
  E = S + W - 69 →
  E = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_lizard_eyes_count_l4170_417027


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l4170_417024

def triangle_numbers : Finset ℕ := {1, 3, 5, 7, 9, 11}

theorem triangle_sum_theorem (vertex_sum midpoint_sum : ℕ → ℕ → ℕ → ℕ) 
  (h1 : vertex_sum 19 19 19 = (vertex_sum 1 3 5 + vertex_sum 7 9 11))
  (h2 : ∀ a b c, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers → 
    a ≠ b ∧ b ≠ c ∧ a ≠ c → vertex_sum a b c + midpoint_sum a b c = 19)
  (h3 : ∀ a b c d e f, a ∈ triangle_numbers → b ∈ triangle_numbers → c ∈ triangle_numbers →
    d ∈ triangle_numbers → e ∈ triangle_numbers → f ∈ triangle_numbers →
    a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a → 
    vertex_sum a c e + midpoint_sum b d f = 19) :
  ∃ a b c, a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ vertex_sum a b c = 21 :=
sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l4170_417024


namespace NUMINAMATH_CALUDE_inequality_properties_l4170_417034

theorem inequality_properties (a b c d : ℝ) :
  (a > b ∧ c > d → a + c > b + d) ∧
  (a > b ∧ b > 0 ∧ c < 0 → c / a > c / b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l4170_417034


namespace NUMINAMATH_CALUDE_rectangle_properties_l4170_417048

theorem rectangle_properties (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : (x + 5) * (y + 5) - (x - 2) * (y - 2) = 196) :
  (2 * (x + y) = 50) ∧ 
  (∃ k : ℤ, (x + 5) * (y + 5) - (x - 2) * (y - 2) = 7 * k) ∧
  (x = y + 5 → ∃ a b : ℕ, a * b = (x + 5) * (y + 5) ∧ (a = x ∨ b = x) ∧ (a = y + 5 ∨ b = y + 5)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_properties_l4170_417048


namespace NUMINAMATH_CALUDE_quadratic_point_on_graph_l4170_417090

theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_on_graph_l4170_417090


namespace NUMINAMATH_CALUDE_negation_of_and_zero_l4170_417019

theorem negation_of_and_zero (x y : ℝ) : ¬(x = 0 ∧ y = 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_and_zero_l4170_417019


namespace NUMINAMATH_CALUDE_loan_problem_l4170_417091

/-- Proves that given the conditions of the loan problem, the time A lent money to C is 2/3 of a year. -/
theorem loan_problem (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  time_B = 2 →
  rate = 9 / 100 →
  total_interest = 1980 →
  total_interest = principal_B * rate * time_B + principal_C * rate * (2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_loan_problem_l4170_417091


namespace NUMINAMATH_CALUDE_min_tangent_product_right_triangle_l4170_417020

theorem min_tangent_product_right_triangle (A B C : Real) : 
  0 < A → A < π / 2 →
  0 < B → B < π / 2 →
  C ≤ π / 2 →
  A + B + C = π →
  (∀ A' B' C', 0 < A' → A' < π / 2 → 0 < B' → B' < π / 2 → C' ≤ π / 2 → A' + B' + C' = π → 
    Real.tan A * Real.tan B ≤ Real.tan A' * Real.tan B') →
  C = π / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_tangent_product_right_triangle_l4170_417020


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l4170_417094

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem four_digit_divisible_by_9 (B : ℕ) :
  B < 10 →
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) →
  B = 6 := by
  sorry

#check four_digit_divisible_by_9

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l4170_417094


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_seven_l4170_417088

theorem complex_magnitude_equals_seven (t : ℝ) (h1 : t > 0) :
  Complex.abs (3 + t * Complex.I) = 7 → t = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_seven_l4170_417088


namespace NUMINAMATH_CALUDE_perfect_negative_correlation_l4170_417037

/-- Represents a pair of data points (x, y) -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Calculates the sample correlation coefficient for a list of data points -/
def sampleCorrelationCoefficient (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem: For any set of paired sample data that fall on a straight line with negative slope,
    the sample correlation coefficient is -1 -/
theorem perfect_negative_correlation 
  (data : List DataPoint) 
  (h_line : ∃ (m : ℝ) (b : ℝ), m < 0 ∧ ∀ (point : DataPoint), point ∈ data → point.y = m * point.x + b) :
  sampleCorrelationCoefficient data = -1 :=
sorry

end NUMINAMATH_CALUDE_perfect_negative_correlation_l4170_417037


namespace NUMINAMATH_CALUDE_negation_of_proposition_l4170_417078

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l4170_417078


namespace NUMINAMATH_CALUDE_circle_properties_l4170_417077

-- Define the circle C
def Circle (t s r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - t)^2 + (p.2 - s)^2 = r^2}

-- Define the condition for independence of x₀ and y₀
def IndependentSum (t s r a b : ℝ) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ Circle t s r →
    ∃ (k : ℝ), |x₀ - y₀ + a| + |x₀ - y₀ + b| = k

-- Main theorem
theorem circle_properties
  (t s r a b : ℝ)
  (h_r : r > 0)
  (h_ab : a ≠ b)
  (h_ind : IndependentSum t s r a b) :
  (|a - b| = 2 * Real.sqrt 2 * r →
    ∃ (m n : ℝ), ∀ (x y : ℝ), (x, y) ∈ Circle t s r → m * x + n * y = 1) ∧
  (|a - b| = 2 * Real.sqrt 2 →
    r ≤ 1 ∧ ∃ (t₀ s₀ : ℝ), r = 1 ∧ (t₀, s₀) ∈ Circle t s r) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l4170_417077


namespace NUMINAMATH_CALUDE_johns_remaining_money_l4170_417060

def remaining_money (initial_amount : ℚ) (snack_fraction : ℚ) (necessity_fraction : ℚ) : ℚ :=
  let after_snacks := initial_amount * (1 - snack_fraction)
  after_snacks * (1 - necessity_fraction)

theorem johns_remaining_money :
  remaining_money 20 (1/5) (3/4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l4170_417060


namespace NUMINAMATH_CALUDE_solve_system_l4170_417021

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_solve_system_l4170_417021


namespace NUMINAMATH_CALUDE_sean_money_difference_l4170_417005

theorem sean_money_difference (fritz_money : ℕ) (rick_sean_total : ℕ) : 
  fritz_money = 40 →
  rick_sean_total = 96 →
  ∃ (sean_money : ℕ),
    sean_money > fritz_money / 2 ∧
    3 * sean_money + sean_money = rick_sean_total ∧
    sean_money - fritz_money / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_sean_money_difference_l4170_417005


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_l4170_417030

/-- Given a right triangle with legs in ratio 3:7 and altitude to hypotenuse of 42,
    prove that the altitude divides the hypotenuse into segments of length 18 and 98 -/
theorem right_triangle_hypotenuse_segments
  (a b c h : ℝ)
  (right_angle : a^2 + b^2 = c^2)
  (leg_ratio : b = 7/3 * a)
  (altitude : h = 42)
  (geo_mean : a * b = h^2) :
  ∃ (x y : ℝ), x + y = c ∧ x * y = h^2 ∧ x = 18 ∧ y = 98 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_segments_l4170_417030


namespace NUMINAMATH_CALUDE_ceiling_minus_x_l4170_417011

theorem ceiling_minus_x (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ α : ℝ, 0 < α ∧ α < 1 ∧ x = ⌊x⌋ + α ∧ ⌈x⌉ - x = 1 - α :=
sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_l4170_417011


namespace NUMINAMATH_CALUDE_autumn_pencil_count_l4170_417017

/-- Calculates the final number of pencils Autumn has -/
def final_pencil_count (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (bought : ℕ) : ℕ :=
  initial - (misplaced + broken) + (found + bought)

/-- Theorem stating that Autumn's final pencil count is correct -/
theorem autumn_pencil_count :
  final_pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencil_count_l4170_417017


namespace NUMINAMATH_CALUDE_count_valid_integers_valid_integers_formula_correct_l4170_417062

/-- The number of n-digit decimal integers using only digits 1, 2, and 3,
    and containing each of these digits at least once. -/
def validIntegers (n : ℕ+) : ℕ :=
  3^n.val - 3 * 2^n.val + 3

/-- Theorem stating that validIntegers gives the correct count. -/
theorem count_valid_integers (n : ℕ+) :
  validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

/-- Proof that the formula is correct for all positive integers n. -/
theorem valid_integers_formula_correct :
  ∀ n : ℕ+, validIntegers n = (3^n.val - 3 * 2^n.val + 3) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_integers_valid_integers_formula_correct_l4170_417062


namespace NUMINAMATH_CALUDE_opposite_sides_line_condition_l4170_417028

theorem opposite_sides_line_condition (a : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = -1 ∧ y2 = -4 ∧ 
    ((a * x1 + 3 * y1 + 1) * (a * x2 + 3 * y2 + 1) < 0)) ↔ 
  (a < -11 ∨ a > -10) := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_condition_l4170_417028


namespace NUMINAMATH_CALUDE_sin_period_2x_minus_pi_div_6_l4170_417039

/-- The minimum positive period of y = sin(2x - π/6) is π -/
theorem sin_period_2x_minus_pi_div_6 (x : ℝ) :
  let f := fun x => Real.sin (2 * x - π / 6)
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = π :=
sorry

end NUMINAMATH_CALUDE_sin_period_2x_minus_pi_div_6_l4170_417039


namespace NUMINAMATH_CALUDE_total_weight_jack_and_sam_l4170_417004

theorem total_weight_jack_and_sam : 
  ∀ (jack_weight sam_weight : ℕ),
  jack_weight = 52 →
  jack_weight = sam_weight + 8 →
  jack_weight + sam_weight = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_total_weight_jack_and_sam_l4170_417004


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4170_417086

/-- Given a hyperbola with the following properties:
    1. A line is drawn through the left focus F₁ at a 30° angle
    2. This line intersects the right branch of the hyperbola at point P
    3. A circle with diameter PF₁ passes through the right focus F₂
    Then the eccentricity of the hyperbola is √3 -/
theorem hyperbola_eccentricity (F₁ F₂ P : ℝ × ℝ) (a b c : ℝ) :
  let e := c / a
  (P.1 = c ∧ P.2 = b^2 / a) →  -- P is on the right branch
  (P.2 / (2 * c) = Real.tan (30 * π / 180)) →  -- Line through F₁ is at 30°
  (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)) →  -- Circle condition
  e = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4170_417086


namespace NUMINAMATH_CALUDE_area_after_shortening_other_side_l4170_417044

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side by 2 -/
def shortened : Rectangle := { length := 3, width := 7 }

/-- The rectangle after shortening the other side by 2 -/
def other_shortened : Rectangle := { length := 5, width := 5 }

theorem area_after_shortening_other_side :
  shortened.area = 21 → other_shortened.area = 25 := by sorry

end NUMINAMATH_CALUDE_area_after_shortening_other_side_l4170_417044


namespace NUMINAMATH_CALUDE_reciprocal_inequality_for_negative_numbers_l4170_417045

theorem reciprocal_inequality_for_negative_numbers (a b : ℝ) 
  (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_for_negative_numbers_l4170_417045


namespace NUMINAMATH_CALUDE_cubic_root_sum_l4170_417009

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 3 + a * (Complex.I + 2) + b = 0 → a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l4170_417009


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4170_417097

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4170_417097


namespace NUMINAMATH_CALUDE_johns_total_earnings_l4170_417029

/-- Calculates the total earnings given the initial bonus, growth rate, and new salary -/
def total_earnings (initial_bonus : ℝ) (growth_rate : ℝ) (new_salary : ℝ) : ℝ :=
  let new_bonus := initial_bonus * (1 + growth_rate)
  new_salary + new_bonus

/-- Theorem: John's total earnings this year are $210,500 -/
theorem johns_total_earnings :
  let initial_bonus : ℝ := 10000
  let growth_rate : ℝ := 0.05
  let new_salary : ℝ := 200000
  total_earnings initial_bonus growth_rate new_salary = 210500 := by
  sorry

#eval total_earnings 10000 0.05 200000

end NUMINAMATH_CALUDE_johns_total_earnings_l4170_417029


namespace NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l4170_417058

def is_single_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem largest_product_of_three_primes_digit_sum :
  ∃ (n a b : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    a ≠ b ∧
    is_prime a ∧
    is_prime b ∧
    is_prime (a + b) ∧
    n = a * b * (a + b) ∧
    (∀ (m : ℕ), 
      (∃ (x y : ℕ), 
        is_single_digit x ∧ 
        is_single_digit y ∧ 
        x ≠ y ∧ 
        is_prime x ∧ 
        is_prime y ∧ 
        is_prime (x + y) ∧ 
        m = x * y * (x + y)) → m ≤ n) ∧
    sum_of_digits n = 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_of_three_primes_digit_sum_l4170_417058


namespace NUMINAMATH_CALUDE_grocer_decaf_percentage_l4170_417052

/-- Calculates the percentage of decaffeinated coffee in a grocer's stock --/
def decaf_percentage (initial_stock : ℝ) (type_a_percent : ℝ) (type_b_percent : ℝ) (type_c_percent : ℝ)
  (type_a_decaf : ℝ) (type_b_decaf : ℝ) (type_c_decaf : ℝ)
  (additional_stock : ℝ) (additional_a_percent : ℝ) (type_d_decaf : ℝ) : ℝ := by
  sorry

theorem grocer_decaf_percentage :
  decaf_percentage 800 0.40 0.35 0.25 0.20 0.30 0.45 200 0.50 0.65 = 32.3 := by
  sorry

end NUMINAMATH_CALUDE_grocer_decaf_percentage_l4170_417052


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_third_sector_l4170_417057

/-- The radius of an inscribed circle in a sector that is one-third of a circle -/
theorem inscribed_circle_radius_in_third_sector (R : ℝ) (h : R = 5) :
  let r := (R * Real.sqrt 3 - R) / 2
  r * (1 + Real.sqrt 3) = R :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_third_sector_l4170_417057


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4170_417007

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l4170_417007


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l4170_417008

/-- Given a quadratic expression 8k^2 - 12k + 20, prove that when rewritten
    in the form c(k + p)^2 + q, the ratio q/p equals -62/3 -/
theorem quadratic_rewrite_ratio (k : ℝ) :
  ∃ (c p q : ℝ), 
    (8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) ∧ 
    (q / p = -62 / 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l4170_417008


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l4170_417089

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_libraries_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_libraries_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l4170_417089


namespace NUMINAMATH_CALUDE_number_problem_l4170_417043

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 17 → (40/100 : ℝ) * N = 204 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4170_417043


namespace NUMINAMATH_CALUDE_simplify_expression_l4170_417081

theorem simplify_expression (a b k : ℝ) (h1 : a + b = -k) (h2 : a * b = -3) :
  (a - 3) * (b - 3) = 6 + 3 * k := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4170_417081


namespace NUMINAMATH_CALUDE_scrap_metal_collection_l4170_417002

theorem scrap_metal_collection (a b : Nat) :
  a < 10 ∧ b < 10 ∧ 
  (900 + 10 * a + b) - (100 * a + 10 * b + 9) = 216 →
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by sorry

end NUMINAMATH_CALUDE_scrap_metal_collection_l4170_417002


namespace NUMINAMATH_CALUDE_sum_of_features_l4170_417054

/-- A pentagonal prism with a pyramid added to one pentagonal face -/
structure PentagonalPrismWithPyramid where
  /-- Number of faces of the original pentagonal prism -/
  prism_faces : ℕ
  /-- Number of vertices of the original pentagonal prism -/
  prism_vertices : ℕ
  /-- Number of edges of the original pentagonal prism -/
  prism_edges : ℕ
  /-- Number of faces added by the pyramid -/
  pyramid_faces : ℕ
  /-- Number of vertices added by the pyramid -/
  pyramid_vertices : ℕ
  /-- Number of edges added by the pyramid -/
  pyramid_edges : ℕ
  /-- The pentagonal prism has 7 faces -/
  prism_faces_eq : prism_faces = 7
  /-- The pentagonal prism has 10 vertices -/
  prism_vertices_eq : prism_vertices = 10
  /-- The pentagonal prism has 15 edges -/
  prism_edges_eq : prism_edges = 15
  /-- The pyramid adds 5 faces -/
  pyramid_faces_eq : pyramid_faces = 5
  /-- The pyramid adds 1 vertex -/
  pyramid_vertices_eq : pyramid_vertices = 1
  /-- The pyramid adds 5 edges -/
  pyramid_edges_eq : pyramid_edges = 5

/-- The sum of exterior faces, vertices, and edges of the resulting shape is 42 -/
theorem sum_of_features (shape : PentagonalPrismWithPyramid) :
  (shape.prism_faces + shape.pyramid_faces - 1) +
  (shape.prism_vertices + shape.pyramid_vertices) +
  (shape.prism_edges + shape.pyramid_edges) = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_features_l4170_417054


namespace NUMINAMATH_CALUDE_initial_kola_solution_volume_l4170_417084

/-- Represents the composition and volume of a kola solution -/
structure KolaSolution where
  initialVolume : ℝ
  waterPercentage : ℝ
  concentratedKolaPercentage : ℝ
  sugarPercentage : ℝ

/-- Theorem stating the initial volume of the kola solution -/
theorem initial_kola_solution_volume
  (solution : KolaSolution)
  (h1 : solution.waterPercentage = 0.88)
  (h2 : solution.concentratedKolaPercentage = 0.05)
  (h3 : solution.sugarPercentage = 1 - solution.waterPercentage - solution.concentratedKolaPercentage)
  (h4 : let newVolume := solution.initialVolume + 3.2 + 10 + 6.8
        (solution.sugarPercentage * solution.initialVolume + 3.2) / newVolume = 0.075) :
  solution.initialVolume = 340 := by
  sorry

end NUMINAMATH_CALUDE_initial_kola_solution_volume_l4170_417084


namespace NUMINAMATH_CALUDE_tourist_guide_groupings_l4170_417093

/-- The number of ways to distribute n distinguishable objects into 2 non-empty groups -/
def distributionCount (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tourists -/
def numTourists : ℕ := 6

/-- The number of guides -/
def numGuides : ℕ := 2

theorem tourist_guide_groupings :
  distributionCount numTourists = 62 :=
sorry

end NUMINAMATH_CALUDE_tourist_guide_groupings_l4170_417093


namespace NUMINAMATH_CALUDE_ellipse_equation_sum_l4170_417035

/-- Given an ellipse with foci at (2, 0) and (2, 4), passing through (7, 2),
    prove that a + k = 7 in its equation (x-h)^2/a^2 + (y-k)^2/b^2 = 1 where a and b are positive. -/
theorem ellipse_equation_sum (h k a b : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ 
    (x - 2)^2 + y^2 + (x - 2)^2 + (y - 4)^2 = 4 * ((x - 2)^2 + y^2)) →
  (7 - h)^2 / a^2 + (2 - k)^2 / b^2 = 1 →
  a > 0 →
  b > 0 →
  a + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_sum_l4170_417035


namespace NUMINAMATH_CALUDE_stating_batsman_average_increase_l4170_417053

/-- 
Represents a batsman's scoring record.
-/
structure BatsmanRecord where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- 
Calculates the increase in average given the batsman's record before and after an inning.
-/
def averageIncrease (before after : BatsmanRecord) : ℚ :=
  after.average - before.average

/-- 
Theorem stating that given a batsman's score of 85 runs in the 17th inning
and an average of 37 runs after the 17th inning, the increase in the batsman's average is 3 runs.
-/
theorem batsman_average_increase :
  ∀ (before : BatsmanRecord),
    before.inningsPlayed = 16 →
    (BatsmanRecord.mk 17 (before.totalRuns + 85) 37).average - before.average = 3 := by
  sorry

end NUMINAMATH_CALUDE_stating_batsman_average_increase_l4170_417053


namespace NUMINAMATH_CALUDE_base_prime_441_l4170_417041

/-- Definition of base prime representation for a natural number -/
def basePrimeRepresentation (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the base prime representation of 441 is [0, 2, 2, 0] -/
theorem base_prime_441 : basePrimeRepresentation 441 = [0, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_441_l4170_417041


namespace NUMINAMATH_CALUDE_tom_solo_time_is_four_l4170_417066

/-- The time it takes Avery to build the wall alone, in hours -/
def avery_time : ℝ := 2

/-- The time Avery and Tom work together, in hours -/
def together_time : ℝ := 1

/-- The time it takes Tom to finish the wall after Avery leaves, in hours -/
def tom_finish_time : ℝ := 1

/-- The time it takes Tom to build the wall alone, in hours -/
def tom_solo_time : ℝ := 4

/-- Theorem stating that Tom's solo time is 4 hours -/
theorem tom_solo_time_is_four :
  (1 / avery_time + 1 / tom_solo_time) * together_time + 
  (1 / tom_solo_time) * tom_finish_time = 1 →
  tom_solo_time = 4 := by
sorry

end NUMINAMATH_CALUDE_tom_solo_time_is_four_l4170_417066


namespace NUMINAMATH_CALUDE_marc_tv_watching_l4170_417051

/-- Given the number of episodes Marc watches per day and the total number of episodes,
    prove the relationship between x, y, and z. -/
theorem marc_tv_watching
  (friends_total : ℕ)
  (seinfeld_total : ℕ)
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)
  (h1 : friends_total = 50)
  (h2 : seinfeld_total = 75)
  (h3 : x * z = friends_total)
  (h4 : y * z = seinfeld_total) :
  y = (3 / 2) * x ∧ z = 50 / x :=
by sorry

end NUMINAMATH_CALUDE_marc_tv_watching_l4170_417051


namespace NUMINAMATH_CALUDE_best_approx_sqrt3_l4170_417061

def best_rational_approx (n : ℕ) (x : ℝ) : ℚ :=
  sorry

theorem best_approx_sqrt3 :
  best_rational_approx 15 (Real.sqrt 3) = 26 / 15 := by
  sorry

end NUMINAMATH_CALUDE_best_approx_sqrt3_l4170_417061


namespace NUMINAMATH_CALUDE_not_perfect_square_l4170_417006

theorem not_perfect_square (a b : ℕ+) (h : (a.val^2 - b.val^2) % 4 ≠ 0) :
  ¬ ∃ (k : ℕ), (a.val + 3*b.val) * (5*a.val + 7*b.val) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l4170_417006


namespace NUMINAMATH_CALUDE_tom_calorie_consumption_l4170_417082

/-- Calculates the total calories consumed by Tom given the specified conditions -/
theorem tom_calorie_consumption : 
  let carrot_calories : ℝ := 51
  let broccoli_calories : ℝ := (1/4) * carrot_calories
  let spinach_calories : ℝ := 0.6 * carrot_calories
  let cauliflower_calories : ℝ := 0.8 * broccoli_calories
  let carrot_pounds : ℝ := 3
  let broccoli_pounds : ℝ := 2
  let spinach_pounds : ℝ := 1
  let cauliflower_pounds : ℝ := 4
  (carrot_calories * carrot_pounds + 
   broccoli_calories * broccoli_pounds + 
   spinach_calories * spinach_pounds + 
   cauliflower_calories * cauliflower_pounds) = 249.9 := by
sorry

end NUMINAMATH_CALUDE_tom_calorie_consumption_l4170_417082


namespace NUMINAMATH_CALUDE_odd_multiple_of_three_l4170_417026

theorem odd_multiple_of_three (a : ℕ) : 
  Odd (88 * a) → (88 * a) % 3 = 0 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_odd_multiple_of_three_l4170_417026
