import Mathlib

namespace NUMINAMATH_CALUDE_apples_distribution_l445_44526

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 5

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 2 * benny_apples

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of apples each friend received -/
def apples_per_friend : ℕ := total_apples / num_friends

theorem apples_distribution :
  apples_per_friend = 5 :=
sorry

end NUMINAMATH_CALUDE_apples_distribution_l445_44526


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l445_44550

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 → 0 < x₂ → y₁ = 6 / x₁ → y₂ = 6 / x₂ → y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l445_44550


namespace NUMINAMATH_CALUDE_emilys_number_proof_l445_44534

theorem emilys_number_proof :
  ∃! n : ℕ, 
    (216 ∣ n) ∧ 
    (45 ∣ n) ∧ 
    (1000 < n) ∧ 
    (n < 3000) ∧ 
    (n = 2160) := by
  sorry

end NUMINAMATH_CALUDE_emilys_number_proof_l445_44534


namespace NUMINAMATH_CALUDE_forty_percent_bought_something_l445_44566

/-- Given advertising costs, number of customers, item price, and profit,
    calculates the percentage of customers who made a purchase. -/
def percentage_of_customers_who_bought (advertising_cost : ℕ) (num_customers : ℕ) 
  (item_price : ℕ) (profit : ℕ) : ℚ :=
  (profit / item_price : ℚ) / num_customers * 100

/-- Theorem stating that under the given conditions, 
    40% of customers made a purchase. -/
theorem forty_percent_bought_something :
  percentage_of_customers_who_bought 1000 100 25 1000 = 40 := by
  sorry

#eval percentage_of_customers_who_bought 1000 100 25 1000

end NUMINAMATH_CALUDE_forty_percent_bought_something_l445_44566


namespace NUMINAMATH_CALUDE_second_grade_volunteers_l445_44528

/-- Given a total population and a subgroup, calculate the proportion of volunteers
    to be selected from the subgroup in a stratified random sampling. -/
def stratified_sampling_proportion (total_population : ℕ) (subgroup : ℕ) (total_volunteers : ℕ) : ℕ :=
  (subgroup * total_volunteers) / total_population

/-- Prove that in a stratified random sampling of 30 volunteers from a population of 3000 students,
    where 1000 students are in the second grade, the number of volunteers to be selected from
    the second grade is 10. -/
theorem second_grade_volunteers :
  stratified_sampling_proportion 3000 1000 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_volunteers_l445_44528


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l445_44565

theorem square_perimeter_sum (a b : ℝ) (h1 : a + b = 85) (h2 : a - b = 41) :
  4 * (Real.sqrt a.toNNReal + Real.sqrt b.toNNReal) = 4 * (Real.sqrt 63 + Real.sqrt 22) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l445_44565


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l445_44559

/-- Given a parallelogram with area 98 sq m and altitude twice the base, prove the base is 7 m -/
theorem parallelogram_base_length : 
  ∀ (base altitude : ℝ), 
  (base * altitude = 98) →  -- Area of parallelogram
  (altitude = 2 * base) →   -- Altitude is twice the base
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l445_44559


namespace NUMINAMATH_CALUDE_seating_arrangements_l445_44513

def total_people : ℕ := 10
def restricted_group : ℕ := 4

theorem seating_arrangements (total : ℕ) (restricted : ℕ) :
  total = total_people ∧ restricted = restricted_group →
  (total.factorial - (total - restricted + 1).factorial * restricted.factorial) = 3507840 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l445_44513


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l445_44592

-- Define the propositions p and q
def p (a b : ℝ) : Prop := a + b > 0
def q (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that p is neither sufficient nor necessary for q
theorem p_neither_sufficient_nor_necessary_for_q :
  (∃ a b : ℝ, p a b ∧ ¬q a b) ∧ (∃ a b : ℝ, q a b ∧ ¬p a b) :=
sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l445_44592


namespace NUMINAMATH_CALUDE_probability_at_least_two_green_l445_44578

theorem probability_at_least_two_green (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) :
  total = 10 ∧ red = 5 ∧ green = 3 ∧ yellow = 2 →
  (Nat.choose total 3 : ℚ) ≠ 0 →
  (Nat.choose green 2 * Nat.choose (total - green) 1 + Nat.choose green 3 : ℚ) / Nat.choose total 3 = 11 / 60 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_green_l445_44578


namespace NUMINAMATH_CALUDE_tommy_savings_needed_l445_44595

def number_of_books : ℕ := 8
def cost_per_book : ℕ := 5
def current_savings : ℕ := 13

theorem tommy_savings_needed : 
  number_of_books * cost_per_book - current_savings = 27 := by
  sorry

end NUMINAMATH_CALUDE_tommy_savings_needed_l445_44595


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l445_44589

theorem min_value_sum_reciprocals (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    1 / (x + y) + 1 / z ≥ 1 / (a + b) + 1 / c) → 
  1 / (a + b) + 1 / c = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l445_44589


namespace NUMINAMATH_CALUDE_sheep_wandered_off_percentage_l445_44571

theorem sheep_wandered_off_percentage 
  (total_sheep : ℕ) 
  (rounded_up_percentage : ℚ) 
  (sheep_in_pen : ℕ) 
  (sheep_in_wilderness : ℕ) 
  (h1 : rounded_up_percentage = 90 / 100) 
  (h2 : sheep_in_pen = 81) 
  (h3 : sheep_in_wilderness = 9) 
  (h4 : ↑sheep_in_pen = rounded_up_percentage * ↑total_sheep) 
  (h5 : total_sheep = sheep_in_pen + sheep_in_wilderness) : 
  (↑sheep_in_wilderness / ↑total_sheep) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_sheep_wandered_off_percentage_l445_44571


namespace NUMINAMATH_CALUDE_min_value_of_f_l445_44536

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem min_value_of_f (a : ℝ) : (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) ↔ a = 1 ∨ a = 7 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l445_44536


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l445_44553

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l445_44553


namespace NUMINAMATH_CALUDE_maggies_total_earnings_l445_44514

/-- Maggie's earnings from selling magazine subscriptions --/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (next_door_neighbor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    next_door_neighbor_subscriptions + 
    (2 * next_door_neighbor_subscriptions)
  total_subscriptions * price_per_subscription

/-- Theorem stating Maggie's earnings --/
theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggies_total_earnings_l445_44514


namespace NUMINAMATH_CALUDE_car_rental_rate_proof_l445_44576

/-- The daily rate of the first car rental company -/
def first_company_rate : ℝ := 17.99

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.18

/-- The daily rate of City Rentals -/
def city_rentals_rate : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.16

/-- The number of miles at which the cost is the same for both companies -/
def equal_cost_miles : ℝ := 48

theorem car_rental_rate_proof :
  first_company_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_rate + city_rentals_per_mile * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rate_proof_l445_44576


namespace NUMINAMATH_CALUDE_table_length_is_77_l445_44501

/-- The length of a rectangular table covered by overlapping paper sheets. -/
def table_length : ℕ :=
  let table_width : ℕ := 80
  let sheet_width : ℕ := 8
  let sheet_height : ℕ := 5
  let offset : ℕ := 1
  let sheets_needed : ℕ := table_width - sheet_width
  sheet_height + sheets_needed

theorem table_length_is_77 : table_length = 77 := by
  sorry

end NUMINAMATH_CALUDE_table_length_is_77_l445_44501


namespace NUMINAMATH_CALUDE_monotonic_function_range_l445_44590

def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonic_function_range (a : ℝ) :
  monotonic_on (f a) (-1) 2 → a ≤ -1 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_range_l445_44590


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l445_44530

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (3 * (a 3)^2 - 11 * (a 3) + 9 = 0) →
  (3 * (a 9)^2 - 11 * (a 9) + 9 = 0) →
  (a 6 = Real.sqrt 3 ∨ a 6 = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l445_44530


namespace NUMINAMATH_CALUDE_remainder_97_37_mod_100_l445_44568

theorem remainder_97_37_mod_100 : 97^37 % 100 = 77 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_37_mod_100_l445_44568


namespace NUMINAMATH_CALUDE_sequence_ratio_l445_44561

def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) :
  is_arithmetic_sequence 1 a₁ a₂ 9 →
  is_geometric_sequence 1 b₁ b₂ b₃ 9 →
  b₂ / (a₁ + a₂) = 3 / 10 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l445_44561


namespace NUMINAMATH_CALUDE_age_ratio_change_l445_44519

/-- Proves the number of years it takes for a parent to become 2.5 times as old as their son -/
theorem age_ratio_change (parent_age son_age : ℕ) (x : ℕ) 
  (h1 : parent_age = 45)
  (h2 : son_age = 15)
  (h3 : parent_age = 3 * son_age) :
  (parent_age + x) = (5/2 : ℚ) * (son_age + x) ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_change_l445_44519


namespace NUMINAMATH_CALUDE_second_street_sales_l445_44591

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  streets : Fin 4 → ℕ
  second_street_sales : ℕ

/-- The conditions of the sales problem. -/
def sales_conditions (data : SalesData) : Prop :=
  data.commission_per_sale = 25 ∧
  data.total_commission = 175 ∧
  data.streets 0 = data.second_street_sales / 2 ∧
  data.streets 1 = data.second_street_sales ∧
  data.streets 2 = 0 ∧
  data.streets 3 = 1

/-- Theorem stating that under the given conditions, the number of security systems sold on the second street is 4. -/
theorem second_street_sales (data : SalesData) :
  sales_conditions data → data.second_street_sales = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_street_sales_l445_44591


namespace NUMINAMATH_CALUDE_M_equals_N_l445_44599

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l445_44599


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l445_44549

theorem real_part_of_i_squared_times_one_plus_i : 
  Complex.re (Complex.I^2 * (1 + Complex.I)) = -1 := by sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_plus_i_l445_44549


namespace NUMINAMATH_CALUDE_seating_theorem_l445_44582

/-- The number of seats in the row -/
def n : ℕ := 8

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for k people in n seats,
    with empty seats required on both sides of each person -/
def seating_arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating that the number of seating arrangements
    for 2 people in 8 seats is 20 -/
theorem seating_theorem : seating_arrangements n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l445_44582


namespace NUMINAMATH_CALUDE_union_equals_interval_l445_44518

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Define the interval [-1, 4]
def interval : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l445_44518


namespace NUMINAMATH_CALUDE_coin_toss_problem_l445_44574

theorem coin_toss_problem (n : ℕ) 
  (total_outcomes : ℕ) 
  (equally_likely : total_outcomes = 8)
  (die_roll_prob : ℚ) 
  (die_roll_prob_value : die_roll_prob = 1/3) :
  (2^n = total_outcomes) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l445_44574


namespace NUMINAMATH_CALUDE_ellipse_constant_product_l445_44525

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line passing through (1, 0) with slope k -/
def line (x y k : ℝ) : Prop := y = k * (x - 1)

/-- The dot product of vectors PE and QE -/
def dot_product (xP yP xQ yQ xE : ℝ) : ℝ :=
  (xE - xP) * (xE - xQ) + (-yP) * (-yQ)

theorem ellipse_constant_product :
  ∀ (xP yP xQ yQ k : ℝ),
    ellipse xP yP →
    ellipse xQ yQ →
    line xP yP k →
    line xQ yQ k →
    xP ≠ xQ →
    dot_product xP yP xQ yQ (17/8) = 33/64 := by sorry

end NUMINAMATH_CALUDE_ellipse_constant_product_l445_44525


namespace NUMINAMATH_CALUDE_mixture_problem_l445_44516

/-- Proves that the initial amount of liquid A is 16 liters given the conditions of the mixture problem -/
theorem mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  (4*x) / x = 4 / 1 ∧ 
  (4*x - 8) / (x + 8) = 2 / 3 → 
  4*x = 16 := by
sorry

end NUMINAMATH_CALUDE_mixture_problem_l445_44516


namespace NUMINAMATH_CALUDE_labor_cost_calculation_l445_44517

def cost_of_seeds : ℝ := 50
def cost_of_fertilizers_and_pesticides : ℝ := 35
def number_of_bags : ℕ := 10
def price_per_bag : ℝ := 11
def profit_percentage : ℝ := 0.1

theorem labor_cost_calculation (labor_cost : ℝ) : 
  (cost_of_seeds + cost_of_fertilizers_and_pesticides + labor_cost) * (1 + profit_percentage) = 
  (number_of_bags : ℝ) * price_per_bag → 
  labor_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_labor_cost_calculation_l445_44517


namespace NUMINAMATH_CALUDE_number_equation_solution_l445_44588

theorem number_equation_solution : 
  ∃! x : ℝ, 45 - (28 - (37 - (x - 18))) = 57 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l445_44588


namespace NUMINAMATH_CALUDE_set_of_positive_rationals_l445_44547

theorem set_of_positive_rationals (S : Set ℚ) :
  (∀ a b : ℚ, a ∈ S → b ∈ S → (a + b) ∈ S ∧ (a * b) ∈ S) →
  (∀ r : ℚ, (r ∈ S ∧ -r ∉ S ∧ r ≠ 0) ∨ (-r ∈ S ∧ r ∉ S ∧ r ≠ 0) ∨ (r = 0 ∧ r ∉ S ∧ -r ∉ S)) →
  S = {r : ℚ | r > 0} :=
by sorry

end NUMINAMATH_CALUDE_set_of_positive_rationals_l445_44547


namespace NUMINAMATH_CALUDE_max_d_value_l445_44511

/-- The sequence term for a given n -/
def a (n : ℕ) : ℕ := 100 + n^2

/-- The greatest common divisor of consecutive terms -/
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The theorem stating the maximum value of d_n -/
theorem max_d_value : ∃ (N : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ 401 ∧ d N = 401 := by
  sorry

end NUMINAMATH_CALUDE_max_d_value_l445_44511


namespace NUMINAMATH_CALUDE_betty_bracelets_l445_44597

/-- Given that Betty has 88.0 pink flower stones and each bracelet requires 11 stones,
    prove that the number of bracelets she can make is 8. -/
theorem betty_bracelets :
  let total_stones : ℝ := 88.0
  let stones_per_bracelet : ℕ := 11
  (total_stones / stones_per_bracelet : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_betty_bracelets_l445_44597


namespace NUMINAMATH_CALUDE_stating_seedling_cost_equations_l445_44509

/-- Represents the cost of seedlings and their price difference -/
structure SeedlingCost where
  x : ℝ  -- Cost of one pine seedling in yuan
  y : ℝ  -- Cost of one tamarisk seedling in yuan
  total_cost : 4 * x + 3 * y = 180  -- Total cost equation
  price_difference : x - y = 10  -- Price difference equation

/-- 
Theorem stating that the given system of equations correctly represents 
the cost of pine and tamarisk seedlings under the given conditions
-/
theorem seedling_cost_equations (cost : SeedlingCost) : 
  (4 * cost.x + 3 * cost.y = 180) ∧ (cost.x - cost.y = 10) := by
  sorry

end NUMINAMATH_CALUDE_stating_seedling_cost_equations_l445_44509


namespace NUMINAMATH_CALUDE_arithmetic_operations_l445_44539

theorem arithmetic_operations :
  ((-12) + (-6) - (-28) = 10) ∧
  ((-8/5) * (15/4) / (-9) = 2/3) ∧
  ((-3/16 - 7/24 + 5/6) * (-48) = -17) ∧
  (-(3^2) + (7/8 - 1) * ((-2)^2) = -19/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l445_44539


namespace NUMINAMATH_CALUDE_total_marbles_l445_44524

def marble_collection (jar1 jar2 jar3 : ℕ) : Prop :=
  (jar1 = 80) ∧
  (jar2 = 2 * jar1) ∧
  (jar3 = jar1 / 4) ∧
  (jar1 + jar2 + jar3 = 260)

theorem total_marbles :
  ∃ (jar1 jar2 jar3 : ℕ), marble_collection jar1 jar2 jar3 :=
sorry

end NUMINAMATH_CALUDE_total_marbles_l445_44524


namespace NUMINAMATH_CALUDE_coin_problem_l445_44551

/-- Proves that Tom has 8 quarters given the conditions of the coin problem -/
theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (quarter_value nickel_value : ℚ) : 
  total_coins = 12 →
  total_value = 11/5 →
  quarter_value = 1/4 →
  nickel_value = 1/20 →
  ∃ (quarters nickels : ℕ),
    quarters + nickels = total_coins ∧
    quarter_value * quarters + nickel_value * nickels = total_value ∧
    quarters = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l445_44551


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l445_44542

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 45 → b = 60 → c^2 = a^2 + b^2 → c = 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l445_44542


namespace NUMINAMATH_CALUDE_peggy_dolls_l445_44512

/-- The number of dolls Peggy has at the end -/
def final_dolls (initial : ℕ) (grandmother : ℕ) : ℕ :=
  initial + grandmother + grandmother / 2

/-- Theorem stating that Peggy ends up with 51 dolls -/
theorem peggy_dolls : final_dolls 6 30 = 51 := by
  sorry

end NUMINAMATH_CALUDE_peggy_dolls_l445_44512


namespace NUMINAMATH_CALUDE_total_decorations_count_l445_44557

/-- The number of decorations in each box -/
def decorations_per_box : ℕ := 4 + 1 + 5

/-- The number of families receiving a box -/
def number_of_families : ℕ := 11

/-- The number of boxes given to the community center -/
def community_center_boxes : ℕ := 1

/-- The total number of decorations handed out -/
def total_decorations : ℕ := decorations_per_box * (number_of_families + community_center_boxes)

theorem total_decorations_count : total_decorations = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_decorations_count_l445_44557


namespace NUMINAMATH_CALUDE_kola_solution_water_percentage_l445_44533

theorem kola_solution_water_percentage
  (initial_volume : ℝ)
  (initial_kola_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_kola_percentage = 5)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 10)
  (h5 : added_kola = 6.8)
  (h6 : final_sugar_percentage = 7.5)
  : ∃ (initial_water_percentage : ℝ),
    initial_water_percentage = 88 ∧
    initial_water_percentage + 
      (100 - initial_water_percentage - initial_kola_percentage) + 
      initial_kola_percentage = 100 ∧
    (100 - initial_water_percentage - initial_kola_percentage) / 100 * initial_volume + added_sugar = 
      final_sugar_percentage / 100 * (initial_volume + added_sugar + added_water + added_kola) :=
by sorry

end NUMINAMATH_CALUDE_kola_solution_water_percentage_l445_44533


namespace NUMINAMATH_CALUDE_circle_tangents_theorem_no_single_common_tangent_l445_44527

/-- Represents the number of common tangents between two circles -/
inductive CommonTangents
  | zero
  | two
  | three
  | four

/-- Represents the configuration of two circles -/
structure CircleConfiguration where
  r1 : ℝ  -- radius of the first circle
  r2 : ℝ  -- radius of the second circle
  d : ℝ   -- distance between the centers of the circles

/-- Function to determine the number of common tangents based on circle configuration -/
def numberOfCommonTangents (config : CircleConfiguration) : CommonTangents :=
  sorry

/-- Theorem stating that two circles with radii 10 and 4 can have 0, 2, 3, or 4 common tangents -/
theorem circle_tangents_theorem :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  (numberOfCommonTangents config = CommonTangents.zero) ∨
  (numberOfCommonTangents config = CommonTangents.two) ∨
  (numberOfCommonTangents config = CommonTangents.three) ∨
  (numberOfCommonTangents config = CommonTangents.four) :=
by sorry

/-- Theorem stating that two circles with radii 10 and 4 cannot have exactly 1 common tangent -/
theorem no_single_common_tangent :
  ∀ (d : ℝ),
  let config := CircleConfiguration.mk 10 4 d
  numberOfCommonTangents config ≠ CommonTangents.zero :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_theorem_no_single_common_tangent_l445_44527


namespace NUMINAMATH_CALUDE_parabola_and_intersection_l445_44587

/-- Parabola with vertex at origin and focus on x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on the parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.equation x y

/-- Line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  k : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = k * x + b
  intersects_twice : ∃ (p1 p2 : PointOnParabola par), p1 ≠ p2 ∧ 
    equation p1.x p1.y ∧ equation p2.x p2.y

theorem parabola_and_intersection 
    (par : Parabola) 
    (A : PointOnParabola par)
    (h1 : A.x = 4)
    (h2 : (A.x + par.p / 2)^2 + A.y^2 = 6^2)
    (line : IntersectingLine par)
    (h3 : line.b = -2)
    (h4 : ∃ (B : PointOnParabola par), 
      line.equation B.x B.y ∧ (A.x + B.x) / 2 = 2) :
  par.p = 4 ∧ line.k = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_l445_44587


namespace NUMINAMATH_CALUDE_factorization_equality_l445_44581

theorem factorization_equality (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = (x^4 + x^2 + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l445_44581


namespace NUMINAMATH_CALUDE_factor_expression_l445_44563

theorem factor_expression (y : ℝ) : 75 * y + 45 = 15 * (5 * y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l445_44563


namespace NUMINAMATH_CALUDE_two_statements_incorrect_l445_44508

-- Define a type for geometric statements
inductive GeometricStatement
  | ParallelogramOppositeAngles
  | PolygonExteriorAngles
  | TriangleRotation
  | AngleMagnification
  | CircleCircumferenceRadiusRatio
  | CircleCircumferenceAreaRatio

-- Define a function to check if a statement is correct
def isCorrect (s : GeometricStatement) : Bool :=
  match s with
  | .ParallelogramOppositeAngles => false
  | .PolygonExteriorAngles => true
  | .TriangleRotation => true
  | .AngleMagnification => true
  | .CircleCircumferenceRadiusRatio => true
  | .CircleCircumferenceAreaRatio => false

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [.ParallelogramOppositeAngles, .PolygonExteriorAngles, .TriangleRotation,
   .AngleMagnification, .CircleCircumferenceRadiusRatio, .CircleCircumferenceAreaRatio]

-- Theorem: Exactly 2 out of 6 statements are incorrect
theorem two_statements_incorrect :
  (allStatements.filter (fun s => ¬(isCorrect s))).length = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_statements_incorrect_l445_44508


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_M_l445_44548

def M : ℕ := sorry

def is_highest_power_of_three (n : ℕ) (j : ℕ) : Prop :=
  3^j ∣ n ∧ ∀ k > j, ¬(3^k ∣ n)

theorem highest_power_of_three_in_M :
  is_highest_power_of_three M 0 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_M_l445_44548


namespace NUMINAMATH_CALUDE_sin_equality_condition_l445_44556

theorem sin_equality_condition :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_condition_l445_44556


namespace NUMINAMATH_CALUDE_percentage_difference_l445_44598

theorem percentage_difference : (0.6 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l445_44598


namespace NUMINAMATH_CALUDE_salary_comparison_l445_44562

def hansel_initial : ℕ := 30000
def hansel_raise : ℚ := 10 / 100

def gretel_initial : ℕ := 30000
def gretel_raise : ℚ := 15 / 100

def rapunzel_initial : ℕ := 40000
def rapunzel_raise : ℚ := 8 / 100

def rumpelstiltskin_initial : ℕ := 35000
def rumpelstiltskin_raise : ℚ := 12 / 100

def new_salary (initial : ℕ) (raise : ℚ) : ℚ :=
  initial * (1 + raise)

theorem salary_comparison :
  (new_salary gretel_initial gretel_raise - new_salary hansel_initial hansel_raise = 1500) ∧
  (new_salary gretel_initial gretel_raise < new_salary rapunzel_initial rapunzel_raise) ∧
  (new_salary gretel_initial gretel_raise < new_salary rumpelstiltskin_initial rumpelstiltskin_raise) :=
by sorry

end NUMINAMATH_CALUDE_salary_comparison_l445_44562


namespace NUMINAMATH_CALUDE_cinema_meeting_day_l445_44506

theorem cinema_meeting_day : Nat.lcm (Nat.lcm 4 5) 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cinema_meeting_day_l445_44506


namespace NUMINAMATH_CALUDE_pony_discount_rate_l445_44540

/-- Represents the discount rate for Fox jeans -/
def F : ℝ := sorry

/-- Represents the discount rate for Pony jeans -/
def P : ℝ := sorry

/-- Regular price of Fox jeans -/
def fox_price : ℝ := 15

/-- Regular price of Pony jeans -/
def pony_price : ℝ := 18

/-- Total savings from purchasing 3 pairs of Fox jeans and 2 pairs of Pony jeans -/
def total_savings : ℝ := 8.64

/-- The sum of discount rates for Fox and Pony jeans -/
def total_discount : ℝ := 22

theorem pony_discount_rate : 
  F + P = total_discount ∧ 
  3 * (fox_price * F / 100) + 2 * (pony_price * P / 100) = total_savings →
  P = 14 := by sorry

end NUMINAMATH_CALUDE_pony_discount_rate_l445_44540


namespace NUMINAMATH_CALUDE_angle_ABC_bisector_l445_44535

theorem angle_ABC_bisector (ABC : Real) : 
  (ABC / 2 = (180 - ABC) / 6) → ABC = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_bisector_l445_44535


namespace NUMINAMATH_CALUDE_xy_value_l445_44594

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : Real.sqrt (Real.log x) + Real.sqrt (Real.log y) + 
          Real.log (Real.sqrt x) + Real.log (Real.sqrt y) + 
          Real.log (x^(1/4)) + Real.log (y^(1/4)) = 150)
  (h_int1 : ∃ n : ℤ, Real.sqrt (Real.log x) = n)
  (h_int2 : ∃ n : ℤ, Real.sqrt (Real.log y) = n)
  (h_int3 : ∃ n : ℤ, Real.log (Real.sqrt x) = n)
  (h_int4 : ∃ n : ℤ, Real.log (Real.sqrt y) = n)
  (h_int5 : ∃ n : ℤ, Real.log (x^(1/4)) = n)
  (h_int6 : ∃ n : ℤ, Real.log (y^(1/4)) = n) :
  x * y = Real.exp 340 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l445_44594


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l445_44575

theorem modular_congruence_solution :
  ∀ m n : ℕ,
  0 ≤ m ∧ m ≤ 17 →
  0 ≤ n ∧ n ≤ 13 →
  m ≡ 98765 [MOD 18] →
  n ≡ 98765 [MOD 14] →
  m = 17 ∧ n = 9 := by
sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l445_44575


namespace NUMINAMATH_CALUDE_joes_team_draws_l445_44593

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  3 * team.wins + team.draws

theorem joes_team_draws : 
  ∀ (joes_team first_place : TeamPerformance),
    joes_team.wins = 1 →
    first_place.wins = 2 →
    first_place.draws = 2 →
    calculatePoints first_place = calculatePoints joes_team + 2 →
    joes_team.draws = 3 := by
  sorry


end NUMINAMATH_CALUDE_joes_team_draws_l445_44593


namespace NUMINAMATH_CALUDE_max_value_implies_a_l445_44579

def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 3/8 :=
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l445_44579


namespace NUMINAMATH_CALUDE_hotel_charges_l445_44531

theorem hotel_charges (G : ℝ) (h1 : G > 0) : 
  let R := 2 * G
  let P := R * (1 - 0.55)
  P = G * (1 - 0.1) := by
sorry

end NUMINAMATH_CALUDE_hotel_charges_l445_44531


namespace NUMINAMATH_CALUDE_rational_sqrt_n_minus_3_over_n_plus_1_l445_44529

theorem rational_sqrt_n_minus_3_over_n_plus_1 
  (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (a b : ℚ), b ≠ 0 ∧ (n - 3) / (n + 1) = (a / b) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_rational_sqrt_n_minus_3_over_n_plus_1_l445_44529


namespace NUMINAMATH_CALUDE_largest_angle_obtuse_triangle_l445_44577

/-- Given an obtuse, scalene triangle ABC with angle A measuring 30 degrees and angle B measuring 55 degrees,
    the measure of the largest interior angle is 95 degrees. -/
theorem largest_angle_obtuse_triangle (A B C : ℝ) (h_obtuse : A + B + C = 180) 
  (h_A : A = 30) (h_B : B = 55) (h_scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  max A (max B C) = 95 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_obtuse_triangle_l445_44577


namespace NUMINAMATH_CALUDE_sci_fi_readers_l445_44564

theorem sci_fi_readers (total : ℕ) (literary : ℕ) (both : ℕ) : 
  total = 650 → literary = 550 → both = 150 → 
  total = literary + (total - literary + both) - both :=
by
  sorry

#check sci_fi_readers

end NUMINAMATH_CALUDE_sci_fi_readers_l445_44564


namespace NUMINAMATH_CALUDE_four_digit_divisible_count_l445_44569

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_all (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem four_digit_divisible_count :
  ∃! (s : Finset ℕ), s.card = 4 ∧
  (∀ n : ℕ, n ∈ s ↔ (is_four_digit n ∧ divisible_by_all n)) :=
sorry

end NUMINAMATH_CALUDE_four_digit_divisible_count_l445_44569


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l445_44500

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 - x)^2) ≥ y ∧
    (∃ (z : ℝ), Real.sqrt (z^2 + (1 - z)^2) + Real.sqrt ((1 - z)^2 + (1 - z)^2) = y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l445_44500


namespace NUMINAMATH_CALUDE_problem_solution_l445_44580

theorem problem_solution : (69842 * 69842 - 30158 * 30158) / (69842 - 30158) = 100000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l445_44580


namespace NUMINAMATH_CALUDE_inequality_proof_l445_44585

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a^2 + 3 * b^2) + (b * c) / (b^2 + 3 * c^2) + (c * a) / (c^2 + 3 * a^2) ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l445_44585


namespace NUMINAMATH_CALUDE_three_intersecting_lines_l445_44546

/-- The parabola defined by y² = 3x -/
def parabola (x y : ℝ) : Prop := y^2 = 3*x

/-- A point lies on a line through (0, 2) -/
def line_through_A (m : ℝ) (x y : ℝ) : Prop := y = m*x + 2

/-- A line intersects the parabola at exactly one point -/
def single_intersection (m : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola p.1 p.2 ∧ line_through_A m p.1 p.2

/-- There are exactly 3 lines through (0, 2) that intersect the parabola at one point -/
theorem three_intersecting_lines : ∃! l : Finset ℝ, 
  l.card = 3 ∧ (∀ m ∈ l, single_intersection m) ∧
  (∀ m : ℝ, single_intersection m → m ∈ l) :=
sorry

end NUMINAMATH_CALUDE_three_intersecting_lines_l445_44546


namespace NUMINAMATH_CALUDE_mall_parking_lot_cars_l445_44504

/-- The number of cars parked in a mall's parking lot -/
def number_of_cars : ℕ := 10

/-- The number of customers in each car -/
def customers_per_car : ℕ := 5

/-- The number of sales made by the sports store -/
def sports_store_sales : ℕ := 20

/-- The number of sales made by the music store -/
def music_store_sales : ℕ := 30

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem mall_parking_lot_cars :
  number_of_cars * customers_per_car = sports_store_sales + music_store_sales :=
by sorry

end NUMINAMATH_CALUDE_mall_parking_lot_cars_l445_44504


namespace NUMINAMATH_CALUDE_ants_after_five_hours_l445_44520

/-- The number of ants in the jar after a given number of hours -/
def antsInJar (initialAnts : ℕ) (hours : ℕ) : ℕ :=
  initialAnts * (2 ^ hours)

/-- Theorem stating that 50 ants doubling every hour for 5 hours results in 1600 ants -/
theorem ants_after_five_hours :
  antsInJar 50 5 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ants_after_five_hours_l445_44520


namespace NUMINAMATH_CALUDE_trig_identity_proof_l445_44503

theorem trig_identity_proof (α : ℝ) : 
  Real.cos (α - 35 * π / 180) * Real.cos (25 * π / 180 + α) + 
  Real.sin (α - 35 * π / 180) * Real.sin (25 * π / 180 + α) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l445_44503


namespace NUMINAMATH_CALUDE_cobbler_friday_hours_l445_44545

/-- Represents the cobbler's work week -/
structure CobblerWeek where
  shoes_per_hour : ℕ
  hours_per_day : ℕ
  days_before_friday : ℕ
  total_shoes_per_week : ℕ

/-- Calculates the number of hours worked on Friday -/
def friday_hours (week : CobblerWeek) : ℕ :=
  (week.total_shoes_per_week - week.shoes_per_hour * week.hours_per_day * week.days_before_friday) / week.shoes_per_hour

/-- Theorem stating that the cobbler works 3 hours on Friday -/
theorem cobbler_friday_hours :
  let week : CobblerWeek := {
    shoes_per_hour := 3,
    hours_per_day := 8,
    days_before_friday := 4,
    total_shoes_per_week := 105
  }
  friday_hours week = 3 := by
  sorry

end NUMINAMATH_CALUDE_cobbler_friday_hours_l445_44545


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l445_44584

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 2*X^2 + 1 = (X^2 - 2*X + 4) * q + (-4*X - 7) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l445_44584


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l445_44586

theorem sum_of_squares_problem (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 64 → 
  a*b + b*c + c*a = 30 → 
  a + b + c = 2 * Real.sqrt 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l445_44586


namespace NUMINAMATH_CALUDE_tan_11_25_decomposition_l445_44532

theorem tan_11_25_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (11.25 * Real.pi / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_11_25_decomposition_l445_44532


namespace NUMINAMATH_CALUDE_sequence_length_l445_44596

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The number of terms in the arithmetic sequence from 5 to 200 with common difference 3 -/
theorem sequence_length : arithmeticSequenceLength 5 200 3 = 66 := by
  sorry

#eval arithmeticSequenceLength 5 200 3

end NUMINAMATH_CALUDE_sequence_length_l445_44596


namespace NUMINAMATH_CALUDE_cantors_theorem_l445_44523

theorem cantors_theorem (X : Type u) : ¬∃(f : X → Set X), Function.Bijective f :=
  sorry

end NUMINAMATH_CALUDE_cantors_theorem_l445_44523


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l445_44552

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = -1 + 3*I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l445_44552


namespace NUMINAMATH_CALUDE_michael_money_ratio_l445_44507

/-- Given the initial conditions and final state of a money transfer between Michael and his brother,
    prove that the ratio of the money Michael gave to his brother to his initial amount is 1/2. -/
theorem michael_money_ratio :
  ∀ (michael_initial brother_initial michael_final brother_final transfer candy : ℕ),
    michael_initial = 42 →
    brother_initial = 17 →
    brother_final = 35 →
    candy = 3 →
    michael_final + transfer = michael_initial →
    brother_final + candy = brother_initial + transfer →
    (transfer : ℚ) / michael_initial = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_money_ratio_l445_44507


namespace NUMINAMATH_CALUDE_cube_coloring_probability_l445_44570

/-- The probability of a single color being chosen for a face -/
def color_probability : ℚ := 1/3

/-- The number of pairs of opposite faces in a cube -/
def opposite_face_pairs : ℕ := 3

/-- The probability that a pair of opposite faces has different colors -/
def diff_color_prob : ℚ := 2/3

/-- The probability that all pairs of opposite faces have different colors -/
def all_diff_prob : ℚ := diff_color_prob ^ opposite_face_pairs

theorem cube_coloring_probability :
  1 - all_diff_prob = 19/27 := by sorry

end NUMINAMATH_CALUDE_cube_coloring_probability_l445_44570


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l445_44521

theorem difference_of_squares_factorization (y : ℝ) : 
  100 - 16 * y^2 = 4 * (5 - 2*y) * (5 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l445_44521


namespace NUMINAMATH_CALUDE_spring_compression_l445_44515

/-- The force-distance relationship for a spring -/
def spring_force (s : ℝ) : ℝ := 16 * s^2

/-- Theorem: When a force of 4 newtons is applied, the spring compresses by 0.5 meters -/
theorem spring_compression :
  spring_force 0.5 = 4 := by sorry

end NUMINAMATH_CALUDE_spring_compression_l445_44515


namespace NUMINAMATH_CALUDE_min_RS_value_l445_44538

/-- Represents a rhombus ABCD with given diagonals -/
structure Rhombus where
  AC : ℝ
  BD : ℝ

/-- Represents a point M on side AB of the rhombus -/
structure PointM where
  BM : ℝ

/-- The minimum value of RS given the rhombus and point M -/
noncomputable def min_RS (r : Rhombus) (m : PointM) : ℝ :=
  Real.sqrt (8 * m.BM^2 - 40 * m.BM + 400)

/-- Theorem stating the minimum value of RS -/
theorem min_RS_value (r : Rhombus) : 
  r.AC = 24 → r.BD = 40 → ∃ (m : PointM), min_RS r m = 5 * Real.sqrt 14 := by
  sorry

#check min_RS_value

end NUMINAMATH_CALUDE_min_RS_value_l445_44538


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l445_44560

/-- A geometric sequence with first term 5 and third term 20 has second term 10 -/
theorem geometric_sequence_second_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 5 →
    a * r^2 = 20 →
    a * r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l445_44560


namespace NUMINAMATH_CALUDE_inequality_properties_l445_44522

theorem inequality_properties (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : d > 0) : 
  (a - d > b - c) ∧ 
  (a * c^2 > b * c^2) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a * c ≤ b * d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a / d ≤ b / c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l445_44522


namespace NUMINAMATH_CALUDE_correct_systematic_sampling_l445_44502

/-- Represents a systematic sampling scheme. -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Generates a sample based on the systematic sampling scheme. -/
def generate_sample (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (λ i => s.start + i * s.interval)

/-- The theorem to be proved. -/
theorem correct_systematic_sampling :
  let s : SystematicSampling := {
    population_size := 60,
    sample_size := 6,
    start := 3,
    interval := 10
  }
  generate_sample s = [3, 13, 23, 33, 43, 53] :=
by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sampling_l445_44502


namespace NUMINAMATH_CALUDE_inequality_proof_l445_44583

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l445_44583


namespace NUMINAMATH_CALUDE_barbaras_to_mikes_age_ratio_l445_44555

/-- Given that Mike is currently 16 years old and Barbara will be 16 years old
    when Mike is 24 years old, prove that the ratio of Barbara's current age
    to Mike's current age is 1:2. -/
theorem barbaras_to_mikes_age_ratio :
  let mike_current_age : ℕ := 16
  let mike_future_age : ℕ := 24
  let barbara_future_age : ℕ := 16
  let age_difference : ℕ := mike_future_age - mike_current_age
  let barbara_current_age : ℕ := barbara_future_age - age_difference
  (barbara_current_age : ℚ) / mike_current_age = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_to_mikes_age_ratio_l445_44555


namespace NUMINAMATH_CALUDE_twelve_million_plus_twelve_thousand_l445_44505

theorem twelve_million_plus_twelve_thousand : 
  12000000 + 12000 = 12012000 := by
  sorry

end NUMINAMATH_CALUDE_twelve_million_plus_twelve_thousand_l445_44505


namespace NUMINAMATH_CALUDE_solve_cupcake_problem_l445_44554

def cupcake_problem (initial_cupcakes : ℕ) (sold_cupcakes : ℕ) (final_cupcakes : ℕ) : Prop :=
  initial_cupcakes - sold_cupcakes + (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20

theorem solve_cupcake_problem :
  cupcake_problem 26 20 26 := by
  sorry

end NUMINAMATH_CALUDE_solve_cupcake_problem_l445_44554


namespace NUMINAMATH_CALUDE_position_of_2013_l445_44537

/-- Represents the position of a number in the arrangement -/
structure Position where
  row : Nat
  column : Nat
  deriving Repr

/-- Calculates the position of a given odd number in the arrangement -/
def position_of_odd_number (n : Nat) : Position :=
  sorry

theorem position_of_2013 : position_of_odd_number 2013 = ⟨45, 17⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2013_l445_44537


namespace NUMINAMATH_CALUDE_first_group_size_l445_44510

/-- The number of people in the first group -/
def P : ℕ := sorry

/-- The amount of work that can be completed by the first group in 3 days -/
def W₁ : ℕ := 3

/-- The number of days it takes the first group to complete W₁ amount of work -/
def D₁ : ℕ := 3

/-- The amount of work that can be completed by 4 people in 3 days -/
def W₂ : ℕ := 4

/-- The number of people in the second group -/
def P₂ : ℕ := 4

/-- The number of days it takes the second group to complete W₂ amount of work -/
def D₂ : ℕ := 3

/-- The theorem stating that the number of people in the first group is 3 -/
theorem first_group_size :
  (P * W₂ * D₁ = P₂ * W₁ * D₂) → P = 3 := by sorry

end NUMINAMATH_CALUDE_first_group_size_l445_44510


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l445_44573

theorem bicycle_price_increase (P : ℝ) : 
  (P * 1.15 = 253) → P = 220 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l445_44573


namespace NUMINAMATH_CALUDE_eight_friends_lineup_l445_44567

theorem eight_friends_lineup (n : ℕ) (h : n = 8) : Nat.factorial n = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_friends_lineup_l445_44567


namespace NUMINAMATH_CALUDE_fifteen_guests_four_rooms_l445_44572

/-- The number of ways to distribute n guests into k rooms such that no room is empty. -/
def distributeGuests (n k : ℕ) : ℕ :=
  (k^n : ℕ) - k * ((k-1)^n : ℕ) + (k.choose 2) * ((k-2)^n : ℕ) - (k.choose 3) * ((k-3)^n : ℕ)

/-- Theorem stating that the number of ways to distribute 15 guests into 4 rooms
    such that no room is empty is equal to 4^15 - 4 * 3^15 + 6 * 2^15 - 4. -/
theorem fifteen_guests_four_rooms :
  distributeGuests 15 4 = 4^15 - 4 * 3^15 + 6 * 2^15 - 4 := by
  sorry

#eval distributeGuests 15 4

end NUMINAMATH_CALUDE_fifteen_guests_four_rooms_l445_44572


namespace NUMINAMATH_CALUDE_problem_1_l445_44544

theorem problem_1 : (1 : ℝ) * (1 + Real.rpow 8 (1/3 : ℝ))^0 + abs (-2) - Real.sqrt 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l445_44544


namespace NUMINAMATH_CALUDE_largest_square_four_digits_base7_l445_44541

/-- Converts a decimal number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 4 digits when written in base 7 -/
def hasFourDigitsBase7 (n : ℕ) : Prop :=
  (toBase7 n).length = 4

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := sorry

theorem largest_square_four_digits_base7 :
  M = (toBase7 66).foldl (fun acc d => acc * 7 + d) 0 ∧
  hasFourDigitsBase7 (M ^ 2) ∧
  ∀ n : ℕ, n > M → ¬hasFourDigitsBase7 (n ^ 2) :=
sorry

end NUMINAMATH_CALUDE_largest_square_four_digits_base7_l445_44541


namespace NUMINAMATH_CALUDE_not_right_triangle_when_A_eq_B_eq_3C_l445_44543

-- Define a triangle with angles A, B, and C
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Theorem statement
theorem not_right_triangle_when_A_eq_B_eq_3C (t : Triangle) 
  (h : t.A = t.B ∧ t.A = 3 * t.C) : 
  ¬ is_right_triangle t := by
  sorry

end NUMINAMATH_CALUDE_not_right_triangle_when_A_eq_B_eq_3C_l445_44543


namespace NUMINAMATH_CALUDE_alexey_min_banks_l445_44558

/-- The minimum number of banks needed to fully insure a given amount of money -/
def min_banks (total_amount : ℕ) (max_payout : ℕ) : ℕ :=
  (total_amount + max_payout - 1) / max_payout

/-- Theorem stating the minimum number of banks needed for Alexey's case -/
theorem alexey_min_banks :
  min_banks 10000000 1400000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_alexey_min_banks_l445_44558
