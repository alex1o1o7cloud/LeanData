import Mathlib

namespace sector_central_angle_l3423_342381

/-- Given a circular sector with perimeter 4 and area 1, 
    its central angle measure is 2 radians. -/
theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end sector_central_angle_l3423_342381


namespace only_solution_is_48_l3423_342385

/-- Product of digits function -/
def p (A : ℕ) : ℕ :=
  sorry

/-- Theorem: 48 is the only natural number satisfying A = 1.5 * p(A) -/
theorem only_solution_is_48 :
  ∀ A : ℕ, A = (3/2 : ℚ) * p A ↔ A = 48 :=
by sorry

end only_solution_is_48_l3423_342385


namespace number_of_candies_bought_l3423_342332

/-- Given the cost of snacks and candies, the total number of items, and the total amount spent,
    prove that the number of candies bought is 3. -/
theorem number_of_candies_bought
  (snack_cost : ℕ)
  (candy_cost : ℕ)
  (total_items : ℕ)
  (total_spent : ℕ)
  (h1 : snack_cost = 300)
  (h2 : candy_cost = 500)
  (h3 : total_items = 8)
  (h4 : total_spent = 3000)
  : ∃ (num_candies : ℕ), num_candies = 3 ∧
    ∃ (num_snacks : ℕ),
      num_snacks + num_candies = total_items ∧
      num_snacks * snack_cost + num_candies * candy_cost = total_spent :=
by
  sorry

end number_of_candies_bought_l3423_342332


namespace fuel_tank_capacity_l3423_342317

/-- The initial capacity of the fuel tank in liters -/
def initial_capacity : ℝ := 3000

/-- The amount of fuel remaining on January 1, 2006 in liters -/
def remaining_jan1 : ℝ := 180

/-- The amount of fuel remaining on May 1, 2006 in liters -/
def remaining_may1 : ℝ := 1238

/-- The total volume of fuel used from November 1, 2005 to May 1, 2006 in liters -/
def total_fuel_used : ℝ := 4582

/-- Proof that the initial capacity of the fuel tank is 3000 liters -/
theorem fuel_tank_capacity : 
  initial_capacity = 
    (total_fuel_used + remaining_may1 + remaining_jan1) / 2 :=
by sorry

end fuel_tank_capacity_l3423_342317


namespace addition_subtraction_ratio_l3423_342382

theorem addition_subtraction_ratio (A B : ℝ) (h : A > 0) (h' : B > 0) (h'' : A / B = 7) : 
  (A + B) / (A - B) = 4 / 3 := by
sorry

end addition_subtraction_ratio_l3423_342382


namespace madeline_max_distance_difference_l3423_342346

-- Define the speeds and durations
def madeline_speed : ℝ := 12
def madeline_time : ℝ := 3
def max_speed : ℝ := 15
def max_time : ℝ := 2

-- Define the distance function
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem madeline_max_distance_difference :
  distance madeline_speed madeline_time - distance max_speed max_time = 6 := by
  sorry

end madeline_max_distance_difference_l3423_342346


namespace total_sheets_l3423_342338

def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41

theorem total_sheets : sheets_in_desk + sheets_in_backpack = 91 := by
  sorry

end total_sheets_l3423_342338


namespace initial_flower_plates_is_four_l3423_342321

/-- Represents the initial number of flower pattern plates Jack has. -/
def initial_flower_plates : ℕ := sorry

/-- Represents the number of checked pattern plates Jack has. -/
def checked_plates : ℕ := 8

/-- Represents the number of polka dotted plates Jack buys. -/
def polka_dotted_plates : ℕ := 2 * checked_plates

/-- Represents the total number of plates Jack has after buying polka dotted plates and smashing one flower plate. -/
def total_plates : ℕ := 27

/-- Theorem stating that the initial number of flower pattern plates is 4. -/
theorem initial_flower_plates_is_four :
  initial_flower_plates = 4 :=
by
  have h1 : initial_flower_plates + checked_plates + polka_dotted_plates - 1 = total_plates := by sorry
  sorry

end initial_flower_plates_is_four_l3423_342321


namespace no_solutions_absolute_value_equation_l3423_342393

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 5| = |x + 3| + 2 := by
sorry

end no_solutions_absolute_value_equation_l3423_342393


namespace power_mod_50_l3423_342302

theorem power_mod_50 : 11^1501 % 50 = 11 := by
  sorry

end power_mod_50_l3423_342302


namespace range_of_m_l3423_342306

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_solvable : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) → (m < -1 ∨ m > 4) :=
by sorry

end range_of_m_l3423_342306


namespace max_value_of_squared_differences_l3423_342369

theorem max_value_of_squared_differences (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) : 
  (∃ (x : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ x) ∧ 
  (∀ (y : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ y → 40 ≤ y) :=
by sorry

end max_value_of_squared_differences_l3423_342369


namespace nested_circles_radius_l3423_342378

theorem nested_circles_radius (B₁ B₃ : ℝ) : 
  B₁ > 0 →
  B₃ > 0 →
  (B₁ + B₃ = π * 6^2) →
  (B₃ - B₁ = (B₁ + B₃) - B₁) →
  (B₁ = π * (3 * Real.sqrt 2)^2) := by
  sorry

end nested_circles_radius_l3423_342378


namespace circular_track_circumference_l3423_342367

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 meeting_time : ℝ) 
  (h1 : speed1 = 7)
  (h2 : speed2 = 8)
  (h3 : meeting_time = 40)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : meeting_time > 0) :
  speed1 * meeting_time + speed2 * meeting_time = 600 :=
by sorry

end circular_track_circumference_l3423_342367


namespace raspberry_juice_volume_l3423_342319

/-- Proves that the original volume of raspberry juice is 6 quarts -/
theorem raspberry_juice_volume : ∀ (original_volume : ℚ),
  (original_volume / 12 + 1 = 3) →
  (original_volume / 4 = 6) := by
  sorry

end raspberry_juice_volume_l3423_342319


namespace increase_average_grades_l3423_342330

theorem increase_average_grades (group_a_avg : ℝ) (group_b_avg : ℝ) 
  (group_a_size : ℕ) (group_b_size : ℕ) (student1_grade : ℝ) (student2_grade : ℝ) :
  group_a_avg = 44.2 →
  group_b_avg = 38.8 →
  group_a_size = 10 →
  group_b_size = 10 →
  student1_grade = 41 →
  student2_grade = 44 →
  let new_group_a_avg := (group_a_avg * group_a_size - student1_grade - student2_grade) / (group_a_size - 2)
  let new_group_b_avg := (group_b_avg * group_b_size + student1_grade + student2_grade) / (group_b_size + 2)
  new_group_a_avg > group_a_avg ∧ new_group_b_avg > group_b_avg := by
  sorry

end increase_average_grades_l3423_342330


namespace power_product_rule_l3423_342345

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_product_rule_l3423_342345


namespace quadratic_root_form_n_l3423_342364

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the form (m ± √n) / p for roots of a quadratic equation -/
structure RootForm where
  m : ℤ
  n : ℕ
  p : ℤ

/-- Check if the given RootForm satisfies the conditions for the quadratic equation -/
def isValidRootForm (eq : QuadraticEquation) (rf : RootForm) : Prop :=
  ∃ (x : ℚ), (eq.a * x^2 + eq.b * x + eq.c = 0) ∧
              (x = (rf.m + Real.sqrt rf.n) / rf.p ∨ x = (rf.m - Real.sqrt rf.n) / rf.p) ∧
              Nat.gcd (Nat.gcd rf.m.natAbs rf.n) rf.p.natAbs = 1

theorem quadratic_root_form_n (eq : QuadraticEquation) (rf : RootForm) :
  eq = QuadraticEquation.mk 3 (-7) 2 →
  isValidRootForm eq rf →
  rf.n = 25 := by
  sorry

end quadratic_root_form_n_l3423_342364


namespace equation_solutions_l3423_342394

theorem equation_solutions :
  (∀ x : ℝ, (x - 5)^2 ≠ -1) ∧
  (∀ x : ℝ, |(-2 * x)| + 7 ≠ 0) ∧
  (∃ x : ℝ, Real.sqrt (2 - x) - 3 = 0) ∧
  (∃ x : ℝ, Real.sqrt (2 * x + 6) - 5 = 0) ∧
  (∃ x : ℝ, |(-2 * x)| - 1 = 0) :=
by sorry

end equation_solutions_l3423_342394


namespace candy_bar_cost_is_131_l3423_342311

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John uses -/
def num_quarters : ℕ := 4

/-- The number of dimes John uses -/
def num_dimes : ℕ := 3

/-- The number of nickels John uses -/
def num_nickels : ℕ := 1

/-- The amount of change John receives in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value - 
  change_received

theorem candy_bar_cost_is_131 : candy_bar_cost = 131 := by
  sorry

end candy_bar_cost_is_131_l3423_342311


namespace remainder_17_pow_63_mod_7_l3423_342336

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l3423_342336


namespace min_value_of_expression_min_value_achievable_l3423_342339

theorem min_value_of_expression (x y : ℝ) : (x*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, (x*y - 1)^2 + (x + y)^2 = 1 := by
  sorry

end min_value_of_expression_min_value_achievable_l3423_342339


namespace union_equals_B_implies_a_range_l3423_342386

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| < 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x - 6 < 0}
def C : Set ℝ := {x | x^2 - 2*x - 15 < 0}

-- State the theorem
theorem union_equals_B_implies_a_range (a : ℝ) :
  A ∪ B a = B a → a ∈ Set.Icc (-5) (-1) :=
by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end union_equals_B_implies_a_range_l3423_342386


namespace square_area_proof_l3423_342300

theorem square_area_proof (side_length : ℝ) (h1 : side_length > 0) : 
  (3 * 4 * side_length - (2 * side_length + 2 * (3 * side_length)) = 28) → 
  side_length^2 = 49 := by
  sorry

#check square_area_proof

end square_area_proof_l3423_342300


namespace first_term_of_constant_ratio_l3423_342396

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (h : d = 5) :
  (∃ (c : ℚ), ∀ (n : ℕ), n > 0 → 
    arithmetic_sum a d (5 * n) / arithmetic_sum a d n = c) →
  a = 5 / 2 :=
sorry

end first_term_of_constant_ratio_l3423_342396


namespace arithmetic_sequence_terms_l3423_342344

theorem arithmetic_sequence_terms (a₁ a₂ aₙ : ℕ) (h1 : a₁ = 6) (h2 : a₂ = 9) (h3 : aₙ = 300) :
  ∃ n : ℕ, n = 99 ∧ aₙ = a₁ + (n - 1) * (a₂ - a₁) :=
sorry

end arithmetic_sequence_terms_l3423_342344


namespace greatest_integer_x_cube_less_than_15_l3423_342323

theorem greatest_integer_x_cube_less_than_15 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) < 15 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) ≥ 15 :=
by sorry

end greatest_integer_x_cube_less_than_15_l3423_342323


namespace transport_tax_calculation_l3423_342312

def calculate_transport_tax (engine_power : ℕ) (tax_rate : ℕ) (ownership_months : ℕ) : ℕ :=
  (engine_power * tax_rate * ownership_months) / 12

theorem transport_tax_calculation :
  calculate_transport_tax 250 75 2 = 3125 := by
  sorry

end transport_tax_calculation_l3423_342312


namespace exist_three_digits_for_infinite_square_representations_l3423_342333

/-- A type representing a digit (0-9) -/
def Digit := Fin 10

/-- A function that checks if a digit is nonzero -/
def isNonzeroDigit (d : Digit) : Prop := d.val ≠ 0

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that represents a natural number as a sequence of digits -/
def toDigitSequence (n : ℕ) : List Digit := sorry

/-- A function that checks if a list of digits contains only the given three digits -/
def containsOnlyGivenDigits (seq : List Digit) (d1 d2 d3 : Digit) : Prop :=
  ∀ d ∈ seq, d = d1 ∨ d = d2 ∨ d = d3

/-- The main theorem -/
theorem exist_three_digits_for_infinite_square_representations :
  ∃ (d1 d2 d3 : Digit),
    isNonzeroDigit d1 ∧ isNonzeroDigit d2 ∧ isNonzeroDigit d3 ∧
    ∀ n : ℕ, ∃ m : ℕ, 
      isPerfectSquare m ∧ 
      containsOnlyGivenDigits (toDigitSequence m) d1 d2 d3 := by
  sorry

end exist_three_digits_for_infinite_square_representations_l3423_342333


namespace area_SUVR_area_SUVR_is_141_44_l3423_342335

/-- Triangle PQR with given properties and points S, T, U, V as described -/
structure TrianglePQR where
  /-- Side length PR -/
  pr : ℝ
  /-- Side length PQ -/
  pq : ℝ
  /-- Area of triangle PQR -/
  area : ℝ
  /-- Point S on PR such that PS = 1/3 * PR -/
  s : ℝ
  /-- Point T on PQ such that PT = 1/3 * PQ -/
  t : ℝ
  /-- Point U on ST -/
  u : ℝ
  /-- Point V on QR -/
  v : ℝ
  /-- PR equals 60 -/
  h_pr : pr = 60
  /-- PQ equals 15 -/
  h_pq : pq = 15
  /-- Area of triangle PQR equals 180 -/
  h_area : area = 180
  /-- PS equals 1/3 of PR -/
  h_s : s = 1/3 * pr
  /-- PT equals 1/3 of PQ -/
  h_t : t = 1/3 * pq
  /-- U is on the angle bisector of angle PQR -/
  h_u_bisector : True  -- Placeholder for the angle bisector condition
  /-- V is on the angle bisector of angle PQR -/
  h_v_bisector : True  -- Placeholder for the angle bisector condition

/-- The area of quadrilateral SUVR in the given triangle PQR is 141.44 -/
theorem area_SUVR (tri : TrianglePQR) : ℝ := 141.44

/-- The main theorem: The area of quadrilateral SUVR is 141.44 -/
theorem area_SUVR_is_141_44 (tri : TrianglePQR) : area_SUVR tri = 141.44 := by
  sorry

end area_SUVR_area_SUVR_is_141_44_l3423_342335


namespace absolute_value_and_trig_calculation_l3423_342309

theorem absolute_value_and_trig_calculation : |(-3 : ℝ)| + 2⁻¹ - Real.cos (π / 3) = 3 := by
  sorry

end absolute_value_and_trig_calculation_l3423_342309


namespace tangent_ratio_range_l3423_342379

open Real

-- Define the function f(x) = |e^x - 1|
noncomputable def f (x : ℝ) : ℝ := abs (exp x - 1)

-- Define the theorem
theorem tangent_ratio_range 
  (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) 
  (h₂ : x₂ > 0) 
  (h_perp : (deriv f x₁) * (deriv f x₂) = -1) :
  ∃ (AM BN : ℝ), 
    AM > 0 ∧ BN > 0 ∧ 
    0 < AM / BN ∧ AM / BN < 1 :=
by sorry


end tangent_ratio_range_l3423_342379


namespace inequalities_proof_l3423_342305

theorem inequalities_proof (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : a + b > 0) :
  a / b > -1 ∧ |a| < |b| := by
  sorry

end inequalities_proof_l3423_342305


namespace product_sum_relation_l3423_342372

theorem product_sum_relation (a b m : ℝ) : 
  a * b = m * (a + b) + 12 → b = 10 → b - a = 6 → m = 2 := by
  sorry

end product_sum_relation_l3423_342372


namespace pirate_treasure_distribution_l3423_342358

def coin_distribution (x : ℕ) : ℕ := x * (x + 1) / 2

theorem pirate_treasure_distribution (x : ℕ) :
  (coin_distribution x = 5 * x) → (x + 5 * x = 54) :=
by
  sorry

end pirate_treasure_distribution_l3423_342358


namespace power_mod_eleven_l3423_342322

theorem power_mod_eleven : (Nat.pow 3 101 + 5) % 11 = 8 := by
  sorry

end power_mod_eleven_l3423_342322


namespace no_member_divisible_by_four_l3423_342395

-- Define the set T
def T : Set ℤ := {s | ∃ n : ℤ, s = (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2}

-- Theorem statement
theorem no_member_divisible_by_four : ∀ s ∈ T, ¬(4 ∣ s) := by
  sorry

end no_member_divisible_by_four_l3423_342395


namespace complement_intersection_theorem_l3423_342354

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {3, 4, 5, 6}
def B : Set Nat := {5, 6, 7, 8, 9}

theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {7, 8, 9} :=
by
  sorry

end complement_intersection_theorem_l3423_342354


namespace jim_ate_15_cookies_l3423_342392

def cookies_problem (cookies_per_batch : ℕ) (flour_per_batch : ℕ) 
  (num_flour_bags : ℕ) (flour_bag_weight : ℕ) (cookies_left : ℕ) : Prop :=
  let total_flour := num_flour_bags * flour_bag_weight
  let num_batches := total_flour / flour_per_batch
  let total_cookies := num_batches * cookies_per_batch
  let cookies_eaten := total_cookies - cookies_left
  cookies_eaten = 15

theorem jim_ate_15_cookies :
  cookies_problem 12 2 4 5 105 := by
  sorry

end jim_ate_15_cookies_l3423_342392


namespace pauls_crayons_left_l3423_342353

/-- Represents the number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 336

/-- Represents the initial number of crayons Paul got -/
def initial_crayons : ℕ := 601

/-- Represents the number of erasers Paul got -/
def erasers : ℕ := 406

theorem pauls_crayons_left :
  crayons_left = 336 ∧
  initial_crayons = 601 ∧
  erasers = 406 ∧
  erasers = crayons_left + 70 :=
by sorry

end pauls_crayons_left_l3423_342353


namespace evaluate_expression_l3423_342310

theorem evaluate_expression (b : ℝ) : 
  let x := b + 9
  (x - b + 4) = 13 := by sorry

end evaluate_expression_l3423_342310


namespace solve_for_a_l3423_342314

theorem solve_for_a (a b c d : ℝ) 
  (eq1 : a + b = d) 
  (eq2 : b + c = 6) 
  (eq3 : c + d = 7) : 
  a = 1 := by
sorry

end solve_for_a_l3423_342314


namespace fractional_equation_solution_l3423_342388

theorem fractional_equation_solution :
  ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ (x = 4) := by
  sorry

end fractional_equation_solution_l3423_342388


namespace earthquake_damage_in_usd_l3423_342331

/-- Converts Euros to US Dollars based on a given exchange rate -/
def euro_to_usd (euro_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  euro_amount * exchange_rate

/-- Theorem: The earthquake damage in USD is $75,000,000 -/
theorem earthquake_damage_in_usd :
  let damage_in_euros : ℝ := 50000000
  let exchange_rate : ℝ := 3/2 -- 2 Euros = 3 USD, so 1 Euro = 3/2 USD
  euro_to_usd damage_in_euros exchange_rate = 75000000 := by
  sorry

end earthquake_damage_in_usd_l3423_342331


namespace fixed_point_on_line_l3423_342313

theorem fixed_point_on_line (m : ℝ) : 
  2 * (1/2 : ℝ) + m * ((1/2 : ℝ) - (1/2 : ℝ)) - 1 = 0 := by
  sorry

end fixed_point_on_line_l3423_342313


namespace pond_water_after_45_days_l3423_342375

def water_amount (initial_amount : ℕ) (days : ℕ) : ℕ :=
  initial_amount - days + 2 * (days / 3)

theorem pond_water_after_45_days :
  water_amount 300 45 = 285 := by
  sorry

end pond_water_after_45_days_l3423_342375


namespace atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l3423_342347

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The sample space of all possible outcomes when tossing 3 coins -/
def sampleSpace : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH, CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At most one heads" -/
def atMostOneHeads : Set CoinToss := {CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At least two heads" -/
def atLeastTwoHeads : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH}

/-- Theorem stating that "At most one heads" and "At least two heads" are mutually exclusive -/
theorem atMostOneHeads_atLeastTwoHeads_mutually_exclusive : 
  atMostOneHeads ∩ atLeastTwoHeads = ∅ := by sorry

end atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l3423_342347


namespace range_when_proposition_false_l3423_342320

theorem range_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 4 := by
  sorry

end range_when_proposition_false_l3423_342320


namespace sum_of_arithmetic_series_l3423_342362

/-- Sum of an arithmetic series with given parameters -/
theorem sum_of_arithmetic_series : 
  ∀ (a l d : ℤ) (n : ℕ+),
  a = -48 →
  d = 4 →
  l = 0 →
  a + (n - 1 : ℤ) * d = l →
  (n : ℤ) * (a + l) / 2 = -312 :=
by
  sorry

end sum_of_arithmetic_series_l3423_342362


namespace quadratic_factorization_l3423_342326

theorem quadratic_factorization (x : ℝ) : x^2 + 14*x + 49 = (x + 7)^2 := by
  sorry

end quadratic_factorization_l3423_342326


namespace cleaning_room_time_l3423_342363

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  bathroom : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the room given the other task times -/
def timeCleaningRoom (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.bathroom + t.homework)

/-- Theorem stating that given the specific task times, the time spent cleaning the room is 35 minutes -/
theorem cleaning_room_time :
  let t : TaskTimes := {
    total := 120,
    laundry := 30,
    bathroom := 15,
    homework := 40
  }
  timeCleaningRoom t = 35 := by sorry

end cleaning_room_time_l3423_342363


namespace student_answer_difference_l3423_342316

theorem student_answer_difference (number : ℕ) (h : number = 288) : 
  (5 : ℚ) / 6 * number - (5 : ℚ) / 16 * number = 150 := by
  sorry

end student_answer_difference_l3423_342316


namespace sum_of_smallest_solutions_l3423_342348

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x - floor x = 1 / (floor x : ℝ)

-- Define a function to check if a real number is a solution
def is_solution (x : ℝ) : Prop := equation x ∧ x > 0

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s1 s2 s3 : ℝ),
    is_solution s1 ∧ is_solution s2 ∧ is_solution s3 ∧
    (∀ (x : ℝ), is_solution x → x ≥ s1) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 → x ≥ s2) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 ∧ x ≠ s2 → x ≥ s3) ∧
    s1 + s2 + s3 = 10 + 1/12 :=
  sorry

end sum_of_smallest_solutions_l3423_342348


namespace calculation_proof_l3423_342343

theorem calculation_proof : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end calculation_proof_l3423_342343


namespace marbles_distribution_l3423_342380

theorem marbles_distribution (x : ℚ) 
  (h1 : x > 0) 
  (h2 : (4 * x + 2) + (2 * x + 1) + 3 * x = 62) : 
  (4 * x + 2 = 254 / 9) ∧ (2 * x + 1 = 127 / 9) ∧ (3 * x = 177 / 9) :=
by sorry

end marbles_distribution_l3423_342380


namespace permutations_of_sees_l3423_342390

theorem permutations_of_sees (n : ℕ) (a b : ℕ) (h1 : n = 4) (h2 : a = 2) (h3 : b = 2) :
  (n.factorial) / (a.factorial * b.factorial) = 6 :=
sorry

end permutations_of_sees_l3423_342390


namespace min_y_value_l3423_342389

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16*x + 40*y) :
  ∃ (y_min : ℝ), y_min = 20 - Real.sqrt 464 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 16*x' + 40*y' → y' ≥ y_min := by
sorry

end min_y_value_l3423_342389


namespace student_congress_sample_size_l3423_342303

theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40) 
  (h2 : students_per_class = 50) 
  (h3 : selected_students = 150) : 
  selected_students = 150 := by
sorry

end student_congress_sample_size_l3423_342303


namespace complex_point_coordinates_l3423_342398

theorem complex_point_coordinates (Z : ℂ) : Z = Complex.I * (1 + Complex.I) → Z.re = -1 ∧ Z.im = 1 := by
  sorry

end complex_point_coordinates_l3423_342398


namespace hyperbola_axis_ratio_l3423_342361

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if the length of the imaginary axis is twice the length of the real axis,
then m = -1/4.
-/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b = 2*a ∧ 
    ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  m = -1/4 := by
sorry

end hyperbola_axis_ratio_l3423_342361


namespace max_value_of_trig_function_l3423_342307

theorem max_value_of_trig_function :
  ∃ (M : ℝ), M = Real.sqrt 5 / 2 ∧ 
  (∀ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 ≤ M) ∧
  (∃ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 = M) := by
sorry

end max_value_of_trig_function_l3423_342307


namespace ab_value_l3423_342368

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B : Set ℝ := {x | ∃ a b : ℝ, x^2 - a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) :
  A ∪ B = {2, 3, 5} →
  A ∩ B = {3} →
  B = {x | x^2 - a*x + b = 0} →
  a * b = 30 := by
  sorry

end ab_value_l3423_342368


namespace mans_age_twice_sons_l3423_342328

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (y : ℕ) : 
  man_age = son_age + 26 →
  son_age = 24 →
  man_age + y = 2 * (son_age + y) →
  y = 2 := by
sorry

end mans_age_twice_sons_l3423_342328


namespace root_square_plus_inverse_square_l3423_342308

theorem root_square_plus_inverse_square (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → m^2 + 1/m^2 = 6 := by
  sorry

end root_square_plus_inverse_square_l3423_342308


namespace binary_digits_difference_l3423_342371

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digits_difference :
  binaryDigits 1500 - binaryDigits 300 = 2 := by
  sorry

end binary_digits_difference_l3423_342371


namespace percy_swims_52_hours_l3423_342350

/-- Represents Percy's swimming schedule and calculates total swimming hours --/
def percy_swimming_hours : ℕ :=
  let weekday_hours := 2  -- 1 hour before school + 1 hour after school
  let weekdays_per_week := 5
  let weekend_hours := 3
  let weeks := 4
  let weekly_hours := weekday_hours * weekdays_per_week + weekend_hours
  weekly_hours * weeks

/-- Theorem stating that Percy swims 52 hours over 4 weeks --/
theorem percy_swims_52_hours : percy_swimming_hours = 52 := by
  sorry

end percy_swims_52_hours_l3423_342350


namespace hadassah_painting_time_l3423_342370

/-- Represents the time taken to paint paintings and take breaks -/
def total_time (small_paint_rate : ℝ) (large_paint_rate : ℝ) (small_count : ℕ) (large_count : ℕ) (break_duration : ℝ) (paintings_per_break : ℕ) : ℝ :=
  let small_time := small_paint_rate * small_count
  let large_time := large_paint_rate * large_count
  let total_paintings := small_count + large_count
  let break_count := total_paintings / paintings_per_break
  let break_time := break_count * break_duration
  small_time + large_time + break_time

/-- Theorem stating the total time Hadassah takes to finish all paintings -/
theorem hadassah_painting_time : 
  let small_paint_rate := 6 / 12
  let large_paint_rate := 8 / 6
  let small_count := 15
  let large_count := 10
  let break_duration := 0.5
  let paintings_per_break := 3
  total_time small_paint_rate large_paint_rate small_count large_count break_duration paintings_per_break = 24.8 := by
  sorry

end hadassah_painting_time_l3423_342370


namespace lower_limit_of_x_l3423_342373

theorem lower_limit_of_x (n x y : ℤ) : 
  x > n → 
  x < 8 → 
  y > 8 → 
  y < 13 → 
  (∀ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 → b - a ≤ 7) → 
  (∃ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 ∧ b - a = 7) → 
  n = 2 := by
sorry

end lower_limit_of_x_l3423_342373


namespace prob_rain_weekend_is_correct_l3423_342387

-- Define the probabilities of rain for each day
def prob_rain_friday : ℝ := 0.30
def prob_rain_saturday : ℝ := 0.60
def prob_rain_sunday : ℝ := 0.40

-- Define the probability of rain on at least one day during the weekend
def prob_rain_weekend : ℝ := 1 - (1 - prob_rain_friday) * (1 - prob_rain_saturday) * (1 - prob_rain_sunday)

-- Theorem statement
theorem prob_rain_weekend_is_correct : 
  prob_rain_weekend = 0.832 := by sorry

end prob_rain_weekend_is_correct_l3423_342387


namespace parallel_line_distance_is_twelve_l3423_342341

/-- Represents a circle with three equally spaced parallel lines intersecting it. -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_length : chord1 = 40
  /-- The second chord has length 36 -/
  chord2_length : chord2 = 36
  /-- The third chord has length 40 -/
  chord3_length : chord3 = 40

/-- Theorem stating that the distance between adjacent parallel lines is 12 -/
theorem parallel_line_distance_is_twelve (c : CircleWithParallelLines) : c.line_distance = 12 := by
  sorry


end parallel_line_distance_is_twelve_l3423_342341


namespace complex_number_existence_l3423_342374

theorem complex_number_existence : ∃ z : ℂ, (z^2).re = 5 ∧ z.im ≠ 0 := by
  sorry

end complex_number_existence_l3423_342374


namespace inverse_contrapositive_equivalence_l3423_342352

theorem inverse_contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end inverse_contrapositive_equivalence_l3423_342352


namespace v_1004_eq_3036_l3423_342360

/-- Defines the nth term of the sequence -/
def v (n : ℕ) : ℕ := sorry

/-- The 1004th term of the sequence is 3036 -/
theorem v_1004_eq_3036 : v 1004 = 3036 := by sorry

end v_1004_eq_3036_l3423_342360


namespace two_marbles_in_two_boxes_proof_l3423_342342

/-- The number of ways to choose 2 marbles out of 3 distinct marbles 
    and place them in 2 indistinguishable boxes -/
def two_marbles_in_two_boxes : ℕ := 3

/-- The number of distinct marbles -/
def total_marbles : ℕ := 3

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 2

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- Boxes are indistinguishable -/
def boxes_indistinguishable : Prop := True

theorem two_marbles_in_two_boxes_proof :
  two_marbles_in_two_boxes = (total_marbles.choose chosen_marbles) :=
by sorry

end two_marbles_in_two_boxes_proof_l3423_342342


namespace pentagonal_tiles_count_l3423_342329

theorem pentagonal_tiles_count (total_tiles total_edges : ℕ) 
  (h1 : total_tiles = 30)
  (h2 : total_edges = 120) : 
  ∃ (triangular_tiles pentagonal_tiles : ℕ),
    triangular_tiles + pentagonal_tiles = total_tiles ∧
    3 * triangular_tiles + 5 * pentagonal_tiles = total_edges ∧
    pentagonal_tiles = 15 := by
  sorry

end pentagonal_tiles_count_l3423_342329


namespace composite_probability_l3423_342301

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of sides on the special die -/
def special_die_sides : ℕ := 10

/-- The number of standard dice -/
def num_standard_dice : ℕ := 5

/-- The total number of dice -/
def total_dice : ℕ := num_standard_dice + 1

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := standard_die_sides ^ num_standard_dice * special_die_sides

/-- The number of outcomes where the product is not composite -/
def non_composite_outcomes : ℕ := 25

/-- The probability of rolling a composite product -/
def prob_composite : ℚ := 1 - (non_composite_outcomes : ℚ) / total_outcomes

theorem composite_probability : prob_composite = 77735 / 77760 := by
  sorry

end composite_probability_l3423_342301


namespace cyclic_quadrilateral_theorem_l3423_342356

-- Define the points and quadrilateral
variable (A B C D P Q R S X Y : Point₂)

-- Define the cyclic quadrilateral property
def is_cyclic_quadrilateral (A B C D : Point₂) : Prop := sorry

-- Define the property of opposite sides not being parallel
def opposite_sides_not_parallel (A B C D : Point₂) : Prop := sorry

-- Define the interior point property
def is_interior_point (P : Point₂) (A B : Point₂) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point₂) : Prop := sorry

-- Define line intersection
def intersects_at (A B C D X : Point₂) : Prop := sorry

-- Define parallel or coincident lines
def parallel_or_coincide (A B C D : Point₂) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_not_parallel : opposite_sides_not_parallel A B C D)
  (h_P_interior : is_interior_point P A B)
  (h_Q_interior : is_interior_point Q B C)
  (h_R_interior : is_interior_point R C D)
  (h_S_interior : is_interior_point S D A)
  (h_angle1 : angle_eq P D A P C B)
  (h_angle2 : angle_eq Q A B Q D C)
  (h_angle3 : angle_eq R B C R A D)
  (h_angle4 : angle_eq S C D S B A)
  (h_intersect1 : intersects_at A Q B S X)
  (h_intersect2 : intersects_at D Q C S Y) :
  parallel_or_coincide P R X Y :=
sorry

end cyclic_quadrilateral_theorem_l3423_342356


namespace max_value_of_f_l3423_342334

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 3 * Real.tan x) * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 6) → f x ≤ M) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 6) ∧ f x = M) := by
sorry

end max_value_of_f_l3423_342334


namespace major_axis_length_tangent_ellipse_major_axis_l3423_342383

/-- An ellipse with foci at (4, 1 + 2√3) and (4, 1 - 2√3), tangent to both x and y axes -/
structure TangentEllipse where
  /-- The ellipse is tangent to the x-axis -/
  tangent_x : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_y : Bool
  /-- The x-coordinate of both foci -/
  focus_x : ℝ
  /-- The y-coordinate of the first focus -/
  focus_y1 : ℝ
  /-- The y-coordinate of the second focus -/
  focus_y2 : ℝ
  /-- Ensure the foci are correctly positioned -/
  foci_constraint : focus_x = 4 ∧ focus_y1 = 1 + 2 * Real.sqrt 3 ∧ focus_y2 = 1 - 2 * Real.sqrt 3

/-- The length of the major axis of the ellipse is 2 -/
theorem major_axis_length (e : TangentEllipse) : ℝ :=
  2

/-- The theorem stating that the major axis length of the given ellipse is 2 -/
theorem tangent_ellipse_major_axis (e : TangentEllipse) (h1 : e.tangent_x = true) (h2 : e.tangent_y = true) :
  major_axis_length e = 2 := by
  sorry

end major_axis_length_tangent_ellipse_major_axis_l3423_342383


namespace at_least_one_greater_than_point_seven_l3423_342351

theorem at_least_one_greater_than_point_seven (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (max a (max (b^2) (1 / (a^2 + b))) : ℝ) > 0.7 := by
  sorry

end at_least_one_greater_than_point_seven_l3423_342351


namespace right_triangle_square_areas_l3423_342365

theorem right_triangle_square_areas (P Q R : ℝ × ℝ) 
  (right_angle_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0)
  (square_QR_area : (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 144)
  (square_PR_area : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 169) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 := by
sorry

end right_triangle_square_areas_l3423_342365


namespace three_digit_squares_divisible_by_12_l3423_342357

theorem three_digit_squares_divisible_by_12 :
  (∃! (l : List Nat), l = (List.range 22).filter (fun n => 
    10 ≤ n ∧ n ≤ 31 ∧ (n^2 % 12 = 0)) ∧ l.length = 4) := by
  sorry

end three_digit_squares_divisible_by_12_l3423_342357


namespace solution_set_equivalence_l3423_342340

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/2

-- Define the solution set of f(10^x) > 0
def solution_set_f_exp (x : ℝ) : Prop := x < -Real.log 2 / Real.log 10

-- Theorem statement
theorem solution_set_equivalence :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (10^x) > 0 ↔ solution_set_f_exp x) :=
sorry

end solution_set_equivalence_l3423_342340


namespace equation_solution_l3423_342376

theorem equation_solution : 
  ∃ x : ℚ, (x - 1) / 2 - (2 - x) / 3 = 2 ∧ x = 19 / 5 := by
sorry

end equation_solution_l3423_342376


namespace range_of_a_l3423_342359

/-- An odd function f(x) = ax³ + bx² + cx + d satisfying certain conditions -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating the range of 'a' given the conditions -/
theorem range_of_a (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is odd
  (f a b c d 1 = 1) →  -- f(1) = 1
  (∀ x ∈ Set.Icc (-1) 1, |f a b c d x| ≤ 1) →  -- |f(x)| ≤ 1 for x ∈ [-1, 1]
  a ∈ Set.Icc (-1/2) 4 :=  -- a ∈ [-1/2, 4]
by sorry

end range_of_a_l3423_342359


namespace three_fish_added_l3423_342337

/-- The number of fish added to a barrel -/
def fish_added (initial_a initial_b final_total : ℕ) : ℕ :=
  final_total - (initial_a + initial_b)

/-- Theorem: Given the initial numbers of fish and the final total, prove that 3 fish were added -/
theorem three_fish_added : fish_added 4 3 10 = 3 := by
  sorry

end three_fish_added_l3423_342337


namespace equation_solution_l3423_342377

theorem equation_solution :
  let f (x : ℝ) := (x^2 - 11*x + 24)/(x-3) + (4*x^2 + 20*x - 32)/(2*x - 4)
  ∃ x₁ x₂ : ℝ, 
    x₁ = (-15 - Real.sqrt 417) / 4 ∧
    x₂ = (-15 + Real.sqrt 417) / 4 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end equation_solution_l3423_342377


namespace allen_blocks_count_l3423_342391

/-- The number of blocks for each color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def number_of_colors : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * number_of_colors

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end allen_blocks_count_l3423_342391


namespace art_students_count_l3423_342349

/-- Given a high school with the following student enrollment:
  * 500 total students
  * 50 students taking music
  * 10 students taking both music and art
  * 440 students taking neither music nor art
  Prove that the number of students taking art is 20. -/
theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 50)
  (h3 : both = 10)
  (h4 : neither = 440) :
  total - music - neither + both = 20 := by
  sorry

#check art_students_count

end art_students_count_l3423_342349


namespace speed_in_still_water_l3423_342366

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 60) 
  (h2 : downstream_speed = 90) : 
  (upstream_speed + downstream_speed) / 2 = 75 := by
  sorry

end speed_in_still_water_l3423_342366


namespace power_function_odd_condition_l3423_342318

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - 5*m + 7) * x^(m-2)
  is_power_function f ∧ is_odd_function f → m = 3 :=
by sorry

end power_function_odd_condition_l3423_342318


namespace expression_evaluation_l3423_342397

theorem expression_evaluation : 200 * (200 - 7) - (200 * 200 - 7) = -1393 := by
  sorry

end expression_evaluation_l3423_342397


namespace fraction_zero_l3423_342315

theorem fraction_zero (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 := by
  sorry

end fraction_zero_l3423_342315


namespace intersection_segment_length_l3423_342399

noncomputable section

/-- Curve C in Cartesian coordinates -/
def C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l in Cartesian coordinates -/
def l (x y : ℝ) : Prop := y = x + 1

/-- Point on both curve C and line l -/
def intersection_point (p : ℝ × ℝ) : Prop :=
  C p.1 p.2 ∧ l p.1 p.2

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_segment_length :
  ∃ (M N : ℝ × ℝ), intersection_point M ∧ intersection_point N ∧ distance M N = 8 :=
sorry

end

end intersection_segment_length_l3423_342399


namespace remainder_problem_l3423_342325

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 11) (h2 : n = 349) : n % 17 = 9 := by
  sorry

end remainder_problem_l3423_342325


namespace equation_roots_existence_l3423_342355

theorem equation_roots_existence :
  (∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0) ∧ 
  (∃ k : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) ∧ 
  (¬ ∃ k : ℝ, ∃! x : ℝ, x^2 - 2*|x| - (2*k + 1)^2 = 0) ∧
  (¬ ∃ k : ℝ, ∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    w^2 - 2*|w| - (2*k + 1)^2 = 0 ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) := by
  sorry

end equation_roots_existence_l3423_342355


namespace sum_of_perpendiculars_equals_height_l3423_342327

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Height of the equilateral triangle -/
  h : ℝ
  /-- Perpendicular distance from the point to side AB -/
  m₁ : ℝ
  /-- Perpendicular distance from the point to side BC -/
  m₂ : ℝ
  /-- Perpendicular distance from the point to side CA -/
  m₃ : ℝ
  /-- The point is inside the triangle -/
  point_inside : 0 < m₁ ∧ 0 < m₂ ∧ 0 < m₃
  /-- The triangle is equilateral -/
  equilateral : h = (Real.sqrt 3 / 2) * a
  /-- The height is positive -/
  height_positive : 0 < h

/-- 
The sum of perpendiculars from any point inside an equilateral triangle 
to its sides equals the triangle's height
-/
theorem sum_of_perpendiculars_equals_height (t : EquilateralTriangleWithPoint) : 
  t.m₁ + t.m₂ + t.m₃ = t.h := by
  sorry

end sum_of_perpendiculars_equals_height_l3423_342327


namespace profit_maximized_at_150_l3423_342304

/-- The profit function for a company based on the number of machines -/
def profit (x : ℝ) : ℝ := -25 * x^2 + 7500 * x

/-- Theorem stating that the profit is maximized when x = 150 -/
theorem profit_maximized_at_150 :
  ∃ (x_max : ℝ), x_max = 150 ∧ ∀ (x : ℝ), profit x ≤ profit x_max :=
sorry

end profit_maximized_at_150_l3423_342304


namespace rationalize_denominator_sqrt3_plus_1_l3423_342324

theorem rationalize_denominator_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end rationalize_denominator_sqrt3_plus_1_l3423_342324


namespace escalator_steps_l3423_342384

/-- The number of steps Al counts walking down the escalator -/
def al_steps : ℕ := 150

/-- The number of steps Bob counts walking up the escalator -/
def bob_steps : ℕ := 75

/-- The ratio of Al's walking speed to Bob's walking speed -/
def speed_ratio : ℕ := 3

/-- The number of steps visible on the escalator at any given time -/
def visible_steps : ℕ := 120

/-- Theorem stating that given the conditions, the number of visible steps on the escalator is 120 -/
theorem escalator_steps : 
  ∀ (al_count bob_count : ℕ) (speed_ratio : ℕ),
    al_count = al_steps →
    bob_count = bob_steps →
    speed_ratio = 3 →
    visible_steps = 120 := by sorry

end escalator_steps_l3423_342384
