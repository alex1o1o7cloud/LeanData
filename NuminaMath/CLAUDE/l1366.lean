import Mathlib

namespace flour_added_l1366_136621

/-- Given that Mary already put in 8 cups of flour and the recipe requires 10 cups in total,
    prove that she added 2 more cups of flour. -/
theorem flour_added (initial_flour : ℕ) (total_flour : ℕ) (h1 : initial_flour = 8) (h2 : total_flour = 10) :
  total_flour - initial_flour = 2 := by
  sorry

end flour_added_l1366_136621


namespace loan_interest_calculation_l1366_136647

/-- Calculates simple interest given principal, rate, and time --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem: The simple interest on a loan of $1200 at 3% for 3 years is $108 --/
theorem loan_interest_calculation :
  let principal : ℝ := 1200
  let rate : ℝ := 0.03
  let time : ℝ := 3
  simple_interest principal rate time = 108 := by
  sorry

end loan_interest_calculation_l1366_136647


namespace percentage_of_12_to_80_l1366_136646

theorem percentage_of_12_to_80 : ∀ (x : ℝ), x = 12 ∧ (x / 80) * 100 = 15 := by sorry

end percentage_of_12_to_80_l1366_136646


namespace circle_bisection_l1366_136656

-- Define the two circles
def circle1 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = b^2 + 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4

-- Define the bisection condition
def bisects (a b : ℝ) : Prop := 
  ∀ x y : ℝ, circle1 a b x y → circle2 x y → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ circle1 a b x' y' ∧ circle2 x' y'

-- State the theorem
theorem circle_bisection (a b : ℝ) :
  bisects a b → a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end circle_bisection_l1366_136656


namespace coffee_mix_price_l1366_136622

theorem coffee_mix_price (P : ℝ) : 
  let price_second : ℝ := 2.45
  let total_weight : ℝ := 18
  let mix_price : ℝ := 2.30
  let weight_each : ℝ := 9
  (weight_each * P + weight_each * price_second = total_weight * mix_price) →
  P = 2.15 := by
sorry

end coffee_mix_price_l1366_136622


namespace total_outstanding_credit_l1366_136691

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 416.67

/-- The percentage of automobile installment credit in total consumer installment credit -/
def auto_credit_percentage : ℝ := 36

/-- The amount of credit extended by automobile finance companies in billions of dollars -/
def auto_finance_credit : ℝ := 75

/-- Theorem stating the total outstanding consumer installment credit -/
theorem total_outstanding_credit : 
  total_credit = (2 * auto_finance_credit) / (auto_credit_percentage / 100) := by
  sorry

end total_outstanding_credit_l1366_136691


namespace product_of_sums_is_even_l1366_136642

/-- A card with two numbers -/
structure Card where
  front : Nat
  back : Nat

/-- The set of all cards -/
def deck : Finset Card := sorry

/-- The theorem to prove -/
theorem product_of_sums_is_even :
  (∀ c ∈ deck, c.front ∈ Finset.range 100 ∧ c.back ∈ Finset.range 100) →
  deck.card = 99 →
  (Finset.range 100).card = deck.card + 1 →
  (∀ n ∈ Finset.range 100, (deck.filter (λ c => c.front = n)).card +
    (deck.filter (λ c => c.back = n)).card = 1) →
  Even ((deck.prod (λ c => c.front + c.back))) :=
by sorry

end product_of_sums_is_even_l1366_136642


namespace jim_lamp_purchase_jim_lamp_purchase_correct_l1366_136652

theorem jim_lamp_purchase (lamp_cost : ℕ) (bulb_cost_difference : ℕ) (num_bulbs : ℕ) (total_paid : ℕ) : ℕ :=
  let bulb_cost := lamp_cost - bulb_cost_difference
  let num_lamps := (total_paid - num_bulbs * bulb_cost) / lamp_cost
  num_lamps

#check jim_lamp_purchase 7 4 6 32

theorem jim_lamp_purchase_correct :
  jim_lamp_purchase 7 4 6 32 = 2 := by
  sorry

end jim_lamp_purchase_jim_lamp_purchase_correct_l1366_136652


namespace power_equality_l1366_136611

theorem power_equality : 32^5 * 4^5 = 2^35 := by
  sorry

end power_equality_l1366_136611


namespace complex_determinant_equation_l1366_136605

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∃ (z : ℂ), det z Complex.I 1 Complex.I = 1 + Complex.I ∧ z = 2 - Complex.I := by
  sorry

end complex_determinant_equation_l1366_136605


namespace ordering_of_abc_l1366_136626

theorem ordering_of_abc : 
  let a : ℝ := (1.7 : ℝ) ^ (0.9 : ℝ)
  let b : ℝ := (0.9 : ℝ) ^ (1.7 : ℝ)
  let c : ℝ := 1
  b < c ∧ c < a := by sorry

end ordering_of_abc_l1366_136626


namespace arithmetic_geometric_mean_inequalities_l1366_136636

theorem arithmetic_geometric_mean_inequalities
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b) / 2 ≥ Real.sqrt (a * b) ∧
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c :=
by sorry

end arithmetic_geometric_mean_inequalities_l1366_136636


namespace greatest_integer_inequality_l1366_136634

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5*x - 4 < 3 - 2*x := by sorry

end greatest_integer_inequality_l1366_136634


namespace greatest_perfect_square_under_1000_l1366_136688

theorem greatest_perfect_square_under_1000 : 
  ∀ n : ℕ, n < 1000 → n ≤ 961 ∨ ¬∃ m : ℕ, n = m^2 := by
  sorry

end greatest_perfect_square_under_1000_l1366_136688


namespace negation_of_proposition_l1366_136668

theorem negation_of_proposition (p : Prop) :
  (¬ (∀ x : ℝ, x < 0 → x^2019 < x^2018)) ↔ 
  (∃ x : ℝ, x < 0 ∧ x^2019 ≥ x^2018) := by
sorry

end negation_of_proposition_l1366_136668


namespace equation_solution_l1366_136614

theorem equation_solution : 
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 := by
  sorry

end equation_solution_l1366_136614


namespace periodic_trig_function_l1366_136695

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) where f(4) = 3,
    prove that f(2017) = -3 -/
theorem periodic_trig_function (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 4 = 3 → f 2017 = -3 := by sorry

end periodic_trig_function_l1366_136695


namespace simple_interest_rate_l1366_136653

/-- Given a principal sum and a time period of 7 years, if the simple interest
    is one-fifth of the principal, prove that the annual interest rate is 20/7. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 7 * (20 / 7) / 100 = P / 5) → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end simple_interest_rate_l1366_136653


namespace least_common_multiple_345667_l1366_136651

theorem least_common_multiple_345667 :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) :=
by
  -- The proof goes here
  sorry

end least_common_multiple_345667_l1366_136651


namespace alcohol_percentage_problem_l1366_136618

/-- Proves that the initial alcohol percentage is 30% given the conditions of the problem -/
theorem alcohol_percentage_problem (initial_volume : ℝ) (added_alcohol : ℝ) (final_percentage : ℝ) :
  initial_volume = 6 →
  added_alcohol = 2.4 →
  final_percentage = 50 →
  (∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_alcohol = 
    final_percentage * (initial_volume + added_alcohol) / 100 ∧
    initial_percentage = 30) :=
by sorry

end alcohol_percentage_problem_l1366_136618


namespace power_of_five_l1366_136629

theorem power_of_five (m : ℕ) : 5^m = 5 * 25^2 * 125^3 → m = 14 := by
  sorry

end power_of_five_l1366_136629


namespace geometric_sequence_first_term_l1366_136669

/-- Definition of the sum of a geometric sequence -/
def geometric_sum (a : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a * (1 - q^n) / (1 - q)

/-- Theorem: Given S_4 = 1 and S_8 = 17, the first term a_1 is either 1/15 or -1/5 -/
theorem geometric_sequence_first_term
  (a : ℚ) (q : ℚ)
  (h1 : geometric_sum a q 4 = 1)
  (h2 : geometric_sum a q 8 = 17) :
  a = 1/15 ∨ a = -1/5 :=
sorry

end geometric_sequence_first_term_l1366_136669


namespace product_of_digits_of_non_divisible_by_four_l1366_136603

def numbers : List Nat := [3612, 3620, 3628, 3636, 3641]

def is_divisible_by_four (n : Nat) : Bool :=
  n % 4 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_of_non_divisible_by_four :
  ∃ n ∈ numbers, ¬is_divisible_by_four n ∧ 
  units_digit n * tens_digit n = 4 := by
  sorry

end product_of_digits_of_non_divisible_by_four_l1366_136603


namespace car_distance_theorem_l1366_136607

/-- Calculates the total distance traveled by a car over a given number of hours,
    where the car's speed increases by a fixed amount each hour. -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (List.range hours).foldl (fun acc i => acc + (initialSpeed + i * speedIncrease)) 0

/-- Proves that a car traveling for 12 hours, starting at 45 km/h and increasing
    speed by 2 km/h each hour, travels a total of 672 km. -/
theorem car_distance_theorem :
  totalDistance 45 2 12 = 672 := by
  sorry

#eval totalDistance 45 2 12

end car_distance_theorem_l1366_136607


namespace triangle_perimeter_l1366_136687

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  right_angle : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0

-- Define the squares PQXY and PRWZ
structure Square (A B C D : ℝ × ℝ) : Prop where
  side_length_eq : (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0

-- Define points on a circle
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Main theorem
theorem triangle_perimeter 
  (P Q R X Y Z W : ℝ × ℝ) 
  (h_triangle : Triangle P Q R)
  (h_pq_length : (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 100)
  (h_square_pq : Square P Q X Y)
  (h_square_pr : Square P R W Z)
  (h_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    OnCircle center radius X ∧ 
    OnCircle center radius Y ∧ 
    OnCircle center radius Z ∧ 
    OnCircle center radius W) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 + 
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (10 + 10 * Real.sqrt 2)^2 :=
by sorry

end triangle_perimeter_l1366_136687


namespace intersection_complement_when_a_2_a_value_when_union_equals_A_l1366_136672

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - (a+1)*x + a = 0}

-- Part 1
theorem intersection_complement_when_a_2 :
  A ∩ (Set.univ \ B 2) = {-3} := by sorry

-- Part 2
theorem a_value_when_union_equals_A :
  ∀ a : ℝ, A ∪ B a = A → a = 1 := by sorry

end intersection_complement_when_a_2_a_value_when_union_equals_A_l1366_136672


namespace expression_equals_five_l1366_136678

theorem expression_equals_five :
  (1 - Real.sqrt 5) ^ 0 + |-Real.sqrt 2| - 2 * Real.cos (π / 4) + (1 / 4)⁻¹ = 5 := by
  sorry

end expression_equals_five_l1366_136678


namespace units_digit_of_product_l1366_136675

theorem units_digit_of_product (n : ℕ) : (3^1001 * 7^1002 * 13^1003) % 10 = 9 := by
  sorry

end units_digit_of_product_l1366_136675


namespace max_roses_for_680_l1366_136696

/-- Represents the price of roses in different quantities -/
structure RosePrices where
  individual : ℝ
  oneDozen : ℝ
  twoDozen : ℝ

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (prices : RosePrices) (budget : ℝ) : ℕ :=
  sorry

/-- The theorem stating the maximum number of roses that can be purchased for $680 -/
theorem max_roses_for_680 (prices : RosePrices) 
  (h1 : prices.individual = 4.5)
  (h2 : prices.oneDozen = 36)
  (h3 : prices.twoDozen = 50) :
  maxRoses prices 680 = 318 :=
sorry

end max_roses_for_680_l1366_136696


namespace max_min_difference_l1366_136666

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * abs (x - a)

-- State the theorem
theorem max_min_difference (a : ℝ) (h : a ≥ 2) :
  ∃ M m : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ M ∧ m ≤ f a x) ∧ M - m = 4 :=
sorry

end max_min_difference_l1366_136666


namespace bottle_caps_problem_l1366_136623

theorem bottle_caps_problem (katherine_initial : ℕ) (hippopotamus_eaten : ℕ) : 
  katherine_initial = 34 →
  hippopotamus_eaten = 8 →
  (katherine_initial / 2 : ℕ) - hippopotamus_eaten = 9 :=
by
  sorry

end bottle_caps_problem_l1366_136623


namespace sterilization_tank_capacity_l1366_136681

/-- The total capacity of the sterilization tank in gallons -/
def tank_capacity : ℝ := 100

/-- The initial concentration of bleach in the tank as a decimal -/
def initial_concentration : ℝ := 0.02

/-- The target concentration of bleach in the tank as a decimal -/
def target_concentration : ℝ := 0.05

/-- The amount of solution drained and replaced with pure bleach in gallons -/
def drained_amount : ℝ := 3.0612244898

theorem sterilization_tank_capacity :
  let initial_bleach := initial_concentration * tank_capacity
  let drained_bleach := initial_concentration * drained_amount
  let added_bleach := drained_amount
  let final_bleach := initial_bleach - drained_bleach + added_bleach
  final_bleach = target_concentration * tank_capacity := by
  sorry

end sterilization_tank_capacity_l1366_136681


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1366_136682

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (7 + Real.sqrt (49 - 48)) / 2
  let r₂ := (7 - Real.sqrt (49 - 48)) / 2
  r₁ + r₂ = 7 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1366_136682


namespace no_equal_tuesdays_fridays_l1366_136649

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in the month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem stating that there are no starting days that result in equal Tuesdays and Fridays -/
theorem no_equal_tuesdays_fridays :
  ∀ startDay : DayOfWeek,
    countDayOccurrences startDay DayOfWeek.Tuesday ≠
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end no_equal_tuesdays_fridays_l1366_136649


namespace intersection_A_complement_B_l1366_136628

open Set

-- Define the universe U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x < 0}

-- Define set B
def B : Set ℝ := {x | x ≤ -1}

-- Theorem statement
theorem intersection_A_complement_B : 
  A ∩ (U \ B) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end intersection_A_complement_B_l1366_136628


namespace choose_three_roles_from_eight_people_l1366_136645

def number_of_people : ℕ := 8
def number_of_roles : ℕ := 3

theorem choose_three_roles_from_eight_people : 
  (number_of_people * (number_of_people - 1) * (number_of_people - 2) = 336) := by
  sorry

end choose_three_roles_from_eight_people_l1366_136645


namespace cubic_function_property_l1366_136670

/-- A cubic function g(x) = ax^3 + bx^2 + cx + d with g(0) = 3 and g(1) = 5 satisfies a + 2b + c + 3d = 0 -/
theorem cubic_function_property (a b c d : ℝ) : 
  let g : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d
  (g 0 = 3) → (g 1 = 5) → (a + 2*b + c + 3*d = 0) := by
sorry

end cubic_function_property_l1366_136670


namespace tan_sin_ratio_equals_three_l1366_136684

theorem tan_sin_ratio_equals_three :
  (Real.tan (20 * π / 180) + 4 * Real.sin (20 * π / 180)) / Real.tan (30 * π / 180) = 3 := by
  sorry

end tan_sin_ratio_equals_three_l1366_136684


namespace tax_calculation_l1366_136679

/-- Given a monthly income and a tax rate, calculates the amount paid in taxes -/
def calculate_tax (income : ℝ) (tax_rate : ℝ) : ℝ :=
  income * tax_rate

/-- Proves that for a monthly income of 2120 dollars and a tax rate of 0.4, 
    the amount paid in taxes is 848 dollars -/
theorem tax_calculation :
  calculate_tax 2120 0.4 = 848 := by
sorry

end tax_calculation_l1366_136679


namespace quadratic_transformation_l1366_136663

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 2 * (x - 4)^2 + 8) →
  ∃ n k, ∀ x, 3 * a * x^2 + 3 * b * x + 3 * c = n * (x - 4)^2 + k :=
by sorry

end quadratic_transformation_l1366_136663


namespace number_difference_l1366_136650

theorem number_difference (a b : ℕ) : 
  b = 10 * a + 5 →
  a + b = 22500 →
  b - a = 18410 := by
sorry

end number_difference_l1366_136650


namespace hyperbola_eccentricity_l1366_136671

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity 
  (h : Hyperbola a b) 
  (F A B P O : Point) -- F is the right focus, O is the origin
  (right_branch : Point → Prop) -- Predicate for points on the right branch
  (on_hyperbola : Point → Prop) -- Predicate for points on the hyperbola
  (line_through : Point → Point → Point → Prop) -- Predicate for collinear points
  (symmetric_to : Point → Point → Point → Prop) -- Predicate for point symmetry
  (perpendicular : Point → Point → Point → Point → Prop) -- Predicate for perpendicular lines
  (h_right_focus : F.x > 0 ∧ F.y = 0)
  (h_AB_on_C : on_hyperbola A ∧ on_hyperbola B)
  (h_AB_right : right_branch A ∧ right_branch B)
  (h_line_FAB : line_through F A B)
  (h_A_symmetric : symmetric_to A O P)
  (h_PF_perp_AB : perpendicular P F A B)
  (h_BF_3AF : distance B F = 3 * distance A F)
  : eccentricity h = Real.sqrt 10 / 2 := by
  sorry

end hyperbola_eccentricity_l1366_136671


namespace inequality_proof_l1366_136694

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end inequality_proof_l1366_136694


namespace geometric_sequence_sum_l1366_136699

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 5 = -3/4 →
  a 2 + a 3 + a 4 + a 5 = 5/4 →
  1/a 2 + 1/a 3 + 1/a 4 + 1/a 5 = -4/3 := by
  sorry

end geometric_sequence_sum_l1366_136699


namespace vector_subtraction_l1366_136673

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := fun m ↦ (4, m)

theorem vector_subtraction (m : ℝ) (h : a.1 * (b m).1 + a.2 * (b m).2 = 0) :
  (a.1 - (b m).1, a.2 - (b m).2) = (-3, -4) := by
  sorry

end vector_subtraction_l1366_136673


namespace power_function_even_l1366_136683

theorem power_function_even (α : ℤ) (h1 : 0 ≤ α) (h2 : α ≤ 5) :
  (∀ x : ℝ, (fun x => x^(3 - α)) (-x) = (fun x => x^(3 - α)) x) → α = 1 := by
  sorry

end power_function_even_l1366_136683


namespace amusement_park_tickets_l1366_136613

theorem amusement_park_tickets :
  ∀ (a b c : ℕ),
  a + b + c = 85 →
  7 * a + 4 * b + 2 * c = 500 →
  a = b + 31 →
  a = 56 :=
by
  sorry

end amusement_park_tickets_l1366_136613


namespace marble_problem_l1366_136676

def initial_red_marbles : ℕ := 33
def initial_green_marbles : ℕ := 22

theorem marble_problem :
  (initial_red_marbles : ℚ) / initial_green_marbles = 3 / 2 ∧
  (initial_red_marbles - 18 : ℚ) / (initial_green_marbles + 15) = 2 / 5 :=
by
  sorry

#check marble_problem

end marble_problem_l1366_136676


namespace max_reciprocal_sum_2024_l1366_136612

/-- Given a quadratic equation with roots satisfying specific conditions, 
    the maximum value of the sum of their reciprocals raised to the 2024th power is 2. -/
theorem max_reciprocal_sum_2024 (s p r₁ r₂ : ℝ) : 
  (r₁^2 - s*r₁ + p = 0) →
  (r₂^2 - s*r₂ + p = 0) →
  (∀ (n : ℕ), n ≤ 2023 → r₁^n + r₂^n = s) →
  (∃ (max : ℝ), max = 2 ∧ 
    ∀ (s' p' r₁' r₂' : ℝ), 
      (r₁'^2 - s'*r₁' + p' = 0) →
      (r₂'^2 - s'*r₂' + p' = 0) →
      (∀ (n : ℕ), n ≤ 2023 → r₁'^n + r₂'^n = s') →
      1/r₁'^2024 + 1/r₂'^2024 ≤ max) :=
by sorry

end max_reciprocal_sum_2024_l1366_136612


namespace new_drive_usage_percentage_l1366_136625

def initial_free_space : ℝ := 324
def initial_used_space : ℝ := 850
def initial_document_size : ℝ := 180
def initial_photo_size : ℝ := 380
def initial_video_size : ℝ := 290
def document_compression_ratio : ℝ := 0.05
def photo_compression_ratio : ℝ := 0.12
def video_compression_ratio : ℝ := 0.20
def deleted_photo_size : ℝ := 65.9
def deleted_video_size : ℝ := 98.1
def added_document_size : ℝ := 20.4
def added_photo_size : ℝ := 37.6
def new_drive_size : ℝ := 1500

theorem new_drive_usage_percentage (ε : ℝ) (hε : ε > 0) :
  ∃ (percentage : ℝ),
    abs (percentage - 43.56) < ε ∧
    percentage = 
      (((initial_document_size + added_document_size) * (1 - document_compression_ratio) +
        (initial_photo_size - deleted_photo_size + added_photo_size) * (1 - photo_compression_ratio) +
        (initial_video_size - deleted_video_size) * (1 - video_compression_ratio)) /
       new_drive_size) * 100 :=
by sorry

end new_drive_usage_percentage_l1366_136625


namespace no_solution_for_equation_l1366_136690

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ x ≠ -1 ∧ (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 := by
  sorry

end no_solution_for_equation_l1366_136690


namespace chair_production_theorem_l1366_136648

/-- Represents the chair production scenario in a furniture factory -/
structure ChairProduction where
  workers : ℕ
  individual_rate : ℕ
  total_time : ℕ
  total_chairs : ℕ

/-- Calculates the frequency of producing an additional chair as a group -/
def group_chair_frequency (cp : ChairProduction) : ℚ :=
  cp.total_time / (cp.total_chairs - cp.workers * cp.individual_rate * cp.total_time)

/-- Theorem stating the group chair frequency for the given scenario -/
theorem chair_production_theorem (cp : ChairProduction) 
  (h1 : cp.workers = 3)
  (h2 : cp.individual_rate = 4)
  (h3 : cp.total_time = 6)
  (h4 : cp.total_chairs = 73) :
  group_chair_frequency cp = 6 := by
  sorry

#eval group_chair_frequency ⟨3, 4, 6, 73⟩

end chair_production_theorem_l1366_136648


namespace fraction_equality_l1366_136689

theorem fraction_equality (m : ℝ) (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 := by
  sorry

end fraction_equality_l1366_136689


namespace quadrilateral_diagonal_length_l1366_136677

structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)
  (PQ QR RS SP : ℝ)
  (PR : ℤ)

theorem quadrilateral_diagonal_length 
  (quad : Quadrilateral) 
  (h1 : quad.PQ = 7)
  (h2 : quad.QR = 15)
  (h3 : quad.RS = 7)
  (h4 : quad.SP = 8) :
  9 ≤ quad.PR ∧ quad.PR ≤ 13 :=
sorry

end quadrilateral_diagonal_length_l1366_136677


namespace smallest_nut_count_l1366_136654

def nut_division (N : ℕ) (i : ℕ) : ℕ :=
  match i with
  | 0 => N
  | i + 1 => (nut_division N i - 1) / 5

theorem smallest_nut_count :
  ∀ N : ℕ, (∀ i : ℕ, i ≤ 5 → nut_division N i % 5 = 1) ↔ N ≥ 15621 :=
sorry

end smallest_nut_count_l1366_136654


namespace pasta_preference_ratio_l1366_136631

theorem pasta_preference_ratio :
  let total_students : ℕ := 1000
  let lasagna_pref : ℕ := 300
  let manicotti_pref : ℕ := 200
  let ravioli_pref : ℕ := 150
  let spaghetti_pref : ℕ := 270
  let fettuccine_pref : ℕ := 80
  (spaghetti_pref : ℚ) / (manicotti_pref : ℚ) = 27 / 20 :=
by sorry

end pasta_preference_ratio_l1366_136631


namespace fruit_count_l1366_136640

/-- Given:
  1. If each bag contains 5 oranges and 7 apples, after packing all the apples, there will be 1 orange left.
  2. If each bag contains 9 oranges and 7 apples, after packing all the oranges, there will be 21 apples left.
Prove that the total number of oranges and apples is 85. -/
theorem fruit_count (oranges apples : ℕ) 
  (h1 : ∃ m : ℕ, oranges = 5 * m + 1 ∧ apples = 7 * m)
  (h2 : ∃ n : ℕ, oranges = 9 * n ∧ apples = 7 * n + 21) :
  oranges + apples = 85 := by
sorry

end fruit_count_l1366_136640


namespace secret_code_count_l1366_136680

/-- The number of colors available -/
def num_colors : ℕ := 7

/-- The number of slots to fill -/
def num_slots : ℕ := 5

/-- The number of possible secret codes -/
def num_codes : ℕ := 2520

/-- Theorem: The number of ways to arrange 5 colors chosen from 7 distinct colors is 2520 -/
theorem secret_code_count :
  (Finset.card (Finset.range num_colors)).factorial / 
  (Finset.card (Finset.range (num_colors - num_slots))).factorial = num_codes :=
by sorry

end secret_code_count_l1366_136680


namespace unique_prime_factor_count_l1366_136657

def count_prime_factors (n : ℕ) : ℕ := sorry

theorem unique_prime_factor_count : 
  ∃! x : ℕ, x > 0 ∧ count_prime_factors ((4^11) * (7^5) * (x^2)) = 29 :=
sorry

end unique_prime_factor_count_l1366_136657


namespace connect_four_ratio_l1366_136697

theorem connect_four_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 30) 
  (h2 : won_games = 18) : 
  (won_games : ℚ) / (total_games - won_games) = 3 / 2 := by
  sorry

end connect_four_ratio_l1366_136697


namespace single_digit_sum_l1366_136639

/-- Given two different single-digit numbers A and B where AB × 6 = BBB, 
    prove that A + B = 11 -/
theorem single_digit_sum (A B : ℕ) : 
  A ≠ B ∧ 
  A < 10 ∧ 
  B < 10 ∧ 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 := by
sorry

end single_digit_sum_l1366_136639


namespace equation_solution_l1366_136664

theorem equation_solution : ∃! x : ℝ, (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ∧ x = -8 := by
  sorry

end equation_solution_l1366_136664


namespace decimal_to_fraction_l1366_136693

theorem decimal_to_fraction : 
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (0.4 + (3 : ℚ) / 99) = (n : ℚ) / d := by
  sorry

end decimal_to_fraction_l1366_136693


namespace polynomial_simplification_l1366_136643

theorem polynomial_simplification (x : ℝ) :
  (x + 1)^4 - 4*(x + 1)^3 + 6*(x + 1)^2 - 4*(x + 1) + 1 = x^4 := by
  sorry

end polynomial_simplification_l1366_136643


namespace consecutive_points_segment_length_l1366_136616

/-- Given five consecutive points on a straight line, prove the length of a specific segment -/
theorem consecutive_points_segment_length 
  (a b c d e : ℝ) -- Define points as real numbers representing their positions on the line
  (consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Ensure points are consecutive
  (bc_cd : c - b = 2 * (d - c)) -- bc = 2 cd
  (de_length : e - d = 4) -- de = 4
  (ab_length : b - a = 5) -- ab = 5
  (ae_length : e - a = 18) -- ae = 18
  : c - a = 11 := by -- Prove that ac = 11
  sorry

end consecutive_points_segment_length_l1366_136616


namespace rectangle_circle_tangency_l1366_136638

theorem rectangle_circle_tangency (r : ℝ) (a b : ℝ) : 
  r = 6 →                             -- Circle radius is 6 cm
  a ≥ b →                             -- a is the longer side, b is the shorter side
  b = 2 * r →                         -- Circle is tangent to shorter side
  a * b = 3 * (π * r^2) →             -- Rectangle area is triple the circle area
  b = 12 :=                           -- Shorter side length is 12 cm
by sorry

end rectangle_circle_tangency_l1366_136638


namespace NQ_passes_through_fixed_point_l1366_136615

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line l
def line (k p : ℝ) (x y : ℝ) : Prop := y = k*(x + p/2)

-- Define the intersection points M and N
def intersection_points (p k : ℝ) (M N : ℝ × ℝ) : Prop :=
  parabola p M.1 M.2 ∧ parabola p N.1 N.2 ∧
  line k p M.1 M.2 ∧ line k p N.1 N.2 ∧
  M ≠ N

-- Define the chord length condition
def chord_length_condition (p : ℝ) (M N : ℝ × ℝ) : Prop :=
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16*15

-- Define the third intersection point Q
def third_intersection (p : ℝ) (M N Q : ℝ × ℝ) : Prop :=
  parabola p Q.1 Q.2 ∧ Q ≠ M ∧ Q ≠ N

-- Define point B
def point_B : ℝ × ℝ := (1, -1)

-- Define the condition that MQ passes through B
def MQ_through_B (M Q : ℝ × ℝ) : Prop :=
  (point_B.2 - M.2) * (Q.1 - M.1) = (Q.2 - M.2) * (point_B.1 - M.1)

-- Theorem statement
theorem NQ_passes_through_fixed_point (p k : ℝ) (M N Q : ℝ × ℝ) :
  p > 0 →
  k = 1/2 →
  intersection_points p k M N →
  chord_length_condition p M N →
  third_intersection p M N Q →
  MQ_through_B M Q →
  ∃ (fixed_point : ℝ × ℝ), fixed_point = (1, -4) ∧
    (fixed_point.2 - N.2) * (Q.1 - N.1) = (Q.2 - N.2) * (fixed_point.1 - N.1) :=
sorry

end NQ_passes_through_fixed_point_l1366_136615


namespace treasure_chest_gems_l1366_136610

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) 
    (h1 : diamonds = 45) 
    (h2 : rubies = 5110) : 
  diamonds + rubies = 5155 := by
  sorry

end treasure_chest_gems_l1366_136610


namespace jesse_room_area_l1366_136644

/-- The length of Jesse's room in feet -/
def room_length : ℝ := 12

/-- The width of Jesse's room in feet -/
def room_width : ℝ := 8

/-- The area of Jesse's room floor in square feet -/
def room_area : ℝ := room_length * room_width

theorem jesse_room_area : room_area = 96 := by
  sorry

end jesse_room_area_l1366_136644


namespace proposition_3_proposition_4_l1366_136632

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (lineParallel : Line → Line → Prop)
variable (linePlaneParallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β : Plane)

-- Axioms
axiom distinct_lines : m ≠ n
axiom non_coincident_planes : α ≠ β

-- Theorem for proposition 3
theorem proposition_3 
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : lineParallel m n) :
  parallel α β :=
sorry

-- Theorem for proposition 4
theorem proposition_4
  (h1 : skew m n)
  (h2 : linePlaneParallel m α)
  (h3 : linePlaneParallel m β)
  (h4 : linePlaneParallel n α)
  (h5 : linePlaneParallel n β) :
  parallel α β :=
sorry

end proposition_3_proposition_4_l1366_136632


namespace perpendicular_line_equation_l1366_136635

/-- A line passing through a point and perpendicular to another line -/
def perpendicular_line (x₀ y₀ a b c : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ 
    (∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (a * x + b * y + c = 0)) ∧
    k * (a / b) = -1

theorem perpendicular_line_equation :
  perpendicular_line 1 3 2 (-5) 1 →
  ∀ x y : ℝ, 5 * x + 2 * y - 11 = 0 :=
by sorry

end perpendicular_line_equation_l1366_136635


namespace abs_equation_solution_difference_l1366_136674

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end abs_equation_solution_difference_l1366_136674


namespace solution_inequality1_solution_inequality2_l1366_136624

-- Define the first inequality
def inequality1 (x : ℝ) : Prop := (5 * (x - 1)) / 6 - 1 < (x + 2) / 3

-- Define the second system of inequalities
def inequality2 (x : ℝ) : Prop := 3 * x - 2 ≤ x + 6 ∧ (5 * x + 3) / 2 > x

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℕ | inequality1 x} = {1, 2, 3, 4} :=
sorry

-- Theorem for the second system of inequalities
theorem solution_inequality2 : 
  {x : ℝ | inequality2 x} = {x : ℝ | -1 < x ∧ x ≤ 4} :=
sorry

end solution_inequality1_solution_inequality2_l1366_136624


namespace milk_for_pizza_dough_l1366_136659

/-- Given a ratio of 50 mL of milk for every 250 mL of flour, 
    calculate the amount of milk needed for 1200 mL of flour. -/
theorem milk_for_pizza_dough (flour : ℝ) (milk : ℝ) : 
  flour = 1200 → 
  (milk / flour = 50 / 250) → 
  milk = 240 := by sorry

end milk_for_pizza_dough_l1366_136659


namespace last_two_digits_of_nine_to_2008_l1366_136617

theorem last_two_digits_of_nine_to_2008 : 9^2008 % 100 = 21 := by
  sorry

end last_two_digits_of_nine_to_2008_l1366_136617


namespace andrew_work_days_l1366_136698

/-- Given that Andrew worked 2.5 hours each day and 7.5 hours in total on his Science report,
    prove that he spent 3 days working on it. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) (h1 : hours_per_day = 2.5) (h2 : total_hours = 7.5) :
  total_hours / hours_per_day = 3 := by
  sorry

end andrew_work_days_l1366_136698


namespace triangle_angle_proof_l1366_136630

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) : 
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
sorry

end triangle_angle_proof_l1366_136630


namespace square_perimeter_is_48_l1366_136601

-- Define a square with side length 12
def square_side_length : ℝ := 12

-- Define the perimeter of a square
def square_perimeter (side_length : ℝ) : ℝ := 4 * side_length

-- Theorem: The perimeter of the square with side length 12 cm is 48 cm
theorem square_perimeter_is_48 : 
  square_perimeter square_side_length = 48 := by
  sorry

end square_perimeter_is_48_l1366_136601


namespace exists_non_one_same_first_digit_l1366_136600

/-- Given a natural number n, returns the first digit of n -/
def firstDigit (n : ℕ) : ℕ := sorry

/-- Returns true if all numbers in the list start with the same digit -/
def sameFirstDigit (numbers : List ℕ) : Bool := sorry

theorem exists_non_one_same_first_digit :
  ∃ x : ℕ, x > 0 ∧ 
  let powers := List.range 2015 |>.map (λ i => x^(i+1))
  sameFirstDigit powers ∧ 
  firstDigit x ≠ 1 := by
  sorry

end exists_non_one_same_first_digit_l1366_136600


namespace simplify_expression_l1366_136641

theorem simplify_expression (y : ℝ) : 5*y + 8*y + 2*y + 7 = 15*y + 7 := by
  sorry

end simplify_expression_l1366_136641


namespace subset_implies_a_value_l1366_136604

def A (a : ℝ) : Set ℝ := {1, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem subset_implies_a_value (a : ℝ) (h : B a ⊆ A a) : a = -1 ∨ a = 0 := by
  sorry

end subset_implies_a_value_l1366_136604


namespace first_term_of_specific_sequence_l1366_136686

/-- A geometric sequence is defined by its fifth and sixth terms -/
structure GeometricSequence where
  fifth_term : ℚ
  sixth_term : ℚ

/-- The first term of a geometric sequence -/
def first_term (seq : GeometricSequence) : ℚ :=
  256 / 27

/-- Theorem: Given a geometric sequence where the fifth term is 48 and the sixth term is 72, 
    the first term is 256/27 -/
theorem first_term_of_specific_sequence :
  ∀ (seq : GeometricSequence), 
    seq.fifth_term = 48 ∧ seq.sixth_term = 72 → first_term seq = 256 / 27 :=
by
  sorry

end first_term_of_specific_sequence_l1366_136686


namespace inequality_statements_truth_l1366_136658

theorem inequality_statements_truth :
  let statement1 := ∀ (a b c d : ℝ), a > b ∧ c > d → a - c > b - d
  let statement2 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d
  let statement3 := ∀ (a b : ℝ), a > b ∧ b > 0 → 3 * a > 3 * b
  let statement4 := ∀ (a b : ℝ), a > b ∧ b > 0 → 1 / (a^2) < 1 / (b^2)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ statement4) :=
by sorry

end inequality_statements_truth_l1366_136658


namespace total_weight_calculation_l1366_136692

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 1184

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := number_of_moles * molecular_weight

theorem total_weight_calculation : total_weight = 4736 := by
  sorry

end total_weight_calculation_l1366_136692


namespace base_addition_theorem_l1366_136627

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

theorem base_addition_theorem :
  let base13_number := [3, 5, 7]
  let base14_number := [4, 12, 13]
  (base_to_decimal base13_number 13) + (base_to_decimal base14_number 14) = 1544 := by
  sorry

end base_addition_theorem_l1366_136627


namespace positive_integer_equation_l1366_136619

theorem positive_integer_equation (N : ℕ+) : 15^4 * 28^2 = 12^2 * N^2 ↔ N = 525 := by
  sorry

end positive_integer_equation_l1366_136619


namespace symmetric_points_of_M_l1366_136637

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Original point M -/
def M : Point3D := ⟨1, -2, 3⟩

/-- Symmetric point with respect to xy-plane -/
def symmetricXY (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

/-- Symmetric point with respect to z-axis -/
def symmetricZ (p : Point3D) : Point3D :=
  ⟨-p.x, -p.y, p.z⟩

theorem symmetric_points_of_M :
  (symmetricXY M = ⟨1, -2, -3⟩) ∧ (symmetricZ M = ⟨-1, 2, 3⟩) := by sorry

end symmetric_points_of_M_l1366_136637


namespace sin_690_degrees_l1366_136667

theorem sin_690_degrees : Real.sin (690 * Real.pi / 180) = -1/2 := by
  sorry

end sin_690_degrees_l1366_136667


namespace square_equation_solution_l1366_136608

theorem square_equation_solution : ∃! (N : ℕ), N > 0 ∧ 36^2 * 60^2 = 30^2 * N^2 := by
  sorry

end square_equation_solution_l1366_136608


namespace bobs_final_salary_l1366_136655

/-- Calculates the final salary after two raises and a pay cut -/
def final_salary (initial_salary : ℝ) (first_raise : ℝ) (second_raise : ℝ) (pay_cut : ℝ) : ℝ :=
  let salary_after_first_raise := initial_salary * (1 + first_raise)
  let salary_after_second_raise := salary_after_first_raise * (1 + second_raise)
  salary_after_second_raise * (1 - pay_cut)

/-- Theorem stating that Bob's final salary is $2541 -/
theorem bobs_final_salary :
  final_salary 3000 0.1 0.1 0.3 = 2541 := by
  sorry

end bobs_final_salary_l1366_136655


namespace employees_abroad_l1366_136606

theorem employees_abroad (total : ℕ) (fraction : ℚ) (abroad : ℕ) : 
  total = 450 → fraction = 0.06 → abroad = (total : ℚ) * fraction → abroad = 27 := by
sorry

end employees_abroad_l1366_136606


namespace lcm_gcd_product_l1366_136602

theorem lcm_gcd_product (a b : ℕ) (ha : a = 11) (hb : b = 12) :
  Nat.lcm a b * Nat.gcd a b = 132 := by
  sorry

end lcm_gcd_product_l1366_136602


namespace cube_root_equation_solution_l1366_136665

theorem cube_root_equation_solution :
  ∃ (a b c : ℕ+),
    (2 * (7^(1/3) + 6^(1/3))^(1/2) : ℝ) = a^(1/3) - b^(1/3) + c^(1/3) ∧
    a + b + c = 42 :=
by sorry

end cube_root_equation_solution_l1366_136665


namespace airplane_seat_ratio_l1366_136661

theorem airplane_seat_ratio :
  ∀ (total_seats coach_seats first_class_seats k : ℕ),
    total_seats = 387 →
    coach_seats = 310 →
    coach_seats = k * first_class_seats + 2 →
    first_class_seats + coach_seats = total_seats →
    (coach_seats - 2) / first_class_seats = 4 := by
  sorry

end airplane_seat_ratio_l1366_136661


namespace apple_students_l1366_136685

theorem apple_students (bananas apples both one_fruit : ℕ) 
  (h1 : bananas = 8)
  (h2 : one_fruit = 10)
  (h3 : both = 5)
  (h4 : one_fruit = (apples - both) + (bananas - both)) :
  apples = 12 := by
  sorry

end apple_students_l1366_136685


namespace unique_prime_cube_l1366_136620

theorem unique_prime_cube : ∃! n : ℕ, ∃ p : ℕ,
  Prime p ∧ n = 2 * p + 1 ∧ ∃ m : ℕ, n = m^3 := by
  sorry

end unique_prime_cube_l1366_136620


namespace digits_of_powers_l1366_136660

/-- A number is even and not divisible by 10 -/
def IsEvenNotDivBy10 (n : ℕ) : Prop :=
  Even n ∧ ¬(10 ∣ n)

/-- The tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem digits_of_powers (N : ℕ) (h : IsEvenNotDivBy10 N) :
  TensDigit (N^20) = 7 ∧ HundredsDigit (N^200) = 3 := by
  sorry

end digits_of_powers_l1366_136660


namespace arithmetic_sequence_increasing_iff_l1366_136662

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, a₁ < a₃ if and only if aₙ < aₙ₊₁ for all n -/
theorem arithmetic_sequence_increasing_iff (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 1 < a 3 ↔ ∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end arithmetic_sequence_increasing_iff_l1366_136662


namespace paint_intensity_problem_l1366_136609

/-- Given an original paint intensity of 50%, a new paint intensity of 30%,
    and 2/3 of the original paint replaced, prove that the intensity of
    the added paint solution is 20%. -/
theorem paint_intensity_problem (original_intensity new_intensity fraction_replaced : ℚ)
    (h1 : original_intensity = 50/100)
    (h2 : new_intensity = 30/100)
    (h3 : fraction_replaced = 2/3) :
    let added_intensity := (new_intensity - original_intensity * (1 - fraction_replaced)) / fraction_replaced
    added_intensity = 20/100 := by
  sorry

end paint_intensity_problem_l1366_136609


namespace four_points_same_inradius_congruent_triangles_l1366_136633

-- Define a structure for a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a structure for a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a function to calculate the inradius of a triangle
noncomputable def inradius (t : Triangle) : ℝ := sorry

-- Define a predicate for triangle congruence
def is_congruent (t1 t2 : Triangle) : Prop := sorry

-- Main theorem
theorem four_points_same_inradius_congruent_triangles 
  (A B C D : Point) 
  (h_same_inradius : ∃ r : ℝ, 
    inradius (Triangle.mk A B C) = r ∧
    inradius (Triangle.mk A B D) = r ∧
    inradius (Triangle.mk A C D) = r ∧
    inradius (Triangle.mk B C D) = r) :
  is_congruent (Triangle.mk A B C) (Triangle.mk A B D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk A C D) ∧
  is_congruent (Triangle.mk A B C) (Triangle.mk B C D) :=
sorry

end four_points_same_inradius_congruent_triangles_l1366_136633
