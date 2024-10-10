import Mathlib

namespace min_value_trigonometric_expression_min_value_achievable_l2443_244312

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
  (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 ≥ 500 :=
by sorry

theorem min_value_achievable :
  ∃ α β : ℝ, (3 * Real.cos α + 6 * Real.sin β - 10)^2 + 
             (3 * Real.sin α + 6 * Real.cos β + 4 * Real.cos (α + β) - 20)^2 = 500 :=
by sorry

end min_value_trigonometric_expression_min_value_achievable_l2443_244312


namespace alice_bob_number_game_l2443_244329

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem alice_bob_number_game (alice_num bob_num : ℕ) : 
  (1 ≤ alice_num ∧ alice_num ≤ 50) →
  (1 ≤ bob_num ∧ bob_num ≤ 50) →
  (alice_num ≠ 1) →
  (is_prime bob_num) →
  (∃ m : ℕ, 100 * bob_num + alice_num = m * m) →
  (alice_num = 24 ∨ alice_num = 61) :=
by sorry

end alice_bob_number_game_l2443_244329


namespace tom_car_lease_annual_cost_l2443_244336

/-- Calculates the annual cost of Tom's car lease -/
theorem tom_car_lease_annual_cost :
  let miles_mon_wed_fri : ℕ := 50
  let miles_other_days : ℕ := 100
  let days_mon_wed_fri : ℕ := 3
  let days_other : ℕ := 4
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52

  let weekly_miles : ℕ := miles_mon_wed_fri * days_mon_wed_fri + miles_other_days * days_other
  let weekly_mileage_cost : ℚ := (weekly_miles : ℚ) * cost_per_mile
  let total_weekly_cost : ℚ := weekly_mileage_cost + weekly_fee
  let annual_cost : ℚ := total_weekly_cost * weeks_per_year

  annual_cost = 8060 := by
sorry


end tom_car_lease_annual_cost_l2443_244336


namespace true_propositions_l2443_244366

theorem true_propositions :
  (∃ x : ℝ, x^3 < 1) ∧
  (∃ x : ℝ, x^2 + 1 > 0) ∧
  ¬(∃ x : ℚ, x^2 = 2) ∧
  ¬(∃ x : ℕ, x^3 > x^2) :=
by sorry

end true_propositions_l2443_244366


namespace parabola_point_x_coordinate_l2443_244395

/-- The x-coordinate of a point on a parabola with a given distance from its focus -/
theorem parabola_point_x_coordinate (x y : ℝ) :
  y^2 = 4*x →  -- Point (x, y) is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 4^2 →  -- Distance from (x, y) to focus (1, 0) is 4
  x = 3 := by sorry

end parabola_point_x_coordinate_l2443_244395


namespace projection_shape_theorem_l2443_244357

/-- Represents a plane in 3D space -/
structure Plane

/-- Represents a point in 3D space -/
structure Point

/-- Represents a triangle in 3D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the projection of a point onto a plane -/
def project (p : Point) (plane : Plane) : Point :=
  sorry

/-- Determines if a point is outside a plane -/
def isOutside (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a point is on a plane -/
def isOn (p : Point) (plane : Plane) : Prop :=
  sorry

/-- Determines if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  sorry

/-- Represents the shape formed by projections -/
inductive ProjectionShape
  | LineSegment
  | ObtuseTriangle

theorem projection_shape_theorem (ABC : Triangle) (a : Plane) :
  isRightTriangle ABC →
  isOn ABC.B a →
  isOn ABC.C a →
  isOutside ABC.A a →
  (project ABC.A a ≠ ABC.B ∧ project ABC.A a ≠ ABC.C) →
  (∃ shape : ProjectionShape, 
    (shape = ProjectionShape.LineSegment ∨ shape = ProjectionShape.ObtuseTriangle)) :=
  sorry

end projection_shape_theorem_l2443_244357


namespace train_length_l2443_244369

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 → time = 16 → speed * time * (5 / 18) = 120 := by
  sorry

end train_length_l2443_244369


namespace odd_prime_sqrt_sum_l2443_244373

theorem odd_prime_sqrt_sum (p : ℕ) : 
  Prime p ↔ (∃ m : ℕ, ∃ n : ℕ, Real.sqrt m + Real.sqrt (m + p) = n) ∧ Odd p := by
  sorry

end odd_prime_sqrt_sum_l2443_244373


namespace isosceles_triangle_perimeter_l2443_244324

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 7) :
  let perimeter := a + 2 * b
  perimeter = 17 := by
  sorry

end isosceles_triangle_perimeter_l2443_244324


namespace diophantine_equation_solutions_l2443_244396

theorem diophantine_equation_solutions : 
  ∀ x y : ℤ, 5 * x^2 + 5 * x * y + 5 * y^2 = 7 * x + 14 * y ↔ 
  (x = -1 ∧ y = 3) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 2) :=
by sorry

end diophantine_equation_solutions_l2443_244396


namespace compound_molecular_weight_l2443_244337

/-- The atomic weight of Hydrogen in atomic mass units (amu) -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of Chromium in atomic mass units (amu) -/
def atomic_weight_Cr : ℝ := 51.996

/-- The atomic weight of Oxygen in atomic mass units (amu) -/
def atomic_weight_O : ℝ := 15.999

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 2

/-- The number of Chromium atoms in the compound -/
def num_Cr : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight : ℝ := num_H * atomic_weight_H + num_Cr * atomic_weight_Cr + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 118.008 := by
  sorry

end compound_molecular_weight_l2443_244337


namespace min_beta_delta_sum_l2443_244311

theorem min_beta_delta_sum (g : ℂ → ℂ) (β δ : ℂ) :
  (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β * z + δ) →
  (g 1).im = 0 →
  (g (-Complex.I)).im = 0 →
  ∃ (β₀ δ₀ : ℂ), Complex.abs β₀ + Complex.abs δ₀ = 2 ∧
    ∀ β' δ', (∀ z, g z = (3 + 2 * Complex.I) * z^2 + β' * z + δ') →
              (g 1).im = 0 →
              (g (-Complex.I)).im = 0 →
              Complex.abs β' + Complex.abs δ' ≥ 2 :=
sorry

end min_beta_delta_sum_l2443_244311


namespace modulus_of_complex_fraction_l2443_244332

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (3 + i) / (i^2)
  Complex.abs z = Real.sqrt 10 := by sorry

end modulus_of_complex_fraction_l2443_244332


namespace P_symmetric_l2443_244377

variable (x y z : ℝ)

noncomputable def P : ℕ → ℝ → ℝ → ℝ → ℝ
| 0, _, _, _ => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem P_symmetric (m : ℕ) : 
  P m x y z = P m y x z ∧ 
  P m x y z = P m x z y ∧ 
  P m x y z = P m z y x :=
by sorry

end P_symmetric_l2443_244377


namespace inequality_proof_l2443_244371

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_abc : a * b + b * c + c * a = a * b * c) :
  (a^a * (b^2 + c^2)) / ((a^a - 1)^2) +
  (b^b * (c^2 + a^2)) / ((b^b - 1)^2) +
  (c^c * (a^2 + b^2)) / ((c^c - 1)^2) ≥
  18 * ((a + b + c) / (a * b * c - 1))^2 := by
sorry

end inequality_proof_l2443_244371


namespace pet_store_house_cats_l2443_244317

theorem pet_store_house_cats 
  (initial_siamese : ℕ)
  (sold : ℕ)
  (remaining : ℕ)
  (h1 : initial_siamese = 19)
  (h2 : sold = 56)
  (h3 : remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 45 ∧ 
    initial_siamese + initial_house = sold + remaining :=
by sorry

end pet_store_house_cats_l2443_244317


namespace parts_cost_is_800_l2443_244342

/-- Represents the business model of John's computer assembly and sales --/
structure ComputerBusiness where
  partsCost : ℝ  -- Cost of parts for each computer
  sellMultiplier : ℝ  -- Multiplier for selling price
  monthlyProduction : ℕ  -- Number of computers produced per month
  monthlyRent : ℝ  -- Monthly rent cost
  monthlyExtraExpenses : ℝ  -- Monthly non-rent extra expenses
  monthlyProfit : ℝ  -- Monthly profit

/-- Calculates the monthly revenue --/
def monthlyRevenue (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * (b.sellMultiplier * b.partsCost)

/-- Calculates the monthly expenses --/
def monthlyExpenses (b : ComputerBusiness) : ℝ :=
  b.monthlyProduction * b.partsCost + b.monthlyRent + b.monthlyExtraExpenses

/-- Theorem stating that the cost of parts for each computer is $800 --/
theorem parts_cost_is_800 (b : ComputerBusiness)
    (h1 : b.sellMultiplier = 1.4)
    (h2 : b.monthlyProduction = 60)
    (h3 : b.monthlyRent = 5000)
    (h4 : b.monthlyExtraExpenses = 3000)
    (h5 : b.monthlyProfit = 11200)
    (h6 : monthlyRevenue b - monthlyExpenses b = b.monthlyProfit) :
    b.partsCost = 800 := by
  sorry

end parts_cost_is_800_l2443_244342


namespace function_linearity_l2443_244322

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h_continuous : Continuous f)
variable (h_additive : ∀ x y : ℝ, f (x + y) = f x + f y)

-- State the theorem
theorem function_linearity :
  ∃ C : ℝ, (∀ x : ℝ, f x = C * x) ∧ C = f 1 :=
sorry

end function_linearity_l2443_244322


namespace function_value_at_specific_point_l2443_244393

/-- The base-3 logarithm -/
noncomputable def log3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

/-- The base-10 logarithm -/
noncomputable def lg (x : ℝ) : ℝ := (Real.log x) / (Real.log 10)

/-- The given function f -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin x - b * log3 (Real.sqrt (x^2 + 1) - x) + 1

theorem function_value_at_specific_point
  (a b : ℝ) (h : f a b (lg (log3 10)) = 5) :
  f a b (lg (lg 3)) = -3 := by
  sorry

end function_value_at_specific_point_l2443_244393


namespace smallest_fraction_between_l2443_244352

theorem smallest_fraction_between (p q : ℕ+) : 
  (4 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (3 : ℚ) / 8 ∧ 
  (∀ p' q' : ℕ+, (4 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (3 : ℚ) / 8 → q ≤ q') →
  q - p = 12 := by
sorry

end smallest_fraction_between_l2443_244352


namespace same_heads_probability_l2443_244350

/-- The number of coins Keiko tosses -/
def keiko_coins : ℕ := 2

/-- The number of coins Ephraim tosses -/
def ephraim_coins : ℕ := 3

/-- The probability of getting the same number of heads -/
def same_heads_prob : ℚ := 3/16

/-- 
Theorem: Given that Keiko tosses 2 coins and Ephraim tosses 3 coins, 
the probability that Ephraim gets the same number of heads as Keiko is 3/16.
-/
theorem same_heads_probability : 
  let outcomes := 2^(keiko_coins + ephraim_coins)
  let favorable_outcomes := (keiko_coins + 1) * (ephraim_coins + 1) / 2
  (favorable_outcomes : ℚ) / outcomes = same_heads_prob := by
  sorry

end same_heads_probability_l2443_244350


namespace element_in_complement_l2443_244392

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,5}
def P : Set Nat := {2,4}

theorem element_in_complement : 3 ∈ (U \ (M ∪ P)) := by
  sorry

end element_in_complement_l2443_244392


namespace geometric_sequence_problem_l2443_244313

theorem geometric_sequence_problem (b : ℝ) (h1 : b > 0) :
  (∃ r : ℝ, 30 * r = b ∧ b * r = 3/8) → b = 15/2 := by
  sorry

end geometric_sequence_problem_l2443_244313


namespace number_categorization_l2443_244301

def given_numbers : List ℚ := [8, -1, -2/5, 3/5, 0, 1/3, -10/7, 5, -20/7]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n
def is_fraction (x : ℚ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

def positive_set : Set ℚ := {x | is_positive x}
def negative_set : Set ℚ := {x | is_negative x}
def integer_set : Set ℚ := {x | is_integer x}
def fraction_set : Set ℚ := {x | is_fraction x}
def non_negative_rational_set : Set ℚ := {x | is_non_negative_rational x}

theorem number_categorization :
  positive_set = {8, 3/5, 1/3, 5} ∧
  negative_set = {-1, -2/5, -10/7, -20/7} ∧
  integer_set = {8, -1, 0, 5} ∧
  fraction_set = {-2/5, 3/5, 1/3, -10/7, -20/7} ∧
  non_negative_rational_set = {8, 3/5, 0, 1/3, 5} := by
  sorry

end number_categorization_l2443_244301


namespace maintenance_check_increase_l2443_244398

theorem maintenance_check_increase (old_time new_time : ℝ) (h1 : old_time = 45) (h2 : new_time = 60) :
  (new_time - old_time) / old_time * 100 = 33.33 := by
sorry

end maintenance_check_increase_l2443_244398


namespace gym_class_students_l2443_244302

theorem gym_class_students (n : ℕ) : 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 2 → 
  n = 165 ∨ n = 237 := by
sorry

end gym_class_students_l2443_244302


namespace pressure_volume_inverse_proportionality_l2443_244361

/-- Given inverse proportionality of pressure and volume, prove that if the initial pressure is 8 kPa
    at 3.5 liters, then the pressure at 7 liters is 4 kPa. -/
theorem pressure_volume_inverse_proportionality
  (pressure volume : ℝ → ℝ) -- Pressure and volume as functions of time
  (t₀ t₁ : ℝ) -- Initial and final times
  (h_inverse_prop : ∀ t, pressure t * volume t = pressure t₀ * volume t₀) -- Inverse proportionality
  (h_init_volume : volume t₀ = 3.5)
  (h_init_pressure : pressure t₀ = 8)
  (h_final_volume : volume t₁ = 7) :
  pressure t₁ = 4 := by
  sorry

end pressure_volume_inverse_proportionality_l2443_244361


namespace smallest_y_value_l2443_244333

theorem smallest_y_value (x y z : ℝ) : 
  (4 < x ∧ x < z ∧ z < y ∧ y < 10) →
  (∀ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) → (⌊b⌋ - ⌊a⌋ : ℤ) ≤ 5) →
  (∃ a b : ℝ, (4 < a ∧ a < z ∧ z < b ∧ b < 10) ∧ (⌊b⌋ - ⌊a⌋ : ℤ) = 5) →
  9 ≤ y :=
by sorry

end smallest_y_value_l2443_244333


namespace chess_group_players_l2443_244368

/-- The number of players in the chess group -/
def n : ℕ := 10

/-- The total number of games played -/
def total_games : ℕ := 45

/-- Theorem: Given the conditions, the number of players in the chess group is 10 -/
theorem chess_group_players :
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (game : ℕ), game < total_games) ∧
  (∀ (game : ℕ), game < total_games → ∃! (i j : ℕ), i < n ∧ j < n ∧ i ≠ j) ∧
  (n * (n - 1) / 2 = total_games) →
  n = 10 := by
  sorry

end chess_group_players_l2443_244368


namespace cows_that_ran_away_l2443_244390

/-- Represents the problem of determining how many cows ran away --/
theorem cows_that_ran_away 
  (initial_cows : ℕ) 
  (feeding_period : ℕ) 
  (days_passed : ℕ) 
  (h1 : initial_cows = 1000)
  (h2 : feeding_period = 50)
  (h3 : days_passed = 10)
  : ∃ (cows_ran_away : ℕ),
    cows_ran_away = 200 ∧ 
    (initial_cows * feeding_period - initial_cows * days_passed) 
    = (initial_cows - cows_ran_away) * feeding_period :=
by sorry


end cows_that_ran_away_l2443_244390


namespace completing_square_proof_l2443_244380

theorem completing_square_proof (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) := by
  sorry

end completing_square_proof_l2443_244380


namespace sector_radius_l2443_244372

/-- Given a circular sector with area 7 square centimeters and arc length 3.5 cm,
    prove that the radius of the circle is 4 cm. -/
theorem sector_radius (area : ℝ) (arc_length : ℝ) (radius : ℝ) 
    (h_area : area = 7) 
    (h_arc_length : arc_length = 3.5) 
    (h_sector_area : area = (arc_length * radius) / 2) : radius = 4 := by
  sorry

end sector_radius_l2443_244372


namespace sock_selection_l2443_244306

theorem sock_selection (n : ℕ) : 
  (Nat.choose 10 n = 90) → n = 2 := by
sorry

end sock_selection_l2443_244306


namespace arithmetic_expression_equality_l2443_244344

theorem arithmetic_expression_equality : 5 * 7 - (3 * 2 + 5 * 4) / 2 = 22 := by
  sorry

end arithmetic_expression_equality_l2443_244344


namespace at_least_one_equation_has_two_roots_l2443_244378

theorem at_least_one_equation_has_two_roots (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end at_least_one_equation_has_two_roots_l2443_244378


namespace total_sides_of_dice_l2443_244364

/-- The number of dice each person brought -/
def dice_per_person : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 6

/-- The number of people who brought dice -/
def number_of_people : ℕ := 2

/-- Theorem: The total number of sides on all dice brought by two people, 
    each bringing 4 six-sided dice, is 48. -/
theorem total_sides_of_dice : 
  number_of_people * dice_per_person * sides_per_die = 48 := by
  sorry

end total_sides_of_dice_l2443_244364


namespace chicken_bucket_capacity_l2443_244328

/-- Represents the cost of a chicken bucket with sides in dollars -/
def bucket_cost : ℚ := 12

/-- Represents the total amount Monty spent in dollars -/
def total_spent : ℚ := 72

/-- Represents the number of family members Monty fed -/
def family_members : ℕ := 36

/-- Represents the number of people one chicken bucket with sides can feed -/
def people_per_bucket : ℕ := 6

/-- Proves that one chicken bucket with sides can feed 6 people -/
theorem chicken_bucket_capacity :
  (total_spent / bucket_cost) * people_per_bucket = family_members :=
by sorry

end chicken_bucket_capacity_l2443_244328


namespace kangaroo_hop_distance_l2443_244381

theorem kangaroo_hop_distance :
  let a : ℚ := 1/4  -- first term
  let r : ℚ := 3/4  -- common ratio
  let n : ℕ := 6    -- number of hops
  (a * (1 - r^n)) / (1 - r) = 3367/4096 := by
sorry

end kangaroo_hop_distance_l2443_244381


namespace hexagon_side_length_l2443_244340

/-- A regular hexagon with perimeter 60 inches has sides of length 10 inches. -/
theorem hexagon_side_length : ∀ (side_length : ℝ), 
  side_length > 0 →
  6 * side_length = 60 →
  side_length = 10 :=
by
  sorry

end hexagon_side_length_l2443_244340


namespace bob_investment_l2443_244335

theorem bob_investment (interest_rate_1 interest_rate_2 total_interest investment_1 : ℝ)
  (h1 : interest_rate_1 = 0.18)
  (h2 : interest_rate_2 = 0.14)
  (h3 : total_interest = 3360)
  (h4 : investment_1 = 7000)
  (h5 : investment_1 * interest_rate_1 + (total_investment - investment_1) * interest_rate_2 = total_interest) :
  ∃ (total_investment : ℝ), total_investment = 22000 := by
sorry

end bob_investment_l2443_244335


namespace sum_of_digits_9ab_l2443_244343

/-- Represents a number with n repetitions of a digit in base 10 -/
def repeatedDigit (digit : Nat) (n : Nat) : Nat :=
  digit * ((10^n - 1) / 9)

/-- Calculates the sum of digits of a number in base 10 -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_9ab :
  let a := repeatedDigit 9 1977
  let b := repeatedDigit 6 1977
  sumOfDigits (9 * a * b) = 25694 := by
  sorry

end sum_of_digits_9ab_l2443_244343


namespace fraction_of_fraction_of_fraction_problem_solution_l2443_244330

theorem fraction_of_fraction_of_fraction (a b c d : ℚ) (n : ℕ) :
  a * (b * (c * n)) = (a * b * c) * n :=
by sorry

theorem problem_solution : (2 / 5 : ℚ) * ((3 / 4 : ℚ) * ((1 / 6 : ℚ) * 120)) = 6 :=
by sorry

end fraction_of_fraction_of_fraction_problem_solution_l2443_244330


namespace sufficient_not_necessary_condition_l2443_244354

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a = 1/4 → ∀ x : ℝ, x > 0 → x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/4 ∧ ∀ x : ℝ, x > 0 → x + a/x ≥ 1) :=
by sorry

end sufficient_not_necessary_condition_l2443_244354


namespace fourth_person_height_l2443_244320

/-- Represents a person with height, weight, and age -/
structure Person where
  height : ℝ
  weight : ℝ
  age : ℕ

/-- Given conditions for the problem -/
def fourPeople (p1 p2 p3 p4 : Person) : Prop :=
  p1.height < p2.height ∧ p2.height < p3.height ∧ p3.height < p4.height ∧
  p2.height - p1.height = 2 ∧
  p3.height - p2.height = 3 ∧
  p4.height - p3.height = 6 ∧
  p1.weight + p2.weight + p3.weight + p4.weight = 600 ∧
  p1.age = 25 ∧ p2.age = 32 ∧ p3.age = 37 ∧ p4.age = 46 ∧
  (p1.height + p2.height + p3.height + p4.height) / 4 = 72 ∧
  ∀ (i j : Fin 4), (i.val < j.val) → 
    (p1.height * p1.age = p2.height * p2.age) ∧
    (p1.height * p2.weight = p2.height * p1.weight)

/-- Theorem: The fourth person's height is 78.5 inches -/
theorem fourth_person_height (p1 p2 p3 p4 : Person) 
  (h : fourPeople p1 p2 p3 p4) : p4.height = 78.5 := by
  sorry

end fourth_person_height_l2443_244320


namespace square_perimeter_greater_than_circle_circumference_l2443_244318

theorem square_perimeter_greater_than_circle_circumference :
  ∀ (a r : ℝ), a > 0 → r > 0 →
  a^2 = π * r^2 →
  4 * a > 2 * π * r :=
by sorry

end square_perimeter_greater_than_circle_circumference_l2443_244318


namespace factorial_divisibility_implies_inequality_l2443_244303

theorem factorial_divisibility_implies_inequality (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : 
  a ≥ 2 * b + 1 := by
sorry

end factorial_divisibility_implies_inequality_l2443_244303


namespace snacks_sold_l2443_244355

/-- Given the initial number of snacks and ramens in a market, and the final total after some transactions, 
    prove that the number of snacks sold is 599. -/
theorem snacks_sold (initial_snacks : ℕ) (initial_ramens : ℕ) (ramens_bought : ℕ) (final_total : ℕ) :
  initial_snacks = 1238 →
  initial_ramens = initial_snacks + 374 →
  ramens_bought = 276 →
  final_total = 2527 →
  (initial_snacks - (initial_snacks - (initial_ramens + ramens_bought - final_total))) = 599 := by
  sorry

end snacks_sold_l2443_244355


namespace hill_depth_ratio_l2443_244339

/-- Given a hill with its base 300m above the seabed and a total height of 900m,
    prove that the ratio of the depth from the base to the seabed
    to the total height of the hill is 1/3. -/
theorem hill_depth_ratio (base_height : ℝ) (total_height : ℝ) :
  base_height = 300 →
  total_height = 900 →
  base_height / total_height = 1 / 3 := by
  sorry

end hill_depth_ratio_l2443_244339


namespace average_pages_proof_l2443_244345

def book_pages : List Nat := [50, 75, 80, 120, 100, 90, 110, 130]

theorem average_pages_proof :
  (book_pages.sum : ℚ) / book_pages.length = 94.375 := by
  sorry

end average_pages_proof_l2443_244345


namespace first_number_is_45_l2443_244326

/-- Given two positive integers with a ratio of 3:4 and LCM 180, prove the first number is 45 -/
theorem first_number_is_45 (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.lcm a.val b.val = 180) : a = 45 := by
  sorry

end first_number_is_45_l2443_244326


namespace f_is_even_and_increasing_l2443_244304

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
sorry

end f_is_even_and_increasing_l2443_244304


namespace polynomial_equality_sum_of_squares_l2443_244379

theorem polynomial_equality_sum_of_squares (p q r s t u : ℤ) :
  (∀ x : ℝ, 512 * x^3 + 125 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) →
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 6410 := by
  sorry

end polynomial_equality_sum_of_squares_l2443_244379


namespace double_reflection_of_D_l2443_244399

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the line y = x - 1 -/
def reflect_line_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 + 1, p.1 - 1)

/-- The composition of two reflections -/
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ :=
  reflect_line_y_eq_x_minus_1 (reflect_y_axis p)

theorem double_reflection_of_D :
  double_reflection (7, 0) = (1, -8) := by
  sorry

end double_reflection_of_D_l2443_244399


namespace prime_sum_of_squares_l2443_244391

theorem prime_sum_of_squares (k : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_form : p = 4 * k + 1) :
  (∃ (x y m : ℕ), x^2 + y^2 = m * p) ∧
  (∀ (x y m : ℕ), x^2 + y^2 = m * p → m > 1 → 
    ∃ (X Y m' : ℕ), X^2 + Y^2 = m' * p ∧ 0 < m' ∧ m' < m) :=
by sorry

end prime_sum_of_squares_l2443_244391


namespace integers_between_cubes_l2443_244387

theorem integers_between_cubes : ∃ n : ℕ, n = 26 ∧ 
  n = (⌊(9.3 : ℝ)^3⌋ - ⌈(9.2 : ℝ)^3⌉ + 1) := by
  sorry

end integers_between_cubes_l2443_244387


namespace ratio_of_divisor_sums_l2443_244348

def N : ℕ := 36 * 72 * 50 * 81

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums : 
  (sum_odd_divisors N) * 126 = sum_even_divisors N := by sorry

end ratio_of_divisor_sums_l2443_244348


namespace school_pairing_fraction_l2443_244325

theorem school_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) : 
  (t / 4 : ℚ) = (s / 3 : ℚ) → 
  ((t / 4 + s / 3) : ℚ) / ((t + s) : ℚ) = 2 / 7 := by
sorry

end school_pairing_fraction_l2443_244325


namespace price_reduction_sales_increase_l2443_244358

/-- Proves that a 30% price reduction and 80% sales increase results in a 26% revenue increase -/
theorem price_reduction_sales_increase (P S : ℝ) (P_pos : P > 0) (S_pos : S > 0) :
  let new_price := 0.7 * P
  let new_sales := 1.8 * S
  let original_revenue := P * S
  let new_revenue := new_price * new_sales
  (new_revenue - original_revenue) / original_revenue = 0.26 := by
  sorry

end price_reduction_sales_increase_l2443_244358


namespace tennis_ball_order_l2443_244321

theorem tennis_ball_order (white yellow : ℕ) (h1 : white = yellow)
  (h2 : (white : ℚ) / ((yellow : ℚ) + 20) = 8 / 13) :
  white + yellow = 64 := by
  sorry

end tennis_ball_order_l2443_244321


namespace blue_balls_count_l2443_244376

/-- The number of boxes a person has -/
def num_boxes : ℕ := 2

/-- The number of blue balls in each box -/
def blue_balls_per_box : ℕ := 5

/-- The total number of blue balls a person has -/
def total_blue_balls : ℕ := num_boxes * blue_balls_per_box

theorem blue_balls_count : total_blue_balls = 10 := by
  sorry

end blue_balls_count_l2443_244376


namespace simplify_fraction_l2443_244308

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end simplify_fraction_l2443_244308


namespace sum_of_first_100_inverse_terms_l2443_244314

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => sequence_a n + n + 1

def sequence_inverse_a (n : ℕ) : ℚ := 1 / sequence_a n

theorem sum_of_first_100_inverse_terms :
  (Finset.range 100).sum sequence_inverse_a = 200 / 101 := by
  sorry

end sum_of_first_100_inverse_terms_l2443_244314


namespace smallest_base_for_200_proof_l2443_244315

/-- The smallest base in which 200 (base 10) has exactly 6 digits -/
def smallest_base_for_200 : ℕ := 2

theorem smallest_base_for_200_proof :
  smallest_base_for_200 = 2 ∧
  2^7 ≤ 200 ∧
  200 < 2^8 ∧
  ∀ b : ℕ, 1 < b → b < 2 →
    (b^5 > 200 ∨ b^6 ≤ 200) :=
by sorry

end smallest_base_for_200_proof_l2443_244315


namespace product_property_l2443_244316

theorem product_property : ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ (∃ (k : ℤ), 4.02 * (n : ℝ) = (k : ℝ)) ∧ 10 * (4.02 * (n : ℝ)) = 2010 := by
  sorry

end product_property_l2443_244316


namespace HN_passes_through_fixed_point_l2443_244362

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the line segment AB
def line_AB (x y : ℝ) : Prop := y = 2/3 * x - 2

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define a point on line AB
def on_line_AB (p : ℝ × ℝ) : Prop := line_AB p.1 p.2

-- Define the property of T being on the line parallel to x-axis through M
def T_on_parallel_line (M T : ℝ × ℝ) : Prop := M.2 = T.2

-- Define the property of H satisfying MT = TH
def H_satisfies_MT_eq_TH (M T H : ℝ × ℝ) : Prop := 
  (H.1 - T.1 = T.1 - M.1) ∧ (H.2 - T.2 = T.2 - M.2)

-- Main theorem
theorem HN_passes_through_fixed_point :
  ∀ (M N T H : ℝ × ℝ),
  on_ellipse M → on_ellipse N →
  on_line_AB T →
  T_on_parallel_line M T →
  H_satisfies_MT_eq_TH M T H →
  ∃ (t : ℝ), (1 - t) * H.1 + t * N.1 = 0 ∧ (1 - t) * H.2 + t * N.2 = -2 :=
sorry

end HN_passes_through_fixed_point_l2443_244362


namespace quadratic_to_linear_equations_l2443_244341

theorem quadratic_to_linear_equations :
  ∀ x y : ℝ, x^2 - 4*x*y + 4*y^2 = 4 ↔ (x - 2*y + 2 = 0 ∨ x - 2*y - 2 = 0) :=
by sorry

end quadratic_to_linear_equations_l2443_244341


namespace loot_box_loss_l2443_244346

/-- Calculates the average amount lost when buying loot boxes --/
theorem loot_box_loss (loot_box_cost : ℝ) (average_item_value : ℝ) (total_spent : ℝ) :
  loot_box_cost = 5 →
  average_item_value = 3.5 →
  total_spent = 40 →
  total_spent - (total_spent / loot_box_cost * average_item_value) = 12 := by
  sorry

#check loot_box_loss

end loot_box_loss_l2443_244346


namespace factorization_sum_l2443_244323

theorem factorization_sum (a b : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -68 := by
sorry

end factorization_sum_l2443_244323


namespace choose_two_from_seven_eq_twentyone_l2443_244374

/-- The number of ways to choose 2 people from 7 -/
def choose_two_from_seven : ℕ := Nat.choose 7 2

/-- Theorem stating that choosing 2 from 7 results in 21 possibilities -/
theorem choose_two_from_seven_eq_twentyone : choose_two_from_seven = 21 := by
  sorry

end choose_two_from_seven_eq_twentyone_l2443_244374


namespace inequality_solution_set_l2443_244384

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 1 / 2) ↔ x ∈ (Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end inequality_solution_set_l2443_244384


namespace weight_ratio_l2443_244363

def student_weight : ℝ := 79
def total_weight : ℝ := 116
def weight_loss : ℝ := 5

def sister_weight : ℝ := total_weight - student_weight
def student_new_weight : ℝ := student_weight - weight_loss

theorem weight_ratio : student_new_weight / sister_weight = 2 := by sorry

end weight_ratio_l2443_244363


namespace largest_integer_inequality_l2443_244389

theorem largest_integer_inequality : 
  ∀ x : ℤ, x ≤ 10 ↔ (x : ℚ) / 4 + 5 / 6 < 7 / 2 :=
by sorry

end largest_integer_inequality_l2443_244389


namespace equation_represents_parabola_l2443_244388

/-- The equation y^4 - 6x^2 = 3y^2 - 2 represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
  (∀ x y : ℝ, y^4 - 6*x^2 = 3*y^2 - 2 ↔ a*y^2 + b*x + c = 0) :=
sorry

end equation_represents_parabola_l2443_244388


namespace afternoon_letters_indeterminate_l2443_244327

/-- Represents the number of items Jack received at different times of the day -/
structure JacksItems where
  morning_emails : ℕ
  morning_letters : ℕ
  afternoon_emails : ℕ
  afternoon_letters : ℕ

/-- The given conditions about Jack's received items -/
def jack_conditions (items : JacksItems) : Prop :=
  items.morning_emails = 10 ∧
  items.morning_letters = 12 ∧
  items.afternoon_emails = 3 ∧
  items.morning_emails = items.afternoon_emails + 7

/-- Theorem stating that the number of afternoon letters cannot be determined -/
theorem afternoon_letters_indeterminate (items : JacksItems) 
  (h : jack_conditions items) : 
  ¬∃ (n : ℕ), ∀ (items' : JacksItems), 
    jack_conditions items' → items'.afternoon_letters = n :=
sorry

end afternoon_letters_indeterminate_l2443_244327


namespace ivan_tsarevich_revival_l2443_244307

/-- Represents the scenario of Wolf, Ivan Tsarevich, and the Raven --/
structure RevivalScenario where
  initialDistance : ℝ
  wolfSpeed : ℝ
  waterNeeded : ℝ
  springFlowRate : ℝ
  ravenSpeed : ℝ
  ravenWaterLossRate : ℝ

/-- Determines if Ivan Tsarevich can be revived after the given time --/
def canRevive (scenario : RevivalScenario) (time : ℝ) : Prop :=
  let waterCollectionTime := scenario.waterNeeded / scenario.springFlowRate
  let wolfDistance := scenario.wolfSpeed * waterCollectionTime
  let remainingDistance := scenario.initialDistance - wolfDistance
  let meetingTime := remainingDistance / (scenario.ravenSpeed + scenario.wolfSpeed)
  let totalTime := waterCollectionTime + meetingTime
  let waterLost := scenario.ravenWaterLossRate * meetingTime
  totalTime ≤ time ∧ scenario.waterNeeded - waterLost > 0

/-- The main theorem stating that Ivan Tsarevich can be revived after 4 hours --/
theorem ivan_tsarevich_revival (scenario : RevivalScenario)
  (h1 : scenario.initialDistance = 20)
  (h2 : scenario.wolfSpeed = 3)
  (h3 : scenario.waterNeeded = 1)
  (h4 : scenario.springFlowRate = 0.5)
  (h5 : scenario.ravenSpeed = 6)
  (h6 : scenario.ravenWaterLossRate = 0.25) :
  canRevive scenario 4 := by
  sorry

end ivan_tsarevich_revival_l2443_244307


namespace total_routes_to_school_l2443_244305

theorem total_routes_to_school (bus_routes subway_routes : ℕ) 
  (h1 : bus_routes = 3) 
  (h2 : subway_routes = 2) : 
  bus_routes + subway_routes = 5 := by
  sorry

end total_routes_to_school_l2443_244305


namespace star_3_2_l2443_244359

-- Define the ★ operation
def star (a b : ℝ) : ℝ := a^3 + 3*a^2*b + 3*a*b^2 + b^3

-- Theorem statement
theorem star_3_2 : star 3 2 = 125 := by sorry

end star_3_2_l2443_244359


namespace total_sleep_l2443_244394

def sleep_pattern (first_night : ℕ) : Fin 4 → ℕ
| 0 => first_night
| 1 => 2 * first_night
| 2 => 2 * first_night - 3
| 3 => 3 * (2 * first_night - 3)

theorem total_sleep (first_night : ℕ) (h : first_night = 6) : 
  (Finset.sum Finset.univ (sleep_pattern first_night)) = 54 := by
  sorry

end total_sleep_l2443_244394


namespace smallest_whole_number_above_max_perimeter_l2443_244360

theorem smallest_whole_number_above_max_perimeter : ∀ s : ℝ,
  s > 0 →
  s + 7 > 21 →
  s + 21 > 7 →
  7 + 21 > s →
  57 > 7 + 21 + s ∧ 
  ∀ n : ℕ, n < 57 → ∃ s' : ℝ, 
    s' > 0 ∧ 
    s' + 7 > 21 ∧ 
    s' + 21 > 7 ∧ 
    7 + 21 > s' ∧ 
    n ≤ 7 + 21 + s' :=
by sorry

end smallest_whole_number_above_max_perimeter_l2443_244360


namespace sales_difference_is_48_l2443_244347

/-- Represents the baker's sales data --/
structure BakerSales where
  usualPastries : ℕ
  usualBread : ℕ
  todayPastries : ℕ
  todayBread : ℕ
  pastryPrice : ℕ
  breadPrice : ℕ

/-- Calculates the difference between today's sales and the daily average sales --/
def salesDifference (sales : BakerSales) : ℕ :=
  let usualTotal := sales.usualPastries * sales.pastryPrice + sales.usualBread * sales.breadPrice
  let todayTotal := sales.todayPastries * sales.pastryPrice + sales.todayBread * sales.breadPrice
  todayTotal - usualTotal

/-- Theorem stating the difference in sales --/
theorem sales_difference_is_48 :
  ∃ (sales : BakerSales),
    sales.usualPastries = 20 ∧
    sales.usualBread = 10 ∧
    sales.todayPastries = 14 ∧
    sales.todayBread = 25 ∧
    sales.pastryPrice = 2 ∧
    sales.breadPrice = 4 ∧
    salesDifference sales = 48 := by
  sorry

end sales_difference_is_48_l2443_244347


namespace fraction_division_simplification_l2443_244334

theorem fraction_division_simplification : (3 / 4) / (5 / 6) = 9 / 10 := by
  sorry

end fraction_division_simplification_l2443_244334


namespace cheryl_strawberries_l2443_244356

theorem cheryl_strawberries (num_buckets : ℕ) (removed_per_bucket : ℕ) (remaining_per_bucket : ℕ) : 
  num_buckets = 5 →
  removed_per_bucket = 20 →
  remaining_per_bucket = 40 →
  num_buckets * (removed_per_bucket + remaining_per_bucket) = 300 := by
  sorry

end cheryl_strawberries_l2443_244356


namespace weeks_to_save_dress_l2443_244349

def original_price : ℚ := 150
def discount_rate : ℚ := 15 / 100
def initial_savings : ℚ := 35
def odd_week_allowance : ℚ := 30
def even_week_allowance : ℚ := 35
def weekly_arcade_expense : ℚ := 20
def weekly_snack_expense : ℚ := 10

def discounted_price : ℚ := original_price * (1 - discount_rate)
def amount_to_save : ℚ := discounted_price - initial_savings
def biweekly_allowance : ℚ := odd_week_allowance + even_week_allowance
def weekly_expenses : ℚ := weekly_arcade_expense + weekly_snack_expense
def biweekly_savings : ℚ := biweekly_allowance - 2 * weekly_expenses
def average_weekly_savings : ℚ := biweekly_savings / 2

theorem weeks_to_save_dress : 
  ⌈amount_to_save / average_weekly_savings⌉ = 37 := by sorry

end weeks_to_save_dress_l2443_244349


namespace tetrahedron_octahedron_volume_ratio_l2443_244319

/-- The volume of a regular tetrahedron -/
def tetrahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron -/
def octahedronVolume (edgeLength : ℝ) : ℝ := sorry

/-- Theorem: The ratio of the volume of a regular tetrahedron to the volume of a regular octahedron 
    with the same edge length is 1/2 -/
theorem tetrahedron_octahedron_volume_ratio (edgeLength : ℝ) (h : edgeLength > 0) : 
  tetrahedronVolume edgeLength / octahedronVolume edgeLength = 1 / 2 := by sorry

end tetrahedron_octahedron_volume_ratio_l2443_244319


namespace jane_shorter_than_sarah_l2443_244351

-- Define the lengths of the sticks and the covered portion
def pat_stick_length : ℕ := 30
def pat_covered_length : ℕ := 7
def jane_stick_length : ℕ := 22

-- Define Sarah's stick length based on Pat's uncovered portion
def sarah_stick_length : ℕ := 2 * (pat_stick_length - pat_covered_length)

-- State the theorem
theorem jane_shorter_than_sarah : sarah_stick_length - jane_stick_length = 24 := by
  sorry

end jane_shorter_than_sarah_l2443_244351


namespace max_value_f_l2443_244382

/-- Given positive real numbers x, y, z satisfying xyz = 1, 
    the maximum value of f(x, y, z) = (1 - yz + z)(1 - zx + x)(1 - xy + y) is 1, 
    and this maximum is achieved when x = y = z = 1. -/
theorem max_value_f (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  let f := fun (a b c : ℝ) => (1 - b*c + c) * (1 - c*a + a) * (1 - a*b + b)
  (∀ a b c, a > 0 → b > 0 → c > 0 → a * b * c = 1 → f a b c ≤ 1) ∧ 
  f x y z ≤ 1 ∧
  f 1 1 1 = 1 :=
sorry

end max_value_f_l2443_244382


namespace cosine_product_square_root_l2443_244300

theorem cosine_product_square_root : 
  Real.sqrt ((2 - Real.cos (π / 9) ^ 2) * (2 - Real.cos (2 * π / 9) ^ 2) * (2 - Real.cos (3 * π / 9) ^ 2)) = Real.sqrt 377 / 8 := by
  sorry

end cosine_product_square_root_l2443_244300


namespace unique_square_divisible_by_nine_between_90_and_200_l2443_244365

theorem unique_square_divisible_by_nine_between_90_and_200 :
  ∃! y : ℕ, 
    90 < y ∧ 
    y < 200 ∧ 
    ∃ n : ℕ, y = n^2 ∧ 
    ∃ k : ℕ, y = 9 * k :=
by
  -- The proof would go here
  sorry

end unique_square_divisible_by_nine_between_90_and_200_l2443_244365


namespace relay_race_arrangements_l2443_244309

/-- The number of students to choose from -/
def total_students : ℕ := 10

/-- The number of legs in the relay race -/
def race_legs : ℕ := 4

/-- Function to calculate permutations -/
def permutations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

/-- The main theorem -/
theorem relay_race_arrangements :
  permutations total_students race_legs
  - permutations (total_students - 1) (race_legs - 1)  -- A not in first leg
  - permutations (total_students - 1) (race_legs - 1)  -- B not in last leg
  + permutations (total_students - 2) (race_legs - 2)  -- Neither A in first nor B in last
  = 4008 := by
  sorry

end relay_race_arrangements_l2443_244309


namespace floor_sum_abcd_l2443_244385

theorem floor_sum_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + b^2 = 2500) (h2 : c^2 + d^2 = 2500) (h3 : a*c + b*d = 1500) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end floor_sum_abcd_l2443_244385


namespace power_multiplication_simplification_l2443_244397

theorem power_multiplication_simplification :
  let a : ℝ := 0.25
  let b : ℝ := -4
  let n : ℕ := 16
  let m : ℕ := 17
  (a ^ n) * (b ^ m) = -4 := by
  sorry

end power_multiplication_simplification_l2443_244397


namespace tournament_distributions_l2443_244386

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

theorem tournament_distributions :
  total_distributions = 32 :=
sorry

end tournament_distributions_l2443_244386


namespace vector_sum_magnitude_l2443_244353

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  let angle := angle_between_vectors a b
  let a_magnitude := Real.sqrt (a.1^2 + a.2^2)
  let b_magnitude := Real.sqrt (b.1^2 + b.2^2)
  angle = π/3 ∧ a = (2, 0) ∧ b_magnitude = 1 →
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l2443_244353


namespace cube_projection_sum_squares_zero_l2443_244383

/-- Represents a vertex of a cube -/
structure CubeVertex where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the orthogonal projection of a cube vertex onto a complex plane -/
def project (v : CubeVertex) : ℂ :=
  Complex.mk v.x v.y

/-- Given four vertices of a cube where three are adjacent to the fourth,
    and their orthogonal projections onto a complex plane,
    the sum of the squares of the projected complex numbers is zero. -/
theorem cube_projection_sum_squares_zero
  (V V₁ V₂ V₃ : CubeVertex)
  (adj₁ : V₁.x = V.x ∨ V₁.y = V.y ∨ V₁.z = V.z)
  (adj₂ : V₂.x = V.x ∨ V₂.y = V.y ∨ V₂.z = V.z)
  (adj₃ : V₃.x = V.x ∨ V₃.y = V.y ∨ V₃.z = V.z)
  (origin_proj : project V = 0)
  : (project V₁)^2 + (project V₂)^2 + (project V₃)^2 = 0 := by
  sorry

end cube_projection_sum_squares_zero_l2443_244383


namespace sum_2x_2y_l2443_244310

theorem sum_2x_2y (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2*x + 2*y = 8/3 := by
  sorry

end sum_2x_2y_l2443_244310


namespace meatballs_cost_is_five_l2443_244338

/-- A dinner consisting of pasta, sauce, and meatballs -/
structure Dinner where
  total_cost : ℝ
  pasta_cost : ℝ
  sauce_cost : ℝ
  meatballs_cost : ℝ

/-- The cost of the dinner components add up to the total cost -/
def cost_sum (d : Dinner) : Prop :=
  d.total_cost = d.pasta_cost + d.sauce_cost + d.meatballs_cost

/-- Theorem: Given the total cost, pasta cost, and sauce cost, 
    prove that the meatballs cost $5 -/
theorem meatballs_cost_is_five (d : Dinner) 
  (h1 : d.total_cost = 8)
  (h2 : d.pasta_cost = 1)
  (h3 : d.sauce_cost = 2)
  (h4 : cost_sum d) : 
  d.meatballs_cost = 5 := by
  sorry


end meatballs_cost_is_five_l2443_244338


namespace box_percentage_difference_l2443_244375

theorem box_percentage_difference
  (stan_boxes : ℕ)
  (john_boxes : ℕ)
  (jules_boxes : ℕ)
  (joseph_boxes : ℕ)
  (h1 : stan_boxes = 100)
  (h2 : john_boxes = 30)
  (h3 : john_boxes = jules_boxes + jules_boxes / 5)
  (h4 : jules_boxes = joseph_boxes + 5) :
  (stan_boxes - joseph_boxes) / stan_boxes = 4/5 :=
sorry

end box_percentage_difference_l2443_244375


namespace apples_left_l2443_244331

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- The number of apples Keith ate -/
def keith_apples : ℝ := 6.0

/-- Theorem: The number of apples left after Mike and Nancy picked apples and Keith ate some -/
theorem apples_left : mike_apples + nancy_apples - keith_apples = 4.0 := by
  sorry

end apples_left_l2443_244331


namespace rectangle_max_area_l2443_244367

theorem rectangle_max_area (perimeter : ℝ) (h_perimeter : perimeter = 40) :
  let short_side := perimeter / 6
  let long_side := 2 * short_side
  let area := short_side * long_side
  area = 800 / 9 := by sorry

end rectangle_max_area_l2443_244367


namespace calculate_brokerage_percentage_l2443_244370

/-- Calculate the brokerage percentage for a stock investment --/
theorem calculate_brokerage_percentage
  (stock_rate : ℝ)
  (income : ℝ)
  (investment : ℝ)
  (market_value : ℝ)
  (h1 : stock_rate = 10.5)
  (h2 : income = 756)
  (h3 : investment = 8000)
  (h4 : market_value = 110.86111111111111)
  : ∃ (brokerage_percentage : ℝ),
    brokerage_percentage = 0.225 ∧
    brokerage_percentage = (investment - (income * 100 / stock_rate) * market_value / 100) / investment * 100 :=
by sorry

end calculate_brokerage_percentage_l2443_244370
