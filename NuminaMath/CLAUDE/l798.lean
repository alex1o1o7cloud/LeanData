import Mathlib

namespace sum_of_squared_residuals_l798_79884

theorem sum_of_squared_residuals 
  (total_sum_squared_deviations : ℝ) 
  (correlation_coefficient : ℝ) 
  (h1 : total_sum_squared_deviations = 100) 
  (h2 : correlation_coefficient = 0.818) : 
  total_sum_squared_deviations * (1 - correlation_coefficient ^ 2) = 33.0876 := by
  sorry

end sum_of_squared_residuals_l798_79884


namespace cookie_store_spending_l798_79869

theorem cookie_store_spending (ben david : ℝ) 
  (h1 : david = ben / 2)
  (h2 : ben = david + 20) : 
  ben + david = 60 := by
sorry

end cookie_store_spending_l798_79869


namespace scientific_notation_of_57277000_l798_79891

theorem scientific_notation_of_57277000 :
  (57277000 : ℝ) = 5.7277 * (10 : ℝ)^7 := by
  sorry

end scientific_notation_of_57277000_l798_79891


namespace largest_gcd_of_sum_1023_l798_79827

theorem largest_gcd_of_sum_1023 :
  ∃ (c d : ℕ+), c + d = 1023 ∧
  ∀ (a b : ℕ+), a + b = 1023 → Nat.gcd a b ≤ Nat.gcd c d ∧
  Nat.gcd c d = 341 :=
sorry

end largest_gcd_of_sum_1023_l798_79827


namespace janous_inequality_janous_equality_l798_79805

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := by
  sorry

theorem janous_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ y = z ∧ x = 2 * y := by
  sorry

end janous_inequality_janous_equality_l798_79805


namespace alice_additional_spend_l798_79888

/-- Represents the grocery store cart with various items and their prices -/
structure GroceryCart where
  chicken : Float
  lettuce : Float
  cherryTomatoes : Float
  sweetPotatoes : Float
  broccoli : Float
  brusselSprouts : Float
  strawberries : Float
  cereal : Float
  groundBeef : Float

/-- Calculates the pre-tax total of the grocery cart -/
def calculatePreTaxTotal (cart : GroceryCart) : Float :=
  cart.chicken + cart.lettuce + cart.cherryTomatoes + cart.sweetPotatoes +
  cart.broccoli + cart.brusselSprouts + cart.strawberries + cart.cereal +
  cart.groundBeef

/-- Theorem: The difference between the minimum spend for free delivery and
    Alice's pre-tax total is $3.02 -/
theorem alice_additional_spend (minSpend : Float) (cart : GroceryCart)
    (h1 : minSpend = 50.00)
    (h2 : cart.chicken = 10.80)
    (h3 : cart.lettuce = 3.50)
    (h4 : cart.cherryTomatoes = 5.00)
    (h5 : cart.sweetPotatoes = 3.75)
    (h6 : cart.broccoli = 6.00)
    (h7 : cart.brusselSprouts = 2.50)
    (h8 : cart.strawberries = 4.80)
    (h9 : cart.cereal = 4.00)
    (h10 : cart.groundBeef = 5.63) :
    minSpend - calculatePreTaxTotal cart = 3.02 := by
  sorry

end alice_additional_spend_l798_79888


namespace power_equality_l798_79808

theorem power_equality : 32^2 * 4^4 = 2^18 := by
  sorry

end power_equality_l798_79808


namespace tan_75_degrees_l798_79826

theorem tan_75_degrees (h1 : 75 = 60 + 15) 
                        (h2 : Real.tan (60 * π / 180) = Real.sqrt 3) 
                        (h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) : 
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by sorry

end tan_75_degrees_l798_79826


namespace floor_plus_self_unique_solution_l798_79822

theorem floor_plus_self_unique_solution : 
  ∃! r : ℝ, (⌊r⌋ : ℝ) + r = 18.75 := by sorry

end floor_plus_self_unique_solution_l798_79822


namespace age_of_B_l798_79854

/-- Given the initial ratio of ages and the ratio after 2 years, prove B's age is 6 years -/
theorem age_of_B (k : ℚ) (x : ℚ) : 
  (5 * k : ℚ) / (3 * k : ℚ) = 5 / 3 →
  (4 * k : ℚ) / (3 * k : ℚ) = 4 / 3 →
  ((5 * k + 2) : ℚ) / ((3 * k + 2) : ℚ) = 3 / 2 →
  ((3 * k + 2) : ℚ) / ((2 * k + 2) : ℚ) = 2 / x →
  (3 * k : ℚ) = 6 := by
  sorry

#check age_of_B

end age_of_B_l798_79854


namespace num_non_mult_6_divisors_l798_79840

/-- The smallest integer satisfying the given conditions -/
def m : ℕ :=
  2^3 * 3^4 * 5^6

/-- m/2 is a perfect square -/
axiom m_div_2_is_square : ∃ k : ℕ, m / 2 = k^2

/-- m/3 is a perfect cube -/
axiom m_div_3_is_cube : ∃ k : ℕ, m / 3 = k^3

/-- m/5 is a perfect fifth -/
axiom m_div_5_is_fifth : ∃ k : ℕ, m / 5 = k^5

/-- The number of divisors of m -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The number of divisors of m that are multiples of 6 -/
def num_divisors_mult_6 (n : ℕ) : ℕ :=
  (Finset.filter (λ x => x ∣ n ∧ 6 ∣ x) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem num_non_mult_6_divisors :
    num_divisors m - num_divisors_mult_6 m = 56 := by
  sorry

end num_non_mult_6_divisors_l798_79840


namespace negation_equivalence_l798_79838

theorem negation_equivalence :
  ¬(∀ (x : ℝ), ∃ (n : ℕ+), (n : ℝ) ≥ x) ↔ 
  ∃ (x : ℝ), ∀ (n : ℕ+), (n : ℝ) < x^2 :=
sorry

end negation_equivalence_l798_79838


namespace two_numbers_difference_l798_79829

theorem two_numbers_difference (a b : ℕ) : 
  a + b = 23976 →
  b % 8 = 0 →
  a = b - b / 8 →
  b - a = 1598 := by
sorry

end two_numbers_difference_l798_79829


namespace y_increase_for_x_increase_l798_79871

/-- Given a line with the following properties:
    1. When x increases by 2 units, y increases by 5 units.
    2. The line passes through the point (1, 1).
    3. We consider an x-value increase of 8 units.
    
    This theorem proves that the y-value will increase by 20 units. -/
theorem y_increase_for_x_increase (slope : ℚ) (x_increase y_increase : ℚ) :
  slope = 5 / 2 →
  x_increase = 8 →
  y_increase = slope * x_increase →
  y_increase = 20 := by
  sorry

end y_increase_for_x_increase_l798_79871


namespace flower_shop_utilities_percentage_l798_79863

/-- Calculates the percentage of rent paid for utilities in James' flower shop --/
theorem flower_shop_utilities_percentage
  (weekly_rent : ℝ)
  (store_hours_per_day : ℝ)
  (store_days_per_week : ℝ)
  (employees_per_shift : ℝ)
  (employee_hourly_wage : ℝ)
  (total_weekly_expenses : ℝ)
  (h1 : weekly_rent = 1200)
  (h2 : store_hours_per_day = 16)
  (h3 : store_days_per_week = 5)
  (h4 : employees_per_shift = 2)
  (h5 : employee_hourly_wage = 12.5)
  (h6 : total_weekly_expenses = 3440)
  : (((total_weekly_expenses - (store_hours_per_day * store_days_per_week * employees_per_shift * employee_hourly_wage)) - weekly_rent) / weekly_rent) * 100 = 20 := by
  sorry

end flower_shop_utilities_percentage_l798_79863


namespace percentage_materialB_in_final_mixture_l798_79874

/-- Represents a mixture of oil and material B -/
structure Mixture where
  total : ℝ
  oil : ℝ
  materialB : ℝ

/-- The initial mixture A -/
def initialMixtureA : Mixture :=
  { total := 8
    oil := 8 * 0.2
    materialB := 8 * 0.8 }

/-- The mixture after adding 2 kg of oil -/
def mixtureAfterOil : Mixture :=
  { total := initialMixtureA.total + 2
    oil := initialMixtureA.oil + 2
    materialB := initialMixtureA.materialB }

/-- The additional 6 kg of mixture A -/
def additionalMixtureA : Mixture :=
  { total := 6
    oil := 6 * 0.2
    materialB := 6 * 0.8 }

/-- The final mixture -/
def finalMixture : Mixture :=
  { total := mixtureAfterOil.total + additionalMixtureA.total
    oil := mixtureAfterOil.oil + additionalMixtureA.oil
    materialB := mixtureAfterOil.materialB + additionalMixtureA.materialB }

theorem percentage_materialB_in_final_mixture :
  finalMixture.materialB / finalMixture.total = 0.7 := by
  sorry

end percentage_materialB_in_final_mixture_l798_79874


namespace max_n_value_l798_79862

theorem max_n_value (a b c d : ℝ) (n : ℕ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (h4 : (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ n / (a - d)) :
  n ≤ 9 :=
sorry

end max_n_value_l798_79862


namespace circle_inequality_l798_79828

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y - 1)^2 = 1 → x + y + c ≥ 0) ↔ c ≥ Real.sqrt 2 - 1 := by
  sorry

end circle_inequality_l798_79828


namespace six_digit_divisibility_l798_79897

theorem six_digit_divisibility (abc : Nat) (h : abc ≥ 100 ∧ abc < 1000) :
  let abcabc := abc * 1000 + abc
  (abcabc % 11 = 0) ∧ (abcabc % 13 = 0) ∧ (abcabc % 1001 = 0) ∧
  ∃ x : Nat, x ≥ 100 ∧ x < 1000 ∧ (x * 1000 + x) % 101 ≠ 0 :=
by sorry

end six_digit_divisibility_l798_79897


namespace walking_distance_problem_l798_79866

theorem walking_distance_problem (x t d : ℝ) 
  (h1 : d = (x + 1) * (3/4 * t))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 18 := by
  sorry

end walking_distance_problem_l798_79866


namespace max_value_inequality_l798_79867

theorem max_value_inequality (x y z : ℝ) :
  ∃ (A : ℝ), A > 0 ∧ 
  (∀ (B : ℝ), B > A → 
    ∃ (a b c : ℝ), a^4 + b^4 + c^4 + a^2*b*c + a*b^2*c + a*b*c^2 - B*(a*b + b*c + c*a)^2 < 0) ∧
  (x^4 + y^4 + z^4 + x^2*y*z + x*y^2*z + x*y*z^2 - A*(x*y + y*z + z*x)^2 ≥ 0) ∧
  A = 2/3 :=
sorry

end max_value_inequality_l798_79867


namespace power_seven_700_mod_100_l798_79877

theorem power_seven_700_mod_100 : 7^700 % 100 = 1 := by
  sorry

end power_seven_700_mod_100_l798_79877


namespace total_rulers_problem_solution_l798_79800

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of these two numbers. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = 
    initial_rulers + added_rulers := by sorry

/-- The specific case for the problem -/
theorem problem_solution : 
  11 + 14 = 25 := by sorry

end total_rulers_problem_solution_l798_79800


namespace product_of_digits_not_divisible_by_five_l798_79813

def numbers : List Nat := [4825, 4835, 4845, 4855, 4865]

def is_divisible_by_five (n : Nat) : Prop :=
  n % 5 = 0

def units_digit (n : Nat) : Nat :=
  n % 10

def tens_digit (n : Nat) : Nat :=
  (n / 10) % 10

theorem product_of_digits_not_divisible_by_five :
  ∃ n ∈ numbers,
    ¬is_divisible_by_five n ∧
    ∀ m ∈ numbers, m ≠ n → is_divisible_by_five m ∧
    units_digit n * tens_digit n = 30 :=
  sorry

end product_of_digits_not_divisible_by_five_l798_79813


namespace equation_solution_l798_79841

theorem equation_solution (x : ℚ) : 5 * x + 3 = 2 * x - 4 → 3 * (x^2 + 6) = 103 / 3 := by
  sorry

end equation_solution_l798_79841


namespace complex_magnitude_l798_79847

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I)^2 = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l798_79847


namespace joan_apples_l798_79832

/-- The number of apples Joan picked -/
def apples_picked : ℕ := 43

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_remaining : ℕ := apples_picked - apples_given

theorem joan_apples : apples_remaining = 16 := by
  sorry

end joan_apples_l798_79832


namespace unique_solution_modular_equation_l798_79817

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 72 % 103 ∧ n = 52 := by
  sorry

end unique_solution_modular_equation_l798_79817


namespace james_has_winning_strategy_l798_79803

/-- Represents a player in the coin-choosing game -/
inductive Player : Type
| John : Player
| James : Player

/-- The state of the game at any point -/
structure GameState :=
  (coins_left : List ℕ)
  (john_kopeks : ℕ)
  (james_kopeks : ℕ)
  (current_chooser : Player)

/-- A strategy is a function that takes the current game state and returns the chosen coin -/
def Strategy := GameState → ℕ

/-- The result of the game -/
inductive GameResult
| JohnWins : GameResult
| JamesWins : GameResult
| Draw : GameResult

/-- Play the game given strategies for both players -/
def play_game (john_strategy : Strategy) (james_strategy : Strategy) : GameResult :=
  sorry

/-- A winning strategy for a player ensures they always win or draw -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop :=
  match player with
  | Player.John => ∀ james_strategy, play_game strategy james_strategy ≠ GameResult.JamesWins
  | Player.James => ∀ john_strategy, play_game john_strategy strategy ≠ GameResult.JohnWins

/-- The main theorem: James has a winning strategy -/
theorem james_has_winning_strategy :
  ∃ (strategy : Strategy), is_winning_strategy Player.James strategy :=
sorry

end james_has_winning_strategy_l798_79803


namespace days_worked_by_c_l798_79810

/-- The number of days worked by person a -/
def days_a : ℕ := 6

/-- The number of days worked by person b -/
def days_b : ℕ := 9

/-- The daily wage of person c in rupees -/
def wage_c : ℕ := 115

/-- The total earnings of all three persons in rupees -/
def total_earnings : ℕ := 1702

/-- The ratio of daily wages for persons a, b, and c -/
def wage_ratio : Fin 3 → ℕ
| 0 => 3
| 1 => 4
| 2 => 5

/-- Theorem stating that person c worked for 4 days -/
theorem days_worked_by_c : 
  ∃ (days_c : ℕ), 
    days_c * wage_c + 
    days_a * (wage_ratio 0 * wage_c / wage_ratio 2) + 
    days_b * (wage_ratio 1 * wage_c / wage_ratio 2) = 
    total_earnings ∧ days_c = 4 := by
  sorry

end days_worked_by_c_l798_79810


namespace solve_for_b_l798_79860

-- Define the functions p and q
def p (x : ℝ) := 3 * x + 5
def q (x b : ℝ) := 4 * x - b

-- State the theorem
theorem solve_for_b :
  ∀ b : ℝ, p (q 3 b) = 29 → b = 4 := by
  sorry

end solve_for_b_l798_79860


namespace hundred_power_ten_as_sum_of_tens_l798_79858

theorem hundred_power_ten_as_sum_of_tens (n : ℕ) : (100 ^ 10) = n * 10 → n = 10 ^ 19 := by
  sorry

end hundred_power_ten_as_sum_of_tens_l798_79858


namespace fruit_arrangement_count_l798_79821

-- Define the number of each type of fruit
def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 3

-- Define the total number of fruits
def total_fruits : ℕ := num_apples + num_oranges + num_bananas

-- Theorem statement
theorem fruit_arrangement_count : 
  (Nat.factorial total_fruits) / (Nat.factorial num_apples * Nat.factorial num_oranges * Nat.factorial num_bananas) = 1260 := by
  sorry

end fruit_arrangement_count_l798_79821


namespace simplest_form_fraction_l798_79885

/-- A fraction is in simplest form if its numerator and denominator have no common factors
    other than 1 and -1, and neither the numerator nor denominator can be factored further. -/
def IsSimplestForm (n d : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, (n x y ≠ 0 ∨ d x y ≠ 0) →
    ∀ f : ℝ → ℝ → ℝ, (f x y ∣ n x y) ∧ (f x y ∣ d x y) → f x y = 1 ∨ f x y = -1

/-- The fraction (x^2 + y^2) / (x + y) is in simplest form. -/
theorem simplest_form_fraction (x y : ℝ) :
    IsSimplestForm (fun x y => x^2 + y^2) (fun x y => x + y) := by
  sorry

#check simplest_form_fraction

end simplest_form_fraction_l798_79885


namespace hyperbola_iff_m_in_range_l798_79886

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents a hyperbola -/
def m_range : Set ℝ := {m | -2 < m ∧ m < -1}

/-- Theorem: The equation represents a hyperbola if and only if m is in the range (-2, -1) -/
theorem hyperbola_iff_m_in_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ m_range :=
by sorry

end hyperbola_iff_m_in_range_l798_79886


namespace max_m_value_l798_79850

theorem max_m_value (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ (x^2 - 4*x - 5 > 0)) →
  (∀ x, q x ↔ (x^2 - 2*x + 1 - m^2 > 0)) →
  (m > 0) →
  (∀ x, p x → q x) →
  (∃ x, q x ∧ ¬(p x)) →
  (∀ m' > m, ∃ x, p x ∧ ¬(q x)) →
  m = 2 :=
by sorry

end max_m_value_l798_79850


namespace lcm_gcd_48_180_l798_79806

theorem lcm_gcd_48_180 : 
  (Nat.lcm 48 180 = 720) ∧ (Nat.gcd 48 180 = 12) := by
  sorry

end lcm_gcd_48_180_l798_79806


namespace equal_opposite_angles_imag_prod_zero_l798_79894

/-- Given complex numbers a, b, c, d where the angles a 0 b and c 0 d are equal and oppositely oriented,
    the imaginary part of their product abcd is zero. -/
theorem equal_opposite_angles_imag_prod_zero
  (a b c d : ℂ)
  (h : ∃ (θ : ℝ), (b / a).arg = θ ∧ (d / c).arg = -θ) :
  (a * b * c * d).im = 0 := by
  sorry

end equal_opposite_angles_imag_prod_zero_l798_79894


namespace childless_count_bertha_l798_79846

structure Family :=
  (daughters : ℕ)
  (total_descendants : ℕ)
  (grandchildren_per_daughter : ℕ)

def childless_count (f : Family) : ℕ :=
  f.total_descendants - f.daughters

theorem childless_count_bertha (f : Family) 
  (h1 : f.daughters = 8)
  (h2 : f.total_descendants = 40)
  (h3 : f.grandchildren_per_daughter = 4)
  (h4 : f.total_descendants = f.daughters + f.daughters * f.grandchildren_per_daughter) :
  childless_count f = 32 := by
  sorry


end childless_count_bertha_l798_79846


namespace quartet_performances_theorem_l798_79861

/-- Represents the number of performances for each friend -/
structure Performances where
  sarah : ℕ
  lily : ℕ
  emma : ℕ
  nora : ℕ
  kate : ℕ

/-- The total number of quartet performances -/
def total_performances (p : Performances) : ℕ :=
  (p.sarah + p.lily + p.emma + p.nora + p.kate) / 4

theorem quartet_performances_theorem (p : Performances) :
  p.nora = 10 →
  p.sarah = 6 →
  p.lily > 6 →
  p.emma > 6 →
  p.kate > 6 →
  p.lily < 10 →
  p.emma < 10 →
  p.kate < 10 →
  (p.sarah + p.lily + p.emma + p.nora + p.kate) % 4 = 0 →
  total_performances p = 10 := by
  sorry

#check quartet_performances_theorem

end quartet_performances_theorem_l798_79861


namespace tiles_per_row_l798_79820

-- Define the room area in square feet
def room_area : ℝ := 144

-- Define the tile size in inches
def tile_size : ℝ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℝ := 12

-- Theorem to prove
theorem tiles_per_row : 
  ⌊(inches_per_foot * (room_area ^ (1/2 : ℝ))) / tile_size⌋ = 18 := by
  sorry

end tiles_per_row_l798_79820


namespace sin_plus_cos_value_l798_79898

theorem sin_plus_cos_value (α : Real) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α * Real.cos α = 1 / 8) : 
  Real.sin α + Real.cos α = Real.sqrt 5 / 2 := by
sorry

end sin_plus_cos_value_l798_79898


namespace intersection_of_A_and_B_l798_79831

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 1)}
def set_B : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 - 1)}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ici 1 := by sorry

end intersection_of_A_and_B_l798_79831


namespace quadratic_equation_result_l798_79802

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 3*x = 4) : 3*x^2 - 9*x + 8 = 20 := by
  sorry

end quadratic_equation_result_l798_79802


namespace perimeter_of_triangle_cos_A_minus_C_l798_79819

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 1 ∧ b = 2 ∧ Real.cos C = 1/4

-- Theorem for the perimeter
theorem perimeter_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : a + b + c = 5 := by
  sorry

-- Theorem for cos(A-C)
theorem cos_A_minus_C (a b c : ℝ) (A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : Real.cos (A - C) = 11/16 := by
  sorry

end perimeter_of_triangle_cos_A_minus_C_l798_79819


namespace largest_number_with_equal_quotient_and_remainder_l798_79833

theorem largest_number_with_equal_quotient_and_remainder :
  ∀ (A B C : ℕ),
    (A = 7 * B + C) →
    (B = C) →
    (C < 7) →
    A ≤ 48 :=
by
  sorry

end largest_number_with_equal_quotient_and_remainder_l798_79833


namespace euler_formula_quadrant_l798_79851

theorem euler_formula_quadrant :
  let θ : ℝ := 2 * Real.pi / 3
  let z : ℂ := Complex.exp (Complex.I * θ)
  z.re < 0 ∧ z.im > 0 :=
by sorry

end euler_formula_quadrant_l798_79851


namespace one_eighth_of_number_l798_79859

theorem one_eighth_of_number (n : ℚ) (h : 6/11 * n = 48) : 1/8 * n = 11 := by
  sorry

end one_eighth_of_number_l798_79859


namespace dining_bill_share_l798_79845

def total_bill : ℝ := 211.00
def num_people : ℕ := 9
def tip_percentage : ℝ := 0.15

theorem dining_bill_share :
  let tip := total_bill * tip_percentage
  let total_with_tip := total_bill + tip
  let share_per_person := total_with_tip / num_people
  ∃ ε > 0, |share_per_person - 26.96| < ε :=
by sorry

end dining_bill_share_l798_79845


namespace endpoint_of_vector_l798_79835

def vector_a : Fin 3 → ℝ := ![3, -4, 2]
def point_A : Fin 3 → ℝ := ![2, -1, 1]
def point_B : Fin 3 → ℝ := ![5, -5, 3]

theorem endpoint_of_vector (i : Fin 3) : 
  point_B i = point_A i + vector_a i :=
by
  sorry

end endpoint_of_vector_l798_79835


namespace piravena_round_trip_cost_l798_79881

/-- Represents the cost of a journey between two cities -/
structure JourneyCost where
  distance : ℝ
  rate : ℝ
  bookingFee : ℝ := 0

def totalCost (journey : JourneyCost) : ℝ :=
  journey.distance * journey.rate + journey.bookingFee

def roundTripCost (outbound outboundRate inbound inboundRate bookingFee : ℝ) : ℝ :=
  totalCost { distance := outbound, rate := outboundRate, bookingFee := bookingFee } +
  totalCost { distance := inbound, rate := inboundRate }

theorem piravena_round_trip_cost :
  let distanceAB : ℝ := 4000
  let distanceAC : ℝ := 3000
  let busRate : ℝ := 0.20
  let planeRate : ℝ := 0.12
  let planeBookingFee : ℝ := 120
  roundTripCost distanceAB planeRate distanceAB busRate planeBookingFee = 1400 := by
  sorry

end piravena_round_trip_cost_l798_79881


namespace smallest_positive_integer_form_l798_79865

theorem smallest_positive_integer_form (m n : ℤ) : ∃ (k : ℕ), k > 0 ∧ ∃ (a b : ℤ), k = 1237 * a + 78653 * b ∧ ∀ (l : ℕ), l > 0 → ∃ (c d : ℤ), l = 1237 * c + 78653 * d → k ≤ l :=
sorry

end smallest_positive_integer_form_l798_79865


namespace circle_equation_coefficients_l798_79852

theorem circle_equation_coefficients (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔ (x + 2)^2 + (y - 3)^2 = 4^2) →
  D = 4 ∧ E = -6 ∧ F = -3 := by
sorry

end circle_equation_coefficients_l798_79852


namespace complex_number_problem_l798_79816

theorem complex_number_problem (a : ℝ) : 
  let z : ℂ := Complex.I * (2 + a * Complex.I)
  (Complex.re z = -Complex.im z) → a = 2 := by
  sorry

end complex_number_problem_l798_79816


namespace divisible_by_24_l798_79809

theorem divisible_by_24 (n : ℤ) : ∃ k : ℤ, n * (n + 2) * (5 * n - 1) * (5 * n + 1) = 24 * k := by
  sorry

end divisible_by_24_l798_79809


namespace divisors_of_power_minus_one_l798_79823

/-- The number of distinct positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem divisors_of_power_minus_one (a n : ℕ) (ha : a > 1) (hn : n > 0) 
  (h_prime : Nat.Prime (a^n + 1)) : num_divisors (a^n - 1) ≥ n := by
  sorry


end divisors_of_power_minus_one_l798_79823


namespace unanswered_questions_l798_79856

/-- Represents the scoring system and results of a math contest --/
structure ContestScore where
  totalQuestions : ℕ
  oldScore : ℕ
  newScore : ℕ

/-- Proves that the number of unanswered questions is 10 given the contest conditions --/
theorem unanswered_questions (score : ContestScore)
  (h1 : score.totalQuestions = 40)
  (h2 : ∃ c w : ℕ, 25 + 3 * c - w = score.oldScore)
  (h3 : score.oldScore = 95)
  (h4 : ∃ c w u : ℕ, 6 * c - 2 * w + 3 * u = score.newScore)
  (h5 : score.newScore = 120)
  (h6 : ∃ c w u : ℕ, c + w + u = score.totalQuestions) :
  ∃ c w : ℕ, c + w + 10 = score.totalQuestions :=
sorry

end unanswered_questions_l798_79856


namespace min_cubes_for_valid_config_l798_79890

/-- Represents a modified cube with two protruding snaps and four receptacle holes. -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (holes : Fin 4)

/-- Represents a configuration of snapped-together cubes. -/
structure CubeConfiguration :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)

/-- Returns true if all snaps are covered in the given configuration. -/
def all_snaps_covered (config : CubeConfiguration) : Bool :=
  config.all_snaps_covered

/-- The minimum number of cubes required for a valid configuration. -/
def min_cubes : Nat := 6

/-- Theorem stating that the minimum number of cubes for a valid configuration is 6. -/
theorem min_cubes_for_valid_config :
  ∀ (config : CubeConfiguration),
    all_snaps_covered config →
    config.cubes.length ≥ min_cubes :=
  sorry

end min_cubes_for_valid_config_l798_79890


namespace min_value_theorem_l798_79893

theorem min_value_theorem (x y z : ℝ) (h : (1 / x) + (2 / y) + (3 / z) = 1) :
  x + y / 2 + z / 3 ≥ 9 ∧
  (x + y / 2 + z / 3 = 9 ↔ x = y / 2 ∧ y / 2 = z / 3) :=
sorry

end min_value_theorem_l798_79893


namespace chosen_number_l798_79848

theorem chosen_number (x : ℝ) : (x / 12)^2 - 240 = 8 → x = 24 * Real.sqrt 62 := by
  sorry

end chosen_number_l798_79848


namespace complex_sum_equals_negative_two_l798_79876

/-- Given that z = cos(6π/11) + i sin(6π/11), prove that z/(1 + z²) + z²/(1 + z⁴) + z³/(1 + z⁶) = -2 -/
theorem complex_sum_equals_negative_two (z : ℂ) (h : z = Complex.exp (Complex.I * (6 * Real.pi / 11))) :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = -2 := by
  sorry

end complex_sum_equals_negative_two_l798_79876


namespace probability_at_least_one_white_ball_l798_79811

theorem probability_at_least_one_white_ball (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = red_balls + white_balls) (h2 : total_balls = 5) 
  (h3 : red_balls = 3) (h4 : white_balls = 2) (h5 : drawn_balls = 3) : 
  1 - (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 9 / 10 := by
  sorry

end probability_at_least_one_white_ball_l798_79811


namespace positive_A_value_l798_79895

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end positive_A_value_l798_79895


namespace function_values_l798_79870

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def g (x : ℝ) : ℝ := 3 * x^2 + 1

-- State the theorem
theorem function_values :
  f 2 = 9 ∧ f (-2) = 25 ∧ g (-1) = 4 := by
  sorry

end function_values_l798_79870


namespace episode_filming_time_increase_l798_79824

/-- The percentage increase in filming time compared to episode duration -/
theorem episode_filming_time_increase (episode_duration : ℕ) (episodes_per_week : ℕ) (filming_time : ℕ) : 
  episode_duration = 20 →
  episodes_per_week = 5 →
  filming_time = 600 →
  (((filming_time / (episodes_per_week * 4)) - episode_duration) / episode_duration) * 100 = 50 := by
sorry

end episode_filming_time_increase_l798_79824


namespace shanghai_population_aging_l798_79872

/-- Represents a city's demographic characteristics -/
structure CityDemographics where
  location : String
  economy : String
  inMigrationRate : String
  mechanicalGrowthRate : String
  naturalGrowthRate : String

/-- Represents possible population issues -/
inductive PopulationIssue
  | SatelliteTownPopulation
  | PopulationAging
  | LargePopulationBase
  | YoungPopulationStructure

/-- Determines the most significant population issue for a given city -/
def mostSignificantIssue (city : CityDemographics) : PopulationIssue :=
  sorry

/-- Shanghai's demographic characteristics -/
def shanghai : CityDemographics := {
  location := "eastern coast of China",
  economy := "developed",
  inMigrationRate := "high",
  mechanicalGrowthRate := "high",
  naturalGrowthRate := "low"
}

/-- Theorem stating that Shanghai's most significant population issue is aging -/
theorem shanghai_population_aging :
  mostSignificantIssue shanghai = PopulationIssue.PopulationAging :=
  sorry

end shanghai_population_aging_l798_79872


namespace comic_stacking_arrangements_l798_79830

def spiderman_comics : ℕ := 8
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 3

def total_comics : ℕ := spiderman_comics + archie_comics + garfield_comics

def garfield_group_positions : ℕ := 3

theorem comic_stacking_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * garfield_group_positions) = 8669760 := by
  sorry

end comic_stacking_arrangements_l798_79830


namespace max_a_for_monotonous_l798_79875

/-- The function f(x) = -x^3 + ax is monotonous (non-increasing) on [1, +∞) -/
def is_monotonous (a : ℝ) : Prop :=
  ∀ x y, x ≥ 1 → y ≥ 1 → x ≤ y → (-x^3 + a*x) ≥ (-y^3 + a*y)

/-- The maximum value of a for which f(x) = -x^3 + ax is monotonous on [1, +∞) is 3 -/
theorem max_a_for_monotonous : (∃ a_max : ℝ, a_max = 3 ∧ 
  (∀ a : ℝ, is_monotonous a → a ≤ a_max) ∧ 
  is_monotonous a_max) :=
sorry

end max_a_for_monotonous_l798_79875


namespace michelle_initial_ride_fee_l798_79878

/-- A taxi ride with an initial fee and per-mile charge. -/
structure TaxiRide where
  distance : ℝ
  chargePerMile : ℝ
  totalPaid : ℝ

/-- Calculate the initial ride fee for a taxi ride. -/
def initialRideFee (ride : TaxiRide) : ℝ :=
  ride.totalPaid - ride.distance * ride.chargePerMile

/-- Theorem: The initial ride fee for Michelle's taxi ride is $2. -/
theorem michelle_initial_ride_fee :
  let ride : TaxiRide := {
    distance := 4,
    chargePerMile := 2.5,
    totalPaid := 12
  }
  initialRideFee ride = 2 := by
  sorry

end michelle_initial_ride_fee_l798_79878


namespace dinner_bill_proof_l798_79879

theorem dinner_bill_proof (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ) : 
  total_friends = 10 → 
  paying_friends = 9 → 
  extra_payment = 3 → 
  ∃ (bill : ℚ), bill = 270 ∧ 
    paying_friends * (bill / total_friends + extra_payment) = bill :=
by sorry

end dinner_bill_proof_l798_79879


namespace min_points_for_top_two_l798_79804

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scoring_system : List Nat)

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (t : Tournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculates the maximum total points possible in the tournament --/
def max_total_points (t : Tournament) : Nat :=
  (total_matches t) * (t.scoring_system.head!)

/-- Theorem: In a 4-team round-robin tournament with the given scoring system,
    a team needs at least 7 points to guarantee a top-two finish --/
theorem min_points_for_top_two (t : Tournament) 
  (h1 : t.num_teams = 4)
  (h2 : t.scoring_system = [3, 1, 0]) : 
  ∃ (min_points : Nat), 
    (min_points = 7) ∧ 
    (∀ (team_points : Nat), 
      team_points ≥ min_points → 
      (max_total_points t - team_points) / (t.num_teams - 1) < team_points) :=
by sorry

end min_points_for_top_two_l798_79804


namespace cos_75_degrees_l798_79844

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l798_79844


namespace john_needs_more_money_l798_79818

/-- Given that John needs $2.5 in total and has $0.75, prove that he needs $1.75 more. -/
theorem john_needs_more_money (total_needed : ℝ) (amount_has : ℝ) 
  (h1 : total_needed = 2.5)
  (h2 : amount_has = 0.75) :
  total_needed - amount_has = 1.75 := by
sorry

end john_needs_more_money_l798_79818


namespace midpoint_ratio_range_l798_79887

/-- Given two lines and a point M that is the midpoint of two points on these lines,
    prove that the ratio of y₀/x₀ falls within a specific range. -/
theorem midpoint_ratio_range (P Q : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + 2 * P.2 - 1 = 0) →  -- P is on the line x + 2y - 1 = 0
  (Q.1 + 2 * Q.2 + 3 = 0) →  -- Q is on the line x + 2y + 3 = 0
  ((x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M(x₀, y₀) is the midpoint of PQ
  (y₀ > x₀ + 2) →  -- Given condition
  (-1/2 < y₀ / x₀) ∧ (y₀ / x₀ < -1/5) :=  -- The range of y₀/x₀
by sorry

end midpoint_ratio_range_l798_79887


namespace wire_cut_ratio_l798_79864

theorem wire_cut_ratio (p q : ℝ) (h : p > 0 ∧ q > 0) : 
  (p^2 / 16 = π * (q / (2 * π))^2) → p / q = 4 / Real.sqrt π := by
  sorry

end wire_cut_ratio_l798_79864


namespace square_sum_value_l798_79825

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 := by
  sorry

end square_sum_value_l798_79825


namespace card_distribution_l798_79857

theorem card_distribution (n : ℕ) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => Nat.choose n (k + 1))) = 2 * (2^(n - 1) - 1) :=
by sorry

end card_distribution_l798_79857


namespace pool_cannot_be_filled_problem_pool_cannot_be_filled_l798_79892

/-- Represents the state of a pool being filled -/
structure PoolFilling where
  capacity : ℝ
  num_hoses : ℕ
  flow_rate_per_hose : ℝ
  leakage_rate : ℝ

/-- Determines if a pool can be filled given its filling conditions -/
def can_be_filled (p : PoolFilling) : Prop :=
  p.num_hoses * p.flow_rate_per_hose > p.leakage_rate

/-- Theorem stating that a pool cannot be filled if inflow rate equals leakage rate -/
theorem pool_cannot_be_filled (p : PoolFilling) 
  (h : p.num_hoses * p.flow_rate_per_hose = p.leakage_rate) : 
  ¬(can_be_filled p) := by
  sorry

/-- The specific pool problem instance -/
def problem_pool : PoolFilling := {
  capacity := 48000
  num_hoses := 6
  flow_rate_per_hose := 3
  leakage_rate := 18
}

/-- Theorem for the specific problem instance -/
theorem problem_pool_cannot_be_filled : 
  ¬(can_be_filled problem_pool) := by
  sorry

end pool_cannot_be_filled_problem_pool_cannot_be_filled_l798_79892


namespace shirt_discount_percentage_l798_79815

theorem shirt_discount_percentage (original_price discounted_price : ℝ) 
  (h1 : original_price = 80)
  (h2 : discounted_price = 68) :
  (original_price - discounted_price) / original_price * 100 = 15 := by
  sorry

end shirt_discount_percentage_l798_79815


namespace hexagon_angle_sum_l798_79801

theorem hexagon_angle_sum (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → 
  34 + 80 + 30 + 90 + x + y = 720 → 
  x + y = 36 := by
sorry

end hexagon_angle_sum_l798_79801


namespace q_value_for_p_seven_l798_79868

/-- Given the equation Q = 3rP - 6, where r is a constant, prove that if Q = 27 when P = 5, then Q = 40 when P = 7 -/
theorem q_value_for_p_seven (r : ℝ) : 
  (∃ Q : ℝ, Q = 3 * r * 5 - 6 ∧ Q = 27) →
  (∃ Q : ℝ, Q = 3 * r * 7 - 6 ∧ Q = 40) :=
by sorry

end q_value_for_p_seven_l798_79868


namespace not_divisible_by_61_l798_79812

theorem not_divisible_by_61 (x y : ℕ) 
  (h1 : ¬(61 ∣ x))
  (h2 : ¬(61 ∣ y))
  (h3 : 61 ∣ (7*x + 34*y)) :
  ¬(61 ∣ (5*x + 16*y)) := by
sorry

end not_divisible_by_61_l798_79812


namespace negative_a_cubed_times_a_squared_l798_79882

theorem negative_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end negative_a_cubed_times_a_squared_l798_79882


namespace polynomial_not_equal_77_l798_79855

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end polynomial_not_equal_77_l798_79855


namespace homework_problem_distribution_l798_79837

theorem homework_problem_distribution (total : ℕ) (finished : ℕ) (pages : ℕ) 
  (h1 : total = 60) 
  (h2 : finished = 20) 
  (h3 : pages = 5) 
  (h4 : pages > 0) :
  (total - finished) / pages = 8 := by
  sorry

end homework_problem_distribution_l798_79837


namespace function_properties_l798_79899

def f (x : ℝ) : ℝ := |2*x + 2| - 5

def g (m : ℝ) (x : ℝ) : ℝ := f x + |x - m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (∀ x, f x - |x - 1| ≥ 0 ↔ x ∈ Set.Iic (-8) ∪ Set.Ici 2) ∧
  (∃ a b c : ℝ, a < b ∧ b < c ∧
    (∀ x, x < a → g m x < 0) ∧
    (∀ x, a < x ∧ x < c → g m x > 0) ∧
    g m a = 0 ∧ g m c = 0) ↔
  3/2 ≤ m ∧ m < 4 :=
sorry

end function_properties_l798_79899


namespace rational_root_of_cubic_l798_79889

theorem rational_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 - 4*x^2 + b*x + c = 0 ∧ x = 4 - Real.sqrt 11) →
  (∃ y : ℚ, y^3 - 4*y^2 + b*y + c = 0) →
  (∃ z : ℚ, z^3 - 4*z^2 + b*z + c = 0 ∧ z = -4) :=
by sorry

end rational_root_of_cubic_l798_79889


namespace cost_formula_l798_79883

def cost (P : ℕ) : ℕ :=
  15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0)

theorem cost_formula (P : ℕ) :
  cost P = 15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0) :=
by sorry

end cost_formula_l798_79883


namespace aq_length_is_112_over_35_l798_79807

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an inscribed right triangle within another triangle -/
structure InscribedRightTriangle where
  outer : Triangle
  pc : ℝ
  bp : ℝ
  cq : ℝ

/-- The length of AQ in the described configuration -/
def aq_length (t : InscribedRightTriangle) : ℝ :=
  -- Definition of aq_length goes here
  sorry

/-- Theorem stating that AQ = 112/35 in the given configuration -/
theorem aq_length_is_112_over_35 :
  let t : InscribedRightTriangle := {
    outer := { a := 6, b := 7, c := 8 },
    pc := 4,
    bp := 3,
    cq := 3
  }
  aq_length t = 112 / 35 := by sorry

end aq_length_is_112_over_35_l798_79807


namespace brad_speed_is_6_l798_79814

-- Define the given conditions
def maxwell_speed : ℝ := 4
def brad_delay : ℝ := 1
def total_distance : ℝ := 34
def meeting_time : ℝ := 4

-- Define Brad's speed as a variable
def brad_speed : ℝ := sorry

-- Theorem to prove
theorem brad_speed_is_6 : brad_speed = 6 := by
  -- The proof goes here
  sorry

end brad_speed_is_6_l798_79814


namespace base3_to_base10_conversion_l798_79873

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 1, 0, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Number = 178 := by
  sorry

end base3_to_base10_conversion_l798_79873


namespace final_top_number_is_16_l798_79836

/-- Represents the state of the paper after folding operations -/
structure PaperState :=
  (top_number : Nat)

/-- Represents a folding operation -/
inductive FoldOperation
  | FoldBottomUp
  | FoldTopDown
  | FoldLeftRight

/-- The initial configuration of the paper -/
def initial_paper : List (List Nat) :=
  [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

/-- Perform a single fold operation -/
def fold (state : PaperState) (op : FoldOperation) : PaperState :=
  match op with
  | FoldOperation.FoldBottomUp => { top_number := 15 }
  | FoldOperation.FoldTopDown => { top_number := 9 }
  | FoldOperation.FoldLeftRight => { top_number := state.top_number + 1 }

/-- Perform a sequence of fold operations -/
def fold_sequence (initial : PaperState) (ops : List FoldOperation) : PaperState :=
  ops.foldl fold initial

/-- The theorem to be proved -/
theorem final_top_number_is_16 :
  (fold_sequence { top_number := 1 }
    [FoldOperation.FoldBottomUp,
     FoldOperation.FoldTopDown,
     FoldOperation.FoldBottomUp,
     FoldOperation.FoldLeftRight]).top_number = 16 := by
  sorry


end final_top_number_is_16_l798_79836


namespace tim_pencils_l798_79849

theorem tim_pencils (tyrah_pencils : ℕ) (sarah_pencils : ℕ) (tim_pencils : ℕ)
  (h1 : tyrah_pencils = 6 * sarah_pencils)
  (h2 : tim_pencils = 8 * sarah_pencils)
  (h3 : tyrah_pencils = 12) :
  tim_pencils = 16 := by
sorry

end tim_pencils_l798_79849


namespace fraction_simplification_l798_79853

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 / (a * b) = a / b := by
  sorry

end fraction_simplification_l798_79853


namespace hamburgers_needed_proof_l798_79896

/-- Calculates the number of additional hamburgers needed to reach a target revenue -/
def additional_hamburgers_needed (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ) : ℕ :=
  ((target_revenue - (price_per_hamburger * hamburgers_sold)) + (price_per_hamburger - 1)) / price_per_hamburger

/-- Proves that 4 additional hamburgers are needed to reach $50 given the conditions -/
theorem hamburgers_needed_proof (target_revenue : ℕ) (price_per_hamburger : ℕ) (hamburgers_sold : ℕ)
  (h1 : target_revenue = 50)
  (h2 : price_per_hamburger = 5)
  (h3 : hamburgers_sold = 6) :
  additional_hamburgers_needed target_revenue price_per_hamburger hamburgers_sold = 4 := by
  sorry

end hamburgers_needed_proof_l798_79896


namespace sin_sum_inverse_sin_tan_l798_79839

theorem sin_sum_inverse_sin_tan (x y : ℝ) 
  (hx : x = 4 / 5) (hy : y = 1 / 2) : 
  Real.sin (Real.arcsin x + Real.arctan y) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sin_sum_inverse_sin_tan_l798_79839


namespace quadratic_inequality_equivalence_l798_79834

theorem quadratic_inequality_equivalence :
  ∃ d : ℝ, ∀ x : ℝ, x * (2 * x + 4) < d ↔ x ∈ Set.Ioo (-4) 1 :=
by
  use 8
  sorry

end quadratic_inequality_equivalence_l798_79834


namespace total_tiles_needed_l798_79842

def room_length : ℕ := 12
def room_width : ℕ := 16
def small_tile_size : ℕ := 1
def large_tile_size : ℕ := 2

theorem total_tiles_needed : 
  (room_length * room_width - (room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size)) + 
  ((room_length - 2 * small_tile_size) * (room_width - 2 * small_tile_size) / (large_tile_size * large_tile_size)) = 87 := by
  sorry

end total_tiles_needed_l798_79842


namespace strawberry_jelly_amount_l798_79843

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := sorry

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := 4518

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- Theorem stating that the amount of strawberry jelly is 1792 grams -/
theorem strawberry_jelly_amount : strawberry_jelly = 1792 :=
  by
    sorry

/-- Lemma stating that the sum of strawberry and blueberry jelly equals the total jelly -/
lemma jelly_sum : strawberry_jelly + blueberry_jelly = total_jelly :=
  by
    sorry

end strawberry_jelly_amount_l798_79843


namespace triangle_shape_l798_79880

theorem triangle_shape (a b : ℝ) (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < a ∧ 0 < b) (h_condition : a * Real.cos A = b * Real.cos B) :
  A = B ∨ A + B = π / 2 := by sorry

end triangle_shape_l798_79880
