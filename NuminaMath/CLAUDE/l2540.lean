import Mathlib

namespace system_solution_l2540_254021

theorem system_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = 4) ∧ (x = -10/13) ∧ (y = 17/13) := by
  sorry

end system_solution_l2540_254021


namespace min_stamps_for_60_cents_l2540_254024

/-- Represents the number of ways to make a certain amount using given denominations -/
def numWays (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Represents the minimum number of coins needed to make a certain amount using given denominations -/
def minCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

theorem min_stamps_for_60_cents :
  minCoins 60 [5, 6] = 10 :=
sorry

end min_stamps_for_60_cents_l2540_254024


namespace ratio_a_to_c_l2540_254017

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 5)
  (hdb : d / b = 1 / 9) :
  a / c = 112.5 / 14 := by
  sorry

end ratio_a_to_c_l2540_254017


namespace roots_sum_and_product_l2540_254067

theorem roots_sum_and_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 4 = 0 → 
  x₂^2 - 2*x₂ - 4 = 0 → 
  x₁ + x₂ + x₁*x₂ = -2 :=
by sorry

end roots_sum_and_product_l2540_254067


namespace absolute_value_equation_product_l2540_254045

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| - 7 = 28) ∧ (|5 * x₂| - 7 = 28) ∧ (x₁ ≠ x₂) ∧ (x₁ * x₂ = -49)) := by
  sorry

end absolute_value_equation_product_l2540_254045


namespace school_students_count_l2540_254012

theorem school_students_count (total : ℕ) 
  (chess_ratio : Real) 
  (swimming_ratio : Real) 
  (swimming_count : ℕ) 
  (h1 : chess_ratio = 0.25)
  (h2 : swimming_ratio = 0.50)
  (h3 : swimming_count = 125)
  (h4 : ↑swimming_count = swimming_ratio * (chess_ratio * total)) :
  total = 1000 := by
sorry

end school_students_count_l2540_254012


namespace unique_solution_l2540_254072

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1) ∧
  2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1) ∧
  2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)

/-- The theorem stating that (1, 1, 1) is the unique positive real solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ system x y z ∧ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end unique_solution_l2540_254072


namespace hay_delivery_ratio_l2540_254089

theorem hay_delivery_ratio : 
  let initial_bales : ℕ := 10
  let initial_cost_per_bale : ℕ := 15
  let new_cost_per_bale : ℕ := 18
  let additional_cost : ℕ := 210
  let new_bales : ℕ := (initial_bales * initial_cost_per_bale + additional_cost) / new_cost_per_bale
  (new_bales : ℚ) / initial_bales = 2 := by
  sorry

end hay_delivery_ratio_l2540_254089


namespace complex_i_plus_i_squared_l2540_254032

theorem complex_i_plus_i_squared : ∃ (i : ℂ), i * i = -1 ∧ i + i * i = -1 + i := by sorry

end complex_i_plus_i_squared_l2540_254032


namespace fraction_decimal_conversions_l2540_254088

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(round (q * 10^n)) / 10^n)

theorem fraction_decimal_conversions :
  -- 1. 60/4 = 15 in both fraction and decimal form
  (60 : ℚ) / 4 = 15 ∧ 
  -- 2. 19/6 ≈ 3.167 when rounded to three decimal places
  round_to_decimal_places ((19 : ℚ) / 6) 3 = (3167 : ℚ) / 1000 ∧
  -- 3. 0.25 = 1/4
  (1 : ℚ) / 4 = (25 : ℚ) / 100 ∧
  -- 4. 0.08 = 2/25
  (2 : ℚ) / 25 = (8 : ℚ) / 100 := by
  sorry

end fraction_decimal_conversions_l2540_254088


namespace intersection_range_l2540_254078

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x ≤ 0 ∧ y = 0)

-- Define the line segment AB
def lineAB (a x y : ℝ) : Prop :=
  y = x + 1 ∧ x ≥ a - 1 ∧ x ≤ a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ lineAB a p.1 p.2) →
  a ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
sorry

end intersection_range_l2540_254078


namespace increasing_function_range_l2540_254070

/-- Given a ∈ (0,1) and f(x) = a^x + (1+a)^x is increasing on (0,+∞), 
    prove that a ∈ [((5^(1/2)) - 1)/2, 1) -/
theorem increasing_function_range (a : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : ∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) : 
  a ∈ Set.Icc ((Real.sqrt 5 - 1) / 2) 1 := by
  sorry

end increasing_function_range_l2540_254070


namespace profit_percent_l2540_254033

theorem profit_percent (P : ℝ) (C : ℝ) (h1 : C > 0) (h2 : (2/3) * P = 0.88 * C) :
  (P - C) / C = 0.32 := by
sorry

end profit_percent_l2540_254033


namespace gcd_840_1764_l2540_254075

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2540_254075


namespace bridge_length_is_4km_l2540_254042

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The length of a bridge is 4 km when crossed by a man walking at 10 km/hr in 24 minutes -/
theorem bridge_length_is_4km (speed : ℝ) (time : ℝ) 
    (h1 : speed = 10) 
    (h2 : time = 24 / 60) : 
  bridge_length speed time = 4 := by
  sorry

end bridge_length_is_4km_l2540_254042


namespace functional_inequality_solution_l2540_254022

theorem functional_inequality_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) ≤ (1/2) * (f x + f y)) : 
  ∃ a c : ℝ, (∀ x : ℝ, x ≠ 0 → f x = c) ∧ (f 0 = a) ∧ (a ≤ c) := by
  sorry

end functional_inequality_solution_l2540_254022


namespace least_addition_for_divisibility_l2540_254041

theorem least_addition_for_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬((1077 + m) % 23 = 0 ∧ (1077 + m) % 17 = 0)) ∧ 
  ((1077 + n) % 23 = 0 ∧ (1077 + n) % 17 = 0) :=
by
  -- Proof goes here
  sorry

end least_addition_for_divisibility_l2540_254041


namespace carter_baseball_cards_l2540_254063

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carter_baseball_cards :
  let marcus_cards : ℕ := 210
  let difference : ℕ := 58
  let carter_cards : ℕ := marcus_cards - difference
  carter_cards = 152 :=
by sorry

end carter_baseball_cards_l2540_254063


namespace abs_two_over_z_minus_z_equals_two_l2540_254020

/-- Given a complex number z = 1 + i, prove that |2/z - z| = 2 -/
theorem abs_two_over_z_minus_z_equals_two :
  let z : ℂ := 1 + Complex.I
  Complex.abs (2 / z - z) = 2 := by
  sorry

end abs_two_over_z_minus_z_equals_two_l2540_254020


namespace problem_solution_l2540_254046

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the conditions of the problem
def is_purely_imaginary (z : ℂ) : Prop := ∃ (a : ℝ), z = a * i

-- State the theorem
theorem problem_solution (x y : ℂ) 
  (h1 : is_purely_imaginary x) 
  (h2 : y.im = 0) 
  (h3 : 2 * x - 1 + i = y - (3 - y) * i) : 
  x + y = -1 - (5/2) * i := by sorry

end problem_solution_l2540_254046


namespace smallest_m_for_exact_tax_l2540_254083

theorem smallest_m_for_exact_tax : ∃ (x : ℕ+), 
  (106 * x : ℕ) % 100 = 0 ∧ 
  (106 * x : ℕ) / 100 = 53 ∧ 
  ∀ (m : ℕ+), m < 53 → ¬∃ (y : ℕ+), (106 * y : ℕ) % 100 = 0 ∧ (106 * y : ℕ) / 100 = m := by
  sorry

end smallest_m_for_exact_tax_l2540_254083


namespace coefficient_b_is_zero_l2540_254090

/-- Given an equation px + qy + bz = 1 with three solutions, prove that b = 0 -/
theorem coefficient_b_is_zero
  (p q b a : ℝ)
  (h1 : q * (3 * a) + b * 1 = 1)
  (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
  (h3 : q * (3 * a) = 1) :
  b = 0 := by
  sorry

end coefficient_b_is_zero_l2540_254090


namespace system_solution_l2540_254099

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 2 ∧ x₁^2 - 2*x₁*y₁ - 3*y₁^2 = 0 ∧ x₁ = 3 ∧ y₁ = 1) ∧
    (x₂ - y₂ = 2 ∧ x₂^2 - 2*x₂*y₂ - 3*y₂^2 = 0 ∧ x₂ = 1 ∧ y₂ = -1) :=
by sorry

end system_solution_l2540_254099


namespace unique_solution_system_l2540_254038

theorem unique_solution_system (x y z : ℝ) : 
  (x + y = 2 ∧ x * y - z^2 = 1) → (x = 1 ∧ y = 1 ∧ z = 0) :=
by sorry

end unique_solution_system_l2540_254038


namespace smallest_y_for_perfect_fourth_power_l2540_254058

def x : ℕ := 5 * 27 * 64

theorem smallest_y_for_perfect_fourth_power (y : ℕ) : 
  y > 0 ∧ 
  (∀ z : ℕ, z > 0 ∧ z < y → ¬ ∃ n : ℕ, x * z = n^4) ∧
  (∃ n : ℕ, x * y = n^4) → 
  y = 1500 := by sorry

end smallest_y_for_perfect_fourth_power_l2540_254058


namespace unique_solution_square_equation_l2540_254002

theorem unique_solution_square_equation :
  ∃! y : ℤ, (2010 + y)^2 = y^2 ∧ y = -1005 := by sorry

end unique_solution_square_equation_l2540_254002


namespace problem_solution_l2540_254094

def A : Set ℝ := {x | x^2 + 5*x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 3 = 0}

theorem problem_solution :
  (A ∪ B 0 = {-6, 1, -3}) ∧
  (∀ m : ℝ, B m ⊆ A ↔ m ≤ -2) := by sorry

end problem_solution_l2540_254094


namespace roots_of_quadratic_l2540_254098

theorem roots_of_quadratic (a b : ℝ) : 
  (a * b ≠ 0) →
  (a^2 + 2*b*a + a = 0) →
  (b^2 + 2*b*b + a = 0) →
  (a = -3 ∧ b = 1) := by
sorry

end roots_of_quadratic_l2540_254098


namespace vector_computation_l2540_254035

theorem vector_computation :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -4]
  2 • v1 + 4 • v2 - 3 • v3 = ![(-4 : ℝ), 26] :=
by sorry

end vector_computation_l2540_254035


namespace min_fraction_sum_l2540_254044

def Digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (hw : W ∈ Digits) (hx : X ∈ Digits) (hy : Y ∈ Digits) (hz : Z ∈ Digits)
  (hdiff : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) :
  (∀ W' X' Y' Z' : ℕ, 
    W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
    W' ≠ X' ∧ W' ≠ Y' ∧ W' ≠ Z' ∧ X' ≠ Y' ∧ X' ≠ Z' ∧ Y' ≠ Z' →
    (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z') →
  (W : ℚ) / X + (Y : ℚ) / Z = 23 / 21 := by
  sorry

end min_fraction_sum_l2540_254044


namespace inequality_implies_k_bound_l2540_254049

theorem inequality_implies_k_bound :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) → k > -3 := by
  sorry

end inequality_implies_k_bound_l2540_254049


namespace quadratic_inequality_solution_set_is_real_line_l2540_254013

/-- The solution set of a quadratic inequality is the entire real line -/
theorem quadratic_inequality_solution_set_is_real_line 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ b^2 - 4*a*c < 0) :=
sorry

end quadratic_inequality_solution_set_is_real_line_l2540_254013


namespace find_g_l2540_254055

-- Define the functions f and g
def f : ℝ → ℝ := λ x ↦ 2 * x + 3

-- Define the property of g
def g_property (g : ℝ → ℝ) : Prop := ∀ x, g (x + 2) = f x

-- Theorem statement
theorem find_g : ∃ g : ℝ → ℝ, g_property g ∧ (∀ x, g x = 2 * x - 1) := by
  sorry

end find_g_l2540_254055


namespace opposite_of_negative_nine_l2540_254053

theorem opposite_of_negative_nine :
  ∃ (x : ℤ), (x + (-9) = 0) ∧ (x = 9) := by
  sorry

end opposite_of_negative_nine_l2540_254053


namespace hourly_runoff_is_1000_l2540_254018

/-- The total capacity of the sewers in gallons -/
def sewer_capacity : ℕ := 240000

/-- The number of days the sewers can handle rain before overflowing -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the hourly runoff rate -/
def hourly_runoff_rate : ℕ := sewer_capacity / (days_before_overflow * hours_per_day)

/-- Theorem stating that the hourly runoff rate is 1000 gallons per hour -/
theorem hourly_runoff_is_1000 : hourly_runoff_rate = 1000 := by
  sorry

end hourly_runoff_is_1000_l2540_254018


namespace remainder_3_pow_2000_mod_17_l2540_254068

theorem remainder_3_pow_2000_mod_17 : 3^2000 % 17 = 2 := by
  sorry

end remainder_3_pow_2000_mod_17_l2540_254068


namespace hex_725_equals_octal_3445_l2540_254069

-- Define a function to convert a base-16 number to base-10
def hexToDecimal (hex : String) : ℕ := sorry

-- Define a function to convert a base-10 number to base-8
def decimalToOctal (decimal : ℕ) : String := sorry

-- Theorem statement
theorem hex_725_equals_octal_3445 :
  decimalToOctal (hexToDecimal "725") = "3445" := by sorry

end hex_725_equals_octal_3445_l2540_254069


namespace parsley_sprigs_left_l2540_254023

/-- Calculates the number of parsley sprigs left after decorating plates --/
theorem parsley_sprigs_left 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 :=
by sorry

end parsley_sprigs_left_l2540_254023


namespace range_of_m_l2540_254048

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

def p (m : ℝ) : Prop := ∀ x ≥ 2, Monotone (f m)

def q (m : ℝ) : Prop := ∀ x, m*x^2 + 2*(m-2)*x + 1 > 0

theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Iic 1 ∪ Set.Ioo 2 4) ↔ (p m ∨ q m) :=
sorry

end range_of_m_l2540_254048


namespace car_rental_rates_l2540_254054

/-- The daily rate of the first car rental company -/
def daily_rate : ℝ := 21.95

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.19

/-- The fixed rate of City Rentals -/
def city_rentals_fixed : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.21

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem car_rental_rates :
  daily_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_fixed + city_rentals_per_mile * equal_cost_miles :=
by sorry

end car_rental_rates_l2540_254054


namespace original_selling_price_l2540_254029

theorem original_selling_price (CP : ℝ) : 
  CP * 0.85 = 544 → CP * 1.25 = 800 := by
  sorry

end original_selling_price_l2540_254029


namespace special_line_equation_l2540_254015

/-- A line passing through (9, 4) with x-intercept 5 units greater than y-intercept -/
def SpecialLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b ∧ (9, 4) ∈ {q : ℝ × ℝ | q.2 = m * q.1 + b} ∧
    ∃ (x y : ℝ), x = y + 5 ∧ 0 = m * x + b ∧ y = b}

/-- The three possible equations of the special line -/
def PossibleEquations : Set (ℝ × ℝ → Prop) :=
  {(λ p : ℝ × ℝ ↦ 2 * p.1 + 3 * p.2 - 30 = 0),
   (λ p : ℝ × ℝ ↦ 2 * p.1 - 3 * p.2 - 6 = 0),
   (λ p : ℝ × ℝ ↦ p.1 - p.2 - 5 = 0)}

theorem special_line_equation :
  ∃ (eq : ℝ × ℝ → Prop), eq ∈ PossibleEquations ∧ ∀ p : ℝ × ℝ, p ∈ SpecialLine ↔ eq p :=
by sorry

end special_line_equation_l2540_254015


namespace tangent_line_equation_l2540_254080

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

theorem tangent_line_equation :
  let A : ℝ × ℝ := (1, f 1)
  let m : ℝ := deriv f 1
  (λ (x y : ℝ) => x + y - 2 = 0) = (λ (x y : ℝ) => y - A.2 = m * (x - A.1)) := by sorry

end tangent_line_equation_l2540_254080


namespace tuesday_extra_minutes_l2540_254085

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of minutes Ayen jogs on a regular weekday -/
def regular_jog : ℕ := 30

/-- The number of extra minutes Ayen jogged on Friday -/
def friday_extra : ℕ := 25

/-- The total number of minutes Ayen jogged this week -/
def total_jog : ℕ := 3 * 60

/-- The number of extra minutes Ayen jogged on Tuesday -/
def tuesday_extra : ℕ := total_jog - (weekdays * regular_jog) - friday_extra

theorem tuesday_extra_minutes : tuesday_extra = 5 := by sorry

end tuesday_extra_minutes_l2540_254085


namespace journey_speed_theorem_l2540_254000

/-- Proves that given a round trip journey where the time taken to go up is twice the time taken to come down,
    the total journey time is 6 hours, and the average speed for the whole journey is 4 km/h,
    then the average speed while going up is 3 km/h. -/
theorem journey_speed_theorem (time_up : ℝ) (time_down : ℝ) (total_distance : ℝ) :
  time_up = 2 * time_down →
  time_up + time_down = 6 →
  total_distance / (time_up + time_down) = 4 →
  (total_distance / 2) / time_up = 3 :=
by sorry

end journey_speed_theorem_l2540_254000


namespace fred_initial_money_l2540_254026

/-- Calculates the initial amount of money Fred had given the number of books bought,
    the average cost per book, and the amount left after buying. -/
def initial_money (num_books : ℕ) (avg_cost : ℕ) (money_left : ℕ) : ℕ :=
  num_books * avg_cost + money_left

/-- Proves that Fred initially had 236 dollars given the problem conditions. -/
theorem fred_initial_money :
  let num_books : ℕ := 6
  let avg_cost : ℕ := 37
  let money_left : ℕ := 14
  initial_money num_books avg_cost money_left = 236 := by
  sorry

end fred_initial_money_l2540_254026


namespace negative_of_negative_equals_absolute_value_l2540_254025

theorem negative_of_negative_equals_absolute_value : -(-5) = |(-5)| := by
  sorry

end negative_of_negative_equals_absolute_value_l2540_254025


namespace line_l1_equation_l2540_254060

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 3 = 0

-- Define the property of being perpendicular
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define the property of being tangent
def tangent (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop := sorry

-- Theorem statement
theorem line_l1_equation :
  ∀ (l1 : ℝ → ℝ → Prop),
  perpendicular l1 line_l2 →
  tangent l1 circle_C →
  (∀ x y, l1 x y ↔ (3*x + 4*y + 14 = 0 ∨ 3*x + 4*y - 6 = 0)) :=
sorry

end line_l1_equation_l2540_254060


namespace book_price_calculation_l2540_254003

def initial_price : ℝ := 250

def week1_decrease : ℝ := 0.125
def week1_increase : ℝ := 0.30
def week2_decrease : ℝ := 0.20
def week3_increase : ℝ := 0.50

def conversion_rate : ℝ := 3
def sales_tax_rate : ℝ := 0.05

def price_after_fluctuations : ℝ :=
  initial_price * (1 - week1_decrease) * (1 + week1_increase) * (1 - week2_decrease) * (1 + week3_increase)

def price_in_currency_b : ℝ := price_after_fluctuations * conversion_rate

def final_price : ℝ := price_in_currency_b * (1 + sales_tax_rate)

theorem book_price_calculation :
  final_price = 1074.9375 := by sorry

end book_price_calculation_l2540_254003


namespace angle_D_measure_l2540_254061

theorem angle_D_measure (A B C D : ℝ) :
  -- ABCD is a convex quadrilateral (implied by the angle sum condition)
  A + B + C + D = 360 →
  -- ∠C = 57°
  C = 57 →
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 →
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2 →
  -- Then ∠D = 168°
  D = 168 := by
sorry

end angle_D_measure_l2540_254061


namespace cyclist_journey_l2540_254050

theorem cyclist_journey (v t : ℝ) 
  (h1 : (v + 1) * (t - 0.5) = v * t)
  (h2 : (v - 1) * (t + 1) = v * t)
  : v * t = 6 := by
  sorry

end cyclist_journey_l2540_254050


namespace purple_sequin_rows_purple_sequin_rows_proof_l2540_254027

theorem purple_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_per_row : Nat) (green_rows : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let green_sequins := green_rows * green_per_row
  let non_purple_sequins := blue_sequins + green_sequins
  let purple_sequins := total_sequins - non_purple_sequins
  purple_sequins / purple_per_row

#check purple_sequin_rows 6 8 12 9 6 162 = 5

theorem purple_sequin_rows_proof :
  purple_sequin_rows 6 8 12 9 6 162 = 5 := by
  sorry

end purple_sequin_rows_purple_sequin_rows_proof_l2540_254027


namespace third_card_different_suit_probability_l2540_254082

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards per suit in a standard deck -/
def CardsPerSuit : ℕ := StandardDeck / NumberOfSuits

/-- The probability of picking a third card of a different suit than the first two,
    given that the first two cards are of different suits -/
def thirdCardDifferentSuitProbability : ℚ :=
  (StandardDeck - 2 - 2 * CardsPerSuit) / (StandardDeck - 2)

/-- Theorem stating that the probability of the third card being of a different suit
    than the first two is 12/25, given the conditions of the problem -/
theorem third_card_different_suit_probability :
  thirdCardDifferentSuitProbability = 12 / 25 := by
  sorry

end third_card_different_suit_probability_l2540_254082


namespace total_cost_of_cows_l2540_254001

-- Define the number of cards in a standard deck
def standard_deck_size : ℕ := 52

-- Define the number of hearts per card
def hearts_per_card : ℕ := 4

-- Define the cost per cow
def cost_per_cow : ℕ := 200

-- Define the number of cows in Devonshire
def cows_in_devonshire : ℕ := 2 * (standard_deck_size * hearts_per_card)

-- State the theorem
theorem total_cost_of_cows : cows_in_devonshire * cost_per_cow = 83200 := by
  sorry

end total_cost_of_cows_l2540_254001


namespace given_equation_is_quadratic_l2540_254081

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x

/-- Theorem: The given equation is a quadratic equation -/
theorem given_equation_is_quadratic : is_quadratic_equation f := by
  sorry

end given_equation_is_quadratic_l2540_254081


namespace polar_equation_defines_parabola_l2540_254014

/-- The polar equation r = 1 / (1 + cos θ) defines a parabola. -/
theorem polar_equation_defines_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ (x y : ℝ), (∃ (r θ : ℝ), r > 0 ∧ 
    r = 1 / (1 + Real.cos θ) ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ) ↔
    a * y^2 + b * x + c = 0) :=
sorry

end polar_equation_defines_parabola_l2540_254014


namespace james_bag_weight_l2540_254087

/-- The weight of James's bag given Oliver's bags' weights -/
theorem james_bag_weight (oliver_bag1 oliver_bag2 james_bag : ℝ) : 
  oliver_bag1 = (1 / 6) * james_bag →
  oliver_bag2 = (1 / 6) * james_bag →
  oliver_bag1 + oliver_bag2 = 6 →
  james_bag = 18 := by
  sorry

end james_bag_weight_l2540_254087


namespace max_k_value_l2540_254097

/-- Given positive real numbers x, y, and k satisfying the equation
    5 = k³(x²/y² + y²/x²) + k(x/y + y/x) + 2k²,
    the maximum possible value of k is approximately 0.8. -/
theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x) + 2 * k^2) :
  k ≤ 0.8 := by sorry

end max_k_value_l2540_254097


namespace hyperbola_real_axis_length_l2540_254039

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : ∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b ∧ (x, y) ≠ parabola_focus) 
  (hperp : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola x₁ y₁ ∧ hyperbola x₁ y₁ a b ∧
    parabola x₂ y₂ ∧ hyperbola x₂ y₂ a b ∧
    (x₁ + x₂) * (1 - x₁) + (y₁ + y₂) * (-y₁) = 0) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end hyperbola_real_axis_length_l2540_254039


namespace bowl_glass_pairing_l2540_254028

theorem bowl_glass_pairing (n : ℕ) (h : n = 5) : 
  (n : ℕ) * (n - 1) = 20 := by
  sorry

end bowl_glass_pairing_l2540_254028


namespace inequality_and_equality_conditions_l2540_254011

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a + b + c = a * b + b * c + c * a) 
  (h5 : a + b + c > 0) : 
  (Real.sqrt (b * c) * (a + 1) ≥ 2) ∧ 
  (Real.sqrt (b * c) * (a + 1) = 2 ↔ 
    (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 0 ∧ b = 2 ∧ c = 2)) := by
  sorry

end inequality_and_equality_conditions_l2540_254011


namespace solve_equations_l2540_254064

theorem solve_equations :
  (∀ x : ℚ, (16 / 5) / x = (12 / 7) / (5 / 8) → x = 7 / 6) ∧
  (∀ x : ℚ, 2 * x + 3 * 0.9 = 24.7 → x = 11) :=
by sorry

end solve_equations_l2540_254064


namespace rationalize_denominator_l2540_254062

theorem rationalize_denominator :
  let x := (Real.sqrt 12 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2)
  ∃ y, y = 4 - Real.sqrt 6 ∧ x = y :=
by sorry

end rationalize_denominator_l2540_254062


namespace vector_addition_proof_l2540_254079

def a : Fin 2 → ℝ := ![1, -2]
def b : Fin 2 → ℝ := ![3, 5]

theorem vector_addition_proof : 
  (2 • a + b) = ![5, 1] := by sorry

end vector_addition_proof_l2540_254079


namespace phyllis_gardens_tomato_percentage_l2540_254030

/-- Represents a garden with a total number of plants and a fraction of tomato plants -/
structure Garden where
  total_plants : ℕ
  tomato_fraction : ℚ

/-- Calculates the percentage of tomato plants in two gardens combined -/
def combined_tomato_percentage (g1 g2 : Garden) : ℚ :=
  let total_plants := g1.total_plants + g2.total_plants
  let total_tomatoes := g1.total_plants * g1.tomato_fraction + g2.total_plants * g2.tomato_fraction
  (total_tomatoes / total_plants) * 100

/-- Theorem stating that the percentage of tomato plants in Phyllis's two gardens is 20% -/
theorem phyllis_gardens_tomato_percentage :
  let garden1 : Garden := { total_plants := 20, tomato_fraction := 1/10 }
  let garden2 : Garden := { total_plants := 15, tomato_fraction := 1/3 }
  combined_tomato_percentage garden1 garden2 = 20 := by
  sorry

end phyllis_gardens_tomato_percentage_l2540_254030


namespace cube_volume_percentage_l2540_254091

theorem cube_volume_percentage (box_length box_width box_height cube_side : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 6)
  (h3 : box_height = 12)
  (h4 : cube_side = 4) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side) * cube_side^3) : ℚ) /
  (box_length * box_width * box_height) = 2/3 := by
sorry

end cube_volume_percentage_l2540_254091


namespace tournament_games_count_l2540_254065

/-- Calculates the number of games in a single-elimination tournament. -/
def singleEliminationGames (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Calculates the total number of games in the tournament. -/
def totalGames (initialTeams : ℕ) : ℕ :=
  let preliminaryGames := initialTeams / 2
  let remainingTeams := initialTeams - preliminaryGames
  preliminaryGames + singleEliminationGames remainingTeams

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 24 = 23 := by sorry

end tournament_games_count_l2540_254065


namespace hard_hats_remaining_l2540_254008

/-- The number of hard hats remaining in a truck after some are removed --/
def remaining_hard_hats (pink_initial green_initial yellow_initial : ℕ)
  (pink_carl pink_john green_john : ℕ) : ℕ :=
  (pink_initial - pink_carl - pink_john) +
  (green_initial - green_john) +
  yellow_initial

theorem hard_hats_remaining :
  remaining_hard_hats 26 15 24 4 6 12 = 43 := by
  sorry

end hard_hats_remaining_l2540_254008


namespace quadratic_no_real_roots_l2540_254005

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, 2 * x^2 - 5 * x + 6 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l2540_254005


namespace cookies_sold_in_morning_l2540_254004

/-- Proves the number of cookies sold in the morning given the total cookies,
    cookies sold during lunch and afternoon, and cookies left at the end of the day. -/
theorem cookies_sold_in_morning 
  (total : ℕ) 
  (lunch_sold : ℕ) 
  (afternoon_sold : ℕ) 
  (left_at_end : ℕ) 
  (h1 : total = 120) 
  (h2 : lunch_sold = 57) 
  (h3 : afternoon_sold = 16) 
  (h4 : left_at_end = 11) : 
  total - lunch_sold - afternoon_sold - left_at_end = 36 := by
  sorry

end cookies_sold_in_morning_l2540_254004


namespace triangle_area_proof_l2540_254066

theorem triangle_area_proof (square_side : ℝ) (overlap_ratio_square : ℝ) (overlap_ratio_triangle : ℝ) : 
  square_side = 8 →
  overlap_ratio_square = 3/4 →
  overlap_ratio_triangle = 1/2 →
  let square_area := square_side * square_side
  let overlap_area := square_area * overlap_ratio_square
  let triangle_area := overlap_area / overlap_ratio_triangle
  triangle_area = 96 := by
sorry

end triangle_area_proof_l2540_254066


namespace complex_equation_problem_l2540_254086

theorem complex_equation_problem (a b : ℝ) : 
  Complex.mk 1 (-2) = Complex.mk a b → a - b = 3 := by
  sorry

end complex_equation_problem_l2540_254086


namespace club_members_problem_l2540_254095

theorem club_members_problem (current_members : ℕ) : 
  (2 * current_members + 5 = current_members + 15) → 
  current_members = 10 := by
sorry

end club_members_problem_l2540_254095


namespace cubic_difference_division_l2540_254052

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end cubic_difference_division_l2540_254052


namespace cycling_competition_problem_l2540_254037

/-- Represents the distance Natalia rode on each day of the week -/
structure CyclingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- The cycling competition problem -/
theorem cycling_competition_problem (d : CyclingDistance) : 
  d.tuesday = 50 ∧ 
  d.wednesday = 0.5 * d.tuesday ∧ 
  d.thursday = d.monday + d.wednesday ∧ 
  d.monday + d.tuesday + d.wednesday + d.thursday = 180 →
  d.monday = 40 := by
sorry


end cycling_competition_problem_l2540_254037


namespace work_completed_by_two_workers_l2540_254009

/-- The fraction of work completed by two workers in one day -/
def work_completed_together (days_a : ℕ) (days_b : ℕ) : ℚ :=
  1 / days_a + 1 / days_b

/-- Theorem: Two workers A and B, where A takes 12 days and B takes half the time of A,
    can complete 1/4 of the work in one day when working together -/
theorem work_completed_by_two_workers :
  let days_a : ℕ := 12
  let days_b : ℕ := days_a / 2
  work_completed_together days_a days_b = 1 / 4 := by
sorry


end work_completed_by_two_workers_l2540_254009


namespace matrix_P_satisfies_conditions_l2540_254077

theorem matrix_P_satisfies_conditions : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2/3; 3, -4]
  (P.mulVec ![4, 0] = ![8, 12]) ∧ 
  (P.mulVec ![2, -3] = ![2, -6]) := by
  sorry

end matrix_P_satisfies_conditions_l2540_254077


namespace partition_rational_points_l2540_254019

/-- Rational points in the plane -/
def RationalPoints : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | true}

/-- The theorem statement -/
theorem partition_rational_points :
  ∃ (A B : Set (ℚ × ℚ)),
    A ∩ B = ∅ ∧
    A ∪ B = RationalPoints ∧
    (∀ t : ℚ, Set.Finite {y : ℚ | (t, y) ∈ A}) ∧
    (∀ t : ℚ, Set.Finite {x : ℚ | (x, t) ∈ B}) :=
sorry

end partition_rational_points_l2540_254019


namespace partnership_investment_l2540_254076

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a b total_profit a_profit : ℕ) (ha : a = 6300) (hb : b = 4200) 
    (h_total_profit : total_profit = 12200) (h_a_profit : a_profit = 3660) : 
    ∃ c : ℕ, c = 10490 ∧ a * total_profit = a_profit * (a + b + c) :=
by sorry

end partnership_investment_l2540_254076


namespace train_crossing_time_l2540_254047

/-- Calculates the time for a train to cross a signal pole given its length, platform length, and time to cross the platform. -/
theorem train_crossing_time (train_length platform_length time_cross_platform : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 600.0000000000001)
  (h3 : time_cross_platform = 54) : 
  ∃ (time_cross_pole : ℝ), 
    (time_cross_pole ≥ 17.99 ∧ time_cross_pole ≤ 18.01) ∧
    time_cross_pole = train_length / ((train_length + platform_length) / time_cross_platform) :=
by sorry

end train_crossing_time_l2540_254047


namespace second_person_share_correct_l2540_254043

/-- Represents the rent sharing scenario -/
structure RentSharing where
  total_rent : ℕ
  base_share : ℕ
  first_multiplier : ℕ
  second_multiplier : ℕ
  third_multiplier : ℕ

/-- Calculates the share of the second person -/
def second_person_share (rs : RentSharing) : ℕ :=
  rs.base_share * rs.second_multiplier

/-- Theorem stating the correct share for the second person -/
theorem second_person_share_correct (rs : RentSharing) 
  (h1 : rs.total_rent = 5400)
  (h2 : rs.first_multiplier = 5)
  (h3 : rs.second_multiplier = 3)
  (h4 : rs.third_multiplier = 1)
  (h5 : rs.total_rent = rs.base_share * (rs.first_multiplier + rs.second_multiplier + rs.third_multiplier)) :
  second_person_share rs = 1800 := by
  sorry

#eval second_person_share { total_rent := 5400, base_share := 600, first_multiplier := 5, second_multiplier := 3, third_multiplier := 1 }

end second_person_share_correct_l2540_254043


namespace children_savings_l2540_254074

def josiah_daily_savings : ℚ := 0.25
def josiah_days : ℕ := 24

def leah_daily_savings : ℚ := 0.50
def leah_days : ℕ := 20

def megan_days : ℕ := 12

def total_savings (j_daily : ℚ) (j_days : ℕ) (l_daily : ℚ) (l_days : ℕ) (m_days : ℕ) : ℚ :=
  j_daily * j_days + l_daily * l_days + 2 * l_daily * m_days

theorem children_savings : 
  total_savings josiah_daily_savings josiah_days leah_daily_savings leah_days megan_days = 28 := by
  sorry

end children_savings_l2540_254074


namespace list_price_correct_l2540_254034

/-- Given a book's cost price, calculates the list price that results in a 40% profit
    after an 18% deduction from the list price -/
def listPrice (costPrice : ℝ) : ℝ :=
  costPrice * 1.7073

theorem list_price_correct (costPrice : ℝ) :
  let listPrice := listPrice costPrice
  let sellingPrice := listPrice * (1 - 0.18)
  sellingPrice = costPrice * 1.4 := by
  sorry

end list_price_correct_l2540_254034


namespace catch_up_distance_l2540_254040

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ

/-- Represents the problem scenario -/
structure TwoCarsScenario where
  carA : Car
  carB : Car
  totalDistance : ℝ

/-- The conditions of the problem -/
def problemConditions (scenario : TwoCarsScenario) : Prop :=
  scenario.totalDistance = 300 ∧
  scenario.carA.startTime = scenario.carB.startTime + 1 ∧
  (scenario.totalDistance / scenario.carA.speed) + scenario.carA.startTime =
    (scenario.totalDistance / scenario.carB.speed) + scenario.carB.startTime - 1

/-- The point where carA catches up with carB -/
def catchUpPoint (scenario : TwoCarsScenario) : ℝ :=
  scenario.totalDistance - (scenario.carA.speed * (scenario.carB.startTime - scenario.carA.startTime))

/-- The theorem to be proved -/
theorem catch_up_distance (scenario : TwoCarsScenario) :
  problemConditions scenario → catchUpPoint scenario = 150 := by
  sorry

end catch_up_distance_l2540_254040


namespace work_rate_problem_l2540_254036

theorem work_rate_problem (a b c : ℝ) 
  (hab : a + b = 1/18)
  (hbc : b + c = 1/24)
  (hac : a + c = 1/36) :
  a + b + c = 1/16 := by
sorry

end work_rate_problem_l2540_254036


namespace sum_of_digits_power_minus_hundred_l2540_254071

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates 10^n - 100 for n ≥ 2 -/
def power_minus_hundred (n : ℕ) : ℕ := 
  if n ≥ 2 then 10^n - 100 else 0

theorem sum_of_digits_power_minus_hundred : 
  sum_of_digits (power_minus_hundred 100) = 882 := by sorry

end sum_of_digits_power_minus_hundred_l2540_254071


namespace partial_fraction_decomposition_l2540_254059

theorem partial_fraction_decomposition :
  ∃! (A B C D : ℚ),
    ∀ (x : ℝ), x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ -1 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 5) * (x + 1)) =
      A / (x - 2) + B / (x - 3) + C / (x - 5) + D / (x + 1) ∧
      A = -5/9 ∧ B = 0 ∧ C = 4/9 ∧ D = -1/9 := by
  sorry

end partial_fraction_decomposition_l2540_254059


namespace equal_roots_quadratic_l2540_254057

/-- 
Given a quadratic equation x^2 - 4x - m = 0 with two equal real roots,
prove that m = -4.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x - m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y - m = 0 → y = x) → 
  m = -4 := by
sorry

end equal_roots_quadratic_l2540_254057


namespace f_5_equals_357_l2540_254016

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + 5 * n + 7

theorem f_5_equals_357 : f 5 = 357 := by sorry

end f_5_equals_357_l2540_254016


namespace mans_rate_in_still_water_l2540_254084

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end mans_rate_in_still_water_l2540_254084


namespace no_real_solutions_greater_than_one_l2540_254092

theorem no_real_solutions_greater_than_one :
  ∀ x : ℝ, x > 1 → (x^10 + 1) * (x^8 + x^6 + x^4 + x^2 + 1) ≠ 22 * x^9 := by
  sorry

end no_real_solutions_greater_than_one_l2540_254092


namespace smaller_fraction_l2540_254031

theorem smaller_fraction (x y : ℚ) (sum_eq : x + y = 13/14) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by sorry

end smaller_fraction_l2540_254031


namespace max_distance_between_functions_l2540_254007

theorem max_distance_between_functions (a : ℝ) : 
  let f (x : ℝ) := 2 * (Real.cos (π / 4 + x))^2
  let g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x)
  let distance := |f a - g a|
  ∃ (max_distance : ℝ), max_distance = 3 ∧ distance ≤ max_distance ∧
    ∀ (b : ℝ), |f b - g b| ≤ max_distance :=
by sorry

end max_distance_between_functions_l2540_254007


namespace jake_and_sister_weight_l2540_254056

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 196

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := (jakes_weight - 8) / 2

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := jakes_weight + sisters_weight

/-- Theorem stating that the combined weight of Jake and his sister is 290 pounds -/
theorem jake_and_sister_weight : combined_weight = 290 := by
  sorry

/-- Lemma stating that if Jake loses 8 pounds, he will weigh twice as much as his sister -/
lemma jake_twice_sister_weight : jakes_weight - 8 = 2 * sisters_weight := by
  sorry

end jake_and_sister_weight_l2540_254056


namespace selection_theorem_l2540_254051

/-- The number of ways to choose a president, vice-president, and 2-person committee from 10 people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating the number of ways to make the selection -/
theorem selection_theorem :
  selection_ways 10 = 2520 :=
by sorry

end selection_theorem_l2540_254051


namespace probability_for_given_dice_l2540_254006

/-- Represents a 20-sided die with color distributions -/
structure TwentySidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total : Nat)
  (sum_eq_total : maroon + teal + cyan + sparkly = total)

/-- Calculate the probability of two 20-sided dice showing the same color
    and a 6-sided die showing a number greater than 4 -/
def probability_same_color_and_high_roll 
  (die1 : TwentySidedDie) 
  (die2 : TwentySidedDie) : ℚ :=
  let same_color_prob := 
    (die1.maroon * die2.maroon + 
     die1.teal * die2.teal + 
     die1.cyan * die2.cyan + 
     die1.sparkly * die2.sparkly : ℚ) / 
    (die1.total * die2.total : ℚ)
  let high_roll_prob : ℚ := 1 / 3
  same_color_prob * high_roll_prob

/-- The main theorem stating the probability for the given dice configuration -/
theorem probability_for_given_dice : 
  let die1 : TwentySidedDie := ⟨3, 9, 7, 1, 20, by norm_num⟩
  let die2 : TwentySidedDie := ⟨5, 6, 8, 1, 20, by norm_num⟩
  probability_same_color_and_high_roll die1 die2 = 21 / 200 := by
  sorry

end probability_for_given_dice_l2540_254006


namespace smallest_value_problem_l2540_254093

theorem smallest_value_problem (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 → m = 77 →
  Nat.gcd m n = x + 7 →
  Nat.lcm m n = x * (x + 7) →
  ∃ (n_min : ℕ), n_min > 0 ∧ n_min ≤ n ∧ n_min = 22 := by
  sorry

end smallest_value_problem_l2540_254093


namespace train_travel_time_l2540_254096

/-- Given a train that travels 270 miles in 3 hours, prove that it takes 2 hours to travel an additional 180 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 270 →
  initial_time = 3 →
  additional_distance = 180 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end train_travel_time_l2540_254096


namespace largest_prime_factor_of_expression_l2540_254073

theorem largest_prime_factor_of_expression : 
  (Nat.factors (16^4 + 2 * 16^2 + 1 - 13^4)).maximum = some 71 := by
  sorry

end largest_prime_factor_of_expression_l2540_254073


namespace population_after_20_years_l2540_254010

/-- The population growth over time with a constant growth rate -/
def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- The theorem stating the population after 20 years with 1% growth rate -/
theorem population_after_20_years :
  population_growth 13 0.01 20 = 13 * (1 + 0.01)^20 := by
  sorry

#eval population_growth 13 0.01 20

end population_after_20_years_l2540_254010
