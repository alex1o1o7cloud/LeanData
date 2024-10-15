import Mathlib

namespace NUMINAMATH_GPT_min_trams_spy_sees_l976_97681

/-- 
   Vasya stood at a bus stop for some time and saw 1 bus and 2 trams.
   Buses run every hour.
   After Vasya left, a spy stood at the bus stop for 10 hours and saw 10 buses.
   Given these conditions, the minimum number of trams that the spy could have seen is 5.
-/
theorem min_trams_spy_sees (bus_interval tram_interval : ℕ) 
  (vasya_buses vasya_trams spy_buses spy_hours min_trams : ℕ) 
  (h1 : bus_interval = 1)
  (h2 : vasya_buses = 1)
  (h3 : vasya_trams = 2)
  (h4 : spy_buses = spy_hours)
  (h5 : spy_buses = 10)
  (h6 : spy_hours = 10)
  (h7 : ∀ t : ℕ, t * tram_interval ≤ 2 → 2 * bus_interval ≤ 2)
  (h8 : min_trams = 5) :
  min_trams = 5 := 
sorry

end NUMINAMATH_GPT_min_trams_spy_sees_l976_97681


namespace NUMINAMATH_GPT_axis_of_symmetry_circle_l976_97614

theorem axis_of_symmetry_circle (a : ℝ) : 
  (2 * a + 0 - 1 = 0) ↔ (a = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_circle_l976_97614


namespace NUMINAMATH_GPT_no_integer_b_satisfies_conditions_l976_97647

theorem no_integer_b_satisfies_conditions :
  ¬ ∃ b : ℕ, b^6 ≤ 196 ∧ 196 < b^7 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_b_satisfies_conditions_l976_97647


namespace NUMINAMATH_GPT_shelly_total_money_l976_97663

-- Define the conditions
def num_of_ten_dollar_bills : ℕ := 10
def num_of_five_dollar_bills : ℕ := num_of_ten_dollar_bills - 4

-- Problem statement: How much money does Shelly have in all?
theorem shelly_total_money :
  (num_of_ten_dollar_bills * 10) + (num_of_five_dollar_bills * 5) = 130 :=
by
  sorry

end NUMINAMATH_GPT_shelly_total_money_l976_97663


namespace NUMINAMATH_GPT_min_value_of_sequence_l976_97646

theorem min_value_of_sequence 
  (a : ℤ) 
  (a_sequence : ℕ → ℤ) 
  (h₀ : a_sequence 0 = a)
  (h_rec : ∀ n, a_sequence (n + 1) = 2 * a_sequence n - n ^ 2)
  (h_pos : ∀ n, a_sequence n > 0) :
  ∃ k, a_sequence k = 3 := 
sorry

end NUMINAMATH_GPT_min_value_of_sequence_l976_97646


namespace NUMINAMATH_GPT_remaining_milk_and_coffee_l976_97664

/-- 
Given:
1. A cup initially contains 1 glass of coffee.
2. A quarter glass of milk is added to the cup.
3. The mixture is thoroughly stirred.
4. One glass of the mixture is poured back.

Prove:
The remaining content in the cup is 1/5 glass of milk and 4/5 glass of coffee. 
--/
theorem remaining_milk_and_coffee :
  let coffee_initial := 1  -- initial volume of coffee
  let milk_added := 1 / 4  -- volume of milk added
  let total_volume := coffee_initial + milk_added  -- total volume after mixing = 5/4 glasses
  let milk_fraction := milk_added / total_volume  -- fraction of milk in the mixture = 1/5
  let coffee_fraction := coffee_initial / total_volume  -- fraction of coffee in the mixture = 4/5
  let volume_poured := 1 / 4  -- volume of mixture poured out
  let milk_poured := (milk_fraction * volume_poured : ℝ)  -- volume of milk poured out = 1/20 glass
  let coffee_poured := (coffee_fraction * volume_poured : ℝ)  -- volume of coffee poured out = 1/5 glass
  let remaining_milk := milk_added - milk_poured  -- remaining volume of milk = 1/5 glass
  let remaining_coffee := coffee_initial - coffee_poured  -- remaining volume of coffee = 4/5 glass
  remaining_milk = 1 / 5 ∧ remaining_coffee = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_remaining_milk_and_coffee_l976_97664


namespace NUMINAMATH_GPT_proof_problem_l976_97693

-- Defining lines l1, l2, l3
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y = 2
def l2 (x y : ℝ) : Prop := 2 * x + y = -2
def l3 (x y : ℝ) : Prop := x - 2 * y = 1

-- Point P being the intersection of l1 and l2
def P : ℝ × ℝ := (-2, 2)

-- Definition of the first required line passing through P and the origin
def line_through_P_and_origin (x y : ℝ) : Prop := x + y = 0

-- Definition of the second required line passing through P and perpendicular to l3
def required_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- The theorem to prove
theorem proof_problem :
  (∃ x y, l1 x y ∧ l2 x y ∧ (x, y) = P) →
  (∀ x y, (x, y) = P → line_through_P_and_origin x y) ∧
  (∀ x y, (x, y) = P → required_line x y) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l976_97693


namespace NUMINAMATH_GPT_find_speed_of_boat_l976_97616

noncomputable def speed_of_boat_in_still_water 
  (v : ℝ) 
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) : Prop :=
  v = 42

theorem find_speed_of_boat 
  (v : ℝ)
  (current_speed : ℝ := 8) 
  (distance : ℝ := 36.67) 
  (time_in_minutes : ℝ := 44) 
  (h1 : v + current_speed = distance / (time_in_minutes / 60)) : 
  speed_of_boat_in_still_water v :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_boat_l976_97616


namespace NUMINAMATH_GPT_words_per_page_l976_97699

theorem words_per_page (p : ℕ) :
  (136 * p) % 203 = 184 % 203 ∧ p ≤ 100 → p = 73 :=
sorry

end NUMINAMATH_GPT_words_per_page_l976_97699


namespace NUMINAMATH_GPT_total_rainfall_2004_l976_97665

def average_rainfall_2003 := 50 -- in mm
def extra_rainfall_2004 := 3 -- in mm
def average_rainfall_2004 := average_rainfall_2003 + extra_rainfall_2004 -- in mm
def days_february_2004 := 29
def days_other_months := 30
def months := 12
def months_without_february := months - 1

theorem total_rainfall_2004 : 
  (average_rainfall_2004 * days_february_2004) + (months_without_february * average_rainfall_2004 * days_other_months) = 19027 := 
by sorry

end NUMINAMATH_GPT_total_rainfall_2004_l976_97665


namespace NUMINAMATH_GPT_water_bill_payment_ratio_l976_97602

variables (electricity_bill gas_bill water_bill internet_bill amount_remaining : ℤ)
variables (paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment : ℤ)

-- Define the given conditions
def stephanie_budget := 
  electricity_bill = 60 ∧
  gas_bill = 40 ∧
  water_bill = 40 ∧
  internet_bill = 25 ∧
  amount_remaining = 30 ∧
  paid_gas_bill_payments = 3 ∧ -- three-quarters
  paid_internet_bill_payments = 4 ∧ -- four payments of $5
  additional_gas_payment = 5

-- Define the given problem as a theorem
theorem water_bill_payment_ratio 
  (h : stephanie_budget electricity_bill gas_bill water_bill internet_bill amount_remaining paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment) :
  ∃ (paid_water_bill : ℤ), paid_water_bill / water_bill = 1 / 2 :=
sorry

end NUMINAMATH_GPT_water_bill_payment_ratio_l976_97602


namespace NUMINAMATH_GPT_triangle_area_correct_l976_97625

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l976_97625


namespace NUMINAMATH_GPT_num_apartments_per_floor_l976_97697

-- Definitions used in the proof
def num_buildings : ℕ := 2
def floors_per_building : ℕ := 12
def doors_per_apartment : ℕ := 7
def total_doors_needed : ℕ := 1008

-- Lean statement to proof the number of apartments per floor
theorem num_apartments_per_floor : 
  (total_doors_needed / (doors_per_apartment * num_buildings * floors_per_building)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_apartments_per_floor_l976_97697


namespace NUMINAMATH_GPT_fraction_defined_iff_l976_97678

theorem fraction_defined_iff (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) :=
by sorry

end NUMINAMATH_GPT_fraction_defined_iff_l976_97678


namespace NUMINAMATH_GPT_option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l976_97677

-- Define the variables
variables (m : ℤ)

-- State the conditions as hypotheses
theorem option_D_correct (m : ℤ) : 
  (m * (m - 1) = m^2 - m) :=
by {
    -- Proof sketch (not implemented):
    -- Use distributive property to demonstrate that both sides are equal.
    sorry
}

theorem option_A_incorrect (m : ℤ) : 
  ¬ (m^4 + m^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Demonstrate that exponents can't be added this way when bases are added.
    sorry
}

theorem option_B_incorrect (m : ℤ) : 
  ¬ ((m^4)^3 = m^7) :=
by {
    -- Proof sketch (not implemented):
    -- Show that raising m^4 to the power of 3 results in m^12.
    sorry
}

theorem option_C_incorrect (m : ℤ) : 
  ¬ (2 * m^5 / m^3 = m^2) :=
by {
    -- Proof sketch (not implemented):
    -- Show that dividing results in 2m^2.
    sorry
}

end NUMINAMATH_GPT_option_D_correct_option_A_incorrect_option_B_incorrect_option_C_incorrect_l976_97677


namespace NUMINAMATH_GPT_votes_cast_l976_97667

theorem votes_cast (A F T : ℕ) (h1 : A = 40 * T / 100) (h2 : F = A + 58) (h3 : T = F + A) : 
  T = 290 := 
by
  sorry

end NUMINAMATH_GPT_votes_cast_l976_97667


namespace NUMINAMATH_GPT_girls_in_wind_band_not_string_band_l976_97686

def M_G : ℕ := 100
def F_G : ℕ := 80
def M_O : ℕ := 80
def F_O : ℕ := 100
def total_students : ℕ := 230
def boys_in_both : ℕ := 60

theorem girls_in_wind_band_not_string_band : (F_G - (total_students - (M_G + F_G + M_O + F_O - boys_in_both - boys_in_both))) = 10 :=
by
  sorry

end NUMINAMATH_GPT_girls_in_wind_band_not_string_band_l976_97686


namespace NUMINAMATH_GPT_anya_initial_seat_l976_97670

theorem anya_initial_seat (V G D E A : ℕ) (A' : ℕ) 
  (h1 : V + G + D + E + A = 15)
  (h2 : V + 1 ≠ A')
  (h3 : G - 3 ≠ A')
  (h4 : (D = A' → E ≠ A') ∧ (E = A' → D ≠ A'))
  (h5 : A = 3 + 2)
  : A = 3 := by
  sorry

end NUMINAMATH_GPT_anya_initial_seat_l976_97670


namespace NUMINAMATH_GPT_triangle_angle_measure_l976_97610

theorem triangle_angle_measure {D E F : ℝ} (hD : D = 90) (hE : E = 2 * F + 15) : 
  D + E + F = 180 → F = 25 :=
by
  intro h_sum
  sorry

end NUMINAMATH_GPT_triangle_angle_measure_l976_97610


namespace NUMINAMATH_GPT_totalLemonProductionIn5Years_l976_97695

-- Definition of a normal lemon tree's production rate
def normalLemonProduction : ℕ := 60

-- Definition of the percentage increase for Jim's lemon trees (50%)
def percentageIncrease : ℕ := 50

-- Calculate Jim's lemon tree production per year
def jimLemonProduction : ℕ := normalLemonProduction * (100 + percentageIncrease) / 100

-- Calculate the total number of trees in Jim's grove
def treesInGrove : ℕ := 50 * 30

-- Calculate the total lemon production by Jim's grove in one year
def annualLemonProduction : ℕ := treesInGrove * jimLemonProduction

-- Calculate the total lemon production by Jim's grove in 5 years
def fiveYearLemonProduction : ℕ := 5 * annualLemonProduction

-- Theorem: Prove that the total lemon production in 5 years is 675000
theorem totalLemonProductionIn5Years : fiveYearLemonProduction = 675000 := by
  -- Proof needs to be filled in
  sorry

end NUMINAMATH_GPT_totalLemonProductionIn5Years_l976_97695


namespace NUMINAMATH_GPT_circumradius_inradius_perimeter_inequality_l976_97621

open Real

variables {R r P : ℝ} -- circumradius, inradius, perimeter
variable (triangle_type : String) -- acute, obtuse, right

def satisfies_inequality (R r P : ℝ) (triangle_type : String) : Prop :=
  if triangle_type = "right" then
    R ≥ (sqrt 2) / 2 * sqrt (P * r)
  else
    R ≥ (sqrt 3) / 3 * sqrt (P * r)

theorem circumradius_inradius_perimeter_inequality :
  ∀ (R r P : ℝ) (triangle_type : String), satisfies_inequality R r P triangle_type :=
by 
  intros R r P triangle_type
  sorry -- proof steps go here

end NUMINAMATH_GPT_circumradius_inradius_perimeter_inequality_l976_97621


namespace NUMINAMATH_GPT_initial_percentage_of_gold_l976_97660

theorem initial_percentage_of_gold (x : ℝ) (h₁ : 48 * x / 100 + 12 = 40 * 60 / 100) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_gold_l976_97660


namespace NUMINAMATH_GPT_three_exp_eq_l976_97632

theorem three_exp_eq (y : ℕ) (h : 3^y + 3^y + 3^y = 2187) : y = 6 :=
by
  sorry

end NUMINAMATH_GPT_three_exp_eq_l976_97632


namespace NUMINAMATH_GPT_parabola_focus_distance_l976_97620

theorem parabola_focus_distance (p : ℝ) (y₀ : ℝ) (h₀ : p > 0) 
  (h₁ : y₀^2 = 2 * p * 4) 
  (h₂ : dist (4, y₀) (p/2, 0) = 3/2 * p) : 
  p = 4 := 
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l976_97620


namespace NUMINAMATH_GPT_find_remainder_l976_97674

theorem find_remainder (G : ℕ) (Q1 Q2 R1 : ℕ) (hG : G = 127) (h1 : 1661 = G * Q1 + R1) (h2 : 2045 = G * Q2 + 13) : R1 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_l976_97674


namespace NUMINAMATH_GPT_stone_reaches_bottom_l976_97692

structure StoneInWater where
  σ : ℝ   -- Density of stone in g/cm³
  d : ℝ   -- Depth of lake in cm
  g : ℝ   -- Acceleration due to gravity in cm/sec²
  σ₁ : ℝ  -- Density of water in g/cm³

noncomputable def time_and_velocity (siw : StoneInWater) : ℝ × ℝ :=
  let g₁ := ((siw.σ - siw.σ₁) / siw.σ) * siw.g
  let t := Real.sqrt ((2 * siw.d) / g₁)
  let v := g₁ * t
  (t, v)

theorem stone_reaches_bottom (siw : StoneInWater)
  (hσ : siw.σ = 2.1)
  (hd : siw.d = 850)
  (hg : siw.g = 980.8)
  (hσ₁ : siw.σ₁ = 1.0) :
  time_and_velocity siw = (1.82, 935) :=
by
  sorry

end NUMINAMATH_GPT_stone_reaches_bottom_l976_97692


namespace NUMINAMATH_GPT_endomorphisms_of_Z2_are_linear_functions_l976_97639

namespace GroupEndomorphism

-- Definition of an endomorphism: a homomorphism from Z² to itself
def is_endomorphism (f : ℤ × ℤ → ℤ × ℤ) : Prop :=
  ∀ a b : ℤ × ℤ, f (a + b) = f a + f b

-- Definition of the specific form of endomorphisms for Z²
def specific_endomorphism_form (u v : ℤ × ℤ) (φ : ℤ × ℤ) : ℤ × ℤ :=
  (φ.1 * u.1 + φ.2 * v.1, φ.1 * u.2 + φ.2 * v.2)

-- Main theorem:
theorem endomorphisms_of_Z2_are_linear_functions :
  ∀ φ : ℤ × ℤ → ℤ × ℤ, is_endomorphism φ →
  ∃ u v : ℤ × ℤ, φ = specific_endomorphism_form u v := by
  sorry

end GroupEndomorphism

end NUMINAMATH_GPT_endomorphisms_of_Z2_are_linear_functions_l976_97639


namespace NUMINAMATH_GPT_large_bucket_capacity_l976_97657

variables (S L : ℝ)

theorem large_bucket_capacity (h1 : L = 2 * S + 3) (h2 : 2 * S + 5 * L = 63) : L = 11 :=
by sorry

end NUMINAMATH_GPT_large_bucket_capacity_l976_97657


namespace NUMINAMATH_GPT_factorization_proof_l976_97605

theorem factorization_proof (a : ℝ) : 2 * a^2 + 4 * a + 2 = 2 * (a + 1)^2 :=
by { sorry }

end NUMINAMATH_GPT_factorization_proof_l976_97605


namespace NUMINAMATH_GPT_parallel_lines_find_m_l976_97661

theorem parallel_lines_find_m (m : ℝ) :
  (((3 + m) / 2 = 4 / (5 + m)) ∧ ((3 + m) / 2 ≠ (5 - 3 * m) / 8)) → m = -7 :=
sorry

end NUMINAMATH_GPT_parallel_lines_find_m_l976_97661


namespace NUMINAMATH_GPT_option_A_is_correct_l976_97683

-- Define propositions p and q
variables (p q : Prop)

-- Option A
def isOptionACorrect: Prop := (¬p ∨ ¬q) → (¬p ∧ ¬q)

theorem option_A_is_correct: isOptionACorrect p q := sorry

end NUMINAMATH_GPT_option_A_is_correct_l976_97683


namespace NUMINAMATH_GPT_workman_problem_l976_97691

theorem workman_problem 
  {A B : Type}
  (W : ℕ)
  (RA RB : ℝ)
  (h1 : RA = (1/2) * RB)
  (h2 : RA + RB = W / 14)
  : W / RB = 21 :=
by
  sorry

end NUMINAMATH_GPT_workman_problem_l976_97691


namespace NUMINAMATH_GPT_total_spent_l976_97673

theorem total_spent (cost_other_toys : ℕ) (cost_lightsaber : ℕ) 
  (h1 : cost_other_toys = 1000) 
  (h2 : cost_lightsaber = 2 * cost_other_toys) : 
  cost_lightsaber + cost_other_toys = 3000 :=
  by
    sorry

end NUMINAMATH_GPT_total_spent_l976_97673


namespace NUMINAMATH_GPT_geometric_sequence_sum_l976_97638

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (a_pos : ∀ n, 0 < a n)
  (h_a2 : a 2 = 1) (h_a3a7_a5 : a 3 * a 7 - a 5 = 56)
  (S_eq : ∀ n, S n = (a 1 * (1 - (2 : ℝ) ^ n)) / (1 - 2)) :
  S 5 = 31 / 2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l976_97638


namespace NUMINAMATH_GPT_find_second_number_l976_97690

theorem find_second_number (x y : ℤ) (h1 : x = -63) (h2 : (2 + y + x) / 3 = 5) : y = 76 :=
sorry

end NUMINAMATH_GPT_find_second_number_l976_97690


namespace NUMINAMATH_GPT_paula_paint_coverage_l976_97685

-- Define the initial conditions
def initial_capacity : ℕ := 36
def lost_cans : ℕ := 4
def reduced_capacity : ℕ := 28

-- Define the proof problem
theorem paula_paint_coverage :
  (initial_capacity - reduced_capacity = lost_cans * (initial_capacity / reduced_capacity)) →
  (reduced_capacity / (initial_capacity / reduced_capacity) = 14) :=
by
  sorry

end NUMINAMATH_GPT_paula_paint_coverage_l976_97685


namespace NUMINAMATH_GPT_roger_remaining_debt_is_correct_l976_97606

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end NUMINAMATH_GPT_roger_remaining_debt_is_correct_l976_97606


namespace NUMINAMATH_GPT_problem_statement_l976_97655

theorem problem_statement (x y z : ℝ) (h : x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0) : (x - y - z) ^ 2002 = 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_l976_97655


namespace NUMINAMATH_GPT_only_prime_satisfying_condition_l976_97615

theorem only_prime_satisfying_condition (p : ℕ) (h_prime : Prime p) : (Prime (p^2 + 14) ↔ p = 3) := 
by
  sorry

end NUMINAMATH_GPT_only_prime_satisfying_condition_l976_97615


namespace NUMINAMATH_GPT_kim_morning_routine_time_l976_97609

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end NUMINAMATH_GPT_kim_morning_routine_time_l976_97609


namespace NUMINAMATH_GPT_yuna_correct_multiplication_l976_97641

theorem yuna_correct_multiplication (x : ℕ) (h : 4 * x = 60) : 8 * x = 120 :=
by
  sorry

end NUMINAMATH_GPT_yuna_correct_multiplication_l976_97641


namespace NUMINAMATH_GPT_divide_P_Q_l976_97617

noncomputable def sequence_of_ones (n : ℕ) : ℕ := (10 ^ n - 1) / 9

theorem divide_P_Q (n : ℕ) (h : 1997 ∣ sequence_of_ones n) :
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*n) + 9 * 10^(2*n) + 9 * 10^n + 7)) ∧
  1997 ∣ (sequence_of_ones (n + 1) * (10^(3*(n + 1)) + 9 * 10^(2*(n + 1)) + 9 * 10^(n + 1) + 7)) := 
by
  sorry

end NUMINAMATH_GPT_divide_P_Q_l976_97617


namespace NUMINAMATH_GPT_problem_condition_l976_97684

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_l976_97684


namespace NUMINAMATH_GPT_factor_expression_l976_97636

-- Define the variables
variables (x : ℝ)

-- State the theorem to prove
theorem factor_expression : 3 * x * (x + 1) + 7 * (x + 1) = (3 * x + 7) * (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l976_97636


namespace NUMINAMATH_GPT_total_birds_caught_l976_97627

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end NUMINAMATH_GPT_total_birds_caught_l976_97627


namespace NUMINAMATH_GPT_car_travel_speed_l976_97687

theorem car_travel_speed (v : ℝ) : 
  (1 / 60) * 3600 + 5 = (1 / v) * 3600 → v = 65 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_car_travel_speed_l976_97687


namespace NUMINAMATH_GPT_quadratic_equation_l976_97689

theorem quadratic_equation (a b c x1 x2 : ℝ) (hx1 : a * x1^2 + b * x1 + c = 0) (hx2 : a * x2^2 + b * x2 + c = 0) :
  ∃ y : ℝ, c * y^2 + b * y + a = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_equation_l976_97689


namespace NUMINAMATH_GPT_incorrect_judgment_l976_97635

-- Define propositions p and q
def p : Prop := 2 + 2 = 5
def q : Prop := 3 > 2

-- The incorrect judgment in Lean statement
theorem incorrect_judgment : ¬((p ∧ q) ∧ ¬p) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_judgment_l976_97635


namespace NUMINAMATH_GPT_tangent_line_eq_l976_97675

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * log x

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ y = 0) :
  ∃ m b, (∀ t, y = m * (t - 1) + b) ∧ (f x = y) ∧ (m = exp 1) ∧ (b = -exp 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l976_97675


namespace NUMINAMATH_GPT_students_shared_cost_l976_97676

theorem students_shared_cost (P n : ℕ) (h_price_range: 100 ≤ P ∧ P ≤ 120)
  (h_div1: P % n = 0) (h_div2: P % (n - 2) = 0) (h_extra_cost: P / n + 1 = P / (n - 2)) : n = 14 := by
  sorry

end NUMINAMATH_GPT_students_shared_cost_l976_97676


namespace NUMINAMATH_GPT_baking_powder_difference_l976_97619

-- Define the known quantities
def baking_powder_yesterday : ℝ := 0.4
def baking_powder_now : ℝ := 0.3

-- Define the statement to prove, i.e., the difference in baking powder
theorem baking_powder_difference : baking_powder_yesterday - baking_powder_now = 0.1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_baking_powder_difference_l976_97619


namespace NUMINAMATH_GPT_approx_equal_e_l976_97612
noncomputable def a : ℝ := 69.28
noncomputable def b : ℝ := 0.004
noncomputable def c : ℝ := 0.03
noncomputable def d : ℝ := a * b
noncomputable def e : ℝ := d / c

theorem approx_equal_e : abs (e - 9.24) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_approx_equal_e_l976_97612


namespace NUMINAMATH_GPT_square_area_ratio_l976_97622

theorem square_area_ratio (n : ℕ) (s₁ s₂: ℕ) (h1 : s₁ = 1) (h2 : s₂ = n^2) (h3 : 2 * s₂ - 1 = 17) :
  s₂ = 81 := 
sorry

end NUMINAMATH_GPT_square_area_ratio_l976_97622


namespace NUMINAMATH_GPT_negation_existence_l976_97640

-- The problem requires showing the equivalence between the negation of an existential
-- proposition and a universal proposition in the context of real numbers.

theorem negation_existence (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - m * x - m < 0) → (∀ x : ℝ, x^2 - m * x - m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_existence_l976_97640


namespace NUMINAMATH_GPT_inequality_solution_l976_97650

theorem inequality_solution (x : ℝ) :
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) < 2 ↔
  4 - Real.sqrt 21 < x ∧ x < 4 + Real.sqrt 21 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l976_97650


namespace NUMINAMATH_GPT_rectangle_cut_l976_97698

def dimensions_ratio (x y : ℕ) : Prop := ∃ (r : ℚ), x = r * y

theorem rectangle_cut (k m n : ℕ) (hk : ℝ) (hm : ℝ) (hn : ℝ) 
  (h1 : k + m + n = 10) 
  (h2 : k * 9 / 10 = hk)
  (h3 : m * 9 / 10 = hm)
  (h4 : n * 9 / 10 = hn)
  (h5 : hk + hm + hn = 9) :
  ∃ (k' m' n' : ℕ), 
    dimensions_ratio k k' ∧ 
    dimensions_ratio m m' ∧
    dimensions_ratio n n' ∧
    k ≠ m ∧ m ≠ n ∧ k ≠ n :=
sorry

end NUMINAMATH_GPT_rectangle_cut_l976_97698


namespace NUMINAMATH_GPT_solution_set_of_inequality_l976_97618

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x + 1) * (1 - 2 * x) > 0} = {x : ℝ | -1 / 3 < x ∧ x < 1 / 2} := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l976_97618


namespace NUMINAMATH_GPT_find_number_l976_97623

-- Defining the constants provided and the related condition
def eight_percent_of (x: ℝ) : ℝ := 0.08 * x
def ten_percent_of_40 : ℝ := 0.10 * 40
def is_solution (x: ℝ) : Prop := (eight_percent_of x) + ten_percent_of_40 = 5.92

-- Theorem statement
theorem find_number : ∃ x : ℝ, is_solution x ∧ x = 24 :=
by sorry

end NUMINAMATH_GPT_find_number_l976_97623


namespace NUMINAMATH_GPT_shonda_kids_calculation_l976_97668

def number_of_kids (B E P F A : Nat) : Nat :=
  let T := B * E
  let total_people := T / P
  total_people - (F + A + 1)

theorem shonda_kids_calculation :
  (number_of_kids 15 12 9 10 7) = 2 :=
by
  unfold number_of_kids
  exact rfl

end NUMINAMATH_GPT_shonda_kids_calculation_l976_97668


namespace NUMINAMATH_GPT_chris_earnings_total_l976_97604

-- Define the conditions
variable (hours_week1 hours_week2 : ℕ) (wage_per_hour earnings_diff : ℝ)
variable (hours_week1_val : hours_week1 = 18)
variable (hours_week2_val : hours_week2 = 30)
variable (earnings_diff_val : earnings_diff = 65.40)
variable (constant_wage : wage_per_hour > 0)

-- Theorem statement
theorem chris_earnings_total (total_earnings : ℝ) :
  hours_week2 - hours_week1 = 12 →
  wage_per_hour = earnings_diff / 12 →
  total_earnings = (hours_week1 + hours_week2) * wage_per_hour →
  total_earnings = 261.60 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_chris_earnings_total_l976_97604


namespace NUMINAMATH_GPT_neg_three_lt_neg_sqrt_eight_l976_97633

theorem neg_three_lt_neg_sqrt_eight : -3 < -Real.sqrt 8 := 
sorry

end NUMINAMATH_GPT_neg_three_lt_neg_sqrt_eight_l976_97633


namespace NUMINAMATH_GPT_no_family_of_lines_exists_l976_97679

theorem no_family_of_lines_exists (k : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n, (1 : ℝ) = k n * (1 : ℝ) + (1 - k n)) ∧
  (∀ n, k (n + 1) = a n - b n ∧ a n = 1 - 1 / k n ∧ b n = 1 - k n) ∧
  (∀ n, k n * k (n + 1) ≥ 0) →
  False :=
by
  sorry

end NUMINAMATH_GPT_no_family_of_lines_exists_l976_97679


namespace NUMINAMATH_GPT_rowing_velocity_l976_97658

theorem rowing_velocity (v : ℝ) : 
  (∀ (d : ℝ) (s : ℝ) (total_time : ℝ), 
    s = 10 ∧ 
    total_time = 30 ∧ 
    d = 144 ∧ 
    (d / (s - v) + d / (s + v)) = total_time) → 
  v = 2 := 
by
  sorry

end NUMINAMATH_GPT_rowing_velocity_l976_97658


namespace NUMINAMATH_GPT_percentage_of_brand_z_l976_97611

/-- Define the initial and subsequent conditions for the fuel tank -/
def initial_fuel_tank : ℕ := 1
def first_stage_z_gasoline : ℚ := 1 / 4
def first_stage_y_gasoline : ℚ := 3 / 4
def second_stage_z_gasoline : ℚ := first_stage_z_gasoline / 2 + 1 / 2
def second_stage_y_gasoline : ℚ := first_stage_y_gasoline / 2
def final_stage_z_gasoline : ℚ := second_stage_z_gasoline / 2
def final_stage_y_gasoline : ℚ := second_stage_y_gasoline / 2 + 1 / 2

/-- Formal statement of the problem: Prove the percentage of Brand Z gasoline -/
theorem percentage_of_brand_z :
  ∃ (percentage : ℚ), percentage = (final_stage_z_gasoline / (final_stage_z_gasoline + final_stage_y_gasoline)) * 100 ∧ percentage = 31.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_brand_z_l976_97611


namespace NUMINAMATH_GPT_factorize_quadratic_l976_97656

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end NUMINAMATH_GPT_factorize_quadratic_l976_97656


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l976_97652

noncomputable def calculate_eccentricity (a b c x0 y0 : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∀ (a b c x0 y0 : ℝ),
    (c = 2) →
    (a^2 + b^2 = 4) →
    (x0 = 3) →
    (y0^2 = 24) →
    (5 = x0 + 2) →
    calculate_eccentricity a b c x0 y0 = 2 := 
by 
  intros a b c x0 y0 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l976_97652


namespace NUMINAMATH_GPT_relationship_among_terms_l976_97645

theorem relationship_among_terms (a : ℝ) (h : a ^ 2 + a < 0) : 
  -a > a ^ 2 ∧ a ^ 2 > -a ^ 2 ∧ -a ^ 2 > a :=
sorry

end NUMINAMATH_GPT_relationship_among_terms_l976_97645


namespace NUMINAMATH_GPT_initial_volume_mixture_l976_97649

theorem initial_volume_mixture (V : ℝ) (h1 : 0.84 * V = 0.6 * (V + 24)) : V = 60 :=
by
  sorry

end NUMINAMATH_GPT_initial_volume_mixture_l976_97649


namespace NUMINAMATH_GPT_find_n_l976_97644

def f (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + n
def g (x n : ℝ) : ℝ := 2 * x^2 - 3 * x + 5 * n

theorem find_n (n : ℝ) (h : 3 * f 3 n = 2 * g 3 n) : n = 9 / 7 := by
  sorry

end NUMINAMATH_GPT_find_n_l976_97644


namespace NUMINAMATH_GPT_NaCl_yield_l976_97624

structure Reaction :=
  (reactant1 : ℕ)
  (reactant2 : ℕ)
  (product : ℕ)

def NaOH := 3
def HCl := 3

theorem NaCl_yield : ∀ (R : Reaction), R.reactant1 = NaOH → R.reactant2 = HCl → R.product = 3 :=
by
  sorry

end NUMINAMATH_GPT_NaCl_yield_l976_97624


namespace NUMINAMATH_GPT_circle_values_of_a_l976_97607

theorem circle_values_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0) ↔ (a = -1 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_circle_values_of_a_l976_97607


namespace NUMINAMATH_GPT_basketball_free_throws_l976_97654

-- Define the given conditions as assumptions
variables {a b x : ℝ}
variables (h1 : 3 * b = 2 * a)
variables (h2 : x = 2 * a - 2)
variables (h3 : 2 * a + 3 * b + x = 78)

-- State the theorem to be proven
theorem basketball_free_throws : x = 74 / 3 :=
by {
  -- We will provide the proof later
  sorry
}

end NUMINAMATH_GPT_basketball_free_throws_l976_97654


namespace NUMINAMATH_GPT_greatest_common_divisor_of_B_l976_97648

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_of_B_l976_97648


namespace NUMINAMATH_GPT_prime_divides_2_pow_n_minus_n_infinte_times_l976_97626

theorem prime_divides_2_pow_n_minus_n_infinte_times (p : ℕ) (hp : Nat.Prime p) : ∃ᶠ n in at_top, p ∣ 2^n - n :=
sorry

end NUMINAMATH_GPT_prime_divides_2_pow_n_minus_n_infinte_times_l976_97626


namespace NUMINAMATH_GPT_fraction_value_l976_97603

theorem fraction_value (x : ℝ) (h : 1 - 6 / x + 9 / (x^2) = 0) : 2 / x = 2 / 3 :=
  sorry

end NUMINAMATH_GPT_fraction_value_l976_97603


namespace NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l976_97666

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1 ∧ Nat.Prime p

theorem largest_mersenne_prime_less_than_500 :
  ∃ p : ℕ, is_mersenne_prime p ∧ p < 500 ∧ ∀ q : ℕ, is_mersenne_prime q ∧ q < 500 → q ≤ p → p = 127 :=
by
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_less_than_500_l976_97666


namespace NUMINAMATH_GPT_min_sum_xyz_l976_97682

theorem min_sum_xyz (x y z : ℝ) 
  (hx : x ≥ 4) (hy : y ≥ 5) (hz : z ≥ 6) 
  (hxyz : x^2 + y^2 + z^2 ≥ 90) : 
  x + y + z ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_sum_xyz_l976_97682


namespace NUMINAMATH_GPT_range_of_a_l976_97601

-- Define sets A and B and the condition A ∩ B = ∅
def set_A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_B (a : ℝ) : Set ℝ := { x | x > a }

-- State the condition: A ∩ B = ∅ implies a ≥ 1
theorem range_of_a (a : ℝ) : (set_A ∩ set_B a = ∅) → a ≥ 1 :=
  by
  sorry

end NUMINAMATH_GPT_range_of_a_l976_97601


namespace NUMINAMATH_GPT_grid_possible_configuration_l976_97608

theorem grid_possible_configuration (m n : ℕ) (hm : m > 100) (hn : n > 100) : 
  ∃ grid : ℕ → ℕ → ℕ,
  (∀ i j, grid i j = (if i > 0 then grid (i - 1) j else 0) + 
                       (if i < m - 1 then grid (i + 1) j else 0) + 
                       (if j > 0 then grid i (j - 1) else 0) + 
                       (if j < n - 1 then grid i (j + 1) else 0)) 
  ∧ (∃ i j, grid i j ≠ 0) 
  ∧ m > 14 
  ∧ n > 14 := 
sorry

end NUMINAMATH_GPT_grid_possible_configuration_l976_97608


namespace NUMINAMATH_GPT_yellow_block_heavier_than_green_l976_97643

theorem yellow_block_heavier_than_green :
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  yellow_block_weight - green_block_weight = 0.2 := by
  let yellow_block_weight := 0.6
  let green_block_weight := 0.4
  show yellow_block_weight - green_block_weight = 0.2
  sorry

end NUMINAMATH_GPT_yellow_block_heavier_than_green_l976_97643


namespace NUMINAMATH_GPT_bob_cleaning_time_l976_97672

theorem bob_cleaning_time (alice_time : ℕ) (h1 : alice_time = 25) (bob_ratio : ℚ) (h2 : bob_ratio = 2 / 5) : 
  bob_time = 10 :=
by
  -- Definitions for conditions
  let bob_time := bob_ratio * alice_time
  -- Sorry to represent the skipped proof
  sorry

end NUMINAMATH_GPT_bob_cleaning_time_l976_97672


namespace NUMINAMATH_GPT_proof_a_minus_b_l976_97662

def S (a : ℕ) : Set ℕ := {1, 2, a}
def T (b : ℕ) : Set ℕ := {2, 3, 4, b}

theorem proof_a_minus_b (a b : ℕ)
  (hS : S a = {1, 2, a})
  (hT : T b = {2, 3, 4, b})
  (h_intersection : S a ∩ T b = {1, 2, 3}) :
  a - b = 2 := by
  sorry

end NUMINAMATH_GPT_proof_a_minus_b_l976_97662


namespace NUMINAMATH_GPT_percent_alcohol_in_new_solution_l976_97653

theorem percent_alcohol_in_new_solution (orig_vol : ℝ) (orig_percent : ℝ) (add_alc : ℝ) (add_water : ℝ) :
  orig_percent = 5 → orig_vol = 40 → add_alc = 5.5 → add_water = 4.5 →
  (((orig_vol * (orig_percent / 100) + add_alc) / (orig_vol + add_alc + add_water)) * 100) = 15 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_percent_alcohol_in_new_solution_l976_97653


namespace NUMINAMATH_GPT_minyoung_division_l976_97631

theorem minyoung_division : 
  ∃ x : ℝ, 107.8 / x = 9.8 ∧ x = 11 :=
by
  use 11
  simp
  sorry

end NUMINAMATH_GPT_minyoung_division_l976_97631


namespace NUMINAMATH_GPT_inequality_proof_l976_97659

theorem inequality_proof (x y z : ℝ) (h : x * y * z + x + y + z = 4) : 
    (y * z + 6)^2 + (z * x + 6)^2 + (x * y + 6)^2 ≥ 8 * (x * y * z + 5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l976_97659


namespace NUMINAMATH_GPT_waiter_net_earning_l976_97669

theorem waiter_net_earning (c1 c2 c3 m : ℤ) (h1 : c1 = 3) (h2 : c2 = 2) (h3 : c3 = 1) (t1 t2 t3 : ℤ) (h4 : t1 = 8) (h5 : t2 = 10) (h6 : t3 = 12) (hmeal : m = 5):
  c1 * t1 + c2 * t2 + c3 * t3 - m = 51 := 
by 
  sorry

end NUMINAMATH_GPT_waiter_net_earning_l976_97669


namespace NUMINAMATH_GPT_fruit_bowl_apples_l976_97629

theorem fruit_bowl_apples (A : ℕ) (total_oranges initial_oranges remaining_oranges : ℕ) (percentage_apples : ℝ) :
  total_oranges = 20 →
  initial_oranges = total_oranges →
  remaining_oranges = initial_oranges - 14 →
  percentage_apples = 0.70 →
  percentage_apples * (A + remaining_oranges) = A →
  A = 14 :=
by 
  intro h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fruit_bowl_apples_l976_97629


namespace NUMINAMATH_GPT_arrangement_count_27_arrangement_count_26_l976_97680

open Int

def valid_arrangement_count (n : ℕ) : ℕ :=
  if n = 27 then 14 else if n = 26 then 105 else 0

theorem arrangement_count_27 : valid_arrangement_count 27 = 14 :=
  by
    sorry

theorem arrangement_count_26 : valid_arrangement_count 26 = 105 :=
  by
    sorry

end NUMINAMATH_GPT_arrangement_count_27_arrangement_count_26_l976_97680


namespace NUMINAMATH_GPT_Eva_is_16_l976_97651

def Clara_age : ℕ := 12
def Nora_age : ℕ := Clara_age + 3
def Liam_age : ℕ := Nora_age - 4
def Eva_age : ℕ := Liam_age + 5

theorem Eva_is_16 : Eva_age = 16 := by
  sorry

end NUMINAMATH_GPT_Eva_is_16_l976_97651


namespace NUMINAMATH_GPT_total_koalas_l976_97600

namespace KangarooKoalaProof

variables {P Q R S T U V p q r s t u v : ℕ}
variables (h₁ : P = q + r + s + t + u + v)
variables (h₂ : Q = p + r + s + t + u + v)
variables (h₃ : R = p + q + s + t + u + v)
variables (h₄ : S = p + q + r + t + u + v)
variables (h₅ : T = p + q + r + s + u + v)
variables (h₆ : U = p + q + r + s + t + v)
variables (h₇ : V = p + q + r + s + t + u)
variables (h_total : P + Q + R + S + T + U + V = 2022)

theorem total_koalas : p + q + r + s + t + u + v = 337 :=
by
  sorry

end KangarooKoalaProof

end NUMINAMATH_GPT_total_koalas_l976_97600


namespace NUMINAMATH_GPT_parabola_focus_standard_equation_l976_97630

theorem parabola_focus_standard_equation :
  ∃ (a b : ℝ), (a = 16 ∧ b = 0) ∨ (a = 0 ∧ b = -8) →
  (∃ (F : ℝ × ℝ), F = (4, 0) ∨ F = (0, -2) ∧ F ∈ {p : ℝ × ℝ | (p.1 - 2 * p.2 - 4 = 0)} →
  (∃ (x y : ℝ), (y^2 = a * x) ∨ (x^2 = b * y))) := sorry

end NUMINAMATH_GPT_parabola_focus_standard_equation_l976_97630


namespace NUMINAMATH_GPT_decompose_number_4705_l976_97642

theorem decompose_number_4705 :
  4.705 = 4 * 1 + 7 * 0.1 + 0 * 0.01 + 5 * 0.001 := by
  sorry

end NUMINAMATH_GPT_decompose_number_4705_l976_97642


namespace NUMINAMATH_GPT_cylinder_original_radius_inch_l976_97694

theorem cylinder_original_radius_inch (r : ℝ) :
  (∃ r : ℝ, (π * (r + 4)^2 * 3 = π * r^2 * 15) ∧ (r > 0)) →
  r = 1 + Real.sqrt 5 :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_original_radius_inch_l976_97694


namespace NUMINAMATH_GPT_opposite_of_neg_abs_is_positive_two_l976_97696

theorem opposite_of_neg_abs_is_positive_two : -(abs (-2)) = -2 :=
by sorry

end NUMINAMATH_GPT_opposite_of_neg_abs_is_positive_two_l976_97696


namespace NUMINAMATH_GPT_only_solution_l976_97688

def phi : ℕ → ℕ := sorry  -- Euler's totient function
def d : ℕ → ℕ := sorry    -- Divisor function

theorem only_solution (n : ℕ) (h1 : n ∣ (phi n)^(d n) + 1) (h2 : ¬ d n ^ 5 ∣ n ^ (phi n) - 1) : n = 2 :=
sorry

end NUMINAMATH_GPT_only_solution_l976_97688


namespace NUMINAMATH_GPT_multiple_of_k_l976_97671

theorem multiple_of_k (k : ℕ) (m : ℕ) (h₁ : 7 ^ k = 2) (h₂ : 7 ^ (m * k + 2) = 784) : m = 2 :=
sorry

end NUMINAMATH_GPT_multiple_of_k_l976_97671


namespace NUMINAMATH_GPT_remainder_of_polynomial_l976_97613

def p (x : ℝ) : ℝ := x^4 + 2*x^2 + 5

theorem remainder_of_polynomial (x : ℝ) : p 2 = 29 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l976_97613


namespace NUMINAMATH_GPT_negation_p_l976_97637

theorem negation_p (p : Prop) : 
  (∃ x : ℝ, x^2 ≥ x) ↔ ¬ (∀ x : ℝ, x^2 < x) :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_negation_p_l976_97637


namespace NUMINAMATH_GPT_four_distinct_real_roots_l976_97634

theorem four_distinct_real_roots (m : ℝ) : 
  (∀ x : ℝ, |(x-1)*(x-3)| = m*x → ∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ↔ 
  0 < m ∧ m < 4 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_four_distinct_real_roots_l976_97634


namespace NUMINAMATH_GPT_first_term_is_5_over_2_l976_97628

-- Define the arithmetic sequence and the sum of the first n terms.
def arith_seq (a d : ℕ) (n : ℕ) := a + (n - 1) * d
def S (a d : ℕ) (n : ℕ) := (n * (2 * a + (n - 1) * d)) / 2

-- Define the constant ratio condition.
def const_ratio (a d : ℕ) (n : ℕ) (c : ℕ) :=
  (S a d (3 * n) * 2) = c * (S a d n * 2)

-- Prove the first term is 5/2 given the conditions.
theorem first_term_is_5_over_2 (c : ℕ) (n : ℕ) (h : const_ratio a 5 n 9) : 
  a = 5 / 2 :=
sorry

end NUMINAMATH_GPT_first_term_is_5_over_2_l976_97628
