import Mathlib

namespace product_binary1101_ternary202_eq_260_l1406_140679

-- Define the binary number 1101 in base 10
def binary1101 := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 202 in base 10
def ternary202 := 2 * 3^2 + 0 * 3^1 + 2 * 3^0

-- Prove that their product in base 10 is 260
theorem product_binary1101_ternary202_eq_260 : binary1101 * ternary202 = 260 := by
  -- Proof 
  sorry

end product_binary1101_ternary202_eq_260_l1406_140679


namespace psychologist_charge_difference_l1406_140621

-- Define the variables and conditions
variables (F A : ℝ)
axiom cond1 : F + 4 * A = 250
axiom cond2 : F + A = 115

theorem psychologist_charge_difference : F - A = 25 :=
by
  -- conditions are already stated as axioms, we'll just provide the target theorem
  sorry

end psychologist_charge_difference_l1406_140621


namespace choir_arrangement_l1406_140683

/-- There are 4 possible row-lengths for arranging 90 choir members such that each row has the same
number of individuals and the number of members per row is between 6 and 15. -/
theorem choir_arrangement (x : ℕ) (h : 6 ≤ x ∧ x ≤ 15 ∧ 90 % x = 0) :
  x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15 :=
by
  sorry

end choir_arrangement_l1406_140683


namespace average_running_time_l1406_140609

variable (s : ℕ) -- Number of seventh graders

-- let sixth graders run 20 minutes per day
-- let seventh graders run 18 minutes per day
-- let eighth graders run 15 minutes per day
-- sixth graders = 3 * seventh graders
-- eighth graders = 2 * seventh graders

def sixthGradersRunningTime : ℕ := 20 * (3 * s)
def seventhGradersRunningTime : ℕ := 18 * s
def eighthGradersRunningTime : ℕ := 15 * (2 * s)

def totalRunningTime : ℕ := sixthGradersRunningTime s + seventhGradersRunningTime s + eighthGradersRunningTime s
def totalStudents : ℕ := 3 * s + s + 2 * s

theorem average_running_time : totalRunningTime s / totalStudents s = 18 :=
by sorry

end average_running_time_l1406_140609


namespace train_length_from_speed_l1406_140622

-- Definitions based on conditions
def seconds_to_cross_post : ℕ := 40
def seconds_to_cross_bridge : ℕ := 480
def bridge_length_meters : ℕ := 7200

-- Theorem statement to be proven
theorem train_length_from_speed :
  (bridge_length_meters / seconds_to_cross_bridge) * seconds_to_cross_post = 600 :=
sorry -- Proof is not provided

end train_length_from_speed_l1406_140622


namespace correct_evaluation_at_3_l1406_140689

noncomputable def polynomial (x : ℝ) : ℝ := 
  (4 * x^3 - 6 * x + 5) * (9 - 3 * x)

def expanded_poly (x : ℝ) : ℝ := 
  -12 * x^4 + 36 * x^3 + 18 * x^2 - 69 * x + 45

theorem correct_evaluation_at_3 :
  polynomial = expanded_poly →
  (12 * (-12) + 6 * 36 + 3 * 18 - 69) = 57 := 
by
  intro h
  sorry

end correct_evaluation_at_3_l1406_140689


namespace candy_total_cents_l1406_140693

def candy_cost : ℕ := 8
def gumdrops : ℕ := 28
def total_cents : ℕ := 224

theorem candy_total_cents : candy_cost * gumdrops = total_cents := by
  sorry

end candy_total_cents_l1406_140693


namespace not_snowing_next_five_days_l1406_140601

-- Define the given condition
def prob_snow : ℚ := 2 / 3

-- Define the question condition regarding not snowing for one day
def prob_no_snow : ℚ := 1 - prob_snow

-- Define the question asking for not snowing over 5 days and the expected probability
def prob_no_snow_five_days : ℚ := prob_no_snow ^ 5

theorem not_snowing_next_five_days :
  prob_no_snow_five_days = 1 / 243 :=
by 
  -- Placeholder for the proof
  sorry

end not_snowing_next_five_days_l1406_140601


namespace expected_difference_l1406_140666

noncomputable def fair_eight_sided_die := [2, 3, 4, 5, 6, 7, 8]

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop := 
  n = 4 ∨ n = 6 ∨ n = 8

def unsweetened_cereal_days := (4 / 7) * 365
def sweetened_cereal_days := (3 / 7) * 365

theorem expected_difference :
  unsweetened_cereal_days - sweetened_cereal_days = 53 := by
  sorry

end expected_difference_l1406_140666


namespace shyam_weight_increase_l1406_140687

theorem shyam_weight_increase (total_weight_after_increase : ℝ) (ram_initial_weight_ratio : ℝ) 
    (shyam_initial_weight_ratio : ℝ) (ram_increase_percent : ℝ) (total_increase_percent : ℝ) 
    (ram_total_weight_ratio : ram_initial_weight_ratio = 6) (shyam_initial_total_weight_ratio : shyam_initial_weight_ratio = 5) 
    (total_weight_after_increase_eq : total_weight_after_increase = 82.8) 
    (ram_increase_percent_eq : ram_increase_percent = 0.10) 
    (total_increase_percent_eq : total_increase_percent = 0.15) : 
  shyam_increase_percent = (21 : ℝ) :=
sorry

end shyam_weight_increase_l1406_140687


namespace teacher_periods_per_day_l1406_140653

noncomputable def periods_per_day (days_per_month : ℕ) (months : ℕ) (period_rate : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_days := days_per_month * months
  let total_periods := total_earnings / period_rate
  let periods_per_day := total_periods / total_days
  periods_per_day

theorem teacher_periods_per_day :
  periods_per_day 24 6 5 3600 = 5 := by
  sorry

end teacher_periods_per_day_l1406_140653


namespace equal_piece_length_l1406_140638

/-- A 1165 cm long rope is cut into 154 pieces, 150 of which are equally sized, and the remaining pieces are 100mm each.
    This theorem proves that the length of each equally sized piece is 75mm. -/
theorem equal_piece_length (total_length_cm : ℕ) (total_pieces : ℕ) (equal_pieces : ℕ) (remaining_piece_length_mm : ℕ) 
  (total_length_mm : ℕ) (remaining_pieces : ℕ) (equal_length_mm : ℕ) : 
  total_length_cm = 1165 ∧ 
  total_pieces = 154 ∧  
  equal_pieces = 150 ∧
  remaining_piece_length_mm = 100 ∧
  total_length_mm = total_length_cm * 10 ∧
  remaining_pieces = total_pieces - equal_pieces ∧ 
  equal_length_mm = (total_length_mm - remaining_pieces * remaining_piece_length_mm) / equal_pieces →
  equal_length_mm = 75 :=
by
  sorry

end equal_piece_length_l1406_140638


namespace incorrect_statement_D_l1406_140615

theorem incorrect_statement_D (k b x : ℝ) (hk : k < 0) (hb : b > 0) (hx : x > -b / k) :
  k * x + b ≤ 0 :=
by
  sorry

end incorrect_statement_D_l1406_140615


namespace quadratic_roots_property_l1406_140658

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l1406_140658


namespace units_digit_a2019_l1406_140602

theorem units_digit_a2019 (a : ℕ → ℝ) (h₁ : ∀ n, a n > 0)
  (h₂ : a 2 ^ 2 + a 4 ^ 2 = 900 - 2 * a 1 * a 5)
  (h₃ : a 5 = 9 * a 3) : (3^(2018) % 10) = 9 := by
  sorry

end units_digit_a2019_l1406_140602


namespace factorize_difference_of_squares_l1406_140625

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := 
by 
  sorry

end factorize_difference_of_squares_l1406_140625


namespace find_stock_face_value_l1406_140631

theorem find_stock_face_value
  (cost_price : ℝ) -- Definition for the cost price
  (discount_rate : ℝ) -- Definition for the discount rate
  (brokerage_rate : ℝ) -- Definition for the brokerage rate
  (h1 : cost_price = 98.2) -- Condition: The cost price is 98.2
  (h2 : discount_rate = 0.02) -- Condition: The discount rate is 2%
  (h3 : brokerage_rate = 0.002) -- Condition: The brokerage rate is 1/5% (0.002)
  : ∃ X : ℝ, 0.982 * X = cost_price ∧ X = 100 := -- Theorem statement to prove
by
  -- Proof omitted
  sorry

end find_stock_face_value_l1406_140631


namespace Ada_initial_seat_l1406_140608

-- We have 6 seats
def Seats := Fin 6

-- Friends' movements expressed in terms of seat positions changes
variable (Bea Ceci Dee Edie Fred Ada : Seats)

-- Conditions about the movements
variable (beMovedRight : Bea.val + 1 = Ada.val)
variable (ceMovedLeft : Ceci.val = Ada.val + 2)
variable (deeMovedRight : Dee.val + 1 = Ada.val)
variable (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
  edie_new = Fred ∧ fred_new = Edie)

-- Ada returns to an end seat (1 or 6)
axiom adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩

-- Theorem to prove Ada's initial position
theorem Ada_initial_seat (Bea Ceci Dee Edie Fred Ada : Seats)
  (beMovedRight : Bea.val + 1 = Ada.val)
  (ceMovedLeft : Ceci.val = Ada.val + 2)
  (deeMovedRight : Dee.val + 1 = Ada.val)
  (edieFredSwitch : ∀ (edie_new fred_new : Seats), 
    edie_new = Fred ∧ fred_new = Edie)
  (adaEndSeat : Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩) :
  Ada = ⟨0, by decide⟩ ∨ Ada = ⟨5, by decide⟩ := sorry

end Ada_initial_seat_l1406_140608


namespace three_digit_sum_26_l1406_140669

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_26 : 
  ∃! (n : ℕ), is_three_digit n ∧ digit_sum n = 26 := 
sorry

end three_digit_sum_26_l1406_140669


namespace time_to_carl_is_28_minutes_l1406_140650

variable (distance_to_julia : ℕ := 1) (time_to_julia : ℕ := 4)
variable (distance_to_carl : ℕ := 7)
variable (rate : ℕ := distance_to_julia * time_to_julia) -- Rate as product of distance and time

theorem time_to_carl_is_28_minutes : (distance_to_carl * time_to_julia) = 28 := by
  sorry

end time_to_carl_is_28_minutes_l1406_140650


namespace find_x_l1406_140640

theorem find_x :
  ∃ x : ℝ, 8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 ∧ x = 1.464 :=
by
  sorry

end find_x_l1406_140640


namespace max_tickets_l1406_140672

/-- Given the cost of each ticket and the total amount of money available, 
    prove that the maximum number of tickets that can be purchased is 8. -/
theorem max_tickets (ticket_cost : ℝ) (total_amount : ℝ) (h1 : ticket_cost = 18.75) (h2 : total_amount = 150) :
  (∃ n : ℕ, ticket_cost * n ≤ total_amount ∧ ∀ m : ℕ, ticket_cost * m ≤ total_amount → m ≤ n) ∧
  ∃ n : ℤ, (n : ℤ) = 8 :=
by
  sorry

end max_tickets_l1406_140672


namespace proposition_B_l1406_140684

-- Definitions of planes and lines
variable {Plane : Type}
variable {Line : Type}
variable (α β : Plane)
variable (m n : Line)

-- Definitions of parallel and perpendicular relationships
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem proposition_B (h1 : perpendicular m α) (h2 : parallel n α) : _perpendicular m n :=
sorry

end proposition_B_l1406_140684


namespace problem_solution_l1406_140636

theorem problem_solution (x : ℝ) (h : 1 - 9 / x + 20 / x^2 = 0) : (2 / x = 1 / 2 ∨ 2 / x = 2 / 5) := 
  sorry

end problem_solution_l1406_140636


namespace inverse_proportion_inequality_l1406_140655

variable (x1 x2 k : ℝ)
variable (y1 y2 : ℝ)

theorem inverse_proportion_inequality (h1 : x1 < 0) (h2 : 0 < x2) (hk : k < 0)
  (hy1 : y1 = k / x1) (hy2 : y2 = k / x2) : y2 < 0 ∧ 0 < y1 := 
by sorry

end inverse_proportion_inequality_l1406_140655


namespace p_sufficient_but_not_necessary_for_q_l1406_140664

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) :
  (|x - 1| < 2 → x ^ 2 - 5 * x - 6 < 0) ∧ ¬ (x ^ 2 - 5 * x - 6 < 0 → |x - 1| < 2) :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l1406_140664


namespace int_solutions_fraction_l1406_140665

theorem int_solutions_fraction :
  ∀ n : ℤ, (∃ k : ℤ, (n - 2) / (n + 1) = k) ↔ n = 0 ∨ n = -2 ∨ n = 2 ∨ n = -4 :=
by
  intro n
  sorry

end int_solutions_fraction_l1406_140665


namespace effective_annual_rate_correct_l1406_140623

noncomputable def nominal_annual_interest_rate : ℝ := 0.10
noncomputable def compounding_periods_per_year : ℕ := 2
noncomputable def effective_annual_rate : ℝ := (1 + nominal_annual_interest_rate / compounding_periods_per_year) ^ compounding_periods_per_year - 1

theorem effective_annual_rate_correct :
  effective_annual_rate = 0.1025 :=
by
  sorry

end effective_annual_rate_correct_l1406_140623


namespace ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l1406_140613

theorem ratio_of_area_to_square_of_perimeter_of_equilateral_triangle :
  let a := 10
  let area := (10 * 10 * (Real.sqrt 3) / 4)
  let perimeter := 3 * 10
  let square_of_perimeter := perimeter * perimeter
  (area / square_of_perimeter) = (Real.sqrt 3 / 36) := by
  -- Proof to be completed
  sorry

end ratio_of_area_to_square_of_perimeter_of_equilateral_triangle_l1406_140613


namespace Mary_work_hours_l1406_140644

variable (H : ℕ)
variable (weekly_earnings hourly_wage : ℕ)
variable (hours_Tuesday hours_Thursday : ℕ)

def weekly_hours (H : ℕ) : ℕ := 3 * H + hours_Tuesday + hours_Thursday

theorem Mary_work_hours:
  weekly_earnings = 11 * weekly_hours H → hours_Tuesday = 5 →
  hours_Thursday = 5 → weekly_earnings = 407 →
  hourly_wage = 11 → H = 9 :=
by
  intros earnings_eq tues_hours thurs_hours total_earn wage
  sorry

end Mary_work_hours_l1406_140644


namespace tom_payment_l1406_140662

theorem tom_payment :
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  total_amount = 1190 :=
by
  let q_apples := 8
  let r_apples := 70
  let q_mangoes := 9
  let r_mangoes := 70
  let cost_apples := q_apples * r_apples
  let cost_mangoes := q_mangoes * r_mangoes
  let total_amount := cost_apples + cost_mangoes
  sorry

end tom_payment_l1406_140662


namespace hyperbola_asymptote_m_value_l1406_140697

theorem hyperbola_asymptote_m_value (m : ℝ) :
  (∀ x y : ℝ, (x^2 / m - y^2 / 6 = 1) → (y = x)) → m = 6 :=
by
  intros hx
  sorry

end hyperbola_asymptote_m_value_l1406_140697


namespace units_digit_product_first_four_composite_numbers_l1406_140612

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l1406_140612


namespace proof_problem_l1406_140690

noncomputable def f (a x : ℝ) : ℝ := a^x
noncomputable def g (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem proof_problem (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_f : f a 2 = 9) : 
    g a (1/9) + f a 3 = 25 :=
by
  -- Definitions and assumptions based on the provided problem
  sorry

end proof_problem_l1406_140690


namespace baseball_games_in_season_l1406_140630

theorem baseball_games_in_season 
  (games_per_month : ℕ) 
  (months_in_season : ℕ)
  (h1 : games_per_month = 7) 
  (h2 : months_in_season = 2) :
  games_per_month * months_in_season = 14 := by
  sorry


end baseball_games_in_season_l1406_140630


namespace expression_evaluation_l1406_140619

variable (a b : ℝ)

theorem expression_evaluation (h : a + b = 1) :
  a^3 + b^3 + 3 * (a^3 * b + a * b^3) + 6 * (a^3 * b^2 + a^2 * b^3) = 1 :=
by
  sorry

end expression_evaluation_l1406_140619


namespace find_xy_l1406_140617

theorem find_xy (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : (x - y)^2 = 9) : x * y = 3 :=
sorry

end find_xy_l1406_140617


namespace metallic_sheet_first_dimension_l1406_140605

-- Given Conditions
variable (x : ℝ) (height width : ℝ)
def metallic_sheet :=
  (x > 0) ∧ (height = 8) ∧ (width = 36 - 2 * height)

-- Volume of the resulting box should be 5760 m³
def volume_box :=
  (width - 2 * height) * (x - 2 * height) * height = 5760

-- Prove the first dimension of the metallic sheet
theorem metallic_sheet_first_dimension (h1 : metallic_sheet x height width) (h2 : volume_box x height width) : 
  x = 52 :=
  sorry

end metallic_sheet_first_dimension_l1406_140605


namespace four_digit_integer_unique_l1406_140685

theorem four_digit_integer_unique (a b c d : ℕ) (h1 : a + b + c + d = 16) (h2 : b + c = 10) (h3 : a - d = 2)
    (h4 : (a - b + c - d) % 11 = 0) : a = 4 ∧ b = 6 ∧ c = 4 ∧ d = 2 := 
  by 
    sorry

end four_digit_integer_unique_l1406_140685


namespace faye_age_l1406_140600

variable (C D E F : ℕ)

-- Conditions
axiom h1 : D = 16
axiom h2 : D = E - 4
axiom h3 : E = C + 5
axiom h4 : F = C + 2

-- Goal: Prove that F = 17
theorem faye_age : F = 17 :=
by
  sorry

end faye_age_l1406_140600


namespace ratio_of_square_areas_l1406_140637

theorem ratio_of_square_areas (d s : ℝ)
  (h1 : d^2 = 2 * s^2) :
  (d^2) / (s^2) = 2 :=
by
  sorry

end ratio_of_square_areas_l1406_140637


namespace solve_real_number_pairs_l1406_140611

theorem solve_real_number_pairs (x y : ℝ) :
  (x^2 + y^2 - 48 * x - 29 * y + 714 = 0 ∧ 2 * x * y - 29 * x - 48 * y + 756 = 0) ↔
  (x = 31.5 ∧ y = 10.5) ∨ (x = 20 ∧ y = 22) ∨ (x = 28 ∧ y = 7) ∨ (x = 16.5 ∧ y = 18.5) :=
by
  sorry

end solve_real_number_pairs_l1406_140611


namespace reciprocal_of_sum_of_repeating_decimals_l1406_140624

theorem reciprocal_of_sum_of_repeating_decimals :
  let x := 5 / 33
  let y := 1 / 3
  1 / (x + y) = 33 / 16 :=
by
  -- The following is the proof, but it will be skipped for this exercise.
  sorry

end reciprocal_of_sum_of_repeating_decimals_l1406_140624


namespace fractions_equivalent_iff_x_eq_zero_l1406_140646

theorem fractions_equivalent_iff_x_eq_zero (x : ℝ) (h : (x + 1) / (x + 3) = 1 / 3) : x = 0 :=
by
  sorry

end fractions_equivalent_iff_x_eq_zero_l1406_140646


namespace simple_interest_rate_l1406_140668

theorem simple_interest_rate (P : ℝ) (increase_time : ℝ) (increase_amount : ℝ) 
(hP : P = 2000) (h_increase_time : increase_time = 4) (h_increase_amount : increase_amount = 40) :
  ∃ R : ℝ, (2000 * R / 100 * (increase_time + 4) - 2000 * R / 100 * increase_time = increase_amount) ∧ (R = 0.5) := 
by
  sorry

end simple_interest_rate_l1406_140668


namespace line_intersects_circle_l1406_140647

variable (x₀ y₀ r : Real)

theorem line_intersects_circle (h : x₀^2 + y₀^2 > r^2) : 
  ∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2) ∧ (x₀ * p.1 + y₀ * p.2 = r^2) := by
  sorry

end line_intersects_circle_l1406_140647


namespace find_alpha_l1406_140651

def demand_function (p : ℝ) : ℝ := 150 - p
def supply_function (p : ℝ) : ℝ := 3 * p - 10

def new_demand_function (p : ℝ) (α : ℝ) : ℝ := α * (150 - p)

theorem find_alpha (α : ℝ) :
  (∃ p₀ p_new, demand_function p₀ = supply_function p₀ ∧ 
    p_new = p₀ * 1.25 ∧ 
    3 * p_new - 10 = new_demand_function p_new α) →
  α = 1.4 :=
by
  sorry

end find_alpha_l1406_140651


namespace instantaneous_velocity_at_t2_l1406_140692

noncomputable def displacement (t : ℝ) : ℝ := t^2 * Real.exp (t - 2)

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2 = 8) :=
by
  sorry

end instantaneous_velocity_at_t2_l1406_140692


namespace duration_trip_for_cyclist1_l1406_140678

-- Definitions
variable (s : ℝ) -- the speed of Cyclist 1 without wind in km/h
variable (t : ℝ) -- the time in hours it takes for Cyclist 1 to travel from A to B
variable (wind_speed : ℝ := 3) -- wind modifies speed by 3 km/h
variable (total_time : ℝ := 4) -- total time after which cyclists meet

-- Conditions
axiom consistent_speed_aid : ∀ (s t : ℝ), t > 0 → (s + wind_speed) * t + (s - wind_speed) * (total_time - t) / 2 = s - wind_speed * total_time

-- Goal (equivalent proof problem)
theorem duration_trip_for_cyclist1 : t = 2 := by
  sorry

end duration_trip_for_cyclist1_l1406_140678


namespace ratio_of_x_to_y_l1406_140604

theorem ratio_of_x_to_y (x y : ℚ) (h : (2 * x - 3 * y) / (x + 2 * y) = 5 / 4) : x / y = 22 / 3 := by
  sorry

end ratio_of_x_to_y_l1406_140604


namespace find_third_angle_l1406_140661

-- Definitions from the problem conditions
def triangle_angle_sum (a b c : ℝ) : Prop := a + b + c = 180

-- Statement of the proof problem
theorem find_third_angle (a b x : ℝ) (h1 : a = 50) (h2 : b = 45) (h3 : triangle_angle_sum a b x) : x = 85 := sorry

end find_third_angle_l1406_140661


namespace inequality_solution_l1406_140691

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem inequality_solution (k : ℝ) (h_pos : 0 < k) :
  (0 < k ∧ k < 1 ∧ (1 : ℝ) < x ∧ x < (1 / k)) ∨
  (k = 1 ∧ False) ∨
  (1 < k ∧ (1 / k) < x ∧ x < 1)
  ∨ False :=
sorry

end inequality_solution_l1406_140691


namespace triangle_proof_l1406_140686

variables (α β γ a b c : ℝ)

-- Definitions based on the given conditions
def angle_relation (α β : ℝ) : Prop := 3 * α + 2 * β = 180
def triangle_angle_sum (α β γ : ℝ) : Prop := α + β + γ = 180

-- Lean statement for the proof problem
theorem triangle_proof
  (h1 : angle_relation α β)
  (h2 : triangle_angle_sum α β γ) :
  a^2 + b * c = c^2 :=
sorry

end triangle_proof_l1406_140686


namespace simplest_form_correct_l1406_140698

variable (A : ℝ)
variable (B : ℝ)
variable (C : ℝ)
variable (D : ℝ)

def is_simplest_form (x : ℝ) : Prop :=
-- define what it means for a square root to be in simplest form
sorry

theorem simplest_form_correct :
  A = Real.sqrt (1 / 2) ∧ B = Real.sqrt 0.2 ∧ C = Real.sqrt 3 ∧ D = Real.sqrt 8 →
  ¬ is_simplest_form A ∧ ¬ is_simplest_form B ∧ is_simplest_form C ∧ ¬ is_simplest_form D :=
by
  -- prove that C is the simplest form and others are not
  sorry

end simplest_form_correct_l1406_140698


namespace algebraic_expression_problem_l1406_140632

-- Define the conditions and the target statement to verify.
theorem algebraic_expression_problem (x : ℝ) 
  (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by 
  -- Add sorry to skip the proof.
  sorry

end algebraic_expression_problem_l1406_140632


namespace maxim_birth_probability_l1406_140614

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l1406_140614


namespace complex_exp_neg_ipi_on_real_axis_l1406_140656

theorem complex_exp_neg_ipi_on_real_axis :
  (Complex.exp (-Real.pi * Complex.I)).im = 0 :=
by 
  sorry

end complex_exp_neg_ipi_on_real_axis_l1406_140656


namespace stratified_sampling_l1406_140695

theorem stratified_sampling (total_students : ℕ) (num_freshmen : ℕ)
                            (freshmen_sample : ℕ) (sample_size : ℕ)
                            (h1 : total_students = 1500)
                            (h2 : num_freshmen = 400)
                            (h3 : freshmen_sample = 12)
                            (h4 : (freshmen_sample : ℚ) / num_freshmen = sample_size / total_students) :
  sample_size = 45 :=
  by
  -- There would be some steps to prove this, but they are omitted.
  sorry

end stratified_sampling_l1406_140695


namespace find_divisor_l1406_140699

theorem find_divisor (x y : ℝ) (hx : x > 0) (hx_val : x = 1.3333333333333333) (h : 4 * x / y = x^2) : y = 3 :=
by 
  sorry

end find_divisor_l1406_140699


namespace blue_balls_taken_out_l1406_140696

theorem blue_balls_taken_out
  (x : ℕ) 
  (balls_initial : ℕ := 18)
  (blue_initial : ℕ := 6)
  (prob_blue : ℚ := 1/5)
  (total : ℕ := balls_initial - x)
  (blue_current : ℕ := blue_initial - x) :
  (↑blue_current / ↑total = prob_blue) → x = 3 :=
by
  sorry

end blue_balls_taken_out_l1406_140696


namespace journey_time_l1406_140603

theorem journey_time
  (t_1 t_2 : ℝ)
  (h1 : t_1 + t_2 = 5)
  (h2 : 40 * t_1 + 60 * t_2 = 240) :
  t_1 = 3 :=
sorry

end journey_time_l1406_140603


namespace fold_creates_bisector_l1406_140633

-- Define an angle α with its vertex located outside the drawing (hence inaccessible)
structure Angle :=
  (theta1 theta2 : ℝ) -- theta1 and theta2 are the measures of the two angle sides

-- Define the condition: there exists an angle on transparent paper
variable (a: Angle)

-- Prove that folding such that the sides of the angle coincide results in the crease formed being the bisector
theorem fold_creates_bisector (a: Angle) :
  ∃ crease, crease = (a.theta1 + a.theta2) / 2 := 
sorry

end fold_creates_bisector_l1406_140633


namespace sum_p_q_eq_21_l1406_140620

theorem sum_p_q_eq_21 (p q : ℤ) :
  {x | x^2 + 6 * x - q = 0} ∩ {x | x^2 - p * x + 6 = 0} = {2} → p + q = 21 :=
by
  sorry

end sum_p_q_eq_21_l1406_140620


namespace solve_equation_l1406_140673

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48 ↔ x = 6 ∨ x = 8 := 
by
  sorry

end solve_equation_l1406_140673


namespace train_car_count_l1406_140607

theorem train_car_count
    (cars_first_15_sec : ℕ)
    (time_first_15_sec : ℕ)
    (total_time_minutes : ℕ)
    (total_additional_seconds : ℕ)
    (constant_speed : Prop)
    (h1 : cars_first_15_sec = 9)
    (h2 : time_first_15_sec = 15)
    (h3 : total_time_minutes = 3)
    (h4 : total_additional_seconds = 30)
    (h5 : constant_speed) :
    0.6 * (3 * 60 + 30) = 126 := by
  sorry

end train_car_count_l1406_140607


namespace find_multiple_of_smaller_integer_l1406_140634

theorem find_multiple_of_smaller_integer (L S k : ℕ) 
  (h1 : S = 10) 
  (h2 : L + S = 30) 
  (h3 : 2 * L = k * S - 10) 
  : k = 5 := 
by
  sorry

end find_multiple_of_smaller_integer_l1406_140634


namespace solve_inequality_l1406_140635

theorem solve_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 ↔ x = -a) ↔ (a = 1 ∨ a = 2) :=
sorry

end solve_inequality_l1406_140635


namespace multiple_of_tickletoe_nails_l1406_140688

def violet_nails := 27
def total_nails := 39
def difference := 3

theorem multiple_of_tickletoe_nails : ∃ (M T : ℕ), violet_nails = M * T + difference ∧ total_nails = violet_nails + T ∧ (M = 2) :=
by
  sorry

end multiple_of_tickletoe_nails_l1406_140688


namespace books_on_shelf_after_removal_l1406_140676

theorem books_on_shelf_after_removal :
  let initial_books : ℝ := 38.0
  let books_removed : ℝ := 10.0
  initial_books - books_removed = 28.0 :=
by 
  sorry

end books_on_shelf_after_removal_l1406_140676


namespace thirty_percent_less_than_80_equals_one_fourth_more_l1406_140628

theorem thirty_percent_less_than_80_equals_one_fourth_more (n : ℝ) :
  80 * 0.30 = 24 → 80 - 24 = 56 → n + n / 4 = 56 → n = 224 / 5 :=
by
  intros h1 h2 h3
  sorry

end thirty_percent_less_than_80_equals_one_fourth_more_l1406_140628


namespace possible_values_of_quadratic_expression_l1406_140660

theorem possible_values_of_quadratic_expression (x : ℝ) (h : 2 < x ∧ x < 3) : 
  20 < x^2 + 5 * x + 6 ∧ x^2 + 5 * x + 6 < 30 :=
by
  sorry

end possible_values_of_quadratic_expression_l1406_140660


namespace profit_ratio_l1406_140616

noncomputable def effective_capital (investment : ℕ) (months : ℕ) : ℕ := investment * months

theorem profit_ratio : 
  let P_investment := 4000
  let P_months := 12
  let Q_investment := 9000
  let Q_months := 8
  let P_effective := effective_capital P_investment P_months
  let Q_effective := effective_capital Q_investment Q_months
  (P_effective / Nat.gcd P_effective Q_effective) = 2 ∧ (Q_effective / Nat.gcd P_effective Q_effective) = 3 :=
sorry

end profit_ratio_l1406_140616


namespace next_term_in_geometric_sequence_l1406_140682

theorem next_term_in_geometric_sequence : 
  ∀ (x : ℕ), (∃ (a : ℕ), a = 768 * x^4) :=
by
  sorry

end next_term_in_geometric_sequence_l1406_140682


namespace same_volume_increase_rate_l1406_140642

def initial_radius := 10
def initial_height := 5 

def volume_increase_rate_new_radius (x : ℝ) :=
  let r' := initial_radius + 2 * x
  (r' ^ 2) * initial_height  - (initial_radius ^ 2) * initial_height

def volume_increase_rate_new_height (x : ℝ) :=
  let h' := initial_height + 3 * x
  (initial_radius ^ 2) * h' - (initial_radius ^ 2) * initial_height

theorem same_volume_increase_rate (x : ℝ) : volume_increase_rate_new_radius x = volume_increase_rate_new_height x → x = 5 := 
  by sorry

end same_volume_increase_rate_l1406_140642


namespace dave_apps_left_l1406_140663

theorem dave_apps_left (initial_apps deleted_apps remaining_apps : ℕ)
  (h_initial : initial_apps = 23)
  (h_deleted : deleted_apps = 18)
  (h_calculation : remaining_apps = initial_apps - deleted_apps) :
  remaining_apps = 5 := 
by 
  sorry

end dave_apps_left_l1406_140663


namespace find_unit_price_B_l1406_140654

/-- Definitions based on the conditions --/
def total_cost_A := 7500
def total_cost_B := 4800
def quantity_difference := 30
def price_ratio : ℝ := 2.5

/-- Define the variable x as the unit price of B type soccer balls --/
def unit_price_B (x : ℝ) : Prop :=
  (total_cost_A / (price_ratio * x)) + 30 = (total_cost_B / x) ∧
  total_cost_A > 0 ∧ total_cost_B > 0 ∧ x > 0

/-- The main statement to prove --/
theorem find_unit_price_B (x : ℝ) : unit_price_B x ↔ x = 60 :=
by
  sorry

end find_unit_price_B_l1406_140654


namespace father_and_daughter_age_l1406_140639

-- A father's age is 5 times the daughter's age.
-- In 30 years, the father will be 3 times as old as the daughter.
-- Prove that the daughter's current age is 30 and the father's current age is 150.

theorem father_and_daughter_age :
  ∃ (d f : ℤ), (f = 5 * d) ∧ (f + 30 = 3 * (d + 30)) ∧ (d = 30 ∧ f = 150) :=
by
  sorry

end father_and_daughter_age_l1406_140639


namespace alpha_in_first_quadrant_l1406_140626

theorem alpha_in_first_quadrant (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 2) < 0) 
  (h2 : Real.tan (Real.pi + α) > 0) : 
  (0 < α ∧ α < Real.pi / 2) ∨ (2 * Real.pi < α ∧ α < 5 * Real.pi / 2) := 
by
  sorry

end alpha_in_first_quadrant_l1406_140626


namespace sallys_change_l1406_140649

-- Given conditions
def frames_bought : ℕ := 3
def cost_per_frame : ℕ := 3
def payment : ℕ := 20

-- The statement to prove
theorem sallys_change : payment - (frames_bought * cost_per_frame) = 11 := by
  sorry

end sallys_change_l1406_140649


namespace total_students_in_school_l1406_140643

theorem total_students_in_school (C1 C2 C3 C4 C5 : ℕ) 
  (h1 : C1 = 23)
  (h2 : C2 = C1 - 2)
  (h3 : C3 = C2 - 2)
  (h4 : C4 = C3 - 2)
  (h5 : C5 = C4 - 2)
  : C1 + C2 + C3 + C4 + C5 = 95 := 
by 
  -- proof details skipped with sorry
  sorry

end total_students_in_school_l1406_140643


namespace function_zeros_condition_l1406_140674

theorem function_zeros_condition (a : ℝ) (H : ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧ 
  2 * Real.exp (2 * x1) - 2 * a * x1 + a - 2 * Real.exp 1 - 1 = 0 ∧ 
  2 * Real.exp (2 * x2) - 2 * a * x2 + a - 2 * Real.exp 1 - 1 = 0) :
  2 * Real.exp 1 - 1 < a ∧ a < 2 * Real.exp (2:ℝ) - 2 * Real.exp 1 - 1 := 
sorry

end function_zeros_condition_l1406_140674


namespace maximize_annual_profit_l1406_140657

noncomputable def profit_function (x : ℝ) : ℝ :=
  - (1 / 3) * x^3 + 81 * x - 234

theorem maximize_annual_profit :
  ∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function x :=
sorry

end maximize_annual_profit_l1406_140657


namespace identity_proof_l1406_140694

theorem identity_proof
  (M N x a b : ℝ)
  (h₀ : x ≠ a)
  (h₁ : x ≠ b)
  (h₂ : a ≠ b) :
  (Mx + N) / ((x - a) * (x - b)) =
  (((M *a + N) / (a - b)) * (1 / (x - a))) - 
  (((M * b + N) / (a - b)) * (1 / (x - b))) :=
sorry

end identity_proof_l1406_140694


namespace find_some_number_l1406_140618

theorem find_some_number (x : ℤ) (h : 45 - (28 - (x - (15 - 20))) = 59) : x = 37 :=
by
  sorry

end find_some_number_l1406_140618


namespace parry_position_probability_l1406_140677

theorem parry_position_probability :
    let total_members := 20
    let positions := ["President", "Vice President", "Secretary", "Treasurer"]
    let remaining_for_secretary := 18
    let remaining_for_treasurer := 17
    let prob_parry_secretary := (1 : ℚ) / remaining_for_secretary
    let prob_parry_treasurer_given_not_secretary := (1 : ℚ) / remaining_for_treasurer
    let overall_probability := prob_parry_secretary + prob_parry_treasurer_given_not_secretary * (remaining_for_treasurer / remaining_for_secretary)
    overall_probability = (1 : ℚ) / 9 := 
by
  sorry

end parry_position_probability_l1406_140677


namespace shortest_distance_Dasha_Vasya_l1406_140680

def distance_Asya_Galia : ℕ := 12
def distance_Galia_Borya : ℕ := 10
def distance_Asya_Borya : ℕ := 8
def distance_Dasha_Galia : ℕ := 15
def distance_Vasya_Galia : ℕ := 17

def distance_Dasha_Vasya : ℕ :=
  distance_Dasha_Galia + distance_Vasya_Galia - distance_Asya_Galia - distance_Galia_Borya + distance_Asya_Borya

theorem shortest_distance_Dasha_Vasya : distance_Dasha_Vasya = 18 :=
by
  -- We assume the distances as given in the conditions. The calculation part is skipped here.
  -- The actual proof steps would go here.
  sorry

end shortest_distance_Dasha_Vasya_l1406_140680


namespace wire_length_around_square_field_l1406_140652

theorem wire_length_around_square_field (area : ℝ) (times : ℕ) (wire_length : ℝ) 
    (h1 : area = 69696) (h2 : times = 15) : wire_length = 15840 :=
by
  sorry

end wire_length_around_square_field_l1406_140652


namespace find_ab_l1406_140648

variable (a b : ℝ)

def point_symmetric_about_line (Px Py Qx Qy : ℝ) (m n c : ℝ) : Prop :=
  ∃ xM yM : ℝ,
  xM = (Px + Qx) / 2 ∧ yM = (Py + Qy) / 2 ∧
  m * xM + n * yM = c ∧
  (Py - Qy) / (Px - Qx) * (-n/m) = -1

theorem find_ab (H : point_symmetric_about_line (a + 2) (b + 2) (b - a) (-b) 4 3 11) :
  a = 4 ∧ b = 2 :=
sorry

end find_ab_l1406_140648


namespace distance_home_to_school_l1406_140667

theorem distance_home_to_school
  (T T' : ℝ)
  (D : ℝ)
  (h1 : D = 6 * T)
  (h2 : D = 12 * T')
  (h3 : T - T' = 0.25) :
  D = 3 :=
by
  -- The proof would go here
  sorry

end distance_home_to_school_l1406_140667


namespace basketball_game_points_l1406_140681

theorem basketball_game_points
  (a b : ℕ) 
  (r : ℕ := 2)
  (S_E : ℕ := a / 2 * (1 + r + r^2 + r^3))
  (S_T : ℕ := 4 * b)
  (h1 : S_E = S_T + 2)
  (h2 : S_E < 100)
  (h3 : S_T < 100)
  : (a / 2 + a / 2 * r + b + b = 19) :=
by sorry

end basketball_game_points_l1406_140681


namespace solve_total_rainfall_l1406_140641

def rainfall_2010 : ℝ := 50.0
def increase_2011 : ℝ := 3.0
def increase_2012 : ℝ := 4.0

def monthly_rainfall_2011 : ℝ := rainfall_2010 + increase_2011
def monthly_rainfall_2012 : ℝ := monthly_rainfall_2011 + increase_2012

def total_rainfall_2011 : ℝ := monthly_rainfall_2011 * 12
def total_rainfall_2012 : ℝ := monthly_rainfall_2012 * 12

def total_rainfall_2011_2012 : ℝ := total_rainfall_2011 + total_rainfall_2012

theorem solve_total_rainfall :
  total_rainfall_2011_2012 = 1320.0 :=
sorry

end solve_total_rainfall_l1406_140641


namespace age_of_b_l1406_140645

variables {a b : ℕ}

theorem age_of_b (h₁ : a + 10 = 2 * (b - 10)) (h₂ : a = b + 11) : b = 41 :=
sorry

end age_of_b_l1406_140645


namespace largest_divisible_by_6_ending_in_4_l1406_140627

theorem largest_divisible_by_6_ending_in_4 : 
  ∃ n, (10 ≤ n) ∧ (n ≤ 99) ∧ (n % 6 = 0) ∧ (n % 10 = 4) ∧ ∀ m, (10 ≤ m) ∧ (m ≤ 99) ∧ (m % 6 = 0) ∧ (m % 10 = 4) → m ≤ n := 
sorry

end largest_divisible_by_6_ending_in_4_l1406_140627


namespace Addison_High_School_college_attendance_l1406_140675

theorem Addison_High_School_college_attendance:
  ∀ (G B : ℕ) (pG_not_college p_total_college : ℚ),
  G = 200 →
  B = 160 →
  pG_not_college = 0.4 →
  p_total_college = 0.6667 →
  ((B * 100) / 160) = 75 := 
by
  intro G B pG_not_college p_total_college G_eq B_eq pG_not_college_eq p_total_college_eq
  -- skipped proof
  sorry

end Addison_High_School_college_attendance_l1406_140675


namespace number_of_months_in_martian_calendar_l1406_140671

theorem number_of_months_in_martian_calendar
  (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) 
  (h2 : x + y = 74) :
  x + y = 74 := 
by
  sorry

end number_of_months_in_martian_calendar_l1406_140671


namespace punger_needs_pages_l1406_140670

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end punger_needs_pages_l1406_140670


namespace team_A_wins_series_4_1_probability_l1406_140606

noncomputable def probability_team_A_wins_series_4_1 : ℝ :=
  let home_win_prob : ℝ := 0.6
  let away_win_prob : ℝ := 0.5
  let home_loss_prob : ℝ := 1 - home_win_prob
  let away_loss_prob : ℝ := 1 - away_win_prob
  -- Scenario 1: L W W W W
  let p1 := home_loss_prob * home_win_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 2: W L W W W
  let p2 := home_win_prob * home_loss_prob * away_win_prob * away_win_prob * home_win_prob
  -- Scenario 3: W W L W W
  let p3 := home_win_prob * home_win_prob * away_loss_prob * away_win_prob * home_win_prob
  -- Scenario 4: W W W L W
  let p4 := home_win_prob * home_win_prob * away_win_prob * away_loss_prob * home_win_prob
  p1 + p2 + p3 + p4

theorem team_A_wins_series_4_1_probability : 
  probability_team_A_wins_series_4_1 = 0.18 :=
by
  -- This where the proof would go
  sorry

end team_A_wins_series_4_1_probability_l1406_140606


namespace relationship_of_sets_l1406_140610

def set_A : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 6 + 1}
def set_B : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 3 + 1 / 2}
def set_C : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k : ℝ) / 3 + 1 / 2}

theorem relationship_of_sets : set_C ⊆ set_B ∧ set_B ⊆ set_A := by
  sorry

end relationship_of_sets_l1406_140610


namespace exam_question_bound_l1406_140629

theorem exam_question_bound (n_students : ℕ) (k_questions : ℕ) (n_answers : ℕ) 
    (H_students : n_students = 25) (H_answers : n_answers = 5) 
    (H_condition : ∀ (i j : ℕ) (H1 : i < n_students) (H2 : j < n_students) (H_neq : i ≠ j), 
      ∀ q : ℕ, q < k_questions → ∀ ai aj : ℕ, ai < n_answers → aj < n_answers → 
      ((ai = aj) → (i = j ∨ q' > 1))) : 
    k_questions ≤ 6 := 
sorry

end exam_question_bound_l1406_140629


namespace circle_tangent_ellipse_l1406_140659

noncomputable def r : ℝ := (Real.sqrt 15) / 2

theorem circle_tangent_ellipse {x y : ℝ} (r : ℝ) (h₁ : r > 0) 
  (h₂ : ∀ x y, x^2 + 4*y^2 = 5 → ((x - r)^2 + y^2 = r^2 ∨ (x + r)^2 + y^2 = r^2))
  (h₃ : ∀ y, 4*(0 - r)^2 + (4*y^2) = 5 → ((-8*r)^2 - 4*3*(4*r^2 - 5) = 0)) :
  r = (Real.sqrt 15) / 2 :=
sorry

end circle_tangent_ellipse_l1406_140659
