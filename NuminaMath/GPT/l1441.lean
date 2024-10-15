import Mathlib

namespace NUMINAMATH_GPT_m_greater_than_p_l1441_144199

theorem m_greater_than_p (p m n : ℕ) (hp : Nat.Prime p) (hm : 0 < m) (hn : 0 < n) (eq : p^2 + m^2 = n^2) : m > p :=
sorry

end NUMINAMATH_GPT_m_greater_than_p_l1441_144199


namespace NUMINAMATH_GPT_total_games_eq_64_l1441_144176

def games_attended : ℕ := 32
def games_missed : ℕ := 32
def total_games : ℕ := games_attended + games_missed

theorem total_games_eq_64 : total_games = 64 := by
  sorry

end NUMINAMATH_GPT_total_games_eq_64_l1441_144176


namespace NUMINAMATH_GPT_pirate_overtakes_at_8pm_l1441_144154

noncomputable def pirate_overtake_trade : Prop :=
  let initial_distance := 15
  let pirate_speed_before_damage := 14
  let trade_speed := 10
  let time_before_damage := 3
  let pirate_distance_before_damage := pirate_speed_before_damage * time_before_damage
  let trade_distance_before_damage := trade_speed * time_before_damage
  let remaining_distance := initial_distance + trade_distance_before_damage - pirate_distance_before_damage
  let pirate_speed_after_damage := (18 / 17) * 10
  let relative_speed_after_damage := pirate_speed_after_damage - trade_speed
  let time_to_overtake_after_damage := remaining_distance / relative_speed_after_damage
  let total_time := time_before_damage + time_to_overtake_after_damage
  total_time = 8

theorem pirate_overtakes_at_8pm : pirate_overtake_trade :=
by
  sorry

end NUMINAMATH_GPT_pirate_overtakes_at_8pm_l1441_144154


namespace NUMINAMATH_GPT_abs_eq_sum_solutions_l1441_144166

theorem abs_eq_sum_solutions (x : ℝ) : (|3*x - 2| + |3*x + 1| = 3) ↔ 
  (x = -1 / 3 ∨ (-1 / 3 < x ∧ x <= 2 / 3)) :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_sum_solutions_l1441_144166


namespace NUMINAMATH_GPT_daisies_left_l1441_144160

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9
def total_daisies : ℕ := initial_daisies + sister_daisies
def daisies_given_to_mother : ℕ := total_daisies / 2
def remaining_daisies : ℕ := total_daisies - daisies_given_to_mother

theorem daisies_left : remaining_daisies = 7 := by
  sorry

end NUMINAMATH_GPT_daisies_left_l1441_144160


namespace NUMINAMATH_GPT_parallelogram_height_l1441_144140

theorem parallelogram_height
  (A b : ℝ)
  (h : ℝ)
  (h_area : A = 120)
  (h_base : b = 12)
  (h_formula : A = b * h) : h = 10 :=
by 
  sorry

end NUMINAMATH_GPT_parallelogram_height_l1441_144140


namespace NUMINAMATH_GPT_complete_square_b_l1441_144161

theorem complete_square_b (a b x : ℝ) (h : x^2 + 6 * x - 3 = 0) : (x + a)^2 = b → b = 12 := by
  sorry

end NUMINAMATH_GPT_complete_square_b_l1441_144161


namespace NUMINAMATH_GPT_greater_number_is_84_l1441_144107

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : x + y - (x - y) = 64) :
  x = 84 :=
by sorry

end NUMINAMATH_GPT_greater_number_is_84_l1441_144107


namespace NUMINAMATH_GPT_value_standard_deviations_from_mean_l1441_144134

-- Define the mean (µ)
def μ : ℝ := 15.5

-- Define the standard deviation (σ)
def σ : ℝ := 1.5

-- Define the value X
def X : ℝ := 12.5

-- Prove that the Z-score is -2
theorem value_standard_deviations_from_mean : (X - μ) / σ = -2 := by
  sorry

end NUMINAMATH_GPT_value_standard_deviations_from_mean_l1441_144134


namespace NUMINAMATH_GPT_tennis_balls_ordered_originally_l1441_144147

-- Definitions according to the conditions in a)
def retailer_ordered_equal_white_yellow_balls (W Y : ℕ) : Prop :=
  W = Y

def dispatch_error (Y : ℕ) : ℕ :=
  Y + 90

def ratio_white_to_yellow (W Y : ℕ) : Prop :=
  W / dispatch_error Y = 8 / 13

-- Main statement
theorem tennis_balls_ordered_originally (W Y : ℕ) (h1 : retailer_ordered_equal_white_yellow_balls W Y)
  (h2 : ratio_white_to_yellow W Y) : W + Y = 288 :=
by
  sorry    -- Placeholder for the actual proof

end NUMINAMATH_GPT_tennis_balls_ordered_originally_l1441_144147


namespace NUMINAMATH_GPT_multiples_of_seven_with_units_digit_three_l1441_144165

theorem multiples_of_seven_with_units_digit_three :
  ∃ n : ℕ, n = 2 ∧ ∀ k : ℕ, (k < 150 ∧ k % 7 = 0 ∧ k % 10 = 3) ↔ (k = 63 ∨ k = 133) := by
  sorry

end NUMINAMATH_GPT_multiples_of_seven_with_units_digit_three_l1441_144165


namespace NUMINAMATH_GPT_number_of_senior_citizen_tickets_sold_on_first_day_l1441_144114

theorem number_of_senior_citizen_tickets_sold_on_first_day 
  (S : ℤ) (x : ℤ)
  (student_ticket_price : ℤ := 9)
  (first_day_sales : ℤ := 79)
  (second_day_sales : ℤ := 246) 
  (first_day_student_tickets_sold : ℤ := 3)
  (second_day_senior_tickets_sold : ℤ := 12)
  (second_day_student_tickets_sold : ℤ := 10) 
  (h1 : 12 * S + 10 * student_ticket_price = second_day_sales)
  (h2 : S * x + first_day_student_tickets_sold * student_ticket_price = first_day_sales) : 
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_senior_citizen_tickets_sold_on_first_day_l1441_144114


namespace NUMINAMATH_GPT_cost_price_is_975_l1441_144128

-- Definitions from the conditions
def selling_price : ℝ := 1170
def profit_percentage : ℝ := 0.20

-- The proof statement
theorem cost_price_is_975 : (selling_price / (1 + profit_percentage)) = 975 := by
  sorry

end NUMINAMATH_GPT_cost_price_is_975_l1441_144128


namespace NUMINAMATH_GPT_juice_profit_eq_l1441_144162

theorem juice_profit_eq (x : ℝ) :
  (70 - x) * (160 + 8 * x) = 16000 :=
sorry

end NUMINAMATH_GPT_juice_profit_eq_l1441_144162


namespace NUMINAMATH_GPT_Gwen_remaining_homework_l1441_144193

def initial_problems_math := 18
def completed_problems_math := 12
def remaining_problems_math := initial_problems_math - completed_problems_math

def initial_problems_science := 11
def completed_problems_science := 6
def remaining_problems_science := initial_problems_science - completed_problems_science

def initial_questions_history := 15
def completed_questions_history := 10
def remaining_questions_history := initial_questions_history - completed_questions_history

def initial_questions_english := 7
def completed_questions_english := 4
def remaining_questions_english := initial_questions_english - completed_questions_english

def total_remaining_problems := remaining_problems_math 
                               + remaining_problems_science 
                               + remaining_questions_history 
                               + remaining_questions_english

theorem Gwen_remaining_homework : total_remaining_problems = 19 :=
by
  sorry

end NUMINAMATH_GPT_Gwen_remaining_homework_l1441_144193


namespace NUMINAMATH_GPT_shared_bill_per_person_l1441_144178

noncomputable def totalBill : ℝ := 139.00
noncomputable def tipPercentage : ℝ := 0.10
noncomputable def totalPeople : ℕ := 5

theorem shared_bill_per_person :
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  amountPerPerson = 30.58 :=
by
  let tipAmount := totalBill * tipPercentage
  let totalBillWithTip := totalBill + tipAmount
  let amountPerPerson := totalBillWithTip / totalPeople
  have h1 : tipAmount = 13.90 := by sorry
  have h2 : totalBillWithTip = 152.90 := by sorry
  have h3 : amountPerPerson = 30.58 := by sorry
  exact h3

end NUMINAMATH_GPT_shared_bill_per_person_l1441_144178


namespace NUMINAMATH_GPT_deepak_current_age_l1441_144101

theorem deepak_current_age (A D : ℕ) (h1 : A / D = 5 / 7) (h2 : A + 6 = 36) : D = 42 :=
sorry

end NUMINAMATH_GPT_deepak_current_age_l1441_144101


namespace NUMINAMATH_GPT_gf_three_l1441_144110

def f (x : ℕ) : ℕ := x^3 - 4 * x + 5
def g (x : ℕ) : ℕ := 3 * x^2 + x + 2

theorem gf_three : g (f 3) = 1222 :=
by {
  -- We would need to prove the given mathematical statement here.
  sorry
}

end NUMINAMATH_GPT_gf_three_l1441_144110


namespace NUMINAMATH_GPT_initial_speed_increase_l1441_144146

variables (S : ℝ) (P : ℝ)

/-- Prove that the initial percentage increase in speed P is 0.3 based on the given conditions: 
1. After the first increase by P, the speed becomes S + PS.
2. After the second increase by 10%, the final speed is (S + PS) * 1.10.
3. The total increase results in a speed that is 1.43 times the original speed S. -/
theorem initial_speed_increase (h : (S + P * S) * 1.1 = 1.43 * S) : P = 0.3 :=
sorry

end NUMINAMATH_GPT_initial_speed_increase_l1441_144146


namespace NUMINAMATH_GPT_initial_number_is_nine_l1441_144180

theorem initial_number_is_nine (x : ℕ) (h : 3 * (2 * x + 9) = 81) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_is_nine_l1441_144180


namespace NUMINAMATH_GPT_intersection_point_of_lines_l1441_144192

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), 
    (3 * y = -2 * x + 6) ∧ 
    (-2 * y = 7 * x + 4) ∧ 
    x = -24 / 17 ∧ 
    y = 50 / 17 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l1441_144192


namespace NUMINAMATH_GPT_first_system_solution_second_system_solution_l1441_144156

theorem first_system_solution (x y : ℝ) (h₁ : 3 * x - y = 8) (h₂ : 3 * x - 5 * y = -20) : 
  x = 5 ∧ y = 7 := 
by
  sorry

theorem second_system_solution (x y : ℝ) (h₁ : x / 3 - y / 2 = -1) (h₂ : 3 * x - 2 * y = 1) : 
  x = 3 ∧ y = 4 := 
by
  sorry

end NUMINAMATH_GPT_first_system_solution_second_system_solution_l1441_144156


namespace NUMINAMATH_GPT_problem_pm_sqrt5_sin_tan_l1441_144189

theorem problem_pm_sqrt5_sin_tan
  (m : ℝ)
  (h_m_nonzero : m ≠ 0)
  (cos_alpha : ℝ)
  (h_cos_alpha : cos_alpha = (Real.sqrt 2 * m) / 4)
  (P : ℝ × ℝ)
  (h_P : P = (m, -Real.sqrt 3))
  (r : ℝ)
  (h_r : r = Real.sqrt (3 + m^2)) :
    (∃ m, m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
    (∃ sin_alpha tan_alpha,
      (sin_alpha = - Real.sqrt 6 / 4 ∧ tan_alpha = -Real.sqrt 15 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_problem_pm_sqrt5_sin_tan_l1441_144189


namespace NUMINAMATH_GPT_domain_of_f_l1441_144152

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x - 2) / Real.log 3 - 1)

theorem domain_of_f :
  {x : ℝ | f x = f x} = {x : ℝ | 2 < x ∧ x ≠ 5} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1441_144152


namespace NUMINAMATH_GPT_option_b_is_correct_l1441_144174

def is_linear (equation : String) : Bool :=
  -- Pretend implementation that checks if the given equation is linear
  -- This function would parse the string and check the linearity condition
  true -- This should be replaced by actual linearity check

def has_two_unknowns (system : List String) : Bool :=
  -- Pretend implementation that checks if the system contains exactly two unknowns
  -- This function would analyze the variables in the system
  true -- This should be replaced by actual unknowns count check

def is_system_of_two_linear_equations (system : List String) : Bool :=
  -- Checking both conditions: Each equation is linear and contains exactly two unknowns
  (system.all is_linear) && (has_two_unknowns system)

def option_b := ["x + y = 1", "x - y = 2"]

theorem option_b_is_correct :
  is_system_of_two_linear_equations option_b := 
  by
    unfold is_system_of_two_linear_equations
    -- Assuming the placeholder implementations of is_linear and has_two_unknowns
    -- actually verify the required properties, this should be true
    sorry

end NUMINAMATH_GPT_option_b_is_correct_l1441_144174


namespace NUMINAMATH_GPT_hyperbola_equation_l1441_144106

theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : (2^2 / a^2) - (1^2 / b^2) = 1) (h₄ : a^2 + b^2 = 3) :
  (∀ x y : ℝ,  (x^2 / 2) - y^2 = 1) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1441_144106


namespace NUMINAMATH_GPT_reflect_across_x_axis_l1441_144119

-- Definitions for the problem conditions
def initial_point : ℝ × ℝ := (-2, 1)
def reflected_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The statement to be proved
theorem reflect_across_x_axis :
  reflected_point initial_point = (-2, -1) :=
  sorry

end NUMINAMATH_GPT_reflect_across_x_axis_l1441_144119


namespace NUMINAMATH_GPT_find_expression_value_l1441_144155

theorem find_expression_value 
  (m : ℝ) 
  (hroot : m^2 - 3 * m + 1 = 0) : 
  (m - 3)^2 + (m + 2) * (m - 2) = 3 := 
sorry

end NUMINAMATH_GPT_find_expression_value_l1441_144155


namespace NUMINAMATH_GPT_probability_of_consecutive_blocks_drawn_l1441_144126

theorem probability_of_consecutive_blocks_drawn :
  let total_ways := (Nat.factorial 12)
  let favorable_ways := (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 5) * (Nat.factorial 3)
  (favorable_ways / total_ways) = 1 / 4620 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_consecutive_blocks_drawn_l1441_144126


namespace NUMINAMATH_GPT_identity_function_l1441_144142

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : ∀ n : ℕ, f n = n :=
by
  sorry

end NUMINAMATH_GPT_identity_function_l1441_144142


namespace NUMINAMATH_GPT_greatest_sum_consecutive_integers_product_less_than_500_l1441_144186

theorem greatest_sum_consecutive_integers_product_less_than_500 :
  ∃ n : ℕ, n * (n + 1) < 500 ∧ (n + (n + 1) = 43) := sorry

end NUMINAMATH_GPT_greatest_sum_consecutive_integers_product_less_than_500_l1441_144186


namespace NUMINAMATH_GPT_stools_chopped_up_l1441_144117

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end NUMINAMATH_GPT_stools_chopped_up_l1441_144117


namespace NUMINAMATH_GPT_profit_percentage_calculation_l1441_144129

noncomputable def profit_percentage (SP CP : ℝ) : ℝ := ((SP - CP) / CP) * 100

theorem profit_percentage_calculation (SP : ℝ) (h : CP = 0.92 * SP) : |profit_percentage SP (0.92 * SP) - 8.70| < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_calculation_l1441_144129


namespace NUMINAMATH_GPT_football_practice_hours_l1441_144181

theorem football_practice_hours (practice_hours_per_day : ℕ) (days_per_week : ℕ) (missed_days_due_to_rain : ℕ) 
  (practice_hours_per_day_eq_six : practice_hours_per_day = 6)
  (days_per_week_eq_seven : days_per_week = 7)
  (missed_days_due_to_rain_eq_one : missed_days_due_to_rain = 1) : 
  practice_hours_per_day * (days_per_week - missed_days_due_to_rain) = 36 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_football_practice_hours_l1441_144181


namespace NUMINAMATH_GPT_solve_equation_l1441_144191

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1441_144191


namespace NUMINAMATH_GPT_angle_ACD_l1441_144182

theorem angle_ACD {α β δ : Type*} [LinearOrderedField α] [CharZero α] (ABC DAB DBA : α)
  (h1 : ABC = 60) (h2 : BAC = 80) (h3 : DAB = 10) (h4 : DBA = 20):
  ACD = 30 := by
  sorry

end NUMINAMATH_GPT_angle_ACD_l1441_144182


namespace NUMINAMATH_GPT_coinCombinationCount_l1441_144171

-- Definitions for the coin values and the target amount
def quarter := 25
def dime := 10
def nickel := 5
def penny := 1
def total := 400

-- Define a function counting the number of ways to reach the total using given coin values
def countWays : Nat := sorry -- placeholder for the actual computation

-- Theorem stating the problem statement
theorem coinCombinationCount (n : Nat) :
  countWays = n :=
sorry

end NUMINAMATH_GPT_coinCombinationCount_l1441_144171


namespace NUMINAMATH_GPT_perpendicular_line_slopes_l1441_144190

theorem perpendicular_line_slopes (α₁ : ℝ) (hα₁ : α₁ = 30) (l₁ : ℝ) (k₁ : ℝ) (k₂ : ℝ) (α₂ : ℝ)
  (h₁ : k₁ = Real.tan (α₁ * Real.pi / 180))
  (h₂ : k₂ = - 1 / k₁)
  (h₃ : k₂ = - Real.sqrt 3)
  (h₄ : 0 < α₂ ∧ α₂ < 180)
  : k₂ = - Real.sqrt 3 ∧ α₂ = 120 := sorry

end NUMINAMATH_GPT_perpendicular_line_slopes_l1441_144190


namespace NUMINAMATH_GPT_calculate_expression_l1441_144125

-- Define the conditions
def exp1 : ℤ := (-1)^(53)
def exp2 : ℤ := 2^(2^4 + 5^2 - 4^3)

-- State and skip the proof
theorem calculate_expression :
  exp1 + exp2 = -1 + 1 / (2^23) :=
by sorry

#check calculate_expression

end NUMINAMATH_GPT_calculate_expression_l1441_144125


namespace NUMINAMATH_GPT_range_of_g_le_2_minus_x_l1441_144124

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ 0 then f x else -f (-x)

theorem range_of_g_le_2_minus_x : {x : ℝ | g x ≤ 2 - x} = {x : ℝ | x ≤ 1} :=
by sorry

end NUMINAMATH_GPT_range_of_g_le_2_minus_x_l1441_144124


namespace NUMINAMATH_GPT_negation_existential_proposition_l1441_144179

theorem negation_existential_proposition :
  ¬(∃ x : ℝ, x^2 - x + 1 = 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_existential_proposition_l1441_144179


namespace NUMINAMATH_GPT_point_after_transformations_l1441_144198

-- Define the initial coordinates of point F
def F : ℝ × ℝ := (-1, -1)

-- Function to reflect a point over the x-axis
def reflect_over_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Function to reflect a point over the line y = x
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Prove that F, when reflected over x-axis and then y=x, results in (1, -1)
theorem point_after_transformations : 
  reflect_over_y_eq_x (reflect_over_x F) = (1, -1) := by
  sorry

end NUMINAMATH_GPT_point_after_transformations_l1441_144198


namespace NUMINAMATH_GPT_calc_f_five_times_l1441_144105

def f (x : ℕ) : ℕ := if x % 2 = 0 then x / 2 else 5 * x + 1

theorem calc_f_five_times : f (f (f (f (f 5)))) = 166 :=
by 
  sorry

end NUMINAMATH_GPT_calc_f_five_times_l1441_144105


namespace NUMINAMATH_GPT_find_multiple_of_y_l1441_144168

noncomputable def multiple_of_y (q m : ℝ) : Prop :=
  ∀ x y : ℝ, (x = 5 - q) → (y = m * q - 1) → (q = 1) → (x = 3 * y) → (m = 7 / 3)

theorem find_multiple_of_y :
  multiple_of_y 1 (7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_y_l1441_144168


namespace NUMINAMATH_GPT_total_distance_correct_l1441_144113

def jonathan_distance : ℝ := 7.5

def mercedes_distance : ℝ := 2 * jonathan_distance

def davonte_distance : ℝ := mercedes_distance + 2

def total_distance : ℝ := mercedes_distance + davonte_distance

theorem total_distance_correct : total_distance = 32 := by
  rw [total_distance, mercedes_distance, davonte_distance]
  norm_num
  sorry

end NUMINAMATH_GPT_total_distance_correct_l1441_144113


namespace NUMINAMATH_GPT_mixture_volume_correct_l1441_144120

-- Define the input values
def water_volume : ℕ := 20
def vinegar_volume : ℕ := 18
def water_ratio : ℚ := 3/5
def vinegar_ratio : ℚ := 5/6

-- Calculate the mixture volume
def mixture_volume : ℚ :=
  (water_volume * water_ratio) + (vinegar_volume * vinegar_ratio)

-- Define the expected result
def expected_mixture_volume : ℚ := 27

-- State the theorem
theorem mixture_volume_correct : mixture_volume = expected_mixture_volume := by
  sorry

end NUMINAMATH_GPT_mixture_volume_correct_l1441_144120


namespace NUMINAMATH_GPT_least_integer_gt_sqrt_700_l1441_144185

theorem least_integer_gt_sqrt_700 : ∃ n : ℕ, (n - 1) < Real.sqrt 700 ∧ Real.sqrt 700 ≤ n ∧ n = 27 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_gt_sqrt_700_l1441_144185


namespace NUMINAMATH_GPT_inequality_solution_equality_condition_l1441_144177

theorem inequality_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) :=
sorry

theorem equality_condition (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d :=
sorry

end NUMINAMATH_GPT_inequality_solution_equality_condition_l1441_144177


namespace NUMINAMATH_GPT_find_a_l1441_144163

def star (a b : ℝ) : ℝ := 2 * a - b^3

theorem find_a (a : ℝ) : star a 3 = 15 → a = 21 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1441_144163


namespace NUMINAMATH_GPT_find_x_value_l1441_144145

/-- Given x, y, z such that x ≠ 0, z ≠ 0, (x / 2) = y^2 + z, and (x / 4) = 4y + 2z, the value of x is 120. -/
theorem find_x_value (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h1 : x / 2 = y^2 + z) (h2 : x / 4 = 4 * y + 2 * z) : x = 120 := 
sorry

end NUMINAMATH_GPT_find_x_value_l1441_144145


namespace NUMINAMATH_GPT_operation_correct_l1441_144102

def operation (x y : ℝ) := x^2 + y^2 + 12

theorem operation_correct :
  operation (Real.sqrt 6) (Real.sqrt 6) = 23.999999999999996 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_operation_correct_l1441_144102


namespace NUMINAMATH_GPT_num_perfect_square_factors_l1441_144143

-- Define the exponents and their corresponding number of perfect square factors
def num_square_factors (exp : ℕ) : ℕ := exp / 2 + 1

-- Define the product of the prime factorization
def product : ℕ := 2^12 * 3^15 * 7^18

-- State the theorem
theorem num_perfect_square_factors :
  (num_square_factors 12) * (num_square_factors 15) * (num_square_factors 18) = 560 := by
  sorry

end NUMINAMATH_GPT_num_perfect_square_factors_l1441_144143


namespace NUMINAMATH_GPT_g_eval_at_neg2_l1441_144132

def g (x : ℝ) : ℝ := x^3 + 2*x - 4

theorem g_eval_at_neg2 : g (-2) = -16 := by
  sorry

end NUMINAMATH_GPT_g_eval_at_neg2_l1441_144132


namespace NUMINAMATH_GPT_pct_three_petals_is_75_l1441_144196

-- Given Values
def total_clovers : Nat := 200
def pct_two_petals : Nat := 24
def pct_four_petals : Nat := 1

-- Statement: Prove that the percentage of clovers with three petals is 75%
theorem pct_three_petals_is_75 :
  (100 - pct_two_petals - pct_four_petals) = 75 := by
  sorry

end NUMINAMATH_GPT_pct_three_petals_is_75_l1441_144196


namespace NUMINAMATH_GPT_range_of_a_l1441_144131

-- Define conditions
def setA : Set ℝ := {x | x^2 - x ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- Problem statement in Lean 4
theorem range_of_a (a : ℝ) (h : setA ⊆ setB a) : a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1441_144131


namespace NUMINAMATH_GPT_min_sum_of_arithmetic_sequence_terms_l1441_144139

open Real

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ d : ℝ, a m = a n + d * (m - n)

theorem min_sum_of_arithmetic_sequence_terms (a : ℕ → ℝ) 
  (hpos : ∀ n, a n > 0) 
  (harith : arithmetic_sequence a) 
  (hprod : a 1 * a 20 = 100) : 
  a 7 + a 14 ≥ 20 := sorry

end NUMINAMATH_GPT_min_sum_of_arithmetic_sequence_terms_l1441_144139


namespace NUMINAMATH_GPT_triangle_shape_l1441_144188

theorem triangle_shape
  (A B C : ℝ) -- Internal angles of triangle ABC
  (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively
  (h1 : a * (Real.cos A) * (Real.cos B) + b * (Real.cos A) * (Real.cos A) = a * (Real.cos A)) :
  (A = Real.pi / 2) ∨ (A = C) :=
sorry

end NUMINAMATH_GPT_triangle_shape_l1441_144188


namespace NUMINAMATH_GPT_sine_cos_suffices_sine_cos_necessary_l1441_144103

theorem sine_cos_suffices
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) :
  c > Real.sqrt (a^2 + b^2) :=
sorry

theorem sine_cos_necessary
  (a b c : ℝ)
  (h : c > Real.sqrt (a^2 + b^2)) :
  ∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0 :=
sorry

end NUMINAMATH_GPT_sine_cos_suffices_sine_cos_necessary_l1441_144103


namespace NUMINAMATH_GPT_rectangle_aspect_ratio_l1441_144150

theorem rectangle_aspect_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y = 2 * y / x) : x / y = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_aspect_ratio_l1441_144150


namespace NUMINAMATH_GPT_triangle_inequality_check_l1441_144149

theorem triangle_inequality_check (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  (a = 5 ∧ b = 8 ∧ c = 12) → (a + b > c ∧ b + c > a ∧ c + a > b) :=
by 
  intros h
  rcases h with ⟨rfl, rfl, rfl⟩
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_triangle_inequality_check_l1441_144149


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_l1441_144135

theorem solve_eq1 (x : ℝ) : 5 * x - 2.9 = 12 → x = 1.82 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq2 (x : ℝ) : 10.5 * x + 0.6 * x = 44 → x = 3 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

theorem solve_eq3 (x : ℝ) : 8 * x / 2 = 1.5 → x = 0.375 :=
by
  intro h
  -- Additional steps to verify should be here
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_l1441_144135


namespace NUMINAMATH_GPT_least_non_lucky_multiple_of_8_l1441_144175

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (· + ·) 0

def lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def multiple_of_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem least_non_lucky_multiple_of_8 : ∃ n > 0, multiple_of_8 n ∧ ¬ lucky n ∧ n = 16 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_least_non_lucky_multiple_of_8_l1441_144175


namespace NUMINAMATH_GPT_calculate_distribution_l1441_144144

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end NUMINAMATH_GPT_calculate_distribution_l1441_144144


namespace NUMINAMATH_GPT_max_area_of_garden_l1441_144148

theorem max_area_of_garden (l w : ℝ) (h : l + 2*w = 270) : l * w ≤ 9112.5 :=
sorry

end NUMINAMATH_GPT_max_area_of_garden_l1441_144148


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1441_144173

def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem intersection_of_M_and_N :
  (M ∩ N = {0, 1}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1441_144173


namespace NUMINAMATH_GPT_fraction_unchanged_l1441_144169

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 * x) / (2 * (x + y)) = x / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_unchanged_l1441_144169


namespace NUMINAMATH_GPT_range_of_f_l1441_144158

open Real

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin x + cos x

theorem range_of_f :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ π/2 → - (sqrt 3) ≤ f x ∧ f x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l1441_144158


namespace NUMINAMATH_GPT_cubic_roots_fraction_l1441_144123

theorem cubic_roots_fraction 
  (a b c d : ℝ)
  (h_eq : ∀ x: ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ (x = -2 ∨ x = 3 ∨ x = 4)) :
  c / d = -1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_fraction_l1441_144123


namespace NUMINAMATH_GPT_math_proof_l1441_144108

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : Nat) : Nat :=
  (factorial n) / ((factorial k) * (factorial (n - k)))

theorem math_proof :
  binom 20 6 * factorial 6 = 27907200 :=
by
  sorry

end NUMINAMATH_GPT_math_proof_l1441_144108


namespace NUMINAMATH_GPT_divisor_of_3825_is_15_l1441_144184

theorem divisor_of_3825_is_15 : ∃ d, 3830 - 5 = 3825 ∧ 3825 % d = 0 ∧ d = 15 := by
  sorry

end NUMINAMATH_GPT_divisor_of_3825_is_15_l1441_144184


namespace NUMINAMATH_GPT_range_m_condition_l1441_144112

theorem range_m_condition {x y m : ℝ} (h1 : x^2 + (y - 1)^2 = 1) (h2 : x + y + m ≥ 0) : -1 < m :=
by
  sorry

end NUMINAMATH_GPT_range_m_condition_l1441_144112


namespace NUMINAMATH_GPT_additional_money_needed_for_free_shipping_l1441_144104

-- Define the prices of the books
def price_book1 : ℝ := 13.00
def price_book2 : ℝ := 15.00
def price_book3 : ℝ := 10.00
def price_book4 : ℝ := 10.00

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Calculate the discounted prices
def discounted_price_book1 : ℝ := price_book1 * (1 - discount_rate)
def discounted_price_book2 : ℝ := price_book2 * (1 - discount_rate)

-- Sum of discounted prices of books
def total_cost : ℝ := discounted_price_book1 + discounted_price_book2 + price_book3 + price_book4

-- Free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- The proof statement
theorem additional_money_needed_for_free_shipping : additional_amount = 9.00 := by
  -- calculation steps omitted
  sorry

end NUMINAMATH_GPT_additional_money_needed_for_free_shipping_l1441_144104


namespace NUMINAMATH_GPT_part1_intersection_part2_range_of_m_l1441_144116

-- Define the universal set and the sets A and B
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0 ∨ x > 3}
def B (m : ℝ) : Set ℝ := {x | x < m - 1 ∨ x > 2 * m}

-- Part (1): When m = 3, find A ∩ B
theorem part1_intersection:
  A ∩ B 3 = {x | x < 0 ∨ x > 6} :=
sorry

-- Part (2): If B ∪ A = B, find the range of values for m
theorem part2_range_of_m (m : ℝ) :
  (B m ∪ A = B m) → (1 ≤ m ∧ m ≤ 3 / 2) :=
sorry

end NUMINAMATH_GPT_part1_intersection_part2_range_of_m_l1441_144116


namespace NUMINAMATH_GPT_jamie_avg_is_correct_l1441_144164

-- Declare the set of test scores and corresponding sums
def test_scores : List ℤ := [75, 78, 82, 85, 88, 91]

-- Alex's average score
def alex_avg : ℤ := 82

-- Total test score sum
def total_sum : ℤ := test_scores.sum

theorem jamie_avg_is_correct (alex_sum : ℤ) :
    alex_sum = 3 * alex_avg →
    (total_sum - alex_sum) / 3 = 253 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jamie_avg_is_correct_l1441_144164


namespace NUMINAMATH_GPT_simplify_fraction_l1441_144100

theorem simplify_fraction (x : ℤ) :
  (2 * x - 3) / 4 + (3 * x + 5) / 5 - (x - 1) / 2 = (12 * x + 15) / 20 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1441_144100


namespace NUMINAMATH_GPT_avg_age_new_students_l1441_144137

-- Definitions for the conditions
def initial_avg_age : ℕ := 14
def initial_student_count : ℕ := 10
def new_student_count : ℕ := 5
def new_avg_age : ℕ := initial_avg_age + 1

-- Lean statement for the proof problem
theorem avg_age_new_students :
  (initial_avg_age * initial_student_count + new_avg_age * new_student_count) / new_student_count = 17 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_new_students_l1441_144137


namespace NUMINAMATH_GPT_largest_prime_factor_4851_l1441_144109

theorem largest_prime_factor_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 4851 → q ≤ p) :=
by
  -- todo: provide actual proof
  sorry

end NUMINAMATH_GPT_largest_prime_factor_4851_l1441_144109


namespace NUMINAMATH_GPT_result_when_7_multiplies_number_l1441_144170

theorem result_when_7_multiplies_number (x : ℤ) (h : x + 45 - 62 = 55) : 7 * x = 504 :=
by sorry

end NUMINAMATH_GPT_result_when_7_multiplies_number_l1441_144170


namespace NUMINAMATH_GPT_find_x_l1441_144121

theorem find_x (x : ℝ) (h : 15 * x + 16 * x + 19 * x + 11 = 161) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1441_144121


namespace NUMINAMATH_GPT_arithmetic_sequence_1001th_term_l1441_144115

theorem arithmetic_sequence_1001th_term (p q : ℤ)
  (h1 : 9 - p = (2 * q - 5))
  (h2 : (3 * p - q + 7) - 9 = (2 * q - 5)) :
  p + (1000 * (2 * q - 5)) = 5004 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_1001th_term_l1441_144115


namespace NUMINAMATH_GPT_column_of_2008_l1441_144157

theorem column_of_2008:
  (∃ k, 2008 = 2 * k) ∧
  ((2 % 8) = 2) ∧ ((4 % 8) = 4) ∧ ((6 % 8) = 6) ∧ ((8 % 8) = 0) ∧
  ((16 % 8) = 0) ∧ ((14 % 8) = 6) ∧ ((12 % 8) = 4) ∧ ((10 % 8) = 2) →
  (2008 % 8 = 4) :=
by
  sorry

end NUMINAMATH_GPT_column_of_2008_l1441_144157


namespace NUMINAMATH_GPT_systematic_sampling_interval_people_l1441_144133

theorem systematic_sampling_interval_people (total_employees : ℕ) (selected_employees : ℕ) (start_interval : ℕ) (end_interval : ℕ)
  (h_total : total_employees = 420)
  (h_selected : selected_employees = 21)
  (h_start_end : start_interval = 281)
  (h_end : end_interval = 420)
  : (end_interval - start_interval + 1) / (total_employees / selected_employees) = 7 := 
by
  -- sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_people_l1441_144133


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1441_144194

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = a n * r

-- Define the statement for the roots of the quadratic function
def is_root (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ :=
  x^2 - x - 2013

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_geometric_sequence a) 
  (h2 : is_root quadratic_function (a 2)) 
  (h3 : is_root quadratic_function (a 3)) : 
  a 1 * a 4 = -2013 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1441_144194


namespace NUMINAMATH_GPT_inequality_four_a_cubed_sub_l1441_144111

theorem inequality_four_a_cubed_sub (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  4 * a^3 * (a - b) ≥ a^4 - b^4 :=
sorry

end NUMINAMATH_GPT_inequality_four_a_cubed_sub_l1441_144111


namespace NUMINAMATH_GPT_minimum_value_expression_l1441_144197

theorem minimum_value_expression (a x1 x2 : ℝ) (h_pos : 0 < a)
  (h1 : x1 + x2 = 4 * a)
  (h2 : x1 * x2 = 3 * a^2)
  (h_ineq : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x1 < x ∧ x < x2) :
  x1 + x2 + a / (x1 * x2) = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l1441_144197


namespace NUMINAMATH_GPT_light_path_in_cube_l1441_144187

/-- Let ABCD and EFGH be two faces of a cube with AB = 10. A beam of light is emitted 
from vertex A and reflects off face EFGH at point Q, which is 6 units from EH and 4 
units from EF. The length of the light path from A until it reaches another vertex of 
the cube for the first time is expressed in the form s√t, where s and t are integers 
with t having no square factors. Provide s + t. -/
theorem light_path_in_cube :
  let AB := 10
  let s := 10
  let t := 152
  s + t = 162 := by
  sorry

end NUMINAMATH_GPT_light_path_in_cube_l1441_144187


namespace NUMINAMATH_GPT_stratified_sampling_l1441_144122

theorem stratified_sampling 
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (selected_first_grade : ℕ)
  (x : ℕ)
  (h1 : students_first_grade = 400)
  (h2 : students_second_grade = 360)
  (h3 : selected_first_grade = 60)
  (h4 : (selected_first_grade / students_first_grade : ℚ) = (x / students_second_grade : ℚ)) :
  x = 54 :=
sorry

end NUMINAMATH_GPT_stratified_sampling_l1441_144122


namespace NUMINAMATH_GPT_upper_limit_of_people_l1441_144127

theorem upper_limit_of_people (T : ℕ) (h1 : (3/7) * T = 24) (h2 : T > 50) : T ≤ 56 :=
by
  -- The steps to solve this proof would go here.
  sorry

end NUMINAMATH_GPT_upper_limit_of_people_l1441_144127


namespace NUMINAMATH_GPT_find_a_l1441_144172

noncomputable def f (x : ℝ) (a : ℝ) := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : ∀ x, f x a ≥ 5) (h₃ : ∃ x, f x a = 5) : a = 9 := by
  sorry

end NUMINAMATH_GPT_find_a_l1441_144172


namespace NUMINAMATH_GPT_rationalize_sqrt_fraction_l1441_144138

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_sqrt_fraction_l1441_144138


namespace NUMINAMATH_GPT_find_number_l1441_144118

variable (n : ℝ)

theorem find_number (h₁ : (0.47 * 1442 - 0.36 * n) + 63 = 3) : 
  n = 2049.28 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1441_144118


namespace NUMINAMATH_GPT_find_a5_of_geometric_sequence_l1441_144195

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = a n * r

theorem find_a5_of_geometric_sequence (a : ℕ → ℝ) (h : geometric_sequence a)
  (h₀ : a 1 = 1) (h₁ : a 9 = 3) : a 5 = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_a5_of_geometric_sequence_l1441_144195


namespace NUMINAMATH_GPT_euclid_1976_part_a_problem_4_l1441_144167

theorem euclid_1976_part_a_problem_4
  (p q y1 y2 : ℝ)
  (h1 : y1 = p * 1^2 + q * 1 + 5)
  (h2 : y2 = p * (-1)^2 + q * (-1) + 5)
  (h3 : y1 + y2 = 14) :
  p = 2 :=
by
  sorry

end NUMINAMATH_GPT_euclid_1976_part_a_problem_4_l1441_144167


namespace NUMINAMATH_GPT_children_total_savings_l1441_144183

theorem children_total_savings :
  let josiah_savings := 0.25 * 24
  let leah_savings := 0.50 * 20
  let megan_savings := (2 * 0.50) * 12
  josiah_savings + leah_savings + megan_savings = 28 := by
{
  -- lean proof goes here
  sorry
}

end NUMINAMATH_GPT_children_total_savings_l1441_144183


namespace NUMINAMATH_GPT_WallLengthBy40Men_l1441_144130

-- Definitions based on the problem conditions
def men1 : ℕ := 20
def length1 : ℕ := 112
def days1 : ℕ := 6

def men2 : ℕ := 40
variable (y : ℕ)  -- given 'y' days

-- Establish the relationship based on the given conditions
theorem WallLengthBy40Men :
  ∃ x : ℕ, x = (men2 / men1) * length1 * (y / days1) :=
by
  sorry

end NUMINAMATH_GPT_WallLengthBy40Men_l1441_144130


namespace NUMINAMATH_GPT_tan_alpha_value_trigonometric_expression_value_l1441_144136

theorem tan_alpha_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  Real.tan α = 2 :=
sorry

theorem trigonometric_expression_value (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (hα3 : Real.sin α = (2 * Real.sqrt 5) / 5) : 
  (4 * Real.sin (π - α) + 2 * Real.cos (2 * π - α)) / (Real.sin (π / 2 - α) + Real.sin (-α)) = -10 := 
sorry

end NUMINAMATH_GPT_tan_alpha_value_trigonometric_expression_value_l1441_144136


namespace NUMINAMATH_GPT_m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l1441_144141

-- Defining the sequence condition
def seq_condition (a : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n ≥ 2, a n ^ 2 - a (n + 1) * a (n - 1) = m * (a 2 - a 1) ^ 2

-- (1) Value of m for an arithmetic sequence with a non-zero common difference
theorem m_value_for_arithmetic_seq {a : ℕ → ℝ} (d : ℝ) (h_nonzero : d ≠ 0) :
  (∀ n, a (n + 1) = a n + d) → seq_condition a 1 :=
by
  sorry

-- (2) Minimum value of t given specific conditions
theorem min_value_t {t p : ℝ} (a : ℕ → ℝ) (h_p : 3 ≤ p ∧ p ≤ 5) :
  a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 4 ∧ (∀ n, t * a n + p ≥ n) → t = 1 / 32 :=
by
  sorry

-- (3) Smallest value of T for non-constant periodic sequence
theorem smallest_T_periodic_seq {a : ℕ → ℝ} {m : ℝ} (h_m_nonzero : m ≠ 0) :
  seq_condition a m → (∀ n, a (n + T) = a n) → (∃ T' > 0, ∀ T'', T'' > 0 → T'' = 3) :=
by
  sorry

end NUMINAMATH_GPT_m_value_for_arithmetic_seq_min_value_t_smallest_T_periodic_seq_l1441_144141


namespace NUMINAMATH_GPT_intersection_question_l1441_144159

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_question : M ∩ N = {1} :=
by sorry

end NUMINAMATH_GPT_intersection_question_l1441_144159


namespace NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l1441_144153

open Real

variables (a b c d : ℝ)

-- Assumptions
axiom a_neg : a < 0
axiom b_neg : b < 0
axiom c_pos : 0 < c
axiom d_pos : 0 < d
axiom abs_conditions : (0 < abs c) ∧ (abs c < 1) ∧ (abs b < 2) ∧ (1 < abs b) ∧ (1 < abs d) ∧ (abs d < 2) ∧ (abs a < 4) ∧ (2 < abs a)

-- Theorem Statements
theorem part_a : abs a < 4 := sorry
theorem part_b : abs b < 2 := sorry
theorem part_c : abs c < 2 := sorry
theorem part_d : abs a > abs b := sorry
theorem part_e : abs c < abs d := sorry
theorem part_f : ¬ (abs a < abs d) := sorry
theorem part_g : abs (a - b) < 4 := sorry
theorem part_h : ¬ (abs (a - b) ≥ 3) := sorry
theorem part_i : ¬ (abs (c - d) < 1) := sorry
theorem part_j : abs (b - c) < 2 := sorry
theorem part_k : ¬ (abs (b - c) > 3) := sorry
theorem part_m : abs (c - a) > 1 := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_part_d_part_e_part_f_part_g_part_h_part_i_part_j_part_k_part_m_l1441_144153


namespace NUMINAMATH_GPT_max_quadratic_function_l1441_144151

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

theorem max_quadratic_function : ∃ x, quadratic_function x = 7 ∧ ∀ x', quadratic_function x' ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_max_quadratic_function_l1441_144151
