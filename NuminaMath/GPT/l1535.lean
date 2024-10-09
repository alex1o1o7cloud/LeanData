import Mathlib

namespace football_problem_l1535_153564

-- Definitions based on conditions
def total_balls (x y : Nat) : Prop := x + y = 200
def total_cost (x y : Nat) : Prop := 80 * x + 60 * y = 14400
def football_A_profit_per_ball : Nat := 96 - 80
def football_B_profit_per_ball : Nat := 81 - 60
def total_profit (x y : Nat) : Nat :=
  football_A_profit_per_ball * x + football_B_profit_per_ball * y

-- Lean statement proving the conditions lead to the solution
theorem football_problem
  (x y : Nat)
  (h1 : total_balls x y)
  (h2 : total_cost x y)
  (h3 : x = 120)
  (h4 : y = 80) :
  total_profit x y = 3600 := by
  sorry

end football_problem_l1535_153564


namespace number_of_toothpicks_l1535_153503

def num_horizontal_toothpicks(lines width : Nat) : Nat := lines * width
def num_vertical_toothpicks(lines height : Nat) : Nat := lines * height

theorem number_of_toothpicks (high wide : Nat) (missing : Nat) 
  (h_high : high = 15) (h_wide : wide = 15) (h_missing : missing = 1) : 
  num_horizontal_toothpicks (high + 1) wide + num_vertical_toothpicks (wide + 1) high - missing = 479 := by
  sorry

end number_of_toothpicks_l1535_153503


namespace absolute_value_condition_l1535_153537

theorem absolute_value_condition (x : ℝ) (h : |x| = 32) : x = 32 ∨ x = -32 :=
sorry

end absolute_value_condition_l1535_153537


namespace base4_to_base10_conversion_l1535_153509

theorem base4_to_base10_conversion : 
  2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582 :=
by 
  sorry

end base4_to_base10_conversion_l1535_153509


namespace non_empty_solution_set_range_l1535_153558

theorem non_empty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 :=
sorry

end non_empty_solution_set_range_l1535_153558


namespace debut_show_tickets_l1535_153588

variable (P : ℕ) -- Number of people who bought tickets for the debut show

-- Conditions
def three_times_more (P : ℕ) : Bool := (3 * P = P + 2 * P)
def ticket_cost : ℕ := 25
def total_revenue (P : ℕ) : ℕ := 4 * P * ticket_cost

-- Main statement
theorem debut_show_tickets (h1 : three_times_more P = true) 
                           (h2 : total_revenue P = 20000) : P = 200 :=
by
  sorry

end debut_show_tickets_l1535_153588


namespace manager_salary_correct_l1535_153589

-- Define the conditions of the problem
def total_salary_of_24_employees : ℕ := 24 * 2400
def new_average_salary_with_manager : ℕ := 2500
def number_of_people_with_manager : ℕ := 25

-- Define the manager's salary to be proved
def managers_salary : ℕ := 4900

-- Statement of the theorem to prove that the manager's salary is Rs. 4900
theorem manager_salary_correct :
  (number_of_people_with_manager * new_average_salary_with_manager) - total_salary_of_24_employees = managers_salary :=
by
  -- Proof to be filled
  sorry

end manager_salary_correct_l1535_153589


namespace closely_related_interval_unique_l1535_153545

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

theorem closely_related_interval_unique :
  closely_related f g 2 3 :=
sorry

end closely_related_interval_unique_l1535_153545


namespace sushil_marks_ratio_l1535_153511

theorem sushil_marks_ratio
  (E M Science : ℕ)
  (h1 : E + M + Science = 170)
  (h2 : E = M / 4)
  (h3 : Science = 17) :
  E = 31 :=
by
  sorry

end sushil_marks_ratio_l1535_153511


namespace bullets_shot_per_person_l1535_153541

-- Definitions based on conditions
def num_people : ℕ := 5
def initial_bullets_per_person : ℕ := 25
def total_remaining_bullets : ℕ := 25

-- Statement to prove
theorem bullets_shot_per_person (x : ℕ) :
  (initial_bullets_per_person * num_people - num_people * x) = total_remaining_bullets → x = 20 :=
by
  sorry

end bullets_shot_per_person_l1535_153541


namespace roots_inequality_l1535_153592

theorem roots_inequality (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) :
  -1 ≤ z ∧ z ≤ 13 / 3 :=
sorry

end roots_inequality_l1535_153592


namespace picnic_recyclable_collected_l1535_153555

theorem picnic_recyclable_collected :
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  soda_drinkers + sparkling_water_drinkers + juice_consumed = 115 :=
by
  let guests := 90
  let soda_cans := 50
  let sparkling_water_bottles := 50
  let juice_bottles := 50
  let soda_drinkers := guests / 2
  let sparkling_water_drinkers := guests / 3
  let juice_consumed := juice_bottles * 4 / 5 
  show soda_drinkers + sparkling_water_drinkers + juice_consumed = 115
  sorry

end picnic_recyclable_collected_l1535_153555


namespace train_speed_is_72_kmph_l1535_153510

-- Define the given conditions in Lean
def crossesMan (L V : ℝ) : Prop := L = 19 * V
def crossesPlatform (L V : ℝ) : Prop := L + 220 = 30 * V

-- The main theorem which states that the speed of the train is 72 km/h under given conditions
theorem train_speed_is_72_kmph (L V : ℝ) (h1 : crossesMan L V) (h2 : crossesPlatform L V) :
  (V * 3.6) = 72 := by
  -- We will provide a full proof here later
  sorry

end train_speed_is_72_kmph_l1535_153510


namespace musical_chairs_l1535_153557

def is_prime_power (m : ℕ) : Prop :=
  ∃ (p k : ℕ), Nat.Prime p ∧ k > 0 ∧ m = p ^ k

theorem musical_chairs (n m : ℕ) (h1 : 1 < m) (h2 : m ≤ n) (h3 : ¬ is_prime_power m) :
  ∃ f : Fin n → Fin n, (∀ x, f x ≠ x) ∧ (∀ x, (f^[m]) x = x) :=
sorry

end musical_chairs_l1535_153557


namespace paperclips_exceed_target_in_days_l1535_153523

def initial_paperclips := 3
def ratio := 2
def target_paperclips := 200

theorem paperclips_exceed_target_in_days :
  ∃ k : ℕ, initial_paperclips * ratio ^ k > target_paperclips ∧ k = 8 :=
by {
  sorry
}

end paperclips_exceed_target_in_days_l1535_153523


namespace calc_fractional_product_l1535_153574

theorem calc_fractional_product (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b :=
by
  sorry

end calc_fractional_product_l1535_153574


namespace min_value_n_constant_term_l1535_153524

-- Define the problem statement
theorem min_value_n_constant_term (n r : ℕ) (h : 2 * n = 5 * r) : n = 5 :=
by sorry

end min_value_n_constant_term_l1535_153524


namespace B_pow_99_identity_l1535_153531

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem B_pow_99_identity : (B ^ 99) = 1 := by
  sorry

end B_pow_99_identity_l1535_153531


namespace hot_drinks_sales_l1535_153580

theorem hot_drinks_sales (x: ℝ) (h: x = 4) : abs ((-2.35 * x + 155.47) - 146) < 1 :=
by sorry

end hot_drinks_sales_l1535_153580


namespace triangle_inequality_l1535_153549

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l1535_153549


namespace hostel_food_duration_l1535_153501

noncomputable def food_last_days (total_food_units daily_consumption_new: ℝ) : ℝ :=
  total_food_units / daily_consumption_new

theorem hostel_food_duration:
  let x : ℝ := 1 -- assuming x is a positive real number
  let men_initial := 100
  let women_initial := 100
  let children_initial := 50
  let total_days := 40
  let consumption_man := 3 * x
  let consumption_woman := 2 * x
  let consumption_child := 1 * x
  let food_sufficient_for := 250
  let total_food_units := 550 * x * 40
  let men_leave := 30
  let women_leave := 20
  let children_leave := 10
  let men_new := men_initial - men_leave
  let women_new := women_initial - women_leave
  let children_new := children_initial - children_leave
  let daily_consumption_new := 210 * x + 160 * x + 40 * x 
  (food_last_days total_food_units daily_consumption_new) = 22000 / 410 := 
by
  sorry

end hostel_food_duration_l1535_153501


namespace largest_pos_int_divisible_l1535_153526

theorem largest_pos_int_divisible (n : ℕ) (h1 : n > 0) (h2 : n + 11 ∣ n^3 + 101) : n = 1098 :=
sorry

end largest_pos_int_divisible_l1535_153526


namespace probability_red_or_blue_is_713_l1535_153514

-- Definition of area ratios
def area_ratio_red : ℕ := 6
def area_ratio_yellow : ℕ := 2
def area_ratio_blue : ℕ := 1
def area_ratio_black : ℕ := 4

-- Total area ratio
def total_area_ratio := area_ratio_red + area_ratio_yellow + area_ratio_blue + area_ratio_black

-- Probability of stopping on either red or blue
def probability_red_or_blue := (area_ratio_red + area_ratio_blue) / total_area_ratio

-- Theorem stating the probability is 7/13
theorem probability_red_or_blue_is_713 : probability_red_or_blue = 7 / 13 :=
by
  unfold probability_red_or_blue total_area_ratio area_ratio_red area_ratio_blue
  simp
  sorry

end probability_red_or_blue_is_713_l1535_153514


namespace horner_eval_at_2_l1535_153561

def poly (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_eval_at_2 : poly 2 = 373 := by
  sorry

end horner_eval_at_2_l1535_153561


namespace solutions_are__l1535_153596

def satisfies_system (x y z : ℝ) : Prop :=
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540

theorem solutions_are_ (x y z : ℝ) :
  satisfies_system x y z ↔ (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by
  sorry

end solutions_are__l1535_153596


namespace amy_local_calls_l1535_153575

-- Define the conditions as hypotheses
variable (L I : ℕ)
variable (h1 : L = (5 / 2 : ℚ) * I)
variable (h2 : L = (5 / 3 : ℚ) * (I + 3))

-- Statement of the theorem
theorem amy_local_calls : L = 15 := by
  sorry

end amy_local_calls_l1535_153575


namespace length_of_MN_l1535_153547

theorem length_of_MN (b : ℝ) (h_focus : ∃ b : ℝ, (3/2, b).1 > 0 ∧ (3/2, b).2 * (3/2, b).2 = 6 * (3 / 2)) : 
  |2 * b| = 6 :=
by sorry

end length_of_MN_l1535_153547


namespace additional_employees_hired_l1535_153519

-- Conditions
def initial_employees : ℕ := 500
def hourly_wage : ℕ := 12
def daily_hours : ℕ := 10
def weekly_days : ℕ := 5
def weekly_hours := daily_hours * weekly_days
def monthly_weeks : ℕ := 4
def monthly_hours_per_employee := weekly_hours * monthly_weeks
def wage_per_employee_per_month := monthly_hours_per_employee * hourly_wage

-- Given new payroll
def new_monthly_payroll : ℕ := 1680000

-- Calculate the initial payroll
def initial_monthly_payroll := initial_employees * wage_per_employee_per_month

-- Statement of the proof problem
theorem additional_employees_hired :
  (new_monthly_payroll - initial_monthly_payroll) / wage_per_employee_per_month = 200 :=
by
  sorry

end additional_employees_hired_l1535_153519


namespace atLeastOneTrueRange_exactlyOneTrueRange_l1535_153560

-- Definitions of Proposition A and B
def propA (a : ℝ) : Prop := ∀ x, x^2 + (a - 1) * x + a^2 ≤ 0 → false
def propB (a : ℝ) : Prop := ∀ x, (2 * a^2 - a)^x < (2 * a^2 - a)^(x + 1)

-- At least one of A or B is true
def atLeastOneTrue (a : ℝ) : Prop :=
  propA a ∨ propB a

-- Exactly one of A or B is true
def exactlyOneTrue (a : ℝ) : Prop := 
  (propA a ∧ ¬ propB a) ∨ (¬ propA a ∧ propB a)

-- Theorems to prove
theorem atLeastOneTrueRange :
  ∃ a : ℝ, atLeastOneTrue a ↔ (a < -1/2 ∨ a > 1/3) := 
sorry

theorem exactlyOneTrueRange :
  ∃ a : ℝ, exactlyOneTrue a ↔ ((1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2)) :=
sorry

end atLeastOneTrueRange_exactlyOneTrueRange_l1535_153560


namespace binary_div_remainder_l1535_153591

theorem binary_div_remainder (n : ℕ) (h : n = 0b101011100101) : n % 8 = 5 :=
by sorry

end binary_div_remainder_l1535_153591


namespace investment_calculation_l1535_153540

theorem investment_calculation
  (face_value : ℝ)
  (market_price : ℝ)
  (rate_of_dividend : ℝ)
  (annual_income : ℝ)
  (h1 : face_value = 10)
  (h2 : market_price = 8.25)
  (h3 : rate_of_dividend = 12)
  (h4 : annual_income = 648) :
  ∃ investment : ℝ, investment = 4455 :=
by
  sorry

end investment_calculation_l1535_153540


namespace fraction_of_juniors_l1535_153515

theorem fraction_of_juniors (J S : ℕ) (h1 : 0 < J) (h2 : 0 < S) (h3 : J = (4 / 3) * S) :
  (J : ℚ) / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l1535_153515


namespace sandy_grew_6_carrots_l1535_153563

theorem sandy_grew_6_carrots (sam_grew : ℕ) (total_grew : ℕ) (h1 : sam_grew = 3) (h2 : total_grew = 9) : ∃ sandy_grew : ℕ, sandy_grew = total_grew - sam_grew ∧ sandy_grew = 6 :=
by
  sorry

end sandy_grew_6_carrots_l1535_153563


namespace find_m_l1535_153528

theorem find_m (m n : ℤ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 0 :=
sorry

end find_m_l1535_153528


namespace no_solution_for_lcm_gcd_eq_l1535_153593

theorem no_solution_for_lcm_gcd_eq (n : ℕ) (h₁ : n ∣ 60) (h₂ : Nat.Prime n) :
  ¬(Nat.lcm n 60 = Nat.gcd n 60 + 200) :=
  sorry

end no_solution_for_lcm_gcd_eq_l1535_153593


namespace maximum_xyzw_l1535_153559

theorem maximum_xyzw (x y z w : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_w : 0 < w)
(h : (x * y * z) + w = (x + w) * (y + w) * (z + w))
(h_sum : x + y + z + w = 1) :
  xyzw = 1 / 256 :=
sorry

end maximum_xyzw_l1535_153559


namespace units_digit_lucas_L10_is_4_l1535_153569

def lucas : ℕ → ℕ 
  | 0 => 2
  | 1 => 1
  | n + 2 => lucas (n + 1) + lucas n

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_lucas_L10_is_4 : units_digit (lucas (lucas 10)) = 4 := 
  sorry

end units_digit_lucas_L10_is_4_l1535_153569


namespace total_cupcakes_l1535_153581

-- Definitions of initial conditions
def cupcakes_initial : ℕ := 42
def cupcakes_sold : ℕ := 22
def cupcakes_made_after : ℕ := 39

-- Proof statement: Total number of cupcakes Robin would have
theorem total_cupcakes : 
  (cupcakes_initial - cupcakes_sold + cupcakes_made_after) = 59 := by
    sorry

end total_cupcakes_l1535_153581


namespace powderman_distance_approximates_275_yards_l1535_153590

noncomputable def distance_run (t : ℝ) : ℝ := 6 * t
noncomputable def sound_distance (t : ℝ) : ℝ := 1080 * (t - 45) / 3

theorem powderman_distance_approximates_275_yards : 
  ∃ t : ℝ, t > 45 ∧ 
  (distance_run t = sound_distance t) → 
  abs (distance_run t - 275) < 1 :=
by
  sorry

end powderman_distance_approximates_275_yards_l1535_153590


namespace range_of_k_l1535_153512

theorem range_of_k (k : ℝ) : (4 < k ∧ k < 9 ∧ k ≠ 13 / 2) ↔ (k ∈ Set.Ioo 4 (13 / 2) ∪ Set.Ioo (13 / 2) 9) :=
by
  sorry

end range_of_k_l1535_153512


namespace exists_natural_numbers_solving_equation_l1535_153542

theorem exists_natural_numbers_solving_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_natural_numbers_solving_equation_l1535_153542


namespace fractions_equiv_conditions_l1535_153529

theorem fractions_equiv_conditions (x y z : ℝ) (h₁ : 2 * x - z ≠ 0) (h₂ : z ≠ 0) : 
  ((2 * x + y) / (2 * x - z) = y / -z) ↔ (y = -z) :=
by
  sorry

end fractions_equiv_conditions_l1535_153529


namespace sqrt_14_bounds_l1535_153534

theorem sqrt_14_bounds : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end sqrt_14_bounds_l1535_153534


namespace length_of_first_video_l1535_153516

theorem length_of_first_video
  (total_time : ℕ)
  (second_video_time : ℕ)
  (last_two_videos_time : ℕ)
  (first_video_time : ℕ)
  (total_seconds : total_time = 510)
  (second_seconds : second_video_time = 4 * 60 + 30)
  (last_videos_seconds : last_two_videos_time = 60 + 60)
  (total_watch_time : total_time = second_video_time + last_two_videos_time + first_video_time) :
  first_video_time = 120 :=
by
  sorry

end length_of_first_video_l1535_153516


namespace calc_eq_neg_ten_thirds_l1535_153530

theorem calc_eq_neg_ten_thirds :
  (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8) + (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12) = -10 / 3 := by 
sorry

end calc_eq_neg_ten_thirds_l1535_153530


namespace handbag_monday_price_l1535_153525

theorem handbag_monday_price (initial_price : ℝ) (primary_discount : ℝ) (additional_discount : ℝ)
(h_initial_price : initial_price = 250)
(h_primary_discount : primary_discount = 0.4)
(h_additional_discount : additional_discount = 0.1) :
(initial_price - initial_price * primary_discount) - ((initial_price - initial_price * primary_discount) * additional_discount) = 135 := by
  sorry

end handbag_monday_price_l1535_153525


namespace simple_interest_time_period_l1535_153543

variable (SI P R T : ℝ)

theorem simple_interest_time_period (h₁ : SI = 4016.25) (h₂ : P = 8925) (h₃ : R = 9) :
  (P * R * T) / 100 = SI ↔ T = 5 := by
  sorry

end simple_interest_time_period_l1535_153543


namespace evaluate_expression_l1535_153576

theorem evaluate_expression (x : ℝ) (h : 3 * x^3 - x = 1) : 9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2001 := 
by
  sorry

end evaluate_expression_l1535_153576


namespace sam_dads_dimes_l1535_153507

theorem sam_dads_dimes (original_dimes new_dimes given_dimes : ℕ) 
  (h1 : original_dimes = 9)
  (h2 : new_dimes = 16)
  (h3 : new_dimes = original_dimes + given_dimes) : 
  given_dimes = 7 := 
by 
  sorry

end sam_dads_dimes_l1535_153507


namespace dog_food_vs_cat_food_l1535_153527

-- Define the quantities of dog food and cat food
def dog_food : ℕ := 600
def cat_food : ℕ := 327

-- Define the problem as a statement asserting the required difference
theorem dog_food_vs_cat_food : dog_food - cat_food = 273 := by
  sorry

end dog_food_vs_cat_food_l1535_153527


namespace incorrect_statement_l1535_153577

-- Define the general rules of program flowcharts
def isValidStart (box : String) : Prop := box = "start"
def isValidEnd (box : String) : Prop := box = "end"
def isInputBox (box : String) : Prop := box = "input"
def isOutputBox (box : String) : Prop := box = "output"

-- Define the statement to be proved incorrect
def statement (boxes : List String) : Prop :=
  ∀ xs ys, boxes = xs ++ ["start", "input"] ++ ys ->
           ∀ zs ws, boxes = zs ++ ["output", "end"] ++ ws

-- The target theorem stating that the statement is incorrect
theorem incorrect_statement (boxes : List String) :
  ¬ statement boxes :=
sorry

end incorrect_statement_l1535_153577


namespace min_value_z_l1535_153505

theorem min_value_z : ∃ (min_z : ℝ), min_z = 24.1 ∧ 
  ∀ (x y : ℝ), (3 * x ^ 2 + 4 * y ^ 2 + 8 * x - 6 * y + 30) ≥ min_z :=
sorry

end min_value_z_l1535_153505


namespace unique_x_intersect_l1535_153532

theorem unique_x_intersect (m : ℝ) (h : ∀ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 → ∀ y : ℝ, (m - 4) * y^2 - 2 * m * y - m - 6 = 0 → x = y) :
  m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end unique_x_intersect_l1535_153532


namespace fraction_value_l1535_153502

theorem fraction_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by sorry

end fraction_value_l1535_153502


namespace percentage_sold_correct_l1535_153548

variables 
  (initial_cost : ℝ) 
  (tripled_value : ℝ) 
  (selling_price : ℝ) 
  (percentage_sold : ℝ)

def game_sold_percentage (initial_cost tripled_value selling_price percentage_sold : ℝ) :=
  tripled_value = initial_cost * 3 ∧ 
  selling_price = 240 ∧ 
  initial_cost = 200 ∧ 
  percentage_sold = (selling_price / tripled_value) * 100

theorem percentage_sold_correct : game_sold_percentage 200 (200 * 3) 240 40 :=
  by simp [game_sold_percentage]; sorry

end percentage_sold_correct_l1535_153548


namespace age_of_father_l1535_153599

theorem age_of_father (F C : ℕ) 
  (h1 : F = C)
  (h2 : C + 5 * 15 = 2 * (F + 15)) : 
  F = 45 := 
by 
sorry

end age_of_father_l1535_153599


namespace square_area_l1535_153568

theorem square_area (x : ℝ) (s : ℝ) 
  (h1 : s^2 + s^2 = (2 * x)^2) 
  (h2 : 4 * s = 16 * x) : s^2 = 16 * x^2 :=
by {
  sorry -- Proof not required
}

end square_area_l1535_153568


namespace coins_after_tenth_hour_l1535_153584

-- Given variables representing the number of coins added or removed each hour.
def coins_put_in : ℕ :=
  20 + 30 + 30 + 40 + 50 + 60 + 70

def coins_taken_out : ℕ :=
  20 + 15 + 25

-- Definition of the full proof problem
theorem coins_after_tenth_hour :
  coins_put_in - coins_taken_out = 240 :=
by
  sorry

end coins_after_tenth_hour_l1535_153584


namespace filling_time_calculation_l1535_153517

namespace TankerFilling

-- Define the filling rates
def fill_rate_A : ℚ := 1 / 60
def fill_rate_B : ℚ := 1 / 40
def combined_fill_rate : ℚ := fill_rate_A + fill_rate_B

-- Define the time variable
variable (T : ℚ)

-- State the theorem to be proved
theorem filling_time_calculation
  (h_fill_rate_A : fill_rate_A = 1 / 60)
  (h_fill_rate_B : fill_rate_B = 1 / 40)
  (h_combined_fill_rate : combined_fill_rate = 1 / 24) :
  (fill_rate_B * (T / 2) + combined_fill_rate * (T / 2)) = 1 → T = 30 :=
by
  intros h
  -- Proof will go here
  sorry

end TankerFilling

end filling_time_calculation_l1535_153517


namespace second_date_sum_eq_80_l1535_153594

theorem second_date_sum_eq_80 (a1 a2 a3 a4 a5 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 = 80)
  (h2 : a2 = a1 + 1) (h3 : a3 = a2 + 1) (h4 : a4 = a3 + 1) (h5 : a5 = a4 + 1): a2 = 15 :=
by
  sorry

end second_date_sum_eq_80_l1535_153594


namespace average_time_per_mile_l1535_153566

-- Define the conditions
def total_distance_miles : ℕ := 24
def total_time_hours : ℕ := 3
def total_time_minutes : ℕ := 36
def total_time_in_minutes : ℕ := (total_time_hours * 60) + total_time_minutes

-- State the theorem
theorem average_time_per_mile : total_time_in_minutes / total_distance_miles = 9 :=
by
  sorry

end average_time_per_mile_l1535_153566


namespace xy_square_sum_l1535_153551

variable (x y : ℝ)

theorem xy_square_sum : (y + 6 = (x - 3)^2) →
                        (x + 6 = (y - 3)^2) →
                        (x ≠ y) →
                        x^2 + y^2 = 43 :=
by
  intros h₁ h₂ h₃
  sorry

end xy_square_sum_l1535_153551


namespace blue_balls_needed_l1535_153550

theorem blue_balls_needed 
  (G B Y W : ℝ)
  (h1 : G = 2 * B)
  (h2 : Y = (8 / 3) * B)
  (h3 : W = (4 / 3) * B) :
  5 * G + 3 * Y + 4 * W = (70 / 3) * B :=
by
  sorry

end blue_balls_needed_l1535_153550


namespace sphere_surface_area_l1535_153513

theorem sphere_surface_area (R h : ℝ) (R_pos : 0 < R) (h_pos : 0 < h) :
  ∃ A : ℝ, A = 2 * Real.pi * R * h := 
sorry

end sphere_surface_area_l1535_153513


namespace units_digit_of_m3_plus_2m_l1535_153520

def m : ℕ := 2021^2 + 2^2021

theorem units_digit_of_m3_plus_2m : (m^3 + 2^m) % 10 = 5 := by
  sorry

end units_digit_of_m3_plus_2m_l1535_153520


namespace clarence_oranges_after_giving_l1535_153571

def initial_oranges : ℝ := 5.0
def oranges_given : ℝ := 3.0

theorem clarence_oranges_after_giving : (initial_oranges - oranges_given) = 2.0 :=
by
  sorry

end clarence_oranges_after_giving_l1535_153571


namespace equivalent_multipliers_l1535_153565

variable (a b : ℝ)

theorem equivalent_multipliers (a b : ℝ) :
  let a_final := 0.93 * a
  let expr := a_final + 0.05 * b
  expr = 0.93 * a + 0.05 * b  :=
by
  -- Proof placeholder
  sorry

end equivalent_multipliers_l1535_153565


namespace tax_amount_is_correct_l1535_153544

def camera_cost : ℝ := 200.00
def tax_rate : ℝ := 0.15

theorem tax_amount_is_correct :
  (camera_cost * tax_rate) = 30.00 :=
sorry

end tax_amount_is_correct_l1535_153544


namespace find_number_l1535_153586

theorem find_number : ∃ x : ℝ, 3550 - (1002 / x) = 3500 ∧ x = 20.04 :=
by
  sorry

end find_number_l1535_153586


namespace fraction_subtraction_simplified_l1535_153585

theorem fraction_subtraction_simplified : (7 / 17) - (4 / 51) = 1 / 3 := by
  sorry

end fraction_subtraction_simplified_l1535_153585


namespace problem_1_part_1_proof_problem_1_part_2_proof_l1535_153598

noncomputable def problem_1_part_1 : Real :=
  2 * Real.sqrt 2 + (Real.sqrt 6) / 2

theorem problem_1_part_1_proof:
  let θ₀ := 3 * Real.pi / 4
  let ρ_A := 4 * Real.cos θ₀
  let ρ_B := Real.sqrt 3 * Real.sin θ₀
  |ρ_A - ρ_B| = 2 * Real.sqrt 2 + (Real.sqrt 6) / 2 :=
  sorry

theorem problem_1_part_2_proof :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x - (Real.sqrt 3)/2 * y = 0) :=
  sorry

end problem_1_part_1_proof_problem_1_part_2_proof_l1535_153598


namespace six_digit_number_count_correct_l1535_153508

-- Defining the 6-digit number formation problem
def count_six_digit_numbers_with_conditions : Nat := 1560

-- Problem statement
theorem six_digit_number_count_correct :
  count_six_digit_numbers_with_conditions = 1560 :=
sorry

end six_digit_number_count_correct_l1535_153508


namespace shipping_cost_correct_l1535_153506

noncomputable def shipping_cost (W : ℝ) : ℕ := 7 + 5 * (⌈W⌉₊ - 1)

theorem shipping_cost_correct (W : ℝ) : shipping_cost W = 5 * ⌈W⌉₊ + 2 :=
by
  sorry

end shipping_cost_correct_l1535_153506


namespace regular_polygon_sides_l1535_153500

theorem regular_polygon_sides (h : ∀ n : ℕ, 140 * n = 180 * (n - 2)) : n = 9 :=
sorry

end regular_polygon_sides_l1535_153500


namespace Rikki_earnings_l1535_153518

theorem Rikki_earnings
  (price_per_word : ℝ := 0.01)
  (words_per_5_minutes : ℕ := 25)
  (total_minutes : ℕ := 120)
  (earning : ℝ := 6)
  : price_per_word * (words_per_5_minutes * (total_minutes / 5)) = earning := by
  sorry

end Rikki_earnings_l1535_153518


namespace sum_of_digits_third_smallest_multiple_l1535_153578

noncomputable def LCM_upto_7 : ℕ := Nat.lcm (Nat.lcm 1 2) (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))

noncomputable def third_smallest_multiple : ℕ := 3 * LCM_upto_7

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_third_smallest_multiple : sum_of_digits third_smallest_multiple = 9 := 
sorry

end sum_of_digits_third_smallest_multiple_l1535_153578


namespace painter_total_rooms_l1535_153535

theorem painter_total_rooms (hours_per_room : ℕ) (rooms_already_painted : ℕ) (additional_painting_hours : ℕ) 
  (h1 : hours_per_room = 8) (h2 : rooms_already_painted = 8) (h3 : additional_painting_hours = 16) : 
  rooms_already_painted + (additional_painting_hours / hours_per_room) = 10 := by
  sorry

end painter_total_rooms_l1535_153535


namespace convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l1535_153538

theorem convert_deg_to_rad1 : 780 * (Real.pi / 180) = (13 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad2 : -1560 * (Real.pi / 180) = -(26 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad3 : 67.5 * (Real.pi / 180) = (3 * Real.pi) / 8 := sorry
theorem convert_rad_to_deg1 : -(10 * Real.pi / 3) * (180 / Real.pi) = -600 := sorry
theorem convert_rad_to_deg2 : (Real.pi / 12) * (180 / Real.pi) = 15 := sorry
theorem convert_rad_to_deg3 : (7 * Real.pi / 4) * (180 / Real.pi) = 315 := sorry

end convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l1535_153538


namespace quadratic_always_positive_l1535_153552

theorem quadratic_always_positive (k : ℝ) :
  ∀ x : ℝ, x^2 - (k - 4) * x + k - 7 > 0 :=
sorry

end quadratic_always_positive_l1535_153552


namespace multiplication_identity_l1535_153522

theorem multiplication_identity : 32519 * 9999 = 324857481 := by
  sorry

end multiplication_identity_l1535_153522


namespace fractional_expression_evaluation_l1535_153587

theorem fractional_expression_evaluation (a : ℝ) (h : a^3 + 3 * a^2 + a = 0) :
  ∃ b : ℝ, b = 0 ∨ b = 1 ∧ b = 2022 * a^2 / (a^4 + 2015 * a^2 + 1) :=
by
  sorry

end fractional_expression_evaluation_l1535_153587


namespace common_fraction_l1535_153567

noncomputable def x : ℚ := 0.6666666 -- represents 0.\overline{6}
noncomputable def y : ℚ := 0.2222222 -- represents 0.\overline{2}
noncomputable def z : ℚ := 0.4444444 -- represents 0.\overline{4}

theorem common_fraction :
  x + y - z = 4 / 9 :=
by
  -- Provide proofs here
  sorry

end common_fraction_l1535_153567


namespace rectangle_area_comparison_l1535_153583

theorem rectangle_area_comparison 
  {A A' B B' C C' D D': ℝ} 
  (h_A: A ≤ A') 
  (h_B: B ≤ B') 
  (h_C: C ≤ C') 
  (h_D: D ≤ B') : 
  A + B + C + D ≤ A' + B' + C' + D' := 
by 
  sorry

end rectangle_area_comparison_l1535_153583


namespace avg_of_other_two_l1535_153533

-- Definitions and conditions from the problem
def avg (l : List ℕ) : ℕ := l.sum / l.length

variables {A B C D E : ℕ}
variables (h_avg_five : avg [A, B, C, D, E] = 20)
variables (h_sum_three : A + B + C = 48)
variables (h_twice : A = 2 * B)

-- Theorem to prove
theorem avg_of_other_two (A B C D E : ℕ) 
  (h_avg_five : avg [A, B, C, D, E] = 20)
  (h_sum_three : A + B + C = 48)
  (h_twice : A = 2 * B) :
  avg [D, E] = 26 := 
  sorry

end avg_of_other_two_l1535_153533


namespace value_of_b_l1535_153572

theorem value_of_b (y : ℝ) (b : ℝ) (h_pos : y > 0) (h_eqn : (7 * y) / b + (3 * y) / 10 = 0.6499999999999999 * y) : 
  b = 70 / 61.99999999999999 :=
sorry

end value_of_b_l1535_153572


namespace zero_in_interval_l1535_153595

theorem zero_in_interval (x y : ℝ) (hx_lt_0 : x < 0) (hy_gt_0 : 0 < y) (hy_lt_1 : y < 1) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) : x^5 < 0 ∧ 0 < y^8 :=
by
  sorry

end zero_in_interval_l1535_153595


namespace sticks_predict_good_fortune_l1535_153556

def good_fortune_probability := 11 / 12

theorem sticks_predict_good_fortune:
  (∃ (α β: ℝ), 0 ≤ α ∧ α ≤ π / 2 ∧ 0 ≤ β ∧ β ≤ π / 2 ∧ (0 ≤ β ∧ β < π - α) ∧ (0 ≤ α ∧ α < π - β)) → 
  good_fortune_probability = 11 / 12 :=
sorry

end sticks_predict_good_fortune_l1535_153556


namespace n_four_plus_n_squared_plus_one_not_prime_l1535_153553

theorem n_four_plus_n_squared_plus_one_not_prime (n : ℤ) (h : n ≥ 2) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end n_four_plus_n_squared_plus_one_not_prime_l1535_153553


namespace boat_speed_greater_than_current_l1535_153539

theorem boat_speed_greater_than_current (U V : ℝ) (hU_gt_V : U > V)
  (h_equation : 1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_current_l1535_153539


namespace average_age_of_two_new_men_l1535_153546

theorem average_age_of_two_new_men :
  ∀ (A N : ℕ), 
    (∀ n : ℕ, n = 12) → 
    (N = 21 + 23 + 12) → 
    (A = N / 2) → 
    A = 28 :=
by
  intros A N twelve men_replace_eq_avg men_avg_eq
  sorry

end average_age_of_two_new_men_l1535_153546


namespace imaginary_part_of_conjugate_l1535_153573

def complex_conjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

theorem imaginary_part_of_conjugate :
  ∀ (z : ℂ), z = (1+i)^2 / (1-i) → (complex_conjugate z).im = -1 :=
by
  sorry

end imaginary_part_of_conjugate_l1535_153573


namespace masha_happy_max_l1535_153597

/-- Masha has 2021 weights, all with unique masses. She places weights one at a 
time on a two-pan balance scale without removing previously placed weights. 
Every time the scale balances, Masha feels happy. Prove that the maximum number 
of times she can find the scales in perfect balance is 673. -/
theorem masha_happy_max (weights : Finset ℕ) (h_unique : weights.card = 2021) : 
  ∃ max_happy_times : ℕ, max_happy_times = 673 := 
sorry

end masha_happy_max_l1535_153597


namespace probability_C_and_D_l1535_153536

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l1535_153536


namespace minimum_value_of_f_l1535_153570

noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem minimum_value_of_f :
  ∃ x ∈ Set.Icc (1/4 : ℝ) (5/4 : ℝ), f x = (4 * Real.sqrt 5 / 5 - 2 * Real.sqrt 10 / 5) :=
sorry

end minimum_value_of_f_l1535_153570


namespace count_possible_x_values_l1535_153504

theorem count_possible_x_values (x y : ℕ) (H : (x + 2) * (y + 2) - x * y = x * y) :
  (∃! x, ∃ y, (x - 2) * (y - 2) = 8) :=
by {
  sorry
}

end count_possible_x_values_l1535_153504


namespace seashells_total_l1535_153562

theorem seashells_total :
  let sally := 9.5
  let tom := 7.2
  let jessica := 5.3
  let alex := 12.8
  sally + tom + jessica + alex = 34.8 :=
by
  sorry

end seashells_total_l1535_153562


namespace sums_solved_correctly_l1535_153582

theorem sums_solved_correctly (x : ℕ) (h : x + 2 * x = 48) : x = 16 := by
  sorry

end sums_solved_correctly_l1535_153582


namespace smallest_floor_sum_l1535_153554

theorem smallest_floor_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (⌊(a + b + d) / c⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + a + d) / b⌋) = 9 :=
sorry

end smallest_floor_sum_l1535_153554


namespace tom_current_yellow_tickets_l1535_153521

-- Definitions based on conditions provided
def yellow_to_red (y : ℕ) : ℕ := y * 10
def red_to_blue (r : ℕ) : ℕ := r * 10
def yellow_to_blue (y : ℕ) : ℕ := (yellow_to_red y) * 10

def tom_red_tickets : ℕ := 3
def tom_blue_tickets : ℕ := 7

def tom_total_blue_tickets : ℕ := (red_to_blue tom_red_tickets) + tom_blue_tickets
def tom_needed_blue_tickets : ℕ := 163

-- Proving that Tom currently has 2 yellow tickets
theorem tom_current_yellow_tickets : (tom_total_blue_tickets + tom_needed_blue_tickets) / yellow_to_blue 1 = 2 :=
by
  sorry

end tom_current_yellow_tickets_l1535_153521


namespace parallelogram_sides_l1535_153579

theorem parallelogram_sides (x y : ℝ) (h1 : 12 * y - 2 = 10) (h2 : 5 * x + 15 = 20) : x + y = 2 :=
by
  sorry

end parallelogram_sides_l1535_153579
