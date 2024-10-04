import Mathlib

namespace complex_sum_power_l212_212314

noncomputable def z : ℂ := sorry

theorem complex_sum_power (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 :=
sorry

end complex_sum_power_l212_212314


namespace solve_system_of_eq_l212_212325

noncomputable def system_of_eq (x y z : ℝ) : Prop :=
  y = x^3 * (3 - 2 * x) ∧
  z = y^3 * (3 - 2 * y) ∧
  x = z^3 * (3 - 2 * z)

theorem solve_system_of_eq (x y z : ℝ) :
  system_of_eq x y z →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1/2 ∧ y = -1/2 ∧ z = -1/2) :=
sorry

end solve_system_of_eq_l212_212325


namespace sum_of_digits_d_l212_212871

theorem sum_of_digits_d (d : ℕ) (exchange_rate : 10 * d / 7 - 60 = d) : 
  (d = 140) -> (Nat.digits 10 140).sum = 5 :=
by
  sorry

end sum_of_digits_d_l212_212871


namespace minyoung_money_l212_212586

theorem minyoung_money (A M : ℕ) (h1 : M = 90 * A) (h2 : M = 60 * A + 270) : M = 810 :=
by 
  sorry

end minyoung_money_l212_212586


namespace resulting_solid_faces_l212_212353

-- Define a cube structure with a given number of faces
structure Cube where
  faces : Nat

-- Define the problem conditions and prove the total faces of the resulting solid
def original_cube := Cube.mk 6

def new_faces_per_cube := 5

def total_new_faces := original_cube.faces * new_faces_per_cube

def total_faces_of_resulting_solid := total_new_faces + original_cube.faces

theorem resulting_solid_faces : total_faces_of_resulting_solid = 36 := by
  sorry

end resulting_solid_faces_l212_212353


namespace fraction_product_equals_64_l212_212919

theorem fraction_product_equals_64 : 
  (1 / 4) * (8 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) * (1 / 8192) * (16384 / 1) = 64 :=
by
  sorry

end fraction_product_equals_64_l212_212919


namespace percentage_increase_240_to_288_l212_212059

theorem percentage_increase_240_to_288 :
  let initial := 240
  let final := 288
  ((final - initial) / initial) * 100 = 20 := by 
  sorry

end percentage_increase_240_to_288_l212_212059


namespace original_number_of_bullets_each_had_l212_212742

theorem original_number_of_bullets_each_had (x : ℕ) (h₁ : 5 * (x - 4) = x) : x = 5 := 
sorry

end original_number_of_bullets_each_had_l212_212742


namespace ratio_saturday_friday_l212_212567

variable (S : ℕ)
variable (soldOnFriday : ℕ := 30)
variable (soldOnSunday : ℕ := S - 15)
variable (totalSold : ℕ := 135)

theorem ratio_saturday_friday (h1 : soldOnFriday = 30)
                              (h2 : totalSold = 135)
                              (h3 : soldOnSunday = S - 15)
                              (h4 : soldOnFriday + S + soldOnSunday = totalSold) :
  (S / soldOnFriday) = 2 :=
by
  -- Prove the theorem here...
  sorry

end ratio_saturday_friday_l212_212567


namespace simplify_expression_l212_212263

theorem simplify_expression : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 :=
by
  sorry

end simplify_expression_l212_212263


namespace min_value_of_quadratic_l212_212298

theorem min_value_of_quadratic (m : ℝ) (x : ℝ) (hx1 : 3 ≤ x) (hx2 : x < 4) (h : x^2 - 4 * x ≥ m) : 
  m ≤ -3 :=
sorry

end min_value_of_quadratic_l212_212298


namespace distinct_zeros_abs_minus_one_l212_212217

theorem distinct_zeros_abs_minus_one : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (|x₁| - 1 = 0) ∧ (|x₂| - 1 = 0) := 
by
  sorry

end distinct_zeros_abs_minus_one_l212_212217


namespace find_real_solutions_l212_212797

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l212_212797


namespace rope_total_in_inches_l212_212676

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l212_212676


namespace weight_of_six_moles_BaF2_l212_212191

variable (atomic_weight_Ba : ℝ := 137.33) -- Atomic weight of Barium in g/mol
variable (atomic_weight_F : ℝ := 19.00) -- Atomic weight of Fluorine in g/mol
variable (moles_BaF2 : ℝ := 6) -- Number of moles of BaF2

theorem weight_of_six_moles_BaF2 :
  moles_BaF2 * (atomic_weight_Ba + 2 * atomic_weight_F) = 1051.98 :=
by sorry

end weight_of_six_moles_BaF2_l212_212191


namespace remove_parentheses_correct_l212_212606

variable {a b c : ℝ}

theorem remove_parentheses_correct :
  -(a - b) = -a + b :=
by sorry

end remove_parentheses_correct_l212_212606


namespace jason_gave_seashells_to_tim_l212_212862

-- Defining the conditions
def original_seashells : ℕ := 49
def current_seashells : ℕ := 36

-- The proof statement
theorem jason_gave_seashells_to_tim :
  original_seashells - current_seashells = 13 :=
by
  sorry

end jason_gave_seashells_to_tim_l212_212862


namespace original_price_correct_percentage_growth_rate_l212_212434

-- Definitions and conditions
def original_price := 45
def sale_discount := 15
def price_after_discount := original_price - sale_discount

def initial_cost_before_event := 90
def final_cost_during_event := 120
def ratio_of_chickens := 2

def initial_buyers := 50
def increase_percentage := 20
def total_sales := 5460
def time_slots := 2  -- 1 hour = 2 slots of 30 minutes each

-- The problem: Prove the original price and growth rate
theorem original_price_correct (x : ℕ) : (120 / (x - 15) = 2 * (90 / x) → x = original_price) :=
by
  sorry

theorem percentage_growth_rate (m : ℕ) :
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = total_sales / (original_price - sale_discount) →
  m = increase_percentage) :=
by
  sorry

end original_price_correct_percentage_growth_rate_l212_212434


namespace simplify_expr1_l212_212057

theorem simplify_expr1 (m n : ℝ) :
  (2 * m + n) ^ 2 - (4 * m + 3 * n) * (m - n) = 8 * m * n + 4 * n ^ 2 := by
  sorry

end simplify_expr1_l212_212057


namespace number_of_red_balls_l212_212670

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l212_212670


namespace find_x_l212_212349

theorem find_x (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_x_l212_212349


namespace base_seven_to_ten_l212_212044

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l212_212044


namespace solution_set_of_inequality_l212_212034

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l212_212034


namespace no_matching_formula_l212_212406

def xy_pairs : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 35), (4, 69), (5, 119)]

def formula_a (x : ℕ) : ℕ := x^3 + x^2 + x + 2
def formula_b (x : ℕ) : ℕ := 3 * x^2 + 2 * x + 1
def formula_c (x : ℕ) : ℕ := 2 * x^3 - x + 4
def formula_d (x : ℕ) : ℕ := 3 * x^3 + 2 * x^2 + x + 1

theorem no_matching_formula :
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_a pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_b pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_c pair.fst) ∧
  ¬ (∀ (pair : ℕ × ℕ), pair ∈ xy_pairs → pair.snd = formula_d pair.fst) :=
by
  sorry

end no_matching_formula_l212_212406


namespace solution_to_cubic_equation_l212_212788

theorem solution_to_cubic_equation :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
begin
  sorry
end

end solution_to_cubic_equation_l212_212788


namespace shaded_quilt_fraction_l212_212721

-- Define the basic structure of the problem using conditions from step a

def is_unit_square (s : ℕ) : Prop := s = 1

def grid_size : ℕ := 4
def total_squares : ℕ := grid_size * grid_size

def shaded_squares : ℕ := 2
def half_shaded_squares : ℕ := 4

def fraction_shaded (shaded: ℕ) (total: ℕ) : ℚ := shaded / total

theorem shaded_quilt_fraction :
  fraction_shaded (shaded_squares + half_shaded_squares / 2) total_squares = 1 / 4 :=
by
  sorry

end shaded_quilt_fraction_l212_212721


namespace quadratic_inequality_solutions_l212_212172

theorem quadratic_inequality_solutions (a x : ℝ) :
  (x^2 - (2+a)*x + 2*a > 0) → (
    (a < 2  → (x < a ∨ x > 2)) ∧
    (a = 2  → (x ≠ 2)) ∧
    (a > 2  → (x < 2 ∨ x > a))
  ) :=
by sorry

end quadratic_inequality_solutions_l212_212172


namespace find_c_l212_212295

theorem find_c (c : ℝ) (h : ∃ (f : ℝ → ℝ), (f = λ x => c * x^3 + 23 * x^2 - 5 * c * x + 55) ∧ f (-5) = 0) : c = 6.3 := 
by {
  sorry
}

end find_c_l212_212295


namespace probability_joint_l212_212454

variables (Ω : Type) [ProbabilitySpace Ω]

def eventA : Event Ω := {ω | passesTest ω}
def eventB : Event Ω := {ω | passesExam ω}

theorem probability_joint :
  (probability eventA = 0.8) →
  (probability (eventB | eventA) = 0.9) →
  probability (eventA ∩ eventB) = 0.72 :=
begin
  sorry
end

end probability_joint_l212_212454


namespace problem_statement_l212_212398

theorem problem_statement (a b c : ℝ) (h1 : a ∈ Set.Ioi 0) (h2 : b ∈ Set.Ioi 0) (h3 : c ∈ Set.Ioi 0) (h4 : a^2 + b^2 + c^2 = 3) : 
  1 / (2 - a) + 1 / (2 - b) + 1 / (2 - c) ≥ 3 := 
sorry

end problem_statement_l212_212398


namespace product_of_two_numbers_l212_212714

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.lcm a b = 72) (h2 : Nat.gcd a b = 8) :
  a * b = 576 :=
by
  sorry

end product_of_two_numbers_l212_212714


namespace magician_identifies_card_l212_212885

def Grid : Type := Fin 6 → Fin 6 → Nat

def choose_card (g : Grid) (c : Fin 6) (r : Fin 6) : Nat := g r c

def rearrange_columns_to_rows (s : List Nat) : Grid :=
  λ r c => s.get! (r.val * 6 + c.val)

theorem magician_identifies_card (g : Grid) (c1 : Fin 6) (r2 : Fin 6) :
  ∃ (card : Nat), (choose_card g c1 r2 = card) :=
  sorry

end magician_identifies_card_l212_212885


namespace fraction_income_spent_on_rent_l212_212211

theorem fraction_income_spent_on_rent
  (hourly_wage : ℕ)
  (work_hours_per_week : ℕ)
  (weeks_in_month : ℕ)
  (food_expense : ℕ)
  (tax_expense : ℕ)
  (remaining_income : ℕ) :
  hourly_wage = 30 →
  work_hours_per_week = 48 →
  weeks_in_month = 4 →
  food_expense = 500 →
  tax_expense = 1000 →
  remaining_income = 2340 →
  ((hourly_wage * work_hours_per_week * weeks_in_month - remaining_income - (food_expense + tax_expense)) / (hourly_wage * work_hours_per_week * weeks_in_month) = 1/3) :=
by
  intros h_wage h_hours h_weeks h_food h_taxes h_remaining
  sorry

end fraction_income_spent_on_rent_l212_212211


namespace arithmetic_sequence_second_term_l212_212595

theorem arithmetic_sequence_second_term (a d : ℤ)
  (h1 : a + 11 * d = 11)
  (h2 : a + 12 * d = 14) :
  a + d = -19 :=
sorry

end arithmetic_sequence_second_term_l212_212595


namespace sequence_general_formula_l212_212547

theorem sequence_general_formula (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  ∀ n, a n = n^2 - n + 1 :=
by sorry

end sequence_general_formula_l212_212547


namespace oldest_child_age_l212_212026

def arithmeticProgression (a d : ℕ) (n : ℕ) : ℕ := 
  a + (n - 1) * d

theorem oldest_child_age (a : ℕ) (d : ℕ) (n : ℕ) 
  (average : (arithmeticProgression a d 1 + arithmeticProgression a d 2 + arithmeticProgression a d 3 + arithmeticProgression a d 4 + arithmeticProgression a d 5) / 5 = 10)
  (distinct : ∀ i j, i ≠ j → arithmeticProgression a d i ≠ arithmeticProgression a d j)
  (constant_difference : d = 3) :
  arithmeticProgression a d 5 = 16 :=
by
  sorry

end oldest_child_age_l212_212026


namespace trigonometric_product_l212_212381

theorem trigonometric_product :
  (1 - Real.sin (Real.pi / 12)) * 
  (1 - Real.sin (5 * Real.pi / 12)) * 
  (1 - Real.sin (7 * Real.pi / 12)) * 
  (1 - Real.sin (11 * Real.pi / 12)) = 1 / 4 :=
by sorry

end trigonometric_product_l212_212381


namespace NumFriendsNextToCaraOnRight_l212_212921

open Nat

def total_people : ℕ := 8
def freds_next_to_Cara : ℕ := 7

theorem NumFriendsNextToCaraOnRight (h : total_people = 8) : freds_next_to_Cara = 7 :=
by
  sorry

end NumFriendsNextToCaraOnRight_l212_212921


namespace unique_n_for_given_divisors_l212_212312

theorem unique_n_for_given_divisors :
  ∃! (n : ℕ), 
    ∀ (k : ℕ) (d : ℕ → ℕ), 
      k ≥ 22 ∧ 
      d 1 = 1 ∧ d k = n ∧ 
      (∀ i j, i < j → d i < d j) ∧ 
      (d 7) ^ 2 + (d 10) ^ 2 = (n / d 22) ^ 2 →
      n = 2^3 * 3 * 5 * 17 :=
sorry

end unique_n_for_given_divisors_l212_212312


namespace leak_empty_time_l212_212016

theorem leak_empty_time (A L : ℝ) (h1 : A = 1 / 8) (h2 : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- The proof will be provided here
  sorry

end leak_empty_time_l212_212016


namespace probability_odd_3_in_6_rolls_l212_212188

-- Definitions based on problem conditions
def probability_of_odd (outcome: ℕ) : ℚ := if outcome % 2 = 1 then 1/2 else 0 

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  ((Nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k)))

-- Given problem
theorem probability_odd_3_in_6_rolls : 
  binomial_probability 6 3 (1/2) = 5 / 16 :=
by
  sorry

end probability_odd_3_in_6_rolls_l212_212188


namespace move_line_down_l212_212997

theorem move_line_down (x : ℝ) : (y = -x + 1) → (y = -x - 2) := by
  sorry

end move_line_down_l212_212997


namespace flour_amount_indeterminable_l212_212431

variable (flour_required : ℕ)
variable (sugar_required : ℕ := 11)
variable (sugar_added : ℕ := 10)
variable (flour_added : ℕ := 12)
variable (sugar_to_add : ℕ := 1)

theorem flour_amount_indeterminable :
  ¬ ∃ (flour_required : ℕ), flour_additional = flour_required - flour_added :=
by
  sorry

end flour_amount_indeterminable_l212_212431


namespace train_speed_proof_l212_212912

noncomputable def speed_of_train (train_length : ℝ) (time_seconds : ℝ) (man_speed : ℝ) : ℝ :=
  let train_length_km := train_length / 1000
  let time_hours := time_seconds / 3600
  let relative_speed := train_length_km / time_hours
  relative_speed - man_speed

theorem train_speed_proof :
  speed_of_train 605 32.99736021118311 6 = 60.028 :=
by
  unfold speed_of_train
  -- Direct substitution and expected numerical simplification
  norm_num
  sorry

end train_speed_proof_l212_212912


namespace water_flow_rate_l212_212740

theorem water_flow_rate
  (depth : ℝ := 4)
  (width : ℝ := 22)
  (flow_rate_kmph : ℝ := 2)
  (flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60)
  (cross_sectional_area : ℝ := depth * width)
  (volume_per_minute : ℝ := cross_sectional_area * flow_rate_mpm) :
  volume_per_minute = 2933.04 :=
  sorry

end water_flow_rate_l212_212740


namespace symmetric_points_on_ellipse_are_m_in_range_l212_212113

open Real

theorem symmetric_points_on_ellipse_are_m_in_range (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1 ∧ 
                   (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1 ∧ 
                   ∃ x0 y0 : ℝ, y0 = 4 * x0 + m ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2) 
  ↔ -2 * sqrt 13 / 13 < m ∧ m < 2 * sqrt 13 / 13 := 
 sorry

end symmetric_points_on_ellipse_are_m_in_range_l212_212113


namespace parallel_lines_condition_l212_212452

-- We define the conditions as Lean definitions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + (a^2 - 1) = 0
def parallel_condition (a : ℝ) : Prop := (a ≠ 0) ∧ (a ≠ 1) ∧ (a ≠ -1) ∧ (a * (a^2 - 1) ≠ 6)

-- Mathematically equivalent Lean 4 statement
theorem parallel_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y → line2 a x y → (line1 a x y ↔ line2 a x y)) ↔ (a = -1) :=
by 
  -- The full proof would be written here
  sorry

end parallel_lines_condition_l212_212452


namespace cars_on_wednesday_more_than_monday_l212_212890

theorem cars_on_wednesday_more_than_monday:
  let cars_tuesday := 25
  let cars_monday := 0.8 * cars_tuesday
  let cars_thursday := 10
  let cars_friday := 10
  let cars_saturday := 5
  let cars_sunday := 5
  let total_cars := 97
  ∃ (cars_wednesday : ℝ), cars_wednesday - cars_monday = 2 :=
by
  sorry

end cars_on_wednesday_more_than_monday_l212_212890


namespace trisha_walked_distance_l212_212012

theorem trisha_walked_distance :
  ∃ x : ℝ, (x + x + 0.67 = 0.89) ∧ (x = 0.11) :=
by sorry

end trisha_walked_distance_l212_212012


namespace max_knights_between_other_knights_l212_212222

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l212_212222


namespace digit_A_divisibility_l212_212684

theorem digit_A_divisibility :
  ∃ (A : ℕ), (0 ≤ A ∧ A < 10) ∧ (∃ k_5 : ℕ, 353809 * 10 + A = 5 * k_5) ∧ 
  (∃ k_7 : ℕ, 353809 * 10 + A = 7 * k_7) ∧ (∃ k_11 : ℕ, 353809 * 10 + A = 11 * k_11) 
  ∧ A = 0 :=
by 
  sorry

end digit_A_divisibility_l212_212684


namespace find_b_days_l212_212347

theorem find_b_days 
  (a_days b_days c_days : ℕ)
  (a_wage b_wage c_wage : ℕ)
  (total_earnings : ℕ)
  (ratio_3_4_5 : a_wage * 5 = b_wage * 4 ∧ b_wage * 5 = c_wage * 4 ∧ a_wage * 5 = c_wage * 3)
  (c_wage_val : c_wage = 110)
  (a_days_val : a_days = 6)
  (c_days_val : c_days = 4) 
  (total_earnings_val : total_earnings = 1628)
  (earnings_eq : a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings) :
  b_days = 9 := by
  sorry

end find_b_days_l212_212347


namespace prove_by_contradiction_l212_212164

-- Statement: To prove "a > b" by contradiction, assuming the negation "a ≤ b".
theorem prove_by_contradiction (a b : ℝ) (h : a ≤ b) : false := sorry

end prove_by_contradiction_l212_212164


namespace probability_at_least_one_coordinate_greater_l212_212000

theorem probability_at_least_one_coordinate_greater (p : ℝ) :
  (∃ (x y : ℝ), (0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ (x > p ∨ y > p))) ↔ p = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end probability_at_least_one_coordinate_greater_l212_212000


namespace average_value_s7_squared_is_3680_l212_212011

def sum_of_digits_base (b n : ℕ) : ℕ := (n.digits b).sum

noncomputable def average_value_s7_squared : ℕ :=
  let N := 7^20 in
  (∑ n in Finset.range N, (sum_of_digits_base 7 n)^2) / N

theorem average_value_s7_squared_is_3680 :
  average_value_s7_squared = 3680 := 
by
  sorry

end average_value_s7_squared_is_3680_l212_212011


namespace eval_difference_of_squares_l212_212509

theorem eval_difference_of_squares :
  (81^2 - 49^2 = 4160) :=
by
  -- Since the exact mathematical content is established in a formal context, 
  -- we omit the detailed proof steps.
  sorry

end eval_difference_of_squares_l212_212509


namespace inequality_x_add_inv_x_ge_two_l212_212995

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end inequality_x_add_inv_x_ge_two_l212_212995


namespace vershoks_per_arshin_l212_212412

theorem vershoks_per_arshin (plank_length_arshins : ℝ) (plank_width_vershoks : ℝ) 
    (room_side_length_arshins : ℝ) (total_planks : ℕ) (n : ℝ)
    (h1 : plank_length_arshins = 6) (h2 : plank_width_vershoks = 6)
    (h3 : room_side_length_arshins = 12) (h4 : total_planks = 64) 
    (h5 : (total_planks : ℝ) * (plank_length_arshins * (plank_width_vershoks / n)) = room_side_length_arshins^2) :
    n = 16 :=
by {
  sorry
}

end vershoks_per_arshin_l212_212412


namespace brown_ball_weight_l212_212991

def total_weight : ℝ := 9.12
def weight_blue : ℝ := 6
def weight_brown : ℝ := 3.12

theorem brown_ball_weight : total_weight - weight_blue = weight_brown :=
by 
  sorry

end brown_ball_weight_l212_212991


namespace no_real_a_b_l212_212315

noncomputable def SetA (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ n : ℤ, p.1 = n ∧ p.2 = n * a + b}

noncomputable def SetB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℤ, p.1 = m ∧ p.2 = 3 * m^2 + 15}

noncomputable def SetC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 144}

theorem no_real_a_b :
  ¬ ∃ (a b : ℝ), (∃ p ∈ SetA a b, p ∈ SetB) ∧ (a, b) ∈ SetC :=
by
    sorry

end no_real_a_b_l212_212315


namespace total_price_of_25_shirts_l212_212968

theorem total_price_of_25_shirts (S W : ℝ) (H1 : W = S + 4) (H2 : 75 * W = 1500) : 
  25 * S = 400 :=
by
  -- Proof would go here
  sorry

end total_price_of_25_shirts_l212_212968


namespace function_machine_output_is_17_l212_212143

def functionMachineOutput (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 <= 22 then step1 + 10 else step1 - 7

theorem function_machine_output_is_17 : functionMachineOutput 8 = 17 := by
  sorry

end function_machine_output_is_17_l212_212143


namespace sum_of_first_3_geometric_terms_eq_7_l212_212649

theorem sum_of_first_3_geometric_terms_eq_7 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_ratio_gt_1 : r > 1)
  (h_eq : (a 0 + a 2 = 5) ∧ (a 0 * a 2 = 4)) 
  : (a 0 + a 1 + a 2) = 7 := 
by
  sorry

end sum_of_first_3_geometric_terms_eq_7_l212_212649


namespace envelope_width_l212_212083

theorem envelope_width (Area Height Width : ℝ) (h_area : Area = 36) (h_height : Height = 6) (h_area_formula : Area = Width * Height) : Width = 6 :=
by
  sorry

end envelope_width_l212_212083


namespace simplify_trig_l212_212574

open Real

theorem simplify_trig : 
  (sin (30 * pi / 180) + sin (60 * pi / 180)) / (cos (30 * pi / 180) + cos (60 * pi / 180)) = tan (45 * pi / 180) :=
by
  sorry

end simplify_trig_l212_212574


namespace distribute_balls_l212_212834

theorem distribute_balls : 
  ∀ (balls boxes: ℕ), 
  balls = 5 → 
  boxes = 4 → 
  (∑ n in (finset.range (balls + 1)).powerset, if n.sum = balls then (n.card!) else 0) = 56 :=
by {
  intros balls boxes h_balls h_boxes,
  sorry
}

end distribute_balls_l212_212834


namespace necessary_and_sufficient_condition_l212_212058

open Real

theorem necessary_and_sufficient_condition 
  {x y : ℝ} (p : x > y) (q : x - y + sin (x - y) > 0) : 
  (x > y) ↔ (x - y + sin (x - y) > 0) :=
sorry

end necessary_and_sufficient_condition_l212_212058


namespace find_K_l212_212147

theorem find_K (Z K : ℕ) (hZ1 : 1000 < Z) (hZ2 : Z < 8000) (hK : Z = K^3) : 11 ≤ K ∧ K ≤ 19 :=
sorry

end find_K_l212_212147


namespace lockers_remaining_open_l212_212141

-- Define the number of lockers and students
def num_lockers : ℕ := 1000

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define a function to count perfect squares up to a given number
def count_perfect_squares_up_to (n : ℕ) : ℕ :=
  Nat.sqrt n

-- Theorem statement
theorem lockers_remaining_open : 
  count_perfect_squares_up_to num_lockers = 31 :=
by
  -- Proof left out because it's not necessary to provide
  sorry

end lockers_remaining_open_l212_212141


namespace max_knights_adjacent_to_two_other_knights_l212_212239

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l212_212239


namespace bees_count_l212_212566

-- Definitions of the conditions
def day1_bees (x : ℕ) := x  -- Number of bees on the first day
def day2_bees (x : ℕ) := 3 * day1_bees x  -- Number of bees on the second day is 3 times that on the first day

theorem bees_count (x : ℕ) (h : day2_bees x = 432) : day1_bees x = 144 :=
by
  dsimp [day1_bees, day2_bees] at h
  have h1 : 3 * x = 432 := h
  sorry

end bees_count_l212_212566


namespace num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l212_212411

theorem num_of_tenths_in_1_9 : (1.9 / 0.1) = 19 :=
by sorry

theorem num_of_hundredths_in_0_8 : (0.8 / 0.01) = 80 :=
by sorry

end num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l212_212411


namespace Julie_initial_savings_l212_212146

theorem Julie_initial_savings (P r : ℝ) 
  (h1 : 100 = P * r * 2) 
  (h2 : 105 = P * (1 + r) ^ 2 - P) : 
  2 * P = 1000 :=
by
  sorry

end Julie_initial_savings_l212_212146


namespace train_speed_l212_212076

theorem train_speed (train_length : ℝ) (man_speed_kmph : ℝ) (passing_time : ℝ) : 
  train_length = 160 → man_speed_kmph = 6 →
  passing_time = 6 → (train_length / passing_time + man_speed_kmph * 1000 / 3600) * 3600 / 1000 = 90 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- further proof steps are omitted
  sorry

end train_speed_l212_212076


namespace combined_original_price_l212_212153

def original_price_shoes (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

def original_price_dress (discount_price : ℚ) (discount_rate : ℚ) : ℚ := discount_price / (1 - discount_rate)

theorem combined_original_price (shoes_price : ℚ) (shoes_discount : ℚ) (dress_price : ℚ) (dress_discount : ℚ) 
  (h_shoes : shoes_discount = 0.20 ∧ shoes_price = 480) 
  (h_dress : dress_discount = 0.30 ∧ dress_price = 350) : 
  original_price_shoes shoes_price shoes_discount + original_price_dress dress_price dress_discount = 1100 := by
  sorry

end combined_original_price_l212_212153


namespace mason_hotdogs_proof_mason_ate_15_hotdogs_l212_212692

-- Define the weights of the items.
def weight_hotdog := 2 -- in ounces
def weight_burger := 5 -- in ounces
def weight_pie := 10 -- in ounces

-- Define Noah's consumption
def noah_burgers := 8

-- Define the total weight of hotdogs Mason ate
def mason_hotdogs_weight := 30

-- Calculate the number of hotdogs Mason ate
def hotdogs_mason_ate := mason_hotdogs_weight / weight_hotdog

-- Calculate the number of pies Jacob ate
def jacob_pies := noah_burgers - 3

-- Given conditions
theorem mason_hotdogs_proof :
  mason_hotdogs_weight / weight_hotdog = 3 * (noah_burgers - 3) :=
by
  sorry

-- Proving the number of hotdogs Mason ate equals 15
theorem mason_ate_15_hotdogs :
  hotdogs_mason_ate = 15 :=
by
  sorry

end mason_hotdogs_proof_mason_ate_15_hotdogs_l212_212692


namespace min_value_expr_l212_212511

theorem min_value_expr : ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by
  sorry

end min_value_expr_l212_212511


namespace minimum_flour_cost_l212_212599

-- Definitions based on conditions provided
def loaves : ℕ := 12
def flour_per_loaf : ℕ := 4
def flour_needed : ℕ := loaves * flour_per_loaf

def ten_pound_bag_weight : ℕ := 10
def ten_pound_bag_cost : ℕ := 10

def twelve_pound_bag_weight : ℕ := 12
def twelve_pound_bag_cost : ℕ := 13

def cost_10_pound_bags : ℕ := (flour_needed + ten_pound_bag_weight - 1) / ten_pound_bag_weight * ten_pound_bag_cost
def cost_12_pound_bags : ℕ := (flour_needed + twelve_pound_bag_weight - 1) / twelve_pound_bag_weight * twelve_pound_bag_cost

theorem minimum_flour_cost : min cost_10_pound_bags cost_12_pound_bags = 50 := by
  sorry

end minimum_flour_cost_l212_212599


namespace solution_inequality_l212_212126

noncomputable def solution_set (a : ℝ) (x : ℝ) := x < (1 - a) / (1 + a)

theorem solution_inequality 
  (a : ℝ) 
  (h1 : a^3 < a) 
  (h2 : a < a^2) :
  ∀ (x : ℝ), x + a > 1 - a * x ↔ solution_set a x :=
sorry

end solution_inequality_l212_212126


namespace exams_in_fourth_year_l212_212366

noncomputable def student_exam_counts 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 ∧ 
  a_5 = 3 * a_1 ∧ 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5

theorem exams_in_fourth_year 
  (a_1 a_2 a_3 a_4 a_5 : ℕ) (h : student_exam_counts a_1 a_2 a_3 a_4 a_5) : 
  a_4 = 8 :=
sorry

end exams_in_fourth_year_l212_212366


namespace prize_winners_l212_212858

theorem prize_winners (n : ℕ) (p1 p2 : ℝ) (h1 : n = 100) (h2 : p1 = 0.4) (h3 : p2 = 0.2) :
  ∃ winners : ℕ, winners = (p2 * (p1 * n)) ∧ winners = 8 :=
by
  sorry

end prize_winners_l212_212858


namespace election_majority_l212_212666

theorem election_majority (total_votes : ℕ) (winning_percentage : ℝ) (losing_percentage : ℝ)
  (h_total_votes : total_votes = 700)
  (h_winning_percentage : winning_percentage = 0.70)
  (h_losing_percentage : losing_percentage = 0.30) :
  (winning_percentage * total_votes - losing_percentage * total_votes) = 280 :=
by
  sorry

end election_majority_l212_212666


namespace ribbons_left_l212_212433

theorem ribbons_left {initial_ribbons morning_giveaway afternoon_giveaway ribbons_left : ℕ} 
    (h1 : initial_ribbons = 38) 
    (h2 : morning_giveaway = 14) 
    (h3 : afternoon_giveaway = 16) 
    (h4 : ribbons_left = initial_ribbons - (morning_giveaway + afternoon_giveaway)) : 
  ribbons_left = 8 := 
by 
  sorry

end ribbons_left_l212_212433


namespace log_w_u_value_l212_212693

noncomputable def log (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem log_w_u_value (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) (hu1 : u ≠ 1) (hv1 : v ≠ 1) (hw1 : w ≠ 1)
    (h1 : log u (v * w) + log v w = 5) (h2 : log v u + log w v = 3) : 
    log w u = 4 / 5 := 
sorry

end log_w_u_value_l212_212693


namespace domain_of_function_l212_212174

theorem domain_of_function :
  ∀ x, (2 * x - 1 ≥ 0) ∧ (x^2 ≠ 1) → (x ≥ 1/2 ∧ x < 1) ∨ (x > 1) := 
sorry

end domain_of_function_l212_212174


namespace polygon_sides_l212_212658

theorem polygon_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 180 - 360 / n = 150) : n = 12 := by
  sorry

end polygon_sides_l212_212658


namespace average_run_per_day_l212_212372

theorem average_run_per_day (n6 n7 n8 : ℕ) 
  (h1 : 3 * n7 = n6) 
  (h2 : 3 * n8 = n7) 
  (h3 : n6 * 20 + n7 * 18 + n8 * 16 = 250 * n8) : 
  (n6 * 20 + n7 * 18 + n8 * 16) / (n6 + n7 + n8) = 250 / 13 :=
by sorry

end average_run_per_day_l212_212372


namespace trigonometric_identity_l212_212573

theorem trigonometric_identity :
  let sin_30 := 1 / 2,
      sin_60 := Real.sqrt 3 / 2,
      cos_30 := Real.sqrt 3 / 2,
      cos_60 := 1 / 2,
      sin_45 := Real.sqrt 2 / 2,
      cos_45 := Real.sqrt 2 / 2 in
  (sin_30 + sin_60) / (cos_30 + cos_60) = Real.tan (Real.pi / 4) := 
  by
    sorry

end trigonometric_identity_l212_212573


namespace max_knights_seated_next_to_two_knights_l212_212247

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l212_212247


namespace blood_expiration_date_l212_212758

noncomputable def factorial (n : Nat) : Nat :=
  if n == 0 then 1 else n * factorial (n - 1)

theorem blood_expiration_date:
  ∀ (donation_day : Nat), 
  ∀ (days_in_month : Nat), 
  ∀ (total_seconds_in_a_day : Nat), 
  total_seconds_in_a_day = 86400 ∧ donation_day = 3 ∧ days_in_month = 31 → 
  (donation_day + factorial 8 / total_seconds_in_a_day) = 3 :=
by {
  intros donation_day days_in_month total_seconds_in_a_day h,
  have fact_eq : factorial 8 = 40320 := rfl,
  have sec_in_day_eq : total_seconds_in_a_day = 86400 := rfl,
  have day_donation : donation_day = 3 := rfl,
  have days_month : days_in_month = 31 := rfl,
  simp [fact_eq, sec_in_day_eq, day_donation, days_month] at *,
  sorry
}

end blood_expiration_date_l212_212758


namespace problem_statement_l212_212270

theorem problem_statement (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 1) (x : ℝ) :
    (m * x^2 - 2 * x - m ≥ 2) ↔ (x ≤ -1) := sorry

end problem_statement_l212_212270


namespace max_knights_seated_next_to_two_knights_l212_212246

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l212_212246


namespace false_statement_l212_212652

-- Define propositions p and q
def p := ∀ x : ℝ, (|x| = x) ↔ (x ≥ 0)
def q := ∀ (f : ℝ → ℝ), (∀ x, f (-x) = -f x) → (∃ origin : ℝ, ∀ y : ℝ, f (origin + y) = f (origin - y))

-- Define the possible answers
def option_A := p ∨ q
def option_B := p ∧ q
def option_C := ¬p ∧ q
def option_D := ¬p ∨ q

-- Define the false option (the correct answer was B)
def false_proposition := option_B

-- The statement to prove
theorem false_statement : false_proposition = false :=
by sorry

end false_statement_l212_212652


namespace linear_system_solution_l212_212202

theorem linear_system_solution (x y : ℝ) (h1 : x = 2) (h2 : y = -3) : x + y = -1 :=
by
  sorry

end linear_system_solution_l212_212202


namespace cubic_solution_l212_212800

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l212_212800


namespace marble_cut_percentage_first_week_l212_212756

theorem marble_cut_percentage_first_week :
  ∀ (W1 W2 : ℝ), 
  W1 = W2 / 0.70 → 
  W2 = 124.95 / 0.85 → 
  (300 - W1) / 300 * 100 = 30 :=
by
  intros W1 W2 h1 h2
  sorry

end marble_cut_percentage_first_week_l212_212756


namespace maciek_total_cost_l212_212068

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l212_212068


namespace eval_floor_ceil_sum_l212_212391

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def ceil (x : ℝ) : ℤ := Int.ceil x

theorem eval_floor_ceil_sum : floor (-3.67) + ceil 34.7 = 31 := by
  sorry

end eval_floor_ceil_sum_l212_212391


namespace at_least_one_not_greater_than_neg_two_l212_212979

open Real

theorem at_least_one_not_greater_than_neg_two
  {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  a + (1 / b) ≤ -2 ∨ b + (1 / c) ≤ -2 ∨ c + (1 / a) ≤ -2 :=
sorry

end at_least_one_not_greater_than_neg_two_l212_212979


namespace evaluate_expression_l212_212262

theorem evaluate_expression :
  (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 :=
by
  sorry

end evaluate_expression_l212_212262


namespace value_of_x_l212_212200

theorem value_of_x (u w z y x : ℤ) (h1 : u = 95) (h2 : w = u + 10) (h3 : z = w + 25) (h4 : y = z + 15) (h5 : x = y + 12) : x = 157 := by
  sorry

end value_of_x_l212_212200


namespace product_of_primes_sum_ten_l212_212458

theorem product_of_primes_sum_ten :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ Prime p1 ∧ Prime p2 ∧ p1 + p2 = 10 ∧ p1 * p2 = 21 := 
by
  sorry

end product_of_primes_sum_ten_l212_212458


namespace max_knights_between_other_knights_l212_212219

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l212_212219


namespace equation_elliptic_and_canonical_form_l212_212258

-- Defining the necessary conditions and setup
def a11 := 1
def a12 := 1
def a22 := 2

def is_elliptic (a11 a12 a22 : ℝ) : Prop :=
  a12^2 - a11 * a22 < 0

def canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) : Prop :=
  let ξ := y - x
  let η := x
  let u_ξξ := u_xx -- Assuming u_xx represents u_ξξ after change of vars
  let u_ξη := u_xy
  let u_ηη := u_yy
  let u_ξ := u_x -- Assuming u_x represents u_ξ after change of vars
  let u_η := u_y
  u_ξξ + u_ηη = -2 * u_η + u + η + (ξ + η)^2

theorem equation_elliptic_and_canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) :
  is_elliptic a11 a12 a22 ∧
  canonical_form u_xx u_xy u_yy u_x u_y u x y :=
by
  sorry -- Proof to be completed

end equation_elliptic_and_canonical_form_l212_212258


namespace probability_no_two_boys_same_cinema_l212_212888

-- Definitions
def total_cinemas := 10
def total_boys := 7

def total_arrangements : ℕ := total_cinemas ^ total_boys
def favorable_arrangements : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4
def probability := (favorable_arrangements : ℚ) / total_arrangements

-- Mathematical proof problem
theorem probability_no_two_boys_same_cinema : 
  probability = 0.06048 := 
by {
  sorry -- Proof goes here
}

end probability_no_two_boys_same_cinema_l212_212888


namespace max_knights_between_knights_l212_212226

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l212_212226


namespace determine_y_l212_212983

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (hxy : x = 2 + (1 / y))
variable (hyx : y = 2 + (2 / x))

theorem determine_y (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 2 + (1 / y)) (hyx : y = 2 + (2 / x)) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 := 
sorry

end determine_y_l212_212983


namespace ironed_clothing_count_l212_212439

theorem ironed_clothing_count : 
  (4 * 2 + 5 * 3) + (3 * 3 + 4 * 2) + (2 * 1 + 3 * 1) = 45 := by
  sorry

end ironed_clothing_count_l212_212439


namespace problem_solution_l212_212410

theorem problem_solution (x : ℝ) (h : x^2 - 8*x - 3 = 0) : (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 :=
by sorry

end problem_solution_l212_212410


namespace knights_max_seated_between_knights_l212_212228

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l212_212228


namespace product_of_roots_eq_25_l212_212268

theorem product_of_roots_eq_25 (t : ℝ) (h : t^2 - 10 * t + 25 = 0) : t * t = 25 :=
sorry

end product_of_roots_eq_25_l212_212268


namespace bernie_savings_l212_212627

-- Defining conditions
def chocolates_per_week : ℕ := 2
def weeks : ℕ := 3
def chocolates_total : ℕ := chocolates_per_week * weeks
def local_store_cost_per_chocolate : ℕ := 3
def different_store_cost_per_chocolate : ℕ := 2

-- Defining the costs in both stores
def local_store_total_cost : ℕ := chocolates_total * local_store_cost_per_chocolate
def different_store_total_cost : ℕ := chocolates_total * different_store_cost_per_chocolate

-- The statement we want to prove
theorem bernie_savings : local_store_total_cost - different_store_total_cost = 6 :=
by
  sorry

end bernie_savings_l212_212627


namespace fraction_to_zero_power_l212_212189

theorem fraction_to_zero_power :
  756321948 ≠ 0 ∧ -3958672103 ≠ 0 →
  (756321948 / -3958672103 : ℝ) ^ 0 = 1 :=
by
  intro h
  have numerator_nonzero : 756321948 ≠ 0 := h.left
  have denominator_nonzero : -3958672103 ≠ 0 := h.right
  -- Skipping the rest of the proof.
  sorry

end fraction_to_zero_power_l212_212189


namespace replace_asterisk_l212_212049

theorem replace_asterisk (star : ℝ) : ((36 / 18) * (star / 72) = 1) → star = 36 :=
by
  intro h
  sorry

end replace_asterisk_l212_212049


namespace rhombus_diagonals_l212_212717

theorem rhombus_diagonals (p d1 d2 : ℝ) (h1 : p = 100) (h2 : abs (d1 - d2) = 34) :
  ∃ d1 d2 : ℝ, d1 = 14 ∧ d2 = 48 :=
by
  -- proof omitted
  sorry

end rhombus_diagonals_l212_212717


namespace elements_of_set_A_l212_212530

theorem elements_of_set_A (A : Set ℝ) (h₁ : ∀ a : ℝ, a ∈ A → (1 + a) / (1 - a) ∈ A)
(h₂ : -3 ∈ A) : A = {-3, -1/2, 1/3, 2} := by
  sorry

end elements_of_set_A_l212_212530


namespace Sarah_l212_212877

variable (s g : ℕ)

theorem Sarah's_score_130 (h1 : s = g + 50) (h2 : (s + g) / 2 = 105) : s = 130 :=
by
  sorry

end Sarah_l212_212877


namespace g_ge_one_l212_212651

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem g_ge_one (x : ℝ) (h : 0 < x) : g x ≥ 1 :=
sorry

end g_ge_one_l212_212651


namespace average_customers_per_day_l212_212073

-- Define the number of customers each day:
def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define the theorem stating the average number of daily customers
theorem average_customers_per_day :
  (customers_per_day.sum : ℚ) / days_in_week = 13.57 :=
by
  sorry

end average_customers_per_day_l212_212073


namespace solution_set_abs_inequality_l212_212036

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l212_212036


namespace probability_correct_l212_212544

noncomputable def probability_two_queens_or_at_least_one_jack : ℚ :=
  let total_cards := 52
  let queens := 3
  let jacks := 1
  let prob_two_queens := (queens * (queens - 1)) / (total_cards * (total_cards - 1))
  let prob_one_jack := jacks / total_cards * (total_cards - jacks) / (total_cards - 1) + (total_cards - jacks) / total_cards * jacks / (total_cards - 1)
  prob_two_queens + prob_one_jack

theorem probability_correct : probability_two_queens_or_at_least_one_jack = 9 / 221 := by
  sorry

end probability_correct_l212_212544


namespace power_identity_l212_212683

-- Define the given definitions
def P (m : ℕ) : ℕ := 5 ^ m
def R (n : ℕ) : ℕ := 7 ^ n

-- The theorem to be proved
theorem power_identity (m n : ℕ) : 35 ^ (m + n) = (P m ^ n * R n ^ m) := 
by sorry

end power_identity_l212_212683


namespace red_balls_approximation_l212_212668

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l212_212668


namespace last_digit_7_powers_l212_212214

theorem last_digit_7_powers :
  (∃ n : ℕ, (∀ k < 4004, k.mod 2002 == n))
  := sorry

end last_digit_7_powers_l212_212214


namespace mrs_peterson_change_l212_212870

def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

theorem mrs_peterson_change : 
  (num_bills * value_per_bill) - (num_tumblers * cost_per_tumbler) = 50 :=
by
  sorry

end mrs_peterson_change_l212_212870


namespace find_u_plus_v_l212_212539

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 7 * v = 17) (h2 : 5 * u + 3 * v = 1) : 
  u + v = - 6 / 11 :=
  sorry

end find_u_plus_v_l212_212539


namespace rohan_monthly_salary_expenses_l212_212021

theorem rohan_monthly_salary_expenses 
    (food_expense_pct : ℝ)
    (house_rent_expense_pct : ℝ)
    (entertainment_expense_pct : ℝ)
    (conveyance_expense_pct : ℝ)
    (utilities_expense_pct : ℝ)
    (misc_expense_pct : ℝ)
    (monthly_saved_amount : ℝ)
    (entertainment_expense_increase_after_6_months : ℝ)
    (conveyance_expense_decrease_after_6_months : ℝ)
    (monthly_salary : ℝ)
    (savings_pct : ℝ)
    (new_savings_pct : ℝ) : 
    (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct = 90) → 
    (100 - (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct) = savings_pct) → 
    (monthly_saved_amount = monthly_salary * savings_pct / 100) → 
    (entertainment_expense_pct + entertainment_expense_increase_after_6_months = 20) → 
    (conveyance_expense_pct - conveyance_expense_decrease_after_6_months = 7) → 
    (new_savings_pct = 100 - (30 + 25 + (entertainment_expense_pct + entertainment_expense_increase_after_6_months) + (conveyance_expense_pct - conveyance_expense_decrease_after_6_months) + 5 + 5)) → 
    monthly_salary = 15000 ∧ new_savings_pct = 8 := 
sorry

end rohan_monthly_salary_expenses_l212_212021


namespace total_number_of_digits_l212_212197

-- Definitions based on identified conditions
def first2500EvenIntegers := {n : ℕ | n % 2 = 0 ∧ 1 ≤ n ∧ n ≤ 5000}

-- Theorem statement based on the equivalent proof problem
theorem total_number_of_digits : 
  (first2500EvenIntegers.count_digits = 9448) :=
sorry

end total_number_of_digits_l212_212197


namespace johns_leisure_travel_miles_per_week_l212_212423

-- Define the given conditions
def mpg : Nat := 30
def work_round_trip_miles : Nat := 20 * 2  -- 20 miles to work + 20 miles back home
def work_days_per_week : Nat := 5
def weekly_fuel_usage_gallons : Nat := 8

-- Define the property to prove
theorem johns_leisure_travel_miles_per_week :
  let work_miles_per_week := work_round_trip_miles * work_days_per_week
  let total_possible_miles := weekly_fuel_usage_gallons * mpg
  let leisure_miles := total_possible_miles - work_miles_per_week
  leisure_miles = 40 :=
by
  sorry

end johns_leisure_travel_miles_per_week_l212_212423


namespace expand_expression_l212_212392

theorem expand_expression (x : ℝ) : 25 * (3 * x - 4) = 75 * x - 100 := 
by 
  sorry

end expand_expression_l212_212392


namespace original_weight_of_potatoes_l212_212477

theorem original_weight_of_potatoes (W : ℝ) (h : W / (W / 2) = 36) : W = 648 := by
  sorry

end original_weight_of_potatoes_l212_212477


namespace investment_of_c_l212_212903

-- Definitions of given conditions
def P_b: ℝ := 4000
def diff_Pa_Pc: ℝ := 1599.9999999999995
def Ca: ℝ := 8000
def Cb: ℝ := 10000

-- Goal to be proved
theorem investment_of_c (C_c: ℝ) : 
  (∃ P_a P_c, (P_a / Ca = P_b / Cb) ∧ (P_c / C_c = P_b / Cb) ∧ (P_a - P_c = diff_Pa_Pc)) → 
  C_c = 4000 :=
sorry

end investment_of_c_l212_212903


namespace range_of_m_l212_212278

-- Definitions based on the conditions
def p (m : ℝ) : Prop := 4 - 4 * m > 0
def q (m : ℝ) : Prop := m + 2 > 0

-- Problem statement in Lean 4
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≤ -2 ∨ m ≥ 1 := by
  sorry

end range_of_m_l212_212278


namespace right_side_longer_l212_212182

/-- The sum of the three sides of a triangle is 50. 
    The right side of the triangle is a certain length longer than the left side, which has a value of 12 cm. 
    The triangle base has a value of 24 cm. 
    Prove that the right side is 2 cm longer than the left side. -/
theorem right_side_longer (L R B : ℝ) (hL : L = 12) (hB : B = 24) (hSum : L + B + R = 50) : R = L + 2 :=
by
  sorry

end right_side_longer_l212_212182


namespace total_amount_earned_l212_212348

-- Definitions of the conditions.
def work_done_per_day (days : ℕ) : ℚ := 1 / days

def total_work_done_per_day : ℚ :=
  work_done_per_day 6 + work_done_per_day 8 + work_done_per_day 12

def b_share : ℚ := work_done_per_day 8

def total_amount (b_earnings : ℚ) : ℚ := b_earnings * (total_work_done_per_day / b_share)

-- Main theorem stating that the total amount earned is $1170 if b's share is $390.
theorem total_amount_earned (h_b : b_share * 390 = 390) : total_amount 390 = 1170 := by sorry

end total_amount_earned_l212_212348


namespace min_value_x_y_l212_212516

theorem min_value_x_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) : x + y ≥ 2 :=
sorry

end min_value_x_y_l212_212516


namespace rectangle_cut_dimensions_l212_212488

-- Define the original dimensions of the rectangle as constants.
def original_length : ℕ := 12
def original_height : ℕ := 6

-- Define the dimensions of the new rectangle after slicing parallel to the longer side.
def new_length := original_length / 2
def new_height := original_height

-- The theorem statement.
theorem rectangle_cut_dimensions :
  new_length = 6 ∧ new_height = 6 :=
by
  sorry

end rectangle_cut_dimensions_l212_212488


namespace eliza_is_18_l212_212497

-- Define the relevant ages
def aunt_ellen_age : ℕ := 48
def dina_age : ℕ := aunt_ellen_age / 2
def eliza_age : ℕ := dina_age - 6

-- Theorem to prove Eliza's age is 18
theorem eliza_is_18 : eliza_age = 18 := by
  sorry

end eliza_is_18_l212_212497


namespace geometric_sequence_extreme_points_l212_212942

-- Given conditions
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

def f (x : ℝ) : ℝ :=
  x^3 / 3 - 5 * x^2 / 2 + 4 * x + 1

def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  deriv f x = 0

theorem geometric_sequence_extreme_points (a : ℕ → ℝ) (r : ℝ) :
  is_geometric_sequence a r →
  is_extreme_point f (a 1) →
  is_extreme_point f (a 5) →
  a 3 = 2 :=
by sorry

end geometric_sequence_extreme_points_l212_212942


namespace manager_salary_3700_l212_212610

theorem manager_salary_3700
  (salary_20_employees_avg : ℕ)
  (salary_increase : ℕ)
  (total_employees : ℕ)
  (manager_salary : ℕ)
  (h_avg : salary_20_employees_avg = 1600)
  (h_increase : salary_increase = 100)
  (h_total_employees : total_employees = 20)
  (h_manager_salary : manager_salary = 21 * (salary_20_employees_avg + salary_increase) - 20 * salary_20_employees_avg) :
  manager_salary = 3700 :=
by
  sorry

end manager_salary_3700_l212_212610


namespace ratio_wy_l212_212590

-- Define the variables and conditions
variables (w x y z : ℚ)
def ratio_wx := w / x = 5 / 4
def ratio_yz := y / z = 7 / 5
def ratio_zx := z / x = 1 / 8

-- Statement to prove
theorem ratio_wy (hwx : ratio_wx w x) (hyz : ratio_yz y z) (hzx : ratio_zx z x) : w / y = 25 / 7 :=
by
  sorry  -- Proof not needed

end ratio_wy_l212_212590


namespace represent_same_function_l212_212344

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l212_212344


namespace length_of_rectangle_l212_212592

theorem length_of_rectangle (L : ℝ) (W : ℝ) (A_triangle : ℝ) (hW : W = 4) (hA_triangle : A_triangle = 60)
  (hRatio : (L * W) / A_triangle = 2 / 5) : L = 6 :=
by
  sorry

end length_of_rectangle_l212_212592


namespace edge_length_of_cube_l212_212419

noncomputable def cost_per_quart : ℝ := 3.20
noncomputable def coverage_per_quart : ℕ := 120
noncomputable def total_cost : ℝ := 16
noncomputable def total_coverage : ℕ := 600 -- From 5 quarts * 120 square feet per quart
noncomputable def surface_area (edge_length : ℝ) : ℝ := 6 * (edge_length ^ 2)

theorem edge_length_of_cube :
  (∃ edge_length : ℝ, surface_area edge_length = total_coverage) → 
  ∃ edge_length : ℝ, edge_length = 10 :=
by
  sorry

end edge_length_of_cube_l212_212419


namespace problem_1_l212_212377

theorem problem_1 : -9 + 5 - (-12) + (-3) = 5 :=
by {
  -- Proof goes here
  sorry
}

end problem_1_l212_212377


namespace is_positive_integer_iff_l212_212541

theorem is_positive_integer_iff (p : ℕ) : 
  (p > 0 → ∃ k : ℕ, (4 * p + 17 = k * (3 * p - 7))) ↔ (3 ≤ p ∧ p ≤ 40) := 
sorry

end is_positive_integer_iff_l212_212541


namespace voldemort_spending_l212_212893

theorem voldemort_spending :
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  (book_price_paid = (original_book_price / 8)) ∧ (total_spent = 24) :=
by
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  have h1 : book_price_paid = (original_book_price / 8) := by
    sorry
  have h2 : total_spent = 24 := by
    sorry
  exact ⟨h1, h2⟩

end voldemort_spending_l212_212893


namespace three_digit_divisible_by_7_iff_last_two_digits_equal_l212_212723

-- Define the conditions as given in the problem
variable (a b c : ℕ)

-- Ensure the sum of the digits is 7, as given by the problem conditions
def sum_of_digits_eq_7 := a + b + c = 7

-- Ensure that it is a three-digit number
def valid_three_digit_number := a ≠ 0

-- Define what it means to be divisible by 7
def divisible_by_7 (n : ℕ) := n % 7 = 0

-- Define the problem statement in Lean
theorem three_digit_divisible_by_7_iff_last_two_digits_equal (h1 : sum_of_digits_eq_7 a b c) (h2 : valid_three_digit_number a) :
  divisible_by_7 (100 * a + 10 * b + c) ↔ b = c :=
by sorry

end three_digit_divisible_by_7_iff_last_two_digits_equal_l212_212723


namespace pentagon_area_is_correct_l212_212122

noncomputable def area_of_pentagon : ℕ :=
  let area_trapezoid := (1 / 2) * (25 + 28) * 30
  let area_triangle := (1 / 2) * 18 * 24
  area_trapezoid + area_triangle

theorem pentagon_area_is_correct (s1 s2 s3 s4 s5 : ℕ) (b1 b2 h1 b3 h2 : ℕ)
  (h₀ : s1 = 18) (h₁ : s2 = 25) (h₂ : s3 = 30) (h₃ : s4 = 28) (h₄ : s5 = 25)
  (h₅ : b1 = 25) (h₆ : b2 = 28) (h₇ : h1 = 30) (h₈ : b3 = 18) (h₉ : h2 = 24) :
  area_of_pentagon = 1011 := by
  -- placeholder for actual proof
  sorry

end pentagon_area_is_correct_l212_212122


namespace exists_positive_ℓ_l212_212427

theorem exists_positive_ℓ (k : ℕ) (h_prime: 0 < k) :
  ∃ ℓ : ℕ, 0 < ℓ ∧ 
  (∀ m n : ℕ, m > 0 → n > 0 → Nat.gcd m ℓ = 1 → Nat.gcd n ℓ = 1 →  m ^ m % ℓ = n ^ n % ℓ → m % k = n % k) :=
sorry

end exists_positive_ℓ_l212_212427


namespace ellen_painted_17_lilies_l212_212261

theorem ellen_painted_17_lilies :
  (∃ n : ℕ, n * 5 + 10 * 7 + 6 * 3 + 20 * 2 = 213) → 
    ∃ n : ℕ, n = 17 := 
by sorry

end ellen_painted_17_lilies_l212_212261


namespace unique_solution_condition_l212_212444

variable (c d x : ℝ)

-- Define the equation
def equation : Prop := 4 * x - 7 + c = d * x + 3

-- Lean theorem for the proof problem
theorem unique_solution_condition :
  (∃! x, equation c d x) ↔ d ≠ 4 :=
sorry

end unique_solution_condition_l212_212444


namespace geometric_seq_a3_l212_212421

theorem geometric_seq_a3 (a : ℕ → ℝ) (r : ℝ)
  (h1 : a 6 = a 3 * r^3)
  (h2 : a 9 = a 3 * r^6)
  (h3 : a 6 = 6)
  (h4 : a 9 = 9) : 
  a 3 = 4 := 
sorry

end geometric_seq_a3_l212_212421


namespace shelves_needed_is_five_l212_212215

-- Definitions for the conditions
def initial_bears : Nat := 15
def additional_bears : Nat := 45
def bears_per_shelf : Nat := 12

-- Adding the number of bears received to the initial stock
def total_bears : Nat := initial_bears + additional_bears

-- Calculating the number of shelves used
def shelves_used : Nat := total_bears / bears_per_shelf

-- Statement to prove
theorem shelves_needed_is_five : shelves_used = 5 :=
by
  -- Insert specific step only if necessary, otherwise use sorry
  sorry

end shelves_needed_is_five_l212_212215


namespace finite_decimal_representation_nat_numbers_l212_212342

theorem finite_decimal_representation_nat_numbers (n : ℕ) : 
  (∀ k : ℕ, k < n → (∃ u v : ℕ, (k + 1 = 2^u ∨ k + 1 = 5^v) ∨ (k - 1 = 2^u ∨ k -1  = 5^v))) ↔ 
  (n = 2 ∨ n = 3 ∨ n = 6) :=
by sorry

end finite_decimal_representation_nat_numbers_l212_212342


namespace A_investment_amount_l212_212369

theorem A_investment_amount
  (B_investment : ℝ) (C_investment : ℝ) 
  (total_profit : ℝ) (A_profit_share : ℝ)
  (h1 : B_investment = 4200)
  (h2 : C_investment = 10500)
  (h3 : total_profit = 14200)
  (h4 : A_profit_share = 4260) :
  ∃ (A_investment : ℝ), 
    A_profit_share / total_profit = A_investment / (A_investment + B_investment + C_investment) ∧ 
    A_investment = 6600 :=
by {
  sorry  -- Proof not required per instructions
}

end A_investment_amount_l212_212369


namespace difference_between_numbers_l212_212887

theorem difference_between_numbers (x y d : ℝ) (h1 : x + y = 10) (h2 : x - y = d) (h3 : x^2 - y^2 = 80) : d = 8 :=
by {
  sorry
}

end difference_between_numbers_l212_212887


namespace find_b_l212_212300

theorem find_b (a c S : ℝ) (h₁ : a = 5) (h₂ : c = 2) (h₃ : S = 4) : 
  b = Real.sqrt 17 ∨ b = Real.sqrt 41 := by
  sorry

end find_b_l212_212300


namespace find_number_l212_212729

theorem find_number (x : ℝ) (h : (x / 4) + 9 = 15) : x = 24 :=
by
  sorry

end find_number_l212_212729


namespace cubic_solution_unique_real_l212_212782

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end cubic_solution_unique_real_l212_212782


namespace brocard_inequalities_l212_212018

theorem brocard_inequalities (α β γ φ: ℝ) (h1: φ > 0) (h2: φ < π / 6)
  (h3: α > 0) (h4: β > 0) (h5: γ > 0) (h6: α + β + γ = π) : 
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := 
by 
  sorry

end brocard_inequalities_l212_212018


namespace maria_sandwich_count_l212_212430

open Nat

noncomputable def numberOfSandwiches (meat_choices cheese_choices topping_choices : Nat) :=
  (choose meat_choices 2) * (choose cheese_choices 2) * (choose topping_choices 2)

theorem maria_sandwich_count : numberOfSandwiches 12 11 8 = 101640 := by
  sorry

end maria_sandwich_count_l212_212430


namespace combined_final_selling_price_correct_l212_212868

def itemA_cost : Float := 180.0
def itemB_cost : Float := 220.0
def itemC_cost : Float := 130.0

def itemA_profit_margin : Float := 0.15
def itemB_profit_margin : Float := 0.20
def itemC_profit_margin : Float := 0.25

def itemA_tax_rate : Float := 0.05
def itemB_discount_rate : Float := 0.10
def itemC_tax_rate : Float := 0.08

def itemA_selling_price_before_tax := itemA_cost * (1 + itemA_profit_margin)
def itemB_selling_price_before_discount := itemB_cost * (1 + itemB_profit_margin)
def itemC_selling_price_before_tax := itemC_cost * (1 + itemC_profit_margin)

def itemA_final_price := itemA_selling_price_before_tax * (1 + itemA_tax_rate)
def itemB_final_price := itemB_selling_price_before_discount * (1 - itemB_discount_rate)
def itemC_final_price := itemC_selling_price_before_tax * (1 + itemC_tax_rate)

def combined_final_price := itemA_final_price + itemB_final_price + itemC_final_price

theorem combined_final_selling_price_correct : 
  combined_final_price = 630.45 :=
by
  -- proof would go here
  sorry

end combined_final_selling_price_correct_l212_212868


namespace total_blue_marbles_correct_l212_212554

def total_blue_marbles (j t e : ℕ) : ℕ :=
  j + t + e

theorem total_blue_marbles_correct :
  total_blue_marbles 44 24 36 = 104 :=
by
  sorry

end total_blue_marbles_correct_l212_212554


namespace transformation_power_of_two_l212_212568

theorem transformation_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ s : ℕ, 2 ^ s ≥ n :=
by sorry

end transformation_power_of_two_l212_212568


namespace total_num_of_cars_l212_212726

-- Define conditions
def row_from_front := 14
def row_from_left := 19
def row_from_back := 11
def row_from_right := 16

-- Compute total number of rows from front to back
def rows_front_to_back : ℕ := (row_from_front - 1) + 1 + (row_from_back - 1)

-- Compute total number of rows from left to right
def rows_left_to_right : ℕ := (row_from_left - 1) + 1 + (row_from_right - 1)

theorem total_num_of_cars :
  rows_front_to_back = 24 ∧
  rows_left_to_right = 34 ∧
  24 * 34 = 816 :=
by
  sorry

end total_num_of_cars_l212_212726


namespace gcf_54_81_l212_212894

theorem gcf_54_81 : Nat.gcd 54 81 = 27 :=
by sorry

end gcf_54_81_l212_212894


namespace min_colors_needed_l212_212643

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  max (abs (c1.1 - c2.1)) (abs (c1.2 - c2.2))

def color (c : cell) : ℤ :=
  (c.1 + c.2) % 4

theorem min_colors_needed : 4 = 4 :=
sorry

end min_colors_needed_l212_212643


namespace hyperbola_equation_slope_of_line_l_l212_212820

-- Define the hyperbola and its conditions
structure Hyperbola :=
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (equation : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)

-- Given conditions
def perpendicular_line (a b : ℝ) := (b / a = 2)
def distance_from_vertex (a : ℝ) := (2 * a / sqrt(5) = 2 * sqrt(5) / 5)

-- Define the points and midpoint condition
structure MidpointCondition (A B M : ℝ × ℝ) :=
  (midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- Prove the equation of the hyperbola
theorem hyperbola_equation : 
  ∃ (C : Hyperbola), 
  perpendicular_line C.a C.b ∧ 
  distance_from_vertex 1 ∧ 
  C.equation = (λ x y, x^2 - y^2 / 4 = 1) :=
sorry

-- Prove the slope of the line l
theorem slope_of_line_l : 
  ∀ (A B M : ℝ × ℝ),
  MidpointCondition A B M → M = (3, 2) → ∃ k : ℝ,
  k = 6 :=
sorry

end hyperbola_equation_slope_of_line_l_l212_212820


namespace rowing_velocity_l212_212618

theorem rowing_velocity (v : ℝ) : 
  (∀ (d : ℝ) (s : ℝ) (total_time : ℝ), 
    s = 10 ∧ 
    total_time = 30 ∧ 
    d = 144 ∧ 
    (d / (s - v) + d / (s + v)) = total_time) → 
  v = 2 := 
by
  sorry

end rowing_velocity_l212_212618


namespace seashells_given_l212_212556

theorem seashells_given (initial left given : ℕ) (h1 : initial = 8) (h2 : left = 2) (h3 : given = initial - left) : given = 6 := by
  sorry

end seashells_given_l212_212556


namespace inequality_abc_l212_212274

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (b * c / a) + (a * c / b) + (a * b / c) ≥ a + b + c := 
  sorry

end inequality_abc_l212_212274


namespace workers_distribution_l212_212908

theorem workers_distribution (x y : ℕ) (h1 : x + y = 32) (h2 : 2 * 5 * x = 6 * y) : 
  (∃ x y : ℕ, x + y = 32 ∧ 2 * 5 * x = 6 * y) :=
sorry

end workers_distribution_l212_212908


namespace division_remainder_l212_212320

theorem division_remainder (dividend divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15) 
  (h_quotient : quotient = 9) 
  (h_dividend_eq : dividend = 136) 
  (h_eq : dividend = (divisor * quotient) + remainder) : 
  remainder = 1 :=
by
  sorry

end division_remainder_l212_212320


namespace solution_set_f_lt_g_l212_212532

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists according to the given conditions

lemma f_at_one : f 1 = -2 := sorry

lemma f_derivative_neg (x : ℝ) : (deriv f x) < 0 := sorry

def g (x : ℝ) : ℝ := x - 3

lemma g_at_one : g 1 = -2 := sorry

theorem solution_set_f_lt_g :
  {x : ℝ | f x < g x} = {x : ℝ | 1 < x} :=
sorry

end solution_set_f_lt_g_l212_212532


namespace bert_average_words_in_crossword_l212_212499

theorem bert_average_words_in_crossword :
  (10 * 35 + 4 * 65) / (10 + 4) = 43.57 :=
by
  sorry

end bert_average_words_in_crossword_l212_212499


namespace minimum_red_chips_l212_212478

variable (w b r : ℕ)

axiom C1 : b ≥ (1 / 3 : ℚ) * w
axiom C2 : b ≤ (1 / 4 : ℚ) * r
axiom C3 : w + b ≥ 75

theorem minimum_red_chips : r = 76 := by sorry

end minimum_red_chips_l212_212478


namespace red_balls_approximation_l212_212667

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l212_212667


namespace circle_radius_l212_212455

theorem circle_radius (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y + 1 = 0) : 
    ∃ r : ℝ, r = 2 ∧ (x - 2)^2 + (y - 1)^2 = r^2 :=
by
  sorry

end circle_radius_l212_212455


namespace correct_average_marks_l212_212027

theorem correct_average_marks
  (n : ℕ) (avg_mks wrong_mk correct_mk correct_avg_mks : ℕ)
  (H1 : n = 10)
  (H2 : avg_mks = 100)
  (H3 : wrong_mk = 50)
  (H4 : correct_mk = 10)
  (H5 : correct_avg_mks = 96) :
  (n * avg_mks - wrong_mk + correct_mk) / n = correct_avg_mks :=
by
  sorry

end correct_average_marks_l212_212027


namespace solution_to_cubic_equation_l212_212786

theorem solution_to_cubic_equation :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
begin
  sorry
end

end solution_to_cubic_equation_l212_212786


namespace least_integer_value_l212_212266

-- Define the condition and then prove the statement
theorem least_integer_value (x : ℤ) (h : 3 * |x| - 2 > 13) : x = -6 :=
by
  sorry

end least_integer_value_l212_212266


namespace trig_product_computation_l212_212380

theorem trig_product_computation :
  (1 - sin (Real.pi / 12)) * (1 - sin (5 * Real.pi / 12)) *
  (1 - sin (7 * Real.pi / 12)) * (1 - sin (11 * Real.pi / 12)) 
  = 1 / 16 :=
by 
  sorry

end trig_product_computation_l212_212380


namespace find_expression_for_x_l212_212529

variable (x : ℝ) (hx : x^3 + (1 / x^3) = -52)

theorem find_expression_for_x : x + (1 / x) = -4 :=
by sorry

end find_expression_for_x_l212_212529


namespace difference_of_sum_l212_212060

theorem difference_of_sum (a b c : ℤ) (h1 : a = 11) (h2 : b = 13) (h3 : c = 15) :
  (b + c) - a = 17 := by
  sorry

end difference_of_sum_l212_212060


namespace ratio_of_areas_of_triangles_l212_212336

noncomputable def area_ratio_triangle_GHI_JKL
  (a_GHI b_GHI c_GHI : ℕ) (a_JKL b_JKL c_JKL : ℕ) 
  (alt_ratio_GHI : ℕ × ℕ) (alt_ratio_JKL : ℕ × ℕ) : ℚ :=
  let area_GHI := (a_GHI * b_GHI) / 2
  let area_JKL := (a_JKL * b_JKL) / 2
  area_GHI / area_JKL

theorem ratio_of_areas_of_triangles :
  let GHI_sides := (7, 24, 25)
  let JKL_sides := (9, 40, 41)
  area_ratio_triangle_GHI_JKL 7 24 25 9 40 41 (2, 3) (4, 5) = 7 / 15 :=
by
  sorry

end ratio_of_areas_of_triangles_l212_212336


namespace LeahsCoinsValueIs68_l212_212559

def LeahsCoinsWorthInCents (p n d : Nat) : Nat :=
  p * 1 + n * 5 + d * 10

theorem LeahsCoinsValueIs68 {p n d : Nat} (h1 : p + n + d = 17) (h2 : n + 2 = p) :
  LeahsCoinsWorthInCents p n d = 68 := by
  sorry

end LeahsCoinsValueIs68_l212_212559


namespace beth_marbles_left_l212_212765

theorem beth_marbles_left :
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  T - (L_red + L_blue + L_yellow) = 42 :=
by
  let T := 72
  let C := T / 3
  let L_red := 5
  let L_blue := 2 * L_red
  let L_yellow := 3 * L_red
  have h1 : T - (L_red + L_blue + L_yellow) = 42 := rfl
  exact h1

end beth_marbles_left_l212_212765


namespace none_takes_own_hat_probability_l212_212039

noncomputable def probability_none_takes_own_hat : ℚ :=
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  probability

theorem none_takes_own_hat_probability : probability_none_takes_own_hat = 1 / 3 :=
by
  have total_arrangements := 3.factorial
  have derangements : ℕ := 2
  have probability := (derangements : ℚ) / (total_arrangements : ℚ)
  show probability_none_takes_own_hat = 1 / 3
  sorry

end none_takes_own_hat_probability_l212_212039


namespace molecular_weight_of_BaF2_l212_212046

theorem molecular_weight_of_BaF2 (mw_6_moles : ℕ → ℕ) (h : mw_6_moles 6 = 1050) : mw_6_moles 1 = 175 :=
by
  sorry

end molecular_weight_of_BaF2_l212_212046


namespace convert_to_rectangular_form_l212_212253

theorem convert_to_rectangular_form :
  2 * Real.sqrt 3 * Complex.exp (13 * Real.pi * Complex.I / 6) = 3 + Complex.I * Real.sqrt 3 :=
by
  sorry

end convert_to_rectangular_form_l212_212253


namespace infinite_rational_set_l212_212874

noncomputable def is_rational (x : ℚ) : Prop :=
  ∃ (a b : ℚ), x - 1 = a^2 ∧ 4 * x + 1 = b^2

theorem infinite_rational_set : 
  {x : ℚ | ∃ (a b : ℚ), x - 1 = a^2 ∧ 4 * x + 1 = b^2}.infinite :=
  sorry

end infinite_rational_set_l212_212874


namespace brad_books_this_month_l212_212346

-- Define the number of books William read last month
def william_books_last_month : ℕ := 6

-- Define the number of books Brad read last month
def brad_books_last_month : ℕ := 3 * william_books_last_month

-- Define the number of books Brad read this month as a variable
variable (B : ℕ)

-- Define the total number of books William read over the two months
def total_william_books (B : ℕ) : ℕ := william_books_last_month + 2 * B

-- Define the total number of books Brad read over the two months
def total_brad_books (B : ℕ) : ℕ := brad_books_last_month + B

-- State the condition that William read 4 more books than Brad
def william_read_more_books_condition (B : ℕ) : Prop := total_william_books B = total_brad_books B + 4

-- State the theorem to be proven
theorem brad_books_this_month (B : ℕ) : william_read_more_books_condition B → B = 16 :=
by
  sorry

end brad_books_this_month_l212_212346


namespace find_rate_of_interest_l212_212373

/-- At what rate percent on simple interest will Rs. 25,000 amount to Rs. 34,500 in 5 years? 
    Given Principal (P) = Rs. 25,000, Amount (A) = Rs. 34,500, Time (T) = 5 years. 
    We need to find the Rate (R). -/
def principal : ℝ := 25000
def amount : ℝ := 34500
def time : ℝ := 5

theorem find_rate_of_interest (P A T : ℝ) : 
  P = principal → 
  A = amount → 
  T = time → 
  ∃ R : ℝ, R = 7.6 :=
by
  intros hP hA hT
  -- proof goes here
  sorry

end find_rate_of_interest_l212_212373


namespace proof_problem_l212_212127

noncomputable def log2 (n : ℝ) : ℝ := Real.log n / Real.log 2

theorem proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/2 * log2 x + 1/3 * log2 y = 1) : x^3 * y^2 = 64 := 
sorry 

end proof_problem_l212_212127


namespace div_by_7_l212_212872

theorem div_by_7 (k : ℕ) : (2^(6*k + 1) + 3^(6*k + 1) + 5^(6*k + 1)) % 7 = 0 := by
  sorry

end div_by_7_l212_212872


namespace unique_3_coloring_edges_count_l212_212990

open GraphTheory

variable (G : SimpleGraph V) [Fintype V]
variable (n : ℕ) (h : 3 ≤ n) (unique_3_coloring : ∀ (c₁ c₂ : V → ℕ), (∀ v, c₁ v < 3 ∧ c₂ v < 3) ∧ G.ProperColoring c₁ ∧ G.ProperColoring c₂ → c₁ = c₂)

theorem unique_3_coloring_edges_count :
    (Fintype.card V ≥ 3 ∧ unique_3_coloring) → G.edgeCount ≥ 2 * Fintype.card V - 3 :=
sorry

end unique_3_coloring_edges_count_l212_212990


namespace min_distance_l212_212850

open Complex

theorem min_distance (z : ℂ) (hz : abs (z + 2 - 2*I) = 1) : abs (z - 2 - 2*I) = 3 :=
sorry

end min_distance_l212_212850


namespace base_digits_equality_l212_212206

theorem base_digits_equality (b : ℕ) (h_condition : b^5 ≤ 200 ∧ 200 < b^6) : b = 2 := 
by {
  sorry -- proof not required as per the instructions
}

end base_digits_equality_l212_212206


namespace initial_books_l212_212364

-- Define the variables and conditions
def B : ℕ := 75
def loaned_books : ℕ := 60
def returned_books : ℕ := (70 * loaned_books) / 100
def not_returned_books : ℕ := loaned_books - returned_books
def end_of_month_books : ℕ := 57

-- State the theorem
theorem initial_books (h1 : returned_books = 42)
                      (h2 : end_of_month_books = 57)
                      (h3 : loaned_books = 60) :
  B = end_of_month_books + not_returned_books :=
by sorry

end initial_books_l212_212364


namespace problem_graph_empty_l212_212257

open Real

theorem problem_graph_empty : ∀ x y : ℝ, ¬ (x^2 + 3 * y^2 - 4 * x - 12 * y + 28 = 0) :=
by
  intro x y
  -- Apply the contradiction argument based on the conditions given
  sorry


end problem_graph_empty_l212_212257


namespace E_union_F_eq_univ_l212_212286

-- Define the given conditions
def E : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def F (a : ℝ) : Set ℝ := { x | x - 5 < a }
def I : Set ℝ := Set.univ
axiom a_gt_6 : ∃ a : ℝ, a > 6 ∧ 11 ∈ F a

-- State the theorem
theorem E_union_F_eq_univ (a : ℝ) (h₁ : a > 6) (h₂ : 11 ∈ F a) : E ∪ F a = I := by
  sorry

end E_union_F_eq_univ_l212_212286


namespace maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l212_212055

theorem maximum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ max_range : ℕ, max_range = a) :=
by {
  use a,
  sorry
}

theorem minimum_range_of_walk (a b : ℕ) (h : a > b) : 
  (∃ min_range : ℕ, min_range = a - b) :=
by {
  use a - b,
  sorry
}

theorem number_of_max_range_sequences (a b : ℕ) (h : a > b) : 
  (∃ num_sequences : ℕ, num_sequences = b + 1) :=
by {
  use b + 1,
  sorry
}

end maximum_range_of_walk_minimum_range_of_walk_number_of_max_range_sequences_l212_212055


namespace tesses_ride_is_longer_l212_212321

noncomputable def tesses_total_distance : ℝ := 0.75 + 0.85 + 1.15
noncomputable def oscars_total_distance : ℝ := 0.25 + 1.35

theorem tesses_ride_is_longer :
  (tesses_total_distance - oscars_total_distance) = 1.15 := by
  sorry

end tesses_ride_is_longer_l212_212321


namespace inequality_solution_set_inequality_range_of_a_l212_212948

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : a = -8) :
  (|x - 3| + |x + 2| ≤ |a + 1|) ↔ (-3 ≤ x ∧ x ≤ 4) :=
by sorry

theorem inequality_range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x + 2| ≤ |a + 1|) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by sorry

end inequality_solution_set_inequality_range_of_a_l212_212948


namespace real_solution_unique_l212_212803

theorem real_solution_unique (x : ℝ) : 
  (x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) ↔ x = 6 := 
begin
  sorry
end

end real_solution_unique_l212_212803


namespace sum_s_r_values_l212_212148

def r_values : List ℤ := [-2, -1, 0, 1, 3]
def r_range : List ℤ := [-1, 0, 1, 3, 5]

def s (x : ℤ) : ℤ := if 1 ≤ x then 2 * x + 1 else 0

theorem sum_s_r_values :
  (s 1) + (s 3) + (s 5) = 21 :=
by
  sorry

end sum_s_r_values_l212_212148


namespace irrational_roots_of_odd_coeff_quad_l212_212165

theorem irrational_roots_of_odd_coeff_quad (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, a * r^2 + b * r + c = 0 := 
sorry

end irrational_roots_of_odd_coeff_quad_l212_212165


namespace ball_distribution_l212_212836

theorem ball_distribution : 
  (finset.sum 
    (finset.image (λ (p : sym2 (fin 4)), 
                    match p with
                    | (a, b, c, d) => 
                      if a + b + c + d = 5 then 1 else 0
                    end) 
    (sym2 (fin 5))).card).to_nat = 56 :=
sorry

end ball_distribution_l212_212836


namespace exists_n_cos_eq_l212_212640

theorem exists_n_cos_eq :
  ∃ (n : ℤ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n:ℝ).to_degrees = Real.cos 942 := by
  sorry

end exists_n_cos_eq_l212_212640


namespace ball_distribution_l212_212835

theorem ball_distribution : 
  (finset.sum 
    (finset.image (λ (p : sym2 (fin 4)), 
                    match p with
                    | (a, b, c, d) => 
                      if a + b + c + d = 5 then 1 else 0
                    end) 
    (sym2 (fin 5))).card).to_nat = 56 :=
sorry

end ball_distribution_l212_212835


namespace KrystianaChargesForSecondFloorRooms_Theorem_l212_212978

noncomputable def KrystianaChargesForSecondFloorRooms (X : ℝ) : Prop :=
  let costFirstFloor := 3 * 15
  let costThirdFloor := 3 * (2 * 15)
  let totalEarnings := costFirstFloor + 3 * X + costThirdFloor
  totalEarnings = 165 → X = 10

-- This is the statement only. The proof is not included.
theorem KrystianaChargesForSecondFloorRooms_Theorem : KrystianaChargesForSecondFloorRooms 10 :=
sorry

end KrystianaChargesForSecondFloorRooms_Theorem_l212_212978


namespace probability_white_given_red_l212_212672

-- Define the total number of balls initially
def total_balls := 10

-- Define the number of red balls, white balls, and black balls
def red_balls := 3
def white_balls := 2
def black_balls := 5

-- Define the event A: Picking a red ball on the first draw
def event_A := red_balls

-- Define the event B: Picking a white ball on the second draw
-- Number of balls left after picking one red ball
def remaining_balls_after_A := total_balls - 1

-- Define the event AB: Picking a red ball first and then a white ball
def event_AB := red_balls * white_balls

-- Calculate the probability P(B|A)
def P_B_given_A := event_AB / (event_A * remaining_balls_after_A)

-- Prove the probability of picking a white ball on the second draw given that the first ball picked is a red ball
theorem probability_white_given_red : P_B_given_A = (2 / 9) := by
  sorry

end probability_white_given_red_l212_212672


namespace packages_ratio_l212_212481

theorem packages_ratio (packages_yesterday packages_today : ℕ)
  (h1 : packages_yesterday = 80)
  (h2 : packages_today + packages_yesterday = 240) :
  (packages_today / packages_yesterday) = 2 :=
by
  sorry

end packages_ratio_l212_212481


namespace gcd_triang_num_gcd_triang_num_max_l212_212396

open Nat

theorem gcd_triang_num (n : ℕ) : n > 0 → gcd (3 * (n * (n + 1) / 2) + n) (n + 3) ≤ 12 :=
by sorry

theorem gcd_triang_num_max (n : ℕ) : ∃ k, n = 6*k - 3 ∧ gcd (3 * (n * (n + 1) / 2) + n) (n + 3) = 12 :=
by sorry

end gcd_triang_num_gcd_triang_num_max_l212_212396


namespace shareCoins_l212_212550

theorem shareCoins (a b c d e d : ℝ)
  (h1 : b = a - d)
  (h2 : ((a-2*d) + b = a + (a+d) + (a+2*d)))
  (h3 : (a-2*d) + b + a + (a+d) + (a+2*d) = 5) :
  b = 7 / 6 :=
by
  sorry

end shareCoins_l212_212550


namespace neither_sufficient_nor_necessary_l212_212646

theorem neither_sufficient_nor_necessary 
  (a b c : ℝ) : 
  ¬ ((∀ x : ℝ, b^2 - 4 * a * c < 0 → a * x^2 + b * x + c > 0) ∧ 
     (∀ x : ℝ, a * x^2 + b * x + c > 0 → b^2 - 4 * a * c < 0)) := 
by
  sorry

end neither_sufficient_nor_necessary_l212_212646


namespace min_value_x_y_l212_212279

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : 
  x + y ≥ 20 :=
sorry

end min_value_x_y_l212_212279


namespace calculate_square_of_complex_l212_212084

theorem calculate_square_of_complex (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end calculate_square_of_complex_l212_212084


namespace solve_for_n_l212_212703

-- Define the equation as a Lean expression
def equation (n : ℚ) : Prop :=
  (2 - n) / (n + 1) + (2 * n - 4) / (2 - n) = 1

theorem solve_for_n : ∃ n : ℚ, equation n ∧ n = -1 / 4 := by
  sorry

end solve_for_n_l212_212703


namespace tetrahedron_circumsphere_radius_l212_212491

theorem tetrahedron_circumsphere_radius :
  ∃ (r : ℝ), 
    (∀ (A B C P : ℝ × ℝ × ℝ),
      (dist A B = 5) ∧
      (dist A C = 5) ∧
      (dist A P = 5) ∧
      (dist B C = 5) ∧
      (dist B P = 5) ∧
      (dist C P = 6) →
      r = (20 * Real.sqrt 39) / 39) :=
sorry

end tetrahedron_circumsphere_radius_l212_212491


namespace probability_non_zero_product_l212_212814

open ProbabilityTheory

noncomputable def probability_no_one (a b c d : ℕ) : ℚ :=
  if h : 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 then
    (5/6) * (5/6) * (5/6) * (5/6)
  else 0

theorem probability_non_zero_product :
  let a b c d : ℕ := -- numbers on the top faces of four standard dice
  probability_no_one a b c d = 625 / 1296 :=
by
  sorry

end probability_non_zero_product_l212_212814


namespace circle_line_intersection_l212_212825

theorem circle_line_intersection (a : ℝ) : 
  (∃ x y : ℝ, (x - y + 1 = 0) ∧ ((x - a)^2 + y^2 = 2)) ↔ -3 ≤ a ∧ a ≤ 1 := 
by
  sorry

end circle_line_intersection_l212_212825


namespace solve_inequality_l212_212705

theorem solve_inequality (x : Real) : 
  (abs ((3 * x + 2) / (x - 2)) > 3) ↔ (x ∈ Set.Ioo (2 / 3) 2) := by
  sorry

end solve_inequality_l212_212705


namespace square_of_binomial_example_l212_212896

theorem square_of_binomial_example : (23^2 + 2 * 23 * 2 + 2^2 = 625) :=
by
  sorry

end square_of_binomial_example_l212_212896


namespace triangle_at_most_one_obtuse_l212_212168

-- Define the notion of a triangle and obtuse angle
def isTriangle (A B C: ℝ) : Prop := (A + B > C) ∧ (A + C > B) ∧ (B + C > A)
def isObtuseAngle (theta: ℝ) : Prop := 90 < theta ∧ theta < 180

-- A theorem to prove that a triangle cannot have more than one obtuse angle 
theorem triangle_at_most_one_obtuse (A B C: ℝ) (angleA angleB angleC : ℝ) 
    (h1 : isTriangle A B C)
    (h2 : isObtuseAngle angleA)
    (h3 : isObtuseAngle angleB)
    (h4 : angleA + angleB + angleC = 180):
    false :=
by
  sorry

end triangle_at_most_one_obtuse_l212_212168


namespace total_cups_used_l212_212591

theorem total_cups_used (butter flour sugar : ℕ) (h1 : 2 * sugar = 3 * butter) (h2 : 5 * sugar = 3 * flour) (h3 : sugar = 12) : butter + flour + sugar = 40 :=
by
  sorry

end total_cups_used_l212_212591


namespace total_rope_in_inches_l212_212680

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l212_212680


namespace bakery_baguettes_l212_212582

theorem bakery_baguettes : 
  ∃ B : ℕ, 
  (∃ B : ℕ, 3 * B - 138 = 6) ∧ 
  B = 48 :=
by
  sorry

end bakery_baguettes_l212_212582


namespace Jamie_liquid_limit_l212_212861

theorem Jamie_liquid_limit :
  let milk_ounces := 8
  let grape_juice_ounces := 16
  let water_bottle_limit := 8
  let already_consumed := milk_ounces + grape_juice_ounces
  let max_before_bathroom := already_consumed + water_bottle_limit
  max_before_bathroom = 32 :=
by
  sorry

end Jamie_liquid_limit_l212_212861


namespace range_of_a3_l212_212952

theorem range_of_a3 (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a (n + 1) + a n = 4 * n + 3)
  (h2 : ∀ n : ℕ, n > 0 → a n + 2 * n^2 ≥ 0) 
  : 2 ≤ a 3 ∧ a 3 ≤ 19 := 
sorry

end range_of_a3_l212_212952


namespace train_length_is_correct_l212_212077

variable (speed_km_hr : ℕ) (time_sec : ℕ)
def convert_speed (speed_km_hr : ℕ) : ℚ :=
  (speed_km_hr * 1000 : ℚ) / 3600

noncomputable def length_of_train (speed_km_hr time_sec : ℕ) : ℚ :=
  convert_speed speed_km_hr * time_sec

theorem train_length_is_correct (speed_km_hr : ℕ) (time_sec : ℕ) (h₁ : speed_km_hr = 300) (h₂ : time_sec = 33) :
  length_of_train speed_km_hr time_sec = 2750 := by
  sorry

end train_length_is_correct_l212_212077


namespace rachel_budget_proof_l212_212698

-- Define the prices Sara paid for shoes and the dress
def shoes_price : ℕ := 50
def dress_price : ℕ := 200

-- Total amount Sara spent
def sara_total : ℕ := shoes_price + dress_price

-- Rachel's budget should be double of Sara's total spending
def rachels_budget : ℕ := 2 * sara_total

-- The theorem statement
theorem rachel_budget_proof : rachels_budget = 500 := by
  unfold rachels_budget sara_total shoes_price dress_price
  rfl

end rachel_budget_proof_l212_212698


namespace tabitha_item_cost_l212_212445

theorem tabitha_item_cost :
  ∀ (start_money gave_mom invest fraction_remain spend item_count remain_money item_cost : ℝ),
    start_money = 25 →
    gave_mom = 8 →
    invest = (start_money - gave_mom) / 2 →
    fraction_remain = start_money - gave_mom - invest →
    spend = fraction_remain - remain_money →
    item_count = 5 →
    remain_money = 6 →
    item_cost = spend / item_count →
    item_cost = 0.5 :=
by
  intros
  sorry

end tabitha_item_cost_l212_212445


namespace sum_of_x_y_l212_212848

theorem sum_of_x_y (x y : ℚ) (h1 : 1/x + 1/y = 5) (h2 : 1/x - 1/y = -9) : x + y = -5/14 := 
by
  sorry

end sum_of_x_y_l212_212848


namespace parabola_focus_is_centroid_l212_212276

-- Define the given points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -6)

-- Calculate the centroid of triangle ABC
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define the focus of the parabola and the centroid of the triangle
def focus_of_parabola (a : ℝ) : ℝ × ℝ := (a / 4, 0)
def centroid_ABC := centroid A B C

-- Statement of the proof problem
theorem parabola_focus_is_centroid :
  focus_of_parabola 8 = centroid_ABC := 
by
  sorry

end parabola_focus_is_centroid_l212_212276


namespace cylinder_radius_and_volume_l212_212748

theorem cylinder_radius_and_volume
  (h : ℝ) (surface_area : ℝ) :
  h = 8 ∧ surface_area = 130 * Real.pi →
  ∃ (r : ℝ) (V : ℝ), r = 5 ∧ V = 200 * Real.pi := by
  sorry

end cylinder_radius_and_volume_l212_212748


namespace real_roots_for_all_K_l212_212271

theorem real_roots_for_all_K (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x-1) * (x-2) + 2 * x :=
sorry

end real_roots_for_all_K_l212_212271


namespace sqrt_eq_l212_212397

noncomputable def sqrt_22500 := 150

theorem sqrt_eq (h : sqrt_22500 = 150) : Real.sqrt 0.0225 = 0.15 :=
sorry

end sqrt_eq_l212_212397


namespace vertex_set_is_parabola_l212_212865

variables (a c k : ℝ) (ha : a > 0) (hc : c > 0) (hk : k ≠ 0)

theorem vertex_set_is_parabola :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) :=
sorry

end vertex_set_is_parabola_l212_212865


namespace find_smaller_number_l212_212712

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  -- Proof steps will be filled in here
  sorry

end find_smaller_number_l212_212712


namespace maciek_total_cost_l212_212064

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l212_212064


namespace minimum_a_l212_212525

noncomputable def f (x : ℝ) := x - Real.exp (x - Real.exp 1)

theorem minimum_a (a : ℝ) (x1 x2 : ℝ) (hx : x2 - x1 ≥ Real.exp 1)
  (hy : Real.exp x1 = 1 + Real.log (x2 - a)) : a ≥ Real.exp 1 - 1 :=
by
  sorry

end minimum_a_l212_212525


namespace volume_of_spheres_l212_212462

noncomputable def sphere_volume (a : ℝ) : ℝ :=
  (4 / 3) * Real.pi * ((3 * a - a * Real.sqrt 3) / 4)^3

theorem volume_of_spheres (a : ℝ) : 
  ∃ r : ℝ, r = (3 * a - a * Real.sqrt 3) / 4 ∧ 
  sphere_volume a = (4 / 3) * Real.pi * r^3 := 
sorry

end volume_of_spheres_l212_212462


namespace mr_desmond_toys_l212_212005

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l212_212005


namespace division_of_floats_l212_212339

theorem division_of_floats : 4.036 / 0.04 = 100.9 :=
by
  sorry

end division_of_floats_l212_212339


namespace max_knights_between_knights_l212_212234

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l212_212234


namespace moles_NaCl_formed_in_reaction_l212_212091

noncomputable def moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) : ℕ :=
  if moles_NaOH = 1 ∧ moles_HCl = 1 then 1 else 0

theorem moles_NaCl_formed_in_reaction : moles_of_NaCl_formed 1 1 = 1 := 
by
  sorry

end moles_NaCl_formed_in_reaction_l212_212091


namespace florida_vs_georgia_license_plates_l212_212690

theorem florida_vs_georgia_license_plates :
  26 ^ 4 * 10 ^ 3 - 26 ^ 3 * 10 ^ 3 = 439400000 := by
  -- proof is omitted as directed
  sorry

end florida_vs_georgia_license_plates_l212_212690


namespace find_m_through_point_l212_212104

theorem find_m_through_point :
  ∃ m : ℝ, ∀ (x y : ℝ), ((y = (m - 1) * x - 4) ∧ (x = 2) ∧ (y = 4)) → m = 5 :=
by 
  -- Sorry can be used here to skip the proof as instructed
  sorry

end find_m_through_point_l212_212104


namespace no_pairs_exist_l212_212265

theorem no_pairs_exist (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : (1/a + 1/b = 2/(a+b)) → False :=
by
  sorry

end no_pairs_exist_l212_212265


namespace fourth_number_is_57_l212_212486

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (.+.) 0

theorem fourth_number_is_57 : 
  ∃ (N : ℕ), N < 100 ∧ 177 + N = 4 * (33 + digit_sum N) ∧ N = 57 :=
by {
  sorry
}

end fourth_number_is_57_l212_212486


namespace find_total_results_l212_212999

noncomputable def total_results (S : ℕ) (n : ℕ) (sum_first6 sum_last6 sixth_result : ℕ) :=
  (S = 52 * n) ∧ (sum_first6 = 6 * 49) ∧ (sum_last6 = 6 * 52) ∧ (sixth_result = 34)

theorem find_total_results {S n sum_first6 sum_last6 sixth_result : ℕ} :
  total_results S n sum_first6 sum_last6 sixth_result → n = 11 :=
by
  intros h
  sorry

end find_total_results_l212_212999


namespace inverse_function_value_l212_212960

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + 4) :
  f⁻¹ 58 = 3 :=
by sorry

end inverse_function_value_l212_212960


namespace positive_solution_x_l212_212405

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 8 - x - 4 * y) (h2 : y * z = 12 - 3 * y - 6 * z) (h3 : x * z = 40 - 5 * x - 2 * z) (hy : y = 3) (hz : z = -1) : x = 6 :=
by
  sorry

end positive_solution_x_l212_212405


namespace ratio_of_mistakes_l212_212856

theorem ratio_of_mistakes (riley_mistakes team_mistakes : ℕ) 
  (h_riley : riley_mistakes = 3) (h_team : team_mistakes = 17) : 
  (team_mistakes - riley_mistakes) / riley_mistakes = 14 / 3 := 
by 
  sorry

end ratio_of_mistakes_l212_212856


namespace probability_at_least_3_speak_l212_212881

-- Define the conditions in Lean 4
def prob_speak_per_baby : ℚ := 1 / 3
def total_babies : ℕ := 7

-- State the problem
theorem probability_at_least_3_speak :
  let prob_no_speak := (2 / 3)^total_babies
  let prob_exactly_1_speak := total_babies * (1 / 3) * (2 / 3)^(total_babies - 1)
  let prob_exactly_2_speak := nat.choose total_babies 2 * (1 / 3)^2 * (2 / 3)^(total_babies - 2)
  prob_no_speak + prob_exactly_1_speak + prob_exactly_2_speak = 1248 / 2187 →
  (1 - (prob_no_speak + prob_exactly_1_speak + prob_exactly_2_speak)) = 939 / 2187 :=
sorry

end probability_at_least_3_speak_l212_212881


namespace cubic_function_decreasing_l212_212711

theorem cubic_function_decreasing (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → (a ≤ 0) := 
by 
  sorry

end cubic_function_decreasing_l212_212711


namespace probability_no_cowboys_picks_own_hat_l212_212038

def derangements (n : Nat) : Nat :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

theorem probability_no_cowboys_picks_own_hat : 
  let total_arrangements := Nat.factorial 3
  let favorable_arrangements := derangements 3
  let probability := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)
  probability = 1 / 3 :=
by
  let total_arrangements := 6
  let favorable_arrangements := 2
  have h : probability = (2 : ℚ) / (6 : ℚ) := by rfl
  rw h
  calc (2 : ℚ) / (6 : ℚ)
          = 1 / 3 : by norm_num
  rw [h]
  sorry

end probability_no_cowboys_picks_own_hat_l212_212038


namespace max_knights_between_knights_l212_212235

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l212_212235


namespace oscar_leap_vs_elmer_stride_l212_212967

/--
Given:
1. The 51st telephone pole is exactly 6600 feet from the first pole.
2. Elmer the emu takes 50 equal strides to walk between consecutive telephone poles.
3. Oscar the ostrich can cover the same distance in 15 equal leaps.
4. There are 50 gaps between the 51 poles.

Prove:
Oscar's leap is 6 feet longer than Elmer's stride.
-/
theorem oscar_leap_vs_elmer_stride : 
  let total_distance := 6600 
  let elmer_strides_per_gap := 50
  let oscar_leaps_per_gap := 15
  let num_gaps := 50
  let elmer_total_strides := elmer_strides_per_gap * num_gaps
  let oscar_total_leaps := oscar_leaps_per_gap * num_gaps
  let elmer_stride_length := total_distance / elmer_total_strides
  let oscar_leap_length := total_distance / oscar_total_leaps
  oscar_leap_length - elmer_stride_length = 6 := 
by {
  -- The proof would go here.
  sorry
}

end oscar_leap_vs_elmer_stride_l212_212967


namespace gold_per_hour_l212_212673

section
variable (t : ℕ) (g_chest g_small_bag total_gold : ℕ)

def g_per_hour (t : ℕ) (g_chest g_small_bag : ℕ) :=
  let total_gold := g_chest + 2 * g_small_bag
  total_gold / t

theorem gold_per_hour (h1 : t = 8) (h2 : g_chest = 100) (h3 : g_small_bag = g_chest / 2) :
  g_per_hour t g_chest g_small_bag = 25 := by
  -- substitute the given values
  have h4 : g_small_bag = 50 := h3
  have h5 : total_gold = g_chest + 2 * g_small_bag := by
    rw [h2, h4]
    rfl
  have h6 : g_per_hour t g_chest g_small_bag = total_gold / t := rfl
  rw [h1, h2, h4, h5] at h6
  exact h6
end

end gold_per_hour_l212_212673


namespace ratio_of_sum_and_difference_l212_212456

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : (x + y) / (x - y) = x / y) : x / y = 1 + Real.sqrt 2 :=
sorry

end ratio_of_sum_and_difference_l212_212456


namespace maciek_total_purchase_cost_l212_212070

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l212_212070


namespace kali_height_now_l212_212879

variable (K_initial J_initial : ℝ)
variable (K_growth J_growth : ℝ)
variable (J_current : ℝ)

theorem kali_height_now :
  J_initial = K_initial →
  J_growth = (2 / 3) * 0.3 * K_initial →
  K_growth = 0.3 * K_initial →
  J_current = 65 →
  J_current = J_initial + J_growth →
  K_current = K_initial + K_growth →
  K_current = 70.42 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end kali_height_now_l212_212879


namespace D_72_eq_93_l212_212313

def D (n : ℕ) : ℕ :=
-- The function definition of D would go here, but we leave it abstract for now.
sorry

theorem D_72_eq_93 : D 72 = 93 :=
sorry

end D_72_eq_93_l212_212313


namespace cone_volume_with_same_radius_and_height_l212_212183

theorem cone_volume_with_same_radius_and_height (r h : ℝ) 
  (Vcylinder : ℝ) (Vcone : ℝ) (h1 : Vcylinder = 54 * Real.pi) 
  (h2 : Vcone = (1 / 3) * Vcylinder) : Vcone = 18 * Real.pi :=
by sorry

end cone_volume_with_same_radius_and_height_l212_212183


namespace combined_salaries_l212_212031

theorem combined_salaries (A B C D E : ℝ) 
  (hA : A = 9000) 
  (h_avg : (A + B + C + D + E) / 5 = 8200) :
  (B + C + D + E) = 32000 :=
by
  sorry

end combined_salaries_l212_212031


namespace max_knights_between_knights_l212_212237

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l212_212237


namespace pythagorean_diagonal_l212_212579

variable (m : ℕ) (h_m : m ≥ 3)

theorem pythagorean_diagonal (h : (2 * m)^2 + a^2 = (a + 2)^2) :
  (a + 2) = m^2 + 1 :=
by
  sorry

end pythagorean_diagonal_l212_212579


namespace distance_A_moves_l212_212213

-- Define the initial conditions and parameters of the rectangular sheet.
def width := 1  -- Width of the sheet in cm
def length := 12  -- Length of the sheet in cm
def mid_length := length / 2  -- Midpoint of the length in cm

-- Calculate the distance A moves using Pythagorean theorem.
def distance_move (a : ℝ) (b : ℝ) := Real.sqrt (a^2 + b^2)

theorem distance_A_moves :
  distance_move width mid_length = Real.sqrt 37 :=
sorry    -- Proof omitted

end distance_A_moves_l212_212213


namespace find_actual_number_of_children_l212_212204

theorem find_actual_number_of_children (B C : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 420)) : C = 840 := 
by
  sorry

end find_actual_number_of_children_l212_212204


namespace inequality_for_M_cap_N_l212_212150

def f (x : ℝ) := 2 * |x - 1| + x - 1
def g (x : ℝ) := 16 * x^2 - 8 * x + 1

def M := {x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3}
def N := {x : ℝ | -1 / 4 ≤ x ∧ x ≤ 3 / 4}
def M_cap_N := {x : ℝ | 0 ≤ x ∧ x ≤ 3 / 4}

theorem inequality_for_M_cap_N (x : ℝ) (hx : x ∈ M_cap_N) : x^2 * f x + x * (f x)^2 ≤ 1 / 4 := 
by 
  sorry

end inequality_for_M_cap_N_l212_212150


namespace max_knights_between_other_knights_l212_212218

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l212_212218


namespace solve_quadratic_l212_212170

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l212_212170


namespace complex_power_sum_l212_212002

open Complex

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨ z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
sorry

end complex_power_sum_l212_212002


namespace less_sum_mult_l212_212292

theorem less_sum_mult {a b : ℝ} (h1 : a < 1) (h2 : b > 1) : a * b < a + b :=
sorry

end less_sum_mult_l212_212292


namespace prove_angle_sum_l212_212522

open Real

theorem prove_angle_sum (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : cos α / sin β + cos β / sin α = 2) : 
  α + β = π / 2 := 
sorry

end prove_angle_sum_l212_212522


namespace correct_value_l212_212359

theorem correct_value (x : ℝ) (h : x / 3.6 = 2.5) : (x * 3.6) / 2 = 16.2 :=
by {
  -- Proof would go here
  sorry
}

end correct_value_l212_212359


namespace ice_cream_flavors_l212_212291

theorem ice_cream_flavors (scoops flavors : ℕ) (h_scoops : scoops = 5) (h_flavors : flavors = 3) : 
  Nat.choose (scoops + flavors - 1) (flavors - 1) = 21 :=
by
  rw [h_scoops, h_flavors]
  -- Here, the actual computation and proof would typically go, but for the purposes 
  -- of this task as per the provided instruction, we leave it to sorry.
  sorry

end ice_cream_flavors_l212_212291


namespace find_prob_ξ_gt_2_l212_212854

-- Given conditions
variables (σ : ℝ) (σ_pos : σ > 0)
def ξ : MeasureTheory.ProbMeasure ℝ := MeasureTheory.probMeasure_normal 3 (σ^2)
axiom P_ξ_gt_4 : MeasureTheory.probability_event (λ x : ℝ, x > 4) ξ = 1 / 5

-- Problem statement
theorem find_prob_ξ_gt_2 :
  MeasureTheory.probability_event (λ x : ℝ, x > 2) ξ = 4 / 5 :=
sorry

end find_prob_ξ_gt_2_l212_212854


namespace split_bill_equally_l212_212394

theorem split_bill_equally :
  let hamburger_cost := 3
  let hamburger_count := 5
  let fries_cost := 1.20
  let fries_count := 4
  let soda_cost := 0.50
  let soda_count := 5
  let spaghetti_cost := 2.70
  let friend_count := 5
  let total_cost := (hamburger_cost * hamburger_count) + (fries_cost * fries_count) + (soda_cost * soda_count) + spaghetti_cost
  in total_cost / friend_count = 5 := 
by
  sorry

end split_bill_equally_l212_212394


namespace initial_cards_collected_l212_212020

  -- Ralph collects some cards.
  variable (initial_cards: ℕ)

  -- Ralph's father gives Ralph 8 more cards.
  variable (added_cards: ℕ := 8)

  -- Now Ralph has 12 cards.
  variable (total_cards: ℕ := 12)

  -- Proof statement: Prove that the initial number of cards Ralph collected plus 8 equals 12.
  theorem initial_cards_collected: initial_cards + added_cards = total_cards := by
    sorry
  
end initial_cards_collected_l212_212020


namespace find_a_l212_212645

theorem find_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a < 13) (h3 : (51 ^ 2016 + a) % 13 = 0) : a = 12 :=
sorry

end find_a_l212_212645


namespace projection_v_w_l212_212813

noncomputable def vector_v : ℝ × ℝ := (3, 4)
noncomputable def vector_w : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := dot_product u v / dot_product v v
  (scalar * v.1, scalar * v.2)

theorem projection_v_w :
  proj vector_v vector_w = (4/5, -2/5) :=
sorry

end projection_v_w_l212_212813


namespace at_least_half_sectors_occupied_l212_212360

theorem at_least_half_sectors_occupied (n : ℕ) (chips : Finset (Fin n.succ)) 
(h_chips_count: chips.card = n + 1) :
  ∃ (steps : ℕ), ∀ (t : ℕ), t ≥ steps → (∃ sector_occupied : Finset (Fin n), sector_occupied.card ≥ n / 2) :=
sorry

end at_least_half_sectors_occupied_l212_212360


namespace each_friend_pays_l212_212393

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end each_friend_pays_l212_212393


namespace two_pow_gt_n_square_plus_one_l212_212438

theorem two_pow_gt_n_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := 
by {
  sorry
}

end two_pow_gt_n_square_plus_one_l212_212438


namespace linette_problem_proof_l212_212691

def boxes_with_neither_markers_nor_stickers (total_boxes markers stickers both : ℕ) : ℕ :=
  total_boxes - (markers + stickers - both)

theorem linette_problem_proof : 
  let total_boxes := 15
  let markers := 9
  let stickers := 5
  let both := 4
  boxes_with_neither_markers_nor_stickers total_boxes markers stickers both = 5 :=
by
  sorry

end linette_problem_proof_l212_212691


namespace interval_of_monotonic_decrease_range_of_k_l212_212955
open Real

noncomputable def f (x : ℝ) : ℝ := 
  let m := (sqrt 3 * sin (x / 4), 1)
  let n := (cos (x / 4), cos (x / 2))
  m.1 * n.1 + m.2 * n.2 -- vector dot product

-- Prove the interval of monotonic decrease for f(x)
theorem interval_of_monotonic_decrease (k : ℤ) : 
  4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3 → f x = sin (x / 2 + π / 6) + 1 / 2 :=
sorry

-- Prove the range of k such that the zero condition is satisfied for g(x) - k
theorem range_of_k (k : ℝ) :
  0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ [0, 7 * π / 3], (sin (x / 2 - π / 6) + 1 / 2) - k = 0 :=
sorry

end interval_of_monotonic_decrease_range_of_k_l212_212955


namespace sequence_bound_l212_212426

theorem sequence_bound (n : ℕ) (a : ℝ) (a_seq : ℕ → ℝ) 
  (h1 : a_seq 1 = a) 
  (h2 : a_seq n = a) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k < n - 1 → a_seq (k + 1) ≤ (a_seq k + a_seq (k + 2)) / 2) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → a_seq k ≤ a := 
by
  sorry

end sequence_bound_l212_212426


namespace Joseph_has_122_socks_l212_212558

def JosephSocks : Nat := 
  let red_pairs := 9 / 2
  let white_pairs := red_pairs + 2
  let green_pairs := 2 * red_pairs
  let blue_pairs := 3 * green_pairs
  let black_pairs := blue_pairs - 5
  (red_pairs + white_pairs + green_pairs + blue_pairs + black_pairs) * 2

theorem Joseph_has_122_socks : JosephSocks = 122 := 
  by
  sorry

end Joseph_has_122_socks_l212_212558


namespace volume_of_cube_is_correct_l212_212611

-- Define necessary constants and conditions
def cost_in_paise : ℕ := 34398
def rate_per_sq_cm : ℕ := 13
def surface_area : ℕ := cost_in_paise / rate_per_sq_cm
def face_area : ℕ := surface_area / 6
def side_length : ℕ := Nat.sqrt face_area
def volume : ℕ := side_length ^ 3

-- Prove the volume of the cube
theorem volume_of_cube_is_correct : volume = 9261 := by
  -- Using given conditions and basic arithmetic 
  sorry

end volume_of_cube_is_correct_l212_212611


namespace zero_not_in_range_of_g_l212_212980

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈(Real.cos x) / (x + 3)⌉
  else if x < -3 then ⌊(Real.cos x) / (x + 3)⌋
  else 0 -- arbitrary value since it's undefined

theorem zero_not_in_range_of_g :
  ¬ (∃ x : ℝ, g x = 0) :=
by
  intro h
  sorry

end zero_not_in_range_of_g_l212_212980


namespace count_multiples_13_9_200_500_l212_212829

def multiple_of_lcm (x y n : ℕ) : Prop :=
  n % (Nat.lcm x y) = 0

theorem count_multiples_13_9_200_500 :
  {n : ℕ | 200 ≤ n ∧ n ≤ 500 ∧ multiple_of_lcm 13 9 n}.toFinset.card = 3 :=
by
  sorry

end count_multiples_13_9_200_500_l212_212829


namespace perfect_square_n_l212_212781

theorem perfect_square_n (n : ℕ) (hn_pos : n > 0) :
  (∃ (m : ℕ), m * m = (n^2 + 11 * n - 4) * n.factorial + 33 * 13^n + 4) ↔ n = 1 ∨ n = 2 :=
by sorry

end perfect_square_n_l212_212781


namespace sahil_selling_price_l212_212876

-- Defining the conditions as variables
def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Defining the total cost
def total_cost : ℕ := purchase_price + repair_cost + transportation_charges

-- Calculating the profit amount
def profit : ℕ := (profit_percentage * total_cost) / 100

-- Calculating the selling price
def selling_price : ℕ := total_cost + profit

-- The Lean statement to prove the selling price is Rs 30,000
theorem sahil_selling_price : selling_price = 30000 :=
by 
  simp [total_cost, profit, selling_price]
  sorry

end sahil_selling_price_l212_212876


namespace extreme_points_inequality_l212_212404

noncomputable def f (a x : ℝ) : ℝ := a * Real.log (x + 1) + 1 / 2 * x ^ 2 - x

theorem extreme_points_inequality 
  (a : ℝ)
  (ha : 0 < a ∧ a < 1)
  (alpha beta : ℝ)
  (h_eq_alpha : alpha = -Real.sqrt (1 - a))
  (h_eq_beta : beta = Real.sqrt (1 - a))
  (h_order : alpha < beta) :
  (f a beta / alpha) < (1 / 2) :=
sorry

end extreme_points_inequality_l212_212404


namespace talia_mom_age_to_talia_age_ratio_l212_212671

-- Definitions for the problem
def Talia_current_age : ℕ := 13
def Talia_mom_current_age : ℕ := 39
def Talia_father_current_age : ℕ := 36

-- These definitions match the conditions in the math problem
def condition1 : Prop := Talia_current_age + 7 = 20
def condition2 : Prop := Talia_father_current_age + 3 = Talia_mom_current_age
def condition3 : Prop := Talia_father_current_age = 36

-- The ratio calculation
def ratio := Talia_mom_current_age / Talia_current_age

-- The main theorem to prove
theorem talia_mom_age_to_talia_age_ratio :
  condition1 ∧ condition2 ∧ condition3 → ratio = 3 := by
  sorry

end talia_mom_age_to_talia_age_ratio_l212_212671


namespace solution_set_of_abs_x_gt_1_l212_212180

theorem solution_set_of_abs_x_gt_1 (x : ℝ) : |x| > 1 ↔ x > 1 ∨ x < -1 := 
sorry

end solution_set_of_abs_x_gt_1_l212_212180


namespace max_knights_between_knights_l212_212225

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l212_212225


namespace solve_equation_l212_212513

theorem solve_equation (x : ℝ) : (2*x - 1)^2 = 81 ↔ (x = 5 ∨ x = -4) :=
by
  sorry

end solve_equation_l212_212513


namespace range_of_expression_l212_212329

theorem range_of_expression (a : ℝ) : (∃ a : ℝ, a + 1 ≥ 0 ∧ a - 2 ≠ 0) → (a ≥ -1 ∧ a ≠ 2) := 
by sorry

end range_of_expression_l212_212329


namespace star_contains_2011_l212_212975

theorem star_contains_2011 :
  ∃ (n : ℕ), n = 183 ∧ 
  (∃ (seq : List ℕ), seq = List.range' (2003) 11 ∧ 2011 ∈ seq) :=
by
  sorry

end star_contains_2011_l212_212975


namespace distance_from_town_l212_212759

theorem distance_from_town (d : ℝ) :
  (7 < d ∧ d < 8) ↔ (d < 8 ∧ d > 7 ∧ d > 6 ∧ d ≠ 9) :=
by sorry

end distance_from_town_l212_212759


namespace girls_first_half_l212_212598

theorem girls_first_half (total_students boys_first_half girls_first_half boys_second_half girls_second_half boys_whole_year : ℕ)
  (h1: total_students = 56)
  (h2: boys_first_half = 25)
  (h3: girls_first_half = 15)
  (h4: boys_second_half = 26)
  (h5: girls_second_half = 25)
  (h6: boys_whole_year = 23) : 
  ∃ girls_first_half_only : ℕ, girls_first_half_only = 3 :=
by {
  sorry
}

end girls_first_half_l212_212598


namespace rolls_sold_to_uncle_l212_212095

theorem rolls_sold_to_uncle (total_rolls : ℕ) (rolls_grandmother : ℕ) (rolls_neighbor : ℕ) (rolls_remaining : ℕ) (rolls_uncle : ℕ) :
  total_rolls = 12 →
  rolls_grandmother = 3 →
  rolls_neighbor = 3 →
  rolls_remaining = 2 →
  rolls_uncle = total_rolls - rolls_remaining - (rolls_grandmother + rolls_neighbor) →
  rolls_uncle = 4 :=
by
  intros h_total h_grandmother h_neighbor h_remaining h_compute
  rw [h_total, h_grandmother, h_neighbor, h_remaining] at h_compute
  exact h_compute

end rolls_sold_to_uncle_l212_212095


namespace find_t_value_l212_212184

theorem find_t_value (x y z t : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (hpos_x : 0 < x) (hpos_y : 0 < y) (hpos_z : 0 < z) :
  x + y + z + t = 10 → t = 4 :=
by
  -- Proof goes here
  sorry

end find_t_value_l212_212184


namespace interest_rate_A_l212_212616

-- Definitions for the conditions
def principal : ℝ := 1000
def rate_C : ℝ := 0.115
def time_period : ℝ := 3
def gain_B : ℝ := 45

-- Main theorem to prove
theorem interest_rate_A {R : ℝ} (h1 : gain_B = (principal * rate_C * time_period - principal * (R / 100) * time_period)) : R = 10 := 
by
  sorry

end interest_rate_A_l212_212616


namespace probability_of_one_unit_apart_l212_212446

theorem probability_of_one_unit_apart : 
  let points := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let total_pairs := (points.card.choose 2 : ℕ) in
  let favorable_pairs := 14 in
  (favorable_pairs : ℚ) / (total_pairs : ℚ) = 14 / 45 :=
by
  sorry

end probability_of_one_unit_apart_l212_212446


namespace price_of_shares_l212_212617

variable (share_value : ℝ) (dividend_rate : ℝ) (tax_rate : ℝ) (effective_return : ℝ) (price : ℝ)

-- Given conditions
axiom H1 : share_value = 50
axiom H2 : dividend_rate = 0.185
axiom H3 : tax_rate = 0.05
axiom H4 : effective_return = 0.25
axiom H5 : 0.25 * price = 0.185 * 50 - (0.05 * (0.185 * 50))

-- Prove that the price at which the investor bought the shares is Rs. 35.15
theorem price_of_shares : price = 35.15 :=
by
  sorry

end price_of_shares_l212_212617


namespace circle_proof_problem_l212_212730

variables {P Q R : Type}
variables {p q r dPQ dPR dQR : ℝ}

-- Given Conditions
variables (hpq : p > q) (hqr : q > r)
variables (hdPQ : ℝ) (hdPR : ℝ) (hdQR : ℝ)

-- Statement of the problem: prove that all conditions can be true
theorem circle_proof_problem :
  (∃ hpq' : dPQ = p + q, true) ∧
  (∃ hqr' : dQR = q + r, true) ∧
  (∃ hpr' : dPR > p + r, true) ∧
  (∃ hpq_diff : dPQ > p - q, true) →
  false := 
sorry

end circle_proof_problem_l212_212730


namespace seating_capacity_for_ten_tables_in_two_rows_l212_212619

-- Definitions based on the problem conditions
def seating_for_one_table : ℕ := 6

def seating_for_two_tables : ℕ := 10

def seating_for_three_tables : ℕ := 14

def additional_people_per_table : ℕ := 4

-- Calculating the seating capacity for n tables based on the pattern
def seating_capacity (n : ℕ) : ℕ :=
  if n = 1 then seating_for_one_table
  else seating_for_one_table + (n - 1) * additional_people_per_table

-- Proof statement without the proof
theorem seating_capacity_for_ten_tables_in_two_rows :
  (seating_capacity 5) * 2 = 44 :=
by sorry

end seating_capacity_for_ten_tables_in_two_rows_l212_212619


namespace balls_into_boxes_l212_212842

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end balls_into_boxes_l212_212842


namespace number_of_classes_l212_212489

theorem number_of_classes (max_val : ℕ) (min_val : ℕ) (class_interval : ℕ) (range : ℕ) (num_classes : ℕ) :
  max_val = 169 → min_val = 143 → class_interval = 3 → range = max_val - min_val → num_classes = (range + 2) / class_interval + 1 :=
sorry

end number_of_classes_l212_212489


namespace arithmetic_sequence_z_l212_212190

-- Define the arithmetic sequence and value of z
theorem arithmetic_sequence_z (z : ℤ) (arith_seq : 9 + 27 = 2 * z) : z = 18 := 
by 
  sorry

end arithmetic_sequence_z_l212_212190


namespace largest_n_for_factorable_poly_l212_212383

theorem largest_n_for_factorable_poly :
  ∃ n : ℤ, (∀ A B : ℤ, (3 * B + A = n) ∧ (A * B = 72) → (A = 1 ∧ B = 72 ∧ n = 217)) ∧
           (∀ A B : ℤ, A * B = 72 → 3 * B + A ≤ 217) :=
by
  sorry

end largest_n_for_factorable_poly_l212_212383


namespace min_value_fraction_l212_212110

theorem min_value_fraction (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + 2 * y = 3) : 
  (∃ t, t = (1 / (x - y) + 9 / (x + 5 * y)) ∧ t = 8 / 3) :=
by 
  sorry

end min_value_fraction_l212_212110


namespace distance_between_foci_of_ellipse_l212_212096

theorem distance_between_foci_of_ellipse : 
  let a := 5
  let b := 3
  2 * Real.sqrt (a^2 - b^2) = 8 := by
  let a := 5
  let b := 3
  sorry

end distance_between_foci_of_ellipse_l212_212096


namespace inequality_preservation_l212_212654

theorem inequality_preservation (a b : ℝ) (h : a > b) : (1/3 : ℝ) * a - 1 > (1/3 : ℝ) * b - 1 := 
by sorry

end inequality_preservation_l212_212654


namespace total_bugs_eaten_l212_212307

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l212_212307


namespace ratio_of_areas_l212_212594

noncomputable def side_length_C := 24 -- cm
noncomputable def side_length_D := 54 -- cm
noncomputable def ratio_areas := (side_length_C / side_length_D) ^ 2

theorem ratio_of_areas : ratio_areas = 16 / 81 := sorry

end ratio_of_areas_l212_212594


namespace parallel_lines_slope_eq_l212_212605

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end parallel_lines_slope_eq_l212_212605


namespace represent_same_function_l212_212343

noncomputable def f1 (x : ℝ) : ℝ := (x^3 + x) / (x^2 + 1)
def f2 (x : ℝ) : ℝ := x

theorem represent_same_function : ∀ x : ℝ, f1 x = f2 x := 
by
  sorry

end represent_same_function_l212_212343


namespace find_b_value_l212_212420

/-- Given a line segment from point (0, b) to (8, 0) with a slope of -3/2, 
    prove that the value of b is 12. -/
theorem find_b_value (b : ℝ) : (8 - 0) ≠ 0 ∧ ((0 - b) / (8 - 0) = -3/2) → b = 12 := 
by
  intro h
  sorry

end find_b_value_l212_212420


namespace no_roots_in_disk_l212_212117

noncomputable def homogeneous_polynomial_deg2 (a b c : ℝ) (x y : ℝ) := a * x^2 + b * x * y + c * y^2
noncomputable def homogeneous_polynomial_deg3 (q : ℝ → ℝ → ℝ) (x y : ℝ) := q x y

theorem no_roots_in_disk 
  (a b c : ℝ) (h_poly_deg2 : ∀ x y, homogeneous_polynomial_deg2 a b c x y = a * x^2 + b * x * y + c * y^2)
  (q : ℝ → ℝ → ℝ) (h_poly_deg3 : ∀ x y, homogeneous_polynomial_deg3 q x y = q x y)
  (h_cond : b^2 < 4 * a * c) :
  ∃ k > 0, ∀ x y, x^2 + y^2 < k → homogeneous_polynomial_deg2 a b c x y ≠ homogeneous_polynomial_deg3 q x y ∨ (x = 0 ∧ y = 0) :=
sorry

end no_roots_in_disk_l212_212117


namespace only_solution_xyz_l212_212779

theorem only_solution_xyz : 
  ∀ (x y z : ℕ), x^3 + 4 * y^3 = 16 * z^3 + 4 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z
  intro h
  sorry

end only_solution_xyz_l212_212779


namespace condition_necessary_but_not_sufficient_l212_212939

variable (m : ℝ)

/-- The problem statement and proof condition -/
theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, |x - 2| + |x + 2| > m) → (∀ x : ℝ, x^2 + m * x + 4 > 0) :=
by {
  sorry
}

end condition_necessary_but_not_sufficient_l212_212939


namespace rectangle_midpoints_sum_l212_212212

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum_l212_212212


namespace algebraic_expression_value_l212_212943

-- Definitions based on the conditions
variable {a : ℝ}
axiom root_equation : 2 * a^2 + 3 * a - 4 = 0

-- Definition of the problem: Proving that 2a^2 + 3a equals 4.
theorem algebraic_expression_value : 2 * a^2 + 3 * a = 4 :=
by 
  have h : 2 * a^2 + 3 * a - 4 = 0 := root_equation
  have h' : 2 * a^2 + 3 * a = 4 := by sorry
  exact h'

end algebraic_expression_value_l212_212943


namespace remainder_correct_l212_212092

noncomputable def P : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^6 
                                  + Polynomial.C 2 * Polynomial.X^5 
                                  - Polynomial.C 3 * Polynomial.X^4 
                                  + Polynomial.C 1 * Polynomial.X^3 
                                  - Polynomial.C 2 * Polynomial.X^2
                                  + Polynomial.C 5 * Polynomial.X 
                                  - Polynomial.C 1

noncomputable def D : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) * 
                                      (Polynomial.X + Polynomial.C 2) * 
                                      (Polynomial.X - Polynomial.C 3)

noncomputable def R : Polynomial ℝ := 17 * Polynomial.X^2 - 52 * Polynomial.X + 38

theorem remainder_correct :
    ∀ (q : Polynomial ℝ), P = D * q + R :=
by sorry

end remainder_correct_l212_212092


namespace intersections_form_trapezoid_l212_212503

noncomputable def pointE : ℝ × ℝ := (0, 0)
noncomputable def pointF : ℝ × ℝ := (0, 5)
noncomputable def pointG : ℝ × ℝ := (8, 5)
noncomputable def pointH : ℝ × ℝ := (8, 0)

-- Lines equations:
noncomputable def line_E_45 (x : ℝ) : ℝ := x
noncomputable def line_E_75 (x : ℝ) : ℝ := 3.732 * x
noncomputable def line_F_neg45 (x : ℝ) : ℝ := 5 - x
noncomputable def line_F_neg75 (x : ℝ) : ℝ := 5 - 3.732 * x

-- Intersection points:
noncomputable def intersection_45_neg45 : ℝ × ℝ := (2.5, 2.5)
noncomputable def intersection_75_neg75 : ℝ × ℝ := (0.67, 2.5)

theorem intersections_form_trapezoid :
  let I1 := intersection_45_neg45
  let I2 := intersection_75_neg75
  let shape_vertices := {pointE, pointF, I1, I2}
  shape vertices = Trapezoid := sorry

end intersections_form_trapezoid_l212_212503


namespace min_value_2x_plus_y_l212_212546

theorem min_value_2x_plus_y (x y : ℝ) (h1 : |y| ≤ 2 - x) (h2 : x ≥ -1) : 
  ∃ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 ∧ (∀ y : ℝ, |y| ≤ 2 - x → x ≥ -1 → 2 * x + y ≥ -5) ∧ (2 * x + y = -5) :=
by
  sorry

end min_value_2x_plus_y_l212_212546


namespace find_h_from_quadratic_l212_212886

theorem find_h_from_quadratic (
  p q r : ℝ) (h₁ : ∀ x, p * x^2 + q * x + r = 7 * (x - 5)^2 + 14) :
  ∀ m k h, (∀ x, 5 * p * x^2 + 5 * q * x + 5 * r = m * (x - h)^2 + k) → h = 5 :=
by
  intros m k h h₂
  sorry

end find_h_from_quadratic_l212_212886


namespace interest_rate_is_10_perc_l212_212852

noncomputable def interest_rate (P : ℝ) (R : ℝ) (T : ℝ := 2) : ℝ := (P * R * T) / 100

theorem interest_rate_is_10_perc (P : ℝ) : 
  (interest_rate P 10) = P / 5 :=
by
  sorry

end interest_rate_is_10_perc_l212_212852


namespace marble_ratio_l212_212623

theorem marble_ratio 
  (K A M : ℕ) 
  (M_has_5_times_as_many_as_K : M = 5 * K)
  (M_has_85_marbles : M = 85)
  (M_has_63_more_than_A : M = A + 63)
  (A_needs_12_more : A + 12 = 34) :
  34 / 17 = 2 := 
by 
  sorry

end marble_ratio_l212_212623


namespace compute_custom_op_l212_212125

def custom_op (x y : ℤ) : ℤ := 
  x * y - y * x - 3 * x + 2 * y

theorem compute_custom_op : (custom_op 9 5) - (custom_op 5 9) = -20 := 
by
  sorry

end compute_custom_op_l212_212125


namespace problem_statement_l212_212388

noncomputable def monochromatic_triangle_in_K6 : Prop :=
  let vertices := {0, 1, 2, 3, 4, 5} in
  ∃ (edges : vertices × vertices → Bool), 
    (∀ e, e ∈ vertices × vertices → Bool = red ∨ Bool = blue) ∧
    (∀ (A B C : vertices), A ≠ B ∧ B ≠ C ∧ C ≠ A →
        ((edges (A, B) = red ∧ edges (B, C) = red ∧ edges (C, A) = red) ∨
         (edges (A, B) = blue ∧ edges (B, C) = blue ∧ edges (C, A) = blue)))

theorem problem_statement :
  (probability monochromatic_triangle_in_K6 vertices × vertices → Bool = 1/2) = 255 / 256 := sorry

end problem_statement_l212_212388


namespace symmetry_axis_of_sine_function_l212_212926

theorem symmetry_axis_of_sine_function (x : ℝ) :
  (∃ k : ℤ, 2 * x + π / 4 = k * π + π / 2) ↔ x = π / 8 :=
by sorry

end symmetry_axis_of_sine_function_l212_212926


namespace weights_equal_weights_equal_ints_weights_equal_rationals_l212_212725

theorem weights_equal (w : Fin 13 → ℝ) (swap_n_weighs_balance : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℝ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_ints (w : Fin 13 → ℤ) (swap_n_weighs_balance_ints : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℤ), ∀ (i : Fin 13), w i = m :=
by
  sorry

theorem weights_equal_rationals (w : Fin 13 → ℚ) (swap_n_weighs_balance_rationals : ∀ (s : Finset (Fin 13)), s.card = 12 → 
  ∃ (t u : Finset (Fin 13)), t.card = 6 ∧ u.card = 6 ∧ t ∪ u = s ∧ t ∩ u = ∅ ∧ Finset.sum t w = Finset.sum u w) :
  ∃ (m : ℚ), ∀ (i : Fin 13), w i = m :=
by
  sorry

end weights_equal_weights_equal_ints_weights_equal_rationals_l212_212725


namespace inequality_proof_l212_212162

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^3 + b^3 = 2) :
  (1 / a) + (1 / b) ≥ 2 * (a^2 - a + 1) * (b^2 - b + 1) := 
by
  sorry

end inequality_proof_l212_212162


namespace find_x_minus_y_l212_212724

theorem find_x_minus_y (x y z : ℤ) (h₁ : x - y - z = 7) (h₂ : x - y + z = 15) : x - y = 11 := by
  sorry

end find_x_minus_y_l212_212724


namespace perimeter_after_adding_tiles_l212_212777

theorem perimeter_after_adding_tiles (init_perimeter new_tiles : ℕ) (cond1 : init_perimeter = 14) (cond2 : new_tiles = 2) :
  ∃ new_perimeter : ℕ, new_perimeter = 18 :=
by
  sorry

end perimeter_after_adding_tiles_l212_212777


namespace real_solution_unique_l212_212805

theorem real_solution_unique (x : ℝ) : 
  (x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) ↔ x = 6 := 
begin
  sorry
end

end real_solution_unique_l212_212805


namespace wizard_viable_combinations_l212_212493

def wizard_combination_problem : Prop :=
  let total_combinations := 4 * 6
  let incompatible_combinations := 3
  let viable_combinations := total_combinations - incompatible_combinations
  viable_combinations = 21

theorem wizard_viable_combinations : wizard_combination_problem :=
by
  sorry

end wizard_viable_combinations_l212_212493


namespace smallest_integer_modulus_l212_212505

theorem smallest_integer_modulus :
  ∃ n : ℕ, 0 < n ∧ (7 ^ n ≡ n ^ 4 [MOD 3]) ∧
  ∀ m : ℕ, 0 < m ∧ (7 ^ m ≡ m ^ 4 [MOD 3]) → n ≤ m :=
by
  sorry

end smallest_integer_modulus_l212_212505


namespace Rachel_budget_twice_Sara_l212_212699

-- Define the cost of Sara's shoes and dress
def s_shoes : ℕ := 50
def s_dress : ℕ := 200

-- Define the target budget for Rachel
def r : ℕ := 500

-- State the theorem to prove
theorem Rachel_budget_twice_Sara :
  2 * s_shoes + 2 * s_dress = r :=
by simp [s_shoes, s_dress, r]; sorry

end Rachel_budget_twice_Sara_l212_212699


namespace similar_pentagon_area_l212_212432

theorem similar_pentagon_area
  (K1 K2 : ℝ) (L1 L2 : ℝ)
  (h_similar : true)  -- simplifying the similarity condition as true for the purpose of this example
  (h_K1 : K1 = 18)
  (h_K2 : K2 = 24)
  (h_L1 : L1 = 8.4375) :
  L2 = 15 :=
by
  sorry

end similar_pentagon_area_l212_212432


namespace find_nat_int_l212_212641

theorem find_nat_int (x y : ℕ) (h : x^2 = y^2 + 7 * y + 6) : x = 6 ∧ y = 3 := 
by
  sorry

end find_nat_int_l212_212641


namespace alpha_plus_2beta_eq_45_l212_212694

theorem alpha_plus_2beta_eq_45 
  (α β : ℝ) 
  (hα_pos : 0 < α ∧ α < π / 2) 
  (hβ_pos : 0 < β ∧ β < π / 2) 
  (tan_alpha : Real.tan α = 1 / 7) 
  (sin_beta : Real.sin β = 1 / Real.sqrt 10)
  : α + 2 * β = π / 4 :=
sorry

end alpha_plus_2beta_eq_45_l212_212694


namespace cards_distribution_l212_212657

theorem cards_distribution (total_cards people : ℕ) (h1 : total_cards = 48) (h2 : people = 7) :
  (people - (total_cards % people)) = 1 :=
by
  sorry

end cards_distribution_l212_212657


namespace max_range_walk_min_range_walk_count_max_range_sequences_l212_212053

variable {a b : ℕ}

-- Condition: a > b
def valid_walk (a b : ℕ) : Prop := a > b

-- Proof that the maximum possible range of the walk is a
theorem max_range_walk (h : valid_walk a b) : 
  (a + b) = a + b := sorry

-- Proof that the minimum possible range of the walk is a - b
theorem min_range_walk (h : valid_walk a b) : 
  (a - b) = a - b := sorry

-- Proof that the number of different sequences with the maximum possible range is b + 1
theorem count_max_range_sequences (h : valid_walk a b) : 
  b + 1 = b + 1 := sorry

end max_range_walk_min_range_walk_count_max_range_sequences_l212_212053


namespace polynomial_unique_l212_212687

noncomputable def lagrange_interpolation {R : Type*} [Field R] (a : Fin n → R) (b : Fin n → R) : Polynomial R :=
  Polynomial.divX (nconn.toPoly ⟨a, sorry⟩ (⟨b, sorry⟩))

theorem polynomial_unique
  {R : Type*} [Field R]
  (a : Fin n → R) (ha : Function.Injective a) (b : Fin n → R) :
  ∃ (P : Polynomial R), 
    (∀ i, P.eval (a i) = b i) ∧
    (∀ P', (∀ i, P'.eval (a i) = b i) → P' = polynomial.prodX a * Q + lagrange_interpolation a b) :=
begin
  sorry
end

end polynomial_unique_l212_212687


namespace geom_seq_prop_l212_212549

-- Definitions from the conditions
def geom_seq (a : ℕ → ℝ) := ∀ (n : ℕ), (a (n + 1)) / (a n) = (a 1) / (a 0) ∧ a n > 0

def condition (a : ℕ → ℝ) :=
  (1 / (a 2 * a 4)) + (2 / (a 4 ^ 2)) + (1 / (a 4 * a 6)) = 81

-- The statement to prove
theorem geom_seq_prop (a : ℕ → ℝ) (hgeom : geom_seq a) (hcond : condition a) :
  (1 / (a 3) + 1 / (a 5)) = 9 :=
sorry

end geom_seq_prop_l212_212549


namespace jake_later_than_austin_by_20_seconds_l212_212498

theorem jake_later_than_austin_by_20_seconds :
  (9 * 30) / 3 - 60 = 20 :=
by
  sorry

end jake_later_than_austin_by_20_seconds_l212_212498


namespace count_decorations_l212_212628

/--
Define a function T(n) that determines the number of ways to decorate the window 
with n stripes according to the given conditions.
--/
def T : ℕ → ℕ
| 0       => 1 -- optional case for completeness
| 1       => 2
| 2       => 2
| (n + 1) => T n + T (n - 1)

theorem count_decorations : T 10 = 110 := by
  sorry

end count_decorations_l212_212628


namespace hyperbola_condition_l212_212028

theorem hyperbola_condition (m : ℝ) : 
  (∃ a b : ℝ, a = m + 4 ∧ b = m - 3 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)) ↔ m > 3 :=
sorry

end hyperbola_condition_l212_212028


namespace ratio_income_to_expenditure_l212_212713

theorem ratio_income_to_expenditure (I E S : ℕ) 
  (h1 : I = 10000) 
  (h2 : S = 3000) 
  (h3 : S = I - E) : I / Nat.gcd I E = 10 ∧ E / Nat.gcd I E = 7 := by 
  sorry

end ratio_income_to_expenditure_l212_212713


namespace painter_total_cost_l212_212075

def south_seq (n : Nat) : Nat :=
  4 + 6 * (n - 1)

def north_seq (n : Nat) : Nat :=
  5 + 6 * (n - 1)

noncomputable def digit_cost (n : Nat) : Nat :=
  String.length (toString n)

noncomputable def total_cost : Nat :=
  let south_cost := (List.range 25).map south_seq |>.map digit_cost |>.sum
  let north_cost := (List.range 25).map north_seq |>.map digit_cost |>.sum
  south_cost + north_cost

theorem painter_total_cost : total_cost = 116 := by
  sorry

end painter_total_cost_l212_212075


namespace probability_at_least_one_head_l212_212984

theorem probability_at_least_one_head : 
  (1 - (1 / 2) * (1 / 2) * (1 / 2) = 7 / 8) :=
by
  sorry

end probability_at_least_one_head_l212_212984


namespace order_of_three_numbers_l212_212716

noncomputable def log_base_2 : ℝ → ℝ := λ x, Real.log x / Real.log 2

theorem order_of_three_numbers : 0 < (0.3 : ℝ)^2 ∧ (0.3 : ℝ)^2 < 1 ∧ log_base_2 0.3 < 0 ∧ 2^(0.3 : ℝ) > 1 →
  log_base_2 0.3 < (0.3 : ℝ)^2 ∧ (0.3 : ℝ)^2 < 2^(0.3 : ℝ) :=
by sorry

end order_of_three_numbers_l212_212716


namespace union_eq_l212_212953

-- Define the sets M and N
def M : Finset ℕ := {0, 3}
def N : Finset ℕ := {1, 2, 3}

-- Define the proof statement
theorem union_eq : M ∪ N = {0, 1, 2, 3} := 
by
  sorry

end union_eq_l212_212953


namespace no_consecutive_nat_mul_eq_25k_plus_1_l212_212695

theorem no_consecutive_nat_mul_eq_25k_plus_1 (k : ℕ) : 
  ¬ ∃ n : ℕ, n * (n + 1) = 25 * k + 1 :=
sorry

end no_consecutive_nat_mul_eq_25k_plus_1_l212_212695


namespace distribute_balls_l212_212833

theorem distribute_balls : 
  ∀ (balls boxes: ℕ), 
  balls = 5 → 
  boxes = 4 → 
  (∑ n in (finset.range (balls + 1)).powerset, if n.sum = balls then (n.card!) else 0) = 56 :=
by {
  intros balls boxes h_balls h_boxes,
  sorry
}

end distribute_balls_l212_212833


namespace probability_two_same_color_balls_l212_212636

open Rat

theorem probability_two_same_color_balls:
  let total_balls := 16,
      prob_blue := (8 / total_balls) * (8 / total_balls),
      prob_green := (5 / total_balls) * (5 / total_balls),
      prob_red := (3 / total_balls) * (3 / total_balls),
      total_prob := prob_blue + prob_green + prob_red in
  total_prob = (49 / 128) := by
  sorry

end probability_two_same_color_balls_l212_212636


namespace find_real_solutions_l212_212796

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l212_212796


namespace range_of_a_l212_212688

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * x - a = 0) ↔ a ≥ -1 :=
by
  sorry

end range_of_a_l212_212688


namespace angles_in_first_or_third_quadrant_l212_212700

noncomputable def angles_first_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}

noncomputable def angles_third_quadrant_set : Set ℝ :=
  {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}

theorem angles_in_first_or_third_quadrant :
  ∃ S1 S2 : Set ℝ, 
    (S1 = {α | ∃ k : ℤ, (2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + (Real.pi / 2))}) ∧
    (S2 = {α | ∃ k : ℤ, (2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + (3 * Real.pi / 2))}) ∧
    (angles_first_quadrant_set = S1 ∧ angles_third_quadrant_set = S2)
  :=
sorry

end angles_in_first_or_third_quadrant_l212_212700


namespace distribute_balls_l212_212832

theorem distribute_balls : 
  ∀ (balls boxes: ℕ), 
  balls = 5 → 
  boxes = 4 → 
  (∑ n in (finset.range (balls + 1)).powerset, if n.sum = balls then (n.card!) else 0) = 56 :=
by {
  intros balls boxes h_balls h_boxes,
  sorry
}

end distribute_balls_l212_212832


namespace distance_between_house_and_school_l212_212490

variable (T D : ℝ)

axiom cond1 : 9 * (T + 20 / 60) = D
axiom cond2 : 12 * (T - 20 / 60) = D
axiom cond3 : 15 * (T - 40 / 60) = D

theorem distance_between_house_and_school : D = 24 := 
by
  sorry

end distance_between_house_and_school_l212_212490


namespace decimal_fraction_to_percentage_l212_212208

theorem decimal_fraction_to_percentage (d : ℝ) (h : d = 0.03) : d * 100 = 3 := by
  sorry

end decimal_fraction_to_percentage_l212_212208


namespace min_pq_sq_min_value_l212_212741

noncomputable def min_pq_sq (α : ℝ) : ℝ :=
  let p := α - 2
  let q := -(α + 1)
  (p + q)^2 - 2 * (p * q)

theorem min_pq_sq_min_value : 
  (∃ (α : ℝ), ∀ p q : ℝ, 
    p^2 + q^2 = (p + q)^2 - 2 * p * q ∧ 
    (p + q = α - 2 ∧ p * q = -(α + 1))) → 
  (min_pq_sq 1) = 5 :=
by
  sorry

end min_pq_sq_min_value_l212_212741


namespace total_points_l212_212924

noncomputable def Darius_points : ℕ := 10
noncomputable def Marius_points : ℕ := Darius_points + 3
noncomputable def Matt_points : ℕ := Darius_points + 5
noncomputable def Sofia_points : ℕ := 2 * Matt_points

theorem total_points : Darius_points + Marius_points + Matt_points + Sofia_points = 68 :=
by
  -- Definitions are directly from the problem statement, proof skipped 
  sorry

end total_points_l212_212924


namespace total_digits_first_2500_even_integers_l212_212195

theorem total_digits_first_2500_even_integers :
  let even_nums := List.range' 2 5000 (λ n, 2*n)  -- List of the first 2500 even integers
  let one_digit_nums := even_nums.filter (λ n, n < 10)
  let two_digit_nums := even_nums.filter (λ n, 10 ≤ n ∧ n < 100)
  let three_digit_nums := even_nums.filter (λ n, 100 ≤ n ∧ n < 1000)
  let four_digit_nums := even_nums.filter (λ n, 1000 ≤ n ∧ n ≤ 5000)
  let sum_digits := one_digit_nums.length * 1 + two_digit_nums.length * 2 + three_digit_nums.length * 3 + four_digit_nums.length * 4
  in sum_digits = 9448 := by sorry

end total_digits_first_2500_even_integers_l212_212195


namespace balls_into_boxes_l212_212841

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end balls_into_boxes_l212_212841


namespace inequality_proof_l212_212277

variable (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)

theorem inequality_proof :
  (a^2 / (c * (b + c)) + b^2 / (a * (c + a)) + c^2 / (b * (a + b))) >= 3 / 2 :=
by
  sorry

end inequality_proof_l212_212277


namespace brick_width_l212_212358

/-- Let dimensions of the wall be 700 cm (length), 600 cm (height), and 22.5 cm (thickness).
    Let dimensions of each brick be 25 cm (length), W cm (width), and 6 cm (height).
    Given that 5600 bricks are required to build the wall, prove that the width of each brick is 11.25 cm. -/
theorem brick_width (W : ℝ)
  (h_wall_dimensions : 700 = 700) (h_wall_height : 600 = 600) (h_wall_thickness : 22.5 = 22.5)
  (h_brick_length : 25 = 25) (h_brick_height : 6 = 6) (h_num_bricks : 5600 = 5600)
  (h_wall_volume : 700 * 600 * 22.5 = 9450000)
  (h_brick_volume : 25 * W * 6 = 9450000 / 5600) :
  W = 11.25 :=
sorry

end brick_width_l212_212358


namespace baseball_card_devaluation_l212_212356

variable (x : ℝ) -- Note: x will represent the yearly percent decrease in decimal form (e.g., x = 0.10 for 10%)

theorem baseball_card_devaluation :
  (1 - x) * (1 - x) = 0.81 → x = 0.10 :=
by
  sorry

end baseball_card_devaluation_l212_212356


namespace inequality_sqrt_sum_l212_212157

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l212_212157


namespace unique_positive_integer_solution_l212_212780

theorem unique_positive_integer_solution :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, n^4 - n^3 + 3*n^2 + 5 = k^2 :=
by
  sorry

end unique_positive_integer_solution_l212_212780


namespace ball_distribution_l212_212837

theorem ball_distribution : 
  (finset.sum 
    (finset.image (λ (p : sym2 (fin 4)), 
                    match p with
                    | (a, b, c, d) => 
                      if a + b + c + d = 5 then 1 else 0
                    end) 
    (sym2 (fin 5))).card).to_nat = 56 :=
sorry

end ball_distribution_l212_212837


namespace find_d_l212_212062

namespace NineDigitNumber

variables {A B C D E F G : ℕ}

theorem find_d 
  (h1 : 6 + A + B = 13) 
  (h2 : A + B + C = 13)
  (h3 : B + C + D = 13)
  (h4 : C + D + E = 13)
  (h5 : D + E + F = 13)
  (h6 : E + F + G = 13)
  (h7 : F + G + 3 = 13) :
  D = 4 :=
sorry

end NineDigitNumber

end find_d_l212_212062


namespace horner_v3_at_2_l212_212819

-- Defining the polynomial f(x).
def f (x : ℝ) := 2 * x^5 + 3 * x^3 - 2 * x^2 + x - 1

-- Defining the Horner's method evaluation up to v3 at x = 2.
def horner_eval (x : ℝ) := (((2 * x + 0) * x + 3) * x - 2) * x + 1

-- The proof statement we need to show.
theorem horner_v3_at_2 : horner_eval 2 = 20 := sorry

end horner_v3_at_2_l212_212819


namespace fraction_subtraction_l212_212042

theorem fraction_subtraction :
  ((2 + 4 + 6 : ℚ) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12) :=
by
  sorry

end fraction_subtraction_l212_212042


namespace combination_identity_l212_212056

theorem combination_identity : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose 6 4) := 
by 
  sorry

end combination_identity_l212_212056


namespace find_a10_l212_212283

noncomputable def ladder_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), (a (n + 3))^2 = a n * a (n + 6)

theorem find_a10 {a : ℕ → ℝ} (h1 : ladder_geometric_sequence a) 
(h2 : a 1 = 1) 
(h3 : a 4 = 2) : a 10 = 8 :=
sorry

end find_a10_l212_212283


namespace maciek_total_cost_l212_212066

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l212_212066


namespace reciprocal_solution_l212_212542

theorem reciprocal_solution {x : ℝ} (h : x * -9 = 1) : x = -1/9 :=
sorry

end reciprocal_solution_l212_212542


namespace distance_missouri_to_new_york_by_car_l212_212450

variable (d_flight d_car : ℚ)

theorem distance_missouri_to_new_york_by_car :
  d_car = 1.4 * d_flight → 
  d_car = 1400 → 
  (d_car / 2 = 700) :=
by
  intros h1 h2
  sorry

end distance_missouri_to_new_york_by_car_l212_212450


namespace sqrt_inequality_l212_212159

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
sorry

end sqrt_inequality_l212_212159


namespace carp_and_population_l212_212822

-- Define the characteristics of an individual and a population
structure Individual where
  birth : Prop
  death : Prop
  gender : Prop
  age : Prop

structure Population where
  birth_rate : Prop
  death_rate : Prop
  gender_ratio : Prop
  age_composition : Prop

-- Define the conditions as hypotheses
axiom a : Individual
axiom b : Population

-- State the theorem: If "a" has characteristics of an individual and "b" has characteristics
-- of a population, then "a" is a carp and "b" is a carp population
theorem carp_and_population : 
  (a.birth ∧ a.death ∧ a.gender ∧ a.age) ∧
  (b.birth_rate ∧ b.death_rate ∧ b.gender_ratio ∧ b.age_composition) →
  (a = ⟨True, True, True, True⟩ ∧ b = ⟨True, True, True, True⟩) := 
by 
  sorry

end carp_and_population_l212_212822


namespace knights_max_seated_between_knights_l212_212229

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l212_212229


namespace cone_base_radius_l212_212402

/--
Given a cone with the following properties:
1. The surface area of the cone is \(3\pi\).
2. The lateral surface of the cone unfolds into a semicircle (which implies the slant height is twice the radius of the base).
Prove that the radius of the base of the cone is \(1\).
-/
theorem cone_base_radius 
  (S : ℝ)
  (r l : ℝ)
  (h1 : S = 3 * Real.pi)
  (h2 : l = 2 * r)
  : r = 1 := 
  sorry

end cone_base_radius_l212_212402


namespace scientific_notation_of_170000_l212_212447

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l212_212447


namespace lunch_cost_before_tip_l212_212461

theorem lunch_cost_before_tip (tip_rate : ℝ) (total_spent : ℝ) (C : ℝ) : 
  tip_rate = 0.20 ∧ total_spent = 72.96 ∧ C + tip_rate * C = total_spent → C = 60.80 :=
by
  intro h
  sorry

end lunch_cost_before_tip_l212_212461


namespace gcd_polynomial_l212_212944

open Nat

theorem gcd_polynomial (b : ℤ) (hb : 1632 ∣ b) : gcd (b^2 + 11 * b + 30) (b + 6) = 6 := by
  sorry

end gcd_polynomial_l212_212944


namespace find_b_and_c_find_b_with_c_range_of_b_l212_212105

-- Part (Ⅰ)
theorem find_b_and_c (b c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_zeros : f (-1) = 0 ∧ f 1 = 0) : b = 0 ∧ c = -1 := sorry

-- Part (Ⅱ)
theorem find_b_with_c (b : ℝ) (f : ℝ → ℝ)
  (x1 x2 : ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + (b^2 + 2 * b + 3))
  (h_eq : (x1 + 1) * (x2 + 1) = 8) 
  (h_roots : f x1 = 0 ∧ f x2 = 0) : b = -2 := sorry

-- Part (Ⅲ)
theorem range_of_b (b : ℝ) (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f_def : ∀ x, f x = x^2 + 2 * b * x + (-1 - 2 * b))
  (h_f_1 : f 1 = 0)
  (h_g_def : ∀ x, g x = f x + x + b)
  (h_intervals : ∀ x, 
    ((-3 < x) ∧ (x < -2) → g x > 0) ∧
    ((-2 < x) ∧ (x < 0) → g x < 0) ∧
    ((0 < x) ∧ (x < 1) → g x < 0) ∧
    ((1 < x) → g x > 0)) : (1/5) < b ∧ b < (5/7) := sorry

end find_b_and_c_find_b_with_c_range_of_b_l212_212105


namespace find_m_l212_212409

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end find_m_l212_212409


namespace linear_dependency_k_l212_212178

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end linear_dependency_k_l212_212178


namespace find_larger_number_l212_212154

-- Define the problem conditions
variable (x y : ℕ)
hypothesis h1 : y = x + 10
hypothesis h2 : x + y = 34

-- Formalize the goal to prove
theorem find_larger_number : y = 22 := by
  -- placeholders in lean statement to skip the proof
  sorry

end find_larger_number_l212_212154


namespace cone_diameter_base_l212_212284

theorem cone_diameter_base 
  (r l : ℝ) 
  (h_semicircle : l = 2 * r) 
  (h_surface_area : π * r ^ 2 + π * r * l = 3 * π) 
  : 2 * r = 2 :=
by
  sorry

end cone_diameter_base_l212_212284


namespace value_of_x_l212_212418

theorem value_of_x (a x y : ℝ) 
  (h1 : a^(x - y) = 343) 
  (h2 : a^(x + y) = 16807) : x = 4 :=
by
  sorry

end value_of_x_l212_212418


namespace toys_total_is_240_l212_212007

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l212_212007


namespace workers_together_time_l212_212473

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 10) :
  let rateA := 1 / A_time
  let rateB := 1 / B_time
  let combined_rate := rateA + rateB
  combined_rate * (40 / 9) = 1 :=
by 
  sorry

end workers_together_time_l212_212473


namespace smallest_other_divisor_of_40_l212_212259

theorem smallest_other_divisor_of_40 (n : ℕ) (h₁ : n > 1) (h₂ : 40 % n = 0) (h₃ : n ≠ 8) :
  (∀ m : ℕ, m > 1 → 40 % m = 0 → m ≠ 8 → n ≤ m) → n = 5 :=
by 
  sorry

end smallest_other_divisor_of_40_l212_212259


namespace calculate_expression_l212_212631

theorem calculate_expression :
  (16/81: ℝ) ^ ((-3: ℝ) / 4) + real.log 3 (5/4) + real.log 3 (4/5) = 27 / 8 :=
by
  sorry

end calculate_expression_l212_212631


namespace inverse_proportion_quadrants_l212_212585

theorem inverse_proportion_quadrants (k : ℝ) (hk : k ≠ 0) :
  (∃ x y : ℝ, x = -2 ∧ y = 3 ∧ y = k / x) →
  (∀ x : ℝ, (x < 0 → k / x > 0) ∧ (x > 0 → k / x < 0)) :=
sorry

end inverse_proportion_quadrants_l212_212585


namespace sequence_general_term_l212_212860

theorem sequence_general_term :
  ∀ (a : ℕ → ℝ), (a 1 = 1) →
    (∀ n : ℕ, n > 0 → (Real.sqrt (a n) - Real.sqrt (a (n + 1)) = Real.sqrt (a n * a (n + 1)))) →
    (∀ n : ℕ, n > 0 → a n = 1 / (n ^ 2)) :=
by
  intros a ha1 hrec n hn
  sorry

end sequence_general_term_l212_212860


namespace max_knights_seated_next_to_two_knights_l212_212243

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l212_212243


namespace minimum_unused_area_for_given_shapes_l212_212483

def remaining_area (side_length : ℕ) (total_area used_area : ℕ) : ℕ :=
  total_area - used_area

theorem minimum_unused_area_for_given_shapes : (remaining_area 5 (5 * 5) (2 * 2 + 1 * 3 + 2 * 1) = 16) :=
by
  -- We skip the proof here, as instructed.
  sorry

end minimum_unused_area_for_given_shapes_l212_212483


namespace jon_weekly_speed_gain_l212_212977

-- Definitions based on the conditions
def initial_speed : ℝ := 80
def speed_increase_percentage : ℝ := 0.20
def training_sessions : ℕ := 4
def weeks_per_session : ℕ := 4
def total_training_duration : ℕ := training_sessions * weeks_per_session

-- The calculated final speed
def final_speed : ℝ := initial_speed + initial_speed * speed_increase_percentage

theorem jon_weekly_speed_gain : 
  (final_speed - initial_speed) / total_training_duration = 1 :=
by
  -- This is the statement we want to prove
  sorry

end jon_weekly_speed_gain_l212_212977


namespace fraction_simplification_l212_212248

theorem fraction_simplification :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) /
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by
  sorry

end fraction_simplification_l212_212248


namespace equal_area_division_l212_212260

theorem equal_area_division (d : ℝ) : 
  (∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 
   (x = d ∨ x = 4) ∧ (y = 4 ∨ y = 0) ∧ 
   (2 : ℝ) * (4 - d) = 4) ↔ d = 2 :=
by
  sorry

end equal_area_division_l212_212260


namespace combined_selling_price_l212_212485

theorem combined_selling_price (C_c : ℕ) (C_s : ℕ) (C_m : ℕ) (L_c L_s L_m : ℕ)
  (hc : C_c = 1600)
  (hs : C_s = 12000)
  (hm : C_m = 45000)
  (hlc : L_c = 15)
  (hls : L_s = 10)
  (hlm : L_m = 5) :
  85 * C_c / 100 + 90 * C_s / 100 + 95 * C_m / 100 = 54910 := by
  sorry

end combined_selling_price_l212_212485


namespace sufficient_but_not_necessary_condition_l212_212293

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₀ : b > a) (h₁ : a > 0) :
  (1 / (a ^ 2) > 1 / (b ^ 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l212_212293


namespace simplify_and_evaluate_l212_212022

-- Define the expression as a function of a and b
def expr (a b : ℚ) : ℚ := 5 * a * b - 2 * (3 * a * b - (4 * a * b^2 + (1/2) * a * b)) - 5 * a * b^2

-- State the condition and the target result
theorem simplify_and_evaluate : 
  let a : ℚ := -1
  let b : ℚ := 1 / 2
  expr a b = -3 / 4 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_l212_212022


namespace algebraic_expression_l212_212102

def a (x : ℕ) := 2005 * x + 2009
def b (x : ℕ) := 2005 * x + 2010
def c (x : ℕ) := 2005 * x + 2011

theorem algebraic_expression (x : ℕ) : 
  a x ^ 2 + b x ^ 2 + c x ^ 2 - a x * b x - b x * c x - c x * a x = 3 :=
by
  sorry

end algebraic_expression_l212_212102


namespace inequality_solution_l212_212297

theorem inequality_solution (a : ℝ) (h : a^2 > 2 * a - 1) : a ≠ 1 := 
sorry

end inequality_solution_l212_212297


namespace multiples_count_l212_212538

theorem multiples_count (count_5 count_7 count_35 count_total : ℕ) :
  count_5 = 600 →
  count_7 = 428 →
  count_35 = 85 →
  count_total = count_5 + count_7 - count_35 →
  count_total = 943 :=
by
  sorry

end multiples_count_l212_212538


namespace bacon_suggestion_count_l212_212577

theorem bacon_suggestion_count (B : ℕ) (h1 : 408 = B + 366) : B = 42 :=
by
  sorry

end bacon_suggestion_count_l212_212577


namespace no_solution_to_equation_l212_212024

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, 8 / (x ^ 2 - 4) + 1 = x / (x - 2) :=
by
  sorry

end no_solution_to_equation_l212_212024


namespace find_radius_l212_212614

-- Definitions and conditions
variables (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 25)

-- Theorem statement
theorem find_radius : r = 50 :=
sorry

end find_radius_l212_212614


namespace jerry_pool_depth_l212_212387

theorem jerry_pool_depth :
  ∀ (total_gallons : ℝ) (gallons_drinking_cooking : ℝ) (gallons_per_shower : ℝ)
    (number_of_showers : ℝ) (pool_length : ℝ) (pool_width : ℝ)
    (gallons_per_cubic_foot : ℝ),
    total_gallons = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    number_of_showers = 15 →
    pool_length = 10 →
    pool_width = 10 →
    gallons_per_cubic_foot = 1 →
    (total_gallons - (gallons_drinking_cooking + gallons_per_shower * number_of_showers)) / 
    (pool_length * pool_width) = 6 := 
by
  intros total_gallons gallons_drinking_cooking gallons_per_shower number_of_showers pool_length pool_width gallons_per_cubic_foot
  intros total_gallons_eq drinking_cooking_eq shower_eq showers_eq length_eq width_eq cubic_foot_eq
  sorry

end jerry_pool_depth_l212_212387


namespace find_t_from_integral_l212_212413

theorem find_t_from_integral :
  (∫ x in (1 : ℝ)..t, (-1 / x + 2 * x)) = (3 - Real.log 2) → t = 2 :=
by
  sorry

end find_t_from_integral_l212_212413


namespace find_y_find_x_l212_212859

section
variables (a b : ℝ × ℝ) (x y : ℝ)

-- Definition of vectors a and b
def vec_a : ℝ × ℝ := (3, -2)
def vec_b (y : ℝ) : ℝ × ℝ := (-1, y)

-- Definition of perpendicular condition
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
-- Proof that y = -3/2 if a is perpendicular to b
theorem find_y (h : perpendicular vec_a (vec_b y)) : y = -3 / 2 :=
sorry

-- Definition of vectors a and c
def vec_c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Definition of parallel condition
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2
-- Proof that x = -15/2 if a is parallel to c
theorem find_x (h : parallel vec_a (vec_c x)) : x = -15 / 2 :=
sorry
end

end find_y_find_x_l212_212859


namespace regular_octahedron_vertices_count_l212_212289

def regular_octahedron_faces := 8
def regular_octahedron_edges := 12
def regular_octahedron_faces_shape := "equilateral triangle"
def regular_octahedron_vertices_meet := 4

theorem regular_octahedron_vertices_count :
  ∀ (F E V : ℕ),
    F = regular_octahedron_faces →
    E = regular_octahedron_edges →
    (∀ (v : ℕ), v = regular_octahedron_vertices_meet) →
    V = 6 :=
by
  intros F E V hF hE hV
  sorry

end regular_octahedron_vertices_count_l212_212289


namespace wreaths_per_greek_l212_212927

variable (m : ℕ) (m_pos : m > 0)

theorem wreaths_per_greek : ∃ x, x = 4 * m := 
sorry

end wreaths_per_greek_l212_212927


namespace trig_identity_l212_212403

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem trig_identity (x : ℝ) (h : f x = 2 * f' x) : 
  (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 := by
  sorry

end trig_identity_l212_212403


namespace simplify_expression_l212_212920

variable (a : ℝ) (ha : a ≠ 0)

theorem simplify_expression : (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end simplify_expression_l212_212920


namespace probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l212_212572

noncomputable def probability_sum_is_multiple_of_3 : ℝ :=
  let total_events := 36
  let favorable_events := 12
  favorable_events / total_events

noncomputable def probability_sum_is_prime : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

noncomputable def probability_second_greater_than_first : ℝ :=
  let total_events := 36
  let favorable_events := 15
  favorable_events / total_events

theorem probability_sum_multiple_of_3_eq_one_third :
  probability_sum_is_multiple_of_3 = 1 / 3 :=
by sorry

theorem probability_sum_prime_eq_five_twelfths :
  probability_sum_is_prime = 5 / 12 :=
by sorry

theorem probability_second_greater_than_first_eq_five_twelfths :
  probability_second_greater_than_first = 5 / 12 :=
by sorry

end probability_sum_multiple_of_3_eq_one_third_probability_sum_prime_eq_five_twelfths_probability_second_greater_than_first_eq_five_twelfths_l212_212572


namespace findAngleC_findPerimeter_l212_212827

noncomputable def triangleCondition (a b c : ℝ) (A B C : ℝ) : Prop :=
  let m := (b+c, Real.sin A)
  let n := (a+b, Real.sin C - Real.sin B)
  m.1 * n.2 = m.2 * n.1 -- m parallel to n

noncomputable def lawOfSines (a b c A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def areaOfTriangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C -- Area calculation by a, b, and angle between them

theorem findAngleC (a b c A B C : ℝ) : 
  triangleCondition a b c A B C ∧ lawOfSines a b c A B C → 
  Real.cos C = -1/2 :=
sorry

theorem findPerimeter (a b c A B C : ℝ) : 
  b = 4 ∧ areaOfTriangle a b c A B C = 4 * Real.sqrt 3 → 
  a = 4 ∧ b = 4 ∧ c = 4 * Real.sqrt 3 ∧ a + b + c = 8 + 4 * Real.sqrt 3 :=
sorry

end findAngleC_findPerimeter_l212_212827


namespace square_side_length_l212_212810

theorem square_side_length (a : ℝ) (n : ℕ) (P : ℝ) (h₀ : n = 5) (h₁ : 15 * (8 * a / 3) = P) (h₂ : P = 800) : a = 20 := 
by sorry

end square_side_length_l212_212810


namespace prove_f_3_equals_11_l212_212650

-- Assuming the given function definition as condition
def f (y : ℝ) : ℝ := sorry

-- The condition provided: f(x - 1/x) = x^2 + 1/x^2.
axiom function_definition (x : ℝ) (h : x ≠ 0): f (x - 1 / x) = x^2 + 1 / x^2

-- The goal is to prove that f(3) = 11
theorem prove_f_3_equals_11 : f 3 = 11 :=
by
  sorry

end prove_f_3_equals_11_l212_212650


namespace decrease_in_radius_l212_212506

theorem decrease_in_radius
  (dist_summer : ℝ)
  (dist_winter : ℝ)
  (radius_summer : ℝ) 
  (mile_to_inch : ℝ)
  (π : ℝ) 
  (δr : ℝ) :
  dist_summer = 560 →
  dist_winter = 570 →
  radius_summer = 20 →
  mile_to_inch = 63360 →
  π = Real.pi →
  δr = 0.33 :=
sorry

end decrease_in_radius_l212_212506


namespace probability_team_A_3_points_probability_team_A_1_point_probability_combined_l212_212040

namespace TeamProbabilities

noncomputable def P_team_A_3_points : ℚ :=
  (1 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_A_1_point : ℚ :=
  (1 / 3) * (2 / 3) * (2 / 3) + (2 / 3) * (1 / 3) * (2 / 3) + (2 / 3) * (2 / 3) * (1 / 3)

noncomputable def P_team_A_2_points : ℚ :=
  (1 / 3) * (1 / 3) * (2 / 3) + (1 / 3) * (2 / 3) * (1 / 3) + (2 / 3) * (1 / 3) * (1 / 3)

noncomputable def P_team_B_1_point : ℚ :=
  (1 / 2) * (2 / 3) * (3 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (1 / 2) * (2 / 3) * (1 / 4) + (1 / 2) * (2 / 3) * (1 / 4) +
  (1 / 2) * (1 / 3) * (1 / 4) + (1 / 2) * (1 / 3) * (3 / 4) + (2 / 3) * (2 / 3) * (1 / 4) + (2 / 3) * (1 / 3) * (1 / 4)

noncomputable def combined_probability : ℚ :=
  P_team_A_2_points * P_team_B_1_point

theorem probability_team_A_3_points :
  P_team_A_3_points = 1 / 27 := by
  sorry

theorem probability_team_A_1_point :
  P_team_A_1_point = 4 / 9 := by
  sorry

theorem probability_combined :
  combined_probability = 11 / 108 := by
  sorry

end TeamProbabilities

end probability_team_A_3_points_probability_team_A_1_point_probability_combined_l212_212040


namespace log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l212_212166

-- Define irrational numbers in Lean
def irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Prove that log base 2 of 3 is irrational
theorem log_two_three_irrational : irrational (Real.log 3 / Real.log 2) := 
sorry

-- Prove that log base sqrt(2) of 3 is irrational
theorem log_sqrt2_three_irrational : 
  irrational (Real.log 3 / (1/2 * Real.log 2)) := 
sorry

-- Prove that log base (5 + 3sqrt(2)) of (3 + 5sqrt(2)) is irrational
theorem log_five_plus_three_sqrt2_irrational :
  irrational (Real.log (3 + 5 * Real.sqrt 2) / Real.log (5 + 3 * Real.sqrt 2)) := 
sorry

end log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l212_212166


namespace perp_line_eq_l212_212176

theorem perp_line_eq (x y : ℝ) (c : ℝ) (hx : x = 1) (hy : y = 2) (hline : 2 * x + y - 5 = 0) :
  x - 2 * y + c = 0 ↔ c = 3 := 
by
  sorry

end perp_line_eq_l212_212176


namespace train_length_l212_212367

/-- Proof problem: 
  Given the speed of a train is 52 km/hr and it crosses a 280-meter long platform in 18 seconds,
  prove that the length of the train is 259.92 meters.
-/
theorem train_length (speed_kmh : ℕ) (platform_length : ℕ) (time_sec : ℕ) (speed_mps : ℝ) 
  (distance_covered : ℝ) (train_length : ℝ) :
  speed_kmh = 52 → platform_length = 280 → time_sec = 18 → 
  speed_mps = (speed_kmh * 1000) / 3600 → distance_covered = speed_mps * time_sec →
  train_length = distance_covered - platform_length →
  train_length = 259.92 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_length_l212_212367


namespace solve_quadratic_l212_212171

theorem solve_quadratic :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end solve_quadratic_l212_212171


namespace boys_and_girls_original_total_l212_212460

theorem boys_and_girls_original_total (b g : ℕ) 
(h1 : b = 3 * g) 
(h2 : b - 4 = 5 * (g - 4)) : 
b + g = 32 := 
sorry

end boys_and_girls_original_total_l212_212460


namespace derivative_at_one_eq_neg_one_l212_212685

variable {α : Type*} [TopologicalSpace α] {f : ℝ → ℝ}
-- condition: f is differentiable
variable (hf_diff : Differentiable ℝ f)
-- condition: limit condition
variable (h_limit : Tendsto (fun Δx => (f (1 + 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-2)))

-- proof goal: f'(1) = -1
theorem derivative_at_one_eq_neg_one : deriv f 1 = -1 := 
by
  sorry

end derivative_at_one_eq_neg_one_l212_212685


namespace diet_soda_bottles_l212_212615

theorem diet_soda_bottles (R D : ℕ) 
  (h1 : R = 60)
  (h2 : R = D + 41) :
  D = 19 :=
by {
  sorry
}

end diet_soda_bottles_l212_212615


namespace find_n_l212_212639

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end find_n_l212_212639


namespace dry_grapes_weight_l212_212905

theorem dry_grapes_weight (W_fresh : ℝ) (W_dry : ℝ) (P_water_fresh : ℝ) (P_water_dry : ℝ) :
  W_fresh = 40 → P_water_fresh = 0.80 → P_water_dry = 0.20 → W_dry = 10 := 
by 
  intros hWf hPwf hPwd 
  sorry

end dry_grapes_weight_l212_212905


namespace least_number_to_add_l212_212205

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) (k : ℕ) (l : ℕ) (h₁ : n = 1077) (h₂ : d = 23) (h₃ : n % d = r) (h₄ : d - r = k) (h₅ : r = 19) (h₆ : k = l) : l = 4 :=
by
  sorry

end least_number_to_add_l212_212205


namespace max_knights_between_knights_l212_212233

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l212_212233


namespace number_of_dress_designs_l212_212749

open Nat

theorem number_of_dress_designs : (3 * 4 = 12) :=
by
  rfl

end number_of_dress_designs_l212_212749


namespace find_multiplier_l212_212747

-- Define the numbers and the equation based on the conditions
def n : ℝ := 3.0
def m : ℝ := 7

-- State the problem in Lean 4
theorem find_multiplier : m * n = 3 * n + 12 := by
  -- Specific steps skipped; only structure is needed
  sorry

end find_multiplier_l212_212747


namespace max_knights_adjacent_to_two_other_knights_l212_212241

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l212_212241


namespace number_of_digits_of_2500_even_integers_l212_212194

theorem number_of_digits_of_2500_even_integers : 
  let even_integers := List.range (5000 : Nat) in
  let first_2500_even := List.filter (fun n => n % 2 = 0) even_integers in
  List.length (List.join (first_2500_even.map (fun n => n.toDigits Nat))) = 9448 :=
by
  sorry

end number_of_digits_of_2500_even_integers_l212_212194


namespace sum_cubes_of_roots_l212_212169

noncomputable def cube_root_sum_cubes (α β γ : ℝ) : ℝ :=
  α^3 + β^3 + γ^3
  
theorem sum_cubes_of_roots : 
  (cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3))) - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + 4/3) = 36 
  ∧
  ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3)) * ((Real.rpow 27 (1/3) + Real.rpow 64 (1/3) + Real.rpow 125 (1/3))^2 - 3 * ((Real.rpow 27 (1/3)) * (Real.rpow 64 (1/3)) + (Real.rpow 64 (1/3)) * (Real.rpow 125 (1/3)) + (Real.rpow 125 (1/3)) * (Real.rpow 27 (1/3)))) = 36) 
  → 
  cube_root_sum_cubes (Real.rpow 27 (1/3)) (Real.rpow 64 (1/3)) (Real.rpow 125 (1/3)) = 220 := 
sorry

end sum_cubes_of_roots_l212_212169


namespace possible_distances_AG_l212_212330

theorem possible_distances_AG (A B V G : ℝ) (AB VG : ℝ) (x AG : ℝ) :
  (AB = 600) →
  (VG = 600) →
  (AG = 3 * x) →
  (AG = 900 ∨ AG = 1800) :=
by
  intros h1 h2 h3
  sorry

end possible_distances_AG_l212_212330


namespace pairs_bought_after_donation_l212_212436

-- Definitions from conditions
def initial_pairs : ℕ := 80
def donation_percentage : ℕ := 30
def post_donation_pairs : ℕ := 62

-- The theorem to be proven
theorem pairs_bought_after_donation : (initial_pairs - (donation_percentage * initial_pairs / 100) + 6 = post_donation_pairs) :=
by
  sorry

end pairs_bought_after_donation_l212_212436


namespace S_n_bounds_l212_212119

open BigOperators

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 1 then 9 else (a_seq (n-1) + 2 * (n-1) + 5)

noncomputable def b_seq (n : ℕ) : ℝ :=
if n = 1 then 1 / 4 else ((n : ℝ) / (n+1)) * b_seq (n-1)

def sqrt (x : ℝ) : ℝ := real.sqrt x

def S (n : ℕ) : ℝ :=
∑ i in finset.range n, (b_seq (i+1) / sqrt (a_seq (i+1)))

theorem S_n_bounds (n : ℕ) : (1 / 12 : ℝ) ≤ S n ∧ S n < (1 / 4 : ℝ) := 
sorry

end S_n_bounds_l212_212119


namespace right_triangle_area_and_hypotenuse_l212_212665

-- Definitions based on given conditions
def a : ℕ := 24
def b : ℕ := 2 * a + 10

-- Statements based on the questions and correct answers
theorem right_triangle_area_and_hypotenuse :
  (1 / 2 : ℝ) * (a : ℝ) * (b : ℝ) = 696 ∧ (Real.sqrt ((a : ℝ)^2 + (b : ℝ)^2) = Real.sqrt 3940) := by
  sorry

end right_triangle_area_and_hypotenuse_l212_212665


namespace find_chocolate_cakes_l212_212914

variable (C : ℕ)
variable (h1 : 12 * C + 6 * 22 = 168)

theorem find_chocolate_cakes : C = 3 :=
by
  -- this is the proof placeholder
  sorry

end find_chocolate_cakes_l212_212914


namespace find_square_l212_212415

theorem find_square (y : ℝ) (h : (y + 5)^(1/3) = 3) : (y + 5)^2 = 729 := 
sorry

end find_square_l212_212415


namespace closed_path_even_length_l212_212209

def is_closed_path (steps : List Char) : Bool :=
  let net_vertical := steps.count 'U' - steps.count 'D'
  let net_horizontal := steps.count 'R' - steps.count 'L'
  net_vertical = 0 ∧ net_horizontal = 0

def move_length (steps : List Char) : Nat :=
  steps.length

theorem closed_path_even_length (steps : List Char) :
  is_closed_path steps = true → move_length steps % 2 = 0 :=
by
  -- Conditions extracted as definitions
  intros h
  -- The proof will handle showing that the length of the closed path is even
  sorry

end closed_path_even_length_l212_212209


namespace question1_question2_question3_l212_212050

-- Question 1
theorem question1 (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2) :
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Question 2
theorem question2 (x m n: ℕ) (h : x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2) :
  (m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7) :=
sorry

-- Question 3
theorem question3 : Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end question1_question2_question3_l212_212050


namespace simplify_complex_expression_l212_212323

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) + 4 * i * (7 - 3 * i) = 40 + 14 * i :=
by
  sorry

end simplify_complex_expression_l212_212323


namespace parabola_intersection_probability_l212_212463

-- Definitions
def parabola1 (a b x : ℝ) : ℝ := x^2 + a * x + b
def parabola2 (c d x : ℝ) : ℝ := x^2 + c * x + d + 2

-- Main Theorem
theorem parabola_intersection_probability :
  (∀ (a b c d : ℕ), 
  a ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  b ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  c ∈ {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  d ∈ {1, 2, 3, 4, 5, 6, 7, 8}) → 
  (prob (parabola1 a b = parabola2 c d) = 57 / 64) :=
sorry

end parabola_intersection_probability_l212_212463


namespace DE_eq_DF_l212_212889

variable {Point : Type}
variable {E A B C D F : Point}
variable (square : Π (A B C D : Point), Prop ) 
variable (is_parallel : Π (A B : Point), Prop) 
variable (E_outside_square : Prop)
variable (BE_eq_BD : Prop)
variable (BE_intersects_AD_at_F : Prop)

theorem DE_eq_DF
  (H1 : square A B C D)
  (H2 : is_parallel AE BD)
  (H3 : BE_eq_BD)
  (H4 : BE_intersects_AD_at_F) :
  DE = DF := 
sorry

end DE_eq_DF_l212_212889


namespace distribute_balls_into_boxes_l212_212845

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l212_212845


namespace only_real_solution_x_eq_6_l212_212790

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l212_212790


namespace find_principal_l212_212901

-- Problem conditions
variables (SI : ℚ := 4016.25) 
variables (R : ℚ := 0.08) 
variables (T : ℚ := 5)

-- The simple interest formula to find Principal
noncomputable def principal (SI : ℚ) (R : ℚ) (T : ℚ) : ℚ := SI * 100 / (R * T)

-- Lean statement to prove
theorem find_principal : principal SI R T = 10040.625 := by
  sorry

end find_principal_l212_212901


namespace volume_parallelepiped_l212_212884

open Real

theorem volume_parallelepiped :
  ∃ (a h : ℝ), 
    let S_base := (4 : ℝ)
    let AB := a
    let AD := 2 * a
    let lateral_face1 := (6 : ℝ)
    let lateral_face2 := (12 : ℝ)
    (AB * h = lateral_face1) ∧
    (AD * h = lateral_face2) ∧
    (1 / 2 * AD * S_base = AB * (1 / 2 * AD)) ∧ 
    (AB^2 + AD^2 - 2 * AB * AD * (cos (π / 6)) = S_base) ∧
    (a = 2) ∧
    (h = 3) ∧ 
    (S_base * h = 12) :=
sorry

end volume_parallelepiped_l212_212884


namespace cannot_fold_patternD_to_cube_l212_212272

def patternA : Prop :=
  -- 5 squares arranged in a cross shape
  let squares := 5
  let shape  := "cross"
  squares = 5 ∧ shape = "cross"

def patternB : Prop :=
  -- 4 squares in a straight line
  let squares := 4
  let shape  := "line"
  squares = 4 ∧ shape = "line"

def patternC : Prop :=
  -- 3 squares in an L shape, and 2 squares attached to one end of the L making a T shape
  let squares := 5
  let shape  := "T"
  squares = 5 ∧ shape = "T"

def patternD : Prop :=
  -- 6 squares in a "+" shape with one extra square
  let squares := 7
  let shape  := "plus"
  squares = 7 ∧ shape = "plus"

theorem cannot_fold_patternD_to_cube :
  patternD → ¬ (patternA ∨ patternB ∨ patternC) :=
by
  sorry

end cannot_fold_patternD_to_cube_l212_212272


namespace rectangular_field_area_l212_212609

theorem rectangular_field_area (a c : ℝ) (h_a : a = 13) (h_c : c = 17) :
  ∃ b : ℝ, (b = 2 * Real.sqrt 30) ∧ (a * b = 26 * Real.sqrt 30) :=
by
  sorry

end rectangular_field_area_l212_212609


namespace fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l212_212097

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 2

theorem fixed_point_when_a_2_b_neg2 :
  (∃ x : ℝ, f 2 (-2) x = x) → (x = -1 ∨ x = 2) :=
sorry

theorem range_of_a_for_two_fixed_points (a : ℝ) :
  (∀ b : ℝ, a ≠ 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b x1 = x1 ∧ f a b x2 = x2)) → (0 < a ∧ a < 2) :=
sorry

end fixed_point_when_a_2_b_neg2_range_of_a_for_two_fixed_points_l212_212097


namespace maciek_total_cost_l212_212063

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l212_212063


namespace leap_years_among_given_years_l212_212761

-- Definitions for conditions
def is_divisible (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def is_leap_year (y : Nat) : Prop :=
  is_divisible y 4 ∧ (¬ is_divisible y 100 ∨ is_divisible y 400)

-- Statement of the problem
theorem leap_years_among_given_years :
  is_leap_year 1996 ∧ is_leap_year 2036 ∧ (¬ is_leap_year 1700) ∧ (¬ is_leap_year 1998) :=
by
  -- Proof would go here
  sorry

end leap_years_among_given_years_l212_212761


namespace probability_six_heads_before_tail_l212_212428

theorem probability_six_heads_before_tail :
  let q := 1 / 64 in q = 1/64 :=
by
  let q := 1 / 64
  sorry

end probability_six_heads_before_tail_l212_212428


namespace find_real_solutions_l212_212795

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l212_212795


namespace find_percentage_l212_212545

theorem find_percentage (x p : ℝ) (h1 : x = 840) (h2 : 0.25 * x + 15 = p / 100 * 1500) : p = 15 := 
by
  sorry

end find_percentage_l212_212545


namespace arithmetic_sequence_property_l212_212304

variable {a : ℕ → ℕ}

-- Given condition in the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∃ d c : ℕ, ∀ n : ℕ, a n = c + n * d

def condition (a : ℕ → ℕ) : Prop := a 4 + a 8 = 16

-- Problem statement
theorem arithmetic_sequence_property (a : ℕ → ℕ)
  (h_arith_seq : arithmetic_sequence a)
  (h_condition : condition a) :
  a 2 + a 6 + a 10 = 24 :=
sorry

end arithmetic_sequence_property_l212_212304


namespace total_inches_of_rope_l212_212678

noncomputable def inches_of_rope (last_week_feet : ℕ) (less_feet : ℕ) (feet_to_inches : ℕ → ℕ) : ℕ :=
  let last_week_inches := feet_to_inches last_week_feet
  let this_week_feet := last_week_feet - less_feet
  let this_week_inches := feet_to_inches this_week_feet
  last_week_inches + this_week_inches

theorem total_inches_of_rope 
  (six_feet : ℕ := 6)
  (four_feet_less : ℕ := 4)
  (conversion : ℕ → ℕ := λ feet, feet * 12) :
  inches_of_rope six_feet four_feet_less conversion = 96 := by
  sorry

end total_inches_of_rope_l212_212678


namespace maciek_total_purchase_cost_l212_212071

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l212_212071


namespace correct_propositions_l212_212954

-- Define the propositions as boolean conditions
def proposition1 (α β : Plane) (m n : Line) : Prop :=
  (α ∩ β = m) → (n ⊆ α) → (m ∥ n) ∨ (∃ p, p ∈ m ∧ p ∈ n)

def proposition2 (α β : Plane) (m n : Line) : Prop :=
  (α ∥ β) → (m ⊆ α) → (n ⊆ β) → (m ∥ n)

def proposition3 (α : Plane) (m n : Line) : Prop :=
  (m ∥ α) → (m ∥ n) → (n ∥ α)

def proposition4 (α β : Plane) (m n : Line) : Prop :=
  (α ∩ β = m) → (m ∥ n) → (n ∥ α) ∨ (n ∥ β)

-- The main theorem statement
theorem correct_propositions (α β : Plane) (m n : Line) :
  (proposition1 α β m n) ∧ (proposition4 α β m n) :=
by
  sorry

end correct_propositions_l212_212954


namespace solve_for_x_l212_212957

noncomputable def find_x (x : ℝ) : Prop :=
  2^12 = 64^x

theorem solve_for_x (x : ℝ) (h : find_x x) : x = 2 :=
by
  sorry

end solve_for_x_l212_212957


namespace product_plus_one_is_square_l212_212130

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end product_plus_one_is_square_l212_212130


namespace units_digit_of_23_mul_51_squared_l212_212269

theorem units_digit_of_23_mul_51_squared : 
  ∀ n m : ℕ, (n % 10 = 3) ∧ ((m^2 % 10) = 1) → (n * m^2 % 10) = 3 :=
by
  intros n m h
  sorry

end units_digit_of_23_mul_51_squared_l212_212269


namespace number_is_2_point_5_l212_212929

theorem number_is_2_point_5 (x : ℝ) (h: x^2 + 50 = (x - 10)^2) : x = 2.5 := 
by
  sorry

end number_is_2_point_5_l212_212929


namespace jenny_correct_number_l212_212555

theorem jenny_correct_number (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 :=
by
  sorry

end jenny_correct_number_l212_212555


namespace bottles_per_day_l212_212254

theorem bottles_per_day (b d : ℕ) (h1 : b = 8066) (h2 : d = 74) : b / d = 109 :=
by {
  sorry
}

end bottles_per_day_l212_212254


namespace two_teachers_place_A_probability_l212_212476

-- Given 3 teachers, each being assigned randomly to one of two places A or B
def probability_two_teachers_place_A : ℚ :=
  let total_assignments := 2^3
  let ways_two_teachers_A := Nat.choose 3 2
  ways_two_teachers_A / total_assignments

-- Proof statement
theorem two_teachers_place_A_probability : probability_two_teachers_place_A = 3 / 8 := by
  sorry

end two_teachers_place_A_probability_l212_212476


namespace exp_ineq_of_r_gt_one_l212_212294

theorem exp_ineq_of_r_gt_one {x r : ℝ} (hx : x > 0) (hr : r > 1) : (1 + x)^r > 1 + r * x :=
by
  sorry

end exp_ineq_of_r_gt_one_l212_212294


namespace product_of_primes_is_582_l212_212601

-- Define the relevant primes based on the conditions.
def smallest_one_digit_prime_1 := 2
def smallest_one_digit_prime_2 := 3
def largest_two_digit_prime := 97

-- Define the product of these primes as stated in the problem.
def product_of_primes := smallest_one_digit_prime_1 * smallest_one_digit_prime_2 * largest_two_digit_prime

-- Prove that this product equals to 582.
theorem product_of_primes_is_582 : product_of_primes = 582 :=
by {
  sorry
}

end product_of_primes_is_582_l212_212601


namespace mailman_should_give_junk_mail_l212_212596

-- Definitions from the conditions
def houses_in_block := 20
def junk_mail_per_house := 32

-- The mathematical equivalent proof problem statement in Lean 4
theorem mailman_should_give_junk_mail : 
  junk_mail_per_house * houses_in_block = 640 :=
  by sorry

end mailman_should_give_junk_mail_l212_212596


namespace sum_arithmetic_sequence_ge_four_l212_212523

theorem sum_arithmetic_sequence_ge_four
  (a_n : ℕ → ℚ) -- arithmetic sequence
  (S : ℕ → ℚ) -- sum of the first n terms of the sequence
  (h_arith_seq : ∀ n, S n = (n * a_n 1) + (n * (n - 1) / 2) * (a_n 2 - a_n 1))
  (p q : ℕ)
  (hpq_ne : p ≠ q)
  (h_sp : S p = p / q)
  (h_sq : S q = q / p) :
  S (p + q) ≥ 4 :=
by
  sorry

end sum_arithmetic_sequence_ge_four_l212_212523


namespace integer_rational_ratio_l212_212470

open Real

theorem integer_rational_ratio (a b : ℤ) (h : (a : ℝ) + sqrt b = sqrt (15 + sqrt 216)) : (a : ℚ) / b = 1 / 2 := 
by 
  -- Omitted proof 
  sorry

end integer_rational_ratio_l212_212470


namespace diagonal_of_rectangular_prism_l212_212755

noncomputable def diagonal_length (a b c : ℕ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 12 18 15 = 3 * Real.sqrt 77 :=
by
  sorry

end diagonal_of_rectangular_prism_l212_212755


namespace circle_arrangement_l212_212769

open Finset

theorem circle_arrangement (σ : Perm (Fin 12)) :
  (∀ i : Fin 12, (σ (i + 1)).val^2 % 13 = (σ i).val * (σ (i + 2)).val % 13) →
  True := sorry

end circle_arrangement_l212_212769


namespace ratio_of_candies_l212_212976

theorem ratio_of_candies (candiesEmily candiesBob : ℕ) (candiesJennifer : ℕ) 
  (hEmily : candiesEmily = 6) 
  (hBob : candiesBob = 4)
  (hJennifer : candiesJennifer = 3 * candiesBob) : 
  (candiesJennifer / Nat.gcd candiesJennifer candiesEmily) = 2 ∧ (candiesEmily / Nat.gcd candiesJennifer candiesEmily) = 1 := 
by
  sorry

end ratio_of_candies_l212_212976


namespace johns_gas_usage_per_week_l212_212675

def mpg : ℕ := 30
def miles_to_work_each_way : ℕ := 20
def days_per_week_to_work : ℕ := 5
def leisure_miles_per_week : ℕ := 40

theorem johns_gas_usage_per_week : 
  (2 * miles_to_work_each_way * days_per_week_to_work + leisure_miles_per_week) / mpg = 8 :=
by
  sorry

end johns_gas_usage_per_week_l212_212675


namespace max_knights_between_knights_l212_212236

theorem max_knights_between_knights (knights samurais total : Nat) (condition_knights_right samurai : Nat) :
  knights = 40 → samurais = 10 → condition_knights_right = 7 → total = knights + samurais →
  ∃ max_knights, max_knights = 32 ∧ 
  (∃ (k : Nat), k ≤ total ∧ (∀ n, (0 < n) → (n < 7) → max_knights = knights - n)) :=
by
  sorry

end max_knights_between_knights_l212_212236


namespace find_m_l212_212970

theorem find_m (a : ℕ → ℝ) (m : ℝ)
  (h1 : (∀ (x : ℝ), x^2 + m * x - 8 = 0 → x = a 2 ∨ x = a 8))
  (h2 : a 4 + a 6 = a 5 ^ 2 + 1) :
  m = -2 :=
sorry

end find_m_l212_212970


namespace short_side_is_7_l212_212754

variable (L S : ℕ)

-- Given conditions
def perimeter : ℕ := 38
def long_side : ℕ := 12

-- In Lean, prove that the short side is 7 given L and P
theorem short_side_is_7 (h1 : 2 * L + 2 * S = perimeter) (h2 : L = long_side) : S = 7 := by
  sorry

end short_side_is_7_l212_212754


namespace max_knights_between_knights_l212_212223

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l212_212223


namespace solve_quadratic_eq1_solve_quadratic_eq2_l212_212704

-- Define the first equation
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 := by
  sorry

-- Define the second equation
theorem solve_quadratic_eq2 (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l212_212704


namespace no_nontrivial_integer_solutions_l212_212697

theorem no_nontrivial_integer_solutions (a b c d : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * d^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
by
  intro h
  sorry

end no_nontrivial_integer_solutions_l212_212697


namespace cost_of_adult_ticket_l212_212078

-- Conditions provided in the original problem.
def total_people : ℕ := 23
def child_tickets_cost : ℕ := 10
def total_money_collected : ℕ := 246
def children_attended : ℕ := 7

-- Define some unknown amount A for the adult tickets cost to be solved.
variable (A : ℕ)

-- Define the Lean statement for the proof problem.
theorem cost_of_adult_ticket :
  16 * A = 176 →
  A = 11 :=
by
  -- Start the proof (this part will be filled out during the proof process).
  sorry

#check cost_of_adult_ticket  -- To ensure it type-checks

end cost_of_adult_ticket_l212_212078


namespace roots_polynomial_sum_cubes_l212_212562

theorem roots_polynomial_sum_cubes (u v w : ℂ) (h : (∀ x, (x = u ∨ x = v ∨ x = w) → 5 * x ^ 3 + 500 * x + 1005 = 0)) :
  (u + v) ^ 3 + (v + w) ^ 3 + (w + u) ^ 3 = 603 := sorry

end roots_polynomial_sum_cubes_l212_212562


namespace train_length_proper_l212_212492

noncomputable def train_length (speed distance_time pass_time : ℝ) : ℝ :=
  speed * pass_time

axiom speed_of_train : ∀ (distance_time : ℝ), 
  (10 * 1000 / (15 * 60)) = 11.11

theorem train_length_proper :
  train_length 11.11 900 10 = 111.1 := by
  sorry

end train_length_proper_l212_212492


namespace sum_of_roots_l212_212193

theorem sum_of_roots : 
  ( ∀ x : ℝ, x^2 - 7*x + 10 = 0 → x = 2 ∨ x = 5 ) → 
  ( 2 + 5 = 7 ) := 
by
  sorry

end sum_of_roots_l212_212193


namespace initial_number_of_students_l212_212998

/-- 
Theorem: If the average mark of the students of a class in an exam is 90, and 2 students whose average mark is 45 are excluded, resulting in the average mark of the remaining students being 95, then the initial number of students is 20.
-/
theorem initial_number_of_students (N : ℕ) (T : ℕ)
  (h1 : T = N * 90)
  (h2 : (T - 90) / (N - 2) = 95) : 
  N = 20 :=
sorry

end initial_number_of_students_l212_212998


namespace mrs_peterson_change_l212_212869

def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

theorem mrs_peterson_change : 
  (num_bills * value_per_bill) - (num_tumblers * cost_per_tumbler) = 50 :=
by
  sorry

end mrs_peterson_change_l212_212869


namespace books_price_arrangement_l212_212514

theorem books_price_arrangement (c : ℝ) (prices : Fin 40 → ℝ)
  (h₁ : ∀ i : Fin 39, prices i.succ = prices i + 3)
  (h₂ : prices ⟨39, by norm_num⟩ = prices ⟨19, by norm_num⟩ + prices ⟨20, by norm_num⟩) :
  prices 20 = prices 19 + 3 := 
sorry

end books_price_arrangement_l212_212514


namespace problem_statement_l212_212816

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem problem_statement : ((U \ A) ∪ (U \ B)) = {0, 1, 3, 4, 5} := by
  sorry

end problem_statement_l212_212816


namespace total_texts_sent_is_97_l212_212010

def textsSentOnMondayAllison := 5
def textsSentOnMondayBrittney := 5
def textsSentOnMondayCarol := 5

def textsSentOnTuesdayAllison := 15
def textsSentOnTuesdayBrittney := 10
def textsSentOnTuesdayCarol := 12

def textsSentOnWednesdayAllison := 20
def textsSentOnWednesdayBrittney := 18
def textsSentOnWednesdayCarol := 7

def totalTextsAllison := textsSentOnMondayAllison + textsSentOnTuesdayAllison + textsSentOnWednesdayAllison
def totalTextsBrittney := textsSentOnMondayBrittney + textsSentOnTuesdayBrittney + textsSentOnWednesdayBrittney
def totalTextsCarol := textsSentOnMondayCarol + textsSentOnTuesdayCarol + textsSentOnWednesdayCarol

def totalTextsAllThree := totalTextsAllison + totalTextsBrittney + totalTextsCarol

theorem total_texts_sent_is_97 : totalTextsAllThree = 97 := by
  sorry

end total_texts_sent_is_97_l212_212010


namespace ant_moves_probability_l212_212371

theorem ant_moves_probability :
  let m := 73
  let n := 48
  m + n = 121 := by
  sorry

end ant_moves_probability_l212_212371


namespace solution_to_cubic_equation_l212_212787

theorem solution_to_cubic_equation :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
begin
  sorry
end

end solution_to_cubic_equation_l212_212787


namespace problem1_problem2_l212_212287

-- Assume x and y are positive numbers
variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

-- Prove that x^3 + y^3 >= x^2*y + y^2*x
theorem problem1 : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
by sorry

-- Prove that m ≤ 2 given the additional condition
variables (m : ℝ)
theorem problem2 (cond : (x/y^2 + y/x^2) ≥ m/2 * (1/x + 1/y)) : m ≤ 2 :=
by sorry

end problem1_problem2_l212_212287


namespace base_seven_to_ten_l212_212043

theorem base_seven_to_ten : 
  (7 * 7^4 + 6 * 7^3 + 5 * 7^2 + 4 * 7^1 + 3 * 7^0) = 19141 := 
by 
  sorry

end base_seven_to_ten_l212_212043


namespace pascal_row_10_sum_l212_212917

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l212_212917


namespace sabrina_basil_leaves_l212_212993

-- Definitions of variables
variables (S B V : ℕ)

-- Conditions as definitions in Lean
def condition1 : Prop := B = 2 * S
def condition2 : Prop := S = V - 5
def condition3 : Prop := B + S + V = 29

-- Problem statement
theorem sabrina_basil_leaves (h1 : condition1 S B) (h2 : condition2 S V) (h3 : condition3 S B V) : B = 12 :=
by {
  sorry
}

end sabrina_basil_leaves_l212_212993


namespace panda_bamboo_consumption_l212_212014

theorem panda_bamboo_consumption (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
  sorry

end panda_bamboo_consumption_l212_212014


namespace sufficient_but_not_necessary_l212_212635

-- Define the equations of the lines
def line1 (a : ℝ) (x y : ℝ) : ℝ := 2 * x + a * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := (a - 1) * x + 3 * y - 2

-- Define the condition for parallel lines by comparing their slopes
def parallel_condition (a : ℝ) : Prop :=  (2 * 3 = a * (a - 1))

theorem sufficient_but_not_necessary (a : ℝ) : 3 ≤ a :=
  sorry

end sufficient_but_not_necessary_l212_212635


namespace conquering_Loulan_necessary_for_returning_home_l212_212136

theorem conquering_Loulan_necessary_for_returning_home : 
  ∀ (P Q : Prop), (¬ Q → ¬ P) → (P → Q) :=
by sorry

end conquering_Loulan_necessary_for_returning_home_l212_212136


namespace maximum_bugs_on_board_l212_212706

-- Definition of the problem board size, bug movement directions, and non-collision rule
def board_size := 10
inductive Direction
| up | down | left | right

-- The main theorem stating the maximum number of bugs on the board
theorem maximum_bugs_on_board (bugs : List (Nat × Nat × Direction)) :
  (∀ (x y : Nat) (d : Direction) (bug : Nat × Nat × Direction), 
    bug = (x, y, d) → 
    x < board_size ∧ y < board_size ∧ 
    (∀ (c : Nat × Nat × Direction), 
      c ∈ bugs → bug ≠ c → bug.1 ≠ c.1 ∨ bug.2 ≠ c.2)) →
  List.length bugs <= 40 :=
sorry

end maximum_bugs_on_board_l212_212706


namespace sequence_solution_l212_212950

theorem sequence_solution (a : ℕ → ℝ)
  (h₁ : a 1 = 0)
  (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 4 * (Real.sqrt (a n + 1)) + 4) :
  ∀ n ≥ 1, a n = 4 * n^2 - 4 * n :=
by
  sorry

end sequence_solution_l212_212950


namespace sequences_count_l212_212121

open BigOperators

def consecutive_blocks (n : ℕ) : ℕ :=
  2 * (n * (n + 1) / 2 - 1) - 2

theorem sequences_count {n : ℕ} (h : n = 15) :
  consecutive_blocks n = 238 :=
by
  sorry

end sequences_count_l212_212121


namespace composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l212_212902

-- Definition for part (a)
def composite_base_greater_than_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + 2*n^2 + 1) = a * b

-- Proof statement for part (a)
theorem composite_10201_in_all_bases_greater_than_two (n : ℕ) (h : n > 2) : composite_base_greater_than_two n :=
by sorry

-- Definition for part (b)
def composite_in_all_bases (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + n^2 + 1) = a * b

-- Proof statement for part (b)
theorem composite_10101_in_all_bases (n : ℕ) : composite_in_all_bases n :=
by sorry

end composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l212_212902


namespace cos_neg_300_l212_212738

theorem cos_neg_300 : Real.cos (-(300 : ℝ) * Real.pi / 180) = 1 / 2 :=
by
  -- Proof goes here
  sorry

end cos_neg_300_l212_212738


namespace extra_kilometers_per_hour_l212_212735

theorem extra_kilometers_per_hour (S a : ℝ) (h : a > 2) : 
  (S / (a - 2)) - (S / a) = (S / (a - 2)) - (S / a) :=
by sorry

end extra_kilometers_per_hour_l212_212735


namespace john_finishes_fourth_task_at_12_18_PM_l212_212310

theorem john_finishes_fourth_task_at_12_18_PM :
  let start_time := 8 * 60 + 45 -- Start time in minutes from midnight
  let third_task_time := 11 * 60 + 25 -- End time of the third task in minutes from midnight
  let total_time_three_tasks := third_task_time - start_time -- Total time in minutes to complete three tasks
  let time_per_task := total_time_three_tasks / 3 -- Time per task in minutes
  let fourth_task_end_time := third_task_time + time_per_task -- End time of the fourth task in minutes from midnight
  fourth_task_end_time = 12 * 60 + 18 := -- Expected end time in minutes from midnight
  sorry

end john_finishes_fourth_task_at_12_18_PM_l212_212310


namespace james_missing_legos_l212_212309

theorem james_missing_legos  (h1 : 500 > 0) (h2 : 500 % 2 = 0) (h3 : 245 < 500)  :
  let total_legos := 500
  let used_legos := total_legos / 2
  let leftover_legos := total_legos - used_legos
  let legos_in_box := 245
  leftover_legos - legos_in_box = 5 := by
{
  sorry
}

end james_missing_legos_l212_212309


namespace students_accounting_majors_l212_212137

theorem students_accounting_majors (p q r s : ℕ) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) (h5 : p * q * r * s = 1365) : p = 3 := 
by 
  sorry

end students_accounting_majors_l212_212137


namespace used_crayons_l212_212440

open Nat

theorem used_crayons (N B T U : ℕ) (h1 : N = 2) (h2 : B = 8) (h3 : T = 14) (h4 : T = N + U + B) : U = 4 :=
by
  -- Proceed with the proof here
  sorry

end used_crayons_l212_212440


namespace total_tickets_sold_l212_212032

theorem total_tickets_sold (A C : ℕ) (hC : C = 16) (h1 : 3 * C = 48) (h2 : 5 * A + 3 * C = 178) : 
  A + C = 42 :=
by
  sorry

end total_tickets_sold_l212_212032


namespace factorize_x_squared_minus_121_l212_212089

theorem factorize_x_squared_minus_121 (x : ℝ) : (x^2 - 121) = (x + 11) * (x - 11) :=
by
  sorry

end factorize_x_squared_minus_121_l212_212089


namespace average_weight_of_16_boys_is_50_25_l212_212333

theorem average_weight_of_16_boys_is_50_25
  (W : ℝ)
  (h1 : 8 * 45.15 = 361.2)
  (h2 : 24 * 48.55 = 1165.2)
  (h3 : 16 * W + 361.2 = 1165.2) :
  W = 50.25 :=
sorry

end average_weight_of_16_boys_is_50_25_l212_212333


namespace assembly_shortest_time_l212_212326

-- Define the times taken for each assembly path
def time_ACD : ℕ := 3 + 4
def time_EDF : ℕ := 4 + 2

-- State the theorem for the shortest time required to assemble the product
theorem assembly_shortest_time : max time_ACD time_EDF + 4 = 13 :=
by {
  -- Introduction of the given conditions and simplified value calculation
  sorry
}

end assembly_shortest_time_l212_212326


namespace programmer_debugging_hours_l212_212775

theorem programmer_debugging_hours 
  (total_hours : ℕ)
  (flow_chart_fraction coding_fraction : ℚ)
  (flow_chart_fraction_eq : flow_chart_fraction = 1/4)
  (coding_fraction_eq : coding_fraction = 3/8)
  (hours_worked : total_hours = 48) :
  ∃ (debugging_hours : ℚ), debugging_hours = 18 := 
by
  sorry

end programmer_debugging_hours_l212_212775


namespace eval_expression_l212_212376

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l212_212376


namespace problem_statement_l212_212407

theorem problem_statement (a b c : ℝ) 
  (h1 : a - 2 * b + c = 0) 
  (h2 : a + 2 * b + c < 0) : b < 0 ∧ b^2 - a * c ≥ 0 :=
by
  sorry

end problem_statement_l212_212407


namespace solve_equation_l212_212806

theorem solve_equation (x : ℝ) (h₀ : x = 46) :
  ( (8 / (Real.sqrt (x - 10) - 10)) + 
    (2 / (Real.sqrt (x - 10) - 5)) + 
    (9 / (Real.sqrt (x - 10) + 5)) + 
    (15 / (Real.sqrt (x - 10) + 10)) = 0) := 
by 
  sorry

end solve_equation_l212_212806


namespace proof_problem_l212_212603

variables (a b : ℝ)

noncomputable def expr := (2 * a⁻¹ + (a⁻¹ / b)) / a

theorem proof_problem (h1 : a = 1/3) (h2 : b = 3) : expr a b = 21 :=
by
  sorry

end proof_problem_l212_212603


namespace trig_identity_l212_212250

theorem trig_identity :
  (Real.tan (30 * Real.pi / 180) * Real.cos (60 * Real.pi / 180) + Real.tan (45 * Real.pi / 180) * Real.cos (30 * Real.pi / 180)) = (2 * Real.sqrt 3) / 3 :=
by
  -- Proof is omitted
  sorry

end trig_identity_l212_212250


namespace number_of_green_hats_l212_212187

variables (B G : ℕ)

-- Given conditions as definitions
def totalHats : Prop := B + G = 85
def totalCost : Prop := 6 * B + 7 * G = 530

-- The statement we need to prove
theorem number_of_green_hats (h1 : totalHats B G) (h2 : totalCost B G) : G = 20 :=
sorry

end number_of_green_hats_l212_212187


namespace inequality_x_add_inv_x_ge_two_l212_212996

theorem inequality_x_add_inv_x_ge_two (x : ℝ) (hx : x > 0) : x + 1/x ≥ 2 :=
  sorry

end inequality_x_add_inv_x_ge_two_l212_212996


namespace question_l212_212163

section

variable (x : ℝ)
variable (p q : Prop)

-- Define proposition p: ∀ x in [0,1], e^x ≥ 1
def Proposition_p : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → Real.exp x ≥ 1

-- Define proposition q: ∃ x in ℝ such that x^2 + x + 1 < 0
def Proposition_q : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- The problem to prove: p ∨ q
theorem question (p q : Prop) (hp : Proposition_p) (hq : ¬ Proposition_q) : p ∨ q := by
  sorry

end

end question_l212_212163


namespace only_real_solution_x_eq_6_l212_212791

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l212_212791


namespace find_k_and_slope_l212_212774

theorem find_k_and_slope : 
  ∃ k : ℝ, (∃ y : ℝ, (3 + y = 8) ∧ (k = -3 * 3 + y)) ∧ (k = -4) ∧ 
  (∀ x y : ℝ, (x + y = 8) → (∃ m b : ℝ, y = m * x + b ∧ m = -1)) :=
by {
  sorry
}

end find_k_and_slope_l212_212774


namespace at_least_one_not_less_than_one_l212_212103

open Real

theorem at_least_one_not_less_than_one (x : ℝ) :
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1 :=
by
  -- Definitions of a, b, and c
  let a := x^2 + 1/2
  let b := 2 - x
  let c := x^2 - x + 1
  -- Proof is omitted
  sorry

end at_least_one_not_less_than_one_l212_212103


namespace additional_savings_l212_212061

-- Defining the conditions
def initial_price : ℝ := 50
def discount_one : ℝ := 6
def discount_percentage : ℝ := 0.15

-- Defining the final prices according to the two methods
def first_method : ℝ := (1 - discount_percentage) * (initial_price - discount_one)
def second_method : ℝ := (1 - discount_percentage) * initial_price - discount_one

-- Defining the savings for the two methods
def savings_first_method : ℝ := initial_price - first_method
def savings_second_method : ℝ := initial_price - second_method

-- Proving that the second method results in an additional 0.90 savings
theorem additional_savings : (savings_second_method - savings_first_method) = 0.90 :=
by
  sorry

end additional_savings_l212_212061


namespace quadratic_equation_C_has_real_solutions_l212_212899

theorem quadratic_equation_C_has_real_solutions :
  ∀ (x : ℝ), ∃ (a b c : ℝ), a = 1 ∧ b = 3 ∧ c = -2 ∧ a*x^2 + b*x + c = 0 :=
by
  sorry

end quadratic_equation_C_has_real_solutions_l212_212899


namespace graduates_distribution_l212_212386

theorem graduates_distribution (n : ℕ) (k : ℕ)
    (h_n : n = 5) (h_k : k = 3)
    (h_dist : ∀ e : Fin k, ∃ g : Finset (Fin n), g.card ≥ 1) :
    ∃ d : ℕ, d = 150 :=
by
  have h_distribution := 150
  use h_distribution
  sorry

end graduates_distribution_l212_212386


namespace number_of_children_l212_212334

-- Definitions of given conditions
def total_passengers := 170
def men := 90
def women := men / 2
def adults := men + women
def children := total_passengers - adults

-- Theorem statement
theorem number_of_children : children = 35 :=
by
  sorry

end number_of_children_l212_212334


namespace exists_n_divisible_by_5_l212_212025

open Int

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h1 : 5 ∣ (a * m^3 + b * m^2 + c * m + d)) 
  (h2 : ¬ (5 ∣ d)) :
  ∃ n : ℤ, 5 ∣ (d * n^3 + c * n^2 + b * n + a) :=
by
  sorry

end exists_n_divisible_by_5_l212_212025


namespace gcd_of_powers_l212_212382

theorem gcd_of_powers (a b c : ℕ) (h1 : a = 2^105 - 1) (h2 : b = 2^115 - 1) (h3 : c = 1023) :
  Nat.gcd a b = c :=
by sorry

end gcd_of_powers_l212_212382


namespace cosine_inequality_l212_212982

theorem cosine_inequality
  (x y z : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2)
  (hz : 0 < z ∧ z < π / 2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤
  (Real.cos x + Real.cos y + Real.cos z) / 3 := sorry

end cosine_inequality_l212_212982


namespace hyperbola_asymptotes_l212_212583

theorem hyperbola_asymptotes:
  (∀ x y : Real, (x^2 / 16 - y^2 / 9 = 1) → (y = 3 / 4 * x ∨ y = -3 / 4 * x)) :=
by {
  sorry
}

end hyperbola_asymptotes_l212_212583


namespace problem1_solution_set_problem2_range_of_a_l212_212106

-- Definitions and statements for Problem 1
def f1 (x : ℝ) : ℝ := -12 * x ^ 2 - 2 * x + 2

theorem problem1_solution_set :
  (∃ a b : ℝ, a = -12 ∧ b = -2 ∧
    ∀ x : ℝ, f1 x > 0 → -1 / 2 < x ∧ x < 1 / 3) :=
by sorry

-- Definitions and statements for Problem 2
def f2 (x a : ℝ) : ℝ := a * x ^ 2 - x + 2

theorem problem2_range_of_a :
  (∃ b : ℝ, b = -1 ∧
    ∀ a : ℝ, (∀ x : ℝ, f2 x a < 0 → false) → a ≥ 1 / 8) :=
by sorry

end problem1_solution_set_problem2_range_of_a_l212_212106


namespace arithmetic_square_root_problem_l212_212710

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end arithmetic_square_root_problem_l212_212710


namespace stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l212_212252

-- Definition of the speeds
def speed_excluding_stoppages_A : ℕ := 60
def speed_including_stoppages_A : ℕ := 48
def speed_excluding_stoppages_B : ℕ := 75
def speed_including_stoppages_B : ℕ := 60
def speed_excluding_stoppages_C : ℕ := 90
def speed_including_stoppages_C : ℕ := 72

-- Theorem to prove the stopped time per hour for each bus
theorem stopped_time_per_hour_A : (speed_excluding_stoppages_A - speed_including_stoppages_A) * 60 / speed_excluding_stoppages_A = 12 := sorry

theorem stopped_time_per_hour_B : (speed_excluding_stoppages_B - speed_including_stoppages_B) * 60 / speed_excluding_stoppages_B = 12 := sorry

theorem stopped_time_per_hour_C : (speed_excluding_stoppages_C - speed_including_stoppages_C) * 60 / speed_excluding_stoppages_C = 12 := sorry

end stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l212_212252


namespace rope_total_in_inches_l212_212677

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l212_212677


namespace min_ab_value_l212_212282

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + 9 * b + 7) : a * b ≥ 49 :=
sorry

end min_ab_value_l212_212282


namespace range_of_a_l212_212686

noncomputable def f (a : ℝ) (x : ℝ) := Real.sqrt (Real.exp x + (Real.exp 1 - 1) * x - a)
def exists_b_condition (a : ℝ) : Prop := ∃ b : ℝ, b ∈ Set.Icc 0 1 ∧ f a b = b

theorem range_of_a (a : ℝ) : exists_b_condition a → a ∈ Set.Icc 1 (2 * Real.exp 1 - 2) :=
sorry

end range_of_a_l212_212686


namespace arithmetic_expression_l212_212767

theorem arithmetic_expression :
  (5^6) / (5^4) + 3^3 - 6^2 = 16 := by
  sorry

end arithmetic_expression_l212_212767


namespace largest_class_students_l212_212608

theorem largest_class_students (x : ℕ) (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 105) : x = 25 :=
by {
  sorry
}

end largest_class_students_l212_212608


namespace product_plus_one_is_square_l212_212129

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l212_212129


namespace bianca_made_after_selling_l212_212812

def bianca_initial_cupcakes : ℕ := 14
def bianca_sold_cupcakes : ℕ := 6
def bianca_final_cupcakes : ℕ := 25

theorem bianca_made_after_selling :
  (bianca_initial_cupcakes - bianca_sold_cupcakes) + (bianca_final_cupcakes - (bianca_initial_cupcakes - bianca_sold_cupcakes)) = bianca_final_cupcakes :=
by
  sorry

end bianca_made_after_selling_l212_212812


namespace sum_of_possible_x_l212_212719

theorem sum_of_possible_x 
  (x : ℝ)
  (squareSide : ℝ) 
  (rectangleLength : ℝ) 
  (rectangleWidth : ℝ) 
  (areaCondition : (rectangleLength * rectangleWidth) = 3 * (squareSide ^ 2)) : 
  6 + 6.5 = 12.5 := 
by 
  sorry

end sum_of_possible_x_l212_212719


namespace number_of_days_worked_l212_212482

theorem number_of_days_worked (total_toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : total_toys_per_week = 6000) (h₂ : toys_per_day = 1500) : (total_toys_per_week / toys_per_day) = 4 :=
by
  sorry

end number_of_days_worked_l212_212482


namespace fred_dimes_l212_212815

theorem fred_dimes (initial_dimes borrowed_dimes : ℕ) (h1 : initial_dimes = 7) (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 :=
by
  sorry

end fred_dimes_l212_212815


namespace knights_in_exchange_l212_212435

noncomputable def count_knights (total_islanders : ℕ) (odd_statements : ℕ) (even_statements : ℕ) : ℕ :=
if total_islanders % 2 = 0 ∧ odd_statements = total_islanders ∧ even_statements = total_islanders then
    total_islanders / 2
else
    0

theorem knights_in_exchange : count_knights 30 30 30 = 15 :=
by
    -- proof part will go here but is not required.
    sorry

end knights_in_exchange_l212_212435


namespace sum_of_non_domain_elements_l212_212093

theorem sum_of_non_domain_elements :
    let f (x : ℝ) : ℝ := 1 / (1 + 1 / (1 + 1 / (1 + 1 / x)))
    let is_not_in_domain (x : ℝ) := x = 0 ∨ x = -1 ∨ x = -1/2 ∨ x = -2/3
    (0 : ℝ) + (-1) + (-1/2) + (-2/3) = -19/6 :=
by 
  sorry

end sum_of_non_domain_elements_l212_212093


namespace maciek_total_purchase_cost_l212_212069

-- Define the cost of pretzels
def pretzel_cost : ℕ := 4

-- Define the cost of chips
def chip_cost : ℕ := pretzel_cost + (75 * pretzel_cost) / 100

-- Calculate the total cost
def total_cost : ℕ := 2 * pretzel_cost + 2 * chip_cost

-- Rewrite the math proof problem statement
theorem maciek_total_purchase_cost : total_cost = 22 :=
by
  -- Skip the proof
  sorry

end maciek_total_purchase_cost_l212_212069


namespace problem_statement_l212_212401

theorem problem_statement (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → a (n + 1) = a (n - 1) / (1 + n * a (n - 1) * a n))
  (h_initial_0 : a 0 = 1)
  (h_initial_1 : a 1 = 1) :
  1 / (a 190 * a 200) = 19901 :=
by
  sorry

end problem_statement_l212_212401


namespace negation_of_proposition_l212_212030

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), (a = b → a^2 = a * b)) = ∀ (a b : ℝ), (a ≠ b → a^2 ≠ a * b) :=
sorry

end negation_of_proposition_l212_212030


namespace new_pie_crust_flour_l212_212863

theorem new_pie_crust_flour :
  ∀ (p1 p2 : ℕ) (f1 f2 : ℚ) (c : ℚ),
  p1 = 40 →
  f1 = 1 / 8 →
  p1 * f1 = c →
  p2 = 25 →
  p2 * f2 = c →
  f2 = 1 / 5 :=
begin
  intros p1 p2 f1 f2 c,
  intros h_p1 h_f1 h_c h_p2 h_new_c,
  sorry
end

end new_pie_crust_flour_l212_212863


namespace sozopolian_ineq_find_p_l212_212718

noncomputable def is_sozopolian (p a b c : ℕ) : Prop :=
  p % 2 = 1 ∧
  Nat.Prime p ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a * b + 1) % p = 0 ∧
  (b * c + 1) % p = 0 ∧
  (c * a + 1) % p = 0

theorem sozopolian_ineq (p a b c : ℕ) (hp : is_sozopolian p a b c) :
  p + 2 ≤ (a + b + c) / 3 :=
sorry

theorem find_p (p : ℕ) :
  (∃ a b c : ℕ, is_sozopolian p a b c ∧ (a + b + c) / 3 = p + 2) ↔ p = 5 :=
sorry

end sozopolian_ineq_find_p_l212_212718


namespace cary_earnings_l212_212379

variable (shoe_cost : ℕ) (saved_amount : ℕ)
variable (lawns_per_weekend : ℕ) (weeks_needed : ℕ)
variable (total_cost_needed : ℕ) (total_lawns : ℕ) (earn_per_lawn : ℕ)
variable (h1 : shoe_cost = 120)
variable (h2 : saved_amount = 30)
variable (h3 : lawns_per_weekend = 3)
variable (h4 : weeks_needed = 6)
variable (h5 : total_cost_needed = shoe_cost - saved_amount)
variable (h6 : total_lawns = lawns_per_weekend * weeks_needed)
variable (h7 : earn_per_lawn = total_cost_needed / total_lawns)

theorem cary_earnings :
  earn_per_lawn = 5 :=
by 
  sorry

end cary_earnings_l212_212379


namespace find_D_l212_212626

theorem find_D (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : D = 1 :=
by
  sorry

end find_D_l212_212626


namespace danny_initial_wrappers_l212_212634

def initial_wrappers (total_wrappers: ℕ) (found_wrappers: ℕ): ℕ :=
  total_wrappers - found_wrappers

theorem danny_initial_wrappers : initial_wrappers 57 30 = 27 :=
by
  exact rfl

end danny_initial_wrappers_l212_212634


namespace ticket_cost_l212_212764

theorem ticket_cost (a : ℝ) (h1 : (6 * a + 5 * (2 / 3 * a) = 47.25)) :
  10 * a + 8 * (2 / 3 * a) = 77.625 :=
by
  sorry

end ticket_cost_l212_212764


namespace sheep_transaction_gain_l212_212751

noncomputable def percent_gain (cost_per_sheep total_sheep sold_sheep remaining_sheep : ℕ) : ℚ :=
let total_cost := (cost_per_sheep : ℚ) * total_sheep
let initial_revenue := total_cost
let price_per_sheep := initial_revenue / sold_sheep
let remaining_revenue := remaining_sheep * price_per_sheep
let total_revenue := initial_revenue + remaining_revenue
let profit := total_revenue - total_cost
(profit / total_cost) * 100

theorem sheep_transaction_gain :
  percent_gain 1 1000 950 50 = -47.37 := sorry

end sheep_transaction_gain_l212_212751


namespace sum_of_first_9_terms_zero_l212_212395

variable (a_n : ℕ → ℝ) (d a₁ : ℝ)
def arithmetic_seq := ∀ n, a_n n = a₁ + (n - 1) * d

def condition (a_n : ℕ → ℝ) := (a_n 2 + a_n 9 = a_n 6)

theorem sum_of_first_9_terms_zero 
  (h_arith : arithmetic_seq a_n d a₁) 
  (h_cond : condition a_n) : 
  (9 * a₁ + (9 * 8 / 2) * d) = 0 :=
by
  sorry

end sum_of_first_9_terms_zero_l212_212395


namespace minimal_blue_chips_value_l212_212746

noncomputable def minimal_blue_chips (r g b : ℕ) : Prop :=
b ≥ r / 3 ∧
b ≤ g / 4 ∧
r + g ≥ 75

theorem minimal_blue_chips_value : ∃ (b : ℕ), minimal_blue_chips 33 44 b ∧ b = 11 :=
by
  have b := 11
  use b
  sorry

end minimal_blue_chips_value_l212_212746


namespace sum_of_arithmetic_sequence_zero_l212_212581

noncomputable def arithmetic_sequence_sum (S : ℕ → ℤ) : Prop :=
S 20 = S 40

theorem sum_of_arithmetic_sequence_zero {S : ℕ → ℤ} (h : arithmetic_sequence_sum S) : 
  S 60 = 0 :=
sorry

end sum_of_arithmetic_sequence_zero_l212_212581


namespace equation_of_line_through_P_l212_212363

theorem equation_of_line_through_P (P : (ℝ × ℝ)) (A B : (ℝ × ℝ))
  (hP : P = (1, 3))
  (hMidpoint : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hA : A.2 = 0)
  (hB : B.1 = 0) :
  ∃ c : ℝ, 3 * c + 1 = 3 ∧ (3 * A.1 / c + A.2 / 6 = 1) ∧ (3 * B.1 / c + B.2 / 6 = 1) := sorry

end equation_of_line_through_P_l212_212363


namespace number_of_ways_songs_can_be_liked_l212_212081

-- Define types and conditions
def Songs := Finset ℕ
def Amy := {a : ℕ | a ∈ Songs}
def Beth := {b : ℕ | b ∈ Songs}
def Jo := {c : ℕ | c ∈ Songs}

noncomputable def exactly_one_from_each_pair (S_AB S_BC S_CA : Songs) : Prop :=
  S_AB.nonempty ∧ S_BC.nonempty ∧ S_CA.nonempty

noncomputable def amy_likes_two (S_A : Songs) : Prop :=
  S_A.card ≥ 2

noncomputable def distinct_songs (S_A S_AB S_BC S_CA : Songs) : Prop :=
  S_A ∪ S_AB ∪ S_BC ∪ S_CA = Songs ∧
  S_A ∩ S_AB = ∅ ∧
  S_A ∩ S_BC = ∅ ∧
  S_A ∩ S_CA = ∅ ∧
  S_AB ∩ S_BC = ∅ ∧
  S_AB ∩ S_CA = ∅ ∧
  S_BC ∩ S_CA = ∅

noncomputable def count_ways_to_satisfy_conditions (Songs : Finset ℕ) : ℕ :=
  let total_songs := 5 in
  if Songs.card = total_songs then
    (nat.choose total_songs 1) * (nat.choose (total_songs - 1) 1) * (nat.choose (total_songs - 2) 1) * (nat.choose (total_songs - 3) 2)
  else 0

theorem number_of_ways_songs_can_be_liked :
  ∀ (S_A S_AB S_BC S_CA : Songs),
  S_A.card + S_AB.card + S_BC.card + S_CA.card = 5 →
  exactly_one_from_each_pair S_AB S_BC S_CA →
  amy_likes_two S_A →
  distinct_songs S_A S_AB S_BC S_CA →
  count_ways_to_satisfy_conditions (S_A ∪ S_AB ∪ S_BC ∪ S_CA) = 60 :=
sorry

end number_of_ways_songs_can_be_liked_l212_212081


namespace maximize_profit_l212_212752

noncomputable def profit (x : ℕ) : ℝ :=
  let price := (180 + 10 * x : ℝ)
  let rooms_occupied := (50 - x : ℝ)
  let expenses := 20
  (price - expenses) * rooms_occupied

theorem maximize_profit :
  ∃ x : ℕ, profit x = profit 17 → (180 + 10 * x) = 350 :=
by
  use 17
  sorry

end maximize_profit_l212_212752


namespace max_knights_seated_next_to_two_knights_l212_212244

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l212_212244


namespace simplify_expression_l212_212576

def expression1 (x : ℝ) : ℝ :=
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6)

def expression2 (x : ℝ) : ℝ :=
  2 * x^3 + 7 * x^2 - 3 * x + 14

theorem simplify_expression (x : ℝ) : expression1 x = expression2 x :=
by 
  sorry

end simplify_expression_l212_212576


namespace cone_height_l212_212480

noncomputable def height_of_cone (r : ℝ) (n : ℕ) : ℝ :=
  let sector_circumference := (2 * Real.pi * r) / n
  let cone_base_radius := sector_circumference / (2 * Real.pi)
  Real.sqrt (r^2 - cone_base_radius^2)

theorem cone_height
  (r_original : ℝ)
  (n : ℕ)
  (h : r_original = 10)
  (hc : n = 4) :
  height_of_cone r_original n = 5 * Real.sqrt 3 := by
  sorry

end cone_height_l212_212480


namespace ratio_of_speeds_l212_212662

theorem ratio_of_speeds :
  ∀ (t : ℝ) (v_A v_B : ℝ),
  (v_A = 360 / t) →
  (v_B = 480 / t) →
  v_A / v_B = 3 / 4 :=
by
  intros t v_A v_B h_A h_B
  rw [←h_A, ←h_B]
  field_simp
  norm_num

# In this theorem, we state that given the conditions of the speeds
# and distances covered in the problem, we prove that the ratio of
# their speeds is 3:4.

end ratio_of_speeds_l212_212662


namespace multiply_exponents_l212_212770

variable (a : ℝ)

theorem multiply_exponents :
  a * a^2 * (-a)^3 = -a^6 := 
sorry

end multiply_exponents_l212_212770


namespace f_divisible_by_64_l212_212019

theorem f_divisible_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end f_divisible_by_64_l212_212019


namespace watch_cost_price_l212_212368

open Real

theorem watch_cost_price (CP SP1 SP2 : ℝ)
    (h1 : SP1 = CP * 0.85)
    (h2 : SP2 = CP * 1.10)
    (h3 : SP2 = SP1 + 450) : CP = 1800 :=
by
  sorry

end watch_cost_price_l212_212368


namespace complex_pow_imaginary_unit_l212_212647

theorem complex_pow_imaginary_unit (i : ℂ) (h : i^2 = -1) : i^2015 = -i :=
sorry

end complex_pow_imaginary_unit_l212_212647


namespace range_of_t_l212_212951

noncomputable def a_n (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-3)
noncomputable def c_n (n : ℕ) (t : ℝ) : ℝ := 
  let a := a_n n t 
  let b := b_n n
  (a + b) / 2 + (|a - b|) / 2

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, n > 0 → c_n n t ≥ c_n 3 t) : 10/3 < t ∧ t < 5 :=
    sorry

end range_of_t_l212_212951


namespace f_3_add_f_10_l212_212281

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f (-x) = f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom f_1 : f 1 = 4

theorem f_3_add_f_10 : f 3 + f 10 = 4 :=
by sorry

end f_3_add_f_10_l212_212281


namespace valerie_laptop_purchase_l212_212466

/-- Valerie wants to buy a new laptop priced at $800. She receives $100 dollars from her parents,
$60 dollars from her uncle, and $40 dollars from her siblings for her graduation.
She also makes $20 dollars each week from tutoring. How many weeks must she save 
her tutoring income, along with her graduation money, to buy the laptop? -/
theorem valerie_laptop_purchase :
  let price_of_laptop : ℕ := 800
  let graduation_money : ℕ := 100 + 60 + 40
  let weekly_tutoring_income : ℕ := 20
  let remaining_amount_needed : ℕ := price_of_laptop - graduation_money
  let weeks_needed := remaining_amount_needed / weekly_tutoring_income
  weeks_needed = 30 :=
by
  sorry

end valerie_laptop_purchase_l212_212466


namespace triangle_area_l212_212487

theorem triangle_area (d : ℝ) (h : d = 8 * Real.sqrt 10) (ang : ∀ {α β γ : ℝ}, α = 45 ∨ β = 45 ∨ γ = 45) :
  ∃ A : ℝ, A = 160 :=
by
  sorry

end triangle_area_l212_212487


namespace average_temperature_second_to_fifth_days_l212_212449

variable (T1 T2 T3 T4 T5 : ℝ)

theorem average_temperature_second_to_fifth_days 
  (h1 : (T1 + T2 + T3 + T4) / 4 = 58)
  (h2 : T1 / T5 = 7 / 8)
  (h3 : T5 = 32) :
  (T2 + T3 + T4 + T5) / 4 = 59 :=
by
  sorry

end average_temperature_second_to_fifth_days_l212_212449


namespace matrix_A_pow_50_l212_212847

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 1],
  ![0, 1]
]

theorem matrix_A_pow_50 : A^50 = ![
  ![1, 50],
  ![0, 1]
] :=
sorry

end matrix_A_pow_50_l212_212847


namespace cubic_solution_l212_212798

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l212_212798


namespace average_speed_correct_l212_212471

variable (t1 t2 : ℝ) -- time components in hours
variable (v1 v2 : ℝ) -- speed components in km/h

-- conditions
def time1 := 20 / 60 -- 20 minutes converted to hours
def time2 := 40 / 60 -- 40 minutes converted to hours
def speed1 := 60 -- speed in km/h for the first segment
def speed2 := 90 -- speed in km/h for the second segment

-- total distance traveled
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def total_distance := distance1 + distance2

-- total time taken
def total_time := time1 + time2

-- average speed
def average_speed := total_distance / total_time

-- proof statement
theorem average_speed_correct : average_speed = 80 := by
  sorry

end average_speed_correct_l212_212471


namespace part1_part2_part3_l212_212173

-- Definitions of conditions
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_even_between (a b : ℕ) : ℕ := sum_even b - sum_even a

-- Problem 1: Prove that for n = 8, S = 72
theorem part1 (n : ℕ) (h : n = 8) : sum_even n = 72 := by
  rw [h]
  exact rfl

-- Problem 2: Prove the general formula for the sum of the first n consecutive even numbers
theorem part2 (n : ℕ) : sum_even n = n * (n + 1) := by
  exact rfl

-- Problem 3: Prove the sum of 102 to 212 is 8792 using the formula
theorem part3 : sum_even_between 50 106 = 8792 := by
  sorry

end part1_part2_part3_l212_212173


namespace symmetry_and_monotonicity_l212_212946

noncomputable def function_f (x : ℝ) : ℝ :=
if x >= 1 then log x else log (2 - x)

theorem symmetry_and_monotonicity :
  function_f (2 - 1 / 2) = function_f (1 / 2) ∧
  function_f (2 - 1 / 3) = function_f (1 / 3) ∧
  function_f (2 - 2) = function_f (2) ∧
  function_f (1 / 2) < function_f (1 / 3) ∧
  function_f (1 / 3) < function_f (2) := 
by
    sorry

end symmetry_and_monotonicity_l212_212946


namespace pyramid_can_be_oblique_l212_212072

-- Define what it means for the pyramid to have a regular triangular base.
def regular_triangular_base (pyramid : Type) : Prop := sorry

-- Define what it means for each lateral face to be an isosceles triangle.
def isosceles_lateral_faces (pyramid : Type) : Prop := sorry

-- Define what it means for a pyramid to be oblique.
def can_be_oblique (pyramid : Type) : Prop := sorry

-- Defining pyramid as a type.
variable (pyramid : Type)

-- The theorem stating the problem's conclusion.
theorem pyramid_can_be_oblique 
  (h1 : regular_triangular_base pyramid) 
  (h2 : isosceles_lateral_faces pyramid) : 
  can_be_oblique pyramid :=
sorry

end pyramid_can_be_oblique_l212_212072


namespace additional_treetags_l212_212302

noncomputable def initial_numerals : Finset ℕ := {1, 2, 3, 4}
noncomputable def initial_letters : Finset Char := {'A', 'E', 'I'}
noncomputable def initial_symbols : Finset Char := {'!', '@', '#', '$'}
noncomputable def added_numeral : Finset ℕ := {5}
noncomputable def added_symbols : Finset Char := {'&'}

theorem additional_treetags : 
  let initial_treetags := initial_numerals.card * initial_letters.card * initial_symbols.card
  let new_numerals := initial_numerals ∪ added_numeral
  let new_symbols := initial_symbols ∪ added_symbols
  let new_treetags := new_numerals.card * initial_letters.card * new_symbols.card
  new_treetags - initial_treetags = 27 := 
by 
  sorry

end additional_treetags_l212_212302


namespace order_abc_l212_212936

noncomputable def a : ℝ := (3 * (2 - Real.log 3)) / Real.exp 2
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.sqrt (Real.exp 1)) / (2 * Real.exp 1)

theorem order_abc : c < a ∧ a < b := by
  sorry

end order_abc_l212_212936


namespace clothing_weight_removed_l212_212589

/-- 
In a suitcase, the initial ratio of books to clothes to electronics, by weight measured in pounds, 
is 7:4:3. The electronics weight 9 pounds. Someone removes some pounds of clothing, doubling the ratio of books to clothes. 
This theorem verifies the weight of clothing removed is 1.5 pounds.
-/
theorem clothing_weight_removed 
  (B C E : ℕ) 
  (initial_ratio : B / 7 = C / 4 ∧ C / 4 = E / 3)
  (E_val : E = 9)
  (new_ratio : ∃ x : ℝ, B / (C - x) = 2) : 
  ∃ x : ℝ, x = 1.5 := 
sorry

end clothing_weight_removed_l212_212589


namespace number_of_ways_to_choose_books_l212_212290

def num_books := 15
def books_to_choose := 3

theorem number_of_ways_to_choose_books : Nat.choose num_books books_to_choose = 455 := by
  sorry

end number_of_ways_to_choose_books_l212_212290


namespace dilution_problem_l212_212588
-- Definitions of the conditions
def volume_initial : ℝ := 15
def concentration_initial : ℝ := 0.60
def concentration_final : ℝ := 0.40
def amount_alcohol_initial : ℝ := volume_initial * concentration_initial

-- Proof problem statement in Lean 4
theorem dilution_problem : 
  ∃ (x : ℝ), x = 7.5 ∧ 
              amount_alcohol_initial = concentration_final * (volume_initial + x) :=
sorry

end dilution_problem_l212_212588


namespace full_price_shoes_l212_212992

variable (P : ℝ)

def full_price (P : ℝ) : ℝ := P
def discount_1_year (P : ℝ) : ℝ := 0.80 * P
def discount_3_years (P : ℝ) : ℝ := 0.75 * discount_1_year P
def price_after_discounts (P : ℝ) : ℝ := 0.60 * P

theorem full_price_shoes : price_after_discounts P = 51 → full_price P = 85 :=
by
  -- Placeholder for proof steps,
  sorry

end full_price_shoes_l212_212992


namespace exponent_problem_l212_212414

theorem exponent_problem (m : ℕ) : 8^2 = 4^2 * 2^m → m = 2 := by
  intro h
  sorry

end exponent_problem_l212_212414


namespace problem_1_problem_2_problem_3_l212_212385

-- Non-computational definitions and conditions for the problem
noncomputable def deck := {card // card.1 ∈ {1, 2, ..., 10} ∧ card.2 ∈ {"Hearts", "Spades", "Diamonds", "Clubs"}}

def drawing_heart (card : deck) : Prop := card.2 = "Hearts"
def drawing_spade (card : deck) : Prop := card.2 = "Spades"
def drawing_red_card (card : deck) : Prop := card.2 ∈ {"Hearts", "Diamonds"}
def drawing_black_card (card : deck) : Prop := card.2 ∈ {"Spades", "Clubs"}
def number_multiple_of_5 (card : deck) : Prop := card.1 % 5 = 0
def number_greater_than_9 (card : deck) : Prop := card.1 > 9

-- Theorems to be proven
theorem problem_1 : 
  (∀ card : deck, drawing_heart card → ¬drawing_spade card) ∧
  ¬(∀ card : deck, drawing_heart card ∨ drawing_spade card) :=
sorry

theorem problem_2 :
  (∀ card : deck, drawing_red_card card → ¬drawing_black_card card) ∧
  (∀ card : deck, drawing_red_card card ∨ drawing_black_card card) :=
sorry 

theorem problem_3 :
  ¬(∀ card : deck, number_multiple_of_5 card → ¬number_greater_than_9 card) ∧
  ¬(∀ card : deck, number_multiple_of_5 card ∨ number_greater_than_9 card) := 
sorry

end problem_1_problem_2_problem_3_l212_212385


namespace area_of_triangle_l212_212807

noncomputable def findAreaOfTriangle (a b : ℝ) (cosAOF : ℝ) : ℝ := sorry

theorem area_of_triangle (a b cosAOF : ℝ)
  (ha : a = 15 / 7)
  (hb : b = Real.sqrt 21)
  (hcos : cosAOF = 2 / 5) :
  findAreaOfTriangle a b cosAOF = 6 := by
  rw [ha, hb, hcos]
  sorry

end area_of_triangle_l212_212807


namespace price_after_discounts_l212_212082

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * (1 - 0.10)
  let second_discount := first_discount * (1 - 0.20)
  second_discount

theorem price_after_discounts (initial_price : ℝ) (h : final_price initial_price = 174.99999999999997) : 
  final_price initial_price = 175 := 
by {
  sorry
}

end price_after_discounts_l212_212082


namespace bus_trip_length_l212_212479

theorem bus_trip_length (v T : ℝ) 
    (h1 : 2 * v + (T - 2 * v) * (3 / (2 * v)) + 1 = T / v + 5)
    (h2 : 2 + 30 / v + (T - (2 * v + 30)) * (3 / (2 * v)) + 1 = T / v + 4) : 
    T = 180 :=
    sorry

end bus_trip_length_l212_212479


namespace gcd_of_gx_and_x_l212_212528

theorem gcd_of_gx_and_x (x : ℤ) (hx : x % 11739 = 0) :
  Int.gcd ((3 * x + 4) * (5 * x + 3) * (11 * x + 5) * (x + 11)) x = 3 :=
sorry

end gcd_of_gx_and_x_l212_212528


namespace jacket_total_price_correct_l212_212074

/-- The original price of the jacket -/
def original_price : ℝ := 120

/-- The initial discount rate -/
def initial_discount_rate : ℝ := 0.15

/-- The additional discount in dollars -/
def additional_discount : ℝ := 10

/-- The sales tax rate -/
def sales_tax_rate : ℝ := 0.10

/-- The calculated total amount the shopper pays for the jacket including all discounts and tax -/
def total_amount_paid : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let price_after_additional_discount := price_after_initial_discount - additional_discount
  price_after_additional_discount * (1 + sales_tax_rate)

theorem jacket_total_price_correct : total_amount_paid = 101.20 :=
  sorry

end jacket_total_price_correct_l212_212074


namespace smallest_sum_of_four_distinct_numbers_l212_212720

theorem smallest_sum_of_four_distinct_numbers 
  (S : Finset ℤ) 
  (h : S = {8, 26, -2, 13, -4, 0}) :
  ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a + b + c + d = 2 :=
sorry

end smallest_sum_of_four_distinct_numbers_l212_212720


namespace log_expression_calculation_l212_212249

theorem log_expression_calculation :
  2 * log 5 10 + log 5 0.25 = 2 := by
  -- assume properties of logarithms and definition of 0.25
  sorry

end log_expression_calculation_l212_212249


namespace fiona_first_to_toss_eight_l212_212086

theorem fiona_first_to_toss_eight :
  (∃ p : ℚ, p = 49/169 ∧
    (∀ n:ℕ, (7/8:ℚ)^(3*n) * (1/8) = if n = 0 then (49/512) else (49/512) * (343/512)^n)) :=
sorry

end fiona_first_to_toss_eight_l212_212086


namespace xy_is_necessary_but_not_sufficient_l212_212818

theorem xy_is_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → xy = 0) ∧ (xy = 0 → ¬(x^2 + y^2 ≠ 0)) := by
  sorry

end xy_is_necessary_but_not_sufficient_l212_212818


namespace solution_set_of_inequality_l212_212033

theorem solution_set_of_inequality :
  { x : ℝ | |1 - 2 * x| < 3 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

end solution_set_of_inequality_l212_212033


namespace parallel_lines_slope_eq_l212_212604

variable (k : ℝ)

theorem parallel_lines_slope_eq (h : 5 = 3 * k) : k = 5 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l212_212604


namespace quadratic_difference_sum_l212_212500

theorem quadratic_difference_sum :
  let a := 2
  let b := -10
  let c := 3
  let Δ := b * b - 4 * a * c
  let root1 := (10 + Real.sqrt Δ) / (2 * a)
  let root2 := (10 - Real.sqrt Δ) / (2 * a)
  let diff := root1 - root2
  let m := 19  -- from the difference calculation
  let n := 1   -- from the simplified form
  m + n = 20 :=
by
  -- Placeholders for calculation and proof steps.
  sorry

end quadratic_difference_sum_l212_212500


namespace no_nat_k_divides_7_l212_212873

theorem no_nat_k_divides_7 (k : ℕ) : ¬ 7 ∣ (2^(2*k - 1) + 2^k + 1) := 
sorry

end no_nat_k_divides_7_l212_212873


namespace machine_A_production_rate_l212_212317

theorem machine_A_production_rate :
  ∀ (A B T_A T_B : ℝ),
    500 = A * T_A →
    500 = B * T_B →
    B = 1.25 * A →
    T_A = T_B + 15 →
    A = 100 / 15 :=
by
  intros A B T_A T_B hA hB hRate hTime
  sorry

end machine_A_production_rate_l212_212317


namespace cubic_solution_unique_real_l212_212783

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end cubic_solution_unique_real_l212_212783


namespace sqrt_of_mixed_number_l212_212087

theorem sqrt_of_mixed_number : sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_of_mixed_number_l212_212087


namespace inequality_sqrt_l212_212161

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l212_212161


namespace number_of_real_solutions_l212_212167

noncomputable def system_of_equations (n : ℕ) (a b c : ℝ) (x : Fin n → ℝ) : Prop :=
∀ i : Fin n, a * (x i) ^ 2 + b * (x i) + c = x (⟨(i + 1) % n, sorry⟩)

theorem number_of_real_solutions
  (a b c : ℝ)
  (h : a ≠ 0)
  (n : ℕ)
  (x : Fin n → ℝ) :
  (b - 1) ^ 2 - 4 * a * c < 0 → ¬(∃ x : Fin n → ℝ, system_of_equations n a b c x) ∧
  (b - 1) ^ 2 - 4 * a * c = 0 → ∃! x : Fin n → ℝ, system_of_equations n a b c x ∧
  (b - 1) ^ 2 - 4 * a * c > 0 → ∃ x : Fin n → ℝ, ∃ y : Fin n → ℝ, x ≠ y ∧ system_of_equations n a b c x ∧ system_of_equations n a b c y := 
sorry

end number_of_real_solutions_l212_212167


namespace eval_expression_l212_212375

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l212_212375


namespace optimal_addition_amount_l212_212465

theorem optimal_addition_amount (a b g : ℝ) (h₁ : a = 628) (h₂ : b = 774) (h₃ : g = 718) : 
    b + a - g = 684 :=
by
  sorry

end optimal_addition_amount_l212_212465


namespace Amelia_weekly_sales_l212_212760

-- Conditions
def monday_sales : ℕ := 45
def tuesday_sales : ℕ := 45 - 16
def remaining_sales : ℕ := 16

-- Question to Answer
def total_weekly_sales : ℕ := 90

-- Lean 4 Statement to Prove
theorem Amelia_weekly_sales : monday_sales + tuesday_sales + remaining_sales = total_weekly_sales :=
by
  sorry

end Amelia_weekly_sales_l212_212760


namespace real_solution_unique_l212_212804

theorem real_solution_unique (x : ℝ) : 
  (x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) ↔ x = 6 := 
begin
  sorry
end

end real_solution_unique_l212_212804


namespace total_rope_in_inches_l212_212681

-- Definitions for conditions
def feet_last_week : ℕ := 6
def feet_less : ℕ := 4
def inches_per_foot : ℕ := 12

-- Condition: rope bought this week
def feet_this_week := feet_last_week - feet_less

-- Condition: total rope bought in feet
def total_feet := feet_last_week + feet_this_week

-- Condition: total rope bought in inches
def total_inches := total_feet * inches_per_foot

-- Theorem statement
theorem total_rope_in_inches : total_inches = 96 := by
  sorry

end total_rope_in_inches_l212_212681


namespace truth_of_compound_proposition_l212_212940

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, x^2 > 0

theorem truth_of_compound_proposition : p ∧ ¬ q :=
by
  sorry

end truth_of_compound_proposition_l212_212940


namespace container_solution_exists_l212_212727

theorem container_solution_exists (x y : ℕ) (h : 130 * x + 160 * y = 3000) : 
  (x = 12) ∧ (y = 9) :=
by sorry

end container_solution_exists_l212_212727


namespace find_B_l212_212361

theorem find_B (A C B : ℕ) (hA : A = 520) (hC : C = A + 204) (hCB : C = B + 179) : B = 545 :=
by
  sorry

end find_B_l212_212361


namespace plane_through_point_and_line_l212_212808

noncomputable def point_on_plane (A B C D : ℤ) (x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

def line_eq_1 (x y : ℤ) : Prop :=
  3 * x + 4 * y - 20 = 0

def line_eq_2 (y z : ℤ) : Prop :=
  -3 * y + 2 * z + 18 = 0

theorem plane_through_point_and_line 
  (A B C D : ℤ)
  (h_point : point_on_plane A B C D 1 9 (-8))
  (h_line1 : ∀ x y, line_eq_1 x y → point_on_plane A B C D x y 0)
  (h_line2 : ∀ y z, line_eq_2 y z → point_on_plane A B C D 0 y z)
  (h_gcd : Int.gcd (Int.gcd (Int.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1) 
  (h_pos : A > 0) :
  A = 75 ∧ B = -29 ∧ C = 86 ∧ D = 274 :=
sorry

end plane_through_point_and_line_l212_212808


namespace sqrt_inequality_l212_212158

theorem sqrt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a = 1) :
  sqrt (a + 1 / a) + sqrt (b + 1 / b) + sqrt (c + 1 / c) ≥ 2 * (sqrt a + sqrt b + sqrt c) :=
sorry

end sqrt_inequality_l212_212158


namespace divides_power_sum_l212_212425

theorem divides_power_sum (a b c : ℤ) (h : a + b + c ∣ a^2 + b^2 + c^2) : ∀ k : ℕ, a + b + c ∣ a^(2^k) + b^(2^k) + c^(2^k) :=
by
  intro k
  induction k with
  | zero =>
    sorry -- Base case proof
  | succ k ih =>
    sorry -- Inductive step proof

end divides_power_sum_l212_212425


namespace book_selection_l212_212709

theorem book_selection :
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  tier1 + tier2 + tier3 = 16 :=
by
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  sorry

end book_selection_l212_212709


namespace sum_nat_numbers_l212_212722

/-- 
If S is the set of all natural numbers n such that 0 ≤ n ≤ 200, n ≡ 7 [MOD 11], 
and n ≡ 5 [MOD 7], then the sum of elements in S is 351.
-/
theorem sum_nat_numbers (S : Finset ℕ) 
  (hs : ∀ n, n ∈ S ↔ n ≤ 200 ∧ n % 11 = 7 ∧ n % 7 = 5) 
  : S.sum id = 351 := 
sorry 

end sum_nat_numbers_l212_212722


namespace jim_gold_per_hour_l212_212674

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end jim_gold_per_hour_l212_212674


namespace quadratic_point_value_l212_212029

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h_min : ∀ x : ℝ, a * x^2 + b * x + c ≥ a * (-1)^2 + b * (-1) + c) 
  (h_at_min : a * (-1)^2 + b * (-1) + c = -3)
  (h_point : a * (1)^2 + b * (1) + c = 7) : 
  a * (3)^2 + b * (3) + c = 37 :=
sorry

end quadratic_point_value_l212_212029


namespace x_equals_y_l212_212251

-- Conditions
def x := 2 * 20212021 * 1011 * 202320232023
def y := 43 * 47 * 20232023 * 202220222022

-- Proof statement
theorem x_equals_y : x = y := sorry

end x_equals_y_l212_212251


namespace systematic_sampling_first_two_numbers_l212_212892

theorem systematic_sampling_first_two_numbers
  (sample_size : ℕ) (population_size : ℕ) (last_sample_number : ℕ)
  (h1 : sample_size = 50) (h2 : population_size = 8000) (h3 : last_sample_number = 7900) :
  ∃ first second : ℕ, first = 60 ∧ second = 220 :=
by
  -- Proof to be provided.
  sorry

end systematic_sampling_first_two_numbers_l212_212892


namespace max_knights_adjacent_to_two_other_knights_l212_212238

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l212_212238


namespace log_lt_x_squared_for_x_gt_zero_l212_212656

theorem log_lt_x_squared_for_x_gt_zero (x : ℝ) (h : x > 0) : Real.log (1 + x) < x^2 :=
sorry

end log_lt_x_squared_for_x_gt_zero_l212_212656


namespace pie_crusts_flour_l212_212864

theorem pie_crusts_flour (initial_crusts : ℕ)
  (initial_flour_per_crust : ℚ)
  (new_crusts : ℕ)
  (total_flour : ℚ)
  (h1 : initial_crusts = 40)
  (h2 : initial_flour_per_crust = 1/8)
  (h3 : new_crusts = 25)
  (h4 : total_flour = initial_crusts * initial_flour_per_crust) :
  (new_crusts * (total_flour / new_crusts) = total_flour) :=
by
  sorry

end pie_crusts_flour_l212_212864


namespace contrapositive_of_x_squared_gt_1_l212_212345

theorem contrapositive_of_x_squared_gt_1 (x : ℝ) (h : x ≤ 1) : x^2 ≤ 1 :=
sorry

end contrapositive_of_x_squared_gt_1_l212_212345


namespace initial_population_l212_212475

theorem initial_population (P : ℝ) (h : 0.72 * P = 3168) : P = 4400 :=
sorry

end initial_population_l212_212475


namespace london_to_baglmintster_distance_l212_212708

variable (D : ℕ) -- distance from London to Baglmintster

-- Conditions
def meeting_point_condition_1 := D ≥ 40
def meeting_point_condition_2 := D ≥ 48
def initial_meeting := D - 40
def return_meeting := D - 48

theorem london_to_baglmintster_distance :
  (D - 40) + 48 = D + 8 ∧ 40 + (D - 48) = D - 8 → D = 72 :=
by
  intros h
  sorry

end london_to_baglmintster_distance_l212_212708


namespace maciek_total_cost_l212_212067

-- Define the cost of pretzels and the additional cost percentage for chips
def cost_pretzel : ℝ := 4
def cost_chip := cost_pretzel + (cost_pretzel * 0.75)

-- Number of packets Maciek bought for pretzels and chips
def num_pretzels : ℕ := 2
def num_chips : ℕ := 2

-- Total cost calculation
def total_cost := (cost_pretzel * num_pretzels) + (cost_chip * num_chips)

-- The final theorem statement
theorem maciek_total_cost :
  total_cost = 22 := by
  sorry

end maciek_total_cost_l212_212067


namespace min_pairs_with_same_sum_l212_212632

theorem min_pairs_with_same_sum (n : ℕ) (h1 : n > 0) :
  (∀ weights : Fin n → ℕ, (∀ i, weights i ≤ 21) → (∃ i j k l : Fin n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    weights i + weights j = weights k + weights l)) ↔ n ≥ 8 :=
by
  sorry

end min_pairs_with_same_sum_l212_212632


namespace throwing_skips_l212_212778

theorem throwing_skips :
  ∃ x y : ℕ, 
  y > x ∧ 
  (∃ z : ℕ, z = 2 * y ∧ 
  (∃ w : ℕ, w = z - 3 ∧ 
  (∃ u : ℕ, u = w + 1 ∧ u = 8))) ∧ 
  x + y + 2 * y + (2 * y - 3) + (2 * y - 2) = 33 ∧ 
  y - x = 2 :=
sorry

end throwing_skips_l212_212778


namespace find_angle_x_l212_212897

theorem find_angle_x (x : ℝ) (h1 : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_angle_x_l212_212897


namespace inequality_proof_l212_212275

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (2 * a^2) / (1 + a + a * b)^2 + (2 * b^2) / (1 + b + b * c)^2 + (2 * c^2) / (1 + c + c * a)^2 +
  9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a)) ≥ 1 :=
by {
  sorry -- The proof goes here
}

end inequality_proof_l212_212275


namespace basketball_game_points_l212_212663

variable (J T K : ℕ)

theorem basketball_game_points (h1 : T = J + 20) (h2 : J + T + K = 100) (h3 : T = 30) : 
  T / K = 1 / 2 :=
by sorry

end basketball_game_points_l212_212663


namespace f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l212_212519
open Real

noncomputable def f : ℝ → ℝ := sorry

theorem f_additive (a b : ℝ) : f (a + b) = f a + f b := sorry
theorem f_positive_lt_x_zero (x : ℝ) (h_pos : 0 < x) : f x < 0 := sorry
theorem f_at_one : f 1 = 1 := sorry

-- Prove that f is an odd function
theorem f_odd (x : ℝ) : f (-x) = -f x :=
  sorry

-- Solve the inequality: f((log2 x)^2 - log2 (x^2)) > 3
theorem f_inequality (x : ℝ) (h_pos : 0 < x) : (f ((log x / log 2)^2 - (log x^2 / log 2))) > 3 ↔ 1 / 2 < x ∧ x < 8 :=
  sorry

end f_additive_f_positive_lt_x_zero_f_at_one_f_odd_f_inequality_l212_212519


namespace find_multiplicand_l212_212933

theorem find_multiplicand (m : ℕ) 
( h : 32519 * m = 325027405 ) : 
m = 9995 := 
by {
  sorry
}

end find_multiplicand_l212_212933


namespace work_together_days_l212_212362

theorem work_together_days (A B : ℝ) (h1 : A = 1/2 * B) (h2 : B = 1/48) :
  1 / (A + B) = 32 :=
by
  sorry

end work_together_days_l212_212362


namespace min_y_value_l212_212267

theorem min_y_value (x : ℝ) : 
  (∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 ∧ (y = 12 ↔ x = -1)) :=
sorry

end min_y_value_l212_212267


namespace range_of_a_l212_212116

noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x) + log (x + sqrt (x^2 + 1))

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → f (x^2 + 2) + f (-2 * a * x) ≥ 0) → a ≤ 3 / 2 :=
begin
  sorry
end

end range_of_a_l212_212116


namespace granger_bought_4_loaves_of_bread_l212_212288

-- Define the prices of items
def price_of_spam : Nat := 3
def price_of_pb : Nat := 5
def price_of_bread : Nat := 2

-- Define the quantities bought by Granger
def qty_spam : Nat := 12
def qty_pb : Nat := 3
def total_amount_paid : Nat := 59

-- The problem statement in Lean: Prove the number of loaves of bread bought
theorem granger_bought_4_loaves_of_bread :
  (qty_spam * price_of_spam) + (qty_pb * price_of_pb) + (4 * price_of_bread) = total_amount_paid :=
sorry

end granger_bought_4_loaves_of_bread_l212_212288


namespace product_of_powers_eq_nine_l212_212739

variable (a : ℕ)

theorem product_of_powers_eq_nine : a^3 * a^6 = a^9 := 
by sorry

end product_of_powers_eq_nine_l212_212739


namespace solve_inequality_l212_212878

theorem solve_inequality :
  ∀ x : ℝ, (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by
  sorry

end solve_inequality_l212_212878


namespace square_of_ratio_is_specified_value_l212_212140

theorem square_of_ratio_is_specified_value (a b c : ℝ) (h1 : c = Real.sqrt (a^2 + b^2)) (h2 : a / b = b / c) :
  (a / b)^2 = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end square_of_ratio_is_specified_value_l212_212140


namespace product_plus_one_is_square_l212_212128

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end product_plus_one_is_square_l212_212128


namespace max_knights_between_knights_l212_212227

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l212_212227


namespace negation_of_p_l212_212527

def p : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - x ≥ 0

theorem negation_of_p : ¬ p ↔ ∃ x : ℝ, x ≥ 0 ∧ x^2 - x < 0 :=
by
  sorry

end negation_of_p_l212_212527


namespace eval_expression_l212_212638

theorem eval_expression : 68 + (156 / 12) + (11 * 19) - 250 - (450 / 9) = -10 := 
by
  sorry

end eval_expression_l212_212638


namespace largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l212_212772

def is_prime (n : ℕ) : Prop := sorry -- Use inbuilt primality function or define it

def expression (n : ℕ) : ℕ := 2^n + n^2 - 1

theorem largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100 :
  ∃ m, is_prime m ∧ (∃ n, is_prime n ∧ expression n = m ∧ m < 100) ∧
        ∀ k, is_prime k ∧ (∃ n, is_prime n ∧ expression n = k ∧ k < 100) → k <= m :=
  sorry

end largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l212_212772


namespace product_of_0_25_and_0_75_is_0_1875_l212_212501

noncomputable def product_of_decimals : ℝ := 0.25 * 0.75

theorem product_of_0_25_and_0_75_is_0_1875 :
  product_of_decimals = 0.1875 :=
by
  sorry

end product_of_0_25_and_0_75_is_0_1875_l212_212501


namespace fresh_grapes_weight_l212_212515

theorem fresh_grapes_weight :
  ∀ (F : ℝ), (∀ (water_content_fresh : ℝ) (water_content_dried : ℝ) (weight_dried : ℝ),
    water_content_fresh = 0.90 → water_content_dried = 0.20 → weight_dried = 3.125 →
    (F * 0.10 = 0.80 * weight_dried) → F = 78.125) := 
by
  intros F
  intros water_content_fresh water_content_dried weight_dried
  intros h1 h2 h3 h4
  sorry

end fresh_grapes_weight_l212_212515


namespace sum_of_legs_le_sqrt2_hypotenuse_l212_212696

theorem sum_of_legs_le_sqrt2_hypotenuse
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2) :
  a + b ≤ Real.sqrt 2 * c :=
sorry

end sum_of_legs_le_sqrt2_hypotenuse_l212_212696


namespace sasha_work_fraction_l212_212437

theorem sasha_work_fraction :
  let sasha_first := 1 / 3
  let sasha_second := 1 / 5
  let sasha_third := 1 / 15
  let total_sasha_contribution := sasha_first + sasha_second + sasha_third
  let fraction_per_car := total_sasha_contribution / 3
  fraction_per_car = 1 / 5 :=
by
  sorry

end sasha_work_fraction_l212_212437


namespace max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l212_212052

-- Definitions based on the given conditions
variables (a b : ℕ) (h : a > b)

-- Proving the maximum possible range equals a
theorem max_range_eq_a : max_range a b = a :=
by sorry

-- Proving the minimum possible range equals a - b
theorem min_range_eq_a_minus_b : min_range a b = a - b :=
by sorry

-- Proving the number of sequences resulting in the maximum range equals b + 1
theorem num_sequences_max_range_eq_b_plus_1 : num_sequences_max_range a b = b + 1 :=
by sorry

end max_range_eq_a_min_range_eq_a_minus_b_num_sequences_max_range_eq_b_plus_1_l212_212052


namespace symmetric_circle_equation_l212_212327

theorem symmetric_circle_equation (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 5 → (x + 1)^2 + (y - 2)^2 = 5 :=
by
  sorry

end symmetric_circle_equation_l212_212327


namespace complex_powers_l212_212508

theorem complex_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^(23 : ℕ) + i^(58 : ℕ) = -1 - i :=
by sorry

end complex_powers_l212_212508


namespace units_digit_of_sum_l212_212737

theorem units_digit_of_sum (a b : ℕ) : 
  (35^87 + 3^45) % 10 = 8 := 
by
  have units_digit_35 := (5 : ℕ)
  have units_digit_3_power_cycle := [3, 9, 7, 1]
  have remainder_45 := 45 % 4
  have units_digit_3_pow_45 := units_digit_3_power_cycle.nth_le remainder_45 sorry
  have add_units_digits := (units_digit_35 + units_digit_3_pow_45) % 10
  exact add_units_digits = 8

end units_digit_of_sum_l212_212737


namespace sum_of_digits_is_15_l212_212301

theorem sum_of_digits_is_15
  (A B C D E : ℕ) 
  (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_digits: A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (h_divisible_by_9: (A * 10000 + B * 1000 + C * 100 + D * 10 + E) % 9 = 0) 
  : A + B + C + D + E = 15 := 
sorry

end sum_of_digits_is_15_l212_212301


namespace max_seq_value_l212_212524

def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + a m

variables (a : ℕ → ℤ)
variables (S : ℕ → ℤ)

axiom distinct_terms (h : is_arithmetic_seq a) : ∀ n m, n ≠ m → a n ≠ a m
axiom condition_1 : ∀ n, a (2 * n) = 2 * a n - 3
axiom condition_2 : a 6 * a 6 = a 1 * a 21
axiom sum_of_first_n_terms : ∀ n, S n = n * (n + 4)

noncomputable def seq (n : ℕ) : ℤ := S n / 2^(n - 1)

theorem max_seq_value : 
  (∀ n, seq n >= seq (n - 1) ∧ seq n >= seq (n + 1)) → 
  (∃ n, seq n = 6) :=
sorry

end max_seq_value_l212_212524


namespace sequence_value_l212_212937

noncomputable def f : ℝ → ℝ := sorry

theorem sequence_value :
  ∃ a : ℕ → ℝ, 
    (a 1 = f 1) ∧ 
    (∀ n : ℕ, f (a (n + 1)) = f (2 * a n + 1)) ∧ 
    (a 2017 = 2 ^ 2016 - 1) := sorry

end sequence_value_l212_212937


namespace number_of_segments_before_returning_to_start_l212_212880

-- Definitions based on the conditions
def concentric_circles (r R : ℝ) (h_circle : r < R) : Prop := true

def tangent_chord (circle1 circle2 : Prop) (A B : Point) : Prop := 
  circle1 ∧ circle2

def angle_ABC_eq_60 (A B C : Point) (angle_ABC : ℝ) : Prop :=
  angle_ABC = 60

noncomputable def number_of_segments (n : ℕ) (m : ℕ) : Prop := 
  120 * n = 360 * m

theorem number_of_segments_before_returning_to_start (r R : ℝ)
  (h_circle : r < R)
  (circle1 circle2 : Prop := concentric_circles r R h_circle)
  (A B C : Point)
  (h_tangent : tangent_chord circle1 circle2 A B)
  (angle_ABC : ℝ := 0)
  (h_ABC_eq_60 : angle_ABC_eq_60 A B C angle_ABC) :
  ∃ n : ℕ, number_of_segments n 1 ∧ n = 3 := by
  sorry

end number_of_segments_before_returning_to_start_l212_212880


namespace candy_distribution_l212_212928

/-! 
    We want to prove that there are exactly 2187 ways to distribute 8 distinct pieces of candy 
    into three bags (red, blue, and white) such that each bag contains at least one piece of candy.
-/

theorem candy_distribution : 
  ∑ r in finset.range (8).filter (λ r, 1 ≤ r ∧ r ≤ 6), 
  ∑ b in finset.range (8 - r).filter (λ b, 1 ≤ b ∧ b ≤ 7 - r),
  nat.choose 8 r * nat.choose (8 - r) b = 2187 :=
by sorry

end candy_distribution_l212_212928


namespace inscribed_square_ratios_l212_212139

theorem inscribed_square_ratios (a b c x y : ℝ) (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sides : a^2 + b^2 = c^2) 
  (h_leg_square : x = a) 
  (h_hyp_square : y = 5 / 18 * c) : 
  x / y = 18 / 13 := by
  sorry

end inscribed_square_ratios_l212_212139


namespace A_inter_B_eq_A_union_C_U_B_eq_l212_212564

section
  -- Define the universal set U
  def U : Set ℝ := { x | x^2 - (5 / 2) * x + 1 ≥ 0 }

  -- Define set A
  def A : Set ℝ := { x | |x - 1| > 1 }

  -- Define set B
  def B : Set ℝ := { x | (x + 1) / (x - 2) ≥ 0 }

  -- Define the complement of B in U
  def C_U_B : Set ℝ := U \ B

  -- Theorem for A ∩ B
  theorem A_inter_B_eq : A ∩ B = { x | x ≤ -1 ∨ x > 2 } := sorry

  -- Theorem for A ∪ (C_U_B)
  theorem A_union_C_U_B_eq : A ∪ C_U_B = U := sorry
end

end A_inter_B_eq_A_union_C_U_B_eq_l212_212564


namespace rectangle_circle_area_ratio_l212_212753

theorem rectangle_circle_area_ratio {d : ℝ} (h : d > 0) :
  let A_rectangle := 2 * d * d
  let A_circle := (π * d^2) / 4
  (A_rectangle / A_circle) = (8 / π) :=
by
  sorry

end rectangle_circle_area_ratio_l212_212753


namespace hayley_initial_meatballs_l212_212828

theorem hayley_initial_meatballs (x : ℕ) (stolen : ℕ) (left : ℕ) (h1 : stolen = 14) (h2 : left = 11) (h3 : x - stolen = left) : x = 25 := 
by 
  sorry

end hayley_initial_meatballs_l212_212828


namespace inequality_sqrt_sum_l212_212156

theorem inequality_sqrt_sum (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_sum_l212_212156


namespace remainder_of_5n_mod_11_l212_212132

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := 
by
  sorry

end remainder_of_5n_mod_11_l212_212132


namespace partI_partII_l212_212947

-- Define the absolute value function
def f (x : ℝ) := |x - 1|

-- Part I: Solve the inequality f(x) - f(x+2) < 1
theorem partI (x : ℝ) (h : f x - f (x + 2) < 1) : x > -1 / 2 := 
sorry

-- Part II: Find the range of values for a such that x - f(x + 1 - a) ≤ 1 for all x in [1,2]
theorem partII (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x - f (x + 1 - a) ≤ 1) : a ≤ 1 ∨ a ≥ 3 := 
sorry

end partI_partII_l212_212947


namespace set_of_a_l212_212296

theorem set_of_a (a : ℝ) :
  (∃ x : ℝ, a * x ^ 2 + a * x + 1 = 0) → -- Set A contains elements
  (a ≠ 0 ∧ a ^ 2 - 4 * a = 0) →           -- Conditions a ≠ 0 and Δ = 0
  a = 4 := 
sorry

end set_of_a_l212_212296


namespace range_of_a_iff_l212_212809

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x → (Real.log x / Real.log a) ≤ x ∧ x ≤ a ^ x

theorem range_of_a_iff (a : ℝ) : (a ≥ Real.exp (Real.exp (-1))) ↔ range_of_a a :=
by
  sorry

end range_of_a_iff_l212_212809


namespace cost_price_of_ball_l212_212988

theorem cost_price_of_ball (x : ℝ) (h : 17 * x - 5 * x = 720) : x = 60 :=
by {
  sorry
}

end cost_price_of_ball_l212_212988


namespace multiplication_sequence_result_l212_212630

theorem multiplication_sequence_result : (1 * 3 * 5 * 7 * 9 * 11 = 10395) :=
by
  sorry

end multiplication_sequence_result_l212_212630


namespace extension_even_function_l212_212689

noncomputable def g (x : ℝ) : ℝ := 
if x ≤ 0 then 2^x else 2^(-x)

theorem extension_even_function (f : ℝ → ℝ) (D_f D_g : set ℝ) (g : ℝ → ℝ)
  (h1 : D_f ⊆ D_g)
  (h2 : ∀ x ∈ D_f, g x = f x)
  (h3 : ∀ x ≤ 0, f x = 2^x)
  (h4 : ∀ x, g x = g (-x)) :
  ∀ x, g x = 2^(-|x|) := 
by
  intros x
  have h5: g(x) = 2^x ∨ g(x) = 2^(-x) := by sorry
  cases h5 with h5left h5right
  · rw h5left
    sorry
  · rw h5right
    sorry

end extension_even_function_l212_212689


namespace abs_neg_five_l212_212467

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l212_212467


namespace triangle_intersect_sum_l212_212303

theorem triangle_intersect_sum (P Q R S T U : ℝ × ℝ) :
  P = (0, 8) →
  Q = (0, 0) →
  R = (10, 0) →
  S = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) →
  T = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  ∃ U : ℝ × ℝ, 
    (U.1 = (P.1 + ((T.2 - P.2) / (T.1 - P.1)) * (U.1 - P.1)) ∧
     U.2 = (R.2 + ((S.2 - R.2) / (S.1 - R.1)) * (U.1 - R.1))) ∧
    (U.1 + U.2) = 6 :=
by
  sorry

end triangle_intersect_sum_l212_212303


namespace blipblish_modulo_l212_212855

-- Definitions from the conditions
inductive Letter
| B | I | L

def is_consonant (c : Letter) : Bool :=
  match c with
  | Letter.B | Letter.L => true
  | _ => false

def is_vowel (v : Letter) : Bool :=
  match v with
  | Letter.I => true
  | _ => false

def is_valid_blipblish_word (word : List Letter) : Bool :=
  -- Check if between any two I's there at least three consonants
  let rec check (lst : List Letter) (cnt : Nat) (during_vowels : Bool) : Bool :=
    match lst with
    | [] => true
    | Letter.I :: xs =>
        if during_vowels then cnt >= 3 && check xs 0 false
        else check xs 0 true
    | x :: xs =>
        if is_consonant x then check xs (cnt + 1) during_vowels
        else check xs cnt during_vowels
  check word 0 false

def number_of_valid_words (n : Nat) : Nat :=
  -- Placeholder function to compute the number of valid Blipblish words of length n
  sorry

-- Statement of the proof problem
theorem blipblish_modulo : number_of_valid_words 12 % 1000 = 312 :=
by sorry

end blipblish_modulo_l212_212855


namespace total_votes_l212_212319

noncomputable def total_votes_proof : Prop :=
  ∃ T A : ℝ, 
    A = 0.40 * T ∧ 
    T = A + (A + 70) ∧ 
    T = 350

theorem total_votes : total_votes_proof :=
sorry

end total_votes_l212_212319


namespace gift_sequence_count_l212_212135

noncomputable def number_of_gift_sequences (students : ℕ) (classes_per_week : ℕ) : ℕ :=
  (students * students) ^ classes_per_week

theorem gift_sequence_count :
  number_of_gift_sequences 15 3 = 11390625 :=
by
  sorry

end gift_sequence_count_l212_212135


namespace avg_mark_excluded_students_l212_212883

-- Define the given conditions
variables (n : ℕ) (A A_remaining : ℕ) (excluded_count : ℕ)
variable (T : ℕ := n * A)
variable (T_remaining : ℕ := (n - excluded_count) * A_remaining)
variable (T_excluded : ℕ := T - T_remaining)

-- Define the problem statement
theorem avg_mark_excluded_students (h1: n = 14) (h2: A = 65) (h3: A_remaining = 90) (h4: excluded_count = 5) :
   T_excluded / excluded_count = 20 :=
by
  sorry

end avg_mark_excluded_students_l212_212883


namespace choose_4_from_7_socks_l212_212701

theorem choose_4_from_7_socks :
  (nat.choose 7 4) = 35 :=
by 
-- Proof skipped
sorry

end choose_4_from_7_socks_l212_212701


namespace initial_boys_count_l212_212593

theorem initial_boys_count (b : ℕ) (h1 : b + 10 - 4 - 3 = 17) : b = 14 :=
by
  sorry

end initial_boys_count_l212_212593


namespace total_bugs_eaten_l212_212306

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l212_212306


namespace deductive_reasoning_example_is_A_l212_212080

def isDeductive (statement : String) : Prop := sorry

-- Define conditions
def optionA : String := "Since y = 2^x is an exponential function, the function y = 2^x passes through the fixed point (0,1)"
def optionB : String := "Guessing the general formula for the sequence 1/(1×2), 1/(2×3), 1/(3×4), ... as a_n = 1/(n(n+1)) (n ∈ ℕ⁺)"
def optionC : String := "Drawing an analogy from 'In a plane, two lines perpendicular to the same line are parallel' to infer 'In space, two planes perpendicular to the same plane are parallel'"
def optionD : String := "From the circle's equation in the Cartesian coordinate plane (x-a)² + (y-b)² = r², predict that the equation of a sphere in three-dimensional Cartesian coordinates is (x-a)² + (y-b)² + (z-c)² = r²"

theorem deductive_reasoning_example_is_A : isDeductive optionA :=
by
  sorry

end deductive_reasoning_example_is_A_l212_212080


namespace four_squares_cover_larger_square_l212_212443

structure Square :=
  (side : ℝ) (h_positive : side > 0)

theorem four_squares_cover_larger_square (large small : Square) 
  (h_side_relation: large.side = 2 * small.side) : 
  large.side^2 = 4 * small.side^2 :=
by
  sorry

end four_squares_cover_larger_square_l212_212443


namespace average_of_measurements_l212_212041

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80.0, 79.6, 80.5]

theorem average_of_measurements : (measurements.sum / measurements.length) = 80 := by sorry

end average_of_measurements_l212_212041


namespace inequality_nonempty_solution_set_l212_212098

theorem inequality_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x-3| + |x-4| < a) ↔ a > 1 :=
by
  sorry

end inequality_nonempty_solution_set_l212_212098


namespace average_of_first_12_even_is_13_l212_212600

-- Define the first 12 even numbers
def first_12_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

-- Define the sum of the first 12 even numbers
def sum_first_12_even : ℕ := first_12_even_numbers.sum

-- Define the number of values
def num_vals : ℕ := first_12_even_numbers.length

-- Define the average calculation
def average_first_12_even : ℕ := sum_first_12_even / num_vals

-- The theorem we want to prove
theorem average_of_first_12_even_is_13 : average_first_12_even = 13 := by
  sorry

end average_of_first_12_even_is_13_l212_212600


namespace prank_combinations_l212_212335

theorem prank_combinations :
  let monday := 1
  let tuesday := 4
  let wednesday := 7
  let thursday := 5
  let friday := 1
  (monday * tuesday * wednesday * thursday * friday) = 140 :=
by
  sorry

end prank_combinations_l212_212335


namespace gcd_of_three_digit_palindromes_l212_212045

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 101 * a + 10 * b

theorem gcd_of_three_digit_palindromes :
  ∀ n, is_palindrome n → Nat.gcd n 1 = 1 := by
  sorry

end gcd_of_three_digit_palindromes_l212_212045


namespace solve_for_x_l212_212956

theorem solve_for_x (x : ℕ) (h : 2^12 = 64^x) : x = 2 :=
by {
  sorry
}

end solve_for_x_l212_212956


namespace solve_system_eq_l212_212578

theorem solve_system_eq (x y z : ℝ) :
  (x * y * z / (x + y) = 6 / 5) ∧
  (x * y * z / (y + z) = 2) ∧
  (x * y * z / (z + x) = 3 / 2) ↔
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) := 
by
  -- proof to be provided
  sorry

end solve_system_eq_l212_212578


namespace toby_candies_left_l212_212733

def total_candies : ℕ := 56 + 132 + 8 + 300
def num_cousins : ℕ := 13

theorem toby_candies_left : total_candies % num_cousins = 2 :=
by sorry

end toby_candies_left_l212_212733


namespace mr_desmond_toys_l212_212006

theorem mr_desmond_toys (toys_for_elder : ℕ) (h1 : toys_for_elder = 60)
  (h2 : ∀ (toys_for_younger : ℕ), toys_for_younger = 3 * toys_for_elder) : 
  ∃ (total_toys : ℕ), total_toys = 240 :=
by {
  sorry
}

end mr_desmond_toys_l212_212006


namespace direct_proportion_l212_212898

theorem direct_proportion : 
  ∃ k, (∀ x, y = k * x) ↔ (y = -2 * x) :=
by
  sorry

end direct_proportion_l212_212898


namespace list_price_correct_l212_212622

noncomputable def list_price_satisfied : Prop :=
∃ x : ℝ, 0.25 * (x - 25) + 0.05 * (x - 5) = 0.15 * (x - 15) ∧ x = 28.33

theorem list_price_correct : list_price_satisfied :=
sorry

end list_price_correct_l212_212622


namespace circle_equation_center_at_1_2_passing_through_origin_l212_212175

theorem circle_equation_center_at_1_2_passing_through_origin :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = 5 ∧
                (0 - 1)^2 + (0 - 2)^2 = 5 :=
by
  sorry

end circle_equation_center_at_1_2_passing_through_origin_l212_212175


namespace children_neither_blue_nor_red_is_20_l212_212496

-- Definitions
def num_children : ℕ := 45
def num_adults : ℕ := num_children / 3
def num_adults_blue : ℕ := num_adults / 3
def num_adults_red : ℕ := 4
def num_adults_other_colors : ℕ := num_adults - num_adults_blue - num_adults_red
def num_children_red : ℕ := 15
def num_remaining_children : ℕ := num_children - num_children_red
def num_children_other_colors : ℕ := num_remaining_children / 2
def num_children_blue : ℕ := 2 * num_adults_blue
def num_children_neither_blue_nor_red : ℕ := num_children - num_children_red - num_children_blue

-- Theorem statement
theorem children_neither_blue_nor_red_is_20 : num_children_neither_blue_nor_red = 20 :=
  by
  sorry

end children_neither_blue_nor_red_is_20_l212_212496


namespace value_of_expression_l212_212201

theorem value_of_expression (x y z : ℤ) (h1 : x = -3) (h2 : y = 5) (h3 : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 :=
by
  -- proof goes here
  sorry

end value_of_expression_l212_212201


namespace balls_into_boxes_l212_212839

theorem balls_into_boxes : 
  ∀ (balls boxes : ℕ), (balls = 5) → (boxes = 4) → 
  (count_distributions balls boxes = 68) :=
begin
  intros balls boxes hballs hboxes,
  sorry
end

end balls_into_boxes_l212_212839


namespace p_expression_l212_212400

theorem p_expression (m n p : ℤ) (r1 r2 : ℝ) 
  (h1 : r1 + r2 = m) 
  (h2 : r1 * r2 = n) 
  (h3 : r1^2 + r2^2 = p) : 
  p = m^2 - 2 * n := by
  sorry

end p_expression_l212_212400


namespace cole_avg_speed_back_home_l212_212922

noncomputable def avg_speed_back_home 
  (speed_to_work : ℚ) 
  (total_round_trip_time : ℚ) 
  (time_to_work : ℚ) 
  (time_in_minutes : ℚ) :=
  let time_to_work_hours := time_to_work / time_in_minutes
  let distance_to_work := speed_to_work * time_to_work_hours
  let time_back_home := total_round_trip_time - time_to_work_hours
  distance_to_work / time_back_home

theorem cole_avg_speed_back_home :
  avg_speed_back_home 75 1 (35/60) 60 = 105 := 
by 
  -- The proof is omitted
  sorry

end cole_avg_speed_back_home_l212_212922


namespace function_D_min_value_is_2_l212_212624

noncomputable def function_A (x : ℝ) : ℝ := x + 2
noncomputable def function_B (x : ℝ) : ℝ := Real.sin x + 2
noncomputable def function_C (x : ℝ) : ℝ := abs x + 2
noncomputable def function_D (x : ℝ) : ℝ := x^2 + 1

theorem function_D_min_value_is_2
  (x : ℝ) :
  ∃ x, function_D x = 2 := by
  sorry
 
end function_D_min_value_is_2_l212_212624


namespace fraction_ratio_l212_212633

theorem fraction_ratio (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 2 :=
by
  sorry

end fraction_ratio_l212_212633


namespace simplify_expression_l212_212324

theorem simplify_expression (p q x : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : x > 0) (h₃ : x ≠ 1) :
  (x^(3 / p) - x^(3 / q)) / ((x^(1 / p) + x^(1 / q))^2 - 2 * x^(1 / q) * (x^(1 / q) + x^(1 / p)))
  + x^(1 / p) / (x^((q - p) / (p * q)) + 1) = x^(1 / p) + x^(1 / q) := 
sorry

end simplify_expression_l212_212324


namespace lock_combination_l212_212322

def valid_combination (T I D E b : ℕ) : Prop :=
  (T > 0) ∧ (I > 0) ∧ (D > 0) ∧ (E > 0) ∧
  (T ≠ I) ∧ (T ≠ D) ∧ (T ≠ E) ∧ (I ≠ D) ∧ (I ≠ E) ∧ (D ≠ E) ∧
  (T * b^3 + I * b^2 + D * b + E) + 
  (E * b^3 + D * b^2 + I * b + T) + 
  (T * b^3 + I * b^2 + D * b + E) = 
  (D * b^3 + I * b^2 + E * b + T)

theorem lock_combination : ∃ (T I D E b : ℕ), valid_combination T I D E b ∧ (T * 100 + I * 10 + D = 984) :=
sorry

end lock_combination_l212_212322


namespace sam_initial_watermelons_l212_212994

theorem sam_initial_watermelons (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  -- proof steps would go here
  sorry

end sam_initial_watermelons_l212_212994


namespace cost_of_bananas_l212_212734

-- Definitions of the conditions from the problem
namespace BananasCost

variables (A B : ℝ)

-- Condition equations
def condition1 : Prop := 2 * A + B = 7
def condition2 : Prop := A + B = 5

-- The theorem to prove the cost of a bunch of bananas
theorem cost_of_bananas (h1 : condition1 A B) (h2 : condition2 A B) : B = 3 := 
  sorry

end BananasCost

end cost_of_bananas_l212_212734


namespace total_digits_used_l212_212196

theorem total_digits_used (n : ℕ) (h : n = 2500) : 
  let first_n_even := (finset.range (2 * n + 1)).filter (λ x, x % 2 = 0)
  let count_digits := λ x, if x < 10 then 1 else if x < 100 then 2 else if x < 1000 then 3 else 4
  let total_digits := first_n_even.sum (λ x, count_digits x)
  total_digits = 9444 :=
by sorry

end total_digits_used_l212_212196


namespace max_knights_adjacent_to_two_other_knights_l212_212240

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l212_212240


namespace range_of_sum_l212_212133

theorem range_of_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + b + 3 = a * b) : 
a + b ≥ 6 := 
sorry

end range_of_sum_l212_212133


namespace min_value_expression_l212_212512

theorem min_value_expression :
  ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ (5 * Real.sqrt 6) / 3 :=
by
  sorry

end min_value_expression_l212_212512


namespace circle_tangent_x_axis_at_origin_l212_212134

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0) ↔ (D = 0 ∧ F = 0 ∧ E ≠ 0) :=
sorry

end circle_tangent_x_axis_at_origin_l212_212134


namespace average_age_of_two_women_is_30_l212_212051

-- Given definitions
def avg_age_before_replacement (A : ℝ) := 8 * A
def avg_age_after_increase (A : ℝ) := 8 * (A + 2)
def ages_of_men_replaced := 20 + 24

-- The theorem to prove: the average age of the two women is 30 years
theorem average_age_of_two_women_is_30 (A : ℝ) :
  (avg_age_after_increase A) - (avg_age_before_replacement A) = 16 →
  (ages_of_men_replaced + 16) / 2 = 30 :=
by
  sorry

end average_age_of_two_women_is_30_l212_212051


namespace find_m_l212_212416

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : ∃ s : ℝ, (s = (m + 1 - 4) / (2 - m)) ∧ s = Real.sqrt 5) :
  m = (10 - Real.sqrt 5) / 4 :=
by
  sorry

end find_m_l212_212416


namespace area_of_EFCD_l212_212974

theorem area_of_EFCD (AB CD h : ℝ) (H_AB : AB = 10) (H_CD : CD = 30) (H_h : h = 15) :
  let EF := (AB + CD) / 2
  let h_EFCD := h / 2
  let area_EFCD := (1 / 2) * (CD + EF) * h_EFCD
  area_EFCD = 187.5 :=
by
  intros EF h_EFCD area_EFCD
  sorry

end area_of_EFCD_l212_212974


namespace jellybean_probability_l212_212907

theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let white_jellybeans := 6
  let total_chosen := 4
  let total_combinations := Nat.choose total_jellybeans total_chosen
  let red_combinations := Nat.choose red_jellybeans 3
  let non_red_combinations := Nat.choose (blue_jellybeans + white_jellybeans) 1
  let successful_outcomes := red_combinations * non_red_combinations
  let probability := (successful_outcomes : ℚ) / total_combinations
  probability = 4 / 91 :=
by 
  sorry

end jellybean_probability_l212_212907


namespace mr_green_expects_expected_potatoes_yield_l212_212318

theorem mr_green_expects_expected_potatoes_yield :
  ∀ (length_steps width_steps: ℕ) (step_length yield_per_sqft: ℝ),
  length_steps = 18 →
  width_steps = 25 →
  step_length = 2.5 →
  yield_per_sqft = 0.75 →
  (length_steps * step_length) * (width_steps * step_length) * yield_per_sqft = 2109.375 :=
by
  intros length_steps width_steps step_length yield_per_sqft
  intros h_length_steps h_width_steps h_step_length h_yield_per_sqft
  rw [h_length_steps, h_width_steps, h_step_length, h_yield_per_sqft]
  sorry

end mr_green_expects_expected_potatoes_yield_l212_212318


namespace solution_in_Quadrant_III_l212_212981

theorem solution_in_Quadrant_III {c x y : ℝ} 
    (h1 : x - y = 4) 
    (h2 : c * x + y = 5) 
    (hx : x < 0) 
    (hy : y < 0) : 
    c < -1 := 
sorry

end solution_in_Quadrant_III_l212_212981


namespace vacation_days_in_march_l212_212625

theorem vacation_days_in_march 
  (days_worked : ℕ) 
  (days_worked_to_vacation_days : ℕ) 
  (vacation_days_left : ℕ) 
  (days_in_march : ℕ) 
  (days_in_september : ℕ)
  (h1 : days_worked = 300)
  (h2 : days_worked_to_vacation_days = 10)
  (h3 : vacation_days_left = 15)
  (h4 : days_in_september = 2 * days_in_march)
  (h5 : days_worked / days_worked_to_vacation_days - (days_in_march + days_in_september) = vacation_days_left) 
  : days_in_march = 5 := 
by
  sorry

end vacation_days_in_march_l212_212625


namespace josh_marbles_l212_212424

theorem josh_marbles (initial_marbles lost_marbles remaining_marbles : ℤ) 
  (h1 : initial_marbles = 19) 
  (h2 : lost_marbles = 11) 
  (h3 : remaining_marbles = initial_marbles - lost_marbles) : 
  remaining_marbles = 8 := 
by
  sorry

end josh_marbles_l212_212424


namespace inequality_solution_l212_212256

theorem inequality_solution (x: ℝ) (h1: x ≠ -1) (h2: x ≠ 0) :
  (x-2)/(x+1) + (x-3)/(3*x) ≥ 2 ↔ x ∈ Set.Iic (-3) ∪ Set.Icc (-1) (-1/2) :=
by
  sorry

end inequality_solution_l212_212256


namespace karen_wins_in_race_l212_212351

theorem karen_wins_in_race (w : ℝ) (h1 : w / 45 > 1 / 15) 
    (h2 : 60 * (w / 45 - 1 / 15) = w + 4) : 
    w = 8 / 3 := 
sorry

end karen_wins_in_race_l212_212351


namespace four_digits_sum_l212_212003

theorem four_digits_sum (A B C D : ℕ) 
  (A_neq_B : A ≠ B) (A_neq_C : A ≠ C) (A_neq_D : A ≠ D) 
  (B_neq_C : B ≠ C) (B_neq_D : B ≠ D) 
  (C_neq_D : C ≠ D)
  (digits_A : A ≤ 9) (digits_B : B ≤ 9) (digits_C : C ≤ 9) (digits_D : D ≤ 9)
  (A_lt_B : A < B) 
  (minimize_fraction : ∃ k : ℕ, (A + B) = k ∧ k ≤ (A + B) ∧ (C + D) ≥ (C + D)) :
  C + D = 17 := 
by
  sorry

end four_digits_sum_l212_212003


namespace inequality_correct_l212_212823

variable (a b : ℝ)

theorem inequality_correct (h : a < b) : 2 - a > 2 - b :=
by
  sorry

end inequality_correct_l212_212823


namespace possible_values_of_x_and_factors_l212_212152

theorem possible_values_of_x_and_factors (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x = p^5 ∧ (∀ (d : ℕ), d ∣ x → d = p^0 ∨ d = p^1 ∨ d = p^2 ∨ d = p^3 ∨ d = p^4 ∨ d = p^5) ∧ Nat.divisors x ≠ ∅ ∧ (Nat.divisors x).card = 6 := 
  by 
    sorry

end possible_values_of_x_and_factors_l212_212152


namespace sum_of_squares_eq_1850_l212_212776

-- Assuming definitions for the rates
variables (b j s h : ℕ)

-- Condition from Ed's activity
axiom ed_condition : 3 * b + 4 * j + 2 * s + 3 * h = 120

-- Condition from Sue's activity
axiom sue_condition : 2 * b + 3 * j + 4 * s + 3 * h = 150

-- Sum of squares of biking, jogging, swimming, and hiking rates
def sum_of_squares (b j s h : ℕ) : ℕ := b^2 + j^2 + s^2 + h^2

-- Assertion we want to prove
theorem sum_of_squares_eq_1850 :
  ∃ b j s h : ℕ, 3 * b + 4 * j + 2 * s + 3 * h = 120 ∧ 2 * b + 3 * j + 4 * s + 3 * h = 150 ∧ sum_of_squares b j s h = 1850 :=
by
  sorry

end sum_of_squares_eq_1850_l212_212776


namespace integers_with_factors_13_9_between_200_500_l212_212830

theorem integers_with_factors_13_9_between_200_500 : 
  card {n : ℕ | 200 ≤ n ∧ n ≤ 500 ∧ 13 ∣ n ∧ 9 ∣ n} = 3 :=
by 
  sorry

end integers_with_factors_13_9_between_200_500_l212_212830


namespace jogging_distance_apart_l212_212621

theorem jogging_distance_apart 
  (anna_rate : ℕ) (mark_rate : ℕ) (time_hours : ℕ) :
  anna_rate = (1 / 20) ∧ mark_rate = (3 / 40) ∧ time_hours = 2 → 
  6 + 3 = 9 :=
by
  -- setting up constants and translating conditions into variables
  have anna_distance : ℕ := 6
  have mark_distance : ℕ := 3
  sorry

end jogging_distance_apart_l212_212621


namespace convex_quad_sum_greater_diff_l212_212989

theorem convex_quad_sum_greater_diff (α β γ δ : ℝ) 
    (h_sum : α + β + γ + δ = 360) 
    (h_convex : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180) :
    ∀ (x y z w : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) → (y = α ∨ y = β ∨ y = γ ∨ y = δ) → 
                     (z = α ∨ z = β ∨ z = γ ∨ z = δ) → (w = α ∨ w = β ∨ w = γ ∨ w = δ) 
                     → x + y > |z - w| := 
by
  sorry

end convex_quad_sum_greater_diff_l212_212989


namespace only_real_solution_x_eq_6_l212_212792

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l212_212792


namespace polynomial_root_interval_l212_212771

open Real

theorem polynomial_root_interval (b : ℝ) (x : ℝ) :
  (x^4 + b*x^3 + x^2 + b*x - 1 = 0) → (b ≤ -2 * sqrt 3 ∨ b ≥ 0) :=
sorry

end polynomial_root_interval_l212_212771


namespace smallest_three_digit_multiple_of_13_l212_212192

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 13 = 0) ∧ (∀ k : ℕ, (100 ≤ k) ∧ (k < 1000) ∧ (k % 13 = 0) → n ≤ k) → n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l212_212192


namespace common_ratio_geom_seq_l212_212972

variable {a : ℕ → ℝ} {q : ℝ}

def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

theorem common_ratio_geom_seq (h₁ : a 5 = 1) (h₂ : a 8 = 8) (hq : geom_seq a q) : q = 2 :=
by
  sorry

end common_ratio_geom_seq_l212_212972


namespace geometric_sum_first_8_terms_eq_17_l212_212111

theorem geometric_sum_first_8_terms_eq_17 (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : a 0 + a 1 + a 2 + a 3 = 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17 :=
sorry

end geometric_sum_first_8_terms_eq_17_l212_212111


namespace alpha_values_perpendicular_l212_212101

theorem alpha_values_perpendicular
  (α : ℝ)
  (h1 : α ∈ Set.Ico 0 (2 * Real.pi))
  (h2 : ∀ (x y : ℝ), x * Real.cos α - y - 1 = 0 → x + y * Real.sin α + 1 = 0 → false):
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end alpha_values_perpendicular_l212_212101


namespace total_inches_of_rope_l212_212679

noncomputable def inches_of_rope (last_week_feet : ℕ) (less_feet : ℕ) (feet_to_inches : ℕ → ℕ) : ℕ :=
  let last_week_inches := feet_to_inches last_week_feet
  let this_week_feet := last_week_feet - less_feet
  let this_week_inches := feet_to_inches this_week_feet
  last_week_inches + this_week_inches

theorem total_inches_of_rope 
  (six_feet : ℕ := 6)
  (four_feet_less : ℕ := 4)
  (conversion : ℕ → ℕ := λ feet, feet * 12) :
  inches_of_rope six_feet four_feet_less conversion = 96 := by
  sorry

end total_inches_of_rope_l212_212679


namespace find_k_l212_212731

-- Assume three lines in the form of equations
def line1 (x y k : ℝ) := x + k * y = 0
def line2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
def line3 (x y : ℝ) := x - y - 1 = 0

-- Assume the intersection point exists
def intersection_point (x y : ℝ) := 
  line2 x y ∧ line3 x y

-- The main theorem statement
theorem find_k (k : ℝ) (x y : ℝ) (h : intersection_point x y) : 
  line1 x y k ↔ k = -1/2 := 
sorry

end find_k_l212_212731


namespace emily_toys_l212_212390

theorem emily_toys (initial_toys sold_toys: Nat) (h₀ : initial_toys = 7) (h₁ : sold_toys = 3) : initial_toys - sold_toys = 4 := by
  sorry

end emily_toys_l212_212390


namespace find_a5_l212_212109

variable (a_n : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

theorem find_a5 
  (h1 : is_arithmetic_sequence a_n d)
  (h2 : a_n 3 + a_n 8 = 22)
  (h3 : a_n 6 = 7) :
  a_n 5 = 15 :=
sorry

end find_a5_l212_212109


namespace hyperbola_equation_l212_212945

-- Definition of the ellipse given in the problem
def ellipse (x y : ℝ) := y^2 / 5 + x^2 = 1

-- Definition of the conditions for the hyperbola:
-- 1. The hyperbola shares a common focus with the ellipse.
-- 2. Distance from the focus to the asymptote of the hyperbola is 1.
def hyperbola (x y : ℝ) (c : ℝ) :=
  ∃ a b : ℝ, c = 2 ∧ a^2 + b^2 = c^2 ∧
             (b = 1 ∧ y = if x = 0 then 0 else x * (a / b))

-- The statement we need to prove
theorem hyperbola_equation : 
  (∃ a b : ℝ, ellipse x y ∧ hyperbola x y 2 ∧ b = 1 ∧ a^2 = 3) → 
  (y^2 / 3 - x^2 = 1) :=
sorry

end hyperbola_equation_l212_212945


namespace find_k_l212_212866

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x + 3
def q (k : ℝ) (x : ℝ) : ℝ := k * x + k
def intersection (x y : ℝ) : Prop := y = p x ∧ ∃ k, y = q k x

-- Proof that based on the intersection at (1, 5), k evaluates to 5/2
theorem find_k : ∃ k : ℝ, intersection 1 5 → k = 5 / 2 := by
  sorry

end find_k_l212_212866


namespace largest_integer_value_n_l212_212090

theorem largest_integer_value_n (n : ℤ) : 
  (n^2 - 9 * n + 18 < 0) → n ≤ 5 := sorry

end largest_integer_value_n_l212_212090


namespace knights_max_seated_between_knights_l212_212230

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l212_212230


namespace quadratic_equation_root_condition_l212_212399

theorem quadratic_equation_root_condition (a : ℝ) :
  (∃ x1 x2 : ℝ, (a - 1) * x1^2 - 4 * x1 - 1 = 0 ∧ (a - 1) * x2^2 - 4 * x2 - 1 = 0) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end quadratic_equation_root_condition_l212_212399


namespace elisa_target_amount_l212_212637

def elisa_current_amount : ℕ := 37
def elisa_additional_amount : ℕ := 16

theorem elisa_target_amount : elisa_current_amount + elisa_additional_amount = 53 :=
by
  sorry

end elisa_target_amount_l212_212637


namespace find_numbers_l212_212457

theorem find_numbers (x y z : ℕ) :
  x + y + z = 35 → 
  2 * y = x + z + 1 → 
  y^2 = (x + 3) * z → 
  (x = 15 ∧ y = 12 ∧ z = 8) ∨ (x = 5 ∧ y = 12 ∧ z = 18) :=
by
  sorry

end find_numbers_l212_212457


namespace Sam_weight_l212_212464

theorem Sam_weight :
  ∃ (sam_weight : ℕ), (∀ (tyler_weight : ℕ), (∀ (peter_weight : ℕ), peter_weight = 65 → tyler_weight = 2 * peter_weight → tyler_weight = sam_weight + 25 → sam_weight = 105)) :=
by {
    sorry
}

end Sam_weight_l212_212464


namespace problem_solution_l212_212607

lemma factor_def (m n : ℕ) : n ∣ m ↔ ∃ k, m = n * k := by sorry

def is_true_A : Prop := 4 ∣ 24
def is_true_B : Prop := 19 ∣ 209 ∧ ¬ (19 ∣ 63)
def is_true_C : Prop := ¬ (30 ∣ 90) ∧ ¬ (30 ∣ 65)
def is_true_D : Prop := 11 ∣ 33 ∧ ¬ (11 ∣ 77)
def is_true_E : Prop := 9 ∣ 180

theorem problem_solution : (is_true_A ∧ is_true_B ∧ is_true_E) ∧ ¬(is_true_C) ∧ ¬(is_true_D) :=
  by sorry

end problem_solution_l212_212607


namespace points_on_circle_l212_212811

theorem points_on_circle (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  sorry

end points_on_circle_l212_212811


namespace find_current_l212_212138

noncomputable def V : ℂ := 2 + 3 * Complex.I
noncomputable def Z : ℂ := 2 - 2 * Complex.I

theorem find_current : (V / Z) = (-1 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I := by
  sorry

end find_current_l212_212138


namespace max_knights_between_other_knights_l212_212221

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l212_212221


namespace negation_of_proposition_l212_212962

open Real

theorem negation_of_proposition : (¬ (∀ x : ℝ, cos x ≤ 1)) ↔ (∃ x : ℝ, cos x > 1) := by
  sorry

end negation_of_proposition_l212_212962


namespace fraction_of_remaining_prize_money_each_winner_receives_l212_212355

-- Definitions based on conditions
def total_prize_money : ℕ := 2400
def first_winner_fraction : ℚ := 1 / 3
def each_following_winner_prize : ℕ := 160

-- Calculate the first winner's prize
def first_winner_prize : ℚ := first_winner_fraction * total_prize_money

-- Calculate the remaining prize money after the first winner
def remaining_prize_money : ℚ := total_prize_money - first_winner_prize

-- Calculate the fraction of the remaining prize money that each of the next ten winners will receive
def following_winner_fraction : ℚ := each_following_winner_prize / remaining_prize_money

-- Theorem statement
theorem fraction_of_remaining_prize_money_each_winner_receives :
  following_winner_fraction = 1 / 10 :=
sorry

end fraction_of_remaining_prize_money_each_winner_receives_l212_212355


namespace min_value_expression_l212_212520

theorem min_value_expression (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 4 * a + b = 1) :
  (1 / a) + (4 / b) = 16 := sorry

end min_value_expression_l212_212520


namespace distance_BC_in_circle_l212_212664

theorem distance_BC_in_circle
    (r : ℝ) (A B C : ℝ × ℝ)
    (h_radius : r = 10)
    (h_diameter : dist A B = 2 * r)
    (h_chord : dist A C = 12) :
    dist B C = 16 := by
  sorry

end distance_BC_in_circle_l212_212664


namespace line_ellipse_common_point_l212_212949

theorem line_ellipse_common_point (k : ℝ) (m : ℝ) :
  (∀ (x y : ℝ), y = k * x + 1 →
    (y^2 / m + x^2 / 5 ≤ 1)) ↔ (m ≥ 1 ∧ m ≠ 5) :=
by sorry

end line_ellipse_common_point_l212_212949


namespace CoastalAcademy_absent_percentage_l212_212762

theorem CoastalAcademy_absent_percentage :
  ∀ (total_students boys girls : ℕ) (absent_boys_ratio absent_girls_ratio : ℚ),
    total_students = 120 →
    boys = 70 →
    girls = 50 →
    absent_boys_ratio = 1/7 →
    absent_girls_ratio = 1/5 →
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    absent_percentage = 16.67 :=
  by
    intros total_students boys girls absent_boys_ratio absent_girls_ratio
           h1 h2 h3 h4 h5
    let absent_boys := absent_boys_ratio * boys
    let absent_girls := absent_girls_ratio * girls
    let total_absent := absent_boys + absent_girls
    let absent_percentage := total_absent / total_students * 100
    sorry

end CoastalAcademy_absent_percentage_l212_212762


namespace consecutive_even_sum_l212_212181

theorem consecutive_even_sum : 
  ∃ n : ℕ, 
  (∃ x : ℕ, (∀ i : ℕ, i < n → (2 * i + x = 14 → i = 2) → 
  2 * x + (n - 1) * n = 52) ∧ n = 4) :=
by
  sorry

end consecutive_even_sum_l212_212181


namespace speed_maintained_l212_212900

-- Given conditions:
def distance : ℕ := 324
def original_time : ℕ := 6
def new_time : ℕ := (3 * original_time) / 2

-- Correct answer:
def required_speed : ℕ := 36

-- Lean 4 statement to prove the equivalence:
theorem speed_maintained :
  (distance / new_time) = required_speed :=
sorry

end speed_maintained_l212_212900


namespace trigonometric_identity_l212_212958

variable (α : Real)

theorem trigonometric_identity (h : Real.tan α = Real.sqrt 2) :
  (1/3) * Real.sin α^2 + Real.cos α^2 = 5/9 :=
sorry

end trigonometric_identity_l212_212958


namespace sarah_score_l212_212442

-- Given conditions
variable (s g : ℕ) -- Sarah's score and Greg's score
variable (h1 : s = g + 60) -- Sarah's score is 60 points more than Greg's
variable (h2 : (s + g) / 2 = 130) -- The average of their two scores is 130

-- Proof statement
theorem sarah_score : s = 160 :=
by
  sorry

end sarah_score_l212_212442


namespace paco_cookies_l212_212013

theorem paco_cookies :
  let initial_cookies := 25
  let ate_cookies := 5
  let remaining_cookies_after_eating := initial_cookies - ate_cookies
  let gave_away_cookies := 4
  let remaining_cookies_after_giving := remaining_cookies_after_eating - gave_away_cookies
  let bought_cookies := 3
  let final_cookies := remaining_cookies_after_giving + bought_cookies
  let combined_bought_and_gave_away := gave_away_cookies + bought_cookies
  (ate_cookies - combined_bought_and_gave_away) = -2 :=
by sorry

end paco_cookies_l212_212013


namespace max_ab_value_l212_212540

theorem max_ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_perpendicular : (2 * a - 1) * b = -1) : ab <= 1 / 8 := by
  sorry

end max_ab_value_l212_212540


namespace vacation_books_pair_count_l212_212123

/-- 
Given three distinct mystery novels, three distinct fantasy novels, and three distinct biographies,
we want to prove that the number of possible pairs of books of different genres is 27.
-/

theorem vacation_books_pair_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let total_books := mystery_books + fantasy_books + biography_books
  let pairs := (total_books * (total_books - 3)) / 2
  pairs = 27 := 
by
  sorry

end vacation_books_pair_count_l212_212123


namespace maciek_total_cost_l212_212065

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end maciek_total_cost_l212_212065


namespace total_prize_money_l212_212969

theorem total_prize_money (P1 P2 P3 : ℕ) (d : ℕ) (total : ℕ) 
(h1 : P1 = 2000) (h2 : d = 400) (h3 : P2 = P1 - d) (h4 : P3 = P2 - d) 
(h5 : total = P1 + P2 + P3) : total = 4800 :=
sorry

end total_prize_money_l212_212969


namespace triangle_area_l212_212620

theorem triangle_area (a b c : ℕ) (h : a = 12) (i : b = 16) (j : c = 20) (hc : c * c = a * a + b * b) :
  ∃ (area : ℕ), area = 96 :=
by
  sorry

end triangle_area_l212_212620


namespace internal_diagonal_cubes_l212_212612

-- Define the dimensions of the rectangular solid
def x_dimension : ℕ := 168
def y_dimension : ℕ := 350
def z_dimension : ℕ := 390

-- Define the GCD calculations for the given dimensions
def gcd_xy : ℕ := Nat.gcd x_dimension y_dimension
def gcd_yz : ℕ := Nat.gcd y_dimension z_dimension
def gcd_zx : ℕ := Nat.gcd z_dimension x_dimension
def gcd_xyz : ℕ := Nat.gcd (Nat.gcd x_dimension y_dimension) z_dimension

-- Define a statement that the internal diagonal passes through a certain number of cubes
theorem internal_diagonal_cubes :
  x_dimension + y_dimension + z_dimension - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 880 :=
by
  -- Configuration of conditions and proof skeleton with sorry
  sorry

end internal_diagonal_cubes_l212_212612


namespace value_of_f_neg2011_l212_212536

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem value_of_f_neg2011 (a b : ℝ) (h : f 2011 a b = 10) : f (-2011) a b = -14 := by
  sorry

end value_of_f_neg2011_l212_212536


namespace bus_journey_distance_l212_212207

theorem bus_journey_distance (x : ℝ) (h1 : 0 ≤ x)
  (h2 : 0 ≤ 250 - x)
  (h3 : x / 40 + (250 - x) / 60 = 5.2) :
  x = 124 :=
sorry

end bus_journey_distance_l212_212207


namespace cubic_solution_unique_real_l212_212784

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end cubic_solution_unique_real_l212_212784


namespace chandra_valid_pairings_l212_212085

def valid_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

theorem chandra_valid_pairings : valid_pairings 6 6 = 36 :=
  by sorry

end chandra_valid_pairings_l212_212085


namespace product_of_fractions_l212_212895

theorem product_of_fractions :
  (1 / 5) * (3 / 7) = 3 / 35 :=
sorry

end product_of_fractions_l212_212895


namespace prove_necessary_but_not_sufficient_l212_212602

noncomputable def necessary_but_not_sufficient_condition (m : ℝ) :=
  (∀ x : ℝ, x^2 + 2*x + m > 0) → (m > 0) ∧ ¬ (∀ x : ℝ, x^2 + 2*x + m > 0 → m <= 1)

theorem prove_necessary_but_not_sufficient
    (m : ℝ) :
    necessary_but_not_sufficient_condition m :=
by
  sorry

end prove_necessary_but_not_sufficient_l212_212602


namespace perpendicular_vectors_k_value_l212_212408

theorem perpendicular_vectors_k_value (k : ℝ) (a b: ℝ × ℝ)
  (h_a : a = (-1, 3)) (h_b : b = (1, k)) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end perpendicular_vectors_k_value_l212_212408


namespace max_marks_exam_l212_212867

theorem max_marks_exam (M : ℝ) 
  (h1 : 0.80 * M = 400) :
  M = 500 := 
by
  sorry

end max_marks_exam_l212_212867


namespace minimum_value_a5_a6_l212_212144

-- Defining the arithmetic geometric sequence relational conditions.
def arithmetic_geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (a 4 + a 3 - 2 * a 2 - 2 * a 1 = 6) ∧ (∀ n, a n > 0)

-- The mathematical problem to prove:
theorem minimum_value_a5_a6 (a : ℕ → ℝ) (q : ℝ) (h : arithmetic_geometric_sequence_condition a q) :
  a 5 + a 6 = 48 :=
sorry

end minimum_value_a5_a6_l212_212144


namespace sum_mod_eleven_l212_212707

variable (x y z : ℕ)

theorem sum_mod_eleven (h1 : (x * y * z) % 11 = 3)
                       (h2 : (7 * z) % 11 = 4)
                       (h3 : (9 * y) % 11 = (5 + y) % 11) :
                       (x + y + z) % 11 = 5 :=
sorry

end sum_mod_eleven_l212_212707


namespace full_time_and_year_l212_212987

variable (Total F Y N FY : ℕ)

theorem full_time_and_year (h1 : Total = 130)
                            (h2 : F = 80)
                            (h3 : Y = 100)
                            (h4 : N = 20)
                            (h5 : Total = FY + (F - FY) + (Y - FY) + N) :
    FY = 90 := 
sorry

end full_time_and_year_l212_212987


namespace find_p_series_l212_212932

theorem find_p_series (p : ℝ) (h : 5 + (5 + p) / 5 + (5 + 2 * p) / 5^2 + (5 + 3 * p) / 5^3 + ∑' (n : ℕ), (5 + (n + 1) * p) / 5^(n + 1) = 10) : p = 16 :=
sorry

end find_p_series_l212_212932


namespace boys_laps_eq_27_l212_212973

noncomputable def miles_per_lap : ℝ := 3 / 4
noncomputable def girls_miles : ℝ := 27
noncomputable def girls_extra_laps : ℝ := 9

theorem boys_laps_eq_27 :
  (∃ boys_laps girls_laps : ℝ, 
    girls_laps = girls_miles / miles_per_lap ∧ 
    boys_laps = girls_laps - girls_extra_laps ∧ 
    boys_laps = 27) :=
by
  sorry

end boys_laps_eq_27_l212_212973


namespace flower_count_l212_212875

theorem flower_count (R L T : ℕ) (h1 : R = L + 22) (h2 : R = T - 20) (h3 : L + R + T = 100) : R = 34 :=
by
  sorry

end flower_count_l212_212875


namespace num_children_attended_show_l212_212587

def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_adults : ℕ := 183
def total_revenue : ℕ := 5122

theorem num_children_attended_show : ∃ C : ℕ, (num_adults * ticket_price_adult + C * ticket_price_child = total_revenue) ∧ C = 28 :=
by
  sorry

end num_children_attended_show_l212_212587


namespace perimeter_F_is_18_l212_212451

-- Define the dimensions of the rectangles.
def vertical_rectangle : ℤ × ℤ := (3, 5)
def horizontal_rectangle : ℤ × ℤ := (1, 5)

-- Define the perimeter calculation for a single rectangle.
def perimeter (width_height : ℤ × ℤ) : ℤ :=
  2 * width_height.1 + 2 * width_height.2

-- The overlapping width and height.
def overlap_width : ℤ := 5
def overlap_height : ℤ := 1

-- Perimeter of the letter F.
def perimeter_F : ℤ :=
  perimeter vertical_rectangle + perimeter horizontal_rectangle - 2 * overlap_width

-- Statement to prove.
theorem perimeter_F_is_18 : perimeter_F = 18 := by sorry

end perimeter_F_is_18_l212_212451


namespace part_one_part_two_l212_212537

def f (x a : ℝ) : ℝ :=
  x^2 + a * (abs x) + x 

theorem part_one (x1 x2 a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

theorem part_two (a : ℝ) (ha : 0 ≤ a) (x1 x2 : ℝ) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

end part_one_part_two_l212_212537


namespace solution_set_of_inequality_l212_212331

theorem solution_set_of_inequality :
  { x : ℝ | 2 / (x - 1) ≥ 1 } = { x : ℝ | 1 < x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l212_212331


namespace eleven_billion_in_scientific_notation_l212_212597

-- Definition: "Billion" is 10^9
def billion : ℝ := 10^9

-- Theorem: 11 billion can be represented as 1.1 * 10^10
theorem eleven_billion_in_scientific_notation : 11 * billion = 1.1 * 10^10 := by
  sorry

end eleven_billion_in_scientific_notation_l212_212597


namespace necessary_but_not_sufficient_l212_212644

def mutually_exclusive (A1 A2 : Prop) : Prop := (A1 ∧ A2) → False
def complementary (A1 A2 : Prop) : Prop := (A1 ∨ A2) ∧ ¬(A1 ∧ A2)

theorem necessary_but_not_sufficient {A1 A2 : Prop}: 
  mutually_exclusive A1 A2 → complementary A1 A2 → (¬(mutually_exclusive A1 A2 → complementary A1 A2) ∧ (complementary A1 A2 → mutually_exclusive A1 A2)) := 
  by
    sorry

end necessary_but_not_sufficient_l212_212644


namespace find_a5_div_a7_l212_212551

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {aₙ} is a positive geometric sequence.
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom pos_seq (n : ℕ) : 0 < a n

-- Given conditions
axiom a2a8_eq_6 : a 2 * a 8 = 6
axiom a4_plus_a6_eq_5 : a 4 + a 6 = 5
axiom decreasing_seq (n : ℕ) : a (n + 1) < a n

theorem find_a5_div_a7 : a 5 / a 7 = 3 / 2 := 
sorry

end find_a5_div_a7_l212_212551


namespace sqrt_mixed_number_simplification_l212_212088

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 :=
by
  sorry

end sqrt_mixed_number_simplification_l212_212088


namespace triangle_area_of_tangent_circles_l212_212891

/-- 
Given three circles with radii 1, 3, and 5, that are mutually externally tangent and all tangent to 
the same line, the area of the triangle determined by the points where each circle is tangent to the line 
is 6.
-/
theorem triangle_area_of_tangent_circles :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  ∃ (A B C : ℝ × ℝ),
    A = (0, -(r1 : ℝ)) ∧ B = (0, -(r2 : ℝ)) ∧ C = (0, -(r3 : ℝ)) ∧
    (∃ (h : ℝ), ∃ (b : ℝ), h = 4 ∧ b = 3 ∧
    (1 / 2) * h * b = 6) := 
by
  sorry

end triangle_area_of_tangent_circles_l212_212891


namespace number_of_chemistry_books_l212_212743

theorem number_of_chemistry_books:
  ∃ (C : ℕ), (∃ (num_biology_books : ℕ) (ways_to_pick_each : ℕ),
    num_biology_books = 12 ∧ ways_to_pick_each = 1848 ∧
    (num_biology_books.choose 2) * (C.choose 2) = ways_to_pick_each) ∧ C = 8 := 
sorry

end number_of_chemistry_books_l212_212743


namespace four_x_plus_t_odd_l212_212472

theorem four_x_plus_t_odd (x t : ℤ) (hx : 2 * x - t = 11) : ¬(∃ n : ℤ, 4 * x + t = 2 * n) :=
by
  -- Since we need to prove the statement, we start a proof block
  sorry -- skipping the actual proof part for this statement

end four_x_plus_t_odd_l212_212472


namespace winning_prizes_l212_212857

theorem winning_prizes (total_people : ℕ) (percentage_with_envelopes : ℝ) (percentage_with_prizes : ℝ) 
    (h_total : total_people = 100) (h_percent_envelopes : percentage_with_envelopes = 0.40)
    (h_percent_prizes : percentage_with_prizes = 0.20) : 
    (total_people * percentage_with_envelopes * percentage_with_prizes).toNat = 8 :=
  by
    -- Proof omitted
    sorry

end winning_prizes_l212_212857


namespace faster_speed_l212_212365

theorem faster_speed (Speed1 : ℝ) (ExtraDistance : ℝ) (ActualDistance : ℝ) (v : ℝ) : 
  Speed1 = 10 ∧ ExtraDistance = 31 ∧ ActualDistance = 20.67 ∧ 
  (ActualDistance / Speed1 = (ActualDistance + ExtraDistance) / v) → 
  v = 25 :=
by
  sorry

end faster_speed_l212_212365


namespace plot_area_in_acres_l212_212911

theorem plot_area_in_acres :
  let scale_cm_to_miles : ℝ := 3
  let base1_cm : ℝ := 20
  let base2_cm : ℝ := 25
  let height_cm : ℝ := 15
  let miles_to_acres : ℝ := 640
  let area_trapezoid_cm2 := (1 / 2) * (base1_cm + base2_cm) * height_cm
  let area_trapezoid_miles2 := area_trapezoid_cm2 * (scale_cm_to_miles ^ 2)
  let area_trapezoid_acres := area_trapezoid_miles2 * miles_to_acres
  area_trapezoid_acres = 1944000 := by
    sorry

end plot_area_in_acres_l212_212911


namespace perfect_square_of_expression_l212_212849

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem perfect_square_of_expression : 
  (∃ k : ℕ, (factorial 19 * 2 = k ∧ (factorial 20 * factorial 19) / 5 = k * k)) := sorry

end perfect_square_of_expression_l212_212849


namespace sphere_surface_area_quadruple_l212_212659

theorem sphere_surface_area_quadruple (r : ℝ) :
  (4 * π * (2 * r)^2) = 4 * (4 * π * r^2) :=
by
  sorry

end sphere_surface_area_quadruple_l212_212659


namespace consecutive_coeff_sum_l212_212682

theorem consecutive_coeff_sum (P : Polynomial ℕ) (hdeg : P.degree = 699)
  (hP : P.eval 1 ≤ 2022) :
  ∃ k : ℕ, k < 700 ∧ (P.coeff (k + 1) + P.coeff k) = 22 ∨
                    (P.coeff (k + 1) + P.coeff k) = 55 ∨
                    (P.coeff (k + 1) + P.coeff k) = 77 :=
by
  sorry

end consecutive_coeff_sum_l212_212682


namespace boxcar_capacity_ratio_l212_212441

theorem boxcar_capacity_ratio :
  ∀ (total_capacity : ℕ)
    (num_red num_blue num_black : ℕ)
    (black_capacity blue_capacity : ℕ)
    (red_capacity : ℕ),
    num_red = 3 →
    num_blue = 4 →
    num_black = 7 →
    black_capacity = 4000 →
    blue_capacity = 2 * black_capacity →
    total_capacity = 132000 →
    total_capacity = num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity →
    (red_capacity / blue_capacity = 3) :=
by
  intros total_capacity num_red num_blue num_black black_capacity blue_capacity red_capacity
         h_num_red h_num_blue h_num_black h_black_capacity h_blue_capacity h_total_capacity h_combined_capacity
  sorry

end boxcar_capacity_ratio_l212_212441


namespace greatest_of_consecutive_integers_sum_18_l212_212468

theorem greatest_of_consecutive_integers_sum_18 
  (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := 
sorry

end greatest_of_consecutive_integers_sum_18_l212_212468


namespace base6_div_by_7_l212_212099

theorem base6_div_by_7 (k d : ℕ) (hk : 0 ≤ k ∧ k ≤ 5) (hd : 0 ≤ d ∧ d ≤ 5) (hkd : k = d) : 
  7 ∣ (217 * k + 42 * d) := 
by 
  rw [hkd]
  sorry

end base6_div_by_7_l212_212099


namespace distribute_balls_into_boxes_l212_212844

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l212_212844


namespace marys_next_birthday_l212_212145

noncomputable def calculate_marys_age (d j s m TotalAge : ℝ) (H1 : j = 1.15 * d) (H2 : s = 1.30 * d) (H3 : m = 1.25 * s) (H4 : j + d + s + m = TotalAge) : ℝ :=
  m + 1

theorem marys_next_birthday (d j s m TotalAge : ℝ) 
  (H1 : j = 1.15 * d)
  (H2 : s = 1.30 * d)
  (H3 : m = 1.25 * s)
  (H4 : j + d + s + m = TotalAge)
  (H5 : TotalAge = 80) :
  calculate_marys_age d j s m TotalAge H1 H2 H3 H4 = 26 :=
sorry

end marys_next_birthday_l212_212145


namespace probability_of_reaching_boundary_within_5_hops_l212_212100

def position := (ℕ × ℕ)  -- Represents a position on the grid

def transition (pos : position) : Probability (position) := 
  Probability.of_fintype $ [(3, 2), (3, 4), (2, 3), (4, 3)] -- Possible moves after initial move from (3, 3)

def is_target (pos : position) : Prop :=
  pos.snd = 1 ∨ pos.snd = 4 ∨ pos.fst = 1 ∨ pos.fst = 4

def reaches_target_within_5_hops : Probability (position) :=
  -- Sum the probabilities of reaching a target state within 5 hops
  sorry

theorem probability_of_reaching_boundary_within_5_hops : reaches_target_within_5_hops = 15/16 :=
  sorry

end probability_of_reaching_boundary_within_5_hops_l212_212100


namespace trapezoid_reassembly_area_conservation_l212_212354

theorem trapezoid_reassembly_area_conservation
  {height length new_width : ℝ}
  (h1 : height = 9)
  (h2 : length = 16)
  (h3 : new_width = y)  -- each base of the trapezoid measures y.
  (div_trapezoids : ∀ (a b c : ℝ), 3 * a = height → a = 9 / 3)
  (area_conserved : length * height = (3 / 2) * (3 * (length + new_width)))
  : new_width = 16 :=
by
  -- The proof is skipped
  sorry

end trapezoid_reassembly_area_conservation_l212_212354


namespace solution_to_cubic_equation_l212_212789

theorem solution_to_cubic_equation :
  ∀ x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ↔ x = 6 :=
begin
  sorry
end

end solution_to_cubic_equation_l212_212789


namespace quadratic_has_two_distinct_real_roots_l212_212934

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (h1 : 4 + 4 * k > 0) (h2 : k ≠ 0) :
  k > -1 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l212_212934


namespace tan_alpha_frac_l212_212941

theorem tan_alpha_frac (α : ℝ) (h : Real.tan α = 2) : (Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 1 / 11 := by
  sorry

end tan_alpha_frac_l212_212941


namespace exists_xy_l212_212094

open Classical

variable (f : ℝ → ℝ)

theorem exists_xy (h : ∃ x₀ y₀ : ℝ, f x₀ ≠ f y₀) : ∃ x y : ℝ, f (x + y) < f (x * y) :=
by
  sorry

end exists_xy_l212_212094


namespace probability_Y_greater_than_6_l212_212821

noncomputable def X : measure_theory.probability_mass_function ℕ := sorry -- X is a binomial with parameters 3 and p
noncomputable def Y : measure_theory.MeasurableSpace ℝ := sorry -- Y is normal with mean 4 and variance σ^2

axiom E_X_eq_1 : measure_theory.expectation X = 1
axiom P_Y_between_2_and_4 : measure_theory.prob_measure Y (set.Ico 2 4) = (1 : ℝ) / 3

theorem probability_Y_greater_than_6 : 
  let p := (1 : ℝ) / 3 in
  measure_theory.prob_measure Y (set.Ioi 6) = (1 : ℝ) / 6 :=
by
  sorry

end probability_Y_greater_than_6_l212_212821


namespace total_gems_in_chest_l212_212216

theorem total_gems_in_chest (diamonds rubies : ℕ) 
  (h_diamonds : diamonds = 45)
  (h_rubies : rubies = 5110) : 
  diamonds + rubies = 5155 := 
by 
  sorry

end total_gems_in_chest_l212_212216


namespace remainder_when_sum_divided_mod7_l212_212561

theorem remainder_when_sum_divided_mod7 (a b c : ℕ)
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (h7 : a * b * c % 7 = 2)
  (h8 : 3 * c % 7 = 1)
  (h9 : 4 * b % 7 = (2 + b) % 7) :
  (a + b + c) % 7 = 3 := by
  sorry

end remainder_when_sum_divided_mod7_l212_212561


namespace arithmetic_sequence_nth_term_l212_212584

theorem arithmetic_sequence_nth_term (a₁ a₂ a₃ : ℤ) (x : ℤ) (n : ℕ)
  (h₁ : a₁ = 3 * x - 4)
  (h₂ : a₂ = 6 * x - 14)
  (h₃ : a₃ = 4 * x + 3)
  (h₄ : ∀ k : ℕ, a₁ + (k - 1) * ((a₂ - a₁) + (a₃ - a₂) / 2) = 3012) :
  n = 247 :=
by {
  -- Proof to be provided
  sorry
}

end arithmetic_sequence_nth_term_l212_212584


namespace total_number_of_animals_l212_212332

-- Definitions based on conditions
def number_of_females : ℕ := 35
def males_outnumber_females_by : ℕ := 7
def number_of_males : ℕ := number_of_females + males_outnumber_females_by

-- Theorem to prove the total number of animals
theorem total_number_of_animals :
  number_of_females + number_of_males = 77 := by
  sorry

end total_number_of_animals_l212_212332


namespace ellipse_nec_but_not_suff_l212_212352

-- Definitions and conditions
def isEllipse (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c > 0 ∧ ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c

/-- Given that the sum of the distances from a moving point P in the plane to two fixed points is constant,
the condition is necessary but not sufficient for the trajectory of the moving point P being an ellipse. -/
theorem ellipse_nec_but_not_suff (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (c : ℝ) :
  (∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) →
  (c > dist F1 F2 → ¬ isEllipse P F1 F2) ∧ (isEllipse P F1 F2 → ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = c) :=
by
  sorry

end ellipse_nec_but_not_suff_l212_212352


namespace find_real_solutions_l212_212794

noncomputable def cubic_eq_solutions (x : ℝ) : Prop := 
  x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3

theorem find_real_solutions : {x : ℝ | cubic_eq_solutions x} = {6} :=
by
  sorry

end find_real_solutions_l212_212794


namespace cubic_solution_l212_212801

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l212_212801


namespace total_participants_l212_212744

theorem total_participants (freshmen sophomores : ℕ) (h1 : freshmen = 8) (h2 : sophomores = 5 * freshmen) : freshmen + sophomores = 48 := 
by
  sorry

end total_participants_l212_212744


namespace polygon_sides_16_l212_212177

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def arithmetic_sequence_sum (a1 an : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a1 + an) / 2

theorem polygon_sides_16 (n : ℕ) (a1 an : ℝ) (d : ℝ) 
  (h1 : d = 5) (h2 : an = 160) (h3 : a1 = 160 - 5 * (n - 1))
  (h4 : arithmetic_sequence_sum a1 an d n = sum_of_interior_angles n)
  : n = 16 :=
sorry

end polygon_sides_16_l212_212177


namespace construct_quadrilateral_l212_212923

variables (α a b c d : ℝ)

-- α represents the sum of angles B and D
-- a represents the length of AB
-- b represents the length of BC
-- c represents the length of CD
-- d represents the length of DA

theorem construct_quadrilateral (α a b c d : ℝ) : 
  ∃ A B C D : ℝ × ℝ, 
    dist A B = a ∧ 
    dist B C = b ∧ 
    dist C D = c ∧ 
    dist D A = d ∧ 
    ∃ β γ δ, β + δ = α := 
sorry

end construct_quadrilateral_l212_212923


namespace percentage_by_which_x_more_than_y_l212_212965

theorem percentage_by_which_x_more_than_y
    (x y z : ℝ)
    (h1 : y = 1.20 * z)
    (h2 : z = 150)
    (h3 : x + y + z = 555) :
    ((x - y) / y) * 100 = 25 :=
by
  sorry

end percentage_by_which_x_more_than_y_l212_212965


namespace fraction_identity_l212_212453

variable (x y z : ℝ)

theorem fraction_identity (h : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 1) :
  (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 0 :=
  sorry

end fraction_identity_l212_212453


namespace courier_cost_formula_l212_212210

def cost (P : ℕ) : ℕ :=
if P = 0 then 0 else max 50 (30 + 7 * (P - 1))

theorem courier_cost_formula (P : ℕ) : cost P = 
  if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  sorry

end courier_cost_formula_l212_212210


namespace number_of_red_balls_l212_212669

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end number_of_red_balls_l212_212669


namespace max_knights_between_knights_l212_212224

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end max_knights_between_knights_l212_212224


namespace strawberries_left_l212_212370

theorem strawberries_left (picked: ℕ) (eaten: ℕ) (initial_count: picked = 35) (eaten_count: eaten = 2) :
  picked - eaten = 33 :=
by
  sorry

end strawberries_left_l212_212370


namespace rate_of_mixed_oil_l212_212203

/-- If 10 litres of an oil at Rs. 50 per litre is mixed with 5 litres of another oil at Rs. 67 per litre,
    then the rate of the mixed oil per litre is Rs. 55.67. --/
theorem rate_of_mixed_oil : 
  let volume1 := 10
  let price1 := 50
  let volume2 := 5
  let price2 := 67
  let total_cost := (volume1 * price1) + (volume2 * price2)
  let total_volume := volume1 + volume2
  (total_cost / total_volume : ℝ) = 55.67 :=
by
  sorry

end rate_of_mixed_oil_l212_212203


namespace problem_inequality_l212_212535

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_inequality (a x : ℝ) (h : a ∈ Set.Iic (-1/Real.exp 2)) :
  f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) := 
sorry

end problem_inequality_l212_212535


namespace short_video_length_l212_212557

theorem short_video_length 
  (videos_per_day : ℕ) 
  (short_videos_factor : ℕ) 
  (weekly_total_minutes : ℕ) 
  (days_in_week : ℕ) 
  (total_videos : videos_per_day = 3)
  (one_video_longer : short_videos_factor = 6)
  (total_weekly_minutes : weekly_total_minutes = 112)
  (days_a_week : days_in_week = 7) :
  ∃ x : ℕ, (videos_per_day * (short_videos_factor + 2)) * days_in_week = weekly_total_minutes ∧ 
            x = 2 := 
by 
  sorry 

end short_video_length_l212_212557


namespace verify_euler_relation_for_transformed_cube_l212_212935

def euler_relation_for_transformed_cube : Prop :=
  let V := 12
  let A := 24
  let F := 14
  V + F = A + 2

theorem verify_euler_relation_for_transformed_cube :
  euler_relation_for_transformed_cube :=
by
  sorry

end verify_euler_relation_for_transformed_cube_l212_212935


namespace six_digit_number_property_l212_212179

theorem six_digit_number_property :
  ∃ N : ℕ, N = 285714 ∧ (∃ x : ℕ, N = 2 * 10^5 + x ∧ M = 10 * x + 2 ∧ M = 3 * N) :=
by
  sorry

end six_digit_number_property_l212_212179


namespace range_of_g_l212_212773

noncomputable def g (a x : ℝ) : ℝ :=
  a * (Real.cos x)^4 - 2 * (Real.sin x) * (Real.cos x) + (Real.sin x)^4

theorem range_of_g (a : ℝ) (h : a > 0) :
  Set.range (g a) = Set.Icc (a - (3 - a) / 2) (a + (a + 1) / 2) :=
sorry

end range_of_g_l212_212773


namespace age_difference_l212_212548

theorem age_difference (B_age : ℕ) (A_age : ℕ) (X : ℕ) : 
  B_age = 42 → 
  A_age = B_age + 12 → 
  A_age + 10 = 2 * (B_age - X) → 
  X = 10 :=
by
  intros hB_age hA_age hEquation 
  -- define variables based on conditions
  have hB : B_age = 42 := hB_age
  have hA : A_age = B_age + 12 := hA_age
  have hEq : A_age + 10 = 2 * (B_age - X) := hEquation
  -- expected result
  sorry

end age_difference_l212_212548


namespace find_positive_integer_l212_212048

theorem find_positive_integer (n : ℕ) (h1 : 100 % n = 3) (h2 : 197 % n = 3) : n = 97 := 
sorry

end find_positive_integer_l212_212048


namespace knights_max_seated_between_knights_l212_212231

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l212_212231


namespace Razorback_shop_total_revenue_l212_212580

theorem Razorback_shop_total_revenue :
  let Tshirt_price := 62
  let Jersey_price := 99
  let Hat_price := 45
  let Keychain_price := 25
  let Tshirt_sold := 183
  let Jersey_sold := 31
  let Hat_sold := 142
  let Keychain_sold := 215
  let revenue := (Tshirt_price * Tshirt_sold) + (Jersey_price * Jersey_sold) + (Hat_price * Hat_sold) + (Keychain_price * Keychain_sold)
  revenue = 26180 :=
by
  sorry

end Razorback_shop_total_revenue_l212_212580


namespace alberto_bikes_more_l212_212552

-- Definitions of given speeds
def alberto_speed : ℝ := 15
def bjorn_speed : ℝ := 11.25

-- The time duration considered
def time_hours : ℝ := 5

-- Calculate the distances each traveled
def alberto_distance : ℝ := alberto_speed * time_hours
def bjorn_distance : ℝ := bjorn_speed * time_hours

-- Calculate the difference in distances
def distance_difference : ℝ := alberto_distance - bjorn_distance

-- The theorem to be proved
theorem alberto_bikes_more : distance_difference = 18.75 := by
    sorry

end alberto_bikes_more_l212_212552


namespace amount_of_brown_paint_l212_212037

-- Definition of the conditions
def white_paint : ℕ := 20
def green_paint : ℕ := 15
def total_paint : ℕ := 69

-- Theorem statement for the amount of brown paint
theorem amount_of_brown_paint : (total_paint - (white_paint + green_paint)) = 34 :=
by
  sorry

end amount_of_brown_paint_l212_212037


namespace sqrt_205_between_14_and_15_l212_212474

theorem sqrt_205_between_14_and_15 : 14 < Real.sqrt 205 ∧ Real.sqrt 205 < 15 := 
by
  sorry

end sqrt_205_between_14_and_15_l212_212474


namespace ratio_of_pieces_l212_212745

-- Define the total length of the wire.
def total_length : ℕ := 14

-- Define the length of the shorter piece.
def shorter_piece_length : ℕ := 4

-- Define the length of the longer piece.
def longer_piece_length : ℕ := total_length - shorter_piece_length

-- Define the expected ratio of the lengths.
def ratio : ℚ := shorter_piece_length / longer_piece_length

-- State the theorem to prove.
theorem ratio_of_pieces : ratio = 2 / 5 := 
by {
  -- skip the proof
  sorry
}

end ratio_of_pieces_l212_212745


namespace f_monotone_decreasing_without_min_value_l212_212660

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem f_monotone_decreasing_without_min_value :
  (∀ x y : ℝ, x < y → f y < f x) ∧ (∃ b : ℝ, ∀ x : ℝ, f x > b) :=
by
  sorry

end f_monotone_decreasing_without_min_value_l212_212660


namespace square_difference_l212_212009

variable (n : ℕ)

theorem square_difference (n : ℕ) : (n + 1)^2 - n^2 = 2 * n + 1 :=
sorry

end square_difference_l212_212009


namespace max_knights_seated_next_to_two_knights_l212_212245

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l212_212245


namespace maximal_value_of_product_l212_212115

theorem maximal_value_of_product (m n : ℤ)
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (1 < x1 ∧ x1 < 3) ∧ (1 < x2 ∧ x2 < 3) ∧ 
    ∀ x : ℝ, (10 * x^2 + m * x + n) = 10 * (x - x1) * (x - x2)) :
  (∃ f1 f3 : ℝ, f1 = 10 * (1 - x1) * (1 - x2) ∧ f3 = 10 * (3 - x1) * (3 - x2) ∧ (f1 * f3 = 99)) := 
sorry

end maximal_value_of_product_l212_212115


namespace ratio_of_x_intercepts_l212_212186

theorem ratio_of_x_intercepts (b s t : ℝ) (h_b : b ≠ 0)
  (h1 : 0 = 8 * s + b)
  (h2 : 0 = 4 * t + b) :
  s / t = 1 / 2 :=
by
  sorry

end ratio_of_x_intercepts_l212_212186


namespace pandemic_cut_percentage_l212_212750

-- Define the conditions
def initial_planned_production : ℕ := 200
def decrease_due_to_metal_shortage : ℕ := 50
def doors_per_car : ℕ := 5
def total_doors_produced : ℕ := 375

-- Define the quantities after metal shortage and before the pandemic
def production_after_metal_shortage : ℕ := initial_planned_production - decrease_due_to_metal_shortage
def doors_after_metal_shortage : ℕ := production_after_metal_shortage * doors_per_car
def cars_after_pandemic : ℕ := total_doors_produced / doors_per_car
def reduction_in_production : ℕ := production_after_metal_shortage - cars_after_pandemic

-- Define the expected percentage cut
def expected_percentage_cut : ℕ := 50

-- Prove that the percentage of production cut due to the pandemic is as required
theorem pandemic_cut_percentage : (reduction_in_production * 100 / production_after_metal_shortage) = expected_percentage_cut := by
  sorry

end pandemic_cut_percentage_l212_212750


namespace circle_area_is_323pi_l212_212155

-- Define points A and B
def A : ℝ × ℝ := (2, 9)
def B : ℝ × ℝ := (14, 7)

-- Define that points A and B lie on circle ω
def on_circle_omega (A B C : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = r ^ 2 ∧
  (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 = r ^ 2

-- Define the tangent lines intersect at a point on the x-axis
def tangents_intersect_on_x_axis (A B : ℝ × ℝ) (C : ℝ × ℝ) (ω : (ℝ × ℝ) → ℝ): Prop := 
  ∃ x : ℝ, (A.1 - C.1) ^ 2 + (A.2 - C.2) ^ 2 = (B.1 - C.1) ^ 2 + (B.2 - C.2) ^ 2 ∧
             C.2 = 0

-- Problem statement to prove
theorem circle_area_is_323pi (C : ℝ × ℝ) (radius : ℝ) (on_circle_omega : on_circle_omega A B C radius)
  (tangents_intersect_on_x_axis : tangents_intersect_on_x_axis A B C omega) :
  π * radius ^ 2 = 323 * π :=
sorry

end circle_area_is_323pi_l212_212155


namespace quadratic_parabola_equation_l212_212285

theorem quadratic_parabola_equation :
  ∃ (a b c : ℝ), 
    (∀ x y, y = 3 * x^2 - 6 * x + 5 → (x - 1)*(x - 1) = (x - 1)^2) ∧ -- Original vertex condition and standard form
    (∀ x y, y = -x - 2 → a = 2) ∧ -- Given intersection point condition
    (∀ x y, y = -3 * (x - 1)^2 + 2 → y = -3 * (x - 1)^2 + b ∧ y = -4) → -- Vertex unchanged and direction reversed
    (a, b, c) = (-3, 6, -4) := -- Resulting equation coefficients
sorry

end quadratic_parabola_equation_l212_212285


namespace range_of_m_l212_212563

theorem range_of_m 
  (m : ℝ)
  (hM : -4 ≤ m ∧ m ≤ 4)
  (ellipse : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 → y = 0) :
  1 ≤ m ∧ m ≤ 4 := sorry

end range_of_m_l212_212563


namespace average_age_in_club_l212_212966

theorem average_age_in_club (women men children : ℕ) 
    (avg_age_women avg_age_men avg_age_children : ℤ)
    (hw : women = 12) (hm : men = 18) (hc : children = 20)
    (haw : avg_age_women = 32) (ham : avg_age_men = 36) (hac : avg_age_children = 10) :
    (12 * 32 + 18 * 36 + 20 * 10) / (12 + 18 + 20) = 24 := by
  sorry

end average_age_in_club_l212_212966


namespace max_score_per_student_l212_212311

theorem max_score_per_student (score_tests : ℕ → ℕ) (avg_score_tests_lt_8 : ℕ) (combined_score_two_tests : ℕ) : (∀ i, 1 ≤ i ∧ i ≤ 8 → score_tests i ≤ 100) ∧ avg_score_tests_lt_8 = 70 ∧ combined_score_two_tests = 290 →
  ∃ max_score : ℕ, max_score = 145 := 
by
  sorry

end max_score_per_student_l212_212311


namespace parallel_and_through_point_l212_212328

-- Defining the given line
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Defining the target line passing through the point (0, 4)
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the point (0, 4)
def point : ℝ × ℝ := (0, 4)

-- Prove that line2 passes through the point (0, 4) and is parallel to line1
theorem parallel_and_through_point (x y : ℝ) 
  (h1 : line1 x y) 
  : line2 (point.fst) (point.snd) := by
  sorry

end parallel_and_through_point_l212_212328


namespace solution_set_inequality_l212_212642

theorem solution_set_inequality (x : ℝ) : (x^2 - 2*x - 8 ≥ 0) ↔ (x ≤ -2 ∨ x ≥ 4) := 
sorry

end solution_set_inequality_l212_212642


namespace product_plus_one_is_square_l212_212131

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : x * y + 1 = (x + 1) ^ 2 :=
by
  sorry

end product_plus_one_is_square_l212_212131


namespace difference_q_r_l212_212904

-- Conditions
variables (p q r : ℕ) (x : ℕ)
variables (h_ratio : 3 * x = p) (h_ratio2 : 7 * x = q) (h_ratio3 : 12 * x = r)
variables (h_diff_pq : q - p = 3200)

-- Proof problem to solve
theorem difference_q_r : q - p = 3200 → 12 * x - 7 * x = 4000 :=
by 
  intro h_diff_pq
  rw [h_ratio, h_ratio2, h_ratio3] at *
  sorry

end difference_q_r_l212_212904


namespace range_of_b_l212_212149

/-- Let A = {x | -1 < x < 1} and B = {x | b - 1 < x < b + 1}.
    We need to show that if A ∩ B ≠ ∅, then b is within the interval (-2, 2). -/
theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ b - 1 < x ∧ x < b + 1) →
  -2 < b ∧ b < 2 :=
sorry

end range_of_b_l212_212149


namespace geometric_sequence_common_ratio_l212_212971

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 1 / 4) : 
  q = 1 / 2 :=
  sorry

end geometric_sequence_common_ratio_l212_212971


namespace xy_sum_one_l212_212826

theorem xy_sum_one (x y : ℝ) (h : x > 0) (k : y > 0) (hx : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : x + y = 1 :=
sorry

end xy_sum_one_l212_212826


namespace total_points_l212_212142

def points_earned (goblins orcs dragons : ℕ): ℕ :=
  goblins * 3 + orcs * 5 + dragons * 10

theorem total_points :
  points_earned 10 7 1 = 75 :=
by
  sorry

end total_points_l212_212142


namespace real_m_of_complex_product_l212_212963

-- Define the conditions that m is a real number and (m^2 + i)(1 - mi) is a real number
def is_real (z : ℂ) : Prop := z.im = 0
def cplx_eq (m : ℝ) : ℂ := (⟨m^2, 1⟩ : ℂ) * (⟨1, -m⟩ : ℂ)

theorem real_m_of_complex_product (m : ℝ) : is_real (cplx_eq m) ↔ m = 1 :=
by
  sorry

end real_m_of_complex_product_l212_212963


namespace variance_probability_binomial_l212_212185

open ProbabilityTheory

noncomputable def variance_of_binomial (n : ℕ) (ξ : ℕ → ℕ) : ℚ :=
  n * (1 / 2 : ℚ) * (1 - 1 / 2 : ℚ)

theorem variance_probability_binomial :
  ∀ (n : ℕ) (ξ : ℕ → ℕ),
  3 ≤ n ∧ n ≤ 8 ∧
  (∀ k : ℕ, Probability.(binomial n (1 / 2 : ℚ)).pmf k = choose n k * (1 / 2 : ℚ) ^ k * (1 / 2 : ℚ) ^ (n - k)) ∧
  Probability.(binomial n (1 / 2 : ℚ)).pmf 1 = 3 / 32
  → variance_of_binomial n ξ = 3 / 2 :=
by
  sorry

end variance_probability_binomial_l212_212185


namespace range_of_a_l212_212108

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0) 
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l212_212108


namespace problem_l212_212824

variable {x : ℝ}

theorem problem (h : x + 1/x = 5) : x^4 + 1/x^4 = 527 :=
by
  sorry

end problem_l212_212824


namespace only_real_solution_x_eq_6_l212_212793

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l212_212793


namespace balls_into_boxes_l212_212840

theorem balls_into_boxes : 
  ∀ (balls boxes : ℕ), (balls = 5) → (boxes = 4) → 
  (count_distributions balls boxes = 68) :=
begin
  intros balls boxes hballs hboxes,
  sorry
end

end balls_into_boxes_l212_212840


namespace sequence_term_geometric_l212_212521

theorem sequence_term_geometric :
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 →
    (∀ n, n ≥ 2 → (a n) / (a (n - 1)) = 2^(n-1)) →
    a 101 = 2^5050 :=
by
  sorry

end sequence_term_geometric_l212_212521


namespace monotonicity_and_extrema_l212_212534

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → f x < f (x + 0.0001)) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → f x > f (x + 0.0001)) ∧
  (∀ x, -1 / 2 < x ∧ x < (Real.exp 2 - 3) / 2 → f x < f (x + 0.0001)) ∧
  ∀ x, x ∈ Set.Icc (-1 : ℝ) ((Real.exp 2 - 3) / 2) →
     (f (x) ≥ Real.log 2 + 1 / 4 → x = -1 / 2) ∧
     (f (x) ≤ 2 + (Real.exp 2 - 3)^2 / 4 → x = (Real.exp 2 - 3) / 2) :=
sorry

end monotonicity_and_extrema_l212_212534


namespace cubic_solution_l212_212799

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end cubic_solution_l212_212799


namespace frequency_third_group_l212_212305

-- Definitions based on conditions:
variables (S : ℕ → ℝ) (m : ℕ)
variables (h1 : m ≥ 3)
variables (h2 : S 1 + S 2 + S 3 = 1 / 4 * (S 4 + ∑ i in finset.range (m-3), S (i+4)))
variables (h3 : ∑ i in finset.range m, S (i + 1) = 1)
variables (h4 : S 1 = 1 / 20)
variables (h5 : S 1 + S 2 + S 3 = 3 * S 2)
variables (sample_size : ℕ := 120)

-- Goal: The frequency of the third group.
theorem frequency_third_group : 120 * S 3 = 10 :=
by
  sorry

end frequency_third_group_l212_212305


namespace calculate_expression_l212_212768

theorem calculate_expression :
  ( ( (1/6) - (1/8) + (1/9) ) / ( (1/3) - (1/4) + (1/5) ) ) * 3 = 55 / 34 :=
by
  sorry

end calculate_expression_l212_212768


namespace initial_honey_amount_l212_212909

variable (H : ℝ)

theorem initial_honey_amount :
  (0.70 * 0.60 * 0.50) * H = 315 → H = 1500 :=
by
  sorry

end initial_honey_amount_l212_212909


namespace minimum_value_l212_212648

theorem minimum_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x - 2 * y + 3 = 0) : 
  ∃ z : ℝ, z = 3 ∧ (∀ z' : ℝ, (z' = y^2 / x) → z ≤ z') :=
sorry

end minimum_value_l212_212648


namespace pool_water_volume_after_evaporation_l212_212916

theorem pool_water_volume_after_evaporation :
  let initial_volume := 300
  let evaporation_first_15_days := 1 -- in gallons per day
  let evaporation_next_15_days := 2 -- in gallons per day
  initial_volume - (15 * evaporation_first_15_days + 15 * evaporation_next_15_days) = 255 :=
by
  sorry

end pool_water_volume_after_evaporation_l212_212916


namespace balls_into_boxes_l212_212838

theorem balls_into_boxes : 
  ∀ (balls boxes : ℕ), (balls = 5) → (boxes = 4) → 
  (count_distributions balls boxes = 68) :=
begin
  intros balls boxes hballs hboxes,
  sorry
end

end balls_into_boxes_l212_212838


namespace min_third_side_of_right_triangle_l212_212337

theorem min_third_side_of_right_triangle (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ c : ℕ, (min c (4 + 5 - 3) - (4 - 3)) = 3 :=
sorry

end min_third_side_of_right_triangle_l212_212337


namespace oranges_per_box_l212_212494

theorem oranges_per_box
  (total_oranges : ℕ)
  (boxes : ℕ)
  (h1 : total_oranges = 35)
  (h2 : boxes = 7) :
  total_oranges / boxes = 5 := by
  sorry

end oranges_per_box_l212_212494


namespace sqrt_div_l212_212378

theorem sqrt_div (a b : ℝ) (h1 : a = 28) (h2 : b = 7) :
  Real.sqrt a / Real.sqrt b = 2 := 
by 
  sorry

end sqrt_div_l212_212378


namespace find_constants_l212_212817

theorem find_constants (a b : ℝ) (h₀ : ∀ x : ℝ, (x^3 + 3*a*x^2 + b*x + a^2 = 0 → x = -1)) :
    a = 2 ∧ b = 9 :=
by
  sorry

end find_constants_l212_212817


namespace unique_x_condition_l212_212264

theorem unique_x_condition (x : ℝ) : 
  (1 ≤ x ∧ x < 2) ∧ (∀ n : ℕ, 0 < n → (⌊2^n * x⌋ % 4 = 1 ∨ ⌊2^n * x⌋ % 4 = 2)) ↔ x = 4/3 := 
by 
  sorry

end unique_x_condition_l212_212264


namespace find_a_l212_212961

theorem find_a (a b c : ℝ) (h1 : b = 15) (h2 : c = 5)
  (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) 
  (result : a * 15 * 5 = 2) : a = 6 := by 
  sorry

end find_a_l212_212961


namespace equal_cubes_l212_212915

theorem equal_cubes (a : ℤ) : -(a ^ 3) = (-a) ^ 3 :=
by
  sorry

end equal_cubes_l212_212915


namespace M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l212_212571

noncomputable def M : ℕ → ℕ → ℕ → ℝ := sorry

theorem M_less_equal_fraction_M (n k h : ℕ) : 
  M n k h ≤ (n / h) * M (n-1) (k-1) (h-1) :=
sorry

theorem M_greater_equal_fraction_M (n k h : ℕ) : 
  M n k h ≥ (n / (n - h)) * M (n-1) k k :=
sorry

theorem M_less_equal_sum_M (n k h : ℕ) : 
  M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h :=
sorry

end M_less_equal_fraction_M_M_greater_equal_fraction_M_M_less_equal_sum_M_l212_212571


namespace age_30_years_from_now_l212_212925

variables (ElderSonAge : ℕ) (DeclanAgeDiff : ℕ) (YoungerSonAgeDiff : ℕ) (ThirdSiblingAgeDiff : ℕ)

-- Given conditions
def elder_son_age : ℕ := 40
def declan_age : ℕ := elder_son_age + 25
def younger_son_age : ℕ := elder_son_age - 10
def third_sibling_age : ℕ := younger_son_age - 5

-- To prove the ages 30 years from now
def younger_son_age_30_years_from_now : ℕ := younger_son_age + 30
def third_sibling_age_30_years_from_now : ℕ := third_sibling_age + 30

-- The proof statement
theorem age_30_years_from_now : 
  younger_son_age_30_years_from_now = 60 ∧ 
  third_sibling_age_30_years_from_now = 55 :=
by
  sorry

end age_30_years_from_now_l212_212925


namespace peak_valley_usage_l212_212882

-- Define the electricity rate constants
def normal_rate : ℝ := 0.5380
def peak_rate : ℝ := 0.5680
def valley_rate : ℝ := 0.2880

-- Define the total consumption and the savings
def total_consumption : ℝ := 200
def savings : ℝ := 16.4

-- Define the theorem to prove the peak and off-peak usage
theorem peak_valley_usage :
  ∃ (x y : ℝ), x + y = total_consumption ∧ peak_rate * x + valley_rate * y = total_consumption * normal_rate - savings ∧ x = 120 ∧ y = 80 :=
by
  sorry

end peak_valley_usage_l212_212882


namespace max_knights_adjacent_to_two_other_knights_l212_212242

theorem max_knights_adjacent_to_two_other_knights
    (total_knights : ℕ)
    (total_samurais : ℕ)
    (knights_with_samurai_on_right : ℕ)
    (total_people := total_knights + total_samurais)
    (total_knights = 40)
    (total_samurais = 10)
    (knights_with_samurai_on_right = 7) : 
    ∃ max_knights_adjacent : ℕ, max_knights_adjacent = 32 :=
by
  sorry

end max_knights_adjacent_to_two_other_knights_l212_212242


namespace determine_triangle_value_l212_212114

theorem determine_triangle_value (p : ℕ) (triangle : ℕ) (h1 : triangle + p = 67) (h2 : 3 * (triangle + p) - p = 185) : triangle = 51 := by
  sorry

end determine_triangle_value_l212_212114


namespace route_one_speed_is_50_l212_212023

noncomputable def speed_route_one (x : ℝ) : Prop :=
  let time_route_one := 75 / x
  let time_route_two := 90 / (1.8 * x)
  time_route_one = time_route_two + 1/2

theorem route_one_speed_is_50 :
  ∃ x : ℝ, speed_route_one x ∧ x = 50 :=
by
  sorry

end route_one_speed_is_50_l212_212023


namespace plane_eq_of_point_and_parallel_l212_212510

theorem plane_eq_of_point_and_parallel (A B C D : ℤ) 
  (h1 : A = 3) (h2 : B = -2) (h3 : C = 4) 
  (point : ℝ × ℝ × ℝ) (hpoint : point = (2, -3, 5))
  (h4 : 3 * (2 : ℝ) - 2 * (-3 : ℝ) + 4 * (5 : ℝ) + (D : ℝ) = 0)
  (hD : D = -32)
  (hGCD : Int.gcd (Int.natAbs 3) (Int.gcd (Int.natAbs (-2)) (Int.gcd (Int.natAbs 4) (Int.natAbs (-32)))) = 1) : 
  3 * (x : ℝ) - 2 * (y : ℝ) + 4 * (z : ℝ) - 32 = 0 :=
sorry

end plane_eq_of_point_and_parallel_l212_212510


namespace solution_set_abs_inequality_l212_212035

theorem solution_set_abs_inequality : {x : ℝ | |1 - 2 * x| < 3} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end solution_set_abs_inequality_l212_212035


namespace negation_proof_l212_212715

theorem negation_proof :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  -- Proof to be filled
  sorry

end negation_proof_l212_212715


namespace simplify_trig_expression_l212_212575

theorem simplify_trig_expression : 
  (sin (Real.pi / 6) + sin (Real.pi / 3)) / (cos (Real.pi / 6) + cos (Real.pi / 3)) = 1 := by
  sorry

end simplify_trig_expression_l212_212575


namespace regular_polygon_sides_l212_212964

theorem regular_polygon_sides (θ : ℝ) (hθ : θ = 45) : 360 / θ = 8 := by
  sorry

end regular_polygon_sides_l212_212964


namespace minimum_value_xy_l212_212531

theorem minimum_value_xy (x y : ℝ) (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) : x + y ≥ 0 :=
sorry

end minimum_value_xy_l212_212531


namespace quadratic_equation_even_coefficient_l212_212338

-- Define the predicate for a rational root
def has_rational_root (a b c : ℤ) : Prop :=
  ∃ (p q : ℤ), (q ≠ 0) ∧ (p.gcd q = 1) ∧ (a * p^2 + b * p * q + c * q^2 = 0)

-- Define the predicate for at least one being even
def at_least_one_even (a b c : ℤ) : Prop :=
  (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0)

theorem quadratic_equation_even_coefficient 
  (a b c : ℤ) (h_non_zero : a ≠ 0) (h_rational_root : has_rational_root a b c) :
  at_least_one_even a b c :=
sorry

end quadratic_equation_even_coefficient_l212_212338


namespace flash_catches_ace_l212_212913

theorem flash_catches_ace (v : ℝ) (x : ℝ) (y : ℝ) (hx : x > 1) :
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  flash_distance = (xy / (x - 1)) :=
by
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  have h1 : x * v * t = xy / (x - 1) := sorry
  exact h1

end flash_catches_ace_l212_212913


namespace sum_of_roots_l212_212518

variables {a b c : ℝ}

-- Conditions
-- The polynomial with roots a, b, c
def poly (x : ℝ) : ℝ := 24 * x^3 - 36 * x^2 + 14 * x - 1

-- The roots are in (0, 1)
def in_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- All roots are distinct
def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- Main Theorem
theorem sum_of_roots :
  (∀ x, poly x = 0 → x = a ∨ x = b ∨ x = c) →
  in_interval a →
  in_interval b →
  in_interval c →
  distinct a b c →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2) :=
by
  intros
  sorry

end sum_of_roots_l212_212518


namespace bacteria_growth_rate_l212_212613

theorem bacteria_growth_rate (r : ℝ) 
  (h1 : ∀ n : ℕ, n = 22 → ∃ c : ℝ, c * r^n = c) 
  (h2 : ∀ n : ℕ, n = 21 → ∃ c : ℝ, 2 * c * r^n = c) : 
  r = 2 := 
by
  sorry

end bacteria_growth_rate_l212_212613


namespace real_solution_unique_l212_212802

theorem real_solution_unique (x : ℝ) : 
  (x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) ↔ x = 6 := 
begin
  sorry
end

end real_solution_unique_l212_212802


namespace trig_identity_l212_212629

open Real

theorem trig_identity :
  sin (21 * π / 180) * cos (9 * π / 180) + sin (69 * π / 180) * sin (9 * π / 180) = 1 / 2 :=
by sorry

end trig_identity_l212_212629


namespace pounds_of_beef_l212_212553

theorem pounds_of_beef (meals_price : ℝ) (total_sales : ℝ) (meat_per_meal : ℝ) (relationship : ℝ) (total_meat_used : ℝ) (beef_pounds : ℝ) :
  (total_sales = 400) → (meals_price = 20) → (meat_per_meal = 1.5) → (relationship = 0.5) → (20 * meals_price = total_sales) → (total_meat_used = 30) →
  (beef_pounds + beef_pounds * relationship = total_meat_used) → beef_pounds = 20 :=
by
  intros
  sorry

end pounds_of_beef_l212_212553


namespace largest_sampled_number_l212_212273

theorem largest_sampled_number (N : ℕ) (a₁ a₂ : ℕ) (k : ℕ) (H_N : N = 1500)
  (H_a₁ : a₁ = 18) (H_a₂ : a₂ = 68) (H_k : k = a₂ - a₁) :
  ∃ m, m ≤ N ∧ (m % k = 18 % k) ∧ ∀ n, (n % k = 18 % k) → n ≤ N → n ≤ m :=
by {
  -- sorry
  sorry
}

end largest_sampled_number_l212_212273


namespace joshua_miles_ratio_l212_212004

-- Definitions corresponding to conditions
def mitch_macarons : ℕ := 20
def joshua_extra : ℕ := 6
def total_kids : ℕ := 68
def macarons_per_kid : ℕ := 2

-- Variables for unspecified amounts
variable (M : ℕ) -- number of macarons Miles made

-- Calculations for Joshua and Renz's macarons based on given conditions
def joshua_macarons := mitch_macarons + joshua_extra
def renz_macarons := (3 * M) / 4 - 1

-- Total macarons calculation
def total_macarons := mitch_macarons + joshua_macarons + renz_macarons + M

-- Proof statement: Showing the ratio of number of macarons Joshua made to the number of macarons Miles made
theorem joshua_miles_ratio : (total_macarons = total_kids * macarons_per_kid) → (joshua_macarons : ℚ) / (M : ℚ) = 1 / 2 :=
by
  sorry

end joshua_miles_ratio_l212_212004


namespace range_of_a_l212_212118

theorem range_of_a :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  intros
  sorry

end range_of_a_l212_212118


namespace inequality_holds_iff_x_in_interval_l212_212255

theorem inequality_holds_iff_x_in_interval (x : ℝ) :
  (∀ n : ℕ, 0 < n → (1 + x)^n ≤ 1 + (2^n - 1) * x) ↔ (0 ≤ x ∧ x ≤ 1) :=
sorry

end inequality_holds_iff_x_in_interval_l212_212255


namespace quadratic_root_ratio_eq_l212_212199

theorem quadratic_root_ratio_eq (k : ℝ) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ (x = 3 * y ∨ y = 3 * x) ∧ x + y = -10 ∧ x * y = k) → k = 18.75 := by
  sorry

end quadratic_root_ratio_eq_l212_212199


namespace common_ratio_geometric_progression_l212_212931

noncomputable def a (x y z w : ℝ) : ℝ := x * (y - z)
noncomputable def ar (x y z w : ℝ) (r : ℝ) : ℝ := y * (z - x)
noncomputable def ar2 (x y z w : ℝ) (r : ℝ) : ℝ := z * (x - y)
noncomputable def ar3 (x y z w : ℝ) (r : ℝ) : ℝ := w * (x - y)

theorem common_ratio_geometric_progression 
  (x y z w : ℝ) (r : ℝ)
  (hxz : x ≠ z) (hxy : x ≠ y) (hyz : y ≠ z) (hwy : w ≠ y)
  (hxy0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (hw0 : w ≠ 0)
  (h1 : a x y z w ≠ 0)
  (ha : a x y z w = x * (y - z))
  (har : ar x y z w r = y * (z - x))
  (har2 : ar2 x y z w r = z * (x - y))
  (har3 : ar3 x y z w r = w * (x - y))
  (hr : ar x y z w r = r * a x y z w) 
  (hr2 : ar2 x y z w r = r * ar x y z w r)
  (hr3 : ar3 x y z w r = r * ar2 x y z w r) :
  r^3 + r^2 + r + 1 = 0 := by
  sorry

end common_ratio_geometric_progression_l212_212931


namespace max_knights_between_other_knights_l212_212220

-- Definitions and conditions derived from the problem
def total_knights := 40
def total_samurais := 10
def knights_with_samurai_on_right := 7

-- Statement to be proved
theorem max_knights_between_other_knights :
  let total_people := total_knights + total_samurais in
  let unaffected_knights := knights_with_samurai_on_right + 1 in
  ∃ (max_knights : ℕ), max_knights = total_knights - unaffected_knights ∧ max_knights = 32 :=
sorry

end max_knights_between_other_knights_l212_212220


namespace trees_in_one_row_l212_212422

variable (total_trees_cleaned : ℕ)
variable (trees_per_row : ℕ)

theorem trees_in_one_row (h1 : total_trees_cleaned = 20) (h2 : trees_per_row = 5) :
  (total_trees_cleaned / trees_per_row) = 4 :=
by
  sorry

end trees_in_one_row_l212_212422


namespace shaded_area_is_correct_l212_212930

-- Definitions based on the conditions
def is_square (s : ℝ) (area : ℝ) : Prop := s * s = area
def rect_area (l w : ℝ) : ℝ := l * w

variables (s : ℝ) (area_s : ℝ) (rect1_l rect1_w rect2_l rect2_w : ℝ)

-- Given conditions
def square := is_square s area_s
def rect1 := rect_area rect1_l rect1_w
def rect2 := rect_area rect2_l rect2_w

-- Problem statement: Prove the area of the shaded region
theorem shaded_area_is_correct
  (s: ℝ)
  (rect1_l rect1_w rect2_l rect2_w : ℝ)
  (h_square: is_square s 16)
  (h_rect1: rect_area rect1_l rect1_w = 6)
  (h_rect2: rect_area rect2_l rect2_w = 2) :
  (16 - (6 + 2) = 8) := 
  sorry

end shaded_area_is_correct_l212_212930


namespace distribute_balls_into_boxes_l212_212846

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l212_212846


namespace inequality_sqrt_l212_212160

theorem inequality_sqrt (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by
  sorry

end inequality_sqrt_l212_212160


namespace trigonometric_inequality_l212_212517

open Real

theorem trigonometric_inequality 
  (x y z : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < π / 2) : 
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
  sorry

end trigonometric_inequality_l212_212517


namespace total_spent_is_correct_l212_212389

def meal_prices : List ℕ := [12, 15, 10, 18, 20]
def ice_cream_prices : List ℕ := [2, 3, 3, 4, 4]
def tip_percentage : ℝ := 0.15
def tax_percentage : ℝ := 0.08

def total_meal_cost (prices : List ℕ) : ℝ :=
  prices.sum

def total_ice_cream_cost (prices : List ℕ) : ℝ :=
  prices.sum

def calculate_tip (total_meal_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  total_meal_cost * tip_percentage

def calculate_tax (total_meal_cost : ℝ) (tax_percentage : ℝ) : ℝ :=
  total_meal_cost * tax_percentage

def total_amount_spent (meal_prices : List ℕ) (ice_cream_prices : List ℕ) (tip_percentage : ℝ) (tax_percentage : ℝ) : ℝ :=
  let total_meal := total_meal_cost meal_prices
  let total_ice_cream := total_ice_cream_cost ice_cream_prices
  let tip := calculate_tip total_meal tip_percentage
  let tax := calculate_tax total_meal tax_percentage
  total_meal + total_ice_cream + tip + tax

theorem total_spent_is_correct :
  total_amount_spent meal_prices ice_cream_prices tip_percentage tax_percentage = 108.25 := 
by
  sorry

end total_spent_is_correct_l212_212389


namespace least_amount_of_money_l212_212495

variable (Money : Type) [LinearOrder Money]
variable (Anne Bo Coe Dan El : Money)

-- Conditions from the problem
axiom anne_less_than_bo : Anne < Bo
axiom dan_less_than_bo : Dan < Bo
axiom coe_less_than_anne : Coe < Anne
axiom coe_less_than_el : Coe < El
axiom coe_less_than_dan : Coe < Dan
axiom dan_less_than_anne : Dan < Anne

theorem least_amount_of_money : (∀ x, x = Anne ∨ x = Bo ∨ x = Coe ∨ x = Dan ∨ x = El → Coe < x) :=
by
  sorry

end least_amount_of_money_l212_212495


namespace total_digits_2500_is_9449_l212_212198

def nth_even (n : ℕ) : ℕ := 2 * n

def count_digits_in_range (start : ℕ) (stop : ℕ) : ℕ :=
  (stop - start) / 2 + 1

def total_digits (n : ℕ) : ℕ :=
  let one_digit := 4
  let two_digit := count_digits_in_range 10 98
  let three_digit := count_digits_in_range 100 998
  let four_digit := count_digits_in_range 1000 4998
  let five_digit := 1
  one_digit * 1 +
  two_digit * 2 +
  (three_digit * 3) +
  (four_digit * 4) +
  (five_digit * 5)

theorem total_digits_2500_is_9449 : total_digits 2500 = 9449 := by
  sorry

end total_digits_2500_is_9449_l212_212198


namespace problem_inequality_l212_212001

theorem problem_inequality (a b c : ℝ) (h₀ : a + b + c = 0) (d : ℝ) (h₁ : d = max (|a|) (max (|b|) (|c|))) : 
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 :=
sorry

end problem_inequality_l212_212001


namespace michael_wants_to_buy_more_packs_l212_212985

theorem michael_wants_to_buy_more_packs
  (initial_packs : ℕ)
  (cost_per_pack : ℝ)
  (total_value_after_purchase : ℝ)
  (value_of_current_packs : ℝ := initial_packs * cost_per_pack)
  (additional_value_needed : ℝ := total_value_after_purchase - value_of_current_packs)
  (packs_to_buy : ℝ := additional_value_needed / cost_per_pack)
  (answer : ℕ := 2) :
  initial_packs = 4 → cost_per_pack = 2.5 → total_value_after_purchase = 15 → packs_to_buy = answer :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end michael_wants_to_buy_more_packs_l212_212985


namespace monotone_increasing_intervals_exists_x0_implies_p_l212_212533

noncomputable def f (x : ℝ) := 6 * Real.log x + x ^ 2 - 8 * x
noncomputable def g (x : ℝ) (p : ℝ) := p / x + x ^ 2

theorem monotone_increasing_intervals :
  (∀ x, (0 < x ∧ x ≤ 1) → ∃ ε > 0, ∀ y, x < y → f y > f x) ∧
  (∀ x, (3 ≤ x) → ∃ ε > 0, ∀ y, x < y → f y > f x) := by
  sorry

theorem exists_x0_implies_p :
  (∃ x0, 1 ≤ x0 ∧ x0 ≤ Real.exp 1 ∧ f x0 > g x0 p) → p < -8 := by
  sorry

end monotone_increasing_intervals_exists_x0_implies_p_l212_212533


namespace rectangle_area_l212_212350

theorem rectangle_area (l w : ℝ) (h₁ : (2 * l + 2 * w) = 46) (h₂ : (l^2 + w^2) = 289) : l * w = 120 :=
by
  sorry

end rectangle_area_l212_212350


namespace gravel_weight_is_correct_l212_212357

def weight_of_gravel (total_weight : ℝ) (fraction_sand : ℝ) (fraction_water : ℝ) : ℝ :=
  total_weight - (fraction_sand * total_weight + fraction_water * total_weight)

theorem gravel_weight_is_correct :
  weight_of_gravel 23.999999999999996 (1 / 3) (1 / 4) = 10 :=
by
  sorry

end gravel_weight_is_correct_l212_212357


namespace find_angle_between_vectors_l212_212107

open Real

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def angle_between_vectors 
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (hmag : ∥b∥ = 4 * ∥a∥) 
  (horth : inner a (2 • a + b) = 0) : ℝ :=
2 * π / 3

theorem find_angle_between_vectors
  {a b : V} (ha : a ≠ 0) (hb : b ≠ 0)
  (hmag : ∥b∥ = 4 * ∥a∥)
  (horth : inner a (2 • a + b) = 0) :
   real.angle a b = 2 * π / 3 :=
sorry

end find_angle_between_vectors_l212_212107


namespace divisibility_1989_l212_212560

theorem divisibility_1989 (n : ℕ) (h1 : n ≥ 3) :
  1989 ∣ n^(n^(n^n)) - n^(n^n) :=
sorry

end divisibility_1989_l212_212560


namespace beth_remaining_marbles_l212_212766

theorem beth_remaining_marbles :
  (∀ (num_colors total_marbles : ℕ),
  total_marbles = 72 →
  num_colors = 3 →
  ∀ (initial_red initial_blue initial_yellow : ℕ),
  initial_red = total_marbles / num_colors →
  initial_blue = total_marbles / num_colors →
  initial_yellow = total_marbles / num_colors →
  ∀ (lost_red : ℕ),
  lost_red = 5 →
  ∀ (lost_blue : ℕ),
  lost_blue = 2 * lost_red →
  ∀ (lost_yellow : ℕ),
  lost_yellow = 3 * lost_red →
  let remaining_red := initial_red - lost_red in
  let remaining_blue := initial_blue - lost_blue in
  let remaining_yellow := initial_yellow - lost_yellow in
  remaining_red + remaining_blue + remaining_yellow = 42) :=
begin
  intros num_colors total_marbles total_marbles_is_72 num_colors_is_3 
         initial_red initial_blue initial_yellow 
         initial_red_is_total_marbles_div_num_colors initial_blue_is_total_marbles_div_num_colors initial_yellow_is_total_marbles_div_num_colors 
         lost_red lost_red_is_5 
         lost_blue lost_blue_is_2_times_lost_red 
         lost_yellow lost_yellow_is_3_times_lost_red,
  
  have h_initial : initial_red = 24 ∧ initial_blue = 24 ∧ initial_yellow = 24,
  { split; try {split}; rw [initial_red_is_total_marbles_div_num_colors, initial_blue_is_total_marbles_div_num_colors, initial_yellow_is_total_marbles_div_num_colors, total_marbles_is_72, num_colors_is_3]; exact rfl },
  
  rw [h_initial.1, h_initial.2.1, h_initial.2.2],
  let remaining_red := 24 - 5,
  let remaining_blue := 24 - 10,
  let remaining_yellow := 24 - 15,
  have h_remaining : remaining_red + remaining_blue + remaining_yellow = 42,
  { calc
    (24 - 5) + (24 - 10) + (24 - 15)
      = 19 + (24 - 10) + (24 - 15) : by rw [nat.sub_eq, rfl]
  ... = 19 + 14 + (24 - 15) : by rw [nat.sub_eq, rfl]
  ... = 19 + 14 + 9 : by rw [nat.sub_eq, rfl]
  ... = 42 : by ring },
  exact h_remaining,
end

end beth_remaining_marbles_l212_212766


namespace max_area_triangle_l212_212938

theorem max_area_triangle (a b c S : ℝ) (h₁ : S = a^2 - (b - c)^2) (h₂ : b + c = 8) :
  S ≤ 64 / 17 :=
sorry

end max_area_triangle_l212_212938


namespace log_inequality_l212_212429

   variable {a b : ℝ}

   theorem log_inequality (h1 : a > b) (h2 : b > 1) (h3 : a > 1) : log 2 a > log 2 b ∧ log 2 b > 0 ↔ a > b ∧ b > 1 :=
   by
     sorry
   
end log_inequality_l212_212429


namespace cubic_solution_unique_real_l212_212785

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end cubic_solution_unique_real_l212_212785


namespace sum_greater_than_four_l212_212570

theorem sum_greater_than_four (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hprod : x * y > x + y) : x + y > 4 :=
by
  sorry

end sum_greater_than_four_l212_212570


namespace zacharys_bus_ride_length_l212_212736

theorem zacharys_bus_ride_length (Vince Zachary : ℝ) (hV : Vince = 0.62) (hDiff : Vince = Zachary + 0.13) : Zachary = 0.49 :=
by
  sorry

end zacharys_bus_ride_length_l212_212736


namespace other_denominations_l212_212757

theorem other_denominations :
  ∀ (total_checks : ℕ) (total_value : ℝ) (fifty_denomination_checks : ℕ) (remaining_avg : ℝ),
    total_checks = 30 →
    total_value = 1800 →
    fifty_denomination_checks = 15 →
    remaining_avg = 70 →
    ∃ (other_denomination : ℝ), other_denomination = 70 :=
by
  intros total_checks total_value fifty_denomination_checks remaining_avg
  intros h1 h2 h3 h4
  let other_denomination := 70
  use other_denomination
  sorry

end other_denominations_l212_212757


namespace fraction_spent_on_food_l212_212910

theorem fraction_spent_on_food (r c f : ℝ) (l s : ℝ)
  (hr : r = 1/10)
  (hc : c = 3/5)
  (hl : l = 16000)
  (hs : s = 160000)
  (heq : f * s + r * s + c * s + l = s) :
  f = 1/5 :=
by
  sorry

end fraction_spent_on_food_l212_212910


namespace complex_magnitude_addition_l212_212507

theorem complex_magnitude_addition :
  (Complex.abs (3 / 4 - 3 * Complex.I) + 5 / 12) = (9 * Real.sqrt 17 + 5) / 12 := 
  sorry

end complex_magnitude_addition_l212_212507


namespace max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l212_212054

section RandomWalk

variables {a b : ℕ} (h : a > b)

def max_range_walk : ℕ := a
def min_range_walk : ℕ := a - b
def count_max_range_sequences : ℕ := b + 1

theorem max_range_of_walk (h : a > b) : max_range_walk h = a :=
by
  sorry

theorem min_range_of_walk (h : a > b) : min_range_walk h = a - b :=
by
  sorry

theorem count_of_max_range_sequences (h : a > b) : count_max_range_sequences h = b + 1 :=
by
  sorry

end RandomWalk

end max_range_of_walk_min_range_of_walk_count_of_max_range_sequences_l212_212054


namespace knights_max_seated_between_knights_l212_212232

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l212_212232


namespace number_of_arrangements_l212_212653

open Nat
open BigOperators

theorem number_of_arrangements :
  (factorial 3) * (factorial 3) * (factorial 4) * (factorial 5) = 103680 := by
  sorry

end number_of_arrangements_l212_212653


namespace original_number_is_fraction_l212_212569

theorem original_number_is_fraction (x : ℚ) (h : 1 + (1 / x) = 9 / 4) : x = 4 / 5 :=
by
  sorry

end original_number_is_fraction_l212_212569


namespace possible_values_of_a_l212_212851

def line1 (x y : ℝ) := x + y + 1 = 0
def line2 (x y : ℝ) := 2 * x - y + 8 = 0
def line3 (a : ℝ) (x y : ℝ) := a * x + 3 * y - 5 = 0

theorem possible_values_of_a :
  {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} ⊆ {1/3, 3, -6} ∧
  {1/3, 3, -6} ⊆ {a : ℝ | ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ line3 a x y} :=
sorry

end possible_values_of_a_l212_212851


namespace bianca_deleted_text_files_l212_212469

theorem bianca_deleted_text_files (pictures songs total : ℕ) (h₁ : pictures = 2) (h₂ : songs = 8) (h₃ : total = 17) :
  total - (pictures + songs) = 7 :=
by {
  sorry
}

end bianca_deleted_text_files_l212_212469


namespace x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l212_212017

theorem x_sq_plus_3x_minus_2_ge_zero (x : ℝ) (h : x ≥ 1) : x^2 + 3 * x - 2 ≥ 0 :=
sorry

theorem neg_x_sq_plus_3x_minus_2_lt_zero (x : ℝ) (h : x < 1) : x^2 + 3 * x - 2 < 0 :=
sorry

end x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l212_212017


namespace find_x_value_l212_212543

noncomputable def solve_some_number (x : ℝ) : Prop :=
  let expr := (x - (8 / 7) * 5 + 10)
  expr = 13.285714285714286

theorem find_x_value : ∃ x : ℝ, solve_some_number x ∧ x = 9 := by
  sorry

end find_x_value_l212_212543


namespace divisor_is_11_l212_212047

noncomputable def least_subtracted_divisor : Nat := 11

def problem_condition (D : Nat) (x : Nat) : Prop :=
  2000 - x = 1989 ∧ (2000 - x) % D = 0

theorem divisor_is_11 (D : Nat) (x : Nat) (h : problem_condition D x) : D = least_subtracted_divisor :=
by
  sorry

end divisor_is_11_l212_212047


namespace melissa_total_cost_l212_212565

-- Definitions based on conditions
def daily_rental_rate : ℝ := 15
def mileage_rate : ℝ := 0.10
def number_of_days : ℕ := 3
def number_of_miles : ℕ := 300

-- Theorem statement to prove the total cost
theorem melissa_total_cost : daily_rental_rate * number_of_days + mileage_rate * number_of_miles = 75 := 
by 
  sorry

end melissa_total_cost_l212_212565


namespace dad_caught_more_l212_212502

theorem dad_caught_more {trouts_caleb : ℕ} (h₁ : trouts_caleb = 2) 
    (h₂ : ∃ trouts_dad : ℕ, trouts_dad = 3 * trouts_caleb) : 
    ∃ more_trouts : ℕ, more_trouts = 4 := by
  sorry

end dad_caught_more_l212_212502


namespace balls_into_boxes_l212_212843

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end balls_into_boxes_l212_212843


namespace complement_of_union_l212_212151

open Set

namespace Proof

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of the union of sets A and B with respect to U
theorem complement_of_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5}) 
  (hA : A = {1, 3}) (hB : B = {3, 5}) : 
  U \ (A ∪ B) = {0, 2, 4} :=
by {
  sorry
}

end Proof

end complement_of_union_l212_212151


namespace multiples_of_lcm_13_9_in_range_l212_212831

theorem multiples_of_lcm_13_9_in_range : 
  {n : ℤ | 200 ≤ n ∧ n ≤ 500 ∧ (13 ∣ n) ∧ (9 ∣ n)}.card = 3 :=
by {
  sorry
}

end multiples_of_lcm_13_9_in_range_l212_212831


namespace holiday_customers_l212_212906

-- Define the normal rate of customers entering the store (175 people/hour)
def normal_rate : ℕ := 175

-- Define the holiday rate of customers entering the store
def holiday_rate : ℕ := 2 * normal_rate

-- Define the duration for which we are calculating the total number of customers (8 hours)
def duration : ℕ := 8

-- Define the correct total number of customers (2800 people)
def correct_total_customers : ℕ := 2800

-- The theorem that asserts the total number of customers in 8 hours during the holiday season is 2800
theorem holiday_customers : holiday_rate * duration = correct_total_customers := by
  sorry

end holiday_customers_l212_212906


namespace find_a1_l212_212112

theorem find_a1 (a_1 : ℕ) (S : ℕ → ℕ) (S_formula : ∀ n : ℕ, S n = (a_1 * (3^n - 1)) / 2)
  (a_4_eq : (S 4) - (S 3) = 54) : a_1 = 2 :=
  sorry

end find_a1_l212_212112


namespace discriminant_square_eq_l212_212417

variable {a b c x : ℝ}

-- Condition: a ≠ 0
axiom h_a : a ≠ 0

-- Condition: x is a root of the quadratic equation ax^2 + bx + c = 0
axiom h_root : a * x^2 + b * x + c = 0

theorem discriminant_square_eq (h_a : a ≠ 0) (h_root : a * x^2 + b * x + c = 0) :
  (2 * a * x + b)^2 = b^2 - 4 * a * c :=
by 
  sorry

end discriminant_square_eq_l212_212417


namespace probability_two_people_between_l212_212341

theorem probability_two_people_between (total_people : ℕ) (favorable_arrangements : ℕ) (total_arrangements : ℕ) :
  total_people = 6 ∧ favorable_arrangements = 144 ∧ total_arrangements = 720 →
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 5 :=
by
  intros h
  -- We substitute the given conditions
  have ht : total_people = 6 := h.1
  have hf : favorable_arrangements = 144 := h.2.1
  have ha : total_arrangements = 720 := h.2.2
  -- We need to calculate the probability considering the favorable and total arrangements
  sorry

end probability_two_people_between_l212_212341


namespace cross_section_area_correct_l212_212732

noncomputable def cross_section_area (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 33) / 8

theorem cross_section_area_correct
  (AB CC1 : ℝ)
  (h1 : AB = a)
  (h2 : CC1 = 2 * a) :
  cross_section_area a = (3 * a^2 * Real.sqrt 33) / 8 :=
by
  sorry

end cross_section_area_correct_l212_212732


namespace min_x_y_l212_212280

theorem min_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 2) :
  x + y ≥ 9 / 2 := 
by 
  sorry

end min_x_y_l212_212280


namespace triangle_area_l212_212299

noncomputable def area_triangle (A B C : ℝ) (b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem triangle_area
  (A B C : ℝ) (b : ℝ) 
  (hA : A = π / 4)
  (h0 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  ∃ c : ℝ, area_triangle A B C b c = 2 :=
by
  sorry

end triangle_area_l212_212299


namespace initial_cost_of_article_l212_212079

variable (P : ℝ)

theorem initial_cost_of_article (h1 : 0.70 * P = 2100) (h2 : 0.50 * (0.70 * P) = 1050) : P = 3000 :=
by
  sorry

end initial_cost_of_article_l212_212079


namespace minimum_cuts_for_48_pieces_l212_212986

theorem minimum_cuts_for_48_pieces 
  (rearrange_without_folding : Prop)
  (can_cut_multiple_layers_simultaneously : Prop)
  (straight_line_cut : Prop)
  (cut_doubles_pieces : ∀ n, ∃ m, m = 2 * n) :
  ∃ n, (2^n ≥ 48 ∧ ∀ m, (m < n → 2^m < 48)) ∧ n = 6 := 
by 
  sorry

end minimum_cuts_for_48_pieces_l212_212986


namespace crayons_more_than_erasers_l212_212015

-- Definitions of the conditions
def initial_crayons := 531
def initial_erasers := 38
def final_crayons := 391
def final_erasers := initial_erasers -- no erasers lost

-- Theorem statement
theorem crayons_more_than_erasers :
  final_crayons - final_erasers = 102 :=
by
  -- Placeholder for the proof
  sorry

end crayons_more_than_erasers_l212_212015


namespace scientific_notation_of_170000_l212_212448

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l212_212448


namespace reciprocal_sum_greater_l212_212655

theorem reciprocal_sum_greater (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
    (1 / a + 1 / b) > 1 / (a + b) :=
sorry

end reciprocal_sum_greater_l212_212655


namespace pascal_triangle_row_10_sum_l212_212918

theorem pascal_triangle_row_10_sum : (∑ (k : Fin 11), nat.choose 10 k) = 1024 := by
  sorry

end pascal_triangle_row_10_sum_l212_212918


namespace value_of_expression_l212_212124

theorem value_of_expression (x y : ℝ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^3 + 3*y^3) / 9 = 24.33 :=
by
  sorry

end value_of_expression_l212_212124


namespace solution_y_amount_l212_212702

-- Definitions based on the conditions
def alcohol_content_x : ℝ := 0.10
def alcohol_content_y : ℝ := 0.30
def initial_volume_x : ℝ := 50
def final_alcohol_percent : ℝ := 0.25

-- Function to calculate the amount of solution y needed
def required_solution_y (y : ℝ) : Prop :=
  (alcohol_content_x * initial_volume_x + alcohol_content_y * y) / (initial_volume_x + y) = final_alcohol_percent

theorem solution_y_amount : ∃ y : ℝ, required_solution_y y ∧ y = 150 := by
  sorry

end solution_y_amount_l212_212702


namespace total_houses_in_lincoln_county_l212_212728

theorem total_houses_in_lincoln_county 
  (original_houses : ℕ) 
  (built_houses : ℕ) 
  (h_original : original_houses = 20817) 
  (h_built : built_houses = 97741) : 
  original_houses + built_houses = 118558 := 
by
  -- Sorry is used to skip the proof.
  sorry

end total_houses_in_lincoln_county_l212_212728


namespace toys_total_is_240_l212_212008

def number_of_toys_elder : Nat := 60
def number_of_toys_younger (toys_elder : Nat) : Nat := 3 * toys_elder
def total_number_of_toys (toys_elder toys_younger : Nat) : Nat := toys_elder + toys_younger

theorem toys_total_is_240 : total_number_of_toys number_of_toys_elder (number_of_toys_younger number_of_toys_elder) = 240 :=
by
  sorry

end toys_total_is_240_l212_212008


namespace max_b_div_a_plus_c_l212_212526

-- Given positive numbers a, b, c
-- equation: b^2 + 2(a + c)b - ac = 0
-- Prove: ∀ a b c : ℝ (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : b^2 + 2*(a + c)*b - a*c = 0),
--         b/(a + c) ≤ (Real.sqrt 5 - 2)/2

theorem max_b_div_a_plus_c (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : b^2 + 2 * (a + c) * b - a * c = 0) :
  b / (a + c) ≤ (Real.sqrt 5 - 2) / 2 :=
sorry

end max_b_div_a_plus_c_l212_212526


namespace num_pos_cubes_ending_in_5_lt_5000_l212_212120

theorem num_pos_cubes_ending_in_5_lt_5000 : 
  (∃ (n1 n2 : ℕ), (n1 ≤ 5000 ∧ n2 ≤ 5000) ∧ (n1^3 % 10 = 5 ∧ n2^3 % 10 = 5) ∧ (n1^3 < 5000 ∧ n2^3 < 5000) ∧ n1 ≠ n2 ∧ 
  ∀ n, (n^3 < 5000 ∧ n^3 % 10 = 5) → (n = n1 ∨ n = n2)) :=
sorry

end num_pos_cubes_ending_in_5_lt_5000_l212_212120


namespace bike_riders_count_l212_212853

variables (B H : ℕ)

theorem bike_riders_count
  (h₁ : H = B + 178)
  (h₂ : H + B = 676) :
  B = 249 :=
sorry

end bike_riders_count_l212_212853


namespace five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l212_212459

-- Prove that the number of five-digit numbers is 27216
theorem five_digit_numbers_count : ∃ n, n = 9 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the number of five-digit numbers greater than or equal to 30000 is 21168
theorem five_digit_numbers_ge_30000 : 
  ∃ n, n = 7 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the rank of 50124 among five-digit numbers with distinct digits in descending order is 15119
theorem rank_of_50124 : 
  ∃ n, n = (Nat.factorial 5) - 1 := by
  sorry

end five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l212_212459


namespace sum_of_valid_a_l212_212661

theorem sum_of_valid_a :
  (∑ x in ({-1, 0, 2, 3, 4, 5} : Finset ℤ), x) = 13 := by
  sorry

end sum_of_valid_a_l212_212661


namespace triangle_shape_l212_212308

theorem triangle_shape
  (A B C : ℝ) -- Internal angles of triangle ABC
  (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively
  (h1 : a * (Real.cos A) * (Real.cos B) + b * (Real.cos A) * (Real.cos A) = a * (Real.cos A)) :
  (A = Real.pi / 2) ∨ (A = C) :=
sorry

end triangle_shape_l212_212308


namespace Lakeview_High_School_Basketball_Team_l212_212763

theorem Lakeview_High_School_Basketball_Team :
  ∀ (total_players taking_physics taking_both statistics: ℕ),
  total_players = 25 →
  taking_physics = 10 →
  taking_both = 5 →
  statistics = 20 :=
sorry

end Lakeview_High_School_Basketball_Team_l212_212763


namespace complex_exponential_sum_l212_212959

theorem complex_exponential_sum (γ δ : ℝ) 
  (h : Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -1 / 2 + 5 / 4 * Complex.I) :
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -1 / 2 - 5 / 4 * Complex.I :=
by
  sorry

end complex_exponential_sum_l212_212959


namespace number_of_zeros_of_f_l212_212384

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem number_of_zeros_of_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l212_212384


namespace practice_match_allocations_count_l212_212374

theorem practice_match_allocations_count :
  ∃ (x : Fin 7 → ℕ), (∑ i, x i) = 270 ∧
                      (∀ i : Fin 4, x i % 7 = 0) ∧
                      (∀ i : Fin (7 - 4), x (i + 4) % 13 = 0) ∧
                      fintype.card {x // 
                        (∑ i, x i = 270) ∧ 
                        (∀ (i : Fin 4), x i % 7 = 0) ∧ 
                        (∀ (i : Fin (7 - 4)), x (i + 4) % 13 = 0)
                      } = 27352 :=
sorry

end practice_match_allocations_count_l212_212374


namespace number_of_intersections_l212_212504

/-- 
  Define the two curves as provided in the problem:
  curve1 is defined by the equation 3x² + 2y² = 6,
  curve2 is defined by the equation x² - 2y² = 1.
  We aim to prove that there are exactly 4 distinct intersection points.
--/
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6

def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

theorem number_of_intersections : ∃ (points : Finset (ℝ × ℝ)), (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 4 :=
sorry

end number_of_intersections_l212_212504


namespace line_intersects_y_axis_at_point_l212_212484

theorem line_intersects_y_axis_at_point :
  let x1 := 3
  let y1 := 20
  let x2 := -7
  let y2 := 2

  -- line equation from 2 points: y - y1 = m * (x - x1)
  -- slope m = (y2 - y1) / (x2 - x1)
  -- y-intercept when x = 0:
  
  (0, 14.6) ∈ { p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ 
    m = (y2 - y1) / (x2 - x1) ∧ 
    b = y1 - m * x1 }
  :=
  sorry

end line_intersects_y_axis_at_point_l212_212484


namespace probability_at_least_one_correct_l212_212316

open Classical

theorem probability_at_least_one_correct :
  let prob_miss := 5 / 6,
      prob_miss_all := (prob_miss ^ 5 : ℚ),
      prob_at_least_one := 1 - prob_miss_all
  in prob_at_least_one = 4651 / 7776 := by
  let prob_miss := 5 / 6 : ℚ
  let prob_miss_all := (prob_miss ^ 5 : ℚ)
  let prob_at_least_one := 1 - prob_miss_all
  show prob_at_least_one = 4651 / 7776
  sorry

end probability_at_least_one_correct_l212_212316


namespace total_volume_of_five_cubes_l212_212340

-- Definition for volume of a cube function
def volume_of_cube (edge_length : ℝ) : ℝ :=
  edge_length ^ 3

-- Conditions
def edge_length : ℝ := 5
def number_of_cubes : ℝ := 5

-- Proof statement
theorem total_volume_of_five_cubes : 
  volume_of_cube edge_length * number_of_cubes = 625 := 
by
  sorry

end total_volume_of_five_cubes_l212_212340
