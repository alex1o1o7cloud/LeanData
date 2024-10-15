import Mathlib

namespace NUMINAMATH_GPT_find_x_l1342_134249

-- Define x as a function of n, where n is an odd natural number
def x (n : ℕ) (h_odd : n % 2 = 1) : ℕ :=
  6^n + 1

-- Define the main theorem
theorem find_x (n : ℕ) (h_odd : n % 2 = 1) (h_prime_div : ∀ p, p.Prime → p ∣ x n h_odd → (p = 11 ∨ p = 7 ∨ p = 101)) : x 1 (by norm_num) = 7777 :=
  sorry

end NUMINAMATH_GPT_find_x_l1342_134249


namespace NUMINAMATH_GPT_three_digit_number_increase_l1342_134209

theorem three_digit_number_increase (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n * 1001 / n) = 1001 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_increase_l1342_134209


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l1342_134283

theorem simplify_and_evaluate_expr :
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 :=
by
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l1342_134283


namespace NUMINAMATH_GPT_sahil_selling_price_correct_l1342_134202

-- Define the conditions as constants
def cost_of_machine : ℕ := 13000
def cost_of_repair : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

-- Define the total cost calculation
def total_cost : ℕ := cost_of_machine + cost_of_repair + transportation_charges

-- Define the profit calculation
def profit : ℕ := total_cost * profit_percentage / 100

-- Define the selling price calculation
def selling_price : ℕ := total_cost + profit

-- Now we express our proof problem
theorem sahil_selling_price_correct :
  selling_price = 28500 := by
  -- sorries to skip the proof.
  sorry

end NUMINAMATH_GPT_sahil_selling_price_correct_l1342_134202


namespace NUMINAMATH_GPT_lcm_150_294_l1342_134270

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end NUMINAMATH_GPT_lcm_150_294_l1342_134270


namespace NUMINAMATH_GPT_max_gcd_of_consecutive_terms_l1342_134271

-- Given conditions
def a (n : ℕ) : ℕ := 2 * (n.factorial) + n

-- Theorem statement
theorem max_gcd_of_consecutive_terms : ∃ (d : ℕ), ∀ n ≥ 0, d ≤ gcd (a n) (a (n + 1)) ∧ d = 1 := by sorry

end NUMINAMATH_GPT_max_gcd_of_consecutive_terms_l1342_134271


namespace NUMINAMATH_GPT_value_of_f_m_plus_one_depends_on_m_l1342_134274

def f (x a : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one_depends_on_m (m a : ℝ) (h : f (-m) a < 0) :
  (∃ m, f (m + 1) a < 0) ∧ (∃ m, f (m + 1) a > 0) :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_m_plus_one_depends_on_m_l1342_134274


namespace NUMINAMATH_GPT_intersection_setA_setB_l1342_134233

-- Define set A
def setA : Set ℝ := {x | 2 * x ≤ 4}

-- Define set B as the domain of the function y = log(x - 1)
def setB : Set ℝ := {x | x > 1}

-- Theorem to prove
theorem intersection_setA_setB : setA ∩ setB = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_setA_setB_l1342_134233


namespace NUMINAMATH_GPT_correct_sampling_methods_l1342_134269

theorem correct_sampling_methods :
  (let num_balls := 1000
   let red_box := 500
   let blue_box := 200
   let yellow_box := 300
   let sample_balls := 100
   let num_students := 20
   let selected_students := 3
   let q1_method := "stratified"
   let q2_method := "simple_random"
   q1_method = "stratified" ∧ q2_method = "simple_random") := sorry

end NUMINAMATH_GPT_correct_sampling_methods_l1342_134269


namespace NUMINAMATH_GPT_alpha_beta_power_eq_sum_power_for_large_p_l1342_134296

theorem alpha_beta_power_eq_sum_power_for_large_p (α β : ℂ) (p : ℕ) (hp : p ≥ 5)
  (hαβ : ∀ x : ℂ, 2 * x^4 - 6 * x^3 + 11 * x^2 - 6 * x - 4 = 0 → x = α ∨ x = β) :
  α^p + β^p = (α + β)^p :=
sorry

end NUMINAMATH_GPT_alpha_beta_power_eq_sum_power_for_large_p_l1342_134296


namespace NUMINAMATH_GPT_total_cost_of_items_l1342_134245

variables (E P M : ℝ)

-- Conditions
def condition1 : Prop := E + 3 * P + 2 * M = 240
def condition2 : Prop := 2 * E + 5 * P + 4 * M = 440

-- Question to prove
def question (E P M : ℝ) : ℝ := 3 * E + 4 * P + 6 * M

theorem total_cost_of_items (E P M : ℝ) :
  condition1 E P M →
  condition2 E P M →
  question E P M = 520 := 
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_total_cost_of_items_l1342_134245


namespace NUMINAMATH_GPT_compute_expression_l1342_134266

variable {R : Type*} [LinearOrderedField R]

theorem compute_expression (r s t : R)
  (h_eq_root: ∀ x, x^3 - 4 * x^2 + 4 * x - 6 = 0)
  (h1: r + s + t = 4)
  (h2: r * s + r * t + s * t = 4)
  (h3: r * s * t = 6) :
  r * s / t + s * t / r + t * r / s = -16 / 3 :=
sorry

end NUMINAMATH_GPT_compute_expression_l1342_134266


namespace NUMINAMATH_GPT_inequality_reversal_l1342_134210

theorem inequality_reversal (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by
  sorry

end NUMINAMATH_GPT_inequality_reversal_l1342_134210


namespace NUMINAMATH_GPT_fewer_twos_to_hundred_l1342_134227

theorem fewer_twos_to_hundred : (222 / 2 - 22 / 2 = 100) :=
by
  sorry

end NUMINAMATH_GPT_fewer_twos_to_hundred_l1342_134227


namespace NUMINAMATH_GPT_max_a_value_l1342_134211

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 - 2 * x - 3 > 0) →
  (¬ (∀ x : ℝ, x^2 - 2 * x - 3 > 0 → x < a)) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_a_value_l1342_134211


namespace NUMINAMATH_GPT_watermelon_sales_correct_l1342_134298

def total_watermelons_sold 
  (customers_one_melon : ℕ) 
  (customers_three_melons : ℕ) 
  (customers_two_melons : ℕ) : ℕ :=
  (customers_one_melon * 1) + (customers_three_melons * 3) + (customers_two_melons * 2)

theorem watermelon_sales_correct :
  total_watermelons_sold 17 3 10 = 46 := by
  sorry

end NUMINAMATH_GPT_watermelon_sales_correct_l1342_134298


namespace NUMINAMATH_GPT_value_standard_deviations_less_than_mean_l1342_134216

-- Definitions of the given conditions
def mean : ℝ := 15
def std_dev : ℝ := 1.5
def value : ℝ := 12

-- Lean 4 statement to prove the question
theorem value_standard_deviations_less_than_mean :
  (mean - value) / std_dev = 2 := by
  sorry

end NUMINAMATH_GPT_value_standard_deviations_less_than_mean_l1342_134216


namespace NUMINAMATH_GPT_cost_per_box_types_l1342_134250

-- Definitions based on conditions
def cost_type_B := 1500
def cost_type_A := cost_type_B + 500

-- Given conditions
def condition1 : cost_type_A = cost_type_B + 500 := by sorry
def condition2 : 6000 / (cost_type_B + 500) = 4500 / cost_type_B := by sorry

-- Theorem to be proved
theorem cost_per_box_types :
  cost_type_A = 2000 ∧ cost_type_B = 1500 ∧
  (∃ (m : ℕ), 20 ≤ m ∧ m ≤ 25 ∧ 2000 * (50 - m) + 1500 * m ≤ 90000) ∧
  (∃ (a b : ℕ), 2500 * a + 3500 * b = 87500 ∧ a + b ≤ 33) :=
sorry

end NUMINAMATH_GPT_cost_per_box_types_l1342_134250


namespace NUMINAMATH_GPT_correct_average_wrong_reading_l1342_134299

theorem correct_average_wrong_reading
  (initial_average : ℕ) (list_length : ℕ) (wrong_number : ℕ) (correct_number : ℕ) (correct_average : ℕ) 
  (h1 : initial_average = 18)
  (h2 : list_length = 10)
  (h3 : wrong_number = 26)
  (h4 : correct_number = 66)
  (h5 : correct_average = 22) :
  correct_average = ((initial_average * list_length) - wrong_number + correct_number) / list_length :=
sorry

end NUMINAMATH_GPT_correct_average_wrong_reading_l1342_134299


namespace NUMINAMATH_GPT_sarah_bus_time_l1342_134207

noncomputable def totalTimeAway : ℝ := (4 + 15/60) + (5 + 15/60)  -- 9.5 hours
noncomputable def totalTimeAwayInMinutes : ℝ := totalTimeAway * 60  -- 570 minutes

noncomputable def timeInClasses : ℝ := 8 * 45  -- 360 minutes
noncomputable def timeInLunch : ℝ := 30  -- 30 minutes
noncomputable def timeInExtracurricular : ℝ := 1.5 * 60  -- 90 minutes
noncomputable def totalTimeInSchoolActivities : ℝ := timeInClasses + timeInLunch + timeInExtracurricular  -- 480 minutes

noncomputable def timeOnBus : ℝ := totalTimeAwayInMinutes - totalTimeInSchoolActivities  -- 90 minutes

theorem sarah_bus_time : timeOnBus = 90 := by
  sorry

end NUMINAMATH_GPT_sarah_bus_time_l1342_134207


namespace NUMINAMATH_GPT_x_equals_l1342_134239

variable (x y: ℝ)

theorem x_equals:
  (x / (x - 2) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 1)) → x = 2 * y^2 + 6 * y + 2 := by
  sorry

end NUMINAMATH_GPT_x_equals_l1342_134239


namespace NUMINAMATH_GPT_paths_from_A_to_B_no_revisits_l1342_134253

noncomputable def numPaths : ℕ :=
  16

theorem paths_from_A_to_B_no_revisits : numPaths = 16 :=
by
  sorry

end NUMINAMATH_GPT_paths_from_A_to_B_no_revisits_l1342_134253


namespace NUMINAMATH_GPT_rainfall_on_tuesday_l1342_134254

noncomputable def R_Tuesday (R_Sunday : ℝ) (D1 : ℝ) : ℝ := 
  R_Sunday + D1

noncomputable def R_Thursday (R_Tuesday : ℝ) (D2 : ℝ) : ℝ :=
  R_Tuesday + D2

noncomputable def total_rainfall (R_Sunday R_Tuesday R_Thursday : ℝ) : ℝ :=
  R_Sunday + R_Tuesday + R_Thursday

theorem rainfall_on_tuesday : R_Tuesday 2 3.75 = 5.75 := 
by 
  sorry -- Proof goes here

end NUMINAMATH_GPT_rainfall_on_tuesday_l1342_134254


namespace NUMINAMATH_GPT_total_pencils_correct_l1342_134290

def initial_pencils : ℕ := 245
def added_pencils : ℕ := 758
def total_pencils : ℕ := initial_pencils + added_pencils

theorem total_pencils_correct : total_pencils = 1003 := 
by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l1342_134290


namespace NUMINAMATH_GPT_chess_program_ratio_l1342_134287

theorem chess_program_ratio {total_students chess_program_absent : ℕ}
  (h_total : total_students = 24)
  (h_absent : chess_program_absent = 4)
  (h_half : chess_program_absent * 2 = chess_program_absent + chess_program_absent) :
  (chess_program_absent * 2 : ℚ) / total_students = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_chess_program_ratio_l1342_134287


namespace NUMINAMATH_GPT_decimal_to_base9_l1342_134297

theorem decimal_to_base9 (n : ℕ) (h : n = 1729) : 
  (2 * 9^3 + 3 * 9^2 + 3 * 9^1 + 1 * 9^0) = n :=
by sorry

end NUMINAMATH_GPT_decimal_to_base9_l1342_134297


namespace NUMINAMATH_GPT_sample_capacity_l1342_134231

theorem sample_capacity (n : ℕ) (A B C : ℕ) (h_ratio : A / (A + B + C) = 3 / 14) (h_A : A = 15) : n = 70 :=
by
  sorry

end NUMINAMATH_GPT_sample_capacity_l1342_134231


namespace NUMINAMATH_GPT_symmetric_function_expression_l1342_134208

variable (f : ℝ → ℝ)
variable (h_sym : ∀ x y, f (-2 - x) = - f x)
variable (h_def : ∀ x, 0 < x → f x = 1 / x)

theorem symmetric_function_expression : ∀ x, x < -2 → f x = 1 / (2 + x) :=
by
  intro x
  intro hx
  sorry

end NUMINAMATH_GPT_symmetric_function_expression_l1342_134208


namespace NUMINAMATH_GPT_larry_wins_game_l1342_134219

-- Defining probabilities for Larry and Julius
def larry_throw_prob : ℚ := 2 / 3
def julius_throw_prob : ℚ := 1 / 3

-- Calculating individual probabilities based on the description
def p1 : ℚ := larry_throw_prob
def p3 : ℚ := (julius_throw_prob ^ 2) * larry_throw_prob
def p5 : ℚ := (julius_throw_prob ^ 4) * larry_throw_prob

-- Aggregating the probability that Larry wins the game
def larry_wins_prob : ℚ := p1 + p3 + p5

-- The proof statement
theorem larry_wins_game : larry_wins_prob = 170 / 243 := by
  sorry

end NUMINAMATH_GPT_larry_wins_game_l1342_134219


namespace NUMINAMATH_GPT_solomon_sale_price_l1342_134267

def original_price : ℝ := 500
def discount_rate : ℝ := 0.10
def sale_price := original_price * (1 - discount_rate)

theorem solomon_sale_price : sale_price = 450 := by
  sorry

end NUMINAMATH_GPT_solomon_sale_price_l1342_134267


namespace NUMINAMATH_GPT_probability_club_then_spade_l1342_134295

/--
   Two cards are dealt at random from a standard deck of 52 cards.
   Prove that the probability that the first card is a club (♣) and the second card is a spade (♠) is 13/204.
-/
theorem probability_club_then_spade :
  let total_cards := 52
  let clubs := 13
  let spades := 13
  let first_card_club_prob := (clubs : ℚ) / total_cards
  let second_card_spade_prob := (spades : ℚ) / (total_cards - 1)
  first_card_club_prob * second_card_spade_prob = 13 / 204 :=
by
  sorry

end NUMINAMATH_GPT_probability_club_then_spade_l1342_134295


namespace NUMINAMATH_GPT_students_shorter_than_yoongi_l1342_134248

variable (total_students taller_than_yoongi : Nat)

theorem students_shorter_than_yoongi (h₁ : total_students = 20) (h₂ : taller_than_yoongi = 11) : 
    total_students - (taller_than_yoongi + 1) = 8 :=
by
  -- Here would be the proof
  sorry

end NUMINAMATH_GPT_students_shorter_than_yoongi_l1342_134248


namespace NUMINAMATH_GPT_pythagorean_theorem_sets_l1342_134288

theorem pythagorean_theorem_sets :
  ¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2) ∧
  (1 ^ 2 + (Real.sqrt 3) ^ 2 = 2 ^ 2) ∧
  ¬ (5 ^ 2 + 6 ^ 2 = 7 ^ 2) ∧
  ¬ (1 ^ 2 + (Real.sqrt 2) ^ 2 = 3 ^ 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_pythagorean_theorem_sets_l1342_134288


namespace NUMINAMATH_GPT_mrs_wilsborough_tickets_l1342_134242

theorem mrs_wilsborough_tickets :
  ∀ (saved vip_ticket_cost regular_ticket_cost vip_tickets left : ℕ),
    saved = 500 →
    vip_ticket_cost = 100 →
    regular_ticket_cost = 50 →
    vip_tickets = 2 →
    left = 150 →
    (saved - left - (vip_tickets * vip_ticket_cost)) / regular_ticket_cost = 3 :=
by
  intros saved vip_ticket_cost regular_ticket_cost vip_tickets left
  sorry

end NUMINAMATH_GPT_mrs_wilsborough_tickets_l1342_134242


namespace NUMINAMATH_GPT_twelve_position_in_circle_l1342_134244

theorem twelve_position_in_circle (a : ℕ → ℕ) (h_cyclic : ∀ i, a (i + 20) = a i)
  (h_sum_six : ∀ i, a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) = 24)
  (h_first : a 1 = 1) :
  a 12 = 7 :=
sorry

end NUMINAMATH_GPT_twelve_position_in_circle_l1342_134244


namespace NUMINAMATH_GPT_min_editors_at_conference_l1342_134281

variable (x E : ℕ)

theorem min_editors_at_conference (h1 : x ≤ 26) 
    (h2 : 100 = 35 + E + x) 
    (h3 : 2 * x ≤ 100 - 35 - E + x) : 
    E ≥ 39 :=
by
  sorry

end NUMINAMATH_GPT_min_editors_at_conference_l1342_134281


namespace NUMINAMATH_GPT_triangle_ABC_is_right_triangle_l1342_134286

theorem triangle_ABC_is_right_triangle (A B C : ℝ) (hA : A = 68) (hB : B = 22) :
  A + B + C = 180 → C = 90 :=
by
  intro hABC
  sorry

end NUMINAMATH_GPT_triangle_ABC_is_right_triangle_l1342_134286


namespace NUMINAMATH_GPT_combined_percentage_grade4_l1342_134268

-- Definitions based on the given conditions
def Pinegrove_total_students : ℕ := 120
def Maplewood_total_students : ℕ := 180

def Pinegrove_grade4_percentage : ℕ := 10
def Maplewood_grade4_percentage : ℕ := 20

theorem combined_percentage_grade4 :
  let combined_total_students := Pinegrove_total_students + Maplewood_total_students
  let Pinegrove_grade4_students := Pinegrove_grade4_percentage * Pinegrove_total_students / 100
  let Maplewood_grade4_students := Maplewood_grade4_percentage * Maplewood_total_students / 100 
  let combined_grade4_students := Pinegrove_grade4_students + Maplewood_grade4_students
  (combined_grade4_students * 100 / combined_total_students) = 16 := by
  sorry

end NUMINAMATH_GPT_combined_percentage_grade4_l1342_134268


namespace NUMINAMATH_GPT_find_other_number_l1342_134237

noncomputable def HCF : ℕ := 14
noncomputable def LCM : ℕ := 396
noncomputable def one_number : ℕ := 154
noncomputable def product_of_numbers : ℕ := HCF * LCM

theorem find_other_number (other_number : ℕ) :
  HCF * LCM = one_number * other_number → other_number = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1342_134237


namespace NUMINAMATH_GPT_number_of_special_permutations_l1342_134204

noncomputable def count_special_permutations : ℕ :=
  (Nat.choose 12 6)

theorem number_of_special_permutations : count_special_permutations = 924 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_special_permutations_l1342_134204


namespace NUMINAMATH_GPT_students_move_bricks_l1342_134273

variable (a b c : ℕ)

theorem students_move_bricks (h : a * b * c ≠ 0) : 
  (by let efficiency := (c : ℚ) / (a * b);
      let total_work := (a : ℚ);
      let required_time := total_work / efficiency;
      exact required_time = (a^2 * b) / (c^2)) := sorry

end NUMINAMATH_GPT_students_move_bricks_l1342_134273


namespace NUMINAMATH_GPT_domain_f_2x_plus_1_eq_l1342_134276

-- Conditions
def domain_fx_plus_1 : Set ℝ := {x : ℝ | -2 < x ∧ x < -1}

-- Question and Correct Answer
theorem domain_f_2x_plus_1_eq :
  (∃ (x : ℝ), x ∈ domain_fx_plus_1) →
  {x : ℝ | -1 < x ∧ x < -1/2} = {x : ℝ | (2*x + 1 ∈ domain_fx_plus_1)} :=
by
  sorry

end NUMINAMATH_GPT_domain_f_2x_plus_1_eq_l1342_134276


namespace NUMINAMATH_GPT_find_Q_l1342_134252

variable (Q U P k : ℝ)

noncomputable def varies_directly_and_inversely : Prop :=
  P = k * (Q / U)

theorem find_Q (h : varies_directly_and_inversely Q U P k)
  (h1 : P = 6) (h2 : Q = 8) (h3 : U = 4)
  (h4 : P = 18) (h5 : U = 9) :
  Q = 54 :=
sorry

end NUMINAMATH_GPT_find_Q_l1342_134252


namespace NUMINAMATH_GPT_correct_statements_l1342_134232
noncomputable def is_pythagorean_triplet (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem correct_statements {a b c : ℕ} (h1 : is_pythagorean_triplet a b c) (h2 : a^2 + b^2 = c^2) :
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → a^2 + b^2 = c^2)) ∧
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → is_pythagorean_triplet (2 * a) (2 * b) (2 * c))) :=
by sorry

end NUMINAMATH_GPT_correct_statements_l1342_134232


namespace NUMINAMATH_GPT_absent_children_l1342_134284

theorem absent_children (A : ℕ) (h1 : 2 * 610 = (610 - A) * 4) : A = 305 := 
by sorry

end NUMINAMATH_GPT_absent_children_l1342_134284


namespace NUMINAMATH_GPT_divisor_of_51234_plus_3_l1342_134203

theorem divisor_of_51234_plus_3 : ∃ d : ℕ, d > 1 ∧ (51234 + 3) % d = 0 ∧ d = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_divisor_of_51234_plus_3_l1342_134203


namespace NUMINAMATH_GPT_angle_ACD_l1342_134226

theorem angle_ACD (E : ℝ) (arc_eq : ∀ (AB BC CD : ℝ), AB = BC ∧ BC = CD) (angle_eq : E = 40) : ∃ (ACD : ℝ), ACD = 15 :=
by
  sorry

end NUMINAMATH_GPT_angle_ACD_l1342_134226


namespace NUMINAMATH_GPT_Cherry_weekly_earnings_l1342_134255

theorem Cherry_weekly_earnings :
  let charge_small_cargo := 2.50
  let charge_large_cargo := 4.00
  let daily_small_cargo := 4
  let daily_large_cargo := 2
  let days_in_week := 7
  let daily_earnings := (charge_small_cargo * daily_small_cargo) + (charge_large_cargo * daily_large_cargo)
  let weekly_earnings := daily_earnings * days_in_week
  weekly_earnings = 126 := sorry

end NUMINAMATH_GPT_Cherry_weekly_earnings_l1342_134255


namespace NUMINAMATH_GPT_mgp_inequality_l1342_134222

theorem mgp_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a * b * c * d = 1) :
  (1 / Real.sqrt (1 / 2 + a + a * b + a * b * c) + 
   1 / Real.sqrt (1 / 2 + b + b * c + b * c * d) + 
   1 / Real.sqrt (1 / 2 + c + c * d + c * d * a) + 
   1 / Real.sqrt (1 / 2 + d + d * a + d * a * b)) 
  ≥ Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_mgp_inequality_l1342_134222


namespace NUMINAMATH_GPT_time_ratio_krishan_nandan_l1342_134234

theorem time_ratio_krishan_nandan 
  (N T k : ℝ) 
  (H1 : N * T = 6000) 
  (H2 : N * T + 6 * N * k * T = 78000) 
  : k = 2 := 
by 
sorry

end NUMINAMATH_GPT_time_ratio_krishan_nandan_l1342_134234


namespace NUMINAMATH_GPT_volume_of_polyhedron_l1342_134282

open Real

-- Define the conditions
def square_side : ℝ := 100  -- in cm, equivalent to 1 meter
def rectangle_length : ℝ := 40  -- in cm
def rectangle_width : ℝ := 20  -- in cm
def trapezoid_leg_length : ℝ := 130  -- in cm

-- Define the question as a theorem statement
theorem volume_of_polyhedron :
  ∃ V : ℝ, V = 552 :=
sorry

end NUMINAMATH_GPT_volume_of_polyhedron_l1342_134282


namespace NUMINAMATH_GPT_pure_gala_trees_l1342_134279

theorem pure_gala_trees (T F G : ℝ) (h1 : F + 0.10 * T = 221)
  (h2 : F = 0.75 * T) : G = T - F - 0.10 * T := 
by 
  -- We define G and show it equals 39
  have eq : T = F / 0.75 := by sorry
  have G_eq : G = T - F - 0.10 * T := by sorry 
  exact G_eq

end NUMINAMATH_GPT_pure_gala_trees_l1342_134279


namespace NUMINAMATH_GPT_max_min_y_l1342_134206

noncomputable def y (x : ℝ) : ℝ :=
  7 - 4 * (Real.sin x) * (Real.cos x) + 4 * (Real.cos x)^2 - 4 * (Real.cos x)^4

theorem max_min_y :
  (∃ x : ℝ, y x = 10) ∧ (∃ x : ℝ, y x = 6) := by
  sorry

end NUMINAMATH_GPT_max_min_y_l1342_134206


namespace NUMINAMATH_GPT_fuel_efficiency_l1342_134224

noncomputable def gas_cost_per_gallon : ℝ := 4
noncomputable def money_spent_on_gas : ℝ := 42
noncomputable def miles_traveled : ℝ := 336

theorem fuel_efficiency : (miles_traveled / (money_spent_on_gas / gas_cost_per_gallon)) = 32 := by
  sorry

end NUMINAMATH_GPT_fuel_efficiency_l1342_134224


namespace NUMINAMATH_GPT_total_animals_correct_l1342_134212

def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

def added_cows : ℕ := 3
def added_pigs : ℕ := 5
def added_goats : ℕ := 2

def total_cows : ℕ := initial_cows + added_cows
def total_pigs : ℕ := initial_pigs + added_pigs
def total_goats : ℕ := initial_goats + added_goats

def total_animals : ℕ := total_cows + total_pigs + total_goats

theorem total_animals_correct : total_animals = 21 := by
  sorry

end NUMINAMATH_GPT_total_animals_correct_l1342_134212


namespace NUMINAMATH_GPT_shortest_side_of_similar_triangle_l1342_134246

def Triangle (a b c : ℤ) : Prop := a^2 + b^2 = c^2
def SimilarTriangles (a b c a' b' c' : ℤ) : Prop := ∃ k : ℤ, k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c 

theorem shortest_side_of_similar_triangle (a b c a' b' c' : ℤ)
  (h₀ : Triangle 15 b 17)
  (h₁ : SimilarTriangles 15 b 17 a' b' c')
  (h₂ : c' = 51) : a' = 24 :=
by
  sorry

end NUMINAMATH_GPT_shortest_side_of_similar_triangle_l1342_134246


namespace NUMINAMATH_GPT_unique_four_digit_perfect_cube_divisible_by_16_and_9_l1342_134228

theorem unique_four_digit_perfect_cube_divisible_by_16_and_9 :
  ∃! n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℕ, n = k^3) ∧ n % 16 = 0 ∧ n % 9 = 0 ∧ n = 1728 :=
by sorry

end NUMINAMATH_GPT_unique_four_digit_perfect_cube_divisible_by_16_and_9_l1342_134228


namespace NUMINAMATH_GPT_area_ratio_of_shapes_l1342_134201

theorem area_ratio_of_shapes (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * π * r) (h2 : l = 3 * w) :
  (l * w) / (π * r^2) = (3 * π) / 16 :=
by sorry

end NUMINAMATH_GPT_area_ratio_of_shapes_l1342_134201


namespace NUMINAMATH_GPT_computation_result_l1342_134277

theorem computation_result :
  (3 + 6 - 12 + 24 + 48 - 96 + 192 - 384) / (6 + 12 - 24 + 48 + 96 - 192 + 384 - 768) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_computation_result_l1342_134277


namespace NUMINAMATH_GPT_simplify_expression_l1342_134278

theorem simplify_expression (x : ℝ) : 
  3 - 5*x - 6*x^2 + 9 + 11*x - 12*x^2 - 15 + 17*x + 18*x^2 - 2*x^3 = -2*x^3 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1342_134278


namespace NUMINAMATH_GPT_trainB_destination_time_l1342_134257

def trainA_speed : ℕ := 90
def trainB_speed : ℕ := 135
def trainA_time_after_meeting : ℕ := 9
def trainB_time_after_meeting (x : ℕ) : ℕ := 18 - 3 * x

theorem trainB_destination_time : (trainA_time_after_meeting, trainA_speed) = (9, 90) → 
  (trainB_speed, trainB_time_after_meeting 3) = (135, 3) := by
  sorry

end NUMINAMATH_GPT_trainB_destination_time_l1342_134257


namespace NUMINAMATH_GPT_problem_solving_ratio_l1342_134265

theorem problem_solving_ratio 
  (total_mcqs : ℕ) (total_psqs : ℕ)
  (written_mcqs_fraction : ℚ) (total_remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_mcqs_fraction = 2/5)
  (h4 : total_remaining_questions = 31) :
  (5 : ℚ) / 15 = (1 : ℚ) / 3 := 
by {
  -- given that 5 is the number of problem-solving questions already written,
  -- and 15 is the total number of problem-solving questions
  sorry
}

end NUMINAMATH_GPT_problem_solving_ratio_l1342_134265


namespace NUMINAMATH_GPT_problem1_problem2_l1342_134247

def f (x a : ℝ) := |x - 1| + |x - a|

/-
  Problem 1:
  Prove that if a = 3, the solution set to the inequality f(x) ≥ 4 is 
  {x | x ≤ 0 ∨ x ≥ 4}.
-/
theorem problem1 (f : ℝ → ℝ → ℝ) (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := 
sorry

/-
  Problem 2:
  Prove that for any x₁ ∈ ℝ, if f(x₁) ≥ 2 holds true, the range of values for
  a is {a | a ≥ 3 ∨ a ≤ -1}.
-/
theorem problem2 (f : ℝ → ℝ → ℝ) (x₁ : ℝ) :
  (∀ x₁ : ℝ, f x₁ a ≥ 2) ↔ (a ≥ 3 ∨ a ≤ -1) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1342_134247


namespace NUMINAMATH_GPT_odd_number_difference_of_squares_not_unique_l1342_134251

theorem odd_number_difference_of_squares_not_unique :
  ∀ n : ℤ, Odd n → ∃ X Y X' Y' : ℤ, (n = X^2 - Y^2) ∧ (n = X'^2 - Y'^2) ∧ (X ≠ X' ∨ Y ≠ Y') :=
sorry

end NUMINAMATH_GPT_odd_number_difference_of_squares_not_unique_l1342_134251


namespace NUMINAMATH_GPT_mary_final_books_l1342_134200

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end NUMINAMATH_GPT_mary_final_books_l1342_134200


namespace NUMINAMATH_GPT_find_a_l1342_134263

theorem find_a (a : ℝ) (h1 : a + 3 > 0) (h2 : abs (a + 3) = 5) : a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1342_134263


namespace NUMINAMATH_GPT_percentage_equivalence_l1342_134260

theorem percentage_equivalence (x : ℝ) :
  (70 / 100) * 600 = (x / 100) * 1050 → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_equivalence_l1342_134260


namespace NUMINAMATH_GPT_total_dress_designs_l1342_134272

def num_colors := 5
def num_patterns := 6
def num_sizes := 3

theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_dress_designs_l1342_134272


namespace NUMINAMATH_GPT_travel_time_on_third_day_l1342_134256

-- Definitions based on conditions
def speed_first_day : ℕ := 5
def time_first_day : ℕ := 7
def distance_first_day : ℕ := speed_first_day * time_first_day

def speed_second_day_part1 : ℕ := 6
def time_second_day_part1 : ℕ := 6
def distance_second_day_part1 : ℕ := speed_second_day_part1 * time_second_day_part1

def speed_second_day_part2 : ℕ := 3
def time_second_day_part2 : ℕ := 3
def distance_second_day_part2 : ℕ := speed_second_day_part2 * time_second_day_part2

def distance_second_day : ℕ := distance_second_day_part1 + distance_second_day_part2
def total_distance_first_two_days : ℕ := distance_first_day + distance_second_day

def total_distance : ℕ := 115
def distance_third_day : ℕ := total_distance - total_distance_first_two_days

def speed_third_day : ℕ := 7
def time_third_day : ℕ := distance_third_day / speed_third_day

-- The statement to be proven
theorem travel_time_on_third_day : time_third_day = 5 := by
  sorry

end NUMINAMATH_GPT_travel_time_on_third_day_l1342_134256


namespace NUMINAMATH_GPT_find_integer_values_l1342_134235

theorem find_integer_values (a : ℤ) (h : ∃ (n : ℤ), (a + 9) = n * (a + 6)) :
  a = -5 ∨ a = -7 ∨ a = -3 ∨ a = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_values_l1342_134235


namespace NUMINAMATH_GPT_value_of_g_3x_minus_5_l1342_134214

variable (R : Type) [Field R]
variable (g : R → R)
variable (x y : R)

-- Given condition: g(x) = -3 for all real numbers x
axiom g_is_constant : ∀ x : R, g x = -3

-- Prove that g(3x - 5) = -3
theorem value_of_g_3x_minus_5 : g (3 * x - 5) = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_g_3x_minus_5_l1342_134214


namespace NUMINAMATH_GPT_teacher_age_is_45_l1342_134215

def avg_age_of_students := 14
def num_students := 30
def avg_age_with_teacher := 15
def num_people_with_teacher := 31

def total_age_of_students := avg_age_of_students * num_students
def total_age_with_teacher := avg_age_with_teacher * num_people_with_teacher

theorem teacher_age_is_45 : (total_age_with_teacher - total_age_of_students = 45) :=
by
  sorry

end NUMINAMATH_GPT_teacher_age_is_45_l1342_134215


namespace NUMINAMATH_GPT_cost_of_each_pant_l1342_134240

theorem cost_of_each_pant (shirts pants : ℕ) (cost_shirt cost_total : ℕ) (cost_pant : ℕ) :
  shirts = 10 ∧ pants = (shirts / 2) ∧ cost_shirt = 6 ∧ cost_total = 100 →
  (shirts * cost_shirt + pants * cost_pant = cost_total) →
  cost_pant = 8 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_each_pant_l1342_134240


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_l1342_134294

variable (b j s : ℤ) -- rates in km/h
-- conditions
def ed_condition : Prop := 3 * b + 4 * j + 2 * s = 86
def sue_condition : Prop := 5 * b + 2 * j + 4 * s = 110

theorem sum_of_squares_of_rates (b j s : ℤ) (hEd : ed_condition b j s) (hSue : sue_condition b j s) : 
  b^2 + j^2 + s^2 = 3349 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_l1342_134294


namespace NUMINAMATH_GPT_no_solution_equation_l1342_134243

theorem no_solution_equation (m : ℝ) : 
  ¬∃ x : ℝ, x ≠ 2 ∧ (x - 3) / (x - 2) = m / (2 - x) → m = 1 := 
by 
  sorry

end NUMINAMATH_GPT_no_solution_equation_l1342_134243


namespace NUMINAMATH_GPT_isosceles_triangle_side_length_l1342_134229

theorem isosceles_triangle_side_length (P : ℕ := 53) (base : ℕ := 11) (x : ℕ)
  (h1 : x + x + base = P) : x = 21 :=
by {
  -- The proof goes here.
  sorry
}

end NUMINAMATH_GPT_isosceles_triangle_side_length_l1342_134229


namespace NUMINAMATH_GPT_min_employees_needed_l1342_134258

-- Definitions for the problem conditions
def hardware_employees : ℕ := 150
def software_employees : ℕ := 130
def both_employees : ℕ := 50

-- Statement of the proof problem
theorem min_employees_needed : hardware_employees + software_employees - both_employees = 230 := 
by 
  -- Calculation skipped with sorry
  sorry

end NUMINAMATH_GPT_min_employees_needed_l1342_134258


namespace NUMINAMATH_GPT_sum_of_squares_of_geometric_progression_l1342_134213

theorem sum_of_squares_of_geometric_progression 
  {b_1 q S_1 S_2 : ℝ} 
  (h1 : |q| < 1) 
  (h2 : S_1 = b_1 / (1 - q))
  (h3 : S_2 = b_1 / (1 + q)) : 
  (b_1^2 / (1 - q^2)) = S_1 * S_2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_geometric_progression_l1342_134213


namespace NUMINAMATH_GPT_geometric_sequence_b_value_l1342_134225

theorem geometric_sequence_b_value :
  ∀ (a b c : ℝ),
  (a = 5 + 2 * Real.sqrt 6) →
  (c = 5 - 2 * Real.sqrt 6) →
  (b * b = a * c) →
  (b = 1 ∨ b = -1) :=
by
  intros a b c ha hc hgeometric
  sorry

end NUMINAMATH_GPT_geometric_sequence_b_value_l1342_134225


namespace NUMINAMATH_GPT_alpha_range_midpoint_trajectory_l1342_134259

noncomputable def circle_parametric_eqn (θ : ℝ) : ℝ × ℝ :=
  ⟨Real.cos θ, Real.sin θ⟩

theorem alpha_range (α : ℝ) (h1 : 0 < α ∧ α < 2 * Real.pi) :
  (Real.tan α) > 1 ∨ (Real.tan α) < -1 ↔ (Real.pi / 4 < α ∧ α < 3 * Real.pi / 4) ∨ 
                                          (5 * Real.pi / 4 < α ∧ α < 7 * Real.pi / 4) := 
  sorry

theorem midpoint_trajectory (m : ℝ) (h2 : -1 < m ∧ m < 1) :
  ∃ x y : ℝ, x = (Real.sqrt 2 * m) / (m^2 + 1) ∧ 
             y = -(Real.sqrt 2 * m^2) / (m^2 + 1) :=
  sorry

end NUMINAMATH_GPT_alpha_range_midpoint_trajectory_l1342_134259


namespace NUMINAMATH_GPT_min_a2_k2b2_l1342_134261

variable (a b t k : ℝ)
variable (hk : 0 < k)
variable (h : a + k * b = t)

theorem min_a2_k2b2 (a b t k : ℝ) (hk : 0 < k) (h : a + k * b = t) :
  a^2 + (k * b)^2 ≥ (1 + k^2) * (t^2) / ((1 + k)^2) :=
sorry

end NUMINAMATH_GPT_min_a2_k2b2_l1342_134261


namespace NUMINAMATH_GPT_three_colored_flag_l1342_134285

theorem three_colored_flag (colors : Finset ℕ) (h : colors.card = 6) : 
  (∃ top middle bottom : ℕ, top ≠ middle ∧ top ≠ bottom ∧ middle ≠ bottom ∧ 
                            top ∈ colors ∧ middle ∈ colors ∧ bottom ∈ colors) → 
  colors.card * (colors.card - 1) * (colors.card - 2) = 120 :=
by 
  intro h_exists
  exact sorry

end NUMINAMATH_GPT_three_colored_flag_l1342_134285


namespace NUMINAMATH_GPT_action_figure_prices_l1342_134275

noncomputable def prices (x y z w : ℝ) : Prop :=
  12 * x + 8 * y + 5 * z + 10 * w = 220 ∧
  x / 4 = y / 3 ∧
  x / 4 = z / 2 ∧
  x / 4 = w / 1

theorem action_figure_prices :
  ∃ x y z w : ℝ, prices x y z w ∧
    x = 220 / 23 ∧
    y = (3 / 4) * (220 / 23) ∧
    z = (1 / 2) * (220 / 23) ∧
    w = (1 / 4) * (220 / 23) :=
  sorry

end NUMINAMATH_GPT_action_figure_prices_l1342_134275


namespace NUMINAMATH_GPT_slope_of_line_with_sine_of_angle_l1342_134289

theorem slope_of_line_with_sine_of_angle (α : ℝ) 
  (hα₁ : 0 ≤ α) (hα₂ : α < Real.pi) 
  (h_sin : Real.sin α = Real.sqrt 3 / 2) : 
  ∃ k : ℝ, k = Real.tan α ∧ (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_with_sine_of_angle_l1342_134289


namespace NUMINAMATH_GPT_find_z_l1342_134221

theorem find_z : 
    ∃ z : ℝ, ( ( 2 ^ 5 ) * ( 9 ^ 2 ) ) / ( z * ( 3 ^ 5 ) ) = 0.16666666666666666 ↔ z = 64 :=
by
    sorry

end NUMINAMATH_GPT_find_z_l1342_134221


namespace NUMINAMATH_GPT_find_c_l1342_134218

def sum_of_digits (n : ℕ) : ℕ := (n.digits 10).sum

theorem find_c :
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  c = 5 :=
by
  let a := sum_of_digits (4568 ^ 777)
  let b := sum_of_digits a
  let c := sum_of_digits b
  sorry

end NUMINAMATH_GPT_find_c_l1342_134218


namespace NUMINAMATH_GPT_regular_nine_sided_polygon_has_27_diagonals_l1342_134236

def is_regular_polygon (n : ℕ) : Prop := n ≥ 3

def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem regular_nine_sided_polygon_has_27_diagonals :
  is_regular_polygon 9 →
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_GPT_regular_nine_sided_polygon_has_27_diagonals_l1342_134236


namespace NUMINAMATH_GPT_nat_numbers_l1342_134217

theorem nat_numbers (n : ℕ) (h1 : n ≥ 2) (h2 : ∃a b : ℕ, a * b = n ∧ ∀ c : ℕ, 1 < c ∧ c ∣ n → a ≤ c ∧ n = a^2 + b^2) : 
  n = 5 ∨ n = 8 ∨ n = 20 :=
by
  sorry

end NUMINAMATH_GPT_nat_numbers_l1342_134217


namespace NUMINAMATH_GPT_triangle_side_length_difference_l1342_134280

theorem triangle_side_length_difference (x : ℤ) :
  (2 < x ∧ x < 16) → (∀ y : ℤ, (2 < y ∧ y < 16) → (3 ≤ y) ∧ (y ≤ 15)) →
  (∀ z : ℤ, (3 ≤ z ∨ z ≤ 15) → (15 - 3 = 12)) := by
  sorry

end NUMINAMATH_GPT_triangle_side_length_difference_l1342_134280


namespace NUMINAMATH_GPT_max_n_is_2_l1342_134262

def is_prime_seq (q : ℕ → ℕ) : Prop :=
  ∀ i, Nat.Prime (q i)

def gen_seq (q0 : ℕ) : ℕ → ℕ
  | 0 => q0
  | (i + 1) => (gen_seq q0 i - 1)^3 + 3

theorem max_n_is_2 (q0 : ℕ) (hq0 : q0 > 0) :
  ∀ (q1 q2 : ℕ), q1 = gen_seq q0 1 → q2 = gen_seq q0 2 → 
  is_prime_seq (gen_seq q0) → q2 = (q1 - 1)^3 + 3 := 
  sorry

end NUMINAMATH_GPT_max_n_is_2_l1342_134262


namespace NUMINAMATH_GPT_fraction_product_l1342_134292

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_l1342_134292


namespace NUMINAMATH_GPT_range_of_a_l1342_134223

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| - 2 * a + 2 < 0) → (a > 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1342_134223


namespace NUMINAMATH_GPT_non_organic_chicken_price_l1342_134205

theorem non_organic_chicken_price :
  ∀ (x : ℝ), (0.75 * x = 9) → (2 * (0.9 * x) = 21.6) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_non_organic_chicken_price_l1342_134205


namespace NUMINAMATH_GPT_problem_l1342_134220

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end NUMINAMATH_GPT_problem_l1342_134220


namespace NUMINAMATH_GPT_factorization_correct_l1342_134264

theorem factorization_correct (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1342_134264


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1342_134293

-- 1. Prove that the solutions to the equation x^2 - 4x - 1 = 0 are x = 2 + sqrt(5) and x = 2 - sqrt(5)
theorem solve_quadratic_1 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
sorry

-- 2. Prove that the solutions to the equation 3(x - 1)^2 = 2(x - 1) are x = 1 and x = 5/3
theorem solve_quadratic_2 (x : ℝ) : 3 * (x - 1) ^ 2 = 2 * (x - 1) ↔ x = 1 ∨ x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l1342_134293


namespace NUMINAMATH_GPT_difference_of_squares_l1342_134238

noncomputable def product_of_consecutive_integers (n : ℕ) := n * (n + 1)

theorem difference_of_squares (h : ∃ n : ℕ, product_of_consecutive_integers n = 2720) :
  ∃ a b : ℕ, product_of_consecutive_integers a = 2720 ∧ (b = a + 1) ∧ (b * b - a * a = 103) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1342_134238


namespace NUMINAMATH_GPT_marbles_problem_l1342_134291

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end NUMINAMATH_GPT_marbles_problem_l1342_134291


namespace NUMINAMATH_GPT_find_a_l1342_134230

-- Definitions from conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2
def directrix : ℝ := 1

-- Statement to prove
theorem find_a (a : ℝ) (h : directrix = 1) : a = -1/4 :=
sorry

end NUMINAMATH_GPT_find_a_l1342_134230


namespace NUMINAMATH_GPT_final_price_is_correct_l1342_134241

def original_price : ℝ := 450
def discounts : List ℝ := [0.10, 0.20, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

noncomputable def final_sale_price (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem final_price_is_correct:
  final_sale_price original_price discounts = 307.8 :=
by
  sorry

end NUMINAMATH_GPT_final_price_is_correct_l1342_134241
