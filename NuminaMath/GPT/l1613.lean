import Mathlib

namespace angle_same_terminal_side_l1613_161351

theorem angle_same_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 → α = 330 :=
by
  sorry

end angle_same_terminal_side_l1613_161351


namespace sum_of_cubes_correct_l1613_161324

noncomputable def expression_for_sum_of_cubes (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) : Prop :=
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + b^3 * d^3) / (a * b * c * d)

theorem sum_of_cubes_correct (x y z w a b c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (hxy : x * y = a) (hxz : x * z = b) (hyz : y * z = c) (hxw : x * w = d) :
  expression_for_sum_of_cubes x y z w a b c d hx hy hz hw ha hb hc hd hxy hxz hyz hxw :=
sorry

end sum_of_cubes_correct_l1613_161324


namespace paper_plate_cup_cost_l1613_161350

variables (P C : ℝ)

theorem paper_plate_cup_cost (h : 100 * P + 200 * C = 6) : 20 * P + 40 * C = 1.20 :=
by sorry

end paper_plate_cup_cost_l1613_161350


namespace percent_change_range_l1613_161381

-- Define initial conditions
def initial_yes_percent : ℝ := 0.60
def initial_no_percent : ℝ := 0.40
def final_yes_percent : ℝ := 0.80
def final_no_percent : ℝ := 0.20

-- Define the key statement to prove
theorem percent_change_range : 
  ∃ y_min y_max : ℝ, 
  y_min = 0.20 ∧ 
  y_max = 0.60 ∧ 
  (y_max - y_min = 0.40) :=
sorry

end percent_change_range_l1613_161381


namespace larger_angle_measure_l1613_161370

theorem larger_angle_measure (x : ℝ) (h : 4 * x + 5 * x = 180) : 5 * x = 100 :=
by
  sorry

end larger_angle_measure_l1613_161370


namespace cycle_reappear_l1613_161305

/-- Given two sequences with cycle lengths 6 and 4, prove the sequences will align on line number 12 -/
theorem cycle_reappear (l1 l2 : ℕ) (h1 : l1 = 6) (h2 : l2 = 4) :
  Nat.lcm l1 l2 = 12 := by
  sorry

end cycle_reappear_l1613_161305


namespace maria_min_score_fifth_term_l1613_161325

theorem maria_min_score_fifth_term (score1 score2 score3 score4 : ℕ) (avg_required : ℕ) 
  (h1 : score1 = 84) (h2 : score2 = 80) (h3 : score3 = 82) (h4 : score4 = 78)
  (h_avg_required : avg_required = 85) :
  ∃ x : ℕ, x ≥ 101 :=
by
  sorry

end maria_min_score_fifth_term_l1613_161325


namespace sum_a10_a11_l1613_161318

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q)

theorem sum_a10_a11 {a : ℕ → ℝ} (h_seq : geometric_sequence a)
  (h1 : a 1 + a 2 = 2)
  (h4 : a 4 + a 5 = 4) :
  a 10 + a 11 = 16 :=
by {
  sorry
}

end sum_a10_a11_l1613_161318


namespace find_angle_D_l1613_161380

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : ∃ B_adj, B_adj = 60 ∧ A + B_adj + B = 180) : D = 25 :=
sorry

end find_angle_D_l1613_161380


namespace find_annual_interest_rate_l1613_161313

-- Define the given conditions
def principal : ℝ := 10000
def time : ℝ := 1  -- since 12 months is 1 year for annual rate
def simple_interest : ℝ := 800

-- Define the annual interest rate to be proved
def annual_interest_rate : ℝ := 0.08

-- The theorem stating the problem
theorem find_annual_interest_rate (P : ℝ) (T : ℝ) (SI : ℝ) : 
  P = principal → 
  T = time → 
  SI = simple_interest → 
  SI = P * annual_interest_rate * T := 
by
  intros hP hT hSI
  rw [hP, hT, hSI]
  unfold annual_interest_rate
  -- here's where we skip the proof
  sorry

end find_annual_interest_rate_l1613_161313


namespace no_solution_for_inequality_system_l1613_161321

theorem no_solution_for_inequality_system (x : ℝ) : 
  ¬ ((2 * x + 3 ≥ x + 11) ∧ (((2 * x + 5) / 3 - 1) < (2 - x))) :=
by
  sorry

end no_solution_for_inequality_system_l1613_161321


namespace find_n_l1613_161340

-- Given Variables
variables (n x y : ℝ)

-- Given Conditions
axiom h1 : n * x = 6 * y
axiom h2 : x * y ≠ 0
axiom h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998

-- Conclusion
theorem find_n : n = 5 := sorry

end find_n_l1613_161340


namespace smallest_multiple_of_6_and_15_l1613_161302

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l1613_161302


namespace ben_gave_18_fish_l1613_161306

variable (initial_fish : ℕ) (total_fish : ℕ) (given_fish : ℕ)

theorem ben_gave_18_fish
    (h1 : initial_fish = 31)
    (h2 : total_fish = 49)
    (h3 : total_fish = initial_fish + given_fish) :
    given_fish = 18 :=
by
  sorry

end ben_gave_18_fish_l1613_161306


namespace functions_eq_l1613_161336

open Function

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem functions_eq (h_surj : Surjective f) (h_inj : Injective g) (h_ge : ∀ n : ℕ, f n ≥ g n) : ∀ n : ℕ, f n = g n :=
sorry

end functions_eq_l1613_161336


namespace sufficient_but_not_necessary_condition_l1613_161377

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 2

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) : 
  a ≤ 0 :=
sorry

end sufficient_but_not_necessary_condition_l1613_161377


namespace Intersection_A_B_l1613_161353

open Set

theorem Intersection_A_B :
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  A ∩ B = {x : ℝ | -3 < x ∧ x < 1} := by
  let A := {x : ℝ | 2 * x + 1 < 3}
  let B := {x : ℝ | -3 < x ∧ x < 2}
  show A ∩ B = {x : ℝ | -3 < x ∧ x < 1}
  sorry

end Intersection_A_B_l1613_161353


namespace ratio_areas_l1613_161386

-- Define the perimeter P
variable (P : ℝ) (hP : P > 0)

-- Define the side lengths
noncomputable def side_length_square := P / 4
noncomputable def side_length_triangle := P / 3

-- Define the radius of the circumscribed circle for the square
noncomputable def radius_square := (P * Real.sqrt 2) / 8
-- Define the area of the circumscribed circle for the square
noncomputable def area_circle_square := Real.pi * (radius_square P)^2

-- Define the radius of the circumscribed circle for the equilateral triangle
noncomputable def radius_triangle := (P * Real.sqrt 3) / 9 
-- Define the area of the circumscribed circle for the equilateral triangle
noncomputable def area_circle_triangle := Real.pi * (radius_triangle P)^2

-- Prove the ratio of the areas is 27/32
theorem ratio_areas (P : ℝ) (hP : P > 0) : 
  (area_circle_square P / area_circle_triangle P) = (27 / 32) := by
  sorry

end ratio_areas_l1613_161386


namespace probability_of_both_chinese_books_l1613_161387

def total_books := 5
def chinese_books := 3
def math_books := 2

theorem probability_of_both_chinese_books (select_books : ℕ) 
  (total_choices : ℕ) (favorable_choices : ℕ) :
  select_books = 2 →
  total_choices = (Nat.choose total_books select_books) →
  favorable_choices = (Nat.choose chinese_books select_books) →
  (favorable_choices : ℚ) / (total_choices : ℚ) = 3 / 10 := by
  intros h1 h2 h3
  sorry

end probability_of_both_chinese_books_l1613_161387


namespace line_bisects_circle_area_l1613_161330

theorem line_bisects_circle_area (b : ℝ) :
  (∀ x y : ℝ, y = 2 * x + b ↔ x^2 + y^2 - 2 * x - 4 * y + 4 = 0) → b = 0 :=
by
  sorry

end line_bisects_circle_area_l1613_161330


namespace tank_weight_when_full_l1613_161308

theorem tank_weight_when_full (p q : ℝ) (x y : ℝ)
  (h1 : x + (3/4) * y = p)
  (h2 : x + (1/3) * y = q) :
  x + y = (8/5) * p - (8/5) * q :=
by
  sorry

end tank_weight_when_full_l1613_161308


namespace sum_lent_l1613_161329

theorem sum_lent (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ)
  (hR: R = 4) 
  (hT: T = 8) 
  (hI1 : I = P - 306) 
  (hI2 : I = P * R * T / 100) :
  P = 450 :=
by
  sorry

end sum_lent_l1613_161329


namespace inequality_for_a_and_b_l1613_161367

theorem inequality_for_a_and_b (a b : ℝ) : 
  (1 / 3 * a - b) ≤ 5 :=
sorry

end inequality_for_a_and_b_l1613_161367


namespace total_area_of_region_l1613_161393

variable (a b c d : ℝ)
variable (ha : a > 0) (hb : b > 0) (hd : d > 0)

theorem total_area_of_region : (a + b) * d + (1 / 2) * Real.pi * c ^ 2 = (a + b) * d + (1 / 2) * Real.pi * c ^ 2 := by
  sorry

end total_area_of_region_l1613_161393


namespace unit_price_of_first_batch_minimum_selling_price_l1613_161326

-- Proof Problem 1
theorem unit_price_of_first_batch :
  (∃ x : ℝ, (3200 / x) * 2 = 7200 / (x + 10) ∧ x = 80) := 
  sorry

-- Proof Problem 2
theorem minimum_selling_price (x : ℝ) (hx : x = 80) :
  (40 * x + 80 * (x + 10) - 3200 - 7200 + 20 * 0.8 * x ≥ 3520) → 
  (∃ y : ℝ, y ≥ 120) :=
  sorry

end unit_price_of_first_batch_minimum_selling_price_l1613_161326


namespace traci_flour_brought_l1613_161394

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end traci_flour_brought_l1613_161394


namespace find_f3_l1613_161323

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 - c * x + 2

theorem find_f3 (a b c : ℝ)
  (h1 : f a b c (-3) = 9) :
  f a b c 3 = -5 :=
by
  sorry

end find_f3_l1613_161323


namespace solve_equation_l1613_161385

theorem solve_equation :
  ∀ x : ℝ, 
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     16 / (Real.sqrt (x - 10) + 10) = 0)
    ↔ 
    x = 1841 / 121 ∨ x = 190 / 9 :=
by
  sorry

end solve_equation_l1613_161385


namespace sqrt_31_estimate_l1613_161345

theorem sqrt_31_estimate : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := 
by
  sorry

end sqrt_31_estimate_l1613_161345


namespace train_speed_kmh_l1613_161316

theorem train_speed_kmh (T P: ℝ) (L: ℝ):
  (T = L + 320) ∧ (L = 18 * P) ->
  P = 20 -> 
  P * 3.6 = 72 := 
by
  sorry

end train_speed_kmh_l1613_161316


namespace eighth_term_matchstick_count_l1613_161359

def matchstick_sequence (n : ℕ) : ℕ := (n + 1) * 3

theorem eighth_term_matchstick_count : matchstick_sequence 8 = 27 :=
by
  -- the proof will go here
  sorry

end eighth_term_matchstick_count_l1613_161359


namespace selling_price_l1613_161396

noncomputable def total_cost_first_mixture : ℝ := 27 * 150
noncomputable def total_cost_second_mixture : ℝ := 36 * 125
noncomputable def total_cost_third_mixture : ℝ := 18 * 175
noncomputable def total_cost_fourth_mixture : ℝ := 24 * 120

noncomputable def total_cost : ℝ := total_cost_first_mixture + total_cost_second_mixture + total_cost_third_mixture + total_cost_fourth_mixture

noncomputable def profit_first_mixture : ℝ := 0.4 * total_cost_first_mixture
noncomputable def profit_second_mixture : ℝ := 0.3 * total_cost_second_mixture
noncomputable def profit_third_mixture : ℝ := 0.2 * total_cost_third_mixture
noncomputable def profit_fourth_mixture : ℝ := 0.25 * total_cost_fourth_mixture

noncomputable def total_profit : ℝ := profit_first_mixture + profit_second_mixture + profit_third_mixture + profit_fourth_mixture

noncomputable def total_weight : ℝ := 27 + 36 + 18 + 24
noncomputable def total_selling_price : ℝ := total_cost + total_profit

noncomputable def selling_price_per_kg : ℝ := total_selling_price / total_weight

theorem selling_price : selling_price_per_kg = 180 := by
  sorry

end selling_price_l1613_161396


namespace christmas_bonus_remainder_l1613_161315

theorem christmas_bonus_remainder (X : ℕ) (h : X % 5 = 2) : (3 * X) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l1613_161315


namespace least_n_satisfies_condition_l1613_161311

theorem least_n_satisfies_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, m > 0 → (m < n) → ¬(1 / (m : ℝ) - 1 / (m + 1) < 1 / 15)) ∧
    (1 / (n : ℝ) - 1 / (n + 1) < 1 / 15) :=
sorry

end least_n_satisfies_condition_l1613_161311


namespace cylinder_heights_relation_l1613_161335

variables {r1 r2 h1 h2 : ℝ}

theorem cylinder_heights_relation 
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = (6 / 5) * r1) :
  h1 = 1.44 * h2 :=
by sorry

end cylinder_heights_relation_l1613_161335


namespace smaller_number_l1613_161374

theorem smaller_number (x y : ℝ) (h1 : x - y = 1650) (h2 : 0.075 * x = 0.125 * y) : y = 2475 := 
sorry

end smaller_number_l1613_161374


namespace notebook_cost_proof_l1613_161314

-- Let n be the cost of the notebook and p be the cost of the pen.
variable (n p : ℝ)

-- Conditions:
def total_cost : Prop := n + p = 2.50
def notebook_more_pen : Prop := n = 2 + p

-- Theorem: Prove that the cost of the notebook is $2.25
theorem notebook_cost_proof (h1 : total_cost n p) (h2 : notebook_more_pen n p) : n = 2.25 := 
by 
  sorry

end notebook_cost_proof_l1613_161314


namespace expected_value_in_classroom_l1613_161341

noncomputable def expected_pairs_next_to_each_other (boys girls : ℕ) : ℕ :=
  if boys = 9 ∧ girls = 14 ∧ boys + girls = 23 then
    10 -- Based on provided conditions and conclusion
  else
    0

theorem expected_value_in_classroom :
  expected_pairs_next_to_each_other 9 14 = 10 :=
by
  sorry

end expected_value_in_classroom_l1613_161341


namespace droid_weekly_coffee_consumption_l1613_161379

noncomputable def weekly_consumption_A : ℕ :=
  (3 * 5) + 4 + 2 + 1 -- Weekdays + Saturday + Sunday + Monday increase

noncomputable def weekly_consumption_B : ℕ :=
  (2 * 5) + 3 + (1 - 1 / 2) -- Weekdays + Saturday + Sunday decrease

noncomputable def weekly_consumption_C : ℕ :=
  (1 * 5) + 2 + 1 -- Weekdays + Saturday + Sunday

theorem droid_weekly_coffee_consumption :
  weekly_consumption_A = 22 ∧ weekly_consumption_B = 14 ∧ weekly_consumption_C = 8 :=
by 
  sorry

end droid_weekly_coffee_consumption_l1613_161379


namespace find_a_of_parabola_l1613_161369

theorem find_a_of_parabola (a b c : ℤ) (h_vertex : (2, 5) = (2, 5)) (h_point : 8 = a * (3 - 2) ^ 2 + 5) :
  a = 3 :=
sorry

end find_a_of_parabola_l1613_161369


namespace correct_statements_count_l1613_161390

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end correct_statements_count_l1613_161390


namespace inequality_bounds_l1613_161310

theorem inequality_bounds (x y : ℝ) : |y - 3 * x| < 2 * x ↔ x > 0 ∧ x < y ∧ y < 5 * x := by
  sorry

end inequality_bounds_l1613_161310


namespace haji_mother_tough_weeks_l1613_161304

/-- Let's define all the conditions: -/
def tough_week_revenue : ℕ := 800
def good_week_revenue : ℕ := 2 * tough_week_revenue
def number_of_good_weeks : ℕ := 5
def total_revenue : ℕ := 10400

/-- Let's define the proofs for intermediate steps: -/
def good_weeks_revenue : ℕ := number_of_good_weeks * good_week_revenue
def tough_weeks_revenue : ℕ := total_revenue - good_weeks_revenue
def number_of_tough_weeks : ℕ := tough_weeks_revenue / tough_week_revenue

/-- Now the theorem which states that the number of tough weeks is 3. -/
theorem haji_mother_tough_weeks : number_of_tough_weeks = 3 := by
  sorry

end haji_mother_tough_weeks_l1613_161304


namespace common_pasture_area_l1613_161357

variable (Area_Ivanov Area_Petrov Area_Sidorov Area_Vasilev Area_Ermolaev : ℝ)
variable (Common_Pasture : ℝ)

theorem common_pasture_area :
  Area_Ivanov = 24 ∧
  Area_Petrov = 28 ∧
  Area_Sidorov = 10 ∧
  Area_Vasilev = 20 ∧
  Area_Ermolaev = 30 →
  Common_Pasture = 17.5 :=
sorry

end common_pasture_area_l1613_161357


namespace proof_problem_l1613_161383

def f (a b c : ℕ) : ℕ :=
  a * 100 + b * 10 + c

def special_op (a b c : ℕ) : ℕ :=
  f (a * b) (b * c / 10) (b * c % 10)

theorem proof_problem :
  special_op 5 7 4 - special_op 7 4 5 = 708 := 
    sorry

end proof_problem_l1613_161383


namespace simplify_to_ap_minus_b_l1613_161388

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((7*p + 3) - 3*p * 2) * 4 + (5 - 2 / 4) * (8*p - 12)

theorem simplify_to_ap_minus_b (p : ℝ) :
  simplify_expression p = 40 * p - 42 :=
by
  -- Proof steps would go here
  sorry

end simplify_to_ap_minus_b_l1613_161388


namespace job_pay_per_pound_l1613_161303

def p := 2
def M := 8 -- Monday
def T := 3 * M -- Tuesday
def W := 0 -- Wednesday
def R := 18 -- Thursday
def total_picked := M + T + W + R -- total berries picked
def money := 100 -- total money wanted

theorem job_pay_per_pound :
  total_picked = 50 → p = money / total_picked :=
by
  intro h
  rw [h]
  norm_num
  exact rfl

end job_pay_per_pound_l1613_161303


namespace interest_percentage_calculation_l1613_161375

-- Definitions based on problem conditions
def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℕ := 12

-- Theorem statement:
theorem interest_percentage_calculation :
  let total_paid := down_payment + (monthly_payment * number_of_monthly_payments)
  let interest_paid := total_paid - purchase_price
  let interest_percent := (interest_paid / purchase_price) * 100
  interest_percent = 18.2 :=
by sorry

end interest_percentage_calculation_l1613_161375


namespace sum_of_reciprocals_of_shifted_roots_l1613_161378

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) (h : ∀ x, x^3 - x + 2 = 0 → x = a ∨ x = b ∨ x = c) :
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 :=
by
  sorry

end sum_of_reciprocals_of_shifted_roots_l1613_161378


namespace f_increasing_on_Ioo_l1613_161389

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_on_Ioo : ∀ x y : ℝ, x < y → f x < f y :=
sorry

end f_increasing_on_Ioo_l1613_161389


namespace slope_of_line_l1613_161395

def point1 : (ℤ × ℤ) := (-4, 6)
def point2 : (ℤ × ℤ) := (3, -4)

def slope_formula (p1 p2 : (ℤ × ℤ)) : ℚ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst : ℚ)

theorem slope_of_line : slope_formula point1 point2 = -10 / 7 := by
  sorry

end slope_of_line_l1613_161395


namespace books_on_each_shelf_l1613_161384

-- Define the conditions and the problem statement
theorem books_on_each_shelf :
  ∀ (M P : ℕ), 
  -- Conditions
  (5 * M + 4 * P = 72) ∧ (M = P) ∧ (∃ B : ℕ, M = B ∧ P = B) ->
  -- Conclusion
  (∃ B : ℕ, B = 8) :=
by
  sorry

end books_on_each_shelf_l1613_161384


namespace homogeneous_variances_l1613_161346

noncomputable def sample_sizes : (ℕ × ℕ × ℕ) := (9, 13, 15)
noncomputable def sample_variances : (ℝ × ℝ × ℝ) := (3.2, 3.8, 6.3)
noncomputable def significance_level : ℝ := 0.05
noncomputable def degrees_of_freedom : ℕ := 2
noncomputable def V : ℝ := 1.43
noncomputable def critical_value : ℝ := 6.0

theorem homogeneous_variances :
  V < critical_value :=
by
  sorry

end homogeneous_variances_l1613_161346


namespace parabolas_intersect_at_points_l1613_161331

theorem parabolas_intersect_at_points :
  ∀ (x y : ℝ), (y = 3 * x^2 - 12 * x - 9) ↔ (y = 2 * x^2 - 8 * x + 5) →
  (x, y) = (2 + 3 * Real.sqrt 2, 66 - 36 * Real.sqrt 2) ∨ (x, y) = (2 - 3 * Real.sqrt 2, 66 + 36 * Real.sqrt 2) :=
by
  sorry

end parabolas_intersect_at_points_l1613_161331


namespace each_friend_should_contribute_equally_l1613_161371

-- Define the total expenses and number of friends
def total_expenses : ℝ := 35 + 9 + 9 + 6 + 2
def number_of_friends : ℕ := 5

-- Define the expected contribution per friend
def expected_contribution : ℝ := 12.20

-- Theorem statement
theorem each_friend_should_contribute_equally :
  total_expenses / number_of_friends = expected_contribution :=
by
  sorry

end each_friend_should_contribute_equally_l1613_161371


namespace desired_cost_of_mixture_l1613_161344

theorem desired_cost_of_mixture 
  (w₈ : ℝ) (c₈ : ℝ) -- weight and cost per pound of the $8 candy
  (w₅ : ℝ) (c₅ : ℝ) -- weight and cost per pound of the $5 candy
  (total_w : ℝ) (desired_cost : ℝ) -- total weight and desired cost per pound of the mixture
  (h₁ : w₈ = 30) (h₂ : c₈ = 8) 
  (h₃ : w₅ = 60) (h₄ : c₅ = 5)
  (h₅ : total_w = w₈ + w₅)
  (h₆ : desired_cost = (w₈ * c₈ + w₅ * c₅) / total_w) :
  desired_cost = 6 := 
by
  sorry

end desired_cost_of_mixture_l1613_161344


namespace product_is_correct_l1613_161312

def number : ℕ := 3460
def multiplier : ℕ := 12
def correct_product : ℕ := 41520

theorem product_is_correct : multiplier * number = correct_product := by
  sorry

end product_is_correct_l1613_161312


namespace initial_price_correct_l1613_161334

-- Definitions based on the conditions
def initial_price : ℝ := 3  -- Rs. 3 per kg
def new_price : ℝ := 5      -- Rs. 5 per kg
def reduction_in_consumption : ℝ := 0.4  -- 40%

-- The main theorem we need to prove
theorem initial_price_correct :
  initial_price = 3 :=
sorry

end initial_price_correct_l1613_161334


namespace number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l1613_161397

noncomputable def a (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem number_of_diagonals_pentagon : a 5 = 5 := sorry

theorem difference_hexagon_pentagon : a 6 - a 5 = 4 := sorry

theorem difference_successive_polygons (n : ℕ) (h : 4 ≤ n) : a (n + 1) - a n = n - 1 := sorry

end number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l1613_161397


namespace jumping_contest_l1613_161391

variables (G F M K : ℤ)

-- Define the conditions
def condition_1 : Prop := G = 39
def condition_2 : Prop := G = F + 19
def condition_3 : Prop := M = F - 12
def condition_4 : Prop := K = 2 * F - 5

-- The theorem asserting the final distances
theorem jumping_contest 
    (h1 : condition_1 G)
    (h2 : condition_2 G F)
    (h3 : condition_3 F M)
    (h4 : condition_4 F K) :
    G = 39 ∧ F = 20 ∧ M = 8 ∧ K = 35 := by
  sorry

end jumping_contest_l1613_161391


namespace factor_difference_of_squares_l1613_161327

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l1613_161327


namespace initial_miles_correct_l1613_161360

-- Definitions and conditions
def miles_per_gallon : ℕ := 30
def gallons_per_tank : ℕ := 20
def current_miles : ℕ := 2928
def tanks_filled : ℕ := 2

-- Question: How many miles were on the car before the road trip?
def initial_miles : ℕ := current_miles - (miles_per_gallon * gallons_per_tank * tanks_filled)

-- Proof problem statement
theorem initial_miles_correct : initial_miles = 1728 :=
by
  -- Here we expect the proof, but are skipping it with 'sorry'
  sorry

end initial_miles_correct_l1613_161360


namespace mechanic_charge_per_hour_l1613_161322

/-- Definitions based on provided conditions -/
def total_amount_paid : ℝ := 300
def part_cost : ℝ := 150
def hours : ℕ := 2

/-- Theorem stating the labor cost per hour is $75 -/
theorem mechanic_charge_per_hour (total_amount_paid part_cost hours : ℝ) : hours = 2 → part_cost = 150 → total_amount_paid = 300 → 
  (total_amount_paid - part_cost) / hours = 75 :=
by
  sorry

end mechanic_charge_per_hour_l1613_161322


namespace lex_read_pages_l1613_161338

theorem lex_read_pages (total_pages days : ℕ) (h1 : total_pages = 240) (h2 : days = 12) :
  total_pages / days = 20 :=
by sorry

end lex_read_pages_l1613_161338


namespace sum_of_all_potential_real_values_of_x_l1613_161319

/-- Determine the sum of all potential real values of x such that when the mean, median, 
and mode of the list [12, 3, 6, 3, 8, 3, x, 15] are arranged in increasing order, they 
form a non-constant arithmetic progression. -/
def sum_potential_x_values : ℚ :=
    let values := [12, 3, 6, 3, 8, 3, 15]
    let mean (x : ℚ) : ℚ := (50 + x) / 8
    let mode : ℚ := 3
    let median (x : ℚ) : ℚ := 
      if x ≤ 3 then 3.5 else if x < 6 then (x + 6) / 2 else 6
    let is_arithmetic_seq (a b c : ℚ) : Prop := 2 * b = a + c
    let valid_x_values : List ℚ := 
      (if is_arithmetic_seq mode 3.5 (mean (3.5)) then [] else []) ++
      (if is_arithmetic_seq mode 6 (mean 6) then [22] else []) ++
      (if is_arithmetic_seq mode (median (50 / 7)) (mean (50 / 7)) then [50 / 7] else [])
    (valid_x_values.sum)
theorem sum_of_all_potential_real_values_of_x :
  sum_potential_x_values = 204 / 7 :=
  sorry

end sum_of_all_potential_real_values_of_x_l1613_161319


namespace inequality_satisfied_l1613_161300

open Real

theorem inequality_satisfied (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  a * sqrt b + b * sqrt c + c * sqrt a ≤ 1 / sqrt 3 :=
sorry

end inequality_satisfied_l1613_161300


namespace find_f_of_500_l1613_161337

theorem find_f_of_500
  (f : ℕ → ℕ)
  (h_pos : ∀ x y : ℕ, f x > 0 ∧ f y > 0) 
  (h_mul : ∀ x y : ℕ, f (x * y) = f x + f y) 
  (h_f10 : f 10 = 15)
  (h_f40 : f 40 = 23) :
  f 500 = 41 :=
sorry

end find_f_of_500_l1613_161337


namespace neq_zero_necessary_not_sufficient_l1613_161398

theorem neq_zero_necessary_not_sufficient (x : ℝ) (h : x ≠ 0) : 
  (¬ (x = 0) ↔ x > 0) ∧ ¬ (x > 0 → x ≠ 0) :=
by sorry

end neq_zero_necessary_not_sufficient_l1613_161398


namespace sum_of_consecutive_naturals_l1613_161301

theorem sum_of_consecutive_naturals (n : ℕ) : 
  ∃ S : ℕ, S = n * (n + 1) / 2 :=
by
  sorry

end sum_of_consecutive_naturals_l1613_161301


namespace max_value_PXQ_l1613_161342

theorem max_value_PXQ :
  ∃ (X P Q : ℕ), (XX = 10 * X + X) ∧ (10 * X + X) * X = 100 * P + 10 * X + Q ∧ 
  (X = 1 ∨ X = 5 ∨ X = 6) ∧ 
  (100 * P + 10 * X + Q) = 396 :=
sorry

end max_value_PXQ_l1613_161342


namespace min_value_f_l1613_161392

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

theorem min_value_f (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f a b x ≤ 5) : 
  ∀ x : ℝ, x < 0 → f a b x ≥ -1 :=
by
  -- Since this is a statement-only problem, we leave the proof to be filled in
  sorry

end min_value_f_l1613_161392


namespace circle_standard_equation_l1613_161358

theorem circle_standard_equation {a : ℝ} :
  (∃ a : ℝ, a ≠ 0 ∧ (a = 2 * a - 3 ∨ a = 3 - 2 * a) ∧ 
  (((x - a)^2 + (y - (2 * a - 3))^2 = a^2) ∧ 
   ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1))) :=
sorry

end circle_standard_equation_l1613_161358


namespace total_votes_cast_is_8200_l1613_161352

variable (V : ℝ) (h1 : 0.35 * V < V) (h2 : 0.35 * V + 2460 = 0.65 * V)

theorem total_votes_cast_is_8200 (V : ℝ)
  (h1 : 0.35 * V < V)
  (h2 : 0.35 * V + 2460 = 0.65 * V) :
  V = 8200 := by
sorry

end total_votes_cast_is_8200_l1613_161352


namespace total_cost_food_l1613_161382

theorem total_cost_food
  (beef_pounds : ℕ)
  (beef_cost_per_pound : ℕ)
  (chicken_pounds : ℕ)
  (chicken_cost_per_pound : ℕ)
  (h_beef : beef_pounds = 1000)
  (h_beef_cost : beef_cost_per_pound = 8)
  (h_chicken : chicken_pounds = 2 * beef_pounds)
  (h_chicken_cost : chicken_cost_per_pound = 3) :
  (beef_pounds * beef_cost_per_pound + chicken_pounds * chicken_cost_per_pound = 14000) :=
by
  sorry

end total_cost_food_l1613_161382


namespace james_baked_multiple_l1613_161328

theorem james_baked_multiple (x : ℕ) (h1 : 115 ≠ 0) (h2 : 1380 = 115 * x) : x = 12 :=
sorry

end james_baked_multiple_l1613_161328


namespace aquariums_have_13_saltwater_animals_l1613_161399

theorem aquariums_have_13_saltwater_animals:
  ∀ x : ℕ, 26 * x = 52 → (∀ n : ℕ, n = 26 → (n * x = 52 ∧ x % 2 = 1 ∧ x > 1)) → x = 13 :=
by
  sorry

end aquariums_have_13_saltwater_animals_l1613_161399


namespace max_expression_value_l1613_161361

theorem max_expression_value (a b c d e f g h k : ℤ) 
  (ha : a = 1 ∨ a = -1)
  (hb : b = 1 ∨ b = -1)
  (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1)
  (he : e = 1 ∨ e = -1)
  (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1)
  (hh : h = 1 ∨ h = -1)
  (hk : k = 1 ∨ k = -1) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 :=
sorry

end max_expression_value_l1613_161361


namespace vector_k_range_l1613_161332

noncomputable def vector_length (v : (ℝ × ℝ)) : ℝ := (v.1 ^ 2 + v.2 ^ 2).sqrt

theorem vector_k_range :
  let a := (-2, 2)
  let b := (5, k)
  vector_length (a.1 + b.1, a.2 + b.2) ≤ 5 → -6 ≤ k ∧ k ≤ 2 := by
  sorry

end vector_k_range_l1613_161332


namespace utility_bills_total_correct_l1613_161373

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end utility_bills_total_correct_l1613_161373


namespace find_y_l1613_161348

def G (a b c d : ℕ) : ℕ := a ^ b + c * d

theorem find_y (y : ℕ) : G 3 y 5 10 = 350 ↔ y = 5 := by
  sorry

end find_y_l1613_161348


namespace management_sampled_count_l1613_161356

variable (total_employees salespeople management_personnel logistical_support staff_sample_size : ℕ)
variable (proportional_sampling : Prop)
variable (n_management_sampled : ℕ)

axiom h1 : total_employees = 160
axiom h2 : salespeople = 104
axiom h3 : management_personnel = 32
axiom h4 : logistical_support = 24
axiom h5 : proportional_sampling
axiom h6 : staff_sample_size = 20

theorem management_sampled_count : n_management_sampled = 4 :=
by
  -- The proof is omitted as per instructions
  sorry

end management_sampled_count_l1613_161356


namespace degree_of_g_l1613_161317

noncomputable def poly_degree (p : Polynomial ℝ) : ℕ :=
  Polynomial.natDegree p

theorem degree_of_g
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ := f.comp g - g)
  (hf : poly_degree f = 3)
  (hh : poly_degree h = 8) :
  poly_degree g = 3 :=
sorry

end degree_of_g_l1613_161317


namespace arithmetic_geometric_mean_inequality_l1613_161339

theorem arithmetic_geometric_mean_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := 
  sorry

end arithmetic_geometric_mean_inequality_l1613_161339


namespace boys_from_Pine_l1613_161355

/-
We need to prove that the number of boys from Pine Middle School is 70
given the following conditions:
1. There were 150 students in total.
2. 90 were boys and 60 were girls.
3. 50 students were from Maple Middle School.
4. 100 students were from Pine Middle School.
5. 30 of the girls were from Maple Middle School.
-/
theorem boys_from_Pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h_total : total_students = 150) (h_boys : total_boys = 90)
  (h_girls : total_girls = 60) (h_maple : maple_students = 50)
  (h_pine : pine_students = 100) (h_maple_girls : maple_girls = 30) :
  total_boys - maple_students + maple_girls = 70 :=
by
  sorry

end boys_from_Pine_l1613_161355


namespace change_factor_l1613_161362

theorem change_factor (avg1 avg2 : ℝ) (n : ℕ) (h_avg1 : avg1 = 40) (h_n : n = 10) (h_avg2 : avg2 = 80) : avg2 * (n : ℝ) / (avg1 * (n : ℝ)) = 2 :=
by
  sorry

end change_factor_l1613_161362


namespace second_quadrant_y_value_l1613_161347

theorem second_quadrant_y_value :
  ∀ (b : ℝ), (-3, b).2 > 0 → b = 2 :=
by
  sorry

end second_quadrant_y_value_l1613_161347


namespace simplify_expression_l1613_161372

theorem simplify_expression (x : ℝ) :
  (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 :=
by
  sorry

end simplify_expression_l1613_161372


namespace fraction_sum_zero_implies_square_sum_zero_l1613_161376

theorem fraction_sum_zero_implies_square_sum_zero (a b c : ℝ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := 
by
  sorry

end fraction_sum_zero_implies_square_sum_zero_l1613_161376


namespace line_through_P0_perpendicular_to_plane_l1613_161349

-- Definitions of the given conditions
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P0 : Point3D := { x := 3, y := 4, z := 2 }

def plane (x y z : ℝ) : Prop := 8 * x - 4 * y + 5 * z - 4 = 0

-- The proof problem statement
theorem line_through_P0_perpendicular_to_plane :
  ∃ t : ℝ, (P0.x + 8 * t = x ∧ P0.y - 4 * t = y ∧ P0.z + 5 * t = z) ↔
    (∃ t : ℝ, x = 3 + 8 * t ∧ y = 4 - 4 * t ∧ z = 2 + 5 * t) → 
    (∃ t : ℝ, (x - 3) / 8 = t ∧ (y - 4) / -4 = t ∧ (z - 2) / 5 = t) := sorry

end line_through_P0_perpendicular_to_plane_l1613_161349


namespace greatest_price_drop_is_april_l1613_161343

-- Define the price changes for each month
def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => 1.00
  | 2 => -1.50
  | 3 => -0.50
  | 4 => -3.75 -- including the -1.25 adjustment
  | 5 => 0.50
  | 6 => -2.25
  | _ => 0 -- default case, although we only deal with months 1-6

-- Define a predicate for the month with the greatest drop
def greatest_drop_month (m : ℕ) : Prop :=
  m = 4

-- Main theorem: Prove that the month with the greatest price drop is April
theorem greatest_price_drop_is_april : greatest_drop_month 4 :=
by
  -- Use Lean tactics to prove the statement
  sorry

end greatest_price_drop_is_april_l1613_161343


namespace LeonaEarnsGivenHourlyRate_l1613_161368

theorem LeonaEarnsGivenHourlyRate :
  (∀ (c: ℝ) (t h e: ℝ), 
    (c = 24.75) → 
    (t = 3) → 
    (h = c / t) → 
    (e = h * 5) →
    e = 41.25) :=
by
  intros c t h e h1 h2 h3 h4
  sorry

end LeonaEarnsGivenHourlyRate_l1613_161368


namespace value_of_x_for_g_equals_g_inv_l1613_161366

noncomputable def g (x : ℝ) : ℝ := 3 * x - 7
  
noncomputable def g_inv (x : ℝ) : ℝ := (x + 7) / 3
  
theorem value_of_x_for_g_equals_g_inv : ∃ x : ℝ, g x = g_inv x ∧ x = 3.5 :=
by
  sorry

end value_of_x_for_g_equals_g_inv_l1613_161366


namespace upgraded_video_card_multiple_l1613_161364

noncomputable def multiple_of_video_card_cost (computer_cost monitor_cost_peripheral_cost base_video_card_cost total_spent upgraded_video_card_cost : ℝ) : ℝ :=
  upgraded_video_card_cost / base_video_card_cost

theorem upgraded_video_card_multiple
  (computer_cost : ℝ)
  (monitor_cost_ratio : ℝ)
  (base_video_card_cost : ℝ)
  (total_spent : ℝ)
  (h1 : computer_cost = 1500)
  (h2 : monitor_cost_ratio = 1/5)
  (h3 : base_video_card_cost = 300)
  (h4 : total_spent = 2100) :
  multiple_of_video_card_cost computer_cost (computer_cost * monitor_cost_ratio) base_video_card_cost total_spent (total_spent - (computer_cost + computer_cost * monitor_cost_ratio)) = 1 :=
by
  sorry

end upgraded_video_card_multiple_l1613_161364


namespace trajectory_of_P_l1613_161354

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (2 * x - 3) ^ 2 + 4 * y ^ 2 = 1

theorem trajectory_of_P (m n x y : ℝ) (hM_on_circle : m^2 + n^2 = 1)
  (hP_midpoint : 2 * x = 3 + m ∧ 2 * y = n) : trajectory_equation x y :=
by 
  sorry

end trajectory_of_P_l1613_161354


namespace pyramid_height_l1613_161365

noncomputable def height_of_pyramid : ℝ :=
  let perimeter := 32
  let pb := 12
  let side := perimeter / 4
  let fb := (side * Real.sqrt 2) / 2
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height :
  height_of_pyramid = 4 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_l1613_161365


namespace minimum_area_of_Archimedean_triangle_l1613_161309

-- Define the problem statement with necessary conditions
theorem minimum_area_of_Archimedean_triangle (p : ℝ) (hp : p > 0) :
  ∃ (ABQ_area : ℝ), ABQ_area = p^2 ∧ 
    (∀ (A B Q : ℝ × ℝ), 
      (A.2 ^ 2 = 2 * p * A.1) ∧
      (B.2 ^ 2 = 2 * p * B.1) ∧
      (0, 0) = (p / 2, p / 2) ∧
      (Q.2 = 0) → 
      ABQ_area = p^2) :=
sorry

end minimum_area_of_Archimedean_triangle_l1613_161309


namespace relationship_among_a_b_c_l1613_161320

noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l1613_161320


namespace min_sets_bound_l1613_161363

theorem min_sets_bound (A : Type) (n k : ℕ) (S : Finset (Finset A))
  (h₁ : S.card = k)
  (h₂ : ∀ x y : A, x ≠ y → ∃ B ∈ S, (x ∈ B ∧ y ∉ B) ∨ (y ∈ B ∧ x ∉ B)) :
  2^k ≥ n :=
sorry

end min_sets_bound_l1613_161363


namespace arithmetic_sequence_8th_term_l1613_161307

theorem arithmetic_sequence_8th_term (a d: ℤ) (h1: a + 3 * d = 23) (h2: a + 5 * d = 47) :
  a + 7 * d = 71 :=
sorry

end arithmetic_sequence_8th_term_l1613_161307


namespace sequence_infinite_coprime_l1613_161333

theorem sequence_infinite_coprime (a : ℤ) (h : a > 1) :
  ∃ (S : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → Int.gcd (a^(S n + 1) + a^S n - 1) (a^(S m + 1) + a^S m - 1) = 1) :=
sorry

end sequence_infinite_coprime_l1613_161333
