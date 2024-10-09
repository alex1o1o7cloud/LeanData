import Mathlib

namespace total_balls_l1393_139338

def num_white : ℕ := 50
def num_green : ℕ := 30
def num_yellow : ℕ := 10
def num_red : ℕ := 7
def num_purple : ℕ := 3

def prob_neither_red_nor_purple : ℝ := 0.9

theorem total_balls (T : ℕ) 
  (h : prob_red_purple = 1 - prob_neither_red_nor_purple) 
  (h_prob : prob_red_purple = (num_red + num_purple : ℝ) / (T : ℝ)) :
  T = 100 :=
by sorry

end total_balls_l1393_139338


namespace proof_problem_l1393_139325

def f (x : ℤ) : ℤ := 3 * x + 5
def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_problem : 
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = (380 / 653) := 
  by 
    sorry

end proof_problem_l1393_139325


namespace milk_needed_for_cookies_l1393_139330

-- Definition of the problem conditions
def cookies_per_milk_usage := 24
def milk_in_liters := 5
def liters_to_milliliters := 1000
def milk_for_6_cookies := 1250

-- Prove that 1250 milliliters of milk are needed to bake 6 cookies
theorem milk_needed_for_cookies
  (h1 : cookies_per_milk_usage = 24)
  (h2 : milk_in_liters = 5)
  (h3 : liters_to_milliliters = 1000) :
  milk_for_6_cookies = 1250 :=
by
  -- Proof is omitted with sorry
  sorry

end milk_needed_for_cookies_l1393_139330


namespace stock_yield_percentage_l1393_139305

def annualDividend (parValue : ℕ) (rate : ℕ) : ℕ :=
  (parValue * rate) / 100

def yieldPercentage (dividend : ℕ) (marketPrice : ℕ) : ℕ :=
  (dividend * 100) / marketPrice

theorem stock_yield_percentage :
  let par_value := 100
  let rate := 8
  let market_price := 80
  yieldPercentage (annualDividend par_value rate) market_price = 10 :=
by
  sorry

end stock_yield_percentage_l1393_139305


namespace cos_of_largest_angle_is_neg_half_l1393_139320

-- Lean does not allow forward references to elements yet to be declared, 
-- hence we keep a strict order for declarations
namespace TriangleCosine

open Real

-- Define the side lengths of the triangle as constants
def a : ℝ := 3
def b : ℝ := 5
def c : ℝ := 7

-- Define the expression using cosine rule to find cos C
noncomputable def cos_largest_angle : ℝ := (a^2 + b^2 - c^2) / (2 * a * b)

-- Declare the theorem statement
theorem cos_of_largest_angle_is_neg_half : cos_largest_angle = -1 / 2 := 
by 
  sorry

end TriangleCosine

end cos_of_largest_angle_is_neg_half_l1393_139320


namespace area_BCD_sixteen_area_BCD_with_new_ABD_l1393_139371

-- Define the conditions and parameters of the problem.
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions from part (a)
variable (AB_length : Real) (BC_length : Real) (area_ABD : Real)

-- Define the lengths and areas in our problem.
axiom AB_eq_five : AB_length = 5
axiom BC_eq_eight : BC_length = 8
axiom area_ABD_eq_ten : area_ABD = 10

-- Part (a) problem statement
theorem area_BCD_sixteen (AB_length BC_length area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → area_ABD = 10 → (∃ area_BCD : Real, area_BCD = 16) :=
by
  sorry

-- Given conditions from part (b)
variable (new_area_ABD : Real)

-- Define the new area.
axiom new_area_ABD_eq_hundred : new_area_ABD = 100

-- Part (b) problem statement
theorem area_BCD_with_new_ABD (AB_length BC_length new_area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → new_area_ABD = 100 → (∃ area_BCD : Real, area_BCD = 160) :=
by
  sorry

end area_BCD_sixteen_area_BCD_with_new_ABD_l1393_139371


namespace find_g_2_l1393_139314

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 2 * g x - 3 * g (1 / x) = x ^ 2

theorem find_g_2 : g 2 = 8.25 :=
by {
  sorry
}

end find_g_2_l1393_139314


namespace total_jellybeans_l1393_139309

theorem total_jellybeans (G : ℕ) (H1 : G = 8 + 2) (H2 : ∀ O : ℕ, O = G - 1) : 
  8 + G + (G - 1) = 27 := 
by 
  sorry

end total_jellybeans_l1393_139309


namespace miles_driven_each_day_l1393_139388

-- Definition of the given conditions
def total_miles : ℝ := 1250
def number_of_days : ℝ := 5.0

-- The statement to be proved
theorem miles_driven_each_day :
  total_miles / number_of_days = 250 :=
by
  sorry

end miles_driven_each_day_l1393_139388


namespace jose_tabs_remaining_l1393_139387

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end jose_tabs_remaining_l1393_139387


namespace DanGreenMarbles_l1393_139352

theorem DanGreenMarbles : 
  ∀ (initial_green marbles_taken remaining_green : ℕ), 
  initial_green = 32 →
  marbles_taken = 23 →
  remaining_green = initial_green - marbles_taken →
  remaining_green = 9 :=
by sorry

end DanGreenMarbles_l1393_139352


namespace max_students_received_less_than_given_l1393_139367

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end max_students_received_less_than_given_l1393_139367


namespace ceil_floor_sum_l1393_139378

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l1393_139378


namespace base_conversion_and_addition_l1393_139312

def C : ℕ := 12

def base9_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 9^2) + (d1 * 9^1) + (d0 * 9^0)

def base13_to_nat (d2 d1 d0 : ℕ) : ℕ := (d2 * 13^2) + (d1 * 13^1) + (d0 * 13^0)

def num1 := base9_to_nat 7 5 2
def num2 := base13_to_nat 6 C 3

theorem base_conversion_and_addition :
  num1 + num2 = 1787 :=
by
  sorry

end base_conversion_and_addition_l1393_139312


namespace convert_spherical_to_cartesian_l1393_139322

theorem convert_spherical_to_cartesian :
  let ρ := 5
  let θ₁ := 3 * Real.pi / 4
  let φ₁ := 9 * Real.pi / 5
  let φ' := 2 * Real.pi - φ₁
  let θ' := θ₁ + Real.pi
  ∃ (θ : ℝ) (φ : ℝ),
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    0 ≤ φ ∧ φ ≤ Real.pi ∧
    (∃ (x y z : ℝ),
      x = ρ * Real.sin φ' * Real.cos θ' ∧
      y = ρ * Real.sin φ' * Real.sin θ' ∧
      z = ρ * Real.cos φ') ∧
    θ = θ' ∧ φ = φ' :=
by
  sorry

end convert_spherical_to_cartesian_l1393_139322


namespace thousandths_place_digit_of_7_div_32_l1393_139332

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end thousandths_place_digit_of_7_div_32_l1393_139332


namespace equation_of_directrix_l1393_139319

theorem equation_of_directrix (x y : ℝ) (h : y^2 = 2 * x) : 
  x = - (1/2) :=
sorry

end equation_of_directrix_l1393_139319


namespace frustum_lateral_area_l1393_139350

def frustum_upper_base_radius : ℝ := 3
def frustum_lower_base_radius : ℝ := 4
def frustum_slant_height : ℝ := 6

theorem frustum_lateral_area : 
  (1 / 2) * (frustum_upper_base_radius + frustum_lower_base_radius) * 2 * Real.pi * frustum_slant_height = 42 * Real.pi :=
by
  sorry

end frustum_lateral_area_l1393_139350


namespace length_of_AB_l1393_139381

theorem length_of_AB 
  (P Q A B : ℝ)
  (h_P_on_AB : P > 0 ∧ P < B)
  (h_Q_on_AB : Q > P ∧ Q < B)
  (h_ratio_P : P = 3 / 7 * B)
  (h_ratio_Q : Q = 4 / 9 * B)
  (h_PQ : Q - P = 3) 
: B = 189 := 
sorry

end length_of_AB_l1393_139381


namespace no_rational_roots_l1393_139351

theorem no_rational_roots (x : ℚ) : ¬(3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1 = 0) :=
by sorry

end no_rational_roots_l1393_139351


namespace simplify_sqrt_neg_five_squared_l1393_139347

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end simplify_sqrt_neg_five_squared_l1393_139347


namespace exterior_angle_BAC_l1393_139382

-- Definitions for the problem conditions
def regular_nonagon_interior_angle :=
  140

def square_interior_angle :=
  90

-- The proof statement
theorem exterior_angle_BAC (regular_nonagon_interior_angle square_interior_angle : ℝ) : 
  regular_nonagon_interior_angle = 140 ∧ square_interior_angle = 90 -> 
  ∃ (BAC : ℝ), BAC = 130 :=
by
  sorry

end exterior_angle_BAC_l1393_139382


namespace solution_set_of_inequality_l1393_139304

noncomputable def f : ℝ → ℝ := sorry 

axiom f_cond : ∀ x : ℝ, f x + deriv f x > 1
axiom f_at_zero : f 0 = 4

theorem solution_set_of_inequality : {x : ℝ | f x > 3 / Real.exp x + 1} = { x : ℝ | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l1393_139304


namespace fraction_of_mothers_with_full_time_jobs_l1393_139397

theorem fraction_of_mothers_with_full_time_jobs :
  (0.4 : ℝ) * M = 0.3 →
  (9 / 10 : ℝ) * 0.6 = 0.54 →
  1 - 0.16 = 0.84 →
  0.84 - 0.54 = 0.3 →
  M = 3 / 4 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here.
  sorry

end fraction_of_mothers_with_full_time_jobs_l1393_139397


namespace machine_performance_l1393_139390

noncomputable def machine_A_data : List ℕ :=
  [4, 1, 0, 2, 2, 1, 3, 1, 2, 4]

noncomputable def machine_B_data : List ℕ :=
  [2, 3, 1, 1, 3, 2, 2, 1, 2, 3]

noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

noncomputable def variance (data : List ℕ) (mean : ℝ) : ℝ :=
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

theorem machine_performance :
  let mean_A := mean machine_A_data
  let mean_B := mean machine_B_data
  let variance_A := variance machine_A_data mean_A
  let variance_B := variance machine_B_data mean_B
  mean_A = 2 ∧ mean_B = 2 ∧ variance_A = 1.6 ∧ variance_B = 0.6 ∧ variance_B < variance_A := 
sorry

end machine_performance_l1393_139390


namespace discount_difference_l1393_139337

theorem discount_difference (bill_amt : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  bill_amt = 12000 → d1 = 0.42 → d2 = 0.35 → d3 = 0.05 →
  (bill_amt * (1 - d2) * (1 - d3) - bill_amt * (1 - d1) = 450) :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end discount_difference_l1393_139337


namespace simplify_expression_l1393_139355

theorem simplify_expression (x : ℝ) (h : x ≤ 2) : 
  (Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9)) = -1 :=
by 
  sorry

end simplify_expression_l1393_139355


namespace purchase_in_april_l1393_139324

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l1393_139324


namespace solve_fractional_eq_l1393_139302

theorem solve_fractional_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) : 
  (2 / (x - 1) = 1 / x) ↔ (x = -1) :=
by 
  sorry

end solve_fractional_eq_l1393_139302


namespace students_taking_neither_l1393_139370

theorem students_taking_neither (total biology chemistry both : ℕ)
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  (total - (biology + chemistry - both)) = 10 :=
by {
  sorry
}

end students_taking_neither_l1393_139370


namespace more_red_than_yellow_l1393_139307

-- Define the number of bouncy balls per pack
def bouncy_balls_per_pack : ℕ := 18

-- Define the number of packs Jill bought
def packs_red : ℕ := 5
def packs_yellow : ℕ := 4

-- Define the total number of bouncy balls purchased for each color
def total_red : ℕ := bouncy_balls_per_pack * packs_red
def total_yellow : ℕ := bouncy_balls_per_pack * packs_yellow

-- The theorem statement indicating how many more red bouncy balls than yellow bouncy balls Jill bought
theorem more_red_than_yellow : total_red - total_yellow = 18 := by
  sorry

end more_red_than_yellow_l1393_139307


namespace value_of_expression_l1393_139356

variable (a b c : ℝ)

theorem value_of_expression (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
                            (h2 : abc = 1)
                            (h3 : a^2 + b^2 + c^2 - ((1 / (a^2)) + (1 / (b^2)) + (1 / (c^2))) = 8 * (a + b + c) - 8 * (ab + bc + ca)) :
                            (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) = -3/2 :=
by
  sorry

end value_of_expression_l1393_139356


namespace number_of_boxes_in_each_case_l1393_139306

theorem number_of_boxes_in_each_case (a b : ℕ) :
    a + b = 2 → 9 = a * 8 + b :=
by
    intro h
    sorry

end number_of_boxes_in_each_case_l1393_139306


namespace skateboard_price_after_discounts_l1393_139372

-- Defining all necessary conditions based on the given problem.
def original_price : ℝ := 150
def discount1 : ℝ := 0.40 * original_price
def price_after_discount1 : ℝ := original_price - discount1
def discount2 : ℝ := 0.25 * price_after_discount1
def final_price : ℝ := price_after_discount1 - discount2

-- Goal: Prove that the final price after both discounts is $67.50.
theorem skateboard_price_after_discounts : final_price = 67.50 := by
  sorry

end skateboard_price_after_discounts_l1393_139372


namespace bob_repay_l1393_139353

theorem bob_repay {x : ℕ} (h : 50 + 10 * x >= 150) : x >= 10 :=
by
  sorry

end bob_repay_l1393_139353


namespace cycling_problem_l1393_139369

theorem cycling_problem (x : ℝ) (h₀ : x > 0) :
  30 / x - 30 / (x + 3) = 2 / 3 :=
sorry

end cycling_problem_l1393_139369


namespace line_equations_through_point_with_intercepts_l1393_139383

theorem line_equations_through_point_with_intercepts (x y : ℝ) :
  (x = -10 ∧ y = 10) ∧ (∃ a : ℝ, 4 * a = intercept_x ∧ a = intercept_y) →
  (x + y = 0 ∨ x + 4 * y - 30 = 0) :=
by
  sorry

end line_equations_through_point_with_intercepts_l1393_139383


namespace range_of_f_l1393_139317

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

theorem range_of_f :
  Set.image f {-1, 0, 1, 2, 3} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l1393_139317


namespace problem_statement_l1393_139300
open Real

noncomputable def l1 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x - (cos α) * y + 1 = 0
noncomputable def l2 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (sin α) * x + (cos α) * y + 1 = 0
noncomputable def l3 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x - (sin α) * y + 1 = 0
noncomputable def l4 (α : ℝ) : ℝ → ℝ → Prop := fun x y => (cos α) * x + (sin α) * y + 1 = 0

theorem problem_statement:
  (∃ (α : ℝ), ∀ (x y : ℝ), l1 α x y → l2 α x y) ∧
  (∀ (α : ℝ), ∀ (x y : ℝ), l1 α x y → (sin α) * (cos α) + (-cos α) * (sin α) = 0) ∧
  (∃ (p : ℝ × ℝ), ∀ (α : ℝ), abs ((sin α) * p.1 - (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((sin α) * p.1 + (cos α) * p.2 + 1) / sqrt ((sin α)^2 + (cos α)^2) = 1 ∧
                        abs ((cos α) * p.1 - (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1 ∧
                        abs ((cos α) * p.1 + (sin α) * p.2 + 1) / sqrt ((cos α)^2 + (sin α)^2) = 1) :=
sorry

end problem_statement_l1393_139300


namespace solve_equation_l1393_139318

theorem solve_equation (x : ℝ) (h : x + 3 ≠ 0) : (2 / (x + 3) = 1) → (x = -1) :=
by
  intro h1
  -- Proof skipped
  sorry

end solve_equation_l1393_139318


namespace probability_xiaoming_l1393_139301

variable (win_probability : ℚ) 
          (xiaoming_goal : ℕ)
          (xiaojie_goal : ℕ)
          (rounds_needed_xiaoming : ℕ)
          (rounds_needed_xiaojie : ℕ)

def probability_xiaoming_wins_2_consecutive_rounds
   (win_probability : ℚ) 
   (rounds_needed_xiaoming : ℕ) : ℚ :=
  (win_probability ^ 2) + 
  2 * win_probability ^ 3 * (1 - win_probability) + 
  win_probability ^ 4

theorem probability_xiaoming :
    win_probability = (1/2) ∧ 
    rounds_needed_xiaoming = 2 ∧
    rounds_needed_xiaojie = 3 →
    probability_xiaoming_wins_2_consecutive_rounds (1 / 2) 2 = 7 / 16 :=
by
  -- Proof steps placeholder
  sorry

end probability_xiaoming_l1393_139301


namespace sum_of_solutions_l1393_139315

-- Define the polynomial equation and the condition
def equation (x : ℝ) : Prop := 3 = (x^3 - 3 * x^2 - 12 * x) / (x + 3)

-- Sum of solutions for the given polynomial equation under the constraint
theorem sum_of_solutions :
  (∀ x : ℝ, equation x → x ≠ -3) →
  ∃ (a b : ℝ), equation a ∧ equation b ∧ a + b = 4 := 
by
  intros h
  sorry

end sum_of_solutions_l1393_139315


namespace value_of_k_l1393_139364

theorem value_of_k (k : ℝ) : (2 - k * 2 = -4 * (-1)) → k = -1 :=
by
  intro h
  sorry

end value_of_k_l1393_139364


namespace lollipops_remainder_l1393_139340

theorem lollipops_remainder :
  let total_lollipops := 8362
  let lollipops_per_package := 12
  total_lollipops % lollipops_per_package = 10 :=
by
  let total_lollipops := 8362
  let lollipops_per_package := 12
  sorry

end lollipops_remainder_l1393_139340


namespace vanya_faster_speed_l1393_139380

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end vanya_faster_speed_l1393_139380


namespace find_c_l1393_139360

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 6) (h3 : ((6 - c) / c) = 4 / 9) : c = 54 / 13 :=
sorry

end find_c_l1393_139360


namespace find_c_l1393_139339

-- Definition of the function f
def f (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

-- Theorem statement
theorem find_c (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : f a a b c = a^3)
  (h4 : f b a b c = b^3) : c = 16 :=
by
  sorry

end find_c_l1393_139339


namespace monitor_height_l1393_139384

theorem monitor_height (width_in_inches : ℕ) (pixels_per_inch : ℕ) (total_pixels : ℕ) 
  (h1 : width_in_inches = 21) (h2 : pixels_per_inch = 100) (h3 : total_pixels = 2520000) : 
  total_pixels / (width_in_inches * pixels_per_inch) / pixels_per_inch = 12 :=
by
  sorry

end monitor_height_l1393_139384


namespace p_finishes_job_after_q_in_24_minutes_l1393_139389

theorem p_finishes_job_after_q_in_24_minutes :
  let P_rate := 1 / 4
  let Q_rate := 1 / 20
  let together_rate := P_rate + Q_rate
  let work_done_in_3_hours := together_rate * 3
  let remaining_work := 1 - work_done_in_3_hours
  let time_for_p_to_finish := remaining_work / P_rate
  let time_in_minutes := time_for_p_to_finish * 60
  time_in_minutes = 24 :=
by
  sorry

end p_finishes_job_after_q_in_24_minutes_l1393_139389


namespace recurring_decimals_sum_l1393_139303

theorem recurring_decimals_sum :
  (0.333333333333 : ℚ) + (0.040404040404 : ℚ) + (0.005005005005 : ℚ) = 42 / 111 :=
by
  sorry

end recurring_decimals_sum_l1393_139303


namespace tic_tac_toe_board_configurations_l1393_139357

theorem tic_tac_toe_board_configurations :
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  total_configurations = 592 :=
by 
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  sorry

end tic_tac_toe_board_configurations_l1393_139357


namespace train_crossing_time_l1393_139310

-- Define the length of the train
def train_length : ℝ := 120

-- Define the speed of the train
def train_speed : ℝ := 15

-- Define the target time to cross the man
def target_time : ℝ := 8

-- Proposition to prove
theorem train_crossing_time :
  target_time = train_length / train_speed :=
by
  sorry

end train_crossing_time_l1393_139310


namespace quadratic_has_real_root_l1393_139393

theorem quadratic_has_real_root (a b : ℝ) : (∃ x : ℝ, x^2 + a * x + b = 0) :=
by
  -- To use contradiction, we assume the negation
  have h : ¬ (∃ x : ℝ, x^2 + a * x + b = 0) := sorry
  -- By contradiction, this assumption should lead to a contradiction
  sorry

end quadratic_has_real_root_l1393_139393


namespace ratio_of_inscribed_to_circumscribed_l1393_139386

theorem ratio_of_inscribed_to_circumscribed (a : ℝ) :
  let r' := a * Real.sqrt 6 / 12
  let R' := a * Real.sqrt 6 / 4
  r' / R' = 1 / 3 := by
  sorry

end ratio_of_inscribed_to_circumscribed_l1393_139386


namespace mystery_book_shelves_l1393_139375

-- Define the conditions from the problem
def total_books : ℕ := 72
def picture_book_shelves : ℕ := 2
def books_per_shelf : ℕ := 9

-- Determine the number of mystery book shelves
theorem mystery_book_shelves : 
  let books_on_picture_shelves := picture_book_shelves * books_per_shelf
  let mystery_books := total_books - books_on_picture_shelves
  let mystery_shelves := mystery_books / books_per_shelf
  mystery_shelves = 6 :=
by {
  -- This space is intentionally left incomplete, as the proof itself is not required.
  sorry
}

end mystery_book_shelves_l1393_139375


namespace melted_ice_cream_depth_l1393_139373

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ),
    r_sphere = 3 ∧ r_cylinder = 12 ∧
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder →
    h_cylinder = 1 / 4 :=
by
  intros r_sphere r_cylinder h_cylinder h
  have r_sphere_eq : r_sphere = 3 := h.1
  have r_cylinder_eq : r_cylinder = 12 := h.2.1
  have volume_eq : (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder := h.2.2
  sorry

end melted_ice_cream_depth_l1393_139373


namespace class_average_gpa_l1393_139333

theorem class_average_gpa (n : ℕ) (hn : 0 < n) :
  ((1/3 * n) * 45 + (2/3 * n) * 60) / n = 55 :=
by
  sorry

end class_average_gpa_l1393_139333


namespace binom_12_11_l1393_139334

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l1393_139334


namespace total_points_l1393_139377

noncomputable def Noa_score : ℕ := 30
noncomputable def Phillip_score : ℕ := 2 * Noa_score
noncomputable def Lucy_score : ℕ := (3 / 2) * Phillip_score

theorem total_points : 
  Noa_score + Phillip_score + Lucy_score = 180 := 
by
  sorry

end total_points_l1393_139377


namespace total_bales_in_barn_l1393_139394

-- Definitions based on the conditions 
def initial_bales : ℕ := 47
def added_bales : ℕ := 35

-- Statement to prove the final number of bales in the barn
theorem total_bales_in_barn : initial_bales + added_bales = 82 :=
by
  sorry

end total_bales_in_barn_l1393_139394


namespace lowest_score_to_average_90_l1393_139396

theorem lowest_score_to_average_90 {s1 s2 s3 max_score avg_score : ℕ} 
    (h1: s1 = 88) 
    (h2: s2 = 96) 
    (h3: s3 = 105) 
    (hmax: max_score = 120) 
    (havg: avg_score = 90) 
    : ∃ s4 s5, s4 ≤ max_score ∧ s5 ≤ max_score ∧ (s1 + s2 + s3 + s4 + s5) / 5 = avg_score ∧ (min s4 s5 = 41) :=
by {
    sorry
}

end lowest_score_to_average_90_l1393_139396


namespace tg_pi_over_12_eq_exists_two_nums_l1393_139327

noncomputable def tg (x : ℝ) := Real.tan x

theorem tg_pi_over_12_eq : tg (Real.pi / 12) = Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

theorem exists_two_nums (a : Fin 13 → ℝ) (h_diff : Function.Injective a) :
  ∃ x y, 0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) < Real.sqrt ((2 - Real.sqrt 3) / (2 + Real.sqrt 3)) :=
sorry

end tg_pi_over_12_eq_exists_two_nums_l1393_139327


namespace wet_surface_area_is_correct_l1393_139329

-- Define the dimensions of the cistern
def cistern_length : ℝ := 6  -- in meters
def cistern_width  : ℝ := 4  -- in meters
def water_depth    : ℝ := 1.25  -- in meters

-- Compute areas for each surface in contact with water
def bottom_area : ℝ := cistern_length * cistern_width
def long_sides_area : ℝ := 2 * (cistern_length * water_depth)
def short_sides_area : ℝ := 2 * (cistern_width * water_depth)

-- Calculate the total area of the wet surface
def total_wet_surface_area : ℝ := bottom_area + long_sides_area + short_sides_area

-- Statement to prove
theorem wet_surface_area_is_correct : total_wet_surface_area = 49 := by
  sorry

end wet_surface_area_is_correct_l1393_139329


namespace solution_set_for_fractional_inequality_l1393_139345

theorem solution_set_for_fractional_inequality :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end solution_set_for_fractional_inequality_l1393_139345


namespace count_valid_triples_l1393_139363

theorem count_valid_triples :
  ∃! (a c : ℕ), a ≤ 101 ∧ 101 ≤ c ∧ a * c = 101^2 :=
sorry

end count_valid_triples_l1393_139363


namespace candidate_total_score_l1393_139321

theorem candidate_total_score (written_score : ℝ) (interview_score : ℝ) (written_weight : ℝ) (interview_weight : ℝ) :
    written_score = 90 → interview_score = 80 → written_weight = 0.70 → interview_weight = 0.30 →
    written_score * written_weight + interview_score * interview_weight = 87 :=
by
  intros
  sorry

end candidate_total_score_l1393_139321


namespace arc_length_of_sector_l1393_139326

theorem arc_length_of_sector (θ r : ℝ) (h1 : θ = 120) (h2 : r = 2) : 
  (θ / 360) * (2 * Real.pi * r) = (4 * Real.pi) / 3 :=
by
  sorry

end arc_length_of_sector_l1393_139326


namespace min_chord_length_l1393_139398

-- Definitions of the problem conditions
def circle_center : ℝ × ℝ := (2, 3)
def circle_radius : ℝ := 3
def point_P : ℝ × ℝ := (1, 1)

-- The mathematical statement to prove
theorem min_chord_length : 
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ ((B.1 - 2)^2 + (B.2 - 3)^2 = 9) ∧ 
  ((A.1 - 1) / (B.1 - 1) = (A.2 - 1) / (B.2 - 1)) → 
  dist A B ≥ 4 := 
sorry

end min_chord_length_l1393_139398


namespace relationship_among_f_l1393_139311

theorem relationship_among_f (
  f : ℝ → ℝ
) (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x - 1) = f (x + 1))
  (h_increasing : ∀ a b, (0 ≤ a ∧ a < b ∧ b ≤ 1) → f a < f b) :
  f 2 < f (-5.5) ∧ f (-5.5) < f (-1) :=
by
  sorry

end relationship_among_f_l1393_139311


namespace melanie_trout_catch_l1393_139392

theorem melanie_trout_catch (T M : ℕ) 
  (h1 : T = 2 * M) 
  (h2 : T = 16) : 
  M = 8 :=
by
  sorry

end melanie_trout_catch_l1393_139392


namespace sum_of_areas_of_circles_l1393_139368

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l1393_139368


namespace find_right_triangle_conditions_l1393_139346

def is_right_triangle (A B C : ℝ) : Prop := 
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem find_right_triangle_conditions (A B C : ℝ):
  (A + B = C ∧ is_right_triangle A B C) ∨ 
  (A = B ∧ B = 2 * C ∧ is_right_triangle A B C) ∨ 
  (A / 30 = 1 ∧ B / 30 = 2 ∧ C / 30 = 3 ∧ is_right_triangle A B C) :=
sorry

end find_right_triangle_conditions_l1393_139346


namespace solve_for_x_l1393_139335

theorem solve_for_x : ∀ (x : ℕ), (1000 = 10^3) → (40 = 2^3 * 5) → 1000^5 = 40^x → x = 15 :=
by
  intros x h1 h2 h3
  sorry

end solve_for_x_l1393_139335


namespace initial_average_l1393_139336

theorem initial_average (A : ℝ) (h : (15 * A + 14 * 15) / 15 = 54) : A = 40 :=
by
  sorry

end initial_average_l1393_139336


namespace least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l1393_139358

noncomputable def sum_of_cubes (n : ℕ) : ℕ :=
  (n * (n + 1)/2)^2

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem least_m_for_sum_of_cubes_is_perfect_cube 
  (h : ∃ m : ℕ, ∀ (a : ℕ), (sum_of_cubes (2*m+1) = a^3) → a = 6):
  m = 1 := sorry

theorem least_k_for_sum_of_squares_is_perfect_square 
  (h : ∃ k : ℕ, ∀ (b : ℕ), (sum_of_squares (2*k+1) = b^2) → b = 77):
  k = 5 := sorry

end least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l1393_139358


namespace num_integers_satisfying_inequality_l1393_139376

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end num_integers_satisfying_inequality_l1393_139376


namespace arithmetic_sequence_a4_l1393_139349

def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else if n = 2 then 4 else 2 + (n - 1) * 2

theorem arithmetic_sequence_a4 :
  a 4 = 8 :=
by {
  sorry
}

end arithmetic_sequence_a4_l1393_139349


namespace probability_no_3x3_red_square_l1393_139344

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end probability_no_3x3_red_square_l1393_139344


namespace Maggie_age_l1393_139313

theorem Maggie_age (Kate Maggie Sue : ℕ) (h1 : Kate + Maggie + Sue = 48) (h2 : Kate = 19) (h3 : Sue = 12) : Maggie = 17 := by
  sorry

end Maggie_age_l1393_139313


namespace arcsin_one_half_l1393_139374

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l1393_139374


namespace derek_dogs_l1393_139316

theorem derek_dogs (d c : ℕ) (h1 : d = 90) 
  (h2 : c = d / 3) 
  (h3 : c + 210 = 2 * (d + 120 - d)) : 
  d + 120 - d = 120 :=
by
  sorry

end derek_dogs_l1393_139316


namespace solve_for_r_l1393_139328

theorem solve_for_r (r : ℚ) (h : 4 * (r - 10) = 3 * (3 - 3 * r) + 9) : r = 58 / 13 :=
by
  sorry

end solve_for_r_l1393_139328


namespace min_value_of_n_l1393_139354

theorem min_value_of_n :
  ∀ (h : ℝ), ∃ n : ℝ, (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -x^2 + 2 * h * x - h ≤ n) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -x^2 + 2 * h * x - h = n) ∧
  n = -1 / 4 := 
by
  sorry

end min_value_of_n_l1393_139354


namespace combined_value_l1393_139343

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l1393_139343


namespace value_of_a_minus_b_l1393_139379

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 :=
sorry

end value_of_a_minus_b_l1393_139379


namespace lcm_is_perfect_square_l1393_139361

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end lcm_is_perfect_square_l1393_139361


namespace sparkling_water_cost_l1393_139331

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end sparkling_water_cost_l1393_139331


namespace real_roots_for_all_K_l1393_139362

theorem real_roots_for_all_K (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x-1) * (x-2) + 2 * x :=
sorry

end real_roots_for_all_K_l1393_139362


namespace rank_friends_l1393_139341

variable (Amy Bill Celine : Prop)

-- Statement definitions
def statement_I := Bill
def statement_II := ¬Amy
def statement_III := ¬Celine

-- Exactly one of the statements is true
def exactly_one_true (s1 s2 s3 : Prop) :=
  (s1 ∧ ¬s2 ∧ ¬s3) ∨ (¬s1 ∧ s2 ∧ ¬s3) ∨ (¬s1 ∧ ¬s2 ∧ s3)

theorem rank_friends (h : exactly_one_true (statement_I Bill) (statement_II Amy) (statement_III Celine)) :
  (Amy ∧ ¬Bill ∧ Celine) :=
sorry

end rank_friends_l1393_139341


namespace solution_exists_solution_unique_l1393_139348

noncomputable def abc_solutions : Finset (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (2, 2, 4), (2, 4, 8), (3, 5, 15), 
   (2, 4, 2), (4, 2, 2), (4, 2, 8), (8, 4, 2), 
   (2, 8, 4), (8, 2, 4), (5, 3, 15), (15, 3, 5), (3, 15, 5),
   (2, 2, 4), (4, 2, 2), (4, 8, 2)}

theorem solution_exists (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a * b * c - 1 = (a - 1) * (b - 1) * (c - 1)) ↔ (a, b, c) ∈ abc_solutions := 
by
  sorry

theorem solution_unique (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a, b, c) ∈ abc_solutions → a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) :=
by
  sorry

end solution_exists_solution_unique_l1393_139348


namespace solve_for_z_l1393_139365

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end solve_for_z_l1393_139365


namespace inverse_of_97_mod_98_l1393_139385

theorem inverse_of_97_mod_98 : 97 * 97 ≡ 1 [MOD 98] :=
by
  sorry

end inverse_of_97_mod_98_l1393_139385


namespace taco_beef_per_taco_l1393_139342

open Real

theorem taco_beef_per_taco
  (total_beef : ℝ)
  (sell_price : ℝ)
  (cost_per_taco : ℝ)
  (profit : ℝ)
  (h1 : total_beef = 100)
  (h2 : sell_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : profit = 200) :
  ∃ (x : ℝ), x = 1/4 := 
by
  -- The proof will go here.
  sorry

end taco_beef_per_taco_l1393_139342


namespace find_n_l1393_139323

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 14 = 56) (h_gcf : Nat.gcd n 14 = 12) : n = 48 :=
by
  sorry

end find_n_l1393_139323


namespace quadruple_solution_l1393_139391

theorem quadruple_solution (a b p n : ℕ) (hp: Nat.Prime p) (hp_pos: p > 0) (ha_pos: a > 0) (hb_pos: b > 0) (hn_pos: n > 0) :
    a^3 + b^3 = p^n →
    (∃ k, k ≥ 1 ∧ (
        (a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k-2) ∨ 
        (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k-1) ∨ 
        (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k-1)
    )) := 
sorry

end quadruple_solution_l1393_139391


namespace larger_square_area_l1393_139395

theorem larger_square_area 
    (s₁ s₂ s₃ s₄ : ℕ) 
    (H1 : s₁ = 20) 
    (H2 : s₂ = 10) 
    (H3 : s₃ = 18) 
    (H4 : s₄ = 12) :
    (s₃ + s₄) > (s₁ + s₂) :=
by
  sorry

end larger_square_area_l1393_139395


namespace total_money_from_tshirts_l1393_139366

def num_tshirts_sold := 20
def money_per_tshirt := 215

theorem total_money_from_tshirts :
  num_tshirts_sold * money_per_tshirt = 4300 :=
by
  sorry

end total_money_from_tshirts_l1393_139366


namespace twenty_is_80_percent_of_what_number_l1393_139359

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l1393_139359


namespace count_indistinguishable_distributions_l1393_139399

theorem count_indistinguishable_distributions (balls : ℕ) (boxes : ℕ) (h_balls : balls = 5) (h_boxes : boxes = 4) : 
  ∃ n : ℕ, n = 6 := by
  sorry

end count_indistinguishable_distributions_l1393_139399


namespace oil_consumption_relation_l1393_139308

noncomputable def initial_oil : ℝ := 62

noncomputable def remaining_oil (x : ℝ) : ℝ :=
  if x = 100 then 50
  else if x = 200 then 38
  else if x = 300 then 26
  else if x = 400 then 14
  else 62 - 0.12 * x

theorem oil_consumption_relation (x : ℝ) :
  remaining_oil x = 62 - 0.12 * x := by
  sorry

end oil_consumption_relation_l1393_139308
