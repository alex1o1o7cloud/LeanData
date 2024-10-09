import Mathlib

namespace cos_pi_plus_2alpha_l1120_112080

theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin ((Real.pi / 2) + α) = 1 / 3) : Real.cos (Real.pi + 2 * α) = 7 / 9 :=
by
  sorry

end cos_pi_plus_2alpha_l1120_112080


namespace bella_steps_l1120_112042

-- Define the conditions and the necessary variables
variable (b : ℝ) (distance : ℝ) (steps_per_foot : ℝ)

-- Given constants
def bella_speed := b
def ella_speed := 4 * b
def combined_speed := bella_speed + ella_speed
def total_distance := 15840
def feet_per_step := 3

-- Define the main theorem to prove the number of steps Bella takes
theorem bella_steps : (total_distance / combined_speed) * bella_speed / feet_per_step = 1056 := by
  sorry

end bella_steps_l1120_112042


namespace min_people_wearing_both_l1120_112031

theorem min_people_wearing_both (n : ℕ) (h_lcm : n % 24 = 0) 
  (h_gloves : 3 * n % 8 = 0) (h_hats : 5 * n % 6 = 0) :
  ∃ x, x = 5 := 
by
  let gloves := 3 * n / 8
  let hats := 5 * n / 6
  let both := gloves + hats - n
  have h1 : both = 5 := sorry
  exact ⟨both, h1⟩

end min_people_wearing_both_l1120_112031


namespace problem_statement_l1120_112088

variable {a b c x y z : ℝ}
variable (h1 : 17 * x + b * y + c * z = 0)
variable (h2 : a * x + 29 * y + c * z = 0)
variable (h3 : a * x + b * y + 53 * z = 0)
variable (ha : a ≠ 17)
variable (hx : x ≠ 0)

theorem problem_statement : 
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end problem_statement_l1120_112088


namespace probability_diff_digits_l1120_112023

open Finset

def two_digit_same_digit (n : ℕ) : Prop :=
  n / 10 = n % 10

def three_digit_same_digit (n : ℕ) : Prop :=
  (n % 100) / 10 = n / 100 ∧ (n / 100) = (n % 10)

def same_digit (n : ℕ) : Prop :=
  two_digit_same_digit n ∨ three_digit_same_digit n

def total_numbers : ℕ :=
  (199 - 10 + 1)

def same_digit_count : ℕ :=
  9 + 9

theorem probability_diff_digits : 
  ((total_numbers - same_digit_count) / total_numbers : ℚ) = 86 / 95 :=
by
  sorry

end probability_diff_digits_l1120_112023


namespace train_length_l1120_112000

theorem train_length (time_crossing : ℕ) (speed_kmh : ℕ) (conversion_factor : ℕ) (expected_length : ℕ) :
  time_crossing = 4 ∧ speed_kmh = 144 ∧ conversion_factor = 1000 / 3600 * 144 →
  expected_length = 160 :=
by
  sorry

end train_length_l1120_112000


namespace sarah_total_pencils_l1120_112004

-- Define the number of pencils Sarah buys on each day
def pencils_monday : ℕ := 35
def pencils_tuesday : ℕ := 42
def pencils_wednesday : ℕ := 3 * pencils_tuesday
def pencils_thursday : ℕ := pencils_wednesday / 2
def pencils_friday : ℕ := 2 * pencils_monday

-- Define the total number of pencils
def total_pencils : ℕ :=
  pencils_monday + pencils_tuesday + pencils_wednesday + pencils_thursday + pencils_friday

-- Theorem statement to prove the total number of pencils equals 336
theorem sarah_total_pencils : total_pencils = 336 :=
by
  -- here goes the proof, but it is not required
  sorry

end sarah_total_pencils_l1120_112004


namespace english_only_students_l1120_112084

theorem english_only_students (T B G_total : ℕ) (hT : T = 40) (hB : B = 12) (hG_total : G_total = 22) :
  (T - (G_total - B) - B) = 18 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end english_only_students_l1120_112084


namespace mrs_peterson_change_l1120_112066

def num_tumblers : ℕ := 10
def cost_per_tumbler : ℕ := 45
def num_bills : ℕ := 5
def value_per_bill : ℕ := 100

theorem mrs_peterson_change : 
  (num_bills * value_per_bill) - (num_tumblers * cost_per_tumbler) = 50 :=
by
  sorry

end mrs_peterson_change_l1120_112066


namespace notification_possible_l1120_112018

-- Define the conditions
def side_length : ℝ := 2
def speed : ℝ := 3
def initial_time : ℝ := 12 -- noon
def arrival_time : ℝ := 19 -- 7 PM
def notification_time : ℝ := arrival_time - initial_time -- total available time for notification

-- Define the proof statement
theorem notification_possible :
  ∃ (partition : ℕ → ℝ) (steps : ℕ → ℝ), (∀ k, steps k * partition k < notification_time) ∧ 
  ∑' k, (steps k * partition k) ≤ 6 :=
by
  sorry

end notification_possible_l1120_112018


namespace correct_statements_l1120_112009

namespace ProofProblem

variable (f : ℕ+ × ℕ+ → ℕ+)
variable (h1 : f (1, 1) = 1)
variable (h2 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2)
variable (h3 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1))

theorem correct_statements :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by
  sorry

end ProofProblem

end correct_statements_l1120_112009


namespace tangent_line_at_01_l1120_112028

noncomputable def tangent_line_equation (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_01 : ∃ (m b : ℝ), (m = 1) ∧ (b = 1) ∧ (∀ x, tangent_line_equation x = m * x + b) :=
by
  sorry

end tangent_line_at_01_l1120_112028


namespace prob_business_less25_correct_l1120_112091

def prob_male : ℝ := 0.4
def prob_female : ℝ := 0.6

def prob_science : ℝ := 0.3
def prob_arts : ℝ := 0.45
def prob_business : ℝ := 0.25

def prob_male_science_25plus : ℝ := 0.4
def prob_male_arts_25plus : ℝ := 0.5
def prob_male_business_25plus : ℝ := 0.35

def prob_female_science_25plus : ℝ := 0.3
def prob_female_arts_25plus : ℝ := 0.45
def prob_female_business_25plus : ℝ := 0.2

def prob_male_science_less25 : ℝ := 1 - prob_male_science_25plus
def prob_male_arts_less25 : ℝ := 1 - prob_male_arts_25plus
def prob_male_business_less25 : ℝ := 1 - prob_male_business_25plus

def prob_female_science_less25 : ℝ := 1 - prob_female_science_25plus
def prob_female_arts_less25 : ℝ := 1 - prob_female_arts_25plus
def prob_female_business_less25 : ℝ := 1 - prob_female_business_25plus

def prob_science_less25 : ℝ := prob_male * prob_science * prob_male_science_less25 + prob_female * prob_science * prob_female_science_less25
def prob_arts_less25 : ℝ := prob_male * prob_arts * prob_male_arts_less25 + prob_female * prob_arts * prob_female_arts_less25
def prob_business_less25 : ℝ := prob_male * prob_business * prob_male_business_less25 + prob_female * prob_business * prob_female_business_less25

theorem prob_business_less25_correct :
    prob_business_less25 = 0.185 :=
by
  -- Theorem statement to be proved (proof omitted)
  sorry

end prob_business_less25_correct_l1120_112091


namespace table_ratio_l1120_112022

theorem table_ratio (L W : ℝ) (h1 : L * W = 128) (h2 : L + 2 * W = 32) : L / W = 2 :=
by
  sorry

end table_ratio_l1120_112022


namespace sin_alpha_value_l1120_112087

-- Define the given conditions
def α : ℝ := sorry -- α is an acute angle
def β : ℝ := sorry -- β has an unspecified value

-- Given conditions translated to Lean
def condition1 : Prop := 2 * Real.tan (Real.pi - α) - 3 * Real.cos (Real.pi / 2 + β) + 5 = 0
def condition2 : Prop := Real.tan (Real.pi + α) + 6 * Real.sin (Real.pi + β) = 1

-- Acute angle condition
def α_acute : Prop := 0 < α ∧ α < Real.pi / 2

-- The proof statement
theorem sin_alpha_value (h1 : condition1) (h2 : condition2) (h3 : α_acute) : Real.sin α = 3 * Real.sqrt 10 / 10 :=
by sorry

end sin_alpha_value_l1120_112087


namespace express_train_leaves_6_hours_later_l1120_112092

theorem express_train_leaves_6_hours_later
  (V_g V_e : ℕ) (t : ℕ) (catch_up_time : ℕ)
  (goods_train_speed : V_g = 36)
  (express_train_speed : V_e = 90)
  (catch_up_in_4_hours : catch_up_time = 4)
  (distance_e : V_e * catch_up_time = 360)
  (distance_g : V_g * (t + catch_up_time) = 360) :
  t = 6 := by
  sorry

end express_train_leaves_6_hours_later_l1120_112092


namespace new_person_weight_l1120_112038

theorem new_person_weight (avg_inc : Real) (num_persons : Nat) (old_weight new_weight : Real)
  (h1 : avg_inc = 2.5)
  (h2 : num_persons = 8)
  (h3 : old_weight = 40)
  (h4 : num_persons * avg_inc = new_weight - old_weight) :
  new_weight = 60 :=
by
  --proof will be done here
  sorry

end new_person_weight_l1120_112038


namespace projectiles_initial_distance_l1120_112067

theorem projectiles_initial_distance (Projectile1_speed Projectile2_speed Time_to_meet : ℕ) 
  (h1 : Projectile1_speed = 444)
  (h2 : Projectile2_speed = 555)
  (h3 : Time_to_meet = 2) : 
  (Projectile1_speed + Projectile2_speed) * Time_to_meet = 1998 := by
  sorry

end projectiles_initial_distance_l1120_112067


namespace usual_eggs_accepted_l1120_112074

theorem usual_eggs_accepted (A R : ℝ) (h1 : A / R = 1 / 4) (h2 : (A + 12) / (R - 4) = 99 / 1) (h3 : A + R = 400) :
  A = 392 :=
by
  sorry

end usual_eggs_accepted_l1120_112074


namespace smallest_b_l1120_112006

-- Define the variables and conditions
variables {a b : ℝ}

-- Assumptions based on the problem conditions
axiom h1 : 2 < a
axiom h2 : a < b

-- The theorems for the triangle inequality violations
theorem smallest_b (h : a ≥ b / (2 * b - 1)) (h' : 2 + a ≤ b) : b = (3 + Real.sqrt 7) / 2 :=
sorry

end smallest_b_l1120_112006


namespace direct_proportion_m_value_l1120_112029

theorem direct_proportion_m_value (m : ℝ) : 
  (∀ x: ℝ, y = -7 * x + 2 + m -> y = k * x) -> m = -2 :=
by
  sorry

end direct_proportion_m_value_l1120_112029


namespace derivative_of_m_l1120_112036

noncomputable def m (x : ℝ) : ℝ := (2 : ℝ)^x / (1 + x)

theorem derivative_of_m (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by
  sorry

end derivative_of_m_l1120_112036


namespace sum_of_remainders_l1120_112032

theorem sum_of_remainders (p : ℕ) (hp : p > 2) (hp_prime : Nat.Prime p)
    (a : ℕ → ℕ) (ha : ∀ k, a k = k^p % p^2) :
    (Finset.sum (Finset.range (p - 1)) a) = (p^3 - p^2) / 2 :=
by
  sorry

end sum_of_remainders_l1120_112032


namespace income_scientific_notation_l1120_112025

theorem income_scientific_notation (avg_income_per_acre : ℝ) (acres : ℝ) (a n : ℝ) :
  avg_income_per_acre = 20000 →
  acres = 8000 → 
  (avg_income_per_acre * acres = a * 10 ^ n ↔ (a = 1.6 ∧ n = 8)) :=
by
  sorry

end income_scientific_notation_l1120_112025


namespace evaluate_custom_op_l1120_112014

def custom_op (a b : ℝ) : ℝ := (a - b)^2

theorem evaluate_custom_op (x y : ℝ) : custom_op ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 :=
by
  sorry

end evaluate_custom_op_l1120_112014


namespace vector_calculation_l1120_112015

variables (a b : ℝ × ℝ)

def a_def : Prop := a = (3, 5)
def b_def : Prop := b = (-2, 1)

theorem vector_calculation (h1 : a_def a) (h2 : b_def b) : a - 2 • b = (7, 3) :=
sorry

end vector_calculation_l1120_112015


namespace rate_is_15_l1120_112044

variable (sum : ℝ) (interest12 : ℝ) (interest_r : ℝ) (r : ℝ)

-- Given conditions
def conditions : Prop :=
  sum = 7000 ∧
  interest12 = 7000 * 0.12 * 2 ∧
  interest_r = 7000 * (r / 100) * 2 ∧
  interest_r = interest12 + 420

-- The rate to prove
def rate_to_prove : Prop := r = 15

theorem rate_is_15 : conditions sum interest12 interest_r r → rate_to_prove r := 
by
  sorry

end rate_is_15_l1120_112044


namespace volume_of_hall_l1120_112098

-- Define the dimensions and areas conditions
def length_hall : ℝ := 15
def breadth_hall : ℝ := 12
def area_floor_ceiling : ℝ := 2 * (length_hall * breadth_hall)
def area_walls (h : ℝ) : ℝ := 2 * (length_hall * h) + 2 * (breadth_hall * h)

-- Given condition: The sum of the areas of the floor and ceiling is equal to the sum of the areas of the four walls
def condition (h : ℝ) : Prop := area_floor_ceiling = area_walls h

-- Define the volume of the hall
def volume_hall (h : ℝ) : ℝ := length_hall * breadth_hall * h

-- The theorem to be proven: given the condition, the volume equals 8004
theorem volume_of_hall : ∃ h : ℝ, condition h ∧ volume_hall h = 8004 := by
  sorry

end volume_of_hall_l1120_112098


namespace rhombus_diagonal_l1120_112012

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 10) (h2 : area = 60) : 
  d1 = 12 :=
by 
  have : (d1 * d2) / 2 = area := sorry
  sorry

end rhombus_diagonal_l1120_112012


namespace sum_of_first_six_terms_l1120_112017

noncomputable def a (n : ℕ) : ℚ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms :
  sum_first_six_terms a = 63 / 32 :=
by
  sorry

end sum_of_first_six_terms_l1120_112017


namespace calc1_calc2_calc3_l1120_112069

theorem calc1 : -4 - 4 = -8 := by
  sorry

theorem calc2 : (-32) / 4 = -8 := by
  sorry

theorem calc3 : -(-2)^3 = 8 := by
  sorry

end calc1_calc2_calc3_l1120_112069


namespace relationship_ab_l1120_112041

-- Define the conditions
variable {f : ℝ → ℝ}
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 1) = -f x)
variable (a b : ℝ)
variable (h_ex : ∃ x : ℝ, f (a + x) = f (b - x))

-- State the conclusion we need to prove
theorem relationship_ab : ∃ k : ℕ, k > 0 ∧ (a + b) = 2 * k + 1 :=
by
  sorry

end relationship_ab_l1120_112041


namespace valid_subsets_12_even_subsets_305_l1120_112016

def valid_subsets_count(n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 4
  else
    valid_subsets_count (n - 1) +
    valid_subsets_count (n - 2) +
    valid_subsets_count (n - 3)
    -- Recurrence relation for valid subsets which satisfy the conditions

theorem valid_subsets_12 : valid_subsets_count 12 = 610 :=
  by sorry
  -- We need to verify recurrence and compute for n = 12 (optional step if just computing, not proving the sequence.)

theorem even_subsets_305 :
  (valid_subsets_count 12) / 2 = 305 :=
  by sorry
  -- Concludes that half the valid subsets for n = 12 are even-sized sets.

end valid_subsets_12_even_subsets_305_l1120_112016


namespace proveCarTransportationProblem_l1120_112021

def carTransportationProblem :=
  ∃ x y a b : ℕ,
  -- Conditions regarding the capabilities of the cars
  (2 * x + 3 * y = 18) ∧
  (x + 2 * y = 11) ∧
  -- Conclusion (question 1)
  (x + y = 7) ∧
  -- Conditions for the rental plan (question 2)
  (3 * a + 4 * b = 27) ∧
  -- Cost optimization
  ((100 * a + 120 * b) = 820 ∨ (100 * a + 120 * b) = 860) ∧
  -- Optimal cost verification
  (100 * a + 120 * b = 820 → a = 1 ∧ b = 6)

theorem proveCarTransportationProblem : carTransportationProblem :=
  sorry

end proveCarTransportationProblem_l1120_112021


namespace sum_of_geometric_sequence_eq_31_over_16_l1120_112011

theorem sum_of_geometric_sequence_eq_31_over_16 (n : ℕ) :
  let a := 1
  let r := (1 / 2 : ℝ)
  let S_n := 2 - 2 * r^n
  (S_n = (31 / 16 : ℝ)) ↔ (n = 5) := by
{
  sorry
}

end sum_of_geometric_sequence_eq_31_over_16_l1120_112011


namespace toys_produced_per_week_l1120_112045

-- Definitions corresponding to the conditions
def days_per_week : ℕ := 2
def toys_per_day : ℕ := 2170

-- Theorem statement corresponding to the question and correct answer
theorem toys_produced_per_week : days_per_week * toys_per_day = 4340 := 
by 
  -- placeholders for the proof steps
  sorry

end toys_produced_per_week_l1120_112045


namespace base_of_parallelogram_l1120_112013

theorem base_of_parallelogram (Area Height : ℕ) (h1 : Area = 44) (h2 : Height = 11) : (Area / Height) = 4 :=
by
  sorry

end base_of_parallelogram_l1120_112013


namespace new_prism_volume_l1120_112060

theorem new_prism_volume (L W H : ℝ) 
  (h_volume : L * W * H = 54)
  (L_new : ℝ := 2 * L)
  (W_new : ℝ := 3 * W)
  (H_new : ℝ := 1.5 * H) :
  L_new * W_new * H_new = 486 := 
by
  sorry

end new_prism_volume_l1120_112060


namespace proj_eq_line_eqn_l1120_112062

theorem proj_eq_line_eqn (x y : ℝ)
  (h : (6 * x + 3 * y) * 6 / 45 = -3 ∧ (6 * x + 3 * y) * 3 / 45 = -3 / 2) :
  y = -2 * x - 15 / 2 :=
by
  sorry

end proj_eq_line_eqn_l1120_112062


namespace each_half_month_has_15_days_l1120_112056

noncomputable def days_in_each_half (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) : ℕ :=
  let first_half_days := total_days / 2
  let second_half_days := total_days - first_half_days
  first_half_days

theorem each_half_month_has_15_days (total_days : ℕ) (mean_profit_total: ℚ) 
  (mean_profit_first_half: ℚ) (mean_profit_last_half: ℚ) :
  total_days = 30 → mean_profit_total = 350 → mean_profit_first_half = 275 → mean_profit_last_half = 425 → 
  days_in_each_half total_days mean_profit_total mean_profit_first_half mean_profit_last_half = 15 :=
by
  intros h_days h_total h_first h_last
  sorry

end each_half_month_has_15_days_l1120_112056


namespace coordinates_equidistant_l1120_112001

-- Define the condition of equidistance
theorem coordinates_equidistant (x y : ℝ) :
  (x + 2) ^ 2 + (y - 2) ^ 2 = (x - 2) ^ 2 + y ^ 2 →
  y = 2 * x + 1 :=
  sorry  -- Proof is omitted

end coordinates_equidistant_l1120_112001


namespace distance_small_ball_to_surface_l1120_112026

-- Define the main variables and conditions
variables (R : ℝ)

-- Define the conditions of the problem
def bottomBallRadius : ℝ := 2 * R
def topBallRadius : ℝ := R
def edgeLengthBaseTetrahedron : ℝ := 4 * R
def edgeLengthLateralTetrahedron : ℝ := 3 * R

-- Define the main statement in Lean format
theorem distance_small_ball_to_surface (R : ℝ) :
  (3 * R) = R + bottomBallRadius R :=
sorry

end distance_small_ball_to_surface_l1120_112026


namespace distance_to_Rock_Mist_Mountains_l1120_112071

theorem distance_to_Rock_Mist_Mountains
  (d_Sky_Falls : ℕ) (d_Sky_Falls_eq : d_Sky_Falls = 8)
  (d_Rock_Mist : ℕ) (d_Rock_Mist_eq : d_Rock_Mist = 50 * d_Sky_Falls)
  (detour_Thunder_Pass : ℕ) (detour_Thunder_Pass_eq : detour_Thunder_Pass = 25) :
  d_Rock_Mist + detour_Thunder_Pass = 425 := by
  sorry

end distance_to_Rock_Mist_Mountains_l1120_112071


namespace nest_building_twig_count_l1120_112057

theorem nest_building_twig_count
    (total_twigs_to_weave : ℕ)
    (found_twigs : ℕ)
    (remaining_twigs : ℕ)
    (n : ℕ)
    (x : ℕ)
    (h1 : total_twigs_to_weave = 12 * x)
    (h2 : found_twigs = (total_twigs_to_weave) / 3)
    (h3 : remaining_twigs = 48)
    (h4 : found_twigs + remaining_twigs = total_twigs_to_weave) :
    x = 18 := 
by
  sorry

end nest_building_twig_count_l1120_112057


namespace counterexample_statement_l1120_112095

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

theorem counterexample_statement (n : ℕ) : is_composite n ∧ (is_prime (n - 3) ∨ is_prime (n - 2)) ↔ n = 22 :=
by
  sorry

end counterexample_statement_l1120_112095


namespace total_rainfall_2004_l1120_112070

theorem total_rainfall_2004 (avg_2003 : ℝ) (increment : ℝ) (months : ℕ) (total_2004 : ℝ) 
  (h1 : avg_2003 = 41.5) 
  (h2 : increment = 2) 
  (h3 : months = 12) 
  (h4 : total_2004 = avg_2003 + increment * months) :
  total_2004 = 522 :=
by 
  sorry

end total_rainfall_2004_l1120_112070


namespace T_8_equals_546_l1120_112019

-- Define the sum of the first n natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the sum of the squares of the first n natural numbers
def sum_squares_first_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Define T_n based on the given formula
def T (n : ℕ) : ℕ := (sum_first_n n ^ 2 - sum_squares_first_n n) / 2

-- The proof statement we need to prove
theorem T_8_equals_546 : T 8 = 546 := sorry

end T_8_equals_546_l1120_112019


namespace base_h_addition_eq_l1120_112030

theorem base_h_addition_eq (h : ℕ) (h_eq : h = 9) : 
  (8 * h^3 + 3 * h^2 + 7 * h + 4) + (6 * h^3 + 9 * h^2 + 2 * h + 5) = 1 * h^4 + 5 * h^3 + 3 * h^2 + 0 * h + 9 :=
by
  rw [h_eq]
  sorry

end base_h_addition_eq_l1120_112030


namespace range_of_a_l1120_112049

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a_l1120_112049


namespace range_of_m_l1120_112059

noncomputable def f (x m : ℝ) := Real.exp x + x^2 / m^2 - x

theorem range_of_m (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 -> b ∈ Set.Icc (-1) 1 -> |f a m - f b m| ≤ Real.exp 1) ↔
  (m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_m_l1120_112059


namespace orange_cost_l1120_112073

-- Definitions based on the conditions
def dollar_per_pound := 5 / 6
def pounds : ℕ := 18
def total_cost := pounds * dollar_per_pound

-- The statement to be proven
theorem orange_cost : total_cost = 15 :=
by
  sorry

end orange_cost_l1120_112073


namespace polygon_sides_l1120_112035

theorem polygon_sides
  (n : ℕ)
  (h1 : 180 * (n - 2) - (2 * (2790 / (n - 1)) - 20) = 2790) :
  n = 18 := sorry

end polygon_sides_l1120_112035


namespace findFirstCarSpeed_l1120_112097

noncomputable def firstCarSpeed (v : ℝ) (blackCarSpeed : ℝ) (initialGap : ℝ) (timeToCatchUp : ℝ) : Prop :=
  blackCarSpeed * timeToCatchUp = initialGap + v * timeToCatchUp → v = 30

theorem findFirstCarSpeed :
  firstCarSpeed 30 50 20 1 :=
by
  sorry

end findFirstCarSpeed_l1120_112097


namespace find_m_and_n_l1120_112051

theorem find_m_and_n (a b c d m n : ℕ) (h1 : a^2 + b^2 + c^2 + d^2 = 1989) 
                    (h2 : a + b + c + d = m^2) 
                    (h3 : max a (max b (max c d)) = n^2) : 
                    m = 9 ∧ n = 6 := 
sorry

end find_m_and_n_l1120_112051


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l1120_112054

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l1120_112054


namespace medium_pizza_slices_l1120_112072

theorem medium_pizza_slices (M : ℕ) 
  (small_pizza_slices : ℕ := 6)
  (large_pizza_slices : ℕ := 12)
  (total_pizzas : ℕ := 15)
  (small_pizzas : ℕ := 4)
  (medium_pizzas : ℕ := 5)
  (total_slices : ℕ := 136) :
  (small_pizzas * small_pizza_slices) + (medium_pizzas * M) + ((total_pizzas - small_pizzas - medium_pizzas) * large_pizza_slices) = total_slices → 
  M = 8 :=
by
  intro h
  sorry

end medium_pizza_slices_l1120_112072


namespace GCF_LCM_18_30_10_45_eq_90_l1120_112099

-- Define LCM and GCF functions
def LCM (a b : ℕ) := a / Nat.gcd a b * b
def GCF (a b : ℕ) := Nat.gcd a b

-- Define the problem
theorem GCF_LCM_18_30_10_45_eq_90 : 
  GCF (LCM 18 30) (LCM 10 45) = 90 := by
sorry

end GCF_LCM_18_30_10_45_eq_90_l1120_112099


namespace triangle_side_eq_median_l1120_112079

theorem triangle_side_eq_median (A B C : Type) (a b c : ℝ) (hAB : a = 2) (hAC : b = 3) (hBC_eq_median : c = (2 * (Real.sqrt (13 / 10)))) :
  c = (Real.sqrt 130) / 5 := by
  sorry

end triangle_side_eq_median_l1120_112079


namespace m_above_x_axis_m_on_line_l1120_112089

namespace ComplexNumberProblem

def above_x_axis (m : ℝ) : Prop :=
  m^2 - 2 * m - 15 > 0

def on_line (m : ℝ) : Prop :=
  2 * m^2 + 3 * m - 4 = 0

theorem m_above_x_axis (m : ℝ) : above_x_axis m → (m < -3 ∨ m > 5) :=
  sorry

theorem m_on_line (m : ℝ) : on_line m → 
  (m = (-3 + Real.sqrt 41) / 4) ∨ (m = (-3 - Real.sqrt 41) / 4) :=
  sorry

end ComplexNumberProblem

end m_above_x_axis_m_on_line_l1120_112089


namespace value_of_b_l1120_112048

-- Defining the number sum in circles and overlap
def circle_sum := 21
def num_circles := 5
def total_sum := 69

-- Overlapping numbers
def overlap_1 := 2
def overlap_2 := 8
def overlap_3 := 9
variable (b d : ℕ)

-- Circle equation containing d
def circle_with_d := d + 5 + 9

-- Prove b = 10 given the conditions
theorem value_of_b (h₁ : num_circles * circle_sum = 105)
    (h₂ : 105 - (overlap_1 + overlap_2 + overlap_3 + b + d) = total_sum)
    (h₃ : circle_with_d d = 21) : b = 10 :=
by sorry

end value_of_b_l1120_112048


namespace young_people_sampled_l1120_112050

def num_young_people := 800
def num_middle_aged_people := 1600
def num_elderly_people := 1400
def sampled_elderly_people := 70

-- Lean statement to prove the number of young people sampled
theorem young_people_sampled : 
  (sampled_elderly_people:ℝ) / num_elderly_people = (1 / 20:ℝ) ->
  num_young_people * (1 / 20:ℝ) = 40 := by
  sorry

end young_people_sampled_l1120_112050


namespace bryce_received_12_raisins_l1120_112024

-- Defining the main entities for the problem
variables {x y z : ℕ} -- number of raisins Bryce, Carter, and Emma received respectively

-- Conditions:
def condition1 (x y : ℕ) : Prop := y = x - 8
def condition2 (x y : ℕ) : Prop := y = x / 3
def condition3 (y z : ℕ) : Prop := z = 2 * y

-- The goal is to prove that Bryce received 12 raisins
theorem bryce_received_12_raisins (x y z : ℕ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) 
  (h3 : condition3 y z) : 
  x = 12 :=
sorry

end bryce_received_12_raisins_l1120_112024


namespace find_original_money_sandy_took_l1120_112033

noncomputable def originalMoney (remainingMoney : ℝ) (clothingPercent electronicsPercent foodPercent additionalSpendPercent salesTaxPercent : ℝ) : Prop :=
  let X := (remainingMoney / (1 - ((clothingPercent + electronicsPercent + foodPercent) + additionalSpendPercent) * (1 + salesTaxPercent)))
  abs (X - 397.73) < 0.01

theorem find_original_money_sandy_took :
  originalMoney 140 0.25 0.15 0.10 0.20 0.08 :=
sorry

end find_original_money_sandy_took_l1120_112033


namespace possible_r_values_l1120_112002

noncomputable def triangle_area (r : ℝ) : ℝ := (r - 3) ^ (3 / 2)

theorem possible_r_values :
  {r : ℝ | 16 ≤ triangle_area r ∧ triangle_area r ≤ 128} = {r : ℝ | 7 ≤ r ∧ r ≤ 19} :=
by
  sorry

end possible_r_values_l1120_112002


namespace fg_of_3_l1120_112047

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem fg_of_3 : f (g 3) = 25 := by
  sorry

end fg_of_3_l1120_112047


namespace new_quadratic_equation_has_square_roots_l1120_112055

theorem new_quadratic_equation_has_square_roots (p q : ℝ) (x : ℝ) :
  (x^2 + px + q = 0 → ∃ x1 x2 : ℝ, x^2 - (p^2 - 2 * q) * x + q^2 = 0 ∧ (x1^2 = x ∨ x2^2 = x)) :=
by sorry

end new_quadratic_equation_has_square_roots_l1120_112055


namespace range_a_inequality_l1120_112083

theorem range_a_inequality (a : ℝ) : (∀ x : ℝ, (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ 1 < a ∧ a ≤ 2 :=
by {
    sorry
}

end range_a_inequality_l1120_112083


namespace value_of_4x_l1120_112075

variable (x : ℤ)

theorem value_of_4x (h : 2 * x - 3 = 10) : 4 * x = 26 := 
by
  sorry

end value_of_4x_l1120_112075


namespace confidence_of_independence_test_l1120_112096

-- Define the observed value of K^2
def K2_obs : ℝ := 5

-- Define the critical value(s) of K^2 for different confidence levels
def K2_critical_0_05 : ℝ := 3.841
def K2_critical_0_01 : ℝ := 6.635

-- Define the confidence levels corresponding to the critical values
def P_K2_ge_3_841 : ℝ := 0.05
def P_K2_ge_6_635 : ℝ := 0.01

-- Define the statement to be proved: there is 95% confidence that "X and Y are related".
theorem confidence_of_independence_test
  (K2_obs K2_critical_0_05 P_K2_ge_3_841 : ℝ)
  (hK2_obs_gt_critical : K2_obs > K2_critical_0_05)
  (hP : P_K2_ge_3_841 = 0.05) :
  1 - P_K2_ge_3_841 = 0.95 :=
by
  -- The proof is omitted
  sorry

end confidence_of_independence_test_l1120_112096


namespace ratio_SP2_SP1_l1120_112003

variable (CP : ℝ)

-- First condition: Sold at a profit of 140%
def SP1 := 2.4 * CP

-- Second condition: Sold at a loss of 20%
def SP2 := 0.8 * CP

-- Statement: The ratio of SP2 to SP1 is 1 to 3
theorem ratio_SP2_SP1 : SP2 / SP1 = 1 / 3 :=
by
  sorry

end ratio_SP2_SP1_l1120_112003


namespace solution_set_condition_l1120_112094

theorem solution_set_condition (a : ℝ) :
  (∀ x : ℝ, x * (x - a + 1) > a ↔ (x < -1 ∨ x > a)) → a > -1 :=
sorry

end solution_set_condition_l1120_112094


namespace sarah_cupcakes_ratio_l1120_112040

theorem sarah_cupcakes_ratio (total_cupcakes : ℕ) (cookies_from_michael : ℕ) 
    (final_desserts : ℕ) (cupcakes_given : ℕ) (h1 : total_cupcakes = 9) 
    (h2 : cookies_from_michael = 5) (h3 : final_desserts = 11) 
    (h4 : total_cupcakes - cupcakes_given + cookies_from_michael = final_desserts) : 
    cupcakes_given / total_cupcakes = 1 / 3 :=
by
  sorry

end sarah_cupcakes_ratio_l1120_112040


namespace find_m_l1120_112077

theorem find_m (x y m : ℝ)
  (h1 : 2 * x + y = 6 * m)
  (h2 : 3 * x - 2 * y = 2 * m)
  (h3 : x / 3 - y / 5 = 4) :
  m = 15 :=
by
  sorry

end find_m_l1120_112077


namespace second_pump_drain_time_l1120_112064

-- Definitions of the rates R1 and R2
def R1 : ℚ := 1 / 12  -- Rate of the first pump
def R2 : ℚ := 1 - R1  -- Rate of the second pump (from the combined rate equation)

-- The time it takes the second pump alone to drain the pond
def time_to_drain_second_pump := 1 / R2

-- The goal is to prove that this value is 12/11
theorem second_pump_drain_time : time_to_drain_second_pump = 12 / 11 := by
  -- The proof is omitted
  sorry

end second_pump_drain_time_l1120_112064


namespace number_of_pencil_boxes_l1120_112082

open Nat

def books_per_box : Nat := 46
def num_book_boxes : Nat := 19
def pencils_per_box : Nat := 170
def total_books_and_pencils : Nat := 1894

theorem number_of_pencil_boxes :
  (total_books_and_pencils - (num_book_boxes * books_per_box)) / pencils_per_box = 6 := 
by
  sorry

end number_of_pencil_boxes_l1120_112082


namespace correct_operation_l1120_112085

theorem correct_operation (a b : ℝ) :
  (a + b) * (b - a) = b^2 - a^2 :=
by
  sorry

end correct_operation_l1120_112085


namespace distance_planes_A_B_l1120_112058

noncomputable def distance_between_planes : ℝ :=
  let d1 := 1
  let d2 := 2
  let a := 1
  let b := 1
  let c := 1
  (|d2 - d1|) / (Real.sqrt (a^2 + b^2 + c^2))

theorem distance_planes_A_B :
  let A := fun (x y z : ℝ) => x + y + z = 1
  let B := fun (x y z : ℝ) => x + y + z = 2
  distance_between_planes = 1 / Real.sqrt 3 :=
  by
    -- Proof steps will be here
    sorry

end distance_planes_A_B_l1120_112058


namespace inequality_pgcd_l1120_112007

theorem inequality_pgcd (a b : ℕ) (h1 : a > b) (h2 : (a - b) ∣ (a^2 + b)) : 
  (a + 1) / (b + 1) ≤ Nat.gcd a b + 1 := 
sorry

end inequality_pgcd_l1120_112007


namespace peregrine_falcon_dive_time_l1120_112090

theorem peregrine_falcon_dive_time 
  (bald_eagle_speed : ℝ := 100) 
  (peregrine_falcon_speed : ℝ := 2 * bald_eagle_speed) 
  (bald_eagle_time : ℝ := 30) : 
  peregrine_falcon_speed = 2 * bald_eagle_speed ∧ peregrine_falcon_speed / bald_eagle_speed = 2 →
  ∃ peregrine_falcon_time : ℝ, peregrine_falcon_time = 15 :=
by
  intro h
  use (bald_eagle_time / 2)
  sorry

end peregrine_falcon_dive_time_l1120_112090


namespace polygon_intersections_inside_circle_l1120_112010

noncomputable def number_of_polygon_intersections
    (polygonSides: List Nat) : Nat :=
  let pairs := [(4,5), (4,7), (4,9), (5,7), (5,9), (7,9)]
  pairs.foldl (λ acc (p1, p2) => acc + 2 * min p1 p2) 0

theorem polygon_intersections_inside_circle :
  number_of_polygon_intersections [4, 5, 7, 9] = 58 :=
by
  sorry

end polygon_intersections_inside_circle_l1120_112010


namespace complement_of_A_relative_to_U_l1120_112068

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {x | 1 ≤ x ∧ x ≤ 3}

theorem complement_of_A_relative_to_U : (U \ A) = {4, 5, 6} := 
by
  sorry

end complement_of_A_relative_to_U_l1120_112068


namespace age_difference_l1120_112053

variables (P M Mo : ℕ)

def patrick_michael_ratio (P M : ℕ) : Prop := (P * 5 = M * 3)
def michael_monica_ratio (M Mo : ℕ) : Prop := (M * 4 = Mo * 3)
def sum_of_ages (P M Mo : ℕ) : Prop := (P + M + Mo = 88)

theorem age_difference (P M Mo : ℕ) : 
  patrick_michael_ratio P M → 
  michael_monica_ratio M Mo → 
  sum_of_ages P M Mo → 
  (Mo - P = 22) :=
by
  sorry

end age_difference_l1120_112053


namespace ratio_of_surface_areas_l1120_112034

theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 4 :=
by
  sorry

end ratio_of_surface_areas_l1120_112034


namespace sin_cos_15_degrees_proof_l1120_112020

noncomputable
def sin_cos_15_degrees : Prop := (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4)

theorem sin_cos_15_degrees_proof : sin_cos_15_degrees :=
by
  sorry

end sin_cos_15_degrees_proof_l1120_112020


namespace fraction_transformation_l1120_112039

variables (a b : ℝ)

theorem fraction_transformation (ha : a ≠ 0) (hb : b ≠ 0) : 
  (4 * a * b) / (2 * (2 * a) + 2 * b) = 2 * (a * b) / (2 * a + b) :=
by
  sorry

end fraction_transformation_l1120_112039


namespace find_k_l1120_112043

variable (k : ℝ) (t : ℝ) (a : ℝ)

theorem find_k (h1 : t = (5 / 9) * (k - 32) + a * k) (h2 : t = 20) (h3 : a = 3) : k = 10.625 := by
  sorry

end find_k_l1120_112043


namespace original_number_l1120_112063

theorem original_number (x : ℝ) (h : 1.4 * x = 700) : x = 500 :=
sorry

end original_number_l1120_112063


namespace simplify_sum_l1120_112076

theorem simplify_sum :
  -2^2004 + (-2)^2005 + 2^2006 - 2^2007 = -2^2004 - 2^2005 + 2^2006 - 2^2007 :=
by
  sorry

end simplify_sum_l1120_112076


namespace find_algebraic_expression_value_l1120_112052

theorem find_algebraic_expression_value (x : ℝ) (h : 3 * x^2 + 5 * x + 1 = 0) : 
  (x + 2) ^ 2 + x * (2 * x + 1) = 3 := 
by 
  -- Proof steps go here
  sorry

end find_algebraic_expression_value_l1120_112052


namespace solve_equation_l1120_112008

theorem solve_equation (x : ℝ) (hx : x ≠ 1) : (x / (x - 1) - 1 = 1) → (x = 2) :=
by
  sorry

end solve_equation_l1120_112008


namespace number_is_16_l1120_112081

theorem number_is_16 (n : ℝ) (h : (1/2) * n + 5 = 13) : n = 16 :=
sorry

end number_is_16_l1120_112081


namespace sum_of_pqrstu_eq_22_l1120_112065

theorem sum_of_pqrstu_eq_22 (p q r s t : ℤ) 
  (h : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48) : 
  p + q + r + s + t = 22 :=
sorry

end sum_of_pqrstu_eq_22_l1120_112065


namespace cubic_inches_in_two_cubic_feet_l1120_112061

-- Define the conversion factor between feet and inches
def foot_to_inch : ℕ := 12
-- Define the conversion factor between cubic feet and cubic inches
def cubic_foot_to_cubic_inch : ℕ := foot_to_inch ^ 3

-- State the theorem to be proved
theorem cubic_inches_in_two_cubic_feet : 2 * cubic_foot_to_cubic_inch = 3456 :=
by
  -- Proof steps go here
  sorry

end cubic_inches_in_two_cubic_feet_l1120_112061


namespace intersection_complement_M_N_eq_456_l1120_112027

def UniversalSet := { n : ℕ | 1 ≤ n ∧ n < 9 }
def M : Set ℕ := { 1, 2, 3 }
def N : Set ℕ := { 3, 4, 5, 6 }

theorem intersection_complement_M_N_eq_456 : 
  (UniversalSet \ M) ∩ N = { 4, 5, 6 } :=
by
  sorry

end intersection_complement_M_N_eq_456_l1120_112027


namespace c_horses_months_l1120_112086

theorem c_horses_months (cost_total Rs_a Rs_b num_horses_a num_months_a num_horses_b num_months_b num_horses_c amount_paid_b : ℕ) (x : ℕ) 
  (h1 : cost_total = 841) 
  (h2 : Rs_a = 12 * 8)
  (h3 : Rs_b = 16 * 9)
  (h4 : amount_paid_b = 348)
  (h5 : 96 * (amount_paid_b / Rs_b) + (18 * x) * (amount_paid_b / Rs_b) = cost_total - amount_paid_b) :
  x = 11 :=
sorry

end c_horses_months_l1120_112086


namespace lisa_eats_correct_number_of_pieces_l1120_112005

variable (M A K R L : ℚ) -- All variables are rational numbers (real numbers could also be used)
variable (n : ℕ) -- n is a natural number (the number of pieces of lasagna)

-- Let's define the conditions succinctly
def manny_wants_one_piece := M = 1
def aaron_eats_nothing := A = 0
def kai_eats_twice_manny := K = 2 * M
def raphael_eats_half_manny := R = 0.5 * M
def lasagna_is_cut_into_6_pieces := n = 6

-- The proof goal is to show Lisa eats 2.5 pieces
theorem lisa_eats_correct_number_of_pieces (M A K R L : ℚ) (n : ℕ) :
  manny_wants_one_piece M →
  aaron_eats_nothing A →
  kai_eats_twice_manny M K →
  raphael_eats_half_manny M R →
  lasagna_is_cut_into_6_pieces n →
  L = n - (M + K + R) →
  L = 2.5 :=
by
  intros hM hA hK hR hn hL
  sorry  -- Proof omitted

end lisa_eats_correct_number_of_pieces_l1120_112005


namespace initial_amount_is_1875_l1120_112093

-- Defining the conditions as given in the problem
def initial_amount : ℝ := sorry
def spent_on_clothes : ℝ := 250
def spent_on_food (remaining : ℝ) : ℝ := 0.35 * remaining
def spent_on_electronics (remaining : ℝ) : ℝ := 0.50 * remaining

-- Given conditions
axiom condition1 : initial_amount - spent_on_clothes = sorry
axiom condition2 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) = sorry
axiom condition3 : initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes) - spent_on_electronics (initial_amount - spent_on_clothes - spent_on_food (initial_amount - spent_on_clothes)) = 200

-- Prove that initial amount is $1875
theorem initial_amount_is_1875 : initial_amount = 1875 :=
sorry

end initial_amount_is_1875_l1120_112093


namespace probability_of_at_least_six_heads_is_correct_l1120_112046

-- Definitions for the given problem
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

def total_possible_outcomes : ℕ :=
  2^8

def favorable_outcomes : ℕ :=
  binomial_coefficient 8 6 + binomial_coefficient 8 7 + binomial_coefficient 8 8

def probability_of_at_least_6_heads : ℚ :=
  favorable_outcomes / total_possible_outcomes

-- The proof statement
theorem probability_of_at_least_six_heads_is_correct :
  probability_of_at_least_6_heads = 37 / 256 :=
by sorry

end probability_of_at_least_six_heads_is_correct_l1120_112046


namespace amount_left_in_wallet_l1120_112037

theorem amount_left_in_wallet
  (initial_amount : ℝ)
  (spent_amount : ℝ)
  (h_initial : initial_amount = 94)
  (h_spent : spent_amount = 16) :
  initial_amount - spent_amount = 78 :=
by
  sorry

end amount_left_in_wallet_l1120_112037


namespace g_triple_application_l1120_112078

def g (x : ℕ) : ℕ := 7 * x + 3

theorem g_triple_application : g (g (g 3)) = 1200 :=
by
  sorry

end g_triple_application_l1120_112078
