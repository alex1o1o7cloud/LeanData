import Mathlib

namespace sugar_needed_for_40_cookies_l258_25814

def num_cookies_per_cup_flour (a : ℕ) (b : ℕ) : ℕ := a / b

def cups_of_flour_needed (num_cookies : ℕ) (cookies_per_cup : ℕ) : ℕ := num_cookies / cookies_per_cup

def cups_of_sugar_needed (cups_flour : ℕ) (flour_to_sugar_ratio_num : ℕ) (flour_to_sugar_ratio_denom : ℕ) : ℚ := 
  (flour_to_sugar_ratio_denom * cups_flour : ℚ) / flour_to_sugar_ratio_num

theorem sugar_needed_for_40_cookies :
  let num_flour_to_make_24_cookies := 3
  let cookies := 24
  let ratio_num := 3
  let ratio_denom := 2
  num_cookies_per_cup_flour cookies num_flour_to_make_24_cookies = 8 →
  cups_of_flour_needed 40 8 = 5 →
  cups_of_sugar_needed 5 ratio_num ratio_denom = 10 / 3 :=
by 
  sorry

end sugar_needed_for_40_cookies_l258_25814


namespace eccentricity_of_ellipse_l258_25877

-- Definitions
variable (a b c : ℝ)  -- semi-major axis, semi-minor axis, and distance from center to a focus
variable (h_c_eq_b : c = b)  -- given condition focal length equals length of minor axis
variable (h_a_eq_sqrt_sum : a = Real.sqrt (c^2 + b^2))  -- relationship in ellipse

-- Question: Prove the eccentricity of the ellipse e = √2 / 2
theorem eccentricity_of_ellipse : (c = b) → (a = Real.sqrt (c^2 + b^2)) → (c / a = Real.sqrt 2 / 2) :=
by
  intros h_c_eq_b h_a_eq_sqrt_sum
  sorry

end eccentricity_of_ellipse_l258_25877


namespace find_marks_in_physics_l258_25818

theorem find_marks_in_physics (P C M : ℕ) (h1 : P + C + M = 225) (h2 : P + M = 180) (h3 : P + C = 140) : 
    P = 95 :=
sorry

end find_marks_in_physics_l258_25818


namespace largest_possible_integer_in_list_l258_25851

theorem largest_possible_integer_in_list :
  ∃ (a b c d e : ℕ), 
  (a = 6) ∧ 
  (b = 6) ∧ 
  (c = 7) ∧ 
  (∀ x, x ≠ a ∨ x ≠ b ∨ x ≠ c → x ≠ 6) ∧ 
  (d > 7) ∧ 
  (12 = (a + b + c + d + e) / 5) ∧ 
  (max a (max b (max c (max d e))) = 33) := by
  sorry

end largest_possible_integer_in_list_l258_25851


namespace utility_bills_total_l258_25886

-- Define the conditions
def fifty_bills := 3
def ten_dollar_bills := 2
def fifty_dollar_value := 50
def ten_dollar_value := 10

-- Prove the total utility bills amount
theorem utility_bills_total : (fifty_bills * fifty_dollar_value + ten_dollar_bills * ten_dollar_value) = 170 := by
  sorry

end utility_bills_total_l258_25886


namespace certainEvent_l258_25835

def scoopingTheMoonOutOfTheWaterMeansCertain : Prop :=
  ∀ (e : String), e = "scooping the moon out of the water" → (∀ (b : Bool), b = true)

theorem certainEvent (e : String) (h : e = "scooping the moon out of the water") : ∀ (b : Bool), b = true :=
  by
  sorry

end certainEvent_l258_25835


namespace binomial_10_3_l258_25893

theorem binomial_10_3 : Nat.choose 10 3 = 120 := 
by 
  sorry

end binomial_10_3_l258_25893


namespace probability_53_sundays_in_leap_year_l258_25879

-- Define the conditions
def num_days_in_leap_year : ℕ := 366
def num_weeks_in_leap_year : ℕ := 52
def extra_days_in_leap_year : ℕ := 2
def num_combinations : ℕ := 7
def num_sunday_combinations : ℕ := 2

-- Define the problem statement
theorem probability_53_sundays_in_leap_year (hdays : num_days_in_leap_year = 52 * 7 + extra_days_in_leap_year) :
  (num_sunday_combinations / num_combinations : ℚ) = 2 / 7 :=
by
  sorry

end probability_53_sundays_in_leap_year_l258_25879


namespace razorback_tshirt_revenue_l258_25819

theorem razorback_tshirt_revenue 
    (total_tshirts : ℕ) (total_money : ℕ) 
    (h1 : total_tshirts = 245) 
    (h2 : total_money = 2205) : 
    (total_money / total_tshirts = 9) := 
by 
    sorry

end razorback_tshirt_revenue_l258_25819


namespace total_hatched_eggs_l258_25802

noncomputable def fertile_eggs (total_eggs : ℕ) (infertility_rate : ℝ) : ℝ :=
  total_eggs * (1 - infertility_rate)

noncomputable def hatching_eggs_after_calcification (fertile_eggs : ℝ) (calcification_rate : ℝ) : ℝ :=
  fertile_eggs * (1 - calcification_rate)

noncomputable def hatching_eggs_after_predator (hatching_eggs : ℝ) (predator_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - predator_rate)

noncomputable def hatching_eggs_after_temperature (hatching_eggs : ℝ) (temperature_rate : ℝ) : ℝ :=
  hatching_eggs * (1 - temperature_rate)

open Nat

theorem total_hatched_eggs :
  let g1_total_eggs := 30
  let g2_total_eggs := 40
  let g1_infertility_rate := 0.20
  let g2_infertility_rate := 0.25
  let g1_calcification_rate := 1.0 / 3.0
  let g2_calcification_rate := 0.25
  let predator_rate := 0.10
  let temperature_rate := 0.05
  let g1_fertile := fertile_eggs g1_total_eggs g1_infertility_rate
  let g1_hatch_calcification := hatching_eggs_after_calcification g1_fertile g1_calcification_rate
  let g1_hatch_predator := hatching_eggs_after_predator g1_hatch_calcification predator_rate
  let g1_hatch_temp := hatching_eggs_after_temperature g1_hatch_predator temperature_rate
  let g2_fertile := fertile_eggs g2_total_eggs g2_infertility_rate
  let g2_hatch_calcification := hatching_eggs_after_calcification g2_fertile g2_calcification_rate
  let g2_hatch_predator := hatching_eggs_after_predator g2_hatch_calcification predator_rate
  let g2_hatch_temp := hatching_eggs_after_temperature g2_hatch_predator temperature_rate
  let total_hatched := g1_hatch_temp + g2_hatch_temp
  floor total_hatched = 32 :=
by
  sorry

end total_hatched_eggs_l258_25802


namespace simplify_expr_l258_25863

-- Define the condition
def y : ℕ := 77

-- Define the expression and the expected result
def expr := (7 * y + 77) / 77

-- The theorem statement
theorem simplify_expr : expr = 8 :=
by
  sorry

end simplify_expr_l258_25863


namespace distance_to_hospital_l258_25801

theorem distance_to_hospital {total_paid base_price price_per_mile : ℝ} (h1 : total_paid = 23) (h2 : base_price = 3) (h3 : price_per_mile = 4) : (total_paid - base_price) / price_per_mile = 5 :=
by
  sorry

end distance_to_hospital_l258_25801


namespace max_triangle_area_l258_25858

noncomputable def max_area_of_triangle (a b c S : ℝ) : ℝ := 
if h : 4 * S = a^2 - (b - c)^2 ∧ b + c = 4 then 
  2 
else
  sorry

-- The statement we want to prove
theorem max_triangle_area : ∀ (a b c S : ℝ),
  (4 * S = a^2 - (b - c)^2) →
  (b + c = 4) →
  S ≤ max_area_of_triangle a b c S ∧ max_area_of_triangle a b c S = 2 :=
by sorry

end max_triangle_area_l258_25858


namespace h_comp_h_3_l258_25855

def h (x : ℕ) : ℕ := 3 * x * x + 5 * x - 3

theorem h_comp_h_3 : h (h 3) = 4755 := by
  sorry

end h_comp_h_3_l258_25855


namespace required_hours_for_fifth_week_l258_25896

def typical_hours_needed (week1 week2 week3 week4 week5 add_hours total_weeks target_avg : ℕ) : ℕ :=
  if (week1 + week2 + week3 + week4 + week5 + add_hours) / total_weeks = target_avg then 
    week5 
  else 
    0

theorem required_hours_for_fifth_week :
  typical_hours_needed 10 14 11 9 x 1 5 12 = 15 :=
by
  sorry

end required_hours_for_fifth_week_l258_25896


namespace extrema_range_l258_25841

noncomputable def hasExtrema (a : ℝ) : Prop :=
  (4 * a^2 + 12 * a > 0)

theorem extrema_range (a : ℝ) : hasExtrema a ↔ (a < -3 ∨ a > 0) := sorry

end extrema_range_l258_25841


namespace first_discount_calculation_l258_25846

-- Define the given conditions and final statement
theorem first_discount_calculation (P : ℝ) (D : ℝ) :
  (1.35 * (1 - D / 100) * 0.85 = 1.03275) → (D = 10.022) :=
by
  -- Proof is not provided, to be done.
  sorry

end first_discount_calculation_l258_25846


namespace range_of_a_l258_25830

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (2^x - 2^(-x))
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * (2^x + 2^(-x))

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * f x + g (2 * x) ≥ 0) ↔ a ≥ -17 / 6 :=
by
  sorry

end range_of_a_l258_25830


namespace gcd_of_36_between_70_and_85_is_81_l258_25825

theorem gcd_of_36_between_70_and_85_is_81 {n : ℕ} (h1 : n ≥ 70) (h2 : n ≤ 85) (h3 : Nat.gcd 36 n = 9) : n = 81 :=
by
  -- proof
  sorry

end gcd_of_36_between_70_and_85_is_81_l258_25825


namespace bob_fencing_needed_l258_25827

-- Problem conditions
def length : ℕ := 225
def width : ℕ := 125
def small_gate : ℕ := 3
def large_gate : ℕ := 10

-- Definition of perimeter
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Total width of the gates
def total_gate_width (g1 g2 : ℕ) : ℕ := g1 + g2

-- Amount of fencing needed
def fencing_needed (p gw : ℕ) : ℕ := p - gw

-- Theorem statement
theorem bob_fencing_needed :
  fencing_needed (perimeter length width) (total_gate_width small_gate large_gate) = 687 :=
by 
  sorry

end bob_fencing_needed_l258_25827


namespace custom_op_example_l258_25888

def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

theorem custom_op_example : custom_op (-4) 6 = 68 :=
by
  sorry

end custom_op_example_l258_25888


namespace quadratic_eq_has_equal_roots_l258_25847

theorem quadratic_eq_has_equal_roots (q : ℚ) :
  (∃ x : ℚ, x^2 - 3 * x + q = 0 ∧ (x^2 - 3 * x + q = 0)) → q = 9 / 4 :=
by
  sorry

end quadratic_eq_has_equal_roots_l258_25847


namespace range_of_x_l258_25840

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + (a - 4) * x + 4 - 2 * a

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  ∀ x : ℝ, (f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  intro x
  sorry

end range_of_x_l258_25840


namespace vessel_base_length_l258_25809

variables (L : ℝ) (edge : ℝ) (W : ℝ) (h : ℝ)
def volume_cube := edge^3
def volume_rise := L * W * h

theorem vessel_base_length :
  (volume_cube 16 = volume_rise L 15 13.653333333333334) →
  L = 20 :=
by sorry

end vessel_base_length_l258_25809


namespace find_deductive_reasoning_l258_25815

noncomputable def is_deductive_reasoning (reasoning : String) : Prop :=
  match reasoning with
  | "B" => true
  | _ => false

theorem find_deductive_reasoning : is_deductive_reasoning "B" = true :=
  sorry

end find_deductive_reasoning_l258_25815


namespace negation_universal_proposition_l258_25839

theorem negation_universal_proposition :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by sorry

end negation_universal_proposition_l258_25839


namespace product_evaluation_l258_25844

theorem product_evaluation :
  (1 / 2) * 4 * (1 / 8) * 16 * (1 / 32) * 64 * (1 / 128) * 256 *
  (1 / 512) * 1024 * (1 / 2048) * 4096 = 64 :=
by
  sorry

end product_evaluation_l258_25844


namespace correct_relationships_l258_25824

open Real

theorem correct_relationships (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a + c > b + c) ∧ (1/a < 1/b) := by
    sorry

end correct_relationships_l258_25824


namespace arithmetic_progression_probability_l258_25820

theorem arithmetic_progression_probability (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_outcomes = 6^4 ∧ favorable_outcomes = 3 →
  favorable_outcomes / total_outcomes = 1 / 432 :=
by
  sorry

end arithmetic_progression_probability_l258_25820


namespace ellipse_eccentricity_l258_25803

theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b : b > 0) (c : ℝ)
  (h_ellipse : (b^2 / c^2) = 3)
  (eccentricity_eq : ∀ (e : ℝ), e = c / a ↔ e = 1 / 2) : 
  ∃ e, e = (c / a) :=
by {
  sorry
}

end ellipse_eccentricity_l258_25803


namespace speed_of_water_l258_25897

variable (v : ℝ)
variable (swimming_speed_still_water : ℝ := 10)
variable (time_against_current : ℝ := 8)
variable (distance_against_current : ℝ := 16)

theorem speed_of_water :
  distance_against_current = (swimming_speed_still_water - v) * time_against_current ↔ v = 8 := by
  sorry

end speed_of_water_l258_25897


namespace possible_values_of_b_l258_25881

theorem possible_values_of_b (r s : ℝ) (t t' : ℝ)
  (hp : ∀ x, x^3 + a * x + b = 0 → (x = r ∨ x = s ∨ x = t))
  (hq : ∀ x, x^3 + a * x + b + 240 = 0 → (x = r + 4 ∨ x = s - 3 ∨ x = t'))
  (h_sum_p : r + s + t = 0)
  (h_sum_q : (r + 4) + (s - 3) + t' = 0)
  (ha_p : a = r * s + r * t + s * t)
  (ha_q : a = (r + 4) * (s - 3) + (r + 4) * (t' - 1) + (s - 3) * (t' - 1))
  (ht'_def : t' = t - 1)
  : b = -330 ∨ b = 90 :=
by
  sorry

end possible_values_of_b_l258_25881


namespace determine_parabola_l258_25837

-- Define the parabola passing through point P(1,1)
def parabola_passing_through (a b c : ℝ) :=
  (1:ℝ)^2 * a + 1 * b + c = 1

-- Define the condition that the tangent line at Q(2, -1) has a slope parallel to y = x - 3, which means slope = 1
def tangent_slope_at_Q (a b : ℝ) :=
  4 * a + b = 1

-- Define the parabola passing through point Q(2, -1)
def parabola_passing_through_Q (a b c : ℝ) :=
  (2:ℝ)^2 * a + (2:ℝ) * b + c = -1

-- The proof statement
theorem determine_parabola (a b c : ℝ):
  parabola_passing_through a b c ∧ 
  tangent_slope_at_Q a b ∧ 
  parabola_passing_through_Q a b c → 
  a = 3 ∧ b = -11 ∧ c = 9 :=
by
  sorry

end determine_parabola_l258_25837


namespace workshop_worker_count_l258_25859

theorem workshop_worker_count (W T N : ℕ) (h1 : T = 7) (h2 : 8000 * W = 7 * 14000 + 6000 * N) (h3 : W = T + N) : W = 28 :=
by
  sorry

end workshop_worker_count_l258_25859


namespace magnitude_of_2a_plus_b_l258_25826

open Real

variables (a b : ℝ × ℝ) (angle : ℝ)

-- Conditions
axiom angle_between_a_b (a b : ℝ × ℝ) : angle = π / 3 -- 60 degrees in radians
axiom norm_a_eq_1 (a : ℝ × ℝ) : ‖a‖ = 1
axiom b_eq (b : ℝ × ℝ) : b = (3, 0)

-- Theorem
theorem magnitude_of_2a_plus_b (h1 : angle = π / 3) (h2 : ‖a‖ = 1) (h3 : b = (3, 0)) :
  ‖2 • a + b‖ = sqrt 19 :=
sorry

end magnitude_of_2a_plus_b_l258_25826


namespace inradius_of_right_triangle_l258_25831

variable (a b c : ℕ) -- Define the sides
def right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

noncomputable def area (a b : ℕ) : ℝ :=
  0.5 * (a : ℝ) * (b : ℝ)

noncomputable def semiperimeter (a b c : ℕ) : ℝ :=
  ((a + b + c) : ℝ) / 2

noncomputable def inradius (a b c : ℕ) : ℝ :=
  let s := semiperimeter a b c
  let A := area a b
  A / s

theorem inradius_of_right_triangle (h : right_triangle 7 24 25) : inradius 7 24 25 = 3 := by
  sorry

end inradius_of_right_triangle_l258_25831


namespace caterpillars_left_on_tree_l258_25875

-- Definitions based on conditions
def initialCaterpillars : ℕ := 14
def hatchedCaterpillars : ℕ := 4
def caterpillarsLeftToCocoon : ℕ := 8

-- The proof problem statement in Lean
theorem caterpillars_left_on_tree : initialCaterpillars + hatchedCaterpillars - caterpillarsLeftToCocoon = 10 :=
by
  -- solution steps will go here eventually
  sorry

end caterpillars_left_on_tree_l258_25875


namespace three_legged_tables_count_l258_25812

theorem three_legged_tables_count (x y : ℕ) (h1 : 3 * x + 4 * y = 23) (h2 : 2 ≤ x) (h3 : 2 ≤ y) : x = 5 := 
sorry

end three_legged_tables_count_l258_25812


namespace smallest_b_value_l258_25860

def triangle_inequality (x y z : ℝ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

def not_triangle (x y z : ℝ) : Prop :=
  ¬triangle_inequality x y z

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
    (h3 : not_triangle 2 a b) (h4 : not_triangle (1 / b) (1 / a) 1) :
    b >= 2 :=
by
  sorry

end smallest_b_value_l258_25860


namespace total_chickens_l258_25884

theorem total_chickens (ducks geese : ℕ) (hens roosters chickens: ℕ) :
  ducks = 45 → geese = 28 →
  hens = ducks - 13 → roosters = geese + 9 →
  chickens = hens + roosters →
  chickens = 69 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_chickens_l258_25884


namespace num_five_digit_palindromes_with_even_middle_l258_25873

theorem num_five_digit_palindromes_with_even_middle :
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ∃ c', c = 2 * c' ∧ 0 ≤ c' ∧ c' ≤ 4 ∧ 10000 * a + 1000 * b + 100 * c + 10 * b + a ≤ 99999) →
  9 * 10 * 5 = 450 :=
by
  sorry

end num_five_digit_palindromes_with_even_middle_l258_25873


namespace box_problem_l258_25856

theorem box_problem 
    (x y : ℕ) 
    (h1 : 10 * x + 20 * y = 18 * (x + y)) 
    (h2 : 10 * x + 20 * (y - 10) = 16 * (x + y - 10)) :
    x + y = 20 :=
sorry

end box_problem_l258_25856


namespace prob_three_heads_is_one_eighth_l258_25892

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l258_25892


namespace find_f_neg1_l258_25822

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f : ℝ → ℝ) (h_odd : odd_function f) (h_f1 : f 1 = 2)

-- Theorem stating the necessary proof
theorem find_f_neg1 : f (-1) = -2 :=
by
  sorry

end find_f_neg1_l258_25822


namespace linear_func_passing_point_l258_25829

theorem linear_func_passing_point :
  ∃ k : ℝ, ∀ x y : ℝ, (y = k * x + 1) → (x = -1 ∧ y = 0) → k = 1 :=
by
  sorry

end linear_func_passing_point_l258_25829


namespace marta_hours_worked_l258_25871

-- Definitions of the conditions in Lean 4
def total_collected : ℕ := 240
def hourly_rate : ℕ := 10
def tips_collected : ℕ := 50
def work_earned : ℕ := total_collected - tips_collected

-- Goal: To prove the number of hours worked by Marta
theorem marta_hours_worked : work_earned / hourly_rate = 19 := by
  sorry

end marta_hours_worked_l258_25871


namespace hockeyPlayers_count_l258_25878

def numPlayers := 50
def cricketPlayers := 12
def footballPlayers := 11
def softballPlayers := 10

theorem hockeyPlayers_count : 
  let hockeyPlayers := numPlayers - (cricketPlayers + footballPlayers + softballPlayers)
  hockeyPlayers = 17 :=
by
  sorry

end hockeyPlayers_count_l258_25878


namespace find_value_of_a_l258_25866

theorem find_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≤ 24) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 24) ∧
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 3) → 
  a = 2 ∨ a = -5 :=
by
  sorry

end find_value_of_a_l258_25866


namespace shaded_area_is_correct_l258_25861

def area_of_rectangle (l w : ℕ) : ℕ := l * w

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

def area_of_shaded_region : ℕ :=
  let length := 8
  let width := 4
  let area_rectangle := area_of_rectangle length width
  let area_triangle := area_of_triangle length width
  area_rectangle - area_triangle

theorem shaded_area_is_correct : area_of_shaded_region = 16 :=
by
  sorry

end shaded_area_is_correct_l258_25861


namespace power_equivalence_l258_25864

theorem power_equivalence (L : ℕ) : 32^4 * 4^5 = 2^L → L = 30 :=
by
  sorry

end power_equivalence_l258_25864


namespace domain_of_function_l258_25828

theorem domain_of_function :
  (∀ x : ℝ, 2 + x ≥ 0 ∧ 3 - x ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry

end domain_of_function_l258_25828


namespace find_length_QR_l258_25810

-- Define the provided conditions as Lean definitions
variables (Q P R : ℝ) (h_cos : Real.cos Q = 0.3) (QP : ℝ) (h_QP : QP = 15)
  
-- State the theorem we need to prove
theorem find_length_QR (QR : ℝ) (h_triangle : QP / QR = Real.cos Q) : QR = 50 := sorry

end find_length_QR_l258_25810


namespace probability_of_mathematics_letter_l258_25845

-- Definitions for the problem
def english_alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def mathematics_letters : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

-- Set the total number of letters in the English alphabet
def total_letters := english_alphabet.card

-- Set the number of unique letters in 'MATHEMATICS'
def mathematics_unique_letters := mathematics_letters.card

-- Statement of the Lean theorem
theorem probability_of_mathematics_letter : (mathematics_unique_letters : ℚ) / total_letters = 4 / 13 :=
by
  sorry

end probability_of_mathematics_letter_l258_25845


namespace length_of_AB_l258_25862

theorem length_of_AB 
  (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 + A.2 ^ 2 = 8)
  (hB : B.1 ^ 2 + B.2 ^ 2 = 8)
  (lA : A.1 - 2 * A.2 + 5 = 0)
  (lB : B.1 - 2 * B.2 + 5 = 0) :
  dist A B = 2 * Real.sqrt 3 := by
  sorry

end length_of_AB_l258_25862


namespace total_sides_tom_tim_l258_25832

def sides_per_die : Nat := 6

def tom_dice_count : Nat := 4
def tim_dice_count : Nat := 4

theorem total_sides_tom_tim : tom_dice_count * sides_per_die + tim_dice_count * sides_per_die = 48 := by
  sorry

end total_sides_tom_tim_l258_25832


namespace first_storm_duration_l258_25848

theorem first_storm_duration
  (x y : ℕ)
  (h1 : 30 * x + 15 * y = 975)
  (h2 : x + y = 45) :
  x = 20 :=
by sorry

end first_storm_duration_l258_25848


namespace find_remainder_l258_25899

theorem find_remainder (P Q R D D' Q' R' C : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  (P % (D * D')) = (D * R' + R + C) :=
sorry

end find_remainder_l258_25899


namespace grandmother_total_payment_l258_25887

theorem grandmother_total_payment
  (senior_discount : Real := 0.30)
  (children_discount : Real := 0.40)
  (num_seniors : Nat := 2)
  (num_children : Nat := 2)
  (num_regular : Nat := 2)
  (senior_ticket_price : Real := 7.50)
  (regular_ticket_price : Real := senior_ticket_price / (1 - senior_discount))
  (children_ticket_price : Real := regular_ticket_price * (1 - children_discount))
  : (num_seniors * senior_ticket_price + num_regular * regular_ticket_price + num_children * children_ticket_price) = 49.27 := 
by
  sorry

end grandmother_total_payment_l258_25887


namespace coin_toss_dice_roll_l258_25817

theorem coin_toss_dice_roll :
  let coin_toss := 2 -- two outcomes for same side coin toss
  let dice_roll := 2 -- two outcomes for multiple of 3 on dice roll
  coin_toss * dice_roll = 4 :=
by
  sorry

end coin_toss_dice_roll_l258_25817


namespace fraction_power_computation_l258_25885

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l258_25885


namespace lines_through_point_l258_25813

theorem lines_through_point (k : ℝ) : ∀ x y : ℝ, (y = k * (x - 1)) ↔ (x = 1 ∧ y = 0) ∨ (x ≠ 1 ∧ y / (x - 1) = k) :=
by
  sorry

end lines_through_point_l258_25813


namespace operation_addition_x_l258_25865

theorem operation_addition_x (x : ℕ) (h : 106 + 106 + x + x = 19872) : x = 9830 :=
sorry

end operation_addition_x_l258_25865


namespace find_d_l258_25852

theorem find_d (a d : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d * x + 12) :
  d = 7 :=
sorry

end find_d_l258_25852


namespace lincoln_high_students_club_overlap_l258_25853

theorem lincoln_high_students_club_overlap (total_students : ℕ)
  (drama_club_students science_club_students both_or_either_club_students : ℕ)
  (h1 : total_students = 500)
  (h2 : drama_club_students = 150)
  (h3 : science_club_students = 200)
  (h4 : both_or_either_club_students = 300) :
  drama_club_students + science_club_students - both_or_either_club_students = 50 :=
by
  sorry

end lincoln_high_students_club_overlap_l258_25853


namespace ice_cream_ordering_ways_l258_25842

def number_of_cone_choices : ℕ := 2
def number_of_flavor_choices : ℕ := 4

theorem ice_cream_ordering_ways : number_of_cone_choices * number_of_flavor_choices = 8 := by
  sorry

end ice_cream_ordering_ways_l258_25842


namespace range_of_a_l258_25800

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 9 * x + a^2 / x + 7
  else 9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8/7 :=
by
  intros h
  -- Detailed proof would go here
  sorry

end range_of_a_l258_25800


namespace calculate_brick_height_cm_l258_25880

noncomputable def wall_length_cm : ℕ := 1000  -- 10 m converted to cm
noncomputable def wall_width_cm : ℕ := 800   -- 8 m converted to cm
noncomputable def wall_height_cm : ℕ := 2450 -- 24.5 m converted to cm

noncomputable def wall_volume_cm3 : ℕ := wall_length_cm * wall_width_cm * wall_height_cm

noncomputable def brick_length_cm : ℕ := 20
noncomputable def brick_width_cm : ℕ := 10
noncomputable def number_of_bricks : ℕ := 12250

noncomputable def brick_area_cm2 : ℕ := brick_length_cm * brick_width_cm

theorem calculate_brick_height_cm (h : ℕ) : brick_area_cm2 * h * number_of_bricks = wall_volume_cm3 → 
  h = wall_volume_cm3 / (brick_area_cm2 * number_of_bricks) := by
  sorry

end calculate_brick_height_cm_l258_25880


namespace Laura_pays_more_l258_25889

theorem Laura_pays_more 
  (slices : ℕ) 
  (cost_plain : ℝ) 
  (cost_mushrooms : ℝ) 
  (laura_mushroom_slices : ℕ) 
  (laura_plain_slices : ℕ) 
  (jessica_plain_slices: ℕ) :
  slices = 12 →
  cost_plain = 12 →
  cost_mushrooms = 3 →
  laura_mushroom_slices = 4 →
  laura_plain_slices = 2 →
  jessica_plain_slices = 6 →
  15 / 12 * (laura_mushroom_slices + laura_plain_slices) - 
  (cost_plain / 12 * jessica_plain_slices) = 1.5 :=
by
  intro slices_eq
  intro cost_plain_eq
  intro cost_mushrooms_eq
  intro laura_mushroom_slices_eq
  intro laura_plain_slices_eq
  intro jessica_plain_slices_eq
  sorry

end Laura_pays_more_l258_25889


namespace pq_true_l258_25816

open Real

def p : Prop := ∃ x0 : ℝ, tan x0 = sqrt 3

def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_true : p ∧ q :=
by
  sorry

end pq_true_l258_25816


namespace sum_of_first_cards_l258_25834

variables (a b c d : ℕ)

theorem sum_of_first_cards (a b c d : ℕ) : 
  ∃ x, x = b * (c + 1) + d - a :=
by
  sorry

end sum_of_first_cards_l258_25834


namespace range_of_a_l258_25882

noncomputable def satisfies_inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → (x^2 + 1) * Real.exp x ≥ a * x^2

theorem range_of_a (a : ℝ) : satisfies_inequality a ↔ a ≤ 2 * Real.exp 1 :=
by
  sorry

end range_of_a_l258_25882


namespace f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l258_25807

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 / x - 1 else 2 / (-x) - 1

-- Assertion 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 := 
sorry

-- Assertion 2: f(x) is a decreasing function on (0, +∞)
theorem f_decreasing_on_positive : ∀ a b : ℝ, 0 < b → b < a → f (a) < f (b) := 
sorry

-- Assertion 3: Expression of the function when x < 0
theorem f_expression_on_negative (x : ℝ) (hx : x < 0) : f x = 2 / (-x) - 1 := 
sorry

end f_neg_one_f_decreasing_on_positive_f_expression_on_negative_l258_25807


namespace part1_part2_l258_25849

def A (x : ℝ) : Prop := x < -3 ∨ x > 7
def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def complement_R_A (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 7

theorem part1 (m : ℝ) :
  (∀ x, complement_R_A x ∨ B m x → complement_R_A x) →
  m ≤ 4 :=
by
  sorry

theorem part2 (m : ℝ) (a b : ℝ) :
  (∀ x, complement_R_A x ∧ B m x ↔ (a ≤ x ∧ x ≤ b)) ∧ (b - a ≥ 1) →
  3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end part1_part2_l258_25849


namespace root_magnitude_conditions_l258_25876

theorem root_magnitude_conditions (p : ℝ) (h : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = -p) ∧ (r1 * r2 = -12)) :
  (∃ r1 r2 : ℝ, (r1 ≠ r2) ∧ |r1| > 2 ∨ |r2| > 2) ∧ (∀ r1 r2 : ℝ, (r1 + r2 = -p) ∧ (r1 * r2 = -12) → |r1| * |r2| ≤ 14) :=
by
  -- Proof of the theorem goes here
  sorry

end root_magnitude_conditions_l258_25876


namespace max_d_n_l258_25823

open Int

def a_n (n : ℕ) : ℤ := 80 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (a_n n) (a_n (n + 1))

theorem max_d_n : ∃ n : ℕ, d_n n = 5 ∧ ∀ m : ℕ, d_n m ≤ 5 := by
  sorry

end max_d_n_l258_25823


namespace problem1_problem2_l258_25867

-- Definitions of vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (-2, 3)
def vec_c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Problem Part 1: Prove m = -1 given a ⊥ (b + c)
theorem problem1 (m : ℝ) (h : vec_a.1 * (vec_b + vec_c m).1 + vec_a.2 * (vec_b + vec_c m).2 = 0) : m = -1 :=
sorry

-- Problem Part 2: Prove k = -2 given k*a + b is collinear with 2*a - b
theorem problem2 (k : ℝ) (h : (k * vec_a.1 + vec_b.1) / (2 * vec_a.1 - vec_b.1) = (k * vec_a.2 + vec_b.2) / (2 * vec_a.2 - vec_b.2)) : k = -2 :=
sorry

end problem1_problem2_l258_25867


namespace joker_then_spade_probability_correct_l258_25869

-- Defining the conditions of the deck
def deck_size : ℕ := 60
def joker_count : ℕ := 4
def suit_count : ℕ := 4
def cards_per_suit : ℕ := 15

-- The probability of drawing a Joker first and then a spade
def prob_joker_then_spade : ℚ :=
  (joker_count * (cards_per_suit - 1) + (deck_size - joker_count) * cards_per_suit) /
  (deck_size * (deck_size - 1))

-- The expected probability according to the solution
def expected_prob : ℚ := 224 / 885

theorem joker_then_spade_probability_correct :
  prob_joker_then_spade = expected_prob :=
by
  -- Skipping the actual proof steps
  sorry

end joker_then_spade_probability_correct_l258_25869


namespace tangent_line_at_e_range_of_a_l258_25843

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x - 2 * a * x

theorem tangent_line_at_e (a : ℝ) :
  a = 0 →
  ∃ m b : ℝ, (∀ x, y = m * x + b) ∧ 
             y = (2 / Real.exp 1 - 2 * Real.exp 1) * x + (Real.exp 1)^2 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 → g a x < 0) →
  a ∈ Set.Icc (-1) 1 :=
sorry

end tangent_line_at_e_range_of_a_l258_25843


namespace baker_bought_131_new_cakes_l258_25821

def number_of_new_cakes_bought (initial_cakes: ℕ) (cakes_sold: ℕ) (excess_sold: ℕ): ℕ :=
    cakes_sold - excess_sold - initial_cakes

theorem baker_bought_131_new_cakes :
    number_of_new_cakes_bought 8 145 6 = 131 :=
by
  -- This is where the proof would normally go
  sorry

end baker_bought_131_new_cakes_l258_25821


namespace find_constants_u_v_l258_25833

theorem find_constants_u_v : 
  ∃ u v : ℝ, (∀ x : ℝ, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 :=
sorry

end find_constants_u_v_l258_25833


namespace triangle_angles_l258_25811

variable (a b c t : ℝ)

def angle_alpha : ℝ := 43

def area_condition (α β : ℝ) : Prop :=
  2 * t = a * b * Real.sqrt (Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin α * Real.sin β)

theorem triangle_angles (α β γ : ℝ) (hα : α = angle_alpha) (h_area : area_condition a b t α β) :
  α = 43 ∧ β = 17 ∧ γ = 120 := sorry

end triangle_angles_l258_25811


namespace geometric_sequence_s6_s4_l258_25874

section GeometricSequence

variables {a : ℕ → ℝ} {a1 : ℝ} {q : ℝ}
variable (h_geom : ∀ n, a (n + 1) = a n * q)
variable (h_q_ne_one : q ≠ 1)
variable (S : ℕ → ℝ)
variable (h_S : ∀ n, S n = a1 * (1 - q^(n + 1)) / (1 - q))
variable (h_ratio : S 4 / S 2 = 3)

theorem geometric_sequence_s6_s4 :
  S 6 / S 4 = 7 / 3 :=
sorry

end GeometricSequence

end geometric_sequence_s6_s4_l258_25874


namespace dot_product_value_l258_25808

variables (a b : ℝ × ℝ)

theorem dot_product_value
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a.1 * b.1 + a.2 * b.2 = -12 :=
sorry

end dot_product_value_l258_25808


namespace total_players_l258_25883

def kabaddi (K : ℕ) (Kho_only : ℕ) (Both : ℕ) : ℕ :=
  K - Both + Kho_only + Both

theorem total_players (K : ℕ) (Kho_only : ℕ) (Both : ℕ)
  (hK : K = 10)
  (hKho_only : Kho_only = 35)
  (hBoth : Both = 5) :
  kabaddi K Kho_only Both = 45 :=
by
  rw [hK, hKho_only, hBoth]
  unfold kabaddi
  norm_num

end total_players_l258_25883


namespace space_filled_with_rhombic_dodecahedra_l258_25854

/-
  Given: Space can be filled completely using cubic cells (cubic lattice).
  To Prove: Space can be filled completely using rhombic dodecahedron cells.
-/

theorem space_filled_with_rhombic_dodecahedra :
  (∀ (cubic_lattice : Type), (∃ fill_space_with_cubes : (cubic_lattice → Prop), 
    ∀ x : cubic_lattice, fill_space_with_cubes x)) →
  (∃ (rhombic_dodecahedra_lattice : Type), 
      (∀ fill_space_with_rhombic_dodecahedra : rhombic_dodecahedra_lattice → Prop, 
        ∀ y : rhombic_dodecahedra_lattice, fill_space_with_rhombic_dodecahedra y)) :=
by {
  sorry
}

end space_filled_with_rhombic_dodecahedra_l258_25854


namespace both_buyers_correct_l258_25894

-- Define the total number of buyers
def total_buyers : ℕ := 100

-- Define the number of buyers who purchase cake mix
def cake_mix_buyers : ℕ := 50

-- Define the number of buyers who purchase muffin mix
def muffin_mix_buyers : ℕ := 40

-- Define the number of buyers who purchase neither cake mix nor muffin mix
def neither_buyers : ℕ := 29

-- Define the number of buyers who purchase both cake and muffin mix
def both_buyers : ℕ := 19

-- The assertion to be proved
theorem both_buyers_correct :
  neither_buyers = total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_buyers) :=
sorry

end both_buyers_correct_l258_25894


namespace determinant_not_sufficient_nor_necessary_l258_25836

-- Definitions of the initial conditions
variables {a1 b1 a2 b2 c1 c2 : ℝ}

-- Conditions given: neither line coefficients form the zero vector
axiom non_zero_1 : a1^2 + b1^2 ≠ 0
axiom non_zero_2 : a2^2 + b2^2 ≠ 0

-- The matrix determinant condition and line parallelism
def determinant_condition (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 ≠ 0

def lines_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  a1 * b2 - a2 * b1 = 0 ∧ a1 * c2 ≠ a2 * c1

-- Proof problem statement: proving equivalence
theorem determinant_not_sufficient_nor_necessary :
  ¬ (∀ a1 b1 a2 b2 c1 c2, (determinant_condition a1 b1 a2 b2 → lines_parallel a1 b1 c1 a2 b2 c2) ∧
                          (lines_parallel a1 b1 c1 a2 b2 c2 → determinant_condition a1 b1 a2 b2)) :=
sorry

end determinant_not_sufficient_nor_necessary_l258_25836


namespace boat_speed_is_20_l258_25805

-- Definitions based on conditions from the problem
def boat_speed_still_water (x : ℝ) : Prop := 
  let current_speed := 5
  let downstream_distance := 8.75
  let downstream_time := 21 / 60
  let downstream_speed := x + current_speed
  downstream_speed * downstream_time = downstream_distance

-- The theorem to prove
theorem boat_speed_is_20 : boat_speed_still_water 20 :=
by 
  unfold boat_speed_still_water
  sorry

end boat_speed_is_20_l258_25805


namespace paulina_convertibles_l258_25895

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end paulina_convertibles_l258_25895


namespace largest_value_x_l258_25804

theorem largest_value_x (x a b c d : ℝ) (h_eq : 7 * x ^ 2 + 15 * x - 20 = 0) (h_form : x = (a + b * Real.sqrt c) / d) (ha : a = -15) (hb : b = 1) (hc : c = 785) (hd : d = 14) : (a * c * d) / b = -164850 := 
sorry

end largest_value_x_l258_25804


namespace difference_of_squares_l258_25870

theorem difference_of_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x y : ℤ, a = x^2 - y^2) ∨ 
  (∃ x y : ℤ, b = x^2 - y^2) ∨ 
  (∃ x y : ℤ, a + b = x^2 - y^2) :=
by
  sorry

end difference_of_squares_l258_25870


namespace andy_incorrect_l258_25891

theorem andy_incorrect (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 8) : a = 14 :=
by
  sorry

end andy_incorrect_l258_25891


namespace monthly_earnings_l258_25806

theorem monthly_earnings (savings_per_month : ℤ) (total_needed : ℤ) (total_earned : ℤ)
  (H1 : savings_per_month = 500)
  (H2 : total_needed = 45000)
  (H3 : total_earned = 360000) :
  total_earned / (total_needed / savings_per_month) = 4000 := by
  sorry

end monthly_earnings_l258_25806


namespace probability_two_faces_no_faces_l258_25868

theorem probability_two_faces_no_faces :
  let side_length := 5
  let total_cubes := side_length ^ 3
  let painted_faces := 2 * (side_length ^ 2)
  let two_painted_faces := 16
  let no_painted_faces := total_cubes - painted_faces + two_painted_faces
  (two_painted_faces = 16) →
  (no_painted_faces = 91) →
  -- Total ways to choose 2 cubes from 125
  let total_ways := (total_cubes * (total_cubes - 1)) / 2
  -- Ways to choose 1 cube with 2 painted faces and 1 with no painted faces
  let successful_ways := two_painted_faces * no_painted_faces
  (successful_ways = 1456) →
  (total_ways = 7750) →
  -- The desired probability
  let probability := successful_ways / (total_ways : ℝ)
  probability = 4 / 21 :=
by
  intros side_length total_cubes painted_faces two_painted_faces no_painted_faces h1 h2 total_ways successful_ways h3 h4 probability
  sorry

end probability_two_faces_no_faces_l258_25868


namespace number_of_possible_triples_l258_25838

-- Given conditions
variables (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)

-- Revenue equation
def revenue_equation : Prop := 10 * x + 5 * y + z = 120

-- Proving the solution
theorem number_of_possible_triples (h : revenue_equation x y z) : 
  ∃ (n : ℕ), n = 121 :=
by
  sorry

end number_of_possible_triples_l258_25838


namespace inequality_min_value_l258_25898

theorem inequality_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, (x + 2 * y) * (2 / x + 1 / y) ≥ m ∧ m ≤ 8 :=
by
  sorry

end inequality_min_value_l258_25898


namespace value_range_of_f_l258_25857

def f (x : ℝ) := 2 * x ^ 2 + 4 * x + 1

theorem value_range_of_f :
  ∀ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 4 → (∃ y ∈ Set.Icc (-1 : ℝ) 49, f x = y) :=
by sorry

end value_range_of_f_l258_25857


namespace asterisk_replacement_l258_25890

theorem asterisk_replacement (x : ℝ) : 
  (x / 20) * (x / 80) = 1 ↔ x = 40 :=
by sorry

end asterisk_replacement_l258_25890


namespace new_phone_plan_cost_l258_25850

def old_plan_cost : ℝ := 150
def increase_percentage : ℝ := 0.30
def new_plan_cost := old_plan_cost + (increase_percentage * old_plan_cost)

theorem new_phone_plan_cost : new_plan_cost = 195 := by
  -- From the condition that the old plan cost is $150 and the increase percentage is 30%
  -- We should prove that the new plan cost is $195
  sorry

end new_phone_plan_cost_l258_25850


namespace scientific_notation_of_neg_0_000008691_l258_25872

theorem scientific_notation_of_neg_0_000008691:
  -0.000008691 = -8.691 * 10^(-6) :=
sorry

end scientific_notation_of_neg_0_000008691_l258_25872
