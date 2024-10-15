import Mathlib

namespace NUMINAMATH_GPT_elsa_ends_with_145_marbles_l227_22755

theorem elsa_ends_with_145_marbles :
  let initial := 150
  let after_breakfast := initial - 7
  let after_lunch := after_breakfast - 57
  let after_afternoon := after_lunch + 25
  let after_evening := after_afternoon + 85
  let after_exchange := after_evening - 9 + 6
  let final := after_exchange - 48
  final = 145 := by
    sorry

end NUMINAMATH_GPT_elsa_ends_with_145_marbles_l227_22755


namespace NUMINAMATH_GPT_not_proportional_eqn_exists_l227_22721

theorem not_proportional_eqn_exists :
  ∀ (x y : ℝ), (4 * x + 2 * y = 8) → ¬ ((∃ k : ℝ, x = k * y) ∨ (∃ k : ℝ, x * y = k)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_not_proportional_eqn_exists_l227_22721


namespace NUMINAMATH_GPT_line_BC_eq_circumscribed_circle_eq_l227_22723

noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0
noncomputable def equation_line_BC (x y : ℝ) : Prop := 3 * x - y - 1 = 0
noncomputable def circumscribed_circle (x y : ℝ) : Prop := (x - 5 / 2)^2 + (y + 7 / 2)^2 = 50 / 4

theorem line_BC_eq :
  ∃ x y : ℝ, altitude_line x y →
             B = (x, y) →
             equation_line_BC x y :=
by sorry

theorem circumscribed_circle_eq :
  ∃ x y : ℝ, altitude_line x y →
             (x - 3)^2 + y^2 = (5 / 2)^2 →
             circumscribed_circle x y :=
by sorry

end NUMINAMATH_GPT_line_BC_eq_circumscribed_circle_eq_l227_22723


namespace NUMINAMATH_GPT_water_intake_proof_l227_22708

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end NUMINAMATH_GPT_water_intake_proof_l227_22708


namespace NUMINAMATH_GPT_vegetable_plot_area_l227_22745

variable (V W : ℝ)

theorem vegetable_plot_area (h1 : (1/2) * V + (1/3) * W = 13) (h2 : (1/2) * W + (1/3) * V = 12) : V = 18 :=
by
  sorry

end NUMINAMATH_GPT_vegetable_plot_area_l227_22745


namespace NUMINAMATH_GPT_triangles_in_divided_square_l227_22767

theorem triangles_in_divided_square (V : ℕ) (marked_points : ℕ) (triangles : ℕ) 
  (h1 : V = 24) -- Vertices - 20 marked points and 4 vertices 
  (h2 : marked_points = 20) -- Marked points
  (h3 : triangles = F - 1) -- Each face (F) except the outer one is a triangle
  (h4 : V - E + F = 2) -- Euler's formula for planar graphs
  (h5 : E = (3*F + 1) / 2) -- Relationship between edges and faces
  (F : ℕ) -- Number of faces including the external face
  (E : ℕ) -- Number of edges
  : triangles = 42 := 
by 
  sorry

end NUMINAMATH_GPT_triangles_in_divided_square_l227_22767


namespace NUMINAMATH_GPT_smallest_n_divisibility_l227_22787

theorem smallest_n_divisibility:
  ∃ (n : ℕ), n > 0 ∧ n^2 % 24 = 0 ∧ n^3 % 540 = 0 ∧ n = 60 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisibility_l227_22787


namespace NUMINAMATH_GPT_welders_started_on_other_project_l227_22702

theorem welders_started_on_other_project
  (r : ℝ) (x : ℝ) (W : ℝ)
  (h1 : 16 * r * 8 = W)
  (h2 : (16 - x) * r * 24 = W - 16 * r) :
  x = 11 :=
by
  sorry

end NUMINAMATH_GPT_welders_started_on_other_project_l227_22702


namespace NUMINAMATH_GPT_car_trader_profit_l227_22751

theorem car_trader_profit (P : ℝ) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.28000000000000004 * P
  let profit := selling_price - purchase_price
  let percentage_increase := (profit / purchase_price) * 100
  percentage_increase = 60 := 
by
  sorry

end NUMINAMATH_GPT_car_trader_profit_l227_22751


namespace NUMINAMATH_GPT_atomic_weight_of_iodine_is_correct_l227_22707

noncomputable def atomic_weight_iodine (atomic_weight_nitrogen : ℝ) (atomic_weight_hydrogen : ℝ) (molecular_weight_compound : ℝ) : ℝ :=
  molecular_weight_compound - (atomic_weight_nitrogen + 4 * atomic_weight_hydrogen)

theorem atomic_weight_of_iodine_is_correct :
  atomic_weight_iodine 14.01 1.008 145 = 126.958 :=
by
  unfold atomic_weight_iodine
  norm_num

end NUMINAMATH_GPT_atomic_weight_of_iodine_is_correct_l227_22707


namespace NUMINAMATH_GPT_sin_cos_identity_l227_22764

theorem sin_cos_identity (α β γ : ℝ) (h : α + β + γ = 180) :
    Real.sin α + Real.sin β + Real.sin γ = 
    4 * Real.cos (α / 2) * Real.cos (β / 2) * Real.cos (γ / 2) := 
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l227_22764


namespace NUMINAMATH_GPT_determine_prices_l227_22765

variable (num_items : ℕ) (cost_keychains cost_plush : ℕ) (x : ℚ) (unit_price_keychains unit_price_plush : ℚ)

noncomputable def price_equation (x : ℚ) : Prop :=
  (cost_keychains / x) + (cost_plush / (1.5 * x)) = num_items

theorem determine_prices 
  (h1 : num_items = 15)
  (h2 : cost_keychains = 240)
  (h3 : cost_plush = 180)
  (h4 : price_equation num_items cost_keychains cost_plush x)
  (hx : x = 24) :
  unit_price_keychains = 24 ∧ unit_price_plush = 36 :=
  by
    sorry

end NUMINAMATH_GPT_determine_prices_l227_22765


namespace NUMINAMATH_GPT_find_n_l227_22746

open Nat

-- Defining the production rates for conditions.
structure Production := 
  (workers : ℕ)
  (gadgets : ℕ)
  (gizmos : ℕ)
  (hours : ℕ)

def condition1 : Production := { workers := 150, gadgets := 450, gizmos := 300, hours := 1 }
def condition2 : Production := { workers := 100, gadgets := 400, gizmos := 500, hours := 2 }
def condition3 : Production := { workers := 75, gadgets := 900, gizmos := 900, hours := 4 }

-- Statement: Finding the value of n.
theorem find_n :
  (75 * ((condition2.gadgets / condition2.workers) * (condition3.hours / condition2.hours))) = 600 := by
  sorry

end NUMINAMATH_GPT_find_n_l227_22746


namespace NUMINAMATH_GPT_number_of_rods_in_one_mile_l227_22768

theorem number_of_rods_in_one_mile :
  (1 : ℤ) * 6 * 60 = 360 :=
by
  sorry

end NUMINAMATH_GPT_number_of_rods_in_one_mile_l227_22768


namespace NUMINAMATH_GPT_ratio_correct_l227_22725

theorem ratio_correct : 
    (2^17 * 3^19) / (6^18) = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_correct_l227_22725


namespace NUMINAMATH_GPT_zoe_bought_bottles_l227_22752

theorem zoe_bought_bottles
  (initial_bottles : ℕ)
  (drank_bottles : ℕ)
  (current_bottles : ℕ)
  (initial_bottles_eq : initial_bottles = 42)
  (drank_bottles_eq : drank_bottles = 25)
  (current_bottles_eq : current_bottles = 47) :
  ∃ bought_bottles : ℕ, bought_bottles = 30 :=
by
  sorry

end NUMINAMATH_GPT_zoe_bought_bottles_l227_22752


namespace NUMINAMATH_GPT_sam_final_amount_l227_22704

def initial_dimes : ℕ := 9
def initial_quarters : ℕ := 5
def initial_nickels : ℕ := 3

def dad_dimes : ℕ := 7
def dad_quarters : ℕ := 2

def mom_nickels : ℕ := 1
def mom_dimes : ℕ := 2

def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5

def initial_amount : ℕ := (initial_dimes * dime_value) + (initial_quarters * quarter_value) + (initial_nickels * nickel_value)
def dad_amount : ℕ := (dad_dimes * dime_value) + (dad_quarters * quarter_value)
def mom_amount : ℕ := (mom_nickels * nickel_value) + (mom_dimes * dime_value)

def final_amount : ℕ := initial_amount + dad_amount - mom_amount

theorem sam_final_amount : final_amount = 325 := by
  sorry

end NUMINAMATH_GPT_sam_final_amount_l227_22704


namespace NUMINAMATH_GPT_average_speed_of_train_l227_22730

theorem average_speed_of_train (x : ℝ) (h₀ : x > 0) :
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  avg_speed = 48 := by
  let time_1 := x / 40
  let time_2 := 2 * x / 20
  let total_time := time_1 + time_2
  let total_distance := 6 * x
  let avg_speed := total_distance / total_time
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l227_22730


namespace NUMINAMATH_GPT_quadratic_h_value_l227_22715

theorem quadratic_h_value (p q r h : ℝ) (hq : p*x^2 + q*x + r = 5*(x - 3)^2 + 15):
  let new_quadratic := 4* (p*x^2 + q*x + r)
  let m := 20
  let k := 60
  new_quadratic = m * (x - h) ^ 2 + k → h = 3 := by
  sorry

end NUMINAMATH_GPT_quadratic_h_value_l227_22715


namespace NUMINAMATH_GPT_pave_square_with_tiles_l227_22772

theorem pave_square_with_tiles (b c : ℕ) (h_right_triangle : (b > 0) ∧ (c > 0)) :
  (∃ (k : ℕ), k^2 = b^2 + c^2) ↔ (∃ (m n : ℕ), m * c * b = 2 * n^2 * (b^2 + c^2)) := 
sorry

end NUMINAMATH_GPT_pave_square_with_tiles_l227_22772


namespace NUMINAMATH_GPT_oranges_cost_l227_22720

def cost_for_multiple_dozens (price_per_dozen: ℝ) (dozens: ℝ) : ℝ := 
    price_per_dozen * dozens

theorem oranges_cost (price_for_4_dozens: ℝ) (price_for_5_dozens: ℝ) :
  price_for_4_dozens = 28.80 →
  price_for_5_dozens = cost_for_multiple_dozens (28.80 / 4) 5 →
  price_for_5_dozens = 36 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_oranges_cost_l227_22720


namespace NUMINAMATH_GPT_probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l227_22724

-- Definitions based on the conditions laid out in the problem
def fly_paths (n_right n_up : ℕ) : ℕ :=
  (Nat.factorial (n_right + n_up)) / ((Nat.factorial n_right) * (Nat.factorial n_up))

-- Probability for part a
theorem probability_at_8_10 : 
  (fly_paths 8 10) / (2 ^ 18) = (Nat.choose 18 8 : ℚ) / 2 ^ 18 := 
sorry

-- Probability for part b
theorem probability_at_8_10_through_5_6 :
  ((fly_paths 5 6) * (fly_paths 1 0) * (fly_paths 2 4)) / (2 ^ 18) = (6930 : ℚ) / 2 ^ 18 :=
sorry

-- Probability for part c
theorem probability_at_8_10_within_circle :
  (2 * fly_paths 2 7 * fly_paths 6 3 + 2 * fly_paths 3 6 * fly_paths 5 3 + (fly_paths 4 6) ^ 2) / (2 ^ 18) = 
  (2 * Nat.choose 9 2 * Nat.choose 9 6 + 2 * Nat.choose 9 3 * Nat.choose 9 5 + (Nat.choose 9 4) ^ 2 : ℚ) / 2 ^ 18 :=
sorry

end NUMINAMATH_GPT_probability_at_8_10_probability_at_8_10_through_5_6_probability_at_8_10_within_circle_l227_22724


namespace NUMINAMATH_GPT_range_of_m_l227_22717

-- Define the points and hyperbola condition
section ProofProblem

variables (m y₁ y₂ : ℝ)

-- Given conditions
def point_A_hyperbola : Prop := y₁ = -3 - m
def point_B_hyperbola : Prop := y₂ = (3 + m) / 2
def y1_greater_than_y2 : Prop := y₁ > y₂

-- The theorem to prove
theorem range_of_m (h1 : point_A_hyperbola m y₁) (h2 : point_B_hyperbola m y₂) (h3 : y1_greater_than_y2 y₁ y₂) : m < -3 :=
by { sorry }

end ProofProblem

end NUMINAMATH_GPT_range_of_m_l227_22717


namespace NUMINAMATH_GPT_equation_of_circle_passing_through_points_l227_22701

theorem equation_of_circle_passing_through_points :
  (∃ D E F : ℝ, 
    (∀ x y : ℝ, x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (∀ x y : ℝ, x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0) ∧
    (x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 4 * x - 6 * y = 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_circle_passing_through_points_l227_22701


namespace NUMINAMATH_GPT_present_cost_after_two_years_l227_22771

-- Defining variables and constants
def initial_cost : ℝ := 75
def inflation_rate : ℝ := 0.05
def first_year_increase1 : ℝ := 0.20
def first_year_decrease1 : ℝ := 0.20
def second_year_increase2 : ℝ := 0.30
def second_year_decrease2 : ℝ := 0.25

theorem present_cost_after_two_years : presents_cost = 77.40 :=
by
  let adjusted_initial_cost := initial_cost + (initial_cost * inflation_rate)
  let increased_cost_year1 := adjusted_initial_cost + (adjusted_initial_cost * first_year_increase1)
  let decreased_cost_year1 := increased_cost_year1 - (increased_cost_year1 * first_year_decrease1)
  let adjusted_cost_year1 := decreased_cost_year1 + (decreased_cost_year1 * inflation_rate)
  let increased_cost_year2 := adjusted_cost_year1 + (adjusted_cost_year1 * second_year_increase2)
  let decreased_cost_year2 := increased_cost_year2 - (increased_cost_year2 * second_year_decrease2)
  let presents_cost := decreased_cost_year2
  have h := (presents_cost : ℝ)
  have h := presents_cost
  sorry

end NUMINAMATH_GPT_present_cost_after_two_years_l227_22771


namespace NUMINAMATH_GPT_problem_statement_l227_22726

theorem problem_statement (a n : ℕ) (h_a : a ≥ 1) (h_n : n ≥ 1) :
  (∃ k : ℕ, (a + 1)^n - a^n = k * n) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l227_22726


namespace NUMINAMATH_GPT_Jakes_brother_has_more_l227_22776

-- Define the number of comic books Jake has
def Jake_comics : ℕ := 36

-- Define the total number of comic books Jake and his brother have together
def total_comics : ℕ := 87

-- Prove Jake's brother has 15 more comic books than Jake
theorem Jakes_brother_has_more : ∃ B, B > Jake_comics ∧ B + Jake_comics = total_comics ∧ B - Jake_comics = 15 :=
by
  sorry

end NUMINAMATH_GPT_Jakes_brother_has_more_l227_22776


namespace NUMINAMATH_GPT_hannah_total_cost_l227_22777

def price_per_kg : ℝ := 5
def discount_rate : ℝ := 0.4
def kilograms : ℝ := 10

theorem hannah_total_cost :
  (price_per_kg * (1 - discount_rate)) * kilograms = 30 := 
by
  sorry

end NUMINAMATH_GPT_hannah_total_cost_l227_22777


namespace NUMINAMATH_GPT_solve_trig_eq_l227_22727

theorem solve_trig_eq (k : ℤ) : 
  ∃ x, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * k * Real.pi :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l227_22727


namespace NUMINAMATH_GPT_max_distance_on_highway_l227_22739

-- Assume there are definitions for the context of this problem
def mpg_highway : ℝ := 12.2
def gallons : ℝ := 24
def max_distance (mpg : ℝ) (gal : ℝ) : ℝ := mpg * gal

theorem max_distance_on_highway :
  max_distance mpg_highway gallons = 292.8 :=
sorry

end NUMINAMATH_GPT_max_distance_on_highway_l227_22739


namespace NUMINAMATH_GPT_slices_per_person_l227_22722

theorem slices_per_person
  (number_of_coworkers : ℕ)
  (number_of_pizzas : ℕ)
  (number_of_slices_per_pizza : ℕ)
  (total_slices : ℕ)
  (slices_per_person : ℕ) :
  number_of_coworkers = 12 →
  number_of_pizzas = 3 →
  number_of_slices_per_pizza = 8 →
  total_slices = number_of_pizzas * number_of_slices_per_pizza →
  slices_per_person = total_slices / number_of_coworkers →
  slices_per_person = 2 :=
by intros; sorry

end NUMINAMATH_GPT_slices_per_person_l227_22722


namespace NUMINAMATH_GPT_calculate_f_2015_l227_22712

noncomputable def f : ℝ → ℝ := sorry

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the periodic function property with period 4
def periodic_4 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 4) = f x

-- Define the given condition for the interval (0, 2)
def interval_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x ^ 2

theorem calculate_f_2015
  (odd_f : odd_function f)
  (periodic_f : periodic_4 f)
  (interval_f : interval_condition f) :
  f 2015 = -2 :=
sorry

end NUMINAMATH_GPT_calculate_f_2015_l227_22712


namespace NUMINAMATH_GPT_correct_calculation_l227_22737

variable (n : ℕ)
variable (h1 : 63 + n = 70)

theorem correct_calculation : 36 * n = 252 :=
by
  -- Here we will need the Lean proof, which we skip using sorry
  sorry

end NUMINAMATH_GPT_correct_calculation_l227_22737


namespace NUMINAMATH_GPT_average_score_l227_22762

theorem average_score 
  (total_students : ℕ)
  (assigned_day_students_pct : ℝ)
  (makeup_day_students_pct : ℝ)
  (assigned_day_avg_score : ℝ)
  (makeup_day_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_students_pct = 0.70)
  (h3 : makeup_day_students_pct = 0.30)
  (h4 : assigned_day_avg_score = 0.60)
  (h5 : makeup_day_avg_score = 0.90) :
  (0.70 * 100 * 0.60 + 0.30 * 100 * 0.90) / 100 = 0.69 := 
sorry


end NUMINAMATH_GPT_average_score_l227_22762


namespace NUMINAMATH_GPT_total_savings_l227_22756

theorem total_savings :
  let josiah_daily := 0.25 
  let josiah_days := 24 
  let leah_daily := 0.50 
  let leah_days := 20 
  let megan_multiplier := 2
  let megan_days := 12 
  let josiah_savings := josiah_daily * josiah_days 
  let leah_savings := leah_daily * leah_days 
  let megan_daily := megan_multiplier * leah_daily 
  let megan_savings := megan_daily * megan_days 
  let total_savings := josiah_savings + leah_savings + megan_savings 
  total_savings = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_l227_22756


namespace NUMINAMATH_GPT_complement_of_set_A_is_34_l227_22793

open Set

noncomputable def U : Set ℕ := {n : ℕ | True}

noncomputable def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Complement of A in U
noncomputable def C_U_A : Set ℕ := U \ A

theorem complement_of_set_A_is_34 : C_U_A = {3, 4} :=
by sorry

end NUMINAMATH_GPT_complement_of_set_A_is_34_l227_22793


namespace NUMINAMATH_GPT_probability_A_given_B_probability_A_or_B_l227_22735

-- Definitions of the given conditions
def PA : ℝ := 0.2
def PB : ℝ := 0.18
def PAB : ℝ := 0.12

-- Theorem to prove the probability that city A also experiences rain when city B is rainy
theorem probability_A_given_B : PA * PB = PAB -> PA = 2 / 3 := by
  sorry

-- Theorem to prove the probability that at least one of the two cities experiences rain
theorem probability_A_or_B (PA PB PAB : ℝ) : (PA + PB - PAB) = 0.26 := by
  sorry

end NUMINAMATH_GPT_probability_A_given_B_probability_A_or_B_l227_22735


namespace NUMINAMATH_GPT_initial_soccer_balls_l227_22774

theorem initial_soccer_balls (x : ℝ) (h1 : 0.40 * x = y) (h2 : 0.20 * (0.60 * x) = z) (h3 : 0.80 * (0.60 * x) = 48) : x = 100 := by
  sorry

end NUMINAMATH_GPT_initial_soccer_balls_l227_22774


namespace NUMINAMATH_GPT_find_bases_l227_22711

theorem find_bases {F1 F2 : ℝ} (R1 R2 : ℕ) 
                   (hR1 : R1 = 9)
                   (hR2 : R2 = 6)
                   (hF1_R1 : F1 = 0.484848 * 9^2 / (9^2 - 1))
                   (hF2_R1 : F2 = 0.848484 * 9^2 / (9^2 - 1))
                   (hF1_R2 : F1 = 0.353535 * 6^2 / (6^2 - 1))
                   (hF2_R2 : F2 = 0.535353 * 6^2 / (6^2 - 1))
                   : R1 + R2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_bases_l227_22711


namespace NUMINAMATH_GPT_minimum_harmonic_sum_l227_22773

theorem minimum_harmonic_sum
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_harmonic_sum_l227_22773


namespace NUMINAMATH_GPT_border_area_is_72_l227_22769

def livingRoomLength : ℝ := 12
def livingRoomWidth : ℝ := 10
def borderWidth : ℝ := 2

def livingRoomArea : ℝ := livingRoomLength * livingRoomWidth
def carpetLength : ℝ := livingRoomLength - 2 * borderWidth
def carpetWidth : ℝ := livingRoomWidth - 2 * borderWidth
def carpetArea : ℝ := carpetLength * carpetWidth
def borderArea : ℝ := livingRoomArea - carpetArea

theorem border_area_is_72 : borderArea = 72 := 
by
  sorry

end NUMINAMATH_GPT_border_area_is_72_l227_22769


namespace NUMINAMATH_GPT_final_number_is_50_l227_22757

theorem final_number_is_50 (initial_ones initial_fours : ℕ) (h1 : initial_ones = 900) (h2 : initial_fours = 100) :
  ∃ (z : ℝ), (900 * (1:ℝ)^2 + 100 * (4:ℝ)^2) = z^2 ∧ z = 50 :=
by
  sorry

end NUMINAMATH_GPT_final_number_is_50_l227_22757


namespace NUMINAMATH_GPT_solution_set_inequality_range_of_m_l227_22788

def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Problem 1
theorem solution_set_inequality (x : ℝ) : 
  (f x 5 > 2) ↔ (-3 / 2 < x ∧ x < 3 / 2) :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = (x^2 + 2 * x + 3) ∧ y = f x m) ↔ (m ≥ 4) :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_range_of_m_l227_22788


namespace NUMINAMATH_GPT_find_antecedent_l227_22797

-- Condition: The ratio is 4:6, simplified to 2:3
def ratio (a b : ℕ) : Prop := (a / gcd a b) = 2 ∧ (b / gcd a b) = 3

-- Condition: The consequent is 30
def consequent (y : ℕ) : Prop := y = 30

-- The problem is to find the antecedent
def antecedent (x : ℕ) (y : ℕ) : Prop := ratio x y

-- The theorem to be proved
theorem find_antecedent:
  ∃ x : ℕ, consequent 30 → antecedent x 30 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_antecedent_l227_22797


namespace NUMINAMATH_GPT_fox_initial_coins_l227_22763

theorem fox_initial_coins :
  ∃ x : ℤ, x - 10 = 0 ∧ 2 * (x - 10) - 50 = 0 ∧ 2 * (2 * (x - 10) - 50) - 50 = 0 ∧
  2 * (2 * (2 * (x - 10) - 50) - 50) - 50 = 0 ∧ 2 * (2 * (2 * (2 * (x - 10) - 50) - 50) - 50) - 50 = 0 ∧
  x = 56 := 
by
  -- we skip the proof here
  sorry

end NUMINAMATH_GPT_fox_initial_coins_l227_22763


namespace NUMINAMATH_GPT_length_AE_l227_22714

-- The given conditions:
def isosceles_triangle (A B C : Type*) (AB BC : ℝ) (h : AB = BC) : Prop := true

def angles_and_lengths (A D C E : Type*) (angle_ADC angle_AEC AD CE DC : ℝ) 
  (h_angles : angle_ADC = 60 ∧ angle_AEC = 60)
  (h_lengths : AD = 13 ∧ CE = 13 ∧ DC = 9) : Prop := true

variables {A B C D E : Type*} (AB BC AD CE DC : ℝ)
  (h_isosceles_triangle : isosceles_triangle A B C AB BC (by sorry))
  (h_angles_and_lengths : angles_and_lengths A D C E 60 60 AD CE DC 
    (by split; norm_num) (by repeat {split}; norm_num))

-- The proof problem:
theorem length_AE : ∃ AE : ℝ, AE = 4 :=
  by sorry

end NUMINAMATH_GPT_length_AE_l227_22714


namespace NUMINAMATH_GPT_toms_dog_age_in_six_years_l227_22779

-- Define the conditions as hypotheses
variables (B T D : ℕ)

-- Conditions
axiom h1 : B = 4 * D
axiom h2 : T = B - 3
axiom h3 : B + 6 = 30

-- The proof goal: Tom's dog's age in six years
theorem toms_dog_age_in_six_years : D + 6 = 12 :=
  sorry -- Proof is omitted based on the instructions

end NUMINAMATH_GPT_toms_dog_age_in_six_years_l227_22779


namespace NUMINAMATH_GPT_ordering_of_powers_l227_22749

theorem ordering_of_powers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by sorry

end NUMINAMATH_GPT_ordering_of_powers_l227_22749


namespace NUMINAMATH_GPT_max_visible_sum_is_128_l227_22719

-- Define the structure of the problem
structure Cube :=
  (faces : Fin 6 → Nat)
  (bottom_face : Nat)
  (all_faces : ∀ i : Fin 6, i ≠ ⟨0, by decide⟩ → faces i = bottom_face → False)

-- Define the problem conditions
noncomputable def problem_conditions : Prop :=
  let cubes := [Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry,
                Cube.mk (fun i => [1, 3, 5, 7, 9, 11].get i) 1 sorry]
  -- Cube stacking in two layers, with two cubes per layer
  
  true

-- Define the theorem to be proved
theorem max_visible_sum_is_128 (h : problem_conditions) : 
  ∃ (total_sum : Nat), total_sum = 128 := 
sorry

end NUMINAMATH_GPT_max_visible_sum_is_128_l227_22719


namespace NUMINAMATH_GPT_basketball_not_table_tennis_l227_22750

theorem basketball_not_table_tennis (total_students likes_basketball likes_table_tennis dislikes_all : ℕ) (likes_basketball_not_tt : ℕ) :
  total_students = 30 →
  likes_basketball = 15 →
  likes_table_tennis = 10 →
  dislikes_all = 8 →
  (likes_basketball - 3 = likes_basketball_not_tt) →
  likes_basketball_not_tt = 12 := by
  intros h_total h_basketball h_table_tennis h_dislikes h_eq
  sorry

end NUMINAMATH_GPT_basketball_not_table_tennis_l227_22750


namespace NUMINAMATH_GPT_violet_children_count_l227_22733

theorem violet_children_count 
  (family_pass_cost : ℕ := 120)
  (adult_ticket_cost : ℕ := 35)
  (child_ticket_cost : ℕ := 20)
  (separate_ticket_total_cost : ℕ := 155)
  (adult_count : ℕ := 1) : 
  ∃ c : ℕ, 35 + 20 * c = 155 ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_violet_children_count_l227_22733


namespace NUMINAMATH_GPT_find_k_l227_22790

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l227_22790


namespace NUMINAMATH_GPT_center_of_circle_l227_22747

theorem center_of_circle (x y : ℝ) :
  x^2 + y^2 - 2 * x - 6 * y + 1 = 0 →
  (1, 3) = (1, 3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_center_of_circle_l227_22747


namespace NUMINAMATH_GPT_jimmy_needs_4_packs_of_bread_l227_22705

theorem jimmy_needs_4_packs_of_bread
  (num_sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (slices_per_pack : ℕ)
  (initial_slices : ℕ)
  (h1 : num_sandwiches = 8)
  (h2 : slices_per_sandwich = 2)
  (h3 : slices_per_pack = 4)
  (h4 : initial_slices = 0) :
  (num_sandwiches * slices_per_sandwich) / slices_per_pack = 4 := by
  sorry

end NUMINAMATH_GPT_jimmy_needs_4_packs_of_bread_l227_22705


namespace NUMINAMATH_GPT_w_share_l227_22786

theorem w_share (k : ℝ) (w x y z : ℝ) (h1 : w = k) (h2 : x = 6 * k) (h3 : y = 2 * k) (h4 : z = 4 * k) (h5 : x - y = 1500):
  w = 375 := by
  /- Lean code to show w = 375 -/
  sorry

end NUMINAMATH_GPT_w_share_l227_22786


namespace NUMINAMATH_GPT_find_missing_id_l227_22718

theorem find_missing_id
  (total_students : ℕ)
  (sample_size : ℕ)
  (known_ids : Finset ℕ)
  (k : ℕ)
  (missing_id : ℕ) : 
  total_students = 52 ∧ 
  sample_size = 4 ∧ 
  known_ids = {3, 29, 42} ∧ 
  k = total_students / sample_size ∧ 
  missing_id = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_id_l227_22718


namespace NUMINAMATH_GPT_cos_double_angle_l227_22703

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l227_22703


namespace NUMINAMATH_GPT_fraction_notation_correct_reading_decimal_correct_l227_22748

-- Define the given conditions
def fraction_notation (num denom : ℕ) : Prop :=
  num / denom = num / denom  -- Essentially stating that in fraction notation, it holds

def reading_decimal (n : ℚ) (s : String) : Prop :=
  if n = 90.58 then s = "ninety point five eight" else false -- Defining the reading rule for this specific case

-- State the theorem using the defined conditions
theorem fraction_notation_correct : fraction_notation 8 9 := 
by 
  sorry

theorem reading_decimal_correct : reading_decimal 90.58 "ninety point five eight" :=
by 
  sorry

end NUMINAMATH_GPT_fraction_notation_correct_reading_decimal_correct_l227_22748


namespace NUMINAMATH_GPT_soft_drink_cost_l227_22754

/-- Benny bought 2 soft drinks for a certain price each and 5 candy bars.
    He spent a total of $28. Each candy bar cost $4. 
    Prove that the cost of each soft drink was $4.
--/
theorem soft_drink_cost (S : ℝ) (h1 : 2 * S + 5 * 4 = 28) : S = 4 := 
by
  sorry

end NUMINAMATH_GPT_soft_drink_cost_l227_22754


namespace NUMINAMATH_GPT_quadratic_has_two_roots_l227_22770

theorem quadratic_has_two_roots 
  (a b c : ℝ) (h : b > a + c ∧ a + c > 0) : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂ :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_roots_l227_22770


namespace NUMINAMATH_GPT_profit_margin_comparison_l227_22716

theorem profit_margin_comparison
    (cost_price_A : ℝ) (selling_price_A : ℝ)
    (cost_price_B : ℝ) (selling_price_B : ℝ)
    (h1 : cost_price_A = 1600)
    (h2 : selling_price_A = 0.9 * 2000)
    (h3 : cost_price_B = 320)
    (h4 : selling_price_B = 0.8 * 460) :
    ((selling_price_B - cost_price_B) / cost_price_B) > ((selling_price_A - cost_price_A) / cost_price_A) := 
by
    sorry

end NUMINAMATH_GPT_profit_margin_comparison_l227_22716


namespace NUMINAMATH_GPT_factorize_n_squared_minus_nine_l227_22753

theorem factorize_n_squared_minus_nine (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := 
sorry

end NUMINAMATH_GPT_factorize_n_squared_minus_nine_l227_22753


namespace NUMINAMATH_GPT_max_and_min_W_l227_22783

noncomputable def W (x y z : ℝ) : ℝ := 2 * x + 6 * y + 4 * z

theorem max_and_min_W {x y z : ℝ} (h1 : x + y + z = 1) (h2 : 3 * y + z ≥ 2) (h3 : 0 ≤ x ∧ x ≤ 1) (h4 : 0 ≤ y ∧ y ≤ 2) :
  ∃ (W_max W_min : ℝ), W_max = 6 ∧ W_min = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_and_min_W_l227_22783


namespace NUMINAMATH_GPT_mrs_heine_dogs_l227_22700

-- Define the number of biscuits per dog
def biscuits_per_dog : ℕ := 3

-- Define the total number of biscuits
def total_biscuits : ℕ := 6

-- Define the number of dogs
def number_of_dogs : ℕ := 2

-- Define the proof statement
theorem mrs_heine_dogs : total_biscuits / biscuits_per_dog = number_of_dogs :=
by
  sorry

end NUMINAMATH_GPT_mrs_heine_dogs_l227_22700


namespace NUMINAMATH_GPT_cuberoot_sum_l227_22736

-- Prove that the sum c + d = 60 for the simplified form of the given expression.
theorem cuberoot_sum :
  let c := 15
  let d := 45
  c + d = 60 :=
by
  sorry

end NUMINAMATH_GPT_cuberoot_sum_l227_22736


namespace NUMINAMATH_GPT_pairs_symmetry_l227_22784

theorem pairs_symmetry (N : ℕ) (hN : N > 2) :
  ∃ f : {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 > 2} ≃ 
           {ab : ℕ × ℕ // ab.1 < ab.2 ∧ ab.2 ≤ N ∧ ab.2 / ab.1 < 2}, 
  true :=
sorry

end NUMINAMATH_GPT_pairs_symmetry_l227_22784


namespace NUMINAMATH_GPT_range_of_m_l227_22761

-- Define the quadratic function f
def f (a c x : ℝ) := a * x^2 - 2 * a * x + c

-- State the theorem
theorem range_of_m (a c : ℝ) (h : f a c 2017 < f a c (-2016)) (m : ℝ) 
  : f a c m ≤ f a c 0 → 0 ≤ m ∧ m ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_m_l227_22761


namespace NUMINAMATH_GPT_smallest_x_satisfying_expression_l227_22799

theorem smallest_x_satisfying_expression :
  ∃ x : ℤ, (∃ k : ℤ, x^2 + x + 7 = k * (x - 2)) ∧ (∀ y : ℤ, (∃ k' : ℤ, y^2 + y + 7 = k' * (y - 2)) → y ≥ x) ∧ x = -11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_satisfying_expression_l227_22799


namespace NUMINAMATH_GPT_smallest_n_with_divisors_l227_22728

-- Definitions of the divisors
def d_total (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)
def d_even (a b c : ℕ) : ℕ := a * (b + 1) * (c + 1)
def d_odd (b c : ℕ) : ℕ := (b + 1) * (c + 1)

-- Math problem and proving smallest n
theorem smallest_n_with_divisors (a b c : ℕ) (n : ℕ) (h_1 : d_odd b c = 8) (h_2 : d_even a b c = 16) : n = 60 :=
  sorry

end NUMINAMATH_GPT_smallest_n_with_divisors_l227_22728


namespace NUMINAMATH_GPT_triangle_side_count_l227_22732

theorem triangle_side_count :
  {b c : ℕ} → b ≤ 5 → 5 ≤ c → c - b < 5 → ∃ t : ℕ, t = 15 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_count_l227_22732


namespace NUMINAMATH_GPT_A_alone_days_l227_22742

noncomputable def days_for_A (r_A r_B r_C : ℝ) : ℝ :=
  1 / r_A

theorem A_alone_days
  (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3)
  (h2 : r_B + r_C = 1 / 6)
  (h3 : r_A + r_C = 1 / 4) :
  days_for_A r_A r_B r_C = 4.8 := by
  sorry

end NUMINAMATH_GPT_A_alone_days_l227_22742


namespace NUMINAMATH_GPT_average_speed_l227_22709

theorem average_speed (D : ℝ) (hD : D > 0) :
  let t1 := (D / 3) / 80
  let t2 := (D / 3) / 15
  let t3 := (D / 3) / 48
  let total_time := t1 + t2 + t3
  let avg_speed := D / total_time
  avg_speed = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l227_22709


namespace NUMINAMATH_GPT_number_of_homes_cleaned_l227_22780

-- Define constants for the amount Mary earns per home and the total amount she made.
def amount_per_home := 46
def total_amount_made := 276

-- Prove that the number of homes Mary cleaned is 6 given the conditions.
theorem number_of_homes_cleaned : total_amount_made / amount_per_home = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_homes_cleaned_l227_22780


namespace NUMINAMATH_GPT_largest_k_no_perpendicular_lines_l227_22759

theorem largest_k_no_perpendicular_lines (n : ℕ) (h : 0 < n) :
  (∃ k, ∀ (l : Fin n → ℝ) (f : Fin n), (∀ i j, i ≠ j → l i ≠ -1 / (l j)) → k = Nat.ceil (n / 2)) :=
sorry

end NUMINAMATH_GPT_largest_k_no_perpendicular_lines_l227_22759


namespace NUMINAMATH_GPT_equation_represents_circle_of_radius_8_l227_22760

theorem equation_represents_circle_of_radius_8 (k : ℝ) : 
  (x^2 + 14 * x + y^2 + 8 * y - k = 0) → k = -1 ↔ (∃ r, r = 8 ∧ (x + 7)^2 + (y + 4)^2 = r^2) :=
by
  sorry

end NUMINAMATH_GPT_equation_represents_circle_of_radius_8_l227_22760


namespace NUMINAMATH_GPT_min_value_of_translated_function_l227_22781

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

theorem min_value_of_translated_function :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ (Real.pi / 2) → ∀ (ϕ : ℝ), |ϕ| < (Real.pi / 2) →
  ∀ (k : ℤ), f (x + (Real.pi / 6)) (ϕ + (Real.pi / 3) + k * Real.pi) = f x ϕ →
  ∃ y : ℝ, y = - Real.sqrt 3 / 2 := sorry

end NUMINAMATH_GPT_min_value_of_translated_function_l227_22781


namespace NUMINAMATH_GPT_boy_overall_average_speed_l227_22743

noncomputable def total_distance : ℝ := 100
noncomputable def distance1 : ℝ := 15
noncomputable def speed1 : ℝ := 12

noncomputable def distance2 : ℝ := 20
noncomputable def speed2 : ℝ := 8

noncomputable def distance3 : ℝ := 10
noncomputable def speed3 : ℝ := 25

noncomputable def distance4 : ℝ := 15
noncomputable def speed4 : ℝ := 18

noncomputable def distance5 : ℝ := 20
noncomputable def speed5 : ℝ := 10

noncomputable def distance6 : ℝ := 20
noncomputable def speed6 : ℝ := 22

noncomputable def time1 : ℝ := distance1 / speed1
noncomputable def time2 : ℝ := distance2 / speed2
noncomputable def time3 : ℝ := distance3 / speed3
noncomputable def time4 : ℝ := distance4 / speed4
noncomputable def time5 : ℝ := distance5 / speed5
noncomputable def time6 : ℝ := distance6 / speed6

noncomputable def total_time : ℝ := time1 + time2 + time3 + time4 + time5 + time6

noncomputable def overall_average_speed : ℝ := total_distance / total_time

theorem boy_overall_average_speed : overall_average_speed = 100 / (15 / 12 + 20 / 8 + 10 / 25 + 15 / 18 + 20 / 10 + 20 / 22) :=
by
  sorry

end NUMINAMATH_GPT_boy_overall_average_speed_l227_22743


namespace NUMINAMATH_GPT_min_gennadys_needed_l227_22791

variables (A B V G : ℕ)

theorem min_gennadys_needed
  (hA : A = 45)
  (hB : B = 122)
  (hV : V = 27)
  (hG : ∀ i, i < 121 → A + V < 121 → G ≥ 49) :
  G = 49 :=
sorry

end NUMINAMATH_GPT_min_gennadys_needed_l227_22791


namespace NUMINAMATH_GPT_hard_candy_food_coloring_l227_22785

theorem hard_candy_food_coloring
  (lollipop_coloring : ℕ) (hard_candy_coloring : ℕ)
  (num_lollipops : ℕ) (num_hardcandies : ℕ)
  (total_coloring : ℕ)
  (H1 : lollipop_coloring = 8)
  (H2 : num_lollipops = 150)
  (H3 : num_hardcandies = 20)
  (H4 : total_coloring = 1800) :
  (20 * hard_candy_coloring + 150 * lollipop_coloring = total_coloring) → 
  hard_candy_coloring = 30 :=
by
  sorry

end NUMINAMATH_GPT_hard_candy_food_coloring_l227_22785


namespace NUMINAMATH_GPT_Smarties_remainder_l227_22734

theorem Smarties_remainder (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end NUMINAMATH_GPT_Smarties_remainder_l227_22734


namespace NUMINAMATH_GPT_inequality_of_ab_l227_22744

theorem inequality_of_ab (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a ≠ b) :
  Real.sqrt (a * b) < (a - b) / (Real.log a - Real.log b) ∧ 
  (a - b) / (Real.log a - Real.log b) < (a + b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_ab_l227_22744


namespace NUMINAMATH_GPT_third_team_pies_l227_22782

theorem third_team_pies (total first_team second_team : ℕ) (h_total : total = 750) (h_first : first_team = 235) (h_second : second_team = 275) :
  total - (first_team + second_team) = 240 := by
  sorry

end NUMINAMATH_GPT_third_team_pies_l227_22782


namespace NUMINAMATH_GPT_ryan_learning_hours_l227_22789

theorem ryan_learning_hours :
  ∀ (e c s : ℕ) , (e = 6) → (s = 58) → (e = c + 3) → (c = 3) :=
by
  intros e c s he hs hc
  sorry

end NUMINAMATH_GPT_ryan_learning_hours_l227_22789


namespace NUMINAMATH_GPT_area_transformation_l227_22796

variables {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x in a..b, g x = 12) :
  ∫ x in c..d, 4 * g (2 * x + 3) = 48 :=
by
  sorry

end NUMINAMATH_GPT_area_transformation_l227_22796


namespace NUMINAMATH_GPT_pentagon_area_l227_22758

theorem pentagon_area 
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) (side5 : ℝ)
  (h1 : side1 = 12) (h2 : side2 = 20) (h3 : side3 = 30) (h4 : side4 = 15) (h5 : side5 = 25)
  (right_angle : ∃ (a b : ℝ), a = side1 ∧ b = side5 ∧ a^2 + b^2 = (a + b)^2) : 
  ∃ (area : ℝ), area = 600 := 
  sorry

end NUMINAMATH_GPT_pentagon_area_l227_22758


namespace NUMINAMATH_GPT_reciprocal_of_neg6_l227_22729

theorem reciprocal_of_neg6 : 1 / (-6 : ℝ) = -1 / 6 := 
sorry

end NUMINAMATH_GPT_reciprocal_of_neg6_l227_22729


namespace NUMINAMATH_GPT_quadrilateral_is_parallelogram_l227_22795

theorem quadrilateral_is_parallelogram 
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 - 2*a*c - 2*b*d = 0) 
  : (a = c ∧ b = d) → parallelogram :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_is_parallelogram_l227_22795


namespace NUMINAMATH_GPT_total_cost_of_constructing_the_path_l227_22766

open Real

-- Define the conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_path_given : ℝ := 1518.72
def cost_per_sq_m : ℝ := 2

-- Define the total cost to be proven
def total_cost : ℝ := 3037.44

-- The statement to be proven
theorem total_cost_of_constructing_the_path :
  let outer_length := length_field + 2 * path_width
  let outer_width := width_field + 2 * path_width
  let total_area_incl_path := outer_length * outer_width
  let area_field := length_field * width_field
  let computed_area_path := total_area_incl_path - area_field
  let given_cost := area_path_given * cost_per_sq_m
  total_cost = given_cost := by
  sorry

end NUMINAMATH_GPT_total_cost_of_constructing_the_path_l227_22766


namespace NUMINAMATH_GPT_problem_I_solution_problem_II_solution_l227_22741

noncomputable def f (x : ℝ) : ℝ := |3 * x - 2| + |x - 2|

-- Problem (I): Solve the inequality f(x) <= 8
theorem problem_I_solution (x : ℝ) : 
  f x ≤ 8 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (II): Find the range of the real number m
theorem problem_II_solution (x m : ℝ) : 
  f x ≥ (m^2 - m + 2) * |x| ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_GPT_problem_I_solution_problem_II_solution_l227_22741


namespace NUMINAMATH_GPT_find_a_value_l227_22710

theorem find_a_value : (15^2 * 8^3 / 256 = 450) :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l227_22710


namespace NUMINAMATH_GPT_lattice_points_non_visible_square_l227_22792

theorem lattice_points_non_visible_square (n : ℕ) (h : n > 0) : 
  ∃ (a b : ℤ), ∀ (x y : ℤ), a < x ∧ x < a + n ∧ b < y ∧ y < b + n → Int.gcd x y > 1 :=
sorry

end NUMINAMATH_GPT_lattice_points_non_visible_square_l227_22792


namespace NUMINAMATH_GPT_jerry_total_shingles_l227_22731

def roof_length : ℕ := 20
def roof_width : ℕ := 40
def num_roofs : ℕ := 3
def shingles_per_square_foot : ℕ := 8

def area_of_one_side (length width : ℕ) : ℕ :=
  length * width

def total_area_one_roof (area_one_side : ℕ) : ℕ :=
  area_one_side * 2

def total_area_three_roofs (total_area_one_roof : ℕ) : ℕ :=
  total_area_one_roof * num_roofs

def total_shingles_needed (total_area_all_roofs shingles_per_square_foot : ℕ) : ℕ :=
  total_area_all_roofs * shingles_per_square_foot

theorem jerry_total_shingles :
  total_shingles_needed (total_area_three_roofs (total_area_one_roof (area_of_one_side roof_length roof_width))) shingles_per_square_foot = 38400 :=
by
  sorry

end NUMINAMATH_GPT_jerry_total_shingles_l227_22731


namespace NUMINAMATH_GPT_louie_share_of_pie_l227_22738

def fraction_of_pie_taken_home (total_pie : ℚ) (shares : ℚ) : ℚ :=
  2 * (total_pie / shares)

theorem louie_share_of_pie : fraction_of_pie_taken_home (8 / 9) 4 = 4 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_louie_share_of_pie_l227_22738


namespace NUMINAMATH_GPT_interest_rate_first_part_eq_3_l227_22798

variable (T P1 P2 r2 I : ℝ)
variable (hT : T = 3400)
variable (hP1 : P1 = 1300)
variable (hP2 : P2 = 2100)
variable (hr2 : r2 = 5)
variable (hI : I = 144)

theorem interest_rate_first_part_eq_3 (r : ℝ) (h : (P1 * r) / 100 + (P2 * r2) / 100 = I) : r = 3 :=
by
  -- leaning in the proof
  sorry

end NUMINAMATH_GPT_interest_rate_first_part_eq_3_l227_22798


namespace NUMINAMATH_GPT_find_some_number_l227_22794

theorem find_some_number (n m : ℕ) (h : (n / 20) * (n / m) = 1) (n_eq_40 : n = 40) : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_some_number_l227_22794


namespace NUMINAMATH_GPT_find_b_perpendicular_lines_l227_22706

variable (b : ℝ)

theorem find_b_perpendicular_lines :
  (2 * b + (-4) * 3 + 7 * (-1) = 0) → b = 19 / 2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_perpendicular_lines_l227_22706


namespace NUMINAMATH_GPT_average_people_per_hour_rounded_l227_22775

def people_moving_per_hour (total_people : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let total_hours := days * hours_per_day
  (total_people / total_hours : ℕ)

theorem average_people_per_hour_rounded :
  people_moving_per_hour 4500 5 24 = 38 := 
  sorry

end NUMINAMATH_GPT_average_people_per_hour_rounded_l227_22775


namespace NUMINAMATH_GPT_light_glow_duration_l227_22713

-- Define the conditions
def total_time_seconds : ℕ := 4969
def glow_times : ℚ := 292.29411764705884

-- Prove the equivalent statement
theorem light_glow_duration :
  (total_time_seconds / glow_times) = 17 := by
  sorry

end NUMINAMATH_GPT_light_glow_duration_l227_22713


namespace NUMINAMATH_GPT_work_day_percentage_l227_22740

theorem work_day_percentage 
  (work_day_hours : ℕ) 
  (first_meeting_minutes : ℕ) 
  (second_meeting_factor : ℕ) 
  (h_work_day : work_day_hours = 10) 
  (h_first_meeting : first_meeting_minutes = 60) 
  (h_second_meeting_factor : second_meeting_factor = 2) :
  ((first_meeting_minutes + second_meeting_factor * first_meeting_minutes) / (work_day_hours * 60) : ℚ) * 100 = 30 :=
sorry

end NUMINAMATH_GPT_work_day_percentage_l227_22740


namespace NUMINAMATH_GPT_NutsInThirdBox_l227_22778

variable (x y z : ℝ)

theorem NutsInThirdBox (h1 : x = (y + z) - 6) (h2 : y = (x + z) - 10) : z = 16 := 
sorry

end NUMINAMATH_GPT_NutsInThirdBox_l227_22778
