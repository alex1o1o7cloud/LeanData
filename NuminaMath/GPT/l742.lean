import Mathlib

namespace length_of_crate_l742_74228

theorem length_of_crate (h crate_dim : ℕ) (radius : ℕ) (h_radius : radius = 8) 
  (h_dims : crate_dim = 18) (h_fit : 2 * radius = 16)
  : h = 18 := 
sorry

end length_of_crate_l742_74228


namespace current_job_wage_l742_74230

variable (W : ℝ) -- Maisy's wage per hour at her current job

-- Define the conditions
def current_job_hours : ℝ := 8
def new_job_hours : ℝ := 4
def new_job_wage_per_hour : ℝ := 15
def new_job_bonus : ℝ := 35
def additional_new_job_earnings : ℝ := 15

-- Assert the given condition
axiom job_earnings_condition : 
  new_job_hours * new_job_wage_per_hour + new_job_bonus 
  = current_job_hours * W + additional_new_job_earnings

-- The theorem we want to prove
theorem current_job_wage : W = 10 := by
  sorry

end current_job_wage_l742_74230


namespace ginger_size_l742_74226

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l742_74226


namespace inequality_unequal_positive_numbers_l742_74289

theorem inequality_unequal_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by
sorry

end inequality_unequal_positive_numbers_l742_74289


namespace middle_number_in_8th_row_l742_74275

-- Define a function that describes the number on the far right of the nth row.
def far_right_number (n : ℕ) : ℕ := n^2

-- Define a function that calculates the number of elements in the nth row.
def row_length (n : ℕ) : ℕ := 2 * n - 1

-- Define the middle number in the nth row.
def middle_number (n : ℕ) : ℕ := 
  let mid_index := (row_length n + 1) / 2
  far_right_number (n - 1) + mid_index

-- Statement to prove the middle number in the 8th row is 57
theorem middle_number_in_8th_row : middle_number 8 = 57 :=
by
  -- Placeholder for proof
  sorry

end middle_number_in_8th_row_l742_74275


namespace simplify_expression_l742_74218

variables {a b : ℝ}

theorem simplify_expression (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := 
by
  sorry

end simplify_expression_l742_74218


namespace problem_l742_74271

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  (1 / (a - b)) + (1 / (b - c)) + (1 / (c - d)) ≥ (9 / (a - d)) :=
sorry

end problem_l742_74271


namespace positively_correlated_variables_l742_74246

-- Define all conditions given in the problem
def weightOfCarVar1 : Type := ℝ
def avgDistPerLiter : Type := ℝ
def avgStudyTime : Type := ℝ
def avgAcademicPerformance : Type := ℝ
def dailySmokingAmount : Type := ℝ
def healthCondition : Type := ℝ
def sideLength : Type := ℝ
def areaOfSquare : Type := ℝ
def fuelConsumptionPerHundredKm : Type := ℝ

-- Define the relationship status between variables
def isPositivelyCorrelated (x y : Type) : Prop := sorry
def isFunctionallyRelated (x y : Type) : Prop := sorry

axiom weight_car_distance_neg : ¬ isPositivelyCorrelated weightOfCarVar1 avgDistPerLiter
axiom study_time_performance_pos : isPositivelyCorrelated avgStudyTime avgAcademicPerformance
axiom smoking_health_neg : ¬ isPositivelyCorrelated dailySmokingAmount healthCondition
axiom side_area_func : isFunctionallyRelated sideLength areaOfSquare
axiom car_weight_fuel_pos : isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm

-- The proof statement to prove C is the correct answer
theorem positively_correlated_variables:
  isPositivelyCorrelated avgStudyTime avgAcademicPerformance ∧
  isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm :=
by
  sorry

end positively_correlated_variables_l742_74246


namespace parabola_point_l742_74277

theorem parabola_point (a b c : ℝ) (hA : 0.64 * a - 0.8 * b + c = 4.132)
  (hB : 1.44 * a + 1.2 * b + c = -1.948) (hC : 7.84 * a + 2.8 * b + c = -3.932) :
  0.5 * (1.8)^2 - 3.24 * 1.8 + 1.22 = -2.992 :=
by
  -- Proof is intentionally omitted
  sorry

end parabola_point_l742_74277


namespace boats_solution_l742_74235

theorem boats_solution (x y : ℕ) (h1 : x + y = 42) (h2 : 6 * x = 8 * y) : x = 24 ∧ y = 18 :=
by
  sorry

end boats_solution_l742_74235


namespace neg_p_l742_74276

variable {α : Type}
variable (x : α)

def p (x : Real) : Prop := ∀ x : Real, x > 1 → x^2 - 1 > 0

theorem neg_p : ¬( ∀ x : Real, x > 1 → x^2 - 1 > 0) ↔ ∃ x : Real, x > 1 ∧ x^2 - 1 ≤ 0 := 
by 
  sorry

end neg_p_l742_74276


namespace sub_from_square_l742_74208

theorem sub_from_square (n : ℕ) (h : n = 17) : (n * n - n) = 272 :=
by 
  -- Proof goes here
  sorry

end sub_from_square_l742_74208


namespace no_such_function_exists_l742_74264

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (f 0 > 0) ∧ (∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :=
by
  -- proof to be completed
  sorry

end no_such_function_exists_l742_74264


namespace intersection_A_B_l742_74265

def A : Set ℤ := { -2, -1, 0, 1, 2 }
def B : Set ℤ := { x : ℤ | x < 1 }

theorem intersection_A_B : A ∩ B = { -2, -1, 0 } :=
by sorry

end intersection_A_B_l742_74265


namespace right_triangle_third_side_square_l742_74282

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) :
  c^2 = 28 ∨ c^2 = 100 :=
by { sorry }

end right_triangle_third_side_square_l742_74282


namespace total_squares_after_erasing_lines_l742_74283

theorem total_squares_after_erasing_lines :
  ∀ (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ), a = 16 → b = 4 → c = 9 → d = 2 → 
  a - b + c - d + (a / 16) = 22 := 
by
  intro a b c d h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_squares_after_erasing_lines_l742_74283


namespace find_m_l742_74267

-- Mathematical definitions from the given conditions
def condition1 (m : ℝ) : Prop := m^2 - 2 * m - 2 = 1
def condition2 (m : ℝ) : Prop := m + 1/2 * m^2 > 0

-- The proof problem summary
theorem find_m (m : ℝ) (h1 : condition1 m) (h2 : condition2 m) : m = 3 :=
by
  sorry

end find_m_l742_74267


namespace triangle_angle_sum_l742_74280

theorem triangle_angle_sum (y : ℝ) (h : 40 + 3 * y + (y + 10) = 180) : y = 32.5 :=
by
  sorry

end triangle_angle_sum_l742_74280


namespace reduced_price_after_exchange_rate_fluctuation_l742_74225

-- Definitions based on conditions
variables (P : ℝ) -- Original price per kg

def reduced_price_per_kg : ℝ := 0.9 * P

axiom six_kg_costs_900 : 6 * reduced_price_per_kg P = 900

-- Additional conditions
def exchange_rate_factor : ℝ := 1.02

-- Question restated as the theorem to prove
theorem reduced_price_after_exchange_rate_fluctuation : 
  ∃ P : ℝ, reduced_price_per_kg P * exchange_rate_factor = 153 :=
sorry

end reduced_price_after_exchange_rate_fluctuation_l742_74225


namespace component_probability_l742_74281

theorem component_probability (p : ℝ) 
  (h : (1 - p)^3 = 0.001) : 
  p = 0.9 :=
sorry

end component_probability_l742_74281


namespace find_a_plus_b_l742_74231

def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem find_a_plus_b (a b : ℝ) (h1 : f (a - 3) = -3) (h2 : f (b - 3) = 1) :
  a + b = 6 :=
sorry

end find_a_plus_b_l742_74231


namespace simplify_polynomial_l742_74247

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l742_74247


namespace problem_part1_problem_part2_l742_74215

-- Define what we need to prove
theorem problem_part1 (x : ℝ) (a b : ℤ) 
  (h : (2*x - 21)*(3*x - 7) - (3*x - 7)*(x - 13) = (3*x + a)*(x + b)): 
  a + 3*b = -31 := 
by {
  -- We know from the problem that h holds,
  -- thus the values of a and b must satisfy the condition.
  sorry
}

theorem problem_part2 (x : ℝ) : 
  (x^2 - 3*x + 2) = (x - 1)*(x - 2) := 
by {
  sorry
}

end problem_part1_problem_part2_l742_74215


namespace guppies_to_angelfish_ratio_l742_74266

noncomputable def goldfish : ℕ := 8
noncomputable def angelfish : ℕ := goldfish + 4
noncomputable def total_fish : ℕ := 44
noncomputable def guppies : ℕ := total_fish - (goldfish + angelfish)

theorem guppies_to_angelfish_ratio :
    guppies / angelfish = 2 := by
    sorry

end guppies_to_angelfish_ratio_l742_74266


namespace find_value_l742_74252

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom symmetric_about_one : ∀ x, f (x - 1) = f (1 - x)
axiom equation_on_interval : ∀ x, 0 < x ∧ x < 1 → f x = 9^x

theorem find_value : f (5 / 2) + f 2 = -3 := 
by sorry

end find_value_l742_74252


namespace stella_dolls_count_l742_74295

variables (D : ℕ) (clocks glasses P_doll P_clock P_glass cost profit : ℕ)

theorem stella_dolls_count (h_clocks : clocks = 2)
                     (h_glasses : glasses = 5)
                     (h_P_doll : P_doll = 5)
                     (h_P_clock : P_clock = 15)
                     (h_P_glass : P_glass = 4)
                     (h_cost : cost = 40)
                     (h_profit : profit = 25) :
  D = 3 :=
by sorry

end stella_dolls_count_l742_74295


namespace cricket_team_average_age_l742_74278

open Real

-- Definitions based on the conditions given
def team_size := 11
def captain_age := 27
def wicket_keeper_age := 30
def remaining_players_size := team_size - 2

-- The mathematically equivalent proof problem in Lean statement
theorem cricket_team_average_age :
  ∃ A : ℝ,
    (A - 1) * remaining_players_size = (A * team_size) - (captain_age + wicket_keeper_age) ∧
    A = 24 :=
by
  sorry

end cricket_team_average_age_l742_74278


namespace linear_regression_equation_demand_prediction_l742_74238

def data_x : List ℝ := [12, 11, 10, 9, 8]
def data_y : List ℝ := [5, 6, 8, 10, 11]

noncomputable def mean_x : ℝ := (12 + 11 + 10 + 9 + 8) / 5
noncomputable def mean_y : ℝ := (5 + 6 + 8 + 10 + 11) / 5

noncomputable def numerator : ℝ := 
  (12 - mean_x) * (5 - mean_y) + 
  (11 - mean_x) * (6 - mean_y) +
  (10 - mean_x) * (8 - mean_y) +
  (9 - mean_x) * (10 - mean_y) +
  (8 - mean_x) * (11 - mean_y)

noncomputable def denominator : ℝ := 
  (12 - mean_x)^2 + 
  (11 - mean_x)^2 +
  (10 - mean_x)^2 +
  (9 - mean_x)^2 +
  (8 - mean_x)^2

noncomputable def slope_b : ℝ := numerator / denominator
noncomputable def intercept_a : ℝ := mean_y - slope_b * mean_x

theorem linear_regression_equation :
  (slope_b = -1.6) ∧ (intercept_a = 24) :=
by
  sorry

noncomputable def predicted_y (x : ℝ) : ℝ :=
  slope_b * x + intercept_a

theorem demand_prediction :
  predicted_y 6 = 14.4 ∧ (predicted_y 6 < 15) :=
by
  sorry

end linear_regression_equation_demand_prediction_l742_74238


namespace probability_of_drawing_2_black_and_2_white_l742_74213

def total_balls : ℕ := 17
def black_balls : ℕ := 9
def white_balls : ℕ := 8
def balls_drawn : ℕ := 4
def favorable_outcomes := (Nat.choose 9 2) * (Nat.choose 8 2)
def total_outcomes := Nat.choose 17 4
def probability_draw : ℚ := favorable_outcomes / total_outcomes

theorem probability_of_drawing_2_black_and_2_white :
  probability_draw = 168 / 397 :=
by
  sorry

end probability_of_drawing_2_black_and_2_white_l742_74213


namespace sum_of_squares_first_28_l742_74224

theorem sum_of_squares_first_28 : 
  (28 * (28 + 1) * (2 * 28 + 1)) / 6 = 7722 := by
  sorry

end sum_of_squares_first_28_l742_74224


namespace point_in_third_quadrant_l742_74250

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 1 + 2 * m < 0) : m < -1 / 2 := 
by 
  sorry

end point_in_third_quadrant_l742_74250


namespace remaining_amount_is_correct_l742_74290

-- Define the original price based on the deposit paid
def original_price : ℝ := 1500

-- Define the discount percentage
def discount_percentage : ℝ := 0.05

-- Define the sales tax percentage
def tax_percentage : ℝ := 0.075

-- Define the deposit already paid
def deposit_paid : ℝ := 150

-- Define the discounted price
def discounted_price : ℝ := original_price * (1 - discount_percentage)

-- Define the sales tax amount
def sales_tax : ℝ := discounted_price * tax_percentage

-- Define the final cost after adding sales tax
def final_cost : ℝ := discounted_price + sales_tax

-- Define the remaining amount to be paid
def remaining_amount : ℝ := final_cost - deposit_paid

-- The statement we need to prove
theorem remaining_amount_is_correct : remaining_amount = 1381.875 :=
by
  -- We'd normally write the proof here, but that's not required for this task.
  sorry

end remaining_amount_is_correct_l742_74290


namespace tony_lift_ratio_l742_74211

noncomputable def curl_weight := 90
noncomputable def military_press_weight := 2 * curl_weight
noncomputable def squat_weight := 900

theorem tony_lift_ratio : 
  squat_weight / military_press_weight = 5 :=
by
  sorry

end tony_lift_ratio_l742_74211


namespace total_fish_in_lake_l742_74293

-- Given conditions:
def initiallyTaggedFish : ℕ := 100
def capturedFish : ℕ := 100
def taggedFishInAugust : ℕ := 5
def taggedFishMortalityRate : ℝ := 0.3
def newcomerFishRate : ℝ := 0.2

-- Proof to show that the total number of fish at the beginning of April is 1120
theorem total_fish_in_lake (initiallyTaggedFish capturedFish taggedFishInAugust : ℕ) 
  (taggedFishMortalityRate newcomerFishRate : ℝ) : 
  (taggedFishInAugust : ℝ) / (capturedFish * (1 - newcomerFishRate)) = 
  ((initiallyTaggedFish * (1 - taggedFishMortalityRate)) : ℝ) / (1120 : ℝ) :=
by 
  sorry

end total_fish_in_lake_l742_74293


namespace sum_of_inverses_gt_one_l742_74292

variable (a1 a2 a3 S : ℝ)

theorem sum_of_inverses_gt_one
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (ineq1 : a1^2 / (a1 - 1) > S)
  (ineq2 : a2^2 / (a2 - 1) > S)
  (ineq3 : a3^2 / (a3 - 1) > S) :
  1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1 := by
  sorry

end sum_of_inverses_gt_one_l742_74292


namespace sqrt_condition_sqrt_not_meaningful_2_l742_74200

theorem sqrt_condition (x : ℝ) : 1 - x ≥ 0 ↔ x ≤ 1 := 
by
  sorry

theorem sqrt_not_meaningful_2 : ¬(1 - 2 ≥ 0) :=
by
  sorry

end sqrt_condition_sqrt_not_meaningful_2_l742_74200


namespace estimate_correctness_l742_74201

noncomputable def total_species_estimate (A B C : ℕ) : Prop :=
  A = 2400 ∧ B = 1440 ∧ C = 3600

theorem estimate_correctness (A B C taggedA taggedB taggedC caught : ℕ) 
  (h1 : taggedA = 40) 
  (h2 : taggedB = 40) 
  (h3 : taggedC = 40)
  (h4 : caught = 180)
  (h5 : 3 * A = taggedA * caught) 
  (h6 : 5 * B = taggedB * caught) 
  (h7 : 2 * C = taggedC * caught) 
  : total_species_estimate A B C := 
by
  sorry

end estimate_correctness_l742_74201


namespace number_of_integer_solutions_l742_74297

theorem number_of_integer_solutions
    (a : ℤ)
    (x : ℤ)
    (h1 : ∃ x : ℤ, (1 - a) / (x - 2) + 2 = 1 / (2 - x))
    (h2 : ∀ x : ℤ, 4 * x ≥ 3 * (x - 1) ∧ x + (2 * x - 1) / 2 < (a - 1) / 2) :
    (a = 4) :=
sorry

end number_of_integer_solutions_l742_74297


namespace fedora_cleaning_time_l742_74257

-- Definitions based on given conditions
def cleaning_time_per_section (total_time sections_cleaned : ℕ) : ℕ :=
  total_time / sections_cleaned

def remaining_sections (total_sections cleaned_sections : ℕ) : ℕ :=
  total_sections - cleaned_sections

def total_cleaning_time (remaining_sections time_per_section : ℕ) : ℕ :=
  remaining_sections * time_per_section

-- Theorem statement
theorem fedora_cleaning_time 
  (total_time : ℕ) 
  (sections_cleaned : ℕ)
  (additional_time : ℕ)
  (additional_sections : ℕ)
  (cleaned_sections : ℕ)
  (total_sections : ℕ)
  (h1 : total_time = 33)
  (h2 : sections_cleaned = 3)
  (h3 : additional_time = 165)
  (h4 : additional_sections = 15)
  (h5 : cleaned_sections = 3)
  (h6 : total_sections = 18)
  (h7 : cleaning_time_per_section total_time sections_cleaned = 11)
  (h8 : remaining_sections total_sections cleaned_sections = additional_sections)
  : total_cleaning_time additional_sections (cleaning_time_per_section total_time sections_cleaned) = additional_time := sorry

end fedora_cleaning_time_l742_74257


namespace total_number_of_employees_l742_74239

theorem total_number_of_employees (n : ℕ) (hm : ℕ) (hd : ℕ) 
  (h_ratio : 4 * hd = hm)
  (h_diff : hm = hd + 72) : n = 120 :=
by
  -- proof steps would go here
  sorry

end total_number_of_employees_l742_74239


namespace inequality_proof_l742_74274

variable {a b c : ℝ}

theorem inequality_proof (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a^2 = b^2 + c^2) :
  a^3 + b^3 + c^3 ≥ (2*Real.sqrt 2 + 1) / 7 * (a^2 * (b + c) + b^2 * (c + a) + c^2 * (a + b)) := 
sorry

end inequality_proof_l742_74274


namespace part1_part2_l742_74229

open Set

variable {α : Type*} [PartialOrder α]

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem part1 : A ∩ B = {x | 2 < x ∧ x < 3} :=
by
  sorry

theorem part2 : (compl B) = {x | x ≤ 1 ∨ x ≥ 3} :=
by
  sorry

end part1_part2_l742_74229


namespace probability_red_or_green_l742_74216

variable (P_brown P_purple P_green P_red P_yellow : ℝ)

def conditions : Prop :=
  P_brown = 0.3 ∧
  P_brown = 3 * P_purple ∧
  P_green = P_purple ∧
  P_red = P_yellow ∧
  P_brown + P_purple + P_green + P_red + P_yellow = 1

theorem probability_red_or_green (h : conditions P_brown P_purple P_green P_red P_yellow) :
  P_red + P_green = 0.35 :=
by
  sorry

end probability_red_or_green_l742_74216


namespace value_of_expression_l742_74285

theorem value_of_expression (a b c d x y : ℤ) 
  (h1 : a = -b) 
  (h2 : c * d = 1)
  (h3 : abs x = 3)
  (h4 : y = -1) : 
  2 * x - c * d + 6 * (a + b) - abs y = 4 ∨ 2 * x - c * d + 6 * (a + b) - abs y = -8 := 
by 
  sorry

end value_of_expression_l742_74285


namespace magic_trick_constant_l742_74279

theorem magic_trick_constant (a : ℚ) : ((2 * a + 8) / 4 - a / 2) = 2 :=
by
  sorry

end magic_trick_constant_l742_74279


namespace y_values_l742_74255

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sin x / |Real.sin x|) + (|Real.cos x| / Real.cos x) + (Real.tan x / |Real.tan x|)

theorem y_values (x : ℝ) (h1 : 0 < x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x ≠ 0) (h4 : Real.cos x ≠ 0) (h5 : Real.tan x ≠ 0) :
  y x = 3 ∨ y x = -1 :=
sorry

end y_values_l742_74255


namespace express_1997_using_elevent_fours_l742_74298

def number_expression_uses_eleven_fours : Prop :=
  (4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997)
  
theorem express_1997_using_elevent_fours : number_expression_uses_eleven_fours :=
by
  sorry

end express_1997_using_elevent_fours_l742_74298


namespace train_speed_kmph_l742_74261

noncomputable def train_speed_mps : ℝ := 60.0048

def conversion_factor : ℝ := 3.6

theorem train_speed_kmph : train_speed_mps * conversion_factor = 216.01728 := by
  sorry

end train_speed_kmph_l742_74261


namespace card_paiting_modulus_l742_74251

theorem card_paiting_modulus (cards : Finset ℕ) (H : cards = Finset.range 61 \ {0}) :
  ∃ d : ℕ, ∀ n ∈ cards, ∃! k, (∀ x ∈ cards, (x + n ≡ k [MOD d])) ∧ (d ∣ 30) ∧ (∃! n : ℕ, 1 ≤ n ∧ n ≤ 8) :=
sorry

end card_paiting_modulus_l742_74251


namespace arithmetic_sequence_first_term_l742_74212

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (k : ℕ) (hk : k ≥ 2)
  (hS : S k = 5)
  (ha_k2_p1 : a (k^2 + 1) = -45)
  (ha_sum : (Finset.range (2 * k + 1) \ Finset.range (k + 1)).sum a = -45) :
  a 1 = 5 := 
sorry

end arithmetic_sequence_first_term_l742_74212


namespace volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l742_74219

namespace RectangularPrism

def length := 4
def width := 2
def height := 1

theorem volume_eq_eight : length * width * height = 8 := sorry

theorem space_diagonal_eq_sqrt21 :
  Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) = Real.sqrt 21 := sorry

theorem surface_area_neq_24 :
  2 * (length * width + width * height + height * length) ≠ 24 := sorry

theorem circumscribed_sphere_area_eq_21pi :
  4 * Real.pi * ((Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2) / 2) ^ 2) = 21 * Real.pi := sorry

end RectangularPrism

end volume_eq_eight_space_diagonal_eq_sqrt21_surface_area_neq_24_circumscribed_sphere_area_eq_21pi_l742_74219


namespace not_right_triangle_l742_74233

theorem not_right_triangle (A B C : ℝ) (hA : A + B = 180 - C) 
  (hB : A = B / 2 ∧ A = C / 3) 
  (hC : A = B / 2 ∧ B = C / 1.5) 
  (hD : A = 2 * B ∧ A = 3 * C):
  (C ≠ 90) :=
by {
  sorry
}

end not_right_triangle_l742_74233


namespace total_strawberries_l742_74288

-- Define the number of original strawberries and the number of picked strawberries
def original_strawberries : ℕ := 42
def picked_strawberries : ℕ := 78

-- Prove the total number of strawberries
theorem total_strawberries : original_strawberries + picked_strawberries = 120 := by
  -- Proof goes here
  sorry

end total_strawberries_l742_74288


namespace calculate_group_A_B_C_and_total_is_correct_l742_74245

def groupA_1week : Int := 175000
def groupA_2week : Int := 107000
def groupA_3week : Int := 35000
def groupB_1week : Int := 100000
def groupB_2week : Int := 70350
def groupB_3week : Int := 19500
def groupC_1week : Int := 45000
def groupC_2week : Int := 87419
def groupC_3week : Int := 14425
def kids_staying_home : Int := 590796
def kids_outside_county : Int := 22

def total_kids_in_A := groupA_1week + groupA_2week + groupA_3week
def total_kids_in_B := groupB_1week + groupB_2week + groupB_3week
def total_kids_in_C := groupC_1week + groupC_2week + groupC_3week
def total_kids_in_camp := total_kids_in_A + total_kids_in_B + total_kids_in_C
def total_kids := total_kids_in_camp + kids_staying_home + kids_outside_county

theorem calculate_group_A_B_C_and_total_is_correct :
  total_kids_in_A = 317000 ∧
  total_kids_in_B = 189850 ∧
  total_kids_in_C = 146844 ∧
  total_kids = 1244512 := by
  sorry

end calculate_group_A_B_C_and_total_is_correct_l742_74245


namespace hcf_of_two_numbers_l742_74284

noncomputable def number1 : ℕ := 414

noncomputable def lcm_factors : Set ℕ := {13, 18}

noncomputable def hcf (a b : ℕ) : ℕ := Nat.gcd a b

-- Statement to prove
theorem hcf_of_two_numbers (Y : ℕ) 
  (H : ℕ) 
  (lcm : ℕ) 
  (H_lcm_factors : lcm = H * 13 * 18)
  (H_lcm_prop : lcm = (number1 * Y) / H)
  (H_Y : Y = (H^2 * 13 * 18) / 414)
  : H = 23 := 
sorry

end hcf_of_two_numbers_l742_74284


namespace chocolate_milk_container_size_l742_74287

/-- Holly's chocolate milk consumption conditions and container size -/
theorem chocolate_milk_container_size
  (morning_initial: ℝ)  -- Initial amount in the morning
  (morning_drink: ℝ)    -- Amount drank in the morning with breakfast
  (lunch_drink: ℝ)      -- Amount drank at lunch
  (dinner_drink: ℝ)     -- Amount drank with dinner
  (end_of_day: ℝ)       -- Amount she ends the day with
  (lunch_container_size: ℝ) -- Size of the container bought at lunch
  (C: ℝ)                -- Container size she bought at lunch
  (h_initial: morning_initial = 16)
  (h_morning_drink: morning_drink = 8)
  (h_lunch_drink: lunch_drink = 8)
  (h_dinner_drink: dinner_drink = 8)
  (h_end_of_day: end_of_day = 56) :
  (morning_initial - morning_drink) + C - lunch_drink - dinner_drink = end_of_day → 
  lunch_container_size = 64 :=
by
  sorry

end chocolate_milk_container_size_l742_74287


namespace smallest_five_digit_multiple_of_18_l742_74209

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, n = 10008 ∧ (n ≥ 10000 ∧ n < 100000) ∧ n % 18 = 0 ∧ (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 18 = 0 → n ≤ m) := sorry

end smallest_five_digit_multiple_of_18_l742_74209


namespace forest_leaves_count_correct_l742_74273

def number_of_trees : ℕ := 20
def number_of_main_branches_per_tree : ℕ := 15
def number_of_sub_branches_per_main_branch : ℕ := 25
def number_of_tertiary_branches_per_sub_branch : ℕ := 30
def number_of_leaves_per_sub_branch : ℕ := 75
def number_of_leaves_per_tertiary_branch : ℕ := 45

def total_leaves_on_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch * number_of_leaves_per_sub_branch

def total_sub_branches_per_tree :=
  number_of_main_branches_per_tree * number_of_sub_branches_per_main_branch

def total_leaves_on_tertiary_branches_per_tree :=
  total_sub_branches_per_tree * number_of_tertiary_branches_per_sub_branch * number_of_leaves_per_tertiary_branch

def total_leaves_per_tree :=
  total_leaves_on_sub_branches_per_tree + total_leaves_on_tertiary_branches_per_tree

def total_leaves_in_forest :=
  total_leaves_per_tree * number_of_trees

theorem forest_leaves_count_correct :
  total_leaves_in_forest = 10687500 := 
by sorry

end forest_leaves_count_correct_l742_74273


namespace triangle_area_from_curve_l742_74232

-- Definition of the curve
def curve (x : ℝ) : ℝ := (x - 2)^2 * (x + 3)

-- Area calculation based on intercepts
theorem triangle_area_from_curve : 
  (1 / 2) * (2 - (-3)) * (curve 0) = 30 :=
by
  sorry

end triangle_area_from_curve_l742_74232


namespace geometric_sequence_from_second_term_l742_74262

theorem geometric_sequence_from_second_term (S : ℕ → ℕ) (a : ℕ → ℕ) :
  S 1 = 1 ∧ S 2 = 2 ∧ (∀ n, n ≥ 2 → S (n + 1) - 3 * S n + 2 * S (n - 1) = 0) →
  (∀ n, n ≥ 2 → a (n + 1) = 2 * a n) :=
by
  sorry

end geometric_sequence_from_second_term_l742_74262


namespace min_expression_value_l742_74272

theorem min_expression_value (a b c : ℝ) (ha : 1 ≤ a) (hbc : b ≥ a) (hcb : c ≥ b) (hc5 : c ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * Real.sqrt (5^(1/4)) + 4 :=
sorry

end min_expression_value_l742_74272


namespace lindsey_exercise_bands_l742_74260

theorem lindsey_exercise_bands (x : ℕ) 
  (h1 : ∀ n, n = 5 * x) 
  (h2 : ∀ m, m = 10 * x) 
  (h3 : ∀ d, d = m + 10) 
  (h4 : d = 30) : 
  x = 2 := 
by 
  sorry

end lindsey_exercise_bands_l742_74260


namespace value_of_A_l742_74244

theorem value_of_A (G F L: ℤ) (H1 : G = 15) (H2 : F + L + 15 = 50) (H3 : F + L + 37 + 15 = 65) (H4 : F + ((58 - F - L) / 2) + ((58 - F - L) / 2) + L = 58) : 
  37 = 37 := 
by 
  sorry

end value_of_A_l742_74244


namespace greatest_x_value_l742_74249

theorem greatest_x_value : 
  (∃ x : ℝ, 2 * x^2 + 7 * x + 3 = 5 ∧ ∀ y : ℝ, (2 * y^2 + 7 * y + 3 = 5) → y ≤ x) → x = 1 / 2 :=
by
  sorry

end greatest_x_value_l742_74249


namespace number_of_solution_pairs_l742_74242

def integer_solutions_on_circle : Set (Int × Int) := {
  (1, 7), (1, -7), (-1, 7), (-1, -7),
  (5, 5), (5, -5), (-5, 5), (-5, -5),
  (7, 1), (7, -1), (-7, 1), (-7, -1) 
}

def system_of_equations_has_integer_solution (a b : ℝ) : Prop :=
  ∃ (x y : ℤ), a * ↑x + b * ↑y = 1 ∧ (↑x ^ 2 + ↑y ^ 2 = 50)

theorem number_of_solution_pairs : ∃ (n : ℕ), n = 72 ∧
  (∀ (a b : ℝ), system_of_equations_has_integer_solution a b → n = 72) := 
sorry

end number_of_solution_pairs_l742_74242


namespace min_value_expression_l742_74286

theorem min_value_expression (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1) ^ 2 + 1 / (Real.sin x1) ^ 2) *
  (2 * (Real.sin x2) ^ 2 + 1 / (Real.sin x2) ^ 2) *
  (2 * (Real.sin x3) ^ 2 + 1 / (Real.sin x3) ^ 2) *
  (2 * (Real.sin x4) ^ 2 + 1 / (Real.sin x4) ^ 2) = 81 := 
sorry

end min_value_expression_l742_74286


namespace xiao_ming_excellent_score_probability_l742_74223

theorem xiao_ming_excellent_score_probability :
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  P_E = 0.2 :=
by
  let P_M : ℝ := 0.5
  let P_L : ℝ := 0.3
  let P_E := 1 - P_M - P_L
  sorry

end xiao_ming_excellent_score_probability_l742_74223


namespace fraction_in_range_l742_74217

theorem fraction_in_range : 
  (2:ℝ) / 5 < (4:ℝ) / 7 ∧ (4:ℝ) / 7 < 3 / 4 := by
  sorry

end fraction_in_range_l742_74217


namespace ethan_presents_l742_74263

variable (A E : ℝ)

theorem ethan_presents (h1 : A = 9) (h2 : A = E - 22.0) : E = 31 := 
by
  sorry

end ethan_presents_l742_74263


namespace regression_lines_intersect_at_average_l742_74296

theorem regression_lines_intersect_at_average
  {x_vals1 x_vals2 : List ℝ} {y_vals1 y_vals2 : List ℝ}
  (n1 : x_vals1.length = 100) (n2 : x_vals2.length = 150)
  (mean_x1 : (List.sum x_vals1 / 100) = s) (mean_x2 : (List.sum x_vals2 / 150) = s)
  (mean_y1 : (List.sum y_vals1 / 100) = t) (mean_y2 : (List.sum y_vals2 / 150) = t)
  (regression_line1 : ℝ → ℝ)
  (regression_line2 : ℝ → ℝ)
  (on_line1 : ∀ x, regression_line1 x = (a1 * x + b1))
  (on_line2 : ∀ x, regression_line2 x = (a2 * x + b2))
  (sample_center1 : regression_line1 s = t)
  (sample_center2 : regression_line2 s = t) :
  regression_line1 s = regression_line2 s := sorry

end regression_lines_intersect_at_average_l742_74296


namespace rectangle_percentage_excess_l742_74269

variable (L W : ℝ) -- The lengths of the sides of the rectangle
variable (x : ℝ) -- The percentage excess for the first side (what we want to prove)

theorem rectangle_percentage_excess 
    (h1 : W' = W * 0.95)                    -- Condition: second side is taken with 5% deficit
    (h2 : L' = L * (1 + x/100))             -- Condition: first side is taken with x% excess
    (h3 : A = L * W)                        -- Actual area of the rectangle
    (h4 : 1.064 = (L' * W') / A) :           -- Condition: error percentage in the area is 6.4%
    x = 12 :=                                -- Proof that x equals 12
sorry

end rectangle_percentage_excess_l742_74269


namespace max_value_a4_b4_c4_d4_l742_74236

theorem max_value_a4_b4_c4_d4 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 16) :
  a^4 + b^4 + c^4 + d^4 ≤ 64 :=
sorry

end max_value_a4_b4_c4_d4_l742_74236


namespace symmetric_point_l742_74294

theorem symmetric_point (P : ℝ × ℝ) (a b : ℝ) (h1 : P = (2, 7)) (h2 : 1 * (a - 2) + (b - 7) * (-1) = 0) (h3 : (a + 2) / 2 + (b + 7) / 2 + 1 = 0) :
  (a, b) = (-8, -3) :=
sorry

end symmetric_point_l742_74294


namespace find_b_l742_74254

theorem find_b (a b : ℝ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5 / 2 := 
by 
  sorry

end find_b_l742_74254


namespace negation_proof_l742_74291

theorem negation_proof :
  (¬ ∃ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∀ x : ℝ, x > 1 ∧ x^2 ≤ 4) :=
by
  sorry

end negation_proof_l742_74291


namespace thabo_number_of_hardcover_nonfiction_books_l742_74214

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end thabo_number_of_hardcover_nonfiction_books_l742_74214


namespace increase_in_circumference_l742_74237

theorem increase_in_circumference (d e : ℝ) : (fun d e => let C := π * d; let C_new := π * (d + e); C_new - C) d e = π * e :=
by sorry

end increase_in_circumference_l742_74237


namespace basketball_teams_l742_74299

theorem basketball_teams (boys girls : ℕ) (total_players : ℕ) (team_size : ℕ) (ways : ℕ) :
  boys = 7 → girls = 3 → total_players = 10 → team_size = 5 → ways = 105 → 
  ∃ (girls_in_team1 girls_in_team2 : ℕ), 
    girls_in_team1 + girls_in_team2 = 3 ∧ 
    1 ≤ girls_in_team1 ∧ 
    1 ≤ girls_in_team2 ∧ 
    girls_in_team1 ≠ 0 ∧ 
    girls_in_team2 ≠ 0 ∧ 
    ways = 105 :=
by 
  sorry

end basketball_teams_l742_74299


namespace regular_polygon_property_l742_74253

variables {n : ℕ}
variables {r : ℝ} -- r is the radius of the circumscribed circle
variables {t_2n : ℝ} -- t_2n is the area of the 2n-gon
variables {k_n : ℝ} -- k_n is the perimeter of the n-gon

theorem regular_polygon_property
  (h1 : t_2n = (n * k_n * r) / 2)
  (h2 : k_n = n * a_n) :
  (t_2n / r^2) = (k_n / (2 * r)) :=
by sorry

end regular_polygon_property_l742_74253


namespace Parkway_Elementary_girls_not_playing_soccer_l742_74243

/-
  In the fifth grade at Parkway Elementary School, there are 500 students. 
  350 students are boys and 250 students are playing soccer.
  86% of the students that play soccer are boys.
  Prove that the number of girl students that are not playing soccer is 115.
-/
theorem Parkway_Elementary_girls_not_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (playing_soccer : ℕ)
  (percentage_boys_playing_soccer : ℝ)
  (H1 : total_students = 500)
  (H2 : boys = 350)
  (H3 : playing_soccer = 250)
  (H4 : percentage_boys_playing_soccer = 0.86) :
  ∃ (girls_not_playing_soccer : ℕ), girls_not_playing_soccer = 115 :=
by
  sorry

end Parkway_Elementary_girls_not_playing_soccer_l742_74243


namespace students_enrolled_for_german_l742_74227

theorem students_enrolled_for_german 
  (total_students : ℕ)
  (both_english_german : ℕ)
  (only_english : ℕ)
  (at_least_one_subject : total_students = 32 ∧ both_english_german = 12 ∧ only_english = 10) :
  ∃ G : ℕ, G = 22 :=
by
  -- Lean proof steps will go here.
  sorry

end students_enrolled_for_german_l742_74227


namespace no_more_than_one_100_l742_74203

-- Define the score variables and the conditions
variables (R P M : ℕ)

-- Given conditions: R = P - 3 and P = M - 7
def score_conditions : Prop := R = P - 3 ∧ P = M - 7

-- The maximum score condition
def max_score_condition : Prop := R ≤ 100 ∧ P ≤ 100 ∧ M ≤ 100

-- The goal: it is impossible for Vanya to have scored 100 in more than one exam
theorem no_more_than_one_100 (R P M : ℕ) (h1 : score_conditions R P M) (h2 : max_score_condition R P M) :
  (R = 100 ∧ P = 100) ∨ (P = 100 ∧ M = 100) ∨ (M = 100 ∧ R = 100) → false :=
sorry

end no_more_than_one_100_l742_74203


namespace pam_bags_count_l742_74202

noncomputable def geralds_bag_apples : ℕ := 40

noncomputable def pams_bag_apples := 3 * geralds_bag_apples

noncomputable def pams_total_apples : ℕ := 1200

theorem pam_bags_count : pams_total_apples / pams_bag_apples = 10 := by 
  sorry

end pam_bags_count_l742_74202


namespace max_cards_from_poster_board_l742_74240

theorem max_cards_from_poster_board (card_length card_width poster_length : ℕ) (h1 : card_length = 2) (h2 : card_width = 3) (h3 : poster_length = 12) : 
  (poster_length / card_length) * (poster_length / card_width) = 24 :=
by
  sorry

end max_cards_from_poster_board_l742_74240


namespace logic_problem_l742_74259

variable (p q : Prop)

theorem logic_problem (h₁ : ¬ p) (h₂ : p ∨ q) : p = False ∧ q = True :=
by
  sorry

end logic_problem_l742_74259


namespace monkey_hop_distance_l742_74205

theorem monkey_hop_distance
    (total_height : ℕ)
    (slip_back : ℕ)
    (hours : ℕ)
    (reach_time : ℕ)
    (hop : ℕ)
    (H1 : total_height = 19)
    (H2 : slip_back = 2)
    (H3 : hours = 17)
    (H4 : reach_time = 16 * (hop - slip_back) + hop)
    (H5 : total_height = reach_time) :
    hop = 3 := by
  sorry

end monkey_hop_distance_l742_74205


namespace lipstick_cost_correct_l742_74256

noncomputable def cost_of_lipsticks (total_cost: ℕ) (cost_slippers: ℚ) (cost_hair_color: ℚ) (paid: ℚ) (number_lipsticks: ℕ) : ℚ :=
  (paid - (6 * cost_slippers + 8 * cost_hair_color)) / number_lipsticks

theorem lipstick_cost_correct :
  cost_of_lipsticks 6 (2.5:ℚ) (3:ℚ) (44:ℚ) 4 = 1.25 := by
  sorry

end lipstick_cost_correct_l742_74256


namespace A_alone_work_days_l742_74222

noncomputable def A_and_B_together : ℕ := 40
noncomputable def A_and_B_worked_together_days : ℕ := 10
noncomputable def B_left_and_C_joined_after_days : ℕ := 6
noncomputable def A_and_C_finish_remaining_work_days : ℕ := 15
noncomputable def C_alone_work_days : ℕ := 60

theorem A_alone_work_days (h1 : A_and_B_together = 40)
                          (h2 : A_and_B_worked_together_days = 10)
                          (h3 : B_left_and_C_joined_after_days = 6)
                          (h4 : A_and_C_finish_remaining_work_days = 15)
                          (h5 : C_alone_work_days = 60) : ∃ (n : ℕ), n = 30 :=
by {
  sorry -- Proof goes here
}

end A_alone_work_days_l742_74222


namespace rectangle_width_is_nine_l742_74248

theorem rectangle_width_is_nine (w l : ℝ) (h1 : l = 2 * w)
  (h2 : l * w = 3 * 2 * (l + w)) : 
  w = 9 :=
by
  sorry

end rectangle_width_is_nine_l742_74248


namespace Bert_total_profit_is_14_90_l742_74234

-- Define the sales price for each item
def sales_price_barrel : ℝ := 90
def sales_price_tools : ℝ := 50
def sales_price_fertilizer : ℝ := 30

-- Define the tax rates for each item
def tax_rate_barrel : ℝ := 0.10
def tax_rate_tools : ℝ := 0.05
def tax_rate_fertilizer : ℝ := 0.12

-- Define the profit added per item
def profit_per_item : ℝ := 10

-- Define the tax amount for each item
def tax_barrel : ℝ := tax_rate_barrel * sales_price_barrel
def tax_tools : ℝ := tax_rate_tools * sales_price_tools
def tax_fertilizer : ℝ := tax_rate_fertilizer * sales_price_fertilizer

-- Define the cost price for each item
def cost_price_barrel : ℝ := sales_price_barrel - profit_per_item
def cost_price_tools : ℝ := sales_price_tools - profit_per_item
def cost_price_fertilizer : ℝ := sales_price_fertilizer - profit_per_item

-- Define the profit for each item
def profit_barrel : ℝ := sales_price_barrel - tax_barrel - cost_price_barrel
def profit_tools : ℝ := sales_price_tools - tax_tools - cost_price_tools
def profit_fertilizer : ℝ := sales_price_fertilizer - tax_fertilizer - cost_price_fertilizer

-- Define the total profit
def total_profit : ℝ := profit_barrel + profit_tools + profit_fertilizer

-- Assert the total profit is $14.90
theorem Bert_total_profit_is_14_90 : total_profit = 14.90 :=
by
  -- Omitted proof
  sorry

end Bert_total_profit_is_14_90_l742_74234


namespace tangent_circle_line_l742_74241

theorem tangent_circle_line (a : ℝ) :
  (∀ x y : ℝ, (x - y + 3 = 0) → (x^2 + y^2 - 2 * x + 2 - a = 0)) →
  a = 9 :=
by
  sorry

end tangent_circle_line_l742_74241


namespace count_two_digit_integers_sum_seven_l742_74207

theorem count_two_digit_integers_sum_seven : 
  ∃ n : ℕ, (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 7 → n = 7) := 
by
  sorry

end count_two_digit_integers_sum_seven_l742_74207


namespace jane_mean_score_l742_74204

-- Define the six quiz scores Jane took
def score1 : ℕ := 86
def score2 : ℕ := 91
def score3 : ℕ := 89
def score4 : ℕ := 95
def score5 : ℕ := 88
def score6 : ℕ := 94

-- The number of quizzes
def num_quizzes : ℕ := 6

-- The sum of all quiz scores
def total_score : ℕ := score1 + score2 + score3 + score4 + score5 + score6 

-- The expected mean score
def mean_score : ℚ := 90.5

-- The proof statement
theorem jane_mean_score (h : total_score = 543) : total_score / num_quizzes = mean_score := 
by sorry

end jane_mean_score_l742_74204


namespace abs_c_eq_181_l742_74206

theorem abs_c_eq_181
  (a b c : ℤ)
  (h_gcd : Int.gcd a (Int.gcd b c) = 1)
  (h_eq : a * (Complex.mk 3 2)^4 + b * (Complex.mk 3 2)^3 + c * (Complex.mk 3 2)^2 + b * (Complex.mk 3 2) + a = 0) :
  |c| = 181 :=
sorry

end abs_c_eq_181_l742_74206


namespace inequality_hold_l742_74221

theorem inequality_hold (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  0 < (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ∧ 
  (1 / (x + y + z + 1)) - (1 / ((x + 1) * (y + 1) * (z + 1))) ≤ 1/8 :=
sorry

end inequality_hold_l742_74221


namespace ratio_platform_to_pole_l742_74268

variables (l t T v : ℝ)
-- Conditions
axiom constant_velocity : ∀ t l, l = v * t
axiom pass_pole : l = v * t
axiom pass_platform : 6 * l = v * T 

theorem ratio_platform_to_pole (h1 : l = v * t) (h2 : 6 * l = v * T) : T / t = 6 := 
  by sorry

end ratio_platform_to_pole_l742_74268


namespace decimal_fraction_eq_l742_74258

theorem decimal_fraction_eq {b : ℕ} (hb : 0 < b) :
  (4 * b + 19 : ℚ) / (6 * b + 11) = 0.76 → b = 19 :=
by
  -- Proof goes here
  sorry

end decimal_fraction_eq_l742_74258


namespace prove_zero_l742_74270

variable {a b c : ℝ}

theorem prove_zero (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
by
  sorry

end prove_zero_l742_74270


namespace forty_percent_of_n_l742_74220

theorem forty_percent_of_n (N : ℝ) (h : (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10) : 0.40 * N = 120 := by
  sorry

end forty_percent_of_n_l742_74220


namespace option_C_correct_l742_74210

theorem option_C_correct (m : ℝ) : (-m + 2) * (-m - 2) = m ^ 2 - 4 :=
by sorry

end option_C_correct_l742_74210
