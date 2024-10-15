import Mathlib

namespace NUMINAMATH_GPT_simplify_and_evaluate_l1137_113722

-- Math proof problem
theorem simplify_and_evaluate :
  ∀ (a : ℤ), a = -1 →
  (2 - a)^2 - (1 + a) * (a - 1) - a * (a - 3) = 5 :=
by
  intros a ha
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1137_113722


namespace NUMINAMATH_GPT_number_of_companies_l1137_113784

theorem number_of_companies (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_companies_l1137_113784


namespace NUMINAMATH_GPT_rubber_bands_per_large_ball_l1137_113755

open Nat

theorem rubber_bands_per_large_ball :
  let total_rubber_bands := 5000
  let small_bands := 50
  let small_balls := 22
  let large_balls := 13
  let used_bands := small_balls * small_bands
  let remaining_bands := total_rubber_bands - used_bands
  let large_bands := remaining_bands / large_balls
  large_bands = 300 :=
by
  sorry

end NUMINAMATH_GPT_rubber_bands_per_large_ball_l1137_113755


namespace NUMINAMATH_GPT_negation_ln_eq_x_minus_1_l1137_113780

theorem negation_ln_eq_x_minus_1 :
  ¬(∃ x : ℝ, 0 < x ∧ Real.log x = x - 1) ↔ ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by 
  sorry

end NUMINAMATH_GPT_negation_ln_eq_x_minus_1_l1137_113780


namespace NUMINAMATH_GPT_cost_of_football_and_basketball_max_number_of_basketballs_l1137_113752

-- Problem 1: Cost of one football and one basketball
theorem cost_of_football_and_basketball (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 310) 
  (h2 : 2 * x + 5 * y = 500) : 
  x = 50 ∧ y = 80 :=
sorry

-- Problem 2: Maximum number of basketballs
theorem max_number_of_basketballs (x : ℝ) 
  (h1 : 50 * (96 - x) + 80 * x ≤ 5800) 
  (h2 : x ≥ 0) 
  (h3 : x ≤ 96) : 
  x ≤ 33 :=
sorry

end NUMINAMATH_GPT_cost_of_football_and_basketball_max_number_of_basketballs_l1137_113752


namespace NUMINAMATH_GPT_exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l1137_113721

/-- There exists a way to completely tile a 5x6 board with dominos without leaving any gaps. -/
theorem exists_tiling_5x6_no_gaps :
  ∃ (tiling : List (Set (Fin 5 × Fin 6))), True := 
sorry

/-- It is not possible to tile a 5x6 board with dominos such that gaps are left. -/
theorem no_tiling_5x6_with_gaps :
  ¬ ∃ (tiling : List (Set (Fin 5 × Fin 6))), False := 
sorry

/-- It is impossible to tile a 6x6 board with dominos. -/
theorem no_tiling_6x6 :
  ¬ ∃ (tiling : List (Set (Fin 6 × Fin 6))), True := 
sorry

end NUMINAMATH_GPT_exists_tiling_5x6_no_gaps_no_tiling_5x6_with_gaps_no_tiling_6x6_l1137_113721


namespace NUMINAMATH_GPT_Shekar_science_marks_l1137_113782

theorem Shekar_science_marks (S : ℕ) : 
  let math_marks := 76
  let social_studies_marks := 82
  let english_marks := 67
  let biology_marks := 75
  let average_marks := 73
  let num_subjects := 5
  ((math_marks + S + social_studies_marks + english_marks + biology_marks) / num_subjects = average_marks) → S = 65 :=
by
  sorry

end NUMINAMATH_GPT_Shekar_science_marks_l1137_113782


namespace NUMINAMATH_GPT_maximum_rectangle_area_l1137_113789

variable (x y : ℝ)

def area (x y : ℝ) : ℝ :=
  x * y

def similarity_condition (x y : ℝ) : Prop :=
  (11 - x) / (y - 6) = 2

theorem maximum_rectangle_area :
  ∃ (x y : ℝ), similarity_condition x y ∧ area x y = 66 :=  by
  sorry

end NUMINAMATH_GPT_maximum_rectangle_area_l1137_113789


namespace NUMINAMATH_GPT_det_calculation_l1137_113730

-- Given conditions
variables (p q r s : ℤ)
variable (h1 : p * s - q * r = -3)

-- Define the matrix and determinant
def matrix_determinant (a b c d : ℤ) := a * d - b * c

-- Problem statement
theorem det_calculation : matrix_determinant (p + 2 * r) (q + 2 * s) r s = -3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_det_calculation_l1137_113730


namespace NUMINAMATH_GPT_sum_of_areas_is_858_l1137_113725

def length1 : ℕ := 1
def length2 : ℕ := 9
def length3 : ℕ := 25
def length4 : ℕ := 49
def length5 : ℕ := 81
def length6 : ℕ := 121

def base_width : ℕ := 3

def area (width : ℕ) (length : ℕ) : ℕ :=
  width * length

def total_area_of_rectangles : ℕ :=
  area base_width length1 +
  area base_width length2 +
  area base_width length3 +
  area base_width length4 +
  area base_width length5 +
  area base_width length6

theorem sum_of_areas_is_858 : total_area_of_rectangles = 858 := by
  sorry

end NUMINAMATH_GPT_sum_of_areas_is_858_l1137_113725


namespace NUMINAMATH_GPT_pipe_B_fill_time_l1137_113714

theorem pipe_B_fill_time (T_B : ℝ) : 
  (1/3 + 1/T_B - 1/4 = 1/3) → T_B = 4 :=
sorry

end NUMINAMATH_GPT_pipe_B_fill_time_l1137_113714


namespace NUMINAMATH_GPT_min_value_of_expression_l1137_113740

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
  (1/x + 4/y + 9/z) >= 36/5 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1137_113740


namespace NUMINAMATH_GPT_base4_base7_digit_difference_l1137_113797

def num_digits_base (n b : ℕ) : ℕ :=
  if b > 1 then Nat.log b n + 1 else 0

theorem base4_base7_digit_difference :
  let n := 1573
  num_digits_base n 4 - num_digits_base n 7 = 2 := by
  sorry

end NUMINAMATH_GPT_base4_base7_digit_difference_l1137_113797


namespace NUMINAMATH_GPT_locus_of_point_is_circle_l1137_113794

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end NUMINAMATH_GPT_locus_of_point_is_circle_l1137_113794


namespace NUMINAMATH_GPT_susie_rhode_island_reds_l1137_113795

variable (R G B_R B_G : ℕ)

def susie_golden_comets := G = 6
def britney_rir := B_R = 2 * R
def britney_golden_comets := B_G = G / 2
def britney_more_chickens := B_R + B_G = R + G + 8

theorem susie_rhode_island_reds
  (h1 : susie_golden_comets G)
  (h2 : britney_rir R B_R)
  (h3 : britney_golden_comets G B_G)
  (h4 : britney_more_chickens R G B_R B_G) :
  R = 11 :=
by
  sorry

end NUMINAMATH_GPT_susie_rhode_island_reds_l1137_113795


namespace NUMINAMATH_GPT_find_k_minus_r_l1137_113744

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end NUMINAMATH_GPT_find_k_minus_r_l1137_113744


namespace NUMINAMATH_GPT_initial_distance_between_Seonghyeon_and_Jisoo_l1137_113748

theorem initial_distance_between_Seonghyeon_and_Jisoo 
  (D : ℝ)
  (h1 : 2000 = (D - 200) + 1000) : 
  D = 1200 :=
by
  sorry

end NUMINAMATH_GPT_initial_distance_between_Seonghyeon_and_Jisoo_l1137_113748


namespace NUMINAMATH_GPT_average_runs_next_10_matches_l1137_113717

theorem average_runs_next_10_matches (avg_first_10 : ℕ) (avg_all_20 : ℕ) (n_matches : ℕ) (avg_next_10 : ℕ) :
  avg_first_10 = 40 ∧ avg_all_20 = 35 ∧ n_matches = 10 → avg_next_10 = 30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_average_runs_next_10_matches_l1137_113717


namespace NUMINAMATH_GPT_range_of_f_l1137_113742

noncomputable def g (x : ℝ) := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

noncomputable def f (x : ℝ) := Real.sqrt (g x ^ 2 - 245)

theorem range_of_f : (Set.range f) = Set.Icc 0 14 := sorry

end NUMINAMATH_GPT_range_of_f_l1137_113742


namespace NUMINAMATH_GPT_family_has_11_eggs_l1137_113769

def initialEggs : ℕ := 10
def eggsUsed : ℕ := 5
def chickens : ℕ := 2
def eggsPerChicken : ℕ := 3

theorem family_has_11_eggs :
  (initialEggs - eggsUsed) + (chickens * eggsPerChicken) = 11 := by
  sorry

end NUMINAMATH_GPT_family_has_11_eggs_l1137_113769


namespace NUMINAMATH_GPT_k_value_l1137_113727

theorem k_value (k : ℝ) (h : 10 * k * (-1)^3 - (-1) - 9 = 0) : k = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_k_value_l1137_113727


namespace NUMINAMATH_GPT_condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l1137_113720

-- Definitions corresponding to each condition
def numMethods_participates_in_one_event (students events : ℕ) : ℕ :=
  events ^ students

def numMethods_event_limit_one_person (students events : ℕ) : ℕ :=
  students * (students - 1) * (students - 2)

def numMethods_person_limit_in_events (students events : ℕ) : ℕ :=
  students ^ events

-- Theorems to be proved
theorem condition1_num_registration_methods : 
  numMethods_participates_in_one_event 6 3 = 729 :=
by
  sorry

theorem condition2_num_registration_methods : 
  numMethods_event_limit_one_person 6 3 = 120 :=
by
  sorry

theorem condition3_num_registration_methods : 
  numMethods_person_limit_in_events 6 3 = 216 :=
by
  sorry

end NUMINAMATH_GPT_condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l1137_113720


namespace NUMINAMATH_GPT_inequality_proof_l1137_113716

variable (a b c : ℝ)

noncomputable def specific_condition (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (1 / a + 1 / b + 1 / c = 1)

theorem inequality_proof (h : specific_condition a b c) :
  (a^a * b * c + b^b * c * a + c^c * a * b) ≥ 27 * (b * c + c * a + a * b) := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1137_113716


namespace NUMINAMATH_GPT_octopus_dressing_orders_l1137_113701

/-- A robotic octopus has four legs, and each leg needs to wear a glove before it can wear a boot.
    Additionally, it has two tentacles that require one bracelet each before putting anything on the legs.
    The total number of valid dressing orders is 1,286,400. -/
theorem octopus_dressing_orders : 
  ∃ (n : ℕ), n = 1286400 :=
by
  sorry

end NUMINAMATH_GPT_octopus_dressing_orders_l1137_113701


namespace NUMINAMATH_GPT_find_initial_passengers_l1137_113779

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_passengers_l1137_113779


namespace NUMINAMATH_GPT_total_flowers_eaten_l1137_113778

theorem total_flowers_eaten :
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = 16.5 :=
by
  let f1 := 2.5
  let f2 := 3.0
  let f3 := 1.5
  let f4 := 2.0
  let f5 := 4.0
  let f6 := 0.5
  let f7 := 3.0
  sorry

end NUMINAMATH_GPT_total_flowers_eaten_l1137_113778


namespace NUMINAMATH_GPT_ratio_of_a_to_b_in_arithmetic_sequence_l1137_113703

theorem ratio_of_a_to_b_in_arithmetic_sequence (a x b : ℝ) (h : a = 0 ∧ b = 2 * x) : (a / b) = 0 :=
  by sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_in_arithmetic_sequence_l1137_113703


namespace NUMINAMATH_GPT_largest_consecutive_odd_integer_sum_l1137_113796

theorem largest_consecutive_odd_integer_sum
  (x : Real)
  (h_sum : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = -378.5) :
  x + 8 = -79.7 + 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_consecutive_odd_integer_sum_l1137_113796


namespace NUMINAMATH_GPT_selection_methods_l1137_113728

theorem selection_methods (students lectures : ℕ) (h_stu : students = 4) (h_lect : lectures = 3) : 
  (lectures ^ students) = 81 := 
by
  rw [h_stu, h_lect]
  rfl

end NUMINAMATH_GPT_selection_methods_l1137_113728


namespace NUMINAMATH_GPT_saroj_age_proof_l1137_113754

def saroj_present_age (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : ℕ :=
  sorry    -- calculation logic would be here but is not needed per instruction

noncomputable def question_conditions (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : Prop :=
  vimal_age_6_years_ago / 6 = saroj_age_6_years_ago / 5 ∧
  (vimal_age_6_years_ago + 10) / 11 = (saroj_age_6_years_ago + 10) / 10 ∧
  saroj_present_age vimal_age_6_years_ago saroj_age_6_years_ago = 16

theorem saroj_age_proof (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) :
  question_conditions vimal_age_6_years_ago saroj_age_6_years_ago :=
  sorry

end NUMINAMATH_GPT_saroj_age_proof_l1137_113754


namespace NUMINAMATH_GPT_seventh_degree_solution_l1137_113792

theorem seventh_degree_solution (a b x : ℝ) :
  (x^7 - 7 * a * x^5 + 14 * a^2 * x^3 - 7 * a^3 * x = b) ↔
  ∃ α β : ℝ, α + β = x ∧ α * β = a ∧ α^7 + β^7 = b :=
by
  sorry

end NUMINAMATH_GPT_seventh_degree_solution_l1137_113792


namespace NUMINAMATH_GPT_joan_gemstone_samples_l1137_113733

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end NUMINAMATH_GPT_joan_gemstone_samples_l1137_113733


namespace NUMINAMATH_GPT_scooter_price_and_installment_l1137_113770

variable {P : ℝ} -- price of the scooter
variable {m : ℝ} -- monthly installment

theorem scooter_price_and_installment (h1 : 0.2 * P = 240) (h2 : (0.8 * P) = 12 * m) : 
  P = 1200 ∧ m = 80 := by
  sorry

end NUMINAMATH_GPT_scooter_price_and_installment_l1137_113770


namespace NUMINAMATH_GPT_value_calculation_l1137_113739

theorem value_calculation :
  6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 :=
by
  sorry

end NUMINAMATH_GPT_value_calculation_l1137_113739


namespace NUMINAMATH_GPT_pushups_fri_is_39_l1137_113710

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end NUMINAMATH_GPT_pushups_fri_is_39_l1137_113710


namespace NUMINAMATH_GPT_g_odd_l1137_113791

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_odd {x₁ x₂ : ℝ} 
  (h₁ : |f x₁ + f x₂| ≥ |g x₁ + g x₂|)
  (hf_odd : ∀ x, f x = -f (-x)) : ∀ x, g x = -g (-x) :=
by
  -- The proof would go here, but it's omitted for the purpose of this translation.
  sorry

end NUMINAMATH_GPT_g_odd_l1137_113791


namespace NUMINAMATH_GPT_census_suitable_survey_l1137_113708

theorem census_suitable_survey (A B C D : Prop) : 
  D := 
sorry

end NUMINAMATH_GPT_census_suitable_survey_l1137_113708


namespace NUMINAMATH_GPT_product_of_two_numbers_l1137_113735

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 :=
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1137_113735


namespace NUMINAMATH_GPT_problem_statement_l1137_113715

noncomputable def count_propositions_and_true_statements 
  (statements : List String)
  (is_proposition : String → Bool)
  (is_true_proposition : String → Bool) 
  : Nat × Nat :=
  let props := statements.filter is_proposition
  let true_props := props.filter is_true_proposition
  (props.length, true_props.length)

theorem problem_statement : 
  (count_propositions_and_true_statements 
     ["Isn't an equilateral triangle an isosceles triangle?",
      "Are two lines perpendicular to the same line necessarily parallel?",
      "A number is either positive or negative",
      "What a beautiful coastal city Zhuhai is!",
      "If x + y is a rational number, then x and y are also rational numbers",
      "Construct △ABC ∼ △A₁B₁C₁"]
     (fun s => 
        s = "A number is either positive or negative" ∨ 
        s = "If x + y is a rational number, then x and y are also rational numbers")
     (fun s => false))
  = (2, 0) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1137_113715


namespace NUMINAMATH_GPT_arc_length_of_circle_l1137_113798

theorem arc_length_of_circle (r : ℝ) (alpha : ℝ) (h_r : r = 10) (h_alpha : alpha = (2 * Real.pi) / 6) : 
  (alpha * r) = (10 * Real.pi) / 3 :=
by
  rw [h_r, h_alpha]
  sorry

end NUMINAMATH_GPT_arc_length_of_circle_l1137_113798


namespace NUMINAMATH_GPT_store_profit_l1137_113729

theorem store_profit 
  (cost_per_item : ℕ)
  (selling_price_decrease : ℕ → ℕ)
  (profit : ℤ)
  (x : ℕ) :
  cost_per_item = 40 →
  (∀ x, selling_price_decrease x = 150 - 5 * (x - 50)) →
  profit = 1500 →
  (((x = 50 ∧ selling_price_decrease 50 = 150) ∨ (x = 70 ∧ selling_price_decrease 70 = 50)) ↔ (x = 50 ∨ x = 70) ∧ profit = 1500) :=
by
  sorry

end NUMINAMATH_GPT_store_profit_l1137_113729


namespace NUMINAMATH_GPT_h_value_l1137_113723

theorem h_value (h : ℝ) : (∃ x : ℝ, x^3 + h * x + 5 = 0 ∧ x = 3) → h = -32 / 3 := by
  sorry

end NUMINAMATH_GPT_h_value_l1137_113723


namespace NUMINAMATH_GPT_S_is_line_l1137_113756

open Complex

noncomputable def S : Set ℂ := { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ 3 * y + 4 * x = 0 }

theorem S_is_line :
  ∃ (m b : ℝ), S = { z : ℂ | ∃ (x y : ℝ), z = x + y * Complex.I ∧ x = m * y + b } :=
sorry

end NUMINAMATH_GPT_S_is_line_l1137_113756


namespace NUMINAMATH_GPT_geometric_series_sum_l1137_113745

theorem geometric_series_sum :
  let a := -1
  let r := -3
  let n := 8
  let S := (a * (r ^ n - 1)) / (r - 1)
  S = 1640 :=
by 
  sorry 

end NUMINAMATH_GPT_geometric_series_sum_l1137_113745


namespace NUMINAMATH_GPT_circle_properties_l1137_113757

noncomputable def circle_eq (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0
noncomputable def line_eq (x y : ℝ) := x + 2*y - 4 = 0
noncomputable def perpendicular (x1 y1 x2 y2 : ℝ) := 
  (x1 * x2 + y1 * y2 = 0)

theorem circle_properties (m : ℝ) (x1 y1 x2 y2 : ℝ) :
  (∀ x y, circle_eq x y m) →
  (∀ x, line_eq x (y1 + y2)) →
  perpendicular (4 - 2*y1) y1 (4 - 2*y2) y2 →
  m = 8 / 5 ∧ 
  (∀ x y, (x^2 + y^2 - (8 / 5) * x - (16 / 5) * y = 0) ↔ 
           (x - (4 - 2*(16/5))) * (x - (4 - 2*(16/5))) + (y - (16/5)) * (y - (16/5)) = 5 - (8/5)) :=
sorry

end NUMINAMATH_GPT_circle_properties_l1137_113757


namespace NUMINAMATH_GPT_scientific_notation_l1137_113747

theorem scientific_notation (n : ℝ) (h : n = 1300000) : n = 1.3 * 10^6 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_l1137_113747


namespace NUMINAMATH_GPT_infinite_nested_radicals_solution_l1137_113753

theorem infinite_nested_radicals_solution :
  ∃ x : ℝ, 
    (∃ y z : ℝ, (y = (x * y)^(1/3) ∧ z = (x + z)^(1/3)) ∧ y = z) ∧ 
    0 < x ∧ x = (3 + Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_infinite_nested_radicals_solution_l1137_113753


namespace NUMINAMATH_GPT_machines_finish_together_in_2_hours_l1137_113765

def machineA_time := 4
def machineB_time := 12
def machineC_time := 6

def machineA_rate := 1 / machineA_time
def machineB_rate := 1 / machineB_time
def machineC_rate := 1 / machineC_time

def combined_rate := machineA_rate + machineB_rate + machineC_rate
def total_time := 1 / combined_rate

-- We want to prove that the total_time for machines A, B, and C to finish the job together is 2 hours.
theorem machines_finish_together_in_2_hours : total_time = 2 := by
  sorry

end NUMINAMATH_GPT_machines_finish_together_in_2_hours_l1137_113765


namespace NUMINAMATH_GPT_tank_fraction_after_adding_water_l1137_113772

noncomputable def fraction_of_tank_full 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  : ℚ :=
(initial_fraction * total_capacity + additional_water) / total_capacity

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  (h_initial : initial_fraction = 3 / 4) 
  (h_addition : additional_water = 4) 
  (h_capacity : total_capacity = 32) 
: fraction_of_tank_full initial_fraction additional_water total_capacity = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_tank_fraction_after_adding_water_l1137_113772


namespace NUMINAMATH_GPT_elena_total_pens_l1137_113731

theorem elena_total_pens (price_x price_y total_cost : ℝ) (num_x : ℕ) (hx1 : price_x = 4.0) (hx2 : price_y = 2.2) 
  (hx3 : total_cost = 42.0) (hx4 : num_x = 6) : 
  ∃ num_total : ℕ, num_total = 14 :=
by
  sorry

end NUMINAMATH_GPT_elena_total_pens_l1137_113731


namespace NUMINAMATH_GPT_neznaika_incorrect_l1137_113705

-- Define the average consumption conditions
def average_consumption_december (total_consumption total_days_cons_december : ℕ) : Prop :=
  total_consumption = 10 * total_days_cons_december

def average_consumption_january (total_consumption total_days_cons_january : ℕ) : Prop :=
  total_consumption = 5 * total_days_cons_january

-- Define the claim to be disproven
def neznaika_claim (days_december_at_least_10 days_january_at_least_10 : ℕ) : Prop :=
  days_december_at_least_10 > days_january_at_least_10

-- Proof statement that the claim is incorrect
theorem neznaika_incorrect (total_days_cons_december total_days_cons_january total_consumption_dec total_consumption_jan : ℕ)
    (days_december_at_least_10 days_january_at_least_10 : ℕ)
    (h1 : average_consumption_december total_consumption_dec total_days_cons_december)
    (h2 : average_consumption_january total_consumption_jan total_days_cons_january)
    (h3 : total_days_cons_december = 31)
    (h4 : total_days_cons_january = 31)
    (h5 : days_december_at_least_10 ≤ total_days_cons_december)
    (h6 : days_january_at_least_10 ≤ total_days_cons_january)
    (h7 : days_december_at_least_10 = 1)
    (h8 : days_january_at_least_10 = 15) : 
    ¬ neznaika_claim days_december_at_least_10 days_january_at_least_10 :=
by
  sorry

end NUMINAMATH_GPT_neznaika_incorrect_l1137_113705


namespace NUMINAMATH_GPT_doll_cost_is_one_l1137_113736

variable (initial_amount : ℕ) (end_amount : ℕ) (number_of_dolls : ℕ)

-- Conditions
def given_conditions : Prop :=
  initial_amount = 100 ∧
  end_amount = 97 ∧
  number_of_dolls = 3

-- Question: Proving the cost of each doll
def cost_per_doll (initial_amount end_amount number_of_dolls : ℕ) : ℕ :=
  (initial_amount - end_amount) / number_of_dolls

theorem doll_cost_is_one (h : given_conditions initial_amount end_amount number_of_dolls) :
  cost_per_doll initial_amount end_amount number_of_dolls = 1 :=
by
  sorry

end NUMINAMATH_GPT_doll_cost_is_one_l1137_113736


namespace NUMINAMATH_GPT_lowest_test_score_dropped_is_35_l1137_113799

theorem lowest_test_score_dropped_is_35 
  (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 50)
  (h2 : min A (min B (min C D)) = D)
  (h3 : (A + B + C) / 3 = 55) : 
  D = 35 := by
  sorry

end NUMINAMATH_GPT_lowest_test_score_dropped_is_35_l1137_113799


namespace NUMINAMATH_GPT_inscribed_circle_implies_rhombus_l1137_113746

theorem inscribed_circle_implies_rhombus (AB : ℝ) (AD : ℝ)
  (h_parallelogram : AB = CD ∧ AD = BC) 
  (h_inscribed : AB + CD = AD + BC) : 
  AB = AD := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_implies_rhombus_l1137_113746


namespace NUMINAMATH_GPT_pencils_per_box_l1137_113750

theorem pencils_per_box (boxes : ℕ) (total_pencils : ℕ) (h1 : boxes = 3) (h2 : total_pencils = 27) : (total_pencils / boxes) = 9 := 
by
  sorry

end NUMINAMATH_GPT_pencils_per_box_l1137_113750


namespace NUMINAMATH_GPT_luke_points_per_round_l1137_113775

-- Definitions for conditions
def total_points : ℤ := 84
def rounds : ℤ := 2
def points_per_round (total_points rounds : ℤ) : ℤ := total_points / rounds

-- Statement of the problem
theorem luke_points_per_round : points_per_round total_points rounds = 42 := 
by 
  sorry

end NUMINAMATH_GPT_luke_points_per_round_l1137_113775


namespace NUMINAMATH_GPT_problem_proof_equality_cases_l1137_113785

theorem problem_proof (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (x * y - 10) ^ 2 ≥ 64 := sorry

theorem equality_cases (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10) ^ 2 = 64 ↔ ((x,y) = (1, 2) ∨ (x,y) = (-3, -6)) := sorry

end NUMINAMATH_GPT_problem_proof_equality_cases_l1137_113785


namespace NUMINAMATH_GPT_linear_function_difference_l1137_113726

-- Define the problem in Lean.
theorem linear_function_difference (g : ℕ → ℝ) (h : ∀ x y : ℕ, g x = 3 * x + g 0) (h_condition : g 4 - g 1 = 9) : g 10 - g 1 = 27 := 
by
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_linear_function_difference_l1137_113726


namespace NUMINAMATH_GPT_Drew_age_is_12_l1137_113773

def Sam_age_current : ℕ := 46
def Sam_age_in_five_years : ℕ := Sam_age_current + 5

def Drew_age_now (D : ℕ) : Prop :=
  Sam_age_in_five_years = 3 * (D + 5)

theorem Drew_age_is_12 (D : ℕ) (h : Drew_age_now D) : D = 12 :=
by
  sorry

end NUMINAMATH_GPT_Drew_age_is_12_l1137_113773


namespace NUMINAMATH_GPT_solve_eqn_l1137_113771

theorem solve_eqn {x : ℝ} : x^4 + (3 - x)^4 = 130 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_eqn_l1137_113771


namespace NUMINAMATH_GPT_exponents_to_99_l1137_113786

theorem exponents_to_99 :
  (1 * 3 / 3^2 / 3^4 / 3^8 * 3^16 * 3^32 * 3^64 = 3^99) :=
sorry

end NUMINAMATH_GPT_exponents_to_99_l1137_113786


namespace NUMINAMATH_GPT_n_squared_plus_one_divides_n_plus_one_l1137_113711

theorem n_squared_plus_one_divides_n_plus_one (n : ℕ) (h : n^2 + 1 ∣ n + 1) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_n_squared_plus_one_divides_n_plus_one_l1137_113711


namespace NUMINAMATH_GPT_num_students_in_research_study_group_prob_diff_classes_l1137_113713

-- Define the number of students in each class and the number of students selected from class (2)
def num_students_class1 : ℕ := 18
def num_students_class2 : ℕ := 27
def selected_from_class2 : ℕ := 3

-- Prove the number of students in the research study group
theorem num_students_in_research_study_group : 
  (∃ (m : ℕ), (m / 18 = 3 / 27) ∧ (m + selected_from_class2 = 5)) := 
by
  sorry

-- Prove the probability that the students speaking in both activities come from different classes
theorem prob_diff_classes : 
  (12 / 25 = 12 / 25) :=
by
  sorry

end NUMINAMATH_GPT_num_students_in_research_study_group_prob_diff_classes_l1137_113713


namespace NUMINAMATH_GPT_range_of_a_l1137_113790

-- Define the conditions and what we want to prove
theorem range_of_a (a : ℝ) (x : ℝ) 
    (h1 : ∀ x, |x - 1| + |x + 1| ≥ 3 * a)
    (h2 : ∀ x, (2 * a - 1) ^ x ≤ 1 → (2 * a - 1) < 1 ∧ (2 * a - 1) > 0) :
    (1 / 2 < a ∧ a ≤ 2 / 3) :=
by
  sorry -- Here will be the proof

end NUMINAMATH_GPT_range_of_a_l1137_113790


namespace NUMINAMATH_GPT_find_origin_coordinates_l1137_113707

variable (x y : ℝ)

def original_eq (x y : ℝ) := x^2 - y^2 - 2*x - 2*y - 1 = 0

def transformed_eq (x' y' : ℝ) := x'^2 - y'^2 = 1

theorem find_origin_coordinates (x y : ℝ) :
  original_eq (x - 1) (y + 1) ↔ transformed_eq x y :=
by
  sorry

end NUMINAMATH_GPT_find_origin_coordinates_l1137_113707


namespace NUMINAMATH_GPT_solution_set_I_range_of_a_II_l1137_113777

def f (x a : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2*x - 1|

theorem solution_set_I (x : ℝ) (a : ℝ) (h : a = 2) :
  f x a ≤ 6 ↔ -1 ≤ x ∧ x ≤ 3 := by
  sorry

theorem range_of_a_II (a : ℝ) :
  (∀ x : ℝ, f x a + g x ≥ 3) ↔ 2 ≤ a := by
  sorry

end NUMINAMATH_GPT_solution_set_I_range_of_a_II_l1137_113777


namespace NUMINAMATH_GPT_find_points_PQ_l1137_113751

-- Define the points A, B, M, and E in 3D space
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨10, 0, 0⟩
def M : Point := ⟨5, 5, 0⟩
def E : Point := ⟨0, 0, 10⟩

-- Define the lines AB and EM
def line_AB (t : ℝ) : Point := ⟨10 * t, 0, 0⟩
def line_EM (s : ℝ) : Point := ⟨5 * s, 5 * s, 10 - 10 * s⟩

-- Define the points P and Q
def P (t : ℝ) : Point := line_AB t
def Q (s : ℝ) : Point := line_EM s

-- Define the distance function in 3D space
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

-- The main theorem
theorem find_points_PQ (t s : ℝ) (h1 : t = 0.4) (h2 : s = 0.8) :
  (P t = ⟨4, 0, 0⟩) ∧ (Q s = ⟨4, 4, 2⟩) ∧
  (distance (P t) (Q s) = distance (line_AB 0.4) (line_EM 0.8)) :=
by
  sorry

end NUMINAMATH_GPT_find_points_PQ_l1137_113751


namespace NUMINAMATH_GPT_find_unknown_number_l1137_113762

theorem find_unknown_number (x : ℕ) (h₁ : (20 + 40 + 60) / 3 = 5 + (10 + 50 + x) / 3) : x = 45 :=
by sorry

end NUMINAMATH_GPT_find_unknown_number_l1137_113762


namespace NUMINAMATH_GPT_males_watch_tvxy_l1137_113761

-- Defining the conditions
def total_watch := 160
def females_watch := 75
def males_dont_watch := 83
def total_dont_watch := 120

-- Proving that the number of males who watch TVXY equals 85
theorem males_watch_tvxy : (total_watch - females_watch) = 85 :=
by sorry

end NUMINAMATH_GPT_males_watch_tvxy_l1137_113761


namespace NUMINAMATH_GPT_express_scientific_notation_l1137_113781

theorem express_scientific_notation : (152300 : ℝ) = 1.523 * 10^5 := 
by
  sorry

end NUMINAMATH_GPT_express_scientific_notation_l1137_113781


namespace NUMINAMATH_GPT_simplify_expr_l1137_113749

theorem simplify_expr (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) =
    8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1137_113749


namespace NUMINAMATH_GPT_equation_satisfied_by_r_l1137_113783

theorem equation_satisfied_by_r {x y z r : ℝ} (h1: x ≠ y) (h2: y ≠ z) (h3: z ≠ x) 
    (h4: x ≠ 0) (h5: y ≠ 0) (h6: z ≠ 0) 
    (h7: ∃ (r: ℝ), x * (y - z) = (y * (z - x)) / r ∧ y * (z - x) = (z * (y - x)) / r ∧ z * (y - x) = (x * (y - z)) * r) 
    : r^2 - r + 1 = 0 := 
sorry

end NUMINAMATH_GPT_equation_satisfied_by_r_l1137_113783


namespace NUMINAMATH_GPT_solve_equations_l1137_113767

theorem solve_equations :
  (∃ x : ℝ, (x + 2) ^ 3 + 1 = 0 ∧ x = -3) ∧
  (∃ x : ℝ, ((3 * x - 2) ^ 2 = 64 ∧ (x = 10/3 ∨ x = -2))) :=
by {
  -- Prove the existence of solutions for both problems
  sorry
}

end NUMINAMATH_GPT_solve_equations_l1137_113767


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l1137_113700

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (Real.cos (34 * Real.pi / 180) - Real.sin (34 * Real.pi / 180))
noncomputable def b : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (128 * Real.pi / 180) + Real.cos (40 * Real.pi / 180) * Real.cos (38 * Real.pi / 180)
noncomputable def c : ℝ := (1 / 2) * (Real.cos (80 * Real.pi / 180) - 2 * (Real.cos (50 * Real.pi / 180))^2 + 1)

theorem relationship_between_a_b_c : b > a ∧ a > c :=
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l1137_113700


namespace NUMINAMATH_GPT_difference_between_percentages_l1137_113712

noncomputable def number : ℝ := 140

noncomputable def percentage_65 (x : ℝ) : ℝ := 0.65 * x

noncomputable def fraction_4_5 (x : ℝ) : ℝ := 0.8 * x

theorem difference_between_percentages 
  (x : ℝ) 
  (hx : x = number) 
  : (fraction_4_5 x) - (percentage_65 x) = 21 := 
by 
  sorry

end NUMINAMATH_GPT_difference_between_percentages_l1137_113712


namespace NUMINAMATH_GPT_Dan_picked_9_plums_l1137_113718

-- Define the constants based on the problem
def M : ℕ := 4 -- Melanie's plums
def S : ℕ := 3 -- Sally's plums
def T : ℕ := 16 -- Total plums picked

-- The number of plums Dan picked
def D : ℕ := T - (M + S)

-- The theorem we want to prove
theorem Dan_picked_9_plums : D = 9 := by
  sorry

end NUMINAMATH_GPT_Dan_picked_9_plums_l1137_113718


namespace NUMINAMATH_GPT_simplify_neg_cube_square_l1137_113704

theorem simplify_neg_cube_square (a : ℝ) : (-a^3)^2 = a^6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_neg_cube_square_l1137_113704


namespace NUMINAMATH_GPT_remainder_of_x_plus_3uy_l1137_113793

-- Given conditions
variables (x y u v : ℕ)
variable (Hdiv : x = u * y + v)
variable (H0_le_v : 0 ≤ v)
variable (Hv_lt_y : v < y)

-- Statement to prove
theorem remainder_of_x_plus_3uy (x y u v : ℕ) (Hdiv : x = u * y + v) (H0_le_v : 0 ≤ v) (Hv_lt_y : v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end NUMINAMATH_GPT_remainder_of_x_plus_3uy_l1137_113793


namespace NUMINAMATH_GPT_find_delta_l1137_113760

theorem find_delta (p q Δ : ℕ) (h₁ : Δ + q = 73) (h₂ : 2 * (Δ + q) + p = 172) (h₃ : p = 26) : Δ = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_delta_l1137_113760


namespace NUMINAMATH_GPT_angle_between_clock_hands_at_7_30_l1137_113702

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end NUMINAMATH_GPT_angle_between_clock_hands_at_7_30_l1137_113702


namespace NUMINAMATH_GPT_first_player_wins_l1137_113774

theorem first_player_wins :
  ∀ {table : Type} {coin : Type} 
  (can_place : table → coin → Prop) -- function defining if a coin can be placed on the table
  (not_overlap : ∀ (t : table) (c1 c2 : coin), (can_place t c1 ∧ can_place t c2) → c1 ≠ c2) -- coins do not overlap
  (first_move_center : table → coin) -- first player places the coin at the center
  (mirror_move : table → coin → coin), -- function to place a coin symmetrically
  (∃ strategy : (table → Prop) → (coin → Prop),
    (∀ (t : table) (p : table → Prop), p t → strategy p (mirror_move t (first_move_center t))) ∧ 
    (∀ (t : table) (p : table → Prop), strategy p (first_move_center t) → p t)) := sorry

end NUMINAMATH_GPT_first_player_wins_l1137_113774


namespace NUMINAMATH_GPT_valid_three_digit_numbers_count_l1137_113788

noncomputable def count_valid_numbers : ℕ :=
  let valid_first_digits := [2, 4, 6, 8].length
  let valid_other_digits := [0, 2, 4, 6, 8].length
  let total_even_digit_3_digit_numbers := valid_first_digits * valid_other_digits * valid_other_digits
  let no_4_or_8_first_digits := [2, 6].length
  let no_4_or_8_other_digits := [0, 2, 6].length
  let numbers_without_4_or_8 := no_4_or_8_first_digits * no_4_or_8_other_digits * no_4_or_8_other_digits
  let numbers_with_4_or_8 := total_even_digit_3_digit_numbers - numbers_without_4_or_8
  let valid_even_sum_count := 50  -- Assumed from the manual checking
  valid_even_sum_count

theorem valid_three_digit_numbers_count :
  count_valid_numbers = 50 :=
by
  sorry

end NUMINAMATH_GPT_valid_three_digit_numbers_count_l1137_113788


namespace NUMINAMATH_GPT_ian_leftover_money_l1137_113719

def ianPayments (initial: ℝ) (colin: ℝ) (helen: ℝ) (benedict: ℝ) (emmaInitial: ℝ) (interest: ℝ) (avaAmount: ℝ) (conversionRate: ℝ) : ℝ :=
  let emmaTotal := emmaInitial + (interest * emmaInitial)
  let avaTotal := (avaAmount * 0.75) * conversionRate
  initial - (colin + helen + benedict + emmaTotal + avaTotal)

theorem ian_leftover_money :
  let initial := 100
  let colin := 20
  let twice_colin := 2 * colin
  let half_helen := twice_colin / 2
  let emmaInitial := 15
  let interest := 0.10
  let avaAmount := 8
  let conversionRate := 1.20
  ianPayments initial colin twice_colin half_helen emmaInitial interest avaAmount conversionRate = -3.70
:= by
  sorry

end NUMINAMATH_GPT_ian_leftover_money_l1137_113719


namespace NUMINAMATH_GPT_decimal_between_0_996_and_0_998_ne_0_997_l1137_113766

theorem decimal_between_0_996_and_0_998_ne_0_997 :
  ∃ x : ℝ, 0.996 < x ∧ x < 0.998 ∧ x ≠ 0.997 :=
by
  sorry

end NUMINAMATH_GPT_decimal_between_0_996_and_0_998_ne_0_997_l1137_113766


namespace NUMINAMATH_GPT_number_of_paths_3x3_l1137_113758

-- Definition of the problem conditions
def grid_moves (n m : ℕ) : ℕ := Nat.choose (n + m) n

-- Lean statement for the proof problem
theorem number_of_paths_3x3 : grid_moves 3 3 = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_paths_3x3_l1137_113758


namespace NUMINAMATH_GPT_averagePrice_is_20_l1137_113787

-- Define the conditions
def books1 : Nat := 32
def cost1 : Nat := 1500

def books2 : Nat := 60
def cost2 : Nat := 340

-- Define the total books and total cost
def totalBooks : Nat := books1 + books2
def totalCost : Nat := cost1 + cost2

-- Define the average price calculation
def averagePrice : Nat := totalCost / totalBooks

-- The statement to prove
theorem averagePrice_is_20 : averagePrice = 20 := by
  -- Sorry is used here as a placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_averagePrice_is_20_l1137_113787


namespace NUMINAMATH_GPT_find_C_marks_l1137_113706

theorem find_C_marks :
  let english := 90
  let math := 92
  let physics := 85
  let biology := 85
  let avg_marks := 87.8
  let total_marks := avg_marks * 5
  let other_marks := english + math + physics + biology
  ∃ C : ℝ, total_marks - other_marks = C ∧ C = 87 :=
by
  sorry

end NUMINAMATH_GPT_find_C_marks_l1137_113706


namespace NUMINAMATH_GPT_tiling_ratio_l1137_113768

theorem tiling_ratio (n a b : ℕ) (ha : a ≠ 0) (H : b = a * 2^(n/2)) :
  b / a = 2^(n/2) :=
  by
  sorry

end NUMINAMATH_GPT_tiling_ratio_l1137_113768


namespace NUMINAMATH_GPT_simplify_polynomial_l1137_113709

theorem simplify_polynomial (x y : ℝ) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1137_113709


namespace NUMINAMATH_GPT_total_wheels_l1137_113776

def num_wheels_in_garage : Nat :=
  let cars := 2 * 4
  let lawnmower := 4
  let bicycles := 3 * 2
  let tricycle := 3
  let unicycle := 1
  let skateboard := 4
  let wheelbarrow := 1
  let wagon := 4
  let dolly := 2
  let shopping_cart := 4
  let scooter := 2
  cars + lawnmower + bicycles + tricycle + unicycle + skateboard + wheelbarrow + wagon + dolly + shopping_cart + scooter

theorem total_wheels : num_wheels_in_garage = 39 := by
  sorry

end NUMINAMATH_GPT_total_wheels_l1137_113776


namespace NUMINAMATH_GPT_find_y_l1137_113738

theorem find_y (x y : ℤ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 :=
sorry

end NUMINAMATH_GPT_find_y_l1137_113738


namespace NUMINAMATH_GPT_proof_C_l1137_113743

variable {a b c : Type} [LinearOrder a] [LinearOrder b] [LinearOrder c]
variable {y : Type}

-- Definitions for parallel and perpendicular relationships
def parallel (x1 x2 : Type) : Prop := sorry
def perp (x1 x2 : Type) : Prop := sorry

theorem proof_C (a b c : Type) [LinearOrder a] [LinearOrder b] [LinearOrder c] (y : Type):
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perp a y ∧ perp b y → parallel a b) :=
by
  sorry

end NUMINAMATH_GPT_proof_C_l1137_113743


namespace NUMINAMATH_GPT_math_problem_l1137_113724

theorem math_problem
  (N O : ℝ)
  (h₁ : 96 / 100 = |(O - 5 * N) / (5 * N)|)
  (h₂ : 5 * N ≠ 0) :
  O = 0.2 * N :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1137_113724


namespace NUMINAMATH_GPT_total_money_made_l1137_113741

-- Define the given conditions.
def total_rooms : ℕ := 260
def single_rooms : ℕ := 64
def single_room_cost : ℕ := 35
def double_room_cost : ℕ := 60

-- Define the number of double rooms.
def double_rooms : ℕ := total_rooms - single_rooms

-- Define the total money made from single and double rooms.
def money_from_single_rooms : ℕ := single_rooms * single_room_cost
def money_from_double_rooms : ℕ := double_rooms * double_room_cost

-- State the theorem we want to prove.
theorem total_money_made : 
  (money_from_single_rooms + money_from_double_rooms) = 14000 :=
  by
    sorry -- Proof is omitted.

end NUMINAMATH_GPT_total_money_made_l1137_113741


namespace NUMINAMATH_GPT_marcus_saves_34_22_l1137_113734

def max_spend : ℝ := 200
def shoe_price : ℝ := 120
def shoe_discount : ℝ := 0.30
def sock_price : ℝ := 25
def sock_discount : ℝ := 0.20
def shirt_price : ℝ := 55
def shirt_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def calc_discounted_price (price discount : ℝ) : ℝ := price * (1 - discount)

def total_cost_before_tax : ℝ :=
  calc_discounted_price shoe_price shoe_discount +
  calc_discounted_price sock_price sock_discount +
  calc_discounted_price shirt_price shirt_discount

def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

def final_cost : ℝ := total_cost_before_tax + sales_tax

def money_saved : ℝ := max_spend - final_cost

theorem marcus_saves_34_22 :
  money_saved = 34.22 :=
by sorry

end NUMINAMATH_GPT_marcus_saves_34_22_l1137_113734


namespace NUMINAMATH_GPT_value_of_a_12_l1137_113759

variable {a : ℕ → ℝ} (h1 : a 6 + a 10 = 20) (h2 : a 4 = 2)

theorem value_of_a_12 : a 12 = 18 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_12_l1137_113759


namespace NUMINAMATH_GPT_TJs_average_time_l1137_113732

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end NUMINAMATH_GPT_TJs_average_time_l1137_113732


namespace NUMINAMATH_GPT_enchilada_taco_cost_l1137_113763

theorem enchilada_taco_cost (e t : ℝ) 
  (h1 : 3 * e + 4 * t = 3.50) 
  (h2 : 4 * e + 3 * t = 3.90) : 
  4 * e + 5 * t = 4.56 := 
sorry

end NUMINAMATH_GPT_enchilada_taco_cost_l1137_113763


namespace NUMINAMATH_GPT_measure_of_theta_l1137_113764

theorem measure_of_theta 
  (ACB FEG DCE DEC : ℝ)
  (h1 : ACB = 10)
  (h2 : FEG = 26)
  (h3 : DCE = 14)
  (h4 : DEC = 33) : θ = 11 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_theta_l1137_113764


namespace NUMINAMATH_GPT_multiple_choice_questions_count_l1137_113737

variable (M F : ℕ)

-- Conditions
def totalQuestions := M + F = 60
def totalStudyTime := 15 * M + 25 * F = 1200

-- Statement to prove
theorem multiple_choice_questions_count (h1 : totalQuestions M F) (h2 : totalStudyTime M F) : M = 30 := by
  sorry

end NUMINAMATH_GPT_multiple_choice_questions_count_l1137_113737
