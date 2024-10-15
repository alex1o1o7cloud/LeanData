import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l132_13221

theorem problem_statement :
  let a := -12
  let b := 45
  let c := -45
  let d := 54
  8 * a + 4 * b + 2 * c + d = 48 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l132_13221


namespace NUMINAMATH_GPT_incorrect_inequality_l132_13283

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_inequality_l132_13283


namespace NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l132_13254

theorem remainder_of_3_pow_2023_mod_7 :
  (3^2023) % 7 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_2023_mod_7_l132_13254


namespace NUMINAMATH_GPT_point_A_lies_on_plane_l132_13258

-- Define the plane equation
def plane (x y z : ℝ) : Prop := 2 * x - y + 2 * z = 7

-- Define the specific point
def point_A : Prop := plane 2 3 3

-- The theorem stating that point A lies on the plane
theorem point_A_lies_on_plane : point_A :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_point_A_lies_on_plane_l132_13258


namespace NUMINAMATH_GPT_identical_remainders_l132_13268

theorem identical_remainders (a : Fin 11 → Fin 11) (h_perm : ∀ n, ∃ m, a m = n) :
  ∃ (i j : Fin 11), i ≠ j ∧ (i * a i) % 11 = (j * a j) % 11 :=
by 
  sorry

end NUMINAMATH_GPT_identical_remainders_l132_13268


namespace NUMINAMATH_GPT_compute_result_l132_13269

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end NUMINAMATH_GPT_compute_result_l132_13269


namespace NUMINAMATH_GPT_scorpion_segments_daily_total_l132_13222

theorem scorpion_segments_daily_total (seg1 : ℕ) (seg2 : ℕ) (additional : ℕ) (total_daily : ℕ) :
  (seg1 = 60) →
  (seg2 = 2 * seg1 * 2) →
  (additional = 10 * 50) →
  (total_daily = seg1 + seg2 + additional) →
  total_daily = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_scorpion_segments_daily_total_l132_13222


namespace NUMINAMATH_GPT_find_natural_number_n_l132_13290

theorem find_natural_number_n (n x y : ℕ) (h1 : n + 195 = x^3) (h2 : n - 274 = y^3) : 
  n = 2002 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_n_l132_13290


namespace NUMINAMATH_GPT_total_pencils_l132_13288

theorem total_pencils (pencils_per_box : ℕ) (friends : ℕ) (total_pencils : ℕ) : 
  pencils_per_box = 7 ∧ friends = 5 → total_pencils = pencils_per_box + friends * pencils_per_box → total_pencils = 42 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_total_pencils_l132_13288


namespace NUMINAMATH_GPT_binary_modulo_eight_l132_13295

theorem binary_modulo_eight : (0b1110101101101 : ℕ) % 8 = 5 := 
by {
  -- This is where the proof would go.
  sorry
}

end NUMINAMATH_GPT_binary_modulo_eight_l132_13295


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l132_13253

theorem geometric_sequence_common_ratio (a_1 a_4 q : ℕ) (h1 : a_1 = 8) (h2 : a_4 = 64) (h3 : a_4 = a_1 * q^3) : q = 2 :=
by {
  -- Given: a_1 = 8
  --        a_4 = 64
  --        a_4 = a_1 * q^3
  -- Prove: q = 2
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l132_13253


namespace NUMINAMATH_GPT_john_overall_profit_l132_13272

-- Definitions based on conditions
def cost_grinder : ℕ := 15000
def cost_mobile : ℕ := 8000
def loss_percentage_grinder : ℚ := 4 / 100
def profit_percentage_mobile : ℚ := 15 / 100

-- Calculations based on the conditions
def loss_amount_grinder := cost_grinder * loss_percentage_grinder
def selling_price_grinder := cost_grinder - loss_amount_grinder
def profit_amount_mobile := cost_mobile * profit_percentage_mobile
def selling_price_mobile := cost_mobile + profit_amount_mobile
def total_cost_price := cost_grinder + cost_mobile
def total_selling_price := selling_price_grinder + selling_price_mobile

-- Overall profit calculation
def overall_profit := total_selling_price - total_cost_price

-- Proof statement to prove the overall profit
theorem john_overall_profit : overall_profit = 600 := by
  sorry

end NUMINAMATH_GPT_john_overall_profit_l132_13272


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l132_13279

-- Define the sets A, B, and C
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

-- The set A ∪ B in terms of Lean
def A_union_B : Set ℝ := A ∪ B

-- State the necessary and sufficient conditions
theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x ∈ A_union_B → x ∈ C) ∧ ¬ (∀ x : ℝ, x ∈ C → x ∈ A_union_B) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l132_13279


namespace NUMINAMATH_GPT_total_area_l132_13291

variable (A : ℝ)

-- Defining the conditions
def first_carpet : Prop := 0.55 * A = 36
def second_carpet : Prop := 0.25 * A = A * 0.25
def third_carpet : Prop := 0.15 * A = 18 + 6
def remaining_floor : Prop := 0.05 * A + 0.55 * A + 0.25 * A + 0.15 * A = A

-- Main theorem to prove the total area
theorem total_area : first_carpet A → second_carpet A → third_carpet A → remaining_floor A → A = 65.45 :=
by
  sorry

end NUMINAMATH_GPT_total_area_l132_13291


namespace NUMINAMATH_GPT_least_three_digit_product_12_l132_13230

-- Problem statement: Find the least three-digit number whose digits multiply to 12
theorem least_three_digit_product_12 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a * b * c = 12 ∧ n = 126) :=
by
  sorry

end NUMINAMATH_GPT_least_three_digit_product_12_l132_13230


namespace NUMINAMATH_GPT_point_in_second_quadrant_l132_13239

def point (x : ℤ) (y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : point (-1) 3 = true := by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l132_13239


namespace NUMINAMATH_GPT_cost_of_adult_ticket_l132_13231

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

end NUMINAMATH_GPT_cost_of_adult_ticket_l132_13231


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l132_13255

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := (y^2 / 4) - (x^2 / 9) = 1

-- Define the standard form of hyperbola asymptotes equations
def asymptotes_eq (x y : ℝ) : Prop := 2 * x + 3 * y = 0 ∨ 2 * x - 3 * y = 0

-- The final proof statement
theorem hyperbola_asymptotes (x y : ℝ) (h : hyperbola_eq x y) : asymptotes_eq x y :=
    sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l132_13255


namespace NUMINAMATH_GPT_computation_result_l132_13246

theorem computation_result :
  2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 :=
by
  sorry

end NUMINAMATH_GPT_computation_result_l132_13246


namespace NUMINAMATH_GPT_correlation_index_l132_13274

-- Define the conditions given in the problem
def height_explains_weight_variation : Prop :=
  ∃ R : ℝ, R^2 = 0.64

-- State the main conjecture (actual proof omitted for simplicity)
theorem correlation_index (R : ℝ) (h : height_explains_weight_variation) : R^2 = 0.64 := by
  sorry

end NUMINAMATH_GPT_correlation_index_l132_13274


namespace NUMINAMATH_GPT_division_addition_problem_l132_13277

-- Define the terms used in the problem
def ten : ℕ := 10
def one_fifth : ℚ := 1 / 5
def six : ℕ := 6

-- Define the math problem
theorem division_addition_problem :
  (ten / one_fifth : ℚ) + six = 56 :=
by sorry

end NUMINAMATH_GPT_division_addition_problem_l132_13277


namespace NUMINAMATH_GPT_proposition_contrapositive_same_truth_value_l132_13247

variable {P : Prop}

theorem proposition_contrapositive_same_truth_value (P : Prop) :
  (P → P) = (¬P → ¬P) := 
sorry

end NUMINAMATH_GPT_proposition_contrapositive_same_truth_value_l132_13247


namespace NUMINAMATH_GPT_Total_Cookies_is_135_l132_13297

-- Define the number of cookies in each pack
def PackA_Cookies : ℕ := 15
def PackB_Cookies : ℕ := 30
def PackC_Cookies : ℕ := 45

-- Define the number of packs bought by Paul and Paula
def Paul_PackA_Count : ℕ := 1
def Paul_PackB_Count : ℕ := 2
def Paula_PackA_Count : ℕ := 1
def Paula_PackC_Count : ℕ := 1

-- Calculate total cookies for Paul
def Paul_Cookies : ℕ := (Paul_PackA_Count * PackA_Cookies) + (Paul_PackB_Count * PackB_Cookies)

-- Calculate total cookies for Paula
def Paula_Cookies : ℕ := (Paula_PackA_Count * PackA_Cookies) + (Paula_PackC_Count * PackC_Cookies)

-- Calculate total cookies for Paul and Paula together
def Total_Cookies : ℕ := Paul_Cookies + Paula_Cookies

theorem Total_Cookies_is_135 : Total_Cookies = 135 := by
  sorry

end NUMINAMATH_GPT_Total_Cookies_is_135_l132_13297


namespace NUMINAMATH_GPT_smallest_number_divisible_by_618_3648_60_l132_13275

theorem smallest_number_divisible_by_618_3648_60 :
  ∃ n : ℕ, (∀ m, (m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0 → m = 1038239) :=
by
  use 1038239
  sorry

end NUMINAMATH_GPT_smallest_number_divisible_by_618_3648_60_l132_13275


namespace NUMINAMATH_GPT_solve_equation_l132_13296

theorem solve_equation (x : ℝ) : x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1 / 2 := 
by {
  sorry -- placeholder for the proof
}

end NUMINAMATH_GPT_solve_equation_l132_13296


namespace NUMINAMATH_GPT_mark_bought_5_pounds_of_apples_l132_13250

noncomputable def cost_of_tomatoes (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) : ℝ :=
  pounds_tomatoes * cost_per_pound_tomato

noncomputable def cost_of_apples (total_spent : ℝ) (cost_of_tomatoes : ℝ) : ℝ :=
  total_spent - cost_of_tomatoes

noncomputable def pounds_of_apples (cost_of_apples : ℝ) (cost_per_pound_apples : ℝ) : ℝ :=
  cost_of_apples / cost_per_pound_apples

theorem mark_bought_5_pounds_of_apples (pounds_tomatoes : ℕ) (cost_per_pound_tomato : ℝ) 
  (total_spent : ℝ) (cost_per_pound_apples : ℝ) :
  pounds_tomatoes = 2 →
  cost_per_pound_tomato = 5 →
  total_spent = 40 →
  cost_per_pound_apples = 6 →
  pounds_of_apples (cost_of_apples total_spent (cost_of_tomatoes pounds_tomatoes cost_per_pound_tomato)) cost_per_pound_apples = 5 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_mark_bought_5_pounds_of_apples_l132_13250


namespace NUMINAMATH_GPT_cube_side_length_l132_13213

theorem cube_side_length (s : ℝ) (h : s^3 = 6 * s^2) : s = 6 := by
  sorry

end NUMINAMATH_GPT_cube_side_length_l132_13213


namespace NUMINAMATH_GPT_field_trip_students_l132_13264

theorem field_trip_students (bus_cost admission_per_student budget : ℕ) (students : ℕ)
  (h1 : bus_cost = 100)
  (h2 : admission_per_student = 10)
  (h3 : budget = 350)
  (total_cost : students * admission_per_student + bus_cost ≤ budget) : 
  students = 25 :=
by
  sorry

end NUMINAMATH_GPT_field_trip_students_l132_13264


namespace NUMINAMATH_GPT_relation_y₁_y₂_y₃_l132_13201

def parabola (x : ℝ) : ℝ := - x^2 - 2 * x + 2
noncomputable def y₁ : ℝ := parabola (-2)
noncomputable def y₂ : ℝ := parabola (1)
noncomputable def y₃ : ℝ := parabola (2)

theorem relation_y₁_y₂_y₃ : y₁ > y₂ ∧ y₂ > y₃ := by
  have h₁ : y₁ = 2 := by
    unfold y₁ parabola
    norm_num
    
  have h₂ : y₂ = -1 := by
    unfold y₂ parabola
    norm_num
    
  have h₃ : y₃ = -6 := by
    unfold y₃ parabola
    norm_num
    
  rw [h₁, h₂, h₃]
  exact ⟨by norm_num, by norm_num⟩

end NUMINAMATH_GPT_relation_y₁_y₂_y₃_l132_13201


namespace NUMINAMATH_GPT_social_gathering_married_men_fraction_l132_13203

theorem social_gathering_married_men_fraction {W : ℝ} {MW : ℝ} {MM : ℝ} 
  (hW_pos : 0 < W)
  (hMW_def : MW = W * (3/7))
  (hMM_def : MM = W - MW)
  (h_total_people : 2 * MM + MW = 11) :
  (MM / 11) = 4/11 :=
by {
  sorry
}

end NUMINAMATH_GPT_social_gathering_married_men_fraction_l132_13203


namespace NUMINAMATH_GPT_average_output_l132_13260

theorem average_output (t1 t2: ℝ) (cogs1 cogs2 : ℕ) (h1 : t1 = cogs1 / 36) (h2 : t2 = cogs2 / 60) (h_sum_cogs : cogs1 = 60) (h_sum_more_cogs : cogs2 = 60) (h_sum_time : t1 + t2 = 60 / 36 + 60 / 60) : 
  (cogs1 + cogs2) / (t1 + t2) = 45 := by
  sorry

end NUMINAMATH_GPT_average_output_l132_13260


namespace NUMINAMATH_GPT_fraction_evaluation_l132_13294

theorem fraction_evaluation :
  (1/5 - 1/7) / (3/8 + 2/9) = 144/1505 := 
  by 
    sorry

end NUMINAMATH_GPT_fraction_evaluation_l132_13294


namespace NUMINAMATH_GPT_allocation_schemes_l132_13224

theorem allocation_schemes (students factories: ℕ) (has_factory_a: Prop) (A_must_have_students: has_factory_a): students = 3 → factories = 4 → has_factory_a → (∃ n: ℕ, n = 4^3 - 3^3 ∧ n = 37) :=
by try { sorry }

end NUMINAMATH_GPT_allocation_schemes_l132_13224


namespace NUMINAMATH_GPT_barbara_typing_time_l132_13215

theorem barbara_typing_time :
  let original_speed := 212
  let reduction := 40
  let num_words := 3440
  let reduced_speed := original_speed - reduction
  let time := num_words / reduced_speed
  time = 20 :=
by
  sorry

end NUMINAMATH_GPT_barbara_typing_time_l132_13215


namespace NUMINAMATH_GPT_find_f1_l132_13257

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x
def odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem find_f1 (f : ℝ → ℝ)
  (h_periodic : periodic f 2)
  (h_odd : odd f) :
  f 1 = 0 :=
sorry

end NUMINAMATH_GPT_find_f1_l132_13257


namespace NUMINAMATH_GPT_problem1_problem2_l132_13216

-- First proof problem
theorem problem1 : - (2^2 : ℚ) + (2/3) * ((1 - 1/3) ^ 2) = -100/27 :=
by sorry

-- Second proof problem
theorem problem2 : (8 : ℚ) ^ (1 / 3) - |2 - (3 : ℚ) ^ (1 / 2)| - (3 : ℚ) ^ (1 / 2) = 0 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l132_13216


namespace NUMINAMATH_GPT_least_perimeter_of_triangle_l132_13234

theorem least_perimeter_of_triangle (a b : ℕ) (a_eq : a = 33) (b_eq : b = 42) (c : ℕ) (h1 : c + a > b) (h2 : c + b > a) (h3 : a + b > c) : a + b + c = 85 :=
sorry

end NUMINAMATH_GPT_least_perimeter_of_triangle_l132_13234


namespace NUMINAMATH_GPT_simon_students_l132_13252

theorem simon_students (S L : ℕ) (h1 : S = 4 * L) (h2 : S + L = 2500) : S = 2000 :=
by {
  sorry
}

end NUMINAMATH_GPT_simon_students_l132_13252


namespace NUMINAMATH_GPT_triangle_perimeter_is_26_l132_13266

-- Define the lengths of the medians as given conditions
def median1 := 3
def median2 := 4
def median3 := 6

-- Define the perimeter of the triangle
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove that the perimeter is 26 cm
theorem triangle_perimeter_is_26 :
  perimeter (2 * median1) (2 * median2) (2 * median3) = 26 :=
by
  -- Calculation follows directly from the definition
  sorry

end NUMINAMATH_GPT_triangle_perimeter_is_26_l132_13266


namespace NUMINAMATH_GPT_intersection_eq_l132_13278

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def M : Set ℝ := { x | -1/2 < x ∧ x < 1/2 }
def N : Set ℝ := { x | 0 ≤ x ∧ x * x ≤ x }

theorem intersection_eq :
  M ∩ N = { x | 0 ≤ x ∧ x < 1/2 } := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l132_13278


namespace NUMINAMATH_GPT_math_problem_l132_13204

theorem math_problem :
  8 / 4 - 3^2 + 4 * 2 + (Nat.factorial 5) = 121 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l132_13204


namespace NUMINAMATH_GPT_tilly_star_count_l132_13219

theorem tilly_star_count (stars_east : ℕ) (stars_west : ℕ) (total_stars : ℕ) 
  (h1 : stars_east = 120)
  (h2 : stars_west = 6 * stars_east)
  (h3 : total_stars = stars_east + stars_west) :
  total_stars = 840 :=
sorry

end NUMINAMATH_GPT_tilly_star_count_l132_13219


namespace NUMINAMATH_GPT_farmer_initial_plan_days_l132_13212

def initialDaysPlan
    (daily_hectares : ℕ)
    (increased_productivity : ℕ)
    (hectares_ploughed_first_two_days : ℕ)
    (hectares_remaining : ℕ)
    (days_ahead_schedule : ℕ)
    (total_hectares : ℕ)
    (days_actual : ℕ) : ℕ :=
  days_actual + days_ahead_schedule

theorem farmer_initial_plan_days : 
  ∀ (x days_ahead_schedule : ℕ) 
    (daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual : ℕ),
  daily_hectares = 120 →
  increased_productivity = daily_hectares + daily_hectares / 4 →
  hectares_ploughed_first_two_days = 2 * daily_hectares →
  total_hectares = 1440 →
  days_ahead_schedule = 2 →
  days_actual = 10 →
  hectares_remaining = total_hectares - hectares_ploughed_first_two_days →
  hectares_remaining = increased_productivity * (days_actual - 2) →
  x = 12 :=
by
  intros x days_ahead_schedule daily_hectares hectares_ploughed_first_two_days increased_productivity hectares_remaining total_hectares days_actual
  intros h_daily_hectares h_increased_productivity h_hectares_ploughed_first_two_days h_total_hectares h_days_ahead_schedule h_days_actual h_hectares_remaining h_hectares_ploughed
  sorry

end NUMINAMATH_GPT_farmer_initial_plan_days_l132_13212


namespace NUMINAMATH_GPT_patio_tiles_l132_13229

theorem patio_tiles (r c : ℕ) (h1 : r * c = 48) (h2 : (r + 4) * (c - 2) = 48) : r = 6 :=
sorry

end NUMINAMATH_GPT_patio_tiles_l132_13229


namespace NUMINAMATH_GPT_download_time_l132_13211

def first_segment_size : ℝ := 30
def first_segment_rate : ℝ := 5
def second_segment_size : ℝ := 40
def second_segment_rate1 : ℝ := 10
def second_segment_rate2 : ℝ := 2
def third_segment_size : ℝ := 20
def third_segment_rate1 : ℝ := 8
def third_segment_rate2 : ℝ := 4

theorem download_time :
  let time_first := first_segment_size / first_segment_rate
  let time_second := (10 / second_segment_rate1) + (10 / second_segment_rate2) + (10 / second_segment_rate1) + (10 / second_segment_rate2)
  let time_third := (10 / third_segment_rate1) + (10 / third_segment_rate2)
  time_first + time_second + time_third = 21.75 :=
by
  sorry

end NUMINAMATH_GPT_download_time_l132_13211


namespace NUMINAMATH_GPT_initial_average_age_is_16_l132_13225

-- Given conditions
variable (N : ℕ) (newPersons : ℕ) (avgNewPersonsAge : ℝ) (totalPersonsAfter : ℕ) (avgAgeAfter : ℝ)
variable (initial_avg_age : ℝ) -- This represents the initial average age (A) we need to prove

-- The specific values from the problem
def N_value : ℕ := 20
def newPersons_value : ℕ := 20
def avgNewPersonsAge_value : ℝ := 15
def totalPersonsAfter_value : ℕ := 40
def avgAgeAfter_value : ℝ := 15.5

-- Theorem statement to prove that the initial average age is 16 years
theorem initial_average_age_is_16 (h1 : N = N_value) (h2 : newPersons = newPersons_value) 
  (h3 : avgNewPersonsAge = avgNewPersonsAge_value) (h4 : totalPersonsAfter = totalPersonsAfter_value) 
  (h5 : avgAgeAfter = avgAgeAfter_value) : initial_avg_age = 16 := by
  sorry

end NUMINAMATH_GPT_initial_average_age_is_16_l132_13225


namespace NUMINAMATH_GPT_boots_ratio_l132_13218

noncomputable def problem_statement : Prop :=
  let total_money : ℝ := 50
  let cost_toilet_paper : ℝ := 12
  let cost_groceries : ℝ := 2 * cost_toilet_paper
  let remaining_after_groceries : ℝ := total_money - cost_toilet_paper - cost_groceries
  let extra_money_per_person : ℝ := 35
  let total_extra_money : ℝ := 2 * extra_money_per_person
  let total_cost_boots : ℝ := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots : ℝ := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3

theorem boots_ratio (total_money : ℝ) (cost_toilet_paper : ℝ) (extra_money_per_person : ℝ) : 
  let cost_groceries := 2 * cost_toilet_paper
  let remaining_after_groceries := total_money - cost_toilet_paper - cost_groceries
  let total_extra_money := 2 * extra_money_per_person
  let total_cost_boots := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3 :=
by
  sorry

end NUMINAMATH_GPT_boots_ratio_l132_13218


namespace NUMINAMATH_GPT_smallest_solution_floor_eq_l132_13298

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_floor_eq_l132_13298


namespace NUMINAMATH_GPT_permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l132_13206

open Finset

def digits : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def permutations_no_repetition : ℤ :=
  (digits.card.factorial) / ((digits.card - 4).factorial)

noncomputable def four_digit_numbers_no_repetition : ℤ :=
  9 * ((digits.card - 1).factorial / ((digits.card - 1 - 3).factorial))

noncomputable def even_four_digit_numbers_gt_3000_no_repetition : ℤ :=
  784 + 1008

theorem permutations_count_5040 : permutations_no_repetition = 5040 := by
  sorry

theorem four_digit_numbers_count_4356 : four_digit_numbers_no_repetition = 4356 := by
  sorry

theorem even_four_digit_numbers_count_1792 : even_four_digit_numbers_gt_3000_no_repetition = 1792 := by
  sorry

end NUMINAMATH_GPT_permutations_count_5040_four_digit_numbers_count_4356_even_four_digit_numbers_count_1792_l132_13206


namespace NUMINAMATH_GPT_problem1_problem2_l132_13265

variable (x y : ℝ)

theorem problem1 :
  x^4 * x^3 * x - (x^4)^2 + (-2 * x)^3 * x^5 = -8 * x^8 :=
by sorry

theorem problem2 :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l132_13265


namespace NUMINAMATH_GPT_grace_wait_time_l132_13261

variable (hose1_rate : ℕ) (hose2_rate : ℕ) (pool_capacity : ℕ) (time_after_second_hose : ℕ)
variable (h : ℕ)

theorem grace_wait_time 
  (h1 : hose1_rate = 50)
  (h2 : hose2_rate = 70)
  (h3 : pool_capacity = 390)
  (h4 : time_after_second_hose = 2) : 
  50 * h + (50 + 70) * 2 = 390 → h = 3 :=
by
  sorry

end NUMINAMATH_GPT_grace_wait_time_l132_13261


namespace NUMINAMATH_GPT_arithmetic_sequence_7th_term_l132_13214

theorem arithmetic_sequence_7th_term 
  (a d : ℝ)
  (n : ℕ)
  (h1 : 5 * a + 10 * d = 34)
  (h2 : 5 * a + 5 * (n - 1) * d = 146)
  (h3 : (n / 2 : ℝ) * (2 * a + (n - 1) * d) = 234) :
  a + 6 * d = 19 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_7th_term_l132_13214


namespace NUMINAMATH_GPT_imaginary_part_of_z_l132_13262

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := i / (i - 1)

theorem imaginary_part_of_z : z.im = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l132_13262


namespace NUMINAMATH_GPT_count_integers_satisfying_conditions_l132_13271

theorem count_integers_satisfying_conditions :
  (∃ (s : Finset ℤ), s.card = 3 ∧
  ∀ x : ℤ, x ∈ s ↔ (-5 ≤ x ∧ x ≤ -3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_count_integers_satisfying_conditions_l132_13271


namespace NUMINAMATH_GPT_solve_abs_inequality_l132_13293

theorem solve_abs_inequality (x : ℝ) : (|x + 3| + |x - 4| < 8) ↔ (4 ≤ x ∧ x < 4.5) := sorry

end NUMINAMATH_GPT_solve_abs_inequality_l132_13293


namespace NUMINAMATH_GPT_sports_club_problem_l132_13241

theorem sports_club_problem (N B T Neither X : ℕ) (hN : N = 42) (hB : B = 20) (hT : T = 23) (hNeither : Neither = 6) :
  (B + T - X + Neither = N) → X = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sports_club_problem_l132_13241


namespace NUMINAMATH_GPT_collinear_points_inverse_sum_half_l132_13217

theorem collinear_points_inverse_sum_half (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
    (collinear : (a - 2) * (b - 2) - (-2) * a = 0) : 
    1 / a + 1 / b = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_collinear_points_inverse_sum_half_l132_13217


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l132_13284

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_variance : (1/5) * ((a 1 - (a 3)) ^ 2 + (a 2 - (a 3)) ^ 2 + (a 3 - (a 3)) ^ 2 + (a 4 - (a 3)) ^ 2 + (a 5 - (a 3)) ^ 2) = 8) :
  d = 2 ∨ d = -2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l132_13284


namespace NUMINAMATH_GPT_billing_error_l132_13256

theorem billing_error (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) 
    (h : 100 * y + x - (100 * x + y) = 2970) : y - x = 30 ∧ 10 ≤ x ∧ x ≤ 69 ∧ 40 ≤ y ∧ y ≤ 99 := 
by
  sorry

end NUMINAMATH_GPT_billing_error_l132_13256


namespace NUMINAMATH_GPT_binomial_12_6_eq_924_l132_13208

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end NUMINAMATH_GPT_binomial_12_6_eq_924_l132_13208


namespace NUMINAMATH_GPT_soccer_tournament_matches_l132_13237

theorem soccer_tournament_matches (x : ℕ) (h : 1 ≤ x) : (1 / 2 : ℝ) * x * (x - 1) = 45 := sorry

end NUMINAMATH_GPT_soccer_tournament_matches_l132_13237


namespace NUMINAMATH_GPT_speed_of_mrs_a_l132_13223

theorem speed_of_mrs_a
  (distance_between : ℝ)
  (speed_mr_a : ℝ)
  (speed_bee : ℝ)
  (distance_bee_travelled : ℝ)
  (time_bee : ℝ)
  (remaining_distance : ℝ)
  (speed_mrs_a : ℝ) :
  distance_between = 120 ∧
  speed_mr_a = 30 ∧
  speed_bee = 60 ∧
  distance_bee_travelled = 180 ∧
  time_bee = distance_bee_travelled / speed_bee ∧
  remaining_distance = distance_between - (speed_mr_a * time_bee) ∧
  speed_mrs_a = remaining_distance / time_bee →
  speed_mrs_a = 10 := by
  sorry

end NUMINAMATH_GPT_speed_of_mrs_a_l132_13223


namespace NUMINAMATH_GPT_smallest_nonfactor_product_of_48_l132_13227

noncomputable def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem smallest_nonfactor_product_of_48
  (m n : ℕ)
  (h1 : m ≠ n)
  (h2 : is_factor_of m 48)
  (h3 : is_factor_of n 48)
  (h4 : ¬is_factor_of (m * n) 48) :
  m * n = 18 :=
sorry

end NUMINAMATH_GPT_smallest_nonfactor_product_of_48_l132_13227


namespace NUMINAMATH_GPT_unique_x_value_l132_13232

theorem unique_x_value (x : ℝ) (h : x ≠ 0) (h_sqrt : Real.sqrt (5 * x / 7) = x) : x = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_unique_x_value_l132_13232


namespace NUMINAMATH_GPT_f_6_plus_f_neg3_l132_13209

noncomputable def f : ℝ → ℝ := sorry

-- f is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- f is increasing in the interval [3,6]
def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) := a ≤ b → ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

-- Define the given conditions
axiom h1 : is_odd_function f
axiom h2 : is_increasing_interval f 3 6
axiom h3 : f 6 = 8
axiom h4 : f 3 = -1

-- The statement to be proved
theorem f_6_plus_f_neg3 : f 6 + f (-3) = 9 :=
by
  sorry

end NUMINAMATH_GPT_f_6_plus_f_neg3_l132_13209


namespace NUMINAMATH_GPT_total_trout_caught_l132_13210

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_trout_caught_l132_13210


namespace NUMINAMATH_GPT_mary_animals_count_l132_13228

def initial_lambs := 18
def initial_alpacas := 5
def initial_baby_lambs := 7 * 4
def traded_lambs := 8
def traded_alpacas := 2
def received_goats := 3
def received_chickens := 10
def chickens_traded_for_alpacas := received_chickens / 2
def additional_lambs := 20
def additional_alpacas := 6

noncomputable def final_lambs := initial_lambs + initial_baby_lambs - traded_lambs + additional_lambs
noncomputable def final_alpacas := initial_alpacas - traded_alpacas + 2 + additional_alpacas
noncomputable def final_goats := received_goats
noncomputable def final_chickens := received_chickens - chickens_traded_for_alpacas

theorem mary_animals_count :
  final_lambs = 58 ∧ 
  final_alpacas = 11 ∧ 
  final_goats = 3 ∧ 
  final_chickens = 5 :=
by 
  sorry

end NUMINAMATH_GPT_mary_animals_count_l132_13228


namespace NUMINAMATH_GPT_snail_kite_first_day_snails_l132_13205

theorem snail_kite_first_day_snails (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 35) : 
  x = 3 :=
sorry

end NUMINAMATH_GPT_snail_kite_first_day_snails_l132_13205


namespace NUMINAMATH_GPT_first_discount_percentage_l132_13235

theorem first_discount_percentage (original_price final_price : ℝ)
  (first_discount second_discount : ℝ) (h_orig : original_price = 200)
  (h_final : final_price = 144) (h_second_disc : second_discount = 0.20) :
  first_discount = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l132_13235


namespace NUMINAMATH_GPT_sum_of_digits_triangular_array_l132_13245

theorem sum_of_digits_triangular_array (N : ℕ) (h : N * (N + 1) / 2 = 5050) : 
  Nat.digits 10 N = [1, 0, 0] := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_triangular_array_l132_13245


namespace NUMINAMATH_GPT_contrapositive_equivalence_l132_13200

-- Definitions based on the conditions
variables (R S : Prop)

-- Statement of the proof
theorem contrapositive_equivalence (h : ¬R → S) : ¬S → R := 
sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l132_13200


namespace NUMINAMATH_GPT_smallest_positive_sum_l132_13280

structure ArithmeticSequence :=
  (a_n : ℕ → ℤ)  -- The sequence is an integer sequence
  (d : ℤ)        -- The common difference of the sequence

def sum_of_first_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a_n 1 + seq.a_n n)) / 2  -- Sum of first n terms

def condition (seq : ArithmeticSequence) : Prop :=
  (seq.a_n 11 < -1 * seq.a_n 10)

theorem smallest_positive_sum (seq : ArithmeticSequence) (H : condition seq) :
  ∃ n, sum_of_first_n seq n > 0 ∧ ∀ m < n, sum_of_first_n seq m ≤ 0 → n = 19 :=
sorry

end NUMINAMATH_GPT_smallest_positive_sum_l132_13280


namespace NUMINAMATH_GPT_find_c_l132_13263

noncomputable def func_condition (f : ℝ → ℝ) (c : ℝ) :=
  ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)

theorem find_c :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), func_condition f c → (c = 1 ∨ c = -1) :=
sorry

end NUMINAMATH_GPT_find_c_l132_13263


namespace NUMINAMATH_GPT_range_of_a_l132_13251

theorem range_of_a (b c a : ℝ) (h_intersect : ∀ x : ℝ, 
  (x ^ 2 - 2 * b * x + b ^ 2 + c = 1 - x → x = b )) 
  (h_vertex : c = a * b ^ 2) :
  a ≥ (-1 / 5) ∧ a ≠ 0 := 
by 
-- Proof skipped
sorry

end NUMINAMATH_GPT_range_of_a_l132_13251


namespace NUMINAMATH_GPT_scientific_notation_of_300670_l132_13259

theorem scientific_notation_of_300670 : ∃ a : ℝ, ∃ n : ℤ, (1 ≤ |a| ∧ |a| < 10) ∧ 300670 = a * 10^n ∧ a = 3.0067 ∧ n = 5 :=
  by
    sorry

end NUMINAMATH_GPT_scientific_notation_of_300670_l132_13259


namespace NUMINAMATH_GPT_sample_size_is_40_l132_13282

theorem sample_size_is_40 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 240) (h2 : sample_students = 40) : sample_students = 40 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_is_40_l132_13282


namespace NUMINAMATH_GPT_find_m_n_sum_product_l132_13292

noncomputable def sum_product_of_roots (m n : ℝ) : Prop :=
  (m^2 - 4*m - 12 = 0) ∧ (n^2 - 4*n - 12 = 0) 

theorem find_m_n_sum_product (m n : ℝ) (h : sum_product_of_roots m n) :
  m + n + m * n = -8 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_n_sum_product_l132_13292


namespace NUMINAMATH_GPT_combined_value_of_cookies_sold_l132_13273

theorem combined_value_of_cookies_sold:
  ∀ (total_boxes : ℝ) (plain_boxes : ℝ) (price_plain : ℝ) (price_choco : ℝ),
    total_boxes = 1585 →
    plain_boxes = 793.125 →
    price_plain = 0.75 →
    price_choco = 1.25 →
    (plain_boxes * price_plain + (total_boxes - plain_boxes) * price_choco) = 1584.6875 :=
by
  intros total_boxes plain_boxes price_plain price_choco
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_combined_value_of_cookies_sold_l132_13273


namespace NUMINAMATH_GPT_algebraic_expression_value_l132_13202

theorem algebraic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 7 = -6 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l132_13202


namespace NUMINAMATH_GPT_nandan_gain_l132_13299

theorem nandan_gain (x t : ℝ) (nandan_gain krishan_gain total_gain : ℝ)
  (h1 : krishan_gain = 12 * x * t)
  (h2 : nandan_gain = x * t)
  (h3 : total_gain = nandan_gain + krishan_gain)
  (h4 : total_gain = 78000) :
  nandan_gain = 6000 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_nandan_gain_l132_13299


namespace NUMINAMATH_GPT_total_time_to_complete_work_l132_13207

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_total_time_to_complete_work_l132_13207


namespace NUMINAMATH_GPT_positive_operation_l132_13267

def operation_a := 1 + (-2)
def operation_b := 1 - (-2)
def operation_c := 1 * (-2)
def operation_d := 1 / (-2)

theorem positive_operation : operation_b > 0 ∧ 
  (operation_a <= 0) ∧ (operation_c <= 0) ∧ (operation_d <= 0) := by
  sorry

end NUMINAMATH_GPT_positive_operation_l132_13267


namespace NUMINAMATH_GPT_number_of_girls_calculation_l132_13242

theorem number_of_girls_calculation : 
  ∀ (number_of_boys number_of_girls total_children : ℕ), 
  number_of_boys = 27 → total_children = 62 → number_of_girls = total_children - number_of_boys → number_of_girls = 35 :=
by
  intros number_of_boys number_of_girls total_children 
  intros h_boys h_total h_calc
  rw [h_boys, h_total] at h_calc
  simp at h_calc
  exact h_calc

end NUMINAMATH_GPT_number_of_girls_calculation_l132_13242


namespace NUMINAMATH_GPT_brokerage_percentage_l132_13270

theorem brokerage_percentage (cash_realized amount_before : ℝ) (h1 : cash_realized = 105.25) (h2 : amount_before = 105) :
  |((amount_before - cash_realized) / amount_before) * 100| = 0.2381 := by
sorry

end NUMINAMATH_GPT_brokerage_percentage_l132_13270


namespace NUMINAMATH_GPT_pair_a_n_uniq_l132_13240

theorem pair_a_n_uniq (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_eq : 3^n = a^2 - 16) : a = 5 ∧ n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_pair_a_n_uniq_l132_13240


namespace NUMINAMATH_GPT_find_exponent_l132_13286

theorem find_exponent (y : ℕ) (b : ℕ) (h_b : b = 2)
  (h : 1 / 8 * 2 ^ 40 = b ^ y) : y = 37 :=
by
  sorry

end NUMINAMATH_GPT_find_exponent_l132_13286


namespace NUMINAMATH_GPT_soccer_league_total_games_l132_13249

theorem soccer_league_total_games :
  let teams := 20
  let regular_games_per_team := 19 * 3
  let total_regular_games := (regular_games_per_team * teams) / 2
  let promotional_games_per_team := 3
  let total_promotional_games := promotional_games_per_team * teams
  let total_games := total_regular_games + total_promotional_games
  total_games = 1200 :=
by
  sorry

end NUMINAMATH_GPT_soccer_league_total_games_l132_13249


namespace NUMINAMATH_GPT_AJHSMETL_19892_reappears_on_line_40_l132_13276
-- Import the entire Mathlib library

-- Define the conditions
def cycleLengthLetters : ℕ := 8
def cycleLengthDigits : ℕ := 5
def lcm_cycles : ℕ := Nat.lcm cycleLengthLetters cycleLengthDigits

-- Problem statement with proof to be filled in later
theorem AJHSMETL_19892_reappears_on_line_40 :
  lcm_cycles = 40 := 
by
  sorry

end NUMINAMATH_GPT_AJHSMETL_19892_reappears_on_line_40_l132_13276


namespace NUMINAMATH_GPT_find_m_l132_13287

-- We define the universal set U, the set A with an unknown m, and the complement of A in U.
def U : Set ℕ := {1, 2, 3}
def A (m : ℕ) : Set ℕ := {1, m}
def complement_U_A (m : ℕ) : Set ℕ := U \ A m

-- The main theorem where we need to prove m = 3 given the conditions.
theorem find_m (m : ℕ) (hU : U = {1, 2, 3})
  (hA : ∀ m, A m = {1, m})
  (h_complement : complement_U_A m = {2}) : m = 3 := sorry

end NUMINAMATH_GPT_find_m_l132_13287


namespace NUMINAMATH_GPT_odd_primes_pq_division_l132_13243

theorem odd_primes_pq_division (p q : ℕ) (m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(hp_odd : ¬Even p) (hq_odd : ¬Even q) (hp_gt_hq : p > q) (hm_pos : 0 < m) : ¬(p * q ∣ m ^ (p - q) + 1) :=
by 
  sorry

end NUMINAMATH_GPT_odd_primes_pq_division_l132_13243


namespace NUMINAMATH_GPT_oliver_earnings_l132_13285

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end NUMINAMATH_GPT_oliver_earnings_l132_13285


namespace NUMINAMATH_GPT_savings_percentage_l132_13233

theorem savings_percentage
  (S : ℝ)
  (last_year_saved : ℝ := 0.06 * S)
  (this_year_salary : ℝ := 1.10 * S)
  (this_year_saved : ℝ := 0.10 * this_year_salary)
  (ratio := this_year_saved / last_year_saved * 100):
  ratio = 183.33 := 
sorry

end NUMINAMATH_GPT_savings_percentage_l132_13233


namespace NUMINAMATH_GPT_value_of_knife_l132_13238

/-- Two siblings sold their flock of sheep. Each sheep was sold for as many florins as 
the number of sheep originally in the flock. They divided the revenue by giving out 
10 florins at a time. First, the elder brother took 10 florins, then the younger brother, 
then the elder again, and so on. In the end, the younger brother received less than 10 florins, 
so the elder brother gave him his knife, making their earnings equal. 
Prove that the value of the knife in florins is 2. -/
theorem value_of_knife (n : ℕ) (k m : ℕ) (h1 : n^2 = 20 * k + 10 + m) (h2 : 1 ≤ m ∧ m ≤ 9) : 
  (∃ b : ℕ, 10 - b = m + b ∧ b = 2) :=
by
  sorry

end NUMINAMATH_GPT_value_of_knife_l132_13238


namespace NUMINAMATH_GPT_sqrt3_mul_sqrt12_eq_6_l132_13220

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sqrt3_mul_sqrt12_eq_6_l132_13220


namespace NUMINAMATH_GPT_solution_to_prime_equation_l132_13236

theorem solution_to_prime_equation (x y : ℕ) (p : ℕ) (h1 : Nat.Prime p) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 = p * (xy + p) ↔ (x = 8 ∧ y = 1 ∧ p = 19) ∨ (x = 1 ∧ y = 8 ∧ p = 19) ∨ 
              (x = 7 ∧ y = 2 ∧ p = 13) ∨ (x = 2 ∧ y = 7 ∧ p = 13) ∨ 
              (x = 5 ∧ y = 4 ∧ p = 7) ∨ (x = 4 ∧ y = 5 ∧ p = 7) := sorry

end NUMINAMATH_GPT_solution_to_prime_equation_l132_13236


namespace NUMINAMATH_GPT_number_of_t_in_T_such_that_f_t_mod_8_eq_0_l132_13248

def f (x : ℤ) : ℤ := x^3 + 2 * x^2 + 3 * x + 4

def T := { n : ℤ | 0 ≤ n ∧ n ≤ 50 }

theorem number_of_t_in_T_such_that_f_t_mod_8_eq_0 : 
  (∃ t ∈ T, f t % 8 = 0) = false := sorry

end NUMINAMATH_GPT_number_of_t_in_T_such_that_f_t_mod_8_eq_0_l132_13248


namespace NUMINAMATH_GPT_installation_cost_l132_13289

-- Definitions
variables (LP : ℝ) (P : ℝ := 16500) (D : ℝ := 0.2) (T : ℝ := 125) (SP : ℝ := 23100) (I : ℝ)

-- Conditions
def purchase_price := P = (1 - D) * LP
def selling_price := SP = 1.1 * LP
def total_cost := P + T + I = SP

-- Proof Statement
theorem installation_cost : I = 6350 :=
  by
    -- sorry is used to skip the proof
    sorry

end NUMINAMATH_GPT_installation_cost_l132_13289


namespace NUMINAMATH_GPT_perimeter_C_correct_l132_13244

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end NUMINAMATH_GPT_perimeter_C_correct_l132_13244


namespace NUMINAMATH_GPT_chime_2203_occurs_on_March_19_l132_13226

-- Define the initial conditions: chime patterns
def chimes_at_half_hour : Nat := 1
def chimes_at_hour (h : Nat) : Nat := if h = 12 then 12 else h % 12

-- Define the start time and the question parameters
def start_time_hours : Nat := 10
def start_time_minutes : Nat := 45
def start_day : Nat := 26 -- Assume February 26 as starting point, to facilitate day count accurately
def target_chime : Nat := 2203

-- Define the date calculation function (based on given solution steps)
noncomputable def calculate_chime_date (start_day : Nat) : Nat := sorry

-- The goal is to prove calculate_chime_date with given start conditions equals 19 (March 19th is the 19th day after the base day assumption of March 0)
theorem chime_2203_occurs_on_March_19 :
  calculate_chime_date start_day = 19 :=
sorry

end NUMINAMATH_GPT_chime_2203_occurs_on_March_19_l132_13226


namespace NUMINAMATH_GPT_find_a_from_quadratic_inequality_l132_13281

theorem find_a_from_quadratic_inequality :
  ∀ (a : ℝ), (∀ x : ℝ, (x > - (1 / 2)) ∧ (x < 1 / 3) → a * x^2 - 2 * x + 2 > 0) → a = -12 :=
by
  intros a h
  have h1 := h (-1 / 2)
  have h2 := h (1 / 3)
  sorry

end NUMINAMATH_GPT_find_a_from_quadratic_inequality_l132_13281
