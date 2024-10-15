import Mathlib

namespace NUMINAMATH_GPT_min_value_quadratic_l2124_212413

theorem min_value_quadratic : 
  ∀ x : ℝ, (4 * x^2 - 12 * x + 9) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l2124_212413


namespace NUMINAMATH_GPT_dylan_speed_constant_l2124_212486

theorem dylan_speed_constant (d t s : ℝ) (h1 : d = 1250) (h2 : t = 25) (h3 : s = d / t) : s = 50 := 
by 
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_dylan_speed_constant_l2124_212486


namespace NUMINAMATH_GPT_negation_equivalence_l2124_212490

theorem negation_equivalence (x : ℝ) : ¬(∀ x, x^2 - x + 2 ≥ 0) ↔ ∃ x, x^2 - x + 2 < 0 :=
sorry

end NUMINAMATH_GPT_negation_equivalence_l2124_212490


namespace NUMINAMATH_GPT_identify_counterfeit_coin_correct_l2124_212439

noncomputable def identify_counterfeit_coin (coins : Fin 8 → ℝ) : ℕ :=
  sorry

theorem identify_counterfeit_coin_correct (coins : Fin 8 → ℝ) (h_fake : 
  ∃ i : Fin 8, ∀ j : Fin 8, j ≠ i → coins i > coins j) : 
  ∃ i : Fin 8, identify_counterfeit_coin coins = i ∧ ∀ j : Fin 8, j ≠ i → coins i > coins j :=
by
  sorry

end NUMINAMATH_GPT_identify_counterfeit_coin_correct_l2124_212439


namespace NUMINAMATH_GPT_sin_alpha_value_l2124_212447

theorem sin_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < Real.pi)
  (h₂ : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_sin_alpha_value_l2124_212447


namespace NUMINAMATH_GPT_find_a_l2124_212436

theorem find_a (a : ℝ) (α : ℝ) (h1 : ∃ (y : ℝ), (a, y) = (a, -2))
(h2 : Real.tan (π + α) = 1 / 3) : a = -6 :=
sorry

end NUMINAMATH_GPT_find_a_l2124_212436


namespace NUMINAMATH_GPT_calculate_inverse_y3_minus_y_l2124_212445

theorem calculate_inverse_y3_minus_y
  (i : ℂ) (y : ℂ)
  (h_i : i = Complex.I)
  (h_y : y = (1 + i * Real.sqrt 3) / 2) :
  (1 / (y^3 - y)) = -1/2 + i * (Real.sqrt 3) / 6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_inverse_y3_minus_y_l2124_212445


namespace NUMINAMATH_GPT_inequality_holds_l2124_212492

theorem inequality_holds (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by 
  sorry

end NUMINAMATH_GPT_inequality_holds_l2124_212492


namespace NUMINAMATH_GPT_coplanar_AD_eq_linear_combination_l2124_212425

-- Define the points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨4, 1, 3⟩
def B : Point3D := ⟨2, 3, 1⟩
def C : Point3D := ⟨3, 7, -5⟩
def D : Point3D := ⟨11, -1, 3⟩

-- Define the vectors
def vector (P Q : Point3D) : Point3D := ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def AB := vector A B
def AC := vector A C
def AD := vector A D

-- Coplanar definition: AD = λ AB + μ AC
theorem coplanar_AD_eq_linear_combination (lambda mu : ℝ) :
  AD = ⟨lambda * 2 + mu * (-1), lambda * (-2) + mu * 6, lambda * (-2) + mu * (-8)⟩ :=
sorry

end NUMINAMATH_GPT_coplanar_AD_eq_linear_combination_l2124_212425


namespace NUMINAMATH_GPT_system_of_equations_correct_l2124_212479

def question_statement (x y : ℕ) : Prop :=
  x + y = 12 ∧ 6 * x = 3 * 4 * y

theorem system_of_equations_correct
  (x y : ℕ)
  (h1 : x + y = 12)
  (h2 : 6 * x = 3 * 4 * y)
: question_statement x y :=
by
  unfold question_statement
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_system_of_equations_correct_l2124_212479


namespace NUMINAMATH_GPT_find_common_remainder_l2124_212469

theorem find_common_remainder :
  ∃ (d : ℕ), 100 ≤ d ∧ d ≤ 999 ∧ (312837 % d = 96) ∧ (310650 % d = 96) :=
sorry

end NUMINAMATH_GPT_find_common_remainder_l2124_212469


namespace NUMINAMATH_GPT_box_dimensions_l2124_212473

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  -- We assume the proof is correct based on given conditions
  sorry

end NUMINAMATH_GPT_box_dimensions_l2124_212473


namespace NUMINAMATH_GPT_most_appropriate_survey_is_D_l2124_212418

-- Define the various scenarios as Lean definitions
def survey_A := "Testing whether a certain brand of fresh milk meets food hygiene standards, using a census method."
def survey_B := "Security check before taking the subway, using a sampling survey method."
def survey_C := "Understanding the sleep time of middle school students in Jiangsu Province, using a census method."
def survey_D := "Understanding the way Nanjing residents commemorate the Qingming Festival, using a sampling survey method."

-- Define the type for specifying which survey method is the most appropriate
def appropriate_survey (survey : String) : Prop := 
  survey = survey_D

-- The theorem statement proving that the most appropriate survey is D
theorem most_appropriate_survey_is_D : appropriate_survey survey_D :=
by sorry

end NUMINAMATH_GPT_most_appropriate_survey_is_D_l2124_212418


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_q_l2124_212464

theorem arithmetic_sequence_ratio_q :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ), 
    (0 < q) →
    (S 2 = 3 * a 2 + 2) →
    (S 4 = 3 * a 4 + 2) →
    (q = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_q_l2124_212464


namespace NUMINAMATH_GPT_coffee_is_32_3_percent_decaf_l2124_212451

def percent_decaf_coffee_stock (total_weight initial_weight : ℕ) (initial_A_rate initial_B_rate initial_C_rate additional_weight additional_A_rate additional_D_rate : ℚ) 
(initial_A_decaf initial_B_decaf initial_C_decaf additional_D_decaf : ℚ) : ℚ :=
  let initial_A_weight := initial_A_rate * initial_weight
  let initial_B_weight := initial_B_rate * initial_weight
  let initial_C_weight := initial_C_rate * initial_weight
  let additional_A_weight := additional_A_rate * additional_weight
  let additional_D_weight := additional_D_rate * additional_weight

  let initial_A_decaf_weight := initial_A_decaf * initial_A_weight
  let initial_B_decaf_weight := initial_B_decaf * initial_B_weight
  let initial_C_decaf_weight := initial_C_decaf * initial_C_weight
  let additional_A_decaf_weight := initial_A_decaf * additional_A_weight
  let additional_D_decaf_weight := additional_D_decaf * additional_D_weight

  let total_decaf_weight := initial_A_decaf_weight + initial_B_decaf_weight + initial_C_decaf_weight + additional_A_decaf_weight + additional_D_decaf_weight

  (total_decaf_weight / total_weight) * 100

theorem coffee_is_32_3_percent_decaf : 
  percent_decaf_coffee_stock 1000 800 (40/100) (35/100) (25/100) 200 (50/100) (50/100) (20/100) (30/100) (45/100) (65/100) = 32.3 := 
  by 
    sorry

end NUMINAMATH_GPT_coffee_is_32_3_percent_decaf_l2124_212451


namespace NUMINAMATH_GPT_average_cost_of_testing_l2124_212485

theorem average_cost_of_testing (total_machines : Nat) (faulty_machines : Nat) (cost_per_test : Nat) 
  (h_total : total_machines = 5) (h_faulty : faulty_machines = 2) (h_cost : cost_per_test = 1000) :
  (2000 * (2 / 5 * 1 / 4) + 3000 * (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3) + 
  4000 * (1 - (2 / 5 * 1 / 4) - (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3))) = 3500 :=
  by
  sorry

end NUMINAMATH_GPT_average_cost_of_testing_l2124_212485


namespace NUMINAMATH_GPT_find_set_A_find_range_a_l2124_212433

-- Define the universal set and the complement condition for A
def universal_set : Set ℝ := {x | true}
def complement_A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 > 0}

-- Define the set A
def set_A : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2}

-- Define the set C
def set_C (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

-- Define the proof problem for part (1)
theorem find_set_A : { x | -1 / 2 ≤ x ∧ x ≤ 2 } = { x | ¬ (2 * x^2 - 3 * x - 2 > 0) } :=
by
  sorry

-- Define the proof problem for part (2)
theorem find_range_a (a : ℝ) (C_ne_empty : (set_C a).Nonempty) (sufficient_not_necessary : ∀ x, x ∈ set_C a → x ∈ set_A → x ∈ set_A) :
  a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_find_set_A_find_range_a_l2124_212433


namespace NUMINAMATH_GPT_straight_flush_probability_l2124_212484

open Classical

noncomputable def number_of_possible_hands : ℕ := Nat.choose 52 5

noncomputable def number_of_straight_flushes : ℕ := 40 

noncomputable def probability_of_straight_flush : ℚ := number_of_straight_flushes / number_of_possible_hands

theorem straight_flush_probability :
  probability_of_straight_flush = 1 / 64974 := by
  sorry

end NUMINAMATH_GPT_straight_flush_probability_l2124_212484


namespace NUMINAMATH_GPT_smallest_t_in_colored_grid_l2124_212410

theorem smallest_t_in_colored_grid :
  ∃ (t : ℕ), (t > 0) ∧
  (∀ (coloring : Fin (100*100) → ℕ),
      (∀ (n : ℕ), (∃ (squares : Finset (Fin (100*100))), squares.card ≤ 104 ∧ ∀ x ∈ squares, coloring x = n)) →
      (∃ (rectangle : Finset (Fin (100*100))),
        (rectangle.card = t ∧ (t = 1 ∨ (t = 2 ∨ ∃ (l : ℕ), (l = 12 ∧ rectangle.card = l) ∧ (∃ (c : ℕ), (c = 3 ∧ ∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ∃(s1 s2 s3 : Fin (100*100)), (s1 ∈ rectangle ∧ coloring s1 = a) ∧ (s2 ∈ rectangle ∧ coloring s2 = b) ∧ (s3 ∈ rectangle ∧ coloring s3 = c))))))))) :=
sorry

end NUMINAMATH_GPT_smallest_t_in_colored_grid_l2124_212410


namespace NUMINAMATH_GPT_find_a_l2124_212462

theorem find_a (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 20) (h2 : (56831742 - a) % 17 = 0) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2124_212462


namespace NUMINAMATH_GPT_compound_proposition_truth_l2124_212446

theorem compound_proposition_truth (p q : Prop) (h1 : ¬p ∨ ¬q = False) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end NUMINAMATH_GPT_compound_proposition_truth_l2124_212446


namespace NUMINAMATH_GPT_carmen_candle_usage_l2124_212476

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_carmen_candle_usage_l2124_212476


namespace NUMINAMATH_GPT_sophie_saves_money_l2124_212452

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end NUMINAMATH_GPT_sophie_saves_money_l2124_212452


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2124_212441

open Real

/-- Given the hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 2 - x^2 / 8 = 1

/-- Prove the eccentricity of the given hyperbola -/
theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_equation x y) : 
  ∃ e : ℝ, e = sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2124_212441


namespace NUMINAMATH_GPT_fraction_of_orange_juice_is_correct_l2124_212467

noncomputable def fraction_of_orange_juice_in_mixture (V1 V2 juice1_ratio juice2_ratio : ℚ) : ℚ :=
  let juice1 := V1 * juice1_ratio
  let juice2 := V2 * juice2_ratio
  let total_juice := juice1 + juice2
  let total_volume := V1 + V2
  total_juice / total_volume

theorem fraction_of_orange_juice_is_correct :
  fraction_of_orange_juice_in_mixture 800 500 (1/4) (1/3) = 7 / 25 :=
by sorry

end NUMINAMATH_GPT_fraction_of_orange_juice_is_correct_l2124_212467


namespace NUMINAMATH_GPT_parabola_c_value_l2124_212444

theorem parabola_c_value (b c : ℝ) 
  (h1 : 20 = 2*(-2)^2 + b*(-2) + c) 
  (h2 : 28 = 2*2^2 + b*2 + c) : 
  c = 16 :=
by
  sorry

end NUMINAMATH_GPT_parabola_c_value_l2124_212444


namespace NUMINAMATH_GPT_smaller_solution_quadratic_equation_l2124_212408

theorem smaller_solution_quadratic_equation :
  (∀ x : ℝ, x^2 + 7 * x - 30 = 0 → x = -10 ∨ x = 3) → -10 = min (-10) 3 :=
by
  sorry

end NUMINAMATH_GPT_smaller_solution_quadratic_equation_l2124_212408


namespace NUMINAMATH_GPT_correct_amendment_statements_l2124_212442

/-- The amendment includes the abuse of administrative power by administrative organs 
    to exclude or limit competition. -/
def abuse_of_power_in_amendment : Prop :=
  true

/-- The amendment includes illegal fundraising. -/
def illegal_fundraising_in_amendment : Prop :=
  true

/-- The amendment includes apportionment of expenses. -/
def apportionment_of_expenses_in_amendment : Prop :=
  true

/-- The amendment includes failure to pay minimum living allowances or social insurance benefits according to law. -/
def failure_to_pay_benefits_in_amendment : Prop :=
  true

/-- The amendment further standardizes the exercise of government power. -/
def standardizes_govt_power : Prop :=
  true

/-- The amendment better protects the legitimate rights and interests of citizens. -/
def protects_rights : Prop :=
  true

/-- The amendment expands the channels for citizens' democratic participation. -/
def expands_democratic_participation : Prop :=
  false

/-- The amendment expands the scope of government functions. -/
def expands_govt_functions : Prop :=
  false

/-- The correct answer to which set of statements is true about the amendment is {②, ③}.
    This is encoded as proving (standardizes_govt_power ∧ protects_rights) = true. -/
theorem correct_amendment_statements : (standardizes_govt_power ∧ protects_rights) ∧ 
                                      ¬(expands_democratic_participation ∧ expands_govt_functions) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_amendment_statements_l2124_212442


namespace NUMINAMATH_GPT_find_m_plus_M_l2124_212423

-- Given conditions
def cond1 (x y z : ℝ) := x + y + z = 4
def cond2 (x y z : ℝ) := x^2 + y^2 + z^2 = 6

-- Proof statement: The sum of the smallest and largest possible values of x is 8/3
theorem find_m_plus_M :
  ∀ (x y z : ℝ), cond1 x y z → cond2 x y z → (min (x : ℝ) (max x y) + max (x : ℝ) (min x y) = 8 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_m_plus_M_l2124_212423


namespace NUMINAMATH_GPT_matrix_multiplication_problem_l2124_212487

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_multiplication_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = ![![5, 2], ![-2, 4]]) :
  B * A = ![![5, 2], ![-2, 4]] :=
sorry

end NUMINAMATH_GPT_matrix_multiplication_problem_l2124_212487


namespace NUMINAMATH_GPT_angie_bought_18_pretzels_l2124_212496

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end NUMINAMATH_GPT_angie_bought_18_pretzels_l2124_212496


namespace NUMINAMATH_GPT_ramesh_transport_cost_l2124_212468

-- Definitions for conditions
def labelled_price (P : ℝ) : Prop := P = 13500 / 0.80
def selling_price (P : ℝ) : Prop := P * 1.10 = 18975
def transport_cost (T : ℝ) (extra_amount : ℝ) (installation_cost : ℝ) : Prop := T = extra_amount - installation_cost

-- The theorem statement to be proved
theorem ramesh_transport_cost (P T extra_amount installation_cost: ℝ) 
  (h1 : labelled_price P) 
  (h2 : selling_price P) 
  (h3 : extra_amount = 18975 - P)
  (h4 : installation_cost = 250) : 
  transport_cost T extra_amount installation_cost :=
by
  sorry

end NUMINAMATH_GPT_ramesh_transport_cost_l2124_212468


namespace NUMINAMATH_GPT_calculate_ray_grocery_bill_l2124_212471

noncomputable def ray_grocery_total_cost : ℝ :=
let hamburger_meat_price := 5.0
let crackers_price := 3.5
let frozen_vegetables_price := 2.0 * 4
let cheese_price := 3.5
let chicken_price := 6.5
let cereal_price := 4.0
let wine_price := 10.0
let cookies_price := 3.0

let discount_hamburger_meat := hamburger_meat_price * 0.10
let discount_crackers := crackers_price * 0.10
let discount_frozen_vegetables := frozen_vegetables_price * 0.10
let discount_cheese := cheese_price * 0.05
let discount_chicken := chicken_price * 0.05
let discount_wine := wine_price * 0.15

let discounted_hamburger_meat_price := hamburger_meat_price - discount_hamburger_meat
let discounted_crackers_price := crackers_price - discount_crackers
let discounted_frozen_vegetables_price := frozen_vegetables_price - discount_frozen_vegetables
let discounted_cheese_price := cheese_price - discount_cheese
let discounted_chicken_price := chicken_price - discount_chicken
let discounted_wine_price := wine_price - discount_wine

let total_discounted_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  discounted_wine_price +
  cookies_price

let food_items_total_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  cookies_price

let food_sales_tax := food_items_total_price * 0.06
let wine_sales_tax := discounted_wine_price * 0.09

let total_with_tax := total_discounted_price + food_sales_tax + wine_sales_tax

total_with_tax

theorem calculate_ray_grocery_bill :
  ray_grocery_total_cost = 42.51 :=
sorry

end NUMINAMATH_GPT_calculate_ray_grocery_bill_l2124_212471


namespace NUMINAMATH_GPT_Teresa_age_at_Michiko_birth_l2124_212460

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_Teresa_age_at_Michiko_birth_l2124_212460


namespace NUMINAMATH_GPT_not_divisible_by_1980_divisible_by_1981_l2124_212455

open Nat

theorem not_divisible_by_1980 (x : ℕ) : ¬ (2^100 * x - 1) % 1980 = 0 := by
sorry

theorem divisible_by_1981 : ∃ x : ℕ, (2^100 * x - 1) % 1981 = 0 := by
sorry

end NUMINAMATH_GPT_not_divisible_by_1980_divisible_by_1981_l2124_212455


namespace NUMINAMATH_GPT_plane_tiled_squares_triangles_percentage_l2124_212453

theorem plane_tiled_squares_triangles_percentage :
    (percent_triangle_area : ℚ) = 625 / 10000 := sorry

end NUMINAMATH_GPT_plane_tiled_squares_triangles_percentage_l2124_212453


namespace NUMINAMATH_GPT_sufficient_not_necessary_range_l2124_212472

variable (x a : ℝ)

theorem sufficient_not_necessary_range (h1 : ∀ x, |x| < 1 → x < a) 
                                       (h2 : ¬(∀ x, x < a → |x| < 1)) :
  a ≥ 1 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_range_l2124_212472


namespace NUMINAMATH_GPT_aku_invited_friends_l2124_212419

def total_cookies (packages : ℕ) (cookies_per_package : ℕ) := packages * cookies_per_package

def total_children (total_cookies : ℕ) (cookies_per_child : ℕ) := total_cookies / cookies_per_child

def invited_friends (total_children : ℕ) := total_children - 1

theorem aku_invited_friends (packages cookies_per_package cookies_per_child : ℕ) (h1 : packages = 3) (h2 : cookies_per_package = 25) (h3 : cookies_per_child = 15) :
  invited_friends (total_children (total_cookies packages cookies_per_package) cookies_per_child) = 4 :=
by
  sorry

end NUMINAMATH_GPT_aku_invited_friends_l2124_212419


namespace NUMINAMATH_GPT_prove_ab_leq_one_l2124_212420

theorem prove_ab_leq_one (a b : ℝ) (h : (a + b + a) * (a + b + b) = 9) : ab ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_prove_ab_leq_one_l2124_212420


namespace NUMINAMATH_GPT_evaluate_f_at_5_l2124_212401

def f (x : ℝ) := 2 * x^5 - 5 * x^4 - 4 * x^3 + 3 * x^2 - 524

theorem evaluate_f_at_5 : f 5 = 2176 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_5_l2124_212401


namespace NUMINAMATH_GPT_question1_question2_l2124_212493

-- Define the conditions
def numTraditionalChinesePaintings : Nat := 5
def numOilPaintings : Nat := 2
def numWatercolorPaintings : Nat := 7

-- Define the number of ways to choose one painting from each category
def numWaysToChooseOnePaintingFromEachCategory : Nat :=
  numTraditionalChinesePaintings * numOilPaintings * numWatercolorPaintings

-- Define the number of ways to choose two paintings of different types
def numWaysToChooseTwoPaintingsOfDifferentTypes : Nat :=
  (numTraditionalChinesePaintings * numOilPaintings) +
  (numTraditionalChinesePaintings * numWatercolorPaintings) +
  (numOilPaintings * numWatercolorPaintings)

-- Theorems to prove the required results
theorem question1 : numWaysToChooseOnePaintingFromEachCategory = 70 := by
  sorry

theorem question2 : numWaysToChooseTwoPaintingsOfDifferentTypes = 59 := by
  sorry

end NUMINAMATH_GPT_question1_question2_l2124_212493


namespace NUMINAMATH_GPT_cos_pi_plus_alpha_l2124_212409

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π + α) = - 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_plus_alpha_l2124_212409


namespace NUMINAMATH_GPT_num_members_in_league_l2124_212417

-- Definitions based on conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def shorts_cost : ℕ := tshirt_cost
def total_cost_per_member : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)
def total_league_cost : ℕ := 4719

-- Theorem statement
theorem num_members_in_league : (total_league_cost / total_cost_per_member) = 74 :=
by
  sorry

end NUMINAMATH_GPT_num_members_in_league_l2124_212417


namespace NUMINAMATH_GPT_scout_troop_net_profit_l2124_212427

theorem scout_troop_net_profit :
  ∃ (cost_per_bar selling_price_per_bar : ℝ),
    cost_per_bar = 1 / 3 ∧
    selling_price_per_bar = 0.6 ∧
    (1500 * selling_price_per_bar - (1500 * cost_per_bar + 50) = 350) :=
by {
  sorry
}

end NUMINAMATH_GPT_scout_troop_net_profit_l2124_212427


namespace NUMINAMATH_GPT_mul_exponents_l2124_212482

theorem mul_exponents (a : ℝ) : ((-2 * a) ^ 2) * (a ^ 4) = 4 * a ^ 6 := by
  sorry

end NUMINAMATH_GPT_mul_exponents_l2124_212482


namespace NUMINAMATH_GPT_abs_twice_sub_pi_l2124_212491

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_abs_twice_sub_pi_l2124_212491


namespace NUMINAMATH_GPT_box_height_l2124_212488

theorem box_height (volume length width : ℝ) (h : ℝ) (h_volume : volume = 315) (h_length : length = 7) (h_width : width = 9) :
  h = 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_box_height_l2124_212488


namespace NUMINAMATH_GPT_min_dot_product_on_hyperbola_l2124_212461

open Real

theorem min_dot_product_on_hyperbola :
  ∀ (P : ℝ × ℝ), (P.1 ≥ 1 ∧ P.1^2 - (P.2^2) / 3 = 1) →
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  ∃ m : ℝ, m = -2 ∧ PA1.1 * PF2.1 + PA1.2 * PF2.2 = m :=
by
  intros P h
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  use -2
  sorry

end NUMINAMATH_GPT_min_dot_product_on_hyperbola_l2124_212461


namespace NUMINAMATH_GPT_reciprocal_relationship_l2124_212474

theorem reciprocal_relationship (a b : ℝ) (h₁ : a = 2 - Real.sqrt 3) (h₂ : b = Real.sqrt 3 + 2) : 
  a * b = 1 :=
by
  rw [h₁, h₂]
  sorry

end NUMINAMATH_GPT_reciprocal_relationship_l2124_212474


namespace NUMINAMATH_GPT_circle_intersection_range_l2124_212480

noncomputable def circleIntersectionRange (r : ℝ) : Prop :=
  1 < r ∧ r < 11

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) :
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) ↔ circleIntersectionRange r :=
by
  sorry

end NUMINAMATH_GPT_circle_intersection_range_l2124_212480


namespace NUMINAMATH_GPT_find_x_for_prime_square_l2124_212443

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_prime_square_l2124_212443


namespace NUMINAMATH_GPT_num_customers_left_more_than_remaining_l2124_212478

theorem num_customers_left_more_than_remaining (initial remaining : ℕ) (h : initial = 11 ∧ remaining = 3) : (initial - remaining) = (remaining + 5) :=
by sorry

end NUMINAMATH_GPT_num_customers_left_more_than_remaining_l2124_212478


namespace NUMINAMATH_GPT_k_range_condition_l2124_212498

theorem k_range_condition (k : ℝ) :
    (∀ x : ℝ, x^2 - (2 * k - 6) * x + k - 3 > 0) ↔ (3 < k ∧ k < 4) :=
by
  sorry

end NUMINAMATH_GPT_k_range_condition_l2124_212498


namespace NUMINAMATH_GPT_Katie_marble_count_l2124_212424

theorem Katie_marble_count :
  ∀ (pink_marbles orange_marbles purple_marbles total_marbles : ℕ),
  pink_marbles = 13 →
  orange_marbles = pink_marbles - 9 →
  purple_marbles = 4 * orange_marbles →
  total_marbles = pink_marbles + orange_marbles + purple_marbles →
  total_marbles = 33 :=
by
  intros pink_marbles orange_marbles purple_marbles total_marbles
  intros hpink horange hpurple htotal
  sorry

end NUMINAMATH_GPT_Katie_marble_count_l2124_212424


namespace NUMINAMATH_GPT_election_total_votes_l2124_212483

theorem election_total_votes (V: ℝ) (valid_votes: ℝ) (candidate_votes: ℝ) (invalid_rate: ℝ) (candidate_rate: ℝ) :
  candidate_rate = 0.75 →
  invalid_rate = 0.15 →
  candidate_votes = 357000 →
  valid_votes = (1 - invalid_rate) * V →
  candidate_votes = candidate_rate * valid_votes →
  V = 560000 :=
by
  intros candidate_rate_eq invalid_rate_eq candidate_votes_eq valid_votes_eq equation
  sorry

end NUMINAMATH_GPT_election_total_votes_l2124_212483


namespace NUMINAMATH_GPT_evaporation_rate_l2124_212437

theorem evaporation_rate (initial_water_volume : ℕ) (days : ℕ) (percentage_evaporated : ℕ) (evaporated_fraction : ℚ)
  (h1 : initial_water_volume = 10)
  (h2 : days = 50)
  (h3 : percentage_evaporated = 3)
  (h4 : evaporated_fraction = percentage_evaporated / 100) :
  (initial_water_volume * evaporated_fraction) / days = 0.06 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_evaporation_rate_l2124_212437


namespace NUMINAMATH_GPT_first_discount_percentage_l2124_212494

theorem first_discount_percentage
  (P : ℝ)
  (initial_price final_price : ℝ)
  (second_discount : ℕ)
  (h1 : initial_price = 200)
  (h2 : final_price = 144)
  (h3 : second_discount = 10)
  (h4 : final_price = (P - (second_discount / 100) * P)) :
  (∃ x : ℝ, P = initial_price - (x / 100) * initial_price ∧ x = 20) :=
sorry

end NUMINAMATH_GPT_first_discount_percentage_l2124_212494


namespace NUMINAMATH_GPT_problem_solution_l2124_212400

theorem problem_solution
  (a b c : ℝ)
  (habc_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2124_212400


namespace NUMINAMATH_GPT_find_x_in_plane_figure_l2124_212459

theorem find_x_in_plane_figure (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 360) 
  (h3 : 2 * x + 160 = 360) : 
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_plane_figure_l2124_212459


namespace NUMINAMATH_GPT_jelly_bean_ratio_l2124_212428

-- Define the number of jelly beans each person has
def napoleon_jelly_beans : ℕ := 17
def sedrich_jelly_beans : ℕ := napoleon_jelly_beans + 4
def mikey_jelly_beans : ℕ := 19

-- Define the sum of jelly beans of Napoleon and Sedrich
def sum_jelly_beans : ℕ := napoleon_jelly_beans + sedrich_jelly_beans

-- Define the ratio of the sum of Napoleon and Sedrich's jelly beans to Mikey's jelly beans
def ratio : ℚ := sum_jelly_beans / mikey_jelly_beans

-- Prove that the ratio is 2
theorem jelly_bean_ratio : ratio = 2 := by
  -- We skip the proof steps since the focus here is on the correct statement
  sorry

end NUMINAMATH_GPT_jelly_bean_ratio_l2124_212428


namespace NUMINAMATH_GPT_coeffs_of_quadratic_eq_l2124_212438

theorem coeffs_of_quadratic_eq :
  ∃ a b c : ℤ, (2 * x^2 + x - 5 = 0) → (a = 2 ∧ b = 1 ∧ c = -5) :=
by
  sorry

end NUMINAMATH_GPT_coeffs_of_quadratic_eq_l2124_212438


namespace NUMINAMATH_GPT_find_other_outlet_rate_l2124_212456

open Real

-- Definitions based on conditions
def V : ℝ := 20 * 1728   -- volume of the tank in cubic inches
def r1 : ℝ := 5          -- rate of inlet pipe in cubic inches/min
def r2 : ℝ := 8          -- rate of one outlet pipe in cubic inches/min
def t : ℝ := 2880        -- time in minutes required to empty the tank
 
-- Mathematically equivalent proof statement
theorem find_other_outlet_rate (x : ℝ) : 
  -- Given conditions
  V = 34560 →
  r1 = 5 →
  r2 = 8 →
  t = 2880 →
  -- Statement to prove
  V = (r2 + x - r1) * t → x = 9 :=
by
  intro hV hr1 hr2 ht hEq
  sorry

end NUMINAMATH_GPT_find_other_outlet_rate_l2124_212456


namespace NUMINAMATH_GPT_number_of_integer_segments_l2124_212411

theorem number_of_integer_segments (DE EF : ℝ) (H1 : DE = 24) (H2 : EF = 25) : 
  ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_segments_l2124_212411


namespace NUMINAMATH_GPT_sums_have_same_remainder_l2124_212429

theorem sums_have_same_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) : 
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i.val) % (2 * n) = (a j + j.val) % (2 * n)) := 
sorry

end NUMINAMATH_GPT_sums_have_same_remainder_l2124_212429


namespace NUMINAMATH_GPT_trigonometric_identity_l2124_212415

theorem trigonometric_identity : 
  Real.cos 6 * Real.cos 36 + Real.sin 6 * Real.cos 54 = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l2124_212415


namespace NUMINAMATH_GPT_sequence_term_306_l2124_212477

theorem sequence_term_306 (a1 a2 : ℤ) (r : ℤ) (n : ℕ) (h1 : a1 = 7) (h2 : a2 = -7) (h3 : r = -1) (h4 : a2 = r * a1) : 
  ∃ a306 : ℤ, a306 = -7 ∧ a306 = a1 * r^305 :=
by
  use -7
  sorry

end NUMINAMATH_GPT_sequence_term_306_l2124_212477


namespace NUMINAMATH_GPT_tan_theta_cos_sin_id_l2124_212434

theorem tan_theta_cos_sin_id (θ : ℝ) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) =
  (17 * (Real.sqrt 10 + 1)) / 24 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_cos_sin_id_l2124_212434


namespace NUMINAMATH_GPT_largest_common_value_less_than_1000_l2124_212435

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, 
    (∃ n : ℕ, a = 4 + 5 * n) ∧
    (∃ m : ℕ, a = 5 + 10 * m) ∧
    a % 4 = 1 ∧
    a < 1000 ∧
    (∀ b : ℕ, 
      (∃ n : ℕ, b = 4 + 5 * n) ∧
      (∃ m : ℕ, b = 5 + 10 * m) ∧
      b % 4 = 1 ∧
      b < 1000 → 
      b ≤ a) ∧ 
    a = 989 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_value_less_than_1000_l2124_212435


namespace NUMINAMATH_GPT_find_weekly_allowance_l2124_212421

noncomputable def weekly_allowance (A : ℝ) : Prop :=
  let spent_at_arcade := (3/5) * A
  let remaining_after_arcade := A - spent_at_arcade
  let spent_at_toy_store := (1/3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - spent_at_toy_store
  remaining_after_toy_store = 1.20

theorem find_weekly_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 4.50 := 
  sorry

end NUMINAMATH_GPT_find_weekly_allowance_l2124_212421


namespace NUMINAMATH_GPT_machine_x_widgets_per_hour_l2124_212432

-- Definitions of the variables and conditions
variable (Wx Wy Tx Ty: ℝ)
variable (h1: Tx = Ty + 60)
variable (h2: Wy = 1.20 * Wx)
variable (h3: Wx * Tx = 1080)
variable (h4: Wy * Ty = 1080)

-- Statement of the problem to prove
theorem machine_x_widgets_per_hour : Wx = 3 := by
  sorry

end NUMINAMATH_GPT_machine_x_widgets_per_hour_l2124_212432


namespace NUMINAMATH_GPT_donation_amount_l2124_212466

theorem donation_amount 
  (total_needed : ℕ) (bronze_amount : ℕ) (silver_amount : ℕ) (raised_so_far : ℕ)
  (bronze_families : ℕ) (silver_families : ℕ) (other_family_donation : ℕ)
  (final_push_needed : ℕ) 
  (h1 : total_needed = 750) 
  (h2 : bronze_amount = 25)
  (h3 : silver_amount = 50)
  (h4 : bronze_families = 10)
  (h5 : silver_families = 7)
  (h6 : raised_so_far = 600)
  (h7 : final_push_needed = 50)
  (h8 : raised_so_far = bronze_families * bronze_amount + silver_families * silver_amount)
  (h9 : total_needed - raised_so_far - other_family_donation = final_push_needed) : 
  other_family_donation = 100 :=
by
  sorry

end NUMINAMATH_GPT_donation_amount_l2124_212466


namespace NUMINAMATH_GPT_calculate_fraction_l2124_212426

theorem calculate_fraction : (10^9 + 10^6) / (3 * 10^4) = 100100 / 3 := by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l2124_212426


namespace NUMINAMATH_GPT_value_three_in_range_of_g_l2124_212431

theorem value_three_in_range_of_g (a : ℝ) : ∀ (a : ℝ), ∃ (x : ℝ), x^2 + a * x + 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_three_in_range_of_g_l2124_212431


namespace NUMINAMATH_GPT_geometric_series_S6_value_l2124_212403

theorem geometric_series_S6_value (S : ℕ → ℝ) (S3 : S 3 = 3) (S9_minus_S6 : S 9 - S 6 = 12) : 
  S 6 = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_S6_value_l2124_212403


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l2124_212402

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l2124_212402


namespace NUMINAMATH_GPT_rectangle_area_l2124_212412

theorem rectangle_area (x w : ℝ) (h₁ : 3 * w = 3 * w) (h₂ : x^2 = 9 * w^2 + w^2) : 
  (3 * w) * w = (3 / 10) * x^2 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2124_212412


namespace NUMINAMATH_GPT_scores_are_sample_l2124_212450

-- Define the total number of students
def total_students : ℕ := 5000

-- Define the number of selected students for sampling
def selected_students : ℕ := 200

-- Define a predicate that checks if a selection is a sample
def is_sample (total selected : ℕ) : Prop :=
  selected < total

-- The proposition that needs to be proven
theorem scores_are_sample : is_sample total_students selected_students := 
by 
  -- Proof of the theorem is omitted.
  sorry

end NUMINAMATH_GPT_scores_are_sample_l2124_212450


namespace NUMINAMATH_GPT_certain_number_is_1_l2124_212449

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_1_l2124_212449


namespace NUMINAMATH_GPT_total_equipment_cost_l2124_212470

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end NUMINAMATH_GPT_total_equipment_cost_l2124_212470


namespace NUMINAMATH_GPT_total_blue_points_l2124_212499

variables (a b c d : ℕ)

theorem total_blue_points (h1 : a * b = 56) (h2 : c * d = 50) (h3 : a + b = c + d) :
  a + b = 15 :=
sorry

end NUMINAMATH_GPT_total_blue_points_l2124_212499


namespace NUMINAMATH_GPT_area_of_intersection_is_zero_l2124_212458

-- Define the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 16
def circle2 (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the theorem to prove
theorem area_of_intersection_is_zero : 
  ∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    x1 = x2 ∧ y1 = -y2 → 
    0 = 0 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_area_of_intersection_is_zero_l2124_212458


namespace NUMINAMATH_GPT_fit_small_boxes_l2124_212465

def larger_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def small_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem fit_small_boxes (L W H l w h : ℕ)
  (larger_box_dim : L = 12 ∧ W = 14 ∧ H = 16)
  (small_box_dim : l = 3 ∧ w = 7 ∧ h = 2)
  (min_boxes : larger_box_volume L W H / small_box_volume l w h = 64) :
  ∃ n, n ≥ 64 :=
by
  sorry

end NUMINAMATH_GPT_fit_small_boxes_l2124_212465


namespace NUMINAMATH_GPT_min_value_expression_is_4_l2124_212463

noncomputable def min_value_expression (x : ℝ) : ℝ :=
(3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1)

theorem min_value_expression_is_4 : ∃ x : ℝ, min_value_expression x = 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_is_4_l2124_212463


namespace NUMINAMATH_GPT_remainder_sum_mod_l2124_212440

theorem remainder_sum_mod (a b c d e : ℕ)
  (h₁ : a = 17145)
  (h₂ : b = 17146)
  (h₃ : c = 17147)
  (h₄ : d = 17148)
  (h₅ : e = 17149)
  : (a + b + c + d + e) % 10 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_mod_l2124_212440


namespace NUMINAMATH_GPT_polygon_diagonals_twice_sides_l2124_212407

theorem polygon_diagonals_twice_sides
  (n : ℕ)
  (h : n * (n - 3) / 2 = 2 * n) :
  n = 7 :=
sorry

end NUMINAMATH_GPT_polygon_diagonals_twice_sides_l2124_212407


namespace NUMINAMATH_GPT_angle_between_vectors_l2124_212430

noncomputable def vec_a : ℝ × ℝ := (-2 * Real.sqrt 3, 2)
noncomputable def vec_b : ℝ × ℝ := (1, - Real.sqrt 3)

-- Define magnitudes
noncomputable def mag_a : ℝ := Real.sqrt ((-2 * Real.sqrt 3) ^ 2 + 2^2)
noncomputable def mag_b : ℝ := Real.sqrt (1^2 + (- Real.sqrt 3) ^ 2)

-- Define the dot product
noncomputable def dot_product : ℝ := (-2 * Real.sqrt 3) * 1 + 2 * (- Real.sqrt 3)

-- Define cosine of the angle theta
-- We use mag_a and mag_b defined above
noncomputable def cos_theta : ℝ := dot_product / (mag_a * mag_b)

-- Define the angle theta, within the range [0, π]
noncomputable def theta : ℝ := Real.arccos cos_theta

-- The expected result is θ = 5π / 6
theorem angle_between_vectors : theta = (5 * Real.pi) / 6 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_vectors_l2124_212430


namespace NUMINAMATH_GPT_balance_scale_cereal_l2124_212422

def scales_are_balanced (left_pan : ℕ) (right_pan : ℕ) : Prop :=
  left_pan = right_pan

theorem balance_scale_cereal (inaccurate_scales : ℕ → ℕ → Prop)
  (cereal : ℕ)
  (correct_weight : ℕ) :
  (∀ left_pan right_pan, inaccurate_scales left_pan right_pan → left_pan = right_pan) →
  (cereal / 2 = 1) →
  true :=
  sorry

end NUMINAMATH_GPT_balance_scale_cereal_l2124_212422


namespace NUMINAMATH_GPT_total_pencils_proof_l2124_212448

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_pencils_proof_l2124_212448


namespace NUMINAMATH_GPT_not_right_triangle_l2124_212481

/-- In a triangle ABC, with angles A, B, C, the condition A = B = 2 * C does not form a right-angled triangle. -/
theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) (h3 : A + B + C = 180) : 
    ¬(A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end NUMINAMATH_GPT_not_right_triangle_l2124_212481


namespace NUMINAMATH_GPT_right_triangle_sides_unique_l2124_212405

theorem right_triangle_sides_unique (a b c : ℕ) 
  (relatively_prime : Int.gcd (Int.gcd a b) c = 1) 
  (right_triangle : a ^ 2 + b ^ 2 = c ^ 2) 
  (increased_right_triangle : (a + 100) ^ 2 + (b + 100) ^ 2 = (c + 140) ^ 2) : 
  (a = 56 ∧ b = 33 ∧ c = 65) :=
by
  sorry 

end NUMINAMATH_GPT_right_triangle_sides_unique_l2124_212405


namespace NUMINAMATH_GPT_slope_of_line_l2124_212475

theorem slope_of_line : ∀ x y : ℝ, 3 * y + 2 * x = 6 * x - 9 → ∃ m b : ℝ, y = m * x + b ∧ m = -4 / 3 :=
by
  -- Sorry to skip proof
  sorry

end NUMINAMATH_GPT_slope_of_line_l2124_212475


namespace NUMINAMATH_GPT_trapezoid_side_length_l2124_212457

theorem trapezoid_side_length (s : ℝ) (A : ℝ) (x : ℝ) (y : ℝ) :
  s = 1 ∧ A = 1 ∧ y = 1/2 ∧ (1/2) * ((x + y) * y) = 1/4 → x = 1/2 :=
by
  intro h
  rcases h with ⟨hs, hA, hy, harea⟩
  sorry

end NUMINAMATH_GPT_trapezoid_side_length_l2124_212457


namespace NUMINAMATH_GPT_pipe_B_filling_time_l2124_212406

theorem pipe_B_filling_time (T_B : ℝ) 
  (A_filling_time : ℝ := 10) 
  (combined_filling_time: ℝ := 20/3)
  (A_rate : ℝ := 1 / A_filling_time)
  (combined_rate : ℝ := 1 / combined_filling_time) : 
  1 / T_B = combined_rate - A_rate → T_B = 20 := by 
  sorry

end NUMINAMATH_GPT_pipe_B_filling_time_l2124_212406


namespace NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l2124_212495

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_arccos_half_eq_pi_div_three_l2124_212495


namespace NUMINAMATH_GPT_part1_part2_l2124_212489

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

theorem part1 {x : ℝ} (hx : x = 0) : 
    f x 0 = 1 :=
by
  sorry

theorem part2 {x k : ℝ} (hx : 0 ≤ x) (hxf : f x k ≥ 1) : 
    k ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2124_212489


namespace NUMINAMATH_GPT_percentage_markup_l2124_212416

theorem percentage_markup (sell_price : ℝ) (cost_price : ℝ)
  (h_sell : sell_price = 8450) (h_cost : cost_price = 6500) : 
  (sell_price - cost_price) / cost_price * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_markup_l2124_212416


namespace NUMINAMATH_GPT_no_common_root_of_quadratics_l2124_212497

theorem no_common_root_of_quadratics (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬ ∃ x₀ : ℝ, (x₀^2 + b * x₀ + c = 0 ∧ x₀^2 + a * x₀ + d = 0) := 
by
  sorry

end NUMINAMATH_GPT_no_common_root_of_quadratics_l2124_212497


namespace NUMINAMATH_GPT_magdalena_fraction_picked_l2124_212414

noncomputable def fraction_picked_first_day
  (produced_apples: ℕ)
  (remaining_apples: ℕ)
  (fraction_picked: ℚ) : Prop :=
  ∃ (f : ℚ),
  produced_apples = 200 ∧
  remaining_apples = 20 ∧
  (f = fraction_picked) ∧
  (200 * f + 2 * 200 * f + (200 * f + 20)) = 200 - remaining_apples ∧
  fraction_picked = 1 / 5

theorem magdalena_fraction_picked :
  fraction_picked_first_day 200 20 (1 / 5) :=
sorry

end NUMINAMATH_GPT_magdalena_fraction_picked_l2124_212414


namespace NUMINAMATH_GPT_percentage_error_square_area_l2124_212404

theorem percentage_error_square_area (s : ℝ) (h : s > 0) :
  let s' := (1.02 * s)
  let actual_area := s^2
  let measured_area := s'^2
  let error_area := measured_area - actual_area
  let percentage_error := (error_area / actual_area) * 100
  percentage_error = 4.04 := 
sorry

end NUMINAMATH_GPT_percentage_error_square_area_l2124_212404


namespace NUMINAMATH_GPT_gcd_is_13_eval_at_neg1_l2124_212454

-- Define the GCD problem
def gcd_117_182 : ℕ := gcd 117 182

-- Define the polynomial evaluation problem
def f (x : ℝ) : ℝ := 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

-- Formalize the statements to be proved
theorem gcd_is_13 : gcd_117_182 = 13 := 
by sorry

theorem eval_at_neg1 : f (-1) = 12 := 
by sorry

end NUMINAMATH_GPT_gcd_is_13_eval_at_neg1_l2124_212454
