import Mathlib

namespace NUMINAMATH_GPT_more_green_peaches_than_red_l1222_122206

theorem more_green_peaches_than_red : 
  let red_peaches := 7
  let green_peaches := 8
  green_peaches - red_peaches = 1 := 
by
  let red_peaches := 7
  let green_peaches := 8
  show green_peaches - red_peaches = 1 
  sorry

end NUMINAMATH_GPT_more_green_peaches_than_red_l1222_122206


namespace NUMINAMATH_GPT_no_representation_of_expr_l1222_122294

theorem no_representation_of_expr :
  ¬ ∃ f g : ℝ → ℝ, (∀ x y : ℝ, 1 + x ^ 2016 * y ^ 2016 = f x * g y) :=
by
  sorry

end NUMINAMATH_GPT_no_representation_of_expr_l1222_122294


namespace NUMINAMATH_GPT_sequence_general_formula_l1222_122287

open Nat

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ (n : ℕ), n > 0 → (n+1) * (a (n + 1))^2 - n * (a n)^2 + (a (n + 1)) * (a n) = 0

theorem sequence_general_formula :
  ∃ (a : ℕ → ℝ), seq a ∧ (a 1 = 1) ∧ (∀ (n : ℕ), n > 0 → a n = 1 / n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1222_122287


namespace NUMINAMATH_GPT_smallest_positive_integer_with_20_divisors_is_432_l1222_122239

-- Define the condition that a number n has exactly 20 positive divisors
def has_exactly_20_divisors (n : ℕ) : Prop :=
  ∃ (a₁ a₂ : ℕ), a₁ + 1 = 5 ∧ a₂ + 1 = 4 ∧
                n = 2^a₁ * 3^a₂

-- The main statement to prove
theorem smallest_positive_integer_with_20_divisors_is_432 :
  ∀ n : ℕ, has_exactly_20_divisors n → n = 432 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_with_20_divisors_is_432_l1222_122239


namespace NUMINAMATH_GPT_random_event_is_B_l1222_122267

variable (isCertain : Event → Prop)
variable (isImpossible : Event → Prop)
variable (isRandom : Event → Prop)

variable (A : Event)
variable (B : Event)
variable (C : Event)
variable (D : Event)

-- Here we set the conditions as definitions in Lean 4:
def condition_A : isCertain A := sorry
def condition_B : isRandom B := sorry
def condition_C : isCertain C := sorry
def condition_D : isImpossible D := sorry

-- The theorem we need to prove:
theorem random_event_is_B : isRandom B := 
by
-- adding sorry to skip the proof
sorry

end NUMINAMATH_GPT_random_event_is_B_l1222_122267


namespace NUMINAMATH_GPT_find_h3_l1222_122280

noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^3 + 1) * (x^9 + 1) - 1) / (x^(3^3 - 1) - 1)

theorem find_h3 : h 3 = 3 := by
  sorry

end NUMINAMATH_GPT_find_h3_l1222_122280


namespace NUMINAMATH_GPT_find_sum_of_numbers_l1222_122241

theorem find_sum_of_numbers (x A B C : ℝ) (h1 : x > 0) (h2 : A = x) (h3 : B = 2 * x) (h4 : C = 3 * x) (h5 : A^2 + B^2 + C^2 = 2016) : A + B + C = 72 :=
sorry

end NUMINAMATH_GPT_find_sum_of_numbers_l1222_122241


namespace NUMINAMATH_GPT_rectangular_prism_dimensions_l1222_122236

theorem rectangular_prism_dimensions (b l h : ℕ) 
  (h1 : l = 3 * b) 
  (h2 : l = 2 * h) 
  (h3 : l * b * h = 12168) :
  b = 14 ∧ l = 42 ∧ h = 21 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_rectangular_prism_dimensions_l1222_122236


namespace NUMINAMATH_GPT_value_of_f_minus_3_l1222_122209

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + x^3 + 1

theorem value_of_f_minus_3 (a b : ℝ) (h : f 3 a b = 7) : f (-3) a b = -5 := 
by
  sorry

end NUMINAMATH_GPT_value_of_f_minus_3_l1222_122209


namespace NUMINAMATH_GPT_least_value_a_l1222_122273

theorem least_value_a (a : ℤ) :
  (∃ a : ℤ, a ≥ 0 ∧ (a ^ 6) % 1920 = 0) → a = 8 ∧ (a ^ 6) % 1920 = 0 :=
by
  sorry

end NUMINAMATH_GPT_least_value_a_l1222_122273


namespace NUMINAMATH_GPT_min_sum_x8y4z_l1222_122265

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end NUMINAMATH_GPT_min_sum_x8y4z_l1222_122265


namespace NUMINAMATH_GPT_number_of_subsets_of_five_element_set_is_32_l1222_122200

theorem number_of_subsets_of_five_element_set_is_32 (M : Finset ℕ) (h : M.card = 5) :
    (2 : ℕ) ^ 5 = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_five_element_set_is_32_l1222_122200


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1222_122258

noncomputable def f (x a : ℝ) := |x + a| + |x - a|

theorem part1_solution : (∀ x : ℝ, f x 1 ≥ 4 ↔ x ∈ Set.Iic (-2) ∨ x ∈ Set.Ici 2) := by
  sorry

theorem part2_solution : (∀ x : ℝ, f x a ≥ 6 → a ∈ Set.Iic (-3) ∨ a ∈ Set.Ici 3) := by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1222_122258


namespace NUMINAMATH_GPT_range_of_a_l1222_122244

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2 * a * x - 2 else x + (36 / x) - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, piecewise_f a x ≥ piecewise_f a 2) ↔ (2 ≤ a ∧ a ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1222_122244


namespace NUMINAMATH_GPT_price_Ramesh_paid_l1222_122251

-- Define the conditions
def labelled_price_sold (P : ℝ) := 1.10 * P
def discount_price_paid (P : ℝ) := 0.80 * P
def additional_costs := 125 + 250
def total_cost (P : ℝ) := discount_price_paid P + additional_costs

-- The main theorem stating that given the conditions,
-- the price Ramesh paid for the refrigerator is Rs. 13175.
theorem price_Ramesh_paid (P : ℝ) (H : labelled_price_sold P = 17600) :
  total_cost P = 13175 :=
by
  -- Providing a placeholder, as we do not need to provide the proof steps in the problem formulation
  sorry

end NUMINAMATH_GPT_price_Ramesh_paid_l1222_122251


namespace NUMINAMATH_GPT_ratio_of_diagonals_to_sides_l1222_122245

-- Define the given parameters and formula
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- State the theorem
theorem ratio_of_diagonals_to_sides (n : ℕ) (h : n = 5) : 
  (num_diagonals n) / n = 1 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_ratio_of_diagonals_to_sides_l1222_122245


namespace NUMINAMATH_GPT_hex_B2F_to_base10_l1222_122286

theorem hex_B2F_to_base10 :
  let b := 11
  let two := 2
  let f := 15
  let base := 16
  (b * base^2 + two * base^1 + f * base^0) = 2863 :=
by
  sorry

end NUMINAMATH_GPT_hex_B2F_to_base10_l1222_122286


namespace NUMINAMATH_GPT_right_triangle_leg_squared_l1222_122276

variable (a b c : ℝ)

theorem right_triangle_leg_squared (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4 * (a + 1) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_squared_l1222_122276


namespace NUMINAMATH_GPT_alex_needs_additional_coins_l1222_122249

theorem alex_needs_additional_coins :
  let n := 15
  let current_coins := 63
  let target_sum := (n * (n + 1)) / 2
  let additional_coins := target_sum - current_coins
  additional_coins = 57 :=
by
  sorry

end NUMINAMATH_GPT_alex_needs_additional_coins_l1222_122249


namespace NUMINAMATH_GPT_speed_ratio_l1222_122263

theorem speed_ratio (v1 v2 : ℝ) (t1 t2 : ℝ) (dist_before dist_after : ℝ) (total_dist : ℝ)
  (h1 : dist_before + dist_after = total_dist)
  (h2 : dist_before = 20)
  (h3 : dist_after = 20)
  (h4 : t2 = t1 + 11)
  (h5 : t2 = 22)
  (h6 : t1 = dist_before / v1)
  (h7 : t2 = dist_after / v2) :
  v1 / v2 = 2 := 
sorry

end NUMINAMATH_GPT_speed_ratio_l1222_122263


namespace NUMINAMATH_GPT_cube_and_difference_of_squares_l1222_122202

theorem cube_and_difference_of_squares (x : ℤ) (h : x^3 = 9261) : (x + 1) * (x - 1) = 440 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_and_difference_of_squares_l1222_122202


namespace NUMINAMATH_GPT_calculator_transform_implication_l1222_122225

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end NUMINAMATH_GPT_calculator_transform_implication_l1222_122225


namespace NUMINAMATH_GPT_S4_equals_15_l1222_122219

noncomputable def S_n (q : ℝ) (n : ℕ) := (1 - q^n) / (1 - q)

theorem S4_equals_15 (q : ℝ) (n : ℕ) (h1 : S_n q 1 = 1) (h2 : S_n q 5 = 5 * S_n q 3 - 4) : 
  S_n q 4 = 15 :=
by
  sorry

end NUMINAMATH_GPT_S4_equals_15_l1222_122219


namespace NUMINAMATH_GPT_hyperbola_a_unique_l1222_122299

-- Definitions from the conditions
def hyperbola (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 4 = 1
def foci (c : ℝ) : Prop := c = 2 * Real.sqrt 3
def a_positive (a : ℝ) : Prop := a > 0

-- Statement to prove
theorem hyperbola_a_unique (a : ℝ) (h : hyperbola 0 0 a ∧ foci (2 * Real.sqrt 3) ∧ a_positive a) : a = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_hyperbola_a_unique_l1222_122299


namespace NUMINAMATH_GPT_quadratic_completing_square_l1222_122298

theorem quadratic_completing_square (b c : ℝ) (h : ∀ x : ℝ, x^2 - 24 * x + 50 = (x + b)^2 + c) :
    b + c = -106 :=
sorry

end NUMINAMATH_GPT_quadratic_completing_square_l1222_122298


namespace NUMINAMATH_GPT_rectangular_park_area_l1222_122257

/-- Define the conditions for the rectangular park -/
def rectangular_park (w l : ℕ) : Prop :=
  l = 3 * w ∧ 2 * (w + l) = 72

/-- Prove that the area of the rectangular park is 243 square meters -/
theorem rectangular_park_area (w l : ℕ) (h : rectangular_park w l) : w * l = 243 := by
  sorry

end NUMINAMATH_GPT_rectangular_park_area_l1222_122257


namespace NUMINAMATH_GPT_sum_of_three_numbers_eq_zero_l1222_122232

theorem sum_of_three_numbers_eq_zero 
  (a b c : ℝ) 
  (h_sorted : a ≤ b ∧ b ≤ c) 
  (h_median : b = 10) 
  (h_mean_least : (a + b + c) / 3 = a + 20) 
  (h_mean_greatest : (a + b + c) / 3 = c - 10) 
  : a + b + c = 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_eq_zero_l1222_122232


namespace NUMINAMATH_GPT_identical_answers_l1222_122208
-- Import necessary libraries

-- Define the entities and conditions
structure Person :=
  (name : String)
  (always_tells_truth : Bool)

def Fyodor : Person := { name := "Fyodor", always_tells_truth := true }
def Sasha : Person := { name := "Sasha", always_tells_truth := false }

def answer (p : Person) : String :=
  if p.always_tells_truth then "Yes" else "No"

-- The theorem statement
theorem identical_answers :
  answer Fyodor = answer Sasha :=
by
  -- Proof steps will be filled in later
  sorry

end NUMINAMATH_GPT_identical_answers_l1222_122208


namespace NUMINAMATH_GPT_express_c_in_terms_of_a_b_l1222_122262

-- Defining the vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)

-- Defining the given vectors
def a := vec 1 1
def b := vec 1 (-1)
def c := vec (-1) 2

-- The statement
theorem express_c_in_terms_of_a_b :
  c = (1/2) • a + (-3/2) • b :=
sorry

end NUMINAMATH_GPT_express_c_in_terms_of_a_b_l1222_122262


namespace NUMINAMATH_GPT_basketball_weight_l1222_122221

theorem basketball_weight (b k : ℝ) (h1 : 6 * b = 4 * k) (h2 : 3 * k = 72) : b = 16 :=
by
  sorry

end NUMINAMATH_GPT_basketball_weight_l1222_122221


namespace NUMINAMATH_GPT_smallest_a_l1222_122255

theorem smallest_a (a b c : ℚ)
  (h1 : a > 0)
  (h2 : b = -2 * a / 3)
  (h3 : c = a / 9 - 5 / 9)
  (h4 : (a + b + c).den = 1) : a = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_l1222_122255


namespace NUMINAMATH_GPT_total_pens_l1222_122295

theorem total_pens (black_pens blue_pens : ℕ) (h1 : black_pens = 4) (h2 : blue_pens = 4) : black_pens + blue_pens = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_pens_l1222_122295


namespace NUMINAMATH_GPT_egg_roll_ratio_l1222_122260

-- Define the conditions as hypotheses 
variables (Matthew_eats Patrick_eats Alvin_eats : ℕ)

-- Define the specific conditions
def conditions : Prop :=
  (Matthew_eats = 6) ∧
  (Patrick_eats = Alvin_eats / 2) ∧
  (Alvin_eats = 4)

-- Define the ratio of Matthew's egg rolls to Patrick's egg rolls
def ratio (a b : ℕ) := a / b

-- State the theorem with the corresponding proof problem
theorem egg_roll_ratio : conditions Matthew_eats Patrick_eats Alvin_eats → ratio Matthew_eats Patrick_eats = 3 :=
by
  -- Proof is not required as mentioned. Adding sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_egg_roll_ratio_l1222_122260


namespace NUMINAMATH_GPT_wrapping_paper_fraction_l1222_122240

theorem wrapping_paper_fraction (s l : ℚ) (h1 : 4 * s + 2 * l = 5 / 12) (h2 : l = 2 * s) :
  s = 5 / 96 ∧ l = 5 / 48 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_fraction_l1222_122240


namespace NUMINAMATH_GPT_shaded_area_percentage_l1222_122224

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_percentage_l1222_122224


namespace NUMINAMATH_GPT_optimal_order_l1222_122296

-- Definition of probabilities
variables (p1 p2 p3 : ℝ) (hp1 : p3 < p1) (hp2 : p1 < p2)

-- The statement to prove
theorem optimal_order (h : p2 > p1) :
  (p2 * (p1 + p3 - p1 * p3)) > (p1 * (p2 + p3 - p2 * p3)) :=
sorry

end NUMINAMATH_GPT_optimal_order_l1222_122296


namespace NUMINAMATH_GPT_cars_on_river_road_l1222_122233

-- Define the given conditions
variables (B C : ℕ)
axiom ratio_condition : B = C / 13
axiom difference_condition : B = C - 60 

-- State the theorem to be proved
theorem cars_on_river_road : C = 65 :=
by
  -- proof would go here 
  sorry

end NUMINAMATH_GPT_cars_on_river_road_l1222_122233


namespace NUMINAMATH_GPT_number_of_types_of_sliced_meat_l1222_122270

-- Define the constants and conditions
def varietyPackCostWithoutRush := 40.00
def rushDeliveryPercentage := 0.30
def costPerTypeWithRush := 13.00
def totalCostWithRush := varietyPackCostWithoutRush + (rushDeliveryPercentage * varietyPackCostWithoutRush)

-- Define the statement that needs to be proven
theorem number_of_types_of_sliced_meat :
  (totalCostWithRush / costPerTypeWithRush) = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_types_of_sliced_meat_l1222_122270


namespace NUMINAMATH_GPT_compute_expression_l1222_122248

theorem compute_expression : 11 * (1 / 17) * 34 = 22 := 
sorry

end NUMINAMATH_GPT_compute_expression_l1222_122248


namespace NUMINAMATH_GPT_correct_transformation_C_l1222_122207

-- Define the conditions as given in the problem
def condition_A (x : ℝ) : Prop := 4 + x = 3 ∧ x = 3 - 4
def condition_B (x : ℝ) : Prop := (1 / 3) * x = 0 ∧ x = 0
def condition_C (y : ℝ) : Prop := 5 * y = -4 * y + 2 ∧ 5 * y + 4 * y = 2
def condition_D (a : ℝ) : Prop := (1 / 2) * a - 1 = 3 * a ∧ a - 2 = 6 * a

-- The theorem to prove that condition_C is correctly transformed
theorem correct_transformation_C : condition_C 1 := 
by sorry

end NUMINAMATH_GPT_correct_transformation_C_l1222_122207


namespace NUMINAMATH_GPT_like_terms_implies_m_minus_n_l1222_122290

/-- If 4x^(2m+2)y^(n-1) and -3x^(3m+1)y^(3n-5) are like terms, then m - n = -1. -/
theorem like_terms_implies_m_minus_n
  (m n : ℤ)
  (h1 : 2 * m + 2 = 3 * m + 1)
  (h2 : n - 1 = 3 * n - 5) :
  m - n = -1 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_implies_m_minus_n_l1222_122290


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l1222_122289

theorem batsman_average_after_12th_innings 
  (A : ℕ) 
  (total_runs_11_innings : ℕ := 11 * A) 
  (new_average : ℕ := A + 2) 
  (total_runs_12_innings : ℕ := total_runs_11_innings + 92) 
  (increased_average_after_12 : 12 * new_average = total_runs_12_innings) 
  : new_average = 70 := 
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l1222_122289


namespace NUMINAMATH_GPT_largest_n_exists_l1222_122211

theorem largest_n_exists :
  ∃ (n : ℕ), 
  (∀ (x y z : ℕ), n^2 = 2*x^2 + 2*y^2 + 2*z^2 + 4*x*y + 4*y*z + 4*z*x + 6*x + 6*y + 6*z - 14) → n = 9 :=
sorry

end NUMINAMATH_GPT_largest_n_exists_l1222_122211


namespace NUMINAMATH_GPT_find_x_l1222_122216

theorem find_x (x : ℕ) 
  (h : (744 + 745 + 747 + 748 + 749 + 752 + 752 + 753 + 755 + x) / 10 = 750) : 
  x = 1255 := 
sorry

end NUMINAMATH_GPT_find_x_l1222_122216


namespace NUMINAMATH_GPT_percent_employed_females_l1222_122288

theorem percent_employed_females (h1 : 96 / 100 > 0) (h2 : 24 / 100 > 0) : 
  (96 - 24) / 96 * 100 = 75 := 
by 
  -- Proof to be filled out
  sorry

end NUMINAMATH_GPT_percent_employed_females_l1222_122288


namespace NUMINAMATH_GPT_sin_pi_minus_alpha_l1222_122223

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 2) : Real.sin (π - α) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_minus_alpha_l1222_122223


namespace NUMINAMATH_GPT_giraffes_count_l1222_122291

def numZebras : ℕ := 12

def numCamels : ℕ := numZebras / 2

def numMonkeys : ℕ := numCamels * 4

def numGiraffes : ℕ := numMonkeys - 22

theorem giraffes_count :
  numGiraffes = 2 :=
by 
  sorry

end NUMINAMATH_GPT_giraffes_count_l1222_122291


namespace NUMINAMATH_GPT_transform_graph_of_g_to_f_l1222_122215

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1

theorem transform_graph_of_g_to_f :
  ∀ (x : ℝ), f x = g (x + (5 * Real.pi) / 12) :=
by
  sorry

end NUMINAMATH_GPT_transform_graph_of_g_to_f_l1222_122215


namespace NUMINAMATH_GPT_smallest_value_m_plus_n_l1222_122234

theorem smallest_value_m_plus_n (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : m + n = 60 :=
sorry

end NUMINAMATH_GPT_smallest_value_m_plus_n_l1222_122234


namespace NUMINAMATH_GPT_no_polynomial_transformation_l1222_122293

-- Define the problem conditions: initial and target sequences
def initial_seq : List ℤ := [-3, -1, 1, 3]
def target_seq : List ℤ := [-3, -1, -3, 3]

-- State the main theorem to be proved
theorem no_polynomial_transformation :
  ¬ (∃ (P : ℤ → ℤ), ∀ x ∈ initial_seq, P x ∈ target_seq) :=
  sorry

end NUMINAMATH_GPT_no_polynomial_transformation_l1222_122293


namespace NUMINAMATH_GPT_compare_abc_l1222_122269

noncomputable def a : ℝ := 2^(4/3)
noncomputable def b : ℝ := 4^(2/5)
noncomputable def c : ℝ := 5^(2/3)

theorem compare_abc : c > a ∧ a > b := 
by
  sorry

end NUMINAMATH_GPT_compare_abc_l1222_122269


namespace NUMINAMATH_GPT_positive_number_square_roots_l1222_122256

theorem positive_number_square_roots (a : ℝ) (x : ℝ) (h1 : x = (a - 7)^2)
  (h2 : x = (2 * a + 1)^2) : x = 25 := by
sorry

end NUMINAMATH_GPT_positive_number_square_roots_l1222_122256


namespace NUMINAMATH_GPT_participants_are_multiple_of_7_l1222_122205

theorem participants_are_multiple_of_7 (P : ℕ) (h1 : P % 2 = 0)
  (h2 : ∀ p, p = P / 2 → P + p / 7 = (4 * P) / 7)
  (h3 : (4 * P) / 7 * 7 = 4 * P) : ∃ k : ℕ, P = 7 * k := 
by
  sorry

end NUMINAMATH_GPT_participants_are_multiple_of_7_l1222_122205


namespace NUMINAMATH_GPT_cycling_problem_l1222_122242

theorem cycling_problem (x : ℚ) (h1 : 25 * x + 15 * (7 - x) = 140) : x = 7 / 2 := 
sorry

end NUMINAMATH_GPT_cycling_problem_l1222_122242


namespace NUMINAMATH_GPT_measure_angle_ACB_l1222_122229

-- Definitions of angles and the conditions
def angle_ABD := 140
def angle_BAC := 105
def supplementary_angle (α β : ℕ) := α + β = 180
def angle_sum_property (α β γ : ℕ) := α + β + γ = 180

-- Theorem to prove the measure of angle ACB
theorem measure_angle_ACB (angle_ABD : ℕ) 
                         (angle_BAC : ℕ) 
                         (h1 : supplementary_angle angle_ABD 40)
                         (h2 : angle_sum_property 40 angle_BAC 35) :
  angle_sum_property 40 105 35 :=
sorry

end NUMINAMATH_GPT_measure_angle_ACB_l1222_122229


namespace NUMINAMATH_GPT_Auston_height_in_cm_l1222_122250

theorem Auston_height_in_cm : 
  (60 : ℝ) * 2.54 = 152.4 :=
by sorry

end NUMINAMATH_GPT_Auston_height_in_cm_l1222_122250


namespace NUMINAMATH_GPT_geometric_sequence_product_l1222_122231

theorem geometric_sequence_product
    (a : ℕ → ℝ)
    (r : ℝ)
    (h₀ : a 1 = 1 / 9)
    (h₃ : a 4 = 3)
    (h_geom : ∀ n, a (n + 1) = a n * r) :
    (a 1) * (a 2) * (a 3) * (a 4) * (a 5) = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1222_122231


namespace NUMINAMATH_GPT_courtyard_width_l1222_122212

theorem courtyard_width 
  (L : ℝ) (N : ℕ) (brick_length brick_width : ℝ) (courtyard_area : ℝ)
  (hL : L = 18)
  (hN : N = 30000)
  (hbrick_length : brick_length = 0.12)
  (hbrick_width : brick_width = 0.06)
  (hcourtyard_area : courtyard_area = (N : ℝ) * (brick_length * brick_width)) :
  (courtyard_area / L) = 12 :=
by
  sorry

end NUMINAMATH_GPT_courtyard_width_l1222_122212


namespace NUMINAMATH_GPT_remainder_of_M_div_by_51_is_zero_l1222_122278

open Nat

noncomputable def M := 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950

theorem remainder_of_M_div_by_51_is_zero :
  M % 51 = 0 :=
sorry

end NUMINAMATH_GPT_remainder_of_M_div_by_51_is_zero_l1222_122278


namespace NUMINAMATH_GPT_find_number_l1222_122297

theorem find_number (x : ℝ) (h : x / 5 + 23 = 42) : x = 95 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_find_number_l1222_122297


namespace NUMINAMATH_GPT_remainder_modulo_l1222_122228

theorem remainder_modulo (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_modulo_l1222_122228


namespace NUMINAMATH_GPT_min_guests_at_banquet_l1222_122230

-- Definitions based on conditions
def total_food : ℕ := 675
def vegetarian_food : ℕ := 195
def pescatarian_food : ℕ := 220
def carnivorous_food : ℕ := 260

def max_vegetarian_per_guest : ℚ := 3
def max_pescatarian_per_guest : ℚ := 2.5
def max_carnivorous_per_guest : ℚ := 4

-- Definition based on the question and the correct answer
def minimum_number_of_guests : ℕ := 218

-- Lean statement to prove the problem
theorem min_guests_at_banquet :
  195 / 3 + 220 / 2.5 + 260 / 4 = 218 :=
by sorry

end NUMINAMATH_GPT_min_guests_at_banquet_l1222_122230


namespace NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l1222_122277

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
  sorry

end NUMINAMATH_GPT_sum_of_seven_consecutive_integers_l1222_122277


namespace NUMINAMATH_GPT_other_root_l1222_122246

theorem other_root (m : ℝ) (h : 1^2 + m*1 + 3 = 0) : 
  ∃ α : ℝ, (1 + α = -m ∧ 1 * α = 3) ∧ α = 3 := 
by 
  sorry

end NUMINAMATH_GPT_other_root_l1222_122246


namespace NUMINAMATH_GPT_minimum_triangle_area_l1222_122272

theorem minimum_triangle_area (r a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a = b) : 
  ∀ T, (T = (a + b) * r / 2) → T = 2 * r * r :=
by 
  sorry

end NUMINAMATH_GPT_minimum_triangle_area_l1222_122272


namespace NUMINAMATH_GPT_solution_existence_l1222_122243

def problem_statement : Prop :=
  ∃ x : ℝ, (0.38 * 80) - (0.12 * x) = 11.2 ∧ x = 160

theorem solution_existence : problem_statement :=
  sorry

end NUMINAMATH_GPT_solution_existence_l1222_122243


namespace NUMINAMATH_GPT_average_death_rate_l1222_122237

-- Definitions of the given conditions
def birth_rate_two_seconds := 10
def net_increase_one_day := 345600
def seconds_per_day := 24 * 60 * 60 

-- Define the theorem to be proven
theorem average_death_rate :
  (birth_rate_two_seconds / 2) - (net_increase_one_day / seconds_per_day) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_average_death_rate_l1222_122237


namespace NUMINAMATH_GPT_soda_cost_l1222_122227

variable {b s f : ℕ}

theorem soda_cost :
    5 * b + 3 * s + 2 * f = 520 ∧
    3 * b + 2 * s + f = 340 →
    s = 80 :=
by
  sorry

end NUMINAMATH_GPT_soda_cost_l1222_122227


namespace NUMINAMATH_GPT_day_53_days_from_thursday_is_monday_l1222_122264

def day_of_week : Type := {n : ℤ // n % 7 = n}

def Thursday : day_of_week := ⟨4, by norm_num⟩
def Monday : day_of_week := ⟨1, by norm_num⟩

theorem day_53_days_from_thursday_is_monday : 
  (⟨(4 + 53) % 7, by norm_num⟩ : day_of_week) = Monday := 
by 
  sorry

end NUMINAMATH_GPT_day_53_days_from_thursday_is_monday_l1222_122264


namespace NUMINAMATH_GPT_closest_point_on_line_l1222_122213

open Real

theorem closest_point_on_line (x y : ℝ) (h_line : y = 4 * x - 3) (h_closest : ∀ p : ℝ × ℝ, (p.snd - -1)^2 + (p.fst - 2)^2 ≥ (y - -1)^2 + (x - 2)^2) :
  x = 10 / 17 ∧ y = 31 / 17 :=
sorry

end NUMINAMATH_GPT_closest_point_on_line_l1222_122213


namespace NUMINAMATH_GPT_complete_square_solution_l1222_122266

-- Define the initial equation 
def equation_to_solve (x : ℝ) : Prop := x^2 - 4 * x = 6

-- Define the transformed equation after completing the square
def transformed_equation (x : ℝ) : Prop := (x - 2)^2 = 10

-- Prove that solving the initial equation using completing the square results in the transformed equation
theorem complete_square_solution : 
  ∀ x : ℝ, equation_to_solve x → transformed_equation x := 
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_complete_square_solution_l1222_122266


namespace NUMINAMATH_GPT_algebraic_identity_l1222_122201

theorem algebraic_identity (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2001 = -2000 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_l1222_122201


namespace NUMINAMATH_GPT_monotonic_intervals_l1222_122252

noncomputable def y : ℝ → ℝ := λ x => x * Real.log x

theorem monotonic_intervals :
  (∀ x : ℝ, 0 < x → x < (1 / Real.exp 1) → y x < -1) ∧ 
  (∀ x : ℝ, (1 / Real.exp 1) < x → x < 5 → y x > 1) := 
by
  sorry -- Proof goes here.

end NUMINAMATH_GPT_monotonic_intervals_l1222_122252


namespace NUMINAMATH_GPT_number_of_bricks_l1222_122247

noncomputable def bricklayer_one_hours : ℝ := 8
noncomputable def bricklayer_two_hours : ℝ := 12
noncomputable def reduction_rate : ℝ := 12
noncomputable def combined_hours : ℝ := 6

theorem number_of_bricks (y : ℝ) :
  ((combined_hours * ((y / bricklayer_one_hours) + (y / bricklayer_two_hours) - reduction_rate)) = y) →
  y = 288 :=
by sorry

end NUMINAMATH_GPT_number_of_bricks_l1222_122247


namespace NUMINAMATH_GPT_function_takes_negative_values_l1222_122204

def f (x a : ℝ) : ℝ := x^2 - a * x + 1

theorem function_takes_negative_values {a : ℝ} :
  (∃ x : ℝ, f x a < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end NUMINAMATH_GPT_function_takes_negative_values_l1222_122204


namespace NUMINAMATH_GPT_part1_part2_l1222_122292

noncomputable def U : Set ℝ := Set.univ

noncomputable def A (a: ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a: ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part1 (a : ℝ) (ha : a = 1/2) :
  (U \ (B a)) ∩ (A a) = {x | 9/4 ≤ x ∧ x < 5/2} :=
sorry

theorem part2 (p q : ℝ → Prop)
  (hp : ∀ x, p x → x ∈ A a) (hq : ∀ x, q x → x ∈ B a)
  (hq_necessary : ∀ x, p x → q x) :
  -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1222_122292


namespace NUMINAMATH_GPT_tan_double_angle_l1222_122283

theorem tan_double_angle (θ : ℝ) 
  (h1 : Real.sin θ = 4 / 5) 
  (h2 : Real.sin θ - Real.cos θ > 1) : 
  Real.tan (2 * θ) = 24 / 7 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_l1222_122283


namespace NUMINAMATH_GPT_domain_of_f_l1222_122274

def condition1 (x : ℝ) : Prop := 4 - |x| ≥ 0
def condition2 (x : ℝ) : Prop := (x^2 - 5 * x + 6) / (x - 3) > 0

theorem domain_of_f (x : ℝ) :
  (condition1 x) ∧ (condition2 x) ↔ ((2 < x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1222_122274


namespace NUMINAMATH_GPT_lemon_juice_fraction_l1222_122279

theorem lemon_juice_fraction :
  ∃ L : ℚ, 30 - 30 * L - (1 / 3) * (30 - 30 * L) = 6 ∧ L = 7 / 10 :=
sorry

end NUMINAMATH_GPT_lemon_juice_fraction_l1222_122279


namespace NUMINAMATH_GPT_simplest_quadratic_radicals_same_type_l1222_122222

theorem simplest_quadratic_radicals_same_type (m n : ℕ)
  (h : ∀ {a : ℕ}, (a = m - 1 → a = 2) ∧ (a = 4 * n - 1 → a = 7)) :
  m + n = 5 :=
sorry

end NUMINAMATH_GPT_simplest_quadratic_radicals_same_type_l1222_122222


namespace NUMINAMATH_GPT_positive_difference_two_solutions_abs_eq_15_l1222_122259

theorem positive_difference_two_solutions_abs_eq_15 :
  ∀ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 > x2) → (x1 - x2 = 30) :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_positive_difference_two_solutions_abs_eq_15_l1222_122259


namespace NUMINAMATH_GPT_find_mistaken_divisor_l1222_122218

-- Define the conditions
def remainder : ℕ := 0
def quotient_correct : ℕ := 32
def divisor_correct : ℕ := 21
def quotient_mistaken : ℕ := 56
def dividend : ℕ := quotient_correct * divisor_correct + remainder

-- Prove the mistaken divisor
theorem find_mistaken_divisor : ∃ x : ℕ, dividend = quotient_mistaken * x + remainder ∧ x = 12 :=
by
  -- We leave this as an exercise to the prover
  sorry

end NUMINAMATH_GPT_find_mistaken_divisor_l1222_122218


namespace NUMINAMATH_GPT_incorrect_fraction_addition_l1222_122214

theorem incorrect_fraction_addition (a b x y : ℤ) (h1 : 0 < b) (h2 : 0 < y) (h3 : (a + x) * (b * y) = (a * y + b * x) * (b + y)) :
  ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by
  sorry

end NUMINAMATH_GPT_incorrect_fraction_addition_l1222_122214


namespace NUMINAMATH_GPT_max_sum_abs_coeff_l1222_122220

theorem max_sum_abs_coeff (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : |f 1| ≤ 1)
  (h3 : |f (1/2)| ≤ 1)
  (h4 : |f 0| ≤ 1) :
  |a| + |b| + |c| ≤ 17 :=
sorry

end NUMINAMATH_GPT_max_sum_abs_coeff_l1222_122220


namespace NUMINAMATH_GPT_smallest_sum_3x3_grid_l1222_122226

-- Define the given conditions
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9] -- List of numbers used in the grid
def total_sum : ℕ := 45 -- Total sum of numbers from 1 to 9
def grid_size : ℕ := 3 -- Size of the grid
def corners_ids : List Nat := [0, 2, 6, 8] -- Indices of the corners in the grid
def remaining_sum : ℕ := 25 -- Sum of the remaining 5 numbers (after excluding the corners)

-- Define the goal: Prove that the smallest sum s is achieved
theorem smallest_sum_3x3_grid : ∃ s : ℕ, 
  (∀ (r : Fin grid_size) (c : Fin grid_size),
    r + c = s) → (s = 12) :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_3x3_grid_l1222_122226


namespace NUMINAMATH_GPT_Sergey_full_years_l1222_122281

def full_years (years months weeks days hours : ℕ) : ℕ :=
  years + months / 12 + (weeks * 7 + days) / 365

theorem Sergey_full_years 
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  years = 36 →
  months = 36 →
  weeks = 36 →
  days = 36 →
  hours = 36 →
  full_years years months weeks days hours = 39 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Sergey_full_years_l1222_122281


namespace NUMINAMATH_GPT_sphere_surface_area_ratios_l1222_122210

theorem sphere_surface_area_ratios
  (s : ℝ)
  (r1 : ℝ)
  (r2 : ℝ)
  (r3 : ℝ)
  (h1 : r1 = s / 4 * Real.sqrt 6)
  (h2 : r2 = s / 4 * Real.sqrt 2)
  (h3 : r3 = s / 12 * Real.sqrt 6) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r3^2) = 9 ∧
  (4 * Real.pi * r2^2) / (4 * Real.pi * r3^2) = 3 ∧
  (4 * Real.pi * r3^2) / (4 * Real.pi * r3^2) = 1 := 
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_ratios_l1222_122210


namespace NUMINAMATH_GPT_smallest_b_for_perfect_square_l1222_122271

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ (∃ (n : ℤ), 4 * b + 5 = n ^ 2) ∧ b = 5 := 
sorry

end NUMINAMATH_GPT_smallest_b_for_perfect_square_l1222_122271


namespace NUMINAMATH_GPT_season_duration_l1222_122268

theorem season_duration (total_games : ℕ) (games_per_month : ℕ) (h1 : total_games = 323) (h2 : games_per_month = 19) :
  (total_games / games_per_month) = 17 :=
by
  sorry

end NUMINAMATH_GPT_season_duration_l1222_122268


namespace NUMINAMATH_GPT_number_of_three_leaf_clovers_l1222_122253

theorem number_of_three_leaf_clovers (total_leaves : ℕ) (three_leaf_clover : ℕ) (four_leaf_clover : ℕ) (n : ℕ)
  (h1 : total_leaves = 40) (h2 : three_leaf_clover = 3) (h3 : four_leaf_clover = 4) (h4: total_leaves = 3 * n + 4) :
  n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_three_leaf_clovers_l1222_122253


namespace NUMINAMATH_GPT_ratio_shirt_to_coat_l1222_122261

-- Define the given conditions
def total_cost := 600
def shirt_cost := 150

-- Define the coat cost based on the given conditions
def coat_cost := total_cost - shirt_cost

-- State the theorem to prove the ratio of shirt cost to coat cost is 1:3
theorem ratio_shirt_to_coat : (shirt_cost : ℚ) / (coat_cost : ℚ) = 1 / 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ratio_shirt_to_coat_l1222_122261


namespace NUMINAMATH_GPT_solution_l1222_122284

-- Define the discount conditions
def discount (price : ℕ) : ℕ :=
  if price > 22 then price * 7 / 10 else
  if price < 20 then price * 8 / 10 else
  price

-- Define the given book prices
def book_prices : List ℕ := [25, 18, 21, 35, 12, 10]

-- Calculate total cost using the discount function
def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (λ acc price => acc + discount price) 0

def problem_statement : Prop :=
  total_cost book_prices = 95

theorem solution : problem_statement :=
  by
  unfold problem_statement
  unfold total_cost
  simp [book_prices, discount]
  sorry

end NUMINAMATH_GPT_solution_l1222_122284


namespace NUMINAMATH_GPT_algebra_expr_value_l1222_122254

theorem algebra_expr_value (x y : ℝ) (h : x - 2 * y = 3) : 4 * y + 1 - 2 * x = -5 :=
sorry

end NUMINAMATH_GPT_algebra_expr_value_l1222_122254


namespace NUMINAMATH_GPT_appropriate_speech_length_l1222_122282

-- Condition 1: Speech duration in minutes
def speech_duration_min : ℝ := 30
def speech_duration_max : ℝ := 45

-- Condition 2: Ideal rate of speech in words per minute
def ideal_rate : ℝ := 150

-- Question translated into Lean proof statement
theorem appropriate_speech_length (n : ℝ) (h : n = 5650) :
  speech_duration_min * ideal_rate ≤ n ∧ n ≤ speech_duration_max * ideal_rate :=
by
  sorry

end NUMINAMATH_GPT_appropriate_speech_length_l1222_122282


namespace NUMINAMATH_GPT_ultramen_defeat_monster_in_5_minutes_l1222_122285

theorem ultramen_defeat_monster_in_5_minutes :
  ∀ (attacksRequired : ℕ) (attackRate1 attackRate2 : ℕ),
    (attacksRequired = 100) →
    (attackRate1 = 12) →
    (attackRate2 = 8) →
    (attacksRequired / (attackRate1 + attackRate2) = 5) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ultramen_defeat_monster_in_5_minutes_l1222_122285


namespace NUMINAMATH_GPT_remainder_when_dividing_698_by_13_is_9_l1222_122217

theorem remainder_when_dividing_698_by_13_is_9 :
  ∃ k m : ℤ, 242 = k * 13 + 8 ∧
             698 = m * 13 + 9 ∧
             (k + m) * 13 + 4 = 940 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_when_dividing_698_by_13_is_9_l1222_122217


namespace NUMINAMATH_GPT_candy_pieces_per_package_l1222_122238

theorem candy_pieces_per_package (packages_gum : ℕ) (packages_candy : ℕ) (total_candies : ℕ) :
  packages_gum = 21 →
  packages_candy = 45 →
  total_candies = 405 →
  total_candies / packages_candy = 9 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_candy_pieces_per_package_l1222_122238


namespace NUMINAMATH_GPT_total_volume_calculation_l1222_122235

noncomputable def total_volume_of_four_cubes (edge_length_in_feet : ℝ) (conversion_factor : ℝ) : ℝ :=
  let edge_length_in_meters := edge_length_in_feet * conversion_factor
  let volume_of_one_cube := edge_length_in_meters^3
  4 * volume_of_one_cube

theorem total_volume_calculation :
  total_volume_of_four_cubes 5 0.3048 = 14.144 :=
by
  -- Proof needs to be filled in.
  sorry

end NUMINAMATH_GPT_total_volume_calculation_l1222_122235


namespace NUMINAMATH_GPT_range_of_a_l1222_122203

theorem range_of_a (a : ℝ) (h : a > 0) :
  let A := {x : ℝ | x^2 + 2 * x - 8 > 0}
  let B := {x : ℝ | x^2 - 2 * a * x + 4 ≤ 0}
  (∃! x : ℤ, (x : ℝ) ∈ A ∩ B) → (13 / 6 ≤ a ∧ a < 5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1222_122203


namespace NUMINAMATH_GPT_leak_out_time_l1222_122275

theorem leak_out_time (T_A T_full : ℝ) (h1 : T_A = 16) (h2 : T_full = 80) :
  ∃ T_B : ℝ, (1 / T_A - 1 / T_B = 1 / T_full) ∧ T_B = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_leak_out_time_l1222_122275
