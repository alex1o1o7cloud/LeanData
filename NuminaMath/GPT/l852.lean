import Mathlib

namespace NUMINAMATH_GPT_math_or_sci_but_not_both_l852_85245

-- Definitions of the conditions
variable (students_math_and_sci : ℕ := 15)
variable (students_math : ℕ := 30)
variable (students_only_sci : ℕ := 18)

-- The theorem to prove
theorem math_or_sci_but_not_both :
  (students_math - students_math_and_sci) + students_only_sci = 33 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_math_or_sci_but_not_both_l852_85245


namespace NUMINAMATH_GPT_L_shape_perimeter_correct_l852_85209

-- Define the dimensions of the rectangles
def rect_height : ℕ := 3
def rect_width : ℕ := 4

-- Define the combined shape and perimeter calculation
def L_shape_perimeter (h w : ℕ) : ℕ := (2 * w) + (2 * h)

theorem L_shape_perimeter_correct : 
  L_shape_perimeter rect_height rect_width = 14 := 
  sorry

end NUMINAMATH_GPT_L_shape_perimeter_correct_l852_85209


namespace NUMINAMATH_GPT_within_acceptable_range_l852_85277

def flour_weight : ℝ := 25.18
def flour_label : ℝ := 25
def tolerance : ℝ := 0.25

theorem within_acceptable_range  :
  (flour_label - tolerance) ≤ flour_weight ∧ flour_weight ≤ (flour_label + tolerance) :=
by
  sorry

end NUMINAMATH_GPT_within_acceptable_range_l852_85277


namespace NUMINAMATH_GPT_max_value_fn_l852_85223

theorem max_value_fn : ∀ x : ℝ, y = 1 / (|x| + 2) → 
  ∃ y : ℝ, y = 1 / 2 ∧ ∀ x : ℝ, 1 / (|x| + 2) ≤ y :=
sorry

end NUMINAMATH_GPT_max_value_fn_l852_85223


namespace NUMINAMATH_GPT_min_val_l852_85257

theorem min_val (x y : ℝ) (h : x + 2 * y = 1) : 2^x + 4^y = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_val_l852_85257


namespace NUMINAMATH_GPT_blue_pens_count_l852_85202

variable (redPenCost bluePenCost totalCost totalPens : ℕ)
variable (numRedPens numBluePens : ℕ)

-- Conditions
axiom PriceOfRedPen : redPenCost = 5
axiom PriceOfBluePen : bluePenCost = 7
axiom TotalCost : totalCost = 102
axiom TotalPens : totalPens = 16
axiom PenCount : numRedPens + numBluePens = totalPens
axiom CostEquation : redPenCost * numRedPens + bluePenCost * numBluePens = totalCost

theorem blue_pens_count : numBluePens = 11 :=
by
  sorry

end NUMINAMATH_GPT_blue_pens_count_l852_85202


namespace NUMINAMATH_GPT_subset_implies_value_l852_85201

theorem subset_implies_value (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 3, 2*m-1}) (hB : B = {3, m}) (hSub : B ⊆ A) : 
  m = -1 ∨ m = 1 := by
  sorry

end NUMINAMATH_GPT_subset_implies_value_l852_85201


namespace NUMINAMATH_GPT_rose_bought_flowers_l852_85240

theorem rose_bought_flowers (F : ℕ) (h1 : ∃ (daisies tulips sunflowers : ℕ), daisies = 2 ∧ sunflowers = 4 ∧ 
  tulips = (3 / 5) * (F - 2) ∧ sunflowers = (2 / 5) * (F - 2)) : F = 12 :=
sorry

end NUMINAMATH_GPT_rose_bought_flowers_l852_85240


namespace NUMINAMATH_GPT_distinct_weights_count_l852_85265

theorem distinct_weights_count (n : ℕ) (h : n = 4) : 
  -- Given four weights and a two-pan balance scale without a pointer,
  ∃ m : ℕ, 
  -- prove that the number of distinct weights of cargo
  (m = 40) ∧  
  -- that can be exactly measured if the weights can be placed on both pans of the scale is 40.
  m = 3^n - 1 ∧ (m / 2 = 40) := by
  sorry

end NUMINAMATH_GPT_distinct_weights_count_l852_85265


namespace NUMINAMATH_GPT_supermarket_sales_l852_85249

theorem supermarket_sales (S_Dec : ℝ) (S_Jan : ℝ) (S_Feb : ℝ) (S_Jan_eq : S_Jan = S_Dec * (1 + x))
  (S_Feb_eq : S_Feb = S_Jan * (1 + x))
  (inc_eq : S_Feb = S_Dec + 0.24 * S_Dec) :
  x = 0.2 ∧ S_Feb = S_Dec * (1 + 0.2)^2 := by
sorry

end NUMINAMATH_GPT_supermarket_sales_l852_85249


namespace NUMINAMATH_GPT_minimum_product_OP_OQ_l852_85239

theorem minimum_product_OP_OQ (a b : ℝ) (P Q : ℝ × ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : P ≠ Q) (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1) (h5 : Q.1 ^ 2 / a ^ 2 + Q.2 ^ 2 / b ^ 2 = 1)
  (h6 : P.1 * Q.1 + P.2 * Q.2 = 0) :
  (P.1 ^ 2 + P.2 ^ 2) * (Q.1 ^ 2 + Q.2 ^ 2) ≥ (2 * a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2)) :=
by sorry

end NUMINAMATH_GPT_minimum_product_OP_OQ_l852_85239


namespace NUMINAMATH_GPT_intersection_A_B_find_coefficients_a_b_l852_85236

open Set

variable {X : Type} (x : X)

def setA : Set ℝ := { x | x^2 < 9 }
def setB : Set ℝ := { x | (x - 2) * (x + 4) < 0 }
def A_inter_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def A_union_B_solution_set : Set ℝ := { x | -4 < x ∧ x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | -3 < x ∧ x < 2 } :=
sorry

theorem find_coefficients_a_b (a b : ℝ) :
  (∀ x, 2 * x^2 + a * x + b < 0 ↔ -4 < x ∧ x < 3) → 
  a = 2 ∧ b = -24 :=
sorry

end NUMINAMATH_GPT_intersection_A_B_find_coefficients_a_b_l852_85236


namespace NUMINAMATH_GPT_combined_girls_avg_l852_85204

variables (A a B b : ℕ) -- Number of boys and girls at Adams and Baker respectively.
variables (avgBoysAdams avgGirlsAdams avgAdams avgBoysBaker avgGirlsBaker avgBaker : ℚ)

-- Conditions
def avgAdamsBoys := 72
def avgAdamsGirls := 78
def avgAdamsCombined := 75
def avgBakerBoys := 84
def avgBakerGirls := 91
def avgBakerCombined := 85
def combinedAvgBoys := 80

-- Equations derived from the problem statement
def equations : Prop :=
  (72 * A + 78 * a) / (A + a) = 75 ∧
  (84 * B + 91 * b) / (B + b) = 85 ∧
  (72 * A + 84 * B) / (A + B) = 80

-- The goal is to show the combined average score of girls
def combinedAvgGirls := 85

theorem combined_girls_avg (h : equations A a B b):
  (78 * (6 * b / 7) + 91 * b) / ((6 * b / 7) + b) = 85 := by
  sorry

end NUMINAMATH_GPT_combined_girls_avg_l852_85204


namespace NUMINAMATH_GPT_find_m_if_parallel_l852_85238

-- Definitions of the lines and the condition for parallel lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m - 1) * x + y + 2
def line2 (m : ℝ) (x y : ℝ) : ℝ := 8 * x + (m + 1) * y + (m - 1)

-- The condition for the lines to be parallel
def parallel (m : ℝ) : Prop :=
  (m - 1) / 8 = 1 / (m + 1) ∧ (m - 1) / 8 ≠ 2 / (m - 1)

-- The main theorem to prove
theorem find_m_if_parallel (m : ℝ) (h : parallel m) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_if_parallel_l852_85238


namespace NUMINAMATH_GPT_solve_for_3x2_plus_6_l852_85295

theorem solve_for_3x2_plus_6 (x : ℚ) (h : 5 * x + 3 = 2 * x - 4) : 3 * (x^2 + 6) = 103 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_3x2_plus_6_l852_85295


namespace NUMINAMATH_GPT_solving_equation_l852_85208

theorem solving_equation (x : ℝ) : 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := 
by
  sorry

end NUMINAMATH_GPT_solving_equation_l852_85208


namespace NUMINAMATH_GPT_geometric_series_sum_l852_85263

theorem geometric_series_sum (a : ℝ) (q : ℝ) (a₁ : ℝ) 
  (h1 : a₁ = 1)
  (h2 : q = a - (3/2))
  (h3 : |q| < 1)
  (h4 : a = a₁ / (1 - q)) :
  a = 2 :=
sorry

end NUMINAMATH_GPT_geometric_series_sum_l852_85263


namespace NUMINAMATH_GPT_problem_solution_l852_85222

lemma factor_def (m n : ℕ) : n ∣ m ↔ ∃ k, m = n * k := by sorry

def is_true_A : Prop := 4 ∣ 24
def is_true_B : Prop := 19 ∣ 209 ∧ ¬ (19 ∣ 63)
def is_true_C : Prop := ¬ (30 ∣ 90) ∧ ¬ (30 ∣ 65)
def is_true_D : Prop := 11 ∣ 33 ∧ ¬ (11 ∣ 77)
def is_true_E : Prop := 9 ∣ 180

theorem problem_solution : (is_true_A ∧ is_true_B ∧ is_true_E) ∧ ¬(is_true_C) ∧ ¬(is_true_D) :=
  by sorry

end NUMINAMATH_GPT_problem_solution_l852_85222


namespace NUMINAMATH_GPT_solve_for_x_l852_85237

theorem solve_for_x (x : ℤ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 9) : x = 72 / 23 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l852_85237


namespace NUMINAMATH_GPT_non_empty_set_A_l852_85227

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {x | x ^ 2 = a}

-- Theorem statement
theorem non_empty_set_A (a : ℝ) (h : (A a).Nonempty) : 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_non_empty_set_A_l852_85227


namespace NUMINAMATH_GPT_solve_for_m_l852_85259

theorem solve_for_m (x m : ℝ) (h : (∃ x, (x - 1) / (x - 4) = m / (x - 4))): 
  m = 3 :=
by {
  sorry -- placeholder to indicate where the proof would go
}

end NUMINAMATH_GPT_solve_for_m_l852_85259


namespace NUMINAMATH_GPT_find_WZ_length_l852_85203

noncomputable def WZ_length (XY YZ XZ WX : ℝ) (theta : ℝ) : ℝ :=
  Real.sqrt ((WX^2 + XZ^2 - 2 * WX * XZ * (-1 / 2)))

-- Define the problem within the context of the provided lengths and condition
theorem find_WZ_length :
  WZ_length 3 5 7 8.5 (-1 / 2) = Real.sqrt 180.75 :=
by 
  -- This "by sorry" is used to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_find_WZ_length_l852_85203


namespace NUMINAMATH_GPT_fraction_white_surface_area_l852_85241

/-- A 4-inch cube is constructed from 64 smaller cubes, each with 1-inch edges.
   48 of these smaller cubes are colored red and 16 are colored white.
   Prove that if the 4-inch cube is constructed to have the smallest possible white surface area showing,
   the fraction of the white surface area is 1/12. -/
theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let white_cubes := 16
  let exposed_white_surface_area := 8
  (exposed_white_surface_area / total_surface_area) = (1 / 12) := 
  sorry

end NUMINAMATH_GPT_fraction_white_surface_area_l852_85241


namespace NUMINAMATH_GPT_area_of_rhombus_enclosed_by_equation_l852_85271

-- Given the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- Define the main theorem to be proven
theorem area_of_rhombus_enclosed_by_equation : 
  (∃ x y : ℝ, equation x y) → ∃ area : ℝ, area = 384 :=
by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_enclosed_by_equation_l852_85271


namespace NUMINAMATH_GPT_g_10_equals_100_l852_85248

-- Define the function g and the conditions it must satisfy.
def g : ℕ → ℝ := sorry

axiom g_2 : g 2 = 4

axiom g_condition : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

-- Prove the required statement.
theorem g_10_equals_100 : g 10 = 100 :=
by sorry

end NUMINAMATH_GPT_g_10_equals_100_l852_85248


namespace NUMINAMATH_GPT_solve_for_x_l852_85233

theorem solve_for_x (y z x : ℝ) (h1 : 2 / 3 = y / 90) (h2 : 2 / 3 = (y + z) / 120) (h3 : 2 / 3 = (x - z) / 150) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l852_85233


namespace NUMINAMATH_GPT_graph_of_cubic_equation_is_three_lines_l852_85274

theorem graph_of_cubic_equation_is_three_lines (x y : ℝ) :
  (x + y) ^ 3 = x ^ 3 + y ^ 3 →
  (y = -x ∨ x = 0 ∨ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_cubic_equation_is_three_lines_l852_85274


namespace NUMINAMATH_GPT_cyclic_trapezoid_radii_relation_l852_85217

variables (A B C D O : Type)
variables (AD BC : Type)
variables (r1 r2 r3 r4 : ℝ)

-- Conditions
def cyclic_trapezoid (A B C D: Type) (AD BC: Type): Prop := sorry
def intersection (A B C D O : Type): Prop := sorry
def radius_incircle (triangle : Type) (radius : ℝ): Prop := sorry

theorem cyclic_trapezoid_radii_relation
  (h1: cyclic_trapezoid A B C D AD BC)
  (h2: intersection A B C D O)
  (hr1: radius_incircle AOD r1)
  (hr2: radius_incircle AOB r2)
  (hr3: radius_incircle BOC r3)
  (hr4: radius_incircle COD r4):
  (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4) :=
sorry

end NUMINAMATH_GPT_cyclic_trapezoid_radii_relation_l852_85217


namespace NUMINAMATH_GPT_parabola_transform_l852_85269

theorem parabola_transform :
  ∀ (x : ℝ),
    ∃ (y : ℝ),
      (y = -2 * x^2) →
      (∃ (y' : ℝ), y' = y - 1 ∧
      ∃ (x' : ℝ), x' = x - 3 ∧
      ∃ (y'' : ℝ), y'' = -2 * (x')^2 - 1) :=
by sorry

end NUMINAMATH_GPT_parabola_transform_l852_85269


namespace NUMINAMATH_GPT_number_of_games_between_men_and_women_l852_85220

theorem number_of_games_between_men_and_women
    (W M : ℕ)
    (hW : W * (W - 1) / 2 = 72)
    (hM : M * (M - 1) / 2 = 288) :
  M * W = 288 :=
by
  sorry

end NUMINAMATH_GPT_number_of_games_between_men_and_women_l852_85220


namespace NUMINAMATH_GPT_area_of_given_field_l852_85268

noncomputable def area_of_field (cost_in_rupees : ℕ) (rate_per_meter_in_paise : ℕ) (ratio_width : ℕ) (ratio_length : ℕ) : ℕ :=
  let cost_in_paise := cost_in_rupees * 100
  let perimeter := (ratio_width + ratio_length) * 2
  let x := cost_in_paise / (perimeter * rate_per_meter_in_paise)
  let width := ratio_width * x
  let length := ratio_length * x
  width * length

theorem area_of_given_field :
  let cost_in_rupees := 105
  let rate_per_meter_in_paise := 25
  let ratio_width := 3
  let ratio_length := 4
  area_of_field cost_in_rupees rate_per_meter_in_paise ratio_width ratio_length = 10800 :=
by
  sorry

end NUMINAMATH_GPT_area_of_given_field_l852_85268


namespace NUMINAMATH_GPT_total_weekly_cups_brewed_l852_85267

-- Define the given conditions
def weekday_cups_per_hour : ℕ := 10
def weekend_total_cups : ℕ := 120
def shop_open_hours_per_day : ℕ := 5
def weekdays_in_week : ℕ := 5

-- Prove the total number of coffee cups brewed in one week
theorem total_weekly_cups_brewed : 
  (weekday_cups_per_hour * shop_open_hours_per_day * weekdays_in_week) 
  + weekend_total_cups = 370 := 
by
  sorry

end NUMINAMATH_GPT_total_weekly_cups_brewed_l852_85267


namespace NUMINAMATH_GPT_distance_to_office_is_18_l852_85294

-- Definitions given in the problem conditions
variables (x t d : ℝ)
-- Conditions based on the problem statements
axiom speed_condition1 : d = x * t
axiom speed_condition2 : d = (x + 1) * (3 / 4 * t)
axiom speed_condition3 : d = (x - 1) * (t + 3)

-- The mathematical proof statement that needs to be shown
theorem distance_to_office_is_18 :
  d = 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_office_is_18_l852_85294


namespace NUMINAMATH_GPT_find_extreme_values_find_m_range_for_zeros_l852_85231

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x + 2

theorem find_extreme_values (m : ℝ) :
  (∀ x > 0, m ≤ 0 → (f x m ≠ 0 ∨ ∀ y > 0, f y m ≥ f x m ∨ f y m ≤ f x m)) ∧
  (∀ x > 0, m > 0 → ∃ x_max, x_max = 1 / m ∧ ∀ y > 0, f y m ≤ f x_max m) := 
sorry

theorem find_m_range_for_zeros (m : ℝ) :
  (∃ a b, a = 1 / Real.exp 2 ∧ b = Real.exp 1 ∧ (f a m = 0 ∧ f b m = 0)) ↔ 
  (m ≥ 3 / Real.exp 1 ∧ m < Real.exp 1) :=
sorry

end NUMINAMATH_GPT_find_extreme_values_find_m_range_for_zeros_l852_85231


namespace NUMINAMATH_GPT_large_rectangle_perimeter_l852_85273

-- Definitions for conditions
def rectangle_area (l b : ℝ) := l * b
def is_large_rectangle_perimeter (l b perimeter : ℝ) := perimeter = 2 * (l + b)

-- Statement of the theorem
theorem large_rectangle_perimeter :
  ∃ (l b : ℝ), rectangle_area l b = 8 ∧ 
               (∀ l_rect b_rect: ℝ, is_large_rectangle_perimeter l_rect b_rect 32) :=
by
  sorry

end NUMINAMATH_GPT_large_rectangle_perimeter_l852_85273


namespace NUMINAMATH_GPT_minimum_length_intersection_l852_85210

def length (a b : ℝ) : ℝ := b - a

def M (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2/3 }
def N (n : ℝ) : Set ℝ := { x | n - 1/2 ≤ x ∧ x ≤ n }

def IntervalSet : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem minimum_length_intersection (m n : ℝ) (hM : M m ⊆ IntervalSet) (hN : N n ⊆ IntervalSet) :
  length (max m (n - 1/2)) (min (m + 2/3) n) = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_length_intersection_l852_85210


namespace NUMINAMATH_GPT_integer_solution_of_inequalities_l852_85284

theorem integer_solution_of_inequalities :
  (∀ x : ℝ, 3 * x - 4 ≤ 6 * x - 2 → (2 * x + 1) / 3 - 1 < (x - 1) / 2 → (x = 0)) :=
sorry

end NUMINAMATH_GPT_integer_solution_of_inequalities_l852_85284


namespace NUMINAMATH_GPT_frog_climbing_time_is_correct_l852_85262

noncomputable def frog_climb_out_time : Nat :=
  let well_depth := 12
  let climb_up := 3
  let slip_down := 1
  let net_gain := climb_up - slip_down
  let total_cycles := (well_depth - 3) / net_gain + 1
  let total_time := total_cycles * 3
  let extra_time := 6
  total_time + extra_time

theorem frog_climbing_time_is_correct :
  frog_climb_out_time = 22 := by
  sorry

end NUMINAMATH_GPT_frog_climbing_time_is_correct_l852_85262


namespace NUMINAMATH_GPT_visitors_current_day_l852_85256

-- Define the number of visitors on the previous day and the additional visitors
def v_prev : ℕ := 600
def v_add : ℕ := 61

-- Prove that the number of visitors on the current day is 661
theorem visitors_current_day : v_prev + v_add = 661 :=
by
  sorry

end NUMINAMATH_GPT_visitors_current_day_l852_85256


namespace NUMINAMATH_GPT_prob_first_question_correct_is_4_5_distribution_of_X_l852_85266

-- Assume probabilities for member A and member B answering correctly.
def prob_A_correct : ℚ := 2 / 5
def prob_B_correct : ℚ := 2 / 3

def prob_A_incorrect : ℚ := 1 - prob_A_correct
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- Given that A answers first, followed by B.
-- Calculate the probability that the first team answers the first question correctly.
def prob_first_question_correct : ℚ :=
  prob_A_correct + (prob_A_incorrect * prob_B_correct)

-- Assert that the calculated probability is equal to 4/5
theorem prob_first_question_correct_is_4_5 :
  prob_first_question_correct = 4 / 5 := by
  sorry

-- Define the possible scores and their probabilities
def prob_X_eq_0 : ℚ := prob_A_incorrect * prob_B_incorrect
def prob_X_eq_10 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_20 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 2 * prob_A_incorrect * prob_B_incorrect
def prob_X_eq_30 : ℚ := (prob_A_correct + prob_A_incorrect * prob_B_correct) ^ 3

-- Assert the distribution probabilities for the random variable X
theorem distribution_of_X :
  prob_X_eq_0 = 1 / 5 ∧
  prob_X_eq_10 = 4 / 25 ∧
  prob_X_eq_20 = 16 / 125 ∧
  prob_X_eq_30 = 64 / 125 := by
  sorry

end NUMINAMATH_GPT_prob_first_question_correct_is_4_5_distribution_of_X_l852_85266


namespace NUMINAMATH_GPT_cats_owners_percentage_l852_85244

noncomputable def percentage_of_students_owning_cats (total_students : ℕ) (cats_owners : ℕ) : ℚ :=
  (cats_owners : ℚ) / (total_students : ℚ) * 100

theorem cats_owners_percentage (total_students : ℕ) (cats_owners : ℕ)
  (dogs_owners : ℕ) (birds_owners : ℕ)
  (h_total_students : total_students = 400)
  (h_cats_owners : cats_owners = 80)
  (h_dogs_owners : dogs_owners = 120)
  (h_birds_owners : birds_owners = 40) :
  percentage_of_students_owning_cats total_students cats_owners = 20 :=
by {
  -- We state the proof but leave it as sorry so it's an incomplete placeholder.
  sorry
}

end NUMINAMATH_GPT_cats_owners_percentage_l852_85244


namespace NUMINAMATH_GPT_expenditure_of_negative_amount_l852_85226

theorem expenditure_of_negative_amount (x : ℝ) (h : x < 0) : 
  ∃ y : ℝ, y > 0 ∧ x = -y :=
by
  sorry

end NUMINAMATH_GPT_expenditure_of_negative_amount_l852_85226


namespace NUMINAMATH_GPT_units_digit_of_quotient_l852_85230

theorem units_digit_of_quotient : 
  (4^1985 + 7^1985) % 7 = 0 → (4^1985 + 7^1985) / 7 % 10 = 2 := 
  by 
    intro h
    sorry

end NUMINAMATH_GPT_units_digit_of_quotient_l852_85230


namespace NUMINAMATH_GPT_symmetric_point_about_x_l852_85288

-- Define the coordinates of the point A
def A : ℝ × ℝ := (-2, 3)

-- Define the function that computes the symmetric point about the x-axis
def symmetric_about_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- The concrete symmetric point of A
def A' := symmetric_about_x A

-- The original problem and proof statement
theorem symmetric_point_about_x :
  A' = (-2, -3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetric_point_about_x_l852_85288


namespace NUMINAMATH_GPT_imo_1990_q31_l852_85298

def A (n : ℕ) : ℕ := sorry -- definition of A(n)
def B (n : ℕ) : ℕ := sorry -- definition of B(n)
def f (n : ℕ) : ℕ := if B n = 1 then 1 else -- largest prime factor of B(n)
  sorry -- logic to find the largest prime factor of B(n)

theorem imo_1990_q31 :
  ∃ (M : ℕ), (∀ n : ℕ, f n ≤ M) ∧ (∀ N, (∀ n, f n ≤ N) → M ≤ N) ∧ M = 1999 :=
by sorry

end NUMINAMATH_GPT_imo_1990_q31_l852_85298


namespace NUMINAMATH_GPT_algebraic_expression_analysis_l852_85235

theorem algebraic_expression_analysis :
  (∀ x y : ℝ, (x - 1/2 * y) * (x + 1/2 * y) = x^2 - (1/2 * y)^2) ∧
  (∀ a b c : ℝ, ¬ ((3 * a + b * c) * (-b * c - 3 * a) = (3 * a + b * c)^2)) ∧
  (∀ x y : ℝ, (3 - x + y) * (3 + x + y) = (3 + y)^2 - x^2) ∧
  ((100 + 1) * (100 - 1) = 100^2 - 1) :=
by
  intros
  repeat { split }; sorry

end NUMINAMATH_GPT_algebraic_expression_analysis_l852_85235


namespace NUMINAMATH_GPT_number_of_boys_is_60_l852_85229

-- Definitions based on conditions
def total_students : ℕ := 150

def number_of_boys (x : ℕ) : Prop :=
  ∃ g : ℕ, x + g = total_students ∧ g = (x * total_students) / 100

-- Theorem statement
theorem number_of_boys_is_60 : number_of_boys 60 := 
sorry

end NUMINAMATH_GPT_number_of_boys_is_60_l852_85229


namespace NUMINAMATH_GPT_range_of_a_l852_85216

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log (x + 1) - x^2

theorem range_of_a (a : ℝ) :
    (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
    a ≥ 18 := sorry

end NUMINAMATH_GPT_range_of_a_l852_85216


namespace NUMINAMATH_GPT_divide_64_to_get_800_l852_85253

theorem divide_64_to_get_800 (x : ℝ) (h : 64 / x = 800) : x = 0.08 :=
sorry

end NUMINAMATH_GPT_divide_64_to_get_800_l852_85253


namespace NUMINAMATH_GPT_major_axis_length_is_three_l852_85270

-- Given the radius of the cylinder
def cylinder_radius : ℝ := 1

-- Given the percentage longer of the major axis than the minor axis
def percentage_longer (r : ℝ) : ℝ := 1.5

-- Given the function to calculate the minor axis using the radius
def minor_axis (r : ℝ) : ℝ := 2 * r

-- Given the function to calculate the major axis using the minor axis
def major_axis (minor_axis : ℝ) (factor : ℝ) : ℝ := minor_axis * factor

-- The conjecture states that the major axis length is 3
theorem major_axis_length_is_three : 
  major_axis (minor_axis cylinder_radius) (percentage_longer cylinder_radius) = 3 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_major_axis_length_is_three_l852_85270


namespace NUMINAMATH_GPT_number_of_fridays_l852_85250

theorem number_of_fridays (jan_1_sat : true) (is_non_leap_year : true) : ∃ (n : ℕ), n = 52 :=
by
  -- Conditions: January 1st is Saturday and it is a non-leap year.
  -- We are given that January 1st is a Saturday.
  have jan_1_sat_condition : true := jan_1_sat
  -- We are given that the year is a non-leap year (365 days).
  have non_leap_condition : true := is_non_leap_year
  -- Therefore, there are 52 Fridays in the year.
  use 52
  done

end NUMINAMATH_GPT_number_of_fridays_l852_85250


namespace NUMINAMATH_GPT_train_speed_l852_85247

theorem train_speed (length : ℕ) (time : ℕ) (h_length : length = 1200) (h_time : time = 15) :
  (length / time) = 80 := by
  sorry

end NUMINAMATH_GPT_train_speed_l852_85247


namespace NUMINAMATH_GPT_sequence_value_of_m_l852_85246

theorem sequence_value_of_m (a : ℕ → ℝ) (m : ℕ) (h1 : a 1 = 1)
                            (h2 : ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n)
                            (h3 : 8 * a m = 1) :
                            m = 8 := by
  sorry

end NUMINAMATH_GPT_sequence_value_of_m_l852_85246


namespace NUMINAMATH_GPT_number_of_pipes_l852_85291

theorem number_of_pipes (L : ℝ) : 
  let r_small := 1
  let r_large := 3
  let len_small := L
  let len_large := 2 * L
  let volume_large := π * r_large^2 * len_large
  let volume_small := π * r_small^2 * len_small
  volume_large = 18 * volume_small :=
by
  sorry

end NUMINAMATH_GPT_number_of_pipes_l852_85291


namespace NUMINAMATH_GPT_circle_area_l852_85221

theorem circle_area :
  let circle := {p : ℝ × ℝ | (p.fst - 8) ^ 2 + p.snd ^ 2 = 64}
  let line := {p : ℝ × ℝ | p.snd = 10 - p.fst}
  ∃ area : ℝ, 
    (area = 8 * Real.pi) ∧ 
    ∀ p : ℝ × ℝ, p ∈ circle → p.snd ≥ 0 → p ∈ line → p.snd ≥ 10 - p.fst →
  sorry := sorry

end NUMINAMATH_GPT_circle_area_l852_85221


namespace NUMINAMATH_GPT_bertha_no_children_count_l852_85281

-- Definitions
def bertha_daughters : ℕ := 6
def granddaughters_per_daughter : ℕ := 6
def total_daughters_and_granddaughters : ℕ := 30

-- Theorem to be proved
theorem bertha_no_children_count : 
  ∃ x : ℕ, (x * granddaughters_per_daughter + bertha_daughters = total_daughters_and_granddaughters) ∧ 
           (bertha_daughters - x + x * granddaughters_per_daughter = 26) :=
sorry

end NUMINAMATH_GPT_bertha_no_children_count_l852_85281


namespace NUMINAMATH_GPT_G_five_times_of_2_l852_85228

def G (x : ℝ) : ℝ := (x - 2) ^ 2 - 1

theorem G_five_times_of_2 : G (G (G (G (G 2)))) = 1179395 := 
by 
  rw [G, G, G, G, G]; 
  sorry

end NUMINAMATH_GPT_G_five_times_of_2_l852_85228


namespace NUMINAMATH_GPT_troll_ratio_l852_85212

theorem troll_ratio 
  (B : ℕ)
  (h1 : 6 + B + (1 / 2 : ℚ) * B = 33) : 
  B / 6 = 3 :=
by
  sorry

end NUMINAMATH_GPT_troll_ratio_l852_85212


namespace NUMINAMATH_GPT_cans_collected_on_first_day_l852_85206

-- Declare the main theorem
theorem cans_collected_on_first_day 
  (x : ℕ) -- Number of cans collected on the first day
  (total_cans : x + (x + 5) + (x + 10) + (x + 15) + (x + 20) = 150) :
  x = 20 :=
sorry

end NUMINAMATH_GPT_cans_collected_on_first_day_l852_85206


namespace NUMINAMATH_GPT_sale_in_first_month_l852_85242

theorem sale_in_first_month
  (s2 : ℕ)
  (s3 : ℕ)
  (s4 : ℕ)
  (s5 : ℕ)
  (s6 : ℕ)
  (required_total_sales : ℕ)
  (average_sales : ℕ)
  : (required_total_sales = 39000) → 
    (average_sales = 6500) → 
    (s2 = 6927) →
    (s3 = 6855) →
    (s4 = 7230) →
    (s5 = 6562) →
    (s6 = 4991) →
    s2 + s3 + s4 + s5 + s6 = 32565 →
    required_total_sales - (s2 + s3 + s4 + s5 + s6) = 6435 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sale_in_first_month_l852_85242


namespace NUMINAMATH_GPT_total_pay_is_correct_l852_85296

-- Define the constants and conditions
def regular_rate := 3  -- $ per hour
def regular_hours := 40  -- hours
def overtime_multiplier := 2  -- overtime pay is twice the regular rate
def overtime_hours := 8  -- hours

-- Calculate regular and overtime pay
def regular_pay := regular_rate * regular_hours
def overtime_rate := regular_rate * overtime_multiplier
def overtime_pay := overtime_rate * overtime_hours

-- Calculate total pay
def total_pay := regular_pay + overtime_pay

-- Prove that the total pay is $168
theorem total_pay_is_correct : total_pay = 168 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_total_pay_is_correct_l852_85296


namespace NUMINAMATH_GPT_esperanzas_gross_monthly_salary_l852_85290

variables (Rent FoodExpenses MortgageBill Savings Taxes GrossSalary : ℝ)

def problem_conditions (Rent FoodExpenses MortgageBill Savings Taxes : ℝ) :=
  Rent = 600 ∧
  FoodExpenses = (3 / 5) * Rent ∧
  MortgageBill = 3 * FoodExpenses ∧
  Savings = 2000 ∧
  Taxes = (2 / 5) * Savings

theorem esperanzas_gross_monthly_salary (h : problem_conditions Rent FoodExpenses MortgageBill Savings Taxes) :
  GrossSalary = Rent + FoodExpenses + MortgageBill + Taxes + Savings → GrossSalary = 4840 :=
by
  sorry

end NUMINAMATH_GPT_esperanzas_gross_monthly_salary_l852_85290


namespace NUMINAMATH_GPT_angle_compute_l852_85260

open Real

noncomputable def a : ℝ × ℝ := (1, -1)
noncomputable def b : ℝ × ℝ := (1, 2)

noncomputable def sub_vec := (b.1 - a.1, b.2 - a.2)
noncomputable def sum_vec := (a.1 + 2 * b.1, a.2 + 2 * b.2)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def angle_between (v₁ v₂ : ℝ × ℝ) : ℝ :=
  arccos (dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂))

theorem angle_compute : angle_between sub_vec sum_vec = π / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_compute_l852_85260


namespace NUMINAMATH_GPT_anne_cleaning_time_l852_85272

variable (B A : ℝ)

theorem anne_cleaning_time :
  (B + A) * 4 = 1 ∧ (B + 2 * A) * 3 = 1 → 1/A = 12 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_anne_cleaning_time_l852_85272


namespace NUMINAMATH_GPT_find_f_log2_20_l852_85225

noncomputable def f (x : ℝ) : ℝ :=
if -1 < x ∧ x < 0 then 2^x + 1 else sorry

lemma f_periodic (x : ℝ) : f (x - 2) = f (x + 2) :=
sorry

lemma f_odd (x : ℝ) : f (-x) = -f (x) :=
sorry

theorem find_f_log2_20 : f (Real.log 20 / Real.log 2) = -1 :=
sorry

end NUMINAMATH_GPT_find_f_log2_20_l852_85225


namespace NUMINAMATH_GPT_number_of_defective_pens_l852_85254

noncomputable def defective_pens (total : ℕ) (prob : ℚ) : ℕ :=
  let N := 6 -- since we already know the steps in the solution leading to N = 6
  let D := total - N
  D

theorem number_of_defective_pens (total : ℕ) (prob : ℚ) :
  (total = 12) → (prob = 0.22727272727272727) → defective_pens total prob = 6 :=
by
  intros ht hp
  unfold defective_pens
  sorry

end NUMINAMATH_GPT_number_of_defective_pens_l852_85254


namespace NUMINAMATH_GPT_clarence_to_matthew_ratio_l852_85293

theorem clarence_to_matthew_ratio (D C M : ℝ) (h1 : D = 6.06) (h2 : D = 1 / 2 * C) (h3 : D + C + M = 20.20) : C / M = 6 := 
by 
  sorry

end NUMINAMATH_GPT_clarence_to_matthew_ratio_l852_85293


namespace NUMINAMATH_GPT_tangent_line_at_zero_decreasing_intervals_l852_85214

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x - 2

theorem tangent_line_at_zero :
  let t : ℝ × ℝ := (0, f 0)
  (∀ x : ℝ, (9 * x - f x - 2 = 0) → t.snd = -2) := by
  sorry

theorem decreasing_intervals :
  ∀ x : ℝ, (-3 * x^2 + 6 * x + 9 < 0) ↔ (x < -1 ∨ x > 3) := by
  sorry

end NUMINAMATH_GPT_tangent_line_at_zero_decreasing_intervals_l852_85214


namespace NUMINAMATH_GPT_remainder_3_pow_500_mod_17_l852_85232

theorem remainder_3_pow_500_mod_17 : (3^500) % 17 = 13 := 
by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_500_mod_17_l852_85232


namespace NUMINAMATH_GPT_maximum_correct_answers_l852_85261

theorem maximum_correct_answers (c w u : ℕ) :
  c + w + u = 25 →
  4 * c - w = 70 →
  c ≤ 19 :=
by
  sorry

end NUMINAMATH_GPT_maximum_correct_answers_l852_85261


namespace NUMINAMATH_GPT_roof_shingles_area_l852_85211

-- Definitions based on given conditions
def base_main_roof : ℝ := 20.5
def height_main_roof : ℝ := 25
def upper_base_porch : ℝ := 2.5
def lower_base_porch : ℝ := 4.5
def height_porch : ℝ := 3
def num_gables_main_roof : ℕ := 2
def num_trapezoids_porch : ℕ := 4

-- Proof problem statement
theorem roof_shingles_area : 
  2 * (1 / 2 * base_main_roof * height_main_roof) +
  4 * (1 / 2 * (upper_base_porch + lower_base_porch) * height_porch) = 554.5 :=
by sorry

end NUMINAMATH_GPT_roof_shingles_area_l852_85211


namespace NUMINAMATH_GPT_evaluate_expression_l852_85258

theorem evaluate_expression : 
    8 * 7 / 8 * 7 = 49 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l852_85258


namespace NUMINAMATH_GPT_sum_base10_to_base4_l852_85278

theorem sum_base10_to_base4 : 
  (31 + 22 : ℕ) = 3 * 4^2 + 1 * 4^1 + 1 * 4^0 :=
by
  sorry

end NUMINAMATH_GPT_sum_base10_to_base4_l852_85278


namespace NUMINAMATH_GPT_slope_of_bisecting_line_l852_85252

theorem slope_of_bisecting_line (m n : ℕ) (hmn : Int.gcd m n = 1) : 
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  -- Define conditions for line through origin (x = 0, y = 0) bisecting the parallelogram
  let b := 135 / 19
  let slope := (90 + b) / 20
  -- The slope must be equal to 369/76 (m = 369, n = 76)
  m = 369 ∧ n = 76 → m + n = 445 := by
  intro m n hmn
  let p1 := (20, 90)
  let p2 := (20, 228)
  let p3 := (56, 306)
  let p4 := (56, 168)
  let b := 135 / 19
  let slope := (90 + b) / 20
  sorry

end NUMINAMATH_GPT_slope_of_bisecting_line_l852_85252


namespace NUMINAMATH_GPT_transformed_average_l852_85283

theorem transformed_average (n : ℕ) (original_average factor : ℝ) 
  (h1 : n = 15) (h2 : original_average = 21.5) (h3 : factor = 7) :
  (original_average * factor) = 150.5 :=
by
  sorry

end NUMINAMATH_GPT_transformed_average_l852_85283


namespace NUMINAMATH_GPT_meaningful_fraction_l852_85280

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 :=
by sorry

end NUMINAMATH_GPT_meaningful_fraction_l852_85280


namespace NUMINAMATH_GPT_gcd_2023_2052_eq_1_l852_85276

theorem gcd_2023_2052_eq_1 : Int.gcd 2023 2052 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_2023_2052_eq_1_l852_85276


namespace NUMINAMATH_GPT_orthocenter_of_ABC_l852_85264

structure Point3D := (x : ℝ) (y : ℝ) (z : ℝ)

def A : Point3D := ⟨-1, 3, 2⟩
def B : Point3D := ⟨4, -2, 2⟩
def C : Point3D := ⟨2, -1, 6⟩

def orthocenter (A B C : Point3D) : Point3D :=
  -- formula to calculate the orthocenter
  sorry

theorem orthocenter_of_ABC :
  orthocenter A B C = ⟨101 / 150, 192 / 150, 232 / 150⟩ :=
by 
  -- proof steps
  sorry

end NUMINAMATH_GPT_orthocenter_of_ABC_l852_85264


namespace NUMINAMATH_GPT_sum_of_squares_l852_85224

theorem sum_of_squares (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 1) (h2 : b^2 + b * c + c^2 = 3) (h3 : c^2 + c * a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l852_85224


namespace NUMINAMATH_GPT_remainder_is_x_plus_2_l852_85234

noncomputable def problem_division := 
  ∀ x : ℤ, ∃ q r : ℤ, (x^3 + 2 * x^2) = q * (x^2 + 3 * x + 2) + r ∧ r < x^2 + 3 * x + 2 ∧ r = x + 2

theorem remainder_is_x_plus_2 : problem_division := sorry

end NUMINAMATH_GPT_remainder_is_x_plus_2_l852_85234


namespace NUMINAMATH_GPT_green_minus_blue_is_40_l852_85282

noncomputable def number_of_green_minus_blue_disks (total_disks : ℕ) (ratio_blue : ℕ) (ratio_yellow : ℕ) (ratio_green : ℕ) : ℕ :=
  let total_ratio := ratio_blue + ratio_yellow + ratio_green
  let disks_per_part := total_disks / total_ratio
  let blue_disks := ratio_blue * disks_per_part
  let green_disks := ratio_green * disks_per_part
  green_disks - blue_disks

theorem green_minus_blue_is_40 :
  number_of_green_minus_blue_disks 144 3 7 8 = 40 :=
sorry

end NUMINAMATH_GPT_green_minus_blue_is_40_l852_85282


namespace NUMINAMATH_GPT_largest_d_l852_85255

theorem largest_d (a b c d : ℤ) 
  (h₁ : a + 1 = b - 2) 
  (h₂ : a + 1 = c + 3) 
  (h₃ : a + 1 = d - 4) : 
  d > a ∧ d > b ∧ d > c := 
by 
  -- Here we would provide the proof, but for now we'll skip it
  sorry

end NUMINAMATH_GPT_largest_d_l852_85255


namespace NUMINAMATH_GPT_price_rollback_is_correct_l852_85218

-- Define the conditions
def liters_today : ℕ := 10
def cost_per_liter_today : ℝ := 1.4
def liters_friday : ℕ := 25
def total_liters : ℕ := 35
def total_cost : ℝ := 39

-- Define the price rollback calculation
noncomputable def price_rollback : ℝ :=
  (cost_per_liter_today - (total_cost - (liters_today * cost_per_liter_today)) / liters_friday)

-- The theorem stating the rollback per liter is $0.4
theorem price_rollback_is_correct : price_rollback = 0.4 := by
  sorry

end NUMINAMATH_GPT_price_rollback_is_correct_l852_85218


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l852_85215

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m + 1)^2) ∨
  (∃ m : ℤ, 49 * m^2 + k * m + 1 = (7 * m - 1)^2) ↔
  k = 14 ∨ k = -14 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l852_85215


namespace NUMINAMATH_GPT_king_lancelot_seats_38_l852_85251

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end NUMINAMATH_GPT_king_lancelot_seats_38_l852_85251


namespace NUMINAMATH_GPT_parallel_lines_m_values_l852_85297

theorem parallel_lines_m_values (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5) ∧ (2 * x + (5 + m) * y = 8) → (m = -1 ∨ m = -7) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_m_values_l852_85297


namespace NUMINAMATH_GPT_perfect_square_unique_n_l852_85275

theorem perfect_square_unique_n (n : ℕ) (hn : n > 0) : 
  (∃ m : ℕ, 2^n + 12^n + 2011^n = m^2) ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_perfect_square_unique_n_l852_85275


namespace NUMINAMATH_GPT_range_of_a_l852_85243

theorem range_of_a
  (a x : ℝ)
  (h_eq : 2 * (1 / 4) ^ (-x) - (1 / 2) ^ (-x) + a = 0)
  (h_x : -1 ≤ x ∧ x ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l852_85243


namespace NUMINAMATH_GPT_stratified_sampling_grade10_sampled_count_l852_85299

def total_students : ℕ := 2000
def grade10_students : ℕ := 600
def grade11_students : ℕ := 680
def grade12_students : ℕ := 720
def total_sampled_students : ℕ := 50

theorem stratified_sampling_grade10_sampled_count :
  15 = (total_sampled_students * grade10_students / total_students) :=
by sorry

end NUMINAMATH_GPT_stratified_sampling_grade10_sampled_count_l852_85299


namespace NUMINAMATH_GPT_no_integers_divisible_by_all_l852_85279

-- Define the list of divisors
def divisors : List ℕ := [2, 3, 4, 5, 7, 11]

-- Define the LCM function
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Calculate the LCM of the given divisors
def lcm_divisors : ℕ := lcm_list divisors

-- Define a predicate to check divisibility by all divisors
def is_divisible_by_all (n : ℕ) (ds : List ℕ) : Prop :=
  ds.all (λ d => n % d = 0)

-- Define the theorem to prove the number of integers between 1 and 1000 divisible by the given divisors
theorem no_integers_divisible_by_all :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 1000 ∧ is_divisible_by_all n divisors) → False := by
  sorry

end NUMINAMATH_GPT_no_integers_divisible_by_all_l852_85279


namespace NUMINAMATH_GPT_fg_of_2_eq_513_l852_85207

def f (x : ℤ) : ℤ := x^3 + 1
def g (x : ℤ) : ℤ := 3*x + 2

theorem fg_of_2_eq_513 : f (g 2) = 513 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_eq_513_l852_85207


namespace NUMINAMATH_GPT_area_of_regionM_l852_85287

/-
Define the conditions as separate predicates in Lean.
-/

def cond1 (x y : ℝ) : Prop := y - x ≥ abs (x + y)

def cond2 (x y : ℝ) : Prop := (x^2 + 8*x + y^2 + 6*y) / (2*y - x - 8) ≤ 0

/-
Define region \( M \) by combining the conditions.
-/

def regionM (x y : ℝ) : Prop := cond1 x y ∧ cond2 x y

/-
Define the main theorem to compute the area of the region \( M \).
-/

theorem area_of_regionM : 
  ∀ x y : ℝ, (regionM x y) → (calculateAreaOfM) := sorry

/-
A placeholder definition to calculate the area of M. 
-/

noncomputable def calculateAreaOfM : ℝ := 8

end NUMINAMATH_GPT_area_of_regionM_l852_85287


namespace NUMINAMATH_GPT_range_of_a_l852_85213

open Set

theorem range_of_a (a x : ℝ) (p : ℝ → Prop) (q : ℝ → ℝ → Prop)
    (hp : p x → |x - a| > 3)
    (hq : q x a → (x + 1) * (2 * x - 1) ≥ 0)
    (hsuff : ∀ x, ¬p x → q x a) :
    {a | ∀ x, (¬ (|x - a| > 3) → (x + 1) * (2 * x - 1) ≥ 0) → (( a ≤ -4) ∨ (a ≥ 7 / 2))} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l852_85213


namespace NUMINAMATH_GPT_trivia_team_points_l852_85285

theorem trivia_team_points 
    (total_members : ℕ) 
    (members_absent : ℕ) 
    (total_points : ℕ) 
    (members_present : ℕ := total_members - members_absent) 
    (points_per_member : ℕ := total_points / members_present) 
    (h1 : total_members = 7) 
    (h2 : members_absent = 2) 
    (h3 : total_points = 20) : 
    points_per_member = 4 :=
by
    sorry

end NUMINAMATH_GPT_trivia_team_points_l852_85285


namespace NUMINAMATH_GPT_simplify_expression_l852_85289

theorem simplify_expression : 
    2 * Real.sqrt 12 + 3 * Real.sqrt (4 / 3) - Real.sqrt (16 / 3) - (2 / 3) * Real.sqrt 48 = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l852_85289


namespace NUMINAMATH_GPT_problem_statement_l852_85219

theorem problem_statement (a b : ℝ) (h1 : 1 / a + 1 / b = Real.sqrt 5) (h2 : a ≠ b) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l852_85219


namespace NUMINAMATH_GPT_intersection_M_complement_N_eq_l852_85286

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def N : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
noncomputable def complement_N : Set ℝ := {y | y < 1}

theorem intersection_M_complement_N_eq : M ∩ complement_N = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_complement_N_eq_l852_85286


namespace NUMINAMATH_GPT_total_cost_proof_l852_85205

def F : ℝ := 20.50
def R : ℝ := 61.50
def M : ℝ := 1476

def total_cost (mangos : ℝ) (rice : ℝ) (flour : ℝ) : ℝ :=
  (M * mangos) + (R * rice) + (F * flour)

theorem total_cost_proof:
  total_cost 4 3 5 = 6191 := by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l852_85205


namespace NUMINAMATH_GPT_hannah_books_per_stocking_l852_85200

theorem hannah_books_per_stocking
  (candy_canes_per_stocking : ℕ)
  (beanie_babies_per_stocking : ℕ)
  (num_kids : ℕ)
  (total_stuffers : ℕ)
  (books_per_stocking : ℕ) :
  candy_canes_per_stocking = 4 →
  beanie_babies_per_stocking = 2 →
  num_kids = 3 →
  total_stuffers = 21 →
  books_per_stocking = (total_stuffers - (candy_canes_per_stocking + beanie_babies_per_stocking) * num_kids) / num_kids →
  books_per_stocking = 1 := 
by 
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h5
  simp at h5
  sorry

end NUMINAMATH_GPT_hannah_books_per_stocking_l852_85200


namespace NUMINAMATH_GPT_fourth_root_of_207360000_l852_85292

theorem fourth_root_of_207360000 :
  120 ^ 4 = 207360000 :=
sorry

end NUMINAMATH_GPT_fourth_root_of_207360000_l852_85292
