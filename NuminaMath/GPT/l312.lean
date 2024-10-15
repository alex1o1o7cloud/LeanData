import Mathlib

namespace NUMINAMATH_GPT_plane_can_be_colored_l312_31220

-- Define a structure for a triangle and the plane divided into triangles
structure Triangle :=
(vertices : Fin 3 → ℕ) -- vertices labeled with ℕ, interpreted as 0, 1, 2

structure Plane :=
(triangles : Set Triangle)
(adjacent : Triangle → Triangle → Prop)
(labels_correct : ∀ {t1 t2 : Triangle}, adjacent t1 t2 → 
  ∀ i j: Fin 3, t1.vertices i ≠ t1.vertices j)
(adjacent_conditions: ∀ t1 t2: Triangle, adjacent t1 t2 → 
  ∃ v, (∃ i: Fin 3, t1.vertices i = v) ∧ (∃ j: Fin 3, t2.vertices j = v))

theorem plane_can_be_colored (p : Plane) : 
  ∃ (c : Triangle → ℕ), (∀ t1 t2, p.adjacent t1 t2 → c t1 ≠ c t2) :=
sorry

end NUMINAMATH_GPT_plane_can_be_colored_l312_31220


namespace NUMINAMATH_GPT_increasing_interval_implication_l312_31298

theorem increasing_interval_implication (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2) 2, (1 / x + 2 * a * x > 0)) → a > -1 / 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_increasing_interval_implication_l312_31298


namespace NUMINAMATH_GPT_concentration_replacement_l312_31215

theorem concentration_replacement 
  (initial_concentration : ℝ)
  (new_concentration : ℝ)
  (fraction_replaced : ℝ)
  (replacing_concentration : ℝ)
  (h1 : initial_concentration = 0.45)
  (h2 : new_concentration = 0.35)
  (h3 : fraction_replaced = 0.5) :
  replacing_concentration = 0.25 := by
  sorry

end NUMINAMATH_GPT_concentration_replacement_l312_31215


namespace NUMINAMATH_GPT_complement_B_intersection_A_complement_B_l312_31258

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x < 0}
noncomputable def B : Set ℝ := {x | x > 1}

theorem complement_B :
  (U \ B) = {x | x ≤ 1} := by
  sorry

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | x < 0} := by
  sorry

end NUMINAMATH_GPT_complement_B_intersection_A_complement_B_l312_31258


namespace NUMINAMATH_GPT_darnell_avg_yards_eq_11_l312_31208

-- Defining the given conditions
def malikYardsPerGame := 18
def josiahYardsPerGame := 22
def numberOfGames := 4
def totalYardsRun := 204

-- Defining the corresponding total yards for Malik and Josiah
def malikTotalYards := malikYardsPerGame * numberOfGames
def josiahTotalYards := josiahYardsPerGame * numberOfGames

-- The combined total yards for Malik and Josiah
def combinedTotal := malikTotalYards + josiahTotalYards

-- Calculate Darnell's total yards and average per game
def darnellTotalYards := totalYardsRun - combinedTotal
def darnellAverageYardsPerGame := darnellTotalYards / numberOfGames

-- Now, we write the theorem to prove darnell's average yards per game
theorem darnell_avg_yards_eq_11 : darnellAverageYardsPerGame = 11 := by
  sorry

end NUMINAMATH_GPT_darnell_avg_yards_eq_11_l312_31208


namespace NUMINAMATH_GPT_total_sugar_l312_31235

theorem total_sugar (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_total_sugar_l312_31235


namespace NUMINAMATH_GPT_loaves_on_friday_l312_31285

theorem loaves_on_friday
  (bread_wed : ℕ)
  (bread_thu : ℕ)
  (bread_sat : ℕ)
  (bread_sun : ℕ)
  (bread_mon : ℕ)
  (inc_wed_thu : bread_thu - bread_wed = 2)
  (inc_sat_sun : bread_sun - bread_sat = 5)
  (inc_sun_mon : bread_mon - bread_sun = 6)
  (pattern : ∀ n : ℕ, bread_wed + (2 + n) + n = bread_thu + n)
  : bread_thu + 3 = 10 := 
sorry

end NUMINAMATH_GPT_loaves_on_friday_l312_31285


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_30_l312_31287

theorem remainder_when_sum_divided_by_30 (x y z : ℕ) (hx : x % 30 = 14) (hy : y % 30 = 5) (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_30_l312_31287


namespace NUMINAMATH_GPT_mortgage_loan_amount_l312_31288

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end NUMINAMATH_GPT_mortgage_loan_amount_l312_31288


namespace NUMINAMATH_GPT_problem_a_problem_d_l312_31299

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_problem_d_l312_31299


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l312_31278

noncomputable def A := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l312_31278


namespace NUMINAMATH_GPT_volleyball_height_30_l312_31224

theorem volleyball_height_30 (t : ℝ) : (60 - 9 * t - 4.5 * t^2 = 30) → t = 1.77 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_volleyball_height_30_l312_31224


namespace NUMINAMATH_GPT_bookshelf_arrangements_l312_31222

theorem bookshelf_arrangements :
  let math_books := 6
  let english_books := 5
  let valid_arrangements := 2400
  (∃ (math_books : Nat) (english_books : Nat) (valid_arrangements : Nat), 
    math_books = 6 ∧ english_books = 5 ∧ valid_arrangements = 2400) :=
by
  sorry

end NUMINAMATH_GPT_bookshelf_arrangements_l312_31222


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l312_31252

-- Declare the repeating decimals as constants
def x : ℚ := 2/3
def y : ℚ := 7/9

-- The problem statement
theorem sum_of_repeating_decimals : x + y = 13 / 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l312_31252


namespace NUMINAMATH_GPT_rectangular_solid_volume_l312_31250

variables {x y z : ℝ}

theorem rectangular_solid_volume :
  x * y = 15 ∧ y * z = 10 ∧ x * z = 6 ∧ x = 3 * y →
  x * y * z = 6 * Real.sqrt 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rectangular_solid_volume_l312_31250


namespace NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l312_31290

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l312_31290


namespace NUMINAMATH_GPT_janice_purchase_l312_31280

theorem janice_purchase : 
  ∃ (a b c : ℕ), a + b + c = 50 ∧ 50 * a + 400 * b + 500 * c = 10000 ∧ a = 23 :=
by
  sorry

end NUMINAMATH_GPT_janice_purchase_l312_31280


namespace NUMINAMATH_GPT_value_of_M_l312_31242

theorem value_of_M (M : ℝ) (h : (25 / 100) * M = (35 / 100) * 1800) : M = 2520 := 
sorry

end NUMINAMATH_GPT_value_of_M_l312_31242


namespace NUMINAMATH_GPT_not_divisible_by_5_count_l312_31277

-- Define the total number of four-digit numbers using the digits 0, 1, 2, 3, 4, 5 without repetition
def total_four_digit_numbers : ℕ := 300

-- Define the number of four-digit numbers ending with 0
def numbers_ending_with_0 : ℕ := 60

-- Define the number of four-digit numbers ending with 5
def numbers_ending_with_5 : ℕ := 48

-- Theorem stating the number of four-digit numbers that cannot be divided by 5
theorem not_divisible_by_5_count : total_four_digit_numbers - numbers_ending_with_0 - numbers_ending_with_5 = 192 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_not_divisible_by_5_count_l312_31277


namespace NUMINAMATH_GPT_books_purchased_with_grant_l312_31262

-- Define the conditions
def total_books_now : ℕ := 8582
def books_before_grant : ℕ := 5935

-- State the theorem that we need to prove
theorem books_purchased_with_grant : (total_books_now - books_before_grant) = 2647 := by
  sorry

end NUMINAMATH_GPT_books_purchased_with_grant_l312_31262


namespace NUMINAMATH_GPT_equilateral_A1C1E1_l312_31253

variables {A B C D E F A₁ B₁ C₁ D₁ E₁ F₁ : Type*}

-- Defining the convex hexagon and the equilateral triangles.
def is_convex_hexagon (A B C D E F : Type*) : Prop := sorry

def is_equilateral (P Q R : Type*) : Prop := sorry

-- Given conditions
variable (h_hexagon : is_convex_hexagon A B C D E F)
variable (h_eq_triangles :
  is_equilateral A B C₁ ∧ is_equilateral B C D₁ ∧ is_equilateral C D E₁ ∧
  is_equilateral D E F₁ ∧ is_equilateral E F A₁ ∧ is_equilateral F A B₁)
variable (h_B1D1F1 : is_equilateral B₁ D₁ F₁)

-- Statement to be proved
theorem equilateral_A1C1E1 :
  is_equilateral A₁ C₁ E₁ :=
sorry

end NUMINAMATH_GPT_equilateral_A1C1E1_l312_31253


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l312_31243

-- Statement for problem 1
theorem problem1 : -12 + (-6) - (-28) = 10 :=
  by sorry

-- Statement for problem 2
theorem problem2 : (-8 / 5) * (15 / 4) / (-9) = 2 / 3 :=
  by sorry

-- Statement for problem 3
theorem problem3 : (-3 / 16 - 7 / 24 + 5 / 6) * (-48) = -17 :=
  by sorry

-- Statement for problem 4
theorem problem4 : -3^2 + (7 / 8 - 1) * (-2)^2 = -9.5 :=
  by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l312_31243


namespace NUMINAMATH_GPT_sled_total_distance_l312_31244

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (a₁ + (a₁ + (n - 1) * d)) / 2

theorem sled_total_distance (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 6 → d = 8 → n = 20 → arithmetic_sequence_sum a₁ d n = 1640 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_sled_total_distance_l312_31244


namespace NUMINAMATH_GPT_picture_books_count_l312_31270

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end NUMINAMATH_GPT_picture_books_count_l312_31270


namespace NUMINAMATH_GPT_cube_sum_minus_triple_product_l312_31223

theorem cube_sum_minus_triple_product (x y z : ℝ) (h1 : x + y + z = 8) (h2 : xy + yz + zx = 20) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 32 :=
sorry

end NUMINAMATH_GPT_cube_sum_minus_triple_product_l312_31223


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_3_sin_cos_ratio_l312_31284

theorem tan_alpha_plus_pi_over_3
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  Real.tan (α + Real.pi / 3) = (48 - 25 * Real.sqrt 3) / 11 := 
sorry

theorem sin_cos_ratio
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17 :=
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_3_sin_cos_ratio_l312_31284


namespace NUMINAMATH_GPT_students_moved_outside_correct_l312_31249

noncomputable def students_total : ℕ := 90
noncomputable def students_cafeteria_initial : ℕ := (2 * students_total) / 3
noncomputable def students_outside_initial : ℕ := students_total - students_cafeteria_initial
noncomputable def students_ran_inside : ℕ := students_outside_initial / 3
noncomputable def students_cafeteria_now : ℕ := 67
noncomputable def students_moved_outside : ℕ := students_cafeteria_initial + students_ran_inside - students_cafeteria_now

theorem students_moved_outside_correct : students_moved_outside = 3 := by
  sorry

end NUMINAMATH_GPT_students_moved_outside_correct_l312_31249


namespace NUMINAMATH_GPT_child_ticket_cost_l312_31276

theorem child_ticket_cost 
    (x : ℝ)
    (adult_ticket_cost : ℝ := 5)
    (total_sales : ℝ := 178)
    (total_tickets_sold : ℝ := 42)
    (child_tickets_sold : ℝ := 16) 
    (adult_tickets_sold : ℝ := total_tickets_sold - child_tickets_sold)
    (total_adult_sales : ℝ := adult_tickets_sold * adult_ticket_cost)
    (sales_equation : total_adult_sales + child_tickets_sold * x = total_sales) : 
    x = 3 :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l312_31276


namespace NUMINAMATH_GPT_dog_ate_cost_6_l312_31216

noncomputable def totalCost : ℝ := 4 + 2 + 0.5 + 2.5
noncomputable def costPerSlice : ℝ := totalCost / 6
noncomputable def slicesEatenByDog : ℕ := 6 - 2
noncomputable def costEatenByDog : ℝ := slicesEatenByDog * costPerSlice

theorem dog_ate_cost_6 : costEatenByDog = 6 := by
  sorry

end NUMINAMATH_GPT_dog_ate_cost_6_l312_31216


namespace NUMINAMATH_GPT_simplify_expression_l312_31279

theorem simplify_expression (x y : ℤ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := 
by 
  sorry 

end NUMINAMATH_GPT_simplify_expression_l312_31279


namespace NUMINAMATH_GPT_sculpture_height_l312_31264

theorem sculpture_height (base_height : ℕ) (total_height_ft : ℝ) (inches_per_foot : ℕ) 
  (h1 : base_height = 8) (h2 : total_height_ft = 3.5) (h3 : inches_per_foot = 12) : 
  (total_height_ft * inches_per_foot - base_height) = 34 := 
by
  sorry

end NUMINAMATH_GPT_sculpture_height_l312_31264


namespace NUMINAMATH_GPT_weight_of_one_baseball_l312_31239

structure Context :=
  (numberBaseballs : ℕ)
  (numberBicycles : ℕ)
  (weightBicycles : ℕ)
  (weightTotalBicycles : ℕ)

def problem (ctx : Context) :=
  ctx.weightTotalBicycles = ctx.numberBicycles * ctx.weightBicycles ∧
  ctx.numberBaseballs * ctx.weightBicycles = ctx.weightTotalBicycles →
  (ctx.weightTotalBicycles / ctx.numberBaseballs) = 8

theorem weight_of_one_baseball (ctx : Context) : problem ctx :=
sorry

end NUMINAMATH_GPT_weight_of_one_baseball_l312_31239


namespace NUMINAMATH_GPT_knight_liar_grouping_l312_31274

noncomputable def can_be_partitioned_into_knight_liar_groups (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : Prop :=
  ∃ t : ℕ, n = (m + 1) * t

-- Show that if the company has n people, where n ≥ 2, and there exists at least one knight,
-- then n can be partitioned into groups where each group contains 1 knight and m liars.
theorem knight_liar_grouping (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : can_be_partitioned_into_knight_liar_groups n m h1 h2 :=
sorry

end NUMINAMATH_GPT_knight_liar_grouping_l312_31274


namespace NUMINAMATH_GPT_moles_of_NH3_formed_l312_31210

-- Conditions
def moles_NH4Cl : ℕ := 3 -- 3 moles of Ammonium chloride
def total_moles_NH3_formed : ℕ := 3 -- The total moles of Ammonia formed

-- The balanced chemical reaction implies a 1:1 molar ratio
lemma reaction_ratio (n : ℕ) : total_moles_NH3_formed = n := by
  sorry

-- Prove that the number of moles of NH3 formed is equal to 3
theorem moles_of_NH3_formed : total_moles_NH3_formed = moles_NH4Cl := 
reaction_ratio moles_NH4Cl

end NUMINAMATH_GPT_moles_of_NH3_formed_l312_31210


namespace NUMINAMATH_GPT_g_zero_g_one_l312_31297

variable (g : ℤ → ℤ)

axiom condition1 (x : ℤ) : g (x + 5) - g x = 10 * x + 30
axiom condition2 (x : ℤ) : g (x^2 - 2) = (g x - x)^2 + x^2 - 4

theorem g_zero_g_one : (g 0, g 1) = (-4, 1) := 
by 
  sorry

end NUMINAMATH_GPT_g_zero_g_one_l312_31297


namespace NUMINAMATH_GPT_largest_base_5_five_digits_base_10_value_l312_31268

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end NUMINAMATH_GPT_largest_base_5_five_digits_base_10_value_l312_31268


namespace NUMINAMATH_GPT_total_shaded_area_l312_31256

-- Problem condition definitions
def side_length_carpet := 12
def ratio_large_square : ℕ := 4
def ratio_small_square : ℕ := 4

-- Problem statement
theorem total_shaded_area : 
  ∃ S T : ℚ, 
    12 / S = ratio_large_square ∧ S / T = ratio_small_square ∧ 
    (12 * (T * T)) + (S * S) = 15.75 := 
sorry

end NUMINAMATH_GPT_total_shaded_area_l312_31256


namespace NUMINAMATH_GPT_larger_number_of_two_with_conditions_l312_31273

theorem larger_number_of_two_with_conditions (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_of_two_with_conditions_l312_31273


namespace NUMINAMATH_GPT_change_factor_l312_31226

theorem change_factor (n : ℕ) (avg_original avg_new : ℕ) (F : ℝ)
  (h1 : n = 10) (h2 : avg_original = 80) (h3 : avg_new = 160) 
  (h4 : F * (n * avg_original) = n * avg_new) :
  F = 2 :=
by
  sorry

end NUMINAMATH_GPT_change_factor_l312_31226


namespace NUMINAMATH_GPT_intersection_complement_eq_l312_31201

open Set

variable (U A B : Set ℕ)
  
theorem intersection_complement_eq : 
  U = {0, 1, 2, 3, 4} → 
  A = {0, 1, 3} → 
  B = {2, 3} → 
  A ∩ (U \ B) = {0, 1} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l312_31201


namespace NUMINAMATH_GPT_sum_of_midpoints_l312_31206

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 15) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l312_31206


namespace NUMINAMATH_GPT_probability_not_within_square_b_l312_31248

noncomputable def prob_not_within_square_b : Prop :=
  let area_A := 121
  let side_length_B := 16 / 4
  let area_B := side_length_B * side_length_B
  let area_not_covered := area_A - area_B
  let prob := area_not_covered / area_A
  prob = (105 / 121)

theorem probability_not_within_square_b : prob_not_within_square_b :=
by
  sorry

end NUMINAMATH_GPT_probability_not_within_square_b_l312_31248


namespace NUMINAMATH_GPT_property1_property2_l312_31283

/-- Given sequence a_n defined as a_n = 3(n^2 + n) + 7 -/
def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

/-- Property 1: Out of any five consecutive terms in the sequence, only one term is divisible by 5. -/
theorem property1 (n : ℕ) : (∃ k : ℕ, a (5 * k + 2) % 5 = 0) ∧ (∀ k : ℕ, ∀ r : ℕ, r ≠ 2 → a (5 * k + r) % 5 ≠ 0) :=
by
  sorry

/-- Property 2: None of the terms in this sequence is a cube of an integer. -/
theorem property2 (n : ℕ) : ¬(∃ t : ℕ, a n = t^3) :=
by
  sorry

end NUMINAMATH_GPT_property1_property2_l312_31283


namespace NUMINAMATH_GPT_original_number_l312_31267

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end NUMINAMATH_GPT_original_number_l312_31267


namespace NUMINAMATH_GPT_expression_equals_two_l312_31204

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem expression_equals_two : simplify_expression = 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_two_l312_31204


namespace NUMINAMATH_GPT_ages_total_l312_31282

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end NUMINAMATH_GPT_ages_total_l312_31282


namespace NUMINAMATH_GPT_remainder_when_15_plus_y_div_31_l312_31294

theorem remainder_when_15_plus_y_div_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + y) % 31 = 24 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_15_plus_y_div_31_l312_31294


namespace NUMINAMATH_GPT_remaining_black_cards_l312_31202

-- Define the conditions of the problem
def total_cards : ℕ := 52
def colors : ℕ := 2
def cards_per_color := total_cards / colors
def black_cards_taken_out : ℕ := 5
def total_black_cards : ℕ := cards_per_color

-- Prove the remaining black cards
theorem remaining_black_cards : total_black_cards - black_cards_taken_out = 21 := 
by
  -- Logic to calculate remaining black cards
  sorry

end NUMINAMATH_GPT_remaining_black_cards_l312_31202


namespace NUMINAMATH_GPT_smallest_x_no_triangle_l312_31217

def triangle_inequality_violated (a b c : ℝ) : Prop :=
a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_x_no_triangle (x : ℕ) (h : ∀ x, triangle_inequality_violated (7 - x : ℝ) (24 - x : ℝ) (26 - x : ℝ)) : x = 5 :=
sorry

end NUMINAMATH_GPT_smallest_x_no_triangle_l312_31217


namespace NUMINAMATH_GPT_map_distance_to_actual_distance_l312_31247

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_map_to_real : ℝ)
  (scale_real_distance : ℝ)
  (H_map_distance : map_distance = 18)
  (H_scale_map : scale_map_to_real = 0.5)
  (H_scale_real : scale_real_distance = 6) :
  (map_distance / scale_map_to_real) * scale_real_distance = 216 :=
by
  sorry

end NUMINAMATH_GPT_map_distance_to_actual_distance_l312_31247


namespace NUMINAMATH_GPT_original_number_of_people_l312_31286

theorem original_number_of_people (x : ℕ) 
  (h1 : (x / 2) - ((x / 2) / 3) = 12) : 
  x = 36 :=
sorry

end NUMINAMATH_GPT_original_number_of_people_l312_31286


namespace NUMINAMATH_GPT_total_pages_in_book_l312_31234

theorem total_pages_in_book (P : ℕ)
  (first_day : P - (P / 5) - 12 = remaining_1)
  (second_day : remaining_1 - (remaining_1 / 4) - 15 = remaining_2)
  (third_day : remaining_2 - (remaining_2 / 3) - 18 = 42) :
  P = 190 := 
sorry

end NUMINAMATH_GPT_total_pages_in_book_l312_31234


namespace NUMINAMATH_GPT_range_of_a_l312_31295

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x + a < 0) ∧ (∀ x : ℝ, 2 * x + 7 > 4 * x - 1) ∧ (∀ x : ℝ, x < 0) → a = 0 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l312_31295


namespace NUMINAMATH_GPT_symmetric_circle_eqn_l312_31257

theorem symmetric_circle_eqn (x y : ℝ) :
  (∃ (x0 y0 : ℝ), (x - 2)^2 + (y - 2)^2 = 7 ∧ x + y = 2) → x^2 + y^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_eqn_l312_31257


namespace NUMINAMATH_GPT_frank_handed_cashier_amount_l312_31209

-- Place conditions as definitions
def cost_chocolate_bar : ℕ := 2
def cost_bag_chip : ℕ := 3
def num_chocolate_bars : ℕ := 5
def num_bag_chips : ℕ := 2
def change_received : ℕ := 4

-- Define the target theorem (Lean 4 statement)
theorem frank_handed_cashier_amount :
  (num_chocolate_bars * cost_chocolate_bar + num_bag_chips * cost_bag_chip + change_received = 20) := 
sorry

end NUMINAMATH_GPT_frank_handed_cashier_amount_l312_31209


namespace NUMINAMATH_GPT_min_value_reciprocals_l312_31214

variable {a b : ℝ}

theorem min_value_reciprocals (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 2) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ∀ x y, x > 0 → y > 0 → x + y = 2 → 
  (1/a + 1/b) ≥ 2) :=
sorry

end NUMINAMATH_GPT_min_value_reciprocals_l312_31214


namespace NUMINAMATH_GPT_find_actual_marks_l312_31227

theorem find_actual_marks (wrong_marks : ℕ) (avg_increase : ℕ) (num_pupils : ℕ) (h_wrong_marks: wrong_marks = 73) (h_avg_increase : avg_increase = 1/2) (h_num_pupils : num_pupils = 16) : 
  ∃ (actual_marks : ℕ), actual_marks = 65 :=
by
  have total_increase := num_pupils * avg_increase
  have eqn := wrong_marks - total_increase
  use eqn
  sorry

end NUMINAMATH_GPT_find_actual_marks_l312_31227


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l312_31219

noncomputable def sum_of_first_n_terms (n : ℕ) (a d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_sum 
  (a_n : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a_n (n + 1) = a_n n + d) 
  (h1 : a_n 1 + a_n 2 + a_n 3 = 3 )
  (h2 : a_n 28 + a_n 29 + a_n 30 = 165 ) 
  : sum_of_first_n_terms 30 (a_n 1) (a_n 2 - a_n 1) = 840 := 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l312_31219


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l312_31292

theorem necessary_but_not_sufficient (x : ℝ) :
  (x - 1) * (x + 2) = 0 → (x = 1 ∨ x = -2) ∧ (x = 1 → (x - 1) * (x + 2) = 0) ∧ ¬((x - 1) * (x + 2) = 0 ↔ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l312_31292


namespace NUMINAMATH_GPT_hyperbola_center_l312_31271

theorem hyperbola_center :
  ∃ c : ℝ × ℝ, (c = (3, 4) ∧ ∀ x y : ℝ, 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0 ↔ (x - 3)^2 / 4 - (y - 4)^2 / 1 = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l312_31271


namespace NUMINAMATH_GPT_range_of_k_l312_31261

theorem range_of_k {k : ℝ} :
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end NUMINAMATH_GPT_range_of_k_l312_31261


namespace NUMINAMATH_GPT_inequality_c_l312_31230

theorem inequality_c (x : ℝ) : x^2 + 1 + 1 / (x^2 + 1) ≥ 2 := sorry

end NUMINAMATH_GPT_inequality_c_l312_31230


namespace NUMINAMATH_GPT_total_money_is_305_l312_31241

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end NUMINAMATH_GPT_total_money_is_305_l312_31241


namespace NUMINAMATH_GPT_students_not_finding_parents_funny_l312_31238

theorem students_not_finding_parents_funny:
  ∀ (total_students funny_dad funny_mom funny_both : ℕ),
  total_students = 50 →
  funny_dad = 25 →
  funny_mom = 30 →
  funny_both = 18 →
  (total_students - (funny_dad + funny_mom - funny_both) = 13) :=
by
  intros total_students funny_dad funny_mom funny_both
  sorry

end NUMINAMATH_GPT_students_not_finding_parents_funny_l312_31238


namespace NUMINAMATH_GPT_solution_set_inequality_l312_31211

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := 
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l312_31211


namespace NUMINAMATH_GPT_correct_operation_l312_31228

theorem correct_operation (x y a b : ℝ) :
  (-2 * x) * (3 * y) = -6 * x * y :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l312_31228


namespace NUMINAMATH_GPT_quadratic_roots_interlace_l312_31255

variable (p1 p2 q1 q2 : ℝ)

theorem quadratic_roots_interlace
(h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + p1 * r1 + q1 = 0 ∧ r2^2 + p1 * r2 + q1 = 0)) ∧
  (∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + p2 * s1 + q2 = 0 ∧ s2^2 + p2 * s2 + q2 = 0)) ∧
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
  (a^2 + p1*a + q1 = 0 ∧ b^2 + p2*b + q2 = 0 ∧ c^2 + p1*c + q1 = 0 ∧ d^2 + p2*d + q2 = 0)) := 
sorry

end NUMINAMATH_GPT_quadratic_roots_interlace_l312_31255


namespace NUMINAMATH_GPT_probability_of_region_F_l312_31221

theorem probability_of_region_F
  (pD pE pG pF : ℚ)
  (hD : pD = 3/8)
  (hE : pE = 1/4)
  (hG : pG = 1/8)
  (hSum : pD + pE + pF + pG = 1) : pF = 1/4 :=
by
  -- we can perform the steps as mentioned in the solution without actually executing them
  sorry

end NUMINAMATH_GPT_probability_of_region_F_l312_31221


namespace NUMINAMATH_GPT_locus_of_intersection_l312_31213

theorem locus_of_intersection (m : ℝ) :
  (∃ x y : ℝ, (m * x - y + 1 = 0) ∧ (x - m * y - 1 = 0)) ↔ (∃ x y : ℝ, (x - y = 0) ∨ (x - y + 1 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_locus_of_intersection_l312_31213


namespace NUMINAMATH_GPT_math_problem_l312_31263

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h_has_derivative_f : ∀ x, Differentiable ℝ f)
variable (h_has_derivative_g : ∀ x, Differentiable ℝ g)
variable (h_deriv_ineq : ∀ x, deriv f x > deriv g x)
variable (h_interval : x ∈ Ioo a b)

theorem math_problem :
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) :=
sorry

end NUMINAMATH_GPT_math_problem_l312_31263


namespace NUMINAMATH_GPT_eval_expr_at_neg3_l312_31225

theorem eval_expr_at_neg3 : 
  (5 + 2 * (-3) * ((-3) + 5) - 5^2) / (2 * (-3) - 5 + 2 * (-3)^3) = 32 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expr_at_neg3_l312_31225


namespace NUMINAMATH_GPT_x_y_quartic_l312_31205

theorem x_y_quartic (x y : ℝ) (h₁ : x - y = 2) (h₂ : x * y = 48) : x^4 + y^4 = 5392 := by
  sorry

end NUMINAMATH_GPT_x_y_quartic_l312_31205


namespace NUMINAMATH_GPT_age_difference_l312_31293

variable (A : ℕ) -- Albert's age
variable (B : ℕ) -- Albert's brother's age
variable (F : ℕ) -- Father's age
variable (M : ℕ) -- Mother's age

def age_conditions : Prop :=
  (B = A - 2) ∧ (F = A + 48) ∧ (M = B + 46)

theorem age_difference (h : age_conditions A B F M) : F - M = 4 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l312_31293


namespace NUMINAMATH_GPT_area_of_formed_triangle_l312_31229

def triangle_area (S R d : ℝ) (S₁ : ℝ) : Prop :=
  S₁ = (S / 4) * |1 - (d^2 / R^2)|

variable (S R d : ℝ)

theorem area_of_formed_triangle (h : S₁ = (S / 4) * |1 - (d^2 / R^2)|) : triangle_area S R d S₁ :=
by
  sorry

end NUMINAMATH_GPT_area_of_formed_triangle_l312_31229


namespace NUMINAMATH_GPT_inequality_of_fractions_l312_31237

theorem inequality_of_fractions
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (c d : ℝ) (h3 : c < d) (h4 : d < 0)
  (e : ℝ) (h5 : e < 0) :
  (e / ((a - c)^2)) > (e / ((b - d)^2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_fractions_l312_31237


namespace NUMINAMATH_GPT_find_q_l312_31281

variable (p q : ℝ)

theorem find_q (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_GPT_find_q_l312_31281


namespace NUMINAMATH_GPT_smallest_d_l312_31246

theorem smallest_d (d t s : ℕ) (h1 : 3 * t - 4 * s = 2023)
                   (h2 : t = s + d) 
                   (h3 : 4 * s > 0)
                   (h4 : d % 3 = 0) :
                   d = 675 := sorry

end NUMINAMATH_GPT_smallest_d_l312_31246


namespace NUMINAMATH_GPT_shifted_parabola_expression_l312_31218

theorem shifted_parabola_expression (x : ℝ) :
  let y_original := x^2
  let y_shifted_right := (x - 1)^2
  let y_shifted_up := y_shifted_right + 2
  y_shifted_up = (x - 1)^2 + 2 :=
by
  sorry

end NUMINAMATH_GPT_shifted_parabola_expression_l312_31218


namespace NUMINAMATH_GPT_value_of_expression_l312_31251

theorem value_of_expression (m n : ℝ) (h : m + n = 3) :
  2 * m^2 + 4 * m * n + 2 * n^2 - 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l312_31251


namespace NUMINAMATH_GPT_ham_and_bread_percentage_l312_31269

-- Defining the different costs as constants
def cost_of_bread : ℝ := 50
def cost_of_ham : ℝ := 150
def cost_of_cake : ℝ := 200

-- Defining the total cost of the items
def total_cost : ℝ := cost_of_bread + cost_of_ham + cost_of_cake

-- Defining the combined cost of ham and bread
def combined_cost_ham_and_bread : ℝ := cost_of_bread + cost_of_ham

-- The theorem stating that the combined cost of ham and bread is 50% of the total cost
theorem ham_and_bread_percentage : (combined_cost_ham_and_bread / total_cost) * 100 = 50 := by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_ham_and_bread_percentage_l312_31269


namespace NUMINAMATH_GPT_inscribed_square_area_l312_31245

theorem inscribed_square_area (R : ℝ) (h : (R^2 * (π - 2) / 4) = (2 * π - 4)) : 
  ∃ (a : ℝ), a^2 = 16 := by
  sorry

end NUMINAMATH_GPT_inscribed_square_area_l312_31245


namespace NUMINAMATH_GPT_base_conversion_subtraction_l312_31240

/-- Definition of base conversion from base 7 and base 5 to base 10. -/
def convert_base_7_to_10 (n : Nat) : Nat :=
  match n with
  | 52103 => 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def convert_base_5_to_10 (n : Nat) : Nat :=
  match n with
  | 43120 => 4 * 5^4 + 3 * 5^3 + 1 * 5^2 + 2 * 5^1 + 0 * 5^0
  | _ => 0

theorem base_conversion_subtraction : 
  convert_base_7_to_10 52103 - convert_base_5_to_10 43120 = 9833 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l312_31240


namespace NUMINAMATH_GPT_negation_of_existential_l312_31254

theorem negation_of_existential (P : Prop) :
  (¬ (∃ x : ℝ, x ^ 3 > 0)) ↔ (∀ x : ℝ, x ^ 3 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l312_31254


namespace NUMINAMATH_GPT_sum_of_roots_l312_31232

variable {p m n : ℝ}

axiom roots_condition (h : m * n = 4) : m + n = -4

theorem sum_of_roots (h : m * n = 4) : m + n = -4 := 
  roots_condition h

end NUMINAMATH_GPT_sum_of_roots_l312_31232


namespace NUMINAMATH_GPT_remainder_divisor_l312_31233

theorem remainder_divisor (d r : ℤ) (h1 : d > 1) 
  (h2 : 2024 % d = r) (h3 : 3250 % d = r) (h4 : 4330 % d = r) : d - r = 2 := 
by
  sorry

end NUMINAMATH_GPT_remainder_divisor_l312_31233


namespace NUMINAMATH_GPT_complete_square_l312_31296

theorem complete_square :
  (∀ x: ℝ, 2 * x^2 - 4 * x + 1 = 2 * (x - 1)^2 - 1) := 
by
  intro x
  sorry

end NUMINAMATH_GPT_complete_square_l312_31296


namespace NUMINAMATH_GPT_Matthew_initial_cakes_l312_31272

theorem Matthew_initial_cakes (n_cakes : ℕ) (n_crackers : ℕ) (n_friends : ℕ) (crackers_per_person : ℕ) :
  n_friends = 4 →
  n_crackers = 32 →
  crackers_per_person = 8 →
  n_crackers = n_friends * crackers_per_person →
  n_cakes = n_friends * crackers_per_person →
  n_cakes = 32 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  exact h5

end NUMINAMATH_GPT_Matthew_initial_cakes_l312_31272


namespace NUMINAMATH_GPT_wilson_pays_total_l312_31200

def hamburger_price : ℝ := 5
def cola_price : ℝ := 2
def fries_price : ℝ := 3
def sundae_price : ℝ := 4
def discount_coupon : ℝ := 4
def loyalty_discount : ℝ := 0.10

def total_cost_before_discounts : ℝ :=
  2 * hamburger_price + 3 * cola_price + fries_price + sundae_price

def total_cost_after_coupon : ℝ :=
  total_cost_before_discounts - discount_coupon

def loyalty_discount_amount : ℝ :=
  loyalty_discount * total_cost_after_coupon

def total_cost_after_all_discounts : ℝ :=
  total_cost_after_coupon - loyalty_discount_amount

theorem wilson_pays_total : total_cost_after_all_discounts = 17.10 :=
  sorry

end NUMINAMATH_GPT_wilson_pays_total_l312_31200


namespace NUMINAMATH_GPT_action_figures_added_l312_31289

-- Definitions according to conditions
def initial_action_figures : ℕ := 4
def books_on_shelf : ℕ := 22 -- This information is not necessary for proving the action figures added
def total_action_figures_after_adding : ℕ := 10

-- Theorem to prove given the conditions
theorem action_figures_added : (total_action_figures_after_adding - initial_action_figures) = 6 := by
  sorry

end NUMINAMATH_GPT_action_figures_added_l312_31289


namespace NUMINAMATH_GPT_abs_diff_mn_sqrt_eight_l312_31236

theorem abs_diff_mn_sqrt_eight {m n p : ℝ} (h1 : m * n = 6) (h2 : m + n + p = 7) (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_mn_sqrt_eight_l312_31236


namespace NUMINAMATH_GPT_distinct_factorizations_72_l312_31203

-- Define the function D that calculates the number of distinct factorizations.
noncomputable def D (n : Nat) : Nat := 
  -- Placeholder function to represent D, the actual implementation is skipped.
  sorry

-- Theorem stating the number of distinct factorizations of 72 considering the order of factors.
theorem distinct_factorizations_72 : D 72 = 119 :=
  sorry

end NUMINAMATH_GPT_distinct_factorizations_72_l312_31203


namespace NUMINAMATH_GPT_intersection_with_y_axis_l312_31265

theorem intersection_with_y_axis :
  ∃ (x y : ℝ), x = 0 ∧ y = 5 * x - 6 ∧ (x, y) = (0, -6) := 
sorry

end NUMINAMATH_GPT_intersection_with_y_axis_l312_31265


namespace NUMINAMATH_GPT_op_example_l312_31231

def op (a b : ℚ) : ℚ := a * b / (a + b)

theorem op_example : op (op 3 5) (op 5 4) = 60 / 59 := by
  sorry

end NUMINAMATH_GPT_op_example_l312_31231


namespace NUMINAMATH_GPT_cat_walking_rate_l312_31212

theorem cat_walking_rate :
  let resisting_time := 20 -- minutes
  let total_distance := 64 -- feet
  let total_time := 28 -- minutes
  let walking_time := total_time - resisting_time
  (total_distance / walking_time = 8) :=
by
  let resisting_time := 20
  let total_distance := 64
  let total_time := 28
  let walking_time := total_time - resisting_time
  have : total_distance / walking_time = 8 :=
    by norm_num [total_distance, walking_time]
  exact this

end NUMINAMATH_GPT_cat_walking_rate_l312_31212


namespace NUMINAMATH_GPT_operation_hash_12_6_l312_31266

axiom operation_hash (r s : ℝ) : ℝ

-- Conditions
axiom condition_1 : ∀ r : ℝ, operation_hash r 0 = r
axiom condition_2 : ∀ r s : ℝ, operation_hash r s = operation_hash s r
axiom condition_3 : ∀ r s : ℝ, operation_hash (r + 2) s = (operation_hash r s) + 2 * s + 2

-- Proof statement
theorem operation_hash_12_6 : operation_hash 12 6 = 168 :=
by
  sorry

end NUMINAMATH_GPT_operation_hash_12_6_l312_31266


namespace NUMINAMATH_GPT_cracker_calories_l312_31275

theorem cracker_calories (cc : ℕ) (hc1 : ∀ (n : ℕ), n = 50 → cc = 50) (hc2 : ∀ (n : ℕ), n = 7 → 7 * 50 = 350) (hc3 : ∀ (n : ℕ), n = 10 * cc → 10 * cc = 10 * cc) (hc4 : 350 + 10 * cc = 500) : cc = 15 :=
by
  sorry

end NUMINAMATH_GPT_cracker_calories_l312_31275


namespace NUMINAMATH_GPT_root_exists_between_a_and_b_l312_31260

variable {α : Type*} [LinearOrderedField α]

theorem root_exists_between_a_and_b (a b p q : α) (h₁ : a^2 + p * a + q = 0) (h₂ : b^2 - p * b - q = 0) (h₃ : q ≠ 0) :
  ∃ c, a < c ∧ c < b ∧ (c^2 + 2 * p * c + 2 * q = 0) := by
  sorry

end NUMINAMATH_GPT_root_exists_between_a_and_b_l312_31260


namespace NUMINAMATH_GPT_canoes_to_kayaks_ratio_l312_31291

theorem canoes_to_kayaks_ratio
  (canoe_cost kayak_cost total_revenue canoes_more_than_kayaks : ℕ)
  (H1 : canoe_cost = 14)
  (H2 : kayak_cost = 15)
  (H3 : total_revenue = 288)
  (H4 : ∃ C K : ℕ, C = K + canoes_more_than_kayaks ∧ 14 * C + 15 * K = 288) :
  ∃ (r : ℚ), r = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_canoes_to_kayaks_ratio_l312_31291


namespace NUMINAMATH_GPT_ratio_of_donations_l312_31259

theorem ratio_of_donations (x : ℝ) (h1 : ∀ (y : ℝ), y = 40) (h2 : ∀ (y : ℝ), y = 40 * x)
  (h3 : ∀ (y : ℝ), y = 0.30 * (40 + 40 * x)) (h4 : ∀ (y : ℝ), y = 36) : x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_donations_l312_31259


namespace NUMINAMATH_GPT_problem_l312_31207

variables (x y z : ℝ)

theorem problem :
  x - y - z = 3 ∧ yz - xy - xz = 3 → x^2 + y^2 + z^2 = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l312_31207
