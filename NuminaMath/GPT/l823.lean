import Mathlib

namespace NUMINAMATH_GPT_moles_of_KCl_formed_l823_82376

variables (NaCl KNO3 KCl NaNO3 : Type) 

-- Define the moles of each compound
variables (moles_NaCl moles_KNO3 moles_KCl moles_NaNO3 : ℕ)

-- Initial conditions
axiom initial_NaCl_condition : moles_NaCl = 2
axiom initial_KNO3_condition : moles_KNO3 = 2

-- Reaction definition
axiom reaction : moles_KCl = moles_NaCl

theorem moles_of_KCl_formed :
  moles_KCl = 2 :=
by sorry

end NUMINAMATH_GPT_moles_of_KCl_formed_l823_82376


namespace NUMINAMATH_GPT_max_area_rect_l823_82301

noncomputable def maximize_area (l w : ℕ) : ℕ :=
  l * w

theorem max_area_rect (l w: ℕ) (hl_even : l % 2 = 0) (h_perim : 2*l + 2*w = 40) :
  maximize_area l w = 100 :=
by
  sorry 

end NUMINAMATH_GPT_max_area_rect_l823_82301


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l823_82357

theorem arithmetic_mean_of_fractions :
  (3/8 + 5/9) / 2 = 67 / 144 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l823_82357


namespace NUMINAMATH_GPT_clock_hands_form_right_angle_at_180_over_11_l823_82383

-- Define the angular speeds as constants
def ω_hour : ℝ := 0.5  -- Degrees per minute
def ω_minute : ℝ := 6  -- Degrees per minute

-- Function to calculate the angle of the hour hand after t minutes
def angle_hour (t : ℝ) : ℝ := ω_hour * t

-- Function to calculate the angle of the minute hand after t minutes
def angle_minute (t : ℝ) : ℝ := ω_minute * t

-- Theorem: Prove the two hands form a right angle at the given time
theorem clock_hands_form_right_angle_at_180_over_11 : 
  ∃ t : ℝ, (6 * t - 0.5 * t = 90) ∧ t = 180 / 11 :=
by 
  -- This is where the proof would go, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_clock_hands_form_right_angle_at_180_over_11_l823_82383


namespace NUMINAMATH_GPT_find_a_if_line_passes_through_center_l823_82338

-- Define the given circle equation
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the given line equation
def line_eqn (x y a : ℝ) : Prop := 3*x + y + a = 0

-- The coordinates of the center of the circle
def center_of_circle : (ℝ × ℝ) := (-1, 2)

-- Prove that a = 1 if the line passes through the center of the circle
theorem find_a_if_line_passes_through_center (a : ℝ) :
  line_eqn (-1) 2 a → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_line_passes_through_center_l823_82338


namespace NUMINAMATH_GPT_exists_three_digit_numbers_with_property_l823_82381

open Nat

def is_three_digit_number (n : ℕ) : Prop := (100 ≤ n ∧ n < 1000)

def distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def inserts_zeros_and_is_square (n : ℕ) (k : ℕ) : Prop :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  let transformed_number := a * 10^(2*k + 2) + b * 10^(k + 1) + c
  ∃ x : ℕ, transformed_number = x * x

theorem exists_three_digit_numbers_with_property:
  ∃ n1 n2 : ℕ, 
    is_three_digit_number n1 ∧ 
    is_three_digit_number n2 ∧ 
    distinct_digits n1 ∧ 
    distinct_digits n2 ∧ 
    ( ∀ k, inserts_zeros_and_is_square n1 k ) ∧ 
    ( ∀ k, inserts_zeros_and_is_square n2 k ) ∧ 
    n1 ≠ n2 := 
sorry

end NUMINAMATH_GPT_exists_three_digit_numbers_with_property_l823_82381


namespace NUMINAMATH_GPT_max_int_difference_l823_82326

theorem max_int_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) : 
  y - x = 5 :=
sorry

end NUMINAMATH_GPT_max_int_difference_l823_82326


namespace NUMINAMATH_GPT_ellipse_eccentricity_proof_l823_82366

theorem ellipse_eccentricity_proof (a b c : ℝ) 
  (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hc_gt_zero : c > 0)
  (h_ellipse : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_r : ∃ r : ℝ, r = (Real.sqrt 2 / 6) * c) :
  (Real.sqrt (1 - b^2 / a^2)) = (2 * Real.sqrt 5 / 5) := by {
  sorry
}

end NUMINAMATH_GPT_ellipse_eccentricity_proof_l823_82366


namespace NUMINAMATH_GPT_ratio_of_areas_l823_82390

theorem ratio_of_areas (r : ℝ) (s1 s2 : ℝ) 
  (h1 : s1^2 = 4 / 5 * r^2)
  (h2 : s2^2 = 2 * r^2) :
  (s1^2 / s2^2) = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l823_82390


namespace NUMINAMATH_GPT_total_notebooks_distributed_l823_82341

/-- Define the parameters for children in Class A and Class B and the conditions given. -/
def ClassAChildren : ℕ := 64
def ClassBChildren : ℕ := 13

/-- Define the conditions as per the problem -/
def notebooksPerChildInClassA (A : ℕ) : ℕ := A / 8
def notebooksPerChildInClassB (A : ℕ) : ℕ := 2 * A
def totalChildrenClasses (A B : ℕ) : ℕ := A + B
def totalChildrenCondition (A : ℕ) : ℕ := 6 * A / 5

/-- Theorem to state the number of notebooks distributed between the two classes -/
theorem total_notebooks_distributed (A : ℕ) (B : ℕ) (H : A = 64) (H1 : B = 13) : 
  (A * (A / 8) + B * (2 * A)) = 2176 := by
  -- Conditions from the problem
  have conditionA : A = 64 := H
  have conditionB : B = 13 := H1
  have classA_notebooks : ℕ := (notebooksPerChildInClassA A) * A
  have classB_notebooks : ℕ := (notebooksPerChildInClassB A) * B
  have total_notebooks : ℕ := classA_notebooks + classB_notebooks
  -- Proof that total notebooks equals 2176
  sorry

end NUMINAMATH_GPT_total_notebooks_distributed_l823_82341


namespace NUMINAMATH_GPT_truck_capacity_rental_plan_l823_82388

-- Define the variables for the number of boxes each type of truck can carry
variables {x y : ℕ}

-- Define the conditions for the number of boxes carried by trucks
axiom cond1 : 15 * x + 25 * y = 750
axiom cond2 : 10 * x + 30 * y = 700

-- Problem 1: Prove x = 25 and y = 15
theorem truck_capacity : x = 25 ∧ y = 15 :=
by
  sorry

-- Define the variables for the number of each type of truck
variables {m : ℕ}

-- Define the conditions for the total number of trucks and boxes to be carried
axiom cond3 : 25 * m + 15 * (70 - m) ≤ 1245
axiom cond4 : 70 - m ≤ 3 * m

-- Problem 2: Prove there is one valid rental plan with m = 18 and 70-m = 52
theorem rental_plan : 17 ≤ m ∧ m ≤ 19 ∧ 70 - m ≤ 3 * m ∧ (70-m = 52 → m = 18) :=
by
  sorry

end NUMINAMATH_GPT_truck_capacity_rental_plan_l823_82388


namespace NUMINAMATH_GPT_total_oak_trees_after_planting_l823_82324

-- Definitions based on conditions
def initial_oak_trees : ℕ := 5
def new_oak_trees : ℕ := 4

-- Statement of the problem and solution
theorem total_oak_trees_after_planting : initial_oak_trees + new_oak_trees = 9 := by
  sorry

end NUMINAMATH_GPT_total_oak_trees_after_planting_l823_82324


namespace NUMINAMATH_GPT_find_y_z_l823_82353

theorem find_y_z (x y z : ℚ) (h1 : (x + y) / (z - x) = 9 / 2) (h2 : (y + z) / (y - x) = 5) (h3 : x = 43 / 4) :
  y = 12 / 17 + 17 ∧ z = 5 / 68 + 17 := 
by sorry

end NUMINAMATH_GPT_find_y_z_l823_82353


namespace NUMINAMATH_GPT_sum_fractions_correct_l823_82374

def sum_of_fractions : Prop :=
  (3 / 15 + 5 / 150 + 7 / 1500 + 9 / 15000 = 0.2386)

theorem sum_fractions_correct : sum_of_fractions :=
by
  sorry

end NUMINAMATH_GPT_sum_fractions_correct_l823_82374


namespace NUMINAMATH_GPT_initial_markup_percentage_l823_82302

-- Conditions:
-- 1. Initial price of the coat is $76.
-- 2. Increasing the price by $4 results in a 100% markup.
-- 3. A 100% markup implies the selling price is double the wholesale price.

theorem initial_markup_percentage (W : ℝ) (h1 : W + (76 - W) = 76)
  (h2 : 2 * W = 76 + 4) : (36 / 40) * 100 = 90 :=
by
  -- Using the conditions directly from the problem, we need to prove the theorem statement.
  sorry

end NUMINAMATH_GPT_initial_markup_percentage_l823_82302


namespace NUMINAMATH_GPT_icosahedron_to_octahedron_l823_82332

theorem icosahedron_to_octahedron : 
  ∃ (f : Finset (Fin 20)), f.card = 8 ∧ 
  (∀ {o : Finset (Fin 8)}, (True ∧ True)) ∧
  (∃ n : ℕ, n = 5) := by
  sorry

end NUMINAMATH_GPT_icosahedron_to_octahedron_l823_82332


namespace NUMINAMATH_GPT_fourth_divisor_of_9600_l823_82315

theorem fourth_divisor_of_9600 (x : ℕ) (h1 : ∀ (d : ℕ), d = 15 ∨ d = 25 ∨ d = 40 → 9600 % d = 0) 
  (h2 : 9600 / Nat.lcm (Nat.lcm 15 25) 40 = x) : x = 16 := by
  sorry

end NUMINAMATH_GPT_fourth_divisor_of_9600_l823_82315


namespace NUMINAMATH_GPT_find_y_l823_82345

theorem find_y (x y : ℤ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 :=
sorry

end NUMINAMATH_GPT_find_y_l823_82345


namespace NUMINAMATH_GPT_determine_angle_B_l823_82343

noncomputable def problem_statement (A B C : ℝ) (a b c : ℝ) : Prop :=
  (2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)
  ∧ (a = 8)
  ∧ (b = Real.sqrt 3)

theorem determine_angle_B (A B C : ℝ) (a b c : ℝ)
  (h : problem_statement A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
by 
  sorry

end NUMINAMATH_GPT_determine_angle_B_l823_82343


namespace NUMINAMATH_GPT_complement_union_l823_82342

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {-1, 2}

def B : Set Int := {-1, 0, 1}

theorem complement_union :
  (U \ B) ∪ A = {-2, -1, 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l823_82342


namespace NUMINAMATH_GPT_geometric_series_sum_l823_82395

theorem geometric_series_sum :
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  (3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = S) → S = 765 :=
by
  -- conditions
  let a := 3
  let r := 2
  let n := 8
  let S := (a * (1 - r^n)) / (1 - r)
  have h : 3 * (1 - 2^n) / (1 - 2) = 765 := sorry
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l823_82395


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l823_82310

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) (h : a + b = 1) (h1 : 2 * a * x - b * y = 1) : x = 1/2 ∧ y = -1 :=
by 
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l823_82310


namespace NUMINAMATH_GPT_infinite_integers_repr_l823_82375

theorem infinite_integers_repr : ∀ (k : ℕ), k > 1 →
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
by
  intros k hk
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  sorry

end NUMINAMATH_GPT_infinite_integers_repr_l823_82375


namespace NUMINAMATH_GPT_least_possible_integer_l823_82333

theorem least_possible_integer (N : ℕ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ N) ∧
  (∀ m : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ m) → N ≤ m) →
  N = 2329089562800 :=
sorry

end NUMINAMATH_GPT_least_possible_integer_l823_82333


namespace NUMINAMATH_GPT_inequality_am_gm_l823_82309

theorem inequality_am_gm (a b c d : ℝ) (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l823_82309


namespace NUMINAMATH_GPT_salary_increase_l823_82372

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 0.65 * S = 0.5 * S + (P / 100) * (0.5 * S)) : P = 30 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_salary_increase_l823_82372


namespace NUMINAMATH_GPT_boats_left_l823_82344

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end NUMINAMATH_GPT_boats_left_l823_82344


namespace NUMINAMATH_GPT_pairs_sold_l823_82316

theorem pairs_sold (total_sales : ℝ) (avg_price_per_pair : ℝ) (h1 : total_sales = 490) (h2 : avg_price_per_pair = 9.8) :
  total_sales / avg_price_per_pair = 50 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_pairs_sold_l823_82316


namespace NUMINAMATH_GPT_ms_emily_inheritance_l823_82363

theorem ms_emily_inheritance :
  ∃ (y : ℝ), 
    (0.25 * y + 0.15 * (y - 0.25 * y) = 19500) ∧
    (y = 53800) :=
by
  sorry

end NUMINAMATH_GPT_ms_emily_inheritance_l823_82363


namespace NUMINAMATH_GPT_balloon_arrangement_count_l823_82347

theorem balloon_arrangement_count :
  let total_permutations := (Nat.factorial 7) / (Nat.factorial 2 * Nat.factorial 3)
  let ways_to_arrange_L_and_O := Nat.choose 4 1 * (Nat.factorial 3)
  let valid_arrangements := ways_to_arrange_L_and_O * total_permutations
  valid_arrangements = 10080 :=
by
  sorry

end NUMINAMATH_GPT_balloon_arrangement_count_l823_82347


namespace NUMINAMATH_GPT_unique_solution_l823_82355

theorem unique_solution (x : ℝ) : (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x ↔ x = 3 := by
  sorry

end NUMINAMATH_GPT_unique_solution_l823_82355


namespace NUMINAMATH_GPT_third_wins_against_seventh_l823_82397

-- Define the participants and their distinct points 
variables (p : ℕ → ℕ) (h_distinct : ∀ i j, i ≠ j → p i ≠ p j)
-- descending order condition
variables (h_order : ∀ i j, i < j → p i > p j)
-- second place points equals sum of last four places
variables (h_second : p 2 = p 5 + p 6 + p 7 + p 8)

-- Theorem stating the third place player won against the seventh place player
theorem third_wins_against_seventh :
  p 3 > p 7 :=
sorry

end NUMINAMATH_GPT_third_wins_against_seventh_l823_82397


namespace NUMINAMATH_GPT_buyers_of_cake_mix_l823_82330

/-
  A certain manufacturer of cake, muffin, and bread mixes has 100 buyers,
  of whom some purchase cake mix, 40 purchase muffin mix, and 17 purchase both cake mix and muffin mix.
  If a buyer is to be selected at random from the 100 buyers, the probability that the buyer selected will be one who purchases 
  neither cake mix nor muffin mix is 0.27.
  Prove that the number of buyers who purchase cake mix is 50.
-/

theorem buyers_of_cake_mix (C M B total : ℕ) (hM : M = 40) (hB : B = 17) (hTotal : total = 100)
    (hProb : (total - (C + M - B) : ℝ) / total = 0.27) : C = 50 :=
by
  -- Definition of the proof is required here
  sorry

end NUMINAMATH_GPT_buyers_of_cake_mix_l823_82330


namespace NUMINAMATH_GPT_max_neg_p_l823_82313

theorem max_neg_p (p : ℤ) (h1 : p < 0) (h2 : ∃ k : ℤ, 2001 + p = k^2) : p ≤ -65 :=
by
  sorry

end NUMINAMATH_GPT_max_neg_p_l823_82313


namespace NUMINAMATH_GPT_solve_for_y_l823_82362

theorem solve_for_y (y : ℤ) : 
  7 * (4 * y + 3) - 3 = -3 * (2 - 9 * y) → y = -24 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_y_l823_82362


namespace NUMINAMATH_GPT_number_of_B_students_l823_82360

theorem number_of_B_students (x : ℝ) (h1 : 0.8 * x + x + 1.2 * x = 40) : x = 13 :=
  sorry

end NUMINAMATH_GPT_number_of_B_students_l823_82360


namespace NUMINAMATH_GPT_simplify_expression_l823_82307

theorem simplify_expression (x : ℝ) (hx : x ≠ 4):
  (x^2 - 4 * x) / (x^2 - 8 * x + 16) = x / (x - 4) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l823_82307


namespace NUMINAMATH_GPT_number_of_six_digit_integers_l823_82329

-- Define the problem conditions
def digits := [1, 1, 3, 3, 7, 8]

-- State the theorem
theorem number_of_six_digit_integers : 
  (List.permutations digits).length = 180 := 
by sorry

end NUMINAMATH_GPT_number_of_six_digit_integers_l823_82329


namespace NUMINAMATH_GPT_contractor_fine_per_absent_day_l823_82380

theorem contractor_fine_per_absent_day :
  ∀ (total_days absent_days wage_per_day total_receipt fine_per_absent_day : ℝ),
    total_days = 30 →
    wage_per_day = 25 →
    absent_days = 4 →
    total_receipt = 620 →
    (total_days - absent_days) * wage_per_day - absent_days * fine_per_absent_day = total_receipt →
    fine_per_absent_day = 7.50 :=
by
  intros total_days absent_days wage_per_day total_receipt fine_per_absent_day
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_contractor_fine_per_absent_day_l823_82380


namespace NUMINAMATH_GPT_abs_val_equality_l823_82335

theorem abs_val_equality (m : ℝ) (h : |m| = |(-3 : ℝ)|) : m = 3 ∨ m = -3 :=
sorry

end NUMINAMATH_GPT_abs_val_equality_l823_82335


namespace NUMINAMATH_GPT_one_sixth_of_x_l823_82399

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 :=
sorry

end NUMINAMATH_GPT_one_sixth_of_x_l823_82399


namespace NUMINAMATH_GPT_mike_eggs_basket_l823_82385

theorem mike_eggs_basket : ∃ k : ℕ, (30 % k = 0) ∧ (42 % k = 0) ∧ k ≥ 4 ∧ (30 / k) ≥ 3 ∧ (42 / k) ≥ 3 ∧ k = 6 := 
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_mike_eggs_basket_l823_82385


namespace NUMINAMATH_GPT_min_cosine_largest_angle_l823_82308

theorem min_cosine_largest_angle (a b c : ℕ → ℝ) 
  (triangle_inequality: ∀ i, a i ≤ b i ∧ b i ≤ c i)
  (pythagorean_inequality: ∀ i, (a i)^2 + (b i)^2 ≥ (c i)^2)
  (A : ℝ := ∑' i, a i)
  (B : ℝ := ∑' i, b i)
  (C : ℝ := ∑' i, c i) :
  (A^2 + B^2 - C^2) / (2 * A * B) ≥ 1 - (Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_min_cosine_largest_angle_l823_82308


namespace NUMINAMATH_GPT_each_client_selected_cars_l823_82321

theorem each_client_selected_cars (cars clients selections : ℕ) (h1 : cars = 16) (h2 : selections = 3 * cars) (h3 : clients = 24) :
  selections / clients = 2 :=
by
  sorry

end NUMINAMATH_GPT_each_client_selected_cars_l823_82321


namespace NUMINAMATH_GPT_relationship_between_3a_3b_4a_l823_82358

variable (a b : ℝ)
variable (h : a > b)
variable (hb : b > 0)

theorem relationship_between_3a_3b_4a (a b : ℝ) (h : a > b) (hb : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := 
by
  sorry

end NUMINAMATH_GPT_relationship_between_3a_3b_4a_l823_82358


namespace NUMINAMATH_GPT_gcd_18_30_l823_82354

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_18_30_l823_82354


namespace NUMINAMATH_GPT_probability_of_divisor_of_6_is_two_thirds_l823_82391

noncomputable def probability_divisor_of_6 : ℚ :=
  have divisors_of_6 : Finset ℕ := {1, 2, 3, 6}
  have total_possible_outcomes : ℕ := 6
  have favorable_outcomes : ℕ := 4
  have probability_event : ℚ := favorable_outcomes / total_possible_outcomes
  2 / 3

theorem probability_of_divisor_of_6_is_two_thirds :
  probability_divisor_of_6 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_probability_of_divisor_of_6_is_two_thirds_l823_82391


namespace NUMINAMATH_GPT_helen_choc_chip_yesterday_l823_82314

variable (total_cookies morning_cookies : ℕ)

theorem helen_choc_chip_yesterday :
  total_cookies = 1081 →
  morning_cookies = 554 →
  total_cookies - morning_cookies = 527 := by
  sorry

end NUMINAMATH_GPT_helen_choc_chip_yesterday_l823_82314


namespace NUMINAMATH_GPT_dress_designs_count_l823_82349

-- Define the number of colors, fabric types, and patterns
def num_colors : Nat := 3
def num_fabric_types : Nat := 4
def num_patterns : Nat := 3

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_fabric_types * num_patterns

-- Define the theorem to prove the equivalence
theorem dress_designs_count :
  total_dress_designs = 36 :=
by
  -- This is to show the theorem's structure; proof will be added here.
  sorry

end NUMINAMATH_GPT_dress_designs_count_l823_82349


namespace NUMINAMATH_GPT_min_value_of_expression_l823_82328

theorem min_value_of_expression {x y z : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) : 
  (x + 1 / y) * (x + 1 / z) >= Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l823_82328


namespace NUMINAMATH_GPT_circles_intersect_l823_82377

noncomputable def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def circle2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 9}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 :=
sorry

end NUMINAMATH_GPT_circles_intersect_l823_82377


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l823_82325

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  a + 10 * d = 5.25 → 
  a + 6 * d = 3.25 → 
  (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 56.25 → 
  n = 15 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l823_82325


namespace NUMINAMATH_GPT_find_pairs_l823_82351

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ n : ℕ, (n > 0) ∧ (a = n ∧ b = n) ∨ (a = n ∧ b = 1)) ↔ 
  (a^3 ∣ b^2) ∧ ((b - 1) ∣ (a - 1)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_pairs_l823_82351


namespace NUMINAMATH_GPT_Seokgi_candies_l823_82389

theorem Seokgi_candies (C : ℕ) 
  (h1 : C / 2 + (C - C / 2) / 3 + 12 = C)
  (h2 : ∃ x, x = 12) :
  C = 36 := 
by 
  sorry

end NUMINAMATH_GPT_Seokgi_candies_l823_82389


namespace NUMINAMATH_GPT_cost_of_tax_free_items_l823_82384

theorem cost_of_tax_free_items (total_cost : ℝ) (tax_40_percent : ℝ) 
  (tax_30_percent : ℝ) (discount : ℝ) : 
  (total_cost = 120) →
  (tax_40_percent = 0.4 * total_cost) →
  (tax_30_percent = 0.3 * total_cost) →
  (discount = 0.05 * tax_30_percent) →
  (tax-free_items = total_cost - (tax_40_percent + (tax_30_percent - discount))) → 
  tax_free_items = 36 :=
by sorry

end NUMINAMATH_GPT_cost_of_tax_free_items_l823_82384


namespace NUMINAMATH_GPT_diversity_values_l823_82361

theorem diversity_values (k : ℕ) (h : 1 ≤ k ∧ k ≤ 4) :
  ∃ (D : ℕ), D = 1000 * (k - 1) := by
  sorry

end NUMINAMATH_GPT_diversity_values_l823_82361


namespace NUMINAMATH_GPT_James_average_speed_l823_82392

theorem James_average_speed (TotalDistance : ℝ) (BreakTime : ℝ) (TotalTripTime : ℝ) (h1 : TotalDistance = 42) (h2 : BreakTime = 1) (h3 : TotalTripTime = 9) :
  (TotalDistance / (TotalTripTime - BreakTime)) = 5.25 :=
by
  sorry

end NUMINAMATH_GPT_James_average_speed_l823_82392


namespace NUMINAMATH_GPT_ratio_of_areas_of_circles_l823_82303

theorem ratio_of_areas_of_circles
    (C_C R_C C_D R_D L : ℝ)
    (hC : C_C = 2 * Real.pi * R_C)
    (hD : C_D = 2 * Real.pi * R_D)
    (hL : (60 / 360) * C_C = L ∧ L = (40 / 360) * C_D) :
    (Real.pi * R_C ^ 2) / (Real.pi * R_D ^ 2) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_circles_l823_82303


namespace NUMINAMATH_GPT_smaller_integer_is_49_l823_82373

theorem smaller_integer_is_49 (m n : ℕ) (hm : 10 ≤ m ∧ m < 100) (hn : 10 ≤ n ∧ n < 100)
  (h : (m + n) / 2 = m + n / 100) : min m n = 49 :=
by
  sorry

end NUMINAMATH_GPT_smaller_integer_is_49_l823_82373


namespace NUMINAMATH_GPT_real_roots_of_quadratics_l823_82359

theorem real_roots_of_quadratics {p1 p2 q1 q2 : ℝ} (h : p1 * p2 = 2 * (q1 + q2)) :
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  have D1 := p1^2 - 4 * q1
  have D2 := p2^2 - 4 * q2
  sorry

end NUMINAMATH_GPT_real_roots_of_quadratics_l823_82359


namespace NUMINAMATH_GPT_pattern_C_not_foldable_without_overlap_l823_82318

-- Define the four patterns, denoted as PatternA, PatternB, PatternC, and PatternD.
inductive Pattern
| A : Pattern
| B : Pattern
| C : Pattern
| D : Pattern

-- Define a predicate for a pattern being foldable into a cube without overlap.
def foldable_into_cube (p : Pattern) : Prop := sorry

theorem pattern_C_not_foldable_without_overlap : ¬ foldable_into_cube Pattern.C := sorry

end NUMINAMATH_GPT_pattern_C_not_foldable_without_overlap_l823_82318


namespace NUMINAMATH_GPT_triangle_base_angles_eq_l823_82334

theorem triangle_base_angles_eq
  (A B C C1 C2 : ℝ)
  (h1 : A > B)
  (h2 : C1 = 2 * C2)
  (h3 : A + B + C = 180)
  (h4 : B + C2 = 90)
  (h5 : C = C1 + C2) :
  A = B := by
  sorry

end NUMINAMATH_GPT_triangle_base_angles_eq_l823_82334


namespace NUMINAMATH_GPT_find_integer_m_l823_82305

theorem find_integer_m 
  (m : ℤ) (h_pos : m > 0) 
  (h_intersect : ∃ (x y : ℤ), 17 * x + 7 * y = 1000 ∧ y = m * x + 2) : 
  m = 68 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_m_l823_82305


namespace NUMINAMATH_GPT_solve_for_x_l823_82394

theorem solve_for_x (x : ℝ) (h : (x / 3) / 3 = 9 / (x / 3)) : x = 3 ^ (5 / 2) ∨ x = -3 ^ (5 / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l823_82394


namespace NUMINAMATH_GPT_max_gcd_lcm_condition_l823_82312

theorem max_gcd_lcm_condition (a b c : ℕ) (h : gcd (lcm a b) c * lcm (gcd a b) c = 200) : gcd (lcm a b) c ≤ 10 := sorry

end NUMINAMATH_GPT_max_gcd_lcm_condition_l823_82312


namespace NUMINAMATH_GPT_number_of_student_tickets_sold_l823_82350

variable (A S : ℝ)

theorem number_of_student_tickets_sold
  (h1 : A + S = 59)
  (h2 : 4 * A + 2.5 * S = 222.50) :
  S = 9 :=
by sorry

end NUMINAMATH_GPT_number_of_student_tickets_sold_l823_82350


namespace NUMINAMATH_GPT_function_does_not_have_property_P_l823_82379

-- Definition of property P
def hasPropertyP (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≠ x2 → f ((x1 + x2) / 2) = (f x1 + f x2) / 2

-- Function in question
def f (x : ℝ) : ℝ :=
  x^2

-- Statement that function f does not have property P
theorem function_does_not_have_property_P : ¬hasPropertyP f :=
  sorry

end NUMINAMATH_GPT_function_does_not_have_property_P_l823_82379


namespace NUMINAMATH_GPT_simplify_expression_l823_82356

theorem simplify_expression :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) *
  (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) * (5^128 + 7^128) = 7^256 - 5^256 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l823_82356


namespace NUMINAMATH_GPT_find_a3_l823_82337

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

noncomputable
def arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_a3 {a : ℕ → ℤ} (d : ℤ) (h6 : a 6 = 6) (h9 : a 9 = 9) :
  (∃ d : ℤ, arithmetic_sequence a d) →
  a 3 = 3 :=
by
  intro h_arith_seq
  sorry

end NUMINAMATH_GPT_find_a3_l823_82337


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l823_82365

variable (a d : ℕ)

def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_15_terms (h : nth_term 4 + nth_term 12 = 16) : sum_of_first_n_terms 15 = 120 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l823_82365


namespace NUMINAMATH_GPT_slope_perpendicular_l823_82396

theorem slope_perpendicular (x1 y1 x2 y2 m : ℚ) 
  (hx1 : x1 = 3) (hy1 : y1 = -4) (hx2 : x2 = -6) (hy2 : y2 = 2) 
  (hm : m = (y2 - y1) / (x2 - x1)) :
  ∀ m_perpendicular: ℚ, m_perpendicular = (-1 / m) → m_perpendicular = 3/2 := 
sorry

end NUMINAMATH_GPT_slope_perpendicular_l823_82396


namespace NUMINAMATH_GPT_mutually_exclusive_event_l823_82364

theorem mutually_exclusive_event (A B C D: Prop) 
  (h_A: ¬ (A ∧ (¬D)) ∧ ¬ ¬ D)
  (h_B: ¬ (B ∧ (¬D)) ∧ ¬ ¬ D)
  (h_C: ¬ (C ∧ (¬D)) ∧ ¬ ¬ D)
  (h_D: ¬ (D ∧ (¬D)) ∧ ¬ ¬ D) :
  D :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_event_l823_82364


namespace NUMINAMATH_GPT_evaluate_expression_l823_82306

theorem evaluate_expression :
  ((gcd 54 42 |> lcm 36) * (gcd 78 66 |> gcd 90) + (lcm 108 72 |> gcd 66 |> gcd 84)) = 24624 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l823_82306


namespace NUMINAMATH_GPT_mul_eight_neg_half_l823_82393

theorem mul_eight_neg_half : 8 * (- (1/2: ℚ)) = -4 := 
by 
  sorry

end NUMINAMATH_GPT_mul_eight_neg_half_l823_82393


namespace NUMINAMATH_GPT_max_bishops_on_chessboard_l823_82339

theorem max_bishops_on_chessboard (N : ℕ) (N_pos: 0 < N) : 
  ∃ max_number : ℕ, max_number = 2 * N - 2 :=
sorry

end NUMINAMATH_GPT_max_bishops_on_chessboard_l823_82339


namespace NUMINAMATH_GPT_determine_n_l823_82311

noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem determine_n :
  (∃ n : ℕ, digit_sum (9 * (10^n - 1)) = 999 ∧ n = 111) :=
sorry

end NUMINAMATH_GPT_determine_n_l823_82311


namespace NUMINAMATH_GPT_min_positive_value_l823_82371

theorem min_positive_value (c d : ℤ) (h : c > d) : 
  ∃ x : ℝ, x = (c + 2 * d) / (c - d) + (c - d) / (c + 2 * d) ∧ x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_positive_value_l823_82371


namespace NUMINAMATH_GPT_three_tenths_of_number_l823_82300

theorem three_tenths_of_number (x : ℝ) (h : (1/3) * (1/4) * x = 15) : (3/10) * x = 54 :=
by
  sorry

end NUMINAMATH_GPT_three_tenths_of_number_l823_82300


namespace NUMINAMATH_GPT_probability_of_diamond_king_ace_l823_82317

noncomputable def probability_three_cards : ℚ :=
  (11 / 52) * (4 / 51) * (4 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (4 / 51) * (3 / 50)

theorem probability_of_diamond_king_ace :
  probability_three_cards = 284 / 132600 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_diamond_king_ace_l823_82317


namespace NUMINAMATH_GPT_paving_stones_correct_l823_82322

def paving_stone_area : ℕ := 3 * 2
def courtyard_breadth : ℕ := 6
def number_of_paving_stones : ℕ := 15
def courtyard_length : ℕ := 15

theorem paving_stones_correct : 
  number_of_paving_stones * paving_stone_area = courtyard_length * courtyard_breadth :=
by
  sorry

end NUMINAMATH_GPT_paving_stones_correct_l823_82322


namespace NUMINAMATH_GPT_total_number_of_students_l823_82378

theorem total_number_of_students 
    (T : ℕ)
    (h1 : ∃ a, a = T / 5) 
    (h2 : ∃ b, b = T / 4) 
    (h3 : ∃ c, c = T / 2) 
    (h4 : T - (T / 5 + T / 4 + T / 2) = 25) : 
  T = 500 := by 
  sorry

end NUMINAMATH_GPT_total_number_of_students_l823_82378


namespace NUMINAMATH_GPT_bottom_right_corner_value_l823_82386

variable (a b c x : ℕ)

/--
Conditions:
- The sums of the numbers in each of the four 2x2 grids forming part of the 3x3 grid are equal.
- Known values for corners: a, b, and c.
Conclusion:
- The bottom right corner value x must be 0.
-/

theorem bottom_right_corner_value (S: ℕ) (A B C D E: ℕ) :
  S = a + A + B + C →
  S = A + b + C + D →
  S = B + C + c + E →
  S = C + D + E + x →
  x = 0 :=
by
  sorry

end NUMINAMATH_GPT_bottom_right_corner_value_l823_82386


namespace NUMINAMATH_GPT_count_triangles_in_figure_l823_82382

/-- 
The figure is a rectangle divided into 8 columns and 2 rows with additional diagonal and vertical lines.
We need to prove that there are 76 triangles in total in the figure.
-/
theorem count_triangles_in_figure : 
  let columns := 8 
  let rows := 2 
  let num_triangles := 76 
  ∃ total_triangles, total_triangles = num_triangles :=
by
  sorry

end NUMINAMATH_GPT_count_triangles_in_figure_l823_82382


namespace NUMINAMATH_GPT_abs_div_one_add_i_by_i_l823_82398

noncomputable def imaginary_unit : ℂ := Complex.I

/-- The absolute value of the complex number (1 + i)/i is √2. -/
theorem abs_div_one_add_i_by_i : Complex.abs ((1 + imaginary_unit) / imaginary_unit) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_abs_div_one_add_i_by_i_l823_82398


namespace NUMINAMATH_GPT_red_ballpoint_pens_count_l823_82369

theorem red_ballpoint_pens_count (R B : ℕ) (h1: R + B = 240) (h2: B = R - 2) : R = 121 :=
by
  sorry

end NUMINAMATH_GPT_red_ballpoint_pens_count_l823_82369


namespace NUMINAMATH_GPT_greatest_common_multiple_of_10_and_15_lt_120_l823_82368

theorem greatest_common_multiple_of_10_and_15_lt_120 : 
  ∃ (m : ℕ), lcm 10 15 = 30 ∧ m ∈ {i | i < 120 ∧ ∃ (k : ℕ), i = k * 30} ∧ m = 90 := 
sorry

end NUMINAMATH_GPT_greatest_common_multiple_of_10_and_15_lt_120_l823_82368


namespace NUMINAMATH_GPT_eq_has_infinite_solutions_l823_82340

theorem eq_has_infinite_solutions (b : ℝ) (x : ℝ) :
  5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 := by
sorry

end NUMINAMATH_GPT_eq_has_infinite_solutions_l823_82340


namespace NUMINAMATH_GPT_third_group_members_l823_82346

theorem third_group_members (total_members first_group second_group : ℕ) (h₁ : total_members = 70) (h₂ : first_group = 25) (h₃ : second_group = 30) : (total_members - (first_group + second_group)) = 15 :=
sorry

end NUMINAMATH_GPT_third_group_members_l823_82346


namespace NUMINAMATH_GPT_intercepts_sum_eq_eight_l823_82336

def parabola_x_y (x y : ℝ) := x = 3 * y^2 - 9 * y + 5

theorem intercepts_sum_eq_eight :
  ∃ (a b c : ℝ), parabola_x_y a 0 ∧ parabola_x_y 0 b ∧ parabola_x_y 0 c ∧ a + b + c = 8 :=
sorry

end NUMINAMATH_GPT_intercepts_sum_eq_eight_l823_82336


namespace NUMINAMATH_GPT_binomial_minus_floor_divisible_by_seven_l823_82352

theorem binomial_minus_floor_divisible_by_seven (n : ℕ) (h : n > 7) :
  ((Nat.choose n 7 : ℤ) - ⌊(n : ℤ) / 7⌋) % 7 = 0 :=
  sorry

end NUMINAMATH_GPT_binomial_minus_floor_divisible_by_seven_l823_82352


namespace NUMINAMATH_GPT_fraction_of_As_l823_82304

-- Define the conditions
def fraction_B (T : ℕ) := 1/4 * T
def fraction_C (T : ℕ) := 1/2 * T
def remaining_D : ℕ := 20
def total_students_approx : ℕ := 400

-- State the theorem
theorem fraction_of_As 
  (F : ℚ) : 
  ∀ T : ℕ, 
  T = F * T + fraction_B T + fraction_C T + remaining_D → 
  T = total_students_approx → 
  F = 1/5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_of_As_l823_82304


namespace NUMINAMATH_GPT_find_b_for_continuity_at_2_l823_82319

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if h : x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem find_b_for_continuity_at_2 (b : ℝ) : (∀ x, f x b = if x ≤ 2 then 4 * x^2 + 5 else b * x + 3) ∧ 
  (f 2 b = 21) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x b - f 2 b| < ε) → 
  b = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_b_for_continuity_at_2_l823_82319


namespace NUMINAMATH_GPT_first_digit_base5_of_312_is_2_l823_82387

theorem first_digit_base5_of_312_is_2 :
  ∃ d : ℕ, d = 2 ∧ (∀ n : ℕ, d * 5 ^ n ≤ 312 ∧ 312 < (d + 1) * 5 ^ n) :=
by
  sorry

end NUMINAMATH_GPT_first_digit_base5_of_312_is_2_l823_82387


namespace NUMINAMATH_GPT_pentagon_area_l823_82320

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℕ := 
  let area_triangle := (1/2) * a * b
  let area_trapezoid := (1/2) * (c + e) * d
  area_triangle + area_trapezoid

theorem pentagon_area : area_of_pentagon 18 25 30 28 25 = 995 :=
by sorry

end NUMINAMATH_GPT_pentagon_area_l823_82320


namespace NUMINAMATH_GPT_find_larger_number_l823_82367

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1355) (h2 : L = 6 * S + 15) : L = 1623 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l823_82367


namespace NUMINAMATH_GPT_smallest_radius_squared_of_sphere_l823_82323

theorem smallest_radius_squared_of_sphere :
  ∃ (x y z : ℤ), 
  (x - 2)^2 + y^2 + z^2 = (x^2 + (y - 4)^2 + z^2) ∧
  (x - 2)^2 + y^2 + z^2 = (x^2 + y^2 + (z - 6)^2) ∧
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
  (∃ r, r^2 = (x - 2)^2 + (0 - y)^2 + (0 - z)^2) ∧
  51 = r^2 :=
sorry

end NUMINAMATH_GPT_smallest_radius_squared_of_sphere_l823_82323


namespace NUMINAMATH_GPT_ratio_addition_l823_82348

theorem ratio_addition (x : ℝ) : 
  (2 + x) / (3 + x) = 4 / 5 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_addition_l823_82348


namespace NUMINAMATH_GPT_mother_age_is_correct_l823_82370

variable (D M : ℕ)

theorem mother_age_is_correct:
  (D + 3 = 26) → (M - 5 = 2 * (D - 5)) → M = 41 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_mother_age_is_correct_l823_82370


namespace NUMINAMATH_GPT_measure_of_angle_B_find_a_and_c_find_perimeter_l823_82331

theorem measure_of_angle_B (a b c : ℝ) (A B C : ℝ) 
    (h : c / (b - a) = (Real.sin A + Real.sin B) / (Real.sin A + Real.sin C)) 
    (cos_B : Real.cos B = -1 / 2) : B = 2 * Real.pi / 3 :=
by
  sorry

theorem find_a_and_c (a c A C : ℝ) (S : ℝ) 
    (h1 : Real.sin C = 2 * Real.sin A) (h2 : S = 2 * Real.sqrt 3) 
    (A' : a * c = 8) : a = 2 ∧ c = 4 :=
by
  sorry

theorem find_perimeter (a b c : ℝ) 
    (h1 : b = Real.sqrt 3) (h2 : a * c = 1) 
    (h3 : a + c = 2) : a + b + c = 2 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_find_a_and_c_find_perimeter_l823_82331


namespace NUMINAMATH_GPT_quadratic_root_k_l823_82327

theorem quadratic_root_k (k : ℝ) : (∃ x : ℝ, x^2 - 2 * x + k = 0 ∧ x = 1) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_k_l823_82327
