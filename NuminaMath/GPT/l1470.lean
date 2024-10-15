import Mathlib

namespace NUMINAMATH_GPT_smallest_positive_value_l1470_147036

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℚ), k = (a^2 + b^2) / (a^2 - b^2) + (a^2 - b^2) / (a^2 + b^2) ∧ k = 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_value_l1470_147036


namespace NUMINAMATH_GPT_sum_of_coordinates_l1470_147047

def g : ℝ → ℝ := sorry
def h (x : ℝ) : ℝ := (g x)^3

theorem sum_of_coordinates (hg : g 4 = 8) : 4 + h 4 = 516 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l1470_147047


namespace NUMINAMATH_GPT_symmetric_sufficient_not_necessary_l1470_147040

theorem symmetric_sufficient_not_necessary (φ : Real) : 
    φ = - (Real.pi / 6) →
    ∃ f : Real → Real, (∀ x, f x = Real.sin (2 * x - φ)) ∧ 
    ∀ x, f (2 * (Real.pi / 6) - x) = f x :=
by
  sorry

end NUMINAMATH_GPT_symmetric_sufficient_not_necessary_l1470_147040


namespace NUMINAMATH_GPT_tan_neg_seven_pi_sixths_l1470_147024

noncomputable def tan_neg_pi_seven_sixths : Real :=
  -Real.sqrt 3 / 3

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_neg_seven_pi_sixths_l1470_147024


namespace NUMINAMATH_GPT_find_x_val_l1470_147017

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_x_val_l1470_147017


namespace NUMINAMATH_GPT_discount_percentage_l1470_147081

noncomputable def cost_price : ℝ := 100
noncomputable def profit_with_discount : ℝ := 0.32 * cost_price
noncomputable def profit_without_discount : ℝ := 0.375 * cost_price

noncomputable def sp_with_discount : ℝ := cost_price + profit_with_discount
noncomputable def sp_without_discount : ℝ := cost_price + profit_without_discount

noncomputable def discount_amount : ℝ := sp_without_discount - sp_with_discount
noncomputable def percentage_discount : ℝ := (discount_amount / sp_without_discount) * 100

theorem discount_percentage : percentage_discount = 4 :=
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_discount_percentage_l1470_147081


namespace NUMINAMATH_GPT_find_a_solution_set_a_negative_l1470_147088

-- Definitions
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  a * x^2 + (a - 1) * x - 1 ≥ 0

-- Problem 1: Prove the value of 'a'
theorem find_a (h : ∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ -1/2)) :
  a = -2 :=
sorry

-- Problem 2: Prove the solution sets when a < 0
theorem solution_set_a_negative (h : a < 0) :
  (a = -1 → (∀ x : ℝ, quadratic_inequality a x ↔ x = -1)) ∧
  (a < -1 → (∀ x : ℝ, quadratic_inequality a x ↔ (-1 ≤ x ∧ x ≤ 1/a))) ∧
  (-1 < a ∧ a < 0 → (∀ x : ℝ, quadratic_inequality a x ↔ (1/a ≤ x ∧ x ≤ -1))) :=
sorry

end NUMINAMATH_GPT_find_a_solution_set_a_negative_l1470_147088


namespace NUMINAMATH_GPT_abc_order_l1470_147038

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := 0.5^3
noncomputable def c : Real := Real.log 3 / Real.log 0.5 -- log_0.5 3 is written as (log 3) / (log 0.5) in Lean

theorem abc_order : a > b ∧ b > c :=
by
  have h1 : a = Real.sqrt 3 := rfl
  have h2 : b = 0.5^3 := rfl
  have h3 : c = Real.log 3 / Real.log 0.5 := rfl
  sorry

end NUMINAMATH_GPT_abc_order_l1470_147038


namespace NUMINAMATH_GPT_johns_age_l1470_147041

theorem johns_age :
  ∃ x : ℕ, (∃ n : ℕ, x - 5 = n^2) ∧ (∃ m : ℕ, x + 3 = m^3) ∧ x = 69 :=
by
  sorry

end NUMINAMATH_GPT_johns_age_l1470_147041


namespace NUMINAMATH_GPT_div_by_1897_l1470_147035

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end NUMINAMATH_GPT_div_by_1897_l1470_147035


namespace NUMINAMATH_GPT_largest_product_of_three_l1470_147011

theorem largest_product_of_three :
  ∃ (a b c : ℤ), a ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 b ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 c ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
                 a * b * c = 90 := 
sorry

end NUMINAMATH_GPT_largest_product_of_three_l1470_147011


namespace NUMINAMATH_GPT_point_D_coordinates_l1470_147095

noncomputable def point := ℝ × ℝ

def A : point := (2, 3)
def B : point := (-1, 5)

def vector_sub (p1 p2 : point) : point := (p1.1 - p2.1, p1.2 - p2.2)
def scalar_mul (k : ℝ) (v : point) : point := (k * v.1, k * v.2)
def vector_add (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)

def D : point := vector_add A (scalar_mul 3 (vector_sub B A))

theorem point_D_coordinates : D = (-7, 9) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_point_D_coordinates_l1470_147095


namespace NUMINAMATH_GPT_find_length_of_street_l1470_147080

-- Definitions based on conditions
def area_street (L : ℝ) : ℝ := L^2
def area_forest (L : ℝ) : ℝ := 3 * (area_street L)
def num_trees (L : ℝ) : ℝ := 4 * (area_forest L)

-- Statement to prove
theorem find_length_of_street (L : ℝ) (h : num_trees L = 120000) : L = 100 := by
  sorry

end NUMINAMATH_GPT_find_length_of_street_l1470_147080


namespace NUMINAMATH_GPT_probability_drawing_red_l1470_147052

/-- The probability of drawing a red ball from a bag that contains 1 red ball and 2 yellow balls. -/
theorem probability_drawing_red : 
  let N_red := 1
  let N_yellow := 2
  let N_total := N_red + N_yellow
  let P_red := (N_red : ℝ) / N_total
  P_red = (1 : ℝ) / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_drawing_red_l1470_147052


namespace NUMINAMATH_GPT_max_truthful_gnomes_l1470_147085

theorem max_truthful_gnomes :
  ∀ (heights : Fin 7 → ℝ), 
    heights 0 = 60 →
    heights 1 = 61 →
    heights 2 = 62 →
    heights 3 = 63 →
    heights 4 = 64 →
    heights 5 = 65 →
    heights 6 = 66 →
      (∃ i : Fin 7, ∀ j : Fin 7, (i ≠ j → heights j ≠ (60 + j.1)) ∧ (i = j → heights j = (60 + j.1))) :=
by
  intro heights h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_max_truthful_gnomes_l1470_147085


namespace NUMINAMATH_GPT_composite_integer_expression_l1470_147069

theorem composite_integer_expression (n : ℕ) (h : n > 1) (hn : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 :=
by
  sorry

end NUMINAMATH_GPT_composite_integer_expression_l1470_147069


namespace NUMINAMATH_GPT_prob_product_less_than_36_is_15_over_16_l1470_147007

noncomputable def prob_product_less_than_36 : ℚ := sorry

theorem prob_product_less_than_36_is_15_over_16 :
  prob_product_less_than_36 = 15 / 16 := 
sorry

end NUMINAMATH_GPT_prob_product_less_than_36_is_15_over_16_l1470_147007


namespace NUMINAMATH_GPT_value_of_c_l1470_147094

theorem value_of_c (a b c : ℚ) (h1 : a / b = 2 / 3) (h2 : b / c = 3 / 7) (h3 : a - b + 3 = c - 2 * b) : c = 21 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_c_l1470_147094


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l1470_147097

theorem tangent_line_to_parabola (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 16 * x) →
  (28 ^ 2 - 4 * 1 * (4 * k) = 0) → k = 49 :=
by
  intro h
  intro h_discriminant
  have discriminant_eq_zero : 28 ^ 2 - 4 * 1 * (4 * k) = 0 := h_discriminant
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l1470_147097


namespace NUMINAMATH_GPT_rotated_ellipse_sum_is_four_l1470_147059

noncomputable def rotated_ellipse_center (h' k' : ℝ) : Prop :=
h' = 3 ∧ k' = -5

noncomputable def rotated_ellipse_axes (a' b' : ℝ) : Prop :=
a' = 4 ∧ b' = 2

noncomputable def rotated_ellipse_sum (h' k' a' b' : ℝ) : ℝ :=
h' + k' + a' + b'

theorem rotated_ellipse_sum_is_four (h' k' a' b' : ℝ) 
  (hc : rotated_ellipse_center h' k') (ha : rotated_ellipse_axes a' b') :
  rotated_ellipse_sum h' k' a' b' = 4 :=
by
  -- The proof would be provided here.
  -- Since we're asked not to provide the proof but just to ensure the statement is correct, we use sorry.
  sorry

end NUMINAMATH_GPT_rotated_ellipse_sum_is_four_l1470_147059


namespace NUMINAMATH_GPT_pants_cost_l1470_147061

theorem pants_cost (starting_amount shirts_cost shirts_count amount_left money_after_shirts pants_cost : ℕ) 
    (h1 : starting_amount = 109)
    (h2 : shirts_cost = 11)
    (h3 : shirts_count = 2)
    (h4 : amount_left = 74)
    (h5 : money_after_shirts = starting_amount - shirts_cost * shirts_count)
    (h6 : pants_cost = money_after_shirts - amount_left) :
  pants_cost = 13 :=
by
  sorry

end NUMINAMATH_GPT_pants_cost_l1470_147061


namespace NUMINAMATH_GPT_complement_intersection_l1470_147039

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_complement_intersection_l1470_147039


namespace NUMINAMATH_GPT_sum_angles_acute_l1470_147071

open Real

theorem sum_angles_acute (A B C : ℝ) (hA_ac : A < π / 2) (hB_ac : B < π / 2) (hC_ac : C < π / 2)
  (h_angle_sum : sin A ^ 2 + sin B ^ 2 + sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end NUMINAMATH_GPT_sum_angles_acute_l1470_147071


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1470_147079

theorem factor_difference_of_squares (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := 
sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1470_147079


namespace NUMINAMATH_GPT_Jill_tax_on_clothing_l1470_147066

theorem Jill_tax_on_clothing 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ) (total_spent : ℝ) (tax_clothing : ℝ) 
  (tax_other_rate : ℝ) (total_tax_rate : ℝ) 
  (h_clothing : spent_clothing = 0.5 * total_spent) 
  (h_food : spent_food = 0.2 * total_spent) 
  (h_other : spent_other = 0.3 * total_spent) 
  (h_other_tax : tax_other_rate = 0.1) 
  (h_total_tax : total_tax_rate = 0.055) 
  (h_total_spent : total_spent = 100):
  (tax_clothing * spent_clothing + tax_other_rate * spent_other) = total_tax_rate * total_spent → 
  tax_clothing = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_Jill_tax_on_clothing_l1470_147066


namespace NUMINAMATH_GPT_conditional_probability_of_wind_given_rain_l1470_147089

theorem conditional_probability_of_wind_given_rain (P_A P_B P_A_and_B : ℚ)
  (h1: P_A = 4/15) (h2: P_B = 2/15) (h3: P_A_and_B = 1/10) :
  P_A_and_B / P_A = 3/8 :=
by
  sorry

end NUMINAMATH_GPT_conditional_probability_of_wind_given_rain_l1470_147089


namespace NUMINAMATH_GPT_area_outside_smaller_squares_l1470_147064

theorem area_outside_smaller_squares (side_large : ℕ) (side_small1 : ℕ) (side_small2 : ℕ)
  (no_overlap : Prop) (side_large_eq : side_large = 9)
  (side_small1_eq : side_small1 = 4)
  (side_small2_eq : side_small2 = 2) :
  (side_large * side_large - (side_small1 * side_small1 + side_small2 * side_small2)) = 61 :=
by
  sorry

end NUMINAMATH_GPT_area_outside_smaller_squares_l1470_147064


namespace NUMINAMATH_GPT_total_degree_difference_l1470_147084

-- Definitions based on conditions
def timeStart : ℕ := 12 * 60  -- noon in minutes
def timeEnd : ℕ := 14 * 60 + 30  -- 2:30 PM in minutes
def numTimeZones : ℕ := 3  -- Three time zones
def degreesInCircle : ℕ := 360  -- Degrees in a full circle

-- Calculate degrees moved by each hand
def degreesMovedByHourHand : ℚ := (timeEnd - timeStart) / (12 * 60) * degreesInCircle
def degreesMovedByMinuteHand : ℚ := (timeEnd - timeStart) % 60 * (degreesInCircle / 60)
def degreesMovedBySecondHand : ℕ := 0  -- At 2:30 PM, second hand is at initial position

-- Calculate total degree difference for all three hands and time zones
def totalDegrees : ℚ := 
  (degreesMovedByHourHand + degreesMovedByMinuteHand + degreesMovedBySecondHand) * numTimeZones

-- Theorem statement to prove
theorem total_degree_difference :
  totalDegrees = 765 := by
  sorry

end NUMINAMATH_GPT_total_degree_difference_l1470_147084


namespace NUMINAMATH_GPT_spontaneous_low_temperature_l1470_147015

theorem spontaneous_low_temperature (ΔH ΔS T : ℝ) (spontaneous : ΔG = ΔH - T * ΔS) :
  (∀ T, T > 0 → ΔG < 0 → ΔH < 0 ∧ ΔS < 0) := 
by 
  sorry

end NUMINAMATH_GPT_spontaneous_low_temperature_l1470_147015


namespace NUMINAMATH_GPT_expand_product_l1470_147048

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5 * x - 36 :=
by
  -- No proof required, just state the theorem
  sorry

end NUMINAMATH_GPT_expand_product_l1470_147048


namespace NUMINAMATH_GPT_tina_spent_on_books_l1470_147054

theorem tina_spent_on_books : 
  ∀ (saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left : ℤ),
  saved_in_june = 27 →
  saved_in_july = 14 →
  saved_in_august = 21 →
  spend_on_shoes = 17 →
  money_left = 40 →
  (saved_in_june + saved_in_july + saved_in_august) - spend_on_books - spend_on_shoes = money_left →
  spend_on_books = 5 :=
by
  intros saved_in_june saved_in_july saved_in_august spend_on_books spend_on_shoes money_left
  intros h_june h_july h_august h_shoes h_money_left h_eq
  sorry

end NUMINAMATH_GPT_tina_spent_on_books_l1470_147054


namespace NUMINAMATH_GPT_maximum_real_roots_maximum_total_real_roots_l1470_147074

variable (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

def quadratic_discriminant (p q r : ℝ) : ℝ := q^2 - 4 * p * r

theorem maximum_real_roots (h1 : quadratic_discriminant a b c < 0)
  (h2 : quadratic_discriminant b c a < 0)
  (h3 : quadratic_discriminant c a b < 0) :
  ∀ (x : ℝ), (a * x^2 + b * x + c ≠ 0) ∧ 
             (b * x^2 + c * x + a ≠ 0) ∧ 
             (c * x^2 + a * x + b ≠ 0) :=
sorry

theorem maximum_total_real_roots :
    ∃ x : ℝ, ∃ y : ℝ, ∃ z : ℝ,
    (a * x^2 + b * x + c = 0) ∧
    (b * y^2 + c * y + a = 0) ∧
    (a * y ≠ x) ∧
    (c * z^2 + a * z + b = 0) ∧
    (b * z ≠ x) ∧
    (c * z ≠ y) :=
sorry

end NUMINAMATH_GPT_maximum_real_roots_maximum_total_real_roots_l1470_147074


namespace NUMINAMATH_GPT_find_angle_x_eq_38_l1470_147030

theorem find_angle_x_eq_38
  (angle_ACD angle_ECB angle_DCE : ℝ)
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  (h3 : angle_ACD + angle_ECB + angle_DCE = 180) :
  angle_DCE = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_x_eq_38_l1470_147030


namespace NUMINAMATH_GPT_more_money_from_mom_is_correct_l1470_147002

noncomputable def more_money_from_mom : ℝ :=
  let money_from_mom := 8.25
  let money_from_dad := 6.50
  let money_from_grandparents := 12.35
  let money_from_aunt := 5.10
  let money_spent_toy := 4.45
  let money_spent_snacks := 6.25
  let total_received := money_from_mom + money_from_dad + money_from_grandparents + money_from_aunt
  let total_spent := money_spent_toy + money_spent_snacks
  let money_remaining := total_received - total_spent
  let money_spent_books := 0.25 * money_remaining
  let money_left_after_books := money_remaining - money_spent_books
  money_from_mom - money_from_dad

theorem more_money_from_mom_is_correct : more_money_from_mom = 1.75 := by
  sorry

end NUMINAMATH_GPT_more_money_from_mom_is_correct_l1470_147002


namespace NUMINAMATH_GPT_face_opposite_to_A_is_D_l1470_147005

-- Definitions of faces
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Given conditions
def C_is_on_top : Face := C
def B_is_to_the_right_of_C : Face := B
def forms_cube (f1 f2 : Face) : Prop := -- Some property indicating that the faces are part of a folded cube
sorry

-- The theorem statement to prove that the face opposite to face A is D
theorem face_opposite_to_A_is_D (h1 : C_is_on_top = C) (h2 : B_is_to_the_right_of_C = B) (h3 : forms_cube A D)
    : ∃ f : Face, f = D := sorry

end NUMINAMATH_GPT_face_opposite_to_A_is_D_l1470_147005


namespace NUMINAMATH_GPT_sushi_father_lollipops_l1470_147008

variable (x : ℕ)

theorem sushi_father_lollipops (h : x - 5 = 7) : x = 12 := by
  sorry

end NUMINAMATH_GPT_sushi_father_lollipops_l1470_147008


namespace NUMINAMATH_GPT_combination_15_choose_3_l1470_147003

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end NUMINAMATH_GPT_combination_15_choose_3_l1470_147003


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1470_147099

theorem quadratic_inequality_solution (m : ℝ) (h : m ≠ 0) : 
  (∃ x : ℝ, m * x^2 - x + 1 < 0) ↔ (m ∈ Set.Iio 0 ∨ m ∈ Set.Ioo 0 (1 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1470_147099


namespace NUMINAMATH_GPT_commercial_break_duration_l1470_147067

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end NUMINAMATH_GPT_commercial_break_duration_l1470_147067


namespace NUMINAMATH_GPT_probability_blue_or_purple_is_4_over_11_l1470_147098

def total_jelly_beans : ℕ := 10 + 12 + 13 + 15 + 5
def blue_or_purple_jelly_beans : ℕ := 15 + 5
def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_4_over_11 :
  probability_blue_or_purple = 4 / 11 :=
sorry

end NUMINAMATH_GPT_probability_blue_or_purple_is_4_over_11_l1470_147098


namespace NUMINAMATH_GPT_mary_earns_per_home_l1470_147014

noncomputable def earnings_per_home (T : ℕ) (n : ℕ) : ℕ := T / n

theorem mary_earns_per_home :
  ∀ (T n : ℕ), T = 276 → n = 6 → earnings_per_home T n = 46 := 
by
  intros T n h1 h2
  -- Placeholder proof step
  sorry

end NUMINAMATH_GPT_mary_earns_per_home_l1470_147014


namespace NUMINAMATH_GPT_problem_statement_l1470_147072

theorem problem_statement (c d : ℤ) (hc : c = 3) (hd : d = 2) :
  (c^2 + d)^2 - (c^2 - d)^2 = 72 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1470_147072


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1470_147043

def expr (a b : ℤ) := -a^2 * b + (3 * a * b^2 - a^2 * b) - 2 * (2 * a * b^2 - a^2 * b)

theorem simplify_and_evaluate : expr (-1) (-2) = -4 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1470_147043


namespace NUMINAMATH_GPT_cloth_length_l1470_147060

theorem cloth_length (L : ℕ) (x : ℕ) :
  32 + x = L ∧ 20 + 3 * x = L → L = 38 :=
by
  sorry

end NUMINAMATH_GPT_cloth_length_l1470_147060


namespace NUMINAMATH_GPT_not_possible_to_cover_l1470_147092

namespace CubeCovering

-- Defining the cube and its properties
def cube_side_length : ℕ := 4
def face_area := cube_side_length * cube_side_length
def total_faces : ℕ := 6
def faces_to_cover : ℕ := 3

-- Defining the paper strips and their properties
def strip_length : ℕ := 3
def strip_width : ℕ := 1
def strip_area := strip_length * strip_width
def num_strips : ℕ := 16

-- Calculate the total area to cover
def total_area_to_cover := faces_to_cover * face_area
def total_area_strips := num_strips * strip_area

-- Statement: Prove that it is not possible to cover the three faces
theorem not_possible_to_cover : total_area_to_cover = 48 → total_area_strips = 48 → false := by
  intro h1 h2
  sorry

end CubeCovering

end NUMINAMATH_GPT_not_possible_to_cover_l1470_147092


namespace NUMINAMATH_GPT_slope_of_line_6x_minus_4y_eq_16_l1470_147025

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_of_line_6x_minus_4y_eq_16 :
  slope_of_line 6 (-4) (-16) = 3 / 2 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_slope_of_line_6x_minus_4y_eq_16_l1470_147025


namespace NUMINAMATH_GPT_magic_square_sum_l1470_147016

-- Definitions based on the conditions outlined in the problem
def magic_sum := 83
def a := 42
def b := 26
def c := 29
def e := 34
def d := 36

theorem magic_square_sum :
  d + e = 70 :=
by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_magic_square_sum_l1470_147016


namespace NUMINAMATH_GPT_weekly_milk_production_l1470_147082

-- Define the conditions
def num_cows : ℕ := 52
def milk_per_cow_per_day : ℕ := 5
def days_per_week : ℕ := 7

-- Define the proof that total weekly milk production is 1820 liters
theorem weekly_milk_production : num_cows * milk_per_cow_per_day * days_per_week = 1820 := by
  sorry

end NUMINAMATH_GPT_weekly_milk_production_l1470_147082


namespace NUMINAMATH_GPT_greatest_xy_value_l1470_147077

theorem greatest_xy_value (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 7 * x + 4 * y = 140) :
  (∀ z : ℕ, (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ z = x * y) → z ≤ 168) ∧
  (∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 7 * x + 4 * y = 140 ∧ 168 = x * y) :=
sorry

end NUMINAMATH_GPT_greatest_xy_value_l1470_147077


namespace NUMINAMATH_GPT_vector_sum_correct_l1470_147056

-- Define the three vectors
def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 6)
def v3 : ℝ × ℝ := (2, -8)

-- Define the expected result
def expected_sum : ℝ × ℝ := (3, -5)

-- Define vector addition (component-wise)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- The theorem statement
theorem vector_sum_correct : vector_add (vector_add v1 v2) v3 = expected_sum := by
  sorry

end NUMINAMATH_GPT_vector_sum_correct_l1470_147056


namespace NUMINAMATH_GPT_total_amount_is_correct_l1470_147065

-- Given conditions
def original_price : ℝ := 200
def discount_rate: ℝ := 0.25
def coupon_value: ℝ := 10
def tax_rate: ℝ := 0.05

-- Define the price calculations
def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
def price_after_coupon (p : ℝ) (c : ℝ) : ℝ := p - c
def final_price_with_tax (p : ℝ) (t : ℝ) : ℝ := p * (1 + t)

-- Goal: Prove the final amount the customer pays
theorem total_amount_is_correct : final_price_with_tax (price_after_coupon (discounted_price original_price discount_rate) coupon_value) tax_rate = 147 := by
  sorry

end NUMINAMATH_GPT_total_amount_is_correct_l1470_147065


namespace NUMINAMATH_GPT_ali_seashells_final_count_l1470_147009

theorem ali_seashells_final_count :
  385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25))) 
  - (1 / 4) * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)))) = 82.485 :=
sorry

end NUMINAMATH_GPT_ali_seashells_final_count_l1470_147009


namespace NUMINAMATH_GPT_average_age_calculated_years_ago_l1470_147029

theorem average_age_calculated_years_ago
  (n m : ℕ) (a b : ℕ) 
  (total_age_original : ℝ)
  (average_age_original : ℝ)
  (average_age_new : ℝ) :
  n = 6 → 
  a = 19 → 
  m = 7 → 
  b = 1 → 
  total_age_original = n * a → 
  average_age_original = a → 
  average_age_new = a →
  (total_age_original + b) / m = a → 
  1 = 1 := 
by
  intros _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_average_age_calculated_years_ago_l1470_147029


namespace NUMINAMATH_GPT_abs_eq_solution_l1470_147051

theorem abs_eq_solution (x : ℚ) : |x - 2| = |x + 3| → x = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_solution_l1470_147051


namespace NUMINAMATH_GPT_a_2n_is_perfect_square_l1470_147078

-- Define the sequence a_n as per the problem's conditions
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n - 1) + a (n - 3) + a (n - 4)

-- Define the Fibonacci sequence for comparison
def fib (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

-- Key theorem to prove: a_{2n} is a perfect square
theorem a_2n_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, a (2 * n) = k * k :=
sorry

end NUMINAMATH_GPT_a_2n_is_perfect_square_l1470_147078


namespace NUMINAMATH_GPT_a_minus_b_range_l1470_147049

noncomputable def range_of_a_minus_b (a b : ℝ) : Set ℝ :=
  {x | -2 < a ∧ a < 1 ∧ 0 < b ∧ b < 4 ∧ x = a - b}

theorem a_minus_b_range (a b : ℝ) (h₁ : -2 < a) (h₂ : a < 1) (h₃ : 0 < b) (h₄ : b < 4) :
  ∃ x, range_of_a_minus_b a b x ∧ (-6 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_range_l1470_147049


namespace NUMINAMATH_GPT_puzzles_pieces_count_l1470_147086

theorem puzzles_pieces_count :
  let pieces_per_hour := 100
  let hours_per_day := 7
  let days := 7
  let total_pieces_can_put_together := pieces_per_hour * hours_per_day * days
  let pieces_per_puzzle1 := 300
  let number_of_puzzles1 := 8
  let total_pieces_puzzles1 := pieces_per_puzzle1 * number_of_puzzles1
  let remaining_pieces := total_pieces_can_put_together - total_pieces_puzzles1
  let number_of_puzzles2 := 5
  remaining_pieces / number_of_puzzles2 = 500
:= by
  sorry

end NUMINAMATH_GPT_puzzles_pieces_count_l1470_147086


namespace NUMINAMATH_GPT_minimum_xy_minimum_x_plus_y_l1470_147070

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
sorry

theorem minimum_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end NUMINAMATH_GPT_minimum_xy_minimum_x_plus_y_l1470_147070


namespace NUMINAMATH_GPT_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1470_147046

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end NUMINAMATH_GPT_coefficient_x3y5_in_expansion_of_x_plus_y_8_l1470_147046


namespace NUMINAMATH_GPT_negation_proposition_l1470_147058

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :
  ¬(∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) = ∃ x : ℝ, x^2 - 2*x + 4 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1470_147058


namespace NUMINAMATH_GPT_walking_rate_ratio_l1470_147053

theorem walking_rate_ratio (R R' : ℝ)
  (h : R * 36 = R' * 32) : R' / R = 9 / 8 :=
sorry

end NUMINAMATH_GPT_walking_rate_ratio_l1470_147053


namespace NUMINAMATH_GPT_person_B_D_coins_l1470_147091

theorem person_B_D_coins
  (a d : ℤ)
  (h1 : a - 3 * d = 58)
  (h2 : a - 2 * d = 58)
  (h3 : a + d = 60)
  (h4 : a + 2 * d = 60)
  (h5 : a + 3 * d = 60) :
  (a - 2 * d = 28) ∧ (a = 24) :=
by
  sorry

end NUMINAMATH_GPT_person_B_D_coins_l1470_147091


namespace NUMINAMATH_GPT_max_plates_l1470_147044

/-- Bill can buy pans, pots, and plates for 3, 5, and 10 dollars each, respectively.
    What is the maximum number of plates he can purchase if he must buy at least
    two of each item and will spend exactly 100 dollars? -/
theorem max_plates (x y z : ℕ) (hx : x ≥ 2) (hy : y ≥ 2) (hz : z ≥ 2) 
  (h_cost : 3 * x + 5 * y + 10 * z = 100) : z = 8 :=
sorry

end NUMINAMATH_GPT_max_plates_l1470_147044


namespace NUMINAMATH_GPT_new_ratio_milk_water_after_adding_milk_l1470_147073

variable (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ)
variable (added_milk_volume : ℕ)

def ratio_of_mix_after_addition (initial_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) 
  (added_milk_volume : ℕ) : ℕ × ℕ :=
  let total_parts := initial_milk_ratio + initial_water_ratio
  let part_volume := initial_volume / total_parts
  let initial_milk_volume := initial_milk_ratio * part_volume
  let initial_water_volume := initial_water_ratio * part_volume
  let new_milk_volume := initial_milk_volume + added_milk_volume
  (new_milk_volume / initial_water_volume, 1)

theorem new_ratio_milk_water_after_adding_milk 
  (h_initial_volume : initial_volume = 20)
  (h_initial_milk_ratio : initial_milk_ratio = 3)
  (h_initial_water_ratio : initial_water_ratio = 1)
  (h_added_milk_volume : added_milk_volume = 5) : 
  ratio_of_mix_after_addition initial_volume initial_milk_ratio initial_water_ratio added_milk_volume = (4, 1) :=
  by
    sorry

end NUMINAMATH_GPT_new_ratio_milk_water_after_adding_milk_l1470_147073


namespace NUMINAMATH_GPT_union_A_B_intersection_A_CI_B_l1470_147021

-- Define the sets
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 5, 6, 7}

-- Define the complement of B in the universal set I
def C_I (I : Set ℕ) (B : Set ℕ) : Set ℕ := {x ∈ I | x ∉ B}

-- The theorem for the union of A and B
theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6, 7} := sorry

-- The theorem for the intersection of A and the complement of B in I
theorem intersection_A_CI_B : A ∩ (C_I I B) = {1, 2, 4} := sorry

end NUMINAMATH_GPT_union_A_B_intersection_A_CI_B_l1470_147021


namespace NUMINAMATH_GPT_factorial_product_trailing_zeros_l1470_147013

def countTrailingZerosInFactorialProduct : ℕ :=
  let countFactorsOfFive (n : ℕ) : ℕ := 
    (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) + (n / 78125) + (n / 390625) 
  List.range 100 -- Generates list [0, 1, ..., 99]
  |> List.map (fun k => countFactorsOfFive (k + 1)) -- Apply countFactorsOfFive to each k+1
  |> List.foldr (· + ·) 0 -- Sum all counts

theorem factorial_product_trailing_zeros : countTrailingZerosInFactorialProduct = 1124 := by
  sorry

end NUMINAMATH_GPT_factorial_product_trailing_zeros_l1470_147013


namespace NUMINAMATH_GPT_find_missing_number_l1470_147045

theorem find_missing_number (x : ℚ) (h : 11 * x + 4 = 7) : x = 9 / 11 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l1470_147045


namespace NUMINAMATH_GPT_number_of_pairs_of_socks_l1470_147037

theorem number_of_pairs_of_socks (n : ℕ) (h : 2 * n^2 - n = 112) : n = 16 := sorry

end NUMINAMATH_GPT_number_of_pairs_of_socks_l1470_147037


namespace NUMINAMATH_GPT_compare_y_values_l1470_147026

theorem compare_y_values (y1 y2 : ℝ) 
  (hA : y1 = (-1)^2 - 4*(-1) - 3) 
  (hB : y2 = 1^2 - 4*1 - 3) : y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_compare_y_values_l1470_147026


namespace NUMINAMATH_GPT_range_of_t_l1470_147010

theorem range_of_t (a t : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  a * x^2 + t * y^2 ≥ (a * x + t * y)^2 ↔ 0 ≤ t ∧ t ≤ 1 - a :=
sorry

end NUMINAMATH_GPT_range_of_t_l1470_147010


namespace NUMINAMATH_GPT_remainder_of_x7_plus_2_div_x_plus_1_l1470_147023

def f (x : ℤ) := x^7 + 2

theorem remainder_of_x7_plus_2_div_x_plus_1 : 
  (f (-1) = 1) := sorry

end NUMINAMATH_GPT_remainder_of_x7_plus_2_div_x_plus_1_l1470_147023


namespace NUMINAMATH_GPT_parallel_planes_mn_l1470_147050

theorem parallel_planes_mn (m n : ℝ) (a b : ℝ × ℝ × ℝ) (α β : Type) (h1 : a = (0, 1, m)) (h2 : b = (0, n, -3)) 
  (h3 : ∃ k : ℝ, a = (k • b)) : m * n = -3 :=
by
  -- Proof would be here
  sorry

end NUMINAMATH_GPT_parallel_planes_mn_l1470_147050


namespace NUMINAMATH_GPT_hours_of_use_per_charge_l1470_147055

theorem hours_of_use_per_charge
  (c h u : ℕ)
  (h_c : c = 10)
  (h_fraction : h = 6)
  (h_use : 6 * u = 12) :
  u = 2 :=
sorry

end NUMINAMATH_GPT_hours_of_use_per_charge_l1470_147055


namespace NUMINAMATH_GPT_number_of_valid_n_l1470_147020

theorem number_of_valid_n : 
  (∃ (n : ℕ), ∀ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 → n = a + 2 * b + 3 * c) ↔
  (∃ (n : ℕ), n = 1000) := by 
  sorry

end NUMINAMATH_GPT_number_of_valid_n_l1470_147020


namespace NUMINAMATH_GPT_simplify_polynomial_l1470_147076

def P (x : ℝ) : ℝ := 3*x^3 + 4*x^2 - 5*x + 8
def Q (x : ℝ) : ℝ := 2*x^3 + x^2 + 3*x - 15

theorem simplify_polynomial (x : ℝ) : P x - Q x = x^3 + 3*x^2 - 8*x + 23 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1470_147076


namespace NUMINAMATH_GPT_solve_quadratic_l1470_147022

theorem solve_quadratic : 
  (∀ x : ℚ, 2 * x^2 - x - 6 = 0 → x = -3 / 2 ∨ x = 2) ∧ 
  (∀ y : ℚ, (y - 2)^2 = 9 * y^2 → y = -1 ∨ y = 1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1470_147022


namespace NUMINAMATH_GPT_op_exp_eq_l1470_147012

-- Define the operation * on natural numbers
def op (a b : ℕ) : ℕ := a ^ b

-- The theorem to be proven
theorem op_exp_eq (a b n : ℕ) : (op a b)^n = op a (b^n) := by
  sorry

end NUMINAMATH_GPT_op_exp_eq_l1470_147012


namespace NUMINAMATH_GPT_value_of_f_m_plus_one_is_negative_l1470_147075

-- Definitions for function and condition
def f (x a : ℝ) := x^2 - x + a 

-- Problem statement: Given that 'f(-m) < 0', prove 'f(m+1) < 0'
theorem value_of_f_m_plus_one_is_negative (a m : ℝ) (h : f (-m) a < 0) : f (m + 1) a < 0 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_f_m_plus_one_is_negative_l1470_147075


namespace NUMINAMATH_GPT_combined_time_third_attempt_l1470_147087

noncomputable def first_lock_initial : ℕ := 5
noncomputable def second_lock_initial : ℕ := 3 * first_lock_initial - 3
noncomputable def combined_initial : ℕ := 5 * second_lock_initial

noncomputable def first_lock_second_attempt : ℝ := first_lock_initial - 0.1 * first_lock_initial
noncomputable def first_lock_third_attempt : ℝ := first_lock_second_attempt - 0.1 * first_lock_second_attempt

noncomputable def second_lock_second_attempt : ℝ := second_lock_initial - 0.15 * second_lock_initial
noncomputable def second_lock_third_attempt : ℝ := second_lock_second_attempt - 0.15 * second_lock_second_attempt

noncomputable def combined_third_attempt : ℝ := 5 * second_lock_third_attempt

theorem combined_time_third_attempt : combined_third_attempt = 43.35 :=
by
  sorry

end NUMINAMATH_GPT_combined_time_third_attempt_l1470_147087


namespace NUMINAMATH_GPT_cars_meet_time_l1470_147093

theorem cars_meet_time 
  (L : ℕ) (v1 v2 : ℕ) (t : ℕ)
  (H1 : L = 333)
  (H2 : v1 = 54)
  (H3 : v2 = 57)
  (H4 : v1 * t + v2 * t = L) : 
  t = 3 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_cars_meet_time_l1470_147093


namespace NUMINAMATH_GPT_probability_fail_then_succeed_l1470_147018

theorem probability_fail_then_succeed
  (P_fail_first : ℚ := 9 / 10)
  (P_succeed_second : ℚ := 1 / 9) :
  P_fail_first * P_succeed_second = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_fail_then_succeed_l1470_147018


namespace NUMINAMATH_GPT_sum_of_coefficients_l1470_147028

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1470_147028


namespace NUMINAMATH_GPT_max_discount_rate_l1470_147006

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end NUMINAMATH_GPT_max_discount_rate_l1470_147006


namespace NUMINAMATH_GPT_gina_total_pay_l1470_147019

noncomputable def gina_painting_pay : ℕ :=
let roses_per_hour := 6
let lilies_per_hour := 7
let rose_order := 6
let lily_order := 14
let pay_per_hour := 30

-- Calculate total time (in hours) Gina spends to complete the order
let time_for_roses := rose_order / roses_per_hour
let time_for_lilies := lily_order / lilies_per_hour
let total_time := time_for_roses + time_for_lilies

-- Calculate the total pay
let total_pay := total_time * pay_per_hour

total_pay

-- The theorem that Gina gets paid $90 for the order
theorem gina_total_pay : gina_painting_pay = 90 := by
  sorry

end NUMINAMATH_GPT_gina_total_pay_l1470_147019


namespace NUMINAMATH_GPT_seating_arrangements_count_l1470_147083

-- Define the main entities: the three teams and the conditions
inductive Person
| Jupitarian
| Saturnian
| Neptunian

open Person

-- Define the seating problem constraints
def valid_arrangement (seating : Fin 12 → Person) : Prop :=
  seating 0 = Jupitarian ∧ seating 11 = Neptunian ∧
  (∀ i, seating (i % 12) = Jupitarian → seating ((i + 11) % 12) ≠ Neptunian) ∧
  (∀ i, seating (i % 12) = Neptunian → seating ((i + 11) % 12) ≠ Saturnian) ∧
  (∀ i, seating (i % 12) = Saturnian → seating ((i + 11) % 12) ≠ Jupitarian)

-- Main theorem: The number of valid arrangements is 225 * (4!)^3
theorem seating_arrangements_count :
  ∃ M : ℕ, (M = 225) ∧ ∃ arrangements : Fin 12 → Person, valid_arrangement arrangements :=
sorry

end NUMINAMATH_GPT_seating_arrangements_count_l1470_147083


namespace NUMINAMATH_GPT_total_annual_car_maintenance_expenses_is_330_l1470_147034

-- Define the conditions as constants
def annualMileage : ℕ := 12000
def milesPerOilChange : ℕ := 3000
def freeOilChangesPerYear : ℕ := 1
def costPerOilChange : ℕ := 50
def milesPerTireRotation : ℕ := 6000
def costPerTireRotation : ℕ := 40
def milesPerBrakePadReplacement : ℕ := 24000
def costPerBrakePadReplacement : ℕ := 200

-- Define the total annual car maintenance expenses calculation
def annualOilChangeExpenses (annualMileage : ℕ) (milesPerOilChange : ℕ) (freeOilChangesPerYear : ℕ) (costPerOilChange : ℕ) : ℕ :=
  let oilChangesNeeded := annualMileage / milesPerOilChange
  let paidOilChanges := oilChangesNeeded - freeOilChangesPerYear
  paidOilChanges * costPerOilChange

def annualTireRotationExpenses (annualMileage : ℕ) (milesPerTireRotation : ℕ) (costPerTireRotation : ℕ) : ℕ :=
  let tireRotationsNeeded := annualMileage / milesPerTireRotation
  tireRotationsNeeded * costPerTireRotation

def annualBrakePadReplacementExpenses (annualMileage : ℕ) (milesPerBrakePadReplacement : ℕ) (costPerBrakePadReplacement : ℕ) : ℕ :=
  let brakePadReplacementInterval := milesPerBrakePadReplacement / annualMileage
  costPerBrakePadReplacement / brakePadReplacementInterval

def totalAnnualCarMaintenanceExpenses : ℕ :=
  annualOilChangeExpenses annualMileage milesPerOilChange freeOilChangesPerYear costPerOilChange +
  annualTireRotationExpenses annualMileage milesPerTireRotation costPerTireRotation +
  annualBrakePadReplacementExpenses annualMileage milesPerBrakePadReplacement costPerBrakePadReplacement

-- Prove the total annual car maintenance expenses equals $330
theorem total_annual_car_maintenance_expenses_is_330 : totalAnnualCarMaintenanceExpenses = 330 := by
  sorry

end NUMINAMATH_GPT_total_annual_car_maintenance_expenses_is_330_l1470_147034


namespace NUMINAMATH_GPT_sales_tax_percentage_l1470_147032

noncomputable def original_price : ℝ := 200
noncomputable def discount : ℝ := 0.25 * original_price
noncomputable def sale_price : ℝ := original_price - discount
noncomputable def total_paid : ℝ := 165
noncomputable def sales_tax : ℝ := total_paid - sale_price

theorem sales_tax_percentage : (sales_tax / sale_price) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_sales_tax_percentage_l1470_147032


namespace NUMINAMATH_GPT_total_weight_l1470_147001

def weight_of_blue_ball : ℝ := 6.0
def weight_of_brown_ball : ℝ := 3.12

theorem total_weight (_ : weight_of_blue_ball = 6.0) (_ : weight_of_brown_ball = 3.12) : 
  weight_of_blue_ball + weight_of_brown_ball = 9.12 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_l1470_147001


namespace NUMINAMATH_GPT_lcm_12_15_18_l1470_147068

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_12_15_18_l1470_147068


namespace NUMINAMATH_GPT_xiaoming_grandfather_age_l1470_147057

-- Define the conditions
def age_cond (x : ℕ) : Prop :=
  ((x - 15) / 4 - 6) * 10 = 100

-- State the problem
theorem xiaoming_grandfather_age (x : ℕ) (h : age_cond x) : x = 79 := 
sorry

end NUMINAMATH_GPT_xiaoming_grandfather_age_l1470_147057


namespace NUMINAMATH_GPT_excluded_angle_sum_1680_degrees_l1470_147027

theorem excluded_angle_sum_1680_degrees (sum_except_one : ℝ) (h : sum_except_one = 1680) : 
  (180 - (1680 % 180)) = 120 :=
by
  have mod_eq : 1680 % 180 = 60 := by sorry
  rw [mod_eq]

end NUMINAMATH_GPT_excluded_angle_sum_1680_degrees_l1470_147027


namespace NUMINAMATH_GPT_minimize_y_l1470_147062

noncomputable def y (x a b : ℝ) : ℝ := 2 * (x - a)^2 + 3 * (x - b)^2

theorem minimize_y (a b : ℝ) : ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) ∧ x = (2 * a + 3 * b) / 5 :=
sorry

end NUMINAMATH_GPT_minimize_y_l1470_147062


namespace NUMINAMATH_GPT_laborer_monthly_income_l1470_147063

variable (I : ℝ)

noncomputable def average_expenditure_six_months := 70 * 6
noncomputable def debt_condition := I * 6 < average_expenditure_six_months
noncomputable def expenditure_next_four_months := 60 * 4
noncomputable def total_income_next_four_months := expenditure_next_four_months + (average_expenditure_six_months - I * 6) + 30

theorem laborer_monthly_income (h1 : debt_condition I) (h2 : total_income_next_four_months I = I * 4) :
  I = 69 :=
by
  sorry

end NUMINAMATH_GPT_laborer_monthly_income_l1470_147063


namespace NUMINAMATH_GPT_field_ratio_l1470_147004

theorem field_ratio (side pond_area_ratio : ℝ) (field_length : ℝ) 
  (pond_is_square: pond_area_ratio = 1/18) 
  (side_length: side = 8) 
  (field_len: field_length = 48) : 
  (field_length / (pond_area_ratio * side ^ 2 / side)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_field_ratio_l1470_147004


namespace NUMINAMATH_GPT_books_price_arrangement_l1470_147033

theorem books_price_arrangement (c : ℝ) (prices : Fin 40 → ℝ)
  (h₁ : ∀ i : Fin 39, prices i.succ = prices i + 3)
  (h₂ : prices ⟨39, by norm_num⟩ = prices ⟨19, by norm_num⟩ + prices ⟨20, by norm_num⟩) :
  prices 20 = prices 19 + 3 := 
sorry

end NUMINAMATH_GPT_books_price_arrangement_l1470_147033


namespace NUMINAMATH_GPT_pond_width_l1470_147090

theorem pond_width
  (L : ℝ) (D : ℝ) (V : ℝ) (W : ℝ)
  (hL : L = 20)
  (hD : D = 5)
  (hV : V = 1000)
  (hVolume : V = L * W * D) :
  W = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_pond_width_l1470_147090


namespace NUMINAMATH_GPT_curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l1470_147042

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2

theorem curve_C1_general_equation (x y : ℝ) (α : ℝ) :
  (2 * Real.cos α = x) ∧ (Real.sqrt 2 * Real.sin α = y) →
  x^2 / 4 + y^2 / 2 = 1 :=
sorry

theorem curve_C2_cartesian_equation (ρ θ : ℝ) (x y : ℝ) :
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ polar_curve_C2 ρ θ →
  x + y = 6 :=
sorry

theorem minimum_distance_P1P2 (P1 P2 : ℝ × ℝ) (d : ℝ) :
  (∃ α, P1 = parametric_curve_C1 α) ∧ (∃ x y, P2 = (x, y) ∧ x + y = 6) →
  d = (3 * Real.sqrt 2 - Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l1470_147042


namespace NUMINAMATH_GPT_fewer_green_pens_than_pink_l1470_147000

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end NUMINAMATH_GPT_fewer_green_pens_than_pink_l1470_147000


namespace NUMINAMATH_GPT_mushrooms_picked_on_second_day_l1470_147031

theorem mushrooms_picked_on_second_day :
  ∃ (n2 : ℕ), (∃ (n1 n3 : ℕ), n3 = 2 * n2 ∧ n1 + n2 + n3 = 65) ∧ n2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_mushrooms_picked_on_second_day_l1470_147031


namespace NUMINAMATH_GPT_no_such_nat_n_l1470_147096

theorem no_such_nat_n :
  ¬ ∃ n : ℕ, ∀ a b : ℕ, (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → (10 * (10 * a + n) + b) % (10 * a + b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_such_nat_n_l1470_147096
