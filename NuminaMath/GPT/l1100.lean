import Mathlib

namespace NUMINAMATH_GPT_algebraic_expression_value_l1100_110080

theorem algebraic_expression_value 
  (p q r s : ℝ) 
  (hpq3 : p^2 / q^3 = 4 / 5) 
  (hrs2 : r^3 / s^2 = 7 / 9) : 
  11 / (7 - r^3 / s^2) + (2 * q^3 - p^2) / (2 * q^3 + p^2) = 123 / 56 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1100_110080


namespace NUMINAMATH_GPT_anand_income_l1100_110014

theorem anand_income (x y : ℕ)
  (income_A : ℕ := 5 * x)
  (income_B : ℕ := 4 * x)
  (expenditure_A : ℕ := 3 * y)
  (expenditure_B : ℕ := 2 * y)
  (savings_A : ℕ := 800)
  (savings_B : ℕ := 800)
  (hA : income_A - expenditure_A = savings_A)
  (hB : income_B - expenditure_B = savings_B) :
  income_A = 2000 := by
  sorry

end NUMINAMATH_GPT_anand_income_l1100_110014


namespace NUMINAMATH_GPT_correct_choice_l1100_110048

theorem correct_choice : 2 ∈ ({0, 1, 2} : Set ℕ) :=
sorry

end NUMINAMATH_GPT_correct_choice_l1100_110048


namespace NUMINAMATH_GPT_tom_remaining_balloons_l1100_110008

def original_balloons : ℕ := 30
def given_balloons : ℕ := 16
def remaining_balloons (original_balloons given_balloons : ℕ) : ℕ := original_balloons - given_balloons

theorem tom_remaining_balloons : remaining_balloons original_balloons given_balloons = 14 :=
by
  -- proof omitted for clarity
  sorry

end NUMINAMATH_GPT_tom_remaining_balloons_l1100_110008


namespace NUMINAMATH_GPT_sin_C_of_arithmetic_sequence_l1100_110075

theorem sin_C_of_arithmetic_sequence 
  (A B C : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = Real.pi) 
  (h3 : Real.cos A = 2 / 3) 
  : Real.sin C = (Real.sqrt 5 + 2 * Real.sqrt 3) / 6 :=
sorry

end NUMINAMATH_GPT_sin_C_of_arithmetic_sequence_l1100_110075


namespace NUMINAMATH_GPT_days_to_finish_by_b_l1100_110038

theorem days_to_finish_by_b (A B C : ℚ) 
  (h1 : A + B + C = 1 / 5) 
  (h2 : A = 1 / 9) 
  (h3 : A + C = 1 / 7) : 
  1 / B = 12.115 :=
by
  sorry

end NUMINAMATH_GPT_days_to_finish_by_b_l1100_110038


namespace NUMINAMATH_GPT_distance_between_points_l1100_110010

theorem distance_between_points : ∀ (A B : ℤ), A = 5 → B = -3 → |A - B| = 8 :=
by
  intros A B hA hB
  rw [hA, hB]
  norm_num

end NUMINAMATH_GPT_distance_between_points_l1100_110010


namespace NUMINAMATH_GPT_astroid_arc_length_l1100_110033

theorem astroid_arc_length (a : ℝ) (h_a : a > 0) :
  ∃ l : ℝ, (l = 6 * a) ∧ 
  ((a = 1 → l = 6) ∧ (a = 2/3 → l = 4)) := 
by
  sorry

end NUMINAMATH_GPT_astroid_arc_length_l1100_110033


namespace NUMINAMATH_GPT_discount_is_20_percent_l1100_110074

noncomputable def discount_percentage 
  (puppy_cost : ℝ := 20.0)
  (dog_food_cost : ℝ := 20.0)
  (treat_cost : ℝ := 2.5)
  (num_treats : ℕ := 2)
  (toy_cost : ℝ := 15.0)
  (crate_cost : ℝ := 20.0)
  (bed_cost : ℝ := 20.0)
  (collar_leash_cost : ℝ := 15.0)
  (total_spent : ℝ := 96.0) : ℝ := 
  let total_cost_before_discount := dog_food_cost + (num_treats * treat_cost) + toy_cost + crate_cost + bed_cost + collar_leash_cost
  let spend_at_store := total_spent - puppy_cost
  let discount_amount := total_cost_before_discount - spend_at_store
  (discount_amount / total_cost_before_discount) * 100

theorem discount_is_20_percent : discount_percentage = 20 := sorry

end NUMINAMATH_GPT_discount_is_20_percent_l1100_110074


namespace NUMINAMATH_GPT_sum_distinct_vars_eq_1716_l1100_110029

open Real

theorem sum_distinct_vars_eq_1716 (p q r s : ℝ) (hpqrs_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s)
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s) :
  p + q + r + s = 1716 :=
sorry

end NUMINAMATH_GPT_sum_distinct_vars_eq_1716_l1100_110029


namespace NUMINAMATH_GPT_integer_values_count_l1100_110012

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end NUMINAMATH_GPT_integer_values_count_l1100_110012


namespace NUMINAMATH_GPT_price_per_strawberry_basket_is_9_l1100_110044

-- Define the conditions
def strawberry_plants := 5
def tomato_plants := 7
def strawberries_per_plant := 14
def tomatoes_per_plant := 16
def items_per_basket := 7
def price_per_tomato_basket := 6
def total_revenue := 186

-- Define the total number of strawberries and tomatoes harvested
def total_strawberries := strawberry_plants * strawberries_per_plant
def total_tomatoes := tomato_plants * tomatoes_per_plant

-- Define the number of baskets of strawberries and tomatoes
def strawberry_baskets := total_strawberries / items_per_basket
def tomato_baskets := total_tomatoes / items_per_basket

-- Define the revenue from tomato baskets
def revenue_tomatoes := tomato_baskets * price_per_tomato_basket

-- Define the revenue from strawberry baskets
def revenue_strawberries := total_revenue - revenue_tomatoes

-- Calculate the price per basket of strawberries (which should be $9)
def price_per_strawberry_basket := revenue_strawberries / strawberry_baskets

theorem price_per_strawberry_basket_is_9 : 
  price_per_strawberry_basket = 9 := by
    sorry

end NUMINAMATH_GPT_price_per_strawberry_basket_is_9_l1100_110044


namespace NUMINAMATH_GPT_racetrack_circumference_diff_l1100_110094

theorem racetrack_circumference_diff (d_inner d_outer width : ℝ) 
(h1 : d_inner = 55) (h2 : width = 15) (h3 : d_outer = d_inner + 2 * width) : 
  (π * d_outer - π * d_inner) = 30 * π :=
by
  sorry

end NUMINAMATH_GPT_racetrack_circumference_diff_l1100_110094


namespace NUMINAMATH_GPT_below_zero_notation_l1100_110009

def celsius_above (x : ℤ) : String := "+" ++ toString x ++ "°C"
def celsius_below (x : ℤ) : String := "-" ++ toString x ++ "°C"

theorem below_zero_notation (h₁ : celsius_above 5 = "+5°C")
  (h₂ : ∀ x : ℤ, x > 0 → celsius_above x = "+" ++ toString x ++ "°C")
  (h₃ : ∀ x : ℤ, x > 0 → celsius_below x = "-" ++ toString x ++ "°C") :
  celsius_below 3 = "-3°C" :=
sorry

end NUMINAMATH_GPT_below_zero_notation_l1100_110009


namespace NUMINAMATH_GPT_code_length_is_4_l1100_110070

-- Definitions based on conditions provided
def code_length : ℕ := 4 -- Each code consists of 4 digits
def total_codes_with_leading_zeros : ℕ := 10^code_length -- Total possible codes allowing leading zeros
def total_codes_without_leading_zeros : ℕ := 9 * 10^(code_length - 1) -- Total possible codes disallowing leading zeros
def codes_lost_if_no_leading_zeros : ℕ := total_codes_with_leading_zeros - total_codes_without_leading_zeros -- Codes lost if leading zeros are disallowed
def manager_measured_codes_lost : ℕ := 10000 -- Manager's incorrect measurement

-- Theorem to be proved based on the problem
theorem code_length_is_4 : code_length = 4 :=
by
  sorry

end NUMINAMATH_GPT_code_length_is_4_l1100_110070


namespace NUMINAMATH_GPT_negation_proof_l1100_110095

theorem negation_proof : 
  (¬(∀ x : ℝ, x < 2^x) ↔ ∃ x : ℝ, x ≥ 2^x) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l1100_110095


namespace NUMINAMATH_GPT_value_of_y_l1100_110090

theorem value_of_y :
  ∃ y : ℝ, (3 * y) / 7 = 12 ∧ y = 28 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l1100_110090


namespace NUMINAMATH_GPT_value_of_f_nine_halves_l1100_110083

noncomputable def f : ℝ → ℝ := sorry  -- Define f with noncomputable since it's not explicitly given

axiom even_function (x : ℝ) : f x = f (-x)  -- Define the even function property
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0 -- Define the property that f is not identically zero
axiom functional_equation (x : ℝ) : x * f (x + 1) = (x + 1) * f x -- Define the given functional equation

theorem value_of_f_nine_halves : f (9 / 2) = 0 := by
  sorry

end NUMINAMATH_GPT_value_of_f_nine_halves_l1100_110083


namespace NUMINAMATH_GPT_solve_system_of_equations_solve_system_of_inequalities_l1100_110036

-- Proof for the system of equations
theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 32) 
  (h2 : 2 * x - y = 0) :
  x = 8 ∧ y = 16 :=
by
  sorry

-- Proof for the system of inequalities
theorem solve_system_of_inequalities (x : ℝ)
  (h3 : 3 * x - 1 < 5 - 2 * x)
  (h4 : 5 * x + 1 ≥ 2 * x + 3) :
  (2 / 3 : ℝ) ≤ x ∧ x < (6 / 5 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_solve_system_of_inequalities_l1100_110036


namespace NUMINAMATH_GPT_complement_of_M_in_U_is_correct_l1100_110051

def U : Set ℤ := {1, -2, 3, -4, 5, -6}
def M : Set ℤ := {1, -2, 3, -4}
def complement_M_in_U : Set ℤ := {5, -6}

theorem complement_of_M_in_U_is_correct : (U \ M) = complement_M_in_U := by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_is_correct_l1100_110051


namespace NUMINAMATH_GPT_quadratic_function_range_l1100_110091

noncomputable def quadratic_range : Set ℝ := {y | -2 ≤ y ∧ y < 2}

theorem quadratic_function_range :
  ∀ y : ℝ, 
    (∃ x : ℝ, -2 < x ∧ x < 1 ∧ y = x^2 + 2 * x - 1) ↔ (y ∈ quadratic_range) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_range_l1100_110091


namespace NUMINAMATH_GPT_polynomial_at_most_one_integer_root_l1100_110026

theorem polynomial_at_most_one_integer_root (n : ℤ) :
  ∀ x1 x2 : ℤ, (x1 ≠ x2) → 
  (x1 ^ 4 - 1993 * x1 ^ 3 + (1993 + n) * x1 ^ 2 - 11 * x1 + n = 0) → 
  (x2 ^ 4 - 1993 * x2 ^ 3 + (1993 + n) * x2 ^ 2 - 11 * x2 + n = 0) → 
  false :=
by
  sorry

end NUMINAMATH_GPT_polynomial_at_most_one_integer_root_l1100_110026


namespace NUMINAMATH_GPT_meaningful_expression_range_l1100_110015

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1100_110015


namespace NUMINAMATH_GPT_Alchemerion_is_3_times_older_than_his_son_l1100_110037

-- Definitions of Alchemerion's age, his father's age and the sum condition
def Alchemerion_age : ℕ := 360
def Father_age (A : ℕ) := 2 * A + 40
def age_sum (A S F : ℕ) := A + S + F

-- Main theorem statement
theorem Alchemerion_is_3_times_older_than_his_son (S : ℕ) (h1 : Alchemerion_age = 360)
    (h2 : Father_age Alchemerion_age = 2 * Alchemerion_age + 40)
    (h3 : age_sum Alchemerion_age S (Father_age Alchemerion_age) = 1240) :
    Alchemerion_age / S = 3 :=
sorry

end NUMINAMATH_GPT_Alchemerion_is_3_times_older_than_his_son_l1100_110037


namespace NUMINAMATH_GPT_total_games_played_l1100_110013

def games_lost : ℕ := 4
def games_won : ℕ := 8

theorem total_games_played : games_lost + games_won = 12 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_games_played_l1100_110013


namespace NUMINAMATH_GPT_radio_show_play_song_duration_l1100_110021

theorem radio_show_play_song_duration :
  ∀ (total_show_time talking_time ad_break_time : ℕ),
  total_show_time = 180 →
  talking_time = 3 * 10 →
  ad_break_time = 5 * 5 →
  total_show_time - (talking_time + ad_break_time) = 125 :=
by
  intros total_show_time talking_time ad_break_time h1 h2 h3
  sorry

end NUMINAMATH_GPT_radio_show_play_song_duration_l1100_110021


namespace NUMINAMATH_GPT_fib_seventh_term_l1100_110055

-- Defining the Fibonacci sequence
def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib n + fib (n + 1)

-- Proving the value of the 7th term given 
-- fib(5) = 5 and fib(6) = 8
theorem fib_seventh_term : fib 7 = 13 :=
by {
    -- Conditions have been used in the definition of Fibonacci sequence
    sorry
}

end NUMINAMATH_GPT_fib_seventh_term_l1100_110055


namespace NUMINAMATH_GPT_jinho_initial_money_l1100_110019

variable (M : ℝ)

theorem jinho_initial_money :
  (M / 2 + 300) + (((M / 2 - 300) / 2) + 400) = M :=
by
  -- This proof is yet to be completed.
  sorry

end NUMINAMATH_GPT_jinho_initial_money_l1100_110019


namespace NUMINAMATH_GPT_total_gold_is_100_l1100_110093

-- Definitions based on conditions
def GregsGold : ℕ := 20
def KatiesGold : ℕ := GregsGold * 4
def TotalGold : ℕ := GregsGold + KatiesGold

-- Theorem to prove
theorem total_gold_is_100 : TotalGold = 100 := by
  sorry

end NUMINAMATH_GPT_total_gold_is_100_l1100_110093


namespace NUMINAMATH_GPT_pure_gala_trees_l1100_110097

variables (T F G : ℕ)

theorem pure_gala_trees :
  (0.1 * T : ℝ) + F = 238 ∧ F = (3 / 4) * ↑T → G = T - F → G = 70 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_pure_gala_trees_l1100_110097


namespace NUMINAMATH_GPT_football_club_balance_l1100_110040

def initial_balance : ℕ := 100
def income := 2 * 10
def cost := 4 * 15
def final_balance := initial_balance + income - cost

theorem football_club_balance : final_balance = 60 := by
  sorry

end NUMINAMATH_GPT_football_club_balance_l1100_110040


namespace NUMINAMATH_GPT_fill_tanker_time_l1100_110077

/-- Given that pipe A can fill the tanker in 60 minutes and pipe B can fill the tanker in 40 minutes,
    prove that the time T to fill the tanker if pipe B is used for half the time and both pipes 
    A and B are used together for the other half is equal to 30 minutes. -/
theorem fill_tanker_time (T : ℝ) (hA : ∀ (a : ℝ), a = 1/60) (hB : ∀ (b : ℝ), b = 1/40) :
  (T / 2) * (1 / 40) + (T / 2) * (1 / 24) = 1 → T = 30 :=
by
  sorry

end NUMINAMATH_GPT_fill_tanker_time_l1100_110077


namespace NUMINAMATH_GPT_Josiah_spent_on_cookies_l1100_110050

theorem Josiah_spent_on_cookies :
  let cookies_per_day := 2
  let cost_per_cookie := 16
  let days_in_march := 31
  2 * days_in_march * cost_per_cookie = 992 := 
by
  sorry

end NUMINAMATH_GPT_Josiah_spent_on_cookies_l1100_110050


namespace NUMINAMATH_GPT_find_A_l1100_110058

def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

theorem find_A (A : ℝ) (h : diamond A 5 = 82) : A = 12 :=
by
  unfold diamond at h
  sorry

end NUMINAMATH_GPT_find_A_l1100_110058


namespace NUMINAMATH_GPT_tangent_line_parabola_l1100_110088

theorem tangent_line_parabola (d : ℝ) :
  (∃ (f g : ℝ → ℝ), (∀ x y, y = f x ↔ y = 3 * x + d) ∧ (∀ x y, y = g x ↔ y ^ 2 = 12 * x)
  ∧ (∀ x y, y = f x ∧ y = g x → y = 3 * x + d ∧ y ^ 2 = 12 * x )) →
  d = 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_parabola_l1100_110088


namespace NUMINAMATH_GPT_intersection_M_N_eq_M_l1100_110072

-- Definition of M
def M := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Definition of N
def N := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_M_N_eq_M : (M ∩ N) = M :=
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_M_l1100_110072


namespace NUMINAMATH_GPT_ten_sided_polygon_diagonals_l1100_110082

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem ten_sided_polygon_diagonals :
  number_of_diagonals 10 = 35 :=
by sorry

end NUMINAMATH_GPT_ten_sided_polygon_diagonals_l1100_110082


namespace NUMINAMATH_GPT_arithmetic_sum_2015_l1100_110049

-- Definitions based on problem conditions
def a1 : ℤ := -2015
def S (n : ℕ) (d : ℤ) : ℤ := n * a1 + n * (n - 1) / 2 * d
def arithmetic_sequence (n : ℕ) (d : ℤ) : ℤ := a1 + (n - 1) * d

-- Proof problem
theorem arithmetic_sum_2015 (d : ℤ) :
  2 * S 6 d - 3 * S 4 d = 24 →
  S 2015 d = -2015 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_2015_l1100_110049


namespace NUMINAMATH_GPT_polynomial_binomial_square_l1100_110000

theorem polynomial_binomial_square (b : ℝ) : 
  (∃ c : ℝ, (3*X + c)^2 = 9*X^2 - 24*X + b) → b = 16 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_binomial_square_l1100_110000


namespace NUMINAMATH_GPT_find_a_l1100_110062

theorem find_a (a : ℝ)
  (hl : ∀ x y : ℝ, ax + 2 * y - a - 2 = 0)
  (hm : ∀ x y : ℝ, 2 * x - y = 0)
  (perpendicular : ∀ x y : ℝ, (2 * - (a / 2)) = -1) : 
  a = 1 := sorry

end NUMINAMATH_GPT_find_a_l1100_110062


namespace NUMINAMATH_GPT_trapezoid_area_calculation_l1100_110028

noncomputable def trapezoid_area : ℝ :=
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2

theorem trapezoid_area_calculation :
  let y1 := 20
  let y2 := 10
  let x1 := y1 / 2
  let x2 := y2 / 2
  let base1 := x1
  let base2 := x2
  let height := y1 - y2
  (base1 + base2) * height / 2 = 75 := 
by
  -- Validation of the translation to Lean 4. Proof steps are omitted.
  sorry

end NUMINAMATH_GPT_trapezoid_area_calculation_l1100_110028


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1100_110059

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) :
  ( ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3 * a) / (a^2 - 1)) = -1/2 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1100_110059


namespace NUMINAMATH_GPT_probability_same_color_probability_different_color_l1100_110076

def count_combinations {α : Type*} (s : Finset α) (k : ℕ) : ℕ :=
  Nat.choose s.card k

noncomputable def count_ways_same_color : ℕ :=
  (count_combinations (Finset.range 3) 2) * 2

noncomputable def count_ways_diff_color : ℕ :=
  (Finset.range 3).card * (Finset.range 3).card

noncomputable def total_ways : ℕ :=
  count_combinations (Finset.range 6) 2

noncomputable def prob_same_color : ℚ :=
  count_ways_same_color / total_ways

noncomputable def prob_diff_color : ℚ :=
  count_ways_diff_color / total_ways

theorem probability_same_color :
  prob_same_color = 2 / 5 := by
  sorry

theorem probability_different_color :
  prob_diff_color = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_probability_different_color_l1100_110076


namespace NUMINAMATH_GPT_tg_half_angle_inequality_l1100_110089

variable (α β γ : ℝ)

theorem tg_half_angle_inequality 
  (h : α + β + γ = 180) : 
  (Real.tan (α / 2)) * (Real.tan (β / 2)) * (Real.tan (γ / 2)) ≤ (Real.sqrt 3) / 9 := 
sorry

end NUMINAMATH_GPT_tg_half_angle_inequality_l1100_110089


namespace NUMINAMATH_GPT_find_adult_ticket_cost_l1100_110043

noncomputable def adult_ticket_cost (A : ℝ) : Prop :=
  let num_adults := 152
  let num_children := num_adults / 2
  let children_ticket_cost := 2.50
  let total_receipts := 1026
  total_receipts = num_adults * A + num_children * children_ticket_cost

theorem find_adult_ticket_cost : adult_ticket_cost 5.50 :=
by
  sorry

end NUMINAMATH_GPT_find_adult_ticket_cost_l1100_110043


namespace NUMINAMATH_GPT_sum_of_digits_is_2640_l1100_110004

theorem sum_of_digits_is_2640 (x : ℕ) (h_cond : (1 + 3 + 4 + 6 + x) * (Nat.factorial 5) = 2640) : x = 8 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_is_2640_l1100_110004


namespace NUMINAMATH_GPT_find_k_l1100_110057

-- Defining the vectors and the condition for parallelism
def vector_a := (2, 1)
def vector_b (k : ℝ) := (k, 3)

def vector_parallel_condition (k : ℝ) : Prop :=
  let a2b := (2 + 2 * k, 7)
  let a2nb := (4 - k, -1)
  (2 + 2 * k) * (-1) = 7 * (4 - k)

theorem find_k (k : ℝ) (h : vector_parallel_condition k) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1100_110057


namespace NUMINAMATH_GPT_solve_fractional_equation_l1100_110064

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 3) : (2 / (x - 3) = 3 / x) → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1100_110064


namespace NUMINAMATH_GPT_functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l1100_110069

noncomputable def daily_sales_profit (x : ℝ) : ℝ :=
  -5 * x^2 + 800 * x - 27500

def profit_maximized (x : ℝ) : Prop :=
  daily_sales_profit x = -5 * (80 - x)^2 + 4500

def sufficient_profit_range (x : ℝ) : Prop :=
  daily_sales_profit x >= 4000 ∧ (x - 50) * (500 - 5 * x) <= 7000

theorem functional_relationship (x : ℝ) : daily_sales_profit x = -5 * x^2 + 800 * x - 27500 :=
  sorry

theorem profit_maximized_at (x : ℝ) : profit_maximized x → x = 80 ∧ daily_sales_profit x = 4500 :=
  sorry

theorem sufficient_profit_range_verified (x : ℝ) : sufficient_profit_range x → 82 ≤ x ∧ x ≤ 90 :=
  sorry

end NUMINAMATH_GPT_functional_relationship_profit_maximized_at_sufficient_profit_range_verified_l1100_110069


namespace NUMINAMATH_GPT_perpendicular_line_eq_l1100_110035

theorem perpendicular_line_eq (x y : ℝ) :
  (∃ (p : ℝ × ℝ), p = (-2, 3) ∧ 
    ∀ y₀ x₀, 3 * x - y = 6 ∧ y₀ = 3 ∧ x₀ = -2 → y = -1 / 3 * x + 7 / 3) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_eq_l1100_110035


namespace NUMINAMATH_GPT_number_of_males_in_village_l1100_110006

-- Given the total population is 800 and it is divided into four equal groups.
def total_population : ℕ := 800
def num_groups : ℕ := 4

-- Proof statement
theorem number_of_males_in_village : (total_population / num_groups) = 200 := 
by sorry

end NUMINAMATH_GPT_number_of_males_in_village_l1100_110006


namespace NUMINAMATH_GPT_number_of_sheets_l1100_110020

theorem number_of_sheets (S E : ℕ) 
  (h1 : S - E = 40)
  (h2 : 5 * E = S) : 
  S = 50 := by 
  sorry

end NUMINAMATH_GPT_number_of_sheets_l1100_110020


namespace NUMINAMATH_GPT_allison_craft_items_l1100_110073

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end NUMINAMATH_GPT_allison_craft_items_l1100_110073


namespace NUMINAMATH_GPT_interval_of_a_l1100_110071

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  f a n.succ  -- since ℕ in Lean includes 0, use n.succ to start from 1

-- The main theorem to prove
theorem interval_of_a (a : ℝ) : (∀ n : ℕ, n ≠ 0 → a_n a n < a_n a (n + 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_interval_of_a_l1100_110071


namespace NUMINAMATH_GPT_find_valid_m_l1100_110034

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem find_valid_m (m : ℝ) : (∀ x, ∃ y, g m x = y ∧ g m y = x) ↔ (m ∈ Set.Iio (-9 / 4) ∪ Set.Ioi (-9 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_find_valid_m_l1100_110034


namespace NUMINAMATH_GPT_last_number_of_ratio_l1100_110061

theorem last_number_of_ratio (A B C : ℕ) (h1 : 5 * B = A) (h2 : 4 * B = C) (h3 : A + B + C = 1000) : C = 400 :=
by
  sorry

end NUMINAMATH_GPT_last_number_of_ratio_l1100_110061


namespace NUMINAMATH_GPT_find_k_value_l1100_110063

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end NUMINAMATH_GPT_find_k_value_l1100_110063


namespace NUMINAMATH_GPT_solve_for_x_l1100_110084

theorem solve_for_x (x : ℤ) (h : 3 * x + 20 = (1/3 : ℚ) * (7 * x + 60)) : x = 0 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1100_110084


namespace NUMINAMATH_GPT_K_set_I_K_set_III_K_set_IV_K_set_V_l1100_110027

-- Definitions for the problem conditions
def K (x y z : ℤ) : ℤ :=
  (x + 2 * y + 3 * z) * (2 * x - y - z) * (y + 2 * z + 3 * x) +
  (y + 2 * z + 3 * x) * (2 * y - z - x) * (z + 2 * x + 3 * y) +
  (z + 2 * x + 3 * y) * (2 * z - x - y) * (x + 2 * y + 3 * z)

-- The equivalent form as a product of terms
def K_equiv (x y z : ℤ) : ℤ :=
  (y + z - 2 * x) * (z + x - 2 * y) * (x + y - 2 * z)

-- Proof statements for each set of numbers
theorem K_set_I : K 1 4 9 = K_equiv 1 4 9 := by
  sorry

theorem K_set_III : K 4 9 1 = K_equiv 4 9 1 := by
  sorry

theorem K_set_IV : K 1 8 11 = K_equiv 1 8 11 := by
  sorry

theorem K_set_V : K 5 8 (-2) = K_equiv 5 8 (-2) := by
  sorry

end NUMINAMATH_GPT_K_set_I_K_set_III_K_set_IV_K_set_V_l1100_110027


namespace NUMINAMATH_GPT_partial_fraction_product_l1100_110068

theorem partial_fraction_product (A B C : ℚ)
  (h_eq : ∀ x, (x^2 - 13) / ((x-2) * (x+2) * (x-3)) = A / (x-2) + B / (x+2) + C / (x-3))
  (h_A : A = 9 / 4)
  (h_B : B = -9 / 20)
  (h_C : C = -4 / 5) :
  A * B * C = 81 / 100 := 
by
  sorry

end NUMINAMATH_GPT_partial_fraction_product_l1100_110068


namespace NUMINAMATH_GPT_likelihood_of_white_crows_at_birch_unchanged_l1100_110085

theorem likelihood_of_white_crows_at_birch_unchanged 
  (a b c d : ℕ) 
  (h1 : a + b = 50) 
  (h2 : c + d = 50) 
  (h3 : b ≥ a) 
  (h4 : d ≥ c - 1) : 
  (bd + ac + a + b : ℝ) / 2550 > (bc + ad : ℝ) / 2550 := by 
  sorry

end NUMINAMATH_GPT_likelihood_of_white_crows_at_birch_unchanged_l1100_110085


namespace NUMINAMATH_GPT_part1_part2_l1100_110045

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 4 ≤ x ∧ x < 8 }
def B : Set ℝ := { x | 3 < x ∧ x < 7 }

theorem part1 :
  (A ∩ B = { x | 4 ≤ x ∧ x < 7 }) ∧
  ((U \ A) ∪ B = { x | x < 7 ∨ x ≥ 8 }) :=
by
  sorry
  
def C (t : ℝ) : Set ℝ := { x | x < t + 1 }

theorem part2 (t : ℝ) :
  (A ∩ C t = ∅) → (t ≤ 3 ∨ t ≥ 7) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1100_110045


namespace NUMINAMATH_GPT_pop_spending_original_l1100_110079

-- Given conditions
def total_spent := 150
def crackle_spending (P : ℝ) := 3 * P
def snap_spending (P : ℝ) := 2 * crackle_spending P

-- Main statement to prove
theorem pop_spending_original : ∃ P : ℝ, snap_spending P + crackle_spending P + P = total_spent ∧ P = 15 :=
by
  sorry

end NUMINAMATH_GPT_pop_spending_original_l1100_110079


namespace NUMINAMATH_GPT_geometric_sequence_term_l1100_110005

theorem geometric_sequence_term :
  ∃ (a_n : ℕ → ℕ),
    -- common ratio condition
    (∀ n, a_n (n + 1) = 2 * a_n n) ∧
    -- sum of first 4 terms condition
    (a_n 1 + a_n 2 + a_n 3 + a_n 4 = 60) ∧
    -- conclusion: value of the third term
    (a_n 3 = 16) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l1100_110005


namespace NUMINAMATH_GPT_soccer_tournament_eq_l1100_110016

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end NUMINAMATH_GPT_soccer_tournament_eq_l1100_110016


namespace NUMINAMATH_GPT_time_saved_correct_l1100_110096

-- Define the conditions as constants
def section1_problems : Nat := 20
def section2_problems : Nat := 15

def time_with_calc_sec1 : Nat := 3
def time_without_calc_sec1 : Nat := 8

def time_with_calc_sec2 : Nat := 5
def time_without_calc_sec2 : Nat := 10

-- Calculate the total times
def total_time_with_calc : Nat :=
  (section1_problems * time_with_calc_sec1) +
  (section2_problems * time_with_calc_sec2)

def total_time_without_calc : Nat :=
  (section1_problems * time_without_calc_sec1) +
  (section2_problems * time_without_calc_sec2)

-- The time saved using a calculator
def time_saved : Nat :=
  total_time_without_calc - total_time_with_calc

-- State the proof problem
theorem time_saved_correct :
  time_saved = 175 := by
  sorry

end NUMINAMATH_GPT_time_saved_correct_l1100_110096


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l1100_110054

theorem abs_inequality_solution_set (x : ℝ) : |x - 1| > 2 ↔ x > 3 ∨ x < -1 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l1100_110054


namespace NUMINAMATH_GPT_find_a_plus_b_l1100_110017

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 2 = a - b / 2) 
  (h2 : 6 = a - b / 3) : 
  a + b = 38 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1100_110017


namespace NUMINAMATH_GPT_sequence_2018_value_l1100_110030

theorem sequence_2018_value (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) - a n = (-1 / 2) ^ n) :
  a 2018 = (2 * (1 - (1 / 2) ^ 2018)) / 3 :=
by sorry

end NUMINAMATH_GPT_sequence_2018_value_l1100_110030


namespace NUMINAMATH_GPT_geometric_mean_l1100_110001

theorem geometric_mean (a b c : ℝ) (h1 : a = 5 + 2 * Real.sqrt 6) (h2 : c = 5 - 2 * Real.sqrt 6) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) (h6 : b^2 = a * c) : b = 1 :=
sorry

end NUMINAMATH_GPT_geometric_mean_l1100_110001


namespace NUMINAMATH_GPT_upstream_distance_18_l1100_110003

theorem upstream_distance_18 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (still_water_speed : ℝ) : 
  upstream_distance = 18 :=
by
  have v := (downstream_distance / downstream_time) - still_water_speed
  have upstream_distance := (still_water_speed - v) * upstream_time
  sorry

end NUMINAMATH_GPT_upstream_distance_18_l1100_110003


namespace NUMINAMATH_GPT_functions_are_same_l1100_110025

def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t - 1

theorem functions_are_same : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_GPT_functions_are_same_l1100_110025


namespace NUMINAMATH_GPT_intersection_m_n_l1100_110052

def M : Set ℝ := { x | (x - 1)^2 < 4 }
def N : Set ℝ := { -1, 0, 1, 2, 3 }

theorem intersection_m_n : M ∩ N = {0, 1, 2} := 
sorry

end NUMINAMATH_GPT_intersection_m_n_l1100_110052


namespace NUMINAMATH_GPT_value_of_f_is_negative_l1100_110060

theorem value_of_f_is_negative {a b c : ℝ} (h1 : a + b < 0) (h2 : b + c < 0) (h3 : c + a < 0) :
  2 * a ^ 3 + 4 * a + 2 * b ^ 3 + 4 * b + 2 * c ^ 3 + 4 * c < 0 := by
sorry

end NUMINAMATH_GPT_value_of_f_is_negative_l1100_110060


namespace NUMINAMATH_GPT_football_team_lineup_ways_l1100_110024

theorem football_team_lineup_ways :
  let members := 12
  let offensive_lineman_options := 4
  let remaining_after_linemen := members - offensive_lineman_options
  let quarterback_options := remaining_after_linemen
  let remaining_after_qb := remaining_after_linemen - 1
  let wide_receiver_options := remaining_after_qb
  let remaining_after_wr := remaining_after_qb - 1
  let tight_end_options := remaining_after_wr
  let lineup_ways := offensive_lineman_options * quarterback_options * wide_receiver_options * tight_end_options
  lineup_ways = 3960 :=
by
  sorry

end NUMINAMATH_GPT_football_team_lineup_ways_l1100_110024


namespace NUMINAMATH_GPT_find_first_offset_l1100_110081

theorem find_first_offset 
  (area : ℝ) (diagonal : ℝ) (offset2 : ℝ) (offset1 : ℝ) 
  (h_area : area = 210) 
  (h_diagonal : diagonal = 28)
  (h_offset2 : offset2 = 6) :
  offset1 = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_first_offset_l1100_110081


namespace NUMINAMATH_GPT_find_c8_l1100_110041

-- Definitions of arithmetic sequences and their products
def arithmetic_seq (a d : ℤ) (n : ℕ) := a + n * d

def c_n (a d1 b d2 : ℤ) (n : ℕ) := arithmetic_seq a d1 n * arithmetic_seq b d2 n

-- Given conditions
variables (a1 d1 a2 d2 : ℤ)
variables (c1 c2 c3 : ℤ)
variables (h1 : c_n a1 d1 a2 d2 1 = 1440)
variables (h2 : c_n a1 d1 a2 d2 2 = 1716)
variables (h3 : c_n a1 d1 a2 d2 3 = 1848)

-- The goal is to prove c_8 = 348
theorem find_c8 : c_n a1 d1 a2 d2 8 = 348 :=
sorry

end NUMINAMATH_GPT_find_c8_l1100_110041


namespace NUMINAMATH_GPT_julie_money_left_l1100_110047

def cost_of_bike : ℕ := 2345
def initial_savings : ℕ := 1500

def mowing_rate : ℕ := 20
def mowing_jobs : ℕ := 20

def paper_rate : ℚ := 0.40
def paper_jobs : ℕ := 600

def dog_rate : ℕ := 15
def dog_jobs : ℕ := 24

def earnings_from_mowing : ℕ := mowing_rate * mowing_jobs
def earnings_from_papers : ℚ := paper_rate * paper_jobs
def earnings_from_dogs : ℕ := dog_rate * dog_jobs

def total_earnings : ℚ := earnings_from_mowing + earnings_from_papers + earnings_from_dogs
def total_money_available : ℚ := initial_savings + total_earnings

def money_left_after_purchase : ℚ := total_money_available - cost_of_bike

theorem julie_money_left : money_left_after_purchase = 155 := sorry

end NUMINAMATH_GPT_julie_money_left_l1100_110047


namespace NUMINAMATH_GPT_correct_order_of_operations_l1100_110042

def order_of_operations (e : String) : String :=
  if e = "38 * 50 - 25 / 5" then
    "multiplication, division, subtraction"
  else
    "unknown"

theorem correct_order_of_operations :
  order_of_operations "38 * 50 - 25 / 5" = "multiplication, division, subtraction" :=
by
  sorry

end NUMINAMATH_GPT_correct_order_of_operations_l1100_110042


namespace NUMINAMATH_GPT_ryan_days_learning_l1100_110078

-- Definitions based on conditions
def hours_per_day_chinese : ℕ := 4
def total_hours_chinese : ℕ := 24

-- Theorem stating the number of days Ryan learns
theorem ryan_days_learning : total_hours_chinese / hours_per_day_chinese = 6 := 
by 
  -- Divide the total hours spent on Chinese learning by hours per day
  sorry

end NUMINAMATH_GPT_ryan_days_learning_l1100_110078


namespace NUMINAMATH_GPT_number_subtracted_from_10000_l1100_110018

theorem number_subtracted_from_10000 (x : ℕ) (h : 10000 - x = 9001) : x = 999 := by
  sorry

end NUMINAMATH_GPT_number_subtracted_from_10000_l1100_110018


namespace NUMINAMATH_GPT_ratio_length_breadth_l1100_110031

theorem ratio_length_breadth
  (b : ℝ) (A : ℝ) (h_b : b = 11) (h_A : A = 363) :
  (∃ l : ℝ, A = l * b ∧ l / b = 3) :=
by
  sorry

end NUMINAMATH_GPT_ratio_length_breadth_l1100_110031


namespace NUMINAMATH_GPT_integer_solution_system_l1100_110053

theorem integer_solution_system (n : ℕ) (H : n ≥ 2) : 
  ∃ (x : ℕ → ℤ), (
    ∀ i : ℕ, x ((i % n) + 1)^2 + x (((i + 1) % n) + 1)^2 + 50 = 16 * x ((i % n) + 1) + 12 * x (((i + 1) % n) + 1)
  ) ↔ n % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_system_l1100_110053


namespace NUMINAMATH_GPT_oak_trees_cut_down_l1100_110098

-- Define the conditions
def initial_oak_trees : ℕ := 9
def final_oak_trees : ℕ := 7

-- Prove that the number of oak trees cut down is 2
theorem oak_trees_cut_down : (initial_oak_trees - final_oak_trees) = 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_oak_trees_cut_down_l1100_110098


namespace NUMINAMATH_GPT_total_toys_l1100_110022

variable (B H : ℕ)

theorem total_toys (h1 : B = 60) (h2 : H = 9 + (B / 2)) : B + H = 99 := by
  sorry

end NUMINAMATH_GPT_total_toys_l1100_110022


namespace NUMINAMATH_GPT_prime_sum_and_difference_l1100_110032

theorem prime_sum_and_difference (m n p : ℕ) (hmprime : Nat.Prime m) (hnprime : Nat.Prime n) (hpprime: Nat.Prime p)
  (h1: m > n)
  (h2: n > p)
  (h3 : m + n + p = 74) 
  (h4 : m - n - p = 44) : 
  m = 59 ∧ n = 13 ∧ p = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_and_difference_l1100_110032


namespace NUMINAMATH_GPT_tooth_extraction_cost_l1100_110056

noncomputable def cleaning_cost : ℕ := 70
noncomputable def filling_cost : ℕ := 120
noncomputable def root_canal_cost : ℕ := 400
noncomputable def crown_cost : ℕ := 600
noncomputable def bridge_cost : ℕ := 800

noncomputable def crown_discount : ℕ := (crown_cost * 20) / 100
noncomputable def bridge_discount : ℕ := (bridge_cost * 10) / 100

noncomputable def total_cost_without_extraction : ℕ := 
  cleaning_cost + 
  3 * filling_cost + 
  root_canal_cost + 
  (crown_cost - crown_discount) + 
  (bridge_cost - bridge_discount)

noncomputable def root_canal_and_one_filling : ℕ := 
  root_canal_cost + filling_cost

noncomputable def dentist_bill : ℕ := 
  11 * root_canal_and_one_filling

theorem tooth_extraction_cost : 
  dentist_bill - total_cost_without_extraction = 3690 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_tooth_extraction_cost_l1100_110056


namespace NUMINAMATH_GPT_sum_coords_B_l1100_110087

def point (A B : ℝ × ℝ) : Prop :=
A = (0, 0) ∧ B.snd = 5 ∧ (B.snd - A.snd) / (B.fst - A.fst) = 3 / 4

theorem sum_coords_B (x : ℝ) :
  point (0, 0) (x, 5) → x + 5 = 35 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_coords_B_l1100_110087


namespace NUMINAMATH_GPT_find_b_value_l1100_110039

theorem find_b_value :
  (∀ x : ℝ, (x < 0 ∨ x > 4) → -x^2 + 4*x - 4 < 0) ↔ b = 4 := by
sorry

end NUMINAMATH_GPT_find_b_value_l1100_110039


namespace NUMINAMATH_GPT_adjacent_abby_bridget_probability_l1100_110086
open Nat

-- Define the conditions
def total_kids := 6
def grid_rows := 3
def grid_cols := 2
def middle_row := 2
def abby_and_bridget := 2

-- Define the probability calculation
theorem adjacent_abby_bridget_probability :
  let total_arrangements := 6!
  let num_ways_adjacent :=
    (2 * abby_and_bridget) * (total_kids - abby_and_bridget)!
  let total_outcomes := total_arrangements
  (num_ways_adjacent / total_outcomes : ℚ) = 4 / 15
:= sorry

end NUMINAMATH_GPT_adjacent_abby_bridget_probability_l1100_110086


namespace NUMINAMATH_GPT_not_possible_to_obtain_target_triple_l1100_110065

def is_target_triple_achievable (a1 a2 a3 b1 b2 b3 : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) = (0.6 * x - 0.8 * y, 0.8 * x + 0.6 * y) →
    (b1^2 + b2^2 + b3^2 = 169 → False)

theorem not_possible_to_obtain_target_triple :
  ¬ is_target_triple_achievable 3 4 12 2 8 10 :=
by sorry

end NUMINAMATH_GPT_not_possible_to_obtain_target_triple_l1100_110065


namespace NUMINAMATH_GPT_jenny_original_amount_half_l1100_110011

-- Definitions based on conditions
def original_amount (x : ℝ) := x
def spent_fraction := 3 / 7
def left_after_spending (x : ℝ) := x * (1 - spent_fraction)

theorem jenny_original_amount_half (x : ℝ) (h : left_after_spending x = 24) : original_amount x / 2 = 21 :=
by
  -- Indicate the intention to prove the statement by sorry
  sorry

end NUMINAMATH_GPT_jenny_original_amount_half_l1100_110011


namespace NUMINAMATH_GPT_perpendicular_line_x_intercept_l1100_110002

theorem perpendicular_line_x_intercept :
  (∃ x : ℝ, ∃ y : ℝ, 4 * x + 5 * y = 10) →
  (∃ y : ℝ, y = (5/4) * x - 3) →
  (∃ x : ℝ, y = 0) →
  x = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_line_x_intercept_l1100_110002


namespace NUMINAMATH_GPT_find_b_of_parabola_axis_of_symmetry_l1100_110007

theorem find_b_of_parabola_axis_of_symmetry (b : ℝ) :
  (∀ (x : ℝ), (x = 1) ↔ (x = - (b / (2 * 2))) ) → b = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_b_of_parabola_axis_of_symmetry_l1100_110007


namespace NUMINAMATH_GPT_fixed_point_of_line_l1100_110066

theorem fixed_point_of_line (a : ℝ) (x y : ℝ)
  (h : ∀ a : ℝ, a * x + y + 1 = 0) :
  x = 0 ∧ y = -1 := 
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_line_l1100_110066


namespace NUMINAMATH_GPT_naturals_less_than_10_l1100_110046

theorem naturals_less_than_10 :
  {n : ℕ | n < 10} = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} :=
by 
  sorry

end NUMINAMATH_GPT_naturals_less_than_10_l1100_110046


namespace NUMINAMATH_GPT_average_death_rate_l1100_110067

-- Definitions and given conditions
def birth_rate_per_two_seconds := 6
def net_increase_per_day := 172800

-- Calculate number of seconds in a day as a constant
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the net increase per second
def net_increase_per_second : ℕ := net_increase_per_day / seconds_per_day

-- Define the birth rate per second
def birth_rate_per_second : ℕ := birth_rate_per_two_seconds / 2

-- The final proof statement
theorem average_death_rate : 
  ∃ (death_rate_per_two_seconds : ℕ), 
    death_rate_per_two_seconds = birth_rate_per_two_seconds - 2 * net_increase_per_second := 
by 
  -- We are required to prove this statement
  use (birth_rate_per_second - net_increase_per_second) * 2
  sorry

end NUMINAMATH_GPT_average_death_rate_l1100_110067


namespace NUMINAMATH_GPT_div_by_eleven_l1100_110023

theorem div_by_eleven (n : ℤ) : 11 ∣ ((n + 11)^2 - n^2) :=
by
  sorry

end NUMINAMATH_GPT_div_by_eleven_l1100_110023


namespace NUMINAMATH_GPT_rectangle_perimeter_l1100_110092

theorem rectangle_perimeter {b : ℕ → ℕ} {W H : ℕ}
  (h1 : ∀ i, b i ≠ b (i+1))
  (h2 : b 9 = W / 2)
  (h3 : gcd W H = 1)

  (h4 : b 1 + b 2 = b 3)
  (h5 : b 1 + b 3 = b 4)
  (h6 : b 3 + b 4 = b 5)
  (h7 : b 4 + b 5 = b 6)
  (h8 : b 2 + b 3 + b 5 = b 7)
  (h9 : b 2 + b 7 = b 8)
  (h10 : b 1 + b 4 + b 6 = b 9)
  (h11 : b 6 + b 9 = b 7 + b 8) : 
  2 * (W + H) = 266 :=
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1100_110092


namespace NUMINAMATH_GPT_soccer_team_points_l1100_110099

theorem soccer_team_points
  (x y : ℕ)
  (h1 : x + y = 8)
  (h2 : 3 * x - y = 12) : 
  (x + y = 8 ∧ 3 * x - y = 12) :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_soccer_team_points_l1100_110099
