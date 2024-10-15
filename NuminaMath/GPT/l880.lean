import Mathlib

namespace NUMINAMATH_GPT_ratio_side_length_to_brush_width_l880_88054

theorem ratio_side_length_to_brush_width (s w : ℝ) (h1 : w = s / 4) (h2 : s^2 / 3 = w^2 + ((s - w)^2) / 2) :
    s / w = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_side_length_to_brush_width_l880_88054


namespace NUMINAMATH_GPT_similar_triangle_legs_l880_88061

theorem similar_triangle_legs {y : ℝ} 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
sorry

end NUMINAMATH_GPT_similar_triangle_legs_l880_88061


namespace NUMINAMATH_GPT_simplify_f_of_alpha_value_of_f_given_cos_l880_88079

variable (α : Real) (f : Real → Real)

def third_quadrant (α : Real) : Prop :=
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

noncomputable def f_def : Real → Real := 
  λ α => (Real.sin (α - Real.pi / 2) * 
           Real.cos (3 * Real.pi / 2 + α) * 
           Real.tan (Real.pi - α)) / 
           (Real.tan (-α - Real.pi) * 
           Real.sin (-Real.pi - α))

theorem simplify_f_of_alpha (h : third_quadrant α) :
  f α = -Real.cos α := sorry

theorem value_of_f_given_cos 
  (h : third_quadrant α) 
  (cos_h : Real.cos (α - 3 / 2 * Real.pi) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := sorry

end NUMINAMATH_GPT_simplify_f_of_alpha_value_of_f_given_cos_l880_88079


namespace NUMINAMATH_GPT_john_total_spent_l880_88000

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end NUMINAMATH_GPT_john_total_spent_l880_88000


namespace NUMINAMATH_GPT_problem_equation_has_solution_l880_88043

noncomputable def x (real_number : ℚ) : ℚ := 210 / 23

theorem problem_equation_has_solution (x_value : ℚ) : 
  (3 / 7) + (7 / x_value) = (10 / x_value) + (1 / 10) → 
  x_value = 210 / 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_equation_has_solution_l880_88043


namespace NUMINAMATH_GPT_flower_count_l880_88004

variables (o y p : ℕ)

theorem flower_count (h1 : y + p = 7) (h2 : o + p = 10) (h3 : o + y = 5) : o + y + p = 11 := sorry

end NUMINAMATH_GPT_flower_count_l880_88004


namespace NUMINAMATH_GPT_bedroom_light_energy_usage_l880_88017

-- Define the conditions and constants
def noahs_bedroom_light_usage (W : ℕ) : ℕ := W
def noahs_office_light_usage (W : ℕ) : ℕ := 3 * W
def noahs_living_room_light_usage (W : ℕ) : ℕ := 4 * W
def total_energy_used (W : ℕ) : ℕ := 2 * (noahs_bedroom_light_usage W + noahs_office_light_usage W + noahs_living_room_light_usage W)
def energy_consumption := 96

-- The main theorem to be proven
theorem bedroom_light_energy_usage : ∃ W : ℕ, total_energy_used W = energy_consumption ∧ W = 6 :=
by
  sorry

end NUMINAMATH_GPT_bedroom_light_energy_usage_l880_88017


namespace NUMINAMATH_GPT_fraction_to_decimal_l880_88071

theorem fraction_to_decimal :
  (7 : ℝ) / (16 : ℝ) = 0.4375 :=
by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l880_88071


namespace NUMINAMATH_GPT_axis_of_symmetry_range_of_t_l880_88045

section
variables (a b m n p t : ℝ)

-- Assume the given conditions
def parabola (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Part (1): Find the axis of symmetry
theorem axis_of_symmetry (h_a_pos : a > 0) 
    (hM : parabola a b 2 = m) 
    (hN : parabola a b 4 = n) 
    (hmn : m = n) : 
    -b / (2 * a) = 3 := 
  sorry

-- Part (2): Find the range of values for t
theorem range_of_t (h_a_pos : a > 0) 
    (hP : parabola a b (-1) = p)
    (axis : -b / (2 * a) = t) 
    (hmn_neg : m * n < 0) 
    (hmpn : m < p ∧ p < n) :
    1 < t ∧ t < 3 / 2 := 
  sorry
end

end NUMINAMATH_GPT_axis_of_symmetry_range_of_t_l880_88045


namespace NUMINAMATH_GPT_parallelogram_area_correct_l880_88091

noncomputable def parallelogram_area (a b : ℝ) (α : ℝ) (h : a < b) : ℝ :=
  (4 * a^2 - b^2) / 4 * (Real.tan α)

theorem parallelogram_area_correct (a b α : ℝ) (h : a < b) :
  parallelogram_area a b α h = (4 * a^2 - b^2) / 4 * (Real.tan α) :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_correct_l880_88091


namespace NUMINAMATH_GPT_area_ratio_triangle_l880_88082

noncomputable def area_ratio (x y : ℝ) (n m : ℕ) : ℝ :=
(x * y) / (2 * n) / ((x * y) / (2 * m))

theorem area_ratio_triangle (x y : ℝ) (n m : ℕ) (hx : 0 ≤ x) (hy : 0 ≤ y) : 
  area_ratio x y n m = (m : ℝ) / (n : ℝ) := by
  sorry

end NUMINAMATH_GPT_area_ratio_triangle_l880_88082


namespace NUMINAMATH_GPT_equal_cost_per_copy_l880_88046

theorem equal_cost_per_copy 
    (x : ℕ) 
    (h₁ : 2000 % x = 0) 
    (h₂ : 3000 % (x + 50) = 0) 
    (h₃ : 2000 / x = 3000 / (x + 50)) :
    (2000 : ℕ) / x = (3000 : ℕ) / (x + 50) :=
by
  sorry

end NUMINAMATH_GPT_equal_cost_per_copy_l880_88046


namespace NUMINAMATH_GPT_num_three_digit_powers_of_three_l880_88020

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end NUMINAMATH_GPT_num_three_digit_powers_of_three_l880_88020


namespace NUMINAMATH_GPT_smallest_n_l880_88021

theorem smallest_n (n : ℕ) (h₁ : ∃ k1 : ℕ, 4 * n = k1 ^ 2) (h₂ : ∃ k2 : ℕ, 3 * n = k2 ^ 3) : n = 18 :=
sorry

end NUMINAMATH_GPT_smallest_n_l880_88021


namespace NUMINAMATH_GPT_equal_charges_at_4_hours_l880_88035

-- Define the charges for both companies
def PaulsPlumbingCharge (h : ℝ) : ℝ := 55 + 35 * h
def ReliablePlumbingCharge (h : ℝ) : ℝ := 75 + 30 * h

-- Prove that for 4 hours of labor, the charges are equal
theorem equal_charges_at_4_hours : PaulsPlumbingCharge 4 = ReliablePlumbingCharge 4 :=
by
  sorry

end NUMINAMATH_GPT_equal_charges_at_4_hours_l880_88035


namespace NUMINAMATH_GPT_negation_of_proposition_l880_88012

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), (a = b → a^2 = a * b)) = ∀ (a b : ℝ), (a ≠ b → a^2 ≠ a * b) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l880_88012


namespace NUMINAMATH_GPT_number_of_real_a_l880_88063

open Int

-- Define the quadratic equation with integer roots
def quadratic_eq_with_integer_roots (a : ℝ) : Prop :=
  ∃ (r s : ℤ), r + s = -a ∧ r * s = 12 * a

-- Prove there are exactly 9 values of a such that the quadratic equation has only integer roots
theorem number_of_real_a (n : ℕ) : n = 9 ↔ ∃ (as : Finset ℝ), as.card = n ∧ ∀ a ∈ as, quadratic_eq_with_integer_roots a :=
by
  -- We can skip the proof with "sorry"
  sorry

end NUMINAMATH_GPT_number_of_real_a_l880_88063


namespace NUMINAMATH_GPT_card_probability_l880_88096

theorem card_probability :
  let totalCards := 52
  let kings := 4
  let jacks := 4
  let queens := 4
  let firstCardKing := kings / totalCards
  let secondCardJack := jacks / (totalCards - 1)
  let thirdCardQueen := queens / (totalCards - 2)
  (firstCardKing * secondCardJack * thirdCardQueen) = (8 / 16575) :=
by
  sorry

end NUMINAMATH_GPT_card_probability_l880_88096


namespace NUMINAMATH_GPT_probability_of_green_ball_l880_88087

-- Define the number of balls in each container
def number_balls_I := (10, 5)  -- (red, green)
def number_balls_II := (3, 6)  -- (red, green)
def number_balls_III := (3, 6)  -- (red, green)

-- Define the probability of selecting each container
noncomputable def probability_container_selected := (1 / 3 : ℝ)

-- Define the probability of drawing a green ball from each container
noncomputable def probability_green_I := (number_balls_I.snd : ℝ) / ((number_balls_I.fst + number_balls_I.snd) : ℝ)
noncomputable def probability_green_II := (number_balls_II.snd : ℝ) / ((number_balls_II.fst + number_balls_II.snd) : ℝ)
noncomputable def probability_green_III := (number_balls_III.snd : ℝ) / ((number_balls_III.fst + number_balls_III.snd) : ℝ)

-- Define the combined probabilities for drawing a green ball and selecting each container
noncomputable def combined_probability_I := probability_container_selected * probability_green_I
noncomputable def combined_probability_II := probability_container_selected * probability_green_II
noncomputable def combined_probability_III := probability_container_selected * probability_green_III

-- Define the total probability of drawing a green ball
noncomputable def total_probability_green := combined_probability_I + combined_probability_II + combined_probability_III

-- The theorem to be proved
theorem probability_of_green_ball : total_probability_green = (5 / 9 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_green_ball_l880_88087


namespace NUMINAMATH_GPT_water_amount_in_sport_formulation_l880_88006

/-
The standard formulation has the ratios:
F : CS : W = 1 : 12 : 30
Where F is flavoring, CS is corn syrup, and W is water.
-/

def standard_flavoring_ratio : ℚ := 1
def standard_corn_syrup_ratio : ℚ := 12
def standard_water_ratio : ℚ := 30

/-
In the sport formulation:
1) The ratio of flavoring to corn syrup is three times as great as in the standard formulation.
2) The ratio of flavoring to water is half that of the standard formulation.
-/
def sport_flavor_to_corn_ratio : ℚ := 3 * (standard_flavoring_ratio / standard_corn_syrup_ratio)
def sport_flavor_to_water_ratio : ℚ := 1 / 2 * (standard_flavoring_ratio / standard_water_ratio)

/-
The sport formulation contains 6 ounces of corn syrup.
The target is to find the amount of water in the sport formulation.
-/
def corn_syrup_in_sport_formulation : ℚ := 6
def flavoring_in_sport_formulation : ℚ := sport_flavor_to_corn_ratio * corn_syrup_in_sport_formulation

def water_in_sport_formulation : ℚ := 
  (flavoring_in_sport_formulation / sport_flavor_to_water_ratio)

theorem water_amount_in_sport_formulation : water_in_sport_formulation = 90 := by
  sorry

end NUMINAMATH_GPT_water_amount_in_sport_formulation_l880_88006


namespace NUMINAMATH_GPT_initial_distance_l880_88083

theorem initial_distance (speed_enrique speed_jamal : ℝ) (hours : ℝ) 
  (h_enrique : speed_enrique = 16) 
  (h_jamal : speed_jamal = 23) 
  (h_time : hours = 8) 
  (h_difference : speed_jamal = speed_enrique + 7) : 
  (speed_enrique * hours + speed_jamal * hours = 312) :=
by 
  sorry

end NUMINAMATH_GPT_initial_distance_l880_88083


namespace NUMINAMATH_GPT_area_of_square_on_PS_l880_88031

-- Given parameters as conditions in the form of hypotheses
variables (PQ QR RS PS PR : ℝ)

-- Hypotheses based on problem conditions
def hypothesis1 : PQ^2 = 25 := sorry
def hypothesis2 : QR^2 = 49 := sorry
def hypothesis3 : RS^2 = 64 := sorry
def hypothesis4 : PR^2 = PQ^2 + QR^2 := sorry
def hypothesis5 : PS^2 = PR^2 - RS^2 := sorry

-- The main theorem we need to prove
theorem area_of_square_on_PS :
  PS^2 = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_area_of_square_on_PS_l880_88031


namespace NUMINAMATH_GPT_speed_comparison_l880_88052

theorem speed_comparison (v v2 : ℝ) (h1 : v2 > 0) (h2 : v = 5 * v2) : v = 5 * v2 :=
by
  exact h2 

end NUMINAMATH_GPT_speed_comparison_l880_88052


namespace NUMINAMATH_GPT_find_a_b_l880_88001

theorem find_a_b (a b : ℝ) : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) → a = 5 ∧ b = -6 :=
sorry

end NUMINAMATH_GPT_find_a_b_l880_88001


namespace NUMINAMATH_GPT_max_bars_scenario_a_max_bars_scenario_b_l880_88074

-- Define the game conditions and the maximum bars Ivan can take in each scenario.

def max_bars_taken (initial_bars : ℕ) : ℕ :=
  if initial_bars = 14 then 13 else 13

theorem max_bars_scenario_a :
  max_bars_taken 13 = 13 :=
by sorry

theorem max_bars_scenario_b :
  max_bars_taken 14 = 13 :=
by sorry

end NUMINAMATH_GPT_max_bars_scenario_a_max_bars_scenario_b_l880_88074


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l880_88053

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l880_88053


namespace NUMINAMATH_GPT_smallest_five_digit_number_divisibility_l880_88072

-- Define the smallest 5-digit number satisfying the conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

theorem smallest_five_digit_number_divisibility :
  ∃ (n : ℕ), isDivisibleBy n 15
          ∧ isDivisibleBy n (2^8)
          ∧ isDivisibleBy n 45
          ∧ isDivisibleBy n 54
          ∧ n >= 10000
          ∧ n < 100000
          ∧ n = 69120 :=
sorry

end NUMINAMATH_GPT_smallest_five_digit_number_divisibility_l880_88072


namespace NUMINAMATH_GPT_greatest_ABCBA_divisible_by_13_l880_88026

theorem greatest_ABCBA_divisible_by_13 :
  ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  1 ≤ A ∧ A < 10 ∧ 0 ≤ B ∧ B < 10 ∧ 0 ≤ C ∧ C < 10 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) % 13 = 0 ∧
  (10000 * A + 1000 * B + 100 * C + 10 * B + A) = 95159 :=
by
  sorry

end NUMINAMATH_GPT_greatest_ABCBA_divisible_by_13_l880_88026


namespace NUMINAMATH_GPT_right_triangle_perimeter_l880_88097

-- Given conditions
variable (x y : ℕ)
def leg1 := 11
def right_triangle := (101 * 11 = 121)

-- The question and answer
theorem right_triangle_perimeter :
  (y + x = 121) ∧ (y - x = 1) → (11 + x + y = 132) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l880_88097


namespace NUMINAMATH_GPT_optimal_saving_is_45_cents_l880_88005

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

def price_after_fixed_discount (price fixed_discount : ℝ) : ℝ :=
  price - fixed_discount

def price_after_percentage_discount (price percentage_discount : ℝ) : ℝ :=
  price * (1 - percentage_discount)

def optimal_saving (initial_price fixed_discount percentage_discount : ℝ) : ℝ :=
  let price1 := price_after_fixed_discount initial_price fixed_discount
  let final_price1 := price_after_percentage_discount price1 percentage_discount
  let price2 := price_after_percentage_discount initial_price percentage_discount
  let final_price2 := price_after_fixed_discount price2 fixed_discount
  final_price1 - final_price2

theorem optimal_saving_is_45_cents : optimal_saving initial_price fixed_discount percentage_discount = 0.45 :=
by 
  sorry

end NUMINAMATH_GPT_optimal_saving_is_45_cents_l880_88005


namespace NUMINAMATH_GPT_four_minus_x_is_five_l880_88085

theorem four_minus_x_is_five (x y : ℤ) (h1 : 4 + x = 5 - y) (h2 : 3 + y = 6 + x) : 4 - x = 5 := by
sorry

end NUMINAMATH_GPT_four_minus_x_is_five_l880_88085


namespace NUMINAMATH_GPT_find_marks_in_biology_l880_88044

-- Definitions based on conditions in a)
def marks_english : ℕ := 76
def marks_math : ℕ := 60
def marks_physics : ℕ := 72
def marks_chemistry : ℕ := 65
def num_subjects : ℕ := 5
def average_marks : ℕ := 71

-- The theorem that needs to be proved
theorem find_marks_in_biology : 
  let total_marks := marks_english + marks_math + marks_physics + marks_chemistry 
  let total_marks_all := average_marks * num_subjects
  let marks_biology := total_marks_all - total_marks
  marks_biology = 82 := 
by
  sorry

end NUMINAMATH_GPT_find_marks_in_biology_l880_88044


namespace NUMINAMATH_GPT_fraction_irreducibility_l880_88078

theorem fraction_irreducibility (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_irreducibility_l880_88078


namespace NUMINAMATH_GPT_greatest_integer_b_for_no_real_roots_l880_88075

theorem greatest_integer_b_for_no_real_roots :
  ∃ (b : ℤ), (b * b < 20) ∧ (∀ (c : ℤ), (c * c < 20) → c ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_b_for_no_real_roots_l880_88075


namespace NUMINAMATH_GPT_oranges_in_box_l880_88033

theorem oranges_in_box :
  ∃ (A P O : ℕ), A + P + O = 60 ∧ A = 3 * (P + O) ∧ P = (A + O) / 5 ∧ O = 5 :=
by
  sorry

end NUMINAMATH_GPT_oranges_in_box_l880_88033


namespace NUMINAMATH_GPT_original_curve_eqn_l880_88094

-- Definitions based on conditions
def scaling_transformation_formula (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

-- The proof problem to be shown in Lean
theorem original_curve_eqn {x y : ℝ} (h : transformed_curve (2 * x) (3 * y)) :
  4 * x^2 + 9 * y^2 = 1 :=
sorry

end NUMINAMATH_GPT_original_curve_eqn_l880_88094


namespace NUMINAMATH_GPT_monotonic_increasing_on_interval_l880_88048

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

theorem monotonic_increasing_on_interval (ω : ℝ) (h1 : ω > 0) (h2 : 2 * Real.pi / (2 * ω) = 4 * Real.pi) :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 2) Real.pi) → (y ∈ Set.Icc (Real.pi / 2) Real.pi) → x ≤ y → f ω x ≤ f ω y := 
by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_on_interval_l880_88048


namespace NUMINAMATH_GPT_value_of_g_3_l880_88019

def g (x : ℚ) : ℚ := (x^2 + x + 1) / (5*x - 3)

theorem value_of_g_3 : g 3 = 13 / 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_g_3_l880_88019


namespace NUMINAMATH_GPT_find_x_l880_88015

theorem find_x : 
  (5 * 12 / (180 / 3) = 1) → (∃ x : ℕ, 1 + x = 81 ∧ x = 80) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l880_88015


namespace NUMINAMATH_GPT_percentage_y_less_than_x_l880_88036

theorem percentage_y_less_than_x (x y : ℝ) (h : x = 11 * y) : 
  ((x - y) / x) * 100 = 90.91 := 
by 
  sorry -- proof to be provided separately

end NUMINAMATH_GPT_percentage_y_less_than_x_l880_88036


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l880_88013

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l880_88013


namespace NUMINAMATH_GPT_michael_left_money_l880_88009

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end NUMINAMATH_GPT_michael_left_money_l880_88009


namespace NUMINAMATH_GPT_barley_percentage_is_80_l880_88030

variables (T C : ℝ) -- Total land and cleared land
variables (B : ℝ) -- Percentage of cleared land planted with barley

-- Given conditions
def cleared_land (T : ℝ) : ℝ := 0.9 * T
def total_land_approx : ℝ := 1000
def potato_land (C : ℝ) : ℝ := 0.1 * C
def tomato_land : ℝ := 90
def barley_percentage (C : ℝ) (B : ℝ) : Prop := C - (potato_land C) - tomato_land = (B / 100) * C

-- Theorem statement to prove
theorem barley_percentage_is_80 :
  cleared_land total_land_approx = 900 → barley_percentage 900 80 :=
by
  intros hC
  rw [cleared_land, total_land_approx] at hC
  simp [barley_percentage, potato_land, tomato_land]
  sorry

end NUMINAMATH_GPT_barley_percentage_is_80_l880_88030


namespace NUMINAMATH_GPT_find_values_of_x_and_y_l880_88023

-- Define the conditions
def first_condition (x : ℝ) : Prop := 0.75 / x = 5 / 7
def second_condition (y : ℝ) : Prop := y / 19 = 11 / 3

-- Define the main theorem to prove
theorem find_values_of_x_and_y (x y : ℝ) (h1 : first_condition x) (h2 : second_condition y) :
  x = 1.05 ∧ y = 209 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_values_of_x_and_y_l880_88023


namespace NUMINAMATH_GPT_central_angle_proof_l880_88014

noncomputable def central_angle (l r : ℝ) : ℝ :=
  l / r

theorem central_angle_proof :
  central_angle 300 100 = 3 :=
by
  -- The statement of the theorem aligns with the given problem conditions and the expected answer.
  sorry

end NUMINAMATH_GPT_central_angle_proof_l880_88014


namespace NUMINAMATH_GPT_largest_possible_b_l880_88040

theorem largest_possible_b (b : ℝ) (h : (3 * b + 6) * (b - 2) = 9 * b) : b ≤ 4 := 
by {
  -- leaving the proof as an exercise, using 'sorry' to complete the statement
  sorry
}

end NUMINAMATH_GPT_largest_possible_b_l880_88040


namespace NUMINAMATH_GPT_mike_oranges_l880_88093

-- Definitions and conditions
variables (O A B : ℕ)
def condition1 := A = 2 * O
def condition2 := B = O + A
def condition3 := O + A + B = 18

-- Theorem to prove that Mike received 3 oranges
theorem mike_oranges (h1 : condition1 O A) (h2 : condition2 O A B) (h3 : condition3 O A B) : 
  O = 3 := 
by 
  sorry

end NUMINAMATH_GPT_mike_oranges_l880_88093


namespace NUMINAMATH_GPT_proof_problem_l880_88008

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l880_88008


namespace NUMINAMATH_GPT_option_A_option_B_option_D_l880_88051

-- Given real numbers a, b, c such that a > b > 1 and c > 0,
-- prove the following inequalities.
variables {a b c : ℝ}

-- Assume the conditions
axiom H1 : a > b
axiom H2 : b > 1
axiom H3 : c > 0

-- Statements to prove
theorem option_A (H1: a > b) (H2: b > 1) (H3: c > 0) : a^2 - bc > b^2 - ac := sorry
theorem option_B (H1: a > b) (H2: b > 1) : a^3 > b^2 := sorry
theorem option_D (H1: a > b) (H2: b > 1) : a + (1/a) > b + (1/b) := sorry
  
end NUMINAMATH_GPT_option_A_option_B_option_D_l880_88051


namespace NUMINAMATH_GPT_midpoints_distance_l880_88081

theorem midpoints_distance
  (A B C D M N : ℝ)
  (h1 : M = (A + C) / 2)
  (h2 : N = (B + D) / 2)
  (h3 : D - A = 68)
  (h4 : C - B = 26)
  : abs (M - N) = 21 := 
sorry

end NUMINAMATH_GPT_midpoints_distance_l880_88081


namespace NUMINAMATH_GPT_consecutive_integers_divisible_by_12_l880_88095

theorem consecutive_integers_divisible_by_12 (a b c d : ℤ) 
  (h1 : b = a + 1) (h2 : c = b + 1) (h3 : d = c + 1) : 
  12 ∣ (a * b + a * c + a * d + b * c + b * d + c * d + 1) := 
sorry

end NUMINAMATH_GPT_consecutive_integers_divisible_by_12_l880_88095


namespace NUMINAMATH_GPT_exactly_one_defective_l880_88039

theorem exactly_one_defective (p_A p_B : ℝ) (hA : p_A = 0.04) (hB : p_B = 0.05) :
  ((p_A * (1 - p_B)) + ((1 - p_A) * p_B)) = 0.086 :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_defective_l880_88039


namespace NUMINAMATH_GPT_year_2013_is_not_special_l880_88086

def is_special_year (year : ℕ) : Prop :=
  ∃ (month day : ℕ), month * day = year % 100 ∧ month ≥ 1 ∧ month ≤ 12 ∧ day ≥ 1 ∧ day ≤ 31

theorem year_2013_is_not_special : ¬ is_special_year 2013 := by
  sorry

end NUMINAMATH_GPT_year_2013_is_not_special_l880_88086


namespace NUMINAMATH_GPT_find_interest_rate_l880_88032

-- Definitions for conditions
def principal : ℝ := 12500
def interest : ℝ := 1500
def time : ℝ := 1

-- Interest rate to prove
def interest_rate : ℝ := 0.12

-- Formal statement to prove
theorem find_interest_rate (P I T : ℝ) (hP : P = principal) (hI : I = interest) (hT : T = time) : I = P * interest_rate * T :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l880_88032


namespace NUMINAMATH_GPT_apples_from_C_to_D_l880_88060

theorem apples_from_C_to_D (n m : ℕ)
  (h_tree_ratio : ∀ (P V : ℕ), P = 2 * V)
  (h_apple_ratio : ∀ (P V : ℕ), P = 7 * V)
  (trees_CD_Petya trees_CD_Vasya : ℕ)
  (h_trees_CD : trees_CD_Petya = 2 * trees_CD_Vasya)
  (apples_CD_Petya apples_CD_Vasya: ℕ)
  (h_apples_CD : apples_CD_Petya = (m / 4) ∧ apples_CD_Vasya = (3 * m / 4)) : 
  apples_CD_Vasya = 3 * apples_CD_Petya := by
  sorry

end NUMINAMATH_GPT_apples_from_C_to_D_l880_88060


namespace NUMINAMATH_GPT_Apollonius_circle_symmetry_l880_88099

theorem Apollonius_circle_symmetry (a : ℝ) (h : a > 1): 
  let F1 := (-1, 0)
  let F2 := (1, 0)
  let locus_C := {P : ℝ × ℝ | ∃ x y, P = (x, y) ∧ (Real.sqrt ((x + 1)^2 + y^2) = a * Real.sqrt ((x - 1)^2 + y^2))}
  let symmetric_y := ∀ (P : ℝ × ℝ), P ∈ locus_C → (P.1, -P.2) ∈ locus_C
  symmetric_y := sorry

end NUMINAMATH_GPT_Apollonius_circle_symmetry_l880_88099


namespace NUMINAMATH_GPT_value_of_expression_l880_88088

theorem value_of_expression : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l880_88088


namespace NUMINAMATH_GPT_cube_faces_consecutive_sum_l880_88041

noncomputable def cube_face_sum (n : ℕ) : ℕ :=
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)

theorem cube_faces_consecutive_sum (n : ℕ) (h1 : ∀ i, i ∈ [0, 5] -> (2 * n + 5 + n + 5 - 6) = 6) (h2 : n = 12) :
  cube_face_sum n = 87 :=
  sorry

end NUMINAMATH_GPT_cube_faces_consecutive_sum_l880_88041


namespace NUMINAMATH_GPT_smallest_difference_l880_88047

variable (DE EF FD : ℕ)

def is_valid_triangle (DE EF FD : ℕ) : Prop :=
  DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference (h1 : DE < EF)
                           (h2 : EF ≤ FD)
                           (h3 : DE + EF + FD = 1024)
                           (h4 : is_valid_triangle DE EF FD) :
  ∃ d, d = EF - DE ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_difference_l880_88047


namespace NUMINAMATH_GPT_largest_prime_divisor_of_360_is_5_l880_88016

theorem largest_prime_divisor_of_360_is_5 (p : ℕ) (hp₁ : Nat.Prime p) (hp₂ : p ∣ 360) : p ≤ 5 :=
by 
sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_360_is_5_l880_88016


namespace NUMINAMATH_GPT_perpendicular_lines_slope_l880_88027

theorem perpendicular_lines_slope :
  ∀ (a : ℚ), (∀ x y : ℚ, y = 3 * x + 5) 
  ∧ (∀ x y : ℚ, 4 * y + a * x = 8) →
  a = 4 / 3 :=
by
  intro a
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l880_88027


namespace NUMINAMATH_GPT_area_of_triangle_is_correct_l880_88028

def vector := (ℝ × ℝ)

def a : vector := (7, 3)
def b : vector := (-1, 5)

noncomputable def det2x2 (v1 v2 : vector) : ℝ :=
  (v1.1 * v2.2) - (v1.2 * v2.1)

theorem area_of_triangle_is_correct :
  let area := (det2x2 a b) / 2
  area = 19 := by
  -- defintions and conditions are set here, proof skipped
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_correct_l880_88028


namespace NUMINAMATH_GPT_calculate_ff2_l880_88038

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 4

theorem calculate_ff2 : f (f 2) = 5450 := by
  sorry

end NUMINAMATH_GPT_calculate_ff2_l880_88038


namespace NUMINAMATH_GPT_average_of_three_marbles_l880_88059

-- Define the conditions as hypotheses
theorem average_of_three_marbles (R Y B : ℕ) 
  (h1 : R + Y = 53)
  (h2 : B + Y = 69)
  (h3 : R + B = 58) :
  (R + Y + B) / 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_average_of_three_marbles_l880_88059


namespace NUMINAMATH_GPT_speed_of_man_l880_88073

open Real Int

/-- 
  A train 110 m long is running with a speed of 40 km/h.
  The train passes a man who is running at a certain speed
  in the direction opposite to that in which the train is going.
  The train takes 9 seconds to pass the man.
  This theorem proves that the speed of the man is 3.992 km/h.
-/
theorem speed_of_man (T_length : ℝ) (T_speed : ℝ) (t_pass : ℝ) (M_speed : ℝ) : 
  T_length = 110 → T_speed = 40 → t_pass = 9 → M_speed = 3.992 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_speed_of_man_l880_88073


namespace NUMINAMATH_GPT_fraction_sum_to_decimal_l880_88037

theorem fraction_sum_to_decimal : 
  (3 / 10 : Rat) + (5 / 100) - (1 / 1000) = 349 / 1000 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_to_decimal_l880_88037


namespace NUMINAMATH_GPT_monotonic_intervals_l880_88022

noncomputable def f (a x : ℝ) : ℝ := x^2 * Real.exp (a * x)

theorem monotonic_intervals (a : ℝ) :
  (a = 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > 0 → f a x > f a 1))) ∧
  (a > 0 → (∀ x : ℝ, (x < -2 / a → f a x < f a (-2 / a - 1)) ∧ (x > 0 → f a x > f a 1) ∧ 
                  ((-2 / a) < x ∧ x < 0 → f a x < f a (-2 / a + 1)))) ∧
  (a < 0 → (∀ x : ℝ, (x < 0 → f a x < f a (-1)) ∧ (x > -2 / a → f a x < f a (-2 / a - 1)) ∧
                  (0 < x ∧ x < -2 / a → f a x > f a (-2 / a + 1))))
:= sorry

end NUMINAMATH_GPT_monotonic_intervals_l880_88022


namespace NUMINAMATH_GPT_smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l880_88024

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem minimum_value_of_f :
  ∃ x, f x = -3 :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ p, (∀ x, f (p + x) = f (p - x)) ∧ p = (Real.pi / 12) + (k * Real.pi / 2) :=
sorry

theorem interval_of_increasing (k : ℤ) :
  ∃ a b, a = -(Real.pi / 6) + k * Real.pi ∧ b = (Real.pi / 3) + k * Real.pi ∧
  ∀ x, (a <= x ∧ x <= b) → StrictMonoOn f (Set.Icc a b) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l880_88024


namespace NUMINAMATH_GPT_chocolates_difference_l880_88080

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ) (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 5) : robert_chocolates - nickel_chocolates = 2 :=
by sorry

end NUMINAMATH_GPT_chocolates_difference_l880_88080


namespace NUMINAMATH_GPT_number_of_throwers_l880_88065

theorem number_of_throwers (T N : ℕ) :
  (T + N = 61) ∧ ((2 * N) / 3 = 53 - T) → T = 37 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_throwers_l880_88065


namespace NUMINAMATH_GPT_sum_of_remainders_l880_88070

theorem sum_of_remainders (a b c d : ℤ) (h1 : a % 53 = 33) (h2 : b % 53 = 25) (h3 : c % 53 = 6) (h4 : d % 53 = 12) : 
  (a + b + c + d) % 53 = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_remainders_l880_88070


namespace NUMINAMATH_GPT_number_of_students_third_l880_88049

-- Define the ratio and the total number of samples.
def ratio_first : ℕ := 3
def ratio_second : ℕ := 3
def ratio_third : ℕ := 4
def total_sample : ℕ := 50

-- Define the condition that the sum of ratios equals the total proportion numerator.
def sum_ratios : ℕ := ratio_first + ratio_second + ratio_third

-- Final proposition: the number of students to be sampled from the third grade.
theorem number_of_students_third :
  (ratio_third * total_sample) / sum_ratios = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_students_third_l880_88049


namespace NUMINAMATH_GPT_total_bill_l880_88058

/-
Ten friends dined at a restaurant and split the bill equally.
One friend, Chris, forgets his money.
Each of the remaining nine friends agreed to pay an extra $3 to cover Chris's share.
How much was the total bill?

Correct answer: 270
-/

theorem total_bill (t : ℕ) (h1 : ∀ x, t = 10 * x) (h2 : ∀ x, t = 9 * (x + 3)) : t = 270 := by
  sorry

end NUMINAMATH_GPT_total_bill_l880_88058


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l880_88025

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 4 + a 8 = 4 →
  S 11 + a 6 = 24 :=
by
  intros a S h1 h2
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l880_88025


namespace NUMINAMATH_GPT_train_distance_problem_l880_88062

theorem train_distance_problem
  (Vx : ℝ) (Vy : ℝ) (t : ℝ) (distanceX : ℝ) 
  (h1 : Vx = 32) 
  (h2 : Vy = 160 / 3) 
  (h3 : 32 * t + (160 / 3) * t = 160) :
  distanceX = Vx * t → distanceX = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_distance_problem_l880_88062


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l880_88077

theorem sum_of_squares_of_roots :
  ∀ (p q r : ℚ), (3 * p^3 + 2 * p^2 - 5 * p - 8 = 0) ∧
                 (3 * q^3 + 2 * q^2 - 5 * q - 8 = 0) ∧
                 (3 * r^3 + 2 * r^2 - 5 * r - 8 = 0) →
                 p^2 + q^2 + r^2 = 34 / 9 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l880_88077


namespace NUMINAMATH_GPT_solution_set_of_inequalities_l880_88089

theorem solution_set_of_inequalities (m n : ℝ) (h1 : m < 0) (h2 : n > 0) (h3 : ∀ x, mx + n > 0 ↔ x < (1/3)) : ∀ x, nx - m < 0 ↔ x < -3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequalities_l880_88089


namespace NUMINAMATH_GPT_little_john_money_left_l880_88002

-- Define the variables with the given conditions
def initAmount : ℚ := 5.10
def spentOnSweets : ℚ := 1.05
def givenToEachFriend : ℚ := 1.00

-- The problem statement
theorem little_john_money_left :
  (initAmount - spentOnSweets - 2 * givenToEachFriend) = 2.05 :=
by
  sorry

end NUMINAMATH_GPT_little_john_money_left_l880_88002


namespace NUMINAMATH_GPT_minutes_practiced_other_days_l880_88003

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end NUMINAMATH_GPT_minutes_practiced_other_days_l880_88003


namespace NUMINAMATH_GPT_tan_7pi_over_4_eq_neg1_l880_88067

theorem tan_7pi_over_4_eq_neg1 : Real.tan (7 * Real.pi / 4) = -1 :=
  sorry

end NUMINAMATH_GPT_tan_7pi_over_4_eq_neg1_l880_88067


namespace NUMINAMATH_GPT_female_employees_sampled_l880_88084

theorem female_employees_sampled
  (T : ℕ) -- Total number of employees
  (M : ℕ) -- Number of male employees
  (F : ℕ) -- Number of female employees
  (S_m : ℕ) -- Number of sampled male employees
  (H_T : T = 140)
  (H_M : M = 80)
  (H_F : F = 60)
  (H_Sm : S_m = 16) :
  ∃ S_f : ℕ, S_f = 12 :=
by
  sorry

end NUMINAMATH_GPT_female_employees_sampled_l880_88084


namespace NUMINAMATH_GPT_average_episodes_per_year_l880_88029

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end NUMINAMATH_GPT_average_episodes_per_year_l880_88029


namespace NUMINAMATH_GPT_problem1_subproblem1_subproblem2_l880_88092

-- Problem 1: Prove that a² + b² = 40 given ab = 30 and a + b = 10
theorem problem1 (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) : a^2 + b^2 = 40 := 
sorry

-- Problem 2: Subproblem 1 - Prove that (40 - x)² + (x - 20)² = 420 given (40 - x)(x - 20) = -10
theorem subproblem1 (x : ℝ) (h : (40 - x) * (x - 20) = -10) : (40 - x)^2 + (x - 20)^2 = 420 := 
sorry

-- Problem 2: Subproblem 2 - Prove that (30 + x)² + (20 + x)² = 120 given (30 + x)(20 + x) = 10
theorem subproblem2 (x : ℝ) (h : (30 + x) * (20 + x) = 10) : (30 + x)^2 + (20 + x)^2 = 120 :=
sorry

end NUMINAMATH_GPT_problem1_subproblem1_subproblem2_l880_88092


namespace NUMINAMATH_GPT_y_intercept_l880_88007

theorem y_intercept : ∀ (x y : ℝ), 4 * x + 7 * y = 28 → (0, 4) = (0, y) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_y_intercept_l880_88007


namespace NUMINAMATH_GPT_consecutive_integers_sum_l880_88098

open Nat

theorem consecutive_integers_sum (n : ℕ) (h : (n - 1) * n * (n + 1) = 336) : (n - 1) + n + (n + 1) = 21 := 
by 
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l880_88098


namespace NUMINAMATH_GPT_distance_sum_is_ten_l880_88018

noncomputable def angle_sum_distance (C A B : ℝ) (d : ℝ) (k : ℝ) : ℝ := 
  let h_A : ℝ := sorry -- replace with expression for h_A based on conditions
  let h_B : ℝ := sorry -- replace with expression for h_B based on conditions
  h_A + h_B

theorem distance_sum_is_ten 
  (A B C : ℝ) 
  (h : ℝ) 
  (k : ℝ) 
  (h_pos : h = 4) 
  (ratio_condition : h_A = 4 * h_B)
  : angle_sum_distance C A B h k = 10 := 
  sorry

end NUMINAMATH_GPT_distance_sum_is_ten_l880_88018


namespace NUMINAMATH_GPT_age_difference_is_100_l880_88057

-- Definition of the ages
variables {X Y Z : ℕ}

-- Conditions from the problem statement
axiom age_condition1 : X + Y > Y + Z
axiom age_condition2 : Z = X - 100

-- Proof to show the difference is 100 years
theorem age_difference_is_100 : (X + Y) - (Y + Z) = 100 :=
by sorry

end NUMINAMATH_GPT_age_difference_is_100_l880_88057


namespace NUMINAMATH_GPT_smallest_divisible_by_1_to_10_l880_88034

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ n) ∧ (∀ k : ℕ, (∀ m : ℕ, (1 ≤ m ∧ m ≤ 10) → m ∣ k) → n ≤ k) ∧ n = 2520 :=
by
  sorry

end NUMINAMATH_GPT_smallest_divisible_by_1_to_10_l880_88034


namespace NUMINAMATH_GPT_milk_for_9_cookies_l880_88069

def quarts_to_pints (q : ℕ) : ℕ := q * 2

def milk_for_cookies (cookies : ℕ) (milk_in_quarts : ℕ) : ℕ :=
  quarts_to_pints milk_in_quarts * cookies / 18

theorem milk_for_9_cookies :
  milk_for_cookies 9 3 = 3 :=
by
  -- We define the conversion and proportional conditions explicitly here.
  unfold milk_for_cookies
  unfold quarts_to_pints
  sorry

end NUMINAMATH_GPT_milk_for_9_cookies_l880_88069


namespace NUMINAMATH_GPT_distinct_square_roots_l880_88011

theorem distinct_square_roots (m : ℝ) (h : 2 * m - 4 ≠ 3 * m - 1) : ∃ n : ℝ, (2 * m - 4) * (2 * m - 4) = n ∧ (3 * m - 1) * (3 * m - 1) = n ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_square_roots_l880_88011


namespace NUMINAMATH_GPT_solution_set_of_inequality_l880_88076

-- Define the conditions and theorem
theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) : (1 / x < x) ↔ ((-1 < x ∧ x < 0) ∨ (1 < x)) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l880_88076


namespace NUMINAMATH_GPT_parabola_axis_of_symmetry_is_x_eq_1_l880_88050

theorem parabola_axis_of_symmetry_is_x_eq_1 :
  ∀ x : ℝ, ∀ y : ℝ, y = -2 * (x - 1)^2 + 3 → (∀ c : ℝ, c = 1 → ∃ x1 x2 : ℝ, x1 = c ∧ x2 = c) := 
by
  sorry

end NUMINAMATH_GPT_parabola_axis_of_symmetry_is_x_eq_1_l880_88050


namespace NUMINAMATH_GPT_pipes_fill_tank_in_8_hours_l880_88066

theorem pipes_fill_tank_in_8_hours (A B C : ℝ) (hA : A = 1 / 56) (hB : B = 2 * A) (hC : C = 2 * B) :
  1 / (A + B + C) = 8 :=
by
  sorry

end NUMINAMATH_GPT_pipes_fill_tank_in_8_hours_l880_88066


namespace NUMINAMATH_GPT_find_x0_l880_88064

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (x : ℝ) : ℝ := 3 * x - 5

theorem find_x0 :
  (∃ x0 : ℝ, f (g x0) = 1) → (∃ x0 : ℝ, x0 = 4/3) :=
by
  sorry

end NUMINAMATH_GPT_find_x0_l880_88064


namespace NUMINAMATH_GPT_find_FC_l880_88090

theorem find_FC 
  (DC CB AD AB ED FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 12)
  (h3 : AB = (1/5) * AD)
  (h4 : ED = (2/3) * AD)
  (h5 : AD = (5/4) * 22)  -- Derived step from solution for full transparency
  (h6 : FC = (ED * (CB + AB)) / AD) : 
  FC = 35 / 3 := 
sorry

end NUMINAMATH_GPT_find_FC_l880_88090


namespace NUMINAMATH_GPT_ratio_of_ages_l880_88056

noncomputable def ratio_4th_to_3rd (age1 age2 age3 age4 age5 : ℕ) : ℚ :=
  age4 / age3

theorem ratio_of_ages
  (age1 age2 age3 age4 age5 : ℕ)
  (h1 : (age1 + age5) / 2 = 18)
  (h2 : age1 = 10)
  (h3 : age2 = age1 - 2)
  (h4 : age3 = age2 + 4)
  (h5 : age4 = age3 / 2)
  (h6 : age5 = age4 + 20) :
  ratio_4th_to_3rd age1 age2 age3 age4 age5 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l880_88056


namespace NUMINAMATH_GPT_molecular_weight_of_3_moles_l880_88055

namespace AscorbicAcid

def molecular_form : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 6)]

def atomic_weight : String → ℝ
| "C" => 12.01
| "H" => 1.008
| "O" => 16.00
| _ => 0

noncomputable def molecular_weight (molecular_form : List (String × ℕ)) : ℝ :=
molecular_form.foldr (λ (x : (String × ℕ)) acc => acc + (x.snd * atomic_weight x.fst)) 0

noncomputable def weight_of_3_moles (mw : ℝ) : ℝ := mw * 3

theorem molecular_weight_of_3_moles :
  weight_of_3_moles (molecular_weight molecular_form) = 528.372 :=
by
  sorry

end AscorbicAcid

end NUMINAMATH_GPT_molecular_weight_of_3_moles_l880_88055


namespace NUMINAMATH_GPT_width_of_metallic_sheet_l880_88068

-- Define the given conditions
def length_of_sheet : ℝ := 48
def side_of_square_cut : ℝ := 7
def volume_of_box : ℝ := 5236

-- Define the question as a Lean theorem
theorem width_of_metallic_sheet : ∃ (w : ℝ), w = 36 ∧
  volume_of_box = (length_of_sheet - 2 * side_of_square_cut) * (w - 2 * side_of_square_cut) * side_of_square_cut := by
  sorry

end NUMINAMATH_GPT_width_of_metallic_sheet_l880_88068


namespace NUMINAMATH_GPT_fraction_left_after_3_days_l880_88010

-- Defining work rates of A and B
def A_rate := 1 / 15
def B_rate := 1 / 20

-- Total work rate of A and B when working together
def combined_rate := A_rate + B_rate

-- Work completed by A and B in 3 days
def work_done := 3 * combined_rate

-- Fraction of work left
def fraction_work_left := 1 - work_done

-- Statement to prove:
theorem fraction_left_after_3_days : fraction_work_left = 13 / 20 :=
by
  have A_rate_def: A_rate = 1 / 15 := rfl
  have B_rate_def: B_rate = 1 / 20 := rfl
  have combined_rate_def: combined_rate = A_rate + B_rate := rfl
  have work_done_def: work_done = 3 * combined_rate := rfl
  have fraction_work_left_def: fraction_work_left = 1 - work_done := rfl
  sorry

end NUMINAMATH_GPT_fraction_left_after_3_days_l880_88010


namespace NUMINAMATH_GPT_samia_walked_distance_l880_88042

theorem samia_walked_distance :
  ∀ (total_distance cycling_speed walking_speed total_time : ℝ), 
  total_distance = 18 → 
  cycling_speed = 20 → 
  walking_speed = 4 → 
  total_time = 1 + 10 / 60 → 
  2 / 3 * total_distance / cycling_speed + 1 / 3 * total_distance / walking_speed = total_time → 
  1 / 3 * total_distance = 6 := 
by
  intros total_distance cycling_speed walking_speed total_time h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_samia_walked_distance_l880_88042
