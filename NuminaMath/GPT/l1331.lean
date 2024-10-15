import Mathlib

namespace NUMINAMATH_GPT_probability_two_red_cards_l1331_133178

theorem probability_two_red_cards : 
  let total_cards := 100;
  let red_cards := 50;
  let black_cards := 50;
  (red_cards / total_cards : ℝ) * ((red_cards - 1) / (total_cards - 1) : ℝ) = 49 / 198 := 
by
  sorry

end NUMINAMATH_GPT_probability_two_red_cards_l1331_133178


namespace NUMINAMATH_GPT_future_age_relation_l1331_133128

-- Conditions
def son_present_age : ℕ := 8
def father_present_age : ℕ := 4 * son_present_age

-- Theorem statement
theorem future_age_relation : ∃ x : ℕ, 32 + x = 3 * (8 + x) ↔ x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_future_age_relation_l1331_133128


namespace NUMINAMATH_GPT_test_average_score_l1331_133152

theorem test_average_score (A : ℝ) (h : 0.90 * A + 5 = 86) : A = 90 := 
by
  sorry

end NUMINAMATH_GPT_test_average_score_l1331_133152


namespace NUMINAMATH_GPT_four_digit_numbers_property_l1331_133106

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end NUMINAMATH_GPT_four_digit_numbers_property_l1331_133106


namespace NUMINAMATH_GPT_monotonic_function_a_range_l1331_133122

theorem monotonic_function_a_range :
  ∀ (f : ℝ → ℝ) (a : ℝ), 
  (f x = x^2 + (2 * a + 1) * x + 1) →
  (∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (f x ≤ f y ∨ f x ≥ f y)) ↔ 
  (a ∈ Set.Ici (-3/2) ∪ Set.Iic (-5/2)) := 
sorry

end NUMINAMATH_GPT_monotonic_function_a_range_l1331_133122


namespace NUMINAMATH_GPT_infinite_solutions_implies_d_eq_five_l1331_133195

theorem infinite_solutions_implies_d_eq_five (d : ℝ) :
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ (d = 5) := by
sorry

end NUMINAMATH_GPT_infinite_solutions_implies_d_eq_five_l1331_133195


namespace NUMINAMATH_GPT_curve_in_second_quadrant_l1331_133177

theorem curve_in_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0) ↔ (a > 2) :=
sorry

end NUMINAMATH_GPT_curve_in_second_quadrant_l1331_133177


namespace NUMINAMATH_GPT_find_p_l1331_133166

theorem find_p (m n p : ℝ) :
  m = (n / 7) - (2 / 5) →
  m + p = ((n + 21) / 7) - (2 / 5) →
  p = 3 := by
  sorry

end NUMINAMATH_GPT_find_p_l1331_133166


namespace NUMINAMATH_GPT_decreasing_interval_implies_range_of_a_l1331_133186

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem decreasing_interval_implies_range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, x ≤ y → y ≤ 4 → f a x ≥ f a y) : a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_implies_range_of_a_l1331_133186


namespace NUMINAMATH_GPT_third_angle_of_triangle_l1331_133116

theorem third_angle_of_triangle (a b : ℝ) (ha : a = 50) (hb : b = 60) : 
  ∃ (c : ℝ), a + b + c = 180 ∧ c = 70 :=
by
  sorry

end NUMINAMATH_GPT_third_angle_of_triangle_l1331_133116


namespace NUMINAMATH_GPT_percentage_increase_l1331_133164

theorem percentage_increase (employees_dec : ℝ) (employees_jan : ℝ) (inc : ℝ) (percentage : ℝ) :
  employees_dec = 470 →
  employees_jan = 408.7 →
  inc = employees_dec - employees_jan →
  percentage = (inc / employees_jan) * 100 →
  percentage = 15 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l1331_133164


namespace NUMINAMATH_GPT_exists_m_in_range_l1331_133127

theorem exists_m_in_range :
  ∃ m : ℝ, 0 ≤ m ∧ m < 1 ∧ ∀ x : ℕ, (x > m ∧ x < 2) ↔ (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_m_in_range_l1331_133127


namespace NUMINAMATH_GPT_system_solution_l1331_133161

theorem system_solution (x b y : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 4 * y = 3 * b) (h3 : x = 3) :
  b = -1 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_system_solution_l1331_133161


namespace NUMINAMATH_GPT_odot_property_l1331_133105

def odot (x y : ℤ) := 2 * x + y

theorem odot_property (a b : ℤ) (h : odot a (-6 * b) = 4) : odot (a - 5 * b) (a + b) = 6 :=
by
  sorry

end NUMINAMATH_GPT_odot_property_l1331_133105


namespace NUMINAMATH_GPT_Q_is_234_l1331_133113

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {z | ∃ x y : ℕ, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_is_234 : Q = {2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_Q_is_234_l1331_133113


namespace NUMINAMATH_GPT_sum_digits_3times_l1331_133102

-- Define the sum of digits function
noncomputable def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the 2006-th power of 2
noncomputable def power_2006 := 2 ^ 2006

-- State the theorem
theorem sum_digits_3times (n : ℕ) (h : n = power_2006) : 
  digit_sum (digit_sum (digit_sum n)) = 4 := by
  -- Add the proof steps here
  sorry

end NUMINAMATH_GPT_sum_digits_3times_l1331_133102


namespace NUMINAMATH_GPT_unique_number_not_in_range_l1331_133170

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_number_not_in_range (p q r s : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : r ≠ 0) (h₃ : s ≠ 0) 
  (h₄ : g p q r s 23 = 23) (h₅ : g p q r s 101 = 101) (h₆ : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  p / r = 62 :=
sorry

end NUMINAMATH_GPT_unique_number_not_in_range_l1331_133170


namespace NUMINAMATH_GPT_distance_from_dorm_to_city_l1331_133123

theorem distance_from_dorm_to_city (D : ℚ) (h1 : (1/3) * D = (1/3) * D) (h2 : (3/5) * D = (3/5) * D) (h3 : D - ((1 / 3) * D + (3 / 5) * D) = 2) :
  D = 30 := 
by sorry

end NUMINAMATH_GPT_distance_from_dorm_to_city_l1331_133123


namespace NUMINAMATH_GPT_no_solution_nat_x_satisfies_eq_l1331_133171

def sum_digits (x : ℕ) : ℕ := x.digits 10 |>.sum

theorem no_solution_nat_x_satisfies_eq (x : ℕ) :
  ¬ (x + sum_digits x + sum_digits (sum_digits x) = 2014) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_nat_x_satisfies_eq_l1331_133171


namespace NUMINAMATH_GPT_exists_integers_a_b_c_d_l1331_133138

-- Define the problem statement in Lean 4

theorem exists_integers_a_b_c_d (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
by
  sorry

end NUMINAMATH_GPT_exists_integers_a_b_c_d_l1331_133138


namespace NUMINAMATH_GPT_inequality_solution_l1331_133176

theorem inequality_solution 
  (x : ℝ) : 
  (x^2 / (x+2)^2 ≥ 0) ↔ x ≠ -2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1331_133176


namespace NUMINAMATH_GPT_brenda_cakes_l1331_133191

-- Definitions based on the given conditions
def cakes_per_day : ℕ := 20
def days : ℕ := 9
def total_cakes_baked : ℕ := cakes_per_day * days
def cakes_sold : ℕ := total_cakes_baked / 2
def cakes_left : ℕ := total_cakes_baked - cakes_sold

-- Formulate the theorem
theorem brenda_cakes : cakes_left = 90 :=
by {
  -- To skip the proof steps
  sorry
}

end NUMINAMATH_GPT_brenda_cakes_l1331_133191


namespace NUMINAMATH_GPT_digit_number_is_203_l1331_133103

theorem digit_number_is_203 {A B C : ℕ} (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) :
  100 * A + 10 * B + C = 203 :=
by
  sorry

end NUMINAMATH_GPT_digit_number_is_203_l1331_133103


namespace NUMINAMATH_GPT_exists_n_sum_three_digit_identical_digit_l1331_133101

theorem exists_n_sum_three_digit_identical_digit:
  ∃ (n : ℕ), (∃ (k : ℕ), (k ≥ 1 ∧ k ≤ 9) ∧ (n*(n+1)/2 = 111*k)) ∧ n = 36 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_exists_n_sum_three_digit_identical_digit_l1331_133101


namespace NUMINAMATH_GPT_fireworks_display_l1331_133145

def num_digits_year : ℕ := 4
def fireworks_per_digit : ℕ := 6
def regular_letters_phrase : ℕ := 12
def fireworks_per_regular_letter : ℕ := 5

def fireworks_H : ℕ := 8
def fireworks_E : ℕ := 7
def fireworks_L : ℕ := 6
def fireworks_O : ℕ := 9

def num_boxes : ℕ := 100
def fireworks_per_box : ℕ := 10

def total_fireworks : ℕ :=
  (num_digits_year * fireworks_per_digit) +
  (regular_letters_phrase * fireworks_per_regular_letter) +
  (fireworks_H + fireworks_E + 2 * fireworks_L + fireworks_O) + 
  (num_boxes * fireworks_per_box)

theorem fireworks_display : total_fireworks = 1120 := by
  sorry

end NUMINAMATH_GPT_fireworks_display_l1331_133145


namespace NUMINAMATH_GPT_yi_jianlian_shots_l1331_133109

theorem yi_jianlian_shots (x y : ℕ) 
  (h1 : x + y = 16 - 3) 
  (h2 : 2 * x + y = 28 - 3 * 3) : 
  x = 6 ∧ y = 7 := 
by 
  sorry

end NUMINAMATH_GPT_yi_jianlian_shots_l1331_133109


namespace NUMINAMATH_GPT_find_radius_and_diameter_l1331_133144

theorem find_radius_and_diameter (M N r d : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 15) : 
  (r = 30) ∧ (d = 60) := by
  sorry

end NUMINAMATH_GPT_find_radius_and_diameter_l1331_133144


namespace NUMINAMATH_GPT_caffeine_over_l1331_133118

section caffeine_problem

-- Definitions of the given conditions
def cups_of_coffee : Nat := 3
def cans_of_soda : Nat := 1
def cups_of_tea : Nat := 2

def caffeine_per_cup_coffee : Nat := 80
def caffeine_per_can_soda : Nat := 40
def caffeine_per_cup_tea : Nat := 50

def caffeine_goal : Nat := 200

-- Calculate the total caffeine consumption
def caffeine_from_coffee : Nat := cups_of_coffee * caffeine_per_cup_coffee
def caffeine_from_soda : Nat := cans_of_soda * caffeine_per_can_soda
def caffeine_from_tea : Nat := cups_of_tea * caffeine_per_cup_tea

def total_caffeine : Nat := caffeine_from_coffee + caffeine_from_soda + caffeine_from_tea

-- Calculate the caffeine amount over the goal
def caffeine_over_goal : Nat := total_caffeine - caffeine_goal

-- Theorem statement
theorem caffeine_over {total_caffeine caffeine_goal : Nat} (h : total_caffeine = 380) (g : caffeine_goal = 200) :
  caffeine_over_goal = 180 := by
  -- The proof goes here.
  sorry

end caffeine_problem

end NUMINAMATH_GPT_caffeine_over_l1331_133118


namespace NUMINAMATH_GPT_initial_members_in_family_c_l1331_133167

theorem initial_members_in_family_c 
  (a b d e f : ℕ)
  (ha : a = 7)
  (hb : b = 8)
  (hd : d = 13)
  (he : e = 6)
  (hf : f = 10)
  (average_after_moving : (a - 1) + (b - 1) + (d - 1) + (e - 1) + (f - 1) + (x : ℕ) - 1 = 48) :
  x = 10 := by
  sorry

end NUMINAMATH_GPT_initial_members_in_family_c_l1331_133167


namespace NUMINAMATH_GPT_not_integer_20_diff_l1331_133156

theorem not_integer_20_diff (a b : ℝ) (hne : a ≠ b) 
  (no_roots1 : ∀ x, x^2 + 20 * a * x + 10 * b ≠ 0) 
  (no_roots2 : ∀ x, x^2 + 20 * b * x + 10 * a ≠ 0) : 
  ¬ (∃ k : ℤ, 20 * (b - a) = k) :=
by
  sorry

end NUMINAMATH_GPT_not_integer_20_diff_l1331_133156


namespace NUMINAMATH_GPT_sandra_oranges_l1331_133187

theorem sandra_oranges (S E B: ℕ) (h1: E = 7 * S) (h2: E = 252) (h3: B = 12) : S / B = 3 := by
  sorry

end NUMINAMATH_GPT_sandra_oranges_l1331_133187


namespace NUMINAMATH_GPT_min_abs_sum_l1331_133135

theorem min_abs_sum (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^2 + b * c = 9) (h2 : b * c + d^2 = 9) (h3 : a * b + b * d = 0) (h4 : a * c + c * d = 0) :
  |a| + |b| + |c| + |d| = 8 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_l1331_133135


namespace NUMINAMATH_GPT_audrey_ratio_in_3_years_l1331_133165

-- Define the ages and the conditions
def Heracles_age : ℕ := 10
def Audrey_age := Heracles_age + 7
def Audrey_age_in_3_years := Audrey_age + 3

-- Statement: Prove that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1
theorem audrey_ratio_in_3_years : (Audrey_age_in_3_years / Heracles_age) = 2 := sorry

end NUMINAMATH_GPT_audrey_ratio_in_3_years_l1331_133165


namespace NUMINAMATH_GPT_cost_condition_shirt_costs_purchasing_plans_maximize_profit_l1331_133160

/-- Define the costs and prices of shirts A and B -/
def cost_A (m : ℝ) : ℝ := m
def cost_B (m : ℝ) : ℝ := m - 10
def price_A : ℝ := 260
def price_B : ℝ := 180

/-- Condition: total cost of 3 A shirts and 2 B shirts is 480 -/
theorem cost_condition (m : ℝ) : 3 * (cost_A m) + 2 * (cost_B m) = 480 := by
  sorry

/-- The cost of each A shirt is 100 and each B shirt is 90 -/
theorem shirt_costs : ∃ m, cost_A m = 100 ∧ cost_B m = 90 := by
  sorry

/-- Number of purchasing plans for at least $34,000 profit with 300 shirts and at most 110 A shirts -/
theorem purchasing_plans : ∃ x, 100 ≤ x ∧ x ≤ 110 ∧ 
  (260 * x + 180 * (300 - x) - 100 * x - 90 * (300 - x) ≥ 34000) := by
  sorry

/- Maximize profit given 60 < a < 80:
   - 60 < a < 70: 110 A shirts, 190 B shirts.
   - a = 70: any combination satisfying conditions.
   - 70 < a < 80: 100 A shirts, 200 B shirts. -/

theorem maximize_profit (a : ℝ) (ha : 60 < a ∧ a < 80) : 
  ∃ x, ((60 < a ∧ a < 70 ∧ x = 110 ∧ (300 - x) = 190) ∨ 
        (a = 70) ∨ 
        (70 < a ∧ a < 80 ∧ x = 100 ∧ (300 - x) = 200)) := by
  sorry

end NUMINAMATH_GPT_cost_condition_shirt_costs_purchasing_plans_maximize_profit_l1331_133160


namespace NUMINAMATH_GPT_verify_sum_l1331_133125

-- Definitions and conditions
def C : ℕ := 1
def D : ℕ := 2
def E : ℕ := 5

-- Base-6 addition representation
def is_valid_base_6_addition (a b c d : ℕ) : Prop :=
  (a + b) % 6 = c ∧ (a + b) / 6 = d

-- Given the addition problem:
def addition_problem : Prop :=
  is_valid_base_6_addition 2 5 C 0 ∧
  is_valid_base_6_addition 4 C E 0 ∧
  is_valid_base_6_addition D 2 4 0

-- Goal to prove
theorem verify_sum : addition_problem → C + D + E = 6 :=
by
  sorry

end NUMINAMATH_GPT_verify_sum_l1331_133125


namespace NUMINAMATH_GPT_pressure_on_trapezoidal_dam_l1331_133100

noncomputable def water_pressure_on_trapezoidal_dam (ρ g h a b : ℝ) : ℝ :=
  ρ * g * (h^2) * (2 * a + b) / 6

theorem pressure_on_trapezoidal_dam
  (ρ g h a b : ℝ) : water_pressure_on_trapezoidal_dam ρ g h a b = ρ * g * (h^2) * (2 * a + b) / 6 := by
  sorry

end NUMINAMATH_GPT_pressure_on_trapezoidal_dam_l1331_133100


namespace NUMINAMATH_GPT_sugar_needed_for_third_layer_l1331_133155

-- Let cups be the amount of sugar, and define the layers
def first_layer_sugar : ℕ := 2
def second_layer_sugar : ℕ := 2 * first_layer_sugar
def third_layer_sugar : ℕ := 3 * second_layer_sugar

-- The theorem we want to prove
theorem sugar_needed_for_third_layer : third_layer_sugar = 12 := by
  sorry

end NUMINAMATH_GPT_sugar_needed_for_third_layer_l1331_133155


namespace NUMINAMATH_GPT_range_x_minus_2y_l1331_133154

variable (x y : ℝ)

def cond1 : Prop := -1 ≤ x ∧ x < 2
def cond2 : Prop := 0 < y ∧ y ≤ 1

theorem range_x_minus_2y 
  (h1 : cond1 x) 
  (h2 : cond2 y) : 
  -3 ≤ x - 2 * y ∧ x - 2 * y < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_x_minus_2y_l1331_133154


namespace NUMINAMATH_GPT_evaluate_expression_l1331_133181

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end NUMINAMATH_GPT_evaluate_expression_l1331_133181


namespace NUMINAMATH_GPT_finish_time_is_1_10_PM_l1331_133174

-- Definitions of the problem conditions
def start_time := 9 * 60 -- 9:00 AM in minutes past midnight
def third_task_finish_time := 11 * 60 + 30 -- 11:30 AM in minutes past midnight
def num_tasks := 5
def tasks1_to_3_duration := third_task_finish_time - start_time
def one_task_duration := tasks1_to_3_duration / 3
def total_duration := one_task_duration * num_tasks

-- Statement to prove the final time when John finishes the fifth task
theorem finish_time_is_1_10_PM : 
  start_time + total_duration = 13 * 60 + 10 := 
by 
  sorry

end NUMINAMATH_GPT_finish_time_is_1_10_PM_l1331_133174


namespace NUMINAMATH_GPT_morgan_hula_hooping_time_l1331_133108

-- Definitions based on conditions
def nancy_can_hula_hoop : ℕ := 10
def casey_can_hula_hoop : ℕ := nancy_can_hula_hoop - 3
def morgan_can_hula_hoop : ℕ := 3 * casey_can_hula_hoop

-- Theorem statement to show the solution is correct
theorem morgan_hula_hooping_time : morgan_can_hula_hoop = 21 :=
by
  sorry

end NUMINAMATH_GPT_morgan_hula_hooping_time_l1331_133108


namespace NUMINAMATH_GPT_initial_depth_dug_l1331_133153

theorem initial_depth_dug :
  (∀ days : ℕ, 75 * 8 * days / D = 140 * 6 * days / 70) → D = 50 :=
by
  sorry

end NUMINAMATH_GPT_initial_depth_dug_l1331_133153


namespace NUMINAMATH_GPT_age_difference_problem_l1331_133134

theorem age_difference_problem 
    (minimum_age : ℕ := 25)
    (current_age_Jane : ℕ := 28)
    (years_ahead : ℕ := 6)
    (Dara_age_in_6_years : ℕ := (current_age_Jane + years_ahead) / 2):
    minimum_age - (Dara_age_in_6_years - years_ahead) = 14 :=
by
  -- all definition parts: minimum_age, current_age_Jane, years_ahead,
  -- Dara_age_in_6_years are present
  sorry

end NUMINAMATH_GPT_age_difference_problem_l1331_133134


namespace NUMINAMATH_GPT_mustard_bottles_total_l1331_133114

theorem mustard_bottles_total (b1 b2 b3 : ℝ) (h1 : b1 = 0.25) (h2 : b2 = 0.25) (h3 : b3 = 0.38) :
  b1 + b2 + b3 = 0.88 :=
by
  sorry

end NUMINAMATH_GPT_mustard_bottles_total_l1331_133114


namespace NUMINAMATH_GPT_candy_distribution_l1331_133126

-- Define the required parameters and conditions.
def num_distinct_candies : ℕ := 9
def num_bags : ℕ := 3

-- The result that we need to prove
theorem candy_distribution :
  (3 ^ num_distinct_candies) - 3 * (2 ^ (num_distinct_candies - 1) - 2) = 18921 := by
  sorry

end NUMINAMATH_GPT_candy_distribution_l1331_133126


namespace NUMINAMATH_GPT_shyam_weight_increase_l1331_133162

theorem shyam_weight_increase (x : ℝ) 
    (h1 : x > 0)
    (ratio : ∀ Ram Shyam : ℝ, (Ram / Shyam) = 7 / 5)
    (ram_increase : ∀ Ram : ℝ, Ram' = Ram + 0.1 * Ram)
    (total_weight_after : Ram' + Shyam' = 82.8)
    (total_weight_increase : 82.8 = 1.15 * total_weight) :
    (Shyam' - Shyam) / Shyam * 100 = 22 :=
by
  sorry

end NUMINAMATH_GPT_shyam_weight_increase_l1331_133162


namespace NUMINAMATH_GPT_taco_price_theorem_l1331_133158

noncomputable def price_hard_shell_taco_proof
  (H : ℤ)
  (price_soft : ℤ := 2)
  (num_hard_tacos_family : ℤ := 4)
  (num_soft_tacos_family : ℤ := 3)
  (num_additional_customers : ℤ := 10)
  (total_earnings : ℤ := 66)
  : Prop :=
  4 * H + 3 * price_soft + 10 * 2 * price_soft = total_earnings → H = 5

theorem taco_price_theorem : price_hard_shell_taco_proof 5 := 
by
  sorry

end NUMINAMATH_GPT_taco_price_theorem_l1331_133158


namespace NUMINAMATH_GPT_number_of_houses_in_block_l1331_133111

theorem number_of_houses_in_block (pieces_per_house pieces_per_block : ℕ) (h1 : pieces_per_house = 32) (h2 : pieces_per_block = 640) :
  pieces_per_block / pieces_per_house = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_houses_in_block_l1331_133111


namespace NUMINAMATH_GPT_ethanol_relationship_l1331_133159

variables (a b c x : ℝ)
def total_capacity := a + b + c = 300
def ethanol_content := x = 0.10 * a + 0.15 * b + 0.20 * c
def ethanol_bounds := 30 ≤ x ∧ x ≤ 60

theorem ethanol_relationship : total_capacity a b c → ethanol_bounds x → ethanol_content a b c x :=
by
  intros h_total h_bounds
  unfold total_capacity at h_total
  unfold ethanol_bounds at h_bounds
  unfold ethanol_content
  sorry

end NUMINAMATH_GPT_ethanol_relationship_l1331_133159


namespace NUMINAMATH_GPT_evaluate_expression_l1331_133168

theorem evaluate_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(2*x)) / (y^(2*y) * x^(2*x)) = (x / y)^(2 * (y - x)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1331_133168


namespace NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l1331_133175

theorem largest_n_for_factorable_polynomial : ∃ n, 
  (∀ A B : ℤ, (6 * B + A = n) → (A * B = 144)) ∧ 
  (∀ n', (∀ A B : ℤ, (6 * B + A = n') → (A * B = 144)) → n' ≤ n) ∧ 
  (n = 865) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_for_factorable_polynomial_l1331_133175


namespace NUMINAMATH_GPT_red_balls_in_total_color_of_158th_ball_l1331_133185

def totalBalls : Nat := 200
def redBallsPerCycle : Nat := 5
def whiteBallsPerCycle : Nat := 4
def blackBallsPerCycle : Nat := 3
def cycleLength : Nat := redBallsPerCycle + whiteBallsPerCycle + blackBallsPerCycle

theorem red_balls_in_total :
  (totalBalls / cycleLength) * redBallsPerCycle + min redBallsPerCycle (totalBalls % cycleLength) = 85 :=
by sorry

theorem color_of_158th_ball :
  let positionInCycle := (158 - 1) % cycleLength + 1
  positionInCycle ≤ redBallsPerCycle := by sorry

end NUMINAMATH_GPT_red_balls_in_total_color_of_158th_ball_l1331_133185


namespace NUMINAMATH_GPT_pens_sold_to_recover_investment_l1331_133146

-- Given the conditions
variables (P C : ℝ) (N : ℝ)
-- P is the total cost of 30 pens
-- C is the cost price of each pen
-- N is the number of pens sold to recover the initial investment

-- Stating the conditions
axiom h1 : P = 30 * C
axiom h2 : N * 1.5 * C = P

-- Proving that N = 20
theorem pens_sold_to_recover_investment (P C N : ℝ) (h1 : P = 30 * C) (h2 : N * 1.5 * C = P) : N = 20 :=
by
  sorry

end NUMINAMATH_GPT_pens_sold_to_recover_investment_l1331_133146


namespace NUMINAMATH_GPT_find_2a_minus_b_l1331_133143

-- Define conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := -5 * x + 7
def h (x : ℝ) (a b : ℝ) := f (g x) a b
def h_inv (x : ℝ) := x - 9

-- Statement to prove
theorem find_2a_minus_b (a b : ℝ) 
(h_eq : ∀ x, h x a b = a * (-5 * x + 7) + b)
(h_inv_eq : ∀ x, h_inv x = x - 9)
(h_hinv_eq : ∀ x, h (h_inv x) a b = x) :
  2 * a - b = -54 / 5 := sorry

end NUMINAMATH_GPT_find_2a_minus_b_l1331_133143


namespace NUMINAMATH_GPT_algorithm_contains_sequential_structure_l1331_133189

theorem algorithm_contains_sequential_structure :
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) ∧
  (∀ algorithm : Type, ∃ sel_struct : Prop, sel_struct ∨ ¬ sel_struct) ∧
  (∀ algorithm : Type, ∃ loop_struct : Prop, loop_struct) →
  (∀ algorithm : Type, ∃ seq_struct : Prop, seq_struct) := by
  sorry

end NUMINAMATH_GPT_algorithm_contains_sequential_structure_l1331_133189


namespace NUMINAMATH_GPT_tracy_customers_l1331_133199

theorem tracy_customers
  (total_customers : ℕ)
  (customers_bought_two_each : ℕ)
  (customers_bought_one_each : ℕ)
  (customers_bought_four_each : ℕ)
  (total_paintings_sold : ℕ)
  (h1 : total_customers = 20)
  (h2 : customers_bought_one_each = 12)
  (h3 : customers_bought_four_each = 4)
  (h4 : total_paintings_sold = 36)
  (h5 : 2 * customers_bought_two_each + customers_bought_one_each + 4 * customers_bought_four_each = total_paintings_sold) :
  customers_bought_two_each = 4 :=
by
  sorry

end NUMINAMATH_GPT_tracy_customers_l1331_133199


namespace NUMINAMATH_GPT_cube_net_count_l1331_133182

/-- A net of a cube is a two-dimensional arrangement of six squares.
    A regular tetrahedron has exactly 2 unique nets.
    For a cube, consider all possible ways in which the six faces can be arranged such that they 
    form a cube when properly folded. -/
theorem cube_net_count : cube_nets_count = 11 :=
sorry

end NUMINAMATH_GPT_cube_net_count_l1331_133182


namespace NUMINAMATH_GPT_savings_for_mother_l1331_133142

theorem savings_for_mother (Liam_oranges Claire_oranges: ℕ) (Liam_price_per_pair Claire_price_per_orange: ℝ) :
  Liam_oranges = 40 ∧ Liam_price_per_pair = 2.50 ∧
  Claire_oranges = 30 ∧ Claire_price_per_orange = 1.20 →
  ((Liam_oranges / 2) * Liam_price_per_pair + Claire_oranges * Claire_price_per_orange) = 86 :=
by
  intros
  sorry

end NUMINAMATH_GPT_savings_for_mother_l1331_133142


namespace NUMINAMATH_GPT_total_pools_l1331_133117

def patsPools (numAStores numPStores poolsA ratio : ℕ) : ℕ :=
  numAStores * poolsA + numPStores * (ratio * poolsA)

theorem total_pools : 
  patsPools 6 4 200 3 = 3600 := 
by 
  sorry

end NUMINAMATH_GPT_total_pools_l1331_133117


namespace NUMINAMATH_GPT_find_integers_satisfying_condition_l1331_133188

-- Define the inequality condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Prove that the set of integers satisfying the condition is {1, 2}
theorem find_integers_satisfying_condition :
  { x : ℤ | condition x } = {1, 2} := 
by {
  sorry
}

end NUMINAMATH_GPT_find_integers_satisfying_condition_l1331_133188


namespace NUMINAMATH_GPT_min_value_a_l1331_133119

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, |x + a| - |x + 1| ≤ 2 * a) → a ≥ 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_l1331_133119


namespace NUMINAMATH_GPT_set_intersection_and_polynomial_solution_l1331_133151

theorem set_intersection_and_polynomial_solution {a b : ℝ} :
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  (A ∩ B = {x | x < -3}) ∧ ((A ∪ B = {x | x < -2 ∨ x > 1}) →
    (a = 2 ∧ b = -4)) :=
by
  let A := {x : ℝ | x + 2 < 0}
  let B := {x : ℝ | (x + 3) * (x - 1) > 0}
  sorry

end NUMINAMATH_GPT_set_intersection_and_polynomial_solution_l1331_133151


namespace NUMINAMATH_GPT_maximum_value_inequality_l1331_133148

theorem maximum_value_inequality (x y : ℝ) : 
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 :=
sorry

end NUMINAMATH_GPT_maximum_value_inequality_l1331_133148


namespace NUMINAMATH_GPT_geometric_mean_condition_l1331_133150

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem geometric_mean_condition
  (h_arith : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) / 6 = (a 3 + a 4) / 2)
  (h_geom_pos : ∀ n, 0 < b n) :
  Real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) = Real.sqrt (b 3 * b 4) :=
sorry

end NUMINAMATH_GPT_geometric_mean_condition_l1331_133150


namespace NUMINAMATH_GPT_find_Y_value_l1331_133104

-- Define the conditions
def P : ℕ := 4020 / 4
def Q : ℕ := P * 2
def Y : ℤ := P - Q

-- State the theorem
theorem find_Y_value : Y = -1005 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_Y_value_l1331_133104


namespace NUMINAMATH_GPT_tangent_normal_lines_l1331_133193

theorem tangent_normal_lines :
  ∃ m_t b_t m_n b_n,
    (∀ x y, y = 1 / (1 + x^2) → y = m_t * x + b_t → 4 * x + 25 * y - 13 = 0) ∧
    (∀ x y, y = 1 / (1 + x^2) → y = m_n * x + b_n → 125 * x - 20 * y - 246 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_normal_lines_l1331_133193


namespace NUMINAMATH_GPT_seven_people_arrangement_l1331_133133

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def perm (n k : Nat) : Nat :=
  factorial n / factorial (n - k)

theorem seven_people_arrangement : 
  (perm 5 5) * (perm 6 2) = 3600 := by
sorry

end NUMINAMATH_GPT_seven_people_arrangement_l1331_133133


namespace NUMINAMATH_GPT_fresh_grapes_weight_l1331_133190

theorem fresh_grapes_weight (F D : ℝ) (h1 : D = 0.625) (h2 : 0.10 * F = 0.80 * D) : F = 5 := by
  -- Using premises h1 and h2, we aim to prove that F = 5
  sorry

end NUMINAMATH_GPT_fresh_grapes_weight_l1331_133190


namespace NUMINAMATH_GPT_rectangle_area_l1331_133169

theorem rectangle_area (length : ℝ) (width : ℝ) (increased_width : ℝ) (area : ℝ)
  (h1 : length = 12)
  (h2 : increased_width = width * 1.2)
  (h3 : increased_width = 12)
  (h4 : area = length * width) : 
  area = 120 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1331_133169


namespace NUMINAMATH_GPT_arithmetic_sequence_a15_l1331_133198

theorem arithmetic_sequence_a15 (a_n S_n : ℕ → ℝ) (a_9 : a_n 9 = 4) (S_15 : S_n 15 = 30) :
  let a_1 := (-12 : ℝ)
  let d := (2 : ℝ)
  a_n 15 = 16 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a15_l1331_133198


namespace NUMINAMATH_GPT_a_beats_b_by_4_rounds_l1331_133147

variable (T_a T_b : ℝ)
variable (race_duration : ℝ) -- duration of the 4-round race in minutes
variable (time_difference : ℝ) -- Time that a beats b by in the 4-round race

open Real

-- Given conditions
def conditions :=
  (T_a = 7.5) ∧                             -- a's time to complete one round
  (race_duration = T_a * 4 + 10) ∧          -- a beats b by 10 minutes in a 4-round race
  (time_difference = T_b - T_a)             -- The time difference per round is T_b - T_a

-- Mathematical proof statement
theorem a_beats_b_by_4_rounds
  (h : conditions T_a T_b race_duration time_difference) :
  10 / time_difference = 4 := by
  sorry

end NUMINAMATH_GPT_a_beats_b_by_4_rounds_l1331_133147


namespace NUMINAMATH_GPT_no_prime_divisible_by_57_l1331_133173

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. --/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Given that 57 is equal to 3 times 19.--/
theorem no_prime_divisible_by_57 : ∀ p : ℕ, is_prime p → ¬ (57 ∣ p) :=
by
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_57_l1331_133173


namespace NUMINAMATH_GPT_problem1_problem2_l1331_133110

-- Define that a quadratic is a root-multiplying equation if one root is twice the other
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 ≠ 0 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)

-- Problem 1: Prove that x^2 - 3x + 2 = 0 is a root-multiplying equation
theorem problem1 : is_root_multiplying 1 (-3) 2 :=
  sorry

-- Problem 2: Given ax^2 + bx - 6 = 0 is a root-multiplying equation with one root being 2, determine a and b
theorem problem2 (a b : ℝ) : is_root_multiplying a b (-6) → (∃ x1 x2 : ℝ, x1 = 2 ∧ x1 ≠ 0 ∧ a * x1^2 + b * x1 - 6 = 0 ∧ a * x2^2 + b * x2 - 6 = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)) →
( (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1331_133110


namespace NUMINAMATH_GPT_probability_of_different_colors_l1331_133136

noncomputable def total_chips := 6 + 5 + 4

noncomputable def prob_diff_color : ℚ :=
  let pr_blue := 6 / total_chips
  let pr_red := 5 / total_chips
  let pr_yellow := 4 / total_chips

  let pr_not_blue := (5 + 4) / total_chips
  let pr_not_red := (6 + 4) / total_chips
  let pr_not_yellow := (6 + 5) / total_chips

  pr_blue * pr_not_blue + pr_red * pr_not_red + pr_yellow * pr_not_yellow

theorem probability_of_different_colors :
  prob_diff_color = 148 / 225 :=
sorry

end NUMINAMATH_GPT_probability_of_different_colors_l1331_133136


namespace NUMINAMATH_GPT_solve_for_a_l1331_133194

noncomputable def area_of_triangle (b c : ℝ) : ℝ :=
  1 / 2 * b * c * Real.sin (Real.pi / 3)

theorem solve_for_a (a b c : ℝ) (hA : 60 = 60) 
  (h_area : area_of_triangle b c = 3 * Real.sqrt 3 / 2)
  (h_sum_bc : b + c = 3 * Real.sqrt 3) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1331_133194


namespace NUMINAMATH_GPT_trigonometry_expression_zero_l1331_133196

variable {r : ℝ} {A B C : ℝ}
variable (a b c : ℝ) (sinA sinB sinC : ℝ)

-- The conditions from the problem
axiom Law_of_Sines_a : a = 2 * r * sinA
axiom Law_of_Sines_b : b = 2 * r * sinB
axiom Law_of_Sines_c : c = 2 * r * sinC

-- The theorem statement
theorem trigonometry_expression_zero :
  a * (sinC - sinB) + b * (sinA - sinC) + c * (sinB - sinA) = 0 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_trigonometry_expression_zero_l1331_133196


namespace NUMINAMATH_GPT_interesting_quadruples_count_l1331_133131

/-- Definition of interesting ordered quadruples (a, b, c, d) where 1 ≤ a < b < c < d ≤ 15 and a + b > c + d --/
def is_interesting_quadruple (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 15 ∧ a + b > c + d

/-- The number of interesting ordered quadruples (a, b, c, d) is 455 --/
theorem interesting_quadruples_count : 
  (∃ (s : Finset (ℕ × ℕ × ℕ × ℕ)), 
    s.card = 455 ∧ ∀ (a b c d : ℕ), 
    ((a, b, c, d) ∈ s ↔ is_interesting_quadruple a b c d)) :=
sorry

end NUMINAMATH_GPT_interesting_quadruples_count_l1331_133131


namespace NUMINAMATH_GPT_initial_percentage_of_water_l1331_133140

/-
Initial conditions:
- Let W be the initial percentage of water in 10 liters of milk.
- The mixture should become 2% water after adding 15 liters of pure milk to it.
-/

theorem initial_percentage_of_water (W : ℚ) 
  (H1 : 0 < W) (H2 : W < 100) 
  (H3 : (10 * (100 - W) / 100 + 15) / 25 = 0.98) : 
  W = 5 := 
sorry

end NUMINAMATH_GPT_initial_percentage_of_water_l1331_133140


namespace NUMINAMATH_GPT_trucks_sold_l1331_133112

-- Definitions for conditions
def cars_and_trucks_total (T C : Nat) : Prop :=
  T + C = 69

def cars_more_than_trucks (T C : Nat) : Prop :=
  C = T + 27

-- Theorem statement
theorem trucks_sold (T C : Nat) (h1 : cars_and_trucks_total T C) (h2 : cars_more_than_trucks T C) : T = 21 :=
by
  -- This will be replaced by the proof
  sorry

end NUMINAMATH_GPT_trucks_sold_l1331_133112


namespace NUMINAMATH_GPT_find_f_neg3_l1331_133192

theorem find_f_neg3 : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 5 * f (1 / x) + 3 * f x / x = 2 * x^2) ∧ f (-3) = 14029 / 72) :=
sorry

end NUMINAMATH_GPT_find_f_neg3_l1331_133192


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1331_133184

theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 9 = 1) → (y = 3/2 * x ∨ y = -3/2 * x) :=
by
  intro x y h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1331_133184


namespace NUMINAMATH_GPT_optimal_direction_l1331_133197

-- Define the conditions as hypotheses
variables (a : ℝ) (V_first V_second : ℝ) (d : ℝ)
variable (speed_rel : V_first = 2 * V_second)
variable (dist : d = a)

-- Create a theorem statement for the problem
theorem optimal_direction (H : d = a) (vel_rel : V_first = 2 * V_second) : true := 
  sorry

end NUMINAMATH_GPT_optimal_direction_l1331_133197


namespace NUMINAMATH_GPT_tables_needed_for_luncheon_l1331_133139

theorem tables_needed_for_luncheon (invited attending remaining tables_needed : ℕ) (H1 : invited = 24) (H2 : remaining = 10) (H3 : attending = invited - remaining) (H4 : tables_needed = attending / 7) : tables_needed = 2 :=
by
  sorry

end NUMINAMATH_GPT_tables_needed_for_luncheon_l1331_133139


namespace NUMINAMATH_GPT_xy_value_l1331_133163

theorem xy_value (x y : ℝ) (h : |x - 5| + |y + 3| = 0) : x * y = -15 := by
  sorry

end NUMINAMATH_GPT_xy_value_l1331_133163


namespace NUMINAMATH_GPT_trigonometric_identity_l1331_133179

theorem trigonometric_identity :
  Real.cos (70 * Real.pi / 180) * Real.sin (80 * Real.pi / 180) +
  Real.cos (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1331_133179


namespace NUMINAMATH_GPT_utility_bills_l1331_133183

-- Definitions for the conditions
def four_hundred := 4 * 100
def five_fifty := 5 * 50
def seven_twenty := 7 * 20
def eight_ten := 8 * 10
def total := four_hundred + five_fifty + seven_twenty + eight_ten

-- Lean statement for the proof problem
theorem utility_bills : total = 870 :=
by
  -- inserting skip proof placeholder
  sorry

end NUMINAMATH_GPT_utility_bills_l1331_133183


namespace NUMINAMATH_GPT_plane_equation_l1331_133129

theorem plane_equation (p q r : ℝ × ℝ × ℝ)
  (h₁ : p = (2, -1, 3))
  (h₂ : q = (0, -1, 5))
  (h₃ : r = (-1, -3, 4)) :
  ∃ A B C D : ℤ, A = 1 ∧ B = 2 ∧ C = -1 ∧ D = 3 ∧
               A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
               ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔
                             (x, y, z) = p ∨ (x, y, z) = q ∨ (x, y, z) = r :=
by
  sorry

end NUMINAMATH_GPT_plane_equation_l1331_133129


namespace NUMINAMATH_GPT_fraction_identity_l1331_133137

variables {a b : ℝ}

theorem fraction_identity (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) :=
by sorry

end NUMINAMATH_GPT_fraction_identity_l1331_133137


namespace NUMINAMATH_GPT_find_roots_l1331_133115

theorem find_roots (x : ℝ) (h : 21 / (x^2 - 9) - 3 / (x - 3) = 1) : x = -3 ∨ x = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_roots_l1331_133115


namespace NUMINAMATH_GPT_smallest_x_inequality_l1331_133132

theorem smallest_x_inequality : ∃ x : ℝ, (x^2 - 8 * x + 15 ≤ 0) ∧ (∀ y : ℝ, (y^2 - 8 * y + 15 ≤ 0) → (3 ≤ y)) ∧ x = 3 := 
sorry

end NUMINAMATH_GPT_smallest_x_inequality_l1331_133132


namespace NUMINAMATH_GPT_given_equation_roots_sum_cubes_l1331_133180

theorem given_equation_roots_sum_cubes (r s t : ℝ) 
    (h1 : 6 * r ^ 3 + 1506 * r + 3009 = 0)
    (h2 : 6 * s ^ 3 + 1506 * s + 3009 = 0)
    (h3 : 6 * t ^ 3 + 1506 * t + 3009 = 0)
    (sum_roots : r + s + t = 0) :
    (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1504.5 := 
by 
  -- proof omitted
  sorry

end NUMINAMATH_GPT_given_equation_roots_sum_cubes_l1331_133180


namespace NUMINAMATH_GPT_find_number_l1331_133130

theorem find_number (x : ℤ) (h : 5 + x * 5 = 15) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1331_133130


namespace NUMINAMATH_GPT_no_intersection_l1331_133141

def f₁ (x : ℝ) : ℝ := abs (3 * x + 6)
def f₂ (x : ℝ) : ℝ := -abs (4 * x - 1)

theorem no_intersection : ∀ x, f₁ x ≠ f₂ x :=
by
  sorry

end NUMINAMATH_GPT_no_intersection_l1331_133141


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l1331_133107

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℤ) (a1 a3 a4 : ℤ) (d : ℤ)
  (h1 : ∀ n, a (n + 1) = a n + d)
  (h2 : d = 2)
  (h3 : a1 = a 1)
  (h4 : a3 = a 3)
  (h5 : a4 = a 4)
  (h6 : a3^2 = a1 * a4) :
  a 6 = 2 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_subsequence_l1331_133107


namespace NUMINAMATH_GPT_abs_abc_eq_one_l1331_133124

variable (a b c : ℝ)

-- Conditions
axiom distinct_nonzero : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)
axiom condition : a^2 + 1/(b^2) = b^2 + 1/(c^2) ∧ b^2 + 1/(c^2) = c^2 + 1/(a^2)

theorem abs_abc_eq_one : |a * b * c| = 1 :=
by
  sorry

end NUMINAMATH_GPT_abs_abc_eq_one_l1331_133124


namespace NUMINAMATH_GPT_range_of_x_l1331_133172

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x (x : ℝ) : 
  (∃ y z : ℝ, y = 2 * x - 1 ∧ f x > f y ∧ x > 1 / 3 ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_range_of_x_l1331_133172


namespace NUMINAMATH_GPT_rectangle_area_l1331_133120

-- Definitions based on the conditions
def radius := 6
def diameter := 2 * radius
def width := diameter
def length := 3 * width

-- Statement of the theorem
theorem rectangle_area : (width * length = 432) := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1331_133120


namespace NUMINAMATH_GPT_xiaoming_interview_pass_probability_l1331_133121

theorem xiaoming_interview_pass_probability :
  let p_correct := 0.7
  let p_fail_per_attempt := 1 - p_correct
  let p_fail_all_attempts := p_fail_per_attempt ^ 3
  let p_pass_interview := 1 - p_fail_all_attempts
  p_pass_interview = 0.973 := by
    let p_correct := 0.7
    let p_fail_per_attempt := 1 - p_correct
    let p_fail_all_attempts := p_fail_per_attempt ^ 3
    let p_pass_interview := 1 - p_fail_all_attempts
    sorry

end NUMINAMATH_GPT_xiaoming_interview_pass_probability_l1331_133121


namespace NUMINAMATH_GPT_neg_prop_p_l1331_133149

def prop_p (x : ℝ) : Prop := x ≥ 0 → Real.log (x^2 + 1) ≥ 0

theorem neg_prop_p : (¬ (∀ x ≥ 0, Real.log (x^2 + 1) ≥ 0)) ↔ (∃ x ≥ 0, Real.log (x^2 + 1) < 0) := by
  sorry

end NUMINAMATH_GPT_neg_prop_p_l1331_133149


namespace NUMINAMATH_GPT_sum_of_arithmetic_progression_l1331_133157

theorem sum_of_arithmetic_progression :
  let a := 30
  let d := -3
  let n := 20
  let S_n := n / 2 * (2 * a + (n - 1) * d)
  S_n = 30 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_progression_l1331_133157
