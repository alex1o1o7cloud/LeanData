import Mathlib

namespace inequality_solution_set_l876_87641

theorem inequality_solution_set : 
  { x : ℝ | (1 - x) * (x + 1) ≤ 0 ∧ x ≠ -1 } = { x : ℝ | x < -1 ∨ x ≥ 1 } :=
sorry

end inequality_solution_set_l876_87641


namespace gun_can_hit_l876_87676

-- Define the constants
variables (v g : ℝ)

-- Define the coordinates in the first quadrant
variables (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0)

-- Prove the condition for a point (x, y) to be in the region that can be hit by the gun
theorem gun_can_hit (hv : v > 0) (hg : g > 0) :
  y ≤ (v^2 / (2 * g)) - (g * x^2 / (2 * v^2)) :=
sorry

end gun_can_hit_l876_87676


namespace polynomial_coeff_sum_l876_87661

theorem polynomial_coeff_sum (a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x: ℝ, (x - 1) ^ 4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_4 - a_3 + a_2 - a_1 + a_0 = 16 :=
by
  intro h
  sorry

end polynomial_coeff_sum_l876_87661


namespace A_and_B_finish_together_in_20_days_l876_87665

noncomputable def W_B : ℝ := 1 / 30

noncomputable def W_A : ℝ := 1 / 2 * W_B

noncomputable def W_A_plus_B : ℝ := W_A + W_B

theorem A_and_B_finish_together_in_20_days :
  (1 / W_A_plus_B) = 20 :=
by
  sorry

end A_and_B_finish_together_in_20_days_l876_87665


namespace proof_mod_55_l876_87688

theorem proof_mod_55 (M : ℕ) (h1 : M % 5 = 3) (h2 : M % 11 = 9) : M % 55 = 53 := 
  sorry

end proof_mod_55_l876_87688


namespace petya_friends_l876_87664

theorem petya_friends (x : ℕ) (s : ℕ) : 
  (5 * x + 8 = s) → 
  (6 * x - 11 = s) → 
  x = 19 :=
by
  intro h1 h2
  -- We would provide the logic to prove here, but the statement is enough
  sorry

end petya_friends_l876_87664


namespace simplify_product_l876_87630

theorem simplify_product (a : ℝ) : (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4) = 120 * a^10 := by
  sorry

end simplify_product_l876_87630


namespace Tony_age_at_end_of_period_l876_87678

-- Definitions based on the conditions in a):
def hours_per_day := 2
def days_worked := 60
def total_earnings := 1140
def earnings_per_hour (age : ℕ) := age

-- The main property we need to prove: Tony's age at the end of the period is 12 years old
theorem Tony_age_at_end_of_period : ∃ age : ℕ, (2 * age * days_worked = total_earnings) ∧ age = 12 :=
by
  sorry

end Tony_age_at_end_of_period_l876_87678


namespace Trent_tears_l876_87675

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end Trent_tears_l876_87675


namespace maria_tom_weather_probability_l876_87654

noncomputable def probability_exactly_two_clear_days (p : ℝ) (n : ℕ) : ℝ :=
  (n.choose 2) * (p ^ (n - 2)) * ((1 - p) ^ 2)

theorem maria_tom_weather_probability :
  probability_exactly_two_clear_days 0.6 5 = 1080 / 3125 :=
by
  sorry

end maria_tom_weather_probability_l876_87654


namespace fraction_passengers_from_asia_l876_87668

theorem fraction_passengers_from_asia (P : ℕ)
  (hP : P = 108)
  (frac_NA : ℚ) (frac_EU : ℚ) (frac_AF : ℚ)
  (Other_continents : ℕ)
  (h_frac_NA : frac_NA = 1/12)
  (h_frac_EU : frac_EU = 1/4)
  (h_frac_AF : frac_AF = 1/9)
  (h_Other_continents : Other_continents = 42) :
  (P * (1 - (frac_NA + frac_EU + frac_AF)) - Other_continents) / P = 1/6 :=
by
  sorry

end fraction_passengers_from_asia_l876_87668


namespace angelina_speed_from_grocery_to_gym_l876_87604

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym_l876_87604


namespace remainder_is_four_l876_87653

def least_number : Nat := 174

theorem remainder_is_four (n : Nat) (m₁ m₂ : Nat) (h₁ : n = least_number / m₁ * m₁ + 4) 
(h₂ : n = least_number / m₂ * m₂ + 4) (h₃ : m₁ = 34) (h₄ : m₂ = 5) : 
  n % m₁ = 4 ∧ n % m₂ = 4 := 
by
  sorry

end remainder_is_four_l876_87653


namespace probability_of_valid_number_l876_87652

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def has_distinct_digits (n : ℕ) : Prop :=
  ∀ (i j : ℕ), i ≠ j → (n % (10^i) / 10^(i-1)) ≠ (n % (10^j) / 10^(j-1))

def digits_in_range (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def valid_number (n : ℕ) : Prop :=
  is_even n ∧ has_distinct_digits n ∧ digits_in_range n

noncomputable def count_valid_numbers : ℕ :=
  2296

noncomputable def total_numbers : ℕ :=
  9000

theorem probability_of_valid_number :
  (count_valid_numbers : ℚ) / total_numbers = 574 / 2250 :=
by sorry

end probability_of_valid_number_l876_87652


namespace maximum_k_inequality_l876_87660

open Real

noncomputable def inequality_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : Prop :=
  (x / sqrt (y + z)) + (y / sqrt (z + x)) + (z / sqrt (x + y)) ≥ sqrt (3 / 2) * sqrt (x + y + z)
 
-- This is the theorem statement
theorem maximum_k_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  inequality_problem x y z h1 h2 h3 :=
  sorry

end maximum_k_inequality_l876_87660


namespace average_speed_uphill_l876_87698

theorem average_speed_uphill (d : ℝ) (v : ℝ) :
  (2 * d) / ((d / v) + (d / 100)) = 9.523809523809524 → v = 5 :=
by
  intro h1
  sorry

end average_speed_uphill_l876_87698


namespace apples_per_case_l876_87621

theorem apples_per_case (total_apples : ℕ) (number_of_cases : ℕ) (h1 : total_apples = 1080) (h2 : number_of_cases = 90) : total_apples / number_of_cases = 12 := by
  sorry

end apples_per_case_l876_87621


namespace building_height_l876_87683

theorem building_height
  (num_stories_1 : ℕ)
  (height_story_1 : ℕ)
  (num_stories_2 : ℕ)
  (height_story_2 : ℕ)
  (h1 : num_stories_1 = 10)
  (h2 : height_story_1 = 12)
  (h3 : num_stories_2 = 10)
  (h4 : height_story_2 = 15)
  :
  num_stories_1 * height_story_1 + num_stories_2 * height_story_2 = 270 :=
by
  sorry

end building_height_l876_87683


namespace find_g_3_l876_87623

theorem find_g_3 (p q r : ℝ) (g : ℝ → ℝ) (h1 : g x = p * x^7 + q * x^3 + r * x + 7) (h2 : g (-3) = -11) (h3 : ∀ x, g (x) + g (-x) = 14) : g 3 = 25 :=
by 
  sorry

end find_g_3_l876_87623


namespace min_value_of_function_l876_87663

noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

theorem min_value_of_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  f x y ≥ 1 / 4 :=
sorry

end min_value_of_function_l876_87663


namespace num_readers_sci_fiction_l876_87655

theorem num_readers_sci_fiction (T L B S: ℕ) (hT: T = 250) (hL: L = 88) (hB: B = 18) (hTotal: T = S + L - B) : 
  S = 180 := 
by 
  sorry

end num_readers_sci_fiction_l876_87655


namespace line_within_plane_l876_87634

variable (a : Set Point) (α : Set Point)

theorem line_within_plane : a ⊆ α :=
by
  sorry

end line_within_plane_l876_87634


namespace bank_balance_after_five_years_l876_87679

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem bank_balance_after_five_years :
  let P0 := 5600
  let r1 := 0.03
  let r2 := 0.035
  let r3 := 0.04
  let r4 := 0.045
  let r5 := 0.05
  let D := 2000
  let A1 := compoundInterest P0 r1 1 1
  let A2 := compoundInterest A1 r2 1 1
  let A3 := compoundInterest (A2 + D) r3 1 1
  let A4 := compoundInterest A3 r4 1 1
  let A5 := compoundInterest A4 r5 1 1
  A5 = 9094.2 := by
  sorry

end bank_balance_after_five_years_l876_87679


namespace scientific_notation_correct_l876_87647

def million : ℝ := 10^6
def num : ℝ := 1.06
def num_in_million : ℝ := num * million
def scientific_notation : ℝ := 1.06 * 10^6

theorem scientific_notation_correct : num_in_million = scientific_notation :=
by 
  -- The proof is skipped, indicated by sorry
  sorry

end scientific_notation_correct_l876_87647


namespace solve_for_x_l876_87632

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - y = 7)
  (h2 : x + 3 * y = 7) :
  x = 2.8 :=
by
  sorry

end solve_for_x_l876_87632


namespace express_y_in_terms_of_x_l876_87602

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 := 
by { sorry }

end express_y_in_terms_of_x_l876_87602


namespace max_value_a_l876_87650

theorem max_value_a (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + y = 1) : 
  ∃ a, a = 16 ∧ (∀ x y, (x > 0 → y > 0 → x + y = 1 → a ≤ (1/x) + (9/y))) :=
by 
  use 16
  sorry

end max_value_a_l876_87650


namespace evaluate_expression_l876_87692

theorem evaluate_expression : 6 - 5 * (9 - 2^3) * 3 = -9 := by
  sorry

end evaluate_expression_l876_87692


namespace simplify_expression_l876_87624

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end simplify_expression_l876_87624


namespace lake_coverage_day_17_l876_87686

-- Define the state of lake coverage as a function of day
def lake_coverage (day : ℕ) : ℝ :=
  if day ≤ 20 then 2 ^ (day - 20) else 0

-- Prove that on day 17, the lake was covered by 12.5% algae
theorem lake_coverage_day_17 : lake_coverage 17 = 0.125 :=
by
  sorry

end lake_coverage_day_17_l876_87686


namespace people_per_seat_l876_87618

def ferris_wheel_seats : ℕ := 4
def total_people_riding : ℕ := 20

theorem people_per_seat : total_people_riding / ferris_wheel_seats = 5 := by
  sorry

end people_per_seat_l876_87618


namespace min_radius_for_area_l876_87669

theorem min_radius_for_area (r : ℝ) (π : ℝ) (A : ℝ) (h1 : A = 314) (h2 : A = π * r^2) : r ≥ 10 :=
by
  sorry

end min_radius_for_area_l876_87669


namespace length_EQ_l876_87614

-- Define the square EFGH with side length 8
def square_EFGH (a : ℝ) (b : ℝ): Prop := a = 8 ∧ b = 8

-- Define the rectangle IJKL with IL = 12 and JK = 8
def rectangle_IJKL (l : ℝ) (w : ℝ): Prop := l = 12 ∧ w = 8

-- Define the perpendicularity of EH and IJ
def perpendicular_EH_IJ : Prop := true

-- Define the shaded area condition
def shaded_area_condition (area_IJKL : ℝ) (shaded_area : ℝ): Prop :=
  shaded_area = (1/3) * area_IJKL

-- Theorem to prove
theorem length_EQ (a b l w area_IJKL shaded_area EH HG HQ EQ : ℝ):
  square_EFGH a b →
  rectangle_IJKL l w →
  perpendicular_EH_IJ →
  shaded_area_condition area_IJKL shaded_area →
  HQ * HG = shaded_area →
  EQ = EH - HQ →
  EQ = 4 := by
  intros hSquare hRectangle hPerpendicular hShadedArea hHQHG hEQ
  sorry

end length_EQ_l876_87614


namespace find_a_l876_87615

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

-- Define the derivative of function f with respect to x
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 8 * x + 3

-- Define the condition for the problem
def condition (a : ℝ) : Prop := f' a 1 = 2

-- The statement to be proved
theorem find_a (a : ℝ) (h : condition a) : a = -3 :=
by {
  -- Proof is omitted
  sorry
}

end find_a_l876_87615


namespace notebooks_last_days_l876_87603

-- Given conditions
def n := 5
def p := 40
def u := 4

-- Derived conditions
def total_pages := n * p
def days := total_pages / u

-- The theorem statement
theorem notebooks_last_days : days = 50 := sorry

end notebooks_last_days_l876_87603


namespace boxes_total_is_correct_l876_87605

def initial_boxes : ℕ := 7
def additional_boxes_per_box : ℕ := 7
def final_non_empty_boxes : ℕ := 10
def total_boxes := 77

theorem boxes_total_is_correct
  (h1 : initial_boxes = 7)
  (h2 : additional_boxes_per_box = 7)
  (h3 : final_non_empty_boxes = 10)
  : total_boxes = 77 :=
by
  -- Proof goes here
  sorry

end boxes_total_is_correct_l876_87605


namespace fraction_of_garden_occupied_by_flowerbeds_is_correct_l876_87651

noncomputable def garden_fraction_occupied : ℚ :=
  let garden_length := 28
  let garden_shorter_length := 18
  let triangle_leg := (garden_length - garden_shorter_length) / 2
  let triangle_area := 1 / 2 * triangle_leg^2
  let flowerbeds_area := 2 * triangle_area
  let garden_width : ℚ := 5  -- Assuming the height of the trapezoid as part of the garden rest
  let garden_area := garden_length * garden_width
  flowerbeds_area / garden_area

theorem fraction_of_garden_occupied_by_flowerbeds_is_correct :
  garden_fraction_occupied = 5 / 28 := by
  sorry

end fraction_of_garden_occupied_by_flowerbeds_is_correct_l876_87651


namespace inequality_problem_l876_87628

theorem inequality_problem
  (a b c d e f : ℝ)
  (h_cond : b^2 ≥ a^2 + c^2) :
  (a * f - c * d)^2 ≤ (a * e - b * d)^2 + (b * f - c * e)^2 :=
by
  sorry

end inequality_problem_l876_87628


namespace total_gymnasts_l876_87612

theorem total_gymnasts (n : ℕ) : 
  (∃ (t : ℕ) (c : t = 4) (h : n * (n-1) / 2 + 4 * 6 = 595), n = 34) :=
by {
  -- skipping the detailed proof here, just ensuring the problem is stated as a theorem
  sorry
}

end total_gymnasts_l876_87612


namespace base_of_isosceles_triangle_l876_87677

theorem base_of_isosceles_triangle (b : ℝ) (h1 : 7 + 7 + b = 22) : b = 8 :=
by {
  sorry
}

end base_of_isosceles_triangle_l876_87677


namespace yanni_money_left_in_cents_l876_87690

-- Conditions
def initial_money : ℝ := 0.85
def money_from_mother : ℝ := 0.40
def money_found : ℝ := 0.50
def cost_per_toy : ℝ := 1.60
def number_of_toys : ℕ := 3
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Prove
theorem yanni_money_left_in_cents : 
  (initial_money + money_from_mother + money_found) * 100 = 175 :=
by
  sorry

end yanni_money_left_in_cents_l876_87690


namespace smallest_four_digit_div_by_53_l876_87662

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l876_87662


namespace no_solution_for_inequalities_l876_87656

theorem no_solution_for_inequalities :
  ¬ ∃ (x y : ℝ), 4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1 :=
by
  sorry

end no_solution_for_inequalities_l876_87656


namespace union_complements_eq_l876_87680

-- Definitions as per conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define complements
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof statement
theorem union_complements_eq :
  (C_UA ∪ C_UB) = {0, 1, 4} :=
by
  sorry

end union_complements_eq_l876_87680


namespace range_of_m_l876_87636

theorem range_of_m {m : ℝ} :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end range_of_m_l876_87636


namespace angle_of_inclination_l876_87667

theorem angle_of_inclination (α : ℝ) (h: 0 ≤ α ∧ α < 180) (slope_eq : Real.tan (Real.pi * α / 180) = Real.sqrt 3) :
  α = 60 :=
sorry

end angle_of_inclination_l876_87667


namespace equation_of_parallel_line_l876_87696

-- Definitions for conditions from the problem
def point_A : ℝ × ℝ := (3, 2)
def line_eq (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def parallel_slope : ℝ := -4

-- Proof problem statement
theorem equation_of_parallel_line (x y : ℝ) :
  (∃ (m b : ℝ), m = parallel_slope ∧ b = 2 + 4 * 3 ∧ y = m * (x - 3) + b) →
  4 * x + y - 14 = 0 :=
sorry

end equation_of_parallel_line_l876_87696


namespace find_x_condition_l876_87657

theorem find_x_condition :
  ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  intros x h
  have num_zero : x^2 - 1 = 0 := by
    -- Proof that the numerator is zero
    sorry
  have denom_nonzero : x ≠ -1 := by
    -- Proof that the denominator is non-zero
    sorry
  have x_solves : x = 1 := by
    -- Final proof to show x = 1
    sorry
  exact x_solves

end find_x_condition_l876_87657


namespace hexagon_area_l876_87633

theorem hexagon_area (A C : ℝ × ℝ) 
  (hA : A = (0, 0)) 
  (hC : C = (2 * Real.sqrt 3, 2)) : 
  6 * Real.sqrt 3 = 6 * Real.sqrt 3 := 
by sorry

end hexagon_area_l876_87633


namespace expression_meaningful_if_not_three_l876_87699

-- Definition of meaningful expression
def meaningful_expr (x : ℝ) : Prop := (x ≠ 3)

theorem expression_meaningful_if_not_three (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) ↔ meaningful_expr x := by
  sorry

end expression_meaningful_if_not_three_l876_87699


namespace cartons_in_a_case_l876_87625

-- Definitions based on problem conditions
def numberOfBoxesInCarton (c : ℕ) (b : ℕ) : ℕ := c * b * 300
def paperClipsInTwoCases (c : ℕ) (b : ℕ) : ℕ := 2 * numberOfBoxesInCarton c b

-- Condition from problem statement: paperClipsInTwoCases c b = 600
theorem cartons_in_a_case 
  (c b : ℕ) 
  (h1 : paperClipsInTwoCases c b = 600) 
  (h2 : b ≥ 1) : 
  c = 1 := 
by
  -- Proof will be provided here
  sorry

end cartons_in_a_case_l876_87625


namespace fourth_vertex_of_square_l876_87637

theorem fourth_vertex_of_square (A B C D : ℂ) : 
  A = (2 + 3 * I) ∧ B = (-3 + 2 * I) ∧ C = (-2 - 3 * I) →
  D = (0 - 0.5 * I) :=
sorry

end fourth_vertex_of_square_l876_87637


namespace find_unique_number_l876_87606

def is_three_digit_number (N : ℕ) : Prop := 100 ≤ N ∧ N < 1000

def nonzero_digits (A B C : ℕ) : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0

def digits_of_number (N A B C : ℕ) : Prop := N = 100 * A + 10 * B + C

def product (N A B : ℕ) := N * (10 * A + B) * A

def divides (n m : ℕ) := ∃ k, n * k = m

theorem find_unique_number (N A B C : ℕ) (h1 : is_three_digit_number N)
    (h2 : nonzero_digits A B C) (h3 : digits_of_number N A B C)
    (h4 : divides 1000 (product N A B)) : N = 875 :=
sorry

end find_unique_number_l876_87606


namespace closest_ratio_adults_children_l876_87608

theorem closest_ratio_adults_children :
  ∃ (a c : ℕ), 25 * a + 15 * c = 1950 ∧ a ≥ 1 ∧ c ≥ 1 ∧ a / c = 24 / 25 := sorry

end closest_ratio_adults_children_l876_87608


namespace count_four_digit_numbers_divisible_by_17_and_end_in_17_l876_87646

theorem count_four_digit_numbers_divisible_by_17_and_end_in_17 :
  ∃ S : Finset ℕ, S.card = 5 ∧ ∀ n ∈ S, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0 ∧ n % 100 = 17 :=
by
  sorry

end count_four_digit_numbers_divisible_by_17_and_end_in_17_l876_87646


namespace find_alpha_beta_l876_87681

-- Define the conditions of the problem
variables (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π)
variable (h_eq : ∀ x : ℝ, cos (x + α) + sin (x + β) + sqrt 2 * cos x = 0)

-- State the required proof as a theorem
theorem find_alpha_beta : α = 3 * π / 4 ∧ β = 7 * π / 4 :=
by
  sorry

end find_alpha_beta_l876_87681


namespace correct_sum_of_integers_l876_87697

theorem correct_sum_of_integers (x y : ℕ) (h1 : x - y = 4) (h2 : x * y = 192) : x + y = 28 := by
  sorry

end correct_sum_of_integers_l876_87697


namespace hyperbola_standard_eq_proof_l876_87631

noncomputable def real_axis_length := 6
noncomputable def asymptote_slope := 3 / 2

def hyperbola_standard_eq (a b : ℝ) :=
  ∀ x y : ℝ, (y^2 / a^2 - x^2 / b^2 = 1)

theorem hyperbola_standard_eq_proof (a b : ℝ) 
  (h_a : 2 * a = real_axis_length)
  (h_b : a / b = asymptote_slope) :
  hyperbola_standard_eq 3 2 := 
by
  sorry

end hyperbola_standard_eq_proof_l876_87631


namespace volume_is_120_l876_87689

namespace volume_proof

-- Definitions from the given conditions
variables (a b c : ℝ)
axiom ab_relation : a * b = 48
axiom bc_relation : b * c = 20
axiom ca_relation : c * a = 15

-- Goal to prove
theorem volume_is_120 : a * b * c = 120 := by
  sorry

end volume_proof

end volume_is_120_l876_87689


namespace false_propositions_count_l876_87616

-- Definitions of the propositions
def proposition1 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition2 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition3 (A B : Prop) : Prop :=
  ¬ (A ∧ B)

def proposition4 (A B : Prop) : Prop :=
  A ∧ B

-- Theorem to prove the total number of false propositions
theorem false_propositions_count (A B : Prop) (P1 P2 P3 P4 : Prop) :
  ¬ (proposition1 A B P1) ∧ ¬ (proposition2 A B P2) ∧ ¬ (proposition3 A B) ∧ proposition4 A B → 3 = 3 :=
by
  intro h
  sorry

end false_propositions_count_l876_87616


namespace algebra_expression_evaluation_l876_87642

theorem algebra_expression_evaluation (a b c d e : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) 
  (h3 : e < 0) 
  (h4 : abs e = 1) : 
  (-a * b) ^ 2009 - (c + d) ^ 2010 - e ^ 2011 = 0 := by 
  sorry

end algebra_expression_evaluation_l876_87642


namespace symmetric_points_l876_87619

theorem symmetric_points (a b : ℤ) (h1 : (a, -2) = (1, -2)) (h2 : (-1, b) = (-1, -2)) :
  (a + b) ^ 2023 = -1 := by
  -- We know from the conditions:
  -- (a, -2) and (1, -2) implies a = 1
  -- (-1, b) and (-1, -2) implies b = -2
  -- Thus it follows that:
  sorry

end symmetric_points_l876_87619


namespace sqrt_41_40_39_38_plus_1_l876_87672

theorem sqrt_41_40_39_38_plus_1 : Real.sqrt ((41 * 40 * 39 * 38) + 1) = 1559 := by
  sorry

end sqrt_41_40_39_38_plus_1_l876_87672


namespace adam_tickets_left_l876_87695

def tickets_left (total_tickets : ℕ) (ticket_cost : ℕ) (total_spent : ℕ) : ℕ :=
  total_tickets - total_spent / ticket_cost

theorem adam_tickets_left :
  tickets_left 13 9 81 = 4 := 
by
  sorry

end adam_tickets_left_l876_87695


namespace least_five_digit_perfect_square_and_cube_l876_87611

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l876_87611


namespace janet_practiced_days_l876_87626

theorem janet_practiced_days (total_miles : ℕ) (miles_per_day : ℕ) (days_practiced : ℕ) :
  total_miles = 72 ∧ miles_per_day = 8 → days_practiced = total_miles / miles_per_day → days_practiced = 9 :=
by
  sorry

end janet_practiced_days_l876_87626


namespace largest_angle_in_ratio_3_4_5_l876_87685

theorem largest_angle_in_ratio_3_4_5 (x : ℝ) (h1 : 3 * x + 4 * x + 5 * x = 180) : 5 * x = 75 :=
by
  sorry

end largest_angle_in_ratio_3_4_5_l876_87685


namespace spherical_to_rectangular_coords_l876_87613

theorem spherical_to_rectangular_coords :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 5 * Real.sin (Real.pi / 3) * Real.cos (Real.pi / 4) ∧
  y = 5 * Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4) ∧
  z = 5 * Real.cos (Real.pi / 3) ∧
  x = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  y = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  z = 2.5 ∧
  (x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 2.5) :=
by {
  sorry
}

end spherical_to_rectangular_coords_l876_87613


namespace ma_m_gt_mb_l876_87659

theorem ma_m_gt_mb (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m * a > m * b) → m ≥ 0 := 
  sorry

end ma_m_gt_mb_l876_87659


namespace optimal_garden_dimensions_l876_87682

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400 ∧
                l ≥ 100 ∧
                w ≥ 0 ∧ 
                l * w = 10000) :=
by
  sorry

end optimal_garden_dimensions_l876_87682


namespace sufficient_but_not_necessary_condition_l876_87673

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a - Real.sin x

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, f' a x > 0 → (a > 1)) ∧ (¬∀ x, f' a x ≥ 0 → (a > 1)) := sorry

end sufficient_but_not_necessary_condition_l876_87673


namespace factor_of_polynomial_l876_87617

theorem factor_of_polynomial (x : ℝ) : 
  (x^2 - 2*x + 2) ∣ (29 * 39 * x^4 + 4) :=
sorry

end factor_of_polynomial_l876_87617


namespace total_artworks_l876_87644

theorem total_artworks (students : ℕ) (group1_artworks : ℕ) (group2_artworks : ℕ) (total_students : students = 10) 
    (artwork_group1 : group1_artworks = 5 * 3) (artwork_group2 : group2_artworks = 5 * 4) : 
    group1_artworks + group2_artworks = 35 :=
by
  sorry

end total_artworks_l876_87644


namespace regression_equation_l876_87640

-- Define the regression coefficient and correlation
def negatively_correlated (x y : ℝ) : Prop :=
  ∃ (a : ℝ), a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100

-- The question is to prove that given x and y are negatively correlated,
-- the regression equation is \hat{y} = -2x + 100
theorem regression_equation (x y : ℝ) (h : negatively_correlated x y) :
  (∃ a, a < 0 ∧ ∀ (x_val : ℝ), y = a * x_val + 100) → ∃ (b : ℝ), b = -2 ∧ ∀ (x_val : ℝ), y = b * x_val + 100 :=
by
  sorry

end regression_equation_l876_87640


namespace cheapest_book_price_l876_87684

theorem cheapest_book_price
  (n : ℕ) (c : ℕ) (d : ℕ)
  (h1 : n = 40)
  (h2 : d = 3)
  (h3 : c + d * 19 = 75) :
  c = 18 :=
sorry

end cheapest_book_price_l876_87684


namespace cuckoo_sounds_from_10_to_16_l876_87687

-- Define a function for the cuckoo sounds per hour considering the clock
def cuckoo_sounds (h : ℕ) : ℕ :=
  if h ≤ 12 then h else h - 12

-- Define the total number of cuckoo sounds from 10:00 to 16:00
def total_cuckoo_sounds : ℕ :=
  (List.range' 10 (16 - 10 + 1)).map cuckoo_sounds |>.sum

theorem cuckoo_sounds_from_10_to_16 : total_cuckoo_sounds = 43 := by
  sorry

end cuckoo_sounds_from_10_to_16_l876_87687


namespace set_intersection_l876_87601

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x * (4 - x) < 0}
def C_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_intersection :
  A ∩ C_R_B = {1, 2, 3, 4} :=
by
  -- Proof goes here
  sorry

end set_intersection_l876_87601


namespace Jimin_addition_l876_87600

theorem Jimin_addition (x : ℕ) (h : 96 / x = 6) : 34 + x = 50 := 
by
  sorry

end Jimin_addition_l876_87600


namespace surface_area_of_rectangular_solid_is_334_l876_87607

theorem surface_area_of_rectangular_solid_is_334
  (l w h : ℕ)
  (h_l_prime : Prime l)
  (h_w_prime : Prime w)
  (h_h_prime : Prime h)
  (volume_eq_385 : l * w * h = 385) : 
  2 * (l * w + l * h + w * h) = 334 := 
sorry

end surface_area_of_rectangular_solid_is_334_l876_87607


namespace total_weight_fruits_in_good_condition_l876_87666

theorem total_weight_fruits_in_good_condition :
  let oranges_initial := 600
  let bananas_initial := 400
  let apples_initial := 300
  let avocados_initial := 200
  let grapes_initial := 100
  let pineapples_initial := 50

  let oranges_rotten := 0.15 * oranges_initial
  let bananas_rotten := 0.05 * bananas_initial
  let apples_rotten := 0.08 * apples_initial
  let avocados_rotten := 0.10 * avocados_initial
  let grapes_rotten := 0.03 * grapes_initial
  let pineapples_rotten := 0.20 * pineapples_initial

  let oranges_good := oranges_initial - oranges_rotten
  let bananas_good := bananas_initial - bananas_rotten
  let apples_good := apples_initial - apples_rotten
  let avocados_good := avocados_initial - avocados_rotten
  let grapes_good := grapes_initial - grapes_rotten
  let pineapples_good := pineapples_initial - pineapples_rotten

  let weight_per_orange := 150 / 1000 -- kg
  let weight_per_banana := 120 / 1000 -- kg
  let weight_per_apple := 100 / 1000 -- kg
  let weight_per_avocado := 80 / 1000 -- kg
  let weight_per_grape := 5 / 1000 -- kg
  let weight_per_pineapple := 1 -- kg

  oranges_good * weight_per_orange +
  bananas_good * weight_per_banana +
  apples_good * weight_per_apple +
  avocados_good * weight_per_avocado +
  grapes_good * weight_per_grape +
  pineapples_good * weight_per_pineapple = 204.585 :=
by
  sorry

end total_weight_fruits_in_good_condition_l876_87666


namespace circle_equation_k_range_l876_87670

theorem circle_equation_k_range (k : ℝ) :
  ∀ x y: ℝ, x^2 + y^2 + 4*k*x - 2*y + 4*k^2 - k = 0 →
  k > -1 := 
sorry

end circle_equation_k_range_l876_87670


namespace andrew_eggs_l876_87638

def andrew_eggs_problem (a b : ℕ) (half_eggs_given_away : ℚ) (remaining_eggs : ℕ) : Prop :=
  a + b - (a + b) * half_eggs_given_away = remaining_eggs

theorem andrew_eggs :
  andrew_eggs_problem 8 62 (1/2 : ℚ) 35 :=
by
  sorry

end andrew_eggs_l876_87638


namespace largest_number_l876_87620

def A : ℚ := 97 / 100
def B : ℚ := 979 / 1000
def C : ℚ := 9709 / 10000
def D : ℚ := 907 / 1000
def E : ℚ := 9089 / 10000

theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_number_l876_87620


namespace factor_and_sum_coeffs_l876_87639

noncomputable def sum_of_integer_coeffs_of_factorization (x y : ℤ) : ℤ :=
  let factors := ([(1 : ℤ), (-1 : ℤ), (5 : ℤ), (1 : ℤ), (6 : ℤ), (1 : ℤ), (1 : ℤ), (5 : ℤ), (-1 : ℤ), (6 : ℤ)])
  factors.sum

theorem factor_and_sum_coeffs (x y : ℤ) :
  (125 * (x^9:ℤ) - 216 * (y^9:ℤ) = (x - y) * (5 * x^2 + x * y + 6 * y^2) * (x + y) * (5 * x^2 - x * y + 6 * y^2))
  ∧ (sum_of_integer_coeffs_of_factorization x y = 24) :=
by
  sorry

end factor_and_sum_coeffs_l876_87639


namespace minimum_value_of_a_plus_b_l876_87609

noncomputable def f (x : ℝ) := Real.log x - (1 / x)
noncomputable def f' (x : ℝ) := 1 / x + 1 / (x^2)

theorem minimum_value_of_a_plus_b (a b m : ℝ) (h1 : a = 1 / m + 1 / (m^2)) 
  (h2 : b = Real.log m - 2 / m - 1) : a + b = -1 :=
by
  sorry

end minimum_value_of_a_plus_b_l876_87609


namespace cookies_left_l876_87635

-- Define the conditions as in the problem
def dozens_to_cookies(dozens : ℕ) : ℕ := dozens * 12
def initial_cookies := dozens_to_cookies 2
def eaten_cookies := 3

-- Prove that John has 21 cookies left
theorem cookies_left : initial_cookies - eaten_cookies = 21 :=
  by
  sorry

end cookies_left_l876_87635


namespace number_divisible_by_19_l876_87643

theorem number_divisible_by_19 (n : ℕ) : (12000 + 3 * 10^n + 8) % 19 = 0 := 
by sorry

end number_divisible_by_19_l876_87643


namespace pieces_per_block_is_32_l876_87694

-- Define the number of pieces of junk mail given to each house
def pieces_per_house : ℕ := 8

-- Define the number of houses in each block
def houses_per_block : ℕ := 4

-- Calculate the total number of pieces of junk mail given to each block
def total_pieces_per_block : ℕ := pieces_per_house * houses_per_block

-- Prove that the total number of pieces of junk mail given to each block is 32
theorem pieces_per_block_is_32 : total_pieces_per_block = 32 := 
by sorry

end pieces_per_block_is_32_l876_87694


namespace negation_universal_prop_l876_87693

theorem negation_universal_prop :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ ∃ x : ℝ, x^2 < 0 :=
sorry

end negation_universal_prop_l876_87693


namespace total_amount_of_currency_notes_l876_87645

theorem total_amount_of_currency_notes (x y : ℕ) (h1 : x + y = 85) (h2 : 50 * y = 3500) : 100 * x + 50 * y = 5000 := by
  sorry

end total_amount_of_currency_notes_l876_87645


namespace is_odd_function_l876_87691

def f (x : ℝ) : ℝ := x^3 - x

theorem is_odd_function : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

end is_odd_function_l876_87691


namespace find_angle_and_area_l876_87627

theorem find_angle_and_area (a b c : ℝ) (C : ℝ)
  (h₁: (a^2 + b^2 - c^2) * Real.tan C = Real.sqrt 2 * a * b)
  (h₂: c = 2)
  (h₃: b = 2 * Real.sqrt 2) : 
  C = Real.pi / 4 ∧ a = 2 ∧ (∃ S : ℝ, S = 1 / 2 * a * c ∧ S = 2) :=
by
  -- We assume sorry here since the focus is on setting up the problem statement correctly
  sorry

end find_angle_and_area_l876_87627


namespace maddy_credits_to_graduate_l876_87610

theorem maddy_credits_to_graduate (semesters : ℕ) (credits_per_class : ℕ) (classes_per_semester : ℕ)
  (semesters_eq : semesters = 8)
  (credits_per_class_eq : credits_per_class = 3)
  (classes_per_semester_eq : classes_per_semester = 5) :
  semesters * (classes_per_semester * credits_per_class) = 120 :=
by
  -- Placeholder for proof
  sorry

end maddy_credits_to_graduate_l876_87610


namespace cos_eq_neg_four_fifths_of_tan_l876_87658

theorem cos_eq_neg_four_fifths_of_tan (α : ℝ) (h_tan : Real.tan α = 3 / 4) (h_interval : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.cos α = -4 / 5 :=
sorry

end cos_eq_neg_four_fifths_of_tan_l876_87658


namespace eyes_that_saw_the_plane_l876_87674

theorem eyes_that_saw_the_plane (students : ℕ) (ratio : ℚ) (eyes_per_student : ℕ) 
  (h1 : students = 200) (h2 : ratio = 3 / 4) (h3 : eyes_per_student = 2) : 
  2 * (ratio * students) = 300 := 
by 
  -- the proof is omitted
  sorry

end eyes_that_saw_the_plane_l876_87674


namespace smallest_integer_solution_l876_87649

theorem smallest_integer_solution (n : ℤ) (h : n^3 - 12 * n^2 + 44 * n - 48 ≤ 0) : n = 2 :=
sorry

end smallest_integer_solution_l876_87649


namespace sum_of_possible_remainders_l876_87671

theorem sum_of_possible_remainders (n : ℕ) (h_even : ∃ k : ℕ, n = 2 * k) : 
  let m := 1000 * (2 * n + 6) + 100 * (2 * n + 4) + 10 * (2 * n + 2) + (2 * n)
  let remainder (k : ℕ) := (1112 * k + 6420) % 29
  23 + 7 + 20 = 50 :=
  by
  sorry

end sum_of_possible_remainders_l876_87671


namespace urn_probability_l876_87629

theorem urn_probability :
  ∀ (urn: Finset (ℕ × ℕ)), 
    urn = {(2, 1)} →
    (∀ (n : ℕ) (urn' : Finset (ℕ × ℕ)), n ≤ 5 → urn = urn' → 
      (∃ (r b : ℕ), (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)} ∨ (r, b) ∈ urn' ∧ urn' = {(r + 1, b), (r, b + 1)}) → 
    ∃ (p : ℚ), p = 8 / 21)
  := by
    sorry

end urn_probability_l876_87629


namespace center_of_circle_l876_87622

-- Defining the equation of the circle as a hypothesis
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y = 0

-- Stating the theorem about the center of the circle
theorem center_of_circle : ∀ x y : ℝ, circle_eq x y → (x = 2 ∧ y = -1) :=
by
  sorry

end center_of_circle_l876_87622


namespace polynomial_sum_l876_87648

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l876_87648
