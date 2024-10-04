import Mathlib

namespace find_p_from_binomial_distribution_l199_199629

theorem find_p_from_binomial_distribution (p : ℝ) (h₁ : 0 ≤ p ∧ p ≤ 1) 
    (h₂ : ∀ n k : ℕ, k ≤ n → 0 ≤ p^(k:ℝ) * (1-p)^((n-k):ℝ)) 
    (h₃ : (1 - (1 - p)^2 = 5 / 9)) : p = 1 / 3 :=
by sorry

end find_p_from_binomial_distribution_l199_199629


namespace horses_put_by_c_l199_199519

theorem horses_put_by_c (a_horses a_months b_horses b_months c_months total_cost b_cost : ℕ) (x : ℕ) 
  (h1 : a_horses = 12) 
  (h2 : a_months = 8) 
  (h3 : b_horses = 16) 
  (h4 : b_months = 9) 
  (h5 : c_months = 6) 
  (h6 : total_cost = 870) 
  (h7 : b_cost = 360) 
  (h8 : 144 / (96 + 144 + 6 * x) = 360 / 870) : 
  x = 18 := 
by 
  sorry

end horses_put_by_c_l199_199519


namespace continuous_on_integrable_l199_199027

theorem continuous_on_integrable {a b : ℝ} (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b)) :
  IntervalIntegrable f measure_theory.measure_space.volume a b := 
sorry

end continuous_on_integrable_l199_199027


namespace sum_of_ages_l199_199460

theorem sum_of_ages (a b c d : ℕ) (h1 : a * b = 20) (h2 : c * d = 28) (distinct : ∀ (x y : ℕ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) : a + b + c + d = 19 :=
sorry

end sum_of_ages_l199_199460


namespace brown_shoes_count_l199_199828

-- Definitions based on given conditions
def total_shoes := 66
def black_shoe_ratio := 2

theorem brown_shoes_count (B : ℕ) (H1 : black_shoe_ratio * B + B = total_shoes) : B = 22 :=
by
  -- Proof here is replaced with sorry for the purpose of this exercise
  sorry

end brown_shoes_count_l199_199828


namespace base10_to_base7_l199_199841

theorem base10_to_base7 : 
  ∃ a b c d : ℕ, a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 = 729 ∧ a = 2 ∧ b = 0 ∧ c = 6 ∧ d = 1 :=
sorry

end base10_to_base7_l199_199841


namespace polynomial_roots_l199_199901

theorem polynomial_roots:
  ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (2*x - 8) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end polynomial_roots_l199_199901


namespace solve_siblings_age_problem_l199_199507

def siblings_age_problem (x : ℕ) : Prop :=
  let age_eldest := 20
  let age_middle := 15
  let age_youngest := 10
  (age_eldest + x) + (age_middle + x) + (age_youngest + x) = 75 → x = 10

theorem solve_siblings_age_problem : siblings_age_problem 10 :=
by
  sorry

end solve_siblings_age_problem_l199_199507


namespace telegraph_longer_than_pardee_l199_199495

theorem telegraph_longer_than_pardee : 
  let telegraph_length_km := 162 in
  let pardee_length_m := 12000 in
  let pardee_length_km := pardee_length_m / 1000 in
  telegraph_length_km - pardee_length_km = 150 :=
by
  sorry

end telegraph_longer_than_pardee_l199_199495


namespace circle_center_radius_sum_correct_l199_199461

noncomputable def circle_center_radius_sum (eq : String) : ℝ :=
  if h : eq = "x^2 + 8x - 2y^2 - 6y = -6" then
    let c : ℝ := -4
    let d : ℝ := -3 / 2
    let s : ℝ := Real.sqrt (47 / 4)
    c + d + s
  else 0

theorem circle_center_radius_sum_correct :
  circle_center_radius_sum "x^2 + 8x - 2y^2 - 6y = -6" = (-11 + Real.sqrt 47) / 2 :=
by
  -- proof omitted
  sorry

end circle_center_radius_sum_correct_l199_199461


namespace find_initial_men_l199_199976

def men_employed (M : ℕ) : Prop :=
  let total_hours := 50 * 8
  let completed_hours := 25 * 8
  let remaining_hours := total_hours - completed_hours
  let new_hours := 25 * 10
  let completed_work := 1 / 3
  let remaining_work := 2 / 3
  let total_work := 2 -- Total work in terms of "work units", assuming 2 km = 2 work units
  let first_eq := M * 25 * 8 = total_work * completed_work
  let second_eq := (M + 60) * 25 * 10 = total_work * remaining_work
  (M = 300 → first_eq ∧ second_eq)

theorem find_initial_men : ∃ M : ℕ, men_employed M := sorry

end find_initial_men_l199_199976


namespace sqrt_225_eq_15_l199_199859

theorem sqrt_225_eq_15 : Real.sqrt 225 = 15 :=
sorry

end sqrt_225_eq_15_l199_199859


namespace price_of_fifth_basket_l199_199815

-- Define the initial conditions
def avg_cost_of_4_baskets (total_cost_4 : ℝ) : Prop :=
  total_cost_4 / 4 = 4

def avg_cost_of_5_baskets (total_cost_5 : ℝ) : Prop :=
  total_cost_5 / 5 = 4.8

-- Theorem statement to be proved
theorem price_of_fifth_basket
  (total_cost_4 : ℝ)
  (h1 : avg_cost_of_4_baskets total_cost_4)
  (total_cost_5 : ℝ)
  (h2 : avg_cost_of_5_baskets total_cost_5) :
  total_cost_5 - total_cost_4 = 8 :=
by
  sorry

end price_of_fifth_basket_l199_199815


namespace least_repeating_block_of_8_over_11_l199_199642

theorem least_repeating_block_of_8_over_11 : (∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∃ a b : ℤ, (10^m - 1) * (8 * 10^n - b * 11 * 10^(n - t)) = a * 11 * 10^(m - t))) ∧ n ≤ 2) :=
by
  sorry

end least_repeating_block_of_8_over_11_l199_199642


namespace product_is_correct_l199_199207

theorem product_is_correct :
  50 * 29.96 * 2.996 * 500 = 2244004 :=
by
  sorry

end product_is_correct_l199_199207


namespace find_distance_IO_l199_199268

noncomputable def problem_statement :=
  let A B C D E I O : Point
  {A B C : Point} {D E I O : Point} in
  -- conditions
  ∃ (A B C D E I O : Point),
    (D ∈ (Segment B C)) ∧ (E ∈ (Segment B C)) ∧
    (Distance A D = 6) ∧ (Distance D B = 6) ∧
    (Distance A E = 8) ∧ (Distance E C = 8) ∧
    (Incenter (Triangle A D E) I) ∧ (Circumcenter (Triangle A B C) O) ∧
    (Distance A I = 5) ∧
    -- question and answer
    Distance I O = 23 / 5

theorem find_distance_IO (A B C D E I O : Point) :
  (D ∈ (Segment B C)) →
  (E ∈ (Segment B C)) →
  (Distance A D = 6) →
  (Distance D B = 6) →
  (Distance A E = 8) →
  (Distance E C = 8) →
  (Incenter (Triangle A D E) I) →
  (Circumcenter (Triangle A B C) O) →
  (Distance A I = 5) →
  Distance I O = 23 / 5 :=
  sorry

end find_distance_IO_l199_199268


namespace mean_of_six_numbers_sum_three_quarters_l199_199501

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l199_199501


namespace z_is_negative_y_intercept_l199_199662

-- Define the objective function as an assumption or condition
def objective_function (x y z : ℝ) : Prop := z = 3 * x - y

-- Define what we need to prove: z is the negative of the y-intercept 
def negative_y_intercept (x y z : ℝ) : Prop := ∃ m b, (y = m * x + b) ∧ m = 3 ∧ b = -z

-- The theorem we need to prove
theorem z_is_negative_y_intercept (x y z : ℝ) (h : objective_function x y z) : negative_y_intercept x y z :=
  sorry

end z_is_negative_y_intercept_l199_199662


namespace number_of_folds_l199_199897

theorem number_of_folds (n : ℕ) :
  (3 * (8 * 8)) / n = 48 → n = 4 :=
by
  sorry

end number_of_folds_l199_199897


namespace p_interval_satisfies_inequality_l199_199317

theorem p_interval_satisfies_inequality :
  ∀ (p q : ℝ), 0 ≤ p ∧ p < 2.232 ∧ q > 0 ∧ p + q ≠ 0 →
    (4 * (p * q ^ 2 + p ^ 2 * q + 4 * q ^ 2 + 4 * p * q)) / (p + q) > 5 * p ^ 2 * q :=
by sorry

end p_interval_satisfies_inequality_l199_199317


namespace candies_remaining_after_yellow_eaten_l199_199345

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l199_199345


namespace ear_muffs_total_l199_199543

theorem ear_muffs_total (a b : ℕ) (h1 : a = 1346) (h2 : b = 6444) : a + b = 7790 :=
by
  sorry

end ear_muffs_total_l199_199543


namespace point_coordinates_l199_199783

-- Definitions based on conditions
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0
def dist_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := abs P.1 = d

-- Lean 4 statement
theorem point_coordinates {P : ℝ × ℝ} (h1 : on_x_axis P) (h2 : dist_to_y_axis P 3) :
  P = (3, 0) ∨ P = (-3, 0) :=
by sorry

end point_coordinates_l199_199783


namespace tangent_line_parallel_x_axis_l199_199949

def f (x : ℝ) : ℝ := x^4 - 4 * x

theorem tangent_line_parallel_x_axis :
  ∃ (m n : ℝ), (n = f m) ∧ (deriv f m = 0) ∧ (m, n) = (1, -3) := by
  sorry

end tangent_line_parallel_x_axis_l199_199949


namespace part1_even_function_part2_two_distinct_zeros_l199_199933

noncomputable def f (x a : ℝ) : ℝ := (4^x + a) / 2^x
noncomputable def g (x a : ℝ) : ℝ := f x a - (a + 1)

theorem part1_even_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) ↔ a = 1 :=
sorry

theorem part2_two_distinct_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (a ∈ Set.Icc (1/2) 1 ∪ Set.Icc 1 2) :=
sorry

end part1_even_function_part2_two_distinct_zeros_l199_199933


namespace base8_operations_l199_199373

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations_l199_199373


namespace triangle_median_perpendicular_l199_199787

theorem triangle_median_perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : (x1 - (x2 + x3) / 2) * (x2 - (x1 + x3) / 2) + (y1 - (y2 + y3) / 2) * (y2 - (y1 + y3) / 2) = 0)
  (h2 : (x2 - x3) ^ 2 + (y2 - y3) ^ 2 = 64)
  (h3 : (x1 - x3) ^ 2 + (y1 - y3) ^ 2 = 25) : 
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 = 22.25 := sorry

end triangle_median_perpendicular_l199_199787


namespace trips_Jean_l199_199545

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l199_199545


namespace regular_polygon_sides_l199_199378

theorem regular_polygon_sides (C : ℕ) (h : (C - 2) * 180 / C = 144) : C = 10 := 
sorry

end regular_polygon_sides_l199_199378


namespace man_speed_l199_199869

theorem man_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h1 : distance = 12)
  (h2 : time_minutes = 72)
  (h3 : time_hours = time_minutes / 60)
  (h4 : speed = distance / time_hours) : speed = 10 :=
by
  sorry

end man_speed_l199_199869


namespace sarah_initial_money_l199_199489

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l199_199489


namespace not_all_same_probability_l199_199059

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l199_199059


namespace friendly_number_pair_a_equals_negative_three_fourths_l199_199056

theorem friendly_number_pair_a_equals_negative_three_fourths (a : ℚ) (h : (a / 2) + (3 / 4) = (a + 3) / 6) : 
  a = -3 / 4 :=
sorry

end friendly_number_pair_a_equals_negative_three_fourths_l199_199056


namespace cookie_calories_l199_199150

theorem cookie_calories 
  (burger_calories : ℕ)
  (carrot_stick_calories : ℕ)
  (num_carrot_sticks : ℕ)
  (total_lunch_calories : ℕ) :
  burger_calories = 400 ∧ 
  carrot_stick_calories = 20 ∧ 
  num_carrot_sticks = 5 ∧ 
  total_lunch_calories = 750 →
  (total_lunch_calories - (burger_calories + num_carrot_sticks * carrot_stick_calories) = 250) :=
by sorry

end cookie_calories_l199_199150


namespace solve_quadratic_l199_199817

theorem solve_quadratic {x : ℚ} (h1 : x > 0) (h2 : 3 * x ^ 2 + 11 * x - 20 = 0) : x = 4 / 3 :=
sorry

end solve_quadratic_l199_199817


namespace angle_C_correct_l199_199411

theorem angle_C_correct (A B C : ℝ) (h1 : A = 65) (h2 : B = 40) (h3 : A + B + C = 180) : C = 75 :=
sorry

end angle_C_correct_l199_199411


namespace andy_coats_l199_199540

theorem andy_coats 
  (initial_minks : ℕ)
  (offspring_4_minks count_4_offspring : ℕ)
  (offspring_6_minks count_6_offspring : ℕ)
  (offspring_8_minks count_8_offspring : ℕ)
  (freed_percentage coat_requirement total_minks offspring_minks freed_minks remaining_minks coats : ℕ) :
  initial_minks = 30 ∧
  offspring_4_minks = 10 ∧ count_4_offspring = 4 ∧
  offspring_6_minks = 15 ∧ count_6_offspring = 6 ∧
  offspring_8_minks = 5 ∧ count_8_offspring = 8 ∧
  freed_percentage = 60 ∧ coat_requirement = 15 ∧
  total_minks = initial_minks + offspring_minks ∧
  offspring_minks = offspring_4_minks * count_4_offspring + offspring_6_minks * count_6_offspring + offspring_8_minks * count_8_offspring ∧
  freed_minks = total_minks * freed_percentage / 100 ∧
  remaining_minks = total_minks - freed_minks ∧
  coats = remaining_minks / coat_requirement →
  coats = 5 :=
sorry

end andy_coats_l199_199540


namespace base_10_to_base_7_l199_199842

theorem base_10_to_base_7 : 
  ∀ (n : ℕ), n = 729 → n = 2 * 7^3 + 0 * 7^2 + 6 * 7^1 + 1 * 7^0 :=
by
  intros n h
  rw h
  sorry

end base_10_to_base_7_l199_199842


namespace cannot_determine_E1_l199_199231

variable (a b c d : ℝ)

theorem cannot_determine_E1 (h1 : a + b - c - d = 5) (h2 : (b - d)^2 = 16) : 
  ¬ ∃ e : ℝ, e = a - b - c + d :=
by
  sorry

end cannot_determine_E1_l199_199231


namespace integer_solutions_zero_l199_199288

theorem integer_solutions_zero (x y u t : ℤ) :
  x^2 + y^2 = 1974 * (u^2 + t^2) → 
  x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 :=
by
  sorry

end integer_solutions_zero_l199_199288


namespace find_m_geq_9_l199_199082

-- Define the real numbers
variables {x m : ℝ}

-- Define the conditions
def p (x : ℝ) := x ≤ 2
def q (x m : ℝ) := x^2 - 2*x + 1 - m^2 ≤ 0

-- Main theorem statement based on the given problem
theorem find_m_geq_9 (m : ℝ) (hm : m > 0) :
  (¬ p x → ¬ q x m) → (p x → q x m) → m ≥ 9 :=
  sorry

end find_m_geq_9_l199_199082


namespace cloth_gain_representation_l199_199370

theorem cloth_gain_representation (C S : ℝ) (h1 : S = 1.20 * C) (h2 : ∃ gain, gain = 60 * S - 60 * C) :
  ∃ meters : ℝ, meters = (60 * S - 60 * C) / S ∧ meters = 12 :=
by
  sorry

end cloth_gain_representation_l199_199370


namespace minimum_sum_of_squares_l199_199146

theorem minimum_sum_of_squares (α p q : ℝ) 
  (h1: p + q = α - 2) (h2: p * q = - (α + 1)) :
  p^2 + q^2 ≥ 5 :=
by
  sorry

end minimum_sum_of_squares_l199_199146


namespace factor_fraction_eq_l199_199554

theorem factor_fraction_eq (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) 
  / ((a + b)^3 + (b + c)^3 + (c + a)^3) = 
  ((a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2)) 
  / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end factor_fraction_eq_l199_199554


namespace binomial_square_expression_l199_199328

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l199_199328


namespace correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l199_199849

theorem correct_calculation_A : (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) :=
by { sorry }

theorem incorrect_calculation_B : (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) :=
by { sorry }

theorem incorrect_calculation_C : ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem incorrect_calculation_D : (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by { sorry }

theorem correct_answer_is_A :
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) ∧
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧
  ((Real.sqrt 2)^2 ≠ 2 * Real.sqrt 2) ∧
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) :=
by {
  exact ⟨correct_calculation_A, incorrect_calculation_B, incorrect_calculation_C, incorrect_calculation_D⟩
}

end correct_calculation_A_incorrect_calculation_B_incorrect_calculation_C_incorrect_calculation_D_correct_answer_is_A_l199_199849


namespace sum_of_digits_of_63_l199_199561

theorem sum_of_digits_of_63 (x y : ℕ) (h : 10 * x + y = 63) (h1 : x + y = 9) (h2 : x - y = 3) : x + y = 9 :=
by
  sorry

end sum_of_digits_of_63_l199_199561


namespace penny_difference_l199_199971

variables (p : ℕ)

/-- Liam and Mia have certain numbers of fifty-cent coins. This theorem proves the difference 
    in their total value in pennies. 
-/
theorem penny_difference:
  (3 * p + 2) * 50 - (2 * p + 7) * 50 = 50 * p - 250 :=
by
  sorry

end penny_difference_l199_199971


namespace inequality_proof_l199_199924

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l199_199924


namespace min_top_block_sum_l199_199533

theorem min_top_block_sum : 
  ∀ (assign_numbers : ℕ → ℕ) 
  (layer_1 : Fin 16 → ℕ) (layer_2 : Fin 9 → ℕ) (layer_3 : Fin 4 → ℕ) (top_block : ℕ),
  (∀ i, layer_3 i = layer_2 (i / 2) + layer_2 ((i / 2) + 1) + layer_2 ((i / 2) + 3) + layer_2 ((i / 2) + 4)) →
  (∀ i, layer_2 i = layer_1 (i / 2) + layer_1 ((i / 2) + 1) + layer_1 ((i / 2) + 3) + layer_1 ((i / 2) + 4)) →
  (top_block = layer_3 0 + layer_3 1 + layer_3 2 + layer_3 3) →
  top_block = 40 :=
sorry

end min_top_block_sum_l199_199533


namespace find_m_l199_199935

def A : Set ℤ := {-1, 1}
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

theorem find_m (m : ℤ) (h : B m ⊆ A) : m = 0 ∨ m = 1 ∨ m = -1 := 
sorry

end find_m_l199_199935


namespace minimum_police_officers_needed_l199_199159

def grid := (5, 8)
def total_intersections : ℕ := 54
def max_distance_to_police := 2

theorem minimum_police_officers_needed (min_police_needed : ℕ) :
  (min_police_needed = 6) := sorry

end minimum_police_officers_needed_l199_199159


namespace carbonated_water_percentage_is_correct_l199_199681

-- Given percentages of lemonade and carbonated water in two solutions
def first_solution : Rat := 0.20 -- Lemonade percentage in the first solution
def second_solution : Rat := 0.45 -- Lemonade percentage in the second solution

-- Calculate percentages of carbonated water
def first_solution_carbonated_water := 1 - first_solution
def second_solution_carbonated_water := 1 - second_solution

-- Assume the mixture is 100 units, with equal parts from both solutions
def volume_mixture : Rat := 100
def volume_first_solution : Rat := volume_mixture * 0.50
def volume_second_solution : Rat := volume_mixture * 0.50

-- Calculate total carbonated water in the mixture
def carbonated_water_in_mixture :=
  (volume_first_solution * first_solution_carbonated_water) +
  (volume_second_solution * second_solution_carbonated_water)

-- Calculate the percentage of carbonated water in the mixture
def percentage_carbonated_water_in_mixture : Rat :=
  (carbonated_water_in_mixture / volume_mixture) * 100

-- Prove the percentage of carbonated water in the mixture is 67.5%
theorem carbonated_water_percentage_is_correct :
  percentage_carbonated_water_in_mixture = 67.5 := by
  sorry

end carbonated_water_percentage_is_correct_l199_199681


namespace numberOfCows_l199_199281

-- Definitions coming from the conditions
def hasFoxes (n : Nat) := n = 15
def zebrasFromFoxes (z f : Nat) := z = 3 * f
def totalAnimalRequirement (total : Nat) := total = 100
def addedSheep (s : Nat) := s = 20

-- Theorem stating the desired proof
theorem numberOfCows (f z total s c : Nat) 
 (h1 : hasFoxes f)
 (h2 : zebrasFromFoxes z f) 
 (h3 : totalAnimalRequirement total) 
 (h4 : addedSheep s) :
 c = total - s - (f + z) := by
 sorry

end numberOfCows_l199_199281


namespace opposite_of_neg_eight_l199_199825

theorem opposite_of_neg_eight : (-(-8)) = 8 :=
by
  sorry

end opposite_of_neg_eight_l199_199825


namespace find_natural_number_l199_199904

-- Definitions reflecting the conditions and result
def is_sum_of_two_squares (n : ℕ) := ∃ a b : ℕ, a * a + b * b = n

def has_exactly_one_not_sum_of_two_squares (n : ℕ) :=
  ∃! x : ℤ, ¬is_sum_of_two_squares (x.natAbs % n)

theorem find_natural_number (n : ℕ) (h : n ≥ 2) : 
  has_exactly_one_not_sum_of_two_squares n ↔ n = 4 :=
sorry

end find_natural_number_l199_199904


namespace part1_l199_199421

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : 
  f x 1 ≥ 1 :=
sorry

end part1_l199_199421


namespace probability_XOXOXOX_l199_199573

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l199_199573


namespace value_of_q_l199_199741

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end value_of_q_l199_199741


namespace elderly_people_sampled_l199_199443

theorem elderly_people_sampled (total_population : ℕ) (children : ℕ) (elderly : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_population = 1500)
  (h2 : ∃ d, children + d = elderly ∧ elderly + d = middle_aged)
  (h3 : total_population = children + elderly + middle_aged)
  (h4 : sample_size = 60) :
  elderly * (sample_size / total_population) = 20 :=
by
  -- Proof will be written here
  sorry

end elderly_people_sampled_l199_199443


namespace product_in_third_quadrant_l199_199465

def z1 : ℂ := 1 - 3 * Complex.I
def z2 : ℂ := 3 - 2 * Complex.I
def z := z1 * z2

theorem product_in_third_quadrant : z.re < 0 ∧ z.im < 0 := 
sorry

end product_in_third_quadrant_l199_199465


namespace probability_XOXOXOX_is_1_div_35_l199_199576

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l199_199576


namespace find_f1_find_f3_range_of_x_l199_199917

-- Define f as described
axiom f : ℝ → ℝ
axiom f_domain : ∀ (x : ℝ), x > 0 → ∃ (y : ℝ), f y = f x

-- Given conditions
axiom condition1 : ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0
axiom condition2 : ∀ (x y : ℝ), 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom condition3 : f (1 / 3) = 1

-- Prove f(1) = 0
theorem find_f1 : f 1 = 0 := by sorry

-- Prove f(3) = -1
theorem find_f3 : f 3 = -1 := by sorry

-- Given inequality condition
axiom condition4 : ∀ x : ℝ, 0 < x → f x < 2 + f (2 - x)

-- Prove range of x for given inequality
theorem range_of_x : ∀ x, x > 1 / 5 ∧ x < 2 ↔ f x < 2 + f (2 - x) := by sorry

end find_f1_find_f3_range_of_x_l199_199917


namespace find_a_plus_d_l199_199605

noncomputable def f (a b c d x : ℚ) : ℚ := (a * x + b) / (c * x + d)

theorem find_a_plus_d (a b c d : ℚ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℚ, f a b c d (f a b c d x) = x) :
  a + d = 0 := by
  sorry

end find_a_plus_d_l199_199605


namespace acute_triangle_sec_csc_inequality_l199_199012

theorem acute_triangle_sec_csc_inequality (A B C : ℝ) (h : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA90 : A < π / 2) (hB90 : B < π / 2) (hC90 : C < π / 2) :
  (1 / Real.cos A) + (1 / Real.cos B) + (1 / Real.cos C) ≥
  (1 / Real.sin (A / 2)) + (1 / Real.sin (B / 2)) + (1 / Real.sin (C / 2)) :=
by sorry

end acute_triangle_sec_csc_inequality_l199_199012


namespace value_of_4m_plus_2n_l199_199594

-- Given that the equation 2kx + 2m = 6 - 2x + nk 
-- has a solution independent of k
theorem value_of_4m_plus_2n (m n : ℝ) 
  (h : ∃ x : ℝ, ∀ k : ℝ, 2 * k * x + 2 * m = 6 - 2 * x + n * k) : 
  4 * m + 2 * n = 12 :=
by
  sorry

end value_of_4m_plus_2n_l199_199594


namespace guides_tourists_grouping_l199_199053

theorem guides_tourists_grouping (tourists : ℕ) (guides : ℕ) (h_t : tourists = 6) (h_g : guides = 2) :
  (∑ k in finset.Ico 1 tourists, nat.choose tourists k) = 62 :=
by
  rw [h_t, h_g]
  -- usual steps to prove the sum, currently omitted
  sorry

end guides_tourists_grouping_l199_199053


namespace brother_raking_time_l199_199455

theorem brother_raking_time (x : ℝ) (hx : x > 0)
  (h_combined : (1 / 30) + (1 / x) = 1 / 18) : x = 45 :=
by
  sorry

end brother_raking_time_l199_199455


namespace range_of_m_l199_199744

theorem range_of_m (x y m : ℝ) (h1 : 2 / x + 1 / y = 1) (h2 : x + y = 2 + 2 * m) : -4 < m ∧ m < 2 :=
sorry

end range_of_m_l199_199744


namespace cross_section_area_l199_199295

open Real

theorem cross_section_area (b α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ (area : ℝ), area = - (b^2 * cos α * tan β) / (2 * cos (3 * α)) :=
by
  sorry

end cross_section_area_l199_199295


namespace board_election_ways_l199_199888

theorem board_election_ways :
  let total_ways := Nat.choose 20 6 in
  let ways_no_prev_serv := Nat.choose 11 6 in
  total_ways - ways_no_prev_serv = 38298 :=
by
  let total_ways := Nat.choose 20 6
  let ways_no_prev_serv := Nat.choose 11 6
  show total_ways - ways_no_prev_serv = 38298
  sorry

end board_election_ways_l199_199888


namespace least_positive_number_of_24x_plus_16y_is_8_l199_199950

theorem least_positive_number_of_24x_plus_16y_is_8 :
  ∃ (x y : ℤ), 24 * x + 16 * y = 8 :=
by
  sorry

end least_positive_number_of_24x_plus_16y_is_8_l199_199950


namespace ricardo_coin_difference_l199_199982

theorem ricardo_coin_difference (p : ℕ) (h1 : 1 ≤ p) (h2 : p ≤ 2299) :
  (11500 - 4 * p) - (11500 - 4 * (2300 - p)) = 9192 :=
by
  sorry

end ricardo_coin_difference_l199_199982


namespace value_of_m_l199_199943

theorem value_of_m (m : ℚ) : 
  (m = - -(-(1/3) : ℚ) → m = -1/3) :=
by
  sorry

end value_of_m_l199_199943


namespace b4_lt_b7_l199_199093

noncomputable def b : ℕ → ℝ
| 1       := 1 + 1 / α 1
| (n + 1) := 1 + 1 / (α 1 + b n)

theorem b4_lt_b7 (α : ℕ → ℝ) (hα : ∀ k, α k > 0) : b α 4 < b α 7 :=
by { sorry }

end b4_lt_b7_l199_199093


namespace count_valid_b_values_l199_199898

-- Definitions of the inequalities and the condition
def inequality1 (x : ℤ) : Prop := 3 * x > 4 * x - 4
def inequality2 (x b: ℤ) : Prop := 4 * x - b > -8

-- The main statement proving that the count of valid b values is 4
theorem count_valid_b_values (x b : ℤ) (h1 : inequality1 x) (h2 : inequality2 x b) :
  ∃ (b_values : Finset ℤ), 
    ((∀ b' ∈ b_values, ∀ x' : ℤ, inequality2 x' b' → x' ≠ 3) ∧ 
     (∀ b' ∈ b_values, 16 ≤ b' ∧ b' < 20) ∧ 
     b_values.card = 4) := by
  sorry

end count_valid_b_values_l199_199898


namespace algebraic_expression_value_l199_199586

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l199_199586


namespace positive_multiples_of_4_with_units_digit_4_l199_199120

theorem positive_multiples_of_4_with_units_digit_4 (n : ℕ) : 
  ∃ n ≤ 15, ∀ m, m = 4 + 10 * (n - 1) → m < 150 ∧ m % 10 = 4 :=
by {
  sorry
}

end positive_multiples_of_4_with_units_digit_4_l199_199120


namespace total_settings_weight_l199_199473

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l199_199473


namespace number_of_real_solutions_l199_199602

theorem number_of_real_solutions :
  (∃ (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) →
  (∃! (x : ℝ), (x ^ 2 + x - 12) ^ 2 = 81) :=
sorry

end number_of_real_solutions_l199_199602


namespace total_salmons_caught_l199_199424

theorem total_salmons_caught :
  let hazel_salmons := 24
  let dad_salmons := 27
  hazel_salmons + dad_salmons = 51 :=
by
  sorry

end total_salmons_caught_l199_199424


namespace find_m_plus_t_l199_199730

-- Define the system of equations represented by the augmented matrix
def equation1 (m t : ℝ) : Prop := 3 * m - t = 22
def equation2 (t : ℝ) : Prop := t = 2

-- State the main theorem with the given conditions and the goal
theorem find_m_plus_t (m t : ℝ) (h1 : equation1 m t) (h2 : equation2 t) : m + t = 10 := 
by
  sorry

end find_m_plus_t_l199_199730


namespace number_of_common_tangents_between_circleC_and_circleD_l199_199394

noncomputable def circleC := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

noncomputable def circleD := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 - 4 = 0 }

theorem number_of_common_tangents_between_circleC_and_circleD : 
    ∃ (num_tangents : ℕ), num_tangents = 2 :=
by
    -- Proving the number of common tangents is 2
    sorry

end number_of_common_tangents_between_circleC_and_circleD_l199_199394


namespace pow_two_grows_faster_than_square_l199_199409

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square_l199_199409


namespace two_digits_same_in_three_digit_numbers_l199_199754

theorem two_digits_same_in_three_digit_numbers (h1 : (100 : ℕ) ≤ n) (h2 : n < 600) : 
  ∃ n, n = 140 := sorry

end two_digits_same_in_three_digit_numbers_l199_199754


namespace polynomial_roots_l199_199907

open Polynomial

theorem polynomial_roots :
  (roots (C 1 * X^4 + C (-3) * X^3 + C 3 * X^2 + C (-1) * X + C (-6))).map (λ x, x.re) = 
    {1 - sqrt 3, 1 + sqrt 3, (1 - sqrt 13) / 2, (1 + sqrt 13) / 2} :=
by sorry

end polynomial_roots_l199_199907


namespace intersection_of_PQ_RS_correct_l199_199960

noncomputable def intersection_point (P Q R S : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let t := 1/9
  let s := 2/3
  (3 + 10 * t, -4 - 10 * t, 4 + 5 * t)

theorem intersection_of_PQ_RS_correct :
  let P := (3, -4, 4)
  let Q := (13, -14, 9)
  let R := (-3, 6, -9)
  let S := (1, -2, 7)
  intersection_point P Q R S = (40/9, -76/9, 49/9) :=
by {
  sorry
}

end intersection_of_PQ_RS_correct_l199_199960


namespace find_a_l199_199726

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }
def setB : Set ℝ := { x | Real.log (x^2 - 5 * x + 8) / Real.log 2 = 1 }
def setC (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }

-- Proof statement to find the value of a
theorem find_a (a : ℝ) : setA ∩ setC a = ∅ → setB ∩ setC a ≠ ∅ → a = -2 := by
  sorry

end find_a_l199_199726


namespace sarah_initial_money_l199_199488

-- Definitions based on conditions
def cost_toy_car := 11
def cost_scarf := 10
def cost_beanie := 14
def remaining_money := 7
def total_cost := 2 * cost_toy_car + cost_scarf + cost_beanie
def initial_money := total_cost + remaining_money

-- Statement of the theorem
theorem sarah_initial_money : initial_money = 53 :=
by
  sorry

end sarah_initial_money_l199_199488


namespace solve_inequality_l199_199711

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 2) / (x - 1)

theorem solve_inequality : 
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x < 1 } ∪ { x : ℝ | x ≥ 2 } :=
by
  sorry

end solve_inequality_l199_199711


namespace probability_more_ones_than_sixes_l199_199776

theorem probability_more_ones_than_sixes :
  let num_faces := 6 in
  let num_rolls := 5 in
  let total_outcomes := num_faces ^ num_rolls in
  let favorable_outcomes := 2711 in
  (favorable_outcomes : ℚ) / total_outcomes = 2711 / 7776 :=
sorry

end probability_more_ones_than_sixes_l199_199776


namespace last_two_digits_of_binom_200_100_l199_199903

open Nat

theorem last_two_digits_of_binom_200_100 :
  (nat.binomial 200 100) % 100 = 20 :=
by
  have h_mod_4 : (nat.binomial 200 100) % 4 = 0 := sorry
  have h_mod_25 : (nat.binomial 200 100) % 25 = 20 := sorry
  exact Nat.modeq.chinese_remainder h_mod_4 h_mod_25 sorry

end last_two_digits_of_binom_200_100_l199_199903


namespace α_plus_β_eq_two_l199_199412

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

theorem α_plus_β_eq_two
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 := 
sorry

end α_plus_β_eq_two_l199_199412


namespace pencils_in_drawer_l199_199312

theorem pencils_in_drawer (P : ℕ) (h1 : P + 19 + 16 = 78) : P = 43 :=
by
  sorry

end pencils_in_drawer_l199_199312


namespace tan_angle_addition_l199_199430

theorem tan_angle_addition (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 :=
sorry

end tan_angle_addition_l199_199430


namespace loaves_of_bread_l199_199808

-- Definitions for the given conditions
def total_flour : ℝ := 5
def flour_per_loaf : ℝ := 2.5

-- The statement of the problem
theorem loaves_of_bread (total_flour : ℝ) (flour_per_loaf : ℝ) : 
  total_flour / flour_per_loaf = 2 :=
by
  -- Proof is not required
  sorry

end loaves_of_bread_l199_199808


namespace total_percent_decrease_l199_199179

theorem total_percent_decrease (initial_value : ℝ) (val1 val2 : ℝ) :
  initial_value > 0 →
  val1 = initial_value * (1 - 0.60) →
  val2 = val1 * (1 - 0.10) →
  (initial_value - val2) / initial_value * 100 = 64 :=
by
  intros h_initial h_val1 h_val2
  sorry

end total_percent_decrease_l199_199179


namespace cori_age_l199_199896

theorem cori_age (C A : ℕ) (hA : A = 19) (hEq : C + 5 = (A + 5) / 3) : C = 3 := by
  rw [hA] at hEq
  norm_num at hEq
  linarith

end cori_age_l199_199896


namespace evaluate_expression_l199_199588

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l199_199588


namespace seq_contains_exactly_16_twos_l199_199346

-- Define a helper function to count occurrences of a digit in a number
def count_digit (d : Nat) (n : Nat) : Nat :=
  (n.digits 10).count d

-- Define a function to sum occurrences of the digit '2' in a list of numbers
def total_twos_in_sequence (seq : List Nat) : Nat :=
  seq.foldl (λ acc n => acc + count_digit 2 n) 0

-- Define the sequence we are interested in
def seq : List Nat := [2215, 2216, 2217, 2218, 2219, 2220, 2221]

-- State the theorem we need to prove
theorem seq_contains_exactly_16_twos : total_twos_in_sequence seq = 16 := 
by
  -- We do not provide the proof here according to the given instructions
  sorry

end seq_contains_exactly_16_twos_l199_199346


namespace area_triangle_CIN_l199_199186

variables (A B C D M N I : Type*)

-- Definitions and assumptions
-- ABCD is a square
def is_square (ABCD : Type*) (side : ℝ) : Prop := sorry
-- M is the midpoint of AB
def midpoint_AB (M A B : Type*) : Prop := sorry
-- N is the midpoint of BC
def midpoint_BC (N B C : Type*) : Prop := sorry
-- Lines CM and DN intersect at I
def lines_intersect_at (C M D N I : Type*) : Prop := sorry

-- Goal
theorem area_triangle_CIN (ABCD : Type*) (side : ℝ) (M N C I : Type*) 
  (h1 : is_square ABCD side)
  (h2 : midpoint_AB M A B)
  (h3 : midpoint_BC N B C)
  (h4 : lines_intersect_at C M D N I) :
  sorry := sorry

end area_triangle_CIN_l199_199186


namespace ellipse_chord_slope_relation_l199_199624

theorem ellipse_chord_slope_relation
    (a b : ℝ) (h : a > b) (h1 : b > 0)
    (A B M : ℝ × ℝ)
    (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (hAB_slope : A.1 ≠ B.1)
    (K_AB K_OM : ℝ)
    (hK_AB : K_AB = (B.2 - A.2) / (B.1 - A.1))
    (hK_OM : K_OM = (M.2 - 0) / (M.1 - 0)) :
  K_AB * K_OM = - (b ^ 2) / (a ^ 2) := 
  sorry

end ellipse_chord_slope_relation_l199_199624


namespace repeating_decimal_eq_fraction_l199_199109

noncomputable def repeating_decimal_to_fraction (x : ℝ) (h : x = 2.353535...) : ℝ :=
  233 / 99

theorem repeating_decimal_eq_fraction :
  (∃ x : ℝ, x = 2.353535... ∧ x = repeating_decimal_to_fraction x (by sorry)) :=
begin
  use 2.353535...,
  split,
  { exact rfl },
  { have h : 2.353535... = 233 / 99, by sorry,
    exact h, }
end

end repeating_decimal_eq_fraction_l199_199109


namespace emily_curtains_purchase_l199_199553

theorem emily_curtains_purchase 
    (c : ℕ) 
    (curtain_cost : ℕ := 30)
    (print_count : ℕ := 9)
    (print_cost_per_unit : ℕ := 15)
    (installation_cost : ℕ := 50)
    (total_cost : ℕ := 245) :
    (curtain_cost * c + print_count * print_cost_per_unit + installation_cost = total_cost) → c = 2 :=
by
  sorry

end emily_curtains_purchase_l199_199553


namespace cube_surface_area_calc_l199_199301

-- Edge length of the cube
def edge_length : ℝ := 7

-- Definition of the surface area formula for a cube
def surface_area (a : ℝ) : ℝ := 6 * (a ^ 2)

-- The main theorem stating the surface area of the cube with given edge length
theorem cube_surface_area_calc : surface_area edge_length = 294 :=
by
  sorry

end cube_surface_area_calc_l199_199301


namespace perfect_square_trinomial_l199_199325

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l199_199325


namespace first_term_of_infinite_geometric_series_l199_199687

theorem first_term_of_infinite_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := 
sorry

end first_term_of_infinite_geometric_series_l199_199687


namespace c_investment_l199_199669

theorem c_investment (x : ℝ) (h1 : 5000 / (5000 + 8000 + x) * 88000 = 36000) : 
  x = 20454.5 :=
by
  sorry

end c_investment_l199_199669


namespace prob_not_all_same_correct_l199_199066

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l199_199066


namespace problem_statement_l199_199718

theorem problem_statement (a : ℤ)
  (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a) ^ 2 + (2004 - a) ^ 2 = 4014 :=
sorry

end problem_statement_l199_199718


namespace area_y_eq_x2_y_eq_x3_l199_199294

noncomputable section

open Real

def area_closed_figure_between_curves : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (x^2 - x^3)

theorem area_y_eq_x2_y_eq_x3 :
  area_closed_figure_between_curves = 1 / 12 := by
  sorry

end area_y_eq_x2_y_eq_x3_l199_199294


namespace upgrade_days_to_sun_l199_199284

/-- 
  Determine the minimum number of additional active days required for 
  a user currently at level 2 moons and 1 star to upgrade to 1 sun.
-/
theorem upgrade_days_to_sun (level_new_star : ℕ) (level_new_moon : ℕ) (active_days_initial : ℕ) : 
  active_days_initial =  9 * (9 + 4) → 
  level_new_star = 1 → 
  level_new_moon = 2 → 
  ∃ (days_required : ℕ), 
    (days_required + active_days_initial = 16 * (16 + 4)) ∧ (days_required = 203) :=
by
  sorry

end upgrade_days_to_sun_l199_199284


namespace binomial_square_evaluation_l199_199331

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l199_199331


namespace not_possible_arrangement_l199_199454

theorem not_possible_arrangement : 
  ¬ ∃ (f : Fin 4026 → Fin 2014), 
    (∀ k : Fin 2014, ∃ i j : Fin 4026, i < j ∧ f i = k ∧ f j = k ∧ (j.val - i.val - 1) = k.val) :=
sorry

end not_possible_arrangement_l199_199454


namespace smaller_solution_of_quadratic_l199_199069

theorem smaller_solution_of_quadratic :
  ∀ x : ℝ, x^2 + 17 * x - 72 = 0 → x = -24 ∨ x = 3 :=
by sorry

end smaller_solution_of_quadratic_l199_199069


namespace probability_more_ones_than_sixes_l199_199770

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end probability_more_ones_than_sixes_l199_199770


namespace derivative_at_2_l199_199991

noncomputable def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

theorem derivative_at_2 : deriv f 2 = 15 := by
  sorry

end derivative_at_2_l199_199991


namespace min_value_of_vectors_l199_199746

theorem min_value_of_vectors (m n : ℝ) (h1 : m > 0) (h2 : n > 0) 
  (h3 : (m * (n - 2)) + 1 = 0) : (1 / m) + (2 / n) = 2 * Real.sqrt 2 + 3 / 2 :=
by sorry

end min_value_of_vectors_l199_199746


namespace students_math_inequality_l199_199200

variables {n x a b c : ℕ}

theorem students_math_inequality (h1 : x + a ≥ 8 * n / 10) 
                                (h2 : x + b ≥ 8 * n / 10) 
                                (h3 : n ≥ a + b + c + x) : 
                                x * 5 ≥ 4 * (x + c) :=
by
  sorry

end students_math_inequality_l199_199200


namespace find_x_l199_199433

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : 
  x = 3 :=
sorry

end find_x_l199_199433


namespace number_of_ways_to_sign_up_probability_student_A_online_journalists_l199_199260

-- Definitions for the conditions
def students : Finset String := {"A", "B", "C", "D", "E"}
def projects : Finset String := {"Online Journalists", "Robot Action", "Sounds of Music"}

-- Function to calculate combinations (nCr)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate arrangements
def arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Proof opportunity for part 1
theorem number_of_ways_to_sign_up : 
  (combinations 5 3 * arrangements 3) + ((combinations 5 2 * combinations 3 2) / arrangements 2 * arrangements 3) = 150 :=
sorry

-- Proof opportunity for part 2
theorem probability_student_A_online_journalists
  (h : (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 243) : 
  ((combinations 4 3 * arrangements 2) * projects.card ^ 3) / 
  (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 1 / 15 :=
sorry

end number_of_ways_to_sign_up_probability_student_A_online_journalists_l199_199260


namespace quadratic_solution_l199_199492

theorem quadratic_solution (x : ℝ) : (x^2 + 6 * x + 8 = -2 * (x + 4) * (x + 5)) ↔ (x = -8 ∨ x = -4) :=
by
  sorry

end quadratic_solution_l199_199492


namespace heather_blocks_l199_199425

theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 86 → shared_blocks = 41 → remaining_blocks = initial_blocks - shared_blocks → remaining_blocks = 45 :=
by
  sorry

end heather_blocks_l199_199425


namespace number_of_divisors_of_2018_or_2019_is_7_l199_199939

theorem number_of_divisors_of_2018_or_2019_is_7 (h1 : Prime 673) (h2 : Prime 1009) : 
  Nat.card {d : Nat | d ∣ 2018 ∨ d ∣ 2019} = 7 := 
  sorry

end number_of_divisors_of_2018_or_2019_is_7_l199_199939


namespace gcd_of_g_y_and_y_l199_199593

theorem gcd_of_g_y_and_y (y : ℤ) (h : 9240 ∣ y) : Int.gcd ((5 * y + 3) * (11 * y + 2) * (17 * y + 8) * (4 * y + 7)) y = 168 := by
  sorry

end gcd_of_g_y_and_y_l199_199593


namespace remainder_sum_of_first_eight_primes_div_tenth_prime_l199_199068

theorem remainder_sum_of_first_eight_primes_div_tenth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) % 29 = 19 :=
by norm_num

end remainder_sum_of_first_eight_primes_div_tenth_prime_l199_199068


namespace min_sum_of_factors_l199_199498

theorem min_sum_of_factors (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 3432) :
  a + b + c ≥ 56 :=
sorry

end min_sum_of_factors_l199_199498


namespace simplify_and_evaluate_l199_199636

theorem simplify_and_evaluate (a : ℝ) (h : a = -3 / 2) : 
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := 
by 
  sorry

end simplify_and_evaluate_l199_199636


namespace original_cards_l199_199016

-- Define the number of cards Jason gave away
def cards_given_away : ℕ := 9

-- Define the number of cards Jason now has
def cards_now : ℕ := 4

-- Prove the original number of Pokemon cards Jason had
theorem original_cards (x : ℕ) : x = cards_given_away + cards_now → x = 13 :=
by {
    sorry
}

end original_cards_l199_199016


namespace least_multiple_greater_than_500_l199_199057

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 0 ∧ 35 * n > 500 ∧ 35 * n = 525 :=
by
  sorry

end least_multiple_greater_than_500_l199_199057


namespace curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l199_199420

noncomputable def curve_C (a x y : ℝ) := a * x ^ 2 + a * y ^ 2 - 2 * x - 2 * y = 0

theorem curve_C_straight_line (a : ℝ) : a = 0 → ∃ x y : ℝ, curve_C a x y :=
by
  intro ha
  use (-1), 1
  rw [curve_C, ha]
  simp

theorem curve_C_not_tangent (a : ℝ) : a = 1 → ¬ ∀ x y, 3 * x + y = 0 → curve_C a x y :=
by
  sorry

theorem curve_C_fixed_point (x y a : ℝ) : curve_C a 0 0 :=
by
  rw [curve_C]
  simp

theorem curve_C_intersect (a : ℝ) : a = 1 → ∃ x y : ℝ, (x + 2 * y = 0) ∧ curve_C a x y :=
by
  sorry

end curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l199_199420


namespace curve_tangents_intersection_l199_199931

theorem curve_tangents_intersection (a : ℝ) :
  (∃ x₀ y₀, y₀ = Real.exp x₀ ∧ y₀ = (x₀ + a)^2 ∧ Real.exp x₀ = 2 * (x₀ + a)) → a = 2 - Real.log 4 :=
by
  sorry

end curve_tangents_intersection_l199_199931


namespace kath_movie_cost_l199_199957

theorem kath_movie_cost :
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  discounted_price * number_of_people = 30 := by
  -- Definitions from conditions
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  -- Derived calculation based on conditions
  have h_discounted_price : discounted_price = 5 := by
    calc
      discounted_price = 8 - 3 : by sorry
      ... = 5 : by sorry
  have h_number_of_people : number_of_people = 6 := by
    calc
      number_of_people = 1 + 2 + 3 : by sorry
      ... = 6 : by sorry
  show 5 * 6 = 30 from sorry

end kath_movie_cost_l199_199957


namespace rectangle_area_increase_l199_199874

theorem rectangle_area_increase (a b : ℝ) :
  let new_length := (1 + 1/4) * a
  let new_width := (1 + 1/5) * b
  let original_area := a * b
  let new_area := new_length * new_width
  let area_increase := new_area - original_area
  (area_increase / original_area) = 1/2 := 
by
  sorry

end rectangle_area_increase_l199_199874


namespace transformed_curve_eq_l199_199161

/-- Given the initial curve equation and the scaling transformation,
    prove that the resulting curve has the transformed equation. -/
theorem transformed_curve_eq 
  (x y x' y' : ℝ)
  (h_curve : x^2 + 9*y^2 = 9)
  (h_transform_x : x' = x)
  (h_transform_y : y' = 3*y) :
  (x')^2 + y'^2 = 9 := 
sorry

end transformed_curve_eq_l199_199161


namespace marked_price_correct_l199_199990

noncomputable def marked_price (cost_price : ℝ) (profit_margin : ℝ) (selling_percentage : ℝ) : ℝ :=
  (cost_price * (1 + profit_margin)) / selling_percentage

theorem marked_price_correct :
  marked_price 1360 0.15 0.8 = 1955 :=
by
  sorry

end marked_price_correct_l199_199990


namespace maximum_garden_area_l199_199019

theorem maximum_garden_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 400) : 
  l * w ≤ 10000 :=
by {
  -- proving the theorem
  sorry
}

end maximum_garden_area_l199_199019


namespace find_m_value_l199_199517

def magic_box (a b : ℝ) : ℝ := a^2 + 2 * b - 3

theorem find_m_value (m : ℝ) :
  magic_box m (-3 * m) = 4 ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end find_m_value_l199_199517


namespace nth_number_in_S_l199_199275

def S : Set ℕ := {n | ∃ k : ℕ, n = 15 * k + 11}

theorem nth_number_in_S (n : ℕ) (hn : n = 127) : ∃ k, 15 * k + 11 = 1901 :=
by
  sorry

end nth_number_in_S_l199_199275


namespace journey_speed_condition_l199_199677

theorem journey_speed_condition (v : ℝ) :
  (10 : ℝ) = 112 / v + 112 / 24 → (224 / 2 = 112) → v = 21 := by
  intros
  apply sorry

end journey_speed_condition_l199_199677


namespace find_x_l199_199525

theorem find_x (x : ℝ) (h : 0.65 * x = 0.2 * 617.50) : x = 190 :=
by
  sorry

end find_x_l199_199525


namespace factor_correct_l199_199555

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l199_199555


namespace trig_identity_l199_199928

variable {α : Real}

theorem trig_identity (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 := 
by
  sorry

end trig_identity_l199_199928


namespace inverse_of_true_implies_negation_true_l199_199609

variable (P : Prop)
theorem inverse_of_true_implies_negation_true (h : ¬ P) : ¬ P :=
by 
  exact h

end inverse_of_true_implies_negation_true_l199_199609


namespace benny_spent_on_baseball_gear_l199_199386

theorem benny_spent_on_baseball_gear (initial_amount left_over spent : ℕ) 
  (h_initial : initial_amount = 67) 
  (h_left : left_over = 33) 
  (h_spent : spent = initial_amount - left_over) : 
  spent = 34 :=
by
  rw [h_initial, h_left] at h_spent
  exact h_spent

end benny_spent_on_baseball_gear_l199_199386


namespace correct_substitution_l199_199522

theorem correct_substitution (x : ℝ) : 
    (2 * x - 7)^2 + (5 * x - 17.5)^2 = 0 → 
    x = 7 / 2 :=
by
  sorry

end correct_substitution_l199_199522


namespace ratio_D_E_equal_l199_199972

variable (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ)

def mary_story_conditions (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) : Prop :=
  total_characters = 60 ∧
  initial_A = 1 / 2 * total_characters ∧
  initial_C = 1 / 2 * initial_A ∧
  initial_D + initial_E = total_characters - (initial_A + initial_C)

theorem ratio_D_E_equal (total_characters initial_A initial_C initial_D initial_E : ℕ) :
  mary_story_conditions total_characters initial_A initial_C initial_D initial_E →
  initial_D = initial_E :=
sorry

end ratio_D_E_equal_l199_199972


namespace max_value_expression_l199_199216

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l199_199216


namespace height_difference_l199_199371

variable (h_A h_B h_D h_E h_F h_G : ℝ)

theorem height_difference :
  (h_A - h_D = 4.5) →
  (h_E - h_D = -1.7) →
  (h_F - h_E = -0.8) →
  (h_G - h_F = 1.9) →
  (h_B - h_G = 3.6) →
  (h_A - h_B > 0) :=
by
  intro h_AD h_ED h_FE h_GF h_BG
  sorry

end height_difference_l199_199371


namespace evaluate_expression_l199_199140

noncomputable def complex_numbers_condition (a b : ℂ) := a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + a * b + b^2 = 0)

theorem evaluate_expression (a b : ℂ) (h : complex_numbers_condition a b) : 
  (a^5 + b^5) / (a + b)^5 = -2 := 
by
  sorry

end evaluate_expression_l199_199140


namespace product_of_ages_l199_199992

theorem product_of_ages (O Y : ℕ) (h1 : O - Y = 12) (h2 : O + Y = (O - Y) + 40) : O * Y = 640 := by
  sorry

end product_of_ages_l199_199992


namespace fixed_point_always_l199_199995

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always_l199_199995


namespace q_value_l199_199241

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l199_199241


namespace algebraic_expression_value_l199_199585

theorem algebraic_expression_value (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 := 
by
  sorry

end algebraic_expression_value_l199_199585


namespace prob_2_lt_X_lt_4_l199_199044

noncomputable def normal_dist_p (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this computes the CDF at x for a normal distribution

variable {X : ℝ → ℝ}
variable {σ : ℝ}

-- Condition: X follows a normal distribution with mean 3 and variance σ^2
axiom normal_distribution_X : ∀ x, X x = normal_dist_p 3 σ x

-- Condition: P(X ≤ 4) = 0.84
axiom prob_X_leq_4 : normal_dist_p 3 σ 4 = 0.84

-- Goal: Prove P(2 < X < 4) = 0.68
theorem prob_2_lt_X_lt_4 : normal_dist_p 3 σ 4 - normal_dist_p 3 σ 2 = 0.68 := by
  sorry

end prob_2_lt_X_lt_4_l199_199044


namespace find_line_equation_through_point_intersecting_hyperbola_l199_199304

theorem find_line_equation_through_point_intersecting_hyperbola 
  (x y : ℝ) 
  (hx : x = -2 / 3)
  (hy : (x : ℝ) = 0) : 
  ∃ k : ℝ, (∀ x y : ℝ, y = k * x - 1 → ((x^2 / 2) - (y^2 / 5) = 1)) ∧ k = 1 := 
sorry

end find_line_equation_through_point_intersecting_hyperbola_l199_199304


namespace find_length_of_bridge_l199_199876

noncomputable def length_of_train : ℝ := 165
noncomputable def speed_of_train_kmph : ℝ := 54
noncomputable def time_to_cross_bridge_seconds : ℝ := 67.66125376636536

noncomputable def speed_of_train_mps : ℝ :=
  speed_of_train_kmph * (1000 / 3600)

noncomputable def total_distance_covered : ℝ :=
  speed_of_train_mps * time_to_cross_bridge_seconds

noncomputable def length_of_bridge : ℝ :=
  total_distance_covered - length_of_train

theorem find_length_of_bridge : length_of_bridge = 849.92 := by
  sorry

end find_length_of_bridge_l199_199876


namespace next_meeting_day_l199_199622

-- Definitions of visit periods
def jia_visit_period : ℕ := 6
def yi_visit_period : ℕ := 8
def bing_visit_period : ℕ := 9

-- August 17th
def initial_meeting_day : ℕ := 17

-- Prove that they will meet again 72 days after August 17th
theorem next_meeting_day : ∃ n : ℕ, n = 72 ∧ (∀ m : ℕ, (m = jia_visit_period ∨
m = yi_visit_period ∨ m = bing_visit_period) → ∃ k : ℕ, initial_meeting_day + 72 = m * k) :=
by
  sorry

end next_meeting_day_l199_199622


namespace action_figure_ratio_l199_199881

variable (initial : ℕ) (sold : ℕ) (remaining : ℕ) (left : ℕ)
variable (h1 : initial = 24)
variable (h2 : sold = initial / 4)
variable (h3 : remaining = initial - sold)
variable (h4 : remaining - left = left)

theorem action_figure_ratio
  (h1 : initial = 24)
  (h2 : sold = initial / 4)
  (h3 : remaining = initial - sold)
  (h4 : remaining - left = left) :
  (remaining - left) * 3 = left :=
by
  sorry

end action_figure_ratio_l199_199881


namespace first_new_player_weight_l199_199311

theorem first_new_player_weight (x : ℝ) :
  (7 * 103) + x + 60 = 9 * 99 → 
  x = 110 := by
  sorry

end first_new_player_weight_l199_199311


namespace emily_orange_count_l199_199212

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l199_199212


namespace positive_integers_satisfying_inequality_l199_199305

theorem positive_integers_satisfying_inequality (x : ℕ) (hx : x > 0) : 4 - x > 1 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end positive_integers_satisfying_inequality_l199_199305


namespace jake_has_3_peaches_l199_199136

-- Define the number of peaches Steven has.
def steven_peaches : ℕ := 13

-- Define the number of peaches Jake has based on the condition.
def jake_peaches (P_S : ℕ) : ℕ := P_S - 10

-- The theorem that states Jake has 3 peaches.
theorem jake_has_3_peaches : jake_peaches steven_peaches = 3 := sorry

end jake_has_3_peaches_l199_199136


namespace bicycle_helmet_savings_l199_199045

theorem bicycle_helmet_savings :
  let bicycle_regular_price := 320
  let bicycle_discount := 0.2
  let helmet_regular_price := 80
  let helmet_discount := 0.1
  let bicycle_savings := bicycle_regular_price * bicycle_discount
  let helmet_savings := helmet_regular_price * helmet_discount
  let total_savings := bicycle_savings + helmet_savings
  let total_regular_price := bicycle_regular_price + helmet_regular_price
  let percentage_savings := (total_savings / total_regular_price) * 100
  percentage_savings = 18 := 
by sorry

end bicycle_helmet_savings_l199_199045


namespace fifth_number_in_tenth_row_l199_199529

def nth_number_in_row (n k : ℕ) : ℕ :=
  7 * n - (7 - k)

theorem fifth_number_in_tenth_row : nth_number_in_row 10 5 = 68 :=
by
  sorry

end fifth_number_in_tenth_row_l199_199529


namespace probability_indep_seq_limit_one_l199_199148

noncomputable def i_o (A : ℕ → Set Ω) : Set Ω :=
  ⋂ k, ⋃ n (hn : k ≤ n), A n

theorem probability_indep_seq_limit_one {Ω : Type*} {A : ℕ → Set Ω} [MeasureSpace Ω] (P : Measure Ω)
  (indep : IndepEvents A P) (h : ∀ n, P (A n) < 1) :
  P (i_o A) = 1 ↔ P (⋃ n, A n) = 1 := 
by
  sorry

end probability_indep_seq_limit_one_l199_199148


namespace point_in_or_on_circle_l199_199010

theorem point_in_or_on_circle (θ : Real) :
  let P := (5 * Real.cos θ, 4 * Real.sin θ)
  let C_eq := ∀ (x y : Real), x^2 + y^2 = 25
  25 * Real.cos θ ^ 2 + 16 * Real.sin θ ^ 2 ≤ 25 := 
by 
  sorry

end point_in_or_on_circle_l199_199010


namespace feeding_amount_per_horse_per_feeding_l199_199459

-- Define the conditions as constants
def num_horses : ℕ := 25
def feedings_per_day : ℕ := 2
def half_ton_in_pounds : ℕ := 1000
def bags_needed : ℕ := 60
def days : ℕ := 60

-- Statement of the problem
theorem feeding_amount_per_horse_per_feeding :
  (bags_needed * half_ton_in_pounds / days / feedings_per_day) / num_horses = 20 := by
  -- Assume conditions are satisfied
  sorry

end feeding_amount_per_horse_per_feeding_l199_199459


namespace not_all_same_probability_l199_199061

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l199_199061


namespace find_missing_figure_l199_199077

theorem find_missing_figure (x : ℝ) (h : 0.003 * x = 0.15) : x = 50 :=
sorry

end find_missing_figure_l199_199077


namespace francie_remaining_money_l199_199913

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end francie_remaining_money_l199_199913


namespace geometric_sequence_sum_l199_199266

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum_l199_199266


namespace hours_per_day_l199_199095

-- Conditions
def days_worked : ℝ := 3
def total_hours_worked : ℝ := 7.5

-- Theorem to prove the number of hours worked each day
theorem hours_per_day : total_hours_worked / days_worked = 2.5 :=
by
  sorry

end hours_per_day_l199_199095


namespace botanical_garden_unique_plants_l199_199790

open Finset

theorem botanical_garden_unique_plants :
  (let A B C : Finset ℕ := 
     {n | n ∈ range 600} ∪ {m | m ∈ range 1200} ∩ {k | k ∈ range 700}, 
   count A = 600 
  ∧ count B = 500 
  ∧ count C = 400 
  ∧ count (A ∩ B) = 70 
  ∧ count (A ∩ C) = 120 
  ∧ count (B ∩ C) = 80 
  ∧ count (A ∩ B ∩ C) = 30 
  → count (A ∪ B ∪ C) = 1260) := sorry

end botanical_garden_unique_plants_l199_199790


namespace pencil_pen_cost_l199_199297

theorem pencil_pen_cost 
  (p q : ℝ) 
  (h1 : 6 * p + 3 * q = 3.90) 
  (h2 : 2 * p + 5 * q = 4.45) :
  3 * p + 4 * q = 3.92 :=
by
  sorry

end pencil_pen_cost_l199_199297


namespace motorbike_speed_l199_199372

noncomputable def speed_of_motorbike 
  (V_train : ℝ) 
  (t_overtake : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  V_train - (train_length_m / 1000) * (3600 / t_overtake)

theorem motorbike_speed : 
  speed_of_motorbike 100 80 800.064 = 63.99712 :=
by
  -- this is where the proof steps would go
  sorry

end motorbike_speed_l199_199372


namespace find_q_l199_199737

theorem find_q (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199737


namespace derivative_at_pi_over_2_l199_199236

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_pi_over_2 : 
  (deriv f (π / 2)) = Real.exp (π / 2) :=
by
  sorry

end derivative_at_pi_over_2_l199_199236


namespace interval_1_max_min_interval_2_max_min_l199_199237

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Prove max and min for interval [-2, 0]
theorem interval_1_max_min : 
  (∀ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x ≤ 10 ∧ f x ≥ 2 ∧ 
   (∃ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x = 10) ∧ (∃ x ∈ set.Icc (-2 : ℝ) (0 : ℝ), f x = 2)) :=
sorry

-- Prove max and min for interval [2, 3]
theorem interval_2_max_min : 
  (∀ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x ≤ 5 ∧ f x ≥ 2 ∧ 
   (∃ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x = 5) ∧ (∃ x ∈ set.Icc (2 : ℝ) (3 : ℝ), f x = 2)) :=
sorry

end interval_1_max_min_interval_2_max_min_l199_199237


namespace cricket_team_average_age_difference_l199_199296

theorem cricket_team_average_age_difference :
  let team_size := 11
  let captain_age := 26
  let keeper_age := captain_age + 3
  let avg_whole_team := 23
  let total_team_age := avg_whole_team * team_size
  let combined_age := captain_age + keeper_age
  let remaining_players := team_size - 2
  let total_remaining_age := total_team_age - combined_age
  let avg_remaining_players := total_remaining_age / remaining_players
  avg_whole_team - avg_remaining_players = 1 :=
by
  -- Proof omitted
  sorry

end cricket_team_average_age_difference_l199_199296


namespace hula_hoop_ratio_l199_199023

variable (Nancy Casey Morgan : ℕ)
variable (hula_hoop_time_Nancy : Nancy = 10)
variable (hula_hoop_time_Casey : Casey = Nancy - 3)
variable (hula_hoop_time_Morgan : Morgan = 21)

theorem hula_hoop_ratio (hula_hoop_time_Nancy : Nancy = 10) (hula_hoop_time_Casey : Casey = Nancy - 3) (hula_hoop_time_Morgan : Morgan = 21) :
  Morgan / Casey = 3 := by
  sorry

end hula_hoop_ratio_l199_199023


namespace crayons_left_l199_199025

-- Define the initial number of crayons
def initial_crayons : ℕ := 440

-- Define the crayons given away
def crayons_given : ℕ := 111

-- Define the crayons lost
def crayons_lost : ℕ := 106

-- Prove the final number of crayons left
theorem crayons_left : (initial_crayons - crayons_given - crayons_lost) = 223 :=
by
  sorry

end crayons_left_l199_199025


namespace solve_abc_l199_199034

theorem solve_abc (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : a + b + c = -1) (h3 : a * b + b * c + a * c = -4) (h4 : a * b * c = -2) :
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 :=
by
  -- Proof goes here
  sorry

end solve_abc_l199_199034


namespace num_partition_sets_correct_l199_199129

noncomputable def num_partition_sets (n : ℕ) : ℕ :=
  2^(n-1) - 1

theorem num_partition_sets_correct (n : ℕ) (hn : n ≥ 2) : 
  num_partition_sets n = 2^(n-1) - 1 := 
by sorry

end num_partition_sets_correct_l199_199129


namespace probability_XOXOXOX_l199_199580

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l199_199580


namespace find_n_eq_seven_l199_199819

theorem find_n_eq_seven (n : ℕ) (h : n ≥ 6) (H : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : n = 7 :=
by
  sorry

end find_n_eq_seven_l199_199819


namespace simplify_neg_x_mul_3_minus_x_l199_199392

theorem simplify_neg_x_mul_3_minus_x (x : ℝ) : -x * (3 - x) = -3 * x + x^2 :=
by
  sorry

end simplify_neg_x_mul_3_minus_x_l199_199392


namespace solve_for_x_l199_199704

theorem solve_for_x (a b c x : ℝ) (h : x^2 + b^2 + c = (a + x)^2) : 
  x = (b^2 + c - a^2) / (2 * a) :=
by sorry

end solve_for_x_l199_199704


namespace count_special_three_digit_numbers_l199_199748

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000
def is_less_than_600 (n : ℕ) := n < 600
def has_at_least_two_same_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

theorem count_special_three_digit_numbers :
  { n : ℕ | is_three_digit n ∧ is_less_than_600 n ∧ has_at_least_two_same_digits n }.to_finset.card = 140 :=
by
  sorry

end count_special_three_digit_numbers_l199_199748


namespace evaluate_expression_l199_199338

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l199_199338


namespace speedster_convertibles_l199_199086

noncomputable def total_inventory (not_speedsters : Nat) (fraction_not_speedsters : ℝ) : ℝ :=
  (not_speedsters : ℝ) / fraction_not_speedsters

noncomputable def number_speedsters (total_inventory : ℝ) (fraction_speedsters : ℝ) : ℝ :=
  total_inventory * fraction_speedsters

noncomputable def number_convertibles (number_speedsters : ℝ) (fraction_convertibles : ℝ) : ℝ :=
  number_speedsters * fraction_convertibles

theorem speedster_convertibles : (not_speedsters = 30) ∧ (fraction_not_speedsters = 2 / 3) ∧ (fraction_speedsters = 1 / 3) ∧ (fraction_convertibles = 4 / 5) →
  number_convertibles (number_speedsters (total_inventory not_speedsters fraction_not_speedsters) fraction_speedsters) fraction_convertibles = 12 :=
by
  intros h
  sorry

end speedster_convertibles_l199_199086


namespace bagel_spending_l199_199911

theorem bagel_spending (B D : ℝ) (h1 : D = 0.5 * B) (h2 : B = D + 15) : B + D = 45 := by
  sorry

end bagel_spending_l199_199911


namespace calculate_total_weight_l199_199470

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l199_199470


namespace perimeter_shaded_region_l199_199265

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def arc_length_per_circle (C : ℝ) : ℝ := C / 4

theorem perimeter_shaded_region (C : ℝ) (hC : C = 48) : 
  3 * arc_length_per_circle C = 36 := by
  sorry

end perimeter_shaded_region_l199_199265


namespace at_least_one_gt_one_l199_199584

theorem at_least_one_gt_one (x y : ℝ) (h : x + y > 2) : ¬(x > 1 ∨ y > 1) → (x ≤ 1 ∧ y ≤ 1) := 
by
  sorry

end at_least_one_gt_one_l199_199584


namespace minimize_expression_l199_199058

theorem minimize_expression (x y : ℝ) (k : ℝ) (h : k = -1) : (xy + k)^2 + (x - y)^2 ≥ 0 ∧ (∀ x y : ℝ, (xy + k)^2 + (x - y)^2 = 0 ↔ k = -1) := 
by {
  sorry
}

end minimize_expression_l199_199058


namespace exists_x_divisible_by_3n_not_by_3np1_l199_199796

noncomputable def f (x : ℕ) : ℕ := x ^ 3 + 17

theorem exists_x_divisible_by_3n_not_by_3np1 (n : ℕ) (hn : 2 ≤ n) : 
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬ (3^(n+1) ∣ f x) :=
sorry

end exists_x_divisible_by_3n_not_by_3np1_l199_199796


namespace net_effect_sale_value_l199_199253

variable (P Q : ℝ) -- New price and quantity sold

theorem net_effect_sale_value (P Q : ℝ) :
  let new_sale_value := (0.75 * P) * (1.75 * Q)
  let original_sale_value := P * Q
  new_sale_value - original_sale_value = 0.3125 * (P * Q) := 
by
  sorry

end net_effect_sale_value_l199_199253


namespace geometric_sequence_common_ratio_l199_199801

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  8 * geometric_sum a₁ q 6 = 7 * geometric_sum a₁ q 3 →
  q = -1/2 :=
by
  sorry

end geometric_sequence_common_ratio_l199_199801


namespace point_inside_circle_l199_199497

theorem point_inside_circle (a : ℝ) (h : 5 * a^2 - 4 * a - 1 < 0) : -1/5 < a ∧ a < 1 :=
    sorry

end point_inside_circle_l199_199497


namespace cube_root_59319_cube_root_103823_l199_199892

theorem cube_root_59319 : ∃ x : ℕ, x ^ 3 = 59319 ∧ x = 39 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

theorem cube_root_103823 : ∃ x : ℕ, x ^ 3 = 103823 ∧ x = 47 :=
by
  -- Sorry used to skip the proof, which is not required 
  sorry

end cube_root_59319_cube_root_103823_l199_199892


namespace matt_new_average_commission_l199_199151

noncomputable def new_average_commission (x : ℝ) : ℝ :=
  (5 * x + 1000) / 6

theorem matt_new_average_commission
  (x : ℝ)
  (h1 : (5 * x + 1000) / 6 = x + 150)
  (h2 : x = 100) :
  new_average_commission x = 250 :=
by
  sorry

end matt_new_average_commission_l199_199151


namespace remainder_mod7_l199_199910

theorem remainder_mod7 (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := 
by
  sorry

end remainder_mod7_l199_199910


namespace emily_orange_count_l199_199211

theorem emily_orange_count
  (betty_oranges : ℕ)
  (h1 : betty_oranges = 12)
  (sandra_oranges : ℕ)
  (h2 : sandra_oranges = 3 * betty_oranges)
  (emily_oranges : ℕ)
  (h3 : emily_oranges = 7 * sandra_oranges) :
  emily_oranges = 252 :=
by
  sorry

end emily_orange_count_l199_199211


namespace painting_time_l199_199552

-- Definitions based on the conditions
def num_people1 := 8
def num_houses1 := 3
def time1 := 12
def num_people2 := 9
def num_houses2 := 4
def k := (num_people1 * time1) / num_houses1

-- The statement we want to prove
theorem painting_time : (num_people2 * t = k * num_houses2) → (t = 128 / 9) :=
by sorry

end painting_time_l199_199552


namespace length_of_greater_segment_l199_199993

-- Definitions based on conditions
variable (shorter longer : ℝ)
variable (h1 : longer = shorter + 2)
variable (h2 : (longer^2) - (shorter^2) = 32)

-- Proof goal
theorem length_of_greater_segment : longer = 9 :=
by
  sorry

end length_of_greater_segment_l199_199993


namespace omar_rolls_l199_199631

-- Define the conditions
def karen_rolls : ℕ := 229
def total_rolls : ℕ := 448

-- Define the main theorem to prove the number of rolls by Omar
theorem omar_rolls : (total_rolls - karen_rolls) = 219 := by
  sorry

end omar_rolls_l199_199631


namespace correct_operation_l199_199667

theorem correct_operation (a b : ℝ) : 
  (2 * a) * (3 * a) = 6 * a^2 :=
by
  -- The proof would be here; using "sorry" to skip the actual proof steps.
  sorry

end correct_operation_l199_199667


namespace bananas_per_friend_l199_199137

theorem bananas_per_friend (total_bananas : ℤ) (total_friends : ℤ) (H1 : total_bananas = 21) (H2 : total_friends = 3) : 
  total_bananas / total_friends = 7 :=
by
  sorry

end bananas_per_friend_l199_199137


namespace math_problem_l199_199589

noncomputable def f : ℝ → ℝ := sorry

theorem math_problem (h_decreasing : ∀ x y : ℝ, 2 < x → x < y → f y < f x)
  (h_even : ∀ x : ℝ, f (-x + 2) = f (x + 2)) :
  f 2 < f 3 ∧ f 3 < f 0 ∧ f 0 < f (-1) :=
by
  sorry

end math_problem_l199_199589


namespace sugar_water_inequality_acute_triangle_inequality_l199_199728

-- Part 1: Proving the inequality \(\frac{a}{b} < \frac{a+m}{b+m}\)
theorem sugar_water_inequality (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
  a / b < (a + m) / (b + m) :=
by
  sorry

-- Part 2: Proving the inequality in an acute triangle \(\triangle ABC\)
theorem acute_triangle_inequality (A B C : ℝ) (hA : A < B + C) (hB : B < C + A) (hC : C < A + B) : 
  (A / (B + C)) + (B / (C + A)) + (C / (A + B)) < 2 :=
by
  sorry

end sugar_water_inequality_acute_triangle_inequality_l199_199728


namespace power_mod_condition_l199_199319

-- Defining the main problem conditions
theorem power_mod_condition (n: ℕ) : 
  (7^2 ≡ 1 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k+1) ≡ 7 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k) ≡ 1 [MOD 12]) →
  7^135 ≡ 7 [MOD 12] :=
by
  intros h1 h2 h3
  sorry

end power_mod_condition_l199_199319


namespace problem1_problem2_l199_199638

open Classical

theorem problem1 (x : ℝ) : -x^2 + 4 * x - 4 < 0 ↔ x ≠ 2 :=
sorry

theorem problem2 (x : ℝ) : (1 - x) / (x - 5) > 0 ↔ 1 < x ∧ x < 5 :=
sorry

end problem1_problem2_l199_199638


namespace geometric_progression_first_term_l199_199655

theorem geometric_progression_first_term (a r : ℝ) 
    (h_sum_inf : a / (1 - r) = 8)
    (h_sum_two : a * (1 + r) = 5) :
    a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) :=
sorry

end geometric_progression_first_term_l199_199655


namespace total_red_papers_l199_199962

-- Defining the number of red papers in one box and the number of boxes Hoseok has
def red_papers_per_box : ℕ := 2
def number_of_boxes : ℕ := 2

-- Statement to prove
theorem total_red_papers : (red_papers_per_box * number_of_boxes) = 4 := by
  sorry

end total_red_papers_l199_199962


namespace mandy_med_school_ratio_l199_199468

theorem mandy_med_school_ratio 
    (researched_schools : ℕ)
    (applied_ratio : ℚ)
    (accepted_schools : ℕ)
    (h1 : researched_schools = 42)
    (h2 : applied_ratio = 1 / 3)
    (h3 : accepted_schools = 7)
    : (accepted_schools : ℚ) / ((researched_schools : ℚ) * applied_ratio) = 1 / 2 :=
by sorry

end mandy_med_school_ratio_l199_199468


namespace perfect_square_trinomial_l199_199324

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l199_199324


namespace max_squares_covered_by_card_l199_199365

theorem max_squares_covered_by_card (side_len : ℕ) (card_side : ℕ) : 
  side_len = 1 → card_side = 2 → n ≤ 12 :=
by
  sorry

end max_squares_covered_by_card_l199_199365


namespace solution_l199_199986

-- Define the equation
def equation (x : ℝ) := x^2 + 4*x + 3 + (x + 3)*(x + 5) = 0

-- State that x = -3 is a solution to the equation
theorem solution : equation (-3) :=
by
  unfold equation
  simp
  sorry

end solution_l199_199986


namespace sarah_initial_money_l199_199491

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l199_199491


namespace Nancy_antacid_consumption_l199_199480

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end Nancy_antacid_consumption_l199_199480


namespace ratio_induction_l199_199316

theorem ratio_induction (k : ℕ) (hk : k > 0) :
    (k + 2) * (k + 3) / (2 * (2 * k + 1)) = 1 := by
sorry

end ratio_induction_l199_199316


namespace francie_has_3_dollars_remaining_l199_199914

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l199_199914


namespace prob_two_heads_l199_199843

theorem prob_two_heads (h : Uniform π) :
  (π (2 heads)) = 1/4 :=
by
  sorry

end prob_two_heads_l199_199843


namespace max_value_f_on_0_4_l199_199041

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_f_on_0_4 : ∃ (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) (4 : ℝ)), ∀ (y : ℝ), y ∈ Set.Icc (0 : ℝ) (4 : ℝ) → f y ≤ f x ∧ f x = 1 / Real.exp 1 :=
by
  sorry

end max_value_f_on_0_4_l199_199041


namespace polygon_sides_sum_l199_199048

theorem polygon_sides_sum (triangle_hexagon_sum : ℕ) (triangle_sides : ℕ) (hexagon_sides : ℕ) 
  (h1 : triangle_hexagon_sum = 1260) 
  (h2 : triangle_sides = 3) 
  (h3 : hexagon_sides = 6) 
  (convex : ∀ n, 3 <= n) : 
  triangle_sides + hexagon_sides + 4 = 13 :=
by 
  sorry

end polygon_sides_sum_l199_199048


namespace problem1_problem2_l199_199548

-- Problem 1: Prove that \(\sqrt{27}+3\sqrt{\frac{1}{3}}-\sqrt{24} \times \sqrt{2} = 0\)
theorem problem1 : Real.sqrt 27 + 3 * Real.sqrt (1 / 3) - Real.sqrt 24 * Real.sqrt 2 = 0 := 
by sorry

-- Problem 2: Prove that \((\sqrt{5}-2)(2+\sqrt{5})-{(\sqrt{3}-1)}^{2} = -3 + 2\sqrt{3}\)
theorem problem2 : (Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1) ^ 2 = -3 + 2 * Real.sqrt 3 := 
by sorry

end problem1_problem2_l199_199548


namespace number_of_tie_games_l199_199449

def total_games (n_teams: ℕ) (games_per_matchup: ℕ) : ℕ :=
  (n_teams * (n_teams - 1) / 2) * games_per_matchup

def theoretical_max_points (total_games: ℕ) (points_per_win: ℕ): ℕ :=
  total_games * points_per_win

def actual_total_points (lions: ℕ) (tigers: ℕ) (mounties: ℕ) (royals: ℕ): ℕ :=
  lions + tigers + mounties + royals

def tie_games (theoretical_points: ℕ) (actual_points: ℕ) (points_per_tie: ℕ): ℕ :=
  (theoretical_points - actual_points) / points_per_tie

theorem number_of_tie_games
  (n_teams: ℕ)
  (games_per_matchup: ℕ)
  (points_per_win: ℕ)
  (points_per_tie: ℕ)
  (lions: ℕ)
  (tigers: ℕ)
  (mounties: ℕ)
  (royals: ℕ)
  (h_teams: n_teams = 4)
  (h_games: games_per_matchup = 4)
  (h_points_win: points_per_win = 3)
  (h_points_tie: points_per_tie = 2)
  (h_lions: lions = 22)
  (h_tigers: tigers = 19)
  (h_mounties: mounties = 14)
  (h_royals: royals = 12) :
  tie_games (theoretical_max_points (total_games n_teams games_per_matchup) points_per_win) 
  (actual_total_points lions tigers mounties royals) points_per_tie = 5 :=
by
  rw [h_teams, h_games, h_points_win, h_points_tie, h_lions, h_tigers, h_mounties, h_royals]
  simp [total_games, theoretical_max_points, actual_total_points, tie_games]
  sorry

end number_of_tie_games_l199_199449


namespace completion_days_together_l199_199350

-- Definitions based on given conditions
variable (W : ℝ) -- Total work
variable (A : ℝ) -- Work done by A in one day
variable (B : ℝ) -- Work done by B in one day

-- Condition 1: A alone completes the work in 20 days
def work_done_by_A := A = W / 20

-- Condition 2: A and B working with B half a day complete the work in 15 days
def work_done_by_A_and_half_B := A + (1 / 2) * B = W / 15

-- Prove: A and B together will complete the work in 60 / 7 days if B works full time
theorem completion_days_together (h1 : work_done_by_A W A) (h2 : work_done_by_A_and_half_B W A B) :
  ∃ D : ℝ, D = 60 / 7 :=
by 
  sorry

end completion_days_together_l199_199350


namespace quadratic_residue_property_l199_199945

theorem quadratic_residue_property (p : ℕ) [hp : Fact (Nat.Prime p)] (a : ℕ)
  (h : ∃ t : ℤ, ∃ k : ℤ, k * k = p * t + a) : (a ^ ((p - 1) / 2)) % p = 1 :=
sorry

end quadratic_residue_property_l199_199945


namespace find_f_5_l199_199994

-- Define the function f satisfying the given conditions
noncomputable def f : ℝ → ℝ :=
sorry

-- Assert the conditions as hypotheses
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x + f y
axiom f_zero : f 0 = 2

-- State the theorem we need to prove
theorem find_f_5 : f 5 = 1 :=
sorry

end find_f_5_l199_199994


namespace fraction_of_work_completed_in_25_days_l199_199974

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end fraction_of_work_completed_in_25_days_l199_199974


namespace find_min_value_l199_199938

-- Define a structure to represent vectors in 2D space
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

-- Define the dot product of two vectors
def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Define the condition for perpendicular vectors (dot product is zero)
def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

-- Define the problem: given vectors a = (m, 1) and b = (1, n - 2)
-- with conditions m > 0, n > 0, and a ⊥ b, then prove the minimum value of 1/m + 2/n
theorem find_min_value (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0)
  (h₂ : perpendicular ⟨m, 1⟩ ⟨1, n - 2⟩) :
  (1 / m + 2 / n) = (3 + 2 * Real.sqrt 2) / 2 :=
  sorry

end find_min_value_l199_199938


namespace probability_more_ones_than_sixes_l199_199762

theorem probability_more_ones_than_sixes :
  (∃ (prob : ℚ), prob = 223 / 648) :=
by
  -- conditions:
  -- let dice := {1, 2, 3, 4, 5, 6}
  
  -- question:
  -- the desired probability is provable to be 223 / 648
  
  have probability : ℚ := 223 / 648,
  use probability,
  sorry

end probability_more_ones_than_sixes_l199_199762


namespace find_x_l199_199658

theorem find_x (x : ℝ) :
  (x * 13.26 + x * 9.43 + x * 77.31 = 470) → (x = 4.7) :=
by
  sorry

end find_x_l199_199658


namespace days_to_complete_l199_199249

variable {m n : ℕ}

theorem days_to_complete (h : ∀ (m n : ℕ), (m + n) * m = 1) : 
  ∀ (n m : ℕ), (m * (m + n)) / n = m * (m + n) / n :=
by
  sorry

end days_to_complete_l199_199249


namespace who_is_murdock_l199_199357

variables (A B C : Prop)
variables (is_murdock : Prop)
variables (is_knight : Prop)
variables (is_liar : Prop)

-- Conditions
def A_statement := is_murdock
def B_statement := A_statement
def C_statement := ¬ is_murdock

-- Knight tells the truth
axiom knight_truth : is_knight → A_statement = true ∨ B_statement = true ∨ C_statement = true

-- Liar tells lies
axiom liar_lying : is_liar → A_statement = false ∨ B_statement = false ∨ C_statement = false

-- Exactly one knight and one liar
axiom unique_knight_liar : ∃ A_is_knight A_is_liar B_is_murdock B_is_knight B_is_liar C_is_murdock C_is_knight C_is_liar, 
  (A_is_knight ≠ B_is_knight ∧ A_is_knight ≠ C_is_knight ∧ B_is_knight ≠ C_is_knight) ∧
  (A_is_liar ≠ B_is_liar ∧ A_is_liar ≠ C_is_liar ∧ B_is_liar ≠ C_is_liar) ∧
  (is_knight ∨ is_liar)

-- The goal: prove B is Murdock
theorem who_is_murdock : is_murdock = B :=
by sorry

end who_is_murdock_l199_199357


namespace average_brown_MnMs_l199_199486

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l199_199486


namespace fill_tank_with_leak_l199_199195

namespace TankFilling

-- Conditions
def pump_fill_rate (P : ℝ) : Prop := P = 1 / 4
def leak_drain_rate (L : ℝ) : Prop := L = 1 / 5
def net_fill_rate (P L R : ℝ) : Prop := P - L = R
def fill_time (R T : ℝ) : Prop := T = 1 / R

-- Statement
theorem fill_tank_with_leak (P L R T : ℝ) (hP : pump_fill_rate P) (hL : leak_drain_rate L) (hR : net_fill_rate P L R) (hT : fill_time R T) :
  T = 20 :=
  sorry

end TankFilling

end fill_tank_with_leak_l199_199195


namespace find_x_l199_199242

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 4, 5)
def vector_b : ℝ × ℝ × ℝ := (1, -2, 2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
def magnitude (a : ℝ × ℝ × ℝ) : ℝ := real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

theorem find_x (x : ℝ) (h : (dot_product (vector_a x) vector_b) = (magnitude (vector_a x)) * (magnitude vector_b) * (√2 / 6))
  : x = 3 :=
by sorry

end find_x_l199_199242


namespace number_to_remove_l199_199071

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem number_to_remove (s : List ℕ) (x : ℕ) 
  (h₀ : s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
  (h₁ : x ∈ s)
  (h₂ : mean (List.erase s x) = 6.1) : x = 5 := sorry

end number_to_remove_l199_199071


namespace largest_K_is_1_l199_199565

noncomputable def largest_K_vip (K : ℝ) : Prop :=
  ∀ (k : ℝ) (a b c : ℝ), 
  0 ≤ k ∧ k ≤ K → 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a^2 + b^2 + c^2 + k * a * b * c = k + 3 → 
  a + b + c ≤ 3

theorem largest_K_is_1 : largest_K_vip 1 :=
sorry

end largest_K_is_1_l199_199565


namespace Kath_payment_l199_199955

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l199_199955


namespace prob_not_all_same_correct_l199_199067

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l199_199067


namespace james_earnings_l199_199456

theorem james_earnings :
  let jan_earn : ℕ := 4000
  let feb_earn := 2 * jan_earn
  let total_earnings : ℕ := 18000
  let earnings_jan_feb := jan_earn + feb_earn
  let mar_earn := total_earnings - earnings_jan_feb
  (feb_earn - mar_earn) = 2000 := by
  sorry

end james_earnings_l199_199456


namespace area_of_triangle_XPQ_l199_199015

noncomputable def area_triangle_XPQ (XY YZ XZ XP XQ : ℝ) (hXY : XY = 12) (hYZ : YZ = 13) (hXZ : XZ = 15) (hXP : XP = 5) (hXQ : XQ = 9) : ℝ :=
  let s := (XY + YZ + XZ) / 2
  let area_XYZ := Real.sqrt (s * (s - XY) * (s - YZ) * (s - XZ))
  let cosX := (XY^2 + YZ^2 - XZ^2) / (2 * XY * YZ)
  let sinX := Real.sqrt (1 - cosX^2)
  (1 / 2) * XP * XQ * sinX

theorem area_of_triangle_XPQ :
  area_triangle_XPQ 12 13 15 5 9 (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) = 45 * Real.sqrt 1400 / 78 :=
by
  sorry

end area_of_triangle_XPQ_l199_199015


namespace ball_travel_distance_l199_199363

noncomputable def total_distance : ℝ :=
  200 + (2 * (200 * (1 / 3))) + (2 * (200 * ((1 / 3) ^ 2))) +
  (2 * (200 * ((1 / 3) ^ 3))) + (2 * (200 * ((1 / 3) ^ 4)))

theorem ball_travel_distance :
  total_distance = 397.2 :=
by
  sorry

end ball_travel_distance_l199_199363


namespace three_legged_extraterrestrials_l199_199400

-- Define the conditions
variables (x y : ℕ)

-- Total number of heads
def heads_equation := x + y = 300

-- Total number of legs
def legs_equation := 3 * x + 4 * y = 846

theorem three_legged_extraterrestrials : heads_equation x y ∧ legs_equation x y → x = 246 :=
by
  sorry

end three_legged_extraterrestrials_l199_199400


namespace base_conversion_least_sum_l199_199647

theorem base_conversion_least_sum :
  ∃ (c d : ℕ), (5 * c + 8 = 8 * d + 5) ∧ c > 0 ∧ d > 0 ∧ (c + d = 15) := by
sorry

end base_conversion_least_sum_l199_199647


namespace sheep_drowned_proof_l199_199863

def animal_problem_statement (S : ℕ) : Prop :=
  let initial_sheep := 20
  let initial_cows := 10
  let initial_dogs := 14
  let total_animals_made_shore := 35
  let sheep_drowned := S
  let cows_drowned := 2 * S
  let dogs_survived := initial_dogs
  let animals_made_shore := initial_sheep + initial_cows + initial_dogs - (sheep_drowned + cows_drowned)
  30 - 3 * S = 35 - 14

theorem sheep_drowned_proof : ∃ S : ℕ, animal_problem_statement S ∧ S = 3 :=
by
  sorry

end sheep_drowned_proof_l199_199863


namespace find_sixth_number_l199_199183

theorem find_sixth_number 
  (A : ℕ → ℝ)
  (h1 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 11 = 60))
  (h2 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6) / 6 = 58))
  (h3 : ((A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 6 = 65)) 
  : A 6 = 78 :=
by
  sorry

end find_sixth_number_l199_199183


namespace parallel_lines_k_value_l199_199252

-- Define the lines and the condition of parallelism
def line1 (x y : ℝ) := x + 2 * y - 1 = 0
def line2 (k x y : ℝ) := k * x - y = 0

-- Define the parallelism condition
def lines_parallel (k : ℝ) := (1 / k) = (2 / -1)

-- Prove that given the parallelism condition, k equals -1/2
theorem parallel_lines_k_value (k : ℝ) (h : lines_parallel k) : k = (-1 / 2) :=
by
  sorry

end parallel_lines_k_value_l199_199252


namespace simplify_expression_l199_199153

theorem simplify_expression (x : ℝ) : (x + 1) ^ 2 + x * (x - 2) = 2 * x ^ 2 + 1 :=
by
  sorry

end simplify_expression_l199_199153


namespace common_chord_eq_l199_199302

theorem common_chord_eq : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0) → 
  (∀ x y : ℝ, x + 2*y - 1 = 0) :=
by 
  sorry

end common_chord_eq_l199_199302


namespace fraction_identity_l199_199756

theorem fraction_identity (a b : ℝ) (hb : b ≠ 0) (h : a / b = 3 / 2) : (a + b) / b = 2.5 :=
by
  sorry

end fraction_identity_l199_199756


namespace tiling_remainder_l199_199187

-- Define the board size
def board_size := 8

-- Define the allowable tile configurations
def tile (n : ℕ) := Fin n

-- We need to prove the following theorem
theorem tiling_remainder :
  let N := 
      ∑ k in {6, 7, 8}, 
        (Nat.choose 7 (k - 1)) * 
        (3^k - (Nat.choose 3 1) * 2^k + (Nat.choose 3 2) * 1^k) in
  N % 1000 = 691 :=
by
  sorry

end tiling_remainder_l199_199187


namespace prove_x3_y3_le_2_l199_199156

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom condition : x^3 + y^4 ≤ x^2 + y^3

theorem prove_x3_y3_le_2 : x^3 + y^3 ≤ 2 := 
by
  sorry

end prove_x3_y3_le_2_l199_199156


namespace solve_fraction_equation_l199_199112

theorem solve_fraction_equation :
  {x : ℝ | (1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 15 * x - 12) = 0)} =
  {1, -12, 12, -1} :=
by
  sorry

end solve_fraction_equation_l199_199112


namespace isabella_hair_length_l199_199135

theorem isabella_hair_length (final_length growth_length initial_length : ℕ) 
  (h1 : final_length = 24) 
  (h2 : growth_length = 6) 
  (h3 : final_length = initial_length + growth_length) : 
  initial_length = 18 :=
by
  sorry

end isabella_hair_length_l199_199135


namespace min_value_fraction_l199_199590

theorem min_value_fraction 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a1 a3 a13 : ℕ)
  (d : ℕ) 
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a1 = 1)
  (h4 : a3 ^ 2 = a1 * a13)
  (h5 : ∀ n, S_n n = n * (a1 + a_n n) / 2) :
  ∃ n, (2 * S_n n + 16) / (a_n n + 3) = 4 := 
sorry

end min_value_fraction_l199_199590


namespace fraction_addition_l199_199181

variable {w x y : ℝ}

theorem fraction_addition (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 := by
  sorry

end fraction_addition_l199_199181


namespace work_days_for_A_and_B_l199_199088

theorem work_days_for_A_and_B (W_A W_B : ℝ) (h1 : W_A = (1/2) * W_B) (h2 : W_B = 1/21) : 
  1 / (W_A + W_B) = 14 := by 
  sorry

end work_days_for_A_and_B_l199_199088


namespace unique_flavors_count_l199_199637

def numOrangeCandies : ℕ := 6
def numPurpleCandies : ℕ := 4

-- Define a set of all possible candy combinations, excluding (0, 0)
def candy_combinations : Set (ℕ × ℕ) :=
  { p | p.1 ≤ numOrangeCandies ∧ p.2 ≤ numPurpleCandies ∧ ¬(p.1 = 0 ∧ p.2 = 0) }

-- Define a function to reduce a ratio to its simplest form
def simplify_ratio (x y : ℕ) : Option (ℕ × ℕ) :=
  if y = 0 then none
  else if x = 0 then some (0, 1)
  else some ((x / (Nat.gcd x y)), (y / (Nat.gcd x y)))

-- Create a set of simplified ratios from the candy combinations
def simplified_ratios : Set (Option (ℕ × ℕ)) :=
  { simplify_ratio p.1 p.2 | p ∈ candy_combinations }

theorem unique_flavors_count :
  simplified_ratios.to_finset.card = 14 := sorry

end unique_flavors_count_l199_199637


namespace mike_can_buy_nine_games_l199_199358

noncomputable def mike_dollars (initial_dollars : ℕ) (spent_dollars : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_dollars - spent_dollars) / game_cost

theorem mike_can_buy_nine_games : mike_dollars 69 24 5 = 9 := by
  sorry

end mike_can_buy_nine_games_l199_199358


namespace solution_count_l199_199003

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem solution_count (a : ℝ) : 
  (∃ x : ℝ, f x = a) ↔ 
  ((a > 2 ∨ a < -2 ∧ ∃! x₁, f x₁ = a) ∨ 
   ((a = 2 ∨ a = -2) ∧ ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ∨ 
   (-2 < a ∧ a < 2 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a)) := 
by sorry

end solution_count_l199_199003


namespace problem_statement_l199_199706

theorem problem_statement (x y z : ℤ) (h1 : x = z - 2) (h2 : y = x + 1) : 
  x * (x - y) + y * (y - z) + z * (z - x) = 1 := 
by
  sorry

end problem_statement_l199_199706


namespace hexagon_largest_angle_l199_199445

-- Definitions for conditions
def hexagon_interior_angle_sum : ℝ := 720  -- Sum of all interior angles of hexagon

def angle_A : ℝ := 100
def angle_B : ℝ := 120

-- Define x for angles C and D
variables (x : ℝ)
def angle_C : ℝ := x
def angle_D : ℝ := x
def angle_F : ℝ := 3 * x + 10

-- The formal statement to prove
theorem hexagon_largest_angle (x : ℝ) : 
  100 + 120 + x + x + (3 * x + 10) = 720 → 
  3 * x + 10 = 304 :=
by 
  sorry

end hexagon_largest_angle_l199_199445


namespace triangle_area_parallel_line_l199_199723

/-- Given line passing through (8, 2) and parallel to y = -x + 1,
    the area of the triangle formed by this line and the coordinate axes is 50. -/
theorem triangle_area_parallel_line :
  ∃ k b : ℝ, k = -1 ∧ (8 * k + b = 2) ∧ (1/2 * 10 * 10 = 50) :=
sorry

end triangle_area_parallel_line_l199_199723


namespace probability_of_earning_exactly_2400_in_3_spins_l199_199977

-- Definitions corresponding to the given conditions
def spinnerAmounts := {Bankrupt, 1500, 200, 6000, 700}

def probabilityOfLandingOn (x : ℝ) : ℝ := 
  if x ∈ spinnerAmounts then 1 / 5 else 0

def eventEarning2400In3Spins (s1 s2 s3 : ℝ) : Prop :=
  s1 + s2 + s3 = 2400 ∧ s1 ∈ spinnerAmounts ∧ s2 ∈ spinnerAmounts ∧ s3 ∈ spinnerAmounts

-- The statement to be proved
theorem probability_of_earning_exactly_2400_in_3_spins :
  (∑ (s1 s2 s3 : ℝ) in spinnerAmounts, 
    if eventEarning2400In3Spins s1 s2 s3 
    then (probabilityOfLandingOn s1 * probabilityOfLandingOn s2 * probabilityOfLandingOn s3) else 0) = 6 / 125 := 
  sorry

end probability_of_earning_exactly_2400_in_3_spins_l199_199977


namespace count_integers_between_cubes_l199_199426

theorem count_integers_between_cubes :
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  n2 - n1 + 1 = 67 :=
by
  let a := 10.5
  let b := 10.7
  let n1 := (a^3).ceil
  let n2 := (b^3).floor
  have h1 : (a^3 = 1157.625) := by sorry
  have h2 : (b^3 = 1225.043) := by sorry
  have h3 : (n1 = 1158) := by sorry
  have h4 : (n2 = 1224) := by sorry
  rw [h1, h2, h3, h4]
  sorry

end count_integers_between_cubes_l199_199426


namespace total_salary_l199_199314

-- Define the salaries and conditions.
def salaryN : ℝ := 280
def salaryM : ℝ := 1.2 * salaryN

-- State the theorem we want to prove
theorem total_salary : salaryM + salaryN = 616 :=
by
  sorry

end total_salary_l199_199314


namespace people_in_gym_l199_199205

-- Define the initial number of people in the gym
def initial_people : ℕ := 16

-- Define the number of additional people entering the gym
def additional_people : ℕ := 5

-- Define the number of people leaving the gym
def people_leaving : ℕ := 2

-- Define the final number of people in the gym as per the conditions
def final_people (initial : ℕ) (additional : ℕ) (leaving : ℕ) : ℕ :=
  initial + additional - leaving

-- The theorem to prove
theorem people_in_gym : final_people initial_people additional_people people_leaving = 19 :=
  by
    sorry

end people_in_gym_l199_199205


namespace probability_more_ones_than_sixes_l199_199771

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end probability_more_ones_than_sixes_l199_199771


namespace parabola_vertex_on_x_axis_l199_199705

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ h k : ℝ, y = (x : ℝ)^2 - 12 * x + c ∧
   (h = -12 / 2) ∧
   (k = c - 144 / 4) ∧
   (k = 0)) ↔ c = 36 :=
by
  sorry

end parabola_vertex_on_x_axis_l199_199705


namespace ticket_savings_l199_199682

def single_ticket_cost : ℝ := 1.50
def package_cost : ℝ := 5.75
def num_tickets_needed : ℝ := 40

theorem ticket_savings :
  (num_tickets_needed * single_ticket_cost) - 
  ((num_tickets_needed / 5) * package_cost) = 14.00 :=
by
  sorry

end ticket_savings_l199_199682


namespace quadratic_roots_l199_199437

theorem quadratic_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2*x1 + 2*m = 0) ∧ (x2^2 + 2*x2 + 2*m = 0)) ↔ m < 1/2 :=
by sorry

end quadratic_roots_l199_199437


namespace xiaohua_apples_l199_199347

theorem xiaohua_apples (x : ℕ) (h1 : ∃ n, (n = 4 * x + 20)) 
                       (h2 : (4 * x + 20 - 8 * (x - 1) > 0) ∧ (4 * x + 20 - 8 * (x - 1) < 8)) : 
                       4 * x + 20 = 44 := by
  sorry

end xiaohua_apples_l199_199347


namespace dice_probability_not_all_same_l199_199064

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l199_199064


namespace correct_graph_is_C_l199_199441

-- Define the years and corresponding remote work percentages
def percentages : List (ℕ × ℝ) := [
  (1960, 0.1),
  (1970, 0.15),
  (1980, 0.12),
  (1990, 0.25),
  (2000, 0.4)
]

-- Define the property of the graph trend
def isCorrectGraph (p : List (ℕ × ℝ)) : Prop :=
  p = [
    (1960, 0.1),
    (1970, 0.15),
    (1980, 0.12),
    (1990, 0.25),
    (2000, 0.4)
  ]

-- State the theorem
theorem correct_graph_is_C : isCorrectGraph percentages = True :=
  sorry

end correct_graph_is_C_l199_199441


namespace cost_of_senior_ticket_l199_199868

theorem cost_of_senior_ticket (x : ℤ) (total_tickets : ℤ) (cost_regular_ticket : ℤ) (total_sales : ℤ) (senior_tickets_sold : ℤ) (regular_tickets_sold : ℤ) :
  total_tickets = 65 →
  cost_regular_ticket = 15 →
  total_sales = 855 →
  senior_tickets_sold = 24 →
  regular_tickets_sold = total_tickets - senior_tickets_sold →
  total_sales = senior_tickets_sold * x + regular_tickets_sold * cost_regular_ticket →
  x = 10 :=
by
  sorry

end cost_of_senior_ticket_l199_199868


namespace sum_of_powers_l199_199818

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 :=
by
  sorry

end sum_of_powers_l199_199818


namespace find_q_l199_199736

theorem find_q (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199736


namespace total_volume_of_five_boxes_l199_199846

theorem total_volume_of_five_boxes 
  (edge_length : ℕ) (number_of_boxes : ℕ) (volume_of_one_box : ℕ) 
  (total_volume : ℕ)
  (h1 : edge_length = 3)
  (h2 : number_of_boxes = 5)
  (h3 : volume_of_one_box = edge_length ^ 3)
  (h4 : total_volume = volume_of_one_box * number_of_boxes) : 
  total_volume = 135 := 
begin
  sorry
end

end total_volume_of_five_boxes_l199_199846


namespace solve_for_x_l199_199035

theorem solve_for_x (x : ℝ) 
  (h : 6 * x + 12 * x = 558 - 9 * (x - 4)) : 
  x = 22 := 
sorry

end solve_for_x_l199_199035


namespace updated_mean_corrected_l199_199999

theorem updated_mean_corrected (mean observations decrement : ℕ) 
  (h1 : mean = 350) (h2 : observations = 100) (h3 : decrement = 63) :
  (mean * observations + decrement * observations) / observations = 413 :=
by
  sorry

end updated_mean_corrected_l199_199999


namespace ratio_of_girls_who_like_pink_l199_199807

theorem ratio_of_girls_who_like_pink 
  (total_students : ℕ) (answered_green : ℕ) (answered_yellow : ℕ) (total_girls : ℕ) (answered_yellow_students : ℕ)
  (portion_girls_pink : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_green = total_students / 2)
  (h3 : total_girls = 18)
  (h4 : answered_yellow_students = 9)
  (answered_pink := total_students - answered_green - answered_yellow_students)
  (ratio_pink : ℚ := answered_pink / total_girls) : 
  ratio_pink = 1 / 3 :=
sorry

end ratio_of_girls_who_like_pink_l199_199807


namespace bone_meal_percentage_growth_l199_199017

-- Definitions for the problem conditions
def control_height : ℝ := 36
def cow_manure_height : ℝ := 90
def bone_meal_to_cow_manure_ratio : ℝ := 0.5 -- since cow manure plant is 200% the height of bone meal plant

noncomputable def bone_meal_height : ℝ := cow_manure_height * bone_meal_to_cow_manure_ratio

-- The main theorem to prove
theorem bone_meal_percentage_growth : 
  ( (bone_meal_height - control_height) / control_height ) * 100 = 25 := 
by
  sorry

end bone_meal_percentage_growth_l199_199017


namespace zaim_larger_part_l199_199518

theorem zaim_larger_part (x y : ℕ) (h_sum : x + y = 20) (h_prod : x * y = 96) : max x y = 12 :=
by
  -- The proof goes here
  sorry

end zaim_larger_part_l199_199518


namespace find_y_intercept_l199_199435

theorem find_y_intercept (a b : ℝ) (h1 : (3 : ℝ) ≠ (7 : ℝ))
  (h2 : -2 = a * 3 + b) (h3 : 14 = a * 7 + b) :
  b = -14 :=
sorry

end find_y_intercept_l199_199435


namespace right_pyramid_sum_edges_l199_199369

theorem right_pyramid_sum_edges (a h : ℝ) (base_side slant_height : ℝ) :
  base_side = 12 ∧ slant_height = 15 ∧ ∀ x : ℝ, a = 117 :=
by
  sorry

end right_pyramid_sum_edges_l199_199369


namespace sequence_not_divisible_by_7_l199_199246

theorem sequence_not_divisible_by_7 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 1200) : ¬ (7 ∣ (9^n + 1)) :=
by
  sorry

end sequence_not_divisible_by_7_l199_199246


namespace triangle_incircle_centers_right_angle_l199_199283

open EuclideanGeometry

variables {A B C D O1 O2 : Point}

/-- Given a triangle ABC with a point D on BC, and the centers of the incircles of triangles ABD and ACD are O1 and O2 respectively.
    We prove that the triangle O1DO2 is right-angled. --/
theorem triangle_incircle_centers_right_angle 
  (hD_on_BC : lies_on D (line B C))
  (hO1 : is_incircle_center A B D O1)
  (hO2 : is_incircle_center A C D O2) :
  angle O1 D O2 = 90 :=
begin
  sorry
end

end triangle_incircle_centers_right_angle_l199_199283


namespace infinitely_many_m_l199_199483

theorem infinitely_many_m (r : ℕ) (n : ℕ) (h_r : r > 1) (h_n : n > 0) : 
  ∃ m, m = 4 * r ^ 4 ∧ ¬Prime (n^4 + m) :=
by
  sorry

end infinitely_many_m_l199_199483


namespace rabbit_speed_final_result_l199_199440

def rabbit_speed : ℕ := 45

def double_speed (speed : ℕ) : ℕ := speed * 2

def add_four (n : ℕ) : ℕ := n + 4

def final_operation : ℕ := double_speed (add_four (double_speed rabbit_speed))

theorem rabbit_speed_final_result : final_operation = 188 := 
by
  sorry

end rabbit_speed_final_result_l199_199440


namespace daisies_sold_on_fourth_day_l199_199700

-- Number of daisies sold on the first day
def first_day_daisies : ℕ := 45

-- Number of daisies sold on the second day
def second_day_daisies : ℕ := first_day_daisies + 20

-- Number of daisies sold on the third day
def third_day_daisies : ℕ := 2 * second_day_daisies - 10

-- Total number of daisies sold in the first three days
def total_first_three_days_daisies : ℕ := first_day_daisies + second_day_daisies + third_day_daisies

-- Total number of daisies sold in four days
def total_four_days_daisies : ℕ := 350

-- Number of daisies sold on the fourth day
def fourth_day_daisies : ℕ := total_four_days_daisies - total_first_three_days_daisies

-- Theorem that states the number of daisies sold on the fourth day is 120
theorem daisies_sold_on_fourth_day : fourth_day_daisies = 120 :=
by sorry

end daisies_sold_on_fourth_day_l199_199700


namespace eight_is_100_discerning_nine_is_not_100_discerning_l199_199670

-- Define what it means to be b-discerning
def is_b_discerning (n b : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧ (∀ (U V : Finset ℕ), U ≠ V ∧ U ⊆ S ∧ V ⊆ S → U.sum id ≠ V.sum id)

-- Prove that 8 is 100-discerning
theorem eight_is_100_discerning : is_b_discerning 8 100 :=
sorry

-- Prove that 9 is not 100-discerning
theorem nine_is_not_100_discerning : ¬is_b_discerning 9 100 :=
sorry

end eight_is_100_discerning_nine_is_not_100_discerning_l199_199670


namespace fraction_of_work_left_l199_199675

variable (A_work_days : ℕ) (B_work_days : ℕ) (work_days_together: ℕ)

theorem fraction_of_work_left (hA : A_work_days = 15) (hB : B_work_days = 20) (hT : work_days_together = 4):
  (1 - 4 * (1 / 15 + 1 / 20)) = (8 / 15) :=
sorry

end fraction_of_work_left_l199_199675


namespace empty_pipe_time_l199_199352

theorem empty_pipe_time (R1 R2 : ℚ) (t1 t2 t_total : ℕ) (h1 : t1 = 60) (h2 : t_total = 180) (H1 : R1 = 1 / t1) (H2 : R1 - R2 = 1 / t_total) :
  1 / R2 = 90 :=
by
  sorry

end empty_pipe_time_l199_199352


namespace neg_p_l199_199805

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end neg_p_l199_199805


namespace root_one_value_of_m_real_roots_range_of_m_l199_199114

variables {m x : ℝ}

-- Part 1: Prove that if 1 is a root of 'mx^2 - 4x + 1 = 0', then m = 3
theorem root_one_value_of_m (h : m * 1^2 - 4 * 1 + 1 = 0) : m = 3 :=
  by sorry

-- Part 2: Prove that 'mx^2 - 4x + 1 = 0' has real roots iff 'm ≤ 4 ∧ m ≠ 0'
theorem real_roots_range_of_m : (∃ x : ℝ, m * x^2 - 4 * x + 1 = 0) ↔ (m ≤ 4 ∧ m ≠ 0) :=
  by sorry

end root_one_value_of_m_real_roots_range_of_m_l199_199114


namespace smallest_n_proof_l199_199703

-- Given conditions and the problem statement in Lean 4
noncomputable def smallest_n : ℕ := 11

theorem smallest_n_proof :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧ (smallest_n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 11) :=
sorry

end smallest_n_proof_l199_199703


namespace no_nat_num_divisible_l199_199131

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l199_199131


namespace increase_in_y_coordinate_l199_199856

theorem increase_in_y_coordinate (m n : ℝ) (h₁ : m = (n / 5) - 2 / 5) : 
  (5 * (m + 3) + 2) - (5 * m + 2) = 15 :=
by
  sorry

end increase_in_y_coordinate_l199_199856


namespace fraction_work_completed_in_25_days_eq_half_l199_199973

theorem fraction_work_completed_in_25_days_eq_half :
  ∀ (men_total men_new men_rem : ℕ) (length : ℝ) (total_days : ℕ) 
    (work_hours_per_day initial_days remaining_days : ℕ)
    (total_man_hours man_hours_in_initial_days: ℝ),
  men_total = 100 →
  length = 2 →
  total_days = 50 →
  work_hours_per_day = 8 →
  initial_days = 25 →
  men_total * total_days * work_hours_per_day = total_man_hours →
  men_total * initial_days * work_hours_per_day = man_hours_in_initial_days →
  men_new = 60 →
  remaining_days = total_days - initial_days →
  man_hours_in_initial_days / total_man_hours = 1 / 2 :=
by
  intro men_total men_new men_rem length total_days work_hours_per_day initial_days remaining_days total_man_hours man_hours_in_initial_days
  intros h_men_total h_length h_total_days h_whpd h_initial_days h_totalmh h_initialmh h_men_new h_remaining_days
  have h1 : total_man_hours = 100 * 50 * 8 := by rw [h_men_total, h_total_days, h_whpd]; norm_num
  have h2 : man_hours_in_initial_days = 100 * 25 * 8 := by rw [h_men_total, h_initial_days, h_whpd]; norm_num
  have h3 : 100 * 50 * 8 = 40_000 := by norm_num
  have h4 : 100 * 25 * 8 = 20_000 := by norm_num
  rw [h1, h3] at h_totalmh
  rw [h2, h4] at h_initialmh
  norm_num at h_totalmh
  norm_num at h_initialmh
  rw [←h_totalmh] at h_initialmh
  rw h_totalmh
  rw h_initialmh
  norm_num
  sorry

end fraction_work_completed_in_25_days_eq_half_l199_199973


namespace first_number_value_l199_199157

theorem first_number_value (A B LCM HCF : ℕ) (h_lcm : LCM = 2310) (h_hcf : HCF = 30) (h_b : B = 210) (h_mul : A * B = LCM * HCF) : A = 330 := 
by
  -- Use sorry to skip the proof
  sorry

end first_number_value_l199_199157


namespace betty_blue_beads_l199_199692

theorem betty_blue_beads (r b : ℕ) (h1 : r = 30) (h2 : 3 * b = 2 * r) : b = 20 :=
by
  sorry

end betty_blue_beads_l199_199692


namespace bottles_needed_exceed_initial_l199_199028

-- Define the initial conditions and their relationships
def initial_bottles : ℕ := 4 * 12 -- four dozen bottles

def bottles_first_break (players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  players * bottles_per_player

def bottles_second_break (total_players : ℕ) (bottles_per_player : ℕ) (exhausted_players : ℕ) (extra_bottles : ℕ) : ℕ :=
  total_players * bottles_per_player + exhausted_players * extra_bottles

def bottles_third_break (remaining_players : ℕ) (bottles_per_player : ℕ) : ℕ :=
  remaining_players * bottles_per_player

-- Prove that the bottles needed exceed the initial amount by 4
theorem bottles_needed_exceed_initial : 
  bottles_first_break 11 2 + bottles_second_break 14 1 4 1 + bottles_third_break 12 1 = initial_bottles + 4 :=
by
  -- Proof will be completed here
  sorry

end bottles_needed_exceed_initial_l199_199028


namespace repeating_decimal_as_fraction_l199_199108

noncomputable def repeating_decimal_value : ℚ :=
  let x := 2.35 + 35 / 99 in
  x

theorem repeating_decimal_as_fraction :
  repeating_decimal_value = 233 / 99 := by
  sorry

end repeating_decimal_as_fraction_l199_199108


namespace shoe_count_l199_199084

theorem shoe_count 
  (pairs : ℕ)
  (total_shoes : ℕ)
  (prob : ℝ)
  (h_pairs : pairs = 12)
  (h_prob : prob = 0.043478260869565216)
  (h_total_shoes : total_shoes = pairs * 2) :
  total_shoes = 24 :=
by
  sorry

end shoe_count_l199_199084


namespace remainder_2n_div_14_l199_199254

theorem remainder_2n_div_14 (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 :=
sorry

end remainder_2n_div_14_l199_199254


namespace evaluate_expression_l199_199336

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l199_199336


namespace solve_system_of_equations_solve_linear_inequality_l199_199523

-- Part 1: System of equations
theorem solve_system_of_equations (x y : ℝ) (h1 : 5 * x + 2 * y = 25) (h2 : 3 * x + 4 * y = 15) : 
  x = 5 ∧ y = 0 := sorry

-- Part 2: Linear inequality
theorem solve_linear_inequality (x : ℝ) (h : 2 * x - 6 < 3 * x) : 
  x > -6 := sorry

end solve_system_of_equations_solve_linear_inequality_l199_199523


namespace line_equation_135_deg_l199_199722

theorem line_equation_135_deg (A : ℝ × ℝ) (theta : ℝ) (l : ℝ → ℝ → Prop) :
  A = (1, -2) →
  theta = 135 →
  (∀ x y, l x y ↔ y = -(x - 1) - 2) →
  ∀ x y, l x y ↔ x + y + 1 = 0 :=
by
  intros hA hTheta hl_form
  sorry

end line_equation_135_deg_l199_199722


namespace ternary_1021_to_decimal_l199_199104

-- Define the function to convert a ternary string to decimal
def ternary_to_decimal (n : String) : Nat :=
  n.foldr (fun c acc => acc * 3 + (c.toNat - '0'.toNat)) 0

-- The statement to prove
theorem ternary_1021_to_decimal : ternary_to_decimal "1021" = 34 := by
  sorry

end ternary_1021_to_decimal_l199_199104


namespace nhai_initial_men_l199_199975

theorem nhai_initial_men (M : ℕ) (W : ℕ) :
  let totalWork := M * 50 * 8 in
  let partialWork := M * 25 * 8 in
  let remainingWork := (M + 60) * 25 * 10 in
  partialWork = totalWork / 3 →
  remainingWork = (2 * totalWork) / 3 →
  M = 100 :=
by
  intros h1 h2
  have eq1 : totalWork = M * 50 * 8 := rfl
  sorry -- Proof is omitted

end nhai_initial_men_l199_199975


namespace constant_function_of_functional_equation_l199_199710

theorem constant_function_of_functional_equation {f : ℝ → ℝ} (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f (x^2 + y^2)) : ∃ c : ℝ, ∀ x : ℝ, 0 < x → f x = c := 
sorry

end constant_function_of_functional_equation_l199_199710


namespace ratio_of_areas_is_correct_l199_199837

-- Definition of the lengths of the sides of the triangles
def triangle_XYZ_sides := (7, 24, 25)
def triangle_PQR_sides := (9, 40, 41)

-- Definition of the areas of the right triangles
def area_triangle_XYZ := (7 * 24) / 2
def area_triangle_PQR := (9 * 40) / 2

-- The ratio of the areas of the triangles
def ratio_of_areas := area_triangle_XYZ / area_triangle_PQR

-- The expected answer
def expected_ratio := 7 / 15

-- The theorem proving that ratio_of_areas is equal to expected_ratio
theorem ratio_of_areas_is_correct :
  ratio_of_areas = expected_ratio := by
  -- Add the proof here
  sorry

end ratio_of_areas_is_correct_l199_199837


namespace all_numbers_positive_l199_199419

theorem all_numbers_positive (n : ℕ) (a : Fin (2 * n + 1) → ℝ) 
  (h : ∀ S : Finset (Fin (2 * n + 1)), 
        S.card = n + 1 → 
        S.sum a > (Finset.univ \ S).sum a) : 
  ∀ i, 0 < a i :=
by
  sorry

end all_numbers_positive_l199_199419


namespace inequality_proof_l199_199923

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l199_199923


namespace find_q_l199_199739

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199739


namespace value_of_a_l199_199306

theorem value_of_a 
  (x y a : ℝ)
  (h1 : 2 * x + y = 3 * a)
  (h2 : x - 2 * y = 9 * a)
  (h3 : x + 3 * y = 24) :
  a = -4 :=
sorry

end value_of_a_l199_199306


namespace items_left_in_cart_l199_199538

-- Define the initial items in the shopping cart
def initial_items : ℕ := 18

-- Define the items deleted from the shopping cart
def deleted_items : ℕ := 10

-- Theorem statement: Prove the remaining items are 8
theorem items_left_in_cart : initial_items - deleted_items = 8 :=
by
  -- Sorry marks the place where the proof would go.
  sorry

end items_left_in_cart_l199_199538


namespace total_money_raised_l199_199836

def maxDonation : ℕ := 1200
def numberMaxDonors : ℕ := 500
def smallerDonation : ℕ := maxDonation / 2
def numberSmallerDonors : ℕ := 3 * numberMaxDonors
def totalMaxDonations : ℕ := maxDonation * numberMaxDonors
def totalSmallerDonations : ℕ := smallerDonation * numberSmallerDonors
def totalDonations : ℕ := totalMaxDonations + totalSmallerDonations
def percentageRaised : ℚ := 0.4  -- using rational number for precise division

theorem total_money_raised : totalDonations / percentageRaised = 3_750_000 := by
  sorry

end total_money_raised_l199_199836


namespace scientific_notation_of_32000000_l199_199382

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l199_199382


namespace certain_number_equals_l199_199008

theorem certain_number_equals (p q : ℚ) (h1 : 3 / p = 8) (h2 : 3 / q = 18) (h3 : p - q = 0.20833333333333334) : q = 1/6 := sorry

end certain_number_equals_l199_199008


namespace integral_semicircle_minus_sin_l199_199102

open Real

noncomputable def semicircle_integral := ∫ x in -1..1, sqrt (1 - x^2)
noncomputable def sine_integral := ∫ x in -1..1, sin x

theorem integral_semicircle_minus_sin :
  (2 * semicircle_integral - sine_integral) = π :=
by
  have semi_circle_area : ∫ x in -1..1, sqrt (1 - x^2) = π / 2 := sorry
  have sine_integral_res : ∫ x in -1..1, sin x = 0 := sorry
  simp [semicircle_integral, sine_integral, semi_circle_area, sine_integral_res]
  linarith

end integral_semicircle_minus_sin_l199_199102


namespace theater_seat_count_l199_199831

theorem theater_seat_count (number_of_people : ℕ) (empty_seats : ℕ) (total_seats : ℕ) 
  (h1 : number_of_people = 532) 
  (h2 : empty_seats = 218) 
  (h3 : total_seats = number_of_people + empty_seats) : 
  total_seats = 750 := 
by 
  sorry

end theater_seat_count_l199_199831


namespace regular_polygon_sides_l199_199197

theorem regular_polygon_sides (n : ℕ) (h : n > 0) (h_exterior_angle : 360 / n = 10) : n = 36 :=
by sorry

end regular_polygon_sides_l199_199197


namespace total_admission_cost_l199_199953

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l199_199953


namespace probability_XOXOXOX_l199_199571

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l199_199571


namespace average_brown_mms_per_bag_l199_199485

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l199_199485


namespace geometric_progression_terms_l199_199037

theorem geometric_progression_terms 
  (q b4 S_n : ℚ) 
  (hq : q = 1/3) 
  (hb4 : b4 = 1/54) 
  (hS : S_n = 121/162) 
  (b1 : ℚ) 
  (hb1 : b1 = b4 * q^3)
  (Sn : ℚ) 
  (hSn : Sn = b1 * (1 - q^5) / (1 - q)) : 
  ∀ (n : ℕ), S_n = Sn → n = 5 :=
by
  intro n hn
  sorry

end geometric_progression_terms_l199_199037


namespace no_such_triples_l199_199984

theorem no_such_triples 
  (a b c : ℕ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : ¬ ∃ k, k ∣ a + c ∧ k ∣ b + c ∧ k ∣ a + b) 
  (h₃ : c^2 ∣ a + b) 
  (h₄ : b^2 ∣ a + c) 
  (h₅ : a^2 ∣ b + c) : 
  false :=
sorry

end no_such_triples_l199_199984


namespace max_possible_b_l199_199308

theorem max_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
by sorry

end max_possible_b_l199_199308


namespace trivia_team_total_members_l199_199201

theorem trivia_team_total_members (x : ℕ) (h1 : 4 ≤ x) (h2 : (x - 4) * 8 = 64) : x = 12 :=
sorry

end trivia_team_total_members_l199_199201


namespace probability_greg_rolls_more_ones_than_sixes_l199_199775

def number_of_outcomes : ℕ := 6^5

def count_combinations_zero_one_six : ℕ := 
  ((choose 5 0) * (4^5))

def count_combinations_one_one_six : ℕ := 
  ((choose 5 1) * (choose 4 1) * (4^3))

def count_combinations_two_one_six : ℕ :=
  ((choose 5 2) * (choose 3 2) * 4)

def total_combinations_equal_one_six : ℕ :=
  count_combinations_zero_one_six + count_combinations_one_one_six + count_combinations_two_one_six

def probability_equal_one_six : ℚ :=
  total_combinations_equal_one_six / number_of_outcomes

def probability_more_ones_than_sixes : ℚ :=
  1 / 2 * (1 - probability_equal_one_six)

theorem probability_greg_rolls_more_ones_than_sixes :
  probability_more_ones_than_sixes = (167 : ℚ) / 486 := by
  sorry

end probability_greg_rolls_more_ones_than_sixes_l199_199775


namespace complex_point_quadrant_l199_199702

def inFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_point_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  inFourthQuadrant z :=
by
  sorry

end complex_point_quadrant_l199_199702


namespace cyclist_avg_speed_l199_199537

theorem cyclist_avg_speed (d : ℝ) (h1 : d > 0) :
  let t_1 := d / 17
  let t_2 := d / 23
  let total_time := t_1 + t_2
  let total_distance := 2 * d
  (total_distance / total_time) = 19.55 :=
by
  -- Proof steps here
  sorry

end cyclist_avg_speed_l199_199537


namespace calculate_value_l199_199391

theorem calculate_value : (535^2 - 465^2) / 70 = 1000 := by
  sorry

end calculate_value_l199_199391


namespace AM_GM_for_x_reciprocal_l199_199804

theorem AM_GM_for_x_reciprocal (x : ℝ) (hx : 0 < x) : x + x⁻¹ ≥ 2 :=
begin
  sorry
end

end AM_GM_for_x_reciprocal_l199_199804


namespace minute_hour_hands_opposite_l199_199887

theorem minute_hour_hands_opposite (x : ℝ) (h1 : 10 * 60 ≤ x) (h2 : x ≤ 11 * 60) : 
  (5.5 * x = 442.5) :=
sorry

end minute_hour_hands_opposite_l199_199887


namespace work_left_fraction_l199_199673

theorem work_left_fraction (A_days B_days total_days : ℕ) (hA : A_days = 15) (hB : B_days = 20) (htotal : total_days = 4) :
  let A_work_per_day := (1 : ℚ) / A_days,
      B_work_per_day := (1 : ℚ) / B_days,
      combined_work_per_day := A_work_per_day + B_work_per_day,
      work_done := combined_work_per_day * total_days,
      work_left := 1 - work_done in
  work_left = 8 / 15 := 
by 
  sorry

end work_left_fraction_l199_199673


namespace problem_l199_199969

def f (n : ℕ) : ℤ := 3 ^ (2 * n) - 32 * n ^ 2 + 24 * n - 1

theorem problem (n : ℕ) (h : 0 < n) : 512 ∣ f n := sorry

end problem_l199_199969


namespace quadratic_has_one_solution_at_zero_l199_199230

theorem quadratic_has_one_solution_at_zero (k : ℝ) :
  ((k - 2) * (0 : ℝ)^2 + 3 * (0 : ℝ) + k^2 - 4 = 0) →
  (3^2 - 4 * (k - 2) * (k^2 - 4) = 0) → k = -2 :=
by
  intro h1 h2
  sorry

end quadratic_has_one_solution_at_zero_l199_199230


namespace initial_bottle_caps_correct_l199_199397

-- Defining the variables based on the conditions
def bottle_caps_found : ℕ := 7
def total_bottle_caps_now : ℕ := 32
def initial_bottle_caps : ℕ := 25

-- Statement of the theorem
theorem initial_bottle_caps_correct:
  total_bottle_caps_now - bottle_caps_found = initial_bottle_caps :=
sorry

end initial_bottle_caps_correct_l199_199397


namespace find_fg_l199_199423

def f (x : ℕ) : ℕ := 3 * x^2 + 2
def g (x : ℕ) : ℕ := 4 * x + 1

theorem find_fg :
  f (g 3) = 509 :=
by
  sorry

end find_fg_l199_199423


namespace rowing_trip_time_l199_199532

theorem rowing_trip_time
  (v_0 : ℝ) -- Rowing speed in still water
  (v_c : ℝ) -- Velocity of current
  (d : ℝ) -- Distance to the place
  (h_v0 : v_0 = 10) -- Given condition that rowing speed is 10 kmph
  (h_vc : v_c = 2) -- Given condition that current speed is 2 kmph
  (h_d : d = 144) -- Given condition that distance is 144 km :
  : (d / (v_0 - v_c) + d / (v_0 + v_c)) = 30 := -- Proving the total round trip time is 30 hours
by
  sorry

end rowing_trip_time_l199_199532


namespace total_swordfish_caught_l199_199032

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l199_199032


namespace probability_more_ones_than_sixes_l199_199761

theorem probability_more_ones_than_sixes :
  (∃ (prob : ℚ), prob = 223 / 648) :=
by
  -- conditions:
  -- let dice := {1, 2, 3, 4, 5, 6}
  
  -- question:
  -- the desired probability is provable to be 223 / 648
  
  have probability : ℚ := 223 / 648,
  use probability,
  sorry

end probability_more_ones_than_sixes_l199_199761


namespace function_intersects_x_axis_l199_199981

theorem function_intersects_x_axis (y : ℝ → ℝ) : ∃ x : ℝ, y = λ x, x + 1 ∧ y x = 0 :=
by
  refine ⟨ -1, _, _ ⟩
  have h : y = λ x, x + 1 := sorry
  rw [h]
  norm_num
  exact hm
  sorry

end function_intersects_x_axis_l199_199981


namespace p_congruent_1_mod_5_infinitely_many_primes_of_form_5n_plus_1_l199_199143

-- Given condition: p is a prime number, p > 5
variable (p : ℕ)
variable (hp_prime : Nat.Prime p)
variable (hp_gt5 : p > 5)

-- Assume that x^4 + x^3 + x^2 + x + 1 ≡ 0 (mod p) is solvable
variable (x : ℕ)
variable (hx_solution : (x^4 + x^3 + x^2 + x + 1) % p = 0)

-- Prove that p ≡ 1 (mod 5)
theorem p_congruent_1_mod_5 : p % 5 = 1 := by
  sorry

-- Infer that there are infinitely many primes of the form 5n + 1
theorem infinitely_many_primes_of_form_5n_plus_1 :
  ∃ f : ℕ → ℕ, (∀ n, Nat.Prime (f n) ∧ f n = 5 * n + 1) :=
  by
  sorry

end p_congruent_1_mod_5_infinitely_many_primes_of_form_5n_plus_1_l199_199143


namespace train_speed_kmph_l199_199092

-- The conditions
def speed_m_s : ℝ := 52.5042
def conversion_factor : ℝ := 3.6

-- The theorem we need to prove
theorem train_speed_kmph : speed_m_s * conversion_factor = 189.01512 := 
  sorry

end train_speed_kmph_l199_199092


namespace rational_includes_integers_and_fractions_l199_199850

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end rational_includes_integers_and_fractions_l199_199850


namespace probability_XOXOXOX_is_one_over_thirty_five_l199_199575

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l199_199575


namespace circle_radius_of_tangent_parabolas_l199_199506

theorem circle_radius_of_tangent_parabolas :
  ∃ r : ℝ, 
  (∀ (x : ℝ), (x^2 + r = x)) →
  r = 1 / 4 :=
by
  sorry

end circle_radius_of_tangent_parabolas_l199_199506


namespace mean_of_six_numbers_l199_199503

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l199_199503


namespace intersection_with_single_element_union_equals_A_l199_199278

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

-- Statement for question (1)
theorem intersection_with_single_element (a : ℝ) (H : A = {1, 2} ∧ A ∩ B a = {2}) : a = -1 ∨ a = -3 :=
by
  sorry

-- Statement for question (2)
theorem union_equals_A (a : ℝ) (H1 : A = {1, 2}) (H2 : A ∪ B a = A) : (a ≥ -3 ∧ a ≤ -1) :=
by
  sorry

end intersection_with_single_element_union_equals_A_l199_199278


namespace quadratic_range_l199_199000

open Real

def quadratic (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 5

theorem quadratic_range :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -8 ≤ quadratic x ∧ quadratic x ≤ 19 :=
by
  intro x h
  sorry

end quadratic_range_l199_199000


namespace neither_necessary_nor_sufficient_l199_199226

noncomputable def C1 (m n : ℝ) :=
  (m ^ 2 - 4 * n ≥ 0) ∧ (m > 0) ∧ (n > 0)

noncomputable def C2 (m n : ℝ) :=
  (m > 0) ∧ (n > 0) ∧ (m ≠ n)

theorem neither_necessary_nor_sufficient (m n : ℝ) :
  ¬(C1 m n → C2 m n) ∧ ¬(C2 m n → C1 m n) :=
sorry

end neither_necessary_nor_sufficient_l199_199226


namespace total_settings_weight_l199_199474

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l199_199474


namespace vertex_of_f_C_l199_199073

def f_A (x : ℝ) : ℝ := (x + 4) ^ 2 - 3
def f_B (x : ℝ) : ℝ := (x + 4) ^ 2 + 3
def f_C (x : ℝ) : ℝ := (x - 4) ^ 2 - 3
def f_D (x : ℝ) : ℝ := (x - 4) ^ 2 + 3

theorem vertex_of_f_C : ∃ (h k : ℝ), h = 4 ∧ k = -3 ∧ ∀ x, f_C x = (x - h) ^ 2 + k :=
by
  sorry

end vertex_of_f_C_l199_199073


namespace graph_union_l199_199398

-- Definitions of the conditions from part a)
def graph1 (z y : ℝ) : Prop := z^4 - 6 * y^4 = 3 * z^2 - 2

def graph_hyperbola (z y : ℝ) : Prop := z^2 - 3 * y^2 = 2

def graph_ellipse (z y : ℝ) : Prop := z^2 - 2 * y^2 = 1

-- Lean statement to prove the question is equivalent to the answer
theorem graph_union (z y : ℝ) : graph1 z y ↔ (graph_hyperbola z y ∨ graph_ellipse z y) := 
sorry

end graph_union_l199_199398


namespace mask_digit_correctness_l199_199643

noncomputable def elephant_mask_digit : ℕ :=
  6
  
noncomputable def mouse_mask_digit : ℕ :=
  4

noncomputable def guinea_pig_mask_digit : ℕ :=
  8

noncomputable def panda_mask_digit : ℕ :=
  1

theorem mask_digit_correctness :
  (∃ (d1 d2 d3 d4 : ℕ), d1 * d1 = 16 ∧ d2 * d2 = 64 ∧ d3 * d3 = 49 ∧ d4 * d4 = 81) →
  elephant_mask_digit = 6 ∧ mouse_mask_digit = 4 ∧ guinea_pig_mask_digit = 8 ∧ panda_mask_digit = 1 :=
by
  -- skip the proof
  sorry

end mask_digit_correctness_l199_199643


namespace distance_amanda_to_kimberly_l199_199619

-- Define the given conditions
def amanda_speed : ℝ := 2 -- miles per hour
def amanda_time : ℝ := 3 -- hours

-- Prove that the distance is 6 miles
theorem distance_amanda_to_kimberly : amanda_speed * amanda_time = 6 := by
  sorry

end distance_amanda_to_kimberly_l199_199619


namespace count_three_digit_integers_with_two_same_digits_l199_199751

def digits (n : ℕ) : List ℕ := [n / 100, (n / 10) % 10, n % 10]

lemma digits_len (n : ℕ) (h : n < 1000) (h1 : n ≥ 100) : (digits n).length = 3 :=
begin 
    sorry
end

def at_least_two_same (n : ℕ) : Prop :=
  let d := digits n in
  d.length = 3 ∧ (d[0] = d[1] ∨ d[1] = d[2] ∨ d[0] = d[2])

theorem count_three_digit_integers_with_two_same_digits :
  (finset.filter at_least_two_same (finset.Icc 100 599)).card = 140 :=
by sorry

end count_three_digit_integers_with_two_same_digits_l199_199751


namespace total_items_washed_l199_199280

def towels := 15
def shirts := 10
def loads := 20

def items_per_load : Nat := towels + shirts
def total_items : Nat := items_per_load * loads

theorem total_items_washed : total_items = 500 :=
by
  rw [total_items, items_per_load]
  -- step expansion:
  -- unfold items_per_load
  -- calc 
  -- 15 + 10 = 25  -- from definition
  -- 25 * 20 = 500  -- from multiplication
  sorry

end total_items_washed_l199_199280


namespace monkeys_and_bananas_l199_199354

theorem monkeys_and_bananas :
  (∀ (m n t : ℕ), m * t = n → (∀ (m' n' t' : ℕ), n = m * (t / t') → n' = (m' * t') / t → n' = n → m' = m)) →
  (6 : ℕ) = 6 :=
by
  intros H
  let m := 6
  let n := 6
  let t := 6
  have H1 : m * t = n := by sorry
  let k := 18
  let t' := 18
  have H2 : n = m * (t / t') := by sorry
  let n' := 18
  have H3 : n' = (m * t') / t := by sorry
  have H4 : n' = n := by sorry
  exact H m n t H1 6 n' t' H2 H3 H4

end monkeys_and_bananas_l199_199354


namespace parabola_equation_l199_199724

theorem parabola_equation (m : ℝ) (focus : ℝ × ℝ) (M : ℝ × ℝ) 
  (h_vertex : (0, 0) = (0, 0))
  (h_focus : focus = (p, 0))
  (h_point : M = (1, m))
  (h_distance : dist M focus = 2) 
  : (forall x y : ℝ, y^2 = 4*x) :=
sorry

end parabola_equation_l199_199724


namespace inequality_proof_l199_199927

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l199_199927


namespace probability_more_ones_than_sixes_l199_199772

theorem probability_more_ones_than_sixes :
  (∃ (p : ℚ), p = 1673 / 3888 ∧ 
  (∃ (d : Fin 6 → ℕ), 
  (∀ i, d i ≤ 4) ∧ 
  (∃ d1 d6 : ℕ, (1 ≤ d1 + d6 ∧ d1 + d6 ≤ 5 ∧ d1 > d6)))) :=
sorry

end probability_more_ones_than_sixes_l199_199772


namespace range_of_k_if_intersection_empty_l199_199466

open Set

variable (k : ℝ)

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem range_of_k_if_intersection_empty (h : M ∩ N k = ∅) : k ≤ -1 :=
by {
  sorry
}

end range_of_k_if_intersection_empty_l199_199466


namespace number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l199_199054

section FiveFives

def five : ℕ := 5

-- Definitions for each number 1 to 17 using five fives.
def one : ℕ := (five / five) * (five / five)
def two : ℕ := (five / five) + (five / five)
def three : ℕ := (five * five - five) / five
def four : ℕ := (five - five / five) * (five / five)
def five_num : ℕ := five + (five - five) * (five / five)
def six : ℕ := five + (five + five) / (five + five)
def seven : ℕ := five + (five * five - five^2) / five
def eight : ℕ := (five + five + five) / five + five
def nine : ℕ := five + (five - five / five)
def ten : ℕ := five + five
def eleven : ℕ := (55 - 55 / five) / five
def twelve : ℕ := five * (five - five / five) / five
def thirteen : ℕ := (five * five - five - five) / five + five
def fourteen : ℕ := five + five + five - (five / five)
def fifteen : ℕ := five + five + five
def sixteen : ℕ := five + five + five + (five / five)
def seventeen : ℕ := five + five + five + ((five / five) + (five / five))

-- Proof statements to be provided
theorem number_one : one = 1 := sorry
theorem number_two : two = 2 := sorry
theorem number_three : three = 3 := sorry
theorem number_four : four = 4 := sorry
theorem number_five : five_num = 5 := sorry
theorem number_six : six = 6 := sorry
theorem number_seven : seven = 7 := sorry
theorem number_eight : eight = 8 := sorry
theorem number_nine : nine = 9 := sorry
theorem number_ten : ten = 10 := sorry
theorem number_eleven : eleven = 11 := sorry
theorem number_twelve : twelve = 12 := sorry
theorem number_thirteen : thirteen = 13 := sorry
theorem number_fourteen : fourteen = 14 := sorry
theorem number_fifteen : fifteen = 15 := sorry
theorem number_sixteen : sixteen = 16 := sorry
theorem number_seventeen : seventeen = 17 := sorry

end FiveFives

end number_one_number_two_number_three_number_four_number_five_number_six_number_seven_number_eight_number_nine_number_ten_number_eleven_number_twelve_number_thirteen_number_fourteen_number_fifteen_number_sixteen_number_seventeen_l199_199054


namespace fish_ratio_l199_199206

theorem fish_ratio (B T S Bo : ℕ) 
  (hBilly : B = 10) 
  (hTonyBilly : T = 3 * B) 
  (hSarahTony : S = T + 5) 
  (hBobbySarah : Bo = 2 * S) 
  (hTotalFish : Bo + S + T + B = 145) : 
  T / B = 3 :=
by sorry

end fish_ratio_l199_199206


namespace number_of_people_chose_pop_l199_199442

theorem number_of_people_chose_pop (total_people : ℕ) (angle_pop : ℕ) (h1 : total_people = 540) (h2 : angle_pop = 270) : (total_people * (angle_pop / 360)) = 405 := by
  sorry

end number_of_people_chose_pop_l199_199442


namespace inequality_proof_l199_199925

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l199_199925


namespace largest_n_unique_k_l199_199663

theorem largest_n_unique_k : 
  ∃ n : ℕ, (∀ k : ℤ, (5 / 12 : ℚ) < n / (n + k) ∧ n / (n + k) < (4 / 9 : ℚ) → k = 9) ∧ n = 7 :=
by
  sorry

end largest_n_unique_k_l199_199663


namespace central_angle_measure_l199_199366

theorem central_angle_measure (p : ℝ) (x : ℝ) (h1 : p = 1 / 8) (h2 : p = x / 360) : x = 45 :=
by
  -- skipping the proof
  sorry

end central_angle_measure_l199_199366


namespace composite_shape_perimeter_l199_199315

theorem composite_shape_perimeter :
  let r1 := 2.1
  let r2 := 3.6
  let π_approx := 3.14159
  let total_perimeter := π_approx * (r1 + r2)
  total_perimeter = 18.31 :=
by
  let radius1 := 2.1
  let radius2 := 3.6
  let total_radius := radius1 + radius2
  let pi_value := 3.14159
  let perimeter := pi_value * total_radius
  have calculation : perimeter = 18.31 := sorry
  exact calculation

end composite_shape_perimeter_l199_199315


namespace simplify_polynomial_l199_199891

theorem simplify_polynomial (x y : ℝ) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 :=
by
  sorry

end simplify_polynomial_l199_199891


namespace problem_statement_l199_199616

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

noncomputable def given_conditions (a : ℕ → ℤ) : Prop :=
a 2 = 2 ∧ a 3 = 4

theorem problem_statement (a : ℕ → ℤ) (h1 : given_conditions a) (h2 : arithmetic_sequence a) :
  a 10 = 18 := by
  sorry

end problem_statement_l199_199616


namespace boys_count_at_table_l199_199359

-- Definitions from conditions
def children_count : ℕ := 13
def alternates (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- The problem to be proven in Lean:
theorem boys_count_at_table : ∃ b g : ℕ, b + g = children_count ∧ alternates b ∧ alternates g ∧ b = 7 :=
by
  sorry

end boys_count_at_table_l199_199359


namespace triangle_angle_equality_l199_199117

theorem triangle_angle_equality (A B C : ℝ) (h : ∃ (x : ℝ), x^2 - x * (Real.cos A * Real.cos B) - Real.cos (C / 2)^2 = 0 ∧ x = 1) : A = B :=
by {
  sorry
}

end triangle_angle_equality_l199_199117


namespace carpet_area_l199_199199

def room_length_ft := 16
def room_width_ft := 12
def column_side_ft := 2
def ft_to_inches := 12

def room_length_in := room_length_ft * ft_to_inches
def room_width_in := room_width_ft * ft_to_inches
def column_side_in := column_side_ft * ft_to_inches

def room_area_in_sq := room_length_in * room_width_in
def column_area_in_sq := column_side_in * column_side_in

def remaining_area_in_sq := room_area_in_sq - column_area_in_sq

theorem carpet_area : remaining_area_in_sq = 27072 := by
  sorry

end carpet_area_l199_199199


namespace scientific_notation_correct_l199_199810

-- Define the number to be converted
def number : ℕ := 3790000

-- Define the correct scientific notation representation
def scientific_notation : ℝ := 3.79 * (10 ^ 6)

-- Statement to prove that number equals scientific_notation
theorem scientific_notation_correct :
  number = 3790000 → scientific_notation = 3.79 * (10 ^ 6) :=
by
  sorry

end scientific_notation_correct_l199_199810


namespace prob_more_1s_than_6s_l199_199758

noncomputable def probability_more_ones_than_sixes (n : ℕ) : ℚ :=
  let total_outcomes := 6^n
  let equal_1s_6s :=  sum_finsupp (λ k1 k6 : _n_, if (k1 = k6) 
    then binom n k1 * binom (n - k1) k6 * (4 ^ (n - k1 - k6)) else 0)
  let prob_equal := equal_1s_6s / total_outcomes
  let final_probability := (1 - prob_equal) / 2
  final_probability

theorem prob_more_1s_than_6s :
  probability_more_ones_than_sixes 5 = 2676 / 7776 :=
sorry

end prob_more_1s_than_6s_l199_199758


namespace intersection_points_vary_with_a_l199_199824

-- Define the lines
def line1 (x : ℝ) : ℝ := x + 1
def line2 (a x : ℝ) : ℝ := a * x + 1

-- Prove that the number of intersection points varies with a
theorem intersection_points_vary_with_a (a : ℝ) : 
  (∃ x : ℝ, line1 x = line2 a x) ↔ 
    (if a = 1 then true else true) :=
by 
  sorry

end intersection_points_vary_with_a_l199_199824


namespace paddington_more_goats_l199_199978

theorem paddington_more_goats (W P total : ℕ) (hW : W = 140) (hTotal : total = 320) (hTotalGoats : W + P = total) : P - W = 40 :=
by
  sorry

end paddington_more_goats_l199_199978


namespace combined_volleyball_percentage_l199_199826

theorem combined_volleyball_percentage (students_north: ℕ) (students_south: ℕ)
(percent_volleyball_north percent_volleyball_south: ℚ)
(H1: students_north = 1800) (H2: percent_volleyball_north = 0.25)
(H3: students_south = 2700) (H4: percent_volleyball_south = 0.35):
  (((students_north * percent_volleyball_north) + (students_south * percent_volleyball_south))
  / (students_north + students_south) * 100) = 31 := 
  sorry

end combined_volleyball_percentage_l199_199826


namespace initial_concentration_l199_199364

theorem initial_concentration (f : ℚ) (C : ℚ) (h₀ : f = 0.7142857142857143) (h₁ : (1 - f) * C + f * 0.25 = 0.35) : C = 0.6 :=
by
  rw [h₀] at h₁
  -- The proof will follow the steps to solve for C
  sorry

end initial_concentration_l199_199364


namespace find_q_l199_199740

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199740


namespace cost_price_l199_199193

/-- A person buys an article at some price. 
They sell the article to make a profit of 24%. 
The selling price of the article is Rs. 595.2. 
Prove that the cost price (CP) is Rs. 480. -/
theorem cost_price (SP CP : ℝ) (h1 : SP = 595.2) (h2 : SP = CP * (1 + 0.24)) : CP = 480 := 
by sorry 

end cost_price_l199_199193


namespace cos_8_minus_sin_8_l199_199601

theorem cos_8_minus_sin_8 (α m : ℝ) (h : Real.cos (2 * α) = m) :
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 :=
by
  sorry

end cos_8_minus_sin_8_l199_199601


namespace quadrilateral_angles_arith_prog_l199_199894

theorem quadrilateral_angles_arith_prog {x a b c : ℕ} (d : ℝ):
  (x^2 = 8^2 + 7^2 + 2 * 8 * 7 * Real.sin (3 * d)) →
  x = a + Real.sqrt b + Real.sqrt c →
  x = Real.sqrt 113 →
  a + b + c = 113 :=
by
  sorry

end quadrilateral_angles_arith_prog_l199_199894


namespace op_example_l199_199524

def op (a b : ℚ) : ℚ := a * b / (a + b)

theorem op_example : op (op 3 5) (op 5 4) = 60 / 59 := by
  sorry

end op_example_l199_199524


namespace binomial_square_evaluation_l199_199332

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l199_199332


namespace sin_2theta_plus_pi_div_2_l199_199248

theorem sin_2theta_plus_pi_div_2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4)
    (h_tan2θ : Real.tan (2 * θ) = Real.cos θ / (2 - Real.sin θ)) :
    Real.sin (2 * θ + π / 2) = 7 / 8 :=
sorry

end sin_2theta_plus_pi_div_2_l199_199248


namespace calculate_expression_l199_199780

variables {a b c : ℤ}
variable (h1 : 5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c) -- a, b, c are multiples of 5
variable (h2 : a < b ∧ b < c) -- a < b < c
variable (h3 : c = a + 10) -- c = a + 10

theorem calculate_expression :
  (a - b) * (a - c) / (b - c) = -10 :=
by
  sorry

end calculate_expression_l199_199780


namespace scientific_notation_of_56_point_5_million_l199_199481

-- Definitions based on conditions
def million : ℝ := 10^6
def number_in_millions : ℝ := 56.5 * million

-- Statement to be proved
theorem scientific_notation_of_56_point_5_million : 
  number_in_millions = 5.65 * 10^7 :=
sorry

end scientific_notation_of_56_point_5_million_l199_199481


namespace car_and_cyclist_speeds_and_meeting_point_l199_199085

/-- 
(1) Distance between points $A$ and $B$ is $80 \mathrm{~km}$.
(2) After one hour, the distance between them reduces to $24 \mathrm{~km}$.
(3) The cyclist takes a 1-hour rest but they meet $90$ minutes after their departure.
-/
def initial_distance : ℝ := 80 -- km
def distance_after_one_hour : ℝ := 24 -- km apart after 1 hour
def cyclist_rest_duration : ℝ := 1 -- hour
def meeting_time : ℝ := 1.5 -- hours (90 minutes after departure)

def car_speed : ℝ := 40 -- km/hr
def cyclist_speed : ℝ := 16 -- km/hr

theorem car_and_cyclist_speeds_and_meeting_point :
  initial_distance = 80 → 
  distance_after_one_hour = 24 → 
  cyclist_rest_duration = 1 → 
  meeting_time = 1.5 → 
  car_speed = 40 ∧ cyclist_speed = 16 ∧ meeting_point_from_A = 60 ∧ meeting_point_from_B = 20 :=
by
  sorry

end car_and_cyclist_speeds_and_meeting_point_l199_199085


namespace find_x_l199_199221

theorem find_x : 2^4 + 3 = 5^2 - 6 :=
by
  sorry

end find_x_l199_199221


namespace probability_three_marbles_same_color_l199_199362

-- Definitions of the counts of each color
def red_marbles : ℕ := 3
def white_marbles : ℕ := 4
def blue_marbles : ℕ := 5
def green_marbles : ℕ := 2

-- Total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

-- Probabilities for drawing three marbles of the same color
def P_all_red : ℚ := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2))
def P_all_white : ℚ := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2))
def P_all_blue : ℚ := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2))
def P_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2))

-- Combined probability of drawing three marbles of the same color
def P_same_color : ℚ := P_all_red + P_all_white + P_all_blue + P_all_green

-- The required theorem
theorem probability_three_marbles_same_color :
  P_same_color = 15 / 364 := by
  sorry  -- Proof goes here but is not required by the task


end probability_three_marbles_same_color_l199_199362


namespace find_y_l199_199006

def is_divisible_by (x y : ℕ) : Prop := x % y = 0

def ends_with_digit (x : ℕ) (d : ℕ) : Prop :=
  x % 10 = d

theorem find_y (y : ℕ) :
  (y > 0) ∧
  is_divisible_by y 4 ∧
  is_divisible_by y 5 ∧
  is_divisible_by y 7 ∧
  is_divisible_by y 13 ∧
  ¬ is_divisible_by y 8 ∧
  ¬ is_divisible_by y 15 ∧
  ¬ is_divisible_by y 50 ∧
  ends_with_digit y 0
  → y = 1820 :=
sorry

end find_y_l199_199006


namespace dice_probability_not_all_same_l199_199063

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l199_199063


namespace sonia_and_joss_time_spent_moving_l199_199493

def total_time_spent_moving (fill_time_per_trip drive_time_per_trip trips : ℕ) :=
  (fill_time_per_trip + drive_time_per_trip) * trips

def total_time_in_hours (total_time_in_minutes : ℕ) : ℚ :=
  total_time_in_minutes / 60

theorem sonia_and_joss_time_spent_moving :
  total_time_in_hours (total_time_spent_moving 15 30 6) = 4.5 :=
by
  sorry

end sonia_and_joss_time_spent_moving_l199_199493


namespace train_to_platform_ratio_l199_199998

-- Define the given conditions as assumptions
def speed_kmh : ℕ := 54 -- speed of the train in km/hr
def train_length_m : ℕ := 450 -- length of the train in meters
def crossing_time_min : ℕ := 1 -- time to cross the platform in minutes

-- Conversion from km/hr to m/min
def speed_mpm : ℕ := (speed_kmh * 1000) / 60

-- Calculate the total distance covered in one minute
def total_distance_m : ℕ := speed_mpm * crossing_time_min

-- Define the length of the platform
def platform_length_m : ℕ := total_distance_m - train_length_m

-- The proof statement to show the ratio of the lengths
theorem train_to_platform_ratio : train_length_m = platform_length_m :=
by 
  -- following from the definition of platform_length_m
  sorry

end train_to_platform_ratio_l199_199998


namespace rectangle_area_l199_199355

theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 56) :
  l * b = 147 := by
  sorry

end rectangle_area_l199_199355


namespace average_infection_l199_199679

theorem average_infection (x : ℕ) (h : 1 + 2 * x + x^2 = 121) : x = 10 :=
by
  sorry -- Proof to be filled.

end average_infection_l199_199679


namespace max_sum_of_solutions_l199_199218

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l199_199218


namespace students_in_zack_classroom_l199_199050

theorem students_in_zack_classroom 
(T M Z : ℕ)
(h1 : T = M)
(h2 : Z = (T + M) / 2)
(h3 : T + M + Z = 69) :
Z = 23 :=
by
  sorry

end students_in_zack_classroom_l199_199050


namespace find_m_l199_199139

-- Definition of vectors in terms of the condition
def vec_a (m : ℝ) : ℝ × ℝ := (2 * m + 1, m)
def vec_b (m : ℝ) : ℝ × ℝ := (1, m)

-- Condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Problem statement: find m such that vec_a is perpendicular to vec_b
theorem find_m (m : ℝ) (h : perpendicular (vec_a m) (vec_b m)) : m = -1 := by
  sorry

end find_m_l199_199139


namespace probability_more_ones_than_sixes_l199_199766

theorem probability_more_ones_than_sixes (total_dice : ℕ) (sides_of_dice : ℕ) 
  (ones : ℕ) (sixes : ℕ) (total_outcomes : ℕ) (equal_outcomes : ℕ) : 
  (total_dice = 5) → 
  (sides_of_dice = 6) → 
  (total_outcomes = 6^total_dice) → 
  (equal_outcomes = 1024 + 1280 + 120) → 
  (ones > sixes) → 
  (prob_more_ones_than_sixes : ℚ) → 
  prob_more_ones_than_sixes = (1/2) * (1 - (equal_outcomes / total_outcomes)) := 
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4],
  sorry,
end

end probability_more_ones_than_sixes_l199_199766


namespace square_area_l199_199356

theorem square_area (side_length : ℕ) (h : side_length = 16) : side_length * side_length = 256 := by
  sorry

end square_area_l199_199356


namespace cost_of_rice_l199_199055

theorem cost_of_rice (x : ℝ) 
  (h : 5 * x + 3 * 5 = 25) : x = 2 :=
by {
  sorry
}

end cost_of_rice_l199_199055


namespace proof_problem_l199_199514

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l199_199514


namespace problem_statement_l199_199415

-- Definitions of conditions
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Statement of the problem
theorem problem_statement (a : ℝ) (h1 : p a) (h2 : q a) : (¬ p a) → (¬ q a) → ∃ x, ¬ (¬ q x) → (¬ (¬ p x)) :=
by
  sorry

end problem_statement_l199_199415


namespace geometric_sequence_common_ratio_l199_199799

theorem geometric_sequence_common_ratio (a₁ : ℚ) (q : ℚ) 
  (S : ℕ → ℚ) (hS : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h : 8 * S 6 = 7 * S 3) : 
  q = -1/2 :=
sorry

end geometric_sequence_common_ratio_l199_199799


namespace deborah_international_letters_l199_199550

theorem deborah_international_letters (standard_postage : ℝ) 
                                      (additional_charge : ℝ) 
                                      (total_letters : ℕ) 
                                      (total_cost : ℝ) 
                                      (h_standard_postage: standard_postage = 1.08)
                                      (h_additional_charge: additional_charge = 0.14)
                                      (h_total_letters: total_letters = 4)
                                      (h_total_cost: total_cost = 4.60) :
                                      ∃ (x : ℕ), x = 2 :=
by
  sorry

end deborah_international_letters_l199_199550


namespace perpendicular_point_sets_l199_199599

-- Define what it means for a set to be a perpendicular point set
def perpendicular_point_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₁ y₁ : ℝ), (x₁, y₁) ∈ M → ∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ M ∧ x₁ * x₂ + y₁ * y₂ = 0

-- Define sets M1, M2, M3, and M4
def M1 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, 1 / x^2) }
def M2 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), x > 0 ∧ p = (x, log x / log 2) }
def M3 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, 2^x - 2) }
def M4 : Set (ℝ × ℝ) := { p | ∃ (x : ℝ), p = (x, sin x + 1) }

theorem perpendicular_point_sets :
  perpendicular_point_set M1 ∧
  ¬ perpendicular_point_set M2 ∧
  perpendicular_point_set M3 ∧
  perpendicular_point_set M4 :=
by {
  sorry -- Proof goes here
}

end perpendicular_point_sets_l199_199599


namespace roots_not_in_interval_l199_199270

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l199_199270


namespace at_least_three_double_marked_l199_199788

noncomputable def grid := Matrix (Fin 10) (Fin 20) ℕ -- 10x20 matrix with natural numbers

def is_red_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 20), k₁ ≠ k₂ ∧ (g i k₁) ≤ g i j ∧ (g i k₂) ≤ g i j ∧ ∀ (k : Fin 20), (k ≠ k₁ ∧ k ≠ k₂) → g i k ≤ g i j

def is_blue_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  ∃ (k₁ k₂ : Fin 10), k₁ ≠ k₂ ∧ (g k₁ j) ≤ g i j ∧ (g k₂ j) ≤ g i j ∧ ∀ (k : Fin 10), (k ≠ k₁ ∧ k ≠ k₂) → g k j ≤ g i j

def is_double_marked (g : grid) (i : Fin 10) (j : Fin 20) : Prop :=
  is_red_marked g i j ∧ is_blue_marked g i j

theorem at_least_three_double_marked (g : grid) :
  (∃ (i₁ i₂ i₃ : Fin 10) (j₁ j₂ j₃ : Fin 20), i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₃ ≠ i₁ ∧ 
    j₁ ≠ j₂ ∧ j₂ ≠ j₃ ∧ j₃ ≠ j₁ ∧ is_double_marked g i₁ j₁ ∧ is_double_marked g i₂ j₂ ∧ is_double_marked g i₃ j₃) :=
sorry

end at_least_three_double_marked_l199_199788


namespace shaded_fraction_is_one_fourth_l199_199830

def quilt_block_shaded_fraction : ℚ :=
  let total_unit_squares := 16
  let triangles_per_unit_square := 2
  let shaded_triangles := 8
  let shaded_unit_squares := shaded_triangles / triangles_per_unit_square
  shaded_unit_squares / total_unit_squares

theorem shaded_fraction_is_one_fourth :
  quilt_block_shaded_fraction = 1 / 4 :=
sorry

end shaded_fraction_is_one_fourth_l199_199830


namespace x_squared_minus_y_squared_l199_199005

open Real

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 4/9)
  (h2 : x - y = 2/9) :
  x^2 - y^2 = 8/81 :=
by
  sorry

end x_squared_minus_y_squared_l199_199005


namespace number_of_solid_shapes_is_three_l199_199879

-- Define the geometric shapes and their dimensionality
inductive GeomShape
| square : GeomShape
| cuboid : GeomShape
| circle : GeomShape
| sphere : GeomShape
| cone : GeomShape

def isSolid (shape : GeomShape) : Bool :=
  match shape with
  | GeomShape.square => false
  | GeomShape.cuboid => true
  | GeomShape.circle => false
  | GeomShape.sphere => true
  | GeomShape.cone => true

-- Formal statement of the problem
theorem number_of_solid_shapes_is_three :
  (List.filter isSolid [GeomShape.square, GeomShape.cuboid, GeomShape.circle, GeomShape.sphere, GeomShape.cone]).length = 3 :=
by
  -- proof omitted
  sorry

end number_of_solid_shapes_is_three_l199_199879


namespace range_of_a_l199_199219

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 3| - |x + 1| ≤ a^2 - 3 * a) ↔ a ≤ -1 ∨ 4 ≤ a := 
sorry

end range_of_a_l199_199219


namespace spring_outing_students_l199_199899

variable (x y : ℕ)

theorem spring_outing_students (hx : x % 10 = 0) (hy : y % 10 = 0) (h1 : x + y = 1008) (h2 : y - x = 133) :
  x = 437 ∧ y = 570 :=
by
  sorry

end spring_outing_students_l199_199899


namespace vacant_seats_l199_199127

theorem vacant_seats (total_seats : ℕ) (filled_percent vacant_percent : ℚ) 
  (h_total : total_seats = 600)
  (h_filled_percent : filled_percent = 75)
  (h_vacant_percent : vacant_percent = 100 - filled_percent)
  (h_vacant_percent_25 : vacant_percent = 25) :
  (25 / 100) * 600 = 150 :=
by 
  -- this is the final answer we want to prove, replace with sorry to skip the proof just for statement validation
  sorry

end vacant_seats_l199_199127


namespace equivalence_of_statements_l199_199177

-- Variables used in the statements
variable (P Q : Prop)

-- Proof problem statement
theorem equivalence_of_statements : (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_of_statements_l199_199177


namespace proof_simplify_expression_l199_199627

noncomputable def simplify_expression (a b : ℝ) : ℝ :=
  (a / b + b / a)^2 - 1 / (a^2 * b^2)

theorem proof_simplify_expression 
  (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = a + b) :
  simplify_expression a b = 2 / (a * b) := by
  sorry

end proof_simplify_expression_l199_199627


namespace smallest_value_not_defined_l199_199512

noncomputable def smallest_undefined_x : ℝ :=
  let a := 6
  let b := -37
  let c := 5
  let discriminant := b * b - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let x1 := (-b + sqrt_discriminant) / (2 * a)
  let x2 := (-b - sqrt_discriminant) / (2 * a)
  if x1 < x2 then x1 else x2

theorem smallest_value_not_defined :
  smallest_undefined_x = 0.1383 :=
by sorry

end smallest_value_not_defined_l199_199512


namespace inversely_proportional_l199_199858

theorem inversely_proportional (X Y K : ℝ) (h : X * Y = K - 1) (hK : K > 1) : 
  (∃ c : ℝ, ∀ x y : ℝ, x * y = c) :=
sorry

end inversely_proportional_l199_199858


namespace total_floors_combined_l199_199640

-- Let C be the number of floors in the Chrysler Building
-- Let L be the number of floors in the Leeward Center
-- Given that C = 23 and C = L + 11
-- Prove that the total floors in both buildings combined equals 35

theorem total_floors_combined (C L : ℕ) (h1 : C = 23) (h2 : C = L + 11) : C + L = 35 :=
by
  sorry

end total_floors_combined_l199_199640


namespace simplify_and_evaluate_l199_199033

def expr (a b : ℤ) := -a^2 * b + (3 * a * b^2 - a^2 * b) - 2 * (2 * a * b^2 - a^2 * b)

theorem simplify_and_evaluate : expr (-1) (-2) = -4 := by
  sorry

end simplify_and_evaluate_l199_199033


namespace total_swordfish_caught_l199_199031

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l199_199031


namespace smallest_divisor_l199_199070

theorem smallest_divisor (N D : ℕ) (hN : N = D * 7) (hD : D > 0) (hsq : (N / D) = 7) :
  D = 7 :=
by 
  sorry

end smallest_divisor_l199_199070


namespace geom_seq_a_n_l199_199919

theorem geom_seq_a_n (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -1) 
  (h_a7 : a 7 = -9) :
  a 5 = -3 :=
sorry

end geom_seq_a_n_l199_199919


namespace solution_set_abs_inequality_l199_199499

theorem solution_set_abs_inequality :
  { x : ℝ | |x - 2| - |2 * x - 1| > 0 } = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_abs_inequality_l199_199499


namespace marble_probability_l199_199844

theorem marble_probability (g w r b : ℕ) (h_g : g = 4) (h_w : w = 3) (h_r : r = 5) (h_b : b = 6) :
  (g + w + r + b = 18) → (g + w = 7) → (7 / 18 = 7 / 18) :=
by
  sorry

end marble_probability_l199_199844


namespace relationship_between_3a_3b_4a_l199_199081

variable (a b : ℝ)
variable (h : a > b)
variable (hb : b > 0)

theorem relationship_between_3a_3b_4a (a b : ℝ) (h : a > b) (hb : b > 0) :
  3 * b < 3 * a ∧ 3 * a < 4 * a := 
by
  sorry

end relationship_between_3a_3b_4a_l199_199081


namespace probability_XOXOXOX_is_one_over_thirty_five_l199_199574

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def probability_of_arrangement : ℤ :=
  let total_arrangements := binomial_coefficient 7 4
  let favorable_outcomes := 1
  favorable_outcomes / total_arrangements

theorem probability_XOXOXOX_is_one_over_thirty_five :
  probability_of_arrangement = (1 : ℤ) / 35 := 
  by
  sorry

end probability_XOXOXOX_is_one_over_thirty_five_l199_199574


namespace new_cost_percentage_l199_199076

variable (t b : ℝ)

-- Define the original cost
def original_cost : ℝ := t * b ^ 4

-- Define the new cost when b is doubled
def new_cost : ℝ := t * (2 * b) ^ 4

-- The theorem statement
theorem new_cost_percentage (t b : ℝ) : new_cost t b = 16 * original_cost t b := 
by
  -- Proof steps are skipped
  sorry

end new_cost_percentage_l199_199076


namespace probability_XOXOXOX_l199_199579

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l199_199579


namespace percentage_seeds_from_dandelions_l199_199101

def Carla_sunflowers := 6
def Carla_dandelions := 8
def seeds_per_sunflower := 9
def seeds_per_dandelion := 12

theorem percentage_seeds_from_dandelions :
  96 / 150 * 100 = 64 := by
  sorry

end percentage_seeds_from_dandelions_l199_199101


namespace trapezoid_midsegment_l199_199130

theorem trapezoid_midsegment (a b : ℝ)
  (AB CD E F: ℝ) -- we need to indicate that E and F are midpoints somehow
  (h1 : AB = a)
  (h2 : CD = b)
  (h3 : AB = CD) 
  (h4 : E = (AB + CD) / 2)
  (h5 : F = (CD + AB) / 2) : 
  EF = (1/2) * (a - b) := sorry

end trapezoid_midsegment_l199_199130


namespace boys_from_other_communities_l199_199182

theorem boys_from_other_communities (total_boys : ℕ) (percent_muslims percent_hindus percent_sikhs : ℕ) 
    (h_total_boys : total_boys = 300)
    (h_percent_muslims : percent_muslims = 44)
    (h_percent_hindus : percent_hindus = 28)
    (h_percent_sikhs : percent_sikhs = 10) :
  ∃ (percent_others : ℕ), percent_others = 100 - (percent_muslims + percent_hindus + percent_sikhs) ∧ 
                             (percent_others * total_boys / 100) = 54 := 
by 
  sorry

end boys_from_other_communities_l199_199182


namespace incorrect_divisor_l199_199951

theorem incorrect_divisor (D x : ℕ) (h1 : D = 24 * x) (h2 : D = 48 * 36) : x = 72 := by
  sorry

end incorrect_divisor_l199_199951


namespace harry_ron_difference_l199_199002

-- Define the amounts each individual paid
def harry_paid : ℕ := 150
def ron_paid : ℕ := 180
def hermione_paid : ℕ := 210

-- Define the total amount
def total_paid : ℕ := harry_paid + ron_paid + hermione_paid

-- Define the amount each should have paid
def equal_share : ℕ := total_paid / 3

-- Define the amount Harry owes to Hermione
def harry_owes : ℕ := equal_share - harry_paid

-- Define the amount Ron owes to Hermione
def ron_owes : ℕ := equal_share - ron_paid

-- Define the difference between what Harry and Ron owe Hermione
def difference : ℕ := harry_owes - ron_owes

-- Prove that the difference is 30
theorem harry_ron_difference : difference = 30 := by
  sorry

end harry_ron_difference_l199_199002


namespace ellipse_parabola_common_point_l199_199947

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔  -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end ellipse_parabola_common_point_l199_199947


namespace find_m_if_f_even_l199_199232

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 2) * x + (m^2 - 7 * m + 12)

theorem find_m_if_f_even :
  (∀ x : ℝ, f m (-x) = f m x) → m = 2 :=
by 
  intro h
  sorry

end find_m_if_f_even_l199_199232


namespace find_remainder_l199_199269

theorem find_remainder (S : Finset ℕ) (h : ∀ n ∈ S, ∃ m, n^2 + 10 * n - 2010 = m^2) :
  (S.sum id) % 1000 = 304 := by
  sorry

end find_remainder_l199_199269


namespace find_value_of_M_l199_199274

theorem find_value_of_M (a b M : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = M) (h4 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y ≤ (M^2) / 4) (h5 : ∀ x y : ℝ, x > 0 → y > 0 → (x + y = M) → x * y = 2) :
  M = 2 * Real.sqrt 2 :=
by
  sorry

end find_value_of_M_l199_199274


namespace geometric_sequence_common_ratio_l199_199444

theorem geometric_sequence_common_ratio
  (a₁ a₂ a₃ : ℝ) (q : ℝ) 
  (h₀ : 0 < a₁) 
  (h₁ : a₂ = a₁ * q) 
  (h₂ : a₃ = a₁ * q^2) 
  (h₃ : 2 * a₁ + a₂ = 2 * (1 / 2 * a₃)) 
  : q = 2 := 
sorry

end geometric_sequence_common_ratio_l199_199444


namespace peanuts_added_correct_l199_199310

-- Define the initial and final number of peanuts
def initial_peanuts : ℕ := 4
def final_peanuts : ℕ := 12

-- Define the number of peanuts Mary added
def peanuts_added : ℕ := final_peanuts - initial_peanuts

-- State the theorem that proves the number of peanuts Mary added
theorem peanuts_added_correct : peanuts_added = 8 :=
by
  -- Add the proof here
  sorry

end peanuts_added_correct_l199_199310


namespace find_b_l199_199648

theorem find_b (a b c : ℝ) (h₁ : c = 3)
  (h₂ : -a / 3 = c)
  (h₃ : -a / 3 = 1 + a + b + c) :
  b = -16 :=
by
  -- The solution steps are not necessary to include here.
  sorry

end find_b_l199_199648


namespace total_swordfish_caught_correct_l199_199029

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l199_199029


namespace problem_part_a_problem_part_b_l199_199180

noncomputable def circular_permutations (n : ℕ) : ℕ :=
  (Fintype.card (Equiv.Perm (Fin n))) / n

theorem problem_part_a : circular_permutations 7 = 720 := by
  sorry

noncomputable def necklace_count (n : ℕ) : ℕ :=
  circular_permutations n / 2

theorem problem_part_b : necklace_count 7 = 360 := by
  sorry

end problem_part_a_problem_part_b_l199_199180


namespace n_to_the_4_plus_4_to_the_n_composite_l199_199635

theorem n_to_the_4_plus_4_to_the_n_composite (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + 4^n) := 
sorry

end n_to_the_4_plus_4_to_the_n_composite_l199_199635


namespace baker_cakes_l199_199385

theorem baker_cakes (initial_cakes sold_cakes remaining_cakes final_cakes new_cakes : ℕ)
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111)
  (h4 : new_cakes = final_cakes - (initial_cakes - sold_cakes)) :
  new_cakes = 76 :=
by {
  sorry
}

end baker_cakes_l199_199385


namespace trig_identity_l199_199696

theorem trig_identity :
  (4 * (1 / 2) - Real.sqrt 2 * (Real.sqrt 2 / 2) - Real.sqrt 3 * (Real.sqrt 3 / 3) + 2 * (Real.sqrt 3 / 2)) = Real.sqrt 3 :=
by sorry

end trig_identity_l199_199696


namespace complex_root_cubic_l199_199779

theorem complex_root_cubic (a b q r : ℝ) (h_b_ne_zero : b ≠ 0)
  (h_root : (Polynomial.C a + Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C a - Polynomial.C b * Polynomial.C I) * 
             (Polynomial.C (-2 * a)) 
             = Polynomial.X^3 + Polynomial.C q * Polynomial.X + Polynomial.C r) :
  q = b^2 - 3 * a^2 :=
sorry

end complex_root_cubic_l199_199779


namespace right_regular_prism_impossible_sets_l199_199341

-- Define a function to check if a given set of numbers {x, y, z} forms an invalid right regular prism
def not_possible (x y z : ℕ) : Prop := (x^2 + y^2 ≤ z^2)

-- Define individual propositions for the given sets of numbers
def set_a : Prop := not_possible 3 4 6
def set_b : Prop := not_possible 5 5 8
def set_e : Prop := not_possible 7 8 12

-- Define our overall proposition that these sets cannot be the lengths of the external diagonals of a right regular prism
theorem right_regular_prism_impossible_sets : 
  set_a ∧ set_b ∧ set_e :=
by
  -- Proof is omitted
  sorry

end right_regular_prism_impossible_sets_l199_199341


namespace factor_correct_l199_199556

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l199_199556


namespace polygon_sides_l199_199871

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end polygon_sides_l199_199871


namespace Q_as_sum_of_squares_Q_sum_of_squares_zero_l199_199043

variables {R : Type*} [CommRing R] (x₁ x₂ x₃ x₄ : R)

def Q (x₁ x₂ x₃ x₄ : R) : R :=
  4 * (x₁^2 + x₂^2 + x₃^2 + x₄^2) - (x₁ + x₂ + x₃ + x₄)^2

theorem Q_as_sum_of_squares :
  Q x₁ x₂ x₃ x₄ =
    (x₁ + x₂ - x₃ - x₄)^2 +
    (x₁ - x₂ + x₃ - x₄)^2 +
    (x₁ - x₂ - x₃ + x₄)^2 :=
by sorry

theorem Q_sum_of_squares_zero (P₁ P₂ P₃ P₄ : R → R → R → R → R) :
  (∀ x₁ x₂ x₃ x₄, Q x₁ x₂ x₃ x₄ = P₁ x₁ x₂ x₃ x₄^2 + P₂ x₁ x₂ x₃ x₄^2 + P₃ x₁ x₂ x₃ x₄^2 + P₄ x₁ x₂ x₃ x₄^2) →
  (∃ i, ∀ x₁ x₂ x₃ x₄, P₁ x₁ x₂ x₃ x₄ = 0 ∧ P₂ x₁ x₂ x₃ x₄ = 0 ∧ P₃ x₁ x₂ x₃ x₄ = 0 ∧ P₄ x₁ x₂ x₃ x₄ = 0) :=
by sorry

end Q_as_sum_of_squares_Q_sum_of_squares_zero_l199_199043


namespace binomial_square_evaluation_l199_199333

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l199_199333


namespace cube_volume_l199_199367

-- Define the surface area constant
def surface_area : ℝ := 725.9999999999998

-- Define the formula for surface area of a cube and solve for volume given the conditions
theorem cube_volume (SA : ℝ) (h : SA = surface_area) : 11^3 = 1331 :=
by sorry

end cube_volume_l199_199367


namespace repeating_decimal_as_fraction_l199_199110

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end repeating_decimal_as_fraction_l199_199110


namespace upper_limit_opinion_l199_199885

theorem upper_limit_opinion (w : ℝ) 
  (H1 : 61 < w ∧ w < 72) 
  (H2 : 60 < w ∧ w < 70) 
  (H3 : (61 + w) / 2 = 63) : w = 65 := 
by
  sorry

end upper_limit_opinion_l199_199885


namespace bird_families_flew_away_to_Asia_l199_199851

-- Defining the given conditions
def Total_bird_families_flew_away_for_winter : ℕ := 118
def Bird_families_flew_away_to_Africa : ℕ := 38

-- Proving the main statement
theorem bird_families_flew_away_to_Asia : 
  (Total_bird_families_flew_away_for_winter - Bird_families_flew_away_to_Africa) = 80 :=
by
  sorry

end bird_families_flew_away_to_Asia_l199_199851


namespace largest_divisor_of_n_l199_199250

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 127 ∣ n^3) : 127 ∣ n :=
sorry

end largest_divisor_of_n_l199_199250


namespace remainder_of_3456_div_97_l199_199665

theorem remainder_of_3456_div_97 :
  3456 % 97 = 61 :=
by
  sorry

end remainder_of_3456_div_97_l199_199665


namespace area_of_large_hexagon_eq_270_l199_199204

noncomputable def area_large_hexagon (area_shaded : ℝ) (n_small_hexagons_shaded : ℕ) (n_small_hexagons_large : ℕ): ℝ :=
  let area_one_small_hexagon := area_shaded / n_small_hexagons_shaded
  area_one_small_hexagon * n_small_hexagons_large

theorem area_of_large_hexagon_eq_270 :
  area_large_hexagon 180 6 7 = 270 := by
  sorry

end area_of_large_hexagon_eq_270_l199_199204


namespace positive_distinct_solutions_of_system_l199_199290

variables {a b x y z : ℝ}

theorem positive_distinct_solutions_of_system
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : xy = z^2) :
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by
  sorry

end positive_distinct_solutions_of_system_l199_199290


namespace solve_linear_combination_l199_199583

theorem solve_linear_combination (x y z : ℤ) 
    (h1 : x + 2 * y - z = 8) 
    (h2 : 2 * x - y + z = 18) : 
    8 * x + y + z = 70 := 
by 
    sorry

end solve_linear_combination_l199_199583


namespace greatest_possible_int_diff_l199_199144

theorem greatest_possible_int_diff (x a y b : ℝ) 
    (hx : 3 < x ∧ x < 4) 
    (ha : 4 < a ∧ a < x) 
    (hy : 6 < y ∧ y < 8) 
    (hb : 8 < b ∧ b < y) 
    (h_ineq : a^2 + b^2 > x^2 + y^2) : 
    abs (⌊x⌋ - ⌈y⌉) = 2 :=
sorry

end greatest_possible_int_diff_l199_199144


namespace vanessas_mother_picked_14_carrots_l199_199166

-- Define the problem parameters
variable (V : Nat := 17)  -- Vanessa picked 17 carrots
variable (G : Nat := 24)  -- Total good carrots
variable (B : Nat := 7)   -- Total bad carrots

-- Define the proof goal: Vanessa's mother picked 14 carrots
theorem vanessas_mother_picked_14_carrots : (G + B) - V = 14 := by
  sorry

end vanessas_mother_picked_14_carrots_l199_199166


namespace tangent_line_eq_max_f_val_in_interval_a_le_2_l199_199234

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) : ℝ := x ^ 3 - a * x ^ 2

def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 2 - 2 * a * x

-- (I) (i) Proof that the tangent line equation is y = 3x - 2 at (1, f(1))
theorem tangent_line_eq (a : ℝ) (h : f_prime 1 a = 3) : y = 3 * x - 2 :=
by sorry

-- (I) (ii) Proof that the max value of f(x) in [0,2] is 8
theorem max_f_val_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x 0 ≤ f 2 0 :=
by sorry

-- (II) Proof that a ≤ 2 if f(x) + x ≥ 0 for all x ∈ [0,2]
theorem a_le_2 (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a + x ≥ 0) : a ≤ 2 :=
by sorry

end tangent_line_eq_max_f_val_in_interval_a_le_2_l199_199234


namespace binomial_square_expression_l199_199327

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l199_199327


namespace expenditure_of_neg_50_l199_199617

/-- In the book "Nine Chapters on the Mathematical Art," it is noted that
"when two calculations have opposite meanings, they should be named positive
and negative." This means: if an income of $80 is denoted as $+80, then $-50
represents an expenditure of $50. -/
theorem expenditure_of_neg_50 :
  (∀ (income : ℤ), income = 80 → -income = -50 → ∃ (expenditure : ℤ), expenditure = 50) := sorry

end expenditure_of_neg_50_l199_199617


namespace value_of_q_l199_199743

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end value_of_q_l199_199743


namespace anya_can_obtain_any_composite_number_l199_199646

theorem anya_can_obtain_any_composite_number (n : ℕ) (h : ∃ k, k > 1 ∧ k < n ∧ n % k = 0) : ∃ m ≥ 4, ∀ k, k > 1 → k < m → m % k = 0 → m = n :=
by
  sorry

end anya_can_obtain_any_composite_number_l199_199646


namespace complement_intersection_l199_199733

def P : Set ℝ := {y | ∃ x, y = (1 / 2) ^ x ∧ 0 < x}
def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_intersection :
  (Set.univ \ P) ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
sorry

end complement_intersection_l199_199733


namespace geometric_sequence_sum_l199_199918

variable {α : Type*} [NormedField α] [CompleteSpace α]

def geometric_sum (a r : α) (n : ℕ) : α :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum (S : ℕ → α) (a r : α) (hS : ∀ n, S n = geometric_sum a r n) :
  S 2 = 6 → S 4 = 30 → S 6 = 126 :=
by
  sorry

end geometric_sequence_sum_l199_199918


namespace luke_fish_fillets_l199_199107

theorem luke_fish_fillets (daily_fish : ℕ) (days : ℕ) (fillets_per_fish : ℕ) 
  (h1 : daily_fish = 2) (h2 : days = 30) (h3 : fillets_per_fish = 2) : 
  daily_fish * days * fillets_per_fish = 120 := 
by 
  sorry

end luke_fish_fillets_l199_199107


namespace cricket_runs_l199_199668

theorem cricket_runs (A B C : ℕ) (h1 : A / B = 1 / 3) (h2 : B / C = 1 / 5) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Skipping proof details
  sorry

end cricket_runs_l199_199668


namespace count_special_numbers_l199_199753

-- Definitions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_less_than_600 (n : ℕ) : Prop := n < 600
def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

-- Theorem to prove
theorem count_special_numbers : 
  ∃! (cnt : ℕ), cnt = 140 ∧ 
  (∀ n, is_three_digit_integer n → is_less_than_600 n → has_at_least_two_identical_digits n) :=
sorry

end count_special_numbers_l199_199753


namespace volume_of_sphere_in_cone_l199_199680

theorem volume_of_sphere_in_cone :
  let diameter_of_base := 16 * Real.sqrt 2
  let radius_of_base := diameter_of_base / 2
  let side_length := radius_of_base * 2 / Real.sqrt 2
  let inradius := side_length / 2
  let r := inradius
  let V := (4 / 3) * Real.pi * r^3
  V = (2048 / 3) * Real.pi := by
  sorry

end volume_of_sphere_in_cone_l199_199680


namespace circle_tangent_to_xaxis_at_origin_l199_199251

theorem circle_tangent_to_xaxis_at_origin (G E F : ℝ)
  (h : ∀ x y: ℝ, x^2 + y^2 + G*x + E*y + F = 0 → y = 0 ∧ x = 0 ∧ 0 < E) :
  G = 0 ∧ F = 0 ∧ E ≠ 0 :=
by
  sorry

end circle_tangent_to_xaxis_at_origin_l199_199251


namespace tangent_circles_m_values_l199_199009

noncomputable def is_tangent (m : ℝ) : Prop :=
  let o1_center := (m, 0)
  let o2_center := (-1, 2 * m)
  let distance := Real.sqrt ((m + 1)^2 + (2 * m)^2)
  (distance = 5 ∨ distance = 1)

theorem tangent_circles_m_values :
  {m : ℝ | is_tangent m} = {-12 / 5, -2 / 5, 0, 2} := by
  sorry

end tangent_circles_m_values_l199_199009


namespace cows_to_eat_grass_in_96_days_l199_199496

theorem cows_to_eat_grass_in_96_days (G r : ℕ) : 
  (∀ N : ℕ, (70 * 24 = G + 24 * r) → (30 * 60 = G + 60 * r) → 
  (∃ N : ℕ, 96 * N = G + 96 * r) → N = 20) :=
by
  intro N
  intro h1 h2 h3
  sorry

end cows_to_eat_grass_in_96_days_l199_199496


namespace winning_probability_is_approx_0_103_l199_199106

/-- Definition of the total number of ways to choose 5 balls out of 10 -/
def total_outcomes : ℕ := nat.choose 10 5

/-- Number of ways to draw 4 red balls and 1 white ball -/
def draw_4_red_1_white : ℕ := nat.choose 5 4 * nat.choose 5 1

/-- Number of ways to draw all 5 red balls -/
def draw_5_red : ℕ := nat.choose 5 5

/-- Total number of winning outcomes -/
def winning_outcomes : ℕ := draw_4_red_1_white + draw_5_red

/-- Definition of the probability of winning as a rational number -/
def winning_probability : ℚ := winning_outcomes / total_outcomes

/-- Theorem: The probability of winning is approximately 0.103, rounded to three decimal places -/
theorem winning_probability_is_approx_0_103 : winning_probability ≈ 0.103 := by
  sorry

end winning_probability_is_approx_0_103_l199_199106


namespace find_f2_of_conditions_l199_199727

theorem find_f2_of_conditions (f g : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) 
                              (h_g : ∀ x, g x = f x + 9) 
                              (h_g_val : g (-2) = 3) : 
                              f 2 = 6 :=
by 
  sorry

end find_f2_of_conditions_l199_199727


namespace probability_age_less_than_20_l199_199793

theorem probability_age_less_than_20 (total_people : ℕ) (people_more_than_30 : ℕ) 
  (h1 : total_people = 130) (h2 : people_more_than_30 = 90) : 
  (130 - 90) / 130 = 4 / 13 := 
by  
  have people_less_than_20 : ℕ := total_people - people_more_than_30
  rw [h1, h2]
  have h3 : people_less_than_20 = 40 := by simp [people_less_than_20, h1, h2]
  have h4 : (40:ℚ) / 130 = (4:ℚ) / 13 := by norm_num
  exact h4

end probability_age_less_than_20_l199_199793


namespace binomial_square_expression_l199_199330

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l199_199330


namespace total_length_segments_in_figure2_l199_199686

-- Define the original dimensions of the figure
def vertical_side : ℕ := 10
def bottom_horizontal_side : ℕ := 3
def middle_horizontal_side : ℕ := 4
def topmost_horizontal_side : ℕ := 2

-- Define the lengths that are removed to form Figure 2
def removed_sides_length : ℕ :=
  bottom_horizontal_side + topmost_horizontal_side + vertical_side

-- Define the remaining lengths in Figure 2
def remaining_vertical_side : ℕ := vertical_side
def remaining_horizontal_side : ℕ := middle_horizontal_side

-- Total length of segments in Figure 2
def total_length_figure2 : ℕ :=
  remaining_vertical_side + remaining_horizontal_side

-- Conjecture that this total length is 14 units
theorem total_length_segments_in_figure2 : total_length_figure2 = 14 := by
  -- Proof goes here
  sorry

end total_length_segments_in_figure2_l199_199686


namespace counterexample_to_prime_condition_l199_199208

theorem counterexample_to_prime_condition :
  ¬(Prime 54) ∧ ¬(Prime 52) ∧ ¬(Prime 51) := by
  -- Proof not required
  sorry

end counterexample_to_prime_condition_l199_199208


namespace relay_team_order_count_l199_199875

theorem relay_team_order_count :
  ∃ (orders : ℕ), orders = 6 :=
by
  let team_members := 4
  let remaining_members := team_members - 1  -- Excluding Lisa
  let first_lap_choices := remaining_members.choose 3  -- Choices for the first lap
  let third_lap_choices := (remaining_members - 1).choose 2  -- Choices for the third lap
  let fourth_lap_choices := (remaining_members - 2).choose 1  -- The last remaining member choices
  have orders := first_lap_choices * third_lap_choices * fourth_lap_choices
  use orders
  sorry

end relay_team_order_count_l199_199875


namespace total_uniform_cost_l199_199716

theorem total_uniform_cost :
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  total_cost = 355 :=
by 
  let pants_cost := 20
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let socks_cost := 3
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
  let total_cost := 5 * uniform_cost
  sorry

end total_uniform_cost_l199_199716


namespace beehive_bee_count_l199_199256

theorem beehive_bee_count {a : ℕ → ℕ} (h₀ : a 0 = 1)
  (h₁ : a 1 = 6)
  (hn : ∀ n, a (n + 1) = a n + 5 * a n) :
  a 6 = 46656 :=
  sorry

end beehive_bee_count_l199_199256


namespace system_has_infinitely_many_solutions_l199_199603

theorem system_has_infinitely_many_solutions :
  ∃ (S : Set (ℝ × ℝ × ℝ)), (∀ x y z : ℝ, (x + y = 2 ∧ xy - z^2 = 1) ↔ (x, y, z) ∈ S) ∧ S.Infinite :=
by
  sorry

end system_has_infinitely_many_solutions_l199_199603


namespace cone_base_diameter_l199_199087

theorem cone_base_diameter
  (h_cone : ℝ) (r_sphere : ℝ) (waste_percentage : ℝ) (d : ℝ) :
  h_cone = 9 → r_sphere = 9 → waste_percentage = 0.75 → 
  (V_cone = 1/3 * π * (d/2)^2 * h_cone) →
  (V_sphere = 4/3 * π * r_sphere^3) →
  (V_cone = (1 - waste_percentage) * V_sphere) →
  d = 9 :=
by
  intros h_cond r_cond waste_cond v_cone_eq v_sphere_eq v_cone_sphere_eq
  sorry

end cone_base_diameter_l199_199087


namespace bracelet_price_l199_199192

theorem bracelet_price 
  (B : ℝ) -- price of each bracelet
  (H1 : B > 0) 
  (H2 : 3 * B + 2 * 10 + 20 = 100 - 15) : 
  B = 15 :=
by
  sorry

end bracelet_price_l199_199192


namespace sum_of_valid_x_sum_of_all_valid_x_l199_199322

open Real

-- Definitions of median and mean for a list of five elements
def median_of_five (a b c d e : ℝ) : ℝ :=
  let lst := List.sort [a, b, c, d, e]
  lst.nthLe 2 sorry -- Since we know the list has exactly 5 elements

def mean_of_five (a b c d e : ℝ) : ℝ :=
  (a + b + c + d + e) / 5

theorem sum_of_valid_x :
  ∀ x : ℝ, (median_of_five 4 6 8 17 x = mean_of_five 4 6 8 17 x) → x = -5 :=
begin
  assume x h,
  sorry -- prove that x can only be -5 under the given conditions
end

theorem sum_of_all_valid_x :
  ∑ x in {x : ℝ | median_of_five 4 6 8 17 x = mean_of_five 4 6 8 17 x}.toFinset = -5 :=
begin
  sorry -- prove that the sum of all valid x is -5
end

end sum_of_valid_x_sum_of_all_valid_x_l199_199322


namespace distinct_non_zero_real_numbers_l199_199592

theorem distinct_non_zero_real_numbers (
  a b c : ℝ
) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2 * b * x1 + c = 0 ∧ ax^2 + 2 * b * x2 + c = 0) 
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ bx^2 + 2 * c * x1 + a = 0 ∧ bx^2 + 2 * c * x2 + a = 0)
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ cx^2 + 2 * a * x1 + b = 0 ∧ cx^2 + 2 * a * x2 + b = 0) :=
sorry

end distinct_non_zero_real_numbers_l199_199592


namespace ratio_of_numbers_l199_199654

theorem ratio_of_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hsum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_numbers_l199_199654


namespace integral_of_exp_plus_linear_l199_199820

theorem integral_of_exp_plus_linear :
  ∫ x in 0..1, (Real.exp x + 2 * x) = Real.exp 1 :=
by
  sorry

end integral_of_exp_plus_linear_l199_199820


namespace Hillary_sunday_minutes_l199_199244

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end Hillary_sunday_minutes_l199_199244


namespace range_of_a_if_exists_x_l199_199940

variable {a x : ℝ}

theorem range_of_a_if_exists_x :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ (a * x^2 - 1 ≥ 0)) → (a > 1) :=
by
  sorry

end range_of_a_if_exists_x_l199_199940


namespace find_values_of_a2_b2_l199_199591

-- Define the conditions
variables {a b : ℝ}
variable (h1 : a > b)
variable (h2 : b > 0)
variable (hP : (-2, (Real.sqrt 14) / 2) ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 })
variable (hCircle : ∀ Q : ℝ × ℝ, (Q ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 2 }) → (∃ tA tB : ℝ × ℝ, (tA ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tB ∈ { p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1 }) ∧ (tA = - tB ∨ tB = - tA) ∧ ((tA.1 + tB.1)/2 = (-2 + tA.1)/2) ))

-- The theorem to be proven
theorem find_values_of_a2_b2 : a^2 + b^2 = 15 :=
sorry

end find_values_of_a2_b2_l199_199591


namespace min_value_function_l199_199916

open Real

theorem min_value_function (x y : ℝ) 
  (hx : x > -2 ∧ x < 2) 
  (hy : y > -2 ∧ y < 2) 
  (hxy : x * y = -1) : 
  (∃ u : ℝ, u = (4 / (4 - x^2) + 9 / (9 - y^2)) ∧ u = 12 / 5) :=
sorry

end min_value_function_l199_199916


namespace f_constant_1_l199_199628

theorem f_constant_1 (f : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → f (n + f n) = f n)
  (h2 : ∃ n0 : ℕ, 0 < n0 ∧ f n0 = 1) : ∀ n : ℕ, f n = 1 := 
by
  sorry

end f_constant_1_l199_199628


namespace numberOfValidFiveDigitNumbers_l199_199905

namespace MathProof

def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def isDivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def firstAndLastDigitsEqual (n : ℕ) : Prop := 
  let firstDigit := (n / 10000) % 10
  let lastDigit := n % 10
  firstDigit = lastDigit

def sumOfDigitsDivisibleBy5 (n : ℕ) : Prop := 
  let d1 := (n / 10000) % 10
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  (d1 + d2 + d3 + d4 + d5) % 5 = 0

theorem numberOfValidFiveDigitNumbers :
  ∃ (count : ℕ), count = 200 ∧ 
  count = Nat.card {n : ℕ // isFiveDigitNumber n ∧ 
                                isDivisibleBy5 n ∧ 
                                firstAndLastDigitsEqual n ∧ 
                                sumOfDigitsDivisibleBy5 n} :=
by
  sorry

end MathProof

end numberOfValidFiveDigitNumbers_l199_199905


namespace dragons_total_games_played_l199_199689

theorem dragons_total_games_played (y x : ℕ)
  (h1 : x = 55 * y / 100)
  (h2 : x + 8 = 60 * (y + 12) / 100) :
  y + 12 = 28 :=
by
  sorry

end dragons_total_games_played_l199_199689


namespace mary_total_payment_l199_199152

def fixed_fee : ℕ := 17
def hourly_charge : ℕ := 7
def rental_duration : ℕ := 9
def total_payment (f : ℕ) (h : ℕ) (r : ℕ) : ℕ := f + (h * r)

theorem mary_total_payment:
  total_payment fixed_fee hourly_charge rental_duration = 80 :=
by
  sorry

end mary_total_payment_l199_199152


namespace evaluate_expression_l199_199587

theorem evaluate_expression (x y : ℝ) (h1 : x * y = -2) (h2 : x + y = 4) : x^2 * y + x * y^2 = -8 :=
by
  sorry

end evaluate_expression_l199_199587


namespace sum_of_three_eq_six_l199_199036

theorem sum_of_three_eq_six
  (a b c : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 150) :
  a + b + c = 6 :=
sorry

end sum_of_three_eq_six_l199_199036


namespace sum_ab_l199_199387

theorem sum_ab (a b : ℕ) (h1 : 1 < b) (h2 : a ^ b < 500) (h3 : ∀ x y : ℕ, (1 < y ∧ x ^ y < 500 ∧ (x + y) % 2 = 0) → a ^ b ≥ x ^ y) (h4 : (a + b) % 2 = 0) : a + b = 24 :=
  sorry

end sum_ab_l199_199387


namespace average_of_remaining_five_l199_199351

open Nat Real

theorem average_of_remaining_five (avg9 avg4 : ℝ) (S S4 : ℝ) 
(h1 : avg9 = 18) (h2 : avg4 = 8) 
(h_sum9 : S = avg9 * 9) 
(h_sum4 : S4 = avg4 * 4) :
(S - S4) / 5 = 26 := by
  sorry

end average_of_remaining_five_l199_199351


namespace translate_point_correct_l199_199264

def P : ℝ × ℝ := (2, 3)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

theorem translate_point_correct :
  translate_down (translate_left P 3) 4 = (-1, -1) :=
by
  sorry

end translate_point_correct_l199_199264


namespace tangent_line_property_l199_199656

variables (a b c : ℝ) (AC1 AB1 : ℝ)
noncomputable def p := (a + b + c) / 2

theorem tangent_line_property (x y : ℝ) (hAC1 : AC1 = x) (hAB1 : AB1 = y) :
  p a b c * x * y - b * c * (x + y) + b * c * (p a b c - a) = 0 :=
by sorry

end tangent_line_property_l199_199656


namespace arithmetic_sequence_evaluation_l199_199174

theorem arithmetic_sequence_evaluation :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by 
-- Proof omitted
sorry

end arithmetic_sequence_evaluation_l199_199174


namespace francie_remaining_money_l199_199912

-- Define the initial weekly allowance for the first period
def initial_weekly_allowance : ℕ := 5
-- Length of the first period in weeks
def first_period_weeks : ℕ := 8
-- Define the raised weekly allowance for the second period
def raised_weekly_allowance : ℕ := 6
-- Length of the second period in weeks
def second_period_weeks : ℕ := 6
-- Cost of the video game
def video_game_cost : ℕ := 35

-- Define the total savings before buying new clothes
def total_savings :=
  first_period_weeks * initial_weekly_allowance + second_period_weeks * raised_weekly_allowance

-- Define the money spent on new clothes
def money_spent_on_clothes :=
  total_savings / 2

-- Define the remaining money after buying the video game
def remaining_money :=
  money_spent_on_clothes - video_game_cost

-- Prove that Francie has $3 remaining after buying the video game
theorem francie_remaining_money : remaining_money = 3 := by
  sorry

end francie_remaining_money_l199_199912


namespace four_digit_num_exists_l199_199214

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem four_digit_num_exists :
  ∃ (n : ℕ), (is_two_digit (n / 100)) ∧ (is_two_digit (n % 100)) ∧
  ((n / 100) + (n % 100))^2 = 100 * (n / 100) + (n % 100) :=
by
  sorry

end four_digit_num_exists_l199_199214


namespace oranges_left_l199_199806

-- Main theorem statement: number of oranges left after specified increases and losses
theorem oranges_left (Mary Jason Tom Sarah : ℕ)
  (hMary : Mary = 122)
  (hJason : Jason = 105)
  (hTom : Tom = 85)
  (hSarah : Sarah = 134) 
  (round : ℝ → ℕ) 
  : round (round ( (Mary : ℝ) * 1.1) 
         + round ((Jason : ℝ) * 1.1) 
         + round ((Tom : ℝ) * 1.1) 
         + round ((Sarah : ℝ) * 1.1) 
         - round (0.15 * (round ((Mary : ℝ) * 1.1) 
                         + round ((Jason : ℝ) * 1.1)
                         + round ((Tom : ℝ) * 1.1) 
                         + round ((Sarah : ℝ) * 1.1)) )) = 417  := 
sorry

end oranges_left_l199_199806


namespace socks_picking_l199_199450

theorem socks_picking : 
  let socks := [ ("white", 5), ("brown", 3), ("blue", 2), ("red", 2) ] in
  ∑ (color in socks), nat.choose color.2 2 = 15 :=
by
  sorry

end socks_picking_l199_199450


namespace max_x_inequality_k_l199_199612

theorem max_x_inequality_k (k : ℝ) (h : ∀ x : ℝ, |x^2 - 4 * x + k| + |x - 3| ≤ 5 → x ≤ 3) : k = 8 :=
sorry

end max_x_inequality_k_l199_199612


namespace problem_l199_199121

variable {a b c d : ℝ}

theorem problem (h1 : a > b) (h2 : c > d) : a - d > b - c := sorry

end problem_l199_199121


namespace bobby_final_paycheck_correct_l199_199694

def bobby_salary : ℕ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 0.08
def health_insurance_deduction : ℕ := 50
def life_insurance_deduction : ℕ := 20
def city_parking_fee : ℕ := 10

def final_paycheck_amount : ℚ :=
  let federal_taxes := federal_tax_rate * bobby_salary
  let state_taxes := state_tax_rate * bobby_salary
  let total_deductions := federal_taxes + state_taxes + health_insurance_deduction + life_insurance_deduction + city_parking_fee
  bobby_salary - total_deductions

theorem bobby_final_paycheck_correct : final_paycheck_amount = 184 := by
  sorry

end bobby_final_paycheck_correct_l199_199694


namespace perpendicular_lines_l199_199436

theorem perpendicular_lines (a : ℝ) :
  (a + 2) * (a - 1) + (1 - a) * (2 * a + 3) = 0 ↔ (a = 1 ∨ a = -1) := 
sorry

end perpendicular_lines_l199_199436


namespace solve_keychain_problem_l199_199983

def keychain_problem : Prop :=
  let f_class := 6
  let f_club := f_class / 2
  let thread_total := 108
  let total_friends := f_class + f_club
  let threads_per_keychain := thread_total / total_friends
  threads_per_keychain = 12

theorem solve_keychain_problem : keychain_problem :=
  by sorry

end solve_keychain_problem_l199_199983


namespace production_rate_l199_199353

theorem production_rate (minutes: ℕ) (machines1 machines2 paperclips1 paperclips2 : ℕ)
  (h1 : minutes = 1) (h2 : machines1 = 8) (h3 : machines2 = 18) (h4 : paperclips1 = 560) 
  (h5 : paperclips2 = (paperclips1 / machines1) * machines2 * minutes) : 
  paperclips2 = 7560 :=
by
  sorry

end production_rate_l199_199353


namespace jordan_rectangle_width_l199_199520

theorem jordan_rectangle_width :
  ∀ (areaC areaJ : ℕ) (lengthC widthC lengthJ widthJ : ℕ), 
    (areaC = lengthC * widthC) →
    (areaJ = lengthJ * widthJ) →
    (areaC = areaJ) →
    (lengthC = 5) →
    (widthC = 24) →
    (lengthJ = 3) →
    widthJ = 40 :=
by
  intros areaC areaJ lengthC widthC lengthJ widthJ
  intro hAreaC
  intro hAreaJ
  intro hEqualArea
  intro hLengthC
  intro hWidthC
  intro hLengthJ
  sorry

end jordan_rectangle_width_l199_199520


namespace problem1_solution_problem2_solution_l199_199291

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : y = 2 * x - 3) (h2 : 2 * x + y = 5) : 
  x = 2 ∧ y = 1 :=
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 3 * x + 4 * y = 5) (h2 : 5 * x - 2 * y = 17) : 
  x = 3 ∧ y = -1 :=
  sorry

end problem1_solution_problem2_solution_l199_199291


namespace true_propositions_l199_199227

def p : Prop :=
  ∀ a b : ℝ, (a > 2 ∧ b > 2) → a + b > 4

def q : Prop :=
  ¬ ∃ x : ℝ, x^2 - x > 0 → ∀ x : ℝ, x^2 - x ≤ 0

theorem true_propositions :
  (¬ p ∨ ¬ q) ∧ (p ∨ ¬ q) := by
  sorry

end true_propositions_l199_199227


namespace cos_value_of_inclined_line_l199_199929

variable (α : ℝ)
variable (l : ℝ) -- representing line as real (though we handle angles here)
variable (h_tan_line : ∃ α, tan α * (-1/2) = -1)

theorem cos_value_of_inclined_line (h_perpendicular : h_tan_line) :
  cos (2015 * Real.pi / 2 + 2 * α) = 4 / 5 := 
sorry

end cos_value_of_inclined_line_l199_199929


namespace smallest_positive_integer_in_form_l199_199172

theorem smallest_positive_integer_in_form :
  ∃ (m n p : ℤ), 1234 * m + 56789 * n + 345 * p = 1 := sorry

end smallest_positive_integer_in_form_l199_199172


namespace inequality_conditions_l199_199970

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + B * C + C * A)) :=
by
  sorry

end inequality_conditions_l199_199970


namespace participants_initial_count_l199_199792

theorem participants_initial_count 
  (x : ℕ) 
  (p1 : x * (2 : ℚ) / 5 * 1 / 4 = 30) :
  x = 300 :=
by
  sorry

end participants_initial_count_l199_199792


namespace tenth_square_tiles_more_than_ninth_l199_199448

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := 2 * n - 1

-- Calculate the number of tiles used in the nth square
def tiles_count (n : ℕ) : ℕ := (side_length n) ^ 2

-- State the theorem that the tenth square requires 72 more tiles than the ninth square
theorem tenth_square_tiles_more_than_ninth : tiles_count 10 - tiles_count 9 = 72 :=
by
  sorry

end tenth_square_tiles_more_than_ninth_l199_199448


namespace q_value_l199_199240

theorem q_value (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end q_value_l199_199240


namespace no_roots_in_interval_l199_199272

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l199_199272


namespace small_boxes_in_large_box_l199_199528

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end small_boxes_in_large_box_l199_199528


namespace no_positive_integers_abc_l199_199399

theorem no_positive_integers_abc :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (a * b + b * c = a * c) ∧ (a * b * c = Nat.factorial 10) :=
sorry

end no_positive_integers_abc_l199_199399


namespace find_greatest_integer_l199_199812

variables (b h x : ℝ)

-- Defining the conditions of the problem
def trapezoid_equation_1 (b : ℝ) : Prop :=
  (b + 75) / (b + 150) = 3 / 4

def trapezoid_equation_2 (b x : ℝ) : Prop :=
  x = 250

-- The main theorem to be proven
theorem find_greatest_integer (b : ℝ) (h : ℝ) (x : ℝ) (h1 : x = 250) : 
  ⌊x^2 / 150⌋ = 416 :=
by
  have b_eq : 4 * (b + 75) = 3 * (b + 150), from sorry,
  have b_val : b = 150, from sorry,
  have x_val : x = 250, from sorry,
  calc
    ⌊(250)^2 / 150⌋ = ⌊62500 / 150⌋ : by rw x_val
    ... = 416 : sorry


end find_greatest_integer_l199_199812


namespace total_weight_of_settings_l199_199475

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l199_199475


namespace find_r_power_4_l199_199434

variable {r : ℝ}

theorem find_r_power_4 (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := 
sorry

end find_r_power_4_l199_199434


namespace carwash_num_cars_l199_199966

variable (C : ℕ)

theorem carwash_num_cars 
    (h1 : 5 * 7 + 5 * 6 + C * 5 = 100)
    : C = 7 := 
by
    sorry

end carwash_num_cars_l199_199966


namespace circumcircle_radius_l199_199794

open Real

theorem circumcircle_radius (a b c A B C S R : ℝ) 
  (h1 : S = (1/2) * sin A * sin B * sin C)
  (h2 : S = (1/2) * a * b * sin C)
  (h3 : ∀ x y, x = y → x * cos 0 = y * cos 0):
  R = (1/2) :=
by
  sorry

end circumcircle_radius_l199_199794


namespace real_solution_l199_199403

noncomputable def condition_1 (x : ℝ) : Prop := 
  4 ≤ x / (2 * x - 7)

noncomputable def condition_2 (x : ℝ) : Prop := 
  x / (2 * x - 7) < 10

noncomputable def solution_set : Set ℝ :=
  { x | (70 / 19 : ℝ) < x ∧ x ≤ 4 }

theorem real_solution (x : ℝ) : 
  (condition_1 x ∧ condition_2 x) ↔ x ∈ solution_set :=
sorry

end real_solution_l199_199403


namespace evaluate_expression_l199_199335

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l199_199335


namespace total_students_l199_199257

theorem total_students (p q r s : ℕ) 
  (h1 : 1 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h5 : p * q * r * s = 1365) :
  p + q + r + s = 28 :=
sorry

end total_students_l199_199257


namespace probability_greg_rolls_more_ones_than_sixes_l199_199773

def number_of_outcomes : ℕ := 6^5

def count_combinations_zero_one_six : ℕ := 
  ((choose 5 0) * (4^5))

def count_combinations_one_one_six : ℕ := 
  ((choose 5 1) * (choose 4 1) * (4^3))

def count_combinations_two_one_six : ℕ :=
  ((choose 5 2) * (choose 3 2) * 4)

def total_combinations_equal_one_six : ℕ :=
  count_combinations_zero_one_six + count_combinations_one_one_six + count_combinations_two_one_six

def probability_equal_one_six : ℚ :=
  total_combinations_equal_one_six / number_of_outcomes

def probability_more_ones_than_sixes : ℚ :=
  1 / 2 * (1 - probability_equal_one_six)

theorem probability_greg_rolls_more_ones_than_sixes :
  probability_more_ones_than_sixes = (167 : ℚ) / 486 := by
  sorry

end probability_greg_rolls_more_ones_than_sixes_l199_199773


namespace ratio_of_altitude_to_radius_l199_199198

theorem ratio_of_altitude_to_radius (r R h : ℝ)
  (hR : R = 2 * r)
  (hV : (1/3) * π * R^2 * h = (1/3) * (4/3) * π * r^3) :
  h / R = 1 / 6 := by
  sorry

end ratio_of_altitude_to_radius_l199_199198


namespace total_swordfish_caught_correct_l199_199030

-- Define the parameters and the catch per trip
def shelly_catch_per_trip : ℕ := 5 - 2
def sam_catch_per_trip (shelly_catch : ℕ) : ℕ := shelly_catch - 1
def total_catch_per_trip (shelly_catch sam_catch : ℕ) : ℕ := shelly_catch + sam_catch
def total_trips : ℕ := 5

-- Define the total number of swordfish caught over the trips
def total_swordfish_caught : ℕ :=
  let shelly_catch := shelly_catch_per_trip
  let sam_catch := sam_catch_per_trip shelly_catch
  let total_catch := total_catch_per_trip shelly_catch sam_catch
  total_catch * total_trips

-- The theorem states that the total swordfish caught is 25
theorem total_swordfish_caught_correct : total_swordfish_caught = 25 := by
  sorry

end total_swordfish_caught_correct_l199_199030


namespace probability_XOXOXOX_l199_199572

theorem probability_XOXOXOX (X O : ℕ) (h1 : X = 4) (h2 : O = 3) :
  let total_ways := Nat.choose (X + O) X,
      favorable_outcomes := 1 in
  (favorable_outcomes / total_ways : ℚ) = 1 / 35 := by
  sorry

end probability_XOXOXOX_l199_199572


namespace sum_largest_smallest_prime_factors_1155_l199_199845

theorem sum_largest_smallest_prime_factors_1155 : 
  ∃ smallest largest : ℕ, 
  smallest ∣ 1155 ∧ largest ∣ 1155 ∧ 
  Prime smallest ∧ Prime largest ∧ 
  smallest <= largest ∧ 
  (∀ p : ℕ, p ∣ 1155 → Prime p → (smallest ≤ p ∧ p ≤ largest)) ∧ 
  (smallest + largest = 14) := 
by {
  sorry
}

end sum_largest_smallest_prime_factors_1155_l199_199845


namespace range_of_derivative_l199_199827

theorem range_of_derivative :
  ∀ x : ℝ, -√3 ≤ 3 * x ^ 2 - √3 :=
by 
  intro x
  have h := (3 * x ^ 2 : ℝ)
  linarith

end range_of_derivative_l199_199827


namespace sqrt_eq_sum_seven_l199_199293

open Real

theorem sqrt_eq_sum_seven (x : ℝ) (h : sqrt (64 - x^2) - sqrt (36 - x^2) = 4) :
    sqrt (64 - x^2) + sqrt (36 - x^2) = 7 :=
by
  sorry

end sqrt_eq_sum_seven_l199_199293


namespace necessary_and_sufficient_condition_l199_199279

def U (a : ℕ) : Set ℕ := { x | x > 0 ∧ x ≤ a }
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}
def C_U (S : Set ℕ) (a : ℕ) : Set ℕ := U a ∩ Sᶜ

theorem necessary_and_sufficient_condition (a : ℕ) (h : 6 ≤ a ∧ a < 7) : 
  C_U P a = Q ↔ (6 ≤ a ∧ a < 7) :=
by
  sorry

end necessary_and_sufficient_condition_l199_199279


namespace brick_length_l199_199713

theorem brick_length (w h SA : ℝ) (h_w : w = 6) (h_h : h = 2) (h_SA : SA = 152) :
  ∃ l : ℝ, 2 * l * w + 2 * l * h + 2 * w * h = SA ∧ l = 8 := 
sorry

end brick_length_l199_199713


namespace prob_more_1s_than_6s_l199_199760

noncomputable def probability_more_ones_than_sixes (n : ℕ) : ℚ :=
  let total_outcomes := 6^n
  let equal_1s_6s :=  sum_finsupp (λ k1 k6 : _n_, if (k1 = k6) 
    then binom n k1 * binom (n - k1) k6 * (4 ^ (n - k1 - k6)) else 0)
  let prob_equal := equal_1s_6s / total_outcomes
  let final_probability := (1 - prob_equal) / 2
  final_probability

theorem prob_more_1s_than_6s :
  probability_more_ones_than_sixes 5 = 2676 / 7776 :=
sorry

end prob_more_1s_than_6s_l199_199760


namespace lily_final_balance_l199_199453

noncomputable def initial_balance : ℝ := 55
noncomputable def shirt_cost : ℝ := 7
noncomputable def shoes_cost : ℝ := 3 * shirt_cost
noncomputable def book_cost : ℝ := 4
noncomputable def books_amount : ℝ := 5
noncomputable def gift_fraction : ℝ := 0.20

noncomputable def remaining_balance : ℝ :=
  initial_balance - 
  shirt_cost - 
  shoes_cost - 
  books_amount * book_cost - 
  gift_fraction * (initial_balance - shirt_cost - shoes_cost - books_amount * book_cost)

theorem lily_final_balance : remaining_balance = 5.60 := 
by 
  sorry

end lily_final_balance_l199_199453


namespace antacids_per_month_proof_l199_199479

structure ConsumptionRates where
  indian : Nat
  mexican : Nat
  other : Nat

structure WeeklyEatingPattern where
  indian_days : Nat
  mexican_days : Nat
  other_days : Nat

def antacids_per_week (rates : ConsumptionRates) (pattern : WeeklyEatingPattern) : Nat :=
  (pattern.indian_days * rates.indian) +
  (pattern.mexican_days * rates.mexican) +
  (pattern.other_days * rates.other)

def weeks_in_month : Nat := 4

theorem antacids_per_month_proof :
  let rates := ConsumptionRates.mk 3 2 1 in
  let pattern := WeeklyEatingPattern.mk 3 2 2 in -- 7 days - 3 Indian days - 2 Mexican days = 2 other days
  let weekly_antacids := antacids_per_week rates pattern in
  weekly_antacids * weeks_in_month = 60 := 
by
  -- let's skip the proof
  sorry

end antacids_per_month_proof_l199_199479


namespace determine_a_value_l199_199930

theorem determine_a_value (a : ℝ) :
  (∀ y₁ y₂ : ℝ, ∃ m₁ m₂ : ℝ, (m₁, y₁) = (a, -2) ∧ (m₂, y₂) = (3, -4) ∧ (m₁ = m₂)) → a = 3 :=
by
  sorry

end determine_a_value_l199_199930


namespace trig_identity_l199_199695

theorem trig_identity :
  (4 * (1 / 2) - Real.sqrt 2 * (Real.sqrt 2 / 2) - Real.sqrt 3 * (Real.sqrt 3 / 3) + 2 * (Real.sqrt 3 / 2)) = Real.sqrt 3 :=
by sorry

end trig_identity_l199_199695


namespace probability_more_ones_than_sixes_l199_199777

theorem probability_more_ones_than_sixes :
  let num_faces := 6 in
  let num_rolls := 5 in
  let total_outcomes := num_faces ^ num_rolls in
  let favorable_outcomes := 2711 in
  (favorable_outcomes : ℚ) / total_outcomes = 2711 / 7776 :=
sorry

end probability_more_ones_than_sixes_l199_199777


namespace problem_statement_l199_199175

def fair_coin_twice := [("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")]

def event_A := [("H", "H"), ("H", "T")]
def event_B := [("H", "H"), ("T", "H")]

def probability (e : List (String × String)) := e.length.toFloat / fair_coin_twice.length.toFloat

theorem problem_statement :
  probability (fair_coin_twice.filter (not ∘ event_A.contains)) = 0.5 ∧
  probability ((event_A ++ event_B).eraseDups) = 0.75 ∧
  ¬ (event_A.filter (event_B.contains) = []) ∧
  probability (event_A.filter (event_B.contains)) = (probability event_A) * (probability event_B) :=
by
  sorry

end problem_statement_l199_199175


namespace greatest_integer_y_l199_199168

-- Define the fraction and inequality condition
def inequality_condition (y : ℤ) : Prop := 8 * 17 > 11 * y

-- Prove the greatest integer y satisfying the condition is 12
theorem greatest_integer_y : ∃ y : ℤ, inequality_condition y ∧ (∀ z : ℤ, inequality_condition z → z ≤ y) ∧ y = 12 :=
by
  exists 12
  sorry

end greatest_integer_y_l199_199168


namespace switches_assembled_are_correct_l199_199833

-- Definitions based on conditions
def total_payment : ℕ := 4700
def first_worker_payment : ℕ := 2000
def second_worker_per_switch_time_min : ℕ := 4
def third_worker_less_payment : ℕ := 300
def overtime_hours : ℕ := 5
def total_minutes (hours : ℕ) : ℕ := hours * 60

-- Function to calculate total switches assembled
noncomputable def total_switches_assembled :=
  let second_worker_payment := (total_payment - first_worker_payment + third_worker_less_payment) / 2
  let third_worker_payment := second_worker_payment - third_worker_less_payment
  let rate_per_switch := second_worker_payment / (total_minutes overtime_hours / second_worker_per_switch_time_min)
  let first_worker_switches := first_worker_payment / rate_per_switch
  let second_worker_switches := total_minutes overtime_hours / second_worker_per_switch_time_min
  let third_worker_switches := third_worker_payment / rate_per_switch
  first_worker_switches + second_worker_switches + third_worker_switches

-- Lean 4 statement to prove the problem
theorem switches_assembled_are_correct : 
  total_switches_assembled = 235 := by
  sorry

end switches_assembled_are_correct_l199_199833


namespace female_athletes_in_sample_l199_199685

theorem female_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ)
  (total_athletes_eq : total_athletes = 98)
  (male_athletes_eq : male_athletes = 56)
  (sample_size_eq : sample_size = 28)
  : (sample_size * (total_athletes - male_athletes) / total_athletes) = 12 :=
by
  sorry

end female_athletes_in_sample_l199_199685


namespace scientific_notation_32000000_l199_199380

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l199_199380


namespace hillary_reading_time_l199_199245

theorem hillary_reading_time :
  let total_minutes := 60
  let friday_minutes := 16
  let saturday_minutes := 28
  let already_read := friday_minutes + saturday_minutes
  in total_minutes - already_read = 16 := by 
  sorry

end hillary_reading_time_l199_199245


namespace part1_part2_l199_199822

-- Part 1: Determining the number of toys A and ornaments B wholesaled
theorem part1 (x y : ℕ) (h₁ : x + y = 100) (h₂ : 60 * x + 50 * y = 5650) : 
  x = 65 ∧ y = 35 := by
  sorry

-- Part 2: Determining the minimum number of toys A to wholesale for a 1400元 profit
theorem part2 (m : ℕ) (h₁ : m ≤ 100) (h₂ : (80 - 60) * m + (60 - 50) * (100 - m) ≥ 1400) : 
  m ≥ 40 := by
  sorry

end part1_part2_l199_199822


namespace point_on_angle_bisector_l199_199429

theorem point_on_angle_bisector (a b : ℝ) (h : (a, b) = (b, a)) : a = b ∨ a = -b := 
by
  sorry

end point_on_angle_bisector_l199_199429


namespace trig_evaluation_l199_199697

noncomputable def sin30 := 1 / 2
noncomputable def cos45 := Real.sqrt 2 / 2
noncomputable def tan30 := Real.sqrt 3 / 3
noncomputable def sin60 := Real.sqrt 3 / 2

theorem trig_evaluation : 4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end trig_evaluation_l199_199697


namespace harish_ganpat_paint_wall_together_l199_199243

theorem harish_ganpat_paint_wall_together :
  let r_h := 1 / 3 -- Harish's rate of work (walls per hour)
  let r_g := 1 / 6 -- Ganpat's rate of work (walls per hour)
  let combined_rate := r_h + r_g -- Combined rate of work when both work together
  let time_to_paint_one_wall := 1 / combined_rate -- Time to paint one wall together
  time_to_paint_one_wall = 2 :=
by
  sorry

end harish_ganpat_paint_wall_together_l199_199243


namespace horizontal_asymptote_at_3_l199_199103

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 2 * x^3 + 11 * x^2 + 6 * x + 4) / (5 * x^4 + x^3 + 10 * x^2 + 4 * x + 2)

theorem horizontal_asymptote_at_3 : 
  (∀ ε > 0, ∃ N > 0, ∀ x > N, |rational_function x - 3| < ε) := 
by
  sorry

end horizontal_asymptote_at_3_l199_199103


namespace mike_toys_l199_199882

theorem mike_toys (M A T : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : T = A + 2)
  (h3 : M + A + T = 56) 
  : M = 6 := 
by 
  sorry

end mike_toys_l199_199882


namespace correct_calculation_l199_199513

theorem correct_calculation : ∀ (a : ℝ), a^3 * a^2 = a^5 := 
by
  intro a
  sorry

end correct_calculation_l199_199513


namespace find_number_l199_199883

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l199_199883


namespace probability_correct_l199_199531

open Classical
noncomputable theory

def a_set : Set ℚ := {1/3, 1/2, 2, 3}
def b_set : Set ℚ := {-1, 1, -2, 2}

def passes_through_third_quadrant (a b : ℚ) : Prop :=
∃ x : ℚ, a ^ x + b < 0 ∧ x < 0 

def valid_cases : Finset (ℚ × ℚ) :=
{((1/3), -2), ((1/2), -2), (2, -1), (2, -2), (3, -1), (3, -2)}

def total_cases : ℕ := 4 * 4

def favorable_cases : ℕ := (valid_cases.card : ℕ)

def probability_third_quadrant : ℚ := favorable_cases / total_cases

theorem probability_correct :
  probability_third_quadrant = 3/8 :=
sorry

end probability_correct_l199_199531


namespace average_of_rest_of_class_l199_199821

def class_average (n : ℕ) (avg : ℕ) := n * avg
def sub_class_average (n : ℕ) (sub_avg : ℕ) := (n / 4) * sub_avg

theorem average_of_rest_of_class (n : ℕ) (h1 : class_average n 80 = 80 * n) (h2 : sub_class_average n 92 = (n / 4) * 92) :
  let A := 76
  A * (3 * n / 4) + (n / 4) * 92 = 80 * n := by
  sorry

end average_of_rest_of_class_l199_199821


namespace ratio_evaluation_l199_199402

theorem ratio_evaluation :
  (10 ^ 2003 + 10 ^ 2001) / (2 * 10 ^ 2002) = 101 / 20 := 
by sorry

end ratio_evaluation_l199_199402


namespace prob_not_all_same_correct_l199_199065

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end prob_not_all_same_correct_l199_199065


namespace each_parent_suitcases_l199_199149

namespace SuitcaseProblem

-- Definitions based on conditions
def siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def total_suitcases : Nat := 14

-- Theorem statement corresponding to the question and correct answer
theorem each_parent_suitcases (suitcases_per_parent : Nat) :
  (siblings * suitcases_per_sibling + 2 * suitcases_per_parent = total_suitcases) →
  suitcases_per_parent = 3 := by
  intro h
  sorry

end SuitcaseProblem

end each_parent_suitcases_l199_199149


namespace train_crossing_time_l199_199997

-- Definitions based on conditions from the problem
def length_of_train_and_platform := 900 -- in meters
def speed_km_per_hr := 108 -- in km/hr
def distance := 2 * length_of_train_and_platform -- distance to be covered
def speed_m_per_s := (speed_km_per_hr * 1000) / 3600 -- converted speed

-- Theorem stating the time to cross the platform is 60 seconds
theorem train_crossing_time : distance / speed_m_per_s = 60 := by
  sorry

end train_crossing_time_l199_199997


namespace average_brown_mms_per_bag_l199_199484

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end average_brown_mms_per_bag_l199_199484


namespace james_initial_amount_l199_199267

noncomputable def initial_amount (total_amount_invested_per_week: ℕ) 
                                (number_of_weeks_in_year: ℕ) 
                                (windfall_factor: ℚ) 
                                (amount_after_windfall: ℕ) : ℚ :=
  let total_investment := total_amount_invested_per_week * number_of_weeks_in_year
  let amount_without_windfall := (amount_after_windfall : ℚ) / (1 + windfall_factor)
  amount_without_windfall - total_investment

theorem james_initial_amount:
  initial_amount 2000 52 0.5 885000 = 250000 := sorry

end james_initial_amount_l199_199267


namespace no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l199_199209

theorem no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square :
  ∀ b : ℤ, ¬ ∃ k : ℤ, b^2 + 3*b + 1 = k^2 :=
by
  sorry

end no_integer_b_such_that_b_sq_plus_3b_plus_1_is_perfect_square_l199_199209


namespace fraction_of_work_left_l199_199676

variable (A_work_days : ℕ) (B_work_days : ℕ) (work_days_together: ℕ)

theorem fraction_of_work_left (hA : A_work_days = 15) (hB : B_work_days = 20) (hT : work_days_together = 4):
  (1 - 4 * (1 / 15 + 1 / 20)) = (8 / 15) :=
sorry

end fraction_of_work_left_l199_199676


namespace afternoon_registration_l199_199024

variable (m a t morning_absent : ℕ)

theorem afternoon_registration (m a t morning_absent afternoon : ℕ) (h1 : m = 25) (h2 : a = 4) (h3 : t = 42) (h4 : morning_absent = 3) : 
  afternoon = t - (m - morning_absent + morning_absent + a) :=
by sorry

end afternoon_registration_l199_199024


namespace snowdrift_ratio_l199_199083

theorem snowdrift_ratio
  (depth_first_day : ℕ := 20)
  (depth_second_day : ℕ)
  (h1 : depth_second_day + 24 = 34)
  (h2 : depth_second_day = 10) :
  depth_second_day / depth_first_day = 1 / 2 := by
  sorry

end snowdrift_ratio_l199_199083


namespace perpendicular_lines_l199_199610

theorem perpendicular_lines (a : ℝ) :
  (if a ≠ 0 then a^2 ≠ 0 else true) ∧ (a^2 * a + (-1/a) * 2 = -1) → (a = 2 ∨ a = 0) :=
by
  sorry

end perpendicular_lines_l199_199610


namespace max_marks_l199_199855

variable (M : ℝ)

def passing_marks (M : ℝ) : ℝ := 0.45 * M

theorem max_marks (h1 : passing_marks M = 225)
  (h2 : 180 + 45 = 225) : M = 500 :=
by
  sorry

end max_marks_l199_199855


namespace find_d_in_triangle_ABC_l199_199613

theorem find_d_in_triangle_ABC (AB BC AC : ℝ) (P : Type) (d : ℝ) 
  (h_AB : AB = 480) (h_BC : BC = 500) (h_AC : AC = 550)
  (h_segments_equal : ∀ (D D' E E' F F' : Type), true) : 
  d = 132000 / 654 :=
sorry

end find_d_in_triangle_ABC_l199_199613


namespace inequality_proof_l199_199926

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) : |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 := 
sorry

end inequality_proof_l199_199926


namespace inverse_36_mod_53_l199_199417

theorem inverse_36_mod_53 (h : 17 * 26 ≡ 1 [MOD 53]) : 36 * 27 ≡ 1 [MOD 53] :=
sorry

end inverse_36_mod_53_l199_199417


namespace gcd_228_1995_l199_199712

theorem gcd_228_1995 :
  Nat.gcd 228 1995 = 21 :=
sorry

end gcd_228_1995_l199_199712


namespace platform_length_l199_199862

theorem platform_length (train_length : ℝ) (time_cross_pole : ℝ) (time_cross_platform : ℝ) (speed : ℝ) 
  (h1 : train_length = 300) 
  (h2 : time_cross_pole = 18) 
  (h3 : time_cross_platform = 54)
  (h4 : speed = train_length / time_cross_pole) :
  train_length + (speed * time_cross_platform) - train_length = 600 := 
by
  sorry

end platform_length_l199_199862


namespace only_one_true_l199_199852

def statement_dong (xi: Prop) := ¬ xi
def statement_xi (nan: Prop) := ¬ nan
def statement_nan (dong: Prop) := ¬ dong
def statement_bei (nan: Prop) := ¬ (statement_nan nan) 

-- Define the main proof problem assuming all statements
theorem only_one_true : (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → true ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → true ∧ statement_nan dong → false ∧ statement_bei nan → false) 
                        ∨ (statement_dong xi → true ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∨ (statement_dong xi → false ∧ statement_xi nan → false ∧ statement_nan dong → false ∧ statement_bei nan → true) 
                        ∧ (statement_nan (statement_dong xi)) = true :=
sorry

end only_one_true_l199_199852


namespace percent_non_filler_l199_199188

def burger_weight : ℕ := 120
def filler_weight : ℕ := 30

theorem percent_non_filler : 
  let total_weight := burger_weight
  let filler := filler_weight
  let non_filler := total_weight - filler
  (non_filler / total_weight : ℚ) * 100 = 75 := by
  sorry

end percent_non_filler_l199_199188


namespace probability_more_ones_than_sixes_l199_199768

open ProbabilityTheory

noncomputable def prob_more_ones_than_sixes : ℚ :=
  let total_outcomes := 6^5 in
  let favorable_cases := 679 in
  favorable_cases / total_outcomes

theorem probability_more_ones_than_sixes (h_dice_fair : ∀ (i : ℕ), i ∈ Finset.range 6 → ℙ (i = 1) = 1 / 6) :
  prob_more_ones_than_sixes = 679 / 1944 :=
by {
  -- placeholder for the actual proof
  sorry
}

end probability_more_ones_than_sixes_l199_199768


namespace points_on_opposite_sides_of_line_range_m_l199_199596

theorem points_on_opposite_sides_of_line_range_m :
  (∀ (m : ℝ), (3 * 3 - 2 * 1 + m) * (3 * -4 - 2 * 6 + m) < 0 → -7 < m ∧ m < 24) := 
by sorry

end points_on_opposite_sides_of_line_range_m_l199_199596


namespace product_of_sum_and_reciprocal_nonneg_l199_199633

theorem product_of_sum_and_reciprocal_nonneg (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 :=
by
  sorry

end product_of_sum_and_reciprocal_nonneg_l199_199633


namespace outlet_pipe_rate_l199_199684

theorem outlet_pipe_rate (V_ft : ℝ) (cf : ℝ) (V_in : ℝ) (r_in : ℝ) (r_out1 : ℝ) (t : ℝ) (r_out2 : ℝ) :
    V_ft = 30 ∧ cf = 1728 ∧
    V_in = V_ft * cf ∧
    r_in = 5 ∧ r_out1 = 9 ∧ t = 4320 ∧
    V_in = (r_out1 + r_out2 - r_in) * t →
    r_out2 = 8 := by
  intros h
  sorry

end outlet_pipe_rate_l199_199684


namespace Kath_payment_l199_199956

noncomputable def reducedPrice (standardPrice discount : ℝ) : ℝ :=
  standardPrice - discount

noncomputable def totalCost (numPeople price : ℝ) : ℝ :=
  numPeople * price

theorem Kath_payment :
  let standardPrice := 8
  let discount := 3
  let numPeople := 6
  let movieTime := 16 -- 4 P.M. in 24-hour format
  let reduced := reducedPrice standardPrice discount
  totalCost numPeople reduced = 30 :=
by
  sorry

end Kath_payment_l199_199956


namespace sin_alpha_through_point_l199_199263

theorem sin_alpha_through_point (α : ℝ) (x y : ℝ) (h : x = -1 ∧ y = 2) (r : ℝ) (h_r : r = Real.sqrt (x^2 + y^2)) :
  Real.sin α = 2 * Real.sqrt 5 / 5 :=
by
  sorry

end sin_alpha_through_point_l199_199263


namespace infinite_series_sum_eq_one_fourth_l199_199388

theorem infinite_series_sum_eq_one_fourth :
  (∑' n : ℕ, 3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+2))) = 1 / 4 :=
sorry

end infinite_series_sum_eq_one_fourth_l199_199388


namespace probability_greg_rolls_more_ones_than_sixes_l199_199774

def number_of_outcomes : ℕ := 6^5

def count_combinations_zero_one_six : ℕ := 
  ((choose 5 0) * (4^5))

def count_combinations_one_one_six : ℕ := 
  ((choose 5 1) * (choose 4 1) * (4^3))

def count_combinations_two_one_six : ℕ :=
  ((choose 5 2) * (choose 3 2) * 4)

def total_combinations_equal_one_six : ℕ :=
  count_combinations_zero_one_six + count_combinations_one_one_six + count_combinations_two_one_six

def probability_equal_one_six : ℚ :=
  total_combinations_equal_one_six / number_of_outcomes

def probability_more_ones_than_sixes : ℚ :=
  1 / 2 * (1 - probability_equal_one_six)

theorem probability_greg_rolls_more_ones_than_sixes :
  probability_more_ones_than_sixes = (167 : ℚ) / 486 := by
  sorry

end probability_greg_rolls_more_ones_than_sixes_l199_199774


namespace max_sum_of_solutions_l199_199217

theorem max_sum_of_solutions (x y : ℤ) (h : 3 * x ^ 2 + 5 * y ^ 2 = 345) :
  x + y ≤ 13 :=
sorry

end max_sum_of_solutions_l199_199217


namespace sqrt_fourth_root_l199_199890

theorem sqrt_fourth_root (h : Real.sqrt (Real.sqrt (0.00000081)) = 0.1732) : Real.sqrt (Real.sqrt (0.00000081)) = 0.2 :=
by
  sorry

end sqrt_fourth_root_l199_199890


namespace proportion_solution_l199_199007

theorem proportion_solution (x : ℝ) (h : 0.6 / x = 5 / 8) : x = 0.96 :=
by 
  -- The proof will go here
  sorry

end proportion_solution_l199_199007


namespace price_decrease_percentage_l199_199649

variables (P Q : ℝ)
variables (Q' R R' : ℝ)

-- Condition: the number sold increased by 60%
def quantity_increase_condition : Prop :=
  Q' = Q * (1 + 0.60)

-- Condition: the total revenue increased by 28.000000000000025%
def revenue_increase_condition : Prop :=
  R' = R * (1 + 0.28000000000000025)

-- Definition: the original revenue R
def original_revenue : Prop :=
  R = P * Q

-- The new price P' after decreasing by x%
variables (P' : ℝ) (x : ℝ)
def new_price_condition : Prop :=
  P' = P * (1 - x / 100)

-- The new revenue R'
def new_revenue : Prop :=
  R' = P' * Q'

-- The proof problem
theorem price_decrease_percentage (P Q Q' R R' P' x : ℝ)
  (h1 : quantity_increase_condition Q Q')
  (h2 : revenue_increase_condition R R')
  (h3 : original_revenue P Q R)
  (h4 : new_price_condition P P' x)
  (h5 : new_revenue P' Q' R') :
  x = 20 :=
sorry

end price_decrease_percentage_l199_199649


namespace steel_bar_lengths_l199_199832

theorem steel_bar_lengths
  (x y z : ℝ)
  (h1 : 2 * x + y + 3 * z = 23)
  (h2 : x + 4 * y + 5 * z = 36) :
  x + 2 * y + 3 * z = 22 := 
sorry

end steel_bar_lengths_l199_199832


namespace storks_initially_l199_199639

-- Definitions for conditions
variable (S : ℕ) -- initial number of storks
variable (B : ℕ) -- initial number of birds

theorem storks_initially (h1 : B = 2) (h2 : S = B + 3 + 1) : S = 6 := by
  -- proof goes here
  sorry

end storks_initially_l199_199639


namespace binary_representation_88_l199_199105

def binary_representation (n : Nat) : String := sorry

theorem binary_representation_88 : binary_representation 88 = "1011000" := sorry

end binary_representation_88_l199_199105


namespace exp_sum_l199_199122

theorem exp_sum (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x + 3 * y) = 108 :=
sorry

end exp_sum_l199_199122


namespace gravitational_force_on_asteroid_l199_199162

theorem gravitational_force_on_asteroid :
  ∃ (k : ℝ), ∃ (f : ℝ), 
  (∀ (d : ℝ), f = k / d^2) ∧
  (d = 5000 → f = 700) →
  (∃ (f_asteroid : ℝ), f_asteroid = k / 300000^2 ∧ f_asteroid = 7 / 36) :=
sorry

end gravitational_force_on_asteroid_l199_199162


namespace money_left_after_expenses_l199_199678

theorem money_left_after_expenses : 
  let salary := 150000.00000000003
  let food := salary * (1 / 5)
  let house_rent := salary * (1 / 10)
  let clothes := salary * (3 / 5)
  let total_spent := food + house_rent + clothes
  let money_left := salary - total_spent
  money_left = 15000.00000000000 :=
by
  sorry

end money_left_after_expenses_l199_199678


namespace motorboat_speed_l199_199530

theorem motorboat_speed 
  (c : ℝ) (h_c : c = 2.28571428571)
  (t_up : ℝ) (h_t_up : t_up = 20 / 60)
  (t_down : ℝ) (h_t_down : t_down = 15 / 60) :
  ∃ v : ℝ, v = 16 :=
by
  sorry

end motorboat_speed_l199_199530


namespace quarters_and_dimes_l199_199438

theorem quarters_and_dimes (n : ℕ) (qval : ℕ := 25) (dval : ℕ := 10) 
  (hq : 20 * qval + 10 * dval = 10 * qval + n * dval) : 
  n = 35 :=
by
  sorry

end quarters_and_dimes_l199_199438


namespace find_unknown_rate_of_two_blankets_l199_199075

-- Definitions of conditions based on the problem statement
def purchased_blankets_at_100 : Nat := 3
def price_per_blanket_at_100 : Nat := 100
def total_cost_at_100 := purchased_blankets_at_100 * price_per_blanket_at_100

def purchased_blankets_at_150 : Nat := 3
def price_per_blanket_at_150 : Nat := 150
def total_cost_at_150 := purchased_blankets_at_150 * price_per_blanket_at_150

def purchased_blankets_at_x : Nat := 2
def blankets_total : Nat := 8
def average_price : Nat := 150
def total_cost := blankets_total * average_price

-- The proof statement
theorem find_unknown_rate_of_two_blankets (x : Nat) 
  (h : purchased_blankets_at_100 * price_per_blanket_at_100 + 
       purchased_blankets_at_150 * price_per_blanket_at_150 + 
       purchased_blankets_at_x * x = total_cost) : x = 225 :=
by sorry

end find_unknown_rate_of_two_blankets_l199_199075


namespace original_prices_sum_l199_199078

theorem original_prices_sum
  (new_price_candy_box : ℝ)
  (new_price_soda_can : ℝ)
  (increase_candy_box : ℝ)
  (increase_soda_can : ℝ)
  (h1 : new_price_candy_box = 10)
  (h2 : new_price_soda_can = 9)
  (h3 : increase_candy_box = 0.25)
  (h4 : increase_soda_can = 0.50) :
  let original_price_candy_box := new_price_candy_box / (1 + increase_candy_box)
  let original_price_soda_can := new_price_soda_can / (1 + increase_soda_can)
  original_price_candy_box + original_price_soda_can = 19 :=
by
  sorry

end original_prices_sum_l199_199078


namespace solve_for_b_l199_199004

theorem solve_for_b (b : ℝ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end solve_for_b_l199_199004


namespace line_parallel_slope_l199_199936

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope_l199_199936


namespace shaded_area_of_four_circles_l199_199615

open Real

noncomputable def area_shaded_region (r : ℝ) (num_circles : ℕ) : ℝ :=
  let area_quarter_circle := (π * r^2) / 4
  let area_triangle := (r * r) / 2
  let area_one_checkered_region := area_quarter_circle - area_triangle
  let num_checkered_regions := num_circles * 2
  num_checkered_regions * area_one_checkered_region

theorem shaded_area_of_four_circles : area_shaded_region 5 4 = 50 * (π - 2) :=
by
  sorry

end shaded_area_of_four_circles_l199_199615


namespace simplify_expression_l199_199154

theorem simplify_expression (x : ℝ) :
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 :=
by sorry

end simplify_expression_l199_199154


namespace always_odd_l199_199021

theorem always_odd (p m : ℕ) (hp : p % 2 = 1) : (p^3 + 3*p*m^2 + 2*m) % 2 = 1 := 
by sorry

end always_odd_l199_199021


namespace wire_division_l199_199194

theorem wire_division (L_wire_ft : Nat) (L_wire_inch : Nat) (L_part : Nat) (H1 : L_wire_ft = 5) (H2 : L_wire_inch = 4) (H3 : L_part = 16) :
  (L_wire_ft * 12 + L_wire_inch) / L_part = 4 :=
by 
  sorry

end wire_division_l199_199194


namespace blue_beads_count_l199_199690

-- Define variables and conditions
variables (r b : ℕ)

-- Define the conditions
def condition1 : Prop := r = 30
def condition2 : Prop := r / 3 = b / 2

-- State the theorem
theorem blue_beads_count (h1 : condition1 r) (h2 : condition2 r b) : b = 20 :=
sorry

end blue_beads_count_l199_199690


namespace acute_angle_sum_equals_pi_over_two_l199_199463

theorem acute_angle_sum_equals_pi_over_two (a b : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2)
  (h1 : 4 * (Real.cos a)^2 + 3 * (Real.cos b)^2 = 1)
  (h2 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 0) :
  2 * a + b = π / 2 := 
sorry

end acute_angle_sum_equals_pi_over_two_l199_199463


namespace find_m_value_l199_199595

noncomputable def hyperbola_m_value (m : ℝ) : Prop :=
  let a := 1
  let b := 2 * a
  m = -(1/4)

theorem find_m_value :
  (∀ x y : ℝ, x^2 + m * y^2 = 1 → b = 2 * a) → hyperbola_m_value m :=
by
  intro h
  sorry

end find_m_value_l199_199595


namespace total_volume_of_5_cubes_is_135_l199_199847

-- Define the edge length of a single cube
def edge_length : ℕ := 3

-- Define the volume of a single cube
def volume_single_cube (s : ℕ) : ℕ := s^3

-- State the total volume for a given number of cubes
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_single_cube s

-- Prove that for 5 cubes with an edge length of 3 meters, the total volume is 135 cubic meters
theorem total_volume_of_5_cubes_is_135 :
    total_volume 5 edge_length = 135 :=
by
  sorry

end total_volume_of_5_cubes_is_135_l199_199847


namespace brownies_cut_into_pieces_l199_199688

theorem brownies_cut_into_pieces (total_amount_made : ℕ) (pans : ℕ) (cost_per_brownie : ℕ) (brownies_sold : ℕ) 
  (h1 : total_amount_made = 32) (h2 : pans = 2) (h3 : cost_per_brownie = 2) (h4 : brownies_sold = total_amount_made / cost_per_brownie) :
  16 = brownies_sold :=
by
  sorry

end brownies_cut_into_pieces_l199_199688


namespace greatest_multiple_of_4_l199_199886

/-- 
Given x is a positive multiple of 4 and x^3 < 2000, 
prove that x is at most 12 and 
x = 12 is the greatest value that satisfies these conditions. 
-/
theorem greatest_multiple_of_4 (x : ℕ) (hx1 : x % 4 = 0) (hx2 : x^3 < 2000) : x ≤ 12 ∧ x = 12 :=
by
  sorry

end greatest_multiple_of_4_l199_199886


namespace water_current_speed_l199_199683

theorem water_current_speed (v : ℝ) (swimmer_speed : ℝ := 4) (time : ℝ := 3.5) (distance : ℝ := 7) :
  (4 - v) = distance / time → v = 2 := 
by
  sorry

end water_current_speed_l199_199683


namespace tan_identity_proof_l199_199657

theorem tan_identity_proof :
  (1 - Real.tan (100 * Real.pi / 180)) * (1 - Real.tan (35 * Real.pi / 180)) = 2 :=
by
  have tan_135 : Real.tan (135 * Real.pi / 180) = -1 := by sorry -- This needs a separate proof.
  have tan_sum_formula : ∀ A B : ℝ, Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B) := by sorry -- This needs a deeper exploration
  sorry -- Main proof to be filled

end tan_identity_proof_l199_199657


namespace convert_base_10_to_base_6_l199_199396

theorem convert_base_10_to_base_6 : 
  ∃ (digits : List ℕ), (digits.length = 4 ∧
    List.foldr (λ (x : ℕ) (acc : ℕ) => acc * 6 + x) 0 digits = 314 ∧
    digits = [1, 2, 4, 2]) := by
  sorry

end convert_base_10_to_base_6_l199_199396


namespace solution_set_of_inequality_l199_199651

theorem solution_set_of_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end solution_set_of_inequality_l199_199651


namespace candies_remaining_after_yellow_eaten_l199_199344

theorem candies_remaining_after_yellow_eaten :
  let red_candies := 40
  let yellow_candies := 3 * red_candies - 20
  let blue_candies := yellow_candies / 2
  red_candies + blue_candies = 90 :=
by
  sorry

end candies_remaining_after_yellow_eaten_l199_199344


namespace point_D_coordinates_l199_199720

-- Define the vectors and points
structure Point where
  x : Int
  y : Int

def vector_add (p1 p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

def scalar_multiply (k : Int) (p : Point) : Point :=
  { x := k * p.x, y := k * p.y }

def ab := Point.mk 5 (-3)
def c := Point.mk (-1) 3
def cd := scalar_multiply 2 ab

def D : Point := vector_add c cd

-- Theorem statement
theorem point_D_coordinates :
  D = Point.mk 9 (-3) :=
sorry

end point_D_coordinates_l199_199720


namespace proof_problem_l199_199515

variable (x y : ℝ)

theorem proof_problem :
  ¬ (x^2 + x^2 = x^4) ∧
  ¬ ((x - y)^2 = x^2 - y^2) ∧
  ¬ ((x^2 * y)^3 = x^6 * y) ∧
  ((-x)^2 * x^3 = x^5) :=
by
  sorry

end proof_problem_l199_199515


namespace max_squares_on_checkerboard_l199_199169

theorem max_squares_on_checkerboard (n : ℕ) (h1 : n = 7) (h2 : ∀ s : ℕ, s = 2) : ∃ max_squares : ℕ, max_squares = 18 := sorry

end max_squares_on_checkerboard_l199_199169


namespace countMultiplesOf30Between900And27000_l199_199178

noncomputable def smallestPerfectSquareDivisibleBy30 : ℕ :=
  900

noncomputable def smallestPerfectCubeDivisibleBy30 : ℕ :=
  27000

theorem countMultiplesOf30Between900And27000 :
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  upper_bound - lower_bound + 1 = 871 :=
  by
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  show upper_bound - lower_bound + 1 = 871;
  sorry

end countMultiplesOf30Between900And27000_l199_199178


namespace geometric_sequence_sum_l199_199803

theorem geometric_sequence_sum (a : Nat → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (hq : q > 1) (h2011_root : 4 * a 2011 ^ 2 - 8 * a 2011 + 3 = 0)
  (h2012_root : 4 * a 2012 ^ 2 - 8 * a 2012 + 3 = 0) :
  a 2013 + a 2014 = 18 :=
sorry

end geometric_sequence_sum_l199_199803


namespace find_particular_number_l199_199428

theorem find_particular_number (x : ℝ) (h : 4 * x * 25 = 812) : x = 8.12 :=
by sorry

end find_particular_number_l199_199428


namespace francie_has_3_dollars_remaining_l199_199915

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l199_199915


namespace area_of_nonagon_on_other_cathetus_l199_199811

theorem area_of_nonagon_on_other_cathetus 
    (A₁ A₂ A₃ : ℝ) 
    (h1 : A₁ = 2019) 
    (h2 : A₂ = 1602) 
    (h3 : A₁ = A₂ + A₃) : 
    A₃ = 417 :=
by
  rw [h1, h2] at h3
  linarith

end area_of_nonagon_on_other_cathetus_l199_199811


namespace probability_XOXOXOX_l199_199581

theorem probability_XOXOXOX :
  let X := 4;
      O := 3;
      total_positions := 7;
      specific_arrangement := 1;
      total_arrangements := Nat.choose total_positions X in
  1 / total_arrangements = 1 / 35 := by
  sorry

end probability_XOXOXOX_l199_199581


namespace relationship_among_log_exp_powers_l199_199225

theorem relationship_among_log_exp_powers :
  let a := Real.log 0.3 / Real.log 2
  let b := Real.exp (0.3 * Real.log 2)
  let c := Real.exp (0.2 * Real.log 0.3)
  a < c ∧ c < b :=
by
  sorry

end relationship_among_log_exp_powers_l199_199225


namespace equal_distances_l199_199482

theorem equal_distances (c : ℝ) (distance : ℝ) :
  abs (2 - -4) = distance ∧ (abs (c - -4) = distance ∨ abs (c - 2) = distance) ↔ (c = -10 ∨ c = 8) :=
by
  sorry

end equal_distances_l199_199482


namespace victor_decks_l199_199510

theorem victor_decks (V : ℕ) (cost_per_deck total_spent friend_decks : ℕ) 
  (h1 : cost_per_deck = 8)
  (h2 : total_spent = 64)
  (h3 : friend_decks = 2) 
  (h4 : 8 * V + 8 * friend_decks = total_spent) : 
  V = 6 :=
by sorry

end victor_decks_l199_199510


namespace baker_sold_cakes_l199_199541

def initialCakes : Nat := 110
def additionalCakes : Nat := 76
def remainingCakes : Nat := 111
def cakesSold : Nat := 75

theorem baker_sold_cakes :
  initialCakes + additionalCakes - remainingCakes = cakesSold := by
  sorry

end baker_sold_cakes_l199_199541


namespace calculate_total_interest_l199_199838

theorem calculate_total_interest :
  let total_money := 9000
  let invested_at_8_percent := 4000
  let invested_at_9_percent := total_money - invested_at_8_percent
  let interest_rate_8 := 0.08
  let interest_rate_9 := 0.09
  let interest_from_8_percent := invested_at_8_percent * interest_rate_8
  let interest_from_9_percent := invested_at_9_percent * interest_rate_9
  let total_interest := interest_from_8_percent + interest_from_9_percent
  total_interest = 770 :=
by
  sorry

end calculate_total_interest_l199_199838


namespace smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l199_199597

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - 3 * Real.pi / 5)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧
  ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T := by
  sorry

theorem axis_of_symmetry :
  ∃ k : ℤ, (∀ x, f x = f (11 * Real.pi / 20 + k * Real.pi / 2)) := by
  sorry

theorem minimum_value_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = -1 := by
  sorry

end smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l199_199597


namespace opposite_quotient_l199_199124

theorem opposite_quotient {a b : ℝ} (h1 : a ≠ b) (h2 : a = -b) : a / b = -1 := 
sorry

end opposite_quotient_l199_199124


namespace correct_statements_l199_199303

-- Definitions for each statement
def statement_1 := ∀ p q : ℤ, q ≠ 0 → (∃ n : ℤ, ∃ d : ℤ, p = n ∧ q = d ∧ (n, d) = (p, q))
def statement_2 := ∀ r : ℚ, (r > 0 ∨ r < 0) ∨ (∃ d : ℚ, d ≥ 0)
def statement_3 := ∀ x y : ℚ, abs x = abs y → x = y
def statement_4 := ∀ x : ℚ, (-x = x ∧ abs x = x) → x = 0
def statement_5 := ∀ x y : ℚ, abs x > abs y → x > y
def statement_6 := (∃ n : ℕ, n > 0) ∧ (∀ r : ℚ, r > 0 → ∃ q : ℚ, q > 0 ∧ q < r)

-- Main theorem: Prove that exactly 3 statements are correct
theorem correct_statements : 
  (statement_1 ∧ statement_4 ∧ statement_6) ∧ 
  (¬ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_5) :=
by
  sorry

end correct_statements_l199_199303


namespace increasing_intervals_l199_199235

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x - Real.pi / 3)

theorem increasing_intervals :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi) 0 →
    (f x > f (x - ε) ∧ f x < f (x + ε) ∧ x ∈ Set.Icc (-Real.pi) (-7 * Real.pi / 12) ∪ Set.Icc (-Real.pi / 12) 0) :=
sorry

end increasing_intervals_l199_199235


namespace average_brown_MnMs_l199_199487

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l199_199487


namespace algebra_expression_value_l199_199721

variable (x : ℝ)

theorem algebra_expression_value (h : x^2 - 3 * x - 12 = 0) : 3 * x^2 - 9 * x + 5 = 41 := 
sorry

end algebra_expression_value_l199_199721


namespace pencils_given_out_l199_199659

-- Defining the conditions
def num_children : ℕ := 4
def pencils_per_child : ℕ := 2

-- Formulating the problem statement, with the goal to prove the total number of pencils
theorem pencils_given_out : num_children * pencils_per_child = 8 := 
by 
  sorry

end pencils_given_out_l199_199659


namespace find_areas_after_shortening_l199_199478

-- Define initial dimensions
def initial_length : ℤ := 5
def initial_width : ℤ := 7
def shortened_by : ℤ := 2

-- Define initial area condition
def initial_area_condition : Prop := 
  initial_length * (initial_width - shortened_by) = 15 ∨ (initial_length - shortened_by) * initial_width = 15

-- Define the resulting areas for shortening each dimension
def area_shortening_length : ℤ := (initial_length - shortened_by) * initial_width
def area_shortening_width : ℤ := initial_length * (initial_width - shortened_by)

-- Statement for proof
theorem find_areas_after_shortening
  (h : initial_area_condition) :
  area_shortening_length = 21 ∧ area_shortening_width = 25 :=
sorry

end find_areas_after_shortening_l199_199478


namespace chord_length_squared_l199_199893

theorem chord_length_squared
  (r5 r10 r15 : ℝ) 
  (externally_tangent : r5 = 5 ∧ r10 = 10)
  (internally_tangent : r15 = 15)
  (common_external_tangent : r15 - r10 - r5 = 0) :
  ∃ PQ_squared : ℝ, PQ_squared = 622.44 :=
by
  sorry

end chord_length_squared_l199_199893


namespace product_units_digit_of_five_consecutive_l199_199173

theorem product_units_digit_of_five_consecutive (n : ℕ) : 
  ((n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10) = 0 := 
sorry

end product_units_digit_of_five_consecutive_l199_199173


namespace trapezoid_area_division_l199_199813

/-- Given a trapezoid where one base is 150 units longer than the other base and the segment joining the midpoints of the legs divides the trapezoid into two regions whose areas are in the ratio 3:4, prove that the greatest integer less than or equal to (x^2 / 150) is 300, where x is the length of the segment that joins the midpoints of the legs and divides the trapezoid into two equal areas. -/
theorem trapezoid_area_division (b h x : ℝ) (h_b : b = 112.5) (h_x : x = 150) :
  ⌊x^2 / 150⌋ = 300 :=
by
  sorry

end trapezoid_area_division_l199_199813


namespace average_age_of_community_l199_199614

theorem average_age_of_community 
  (k : ℕ) 
  (hwomen : ℕ := 7 * k)
  (hmen : ℕ := 5 * k)
  (avg_age_women : ℝ := 30)
  (avg_age_men : ℝ := 35)
  (total_women_age : ℝ := avg_age_women * hwomen)
  (total_men_age : ℝ := avg_age_men * hmen)
  (total_population : ℕ := hwomen + hmen)
  (total_age : ℝ := total_women_age + total_men_age) : 
  total_age / total_population = 32 + 1 / 12 :=
by
  sorry

end average_age_of_community_l199_199614


namespace polar_to_rect_l199_199895

open Real 

theorem polar_to_rect (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 3 * π / 4) : 
  (r * cos θ, r * sin θ) = (-3 / Real.sqrt 2, 3 / Real.sqrt 2) :=
by
  -- Optional step: you can introduce the variables as they have already been proved using the given conditions
  have hr : r = 3 := h_r
  have hθ : θ = 3 * π / 4 := h_θ
  -- Goal changes according to the values of r and θ derived from the conditions
  sorry

end polar_to_rect_l199_199895


namespace binomial_square_expression_l199_199329

theorem binomial_square_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end binomial_square_expression_l199_199329


namespace greatest_number_of_fruit_baskets_l199_199979

def number_of_oranges : ℕ := 18
def number_of_pears : ℕ := 27
def number_of_bananas : ℕ := 12

theorem greatest_number_of_fruit_baskets :
  Nat.gcd (Nat.gcd number_of_oranges number_of_pears) number_of_bananas = 3 :=
by
  sorry

end greatest_number_of_fruit_baskets_l199_199979


namespace find_angle_MBA_l199_199452

-- Define the angles and the triangle
def triangle (A B C : Type) := true

-- Define the angles in degrees
def angle (deg : ℝ) := deg

-- Assume angles' degrees as given in the problem
variables {A B C M : Type}
variable {BAC ABC MAB MCA MBA : ℝ}

-- Given conditions
axiom angle_BAC : angle BAC = 30
axiom angle_ABC : angle ABC = 70
axiom angle_MAB : angle MAB = 20
axiom angle_MCA : angle MCA = 20

-- Prove that angle MBA is 30 degrees
theorem find_angle_MBA : angle MBA = 30 := 
by 
  sorry

end find_angle_MBA_l199_199452


namespace remaining_miles_l199_199018

theorem remaining_miles (total_miles : ℕ) (driven_miles : ℕ) (h1: total_miles = 1200) (h2: driven_miles = 642) :
  total_miles - driven_miles = 558 :=
by
  sorry

end remaining_miles_l199_199018


namespace total_payment_leila_should_pay_l199_199375

-- Definitions of the conditions
def chocolateCakes := 3
def chocolatePrice := 12
def strawberryCakes := 6
def strawberryPrice := 22

-- Mathematical equivalent proof problem
theorem total_payment_leila_should_pay : 
  chocolateCakes * chocolatePrice + strawberryCakes * strawberryPrice = 168 := 
by 
  sorry

end total_payment_leila_should_pay_l199_199375


namespace derivative_u_l199_199111

noncomputable def u (x : ℝ) : ℝ :=
  let z := Real.sin x
  let y := x^2
  Real.exp (z - 2 * y)

theorem derivative_u (x : ℝ) :
  deriv u x = Real.exp (Real.sin x - 2 * x^2) * (Real.cos x - 4 * x) :=
by
  sorry

end derivative_u_l199_199111


namespace fraction_is_three_eighths_l199_199861

theorem fraction_is_three_eighths (F N : ℝ) 
  (h1 : (4 / 5) * F * N = 24) 
  (h2 : (250 / 100) * N = 199.99999999999997) : 
  F = 3 / 8 :=
by 
  sorry

end fraction_is_three_eighths_l199_199861


namespace infinite_series_sum_l199_199549

theorem infinite_series_sum : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / ((4 * n - 1)^3 * (4 * n + 3)^3)) = 1 / 972 := 
by 
  sorry

end infinite_series_sum_l199_199549


namespace factor_fraction_l199_199560

/- Definitions based on conditions -/
variables {a b c : ℝ}

theorem factor_fraction :
  ( (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 ) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
begin
  sorry
end

end factor_fraction_l199_199560


namespace triangle_smallest_side_l199_199047

theorem triangle_smallest_side (a b c : ℝ) (h : b^2 + c^2 ≥ 5 * a^2) : 
    (a ≤ b ∧ a ≤ c) := 
sorry

end triangle_smallest_side_l199_199047


namespace values_of_x_for_f_l199_199414

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

noncomputable def is_monotonically_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

theorem values_of_x_for_f (f : ℝ → ℝ) 
  (h1 : is_even_function f) 
  (h2 : is_monotonically_increasing_on_nonneg f) : 
  (∀ x : ℝ, f (2*x - 1) < f 3 ↔ (-1 < x ∧ x < 2)) :=
by
  sorry

end values_of_x_for_f_l199_199414


namespace neg_exists_exp_l199_199934

theorem neg_exists_exp (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x < 0)) = (∀ x : ℝ, Real.exp x ≥ 0) :=
by
  sorry

end neg_exists_exp_l199_199934


namespace find_four_digit_number_l199_199526

theorem find_four_digit_number : ∃ x : ℕ, (1000 ≤ x ∧ x ≤ 9999) ∧ (x % 7 = 0) ∧ (x % 29 = 0) ∧ (19 * x % 37 = 3) ∧ x = 5075 :=
by
  sorry

end find_four_digit_number_l199_199526


namespace polygon_sides_l199_199870

theorem polygon_sides (n : ℕ) (exterior_angle : ℝ) : exterior_angle = 36 ∧ (∑ i in finset.range n, 360 / n) = 360 → n = 10 := 
by
  intros h
  sorry

end polygon_sides_l199_199870


namespace min_value_expression_l199_199462

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 10)^2 + (3 * sin α + 4 * cos β - 20)^2 ≥ 100 :=
sorry

end min_value_expression_l199_199462


namespace distance_from_neg2_l199_199282

theorem distance_from_neg2 (x : ℝ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 := 
by sorry

end distance_from_neg2_l199_199282


namespace no_nat_number_satisfies_l199_199134

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l199_199134


namespace count_special_numbers_l199_199752

-- Definitions
def is_three_digit_integer (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000
def is_less_than_600 (n : ℕ) : Prop := n < 600
def has_at_least_two_identical_digits (n : ℕ) : Prop :=
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

-- Theorem to prove
theorem count_special_numbers : 
  ∃! (cnt : ℕ), cnt = 140 ∧ 
  (∀ n, is_three_digit_integer n → is_less_than_600 n → has_at_least_two_identical_digits n) :=
sorry

end count_special_numbers_l199_199752


namespace common_ratio_geometric_sequence_l199_199797

theorem common_ratio_geometric_sequence 
  (a1 : ℝ) 
  (q : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h2 : 8 * S 6 = 7 * S 3) 
  (hq : q ≠ 1) : 
  q = -1 / 2 := 
sorry

end common_ratio_geometric_sequence_l199_199797


namespace no_nat_number_satisfies_l199_199133

theorem no_nat_number_satisfies (n : ℕ) : ¬ ((n^2 + 6 * n + 2019) % 100 = 0) :=
sorry

end no_nat_number_satisfies_l199_199133


namespace eval_expression_l199_199393

theorem eval_expression : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 :=
by
  sorry

end eval_expression_l199_199393


namespace tim_campaign_funds_l199_199834

theorem tim_campaign_funds :
  let max_donors := 500
  let max_donation := 1200
  let half_donation := max_donation / 2
  let half_donors := 3 * max_donors
  let total_from_max := max_donors * max_donation
  let total_from_half := half_donors * half_donation
  let total_raised := (total_from_max + total_from_half) / 0.4
  in total_raised = 3750000 := by
  have h1 : max_donation = 1200 := rfl
  have h2 : max_donors = 500 := rfl
  have h3 : half_donation = 600 := by norm_num [half_donation, h1]
  have h4 : half_donors = 1500 := by norm_num [half_donors, h2]
  have h5 : total_from_max = 600000 := by norm_num [total_from_max, h1, h2]
  have h6 : total_from_half = 900000 := by norm_num [total_from_half, h3, h4]
  have h7 : total_raised = (600000 + 900000) / 0.4 := rfl
  have h8 : total_raised = 3750000 := by norm_num [h7]
  exact h8

end tim_campaign_funds_l199_199834


namespace roger_final_money_is_correct_l199_199285

noncomputable def initial_money : ℝ := 84
noncomputable def birthday_money : ℝ := 56
noncomputable def found_money : ℝ := 20
noncomputable def spent_on_game : ℝ := 35
noncomputable def spent_percentage : ℝ := 0.15

noncomputable def final_money 
  (initial_money birthday_money found_money spent_on_game spent_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_money + birthday_money + found_money
  let remaining_after_game := total_before_spending - spent_on_game
  let spent_on_gift := spent_percentage * remaining_after_game
  remaining_after_game - spent_on_gift

theorem roger_final_money_is_correct :
  final_money initial_money birthday_money found_money spent_on_game spent_percentage = 106.25 :=
by
  sorry

end roger_final_money_is_correct_l199_199285


namespace x_cube_plus_y_cube_l199_199944

theorem x_cube_plus_y_cube (x y : ℝ) (h₁ : x + y = 1) (h₂ : x^2 + y^2 = 3) : x^3 + y^3 = 4 :=
sorry

end x_cube_plus_y_cube_l199_199944


namespace no_roots_in_interval_l199_199273

theorem no_roots_in_interval (a : ℝ) (x : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h_eq: a ^ x + a ^ (-x) = 2 * a) : x < -1 ∨ x > 1 :=
sorry

end no_roots_in_interval_l199_199273


namespace values_of_j_for_exactly_one_real_solution_l199_199715

open Real

theorem values_of_j_for_exactly_one_real_solution :
  ∀ j : ℝ, (∀ x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) → (j = 0 ∨ j = -36) := by
sorry

end values_of_j_for_exactly_one_real_solution_l199_199715


namespace product_factors_eq_l199_199389

theorem product_factors_eq :
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) * (1 - 1/8) * (1 - 1/9) * (1 - 1/10) * (1 - 1/11) = 1 / 11 := 
by
  sorry

end product_factors_eq_l199_199389


namespace mean_of_six_numbers_l199_199504

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l199_199504


namespace inequality_proof_l199_199920

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l199_199920


namespace prob_more_1s_than_6s_l199_199759

noncomputable def probability_more_ones_than_sixes (n : ℕ) : ℚ :=
  let total_outcomes := 6^n
  let equal_1s_6s :=  sum_finsupp (λ k1 k6 : _n_, if (k1 = k6) 
    then binom n k1 * binom (n - k1) k6 * (4 ^ (n - k1 - k6)) else 0)
  let prob_equal := equal_1s_6s / total_outcomes
  let final_probability := (1 - prob_equal) / 2
  final_probability

theorem prob_more_1s_than_6s :
  probability_more_ones_than_sixes 5 = 2676 / 7776 :=
sorry

end prob_more_1s_than_6s_l199_199759


namespace water_height_in_cylinder_l199_199880

theorem water_height_in_cylinder :
  let r_cone := 10 -- Radius of the cone in cm
  let h_cone := 15 -- Height of the cone in cm
  let r_cylinder := 20 -- Radius of the cylinder in cm
  let volume_cone := (1 / 3) * Real.pi * r_cone^2 * h_cone
  volume_cone = 500 * Real.pi -> 
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  h_cylinder = 1.25 := 
by
  intros r_cone h_cone r_cylinder volume_cone h_volume
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  have : h_cylinder = 1.25 := by
    sorry
  exact this

end water_height_in_cylinder_l199_199880


namespace base10_to_base7_l199_199840

-- Definition of base conversion
def base7_representation (n : ℕ) : ℕ :=
  match n with
  | 729 => 2 * 7^3 + 6 * 7^1 + 1 * 7^0
  | _   => sorry  -- other cases are not required for the given problem

theorem base10_to_base7 (n : ℕ) (h1 : n = 729) : base7_representation n = 261 := by
  rw [h1]
  unfold base7_representation
  norm_num
  rfl

end base10_to_base7_l199_199840


namespace least_possible_integral_BC_l199_199860

theorem least_possible_integral_BC :
  ∃ (BC : ℕ), (BC > 0) ∧ (BC ≥ 15) ∧ 
    (7 + BC > 15) ∧ (25 + 10 > BC) ∧ 
    (7 + 15 > BC) ∧ (25 + BC > 10) := by
    sorry

end least_possible_integral_BC_l199_199860


namespace initial_goal_proof_l199_199816

def marys_collection (k : ℕ) : ℕ := 5 * k
def scotts_collection (m : ℕ) : ℕ := m / 3
def total_collected (k : ℕ) (m : ℕ) (s : ℕ) : ℕ := k + m + s
def initial_goal (total : ℕ) (excess : ℕ) : ℕ := total - excess

theorem initial_goal_proof : 
  initial_goal (total_collected 600 (marys_collection 600) (scotts_collection (marys_collection 600))) 600 = 4000 :=
by
  sorry

end initial_goal_proof_l199_199816


namespace not_all_zero_implies_at_least_one_nonzero_l199_199508

variable {a b c : ℤ}

theorem not_all_zero_implies_at_least_one_nonzero (h : ¬ (a = 0 ∧ b = 0 ∧ c = 0)) : 
  a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 := 
by 
  sorry

end not_all_zero_implies_at_least_one_nonzero_l199_199508


namespace base_conversion_l199_199607

theorem base_conversion (k : ℕ) (h : 26 = 3*k + 2) : k = 8 := 
by 
  sorry

end base_conversion_l199_199607


namespace quadratic_equation_solution_unique_l199_199494

theorem quadratic_equation_solution_unique (b : ℝ) (hb : b ≠ 0) (h1_sol : ∀ x1 x2 : ℝ, 2*b*x1^2 + 16*x1 + 5 = 0 → 2*b*x2^2 + 16*x2 + 5 = 0 → x1 = x2) :
  ∃ x : ℝ, x = -5/8 ∧ 2*b*x^2 + 16*x + 5 = 0 :=
by
  sorry

end quadratic_equation_solution_unique_l199_199494


namespace total_cost_of_soup_l199_199660

theorem total_cost_of_soup 
  (pounds_beef : ℕ) (pounds_veg : ℕ) (cost_veg_per_pound : ℕ) (beef_price_multiplier : ℕ)
  (h1 : pounds_beef = 4)
  (h2 : pounds_veg = 6)
  (h3 : cost_veg_per_pound = 2)
  (h4 : beef_price_multiplier = 3):
  (pounds_veg * cost_veg_per_pound + pounds_beef * (cost_veg_per_pound * beef_price_multiplier)) = 36 :=
by
  sorry

end total_cost_of_soup_l199_199660


namespace complement_A_union_B_l199_199968

def is_positive_integer_less_than_9 (n : ℕ) : Prop :=
  n > 0 ∧ n < 9

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def U := {n : ℕ | is_positive_integer_less_than_9 n}
noncomputable def A := {n ∈ U | is_odd n}
noncomputable def B := {n ∈ U | is_multiple_of_3 n}

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {2, 4, 8} :=
sorry

end complement_A_union_B_l199_199968


namespace roads_probability_l199_199079

theorem roads_probability :
  ∀ (P_A P_B : ℝ), P_A = 2 / 3 → P_B = 3 / 4 →
    (1 - (1 - P_A) * (1 - P_B) = 11 / 12) :=
by
  intros P_A P_B hA hB
  rw [hA, hB]
  -- rest will be the proof, which we skip with sorry
  sorry

end roads_probability_l199_199079


namespace probability_XOXOXOX_l199_199570

theorem probability_XOXOXOX (arrangement : list char) 
  (h_len : arrangement.length = 7) 
  (h_X_count : arrangement.count 'X' = 4) 
  (h_O_count : arrangement.count 'O' = 3) :
  let total_arrangements := nat.choose 7 4 in 
  let favorable_outcomes := 1 in
  favorable_outcomes / total_arrangements = 1 / 35 :=
by
  -- proof
  sorry

end probability_XOXOXOX_l199_199570


namespace calories_per_burger_l199_199210

-- Conditions given in the problem
def burgers_per_day : Nat := 3
def days : Nat := 2
def total_calories : Nat := 120

-- Total burgers Dimitri will eat in the given period
def total_burgers := burgers_per_day * days

-- Prove that the number of calories per burger is 20
theorem calories_per_burger : total_calories / total_burgers = 20 := 
by 
  -- Skipping the proof with 'sorry' as instructed
  sorry

end calories_per_burger_l199_199210


namespace income_calculation_l199_199040

theorem income_calculation (x : ℕ) (h1 : ∃ x : ℕ, income = 8*x ∧ expenditure = 7*x)
  (h2 : savings = 5000)
  (h3 : income = expenditure + savings) : income = 40000 :=
by {
  sorry
}

end income_calculation_l199_199040


namespace barbara_total_candies_l199_199542

-- Condition: Barbara originally has 9 candies.
def C1 := 9

-- Condition: Barbara buys 18 more candies.
def C2 := 18

-- Question (proof problem): Prove that the total number of candies Barbara has is 27.
theorem barbara_total_candies : C1 + C2 = 27 := by
  -- Proof steps are not required, hence using sorry.
  sorry

end barbara_total_candies_l199_199542


namespace smallest_apples_l199_199848

theorem smallest_apples (A : ℕ) (h1 : A % 9 = 2) (h2 : A % 10 = 2) (h3 : A % 11 = 2) (h4 : A > 2) : A = 992 :=
sorry

end smallest_apples_l199_199848


namespace expected_value_of_8_sided_die_l199_199539

noncomputable def expected_value_of_winnings : ℝ := 
  let multiples_of_three := [3, 6]
  let probability_of_multiple_of_three := 2/8
  let expected_winnings := (probability_of_multiple_of_three * (3 + 6 : ℝ)) / 2
  expected_winnings

theorem expected_value_of_8_sided_die : expected_value_of_winnings = 2.25 := 
  by
    sorry

end expected_value_of_8_sided_die_l199_199539


namespace find_xyz_l199_199416

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := 
by 
  sorry

end find_xyz_l199_199416


namespace milk_production_days_l199_199781

theorem milk_production_days (y : ℕ) :
  (y + 4) * (y + 2) * (y + 6) / (y * (y + 3) * (y + 4)) = y * (y + 3) * (y + 6) / ((y + 2) * (y + 4)) :=
sorry

end milk_production_days_l199_199781


namespace not_all_same_probability_l199_199060

-- Definition of the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8^5

-- Definition of the number of outcomes where all five dice show the same number
def same_number_outcomes : ℕ := 8

-- Definition to find the probability that not all 5 dice show the same number
def probability_not_all_same : ℚ := 1 - (same_number_outcomes / total_outcomes)

-- Statement of the main theorem
theorem not_all_same_probability : probability_not_all_same = (4095 : ℚ) / 4096 :=
by
  rw [probability_not_all_same, same_number_outcomes, total_outcomes]
  -- Simplification steps would go here, but we use sorry to skip the proof
  sorry

end not_all_same_probability_l199_199060


namespace scientific_notation_of_32000000_l199_199381

theorem scientific_notation_of_32000000 : 32000000 = 3.2 * 10^7 := 
  sorry

end scientific_notation_of_32000000_l199_199381


namespace prob_f_has_root_l199_199413

-- Let X be a random variable that follows a binomial distribution with n = 5 and p = 1/2
def X : Probability Mass Function nat := binomial 5 (1/2)

-- Define the function f(x) = x^2 + 4x + X
def f (x : ℝ) (X : ℝ) := x^2 + 4 * x + X

-- Define the event Ω where f(x) has a root
def has_root (X : ℝ) := ∃ x : ℝ, f x X = 0

-- Probability that has_root occurs; we provide the solution in the question
theorem prob_f_has_root : ∀ X : ℕ, X ∼ binomial 5 (1/2) -> P(has_root X) = 31 / 32 := by
  -- The proof is omitted
  sorry

end prob_f_has_root_l199_199413


namespace diameter_bisects_and_perpendicular_l199_199340

-- Define relevant concepts and conditions:
variables {α : Type*} [MetricSpace α] {circle : Circle α} {d1 d2 : Point α} 

-- Definition of the problem statement to be proved.
-- We have that d1 is the midpoint of chord d2 and the statement to prove is defined below.
theorem diameter_bisects_and_perpendicular
  (h1 : ∃ (A B C : Point α), ¬Collinear A B C)  -- There exist three non-collinear points
  (h2 : ∀ {A B C : Triangle α}, Circumcenter A B C = Intersection (PerpendicularBisector A B) (PerpendicularBisector B C)) -- Definition of circumcenter
  (h3 : ∀ {A B : LineSegment α}, A = PerpendicularBisector A B → d1 ∈ A → dist d1 = dist d2) -- Points on the perpendicular bisector are equidistant from endpoints
  (h4 : ∀ {C1 C2 : Circle α},  Radius C1 = Radius C2 → CentralAngle C1 = CentralAngle C2) -- In congruent circles, equal central angles correspond to equal arcs
  : ∀ {C : Circle α} {P Q : LineSegment α}, (Diameter P Q) = PerpendicularBisector P Q → Perpendicular P Q := 
sorry -- Details of the proof are skipped

end diameter_bisects_and_perpendicular_l199_199340


namespace trigonometric_identity_l199_199247

theorem trigonometric_identity (φ : ℝ) 
  (h : Real.cos (π / 2 + φ) = (Real.sqrt 3) / 2) : 
  Real.cos (3 * π / 2 - φ) + Real.sin (φ - π) = Real.sqrt 3 := 
by
  sorry

end trigonometric_identity_l199_199247


namespace perfect_square_trinomial_l199_199326

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l199_199326


namespace value_of_q_l199_199742

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end value_of_q_l199_199742


namespace total_admission_cost_l199_199954

-- Define the standard admission cost
def standard_admission_cost : ℕ := 8

-- Define the discount if watched before 6 P.M.
def discount : ℕ := 3

-- Define the number of people in Kath's group
def number_of_people : ℕ := 1 + 2 + 3

-- Define the movie starting time (not directly used in calculation but part of the conditions)
def movie_start_time : ℕ := 16

-- Define the discounted admission cost
def discounted_admission_cost : ℕ := standard_admission_cost - discount

-- Prove that the total cost Kath will pay is $30
theorem total_admission_cost : number_of_people * discounted_admission_cost = 30 := by
  -- sorry is used to skip the proof
  sorry

end total_admission_cost_l199_199954


namespace probability_of_nonzero_product_probability_of_valid_dice_values_l199_199222

def dice_values := {x : ℕ | 1 ≤ x ∧ x ≤ 6}

def valid_dice_values := {x : ℕ | 2 ≤ x ∧ x ≤ 6}

noncomputable def probability_no_one : ℚ := 625 / 1296

theorem probability_of_nonzero_product (a b c d : ℕ) 
  (ha : a ∈ dice_values) (hb : b ∈ dice_values) 
  (hc : c ∈ dice_values) (hd : d ∈ dice_values) : 
  (a - 1) * (b - 1) * (c - 1) * (d - 1) ≠ 0 ↔ 
  (a ∈ valid_dice_values ∧ b ∈ valid_dice_values ∧ 
   c ∈ valid_dice_values ∧ d ∈ valid_dice_values) :=
sorry

theorem probability_of_valid_dice_values : 
  probability_no_one = (5 / 6) ^ 4 :=
sorry

end probability_of_nonzero_product_probability_of_valid_dice_values_l199_199222


namespace triangle_angle_B_l199_199963

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h : a / b = 3 / Real.sqrt 7) (h2 : b / c = Real.sqrt 7 / 2) : B = Real.pi / 3 :=
by
  sorry

end triangle_angle_B_l199_199963


namespace cost_of_article_l199_199757

variable (C : ℝ)
variable (SP1 SP2 : ℝ)
variable (G : ℝ)

theorem cost_of_article (h1 : SP1 = 380) 
                        (h2 : SP2 = 420)
                        (h3 : SP1 = C + G)
                        (h4 : SP2 = C + G + 0.08 * G) :
  C = 120 :=
by
  sorry

end cost_of_article_l199_199757


namespace no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l199_199360

theorem no_integer_solutions_x_x_plus_1_eq_13y_plus_1 :
  ¬ ∃ x y : ℤ, x * (x + 1) = 13 * y + 1 :=
by sorry

end no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l199_199360


namespace smallest_positive_n_l199_199511

theorem smallest_positive_n (n : ℕ) (h : 19 * n ≡ 789 [MOD 11]) : n = 1 := 
by
  sorry

end smallest_positive_n_l199_199511


namespace max_correct_answers_l199_199535

variable (c w b : ℕ) -- Define c, w, b as natural numbers

theorem max_correct_answers (h1 : c + w + b = 30) (h2 : 4 * c - w = 70) : c ≤ 20 := by
  sorry

end max_correct_answers_l199_199535


namespace rectangle_perimeter_bounds_l199_199606

/-- Given 12 rectangular cardboard pieces, each measuring 4 cm in length and 3 cm in width,
  if these pieces are assembled to form a larger rectangle (possibly including squares),
  without overlapping or leaving gaps, then the minimum possible perimeter of the resulting 
  rectangle is 48 cm and the maximum possible perimeter is 102 cm. -/
theorem rectangle_perimeter_bounds (n : ℕ) (l w : ℝ) (total_area : ℝ) :
  n = 12 ∧ l = 4 ∧ w = 3 ∧ total_area = n * l * w →
  ∃ (min_perimeter max_perimeter : ℝ),
    min_perimeter = 48 ∧ max_perimeter = 102 :=
by
  intros
  sorry

end rectangle_perimeter_bounds_l199_199606


namespace empty_with_three_pumps_in_12_minutes_l199_199384

-- Define the conditions
def conditions (a b x : ℝ) : Prop :=
  x = a + b ∧ 2 * x = 3 * a + b

-- Define the main theorem to prove
theorem empty_with_three_pumps_in_12_minutes (a b x : ℝ) (h : conditions a b x) : 
  (3 * (1 / 5) * x = a + (1 / 5) * b) ∧ ((1 / 5) * 60 = 12) := 
by
  -- Use the given conditions in the proof.
  sorry

end empty_with_three_pumps_in_12_minutes_l199_199384


namespace meaningful_fraction_range_l199_199786

theorem meaningful_fraction_range (x : ℝ) : (3 - x) ≠ 0 ↔ x ≠ 3 :=
by sorry

end meaningful_fraction_range_l199_199786


namespace height_of_stack_correct_l199_199959

namespace PaperStack

-- Define the problem conditions
def sheets_per_package : ℕ := 500
def thickness_per_sheet_mm : ℝ := 0.1
def packages_per_stack : ℕ := 60
def mm_to_m : ℝ := 1000.0

-- Statement: the height of the stack of 60 paper packages
theorem height_of_stack_correct :
  (sheets_per_package * thickness_per_sheet_mm * packages_per_stack) / mm_to_m = 3 :=
sorry

end PaperStack

end height_of_stack_correct_l199_199959


namespace private_schools_in_district_B_l199_199451

theorem private_schools_in_district_B :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  remaining_private_schools = 4 :=
by
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_C_schools := total_schools - district_A_schools - district_B_schools
  let schools_per_kind_in_C := district_C_schools / 3
  let private_schools_in_C := schools_per_kind_in_C
  let remaining_private_schools := private_schools - private_schools_in_C
  sorry

end private_schools_in_district_B_l199_199451


namespace yogurt_banana_slices_l199_199707

/--
Given:
1. Each banana yields 10 slices.
2. Vivian needs to make 5 yogurts.
3. She needs to buy 4 bananas.

Prove:
The number of banana slices needed for each yogurt is 8.
-/
theorem yogurt_banana_slices 
    (slices_per_banana : ℕ)
    (bananas_bought : ℕ)
    (yogurts_needed : ℕ)
    (h1 : slices_per_banana = 10)
    (h2 : yogurts_needed = 5)
    (h3 : bananas_bought = 4) : 
    (bananas_bought * slices_per_banana) / yogurts_needed = 8 :=
by
  sorry

end yogurt_banana_slices_l199_199707


namespace solve_rational_eq_l199_199405

theorem solve_rational_eq {x : ℝ} (h : 1 / (x^2 + 9 * x - 12) + 1 / (x^2 + 4 * x - 5) + 1 / (x^2 - 15 * x - 12) = 0) :
  x = 3 ∨ x = -4 ∨ x = 1 ∨ x = -5 :=
by {
  sorry
}

end solve_rational_eq_l199_199405


namespace pipe_a_fills_cistern_l199_199509

theorem pipe_a_fills_cistern :
  ∀ (x : ℝ), (1 / x + 1 / 120 - 1 / 120 = 1 / 60) → x = 60 :=
by
  intro x
  intro h
  sorry

end pipe_a_fills_cistern_l199_199509


namespace total_settings_weight_l199_199472

/-- 
Each piece of silverware weighs 4 ounces and there are three pieces of silverware per setting.
Each plate weighs 12 ounces and there are two plates per setting.
Mason needs enough settings for 15 tables with 8 settings each, plus 20 backup settings in case of breakage.
Prove the total weight of all the settings equals 5040 ounces.
-/
theorem total_settings_weight
    (silverware_weight : ℝ := 4) (pieces_per_setting : ℕ := 3)
    (plate_weight : ℝ := 12) (plates_per_setting : ℕ := 2)
    (tables : ℕ := 15) (settings_per_table : ℕ := 8) (backup_settings : ℕ := 20) :
    let settings_needed := (tables * settings_per_table) + backup_settings
    let weight_per_setting := (silverware_weight * pieces_per_setting) + (plate_weight * plates_per_setting)
    settings_needed * weight_per_setting = 5040 :=
by
  sorry

end total_settings_weight_l199_199472


namespace count_special_three_digit_numbers_l199_199749

def is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000
def is_less_than_600 (n : ℕ) := n < 600
def has_at_least_two_same_digits (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 = d2 ∨ d2 = d3 ∨ d1 = d3

theorem count_special_three_digit_numbers :
  { n : ℕ | is_three_digit n ∧ is_less_than_600 n ∧ has_at_least_two_same_digits n }.to_finset.card = 140 :=
by
  sorry

end count_special_three_digit_numbers_l199_199749


namespace Zhu_Zaiyu_problem_l199_199074

theorem Zhu_Zaiyu_problem
  (f : ℕ → ℝ) 
  (q : ℝ)
  (h_geom_seq : ∀ n, f (n+1) = q * f n)
  (h_octave : f 13 = 2 * f 1) :
  (f 7) / (f 3) = 2^(1/3) :=
by
  sorry

end Zhu_Zaiyu_problem_l199_199074


namespace find_b_eq_five_l199_199013

/--
Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
and the condition that the distances from O (the origin) to B and from B to A are equal,
prove that b = 5.
-/
theorem find_b_eq_five : ∃ b : ℝ, (dist (0, 0) (0, b) = dist (0, b) (4, 2)) ∧ b = 5 :=
by
  sorry

end find_b_eq_five_l199_199013


namespace common_difference_d_l199_199653

theorem common_difference_d (a_1 d : ℝ) (h1 : a_1 + 2 * d = 4) (h2 : 9 * a_1 + 36 * d = 18) : d = -1 :=
by sorry

end common_difference_d_l199_199653


namespace fixed_point_on_line_AB_always_exists_l199_199229

-- Define the line where P lies
def line (x y : ℝ) : Prop := x + 2 * y = 4

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4

-- Define the point P
def moving_point_P (x y : ℝ) : Prop := line x y

-- Define the function that checks if a point is a tangent to the ellipse
def is_tangent (x0 y0 x y : ℝ) : Prop :=
  moving_point_P x0 y0 → (x * x0 + 4 * y * y0 = 4)

-- Statement: There exists a fixed point (1, 1/2) through which the line AB always passes
theorem fixed_point_on_line_AB_always_exists :
  ∀ (P A B : ℝ × ℝ),
    moving_point_P P.1 P.2 →
    is_tangent P.1 P.2 A.1 A.2 →
    is_tangent P.1 P.2 B.1 B.2 →
    ∃ (F : ℝ × ℝ), F = (1, 1/2) ∧ (F.1 - A.1) / (F.2 - A.2) = (F.1 - B.1) / (F.2 - B.2) :=
by
  sorry

end fixed_point_on_line_AB_always_exists_l199_199229


namespace solve_equation_l199_199155

theorem solve_equation:
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ 
    (x - y - x / y - (x^3 / y^3) + (x^4 / y^4) = 2017) ∧ 
    ((x = 2949 ∧ y = 983) ∨ (x = 4022 ∧ y = 2011)) :=
sorry

end solve_equation_l199_199155


namespace min_M_value_l199_199408

variable {a b c t : ℝ}

theorem min_M_value (h1 : a < b)
                    (h2 : a > 0)
                    (h3 : b^2 - 4 * a * c ≤ 0)
                    (h4 : b = t + a)
                    (h5 : t > 0)
                    (h6 : c ≥ (t + a)^2 / (4 * a)) :
    ∃ M : ℝ, (∀ x : ℝ, (a * x^2 + b * x + c) ≥ 0) → M = 3 := 
  sorry

end min_M_value_l199_199408


namespace range_of_a_minus_abs_b_l199_199942

theorem range_of_a_minus_abs_b (a b : ℝ) (h₁ : 1 < a ∧ a < 3) (h₂ : -4 < b ∧ b < 2) : 
  -3 < a - |b| ∧ a - |b| < 3 :=
sorry

end range_of_a_minus_abs_b_l199_199942


namespace fewer_people_third_bus_l199_199046

noncomputable def people_first_bus : Nat := 12
noncomputable def people_second_bus : Nat := 2 * people_first_bus
noncomputable def people_fourth_bus : Nat := people_first_bus + 9
noncomputable def total_people : Nat := 75
noncomputable def people_other_buses : Nat := people_first_bus + people_second_bus + people_fourth_bus
noncomputable def people_third_bus : Nat := total_people - people_other_buses

theorem fewer_people_third_bus :
  people_second_bus - people_third_bus = 6 :=
by
  sorry

end fewer_people_third_bus_l199_199046


namespace Anna_s_wear_size_l199_199203

theorem Anna_s_wear_size
  (A : ℕ)
  (Becky_size : ℕ)
  (Ginger_size : ℕ)
  (h1 : Becky_size = 3 * A)
  (h2 : Ginger_size = 2 * Becky_size - 4)
  (h3 : Ginger_size = 8) :
  A = 2 :=
by
  sorry

end Anna_s_wear_size_l199_199203


namespace simplified_sum_l199_199171

def exp1 := -( -1 ^ 2006 )
def exp2 := -( -1 ^ 2007 )
def exp3 := -( 1 ^ 2008 )
def exp4 := -( -1 ^ 2009 )

theorem simplified_sum : 
  exp1 + exp2 + exp3 + exp4 = 0 := 
by 
  sorry

end simplified_sum_l199_199171


namespace sarah_initial_money_l199_199490

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l199_199490


namespace matrix_power_50_l199_199020

def P : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-4, -3]
]

theorem matrix_power_50 :
  P ^ 50 = ![
    ![1, 0],
    ![0, 1]
  ] :=
sorry

end matrix_power_50_l199_199020


namespace score_order_l199_199965

-- Definitions that come from the problem conditions
variables (M Q S K : ℝ)
variables (hQK : Q = K) (hMK : M > K) (hSK : S < K)

-- The theorem to prove
theorem score_order (hQK : Q = K) (hMK : M > K) (hSK : S < K) : S < Q ∧ Q < M :=
by {
  sorry
}

end score_order_l199_199965


namespace correct_calculation_l199_199516

theorem correct_calculation :
  (∀ (x : ℝ), (x^3 * 2 * x^4 = 2 * x^7) ∧
  (x^6 / x^3 = x^2) ∧
  ((x^3)^4 = x^7) ∧
  (x^2 + x = x^3)) → 
  (∀ (x : ℝ), x^3 * 2 * x^4 = 2 * x^7) :=
by
  intros h x
  have A := h x
  exact A.1

end correct_calculation_l199_199516


namespace units_digit_of_expression_l199_199098

theorem units_digit_of_expression :
  (4 ^ 101 * 5 ^ 204 * 9 ^ 303 * 11 ^ 404) % 10 = 0 := 
sorry

end units_digit_of_expression_l199_199098


namespace questionnaires_drawn_from_unit_D_l199_199255

theorem questionnaires_drawn_from_unit_D 
  (arith_seq_collected : ∃ a1 d : ℕ, [a1, a1 + d, a1 + 2 * d, a1 + 3 * d] = [aA, aB, aC, aD] ∧ aA + aB + aC + aD = 1000)
  (stratified_sample : [30 - d, 30, 30 + d, 30 + 2 * d] = [sA, sB, sC, sD] ∧ sA + sB + sC + sD = 150)
  (B_drawn : 30 = sB) :
  sD = 60 := 
by {
  sorry
}

end questionnaires_drawn_from_unit_D_l199_199255


namespace find_some_number_l199_199125

-- Definitions of symbol replacements
def replacement_minus (a b : Nat) := a + b
def replacement_plus (a b : Nat) := a * b
def replacement_times (a b : Nat) := a / b
def replacement_div (a b : Nat) := a - b

-- The transformed equation using the replacements
def transformed_equation (some_number : Nat) :=
  replacement_minus
    some_number
    (replacement_div
      (replacement_plus 9 (replacement_times 8 3))
      25) = 5

theorem find_some_number : ∃ some_number : Nat, transformed_equation some_number ∧ some_number = 6 :=
by
  exists 6
  unfold transformed_equation
  unfold replacement_minus replacement_plus replacement_times replacement_div
  sorry

end find_some_number_l199_199125


namespace positive_difference_of_two_numbers_l199_199307

theorem positive_difference_of_two_numbers
  (x y : ℝ)
  (h₁ : x + y = 10)
  (h₂ : x^2 - y^2 = 24) :
  |x - y| = 12 / 5 :=
sorry

end positive_difference_of_two_numbers_l199_199307


namespace arithmetic_sequence_ratio_l199_199119

-- Definitions based on conditions
variables {a_n b_n : ℕ → ℕ} -- Arithmetic sequences
variables {A_n B_n : ℕ → ℕ} -- Sums of the first n terms

-- Given condition
axiom sums_of_arithmetic_sequences (n : ℕ) : A_n n / B_n n = (7 * n + 1) / (4 * n + 27)

-- Theorem to prove
theorem arithmetic_sequence_ratio :
  ∀ (a_n b_n : ℕ → ℕ) (A_n B_n : ℕ → ℕ), 
    (∀ n, A_n n / B_n n = (7 * n + 1) / (4 * n + 27)) → 
    a_6 / b_6 = 78 / 71 := 
by {
  sorry
}

end arithmetic_sequence_ratio_l199_199119


namespace equal_potatoes_l199_199964

theorem equal_potatoes (total_potatoes : ℕ) (total_people : ℕ) (h_potatoes : total_potatoes = 24) (h_people : total_people = 3) :
  (total_potatoes / total_people) = 8 :=
by {
  sorry
}

end equal_potatoes_l199_199964


namespace problem1_problem2_l199_199418

-- Define the permutations
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem1 (hnondefects : 6 ≥ 4)
    (hfirst : 10 ≥ 1) (hlast : 8 ≥ 4) :
    ((A 4 2) * (A 5 2) * (A 6 4)) = A 4 2 * A 5 2 * A 6 4 := sorry

theorem problem2 (hnondefects : 6 ≥ 4)
    (hfour : 10 ≥ 4) :
    ((A 4 4) + 4 * (A 4 3) * (A 6 1) + 4 * (A 5 3) * (A 6 2) + (A 6 6)) =
    A 4 4 + 4 * A 4 3 * A 6 1 + 4 * A 5 3 * A 6 2 + A 6 6 := sorry

end problem1_problem2_l199_199418


namespace probability_XOXOXOX_l199_199578

noncomputable def binomial (n k : ℕ) : ℕ := nat.choose n k

theorem probability_XOXOXOX :
  let num_X := 4,
      num_O := 3,
      total_arrangements := binomial 7 num_X in
  total_arrangements = 35 ∧
  1 / total_arrangements = (1 : ℚ) / 35 :=
by
  sorry

end probability_XOXOXOX_l199_199578


namespace probability_more_ones_than_sixes_l199_199767

open ProbabilityTheory

noncomputable def prob_more_ones_than_sixes : ℚ :=
  let total_outcomes := 6^5 in
  let favorable_cases := 679 in
  favorable_cases / total_outcomes

theorem probability_more_ones_than_sixes (h_dice_fair : ∀ (i : ℕ), i ∈ Finset.range 6 → ℙ (i = 1) = 1 / 6) :
  prob_more_ones_than_sixes = 679 / 1944 :=
by {
  -- placeholder for the actual proof
  sorry
}

end probability_more_ones_than_sixes_l199_199767


namespace find_q_l199_199238

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l199_199238


namespace henry_added_5_gallons_l199_199747

theorem henry_added_5_gallons (capacity : ℚ) (initial_fraction full_fraction added : ℚ) 
  (h_capacity : capacity = 40)
  (h_initial_fraction : initial_fraction = 3 / 4)
  (h_full_fraction : full_fraction = 7 / 8) :
  added = 5 :=
by
  have h_initial_volume : (initial_fraction * capacity) = 30 := by
    calc
      initial_fraction * capacity = (3 / 4) * 40 : by rw [h_initial_fraction, h_capacity]
      ...                        = 30           : by norm_num
  have h_full_volume : (full_fraction * capacity) = 35 := by
    calc
      full_fraction * capacity = (7 / 8) * 40 : by rw [h_full_fraction, h_capacity]
      ...                      = 35           : by norm_num
  have h_added : added = full_fraction * capacity - initial_fraction * capacity := by
    rw [h_full_fraction, h_initial_fraction, h_capacity]
  simp only [mul_sub, h_initial_volume, h_full_volume] at h_added
  have : added = 5 := by simp only [h_added]
  exact this

end henry_added_5_gallons_l199_199747


namespace num_integers_between_cubed_values_l199_199427

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end num_integers_between_cubed_values_l199_199427


namespace distance_from_highest_point_of_sphere_to_bottom_of_glass_l199_199298

theorem distance_from_highest_point_of_sphere_to_bottom_of_glass :
  ∀ (x y : ℝ),
  x^2 = 2 * y →
  0 ≤ y ∧ y < 15 →
  ∃ b : ℝ, (x^2 + (y - b)^2 = 9) ∧ b = 5 ∧ (b + 3 = 8) :=
by
  sorry

end distance_from_highest_point_of_sphere_to_bottom_of_glass_l199_199298


namespace calculation_correct_l199_199097

theorem calculation_correct : 2 * (3 ^ 2) ^ 4 = 13122 := by
  sorry

end calculation_correct_l199_199097


namespace gcd_calculation_l199_199167

theorem gcd_calculation : 
  Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := 
by
  sorry

end gcd_calculation_l199_199167


namespace part_a_part_b_part_c_l199_199447

-- Defining a structure for the problem
structure Rectangle :=
(area : ℝ)

structure Figure :=
(area : ℝ)

-- Defining the conditions
variables (R : Rectangle) 
  (F1 F2 F3 F4 F5 : Figure)
  (overlap_area_pair : Figure → Figure → ℝ)
  (overlap_area_triple : Figure → Figure → Figure → ℝ)

-- Given conditions
axiom R_area : R.area = 1
axiom F1_area : F1.area = 0.5
axiom F2_area : F2.area = 0.5
axiom F3_area : F3.area = 0.5
axiom F4_area : F4.area = 0.5
axiom F5_area : F5.area = 0.5

-- Statements to prove
theorem part_a : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 3 / 20 := sorry
theorem part_b : ∃ (F1 F2 : Figure), overlap_area_pair F1 F2 ≥ 1 / 5 := sorry
theorem part_c : ∃ (F1 F2 F3 : Figure), overlap_area_triple F1 F2 F3 ≥ 1 / 20 := sorry

end part_a_part_b_part_c_l199_199447


namespace betty_blue_beads_l199_199693

theorem betty_blue_beads (r b : ℕ) (h1 : r = 30) (h2 : 3 * b = 2 * r) : b = 20 :=
by
  sorry

end betty_blue_beads_l199_199693


namespace find_B_in_product_l199_199220

theorem find_B_in_product (B : ℕ) (hB : B < 10) (h : (B * 100 + 2) * (900 + B) = 8016) : B = 8 := by
  sorry

end find_B_in_product_l199_199220


namespace jacob_total_distance_l199_199620

/- Jacob jogs at a constant rate of 4 miles per hour.
   He jogs for 2 hours, then stops to take a rest for 30 minutes.
   After the break, he continues jogging for another 1 hour.
   Prove that the total distance jogged by Jacob is 12.0 miles.
-/
theorem jacob_total_distance :
  let joggingSpeed := 4 -- in miles per hour
  let jogBeforeBreak := 2 -- in hours
  let restDuration := 0.5 -- in hours (though it does not affect the distance)
  let jogAfterBreak := 1 -- in hours
  let totalDistance := joggingSpeed * jogBeforeBreak + joggingSpeed * jogAfterBreak
  totalDistance = 12.0 := 
by
  sorry

end jacob_total_distance_l199_199620


namespace Democrats_in_House_l199_199049

-- Let D be the number of Democrats.
-- Let R be the number of Republicans.
-- Given conditions.

def Democrats (D R : ℕ) : Prop := 
  D + R = 434 ∧ R = D + 30

theorem Democrats_in_House : ∃ D, ∃ R, Democrats D R ∧ D = 202 :=
by
  -- skip the proof
  sorry

end Democrats_in_House_l199_199049


namespace roots_not_in_interval_l199_199271

theorem roots_not_in_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2 * a) → (x < -1 ∨ x > 1) :=
by
  sorry

end roots_not_in_interval_l199_199271


namespace intersection_eq_l199_199719

def M : Set (ℝ × ℝ) := { p | ∃ x, p.2 = x^2 }
def N : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 2 }
def Intersect : Set (ℝ × ℝ) := { p | (M p) ∧ (N p)}

theorem intersection_eq : Intersect = { p : ℝ × ℝ | p = (1,1) ∨ p = (-1, 1) } :=
  sorry

end intersection_eq_l199_199719


namespace star_addition_l199_199432

-- Definition of the binary operation "star"
def star (x y : ℤ) := 5 * x - 2 * y

-- Statement of the problem
theorem star_addition : star 3 4 + star 2 2 = 13 :=
by
  -- By calculation, we have:
  -- star 3 4 = 7 and star 2 2 = 6
  -- Thus, star 3 4 + star 2 2 = 7 + 6 = 13
  sorry

end star_addition_l199_199432


namespace trips_Jean_l199_199544

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l199_199544


namespace perfect_square_trinomial_l199_199323

theorem perfect_square_trinomial :
  15^2 + 2 * 15 * 3 + 3^2 = 324 := 
by
  sorry

end perfect_square_trinomial_l199_199323


namespace number_of_valid_permutations_l199_199625

-- Definition of the problem in terms of Lean
def is_valid_permutation (p : List ℕ) : Prop :=
p.perm  (List.range 5).map (· + 1) ∧
∀ i j k : ℕ, i < j → j < k → k < 5 → p.get? i < p.get? j → p.get? j < p.get? k → False

-- Problem statement in Lean
theorem number_of_valid_permutations : 
   (Finset.univ.filter is_valid_permutation).card = 42 :=
sorry

end number_of_valid_permutations_l199_199625


namespace complex_number_equality_l199_199941

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem complex_number_equality (a b : ℝ) (i : ℂ) (h1 : is_imaginary_unit i) (h2 : (a + 4 * i) * i = b + i) : a + b = -3 :=
sorry

end complex_number_equality_l199_199941


namespace average_of_remaining_six_is_correct_l199_199854

noncomputable def average_of_remaining_six (s20 s14: ℕ) (avg20 avg14: ℚ) : ℚ :=
  let sum20 := s20 * avg20
  let sum14 := s14 * avg14
  let sum_remaining := sum20 - sum14
  (sum_remaining / (s20 - s14))

theorem average_of_remaining_six_is_correct : 
  average_of_remaining_six 20 14 500 390 = 756.67 :=
by 
  sorry

end average_of_remaining_six_is_correct_l199_199854


namespace angles_terminal_side_equiv_l199_199909

theorem angles_terminal_side_equiv (k : ℤ) : (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi + Real.pi) % (2 * Real.pi) ∨ (2 * k * Real.pi + Real.pi) % (2 * Real.pi) = (4 * k * Real.pi - Real.pi) % (2 * Real.pi) :=
sorry

end angles_terminal_side_equiv_l199_199909


namespace correct_operation_l199_199072

theorem correct_operation : -5 * 3 = -15 :=
by sorry

end correct_operation_l199_199072


namespace gcd_16_12_eq_4_l199_199564

theorem gcd_16_12_eq_4 : Int.gcd 16 12 = 4 := by
  sorry

end gcd_16_12_eq_4_l199_199564


namespace intersection_A_B_l199_199001

-- Define set A based on the given condition.
def setA : Set ℝ := {x | x^2 - 4 < 0}

-- Define set B based on the given condition.
def setB : Set ℝ := {x | x < 0}

-- Prove that the intersection of sets A and B is the given set.
theorem intersection_A_B : setA ∩ setB = {x | -2 < x ∧ x < 0} := by
  sorry

end intersection_A_B_l199_199001


namespace increasing_sequences_mod_1000_l199_199547

theorem increasing_sequences_mod_1000 :
  (nat.choose 680 12) % 1000 = 680 :=
by sorry

end increasing_sequences_mod_1000_l199_199547


namespace polynomial_properties_l199_199634

noncomputable def polynomial : Polynomial ℚ :=
  -3/8 * (Polynomial.X ^ 5) + 5/4 * (Polynomial.X ^ 3) - 15/8 * (Polynomial.X)

theorem polynomial_properties (f : Polynomial ℚ) :
  (Polynomial.degree f = 5) ∧
  (∃ q : Polynomial ℚ, f + 1 = Polynomial.X - 1 ^ 3 * q) ∧
  (∃ p : Polynomial ℚ, f - 1 = Polynomial.X + 1 ^ 3 * p) ↔
  f = polynomial :=
by sorry

end polynomial_properties_l199_199634


namespace least_number_to_subtract_l199_199339

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (r : ℕ) (h1 : n = 42398) (h2 : d = 15) (h3 : r = 8) : 
  ∃ k, n - r = k * d :=
by
  sorry

end least_number_to_subtract_l199_199339


namespace percentage_2x_minus_y_of_x_l199_199439

noncomputable def x_perc_of_2x_minus_y (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) : ℤ :=
  (2 * x - y) * 100 / x

theorem percentage_2x_minus_y_of_x (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) :
  x_perc_of_2x_minus_y x y z h1 h2 h3 h4 = 175 :=
  sorry

end percentage_2x_minus_y_of_x_l199_199439


namespace yard_length_l199_199011

theorem yard_length
  (trees : ℕ) (gaps : ℕ) (distance_between_trees : ℕ) :
  trees = 26 → 
  gaps = trees - 1 → 
  distance_between_trees = 14 → 
  length_of_yard = gaps * distance_between_trees → 
  length_of_yard = 350 :=
by
  intros h_trees h_gaps h_distance h_length
  sorry

end yard_length_l199_199011


namespace count_three_digit_integers_with_two_same_digits_l199_199750

def digits (n : ℕ) : List ℕ := [n / 100, (n / 10) % 10, n % 10]

lemma digits_len (n : ℕ) (h : n < 1000) (h1 : n ≥ 100) : (digits n).length = 3 :=
begin 
    sorry
end

def at_least_two_same (n : ℕ) : Prop :=
  let d := digits n in
  d.length = 3 ∧ (d[0] = d[1] ∨ d[1] = d[2] ∨ d[0] = d[2])

theorem count_three_digit_integers_with_two_same_digits :
  (finset.filter at_least_two_same (finset.Icc 100 599)).card = 140 :=
by sorry

end count_three_digit_integers_with_two_same_digits_l199_199750


namespace number_of_valid_triangles_eq_95_l199_199277

theorem number_of_valid_triangles_eq_95 :
  let T := { (a, b, c) // 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 
                    ∧ a + b > c ∧ b + c > a ∧ a + c > b } in
  T.card = 95 :=
by
  sorry

end number_of_valid_triangles_eq_95_l199_199277


namespace arithmetic_sequence_sum_l199_199128

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 8 = 8)
  (h2 : ∀ n, S n = n * (a 1 + a n) / 2) (h3 : a 1 + a 15 = 2 * a 8) :
  S 15 = 120 := sorry

end arithmetic_sequence_sum_l199_199128


namespace pepperoni_slices_left_l199_199980

theorem pepperoni_slices_left :
  ∀ (total_friends : ℕ) (total_slices : ℕ) (cheese_left : ℕ),
    (total_friends = 4) →
    (total_slices = 16) →
    (cheese_left = 7) →
    (∃ p_slices_left : ℕ, p_slices_left = 4) :=
by
  intros total_friends total_slices cheese_left h_friends h_slices h_cheese
  sorry

end pepperoni_slices_left_l199_199980


namespace probability_prime_multiple_assignment_l199_199878

theorem probability_prime_multiple_assignment :
  let numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let primes : Finset ℕ := {2, 3, 5}
  let valid_assignments :=
        { (a, b, c) | a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧
          a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
          ∃ p1 ∈ primes, p1 * b = a ∧
          ∃ p2 ∈ primes, p2 * c = b }

  (valid_assignments.card : ℚ) / (numbers.card * (numbers.card - 1) * (numbers.card - 2)) = 1 / 40 :=
by
  sorry

end probability_prime_multiple_assignment_l199_199878


namespace blue_beads_count_l199_199691

-- Define variables and conditions
variables (r b : ℕ)

-- Define the conditions
def condition1 : Prop := r = 30
def condition2 : Prop := r / 3 = b / 2

-- State the theorem
theorem blue_beads_count (h1 : condition1 r) (h2 : condition2 r b) : b = 20 :=
sorry

end blue_beads_count_l199_199691


namespace sheets_in_stack_l199_199089

theorem sheets_in_stack (thickness_per_500_sheets : ℝ) (stack_height : ℝ) (total_sheets : ℕ) :
  thickness_per_500_sheets = 4 → stack_height = 10 → total_sheets = 1250 :=
by
  intros h1 h2
  -- We will provide the mathematical proof steps here.
  sorry

end sheets_in_stack_l199_199089


namespace work_left_fraction_l199_199674

theorem work_left_fraction (A_days B_days total_days : ℕ) (hA : A_days = 15) (hB : B_days = 20) (htotal : total_days = 4) :
  let A_work_per_day := (1 : ℚ) / A_days,
      B_work_per_day := (1 : ℚ) / B_days,
      combined_work_per_day := A_work_per_day + B_work_per_day,
      work_done := combined_work_per_day * total_days,
      work_left := 1 - work_done in
  work_left = 8 / 15 := 
by 
  sorry

end work_left_fraction_l199_199674


namespace common_roots_of_cubic_polynomials_l199_199566

/-- The polynomials \( x^3 + 6x^2 + 11x + 6 \) and \( x^3 + 7x^2 + 14x + 8 \) have two distinct roots in common. -/
theorem common_roots_of_cubic_polynomials :
  ∃ r s : ℝ, r ≠ s ∧ (r^3 + 6 * r^2 + 11 * r + 6 = 0) ∧ (s^3 + 6 * s^2 + 11 * s + 6 = 0)
  ∧ (r^3 + 7 * r^2 + 14 * r + 8 = 0) ∧ (s^3 + 7 * s^2 + 14 * s + 8 = 0) :=
sorry

end common_roots_of_cubic_polynomials_l199_199566


namespace football_game_wristbands_l199_199258

theorem football_game_wristbands (total_wristbands wristbands_per_person : Nat) (h1 : total_wristbands = 290) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 145 :=
by
  sorry

end football_game_wristbands_l199_199258


namespace triangle_count_l199_199276

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def P : ℕ :=
  { n : ℕ // ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ valid_triangle a b c ∧ n = a + b + c }.to_finset.card

theorem triangle_count :
  P = 95 :=
sorry

end triangle_count_l199_199276


namespace first_team_odd_is_correct_l199_199261

noncomputable def odd_for_first_team : Real := 
  let odd2 := 5.23
  let odd3 := 3.25
  let odd4 := 2.05
  let bet_amount := 5.00
  let expected_win := 223.0072
  let total_odds := expected_win / bet_amount
  let denominator := odd2 * odd3 * odd4
  total_odds / denominator

theorem first_team_odd_is_correct : 
  odd_for_first_team = 1.28 := by 
  sorry

end first_team_odd_is_correct_l199_199261


namespace alpha_plus_beta_eq_l199_199582

variable {α β : ℝ}
variable (h1 : 0 < α ∧ α < π)
variable (h2 : 0 < β ∧ β < π)
variable (h3 : Real.sin (α - β) = 5 / 6)
variable (h4 : Real.tan α / Real.tan β = -1 / 4)

theorem alpha_plus_beta_eq : α + β = 7 * Real.pi / 6 := by
  sorry

end alpha_plus_beta_eq_l199_199582


namespace intersection_eq_expected_l199_199228

def setA := { x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def setB := { x : ℝ | 1 ≤ x ∧ x < 4 }
def expectedSet := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq_expected :
  {x : ℝ | x ∈ setA ∧ x ∈ setB} = expectedSet :=
by
  sorry

end intersection_eq_expected_l199_199228


namespace portia_high_school_students_l199_199026

theorem portia_high_school_students
  (L P M : ℕ)
  (h1 : P = 4 * L)
  (h2 : M = 2 * L)
  (h3 : P + L + M = 4200) :
  P = 2400 :=
sorry

end portia_high_school_students_l199_199026


namespace problem1_l199_199185

theorem problem1 (m n : ℕ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m + 4 * n) = 324) : 2016 ^ n = 2016 := 
by 
  sorry

end problem1_l199_199185


namespace sales_tax_percentage_l199_199865

theorem sales_tax_percentage 
  (total_spent : ℝ)
  (tip_percent : ℝ)
  (food_price : ℝ) 
  (total_with_tip : total_spent = food_price * (1 + tip_percent / 100))
  (sales_tax_percent : ℝ) 
  (total_paid : total_spent = food_price * (1 + sales_tax_percent / 100) * (1 + tip_percent / 100)) :
  sales_tax_percent = 10 :=
by sorry

end sales_tax_percentage_l199_199865


namespace solve_n_m_equation_l199_199987

theorem solve_n_m_equation : 
  ∃ (n m : ℤ), n^4 - 2*n^2 = m^2 + 38 ∧ ((n, m) = (3, 5) ∨ (n, m) = (3, -5) ∨ (n, m) = (-3, 5) ∨ (n, m) = (-3, -5)) :=
by { sorry }

end solve_n_m_equation_l199_199987


namespace ball_total_distance_l199_199361

def total_distance (initial_height : ℝ) (bounce_factor : ℝ) (bounces : ℕ) : ℝ :=
  let rec loop (height : ℝ) (total : ℝ) (remaining : ℕ) : ℝ :=
    if remaining = 0 then total
    else loop (height * bounce_factor) (total + height + height * bounce_factor) (remaining - 1)
  loop initial_height 0 bounces

theorem ball_total_distance : 
  total_distance 20 0.8 4 = 106.272 :=
by
  sorry

end ball_total_distance_l199_199361


namespace area_of_smallest_square_containing_circle_l199_199839

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ s, s = 14 ∧ s * s = 196 :=
by
  sorry

end area_of_smallest_square_containing_circle_l199_199839


namespace simplify_and_evaluate_expression_l199_199985

theorem simplify_and_evaluate_expression (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) : 
  (x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = (x + 2) / (x - 2) :=
by
  sorry

example : (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ (x = 3) ∧ ((x + 1 - 3 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1)) = 5)) :=
  ⟨3, by norm_num, by norm_num, rfl, by norm_num⟩

end simplify_and_evaluate_expression_l199_199985


namespace find_q_l199_199735

theorem find_q (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199735


namespace train_length_l199_199527

theorem train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (distance_covered : ℝ)
  (train_length : ℝ) :
  speed_kmph = 72 →
  platform_length = 240 →
  crossing_time = 26 →
  conversion_factor = 5 / 18 →
  speed_mps = speed_kmph * conversion_factor →
  distance_covered = speed_mps * crossing_time →
  train_length = distance_covered - platform_length →
  train_length = 280 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end train_length_l199_199527


namespace tim_total_money_raised_l199_199835

-- Definitions based on conditions
def maxDonation : ℤ := 1200
def numMaxDonors : ℤ := 500
def numHalfDonors : ℤ := 3 * numMaxDonors
def halfDonation : ℤ := maxDonation / 2
def totalPercent : ℚ := 0.4

def totalDonationFromMaxDonors : ℤ := numMaxDonors * maxDonation
def totalDonationFromHalfDonors : ℤ := numHalfDonors * halfDonation
def totalDonation : ℤ := totalDonationFromMaxDonors + totalDonationFromHalfDonors

-- Proposition that Tim's total money raised is $3,750,000
theorem tim_total_money_raised : (totalDonation : ℚ) / totalPercent = 3750000 := by
  -- Verified in the proof steps
  sorry

end tim_total_money_raised_l199_199835


namespace find_incorrect_observation_l199_199042

theorem find_incorrect_observation (n : ℕ) (initial_mean new_mean : ℝ) (correct_value incorrect_value : ℝ) (observations_count : ℕ)
  (h1 : observations_count = 50)
  (h2 : initial_mean = 36)
  (h3 : new_mean = 36.5)
  (h4 : correct_value = 44) :
  incorrect_value = 19 :=
by
  sorry

end find_incorrect_observation_l199_199042


namespace find_c_d_l199_199022

noncomputable def g (c d x : ℝ) : ℝ := c * x^3 + 5 * x^2 + d * x + 7

theorem find_c_d : ∃ (c d : ℝ), 
  (g c d 2 = 11) ∧ (g c d (-3) = 134) ∧ c = -35 / 13 ∧ d = 16 / 13 :=
  by
  sorry

end find_c_d_l199_199022


namespace travel_distance_l199_199864

-- Define the average speed of the car
def speed : ℕ := 68

-- Define the duration of the trip in hours
def time : ℕ := 12

-- Define the distance formula for constant speed
def distance (speed time : ℕ) : ℕ := speed * time

-- Proof statement
theorem travel_distance : distance speed time = 756 := by
  -- Provide a placeholder for the proof
  sorry

end travel_distance_l199_199864


namespace square_of_binomial_l199_199431

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b)^2 = 9 * x^2 + 30 * x + a) → a = 25 :=
by
  sorry

end square_of_binomial_l199_199431


namespace stella_spent_amount_l199_199292

-- Definitions
def num_dolls : ℕ := 3
def num_clocks : ℕ := 2
def num_glasses : ℕ := 5

def price_doll : ℕ := 5
def price_clock : ℕ := 15
def price_glass : ℕ := 4

def profit : ℕ := 25

-- Calculation of total revenue from profit
def total_revenue : ℕ := num_dolls * price_doll + num_clocks * price_clock + num_glasses * price_glass

-- Proposition to be proved
theorem stella_spent_amount : total_revenue - profit = 40 :=
by sorry

end stella_spent_amount_l199_199292


namespace coordinates_of_point_l199_199785

noncomputable def point_on_x_axis (x : ℝ) :=
  (x, 0)

theorem coordinates_of_point (x : ℝ) (hx : abs x = 3) :
  point_on_x_axis x = (3, 0) ∨ point_on_x_axis x = (-3, 0) :=
  sorry

end coordinates_of_point_l199_199785


namespace apples_in_box_at_first_l199_199672

noncomputable def initial_apples (X : ℕ) : Prop :=
  (X / 2 - 25 = 6)

theorem apples_in_box_at_first (X : ℕ) : initial_apples X ↔ X = 62 :=
by
  sorry

end apples_in_box_at_first_l199_199672


namespace union_sets_l199_199118

open Set

/-- Given sets A and B defined as follows:
    A = {x | -1 ≤ x ∧ x ≤ 2}
    B = {x | x ≤ 4}
    Prove that A ∪ B = {x | x ≤ 4}
--/
theorem union_sets  :
    let A := {x | -1 ≤ x ∧ x ≤ 2}
    let B := {x | x ≤ 4}
    A ∪ B = {x | x ≤ 4} :=
by
    intros A B
    have : A = {x | -1 ≤ x ∧ x ≤ 2} := rfl
    have : B = {x | x ≤ 4} := rfl
    sorry

end union_sets_l199_199118


namespace bike_ride_distance_l199_199967

-- Definitions for conditions from a)
def speed_out := 24 -- miles per hour
def speed_back := 18 -- miles per hour
def total_time := 7 -- hours

-- Problem statement for the proof problem
theorem bike_ride_distance :
  ∃ (D : ℝ), (D / speed_out) + (D / speed_back) = total_time ∧ 2 * D = 144 :=
by {
  sorry
}

end bike_ride_distance_l199_199967


namespace geometry_problem_l199_199224

/-- Given:
  DC = 5
  CB = 9
  AB = 1/3 * AD
  ED = 2/3 * AD
  Prove: FC = 10.6667 -/
theorem geometry_problem
  (DC CB AD FC : ℝ) (hDC : DC = 5) (hCB : CB = 9) (hAB : AB = 1 / 3 * AD) (hED : ED = 2 / 3 * AD)
  (AB ED: ℝ):
  FC = 10.6667 :=
by
  sorry

end geometry_problem_l199_199224


namespace common_ratio_geometric_sequence_l199_199798

theorem common_ratio_geometric_sequence 
  (a1 : ℝ) 
  (q : ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S n = a1 * (1 - q^n) / (1 - q)) 
  (h2 : 8 * S 6 = 7 * S 3) 
  (hq : q ≠ 1) : 
  q = -1 / 2 := 
sorry

end common_ratio_geometric_sequence_l199_199798


namespace no_such_function_exists_l199_199626

open Set

theorem no_such_function_exists
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → y > x → f y > (y - x) * f x ^ 2) :
  False :=
sorry

end no_such_function_exists_l199_199626


namespace binomial_square_evaluation_l199_199334

theorem binomial_square_evaluation : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end binomial_square_evaluation_l199_199334


namespace problem1_problem2_l199_199099

-- Proof Problem 1:

theorem problem1 : (5 / 3) ^ 2004 * (3 / 5) ^ 2003 = 5 / 3 := by
  sorry

-- Proof Problem 2:

theorem problem2 (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end problem1_problem2_l199_199099


namespace find_g_30_l199_199996

def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) = x * g y

axiom g_one : g 1 = 10

theorem find_g_30 : g 30 = 300 := by
  sorry

end find_g_30_l199_199996


namespace equation_solution_l199_199289

noncomputable def solve_equation (x : ℝ) : Prop :=
  (4 / (x - 1) + 1 / (1 - x) = 1) → x = 4

theorem equation_solution (x : ℝ) (h : 4 / (x - 1) + 1 / (1 - x) = 1) : x = 4 := by
  sorry

end equation_solution_l199_199289


namespace two_digits_same_in_three_digit_numbers_l199_199755

theorem two_digits_same_in_three_digit_numbers (h1 : (100 : ℕ) ≤ n) (h2 : n < 600) : 
  ∃ n, n = 140 := sorry

end two_digits_same_in_three_digit_numbers_l199_199755


namespace inequality_proof_l199_199922

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l199_199922


namespace kath_movie_cost_l199_199958

theorem kath_movie_cost :
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  discounted_price * number_of_people = 30 := by
  -- Definitions from conditions
  let standard_admission := 8
  let discount := 3
  let movie_start_time := 4
  let before_six_pm := movie_start_time < 6
  let discounted_price := standard_admission - discount
  let number_of_people := 1 + 2 + 3
  -- Derived calculation based on conditions
  have h_discounted_price : discounted_price = 5 := by
    calc
      discounted_price = 8 - 3 : by sorry
      ... = 5 : by sorry
  have h_number_of_people : number_of_people = 6 := by
    calc
      number_of_people = 1 + 2 + 3 : by sorry
      ... = 6 : by sorry
  show 5 * 6 = 30 from sorry

end kath_movie_cost_l199_199958


namespace equilateral_triangle_l199_199262

theorem equilateral_triangle (a b c : ℝ) (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c ∧ a = c := 
by
  sorry

end equilateral_triangle_l199_199262


namespace find_range_of_m_l199_199714

noncomputable def quadratic_equation := 
  ∀ (m : ℝ), 
  ∃ x y : ℝ, 
  (m + 3) * x^2 - 4 * m * x + (2 * m - 1) = 0 ∧ 
  (m + 3) * y^2 - 4 * m * y + (2 * m - 1) = 0 ∧ 
  x * y < 0 ∧ 
  |x| > |y| ∧ 
  m ∈ Set.Ioo (-3:ℝ) (0:ℝ)

theorem find_range_of_m : quadratic_equation := 
by
  sorry

end find_range_of_m_l199_199714


namespace point_coordinates_l199_199782

-- Definitions based on conditions
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0
def dist_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := abs P.1 = d

-- Lean 4 statement
theorem point_coordinates {P : ℝ × ℝ} (h1 : on_x_axis P) (h2 : dist_to_y_axis P 3) :
  P = (3, 0) ∨ P = (-3, 0) :=
by sorry

end point_coordinates_l199_199782


namespace calculate_sum_of_powers_l199_199052

theorem calculate_sum_of_powers :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 :=
by
  sorry

end calculate_sum_of_powers_l199_199052


namespace leila_total_payment_l199_199376

theorem leila_total_payment:
  (choc_cost : ℕ) (choc_quantity : ℕ) (straw_cost : ℕ) (straw_quantity : ℕ)
  (h_choc : choc_cost = 12) (h_choc_qty : choc_quantity = 3)
  (h_straw : straw_cost = 22) (h_straw_qty : straw_quantity = 6) :
  choc_cost * choc_quantity + straw_cost * straw_quantity = 168 := 
by
  sorry

end leila_total_payment_l199_199376


namespace waiter_tips_earned_l199_199857

theorem waiter_tips_earned (total_customers tips_left no_tip_customers tips_per_customer : ℕ) :
  no_tip_customers + tips_left = total_customers ∧ tips_per_customer = 3 ∧ no_tip_customers = 5 ∧ total_customers = 7 → 
  tips_left * tips_per_customer = 6 :=
by
  intro h
  sorry

end waiter_tips_earned_l199_199857


namespace interest_difference_l199_199299

noncomputable def principal := 63100
noncomputable def rate := 10 / 100
noncomputable def time := 2

noncomputable def simple_interest := principal * rate * time
noncomputable def compound_interest := principal * (1 + rate)^time - principal

theorem interest_difference :
  (compound_interest - simple_interest) = 671 := by
  sorry

end interest_difference_l199_199299


namespace eval_fraction_power_l199_199213

theorem eval_fraction_power : (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end eval_fraction_power_l199_199213


namespace original_combined_price_l199_199568

theorem original_combined_price (C S : ℝ) 
  (candy_box_increased : C * 1.25 = 18.75)
  (soda_can_increased : S * 1.50 = 9) : 
  C + S = 21 :=
by
  sorry

end original_combined_price_l199_199568


namespace total_votes_l199_199014

theorem total_votes (total_votes : ℕ) (brenda_votes : ℕ) (fraction : ℚ) (h : brenda_votes = fraction * total_votes) (h_fraction : fraction = 1 / 5) (h_brenda : brenda_votes = 15) : 
  total_votes = 75 := 
by
  sorry

end total_votes_l199_199014


namespace minimize_expr_l199_199141

-- Define the problem conditions
variables (a b c : ℝ)
variables (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variables (h4 : a * b * c = 8)

-- Define the target expression and the proof goal
def expr := (3 * a + b) * (a + 3 * c) * (2 * b * c + 4)

-- Prove the main statement
theorem minimize_expr : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c = 8) ∧ expr a b c = 384 :=
sorry

end minimize_expr_l199_199141


namespace uma_income_is_20000_l199_199650

/-- Given that the ratio of the incomes of Uma and Bala is 4 : 3, 
the ratio of their expenditures is 3 : 2, and both save $5000 at the end of the year, 
prove that Uma's income is $20000. -/
def uma_bala_income : Prop :=
  ∃ (x y : ℕ), (4 * x - 3 * y = 5000) ∧ (3 * x - 2 * y = 5000) ∧ (4 * x = 20000)
  
theorem uma_income_is_20000 : uma_bala_income :=
  sorry

end uma_income_is_20000_l199_199650


namespace find_x_values_l199_199404

theorem find_x_values (
  x : ℝ
) (h₁ : x ≠ 0) (h₂ : x ≠ 1) (h₃ : x ≠ 2) :
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 4) ↔ 
  (x < (1 - Real.sqrt 17) / 2 ∨ (0 < x ∧ x < 1) ∨ (2 < x ∧ x < (1 + Real.sqrt 17) / 2)) :=
by
  sorry

end find_x_values_l199_199404


namespace least_positive_integer_solution_l199_199664

theorem least_positive_integer_solution :
  ∃ x : ℤ, x > 0 ∧ ∃ n : ℤ, (3 * x + 29)^2 = 43 * n ∧ x = 19 :=
by
  sorry

end least_positive_integer_solution_l199_199664


namespace trips_Jean_l199_199546

theorem trips_Jean (x : ℕ) (h1 : x + (x + 6) = 40) : x + 6 = 23 := by
  sorry

end trips_Jean_l199_199546


namespace range_of_3x_plus_2y_l199_199410

theorem range_of_3x_plus_2y (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : -1 ≤ y ∧ y ≤ 4) :
  1 ≤ 3 * x + 2 * y ∧ 3 * x + 2 * y ≤ 17 :=
sorry

end range_of_3x_plus_2y_l199_199410


namespace green_light_probability_l199_199096

-- Define the durations of the red, green, and yellow lights
def red_light_duration : ℕ := 30
def green_light_duration : ℕ := 25
def yellow_light_duration : ℕ := 5

-- Define the total cycle time
def total_cycle_time : ℕ := red_light_duration + green_light_duration + yellow_light_duration

-- Define the expected probability
def expected_probability : ℚ := 5 / 12

-- Prove the probability of seeing a green light equals the expected_probability
theorem green_light_probability :
  (green_light_duration : ℚ) / (total_cycle_time : ℚ) = expected_probability :=
by
  sorry

end green_light_probability_l199_199096


namespace oil_spending_l199_199534

-- Define the original price per kg of oil
def original_price (P : ℝ) := P

-- Define the reduced price per kg of oil
def reduced_price (P : ℝ) := 0.75 * P

-- Define the reduced price as Rs. 60
def reduced_price_fixed := 60

-- State the condition that reduced price enables 5 kgs more oil
def extra_kg := 5

-- The amount of money spent by housewife at reduced price which is to be proven as Rs. 1200
def amount_spent (M : ℝ) := M

-- Define the problem to prove in Lean 4
theorem oil_spending (P X : ℝ) (h1 : reduced_price P = reduced_price_fixed) (h2 : X * original_price P = (X + extra_kg) * reduced_price_fixed) : amount_spent ((X + extra_kg) * reduced_price_fixed) = 1200 :=
  sorry

end oil_spending_l199_199534


namespace mean_of_six_numbers_sum_three_quarters_l199_199502

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l199_199502


namespace find_q_l199_199239

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l199_199239


namespace building_height_l199_199191

theorem building_height (h : ℕ) 
  (flagpole_height flagpole_shadow building_shadow : ℕ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70) 
  (condition : flagpole_height / flagpole_shadow = h / building_shadow) :
  h = 28 := by
  sorry

end building_height_l199_199191


namespace mean_of_six_numbers_sum_three_quarters_l199_199500

theorem mean_of_six_numbers_sum_three_quarters :
  let sum := (3 / 4 : ℝ)
  let n := 6
  (sum / n) = (1 / 8 : ℝ) :=
by
  sorry

end mean_of_six_numbers_sum_three_quarters_l199_199500


namespace find_star_l199_199644

-- Define the problem conditions and statement
theorem find_star (x : ℤ) (star : ℤ) (h1 : x = 5) (h2 : -3 * (star - 9) = 5 * x - 1) : star = 1 :=
by
  sorry -- Proof to be filled in

end find_star_l199_199644


namespace probability_more_ones_than_sixes_l199_199764

theorem probability_more_ones_than_sixes (total_dice : ℕ) (sides_of_dice : ℕ) 
  (ones : ℕ) (sixes : ℕ) (total_outcomes : ℕ) (equal_outcomes : ℕ) : 
  (total_dice = 5) → 
  (sides_of_dice = 6) → 
  (total_outcomes = 6^total_dice) → 
  (equal_outcomes = 1024 + 1280 + 120) → 
  (ones > sixes) → 
  (prob_more_ones_than_sixes : ℚ) → 
  prob_more_ones_than_sixes = (1/2) * (1 - (equal_outcomes / total_outcomes)) := 
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4],
  sorry,
end

end probability_more_ones_than_sixes_l199_199764


namespace laura_owes_correct_amount_l199_199138

def principal : ℝ := 35
def annual_rate : ℝ := 0.07
def time_years : ℝ := 1
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T
def total_amount_owed (P : ℝ) (I : ℝ) : ℝ := P + I

theorem laura_owes_correct_amount :
  total_amount_owed principal (interest principal annual_rate time_years) = 37.45 :=
sorry

end laura_owes_correct_amount_l199_199138


namespace probability_black_ball_l199_199789

variable (total_balls : ℕ)
variable (red_balls : ℕ)
variable (white_probability : ℝ)

def number_of_balls : Prop := total_balls = 100
def red_ball_count : Prop := red_balls = 45
def white_ball_probability : Prop := white_probability = 0.23

theorem probability_black_ball 
  (h1 : number_of_balls total_balls)
  (h2 : red_ball_count red_balls)
  (h3 : white_ball_probability white_probability) :
  let white_balls := white_probability * total_balls 
  let black_balls := total_balls - red_balls - white_balls
  let black_ball_prob := black_balls / total_balls
  black_ball_prob = 0.32 :=
sorry

end probability_black_ball_l199_199789


namespace number_added_to_x_l199_199567

theorem number_added_to_x (x : ℕ) (some_number : ℕ) (h1 : x = 3) (h2 : x + some_number = 4) : some_number = 1 := 
by
  -- Given hypotheses can be used here
  sorry

end number_added_to_x_l199_199567


namespace zero_point_interval_l199_199309

noncomputable def f (x : ℝ) : ℝ := Real.pi * x + Real.log x / Real.log 2

theorem zero_point_interval : 
  f (1/4) < 0 ∧ f (1/2) > 0 → ∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 1/2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l199_199309


namespace find_number_l199_199884

theorem find_number (x : ℕ) (h : 3 * x = 2 * 51 - 3) : x = 33 :=
by
  sorry

end find_number_l199_199884


namespace circle_intersects_cells_l199_199809

/-- On a grid with 1 cm x 1 cm cells, a circle with a radius of 100 cm is drawn.
    The circle does not pass through any vertices of the cells and does not touch the sides of the cells.
    Prove that the number of cells the circle can intersect is either 800 or 799. -/
theorem circle_intersects_cells (r : ℝ) (gsize : ℝ) (cells : ℕ) :
  r = 100 ∧ gsize = 1 ∧ cells = 800 ∨ cells = 799 :=
by
  sorry

end circle_intersects_cells_l199_199809


namespace brown_eyed_brunettes_l199_199952

theorem brown_eyed_brunettes (total_girls blondes brunettes blue_eyed_blondes brown_eyed_girls : ℕ) 
    (h1 : total_girls = 60) 
    (h2 : blondes + brunettes = total_girls) 
    (h3 : blue_eyed_blondes = 20) 
    (h4 : brunettes = 35) 
    (h5 : brown_eyed_girls = 22) 
    (h6 : blondes = total_girls - brunettes) 
    (h7 : brown_eyed_blondes = blondes - blue_eyed_blondes) :
  brunettes - (brown_eyed_girls - brown_eyed_blondes) = 17 :=
by sorry  -- Proof is not required

end brown_eyed_brunettes_l199_199952


namespace pastries_solution_l199_199100

def pastries_problem : Prop :=
  ∃ (F Calvin Phoebe Grace : ℕ),
  (Calvin = F + 8) ∧
  (Phoebe = F + 8) ∧
  (Grace = 30) ∧
  (F + Calvin + Phoebe + Grace = 97) ∧
  (Grace - Calvin = 5) ∧
  (Grace - Phoebe = 5)

theorem pastries_solution : pastries_problem :=
by
  sorry

end pastries_solution_l199_199100


namespace domain_of_g_l199_199729

def f : ℝ → ℝ := sorry  -- Placeholder for the function f

noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

theorem domain_of_g :
  ∀ x : ℝ, g x ≠ 0 → (-1/2 < x ∧ x ≤ 3) :=
by
  intro x hx
  sorry

end domain_of_g_l199_199729


namespace train_speed_correct_l199_199536

noncomputable def train_speed : ℝ :=
  let distance := 120 -- meters
  let time := 5.999520038396929 -- seconds
  let speed_m_s := distance / time -- meters per second
  speed_m_s * 3.6 -- converting to km/hr

theorem train_speed_correct : train_speed = 72.004800384 := by
  simp [train_speed]
  sorry

end train_speed_correct_l199_199536


namespace instantaneous_velocity_at_t2_l199_199176

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2

theorem instantaneous_velocity_at_t2 :
  (deriv displacement 2) = 4 :=
by
  sorry

end instantaneous_velocity_at_t2_l199_199176


namespace tangent_line_slope_eq_k_l199_199611

noncomputable def tangent_slope_at_ln_x (x : ℝ) (hx : x > 0) : ℝ := by
  exact 1 / x

theorem tangent_line_slope_eq_k (k : ℝ) (x : ℝ) (hx : x > 0) (h_tangent : k = tangent_slope_at_ln_x x hx) : k = 1 / Real.exp(1) := by
  have : x = Real.exp(1) := by
    sorry -- proof that x = e
  rw [this] at h_tangent
  exact h_tangent
  sorry -- fill in the rest of the proof


end tangent_line_slope_eq_k_l199_199611


namespace find_multiple_l199_199872

theorem find_multiple (x m : ℝ) (hx : x = 3) (h : x + 17 = m * (1 / x)) : m = 60 := 
by
  sorry

end find_multiple_l199_199872


namespace distance_from_mo_l199_199618

-- Definitions based on conditions
-- 1. Grid squares have side length 1 cm.
-- 2. Shape shaded gray on the grid.
-- 3. The total shaded area needs to be divided into two equal parts.
-- 4. The line to be drawn is parallel to line MO.

noncomputable def grid_side_length : ℝ := 1.0
noncomputable def shaded_area : ℝ := 10.0
noncomputable def line_mo_distance (d : ℝ) : Prop := 
  ∃ parallel_line_distance, parallel_line_distance = d ∧ 
    ∃ equal_area, 2 * equal_area = shaded_area ∧ equal_area = 5.0

-- Theorem: The parallel line should be drawn at 2.6 cm 
theorem distance_from_mo (d : ℝ) : 
  d = 2.6 ↔ line_mo_distance d := 
by
  sorry

end distance_from_mo_l199_199618


namespace complex_solution_l199_199184

theorem complex_solution (i z : ℂ) (h : i^2 = -1) (hz : (z - 2 * i) * (2 - i) = 5) : z = 2 + 3 * i :=
sorry

end complex_solution_l199_199184


namespace coefficient_x2_expansion_l199_199989

-- Define the problem statement
theorem coefficient_x2_expansion (m : ℝ) 
  (h : binomial 5 3 * m^3 = -10) : 
  m = -1 :=
by
  sorry

end coefficient_x2_expansion_l199_199989


namespace final_portfolio_value_l199_199457

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end final_portfolio_value_l199_199457


namespace min_correct_all_four_l199_199126

def total_questions : ℕ := 15
def correct_xiaoxi : ℕ := 11
def correct_xiaofei : ℕ := 12
def correct_xiaomei : ℕ := 13
def correct_xiaoyang : ℕ := 14

theorem min_correct_all_four : 
(∀ total_questions correct_xiaoxi correct_xiaofei correct_xiaomei correct_xiaoyang, 
  total_questions = 15 → correct_xiaoxi = 11 → 
  correct_xiaofei = 12 → correct_xiaomei = 13 → 
  correct_xiaoyang = 14 → 
  ∃ k : ℕ, k = 5 ∧ 
    k = total_questions - ((total_questions - correct_xiaoxi) + 
    (total_questions - correct_xiaofei) + 
    (total_questions - correct_xiaomei) + 
    (total_questions - correct_xiaoyang)) / 4) := 
sorry

end min_correct_all_four_l199_199126


namespace find_divisor_l199_199318

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h_dividend : dividend = 125) 
  (h_quotient : quotient = 8) 
  (h_remainder : remainder = 5) 
  (h_equation : dividend = (divisor * quotient) + remainder) : 
  divisor = 15 := by
  sorry

end find_divisor_l199_199318


namespace no_distinct_triple_exists_for_any_quadratic_trinomial_l199_199113

theorem no_distinct_triple_exists_for_any_quadratic_trinomial (f : ℝ → ℝ) 
    (hf : ∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) :
    ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = b ∧ f b = c ∧ f c = a := 
by 
  sorry

end no_distinct_triple_exists_for_any_quadratic_trinomial_l199_199113


namespace negation_of_p_equiv_l199_199147

-- Define the initial proposition p
def p : Prop := ∃ x : ℝ, x^2 - 5*x - 6 < 0

-- State the theorem for the negation of p
theorem negation_of_p_equiv : ¬p ↔ ∀ x : ℝ, x^2 - 5*x - 6 ≥ 0 :=
by
  sorry

end negation_of_p_equiv_l199_199147


namespace correct_statement_l199_199094

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end correct_statement_l199_199094


namespace max_value_expression_l199_199215

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l199_199215


namespace find_roots_l199_199906

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_l199_199906


namespace solve_equation_l199_199562

theorem solve_equation :
  ∀ x : ℝ,
  (1 / (x^2 + 12 * x - 9) + 
   1 / (x^2 + 3 * x - 9) + 
   1 / (x^2 - 12 * x - 9) = 0) ↔ 
  (x = 1 ∨ x = -9 ∨ x = 3 ∨ x = -3) := 
by
  sorry

end solve_equation_l199_199562


namespace collinear_points_l199_199946

theorem collinear_points (x y : ℝ) (h_collinear : ∃ k : ℝ, (x + 1, y, 3) = (2 * k, 4 * k, 6 * k)) : x - y = -2 := 
by 
  sorry

end collinear_points_l199_199946


namespace scientific_notation_32000000_l199_199379

def scientific_notation (n : ℕ) : String := sorry

theorem scientific_notation_32000000 :
  scientific_notation 32000000 = "3.2 × 10^7" :=
sorry

end scientific_notation_32000000_l199_199379


namespace original_selling_price_l199_199889

theorem original_selling_price (P SP1 SP2 : ℝ) (h1 : SP1 = 1.10 * P)
    (h2 : SP2 = 1.17 * P) (h3 : SP2 - SP1 = 35) : SP1 = 550 :=
by
  sorry

end original_selling_price_l199_199889


namespace valid_numbers_count_l199_199406

def count_valid_numbers (n : ℕ) : ℕ := 2 ^ (n + 1) - 2 * n - 2

theorem valid_numbers_count {n : ℕ} (h: n > 0) :
    ∑ k in finset.range n, (n - k + 1) * 2^(k - 1) - 1 = count_valid_numbers n :=
  sorry

end valid_numbers_count_l199_199406


namespace calculate_total_weight_l199_199469

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l199_199469


namespace five_ab4_is_perfect_square_l199_199608

theorem five_ab4_is_perfect_square (a b : ℕ) (h : 5000 ≤ 5000 + 100 * a + 10 * b + 4 ∧ 5000 + 100 * a + 10 * b + 4 ≤ 5999) :
    ∃ n, n^2 = 5000 + 100 * a + 10 * b + 4 → a + b = 9 :=
by
  sorry

end five_ab4_is_perfect_square_l199_199608


namespace jane_last_day_vases_l199_199621

def vasesPerDay : Nat := 16
def totalVases : Nat := 248

theorem jane_last_day_vases : totalVases % vasesPerDay = 8 := by
  sorry

end jane_last_day_vases_l199_199621


namespace factor_expression_l199_199557

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l199_199557


namespace slide_vs_slip_l199_199867

noncomputable def ladder : Type := sorry

def slide_distance (ladder : ladder) : ℝ := sorry
def slip_distance (ladder : ladder) : ℝ := sorry
def is_right_triangle (ladder : ladder) : Prop := sorry

theorem slide_vs_slip (l : ladder) (h : is_right_triangle l) : slip_distance l > slide_distance l :=
sorry

end slide_vs_slip_l199_199867


namespace closest_point_on_parabola_to_line_l199_199641

theorem closest_point_on_parabola_to_line :
  ∃ (x : ℝ), (∀ y, y = x^2) ∧ (d * ((2 * x) - y - 4) / sqrt 5) :=
sorry

end closest_point_on_parabola_to_line_l199_199641


namespace find_y_z_l199_199745

theorem find_y_z (y z : ℝ) : 
  (∃ k : ℝ, (1:ℝ) = -k ∧ (2:ℝ) = k * y ∧ (3:ℝ) = k * z) → y = -2 ∧ z = -3 :=
by
  sorry

end find_y_z_l199_199745


namespace probability_more_ones_than_sixes_l199_199765

theorem probability_more_ones_than_sixes (total_dice : ℕ) (sides_of_dice : ℕ) 
  (ones : ℕ) (sixes : ℕ) (total_outcomes : ℕ) (equal_outcomes : ℕ) : 
  (total_dice = 5) → 
  (sides_of_dice = 6) → 
  (total_outcomes = 6^total_dice) → 
  (equal_outcomes = 1024 + 1280 + 120) → 
  (ones > sixes) → 
  (prob_more_ones_than_sixes : ℚ) → 
  prob_more_ones_than_sixes = (1/2) * (1 - (equal_outcomes / total_outcomes)) := 
begin
  intros h1 h2 h3 h4 h5 h6,
  rw [h1, h2, h3, h4],
  sorry,
end

end probability_more_ones_than_sixes_l199_199765


namespace trig_evaluation_l199_199698

noncomputable def sin30 := 1 / 2
noncomputable def cos45 := Real.sqrt 2 / 2
noncomputable def tan30 := Real.sqrt 3 / 3
noncomputable def sin60 := Real.sqrt 3 / 2

theorem trig_evaluation : 4 * sin30 - Real.sqrt 2 * cos45 - Real.sqrt 3 * tan30 + 2 * sin60 = Real.sqrt 3 := by
  sorry

end trig_evaluation_l199_199698


namespace math_proof_problem_l199_199349

noncomputable def expr : ℚ :=
  ((5 / 8 * (3 / 7) + 1 / 4 * (2 / 6)) - (2 / 3 * (1 / 4) - 1 / 5 * (4 / 9))) * 
  ((7 / 9 * (2 / 5) * (1 / 2) * 5040 + 1 / 3 * (3 / 8) * (9 / 11) * 4230))

theorem math_proof_problem : expr = 336 := 
  by
  sorry

end math_proof_problem_l199_199349


namespace vector_dot_product_sum_l199_199600

noncomputable def points_in_plane (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) : Prop :=
  dist_AB = 3 ∧ dist_BC = 5 ∧ dist_CA = 6

theorem vector_dot_product_sum (A B C : Type) (dist_AB dist_BC dist_CA : ℝ) (HA : points_in_plane A B C dist_AB dist_BC dist_CA) :
    ∃ (AB BC CA : ℝ), AB * BC + BC * CA + CA * AB = -35 :=
by
  sorry

end vector_dot_product_sum_l199_199600


namespace selection_plans_count_l199_199287

-- Define the number of boys and girls
def num_boys : ℕ := 3
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the number of subjects
def num_subjects : ℕ := 3

-- Prove that the number of selection plans is 120
theorem selection_plans_count :
  (Nat.choose total_students num_subjects) * (num_subjects.factorial) = 120 := 
by
  sorry

end selection_plans_count_l199_199287


namespace continuous_at_5_l199_199569

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x > 5 then x - 2 else 3 * x + b

theorem continuous_at_5 {b : ℝ} : ContinuousAt (fun x => f x b) 5 ↔ b = -12 := by
  sorry

end continuous_at_5_l199_199569


namespace factor_expression_l199_199558

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression_l199_199558


namespace domain_f_l199_199300

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x-1) / Real.log 2) + 1

theorem domain_f : domain f = {x | 1 < x} :=
by {
  sorry
}

end domain_f_l199_199300


namespace intersect_complementB_l199_199734

def setA (x : ℝ) : Prop := ∃ y : ℝ, y = Real.log (9 - x^2)

def setB (x : ℝ) : Prop := ∃ y : ℝ, y = Real.sqrt (4 * x - x^2)

def complementB (x : ℝ) : Prop := x < 0 ∨ 4 < x

theorem intersect_complementB :
  { x : ℝ | setA x } ∩ { x : ℝ | complementB x } = { x : ℝ | -3 < x ∧ x < 0 } :=
sorry

end intersect_complementB_l199_199734


namespace parallel_vectors_m_l199_199937

theorem parallel_vectors_m (m : ℝ) :
  let a := (1, 2)
  let b := (m, m + 1)
  a.1 * b.2 = a.2 * b.1 → m = 1 :=
by
  intros a b h
  dsimp at *
  sorry

end parallel_vectors_m_l199_199937


namespace tangent_line_eq_l199_199038

open Real

noncomputable def f (x : ℝ) : ℝ := exp x * log x

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ y = 0) :
  ∃ m b, (∀ t, y = m * (t - 1) + b) ∧ (f x = y) ∧ (m = exp 1) ∧ (b = -exp 1) :=
by
  sorry

end tangent_line_eq_l199_199038


namespace sum_log_base_5_divisors_l199_199652

theorem sum_log_base_5_divisors (n : ℕ) (h : n * (n + 1) / 2 = 264) : n = 23 :=
by
  sorry

end sum_log_base_5_divisors_l199_199652


namespace evaluate_g_at_3_l199_199731

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem evaluate_g_at_3 : g 3 = 109 := by
  sorry

end evaluate_g_at_3_l199_199731


namespace radius_of_outer_circle_l199_199645

theorem radius_of_outer_circle (C_inner : ℝ) (width : ℝ) (h : C_inner = 880) (w : width = 25) :
  ∃ r_outer : ℝ, r_outer = 165 :=
by
  have r_inner := C_inner / (2 * Real.pi)
  have r_outer := r_inner + width
  use r_outer
  sorry

end radius_of_outer_circle_l199_199645


namespace plane_equation_through_point_parallel_l199_199902

theorem plane_equation_through_point_parallel (A B C D : ℤ) (hx hy hz : ℤ) (x y z : ℤ)
  (h_point : (A, B, C, D) = (-2, 1, -3, 10))
  (h_coordinates : (hx, hy, hz) = (2, -3, 1))
  (h_plane_parallel : ∀ x y z, -2 * x + y - 3 * z = 7 ↔ A * x + B * y + C * z + D = 0)
  (h_form : A > 0):
  ∃ A' B' C' D', A' * (x : ℤ) + B' * (y : ℤ) + C' * (z : ℤ) + D' = 0 :=
by
  sorry

end plane_equation_through_point_parallel_l199_199902


namespace general_term_formula_l199_199598

variable {a_n : ℕ → ℕ} -- Sequence {a_n}
variable {S_n : ℕ → ℕ} -- Sum of the first n terms

-- Condition given in the problem
def S_n_condition (n : ℕ) : ℕ :=
  2 * n^2 + n

theorem general_term_formula (n : ℕ) (h₀ : ∀ (n : ℕ), S_n n = 2 * n^2 + n) :
  a_n n = 4 * n - 1 :=
sorry

end general_term_formula_l199_199598


namespace ratio_horizontal_to_checkered_l199_199791

/--
In a cafeteria, 7 people are wearing checkered shirts, while the rest are wearing vertical stripes
and horizontal stripes. There are 40 people in total, and 5 of them are wearing vertical stripes.
What is the ratio of the number of people wearing horizontal stripes to the number of people wearing
checkered shirts?
-/
theorem ratio_horizontal_to_checkered
  (total_people : ℕ)
  (checkered_people : ℕ)
  (vertical_people : ℕ)
  (horizontal_people : ℕ)
  (ratio : ℕ)
  (h_total : total_people = 40)
  (h_checkered : checkered_people = 7)
  (h_vertical : vertical_people = 5)
  (h_horizontal : horizontal_people = total_people - checkered_people - vertical_people)
  (h_ratio : ratio = horizontal_people / checkered_people) :
  ratio = 4 :=
by
  sorry

end ratio_horizontal_to_checkered_l199_199791


namespace algae_plants_in_milford_lake_l199_199630

theorem algae_plants_in_milford_lake (original : ℕ) (increase : ℕ) : (original = 809) → (increase = 2454) → (original + increase = 3263) :=
by
  sorry

end algae_plants_in_milford_lake_l199_199630


namespace car_race_probability_l199_199446

theorem car_race_probability :
  let pX := 1/8
  let pY := 1/12
  let pZ := 1/6
  pX + pY + pZ = 3/8 :=
by
  sorry

end car_race_probability_l199_199446


namespace anne_total_bottle_caps_l199_199374

def initial_bottle_caps_anne : ℕ := 10
def found_bottle_caps_anne : ℕ := 5

theorem anne_total_bottle_caps : initial_bottle_caps_anne + found_bottle_caps_anne = 15 := 
by
  sorry

end anne_total_bottle_caps_l199_199374


namespace total_weight_of_settings_l199_199476

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l199_199476


namespace rainfall_mondays_l199_199632

theorem rainfall_mondays
  (M : ℕ)
  (rain_monday : ℝ)
  (rain_tuesday : ℝ)
  (num_tuesdays : ℕ)
  (extra_rain_tuesdays : ℝ)
  (h1 : rain_monday = 1.5)
  (h2 : rain_tuesday = 2.5)
  (h3 : num_tuesdays = 9)
  (h4 : num_tuesdays * rain_tuesday = rain_monday * M + extra_rain_tuesdays)
  (h5 : extra_rain_tuesdays = 12) :
  M = 7 := 
sorry

end rainfall_mondays_l199_199632


namespace greatest_integer_difference_l199_199853

theorem greatest_integer_difference (x y : ℤ) (hx : 7 < x ∧ x < 9) (hy : 9 < y ∧ y < 15) :
  ∀ d : ℤ, (d = y - x) → d ≤ 6 := 
sorry

end greatest_integer_difference_l199_199853


namespace net_effect_transactions_l199_199377

theorem net_effect_transactions {a o : ℝ} (h1 : 3 * a / 4 = 15000) (h2 : 5 * o / 4 = 15000) :
  a + o - (2 * 15000) = 2000 :=
by
  sorry

end net_effect_transactions_l199_199377


namespace final_portfolio_value_l199_199458

theorem final_portfolio_value (initial_amount : ℕ) (growth_1 : ℕ) (additional_funds : ℕ) (growth_2 : ℕ) :
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100)
  let after_adding_funds := after_first_year_growth + additional_funds
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100)
  after_second_year_growth = 132 :=
by
  let after_first_year_growth := initial_amount + (initial_amount * growth_1 / 100);
  let after_adding_funds := after_first_year_growth + additional_funds;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * growth_2 / 100);
  trivial

-- substituting the values as per conditions:
example : final_portfolio_value 80 15 28 10 = 132 := by
  let after_first_year_growth := 80 + (80 * 15 / 100);
  let after_adding_funds := after_first_year_growth + 28;
  let after_second_year_growth := after_adding_funds + (after_adding_funds * 10 / 100);
  trivial

end final_portfolio_value_l199_199458


namespace fraction_sum_neg_one_l199_199390

variable (a : ℚ)

theorem fraction_sum_neg_one (h : a ≠ 1/2) : (a / (1 - 2 * a)) + ((a - 1) / (1 - 2 * a)) = -1 := 
sorry

end fraction_sum_neg_one_l199_199390


namespace edward_games_start_l199_199551

theorem edward_games_start (sold_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h_sold : sold_games = 19) (h_boxes : boxes = 2) (h_game_box : games_per_box = 8) : 
  sold_games + boxes * games_per_box = 35 := 
  by 
    sorry

end edward_games_start_l199_199551


namespace probability_both_companies_two_correct_lower_variance_greater_chance_l199_199189

-- Definitions
def num_questions := 6
def questions_drawn := 3
def company_a_correct := 4
def company_b_prob := 2 / 3

-- Calculating number of combinations
def comb (n r : ℕ) := n.choose r

-- Probability that Company A answers i questions correctly
def prob_A (i : ℕ) : ℝ :=
  comb company_a_correct i * comb (num_questions - company_a_correct) (questions_drawn - i) / comb num_questions questions_drawn

-- Probability that Company B answers j questions correctly
def prob_B (j : ℕ) : ℝ :=
  comb questions_drawn j * company_b_prob^j * (1 - company_b_prob)^(questions_drawn - j)

-- Main statements
theorem probability_both_companies_two_correct :
  prob_A 1 * prob_B 1 + prob_A 2 * prob_B 0 = 1 / 15 := sorry

theorem lower_variance_greater_chance :
  let E_a := (1 : ℝ) * (1/5) + (2 : ℝ) * (3/5) + (3 : ℝ) * (1/5),
      Var_a := (1 - E_a)^2 * (1/5) + (2 - E_a)^2 * (3/5) + (3 - E_a)^0^2 * (1/5),
      E_b := (0 : ℝ) * (1/27) + (1 : ℝ) * (2/9) + (2 : ℝ) * (4/9) + (3 : ℝ) * (8/27),
      Var_b := (0 - E_b)^2 * (1/27) + (1 - E_b)^2 * (2/9) + (2 - E_b)^2 * (4/9) + (3 - E_b)^0^2 * (8/27)
  in Var_a < Var_b := sorry

end probability_both_companies_two_correct_lower_variance_greater_chance_l199_199189


namespace find_f_value_l199_199604

theorem find_f_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / x^2) : 
  f (1 / 2) = 15 :=
sorry

end find_f_value_l199_199604


namespace rectangle_side_ratio_l199_199259

theorem rectangle_side_ratio
  (s : ℝ)  -- the side length of the inner square
  (y x : ℝ) -- the side lengths of the rectangles (y: shorter, x: longer)
  (h1 : 9 * s^2 = (3 * s)^2)  -- the area of the outer square is 9 times that of the inner square
  (h2 : s + 2*y = 3*s)  -- the total side length relation due to geometry
  (h3 : x + y = 3*s)  -- another side length relation
: x / y = 2 :=
by
  sorry

end rectangle_side_ratio_l199_199259


namespace problem_solution_l199_199666

theorem problem_solution 
  (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) :
  4 * x^4 + 17 * x^2 * y + 4 * y^2 < (m / 4) * (x^4 + 2 * x^2 * y + y^2) ↔ 25 < m :=
sorry

end problem_solution_l199_199666


namespace updated_mean_166_l199_199823

/-- The mean of 50 observations is 200. Later, it was found that there is a decrement of 34 
from each observation. Prove that the updated mean of the observations is 166. -/
theorem updated_mean_166
  (mean : ℝ) (n : ℕ) (decrement : ℝ) (updated_mean : ℝ)
  (h1 : mean = 200) (h2 : n = 50) (h3 : decrement = 34) (h4 : updated_mean = 166) :
  mean - (decrement * n) / n = updated_mean :=
by
  sorry

end updated_mean_166_l199_199823


namespace dividend_percentage_l199_199368

theorem dividend_percentage (investment_amount market_value : ℝ) (interest_rate : ℝ) 
  (h1 : investment_amount = 44) (h2 : interest_rate = 12) (h3 : market_value = 33) : 
  ((interest_rate / 100) * investment_amount / market_value) * 100 = 16 := 
by
  sorry

end dividend_percentage_l199_199368


namespace problem_1_problem_2_l199_199708

def f (x : ℝ) : ℝ := abs (2 * x + 3) + abs (2 * x - 1)

theorem problem_1 (x : ℝ) : (f x ≤ 5) ↔ (-7/4 ≤ x ∧ x ≤ 3/4) :=
by sorry

theorem problem_2 (m : ℝ) : (∃ x, f x < abs (m - 1)) ↔ (m > 5 ∨ m < -3) :=
by sorry

end problem_1_problem_2_l199_199708


namespace dice_probability_not_all_same_l199_199062

theorem dice_probability_not_all_same : 
  let total_outcomes := (8 : ℕ)^5 in
  let same_number_outcomes := 8 in
  let probability_all_same := (same_number_outcomes : ℚ) / total_outcomes in
  let probability_not_all_same := 1 - probability_all_same in
  probability_not_all_same = 4095 / 4096 :=
by
  sorry

end dice_probability_not_all_same_l199_199062


namespace linear_eq_substitution_l199_199948

theorem linear_eq_substitution (x y : ℝ) (h1 : 3 * x - 4 * y = 2) (h2 : x = 2 * y - 1) :
  3 * (2 * y - 1) - 4 * y = 2 :=
by
  sorry

end linear_eq_substitution_l199_199948


namespace article_large_font_pages_l199_199795

theorem article_large_font_pages (L S : ℕ) 
  (pages_eq : L + S = 21) 
  (words_eq : 1800 * L + 2400 * S = 48000) : 
  L = 4 := 
by 
  sorry

end article_large_font_pages_l199_199795


namespace expected_points_A_correct_prob_A_B_same_points_correct_l199_199699

-- Conditions
def game_is_independent := true

def prob_A_B_win := 2/5
def prob_A_B_draw := 1/5

def prob_A_C_win := 1/3
def prob_A_C_draw := 1/3

def prob_B_C_win := 1/2
def prob_B_C_draw := 1/6

noncomputable def prob_A_B_lose := 1 - prob_A_B_win - prob_A_B_draw
noncomputable def prob_A_C_lose := 1 - prob_A_C_win - prob_A_C_draw
noncomputable def prob_B_C_lose := 1 - prob_B_C_win - prob_B_C_draw

noncomputable def expected_points_A : ℚ := 0 * (prob_A_B_lose * prob_A_C_lose)        /- P(ξ=0) = 2/15 -/
                                       + 1 * ((prob_A_B_draw * prob_A_C_lose) +
                                              (prob_A_B_lose * prob_A_C_draw))        /- P(ξ=1) = 1/5 -/
                                       + 2 * (prob_A_B_draw * prob_A_C_draw)         /- P(ξ=2) = 1/15 -/
                                       + 3 * ((prob_A_B_win * prob_A_C_lose) + 
                                              (prob_A_B_win * prob_A_C_draw) + 
                                              (prob_A_C_win * prob_A_B_lose))        /- P(ξ=3) = 4/15 -/
                                       + 4 * ((prob_A_B_draw * prob_A_C_win) +
                                              (prob_A_B_win * prob_A_C_win))         /- P(ξ=4) = 1/5 -/
                                       + 6 * (prob_A_B_win * prob_A_C_win)           /- P(ξ=6) = 2/15 -/

theorem expected_points_A_correct : expected_points_A = 41 / 15 :=
by
  sorry

noncomputable def prob_A_B_same_points: ℚ := ((prob_A_B_draw * prob_A_C_lose) * prob_B_C_lose)  /- both 1 point -/
                                            + ((prob_A_B_draw * prob_A_C_draw) * prob_B_C_draw)/- both 2 points -/
                                            + ((prob_A_B_win * prob_B_C_win) * prob_A_C_lose)  /- both 3 points -/
                                            + ((prob_A_B_win * prob_A_C_lose) * prob_B_C_win)  /- both 3 points -/
                                            + ((prob_A_B_draw * prob_A_C_win) * prob_B_C_win)  /- both 4 points -/

theorem prob_A_B_same_points_correct : prob_A_B_same_points = 8 / 45 :=
by
  sorry

end expected_points_A_correct_prob_A_B_same_points_correct_l199_199699


namespace evaluate_expression_l199_199337

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l199_199337


namespace factor_fraction_l199_199559

/- Definitions based on conditions -/
variables {a b c : ℝ}

theorem factor_fraction :
  ( (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 ) / 
  ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) :=
begin
  sorry
end

end factor_fraction_l199_199559


namespace parabola_c_value_l199_199090

theorem parabola_c_value {b c : ℝ} :
  (1:ℝ)^2 + b * (1:ℝ) + c = 2 → 
  (4:ℝ)^2 + b * (4:ℝ) + c = 5 → 
  (7:ℝ)^2 + b * (7:ℝ) + c = 2 →
  c = 9 :=
by
  intros h1 h2 h3
  sorry

end parabola_c_value_l199_199090


namespace arc_length_l199_199988

theorem arc_length (circumference : ℝ) (angle : ℝ) (h1 : circumference = 72) (h2 : angle = 45) :
  ∃ length : ℝ, length = 9 :=
by
  sorry

end arc_length_l199_199988


namespace primes_unique_l199_199563

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique_l199_199563


namespace bess_milk_daily_l199_199900

-- Definitions based on conditions from step a)
variable (B : ℕ) -- B is the number of pails Bess gives every day

def BrownieMilk : ℕ := 3 * B
def DaisyMilk : ℕ := B + 1
def TotalDailyMilk : ℕ := B + BrownieMilk B + DaisyMilk B

-- Conditions definition to be used in Lean to ensure the equivalence
axiom weekly_milk_total : 7 * TotalDailyMilk B = 77
axiom daily_milk_eq : TotalDailyMilk B = 11

-- Prove that Bess gives 2 pails of milk everyday
theorem bess_milk_daily : B = 2 :=
by
  sorry

end bess_milk_daily_l199_199900


namespace loop_condition_l199_199932

theorem loop_condition (b : ℕ) : (b = 10 ∧ ∀ n, b = 10 + 3 * n ∧ b < 16 → n + 1 = 16) → ∀ (condition : ℕ → Prop), condition b → b = 16 :=
by sorry

end loop_condition_l199_199932


namespace subtract_add_example_l199_199320

theorem subtract_add_example : (3005 - 3000) + 10 = 15 :=
by
  sorry

end subtract_add_example_l199_199320


namespace mean_of_six_numbers_l199_199505

theorem mean_of_six_numbers (sum_of_six: ℚ) (H: sum_of_six = 3 / 4) : sum_of_six / 6 = 1 / 8 := by
  sorry

end mean_of_six_numbers_l199_199505


namespace intersection_point_exists_l199_199407

def line_param_eq (x y z : ℝ) (t : ℝ) := x = 5 + t ∧ y = 3 - t ∧ z = 2
def plane_eq (x y z : ℝ) := 3 * x + y - 5 * z - 12 = 0

theorem intersection_point_exists : 
  ∃ t : ℝ, ∃ x y z : ℝ, line_param_eq x y z t ∧ plane_eq x y z ∧ x = 7 ∧ y = 1 ∧ z = 2 :=
by {
  -- Skipping the proof
  sorry
}

end intersection_point_exists_l199_199407


namespace quiz_probability_l199_199091

theorem quiz_probability :
  let probMCQ := 1/3
  let probTF1 := 1/2
  let probTF2 := 1/2
  probMCQ * probTF1 * probTF2 = 1/12 := by
  sorry

end quiz_probability_l199_199091


namespace soda_difference_l199_199866

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := 
  by
  sorry

end soda_difference_l199_199866


namespace new_average_score_l199_199158

theorem new_average_score (n : ℕ) (initial_avg : ℕ) (grace_marks : ℕ) (h1 : n = 35) (h2 : initial_avg = 37) (h3 : grace_marks = 3) : initial_avg + grace_marks = 40 := by
  sorry

end new_average_score_l199_199158


namespace two_distinct_solutions_diff_l199_199145

theorem two_distinct_solutions_diff (a b : ℝ) (h1 : a ≠ b) (h2 : a > b)
  (h3 : ∀ x, (x = a ∨ x = b) ↔ (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3) :
  a - b = 3 :=
by
  -- Proof will be provided here.
  sorry

end two_distinct_solutions_diff_l199_199145


namespace proposition_p_and_not_q_l199_199123

theorem proposition_p_and_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬ (P ∧ Q)) : (P ↔ ¬ Q) :=
sorry

end proposition_p_and_not_q_l199_199123


namespace xiaoxia_exceeds_xiaoming_l199_199348

theorem xiaoxia_exceeds_xiaoming (n : ℕ) : 
  52 + 15 * n > 70 + 12 * n := 
sorry

end xiaoxia_exceeds_xiaoming_l199_199348


namespace maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l199_199961

-- Part (a): One blue cube
theorem maximum_amount_one_blue_cube : 
  ∃ (B : ℕ → ℚ) (P : ℕ → ℕ), (B 1 = 2) ∧ (∀ m > 1, B m = 2^m / P m) ∧ (P 1 = 1) ∧ (∀ m > 1, P m = m) ∧ B 100 = 2^100 / 100 :=
by
  sorry

-- Part (b): Exactly n blue cubes
theorem maximum_amount_n_blue_cubes (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 100) : 
  ∃ (B : ℕ × ℕ → ℚ) (P : ℕ × ℕ → ℕ), (B (1, 0) = 2) ∧ (B (1, 1) = 2) ∧ (∀ m > 1, B (m, 0) = 2^m) ∧ (P (1, 0) = 1) ∧ (P (1, 1) = 1) ∧ (∀ m > 1, P (m, 0) = 1) ∧ B (100, n) = 2^100 / Nat.choose 100 n :=
by
  sorry

end maximum_amount_one_blue_cube_maximum_amount_n_blue_cubes_l199_199961


namespace gcd_98_63_l199_199661

-- Definition of gcd
def gcd_euclidean := ∀ (a b : ℕ), ∃ (g : ℕ), gcd a b = g

-- Statement of the problem using Lean
theorem gcd_98_63 : gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l199_199661


namespace problem_statement_l199_199202

noncomputable def f1 (x : ℝ) := x + (1 / x)
noncomputable def f2 (x : ℝ) := 1 / (x ^ 2)
noncomputable def f3 (x : ℝ) := x ^ 3 - 2 * x
noncomputable def f4 (x : ℝ) := x ^ 2

theorem problem_statement : ∀ (x : ℝ), f2 (-x) = f2 x := by 
  sorry

end problem_statement_l199_199202


namespace range_of_function_l199_199164

noncomputable def function_range (x : ℝ) : ℝ :=
    (1 / 2) ^ (-x^2 + 2 * x)

theorem range_of_function : 
    (Set.range function_range) = Set.Ici (1 / 2) :=
by
    sorry

end range_of_function_l199_199164


namespace right_triangle_area_and_perimeter_l199_199039

theorem right_triangle_area_and_perimeter (a c : ℕ) (h₁ : c = 13) (h₂ : a = 5) :
  ∃ (b : ℕ), b^2 = c^2 - a^2 ∧
             (1/2 : ℝ) * (a : ℝ) * (b : ℝ) = 30 ∧
             (a + b + c : ℕ) = 30 :=
by
  sorry

end right_triangle_area_and_perimeter_l199_199039


namespace initial_investment_l199_199671

theorem initial_investment (b : ℝ) (t_b : ℝ) (t_a : ℝ) (ratio_profit : ℝ) (x : ℝ) :
  b = 36000 → t_b = 4.5 → t_a = 12 → ratio_profit = 2 →
  (x * t_a) / (b * t_b) = ratio_profit → x = 27000 := 
by
  intros hb ht_b ht_a hr hp
  rw [hb, ht_b, ht_a, hr] at hp
  sorry

end initial_investment_l199_199671


namespace total_weight_of_settings_l199_199477

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end total_weight_of_settings_l199_199477


namespace parabola_chord_solution_l199_199160

noncomputable def parabola_chord : Prop :=
  ∃ x_A x_B : ℝ, (140 = 5 * x_B^2 + 2 * x_A^2) ∧ 
  ((x_A = -5 * Real.sqrt 2 ∧ x_B = 2 * Real.sqrt 2) ∨ 
   (x_A = 5 * Real.sqrt 2 ∧ x_B = -2 * Real.sqrt 2))

theorem parabola_chord_solution : parabola_chord := 
sorry

end parabola_chord_solution_l199_199160


namespace prism_faces_same_color_l199_199873

structure PrismColoring :=
  (A : Fin 5 → Fin 5 → Bool)
  (B : Fin 5 → Fin 5 → Bool)
  (A_to_B : Fin 5 → Fin 5 → Bool)

def all_triangles_diff_colors (pc : PrismColoring) : Prop :=
  ∀ i j k : Fin 5, i ≠ j → j ≠ k → k ≠ i →
    (pc.A i j = !pc.A i k ∨ pc.A i j = !pc.A j k) ∧
    (pc.B i j = !pc.B i k ∨ pc.B i j = !pc.B j k) ∧
    (pc.A_to_B i j = !pc.A_to_B i k ∨ pc.A_to_B i j = !pc.A_to_B j k)

theorem prism_faces_same_color (pc : PrismColoring) (h : all_triangles_diff_colors pc) :
  (∀ i j : Fin 5, pc.A i j = pc.A 0 1) ∧ (∀ i j : Fin 5, pc.B i j = pc.B 0 1) :=
sorry

end prism_faces_same_color_l199_199873


namespace probability_more_ones_than_sixes_l199_199778

theorem probability_more_ones_than_sixes :
  let num_faces := 6 in
  let num_rolls := 5 in
  let total_outcomes := num_faces ^ num_rolls in
  let favorable_outcomes := 2711 in
  (favorable_outcomes : ℚ) / total_outcomes = 2711 / 7776 :=
sorry

end probability_more_ones_than_sixes_l199_199778


namespace probability_more_ones_than_sixes_l199_199769

open ProbabilityTheory

noncomputable def prob_more_ones_than_sixes : ℚ :=
  let total_outcomes := 6^5 in
  let favorable_cases := 679 in
  favorable_cases / total_outcomes

theorem probability_more_ones_than_sixes (h_dice_fair : ∀ (i : ℕ), i ∈ Finset.range 6 → ℙ (i = 1) = 1 / 6) :
  prob_more_ones_than_sixes = 679 / 1944 :=
by {
  -- placeholder for the actual proof
  sorry
}

end probability_more_ones_than_sixes_l199_199769


namespace function_range_l199_199233

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x^2 - 1) * (x^2 + a * x + b)

theorem function_range (a b : ℝ) (h_symm : ∀ x : ℝ, f (6 - x) a b = f x a b) :
  a = -12 ∧ b = 35 ∧ (∀ y, ∃ x : ℝ, f x (-12) 35 = y ↔ -36 ≤ y) :=
by
  sorry

end function_range_l199_199233


namespace solution_set_bf_x2_solution_set_g_l199_199142

def f (x : ℝ) := x^2 - 5 * x + 6

theorem solution_set_bf_x2 (x : ℝ) : (2 < x ∧ x < 3) ↔ f x < 0 := sorry

noncomputable def g (x : ℝ) := 6 * x^2 - 5 * x + 1

theorem solution_set_g (x : ℝ) : (1 / 3 < x ∧ x < 1 / 2) ↔ g x < 0 := sorry

end solution_set_bf_x2_solution_set_g_l199_199142


namespace function_is_linear_l199_199395

theorem function_is_linear (f : ℝ → ℝ) :
  (∀ a b c d : ℝ,
    a ≠ b → b ≠ c → c ≠ d → d ≠ a →
    (a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ b ≠ c) →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d) →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ m c : ℝ, ∀ x : ℝ, f x = m * x + c :=
by
  sorry

end function_is_linear_l199_199395


namespace trapezoid_area_l199_199877

theorem trapezoid_area (h : ℝ) : 
  let base1 := 3 * h 
  let base2 := 4 * h 
  let average_base := (base1 + base2) / 2 
  let area := average_base * h 
  area = (7 * h^2) / 2 := 
by
  sorry

end trapezoid_area_l199_199877


namespace number_of_integers_with_abs_le_4_l199_199163

theorem number_of_integers_with_abs_le_4 : 
  ∃ (S : Set Int), (∀ x ∈ S, |x| ≤ 4) ∧ S.card = 9 :=
by
  let S := {x : Int | |x| ≤ 4}
  use S
  have h1: ∀ x ∈ S, |x| ≤ 4 := by
    intros x hx
    exact hx
  have h2: S.card = 9 := sorry
  exact ⟨h1, h2⟩

end number_of_integers_with_abs_le_4_l199_199163


namespace proof_goats_minus_pigs_l199_199383

noncomputable def number_of_goats : ℕ := 66
noncomputable def number_of_chickens : ℕ := 2 * number_of_goats - 10
noncomputable def number_of_ducks : ℕ := (number_of_goats + number_of_chickens) / 2
noncomputable def number_of_pigs : ℕ := number_of_ducks / 3
noncomputable def number_of_rabbits : ℕ := Nat.floor (Real.sqrt (2 * number_of_ducks - number_of_pigs))
noncomputable def number_of_cows : ℕ := number_of_rabbits ^ number_of_pigs / Nat.factorial (number_of_goats / 2)

theorem proof_goats_minus_pigs : number_of_goats - number_of_pigs = 35 := by
  sorry

end proof_goats_minus_pigs_l199_199383


namespace tangent_slope_of_circle_l199_199321

theorem tangent_slope_of_circle {x1 y1 x2 y2 : ℝ}
  (hx1 : x1 = 1) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 4) :
  ∀ m : ℝ, m = -5 / 3 ↔
    (∃ (r : ℝ), r = (y2 - y1) / (x2 - x1) ∧ m = -1 / r) :=
by
  sorry

end tangent_slope_of_circle_l199_199321


namespace evaluate_expression_l199_199401

-- Definitions based on conditions
def a : ℤ := 5
def b : ℤ := -3
def c : ℤ := 2

-- Theorem to be proved: evaluate the expression
theorem evaluate_expression : (3 : ℚ) / (a + b + c) = 3 / 4 := by
  sorry

end evaluate_expression_l199_199401


namespace coordinates_of_point_l199_199784

noncomputable def point_on_x_axis (x : ℝ) :=
  (x, 0)

theorem coordinates_of_point (x : ℝ) (hx : abs x = 3) :
  point_on_x_axis x = (3, 0) ∨ point_on_x_axis x = (-3, 0) :=
  sorry

end coordinates_of_point_l199_199784


namespace solution_set_of_inequality_l199_199829

theorem solution_set_of_inequality : 
  {x : ℝ | x < x^2} = {x | x < 0} ∪ {x | x > 1} :=
by sorry

end solution_set_of_inequality_l199_199829


namespace find_q_l199_199738

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l199_199738


namespace iterate_g_eq_2_l199_199701

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

theorem iterate_g_eq_2 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 100): 
  (∃ m : ℕ, (Nat.iterate g m n) = 2) ↔ n = 1 :=
by
sorry

end iterate_g_eq_2_l199_199701


namespace cups_filled_with_tea_l199_199313

theorem cups_filled_with_tea (total_tea ml_each_cup : ℕ)
  (h1 : total_tea = 1050)
  (h2 : ml_each_cup = 65) :
  total_tea / ml_each_cup = 16 := sorry

end cups_filled_with_tea_l199_199313


namespace m_perpendicular_beta_l199_199725

variables {Plane : Type*} {Line : Type*}

-- Definitions of the perpendicularity and parallelism
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p1 p2 : Plane) : Prop := sorry

-- Given variables
variables (α β : Plane) (m : Line)

-- Conditions
axiom M_perpendicular_Alpha : perpendicular m α
axiom Alpha_parallel_Beta : parallel α β

-- Proof goal
theorem m_perpendicular_beta 
  (h1 : perpendicular m α) 
  (h2 : parallel α β) : 
  perpendicular m β := 
  sorry

end m_perpendicular_beta_l199_199725


namespace linear_eq_k_l199_199116

theorem linear_eq_k (k : ℝ) : (k - 3) * x ^ (|k| - 2) + 5 = k - 4 → |k| = 3 → k ≠ 3 → k = -3 :=
by
  intros h1 h2 h3
  sorry

end linear_eq_k_l199_199116


namespace sin_add_alpha_l199_199223

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha_l199_199223


namespace sin_theta_l199_199467

open Real

variables {a b c : E ℝ 3}

noncomputable def angle_between (b c : E ℝ 3) : ℝ :=
  acos ((b ⬝ c) / (∥b∥ * ∥c∥))

theorem sin_theta (a b c : E ℝ 3) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
(h4 : ¬Collinear ℝ ({a, b, c} : Set (E ℝ 3)))
(h5 : (a × b) × c = -((1 / 2) * (∥b∥ * ∥c∥) • a)) :
  sin (angle_between b c) = sqrt 3 / 2 :=
sorry

end sin_theta_l199_199467


namespace fewest_tiles_needed_l199_199196

theorem fewest_tiles_needed 
  (tile_len : ℝ) (tile_wid : ℝ) (region_len : ℝ) (region_wid : ℝ)
  (h_tile_dims : tile_len = 2 ∧ tile_wid = 3)
  (h_region_dims : region_len = 48 ∧ region_wid = 72) :
  (region_len * region_wid) / (tile_len * tile_wid) = 576 :=
by {
  sorry
}

end fewest_tiles_needed_l199_199196


namespace inequality_proof_l199_199921

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end inequality_proof_l199_199921


namespace number_of_dimes_l199_199717

-- Definitions based on conditions
def total_coins : Nat := 28
def nickels : Nat := 4

-- Definition of the number of dimes.
def dimes : Nat := total_coins - nickels

-- Theorem statement with the expected answer
theorem number_of_dimes : dimes = 24 := by
  -- Proof is skipped with sorry
  sorry

end number_of_dimes_l199_199717


namespace expand_product_l199_199709

variable (x : ℝ)

theorem expand_product :
  (x + 3) * (x^2 + 4 * x + 6) = x^3 + 7 * x^2 + 18 * x + 18 := 
  sorry

end expand_product_l199_199709


namespace hyperbola_focus_l199_199732

theorem hyperbola_focus :
    ∃ (f : ℝ × ℝ), f = (-2 - Real.sqrt 6, -2) ∧
    ∀ (x y : ℝ), 2 * x^2 - y^2 + 8 * x - 4 * y - 8 = 0 → 
    ∃ a b h k : ℝ, 
        (a = Real.sqrt 2) ∧ (b = 2) ∧ (h = -2) ∧ (k = -2) ∧
        ((2 * (x + h)^2 - (y + k)^2 = 4) ∧ 
         (x, y) = f) :=
sorry

end hyperbola_focus_l199_199732


namespace tangent_line_on_x_axis_l199_199115

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 1/4

theorem tangent_line_on_x_axis (x0 a : ℝ) (h1: f x0 a = 0) (h2: (3 * x0^2 + a) = 0) : a = -3/4 :=
by sorry

end tangent_line_on_x_axis_l199_199115


namespace smaller_root_of_quadratic_l199_199908

theorem smaller_root_of_quadratic :
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (x₁^2 - 14 * x₁ + 45 = 0) ∧ (x₂^2 - 14 * x₂ + 45 = 0) ∧ (min x₁ x₂ = 5) :=
sorry

end smaller_root_of_quadratic_l199_199908


namespace geometric_sequence_common_ratio_l199_199802

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_common_ratio (a₁ : ℝ) (q : ℝ) :
  8 * geometric_sum a₁ q 6 = 7 * geometric_sum a₁ q 3 →
  q = -1/2 :=
by
  sorry

end geometric_sequence_common_ratio_l199_199802


namespace ganesh_speed_x_to_y_l199_199080

-- Define the conditions
variables (D : ℝ) (V : ℝ)

-- Theorem statement: Prove that Ganesh's average speed from x to y is 44 km/hr
theorem ganesh_speed_x_to_y
  (H1 : 39.6 = 2 * D / (D / V + D / 36))
  (H2 : V = 44) :
  true :=
sorry

end ganesh_speed_x_to_y_l199_199080


namespace cake_area_l199_199170

theorem cake_area (n : ℕ) (a area_per_piece : ℕ) 
  (h1 : n = 25) 
  (h2 : a = 16) 
  (h3 : area_per_piece = 4 * 4) 
  (h4 : a = area_per_piece) : 
  n * a = 400 := 
by
  sorry

end cake_area_l199_199170


namespace geometric_sequence_common_ratio_l199_199800

theorem geometric_sequence_common_ratio (a₁ : ℚ) (q : ℚ) 
  (S : ℕ → ℚ) (hS : ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) 
  (h : 8 * S 6 = 7 * S 3) : 
  q = -1/2 :=
sorry

end geometric_sequence_common_ratio_l199_199800


namespace calculate_total_weight_l199_199471

-- Define the given conditions as constants and calculations
def silverware_weight_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def plate_weight_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Calculate individual weights and total settings
def silverware_weight_per_setting := silverware_weight_per_piece * pieces_per_setting
def plate_weight_per_setting := plate_weight_per_piece * plates_per_setting
def weight_per_setting := silverware_weight_per_setting + plate_weight_per_setting
def total_settings := (tables * settings_per_table) + backup_settings

-- Calculate the total weight of all settings
def total_weight : ℕ := total_settings * weight_per_setting

-- The theorem to prove that the total weight is 5040 ounces
theorem calculate_total_weight : total_weight = 5040 :=
by
  -- The proof steps are omitted
  sorry

end calculate_total_weight_l199_199471


namespace probability_more_ones_than_sixes_l199_199763

theorem probability_more_ones_than_sixes :
  (∃ (prob : ℚ), prob = 223 / 648) :=
by
  -- conditions:
  -- let dice := {1, 2, 3, 4, 5, 6}
  
  -- question:
  -- the desired probability is provable to be 223 / 648
  
  have probability : ℚ := 223 / 648,
  use probability,
  sorry

end probability_more_ones_than_sixes_l199_199763


namespace Bob_walked_35_miles_l199_199521

theorem Bob_walked_35_miles (distance : ℕ) 
  (Yolanda_rate Bob_rate : ℕ) (Bob_start_after : ℕ) (Yolanda_initial_walk : ℕ)
  (h1 : distance = 65) 
  (h2 : Yolanda_rate = 5) 
  (h3 : Bob_rate = 7) 
  (h4 : Bob_start_after = 1)
  (h5 : Yolanda_initial_walk = Yolanda_rate * Bob_start_after) :
  Bob_rate * (distance - Yolanda_initial_walk) / (Yolanda_rate + Bob_rate) = 35 := 
by 
  sorry

end Bob_walked_35_miles_l199_199521


namespace candies_remaining_l199_199342

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l199_199342


namespace find_a_l199_199623

open Set

theorem find_a (a : ℝ) : let A := {-4, 2 * a - 1, a^2}
                        let B := {a - 1, 1 - a, 9}
                        (A ∩ B = {9}) → a = 3 :=
by
  intro A B h
  sorry

end find_a_l199_199623


namespace no_nat_num_divisible_l199_199132

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end no_nat_num_divisible_l199_199132


namespace candies_remaining_l199_199343

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l199_199343


namespace P_B_given_A_correct_l199_199286

def die_faces := {1, 2, 3, 4, 5, 6}

noncomputable def P_B_given_A : Rational := 
  let event_A : Set (ℕ × ℕ) := {⟨x, y⟩ | x ∈ die_faces ∧ y ∈ die_faces ∧ (x + y) % 2 = 0}
  let event_B : Set (ℕ × ℕ) := {⟨x, y⟩ | x ∈ die_faces ∧ y ∈ die_faces ∧ (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y}
  let P_A := Rational.mk 1 2 -- Probability of event A
  let P_AB := Rational.mk 1 6 -- Probability of events A and B occurring simultaneously
  let P_B_given_A := P_AB / P_A
  P_B_given_A

theorem P_B_given_A_correct : P_B_given_A = 1 / 3 :=
by sorry

end P_B_given_A_correct_l199_199286


namespace probability_XOXOXOX_is_1_div_35_l199_199577

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l199_199577


namespace spring_compression_l199_199051

theorem spring_compression (s F : ℝ) (h : F = 16 * s^2) (hF : F = 4) : s = 0.5 :=
by
  sorry

end spring_compression_l199_199051


namespace profit_percentage_correct_l199_199190

def SP : ℝ := 900
def P : ℝ := 100

theorem profit_percentage_correct : (P / (SP - P)) * 100 = 12.5 := sorry

end profit_percentage_correct_l199_199190


namespace at_least_one_not_less_than_neg_two_l199_199464

theorem at_least_one_not_less_than_neg_two (a b c : ℝ) (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1/b ≥ -2 ∨ b + 1/c ≥ -2 ∨ c + 1/a ≥ -2) :=
sorry

end at_least_one_not_less_than_neg_two_l199_199464


namespace find_c_for_minimum_value_l199_199422

-- Definitions based on the conditions
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Main statement to be proved
theorem find_c_for_minimum_value (c : ℝ) : (∀ x, (3*x^2 - 4*c*x + c^2) = 0) → c = 3 :=
by
  sorry

end find_c_for_minimum_value_l199_199422


namespace box_one_contains_at_least_one_ball_l199_199814

-- Define the conditions
def boxes : List ℕ := [1, 2, 3, 4]
def balls : List ℕ := [1, 2, 3]

-- Define the problem
def count_ways_box_one_contains_ball :=
  let total_ways := (boxes.length)^(balls.length)
  let ways_box_one_empty := (boxes.length - 1)^(balls.length)
  total_ways - ways_box_one_empty

-- The proof problem statement
theorem box_one_contains_at_least_one_ball : count_ways_box_one_contains_ball = 37 := by
  sorry

end box_one_contains_at_least_one_ball_l199_199814


namespace resistor_value_l199_199165

/-- Two resistors with resistance R are connected in series to a DC voltage source U.
    An ideal voltmeter connected in parallel to one resistor shows a reading of 10V.
    The voltmeter is then replaced by an ideal ammeter, which shows a reading of 10A.
    Prove that the resistance R of each resistor is 2Ω. -/
theorem resistor_value (R U U_v I_A : ℝ)
  (hU_v : U_v = 10)
  (hI_A : I_A = 10)
  (hU : U = 2 * U_v)
  (hU_total : U = R * I_A) : R = 2 :=
by
  sorry

end resistor_value_l199_199165
