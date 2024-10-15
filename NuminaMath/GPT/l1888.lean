import Mathlib

namespace NUMINAMATH_GPT_grade_assignment_ways_l1888_188877

-- Definitions
def num_students : ℕ := 10
def num_choices_per_student : ℕ := 3

-- Theorem statement
theorem grade_assignment_ways : num_choices_per_student ^ num_students = 59049 := by
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l1888_188877


namespace NUMINAMATH_GPT_percentage_chromium_first_alloy_l1888_188891

theorem percentage_chromium_first_alloy 
  (x : ℝ) (w1 w2 : ℝ) (p2 p_new : ℝ) 
  (h1 : w1 = 10) 
  (h2 : w2 = 30) 
  (h3 : p2 = 0.08)
  (h4 : p_new = 0.09):
  ((x / 100) * w1 + p2 * w2) = p_new * (w1 + w2) → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_percentage_chromium_first_alloy_l1888_188891


namespace NUMINAMATH_GPT_sequence_sum_consecutive_l1888_188895

theorem sequence_sum_consecutive 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 20) 
  (h8 : a 8 = 16) 
  (h_sum : ∀ i, 1 ≤ i ∧ i ≤ 6 → a i + a (i+1) + a (i+2) = 100) :
  a 2 = 16 ∧ a 3 = 64 ∧ a 4 = 20 ∧ a 5 = 16 ∧ a 6 = 64 ∧ a 7 = 20 :=
  sorry

end NUMINAMATH_GPT_sequence_sum_consecutive_l1888_188895


namespace NUMINAMATH_GPT_cost_price_of_apple_l1888_188859

theorem cost_price_of_apple (C : ℚ) (h1 : 19 = 5/6 * C) : C = 22.8 := by
  sorry

end NUMINAMATH_GPT_cost_price_of_apple_l1888_188859


namespace NUMINAMATH_GPT_total_carrots_l1888_188871

-- Define constants for the number of carrots grown by each person
def Joan_carrots : ℕ := 29
def Jessica_carrots : ℕ := 11
def Michael_carrots : ℕ := 37
def Taylor_carrots : ℕ := 24

-- The proof problem: Prove that the total number of carrots grown is 101
theorem total_carrots : Joan_carrots + Jessica_carrots + Michael_carrots + Taylor_carrots = 101 :=
by
  sorry

end NUMINAMATH_GPT_total_carrots_l1888_188871


namespace NUMINAMATH_GPT_binomial_sum_zero_l1888_188856

open BigOperators

theorem binomial_sum_zero {n m : ℕ} (h1 : 1 ≤ m) (h2 : m < n) :
  ∑ k in Finset.range (n + 1), (-1 : ℤ) ^ k * k ^ m * Nat.choose n k = 0 :=
by
  sorry

end NUMINAMATH_GPT_binomial_sum_zero_l1888_188856


namespace NUMINAMATH_GPT_find_real_a_l1888_188847

theorem find_real_a (a : ℝ) : 
  (a ^ 2 + 2 * a - 15 = 0) ∧ (a ^ 2 + 4 * a - 5 ≠ 0) → a = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_real_a_l1888_188847


namespace NUMINAMATH_GPT_relationship_between_y1_y2_l1888_188809

variable (k b y1 y2 : ℝ)

-- Let A = (-3, y1) and B = (4, y2) be points on the line y = kx + b, with k < 0
axiom A_on_line : y1 = k * -3 + b
axiom B_on_line : y2 = k * 4 + b
axiom k_neg : k < 0

theorem relationship_between_y1_y2 : y1 > y2 :=
by sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_l1888_188809


namespace NUMINAMATH_GPT_function_inequality_m_l1888_188802

theorem function_inequality_m (m : ℝ) : (∀ x : ℝ, (1 / 2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) ↔ m ≥ (3 / 2) := sorry

end NUMINAMATH_GPT_function_inequality_m_l1888_188802


namespace NUMINAMATH_GPT_hyperbola_eqn_l1888_188850

theorem hyperbola_eqn
  (P : ℝ × ℝ) (Q : ℝ × ℝ)
  (C1 : P = (-3, 2 * Real.sqrt 7))
  (C2 : Q = (-6 * Real.sqrt 2, -7))
  (asymptote_hyperbola : ∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1)
  (special_point : ℝ × ℝ)
  (C3 : special_point = (2, 2 * Real.sqrt 3)) :
  ∃ (a b : ℝ), ¬(a = 0) ∧ ¬(b = 0) ∧ 
  (∀ x y : ℝ, (y^2 / b - x^2 / a = 1 → 
    ((y^2 / 25 - x^2 / 75 = 1) ∨ 
    (y^2 / 9 - x^2 / 12 = 1)))) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eqn_l1888_188850


namespace NUMINAMATH_GPT_min_value_a_b_inv_a_inv_b_l1888_188822

theorem min_value_a_b_inv_a_inv_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_a_b_inv_a_inv_b_l1888_188822


namespace NUMINAMATH_GPT_repeating_decimals_sum_l1888_188883

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end NUMINAMATH_GPT_repeating_decimals_sum_l1888_188883


namespace NUMINAMATH_GPT_total_respondents_l1888_188881

theorem total_respondents (X Y : ℕ) (h1 : X = 60) (h2 : 3 * Y = X) : X + Y = 80 :=
by
  sorry

end NUMINAMATH_GPT_total_respondents_l1888_188881


namespace NUMINAMATH_GPT_length_of_first_train_solution_l1888_188834

noncomputable def length_of_first_train (speed1_kmph speed2_kmph : ℝ) (length2_m time_s : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * (5 / 18)
  let speed2_mps := speed2_kmph * (5 / 18)
  let relative_speed_mps := speed1_mps + speed2_mps
  let combined_length_m := relative_speed_mps * time_s
  combined_length_m - length2_m

theorem length_of_first_train_solution 
  (speed1_kmph : ℝ) 
  (speed2_kmph : ℝ) 
  (length2_m : ℝ) 
  (time_s : ℝ) 
  (h₁ : speed1_kmph = 42) 
  (h₂ : speed2_kmph = 30) 
  (h₃ : length2_m = 120) 
  (h₄ : time_s = 10.999120070394369) : 
  length_of_first_train speed1_kmph speed2_kmph length2_m time_s = 99.98 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_first_train_solution_l1888_188834


namespace NUMINAMATH_GPT_ellipse_sum_l1888_188828

theorem ellipse_sum (h k a b : ℤ) (h_val : h = 3) (k_val : k = -5) (a_val : a = 7) (b_val : b = 4) : 
  h + k + a + b = 9 :=
by
  rw [h_val, k_val, a_val, b_val]
  norm_num

end NUMINAMATH_GPT_ellipse_sum_l1888_188828


namespace NUMINAMATH_GPT_problem_l1888_188842

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1) 
  : f 2019 = -1 := 
sorry

end NUMINAMATH_GPT_problem_l1888_188842


namespace NUMINAMATH_GPT_find_vanessa_age_l1888_188876

/-- Define the initial conditions and goal -/
theorem find_vanessa_age (V : ℕ) (Kevin_age current_time future_time : ℕ) :
  Kevin_age = 16 ∧ future_time = current_time + 5 ∧
  (Kevin_age + future_time - current_time) = 3 * (V + future_time - current_time) →
  V = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_vanessa_age_l1888_188876


namespace NUMINAMATH_GPT_initial_oranges_count_l1888_188800

theorem initial_oranges_count 
  (O : ℕ)
  (h1 : 10 = O - 13) : 
  O = 23 := 
sorry

end NUMINAMATH_GPT_initial_oranges_count_l1888_188800


namespace NUMINAMATH_GPT_total_money_divided_l1888_188843

theorem total_money_divided (x y : ℕ) (hx : x = 1000) (ratioxy : 2 * y = 8 * x) : x + y = 5000 := 
by
  sorry

end NUMINAMATH_GPT_total_money_divided_l1888_188843


namespace NUMINAMATH_GPT_turtles_remaining_on_log_l1888_188824

-- Definition of the problem parameters
def initial_turtles : ℕ := 9
def additional_turtles : ℕ := (3 * initial_turtles) - 2
def total_turtles : ℕ := initial_turtles + additional_turtles
def frightened_turtles : ℕ := total_turtles / 2
def remaining_turtles : ℕ := total_turtles - frightened_turtles

-- The final theorem stating the number of turtles remaining
theorem turtles_remaining_on_log : remaining_turtles = 17 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_turtles_remaining_on_log_l1888_188824


namespace NUMINAMATH_GPT_problem_statement_l1888_188894

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^4 + 2*m^3 - m + 2007 = 2007 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1888_188894


namespace NUMINAMATH_GPT_skateboard_weight_is_18_l1888_188803

def weight_of_canoe : Nat := 45
def weight_of_four_canoes := 4 * weight_of_canoe
def weight_of_ten_skateboards := weight_of_four_canoes
def weight_of_one_skateboard := weight_of_ten_skateboards / 10

theorem skateboard_weight_is_18 : weight_of_one_skateboard = 18 := by
  sorry

end NUMINAMATH_GPT_skateboard_weight_is_18_l1888_188803


namespace NUMINAMATH_GPT_angle_sum_impossible_l1888_188846

theorem angle_sum_impossible (A1 A2 A3 : ℝ) (h : A1 + A2 + A3 = 180) :
  ¬ ((A1 > 90 ∧ A2 > 90 ∧ A3 < 90) ∨ (A1 > 90 ∧ A3 > 90 ∧ A2 < 90) ∨ (A2 > 90 ∧ A3 > 90 ∧ A1 < 90)) :=
sorry

end NUMINAMATH_GPT_angle_sum_impossible_l1888_188846


namespace NUMINAMATH_GPT_find_x_l1888_188848

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1888_188848


namespace NUMINAMATH_GPT_total_surface_area_of_modified_cube_l1888_188830

-- Define the side length of the original cube
def side_length_cube := 3

-- Define the side length of the holes
def side_length_hole := 1

-- Define the condition of the surface area calculation
def total_surface_area_including_internal (side_length_cube side_length_hole : ℕ) : ℕ :=
  let original_surface_area := 6 * (side_length_cube * side_length_cube)
  let reduction_area := 6 * (side_length_hole * side_length_hole)
  let remaining_surface_area := original_surface_area - reduction_area
  let interior_surface_area := 6 * (4 * side_length_hole * side_length_cube)
  remaining_surface_area + interior_surface_area

-- Statement for the proof
theorem total_surface_area_of_modified_cube : total_surface_area_including_internal 3 1 = 72 :=
by
  -- This is the statement; the proof is omitted as "sorry"
  sorry

end NUMINAMATH_GPT_total_surface_area_of_modified_cube_l1888_188830


namespace NUMINAMATH_GPT_parabolic_arch_height_l1888_188831

noncomputable def arch_height (a : ℝ) : ℝ :=
  a * (0 : ℝ)^2

theorem parabolic_arch_height :
  ∃ (a : ℝ), (∫ x in (-4 : ℝ)..4, a * x^2) = (160 : ℝ) ∧ arch_height a = 30 :=
by
  sorry

end NUMINAMATH_GPT_parabolic_arch_height_l1888_188831


namespace NUMINAMATH_GPT_count_valid_triples_l1888_188898

def S (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def satisfies_conditions (a b c : ℕ) : Prop :=
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c ∧ 
  (a + b + c = 2005) ∧ (S a + S b + S c = 61)

def number_of_valid_triples : ℕ := sorry

theorem count_valid_triples : number_of_valid_triples = 17160 :=
sorry

end NUMINAMATH_GPT_count_valid_triples_l1888_188898


namespace NUMINAMATH_GPT_base_of_first_term_l1888_188811

theorem base_of_first_term (e : ℕ) (b : ℝ) (h : e = 35) :
  b^e * (1/4)^18 = 1/(2 * 10^35) → b = 1/5 :=
by
  sorry

end NUMINAMATH_GPT_base_of_first_term_l1888_188811


namespace NUMINAMATH_GPT_quadratic_function_vertex_form_l1888_188840

theorem quadratic_function_vertex_form :
  ∃ f : ℝ → ℝ, (∀ x, f x = (x - 2)^2 - 2) ∧ (f 0 = 2) ∧ (∀ x, f x = a * (x - 2)^2 - 2 → a = 1) := by
  sorry

end NUMINAMATH_GPT_quadratic_function_vertex_form_l1888_188840


namespace NUMINAMATH_GPT_focus_of_given_parabola_is_correct_l1888_188888

-- Define the problem conditions
def parabolic_equation (x y : ℝ) : Prop := y = 4 * x^2

-- Define what it means for a point to be the focus of the given parabola
def is_focus_of_parabola (x0 y0 : ℝ) : Prop := 
    x0 = 0 ∧ y0 = 1 / 16

-- Define the theorem to be proven
theorem focus_of_given_parabola_is_correct : 
  ∃ x0 y0, parabolic_equation x0 y0 ∧ is_focus_of_parabola x0 y0 :=
sorry

end NUMINAMATH_GPT_focus_of_given_parabola_is_correct_l1888_188888


namespace NUMINAMATH_GPT_total_trees_after_planting_l1888_188813

theorem total_trees_after_planting
  (initial_walnut_trees : ℕ) (initial_oak_trees : ℕ) (initial_maple_trees : ℕ)
  (plant_walnut_trees : ℕ) (plant_oak_trees : ℕ) (plant_maple_trees : ℕ) :
  (initial_walnut_trees = 107) →
  (initial_oak_trees = 65) →
  (initial_maple_trees = 32) →
  (plant_walnut_trees = 104) →
  (plant_oak_trees = 79) →
  (plant_maple_trees = 46) →
  initial_walnut_trees + plant_walnut_trees +
  initial_oak_trees + plant_oak_trees +
  initial_maple_trees + plant_maple_trees = 433 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_trees_after_planting_l1888_188813


namespace NUMINAMATH_GPT_morgan_change_l1888_188837

theorem morgan_change:
  let hamburger := 5.75
  let onion_rings := 2.50
  let smoothie := 3.25
  let side_salad := 3.75
  let cake := 4.20
  let total_cost := hamburger + onion_rings + smoothie + side_salad + cake
  let payment := 50
  let change := payment - total_cost
  ℝ := by
    exact sorry

end NUMINAMATH_GPT_morgan_change_l1888_188837


namespace NUMINAMATH_GPT_closing_price_l1888_188832

theorem closing_price
  (opening_price : ℝ)
  (increase_percentage : ℝ)
  (h_opening_price : opening_price = 15)
  (h_increase_percentage : increase_percentage = 6.666666666666665) :
  opening_price * (1 + increase_percentage / 100) = 16 :=
by
  sorry

end NUMINAMATH_GPT_closing_price_l1888_188832


namespace NUMINAMATH_GPT_sum_possible_n_k_l1888_188853

theorem sum_possible_n_k (n k : ℕ) (h1 : 3 * k + 3 = n) (h2 : 5 * (k + 2) = 3 * (n - k - 1)) : 
  n + k = 19 := 
by {
  sorry -- proof steps would go here
}

end NUMINAMATH_GPT_sum_possible_n_k_l1888_188853


namespace NUMINAMATH_GPT_number_of_even_factors_of_n_l1888_188838

noncomputable def n := 2^3 * 3^2 * 7^3

theorem number_of_even_factors_of_n : 
  (∃ (a : ℕ), (1 ≤ a ∧ a ≤ 3)) ∧ 
  (∃ (b : ℕ), (0 ≤ b ∧ b ≤ 2)) ∧ 
  (∃ (c : ℕ), (0 ≤ c ∧ c ≤ 3)) → 
  (even_nat_factors_count : ℕ) = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_even_factors_of_n_l1888_188838


namespace NUMINAMATH_GPT_num_real_numbers_l1888_188833

theorem num_real_numbers (x : ℝ) : 
  (∃ n : ℕ, n ≤ 11 ∧ x = (123 - n^2)^2) ↔ 
  (∃ s : ℕ, 0 ≤ s ∧ s ≤ 11 ∧ (x = (123 - s^2)^2) ∧ 
  (∃! x, ∃ k : ℕ, 0 ≤ k ∧ k ≤ 11 ∧ x = (123 - k^2)^2)) :=
by
  sorry

end NUMINAMATH_GPT_num_real_numbers_l1888_188833


namespace NUMINAMATH_GPT_no_such_reals_exist_l1888_188872

-- Define the existence of distinct real numbers such that the given condition holds
theorem no_such_reals_exist :
  ¬ ∃ x y z : ℝ, (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧ 
  (1 / (x^2 - y^2) + 1 / (y^2 - z^2) + 1 / (z^2 - x^2) = 0) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_no_such_reals_exist_l1888_188872


namespace NUMINAMATH_GPT_range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l1888_188835

theorem range_of_x_if_p_and_q_true (a : ℝ) (p q : ℝ → Prop) (h_a : a = 1) (h_p : ∀ x, p x ↔ x^2 - 4*a*x + 3*a^2 < 0)
  (h_q : ∀ x, q x ↔ (x-3)^2 < 1) (h_pq : ∀ x, p x ∧ q x) :
  ∀ x, 2 < x ∧ x < 3 :=
by
  sorry

theorem range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q (p q : ℝ → Prop) (h_neg : ∀ x, ¬p x → ¬q x) : 
  ∀ a : ℝ, a > 0 → (a ≥ 4/3 ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_if_p_and_q_true_range_of_a_if_neg_p_necessary_not_sufficient_for_neg_q_l1888_188835


namespace NUMINAMATH_GPT_cubic_polynomial_inequality_l1888_188879

theorem cubic_polynomial_inequality
  (A B C : ℝ)
  (h : ∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    A = -(a + b + c) ∧ B = ab + bc + ca ∧ C = -abc) :
  A^2 + B^2 + 18 * C > 0 :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_inequality_l1888_188879


namespace NUMINAMATH_GPT_john_read_bible_in_weeks_l1888_188836

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end NUMINAMATH_GPT_john_read_bible_in_weeks_l1888_188836


namespace NUMINAMATH_GPT_possible_values_of_5x_plus_2_l1888_188892

theorem possible_values_of_5x_plus_2 (x : ℝ) :
  (x - 4) * (5 * x + 2) = 0 →
  (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_possible_values_of_5x_plus_2_l1888_188892


namespace NUMINAMATH_GPT_max_distance_from_earth_to_sun_l1888_188849

-- Assume the semi-major axis 'a' and semi-minor axis 'b' specified in the problem.
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_minor_axis : ℝ := 3 * 10^6

-- Define the theorem stating the maximum distance from the Earth to the Sun.
theorem max_distance_from_earth_to_sun :
  let a := semi_major_axis
  let b := semi_minor_axis
  a + b = 1.53 * 10^8 :=
by
  -- Proof will be completed
  sorry

end NUMINAMATH_GPT_max_distance_from_earth_to_sun_l1888_188849


namespace NUMINAMATH_GPT_number_of_cities_from_group_B_l1888_188826

theorem number_of_cities_from_group_B
  (total_cities : ℕ)
  (cities_in_A : ℕ)
  (cities_in_B : ℕ)
  (cities_in_C : ℕ)
  (sampled_cities : ℕ)
  (h1 : total_cities = cities_in_A + cities_in_B + cities_in_C)
  (h2 : total_cities = 24)
  (h3 : cities_in_A = 4)
  (h4 : cities_in_B = 12)
  (h5 : cities_in_C = 8)
  (h6 : sampled_cities = 6) :
  cities_in_B * sampled_cities / total_cities = 3 := 
  by 
    sorry

end NUMINAMATH_GPT_number_of_cities_from_group_B_l1888_188826


namespace NUMINAMATH_GPT_minimize_sum_of_digits_l1888_188841

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Define the expression in the problem
def expression (p : ℕ) : ℕ :=
  p^4 - 5 * p^2 + 13

-- Proposition stating the conditions and the expected result
theorem minimize_sum_of_digits (p : ℕ) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∀ q : ℕ, Nat.Prime q → q % 2 = 1 → sum_of_digits (expression q) ≥ sum_of_digits (expression 5)) →
  p = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_of_digits_l1888_188841


namespace NUMINAMATH_GPT_john_february_bill_l1888_188897

-- Define the conditions as constants
def base_cost : ℝ := 25
def cost_per_text : ℝ := 0.1 -- 10 cents
def cost_per_over_minute : ℝ := 0.1 -- 10 cents
def texts_sent : ℝ := 200
def hours_talked : ℝ := 51
def included_hours : ℝ := 50
def minutes_per_hour : ℝ := 60

-- Total cost computation
def total_cost : ℝ :=
  base_cost +
  (texts_sent * cost_per_text) +
  ((hours_talked - included_hours) * minutes_per_hour * cost_per_over_minute)

-- Proof statement
theorem john_february_bill : total_cost = 51 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_john_february_bill_l1888_188897


namespace NUMINAMATH_GPT_division_problem_l1888_188893

theorem division_problem 
  (a b c d e f g h i : ℕ) 
  (h1 : a = 7) 
  (h2 : b = 9) 
  (h3 : c = 8) 
  (h4 : d = 1) 
  (h5 : e = 2) 
  (h6 : f = 3) 
  (h7 : g = 4) 
  (h8 : h = 6) 
  (h9 : i = 0) 
  : 7981 / 23 = 347 := 
by 
  sorry

end NUMINAMATH_GPT_division_problem_l1888_188893


namespace NUMINAMATH_GPT_problem1_problem2_l1888_188852

theorem problem1 (x : ℕ) : 
  2 / 8^x * 16^x = 2^5 → x = 4 := 
by
  sorry

theorem problem2 (x : ℕ) : 
  2^(x+2) + 2^(x+1) = 24 → x = 2 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1888_188852


namespace NUMINAMATH_GPT_correct_propositions_l1888_188860

-- Definitions of the conditions in the Math problem

variable (triangle_outside_plane : Prop)
variable (triangle_side_intersections_collinear : Prop)
variable (parallel_lines_coplanar : Prop)
variable (noncoplanar_points_planes : Prop)

-- Math proof problem statement
theorem correct_propositions :
  (triangle_outside_plane ∧ 
   parallel_lines_coplanar ∧ 
   ¬noncoplanar_points_planes) →
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l1888_188860


namespace NUMINAMATH_GPT_ratio_of_areas_l1888_188812

-- Definitions of conditions
def side_length (s : ℝ) : Prop := s > 0
def original_area (A s : ℝ) : Prop := A = s^2

-- Definition of the new area after folding
def new_area (B A s : ℝ) : Prop := B = (7/8) * s^2

-- The proof statement to show the ratio B/A is 7/8
theorem ratio_of_areas (s A B : ℝ) (h_side : side_length s) (h_area : original_area A s) (h_B : new_area B A s) : 
  B / A = 7 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1888_188812


namespace NUMINAMATH_GPT_soaps_in_one_package_l1888_188823

theorem soaps_in_one_package (boxes : ℕ) (packages_per_box : ℕ) (total_packages : ℕ) (total_soaps : ℕ) : 
  boxes = 2 → packages_per_box = 6 → total_packages = boxes * packages_per_box → total_soaps = 2304 → (total_soaps / total_packages) = 192 :=
by
  intros h_boxes h_packages_per_box h_total_packages h_total_soaps
  sorry

end NUMINAMATH_GPT_soaps_in_one_package_l1888_188823


namespace NUMINAMATH_GPT_branches_on_main_stem_l1888_188887

theorem branches_on_main_stem (x : ℕ) (h : 1 + x + x^2 = 57) : x = 7 :=
  sorry

end NUMINAMATH_GPT_branches_on_main_stem_l1888_188887


namespace NUMINAMATH_GPT_sum_of_two_relatively_prime_integers_l1888_188827

theorem sum_of_two_relatively_prime_integers (x y : ℕ) : 0 < x ∧ x < 30 ∧ 0 < y ∧ y < 30 ∧
  gcd x y = 1 ∧ x * y + x + y = 119 ∧ x + y = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_relatively_prime_integers_l1888_188827


namespace NUMINAMATH_GPT_binomial_coefficients_sum_l1888_188865

noncomputable def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem binomial_coefficients_sum : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := 
by
  sorry

end NUMINAMATH_GPT_binomial_coefficients_sum_l1888_188865


namespace NUMINAMATH_GPT_original_rice_amount_l1888_188858

theorem original_rice_amount (x : ℝ) 
  (h1 : (x / 2) - 3 = 18) : 
  x = 42 :=
sorry

end NUMINAMATH_GPT_original_rice_amount_l1888_188858


namespace NUMINAMATH_GPT_gcd_40_120_80_l1888_188818

-- Given numbers
def n1 := 40
def n2 := 120
def n3 := 80

-- The problem we want to prove:
theorem gcd_40_120_80 : Int.gcd (Int.gcd n1 n2) n3 = 40 := by
  sorry

end NUMINAMATH_GPT_gcd_40_120_80_l1888_188818


namespace NUMINAMATH_GPT_binary_div_four_remainder_l1888_188839

theorem binary_div_four_remainder (n : ℕ) (h : n = 0b111001001101) : n % 4 = 1 := 
sorry

end NUMINAMATH_GPT_binary_div_four_remainder_l1888_188839


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l1888_188807

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the condition for the problem
def condition (x : ℝ) : Prop := -2 < x ∧ x < 3

-- State the proof problem: Prove that the interval is a necessary but not sufficient condition for f(x) < 0
theorem necessary_not_sufficient_condition : 
  ∀ x : ℝ, condition x → ¬ (∀ y : ℝ, condition y → f y < 0) :=
sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l1888_188807


namespace NUMINAMATH_GPT_g_inv_f_7_l1888_188869

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g (x : ℝ) : f_inv (g x) = x^3 - 1
axiom g_exists_inv : ∀ y : ℝ, ∃ x : ℝ, g x = y

theorem g_inv_f_7 : g_inv (f 7) = 2 :=
by
  sorry

end NUMINAMATH_GPT_g_inv_f_7_l1888_188869


namespace NUMINAMATH_GPT_range_of_m_l1888_188806

theorem range_of_m (m : ℝ) :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + m ≤ 0) → 1 < m := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1888_188806


namespace NUMINAMATH_GPT_no_intersection_points_l1888_188885

def intersection_points_eq_zero : Prop :=
∀ x y : ℝ, (y = abs (3 * x + 6)) ∧ (y = -abs (4 * x - 3)) → false

theorem no_intersection_points :
  intersection_points_eq_zero :=
by
  intro x y h
  cases h
  sorry

end NUMINAMATH_GPT_no_intersection_points_l1888_188885


namespace NUMINAMATH_GPT_cups_of_baking_mix_planned_l1888_188875

-- Definitions
def butter_per_cup := 2 -- 2 ounces of butter per 1 cup of baking mix
def coconut_oil_per_butter := 2 -- 2 ounces of coconut oil can substitute 2 ounces of butter
def butter_remaining := 4 -- Chef had 4 ounces of butter
def coconut_oil_used := 8 -- Chef used 8 ounces of coconut oil

-- Statement to be proven
theorem cups_of_baking_mix_planned : 
  (butter_remaining / butter_per_cup) + (coconut_oil_used / coconut_oil_per_butter) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_cups_of_baking_mix_planned_l1888_188875


namespace NUMINAMATH_GPT_maria_sandwich_count_l1888_188899

open Nat

noncomputable def numberOfSandwiches (meat_choices cheese_choices topping_choices : Nat) :=
  (choose meat_choices 2) * (choose cheese_choices 2) * (choose topping_choices 2)

theorem maria_sandwich_count : numberOfSandwiches 12 11 8 = 101640 := by
  sorry

end NUMINAMATH_GPT_maria_sandwich_count_l1888_188899


namespace NUMINAMATH_GPT_price_of_first_tea_x_l1888_188870

theorem price_of_first_tea_x (x : ℝ) :
  let price_second := 135
  let price_third := 173.5
  let avg_price := 152
  let ratio := [1, 1, 2]
  1 * x + 1 * price_second + 2 * price_third = 4 * avg_price -> x = 126 :=
by
  intros price_second price_third avg_price ratio h
  sorry

end NUMINAMATH_GPT_price_of_first_tea_x_l1888_188870


namespace NUMINAMATH_GPT_find_k_l1888_188815

theorem find_k (k : ℝ) :
  (∃ x y : ℝ, y = x + 2 * k ∧ y = 2 * x + k + 1 ∧ x^2 + y^2 = 4) ↔
  (k = 1 ∨ k = -1/5) := 
sorry

end NUMINAMATH_GPT_find_k_l1888_188815


namespace NUMINAMATH_GPT_am_gm_hm_inequality_l1888_188866

variable {x y : ℝ}

-- Conditions: x and y are positive real numbers and x < y
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x < y

-- Proof statement: A.M. > G.M. > H.M. under given conditions
theorem am_gm_hm_inequality (x y : ℝ) (h : conditions x y) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > (2 * x * y) / (x + y) :=
sorry

end NUMINAMATH_GPT_am_gm_hm_inequality_l1888_188866


namespace NUMINAMATH_GPT_decreasing_interval_of_f_l1888_188882

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x)

theorem decreasing_interval_of_f :
  ∀ x y : ℝ, (1 < x ∧ x < y) → f y < f x :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_of_f_l1888_188882


namespace NUMINAMATH_GPT_vector_dot_product_calculation_l1888_188874

theorem vector_dot_product_calculation : 
  let a := (2, 3, -1)
  let b := (2, 0, 3)
  let c := (0, 2, 2)
  (2 * (2 + 0) + 3 * (0 + 2) + -1 * (3 + 2)) = 5 := 
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_calculation_l1888_188874


namespace NUMINAMATH_GPT_fraction_equality_l1888_188867

def at_op (a b : ℕ) : ℕ := a * b - b^2 + b^3
def hash_op (a b : ℕ) : ℕ := a + b - a * b^2 + a * b^3

theorem fraction_equality : 
  ∀ (a b : ℕ), a = 7 → b = 3 → (at_op a b : ℚ) / (hash_op a b : ℚ) = 39 / 136 :=
by
  intros a b h_a h_b
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_fraction_equality_l1888_188867


namespace NUMINAMATH_GPT_S6_is_48_l1888_188814

-- Define the first term and common difference
def a₁ : ℕ := 3
def d : ℕ := 2

-- Define the formula for sum of the first n terms of an arithmetic sequence
def sum_of_arithmetic_sequence (n : ℕ) : ℕ :=
  n / 2 * (2 * a₁ + (n - 1) * d)

-- Prove that the sum of the first 6 terms is 48
theorem S6_is_48 : sum_of_arithmetic_sequence 6 = 48 := by
  sorry

end NUMINAMATH_GPT_S6_is_48_l1888_188814


namespace NUMINAMATH_GPT_xy_identity_l1888_188878

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = 6) : x^2 + y^2 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_xy_identity_l1888_188878


namespace NUMINAMATH_GPT_cement_tesss_street_l1888_188805

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end NUMINAMATH_GPT_cement_tesss_street_l1888_188805


namespace NUMINAMATH_GPT_lcm_12_18_l1888_188873

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_GPT_lcm_12_18_l1888_188873


namespace NUMINAMATH_GPT_ratio_xz_y2_l1888_188844

-- Define the system of equations
def system (k x y z : ℝ) : Prop := 
  x + k * y + 4 * z = 0 ∧ 
  4 * x + k * y - 3 * z = 0 ∧ 
  3 * x + 5 * y - 4 * z = 0

-- Our main theorem to prove the value of xz / y^2 given the system with k = 7.923
theorem ratio_xz_y2 (x y z : ℝ) (h : system 7.923 x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ r : ℝ, r = (x * z) / (y ^ 2) :=
sorry

end NUMINAMATH_GPT_ratio_xz_y2_l1888_188844


namespace NUMINAMATH_GPT_pies_sold_by_mcgee_l1888_188886

/--
If Smith's Bakery sold 70 pies, and they sold 6 more than four times the number of pies that Mcgee's Bakery sold,
prove that Mcgee's Bakery sold 16 pies.
-/
theorem pies_sold_by_mcgee (x : ℕ) (h1 : 4 * x + 6 = 70) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_pies_sold_by_mcgee_l1888_188886


namespace NUMINAMATH_GPT_dog_food_packages_l1888_188810

theorem dog_food_packages
  (packages_cat_food : Nat := 9)
  (cans_per_package_cat_food : Nat := 10)
  (cans_per_package_dog_food : Nat := 5)
  (more_cans_cat_food : Nat := 55)
  (total_cans_cat_food : Nat := packages_cat_food * cans_per_package_cat_food)
  (total_cans_dog_food : Nat := d * cans_per_package_dog_food)
  (h : total_cans_cat_food = total_cans_dog_food + more_cans_cat_food) :
  d = 7 :=
by
  sorry

end NUMINAMATH_GPT_dog_food_packages_l1888_188810


namespace NUMINAMATH_GPT_not_collinear_C_vector_decomposition_l1888_188880

namespace VectorProof

open Function

structure Vector2 where
  x : ℝ
  y : ℝ

def add (v1 v2 : Vector2) : Vector2 := ⟨v1.x + v2.x, v1.y + v2.y⟩
def scale (c : ℝ) (v : Vector2) : Vector2 := ⟨c * v.x, c * v.y⟩

def collinear (v1 v2 : Vector2) : Prop :=
  ∃ k : ℝ, v2 = scale k v1

def vector_a : Vector2 := ⟨3, 4⟩
def e₁_C : Vector2 := ⟨-1, 2⟩
def e₂_C : Vector2 := ⟨3, -1⟩

theorem not_collinear_C :
  ¬ collinear e₁_C e₂_C :=
sorry

theorem vector_decomposition :
  ∃ (x y : ℝ), vector_a = add (scale x e₁_C) (scale y e₂_C) :=
sorry

end VectorProof

end NUMINAMATH_GPT_not_collinear_C_vector_decomposition_l1888_188880


namespace NUMINAMATH_GPT_find_c_l1888_188819

-- Define that \( r \) and \( s \) are roots of \( 2x^2 - 4x - 5 \)
variables (r s : ℚ)
-- Condition: sum of roots \( r + s = 2 \)
axiom sum_of_roots : r + s = 2
-- Condition: product of roots \( rs = -5/2 \)
axiom product_of_roots : r * s = -5 / 2

-- Definition of \( c \) based on the roots \( r-3 \) and \( s-3 \)
def c : ℚ := (r - 3) * (s - 3)

-- The theorem to be proved
theorem find_c : c = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1888_188819


namespace NUMINAMATH_GPT_common_root_for_permutations_of_coeffs_l1888_188801

theorem common_root_for_permutations_of_coeffs :
  ∀ (a b c d : ℤ), (a = -7 ∨ a = 4 ∨ a = -3 ∨ a = 6) ∧ 
                   (b = -7 ∨ b = 4 ∨ b = -3 ∨ b = 6) ∧
                   (c = -7 ∨ c = 4 ∨ c = -3 ∨ c = 6) ∧
                   (d = -7 ∨ d = 4 ∨ d = -3 ∨ d = 6) ∧
                   (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (a * 1^3 + b * 1^2 + c * 1 + d = 0) :=
by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_common_root_for_permutations_of_coeffs_l1888_188801


namespace NUMINAMATH_GPT_experienced_sailors_monthly_earnings_l1888_188890

theorem experienced_sailors_monthly_earnings :
  let total_sailors : Nat := 17
  let inexperienced_sailors : Nat := 5
  let hourly_wage_inexperienced : Nat := 10
  let workweek_hours : Nat := 60
  let weeks_in_month : Nat := 4
  let experienced_sailors : Nat := total_sailors - inexperienced_sailors
  let hourly_wage_experienced := hourly_wage_inexperienced + (hourly_wage_inexperienced / 5)
  let weekly_earnings_experienced := hourly_wage_experienced * workweek_hours
  let total_weekly_earnings_experienced := weekly_earnings_experienced * experienced_sailors
  let monthly_earnings_experienced := total_weekly_earnings_experienced * weeks_in_month
  monthly_earnings_experienced = 34560 := by
  sorry

end NUMINAMATH_GPT_experienced_sailors_monthly_earnings_l1888_188890


namespace NUMINAMATH_GPT_value_of_4m_plus_2n_l1888_188851

-- Given that the equation 2kx + 2m = 6 - 2x + nk 
-- has a solution independent of k
theorem value_of_4m_plus_2n (m n : ℝ) 
  (h : ∃ x : ℝ, ∀ k : ℝ, 2 * k * x + 2 * m = 6 - 2 * x + n * k) : 
  4 * m + 2 * n = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_4m_plus_2n_l1888_188851


namespace NUMINAMATH_GPT_smallest_d_l1888_188829

theorem smallest_d (d : ℝ) : 
  (∃ d, d > 0 ∧ (4 * d = Real.sqrt ((4 * Real.sqrt 3)^2 + (d - 2)^2))) → d = 2 :=
sorry

end NUMINAMATH_GPT_smallest_d_l1888_188829


namespace NUMINAMATH_GPT_fraction_of_girls_l1888_188855

variable (T G B : ℕ) -- The total number of students, number of girls, and number of boys
variable (x : ℚ) -- The fraction of the number of girls

-- Definitions based on the given conditions
def fraction_condition : Prop := x * G = (1/6) * T
def ratio_condition : Prop := (B : ℚ) / (G : ℚ) = 2
def total_students : Prop := T = B + G

-- The statement we need to prove
theorem fraction_of_girls (h1 : fraction_condition T G x)
                          (h2 : ratio_condition B G)
                          (h3 : total_students T G B):
  x = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_girls_l1888_188855


namespace NUMINAMATH_GPT_chord_length_sqrt_10_l1888_188821

/-
  Given a line L: 3x - y - 6 = 0 and a circle C: x^2 + y^2 - 2x - 4y = 0,
  prove that the length of the chord AB formed by their intersection is sqrt(10).
-/

noncomputable def line_L : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ 3 * x - y - 6 = 0}

noncomputable def circle_C : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ x^2 + y^2 - 2 * x - 4 * y = 0}

noncomputable def chord_length (L C : Set (ℝ × ℝ)) : ℝ :=
  let center := (1, 2)
  let r := Real.sqrt 5
  let d := |3 * 1 - 2 - 6| / Real.sqrt (1 + 3^2)
  2 * Real.sqrt (r^2 - d^2)

theorem chord_length_sqrt_10 : chord_length line_L circle_C = Real.sqrt 10 := sorry

end NUMINAMATH_GPT_chord_length_sqrt_10_l1888_188821


namespace NUMINAMATH_GPT_out_of_pocket_expense_l1888_188863

theorem out_of_pocket_expense :
  let initial_purchase := 3000
  let tv_return := 700
  let bike_return := 500
  let sold_bike_cost := bike_return + (0.20 * bike_return)
  let sold_bike_sell_price := 0.80 * sold_bike_cost
  let toaster_purchase := 100
  (initial_purchase - tv_return - bike_return - sold_bike_sell_price + toaster_purchase) = 1420 :=
by
  sorry

end NUMINAMATH_GPT_out_of_pocket_expense_l1888_188863


namespace NUMINAMATH_GPT_fraction_identity_l1888_188864

variable (x y z : ℝ)

theorem fraction_identity (h : (x / (y + z)) + (y / (z + x)) + (z / (x + y)) = 1) :
  (x^2 / (y + z)) + (y^2 / (z + x)) + (z^2 / (x + y)) = 0 :=
  sorry

end NUMINAMATH_GPT_fraction_identity_l1888_188864


namespace NUMINAMATH_GPT_solve_inner_circle_radius_l1888_188820

noncomputable def isosceles_trapezoid_radius := 
  let AB := 8
  let BC := 7
  let DA := 7
  let CD := 6
  let radiusA := 4
  let radiusB := 4
  let radiusC := 3
  let radiusD := 3
  let r := (-72 + 60 * Real.sqrt 3) / 26
  r

theorem solve_inner_circle_radius :
  let k := 72
  let m := 60
  let n := 3
  let p := 26
  gcd k p = 1 → -- explicit gcd calculation between k and p 
  (isosceles_trapezoid_radius = (-k + m * Real.sqrt n) / p) ∧ (k + m + n + p = 161) :=
by
  sorry

end NUMINAMATH_GPT_solve_inner_circle_radius_l1888_188820


namespace NUMINAMATH_GPT_pipe_tank_overflow_l1888_188861

theorem pipe_tank_overflow (t : ℕ) :
  let rateA := 1 / 30
  let rateB := 1 / 60
  let combined_rate := rateA + rateB
  let workA := rateA * (t - 15)
  let workB := rateB * t
  (workA + workB = 1) ↔ (t = 25) := by
  sorry

end NUMINAMATH_GPT_pipe_tank_overflow_l1888_188861


namespace NUMINAMATH_GPT_real_root_quadratic_complex_eq_l1888_188817

open Complex

theorem real_root_quadratic_complex_eq (a : ℝ) :
  ∀ x : ℝ, a * (1 + I) * x^2 + (1 + a^2 * I) * x + (a^2 + I) = 0 →
  a = -1 :=
by
  intros x h
  -- We need to prove this, but we're skipping the proof for now.
  sorry

end NUMINAMATH_GPT_real_root_quadratic_complex_eq_l1888_188817


namespace NUMINAMATH_GPT_fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l1888_188804

theorem fewer_than_ten_sevens_example1 : (777 / 7) - (77 / 7) = 100 :=
  by sorry

theorem fewer_than_ten_sevens_example2 : (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7) = 100 :=
  by sorry

end NUMINAMATH_GPT_fewer_than_ten_sevens_example1_fewer_than_ten_sevens_example2_l1888_188804


namespace NUMINAMATH_GPT_compute_58_sq_pattern_l1888_188825

theorem compute_58_sq_pattern : (58 * 58 = 56 * 60 + 4) :=
by
  sorry

end NUMINAMATH_GPT_compute_58_sq_pattern_l1888_188825


namespace NUMINAMATH_GPT_cost_of_50_snacks_l1888_188845

-- Definitions based on conditions
def travel_time_to_work : ℕ := 2 -- hours
def cost_of_snack : ℕ := 10 * (2 * travel_time_to_work) -- Ten times the round trip time

-- The theorem to prove
theorem cost_of_50_snacks : (50 * cost_of_snack) = 2000 := by
  sorry

end NUMINAMATH_GPT_cost_of_50_snacks_l1888_188845


namespace NUMINAMATH_GPT_express_as_scientific_notation_l1888_188808

-- Define the question and condition
def trillion : ℝ := 1000000000000
def num := 6.13 * trillion

-- The main statement to be proven
theorem express_as_scientific_notation : num = 6.13 * 10^12 :=
by
  sorry

end NUMINAMATH_GPT_express_as_scientific_notation_l1888_188808


namespace NUMINAMATH_GPT_scientific_notation_of_number_l1888_188857

theorem scientific_notation_of_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000002 = a * 10^n ∧ a = 2 ∧ n = -8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_number_l1888_188857


namespace NUMINAMATH_GPT_roots_of_polynomial_l1888_188862

theorem roots_of_polynomial :
  (x^2 - 5 * x + 6) * (x - 1) * (x + 3) = 0 ↔ (x = -3 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_roots_of_polynomial_l1888_188862


namespace NUMINAMATH_GPT_alan_tickets_l1888_188896

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end NUMINAMATH_GPT_alan_tickets_l1888_188896


namespace NUMINAMATH_GPT_lowest_possible_number_of_students_l1888_188816

theorem lowest_possible_number_of_students :
  ∃ n : ℕ, (n % 12 = 0 ∧ n % 24 = 0) ∧ ∀ m : ℕ, ((m % 12 = 0 ∧ m % 24 = 0) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_lowest_possible_number_of_students_l1888_188816


namespace NUMINAMATH_GPT_solve_b_values_l1888_188868

open Int

theorem solve_b_values :
  {b : ℤ | ∃ x1 x2 x3 : ℤ, x1^2 + b * x1 - 2 ≤ 0 ∧ x2^2 + b * x2 - 2 ≤ 0 ∧ x3^2 + b * x3 - 2 ≤ 0 ∧
  ∀ x : ℤ, x ≠ x1 ∧ x ≠ x2 ∧ x ≠ x3 → x^2 + b * x - 2 > 0} = { -4, -3 } :=
by sorry

end NUMINAMATH_GPT_solve_b_values_l1888_188868


namespace NUMINAMATH_GPT_mikko_should_attempt_least_questions_l1888_188889

theorem mikko_should_attempt_least_questions (p : ℝ) (h_p : 0 < p ∧ p < 1) : 
  ∃ (x : ℕ), x ≥ ⌈1 / (2 * p - 1)⌉ :=
by
  sorry

end NUMINAMATH_GPT_mikko_should_attempt_least_questions_l1888_188889


namespace NUMINAMATH_GPT_bounces_less_than_50_l1888_188884

noncomputable def minBouncesNeeded (initialHeight : ℝ) (bounceFactor : ℝ) (thresholdHeight : ℝ) : ℕ :=
  ⌈(Real.log (thresholdHeight / initialHeight) / Real.log (bounceFactor))⌉₊

theorem bounces_less_than_50 :
  minBouncesNeeded 360 (3/4 : ℝ) 50 = 8 :=
by
  sorry

end NUMINAMATH_GPT_bounces_less_than_50_l1888_188884


namespace NUMINAMATH_GPT_fishing_problem_l1888_188854

theorem fishing_problem (a b c d : ℕ)
  (h1 : a + b + c + d = 11)
  (h2 : 1 ≤ a) 
  (h3 : 1 ≤ b) 
  (h4 : 1 ≤ c) 
  (h5 : 1 ≤ d) : 
  a < 3 ∨ b < 3 ∨ c < 3 ∨ d < 3 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_fishing_problem_l1888_188854
