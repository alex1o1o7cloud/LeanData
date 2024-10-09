import Mathlib

namespace kevin_hops_7_times_l2089_208973

noncomputable def distance_hopped_after_n_hops (n : ℕ) : ℚ :=
  4 * (1 - (3 / 4) ^ n)

theorem kevin_hops_7_times :
  distance_hopped_after_n_hops 7 = 7086 / 2048 := 
by
  sorry

end kevin_hops_7_times_l2089_208973


namespace travel_time_seattle_to_lasvegas_l2089_208937

def distance_seattle_boise : ℝ := 640
def distance_boise_saltlakecity : ℝ := 400
def distance_saltlakecity_phoenix : ℝ := 750
def distance_phoenix_lasvegas : ℝ := 300

def speed_highway_seattle_boise : ℝ := 80
def speed_city_seattle_boise : ℝ := 35

def speed_highway_boise_saltlakecity : ℝ := 65
def speed_city_boise_saltlakecity : ℝ := 25

def speed_highway_saltlakecity_denver : ℝ := 75
def speed_city_saltlakecity_denver : ℝ := 30

def speed_highway_denver_phoenix : ℝ := 70
def speed_city_denver_phoenix : ℝ := 20

def speed_highway_phoenix_lasvegas : ℝ := 50
def speed_city_phoenix_lasvegas : ℝ := 30

def city_distance_estimate : ℝ := 10

noncomputable def total_time : ℝ :=
  let time_seattle_boise := ((distance_seattle_boise - city_distance_estimate) / speed_highway_seattle_boise) + (city_distance_estimate / speed_city_seattle_boise)
  let time_boise_saltlakecity := ((distance_boise_saltlakecity - city_distance_estimate) / speed_highway_boise_saltlakecity) + (city_distance_estimate / speed_city_boise_saltlakecity)
  let time_saltlakecity_phoenix := ((distance_saltlakecity_phoenix - city_distance_estimate) / speed_highway_saltlakecity_denver) + (city_distance_estimate / speed_city_saltlakecity_denver)
  let time_phoenix_lasvegas := ((distance_phoenix_lasvegas - city_distance_estimate) / speed_highway_phoenix_lasvegas) + (city_distance_estimate / speed_city_phoenix_lasvegas)
  time_seattle_boise + time_boise_saltlakecity + time_saltlakecity_phoenix + time_phoenix_lasvegas

theorem travel_time_seattle_to_lasvegas :
  total_time = 30.89 :=
sorry

end travel_time_seattle_to_lasvegas_l2089_208937


namespace hypotenuse_length_l2089_208946

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 32) (h2 : a * b = 40) (h3 : a^2 + b^2 = c^2) : 
  c = 59 / 4 :=
by
  sorry

end hypotenuse_length_l2089_208946


namespace cone_new_height_l2089_208994

noncomputable def new_cone_height : ℝ := 6

theorem cone_new_height (r h V : ℝ) (circumference : 2 * Real.pi * r = 24 * Real.pi)
  (original_height : h = 40) (same_base_circumference : 2 * Real.pi * r = 24 * Real.pi)
  (volume : (1 / 3) * Real.pi * (r ^ 2) * new_cone_height = 288 * Real.pi) :
    new_cone_height = 6 := 
sorry

end cone_new_height_l2089_208994


namespace strawberries_picking_problem_l2089_208972

noncomputable def StrawberriesPicked : Prop :=
  let kg_to_lb := 2.2
  let marco_pounds := 1 + 3 * kg_to_lb
  let sister_pounds := 1.5 * marco_pounds
  let father_pounds := 2 * sister_pounds
  marco_pounds = 7.6 ∧ sister_pounds = 11.4 ∧ father_pounds = 22.8

theorem strawberries_picking_problem : StrawberriesPicked :=
  sorry

end strawberries_picking_problem_l2089_208972


namespace exists_irrational_an_l2089_208959

theorem exists_irrational_an (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n ≥ 1, a (n + 1)^2 = a n + 1) :
  ∃ n, ¬ ∃ q : ℚ, a n = q :=
sorry

end exists_irrational_an_l2089_208959


namespace intersecting_lines_solution_l2089_208945

theorem intersecting_lines_solution (x y b : ℝ) 
  (h₁ : y = 2 * x - 5)
  (h₂ : y = 3 * x + b)
  (hP : x = 1 ∧ y = -3) : 
  b = -6 ∧ x = 1 ∧ y = -3 := by
  sorry

end intersecting_lines_solution_l2089_208945


namespace eighth_arithmetic_term_l2089_208989

theorem eighth_arithmetic_term (a₂ a₁₄ a₈ : ℚ) 
  (h2 : a₂ = 8 / 11)
  (h14 : a₁₄ = 9 / 13) :
  a₈ = 203 / 286 :=
by
  sorry

end eighth_arithmetic_term_l2089_208989


namespace average_speed_sf_l2089_208942

variables
  (v d t : ℝ)  -- Representing the average speed to SF, the distance, and time to SF
  (h1 : 42 = (2 * d) / (3 * t))  -- Condition: Average speed of the round trip is 42 mph
  (h2 : t = d / v)  -- Definition of time t in terms of distance and speed

theorem average_speed_sf : v = 63 :=
by
  sorry

end average_speed_sf_l2089_208942


namespace nine_otimes_three_l2089_208967

def otimes (a b : ℤ) : ℤ := a + (4 * a) / (3 * b)

theorem nine_otimes_three : otimes 9 3 = 13 := by
  sorry

end nine_otimes_three_l2089_208967


namespace largest_four_digit_number_divisible_by_33_l2089_208983

theorem largest_four_digit_number_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (33 ∣ n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ 33 ∣ m → m ≤ 9999) :=
by
  sorry

end largest_four_digit_number_divisible_by_33_l2089_208983


namespace license_plate_increase_l2089_208996

theorem license_plate_increase :
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 26^2 / 10 :=
by
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4
  show new_plates / old_plates = 26^2 / 10
  sorry

end license_plate_increase_l2089_208996


namespace gas_pipe_probability_l2089_208915

-- Define the problem statement in Lean.
theorem gas_pipe_probability :
  let total_area := 400 * 400 / 2
  let usable_area := (300 - 100) * (300 - 100) / 2
  usable_area / total_area = 1 / 4 :=
by
  -- Sorry will be placeholder for the proof
  sorry

end gas_pipe_probability_l2089_208915


namespace outfit_count_correct_l2089_208966

def total_shirts : ℕ := 8
def total_pants : ℕ := 4
def total_hats : ℕ := 6
def shirt_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Set (String) := {"tan", "black", "blue", "gray", "white", "yellow"}
def conflict_free_outfits (total_shirts total_pants total_hats : ℕ) : ℕ :=
  let total_outfits := total_shirts * total_pants * total_hats
  let matching_outfits := (2 * 1 * 4) * total_pants
  total_outfits - matching_outfits

theorem outfit_count_correct :
  conflict_free_outfits total_shirts total_pants total_hats = 160 :=
by
  unfold conflict_free_outfits
  norm_num
  sorry

end outfit_count_correct_l2089_208966


namespace area_of_region_bounded_by_circle_l2089_208902

theorem area_of_region_bounded_by_circle :
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 9 = 0) →
  ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_bounded_by_circle_l2089_208902


namespace part1_part2_l2089_208919

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l2089_208919


namespace average_of_25_results_l2089_208951

theorem average_of_25_results (first12_avg : ℕ -> ℕ -> ℕ)
                             (last12_avg : ℕ -> ℕ -> ℕ) 
                             (res13 : ℕ)
                             (avg_of_25 : ℕ) :
                             first12_avg 12 10 = 120
                             ∧ last12_avg 12 20 = 240
                             ∧ res13 = 90
                             ∧ avg_of_25 = (first12_avg 12 10 + last12_avg 12 20 + res13) / 25
                             → avg_of_25 = 18 := by
  sorry

end average_of_25_results_l2089_208951


namespace chef_pies_total_l2089_208992

def chefPieSales : ℕ :=
  let small_shepherd_pies := 52 / 4
  let large_shepherd_pies := 76 / 8
  let small_chicken_pies := 80 / 5
  let large_chicken_pies := 130 / 10
  let small_vegetable_pies := 42 / 6
  let large_vegetable_pies := 96 / 12
  let small_beef_pies := 35 / 7
  let large_beef_pies := 105 / 14

  small_shepherd_pies + large_shepherd_pies + small_chicken_pies + large_chicken_pies +
  small_vegetable_pies + large_vegetable_pies +
  small_beef_pies + large_beef_pies

theorem chef_pies_total : chefPieSales = 80 := by
  unfold chefPieSales
  have h1 : 52 / 4 = 13 := by norm_num
  have h2 : 76 / 8 = 9 ∨ 76 / 8 = 10 := by norm_num -- rounding consideration
  have h3 : 80 / 5 = 16 := by norm_num
  have h4 : 130 / 10 = 13 := by norm_num
  have h5 : 42 / 6 = 7 := by norm_num
  have h6 : 96 / 12 = 8 := by norm_num
  have h7 : 35 / 7 = 5 := by norm_num
  have h8 : 105 / 14 = 7 ∨ 105 / 14 = 8 := by norm_num -- rounding consideration
  sorry

end chef_pies_total_l2089_208992


namespace geometric_sequence_sum_eq_five_l2089_208932

/-- Given that {a_n} is a geometric sequence where each a_n > 0
    and the equation a_2 * a_4 + 2 * a_3 * a_5 + a_4 * a_6 = 25 holds,
    we want to prove that a_3 + a_5 = 5. -/
theorem geometric_sequence_sum_eq_five
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a n = a 1 * r ^ (n - 1))
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : a 3 + a 5 = 5 :=
sorry

end geometric_sequence_sum_eq_five_l2089_208932


namespace cos_double_angle_l2089_208935

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4 / 5 := 
  sorry

end cos_double_angle_l2089_208935


namespace direction_vector_of_line_m_l2089_208933

noncomputable def projectionMatrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![ 5 / 21, -2 / 21, -2 / 7 ],
    ![ -2 / 21, 1 / 42, 1 / 14 ],
    ![ -2 / 7,  1 / 14, 4 / 7 ]
  ]

noncomputable def vectorI : Fin 3 → ℚ
  | 0 => 1
  | _ => 0

noncomputable def projectedVector : Fin 3 → ℚ :=
  fun i => (projectionMatrix.mulVec vectorI) i

theorem direction_vector_of_line_m :
  (projectedVector 0 = 5 / 21) ∧ 
  (projectedVector 1 = -2 / 21) ∧
  (projectedVector 2 = -6 / 21) ∧
  Nat.gcd (Nat.gcd 5 2) 6 = 1 :=
by
  sorry

end direction_vector_of_line_m_l2089_208933


namespace intersection_A_B_l2089_208978

open Set

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def B : Set ℕ := {x | 2 ≤ x ∧ x < 6}

theorem intersection_A_B : A ∩ B = {2, 3, 4, 5} := by
  sorry

end intersection_A_B_l2089_208978


namespace range_of_a_l2089_208926

noncomputable def p (a : ℝ) : Prop := 
  (1 + a)^2 + (1 - a)^2 < 4

noncomputable def q (a : ℝ) : Prop := 
  ∀ x : ℝ, x^2 + a * x + 1 ≥ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) ↔ (-2 ≤ a ∧ a ≤ -1) ∨ (1 ≤ a ∧ a ≤ 2) := 
by
  sorry

end range_of_a_l2089_208926


namespace factor_quadratic_l2089_208968

theorem factor_quadratic : ∀ (x : ℝ), 4 * x^2 - 20 * x + 25 = (2 * x - 5)^2 :=
by
  intro x
  sorry

end factor_quadratic_l2089_208968


namespace min_value_fraction_l2089_208958

theorem min_value_fraction (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + 2 * b + 3 * c = 1) : 
  (1/a + 2/b + 3/c) ≥ 36 := 
sorry

end min_value_fraction_l2089_208958


namespace largest_integer_sol_l2089_208909

theorem largest_integer_sol (x : ℤ) : (3 * x + 4 < 5 * x - 2) -> x = 3 :=
by
  sorry

end largest_integer_sol_l2089_208909


namespace problem_statement_l2089_208908

theorem problem_statement (a b : ℕ) (m n : ℕ)
  (h1 : 32 + (2 / 7 : ℝ) = 3 * (2 / 7 : ℝ))
  (h2 : 33 + (3 / 26 : ℝ) = 3 * (3 / 26 : ℝ))
  (h3 : 34 + (4 / 63 : ℝ) = 3 * (4 / 63 : ℝ))
  (h4 : 32014 + (m / n : ℝ) = 2014 * 3 * (m / n : ℝ))
  (h5 : 32016 + (a / b : ℝ) = 2016 * 3 * (a / b : ℝ)) :
  (b + 1) / (a * a) = 2016 :=
sorry

end problem_statement_l2089_208908


namespace sum_of_consecutive_integers_l2089_208980

theorem sum_of_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 := by
  sorry

end sum_of_consecutive_integers_l2089_208980


namespace store_discount_percentage_l2089_208927

theorem store_discount_percentage
  (total_without_discount : ℝ := 350)
  (final_price : ℝ := 252)
  (coupon_percentage : ℝ := 0.1) :
  ∃ (x : ℝ), total_without_discount * (1 - x / 100) * (1 - coupon_percentage) = final_price ∧ x = 20 :=
by
  use 20
  sorry

end store_discount_percentage_l2089_208927


namespace smallest_prime_divides_sum_l2089_208925

theorem smallest_prime_divides_sum :
  ∃ a, Prime a ∧ a ∣ (3 ^ 11 + 5 ^ 13) ∧
       ∀ b, Prime b → b ∣ (3 ^ 11 + 5 ^ 13) → a ≤ b :=
sorry

end smallest_prime_divides_sum_l2089_208925


namespace production_difference_correct_l2089_208963

variable (w t M T : ℕ)

-- Condition: w = 2t
def condition_w := w = 2 * t

-- Widgets produced on Monday
def widgets_monday := M = w * t

-- Widgets produced on Tuesday
def widgets_tuesday := T = (w + 5) * (t - 3)

-- Difference in production
def production_difference := M - T = t + 15

theorem production_difference_correct
  (h1 : condition_w w t)
  (h2 : widgets_monday M w t)
  (h3 : widgets_tuesday T w t) :
  production_difference M T t :=
sorry

end production_difference_correct_l2089_208963


namespace ellen_bakes_6_balls_of_dough_l2089_208939

theorem ellen_bakes_6_balls_of_dough (rising_time baking_time total_time : ℕ) (h_rise : rising_time = 3) (h_bake : baking_time = 2) (h_total : total_time = 20) :
  ∃ n : ℕ, (rising_time + baking_time) + rising_time * (n - 1) = total_time ∧ n = 6 :=
by sorry

end ellen_bakes_6_balls_of_dough_l2089_208939


namespace sequence_equal_l2089_208912

variable {n : ℕ} (h1 : 2 ≤ n)
variable (a : ℕ → ℝ)
variable (h2 : ∀ i, a i ≠ -1)
variable (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
variable (h4 : a n = a 0)
variable (h5 : a (n + 1) = a 1)

theorem sequence_equal 
  (h1 : 2 ≤ n)
  (h2 : ∀ i, a i ≠ -1) 
  (h3 : ∀ i, a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1))
  (h4 : a n = a 0)
  (h5 : a (n + 1) = a 1) :
  ∀ i, a i = a 0 := 
sorry

end sequence_equal_l2089_208912


namespace no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l2089_208986

theorem no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), n^2012 - 1 = 2^k :=
by
  sorry  

end no_natural_number_such_that_n_pow_2012_minus_1_is_power_of_two_l2089_208986


namespace five_star_three_l2089_208993

def star (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2

theorem five_star_three : star 5 3 = 4 := by
  sorry

end five_star_three_l2089_208993


namespace air_quality_probability_l2089_208976

variable (p_good_day : ℝ) (p_good_two_days : ℝ)

theorem air_quality_probability
  (h1 : p_good_day = 0.75)
  (h2 : p_good_two_days = 0.6) :
  (p_good_two_days / p_good_day = 0.8) :=
by
  rw [h1, h2]
  norm_num

end air_quality_probability_l2089_208976


namespace percentage_discount_is_12_l2089_208977

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 67.47
noncomputable def desired_selling_price : ℝ := cost_price + 0.25 * cost_price
noncomputable def actual_selling_price : ℝ := 59.375

theorem percentage_discount_is_12 :
  ∃ D : ℝ, desired_selling_price = list_price - (list_price * D) ∧ D = 0.12 := 
by 
  sorry

end percentage_discount_is_12_l2089_208977


namespace left_handed_women_percentage_l2089_208990

theorem left_handed_women_percentage
  (x y : ℕ)
  (h1 : 4 * x = 5 * y)
  (h2 : 3 * x ≥ 3 * y) :
  (x / (4 * x) : ℚ) * 100 = 25 :=
by
  sorry

end left_handed_women_percentage_l2089_208990


namespace tan_triple_angle_l2089_208947

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l2089_208947


namespace find_cows_l2089_208900

theorem find_cows :
  ∃ (D C : ℕ), (2 * D + 4 * C = 2 * (D + C) + 30) → C = 15 := 
sorry

end find_cows_l2089_208900


namespace circle_center_coordinates_l2089_208974

theorem circle_center_coordinates (h k r : ℝ) :
  (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 1 → (x - h)^2 + (y - k)^2 = r^2) →
  (h, k) = (2, -3) :=
by
  intro H
  sorry

end circle_center_coordinates_l2089_208974


namespace cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l2089_208923

-- Define the conditions
def ticket_full_price : ℕ := 240
def discount_A : ℕ := ticket_full_price / 2
def discount_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Algebraic expressions provided in the answer
def cost_A (x : ℕ) : ℕ := discount_A * x + ticket_full_price
def cost_B (x : ℕ) : ℕ := 144 * (x + 1)

-- Proofs for the specific cases
theorem cost_expression_A (x : ℕ) : cost_A x = 120 * x + 240 := by
  sorry

theorem cost_expression_B (x : ℕ) : cost_B x = 144 * (x + 1) := by
  sorry

theorem cost_comparison_10_students : cost_A 10 < cost_B 10 := by
  sorry

theorem cost_comparison_4_students : cost_A 4 = cost_B 4 := by
  sorry

end cost_expression_A_cost_expression_B_cost_comparison_10_students_cost_comparison_4_students_l2089_208923


namespace find_m_l2089_208969

theorem find_m (a b c m x : ℂ) :
  ( (2 * m + 1) * (x^2 - (b + 1) * x) = (2 * m - 3) * (2 * a * x - c) )
  →
  (x = (b + 1)) 
  →
  m = 1.5 := by
  sorry

end find_m_l2089_208969


namespace range_of_eccentricity_l2089_208995

theorem range_of_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (c : ℝ := Real.sqrt (a^2 - b^2)) 
  (h_dot_product : ∀ (x y: ℝ) (h_point : x^2 / a^2 + y^2 / b^2 = 1), 
    let PF1 : ℝ × ℝ := (-c - x, -y)
    let PF2 : ℝ × ℝ := (c - x, -y)
    PF1.1 * PF2.1 + PF1.2 * PF2.2 ≤ a * c) : 
  ∀ (e : ℝ := c / a), (Real.sqrt 5 - 1) / 2 ≤ e ∧ e < 1 := 
by 
  sorry

end range_of_eccentricity_l2089_208995


namespace f_3_eq_4_l2089_208904

noncomputable def f : ℝ → ℝ := sorry

theorem f_3_eq_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 3 = 4 :=
by
  sorry

end f_3_eq_4_l2089_208904


namespace maria_total_baggies_l2089_208953

def choc_chip_cookies := 33
def oatmeal_cookies := 2
def cookies_per_bag := 5

def total_cookies := choc_chip_cookies + oatmeal_cookies

def total_baggies (total_cookies : Nat) (cookies_per_bag : Nat) : Nat :=
  total_cookies / cookies_per_bag

theorem maria_total_baggies : total_baggies total_cookies cookies_per_bag = 7 :=
  by
    -- Steps proving the equivalence can be done here
    sorry

end maria_total_baggies_l2089_208953


namespace compute_a1d1_a2d2_a3d3_l2089_208971

theorem compute_a1d1_a2d2_a3d3
  (a1 a2 a3 d1 d2 d3 : ℝ)
  (h : ∀ x : ℝ, x^6 + 2 * x^5 + x^4 + x^3 + x^2 + 2 * x + 1 = (x^2 + a1*x + d1) * (x^2 + a2*x + d2) * (x^2 + a3*x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 2 :=
by
  sorry

end compute_a1d1_a2d2_a3d3_l2089_208971


namespace no_intersection_range_k_l2089_208916

def problem_statement (k : ℝ) : Prop :=
  ∀ (x : ℝ),
    ¬(x > 1 ∧ x + 1 = k * x + 2) ∧ ¬(x < 1 ∧ -x - 1 = k * x + 2) ∧ 
    (x = 1 → (x + 1 ≠ k * x + 2 ∧ -x - 1 ≠ k * x + 2))

theorem no_intersection_range_k :
  ∀ (k : ℝ), problem_statement k ↔ -4 ≤ k ∧ k < -1 :=
sorry

end no_intersection_range_k_l2089_208916


namespace functional_equation_solution_l2089_208961

theorem functional_equation_solution :
  ∀ (f : ℚ → ℝ), (∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) →
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x :=
by
  sorry

end functional_equation_solution_l2089_208961


namespace paula_cans_used_l2089_208920

/-- 
  Paula originally had enough paint to cover 42 rooms. 
  Unfortunately, she lost 4 cans of paint on her way, 
  and now she can only paint 34 rooms. 
  Prove the number of cans she used for these 34 rooms is 17.
-/
theorem paula_cans_used (R L P C : ℕ) (hR : R = 42) (hL : L = 4) (hP : P = 34)
    (hRooms : R - ((R - P) / L) * L = P) :
  C = 17 :=
by
  sorry

end paula_cans_used_l2089_208920


namespace susie_rooms_l2089_208941

theorem susie_rooms
  (house_vacuum_time_hours : ℕ)
  (room_vacuum_time_minutes : ℕ)
  (total_vacuum_time_minutes : ℕ)
  (total_vacuum_time_computed : house_vacuum_time_hours * 60 = total_vacuum_time_minutes)
  (rooms_count : ℕ)
  (rooms_count_computed : total_vacuum_time_minutes / room_vacuum_time_minutes = rooms_count) :
  house_vacuum_time_hours = 2 →
  room_vacuum_time_minutes = 20 →
  rooms_count = 6 :=
by
  intros h1 h2
  sorry

end susie_rooms_l2089_208941


namespace correct_decimal_multiplication_l2089_208975

theorem correct_decimal_multiplication : 0.085 * 3.45 = 0.29325 := 
by 
  sorry

end correct_decimal_multiplication_l2089_208975


namespace avg_age_new_students_l2089_208907

theorem avg_age_new_students :
  ∀ (O A_old A_new_avg : ℕ) (A_new : ℕ),
    O = 12 ∧ A_old = 40 ∧ A_new_avg = (A_old - 4) ∧ A_new_avg = 36 →
    A_new * 12 = (24 * A_new_avg) - (O * A_old) →
    A_new = 32 :=
by
  intros O A_old A_new_avg A_new
  intro h
  rcases h with ⟨hO, hA_old, hA_new_avg, h36⟩
  sorry

end avg_age_new_students_l2089_208907


namespace harriet_trip_time_l2089_208964

theorem harriet_trip_time :
  ∀ (t1 : ℝ) (s1 s2 t2 d : ℝ), 
  t1 = 2.8 ∧ 
  s1 = 110 ∧ 
  s2 = 140 ∧ 
  d = s1 * t1 ∧ 
  t2 = d / s2 → 
  t1 + t2 = 5 :=
by intros t1 s1 s2 t2 d
   sorry

end harriet_trip_time_l2089_208964


namespace mailman_total_delivered_l2089_208991

def pieces_of_junk_mail : Nat := 6
def magazines : Nat := 5
def newspapers : Nat := 3
def bills : Nat := 4
def postcards : Nat := 2

def total_pieces_of_mail : Nat := pieces_of_junk_mail + magazines + newspapers + bills + postcards

theorem mailman_total_delivered : total_pieces_of_mail = 20 := by
  sorry

end mailman_total_delivered_l2089_208991


namespace cos_pi_minus_2alpha_eq_seven_over_twentyfive_l2089_208928

variable (α : ℝ)

theorem cos_pi_minus_2alpha_eq_seven_over_twentyfive 
  (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π - 2 * α) = 7 / 25 := 
by
  sorry

end cos_pi_minus_2alpha_eq_seven_over_twentyfive_l2089_208928


namespace opposite_neg_two_l2089_208962

theorem opposite_neg_two : -(-2) = 2 := by
  sorry

end opposite_neg_two_l2089_208962


namespace eleven_pow_2048_mod_17_l2089_208960

theorem eleven_pow_2048_mod_17 : 11^2048 % 17 = 1 := by
  sorry

end eleven_pow_2048_mod_17_l2089_208960


namespace binary_1101_to_decimal_l2089_208950

theorem binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 13 := by
  -- To convert a binary number to its decimal equivalent, we multiply each digit by its corresponding power of 2 based on its position and then sum the results.
  sorry

end binary_1101_to_decimal_l2089_208950


namespace mrs_evans_class_l2089_208922

def students_enrolled_in_class (S Q1 Q2 missing both: ℕ) : Prop :=
  25 = Q1 ∧ 22 = Q2 ∧ 5 = missing ∧ 22 = both → S = Q1 + Q2 - both + missing

theorem mrs_evans_class (S : ℕ) : students_enrolled_in_class S 25 22 5 22 :=
by
  sorry

end mrs_evans_class_l2089_208922


namespace min_value_geom_seq_l2089_208987

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  0 < a 4 ∧ 0 < a 14 ∧ a 4 * a 14 = 8 ∧ 0 < a 7 ∧ 0 < a 11 ∧ a 7 * a 11 = 8

theorem min_value_geom_seq {a : ℕ → ℝ} (h : geom_seq a) :
  2 * a 7 + a 11 = 8 :=
by
  sorry

end min_value_geom_seq_l2089_208987


namespace Doug_money_l2089_208931

theorem Doug_money (B D : ℝ) (h1 : B + 2*B + D = 68) (h2 : 2*B = (3/4)*D) : D = 32 := by
  sorry

end Doug_money_l2089_208931


namespace seq_a10_eq_90_l2089_208982

noncomputable def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem seq_a10_eq_90 {a : ℕ → ℕ} (h : seq a) : a 10 = 90 :=
  sorry

end seq_a10_eq_90_l2089_208982


namespace sqrt_neg3_squared_l2089_208999

theorem sqrt_neg3_squared : Real.sqrt ((-3)^2) = 3 :=
by sorry

end sqrt_neg3_squared_l2089_208999


namespace div_by_66_l2089_208965

theorem div_by_66 :
  (43 ^ 23 + 23 ^ 43) % 66 = 0 := 
sorry

end div_by_66_l2089_208965


namespace roots_of_polynomial_l2089_208957

noncomputable def polynomial (m z : ℝ) : ℝ :=
  z^3 - (m^2 - m + 7) * z - (3 * m^2 - 3 * m - 6)

theorem roots_of_polynomial (m z : ℝ) (h : polynomial m (-1) = 0) :
  (m = 3 ∧ z = 4 ∨ z = -3) ∨ (m = -2 ∧ sorry) :=
sorry

end roots_of_polynomial_l2089_208957


namespace most_probable_sellable_samples_l2089_208911

/-- Prove that the most probable number k of sellable samples out of 24,
given each has a 0.6 probability of being sellable, is either 14 or 15. -/
theorem most_probable_sellable_samples (n : ℕ) (p : ℝ) (q : ℝ) (k₀ k₁ : ℕ) 
  (h₁ : n = 24) (h₂ : p = 0.6) (h₃ : q = 1 - p)
  (h₄ : 24 * p - q < k₀) (h₅ : k₀ < 24 * p + p) 
  (h₆ : k₀ = 14) (h₇ : k₁ = 15) :
  (k₀ = 14 ∨ k₀ = 15) :=
  sorry

end most_probable_sellable_samples_l2089_208911


namespace value_of_b_div_a_l2089_208934

theorem value_of_b_div_a (a b : ℝ) (h : |5 - a| + (b + 3)^2 = 0) : b / a = -3 / 5 :=
by
  sorry

end value_of_b_div_a_l2089_208934


namespace smallest_three_digit_divisible_l2089_208944

theorem smallest_three_digit_divisible :
  ∃ (A B C : Nat), A ≠ 0 ∧ 100 ≤ (100 * A + 10 * B + C) ∧ (100 * A + 10 * B + C) < 1000 ∧
  (10 * A + B) > 9 ∧ (10 * B + C) > 9 ∧ 
  (100 * A + 10 * B + C) % (10 * A + B) = 0 ∧ (100 * A + 10 * B + C) % (10 * B + C) = 0 ∧
  (100 * A + 10 * B + C) = 110 :=
by
  sorry

end smallest_three_digit_divisible_l2089_208944


namespace factorization_correct_l2089_208943

theorem factorization_correct (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) :=
by
  sorry

end factorization_correct_l2089_208943


namespace marble_probability_l2089_208905

theorem marble_probability :
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  first_red_prob * second_white_given_first_red_prob * third_red_given_first_red_and_second_white_prob = (40 : ℚ) / 429 :=
by
  let total_marbles := 13
  let red_marbles := 5
  let white_marbles := 8
  let first_red_prob := (red_marbles:ℚ) / total_marbles
  let second_white_given_first_red_prob := (white_marbles:ℚ) / (total_marbles - 1)
  let third_red_given_first_red_and_second_white_prob := (red_marbles - 1:ℚ) / (total_marbles - 2)
  -- Adding sorry to skip the proof
  sorry

end marble_probability_l2089_208905


namespace average_price_of_six_toys_l2089_208998

/-- Define the average cost of toys given the number of toys and their total cost -/
def avg_cost (total_cost : ℕ) (num_toys : ℕ) : ℕ :=
  total_cost / num_toys

/-- Define the total cost of toys given a list of individual toy costs -/
def total_cost (costs : List ℕ) : ℕ :=
  costs.foldl (· + ·) 0

/-- The main theorem -/
theorem average_price_of_six_toys :
  let dhoni_toys := 5
  let avg_cost_dhoni := 10
  let total_cost_dhoni := dhoni_toys * avg_cost_dhoni
  let david_toy_cost := 16
  let total_toys := dhoni_toys + 1
  total_cost_dhoni + david_toy_cost = 66 →
  avg_cost (66) (total_toys) = 11 :=
by
  -- Introduce the conditions and hypothesis
  intros total_cost_of_6_toys H
  -- Simplify the expression
  sorry  -- Proof skipped

end average_price_of_six_toys_l2089_208998


namespace average_loss_l2089_208930

theorem average_loss (cost_per_lootbox : ℝ) (average_value_per_lootbox : ℝ) (total_spent : ℝ)
                      (h1 : cost_per_lootbox = 5)
                      (h2 : average_value_per_lootbox = 3.5)
                      (h3 : total_spent = 40) :
  (total_spent - (average_value_per_lootbox * (total_spent / cost_per_lootbox))) = 12 :=
by
  sorry

end average_loss_l2089_208930


namespace james_earnings_l2089_208970

theorem james_earnings :
  let jan_earn : ℕ := 4000
  let feb_earn := 2 * jan_earn
  let total_earnings : ℕ := 18000
  let earnings_jan_feb := jan_earn + feb_earn
  let mar_earn := total_earnings - earnings_jan_feb
  (feb_earn - mar_earn) = 2000 := by
  sorry

end james_earnings_l2089_208970


namespace sally_spent_eur_l2089_208979

-- Define the given conditions
def coupon_value : ℝ := 3
def peaches_total_usd : ℝ := 12.32
def cherries_original_usd : ℝ := 11.54
def discount_rate : ℝ := 0.1
def conversion_rate : ℝ := 0.85

-- Define the intermediate calculations
def cherries_discount_usd : ℝ := cherries_original_usd * discount_rate
def cherries_final_usd : ℝ := cherries_original_usd - cherries_discount_usd
def total_usd : ℝ := peaches_total_usd + cherries_final_usd
def total_eur : ℝ := total_usd * conversion_rate

-- The final statement to be proven
theorem sally_spent_eur : total_eur = 19.30 := by
  sorry

end sally_spent_eur_l2089_208979


namespace cistern_fill_time_l2089_208918

variable (C : ℝ) -- Volume of the cistern
variable (X Y Z : ℝ) -- Rates at which pipes X, Y, and Z fill the cistern

-- Pipes X and Y together, pipes X and Z together, and pipes Y and Z together conditions
def condition1 := X + Y = C / 3
def condition2 := X + Z = C / 4
def condition3 := Y + Z = C / 5

theorem cistern_fill_time (h1 : condition1 C X Y) (h2 : condition2 C X Z) (h3 : condition3 C Y Z) :
  1 / (X + Y + Z) = 120 / 47 :=
by
  sorry

end cistern_fill_time_l2089_208918


namespace inequality_of_powers_l2089_208988

theorem inequality_of_powers (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end inequality_of_powers_l2089_208988


namespace find_x_plus_y_l2089_208914

theorem find_x_plus_y (x y : ℝ) (hx : |x| + x + y = 14) (hy : x + |y| - y = 16) : x + y = 26 / 5 := 
sorry

end find_x_plus_y_l2089_208914


namespace total_price_paid_l2089_208903

noncomputable def total_price
    (price_rose : ℝ) (qty_rose : ℕ) (discount_rose : ℝ)
    (price_lily : ℝ) (qty_lily : ℕ) (discount_lily : ℝ)
    (price_sunflower : ℝ) (qty_sunflower : ℕ)
    (store_discount : ℝ) (tax_rate : ℝ)
    : ℝ :=
  let total_rose := qty_rose * price_rose
  let total_lily := qty_lily * price_lily
  let total_sunflower := qty_sunflower * price_sunflower
  let total := total_rose + total_lily + total_sunflower
  let total_disc_rose := total_rose * discount_rose
  let total_disc_lily := total_lily * discount_lily
  let discounted_total := total - total_disc_rose - total_disc_lily
  let store_discount_amount := discounted_total * store_discount
  let after_store_discount := discounted_total - store_discount_amount
  let tax_amount := after_store_discount * tax_rate
  after_store_discount + tax_amount

theorem total_price_paid :
  total_price 20 3 0.15 15 5 0.10 10 2 0.05 0.07 = 140.79 :=
by
  apply sorry

end total_price_paid_l2089_208903


namespace bicycle_cost_price_l2089_208929

theorem bicycle_cost_price (CP_A : ℝ) 
    (h1 : ∀ SP_B, SP_B = 1.20 * CP_A)
    (h2 : ∀ CP_C SP_B, CP_C = 1.40 * SP_B ∧ SP_B = 1.20 * CP_A)
    (h3 : ∀ SP_D CP_C, SP_D = 1.30 * CP_C ∧ CP_C = 1.40 * 1.20 * CP_A)
    (h4 : ∀ SP_D', SP_D' = 350 / 0.90) :
    CP_A = 350 / 1.9626 :=
by
  sorry

end bicycle_cost_price_l2089_208929


namespace baseball_cards_start_count_l2089_208936

theorem baseball_cards_start_count (X : ℝ) 
  (h1 : ∃ (x : ℝ), x = (X + 1) / 2)
  (h2 : ∃ (x' : ℝ), x' = X - ((X + 1) / 2) - 1)
  (h3 : ∃ (y : ℝ), y = 3 * (X - ((X + 1) / 2) - 1))
  (h4 : ∃ (z : ℝ), z = 18) : 
  X = 15 :=
by
  sorry

end baseball_cards_start_count_l2089_208936


namespace initial_books_l2089_208924

theorem initial_books (added_books : ℝ) (books_per_shelf : ℝ) (shelves : ℝ) 
  (total_books : ℝ) : total_books = shelves * books_per_shelf → 
  shelves = 14 → books_per_shelf = 4.0 → added_books = 10.0 → 
  total_books - added_books = 46.0 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_books_l2089_208924


namespace sum_of_consecutive_neg_ints_l2089_208910

theorem sum_of_consecutive_neg_ints (n : ℤ) (h : n * (n + 1) = 2720) (hn : n < 0) (hn_plus1 : n + 1 < 0) :
  n + (n + 1) = -105 :=
sorry

end sum_of_consecutive_neg_ints_l2089_208910


namespace roots_of_polynomial_l2089_208955

theorem roots_of_polynomial : {x : ℝ | (x^2 - 5*x + 6)*(x - 1)*(x - 6) = 0} = {1, 2, 3, 6} :=
by
  -- proof goes here
  sorry

end roots_of_polynomial_l2089_208955


namespace difference_in_students_and_guinea_pigs_l2089_208954

def num_students (classrooms : ℕ) (students_per_classroom : ℕ) : ℕ := classrooms * students_per_classroom
def num_guinea_pigs (classrooms : ℕ) (guinea_pigs_per_classroom : ℕ) : ℕ := classrooms * guinea_pigs_per_classroom
def difference_students_guinea_pigs (students : ℕ) (guinea_pigs : ℕ) : ℕ := students - guinea_pigs

theorem difference_in_students_and_guinea_pigs :
  ∀ (classrooms : ℕ) (students_per_classroom : ℕ) (guinea_pigs_per_classroom : ℕ),
  classrooms = 6 →
  students_per_classroom = 24 →
  guinea_pigs_per_classroom = 3 →
  difference_students_guinea_pigs (num_students classrooms students_per_classroom) (num_guinea_pigs classrooms guinea_pigs_per_classroom) = 126 :=
by
  intros
  sorry

end difference_in_students_and_guinea_pigs_l2089_208954


namespace number_of_boys_in_second_group_l2089_208952

noncomputable def daily_work_done_by_man (M : ℝ) (B : ℝ) : Prop :=
  M = 2 * B

theorem number_of_boys_in_second_group
  (M B : ℝ)
  (h1 : (12 * M + 16 * B) * 5 = (13 * M + 24 * B) * 4)
  (h2 : daily_work_done_by_man M B) :
  24 = 24 :=
by
  -- The proof is omitted.
  sorry

end number_of_boys_in_second_group_l2089_208952


namespace percentage_problem_l2089_208985

theorem percentage_problem (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42) : P = 35 := 
by
  -- Proof goes here
  sorry

end percentage_problem_l2089_208985


namespace speed_second_boy_l2089_208997

theorem speed_second_boy (v : ℝ) (t : ℝ) (d : ℝ) (s₁ : ℝ) :
  s₁ = 4.5 ∧ t = 9.5 ∧ d = 9.5 ∧ (d = (v - s₁) * t) → v = 5.5 :=
by
  intros h
  obtain ⟨hs₁, ht, hd, hev⟩ := h
  sorry

end speed_second_boy_l2089_208997


namespace linear_eq_conditions_l2089_208984

theorem linear_eq_conditions (m : ℤ) (h : abs m = 1) (h₂ : m + 1 ≠ 0) : m = 1 :=
by
  sorry

end linear_eq_conditions_l2089_208984


namespace distance_from_A_to_C_correct_total_distance_traveled_correct_l2089_208913

-- Define the conditions
def distance_to_A : ℕ := 30
def distance_to_B : ℕ := 20
def distance_to_C : ℤ := -15
def times_to_C : ℕ := 3

-- Define the resulting calculated distances based on the conditions
def distance_A_to_C : ℕ := distance_to_A + distance_to_C.natAbs
def total_distance_traveled : ℕ := (distance_to_A + distance_to_B) * 2 + distance_to_C.natAbs * (times_to_C * 2)

-- The proof problems (statements) based on the problem's questions
theorem distance_from_A_to_C_correct : distance_A_to_C = 45 := by
  sorry

theorem total_distance_traveled_correct : total_distance_traveled = 190 := by
  sorry

end distance_from_A_to_C_correct_total_distance_traveled_correct_l2089_208913


namespace max_vouchers_with_680_l2089_208956

def spend_to_voucher (spent : ℕ) : ℕ := (spent / 100) * 20

theorem max_vouchers_with_680 : spend_to_voucher 680 = 160 := by
  sorry

end max_vouchers_with_680_l2089_208956


namespace hotel_room_assignment_even_hotel_room_assignment_odd_l2089_208938

def smallest_n_even (k : ℕ) (m : ℕ) (h1 : k = 2 * m) : ℕ :=
  100 * (m + 1)

def smallest_n_odd (k : ℕ) (m : ℕ) (h1 : k = 2 * m + 1) : ℕ :=
  100 * (m + 1) + 1

theorem hotel_room_assignment_even (k m : ℕ) (h1 : k = 2 * m) :
  ∃ n, n = smallest_n_even k m h1 ∧ n >= 100 :=
  by
  sorry

theorem hotel_room_assignment_odd (k m : ℕ) (h1 : k = 2 * m + 1) :
  ∃ n, n = smallest_n_odd k m h1 ∧ n >= 100 :=
  by
  sorry

end hotel_room_assignment_even_hotel_room_assignment_odd_l2089_208938


namespace max_d_minus_r_proof_l2089_208917

noncomputable def max_d_minus_r : ℕ := 35

theorem max_d_minus_r_proof (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) :
  d - r ≤ max_d_minus_r :=
  sorry

end max_d_minus_r_proof_l2089_208917


namespace height_of_shorter_pot_is_20_l2089_208901

-- Define the conditions as given
def height_of_taller_pot := 40
def shadow_of_taller_pot := 20
def shadow_of_shorter_pot := 10

-- Define the height of the shorter pot to be determined
def height_of_shorter_pot (h : ℝ) := h

-- Define the relationship using the concept of similar triangles
theorem height_of_shorter_pot_is_20 (h : ℝ) :
  (height_of_taller_pot / shadow_of_taller_pot = height_of_shorter_pot h / shadow_of_shorter_pot) → h = 20 :=
by
  intros
  sorry

end height_of_shorter_pot_is_20_l2089_208901


namespace D_72_eq_93_l2089_208940

def D (n : ℕ) : ℕ :=
-- The function definition of D would go here, but we leave it abstract for now.
sorry

theorem D_72_eq_93 : D 72 = 93 :=
sorry

end D_72_eq_93_l2089_208940


namespace part1_part2_l2089_208948

theorem part1 (n : Nat) (hn : 0 < n) : 
  (∃ k, -5^4 + 5^5 + 5^n = k^2) -> n = 5 :=
by
  sorry

theorem part2 (n : Nat) (hn : 0 < n) : 
  (∃ m, 2^4 + 2^7 + 2^n = m^2) -> n = 8 :=
by
  sorry

end part1_part2_l2089_208948


namespace percentage_sum_l2089_208906

theorem percentage_sum {A B : ℝ} 
  (hA : 0.40 * A = 160) 
  (hB : (2/3) * B = 160) : 
  0.60 * (A + B) = 384 :=
by
  sorry

end percentage_sum_l2089_208906


namespace total_vegetables_l2089_208981

-- Definitions for the conditions in the problem
def cucumbers := 58
def carrots := cucumbers - 24
def tomatoes := cucumbers + 49
def radishes := carrots

-- Statement for the proof problem
theorem total_vegetables :
  cucumbers + carrots + tomatoes + radishes = 233 :=
by sorry

end total_vegetables_l2089_208981


namespace possible_dimensions_of_plot_l2089_208921

theorem possible_dimensions_of_plot (x : ℕ) :
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ 1000 * a + 100 * a + 10 * b + b = x * (x + 1)) →
  x = 33 ∨ x = 66 ∨ x = 99 :=
sorry

end possible_dimensions_of_plot_l2089_208921


namespace eval_expr_l2089_208949

def a := -1
def b := 1 / 7
def expr := (3 * a^3 - 2 * a * b + b^2) - 2 * (-a^3 - a * b + 4 * b^2)

theorem eval_expr : expr = -36 / 7 := by
  -- Inserting the proof using the original mathematical solution steps is not required here.
  sorry

end eval_expr_l2089_208949
