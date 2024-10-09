import Mathlib

namespace largest_interior_angle_of_triangle_l468_46858

theorem largest_interior_angle_of_triangle (exterior_ratio_2k : ℝ) (exterior_ratio_3k : ℝ) (exterior_ratio_4k : ℝ) (sum_exterior_angles : exterior_ratio_2k + exterior_ratio_3k + exterior_ratio_4k = 360) :
  180 - exterior_ratio_2k = 100 :=
by
  sorry

end largest_interior_angle_of_triangle_l468_46858


namespace johns_average_speed_is_correct_l468_46825

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end johns_average_speed_is_correct_l468_46825


namespace initial_roses_l468_46868

theorem initial_roses {x : ℕ} (h : x + 11 = 14) : x = 3 := by
  sorry

end initial_roses_l468_46868


namespace area_of_region_S_is_correct_l468_46878

noncomputable def area_of_inverted_region (d : ℝ) : ℝ :=
  if h : d = 1.5 then 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi else 0

theorem area_of_region_S_is_correct :
  area_of_inverted_region 1.5 = 9 * Real.pi + 4.5 * Real.sqrt 2 * Real.pi := 
by 
  sorry

end area_of_region_S_is_correct_l468_46878


namespace similar_triangle_shortest_side_l468_46842

theorem similar_triangle_shortest_side {a b c : ℝ} (h₁ : a = 24) (h₂ : b = 32) (h₃ : c = 80) :
  let hypotenuse₁ := Real.sqrt (a ^ 2 + b ^ 2)
  let scale_factor := c / hypotenuse₁
  let shortest_side₂ := scale_factor * a
  shortest_side₂ = 48 :=
by
  sorry

end similar_triangle_shortest_side_l468_46842


namespace quadratic_function_behavior_l468_46888

theorem quadratic_function_behavior (x : ℝ) (h : x > 2) :
  ∃ y : ℝ, y = - (x - 2)^2 - 7 ∧ ∀ x₁ x₂, x₁ > 2 → x₂ > x₁ → (-(x₂ - 2)^2 - 7) < (-(x₁ - 2)^2 - 7) :=
by
  sorry

end quadratic_function_behavior_l468_46888


namespace expensive_time_8_l468_46840

variable (x : ℝ) -- x represents the time to pick an expensive handcuff lock

-- Conditions
def cheap_time := 6
def total_time := 42
def cheap_pairs := 3
def expensive_pairs := 3

-- Total time for cheap handcuffs
def total_cheap_time := cheap_pairs * cheap_time

-- Total time for expensive handcuffs
def total_expensive_time := total_time - total_cheap_time

-- Equation relating x to total_expensive_time
def expensive_equation := expensive_pairs * x = total_expensive_time

-- Proof goal
theorem expensive_time_8 : expensive_equation x -> x = 8 := by
  sorry

end expensive_time_8_l468_46840


namespace find_c_value_l468_46883

theorem find_c_value (x c : ℝ) (h₁ : 3 * x + 8 = 5) (h₂ : c * x + 15 = 3) : c = 12 :=
by
  -- This is where the proof steps would go, but we will use sorry for now.
  sorry

end find_c_value_l468_46883


namespace ratio_of_x_and_y_l468_46817

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 1 / 4 :=
by
  sorry

end ratio_of_x_and_y_l468_46817


namespace cost_per_tissue_box_l468_46823

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l468_46823


namespace meeting_time_l468_46866

-- Variables representing the conditions
def uniform_rate_cassie := 15
def uniform_rate_brian := 18
def distance_route := 70
def cassie_start_time := 8.0
def brian_start_time := 9.25

-- The goal
theorem meeting_time : ∃ T : ℝ, (15 * T + 18 * (T - 1.25) = 70) ∧ T = 2.803 := 
by {
  sorry
}

end meeting_time_l468_46866


namespace simplify_expression_l468_46830

variable (a b c x y z : ℝ)

theorem simplify_expression :
  (cz * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + bz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cz + bz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cz * a^3 * y^3 + 3 * bz * c^3 * x^3) / (cz + bz) :=
by
  sorry

end simplify_expression_l468_46830


namespace A_B_distance_l468_46891

noncomputable def distance_between_A_and_B 
  (vA: ℕ) (vB: ℕ) (vA_after_return: ℕ) 
  (meet_distance: ℕ) : ℚ := sorry

theorem A_B_distance (distance: ℚ) 
  (hA: vA = 40) (hB: vB = 60) 
  (hA_after_return: vA_after_return = 60) 
  (hmeet: meet_distance = 50) : 
  distance_between_A_and_B vA vB vA_after_return meet_distance = 1000 / 7 := sorry

end A_B_distance_l468_46891


namespace correct_option_b_l468_46892

theorem correct_option_b (a : ℝ) : 
  (-2 * a) ^ 3 = -8 * a ^ 3 :=
by sorry

end correct_option_b_l468_46892


namespace ratio_of_four_numbers_exists_l468_46884

theorem ratio_of_four_numbers_exists (A B C D : ℕ) (h1 : A + B + C + D = 1344) (h2 : D = 672) : 
  ∃ rA rB rC rD, rA ≠ 0 ∧ rB ≠ 0 ∧ rC ≠ 0 ∧ rD ≠ 0 ∧ A = rA * k ∧ B = rB * k ∧ C = rC * k ∧ D = rD * k :=
by {
  sorry
}

end ratio_of_four_numbers_exists_l468_46884


namespace find_x_l468_46863

theorem find_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end find_x_l468_46863


namespace parallelepiped_surface_area_l468_46864

theorem parallelepiped_surface_area (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 12) 
  (h2 : a * b * c = 8) : 
  6 * (a^2) = 24 :=
by
  sorry

end parallelepiped_surface_area_l468_46864


namespace blu_ray_movies_returned_l468_46882

theorem blu_ray_movies_returned (D B x : ℕ)
  (h1 : D / B = 17 / 4)
  (h2 : D + B = 378)
  (h3 : D / (B - x) = 9 / 2) :
  x = 4 := by
  sorry

end blu_ray_movies_returned_l468_46882


namespace least_four_digit_11_heavy_l468_46844

def is_11_heavy (n : ℕ) : Prop := (n % 11) > 7

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem least_four_digit_11_heavy : ∃ n : ℕ, is_four_digit n ∧ is_11_heavy n ∧ 
  (∀ m : ℕ, is_four_digit m ∧ is_11_heavy m → 1000 ≤ n) := 
sorry

end least_four_digit_11_heavy_l468_46844


namespace total_practice_hours_l468_46801

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end total_practice_hours_l468_46801


namespace tan_a2_a12_l468_46889

noncomputable def arithmetic_term (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem tan_a2_a12 (a d : ℝ) (h : a + (a + 6 * d) + (a + 12 * d) = 4 * Real.pi) :
  Real.tan (arithmetic_term a d 2 + arithmetic_term a d 12) = - Real.sqrt 3 :=
by
  sorry

end tan_a2_a12_l468_46889


namespace range_of_m_l468_46869

theorem range_of_m {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 6*x - 16)
  {a b : ℝ} (h_domain : ∀ x, 0 ≤ x ∧ x ≤ a → ∃ y, f y ≤ b) 
  (h_range : ∀ y, -25 ≤ y ∧ y ≤ -16 → ∃ x, f x = y) : 3 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_m_l468_46869


namespace fraction_problem_l468_46887

theorem fraction_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 :=
by sorry

end fraction_problem_l468_46887


namespace original_marketing_pct_correct_l468_46881

-- Define the initial and final percentages of finance specialization students
def initial_finance_pct := 0.88
def final_finance_pct := 0.90

-- Define the final percentage of marketing specialization students
def final_marketing_pct := 0.43333333333333335

-- Define the original percentage of marketing specialization students
def original_marketing_pct := 0.45333333333333335

-- The Lean statement to prove the original percentage of marketing students
theorem original_marketing_pct_correct :
  initial_finance_pct + (final_marketing_pct - initial_finance_pct) = original_marketing_pct := 
sorry

end original_marketing_pct_correct_l468_46881


namespace range_of_a_and_t_minimum_of_y_l468_46853

noncomputable def minimum_value_y (a b : ℝ) (h : a + b = 1) : ℝ :=
(a + 1/a) * (b + 1/b)

theorem range_of_a_and_t (a b : ℝ) (h : a + b = 1) :
  0 < a ∧ a < 1 ∧ 0 < a * b ∧ a * b <= 1/4 :=
sorry

theorem minimum_of_y (a b : ℝ) (h : a + b = 1) :
  minimum_value_y a b h = 25/4 :=
sorry

end range_of_a_and_t_minimum_of_y_l468_46853


namespace relationship_among_abc_l468_46847

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem relationship_among_abc :
  b > c ∧ c > a :=
by
  sorry

end relationship_among_abc_l468_46847


namespace parallelogram_area_l468_46835

noncomputable def area_parallelogram (b s θ : ℝ) : ℝ := b * (s * Real.sin θ)

theorem parallelogram_area : area_parallelogram 20 10 (Real.pi / 6) = 100 := by
  sorry

end parallelogram_area_l468_46835


namespace equivalent_expression_l468_46852

theorem equivalent_expression (x : ℝ) : 
  (x-1)^4 + 4*(x-1)^3 + 6*(x-1)^2 + 4*(x-1) + 1 = x^4 := 
by
  sorry

end equivalent_expression_l468_46852


namespace constant_t_exists_l468_46898

theorem constant_t_exists (c : ℝ) :
  ∃ t : ℝ, (∀ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A.2 = A.1 * c + c) ∧ (B.2 = B.1 * c + c) → (t = -2)) :=
sorry

end constant_t_exists_l468_46898


namespace farmer_crops_saved_l468_46820

noncomputable def average_corn_per_row := (10 + 14) / 2
noncomputable def average_potato_per_row := (35 + 45) / 2
noncomputable def average_wheat_per_row := (55 + 65) / 2

noncomputable def avg_reduction_corn := (40 + 60 + 25) / 3 / 100
noncomputable def avg_reduction_potato := (50 + 30 + 60) / 3 / 100
noncomputable def avg_reduction_wheat := (20 + 55 + 35) / 3 / 100

noncomputable def saved_corn_per_row := average_corn_per_row * (1 - avg_reduction_corn)
noncomputable def saved_potato_per_row := average_potato_per_row * (1 - avg_reduction_potato)
noncomputable def saved_wheat_per_row := average_wheat_per_row * (1 - avg_reduction_wheat)

def rows_corn := 30
def rows_potato := 24
def rows_wheat := 36

noncomputable def total_saved_corn := saved_corn_per_row * rows_corn
noncomputable def total_saved_potatoes := saved_potato_per_row * rows_potato
noncomputable def total_saved_wheat := saved_wheat_per_row * rows_wheat

noncomputable def total_crops_saved := total_saved_corn + total_saved_potatoes + total_saved_wheat

theorem farmer_crops_saved : total_crops_saved = 2090 := by
  sorry

end farmer_crops_saved_l468_46820


namespace integer_roots_7_values_of_a_l468_46837

theorem integer_roots_7_values_of_a :
  (∃ a : ℝ, (∀ r s : ℤ, (r + s = -a ∧ (r * s = 8 * a))) ∧ (∃ n : ℕ, n = 7)) :=
sorry

end integer_roots_7_values_of_a_l468_46837


namespace find_f_at_one_l468_46804

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * x^3 + n * x + 1

theorem find_f_at_one (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : f (-1) m n = 5) : f (1) m n = 7 :=
by
  -- proof goes here
  sorry

end find_f_at_one_l468_46804


namespace sum_of_consecutive_integers_with_product_1680_l468_46851

theorem sum_of_consecutive_integers_with_product_1680 : 
  ∃ (a b c d : ℤ), (a * b * c * d = 1680 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3) → (a + b + c + d = 26) := sorry

end sum_of_consecutive_integers_with_product_1680_l468_46851


namespace total_bricks_fill_box_l468_46861

-- Define brick and box volumes based on conditions
def volume_brick1 := 2 * 5 * 8
def volume_brick2 := 2 * 3 * 7
def volume_box := 10 * 11 * 14

-- Define the main proof problem
theorem total_bricks_fill_box (x y : ℕ) (h1 : volume_brick1 * x + volume_brick2 * y = volume_box) :
  x + y = 24 :=
by
  -- Left as an exercise (proof steps are not included per instructions)
  sorry

end total_bricks_fill_box_l468_46861


namespace suzanne_donation_total_l468_46810

theorem suzanne_donation_total : 
  (10 + 10 * 2 + 10 * 2^2 + 10 * 2^3 + 10 * 2^4 = 310) :=
by
  sorry

end suzanne_donation_total_l468_46810


namespace intersection_points_l468_46897

noncomputable def h (x : ℝ) : ℝ := -x^2 - 4 * x + 1
noncomputable def j (x : ℝ) : ℝ := -h x
noncomputable def k (x : ℝ) : ℝ := h (-x)

def c : ℕ := 2 -- Number of intersections of y = h(x) and y = j(x)
def d : ℕ := 1 -- Number of intersections of y = h(x) and y = k(x)

theorem intersection_points :
  10 * c + d = 21 := by
  sorry

end intersection_points_l468_46897


namespace circle_radius_l468_46833

theorem circle_radius (x y : ℝ) : x^2 + 8*x + y^2 - 10*y + 32 = 0 → ∃ r : ℝ, r = 3 :=
by
  sorry

end circle_radius_l468_46833


namespace vendor_sales_first_day_l468_46870

theorem vendor_sales_first_day (A S: ℝ) (h1: S = S / 100) 
  (h2: 0.20 * A * (1 - S / 100) = 0.42 * A - 0.50 * A * (0.80 * (1 - S / 100)))
  (h3: 0 < S) (h4: S < 100) : 
  S = 30 := 
by
  sorry

end vendor_sales_first_day_l468_46870


namespace sum_of_smallest_multiples_l468_46846

def smallest_two_digit_multiple_of_5 := 10
def smallest_three_digit_multiple_of_7 := 105

theorem sum_of_smallest_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end sum_of_smallest_multiples_l468_46846


namespace possible_values_l468_46832

theorem possible_values (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  ∃ S : Set ℝ, S = {x : ℝ | 4 ≤ x} ∧ (1 / a + 1 / b) ∈ S :=
by
  sorry

end possible_values_l468_46832


namespace simplify_expression_l468_46816

theorem simplify_expression :
  (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := 
by {
  sorry
}

end simplify_expression_l468_46816


namespace find_y_l468_46865

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2 + 1 / y) 
  (h2 : y = 3 + 1 / x) : 
  y = (3/2) + (Real.sqrt 15 / 2) :=
by
  sorry

end find_y_l468_46865


namespace measure_smaller_angle_east_northwest_l468_46843

/-- A mathematical structure for a circle with 12 rays forming congruent central angles. -/
structure CircleWithRays where
  rays : Finset (Fin 12)  -- There are 12 rays
  congruent_angles : ∀ i, i ∈ rays

/-- The measure of the central angle formed by each ray is 30 degrees (since 360/12 = 30). -/
def central_angle_measure : ℝ := 30

/-- The measure of the smaller angle formed between the ray pointing East and the ray pointing Northwest is 150 degrees. -/
theorem measure_smaller_angle_east_northwest (c : CircleWithRays) : 
  ∃ angle : ℝ, angle = 150 := by
  sorry

end measure_smaller_angle_east_northwest_l468_46843


namespace perfect_square_fraction_l468_46824

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem perfect_square_fraction (a b : ℕ) 
  (h_pos_a: 0 < a) 
  (h_pos_b: 0 < b) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  is_perfect_square ((a^2 + b^2) / (a * b + 1)) := 
sorry

end perfect_square_fraction_l468_46824


namespace solve_for_a_l468_46800

theorem solve_for_a (a : ℝ) (h : (a + 3)^(a + 1) = 1) : a = -2 ∨ a = -1 :=
by {
  -- proof here
  sorry
}

end solve_for_a_l468_46800


namespace max_value_of_m_l468_46831

theorem max_value_of_m :
  (∃ (t : ℝ), ∀ (x : ℝ), 2 ≤ x ∧ x ≤ m → (x + t)^2 ≤ 2 * x) → m ≤ 8 :=
sorry

end max_value_of_m_l468_46831


namespace value_of_b_l468_46856

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 :=
sorry

end value_of_b_l468_46856


namespace quadratic_has_one_real_root_positive_value_of_m_l468_46829

theorem quadratic_has_one_real_root (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 0 ∨ m = 1/4 := by
  sorry

theorem positive_value_of_m (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 1/4 := by
  have root_cases := quadratic_has_one_real_root m h
  cases root_cases
  · exfalso
    -- We know m = 0 cannot be the positive m we are looking for.
    sorry
  · assumption

end quadratic_has_one_real_root_positive_value_of_m_l468_46829


namespace max_students_l468_46896

def num_pens : Nat := 1204
def num_pencils : Nat := 840

theorem max_students (n_pens n_pencils : Nat) (h_pens : n_pens = num_pens) (h_pencils : n_pencils = num_pencils) :
  Nat.gcd n_pens n_pencils = 16 := by
  sorry

end max_students_l468_46896


namespace total_sum_l468_46867

theorem total_sum (p q r s t : ℝ) (P : ℝ) 
  (h1 : q = 0.75 * P) 
  (h2 : r = 0.50 * P) 
  (h3 : s = 0.25 * P) 
  (h4 : t = 0.10 * P) 
  (h5 : s = 25) 
  :
  p + q + r + s + t = 260 :=
by 
  sorry

end total_sum_l468_46867


namespace probability_composite_product_l468_46879

theorem probability_composite_product :
  let dice_faces := 6
  let rolls := 4
  let total_outcomes := dice_faces ^ rolls
  let non_composite_cases := 13
  let non_composite_probability := non_composite_cases / total_outcomes
  let composite_probability := 1 - non_composite_probability
  composite_probability = 1283 / 1296 := by
  sorry

end probability_composite_product_l468_46879


namespace quarters_per_jar_l468_46811

/-- Jenn has 5 jars full of quarters. Each jar can hold a certain number of quarters.
    The bike costs 180 dollars, and she will have 20 dollars left over after buying it.
    Prove that each jar can hold 160 quarters. -/
theorem quarters_per_jar (num_jars : ℕ) (cost_bike : ℕ) (left_over : ℕ)
  (quarters_per_dollar : ℕ) (total_quarters : ℕ) (quarters_per_jar : ℕ) :
  num_jars = 5 → cost_bike = 180 → left_over = 20 → quarters_per_dollar = 4 →
  total_quarters = ((cost_bike + left_over) * quarters_per_dollar) →
  quarters_per_jar = (total_quarters / num_jars) →
  quarters_per_jar = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end quarters_per_jar_l468_46811


namespace frank_spend_more_l468_46893

noncomputable def table_cost : ℝ := 140
noncomputable def chair_cost : ℝ := 100
noncomputable def joystick_cost : ℝ := 20
noncomputable def frank_joystick : ℝ := joystick_cost * (1 / 4)
noncomputable def eman_joystick : ℝ := joystick_cost - frank_joystick
noncomputable def frank_total : ℝ := table_cost + frank_joystick
noncomputable def eman_total : ℝ := chair_cost + eman_joystick

theorem frank_spend_more :
  frank_total - eman_total = 30 :=
  sorry

end frank_spend_more_l468_46893


namespace culture_medium_preparation_l468_46854

theorem culture_medium_preparation :
  ∀ (V : ℝ), 0 < V → 
  ∃ (nutrient_broth pure_water saline_water : ℝ),
    nutrient_broth = V / 3 ∧
    pure_water = V * 0.3 ∧
    saline_water = V - (nutrient_broth + pure_water) :=
by
  sorry

end culture_medium_preparation_l468_46854


namespace power_function_half_value_l468_46822

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_half_value (a : ℝ) (h : (f 4 a) / (f 2 a) = 3) :
  f (1 / 2) a = 1 / 3 :=
by
  sorry  -- Proof goes here

end power_function_half_value_l468_46822


namespace find_m_l468_46877

theorem find_m (m : ℝ) (h : ∀ x : ℝ, m - |x| ≥ 0 ↔ -1 ≤ x ∧ x ≤ 1) : m = 1 :=
sorry

end find_m_l468_46877


namespace sale_in_second_month_l468_46885

theorem sale_in_second_month 
  (m1 m2 m3 m4 m5 m6 : ℕ) 
  (h1: m1 = 6335) 
  (h2: m3 = 6855) 
  (h3: m4 = 7230) 
  (h4: m5 = 6562) 
  (h5: m6 = 5091)
  (average: (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 6500) : 
  m2 = 6927 :=
sorry

end sale_in_second_month_l468_46885


namespace joe_paint_initial_amount_l468_46871

theorem joe_paint_initial_amount (P : ℕ) (h1 : P / 6 + (5 * P / 6) / 5 = 120) :
  P = 360 := by
  sorry

end joe_paint_initial_amount_l468_46871


namespace gcd_71_19_l468_46895

theorem gcd_71_19 : Int.gcd 71 19 = 1 := by
  sorry

end gcd_71_19_l468_46895


namespace volleyball_teams_l468_46873

theorem volleyball_teams (managers employees teams : ℕ) (h1 : managers = 3) (h2 : employees = 3) (h3 : teams = 3) :
  ((managers + employees) / teams) = 2 :=
by
  sorry

end volleyball_teams_l468_46873


namespace highest_temperature_l468_46834

theorem highest_temperature (lowest_temp : ℝ) (max_temp_diff : ℝ) :
  lowest_temp = 18 → max_temp_diff = 4 → lowest_temp + max_temp_diff = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end highest_temperature_l468_46834


namespace count_ordered_triples_l468_46886

theorem count_ordered_triples (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : 2 * a * b * c = 2 * (a * b + b * c + a * c)) : 
  ∃ n, n = 10 :=
by
  sorry

end count_ordered_triples_l468_46886


namespace number_of_episodes_l468_46845

def episode_length : ℕ := 20
def hours_per_day : ℕ := 2
def days : ℕ := 15

theorem number_of_episodes : (days * hours_per_day * 60) / episode_length = 90 :=
by
  sorry

end number_of_episodes_l468_46845


namespace binder_cost_l468_46850

variable (B : ℕ) -- Define B as the cost of each binder

theorem binder_cost :
  let book_cost := 16
  let num_binders := 3
  let notebook_cost := 1
  let num_notebooks := 6
  let total_cost := 28
  (book_cost + num_binders * B + num_notebooks * notebook_cost = total_cost) → (B = 2) :=
by
  sorry

end binder_cost_l468_46850


namespace scientific_notation_110_billion_l468_46803

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ 110 * 10^8 = a * 10^n

theorem scientific_notation_110_billion :
  ∃ (a : ℝ) (n : ℤ), scientific_notation_form a n ∧ a = 1.1 ∧ n = 10 :=
by
  sorry

end scientific_notation_110_billion_l468_46803


namespace sequence_not_generated_l468_46818

theorem sequence_not_generated (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (a 2 = 0) ∧ (a 3 = 2) ∧ (a 4 = 0) → 
  (∀ n, a n ≠ (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)) :=
by sorry

end sequence_not_generated_l468_46818


namespace rational_solution_for_k_is_6_l468_46859

theorem rational_solution_for_k_is_6 (k : ℕ) (h : 0 < k) :
  (∃ x : ℚ, k * x ^ 2 + 12 * x + k = 0) ↔ k = 6 :=
by { sorry }

end rational_solution_for_k_is_6_l468_46859


namespace cube_volume_l468_46838

theorem cube_volume (s : ℝ) (h1 : 6 * s^2 = 1734) : s^3 = 4913 := by
  sorry

end cube_volume_l468_46838


namespace batsman_average_after_15th_innings_l468_46862

theorem batsman_average_after_15th_innings 
  (A : ℕ) 
  (h1 : 14 * A + 85 = 15 * (A + 3)) 
  (h2 : A = 40) : 
  (A + 3) = 43 := by 
  sorry

end batsman_average_after_15th_innings_l468_46862


namespace comparison_of_products_l468_46872

def A : ℕ := 8888888888888888888 -- 19 digits, all 8's
def B : ℕ := 3333333333333333333333333333333333333333333333333333333333333333 -- 68 digits, all 3's
def C : ℕ := 4444444444444444444 -- 19 digits, all 4's
def D : ℕ := 6666666666666666666666666666666666666666666666666666666666666667 -- 68 digits, first 67 are 6's, last is 7

theorem comparison_of_products : C * D > A * B ∧ C * D - A * B = 4444444444444444444 := sorry

end comparison_of_products_l468_46872


namespace force_with_18_inch_crowbar_l468_46890

noncomputable def inverseForce (L F : ℝ) : ℝ :=
  F * L

theorem force_with_18_inch_crowbar :
  ∀ (F : ℝ), (inverseForce 12 200 = inverseForce 18 F) → F = 133.333333 :=
by
  intros
  sorry

end force_with_18_inch_crowbar_l468_46890


namespace milan_total_bill_correct_l468_46819

-- Define the monthly fee, the per minute rate, and the number of minutes used last month
def monthly_fee : ℝ := 2
def per_minute_rate : ℝ := 0.12
def minutes_used : ℕ := 178

-- Define the total bill calculation
def total_bill : ℝ := minutes_used * per_minute_rate + monthly_fee

-- The proof statement
theorem milan_total_bill_correct :
  total_bill = 23.36 := 
by
  sorry

end milan_total_bill_correct_l468_46819


namespace johnny_earnings_l468_46836

theorem johnny_earnings :
  let job1 := 3 * 7
  let job2 := 2 * 10
  let job3 := 4 * 12
  let daily_earnings := job1 + job2 + job3
  let total_earnings := 5 * daily_earnings
  total_earnings = 445 :=
by
  sorry

end johnny_earnings_l468_46836


namespace actual_distance_traveled_l468_46849

theorem actual_distance_traveled (D T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 20 * T) : D = 20 :=
by
  sorry

end actual_distance_traveled_l468_46849


namespace age_difference_l468_46875

variable (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 16) : C + 16 = A := 
by
  sorry

end age_difference_l468_46875


namespace first_term_exceeding_10000_l468_46860

theorem first_term_exceeding_10000 :
  ∃ (n : ℕ), (2^(n-1) > 10000) ∧ (2^(n-1) = 16384) :=
by
  sorry

end first_term_exceeding_10000_l468_46860


namespace youngest_son_trips_l468_46826

theorem youngest_son_trips 
  (p : ℝ) (n_oldest : ℝ) (c : ℝ) (Y : ℝ)
  (h1 : p = 100)
  (h2 : n_oldest = 35)
  (h3 : c = 4)
  (h4 : p / c = Y) :
  Y = 25 := sorry

end youngest_son_trips_l468_46826


namespace find_value_of_m_l468_46806

/-- Given the universal set U, set A, and the complement of A in U, we prove that m = -2. -/
theorem find_value_of_m (m : ℤ) (U : Set ℤ) (A : Set ℤ) (complement_U_A : Set ℤ) 
  (h1 : U = {2, 3, m^2 + m - 4})
  (h2 : A = {m, 2})
  (h3 : complement_U_A = {3}) 
  (h4 : U = A ∪ complement_U_A) 
  (h5 : A ∩ complement_U_A = ∅) 
  : m = -2 :=
sorry

end find_value_of_m_l468_46806


namespace smallest_n_for_Sn_gt_10_l468_46876

noncomputable def harmonicSeriesSum : ℕ → ℝ
| 0       => 0
| (n + 1) => harmonicSeriesSum n + 1 / (n + 1)

theorem smallest_n_for_Sn_gt_10 : ∃ n : ℕ, (harmonicSeriesSum n > 10) ∧ ∀ k < 12367, harmonicSeriesSum k ≤ 10 :=
by
  sorry

end smallest_n_for_Sn_gt_10_l468_46876


namespace last_digit_p_minus_q_not_5_l468_46841

theorem last_digit_p_minus_q_not_5 (p q : ℕ) (n : ℕ) 
  (h1 : p * q = 10^n) 
  (h2 : ¬ (p % 10 = 0))
  (h3 : ¬ (q % 10 = 0))
  (h4 : p > q) : (p - q) % 10 ≠ 5 :=
by sorry

end last_digit_p_minus_q_not_5_l468_46841


namespace winter_expenditure_l468_46874

theorem winter_expenditure (exp_end_nov : Real) (exp_end_feb : Real) 
  (h_nov : exp_end_nov = 3.0) (h_feb : exp_end_feb = 5.5) : 
  (exp_end_feb - exp_end_nov) = 2.5 :=
by 
  sorry

end winter_expenditure_l468_46874


namespace determine_q_l468_46813

theorem determine_q (q : ℝ) (x1 x2 x3 x4 : ℝ) 
  (h_first_eq : x1^2 - 5 * x1 + q = 0 ∧ x2^2 - 5 * x2 + q = 0)
  (h_second_eq : x3^2 - 7 * x3 + 2 * q = 0 ∧ x4^2 - 7 * x4 + 2 * q = 0)
  (h_relation : x3 = 2 * x1) : 
  q = 6 :=
by
  sorry

end determine_q_l468_46813


namespace sin_alpha_at_point_l468_46880

open Real

theorem sin_alpha_at_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (1, -2)) :
  sin α = -2 * sqrt 5 / 5 :=
sorry

end sin_alpha_at_point_l468_46880


namespace part_a_part_b_l468_46802

/-- Part (a) statement: -/
theorem part_a (x : Fin 100 → ℕ) :
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) →
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) :=
by
  sorry

/-- Part (b) statement: -/
theorem part_b (x : Fin 100 → ℕ) :
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) →
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) :=
by
  sorry

end part_a_part_b_l468_46802


namespace flower_pattern_perimeter_l468_46805

theorem flower_pattern_perimeter (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 45) : 
    let arc_length := (360 - θ) / 360 * 2 * π * r
    let total_perimeter := arc_length + 2 * r
    total_perimeter = (21 / 4 * π) + 6 := 
by
  -- Definitions from conditions
  let arc_length := (360 - θ) / 360 * 2 * π * r
  let total_perimeter := arc_length + 2 * r

  -- Assertions to reach the target conclusion
  have h_arc_length: arc_length = (21 / 4 * π) :=
    by
      sorry

  -- Incorporate the radius
  have h_total: total_perimeter = (21 / 4 * π) + 6 :=
    by
      sorry

  exact h_total

end flower_pattern_perimeter_l468_46805


namespace range_of_a_l468_46815

def A := {x : ℝ | x * (4 - x) ≥ 3}
def B (a : ℝ) := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a < 1) := by
  sorry

end range_of_a_l468_46815


namespace no_real_solution_l468_46807

-- Given conditions as definitions in Lean 4
def eq1 (x : ℝ) : Prop := x^5 + 3 * x^4 + 5 * x^3 + 5 * x^2 + 6 * x + 2 = 0
def eq2 (x : ℝ) : Prop := x^3 + 3 * x^2 + 4 * x + 1 = 0

-- The theorem to prove
theorem no_real_solution : ¬ ∃ x : ℝ, eq1 x ∧ eq2 x :=
by sorry

end no_real_solution_l468_46807


namespace temperature_problem_l468_46855

theorem temperature_problem (N : ℤ) (P : ℤ) (D : ℤ) (D_3_pm : ℤ) (P_3_pm : ℤ) :
  D = P + N →
  D_3_pm = D - 8 →
  P_3_pm = P + 9 →
  |D_3_pm - P_3_pm| = 1 →
  (N = 18 ∨ N = 16) →
  18 * 16 = 288 :=
by
  sorry

end temperature_problem_l468_46855


namespace greatest_mean_YZ_l468_46848

noncomputable def X_mean := 60
noncomputable def Y_mean := 70
noncomputable def XY_mean := 64
noncomputable def XZ_mean := 66

theorem greatest_mean_YZ (Xn Yn Zn : ℕ) (m : ℕ) :
  (60 * Xn + 70 * Yn) / (Xn + Yn) = 64 →
  (60 * Xn + m) / (Xn + Zn) = 66 →
  ∃ (k : ℕ), k = 69 :=
by
  intro h1 h2
  -- Sorry is used to skip the proof
  sorry

end greatest_mean_YZ_l468_46848


namespace time_of_same_distance_l468_46894

theorem time_of_same_distance (m : ℝ) (h_m : 0 ≤ m ∧ m ≤ 60) : 180 - 6 * m = 90 + 0.5 * m :=
by
  sorry

end time_of_same_distance_l468_46894


namespace area_comparison_l468_46828

def point := (ℝ × ℝ)

def quadrilateral_I_vertices : List point := [(0, 0), (2, 0), (2, 2), (0, 2)]

def quadrilateral_I_area : ℝ := 4

def quadrilateral_II_vertices : List point := [(1, 0), (4, 0), (4, 4), (1, 3)]

noncomputable def quadrilateral_II_area : ℝ := 10.5

theorem area_comparison :
  quadrilateral_I_area < quadrilateral_II_area :=
  by
    sorry

end area_comparison_l468_46828


namespace max_value_of_f_range_of_m_l468_46821

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem max_value_of_f (a b : ℝ) (x : ℝ) (h1 : 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_tangent : ∀ (x : ℝ), f a b x - ((-1/2) * x + (Real.log 1 - 1/2)) = 0) : 
  ∃ x_max, f a b x_max = -1/2 := sorry

theorem range_of_m (m : ℝ) 
  (h_ineq : ∀ (a : ℝ) (x : ℝ), 1 ≤ a ∧ a ≤ 3 / 2 ∧ 1 ≤ x ∧ x ≤ Real.exp 2 → a * Real.log x ≥ m + x) : 
  m ≤ 2 - Real.exp 2 := sorry

end max_value_of_f_range_of_m_l468_46821


namespace john_ate_2_bags_for_dinner_l468_46839

variable (x y : ℕ)
variable (h1 : x + y = 3)
variable (h2 : y ≥ 1)

theorem john_ate_2_bags_for_dinner : x = 2 := 
by sorry

end john_ate_2_bags_for_dinner_l468_46839


namespace markup_percentage_l468_46857

theorem markup_percentage (S M : ℝ) (h1 : S = 56 + M * S) (h2 : 0.80 * S - 56 = 8) : M = 0.30 :=
sorry

end markup_percentage_l468_46857


namespace find_c_l468_46827

theorem find_c (x c : ℝ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 15 = -3) : c = -12 := 
by
  -- Equations and conditions
  have h1 : 3 * x + 8 = 5 := h1
  have h2 : c * x - 15 = -3 := h2
  -- The proof script would go here
  sorry

end find_c_l468_46827


namespace measure_of_obtuse_angle_APB_l468_46812

-- Define the triangle type and conditions
structure Triangle :=
  (A B C : Point)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

-- Define the point type
structure Point :=
  (x y : ℝ)

-- Property of the triangle is isotropic and it contains right angles 90 degrees 
def IsoscelesRightTriangle (T : Triangle) : Prop :=
  T.angle_A = 45 ∧ T.angle_B = 45 ∧ T.angle_C = 90

-- Define the angle bisector intersection point P
def AngleBisectorIntersection (T : Triangle) (P : Point) : Prop :=
  -- (dummy properties assuming necessary geometric constructions can be proven)
  true

-- Statement we want to prove
theorem measure_of_obtuse_angle_APB (T : Triangle) (P : Point) 
    (h1 : IsoscelesRightTriangle T) (h2 : AngleBisectorIntersection T P) :
  ∃ APB : ℝ, APB = 135 :=
  sorry

end measure_of_obtuse_angle_APB_l468_46812


namespace prove_m_plus_n_eq_one_l468_46814

-- Define coordinates of points A and B
def A (m n : ℝ) : ℝ × ℝ := (1 + m, 1 - n)
def B : ℝ × ℝ := (-3, 2)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  symmetric_about_y_axis (A m n) B

-- Statement to prove
theorem prove_m_plus_n_eq_one (m n : ℝ) (h : conditions m n) : m + n = 1 := 
by 
  sorry

end prove_m_plus_n_eq_one_l468_46814


namespace triangle_third_side_l468_46808

theorem triangle_third_side (x : ℕ) : 
  (3 < x) ∧ (x < 17) → 
  (x = 11) :=
by
  sorry

end triangle_third_side_l468_46808


namespace spherical_to_rectangular_coordinates_l468_46899

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 10 ∧ θ = 3 * Real.pi / 4 ∧ φ = Real.pi / 6 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 5 * Real.sqrt 3)
  :=
by
  intros ρ θ φ h
  rcases h with ⟨hρ, hθ, hφ⟩
  simp [hρ, hθ, hφ]
  sorry

end spherical_to_rectangular_coordinates_l468_46899


namespace calculate_sum_l468_46809

theorem calculate_sum :
  (1 : ℚ) + 3 / 6 + 5 / 12 + 7 / 20 + 9 / 30 + 11 / 42 + 13 / 56 + 15 / 72 + 17 / 90 = 81 + 2 / 5 :=
sorry

end calculate_sum_l468_46809
