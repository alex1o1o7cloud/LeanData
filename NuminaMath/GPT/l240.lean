import Mathlib

namespace overall_winning_percentage_is_fifty_l240_240245

def winning_percentage_of_first_games := (40 / 100) * 30
def total_games_played := 40
def remaining_games := total_games_played - 30
def winning_percentage_of_remaining_games := (80 / 100) * remaining_games
def total_games_won := winning_percentage_of_first_games + winning_percentage_of_remaining_games

theorem overall_winning_percentage_is_fifty : 
  (total_games_won / total_games_played) * 100 = 50 := 
by
  sorry

end overall_winning_percentage_is_fifty_l240_240245


namespace geometric_series_sum_frac_l240_240710

open BigOperators

theorem geometric_series_sum_frac (q : ℚ) (a1 : ℚ) (a_list: List ℚ) (h_theta : q = 1 / 2) 
(h_a_list : a_list ⊆ [-4, -3, -2, 0, 1, 23, 4]) : 
  a1 * (1 + q^5) / (1 - q) = 33 / 4 := by
  sorry

end geometric_series_sum_frac_l240_240710


namespace part1_part2_l240_240444

def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * m * x - 4 ≤ 0}

-- Problem 1
theorem part1 (m : ℝ) : 
  (A ∩ B m = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) → m = 3 :=
by sorry

-- Problem 2
theorem part2 (m : ℝ) : 
  (A ⊆ (B m)ᶜ) → (m < -3 ∨ m > 5) :=
by sorry

end part1_part2_l240_240444


namespace ratio_M_N_l240_240450

variable (M Q P N : ℝ)

-- Conditions
axiom h1 : M = 0.40 * Q
axiom h2 : Q = 0.25 * P
axiom h3 : N = 0.60 * P

theorem ratio_M_N : M / N = 1 / 6 :=
by
  sorry

end ratio_M_N_l240_240450


namespace circle_radius_of_square_perimeter_eq_area_l240_240351

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l240_240351


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l240_240616

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l240_240616


namespace sample_size_correct_l240_240532

def total_students (freshmen sophomores juniors : ℕ) : ℕ :=
  freshmen + sophomores + juniors

def sample_size (total : ℕ) (prob : ℝ) : ℝ :=
  total * prob

theorem sample_size_correct (f : ℕ) (s : ℕ) (j : ℕ) (p : ℝ) (h_f : f = 400) (h_s : s = 320) (h_j : j = 280) (h_p : p = 0.2) :
  sample_size (total_students f s j) p = 200 :=
by
  sorry

end sample_size_correct_l240_240532


namespace regular_polygon_sides_l240_240836

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240836


namespace tan_expression_val_l240_240729

theorem tan_expression_val (A B : ℝ) (hA : A = 30) (hB : B = 15) :
  (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
by
  sorry

end tan_expression_val_l240_240729


namespace prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l240_240805

def num_outcomes := 36

def same_points_events := 6
def less_than_seven_events := 15
def greater_than_or_equal_eleven_events := 3

def prob_same_points := (same_points_events : ℚ) / num_outcomes
def prob_less_than_seven := (less_than_seven_events : ℚ) / num_outcomes
def prob_greater_or_equal_eleven := (greater_than_or_equal_eleven_events : ℚ) / num_outcomes

theorem prob_same_points_eq : prob_same_points = 1 / 6 := by
  sorry

theorem prob_less_than_seven_eq : prob_less_than_seven = 5 / 12 := by
  sorry

theorem prob_greater_or_equal_eleven_eq : prob_greater_or_equal_eleven = 1 / 12 := by
  sorry

end prob_same_points_eq_prob_less_than_seven_eq_prob_greater_or_equal_eleven_eq_l240_240805


namespace find_width_l240_240503

variable (L W : ℕ)

def perimeter (L W : ℕ) : ℕ := 2 * L + 2 * W

theorem find_width (h1 : perimeter L W = 46) (h2 : W = L + 7) : W = 15 :=
sorry

end find_width_l240_240503


namespace number_of_isosceles_triangles_with_perimeter_25_l240_240008

def is_isosceles_triangle (a b : ℕ) : Prop :=
  a + 2 * b = 25 ∧ 2 * b > a ∧ a < 2 * b

theorem number_of_isosceles_triangles_with_perimeter_25 :
  (finset.filter (λ b, ∃ a, is_isosceles_triangle a b)
                 (finset.range 13)).card = 6 := by
sorry

end number_of_isosceles_triangles_with_perimeter_25_l240_240008


namespace geometric_sequence_a3_value_l240_240990

noncomputable def geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem geometric_sequence_a3_value :
  ∃ a : ℕ → ℝ, ∃ r : ℝ,
  geometric_seq a r ∧
  a 1 = 2 ∧
  (a 3) * (a 5) = 4 * (a 6)^2 →
  a 3 = 1 :=
sorry

end geometric_sequence_a3_value_l240_240990


namespace initial_pages_l240_240653

variable (P : ℕ)
variable (h : 20 * P - 20 = 220)

theorem initial_pages (h : 20 * P - 20 = 220) : P = 12 := by
  sorry

end initial_pages_l240_240653


namespace square_feet_per_acre_l240_240672

-- Define the conditions
def rent_per_acre_per_month : ℝ := 60
def total_rent_per_month : ℝ := 600
def length_of_plot : ℝ := 360
def width_of_plot : ℝ := 1210

-- Translate the problem to a Lean theorem
theorem square_feet_per_acre :
  (length_of_plot * width_of_plot) / (total_rent_per_month / rent_per_acre_per_month) = 43560 :=
by {
  -- skipping the proof steps
  sorry
}

end square_feet_per_acre_l240_240672


namespace division_identity_l240_240377

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end division_identity_l240_240377


namespace circle_radius_eq_l240_240348

theorem circle_radius_eq {s r : ℝ} 
  (h1 : s * real.sqrt 2 = 2 * r)
  (h2 : 4 * s = real.pi * r^2) :
  r = 2 * real.sqrt 2 / real.pi :=
by
  sorry

end circle_radius_eq_l240_240348


namespace coordinates_C_on_segment_AB_l240_240474

theorem coordinates_C_on_segment_AB :
  ∃ C : (ℝ × ℝ), 
  (C.1 = 2 ∧ C.2 = 6) ∧
  ∃ A B : (ℝ × ℝ), 
  (A = (-1, 0)) ∧ 
  (B = (3, 8)) ∧ 
  (∃ k : ℝ, (k = 3) ∧ dist (C) (A) = k * dist (C) (B)) :=
by
  sorry

end coordinates_C_on_segment_AB_l240_240474


namespace x_squared_plus_y_squared_l240_240180

theorem x_squared_plus_y_squared (x y : ℝ) (h₁ : x - y = 18) (h₂ : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end x_squared_plus_y_squared_l240_240180


namespace an_is_arithmetic_sum_bn_l240_240277

noncomputable def an_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n:ℕ, 8 * S n = (a n + 2) ^ 2) ∧ ∃ d, ∀ n:ℕ, a (n + 1) = a n + d

theorem an_is_arithmetic (a S : ℕ → ℝ) (h : ∀ n:ℕ, 8 * S n = (a n + 2) ^ 2) :
  ∃ d, ∀ n:ℕ, a (n + 1) = a n + d := sorry

noncomputable def bn_sum (a b : ℕ → ℝ) (T : ℕ → ℝ) : Prop :=
  (∀ n:ℕ, a n = Real.log (b n) / Real.log (Real.sqrt 3)) ∧
  (T = λ n, (3 * (9^n - 1)) / 8)

theorem sum_bn (a b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_a : ∀ n:ℕ, a n = Real.log (b n) / Real.log (Real.sqrt 3))
  (h_a_seq : ∃ d, ∀ n:ℕ, a (n + 1) = a n + d) :
  T = λ n, (3 * (9^n - 1)) / 8 := sorry

end an_is_arithmetic_sum_bn_l240_240277


namespace regular_polygon_sides_l240_240871

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240871


namespace retailer_profit_percentage_l240_240678

theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h_wholesale_price : wholesale_price = 108)
  (h_retail_price : retail_price = 144)
  (h_discount_rate : discount_rate = 0.10) :
  (retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price * 100 = 20 :=
by
  sorry

end retailer_profit_percentage_l240_240678


namespace multiplicative_inverse_137_391_l240_240690

theorem multiplicative_inverse_137_391 :
  ∃ (b : ℕ), (b ≤ 390) ∧ (137 * b) % 391 = 1 :=
sorry

end multiplicative_inverse_137_391_l240_240690


namespace integer_root_b_l240_240996

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l240_240996


namespace population_of_missing_village_l240_240366

theorem population_of_missing_village 
  (p1 p2 p3 p4 p5 p6 : ℕ) 
  (h1 : p1 = 803) 
  (h2 : p2 = 900) 
  (h3 : p3 = 1100) 
  (h4 : p4 = 1023) 
  (h5 : p5 = 945) 
  (h6 : p6 = 1249) 
  (avg_population : ℕ) 
  (h_avg : avg_population = 1000) :
  ∃ p7 : ℕ, p7 = 980 ∧ avg_population * 7 = p1 + p2 + p3 + p4 + p5 + p6 + p7 :=
by
  sorry

end population_of_missing_village_l240_240366


namespace range_of_constant_c_in_quadrant_I_l240_240584

theorem range_of_constant_c_in_quadrant_I (c : ℝ) (x y : ℝ)
  (h1 : x - 2 * y = 4)
  (h2 : 2 * c * x + y = 5)
  (hx_pos : x > 0)
  (hy_pos : y > 0) : 
  -1 / 4 < c ∧ c < 5 / 8 := 
sorry

end range_of_constant_c_in_quadrant_I_l240_240584


namespace mason_savings_fraction_l240_240757

theorem mason_savings_fraction (M p b : ℝ) (h : (1 / 4) * M = (2 / 5) * b * p) : 
  (M - b * p) / M = 3 / 8 :=
by 
  sorry

end mason_savings_fraction_l240_240757


namespace triangle_isosceles_l240_240187

theorem triangle_isosceles
  (A B C : ℝ) -- Angles of the triangle, A, B, and C
  (h1 : A = 2 * C) -- Condition 1: Angle A equals twice angle C
  (h2 : B = 2 * C) -- Condition 2: Angle B equals twice angle C
  (h3 : A + B + C = 180) -- Sum of angles in a triangle equals 180 degrees
  : A = B := -- Conclusion: with the conditions above, angles A and B are equal
by
  sorry

end triangle_isosceles_l240_240187


namespace find_a_equiv_l240_240993

noncomputable def A (a : ℝ) : Set ℝ := {1, 3, a^2}
noncomputable def B (a : ℝ) : Set ℝ := {1, 2 + a}

theorem find_a_equiv (a : ℝ) (h : A a ∪ B a = A a) : a = 2 :=
by
  sorry

end find_a_equiv_l240_240993


namespace no_real_roots_for_polynomial_l240_240255

theorem no_real_roots_for_polynomial :
  (∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + (5/2) ≠ 0) :=
by
  sorry

end no_real_roots_for_polynomial_l240_240255


namespace regular_polygon_sides_l240_240903

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240903


namespace T_5_3_l240_240152

def T (x y : ℕ) : ℕ := 4 * x + 5 * y + x * y

theorem T_5_3 : T 5 3 = 50 :=
by
  sorry

end T_5_3_l240_240152


namespace radius_of_circumscribed_circle_l240_240342

noncomputable def circumscribed_circle_radius (r : ℝ) : Prop :=
  let s := r * Real.sqrt 2 in
  4 * s = π * r ^ 2 -> r = 4 * Real.sqrt 2 / π

-- Statement of the theorem to be proved
theorem radius_of_circumscribed_circle (r : ℝ) : circumscribed_circle_radius r := 
sorry

end radius_of_circumscribed_circle_l240_240342


namespace at_least_two_consecutive_heads_probability_l240_240124

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability_l240_240124


namespace root_expression_eq_l240_240461

theorem root_expression_eq (p q α β γ δ : ℝ) 
  (h1 : ∀ x, (x - α) * (x - β) = x^2 + p * x + 2)
  (h2 : ∀ x, (x - γ) * (x - δ) = x^2 + q * x + 2) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 4 + 2 * (p^2 - q^2) := 
sorry

end root_expression_eq_l240_240461


namespace total_dogs_equation_l240_240970

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l240_240970


namespace factorize_poly_l240_240520

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l240_240520


namespace more_people_needed_to_paint_fence_l240_240976

theorem more_people_needed_to_paint_fence :
  ∀ (n t m t' : ℕ), n = 8 → t = 3 → t' = 2 → (n * t = m * t') → m - n = 4 :=
by
  intros n t m t'
  intro h1
  intro h2
  intro h3
  intro h4
  sorry

end more_people_needed_to_paint_fence_l240_240976


namespace pieces_brought_to_school_on_friday_l240_240756

def pieces_of_fruit_mark_had := 10
def pieces_eaten_first_four_days := 5
def pieces_kept_for_next_week := 2

theorem pieces_brought_to_school_on_friday :
  pieces_of_fruit_mark_had - pieces_eaten_first_four_days - pieces_kept_for_next_week = 3 :=
by
  sorry

end pieces_brought_to_school_on_friday_l240_240756


namespace find_number_l240_240512

theorem find_number (x : ℝ) (h : x / 3 = x - 4) : x = 6 := 
by 
  sorry

end find_number_l240_240512


namespace regular_polygon_sides_l240_240912

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240912


namespace notebook_and_pen_prices_l240_240053

theorem notebook_and_pen_prices (x y : ℕ) (h1 : 2 * x + y = 30) (h2 : x = 2 * y) :
  x = 12 ∧ y = 6 :=
by
  sorry

end notebook_and_pen_prices_l240_240053


namespace rabbit_distribution_count_l240_240261

universe u

def rabbits : Finset String := {"Nina", "Tony", "Fluffy", "Snowy", "Brownie"}

structure StoreDistribution :=
  (store : Fin 4 → Finset String)
  (no_more_than_two : ∀ (i : Fin 4), (store i).card ≤ 2)
  (nina_and_tony_not_together : (store 0).disjoint (store 0))

theorem rabbit_distribution_count : ∃ (d : Finset StoreDistribution), d.card = 768 :=
by
  sorry

end rabbit_distribution_count_l240_240261


namespace num_divisors_8_fact_l240_240020

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l240_240020


namespace fernanda_savings_before_payments_l240_240140

open Real

theorem fernanda_savings_before_payments (aryan_debt kyro_debt aryan_payment kyro_payment total_savings before_savings : ℝ) 
  (h1: aryan_debt = 1200)
  (h2: aryan_debt = 2 * kyro_debt)
  (h3: aryan_payment = 0.6 * aryan_debt)
  (h4: kyro_payment = 0.8 * kyro_debt)
  (h5: total_savings = before_savings + aryan_payment + kyro_payment)
  (h6: total_savings = 1500) :
  before_savings = 300 :=
by
  sorry

end fernanda_savings_before_payments_l240_240140


namespace isosceles_triangle_count_l240_240010

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l240_240010


namespace circle_radius_eq_l240_240349

theorem circle_radius_eq {s r : ℝ} 
  (h1 : s * real.sqrt 2 = 2 * r)
  (h2 : 4 * s = real.pi * r^2) :
  r = 2 * real.sqrt 2 / real.pi :=
by
  sorry

end circle_radius_eq_l240_240349


namespace vector_parallel_dot_product_l240_240593

theorem vector_parallel_dot_product (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (x, 1))
  (h2 : b = (4, 2))
  (h3 : x / 4 = 1 / 2) : 
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) = 5 := 
by 
  sorry

end vector_parallel_dot_product_l240_240593


namespace value_of_y_l240_240042

theorem value_of_y (x y : ℝ) (h1 : x ^ (2 * y) = 81) (h2 : x = 9) : y = 1 :=
sorry

end value_of_y_l240_240042


namespace inequality_solution_l240_240796

theorem inequality_solution (x : ℝ) : x + 1 < (4 + 3 * x) / 2 → x > -2 :=
by
  intros h
  sorry

end inequality_solution_l240_240796


namespace bottle_caps_per_box_l240_240313

theorem bottle_caps_per_box (total_caps : ℕ) (total_boxes : ℕ) (h_total_caps : total_caps = 60) (h_total_boxes : total_boxes = 60) :
  (total_caps / total_boxes) = 1 :=
by {
  sorry
}

end bottle_caps_per_box_l240_240313


namespace sum_alternating_sequence_l240_240250

theorem sum_alternating_sequence : (Finset.range 2012).sum (λ k => (-1 : ℤ)^(k + 1)) = 0 :=
by
  sorry

end sum_alternating_sequence_l240_240250


namespace flat_fee_shipping_l240_240145

theorem flat_fee_shipping (w : ℝ) (c : ℝ) (C : ℝ) (F : ℝ) 
  (h_w : w = 5) 
  (h_c : c = 0.80) 
  (h_C : C = 9)
  (h_shipping : C = F + (c * w)) :
  F = 5 :=
by
  -- proof skipped
  sorry

end flat_fee_shipping_l240_240145


namespace regular_polygon_sides_l240_240929

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240929


namespace part_one_part_two_l240_240433

variable (α : Real) (h : Real.tan α = 2)

theorem part_one (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6 / 11 := 
by
  sorry

theorem part_two (h : Real.tan α = 2) : 
  (1 / 4 * Real.sin α ^ 2 + 1 / 3 * Real.sin α * Real.cos α + 1 / 2 * Real.cos α ^ 2 + 1) = 43 / 30 := 
by
  sorry

end part_one_part_two_l240_240433


namespace minimum_distance_from_circle_to_line_l240_240281

noncomputable def point_on_circle (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def line_eq (p : ℝ × ℝ) : ℝ :=
  p.1 - p.2 + 4

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt (1^2 + 1^2)

theorem minimum_distance_from_circle_to_line :
  ∀ θ : ℝ, (∃ θ, distance_from_point_to_line (point_on_circle θ) = 2 * Real.sqrt 2 - 2) :=
by
  sorry

end minimum_distance_from_circle_to_line_l240_240281


namespace sarah_gave_away_16_apples_to_teachers_l240_240316

def initial_apples : Nat := 25
def apples_given_to_friends : Nat := 5
def apples_eaten : Nat := 1
def apples_left_after_journey : Nat := 3

theorem sarah_gave_away_16_apples_to_teachers :
  let apples_after_giving_to_friends := initial_apples - apples_given_to_friends
  let apples_after_eating := apples_after_giving_to_friends - apples_eaten
  apples_after_eating - apples_left_after_journey = 16 :=
by
  sorry

end sarah_gave_away_16_apples_to_teachers_l240_240316


namespace rectangle_area_is_1600_l240_240338

theorem rectangle_area_is_1600 (l w : ℕ) 
  (h₁ : l = 4 * w)
  (h₂ : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_is_1600_l240_240338


namespace prob_divisors_8_fact_l240_240023

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l240_240023


namespace abs_diff_p_q_l240_240465

theorem abs_diff_p_q (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
by 
  sorry

end abs_diff_p_q_l240_240465


namespace range_of_a_l240_240992

-- Define the propositions
def p (x : ℝ) := (x - 1) * (x - 2) > 0
def q (a x : ℝ) := x^2 + (a - 1) * x - a > 0

-- Define the solution sets
def A := {x : ℝ | p x}
def B (a : ℝ) := {x : ℝ | q a x}

-- State the proof problem
theorem range_of_a (a : ℝ) : 
  (∀ x, p x → q a x) ∧ (∃ x, ¬p x ∧ q a x) → -2 < a ∧ a ≤ -1 :=
by
  sorry

end range_of_a_l240_240992


namespace three_digit_number_constraint_l240_240652

theorem three_digit_number_constraint (B : ℕ) (h1 : 30 ≤ B ∧ B < 40) (h2 : (330 + B) % 3 = 0) (h3 : (330 + B) % 7 = 0) : B = 6 :=
sorry

end three_digit_number_constraint_l240_240652


namespace polynomial_has_real_root_l240_240478

theorem polynomial_has_real_root (a b : ℝ) :
  ∃ x : ℝ, x^3 + a * x + b = 0 :=
sorry

end polynomial_has_real_root_l240_240478


namespace volleyball_team_total_score_l240_240734

-- Define the conditions
def LizzieScore := 4
def NathalieScore := LizzieScore + 3
def CombinedLizzieNathalieScore := LizzieScore + NathalieScore
def AimeeScore := 2 * CombinedLizzieNathalieScore
def TeammatesScore := 17

-- Prove that the total team score is 50
theorem volleyball_team_total_score :
  LizzieScore + NathalieScore + AimeeScore + TeammatesScore = 50 :=
by
  sorry

end volleyball_team_total_score_l240_240734


namespace smallest_multiple_9_11_13_l240_240425

theorem smallest_multiple_9_11_13 : ∃ n : ℕ, n > 0 ∧ (9 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1287 := 
 by {
   sorry
 }

end smallest_multiple_9_11_13_l240_240425


namespace pens_at_end_l240_240065

-- Define the main variable
variable (x : ℝ)

-- Define the conditions as functions
def initial_pens (x : ℝ) := x
def mike_gives (x : ℝ) := 0.5 * x
def after_mike (x : ℝ) := x + (mike_gives x)
def after_cindy (x : ℝ) := 2 * (after_mike x)
def give_sharon (x : ℝ) := 0.25 * (after_cindy x)

-- Define the final number of pens
def final_pens (x : ℝ) := (after_cindy x) - (give_sharon x)

-- The theorem statement
theorem pens_at_end (x : ℝ) : final_pens x = 2.25 * x :=
by sorry

end pens_at_end_l240_240065


namespace unique_real_solution_l240_240696

theorem unique_real_solution (x y : ℝ) (h1 : x^3 = 2 - y) (h2 : y^3 = 2 - x) : x = 1 ∧ y = 1 :=
sorry

end unique_real_solution_l240_240696


namespace move_digit_produces_ratio_l240_240985

theorem move_digit_produces_ratio
  (a b : ℕ)
  (h_original_eq : ∃ x : ℕ, x = 10 * a + b)
  (h_new_eq : ∀ (n : ℕ), 10^n * b + a = (3 * (10 * a + b)) / 2):
  285714 = 10 * a + b :=
by
  -- proof steps would go here
  sorry

end move_digit_produces_ratio_l240_240985


namespace radius_of_circumscribed_circle_l240_240364

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l240_240364


namespace value_of_fg3_l240_240290

namespace ProofProblem

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2

theorem value_of_fg3 : f (g 3) = 83 := 
by 
  sorry -- Proof not needed

end ProofProblem

end value_of_fg3_l240_240290


namespace problem1_general_formula_problem2_range_of_a_problem3_sum_of_first_n_terms_l240_240586

noncomputable def f (x : ℝ) : ℝ :=
  2 * |x + 2| - |x + 1|

noncomputable def a_seq (n : ℕ) : ℝ := by
  apply ite (n = 0) 0
  apply f (n - 1)

theorem problem1_general_formula (n : ℕ) (h : n ≠ 0) :
  a_seq n = n + 3 := by
  sorry

theorem problem2_range_of_a (a : ℝ) :
  (∀n > 1, a_seq (n + 1) = f (a_seq n) →
      (∃ d, a_seq (n + 1) = a_seq n + d)) →
  a ≥ -1 ∨ a = -3 := by
  sorry

theorem problem3_sum_of_first_n_terms (a : ℝ) (n : ℕ) :
  S_n a n = ∑ i in range n, a_seq i =
    if a ≥ -1 then
      (3/2 : ℝ) * (n ^ 2) + (a - 3 / 2) * n
    else if -2 < a ∧ a ≤ -1 then
      (3/2 : ℝ) * (n ^ 2) + (1/2 + 3 * a) * n - 2 * a - 2
    else
      (3/2 : ℝ) * (n ^ 2) - (a + 15 / 2) * n + 2 * a + 6 := by
  sorry

end problem1_general_formula_problem2_range_of_a_problem3_sum_of_first_n_terms_l240_240586


namespace sandy_books_second_shop_l240_240637

theorem sandy_books_second_shop (x : ℕ) (h1 : 65 = 1080 / 16) 
                                (h2 : x * 16 = 840) 
                                (h3 : (1080 + 840) / 16 = 120) : 
                                x = 55 :=
by
  sorry

end sandy_books_second_shop_l240_240637


namespace factorize_poly_l240_240518

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l240_240518


namespace range_of_a_l240_240730

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ 1 :=
sorry

end range_of_a_l240_240730


namespace cos_pi_over_2_minus_l240_240273

theorem cos_pi_over_2_minus (A : ℝ) (h : Real.sin A = 1 / 2) : Real.cos (3 * Real.pi / 2 - A) = -1 / 2 :=
  sorry

end cos_pi_over_2_minus_l240_240273


namespace find_c_and_d_l240_240601

theorem find_c_and_d :
  ∀ (y c d : ℝ), (y^2 - 5 * y + 5 / y + 1 / (y^2) = 17) ∧ (y = c - Real.sqrt d) ∧ (0 < c) ∧ (0 < d) → (c + d = 106) :=
by
  intros y c d h
  sorry

end find_c_and_d_l240_240601


namespace total_expenditure_now_l240_240048

-- Define the conditions in Lean
def original_student_count : ℕ := 100
def additional_students : ℕ := 25
def decrease_in_average_expenditure : ℤ := 10
def increase_in_total_expenditure : ℤ := 500

-- Let's denote the original average expenditure per student as A rupees
variable (A : ℤ)

-- Define the old and new expenditures
def original_total_expenditure := original_student_count * A
def new_average_expenditure := A - decrease_in_average_expenditure
def new_total_expenditure := (original_student_count + additional_students) * new_average_expenditure

-- The theorem to prove
theorem total_expenditure_now :
  new_total_expenditure A - original_total_expenditure A = increase_in_total_expenditure →
  new_total_expenditure A = 7500 :=
by
  sorry

end total_expenditure_now_l240_240048


namespace quadratic_roots_relation_l240_240702

noncomputable def roots_relation (a b c : ℝ) : Prop :=
  ∃ α β : ℝ, (α * β = c / a) ∧ (α + β = -b / a) ∧ β = 3 * α

theorem quadratic_roots_relation (a b c : ℝ) (h : roots_relation a b c) : 3 * b^2 = 16 * a * c :=
by
  sorry

end quadratic_roots_relation_l240_240702


namespace smallest_n_contains_constant_term_l240_240426

theorem smallest_n_contains_constant_term :
  ∃ n : ℕ, (∀ x : ℝ, x ≠ 0 → (2 * x^3 + 1 / x^(1/2))^n = c ↔ n = 7) :=
by
  sorry

end smallest_n_contains_constant_term_l240_240426


namespace num_divisors_fact8_l240_240034

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l240_240034


namespace regular_polygon_sides_l240_240862

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240862


namespace set_representation_equiv_l240_240550

open Nat

theorem set_representation_equiv :
  {x : ℕ | (0 < x) ∧ (x - 3 < 2)} = {1, 2, 3, 4} :=
by
  sorry

end set_representation_equiv_l240_240550


namespace regular_polygon_sides_l240_240853

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240853


namespace circle_radius_l240_240360

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l240_240360


namespace percent_of_g_is_h_l240_240752

variable (a b c d e f g h : ℝ)

-- Conditions
def cond1a : f = 0.60 * a := sorry
def cond1b : f = 0.45 * b := sorry
def cond2a : g = 0.70 * b := sorry
def cond2b : g = 0.30 * c := sorry
def cond3a : h = 0.80 * c := sorry
def cond3b : h = 0.10 * f := sorry
def cond4a : c = 0.30 * a := sorry
def cond4b : c = 0.25 * b := sorry
def cond5a : d = 0.40 * a := sorry
def cond5b : d = 0.35 * b := sorry
def cond6a : e = 0.50 * b := sorry
def cond6b : e = 0.20 * c := sorry

-- Theorem to prove
theorem percent_of_g_is_h (h_percent_g : ℝ) 
  (h_formula : h = h_percent_g * g) : 
  h = 0.285714 * g :=
by
  sorry

end percent_of_g_is_h_l240_240752


namespace calc_30_exp_l240_240144

theorem calc_30_exp :
  30 * 30 ^ 10 = 30 ^ 11 :=
by sorry

end calc_30_exp_l240_240144


namespace charles_total_earnings_l240_240689

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l240_240689


namespace sufficient_condition_l240_240270

-- Definitions of propositions p and q
variables (p q : Prop)

-- Theorem statement
theorem sufficient_condition (h : ¬(p ∨ q)) : ¬p :=
by sorry

end sufficient_condition_l240_240270


namespace simplify_expression_l240_240072

theorem simplify_expression (x y : ℝ) : 3 * y - 5 * x + 2 * y + 4 * x = 5 * y - x :=
by
  sorry

end simplify_expression_l240_240072


namespace division_of_powers_l240_240151

theorem division_of_powers :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 :=
by sorry

end division_of_powers_l240_240151


namespace infinite_solutions_imply_values_l240_240603

theorem infinite_solutions_imply_values (a b : ℝ) :
  (∀ x : ℝ, a * (2 * x + b) = 12 * x + 5) ↔ (a = 6 ∧ b = 5 / 6) :=
by
  sorry

end infinite_solutions_imply_values_l240_240603


namespace perp_vector_k_l240_240269

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_vector_k :
  ∀ k : ℝ, dot_product (1, 2) (-2, k) = 0 → k = 1 :=
by
  intro k h₀
  sorry

end perp_vector_k_l240_240269


namespace regular_polygon_sides_l240_240852

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240852


namespace regular_polygon_sides_l240_240848

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240848


namespace positive_divisors_8_factorial_l240_240025

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l240_240025


namespace problem1_problem2_l240_240989

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}

-- Define the conditions and questions as Lean statements

-- First problem: Prove that if A ∩ B = ∅ and A ∪ B = ℝ, then a = 2
theorem problem1 (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := 
  sorry

-- Second problem: Prove that if A a ⊆ B, then a ∈ (-∞, 0] ∪ [4, ∞)
theorem problem2 (a : ℝ) (h1 : A a ⊆ B) : a ≤ 0 ∨ a ≥ 4 := 
  sorry

end problem1_problem2_l240_240989


namespace find_m_l240_240282

theorem find_m (m : ℝ) :
  (∀ x y, x + (m^2 - m) * y = 4 * m - 1 → ∀ x y, 2 * x - y - 5 = 0 → (-1 / (m^2 - m)) = -1 / 2) → 
  (m = -1 ∨ m = 2) :=
sorry

end find_m_l240_240282


namespace bookshop_shipment_correct_l240_240128

noncomputable def bookshop_shipment : ℕ :=
  let Initial_books := 743
  let Saturday_instore_sales := 37
  let Saturday_online_sales := 128
  let Sunday_instore_sales := 2 * Saturday_instore_sales
  let Sunday_online_sales := Saturday_online_sales + 34
  let books_sold := Saturday_instore_sales + Saturday_online_sales + Sunday_instore_sales + Sunday_online_sales
  let Final_books := 502
  Final_books - (Initial_books - books_sold)

theorem bookshop_shipment_correct : bookshop_shipment = 160 := by
  sorry

end bookshop_shipment_correct_l240_240128


namespace regular_polygon_sides_l240_240939

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240939


namespace sin_pi_minus_alpha_l240_240274

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin α = 1 / 2) : Real.sin (π - α) = 1 / 2 :=
by
  sorry

end sin_pi_minus_alpha_l240_240274


namespace inverse_proportional_ratios_l240_240485

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l240_240485


namespace smallest_number_conditions_l240_240808

theorem smallest_number_conditions :
  ∃ b : ℕ, 
    (b % 3 = 2) ∧ 
    (b % 4 = 2) ∧
    (b % 5 = 3) ∧
    (∀ b' : ℕ, 
      (b' % 3 = 2) ∧ 
      (b' % 4 = 2) ∧
      (b' % 5 = 3) → b ≤ b') :=
begin
  use 38,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros b' hb',
    have h3 := (hb'.left),
    have h4 := (hb'.right.left),
    have h5 := (hb'.right.right),
    -- The raw machinery for showing that 38 is the smallest may require more definition
    sorry
  }
end

end smallest_number_conditions_l240_240808


namespace regular_polygon_sides_l240_240960

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240960


namespace regular_polygon_sides_l240_240861

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240861


namespace jamie_workday_percent_l240_240415

theorem jamie_workday_percent
  (total_work_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_multiplier : ℕ)
  (break_minutes : ℕ)
  (total_minutes_per_hour : ℕ)
  (total_work_minutes : ℕ)
  (first_meeting_duration : ℕ)
  (second_meeting_duration : ℕ)
  (total_meeting_time : ℕ)
  (percentage_spent : ℚ) :
  total_work_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 2 →
  break_minutes = 30 →
  total_minutes_per_hour = 60 →
  total_work_minutes = total_work_hours * total_minutes_per_hour →
  first_meeting_duration = first_meeting_minutes →
  second_meeting_duration = second_meeting_multiplier * first_meeting_duration →
  total_meeting_time = first_meeting_duration + second_meeting_duration + break_minutes →
  percentage_spent = (total_meeting_time : ℚ) / (total_work_minutes : ℚ) * 100 →
  percentage_spent = 35 :=
sorry

end jamie_workday_percent_l240_240415


namespace radius_of_circle_l240_240453

-- Define the problem condition
def diameter_of_circle : ℕ := 14

-- State the problem as a theorem
theorem radius_of_circle (d : ℕ) (hd : d = diameter_of_circle) : d / 2 = 7 := by 
  sorry

end radius_of_circle_l240_240453


namespace find_length_AB_l240_240260

open Real

noncomputable def AB_length := 
  let r := 4
  let V_total := 320 * π
  ∃ (L : ℝ), 16 * π * L + (256 / 3) * π = V_total ∧ L = 44 / 3

theorem find_length_AB :
  AB_length := by
  sorry

end find_length_AB_l240_240260


namespace find_number_l240_240326

def digits_form_geometric_progression (x y z : ℕ) : Prop :=
  x * z = y * y

def swapped_hundreds_units (x y z : ℕ) : Prop :=
  100 * z + 10 * y + x = 100 * x + 10 * y + z - 594

def reversed_post_removal (x y z : ℕ) : Prop :=
  10 * z + y = 10 * y + z - 18

theorem find_number (x y z : ℕ) (h1 : digits_form_geometric_progression x y z) 
  (h2 : swapped_hundreds_units x y z) 
  (h3 : reversed_post_removal x y z) :
  100 * x + 10 * y + z = 842 := by
  sorry

end find_number_l240_240326


namespace net_moles_nh3_after_reactions_l240_240691

/-- Define the stoichiometry of the reactions and available amounts of reactants -/
def step1_reaction (nh4cl na2co3 : ℕ) : ℕ :=
  if nh4cl / 2 >= na2co3 then 
    2 * na2co3
  else 
    2 * (nh4cl / 2)

def step2_reaction (koh h3po4 : ℕ) : ℕ :=
  0  -- No NH3 produced in this step

theorem net_moles_nh3_after_reactions :
  let nh4cl := 3
  let na2co3 := 1
  let koh := 3
  let h3po4 := 1
  let nh3_after_step1 := step1_reaction nh4cl na2co3
  let nh3_after_step2 := step2_reaction koh h3po4
  nh3_after_step1 + nh3_after_step2 = 2 :=
by
  sorry

end net_moles_nh3_after_reactions_l240_240691


namespace campaign_fliers_l240_240516

theorem campaign_fliers (total_fliers : ℕ) (fraction_morning : ℚ) (fraction_afternoon : ℚ) 
  (remaining_fliers_after_morning : ℕ) (remaining_fliers_after_afternoon : ℕ) :
  total_fliers = 1000 → fraction_morning = 1/5 → fraction_afternoon = 1/4 → 
  remaining_fliers_after_morning = total_fliers - total_fliers * fraction_morning → 
  remaining_fliers_after_afternoon = remaining_fliers_after_morning - remaining_fliers_after_morning * fraction_afternoon → 
  remaining_fliers_after_afternoon = 600 := 
by
  sorry

end campaign_fliers_l240_240516


namespace geom_sequence_sum_l240_240608

theorem geom_sequence_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (r : ℤ) 
    (h1 : ∀ n : ℕ, n ≥ 1 → S n = 3^n + r) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n = S n - S (n - 1)) 
    (h3 : a 1 = S 1) :
  r = -1 := 
sorry

end geom_sequence_sum_l240_240608


namespace product_is_two_l240_240408

theorem product_is_two : 
  ((10 : ℚ) * (1/5) * 4 * (1/16) * (1/2) * 8 = 2) :=
sorry

end product_is_two_l240_240408


namespace am_gm_inequality_l240_240477

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^a * b^b * c^c ≥ (a * b * c)^((a + b + c) / 3) :=
sorry

end am_gm_inequality_l240_240477


namespace probability_triangle_nonagon_l240_240569

-- Define the total number of ways to choose 3 vertices from 9 vertices
def total_ways_to_choose_triangle : ℕ := Nat.choose 9 3

-- Define the number of favorable outcomes
def favorable_outcomes_one_side : ℕ := 9 * 5
def favorable_outcomes_two_sides : ℕ := 9

def total_favorable_outcomes : ℕ := favorable_outcomes_one_side + favorable_outcomes_two_sides

-- Define the probability as a rational number
def probability_at_least_one_side_nonagon (total: ℕ) (favorable: ℕ) : ℚ :=
  favorable / total
  
-- Theorem stating the probability
theorem probability_triangle_nonagon :
  probability_at_least_one_side_nonagon total_ways_to_choose_triangle total_favorable_outcomes = 9 / 14 :=
by
  sorry

end probability_triangle_nonagon_l240_240569


namespace median_first_fifteen_positive_integers_l240_240664

-- Define the list of the first fifteen positive integers
def first_fifteen_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

-- Define the property that the median of the list is 8.0
theorem median_first_fifteen_positive_integers : median(first_fifteen_positive_integers) = 8.0 := 
sorry

end median_first_fifteen_positive_integers_l240_240664


namespace least_number_to_add_l240_240813

theorem least_number_to_add (n : ℕ) (sum_digits : ℕ) (next_multiple : ℕ) 
  (h1 : n = 51234) 
  (h2 : sum_digits = 5 + 1 + 2 + 3 + 4) 
  (h3 : next_multiple = 18) :
  ∃ k, (k = next_multiple - sum_digits) ∧ (n + k) % 9 = 0 :=
sorry

end least_number_to_add_l240_240813


namespace circumscribed_circle_radius_l240_240455

-- Definitions of side lengths
def a : ℕ := 5
def b : ℕ := 12

-- Defining the hypotenuse based on the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt (a * a + b * b)

-- Radius of the circumscribed circle of a right triangle
def radius (hypotenuse : ℕ) : ℕ := hypotenuse / 2

-- Theorem: The radius of the circumscribed circle of the right triangle is 13 / 2 = 6.5
theorem circumscribed_circle_radius : 
  radius (hypotenuse a b) = 13 / 2 :=
by
  sorry

end circumscribed_circle_radius_l240_240455


namespace smallest_number_conditions_l240_240807

theorem smallest_number_conditions :
  ∃ b : ℕ, 
    (b % 3 = 2) ∧ 
    (b % 4 = 2) ∧
    (b % 5 = 3) ∧
    (∀ b' : ℕ, 
      (b' % 3 = 2) ∧ 
      (b' % 4 = 2) ∧
      (b' % 5 = 3) → b ≤ b') :=
begin
  use 38,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num },
  { intros b' hb',
    have h3 := (hb'.left),
    have h4 := (hb'.right.left),
    have h5 := (hb'.right.right),
    -- The raw machinery for showing that 38 is the smallest may require more definition
    sorry
  }
end

end smallest_number_conditions_l240_240807


namespace greatest_whole_number_satisfies_inequality_l240_240423

theorem greatest_whole_number_satisfies_inequality : 
  ∃ (x : ℕ), (∀ (y : ℕ), (6 * y - 4 < 5 - 3 * y) → y ≤ x) ∧ x = 0 := 
sorry

end greatest_whole_number_satisfies_inequality_l240_240423


namespace compare_trig_values_l240_240164

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values_l240_240164


namespace mia_socks_l240_240631

-- Defining the number of each type of socks
variables {a b c : ℕ}

-- Conditions and constraints
def total_pairs (a b c : ℕ) : Prop := a + b + c = 15
def total_cost (a b c : ℕ) : Prop := 2 * a + 3 * b + 5 * c = 35
def at_least_one (a b c : ℕ) : Prop := a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1

-- Main theorem to prove the number of 2-dollar pairs of socks
theorem mia_socks : 
  ∀ (a b c : ℕ), 
  total_pairs a b c → 
  total_cost a b c → 
  at_least_one a b c → 
  a = 12 :=
by
  sorry

end mia_socks_l240_240631


namespace regular_polygon_sides_l240_240933

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240933


namespace phoenix_hike_distance_l240_240473

variable (a b c d : ℕ)

theorem phoenix_hike_distance
  (h1 : a + b = 24)
  (h2 : b + c = 30)
  (h3 : c + d = 32)
  (h4 : a + c = 28) :
  a + b + c + d = 56 :=
by
  sorry

end phoenix_hike_distance_l240_240473


namespace regular_polygon_sides_l240_240956

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240956


namespace number_of_children_l240_240655
-- Import the entirety of the Mathlib library

-- Define the conditions and the theorem to be proven
theorem number_of_children (C n : ℕ) 
  (h1 : C = 8 * n + 4) 
  (h2 : C = 11 * (n - 1)) : 
  n = 5 :=
by sorry

end number_of_children_l240_240655


namespace percentage_loss_l240_240825

theorem percentage_loss (SP_loss SP_profit CP : ℝ) 
  (h₁ : SP_loss = 9) 
  (h₂ : SP_profit = 11.8125) 
  (h₃ : SP_profit = CP * 1.05) : 
  (CP - SP_loss) / CP * 100 = 20 :=
by sorry

end percentage_loss_l240_240825


namespace gcd_of_polynomials_l240_240578

theorem gcd_of_polynomials (b : ℤ) (h : 2460 ∣ b) : 
  Int.gcd (b^2 + 6 * b + 30) (b + 5) = 30 :=
sorry

end gcd_of_polynomials_l240_240578


namespace smallest_number_l240_240810

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l240_240810


namespace ratio_costs_equal_l240_240302

noncomputable def cost_first_8_years : ℝ := 10000 * 8
noncomputable def john_share_first_8_years : ℝ := cost_first_8_years / 2
noncomputable def university_tuition : ℝ := 250000
noncomputable def john_share_university : ℝ := university_tuition / 2
noncomputable def total_paid_by_john : ℝ := 265000
noncomputable def cost_between_8_and_18 : ℝ := total_paid_by_john - john_share_first_8_years - john_share_university
noncomputable def cost_per_year_8_to_18 : ℝ := cost_between_8_and_18 / 10
noncomputable def cost_per_year_first_8_years : ℝ := 10000

theorem ratio_costs_equal : cost_per_year_8_to_18 / cost_per_year_first_8_years = 1 := by
  sorry

end ratio_costs_equal_l240_240302


namespace find_f_1_minus_a_l240_240988

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

theorem find_f_1_minus_a 
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_period : periodic_function f 2)
  (h_value : ∃ a : ℝ, f (1 + a) = 1) :
  ∃ a : ℝ, f (1 - a) = -1 :=
by
  sorry

end find_f_1_minus_a_l240_240988


namespace equation_roots_l240_240795

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l240_240795


namespace problem1_problem2a_problem2b_problem2c_l240_240280

theorem problem1 {x : ℝ} : 3 * x ^ 2 - 5 * x - 2 < 0 → -1 / 3 < x ∧ x < 2 :=
sorry

theorem problem2a {x a : ℝ} (ha : -1 / 2 < a ∧ a < 0) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x < 2 ∨ x > -1 / a :=
sorry

theorem problem2b {x a : ℝ} (ha : a = -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x ≠ 2 :=
sorry

theorem problem2c {x a : ℝ} (ha : a < -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x > 2 ∨ x < -1 / a :=
sorry

end problem1_problem2a_problem2b_problem2c_l240_240280


namespace range_of_m_for_decreasing_interval_l240_240001

def function_monotonically_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def f (x : ℝ) : ℝ := x ^ 3 - 12 * x

theorem range_of_m_for_decreasing_interval :
  ∀ m : ℝ, function_monotonically_decreasing_in_interval f (2 * m) (m + 1) → -1 ≤ m ∧ m < 1 :=
by
  sorry

end range_of_m_for_decreasing_interval_l240_240001


namespace Brandy_can_safely_drink_20_mg_more_l240_240491

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end Brandy_can_safely_drink_20_mg_more_l240_240491


namespace value_diff_l240_240090

theorem value_diff (a b : ℕ) (h1 : a * b = 2 * (a + b) + 14) (h2 : b = 8) : b - a = 3 :=
by
  sorry

end value_diff_l240_240090


namespace multiplication_in_P_l240_240630

-- Define the set P as described in the problem
def P := {x : ℕ | ∃ n : ℕ, x = n^2}

-- Prove that for all a, b in P, a * b is also in P
theorem multiplication_in_P {a b : ℕ} (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P :=
sorry

end multiplication_in_P_l240_240630


namespace equation_of_line_l240_240563

theorem equation_of_line (a b : ℝ) (h1 : a = -2) (h2 : b = 2) :
  (∀ x y : ℝ, (x / a + y / b = 1) → x - y + 2 = 0) :=
by
  sorry

end equation_of_line_l240_240563


namespace backpack_pencil_case_combinations_l240_240743

theorem backpack_pencil_case_combinations (backpacks pencil_cases : Fin 2) : 
  (backpacks * pencil_cases) = 4 :=
by 
  sorry

end backpack_pencil_case_combinations_l240_240743


namespace monotonicity_and_range_of_k_l240_240720

noncomputable def f (x k : ℝ) := 2 * real.exp x - k * x - 2

theorem monotonicity_and_range_of_k (k : ℝ) :
  (∀ x > 0, (2 * real.exp x - k) > 0 ∨ (2 * real.exp x - k) < 0) ∧
  (∃ m > 0, ∀ x ∈ set.Ioo 0 m, |f x k| > 2 * x) → k > 4 :=
by
  sorry

end monotonicity_and_range_of_k_l240_240720


namespace power_function_value_l240_240644

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end power_function_value_l240_240644


namespace total_coins_l240_240972

-- Defining the conditions
def stack1 : Nat := 4
def stack2 : Nat := 8

-- Statement of the proof problem
theorem total_coins : stack1 + stack2 = 12 :=
by
  sorry

end total_coins_l240_240972


namespace positive_divisors_of_8_factorial_l240_240022

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l240_240022


namespace base_conversion_difference_l240_240552

-- Definitions
def base9_to_base10 (n : ℕ) : ℕ := 3 * (9^2) + 2 * (9^1) + 7 * (9^0)
def base8_to_base10 (m : ℕ) : ℕ := 2 * (8^2) + 5 * (8^1) + 3 * (8^0)

-- Statement
theorem base_conversion_difference :
  base9_to_base10 327 - base8_to_base10 253 = 97 :=
by sorry

end base_conversion_difference_l240_240552


namespace equation_of_tangent_line_l240_240718

noncomputable def f (m x : ℝ) := m * Real.exp x - x - 1

def passes_through_P (m : ℝ) : Prop :=
  f m 0 = 1

theorem equation_of_tangent_line (m : ℝ) (h : passes_through_P m) :
  (f m) 0 = 1 → (2 - 1 = 1) ∧ ((y - 1 = x) → (x - y + 1 = 0)) :=
by
  intro h
  sorry

end equation_of_tangent_line_l240_240718


namespace usual_time_cover_journey_l240_240221

theorem usual_time_cover_journey (S T : ℝ) (H : S / T = (5/6 * S) / (T + 8)) : T = 48 :=
by
  sorry

end usual_time_cover_journey_l240_240221


namespace median_of_first_fifteen_integers_l240_240661

theorem median_of_first_fifteen_integers : 
  let L := (list.range 15).map (λ n, n + 1)
  in list.median L = 8.0 :=
by 
  sorry

end median_of_first_fifteen_integers_l240_240661


namespace solve_for_y_l240_240592

theorem solve_for_y (x y : ℝ) (hx : x > 1) (hy : y > 1) (h1 : 1 / x + 1 / y = 1) (h2 : x * y = 9) :
  y = (9 + 3 * Real.sqrt 5) / 2 :=
by
  sorry

end solve_for_y_l240_240592


namespace remainder_n_squared_l240_240108

theorem remainder_n_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := 
    sorry

end remainder_n_squared_l240_240108


namespace tomato_count_after_harvest_l240_240746

theorem tomato_count_after_harvest :
  let plant_A_initial := 150
  let plant_B_initial := 200
  let plant_C_initial := 250
  -- Day 1
  let plant_A_after_day1 := plant_A_initial - (plant_A_initial * 3 / 10)
  let plant_B_after_day1 := plant_B_initial - (plant_B_initial * 1 / 4)
  let plant_C_after_day1 := plant_C_initial - (plant_C_initial * 4 / 25)
  -- Day 7
  let plant_A_after_day7 := plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)
  let plant_B_after_day7 := plant_B_after_day1 - (plant_B_after_day1 * 1 / 5)
  let plant_C_after_day7 := plant_C_after_day1 - ((plant_C_initial * 4 / 25) * 2)
  -- Day 14
  let plant_A_after_day14 := plant_A_after_day7 - ((plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)) * 3)
  let plant_B_after_day14 := plant_B_after_day7 - ((plant_B_after_day1 * 1 / 5) + 15)
  let plant_C_after_day14 := plant_C_after_day7 - (plant_C_after_day7 * 1 / 5)
  (plant_A_after_day14 = 0) ∧ (plant_B_after_day14 = 75) ∧ (plant_C_after_day14 = 104) :=
by
  sorry

end tomato_count_after_harvest_l240_240746


namespace solution_set_l240_240980

theorem solution_set (x : ℝ) : (x + 1 = |x + 3| - |x - 1|) ↔ (x = 3 ∨ x = -1 ∨ x = -5) :=
by
  sorry

end solution_set_l240_240980


namespace equation_roots_l240_240792

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l240_240792


namespace point_in_third_quadrant_l240_240165

noncomputable def is_second_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b > 0

noncomputable def is_third_quadrant (a b : ℝ) : Prop :=
a < 0 ∧ b < 0

theorem point_in_third_quadrant (a b : ℝ) (h : is_second_quadrant a b) : is_third_quadrant a (-b) :=
by
  sorry

end point_in_third_quadrant_l240_240165


namespace length_of_side_of_largest_square_l240_240818

-- Definitions based on the conditions
def string_length : ℕ := 24

-- The main theorem corresponding to the problem statement.
theorem length_of_side_of_largest_square (h: string_length = 24) : 24 / 4 = 6 :=
by
  sorry

end length_of_side_of_largest_square_l240_240818


namespace regular_polygon_sides_l240_240875

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240875


namespace max_value_condition_l240_240708

variable {m n : ℝ}

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  m * n > 0 ∧ m + n = -1

-- Statement of the proof problem
theorem max_value_condition (h : conditions m n) : (1/m + 1/n) ≤ 4 :=
sorry

end max_value_condition_l240_240708


namespace roots_of_quadratic_eq_l240_240781

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l240_240781


namespace rectangle_area_1600_l240_240336

theorem rectangle_area_1600
  (l w : ℝ)
  (h1 : l = 4 * w)
  (h2 : 2 * l + 2 * w = 200) :
  l * w = 1600 :=
by
  sorry

end rectangle_area_1600_l240_240336


namespace roots_of_polynomial_l240_240153

theorem roots_of_polynomial :
  {x : ℝ | x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256 = 0} = {x | x = 1 ∨ x = -1 ∨ x = 2 ∨ x = -2} :=
by
  sorry

end roots_of_polynomial_l240_240153


namespace monomial_2023rd_l240_240471

theorem monomial_2023rd : ∀ (x : ℝ), (2 * 2023 + 1) / 2023 * x ^ 2023 = (4047 / 2023) * x ^ 2023 :=
by
  intro x
  sorry

end monomial_2023rd_l240_240471


namespace roots_of_quadratic_eq_l240_240784

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l240_240784


namespace salary_increase_correct_l240_240771

noncomputable def old_average_salary : ℕ := 1500
noncomputable def number_of_employees : ℕ := 24
noncomputable def manager_salary : ℕ := 11500
noncomputable def new_total_salary := (number_of_employees * old_average_salary) + manager_salary
noncomputable def new_number_of_people := number_of_employees + 1
noncomputable def new_average_salary := new_total_salary / new_number_of_people
noncomputable def salary_increase := new_average_salary - old_average_salary

theorem salary_increase_correct : salary_increase = 400 := by
sorry

end salary_increase_correct_l240_240771


namespace regular_polygon_sides_l240_240945

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240945


namespace number_of_figures_l240_240396

theorem number_of_figures (num_squares num_rectangles : ℕ) 
  (h1 : 8 * 8 / 4 = num_squares + num_rectangles) 
  (h2 : 2 * 54 + 4 * 8 = 8 * num_squares + 10 * num_rectangles) :
  num_squares = 10 ∧ num_rectangles = 6 :=
sorry

end number_of_figures_l240_240396


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l240_240615

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l240_240615


namespace charles_earnings_l240_240687

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l240_240687


namespace ball_hits_ground_time_l240_240641

noncomputable def find_time_when_ball_hits_ground (a b c : ℝ) : ℝ :=
  (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)

theorem ball_hits_ground_time :
  find_time_when_ball_hits_ground (-16) 40 50 = (5 + 5 * Real.sqrt 3) / 4 :=
by
  sorry

end ball_hits_ground_time_l240_240641


namespace copper_zinc_mixture_mass_bounds_l240_240371

theorem copper_zinc_mixture_mass_bounds :
  ∀ (x y : ℝ) (D1 D2 : ℝ),
    (400 = x + y) →
    (50 = x / D1 + y / D2) →
    (8.8 ≤ D1 ∧ D1 ≤ 9) →
    (7.1 ≤ D2 ∧ D2 ≤ 7.2) →
    (200 ≤ x ∧ x ≤ 233) ∧ (167 ≤ y ∧ y ≤ 200) :=
sorry

end copper_zinc_mixture_mass_bounds_l240_240371


namespace similar_triangles_x_value_l240_240242

theorem similar_triangles_x_value : ∃ (x : ℝ), (12 / x = 9 / 6) ∧ x = 8 := by
  use 8
  constructor
  · sorry
  · rfl

end similar_triangles_x_value_l240_240242


namespace problem1_problem2_l240_240409

-- Problem 1
theorem problem1 : 2 * Real.cos (30 * Real.pi / 180) - Real.tan (60 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) = 1 / 2 :=
by sorry

-- Problem 2
theorem problem2 : (-1) ^ 2023 + 2 * Real.sin (45 * Real.pi / 180) - Real.cos (30 * Real.pi / 180) + Real.sin (60 * Real.pi / 180) + (Real.tan (60 * Real.pi / 180)) ^ 2 = 2 + Real.sqrt 2 :=
by sorry

end problem1_problem2_l240_240409


namespace find_m_l240_240271

noncomputable def quadratic_eq (x : ℝ) (m : ℝ) : ℝ := 2 * x^2 + 4 * x + m

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : quadratic_eq x₁ m = 0)
  (h2 : quadratic_eq x₂ m = 0)
  (h3 : 16 - 8 * m ≥ 0)
  (h4 : x₁^2 + x₂^2 + 2 * x₁ * x₂ - x₁^2 * x₂^2 = 0) 
  : m = -4 :=
sorry

end find_m_l240_240271


namespace fruit_difference_l240_240758

noncomputable def apples : ℕ := 60
noncomputable def peaches : ℕ := 3 * apples

theorem fruit_difference : peaches - apples = 120 :=
by
  have h1 : apples = 60 := rfl
  have h2 : peaches = 3 * apples := rfl
  calc
    peaches - apples = 3 * apples - apples : by rw [h2]
                ... = 3 * 60 - 60        : by rw [h1]
                ... = 180 - 60           : by norm_num
                ... = 120                : by norm_num

end fruit_difference_l240_240758


namespace counterexample_to_proposition_l240_240109

theorem counterexample_to_proposition (x y : ℤ) (h1 : x = -1) (h2 : y = -2) : x > y ∧ ¬ (x^2 > y^2) := by
  sorry

end counterexample_to_proposition_l240_240109


namespace regular_polygon_sides_l240_240831

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240831


namespace number_of_quartets_l240_240297

theorem number_of_quartets :
  let n := 5
  let factorial (x : Nat) := Nat.factorial x
  factorial n ^ 3 = 120 ^ 3 :=
by
  sorry

end number_of_quartets_l240_240297


namespace estate_value_l240_240470

theorem estate_value (E : ℝ) (x : ℝ) (y: ℝ) (z: ℝ) 
  (h1 : 9 * x = 3 / 4 * E) 
  (h2 : z = 8 * x) 
  (h3 : y = 600) 
  (h4 : E = z + 9 * x + y):
  E = 1440 := 
sorry

end estate_value_l240_240470


namespace regular_polygon_sides_l240_240849

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240849


namespace regular_polygon_sides_l240_240867

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240867


namespace regular_polygon_sides_l240_240899

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240899


namespace min_value_inequality_l240_240167

noncomputable def minValue : ℝ := 17 / 2

theorem min_value_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_cond : a + 2 * b = 1) :
  a^2 + 4 * b^2 + 1 / (a * b) = minValue := 
by
  sorry

end min_value_inequality_l240_240167


namespace ratio_of_speeds_l240_240657

theorem ratio_of_speeds (v1 v2 : ℝ) (h1 : v1 > v2) (h2 : 8 = (v1 + v2) * 2) (h3 : 8 = (v1 - v2) * 4) : v1 / v2 = 3 :=
by
  sorry

end ratio_of_speeds_l240_240657


namespace dodecahedron_equilateral_triangles_l240_240726

-- Definitions reflecting the conditions
def vertices_of_dodecahedron := 20
def faces_of_dodecahedron := 12
def vertices_per_face := 5
def equilateral_triangles_per_face := 5

theorem dodecahedron_equilateral_triangles :
  (faces_of_dodecahedron * equilateral_triangles_per_face) = 60 := by
  sorry

end dodecahedron_equilateral_triangles_l240_240726


namespace yara_total_earnings_l240_240693

-- Lean code to represent the conditions and the proof statement

theorem yara_total_earnings
  (x : ℕ)  -- Yara's hourly wage
  (third_week_hours : ℕ := 18)
  (previous_week_hours : ℕ := 12)
  (extra_earnings : ℕ := 36)
  (third_week_earning : ℕ := third_week_hours * x)
  (previous_week_earning : ℕ := previous_week_hours * x)
  (total_earning : ℕ := third_week_earning + previous_week_earning) :
  third_week_earning = previous_week_earning + extra_earnings → 
  total_earning = 180 := 
by
  -- Proof here
  sorry

end yara_total_earnings_l240_240693


namespace first_mathematician_correct_second_mathematician_correct_third_mathematician_incorrect_l240_240329

def first_packet_blue_candies_1 : ℕ := 2
def first_packet_total_candies_1 : ℕ := 5

def second_packet_blue_candies_1 : ℕ := 3
def second_packet_total_candies_1 : ℕ := 8

def first_packet_blue_candies_2 : ℕ := 4
def first_packet_total_candies_2 : ℕ := 10

def second_packet_blue_candies_2 : ℕ := 3
def second_packet_total_candies_2 : ℕ := 8

def total_blue_candies_1 : ℕ := first_packet_blue_candies_1 + second_packet_blue_candies_1
def total_candies_1 : ℕ := first_packet_total_candies_1 + second_packet_total_candies_1

def total_blue_candies_2 : ℕ := first_packet_blue_candies_2 + second_packet_blue_candies_2
def total_candies_2 : ℕ := first_packet_total_candies_2 + second_packet_total_candies_2

def prob_first : ℚ := total_blue_candies_1 / total_candies_1
def prob_second : ℚ := total_blue_candies_2 / total_candies_2

def lower_bound : ℚ := 3 / 8
def upper_bound : ℚ := 2 / 5
def third_prob : ℚ := 17 / 40

theorem first_mathematician_correct : prob_first = 5 / 13 := 
begin
  unfold prob_first,
  unfold total_blue_candies_1 total_candies_1,
  simp [first_packet_blue_candies_1, second_packet_blue_candies_1,
    first_packet_total_candies_1, second_packet_total_candies_1],
end

theorem second_mathematician_correct : prob_second = 7 / 18 := 
begin
  unfold prob_second,
  unfold total_blue_candies_2 total_candies_2,
  simp [first_packet_blue_candies_2, second_packet_blue_candies_2,
    first_packet_total_candies_2, second_packet_total_candies_2],
end

theorem third_mathematician_incorrect : ¬ (lower_bound < third_prob ∧ third_prob < upper_bound) :=
by simp [lower_bound, upper_bound, third_prob]; linarith

end first_mathematician_correct_second_mathematician_correct_third_mathematician_incorrect_l240_240329


namespace conditional_probability_correct_l240_240487

noncomputable def total_products : ℕ := 8
noncomputable def first_class_products : ℕ := 6
noncomputable def chosen_products : ℕ := 2

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def P_A : ℚ := 1 - (combination first_class_products chosen_products) / (combination total_products chosen_products)
noncomputable def P_AB : ℚ := (combination 2 1 * combination first_class_products 1) / (combination total_products chosen_products)

noncomputable def conditional_probability : ℚ := P_AB / P_A

theorem conditional_probability_correct :
  conditional_probability = 12 / 13 :=
  sorry

end conditional_probability_correct_l240_240487


namespace ravi_first_has_more_than_500_paperclips_on_wednesday_l240_240315

noncomputable def paperclips (k : Nat) : Nat :=
  5 * 4^k

theorem ravi_first_has_more_than_500_paperclips_on_wednesday :
  ∃ k : Nat, paperclips k > 500 ∧ k = 3 :=
by
  sorry

end ravi_first_has_more_than_500_paperclips_on_wednesday_l240_240315


namespace probability_two_copresidents_l240_240370

theorem probability_two_copresidents :
  let club_sizes := [6, 9, 10]
  let copresidents := 3
  let selected := 4
  let total_probability :=
    (1 / 3) * ((3 / 5) + (5 / 14) + (3 / 10))
  in total_probability = 44 / 105 :=
by
  sorry

end probability_two_copresidents_l240_240370


namespace weekly_earnings_l240_240446

theorem weekly_earnings :
  let hours_Monday := 2
  let minutes_Tuesday := 75
  let start_Thursday := (15, 10) -- 3:10 PM in (hour, minute) format
  let end_Thursday := (17, 45) -- 5:45 PM in (hour, minute) format
  let minutes_Saturday := 45

  let pay_rate_weekday := 4 -- \$4 per hour
  let pay_rate_weekend := 5 -- \$5 per hour

  -- Convert time to hours
  let hours_Tuesday := minutes_Tuesday / 60.0
  let Thursday_work_minutes := (end_Thursday.1 * 60 + end_Thursday.2) - (start_Thursday.1 * 60 + start_Thursday.2)
  let hours_Thursday := Thursday_work_minutes / 60.0
  let hours_Saturday := minutes_Saturday / 60.0

  -- Calculate earnings
  let earnings_Monday := hours_Monday * pay_rate_weekday
  let earnings_Tuesday := hours_Tuesday * pay_rate_weekday
  let earnings_Thursday := hours_Thursday * pay_rate_weekday
  let earnings_Saturday := hours_Saturday * pay_rate_weekend

  -- Total earnings
  let total_earnings := earnings_Monday + earnings_Tuesday + earnings_Thursday + earnings_Saturday

  total_earnings = 27.08 := by sorry

end weekly_earnings_l240_240446


namespace number_of_divisors_8_factorial_l240_240038

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l240_240038


namespace fruit_difference_l240_240759

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end fruit_difference_l240_240759


namespace minimize_quadratic_l240_240379

theorem minimize_quadratic (x : ℝ) :
  (∀ y : ℝ, x^2 + 14*x + 6 ≤ y^2 + 14*y + 6) ↔ x = -7 :=
by
  sorry

end minimize_quadratic_l240_240379


namespace smallest_value_x_squared_plus_six_x_plus_nine_l240_240105

theorem smallest_value_x_squared_plus_six_x_plus_nine : ∀ x : ℝ, x^2 + 6 * x + 9 ≥ 0 :=
by sorry

end smallest_value_x_squared_plus_six_x_plus_nine_l240_240105


namespace find_a_range_l240_240442

noncomputable
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≤ 0 then x * Real.exp x else a * x ^ 2 - 2 * x

theorem find_a_range (a : ℝ) : (∀ x : ℝ, -1 / Real.exp 1 ≤ f x a) → a ∈ Set.Ici (Real.exp 1) :=
  sorry

end find_a_range_l240_240442


namespace theater_cost_per_square_foot_l240_240373

theorem theater_cost_per_square_foot
    (n_seats : ℕ)
    (space_per_seat : ℕ)
    (cost_ratio : ℕ)
    (partner_coverage : ℕ)
    (tom_expense : ℕ)
    (total_seats := 500)
    (square_footage := total_seats * space_per_seat)
    (construction_cost := cost_ratio * land_cost)
    (total_cost := land_cost + construction_cost)
    (partner_expense := total_cost * partner_coverage / 100)
    (tom_expense_ratio := 100 - partner_coverage)
    (cost_equation := tom_expense = total_cost * tom_expense_ratio / 100)
    (land_cost := 30000) :
    tom_expense = 54000 → 
    space_per_seat = 12 → 
    cost_ratio = 2 →
    partner_coverage = 40 → 
    tom_expense_ratio = 60 → 
    total_cost = 90000 → 
    total_cost / 3 = land_cost →
    land_cost / square_footage = 5 :=
    sorry

end theater_cost_per_square_foot_l240_240373


namespace regular_polygon_sides_l240_240921

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240921


namespace inequality_for_positive_reals_l240_240981

theorem inequality_for_positive_reals (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
    (x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y)) 
    ≥ (Real.sqrt (3 / 2) * Real.sqrt (x + y + z)) := 
sorry

end inequality_for_positive_reals_l240_240981


namespace probability_of_specific_event_l240_240546

noncomputable def adam_probability := 1 / 5
noncomputable def beth_probability := 2 / 9
noncomputable def jack_probability := 1 / 6
noncomputable def jill_probability := 1 / 7
noncomputable def sandy_probability := 1 / 8

theorem probability_of_specific_event :
  (1 - adam_probability) * beth_probability * (1 - jack_probability) * jill_probability * sandy_probability = 1 / 378 := by
  sorry

end probability_of_specific_event_l240_240546


namespace range_of_x_l240_240436

open Real

def p (x : ℝ) : Prop := log (x^2 - 2 * x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4
def not_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 4

theorem range_of_x (x : ℝ) :
  (¬ p x ∧ ¬ q x ∧ (p x ∨ q x)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
sorry

end range_of_x_l240_240436


namespace triangle_larger_segment_cutoff_l240_240217

open Real

theorem triangle_larger_segment_cutoff (a b c h s₁ s₂ : ℝ) (habc : a = 35) (hbc : b = 85) (hca : c = 90)
  (hh : h = 90)
  (eq₁ : a^2 = s₁^2 + h^2)
  (eq₂ : b^2 = s₂^2 + h^2)
  (h_sum : s₁ + s₂ = c) :
  max s₁ s₂ = 78.33 :=
by
  sorry

end triangle_larger_segment_cutoff_l240_240217


namespace paint_grid_condition_l240_240975

variables {a b c d e A B C D E : ℕ}

def is_valid (n : ℕ) : Prop := n = 2 ∨ n = 3

theorem paint_grid_condition 
  (ha : is_valid a) (hb : is_valid b) (hc : is_valid c) 
  (hd : is_valid d) (he : is_valid e) (hA : is_valid A) 
  (hB : is_valid B) (hC : is_valid C) (hD : is_valid D) 
  (hE : is_valid E) :
  a + b + c + d + e = A + B + C + D + E :=
sorry

end paint_grid_condition_l240_240975


namespace evaluate_expression_l240_240561

theorem evaluate_expression (a x : ℤ) (h : x = a + 7) : x - a + 3 = 10 := by
  sorry

end evaluate_expression_l240_240561


namespace roots_of_equation_l240_240788

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l240_240788


namespace pos_divisors_8_factorial_l240_240030

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l240_240030


namespace solve_for_y_l240_240638

theorem solve_for_y : ∀ y : ℚ, (8 * y^2 + 78 * y + 5) / (2 * y + 19) = 4 * y + 2 → y = -16.5 :=
by
  intro y
  intro h
  sorry

end solve_for_y_l240_240638


namespace seq_not_square_l240_240475

open Nat

theorem seq_not_square (n : ℕ) (r : ℕ) :
  (r = 11 ∨ r = 111 ∨ r = 1111 ∨ ∃ k : ℕ, r = k * 10^(n + 1) + 1) →
  (r % 4 = 3) →
  (¬ ∃ m : ℕ, r = m^2) :=
by
  intro h_seq h_mod
  intro h_square
  sorry

end seq_not_square_l240_240475


namespace rewrite_sum_l240_240570

theorem rewrite_sum (S_b S : ℕ → ℕ) (n S_1 : ℕ) (a b c : ℕ) :
  b = 4 → (a + b + c) / 3 = 6 →
  S_b n = b * n + (a + b + c) / 3 * (S n - n * S_1) →
  S_b n = 4 * n + 6 * (S n - n * S_1) := by
sorry

end rewrite_sum_l240_240570


namespace geom_seq_sum_relation_l240_240286

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_sum_relation (h_geom : is_geometric_sequence a q)
  (h_pos : ∀ n, a n > 0) (h_q_ne_one : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 :=
by
  sorry

end geom_seq_sum_relation_l240_240286


namespace ratio_of_girls_to_boys_l240_240760

variable (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : g + b = 36)
                               (h₂ : g = b + 6) : g / b = 7 / 5 :=
by sorry

end ratio_of_girls_to_boys_l240_240760


namespace solve_for_x_l240_240482

theorem solve_for_x (x : ℝ) (h : (3 * x - 17) / 4 = (x + 12) / 5) : x = 12.09 :=
by
  sorry

end solve_for_x_l240_240482


namespace pyramid_base_side_length_l240_240080

theorem pyramid_base_side_length
  (lateral_face_area : Real)
  (slant_height : Real)
  (s : Real)
  (h_lateral_face_area : lateral_face_area = 200)
  (h_slant_height : slant_height = 40)
  (h_area_formula : lateral_face_area = 0.5 * s * slant_height) :
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l240_240080


namespace kendall_nickels_l240_240622

def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

theorem kendall_nickels (q d : ℕ) (total : ℝ) (hq : q = 10) (hd : d = 12) (htotal : total = 4) : 
  ∃ n : ℕ, value_of_nickels n = total - (value_of_quarters q + value_of_dimes d) ∧ n = 6 :=
by
  sorry

end kendall_nickels_l240_240622


namespace find_center_of_circle_l240_240673

noncomputable def center_of_circle_tangent_to_parabola : (ℝ × ℝ) :=
  let a : ℝ := 3 in
  let b : ℝ := 97 / 10 in
  (a, b)

theorem find_center_of_circle :
  ∃ a b : ℝ, (0, 3) ∈ {p : ℝ × ℝ | (p.1 - a) ^ 2 + (p.2 - b) ^ 2 = (3 - a) ^ 2 + (9 - b) ^ 2} ∧
              ∀ y, y = x^2 → deriv y 3 = 6 →
              a = 6b - 57 ∧
              (0 - a)^2 + (3 - b)^2 = (3 - a)^2 + (9 - b)^2 ∧
              (a, b) = (3, 97/10) :=
by
  sorry

end find_center_of_circle_l240_240673


namespace equivalent_single_discount_l240_240393

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.25
noncomputable def coupon_discount : ℝ := 0.10
noncomputable def final_price : ℝ := 33.75

theorem equivalent_single_discount :
  (1 - final_price / original_price) * 100 = 32.5 :=
by
  sorry

end equivalent_single_discount_l240_240393


namespace f_is_even_if_g_is_odd_l240_240463

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l240_240463


namespace solve_for_x_l240_240437

theorem solve_for_x (x y : ℝ) (h1 : 3 * x + y = 75) (h2 : 2 * (3 * x + y) - y = 138) : x = 21 :=
  sorry

end solve_for_x_l240_240437


namespace unique_k_for_triangle_inequality_l240_240695

theorem unique_k_for_triangle_inequality (k : ℕ) (h : 0 < k) :
  (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → k * (a * b + b * c + c * a) > 5 * (a * b + b * b + c * c) → a + b > c ∧ b + c > a ∧ c + a > b) ↔ (k = 6) :=
by
  sorry

end unique_k_for_triangle_inequality_l240_240695


namespace lava_lamps_probability_l240_240636

noncomputable def lampProbability : ℚ :=
  let total_arrangements := (Nat.choose 8 4) * (Nat.choose 8 4)
  let favorable_arrangements := (Nat.choose 6 3) * (Nat.choose 6 3)
  favorable_arrangements / total_arrangements

theorem lava_lamps_probability : lampProbability = 4 / 49 := by
  sorry

end lava_lamps_probability_l240_240636


namespace number_of_valid_sets_l240_240215

open Set

variable {α : Type} (a b : α)

def is_valid_set (M : Set α) : Prop := M ∪ {a} = {a, b}

theorem number_of_valid_sets (a b : α) : (∃! M : Set α, is_valid_set a b M) := 
sorry

end number_of_valid_sets_l240_240215


namespace percent_sold_second_day_l240_240135

-- Defining the problem conditions
def initial_pears (x : ℕ) : ℕ := x
def pears_sold_first_day (x : ℕ) : ℕ := (20 * x) / 100
def pears_remaining_after_first_sale (x : ℕ) : ℕ := x - pears_sold_first_day x
def pears_thrown_away_first_day (x : ℕ) : ℕ := (50 * pears_remaining_after_first_sale x) / 100
def pears_remaining_after_first_day (x : ℕ) : ℕ := pears_remaining_after_first_sale x - pears_thrown_away_first_day x
def total_pears_thrown_away (x : ℕ) : ℕ := (72 * x) / 100
def pears_thrown_away_second_day (x : ℕ) : ℕ := total_pears_thrown_away x - pears_thrown_away_first_day x
def pears_remaining_after_second_day (x : ℕ) : ℕ := pears_remaining_after_first_day x - pears_thrown_away_second_day x

-- Prove that the vendor sold 20% of the remaining pears on the second day
theorem percent_sold_second_day (x : ℕ) (h : x > 0) :
  ((pears_remaining_after_second_day x * 100) / pears_remaining_after_first_day x) = 20 :=
by 
  sorry

end percent_sold_second_day_l240_240135


namespace regular_polygon_sides_l240_240880

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240880


namespace difference_of_interchanged_digits_l240_240083

theorem difference_of_interchanged_digits (X Y : ℕ) (h1 : X - Y = 3) :
  (10 * X + Y) - (10 * Y + X) = 27 := by
  sorry

end difference_of_interchanged_digits_l240_240083


namespace maddie_milk_usage_l240_240066

-- Define the constants based on the problem conditions
def cups_per_day : ℕ := 2
def ounces_per_cup : ℝ := 1.5
def bag_cost : ℝ := 8
def ounces_per_bag : ℝ := 10.5
def weekly_coffee_expense : ℝ := 18
def gallon_milk_cost : ℝ := 4

-- Define the proof problem
theorem maddie_milk_usage : 
  (0.5 : ℝ) = (weekly_coffee_expense - 2 * ((cups_per_day * ounces_per_cup * 7) / ounces_per_bag * bag_cost)) / gallon_milk_cost :=
by 
  sorry

end maddie_milk_usage_l240_240066


namespace max_value_of_xyz_l240_240627

theorem max_value_of_xyz (x y z : ℝ) (h : x + 3 * y + z = 5) : xy + xz + yz ≤ 125 / 4 := 
sorry

end max_value_of_xyz_l240_240627


namespace radius_of_circumscribed_circle_l240_240363

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l240_240363


namespace smallest_class_size_l240_240737

/--
In a science class, students are separated into five rows for an experiment. 
The class size must be greater than 50. 
Three rows have the same number of students, one row has two more students than the others, 
and another row has three more students than the others.
Prove that the smallest possible class size for this science class is 55.
-/
theorem smallest_class_size (class_size : ℕ) (n : ℕ) 
  (h1 : class_size = 3 * n + (n + 2) + (n + 3))
  (h2 : class_size > 50) :
  class_size = 55 :=
sorry

end smallest_class_size_l240_240737


namespace similar_triangles_area_ratio_l240_240094

theorem similar_triangles_area_ratio (r : ℚ) (h : r = 1/3) : (r^2) = 1/9 :=
by
  sorry

end similar_triangles_area_ratio_l240_240094


namespace range_of_m_l240_240573

theorem range_of_m {m : ℝ} (h1 : ∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1)))
                   (h2 : ∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))
                   (h3 : ¬(∀ (x : ℝ), 1 < x → (2 / (x - m) < 2 / (x - m + 1))) ∧
                           (∃ (a : ℝ), -1 ≤ a ∧ a ≤ 1 ∧ ∀ (x1 x2 : ℝ), x1 + x2 = a ∧ x1 * x2 = -2 → m^2 + 5 * m - 3 ≥ abs (x1 - x2))) :
  m > 1 :=
by
  sorry

end range_of_m_l240_240573


namespace circle_radius_l240_240358

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l240_240358


namespace polynomial_divisible_l240_240476

theorem polynomial_divisible (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, (x-1)^3 ∣ x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 :=
by
  sorry

end polynomial_divisible_l240_240476


namespace number_exceeds_its_fraction_by_35_l240_240239

theorem number_exceeds_its_fraction_by_35 (x : ℝ) (h : x = (3 / 8) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_its_fraction_by_35_l240_240239


namespace solve_abcd_l240_240421

theorem solve_abcd : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 - d * x| ≤ 1) ∧ 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 + a * x^2 + b * x + c| ≤ 1) →
  d = 3 ∧ b = -3 ∧ a = 0 ∧ c = 0 :=
by
  sorry

end solve_abcd_l240_240421


namespace circle_radius_of_square_perimeter_eq_area_l240_240350

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l240_240350


namespace garden_area_increase_l240_240386

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 20
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def circle_radius : ℝ := original_garden_perimeter / (2 * Real.pi)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius ^ 2)

noncomputable def area_increase : ℝ := circle_area - original_garden_area

theorem garden_area_increase :
  area_increase = (6400 / Real.pi) - 1200 :=
by 
  sorry -- proof goes here

end garden_area_increase_l240_240386


namespace regular_polygon_sides_l240_240946

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240946


namespace arithmetic_sequence_geometric_mean_l240_240310

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9 * d)
  (h3 : a (k + 1) = a 1 + k * d)
  (h4 : a (2 * k + 1) = a 1 + (2 * k) * d)
  (h_gm : (a k) ^ 2 = a 1 * a (2 * k)) :
  k = 4 :=
sorry

end arithmetic_sequence_geometric_mean_l240_240310


namespace A_beats_B_by_seconds_l240_240049

theorem A_beats_B_by_seconds :
  ∀ (t_A : ℝ) (distance_A distance_B : ℝ),
  t_A = 156.67 →
  distance_A = 1000 →
  distance_B = 940 →
  (distance_A * t_A = 60 * (distance_A / t_A)) →
  t_A ≠ 0 →
  ((60 * t_A / distance_A) = 9.4002) :=
by
  intros t_A distance_A distance_B h1 h2 h3 h4 h5
  sorry

end A_beats_B_by_seconds_l240_240049


namespace grazing_b_l240_240136

theorem grazing_b (A_oxen_months B_oxen_months C_oxen_months total_months total_rent C_rent B_oxen : ℕ) 
  (hA : A_oxen_months = 10 * 7)
  (hB : B_oxen_months = B_oxen * 5)
  (hC : C_oxen_months = 15 * 3)
  (htotal : total_months = A_oxen_months + B_oxen_months + C_oxen_months)
  (hrent : total_rent = 175)
  (hC_rent : C_rent = 45)
  (hC_share : C_oxen_months / total_months = C_rent / total_rent) :
  B_oxen = 12 :=
by
  sorry

end grazing_b_l240_240136


namespace value_of_f_g_3_l240_240288

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x + 2

theorem value_of_f_g_3 : f (g 3) = 83 := by
  sorry

end value_of_f_g_3_l240_240288


namespace quadratic_inequality_solution_l240_240722

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x ^ 2 + 4 * x + 1 > 0) ↔ (a > 4) :=
sorry

end quadratic_inequality_solution_l240_240722


namespace circle_radius_eq_l240_240346

theorem circle_radius_eq {s r : ℝ} 
  (h1 : s * real.sqrt 2 = 2 * r)
  (h2 : 4 * s = real.pi * r^2) :
  r = 2 * real.sqrt 2 / real.pi :=
by
  sorry

end circle_radius_eq_l240_240346


namespace values_of_a_for_single_root_l240_240006

theorem values_of_a_for_single_root (a : ℝ) :
  (∃ (x : ℝ), ax^2 - 4 * x + 2 = 0) ∧ (∀ (x1 x2 : ℝ), ax^2 - 4 * x1 + 2 = 0 → ax^2 - 4 * x2 + 2 = 0 → x1 = x2) ↔ a = 0 ∨ a = 2 :=
sorry

end values_of_a_for_single_root_l240_240006


namespace sum_of_intercepts_l240_240335

theorem sum_of_intercepts (a b c : ℕ) :
  (∃ y, x = 2 * y^2 - 6 * y + 3 ∧ x = a ∧ y = 0) ∧
  (∃ y1 y2, x = 0 ∧ 2 * y1^2 - 6 * y1 + 3 = 0 ∧ 2 * y2^2 - 6 * y2 + 3 = 0 ∧ y1 + y2 = b + c) →
  a + b + c = 6 :=
by 
  sorry

end sum_of_intercepts_l240_240335


namespace regular_polygon_sides_l240_240934

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240934


namespace number_of_moles_of_NaCl_l240_240424

theorem number_of_moles_of_NaCl
  (moles_NaOH : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : 2 * moles_NaOH + moles_Cl2 = 2 * moles_NaOH + 1) :
  2 * moles_Cl2 = 2 := by 
  sorry

end number_of_moles_of_NaCl_l240_240424


namespace expected_value_of_days_eq_m_plus_denominator_eq_l240_240304

-- Define the size of the set
def n : ℕ := 8

-- Define the number of non-empty subsets of the set {1,2,3,4,5,6,7,8}
def subsets_count (n : ℕ) : ℕ := 2^n - 1

-- Define the expected number of days
noncomputable def expected_days (n : ℕ) : ℚ :=
  let sum := ∑ k in Finset.range n, (Nat.choose n (k+1) : ℚ) * (1 / 2^(n - (k+1)))
  in sum

-- Define m and n
def m : ℕ := 205
def denominator : ℕ := 8

-- Define the expected value in fractional format
def expected_value := (m : ℚ) / denominator

-- Prove that the expected value of days matches the computed expected value
theorem expected_value_of_days_eq : expected_days 8 = expected_value := 
  by sorry

-- Prove that m + denominators is 213
theorem m_plus_denominator_eq : m + denominator = 213 :=
  by sorry

end expected_value_of_days_eq_m_plus_denominator_eq_l240_240304


namespace cosine_of_half_pi_minus_double_alpha_l240_240577

theorem cosine_of_half_pi_minus_double_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (π / 2 - 2 * α) = 4 / 5 :=
sorry

end cosine_of_half_pi_minus_double_alpha_l240_240577


namespace x_intercept_of_quadratic_l240_240703

theorem x_intercept_of_quadratic (a b c : ℝ) (h_vertex : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 4 ∧ y = -2) 
(h_intercept : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 1 ∧ y = 0) : 
∃ x : ℝ, x = 7 ∧ ∃ y : ℝ, y = a * x^2 + b * x + c ∧ y = 0 :=
sorry

end x_intercept_of_quadratic_l240_240703


namespace minimum_cubes_required_l240_240674

def box_length := 12
def box_width := 16
def box_height := 6
def cube_volume := 3

def volume_box := box_length * box_width * box_height

theorem minimum_cubes_required : volume_box / cube_volume = 384 := by
  sorry

end minimum_cubes_required_l240_240674


namespace pizza_slices_left_l240_240113

def initial_slices : ℕ := 16
def eaten_during_dinner : ℕ := initial_slices / 4
def remaining_after_dinner : ℕ := initial_slices - eaten_during_dinner
def yves_eaten : ℕ := remaining_after_dinner / 4
def remaining_after_yves : ℕ := remaining_after_dinner - yves_eaten
def siblings_eaten : ℕ := 2 * 2
def remaining_after_siblings : ℕ := remaining_after_yves - siblings_eaten

theorem pizza_slices_left : remaining_after_siblings = 5 := by
  sorry

end pizza_slices_left_l240_240113


namespace division_identity_l240_240376

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end division_identity_l240_240376


namespace roots_of_equation_l240_240789

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l240_240789


namespace inequality_division_by_two_l240_240600

theorem inequality_division_by_two (x y : ℝ) (h : x > y) : (x / 2) > (y / 2) := 
sorry

end inequality_division_by_two_l240_240600


namespace regular_polygon_sides_l240_240961

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240961


namespace regular_polygon_sides_l240_240887

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240887


namespace evaluate_expression_l240_240418

theorem evaluate_expression : (8^6 / 8^4) * 3^10 = 3783136 := by
  sorry

end evaluate_expression_l240_240418


namespace visible_product_divisible_by_48_l240_240401

-- We represent the eight-sided die as the set {1, 2, 3, 4, 5, 6, 7, 8}.
-- Q is the product of any seven numbers from this set.

theorem visible_product_divisible_by_48 
   (Q : ℕ)
   (H : ∃ (numbers : Finset ℕ), numbers ⊆ ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ) ∧ numbers.card = 7 ∧ Q = numbers.prod id) :
   48 ∣ Q :=
by
  sorry

end visible_product_divisible_by_48_l240_240401


namespace problem_1_problem_2_l240_240498

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h₀ : a = 1) (h₁ : ∀ x : ℝ, x^2 - 5 * a * x + 4 * a^2 < 0)
                                    (h₂ : ∀ x : ℝ, (x - 2) * (x - 5) < 0) :
  ∃ x : ℝ, 2 < x ∧ x < 4 :=
by sorry

-- Proof Problem 2
theorem problem_2 (p q : ℝ → Prop) (h₀ : ∀ x : ℝ, p x → q x) 
                                (p_def : ∀ (a : ℝ) (x : ℝ), 0 < a → p x ↔ a < x ∧ x < 4 * a) 
                                (q_def : ∀ x : ℝ, q x ↔ 2 < x ∧ x < 5) :
  ∃ a : ℝ, (5 / 4) ≤ a ∧ a ≤ 2 :=
by sorry

end problem_1_problem_2_l240_240498


namespace polynomial_remainder_l240_240984

theorem polynomial_remainder (x : ℂ) : (x^1500) % (x^3 - 1) = 1 := 
sorry

end polynomial_remainder_l240_240984


namespace regular_polygon_sides_l240_240846

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240846


namespace smallest_number_remainder_l240_240812

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l240_240812


namespace cyclic_quadrilateral_equality_l240_240314

variables {A B C D : ℝ} (AB BC CD DA AC BD : ℝ)

theorem cyclic_quadrilateral_equality 
  (h_cyclic: A * B * C * D = AB * BC * CD * DA)
  (h_sides: AB = A ∧ BC = B ∧ CD = C ∧ DA = D)
  (h_diagonals: AC = E ∧ BD = F) :
  E * (A * B + C * D) = F * (D * A + B * C) :=
sorry

end cyclic_quadrilateral_equality_l240_240314


namespace g_odd_l240_240751

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem g_odd {x₁ x₂ : ℝ} 
  (h₁ : |f x₁ + f x₂| ≥ |g x₁ + g x₂|)
  (hf_odd : ∀ x, f x = -f (-x)) : ∀ x, g x = -g (-x) :=
by
  -- The proof would go here, but it's omitted for the purpose of this translation.
  sorry

end g_odd_l240_240751


namespace regular_polygon_sides_l240_240858

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240858


namespace number_of_integers_in_double_inequality_l240_240011

noncomputable def pi_approx : ℝ := 3.14
noncomputable def sqrt_pi_approx : ℝ := Real.sqrt pi_approx
noncomputable def lower_bound : ℝ := -12 * sqrt_pi_approx
noncomputable def upper_bound : ℝ := 15 * pi_approx

theorem number_of_integers_in_double_inequality : 
  ∃ n : ℕ, n = 13 ∧ ∀ k : ℤ, lower_bound ≤ (k^2 : ℝ) ∧ (k^2 : ℝ) ≤ upper_bound → (-6 ≤ k ∧ k ≤ 6) :=
by
  sorry

end number_of_integers_in_double_inequality_l240_240011


namespace rectangle_area_1600_l240_240337

theorem rectangle_area_1600
  (l w : ℝ)
  (h1 : l = 4 * w)
  (h2 : 2 * l + 2 * w = 200) :
  l * w = 1600 :=
by
  sorry

end rectangle_area_1600_l240_240337


namespace regular_polygon_sides_l240_240925

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240925


namespace number_of_adults_l240_240141

theorem number_of_adults (total_bill : ℕ) (cost_per_meal : ℕ) (num_children : ℕ) (total_cost_children : ℕ) 
  (remaining_cost_for_adults : ℕ) (num_adults : ℕ) 
  (H1 : total_bill = 56)
  (H2 : cost_per_meal = 8)
  (H3 : num_children = 5)
  (H4 : total_cost_children = num_children * cost_per_meal)
  (H5 : remaining_cost_for_adults = total_bill - total_cost_children)
  (H6 : num_adults = remaining_cost_for_adults / cost_per_meal) :
  num_adults = 2 :=
by
  sorry

end number_of_adults_l240_240141


namespace angle_ratio_in_triangle_l240_240186

theorem angle_ratio_in_triangle
  (triangle : Type)
  (A B C P Q M : triangle)
  (angle : triangle → triangle → triangle → ℝ)
  (ABC_half : angle A B Q = angle Q B C)
  (BP_BQ_bisect_ABC : angle A B P = angle P B Q)
  (BM_bisects_PBQ : angle M B Q = angle M B P)
  : angle M B Q / angle A B Q = 1 / 4 :=
by 
  sorry

end angle_ratio_in_triangle_l240_240186


namespace sequence_50th_term_l240_240118

def sequence_term (n : ℕ) : ℕ × ℕ :=
  (5 + (n - 1), n - 1)

theorem sequence_50th_term :
  sequence_term 50 = (54, 49) :=
by
  sorry

end sequence_50th_term_l240_240118


namespace division_remainder_l240_240761

theorem division_remainder (dividend quotient divisor remainder : ℕ) 
  (h_dividend : dividend = 12401) 
  (h_quotient : quotient = 76) 
  (h_divisor : divisor = 163) 
  (h_remainder : dividend = quotient * divisor + remainder) : 
  remainder = 13 := 
by
  sorry

end division_remainder_l240_240761


namespace units_digits_no_match_l240_240534

theorem units_digits_no_match : ∀ x : ℕ, 1 ≤ x ∧ x ≤ 100 → (x % 10 ≠ (101 - x) % 10) :=
by
  intro x hx
  sorry

end units_digits_no_match_l240_240534


namespace regular_polygon_sides_l240_240842

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240842


namespace graph_is_hyperbola_l240_240110

theorem graph_is_hyperbola : ∀ x y : ℝ, (x + y) ^ 2 = x ^ 2 + y ^ 2 + 2 * x + 2 * y ↔ (x - 1) * (y - 1) = 1 := 
by {
  sorry
}

end graph_is_hyperbola_l240_240110


namespace inverse_proportional_ratios_l240_240486

variables {x y x1 x2 y1 y2 : ℝ}
variables (h_inv_prop : ∀ (x y : ℝ), x * y = 1) (hx1_ne : x1 ≠ 0) (hx2_ne : x2 ≠ 0) (hy1_ne : y1 ≠ 0) (hy2_ne : y2 ≠ 0)

theorem inverse_proportional_ratios 
  (h1 : x1 * y1 = x2 * y2)
  (h2 : (x1 / x2) = (3 / 4)) : 
  (y1 / y2) = (4 / 3) :=
by 

sorry

end inverse_proportional_ratios_l240_240486


namespace shorter_piece_length_correct_l240_240385

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ := 
  total_length * ratio / (ratio + 1)

theorem shorter_piece_length_correct :
  shorter_piece_length 57.134 (3.25678 / 7.81945) = 16.790 :=
by
  sorry

end shorter_piece_length_correct_l240_240385


namespace regular_polygon_sides_l240_240910

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240910


namespace natural_number_increased_by_one_l240_240827

theorem natural_number_increased_by_one (a : ℕ) 
  (h : (a + 1) ^ 2 - a ^ 2 = 1001) : 
  a = 500 := 
sorry

end natural_number_increased_by_one_l240_240827


namespace regular_polygon_sides_l240_240908

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240908


namespace probability_of_different_groups_is_correct_l240_240798

-- Define the number of total members and groups
def num_groups : ℕ := 6
def members_per_group : ℕ := 3
def total_members : ℕ := num_groups * members_per_group

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 3 people from different groups
noncomputable def probability_different_groups : ℚ :=
  binom num_groups 3 / binom total_members 3

-- State the theorem we want to prove
theorem probability_of_different_groups_is_correct :
  probability_different_groups = 5 / 204 :=
by
  sorry

end probability_of_different_groups_is_correct_l240_240798


namespace common_ratio_value_l240_240715

variable {α : Type*} [Field α]
variable {a : ℕ → α} {q : α} (neq1 : q ≠ 1)

-- Definition for geometric sequence
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Definition for arithmetic sequence condition
def is_arithmetic_sequence (a1 a2 a3 : α) : Prop :=
  2 * a3 = a1 + a2

-- Main theorem
theorem common_ratio_value (h1 : is_geometric_sequence a q) (h2 : is_arithmetic_sequence (a 1) (a 2) (a 3)) : q = - (1 / 2) :=
by
  sorry

end common_ratio_value_l240_240715


namespace arithmetic_sequence_geometric_sequence_sum_first_n_terms_l240_240276

noncomputable def S (n : ℕ) := nat.log (nat.sqrt n) -- Dummy definition for S_n

axiom sum_of_sequence (n : ℕ) : 8 * S n = (a n + 2) ^ 2
axiom a_def (n : ℕ) : a n = log (sqrt 3) (b n)

theorem arithmetic_sequence (n : ℕ) : 
  ∀ (a : ℕ → ℝ), 8 * S n = (a n + 2) ^ 2 → 
  ∃ d : ℝ, ∃ a0 : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem geometric_sequence (n : ℕ) :
  ∀ (b : ℕ → ℝ), (a n = log (sqrt 3) (b n)) → 
  T n = (∑ i in range (n + 1), b i) :=
sorry

theorem sum_first_n_terms (n : ℕ) :
  T n = 3 * (9 ^ n - 1) / 8 :=
sorry

end arithmetic_sequence_geometric_sequence_sum_first_n_terms_l240_240276


namespace avg_rate_change_l240_240204

def f (x : ℝ) : ℝ := x^2 + x

theorem avg_rate_change : (f 2 - f 1) / (2 - 1) = 4 := by
  -- here the proof steps should follow
  sorry

end avg_rate_change_l240_240204


namespace speed_of_A_l240_240543

theorem speed_of_A (V_B : ℝ) (h_VB : V_B = 4.555555555555555)
  (h_B_overtakes: ∀ (t_A t_B : ℝ), t_A = t_B + 0.5 → t_B = 1.8) 
  : ∃ V_A : ℝ, V_A = 3.57 :=
by
  sorry

end speed_of_A_l240_240543


namespace mila_calculator_sum_l240_240632

theorem mila_calculator_sum :
  let n := 60
  let calc1_start := 2
  let calc2_start := 0
  let calc3_start := -1
  calc1_start^(3^n) + calc2_start^2^(n) + (-calc3_start)^n = 2^(3^60) + 1 :=
by {
  sorry
}

end mila_calculator_sum_l240_240632


namespace sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l240_240232

def avg_daily_production := 400
def weekly_planned_production := 2800
def daily_deviations := [15, -5, 21, 16, -7, 0, -8]
def total_weekly_deviation := 80

-- Calculation for sets produced on Saturday
def sat_production_exceeds_plan := total_weekly_deviation - (daily_deviations.take (daily_deviations.length - 1)).sum
def sat_production := avg_daily_production + sat_production_exceeds_plan

-- Calculation for the difference between the max and min production days
def max_deviation := max sat_production_exceeds_plan (daily_deviations.maximum.getD 0)
def min_deviation := min sat_production_exceeds_plan (daily_deviations.minimum.getD 0)
def highest_lowest_diff := max_deviation - min_deviation

-- Calculation for the weekly wage for each worker
def workers := 20
def daily_wage := 200
def basic_weekly_wage := daily_wage * 7
def additional_wage := (15 + 21 + 16 + sat_production_exceeds_plan) * 10 - (5 + 7 + 8) * 15
def total_bonus := additional_wage / workers
def total_weekly_wage := basic_weekly_wage + total_bonus

theorem sat_production_correct : sat_production = 448 := by
  sorry

theorem highest_lowest_diff_correct : highest_lowest_diff = 56 := by
  sorry

theorem total_weekly_wage_correct : total_weekly_wage = 1435 := by
  sorry

end sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l240_240232


namespace regular_polygon_sides_l240_240892

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240892


namespace sequence_noncongruent_modulo_l240_240500

theorem sequence_noncongruent_modulo 
  (a : ℕ → ℕ)
  (h0 : a 1 = 1)
  (h1 : ∀ n, a (n + 1) = a n + 2^(a n)) :
  ∀ (i j : ℕ), i ≠ j → i ≤ 32021 → j ≤ 32021 →
  (a i) % (3^2021) ≠ (a j) % (3^2021) := 
by
  sorry

end sequence_noncongruent_modulo_l240_240500


namespace factorize_problem_1_factorize_problem_2_l240_240420

-- Problem 1 Statement
theorem factorize_problem_1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := 
sorry

-- Problem 2 Statement
theorem factorize_problem_2 (x y : ℝ) : (x - y)^2 + 4 * (x * y) = (x + y)^2 := 
sorry

end factorize_problem_1_factorize_problem_2_l240_240420


namespace right_triangle_side_length_l240_240211

theorem right_triangle_side_length (c a b : ℕ) (hc : c = 13) (ha : a = 12) (hypotenuse_eq : c ^ 2 = a ^ 2 + b ^ 2) : b = 5 :=
sorry

end right_triangle_side_length_l240_240211


namespace regular_polygon_sides_l240_240915

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240915


namespace shift_parabola_l240_240292

theorem shift_parabola (x : ℝ) : 
  let y := -x^2
  let y_shifted_left := -((x + 3)^2)
  let y_shifted := y_shifted_left + 5
  y_shifted = -(x + 3)^2 + 5 := 
by {
  sorry
}

end shift_parabola_l240_240292


namespace max_n_plus_m_l240_240209

theorem max_n_plus_m : 
  ∀ (m n : ℝ), (∀ x ∈ set.Icc m n, 3 ≤ 2^(2 * x) - 2^(x + 2) + 7 ∧ 2^(2 * x) - 2^(x + 2) + 7 ≤ 7) → n + m = 3 :=
begin
  intros m n h,
  sorry
end

end max_n_plus_m_l240_240209


namespace sum_common_seq_l240_240138

-- Define the arithmetic sequences
def seq1 (n : ℕ) : ℕ := 2 * n - 1
def seq2 (n : ℕ) : ℕ := 3 * n - 2

-- Define the common term sequence
def common_seq (a : ℕ) : Prop :=
  ∃ (n1 n2 : ℕ), seq1 n1 = a ∧ seq2 n2 = a

-- Prove the sum of the first n terms of the sequence common_seq is 3n^2 - 2n
theorem sum_common_seq (n : ℕ) : ∃ S : ℕ, 
  S = (3 * n ^ 2 - 2 * n) ∧ 
  ∀ (a i : ℕ), (a = 1 + 6 * (i - 1)) → i ∈ (fin n) → common_seq a :=
sorry

end sum_common_seq_l240_240138


namespace pushing_car_effort_l240_240642

theorem pushing_car_effort (effort constant : ℕ) (people1 people2 : ℕ) 
  (h1 : constant = people1 * effort)
  (h2 : people1 = 4)
  (h3 : effort = 120)
  (h4 : people2 = 6) :
  effort * people1 = constant → constant = people2 * 80 :=
by
  sorry

end pushing_car_effort_l240_240642


namespace find_larger_number_l240_240564

theorem find_larger_number 
  (L S : ℕ) 
  (h1 : L - S = 2342) 
  (h2 : L = 9 * S + 23) : 
  L = 2624 := 
sorry

end find_larger_number_l240_240564


namespace pencil_cost_l240_240542

theorem pencil_cost (P : ℝ) : 
  (∀ pen_cost total : ℝ, pen_cost = 3.50 → total = 291 → 38 * P + 56 * pen_cost = total → P = 2.50) :=
by
  intros pen_cost total h1 h2 h3
  sorry

end pencil_cost_l240_240542


namespace baba_yaga_powder_problem_l240_240529

theorem baba_yaga_powder_problem (A B d : ℝ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 := 
sorry

end baba_yaga_powder_problem_l240_240529


namespace unique_solution_range_l240_240721
-- import relevant libraries

-- define the functions
def f (a x : ℝ) : ℝ := 2 * a * x ^ 3 + 3
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 2

-- state and prove the main theorem (statement only)
theorem unique_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = g x ∧ ∀ y : ℝ, y > 0 → f a y = g y → y = x) ↔ a ∈ Set.Iio (-1) :=
sorry

end unique_solution_range_l240_240721


namespace roots_of_quadratic_eq_l240_240786

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l240_240786


namespace spinner_sections_equal_size_l240_240799

theorem spinner_sections_equal_size 
  (p : ℕ → Prop)
  (h1 : ∀ n, p n ↔ (1 - (1: ℝ) / n) ^ 2 = 0.5625) : 
  p 4 :=
by
  sorry

end spinner_sections_equal_size_l240_240799


namespace regular_polygon_sides_l240_240920

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240920


namespace solve_trig_eq_l240_240158

open Real

theorem solve_trig_eq (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 :=
by
  sorry

end solve_trig_eq_l240_240158


namespace stratified_sampling_grade10_students_l240_240538

-- Definitions based on the given problem
def total_students := 900
def grade10_students := 300
def sample_size := 45

-- Calculation of the number of Grade 10 students in the sample
theorem stratified_sampling_grade10_students : (grade10_students * sample_size) / total_students = 15 := by
  sorry

end stratified_sampling_grade10_students_l240_240538


namespace original_number_is_two_over_three_l240_240068

theorem original_number_is_two_over_three (x : ℚ) (h : 1 + 1/x = 5/2) : x = 2/3 :=
sorry

end original_number_is_two_over_three_l240_240068


namespace regular_polygon_sides_l240_240947

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240947


namespace find_number_l240_240648

theorem find_number (x : ℝ) (h : (3.242 * 16) / x = 0.051871999999999995) : x = 1000 :=
by
  sorry

end find_number_l240_240648


namespace actual_distance_in_km_l240_240472

-- Given conditions
def scale_factor : ℕ := 200000
def map_distance_cm : ℚ := 3.5

-- Proof goal: the actual distance in kilometers
theorem actual_distance_in_km : (map_distance_cm * scale_factor) / 100000 = 7 := 
by
  sorry

end actual_distance_in_km_l240_240472


namespace f_decreasing_max_k_value_l240_240279

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing : ∀ x > 0, ∀ y > 0, x < y → f x > f y := by
  sorry

theorem max_k_value : ∀ x > 0, f x > k / (x + 1) → k ≤ 3 := by
  sorry

end f_decreasing_max_k_value_l240_240279


namespace wolf_nobel_laureates_l240_240384

/-- 31 scientists that attended a certain workshop were Wolf Prize laureates,
and some of them were also Nobel Prize laureates. Of the scientists who attended
that workshop and had not received the Wolf Prize, the number of scientists who had
received the Nobel Prize was 3 more than the number of scientists who had not received
the Nobel Prize. In total, 50 scientists attended that workshop, and 25 of them were
Nobel Prize laureates. Prove that the number of Wolf Prize laureates who were also
Nobel Prize laureates is 3. -/
theorem wolf_nobel_laureates (W N total W' N' W_N : ℕ)  
  (hW : W = 31) (hN : N = 25) (htotal : total = 50) 
  (hW' : W' = total - W) (hN' : N' = total - N) 
  (hcondition : N' - W' = 3) :
  W_N = N - W' :=
by
  sorry

end wolf_nobel_laureates_l240_240384


namespace regular_polygon_sides_l240_240834

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240834


namespace find_positive_n_for_quadratic_l240_240559

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

-- Define the condition: the quadratic equation has exactly one real root if its discriminant is zero
def has_one_real_root (a b c : ℝ) : Prop := discriminant a b c = 0

-- The specific quadratic equation y^2 + 6ny + 9n
def my_quadratic (n : ℝ) : Prop := has_one_real_root 1 (6 * n) (9 * n)

-- The statement to be proven: for the quadratic equation y^2 + 6ny + 9n to have one real root, n must be 1
theorem find_positive_n_for_quadratic : ∃ (n : ℝ), my_quadratic n ∧ n > 0 ∧ n = 1 := 
by
  sorry

end find_positive_n_for_quadratic_l240_240559


namespace cyclist_club_member_count_l240_240177

-- Define the set of valid digits.
def valid_digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 9}

-- Define the problem statement
theorem cyclist_club_member_count : valid_digits.card ^ 3 = 512 :=
by
  -- Placeholder for the proof
  sorry

end cyclist_club_member_count_l240_240177


namespace main_theorem_l240_240175

-- Define the sets M and N
def M : Set ℝ := { x | 0 < x ∧ x < 10 }
def N : Set ℝ := { x | x < -4/3 ∨ x > 3 }

-- Define the complement of N in ℝ
def comp_N : Set ℝ := { x | ¬ (x < -4/3 ∨ x > 3) }

-- The main theorem to be proved
theorem main_theorem : M ∩ comp_N = { x | 0 < x ∧ x ≤ 3 } := 
by
  sorry

end main_theorem_l240_240175


namespace average_score_of_entire_class_l240_240605

theorem average_score_of_entire_class :
  ∀ (num_students num_boys : ℕ) (avg_score_girls avg_score_boys : ℝ),
  num_students = 50 →
  num_boys = 20 →
  avg_score_girls = 85 →
  avg_score_boys = 80 →
  (avg_score_boys * num_boys + avg_score_girls * (num_students - num_boys)) / num_students = 83 :=
by
  intros num_students num_boys avg_score_girls avg_score_boys
  sorry

end average_score_of_entire_class_l240_240605


namespace charles_total_earnings_l240_240688

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end charles_total_earnings_l240_240688


namespace line_of_intersection_l240_240817

theorem line_of_intersection :
  ∀ (x y z : ℝ),
    (3 * x + 4 * y - 2 * z + 1 = 0) ∧ (2 * x - 4 * y + 3 * z + 4 = 0) →
    (∃ t : ℝ, x = -1 + 4 * t ∧ y = 1 / 2 - 13 * t ∧ z = -20 * t) :=
by
  intro x y z
  intro h
  cases h
  sorry

end line_of_intersection_l240_240817


namespace part1_part2_l240_240220

noncomputable theory

section Proof

-- Part 1
def proof1 (μ σ : ℝ) : ℝ :=
  let p_outside := 1 - 0.9974
  let X := Binomial 10 p_outside
  1 - (0.9974 ^ 10)

theorem part1 (μ σ : ℝ) : proof1 μ σ = 0.0257 :=
by sorry

-- Part 2
def possible_Y_distribution : pmf ℕ :=
  pmf.of_finite (λ y, match y with
    | 0 => 1 / 42
    | 1 => 5 / 21
    | 2 => 10 / 21
    | 3 => 5 / 21
    | 4 => 1 / 42
    | _ => 0
  end)

def expectation_Y : ℝ :=
  pmf.expected_value possible_Y_distribution (λ y, y)

theorem part2 : expectation_Y = 2 :=
by sorry

end Proof

end part1_part2_l240_240220


namespace length_of_one_side_of_regular_pentagon_l240_240181

-- Define the conditions
def is_regular_pentagon (P : ℝ) (n : ℕ) : Prop := n = 5 ∧ P = 23.4

-- State the theorem
theorem length_of_one_side_of_regular_pentagon (P : ℝ) (n : ℕ) 
  (h : is_regular_pentagon P n) : P / n = 4.68 :=
by
  sorry

end length_of_one_side_of_regular_pentagon_l240_240181


namespace museum_wings_paintings_l240_240129

theorem museum_wings_paintings (P A : ℕ) (h1: P + A = 8) (h2: P = 1 + 2) : P = 3 :=
by
  -- Proof here
  sorry

end museum_wings_paintings_l240_240129


namespace student_answered_two_questions_incorrectly_l240_240096

/-
  Defining the variables and conditions for the problem.
  x: number of questions answered correctly,
  y: number of questions not answered,
  z: number of questions answered incorrectly.
-/

theorem student_answered_two_questions_incorrectly (x y z : ℕ) 
  (h1 : x + y + z = 6) 
  (h2 : 8 * x + 2 * y = 20) : z = 2 :=
by
  /- We know the total number of questions is 6.
     And the total score is 20 with the given scoring rules.
     Thus, we need to prove that z = 2 under these conditions. -/
  sorry

end student_answered_two_questions_incorrectly_l240_240096


namespace cara_total_amount_owed_l240_240411

-- Define the conditions
def principal : ℝ := 54
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the simple interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the total amount owed calculation
def total_amount_owed (P R T : ℝ) : ℝ := P + (interest P R T)

-- The proof statement
theorem cara_total_amount_owed : total_amount_owed principal rate time = 56.70 := by
  sorry

end cara_total_amount_owed_l240_240411


namespace merge_coins_n_ge_3_merge_coins_n_eq_2_l240_240218

-- For Part 1
theorem merge_coins_n_ge_3 (n : ℕ) (hn : n ≥ 3) :
  ∃ (m : ℕ), m = 1 ∨ m = 2 :=
sorry

-- For Part 2
theorem merge_coins_n_eq_2 (r s : ℕ) :
  ∃ (k : ℕ), r + s = 2^k * Nat.gcd r s :=
sorry

end merge_coins_n_ge_3_merge_coins_n_eq_2_l240_240218


namespace roots_of_quadratic_eq_l240_240780

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l240_240780


namespace sum_of_longest_altitudes_l240_240039

theorem sum_of_longest_altitudes (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let h1 := a,
      h2 := b,
      h := (a * b) / c in
  h1 + h2 = 21 := by
{
  sorry
}

end sum_of_longest_altitudes_l240_240039


namespace regular_polygon_sides_l240_240931

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240931


namespace distribute_paper_clips_l240_240753

theorem distribute_paper_clips (total_clips : ℕ) (boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : boxes = 9) :
  total_clips / boxes = clips_per_box ↔ clips_per_box = 9 :=
by
  sorry

end distribute_paper_clips_l240_240753


namespace translation_symmetric_y_axis_phi_l240_240208

theorem translation_symmetric_y_axis_phi :
  ∀ (f : ℝ → ℝ) (φ : ℝ),
    (∀ x : ℝ, f x = Real.sin (2 * x + π / 6)) →
    (0 < φ ∧ φ ≤ π / 2) →
    (∀ x, Real.sin (2 * (x + φ) + π / 6) = Real.sin (2 * (-x + φ) + π / 6)) →
    φ = π / 6 :=
by
  intros f φ f_def φ_bounds symmetry
  sorry

end translation_symmetric_y_axis_phi_l240_240208


namespace solve_for_x_l240_240074

theorem solve_for_x (x : ℤ) (h_eq : (7 * x - 5) / (x - 2) = 2 / (x - 2)) (h_cond : x ≠ 2) : x = 1 := by
  sorry

end solve_for_x_l240_240074


namespace num_coloring_l240_240448

-- Define the set of numbers to be colored
def numbers_to_color : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

-- Define the set of colors
inductive Color
| red
| green
| blue

-- Define proper divisors for the numbers in the list
def proper_divisors (n : ℕ) : List ℕ :=
  match n with
  | 2 => []
  | 3 => []
  | 4 => [2]
  | 5 => []
  | 6 => [2, 3]
  | 7 => []
  | 8 => [2, 4]
  | 9 => [3]
  | _ => []

-- The proof statement
theorem num_coloring (h : ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, n ≠ d) :
  ∃ f : ℕ → Color, ∀ n ∈ numbers_to_color, ∀ d ∈ proper_divisors n, f n ≠ f d :=
  sorry

end num_coloring_l240_240448


namespace regular_polygon_sides_l240_240918

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240918


namespace find_value_b_in_geometric_sequence_l240_240646

theorem find_value_b_in_geometric_sequence
  (b : ℝ)
  (h1 : 15 ≠ 0) -- to ensure division by zero does not occur
  (h2 : b ≠ 0)  -- to ensure division by zero does not occur
  (h3 : 15 * (b / 15) = b) -- 15 * r = b
  (h4 : b * (b / 15) = 45 / 4) -- b * r = 45 / 4
  : b = 15 * Real.sqrt 3 / 2 :=
sorry

end find_value_b_in_geometric_sequence_l240_240646


namespace sum_of_first_six_terms_l240_240190

theorem sum_of_first_six_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (hS : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 2 = 2 → S 4 = 10 → S 6 = 24 := 
by
  intros h1 h2
  sorry

end sum_of_first_six_terms_l240_240190


namespace find_range_m_l240_240723

noncomputable def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem find_range_m (m : ℝ) : ¬ (p m ∨ ¬ q m) ↔ -2 ≤ m ∧ m ≤ 2 :=
  sorry

end find_range_m_l240_240723


namespace find_sum_of_cubes_l240_240191

theorem find_sum_of_cubes (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : (a^3 + 9) / a = (b^3 + 9) / b)
  (h₅ : (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 :=
by
  sorry

end find_sum_of_cubes_l240_240191


namespace temperature_43_l240_240205

theorem temperature_43 (T W Th F : ℝ)
  (h1 : (T + W + Th) / 3 = 42)
  (h2 : (W + Th + F) / 3 = 44)
  (h3 : T = 37) : F = 43 :=
by
  sorry

end temperature_43_l240_240205


namespace second_vote_difference_l240_240454

-- Define the total number of members
def total_members : ℕ := 300

-- Define the votes for and against in the initial vote
structure votes_initial :=
  (a : ℕ) (b : ℕ) (h : a + b = total_members) (rejected : b > a)

-- Define the votes for and against in the second vote
structure votes_second :=
  (a' : ℕ) (b' : ℕ) (h : a' + b' = total_members)

-- Define the margin and condition of passage by three times the margin
def margin (vi : votes_initial) : ℕ := vi.b - vi.a

def passage_by_margin (vi : votes_initial) (vs : votes_second) : Prop :=
  vs.a' - vs.b' = 3 * margin vi

-- Define the condition that a' is 7/6 times b
def proportion (vs : votes_second) (vi : votes_initial) : Prop :=
  vs.a' = (7 * vi.b) / 6

-- The final proof statement
theorem second_vote_difference (vi : votes_initial) (vs : votes_second)
  (h_margin : passage_by_margin vi vs)
  (h_proportion : proportion vs vi) :
  vs.a' - vi.a = 55 :=
by
  sorry  -- This is where the proof would go

end second_vote_difference_l240_240454


namespace problem_l240_240511

theorem problem (K : ℕ) : 16 ^ 3 * 8 ^ 3 = 2 ^ K → K = 21 := by
  sorry

end problem_l240_240511


namespace regular_polygon_sides_l240_240888

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240888


namespace function_relationship_l240_240172

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- The main statement we want to prove
theorem function_relationship (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on f (Set.Ici 0)) :
  f 1 > f (-10) :=
by sorry

end function_relationship_l240_240172


namespace am_gm_hm_inequality_l240_240554

variable {x y : ℝ}

-- Conditions: x and y are positive real numbers and x < y
def conditions (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ x < y

-- Proof statement: A.M. > G.M. > H.M. under given conditions
theorem am_gm_hm_inequality (x y : ℝ) (h : conditions x y) :
  (x + y) / 2 > Real.sqrt (x * y) ∧ Real.sqrt (x * y) > (2 * x * y) / (x + y) :=
sorry

end am_gm_hm_inequality_l240_240554


namespace smallest_b_for_undefined_inverse_mod_70_77_l240_240665

theorem smallest_b_for_undefined_inverse_mod_70_77 (b : ℕ) :
  (∀ k, k < b → k * 1 % 70 ≠ 1 ∧ k * 1 % 77 ≠ 1) ∧ (b * 1 % 70 ≠ 1) ∧ (b * 1 % 77 ≠ 1) → b = 7 :=
by sorry

end smallest_b_for_undefined_inverse_mod_70_77_l240_240665


namespace probability_of_arrangement_XXOXOO_l240_240769

noncomputable def probability_of_XXOXOO : ℚ :=
  let total_arrangements := (Nat.factorial 6) / ((Nat.factorial 4) * (Nat.factorial 2))
  let favorable_arrangements := 1
  favorable_arrangements / total_arrangements

theorem probability_of_arrangement_XXOXOO :
  probability_of_XXOXOO = 1 / 15 :=
by
  sorry

end probability_of_arrangement_XXOXOO_l240_240769


namespace normals_intersect_at_single_point_l240_240531

-- Definitions of points on the parabola and distinct condition
variables {a b c : ℝ}

-- Condition stating that A, B, C are distinct points
def distinct_points (a b c : ℝ) : Prop :=
  (a - b) ≠ 0 ∧ (b - c) ≠ 0 ∧ (c - a) ≠ 0

-- Statement to be proved
theorem normals_intersect_at_single_point (habc : distinct_points a b c) :
  a + b + c = 0 :=
sorry

end normals_intersect_at_single_point_l240_240531


namespace largest_n_l240_240557

theorem largest_n {x y z n : ℕ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (n:ℤ)^2 = (x:ℤ)^2 + (y:ℤ)^2 + (z:ℤ)^2 + 2*(x:ℤ)*(y:ℤ) + 2*(y:ℤ)*(z:ℤ) + 2*(z:ℤ)*(x:ℤ) + 6*(x:ℤ) + 6*(y:ℤ) + 6*(z:ℤ) - 12
  → n = 13 :=
sorry

end largest_n_l240_240557


namespace surface_area_parallelepiped_l240_240241

theorem surface_area_parallelepiped (a b : ℝ) :
  ∃ S : ℝ, (S = 3 * a * b) :=
sorry

end surface_area_parallelepiped_l240_240241


namespace question1_question2_l240_240003

noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
noncomputable def f2 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2 - 2*x

theorem question1 (a : ℝ) : 
  (∀ x : ℝ, f1 a x = 0 → ∀ y : ℝ, f1 a y = 0 → x = y) ↔ (a = 0 ∨ a < -4 / Real.exp 2) :=
sorry -- Proof of theorem 1

theorem question2 (a m n x0 : ℝ) (h : a ≠ 0) :
  (f2 a x0 = f2 a ((x0 + m) / 2) * (x0 - m) + n ∧ x0 ≠ m) → False :=
sorry -- Proof of theorem 2

end question1_question2_l240_240003


namespace factorize_poly_l240_240519

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l240_240519


namespace num_divisors_8_fact_l240_240019

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l240_240019


namespace ethan_expected_wins_l240_240417

-- Define the conditions
def P_win := 2 / 5
def P_tie := 2 / 5
def P_loss := 1 / 5

-- Define the adjusted probabilities
def adj_P_win := P_win / (P_win + P_loss)
def adj_P_loss := P_loss / (P_win + P_loss)

-- Define Ethan's expected number of wins before losing
def expected_wins_before_loss : ℚ := 2

-- The theorem to prove 
theorem ethan_expected_wins :
  ∃ E : ℚ, 
    E = (adj_P_win * (E + 1) + adj_P_loss * 0) ∧ 
    E = expected_wins_before_loss :=
by
  sorry

end ethan_expected_wins_l240_240417


namespace find_f_2017_l240_240719

noncomputable def f (x : ℤ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2017
(x : ℤ)
(a α b β : ℝ)
(h : f 4 a α b β = 3) :
f 2017 a α b β = -3 := 
sorry

end find_f_2017_l240_240719


namespace correct_calculation_l240_240224

theorem correct_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 :=
by sorry

end correct_calculation_l240_240224


namespace unit_vector_parallel_to_d_l240_240368

theorem unit_vector_parallel_to_d (x y: ℝ): (4 * x - 3 * y = 0) ∧ (x^2 + y^2 = 1) → (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) :=
by sorry

end unit_vector_parallel_to_d_l240_240368


namespace new_person_weight_l240_240228

theorem new_person_weight (weights : List ℝ) (len_weights : weights.length = 8) (replace_weight : ℝ) (new_weight : ℝ)
  (weight_diff :  (weights.sum - replace_weight + new_weight) / 8 = (weights.sum / 8) + 3) 
  (replace_weight_eq : replace_weight = 70):
  new_weight = 94 :=
sorry

end new_person_weight_l240_240228


namespace smallest_of_seven_consecutive_even_numbers_l240_240766

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end smallest_of_seven_consecutive_even_numbers_l240_240766


namespace max_value_of_expression_l240_240704

theorem max_value_of_expression (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁^2 / 4 + 9 * y₁^2 / 4 = 1) 
  (h₂ : x₂^2 / 4 + 9 * y₂^2 / 4 = 1) 
  (h₃ : x₁ * x₂ + 9 * y₁ * y₂ = -2) :
  (|2 * x₁ + 3 * y₁ - 3| + |2 * x₂ + 3 * y₂ - 3|) ≤ 6 + 2 * Real.sqrt 5 :=
sorry

end max_value_of_expression_l240_240704


namespace unique_root_value_l240_240986

theorem unique_root_value {x n : ℝ} (h : (15 - n) = 15 - (35 / 4)) :
  (x + 5) * (x + 3) = n + 3 * x → n = 35 / 4 :=
sorry

end unique_root_value_l240_240986


namespace no_monochromatic_arith_progression_l240_240318

theorem no_monochromatic_arith_progression :
  ∃ (coloring : ℕ → ℕ), (∀ n, 1 ≤ n ∧ n ≤ 2014 → coloring n ∈ {1, 2, 3, 4}) ∧
  ¬∃ (a r : ℕ), r > 0 ∧ (∀ i, i < 11 → 1 ≤ a + i * r ∧ a + i * r ≤ 2014) ∧
  (∀ i j, i < 11 → j < 11 → coloring (a + i * r) = coloring (a + j * r)) :=
begin
  -- Proof omitted
  sorry
end

end no_monochromatic_arith_progression_l240_240318


namespace farmer_harvested_correctly_l240_240537

def estimated_harvest : ℕ := 213489
def additional_harvest : ℕ := 13257
def total_harvest : ℕ := 226746

theorem farmer_harvested_correctly :
  estimated_harvest + additional_harvest = total_harvest :=
by
  sorry

end farmer_harvested_correctly_l240_240537


namespace regular_polygon_sides_l240_240900

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240900


namespace spinner_probabilities_l240_240230

theorem spinner_probabilities (pA pB pC pD : ℚ) (h1 : pA = 1/4) (h2 : pB = 1/3) (h3 : pA + pB + pC + pD = 1) :
  pC + pD = 5/12 :=
by
  -- Here you would construct the proof (left as sorry for this example)
  sorry

end spinner_probabilities_l240_240230


namespace range_of_a_l240_240283

theorem range_of_a {A B : Set ℝ} (hA : A = {x | x > 5}) (hB : B = {x | x > a}) 
  (h_sufficient_not_necessary : A ⊆ B ∧ ¬(B ⊆ A)) 
  : a < 5 :=
sorry

end range_of_a_l240_240283


namespace regular_polygon_sides_l240_240901

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240901


namespace prob_divisors_8_fact_l240_240024

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l240_240024


namespace find_purchase_price_l240_240216

noncomputable def purchase_price (a : ℝ) : ℝ := a
def retail_price : ℝ := 1100
def discount_rate : ℝ := 0.8
def profit_rate : ℝ := 0.1

theorem find_purchase_price (a : ℝ) (h : purchase_price a * (1 + profit_rate) = retail_price * discount_rate) : a = 800 := by
  sorry

end find_purchase_price_l240_240216


namespace find_second_expression_l240_240640

theorem find_second_expression (a : ℕ) (x : ℕ) 
  (h1 : (2 * a + 16 + x) / 2 = 74) (h2 : a = 28) : x = 76 := 
by
  sorry

end find_second_expression_l240_240640


namespace warehouse_capacity_l240_240391

theorem warehouse_capacity (total_bins : ℕ) (bins_20_tons : ℕ) (bins_15_tons : ℕ)
    (total_capacity : ℕ) (h1 : total_bins = 30) (h2 : bins_20_tons = 12) 
    (h3 : bins_15_tons = total_bins - bins_20_tons) 
    (h4 : total_capacity = (bins_20_tons * 20) + (bins_15_tons * 15)) : 
    total_capacity = 510 :=
by {
  sorry
}

end warehouse_capacity_l240_240391


namespace regular_polygon_sides_l240_240856

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240856


namespace radius_of_circumscribed_circle_l240_240365

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l240_240365


namespace regular_polygon_sides_l240_240930

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240930


namespace car_winning_probability_l240_240736

noncomputable def probability_of_winning (P_X P_Y P_Z : ℚ) : ℚ :=
  P_X + P_Y + P_Z

theorem car_winning_probability :
  let P_X := (1 : ℚ) / 6
  let P_Y := (1 : ℚ) / 10
  let P_Z := (1 : ℚ) / 8
  probability_of_winning P_X P_Y P_Z = 47 / 120 :=
by
  sorry

end car_winning_probability_l240_240736


namespace regular_polygon_sides_l240_240917

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240917


namespace find_q_l240_240445

noncomputable def solution_condition (p q : ℝ) : Prop :=
  (p > 1) ∧ (q > 1) ∧ (1 / p + 1 / q = 1) ∧ (p * q = 9)

theorem find_q (p q : ℝ) (h : solution_condition p q) : 
  q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l240_240445


namespace simplify_expression_l240_240481

theorem simplify_expression (n : ℕ) : 
  (3^(n + 3) - 3 * 3^n) / (3 * 3^(n + 2)) = 8 / 3 := 
sorry

end simplify_expression_l240_240481


namespace gcd_fx_x_l240_240169

noncomputable def f (x : ℕ) : ℕ := (5 * x + 3) * (8 * x + 2) * (12 * x + 7) * (3 * x + 11)

theorem gcd_fx_x (x : ℕ) (h : ∃ k : ℕ, x = 18720 * k) : Nat.gcd (f x) x = 462 :=
sorry

end gcd_fx_x_l240_240169


namespace factory_production_eq_l240_240671

theorem factory_production_eq (x : ℝ) (h1 : x > 50) : 450 / (x - 50) - 400 / x = 1 := 
by 
  sorry

end factory_production_eq_l240_240671


namespace probability_white_ball_l240_240456

theorem probability_white_ball :
  let total_balls := 3 + 2 + 1 in
  let white_balls := 2 in
  (white_balls / total_balls : ℚ) = 1 / 3 :=
by
  let total_balls := 3 + 2 + 1
  let white_balls := 2
  have h1 : total_balls = 6 := by rfl
  have h2 : white_balls = 2 := by rfl
  calc
    (2 / 6 : ℚ) = 1 / 3 : by norm_num

end probability_white_ball_l240_240456


namespace youngest_is_dan_l240_240772

notation "alice" => 21
notation "bob" => 18
notation "clare" => 22
notation "dan" => 16
notation "eve" => 28

theorem youngest_is_dan :
  let a := alice
  let b := bob
  let c := clare
  let d := dan
  let e := eve
  a + b = 39 ∧
  b + c = 40 ∧
  c + d = 38 ∧
  d + e = 44 ∧
  a + b + c + d + e = 105 →
  min (min (min (min a b) c) d) e = d :=
by {
  sorry
}

end youngest_is_dan_l240_240772


namespace product_ab_l240_240435

noncomputable def median_of_four_numbers (a b : ℕ) := 3
noncomputable def mean_of_four_numbers (a b : ℕ) := 4

theorem product_ab (a b : ℕ)
  (h1 : 1 + 2 + a + b = 4 * 4)
  (h2 : median_of_four_numbers a b = 3)
  (h3 : mean_of_four_numbers a b = 4) : (a * b = 36) :=
by sorry

end product_ab_l240_240435


namespace infinitely_many_not_2a_3b_5c_l240_240763

theorem infinitely_many_not_2a_3b_5c : ∃ᶠ x : ℤ in Filter.cofinite, ∀ a b c : ℕ, x % 120 ≠ (2^a + 3^b - 5^c) % 120 :=
by
  sorry

end infinitely_many_not_2a_3b_5c_l240_240763


namespace subset_property_l240_240514

theorem subset_property : {2} ⊆ {x | x ≤ 10} := 
by 
  sorry

end subset_property_l240_240514


namespace range_of_x_l240_240749

noncomputable def T (x : ℝ) : ℝ := |(2 * x - 1)|

theorem range_of_x (x : ℝ) (h : ∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) : 
  x ≤ -1 ∨ 2 ≤ x :=
by
  sorry

end range_of_x_l240_240749


namespace soccer_league_games_l240_240243

theorem soccer_league_games : 
  (∃ n : ℕ, n = 12) → 
  (∀ i j : ℕ, i ≠ j → i < 12 → j < 12 → (games_played i j = 4)) → 
  total_games_played = 264 :=
by
  sorry

end soccer_league_games_l240_240243


namespace find_value_of_expression_l240_240394

variable (p q r s : ℝ)

def g (x : ℝ) : ℝ := p * x ^ 3 + q * x ^ 2 + r * x + s

-- We state the condition that g(1) = 1
axiom g_at_one : g p q r s 1 = 1

-- Now, we state the problem we need to prove:
theorem find_value_of_expression : 5 * p - 3 * q + 2 * r - s = 5 :=
by
  -- We skip the proof here
  exact sorry

end find_value_of_expression_l240_240394


namespace complementary_angle_difference_l240_240779

theorem complementary_angle_difference (x : ℝ) (h : 3 * x + 5 * x = 90) : 
    abs ((5 * x) - (3 * x)) = 22.5 :=
by
  -- placeholder proof
  sorry

end complementary_angle_difference_l240_240779


namespace find_number_l240_240982

theorem find_number (x : ℝ) (h : x - (3/5) * x = 50) : x = 125 := by
  sorry

end find_number_l240_240982


namespace boxes_containing_neither_l240_240060

-- Define the conditions
def total_boxes : ℕ := 15
def boxes_with_pencils : ℕ := 8
def boxes_with_pens : ℕ := 5
def boxes_with_markers : ℕ := 3
def boxes_with_pencils_and_pens : ℕ := 2
def boxes_with_pencils_and_markers : ℕ := 1
def boxes_with_pens_and_markers : ℕ := 1
def boxes_with_all_three : ℕ := 0

-- The proof problem
theorem boxes_containing_neither (h: total_boxes = 15) : 
  total_boxes - ((boxes_with_pencils - boxes_with_pencils_and_pens - boxes_with_pencils_and_markers) + 
  (boxes_with_pens - boxes_with_pencils_and_pens - boxes_with_pens_and_markers) + 
  (boxes_with_markers - boxes_with_pencils_and_markers - boxes_with_pens_and_markers) + 
  boxes_with_pencils_and_pens + boxes_with_pencils_and_markers + boxes_with_pens_and_markers) = 3 := 
by
  -- Specify that we want to use the equality of the number of boxes
  sorry

end boxes_containing_neither_l240_240060


namespace frac_multiplication_l240_240252

theorem frac_multiplication : 
    ((2/3:ℚ)^4 * (1/5) * (3/4) = 4/135) :=
by
  sorry

end frac_multiplication_l240_240252


namespace regular_polygon_sides_l240_240840

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240840


namespace inverse_proportion_inequality_l240_240995

theorem inverse_proportion_inequality {x1 x2 : ℝ} (h1 : x1 > x2) (h2 : x2 > 0) : 
    -3 / x1 > -3 / x2 := 
by 
  sorry

end inverse_proportion_inequality_l240_240995


namespace no_four_points_with_all_odd_distances_l240_240055

theorem no_four_points_with_all_odd_distances :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (x y z p q r : ℕ),
      (x = dist A B ∧ x % 2 = 1) ∧
      (y = dist B C ∧ y % 2 = 1) ∧
      (z = dist C D ∧ z % 2 = 1) ∧
      (p = dist D A ∧ p % 2 = 1) ∧
      (q = dist A C ∧ q % 2 = 1) ∧
      (r = dist B D ∧ r % 2 = 1))
    → false :=
by
  sorry

end no_four_points_with_all_odd_distances_l240_240055


namespace area_of_WXYZ_l240_240051

structure Quadrilateral (α : Type _) :=
  (W : α) (X : α) (Y : α) (Z : α)
  (WZ ZW' WX XX' XY YY' YZ Z'W : ℝ)
  (area_WXYZ : ℝ)

theorem area_of_WXYZ' (WXYZ : Quadrilateral ℝ) 
  (h1 : WXYZ.WZ = 10) 
  (h2 : WXYZ.ZW' = 10)
  (h3 : WXYZ.WX = 6)
  (h4 : WXYZ.XX' = 6)
  (h5 : WXYZ.XY = 7)
  (h6 : WXYZ.YY' = 7)
  (h7 : WXYZ.YZ = 12)
  (h8 : WXYZ.Z'W = 12)
  (h9 : WXYZ.area_WXYZ = 15) : 
  ∃ area_WXZY' : ℝ, area_WXZY' = 45 :=
sorry

end area_of_WXYZ_l240_240051


namespace mathematicians_correctness_l240_240334

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  ¬ (3 / 8 < 17 / 40 ∧ 17 / 40 < 2 / 5) :=
by {
  sorry
}

end mathematicians_correctness_l240_240334


namespace regular_polygon_sides_l240_240850

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240850


namespace aniyah_more_candles_l240_240403

theorem aniyah_more_candles (x : ℝ) (h1 : 4 + 4 * x = 14) : x = 2.5 :=
sorry

end aniyah_more_candles_l240_240403


namespace midpoint_lattice_point_exists_l240_240063

theorem midpoint_lattice_point_exists (S : Finset (ℤ × ℤ)) (hS : S.card = 5) :
  ∃ (p1 p2 : ℤ × ℤ), p1 ∈ S ∧ p2 ∈ S ∧ p1 ≠ p2 ∧
  (∃ (x_mid y_mid : ℤ), 
    (p1.1 + p2.1) = 2 * x_mid ∧
    (p1.2 + p2.2) = 2 * y_mid) :=
by
  sorry

end midpoint_lattice_point_exists_l240_240063


namespace convert_deg_to_rad_l240_240969

theorem convert_deg_to_rad (deg : ℝ) (π : ℝ) (h : deg = 50) : (deg * (π / 180) = 5 / 18 * π) :=
by
  -- Conditions
  sorry

end convert_deg_to_rad_l240_240969


namespace partial_fraction_decomposition_l240_240698

theorem partial_fraction_decomposition :
  ∃ A B C : ℚ, (∀ x : ℚ, x ≠ -1 ∧ x^2 - x + 2 ≠ 0 →
          (x^2 + 2 * x - 8) / (x^3 - x - 2) = A / (x + 1) + (B * x + C) / (x^2 - x + 2)) ∧
          A = -9/4 ∧ B = 13/4 ∧ C = -7/2 :=
sorry

end partial_fraction_decomposition_l240_240698


namespace tangent_points_are_on_locus_l240_240609

noncomputable def tangent_points_locus (d : ℝ) : Prop :=
∀ (x y : ℝ), 
((x ≠ 0 ∨ y ≠ 0) ∧ (x-d ≠ 0)) ∧ (y = x) 
→ (y^2 - x*y + d*(x + y) = 0)

theorem tangent_points_are_on_locus (d : ℝ) : 
  tangent_points_locus d :=
by sorry

end tangent_points_are_on_locus_l240_240609


namespace regular_polygon_sides_l240_240876

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240876


namespace expected_number_of_hits_l240_240741

variable (W : ℝ) (n : ℕ)
def expected_hits (W : ℝ) (n : ℕ) : ℝ := W * n

theorem expected_number_of_hits :
  W = 0.75 → n = 40 → expected_hits W n = 30 :=
by
  intros hW hn
  rw [hW, hn]
  norm_num
  sorry

end expected_number_of_hits_l240_240741


namespace mean_eq_value_of_z_l240_240341

theorem mean_eq_value_of_z (z : ℤ) : 
  ((6 + 15 + 9 + 20) / 4 : ℚ) = ((13 + z) / 2 : ℚ) → (z = 12) := by
  sorry

end mean_eq_value_of_z_l240_240341


namespace regular_polygon_sides_l240_240895

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240895


namespace contrapositive_correct_l240_240081

-- Define the main condition: If a ≥ 1/2, then ∀ x ≥ 0, f(x) ≥ 0
def main_condition (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≥ 1/2 → ∀ x : ℝ, x ≥ 0 → f x ≥ 0

-- Define the contrapositive statement: If ∃ x ≥ 0 such that f(x) < 0, then a < 1/2
def contrapositive (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, x ≥ 0 ∧ f x < 0) → a < 1/2

-- Theorem to prove that the contrapositive statement is correct
theorem contrapositive_correct (a : ℝ) (f : ℝ → ℝ) :
  main_condition a f ↔ contrapositive a f :=
by
  sorry

end contrapositive_correct_l240_240081


namespace exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l240_240556

def omega (n : Nat) : Nat :=
  if n = 1 then 0 else n.factors.toFinset.card

theorem exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2 :
  ∃ᶠ n in atTop, ∃ k : Nat, n = 2^k ∧
    omega n < omega (n + 1) ∧
    omega (n + 1) < omega (n + 2) :=
sorry

end exists_infinite_n_ωn_less_ωn_add1_less_ωn_add2_l240_240556


namespace initial_students_count_l240_240488

theorem initial_students_count (n : ℕ) (W : ℝ) 
  (h1 : W = n * 28) 
  (h2 : W + 1 = (n + 1) * 27.1) : 
  n = 29 := by
  sorry

end initial_students_count_l240_240488


namespace value_of_f_g_3_l240_240289

def g (x : ℝ) : ℝ := x^3
def f (x : ℝ) : ℝ := 3*x + 2

theorem value_of_f_g_3 : f (g 3) = 83 := by
  sorry

end value_of_f_g_3_l240_240289


namespace students_play_at_least_one_sport_l240_240606

def B := 12
def C := 10
def S := 9
def Ba := 6

def B_and_C := 5
def B_and_S := 4
def B_and_Ba := 3
def C_and_S := 2
def C_and_Ba := 3
def S_and_Ba := 2

def B_and_C_and_S_and_Ba := 1

theorem students_play_at_least_one_sport : 
  B + C + S + Ba - B_and_C - B_and_S - B_and_Ba - C_and_S - C_and_Ba - S_and_Ba + B_and_C_and_S_and_Ba = 19 :=
by
  sorry

end students_play_at_least_one_sport_l240_240606


namespace minimum_box_value_l240_240595

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l240_240595


namespace total_dogs_equation_l240_240971

/-- Definition of the number of boxes and number of dogs per box. --/
def num_boxes : ℕ := 7
def dogs_per_box : ℕ := 4

/-- The total number of dogs --/
theorem total_dogs_equation : num_boxes * dogs_per_box = 28 := by 
  sorry

end total_dogs_equation_l240_240971


namespace no_real_solution_condition_l240_240568

def no_real_solution (k : ℝ) : Prop :=
  let discriminant := 25 + 4 * k
  discriminant < 0

theorem no_real_solution_condition (k : ℝ) : no_real_solution k ↔ k < -25 / 4 := 
sorry

end no_real_solution_condition_l240_240568


namespace sin_nine_pi_over_two_plus_theta_l240_240440

variable (θ : ℝ)

-- Conditions: Point A(4, -3) lies on the terminal side of angle θ
def terminal_point_on_angle (θ : ℝ) : Prop :=
  let x := 4
  let y := -3
  let hypotenuse := Real.sqrt ((x ^ 2) + (y ^ 2))
  hypotenuse = 5 ∧ Real.cos θ = x / hypotenuse

theorem sin_nine_pi_over_two_plus_theta (θ : ℝ) 
  (h : terminal_point_on_angle θ) : 
  Real.sin (9 * Real.pi / 2 + θ) = 4 / 5 :=
sorry

end sin_nine_pi_over_two_plus_theta_l240_240440


namespace range_of_a_l240_240043

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 :=
by
  sorry

end range_of_a_l240_240043


namespace lance_pennies_saved_l240_240305

theorem lance_pennies_saved :
  let a := 5
  let d := 2
  let n := 20
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n = 480 :=
by
  sorry

end lance_pennies_saved_l240_240305


namespace solve_system_l240_240553

theorem solve_system :
  ∀ x y : ℚ, (3 * x + 4 * y = 12) ∧ (9 * x - 12 * y = -24) →
  (x = 2 / 3) ∧ (y = 5 / 2) :=
by
  intro x y
  intro h
  sorry

end solve_system_l240_240553


namespace smallest_number_l240_240526

theorem smallest_number (x : ℕ) (h1 : 2 * x = third) (h2 : 4 * x = second) (h3 : 7 * x = fourth) (h4 : (x + second + third + fourth) / 4 = 77) :
  x = 22 :=
by sorry

end smallest_number_l240_240526


namespace BBB_div_by_9_l240_240087

open Nat

theorem BBB_div_by_9 (B : ℕ) (h1 : 4 * 10^4 + B * 10^3 + B * 10^2 + B * 10 + 2 ≡ 0 [MOD 9]) (h2 : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  B = 4 :=
by
  have mod9_eq : (4 * 10^4 + (B + B + B) * 10^2 + 2) ≡ (4 + B + B + B + 2) [MOD 9] := Nat.mod_eq_of_lt
  sorry

end BBB_div_by_9_l240_240087


namespace vote_difference_60_l240_240515

theorem vote_difference_60
    (total_members : ℕ)
    (x y x' y' m : ℕ)
    (total_eq : x + y = total_members)
    (initial_defeat : y > x)
    (margin_defeat : y - x = m)
    (revote_pass_margin : x' - y' = 2 * m)
    (revote_total_eq : x' + y' = total_members)
    (revote_ratio : x' = 12 * y / 11) :
    x' - x = 60 :=
by
  sorry

end vote_difference_60_l240_240515


namespace max_10a_3b_15c_l240_240308

theorem max_10a_3b_15c (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) : 
  10 * a + 3 * b + 15 * c ≤ (Real.sqrt 337) / 6 := 
sorry

end max_10a_3b_15c_l240_240308


namespace find_b_value_l240_240583

theorem find_b_value (a b : ℤ) (h₁ : a + 2 * b = 32) (h₂ : |a| > 2) (h₃ : a = 4) : b = 14 :=
by
  -- proof goes here
  sorry

end find_b_value_l240_240583


namespace word_value_at_l240_240419

def letter_value (c : Char) : ℕ :=
  if 'A' ≤ c ∧ c ≤ 'Z' then c.toNat - 'A'.toNat + 1 else 0

def word_value (s : String) : ℕ :=
  let sum_values := s.toList.map letter_value |>.sum
  sum_values * s.length

theorem word_value_at : word_value "at" = 42 := by
  sorry

end word_value_at_l240_240419


namespace quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l240_240005

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Problem 1: Prove that the quadratic function passes through the origin for m = 1 or m = -2
theorem quadratic_passes_through_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -2) ∧ quadratic m 0 = 0 := by
  sorry

-- Problem 2: Prove that the quadratic function is symmetric about the y-axis for m = 0
theorem quadratic_symmetric_about_y_axis :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, quadratic m x = quadratic m (-x) := by
  sorry

end quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l240_240005


namespace intersection_domains_l240_240002

def domain_f : Set ℝ := {x : ℝ | x < 1}
def domain_g : Set ℝ := {x : ℝ | x > -1}

theorem intersection_domains : {x : ℝ | x < 1} ∩ {x : ℝ | x > -1} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end intersection_domains_l240_240002


namespace circle_equation_of_tangent_circle_l240_240580

theorem circle_equation_of_tangent_circle
  (h : ∀ x y: ℝ, x^2/4 - y^2 = 1 → (x = 2 ∨ x = -2) → y = 0)
  (asymptote : ∀ x y : ℝ, (y = (1/2)*x ∨ y = -(1/2)*x) → (x - 2)^2 + y^2 = (4/5))
  : ∃ k : ℝ, (∀ x y : ℝ, (x - 2)^2 + y^2 = k) → k = 4/5 := by
  sorry

end circle_equation_of_tangent_circle_l240_240580


namespace trigonometric_identities_l240_240576

theorem trigonometric_identities
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (sinα : Real.sin α = 4 / 5)
  (cosβ : Real.cos β = 12 / 13) :
  Real.sin (α + β) = 63 / 65 ∧ Real.tan (α - β) = 33 / 56 := by
  sorry

end trigonometric_identities_l240_240576


namespace cheerleaders_uniforms_l240_240206

theorem cheerleaders_uniforms (total_cheerleaders : ℕ) (size_6_cheerleaders : ℕ) (half_size_6_cheerleaders : ℕ) (size_2_cheerleaders : ℕ) : 
  total_cheerleaders = 19 →
  size_6_cheerleaders = 10 →
  half_size_6_cheerleaders = size_6_cheerleaders / 2 →
  size_2_cheerleaders = total_cheerleaders - (size_6_cheerleaders + half_size_6_cheerleaders) →
  size_2_cheerleaders = 4 :=
by
  intros
  sorry

end cheerleaders_uniforms_l240_240206


namespace john_blue_pens_l240_240058

variables (R B Bl : ℕ)

axiom total_pens : R + B + Bl = 31
axiom black_more_red : B = R + 5
axiom blue_twice_black : Bl = 2 * B

theorem john_blue_pens : Bl = 18 :=
by
  apply sorry

end john_blue_pens_l240_240058


namespace regular_polygon_sides_l240_240891

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240891


namespace speeds_of_cars_l240_240506

theorem speeds_of_cars (d_A d_B : ℝ) (v_A v_B : ℝ) (h1 : d_A = 300) (h2 : d_B = 250) (h3 : v_A = v_B + 5) (h4 : d_A / v_A = d_B / v_B) :
  v_B = 25 ∧ v_A = 30 :=
by
  sorry

end speeds_of_cars_l240_240506


namespace probability_within_range_l240_240299

noncomputable def normal_dist := measure_theory.probability_distribution.normal 30 0.7

theorem probability_within_range (X : ℝ → ℝ) (hX : ∀ x, normal_dist.density x = X x) :
  measure_theory.probability ((λ x, 28 < x ∧ x < 31) X) = 0.9215 :=
sorry

end probability_within_range_l240_240299


namespace volume_of_blue_tetrahedron_in_cube_l240_240235

theorem volume_of_blue_tetrahedron_in_cube (side_length : ℝ) (h : side_length = 8) :
  let cube_volume := side_length^3
  let tetrahedra_volume := 4 * (1/3 * (1/2 * side_length * side_length) * side_length)
  cube_volume - tetrahedra_volume = 512/3 :=
by
  sorry

end volume_of_blue_tetrahedron_in_cube_l240_240235


namespace sum_of_three_numbers_l240_240212

theorem sum_of_three_numbers (a b c : ℕ) (h1 : b = 10)
                            (h2 : (a + b + c) / 3 = a + 15)
                            (h3 : (a + b + c) / 3 = c - 25) :
                            a + b + c = 60 :=
sorry

end sum_of_three_numbers_l240_240212


namespace smallest_n_divides_l240_240625

theorem smallest_n_divides (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  ∃ n : ℕ, 2^(1988) = n ∧ 2^1989 ∣ m^n - 1 :=
by
  sorry

end smallest_n_divides_l240_240625


namespace regular_polygon_sides_l240_240843

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240843


namespace graph_does_not_pass_through_third_quadrant_l240_240667

theorem graph_does_not_pass_through_third_quadrant (k x y : ℝ) (hk : k < 0) :
  y = k * x - k → (¬ (x < 0 ∧ y < 0)) :=
by
  sorry

end graph_does_not_pass_through_third_quadrant_l240_240667


namespace travel_time_K_l240_240229

theorem travel_time_K (d x : ℝ) (h_pos_d : d > 0) (h_x_pos : x > 0) (h_time_diff : (d / (x - 1/2)) - (d / x) = 1/2) : d / x = 40 / x :=
by
  sorry

end travel_time_K_l240_240229


namespace find_a_l240_240192

def f(x : ℚ) : ℚ := x / 3 + 2
def g(x : ℚ) : ℚ := 5 - 2 * x

theorem find_a (a : ℚ) (h : f (g a) = 4) : a = -1 / 2 :=
by
  sorry

end find_a_l240_240192


namespace common_solutions_y_values_l240_240369

theorem common_solutions_y_values :
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by {
  sorry
}

end common_solutions_y_values_l240_240369


namespace time_to_pass_platform_l240_240669

-- Conditions of the problem
def length_of_train : ℕ := 1500
def time_to_cross_tree : ℕ := 100
def length_of_platform : ℕ := 500

-- Derived values according to solution steps
def speed_of_train : ℚ := length_of_train / time_to_cross_tree
def total_distance_to_pass_platform : ℕ := length_of_train + length_of_platform

-- The theorem to be proved
theorem time_to_pass_platform :
  (total_distance_to_pass_platform / speed_of_train : ℚ) = 133.33 := sorry

end time_to_pass_platform_l240_240669


namespace boston_trip_distance_l240_240374

theorem boston_trip_distance :
  ∃ d : ℕ, 40 * d = 440 :=
by
  sorry

end boston_trip_distance_l240_240374


namespace BBB_div_by_9_l240_240086

open Nat

theorem BBB_div_by_9 (B : ℕ) (h1 : 4 * 10^4 + B * 10^3 + B * 10^2 + B * 10 + 2 ≡ 0 [MOD 9]) (h2 : B ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  B = 4 :=
by
  have mod9_eq : (4 * 10^4 + (B + B + B) * 10^2 + 2) ≡ (4 + B + B + B + 2) [MOD 9] := Nat.mod_eq_of_lt
  sorry

end BBB_div_by_9_l240_240086


namespace problem_statement_l240_240533

theorem problem_statement : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end problem_statement_l240_240533


namespace tax_paid_at_fifth_checkpoint_l240_240742

variable {x : ℚ}

theorem tax_paid_at_fifth_checkpoint (x : ℚ) (h : (x / 2) + (x / 2 * 1 / 3) + (x / 3 * 1 / 4) + (x / 4 * 1 / 5) + (x / 5 * 1 / 6) = 1) :
  (x / 5 * 1 / 6) = 1 / 25 :=
sorry

end tax_paid_at_fifth_checkpoint_l240_240742


namespace evaluate_expression_l240_240562

theorem evaluate_expression :
  8^6 * 27^6 * 8^15 * 27^15 = 216^21 :=
by
  sorry

end evaluate_expression_l240_240562


namespace regular_polygon_sides_l240_240959

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240959


namespace geometric_sequence_sum_l240_240000

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : (a 1) = 1
axiom a2 : ∀ (n : ℕ), n ≥ 2 → 2 * a (n + 1) + 2 * a (n - 1) = 5 * a n
axiom increasing : ∀ (n m : ℕ), n < m → a n < a m

-- Target
theorem geometric_sequence_sum : S 5 = 31 := by
  sorry

end geometric_sequence_sum_l240_240000


namespace find_grazing_months_l240_240226

def oxen_months_A := 10 * 7
def oxen_months_B := 12 * 5
def total_rent := 175
def rent_C := 45

def proportion_equation (x : ℕ) : Prop :=
  45 / 175 = (15 * x) / (oxen_months_A + oxen_months_B + 15 * x)

theorem find_grazing_months (x : ℕ) (h : proportion_equation x) : x = 3 :=
by
  -- We will need to involve some calculations leading to x = 3
  sorry

end find_grazing_months_l240_240226


namespace sin_and_tan_alpha_l240_240581

variable (x : ℝ) (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_is_non_negative_half_axis : Prop := true
def terminal_side_passes_through_P : Prop := ∃ (P : ℝ × ℝ), P = (x, -Real.sqrt 2)
def cos_alpha_eq : Prop := x ≠ 0 ∧ Real.cos α = (Real.sqrt 3 / 6) * x

-- Proof Problem Statement
theorem sin_and_tan_alpha (h1 : vertex_is_origin) 
                         (h2 : initial_side_is_non_negative_half_axis) 
                         (h3 : terminal_side_passes_through_P x) 
                         (h4 : cos_alpha_eq x α) 
                         : Real.sin α = -Real.sqrt 6 / 6 ∧ (Real.tan α = Real.sqrt 5 / 5 ∨ Real.tan α = -Real.sqrt 5 / 5) := 
sorry

end sin_and_tan_alpha_l240_240581


namespace inversely_proportional_y_ratio_l240_240484

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l240_240484


namespace age_of_youngest_l240_240095

theorem age_of_youngest
  (y : ℕ)
  (h1 : 4 * 25 = y + (y + 2) + (y + 7) + (y + 11)) : y = 20 :=
by
  sorry

end age_of_youngest_l240_240095


namespace original_price_of_shoes_l240_240133

theorem original_price_of_shoes (P : ℝ) (h : 0.08 * P = 16) : P = 200 :=
sorry

end original_price_of_shoes_l240_240133


namespace calculate_kevin_training_time_l240_240459

theorem calculate_kevin_training_time : 
  ∀ (laps : ℕ) 
    (track_length : ℕ) 
    (run1_distance : ℕ) 
    (run1_speed : ℕ) 
    (walk_distance : ℕ) 
    (walk_speed : Real) 
    (run2_distance : ℕ) 
    (run2_speed : ℕ) 
    (minutes : ℕ) 
    (seconds : Real),
    laps = 8 →
    track_length = 500 →
    run1_distance = 200 →
    run1_speed = 3 →
    walk_distance = 100 →
    walk_speed = 1.5 →
    run2_distance = 200 →
    run2_speed = 4 →
    minutes = 24 →
    seconds = 27 →
    (∀ (t1 t2 t3 t_total t_training : Real),
      t1 = run1_distance / run1_speed →
      t2 = walk_distance / walk_speed →
      t3 = run2_distance / run2_speed →
      t_total = t1 + t2 + t3 →
      t_training = laps * t_total →
      t_training = (minutes * 60 + seconds)) := 
by
  intros laps track_length run1_distance run1_speed walk_distance walk_speed run2_distance run2_speed minutes seconds
  intros h_laps h_track_length h_run1_distance h_run1_speed h_walk_distance h_walk_speed h_run2_distance h_run2_speed h_minutes h_seconds
  intros t1 t2 t3 t_total t_training
  intros h_t1 h_t2 h_t3 h_t_total h_t_training
  sorry

end calculate_kevin_training_time_l240_240459


namespace total_fruit_cost_is_173_l240_240311

-- Define the cost of a single orange and a single apple
def orange_cost := 2
def apple_cost := 3
def banana_cost := 1

-- Define the number of fruits each person has
def louis_oranges := 5
def louis_apples := 3

def samantha_oranges := 8
def samantha_apples := 7

def marley_oranges := 2 * louis_oranges
def marley_apples := 3 * samantha_apples

def edward_oranges := 3 * louis_oranges
def edward_bananas := 4

-- Define the cost of fruits for each person
def louis_cost := (louis_oranges * orange_cost) + (louis_apples * apple_cost)
def samantha_cost := (samantha_oranges * orange_cost) + (samantha_apples * apple_cost)
def marley_cost := (marley_oranges * orange_cost) + (marley_apples * apple_cost)
def edward_cost := (edward_oranges * orange_cost) + (edward_bananas * banana_cost)

-- Define the total cost for all four people
def total_cost := louis_cost + samantha_cost + marley_cost + edward_cost

-- Statement to prove that the total cost is $173
theorem total_fruit_cost_is_173 : total_cost = 173 :=
by
  sorry

end total_fruit_cost_is_173_l240_240311


namespace sum_of_integer_pair_l240_240814

theorem sum_of_integer_pair (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 10) (h3 : 1 ≤ b) (h4 : b ≤ 10) (h5 : a * b = 14) : a + b = 9 := 
sorry

end sum_of_integer_pair_l240_240814


namespace b_catches_a_distance_l240_240115

-- Define the initial conditions
def a_speed : ℝ := 10  -- A's speed in km/h
def b_speed : ℝ := 20  -- B's speed in km/h
def start_delay : ℝ := 3  -- B starts cycling 3 hours after A in hours

-- Define the target distance to prove
theorem b_catches_a_distance : ∃ (d : ℝ), d = 60 := 
by 
  sorry

end b_catches_a_distance_l240_240115


namespace candies_remaining_l240_240816

theorem candies_remaining 
    (red_candies : ℕ)
    (yellow_candies : ℕ)
    (blue_candies : ℕ)
    (yellow_condition : yellow_candies = 3 * red_candies - 20)
    (blue_condition : blue_candies = yellow_candies / 2)
    (initial_red_candies : red_candies = 40) :
    (red_candies + yellow_candies + blue_candies - yellow_candies) = 90 := 
by
  sorry

end candies_remaining_l240_240816


namespace books_on_each_shelf_l240_240551

-- Define the conditions and the problem statement
theorem books_on_each_shelf :
  ∀ (M P : ℕ), 
  -- Conditions
  (5 * M + 4 * P = 72) ∧ (M = P) ∧ (∃ B : ℕ, M = B ∧ P = B) ->
  -- Conclusion
  (∃ B : ℕ, B = 8) :=
by
  sorry

end books_on_each_shelf_l240_240551


namespace number_whose_multiples_are_considered_for_calculating_the_average_l240_240524

theorem number_whose_multiples_are_considered_for_calculating_the_average
  (x : ℕ)
  (n : ℕ)
  (a : ℕ)
  (b : ℕ)
  (h1 : n = 10)
  (h2 : a = (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7)
  (h3 : b = 2*n)
  (h4 : a^2 - b^2 = 0) :
  x = 5 := 
sorry

end number_whose_multiples_are_considered_for_calculating_the_average_l240_240524


namespace smallest_divisor_of_7614_l240_240496

theorem smallest_divisor_of_7614 (h : Nat) (H_h_eq : h = 1) (n : Nat) (H_n_eq : n = (7600 + 10 * h + 4)) :
  ∃ d, d > 1 ∧ d ∣ n ∧ ∀ x, x > 1 ∧ x ∣ n → d ≤ x :=
by
  sorry

end smallest_divisor_of_7614_l240_240496


namespace positive_divisors_8_factorial_l240_240026

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l240_240026


namespace marked_price_is_300_max_discount_is_50_l240_240544

-- Definition of the conditions given in the problem:
def loss_condition (x : ℝ) : Prop := 0.4 * x - 30 = 0.7 * x - 60
def profit_condition (x : ℝ) : Prop := 0.7 * x - 60 - (0.4 * x - 30) = 90

-- Statement for the first problem: Prove the marked price is 300 yuan.
theorem marked_price_is_300 : ∃ x : ℝ, loss_condition x ∧ profit_condition x ∧ x = 300 := by
  exists 300
  simp [loss_condition, profit_condition]
  sorry

noncomputable def max_discount (x : ℝ) : ℝ := 100 - (30 + 0.4 * x) / x * 100

def no_loss_max_discount (d : ℝ) : Prop := d = 50

-- Statement for the second problem: Prove the maximum discount is 50%.
theorem max_discount_is_50 (x : ℝ) (h_loss : loss_condition x) (h_profit : profit_condition x) : no_loss_max_discount (max_discount x) := by
  simp [max_discount, no_loss_max_discount]
  sorry

end marked_price_is_300_max_discount_is_50_l240_240544


namespace proportion_of_bike_riders_is_correct_l240_240502

-- Define the given conditions as constants
def total_students : ℕ := 92
def bus_riders : ℕ := 20
def walkers : ℕ := 27

-- Define the remaining students after bus riders and after walkers
def remaining_after_bus_riders : ℕ := total_students - bus_riders
def bike_riders : ℕ := remaining_after_bus_riders - walkers

-- Define the expected proportion
def expected_proportion : ℚ := 45 / 72

-- State the theorem to be proved
theorem proportion_of_bike_riders_is_correct :
  (↑bike_riders / ↑remaining_after_bus_riders : ℚ) = expected_proportion := 
by
  sorry

end proportion_of_bike_riders_is_correct_l240_240502


namespace inequality_ab5_bc5_ca5_l240_240635

theorem inequality_ab5_bc5_ca5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) :=
sorry

end inequality_ab5_bc5_ca5_l240_240635


namespace eliza_ironing_hours_l240_240416

theorem eliza_ironing_hours (h : ℕ) 
  (blouse_minutes : ℕ := 15) 
  (dress_minutes : ℕ := 20) 
  (hours_ironing_blouses : ℕ := h)
  (hours_ironing_dresses : ℕ := 3)
  (total_clothes : ℕ := 17) :
  ((60 / blouse_minutes) * hours_ironing_blouses) + ((60 / dress_minutes) * hours_ironing_dresses) = total_clothes →
  hours_ironing_blouses = 2 := 
sorry

end eliza_ironing_hours_l240_240416


namespace tourists_escape_l240_240770

theorem tourists_escape (T : ℕ) (hT : T = 10) (hats : Fin T → Bool) (could_see : ∀ (i : Fin T), Fin (i) → Bool) :
  ∃ strategy : (Fin T → Bool), (∀ (i : Fin T), (strategy i = hats i) ∨ (strategy i ≠ hats i)) →
  (∀ (i : Fin T), (∀ (j : Fin T), i < j → strategy i = hats i) → ∃ count : ℕ, count ≥ 9 ∧ ∀ (i : Fin T), count ≥ i → strategy i = hats i) := sorry

end tourists_escape_l240_240770


namespace total_fruit_salads_is_1800_l240_240402

def Alaya_fruit_salads := 200
def Angel_fruit_salads := 2 * Alaya_fruit_salads
def Betty_fruit_salads := 3 * Angel_fruit_salads
def Total_fruit_salads := Alaya_fruit_salads + Angel_fruit_salads + Betty_fruit_salads

theorem total_fruit_salads_is_1800 : Total_fruit_salads = 1800 := by
  sorry

end total_fruit_salads_is_1800_l240_240402


namespace cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l240_240974

-- Definition of size-n tromino
def tromino_area (n : ℕ) := (4 * 4 * n - 1)

-- Problem (a): Can a size-5 tromino be tiled by size-1 trominos
theorem cannot_tile_size5_with_size1_trominos :
  ¬ (∃ (count : ℕ), count * 3 = tromino_area 5) :=
by sorry

-- Problem (b): Can a size-2013 tromino be tiled by size-1 trominos
theorem can_tile_size2013_with_size1_trominos :
  ∃ (count : ℕ), count * 3 = tromino_area 2013 :=
by sorry

end cannot_tile_size5_with_size1_trominos_can_tile_size2013_with_size1_trominos_l240_240974


namespace eccentricity_of_ellipse_l240_240713

theorem eccentricity_of_ellipse (a c : ℝ) (h : 4 * a = 7 * 2 * (a - c)) : 
    c / a = 5 / 7 :=
by {
  sorry
}

end eccentricity_of_ellipse_l240_240713


namespace regular_polygon_sides_l240_240874

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240874


namespace regular_polygon_sides_l240_240938

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240938


namespace price_of_other_stamp_l240_240541

-- Define the conditions
def total_stamps : ℕ := 75
def total_value_cents : ℕ := 480
def known_stamp_price : ℕ := 8
def known_stamp_count : ℕ := 40
def unknown_stamp_count : ℕ := total_stamps - known_stamp_count

-- The problem to solve
theorem price_of_other_stamp (x : ℕ) :
  (known_stamp_count * known_stamp_price) + (unknown_stamp_count * x) = total_value_cents → x = 5 :=
by
  sorry

end price_of_other_stamp_l240_240541


namespace area_of_tangency_triangle_l240_240146

noncomputable def area_of_triangle : ℝ :=
  let r1 := 2
  let r2 := 3
  let r3 := 4
  let s := (r1 + r2 + r3) / 2
  let A := Real.sqrt (s * (s - (r1 + r2)) * (s - (r2 + r3)) * (s - (r1 + r3)))
  let inradius := A / s
  let area_points_of_tangency := A * (inradius / r1) * (inradius / r2) * (inradius / r3)
  area_points_of_tangency

theorem area_of_tangency_triangle :
  area_of_triangle = (16 * Real.sqrt 6) / 3 :=
sorry

end area_of_tangency_triangle_l240_240146


namespace solution_set_of_inequality_l240_240414

theorem solution_set_of_inequality : {x : ℝ | 8 * x^2 + 6 * x ≤ 2} = { x : ℝ | -1 ≤ x ∧ x ≤ (1/4) } :=
sorry

end solution_set_of_inequality_l240_240414


namespace find_last_four_digits_of_N_l240_240123

def P (n : Nat) : Nat :=
  match n with
  | 0     => 1 -- usually not needed but for completeness
  | 1     => 2
  | _     => 2 + (n - 1) * n

theorem find_last_four_digits_of_N : (P 2011) % 10000 = 2112 := by
  -- we define P(2011) as per the general formula derived and then verify the modulo operation
  sorry

end find_last_four_digits_of_N_l240_240123


namespace journey_time_ratio_l240_240670

theorem journey_time_ratio (D : ℝ) (hD_pos : D > 0) :
  let T1 := D / 45
  let T2 := D / 30
  (T2 / T1) = (3 / 2) := 
by
  sorry

end journey_time_ratio_l240_240670


namespace regular_polygon_sides_l240_240878

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240878


namespace triangle_cos_b_l240_240046

-- Definitions according to the conditions
variables {a b c : ℝ}
variables (h1 : b^2 = a * c) (h2 : c = 2 * a)

-- The theorem to be proven
theorem triangle_cos_b (h1 : b^2 = a * c) (h2 : c = 2 * a) : 
  Real.cos_angle (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 3 / 4 :=
  sorry

end triangle_cos_b_l240_240046


namespace num_divisors_of_8_factorial_l240_240031

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l240_240031


namespace product_of_distances_is_one_l240_240098

theorem product_of_distances_is_one (k : ℝ) (x1 x2 : ℝ)
  (h1 : x1^2 - k*x1 - 1 = 0)
  (h2 : x2^2 - k*x2 - 1 = 0)
  (h3 : x1 ≠ x2) :
  (|x1| * |x2| = 1) :=
by
  -- Proof goes here
  sorry

end product_of_distances_is_one_l240_240098


namespace radius_of_circumscribed_circle_l240_240355

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l240_240355


namespace donna_pizza_slices_left_l240_240155

def total_slices_initial : ℕ := 12
def slices_eaten_lunch (slices : ℕ) : ℕ := slices / 2
def slices_remaining_after_lunch (slices : ℕ) : ℕ := slices - slices_eaten_lunch slices
def slices_eaten_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices / 3
def slices_remaining_after_dinner (slices : ℕ) : ℕ := slices_remaining_after_lunch slices - slices_eaten_dinner slices
def slices_shared_friend (slices : ℕ) : ℕ := slices_remaining_after_dinner slices / 4
def slices_remaining_final (slices : ℕ) : ℕ := slices_remaining_after_dinner slices - slices_shared_friend slices

theorem donna_pizza_slices_left : slices_remaining_final total_slices_initial = 3 :=
sorry

end donna_pizza_slices_left_l240_240155


namespace graph_sequence_periodic_l240_240800

open Finset

/-- Given a graph G_0 on vertices A_1, A_2, ..., A_n,
    and a sequence G_{n+1} constructed such that A_i and A_j are joined only if
    in G_n there is a vertex A_k ≠ A_i, A_j such that A_k is joined with both A_i and A_j,
    prove that the sequence {G_n} is periodic after some term with period T ≤ 2^n. -/
theorem graph_sequence_periodic (n : ℕ) (G_0 : SimpleGraph (Fin n))
  (G_seq : ℕ → SimpleGraph (Fin n))
  (h : ∀ k, ∀ (A_i A_j : Fin n), (A_i ∈ G_seq k.edges) → (A_j ∈ G_seq k.edges) →
  ∃ (A_k : Fin n), A_k ≠ A_i ∧ A_k ≠ A_j ∧ A_k ∈ G_seq k.edges ∧
  A_i ∈ G_seq (k+1).edges ∧ A_j ∈ G_seq (k+1).edges)
  : ∃ T ≤ 2^n, ∀ t ≥ T, G_seq t = G_seq T := sorry

end graph_sequence_periodic_l240_240800


namespace probability_leftmost_blue_off_rightmost_red_on_l240_240197

noncomputable def probability_specific_arrangement : ℚ :=
  let total_ways_arrange_colors := combinatorial.choose 6 3 in
  let total_ways_choose_on := combinatorial.choose 6 3 in
  let ways_arrange_colors_given_restrictions := combinatorial.choose 4 2 in
  let ways_choose_on_given_restrictions := combinatorial.choose 4 2 in
  (ways_arrange_colors_given_restrictions * ways_choose_on_given_restrictions : ℚ) / (total_ways_arrange_colors * total_ways_choose_on : ℚ)

theorem probability_leftmost_blue_off_rightmost_red_on :
  probability_specific_arrangement = 9 / 100 :=
begin
  -- The proof will be placed here
  sorry
end

end probability_leftmost_blue_off_rightmost_red_on_l240_240197


namespace option_c_same_function_l240_240963

-- Definitions based on conditions
def f_c (x : ℝ) : ℝ := x^2
def g_c (x : ℝ) : ℝ := 3 * x^6

-- Theorem statement that Option C f(x) and g(x) represent the same function
theorem option_c_same_function : ∀ x : ℝ, f_c x = g_c x := by
  sorry

end option_c_same_function_l240_240963


namespace num_valid_arrangements_l240_240077

def valid_arrangements : Nat := 5! - 2 * 4! + 3!

theorem num_valid_arrangements : 
  valid_arrangements = 78 :=
by
  unfold valid_arrangements
  rw [Nat.factorial_succ, Nat.factorial_succ, Nat.factorial_three, Nat.factorial_four]
  calc
    5 * 24 - 2 * 24 + 6 = 120 - 48 + 6 : by norm_num
    ... = 78 : by norm_num
  sorry

end num_valid_arrangements_l240_240077


namespace probability_blue_or_purple_is_4_over_11_l240_240387

def total_jelly_beans : ℕ := 10 + 12 + 13 + 15 + 5
def blue_or_purple_jelly_beans : ℕ := 15 + 5
def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_4_over_11 :
  probability_blue_or_purple = 4 / 11 :=
sorry

end probability_blue_or_purple_is_4_over_11_l240_240387


namespace regular_polygon_sides_l240_240927

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240927


namespace radius_of_circumscribed_circle_l240_240356

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l240_240356


namespace number_of_cars_washed_l240_240623

theorem number_of_cars_washed (cars trucks suvs total raised_per_car raised_per_truck raised_per_suv : ℕ)
  (hc : cars = 5)
  (ht : trucks = 5)
  (ha : cars + trucks + suvs = total)
  (h_cost_car : raised_per_car = 5)
  (h_cost_truck : raised_per_truck = 6)
  (h_cost_suv : raised_per_suv = 7)
  (h_amount_total : total = 100)
  (h_raised_trucks : trucks * raised_per_truck = 30)
  (h_raised_suvs : suvs * raised_per_suv = 35) :
  suvs + trucks + cars = 7 :=
by
  sorry

end number_of_cars_washed_l240_240623


namespace sixth_employee_salary_l240_240494

-- We define the salaries of the five employees
def salaries : List ℝ := [1000, 2500, 3100, 1500, 2000]

-- The mean of the salaries of these 5 employees and another employee
def mean_salary : ℝ := 2291.67

-- The number of employees
def number_of_employees : ℝ := 6

-- The total salary of the first five employees
def total_salary_5 : ℝ := salaries.sum

-- The total salary based on the given mean and number of employees
def total_salary_all : ℝ := mean_salary * number_of_employees

-- The statement to prove: The salary of the sixth employee
theorem sixth_employee_salary :
  total_salary_all - total_salary_5 = 3650.02 := 
  sorry

end sixth_employee_salary_l240_240494


namespace value_of_a_squared_b_plus_ab_squared_eq_4_l240_240705

variable (a b : ℝ)
variable (h_a : a = 2 + Real.sqrt 3)
variable (h_b : b = 2 - Real.sqrt 3)

theorem value_of_a_squared_b_plus_ab_squared_eq_4 :
  a^2 * b + a * b^2 = 4 := by
  sorry

end value_of_a_squared_b_plus_ab_squared_eq_4_l240_240705


namespace sum_m_b_eq_neg_five_halves_l240_240427

theorem sum_m_b_eq_neg_five_halves : 
  let x1 := 1 / 2
  let y1 := -1
  let x2 := -1 / 2
  let y2 := 2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m + b = -5 / 2 :=
by 
  sorry

end sum_m_b_eq_neg_five_halves_l240_240427


namespace regular_polygon_sides_l240_240950

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240950


namespace hyperbola_eccentricity_l240_240589

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (c : ℝ) (h3 : a^2 + b^2 = c^2) 
  (h4 : ∃ M : ℝ × ℝ, (M.fst^2 / a^2 - M.snd^2 / b^2 = 1) ∧ (M.snd^2 = 8 * M.fst)
    ∧ (|M.fst - 2| + |M.snd| = 5)) : 
  (c / a = 2) :=
by
  sorry

end hyperbola_eccentricity_l240_240589


namespace regular_polygon_sides_l240_240839

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240839


namespace alicia_total_deductions_in_cents_l240_240399

def Alicia_hourly_wage : ℝ := 25
def local_tax_rate : ℝ := 0.015
def retirement_contribution_rate : ℝ := 0.03

theorem alicia_total_deductions_in_cents :
  let wage_cents := Alicia_hourly_wage * 100
  let tax_deduction := wage_cents * local_tax_rate
  let after_tax_earnings := wage_cents - tax_deduction
  let retirement_contribution := after_tax_earnings * retirement_contribution_rate
  let total_deductions := tax_deduction + retirement_contribution
  total_deductions = 111 :=
by
  sorry

end alicia_total_deductions_in_cents_l240_240399


namespace value_of_fg3_l240_240291

namespace ProofProblem

def g (x : ℕ) : ℕ := x ^ 3
def f (x : ℕ) : ℕ := 3 * x + 2

theorem value_of_fg3 : f (g 3) = 83 := 
by 
  sorry -- Proof not needed

end ProofProblem

end value_of_fg3_l240_240291


namespace regular_polygon_sides_l240_240869

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240869


namespace min_value_abs_sum_pqr_inequality_l240_240170

theorem min_value_abs_sum (x : ℝ) : |x + 1| + |x - 2| ≥ 3 :=
by
  sorry

theorem pqr_inequality (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := 
by
  have f_min : ∀ x, |x + 1| + |x - 2| ≥ 3 := min_value_abs_sum
  sorry

end min_value_abs_sum_pqr_inequality_l240_240170


namespace product_of_solutions_l240_240565

theorem product_of_solutions : 
  ∀ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47) ∧ (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 :=
by
  sorry

end product_of_solutions_l240_240565


namespace number_of_ordered_triples_l240_240159

theorem number_of_ordered_triples :
  ∃ n, (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.lcm a b = 12 ∧ Nat.gcd b c = 6 ∧ Nat.lcm c a = 24) ∧ n = 4 :=
sorry

end number_of_ordered_triples_l240_240159


namespace chase_travel_time_l240_240410

-- Define the necessary mathematical structures
variable (time : Type)

-- Conditions
variable (chase_time cameron_time danielle_time : time)
variable (relation1 : cameron_time = 2 * chase_time)
variable (relation2 : danielle_time = 3 * cameron_time)
variable (danielle_given_time : danielle_time = 30)

-- Theorem statement
theorem chase_travel_time : chase_time = 180 := by
  sorry

end chase_travel_time_l240_240410


namespace regular_polygon_sides_l240_240916

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240916


namespace line_parallel_condition_l240_240750

theorem line_parallel_condition (a : ℝ) :
    (a = 1) → (∀ (x y : ℝ), (ax + 2 * y - 1 = 0) ∧ (x + (a + 1) * y + 4 = 0)) → (a = 1 ∨ a = -2) :=
by
sorry

end line_parallel_condition_l240_240750


namespace value_of_b_l240_240998

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l240_240998


namespace ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l240_240983

theorem ones_digit_largest_power_of_three_divides_factorial_3_pow_3 :
  (3 ^ 13) % 10 = 3 := by
  sorry

end ones_digit_largest_power_of_three_divides_factorial_3_pow_3_l240_240983


namespace swan_count_l240_240160

theorem swan_count (total_birds : ℕ) (fraction_ducks : ℚ):
  fraction_ducks = 5 / 6 →
  total_birds = 108 →
  ∃ (num_swans : ℕ), num_swans = 18 :=
by
  intro h_fraction_ducks h_total_birds
  sorry

end swan_count_l240_240160


namespace david_age_uniq_l240_240251

theorem david_age_uniq (C D E : ℚ) (h1 : C = 4 * D) (h2 : E = D + 7) (h3 : C = E + 1) : D = 8 / 3 := 
by 
  sorry

end david_age_uniq_l240_240251


namespace cube_volume_from_surface_area_l240_240293

theorem cube_volume_from_surface_area (SA : ℕ) (h : SA = 600) :
  ∃ V : ℕ, V = 1000 := by
  sorry

end cube_volume_from_surface_area_l240_240293


namespace intersection_A_B_l240_240574

-- Definition of sets A and B
def A := {x : ℝ | x > 2}
def B := { x : ℝ | (x - 1) * (x - 3) < 0 }

-- Claim that A ∩ B = {x : ℝ | 2 < x < 3}
theorem intersection_A_B :
  {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l240_240574


namespace marble_count_l240_240047

variable (r b g : ℝ)

-- Conditions
def condition1 : b = r / 1.3 := sorry
def condition2 : g = 1.5 * r := sorry

-- Theorem statement
theorem marble_count (h1 : b = r / 1.3) (h2 : g = 1.5 * r) :
  r + b + g = 3.27 * r :=
by sorry

end marble_count_l240_240047


namespace identify_incorrect_propositions_l240_240449

-- Definitions for parallel lines and planes
def line := Type -- Define a line type
def plane := Type -- Define a plane type
def parallel_to (l1 l2 : line) : Prop := sorry -- Assume a definition for parallel lines
def parallel_to_plane (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line parallel to a plane
def contained_in (l : line) (pl : plane) : Prop := sorry -- Assume a definition for a line contained in a plane

theorem identify_incorrect_propositions (a b : line) (α : plane) :
  (parallel_to_plane a α ∧ parallel_to_plane b α → ¬parallel_to a b) ∧
  (parallel_to_plane a α ∧ contained_in b α → ¬parallel_to a b) ∧
  (parallel_to a b ∧ contained_in b α → ¬parallel_to_plane a α) ∧
  (parallel_to a b ∧ parallel_to_plane b α → ¬parallel_to_plane a α) :=
by
  sorry -- The proof is not required

end identify_incorrect_propositions_l240_240449


namespace leak_empties_cistern_in_12_hours_l240_240819

theorem leak_empties_cistern_in_12_hours 
  (R : ℝ) (L : ℝ)
  (h1 : R = 1 / 4) 
  (h2 : R - L = 1 / 6) : 
  1 / L = 12 := 
by
  -- proof will go here
  sorry

end leak_empties_cistern_in_12_hours_l240_240819


namespace find_k_l240_240994

def A (a b : ℤ) : Prop := 3 * a + b - 2 = 0
def B (a b : ℤ) (k : ℤ) : Prop := k * (a^2 - a + 1) - b = 0

theorem find_k (k : ℤ) (h : ∃ a b : ℤ, A a b ∧ B a b k ∧ a > 0) : k = -1 ∨ k = 2 :=
by
  sorry

end find_k_l240_240994


namespace part1_part2_part3_l240_240585

open Real

def f (x : ℝ) : ℝ := 2 * (cos x)^2 - 2 * sqrt 3 * (sin x) * (cos x) - 1

theorem part1 : f (-π / 12) = sqrt 3 :=
by {
  sorry
}

theorem part2 : ∃ (p : ℝ), 0 < p ∧ ∀ (x : ℝ), f (x + p) = f x :=
by {
  use π,
  sorry
}

theorem part3 : ∀ (k : ℤ), k * π - (2 * π) / 3 ≤ x → x ≤ k * π - π / 6 → 
  ∃ (a b : ℝ), k * π - (2 * π) / 3 = a ∧ k * π - π / 6 = b ∧ 
  ∀ (x : ℝ), a ≤ x ∧ x ≤ b → f(x) is_strictly_increasing_on (set.Icc a b) :=
by {
  intros k x H1 H2,
  use [(k : ℝ) * π - (2 * π) / 3, (k : ℝ) * π - π / 6],
  sorry
}

end part1_part2_part3_l240_240585


namespace linear_function_not_in_second_quadrant_l240_240643

-- Define the linear function y = x - 1.
def linear_function (x : ℝ) : ℝ := x - 1

-- Define the condition for a point to be in the second quadrant.
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State that for any point (x, y) in the second quadrant, it does not satisfy y = x - 1.
theorem linear_function_not_in_second_quadrant {x y : ℝ} (h : in_second_quadrant x y) : linear_function x ≠ y :=
sorry

end linear_function_not_in_second_quadrant_l240_240643


namespace regular_polygon_sides_l240_240932

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240932


namespace part_I_part_II_l240_240156

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II_l240_240156


namespace B_values_for_divisibility_l240_240084

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l240_240084


namespace regular_polygon_sides_l240_240837

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240837


namespace flask_forces_l240_240701

theorem flask_forces (r : ℝ) (ρ g h_A h_B h_C V : ℝ) (A : ℝ) (FA FB FC : ℝ) (h1 : r = 2)
  (h2 : A = π * r^2)
  (h3 : V = A * h_A ∧ V = A * h_B ∧ V = A * h_C)
  (h4 : FC = ρ * g * h_C * A)
  (h5 : FA = ρ * g * h_A * A)
  (h6 : FB = ρ * g * h_B * A)
  (h7 : h_C > h_A ∧ h_A > h_B) : FC > FA ∧ FA > FB := 
sorry

end flask_forces_l240_240701


namespace absolute_value_expression_evaluation_l240_240694

theorem absolute_value_expression_evaluation : abs (-2) * (abs (-Real.sqrt 25) - abs (Real.sin (5 * Real.pi / 2))) = 8 := by
  sorry

end absolute_value_expression_evaluation_l240_240694


namespace simple_interest_rate_l240_240116

theorem simple_interest_rate (P A : ℝ) (T : ℕ) (R : ℝ) 
  (P_pos : P = 800) (A_pos : A = 950) (T_pos : T = 5) :
  R = 3.75 :=
by
  sorry

end simple_interest_rate_l240_240116


namespace regular_polygon_sides_l240_240909

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240909


namespace Maria_needs_more_l240_240735

def num_mechanics : Nat := 20
def num_thermodynamics : Nat := 50
def num_optics : Nat := 30
def total_questions : Nat := num_mechanics + num_thermodynamics + num_optics

def correct_mechanics : Nat := (80 * num_mechanics) / 100
def correct_thermodynamics : Nat := (50 * num_thermodynamics) / 100
def correct_optics : Nat := (70 * num_optics) / 100
def correct_total : Nat := correct_mechanics + correct_thermodynamics + correct_optics

def correct_for_passing : Nat := (65 * total_questions) / 100
def additional_needed : Nat := correct_for_passing - correct_total

theorem Maria_needs_more:
  additional_needed = 3 := by
  sorry

end Maria_needs_more_l240_240735


namespace regular_polygon_sides_l240_240866

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240866


namespace min_a2_b2_l240_240711

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + 2 * x^2 + b * x + 1 = 0) : a^2 + b^2 ≥ 8 :=
sorry

end min_a2_b2_l240_240711


namespace beads_needed_for_jewelry_l240_240075

/-
  We define the parameters based on the problem statement.
-/

def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def total_beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

def repeats_per_bracelet : ℕ := 3
def repeats_per_necklace : ℕ := 5

/-
  We calculate the total number of beads for 1 bracelet and 10 necklaces.
-/

def beads_per_bracelet : ℕ := total_beads_per_pattern * repeats_per_bracelet
def beads_per_necklace : ℕ := total_beads_per_pattern * repeats_per_necklace
def total_beads_needed : ℕ := beads_per_bracelet + beads_per_necklace * 10

theorem beads_needed_for_jewelry:
  total_beads_needed = 742 :=
by 
  sorry

end beads_needed_for_jewelry_l240_240075


namespace find_finite_sets_l240_240259

open Set

theorem find_finite_sets (X : Set ℝ) (h1 : X.Nonempty) (h2 : X.Finite)
  (h3 : ∀ x ∈ X, (x + |x|) ∈ X) :
  ∃ (F : Set ℝ), F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = insert 0 F :=
sorry

end find_finite_sets_l240_240259


namespace num_positive_divisors_8_factorial_l240_240018

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l240_240018


namespace quadratic_roots_sum_product_l240_240429

theorem quadratic_roots_sum_product {p q : ℝ} 
  (h1 : p / 3 = 10) 
  (h2 : q / 3 = 15) : 
  p + q = 75 := sorry

end quadratic_roots_sum_product_l240_240429


namespace regular_polygon_sides_l240_240943

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240943


namespace cubes_with_no_colored_faces_l240_240677

theorem cubes_with_no_colored_faces (width length height : ℕ) (total_cubes cube_side : ℕ) :
  width = 6 ∧ length = 5 ∧ height = 4 ∧ total_cubes = 120 ∧ cube_side = 1 →
  (width - 2) * (length - 2) * (height - 2) = 24 :=
by
  intros h
  sorry

end cubes_with_no_colored_faces_l240_240677


namespace problem_1_problem_2_l240_240587

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 1
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^3 + b * x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x + g (4 * a^2 / 4) x

theorem problem_1 (a b : ℝ) (h1 : a > 0) (h2 : f a 1 = g b 1) 
  (h3 : deriv (f a) 1 = deriv (g b) 1) : 
  a = 3 ∧ b = 3 :=
sorry

theorem problem_2 (a : ℝ) (h4 : a > 0) (h5 : a^2 = 4 * (4 * a^2 / 4)) : 
  (∀ x, x < -a / 2 → deriv (h a) x < 0) ∧ 
  (∀ x, x > -a / 2 ∧ x < -a / 6 → deriv (h a) x > 0) ∧ 
  (∀ x, x > -a / 6 → deriv (h a) x < 0) ∧ 
  (∀ x, x ≤ -1 → h(a) x ≤ h(a) (-a / 2) ∧ x = -a / 2 → h(a) x = 1) :=
sorry

end problem_1_problem_2_l240_240587


namespace regular_polygon_sides_l240_240913

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240913


namespace regular_polygon_sides_l240_240832

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240832


namespace regular_polygon_sides_l240_240885

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240885


namespace find_ab_l240_240107

theorem find_ab (a b : ℕ) (h1 : 1 <= a) (h2 : a < 10) (h3 : 0 <= b) (h4 : b < 10) (h5 : 66 * ((1 : ℝ) + ((10 * a + b : ℕ) / 100) - (↑(10 * a + b) / 99)) = 0.5) : 10 * a + b = 75 :=
by
  sorry

end find_ab_l240_240107


namespace regular_polygon_sides_l240_240844

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240844


namespace bottle_caps_per_group_l240_240747

theorem bottle_caps_per_group (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 12) (h2 : num_groups = 6) : 
  total_caps / num_groups = caps_per_group := by
  sorry

end bottle_caps_per_group_l240_240747


namespace regular_polygon_sides_l240_240955

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240955


namespace number_of_divisors_of_8_fact_l240_240036

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l240_240036


namespace regular_polygon_sides_l240_240904

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240904


namespace inequality_property_l240_240683

theorem inequality_property (a b : ℝ) (h : a > b) : -5 * a < -5 * b := sorry

end inequality_property_l240_240683


namespace sum_T_19_34_51_l240_240684

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2 : ℕ) else (n + 1) / 2

def T (n : ℕ) : ℤ :=
  2 + S n

theorem sum_T_19_34_51 : T 19 + T 34 + T 51 = 25 := 
by
  -- Add the steps here
  sorry

end sum_T_19_34_51_l240_240684


namespace integer_root_b_l240_240997

theorem integer_root_b (a1 a2 a3 a4 a5 b : ℤ)
  (h_diff : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a4 ≠ a5)
  (h_sum : a1 + a2 + a3 + a4 + a5 = 9)
  (h_prod : (b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) :
  b = 10 :=
sorry

end integer_root_b_l240_240997


namespace max_value_min_expression_l240_240100

def f (x y : ℝ) : ℝ :=
  x^3 + (y-4)*x^2 + (y^2-4*y+4)*x + (y^3-4*y^2+4*y)

theorem max_value_min_expression (a b c : ℝ) (h₁: a ≠ b) (h₂: b ≠ c) (h₃: c ≠ a)
  (hab : f a b = f b c) (hbc : f b c = f c a) :
  (max (min (a^4 - 4*a^3 + 4*a^2) (min (b^4 - 4*b^3 + 4*b^2) (c^4 - 4*c^3 + 4*c^2))) 1) = 1 :=
sorry

end max_value_min_expression_l240_240100


namespace num_pos_divisors_fact8_l240_240027

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l240_240027


namespace regular_polygon_sides_l240_240953

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240953


namespace second_discount_percentage_l240_240649

theorem second_discount_percentage
    (original_price : ℝ)
    (first_discount : ℝ)
    (final_sale_price : ℝ)
    (second_discount : ℝ)
    (h1 : original_price = 390)
    (h2 : first_discount = 14)
    (h3 : final_sale_price = 285.09) :
    second_discount = 15 :=
by
  -- Since we are not providing the full proof, we assume the steps to be correct
  sorry

end second_discount_percentage_l240_240649


namespace factorize_poly_l240_240517

theorem factorize_poly : 
  (x : ℤ) → (x^12 + x^6 + 1) = (x^4 - x^2 + 1) * (x^8 + x^4 + 1) :=
by
  sorry

end factorize_poly_l240_240517


namespace simplify_expression_l240_240073

theorem simplify_expression :
  ((4 * 7) / (12 * 14)) * ((9 * 12 * 14) / (4 * 7 * 9)) ^ 2 = 1 := 
by
  sorry

end simplify_expression_l240_240073


namespace original_weight_of_apple_box_l240_240654

theorem original_weight_of_apple_box:
  ∀ (x : ℕ), (3 * x - 12 = x) → x = 6 :=
by
  intros x h
  sorry

end original_weight_of_apple_box_l240_240654


namespace roots_of_quadratic_eq_l240_240785

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l240_240785


namespace caffeine_safe_amount_l240_240492

theorem caffeine_safe_amount (max_caffeine : ℕ) (caffeine_per_drink : ℕ) (num_drinks : ℕ) :
    max_caffeine = 500 →
    caffeine_per_drink = 120 →
    num_drinks = 4 →
    max_caffeine - (caffeine_per_drink * num_drinks) = 20 :=
by
  intros h_max h_caffeine h_num
  rw [h_max, h_caffeine, h_num]
  norm_num
  sorry

end caffeine_safe_amount_l240_240492


namespace problem_1_problem_2_problem_3_l240_240120

-- Problem 1: Prove that if the inequality |x-1| - |x-2| < a holds for all x in ℝ, then a > 1.
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a) → a > 1 :=
sorry

-- Problem 2: Prove that if the inequality |x-1| - |x-2| < a has at least one real solution, then a > -1.
theorem problem_2 (a : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| < a) → a > -1 :=
sorry

-- Problem 3: Prove that if the solution set of the inequality |x-1| - |x-2| < a is empty, then a ≤ -1.
theorem problem_3 (a : ℝ) :
  (¬∃ x : ℝ, |x - 1| - |x - 2| < a) → a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l240_240120


namespace correct_fraction_l240_240050

theorem correct_fraction (x y : ℕ) (h1 : 480 * 5 / 6 = 480 * x / y + 250) : x / y = 5 / 16 :=
by
  sorry

end correct_fraction_l240_240050


namespace solve_for_x_l240_240044

theorem solve_for_x (x : ℚ) :  (1/2) * (12 * x + 3) = 3 * x + 2 → x = 1/6 := by
  intro h
  sorry

end solve_for_x_l240_240044


namespace age_ratio_l240_240965

-- Define the conditions
def ArunCurrentAgeAfter6Years (A: ℕ) : Prop := A + 6 = 36
def DeepakCurrentAge : ℕ := 42

-- Define the goal statement
theorem age_ratio (A: ℕ) (hc: ArunCurrentAgeAfter6Years A) : A / gcd A DeepakCurrentAge = 5 ∧ DeepakCurrentAge / gcd A DeepakCurrentAge = 7 :=
by
  sorry

end age_ratio_l240_240965


namespace find_a_l240_240599

theorem find_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 1) : a = 3 :=
by
  sorry

end find_a_l240_240599


namespace find_length_AB_l240_240618

noncomputable def length_of_AB (DE DF : ℝ) (AC : ℝ) : ℝ :=
  (AC * DE) / DF

theorem find_length_AB (DE DF AC : ℝ) (pro1 : DE = 9) (pro2 : DF = 17) (pro3 : AC = 10) :
    length_of_AB DE DF AC = 90 / 17 :=
  by
    rw [pro1, pro2, pro3]
    unfold length_of_AB
    norm_num

end find_length_AB_l240_240618


namespace verify_extending_points_l240_240754

noncomputable def verify_P_and_Q (A B P Q : ℝ → ℝ → ℝ) : Prop := 
  let vector_relation_P := P = - (2/5) • A + (7/5) • B
  let vector_relation_Q := Q = - (1/4) • A + (5/4) • B 
  vector_relation_P ∧ vector_relation_Q

theorem verify_extending_points 
  (A B P Q : ℝ → ℝ → ℝ)
  (h1 : 7 • (P - A) = 2 • (B - P))
  (h2 : 5 • (Q - A) = 1 • (Q - B)) :
  verify_P_and_Q A B P Q := 
by
  sorry  

end verify_extending_points_l240_240754


namespace circle_radius_l240_240361

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l240_240361


namespace regular_polygon_sides_l240_240911

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240911


namespace equation_roots_l240_240793

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l240_240793


namespace find_k_l240_240004

theorem find_k
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), (x - k * y - 5 = 0 ∧ x^2 + y^2 = 10 ∧ (A = (x, y) ∨ B = (x, y))))
  (h2 : (A.fst^2 + A.snd^2 = 10) ∧ (B.fst^2 + B.snd^2 = 10))
  (h3 : (A.fst - k * A.snd - 5 = 0) ∧ (B.fst - k * B.snd - 5 = 0))
  (h4 : A.fst * B.fst + A.snd * B.snd = 0) :
  k = 2 ∨ k = -2 :=
by
  sorry

end find_k_l240_240004


namespace yearly_return_of_1500_investment_l240_240225

theorem yearly_return_of_1500_investment 
  (combined_return_percent : ℝ)
  (total_investment : ℕ)
  (return_500 : ℕ)
  (investment_500 : ℕ)
  (investment_1500 : ℕ) :
  combined_return_percent = 0.085 →
  total_investment = (investment_500 + investment_1500) →
  return_500 = (investment_500 * 7 / 100) →
  investment_500 = 500 →
  investment_1500 = 1500 →
  total_investment = 2000 →
  (return_500 + investment_1500 * combined_return_percent * 100) = (combined_return_percent * total_investment * 100) →
  ((investment_1500 * (9 : ℝ)) / 100) + return_500 = 0.085 * total_investment →
  (investment_1500 * 7 / 100) = investment_1500 →
  (investment_1500 / investment_1500) = (13500 / 1500) →
  (9 : ℝ) = 9 :=
sorry

end yearly_return_of_1500_investment_l240_240225


namespace number_of_positive_solutions_l240_240214

theorem number_of_positive_solutions (x y z : ℕ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 12) :
    ∃ (n : ℕ), n = 55 :=
by 
  sorry

end number_of_positive_solutions_l240_240214


namespace regular_polygon_sides_l240_240905

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240905


namespace initial_paintings_l240_240539

theorem initial_paintings (x : ℕ) (h : x - 3 = 95) : x = 98 :=
sorry

end initial_paintings_l240_240539


namespace kaleb_initial_cherries_l240_240620

/-- Kaleb's initial number of cherries -/
def initial_cherries : ℕ := 67

/-- Cherries that Kaleb ate -/
def eaten_cherries : ℕ := 25

/-- Cherries left after eating -/
def left_cherries : ℕ := 42

/-- Prove that the initial number of cherries is 67 given the conditions. -/
theorem kaleb_initial_cherries :
  eaten_cherries + left_cherries = initial_cherries :=
by
  sorry

end kaleb_initial_cherries_l240_240620


namespace regular_polygon_sides_l240_240872

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240872


namespace arithmetic_seq_b3_b6_l240_240626

theorem arithmetic_seq_b3_b6 (b : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, b n = b 1 + n * d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_b4_b5 : b 4 * b 5 = 30) :
  b 3 * b 6 = 28 := 
sorry

end arithmetic_seq_b3_b6_l240_240626


namespace GIMPS_meaning_l240_240659

/--
  Curtis Cooper's team discovered the largest prime number known as \( 2^{74,207,281} - 1 \), which is a Mersenne prime.
  GIMPS stands for "Great Internet Mersenne Prime Search."

  Prove that GIMPS means "Great Internet Mersenne Prime Search".
-/
theorem GIMPS_meaning : GIMPS = "Great Internet Mersenne Prime Search" :=
  sorry

end GIMPS_meaning_l240_240659


namespace transformed_curve_l240_240441

variables (x y x' y' : ℝ)

def original_curve := (x^2) / 4 - y^2 = 1
def transformation_x := x' = (1/2) * x
def transformation_y := y' = 2 * y

theorem transformed_curve : original_curve x y → transformation_x x x' → transformation_y y y' → x^2 - (y^2) / 4 = 1 := 
sorry

end transformed_curve_l240_240441


namespace alice_wins_l240_240268

noncomputable def game_condition (r : ℝ) (f : ℕ → ℝ) : Prop :=
∀ n, 0 ≤ f n ∧ f n ≤ 1

theorem alice_wins (r : ℝ) (f : ℕ → ℝ) (hf : game_condition r f) :
  r ≤ 3 → (∃ x : ℕ → ℝ, game_condition 3 x ∧ (abs (x 0 - x 1) + abs (x 2 - x 3) + abs (x 4 - x 5) ≥ r)) :=
by
  sorry

end alice_wins_l240_240268


namespace proof_problem_l240_240582

theorem proof_problem (a b : ℝ) (h : a^2 + b^2 + 2*a - 4*b + 5 = 0) : 2*a^2 + 4*b - 3 = 7 :=
sorry

end proof_problem_l240_240582


namespace unit_digit_div_l240_240530

theorem unit_digit_div (n : ℕ) : (33 * 10) % (2 ^ 1984) = n % 10 :=
by
  have h := 2 ^ 1984
  have u_digit_2_1984 := 6 -- Since 1984 % 4 = 0, last digit in the cycle of 2^n for n ≡ 0 [4] is 6
  sorry
  
example : (33 * 10) / (2 ^ 1984) % 10 = 6 :=
by sorry

end unit_digit_div_l240_240530


namespace regular_polygon_sides_l240_240924

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240924


namespace A_equals_k_with_conditions_l240_240466

theorem A_equals_k_with_conditions (n m : ℕ) (h_n : 1 < n) (h_m : 1 < m) :
  ∃ k : ℤ, (1 : ℝ) < k ∧ (( (n + Real.sqrt (n^2 - 4)) / 2 ) ^ m = (k + Real.sqrt (k^2 - 4)) / 2) :=
sorry

end A_equals_k_with_conditions_l240_240466


namespace solution_inequality_l240_240566

theorem solution_inequality (θ x : ℝ)
  (h : |x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) : 
  -1 ≤ x ∧ x ≤ -Real.cos (2 * θ) :=
sorry

end solution_inequality_l240_240566


namespace square_area_l240_240668

theorem square_area (x y : ℝ) 
  (h1 : x = 20 ∧ y = 20)
  (h2 : x = 20 ∧ y = 5)
  (h3 : x = x ∧ y = 5)
  (h4 : x = x ∧ y = 20)
  : (∃ a : ℝ, a = 225) :=
sorry

end square_area_l240_240668


namespace Jack_emails_evening_l240_240300

theorem Jack_emails_evening : 
  ∀ (morning_emails evening_emails : ℕ), 
  (morning_emails = 9) ∧ 
  (evening_emails = morning_emails - 2) → 
  evening_emails = 7 := 
by
  intros morning_emails evening_emails
  sorry

end Jack_emails_evening_l240_240300


namespace eduardo_ate_fraction_of_remaining_l240_240219

theorem eduardo_ate_fraction_of_remaining (init_cookies : ℕ) (nicole_fraction : ℚ) (remaining_percent : ℚ) :
  init_cookies = 600 →
  nicole_fraction = 2 / 5 →
  remaining_percent = 24 / 100 →
  (360 - (600 * 24 / 100)) / 360 = 3 / 5 := by
  sorry

end eduardo_ate_fraction_of_remaining_l240_240219


namespace garden_ratio_length_to_width_l240_240340

theorem garden_ratio_length_to_width (width length : ℕ) (area : ℕ) 
  (h1 : area = 507) 
  (h2 : width = 13) 
  (h3 : length * width = area) :
  length / width = 3 :=
by
  -- Proof to be filled in.
  sorry

end garden_ratio_length_to_width_l240_240340


namespace cart_total_distance_l240_240231

-- Definitions for the conditions
def first_section_distance := (15/2) * (8 + (8 + 14 * 10))
def second_section_distance := (15/2) * (148 + (148 + 14 * 6))

-- Combining both distances
def total_distance := first_section_distance + second_section_distance

-- Statement to be proved
theorem cart_total_distance:
  total_distance = 4020 :=
by
  sorry

end cart_total_distance_l240_240231


namespace solve_integer_pairs_l240_240820

-- Definition of the predicate that (m, n) satisfies the given equation
def satisfies_equation (m n : ℤ) : Prop :=
  m * n^2 = 2009 * (n + 1)

-- Theorem stating that the only solutions are (4018, 1) and (0, -1)
theorem solve_integer_pairs :
  ∀ (m n : ℤ), satisfies_equation m n ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by
  sorry

end solve_integer_pairs_l240_240820


namespace regular_polygon_sides_l240_240941

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240941


namespace smallest_even_number_of_seven_l240_240767

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l240_240767


namespace B_values_for_divisibility_l240_240085

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l240_240085


namespace sequence_an_sequence_Tn_l240_240716

theorem sequence_an (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, 2 * S n = a n ^ 2 + a n):
  ∀ n, a n = n :=
sorry

theorem sequence_Tn (b : ℕ → ℕ) (T : ℕ → ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, 2 * S n = a n ^ 2 + a n) (h2 : ∀ n, a n = n) (h3 : ∀ n, b n = 2^n * a n):
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
sorry

end sequence_an_sequence_Tn_l240_240716


namespace largest_divisor_of_prime_squares_l240_240464

theorem largest_divisor_of_prime_squares (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q < p) : 
  ∃ d : ℕ, ∀ p q : ℕ, Prime p → Prime q → q < p → d ∣ (p^2 - q^2) ∧ ∀ k : ℕ, (∀ p q : ℕ, Prime p → Prime q → q < p → k ∣ (p^2 - q^2)) → k ≤ d :=
by 
  use 2
  {
    sorry
  }

end largest_divisor_of_prime_squares_l240_240464


namespace abc_inequality_l240_240468

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3 / 4 :=
by
  sorry

end abc_inequality_l240_240468


namespace number_of_chips_per_day_l240_240469

def total_chips : ℕ := 100
def chips_first_day : ℕ := 10
def total_days : ℕ := 10
def days_remaining : ℕ := total_days - 1
def chips_remaining : ℕ := total_chips - chips_first_day

theorem number_of_chips_per_day : 
  chips_remaining / days_remaining = 10 :=
by 
  unfold chips_remaining days_remaining total_chips chips_first_day total_days
  sorry

end number_of_chips_per_day_l240_240469


namespace find_minimal_sum_l240_240709

theorem find_minimal_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * (x + 1)) ∣ (y * (y + 1)) →
  ¬(x ∣ y ∨ x ∣ (y + 1)) →
  ¬((x + 1) ∣ y ∨ (x + 1) ∣ (y + 1)) →
  x = 14 ∧ y = 35 ∧ x^2 + y^2 = 1421 :=
sorry

end find_minimal_sum_l240_240709


namespace max_of_2x_plus_y_l240_240579

theorem max_of_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y / 2 + 1 / x + 8 / y = 10) : 
  2 * x + y ≤ 18 :=
sorry

end max_of_2x_plus_y_l240_240579


namespace regular_polygon_sides_l240_240937

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240937


namespace mathematicians_correctness_l240_240331

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  (3 / 8 < 2 / 5 ∧ 2 / 5 < 17 / 40) → false :=
by 
  sorry

end mathematicians_correctness_l240_240331


namespace A_is_9_years_older_than_B_l240_240296

-- Define the conditions
variables (A_years B_years : ℕ)

def given_conditions : Prop :=
  B_years = 39 ∧ A_years + 10 = 2 * (B_years - 10)

-- Theorem to prove the correct answer
theorem A_is_9_years_older_than_B (h : given_conditions A_years B_years) : A_years - B_years = 9 :=
by
  sorry

end A_is_9_years_older_than_B_l240_240296


namespace find_r_in_geometric_sequence_l240_240607

noncomputable def sum_of_geometric_sequence (n : ℕ) (r : ℝ) := 3^n + r

theorem find_r_in_geometric_sequence:
  ∃ r : ℝ, 
  (∀ n : ℕ, n >= 1 → sum_of_geometric_sequence n r = 3^n +  r) ∧
  (∀ n : ℕ, n >= 2 → 
    let S_n := sum_of_geometric_sequence n r in
    let S_n_minus_1 := sum_of_geometric_sequence (n - 1) r in
    let a_n := S_n - S_n_minus_1 in
    a_n = 2 * 3^(n - 1)) ∧
  (∃ a1 : ℝ, a1 = 3 + r ∧ a1 * 3 = 6) ∧
  r = -1 :=
by sorry

end find_r_in_geometric_sequence_l240_240607


namespace one_kid_six_whiteboards_l240_240264

theorem one_kid_six_whiteboards (k: ℝ) (b1 b2: ℝ) (t1 t2: ℝ) 
  (hk: k = 1) (hb1: b1 = 3) (hb2: b2 = 6) 
  (ht1: t1 = 20) 
  (H: 4 * t1 / b1 = t2 / b2) : 
  t2 = 160 := 
by
  -- provide the proof here
  sorry

end one_kid_six_whiteboards_l240_240264


namespace real_numbers_correspond_to_number_line_l240_240773

noncomputable def number_line := ℝ

def real_numbers := ℝ

theorem real_numbers_correspond_to_number_line :
  ∀ (p : ℝ), ∃ (r : real_numbers), r = p ∧ ∀ (r : real_numbers), ∃ (p : ℝ), p = r :=
by
  sorry

end real_numbers_correspond_to_number_line_l240_240773


namespace minimize_PA2_plus_PB2_plus_PC2_l240_240294

def PA (x y : ℝ) : ℝ := (x - 3) ^ 2 + (y + 1) ^ 2
def PB (x y : ℝ) : ℝ := (x + 1) ^ 2 + (y - 4) ^ 2
def PC (x y : ℝ) : ℝ := (x - 1) ^ 2 + (y + 6) ^ 2

theorem minimize_PA2_plus_PB2_plus_PC2 :
  ∃ x y : ℝ, (PA x y + PB x y + PC x y) = 64 :=
by
  use 1
  use -1
  simp [PA, PB, PC]
  sorry

end minimize_PA2_plus_PB2_plus_PC2_l240_240294


namespace age_ratio_l240_240801

theorem age_ratio (s a : ℕ) (h1 : s - 3 = 2 * (a - 3)) (h2 : s - 7 = 3 * (a - 7)) :
  ∃ x : ℕ, (x = 23) ∧ (s + x) / (a + x) = 3 / 2 :=
by
  sorry

end age_ratio_l240_240801


namespace minimize_quadratic_expression_l240_240161

noncomputable def quadratic_expression (b : ℝ) : ℝ :=
  (1 / 3) * b^2 + 7 * b - 6

theorem minimize_quadratic_expression : ∃ b : ℝ, quadratic_expression b = -10.5 :=
  sorry

end minimize_quadratic_expression_l240_240161


namespace JakeMowingEarnings_l240_240458

theorem JakeMowingEarnings :
  (∀ rate hours_mowing hours_planting (total_charge : ℝ),
      rate = 20 →
      hours_mowing = 1 →
      hours_planting = 2 →
      total_charge = 45 →
      (total_charge = hours_planting * rate + 5) →
      hours_mowing * rate = 20) :=
by
  intros rate hours_mowing hours_planting total_charge
  sorry

end JakeMowingEarnings_l240_240458


namespace distinct_integers_real_roots_l240_240504

theorem distinct_integers_real_roots (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > b) (h5 : b > c) :
    (∃ x : ℝ, x^2 + 2 * a * x + 3 * (b + c) = 0) :=
sorry

end distinct_integers_real_roots_l240_240504


namespace mult_xy_eq_200_over_3_l240_240497

def hash_op (a b : ℚ) : ℚ := a + a / b

def x : ℚ := hash_op 8 3

def y : ℚ := hash_op 5 4

theorem mult_xy_eq_200_over_3 : x * y = 200 / 3 := 
by 
  -- lean uses real division operator, and hash_op must remain rational
  sorry

end mult_xy_eq_200_over_3_l240_240497


namespace inequality_holds_l240_240602

theorem inequality_holds 
  (a b c : ℝ) 
  (h1 : a > 0)
  (h2 : b < 0) 
  (h3 : b > c) : 
  (a / (c^2)) > (b / (c^2)) :=
by
  sorry

end inequality_holds_l240_240602


namespace solved_fraction_equation_l240_240428

theorem solved_fraction_equation :
  ∀ (x : ℚ),
    x ≠ 2 →
    x ≠ 7 →
    x ≠ -5 →
    (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) →
    x = 55 / 13 := by
  sorry

end solved_fraction_equation_l240_240428


namespace largest_five_digit_congruent_to_31_modulo_26_l240_240510

theorem largest_five_digit_congruent_to_31_modulo_26 :
  ∃ x : ℕ, (10000 ≤ x ∧ x < 100000) ∧ x % 26 = 31 ∧ x = 99975 :=
by
  sorry

end largest_five_digit_congruent_to_31_modulo_26_l240_240510


namespace regular_polygon_sides_l240_240948

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240948


namespace cos_75_deg_l240_240412

theorem cos_75_deg : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l240_240412


namespace necessary_and_sufficient_condition_l240_240275

variable (f : ℝ → ℝ)

-- Define even function
def even_function : Prop := ∀ x, f x = f (-x)

-- Define periodic function with period 2
def periodic_function : Prop := ∀ x, f (x + 2) = f x

-- Define increasing function on [0, 1]
def increasing_on_0_1 : Prop := ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f x ≤ f y

-- Define decreasing function on [3, 4]
def decreasing_on_3_4 : Prop := ∀ x y, 3 ≤ x → x ≤ y → y ≤ 4 → f x ≥ f y

theorem necessary_and_sufficient_condition :
  even_function f →
  periodic_function f →
  (increasing_on_0_1 f ↔ decreasing_on_3_4 f) :=
by
  intros h_even h_periodic
  sorry

end necessary_and_sufficient_condition_l240_240275


namespace discount_percentage_l240_240134

variable (P : ℝ)  -- Original price of the car
variable (D : ℝ)  -- Discount percentage in decimal form
variable (S : ℝ)  -- Selling price of the car

theorem discount_percentage
  (h1 : S = P * (1 - D) * 1.70)
  (h2 : S = P * 1.1899999999999999) :
  D = 0.3 :=
by
  -- The proof goes here
  sorry

end discount_percentage_l240_240134


namespace factorize_poly_l240_240522

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l240_240522


namespace students_taking_history_but_not_statistics_l240_240183

-- Definitions based on conditions
def T : Nat := 150
def H : Nat := 58
def S : Nat := 42
def H_union_S : Nat := 95

-- Statement to prove
theorem students_taking_history_but_not_statistics : H - (H + S - H_union_S) = 53 :=
by
  sorry

end students_taking_history_but_not_statistics_l240_240183


namespace period_of_function_is_2pi_over_3_l240_240650

noncomputable def period_of_f (x : ℝ) : ℝ :=
  4 * (Real.sin x)^3 - Real.sin x + 2 * (Real.sin (x / 2) - Real.cos (x / 2))^2

theorem period_of_function_is_2pi_over_3 : ∀ x, period_of_f (x + (2 * Real.pi) / 3) = period_of_f x :=
by sorry

end period_of_function_is_2pi_over_3_l240_240650


namespace isosceles_triangles_with_perimeter_25_l240_240009

/-- Prove that there are 6 distinct isosceles triangles with integer side lengths 
and a perimeter of 25 -/
theorem isosceles_triangles_with_perimeter_25 :
  ∃ (count : ℕ), 
    count = 6 ∧ 
    (∀ (a b : ℕ), 
      let a1 := a,
          a2 := a,
          b3 := b in
      2 * a + b = 25 → 
      2 * a > b ∧ a + b > a ∧ b < 2 * a ∧
      a > 0 ∧ b > 0 ∧ a ∈ finset.Icc 7 12) :=
by sorry

end isosceles_triangles_with_perimeter_25_l240_240009


namespace sum_of_squares_eq_zero_iff_all_zero_l240_240447

theorem sum_of_squares_eq_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end sum_of_squares_eq_zero_iff_all_zero_l240_240447


namespace exponent_multiplication_rule_l240_240405

theorem exponent_multiplication_rule :
  3000 * (3000 ^ 3000) = 3000 ^ 3001 := 
by {
  sorry
}

end exponent_multiplication_rule_l240_240405


namespace radius_of_circumscribed_circle_l240_240345

noncomputable def circumscribed_circle_radius (r : ℝ) : Prop :=
  let s := r * Real.sqrt 2 in
  4 * s = π * r ^ 2 -> r = 4 * Real.sqrt 2 / π

-- Statement of the theorem to be proved
theorem radius_of_circumscribed_circle (r : ℝ) : circumscribed_circle_radius r := 
sorry

end radius_of_circumscribed_circle_l240_240345


namespace regular_polygon_sides_l240_240873

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240873


namespace train_speed_is_25_kmph_l240_240398

noncomputable def train_speed_kmph (train_length_m : ℕ) (man_speed_kmph : ℕ) (cross_time_s : ℕ) : ℕ :=
  let man_speed_mps := (man_speed_kmph * 1000) / 3600
  let relative_speed_mps := train_length_m / cross_time_s
  let train_speed_mps := relative_speed_mps - man_speed_mps
  let train_speed_kmph := (train_speed_mps * 3600) / 1000
  train_speed_kmph

theorem train_speed_is_25_kmph : train_speed_kmph 270 2 36 = 25 := by
  sorry

end train_speed_is_25_kmph_l240_240398


namespace number_of_girls_l240_240119

variable (boys : ℕ) (total_children : ℕ)

theorem number_of_girls (h1 : boys = 40) (h2 : total_children = 117) : total_children - boys = 77 :=
by
  sorry

end number_of_girls_l240_240119


namespace fraction_to_decimal_l240_240253

theorem fraction_to_decimal :
  (7 / 125 : ℚ) = 0.056 :=
sorry

end fraction_to_decimal_l240_240253


namespace regular_polygon_sides_l240_240942

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240942


namespace regular_polygon_sides_l240_240855

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240855


namespace solve_inequality_l240_240322

theorem solve_inequality (x : ℝ) : 
  (let a := log 2 (x^6),
       b := log (1/2) (x^2),
       c := log (1/2) (x^6),
       d := 8 : ℝ,
       e := (log (1/2) (x^2))^3
   in  (a * b - c - 8 * log 2 (x^2) + 2) / (8 + e) ≤ 0) ↔ 
   x ∈ set.Icc (-2 : ℝ) (-2 ^ (1/6)) ∪ set.Ico (-1/2) 0 ∪ set.Ioc 0 (1/2) ∪ set.Icc (2 ^ (1/6)) 2 :=
by sorry

end solve_inequality_l240_240322


namespace boat_shipments_divisor_l240_240656

/-- 
Given:
1. There exists an integer B representing the number of boxes that can be divided into S equal shipments by boat.
2. B can be divided into 24 equal shipments by truck.
3. The smallest number of boxes B is 120.
Prove that S, the number of equal shipments by boat, is 60.
--/
theorem boat_shipments_divisor (B S : ℕ) (h1 : B % S = 0) (h2 : B % 24 = 0) (h3 : B = 120) : S = 60 := 
sorry

end boat_shipments_divisor_l240_240656


namespace polynomial_q_correct_l240_240168

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct_l240_240168


namespace regular_polygon_sides_l240_240877

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240877


namespace hyperbola_range_of_m_l240_240728

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (1 + m) + (y^2) / (1 - m) = 1) → 
  (m < -1 ∨ m > 1) :=
by 
sorry

end hyperbola_range_of_m_l240_240728


namespace regular_polygon_sides_l240_240951

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240951


namespace radius_of_circumscribed_circle_l240_240357

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l240_240357


namespace hundred_squared_plus_two_hundred_one_is_composite_l240_240188

theorem hundred_squared_plus_two_hundred_one_is_composite : 
    ¬ Prime (100^2 + 201) :=
by {
  sorry
}

end hundred_squared_plus_two_hundred_one_is_composite_l240_240188


namespace expected_winnings_l240_240389

theorem expected_winnings :
  let p_heads : ℚ := 1 / 4
  let p_tails : ℚ := 1 / 2
  let p_edge : ℚ := 1 / 4
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let loss_edge : ℚ := -8
  (p_heads * win_heads + p_tails * win_tails + p_edge * loss_edge) = -0.25 := 
by sorry

end expected_winnings_l240_240389


namespace correct_statements_l240_240962

-- A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test.
def statement1 := false -- This statement is incorrect because this is systematic sampling, not stratified sampling.

-- In the frequency distribution histogram, the sum of the areas of all small rectangles is 1.
def statement2 := true -- This is correct.

-- In the regression line equation \(\hat{y} = 0.2x + 12\), when the variable \(x\) increases by one unit, the variable \(y\) definitely increases by 0.2 units.
def statement3 := false -- This is incorrect because y increases on average by 0.2 units, not definitely.

-- For two categorical variables \(X\) and \(Y\), calculating the statistic \(K^2\) and its observed value \(k\), the larger the observed value \(k\), the more confident we are that “X and Y are related”.
def statement4 := true -- This is correct.

-- We need to prove that the correct statements are only statement2 and statement4.
theorem correct_statements : (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) → (statement2 ∧ statement4) :=
by sorry

end correct_statements_l240_240962


namespace height_of_frustum_l240_240540

-- Definitions based on the given conditions
def cuts_parallel_to_base (height: ℕ) (ratio: ℕ) : ℕ := 
  height * ratio

-- Define the problem
theorem height_of_frustum 
  (height_smaller_pyramid : ℕ) 
  (ratio_upper_to_lower: ℕ) 
  (h : height_smaller_pyramid = 3) 
  (r : ratio_upper_to_lower = 4) 
  : (cuts_parallel_to_base 3 2) - height_smaller_pyramid = 3 := 
by
  sorry

end height_of_frustum_l240_240540


namespace sum_of_T_l240_240306

noncomputable def T : finset ℕ :=
  finset.Icc (2^4) (2^5 - 1)

theorem sum_of_T :
  ∑ x in T, x = 376 :=
by
  sorry

end sum_of_T_l240_240306


namespace regular_polygon_sides_l240_240923

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240923


namespace atomic_weight_Ca_l240_240700

def molecular_weight_CaH2 : ℝ := 42
def atomic_weight_H : ℝ := 1.008

theorem atomic_weight_Ca : atomic_weight_H * 2 < molecular_weight_CaH2 :=
by sorry

end atomic_weight_Ca_l240_240700


namespace cost_of_camel_proof_l240_240823

noncomputable def cost_of_camel (C H O E : ℕ) : ℕ :=
  if 10 * C = 24 * H ∧ 16 * H = 4 * O ∧ 6 * O = 4 * E ∧ 10 * E = 120000 then 4800 else 0

theorem cost_of_camel_proof (C H O E : ℕ) 
  (h1 : 10 * C = 24 * H) (h2 : 16 * H = 4 * O) (h3 : 6 * O = 4 * E) (h4 : 10 * E = 120000) :
  cost_of_camel C H O E = 4800 :=
by
  sorry

end cost_of_camel_proof_l240_240823


namespace simplify_expression_l240_240207

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  (3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6)) / ((x - 3) * (x + 2)) :=
by
  sorry

end simplify_expression_l240_240207


namespace expand_expression_l240_240977

theorem expand_expression : ∀ (x : ℝ), (17 * x + 21) * 3 * x = 51 * x^2 + 63 * x :=
by
  intro x
  sorry

end expand_expression_l240_240977


namespace equivalent_single_discount_l240_240132

variable (original_price : ℝ)
variable (first_discount : ℝ)
variable (second_discount : ℝ)

-- Conditions
def sale_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)

def final_price (p : ℝ) (d1 d2 : ℝ) : ℝ :=
  let sale1 := sale_price p d1
  sale_price sale1 d2

-- Prove the equivalent single discount is as described
theorem equivalent_single_discount :
  original_price = 30 → first_discount = 0.2 → second_discount = 0.25 →
  (1 - final_price original_price first_discount second_discount / original_price) * 100 = 40 :=
by
  intros
  sorry

end equivalent_single_discount_l240_240132


namespace find_d_l240_240571

theorem find_d (c a m d : ℝ) (h : m = (c * a * d) / (a - d)) : d = (m * a) / (m + c * a) :=
by sorry

end find_d_l240_240571


namespace find_percentage_l240_240233

theorem find_percentage (p : ℝ) (h : (p / 100) * 8 = 0.06) : p = 0.75 := 
by 
  sorry

end find_percentage_l240_240233


namespace trigonometric_identity_l240_240567

theorem trigonometric_identity 
  (deg7 deg37 deg83 : ℝ)
  (h7 : deg7 = 7) 
  (h37 : deg37 = 37) 
  (h83 : deg83 = 83) 
  : (Real.sin (deg7 * Real.pi / 180) * Real.cos (deg37 * Real.pi / 180) - Real.sin (deg83 * Real.pi / 180) * Real.sin (deg37 * Real.pi / 180) = -1/2) :=
sorry

end trigonometric_identity_l240_240567


namespace factorize_poly_l240_240521

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l240_240521


namespace percent_of_12356_equals_1_2356_l240_240658

theorem percent_of_12356_equals_1_2356 (p : ℝ) (h : p * 12356 = 1.2356) : p = 0.0001 := sorry

end percent_of_12356_equals_1_2356_l240_240658


namespace range_of_alpha_l240_240987

open Real

theorem range_of_alpha 
  (α : ℝ) (k : ℤ) :
  (sin α > 0) ∧ (cos α < 0) ∧ (sin α > cos α) →
  (∃ k : ℤ, (2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) ∨ 
  (2 * k * π + (3 * π / 2) < α ∧ α < 2 * k * π + 2 * π)) := 
by 
  sorry

end range_of_alpha_l240_240987


namespace solve_f_sqrt_2009_l240_240064

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_never_zero : ∀ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

theorem solve_f_sqrt_2009 :
  f (sqrt 2009) = 1 / 2009 := sorry

end solve_f_sqrt_2009_l240_240064


namespace mathematicians_probabilities_l240_240327

theorem mathematicians_probabilities:
  (let p1_b1 := 2 in let t1 := 5 in
   let p2_b1 := 3 in let t2 := 8 in
   let P1 := p1_b1 + p2_b1 in let T1 := t1 + t2 in
   P1 / T1 = 5 / 13) ∧
  (let p1_b2 := 4 in let t1 := 10 in
   let p2_b2 := 3 in let t2 := 8 in
   let P2 := p1_b2 + p2_b2 in let T2 := t1 + t2 in
   P2 / T2 = 7 / 18) ∧
  (let lb := (3 : ℚ) / 8 in let ub := (2 : ℚ) / 5 in let p3 := (17 : ℚ) / 40 in
   ¬ (lb < p3 ∧ p3 < ub)) :=
by {
  split;
  {
    let p1_b1 := 2;
    let t1 := 5;
    let p2_b1 := 3;
    let t2 := 8;
    let P1 := p1_b1 + p2_b1;
    let T1 := t1 + t2;
    exact P1 / T1 = 5 / 13;
  },
  {
    let p1_b2 := 4;
    let t1 := 10;
    let p2_b2 := 3;
    let t2 := 8;
    let P2 := p1_b2 + p2_b2;
    let T2 := t1 + t2;
    exact P2 / T2 = 7 / 18;
  },
  {
    let lb := (3 : ℚ) / 8;
    let ub := (2 : ℚ) / 5;
    let p3 := (17 : ℚ) / 40;
    exact ¬ (lb < p3 ∧ p3 < ub);
  }
}

end mathematicians_probabilities_l240_240327


namespace rectangle_perimeter_l240_240131

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 2 * (2 * a + 2 * b)) : 2 * (a + b) = 36 :=
by
  sorry

end rectangle_perimeter_l240_240131


namespace regular_polygon_sides_l240_240847

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240847


namespace passengers_got_on_in_Texas_l240_240400

theorem passengers_got_on_in_Texas (start_pax : ℕ) 
  (texas_depart_pax : ℕ) 
  (nc_depart_pax : ℕ) 
  (nc_board_pax : ℕ) 
  (virginia_total_people : ℕ) 
  (crew_members : ℕ) 
  (final_pax_virginia : ℕ) 
  (X : ℕ) :
  start_pax = 124 →
  texas_depart_pax = 58 →
  nc_depart_pax = 47 →
  nc_board_pax = 14 →
  virginia_total_people = 67 →
  crew_members = 10 →
  final_pax_virginia = virginia_total_people - crew_members →
  X + 33 = final_pax_virginia →
  X = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end passengers_got_on_in_Texas_l240_240400


namespace num_divisors_8_factorial_l240_240016

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l240_240016


namespace root_abs_sum_l240_240262

-- Definitions and conditions
variable (p q r n : ℤ)
variable (h_root : (x^3 - 2018 * x + n).coeffs[0] = 0)  -- This needs coefficient definition (simplified for clarity)
variable (h_vieta1 : p + q + r = 0)
variable (h_vieta2 : p * q + q * r + r * p = -2018)

theorem root_abs_sum :
  |p| + |q| + |r| = 100 :=
sorry

end root_abs_sum_l240_240262


namespace a_5_value_l240_240434

noncomputable def seq : ℕ → ℤ
| 0       => 1
| (n + 1) => (seq n) ^ 2 - 1

theorem a_5_value : seq 4 = -1 :=
by
  sorry

end a_5_value_l240_240434


namespace average_of_w_x_z_l240_240383

theorem average_of_w_x_z (w x z y a : ℝ) (h1 : 2 / w + 2 / x + 2 / z = 2 / y)
  (h2 : w * x * z = y) (h3 : w + x + z = a) : (w + x + z) / 3 = a / 3 :=
by sorry

end average_of_w_x_z_l240_240383


namespace recurring_decimal_mul_seven_l240_240966

-- Declare the repeating decimal as a definition
def recurring_decimal_0_3 : ℚ := 1 / 3

-- Theorem stating that the product of 0.333... and 7 is 7/3
theorem recurring_decimal_mul_seven : recurring_decimal_0_3 * 7 = 7 / 3 :=
by
  -- Insert proof here
  sorry

end recurring_decimal_mul_seven_l240_240966


namespace number_of_divisors_8_factorial_l240_240037

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l240_240037


namespace john_plays_periods_l240_240059

theorem john_plays_periods
  (PointsPer4Minutes : ℕ := 7)
  (PeriodDurationMinutes : ℕ := 12)
  (TotalPoints : ℕ := 42) :
  (TotalPoints / PointsPer4Minutes) / (PeriodDurationMinutes / 4) = 2 := by
  sorry

end john_plays_periods_l240_240059


namespace arithmetic_sequence_sum_l240_240738

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 4 + a 8 = 4 →
  S 11 + a 6 = 24 :=
by
  intros a S h1 h2
  sorry

end arithmetic_sequence_sum_l240_240738


namespace cube_plane_probability_l240_240162

theorem cube_plane_probability : 
  let n := 8
  let k := 4
  let total_ways := Nat.choose n k
  let favorable_ways := 12
  ∃ p : ℚ, p = favorable_ways / total_ways ∧ p = 6 / 35 :=
sorry

end cube_plane_probability_l240_240162


namespace modulo_multiplication_l240_240324

theorem modulo_multiplication (m : ℕ) (h : 0 ≤ m ∧ m < 50) :
  152 * 936 % 50 = 22 :=
by
  sorry

end modulo_multiplication_l240_240324


namespace range_of_f_l240_240443

noncomputable def f (x : ℝ) : ℝ := 2^(x + 2) - 4^x

def domain_M (x : ℝ) : Prop := 1 < x ∧ x < 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, domain_M x ∧ f x = y) ↔ -32 < y ∧ y < 4 :=
sorry

end range_of_f_l240_240443


namespace regular_polygon_sides_l240_240952

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240952


namespace find_all_functions_l240_240979

noncomputable def is_solution (f : ℕ → ℕ) : Prop :=
  f(1) = 1 ∧
  (∀ n, f(n + 2) + (n^2 + 4*n + 3) * f(n) = (2*n + 5) * f(n + 1)) ∧
  (∀ m n, m > n → f(n) ∣ f(m))

def factorial_solution (f : ℕ → ℕ) := 
  ∀ n, f(n) = n! ∨ f(n) = (n+2)! / 6

theorem find_all_functions (f : ℕ → ℕ) (n : ℕ) :
  is_solution f → factorial_solution f := by
  sorry

end find_all_functions_l240_240979


namespace regular_polygon_sides_l240_240865

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240865


namespace avery_work_time_l240_240057

theorem avery_work_time :
  ∀ (t : ℝ),
    (1/2 * t + 1/4 * 1 = 1) → t = 1 :=
by
  intros t h
  sorry

end avery_work_time_l240_240057


namespace triangle_angle_contradiction_l240_240815

theorem triangle_angle_contradiction (a b c : ℝ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : a + b + c = 180) (h₃ : 60 < a ∧ 60 < b ∧ 60 < c) : false :=
by
  sorry

end triangle_angle_contradiction_l240_240815


namespace sum_of_interior_angles_of_regular_polygon_l240_240778

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : n = 360 / 20) :
  (∑ i in finset.range n, 180 - 360 / n) = 2880 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l240_240778


namespace regular_polygon_sides_l240_240926

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240926


namespace sidney_thursday_jacks_l240_240199

open Nat

-- Define the number of jumping jacks Sidney did on each day
def monday_jacks := 20
def tuesday_jacks := 36
def wednesday_jacks := 40

-- Define the total number of jumping jacks done by Sidney
-- on Monday, Tuesday, and Wednesday
def sidney_mon_wed_jacks := monday_jacks + tuesday_jacks + wednesday_jacks

-- Define the total number of jumping jacks done by Brooke
def brooke_jacks := 438

-- Define the relationship between Brooke's and Sidney's total jumping jacks
def sidney_total_jacks := brooke_jacks / 3

-- Prove the number of jumping jacks Sidney did on Thursday
theorem sidney_thursday_jacks :
  sidney_total_jacks - sidney_mon_wed_jacks = 50 :=
by
  sorry

end sidney_thursday_jacks_l240_240199


namespace regular_polygon_sides_l240_240882

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240882


namespace determine_right_triangle_l240_240613

variable (A B C : ℝ)
variable (AB BC AC : ℝ)

-- Conditions as definitions
def condition1 : Prop := A + C = B
def condition2 : Prop := A = 30 ∧ B = 60 ∧ C = 90 -- Since ratio 1:2:3 means A = 30, B = 60, C = 90

-- Proof problem statement
theorem determine_right_triangle (h1 : condition1 A B C) (h2 : condition2 A B C) : (B = 90) :=
sorry

end determine_right_triangle_l240_240613


namespace circle_radius_of_square_perimeter_eq_area_l240_240353

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l240_240353


namespace miss_hilt_apples_l240_240312

theorem miss_hilt_apples (h : ℕ) (a_per_hour : ℕ) (total_apples : ℕ) 
    (H1 : a_per_hour = 5) (H2 : total_apples = 15) (H3 : total_apples = h * a_per_hour) : 
  h = 3 :=
by
  sorry

end miss_hilt_apples_l240_240312


namespace total_kids_played_with_l240_240061

-- Define the conditions as separate constants
def kidsMonday : Nat := 12
def kidsTuesday : Nat := 7

-- Prove the total number of kids Julia played with
theorem total_kids_played_with : kidsMonday + kidsTuesday = 19 := 
by
  sorry

end total_kids_played_with_l240_240061


namespace largest_divisor_even_triplet_l240_240062

theorem largest_divisor_even_triplet :
  ∀ (n : ℕ), 24 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) :=
by intros; sorry

end largest_divisor_even_triplet_l240_240062


namespace find_D_l240_240745

-- Definitions from conditions
def is_different (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

-- The proof problem
theorem find_D (A B C D : ℕ) (h_diff: is_different A B C D) (h_eq : 700 + 10 * A + 5 + 100 * B + 70 + C = 100 * D + 38) : D = 9 :=
sorry

end find_D_l240_240745


namespace smallest_of_seven_consecutive_even_numbers_l240_240765

theorem smallest_of_seven_consecutive_even_numbers (a b c d e f g : ℤ)
  (h₁ : a + b + c + d + e + f + g = 700)
  (h₂ : b = a + 2)
  (h₃ : c = a + 4)
  (h₄ : d = a + 6)
  (h₅ : e = a + 8)
  (h₆ : f = a + 10)
  (h₇ : g = a + 12)
  : a = 94 :=
by
  -- Proof is omitted, this is just the statement.
  sorry

end smallest_of_seven_consecutive_even_numbers_l240_240765


namespace regular_polygon_sides_l240_240884

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240884


namespace negation_universal_proposition_l240_240645

theorem negation_universal_proposition :
  (¬∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end negation_universal_proposition_l240_240645


namespace roots_of_quadratic_eq_l240_240787

theorem roots_of_quadratic_eq (x : ℝ) : x^2 - 2 * x = 0 → (x = 0 ∨ x = 2) :=
by sorry

end roots_of_quadratic_eq_l240_240787


namespace units_digit_17_pow_31_l240_240666

theorem units_digit_17_pow_31 : (17 ^ 31) % 10 = 3 := by
  sorry

end units_digit_17_pow_31_l240_240666


namespace max_area_quad_l240_240460

noncomputable def MaxAreaABCD : ℝ :=
  let x : ℝ := 3
  let θ : ℝ := Real.pi / 2
  let φ : ℝ := Real.pi
  let area_ABC := (1/2) * x * 3 * Real.sin θ
  let area_BCD := (1/2) * 3 * 5 * Real.sin (φ - θ)
  area_ABC + area_BCD

theorem max_area_quad (x : ℝ) (h : x > 0)
  (BC_eq_3 : True)
  (CD_eq_5 : True)
  (centroids_form_isosceles : True) :
  MaxAreaABCD = 12 := by
  sorry

end max_area_quad_l240_240460


namespace power_difference_divisible_l240_240287

-- Define the variables and conditions
variables {a b c : ℤ} {n : ℕ}

-- Condition: a - b is divisible by c
def is_divisible (a b c : ℤ) : Prop := ∃ k : ℤ, a - b = k * c

-- Lean proof statement
theorem power_difference_divisible {a b c : ℤ} {n : ℕ} (h : is_divisible a b c) : c ∣ (a^n - b^n) :=
  sorry

end power_difference_divisible_l240_240287


namespace time_for_one_kid_to_wash_six_whiteboards_l240_240267

-- Define the conditions as a function
def time_taken (k : ℕ) (w : ℕ) : ℕ := 20 * 4 * w / k

theorem time_for_one_kid_to_wash_six_whiteboards :
  time_taken 1 6 = 160 := by
-- Proof omitted
sorry

end time_for_one_kid_to_wash_six_whiteboards_l240_240267


namespace negation_of_P_equiv_l240_240590

-- Define the proposition P
def P : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of P equivalently
theorem negation_of_P_equiv :
  ¬ P ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := 
sorry

end negation_of_P_equiv_l240_240590


namespace num_combinations_l240_240744

theorem num_combinations (backpacks pencil_cases : ℕ) (h_backpacks : backpacks = 2) (h_pencil_cases : pencil_cases = 2) : 
  backpacks * pencil_cases = 4 :=
by
  rw [h_backpacks, h_pencil_cases]
  exact Nat.mul_self_eq 2

end num_combinations_l240_240744


namespace find_first_term_of_sequence_l240_240499

theorem find_first_term_of_sequence (a : ℕ → ℝ)
  (h_rec : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h_a8 : a 8 = 2) :
  a 1 = 1 / 2 :=
sorry

end find_first_term_of_sequence_l240_240499


namespace stream_speed_l240_240238

/-- The speed of the stream problem -/
theorem stream_speed 
    (b s : ℝ) 
    (downstream_time : ℝ := 3)
    (upstream_time : ℝ := 3)
    (downstream_distance : ℝ := 60)
    (upstream_distance : ℝ := 30)
    (h1 : downstream_distance = (b + s) * downstream_time)
    (h2 : upstream_distance = (b - s) * upstream_time) : 
    s = 5 := 
by {
  -- The proof can be filled here
  sorry
}

end stream_speed_l240_240238


namespace solve_first_equation_solve_second_equation_l240_240202

theorem solve_first_equation (x : ℝ) : (8 * x = -2 * (x + 5)) → (x = -1) :=
by
  intro h
  sorry

theorem solve_second_equation (x : ℝ) : ((x - 1) / 4 = (5 * x - 7) / 6 + 1) → (x = -1 / 7) :=
by
  intro h
  sorry

end solve_first_equation_solve_second_equation_l240_240202


namespace rent_increase_percentage_l240_240236

theorem rent_increase_percentage (a x: ℝ) (h1: a ≠ 0) (h2: (9 / 10) * a = (4 / 5) * a * (1 + x / 100)) : x = 12.5 :=
sorry

end rent_increase_percentage_l240_240236


namespace zach_needs_more_money_l240_240523

noncomputable def cost_of_bike : ℕ := 100
noncomputable def weekly_allowance : ℕ := 5
noncomputable def mowing_income : ℕ := 10
noncomputable def babysitting_rate_per_hour : ℕ := 7
noncomputable def initial_savings : ℕ := 65
noncomputable def hours_babysitting : ℕ := 2

theorem zach_needs_more_money : 
  cost_of_bike - (initial_savings + weekly_allowance + mowing_income + (babysitting_rate_per_hour * hours_babysitting)) = 6 :=
by
  sorry

end zach_needs_more_money_l240_240523


namespace fg_difference_l240_240438

def f (x : ℝ) : ℝ := x^2 - 4 * x + 7
def g (x : ℝ) : ℝ := x + 4

theorem fg_difference : f (g 3) - g (f 3) = 20 :=
by
  sorry

end fg_difference_l240_240438


namespace satisfies_differential_eqn_l240_240198

noncomputable def y (x : ℝ) : ℝ := 5 * Real.exp (-2 * x) + (1 / 3) * Real.exp x

theorem satisfies_differential_eqn : ∀ x : ℝ, (deriv y x) + 2 * y x = Real.exp x :=
by
  -- The proof is to be provided
  sorry

end satisfies_differential_eqn_l240_240198


namespace solve_for_x_l240_240452

theorem solve_for_x (x : ℝ) (h : x + 2 = 7) : x = 5 := 
by
  sorry

end solve_for_x_l240_240452


namespace sum_of_interior_angles_l240_240777

theorem sum_of_interior_angles (h : ∀ (n : ℕ), 360 / 20 = n) : 
  ∃ (s : ℕ), s = 2880 :=
by
  have n := 360 / 20
  have sum := 180 * (n - 2)
  use sum
  sorry

end sum_of_interior_angles_l240_240777


namespace find_inverse_sum_l240_240467

def f (x : ℝ) : ℝ := x * |x|^2

theorem find_inverse_sum :
  (∃ x : ℝ, f x = 8) ∧ (∃ y : ℝ, f y = -64) → 
  (∃ a b : ℝ, f a = 8 ∧ f b = -64 ∧ a + b = 6) :=
sorry

end find_inverse_sum_l240_240467


namespace car_robot_collections_l240_240803

variable (t m b s j : ℕ)

axiom tom_has_15 : t = 15
axiom michael_robots : m = 3 * t - 5
axiom bob_robots : b = 8 * (t + m)
axiom sarah_robots : s = b / 2 - 7
axiom jane_robots : j = (s - t) / 3

theorem car_robot_collections :
  t = 15 ∧
  m = 40 ∧
  b = 440 ∧
  s = 213 ∧
  j = 66 :=
  by
    sorry

end car_robot_collections_l240_240803


namespace prove_x_value_l240_240256

-- Definitions of the conditions
variable (x y z w : ℕ)
variable (h1 : x = y + 8)
variable (h2 : y = z + 15)
variable (h3 : z = w + 25)
variable (h4 : w = 90)

-- The goal is to prove x = 138 given the conditions
theorem prove_x_value : x = 138 := by
  sorry

end prove_x_value_l240_240256


namespace regular_polygon_sides_l240_240868

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240868


namespace parallel_condition_perpendicular_condition_l240_240007

theorem parallel_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → a.1 * b.2 = a.2 * b.1 → x = 2 := 
sorry

theorem perpendicular_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → ((a.1 - b.1) * b.1 + (a.2 - b.2) * b.2) = 0 → x = 1 / 3 :=
sorry

end parallel_condition_perpendicular_condition_l240_240007


namespace new_machine_rate_l240_240676

def old_machine_rate : ℕ := 100
def total_bolts : ℕ := 500
def time_hours : ℕ := 2

theorem new_machine_rate (R : ℕ) : 
  (old_machine_rate * time_hours + R * time_hours = total_bolts) → 
  R = 150 := 
by
  sorry

end new_machine_rate_l240_240676


namespace regular_polygon_sides_l240_240879

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240879


namespace builder_total_amount_paid_l240_240535

theorem builder_total_amount_paid :
  let cost_drill_bits := 5 * 6
  let tax_drill_bits := 0.10 * cost_drill_bits
  let total_cost_drill_bits := cost_drill_bits + tax_drill_bits

  let cost_hammers := 3 * 8
  let discount_hammers := 0.05 * cost_hammers
  let total_cost_hammers := cost_hammers - discount_hammers

  let cost_toolbox := 25
  let tax_toolbox := 0.15 * cost_toolbox
  let total_cost_toolbox := cost_toolbox + tax_toolbox

  let total_amount_paid := total_cost_drill_bits + total_cost_hammers + total_cost_toolbox

  total_amount_paid = 84.55 :=
by
  sorry

end builder_total_amount_paid_l240_240535


namespace regular_polygon_sides_l240_240851

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240851


namespace regular_polygon_sides_l240_240940

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240940


namespace inversely_proportional_y_ratio_l240_240483

variable {k : ℝ}
variable {x₁ x₂ y₁ y₂ : ℝ}
variable (h_inv_prop : ∀ (x y : ℝ), x * y = k)
variable (hx₁x₂ : x₁ ≠ 0 ∧ x₂ ≠ 0)
variable (hy₁y₂ : y₁ ≠ 0 ∧ y₂ ≠ 0)
variable (hx_ratio : x₁ / x₂ = 3 / 4)

theorem inversely_proportional_y_ratio :
  y₁ / y₂ = 4 / 3 :=
by
  sorry

end inversely_proportional_y_ratio_l240_240483


namespace regular_pentagon_cannot_tessellate_l240_240964

-- Definitions of polygons
def is_regular_triangle (angle : ℝ) : Prop := angle = 60
def is_square (angle : ℝ) : Prop := angle = 90
def is_regular_pentagon (angle : ℝ) : Prop := angle = 108
def is_hexagon (angle : ℝ) : Prop := angle = 120

-- Tessellation condition
def divides_evenly (a b : ℝ) : Prop := ∃ k : ℕ, b = k * a

-- The main statement
theorem regular_pentagon_cannot_tessellate :
  ¬ divides_evenly 108 360 :=
sorry

end regular_pentagon_cannot_tessellate_l240_240964


namespace nonnegative_fraction_interval_l240_240431

theorem nonnegative_fraction_interval : 
  ∀ x : ℝ, (0 ≤ x ∧ x < 3) ↔ (0 ≤ (x - 15 * x^2 + 36 * x^3) / (9 - x^3)) := by
sorry

end nonnegative_fraction_interval_l240_240431


namespace factorize_1_factorize_2_factorize_3_l240_240978

-- Problem 1: Factorize 3a^3 - 6a^2 + 3a
theorem factorize_1 (a : ℝ) : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
sorry

-- Problem 2: Factorize a^2(x - y) + b^2(y - x)
theorem factorize_2 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
sorry

-- Problem 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factorize_3 (a b : ℝ) : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
sorry

end factorize_1_factorize_2_factorize_3_l240_240978


namespace regular_polygon_sides_l240_240854

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240854


namespace axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l240_240614

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l240_240614


namespace polyhedron_edges_faces_vertices_l240_240629

theorem polyhedron_edges_faces_vertices
  (E F V n m : ℕ)
  (h1 : n * F = 2 * E)
  (h2 : m * V = 2 * E)
  (h3 : V + F = E + 2) :
  ¬(m * F = 2 * E) :=
sorry

end polyhedron_edges_faces_vertices_l240_240629


namespace regular_polygon_sides_l240_240896

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240896


namespace solve_system_of_inequalities_l240_240323

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x - 2 > 0) ∧ (3 * (x - 1) - 7 < -2 * x) → 1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l240_240323


namespace sum_of_two_longest_altitudes_l240_240041

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l240_240041


namespace students_in_hollow_square_are_160_l240_240126

-- Define the problem conditions
def hollow_square_formation (outer_layer : ℕ) (inner_layer : ℕ) : Prop :=
  outer_layer = 52 ∧ inner_layer = 28

-- Define the total number of students in the group based on the given condition
def total_students (n : ℕ) : Prop := n = 160

-- Prove that the total number of students is 160 given the hollow square formation conditions
theorem students_in_hollow_square_are_160 : ∀ (outer_layer inner_layer : ℕ),
  hollow_square_formation outer_layer inner_layer → total_students 160 :=
by
  intros outer_layer inner_layer h
  sorry

end students_in_hollow_square_are_160_l240_240126


namespace suitable_for_lottery_method_B_l240_240137

def total_items_A : Nat := 3000
def samples_A : Nat := 600

def total_items_B (n: Nat) : Nat := 2 * 15
def samples_B : Nat := 6

def total_items_C : Nat := 2 * 15
def samples_C : Nat := 6

def total_items_D : Nat := 3000
def samples_D : Nat := 10

def is_lottery_suitable (total_items : Nat) (samples : Nat) (different_factories : Bool) : Bool :=
  total_items <= 30 && samples <= total_items && !different_factories

theorem suitable_for_lottery_method_B : 
  is_lottery_suitable (total_items_B 2) samples_B false = true :=
  sorry

end suitable_for_lottery_method_B_l240_240137


namespace regular_polygon_sides_l240_240864

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240864


namespace arithmetic_sequence_inequality_l240_240439

variable {a : ℕ → ℝ}
variable {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality 
  (h : is_arithmetic_sequence a d)
  (d_pos : d ≠ 0)
  (a_pos : ∀ n, a n > 0) :
  (a 1) * (a 8) < (a 4) * (a 5) := 
by
  sorry

end arithmetic_sequence_inequality_l240_240439


namespace median_of_fifteen_is_eight_l240_240663

def median_of_first_fifteen_positive_integers : ℝ :=
  let lst := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median_pos := (list.length lst + 1) / 2  
  lst.get (median_pos - 1)

theorem median_of_fifteen_is_eight : median_of_first_fifteen_positive_integers = 8.0 := 
  by 
    -- Proof omitted    
    sorry

end median_of_fifteen_is_eight_l240_240663


namespace electric_car_travel_distance_l240_240828

theorem electric_car_travel_distance {d_electric d_diesel : ℕ} 
  (h1 : d_diesel = 120) 
  (h2 : d_electric = d_diesel + 50 * d_diesel / 100) : 
  d_electric = 180 := 
by 
  sorry

end electric_car_travel_distance_l240_240828


namespace andre_tuesday_ladybugs_l240_240549

theorem andre_tuesday_ladybugs (M T : ℕ) (dots_per_ladybug total_dots monday_dots tuesday_dots : ℕ)
  (h1 : M = 8)
  (h2 : dots_per_ladybug = 6)
  (h3 : total_dots = 78)
  (h4 : monday_dots = M * dots_per_ladybug)
  (h5 : tuesday_dots = total_dots - monday_dots)
  (h6 : tuesday_dots = T * dots_per_ladybug) :
  T = 5 :=
sorry

end andre_tuesday_ladybugs_l240_240549


namespace perpendicular_lines_l240_240513

section
variables {l1 l2 : Type} 

-- Condition A: Slope of l1 is 1, slope of l2 is 1
def condition_A (k1 k2 : ℝ) : Prop :=
  k1 = 1 ∧ k2 = 1

-- Condition B: Slope of l1 is -√3/3, l2 passes through points (2,0) and (3,√3)
def condition_B (k1 : ℝ) (A B : ℝ × ℝ) : Prop :=
  k1 = - (real.sqrt 3) / 3 ∧ A = (2,0) ∧ B = (3, real.sqrt 3)

-- Condition C: l1 passes through points (2,1) and (-4,-5), and l2 passes through (1,2) and (1,0)
def condition_C (P Q M N : ℝ × ℝ) : Prop :=
  P = (2,1) ∧ Q = (-4,-5) ∧ M = (-1,2) ∧ N = (1,0)
  
-- Condition D: Direction vectors
def condition_D (m : ℝ) : Prop :=
  (1, m) ∧ (1, -1 / m)

-- Perpendicular if and only if correct conditions are met (Option B, C, D)
theorem perpendicular_lines (k1 k2 : ℝ) (A B P Q M N : ℝ × ℝ) (m : ℝ) :
  (∃ k1 k2, condition_A k1 k2 ∧ ¬ (k1 * k2 = -1)) →
  (∃ k1 A B, condition_B k1 A B ∧ k1 * ((B.2 - A.2) / (B.1 - A.1)) = -1) →
  (∃ P Q M N, condition_C P Q M N ∧ (((Q.2 - P.2) / (Q.1 - P.1)) * ((N.2 - M.2) / (N.1 - M.1)) = -1)) →
  (∃ m, condition_D m ∧ (1 - (1 / m)) = 0) →
  true :=
by exactly sorry
end

end perpendicular_lines_l240_240513


namespace mathematicians_probabilities_l240_240328

theorem mathematicians_probabilities:
  (let p1_b1 := 2 in let t1 := 5 in
   let p2_b1 := 3 in let t2 := 8 in
   let P1 := p1_b1 + p2_b1 in let T1 := t1 + t2 in
   P1 / T1 = 5 / 13) ∧
  (let p1_b2 := 4 in let t1 := 10 in
   let p2_b2 := 3 in let t2 := 8 in
   let P2 := p1_b2 + p2_b2 in let T2 := t1 + t2 in
   P2 / T2 = 7 / 18) ∧
  (let lb := (3 : ℚ) / 8 in let ub := (2 : ℚ) / 5 in let p3 := (17 : ℚ) / 40 in
   ¬ (lb < p3 ∧ p3 < ub)) :=
by {
  split;
  {
    let p1_b1 := 2;
    let t1 := 5;
    let p2_b1 := 3;
    let t2 := 8;
    let P1 := p1_b1 + p2_b1;
    let T1 := t1 + t2;
    exact P1 / T1 = 5 / 13;
  },
  {
    let p1_b2 := 4;
    let t1 := 10;
    let p2_b2 := 3;
    let t2 := 8;
    let P2 := p1_b2 + p2_b2;
    let T2 := t1 + t2;
    exact P2 / T2 = 7 / 18;
  },
  {
    let lb := (3 : ℚ) / 8;
    let ub := (2 : ℚ) / 5;
    let p3 := (17 : ℚ) / 40;
    exact ¬ (lb < p3 ∧ p3 < ub);
  }
}

end mathematicians_probabilities_l240_240328


namespace complete_the_square_l240_240507

-- Definition of the initial condition
def eq1 : Prop := ∀ x : ℝ, x^2 + 4 * x + 1 = 0

-- The goal is to prove if the initial condition holds, then the desired result holds.
theorem complete_the_square (x : ℝ) (h : x^2 + 4 * x + 1 = 0) : (x + 2)^2 = 3 := by
  sorry

end complete_the_square_l240_240507


namespace find_minimal_positive_n_l240_240739

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Define the conditions
variables (a1 d : ℤ)
axiom condition_1 : arithmetic_seq a1 d 11 / arithmetic_seq a1 d 10 < -1
axiom condition_2 : ∃ n : ℕ, ∀ k : ℕ, k ≤ n → sum_arithmetic_seq a1 d k ≤ sum_arithmetic_seq a1 d n

-- Prove the statement
theorem find_minimal_positive_n : ∃ n : ℕ, n = 19 ∧ sum_arithmetic_seq a1 d n = 0 ∧
  (∀ m : ℕ, 0 < sum_arithmetic_seq a1 d m ∧ sum_arithmetic_seq a1 d m < sum_arithmetic_seq a1 d n) :=
sorry

end find_minimal_positive_n_l240_240739


namespace trays_needed_l240_240194

theorem trays_needed (cookies_classmates cookies_teachers cookies_per_tray : ℕ) 
  (hc1 : cookies_classmates = 276) 
  (hc2 : cookies_teachers = 92) 
  (hc3 : cookies_per_tray = 12) : 
  (cookies_classmates + cookies_teachers + cookies_per_tray - 1) / cookies_per_tray = 31 :=
by
  sorry

end trays_needed_l240_240194


namespace train_speed_in_kmph_l240_240397

-- Definitions for the given problem conditions
def length_of_train : ℝ := 110
def length_of_bridge : ℝ := 240
def time_to_cross_bridge : ℝ := 20.99832013438925

-- Main theorem statement
theorem train_speed_in_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60.0084 := 
by
  sorry

end train_speed_in_kmph_l240_240397


namespace regular_polygon_sides_l240_240830

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240830


namespace simon_change_l240_240764

def pansy_price : ℝ := 2.50
def pansy_count : ℕ := 5
def hydrangea_price : ℝ := 12.50
def hydrangea_count : ℕ := 1
def petunia_price : ℝ := 1.00
def petunia_count : ℕ := 5
def discount_rate : ℝ := 0.10
def initial_payment : ℝ := 50.00

theorem simon_change : 
  let total_cost := (pansy_count * pansy_price) + (hydrangea_count * hydrangea_price) + (petunia_count * petunia_price)
  let discount := total_cost * discount_rate
  let cost_after_discount := total_cost - discount
  let change := initial_payment - cost_after_discount
  change = 23.00 :=
by
  sorry

end simon_change_l240_240764


namespace savings_calculation_l240_240525

-- Define the conditions as given in the problem
def income_expenditure_ratio (income expenditure : ℝ) : Prop :=
  ∃ x : ℝ, income = 10 * x ∧ expenditure = 4 * x

def income_value : ℝ := 19000

-- The final statement for the savings, where we will prove the above question == answer
theorem savings_calculation (income expenditure savings : ℝ)
  (h_ratio : income_expenditure_ratio income expenditure)
  (h_income : income = income_value) : savings = 11400 :=
by
  sorry

end savings_calculation_l240_240525


namespace solve_problem_l240_240697

theorem solve_problem (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  7^m - 3 * 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := sorry

end solve_problem_l240_240697


namespace regular_polygon_sides_l240_240954

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240954


namespace power_difference_divisible_by_35_l240_240762

theorem power_difference_divisible_by_35 (n : ℕ) : (3^(6*n) - 2^(6*n)) % 35 = 0 := 
by sorry

end power_difference_divisible_by_35_l240_240762


namespace increasing_function_range_of_a_l240_240088

variable {f : ℝ → ℝ}

theorem increasing_function_range_of_a (a : ℝ) (h : ∀ x : ℝ, 3 * a * x^2 ≥ 0) : a > 0 :=
sorry

end increasing_function_range_of_a_l240_240088


namespace quadrilateral_sides_equality_l240_240089

theorem quadrilateral_sides_equality 
  (a b c d : ℕ) 
  (h1 : (b + c + d) % a = 0) 
  (h2 : (a + c + d) % b = 0) 
  (h3 : (a + b + d) % c = 0) 
  (h4 : (a + b + c) % d = 0) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equality_l240_240089


namespace radius_of_circumscribed_circle_l240_240343

noncomputable def circumscribed_circle_radius (r : ℝ) : Prop :=
  let s := r * Real.sqrt 2 in
  4 * s = π * r ^ 2 -> r = 4 * Real.sqrt 2 / π

-- Statement of the theorem to be proved
theorem radius_of_circumscribed_circle (r : ℝ) : circumscribed_circle_radius r := 
sorry

end radius_of_circumscribed_circle_l240_240343


namespace aftershave_alcohol_concentration_l240_240213

def initial_volume : ℝ := 12
def initial_concentration : ℝ := 0.60
def desired_concentration : ℝ := 0.40
def water_added : ℝ := 6
def final_volume : ℝ := initial_volume + water_added

theorem aftershave_alcohol_concentration :
  initial_concentration * initial_volume = desired_concentration * final_volume :=
by
  sorry

end aftershave_alcohol_concentration_l240_240213


namespace regular_polygon_sides_l240_240893

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240893


namespace cube_edge_length_close_to_six_l240_240045

theorem cube_edge_length_close_to_six
  (a V S : ℝ)
  (h1 : V = a^3)
  (h2 : S = 6 * a^2)
  (h3 : V = S + 1) : abs (a - 6) < 1 :=
by
  sorry

end cube_edge_length_close_to_six_l240_240045


namespace regular_polygon_sides_l240_240958

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240958


namespace cost_comparison_l240_240234

def cost_function_A (x : ℕ) : ℕ := 450 * x + 1000
def cost_function_B (x : ℕ) : ℕ := 500 * x

theorem cost_comparison (x : ℕ) : 
  if x = 20 then cost_function_A x = cost_function_B x 
  else if x < 20 then cost_function_A x > cost_function_B x 
  else cost_function_A x < cost_function_B x :=
sorry

end cost_comparison_l240_240234


namespace simplify_and_evaluate_l240_240201

-- Definitions and conditions 
def x := ℝ
def given_condition (x: ℝ) : Prop := x + 2 = Real.sqrt 2

-- The problem statement translated into Lean 4
theorem simplify_and_evaluate (x: ℝ) (h: given_condition x) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3 * x)) = Real.sqrt 2 - 1 :=
sorry

end simplify_and_evaluate_l240_240201


namespace complex_fourth_power_l240_240149

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l240_240149


namespace parallel_tangents_a3_plus_b2_plus_d_eq_seven_l240_240166

theorem parallel_tangents_a3_plus_b2_plus_d_eq_seven:
  ∃ (a b d : ℝ),
  (1, 1).snd = a * (1:ℝ)^3 + b * (1:ℝ)^2 + d ∧
  (-1, -3).snd = a * (-1:ℝ)^3 + b * (-1:ℝ)^2 + d ∧
  (3 * a * (1:ℝ)^2 + 2 * b * 1 = 3 * a * (-1:ℝ)^2 + 2 * b * -1) ∧
  a^3 + b^2 + d = 7 := 
sorry

end parallel_tangents_a3_plus_b2_plus_d_eq_seven_l240_240166


namespace find_C_coordinates_l240_240069

noncomputable def pointC_coordinates : Prop :=
  let A : (ℝ × ℝ) := (-2, 1)
  let B : (ℝ × ℝ) := (4, 9)
  ∃ C : (ℝ × ℝ), 
    (dist (A.1, A.2) (C.1, C.2) = 2 * dist (B.1, B.2) (C.1, C.2)) ∧ 
    C = (2, 19 / 3)

theorem find_C_coordinates : pointC_coordinates :=
  sorry

end find_C_coordinates_l240_240069


namespace bianca_picture_books_shelves_l240_240682

theorem bianca_picture_books_shelves (total_shelves : ℕ) (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 8 →
  mystery_shelves = 5 →
  total_books = 72 →
  total_shelves = (total_books - (mystery_shelves * books_per_shelf)) / books_per_shelf →
  total_shelves = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end bianca_picture_books_shelves_l240_240682


namespace part1_part2_l240_240285

-- Definitions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b + a + 3

-- First proof: When a = -1 and b = 10, prove 4A - (3A - 2B) = -45
theorem part1 : 4 * A (-1) 10 - (3 * A (-1) 10 - 2 * B (-1) 10) = -45 := by
  sorry

-- Second proof: If a and b are reciprocal, prove 4A - (3A - 2B) = 10
theorem part2 (a b : ℝ) (hab : a * b = 1) : 4 * A a b - (3 * A a b - 2 * B a b) = 10 := by
  sorry

end part1_part2_l240_240285


namespace district_B_high_schools_l240_240185

theorem district_B_high_schools :
  ∀ (total_schools public_schools parochial_schools private_schools districtA_schools districtB_private_schools: ℕ),
  total_schools = 50 ∧ 
  public_schools = 25 ∧ 
  parochial_schools = 16 ∧ 
  private_schools = 9 ∧ 
  districtA_schools = 18 ∧ 
  districtB_private_schools = 2 ∧ 
  (∃ districtC_schools, 
     districtC_schools = public_schools / 3 + parochial_schools / 3 + private_schools / 3) →
  ∃ districtB_schools, 
    districtB_schools = total_schools - districtA_schools - (public_schools / 3 + parochial_schools / 3 + private_schools / 3) ∧ 
    districtB_schools = 5 := by
  sorry

end district_B_high_schools_l240_240185


namespace largest_prime_factor_among_numbers_l240_240111

-- Definitions of the numbers with their prime factors
def num1 := 39
def num2 := 51
def num3 := 77
def num4 := 91
def num5 := 121

def prime_factors (n : ℕ) : List ℕ := sorry  -- Placeholder for the prime factors function

-- Prime factors for the given numbers
def factors_num1 := prime_factors num1
def factors_num2 := prime_factors num2
def factors_num3 := prime_factors num3
def factors_num4 := prime_factors num4
def factors_num5 := prime_factors num5

-- Extract the largest prime factor from a list of factors
def largest_prime_factor (factors : List ℕ) : ℕ := sorry  -- Placeholder for the largest_prime_factor function

-- Largest prime factors for each number
def largest_prime_factor_num1 := largest_prime_factor factors_num1
def largest_prime_factor_num2 := largest_prime_factor factors_num2
def largest_prime_factor_num3 := largest_prime_factor factors_num3
def largest_prime_factor_num4 := largest_prime_factor factors_num4
def largest_prime_factor_num5 := largest_prime_factor factors_num5

theorem largest_prime_factor_among_numbers :
  largest_prime_factor_num2 = 17 ∧
  largest_prime_factor_num1 = 13 ∧
  largest_prime_factor_num3 = 11 ∧
  largest_prime_factor_num4 = 13 ∧
  largest_prime_factor_num5 = 11 ∧
  (largest_prime_factor_num2 > largest_prime_factor_num1) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num3) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num4) ∧
  (largest_prime_factor_num2 > largest_prime_factor_num5)
:= by
  -- skeleton proof, details to be filled in
  sorry

end largest_prime_factor_among_numbers_l240_240111


namespace smallest_number_l240_240809

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l240_240809


namespace find_q_l240_240278

def f (q : ℝ) : ℝ := 3 * q - 3

theorem find_q (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end find_q_l240_240278


namespace pos_divisors_8_factorial_l240_240029

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l240_240029


namespace dave_hourly_wage_l240_240254

theorem dave_hourly_wage :
  ∀ (hours_monday hours_tuesday total_money : ℝ),
  hours_monday = 6 → hours_tuesday = 2 → total_money = 48 →
  (total_money / (hours_monday + hours_tuesday) = 6) :=
by
  intros hours_monday hours_tuesday total_money h_monday h_tuesday h_money
  sorry

end dave_hourly_wage_l240_240254


namespace regular_polygon_sides_l240_240835

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240835


namespace regular_polygon_sides_l240_240860

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240860


namespace polynomial_evaluation_l240_240628

def p (x : ℝ) (a b c d : ℝ) := x^4 + a * x^3 + b * x^2 + c * x + d

theorem polynomial_evaluation
  (a b c d : ℝ)
  (h1 : p 1 a b c d = 1993)
  (h2 : p 2 a b c d = 3986)
  (h3 : p 3 a b c d = 5979) :
  (1 / 4 : ℝ) * (p 11 a b c d + p (-7) a b c d) = 5233 := by
  sorry

end polynomial_evaluation_l240_240628


namespace regular_polygon_sides_l240_240833

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240833


namespace range_of_k_l240_240714

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, k^2 * x^2 + y^2 - 4 * k * x + 2 * k * y + k^2 - 1 = 0 ∧ (x, y) = (0, 0)) →
  0 < |k| ∧ |k| < 1 :=
by
  intros
  sorry

end range_of_k_l240_240714


namespace problem1_problem2_l240_240822

noncomputable def tan_inv_3_value : ℝ := -4 / 5

theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  Real.cos α ^ 2 - 3 * Real.sin α * Real.cos α = tan_inv_3_value := 
sorry

noncomputable def f (θ : ℝ) : ℝ := 
  (2 * Real.cos θ ^ 3 + Real.sin (2 * Real.pi - θ) ^ 2 + 
   Real.sin (Real.pi / 2 + θ) - 3) / 
  (2 + 2 * Real.cos (Real.pi + θ) ^ 2 + Real.cos (-θ))

theorem problem2 :
  f (Real.pi / 3) = -1 / 2 :=
sorry

end problem1_problem2_l240_240822


namespace pirate_15_gets_coins_l240_240125

def coins_required_for_pirates : ℕ :=
  Nat.factorial 14 * ((2 ^ 4) * (3 ^ 9)) / 15 ^ 14

theorem pirate_15_gets_coins :
  coins_required_for_pirates = 314928 := 
by sorry

end pirate_15_gets_coins_l240_240125


namespace pond_water_amount_l240_240826

theorem pond_water_amount : 
  let initial_water := 500 
  let evaporation_rate := 4
  let rain_amount := 2
  let days := 40
  initial_water - days * (evaporation_rate - rain_amount) = 420 :=
by
  sorry

end pond_water_amount_l240_240826


namespace num_divisors_fact8_l240_240033

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l240_240033


namespace parallel_lines_condition_l240_240821

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y: ℝ, (x + a * y + 6 = 0) ↔ ((a - 2) * x + 3 * y + 2 * a = 0)) ↔ a = -1 :=
by
  sorry

end parallel_lines_condition_l240_240821


namespace length_of_room_l240_240189

theorem length_of_room (Area Width Length : ℝ) (h1 : Area = 10) (h2 : Width = 2) (h3 : Area = Length * Width) : Length = 5 :=
by
  sorry

end length_of_room_l240_240189


namespace sum_x1_x2_eq_five_l240_240451

theorem sum_x1_x2_eq_five {x1 x2 : ℝ} 
  (h1 : 2^x1 = 5 - x1)
  (h2 : x2 + Real.log x2 / Real.log 2 = 5) : 
  x1 + x2 = 5 := 
sorry

end sum_x1_x2_eq_five_l240_240451


namespace game_a_greater_than_game_c_l240_240824

-- Definitions of probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probabilities for Game A and Game C based on given conditions
def prob_game_a : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)
def prob_game_c : ℚ :=
  (prob_heads ^ 5) +
  (prob_tails ^ 5) +
  (prob_heads ^ 3 * prob_tails ^ 2) +
  (prob_tails ^ 3 * prob_heads ^ 2)

-- Define the difference
def prob_difference : ℚ := prob_game_a - prob_game_c

-- The theorem to be proved
theorem game_a_greater_than_game_c :
  prob_difference = 3 / 64 :=
by
  sorry

end game_a_greater_than_game_c_l240_240824


namespace number_of_divisors_of_8_fact_l240_240035

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l240_240035


namespace question_correct_statements_l240_240182

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (f : ℝ → ℝ) : ∀ x y : ℝ, f (x + y) = f x + f y
axiom periodicity (f : ℝ → ℝ) : f 2 = 0

theorem question_correct_statements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- ensuring the function is periodic
  (∀ x : ℝ, f x = -f (-x)) ∧ -- ensuring the function is odd
  (∀ x : ℝ, f (x+2) = -f (-x)) :=  -- ensuring symmetry about point (1,0)
by
  -- We'll prove this using the conditions given and properties derived from it
  sorry 

end question_correct_statements_l240_240182


namespace initial_tickets_count_l240_240142

def spent_tickets : ℕ := 5
def additional_tickets : ℕ := 10
def current_tickets : ℕ := 16

theorem initial_tickets_count (initial_tickets : ℕ) :
  initial_tickets - spent_tickets + additional_tickets = current_tickets ↔ initial_tickets = 11 :=
by
  sorry

end initial_tickets_count_l240_240142


namespace find_integer_divisible_by_18_and_square_root_in_range_l240_240157

theorem find_integer_divisible_by_18_and_square_root_in_range :
  ∃ x : ℕ, 28 < Real.sqrt x ∧ Real.sqrt x < 28.2 ∧ 18 ∣ x ∧ x = 792 :=
by
  sorry

end find_integer_divisible_by_18_and_square_root_in_range_l240_240157


namespace cone_plane_distance_l240_240210

theorem cone_plane_distance (H α : ℝ) : 
  (x = 2 * H * (Real.sin (α / 4)) ^ 2) :=
sorry

end cone_plane_distance_l240_240210


namespace ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1_l240_240594

theorem ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1 :
  {p : ℕ × ℕ // 3 / p.1 + 6 / p.2 = 1}.toFinset.card = 6 :=
sorry

end ordered_pairs_sol_3_over_m_plus_6_over_n_eq_1_l240_240594


namespace regular_polygon_sides_l240_240936

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240936


namespace initial_cake_pieces_l240_240432

-- Define the initial number of cake pieces
variable (X : ℝ)

-- Define the conditions as assumptions
def cake_conditions (X : ℝ) : Prop :=
  0.60 * X + 3 * 32 = X 

theorem initial_cake_pieces (X : ℝ) (h : cake_conditions X) : X = 240 := sorry

end initial_cake_pieces_l240_240432


namespace solve_for_x_l240_240179

theorem solve_for_x {x : ℝ} (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end solve_for_x_l240_240179


namespace simplify_fractions_l240_240321

theorem simplify_fractions:
  (3 / 462 : ℚ) + (28 / 42 : ℚ) = 311 / 462 := sorry

end simplify_fractions_l240_240321


namespace bumper_cars_line_l240_240143

theorem bumper_cars_line (initial in_line_leaving newcomers : ℕ) 
  (h_initial : initial = 9)
  (h_leaving : in_line_leaving = 6)
  (h_newcomers : newcomers = 3) :
  initial - in_line_leaving + newcomers = 6 :=
by
  sorry

end bumper_cars_line_l240_240143


namespace slope_of_AB_l240_240572

theorem slope_of_AB (A B : (ℕ × ℕ)) (hA : A = (3, 4)) (hB : B = (2, 3)) : 
  (B.2 - A.2) / (B.1 - A.1) = 1 := 
by 
  sorry

end slope_of_AB_l240_240572


namespace integral_1_integral_2_integral_3_integral_4_integral_5_l240_240407
open Real

-- Integral 1
theorem integral_1 : ∫ (x : ℝ), sin x * cos x ^ 3 = -1 / 4 * cos x ^ 4 + C :=
by sorry

-- Integral 2
theorem integral_2 : ∫ (x : ℝ), 1 / ((1 + sqrt x) * sqrt x) = 2 * log (1 + sqrt x) + C :=
by sorry

-- Integral 3
theorem integral_3 : ∫ (x : ℝ), x ^ 2 * sqrt (x ^ 3 + 1) = 2 / 9 * (x ^ 3 + 1) ^ (3/2) + C :=
by sorry

-- Integral 4
theorem integral_4 : ∫ (x : ℝ), (exp (2 * x) - 3 * exp x) / exp x = exp x - 3 * x + C :=
by sorry

-- Integral 5
theorem integral_5 : ∫ (x : ℝ), (1 - x ^ 2) * exp x = - (x - 1) ^ 2 * exp x + C :=
by sorry

end integral_1_integral_2_integral_3_integral_4_integral_5_l240_240407


namespace num_divisors_8_factorial_l240_240013

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l240_240013


namespace applesGivenToTeachers_l240_240317

/-- Define the initial number of apples Sarah had. --/
def initialApples : ℕ := 25

/-- Define the number of apples given to friends. --/
def applesGivenToFriends : ℕ := 5

/-- Define the number of apples Sarah ate. --/
def applesEaten : ℕ := 1

/-- Define the number of apples left when Sarah got home. --/
def applesLeftAtHome : ℕ := 3

/--
Use the given conditions to prove that Sarah gave away 16 apples to teachers.
--/
theorem applesGivenToTeachers :
  (initialApples - applesGivenToFriends - applesEaten - applesLeftAtHome) = 16 := by
  calc
    initialApples - applesGivenToFriends - applesEaten - applesLeftAtHome
    = 25 - 5 - 1 - 3 : by sorry
    ... = 16 : by sorry

end applesGivenToTeachers_l240_240317


namespace roots_of_equation_l240_240791

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l240_240791


namespace solve_for_a_when_diamond_eq_6_l240_240973

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem solve_for_a_when_diamond_eq_6 (a : ℝ) : diamond a 3 = 6 → a = 8 :=
by
  intros h
  simp [diamond] at h
  sorry

end solve_for_a_when_diamond_eq_6_l240_240973


namespace find_x_y_l240_240422

theorem find_x_y (x y : ℝ) : 
  (x - 12) ^ 2 + (y - 13) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ (x = 37 / 3 ∧ y = 38 / 3) :=
by
  sorry

end find_x_y_l240_240422


namespace circle_radius_eq_l240_240347

theorem circle_radius_eq {s r : ℝ} 
  (h1 : s * real.sqrt 2 = 2 * r)
  (h2 : 4 * s = real.pi * r^2) :
  r = 2 * real.sqrt 2 / real.pi :=
by
  sorry

end circle_radius_eq_l240_240347


namespace regular_polygon_sides_l240_240859

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240859


namespace average_visitors_per_day_l240_240127

/-- A library has different visitor numbers depending on the day of the week.
  - On Sundays, the library has an average of 660 visitors.
  - On Mondays through Thursdays, there are 280 visitors on average.
  - Fridays and Saturdays see an increase to an average of 350 visitors.
  - This month has a special event on the third Saturday, bringing an extra 120 visitors that day.
  - The month has 30 days and begins with a Sunday.
  We want to calculate the average number of visitors per day for the entire month. -/
theorem average_visitors_per_day
  (num_days : ℕ) (starts_on_sunday : Bool)
  (sundays_visitors : ℕ) (weekdays_visitors : ℕ) (weekend_visitors : ℕ)
  (special_event_extra_visitors : ℕ) (sundays : ℕ) (mondays : ℕ)
  (tuesdays : ℕ) (wednesdays : ℕ) (thursdays : ℕ) (fridays : ℕ)
  (saturdays : ℕ) :
  num_days = 30 → starts_on_sunday = true →
  sundays_visitors = 660 → weekdays_visitors = 280 → weekend_visitors = 350 →
  special_event_extra_visitors = 120 →
  sundays = 4 → mondays = 5 →
  tuesdays = 4 → wednesdays = 4 → thursdays = 4 → fridays = 4 → saturdays = 4 →
  ((sundays * sundays_visitors +
    mondays * weekdays_visitors +
    tuesdays * weekdays_visitors +
    wednesdays * weekdays_visitors +
    thursdays * weekdays_visitors +
    fridays * weekend_visitors +
    saturdays * weekend_visitors +
    special_event_extra_visitors) / num_days = 344) :=
by
  intros
  sorry

end average_visitors_per_day_l240_240127


namespace a_values_unique_solution_l240_240724

theorem a_values_unique_solution :
  (∀ a : ℝ, ∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) →
  (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end a_values_unique_solution_l240_240724


namespace find_other_endpoint_l240_240493

theorem find_other_endpoint :
  ∀ (A B M : ℝ × ℝ),
  M = (2, 3) →
  A = (7, -4) →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  B = (-3, 10) :=
by
  intros A B M hM1 hA hM2
  sorry

end find_other_endpoint_l240_240493


namespace regular_polygon_sides_l240_240845

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240845


namespace median_first_fifteen_integers_l240_240662

theorem median_first_fifteen_integers :
  let l := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] in
  let seventh := l.nth 6 in
  let eighth := l.nth 7 in
  (seventh.is_some ∧ eighth.is_some) →
  (seventh.get_or_else 0 + eighth.get_or_else 0) / 2 = 7.5 :=
by
  sorry

end median_first_fifteen_integers_l240_240662


namespace length_of_DE_in_triangle_l240_240054

noncomputable def triangle_length_DE (BC : ℝ) (C_deg: ℝ) (DE : ℝ) : Prop :=
  BC = 24 * Real.sqrt 2 ∧ C_deg = 45 ∧ DE = 12 * Real.sqrt 2

theorem length_of_DE_in_triangle :
  ∀ (BC : ℝ) (C_deg: ℝ) (DE : ℝ), (BC = 24 * Real.sqrt 2 ∧ C_deg = 45) → DE = 12 * Real.sqrt 2 :=
by
  intros BC C_deg DE h_cond
  have h_length := h_cond.2
  sorry

end length_of_DE_in_triangle_l240_240054


namespace minimum_possible_value_of_Box_l240_240597

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l240_240597


namespace triangle_ABC_properties_l240_240717

open Real

theorem triangle_ABC_properties
  (a b c : ℝ) 
  (A B C : ℝ) 
  (A_eq : A = π / 3) 
  (b_eq : b = sqrt 2) 
  (cond1 : b^2 + sqrt 2 * a * c = a^2 + c^2) 
  (cond2 : a * cos B = b * sin A) 
  (cond3 : sin B + cos B = sqrt 2) : 
  B = π / 4 ∧ (1 / 2) * a * b * sin (π - A - B) = (3 + sqrt 3) / 4 := 
by 
  sorry

end triangle_ABC_properties_l240_240717


namespace optimal_strategy_l240_240382

-- Define the conditions
def valid_N (N : ℤ) : Prop :=
  0 ≤ N ∧ N ≤ 20

def score (N : ℤ) (other_teams_count : ℤ) : ℤ :=
  if other_teams_count > N then N else 0

-- The mathematical problem statement
theorem optimal_strategy : ∃ N : ℤ, valid_N N ∧ (∀ other_teams_count : ℤ, score 1 other_teams_count ≥ score N other_teams_count ∧ score 1 other_teams_count ≠ 0) :=
sorry

end optimal_strategy_l240_240382


namespace regular_polygon_sides_l240_240944

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240944


namespace convert_to_standard_spherical_coordinates_l240_240298

theorem convert_to_standard_spherical_coordinates :
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  (ρ, adjusted_θ, adjusted_φ) = (4, (7 * Real.pi) / 4, Real.pi / 5) :=
by
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  sorry

end convert_to_standard_spherical_coordinates_l240_240298


namespace regular_polygon_sides_l240_240886

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240886


namespace regular_polygon_sides_l240_240907

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240907


namespace regular_polygon_sides_l240_240863

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240863


namespace total_marks_l240_240528

-- Define the conditions
def average_marks : ℝ := 35
def number_of_candidates : ℕ := 120

-- Define the total marks as a goal to prove
theorem total_marks : number_of_candidates * average_marks = 4200 :=
by
  sorry

end total_marks_l240_240528


namespace no_monochromatic_ap_11_l240_240319

open Function

theorem no_monochromatic_ap_11 :
  ∃ (coloring : ℕ → Fin 4), (∀ a r : ℕ, r > 0 → a + 10 * r ≤ 2014 → ∃ i j : ℕ, (i ≠ j) ∧ (a + i * r < 1 ∨ a + j * r > 2014 ∨ coloring (a + i * r) ≠ coloring (a + j * r))) :=
sorry

end no_monochromatic_ap_11_l240_240319


namespace equation_no_solution_at_5_l240_240174

theorem equation_no_solution_at_5 :
  ∀ (some_expr : ℝ), ¬(1 / (5 + 5) + some_expr = 1 / (5 - 5)) :=
by
  intro some_expr
  sorry

end equation_no_solution_at_5_l240_240174


namespace regular_polygon_sides_l240_240881

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240881


namespace compute_complex_power_l240_240148

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l240_240148


namespace amount_spent_on_candy_l240_240755

-- Define the given conditions
def amount_from_mother := 80
def amount_from_father := 40
def amount_from_uncle := 70
def final_amount := 140 

-- Define the initial amount
def initial_amount := amount_from_mother + amount_from_father 

-- Prove the amount spent on candy
theorem amount_spent_on_candy : 
  initial_amount - (final_amount - amount_from_uncle) = 50 := 
by
  -- Placeholder for proof
  sorry

end amount_spent_on_candy_l240_240755


namespace circle_radius_l240_240359

theorem circle_radius {s r : ℝ} (h : 4 * s = π * r^2) (h1 : 2 * r = s * Real.sqrt 2) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end circle_radius_l240_240359


namespace john_chips_consumption_l240_240303

/-- John starts the week with a routine. Every day, he eats one bag of chips for breakfast,
  two bags for lunch, and doubles the amount he had for lunch for dinner.
  Prove that by the end of the week, John consumed 49 bags of chips. --/
theorem john_chips_consumption : 
  ∀ (days_in_week : ℕ) (chips_breakfast : ℕ) (chips_lunch : ℕ) (chips_dinner : ℕ), 
    days_in_week = 7 ∧ chips_breakfast = 1 ∧ chips_lunch = 2 ∧ chips_dinner = 2 * chips_lunch →
    days_in_week * (chips_breakfast + chips_lunch + chips_dinner) = 49 :=
by
  intros days_in_week chips_breakfast chips_lunch chips_dinner
  sorry

end john_chips_consumption_l240_240303


namespace equation_roots_l240_240794

theorem equation_roots (x : ℝ) : x * x = 2 * x ↔ (x = 0 ∨ x = 2) := by
  sorry

end equation_roots_l240_240794


namespace anusha_share_l240_240071

theorem anusha_share (A B E D G X : ℝ) 
  (h1: 20 * A = X)
  (h2: 15 * B = X)
  (h3: 8 * E = X)
  (h4: 12 * D = X)
  (h5: 10 * G = X)
  (h6: A + B + E + D + G = 950) : 
  A = 112 := 
by 
  sorry

end anusha_share_l240_240071


namespace sequence_properties_l240_240272

/-- Theorem setup:
Assume a sequence {a_n} with a_1 = 1 and a_{n+1} = 2a_n / (a_n + 2)
Also, define b_n = 1 / a_n
-/
theorem sequence_properties 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  -- Prove that {b_n} (b n = 1 / a n) is arithmetic with common difference 1/2
  (∃ b : ℕ → ℝ, (∀ n : ℕ, b n = 1 / a n) ∧ (∀ n : ℕ, b (n + 1) = b n + 1 / 2)) ∧ 
  -- Prove the general formula for a_n
  (∀ n : ℕ, a (n + 1) = 2 / (n + 1)) := 
sorry


end sequence_properties_l240_240272


namespace geometric_sequence_sum_range_l240_240732

theorem geometric_sequence_sum_range (a b c : ℝ) 
  (h1 : ∃ q : ℝ, q ≠ 0 ∧ a = b * q ∧ c = b / q) 
  (h2 : a + b + c = 1) : 
  a + c ∈ (Set.Icc (2 / 3 : ℝ) 1 \ Set.Iio 1) ∪ (Set.Ioo 1 2) :=
sorry

end geometric_sequence_sum_range_l240_240732


namespace roots_of_quadratic_eq_l240_240782

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l240_240782


namespace simplify_fraction_l240_240200

theorem simplify_fraction :
  (5^5 + 5^3) / (5^4 - 5^2) = (65 : ℚ) / 12 := 
by
  sorry

end simplify_fraction_l240_240200


namespace fraction_subtraction_l240_240509

theorem fraction_subtraction : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) 
  = 9 / 20 := by
  sorry

end fraction_subtraction_l240_240509


namespace find_y_l240_240203

-- Define the sequence from 1 to 50
def seq_sum : ℕ := (50 * 51) / 2

-- Define y and the average condition
def average_condition (y : ℚ) : Prop :=
  (seq_sum + y) / 51 = 51 * y

-- Theorem statement
theorem find_y (y : ℚ) (h : average_condition y) : y = 51 / 104 :=
by
  sorry

end find_y_l240_240203


namespace regular_polygon_sides_l240_240897

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240897


namespace determinant_inequality_l240_240413

open Real

def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det 7 (x^2) 2 1 > det 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 :=
by
  sorry

end determinant_inequality_l240_240413


namespace star_point_angle_l240_240093

theorem star_point_angle (n : ℕ) (h : n > 4) (h₁ : n ≥ 3) :
  ∃ θ : ℝ, θ = (n-2) * 180 / n :=
by
  sorry

end star_point_angle_l240_240093


namespace prob_from_boxes_l240_240097

-- Define the probability theory space and events
open_locale big_operators

def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def boxA_prob : ℚ := 19 / 30
noncomputable def boxB_prob : ℚ := (11 : ℚ) / 25

theorem prob_from_boxes :
  let boxA : finset ℕ := finset.range 31,
      boxB := finset.range' 10 35,
      prime_or_gt28 := (boxB.filter (λ n, is_prime n)).union (boxB.filter (λ n, n > 28)),
      boxB_prime_or_gt28 := prime_or_gt28.card
  in boxA_prob * boxB_prob = 209 / 750 :=
by {
  let boxA := finset.range 31,
  let boxB := finset.range' 10 35,
  let prime_or_gt28 := (boxB.filter (λ n, is_prime n)).union (boxB.filter (λ n, n > 28)),
  let boxB_prime_or_gt28 := prime_or_gt28.card,
  have A_prob := boxA_prob,
  have B_prob := boxB_prob,
  linarith,
  sorry
}

end prob_from_boxes_l240_240097


namespace factors_of_48_multiples_of_8_l240_240178

theorem factors_of_48_multiples_of_8 : 
  ∃ count : ℕ, count = 4 ∧ (∀ d ∈ {d | d ∣ 48 ∧ (∃ k, d = 8 * k)}, true) :=
by {
  sorry  -- This is a placeholder for the actual proof
}

end factors_of_48_multiples_of_8_l240_240178


namespace second_interest_rate_exists_l240_240806

theorem second_interest_rate_exists (X Y : ℝ) (H : 0 < X ∧ X ≤ 10000) : ∃ Y, 8 * X + Y * (10000 - X) = 85000 :=
by
  sorry

end second_interest_rate_exists_l240_240806


namespace math_marks_is_95_l240_240555

-- Define the conditions as Lean assumptions
variables (english_marks math_marks physics_marks chemistry_marks biology_marks : ℝ)
variable (average_marks : ℝ)
variable (num_subjects : ℝ)

-- State the conditions
axiom h1 : english_marks = 96
axiom h2 : physics_marks = 82
axiom h3 : chemistry_marks = 97
axiom h4 : biology_marks = 95
axiom h5 : average_marks = 93
axiom h6 : num_subjects = 5

-- Formalize the problem: Prove that math_marks = 95
theorem math_marks_is_95 : math_marks = 95 :=
by
  sorry

end math_marks_is_95_l240_240555


namespace smallest_perimeter_consecutive_integers_triangle_l240_240154

theorem smallest_perimeter_consecutive_integers_triangle :
  ∃ (a b c : ℕ), 
    1 < a ∧ a + 1 = b ∧ b + 1 = c ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    a + b + c = 12 :=
by
  -- proof placeholder
  sorry

end smallest_perimeter_consecutive_integers_triangle_l240_240154


namespace radiator_initial_fluid_l240_240392

theorem radiator_initial_fluid (x : ℝ)
  (h1 : (0.10 * x - 0.10 * 2.2857 + 0.80 * 2.2857) = 0.50 * x) :
  x = 4 :=
sorry

end radiator_initial_fluid_l240_240392


namespace regular_polygon_sides_l240_240902

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240902


namespace arithmetic_sequence_common_difference_l240_240612

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_a2 : a 2 = 10) (h_a4 : a 4 = 18) : 
  d = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l240_240612


namespace books_bought_l240_240082

def cost_price_of_books (n : ℕ) (C : ℝ) (S : ℝ) : Prop :=
  n * C = 16 * S

def gain_or_loss_percentage (gain_loss_percent : ℝ) : Prop :=
  gain_loss_percent = 0.5

def loss_selling_price (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) : Prop :=
  S = (1 - gain_loss_percent) * C
  
theorem books_bought (n : ℕ) (C : ℝ) (S : ℝ) (gain_loss_percent : ℝ) 
  (h1 : cost_price_of_books n C S) 
  (h2 : gain_or_loss_percentage gain_loss_percent) 
  (h3 : loss_selling_price C S gain_loss_percent) : 
  n = 8 := 
sorry 

end books_bought_l240_240082


namespace carwash_problem_l240_240624

theorem carwash_problem
(h1 : ∀ (n : ℕ), 5 * n + 6 * 5 + 7 * 5 = 100)
(h2 : 5 * 5 = 25)
(h3 : 7 * 5 = 35)
(h4 : 100 - 35 - 30 = 35):
(n = 7) :=
by
  have h : 5 * n = 35 := by sorry
  exact eq_of_mul_eq_mul_left (by sorry) h

end carwash_problem_l240_240624


namespace part1_arithmetic_sequence_part2_general_term_part3_max_m_l240_240991

-- Part (1)
theorem part1_arithmetic_sequence (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : a 1 + a 2 = 2 * m) : 
  m = 9 / 8 := 
sorry

-- Part (2)
theorem part2_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2) : 
  ∀ n, a n = 8 ^ (1 - 2 ^ (n - 1)) := 
sorry

-- Part (3)
theorem part3_max_m (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : ∀ n, a n < 4) : 
  m ≤ 2 := 
sorry

end part1_arithmetic_sequence_part2_general_term_part3_max_m_l240_240991


namespace equation_parallel_equation_perpendicular_l240_240176

variables {x y : ℝ}

def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x - 5 * y + 14 = 0
def l3 (x y : ℝ) := 2 * x - y + 7 = 0

theorem equation_parallel {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : 2 * x - y + 6 = 0 :=
sorry

theorem equation_perpendicular {x y : ℝ} (hx : l1 x y) (hy : l2 x y) : x + 2 * y - 2 = 0 :=
sorry

end equation_parallel_equation_perpendicular_l240_240176


namespace find_m_l240_240193

theorem find_m (m : ℝ) (A B C D : ℝ × ℝ)
  (h1 : A = (m, 1)) (h2 : B = (-3, 4))
  (h3 : C = (0, 2)) (h4 : D = (1, 1))
  (h_parallel : (4 - 1) / (-3 - m) = (1 - 2) / (1 - 0)) :
  m = 0 :=
  by
  sorry

end find_m_l240_240193


namespace eval_expression_l240_240258

theorem eval_expression : (5 + 2 + 6) * 2 / 3 - 4 / 3 = 22 / 3 := sorry

end eval_expression_l240_240258


namespace andy_wrong_questions_l240_240610

theorem andy_wrong_questions (a b c d : ℕ) (h1 : a + b = c + d) (h2 : a + d = b + c + 6) (h3 : c = 3) : a = 6 := by
  sorry

end andy_wrong_questions_l240_240610


namespace first_mathematician_correct_second_mathematician_correct_third_mathematician_incorrect_l240_240330

def first_packet_blue_candies_1 : ℕ := 2
def first_packet_total_candies_1 : ℕ := 5

def second_packet_blue_candies_1 : ℕ := 3
def second_packet_total_candies_1 : ℕ := 8

def first_packet_blue_candies_2 : ℕ := 4
def first_packet_total_candies_2 : ℕ := 10

def second_packet_blue_candies_2 : ℕ := 3
def second_packet_total_candies_2 : ℕ := 8

def total_blue_candies_1 : ℕ := first_packet_blue_candies_1 + second_packet_blue_candies_1
def total_candies_1 : ℕ := first_packet_total_candies_1 + second_packet_total_candies_1

def total_blue_candies_2 : ℕ := first_packet_blue_candies_2 + second_packet_blue_candies_2
def total_candies_2 : ℕ := first_packet_total_candies_2 + second_packet_total_candies_2

def prob_first : ℚ := total_blue_candies_1 / total_candies_1
def prob_second : ℚ := total_blue_candies_2 / total_candies_2

def lower_bound : ℚ := 3 / 8
def upper_bound : ℚ := 2 / 5
def third_prob : ℚ := 17 / 40

theorem first_mathematician_correct : prob_first = 5 / 13 := 
begin
  unfold prob_first,
  unfold total_blue_candies_1 total_candies_1,
  simp [first_packet_blue_candies_1, second_packet_blue_candies_1,
    first_packet_total_candies_1, second_packet_total_candies_1],
end

theorem second_mathematician_correct : prob_second = 7 / 18 := 
begin
  unfold prob_second,
  unfold total_blue_candies_2 total_candies_2,
  simp [first_packet_blue_candies_2, second_packet_blue_candies_2,
    first_packet_total_candies_2, second_packet_total_candies_2],
end

theorem third_mathematician_incorrect : ¬ (lower_bound < third_prob ∧ third_prob < upper_bound) :=
by simp [lower_bound, upper_bound, third_prob]; linarith

end first_mathematician_correct_second_mathematician_correct_third_mathematician_incorrect_l240_240330


namespace total_cost_of_apples_l240_240122

variable (num_apples_per_bag cost_per_bag num_apples : ℕ)
#check num_apples_per_bag = 50
#check cost_per_bag = 8
#check num_apples = 750

theorem total_cost_of_apples : 
  (num_apples_per_bag = 50) → 
  (cost_per_bag = 8) → 
  (num_apples = 750) → 
  (num_apples / num_apples_per_bag * cost_per_bag = 120) :=
by
  intros
  sorry

end total_cost_of_apples_l240_240122


namespace sum_of_even_factors_420_l240_240967

def sum_even_factors (n : ℕ) : ℕ :=
  if n ≠ 420 then 0
  else 
    let even_factors_sum :=
      (2 + 4) * (1 + 3) * (1 + 5) * (1 + 7)
    even_factors_sum

theorem sum_of_even_factors_420 : sum_even_factors 420 = 1152 :=
by {
  -- Proof skipped
  sorry
}

end sum_of_even_factors_420_l240_240967


namespace length_of_train_is_correct_l240_240680

noncomputable def speed_in_m_per_s (speed_in_km_per_hr : ℝ) : ℝ := speed_in_km_per_hr * 1000 / 3600

noncomputable def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

noncomputable def length_of_train (total_distance : ℝ) (length_of_bridge : ℝ) : ℝ := total_distance - length_of_bridge

theorem length_of_train_is_correct :
  ∀ (speed_in_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) (length_of_bridge : ℝ),
  speed_in_km_per_hr = 72 →
  time_to_cross_bridge = 12.199024078073753 →
  length_of_bridge = 134 →
  length_of_train (total_distance (speed_in_m_per_s speed_in_km_per_hr) time_to_cross_bridge) length_of_bridge = 110.98048156147506 :=
by 
  intros speed_in_km_per_hr time_to_cross_bridge length_of_bridge hs ht hl;
  rw [hs, ht, hl];
  sorry

end length_of_train_is_correct_l240_240680


namespace trench_digging_l240_240244

theorem trench_digging 
  (t : ℝ) (T : ℝ) (work_units : ℝ)
  (h1 : 4 * t = 10)
  (h2 : T = 5 * t) :
  work_units = 80 :=
by
  sorry

end trench_digging_l240_240244


namespace P_sufficient_but_not_necessary_for_Q_l240_240706

def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) := 
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l240_240706


namespace exists_increasing_sequences_l240_240480

theorem exists_increasing_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) :
  (∀ n : ℕ, a n < a (n + 1)) ∧ (∀ n : ℕ, b n < b (n + 1)) ∧
  (∀ n : ℕ, a n * (a n + 1) ∣ b n ^ 2 + 1) :=
sorry

end exists_increasing_sequences_l240_240480


namespace complex_modulus_to_real_l240_240731

theorem complex_modulus_to_real (a : ℝ) (h : (a + 1)^2 + (1 - a)^2 = 10) : a = 2 ∨ a = -2 :=
sorry

end complex_modulus_to_real_l240_240731


namespace trap_area_BCDK_l240_240536

-- Definitions
variables {A B C D E K M : Point}
variable {circle : Circle}
variable {rect : Rectangle}
variable {ab_len ke_ka_ratio : ℝ}
variable {area_BCKD : ℝ}

noncomputable def circle_pass_through_A_B_and_touch_CD_midpoint
  (hA : A ∈ circle)
  (hB : B ∈ circle)
  (hMidCD : M ∈ circle)
  (hMidCDCD : M.is_midpoint CD) : Prop := sorry

noncomputable def tangent_line_from_D
  (h_tangent_circle : is_tangent D E circle)
  (h_extend_AB : is_intersection K (extension AB) E) : Prop := sorry

-- Hypotheses
def problem_hypotheses : Prop :=
  circle_pass_through_A_B_and_touch_CD_midpoint
    (by simp [A])
    (by simp [B])
    (by simp [M])
    (by simp)
  ∧ tangent_line_from_D
    (by simp)
    (by simp)
  ∧ AB = 10
  ∧ KE / KA = 3 / 2

-- Goal: area_BCDK == 210
theorem trap_area_BCDK
  (h : problem_hypotheses)
  : area_BCKD = 210 :=
begin
  sorry
end

end trap_area_BCDK_l240_240536


namespace leak_drain_time_l240_240114

noncomputable def pump_rate : ℚ := 1/2
noncomputable def leak_empty_rate : ℚ := 1 / (1 / pump_rate - 5/11)

theorem leak_drain_time :
  let pump_rate := 1/2
  let combined_rate := 5/11
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate = 22 :=
  by
    -- Definition of pump rate
    let pump_rate := 1/2
    -- Definition of combined rate
    let combined_rate := 5/11
    -- Definition of leak rate
    let leak_rate := pump_rate - combined_rate
    -- Calculate leak drain time
    show 1 / leak_rate = 22
    sorry

end leak_drain_time_l240_240114


namespace police_officers_on_duty_l240_240634

theorem police_officers_on_duty
  (female_officers : ℕ)
  (percent_female_on_duty : ℚ)
  (total_female_on_duty : ℕ)
  (total_officers_on_duty : ℕ)
  (H1 : female_officers = 1000)
  (H2 : percent_female_on_duty = 15 / 100)
  (H3 : total_female_on_duty = percent_female_on_duty * female_officers)
  (H4 : 2 * total_female_on_duty = total_officers_on_duty) :
  total_officers_on_duty = 300 :=
by
  sorry

end police_officers_on_duty_l240_240634


namespace problem_conditions_l240_240489

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (1 + x)
else if -1 < x ∧ x < 0 then x / (1 - x)
else 0

theorem problem_conditions (a b : ℝ) (x : ℝ) :
  (∀ x : ℝ, -1 < x → x < 1 → f (-x) = -f x) ∧ 
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = (-a * x - b) / (1 + x)) ∧ 
  (f (1 / 2) = 1 / 3) →
  (a = -1) ∧ (b = 0) ∧
  (∀ x :  ℝ, -1 < x ∧ x < 1 → 
    (if 0 ≤ x ∧ x < 1 then f x = x / (1 + x) else if -1 < x ∧ x < 0 then f x = x / (1 - x) else True)) ∧ 
  (∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2) ∧ 
  (∀ x : ℝ, f (x - 1) + f x > 0 → (1 / 2 < x ∧ x < 1)) :=
by
  sorry

end problem_conditions_l240_240489


namespace no_solution_equation_l240_240263

theorem no_solution_equation (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 → x ≠ 8 → (x - 3) / (x - 4) = (x - m) / (x - 8) → false) ↔ m = 7 :=
by
  sorry

end no_solution_equation_l240_240263


namespace pizza_slices_left_l240_240112

-- Defining the initial number of pizza slices
def initial_slices : ℕ := 16

-- Defining one-fourth of the pizza during dinner time
def dinner_fraction : ℚ := 1 / 4

-- Defining one-fourth of the remaining pizza eaten by Yves
def yves_fraction : ℚ := 1 / 4

-- Defining the slices eaten by each sibling
def slices_per_sibling : ℕ := 2

-- Theorem to prove the number of slices of pizza left is 5
theorem pizza_slices_left :
    let eaten_at_dinner := initial_slices * dinner_fraction
    let remaining_after_dinner := initial_slices - eaten_at_dinner
    let eaten_by_yves := remaining_after_dinner * yves_fraction
    let remaining_after_yves := remaining_after_dinner - eaten_by_yves
    let eaten_by_siblings := 2 * slices_per_sibling
    let final_remaining := remaining_after_yves - eaten_by_siblings
    final_remaining = 5 :=
by
  have step1 : let eaten_at_dinner := initial_slices * dinner_fraction
               let remaining_after_dinner := initial_slices - eaten_at_dinner
               eaten_at_dinner = 4 := by norm_num
  have step2 : let remaining_after_dinner := initial_slices - eaten_at_dinner
               remaining_after_dinner = 12 := by norm_num
  have step3 : let eaten_by_yves := remaining_after_dinner * yves_fraction
               eaten_by_yves = 3 := by norm_num
  have step4 : let remaining_after_yves := remaining_after_dinner - eaten_by_yves
               remaining_after_yves = 9 := by norm_num
  have step5 : let eaten_by_siblings := 2 * slices_per_sibling
               final_remaining = remaining_after_yves - eaten_by_siblings
               eaten_by_siblings = 4 := by norm_num
  show final_remaining = 5 from calc
    final_remaining 
      = remaining_after_yves - eaten_by_siblings := by norm_num
      ... = 5 := by norm_num

end pizza_slices_left_l240_240112


namespace roots_of_equation_l240_240790

theorem roots_of_equation (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) :=
by
  -- Proof omitted
  sorry

end roots_of_equation_l240_240790


namespace point_B_value_l240_240195

theorem point_B_value (A : ℝ) (B : ℝ) (hA : A = -5) (hB : B = -1 ∨ B = -9) :
  ∃ B : ℝ, (B = A + 4 ∨ B = A - 4) :=
by sorry

end point_B_value_l240_240195


namespace collinear_points_l240_240284

theorem collinear_points (k : ℝ) (OA OB OC : ℝ × ℝ) 
  (hOA : OA = (1, -3)) 
  (hOB : OB = (2, -1))
  (hOC : OC = (k + 1, k - 2))
  (h_collinear : ∃ t : ℝ, OC - OA = t • (OB - OA)) : 
  k = 1 :=
by
  have := h_collinear
  sorry

end collinear_points_l240_240284


namespace reduced_flow_rate_is_correct_l240_240246

-- Define the original flow rate
def original_flow_rate : ℝ := 5.0

-- Define the function for the reduced flow rate
def reduced_flow_rate (x : ℝ) : ℝ := 0.6 * x - 1

-- Prove that the reduced flow rate is 2.0 gallons per minute
theorem reduced_flow_rate_is_correct : reduced_flow_rate original_flow_rate = 2.0 := by
  sorry

end reduced_flow_rate_is_correct_l240_240246


namespace jam_cost_is_162_l240_240560

theorem jam_cost_is_162 (N B J : ℕ) (h1 : N > 1) (h2 : 4 * B + 6 * J = 39) (h3 : N = 9) : 
  6 * N * J = 162 := 
by sorry

end jam_cost_is_162_l240_240560


namespace radius_of_circumscribed_circle_l240_240362

theorem radius_of_circumscribed_circle 
    (r : ℝ)
    (h : 8 * r = π * r^2) : 
    r = 8 / π :=
by
    sorry

end radius_of_circumscribed_circle_l240_240362


namespace part1_ABC_inquality_part2_ABCD_inquality_l240_240196

theorem part1_ABC_inquality (a b c ABC : ℝ) : 
  (ABC <= (a^2 + b^2) / 4) -> 
  (ABC <= (b^2 + c^2) / 4) -> 
  (ABC <= (a^2 + c^2) / 4) -> 
    (ABC < (a^2 + b^2 + c^2) / 6) :=
sorry

theorem part2_ABCD_inquality (a b c d ABC BCD CDA DAB ABCD : ℝ) :
  (ABCD = 1/2 * ((ABC) + (BCD) + (CDA) + (DAB))) -> 
  (ABC < (a^2 + b^2 + c^2) / 6) -> 
  (BCD < (b^2 + c^2 + d^2) / 6) -> 
  (CDA < (c^2 + d^2 + a^2) / 6) -> 
  (DAB < (d^2 + a^2 + b^2) / 6) -> 
    (ABCD < (a^2 + b^2 + c^2 + d^2) / 6) :=
sorry

end part1_ABC_inquality_part2_ABCD_inquality_l240_240196


namespace original_price_l240_240548

theorem original_price (P : ℝ) (h1 : 0.76 * P = 820) : P = 1079 :=
by
  sorry

end original_price_l240_240548


namespace regular_polygon_sides_l240_240870

theorem regular_polygon_sides (n : ℕ) : (360 : ℝ) / n = 18 → n = 20 :=
by
  intros h
  -- Start the proof here
  sorry

end regular_polygon_sides_l240_240870


namespace not_a_perfect_square_l240_240380

theorem not_a_perfect_square :
  ¬ (∃ x, (x: ℝ)^2 = 5^2025) :=
by
  sorry

end not_a_perfect_square_l240_240380


namespace regular_polygon_sides_l240_240928

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240928


namespace area_of_region_l240_240406

noncomputable def region_area : ℝ :=
  ∫ t in (2 * Real.pi / 3)..(4 * Real.pi / 3), 16 * (1 - cos t) ^ 2

theorem area_of_region :
  region_area = 16 * Real.pi :=
by
  -- proof steps will go here
  sorry

end area_of_region_l240_240406


namespace father_three_times_marika_in_year_l240_240067

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year_l240_240067


namespace measure_angle_ACB_l240_240052

-- Definitions of angles and the conditions
def angle_ABD := 140
def angle_BAC := 105
def supplementary_angle (α β : ℕ) := α + β = 180
def angle_sum_property (α β γ : ℕ) := α + β + γ = 180

-- Theorem to prove the measure of angle ACB
theorem measure_angle_ACB (angle_ABD : ℕ) 
                         (angle_BAC : ℕ) 
                         (h1 : supplementary_angle angle_ABD 40)
                         (h2 : angle_sum_property 40 angle_BAC 35) :
  angle_sum_property 40 105 35 :=
sorry

end measure_angle_ACB_l240_240052


namespace sum_of_cubes_eq_three_l240_240320

theorem sum_of_cubes_eq_three (k : ℤ) : 
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 :=
by 
  sorry

end sum_of_cubes_eq_three_l240_240320


namespace different_testing_methods_1_different_testing_methods_2_l240_240056

-- Definitions used in Lean 4 statement should be derived from the conditions in a).
def total_products := 10
def defective_products := 4
def non_defective_products := total_products - defective_products
def choose (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement (1)
theorem different_testing_methods_1 :
  let first_defective := 5
  let last_defective := 10
  let non_defective_in_first_4 := choose 6 4
  let defective_in_middle_5 := choose 5 3
  let total_methods := non_defective_in_first_4 * defective_in_middle_5 * Nat.factorial 5 * Nat.factorial 4
  total_methods = 103680 := sorry

-- Statement (2)
theorem different_testing_methods_2 :
  let first_defective := 5
  let remaining_defective := 4
  let non_defective_in_first_4 := choose 6 4
  let total_methods := non_defective_in_first_4 * Nat.factorial 5
  total_methods = 576 := sorry

end different_testing_methods_1_different_testing_methods_2_l240_240056


namespace mathematicians_correctness_l240_240332

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  (3 / 8 < 2 / 5 ∧ 2 / 5 < 17 / 40) → false :=
by 
  sorry

end mathematicians_correctness_l240_240332


namespace ninth_term_arithmetic_sequence_l240_240651

variable (a d : ℕ)

def arithmetic_sequence_sum (a d : ℕ) : ℕ :=
  a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d)

theorem ninth_term_arithmetic_sequence (h1 : arithmetic_sequence_sum a d = 21) (h2 : a + 6 * d = 7) : a + 8 * d = 9 :=
by
  sorry

end ninth_term_arithmetic_sequence_l240_240651


namespace female_athletes_in_sample_l240_240679

theorem female_athletes_in_sample (M F S : ℕ) (hM : M = 56) (hF : F = 42) (hS : S = 28) :
  (F * (S / (M + F))) = 12 :=
by
  rw [hM, hF, hS]
  norm_num
  sorry

end female_athletes_in_sample_l240_240679


namespace travel_distance_l240_240829

variables (speed time : ℕ) (distance : ℕ)

theorem travel_distance (hspeed : speed = 75) (htime : time = 4) : distance = speed * time → distance = 300 :=
by
  intros hdist
  rw [hspeed, htime] at hdist
  simp at hdist
  assumption

end travel_distance_l240_240829


namespace f_is_even_if_g_is_odd_l240_240462

variable (g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) :=
  ∀ x : ℝ, h (-x) = -h x

def is_even (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = f x

theorem f_is_even_if_g_is_odd (hg : is_odd g) :
  is_even (fun x => |g (x^4)|) :=
by
  sorry

end f_is_even_if_g_is_odd_l240_240462


namespace increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l240_240070

open Real

-- Defining the sequences
noncomputable def a_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ n
noncomputable def b_seq (n : ℕ) : ℝ := (1 + (1 : ℝ) / n) ^ (n + 1)

theorem increase_function (x : ℝ) (hx : 0 < x) : 
  ((1:ℝ) + 1 / x) ^ x < (1 + 1 / (x + 1)) ^ (x + 1) := sorry

theorem a_seq_increasing (n : ℕ) (hn : 0 < n) : 
  a_seq n < a_seq (n + 1) := sorry

theorem b_seq_decreasing (n : ℕ) (hn : 0 < n) : 
  b_seq (n + 1) < b_seq n := sorry

theorem seq_relation (n : ℕ) (hn : 0 < n) : 
  a_seq n < b_seq n := sorry

end increase_function_a_seq_increasing_b_seq_decreasing_seq_relation_l240_240070


namespace expression_in_scientific_notation_l240_240076

-- Conditions
def billion : ℝ := 10^9
def a : ℝ := 20.8

-- Statement
theorem expression_in_scientific_notation : a * billion = 2.08 * 10^10 := by
  sorry

end expression_in_scientific_notation_l240_240076


namespace ordered_pairs_count_l240_240012

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count_l240_240012


namespace value_of_b_l240_240999

noncomputable def problem (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :=
  (a1 ≠ a2) ∧ (a1 ≠ a3) ∧ (a1 ≠ a4) ∧ (a1 ≠ a5) ∧
  (a2 ≠ a3) ∧ (a2 ≠ a4) ∧ (a2 ≠ a5) ∧
  (a3 ≠ a4) ∧ (a3 ≠ a5) ∧
  (a4 ≠ a5) ∧
  (a1 + a2 + a3 + a4 + a5 = 9) ∧
  ((b - a1) * (b - a2) * (b - a3) * (b - a4) * (b - a5) = 2009) ∧
  (∃ b : ℤ, b = 10)

theorem value_of_b (a1 a2 a3 a4 a5 : ℤ) (b : ℤ) :
  problem a1 a2 a3 a4 a5 b → b = 10 :=
  sorry

end value_of_b_l240_240999


namespace sqrt_div_l240_240106

theorem sqrt_div (x: ℕ) (h1: Nat.sqrt 144 * Nat.sqrt 144 = 144) (h2: 144 = 12 * 12) (h3: 2 * x = 12) : x = 6 :=
sorry

end sqrt_div_l240_240106


namespace sequence_periodicity_l240_240619

theorem sequence_periodicity : 
  let a : ℕ → ℤ := λ n, 
    if n = 1 then 13 
    else if n = 2 then 56 
    else if n > 2 ∧ n % 6 = 5 then a (n - 1) + a (n - 2) 
    else a (n - 1) - a (n - 2)
  in a 1934 = 56 :=
by
  sorry

end sequence_periodicity_l240_240619


namespace trapezoidal_field_base_count_l240_240079

theorem trapezoidal_field_base_count
  (A : ℕ) (h : ℕ) (b1 b2 : ℕ)
  (hdiv8 : ∃ m n : ℕ, b1 = 8 * m ∧ b2 = 8 * n)
  (area_eq : A = (h * (b1 + b2)) / 2)
  (A_val : A = 1400)
  (h_val : h = 50) :
  (∃ pair1 pair2 pair3, (pair1 + pair2 + pair3 = (b1 + b2))) :=
by
  sorry

end trapezoidal_field_base_count_l240_240079


namespace total_weight_collected_l240_240163

def GinaCollectedBags : ℕ := 8
def NeighborhoodFactor : ℕ := 120
def WeightPerBag : ℕ := 6

theorem total_weight_collected :
  (GinaCollectedBags * NeighborhoodFactor + GinaCollectedBags) * WeightPerBag = 5808 :=
by
  sorry

end total_weight_collected_l240_240163


namespace interior_angles_sum_l240_240776

theorem interior_angles_sum (h : ∀ (n : ℕ), n = 360 / 20) : 
  180 * (h 18 - 2) = 2880 :=
by
  sorry

end interior_angles_sum_l240_240776


namespace regular_polygon_sides_l240_240949

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240949


namespace kangaroo_fiber_intake_l240_240325

-- Suppose kangaroos absorb only 30% of the fiber they eat
def absorption_rate : ℝ := 0.30

-- If a kangaroo absorbed 15 ounces of fiber in one day
def absorbed_fiber : ℝ := 15.0

-- Prove the kangaroo ate 50 ounces of fiber that day
theorem kangaroo_fiber_intake (x : ℝ) (hx : absorption_rate * x = absorbed_fiber) : x = 50 :=
by
  sorry

end kangaroo_fiber_intake_l240_240325


namespace solution_is_unique_l240_240707

noncomputable def solution (f : ℝ → ℝ) (α : ℝ) :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

theorem solution_is_unique (f : ℝ → ℝ) (α : ℝ)
  (h : solution f α) :
  f = id ∧ α = -1 :=
sorry

end solution_is_unique_l240_240707


namespace minimum_possible_value_of_Box_l240_240598

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end minimum_possible_value_of_Box_l240_240598


namespace wholesome_bakery_loaves_on_wednesday_l240_240078

theorem wholesome_bakery_loaves_on_wednesday :
  ∀ (L_wed L_thu L_fri L_sat L_sun L_mon : ℕ),
    L_thu = 7 →
    L_fri = 10 →
    L_sat = 14 →
    L_sun = 19 →
    L_mon = 25 →
    L_thu - L_wed = 2 →
    L_wed = 5 :=
by intros L_wed L_thu L_fri L_sat L_sun L_mon;
   intros H_thu H_fri H_sat H_sun H_mon H_diff;
   sorry

end wholesome_bakery_loaves_on_wednesday_l240_240078


namespace sum_of_interior_angles_l240_240775

theorem sum_of_interior_angles (ext_angle : ℝ) (h : ext_angle = 20) : 
  let n := 360 / ext_angle in
  let int_sum := 180 * (n - 2) in
  int_sum = 2880 := 
by 
  sorry

end sum_of_interior_angles_l240_240775


namespace regular_polygon_sides_l240_240922

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240922


namespace skill_of_passing_through_walls_l240_240295

theorem skill_of_passing_through_walls (k n : ℕ) (h : k = 8) (h_eq : k * Real.sqrt (k / (k * k - 1)) = Real.sqrt (k * k / (k * k - 1))) : n = k * k - 1 :=
by sorry

end skill_of_passing_through_walls_l240_240295


namespace regular_polygon_sides_l240_240898

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240898


namespace determine_number_l240_240117

noncomputable def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  (∃ d1 d2 d3, 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    (
      (d1 = 5 ∨ d1 = 1 ∨ d1 = 5 ∨ d1 = 2) ∧
      (d2 = 4 ∨ d2 = 4 ∨ d2 = 4) ∧
      (d3 = 3 ∨ d3 = 2 ∨ d3 = 6)
    ) ∧
    (
      (d1 ≠ 1 ∧ d1 ≠ 2 ∧ d1 ≠ 6) ∧
      (d2 ≠ 5 ∧ d2 ≠ 4 ∧ d2 ≠ 6 ∧ d2 ≠ 2) ∧
      (d3 ≠ 5 ∧ d3 ≠ 4 ∧ d3 ≠ 1 ∧ d3 ≠ 2)
    )
  )

theorem determine_number : ∃ n : ℕ, is_valid_number n ∧ n = 163 :=
by 
  existsi 163
  unfold is_valid_number
  sorry

end determine_number_l240_240117


namespace integer_values_not_satisfying_inequality_l240_240430

theorem integer_values_not_satisfying_inequality :
  (∃ x : ℤ, ¬(3 * x^2 + 17 * x + 28 > 25)) ∧ (∃ x1 x2 : ℤ, x1 = -2 ∧ x2 = -1) ∧
  ∀ x : ℤ, (x = -2 ∨ x = -1) -> ¬(3 * x^2 + 17 * x + 28 > 25) :=
by
  sorry

end integer_values_not_satisfying_inequality_l240_240430


namespace ball_bounce_height_l240_240388

theorem ball_bounce_height :
  ∃ k : ℕ, 800 * (1 / 2 : ℝ)^k < 2 ∧ k ≥ 9 :=
by
  sorry

end ball_bounce_height_l240_240388


namespace consecutive_even_numbers_divisible_by_384_l240_240647

theorem consecutive_even_numbers_divisible_by_384 (n : Nat) (h1 : n * (n + 2) * (n + 4) * (n + 6) * (n + 8) * (n + 10) * (n + 12) = 384) : n = 6 :=
sorry

end consecutive_even_numbers_divisible_by_384_l240_240647


namespace area_of_ring_between_concentric_circles_l240_240804

theorem area_of_ring_between_concentric_circles :
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  area_ring = 95 * Real.pi :=
by
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  show area_ring = 95 * Real.pi
  sorry

end area_of_ring_between_concentric_circles_l240_240804


namespace sandcastle_ratio_l240_240501

-- Definitions based on conditions in a)
def sandcastles_on_marks_beach : ℕ := 20
def towers_per_sandcastle_marks_beach : ℕ := 10
def towers_per_sandcastle_jeffs_beach : ℕ := 5
def total_combined_sandcastles_and_towers : ℕ := 580

-- The main statement to prove
theorem sandcastle_ratio : 
  ∃ (J : ℕ), 
  (sandcastles_on_marks_beach + (towers_per_sandcastle_marks_beach * sandcastles_on_marks_beach) + J + (towers_per_sandcastle_jeffs_beach * J) = total_combined_sandcastles_and_towers) ∧ 
  (J / sandcastles_on_marks_beach = 3) :=
by 
  sorry

end sandcastle_ratio_l240_240501


namespace regular_polygon_sides_l240_240890

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240890


namespace ratio_of_volume_to_surface_area_l240_240130

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end ratio_of_volume_to_surface_area_l240_240130


namespace sum4_l240_240307

noncomputable def alpha : ℂ := sorry
noncomputable def beta : ℂ := sorry
noncomputable def gamma : ℂ := sorry

axiom sum1 : alpha + beta + gamma = 1
axiom sum2 : alpha^2 + beta^2 + gamma^2 = 5
axiom sum3 : alpha^3 + beta^3 + gamma^3 = 9

theorem sum4 : alpha^4 + beta^4 + gamma^4 = 56 := by
  sorry

end sum4_l240_240307


namespace dvd_cost_l240_240301

-- Given conditions
def vhs_trade_in_value : Int := 2
def number_of_movies : Int := 100
def total_replacement_cost : Int := 800

-- Statement to prove
theorem dvd_cost :
  ((number_of_movies * vhs_trade_in_value) + (number_of_movies * 6) = total_replacement_cost) :=
by
  sorry

end dvd_cost_l240_240301


namespace ending_number_of_range_l240_240367

/-- The sum of the first n consecutive odd integers is n^2. -/
def sum_first_n_odd : ℕ → ℕ 
| 0       => 0
| (n + 1) => (2 * n + 1) + sum_first_n_odd n

/-- The sum of all odd integers between 11 and the ending number is 416. -/
def sum_odd_integers (a b : ℕ) : ℕ :=
  let s := (1 + b) / 2 - (1 + a) / 2 + 1
  sum_first_n_odd s

theorem ending_number_of_range (n : ℕ) (h1 : sum_first_n_odd n = n^2) 
  (h2 : sum_odd_integers 11 n = 416) : 
  n = 67 :=
sorry

end ending_number_of_range_l240_240367


namespace find_prob_A_l240_240173

variable (P : String → ℝ)
variable (A B : String)

-- Conditions
axiom prob_complement_twice : P B = 2 * P A
axiom prob_sum_to_one : P A + P B = 1

-- Statement to be proved
theorem find_prob_A : P A = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end find_prob_A_l240_240173


namespace remainder_of_large_number_l240_240968

noncomputable def X (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 32
  | 6 => 64
  | 7 => 128
  | 8 => 256
  | 9 => 512
  | 10 => 1024
  | 11 => 2048
  | 12 => 4096
  | 13 => 8192
  | _ => 0

noncomputable def concatenate_X (k : ℕ) : ℕ :=
  if k = 5 then 
    100020004000800160032
  else if k = 11 then 
    100020004000800160032006401280256051210242048
  else if k = 13 then 
    10002000400080016003200640128025605121024204840968192
  else 
    0

theorem remainder_of_large_number :
  (concatenate_X 13) % (concatenate_X 5) = 40968192 :=
by
  sorry

end remainder_of_large_number_l240_240968


namespace good_subset_divisible_by_5_l240_240591

noncomputable def num_good_subsets : ℕ :=
  (Nat.factorial 1000) / ((Nat.factorial 201) * (Nat.factorial (1000 - 201)))

theorem good_subset_divisible_by_5 : num_good_subsets / 5 = (1 / 5) * num_good_subsets := 
sorry

end good_subset_divisible_by_5_l240_240591


namespace num_divisors_8_factorial_l240_240015

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l240_240015


namespace probability_x_lt_2y_in_rectangle_l240_240240

-- Define the rectangle and the conditions
def in_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3

-- Define the condition x < 2y
def condition_x_lt_2y (x y : ℝ) : Prop :=
  x < 2 * y

-- Define the probability calculation
theorem probability_x_lt_2y_in_rectangle :
  let rectangle_area := (4:ℝ) * 3
  let triangle_area := (1:ℝ) / 2 * 4 * 2
  let probability := triangle_area / rectangle_area
  probability = 1 / 3 :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l240_240240


namespace male_percentage_l240_240740

theorem male_percentage (total_employees : ℕ)
  (males_below_50 : ℕ)
  (percentage_males_at_least_50 : ℝ)
  (male_percentage : ℝ) :
  total_employees = 2200 →
  males_below_50 = 616 →
  percentage_males_at_least_50 = 0.3 → 
  male_percentage = 40 :=
by
  sorry

end male_percentage_l240_240740


namespace total_students_appeared_l240_240611

variable (T : ℝ) -- total number of students

def fraction_failed := 0.65
def num_failed := 546

theorem total_students_appeared :
  0.65 * T = 546 → T = 840 :=
by
  intro h
  sorry

end total_students_appeared_l240_240611


namespace exists_diff_shape_and_color_l240_240248

variable (Pitcher : Type) 
variable (shape color : Pitcher → Prop)
variable (exists_diff_shape : ∃ (A B : Pitcher), shape A ≠ shape B)
variable (exists_diff_color : ∃ (A B : Pitcher), color A ≠ color B)

theorem exists_diff_shape_and_color : ∃ (A B : Pitcher), shape A ≠ shape B ∧ color A ≠ color B :=
  sorry

end exists_diff_shape_and_color_l240_240248


namespace largest_c_value_l240_240699

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := x^2 + 3 * x + c

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, f x c = -2) ↔ c ≤ 1/4 := by
sorry

end largest_c_value_l240_240699


namespace sum_common_seq_first_n_l240_240139

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end sum_common_seq_first_n_l240_240139


namespace regular_polygon_sides_l240_240841

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ (x : ℕ), x > 2 → n = x)
  (h2 : ∀ (θ : ℕ), θ = 18 → 360 / n = θ) : n = 20 := by
  sorry

end regular_polygon_sides_l240_240841


namespace regular_polygon_sides_l240_240906

theorem regular_polygon_sides (n : ℕ) (h : 360 % n = 0) (h₀ : 360 / n = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240906


namespace bert_toy_phones_l240_240404

theorem bert_toy_phones (P : ℕ) (berts_price_per_phone : ℕ) (berts_earning : ℕ)
                        (torys_price_per_gun : ℕ) (torys_earning : ℕ) (tory_guns : ℕ)
                        (earnings_difference : ℕ)
                        (h1 : berts_price_per_phone = 18)
                        (h2 : torys_price_per_gun = 20)
                        (h3 : tory_guns = 7)
                        (h4 : torys_earning = tory_guns * torys_price_per_gun)
                        (h5 : berts_earning = torys_earning + earnings_difference)
                        (h6 : earnings_difference = 4)
                        (h7 : P = berts_earning / berts_price_per_phone) :
  P = 8 := by sorry

end bert_toy_phones_l240_240404


namespace regular_polygon_sides_l240_240914

-- Definition of a regular polygon and its properties
def regular_polygon (n : ℕ) : Prop :=
  ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → true

-- Definition of the exterior angle of a regular polygon
def exterior_angle (n : ℕ) : ℝ :=
  360 / n

-- Theorem stating that a regular polygon with an exterior angle of 18 degrees has 20 sides
theorem regular_polygon_sides (n : ℕ) (h_reg : regular_polygon n) (h_angle : exterior_angle n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240914


namespace division_identity_l240_240375

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end division_identity_l240_240375


namespace Turner_Catapult_rides_l240_240505

def tickets_needed (rollercoaster_rides Ferris_wheel_rides Catapult_rides : ℕ) : ℕ :=
  4 * rollercoaster_rides + 1 * Ferris_wheel_rides + 4 * Catapult_rides

theorem Turner_Catapult_rides :
  ∀ (x : ℕ), tickets_needed 3 1 x = 21 → x = 2 := by
  intros x h
  sorry

end Turner_Catapult_rides_l240_240505


namespace time_for_one_kid_to_wash_six_whiteboards_l240_240266

-- Define the conditions as a function
def time_taken (k : ℕ) (w : ℕ) : ℕ := 20 * 4 * w / k

theorem time_for_one_kid_to_wash_six_whiteboards :
  time_taken 1 6 = 160 := by
-- Proof omitted
sorry

end time_for_one_kid_to_wash_six_whiteboards_l240_240266


namespace smallest_even_number_of_seven_l240_240768

-- Conditions: The sum of seven consecutive even numbers is 700.
-- We need to prove that the smallest of these numbers is 94.

theorem smallest_even_number_of_seven (n : ℕ) (hn : 7 * n = 700) :
  ∃ (a b c d e f g : ℕ), 
  (2 * a + 4 * b + 6 * c + 8 * d + 10 * e + 12 * f + 14 * g = 700) ∧ 
  (a = b - 1) ∧ (b = c - 1) ∧ (c = d - 1) ∧ (d = e - 1) ∧ (e = f - 1) ∧ 
  (f = g - 1) ∧ (g = 100) ∧ (a = 94) :=
by
  -- This is the theorem statement. 
  sorry

end smallest_even_number_of_seven_l240_240768


namespace greatest_power_of_two_factor_l240_240102

theorem greatest_power_of_two_factor (a b c d : ℕ) (h1 : a = 10) (h2 : b = 1006) (h3 : c = 6) (h4 : d = 503) :
  ∃ k : ℕ, 2^k ∣ (a^b - c^d) ∧ ∀ j : ℕ, 2^j ∣ (a^b - c^d) → j ≤ 503 :=
sorry

end greatest_power_of_two_factor_l240_240102


namespace point_on_x_axis_l240_240712

theorem point_on_x_axis (a : ℝ) (h : a + 2 = 0) : (a - 1, a + 2) = (-3, 0) :=
by
  sorry

end point_on_x_axis_l240_240712


namespace factor_quadratic_l240_240247

theorem factor_quadratic (x : ℝ) : 
  x^2 + 6 * x = 1 → (x + 3)^2 = 10 := 
by
  intro h
  sorry

end factor_quadratic_l240_240247


namespace kendall_nickels_count_l240_240621

theorem kendall_nickels_count :
  ∃ (n : ℕ), n * 0.05 = 4 - (10 * 0.25 + 12 * 0.10) ∧ n = 6 :=
by
  have quarters_value : ℝ := 10 * 0.25
  have dimes_value : ℝ := 12 * 0.10
  have total_value : ℝ := 4
  have nickels_value : ℝ := total_value - (quarters_value + dimes_value)
  use 6
  split
  sorry
  sorry

end kendall_nickels_count_l240_240621


namespace circle_radius_of_square_perimeter_eq_area_l240_240352

theorem circle_radius_of_square_perimeter_eq_area (r : ℝ) (s : ℝ) (h1 : 2 * r = s) (h2 : 4 * s = 8 * r) (h3 : π * r ^ 2 = 8 * r) : r = 8 / π := by
  sorry

end circle_radius_of_square_perimeter_eq_area_l240_240352


namespace initial_number_of_peanuts_l240_240457

theorem initial_number_of_peanuts (x : ℕ) (h : x + 2 = 6) : x = 4 :=
sorry

end initial_number_of_peanuts_l240_240457


namespace initial_amount_calc_l240_240508

theorem initial_amount_calc 
  (M : ℝ)
  (H1 : M * 0.3675 = 350) :
  M = 952.38 :=
by
  sorry

end initial_amount_calc_l240_240508


namespace radius_of_circumscribed_circle_l240_240344

noncomputable def circumscribed_circle_radius (r : ℝ) : Prop :=
  let s := r * Real.sqrt 2 in
  4 * s = π * r ^ 2 -> r = 4 * Real.sqrt 2 / π

-- Statement of the theorem to be proved
theorem radius_of_circumscribed_circle (r : ℝ) : circumscribed_circle_radius r := 
sorry

end radius_of_circumscribed_circle_l240_240344


namespace smallest_b_for_perfect_square_l240_240104

theorem smallest_b_for_perfect_square (b : ℤ) (h1 : b > 4) (h2 : ∃ n : ℤ, 3 * b + 4 = n * n) : b = 7 :=
by
  sorry

end smallest_b_for_perfect_square_l240_240104


namespace num_divisors_of_8_factorial_l240_240032

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l240_240032


namespace milk_butterfat_problem_l240_240727

-- Define the values given in the problem
def b1 : ℝ := 0.35  -- butterfat percentage of initial milk
def v1 : ℝ := 8     -- volume of initial milk in gallons
def b2 : ℝ := 0.10  -- butterfat percentage of milk to be added
def bf : ℝ := 0.20  -- desired butterfat percentage of the final mixture

-- Define the proof statement
theorem milk_butterfat_problem :
  ∃ x : ℝ, (2.8 + 0.1 * x) / (v1 + x) = bf ↔ x = 12 :=
by {
  sorry
}

end milk_butterfat_problem_l240_240727


namespace decimal_equivalent_of_squared_fraction_l240_240378

theorem decimal_equivalent_of_squared_fraction : (1 / 5 : ℝ)^2 = 0.04 :=
by
  sorry

end decimal_equivalent_of_squared_fraction_l240_240378


namespace geometric_sequence_a9_l240_240588

open Nat

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 3 = 20) (h2 : a 6 = 5) 
  (h_geometric : ∀ m n, a ((m + n) / 2) ^ 2 = a m * a n) : 
  a 9 = 5 / 4 := 
by
  sorry

end geometric_sequence_a9_l240_240588


namespace problem1_problem2_l240_240121

-- Problem (1)
theorem problem1 (a b : ℝ) (h : 2 * a^2 + 3 * b = 6) : a^2 + (3 / 2) * b - 5 = -2 := 
sorry

-- Problem (2)
theorem problem2 (x : ℝ) (h : 14 * x + 5 - 21 * x^2 = -2) : 6 * x^2 - 4 * x + 5 = 7 := 
sorry

end problem1_problem2_l240_240121


namespace sphere_radius_l240_240237

-- Define the conditions
variable (r : ℝ) -- Radius of the sphere
variable (sphere_shadow : ℝ) (stick_height : ℝ) (stick_shadow : ℝ)

-- Given conditions
axiom sphere_shadow_equals_10 : sphere_shadow = 10
axiom stick_height_equals_1 : stick_height = 1
axiom stick_shadow_equals_2 : stick_shadow = 2

-- Using similar triangles and tangent relations, we want to prove the radius of sphere.
theorem sphere_radius (h1 : sphere_shadow = 10)
    (h2 : stick_height = 1)
    (h3 : stick_shadow = 2) : r = 5 :=
by
  -- Placeholder for the proof
  sorry

end sphere_radius_l240_240237


namespace trig_identity_condition_l240_240733

open Real

theorem trig_identity_condition (a : Real) (h : ∃ x ≥ 0, (tan a = -1 ∧ cos a ≠ 0)) :
  (sin a / sqrt (1 - sin a ^ 2) + sqrt (1 - cos a ^ 2) / cos a) = 0 :=
by
  sorry

end trig_identity_condition_l240_240733


namespace operation_result_l240_240692

-- Define the new operation x # y
def op (x y : ℕ) : ℤ := 2 * x * y - 3 * x + y

-- Prove that (6 # 4) - (4 # 6) = -8
theorem operation_result : op 6 4 - op 4 6 = -8 :=
by
  sorry

end operation_result_l240_240692


namespace john_saves_1200_yearly_l240_240748

noncomputable def former_rent_per_month (sq_ft_cost : ℝ) (sq_ft : ℝ) : ℝ :=
  sq_ft_cost * sq_ft

noncomputable def new_rent_per_month (total_cost : ℝ) (roommates : ℝ) : ℝ :=
  total_cost / roommates

noncomputable def monthly_savings (former_rent : ℝ) (new_rent : ℝ) : ℝ :=
  former_rent - new_rent

noncomputable def annual_savings (monthly_savings : ℝ) : ℝ :=
  monthly_savings * 12

theorem john_saves_1200_yearly :
  let former_rent := former_rent_per_month 2 750
  let new_rent := new_rent_per_month 2800 2
  let monthly_savings := monthly_savings former_rent new_rent
  annual_savings monthly_savings = 1200 := 
by 
  sorry

end john_saves_1200_yearly_l240_240748


namespace cookies_batches_needed_l240_240681

noncomputable def number_of_recipes (total_students : ℕ) (attendance_drop : ℝ) (cookies_per_batch : ℕ) : ℕ :=
  let remaining_students := (total_students : ℝ) * (1 - attendance_drop)
  let total_cookies := remaining_students * 2
  let recipes_needed := total_cookies / cookies_per_batch
  (Nat.ceil recipes_needed : ℕ)

theorem cookies_batches_needed :
  number_of_recipes 150 0.40 18 = 10 :=
by
  sorry

end cookies_batches_needed_l240_240681


namespace sphere_ratios_l240_240091

theorem sphere_ratios (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 ∧ (4 / 3 * π * r1^3) / (4 / 3 * π * r2^3) = 1 / 27 :=
by
  sorry

end sphere_ratios_l240_240091


namespace profit_percentage_is_23_16_l240_240675

   noncomputable def cost_price (mp : ℝ) : ℝ := 95 * mp
   noncomputable def selling_price (mp : ℝ) : ℝ := 120 * (mp - (0.025 * mp))
   noncomputable def profit_percent (cp sp : ℝ) : ℝ := ((sp - cp) / cp) * 100

   theorem profit_percentage_is_23_16 
     (mp : ℝ) (h_mp_gt_zero : mp > 0) : 
       profit_percent (cost_price mp) (selling_price mp) = 23.16 :=
   by 
     sorry
   
end profit_percentage_is_23_16_l240_240675


namespace minimum_box_value_l240_240596

def is_valid_pair (a b : ℤ) : Prop :=
  a * b = 15 ∧ (a^2 + b^2 ≥ 34)

theorem minimum_box_value :
  ∃ (a b : ℤ), is_valid_pair a b ∧ (∀ (a' b' : ℤ), is_valid_pair a' b' → a^2 + b^2 ≤ a'^2 + b'^2) ∧ a^2 + b^2 = 34 :=
by
  sorry

end minimum_box_value_l240_240596


namespace graph_passes_through_point_l240_240490

noncomputable def exponential_shift (a : ℝ) (x : ℝ) := a^(x - 2)

theorem graph_passes_through_point (a : ℝ) (h : a > 0) (h1 : a ≠ 1) : exponential_shift a 2 = 1 :=
by
  unfold exponential_shift
  sorry

end graph_passes_through_point_l240_240490


namespace regular_polygon_sides_l240_240919

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l240_240919


namespace roots_of_quadratic_eq_l240_240783

theorem roots_of_quadratic_eq (x : ℝ) : x^2 = 2 * x ↔ x = 0 ∨ x = 2 := by
  sorry

end roots_of_quadratic_eq_l240_240783


namespace num_pos_divisors_fact8_l240_240028

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l240_240028


namespace median_of_first_fifteen_positive_integers_l240_240660

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l240_240660


namespace expression_evaluation_l240_240101

theorem expression_evaluation : 4 * 10 + 5 * 11 + 12 * 4 + 4 * 9 = 179 :=
by
  sorry

end expression_evaluation_l240_240101


namespace fraction_integer_condition_special_integers_l240_240223

theorem fraction_integer_condition (p : ℕ) (h : (p + 2) % (p + 1) = 0) : p = 2 :=
by
  sorry

theorem special_integers (N : ℕ) (h1 : ∀ q : ℕ, N = 2 ^ p * 3 ^ q ∧ (2 * p + 1) * (2 * q + 1) = 3 * (p + 1) * (q + 1)) : 
  N = 144 ∨ N = 324 :=
by
  sorry

end fraction_integer_condition_special_integers_l240_240223


namespace arithmetic_sequences_ratio_l240_240725

theorem arithmetic_sequences_ratio (a b S T : ℕ → ℕ) (h : ∀ n, S n / T n = 2 * n / (3 * n + 1)) :
  (a 2) / (b 3 + b 7) + (a 8) / (b 4 + b 6) = 9 / 14 :=
  sorry

end arithmetic_sequences_ratio_l240_240725


namespace revenue_increase_20_percent_l240_240527

variable (P Q : ℝ)

def original_revenue (P Q : ℝ) : ℝ := P * Q
def new_price (P : ℝ) : ℝ := P * 1.5
def new_quantity (Q : ℝ) : ℝ := Q * 0.8
def new_revenue (P Q : ℝ) : ℝ := (new_price P) * (new_quantity Q)

theorem revenue_increase_20_percent (P Q : ℝ) : 
  (new_revenue P Q) = 1.2 * (original_revenue P Q) := by
  sorry

end revenue_increase_20_percent_l240_240527


namespace rational_iff_arithmetic_progression_l240_240495

theorem rational_iff_arithmetic_progression (x : ℝ) : 
  (∃ (i j k : ℤ), i < j ∧ j < k ∧ (x + i) + (x + k) = 2 * (x + j)) ↔ 
  (∃ n d : ℤ, d ≠ 0 ∧ x = n / d) := 
sorry

end rational_iff_arithmetic_progression_l240_240495


namespace snooker_tournament_l240_240395

theorem snooker_tournament : 
  ∀ (V G : ℝ),
    V + G = 320 →
    40 * V + 15 * G = 7500 →
    V ≥ 80 →
    G ≥ 100 →
    G - V = 104 :=
by
  intros V G h1 h2 h3 h4
  sorry

end snooker_tournament_l240_240395


namespace sum_of_base_radii_l240_240257

theorem sum_of_base_radii (R : ℝ) (hR : R = 5) (a b c : ℝ) 
  (h_ratios : a = 1 ∧ b = 2 ∧ c = 3) 
  (r1 r2 r3 : ℝ) 
  (h_r1 : r1 = (a / (a + b + c)) * R)
  (h_r2 : r2 = (b / (a + b + c)) * R)
  (h_r3 : r3 = (c / (a + b + c)) * R) : 
  r1 + r2 + r3 = 5 := 
by
  subst hR
  simp [*, ←add_assoc, add_comm]
  sorry

end sum_of_base_radii_l240_240257


namespace inverse_of_g_l240_240774

noncomputable def u (x : ℝ) : ℝ := sorry
noncomputable def v (x : ℝ) : ℝ := sorry
noncomputable def w (x : ℝ) : ℝ := sorry

noncomputable def u_inv (x : ℝ) : ℝ := sorry
noncomputable def v_inv (x : ℝ) : ℝ := sorry
noncomputable def w_inv (x : ℝ) : ℝ := sorry

lemma u_inverse : ∀ x, u_inv (u x) = x ∧ u (u_inv x) = x := sorry
lemma v_inverse : ∀ x, v_inv (v x) = x ∧ v (v_inv x) = x := sorry
lemma w_inverse : ∀ x, w_inv (w x) = x ∧ w (w_inv x) = x := sorry

noncomputable def g (x : ℝ) : ℝ := v (u (w x))

noncomputable def g_inv (x : ℝ) : ℝ := w_inv (u_inv (v_inv x))

theorem inverse_of_g :
  ∀ x : ℝ, g_inv (g x) = x ∧ g (g_inv x) = x :=
by
  intro x
  -- proof omitted
  sorry

end inverse_of_g_l240_240774


namespace smallest_number_remainder_l240_240811

open Nat

theorem smallest_number_remainder
  (b : ℕ)
  (h1 : b % 4 = 2)
  (h2 : b % 3 = 2)
  (h3 : b % 5 = 3) :
  b = 38 :=
sorry

end smallest_number_remainder_l240_240811


namespace radius_of_circumscribed_circle_l240_240354

theorem radius_of_circumscribed_circle (r : ℝ) (π : ℝ) (h : 4 * r * Real.sqrt 2 = π * r * r) : 
  r = 4 * Real.sqrt 2 / π :=
by
  sorry

end radius_of_circumscribed_circle_l240_240354


namespace lake_view_population_l240_240797

-- Define the populations of the cities
def population_of_Seattle : ℕ := 20000 -- Derived from the solution
def population_of_Boise : ℕ := (3 / 5) * population_of_Seattle
def population_of_Lake_View : ℕ := population_of_Seattle + 4000
def total_population : ℕ := population_of_Seattle + population_of_Boise + population_of_Lake_View

-- Statement to prove
theorem lake_view_population :
  total_population = 56000 →
  population_of_Lake_View = 24000 :=
sorry

end lake_view_population_l240_240797


namespace num_divisors_8_factorial_l240_240014

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l240_240014


namespace positive_divisors_of_8_factorial_l240_240021

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l240_240021


namespace determine_p_l240_240558

variable (x y z p : ℝ)

theorem determine_p (h1 : 8 / (x + y) = p / (x + z)) (h2 : p / (x + z) = 12 / (z - y)) : p = 20 :=
sorry

end determine_p_l240_240558


namespace tip_percentage_is_30_l240_240249

theorem tip_percentage_is_30
  (appetizer_cost : ℝ)
  (entree_cost : ℝ)
  (num_entrees : ℕ)
  (dessert_cost : ℝ)
  (total_price_including_tip : ℝ)
  (h_appetizer : appetizer_cost = 9.0)
  (h_entree : entree_cost = 20.0)
  (h_num_entrees : num_entrees = 2)
  (h_dessert : dessert_cost = 11.0)
  (h_total : total_price_including_tip = 78.0) :
  let total_before_tip := appetizer_cost + num_entrees * entree_cost + dessert_cost
  let tip_amount := total_price_including_tip - total_before_tip
  let tip_percentage := (tip_amount / total_before_tip) * 100
  tip_percentage = 30 :=
by
  sorry

end tip_percentage_is_30_l240_240249


namespace regular_polygon_sides_l240_240883

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l240_240883


namespace find_x_squared_plus_y_squared_l240_240575

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = -8) : x^2 + y^2 = 33 := 
by 
  sorry

end find_x_squared_plus_y_squared_l240_240575


namespace power_function_value_at_two_l240_240171

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_value_at_two (a : ℝ) (h : f (1/2) a = 8) : f 2 a = 1 / 8 := by
  sorry

end power_function_value_at_two_l240_240171


namespace one_third_of_1206_is_100_5_percent_of_400_l240_240227

theorem one_third_of_1206_is_100_5_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end one_third_of_1206_is_100_5_percent_of_400_l240_240227


namespace regular_polygon_sides_l240_240889

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240889


namespace area_enclosed_by_equation_is_96_l240_240685

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- The theorem to prove the area enclosed by the graph is 96 square units
theorem area_enclosed_by_equation_is_96 :
  (∃ x y : ℝ, equation x y) → ∃ A : ℝ, A = 96 :=
sorry

end area_enclosed_by_equation_is_96_l240_240685


namespace single_dog_barks_per_minute_l240_240390

theorem single_dog_barks_per_minute (x : ℕ) (h : 10 * 2 * x = 600) : x = 30 :=
by
  sorry

end single_dog_barks_per_minute_l240_240390


namespace molecular_weight_CCl4_l240_240103

theorem molecular_weight_CCl4 (MW_7moles_CCl4 : ℝ) (h : MW_7moles_CCl4 = 1064) : 
  MW_7moles_CCl4 / 7 = 152 :=
by
  sorry

end molecular_weight_CCl4_l240_240103


namespace regular_polygon_sides_l240_240857

theorem regular_polygon_sides (n : ℕ) (h1 : 0 < n) (h2 : 18 = 360 / n) : n = 20 :=
sorry

end regular_polygon_sides_l240_240857


namespace A_union_B_eq_B_l240_240479

-- Define set A
def A : Set ℝ := {-1, 0, 1}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- The proof problem
theorem A_union_B_eq_B : A ∪ B = B := 
  sorry

end A_union_B_eq_B_l240_240479


namespace regular_polygon_sides_l240_240935

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l240_240935


namespace rectangle_area_is_1600_l240_240339

theorem rectangle_area_is_1600 (l w : ℕ) 
  (h₁ : l = 4 * w)
  (h₂ : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end rectangle_area_is_1600_l240_240339


namespace num_positive_divisors_8_factorial_l240_240017

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l240_240017


namespace outfits_count_l240_240639

theorem outfits_count (shirts ties : ℕ) (h_shirts : shirts = 7) (h_ties : ties = 6) : 
  (shirts * (ties + 1) = 49) :=
by
  sorry

end outfits_count_l240_240639


namespace incorrect_statement_count_l240_240547

theorem incorrect_statement_count :
  let statements := ["Every number has a square root",
                     "The square root of a number must be positive",
                     "The square root of a^2 is a",
                     "The square root of (π - 4)^2 is π - 4",
                     "A square root cannot be negative"]
  let incorrect := [statements.get! 0, statements.get! 1, statements.get! 2, statements.get! 3]
  incorrect.length = 4 :=
by
  sorry

end incorrect_statement_count_l240_240547


namespace regular_polygon_sides_l240_240838

theorem regular_polygon_sides (n : ℕ) (h_regular : regular_polygon P) (h_exterior_angle : exterior_angle P = 18) : n = 20 := 
by
  sorry

end regular_polygon_sides_l240_240838


namespace charles_earnings_l240_240686

def housesit_rate : ℝ := 15
def dog_walk_rate : ℝ := 22
def hours_housesit : ℝ := 10
def num_dogs : ℝ := 3

theorem charles_earnings :
  housesit_rate * hours_housesit + dog_walk_rate * num_dogs = 216 :=
by
  sorry

end charles_earnings_l240_240686


namespace motorcyclist_travel_time_l240_240372

-- Define the conditions and the proof goal:
theorem motorcyclist_travel_time :
  ∀ (z : ℝ) (t₁ t₂ t₃ : ℝ),
    t₂ = 60 →
    t₃ = 3240 →
    (t₃ - 5) / (z / 40 - z / t₁) = 10 →
    t₃ / (z / 40) = 10 + t₂ / (z / 60 - z / t₁) →
    t₁ = 80 :=
by
  intros z t₁ t₂ t₃ h1 h2 h3 h4
  sorry

end motorcyclist_travel_time_l240_240372


namespace sum_of_geometric_ratios_l240_240309

theorem sum_of_geometric_ratios (k a2 a3 b2 b3 p r : ℝ)
  (h_seq1 : a2 = k * p)
  (h_seq2 : a3 = k * p^2)
  (h_seq3 : b2 = k * r)
  (h_seq4 : b3 = k * r^2)
  (h_diff : a3 - b3 = 3 * (a2 - b2) - k) :
  p + r = 2 :=
by
  sorry

end sum_of_geometric_ratios_l240_240309


namespace length_of_midsegment_l240_240184

/-- Given a quadrilateral ABCD where sides AB and CD are parallel with lengths 7 and 3 
    respectively, and the other sides BC and DA are of lengths 5 and 4 respectively, 
    prove that the length of the segment joining the midpoints of sides BC and DA is 5. -/
theorem length_of_midsegment (A B C D : ℝ × ℝ)
  (HAB : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 7 ∧ B.2 = 0)
  (HBC : dist B C = 5)
  (HCD : dist C D = 3)
  (HDA : dist D A = 4)
  (Hparallel : B.2 = 0 ∧ D.2 ≠ 0 → C.2 = D.2) :
  dist ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((A.1 + D.1) / 2, (A.2 + D.2) / 2) = 5 :=
sorry

end length_of_midsegment_l240_240184


namespace dodgeball_cost_l240_240092

theorem dodgeball_cost (B : ℝ) 
  (hb1 : 1.20 * B = 90) 
  (hb2 : B / 15 = 5) :
  ∃ (cost_per_dodgeball : ℝ), cost_per_dodgeball = 5 := by
sorry

end dodgeball_cost_l240_240092


namespace parabola_translation_l240_240099

theorem parabola_translation :
  ∀(x y : ℝ), y = - (1 / 3) * (x - 5) ^ 2 + 3 →
  ∃(x' y' : ℝ), y' = -(1/3) * x'^2 + 6 := by
  sorry

end parabola_translation_l240_240099


namespace same_last_k_digits_pow_l240_240604

theorem same_last_k_digits_pow (A B : ℤ) (k n : ℕ) 
  (h : A % 10^k = B % 10^k) : 
  (A^n % 10^k = B^n % 10^k) := 
by
  sorry

end same_last_k_digits_pow_l240_240604


namespace sum_of_longest_altitudes_l240_240040

-- Defines the sides of the triangle
def side_a : ℕ := 9
def side_b : ℕ := 12
def side_c : ℕ := 15

-- States it is a right triangle (by Pythagorean triple)
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the altitude lengths in a right triangle
def altitude_a (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := a
def altitude_b (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := b

-- Problem statement
theorem sum_of_longest_altitudes :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧ a = side_a ∧ b = side_b ∧ c = side_c ∧
  altitude_a a b c sorry + altitude_b a b c sorry = 21 :=
by
  use side_a, side_b, side_c
  split
  sorry -- Proof that 9, 12, and 15 form a right triangle.
  split; refl
  split; refl
  sorry -- Proof that the sum of altitudes is 21.

end sum_of_longest_altitudes_l240_240040


namespace one_kid_six_whiteboards_l240_240265

theorem one_kid_six_whiteboards (k: ℝ) (b1 b2: ℝ) (t1 t2: ℝ) 
  (hk: k = 1) (hb1: b1 = 3) (hb2: b2 = 6) 
  (ht1: t1 = 20) 
  (H: 4 * t1 / b1 = t2 / b2) : 
  t2 = 160 := 
by
  -- provide the proof here
  sorry

end one_kid_six_whiteboards_l240_240265


namespace complex_fourth_power_l240_240150

theorem complex_fourth_power (i : ℂ) (hi : i^2 = -1) : (1 - i)^4 = -4 := 
sorry

end complex_fourth_power_l240_240150


namespace mathematicians_correctness_l240_240333

theorem mathematicians_correctness :
  (2 / 5 + 3 / 8) / (5 + 8) = 5 / 13 ∧
  (4 / 10 + 3 / 8) / (10 + 8) = 7 / 18 ∧
  ¬ (3 / 8 < 17 / 40 ∧ 17 / 40 < 2 / 5) :=
by {
  sorry
}

end mathematicians_correctness_l240_240333


namespace compute_complex_power_l240_240147

theorem compute_complex_power (i : ℂ) (h : i^2 = -1) : (1 - i)^4 = -4 :=
by
  sorry

end compute_complex_power_l240_240147


namespace regular_polygon_sides_l240_240957

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end regular_polygon_sides_l240_240957


namespace regular_polygon_sides_l240_240894

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l240_240894


namespace division_addition_correct_l240_240222

theorem division_addition_correct : 0.2 / 0.005 + 0.1 = 40.1 :=
by
  sorry

end division_addition_correct_l240_240222


namespace algae_colony_growth_l240_240545

def initial_cells : ℕ := 5
def days : ℕ := 10
def tripling_period : ℕ := 3
def cell_growth_ratio : ℕ := 3

noncomputable def cells_after_n_days (init_cells : ℕ) (day_count : ℕ) (period : ℕ) (growth_ratio : ℕ) : ℕ :=
  let steps := day_count / period
  init_cells * growth_ratio^steps

theorem algae_colony_growth : cells_after_n_days initial_cells days tripling_period cell_growth_ratio = 135 :=
  by sorry

end algae_colony_growth_l240_240545


namespace sum_of_other_endpoint_coordinates_l240_240633

theorem sum_of_other_endpoint_coordinates {x y : ℝ} :
  let P1 := (1, 2)
  let M := (5, 6)
  let P2 := (x, y)
  (M.1 = (P1.1 + P2.1) / 2 ∧ M.2 = (P1.2 + P2.2) / 2) → (x + y) = 19 :=
by
  intros P1 M P2 h
  sorry

end sum_of_other_endpoint_coordinates_l240_240633


namespace correct_operation_l240_240381

-- Definitions based on conditions
def exprA (a b : ℤ) : ℤ := 3 * a * b - a * b
def exprB (a : ℤ) : ℤ := -3 * a^2 - 5 * a^2
def exprC (x : ℤ) : ℤ := -3 * x - 2 * x

-- Statement to prove that exprB is correct
theorem correct_operation (a : ℤ) : exprB a = -8 * a^2 := by
  sorry

end correct_operation_l240_240381


namespace total_weight_lifted_l240_240802

-- Definitions based on conditions
def original_lift : ℝ := 80
def after_training : ℝ := original_lift * 2
def specialization_increment : ℝ := after_training * 0.10
def specialized_lift : ℝ := after_training + specialization_increment

-- Statement of the theorem to prove total weight lifted
theorem total_weight_lifted : 
  (specialized_lift * 2) = 352 :=
sorry

end total_weight_lifted_l240_240802


namespace axis_of_symmetry_compare_m_n_range_t_max_t_l240_240617

-- Condition: Definition of the parabola
def parabola (t x : ℝ) := x^2 - 2 * t * x + 1

-- Problem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) : 
  ∀ (y x : ℝ), parabola t x = y -> x = t :=
sorry

-- Problem 2: Comparing m and n
theorem compare_m_n (t m n : ℝ) :
  parabola t (t - 2) = m ∧ parabola t (t + 3) = n -> n > m := 
sorry

-- Problem 3: Range of t for y₁ ≤ y₂
theorem range_t (t x₁ y₁ y₂ : ℝ) :
  -1 ≤ x₁ ∧ x₁ < 3 ∧ parabola t x₁ = y₁ ∧ parabola t 3 = y₂ -> y₁ ≤ y₂ → t ≤ 1 :=
sorry

-- Problem 4: Maximum t for y₁ ≥ y₂
theorem max_t (t y₁ y₂ : ℝ) :
  (parabola t (t + 1) = y₁ ∧ parabola t (2 * t - 4) = y₂) → y₁ ≥ y₂ → t ≤ 5 :=
sorry

end axis_of_symmetry_compare_m_n_range_t_max_t_l240_240617
