import Mathlib

namespace students_like_both_l2134_213488

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_like_mountains : ℕ := 289
def students_like_sea : ℕ := 337
def students_like_neither : ℕ := 56

-- Statement to prove
theorem students_like_both : 
  students_like_mountains + students_like_sea - 182 + students_like_neither = total_students := 
by
  sorry

end students_like_both_l2134_213488


namespace range_of_m_l2134_213481

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, (x^2 + 1) * (x^2 - 8*x - 20) ≤ 0 → (-2 ≤ x → x ≤ 10))
    (h2 : ∀ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 → (1 - m ≤ x → x ≤ 1 + m))
    (h3 : m > 0)
    (h4 : ∀ x : ℝ, ¬ ((x^2 + 1) * (x^2 - 8*x - 20) ≤ 0) → ¬ (x^2 - 2*x + 1 - m^2 ≤ 0) → (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
  m ≥ 9 := 
sorry

end range_of_m_l2134_213481


namespace find_range_of_x_l2134_213470

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 ^ x else 2 ^ (-x)

theorem find_range_of_x (x : ℝ) : 
  f (1 - 2 * x) < f 3 ↔ (-1 < x ∧ x < 2) := 
sorry

end find_range_of_x_l2134_213470


namespace jerry_daughters_games_l2134_213476

theorem jerry_daughters_games (x y : ℕ) (h : 4 * x + 2 * x + 4 * y + 2 * y = 96) (hx : x = y) :
  x = 8 ∧ y = 8 :=
by
  have h1 : 6 * x + 6 * y = 96 := by linarith
  have h2 : x = y := hx
  sorry

end jerry_daughters_games_l2134_213476


namespace part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l2134_213474

open Set Real

def setA (a : ℝ) : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ a + 5}
def setB : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem part_one_a_two_complement_union (a : ℝ) (h : a = 2) :
  compl (setA a ∪ setB) = Iic 2 ∪ Ici 10 := sorry

theorem part_one_a_two_complement_intersection (a : ℝ) (h : a = 2) :
  compl (setA a) ∩ setB = Ioo 2 3 ∪ Ioo 7 10 := sorry

theorem part_two_subset (a : ℝ) (h : setA a ⊆ setB) :
  a < 5 := sorry

end part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l2134_213474


namespace parabola_focus_distance_l2134_213453

theorem parabola_focus_distance (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) (h_dist_y_axis : |P.1| = 4) : 
  dist P (4, 0) = 5 :=
sorry

end parabola_focus_distance_l2134_213453


namespace digits_divisible_by_101_l2134_213413

theorem digits_divisible_by_101 :
  ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ (2013 * 100 + 10 * x + y) % 101 = 0 ∧ x = 9 ∧ y = 4 := by
  sorry

end digits_divisible_by_101_l2134_213413


namespace find_x_l2134_213436

variable (N x : ℕ)
variable (h1 : N = 500 * x + 20)
variable (h2 : 4 * 500 + 20 = 2020)

theorem find_x : x = 4 := by
  -- The proof code will go here
  sorry

end find_x_l2134_213436


namespace correct_option_c_l2134_213455

theorem correct_option_c (x : ℝ) : -2 * (x + 1) = -2 * x - 2 :=
  by
  -- Proof can be omitted
  sorry

end correct_option_c_l2134_213455


namespace camille_saw_31_birds_l2134_213449

def num_cardinals : ℕ := 3
def num_robins : ℕ := 4 * num_cardinals
def num_blue_jays : ℕ := 2 * num_cardinals
def num_sparrows : ℕ := 3 * num_cardinals + 1
def total_birds : ℕ := num_cardinals + num_robins + num_blue_jays + num_sparrows

theorem camille_saw_31_birds : total_birds = 31 := by
  sorry

end camille_saw_31_birds_l2134_213449


namespace train_length_is_correct_l2134_213499

noncomputable def length_of_train 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * time_to_cross
  distance_covered - bridge_length

theorem train_length_is_correct :
  length_of_train 23.998080153587715 140 36 = 99.98080153587715 :=
by sorry

end train_length_is_correct_l2134_213499


namespace find_a10_l2134_213441

def seq (a : ℕ → ℝ) : Prop :=
∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q

theorem find_a10 (a : ℕ → ℝ) (h_seq : seq a) (h_a2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end find_a10_l2134_213441


namespace find_d_minus_r_l2134_213421

theorem find_d_minus_r :
  ∃ d r : ℕ, 1 < d ∧ 1223 % d = r ∧ 1625 % d = r ∧ 2513 % d = r ∧ d - r = 1 :=
by
  sorry

end find_d_minus_r_l2134_213421


namespace intersection_complement_A_U_B_l2134_213431

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def set_A : Set ℕ := {2, 4, 6}
def set_B : Set ℕ := {1, 3, 5, 7}

theorem intersection_complement_A_U_B :
  set_A ∩ (universal_set \ set_B) = {2, 4, 6} :=
by {
  sorry
}

end intersection_complement_A_U_B_l2134_213431


namespace length_BC_fraction_AD_l2134_213405

theorem length_BC_fraction_AD {A B C D : Type} {AB BD AC CD AD BC : ℕ} 
  (h1 : AB = 4 * BD) (h2 : AC = 9 * CD) (h3 : AD = AB + BD) (h4 : AD = AC + CD)
  (h5 : B ≠ A) (h6 : C ≠ A) (h7 : A ≠ D) : BC = AD / 10 :=
by
  sorry

end length_BC_fraction_AD_l2134_213405


namespace gcd_12345_6789_l2134_213467

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l2134_213467


namespace alyssa_spent_on_grapes_l2134_213493

theorem alyssa_spent_on_grapes (t c g : ℝ) (h1 : t = 21.93) (h2 : c = 9.85) (h3 : t = g + c) : g = 12.08 :=
by
  sorry

end alyssa_spent_on_grapes_l2134_213493


namespace hyperbola_asymptote_l2134_213412

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y, 3 * x + 2 * y = 0 ∨ 3 * x - 2 * y = 0) →
  (∀ x y, y * y = 9 * (x * x / (a * a) - 1)) →
  a = 2 :=
by
  intros asymptote_constr hyp
  sorry

end hyperbola_asymptote_l2134_213412


namespace area_of_rectangle_l2134_213495

namespace RectangleArea

variable (l b : ℕ)
variable (h1 : l = 3 * b)
variable (h2 : 2 * (l + b) = 88)

theorem area_of_rectangle : l * b = 363 :=
by
  -- We will prove this in Lean 
  sorry

end RectangleArea

end area_of_rectangle_l2134_213495


namespace min_cuts_to_one_meter_pieces_l2134_213422

theorem min_cuts_to_one_meter_pieces (x y : ℕ) (hx : x + y = 30) (hl : 3 * x + 4 * y = 100) : (2 * x + 3 * y) = 70 := 
by sorry

end min_cuts_to_one_meter_pieces_l2134_213422


namespace pizza_problem_l2134_213461

theorem pizza_problem (m d : ℕ) :
  (7 * m + 2 * d > 36) ∧ (8 * m + 4 * d < 48) ↔ (m = 5) ∧ (d = 1) := by
  sorry

end pizza_problem_l2134_213461


namespace coupon1_best_discount_l2134_213419

noncomputable def listed_prices : List ℝ := [159.95, 179.95, 199.95, 219.95, 239.95]

theorem coupon1_best_discount (x : ℝ) (h₁ : x ∈ listed_prices) (h₂ : x > 120) :
  0.15 * x > 25 ∧ 0.15 * x > 0.20 * (x - 120) ↔ 
  x = 179.95 ∨ x = 199.95 ∨ x = 219.95 ∨ x = 239.95 :=
sorry

end coupon1_best_discount_l2134_213419


namespace min_x_plus_y_l2134_213439

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 := by
  sorry

end min_x_plus_y_l2134_213439


namespace average_rainfall_virginia_l2134_213477

noncomputable def average_rainfall : ℝ :=
  (3.79 + 4.5 + 3.95 + 3.09 + 4.67) / 5

theorem average_rainfall_virginia : average_rainfall = 4 :=
by
  sorry

end average_rainfall_virginia_l2134_213477


namespace number_of_cubes_with_three_faces_painted_l2134_213401

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end number_of_cubes_with_three_faces_painted_l2134_213401


namespace absolute_value_inequality_l2134_213469

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 4) ↔ (-1 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 7) := 
by sorry

end absolute_value_inequality_l2134_213469


namespace total_population_of_city_l2134_213487

theorem total_population_of_city (P : ℝ) (h : 0.85 * P = 85000) : P = 100000 :=
  by
  sorry

end total_population_of_city_l2134_213487


namespace max_fraction_diagonals_sides_cyclic_pentagon_l2134_213434

theorem max_fraction_diagonals_sides_cyclic_pentagon (a b c d e A B C D E : ℝ)
  (h1 : b * e + a * A = C * D)
  (h2 : c * a + b * B = D * E)
  (h3 : d * b + c * C = E * A)
  (h4 : e * c + d * D = A * B)
  (h5 : a * d + e * E = B * C) :
  (a * b * c * d * e) / (A * B * C * D * E) ≤ (5 * Real.sqrt 5 - 11) / 2 :=
sorry

end max_fraction_diagonals_sides_cyclic_pentagon_l2134_213434


namespace closest_to_fraction_l2134_213420

theorem closest_to_fraction (options : List ℝ) (h1 : options = [2000, 1500, 200, 2500, 3000]) :
  ∃ closest : ℝ, closest ∈ options ∧ closest = 2000 :=
by
  sorry

end closest_to_fraction_l2134_213420


namespace evaluate_magnitude_of_product_l2134_213402

theorem evaluate_magnitude_of_product :
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  Complex.abs (z1 * z2) = 4 * Real.sqrt 43 := by
  let z1 := (3 * Real.sqrt 2 - 5 * Complex.I)
  let z2 := (2 * Real.sqrt 3 + 2 * Complex.I)
  suffices Complex.abs z1 * Complex.abs z2 = 4 * Real.sqrt 43 by sorry
  sorry

end evaluate_magnitude_of_product_l2134_213402


namespace joy_reading_rate_l2134_213454

theorem joy_reading_rate
  (h1 : ∀ t: ℕ, t = 20 → ∀ p: ℕ, p = 8 → ∀ t': ℕ, t' = 60 → ∃ p': ℕ, p' = (p * t') / t)
  (h2 : ∀ t: ℕ, t = 5 * 60 → ∀ p: ℕ, p = 120):
  ∃ r: ℕ, r = 24 :=
by
  sorry

end joy_reading_rate_l2134_213454


namespace find_m_and_c_l2134_213443

-- Definitions & conditions
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := -1, y := 3 }
def B (m : ℝ) : Point := { x := -6, y := m }

def line (c : ℝ) (p : Point) : Prop := p.x + p.y + c = 0

-- Theorem statement
theorem find_m_and_c (m : ℝ) (c : ℝ) (hc : line c A) (hcB : line c (B m)) :
  m = 3 ∧ c = -2 :=
  by
  sorry

end find_m_and_c_l2134_213443


namespace num_convex_pentagons_l2134_213458

theorem num_convex_pentagons (n m : ℕ) (hn : n = 15) (hm : m = 5) : 
  Nat.choose n m = 3003 := by
  sorry

end num_convex_pentagons_l2134_213458


namespace equation_of_line_through_point_with_equal_intercepts_l2134_213478

open LinearAlgebra

theorem equation_of_line_through_point_with_equal_intercepts :
  ∃ (a b c : ℝ), (a * 1 + b * 2 + c = 0) ∧ (a * b < 0) ∧ ∀ x y : ℝ, 
  (a * x + b * y + c = 0 ↔ (2 * x - y = 0 ∨ x + y - 3 = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l2134_213478


namespace non_integer_x_and_y_impossible_l2134_213428

theorem non_integer_x_and_y_impossible 
  (x y : ℚ) (m n : ℤ) 
  (h1 : 5 * x + 7 * y = m)
  (h2 : 7 * x + 10 * y = n) : 
  ∃ (x y : ℤ), 5 * x + 7 * y = m ∧ 7 * x + 10 * y = n := 
sorry

end non_integer_x_and_y_impossible_l2134_213428


namespace orchard_produce_l2134_213437

theorem orchard_produce (num_apple_trees num_orange_trees apple_baskets_per_tree apples_per_basket orange_baskets_per_tree oranges_per_basket : ℕ) 
  (h1 : num_apple_trees = 50) 
  (h2 : num_orange_trees = 30) 
  (h3 : apple_baskets_per_tree = 25) 
  (h4 : apples_per_basket = 18)
  (h5 : orange_baskets_per_tree = 15) 
  (h6 : oranges_per_basket = 12) 
: (num_apple_trees * (apple_baskets_per_tree * apples_per_basket) = 22500) ∧ 
  (num_orange_trees * (orange_baskets_per_tree * oranges_per_basket) = 5400) :=
  by 
  sorry

end orchard_produce_l2134_213437


namespace expand_expression_l2134_213450

theorem expand_expression : (x-3)*(x+3)*(x^2 + 9) = x^4 - 81 :=
by
  sorry

end expand_expression_l2134_213450


namespace day_after_75_days_l2134_213472

theorem day_after_75_days (day_of_week : ℕ → String) (h : day_of_week 0 = "Tuesday") :
  day_of_week 75 = "Sunday" :=
sorry

end day_after_75_days_l2134_213472


namespace perpendicular_chords_square_sum_l2134_213484

theorem perpendicular_chords_square_sum (d : ℝ) (r : ℝ) (x y : ℝ) 
  (h1 : r = d / 2)
  (h2 : x = r)
  (h3 : y = r) 
  : (x^2 + y^2) + (x^2 + y^2) = d^2 :=
by
  sorry

end perpendicular_chords_square_sum_l2134_213484


namespace M_lies_in_third_quadrant_l2134_213482

noncomputable def harmonious_point (a b : ℝ) : Prop :=
  3 * a = 2 * b + 5

noncomputable def point_M_harmonious (m : ℝ) : Prop :=
  harmonious_point (m - 1) (3 * m + 2)

theorem M_lies_in_third_quadrant (m : ℝ) (hM : point_M_harmonious m) : 
  (m - 1 < 0 ∧ 3 * m + 2 < 0) :=
by {
  sorry
}

end M_lies_in_third_quadrant_l2134_213482


namespace simplify_expression_l2134_213416

variable (a : ℝ)

theorem simplify_expression : a * (a + 2) - 2 * a = a^2 := by 
  sorry

end simplify_expression_l2134_213416


namespace intersection_A_B_l2134_213466

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { x | x < 0 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 0 } :=
sorry

end intersection_A_B_l2134_213466


namespace n_fifth_plus_4n_mod_5_l2134_213457

theorem n_fifth_plus_4n_mod_5 (n : ℕ) : (n^5 + 4 * n) % 5 = 0 := 
by
  sorry

end n_fifth_plus_4n_mod_5_l2134_213457


namespace depreciation_rate_l2134_213442

theorem depreciation_rate (initial_value final_value : ℝ) (years : ℕ) (r : ℝ)
  (h_initial : initial_value = 128000)
  (h_final : final_value = 54000)
  (h_years : years = 3)
  (h_equation : final_value = initial_value * (1 - r) ^ years) :
  r = 0.247 :=
sorry

end depreciation_rate_l2134_213442


namespace tank_filled_to_depth_l2134_213497

noncomputable def tank_volume (R H r d : ℝ) : ℝ := R^2 * H * Real.pi - (r^2 * H * Real.pi)

theorem tank_filled_to_depth (R H r d : ℝ) (h_cond : R = 5 ∧ H = 12 ∧ r = 2 ∧ d = 3) :
  tank_volume R H r d = 110 * Real.pi - 96 :=
sorry

end tank_filled_to_depth_l2134_213497


namespace non_zero_x_satisfies_equation_l2134_213403

theorem non_zero_x_satisfies_equation :
  ∃ (x : ℝ), (x ≠ 0) ∧ (7 * x)^5 = (14 * x)^4 ∧ x = 16 / 7 :=
by {
  sorry
}

end non_zero_x_satisfies_equation_l2134_213403


namespace option_A_option_B_option_C_option_D_l2134_213429

variables {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b)

-- A: Prove that \(a(6 - a) \leq 9\).
theorem option_A (h : 0 < a ∧ 0 < b) : a * (6 - a) ≤ 9 := sorry

-- B: Prove that if \(ab = a + b + 3\), then \(ab \geq 9\).
theorem option_B (h : ab = a + b + 3) : ab ≥ 9 := sorry

-- C: Prove that the minimum value of \(a^2 + \frac{4}{a^2 + 3}\) is not equal to 1.
theorem option_C : ∀ a > 0, (a^2 + 4 / (a^2 + 3) ≠ 1) := sorry

-- D: Prove that if \(a + b = 2\), then \(\frac{1}{a} + \frac{2}{b} \geq \frac{3}{2} + \sqrt{2}\).
theorem option_D (h : a + b = 2) : (1 / a + 2 / b) ≥ (3 / 2 + Real.sqrt 2) := sorry

end option_A_option_B_option_C_option_D_l2134_213429


namespace ratio_of_numbers_l2134_213445

theorem ratio_of_numbers
  (greater less : ℕ)
  (h1 : greater = 64)
  (h2 : less = 32)
  (h3 : greater + less = 96)
  (h4 : ∃ k : ℕ, greater = k * less) :
  greater / less = 2 := by
  sorry

end ratio_of_numbers_l2134_213445


namespace expression_value_l2134_213424

theorem expression_value : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end expression_value_l2134_213424


namespace cost_price_correct_l2134_213417

noncomputable def cost_price_per_meter (selling_price_per_meter : ℝ) (total_meters : ℝ) (loss_per_meter : ℝ) :=
  (selling_price_per_meter * total_meters + loss_per_meter * total_meters) / total_meters

theorem cost_price_correct :
  cost_price_per_meter 18000 500 5 = 41 :=
by 
  sorry

end cost_price_correct_l2134_213417


namespace debby_drinking_days_l2134_213471

def starting_bottles := 264
def daily_consumption := 15
def bottles_left := 99

theorem debby_drinking_days : (starting_bottles - bottles_left) / daily_consumption = 11 :=
by
  -- proof steps will go here
  sorry

end debby_drinking_days_l2134_213471


namespace constant_expression_l2134_213489

variable {x y m n : ℝ}

theorem constant_expression (hx : x^2 = 25) (hy : ∀ y : ℝ, (x + y) * (x - 2 * y) - m * y * (n * x - y) = 25) :
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end constant_expression_l2134_213489


namespace probability_of_sum_six_two_dice_l2134_213440

noncomputable def probability_sum_six : ℚ := 5 / 36

theorem probability_of_sum_six_two_dice (dice_faces : ℕ := 6) : 
  ∃ (p : ℚ), p = probability_sum_six :=
by
  sorry

end probability_of_sum_six_two_dice_l2134_213440


namespace sum_f_positive_l2134_213415

noncomputable def f (x : ℝ) : ℝ := (x ^ 3) / (Real.cos x)

theorem sum_f_positive 
  (x1 x2 x3 : ℝ)
  (hdom1 : abs x1 < Real.pi / 2)
  (hdom2 : abs x2 < Real.pi / 2)
  (hdom3 : abs x3 < Real.pi / 2)
  (hx1x2 : x1 + x2 > 0)
  (hx2x3 : x2 + x3 > 0)
  (hx1x3 : x1 + x3 > 0) :
  f x1 + f x2 + f x3 > 0 :=
sorry

end sum_f_positive_l2134_213415


namespace draw_at_least_one_red_card_l2134_213447

-- Define the deck and properties
def total_cards := 52
def red_cards := 26
def black_cards := 26

-- Define the calculation for drawing three cards sequentially
def total_ways_draw3 := total_cards * (total_cards - 1) * (total_cards - 2)
def black_only_ways_draw3 := black_cards * (black_cards - 1) * (black_cards - 2)

-- Define the main proof statement
theorem draw_at_least_one_red_card : 
    total_ways_draw3 - black_only_ways_draw3 = 117000 := by
    -- Proof is omitted
    sorry

end draw_at_least_one_red_card_l2134_213447


namespace jill_spent_10_percent_on_food_l2134_213423

theorem jill_spent_10_percent_on_food 
  (T : ℝ)                         
  (h1 : 0.60 * T = 0.60 * T)    -- 60% on clothing
  (h2 : 0.30 * T = 0.30 * T)    -- 30% on other items
  (h3 : 0.04 * (0.60 * T) = 0.024 * T)  -- 4% tax on clothing
  (h4 : 0.08 * (0.30 * T) = 0.024 * T)  -- 8% tax on other items
  (h5 : 0.048 * T = (0.024 * T + 0.024 * T)) -- total tax is 4.8%
  : 0.10 * T = (T - (0.60*T + 0.30*T)) :=
by
  -- Proof is omitted
  sorry

end jill_spent_10_percent_on_food_l2134_213423


namespace lcm_of_54_and_198_l2134_213426

theorem lcm_of_54_and_198 : Nat.lcm 54 198 = 594 :=
by
  have fact1 : 54 = 2 ^ 1 * 3 ^ 3 := by norm_num
  have fact2 : 198 = 2 ^ 1 * 3 ^ 2 * 11 ^ 1 := by norm_num
  have lcm_prime : Nat.lcm 54 198 = 594 := by
    sorry -- Proof skipped
  exact lcm_prime

end lcm_of_54_and_198_l2134_213426


namespace greatest_integer_function_of_pi_plus_3_l2134_213483

noncomputable def pi_plus_3 : Real := Real.pi + 3

theorem greatest_integer_function_of_pi_plus_3 : Int.floor pi_plus_3 = 6 := 
by
  -- sorry is used to skip the proof
  sorry

end greatest_integer_function_of_pi_plus_3_l2134_213483


namespace smallest_k_l2134_213494

theorem smallest_k (M : Finset ℕ) (H : ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d)) :
  ∃ k, k = 7 ∧ ∀ (M' : Finset ℕ), M'.card = k → ∀ (a b c d : ℕ), a ∈ M' → b ∈ M' → c ∈ M' → d ∈ M' → a ≠ b → b ≠ c → c ≠ d → d ≠ a → 20 ∣ (a - b + c - d) :=
sorry

end smallest_k_l2134_213494


namespace files_deleted_is_3_l2134_213456

-- Define the initial number of files
def initial_files : Nat := 24

-- Define the remaining number of files
def remaining_files : Nat := 21

-- Define the number of files deleted
def files_deleted : Nat := initial_files - remaining_files

-- Prove that the number of files deleted is 3
theorem files_deleted_is_3 : files_deleted = 3 :=
by
  sorry

end files_deleted_is_3_l2134_213456


namespace smallest_triangle_perimeter_l2134_213408

theorem smallest_triangle_perimeter : 
  ∀ (a b c : ℕ), 
    (2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c) ∧ (a = b - 2 ∨ a = b + 2) ∧ (b = c - 2 ∨ b = c + 2) ∧ (a + b > c) ∧ (a + c > b) ∧ (b + c > a) 
    → a + b + c = 12 := 
  sorry

end smallest_triangle_perimeter_l2134_213408


namespace solve_for_y_l2134_213435

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y_l2134_213435


namespace general_term_sequence_l2134_213432

-- Definition of the sequence conditions
def seq (n : ℕ) : ℤ :=
  (-1)^(n+1) * (2*n + 1)

-- The main statement to be proved
theorem general_term_sequence (n : ℕ) : seq n = (-1)^(n+1) * (2 * n + 1) :=
sorry

end general_term_sequence_l2134_213432


namespace zachary_pushups_l2134_213404

theorem zachary_pushups (C P : ℕ) (h1 : C = 14) (h2 : P + C = 67) : P = 53 :=
by
  rw [h1] at h2
  linarith

end zachary_pushups_l2134_213404


namespace find_vertex_angle_of_cone_l2134_213460

noncomputable def vertexAngleCone (r1 r2 : ℝ) (O1 O2 : ℝ) (touching : Prop) (Ctable : Prop) (equalAngles : Prop) : Prop :=
  -- The given conditions:
  -- r1, r2 are the radii of the spheres, where r1 = 4 and r2 = 1.
  -- O1, O2 are the centers of the spheres.
  -- touching indicates the spheres touch externally.
  -- Ctable indicates that vertex C of the cone is on the segment connecting the points where the spheres touch the table.
  -- equalAngles indicates that the rays CO1 and CO2 form equal angles with the table.
  touching → 
  Ctable → 
  equalAngles →
  -- The target to prove:
  ∃ α : ℝ, 2 * α = 2 * Real.arctan (2 / 5)

theorem find_vertex_angle_of_cone (r1 r2 : ℝ) (O1 O2 : ℝ) :
  let touching : Prop := (r1 = 4 ∧ r2 = 1 ∧ abs (O1 - O2) = r1 + r2)
  let Ctable : Prop := (True)  -- Provided by problem conditions, details can be expanded
  let equalAngles : Prop := (True)  
  vertexAngleCone r1 r2 O1 O2 touching Ctable equalAngles := 
by
  sorry

end find_vertex_angle_of_cone_l2134_213460


namespace last_three_digits_of_2_pow_6000_l2134_213448

theorem last_three_digits_of_2_pow_6000 (h : 2^200 ≡ 1 [MOD 800]) : (2^6000 ≡ 1 [MOD 800]) :=
sorry

end last_three_digits_of_2_pow_6000_l2134_213448


namespace problem1_problem2_problem3_problem4_l2134_213492

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l2134_213492


namespace range_of_a_l2134_213468

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 2| + |x + 2| ≤ a^2 - 3 * a) ↔ (a ≥ 4 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l2134_213468


namespace cakes_given_away_l2134_213473

theorem cakes_given_away 
  (cakes_baked : ℕ) 
  (candles_per_cake : ℕ) 
  (total_candles : ℕ) 
  (cakes_given : ℕ) 
  (cakes_left : ℕ) 
  (h1 : cakes_baked = 8) 
  (h2 : candles_per_cake = 6) 
  (h3 : total_candles = 36) 
  (h4 : total_candles = candles_per_cake * cakes_left) 
  (h5 : cakes_given = cakes_baked - cakes_left) 
  : cakes_given = 2 :=
sorry

end cakes_given_away_l2134_213473


namespace max_peaceful_clients_kept_l2134_213459

-- Defining the types for knights, liars, and troublemakers
def Person : Type := ℕ

noncomputable def isKnight : Person → Prop := sorry
noncomputable def isLiar : Person → Prop := sorry
noncomputable def isTroublemaker : Person → Prop := sorry

-- Total number of people in the bar
def totalPeople : ℕ := 30

-- Number of knights, liars, and troublemakers
def numberKnights : ℕ := 10
def numberLiars : ℕ := 10
def numberTroublemakers : ℕ := 10

-- The bartender's goal: get rid of all troublemakers and keep as many peaceful clients as possible
def maxPeacefulClients (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ): ℕ :=
  total - troublemakers

-- Statement to be proved
theorem max_peaceful_clients_kept (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ)
  (h_total : total = 30)
  (h_knights : knights = 10)
  (h_liars : liars = 10)
  (h_troublemakers : troublemakers = 10) :
  maxPeacefulClients total knights liars troublemakers = 19 :=
by
  -- Proof steps go here
  sorry

end max_peaceful_clients_kept_l2134_213459


namespace jersey_to_shoes_ratio_l2134_213465

theorem jersey_to_shoes_ratio
  (pairs_shoes: ℕ) (jerseys: ℕ) (total_cost: ℝ) (total_cost_shoes: ℝ) 
  (shoes: pairs_shoes = 6) (jer: jerseys = 4) (total: total_cost = 560) (cost_sh: total_cost_shoes = 480) :
  ((total_cost - total_cost_shoes) / jerseys) / (total_cost_shoes / pairs_shoes) = 1 / 4 := 
by 
  sorry

end jersey_to_shoes_ratio_l2134_213465


namespace average_production_l2134_213451

theorem average_production (n : ℕ) (P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 62) : n = 14 :=
  sorry

end average_production_l2134_213451


namespace max_sum_cubes_l2134_213406

theorem max_sum_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 ≤ 8 :=
sorry

end max_sum_cubes_l2134_213406


namespace power_function_evaluation_l2134_213411

theorem power_function_evaluation (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 4 = 2) : f 16 = 4 :=
by
  sorry

end power_function_evaluation_l2134_213411


namespace perfect_square_of_seq_l2134_213479

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n ≥ 3, a n = 7 * a (n - 1) - a (n - 2)

theorem perfect_square_of_seq (a : ℕ → ℤ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  ∃ k : ℤ, k * k = a n + 2 + a (n + 1) :=
sorry

end perfect_square_of_seq_l2134_213479


namespace students_from_other_communities_l2134_213444

noncomputable def percentageMuslims : ℝ := 0.41
noncomputable def percentageHindus : ℝ := 0.32
noncomputable def percentageSikhs : ℝ := 0.12
noncomputable def totalStudents : ℝ := 1520

theorem students_from_other_communities : 
  totalStudents * (1 - (percentageMuslims + percentageHindus + percentageSikhs)) = 228 := 
by 
  sorry

end students_from_other_communities_l2134_213444


namespace problem_proof_l2134_213490

variable (a b c : ℝ)

-- Given conditions
def conditions (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c) ∧ ((a + 1) * (b + 1) * (c + 1) = 8)

-- The proof problem
theorem problem_proof (h : conditions a b c) : a + b + c ≥ 3 ∧ a * b * c ≤ 1 :=
  sorry

end problem_proof_l2134_213490


namespace find_k_and_f_min_total_cost_l2134_213464

-- Define the conditions
def construction_cost (x : ℝ) : ℝ := 60 * x
def energy_consumption_cost (x : ℝ) : ℝ := 40 - 4 * x
def total_cost (x : ℝ) : ℝ := construction_cost x + 20 * energy_consumption_cost x

theorem find_k_and_f :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → energy_consumption_cost 0 = 8 → energy_consumption_cost x = 40 - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 10 → total_cost x = 800 - 74 * x) :=
by
  sorry

theorem min_total_cost :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → 800 - 74 * x ≥ 70) ∧
  total_cost 5 = 70 :=
by
  sorry

end find_k_and_f_min_total_cost_l2134_213464


namespace clubsuit_subtraction_l2134_213496

def clubsuit (x y : ℕ) := 4 * x + 6 * y

theorem clubsuit_subtraction :
  (clubsuit 5 3) - (clubsuit 1 4) = 10 :=
by
  sorry

end clubsuit_subtraction_l2134_213496


namespace sausage_thickness_correct_l2134_213463

noncomputable def earth_radius := 6000 -- in km
noncomputable def distance_to_sun := 150000000 -- in km
noncomputable def sausage_thickness := 44 -- in km

theorem sausage_thickness_correct :
  let R := earth_radius
  let L := distance_to_sun
  let r := Real.sqrt ((4 * R^3) / (3 * L))
  abs (r - sausage_thickness) < 10 * sausage_thickness :=
by
  sorry

end sausage_thickness_correct_l2134_213463


namespace sum_abs_coeffs_l2134_213480

theorem sum_abs_coeffs (a : ℝ → ℝ) :
  (∀ x, (1 - 3 * x)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| = 4^9 := by
  sorry

end sum_abs_coeffs_l2134_213480


namespace repeating_decimal_sum_correct_l2134_213418

noncomputable def repeating_decimal_sum : ℚ :=
  let x := 2 / 3
  let y := 2 / 9
  let z := 4 / 9
  x + y - z

theorem repeating_decimal_sum_correct :
  repeating_decimal_sum = 4 / 9 :=
by
  sorry

end repeating_decimal_sum_correct_l2134_213418


namespace sum_of_remainders_l2134_213407

theorem sum_of_remainders (n : ℤ) (h : n % 12 = 5) :
  (n % 4) + (n % 3) = 3 :=
by
  sorry

end sum_of_remainders_l2134_213407


namespace number_of_girls_in_group_l2134_213430

-- Define the given conditions
def total_students : ℕ := 20
def prob_of_selecting_girl : ℚ := 2/5

-- State the lean problem for the proof
theorem number_of_girls_in_group : (total_students : ℚ) * prob_of_selecting_girl = 8 := by
  sorry

end number_of_girls_in_group_l2134_213430


namespace probability_floor_sqrt_even_l2134_213462

/-- Suppose x and y are chosen randomly and uniformly from (0,1). The probability that
    ⌊√(x/y)⌋ is even is 1 - π²/24. -/
theorem probability_floor_sqrt_even (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (1 - Real.pi ^ 2 / 24) = sorry :=
sorry

end probability_floor_sqrt_even_l2134_213462


namespace highest_throw_is_37_feet_l2134_213486

theorem highest_throw_is_37_feet :
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  max (max C1 (max C2 C3)) (max J1 (max J2 J3)) = 37 := by
  let C1 := 20
  let J1 := C1 - 4
  let C2 := C1 + 10
  let J2 := J1 * 2
  let C3 := C2 + 4
  let J3 := C1 + 17
  sorry

end highest_throw_is_37_feet_l2134_213486


namespace fraction_pow_rule_l2134_213452

theorem fraction_pow_rule :
  (5 / 7)^4 = 625 / 2401 :=
by
  sorry

end fraction_pow_rule_l2134_213452


namespace range_of_k_l2134_213438

theorem range_of_k (k : Real) : 
  (∀ (x y : Real), x^2 + y^2 - 12 * x - 4 * y + 37 = 0)
  → ((k < -Real.sqrt 2) ∨ (k > Real.sqrt 2)) :=
by
  sorry

end range_of_k_l2134_213438


namespace quadratic_solution_exists_l2134_213433

theorem quadratic_solution_exists (a b : ℝ) : ∃ (x : ℝ), (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 :=
by
  sorry

end quadratic_solution_exists_l2134_213433


namespace speed_limit_l2134_213498

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end speed_limit_l2134_213498


namespace increasing_ω_l2134_213425

noncomputable def f (ω x : ℝ) : ℝ := (1 / 2) * (Real.sin ((ω * x) / 2)) * (Real.cos ((ω * x) / 2))

theorem increasing_ω (ω : ℝ) (hω : 0 < ω) :
  (∀ x y, - (Real.pi / 3) ≤ x → x ≤ y → y ≤ (Real.pi / 4) → f ω x ≤ f ω y)
  ↔ 0 < ω ∧ ω ≤ (3 / 2) :=
sorry

end increasing_ω_l2134_213425


namespace smallest_x_division_remainder_l2134_213427

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l2134_213427


namespace work_done_by_force_l2134_213485

def F (x : ℝ) := 4 * x - 1

theorem work_done_by_force :
  let a := 1
  let b := 3
  (∫ x in a..b, F x) = 14 := by
  sorry

end work_done_by_force_l2134_213485


namespace product_of_numbers_l2134_213410

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := 
sorry

end product_of_numbers_l2134_213410


namespace sally_students_are_30_l2134_213491

-- Define the conditions given in the problem
def school_money : ℕ := 320
def book_cost : ℕ := 12
def sally_money : ℕ := 40
def total_students : ℕ := 30

-- Define the total amount Sally can spend on books
def total_amount_available : ℕ := school_money + sally_money

-- The total cost of books for S students
def total_cost (S : ℕ) : ℕ := book_cost * S

-- The main theorem stating that S students will cost the same as the amount Sally can spend
theorem sally_students_are_30 : total_cost 30 = total_amount_available :=
by
  sorry

end sally_students_are_30_l2134_213491


namespace data_transmission_time_l2134_213446

def chunks_per_block : ℕ := 1024
def blocks : ℕ := 30
def transmission_rate : ℕ := 256
def seconds_in_minute : ℕ := 60

theorem data_transmission_time :
  (blocks * chunks_per_block) / transmission_rate / seconds_in_minute = 2 :=
by
  sorry

end data_transmission_time_l2134_213446


namespace remainder_of_sum_is_zero_l2134_213400

-- Define the properties of m and n according to the conditions of the problem
def m : ℕ := 2 * 1004 ^ 2
def n : ℕ := 2007 * 1003

-- State the theorem that proves the remainder of (m + n) divided by 1004 is 0
theorem remainder_of_sum_is_zero : (m + n) % 1004 = 0 := by
  sorry

end remainder_of_sum_is_zero_l2134_213400


namespace part1a_part1b_part2_part3_l2134_213475

-- Definitions for the sequences in columns ①, ②, and ③
def col1 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1)
def col2 (n : ℕ) : ℤ := ((-1 : ℤ) ^ n * (2 * n - 1)) - 2
def col3 (n : ℕ) : ℤ := (-1 : ℤ) ^ n * (2 * n - 1) * 3

-- Problem statements
theorem part1a : col1 10 = 19 :=
sorry

theorem part1b : col2 15 = -31 :=
sorry

theorem part2 : ¬ ∃ n : ℕ, col2 (n - 1) + col2 n + col2 (n + 1) = 1001 :=
sorry

theorem part3 : ∃ k : ℕ, col1 k + col2 k + col3 k = 599 ∧ k = 301 :=
sorry

end part1a_part1b_part2_part3_l2134_213475


namespace function_monotone_increasing_l2134_213409

open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - log x

theorem function_monotone_increasing : ∀ x, 1 ≤ x → (0 < x) → (1 / 2) * x^2 - log x = f x → (∀ y, 1 ≤ y → (0 < y) → (f y ≤ f x)) :=
sorry

end function_monotone_increasing_l2134_213409


namespace simplify_evaluate_expression_l2134_213414

theorem simplify_evaluate_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (2 / (x + 1) + 1 / (x - 2)) / (x - 1) / (x - 2) = Real.sqrt 3 := by
  sorry

end simplify_evaluate_expression_l2134_213414
