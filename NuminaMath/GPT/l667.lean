import Mathlib

namespace NUMINAMATH_GPT_positive_solution_unique_m_l667_66775

theorem positive_solution_unique_m (m : ℝ) : ¬ (4 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_positive_solution_unique_m_l667_66775


namespace NUMINAMATH_GPT_trevor_spending_proof_l667_66751

def trevor_spends (T R Q : ℕ) : Prop :=
  T = R + 20 ∧ R = 2 * Q ∧ 4 * T + 4 * R + 2 * Q = 680

theorem trevor_spending_proof (T R Q : ℕ) (h : trevor_spends T R Q) : T = 80 :=
by sorry

end NUMINAMATH_GPT_trevor_spending_proof_l667_66751


namespace NUMINAMATH_GPT_number_of_tins_per_day_for_rest_of_week_l667_66752
-- Import necessary library

-- Define conditions as Lean definitions
def d1 : ℕ := 50
def d2 : ℕ := 3 * d1
def d3 : ℕ := d2 - 50
def total_target : ℕ := 500

-- Define what we need to prove
theorem number_of_tins_per_day_for_rest_of_week :
  ∃ (dr : ℕ), d1 + d2 + d3 + 4 * dr = total_target ∧ dr = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_tins_per_day_for_rest_of_week_l667_66752


namespace NUMINAMATH_GPT_cos_arcsin_l667_66773

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_arcsin_l667_66773


namespace NUMINAMATH_GPT_selling_price_of_book_l667_66716

theorem selling_price_of_book (cost_price : ℕ) (profit_rate : ℕ) (profit : ℕ) (selling_price : ℕ) :
  cost_price = 50 → profit_rate = 80 → profit = (profit_rate * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 90 :=
by
  intros h_cost_price h_profit_rate h_profit h_selling_price
  rw [h_cost_price, h_profit_rate] at h_profit
  simp at h_profit
  rw [h_cost_price, h_profit] at h_selling_price
  exact h_selling_price

end NUMINAMATH_GPT_selling_price_of_book_l667_66716


namespace NUMINAMATH_GPT_form_regular_octagon_l667_66788

def concentric_squares_form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) : Prop :=
  ∀ (p : ℂ), ∃ (h₃ : ∀ (pvertices : ℤ → ℂ), -- vertices of the smaller square
                ∀ (lperpendiculars : ℤ → ℂ), -- perpendicular line segments
                true), -- additional conditions representing the perpendicular lines construction
    -- proving that the formed shape is a regular octagon:
    true -- Placeholder for actual condition/check for regular octagon

theorem form_regular_octagon (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^2 / b^2 = 2) :
  concentric_squares_form_regular_octagon a b h₀ h₁ h₂ :=
by sorry

end NUMINAMATH_GPT_form_regular_octagon_l667_66788


namespace NUMINAMATH_GPT_minute_hour_hands_opposite_l667_66776

theorem minute_hour_hands_opposite (x : ℝ) (h1 : 10 * 60 ≤ x) (h2 : x ≤ 11 * 60) : 
  (5.5 * x = 442.5) :=
sorry

end NUMINAMATH_GPT_minute_hour_hands_opposite_l667_66776


namespace NUMINAMATH_GPT_evaluation_at_x_4_l667_66724

noncomputable def simplified_expression (x : ℝ) :=
  (x - 1 - (3 / (x + 1))) / ((x^2 + 2 * x) / (x + 1))

theorem evaluation_at_x_4 : simplified_expression 4 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluation_at_x_4_l667_66724


namespace NUMINAMATH_GPT_total_wings_count_l667_66784

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) (h_planes : num_planes = 54) (h_wings : wings_per_plane = 2) : num_planes * wings_per_plane = 108 :=
by 
  sorry

end NUMINAMATH_GPT_total_wings_count_l667_66784


namespace NUMINAMATH_GPT_approximate_roots_l667_66783

noncomputable def f (x : ℝ) : ℝ := 0.3 * x^3 - 2 * x^2 - 0.2 * x + 0.5

theorem approximate_roots : 
  ∃ x₁ x₂ x₃ : ℝ, 
    (f x₁ = 0 ∧ |x₁ + 0.4| < 0.1) ∧ 
    (f x₂ = 0 ∧ |x₂ - 0.5| < 0.1) ∧ 
    (f x₃ = 0 ∧ |x₃ - 2.6| < 0.1) :=
by
  sorry

end NUMINAMATH_GPT_approximate_roots_l667_66783


namespace NUMINAMATH_GPT_integral_fx_l667_66747

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_fx :
  ∫ x in -Real.pi..0, f x = -2 - (1/2) * Real.pi ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_integral_fx_l667_66747


namespace NUMINAMATH_GPT_complex_number_solution_l667_66738

open Complex

theorem complex_number_solution (z : ℂ) (h : z^2 = -99 - 40 * I) : z = 2 - 10 * I ∨ z = -2 + 10 * I :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l667_66738


namespace NUMINAMATH_GPT_find_b_l667_66721

theorem find_b (b p : ℚ) :
  (∀ x : ℚ, (2 * x^3 + b * x + 7 = (x^2 + p * x + 1) * (2 * x + 7))) →
  b = -45 / 2 :=
sorry

end NUMINAMATH_GPT_find_b_l667_66721


namespace NUMINAMATH_GPT_treasures_first_level_is_4_l667_66761

-- Definitions based on conditions
def points_per_treasure : ℕ := 5
def treasures_second_level : ℕ := 3
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := 35
def points_first_level : ℕ := total_score - score_second_level

-- Main statement to prove
theorem treasures_first_level_is_4 : points_first_level / points_per_treasure = 4 := 
by
  -- We are skipping the proof here and using sorry.
  sorry

end NUMINAMATH_GPT_treasures_first_level_is_4_l667_66761


namespace NUMINAMATH_GPT_find_PO_l667_66799

variables {P : ℝ × ℝ} {O F : ℝ × ℝ}

def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def isosceles_triangle (O P F : ℝ × ℝ) : Prop :=
  dist O P = dist O F ∨ dist O P = dist P F

theorem find_PO
  (P : ℝ × ℝ) (O : ℝ × ℝ) (F : ℝ × ℝ)
  (hO : origin O) (hF : focus F) (hP : on_parabola P) (h_iso : isosceles_triangle O P F) :
  dist O P = 1 ∨ dist O P = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_PO_l667_66799


namespace NUMINAMATH_GPT_problem1_eval_problem2_eval_l667_66701

-- Problem 1
theorem problem1_eval :
  (1 : ℚ) * (-4.5) - (-5.6667) - (2.5) - 7.6667 = -9 := 
by
  sorry

-- Problem 2
theorem problem2_eval :
  (-(4^2) / (-2)^3) - ((4 / 9) * ((-3 / 2)^2)) = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem1_eval_problem2_eval_l667_66701


namespace NUMINAMATH_GPT_gauss_polynomial_reciprocal_l667_66778

def gauss_polynomial (k l : ℤ) (x : ℝ) : ℝ := sorry -- Placeholder for actual polynomial definition

theorem gauss_polynomial_reciprocal (k l : ℤ) (x : ℝ) : 
  x^(k * l) * gauss_polynomial k l (1 / x) = gauss_polynomial k l x :=
sorry

end NUMINAMATH_GPT_gauss_polynomial_reciprocal_l667_66778


namespace NUMINAMATH_GPT_max_participants_win_at_least_three_matches_l667_66774

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end NUMINAMATH_GPT_max_participants_win_at_least_three_matches_l667_66774


namespace NUMINAMATH_GPT_degenerate_ellipse_single_point_c_l667_66764

theorem degenerate_ellipse_single_point_c (c : ℝ) :
  (∀ x y : ℝ, 2 * x^2 + y^2 + 8 * x - 10 * y + c = 0 → x = -2 ∧ y = 5) →
  c = 33 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_degenerate_ellipse_single_point_c_l667_66764


namespace NUMINAMATH_GPT_joan_gave_melanie_apples_l667_66703

theorem joan_gave_melanie_apples (original_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : original_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  sorry

end NUMINAMATH_GPT_joan_gave_melanie_apples_l667_66703


namespace NUMINAMATH_GPT_solve_inequality_find_m_range_l667_66711

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem solve_inequality (a : ℝ) : 
  ∀ x : ℝ, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨ 
    (a > 1) ∨ 
    (a < 1 ∧ (x > 3 - a ∨ x < a + 1)) :=
sorry

theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, f x > g x m) ↔ m < 5 :=
sorry

end NUMINAMATH_GPT_solve_inequality_find_m_range_l667_66711


namespace NUMINAMATH_GPT_range_of_m_l667_66736

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l667_66736


namespace NUMINAMATH_GPT_guppies_eaten_by_moray_eel_l667_66706

-- Definitions based on conditions
def moray_eel_guppies_per_day : ℕ := sorry -- Number of guppies the moray eel eats per day

def number_of_betta_fish : ℕ := 5

def guppies_per_betta : ℕ := 7

def total_guppies_needed_per_day : ℕ := 55

-- Theorem based on the question
theorem guppies_eaten_by_moray_eel :
  moray_eel_guppies_per_day = total_guppies_needed_per_day - (number_of_betta_fish * guppies_per_betta) :=
sorry

end NUMINAMATH_GPT_guppies_eaten_by_moray_eel_l667_66706


namespace NUMINAMATH_GPT_sum_abc_l667_66743

theorem sum_abc (a b c : ℝ) 
  (h : (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0) : 
  a + b + c = 11 := 
by 
  sorry

end NUMINAMATH_GPT_sum_abc_l667_66743


namespace NUMINAMATH_GPT_exists_special_number_l667_66777

theorem exists_special_number :
  ∃ N : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ 149 → k ∣ N) ∨ (k + 1 ∣ N) = false) :=
sorry

end NUMINAMATH_GPT_exists_special_number_l667_66777


namespace NUMINAMATH_GPT_g_neg6_eq_neg28_l667_66757

-- Define the given function g
def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

-- State the main theorem to prove g(-6) = -28 under the given conditions
theorem g_neg6_eq_neg28 (h1 : g 6 = 12) : g (-6) = -28 :=
by
  sorry

end NUMINAMATH_GPT_g_neg6_eq_neg28_l667_66757


namespace NUMINAMATH_GPT_range_of_k_for_real_roots_l667_66749

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_k_for_real_roots_l667_66749


namespace NUMINAMATH_GPT_license_plate_increase_l667_66770

-- definitions from conditions
def old_plates_count : ℕ := 26 ^ 2 * 10 ^ 3
def new_plates_count : ℕ := 26 ^ 4 * 10 ^ 2

-- theorem stating the increase in the number of license plates
theorem license_plate_increase : 
  (new_plates_count : ℚ) / (old_plates_count : ℚ) = 26 ^ 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_increase_l667_66770


namespace NUMINAMATH_GPT_distance_from_negative_two_is_three_l667_66772

theorem distance_from_negative_two_is_three (x : ℝ) : abs (x + 2) = 3 → (x = -5) ∨ (x = 1) :=
  sorry

end NUMINAMATH_GPT_distance_from_negative_two_is_three_l667_66772


namespace NUMINAMATH_GPT_evaluate_exponent_l667_66760

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end NUMINAMATH_GPT_evaluate_exponent_l667_66760


namespace NUMINAMATH_GPT_sandro_children_ratio_l667_66750

theorem sandro_children_ratio (d : ℕ) (h1 : d + 3 = 21) : d / 3 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sandro_children_ratio_l667_66750


namespace NUMINAMATH_GPT_find_f_5_l667_66713

def f (x : ℝ) : ℝ := sorry -- we need to create a function under our condition

theorem find_f_5 : f 5 = 0 :=
sorry

end NUMINAMATH_GPT_find_f_5_l667_66713


namespace NUMINAMATH_GPT_end_of_month_books_count_l667_66700

theorem end_of_month_books_count:
  ∀ (initial_books : ℝ) (loaned_out_books : ℝ) (return_rate : ℝ)
    (rounded_loaned_out_books : ℝ) (returned_books : ℝ)
    (not_returned_books : ℝ) (end_of_month_books : ℝ),
    initial_books = 75 →
    loaned_out_books = 60.00000000000001 →
    return_rate = 0.65 →
    rounded_loaned_out_books = 60 →
    returned_books = return_rate * rounded_loaned_out_books →
    not_returned_books = rounded_loaned_out_books - returned_books →
    end_of_month_books = initial_books - not_returned_books →
    end_of_month_books = 54 :=
by
  intros initial_books loaned_out_books return_rate
         rounded_loaned_out_books returned_books
         not_returned_books end_of_month_books
  intros h_initial_books h_loaned_out_books h_return_rate
         h_rounded_loaned_out_books h_returned_books
         h_not_returned_books h_end_of_month_books
  sorry

end NUMINAMATH_GPT_end_of_month_books_count_l667_66700


namespace NUMINAMATH_GPT_car_speed_in_kmh_l667_66765

theorem car_speed_in_kmh (rev_per_min : ℕ) (circumference : ℕ) (speed : ℕ) 
  (h1 : rev_per_min = 400) (h2 : circumference = 4) : speed = 96 :=
  sorry

end NUMINAMATH_GPT_car_speed_in_kmh_l667_66765


namespace NUMINAMATH_GPT_rational_sum_of_squares_is_square_l667_66707

theorem rational_sum_of_squares_is_square (a b c : ℚ) :
  ∃ r : ℚ, r ^ 2 = (1 / (b - c) ^ 2 + 1 / (c - a) ^ 2 + 1 / (a - b) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_rational_sum_of_squares_is_square_l667_66707


namespace NUMINAMATH_GPT_readers_all_three_l667_66731

def total_readers : ℕ := 500
def readers_science_fiction : ℕ := 320
def readers_literary_works : ℕ := 200
def readers_non_fiction : ℕ := 150
def readers_sf_and_lw : ℕ := 120
def readers_sf_and_nf : ℕ := 80
def readers_lw_and_nf : ℕ := 60

theorem readers_all_three :
  total_readers = readers_science_fiction + readers_literary_works + readers_non_fiction - (readers_sf_and_lw + readers_sf_and_nf + readers_lw_and_nf) + 90 :=
by
  sorry

end NUMINAMATH_GPT_readers_all_three_l667_66731


namespace NUMINAMATH_GPT_length_of_AB_l667_66755

noncomputable def AB_CD_sum_240 (AB CD : ℝ) (h : ℝ) : Prop :=
  AB + CD = 240

noncomputable def ratio_of_areas (AB CD : ℝ) : Prop :=
  AB / CD = 5 / 3

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (h_ratio : ratio_of_areas AB CD) (h_sum : AB_CD_sum_240 AB CD h) : AB = 150 :=
by
  unfold ratio_of_areas at h_ratio
  unfold AB_CD_sum_240 at h_sum
  sorry

end NUMINAMATH_GPT_length_of_AB_l667_66755


namespace NUMINAMATH_GPT_ones_digit_9_pow_53_l667_66756

theorem ones_digit_9_pow_53 :
  (9 ^ 53) % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_9_pow_53_l667_66756


namespace NUMINAMATH_GPT_max_figures_in_grid_l667_66705

-- Definition of the grid size
def grid_size : ℕ := 9

-- Definition of the figure coverage
def figure_coverage : ℕ := 4

-- The total number of unit squares in the grid is 9 * 9 = 81
def total_unit_squares : ℕ := grid_size * grid_size

-- Each figure covers exactly 4 unit squares
def units_per_figure : ℕ := figure_coverage

-- The number of such 2x2 blocks that can be formed in 9x9 grid.
def maximal_figures_possible : ℕ := (grid_size / 2) * (grid_size / 2)

-- The main theorem to be proved
theorem max_figures_in_grid : 
  maximal_figures_possible = total_unit_squares / units_per_figure := by
  sorry

end NUMINAMATH_GPT_max_figures_in_grid_l667_66705


namespace NUMINAMATH_GPT_newer_model_distance_l667_66785

theorem newer_model_distance (d_old : ℝ) (p_increase : ℝ) (d_new : ℝ) (h1 : d_old = 300) (h2 : p_increase = 0.30) (h3 : d_new = d_old * (1 + p_increase)) : d_new = 390 :=
by
  sorry

end NUMINAMATH_GPT_newer_model_distance_l667_66785


namespace NUMINAMATH_GPT_find_a1_l667_66728

open Nat

theorem find_a1 (a : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n)
  (h2 : a 3 = 12) : a 1 = 3 :=
sorry

end NUMINAMATH_GPT_find_a1_l667_66728


namespace NUMINAMATH_GPT_exists_two_points_same_color_l667_66734

theorem exists_two_points_same_color :
  ∀ (x : ℝ), ∀ (color : ℝ × ℝ → Prop),
  (∀ (p : ℝ × ℝ), color p = red ∨ color p = blue) →
  (∃ (p1 p2 : ℝ × ℝ), dist p1 p2 = x ∧ color p1 = color p2) :=
by
  intro x color color_prop
  sorry

end NUMINAMATH_GPT_exists_two_points_same_color_l667_66734


namespace NUMINAMATH_GPT_line_tangent_72_l667_66735

theorem line_tangent_72 (k : ℝ) : 4 * x + 6 * y + k = 0 → y^2 = 32 * x → (48^2 - 4 * (8 * k) = 0 ↔ k = 72) :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_72_l667_66735


namespace NUMINAMATH_GPT_candy_pebbles_l667_66791

theorem candy_pebbles (C L : ℕ) 
  (h1 : L = 3 * C)
  (h2 : L = C + 8) :
  C = 4 :=
by
  sorry

end NUMINAMATH_GPT_candy_pebbles_l667_66791


namespace NUMINAMATH_GPT_enrique_shredder_Y_feeds_l667_66732

theorem enrique_shredder_Y_feeds :
  let typeB_contracts := 350
  let pages_per_TypeB := 10
  let shredderY_capacity := 8
  let total_pages_TypeB := typeB_contracts * pages_per_TypeB
  let feeds_ShredderY := (total_pages_TypeB + shredderY_capacity - 1) / shredderY_capacity
  feeds_ShredderY = 438 := sorry

end NUMINAMATH_GPT_enrique_shredder_Y_feeds_l667_66732


namespace NUMINAMATH_GPT_squares_with_center_25_60_l667_66740

theorem squares_with_center_25_60 :
  let center_x := 25
  let center_y := 60
  let non_neg_int_coords (x : ℤ) (y : ℤ) := x ≥ 0 ∧ y ≥ 0
  let is_center (x : ℤ) (y : ℤ) := x = center_x ∧ y = center_y
  let num_squares := 650
  ∃ n : ℤ, (n = num_squares) ∧ ∀ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ), 
    non_neg_int_coords x₁ y₁ ∧ non_neg_int_coords x₂ y₂ ∧ 
    non_neg_int_coords x₃ y₃ ∧ non_neg_int_coords x₄ y₄ ∧ 
    is_center ((x₁ + x₂ + x₃ + x₄) / 4) ((y₁ + y₂ + y₃ + y₄) / 4) → 
    ∃ (k : ℤ), n = 650 :=
sorry

end NUMINAMATH_GPT_squares_with_center_25_60_l667_66740


namespace NUMINAMATH_GPT_factorization_example_l667_66792

theorem factorization_example (a b : ℕ) : (a - 2*b)^2 = a^2 - 4*a*b + 4*b^2 := 
by sorry

end NUMINAMATH_GPT_factorization_example_l667_66792


namespace NUMINAMATH_GPT_y_is_defined_iff_x_not_equal_to_10_l667_66795

def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 10

theorem y_is_defined_iff_x_not_equal_to_10 (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 10)) ↔ range_of_independent_variable x :=
by sorry

end NUMINAMATH_GPT_y_is_defined_iff_x_not_equal_to_10_l667_66795


namespace NUMINAMATH_GPT_total_amount_returned_l667_66722

noncomputable def continuous_compounding_interest : ℝ :=
  let P : ℝ := 325 / (Real.exp 0.12 - 1)
  let A1 : ℝ := P * Real.exp 0.04
  let A2 : ℝ := A1 * Real.exp 0.05
  let A3 : ℝ := A2 * Real.exp 0.03
  let total_interest : ℝ := 325
  let total_amount : ℝ := P + total_interest
  total_amount

theorem total_amount_returned :
  continuous_compounding_interest = 2874.02 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_returned_l667_66722


namespace NUMINAMATH_GPT_movie_ticket_cost_l667_66789

-- Definitions from conditions
def total_spending : ℝ := 36
def combo_meal_cost : ℝ := 11
def candy_cost : ℝ := 2.5
def total_food_cost : ℝ := combo_meal_cost + 2 * candy_cost
def total_ticket_cost (x : ℝ) : ℝ := 2 * x

-- The theorem stating the proof problem
theorem movie_ticket_cost :
  ∃ (x : ℝ), total_ticket_cost x + total_food_cost = total_spending ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_movie_ticket_cost_l667_66789


namespace NUMINAMATH_GPT_range_of_a_for_three_distinct_real_roots_l667_66719

theorem range_of_a_for_three_distinct_real_roots (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 - 3*x^2 - a ∧ ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_three_distinct_real_roots_l667_66719


namespace NUMINAMATH_GPT_actual_diameter_layer_3_is_20_micrometers_l667_66710

noncomputable def magnified_diameter_to_actual (magnified_diameter_cm : ℕ) (magnification_factor : ℕ) : ℕ :=
  (magnified_diameter_cm * 10000) / magnification_factor

def layer_3_magnified_diameter_cm : ℕ := 3
def layer_3_magnification_factor : ℕ := 1500

theorem actual_diameter_layer_3_is_20_micrometers :
  magnified_diameter_to_actual layer_3_magnified_diameter_cm layer_3_magnification_factor = 20 :=
by
  sorry

end NUMINAMATH_GPT_actual_diameter_layer_3_is_20_micrometers_l667_66710


namespace NUMINAMATH_GPT_smallest_multiple_of_7_greater_than_500_l667_66708

theorem smallest_multiple_of_7_greater_than_500 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n > 500 ∧ n = 504 := 
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_7_greater_than_500_l667_66708


namespace NUMINAMATH_GPT_find_constant_a_l667_66782

theorem find_constant_a :
  (∃ (a : ℝ), a > 0 ∧ (a + 2 * a + 3 * a + 4 * a = 1)) →
  ∃ (a : ℝ), a = 1 / 10 :=
sorry

end NUMINAMATH_GPT_find_constant_a_l667_66782


namespace NUMINAMATH_GPT_first_team_engineers_l667_66797

theorem first_team_engineers (E : ℕ) 
  (teamQ_engineers : ℕ := 16) 
  (work_days_teamQ : ℕ := 30) 
  (work_days_first_team : ℕ := 32) 
  (working_capacity_ratio : ℚ := 3 / 2) :
  E * work_days_first_team * 3 = teamQ_engineers * work_days_teamQ * 2 → 
  E = 10 :=
by
  sorry

end NUMINAMATH_GPT_first_team_engineers_l667_66797


namespace NUMINAMATH_GPT_product_of_real_roots_l667_66748

theorem product_of_real_roots (x : ℝ) (h : x^5 = 100) : x = 10^(2/5) := by
  sorry

end NUMINAMATH_GPT_product_of_real_roots_l667_66748


namespace NUMINAMATH_GPT_problem_1_solution_set_problem_2_minimum_value_a_l667_66766

-- Define the function f with given a value
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - a * |x - 1|

-- Problem 1: Prove the solution set for f(x) > 5 when a = -2 is {x | x < -4/3 ∨ x > 2}
theorem problem_1_solution_set (x : ℝ) : f x (-2) > 5 ↔ x < -4 / 3 ∨ x > 2 :=
by
  sorry

-- Problem 2: Prove the minimum value of a ensures f(x) ≤ a * |x + 3| is 1/2
theorem problem_2_minimum_value_a : (∀ x : ℝ, f x a ≤ a * |x + 3| ∨ a ≥ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_solution_set_problem_2_minimum_value_a_l667_66766


namespace NUMINAMATH_GPT_find_f_neg_two_l667_66733

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - 1 else sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)

axiom f_odd : is_odd_function f
axiom f_pos : ∀ x, x > 0 → f x = x^2 - 1

theorem find_f_neg_two : f (-2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_two_l667_66733


namespace NUMINAMATH_GPT_books_at_end_of_year_l667_66754

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end NUMINAMATH_GPT_books_at_end_of_year_l667_66754


namespace NUMINAMATH_GPT_bob_spends_more_time_l667_66767

def pages := 760
def time_per_page_bob := 45
def time_per_page_chandra := 30
def total_time_bob := pages * time_per_page_bob
def total_time_chandra := pages * time_per_page_chandra
def time_difference := total_time_bob - total_time_chandra

theorem bob_spends_more_time : time_difference = 11400 :=
by
  sorry

end NUMINAMATH_GPT_bob_spends_more_time_l667_66767


namespace NUMINAMATH_GPT_interval_of_x_l667_66779

theorem interval_of_x (x : ℝ) : (4 * x > 2) ∧ (4 * x < 5) ∧ (5 * x > 2) ∧ (5 * x < 5) ↔ (x > 1/2) ∧ (x < 1) := 
by 
  sorry

end NUMINAMATH_GPT_interval_of_x_l667_66779


namespace NUMINAMATH_GPT_possible_values_on_Saras_card_l667_66742

theorem possible_values_on_Saras_card :
  ∀ (y : ℝ), (0 < y ∧ y < π / 2) →
  let sin_y := Real.sin y
  let cos_y := Real.cos y
  let tan_y := Real.tan y
  (∃ (s l k : ℝ), s = sin_y ∧ l = cos_y ∧ k = tan_y ∧
  (s = l ∨ s = k ∨ l = k) ∧ (s = l ∧ l ≠ k) ∧ s = l ∧ s = 1) :=
sorry

end NUMINAMATH_GPT_possible_values_on_Saras_card_l667_66742


namespace NUMINAMATH_GPT_area_of_region_l667_66781

theorem area_of_region : 
  (∃ x y : ℝ, (x + 5)^2 + (y - 3)^2 = 32) → (π * 32 = 32 * π) :=
by 
  sorry

end NUMINAMATH_GPT_area_of_region_l667_66781


namespace NUMINAMATH_GPT_algebraic_expression_1_algebraic_expression_2_l667_66744

-- Problem 1
theorem algebraic_expression_1 (a : ℚ) (h : a = 4 / 5) : -24.7 * a + 1.3 * a - (33 / 5) * a = -24 := 
by 
  sorry

-- Problem 2
theorem algebraic_expression_2 (a b : ℕ) (ha : a = 899) (hb : b = 101) : a^2 + 2 * a * b + b^2 = 1000000 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_1_algebraic_expression_2_l667_66744


namespace NUMINAMATH_GPT_n_is_900_l667_66780

theorem n_is_900 
  (m n : ℕ) 
  (h1 : ∃ x y : ℤ, m = x^2 ∧ n = y^2) 
  (h2 : Prime (m - n)) : n = 900 := 
sorry

end NUMINAMATH_GPT_n_is_900_l667_66780


namespace NUMINAMATH_GPT_price_of_wheat_flour_l667_66762

theorem price_of_wheat_flour
  (initial_amount : ℕ)
  (price_rice : ℕ)
  (num_rice : ℕ)
  (price_soda : ℕ)
  (num_soda : ℕ)
  (num_wheat_flour : ℕ)
  (remaining_balance : ℕ)
  (total_spent : ℕ)
  (amount_spent_on_rice_and_soda : ℕ)
  (amount_spent_on_wheat_flour : ℕ)
  (price_per_packet_wheat_flour : ℕ) 
  (h_initial_amount : initial_amount = 500)
  (h_price_rice : price_rice = 20)
  (h_num_rice : num_rice = 2)
  (h_price_soda : price_soda = 150)
  (h_num_soda : num_soda = 1)
  (h_num_wheat_flour : num_wheat_flour = 3)
  (h_remaining_balance : remaining_balance = 235)
  (h_total_spent : total_spent = initial_amount - remaining_balance)
  (h_amount_spent_on_rice_and_soda : amount_spent_on_rice_and_soda = price_rice * num_rice + price_soda * num_soda)
  (h_amount_spent_on_wheat_flour : amount_spent_on_wheat_flour = total_spent - amount_spent_on_rice_and_soda)
  (h_price_per_packet_wheat_flour : price_per_packet_wheat_flour = amount_spent_on_wheat_flour / num_wheat_flour) :
  price_per_packet_wheat_flour = 25 :=
by 
  sorry

end NUMINAMATH_GPT_price_of_wheat_flour_l667_66762


namespace NUMINAMATH_GPT_smallest_expression_l667_66723

theorem smallest_expression (a b : ℝ) (h : b < 0) : a + b < a ∧ a < a - b :=
by
  sorry

end NUMINAMATH_GPT_smallest_expression_l667_66723


namespace NUMINAMATH_GPT_angle_BPE_l667_66758

-- Define the conditions given in the problem
def triangle_ABC (A B C : ℝ) : Prop := A = 60 ∧ 
  (∃ (B₁ B₂ B₃ : ℝ), B₁ = B / 3 ∧ B₂ = B / 3 ∧ B₃ = B / 3) ∧ 
  (∃ (C₁ C₂ C₃ : ℝ), C₁ = C / 3 ∧ C₂ = C / 3 ∧ C₃ = C / 3) ∧ 
  (B + C = 120)

-- State the theorem to proof
theorem angle_BPE (A B C x : ℝ) (h : triangle_ABC A B C) : x = 50 := by
  sorry

end NUMINAMATH_GPT_angle_BPE_l667_66758


namespace NUMINAMATH_GPT_mean_of_five_numbers_l667_66759

theorem mean_of_five_numbers (x1 x2 x3 x4 x5 : ℚ) (h_sum : x1 + x2 + x3 + x4 + x5 = 1/3) : 
  (x1 + x2 + x3 + x4 + x5) / 5 = 1/15 :=
by 
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_l667_66759


namespace NUMINAMATH_GPT_age_difference_l667_66726

-- Defining the age variables as fractions
variables (x y : ℚ)

-- Given conditions
axiom ratio1 : 2 * x / y = 2 / y
axiom ratio2 : (5 * x + 20) / (y + 20) = 8 / 3

-- The main theorem to prove the difference between Mahesh's and Suresh's ages.
theorem age_difference : 5 * x - y = (125 / 8) := sorry

end NUMINAMATH_GPT_age_difference_l667_66726


namespace NUMINAMATH_GPT_rationalize_sqrt_three_sub_one_l667_66727

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_sqrt_three_sub_one_l667_66727


namespace NUMINAMATH_GPT_value_of_f_at_neg1_l667_66741

def f (x : ℤ) : ℤ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

theorem value_of_f_at_neg1 : f (-1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_neg1_l667_66741


namespace NUMINAMATH_GPT_root_of_quadratic_l667_66702

theorem root_of_quadratic (a b c : ℝ) :
  (4 * a + 2 * b + c = 0) ↔ (a * 2^2 + b * 2 + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_root_of_quadratic_l667_66702


namespace NUMINAMATH_GPT_solve_quadratic_polynomial_l667_66763

noncomputable def q (x : ℝ) : ℝ := -4.5 * x^2 + 4.5 * x + 135

theorem solve_quadratic_polynomial : 
  (q (-5) = 0) ∧ (q 6 = 0) ∧ (q 7 = -54) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_polynomial_l667_66763


namespace NUMINAMATH_GPT_quiz_scores_dropped_students_l667_66725

theorem quiz_scores_dropped_students (T S : ℝ) :
  T = 30 * 60.25 →
  T - S = 26 * 63.75 →
  S = 150 :=
by
  intros hT h_rem
  -- Additional steps would be implemented here.
  sorry

end NUMINAMATH_GPT_quiz_scores_dropped_students_l667_66725


namespace NUMINAMATH_GPT_S_2011_value_l667_66745

-- Definitions based on conditions provided in the problem
def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sum_seq (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Problem statement
theorem S_2011_value
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h_arith : arithmetic_seq a_n)
  (h_sum : sum_seq S_n a_n)
  (h_init : a_n 1 = -2011)
  (h_cond : (S_n 2010) / 2010 - (S_n 2008) / 2008 = 2) :
  S_n 2011 = -2011 := 
sorry

end NUMINAMATH_GPT_S_2011_value_l667_66745


namespace NUMINAMATH_GPT_mul_65_35_eq_2275_l667_66786

theorem mul_65_35_eq_2275 : 65 * 35 = 2275 := by
  sorry

end NUMINAMATH_GPT_mul_65_35_eq_2275_l667_66786


namespace NUMINAMATH_GPT_total_messages_three_days_l667_66790

theorem total_messages_three_days :
  ∀ (A1 A2 A3 L1 L2 L3 : ℕ),
  A1 = L1 - 20 →
  L1 = 120 →
  L2 = (1 / 3 : ℚ) * L1 →
  A2 = 2 * A1 →
  A1 + L1 = A3 + L3 →
  (A1 + L1 + A2 + L2 + A3 + L3 = 680) := by
  intros A1 A2 A3 L1 L2 L3 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_messages_three_days_l667_66790


namespace NUMINAMATH_GPT_area_of_gray_region_is_27pi_l667_66715

-- Define the conditions
def concentric_circles (inner_radius outer_radius : ℝ) :=
  2 * inner_radius = outer_radius

def width_of_gray_region (inner_radius outer_radius width : ℝ) :=
  width = outer_radius - inner_radius

-- Define the proof problem
theorem area_of_gray_region_is_27pi
(inner_radius outer_radius : ℝ) 
(h1 : concentric_circles inner_radius outer_radius)
(h2 : width_of_gray_region inner_radius outer_radius 3) :
π * outer_radius^2 - π * inner_radius^2 = 27 * π :=
by
  -- Proof goes here, but it is not required as per instructions
  sorry

end NUMINAMATH_GPT_area_of_gray_region_is_27pi_l667_66715


namespace NUMINAMATH_GPT_proof_problem_l667_66729

-- Definitions of the propositions
def p : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → y = 5 - 3 * x
def q : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → 2 * x + 6 * y - 4 = 0

-- Translate the mathematical proof problem into a Lean theorem
theorem proof_problem : 
  (p ∧ ¬q) ∧ ¬((¬p) ∧ q) :=
by
  -- You can fill in the exact proof steps here
  sorry

end NUMINAMATH_GPT_proof_problem_l667_66729


namespace NUMINAMATH_GPT_six_digit_permutation_reverse_div_by_11_l667_66717

theorem six_digit_permutation_reverse_div_by_11 
  (a b c : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 9)
  (h_b : 0 ≤ b ∧ b ≤ 9)
  (h_c : 0 ≤ c ∧ c ≤ 9)
  (X : ℕ)
  (h_X : X = 100001 * a + 10010 * b + 1100 * c) :
  11 ∣ X :=
by 
  sorry

end NUMINAMATH_GPT_six_digit_permutation_reverse_div_by_11_l667_66717


namespace NUMINAMATH_GPT_remaining_load_after_three_deliveries_l667_66771

def initial_load : ℝ := 50000
def unload_first_store (load : ℝ) : ℝ := load - 0.10 * load
def unload_second_store (load : ℝ) : ℝ := load - 0.20 * load
def unload_third_store (load : ℝ) : ℝ := load - 0.15 * load

theorem remaining_load_after_three_deliveries : 
  unload_third_store (unload_second_store (unload_first_store initial_load)) = 30600 := 
by
  sorry

end NUMINAMATH_GPT_remaining_load_after_three_deliveries_l667_66771


namespace NUMINAMATH_GPT_roots_quadratic_eq_value_l667_66787

theorem roots_quadratic_eq_value (d e : ℝ) (h : 3 * d^2 + 4 * d - 7 = 0) (h' : 3 * e^2 + 4 * e - 7 = 0) : 
  (d - 2) * (e - 2) = 13 / 3 := 
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_eq_value_l667_66787


namespace NUMINAMATH_GPT_eq_solution_set_l667_66709

theorem eq_solution_set :
  {x : ℝ | (2 / (x + 2)) + (4 / (x + 8)) ≥ 3 / 4} = {x : ℝ | -2 < x ∧ x ≤ 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_eq_solution_set_l667_66709


namespace NUMINAMATH_GPT_find_p_l667_66796

theorem find_p (m n p : ℝ) 
  (h1 : m = (n / 2) - (2 / 5)) 
  (h2 : m + p = ((n + 4) / 2) - (2 / 5)) :
  p = 2 :=
sorry

end NUMINAMATH_GPT_find_p_l667_66796


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l667_66794

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h : ∀ n, a n = a 1 + (n - 1) * d) (h_6 : a 6 = 1) :
  a 2 + a 10 = 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l667_66794


namespace NUMINAMATH_GPT_maximum_weekly_hours_l667_66718

-- Conditions
def regular_rate : ℝ := 8 -- $8 per hour for the first 20 hours
def overtime_rate : ℝ := regular_rate * 1.25 -- 25% higher than the regular rate
def max_weekly_earnings : ℝ := 460 -- Maximum of $460 in a week
def regular_hours : ℕ := 20 -- First 20 hours are regular hours
def regular_earnings : ℝ := regular_hours * regular_rate -- Earnings for regular hours
def max_overtime_earnings : ℝ := max_weekly_earnings - regular_earnings -- Maximum overtime earnings

-- Proof problem statement
theorem maximum_weekly_hours : regular_hours + (max_overtime_earnings / overtime_rate) = 50 := by
  sorry

end NUMINAMATH_GPT_maximum_weekly_hours_l667_66718


namespace NUMINAMATH_GPT_outer_perimeter_fence_l667_66753

-- Definitions based on given conditions
def total_posts : Nat := 16
def post_width_feet : Real := 0.5 -- 6 inches converted to feet
def gap_length_feet : Real := 6 -- gap between posts in feet
def num_sides : Nat := 4 -- square field has 4 sides

-- Hypotheses that capture conditions and intermediate calculations
def num_corners : Nat := 4
def non_corner_posts : Nat := total_posts - num_corners
def non_corner_posts_per_side : Nat := non_corner_posts / num_sides
def posts_per_side : Nat := non_corner_posts_per_side + 2
def gaps_per_side : Nat := posts_per_side - 1
def length_gaps_per_side : Real := gaps_per_side * gap_length_feet
def total_post_width_per_side : Real := posts_per_side * post_width_feet
def length_one_side : Real := length_gaps_per_side + total_post_width_per_side
def perimeter : Real := num_sides * length_one_side

-- The theorem to prove
theorem outer_perimeter_fence : perimeter = 106 := by
  sorry

end NUMINAMATH_GPT_outer_perimeter_fence_l667_66753


namespace NUMINAMATH_GPT_value_of_a_l667_66712
noncomputable def find_a (a b c : ℝ) : ℝ :=
if 2 * b = a + c ∧ (a * c) * (b * c) = ((a * b) ^ 2) ∧ a + b + c = 6 then a else 0

theorem value_of_a (a b c : ℝ) :
  (2 * b = a + c) ∧ ((a * c) * (b * c) = (a * b) ^ 2) ∧ (a + b + c = 6) ∧ (a ≠ c) ∧ (a ≠ b) ∧ (b ≠ c) → a = 4 :=
by sorry

end NUMINAMATH_GPT_value_of_a_l667_66712


namespace NUMINAMATH_GPT_volume_of_resulting_solid_is_9_l667_66768

-- Defining the initial cube with edge length 3
def initial_cube_edge_length : ℝ := 3

-- Defining the volume of the initial cube
def initial_cube_volume : ℝ := initial_cube_edge_length^3

-- Defining the volume of the resulting solid after some parts are cut off
def resulting_solid_volume : ℝ := 9

-- Theorem stating that given the initial conditions, the volume of the resulting solid is 9
theorem volume_of_resulting_solid_is_9 : resulting_solid_volume = 9 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_resulting_solid_is_9_l667_66768


namespace NUMINAMATH_GPT_eighteenth_entry_l667_66737

def r_8 (n : ℕ) : ℕ := n % 8

theorem eighteenth_entry (n : ℕ) (h : r_8 (3 * n) ≤ 3) : n = 17 :=
sorry

end NUMINAMATH_GPT_eighteenth_entry_l667_66737


namespace NUMINAMATH_GPT_minimum_sum_of_box_dimensions_l667_66730

theorem minimum_sum_of_box_dimensions :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end NUMINAMATH_GPT_minimum_sum_of_box_dimensions_l667_66730


namespace NUMINAMATH_GPT_coat_shirt_ratio_l667_66714

variable (P S C k : ℕ)

axiom h1 : P + S = 100
axiom h2 : P + C = 244
axiom h3 : C = k * S
axiom h4 : C = 180

theorem coat_shirt_ratio (P S C k : ℕ) (h1 : P + S = 100) (h2 : P + C = 244) (h3 : C = k * S) (h4 : C = 180) :
  C / S = 5 :=
sorry

end NUMINAMATH_GPT_coat_shirt_ratio_l667_66714


namespace NUMINAMATH_GPT_range_of_a_l667_66746

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (2 * a * x + 3 * x > 2 * a + 3) ↔ (x < 1)) → (a < -3 / 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l667_66746


namespace NUMINAMATH_GPT_books_per_shelf_l667_66793

theorem books_per_shelf 
  (initial_books : ℕ) 
  (sold_books : ℕ) 
  (num_shelves : ℕ) 
  (remaining_books : ℕ := initial_books - sold_books) :
  initial_books = 40 → sold_books = 20 → num_shelves = 5 → remaining_books / num_shelves = 4 :=
by
  sorry

end NUMINAMATH_GPT_books_per_shelf_l667_66793


namespace NUMINAMATH_GPT_bob_height_in_inches_l667_66720

theorem bob_height_in_inches (tree_height shadow_tree bob_shadow : ℝ)
  (h1 : tree_height = 50)
  (h2 : shadow_tree = 25)
  (h3 : bob_shadow = 6) :
  (12 * (tree_height / shadow_tree) * bob_shadow) = 144 :=
by sorry

end NUMINAMATH_GPT_bob_height_in_inches_l667_66720


namespace NUMINAMATH_GPT_meteorological_forecast_l667_66798

theorem meteorological_forecast (prob_rain : ℝ) (h1 : prob_rain = 0.7) :
  (prob_rain = 0.7) → "There is a high probability of needing to carry rain gear when going out tomorrow." = "Correct" :=
by
  intro h
  sorry

end NUMINAMATH_GPT_meteorological_forecast_l667_66798


namespace NUMINAMATH_GPT_function_properties_and_k_range_l667_66704

theorem function_properties_and_k_range :
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 ^ x) ∧ (∀ y, y > 0)) ∧
  (∀ k : ℝ, (∃ t : ℝ, t > 0 ∧ (t^2 - 2*t + k = 0)) ↔ (0 < k ∧ k < 1)) :=
by sorry

end NUMINAMATH_GPT_function_properties_and_k_range_l667_66704


namespace NUMINAMATH_GPT_peter_pizza_fraction_l667_66769

def pizza_slices : ℕ := 16
def peter_initial_slices : ℕ := 2
def shared_slices : ℕ := 2
def shared_with_paul : ℕ := shared_slices / 2
def total_slices_peter_ate := peter_initial_slices + shared_with_paul
def fraction_peter_ate : ℚ := total_slices_peter_ate / pizza_slices

theorem peter_pizza_fraction :
  fraction_peter_ate = 3 / 16 :=
by
  -- Leave space for the proof, which is not required.
  sorry

end NUMINAMATH_GPT_peter_pizza_fraction_l667_66769


namespace NUMINAMATH_GPT_y_power_x_equals_49_l667_66739

theorem y_power_x_equals_49 (x y : ℝ) (h : |x - 2| = -(y + 7)^2) : y ^ x = 49 := by
  sorry

end NUMINAMATH_GPT_y_power_x_equals_49_l667_66739
