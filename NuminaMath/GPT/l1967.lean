import Mathlib

namespace smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l1967_196743

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

end smallest_positive_period_axis_of_symmetry_minimum_value_on_interval_l1967_196743


namespace books_sold_on_wednesday_l1967_196790

theorem books_sold_on_wednesday
  (initial_stock : ℕ)
  (sold_monday : ℕ)
  (sold_tuesday : ℕ)
  (sold_thursday : ℕ)
  (sold_friday : ℕ)
  (percent_unsold : ℚ) :
  initial_stock = 900 →
  sold_monday = 75 →
  sold_tuesday = 50 →
  sold_thursday = 78 →
  sold_friday = 135 →
  percent_unsold = 55.333333333333336 →
  ∃ (sold_wednesday : ℕ), sold_wednesday = 64 :=
by
  sorry

end books_sold_on_wednesday_l1967_196790


namespace solution_set_of_inequality_l1967_196719

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
sorry

end solution_set_of_inequality_l1967_196719


namespace enlarged_decal_height_l1967_196733

theorem enlarged_decal_height (original_width original_height new_width : ℕ)
  (original_width_eq : original_width = 3)
  (original_height_eq : original_height = 2)
  (new_width_eq : new_width = 15)
  (proportions_consistent : ∀ h : ℕ, new_width * original_height = original_width * h) :
  ∃ new_height, new_height = 10 :=
by sorry

end enlarged_decal_height_l1967_196733


namespace deshaun_read_books_over_summer_l1967_196752

theorem deshaun_read_books_over_summer 
  (summer_days : ℕ)
  (average_pages_per_book : ℕ)
  (ratio_closest_person : ℝ)
  (pages_read_per_day_second_person : ℕ)
  (books_read : ℕ)
  (total_pages_second_person_read : ℕ)
  (h1 : summer_days = 80)
  (h2 : average_pages_per_book = 320)
  (h3 : ratio_closest_person = 0.75)
  (h4 : pages_read_per_day_second_person = 180)
  (h5 : total_pages_second_person_read = pages_read_per_day_second_person * summer_days)
  (h6 : books_read * average_pages_per_book = total_pages_second_person_read / ratio_closest_person) :
  books_read = 60 :=
by {
  sorry
}

end deshaun_read_books_over_summer_l1967_196752


namespace candy_bar_cost_l1967_196770

theorem candy_bar_cost (initial_amount change : ℕ) (h : initial_amount = 50) (hc : change = 5) : 
  initial_amount - change = 45 :=
by
  -- sorry is used to skip the proof
  sorry

end candy_bar_cost_l1967_196770


namespace initial_percentage_water_is_80_l1967_196738

noncomputable def initial_kola_solution := 340
noncomputable def added_sugar := 3.2
noncomputable def added_water := 10
noncomputable def added_kola := 6.8
noncomputable def final_percentage_sugar := 14.111111111111112
noncomputable def percentage_kola := 6

theorem initial_percentage_water_is_80 :
  ∃ (W : ℝ), W = 80 :=
by
  sorry

end initial_percentage_water_is_80_l1967_196738


namespace find_x_for_sin_minus_cos_eq_sqrt2_l1967_196731

theorem find_x_for_sin_minus_cos_eq_sqrt2 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
by
  sorry

end find_x_for_sin_minus_cos_eq_sqrt2_l1967_196731


namespace coordinates_of_point_A_l1967_196747

def f (x : ℝ) : ℝ := x^2 + 3 * x

theorem coordinates_of_point_A (a : ℝ) (b : ℝ) 
    (slope_condition : deriv f a = 7) 
    (point_condition : f a = b) : 
    a = 2 ∧ b = 10 := 
by {
    sorry
}

end coordinates_of_point_A_l1967_196747


namespace Tom_Brady_passing_yards_l1967_196754

-- Definitions
def record := 5999
def current_yards := 4200
def games_left := 6

-- Proof problem statement
theorem Tom_Brady_passing_yards :
  (record + 1 - current_yards) / games_left = 300 := by
  sorry

end Tom_Brady_passing_yards_l1967_196754


namespace find_x_plus_y_l1967_196740

theorem find_x_plus_y (x y : ℕ) 
  (h1 : 4^x = 16^(y + 1)) 
  (h2 : 5^(2 * y) = 25^(x - 2)) : 
  x + y = 2 := 
sorry

end find_x_plus_y_l1967_196740


namespace boys_in_class_is_120_l1967_196714

-- Definitions from conditions
def num_boys_in_class (number_of_girls number_of_boys : Nat) : Prop :=
  ∃ x : Nat, number_of_girls = 5 * x ∧ number_of_boys = 6 * x ∧
             (5 * x - 20) * 3 = 2 * (6 * x)

-- The theorem proving that given the conditions, the number of boys in the class is 120.
theorem boys_in_class_is_120 (number_of_girls number_of_boys : Nat) (h : num_boys_in_class number_of_girls number_of_boys) :
  number_of_boys = 120 :=
by
  sorry

end boys_in_class_is_120_l1967_196714


namespace five_digit_divisible_by_four_digit_l1967_196711

theorem five_digit_divisible_by_four_digit (x y z u v : ℕ) (h1 : 1 ≤ x) (h2 : x < 10) (h3 : y < 10) (h4 : z < 10) (h5 : u < 10) (h6 : v < 10)
  (h7 : (x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v) % (x * 10^3 + y * 10^2 + u * 10 + v) = 0) : 
  ∃ N, 10 ≤ N ∧ N ≤ 99 ∧ 
  x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v = N * 10^3 ∧
  10 * (x * 10^3 + y * 10^2 + u * 10 + v) = x * 10^4 + y * 10^3 + z * 10^2 + u * 10 + v :=
sorry

end five_digit_divisible_by_four_digit_l1967_196711


namespace prize_calculations_l1967_196763

-- Definitions for the conditions
def total_prizes := 50
def first_prize_unit_price := 20
def second_prize_unit_price := 14
def third_prize_unit_price := 8
def num_second_prize (x : ℕ) := 3 * x - 2
def num_third_prize (x : ℕ) := total_prizes - x - num_second_prize x
def total_cost (x : ℕ) := first_prize_unit_price * x + second_prize_unit_price * num_second_prize x + third_prize_unit_price * num_third_prize x

-- Proof problem statement
theorem prize_calculations (x : ℕ) (h : num_second_prize x = 22) : 
  num_second_prize x = 3 * x - 2 ∧ 
  num_third_prize x = 52 - 4 * x ∧ 
  total_cost x = 30 * x + 388 ∧ 
  total_cost 8 = 628 :=
by
  sorry

end prize_calculations_l1967_196763


namespace total_bouncy_balls_l1967_196769

-- Definitions based on the conditions of the problem
def packs_of_red := 4
def packs_of_yellow := 8
def packs_of_green := 4
def balls_per_pack := 10

-- Theorem stating the conclusion to be proven
theorem total_bouncy_balls :
  (packs_of_red + packs_of_yellow + packs_of_green) * balls_per_pack = 160 := 
by
  sorry

end total_bouncy_balls_l1967_196769


namespace stan_needs_more_minutes_l1967_196793

/-- Stan has 10 songs each of 3 minutes and 15 songs each of 2 minutes. His run takes 100 minutes.
    Prove that he needs 40 more minutes of songs in his playlist. -/
theorem stan_needs_more_minutes 
    (num_3min_songs : ℕ) 
    (num_2min_songs : ℕ) 
    (time_per_3min_song : ℕ) 
    (time_per_2min_song : ℕ) 
    (total_run_time : ℕ) 
    (given_minutes_3min_songs : num_3min_songs = 10)
    (given_minutes_2min_songs : num_2min_songs = 15)
    (given_time_per_3min_song : time_per_3min_song = 3)
    (given_time_per_2min_song : time_per_2min_song = 2)
    (given_total_run_time : total_run_time = 100)
    : num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song = 60 →
      total_run_time - (num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song) = 40 := 
by
    sorry

end stan_needs_more_minutes_l1967_196793


namespace find_a15_l1967_196771

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

def arithmetic_sequence (an : ℕ → ℝ) := ∃ (a₁ d : ℝ), ∀ n, an n = a₁ + (n - 1) * d

theorem find_a15
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a9 : a 9 = 10) :
  a 15 = 16 := 
sorry

end ArithmeticSequence

end find_a15_l1967_196771


namespace mary_needs_more_sugar_l1967_196746

theorem mary_needs_more_sugar 
  (sugar_needed flour_needed salt_needed already_added_flour : ℕ)
  (h1 : sugar_needed = 11)
  (h2 : flour_needed = 6)
  (h3 : salt_needed = 9)
  (h4 : already_added_flour = 12) :
  (sugar_needed - salt_needed) = 2 :=
by
  sorry

end mary_needs_more_sugar_l1967_196746


namespace equation_of_perpendicular_line_l1967_196720

theorem equation_of_perpendicular_line (c : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0 ∧ 2 * x + y - 5 = 0) → (x - 2 * y - 3 = 0) := 
by
  sorry

end equation_of_perpendicular_line_l1967_196720


namespace predict_monthly_savings_l1967_196762

noncomputable def sum_x_i := 80
noncomputable def sum_y_i := 20
noncomputable def sum_x_i_y_i := 184
noncomputable def sum_x_i_sq := 720
noncomputable def n := 10
noncomputable def x_bar := sum_x_i / n
noncomputable def y_bar := sum_y_i / n
noncomputable def b := (sum_x_i_y_i - n * x_bar * y_bar) / (sum_x_i_sq - n * x_bar^2)
noncomputable def a := y_bar - b * x_bar
noncomputable def regression_eqn(x: ℝ) := b * x + a

theorem predict_monthly_savings :
  regression_eqn 7 = 1.7 :=
by
  sorry

end predict_monthly_savings_l1967_196762


namespace find_k_l1967_196710

theorem find_k (k : ℤ) :
  (-x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) → k = -16 :=
by
  sorry

end find_k_l1967_196710


namespace portion_of_profit_divided_equally_l1967_196775

-- Definitions for the given conditions
def total_investment_mary : ℝ := 600
def total_investment_mike : ℝ := 400
def total_profit : ℝ := 7500
def profit_diff : ℝ := 1000

-- Main statement
theorem portion_of_profit_divided_equally (E P : ℝ) 
  (h1 : total_profit = E + P)
  (h2 : E + (3/5) * P = E + (2/5) * P + profit_diff) :
  E = 2500 :=
by
  sorry

end portion_of_profit_divided_equally_l1967_196775


namespace area_of_gray_region_l1967_196707

open Real

-- Define the circles and the radii.
def circleC_center : Prod Real Real := (5, 5)
def radiusC : Real := 5

def circleD_center : Prod Real Real := (15, 5)
def radiusD : Real := 5

-- The main theorem stating the area of the gray region bound by the circles and the x-axis.
theorem area_of_gray_region : 
  let area_rectangle := (10:Real) * (5:Real)
  let area_sectors := (2:Real) * ((1/4) * (5:Real)^2 * π)
  area_rectangle - area_sectors = 50 - 12.5 * π :=
by
  sorry

end area_of_gray_region_l1967_196707


namespace find_m_from_inequality_l1967_196788

theorem find_m_from_inequality :
  (∀ x, x^2 - (m+2)*x > 0 ↔ (x < 0 ∨ x > 2)) → m = 0 :=
by
  sorry

end find_m_from_inequality_l1967_196788


namespace sequence_formula_l1967_196737

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4 ^ n - 1 :=
by
  sorry

end sequence_formula_l1967_196737


namespace find_first_number_of_sequence_l1967_196776

theorem find_first_number_of_sequence
    (a : ℕ → ℕ)
    (h1 : ∀ n, 3 ≤ n → a n = a (n-1) * a (n-2))
    (h2 : a 8 = 36)
    (h3 : a 9 = 1296)
    (h4 : a 10 = 46656) :
    a 1 = 60466176 := 
sorry

end find_first_number_of_sequence_l1967_196776


namespace twenty_five_point_zero_six_million_in_scientific_notation_l1967_196730

theorem twenty_five_point_zero_six_million_in_scientific_notation :
  (25.06e6 : ℝ) = 2.506 * 10^7 :=
by
  -- The proof would go here, but we use sorry to skip the proof.
  sorry

end twenty_five_point_zero_six_million_in_scientific_notation_l1967_196730


namespace problem_inequality_l1967_196767

theorem problem_inequality (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_abc : a * b * c = 1) :
  (a - 1 + (1 / b)) * (b - 1 + (1 / c)) * (c - 1 + (1 / a)) ≤ 1 :=
sorry

end problem_inequality_l1967_196767


namespace range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l1967_196749

open Real

theorem range_a_of_abs_2x_minus_a_eq_1_two_real_solutions :
  {a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (abs (2^x1 - a) = 1) ∧ (abs (2^x2 - a) = 1)} = {a : ℝ | 1 < a} :=
by
  sorry

end range_a_of_abs_2x_minus_a_eq_1_two_real_solutions_l1967_196749


namespace calculate_expression_l1967_196708

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1 / x^2) * (y^2 + 1 / y^2) = x^4 - y^4 := by
  sorry

end calculate_expression_l1967_196708


namespace ajax_store_price_l1967_196758

theorem ajax_store_price (original_price : ℝ) (first_discount_rate : ℝ) (second_discount_rate : ℝ)
    (h_original: original_price = 180)
    (h_first_discount : first_discount_rate = 0.5)
    (h_second_discount : second_discount_rate = 0.2) :
    let first_discount_price := original_price * (1 - first_discount_rate)
    let saturday_price := first_discount_price * (1 - second_discount_rate)
    saturday_price = 72 :=
by
    sorry

end ajax_store_price_l1967_196758


namespace area_of_square_with_diagonal_40_l1967_196739

theorem area_of_square_with_diagonal_40 {d : ℝ} (h : d = 40) : ∃ A : ℝ, A = 800 :=
by
  sorry

end area_of_square_with_diagonal_40_l1967_196739


namespace find_a_and_b_l1967_196742

theorem find_a_and_b (a b : ℤ) (h : ∀ x : ℝ, x ≤ 0 → (a*x + 2)*(x^2 + 2*b) ≤ 0) : a = 1 ∧ b = -2 := 
by 
  -- Proof steps would go here, but they are omitted as per instructions.
  sorry

end find_a_and_b_l1967_196742


namespace line_point_t_l1967_196784

theorem line_point_t (t : ℝ) : 
  (∃ t, (0, 3) = (0, 3) ∧ (-8, 0) = (-8, 0) ∧ (5 - 3) / t = 3 / 8) → (t = 16 / 3) :=
by
  sorry

end line_point_t_l1967_196784


namespace triangle_area_is_correct_l1967_196780

noncomputable def isosceles_triangle_area : Prop :=
  let side_large_square := 6 -- sides of the large square WXYZ
  let area_large_square := side_large_square * side_large_square
  let side_small_square := 2 -- sides of the smaller squares
  let BC := side_large_square - 2 * side_small_square -- length of BC
  let height_AM := side_large_square / 2 + side_small_square -- height of the triangle from A to M
  let area_ABC := (BC * height_AM) / 2 -- area of the triangle ABC
  area_large_square = 36 ∧ BC = 2 ∧ height_AM = 5 ∧ area_ABC = 5

theorem triangle_area_is_correct : isosceles_triangle_area := sorry

end triangle_area_is_correct_l1967_196780


namespace initial_number_of_friends_is_six_l1967_196792

theorem initial_number_of_friends_is_six
  (car_cost : ℕ)
  (car_wash_earnings : ℕ)
  (F : ℕ)
  (additional_cost_when_one_friend_leaves : ℕ)
  (h1 : car_cost = 1700)
  (h2 : car_wash_earnings = 500)
  (remaining_cost := car_cost - car_wash_earnings)
  (cost_per_friend_before := remaining_cost / F)
  (cost_per_friend_after := remaining_cost / (F - 1))
  (h3 : additional_cost_when_one_friend_leaves = 40)
  (h4 : cost_per_friend_after = cost_per_friend_before + additional_cost_when_one_friend_leaves) :
  F = 6 :=
by
  sorry

end initial_number_of_friends_is_six_l1967_196792


namespace product_ne_sum_11_times_l1967_196778

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0
def prime_sum_product_condition (a b c d : ℕ) : Prop := 
  a * b * c * d = 11 * (a + b + c + d)

theorem product_ne_sum_11_times (a b c d : ℕ)
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (hd : is_prime d)
  (h : prime_sum_product_condition a b c d) :
  (a + b + c + d ≠ 46) ∧ (a + b + c + d ≠ 47) ∧ (a + b + c + d ≠ 48) :=
by  
  sorry

end product_ne_sum_11_times_l1967_196778


namespace problem_statement_l1967_196764

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 4
def g (x : ℝ) : ℝ := 2*x - 1

-- State the theorem and provide the necessary conditions
theorem problem_statement : f (g 5) - g (f 5) = 381 :=
by
  sorry

end problem_statement_l1967_196764


namespace f_inv_f_inv_17_l1967_196729

noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def f_inv (y : ℝ) : ℝ := (y + 3) / 4

theorem f_inv_f_inv_17 : f_inv (f_inv 17) = 2 := by
  sorry

end f_inv_f_inv_17_l1967_196729


namespace trig_identity_proof_l1967_196760

theorem trig_identity_proof (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) :
  Real.sin (2 * α - π / 6) + Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end trig_identity_proof_l1967_196760


namespace find_natural_numbers_satisfying_prime_square_l1967_196744

-- Define conditions as a Lean statement
theorem find_natural_numbers_satisfying_prime_square (n : ℕ) (h : ∃ p : ℕ, Prime p ∧ (2 * n^2 + 3 * n - 35 = p^2)) :
  n = 4 ∨ n = 12 :=
sorry

end find_natural_numbers_satisfying_prime_square_l1967_196744


namespace equal_diagonals_implies_quad_or_pent_l1967_196789

-- Define a convex polygon with n edges and equal diagonals
structure ConvexPolygon (n : ℕ) :=
(edges : ℕ)
(convex : Prop)
(diagonalsEqualLength : Prop)

-- State the theorem to prove
theorem equal_diagonals_implies_quad_or_pent (n : ℕ) (poly : ConvexPolygon n) 
    (h1 : poly.convex) 
    (h2 : poly.diagonalsEqualLength) :
    (n = 4) ∨ (n = 5) :=
sorry

end equal_diagonals_implies_quad_or_pent_l1967_196789


namespace additional_carpet_needed_is_94_l1967_196700

noncomputable def area_room_a : ℝ := 4 * 20

noncomputable def area_room_b : ℝ := area_room_a / 2.5

noncomputable def total_area : ℝ := area_room_a + area_room_b

noncomputable def carpet_jessie_has : ℝ := 18

noncomputable def additional_carpet_needed : ℝ := total_area - carpet_jessie_has

theorem additional_carpet_needed_is_94 :
  additional_carpet_needed = 94 := by
  sorry

end additional_carpet_needed_is_94_l1967_196700


namespace sum_of_given_geom_series_l1967_196777

-- Define the necessary conditions
def first_term (a : ℕ) := a = 2
def common_ratio (r : ℕ) := r = 3
def number_of_terms (n : ℕ) := n = 6

-- Define the sum of the geometric series
def sum_geom_series (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

-- State the theorem
theorem sum_of_given_geom_series :
  first_term 2 → common_ratio 3 → number_of_terms 6 → sum_geom_series 2 3 6 = 728 :=
by
  intros h1 h2 h3
  rw [first_term] at h1
  rw [common_ratio] at h2
  rw [number_of_terms] at h3
  have h1 : 2 = 2 := by exact h1
  have h2 : 3 = 3 := by exact h2
  have h3 : 6 = 6 := by exact h3
  exact sorry

end sum_of_given_geom_series_l1967_196777


namespace eval_expression_l1967_196713

theorem eval_expression (a b : ℤ) (h₁ : a = 4) (h₂ : b = -2) : -a - b^2 + a*b + a^2 = 0 := by
  sorry

end eval_expression_l1967_196713


namespace trig_cos_sum_l1967_196785

open Real

theorem trig_cos_sum :
  cos (37 * (π / 180)) * cos (23 * (π / 180)) - sin (37 * (π / 180)) * sin (23 * (π / 180)) = 1 / 2 :=
by
  sorry

end trig_cos_sum_l1967_196785


namespace problem_l1967_196750

def f (x : ℝ) : ℝ := sorry -- We assume f is defined as per the given condition but do not provide an implementation.

theorem problem (h : ∀ x : ℝ, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (Real.pi / 12)) = - (Real.sqrt 3) / 2 :=
by
  sorry -- The proof is omitted

end problem_l1967_196750


namespace classification_of_square_and_cube_roots_l1967_196723

-- Define the three cases: positive, zero, and negative
inductive NumberCase
| positive 
| zero 
| negative 

-- Define the concept of "classification and discussion thinking"
def is_classification_and_discussion_thinking (cases : List NumberCase) : Prop :=
  cases = [NumberCase.positive, NumberCase.zero, NumberCase.negative]

-- The main statement to be proven
theorem classification_of_square_and_cube_roots :
  is_classification_and_discussion_thinking [NumberCase.positive, NumberCase.zero, NumberCase.negative] :=
by
  sorry

end classification_of_square_and_cube_roots_l1967_196723


namespace turnover_june_l1967_196794

variable (TurnoverApril TurnoverMay : ℝ)

theorem turnover_june (h1 : TurnoverApril = 10) (h2 : TurnoverMay = 12) :
  TurnoverMay * (1 + (TurnoverMay - TurnoverApril) / TurnoverApril) = 14.4 := by
  sorry

end turnover_june_l1967_196794


namespace same_terminal_side_l1967_196772

theorem same_terminal_side (α : ℝ) (k : ℤ) (h : α = -51) : 
  ∃ (m : ℤ), α + m * 360 = k * 360 - 51 :=
by {
    sorry
}

end same_terminal_side_l1967_196772


namespace mario_garden_total_blossoms_l1967_196727

def hibiscus_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

def rose_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

theorem mario_garden_total_blossoms :
  let weeks := 2
  let hibiscus1 := hibiscus_growth 2 3 weeks
  let hibiscus2 := hibiscus_growth (2 * 2) 4 weeks
  let hibiscus3 := hibiscus_growth (4 * (2 * 2)) 5 weeks
  let rose1 := rose_growth 3 2 weeks
  let rose2 := rose_growth 5 3 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 = 64 := 
by
  sorry

end mario_garden_total_blossoms_l1967_196727


namespace train_speed_calculation_l1967_196759

open Real

noncomputable def train_speed_in_kmph (V : ℝ) : ℝ := V * 3.6

theorem train_speed_calculation (L V : ℝ) (h1 : L = 16 * V) (h2 : L + 280 = 30 * V) :
  train_speed_in_kmph V = 72 :=
by
  sorry

end train_speed_calculation_l1967_196759


namespace quadratic_identity_l1967_196761

variables {R : Type*} [CommRing R] [IsDomain R]

-- Define the quadratic polynomial P
def P (a b c x : R) : R := a * x^2 + b * x + c

-- Conditions as definitions in Lean
variables (a b c : R) (h₁ : P a b c a = 2021 * b * c)
                (h₂ : P a b c b = 2021 * c * a)
                (h₃ : P a b c c = 2021 * a * b)
                (dist : (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c))

-- The main theorem statement
theorem quadratic_identity : a + 2021 * b + c = 0 :=
sorry

end quadratic_identity_l1967_196761


namespace inverse_proportion_passing_through_l1967_196725

theorem inverse_proportion_passing_through (k : ℝ) :
  (∀ x y : ℝ, (y = k / x) → (x = 3 → y = 2)) → k = 6 := 
by
  sorry

end inverse_proportion_passing_through_l1967_196725


namespace triangle_side_b_value_l1967_196709

theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) (h1 : a = Real.sqrt 3) (h2 : A = 60) (h3 : C = 75) : b = Real.sqrt 2 :=
sorry

end triangle_side_b_value_l1967_196709


namespace range_of_a_l1967_196705

noncomputable def setA (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (setA a ∩ setB = ∅) ↔ (a > 3 ∨ (-1 / 2 ≤ a ∧ a ≤ 2)) := 
  sorry

end range_of_a_l1967_196705


namespace rogers_parents_paid_percentage_l1967_196704

variables 
  (house_cost : ℝ)
  (down_payment_percentage : ℝ)
  (remaining_balance_owed : ℝ)
  (down_payment : ℝ := down_payment_percentage * house_cost)
  (remaining_balance_after_down : ℝ := house_cost - down_payment)
  (parents_payment : ℝ := remaining_balance_after_down - remaining_balance_owed)
  (percentage_paid_by_parents : ℝ := (parents_payment / remaining_balance_after_down) * 100)

theorem rogers_parents_paid_percentage
  (h1 : house_cost = 100000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : remaining_balance_owed = 56000) :
  percentage_paid_by_parents = 30 :=
sorry

end rogers_parents_paid_percentage_l1967_196704


namespace find_distance_l1967_196701

variable (D V : ℕ)

axiom normal_speed : V = 25
axiom time_difference : (D / V) - (D / (V + 5)) = 2

theorem find_distance : D = 300 :=
by
  sorry

end find_distance_l1967_196701


namespace people_in_room_l1967_196797

/-- 
   Problem: Five-sixths of the people in a room are seated in five-sixths of the chairs.
   The rest of the people are standing. If there are 10 empty chairs, 
   prove that there are 60 people in the room.
-/
theorem people_in_room (people chairs : ℕ) 
  (h_condition1 : 5 / 6 * people = 5 / 6 * chairs) 
  (h_condition2 : chairs = 60) :
  people = 60 :=
by
  sorry

end people_in_room_l1967_196797


namespace total_students_is_45_l1967_196768

-- Define the initial conditions with the definitions provided
def drunk_drivers : Nat := 6
def speeders : Nat := 7 * drunk_drivers - 3
def total_students : Nat := drunk_drivers + speeders

-- The theorem to prove that the total number of students is 45
theorem total_students_is_45 : total_students = 45 :=
by
  sorry

end total_students_is_45_l1967_196768


namespace repair_time_calculation_l1967_196756

-- Assume amount of work is represented as units
def work_10_people_45_minutes := 10 * 45
def work_20_people_20_minutes := 20 * 20

-- Assuming the flood destroys 2 units per minute as calculated in the solution
def flood_rate := 2

-- Calculate total initial units of the dike
def dike_initial_units :=
  work_10_people_45_minutes - flood_rate * 45

-- Given 14 people are repairing the dam
def repair_rate_14_people := 14 - flood_rate

-- Statement to prove that 14 people need 30 minutes to repair the dam
theorem repair_time_calculation :
  dike_initial_units / repair_rate_14_people = 30 :=
by
  sorry

end repair_time_calculation_l1967_196756


namespace trigonometric_ratio_l1967_196751

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end trigonometric_ratio_l1967_196751


namespace unique_solution_abc_l1967_196748

theorem unique_solution_abc (a b c : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
(h1 : b ∣ 2^a - 1) 
(h2 : c ∣ 2^b - 1) 
(h3 : a ∣ 2^c - 1) : 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end unique_solution_abc_l1967_196748


namespace possible_values_of_a_l1967_196783

theorem possible_values_of_a (a : ℚ) : 
  (a^2 = 9 * 16) ∨ (16 * a = 81) ∨ (9 * a = 256) → 
  a = 12 ∨ a = -12 ∨ a = 81 / 16 ∨ a = 256 / 9 :=
by
  intros h
  sorry

end possible_values_of_a_l1967_196783


namespace hyperbola_equation_l1967_196741

theorem hyperbola_equation 
  (vertex : ℝ × ℝ) 
  (asymptote_slope : ℝ) 
  (h_vertex : vertex = (2, 0))
  (h_asymptote : asymptote_slope = Real.sqrt 2) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 8 = 1) := 
by
    sorry

end hyperbola_equation_l1967_196741


namespace check_perfect_squares_l1967_196765

-- Define the prime factorizations of each option
def optionA := 3^3 * 4^5 * 7^7
def optionB := 3^4 * 4^4 * 7^6
def optionC := 3^6 * 4^3 * 7^8
def optionD := 3^5 * 4^6 * 7^5
def optionE := 3^4 * 4^6 * 7^7

-- Definition of a perfect square (all exponents in prime factorization are even)
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p : ℕ, (p ^ 2 ∣ n) -> (p ∣ n)

-- The Lean statement asserting which options are perfect squares
theorem check_perfect_squares :
  (is_perfect_square optionB) ∧ (is_perfect_square optionC) ∧
  ¬(is_perfect_square optionA) ∧ ¬(is_perfect_square optionD) ∧ ¬(is_perfect_square optionE) :=
by sorry

end check_perfect_squares_l1967_196765


namespace marked_price_percentage_l1967_196721

theorem marked_price_percentage (L C M S : ℝ) 
  (h1 : C = 0.7 * L) 
  (h2 : C = 0.7 * S) 
  (h3 : S = 0.9 * M) 
  (h4 : S = L) 
  : M = (10 / 9) * L := 
by
  sorry

end marked_price_percentage_l1967_196721


namespace cost_of_plastering_is_334_point_8_l1967_196712

def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.45

def bottom_area : ℝ := tank_length * tank_width
def long_wall_area : ℝ := 2 * (tank_length * tank_depth)
def short_wall_area : ℝ := 2 * (tank_width * tank_depth)
def total_surface_area : ℝ := bottom_area + long_wall_area + short_wall_area
def total_cost : ℝ := total_surface_area * cost_per_sq_meter

theorem cost_of_plastering_is_334_point_8 :
  total_cost = 334.8 :=
by
  sorry

end cost_of_plastering_is_334_point_8_l1967_196712


namespace fraction_zero_x_value_l1967_196716

theorem fraction_zero_x_value (x : ℝ) (h : (x^2 - 4) / (x - 2) = 0) (h2 : x ≠ 2) : x = -2 :=
sorry

end fraction_zero_x_value_l1967_196716


namespace rolls_sold_to_grandmother_l1967_196755

theorem rolls_sold_to_grandmother (t u n s g : ℕ) 
  (h1 : t = 45)
  (h2 : u = 10)
  (h3 : n = 6)
  (h4 : s = 28)
  (total_sold : t - s = g + u + n) : 
  g = 1 := 
  sorry

end rolls_sold_to_grandmother_l1967_196755


namespace like_terms_expression_value_l1967_196745

theorem like_terms_expression_value (m n : ℤ) (h1 : m = 3) (h2 : n = 1) :
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 33 := by
  sorry

end like_terms_expression_value_l1967_196745


namespace crayon_production_correct_l1967_196798

def numColors := 4
def crayonsPerColor := 2
def boxesPerHour := 5
def hours := 4

def crayonsPerBox := numColors * crayonsPerColor
def crayonsPerHour := boxesPerHour * crayonsPerBox
def totalCrayons := hours * crayonsPerHour

theorem crayon_production_correct :
  totalCrayons = 160 :=  
by
  sorry

end crayon_production_correct_l1967_196798


namespace find_b_l1967_196774

theorem find_b (b : ℝ) (h : ∃ (f_inv : ℝ → ℝ), (∀ x y, f_inv (2^x + b) = y) ∧ f_inv 5 = 2) :
    b = 1 := by
  sorry

end find_b_l1967_196774


namespace value_of_expression_l1967_196702

noncomputable def f : ℝ → ℝ
| x => if x > 0 then -1 else if x < 0 then 1 else 0

theorem value_of_expression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b := 
sorry

end value_of_expression_l1967_196702


namespace avg_rate_first_half_l1967_196753

theorem avg_rate_first_half (total_distance : ℕ) (avg_rate : ℕ) (first_half_distance : ℕ) (second_half_distance : ℕ)
  (rate_first_half : ℕ) (time_first_half : ℕ) (time_second_half : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  second_half_distance = first_half_distance →
  time_second_half = 3 * time_first_half →
  avg_rate = 40 →
  total_distance = first_half_distance + second_half_distance →
  total_time = time_first_half + time_second_half →
  avg_rate = total_distance / total_time →
  time_first_half = first_half_distance / rate_first_half →
  rate_first_half = 80
  :=
  sorry

end avg_rate_first_half_l1967_196753


namespace gcd_pow_sub_one_l1967_196791

theorem gcd_pow_sub_one (n m : ℕ) (h1 : n = 1005) (h2 : m = 1016) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2047 := by
  rw [h1, h2]
  sorry

end gcd_pow_sub_one_l1967_196791


namespace value_of_frac_mul_l1967_196781

theorem value_of_frac_mul (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 2 * d) :
  (a * c) / (b * d) = 8 :=
by
  sorry

end value_of_frac_mul_l1967_196781


namespace alpha_value_l1967_196724

theorem alpha_value
  (β γ δ α : ℝ) 
  (h1 : β = 100)
  (h2 : γ = 30)
  (h3 : δ = 150)
  (h4 : α + β + γ + 0.5 * γ = 360) : 
  α = 215 :=
by
  sorry

end alpha_value_l1967_196724


namespace relationship_log2_2_pow_03_l1967_196722

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem relationship_log2_2_pow_03 : 
  log_base_2 0.3 < (0.3)^2 ∧ (0.3)^2 < 2^(0.3) :=
by
  sorry

end relationship_log2_2_pow_03_l1967_196722


namespace minimum_value_is_81_l1967_196782

noncomputable def minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) : ℝ :=
a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem minimum_value_is_81 {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minimum_value a b c h1 h2 h3 h4 = 81 :=
sorry

end minimum_value_is_81_l1967_196782


namespace infinite_solutions_l1967_196728

theorem infinite_solutions (b : ℤ) : 
  (∀ x : ℤ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := 
by sorry

end infinite_solutions_l1967_196728


namespace ab_sum_l1967_196736

theorem ab_sum (a b : ℝ) (h : |a - 4| + (b + 1)^2 = 0) : a + b = 3 :=
by
  sorry -- this is where the proof would go

end ab_sum_l1967_196736


namespace f_increasing_on_neg_inf_to_one_l1967_196757

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 8

theorem f_increasing_on_neg_inf_to_one :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end f_increasing_on_neg_inf_to_one_l1967_196757


namespace fourth_root_difference_l1967_196734

theorem fourth_root_difference : (81 : ℝ) ^ (1 / 4 : ℝ) - (1296 : ℝ) ^ (1 / 4 : ℝ) = -3 :=
by
  sorry

end fourth_root_difference_l1967_196734


namespace find_b_of_perpendicular_lines_l1967_196718

theorem find_b_of_perpendicular_lines (b : ℝ) (h : 4 * b - 8 = 0) : b = 2 := 
by 
  sorry

end find_b_of_perpendicular_lines_l1967_196718


namespace original_triangle_area_l1967_196786

theorem original_triangle_area (A_new : ℝ) (scale_factor : ℝ) (A_original : ℝ) 
  (h1: scale_factor = 5) (h2: A_new = 200) (h3: A_new = scale_factor^2 * A_original) : 
  A_original = 8 :=
by
  sorry

end original_triangle_area_l1967_196786


namespace markup_percentage_is_ten_l1967_196766

theorem markup_percentage_is_ten (S C : ℝ)
  (h1 : S - C = 0.0909090909090909 * S) :
  (S - C) / C * 100 = 10 :=
by
  sorry

end markup_percentage_is_ten_l1967_196766


namespace range_of_3a_minus_b_l1967_196795

theorem range_of_3a_minus_b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3)
                             (h3 : 2 < a - b) (h4 : a - b < 4) :
    ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 11 ∧ x = 3 * a - b :=
sorry

end range_of_3a_minus_b_l1967_196795


namespace sufficient_but_not_necessary_condition_l1967_196787

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1967_196787


namespace evaluate_floor_of_negative_seven_halves_l1967_196773

def floor_of_negative_seven_halves : ℤ :=
  Int.floor (-7 / 2)

theorem evaluate_floor_of_negative_seven_halves :
  floor_of_negative_seven_halves = -4 :=
by
  sorry

end evaluate_floor_of_negative_seven_halves_l1967_196773


namespace number_of_arrangements_SEES_l1967_196799

theorem number_of_arrangements_SEES : 
  ∃ n : ℕ, 
    (∀ (total_letters E S : ℕ), 
      total_letters = 4 ∧ E = 2 ∧ S = 2 → 
      n = Nat.factorial total_letters / (Nat.factorial E * Nat.factorial S)) → 
    n = 6 := 
by 
  sorry

end number_of_arrangements_SEES_l1967_196799


namespace Jen_visits_either_but_not_both_l1967_196703

-- Define the events and their associated probabilities
def P_Chile : ℝ := 0.30
def P_Madagascar : ℝ := 0.50

-- Define the probability of visiting both assuming independence
def P_both : ℝ := P_Chile * P_Madagascar

-- Define the probability of visiting either but not both
def P_either_but_not_both : ℝ := P_Chile + P_Madagascar - 2 * P_both

-- The problem statement
theorem Jen_visits_either_but_not_both : P_either_but_not_both = 0.65 := by
  /- The proof goes here -/
  sorry

end Jen_visits_either_but_not_both_l1967_196703


namespace angle_same_terminal_side_l1967_196726

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ (θ : ℤ), θ = -324 ∧ 
    ∀ α : ℤ, α = 36 + k * 360 → 
            ( (α % 360 = θ % 360) ∨ (α % 360 + 360 = θ % 360) ∨ (θ % 360 + 360 = α % 360)) :=
by
  sorry

end angle_same_terminal_side_l1967_196726


namespace stone_width_l1967_196735

theorem stone_width (length_hall breadth_hall : ℝ) (num_stones length_stone : ℝ) (total_area_hall total_area_stones area_stone : ℝ)
  (h1 : length_hall = 36) (h2 : breadth_hall = 15) (h3 : num_stones = 5400) (h4 : length_stone = 2) 
  (h5 : total_area_hall = length_hall * breadth_hall * (10 * 10))
  (h6 : total_area_stones = num_stones * area_stone) 
  (h7 : area_stone = length_stone * (5 : ℝ)) 
  (h8 : total_area_stones = total_area_hall) : 
  (5 : ℝ) = 5 :=  
by sorry

end stone_width_l1967_196735


namespace area_of_ABCM_l1967_196796

-- Definitions of the problem conditions
def length_of_sides (P : ℕ) := 4
def forms_right_angle (P : ℕ) := True
def M_intersection (AG CH : ℝ) := True

-- Proposition that quadrilateral ABCM has the correct area
theorem area_of_ABCM (a b c m : ℝ) :
  (length_of_sides 12 = 4) ∧
  (forms_right_angle 12) ∧
  (M_intersection a b) →
  ∃ area_ABCM : ℝ, area_ABCM = 88/5 :=
by
  sorry

end area_of_ABCM_l1967_196796


namespace servings_per_bottle_l1967_196715

-- Definitions based on conditions
def total_guests : ℕ := 120
def servings_per_guest : ℕ := 2
def total_bottles : ℕ := 40

-- Theorem stating that given the conditions, the servings per bottle is 6
theorem servings_per_bottle : (total_guests * servings_per_guest) / total_bottles = 6 := by
  sorry

end servings_per_bottle_l1967_196715


namespace avg_class_weight_is_46_67_l1967_196706

-- Define the total number of students in section A
def num_students_a : ℕ := 40

-- Define the average weight of students in section A
def avg_weight_a : ℚ := 50

-- Define the total number of students in section B
def num_students_b : ℕ := 20

-- Define the average weight of students in section B
def avg_weight_b : ℚ := 40

-- Calculate the total weight of section A
def total_weight_a : ℚ := num_students_a * avg_weight_a

-- Calculate the total weight of section B
def total_weight_b : ℚ := num_students_b * avg_weight_b

-- Calculate the total weight of the entire class
def total_weight_class : ℚ := total_weight_a + total_weight_b

-- Calculate the total number of students in the entire class
def total_students_class : ℕ := num_students_a + num_students_b

-- Calculate the average weight of the entire class
def avg_weight_class : ℚ := total_weight_class / total_students_class

-- Theorem to prove
theorem avg_class_weight_is_46_67 :
  avg_weight_class = 46.67 := sorry

end avg_class_weight_is_46_67_l1967_196706


namespace find_common_ratio_l1967_196717

-- Declare the sequence and conditions
variables {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions of the problem 
def positive_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ m n : ℕ, a m = a 0 * q ^ m) ∧ q > 0

def third_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 3 + a 5 = 5

def fifth_term_seventh_term_condition (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 5 + a 7 = 20

-- The final lean statement proving the common ratio is 2
theorem find_common_ratio 
  (h1 : positive_geometric_sequence a q) 
  (h2 : third_term_condition a q) 
  (h3 : fifth_term_seventh_term_condition a q) : 
  q = 2 :=
sorry

end find_common_ratio_l1967_196717


namespace age_difference_l1967_196779

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 11) : C + 11 = A :=
by {
  sorry
}

end age_difference_l1967_196779


namespace geometric_sequence_sum_l1967_196732

-- Define the relations for geometric sequences
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (m n p q : ℕ), m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : is_geometric_sequence a) (h_pos : ∀ n, a n > 0)
  (h_cond : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) : a 5 + a 7 = 6 :=
sorry

end geometric_sequence_sum_l1967_196732
