import Mathlib

namespace locus_of_P_coordinates_of_P_l69_69398

-- Define the points A and B
def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (2, -1)

-- Define the line l : 4x + 3y - 2 = 0
def l (x y: ℝ) := 4 * x + 3 * y - 2 = 0

-- Problem (1): Equation of the locus of point P such that |PA| = |PB|
theorem locus_of_P (P : ℝ × ℝ) :
  (∃ P, dist P A = dist P B) ↔ (∀ x y : ℝ, P = (x, y) → x - y - 5 = 0) :=
sorry

-- Problem (2): Coordinates of P such that |PA| = |PB| and the distance from P to line l is 2
theorem coordinates_of_P (a b : ℝ):
  (dist (a, b) A = dist (a, b) B ∧ abs (4 * a + 3 * b - 2) / 5 = 2) ↔
  ((a = 1 ∧ b = -4) ∨ (a = 27 / 7 ∧ b = -8 / 7)) :=
sorry

end locus_of_P_coordinates_of_P_l69_69398


namespace king_lancelot_seats_38_l69_69914

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69914


namespace robert_more_photos_than_claire_l69_69329

theorem robert_more_photos_than_claire
  (claire_photos : ℕ)
  (Lisa_photos : ℕ)
  (Robert_photos : ℕ)
  (Claire_takes_photos : claire_photos = 12)
  (Lisa_takes_photos : Lisa_photos = 3 * claire_photos)
  (Lisa_and_Robert_same_photos : Lisa_photos = Robert_photos) :
  Robert_photos - claire_photos = 24 := by
    sorry

end robert_more_photos_than_claire_l69_69329


namespace king_arthur_round_table_seats_l69_69897

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69897


namespace factorial_expression_l69_69949

open Nat

theorem factorial_expression :
  7 * (6!) + 6 * (5!) + 2 * (5!) = 6000 :=
by
  sorry

end factorial_expression_l69_69949


namespace calculation_division_l69_69506

theorem calculation_division :
  ((27 * 0.92 * 0.85) / (23 * 1.7 * 1.8)) = 0.3 :=
by
  sorry

end calculation_division_l69_69506


namespace men_meet_4_miles_nearer_R_than_S_l69_69213

def distance_between_points : ℝ := 76
def rate_at_R : ℝ := 4.5
def initial_rate_at_S : ℝ := 3.25
def rate_increase_at_S_per_hour : ℝ := 0.5

theorem men_meet_4_miles_nearer_R_than_S :
  ∃ (h : ℕ) (x : ℝ), x = 4 ∧
  (rate_at_R * h + (h / 2 * (2 * initial_rate_at_S + (h - 1) * rate_increase_at_S_per_hour)) = distance_between_points) :=
begin
  sorry
end

end men_meet_4_miles_nearer_R_than_S_l69_69213


namespace greatest_divisor_four_consecutive_integers_l69_69726

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69726


namespace quadratic_function_range_l69_69597

theorem quadratic_function_range (x : ℝ) (h : x ≥ 0) : 
  3 ≤ x^2 + 2 * x + 3 :=
by {
  sorry
}

end quadratic_function_range_l69_69597


namespace original_price_l69_69067

theorem original_price (x : ℝ) (h1 : x > 0) (h2 : 1.12 * x - x = 270) : x = 2250 :=
by
  sorry

end original_price_l69_69067


namespace xy_eq_119_imp_sum_values_l69_69293

theorem xy_eq_119_imp_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0)
(hx_lt_30 : x < 30) (hy_lt_30 : y < 30) (h : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 := 
sorry

end xy_eq_119_imp_sum_values_l69_69293


namespace area_enclosed_by_abs_eq_12_l69_69030

theorem area_enclosed_by_abs_eq_12 :
  let A := { p : ℝ × ℝ | abs p.1 + abs (3 * p.2) = 12 } in
  ∃ area : ℝ, area = 96 ∧
    (∀ (triangle : set (ℝ × ℝ)),
      triangle ⊆ A →
      is_triangle triangle →
      area_of triangle = 24) →
    (∃ (number_of_triangles : ℕ), number_of_triangles = 4) :=
by
  -- Definitions and steps would go here in a proper proof.
  sorry

end area_enclosed_by_abs_eq_12_l69_69030


namespace chocolate_distribution_l69_69491

theorem chocolate_distribution (n : ℕ) 
  (h1 : 12 * 2 ≤ n * 2 ∨ n * 2 ≤ 12 * 2) 
  (h2 : ∃ d : ℚ, (12 / n) = d ∧ d * n = 12) : 
  n = 15 :=
by 
  sorry

end chocolate_distribution_l69_69491


namespace circle_regions_l69_69153

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l69_69153


namespace largest_possible_s_l69_69433

theorem largest_possible_s (r s : ℕ) (h1 : 3 ≤ s) (h2 : s ≤ r) (h3 : s < 122)
    (h4 : ∀ r s, (61 * (s - 2) * r = 60 * (r - 2) * s)) : s ≤ 121 :=
by
  sorry

end largest_possible_s_l69_69433


namespace quadratic_sequence_exists_l69_69243

theorem quadratic_sequence_exists (b c : ℤ) : 
  ∃ (n : ℕ) (a : ℕ → ℤ), 
  a 0 = b ∧ 
  a n = c ∧ 
  ∀ i, 1 ≤ i → i ≤ n → |a i - a (i - 1)| = i^2 :=
sorry

end quadratic_sequence_exists_l69_69243


namespace polynomial_product_roots_l69_69518

theorem polynomial_product_roots (a b c : ℝ) : 
  (∀ x, (x - (Real.sin (Real.pi / 6))) * (x - (Real.sin (Real.pi / 3))) * (x - (Real.sin (5 * Real.pi / 6))) = x^3 + a * x^2 + b * x + c) → 
  a * b * c = Real.sqrt 3 / 2 :=
by
  sorry

end polynomial_product_roots_l69_69518


namespace div_product_four_consecutive_integers_l69_69850

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69850


namespace bella_bakes_most_cookies_per_batch_l69_69423

theorem bella_bakes_most_cookies_per_batch (V : ℝ) :
  let alex_cookies := V / 9
  let bella_cookies := V / 7
  let carlo_cookies := V / 8
  let dana_cookies := V / 10
  alex_cookies < bella_cookies ∧ carlo_cookies < bella_cookies ∧ dana_cookies < bella_cookies :=
sorry

end bella_bakes_most_cookies_per_batch_l69_69423


namespace horizontal_asymptote_exists_x_intercepts_are_roots_l69_69321

noncomputable def given_function (x : ℝ) : ℝ :=
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_exists :
  ∃ L : ℝ, ∀ x : ℝ, (∃ M : ℝ, M > 0 ∧ (∀ x > M, abs (given_function x - L) < 1)) ∧ L = 0 := 
sorry

theorem x_intercepts_are_roots :
  ∀ y, y = 0 ↔ ∃ x : ℝ, x ≠ 0 ∧ 15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5 = 0 :=
sorry

end horizontal_asymptote_exists_x_intercepts_are_roots_l69_69321


namespace units_digit_of_sum_sequence_is_8_l69_69215

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit_sum_sequence : ℕ :=
  let term (n : ℕ) := (factorial n + n * n) % 10
  (term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7 + term 8 + term 9) % 10

theorem units_digit_of_sum_sequence_is_8 :
  units_digit_sum_sequence = 8 :=
sorry

end units_digit_of_sum_sequence_is_8_l69_69215


namespace number_of_correct_propositions_is_one_l69_69455

def obtuse_angle_is_second_quadrant (θ : ℝ) : Prop :=
  θ > 90 ∧ θ < 180

def acute_angle (θ : ℝ) : Prop :=
  θ < 90

def first_quadrant_not_negative (θ : ℝ) : Prop :=
  θ > 0 ∧ θ < 90

def second_quadrant_greater_first (θ₁ θ₂ : ℝ) : Prop :=
  (θ₁ > 90 ∧ θ₁ < 180) → (θ₂ > 0 ∧ θ₂ < 90) → θ₁ > θ₂

theorem number_of_correct_propositions_is_one :
  (¬ ∀ θ, obtuse_angle_is_second_quadrant θ) ∧
  (∀ θ, acute_angle θ → θ < 90) ∧
  (¬ ∀ θ, first_quadrant_not_negative θ) ∧
  (¬ ∀ θ₁ θ₂, second_quadrant_greater_first θ₁ θ₂) →
  1 = 1 :=
by
  sorry

end number_of_correct_propositions_is_one_l69_69455


namespace four_consecutive_integers_divisible_by_12_l69_69686

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69686


namespace possible_values_a1_l69_69434

def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem possible_values_a1 {a : ℕ → ℤ} (h1 : ∀ n : ℕ, a n + a (n + 1) = 2 * n - 1)
  (h2 : ∃ k : ℕ, sequence_sum a k = 190 ∧ sequence_sum a (k + 1) = 190) :
  (a 0 = -20 ∨ a 0 = 19) :=
sorry

end possible_values_a1_l69_69434


namespace average_runs_per_game_l69_69360

-- Define the number of games
def games : ℕ := 6

-- Define the list of runs scored in each game
def runs : List ℕ := [1, 4, 4, 5, 5, 5]

-- The sum of the runs
def total_runs : ℕ := List.sum runs

-- The average runs per game
def avg_runs : ℚ := total_runs / games

-- The theorem to prove
theorem average_runs_per_game : avg_runs = 4 := by sorry

end average_runs_per_game_l69_69360


namespace time_to_cross_man_l69_69229

-- Define the conversion from km/h to m/s
def kmh_to_ms (speed_kmh : ℕ) : ℕ := (speed_kmh * 1000) / 3600

-- Given conditions
def length_of_train : ℕ := 150
def speed_of_train_kmh : ℕ := 180

-- Calculate speed in m/s
def speed_of_train_ms : ℕ := kmh_to_ms speed_of_train_kmh

-- Proof problem statement
theorem time_to_cross_man : (length_of_train : ℕ) / (speed_of_train_ms : ℕ) = 3 := by
  sorry

end time_to_cross_man_l69_69229


namespace area_enclosed_by_abs_linear_eq_l69_69022

theorem area_enclosed_by_abs_linear_eq (x y : ℝ) :
  |x| + |3 * y| = 12 → (enclosure_area := 4 * (1 / 2 * 12 * 4)) = 96 := by
  sorry

end area_enclosed_by_abs_linear_eq_l69_69022


namespace first_discount_percentage_l69_69995

theorem first_discount_percentage (x : ℝ) :
  let initial_price := 26.67
  let final_price := 15.0
  let second_discount := 0.25
  (initial_price * (1 - x / 100) * (1 - second_discount) = final_price) → x = 25 :=
by
  intros
  sorry

end first_discount_percentage_l69_69995


namespace greatest_divisor_of_consecutive_product_l69_69856

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69856


namespace divisor_of_four_consecutive_integers_l69_69752

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69752


namespace greatest_divisor_four_consecutive_integers_l69_69731

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69731


namespace arithmetic_mean_l69_69262

theorem arithmetic_mean (x y : ℝ) (h1 : x = Real.sqrt 2 - 1) (h2 : y = 1 / (Real.sqrt 2 - 1)) :
  (x + y) / 2 = Real.sqrt 2 := sorry

end arithmetic_mean_l69_69262


namespace negation_of_p_l69_69543

def p (x : ℝ) : Prop := x^3 - x^2 + 1 < 0

theorem negation_of_p : (¬ ∀ x : ℝ, p x) ↔ ∃ x : ℝ, ¬ p x := by
  sorry

end negation_of_p_l69_69543


namespace circumscribed_circle_radius_l69_69531

noncomputable def radius_of_circumscribed_circle 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) : ℝ :=
2

theorem circumscribed_circle_radius 
  {a b c A B C : ℝ} 
  (h1 : b = 2 * Real.sqrt 3) 
  (h2 : A + B + C = Real.pi) 
  (h3 : 2 * B = A + C) :
  radius_of_circumscribed_circle a b c A B C h1 h2 h3 = 2 :=
sorry

end circumscribed_circle_radius_l69_69531


namespace leonid_painted_cells_l69_69315

theorem leonid_painted_cells (k l : ℕ) (hkl : k * l = 74) :
  ∃ (painted_cells : ℕ), painted_cells = ((2 * k + 1) * (2 * l + 1) - 74) ∧ (painted_cells = 373 ∨ painted_cells = 301) :=
by
  sorry

end leonid_painted_cells_l69_69315


namespace problem1_problem2_part1_problem2_part2_l69_69362

-- Problem 1
theorem problem1 (x : ℚ) (h : x = 11 / 12) : 
  (2 * x - 5) * (2 * x + 5) - (2 * x - 3) ^ 2 = -23 := 
by sorry

-- Problem 2
theorem problem2_part1 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  a^2 + b^2 = 22 := 
by sorry

theorem problem2_part2 (a b : ℚ) (h1 : a + b = 6) (h2 : a * b = 7) : 
  (a - b)^2 = 8 := 
by sorry

end problem1_problem2_part1_problem2_part2_l69_69362


namespace sum_of_d_and_e_l69_69002

-- Define the original numbers and their sum
def original_first := 3742586
def original_second := 4829430
def correct_sum := 8572016

-- The given incorrect addition result
def given_sum := 72120116

-- Define the digits d and e
def d := 2
def e := 8

-- Define the correct adjusted sum if we replace d with e
def adjusted_first := 3782586
def adjusted_second := 4889430
def adjusted_sum := 8672016

-- State the final theorem
theorem sum_of_d_and_e : 
  (given_sum != correct_sum) → 
  (original_first + original_second = correct_sum) → 
  (adjusted_first + adjusted_second = adjusted_sum) → 
  (d + e = 10) :=
by
  sorry

end sum_of_d_and_e_l69_69002


namespace solve_for_q_l69_69540

variable (R t m q : ℝ)

def given_condition : Prop :=
  R = t / ((2 + m) ^ q)

theorem solve_for_q (h : given_condition R t m q) : 
  q = (Real.log (t / R)) / (Real.log (2 + m)) := 
sorry

end solve_for_q_l69_69540


namespace factor_complete_polynomial_l69_69100

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l69_69100


namespace circle_region_count_l69_69160

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l69_69160


namespace regions_formed_l69_69163

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l69_69163


namespace product_of_roots_l69_69265

noncomputable def quadratic_has_product_of_roots (A B C : ℤ) : ℚ :=
  C / A

theorem product_of_roots (α β : ℚ) (h : 12 * α^2 + 28 * α - 320 = 0) (h2 : 12 * β^2 + 28 * β - 320 = 0) :
  quadratic_has_product_of_roots 12 28 (-320) = -80 / 3 :=
by
  -- Insert proof here
  sorry

end product_of_roots_l69_69265


namespace negation_of_proposition_l69_69981

theorem negation_of_proposition : 
  ¬ (∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 > 0 := by
  sorry

end negation_of_proposition_l69_69981


namespace total_bags_sold_l69_69061

theorem total_bags_sold (first_week second_week third_week fourth_week total : ℕ) 
  (h1 : first_week = 15) 
  (h2 : second_week = 3 * first_week) 
  (h3 : third_week = 20) 
  (h4 : fourth_week = 20) 
  (h5 : total = first_week + second_week + third_week + fourth_week) : 
  total = 100 := 
sorry

end total_bags_sold_l69_69061


namespace jasmine_percentage_new_solution_l69_69242

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_jasmine_percent : ℝ := 0.10
def added_jasmine : ℝ := 5
def added_water : ℝ := 15

-- Define the correct answer
theorem jasmine_percentage_new_solution :
  let initial_jasmine := initial_jasmine_percent * initial_volume
  let new_jasmine := initial_jasmine + added_jasmine
  let total_new_volume := initial_volume + added_jasmine + added_water
  (new_jasmine / total_new_volume) * 100 = 13 := 
by 
  sorry

end jasmine_percentage_new_solution_l69_69242


namespace num_int_values_not_satisfying_l69_69106

theorem num_int_values_not_satisfying:
  (∃ n : ℕ, n = 7 ∧ (∃ x : ℤ, 7 * x^2 + 25 * x + 24 ≤ 30)) :=
sorry

end num_int_values_not_satisfying_l69_69106


namespace king_lancelot_seats_38_l69_69919

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69919


namespace expected_number_of_2s_when_three_dice_rolled_l69_69210

def probability_of_rolling_2 : ℚ := 1 / 6
def probability_of_not_rolling_2 : ℚ := 5 / 6

theorem expected_number_of_2s_when_three_dice_rolled :
  (0 * (probability_of_not_rolling_2)^3 + 
   1 * 3 * (probability_of_rolling_2) * (probability_of_not_rolling_2)^2 + 
   2 * 3 * (probability_of_rolling_2)^2 * (probability_of_not_rolling_2) + 
   3 * (probability_of_rolling_2)^3) = 
   1 / 2 :=
by
  sorry

end expected_number_of_2s_when_three_dice_rolled_l69_69210


namespace find_divisors_l69_69299

theorem find_divisors (N : ℕ) :
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ (N = 2013 ∨ N = 1006 ∨ N = 105 ∨ N = 52) := by
  sorry

end find_divisors_l69_69299


namespace function_order_l69_69964

theorem function_order (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (2 - x) = f x)
  (h2 : ∀ x : ℝ, f (x + 2) = f (x - 2))
  (h3 : ∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 ∧ 1 ≤ x2 ∧ x2 ≤ 3 → (f x1 - f x2) / (x1 - x2) < 0) :
  f 2016 = f 2014 ∧ f 2014 > f 2015 :=
by
  sorry

end function_order_l69_69964


namespace cheaper_to_buy_more_books_l69_69492

def C (n : ℕ) : ℕ :=
  if n < 1 then 0
  else if n ≤ 20 then 15 * n
  else if n ≤ 40 then 14 * n - 5
  else 13 * n

noncomputable def apply_discount (n : ℕ) (cost : ℕ) : ℕ :=
  cost - 10 * (n / 10)

theorem cheaper_to_buy_more_books : 
  ∃ (n_vals : Finset ℕ), n_vals.card = 5 ∧ ∀ n ∈ n_vals, apply_discount (n + 1) (C (n + 1)) < apply_discount n (C n) :=
sorry

end cheaper_to_buy_more_books_l69_69492


namespace arithmetic_sequence_diff_l69_69144

theorem arithmetic_sequence_diff (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 7 = a 3 + 4 * d) :
  a 2008 - a 2000 = 8 * d :=
by
  sorry

end arithmetic_sequence_diff_l69_69144


namespace greatest_divisor_four_consecutive_integers_l69_69727

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69727


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69774

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69774


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69711

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69711


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69806

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69806


namespace divisor_of_product_of_four_consecutive_integers_l69_69653

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69653


namespace cos_beta_of_acute_angles_l69_69986

theorem cos_beta_of_acute_angles (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = Real.sqrt 5 / 5)
  (hsin_alpha_minus_beta : Real.sin (α - β) = 3 * Real.sqrt 10 / 10) :
  Real.cos β = 7 * Real.sqrt 2 / 10 :=
sorry

end cos_beta_of_acute_angles_l69_69986


namespace greatest_divisor_four_consecutive_l69_69744

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69744


namespace devah_erases_895_dots_l69_69257

theorem devah_erases_895_dots :
  let number_of_dots := 1000
  ∃ erased_count, erased_count = number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2)) :=
by {
  let number_of_dots := 1000;
  have h : (number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2))) = 895,
  {
    sorry,
  },
  use (number_of_dots - (number_of_dots - (Nat.range number_of_dots).count (λ n, ∀ b ∈ nat.digits 3 n, b ≠ 2))),
  exact h,
}

end devah_erases_895_dots_l69_69257


namespace probability_angie_carlos_two_seats_apart_l69_69943

theorem probability_angie_carlos_two_seats_apart :
  let people := ["Angie", "Bridget", "Carlos", "Diego", "Edwin"]
  let table_size := people.length
  let total_arrangements := (Nat.factorial (table_size - 1))
  let favorable_arrangements := 2 * (Nat.factorial (table_size - 2))
  total_arrangements > 0 ∧
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 2 :=
by {
  sorry
}

end probability_angie_carlos_two_seats_apart_l69_69943


namespace sequence_sum_formula_l69_69532

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_S : ∀ n, S n = (1 / 6) * (a n ^ 2 + 3 * a n - 4)) : 
  ∀ n, S n = (3 / 2) * n ^ 2 + (5 / 2) * n :=
by
  sorry

end sequence_sum_formula_l69_69532


namespace hexagon_same_length_probability_l69_69557

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l69_69557


namespace greatest_divisor_of_consecutive_product_l69_69851

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69851


namespace typing_speed_ratio_l69_69222

-- Define Tim's and Tom's typing speeds
variables (T t : ℝ)

-- Conditions from the problem
def condition1 : Prop := T + t = 15
def condition2 : Prop := T + 1.6 * t = 18

-- The proposition to prove: the ratio of Tom's typing speed to Tim's is 1:2
theorem typing_speed_ratio (h1 : condition1 T t) (h2 : condition2 T t) : t / T = 1 / 2 :=
sorry

end typing_speed_ratio_l69_69222


namespace perfect_square_expression_l69_69375

theorem perfect_square_expression (x y : ℕ) (p : ℕ) [Fact (Nat.Prime p)]
    (h : 4 * x^2 + 8 * y^2 + (2 * x - 3 * y) * p - 12 * x * y = 0) :
    ∃ (n : ℕ), 4 * y + 1 = n^2 :=
sorry

end perfect_square_expression_l69_69375


namespace box_dimensions_l69_69550

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) : 
  a = 5 ∧ b = 8 ∧ c = 12 := 
by
  sorry

end box_dimensions_l69_69550


namespace cubic_poly_sum_l69_69395

noncomputable def q (x : ℕ) : ℤ := sorry

axiom h0 : q 1 = 5
axiom h1 : q 6 = 24
axiom h2 : q 10 = 16
axiom h3 : q 15 = 34

theorem cubic_poly_sum :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) +
  (q 7) + (q 8) + (q 9) + (q 10) + (q 11) + (q 12) + (q 13) +
  (q 14) + (q 15) + (q 16) = 340 :=
by
  sorry

end cubic_poly_sum_l69_69395


namespace correct_choice_2point5_l69_69288

def set_M : Set ℝ := {x | -2 < x ∧ x < 3}

theorem correct_choice_2point5 : 2.5 ∈ set_M :=
by {
  -- sorry is added to close the proof for now
  sorry
}

end correct_choice_2point5_l69_69288


namespace deny_evenness_l69_69379

-- We need to define the natural numbers and their parity.
variables {a b c : ℕ}

-- Define what it means for a number to be odd and even.
def is_odd (n : ℕ) := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) := ∃ k, n = 2 * k

-- The Lean theorem statement translating the given problem.
theorem deny_evenness :
  (is_odd a ∧ is_odd b ∧ is_odd c) → ¬(is_even a ∨ is_even b ∨ is_even c) :=
by sorry

end deny_evenness_l69_69379


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69815

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69815


namespace necessary_but_not_sufficient_condition_l69_69226

theorem necessary_but_not_sufficient_condition (x : ℝ) : 
  (∃ x, x > 2 ∧ ¬ (x > 3)) ∧ 
  (∀ x, x > 3 → x > 2) := by sorry

end necessary_but_not_sufficient_condition_l69_69226


namespace sequence_general_term_l69_69429

theorem sequence_general_term {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 :=
by
  -- skip the proof with sorry
  sorry

end sequence_general_term_l69_69429


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69604

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69604


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69800

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69800


namespace total_seats_round_table_l69_69934

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69934


namespace greatest_divisor_of_four_consecutive_integers_l69_69835

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69835


namespace fixed_point_of_inverse_l69_69283

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 4

theorem fixed_point_of_inverse (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  f a (5) = 1 :=
by
  unfold f
  sorry

end fixed_point_of_inverse_l69_69283


namespace greatest_divisor_four_consecutive_integers_l69_69733

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69733


namespace greatest_divisor_of_four_consecutive_integers_l69_69660

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69660


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69720

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69720


namespace function_not_strictly_decreasing_l69_69285

theorem function_not_strictly_decreasing (b : ℝ)
  (h : ¬ ∀ x1 x2 : ℝ, x1 < x2 → (-x1^3 + b*x1^2 - (2*b + 3)*x1 + 2 - b > -x2^3 + b*x2^2 - (2*b + 3)*x2 + 2 - b)) : 
  b < -1 ∨ b > 3 :=
by
  sorry

end function_not_strictly_decreasing_l69_69285


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69639

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69639


namespace divisor_of_four_consecutive_integers_l69_69755

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69755


namespace sum_of_roots_of_quadratic_l69_69421

theorem sum_of_roots_of_quadratic :
  ∀ x1 x2 : ℝ, (∃ a b c, a = -1 ∧ b = 2 ∧ c = 4 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 + x2 = 2) :=
by
  sorry

end sum_of_roots_of_quadratic_l69_69421


namespace probability_of_pairing_long_with_short_l69_69450

theorem probability_of_pairing_long_with_short :
  let S := finset.range 5, L := finset.range 5
  let total_permutations := (10.factorial : ℚ)
  let distinct_permutations := total_permutations / ((5.factorial : ℚ) * (5.factorial : ℚ))
  let acceptable_pairings := 2^5
  let P := acceptable_pairings / distinct_permutations
  P = 8 / 63 := by
  sorry

end probability_of_pairing_long_with_short_l69_69450


namespace income_ratio_l69_69347

theorem income_ratio (I1 I2 E1 E2 : ℝ) (h1 : I1 = 5500) (h2 : E1 = I1 - 2200) (h3 : E2 = I2 - 2200) (h4 : E1 / E2 = 3 / 2) : I1 / I2 = 5 / 4 := by
  -- This is where the proof would go, but it's omitted for brevity.
  sorry

end income_ratio_l69_69347


namespace fraction_eq_l69_69581

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l69_69581


namespace probability_different_colors_l69_69547

def total_chips := 7 + 5 + 4

def probability_blue_draw : ℚ := 7 / total_chips
def probability_red_draw : ℚ := 5 / total_chips
def probability_yellow_draw : ℚ := 4 / total_chips
def probability_different_color (color1_prob color2_prob : ℚ) : ℚ := color1_prob * (1 - color2_prob)

theorem probability_different_colors :
  (probability_blue_draw * probability_different_color 7 (7 / total_chips)) +
  (probability_red_draw * probability_different_color 5 (5 / total_chips)) +
  (probability_yellow_draw * probability_different_color 4 (4 / total_chips)) 
  = 83 / 128 := 
by 
  sorry

end probability_different_colors_l69_69547


namespace chickens_do_not_lay_eggs_l69_69326

theorem chickens_do_not_lay_eggs (total_chickens : ℕ) 
  (roosters : ℕ) (hens : ℕ) (hens_lay_eggs : ℕ) (hens_do_not_lay_eggs : ℕ) 
  (chickens_do_not_lay_eggs : ℕ) :
  total_chickens = 80 →
  roosters = total_chickens / 4 →
  hens = total_chickens - roosters →
  hens_lay_eggs = 3 * hens / 4 →
  hens_do_not_lay_eggs = hens - hens_lay_eggs →
  chickens_do_not_lay_eggs = hens_do_not_lay_eggs + roosters →
  chickens_do_not_lay_eggs = 35 :=
by
  intros h0 h1 h2 h3 h4 h5
  sorry

end chickens_do_not_lay_eggs_l69_69326


namespace treaty_of_versailles_signed_on_wednesday_l69_69590

/-- The Treaty of Versailles was signed on June 28, 1919, marking the official end of World War I.
   The war had begun 1,566 days earlier, on a Wednesday, July 28, 1914.
   Determine the day of the week on which the treaty was signed. -/
theorem treaty_of_versailles_signed_on_wednesday :
  let days_in_week := 7
  let num_days := 1566
  let starting_day := 3 -- Representing Wednesday as an integer (e.g., Sun = 0, Mon = 1, ..., Wed = 3, ...)
  (starting_day + num_days) % days_in_week = 3 :=
by
  let days_in_week := 7
  let num_days := 1566
  let starting_day := 3
  have mod_eq_4 : num_days % days_in_week = 4 := by norm_num
  rw [mod_eq_4, nat.add_mod]; norm_num
  sorry

end treaty_of_versailles_signed_on_wednesday_l69_69590


namespace intersection_is_N_l69_69120

-- Define the sets M and N as given in the problem
def M := {x : ℝ | x > 0}
def N := {x : ℝ | Real.log x > 0}

-- State the theorem for the intersection of M and N
theorem intersection_is_N : (M ∩ N) = N := 
  by 
    sorry

end intersection_is_N_l69_69120


namespace largest_perfect_square_factor_of_1800_l69_69041

theorem largest_perfect_square_factor_of_1800 :
  ∃ k, k ∣ 1800 ∧ is_square k ∧ ∀ m, (m ∣ 1800 ∧ is_square m) → m ≤ k :=
begin
  use 900,
  split,
  { -- 900 divides 1800
    rw dvd_iff_mod_eq_zero,
    norm_num,
  },
  split,
  { -- 900 is a perfect square
    exact is_square.mk' 30, -- since 30 * 30 = 900
  },
  { -- 900 is the largest perfect square factor of 1800
    intros m hm,
    rcases hm with ⟨hdvd, hsquare⟩,
    rw dvd_iff_mod_eq_zero at hdvd,
    cases hsquare with n hn,
    rw hn at hdvd ⊢,
    have h : n^2 ∣ 2^3 * 3^2 * 5^2 := by norm_num at hdvd ⊢; sorry,
    sorry
  }
end

end largest_perfect_square_factor_of_1800_l69_69041


namespace smallest_non_factor_product_of_48_l69_69020

theorem smallest_non_factor_product_of_48 :
  ∃ (x y : ℕ), x ≠ y ∧ x * y ≤ 48 ∧ (x ∣ 48) ∧ (y ∣ 48) ∧ ¬ (x * y ∣ 48) ∧ x * y = 18 :=
by
  sorry

end smallest_non_factor_product_of_48_l69_69020


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69630

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69630


namespace abs_inequality_solution_l69_69135

theorem abs_inequality_solution {a : ℝ} (h : ∀ x : ℝ, |2 - x| + |x + 1| ≥ a) : a ≤ 3 :=
sorry

end abs_inequality_solution_l69_69135


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69705

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69705


namespace whale_consumption_l69_69503

-- Define the conditions
def first_hour_consumption (x : ℕ) := x
def second_hour_consumption (x : ℕ) := x + 3
def third_hour_consumption (x : ℕ) := x + 6
def fourth_hour_consumption (x : ℕ) := x + 9
def fifth_hour_consumption (x : ℕ) := x + 12
def sixth_hour_consumption (x : ℕ) := x + 15
def seventh_hour_consumption (x : ℕ) := x + 18
def eighth_hour_consumption (x : ℕ) := x + 21
def ninth_hour_consumption (x : ℕ) := x + 24

def total_consumed (x : ℕ) := 
  first_hour_consumption x + 
  second_hour_consumption x + 
  third_hour_consumption x + 
  fourth_hour_consumption x + 
  fifth_hour_consumption x + 
  sixth_hour_consumption x + 
  seventh_hour_consumption x + 
  eighth_hour_consumption x + 
  ninth_hour_consumption x

-- Prove that the total sum consumed equals 540
theorem whale_consumption : ∃ x : ℕ, total_consumed x = 540 ∧ sixth_hour_consumption x = 63 :=
by
  sorry

end whale_consumption_l69_69503


namespace smallest_nonnegative_a_l69_69567

open Real

theorem smallest_nonnegative_a (a b : ℝ) (h_b : b = π / 4)
(sin_eq : ∀ (x : ℤ), sin (a * x + b) = sin (17 * x)) : 
a = 17 - π / 4 := by 
  sorry

end smallest_nonnegative_a_l69_69567


namespace problem_1_l69_69871

theorem problem_1 : (-(5 / 8) / (14 / 3) * (-(16 / 5)) / (-(6 / 7))) = -1 / 2 :=
  sorry

end problem_1_l69_69871


namespace greatest_divisor_of_four_consecutive_integers_l69_69837

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69837


namespace min_value_of_z_l69_69965

theorem min_value_of_z (x y : ℝ) (h : y^2 = 4 * x) : 
  ∃ (z : ℝ), z = 3 ∧ ∀ (x' : ℝ) (hx' : x' ≥ 0), ∃ (y' : ℝ), y'^2 = 4 * x' → z ≤ (1/2) * y'^2 + x'^2 + 3 :=
by sorry

end min_value_of_z_l69_69965


namespace greatest_divisor_of_consecutive_product_l69_69860

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69860


namespace problem_solution_l69_69198

theorem problem_solution (a b : ℤ) (h1 : 6 * b + 4 * a = -50) (h2 : a * b = -84) : a + 2 * b = -17 := 
  sorry

end problem_solution_l69_69198


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69623

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69623


namespace four_digit_number_exists_l69_69224

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), 
  B = 3 * A ∧ 
  C = A + B ∧ 
  D = 3 * B ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ 
  1000 * A + 100 * B + 10 * C + D = 1349 :=
by {
  sorry 
}

end four_digit_number_exists_l69_69224


namespace geometric_progression_fraction_l69_69438

theorem geometric_progression_fraction (a₁ a₂ a₃ a₄ : ℝ) (h1 : a₂ = 2 * a₁) (h2 : a₃ = 2 * a₂) (h3 : a₄ = 2 * a₃) : 
  (2 * a₁ + a₂) / (2 * a₃ + a₄) = 1 / 4 := 
by 
  sorry

end geometric_progression_fraction_l69_69438


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69775

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69775


namespace proof_problem_l69_69399

variable (a b c d x : ℤ)

-- Conditions
def are_opposite (a b : ℤ) : Prop := a + b = 0
def are_reciprocals (c d : ℤ) : Prop := c * d = 1
def largest_negative_integer (x : ℤ) : Prop := x = -1

theorem proof_problem 
  (h1 : are_opposite a b) 
  (h2 : are_reciprocals c d) 
  (h3 : largest_negative_integer x) :
  x^2 - (a + b - c * d)^(2012 : ℕ) + (-c * d)^(2011 : ℕ) = -1 :=
by
  sorry

end proof_problem_l69_69399


namespace intersection_eq_l69_69109

noncomputable def A : Set ℝ := { x | x < 2 }
noncomputable def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_eq : A ∩ B = {-1, 0, 1} :=
by
  sorry

end intersection_eq_l69_69109


namespace area_of_rhombus_enclosed_by_equation_l69_69028

-- Given the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- Define the main theorem to be proven
theorem area_of_rhombus_enclosed_by_equation : 
  (∃ x y : ℝ, equation x y) → ∃ area : ℝ, area = 384 :=
by
  sorry

end area_of_rhombus_enclosed_by_equation_l69_69028


namespace b_share_in_profit_l69_69887

theorem b_share_in_profit (A B C : ℝ) (p : ℝ := 4400) (x : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = (2 / 3) * C)
  (h3 : C = x) :
  B / (A + B + C) * p = 800 :=
by
  sorry

end b_share_in_profit_l69_69887


namespace divisor_of_product_of_four_consecutive_integers_l69_69652

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69652


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69696

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69696


namespace original_average_l69_69003

theorem original_average (A : ℝ) (h : 5 * A = 130) : A = 26 :=
by
  have h1 : 5 * A / 5 = 130 / 5 := by sorry
  sorry

end original_average_l69_69003


namespace domain_of_tan_2x_plus_pi_over_3_l69_69593

noncomputable def domain_tan_transformed : Set ℝ :=
  {x : ℝ | ∀ (k : ℤ), x ≠ k * (Real.pi / 2) + (Real.pi / 12)}

theorem domain_of_tan_2x_plus_pi_over_3 :
  (∀ x : ℝ, x ∉ domain_tan_transformed ↔ ∃ (k : ℤ), x = k * (Real.pi / 2) + (Real.pi / 12)) :=
sorry

end domain_of_tan_2x_plus_pi_over_3_l69_69593


namespace greatest_divisor_of_four_consecutive_integers_l69_69825

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69825


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69778

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69778


namespace total_seats_round_table_l69_69937

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69937


namespace hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l69_69364

theorem hyperbola_shares_focus_with_eccentricity 
  (a1 b1 : ℝ) (h1 : a1 = 3 ∧ b1 = 2)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 5) / 2)
  (c : ℝ) (h_focus : c = Real.sqrt (a1^2 - b1^2)) :
  (∃ a b : ℝ, a^2 - b^2 = c^2 ∧ c/a = e ∧ a = 2 ∧ b = 1) :=
sorry

theorem length_of_chord_AB 
  (a b : ℝ) (h_ellipse : a^2 = 4 ∧ b^2 = 1)
  (c : ℝ) (h_focus : c = Real.sqrt (a^2 - b^2))
  (f : ℝ) (h_f : f = Real.sqrt 3)
  (line_eq : ℝ -> ℝ) (h_line_eq : ∀ x, line_eq x = x - f) :
  (∃ x1 x2 : ℝ, 
    x1 + x2 = (8 * Real.sqrt 3) / 5 ∧
    x1 * x2 = 8 / 5 ∧
    Real.sqrt ((x1 - x2)^2 + (line_eq x1 - line_eq x2)^2) = 8 / 5) :=
sorry

end hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l69_69364


namespace four_consecutive_product_divisible_by_12_l69_69795

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69795


namespace greatest_divisor_four_consecutive_integers_l69_69730

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69730


namespace divisor_of_product_of_four_consecutive_integers_l69_69647

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69647


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69637

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69637


namespace coloring_impossible_l69_69485

-- Define vertices for the outer pentagon and inner star
inductive Vertex
| A | B | C | D | E | A' | B' | C' | D' | E'

open Vertex

-- Define segments in the figure
def Segments : List (Vertex × Vertex) :=
  [(A, B), (B, C), (C, D), (D, E), (E, A),
   (A, A'), (B, B'), (C, C'), (D, D'), (E, E'),
   (A', C), (C, E'), (E, B'), (B, D'), (D, A')]

-- Color type
inductive Color
| Red | Green | Blue

open Color

-- Condition for coloring: no two segments of the same color share a common endpoint
def distinct_color (c : Vertex → Color) : Prop :=
  ∀ (v1 v2 v3 : Vertex) (h1 : (v1, v2) ∈ Segments) (h2 : (v2, v3) ∈ Segments),
  c v1 ≠ c v2 ∧ c v2 ≠ c v3 ∧ c v1 ≠ c v3

-- Statement of the proof problem
theorem coloring_impossible : ¬ ∃ (c : Vertex → Color), distinct_color c := 
by 
  sorry

end coloring_impossible_l69_69485


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69807

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69807


namespace area_enclosed_abs_eq_96_l69_69031

theorem area_enclosed_abs_eq_96 :
  (∃ (S : Set (ℝ × ℝ)), ∀ (x y : ℝ), (x, y) ∈ S ↔ |x| + |3 * y| = 12) →
  (let area := 96 in true) :=
begin
  sorry
end

end area_enclosed_abs_eq_96_l69_69031


namespace enclosed_area_abs_x_abs_3y_eq_12_l69_69023

theorem enclosed_area_abs_x_abs_3y_eq_12 : 
  let f (x y : ℝ) := |x| + |3 * y|
  ∃ (A : ℝ), ∀ (x y : ℝ), f x y = 12 → A = 96 := 
sorry

end enclosed_area_abs_x_abs_3y_eq_12_l69_69023


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69618

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69618


namespace find_speed_l69_69239

theorem find_speed (v d : ℝ) (h1 : d > 0) (h2 : 1.10 * v > 0) (h3 : 84 = 2 * d / (d / v + d / (1.10 * v))) : v = 80.18 := 
sorry

end find_speed_l69_69239


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69612

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69612


namespace odd_function_extended_l69_69112

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≥ 0 then 
    x * Real.log (x + 1)
  else 
    x * Real.log (-x + 1)

theorem odd_function_extended : (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, x ≥ 0 → f x = x * Real.log (x + 1)) →
  (∀ x : ℝ, x < 0 → f x = x * Real.log (-x + 1)) :=
by
  intros h_odd h_def_neg
  sorry

end odd_function_extended_l69_69112


namespace total_seats_at_round_table_l69_69908

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69908


namespace no_solution_in_positive_rationals_l69_69176

theorem no_solution_in_positive_rationals (n : ℕ) (hn : n > 0) (x y : ℚ) (hx : x > 0) (hy : y > 0) :
  x + y + (1 / x) + (1 / y) ≠ 3 * n :=
sorry

end no_solution_in_positive_rationals_l69_69176


namespace four_consecutive_product_divisible_by_12_l69_69786

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69786


namespace divisor_of_four_consecutive_integers_l69_69757

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69757


namespace area_of_triangle_given_conditions_l69_69146

noncomputable def area_triangle_ABC (a b B : ℝ) : ℝ :=
  0.5 * a * b * Real.sin B

theorem area_of_triangle_given_conditions :
  area_triangle_ABC 2 (Real.sqrt 3) (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end area_of_triangle_given_conditions_l69_69146


namespace hcf_of_two_numbers_l69_69869
-- Importing the entire Mathlib library for mathematical functions

-- Define the two numbers and the conditions given in the problem
variables (x y : ℕ)

-- State the conditions as hypotheses
def conditions (h1 : x + y = 45) (h2 : Nat.lcm x y = 120) (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Prop :=
  True

-- State the theorem we want to prove
theorem hcf_of_two_numbers (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : Nat.lcm x y = 120)
  (h3 : (1 / (x : ℚ)) + (1 / (y : ℚ)) = 11 / 120) : Nat.gcd x y = 1 :=
  sorry

end hcf_of_two_numbers_l69_69869


namespace not_necessarily_divisible_by_28_l69_69587

theorem not_necessarily_divisible_by_28 (k : ℤ) (h : 7 ∣ (k * (k + 1) * (k + 2))) : ¬ (28 ∣ (k * (k + 1) * (k + 2))) :=
sorry

end not_necessarily_divisible_by_28_l69_69587


namespace cylinder_height_to_radius_ratio_l69_69954

theorem cylinder_height_to_radius_ratio (V r h : ℝ) (hV : V = π * r^2 * h) (hS : sorry) :
  h / r = 2 :=
sorry

end cylinder_height_to_radius_ratio_l69_69954


namespace kernels_popped_in_first_bag_l69_69182

theorem kernels_popped_in_first_bag :
  ∀ (x : ℕ), 
    (total_kernels : ℕ := 75 + 50 + 100) →
    (total_popped : ℕ := x + 42 + 82) →
    (average_percentage_popped : ℚ := 82) →
    ((total_popped : ℚ) / total_kernels) * 100 = average_percentage_popped →
    x = 61 :=
by
  sorry

end kernels_popped_in_first_bag_l69_69182


namespace graph_of_equation_is_two_intersecting_lines_l69_69218

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x - 2 * y)^2 = x^2 + y^2 ↔ (y = 0 ∨ y = 4 / 3 * x) :=
by
  sorry

end graph_of_equation_is_two_intersecting_lines_l69_69218


namespace arithmetic_sequence_fifth_term_l69_69350

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 3) 
  (h2 : a + 11 * d = 9) : 
  a + 4 * d = -12 :=
by
  sorry

end arithmetic_sequence_fifth_term_l69_69350


namespace point_M_on_y_axis_l69_69142

theorem point_M_on_y_axis (t : ℝ) (h : t - 3 = 0) : (t-3, 5-t) = (0, 2) :=
by
  sorry

end point_M_on_y_axis_l69_69142


namespace ticket_representation_l69_69542

-- Define a structure for representing a movie ticket
structure Ticket where
  rows : Nat
  seats : Nat

-- Define the specific instance of representing 7 rows and 5 seats
def ticket_7_5 : Ticket := ⟨7, 5⟩

-- The theorem stating our problem: the representation of 7 rows and 5 seats is (7,5)
theorem ticket_representation : ticket_7_5 = ⟨7, 5⟩ :=
  by
    -- Proof goes here (omitted as per instructions)
    sorry

end ticket_representation_l69_69542


namespace f_eq_g_iff_l69_69568

noncomputable def f (m n x : ℝ) := m * x^2 + n * x
noncomputable def g (p q x : ℝ) := p * x + q

theorem f_eq_g_iff (m n p q : ℝ) :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n := by
  sorry

end f_eq_g_iff_l69_69568


namespace num_ways_to_use_100_yuan_l69_69475

noncomputable def x : ℕ → ℝ
| 0       => 0
| 1       => 1
| 2       => 3
| (n + 3) => x (n + 2) + 2 * x (n + 1)

theorem num_ways_to_use_100_yuan :
  x 100 = (1 / 3) * (2 ^ 101 + 1) :=
sorry

end num_ways_to_use_100_yuan_l69_69475


namespace necessary_and_sufficient_for_Sn_lt_an_l69_69462

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1)) / 2

theorem necessary_and_sufficient_for_Sn_lt_an
  (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h_arith_seq : arithmetic_seq a d)
  (h_d_neg : d < 0)
  (m n : ℕ)
  (h_pos_m : m ≥ 3)
  (h_am_eq_Sm : a m = S m) :
  n > m ↔ S n < a n := sorry

end necessary_and_sufficient_for_Sn_lt_an_l69_69462


namespace prove_option_d_l69_69274

-- Definitions of conditions
variables (a b : ℝ)
variable (h_nonzero : a ≠ 0 ∧ b ≠ 0)
variable (h_lt : a < b)

-- The theorem to be proved
theorem prove_option_d : a^3 < b^3 :=
sorry

end prove_option_d_l69_69274


namespace worth_of_each_gold_bar_l69_69582

theorem worth_of_each_gold_bar
  (rows : ℕ) (gold_bars_per_row : ℕ) (total_worth : ℕ)
  (h1 : rows = 4) (h2 : gold_bars_per_row = 20) (h3 : total_worth = 1600000)
  (total_gold_bars : ℕ) (h4 : total_gold_bars = rows * gold_bars_per_row) :
  total_worth / total_gold_bars = 20000 :=
by sorry

end worth_of_each_gold_bar_l69_69582


namespace greatest_divisor_of_four_consecutive_integers_l69_69834

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69834


namespace greatest_divisor_of_four_consecutive_integers_l69_69826

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69826


namespace false_proposition_is_C_l69_69372

theorem false_proposition_is_C : ¬ (∀ x : ℝ, x^3 > 0) :=
sorry

end false_proposition_is_C_l69_69372


namespace johnny_years_ago_l69_69356

theorem johnny_years_ago 
  (J : ℕ) (hJ : J = 8) (X : ℕ) 
  (h : J + 2 = 2 * (J - X)) : 
  X = 3 := by
  sorry

end johnny_years_ago_l69_69356


namespace determine_k_l69_69382

variables (x y z k : ℝ)

theorem determine_k (h1 : (5 / (x - z)) = (k / (y + z))) 
                    (h2 : (k / (y + z)) = (12 / (x + y))) 
                    (h3 : y + z = 2 * x) : 
                    k = 17 := 
by 
  sorry

end determine_k_l69_69382


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69617

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69617


namespace greatest_divisor_of_consecutive_product_l69_69859

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69859


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69765

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69765


namespace circle_region_count_l69_69159

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l69_69159


namespace event_relationship_l69_69187

def die_events : set ℕ := {1, 2, 3, 4, 5, 6}
def event_A : set ℕ := {n | n ∈ die_events ∧ n % 2 = 1}
def event_B : set ℕ := {n | n = 3 ∨ n = 4}

lemma not_mutually_exclusive : ∃ n ∈ die_events, n ∈ event_A ∧ n ∈ event_B :=
by {
  use 3,
  split,
  { unfold die_events, finish, },
  { unfold event_A, unfold event_B, finish, }
}

lemma independent_events : independent_events event_A event_B :=
by {
  sorry
}

theorem event_relationship : ¬ (disjoint event_A event_B) ∧ independent_events event_A event_B :=
by {
  split,
  { exact not_mutually_exclusive, },
  { exact independent_events, }
}


end event_relationship_l69_69187


namespace exists_five_points_with_unique_distances_and_same_perimeter_l69_69993

theorem exists_five_points_with_unique_distances_and_same_perimeter :
  ∃ (A B C D E : ℝ³),
  (∀ (P Q : ℝ³), P ≠ Q → dist P Q ≠ dist (P + 1) Q) ∧
  (∀ (P Q R S T : ℝ³), dist P Q + dist Q R + dist R S + dist S T + dist T P = dist A B + dist B C + dist C D + dist D E + dist E A) :=
sorry

end exists_five_points_with_unique_distances_and_same_perimeter_l69_69993


namespace ordered_pairs_m_n_l69_69123

theorem ordered_pairs_m_n :
  ∃ (s : Finset (ℕ × ℕ)), 
  (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ (p.1 ^ 2 - p.2 ^ 2 = 72)) ∧ s.card = 3 :=
by
  sorry

end ordered_pairs_m_n_l69_69123


namespace divisor_of_four_consecutive_integers_l69_69748

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69748


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69616

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69616


namespace fraction_of_red_knights_magical_l69_69301

variable {knights : ℕ}
variable {red_knights : ℕ}
variable {blue_knights : ℕ}
variable {magical_knights : ℕ}
variable {magical_red_knights : ℕ}
variable {magical_blue_knights : ℕ}

axiom total_knights : knights > 0
axiom red_knights_fraction : red_knights = (3 * knights) / 8
axiom blue_knights_fraction : blue_knights = (5 * knights) / 8
axiom magical_knights_fraction : magical_knights = knights / 4
axiom magical_fraction_relation : 3 * magical_blue_knights = magical_red_knights

theorem fraction_of_red_knights_magical :
  (magical_red_knights : ℚ) / red_knights = 3 / 7 :=
by
  sorry

end fraction_of_red_knights_magical_l69_69301


namespace probability_correct_l69_69571

structure SockDrawSetup where
  total_socks : ℕ
  color_pairs : ℕ
  socks_per_color : ℕ
  draw_size : ℕ

noncomputable def probability_one_pair (S : SockDrawSetup) : ℚ :=
  let total_combinations := Nat.choose S.total_socks S.draw_size
  let favorable_combinations := (Nat.choose S.color_pairs 3) * (Nat.choose 3 1) * 2 * 2
  favorable_combinations / total_combinations

theorem probability_correct (S : SockDrawSetup) (h1 : S.total_socks = 12) (h2 : S.color_pairs = 6) (h3 : S.socks_per_color = 2) (h4 : S.draw_size = 6) :
  probability_one_pair S = 20 / 77 :=
by
  apply sorry

end probability_correct_l69_69571


namespace sample_size_l69_69107

theorem sample_size (F n : ℕ) (FR : ℚ) (h1: F = 36) (h2: FR = 1/4) (h3: FR = F / n) : n = 144 :=
by 
  sorry

end sample_size_l69_69107


namespace square_root_calc_l69_69113

theorem square_root_calc (x : ℤ) (hx : x^2 = 1764) : (x + 2) * (x - 2) = 1760 := by
  sorry

end square_root_calc_l69_69113


namespace max_students_l69_69012

theorem max_students (pens pencils : ℕ) (h_pens : pens = 1340) (h_pencils : pencils = 1280) : Nat.gcd pens pencils = 20 := by
    sorry

end max_students_l69_69012


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69718

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69718


namespace parallelogram_area_l69_69072

noncomputable def area_parallelogram (b s θ : ℝ) : ℝ := b * (s * Real.sin θ)

theorem parallelogram_area : area_parallelogram 20 10 (Real.pi / 6) = 100 := by
  sorry

end parallelogram_area_l69_69072


namespace geometric_sequence_first_term_l69_69992

theorem geometric_sequence_first_term (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 * a 2 * a 3 = 27) (h3 : a 6 = 27) : a 0 = 1 :=
by
  sorry

end geometric_sequence_first_term_l69_69992


namespace probability_heads_at_least_twice_in_five_tosses_l69_69069

noncomputable def fair_coin := Prob.one_half

def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_heads_at_least_twice_in_five_tosses :
  let p := fair_coin in
  1 - (binomialProbability 5 0 p) - (binomialProbability 5 1 p) = 0.8125 := by
  sorry

end probability_heads_at_least_twice_in_five_tosses_l69_69069


namespace socks_ratio_l69_69572

theorem socks_ratio 
  (g : ℕ) -- number of pairs of green socks
  (y : ℝ) -- price per pair of green socks
  (h1 : y > 0) -- price per pair of green socks is positive
  (h2 : 3 * g * y + 3 * y = 1.2 * (9 * y + g * y)) -- swapping resulted in a 20% increase in the bill
  : 3 / g = 3 / 4 :=
by sorry

end socks_ratio_l69_69572


namespace total_seats_round_table_l69_69940

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69940


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69707

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69707


namespace four_consecutive_integers_divisible_by_12_l69_69691

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69691


namespace divisible_by_133_l69_69445

theorem divisible_by_133 (n : ℕ) : (11^(n + 2) + 12^(2*n + 1)) % 133 = 0 :=
by
  sorry

end divisible_by_133_l69_69445


namespace raja_monthly_income_l69_69482

noncomputable def monthly_income (household_percentage clothes_percentage medicines_percentage savings : ℝ) : ℝ :=
  let spending_percentage := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage := 1 - spending_percentage
  savings / savings_percentage

theorem raja_monthly_income :
  monthly_income 0.35 0.20 0.05 15000 = 37500 :=
by
  sorry

end raja_monthly_income_l69_69482


namespace cats_remained_on_island_l69_69081

theorem cats_remained_on_island : 
  ∀ (n m1 : ℕ), 
  n = 1800 → 
  m1 = 600 → 
  (n - m1) / 2 = 600 → 
  (n - m1) - ((n - m1) / 2) = 600 :=
by sorry

end cats_remained_on_island_l69_69081


namespace minimum_deposits_needed_l69_69241

noncomputable def annual_salary_expense : ℝ := 100000
noncomputable def annual_fixed_expense : ℝ := 170000
noncomputable def interest_rate_paid : ℝ := 0.0225
noncomputable def interest_rate_earned : ℝ := 0.0405

theorem minimum_deposits_needed :
  ∃ (x : ℝ), 
    (interest_rate_earned * x = annual_salary_expense + annual_fixed_expense + interest_rate_paid * x) →
    x = 1500 :=
by
  sorry

end minimum_deposits_needed_l69_69241


namespace largest_perfect_square_factor_1800_l69_69039

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  if n = 1800 then 900 else sorry

theorem largest_perfect_square_factor_1800 : 
  largest_perfect_square_factor 1800 = 900 :=
by
  -- Proof is not needed, so we use sorry
  sorry

end largest_perfect_square_factor_1800_l69_69039


namespace num_female_managers_l69_69868

-- Definitions based on the conditions
def total_employees : ℕ := 250
def female_employees : ℕ := 90
def total_managers : ℕ := 40
def male_associates : ℕ := 160

-- Proof statement that computes the number of female managers
theorem num_female_managers : 
  (total_managers - (total_employees - female_employees - male_associates)) = 40 := 
by 
  sorry

end num_female_managers_l69_69868


namespace avg_ac_l69_69196

-- Define the ages of a, b, and c as variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def avg_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 26
def age_b (B : ℕ) : Prop := B = 20

-- State the theorem to prove
theorem avg_ac {A B C : ℕ} (h1 : avg_abc A B C) (h2 : age_b B) : (A + C) / 2 = 29 := 
by sorry

end avg_ac_l69_69196


namespace total_seats_round_table_l69_69903

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69903


namespace greatest_divisor_four_consecutive_l69_69745

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69745


namespace product_of_consecutive_integers_l69_69675

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69675


namespace total_seats_l69_69932

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69932


namespace total_seats_at_round_table_l69_69911

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69911


namespace four_consecutive_product_divisible_by_12_l69_69793

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69793


namespace jana_distance_travel_in_20_minutes_l69_69430

theorem jana_distance_travel_in_20_minutes :
  ∀ (usual_pace half_pace double_pace : ℚ)
    (first_15_minutes_distance second_5_minutes_distance total_distance : ℚ),
  usual_pace = 1 / 30 →
  half_pace = usual_pace / 2 →
  double_pace = usual_pace * 2 →
  first_15_minutes_distance = 15 * half_pace →
  second_5_minutes_distance = 5 * double_pace →
  total_distance = first_15_minutes_distance + second_5_minutes_distance →
  total_distance = 7 / 12 := 
by
  intros
  sorry

end jana_distance_travel_in_20_minutes_l69_69430


namespace king_arthur_round_table_seats_l69_69894

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69894


namespace correct_factorization_A_l69_69219

theorem correct_factorization_A (x : ℝ) : x^2 - 4 * x + 4 = (x - 2)^2 :=
by sorry

end correct_factorization_A_l69_69219


namespace inscribed_circle_radius_inequality_l69_69489

open Real

variables (ABC ABD BDC : Type) -- Representing the triangles

noncomputable def r (ABC : Type) : ℝ := sorry -- radius of the inscribed circle in ABC
noncomputable def r1 (ABD : Type) : ℝ := sorry -- radius of the inscribed circle in ABD
noncomputable def r2 (BDC : Type) : ℝ := sorry -- radius of the inscribed circle in BDC

noncomputable def p (ABC : Type) : ℝ := sorry -- semiperimeter of ABC
noncomputable def p1 (ABD : Type) : ℝ := sorry -- semiperimeter of ABD
noncomputable def p2 (BDC : Type) : ℝ := sorry -- semiperimeter of BDC

noncomputable def S (ABC : Type) : ℝ := sorry -- area of ABC
noncomputable def S1 (ABD : Type) : ℝ := sorry -- area of ABD
noncomputable def S2 (BDC : Type) : ℝ := sorry -- area of BDC

lemma triangle_area_sum (ABC ABD BDC : Type) :
  S ABC = S1 ABD + S2 BDC := sorry

lemma semiperimeter_area_relation (ABC ABD BDC : Type) :
  S ABC = p ABC * r ABC ∧
  S1 ABD = p1 ABD * r1 ABD ∧
  S2 BDC = p2 BDC * r2 BDC := sorry

theorem inscribed_circle_radius_inequality (ABC ABD BDC : Type) :
  r1 ABD + r2 BDC > r ABC := sorry

end inscribed_circle_radius_inequality_l69_69489


namespace adam_apples_l69_69888

theorem adam_apples (x : ℕ) 
  (h1 : 15 + 75 * x = 240) : x = 3 :=
sorry

end adam_apples_l69_69888


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69633

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69633


namespace distinct_rational_numbers_count_l69_69511

theorem distinct_rational_numbers_count :
  ∃ N : ℕ, 
    (N = 49) ∧
    ∀ (k : ℚ), |k| < 50 →
      (∃ x : ℤ, x^2 - k * x + 18 = 0) →
        ∃ m: ℤ, k = 2 * m ∧ |m| < 25 :=
sorry

end distinct_rational_numbers_count_l69_69511


namespace cost_of_paintbrush_l69_69083

noncomputable def cost_of_paints : ℝ := 4.35
noncomputable def cost_of_easel : ℝ := 12.65
noncomputable def amount_already_has : ℝ := 6.50
noncomputable def additional_amount_needed : ℝ := 12.00

-- Let's define the total cost needed and the total costs of items
noncomputable def total_cost_of_paints_and_easel : ℝ := cost_of_paints + cost_of_easel
noncomputable def total_amount_needed : ℝ := amount_already_has + additional_amount_needed

-- And now we can state our theorem that needs to be proved.
theorem cost_of_paintbrush : total_amount_needed - total_cost_of_paints_and_easel = 1.50 :=
by
  sorry

end cost_of_paintbrush_l69_69083


namespace proof_problem_l69_69280

theorem proof_problem 
  (a1 a2 b2 : ℚ)
  (ha1 : a1 = -9 + (8/3))
  (ha2 : a2 = -9 + 2 * (8/3))
  (hb2 : b2 = -3) :
  b2 * (a1 + a2) = 30 :=
by
  sorry

end proof_problem_l69_69280


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69772

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69772


namespace div_product_four_consecutive_integers_l69_69848

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69848


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69814

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69814


namespace max_integer_solutions_l69_69522

def quad_func (x : ℝ) : ℝ := x^2 - 6 * x + 1

theorem max_integer_solutions (p : ℝ → ℝ) : 
  (p = quad_func) →
  (∃ n1 n2 n3 n4 : ℤ, 
    ((p n1 = p (n1 ^ 2)) ∧ (p n2 = p (n2 ^ 2)) ∧ 
    (p n3 = p (n3 ^ 2)) ∧ (p n4 = p (n4 ^ 2))) ∧ 
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ 
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ 
    n3 ≠ n4) :=
by
  sorry

end max_integer_solutions_l69_69522


namespace anthony_more_than_mabel_l69_69444

noncomputable def transactions := 
  let M := 90  -- Mabel's transactions
  let J := 82  -- Jade's transactions
  let C := J - 16  -- Cal's transactions
  let A := (3 / 2) * C  -- Anthony's transactions
  let P := ((A - M) / M) * 100 -- Percentage more transactions Anthony handled than Mabel
  P

theorem anthony_more_than_mabel : transactions = 10 := by
  sorry

end anthony_more_than_mabel_l69_69444


namespace total_number_of_seats_l69_69920

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69920


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69762

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69762


namespace four_consecutive_product_divisible_by_12_l69_69787

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69787


namespace probability_of_drawing_stamps_l69_69588

theorem probability_of_drawing_stamps : 
  let stamps := ["Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold"]
  in (2 / (list.permutations stamps).length) = (1 / 6) := by
  sorry

end probability_of_drawing_stamps_l69_69588


namespace smallest_four_digit_divisible_by_primes_l69_69961

theorem smallest_four_digit_divisible_by_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ n) ∧ n = 1050 :=
by
  sorry

end smallest_four_digit_divisible_by_primes_l69_69961


namespace james_speed_is_16_l69_69172

theorem james_speed_is_16
  (distance : ℝ)
  (time : ℝ)
  (distance_eq : distance = 80)
  (time_eq : time = 5) :
  (distance / time = 16) :=
by
  rw [distance_eq, time_eq]
  norm_num

end james_speed_is_16_l69_69172


namespace multiple_7_proposition_l69_69202

theorem multiple_7_proposition : (47 % 7 ≠ 0 ∨ 49 % 7 = 0) → True :=
by
  intros h
  sorry

end multiple_7_proposition_l69_69202


namespace value_of_coins_l69_69546

theorem value_of_coins (m : ℕ) : 25 * 25 + 15 * 10 = m * 25 + 40 * 10 ↔ m = 15 :=
by
sorry

end value_of_coins_l69_69546


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69704

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69704


namespace king_arthur_round_table_seats_l69_69892

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69892


namespace total_visitors_l69_69599

noncomputable def visitors_questionnaire (V E U : ℕ) : Prop :=
  (130 ≠ E ∧ E ≠ U) ∧ 
  (E = U) ∧ 
  (3 * V = 4 * E) ∧ 
  (V = 130 + 3 / 4 * V)

theorem total_visitors (V : ℕ) : visitors_questionnaire V V V → V = 520 :=
by sorry

end total_visitors_l69_69599


namespace proof_star_ast_l69_69514

noncomputable def star (a b : ℕ) : ℕ := sorry  -- representing binary operation for star
noncomputable def ast (a b : ℕ) : ℕ := sorry  -- representing binary operation for ast

theorem proof_star_ast :
  star 12 2 * ast 9 3 = 2 →
  (star 7 3 * ast 12 6) = 7 / 6 :=
by
  sorry

end proof_star_ast_l69_69514


namespace integer_values_of_x_for_equation_l69_69509

theorem integer_values_of_x_for_equation 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : a = b + c ∨ b = c + a ∨ c = b + a) : 
  ∃ x : ℤ, a * x + b = c :=
sorry

end integer_values_of_x_for_equation_l69_69509


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69818

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69818


namespace relationship_y1_y2_l69_69294

theorem relationship_y1_y2 (y1 y2 : ℝ) (m : ℝ) (h_m : m ≠ 0) 
  (hA : y1 = m * (-2) + 4) (hB : 3 = m * 1 + 4) (hC : y2 = m * 3 + 4) : y1 > y2 :=
by
  sorry

end relationship_y1_y2_l69_69294


namespace steps_from_center_to_square_l69_69082

-- Define the conditions and question in Lean 4
def steps_to_center := 354
def total_steps := 582

-- Prove that the steps from Rockefeller Center to Times Square is 228
theorem steps_from_center_to_square : (total_steps - steps_to_center) = 228 := by
  sorry

end steps_from_center_to_square_l69_69082


namespace min_sum_of_M_and_N_l69_69000

noncomputable def Alice (x : ℕ) : ℕ := 3 * x + 2
noncomputable def Bob (x : ℕ) : ℕ := 2 * x + 27

-- Define the result after 4 moves
noncomputable def Alice_4_moves (M : ℕ) : ℕ := Alice (Alice (Alice (Alice M)))
noncomputable def Bob_4_moves (N : ℕ) : ℕ := Bob (Bob (Bob (Bob N)))

theorem min_sum_of_M_and_N :
  ∃ (M N : ℕ), Alice_4_moves M = Bob_4_moves N ∧ M + N = 10 :=
sorry

end min_sum_of_M_and_N_l69_69000


namespace remainder_14_plus_x_mod_31_l69_69178

theorem remainder_14_plus_x_mod_31 (x : ℕ) (hx : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
sorry

end remainder_14_plus_x_mod_31_l69_69178


namespace kendra_more_buttons_l69_69556

theorem kendra_more_buttons {K M S : ℕ} (hM : M = 8) (hS : S = 22) (hHalfK : S = K / 2) :
  K - 5 * M = 4 :=
by
  sorry

end kendra_more_buttons_l69_69556


namespace range_of_a_l69_69989

theorem range_of_a (a : ℝ) :
  ¬ (∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l69_69989


namespace intersection_eq_l69_69570

def setM : Set ℝ := { x | x^2 - 2*x < 0 }
def setN : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_eq : setM ∩ setN = { x | 0 < x ∧ x ≤ 1 } := sorry

end intersection_eq_l69_69570


namespace trajectory_midpoint_l69_69517

-- Define the hyperbola equation
def hyperbola (x y : ℝ) := x^2 - (y^2 / 4) = 1

-- Define the condition that a line passes through the point (0, 1)
def line_through_fixed_point (k x y : ℝ) := y = k * x + 1

-- Define the theorem to prove the trajectory of the midpoint of the chord
theorem trajectory_midpoint (x y k : ℝ) (h : ∃ x y, hyperbola x y ∧ line_through_fixed_point k x y) : 
    4 * x^2 - y^2 + y = 0 := 
sorry

end trajectory_midpoint_l69_69517


namespace regions_formed_l69_69162

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l69_69162


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69706

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69706


namespace triangle_count_l69_69127

theorem triangle_count (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 6) (h₂ : 1 ≤ j ∧ j ≤ 6): 
  let points := { (x, y) | 1 ≤ x ∧ x ≤ 6 ∧ 1 ≤ y ∧ y ≤ 6 } in
  fintype.card { t : finset (ℕ × ℕ) // t.card = 3 ∧ ∃ a b c : ℕ × ℕ, a ∉ t ∧ b ∉ t ∧ c ∉ t ∧ abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) ≠ 0 } = 6800 := 
by
  sorry

end triangle_count_l69_69127


namespace quadratic_inequality_l69_69577

theorem quadratic_inequality (a b c : ℝ) (h : a^2 + a * b + a * c < 0) : b^2 > 4 * a * c := 
sorry

end quadratic_inequality_l69_69577


namespace average_of_remaining_two_numbers_l69_69339

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95) 
  (h_avg_ab : (a + b) / 2 = 3.8) 
  (h_avg_cd : (c + d) / 2 = 3.85) :
  ((e + f) / 2) = 4.2 := 
by 
  sorry

end average_of_remaining_two_numbers_l69_69339


namespace find_m_for_parallel_lines_l69_69410

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 3 * x - y + 2 = 0 → x + m * y - 3 = 0) →
  m = -1 / 3 := sorry

end find_m_for_parallel_lines_l69_69410


namespace solve_problem_l69_69380

-- Conditions from the problem
def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (n p : ℕ) : Prop := 
  (p > 1) ∧ is_prime p ∧ (n > 0) ∧ (n ≤ 2 * p)

-- Main proof statement
theorem solve_problem (n p : ℕ) (h1 : satisfies_conditions n p)
    (h2 : (p - 1) ^ n + 1 ∣ n ^ (p - 1)) :
    (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
sorry

end solve_problem_l69_69380


namespace min_value_S_max_value_m_l69_69976

noncomputable def S (x : ℝ) : ℝ := abs (x - 2) + abs (x - 4)

theorem min_value_S : ∃ x, S x = 2 ∧ ∀ x, S x ≥ 2 := by
  sorry

theorem max_value_m : ∀ x y, S x ≥ m * (-y^2 + 2*y) → 0 ≤ m ∧ m ≤ 2 := by
  sorry

end min_value_S_max_value_m_l69_69976


namespace expected_worth_of_coin_flip_l69_69244

theorem expected_worth_of_coin_flip :
  let p_heads := 2 / 3
  let p_tails := 1 / 3
  let gain_heads := 5
  let loss_tails := -9
  (p_heads * gain_heads) + (p_tails * loss_tails) = 1 / 3 :=
by
  -- Proof will be here
  sorry

end expected_worth_of_coin_flip_l69_69244


namespace total_seats_at_round_table_l69_69912

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69912


namespace div_product_four_consecutive_integers_l69_69841

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69841


namespace repeating_decimal_fraction_sum_l69_69131

theorem repeating_decimal_fraction_sum : 
  let x := 36 / 99 in
  let a := 4 in
  let b := 11 in
  gcd a b = 1 ∧ (a : ℚ) / (b : ℚ) = x → a + b = 15 :=
by
  sorry

end repeating_decimal_fraction_sum_l69_69131


namespace total_seats_l69_69927

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69927


namespace divisor_of_product_of_four_consecutive_integers_l69_69654

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69654


namespace total_seats_round_table_l69_69901

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69901


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69767

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69767


namespace percentage_of_x_eq_y_l69_69539

theorem percentage_of_x_eq_y
  (x y : ℝ) 
  (h : 0.60 * (x - y) = 0.20 * (x + y)) :
  y = 0.50 * x := 
sorry

end percentage_of_x_eq_y_l69_69539


namespace rectangle_perimeter_of_right_triangle_l69_69370

noncomputable def right_triangle_area (a b: ℕ) : ℝ := (1/2 : ℝ) * a * b

noncomputable def rectangle_length (area width: ℝ) : ℝ := area / width

noncomputable def rectangle_perimeter (length width: ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter_of_right_triangle :
  rectangle_perimeter (rectangle_length (right_triangle_area 7 24) 5) 5 = 43.6 :=
by
  sorry

end rectangle_perimeter_of_right_triangle_l69_69370


namespace calculate_x_minus_y_l69_69349

theorem calculate_x_minus_y (x y z : ℝ) 
    (h1 : x - y + z = 23) 
    (h2 : x - y - z = 7) : 
    x - y = 15 :=
by
  sorry

end calculate_x_minus_y_l69_69349


namespace arith_seq_ratio_l69_69108

variables {a₁ d : ℝ} (h₁ : d ≠ 0) (h₂ : (a₁ + 2*d)^2 ≠ a₁ * (a₁ + 8*d))

theorem arith_seq_ratio:
  (a₁ + 2*d) / (a₁ + 5*d) = 1 / 2 :=
sorry

end arith_seq_ratio_l69_69108


namespace exists_m_square_between_l69_69118

theorem exists_m_square_between (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, a < m^2 ∧ m^2 < d := 
sorry

end exists_m_square_between_l69_69118


namespace circles_radii_divide_regions_l69_69150

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l69_69150


namespace more_tails_than_heads_l69_69418

theorem more_tails_than_heads (total_flips : ℕ) (heads : ℕ) (tails : ℕ) :
  total_flips = 211 → heads = 65 → tails = (total_flips - heads) → (tails - heads) = 81 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h1, h2]
  exact h3.trans (show 211 - 65 - 65 = 81 by norm_num)

end more_tails_than_heads_l69_69418


namespace greatest_divisor_of_four_consecutive_integers_l69_69662

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69662


namespace rahim_average_price_l69_69447

def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def books_shop2 : ℕ := 40
def cost_shop2 : ℕ := 800

def total_books : ℕ := books_shop1 + books_shop2
def total_cost : ℕ := cost_shop1 + cost_shop2
def average_price_per_book : ℕ := total_cost / total_books

theorem rahim_average_price :
  average_price_per_book = 20 := by
  sorry

end rahim_average_price_l69_69447


namespace net_effect_on_sale_l69_69870

theorem net_effect_on_sale (P Q : ℝ) :
  let new_price := 0.65 * P
  let new_quantity := 1.8 * Q
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  new_revenue - original_revenue = 0.17 * original_revenue :=
by
  sorry

end net_effect_on_sale_l69_69870


namespace fraction_inequality_l69_69967

variables (a b m : ℝ)

theorem fraction_inequality (h1 : a > b) (h2 : m > 0) : (b + m) / (a + m) > b / a :=
sorry

end fraction_inequality_l69_69967


namespace parabola_hyperbola_focus_l69_69296

noncomputable def focus_left (p : ℝ) : ℝ × ℝ :=
  (-p / 2, 0)

theorem parabola_hyperbola_focus (p : ℝ) (hp : p > 0) : 
  focus_left p = (-2, 0) ↔ p = 4 :=
by 
  sorry

end parabola_hyperbola_focus_l69_69296


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69781

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69781


namespace divisor_of_product_of_four_consecutive_integers_l69_69649

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69649


namespace largest_perfect_square_factor_1800_l69_69038

def largest_perfect_square_factor (n : ℕ) : ℕ :=
  if n = 1800 then 900 else sorry

theorem largest_perfect_square_factor_1800 : 
  largest_perfect_square_factor 1800 = 900 :=
by
  -- Proof is not needed, so we use sorry
  sorry

end largest_perfect_square_factor_1800_l69_69038


namespace find_n_tan_eq_l69_69959

theorem find_n_tan_eq (n : ℝ) (h1 : -180 < n) (h2 : n < 180) (h3 : Real.tan (n * Real.pi / 180) = Real.tan (678 * Real.pi / 180)) : 
  n = 138 := 
sorry

end find_n_tan_eq_l69_69959


namespace four_consecutive_product_divisible_by_12_l69_69796

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69796


namespace hexagon_same_length_probability_l69_69565

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l69_69565


namespace train_speed_l69_69079

/-- Define the lengths of the train and the bridge and the time taken to cross the bridge. --/
def len_train : ℕ := 360
def len_bridge : ℕ := 240
def time_minutes : ℕ := 4
def time_seconds : ℕ := 240 -- 4 minutes converted to seconds

/-- Define the speed calculation based on the given domain. --/
def total_distance : ℕ := len_train + len_bridge
def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

/-- The statement to prove that the speed of the train is 2.5 m/s. --/
theorem train_speed :
  speed total_distance time_seconds = 2.5 := sorry

end train_speed_l69_69079


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69713

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69713


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69629

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69629


namespace correct_exponent_calculation_l69_69474

theorem correct_exponent_calculation (a : ℝ) : 
  (a^5 * a^2 = a^7) :=
by
  sorry

end correct_exponent_calculation_l69_69474


namespace integer_solutions_xy_l69_69957

theorem integer_solutions_xy :
  ∃ (x y : ℤ), (x + y + x * y = 500) ∧ 
               ((x = 0 ∧ y = 500) ∨ 
                (x = -2 ∧ y = -502) ∨ 
                (x = 2 ∧ y = 166) ∨ 
                (x = -4 ∧ y = -168)) :=
by
  sorry

end integer_solutions_xy_l69_69957


namespace find_number_l69_69235

theorem find_number (Number : ℝ) (h : Number / 5 = 30 / 600) : Number = 1 / 4 :=
by sorry

end find_number_l69_69235


namespace tapA_fill_time_l69_69469

-- Define the conditions
def fillTapA (t : ℕ) := 1 / t
def fillTapB := 1 / 40
def fillCombined (t : ℕ) := 9 * (fillTapA t + fillTapB)
def fillRemaining := 23 * fillTapB

-- Main theorem statement
theorem tapA_fill_time : ∀ (t : ℕ), fillCombined t + fillRemaining = 1 → t = 45 := by
  sorry

end tapA_fill_time_l69_69469


namespace rational_sum_of_squares_is_square_l69_69987

theorem rational_sum_of_squares_is_square (a b c : ℚ) :
  ∃ r : ℚ, r ^ 2 = (1 / (b - c) ^ 2 + 1 / (c - a) ^ 2 + 1 / (a - b) ^ 2) :=
by
  sorry

end rational_sum_of_squares_is_square_l69_69987


namespace num_common_tangents_l69_69512

-- Define the first circle
def circle1 (x y : ℝ) : Prop := (x + 2) ^ 2 + y ^ 2 = 4
-- Define the second circle
def circle2 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 9

-- Prove that the number of common tangent lines between the given circles is 2
theorem num_common_tangents : ∃ (n : ℕ), n = 2 ∧
  -- The circles do not intersect nor are they internally tangent
  (∀ (x y : ℝ), ¬(circle1 x y ∧ circle2 x y) ∧ 
  -- There exist exactly n common tangent lines
  ∃ (C : ℕ), C = n) :=
sorry

end num_common_tangents_l69_69512


namespace prove_total_bill_is_correct_l69_69085

noncomputable def totalCostAfterDiscounts : ℝ :=
  let adultsMealsCost := 8 * 12
  let teenagersMealsCost := 4 * 10
  let childrenMealsCost := 3 * 7
  let adultsSodasCost := 8 * 3.5
  let teenagersSodasCost := 4 * 3.5
  let childrenSodasCost := 3 * 1.8
  let appetizersCost := 4 * 8
  let dessertsCost := 5 * 5

  let subtotal := adultsMealsCost + teenagersMealsCost + childrenMealsCost +
                  adultsSodasCost + teenagersSodasCost + childrenSodasCost +
                  appetizersCost + dessertsCost

  let discountAdultsMeals := 0.10 * adultsMealsCost
  let discountDesserts := 5
  let discountChildrenMealsAndSodas := 0.15 * (childrenMealsCost + childrenSodasCost)

  let adjustedSubtotal := subtotal - discountAdultsMeals - discountDesserts - discountChildrenMealsAndSodas

  let additionalDiscount := if subtotal > 200 then 0.05 * adjustedSubtotal else 0
  let total := adjustedSubtotal - additionalDiscount

  total

theorem prove_total_bill_is_correct : totalCostAfterDiscounts = 230.70 :=
by sorry

end prove_total_bill_is_correct_l69_69085


namespace angle_bisector_form_l69_69885

noncomputable def P : ℝ × ℝ := (-8, 5)
noncomputable def Q : ℝ × ℝ := (-15, -19)
noncomputable def R : ℝ × ℝ := (1, -7)

-- Function to check if the given equation can be in the form ax + 2y + c = 0
-- and that a + c equals 89.
theorem angle_bisector_form (a c : ℝ) : a + c = 89 :=
by
   sorry

end angle_bisector_form_l69_69885


namespace number_of_blue_spotted_fish_l69_69351

theorem number_of_blue_spotted_fish : 
  ∀ (fish_total : ℕ) (one_third_blue : ℕ) (half_spotted : ℕ),
    fish_total = 30 →
    one_third_blue = fish_total / 3 →
    half_spotted = one_third_blue / 2 →
    half_spotted = 5 := 
by
  intros fish_total one_third_blue half_spotted ht htb hhs
  sorry

end number_of_blue_spotted_fish_l69_69351


namespace range_of_a_l69_69275

/-- Given a fixed point A(a, 3) is outside the circle x^2 + y^2 - 2ax - 3y + a^2 + a = 0,
we want to show that the range of values for a is (0, 9/4). -/
theorem range_of_a (a : ℝ) :
  (∃ (A : ℝ × ℝ), A = (a, 3) ∧ ¬(∃ (x y : ℝ), x^2 + y^2 - 2*a*x - 3*y + a^2 + a = 0))
  ↔ (0 < a ∧ a < 9/4) :=
sorry

end range_of_a_l69_69275


namespace range_of_decreasing_function_l69_69284

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x < 3 → (deriv (f a) x) ≤ 0) ↔ 0 ≤ a ∧ a ≤ 3/4 := 
sorry

end range_of_decreasing_function_l69_69284


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69813

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69813


namespace find_m_l69_69110

open Classical

noncomputable def vec_a : ℝ × ℝ × ℝ := (-1, 1, 3)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ × ℝ := (1, 3, m)

def perp (u v : ℝ × ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0)

theorem find_m (m : ℝ) : perp (2 • vec_a + vec_b m) vec_a ↔ m = -8 := by
  sorry

end find_m_l69_69110


namespace spent_on_video_games_l69_69997

-- Defining the given amounts
def initial_amount : ℕ := 84
def grocery_spending : ℕ := 21
def final_amount : ℕ := 39

-- The proof statement: Proving Lenny spent $24 on video games.
theorem spent_on_video_games : initial_amount - final_amount - grocery_spending = 24 :=
by
  sorry

end spent_on_video_games_l69_69997


namespace how_many_toys_l69_69175

theorem how_many_toys (initial_savings : ℕ) (allowance : ℕ) (toy_cost : ℕ)
  (h1 : initial_savings = 21)
  (h2 : allowance = 15)
  (h3 : toy_cost = 6) :
  (initial_savings + allowance) / toy_cost = 6 :=
by
  sorry

end how_many_toys_l69_69175


namespace Brittany_older_by_3_years_l69_69247

-- Define the necessary parameters as assumptions
variable (Rebecca_age : ℕ) (Brittany_return_age : ℕ) (vacation_years : ℕ)

-- Initial conditions
axiom h1 : Rebecca_age = 25
axiom h2 : Brittany_return_age = 32
axiom h3 : vacation_years = 4

-- Definition to capture Brittany's age before vacation
def Brittany_age_before_vacation (return_age vacation_period : ℕ) : ℕ := return_age - vacation_period

-- Theorem stating that Brittany is 3 years older than Rebecca
theorem Brittany_older_by_3_years :
  Brittany_age_before_vacation Brittany_return_age vacation_years - Rebecca_age = 3 :=
by
  sorry

end Brittany_older_by_3_years_l69_69247


namespace greatest_divisor_of_consecutive_product_l69_69854

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69854


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69636

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69636


namespace divisor_of_four_consecutive_integers_l69_69753

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69753


namespace greatest_divisor_of_consecutive_product_l69_69862

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69862


namespace hexagon_same_length_probability_l69_69558

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l69_69558


namespace greatest_divisor_of_four_consecutive_integers_l69_69658

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69658


namespace dots_not_visible_l69_69352

theorem dots_not_visible (visible_sum : ℕ) (total_faces_sum : ℕ) (num_dice : ℕ) (total_visible_faces : ℕ)
  (h1 : total_faces_sum = 21)
  (h2 : visible_sum = 22) 
  (h3 : num_dice = 3)
  (h4 : total_visible_faces = 7) :
  (num_dice * total_faces_sum - visible_sum) = 41 :=
sorry

end dots_not_visible_l69_69352


namespace company_x_installation_charge_l69_69212

theorem company_x_installation_charge:
  let price_X := 575
  let surcharge_X := 0.04 * price_X
  let installation_charge_X := 82.50
  let total_cost_X := price_X + surcharge_X + installation_charge_X
  let price_Y := 530
  let surcharge_Y := 0.03 * price_Y
  let installation_charge_Y := 93.00
  let total_cost_Y := price_Y + surcharge_Y + installation_charge_Y
  let savings := 41.60
  total_cost_X - total_cost_Y = savings → installation_charge_X = 82.50 :=
by
  intros h
  sorry

end company_x_installation_charge_l69_69212


namespace volume_in_cubic_yards_l69_69074

-- Define the conditions
def volume_in_cubic_feet : ℕ := 162
def cubic_feet_per_cubic_yard : ℕ := 27

-- Problem statement in Lean 4
theorem volume_in_cubic_yards : volume_in_cubic_feet / cubic_feet_per_cubic_yard = 6 := 
  by
    sorry

end volume_in_cubic_yards_l69_69074


namespace daughterAgeThreeYearsFromNow_l69_69055

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l69_69055


namespace inequality_a_neg_one_inequality_general_a_l69_69406

theorem inequality_a_neg_one : ∀ x : ℝ, (x^2 + x - 2 > 0) ↔ (x < -2 ∨ x > 1) :=
by { sorry }

theorem inequality_general_a : 
∀ (a x : ℝ), ax^2 - (a + 2)*x + 2 < 0 ↔ 
  if a = 0 then x > 1
  else if a < 0 then x < (2 / a) ∨ x > 1
  else if 0 < a ∧ a < 2 then 1 < x ∧ x < (2 / a)
  else if a = 2 then False
  else (2 / a) < x ∧ x < 1 :=
by { sorry }

end inequality_a_neg_one_inequality_general_a_l69_69406


namespace player_A_wins_iff_n_is_odd_l69_69494

-- Definitions of the problem conditions
structure ChessboardGame (n : ℕ) :=
  (stones : ℕ := 99)
  (playerA_first : Prop := true)
  (turns : ℕ := n * 99)

-- Statement of the problem
theorem player_A_wins_iff_n_is_odd (n : ℕ) (g : ChessboardGame n) : 
  PlayerA_has_winning_strategy ↔ n % 2 = 1 := 
sorry

end player_A_wins_iff_n_is_odd_l69_69494


namespace perfect_square_trinomial_l69_69538

theorem perfect_square_trinomial (m : ℤ) : 
  (x^2 - (m - 3) * x + 16 = (x - 4)^2) ∨ (x^2 - (m - 3) * x + 16 = (x + 4)^2) ↔ (m = -5 ∨ m = 11) := by
  sorry

end perfect_square_trinomial_l69_69538


namespace divisor_of_product_of_four_consecutive_integers_l69_69648

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69648


namespace red_light_probability_l69_69883

theorem red_light_probability (n : ℕ) (p_r : ℚ) (waiting_time_for_two_red : ℚ) 
    (prob_two_red : ℚ) :
    n = 4 →
    p_r = (1/3 : ℚ) →
    waiting_time_for_two_red = 4 →
    prob_two_red = (8/27 : ℚ) :=
by
  intros hn hp hw
  sorry

end red_light_probability_l69_69883


namespace james_vs_combined_l69_69171

def james_balloons : ℕ := 1222
def amy_balloons : ℕ := 513
def felix_balloons : ℕ := 687
def olivia_balloons : ℕ := 395
def combined_balloons : ℕ := amy_balloons + felix_balloons + olivia_balloons

theorem james_vs_combined :
  1222 = 1222 ∧ 513 = 513 ∧ 687 = 687 ∧ 395 = 395 → combined_balloons - james_balloons = 373 := by
  sorry

end james_vs_combined_l69_69171


namespace iterate_fixed_point_l69_69437

theorem iterate_fixed_point {f : ℤ → ℤ} (a : ℤ) :
  (∀ n, f^[n] a = a → f a = a) ∧ (f a = a → f^[22000] a = a) :=
sorry

end iterate_fixed_point_l69_69437


namespace James_watch_time_l69_69552

def Jeopardy_length : ℕ := 20
def Wheel_of_Fortune_length : ℕ := Jeopardy_length * 2
def Jeopardy_episodes : ℕ := 2
def Wheel_of_Fortune_episodes : ℕ := 2

theorem James_watch_time :
  (Jeopardy_episodes * Jeopardy_length + Wheel_of_Fortune_episodes * Wheel_of_Fortune_length) / 60 = 2 :=
by
  sorry

end James_watch_time_l69_69552


namespace reciprocal_of_sum_l69_69047

theorem reciprocal_of_sum :
  (1 / ((3 : ℚ) / 4 + (5 : ℚ) / 6)) = (12 / 19) :=
by
  sorry

end reciprocal_of_sum_l69_69047


namespace grant_earnings_proof_l69_69122

noncomputable def total_earnings (X Y Z W : ℕ): ℕ :=
  let first_month := X
  let second_month := 3 * X + Y
  let third_month := 2 * second_month - Z
  let average := (first_month + second_month + third_month) / 3
  let fourth_month := average + W
  first_month + second_month + third_month + fourth_month

theorem grant_earnings_proof : total_earnings 350 30 20 50 = 5810 := by
  sorry

end grant_earnings_proof_l69_69122


namespace four_consecutive_integers_divisible_by_12_l69_69687

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69687


namespace product_of_consecutive_integers_l69_69670

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69670


namespace smaller_angle_at_8_15_l69_69084

noncomputable def hour_hand_position (h m : ℕ) : ℝ := (↑h % 12) * 30 + (↑m / 60) * 30

noncomputable def minute_hand_position (m : ℕ) : ℝ := ↑m / 60 * 360

noncomputable def angle_between_hands (h m : ℕ) : ℝ :=
  let θ := |hour_hand_position h m - minute_hand_position m|
  min θ (360 - θ)

theorem smaller_angle_at_8_15 : angle_between_hands 8 15 = 157.5 := by
  sorry

end smaller_angle_at_8_15_l69_69084


namespace divisor_of_product_of_four_consecutive_integers_l69_69645

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69645


namespace divisor_of_product_of_four_consecutive_integers_l69_69643

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69643


namespace integer_roots_polynomial_l69_69516

theorem integer_roots_polynomial 
(m n : ℕ) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  (∃ a b c : ℤ, a + b + c = 17 ∧ a * b * c = n^2 ∧ a * b + b * c + c * a = m) ↔ 
  (m, n) = (80, 10) ∨ (m, n) = (88, 12) ∨ (m, n) = (80, 8) ∨ (m, n) = (90, 12) := 
sorry

end integer_roots_polynomial_l69_69516


namespace chord_length_proof_tangent_lines_through_M_l69_69394

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_l (x y : ℝ) : Prop := 2*x - y + 4 = 0

noncomputable def point_M : (ℝ × ℝ) := (3, 1)

noncomputable def chord_length : ℝ := 4 * Real.sqrt (5) / 5

noncomputable def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0
noncomputable def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem chord_length_proof :
  ∀ x y : ℝ, circle_C x y → line_l x y → chord_length = 4 * Real.sqrt (5) / 5 :=
by sorry

theorem tangent_lines_through_M :
  ∀ x y : ℝ, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x) :=
by sorry

end chord_length_proof_tangent_lines_through_M_l69_69394


namespace max_profit_l69_69367

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def revenue (x : ℝ) : ℝ := 0.05 * 1000 * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x - fixed_cost * 10

theorem max_profit : ∃ x_opt : ℝ, ∀ x : ℝ, 0 < x → 
  profit x ≤ profit 100 ∧ x_opt = 100 :=
by
  sorry

end max_profit_l69_69367


namespace find_alpha_l69_69586

theorem find_alpha (n : ℕ) (h : ∀ x : ℤ, x * x * x + α * x + 4 - 2 * 2016 ^ n = 0 → ∀ r : ℤ, x = r)
  : α = -3 :=
sorry

end find_alpha_l69_69586


namespace range_of_a_l69_69966

noncomputable def f (a x : ℝ) := a * x - 1
noncomputable def g (x : ℝ) := -x^2 + 2 * x + 1

theorem range_of_a (a : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ (Set.Icc (-1 : ℝ) 1) → ∃ (x2 : ℝ), x2 ∈ (Set.Icc (0 : ℝ) 2) ∧ f a x1 < g x2) ↔ a ∈ Set.Ioo (-3 : ℝ) 3 :=
sorry

end range_of_a_l69_69966


namespace main_theorem_l69_69942

/-- A good integer is an integer whose absolute value is not a perfect square. -/
def good (n : ℤ) : Prop := ∀ k : ℤ, k^2 ≠ |n|

/-- Integer m can be represented as a sum of three distinct good integers u, v, w whose product is the square of an odd integer. -/
def special_representation (m : ℤ) : Prop :=
  ∃ u v w : ℤ,
    good u ∧ good v ∧ good w ∧
    (u ≠ v ∧ u ≠ w ∧ v ≠ w) ∧
    (∃ k : ℤ, (u * v * w = k^2 ∧ k % 2 = 1)) ∧
    (m = u + v + w)

/-- All integers m having the property that they can be represented in infinitely many ways as a sum of three distinct good integers whose product is the square of an odd integer are those which are congruent to 3 modulo 4. -/
theorem main_theorem (m : ℤ) : special_representation m ↔ m % 4 = 3 := sorry

end main_theorem_l69_69942


namespace sum_ai_le_sum_bi_l69_69330

open BigOperators

variable {α : Type*} [LinearOrderedField α]

theorem sum_ai_le_sum_bi {n : ℕ} {a b : Fin n → α}
  (h1 : ∀ i, 0 < a i)
  (h2 : ∀ i, 0 < b i)
  (h3 : ∑ i, (a i)^2 / b i ≤ ∑ i, b i) :
  ∑ i, a i ≤ ∑ i, b i :=
sorry

end sum_ai_le_sum_bi_l69_69330


namespace expressible_numbers_count_l69_69960

theorem expressible_numbers_count : ∃ k : ℕ, k = 2222 ∧ ∀ n : ℕ, n ≤ 2000 → ∃ x : ℝ, n = Int.floor x + Int.floor (3 * x) + Int.floor (5 * x) :=
by sorry

end expressible_numbers_count_l69_69960


namespace hexagon_diagonal_extension_projections_l69_69181

open Geometry

theorem hexagon_diagonal_extension_projections
{ABCDEF : RegularPolygon (Fin 6) ℝ} -- original hexagon ABCDEF
(K L M : Point ℝ ℝ) -- points K, L, M
(H : RegularPolygon (Fin 6) ℝ) -- hexagon H formed as described
(P Q R : Point ℝ ℝ) -- points P, Q, R
(h1 : K ∈ Line ABCDEF.diagonal₁) -- assumption for K
(h2 : L ∈ Line ABCDEF.diagonal₂) -- assumption for L
(h3 : M ∈ Line ABCDEF.diagonal₃) -- assumption for M
(h4 : H.formado_sub_intersec_sector KLM ABCDEF) -- H formation condition
(h5 : H.si_extensión ∉ Triangle KLM) -- extension condition for H
(h6 : P ∈ Intersection (Extension (H.edge₁)) (Extension (H.edge₂))) -- P intersection condition
(h7 : Q ∈ Intersection (Extension (H.edge₃)) (Extension (H.edge₄))) -- Q intersection condition
(h8 : R ∈ Intersection (Extension (H.edge₅)) (Extension (H.edge₆))) -- R intersection condition
: P ∈ Line (Extension ABCDEF.diagonal₁) ∧
  Q ∈ Line (Extension ABCDEF.diagonal₂) ∧
  R ∈ Line (Extension ABCDEF.diagonal₃) :=
by
  sorry

end hexagon_diagonal_extension_projections_l69_69181


namespace intersection_at_most_one_l69_69116

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the statement to be proved
theorem intersection_at_most_one (a : ℝ) :
  ∀ x1 x2 : ℝ, f x1 = f x2 → x1 = x2 :=
by
  sorry

end intersection_at_most_one_l69_69116


namespace find_second_number_l69_69454

theorem find_second_number (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 13) / 3 + 9 → x = 40 :=
by
  sorry

end find_second_number_l69_69454


namespace Nick_riding_speed_l69_69889

theorem Nick_riding_speed (Alan_speed Maria_ratio Nick_ratio : ℝ) 
(h1 : Alan_speed = 6) (h2 : Maria_ratio = 3/4) (h3 : Nick_ratio = 4/3) : 
Nick_ratio * (Maria_ratio * Alan_speed) = 6 := 
by 
  sorry

end Nick_riding_speed_l69_69889


namespace dhoni_spent_300_dollars_l69_69513

theorem dhoni_spent_300_dollars :
  ∀ (L S X : ℝ),
  L = 6 →
  S = L - 2 →
  (X / S) - (X / L) = 25 →
  X = 300 :=
by
intros L S X hL hS hEquation
sorry

end dhoni_spent_300_dollars_l69_69513


namespace statement_B_is_false_l69_69256

def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

theorem statement_B_is_false (x y : ℝ) : 3 * (heartsuit x y) ≠ heartsuit (3 * x) y := by
  sorry

end statement_B_is_false_l69_69256


namespace four_consecutive_integers_divisible_by_12_l69_69694

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69694


namespace foreign_objects_total_l69_69245

theorem foreign_objects_total (burrs ticks : ℕ) (h1 : burrs = 12) (h2 : ticks = 6 * burrs) : burrs + ticks = 84 :=
by {
  subst h1,
  subst h2,
  simp,
}

end foreign_objects_total_l69_69245


namespace divisor_of_four_consecutive_integers_l69_69754

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69754


namespace area_of_rhombus_enclosed_by_equation_l69_69027

-- Given the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- Define the main theorem to be proven
theorem area_of_rhombus_enclosed_by_equation : 
  (∃ x y : ℝ, equation x y) → ∃ area : ℝ, area = 384 :=
by
  sorry

end area_of_rhombus_enclosed_by_equation_l69_69027


namespace sum_of_consecutive_evens_l69_69461

theorem sum_of_consecutive_evens (E1 E2 E3 E4 : ℕ) (h1 : E4 = 38) (h2 : E3 = E4 - 2) (h3 : E2 = E3 - 2) (h4 : E1 = E2 - 2) : 
  E1 + E2 + E3 + E4 = 140 := 
by 
  sorry

end sum_of_consecutive_evens_l69_69461


namespace length_of_side_of_regular_tetradecagon_l69_69984

theorem length_of_side_of_regular_tetradecagon (P : ℝ) (n : ℕ) (h₀ : n = 14) (h₁ : P = 154) : P / n = 11 := 
by
  sorry

end length_of_side_of_regular_tetradecagon_l69_69984


namespace four_consecutive_integers_divisible_by_12_l69_69690

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69690


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69810

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69810


namespace area_enclosed_by_abs_eq_12_l69_69029

theorem area_enclosed_by_abs_eq_12 :
  let A := { p : ℝ × ℝ | abs p.1 + abs (3 * p.2) = 12 } in
  ∃ area : ℝ, area = 96 ∧
    (∀ (triangle : set (ℝ × ℝ)),
      triangle ⊆ A →
      is_triangle triangle →
      area_of triangle = 24) →
    (∃ (number_of_triangles : ℕ), number_of_triangles = 4) :=
by
  -- Definitions and steps would go here in a proper proof.
  sorry

end area_enclosed_by_abs_eq_12_l69_69029


namespace geometric_seq_arithmetic_condition_l69_69548

open Real

noncomputable def common_ratio (q : ℝ) := (q > 0) ∧ (q^2 - q - 1 = 0)

def arithmetic_seq_condition (a1 a2 a3 : ℝ) := (a2 = (a1 + a3) / 2)

theorem geometric_seq_arithmetic_condition (a1 a2 a3 a4 a5 : ℝ) (q : ℝ)
  (h1 : 0 < q)
  (h2 : q^2 - q - 1 = 0)
  (h3 : a2 = q * a1)
  (h4 : a3 = q * a2)
  (h5 : a4 = q * a3)
  (h6 : a5 = q * a4)
  (h7 : arithmetic_seq_condition a1 a2 a3) :
  (a4 + a5) / (a3 + a4) = (1 + sqrt 5) / 2 := 
sorry

end geometric_seq_arithmetic_condition_l69_69548


namespace lucy_50_cent_items_l69_69440

theorem lucy_50_cent_items :
  ∃ (a b c : ℕ), a + b + c = 30 ∧ 50 * a + 150 * b + 300 * c = 4500 ∧ a = 6 :=
by
  sorry

end lucy_50_cent_items_l69_69440


namespace factorize_expression_l69_69261

theorem factorize_expression (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := 
by sorry

end factorize_expression_l69_69261


namespace arithmetic_sequence_max_sum_l69_69527

theorem arithmetic_sequence_max_sum (a d t : ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a > 0) 
  (h2 : (9 * t) = a + 5 * d) 
  (h3 : (11 * t) = a + 4 * d) 
  (h4 : ∀ n, S n = (n * (2 * a + (n - 1) * d)) / 2) :
  n = 10 :=
sorry

end arithmetic_sequence_max_sum_l69_69527


namespace greatest_divisor_of_four_consecutive_integers_l69_69831

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69831


namespace reeya_average_score_l69_69448

theorem reeya_average_score :
  let scores := [50, 60, 70, 80, 80]
  let sum_scores := scores.sum
  let num_scores := scores.length
  sum_scores / num_scores = 68 :=
by
  sorry

end reeya_average_score_l69_69448


namespace g_diff_eq_neg8_l69_69456

noncomputable def g : ℝ → ℝ := sorry

axiom linear_g : ∀ x y : ℝ, g (x + y) = g x + g y

axiom condition_g : ∀ x : ℝ, g (x + 2) - g x = 4

theorem g_diff_eq_neg8 : g 2 - g 6 = -8 :=
by
  sorry

end g_diff_eq_neg8_l69_69456


namespace regions_formed_l69_69164

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l69_69164


namespace shirt_selling_price_l69_69233

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l69_69233


namespace standard_equation_line_standard_equation_circle_intersection_range_a_l69_69407

theorem standard_equation_line (a t x y : ℝ) (h1 : x = a - 2 * t * y) (h2 : y = -4 * t) : 
    2 * x - y - 2 * a = 0 :=
sorry

theorem standard_equation_circle (θ x y : ℝ) (h1 : x = 4 * Real.cos θ) (h2 : y = 4 * Real.sin θ) : 
    x ^ 2 + y ^ 2 = 16 :=
sorry

theorem intersection_range_a (a : ℝ) (h : ∃ (t θ : ℝ), (a - 2 * t * (-4 * t)) = 4 * (Real.cos θ) ∧ (-4 * t) = 4 * (Real.sin θ)) :
    -4 * Real.sqrt 5 <= a ∧ a <= 4 * Real.sqrt 5 :=
sorry

end standard_equation_line_standard_equation_circle_intersection_range_a_l69_69407


namespace train_car_passengers_l69_69369

theorem train_car_passengers (x : ℕ) (h : 60 * x = 732 + 228) : x = 16 :=
by
  sorry

end train_car_passengers_l69_69369


namespace number_of_squares_in_grid_l69_69953

-- Grid of size 6 × 6 composed entirely of squares.
def grid_size : Nat := 6

-- Definition of the function that counts the number of squares of a given size in an n × n grid.
def count_squares (n : Nat) (size : Nat) : Nat :=
  (n - size + 1) * (n - size + 1)

noncomputable def total_squares : Nat :=
  List.sum (List.map (count_squares grid_size) (List.range grid_size).tail)  -- Using tail to skip zero size

theorem number_of_squares_in_grid : total_squares = 86 := by
  sorry

end number_of_squares_in_grid_l69_69953


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69804

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69804


namespace solve_system_l69_69383

theorem solve_system : ∃ x y : ℚ, 
  (2 * x + 3 * y = 7 - 2 * x + 7 - 3 * y) ∧ 
  (3 * x - 2 * y = x - 2 + y - 2) ∧ 
  x = 3 / 4 ∧ 
  y = 11 / 6 := 
by 
  sorry

end solve_system_l69_69383


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69717

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69717


namespace total_seats_round_table_l69_69904

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69904


namespace circle_properties_l69_69005

theorem circle_properties (C : ℝ) (hC : C = 36) :
  let r := 18 / π
  let d := 36 / π
  let A := 324 / π
  2 * π * r = 36 ∧ d = 2 * r ∧ A = π * r^2 :=
by
  sorry

end circle_properties_l69_69005


namespace initial_money_eq_l69_69322

-- Definitions for the problem conditions
def spent_on_sweets : ℝ := 1.25
def spent_on_friends : ℝ := 2 * 1.20
def money_left : ℝ :=  4.85

-- Statement of the problem to prove
theorem initial_money_eq :
  spent_on_sweets + spent_on_friends + money_left = 8.50 := 
sorry

end initial_money_eq_l69_69322


namespace even_sum_of_digits_residue_l69_69318

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem even_sum_of_digits_residue (k : ℕ) (h : 2 ≤ k) (r : ℕ) (hr : r < k) :
  ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r := 
sorry

end even_sum_of_digits_residue_l69_69318


namespace FI_squared_correct_l69_69307

noncomputable def FI_squared : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 4)
  let D : ℝ × ℝ := (0, 4)
  let E : ℝ × ℝ := (3, 0)
  let H : ℝ × ℝ := (0, 1)
  let F : ℝ × ℝ := (4, 1)
  let G : ℝ × ℝ := (1, 4)
  let I : ℝ × ℝ := (3, 0)
  let J : ℝ × ℝ := (0, 1)
  let FI_squared := (4 - 3)^2 + (1 - 0)^2
  FI_squared

theorem FI_squared_correct : FI_squared = 2 :=
by
  sorry

end FI_squared_correct_l69_69307


namespace jesse_rooms_l69_69173

theorem jesse_rooms:
  ∀ (l w A n: ℕ), 
  l = 19 ∧ 
  w = 18 ∧ 
  A = 6840 ∧ 
  n = A / (l * w) → 
  n = 20 :=
by
  intros
  sorry

end jesse_rooms_l69_69173


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69822

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69822


namespace length_of_second_edge_l69_69008

-- Define the edge lengths and volume
def edge1 : ℕ := 6
def edge3 : ℕ := 6
def volume : ℕ := 180

-- The theorem to state the length of the second edge
theorem length_of_second_edge (edge2 : ℕ) (h : edge1 * edge2 * edge3 = volume) :
  edge2 = 5 :=
by
  -- Skipping the proof
  sorry

end length_of_second_edge_l69_69008


namespace div_product_four_consecutive_integers_l69_69840

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69840


namespace three_colors_sufficient_l69_69046

-- Definition of the tessellation problem with specified conditions.
def tessellation (n : ℕ) (x_divisions : ℕ) (y_divisions : ℕ) : Prop :=
  n = 8 ∧ x_divisions = 2 ∧ y_divisions = 2

-- Definition of the adjacency property.
def no_adjacent_same_color {α : Type} (coloring : ℕ → ℕ → α) : Prop :=
  ∀ (i j : ℕ), i < 8 → j < 8 →
  (i > 0 → coloring i j ≠ coloring (i-1) j) ∧ 
  (j > 0 → coloring i j ≠ coloring i (j-1)) ∧
  (i < 7 → coloring i j ≠ coloring (i+1) j) ∧ 
  (j < 7 → coloring i j ≠ coloring i (j+1)) ∧
  (i > 0 ∧ j > 0 → coloring i j ≠ coloring (i-1) (j-1)) ∧
  (i < 7 ∧ j < 7 → coloring i j ≠ coloring (i+1) (j+1)) ∧
  (i > 0 ∧ j < 7 → coloring i j ≠ coloring (i-1) (j+1)) ∧
  (i < 7 ∧ j > 0 → coloring i j ≠ coloring (i+1) (j-1))

-- The main theorem that needs to be proved.
theorem three_colors_sufficient : ∃ (k : ℕ) (coloring : ℕ → ℕ → ℕ), k = 3 ∧ 
  tessellation 8 2 2 ∧ 
  no_adjacent_same_color coloring := by
  sorry 

end three_colors_sufficient_l69_69046


namespace ratio_a2_a3_l69_69419

namespace SequenceProof

def a (n : ℕ) : ℤ := 3 - 2^n

theorem ratio_a2_a3 : a 2 / a 3 = 1 / 5 := by
  sorry

end SequenceProof

end ratio_a2_a3_l69_69419


namespace arithmetic_progression_squares_l69_69389

theorem arithmetic_progression_squares :
  ∃ (n : ℤ), ((3 * n^2 + 8 = 1111 * 5) ∧ (n-2, n, n+2) = (41, 43, 45)) :=
by
  sorry

end arithmetic_progression_squares_l69_69389


namespace inequality_system_solution_l69_69195

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 ↔ -1 ≤ x ∧ x < 1 :=
by
  sorry

end inequality_system_solution_l69_69195


namespace area_of_abs_sum_l69_69026

theorem area_of_abs_sum (x y : ℝ) (h : |x| + |3 * y| = 12) : 
  let area := 96 in
  True :=
begin
  sorry
end

end area_of_abs_sum_l69_69026


namespace greatest_divisor_four_consecutive_integers_l69_69722

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69722


namespace alice_basketball_probability_l69_69505

/-- Alice and Bob play a game with a basketball. On each turn, if Alice has the basketball,
 there is a 5/8 chance that she will toss it to Bob and a 3/8 chance that she will keep the basketball.
 If Bob has the basketball, there is a 1/4 chance that he will toss it to Alice, and if he doesn't toss it to Alice,
 he keeps it. Alice starts with the basketball. What is the probability that Alice has the basketball again after two turns? -/
theorem alice_basketball_probability :
  (5 / 8) * (1 / 4) + (3 / 8) * (3 / 8) = 19 / 64 := 
by
  sorry

end alice_basketball_probability_l69_69505


namespace age_difference_ratio_l69_69983

theorem age_difference_ratio (h : ℕ) (f : ℕ) (m : ℕ) 
  (harry_age : h = 50) 
  (father_age : f = h + 24) 
  (mother_age : m = 22 + h) :
  (f - m) / h = 1 / 25 := 
by 
  sorry

end age_difference_ratio_l69_69983


namespace waiter_tables_l69_69080

theorem waiter_tables (total_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (number_of_tables : ℕ) 
  (h1 : total_customers = 22)
  (h2 : customers_left = 14)
  (h3 : people_per_table = 4)
  (h4 : remaining_customers = total_customers - customers_left)
  (h5 : number_of_tables = remaining_customers / people_per_table) :
  number_of_tables = 2 :=
by
  sorry

end waiter_tables_l69_69080


namespace Mark_speeding_ticket_owed_amount_l69_69324

theorem Mark_speeding_ticket_owed_amount :
  let base_fine := 50
  let additional_penalty_per_mph := 2
  let mph_over_limit := 45
  let school_zone_multiplier := 2
  let court_costs := 300
  let lawyer_fee_per_hour := 80
  let lawyer_hours := 3
  let additional_penalty := additional_penalty_per_mph * mph_over_limit
  let pre_school_zone_fine := base_fine + additional_penalty
  let doubled_fine := pre_school_zone_fine * school_zone_multiplier
  let total_fine_with_court_costs := doubled_fine + court_costs
  let lawyer_total_fee := lawyer_fee_per_hour * lawyer_hours
  let total_owed := total_fine_with_court_costs + lawyer_total_fee
  total_owed = 820 :=
by
  sorry

end Mark_speeding_ticket_owed_amount_l69_69324


namespace total_seats_round_table_l69_69905

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69905


namespace initial_amount_l69_69353

theorem initial_amount (X : ℝ) (h : 0.7 * X = 3500) : X = 5000 :=
by
  sorry

end initial_amount_l69_69353


namespace temperature_difference_l69_69345

theorem temperature_difference (highest lowest : ℝ) (h_high : highest = 27) (h_low : lowest = 17) :
  highest - lowest = 10 :=
by
  sorry

end temperature_difference_l69_69345


namespace proper_divisors_condition_l69_69068

theorem proper_divisors_condition (N : ℕ) :
  ∀ x : ℕ, (x ∣ N ∧ x ≠ 1 ∧ x ≠ N) → 
  (∀ L : ℕ, (L ∣ N ∧ L ≠ 1 ∧ L ≠ N) → (L = x^3 + 3 ∨ L = x^3 - 3)) → 
  (N = 10 ∨ N = 22) :=
by
  sorry

end proper_divisors_condition_l69_69068


namespace elmer_saves_14_3_percent_l69_69096

-- Define the problem statement conditions and goal
theorem elmer_saves_14_3_percent (old_efficiency new_efficiency : ℝ) (old_cost new_cost : ℝ) :
  new_efficiency = 1.75 * old_efficiency →
  new_cost = 1.5 * old_cost →
  (500 / old_efficiency * old_cost - 500 / new_efficiency * new_cost) / (500 / old_efficiency * old_cost) * 100 = 14.3 := by
  -- sorry to skip the actual proof
  sorry

end elmer_saves_14_3_percent_l69_69096


namespace sin_cos_sum_l69_69136

theorem sin_cos_sum (x y r : ℝ) (h : r = Real.sqrt (x^2 + y^2)) (ha : (x = 5) ∧ (y = -12)) :
  (y / r) + (x / r) = -7 / 13 :=
by
  sorry

end sin_cos_sum_l69_69136


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69802

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69802


namespace manfred_total_paychecks_l69_69996

-- Define the conditions
def first_paychecks : ℕ := 6
def first_paycheck_amount : ℕ := 750
def remaining_paycheck_amount : ℕ := first_paycheck_amount + 20
def average_amount : ℝ := 765.38

-- Main theorem statement
theorem manfred_total_paychecks (x : ℕ) (h : (first_paychecks * first_paycheck_amount + x * remaining_paycheck_amount) / (first_paychecks + x) = average_amount) : first_paychecks + x = 26 :=
by
  sorry

end manfred_total_paychecks_l69_69996


namespace prob_xi_range_negative_2_to_4_l69_69402

noncomputable
def normal_distribution_mean : ℝ := 1
def normal_distribution_variance : ℝ := 4

axiom xi_follows_normal_distribution (ξ : ℝ) : True

axiom prob_xi_greater_than_4 (ξ : ℝ) : ℙ {x | x > 4} = 0.1

theorem prob_xi_range_negative_2_to_4 (ξ : ℝ) 
  (H1 : xi_follows_normal_distribution ξ) 
  (H2 : prob_xi_greater_than_4 ξ) : 
  ℙ ((-2 : ℝ) ≤ ξ ∧ ξ ≤ 4) = 0.8 := 
sorry

end prob_xi_range_negative_2_to_4_l69_69402


namespace complete_square_ratio_l69_69579

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l69_69579


namespace largest_perfect_square_factor_of_1800_l69_69043

theorem largest_perfect_square_factor_of_1800 :
  ∃ k : ℕ, k ^ 2 ∣ 1800 ∧ (∀ n : ℕ, n ^ 2 ∣ 1800 → n ^ 2 ≤ k ^ 2) ∧ k ^ 2 = 900 :=
begin
  sorry
end

end largest_perfect_square_factor_of_1800_l69_69043


namespace completion_time_C_l69_69363

theorem completion_time_C (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3) 
  (h2 : r_B + r_C = 1 / 3) 
  (h3 : r_A + r_C = 1 / 3) :
  1 / r_C = 6 :=
by
  sorry

end completion_time_C_l69_69363


namespace total_number_of_seats_l69_69922

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69922


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69799

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69799


namespace greatest_divisor_of_four_consecutive_integers_l69_69664

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69664


namespace total_boxes_l69_69016

theorem total_boxes (w1 w2 : ℕ) (h1 : w1 = 400) (h2 : w1 = 2 * w2) : w1 + w2 = 600 := 
by
  sorry

end total_boxes_l69_69016


namespace total_seats_l69_69929

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69929


namespace original_proposition_converse_inverse_contrapositive_l69_69358

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n
def is_real (x : ℝ) : Prop := true

theorem original_proposition (x : ℝ) : is_integer x → is_real x := 
by sorry

theorem converse (x : ℝ) : ¬(is_real x → is_integer x) := 
by sorry

theorem inverse (x : ℝ) : ¬((¬ is_integer x) → (¬ is_real x)) := 
by sorry

theorem contrapositive (x : ℝ) : (¬ is_real x) → (¬ is_integer x) := 
by sorry

end original_proposition_converse_inverse_contrapositive_l69_69358


namespace lattice_point_in_PQE_l69_69876

-- Define points and their integer coordinates
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a convex quadrilateral with integer coordinates
structure ConvexQuadrilateral :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

-- Define the intersection point of diagonals as another point
def diagIntersection (quad: ConvexQuadrilateral) : Point := sorry

-- Define the condition for the sum of angles at P and Q being less than 180 degrees
def sumAnglesLessThan180 (quad : ConvexQuadrilateral) : Prop := sorry

-- Define a function to check if a point is a lattice point
def isLatticePoint (p : Point) : Prop := true  -- Since all points are lattice points by definition

-- Define the proof problem
theorem lattice_point_in_PQE (quad : ConvexQuadrilateral) (E : Point) :
  sumAnglesLessThan180 quad →
  ∃ p : Point, p ≠ quad.P ∧ p ≠ quad.Q ∧ isLatticePoint p ∧ sorry := sorry -- (prove the point is in PQE)

end lattice_point_in_PQE_l69_69876


namespace area_of_rhombus_l69_69036

theorem area_of_rhombus : 
  ∀ (x y : ℝ), (|x| + |3 * y| = 12) → 
  (area (x, y) = 96) :=
by sorry

-- Define the area function as used in the context of this problem:
noncomputable def area (p : ℝ × ℝ) : ℝ :=
if |p.1| + |3 * p.2| = 12 then 96
else 0

attribute [simp] abs_zero

end area_of_rhombus_l69_69036


namespace other_factor_of_lcm_l69_69011

theorem other_factor_of_lcm (A B : ℕ) 
  (hcf : Nat.gcd A B = 23) 
  (hA : A = 345) 
  (hcf_factor : 15 ∣ Nat.lcm A B) 
  : 23 ∣ Nat.lcm A B / 15 :=
sorry

end other_factor_of_lcm_l69_69011


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69606

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69606


namespace divisor_of_product_of_four_consecutive_integers_l69_69655

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69655


namespace circle_regions_l69_69155

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l69_69155


namespace percent_increase_sales_l69_69050

theorem percent_increase_sales (sales_this_year sales_last_year : ℝ) (h1 : sales_this_year = 460) (h2 : sales_last_year = 320) :
  (sales_this_year - sales_last_year) / sales_last_year * 100 = 43.75 :=
by
  sorry

end percent_increase_sales_l69_69050


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69624

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69624


namespace geometric_sequence_a5_l69_69341

theorem geometric_sequence_a5
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_ratio : ∀ n, a (n + 1) = 2 * a n)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := 
sorry

end geometric_sequence_a5_l69_69341


namespace maximum_value_of_f_l69_69009

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - 3 * x else -2 * x + 1

theorem maximum_value_of_f : ∃ (m : ℝ), (∀ x : ℝ, f x ≤ m) ∧ (m = 2) := by
  sorry

end maximum_value_of_f_l69_69009


namespace game_of_24_l69_69601

theorem game_of_24 : 
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  ((b + c / a) * d = 24) :=
by
  let a := 3
  let b := -5
  let c := 6
  let d := -8
  show (b + c / a) * d = 24
  sorry

end game_of_24_l69_69601


namespace expected_twos_three_dice_l69_69209

def expected_twos (n : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), k * (nat.choose n k) * (1/6)^k * (5/6)^(n - k)

theorem expected_twos_three_dice : expected_twos 3 = 1/2 :=
by
  sorry

end expected_twos_three_dice_l69_69209


namespace problem_l69_69051

theorem problem (a b c d : ℝ) (h1 : a - b - c + d = 18) (h2 : a + b - c - d = 6) : (b - d) ^ 2 = 36 :=
by
  sorry

end problem_l69_69051


namespace not_divides_two_pow_n_sub_one_l69_69333

theorem not_divides_two_pow_n_sub_one (n : ℕ) (h1 : n > 1) : ¬ n ∣ (2^n - 1) :=
sorry

end not_divides_two_pow_n_sub_one_l69_69333


namespace find_selling_price_l69_69231

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l69_69231


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69609

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69609


namespace king_lancelot_seats_38_l69_69915

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69915


namespace greatest_divisor_of_four_consecutive_integers_l69_69656

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69656


namespace building_height_l69_69211

theorem building_height (H : ℝ) 
                        (bounced_height : ℕ → ℝ) 
                        (h_bounce : ∀ n, bounced_height n = H / 2 ^ (n + 1)) 
                        (h_fifth : bounced_height 5 = 3) : 
    H = 96 := 
by {
  sorry
}

end building_height_l69_69211


namespace sum_local_values_of_digits_l69_69355

theorem sum_local_values_of_digits :
  let d2 := 2000
  let d3 := 300
  let d4 := 40
  let d5 := 5
  d2 + d3 + d4 + d5 = 2345 :=
by
  sorry

end sum_local_values_of_digits_l69_69355


namespace arithmetic_geometric_sequence_relation_l69_69277

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (hA : ∀ n: ℕ, a (n + 1) - a n = a 1) 
  (hG : ∀ n: ℕ, b (n + 1) / b n = b 1) 
  (h1 : a 1 = b 1) 
  (h11 : a 11 = b 11) 
  (h_pos : 0 < a 1 ∧ 0 < a 11 ∧ 0 < b 11 ∧ 0 < b 1) :
  a 6 ≥ b 6 := sorry

end arithmetic_geometric_sequence_relation_l69_69277


namespace perfect_cube_divisor_l69_69401

theorem perfect_cube_divisor (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + 3*a*b + 3*b^2 - 1 ∣ a + b^3) :
  ∃ k > 1, ∃ m : ℕ, a^2 + 3*a*b + 3*b^2 - 1 = k^3 * m := 
sorry

end perfect_cube_divisor_l69_69401


namespace total_seats_l69_69930

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69930


namespace negation_of_proposition_l69_69013

theorem negation_of_proposition :
  (¬ (∃ x_0 : ℝ, x_0 ≤ 0 ∧ x_0^2 ≥ 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end negation_of_proposition_l69_69013


namespace four_consecutive_product_divisible_by_12_l69_69794

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69794


namespace solution_exists_in_interval_l69_69348

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 3

theorem solution_exists_in_interval : ∃ x, 0 < x ∧ x < 1 ∧ f x = 0 :=
by {
  -- placeholder for the skipped proof
  sorry
}

end solution_exists_in_interval_l69_69348


namespace polynomial_range_open_interval_l69_69951

theorem polynomial_range_open_interval :
  ∀ (k : ℝ), k > 0 → ∃ (x y : ℝ), (1 - x * y)^2 + x^2 = k :=
by
  sorry

end polynomial_range_open_interval_l69_69951


namespace time_to_boil_l69_69941

def T₀ : ℝ := 20
def Tₘ : ℝ := 100
def t : ℝ := 10 * 60 -- 10 minutes converted to seconds
def c : ℝ := 4200 -- Specific heat capacity of water in J/(kg·K)
def L : ℝ := 2.3 * 10^6 -- Specific heat of vaporization of water in J/kg

theorem time_to_boil (m : ℝ) : 
  t₁ = t * (L / (c * (Tₘ - T₀))) ->
  m > 0 -> -- Assuming m (mass) is positive
  t₁ ≈ 68 * 60 :=
by
  sorry

end time_to_boil_l69_69941


namespace circle_division_l69_69167

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l69_69167


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69698

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69698


namespace quadratic_root_condition_l69_69988

theorem quadratic_root_condition (a b : ℝ) (h : (3:ℝ)^2 + 2 * a * 3 + 3 * b = 0) : 2 * a + b = -3 :=
by
  sorry

end quadratic_root_condition_l69_69988


namespace total_seats_l69_69928

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69928


namespace inscribed_polygon_cosine_l69_69487

noncomputable def angle_B (ABC : ℝ) : ℝ := 
  let B := 18 (1 - Mathlib.cos ABC) in
  B

noncomputable def angle_ACE (AC : ℝ) : ℝ :=
  let ACE := 2*AC^2 * (1 - Mathlib.cos AC) = 4 in
  ACE

theorem inscribed_polygon_cosine :
  ∀ (A B C D E : ℝ), A ∈ Circle ∧ B ∈ Circle ∧ C ∈ Circle ∧ D ∈ Circle ∧ E ∈ Circle ∧
    (AB = 3) ∧ (BC = 3) ∧ (CD = 3) ∧ (DE = 3) ∧ (AE = 2) →
    (1 - Mathlib.cos (angle_B 3)) * (1 - Mathlib.cos (angle_ACE 3)) = (1 / 9) :=
  by
  sorry

end inscribed_polygon_cosine_l69_69487


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69619

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69619


namespace div_product_four_consecutive_integers_l69_69845

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69845


namespace four_consecutive_product_divisible_by_12_l69_69789

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69789


namespace distinct_sum_product_problem_l69_69185

theorem distinct_sum_product_problem (S : ℤ) (hS : S ≥ 100) :
  ∃ a b c P : ℤ, a > b ∧ b > c ∧ a + b + c = S ∧ a * b * c = P ∧ 
    ¬(∀ x y z : ℤ, x > y ∧ y > z ∧ x + y + z = S → a = x ∧ b = y ∧ c = z) := 
sorry

end distinct_sum_product_problem_l69_69185


namespace probability_of_same_length_segments_l69_69564

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l69_69564


namespace daughter_age_in_3_years_l69_69056

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l69_69056


namespace packs_with_extra_red_pencils_eq_3_l69_69555

def total_packs : Nat := 15
def regular_red_per_pack : Nat := 1
def total_red_pencils : Nat := 21
def extra_red_per_pack : Nat := 2

theorem packs_with_extra_red_pencils_eq_3 :
  ∃ (packs_with_extra : Nat), packs_with_extra * extra_red_per_pack + (total_packs - packs_with_extra) * regular_red_per_pack = total_red_pencils ∧ packs_with_extra = 3 :=
by
  sorry

end packs_with_extra_red_pencils_eq_3_l69_69555


namespace calculate_result_l69_69249

theorem calculate_result :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = Real.cos (Real.pi / 4) :=
by
  sorry

end calculate_result_l69_69249


namespace domain_of_f_l69_69343

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | 4 - x ^ 2 ≥ 0 ∧ x ≠ 1}

theorem domain_of_f (x : ℝ) : domain_of_function x = {x | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l69_69343


namespace chloromethane_formation_l69_69103

variable (CH₄ Cl₂ CH₃Cl : Type)
variable (molesCH₄ molesCl₂ molesCH₃Cl : ℕ)

theorem chloromethane_formation 
  (h₁ : molesCH₄ = 3)
  (h₂ : molesCl₂ = 3)
  (reaction : CH₄ → Cl₂ → CH₃Cl)
  (one_to_one : ∀ (x y : ℕ), x = y → x = y): 
  molesCH₃Cl = 3 :=
by
  sorry

end chloromethane_formation_l69_69103


namespace value_of_D_l69_69519

theorem value_of_D (D : ℤ) (h : 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89) : D = -5 :=
by sorry

end value_of_D_l69_69519


namespace sum_fractions_l69_69115

noncomputable def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

theorem sum_fractions : 
  ∑ k in finset.range 2017, f ((k + 1) / 2018) = 1008.5 :=
by
  sorry

end sum_fractions_l69_69115


namespace integer_values_of_f_l69_69376

noncomputable def f (x : ℝ) : ℝ := (1 + x)^(1/3) + (3 - x)^(1/3)

theorem integer_values_of_f : 
  {x : ℝ | ∃ k : ℤ, f x = k} = {1 + Real.sqrt 5, 1 - Real.sqrt 5, 1 + (10/9) * Real.sqrt 3, 1 - (10/9) * Real.sqrt 3} :=
by
  sorry

end integer_values_of_f_l69_69376


namespace JessieScore_l69_69576

-- Define the conditions as hypotheses
variables (correct_answers : ℕ) (incorrect_answers : ℕ) (unanswered_questions : ℕ)
variables (points_per_correct : ℕ) (points_deducted_per_incorrect : ℤ)

-- Define the values for the specific problem instance
def JessieCondition := correct_answers = 16 ∧ incorrect_answers = 4 ∧ unanswered_questions = 10 ∧
                       points_per_correct = 2 ∧ points_deducted_per_incorrect = -1 / 2

-- Define the statement that Jessie's score is 30 given the conditions
theorem JessieScore (h : JessieCondition correct_answers incorrect_answers unanswered_questions points_per_correct points_deducted_per_incorrect) :
  (correct_answers * points_per_correct : ℤ) + (incorrect_answers * points_deducted_per_incorrect) = 30 :=
by
  sorry

end JessieScore_l69_69576


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69819

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69819


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69614

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69614


namespace inverse_value_exists_l69_69282

noncomputable def f (a x : ℝ) := a^x - 1

theorem inverse_value_exists (a : ℝ) (h : f a 1 = 1) : (f a)⁻¹ 3 = 2 :=
by
  sorry

end inverse_value_exists_l69_69282


namespace frac_left_handed_l69_69373

variable (x : ℕ)

def red_participants := 10 * x
def blue_participants := 5 * x
def total_participants := red_participants x + blue_participants x

def left_handed_red := (1 / 3 : ℚ) * red_participants x
def left_handed_blue := (2 / 3 : ℚ) * blue_participants x
def total_left_handed := left_handed_red x + left_handed_blue x

theorem frac_left_handed :
  total_left_handed x / total_participants x = (4 / 9 : ℚ) := by
  sorry

end frac_left_handed_l69_69373


namespace shirt_selling_price_l69_69232

theorem shirt_selling_price (x : ℝ)
  (cost_price : x = 80)
  (initial_shirts_sold : ∃ s : ℕ, s = 30)
  (profit_per_shirt : ∃ p : ℝ, p = 50)
  (additional_shirts_per_dollar_decrease : ∃ a : ℕ, a = 2)
  (target_daily_profit : ∃ t : ℝ, t = 2000) :
  (x = 105 ∨ x = 120) := 
sorry

end shirt_selling_price_l69_69232


namespace find_remainder_l69_69864

def mod_condition : Prop :=
  (764251 % 31 = 5) ∧
  (1095223 % 31 = 6) ∧
  (1487719 % 31 = 1) ∧
  (263311 % 31 = 0) ∧
  (12097 % 31 = 25) ∧
  (16817 % 31 = 26) ∧
  (23431 % 31 = 0) ∧
  (305643 % 31 = 20)

theorem find_remainder (h : mod_condition) : 
  ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := 
by
  sorry

end find_remainder_l69_69864


namespace four_consecutive_integers_divisible_by_12_l69_69692

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69692


namespace subtract_three_from_binary_l69_69132

theorem subtract_three_from_binary (M : ℕ) (M_binary: M = 0b10110000) : (M - 3) = 0b10101101 := by
  sorry

end subtract_three_from_binary_l69_69132


namespace div_product_four_consecutive_integers_l69_69843

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69843


namespace greatest_divisor_of_four_consecutive_integers_l69_69828

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69828


namespace arithmetic_sequence_problem_l69_69309

theorem arithmetic_sequence_problem
  (a : ℕ → ℚ)
  (h : a 2 + a 4 + a 9 + a 11 = 32) :
  a 6 + a 7 = 16 :=
sorry

end arithmetic_sequence_problem_l69_69309


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69816

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69816


namespace four_consecutive_product_divisible_by_12_l69_69791

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69791


namespace third_derivative_y_l69_69102

noncomputable def y (x : ℝ) : ℝ := x * Real.cos (x^2)

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = (8 * x^4 - 6) * Real.sin (x^2) - 24 * x^2 * Real.cos (x^2) :=
by
  sorry

end third_derivative_y_l69_69102


namespace count_positive_area_triangles_l69_69126

-- Define the grid size
def grid_size : ℕ := 6

-- Defining the main theorem
theorem count_positive_area_triangles :
  (set.univ.powerset.to_finset.filter (λ s : finset (ℤ × ℤ), s.card = 3 ∧ ¬collinear s)).card = 6628 :=
sorry

end count_positive_area_triangles_l69_69126


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69780

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69780


namespace razorback_tshirt_revenue_l69_69452

theorem razorback_tshirt_revenue 
    (total_tshirts : ℕ) (total_money : ℕ) 
    (h1 : total_tshirts = 245) 
    (h2 : total_money = 2205) : 
    (total_money / total_tshirts = 9) := 
by 
    sorry

end razorback_tshirt_revenue_l69_69452


namespace num_fixed_last_two_digits_l69_69411

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits_l69_69411


namespace side_length_of_square_perimeter_of_square_l69_69342

theorem side_length_of_square {d s: ℝ} (h: d = 2 * Real.sqrt 2): s = 2 :=
by
  sorry

theorem perimeter_of_square {s P: ℝ} (h: s = 2): P = 8 :=
by
  sorry

end side_length_of_square_perimeter_of_square_l69_69342


namespace investment_ratio_l69_69549

theorem investment_ratio (total_profit b_profit : ℝ) (a c b : ℝ) :
  total_profit = 150000 ∧ b_profit = 75000 ∧ a / c = 2 ∧ a + b + c = total_profit →
  a / b = 2 / 3 :=
by
  sorry

end investment_ratio_l69_69549


namespace part1_part2_l69_69258

def unitPrices (x : ℕ) (y : ℕ) : Prop :=
  (20 * x = 16 * (y + 20)) ∧ (x = y + 20)

def maxBoxes (a : ℕ) : Prop :=
  ∀ b, (100 * a + 80 * b ≤ 4600) → (a + b = 50)

theorem part1 (x : ℕ) :
  unitPrices x (x - 20) → x = 100 ∧ (x - 20 = 80) :=
by
  sorry

theorem part2 :
  maxBoxes 30 :=
by
  sorry

end part1_part2_l69_69258


namespace largest_square_factor_of_1800_l69_69044

theorem largest_square_factor_of_1800 : 
  ∃ n, n^2 ∣ 1800 ∧ ∀ m, m^2 ∣ 1800 → m^2 ≤ n^2 :=
sorry

end largest_square_factor_of_1800_l69_69044


namespace jack_jill_same_speed_l69_69431

-- Definitions for Jack and Jill's conditions
def jacks_speed (x : ℝ) : ℝ := x^2 - 13*x - 48
def jills_distance (x : ℝ) : ℝ := x^2 - 5*x - 84
def jills_time (x : ℝ) : ℝ := x + 8

-- Theorem stating the same walking speed given the conditions
theorem jack_jill_same_speed (x : ℝ) (h : jacks_speed x = jills_distance x / jills_time x) : 
  jacks_speed x = 6 :=
by
  sorry

end jack_jill_same_speed_l69_69431


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69635

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69635


namespace four_consecutive_product_divisible_by_12_l69_69797

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69797


namespace feet_count_l69_69368

-- We define the basic quantities
def total_heads : ℕ := 50
def num_hens : ℕ := 30
def num_cows : ℕ := total_heads - num_hens
def hens_feet : ℕ := num_hens * 2
def cows_feet : ℕ := num_cows * 4
def total_feet : ℕ := hens_feet + cows_feet

-- The theorem we want to prove
theorem feet_count : total_feet = 140 :=
  by
  sorry

end feet_count_l69_69368


namespace remainder_of_four_m_plus_five_l69_69001

theorem remainder_of_four_m_plus_five (m : ℤ) (h : m % 5 = 3) : (4 * m + 5) % 5 = 2 :=
by
  -- Proof steps would go here
  sorry

end remainder_of_four_m_plus_five_l69_69001


namespace problem_statement_l69_69114

open Complex

noncomputable def z : ℂ := ((1 - I)^2 + 3 * (1 + I)) / (2 - I)

theorem problem_statement :
  z = 1 + I ∧ (∀ (a b : ℝ), (z^2 + a * z + b = 1 - I) → (a = -3 ∧ b = 4)) :=
by
  sorry

end problem_statement_l69_69114


namespace simplify_and_evaluate_l69_69585

theorem simplify_and_evaluate (a b : ℤ) (h₁ : a = -1) (h₂ : b = 3) :
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 :=
by
  sorry

end simplify_and_evaluate_l69_69585


namespace combined_work_rate_l69_69052

theorem combined_work_rate (x_rate y_rate : ℚ) (h1 : x_rate = 1 / 15) (h2 : y_rate = 1 / 45) :
    1 / (x_rate + y_rate) = 11.25 :=
by
  -- Proof goes here
  sorry

end combined_work_rate_l69_69052


namespace x_gt_zero_sufficient_but_not_necessary_l69_69985

theorem x_gt_zero_sufficient_but_not_necessary (x : ℝ): 
  (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬ (x > 0)) → 
  ((x > 0 ↔ x ≠ 0) = false) :=
by
  intro h
  sorry

end x_gt_zero_sufficient_but_not_necessary_l69_69985


namespace factor_complete_polynomial_l69_69101

theorem factor_complete_polynomial :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 =
  (5 * x^2 + 94 * x + 385) * (x^2 - 20 * x + 77) :=
sorry

end factor_complete_polynomial_l69_69101


namespace polynomial_factorization_l69_69099

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l69_69099


namespace seq_periodic_l69_69119

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1/4
  else ite (n > 1) (1 - (1 / (seq (n-1)))) 0 -- handle invalid cases with a default zero

theorem seq_periodic {n : ℕ} (h : seq 1 = 1/4) (h2 : ∀ k ≥ 2, seq k = 1 - (1 / (seq (k-1)))) :
  seq 2014 = 1/4 :=
sorry

end seq_periodic_l69_69119


namespace range_of_m_l69_69545

theorem range_of_m (x y m : ℝ) (h1 : x - 2 * y = 1) (h2 : 2 * x + y = 4 * m) (h3 : x + 3 * y < 6) : m < 7 / 4 :=
sorry

end range_of_m_l69_69545


namespace circle_division_l69_69165

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l69_69165


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69805

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69805


namespace greatest_divisor_of_four_consecutive_integers_l69_69665

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69665


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69631

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69631


namespace dice_probability_green_l69_69882

theorem dice_probability_green :
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  probability = 1 / 2 :=
by
  let total_faces := 6
  let green_faces := 3
  let probability := green_faces / total_faces
  have h : probability = 1 / 2 := by sorry
  exact h

end dice_probability_green_l69_69882


namespace four_consecutive_integers_divisible_by_12_l69_69689

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69689


namespace greatest_divisor_of_four_consecutive_integers_l69_69668

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69668


namespace cody_games_still_has_l69_69087

def initial_games : ℕ := 9
def games_given_away_to_jake : ℕ := 4
def games_given_away_to_sarah : ℕ := 2
def games_bought_over_weekend : ℕ := 3

theorem cody_games_still_has : 
  initial_games - (games_given_away_to_jake + games_given_away_to_sarah) + games_bought_over_weekend = 6 := 
by
  sorry

end cody_games_still_has_l69_69087


namespace greatest_divisor_of_four_consecutive_integers_l69_69659

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69659


namespace div_product_four_consecutive_integers_l69_69849

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69849


namespace chinese_chess_sets_l69_69075

theorem chinese_chess_sets (x y : ℕ) 
  (h1 : 24 * x + 18 * y = 300) 
  (h2 : x + y = 14) : 
  y = 6 := 
sorry

end chinese_chess_sets_l69_69075


namespace smallest_w_l69_69133

theorem smallest_w (w : ℕ) (w_pos : 0 < w) : 
  (∀ n : ℕ, (2^5 ∣ 936 * n) ∧ (3^3 ∣ 936 * n) ∧ (11^2 ∣ 936 * n) ↔ n = w) → w = 4356 :=
sorry

end smallest_w_l69_69133


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69632

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69632


namespace jason_games_planned_last_month_l69_69432

-- Define the conditions
variable (games_planned_this_month : Nat) (games_missed : Nat) (games_attended : Nat)

-- Define what we want to prove
theorem jason_games_planned_last_month (h1 : games_planned_this_month = 11)
                                        (h2 : games_missed = 16)
                                        (h3 : games_attended = 12) :
                                        (games_attended + games_missed - games_planned_this_month = 17) := 
by
  sorry

end jason_games_planned_last_month_l69_69432


namespace circle_regions_division_l69_69158

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l69_69158


namespace geometric_sequence_general_term_l69_69310

theorem geometric_sequence_general_term (a : ℕ → ℕ) (q : ℕ) (h_q : q = 4) (h_sum : a 0 + a 1 + a 2 = 21)
  (h_geo : ∀ n, a (n + 1) = a n * q) : ∀ n, a n = 4 ^ n :=
by {
  sorry
}

end geometric_sequence_general_term_l69_69310


namespace greatest_divisor_four_consecutive_l69_69740

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69740


namespace find_k_l69_69439

theorem find_k : 
  ∃ (k : ℚ), 
    (∃ (x y : ℚ), y = 3 * x + 7 ∧ y = -4 * x + 1) ∧ 
    ∃ (x y : ℚ), y = 3 * x + 7 ∧ y = 2 * x + k ∧ k = 43 / 7 := 
sorry

end find_k_l69_69439


namespace increase_in_lines_l69_69458

variable (L : ℝ)
variable (h1 : L + (1 / 3) * L = 240)

theorem increase_in_lines : (240 - L) = 60 := by
  sorry

end increase_in_lines_l69_69458


namespace greatest_divisor_four_consecutive_integers_l69_69725

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69725


namespace inequality_solution_l69_69336

theorem inequality_solution (x : ℝ) : 
  (∃ (y : ℝ), y = 1 / (3 ^ x) ∧ y * (y - 2) < 15) ↔ x > - (Real.log 5 / Real.log 3) :=
by 
    sorry

end inequality_solution_l69_69336


namespace intercept_condition_l69_69298

theorem intercept_condition (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ x = -c / a ∧ y = -c / b ∧ x = y) → (c = 0 ∨ a = b) :=
by
  sorry

end intercept_condition_l69_69298


namespace total_number_of_seats_l69_69925

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69925


namespace expression_evaluation_l69_69335

theorem expression_evaluation (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (2 * a + Real.sqrt 3) * (2 * a - Real.sqrt 3) - 3 * a * (a - 2) + 3 = -7 :=
by
  sorry

end expression_evaluation_l69_69335


namespace unique_digits_addition_l69_69427

theorem unique_digits_addition :
  ∃ (X Y B M C : ℕ), 
    -- Conditions
    X ≠ 0 ∧ Y ≠ 0 ∧ B ≠ 0 ∧ M ≠ 0 ∧ C ≠ 0 ∧
    X ≠ Y ∧ X ≠ B ∧ X ≠ M ∧ X ≠ C ∧ Y ≠ B ∧ Y ≠ M ∧ Y ≠ C ∧ B ≠ M ∧ B ≠ C ∧ M ≠ C ∧
    -- Addition equation with distinct digits
    (X * 1000 + Y * 100 + 70) + (B * 100 + M * 10 + C) = (B * 1000 + M * 100 + C * 10 + 0) ∧
    -- Correct Answer
    X = 9 ∧ Y = 8 ∧ B = 3 ∧ M = 8 ∧ C = 7 :=
sorry

end unique_digits_addition_l69_69427


namespace perfect_power_transfer_l69_69279

-- Given Conditions
variables {x y z : ℕ}

-- Definition of what it means to be a perfect seventh power
def is_perfect_seventh_power (n : ℕ) :=
  ∃ k : ℕ, n = k^7

-- The proof problem
theorem perfect_power_transfer 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : is_perfect_seventh_power (x^3 * y^5 * z^6)) :
  is_perfect_seventh_power (x^5 * y^6 * z^3) := by
  sorry

end perfect_power_transfer_l69_69279


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69776

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69776


namespace greatest_divisor_of_four_consecutive_integers_l69_69666

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69666


namespace area_enclosed_abs_eq_96_l69_69032

theorem area_enclosed_abs_eq_96 :
  (∃ (S : Set (ℝ × ℝ)), ∀ (x y : ℝ), (x, y) ∈ S ↔ |x| + |3 * y| = 12) →
  (let area := 96 in true) :=
begin
  sorry
end

end area_enclosed_abs_eq_96_l69_69032


namespace cost_of_remaining_shirt_l69_69206

theorem cost_of_remaining_shirt :
  ∀ (shirts total_cost cost_per_shirt remaining_shirt_cost : ℕ),
  shirts = 5 →
  total_cost = 85 →
  cost_per_shirt = 15 →
  (3 * cost_per_shirt) + (2 * remaining_shirt_cost) = total_cost →
  remaining_shirt_cost = 20 :=
by
  intros shirts total_cost cost_per_shirt remaining_shirt_cost
  intros h_shirts h_total h_cost_per_shirt h_equation
  sorry

end cost_of_remaining_shirt_l69_69206


namespace probability_favorite_track_before_eighth_l69_69053

/-- Pete's favorite track is the 8th track on an 11 track CD.
    When the CD is in random mode, what is the probability
    that he will reach his favorite track with fewer than 8 button presses? --/
theorem probability_favorite_track_before_eighth :
  let n := 11 in
  let favorite := 8 in
  (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) + (1 / n) = 7 / 11 :=
by sorry

end probability_favorite_track_before_eighth_l69_69053


namespace inclination_of_line_l69_69974

theorem inclination_of_line (α : ℝ) (h1 : ∃ l : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → y = -x - 1) : α = 135 :=
by
  sorry

end inclination_of_line_l69_69974


namespace four_consecutive_product_divisible_by_12_l69_69792

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69792


namespace binders_required_l69_69365

variables (b1 b2 B1 B2 d1 d2 b3 : ℕ)

def binding_rate_per_binder_per_day : ℚ := B1 / (↑b1 * d1)

def books_per_binder_in_d2_days : ℚ := binding_rate_per_binder_per_day b1 B1 d1 * ↑d2

def binding_rate_for_b2_binders : ℚ := B2 / ↑b2

theorem binders_required (b1 b2 B1 B2 d1 d2 b3 : ℕ)
  (h1 : binding_rate_per_binder_per_day b1 B1 d1 = binding_rate_for_b2_binders b2 B2)
  (h2 : books_per_binder_in_d2_days b1 B1 d1 d2 = binding_rate_for_b2_binders b2 B2) :
  b3 = b2 :=
sorry

end binders_required_l69_69365


namespace product_of_consecutive_integers_l69_69676

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69676


namespace greatest_divisor_four_consecutive_integers_l69_69723

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69723


namespace cube_difference_l69_69111

theorem cube_difference {a b : ℝ} (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 :=
sorry

end cube_difference_l69_69111


namespace find_f3_l69_69201

noncomputable def f : ℝ → ℝ := sorry

theorem find_f3 (h : ∀ x : ℝ, x ≠ 0 → f x - 2 * f (1 / x) = 3 ^ x) : f 3 = -11 :=
sorry

end find_f3_l69_69201


namespace compare_abc_l69_69525

theorem compare_abc (a b c : Real) (h1 : a = Real.sqrt 3) (h2 : b = Real.log 2) (h3 : c = Real.logb 3 (Real.sin (Real.pi / 6))) :
  a > b ∧ b > c :=
by
  sorry

end compare_abc_l69_69525


namespace find_multiple_of_t_l69_69137

theorem find_multiple_of_t (k t x y : ℝ) (h1 : x = 1 - k * t) (h2 : y = 2 * t - 2) :
  t = 0.5 → x = y → k = 4 :=
by
  intros ht hxy
  sorry

end find_multiple_of_t_l69_69137


namespace transformed_center_is_correct_l69_69952

-- Definition for transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

def translate_up (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Given conditions
def initial_center : ℝ × ℝ := (4, -3)
def reflection_center := reflect_x initial_center
def translated_right_center := translate_right reflection_center 5
def final_center := translate_up translated_right_center 3

-- The statement to be proved
theorem transformed_center_is_correct : final_center = (9, 6) :=
by
  sorry

end transformed_center_is_correct_l69_69952


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69779

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69779


namespace fraction_solution_l69_69199

theorem fraction_solution (a : ℕ) (h : a > 0) (h_eq : (a : ℚ) / (a + 45) = 0.75) : a = 135 :=
sorry

end fraction_solution_l69_69199


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69777

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69777


namespace find_coordinates_of_P_l69_69404

theorem find_coordinates_of_P : 
  ∃ P: ℝ × ℝ, 
  (∃ θ: ℝ, 0 ≤ θ ∧ θ ≤ π ∧ P = (3 * Real.cos θ, 4 * Real.sin θ)) ∧ 
  ∃ m: ℝ, m = 1 ∧ P.fst = P.snd ∧ P = (12/5, 12/5) :=
by {
  sorry -- Proof is omitted as per instruction
}

end find_coordinates_of_P_l69_69404


namespace circle_regions_division_l69_69157

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l69_69157


namespace find_a_value_l69_69524

theorem find_a_value (a : ℝ) (A B : Set ℝ) (hA : A = {3, 5}) (hB : B = {x | a * x - 1 = 0}) :
  B ⊆ A → a = 0 ∨ a = 1/3 ∨ a = 1/5 :=
by sorry

end find_a_value_l69_69524


namespace work_problem_l69_69493

theorem work_problem (x : ℝ) (h1 : x > 0) 
                      (h2 : (2 * (1 / 4 + 1 / x) + 2 * (1 / x) = 1)) : 
                      x = 8 := sorry

end work_problem_l69_69493


namespace log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l69_69416

theorem log_one_plus_xsq_lt_xsq_over_one_plus_xsq (x : ℝ) (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 / (1 + x^2) :=
sorry

end log_one_plus_xsq_lt_xsq_over_one_plus_xsq_l69_69416


namespace no_infinite_positive_integer_sequence_l69_69490

theorem no_infinite_positive_integer_sequence (a : ℕ → ℕ) :
  ¬(∀ n, a (n - 1) ^ 2 ≥ 2 * a n * a (n + 2)) :=
sorry

end no_infinite_positive_integer_sequence_l69_69490


namespace square_TU_squared_l69_69337

theorem square_TU_squared (P Q R S T U : ℝ × ℝ)
  (side : ℝ) (RT SU PT QU : ℝ)
  (hpqrs : (P.1 - S.1)^2 + (P.2 - S.2)^2 = side^2 ∧ (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = side^2 ∧ 
            (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = side^2 ∧ (S.1 - R.1)^2 + (S.2 - R.2)^2 = side^2)
  (hRT : (R.1 - T.1)^2 + (R.2 - T.2)^2 = RT^2)
  (hSU : (S.1 - U.1)^2 + (S.2 - U.2)^2 = SU^2)
  (hPT : (P.1 - T.1)^2 + (P.2 - T.2)^2 = PT^2)
  (hQU : (Q.1 - U.1)^2 + (Q.2 - U.2)^2 = QU^2)
  (side_eq_17 : side = 17) (RT_SU_eq_8 : RT = 8) (PT_QU_eq_15 : PT = 15) :
  (T.1 - U.1)^2 + (T.2 - U.2)^2 = 979.5 :=
by
  -- proof to be filled in
  sorry

end square_TU_squared_l69_69337


namespace negation_of_p_l69_69408

def p := ∀ x, x ≤ 0 → Real.exp x ≤ 1

theorem negation_of_p : ¬ p ↔ ∃ x, x ≤ 0 ∧ Real.exp x > 1 := 
by
  sorry

end negation_of_p_l69_69408


namespace faster_train_speed_l69_69602

theorem faster_train_speed
  (length_per_train : ℝ)
  (speed_slower_train : ℝ)
  (passing_time_secs : ℝ)
  (speed_faster_train : ℝ) :
  length_per_train = 80 / 1000 →
  speed_slower_train = 36 →
  passing_time_secs = 36 →
  speed_faster_train = 52 :=
by
  intro h_length_per_train h_speed_slower_train h_passing_time_secs
  -- Skipped steps would go here
  sorry

end faster_train_speed_l69_69602


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69769

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69769


namespace find_difference_l69_69386

theorem find_difference (x0 y0 : ℝ) 
  (h1 : x0^3 - 2023 * x0 = y0^3 - 2023 * y0 + 2020)
  (h2 : x0^2 + x0 * y0 + y0^2 = 2022) : 
  x0 - y0 = -2020 :=
by
  sorry

end find_difference_l69_69386


namespace largest_multiple_of_9_less_than_110_l69_69472

theorem largest_multiple_of_9_less_than_110 : ∃ x, (x < 110 ∧ x % 9 = 0 ∧ ∀ y, (y < 110 ∧ y % 9 = 0) → y ≤ x) ∧ x = 108 :=
by
  sorry

end largest_multiple_of_9_less_than_110_l69_69472


namespace division_problem_l69_69428

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

end division_problem_l69_69428


namespace divisor_of_four_consecutive_integers_l69_69756

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69756


namespace length_of_each_brick_l69_69467

theorem length_of_each_brick (wall_length wall_height wall_thickness : ℝ) (brick_length brick_width brick_height : ℝ) (num_bricks_used : ℝ) 
  (h1 : wall_length = 8) 
  (h2 : wall_height = 6) 
  (h3 : wall_thickness = 0.02) 
  (h4 : brick_length = 0.11) 
  (h5 : brick_width = 0.05) 
  (h6 : brick_height = 0.06) 
  (h7 : num_bricks_used = 2909.090909090909) : 
  brick_length = 0.11 :=
by
  -- variables and assumptions
  have vol_wall : ℝ := wall_length * wall_height * wall_thickness
  have vol_brick : ℝ := brick_length * brick_width * brick_height
  have calc_bricks : ℝ := vol_wall / vol_brick
  -- skipping proof
  sorry

end length_of_each_brick_l69_69467


namespace pentagon_cosine_identity_l69_69488

    variable (A B C D E : Point)
    variable (circle : Circle)

    -- Given conditions
    variable (inscribed : Inscribed circle [A, B, C, D, E])
    variable (AB_eq : AB = 3)
    variable (BC_eq : BC = 3)
    variable (CD_eq : CD = 3)
    variable (DE_eq : DE = 3)
    variable (AE_eq : AE = 2)

    -- Goal: prove the given equation
    theorem pentagon_cosine_identity : 
      (1 - cos (∠ B)) * (1 - cos (∠ ACE)) = 1 / 9 := 
    by
      sorry
    
end pentagon_cosine_identity_l69_69488


namespace find_angle_F_l69_69312

-- Define the angles of the triangle
variables (D E F : ℝ)

-- Define the conditions given in the problem
def angle_conditions (D E F : ℝ) : Prop :=
  (D = 3 * E) ∧ (E = 18) ∧ (D + E + F = 180)

-- The theorem to prove that angle F is 108 degrees
theorem find_angle_F (D E F : ℝ) (h : angle_conditions D E F) : 
  F = 108 :=
by
  -- The proof body is omitted
  sorry

end find_angle_F_l69_69312


namespace div_product_four_consecutive_integers_l69_69838

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69838


namespace tan_add_pi_over_3_l69_69541

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_add_pi_over_3_l69_69541


namespace value_of_y_l69_69415

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 :=
by
  sorry

end value_of_y_l69_69415


namespace largest_perfect_square_factor_of_1800_l69_69040

theorem largest_perfect_square_factor_of_1800 :
  ∃ k, k ∣ 1800 ∧ is_square k ∧ ∀ m, (m ∣ 1800 ∧ is_square m) → m ≤ k :=
begin
  use 900,
  split,
  { -- 900 divides 1800
    rw dvd_iff_mod_eq_zero,
    norm_num,
  },
  split,
  { -- 900 is a perfect square
    exact is_square.mk' 30, -- since 30 * 30 = 900
  },
  { -- 900 is the largest perfect square factor of 1800
    intros m hm,
    rcases hm with ⟨hdvd, hsquare⟩,
    rw dvd_iff_mod_eq_zero at hdvd,
    cases hsquare with n hn,
    rw hn at hdvd ⊢,
    have h : n^2 ∣ 2^3 * 3^2 * 5^2 := by norm_num at hdvd ⊢; sorry,
    sorry
  }
end

end largest_perfect_square_factor_of_1800_l69_69040


namespace greatest_divisor_four_consecutive_l69_69738

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69738


namespace product_of_consecutive_integers_l69_69680

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69680


namespace max_min_value_l69_69436

theorem max_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 12) (h5 : x * y + y * z + z * x = 30) :
  ∃ n : ℝ, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
sorry

end max_min_value_l69_69436


namespace triangle_construction_possible_l69_69254

theorem triangle_construction_possible (r l_alpha k_alpha : ℝ) (h1 : r > 0) (h2 : l_alpha > 0) (h3 : k_alpha > 0) :
  l_alpha^2 < (4 * k_alpha^2 * r^2) / (k_alpha^2 + r^2) :=
sorry

end triangle_construction_possible_l69_69254


namespace total_seats_round_table_l69_69902

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69902


namespace fraction_sum_l69_69128

theorem fraction_sum (x a b : ℕ) (h1 : x = 36 / 99) (h2 : a = 4) (h3 : b = 11) (h4 : Nat.gcd a b = 1) : a + b = 15 :=
by
  sorry

end fraction_sum_l69_69128


namespace CarlaDailyItems_l69_69508

theorem CarlaDailyItems (leaves bugs days : ℕ) 
  (h_leaves : leaves = 30) 
  (h_bugs : bugs = 20) 
  (h_days : days = 10) : 
  (leaves + bugs) / days = 5 := 
by 
  sorry

end CarlaDailyItems_l69_69508


namespace total_seats_round_table_l69_69936

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69936


namespace quadratic_equation_completing_square_l69_69553

theorem quadratic_equation_completing_square :
  ∃ a b c : ℤ, a > 0 ∧ (25 * x^2 + 30 * x - 75 = 0 → (a * x + b)^2 = c) ∧ a + b + c = -58 :=
  sorry

end quadratic_equation_completing_square_l69_69553


namespace largest_perfect_square_factor_of_1800_l69_69042

theorem largest_perfect_square_factor_of_1800 :
  ∃ k : ℕ, k ^ 2 ∣ 1800 ∧ (∀ n : ℕ, n ^ 2 ∣ 1800 → n ^ 2 ≤ k ^ 2) ∧ k ^ 2 = 900 :=
begin
  sorry
end

end largest_perfect_square_factor_of_1800_l69_69042


namespace daughter_age_in_3_years_l69_69058

variable (mother_age_now : ℕ) (gap_years : ℕ) (ratio : ℕ)

theorem daughter_age_in_3_years
  (h1 : mother_age_now = 41) 
  (h2 : gap_years = 5)
  (h3 : ratio = 2) :
  let mother_age_then := mother_age_now - gap_years in
  let daughter_age_then := mother_age_then / ratio in
  let daughter_age_now := daughter_age_then + gap_years in
  let daughter_age_in_3_years := daughter_age_now + 3 in
  daughter_age_in_3_years = 26 :=
  by
    sorry

end daughter_age_in_3_years_l69_69058


namespace range_of_a_for_increasing_l69_69978

noncomputable def f (a x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

theorem range_of_a_for_increasing (a : ℝ) :
  -1 ≤ a ∧ a ≤ 1 ↔ ∀ x y : ℝ, x < y → f a x ≤ f a y :=
sorry

end range_of_a_for_increasing_l69_69978


namespace number_of_triangles_is_correct_l69_69125

def points := Fin 6 × Fin 6

def is_collinear (p1 p2 p3 : points) : Prop :=
  (p2.1 - p1.1) * (p3.2 - p1.2) = (p3.1 - p1.1) * (p2.2 - p1.2)

noncomputable def count_triangles_with_positive_area : Nat :=
  let all_points := Finset.univ.product Finset.univ
  let all_combinations := all_points.powerset.filter (λ s, s.card = 3)
  let valid_triangles := all_combinations.filter (λ s, ¬is_collinear (s.choose 0) (s.choose 1) (s.choose 2))
  valid_triangles.card

theorem number_of_triangles_is_correct :
  count_triangles_with_positive_area = 6804 :=
by
  sorry

end number_of_triangles_is_correct_l69_69125


namespace divisor_of_four_consecutive_integers_l69_69747

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69747


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69634

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69634


namespace main_inequality_l69_69317

noncomputable def b (c : ℝ) : ℝ := (1 + c) / (2 + c)

def f (c : ℝ) (x : ℝ) : ℝ := sorry

lemma f_continuous (c : ℝ) (h_c : 0 < c) : Continuous (f c) := sorry

lemma condition1 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1/2) : 
  b c * f c (2 * x) = f c x := sorry

lemma condition2 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 1/2 ≤ x ∧ x ≤ 1) : 
  f c x = b c + (1 - b c) * f c (2 * x - 1) := sorry

theorem main_inequality (c : ℝ) (h_c : 0 < c) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) → (0 < f c x - x ∧ f c x - x < c) := sorry

end main_inequality_l69_69317


namespace symmetric_point_proof_l69_69311

def Point3D := (ℝ × ℝ × ℝ)

def symmetric_point_yOz (p : Point3D) : Point3D :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetric_point_proof :
  symmetric_point_yOz (1, -2, 3) = (-1, -2, 3) :=
by
  sorry

end symmetric_point_proof_l69_69311


namespace angle_line_plane_l69_69295

theorem angle_line_plane {l α : Type} (θ : ℝ) (h : θ = 150) : 
  ∃ φ : ℝ, φ = 60 := 
by
  -- This part would require the actual proof.
  sorry

end angle_line_plane_l69_69295


namespace find_k_l69_69973

theorem find_k (k : ℤ) (x : ℚ) (h1 : 5 * x + 3 * k = 24) (h2 : 5 * x + 3 = 0) : k = 9 := 
by
  sorry

end find_k_l69_69973


namespace prop1_prop2_l69_69476

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l69_69476


namespace quadratic_equals_binomial_square_l69_69384

theorem quadratic_equals_binomial_square (d : ℝ) : 
  (∃ b : ℝ, (x^2 + 60 * x + d) = (x + b)^2) → d = 900 :=
by
  sorry

end quadratic_equals_binomial_square_l69_69384


namespace arithmetic_seq_problem_l69_69990

theorem arithmetic_seq_problem
  (a : ℕ → ℤ)  -- sequence a_n is an arithmetic sequence
  (h0 : ∃ (a1 d : ℤ), ∀ (n : ℕ), a n = a1 + n * d)  -- exists a1 and d such that a_n = a1 + n * d
  (h1 : a 0 + 3 * a 7 + a 14 = 120) :                -- given a1 + 3a8 + a15 = 120
  3 * a 8 - a 10 = 48 :=                             -- prove 3a9 - a11 = 48
sorry

end arithmetic_seq_problem_l69_69990


namespace four_consecutive_integers_divisible_by_12_l69_69683

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69683


namespace form_square_from_trapezoid_l69_69093

noncomputable def trapezoid_area (a b h : ℝ) : ℝ :=
  (a + b) * h / 2

theorem form_square_from_trapezoid (a b h : ℝ) (trapezoid_area_eq_five : trapezoid_area a b h = 5) :
  ∃ s : ℝ, s^2 = 5 :=
by
  use (Real.sqrt 5)
  sorry

end form_square_from_trapezoid_l69_69093


namespace sale_in_fifth_month_l69_69495

-- Define the sales in the first, second, third, fourth, and sixth months
def a1 : ℕ := 7435
def a2 : ℕ := 7927
def a3 : ℕ := 7855
def a4 : ℕ := 8230
def a6 : ℕ := 5991

-- Define the average sale
def avg_sale : ℕ := 7500

-- Define the number of months
def months : ℕ := 6

-- The total sales required for the average sale to be 7500 over 6 months.
def total_sales : ℕ := avg_sale * months

-- Calculate the sales in the first four months
def sales_first_four_months : ℕ := a1 + a2 + a3 + a4

-- Calculate the total sales for the first four months plus the sixth month.
def sales_first_four_and_sixth : ℕ := sales_first_four_months + a6

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : ∃ a5 : ℕ, total_sales = sales_first_four_and_sixth + a5 ∧ a5 = 7562 :=
by
  sorry


end sale_in_fifth_month_l69_69495


namespace valid_votes_correct_l69_69426

noncomputable def Total_votes : ℕ := 560000
noncomputable def Percentages_received : Fin 4 → ℚ 
| 0 => 0.4
| 1 => 0.35
| 2 => 0.15
| 3 => 0.1

noncomputable def Percentages_invalid : Fin 4 → ℚ 
| 0 => 0.12
| 1 => 0.18
| 2 => 0.25
| 3 => 0.3

noncomputable def Votes_received (i : Fin 4) : ℚ := Total_votes * Percentages_received i

noncomputable def Invalid_votes (i : Fin 4) : ℚ := Votes_received i * Percentages_invalid i

noncomputable def Valid_votes (i : Fin 4) : ℚ := Votes_received i - Invalid_votes i

def A_valid_votes := 197120
def B_valid_votes := 160720
def C_valid_votes := 63000
def D_valid_votes := 39200

theorem valid_votes_correct :
  Valid_votes 0 = A_valid_votes ∧
  Valid_votes 1 = B_valid_votes ∧
  Valid_votes 2 = C_valid_votes ∧
  Valid_votes 3 = D_valid_votes := by
  sorry

end valid_votes_correct_l69_69426


namespace X_independent_iff_independent_from_prefix_l69_69192

variables {Ω : Type*} {X : ℕ → Ω → ℝ}

def independent_system (X : ℕ → Ω → ℝ) : Prop := 
  ∀ s : finset ℕ, ∀ A : Π i ∈ s, set (Ω → ℝ),
  probability_theory.indep_set (λ i, X i) s A 

def independent_from_prefix (X : ℕ → Ω → ℝ) (n : ℕ) : Prop := 
  ∀ A_n B_{n-1}, 
  probability_theory.indep_set (X n) (λ i, X i) (fin.range (n-1)) (λ _, set.univ)

theorem X_independent_iff_independent_from_prefix : 
  independent_system X ↔ ∀ n ≥ 1, independent_from_prefix X n :=
sorry

end X_independent_iff_independent_from_prefix_l69_69192


namespace books_per_shelf_l69_69078

theorem books_per_shelf 
  (initial_books : ℕ) 
  (sold_books : ℕ) 
  (num_shelves : ℕ) 
  (remaining_books : ℕ := initial_books - sold_books) :
  initial_books = 40 → sold_books = 20 → num_shelves = 5 → remaining_books / num_shelves = 4 :=
by
  sorry

end books_per_shelf_l69_69078


namespace problem1_problem2_problem3_l69_69520

theorem problem1 (x : ℤ) (h : 263 - x = 108) : x = 155 :=
by sorry

theorem problem2 (x : ℤ) (h : 25 * x = 1950) : x = 78 :=
by sorry

theorem problem3 (x : ℤ) (h : x / 15 = 64) : x = 960 :=
by sorry

end problem1_problem2_problem3_l69_69520


namespace no_square_from_vertices_of_equilateral_triangles_l69_69015

-- Definitions
def equilateral_triangle_grid (p : ℝ × ℝ) : Prop := 
  ∃ k l : ℤ, p.1 = k * (1 / 2) ∧ p.2 = l * (Real.sqrt 3 / 2)

def form_square_by_vertices (A B C D : ℝ × ℝ) : Prop := 
  (A.1 = 0 ∧ A.2 = 0) ∧ 
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 ∧ 
  (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 = (D.1 - B.1) ^ 2 + (D.2 - B.2) ^ 2 ∧ 
  (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 = (A.1 - D.1) ^ 2 + (A.2 - D.2) ^ 2
  
-- Problem Statement
theorem no_square_from_vertices_of_equilateral_triangles :
  ¬ ∃ (A B C D : ℝ × ℝ), 
    equilateral_triangle_grid A ∧ 
    equilateral_triangle_grid B ∧ 
    equilateral_triangle_grid C ∧ 
    equilateral_triangle_grid D ∧ 
    form_square_by_vertices A B C D :=
by
  sorry

end no_square_from_vertices_of_equilateral_triangles_l69_69015


namespace distance_from_origin_12_5_l69_69424

def distance_from_origin (x y : ℕ) : ℕ := 
  Int.natAbs (Nat.sqrt (x * x + y * y))

theorem distance_from_origin_12_5 : distance_from_origin 12 5 = 13 := by
  sorry

end distance_from_origin_12_5_l69_69424


namespace fifteenth_term_geometric_sequence_l69_69214

theorem fifteenth_term_geometric_sequence :
  let a1 := 5
  let r := (1 : ℝ) / 2
  let fifteenth_term := a1 * r^(14 : ℕ)
  fifteenth_term = (5 : ℝ) / 16384 := by
sorry

end fifteenth_term_geometric_sequence_l69_69214


namespace three_monotonic_intervals_iff_a_lt_zero_l69_69200

-- Definition of the function f
def f (a x : ℝ) : ℝ := a * x^3 + x

-- Definition of the first derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main statement: Prove that f(x) has exactly three monotonic intervals if and only if a < 0.
theorem three_monotonic_intervals_iff_a_lt_zero (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' a x1 = 0 ∧ f' a x2 = 0) ↔ a < 0 :=
by
  sorry

end three_monotonic_intervals_iff_a_lt_zero_l69_69200


namespace minimum_value_a_l69_69980

theorem minimum_value_a (a : ℝ) : (∃ x0 : ℝ, |x0 + 1| + |x0 - 2| ≤ a) → a ≥ 3 :=
by 
  sorry

end minimum_value_a_l69_69980


namespace xiaoming_comprehensive_score_l69_69947

theorem xiaoming_comprehensive_score :
  ∀ (a b c d : ℝ),
  a = 92 → b = 90 → c = 88 → d = 95 →
  (0.4 * a + 0.3 * b + 0.2 * c + 0.1 * d) = 90.9 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  norm_num
  done

end xiaoming_comprehensive_score_l69_69947


namespace greatest_divisor_four_consecutive_integers_l69_69732

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69732


namespace nh4cl_formed_l69_69958

theorem nh4cl_formed :
  (∀ (nh3 hcl nh4cl : ℝ), nh3 = 1 ∧ hcl = 1 → nh3 + hcl = nh4cl → nh4cl = 1) :=
by
  intros nh3 hcl nh4cl
  sorry

end nh4cl_formed_l69_69958


namespace range_of_m_if_neg_proposition_false_l69_69420

theorem range_of_m_if_neg_proposition_false :
  (¬ ∃ x_0 : ℝ, x_0^2 + m * x_0 + 2 * m - 3 < 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
by
  sorry

end range_of_m_if_neg_proposition_false_l69_69420


namespace sum_is_correct_l69_69204

def number : ℕ := 81
def added_number : ℕ := 15
def sum_value (x : ℕ) (y : ℕ) : ℕ := x + y

theorem sum_is_correct : sum_value number added_number = 96 := 
by 
  sorry

end sum_is_correct_l69_69204


namespace complete_square_ratio_l69_69578

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end complete_square_ratio_l69_69578


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69627

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69627


namespace greatest_divisor_four_consecutive_l69_69742

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69742


namespace minimum_routes_A_C_l69_69304

namespace SettlementRoutes

-- Define three settlements A, B, and C
variable (A B C : Type)

-- Assume there are more than one roads connecting each settlement pair directly
variable (k m n : ℕ) -- k: roads between A and B, m: roads between B and C, n: roads between A and C

-- Conditions: Total paths including intermediate nodes
axiom h1 : k + m * n = 34
axiom h2 : m + k * n = 29

-- Theorem: Minimum number of routes connecting A and C is 26
theorem minimum_routes_A_C : ∃ n k m : ℕ, k + m * n = 34 ∧ m + k * n = 29 ∧ n + k * m = 26 := sorry

end SettlementRoutes

end minimum_routes_A_C_l69_69304


namespace problem_equivalent_l69_69962

def modified_op (a b : ℝ) : ℝ := (a + b) ^ 2

theorem problem_equivalent (x y : ℝ) : 
  modified_op ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 := 
by 
  sorry

end problem_equivalent_l69_69962


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69613

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69613


namespace integer_values_abc_l69_69955

theorem integer_values_abc (a b c : ℤ) :
  1 < a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) →
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  sorry

end integer_values_abc_l69_69955


namespace domain_of_h_l69_69603

noncomputable def h (x : ℝ) : ℝ := (5 * x - 2) / (2 * x - 10)

theorem domain_of_h :
  {x : ℝ | 2 * x - 10 ≠ 0} = {x : ℝ | x ≠ 5} :=
by
  sorry

end domain_of_h_l69_69603


namespace tan_theta_l69_69403

theorem tan_theta (θ : ℝ) (x y : ℝ) (hx : x = - (Real.sqrt 3) / 2) (hy : y = 1 / 2) (h_terminal : True) : 
  Real.tan θ = - (Real.sqrt 3) / 3 :=
sorry

end tan_theta_l69_69403


namespace total_seats_at_round_table_l69_69907

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69907


namespace total_turnips_l69_69575

theorem total_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := 
by sorry

end total_turnips_l69_69575


namespace circles_radii_divide_regions_l69_69152

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l69_69152


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69808

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69808


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69821

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69821


namespace probability_heads_and_three_l69_69289

open ProbabilityTheory

-- Define the sample space for the coin flips and the die roll
def coin_flip_space := {HH, HT, TH, TT}
def die_roll_space := {1, 2, 3, 4, 5, 6}

-- Define the event of interest: flipping two heads and rolling a 3
def event := { (HH, 3) }

-- Define the probability measure
def p : finset (string × ℕ) → ℚ := λ s,
  if s = { (HH, 3) } then 1 / 24 else 0

theorem probability_heads_and_three : 
  p { (HH, 3) } = 1 / 24 := 
by
  sorry

end probability_heads_and_three_l69_69289


namespace smallest_three_digit_candy_number_l69_69891

theorem smallest_three_digit_candy_number (n : ℕ) (hn1 : 100 ≤ n) (hn2 : n ≤ 999)
    (h1 : (n + 6) % 9 = 0) (h2 : (n - 9) % 6 = 0) : n = 111 := by
  sorry

end smallest_three_digit_candy_number_l69_69891


namespace mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l69_69950

def molar_mass_carbon : Float := 12.01
def molar_mass_hydrogen : Float := 1.008
def moles_of_nonane : Float := 23.0
def num_carbons_in_nonane : Float := 9.0
def num_hydrogens_in_nonane : Float := 20.0

theorem mass_of_23_moles_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  mass_23_moles = 2950.75 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  have molar_mass_C9H20_val : molar_mass_C9H20 = 128.25 := sorry
  have mass_23_moles_val : mass_23_moles = 2950.75 := sorry
  exact mass_23_moles_val

theorem percentage_composition_C_H_O_in_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  percentage_carbon = 84.27 ∧ percentage_hydrogen = 15.73 ∧ percentage_oxygen = 0 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  have percentage_carbon_val : percentage_carbon = 84.27 := sorry
  have percentage_hydrogen_val : percentage_hydrogen = 15.73 := sorry
  have percentage_oxygen_val : percentage_oxygen = 0 := by rfl
  exact ⟨percentage_carbon_val, percentage_hydrogen_val, percentage_oxygen_val⟩

end mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l69_69950


namespace cylinder_surface_area_l69_69064

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l69_69064


namespace sum_of_first_1996_terms_l69_69018

noncomputable def sequence (n : ℕ) : List ℚ :=
  List.flatten (List.map (λ k => List.range (k + 1) |>.map (λ i => (i + 1 : ℕ) / (k + 1 : ℕ))) (List.range n))

noncomputable def sum_sequence (n : ℕ) : ℚ :=
  (sequence n).take 1996 |>.sum

theorem sum_of_first_1996_terms : sum_sequence 100 > 1022.51 ∧ sum_sequence 100 < 1022.53 := 
by
  sorry

end sum_of_first_1996_terms_l69_69018


namespace carmen_candle_burn_time_l69_69251

theorem carmen_candle_burn_time 
  (burn_time_first_scenario : ℕ)
  (nights_per_candle : ℕ)
  (total_candles_second_scenario : ℕ)
  (total_nights_second_scenario : ℕ)
  (h1 : burn_time_first_scenario = 1)
  (h2 : nights_per_candle = 8)
  (h3 : total_candles_second_scenario = 6)
  (h4 : total_nights_second_scenario = 24) :
  (total_candles_second_scenario * nights_per_candle) / total_nights_second_scenario = 2 :=
by
  sorry

end carmen_candle_burn_time_l69_69251


namespace angle_condition_l69_69515

theorem angle_condition
  {θ : ℝ}
  (h₀ : 0 ≤ θ)
  (h₁ : θ < π)
  (h₂ : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :
  0 < θ ∧ θ < π / 2 :=
by
  sorry

end angle_condition_l69_69515


namespace inequality_solution_range_l69_69270

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℤ, 6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) ∧
  (∃ x1 x2 x3 : ℤ, (x1 = 3 ∧ x2 = 4 ∧ x3 = 5) ∧
   (6 - 3 * (x1 : ℝ) < 0 ∧ 2 * (x1 : ℝ) ≤ a) ∧
   (6 - 3 * (x2 : ℝ) < 0 ∧ 2 * (x2 : ℝ) ≤ a) ∧
   (6 - 3 * (x3 : ℝ) < 0 ∧ 2 * (x3 : ℝ) ≤ a) ∧
   (∀ x : ℤ, (6 - 3 * (x : ℝ) < 0 ∧ 2 * (x : ℝ) ≤ a) → 
     (x = 3 ∨ x = 4 ∨ x = 5)))
  → 10 ≤ a ∧ a < 12 :=
sorry

end inequality_solution_range_l69_69270


namespace exists_line_with_two_colors_l69_69465

open Classical

/-- Given a grid with 1x1 squares where each vertex is painted one of four colors such that each 1x1 square's vertices are all different colors, 
    there exists a line in the grid with nodes of exactly two different colors. -/
theorem exists_line_with_two_colors 
  (A : Type)
  [Inhabited A]
  [DecidableEq A]
  (colors : Finset A) 
  (h_col : colors.card = 4) 
  (grid : ℤ × ℤ → A) 
  (h_diff_colors : ∀ (i j : ℤ), i ≠ j → ∀ (k l : ℤ), grid (i, k) ≠ grid (j, k) ∧ grid (i, l) ≠ grid (i, k)) :
  ∃ line : ℤ → ℤ × ℤ, ∃ a b : A, a ≠ b ∧ ∀ n : ℤ, grid (line n) = a ∨ grid (line n) = b :=
sorry

end exists_line_with_two_colors_l69_69465


namespace yellow_balls_in_bag_l69_69526

open Classical

theorem yellow_balls_in_bag (Y : ℕ) (hY1 : (Y/(Y+2): ℝ) * ((Y-1)/(Y+1): ℝ) = 0.5) : Y = 5 := by
  sorry

end yellow_balls_in_bag_l69_69526


namespace max_original_chess_pieces_l69_69323

theorem max_original_chess_pieces (m n M N : ℕ) (h1 : m ≤ 19) (h2 : n ≤ 19) (h3 : M ≤ 19) (h4 : N ≤ 19) (h5 : M * N = m * n + 45) (h6 : M = m ∨ N = n) : m * n ≤ 285 :=
by
  sorry

end max_original_chess_pieces_l69_69323


namespace greatest_divisor_four_consecutive_l69_69741

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69741


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69703

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69703


namespace greatest_divisor_of_four_consecutive_integers_l69_69827

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69827


namespace linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l69_69048

variable {α : Type*} [Ring α]

def linear_function (a b x : α) : α :=
  a * x + b

def special_case_linear_function (a x : α) : α :=
  a * x

def quadratic_function (a b c x : α) : α :=
  a * x^2 + b * x + c

def special_case_quadratic_function1 (a c x : α) : α :=
  a * x^2 + c

def special_case_quadratic_function2 (a x : α) : α :=
  a * x^2

theorem linear_function_general_form (a b x : α) :
  ∃ y, y = linear_function a b x := by
  sorry

theorem special_case_linear_function_proof (a x : α) :
  ∃ y, y = special_case_linear_function a x := by
  sorry

theorem quadratic_function_general_form (a b c x : α) :
  a ≠ 0 → ∃ y, y = quadratic_function a b c x := by
  sorry

theorem special_case_quadratic_function1_proof (a b c x : α) :
  a ≠ 0 → b = 0 → ∃ y, y = special_case_quadratic_function1 a c x := by
  sorry

theorem special_case_quadratic_function2_proof (a b c x : α) :
  a ≠ 0 → b = 0 → c = 0 → ∃ y, y = special_case_quadratic_function2 a x := by
  sorry

end linear_function_general_form_special_case_linear_function_proof_quadratic_function_general_form_special_case_quadratic_function1_proof_special_case_quadratic_function2_proof_l69_69048


namespace prime_factorization_2006_expr_l69_69264

theorem prime_factorization_2006_expr :
  let a := 2006
  let b := 669
  let c := 1593
  (a^2 * (b + c) - b^2 * (c + a) + c^2 * (a - b)) =
  2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 :=
by
  let a := 2006
  let b := 669
  let c := 1593
  have h1 : 2262 = b + c := by norm_num
  have h2 : 3599 = c + a := by norm_num
  have h3 : 1337 = a - b := by norm_num
  sorry

end prime_factorization_2006_expr_l69_69264


namespace probability_of_same_length_segments_l69_69563

-- Define the conditions of the problem.
def regular_hexagon_segments : list ℕ :=
  [6, 6, 3]  -- 6 sides, 6 shorter diagonals, 3 longer diagonals

def total_segments (segments : list ℕ) : ℕ :=
  segments.sum

def single_segment_probability (n : ℕ) (total_segs : ℕ) : ℕ × ℕ :=
  (n - 1, total_segs - 1)

def combined_probability : ℚ :=
  let sides := 6
      short_diagonals := 6
      long_diagonals := 3
      total_segs := 15
      prob_side := (sides / total_segs) * (5 / (total_segs - 1))
      prob_short_diag := (short_diagonals / total_segs) * (5 / (total_segs - 1))
      prob_long_diag := (long_diagonals / total_segs) * (2 / (total_segs - 1))
  in prob_side + prob_short_diag + prob_long_diag

def expected_probability : ℚ :=
  33 / 105

-- The theorem we need to prove.
theorem probability_of_same_length_segments :
  combined_probability = expected_probability :=
by
  -- We will put the proof steps here.
  sorry

end probability_of_same_length_segments_l69_69563


namespace find_value_of_a_l69_69502

theorem find_value_of_a (a : ℝ) (h: (1 + 3 + 2 + 5 + a) / 5 = 3) : a = 4 :=
by
  sorry

end find_value_of_a_l69_69502


namespace tom_hockey_games_l69_69468

def tom_hockey_games_last_year (games_this_year missed_this_year total_games : Nat) : Nat :=
  total_games - games_this_year

theorem tom_hockey_games :
  ∀ (games_this_year missed_this_year total_games : Nat),
    games_this_year = 4 →
    missed_this_year = 7 →
    total_games = 13 →
    tom_hockey_games_last_year games_this_year total_games = 9 := by
  intros games_this_year missed_this_year total_games h1 h2 h3
  -- The proof steps would go here
  sorry

end tom_hockey_games_l69_69468


namespace div_product_four_consecutive_integers_l69_69839

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69839


namespace percentage_of_500_l69_69220

theorem percentage_of_500 : (110 * 500) / 100 = 550 :=
by
  sorry

end percentage_of_500_l69_69220


namespace number_of_sides_on_die_l69_69060

theorem number_of_sides_on_die (n : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : (∃ k : ℕ, k = 5) → (5 : ℚ) / (n ^ 2 : ℚ) = (5 : ℚ) / (36 : ℚ)) 
  : n = 6 :=
sorry

end number_of_sides_on_die_l69_69060


namespace surface_area_correct_l69_69396

def radius_hemisphere : ℝ := 9
def height_cone : ℝ := 12
def radius_cone_base : ℝ := 9

noncomputable def total_surface_area : ℝ := 
  let base_area : ℝ := radius_hemisphere^2 * Real.pi
  let curved_area_hemisphere : ℝ := 2 * radius_hemisphere^2 * Real.pi
  let slant_height_cone : ℝ := Real.sqrt (radius_cone_base^2 + height_cone^2)
  let lateral_area_cone : ℝ := radius_cone_base * slant_height_cone * Real.pi
  base_area + curved_area_hemisphere + lateral_area_cone

theorem surface_area_correct : total_surface_area = 378 * Real.pi := by
  sorry

end surface_area_correct_l69_69396


namespace daughter_age_in_3_years_l69_69057

theorem daughter_age_in_3_years (mother_age_now : ℕ) (h1 : mother_age_now = 41)
  (h2 : ∃ daughter_age_5_years_ago : ℕ, mother_age_now - 5 = 2 * daughter_age_5_years_ago) :
  ∃ daughter_age_in_3_years : ℕ, daughter_age_in_3_years = 26 :=
by {
  sorry
}

end daughter_age_in_3_years_l69_69057


namespace tan_product_l69_69291

open Real

theorem tan_product (x y : ℝ) 
(h1 : sin x * sin y = 24 / 65) 
(h2 : cos x * cos y = 48 / 65) :
tan x * tan y = 1 / 2 :=
by
  sorry

end tan_product_l69_69291


namespace inverse_proportion_point_l69_69287

theorem inverse_proportion_point (a : ℝ) (h : (a, 7) ∈ {p : ℝ × ℝ | ∃ x y, y = 14 / x ∧ p = (x, y)}) : a = 2 :=
by
  sorry

end inverse_proportion_point_l69_69287


namespace new_person_weight_l69_69484

theorem new_person_weight (avg_inc : Real) (num_persons : Nat) (old_weight new_weight : Real)
  (h1 : avg_inc = 2.5)
  (h2 : num_persons = 8)
  (h3 : old_weight = 40)
  (h4 : num_persons * avg_inc = new_weight - old_weight) :
  new_weight = 60 :=
by
  --proof will be done here
  sorry

end new_person_weight_l69_69484


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69812

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69812


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69607

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69607


namespace total_eyes_correct_l69_69412

-- Conditions
def boys := 21 * 2 + 2 * 1
def girls := 15 * 2 + 3 * 1
def cats := 8 * 2 + 2 * 1
def spiders := 4 * 8 + 1 * 6

-- Total count of eyes
def total_eyes := boys + girls + cats + spiders

theorem total_eyes_correct: total_eyes = 133 :=
by 
  -- Here the proof steps would go, which we are skipping
  sorry

end total_eyes_correct_l69_69412


namespace minimize_f_minimize_f_exact_l69_69865

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 14 * x - 20

-- State the theorem that x = -7 minimizes the function f(x)
theorem minimize_f : ∀ x : ℝ, f x ≥ f (-7) :=
by
  intro x
  unfold f
  sorry

-- An alternative statement could include the exact condition for the minimum value
theorem minimize_f_exact : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ x = -7 :=
by
  use -7
  intro y
  unfold f
  sorry

end minimize_f_minimize_f_exact_l69_69865


namespace greatest_divisor_of_consecutive_product_l69_69855

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69855


namespace greatest_divisor_four_consecutive_l69_69746

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69746


namespace license_plate_count_l69_69498

-- Define the conditions
def num_digits : ℕ := 5
def num_letters : ℕ := 2
def digit_choices : ℕ := 10
def letter_choices : ℕ := 26

-- Define the statement to prove the total number of distinct licenses plates
theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * 2 = 2704000 :=
by
  sorry

end license_plate_count_l69_69498


namespace derivative_at_1_of_f_l69_69393

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1_of_f :
  (deriv f 1) = 2 * Real.log 2 - 3 :=
sorry

end derivative_at_1_of_f_l69_69393


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69760

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69760


namespace tournament_players_l69_69306

theorem tournament_players (n : ℕ) :
  (∃ k : ℕ, k = n + 12 ∧
    -- Exactly one-third of the points earned by each player were earned against the twelve players with the least number of points.
    (2 * (1 / 3 * (n * (n - 1) / 2)) + 2 / 3 * 66 + 66 = (k * (k - 1)) / 2) ∧
    --- Solving the quadratic equation derived
    (n = 4)) → 
    k = 16 :=
by
  sorry

end tournament_players_l69_69306


namespace most_pieces_day_and_maximum_number_of_popular_days_l69_69139

-- Definitions for conditions:
def a_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then 3 * n
else 65 - 2 * n

def S_n (n : ℕ) : ℕ :=
if h : n ≤ 13 then (3 + 3 * n) * n / 2
else 273 + (51 - n) * (n - 13)

-- Propositions to prove:
theorem most_pieces_day_and_maximum :
  ∃ k a_k, (1 ≤ k ∧ k ≤ 31) ∧
           (a_k = a_n k) ∧
           (∀ n, 1 ≤ n ∧ n ≤ 31 → a_n n ≤ a_k) ∧
           k = 13 ∧ a_k = 39 := 
sorry

theorem number_of_popular_days :
  ∃ days_popular,
    (∃ n1, 1 ≤ n1 ∧ n1 ≤ 13 ∧ S_n n1 > 200) ∧
    (∃ n2, 14 ≤ n2 ∧ n2 ≤ 31 ∧ a_n n2 < 20) ∧
    days_popular = (22 - 12 + 1) :=
sorry

end most_pieces_day_and_maximum_number_of_popular_days_l69_69139


namespace abc_eq_bc_l69_69595

theorem abc_eq_bc (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c) 
(h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c :=
by 
  sorry

end abc_eq_bc_l69_69595


namespace area_enclosed_by_graph_l69_69034

theorem area_enclosed_by_graph : 
  (∃ (A : ℝ), A = 96) ↔ (∃ x y : ℝ, abs x + abs(3 * y) = 12) := 
sorry

end area_enclosed_by_graph_l69_69034


namespace total_children_l69_69191

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l69_69191


namespace binomial_mod_prime_eq_floor_l69_69569

-- Define the problem's conditions and goal in Lean.
theorem binomial_mod_prime_eq_floor (n p : ℕ) (hp : Nat.Prime p) : (Nat.choose n p) % p = n / p := by
  sorry

end binomial_mod_prime_eq_floor_l69_69569


namespace prop1_prop2_l69_69477

-- Proposition 1: Prove the contrapositive
theorem prop1 (q : ℝ) (h : ¬(∃ x : ℝ, x^2 + 2 * x + q = 0)) : q ≥ 1 :=
sorry

-- Proposition 2: Prove the contrapositive
theorem prop2 (x y : ℝ) (h : ¬(x = 0 ∧ y = 0)) : x^2 + y^2 ≠ 0 :=
sorry

end prop1_prop2_l69_69477


namespace monomial_addition_l69_69970

-- Definition of a monomial in Lean
def isMonomial (p : ℕ → ℝ) : Prop := ∃ c n, ∀ x, p x = c * x^n

theorem monomial_addition (A : ℕ → ℝ) :
  (isMonomial (fun x => -3 * x + A x)) → isMonomial A :=
sorry

end monomial_addition_l69_69970


namespace reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l69_69203

theorem reciprocal_opposite_of_neg_neg_3_is_neg_one_third : 
  (1 / (-(-3))) = -1 / 3 :=
by
  sorry

end reciprocal_opposite_of_neg_neg_3_is_neg_one_third_l69_69203


namespace total_seats_at_round_table_l69_69910

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69910


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69641

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69641


namespace total_seats_round_table_l69_69935

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69935


namespace eq_proof_l69_69470

noncomputable def S_even : ℚ := 28
noncomputable def S_odd : ℚ := 24

theorem eq_proof : ( (S_even / S_odd - S_odd / S_even) * 2 ) = (13 / 21) :=
by
  sorry

end eq_proof_l69_69470


namespace repeating_decimals_subtraction_l69_69097

def x : Rat := 1 / 3
def y : Rat := 2 / 99

theorem repeating_decimals_subtraction :
  x - y = 31 / 99 :=
sorry

end repeating_decimals_subtraction_l69_69097


namespace conditional_probability_even_given_six_l69_69090

open ProbabilityTheory

-- Define the sample space of rolling a six-sided die twice
def sampleSpace : Set (ℕ × ℕ) := {p | p.1 ∈ {1, 2, 3, 4, 5, 6} ∧ p.2 ∈ {1, 2, 3, 4, 5, 6}}

-- Event A: First roll results in a six
def EventA : Set (ℕ × ℕ) := {p | p.1 = 6}

-- Event B: Sum of the two rolls is even
def EventB : Set (ℕ × ℕ) := {p | (p.1 + p.2) % 2 = 0}

-- Conditional probability P(B|A) = 1/2
theorem conditional_probability_even_given_six :
  P(EventB | EventA) = 1/2 := sorry

end conditional_probability_even_given_six_l69_69090


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69622

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69622


namespace diameter_of_larger_sphere_l69_69592

theorem diameter_of_larger_sphere (r : ℝ) (a b : ℕ) (hr : r = 9)
    (h1 : 3 * (4/3) * π * r^3 = (4/3) * π * ((2 * a * b^(1/3)) / 2)^3) 
    (h2 : ¬∃ c : ℕ, c^3 = b) : a + b = 21 :=
sorry

end diameter_of_larger_sphere_l69_69592


namespace problem_solution_l69_69536

noncomputable def f (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 - p * x + q

theorem problem_solution
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : p > 0)
  (h3 : q > 0)
  (h4 : f a p q = 0)
  (h5 : f b p q = 0)
  (h6 : ∃ k : ℝ, (a = -2 + k ∧ b = -2 - k) ∨ (a = -2 - k ∧ b = -2 + k))
  (h7 : ∃ l : ℝ, (a = -2 * l ∧ b = 4 * l) ∨ (a = 4 * l ∧ b = -2 * l))
  : p + q = 9 :=
sorry

end problem_solution_l69_69536


namespace Eli_saves_more_with_discount_A_l69_69873

-- Define the prices and discounts
def price_book : ℝ := 25
def discount_A (price : ℝ) : ℝ := price * 0.4
def discount_B : ℝ := 5

-- Define the cost calculations:
def cost_with_discount_A (price : ℝ) : ℝ := price + (price - discount_A price)
def cost_with_discount_B (price : ℝ) : ℝ := price + (price - discount_B)

-- Define the savings calculation:
def savings (cost_B : ℝ) (cost_A : ℝ) : ℝ := cost_B - cost_A

-- The main statement to prove:
theorem Eli_saves_more_with_discount_A :
  savings (cost_with_discount_B price_book) (cost_with_discount_A price_book) = 5 :=
by
  sorry

end Eli_saves_more_with_discount_A_l69_69873


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69820

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69820


namespace arithmetic_sum_S11_l69_69143

theorem arithmetic_sum_S11 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith : ∀ n, a (n+1) - a n = d) -- The sequence is arithmetic with common difference d
  (h_sum : S n = n * (a 1 + a n) / 2) -- Sum of the first n terms definition
  (h_condition: a 3 + a 6 + a 9 = 54) :
  S 11 = 198 := 
sorry

end arithmetic_sum_S11_l69_69143


namespace smallest_gcd_qr_l69_69292

theorem smallest_gcd_qr {p q r : ℕ} (hpq : Nat.gcd p q = 300) (hpr : Nat.gcd p r = 450) : 
  ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 150 :=
by
  sorry

end smallest_gcd_qr_l69_69292


namespace hexagon_probability_l69_69559

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l69_69559


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69782

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69782


namespace yellow_tiles_count_l69_69314

theorem yellow_tiles_count
  (total_tiles : ℕ)
  (yellow_tiles : ℕ)
  (blue_tiles : ℕ)
  (purple_tiles : ℕ)
  (white_tiles : ℕ)
  (h1 : total_tiles = 20)
  (h2 : blue_tiles = yellow_tiles + 1)
  (h3 : purple_tiles = 6)
  (h4 : white_tiles = 7)
  (h5 : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  yellow_tiles = 3 :=
by sorry

end yellow_tiles_count_l69_69314


namespace race_total_distance_l69_69303

theorem race_total_distance (D : ℝ) 
  (A_time : D / 20 = D / 25 + 1) 
  (beat_distance : D / 20 * 25 = D + 20) : 
  D = 80 :=
sorry

end race_total_distance_l69_69303


namespace correct_misread_number_l69_69197

theorem correct_misread_number (s : List ℕ) (wrong_avg correct_avg n wrong_num correct_num : ℕ) 
  (h1 : s.length = 10) 
  (h2 : (s.sum) / n = wrong_avg) 
  (h3 : wrong_num = 26) 
  (h4 : correct_avg = 16) 
  (h5 : n = 10) 
  : correct_num = 36 :=
sorry

end correct_misread_number_l69_69197


namespace polynomial_decomposition_l69_69510

theorem polynomial_decomposition :
  (x^3 - 2*x^2 + 3*x + 5) = 11 + 7*(x - 2) + 4*(x - 2)^2 + (x - 2)^3 :=
by sorry

end polynomial_decomposition_l69_69510


namespace mold_growth_problem_l69_69583

/-- Given the conditions:
    - Initial mold spores: 50 at 9:00 a.m.
    - Colony doubles in size every 10 minutes.
    - Time elapsed: 70 minutes from 9:00 a.m. to 10:10 a.m.,

    Prove that the number of mold spores at 10:10 a.m. is 6400 -/
theorem mold_growth_problem : 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  final_population = 6400 :=
by 
  let initial_mold_spores := 50
  let doubling_period_minutes := 10
  let elapsed_minutes := 70
  let doublings := elapsed_minutes / doubling_period_minutes
  let final_population := initial_mold_spores * (2 ^ doublings)
  sorry

end mold_growth_problem_l69_69583


namespace largest_square_factor_of_1800_l69_69045

theorem largest_square_factor_of_1800 : 
  ∃ n, n^2 ∣ 1800 ∧ ∀ m, m^2 ∣ 1800 → m^2 ≤ n^2 :=
sorry

end largest_square_factor_of_1800_l69_69045


namespace roots_condition_l69_69534

theorem roots_condition (m : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = x^2 + 2*(m - 1)*x - 5*m - 2) 
  (h_roots : ∃ x1 x2, x1 < 1 ∧ 1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) : 
  m > 1 := 
by
  sorry

end roots_condition_l69_69534


namespace simplify_expression_l69_69507

theorem simplify_expression (a : Int) : 2 * a - a = a :=
by
  sorry

end simplify_expression_l69_69507


namespace area_of_rhombus_l69_69035

theorem area_of_rhombus : 
  ∀ (x y : ℝ), (|x| + |3 * y| = 12) → 
  (area (x, y) = 96) :=
by sorry

-- Define the area function as used in the context of this problem:
noncomputable def area (p : ℝ × ℝ) : ℝ :=
if |p.1| + |3 * p.2| = 12 then 96
else 0

attribute [simp] abs_zero

end area_of_rhombus_l69_69035


namespace total_foreign_objects_l69_69246

-- Definitions based on the conditions
def burrs := 12
def ticks := 6 * burrs

-- Theorem to prove the total number of foreign objects
theorem total_foreign_objects : burrs + ticks = 84 :=
by
  sorry -- Proof omitted

end total_foreign_objects_l69_69246


namespace episodes_per_season_before_loss_l69_69378

-- Define the given conditions
def initial_total_seasons : ℕ := 12 + 14
def episodes_lost_per_season : ℕ := 2
def remaining_episodes : ℕ := 364
def total_episodes_lost : ℕ := 12 * episodes_lost_per_season + 14 * episodes_lost_per_season
def initial_total_episodes : ℕ := remaining_episodes + total_episodes_lost

-- Define the theorem to prove
theorem episodes_per_season_before_loss : initial_total_episodes / initial_total_seasons = 16 :=
by
  sorry

end episodes_per_season_before_loss_l69_69378


namespace evaluate_expression_l69_69260

noncomputable def expr : ℚ := (3 ^ 512 + 7 ^ 513) ^ 2 - (3 ^ 512 - 7 ^ 513) ^ 2
noncomputable def k : ℚ := 28 * 2.1 ^ 512

theorem evaluate_expression : expr = k * 10 ^ 513 :=
by
  sorry

end evaluate_expression_l69_69260


namespace expected_value_dodecahedral_die_l69_69037

-- Define the faces of the die
def faces : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the scoring rule
def score (n : ℕ) : ℕ :=
  if n ≤ 6 then 2 * n else n

-- The probability of each face
def prob : ℚ := 1 / 12

-- Calculate the expected value
noncomputable def expected_value : ℚ :=
  prob * (score 1 + score 2 + score 3 + score 4 + score 5 + score 6 + 
          score 7 + score 8 + score 9 + score 10 + score 11 + score 12)

-- State the theorem to be proved
theorem expected_value_dodecahedral_die : expected_value = 8.25 := 
  sorry

end expected_value_dodecahedral_die_l69_69037


namespace total_amount_correct_l69_69466

noncomputable def total_amount_collected
    (single_ticket_price : ℕ)
    (couple_ticket_price : ℕ)
    (total_people : ℕ)
    (couple_tickets_sold : ℕ) : ℕ :=
  let single_tickets_sold := total_people - (couple_tickets_sold * 2)
  let amount_from_couple_tickets := couple_tickets_sold * couple_ticket_price
  let amount_from_single_tickets := single_tickets_sold * single_ticket_price
  amount_from_couple_tickets + amount_from_single_tickets

theorem total_amount_correct :
  total_amount_collected 20 35 128 16 = 2480 := by
  sorry

end total_amount_correct_l69_69466


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69811

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69811


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69611

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69611


namespace circle_regions_l69_69154

theorem circle_regions (radii : ℕ) (circles : ℕ) (regions : ℕ) :
  radii = 16 → circles = 10 → regions = 11 * 16 → regions = 176 :=
by
  intros h_radii h_circles h_regions
  rw [h_radii, h_circles] at h_regions
  exact h_regions

end circle_regions_l69_69154


namespace only_book_A_l69_69361

theorem only_book_A (purchasedBoth : ℕ) (purchasedOnlyB : ℕ) (purchasedA : ℕ) (purchasedB : ℕ) 
  (h1 : purchasedBoth = 500)
  (h2 : 2 * purchasedOnlyB = purchasedBoth)
  (h3 : purchasedA = 2 * purchasedB)
  (h4 : purchasedB = purchasedOnlyB + purchasedBoth) :
  purchasedA - purchasedBoth = 1000 :=
by
  sorry

end only_book_A_l69_69361


namespace possible_values_of_a_l69_69286

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem possible_values_of_a 
    (a b c : ℝ) 
    (h1 : f a b c (Real.pi / 2) = 1) 
    (h2 : f a b c Real.pi = 1) 
    (h3 : ∀ x : ℝ, |f a b c x| ≤ 2) :
    4 - 3 * Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end possible_values_of_a_l69_69286


namespace sequence_sum_l69_69971

-- Definitions for the sequences
def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

-- The theorem we need to prove
theorem sequence_sum : a (b 1) + a (b 2) + a (b 3) + a (b 4) = 19 := by
  sorry

end sequence_sum_l69_69971


namespace chipmunk_acorns_l69_69269

-- Define the conditions and goal for the proof
theorem chipmunk_acorns :
  ∃ x : ℕ, (∀ h_c h_s : ℕ, h_c = h_s + 4 → 3 * h_c = x ∧ 4 * h_s = x) → x = 48 :=
by {
  -- We assume the problem conditions as given
  sorry
}

end chipmunk_acorns_l69_69269


namespace total_number_of_seats_l69_69923

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69923


namespace circle_region_count_l69_69161

-- Definitions of the conditions
def has_16_radii (circle : Type) [IsCircle circle] : Prop :=
  ∃ r : Radii, r.card = 16

def has_10_concentric_circles (circle : Type) [IsCircle circle] : Prop :=
  ∃ c : ConcentricCircles, c.card = 10

-- Theorem statement: Given the conditions, the circle is divided into 176 regions
theorem circle_region_count (circle : Type) [IsCircle circle]
  (h_radii : has_16_radii circle)
  (h_concentric : has_10_concentric_circles circle) :
  num_regions circle = 176 := 
sorry

end circle_region_count_l69_69161


namespace greatest_possible_integer_l69_69441

theorem greatest_possible_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 1) (h3 : ∃ l : ℕ, n = 10 * l - 4) : n = 86 := 
sorry

end greatest_possible_integer_l69_69441


namespace card_probability_is_correct_l69_69208

section CardProbability

open ProbabilityTheory

-- Definitions for the question conditions
def total_cards := 52
def spades_count := 13
def hearts_count := 13
def kings_count := 4
def non_king_spades := 12
def non_king_hearts := 12
def king_spades := 1

-- Correct answer definition as rational number
def correct_answer : ℚ := 17 / 3683

-- The probability calculation as described in the problem statement
def calculate_probability : ℚ :=
  (non_king_spades / total_cards) * (non_king_hearts / (total_cards - 1)) * (kings_count / (total_cards - 2)) +
  (king_spades / total_cards) * (non_king_hearts / (total_cards - 1)) * ((kings_count - 1) / (total_cards - 2))

-- The theorem stating that the calculated probability is equal to the correct answer
theorem card_probability_is_correct :
  calculate_probability = correct_answer :=
by
  -- skipping the proof steps
  sorry

end CardProbability

end card_probability_is_correct_l69_69208


namespace four_consecutive_integers_divisible_by_12_l69_69682

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69682


namespace quadratic_coefficients_l69_69092

theorem quadratic_coefficients :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) → ∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 10 ∧ a * x^2 + b * x + c = 0 := by
  intros x h
  use 1, -3, 10
  sorry

end quadratic_coefficients_l69_69092


namespace circle_regions_division_l69_69156

theorem circle_regions_division (radii : ℕ) (con_circles : ℕ)
  (h1 : radii = 16) (h2 : con_circles = 10) :
  radii * (con_circles + 1) = 176 := 
by
  -- placeholder for proof
  sorry

end circle_regions_division_l69_69156


namespace mean_of_three_l69_69533

theorem mean_of_three (x y z a : ℝ)
  (h₁ : (x + y) / 2 = 5)
  (h₂ : (y + z) / 2 = 9)
  (h₃ : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 :=
by
  sorry

end mean_of_three_l69_69533


namespace smallest_number_with_property_l69_69377

theorem smallest_number_with_property: 
  ∃ (N : ℕ), N = 25 ∧ (∀ (x : ℕ) (h : N = x + (x / 5)), N ≤ x) := 
  sorry

end smallest_number_with_property_l69_69377


namespace sum_area_of_R_eq_20_l69_69497

noncomputable def sum_m_n : ℝ := 
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  let m := 20
  let n := 12 * Real.sqrt 2
  m + n

theorem sum_area_of_R_eq_20 :
  let s := 4 + 2 * Real.sqrt 2
  let total_area := s ^ 2
  let small_square_area := 4
  let given_rectangle_area := 4 * Real.sqrt 2
  let area_R := total_area - (small_square_area + given_rectangle_area)
  area_R = 20 + 12 * Real.sqrt 2 :=
by
  sorry

end sum_area_of_R_eq_20_l69_69497


namespace regression_decrease_by_5_l69_69397

theorem regression_decrease_by_5 (x y : ℝ) (h : y = 2 - 2.5 * x) :
  y = 2 - 2.5 * (x + 2) → y ≠ 2 - 2.5 * x - 5 :=
by sorry

end regression_decrease_by_5_l69_69397


namespace jason_borrowed_amount_l69_69554

def earning_per_six_hours : ℤ :=
  2 + 4 + 6 + 2 + 4 + 6

def total_hours_worked : ℤ :=
  48

def cycle_length : ℤ :=
  6

def total_cycles : ℤ :=
  total_hours_worked / cycle_length

def total_amount_borrowed : ℤ :=
  total_cycles * earning_per_six_hours

theorem jason_borrowed_amount : total_amount_borrowed = 192 :=
  by
    -- Here we use the definition and conditions to prove the equivalence
    -- of the calculation to the problem statement.
    sorry

end jason_borrowed_amount_l69_69554


namespace geometry_problem_l69_69998

open EuclideanGeometry

variables {A B C O H M H' A' : Point}

-- Assume all given conditions
variables (circumcenter_circle : Circle)
variables (H : orthocenter_triangle A B C)
variables (M : midpoint B C)
variables (A' : diametrically_opposite_point A circumcenter_circle)
variables (H' : reflection_point H B C)

theorem geometry_problem :
  (midpoint M H A') ∧
  (on_circumcircle H' circumcenter_circle) ∧
  (symmetric_perpendicular_bisector H' A' B C) ∧
  (symmetric_angle_bisector AO AH angle_BAC) :=
by sorry

end geometry_problem_l69_69998


namespace regions_divided_by_radii_circles_l69_69168

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l69_69168


namespace total_seats_round_table_l69_69900

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69900


namespace circle_divided_into_regions_l69_69149

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l69_69149


namespace truck_capacities_transportation_plan_l69_69884

-- Definitions of given conditions
def A_truck_capacity (x y : ℕ) : Prop := x + 2 * y = 50
def B_truck_capacity (x y : ℕ) : Prop := 5 * x + 4 * y = 160
def total_transport_cost (m n : ℕ) : ℕ := 500 * m + 400 * n
def most_cost_effective_plan (m n cost : ℕ) : Prop := 
  m + 2 * n = 10 ∧ (20 * m + 15 * n = 190) ∧ cost = total_transport_cost m n ∧ cost = 4800

-- Proving the capacities of trucks A and B
theorem truck_capacities : 
  ∃ x y : ℕ, A_truck_capacity x y ∧ B_truck_capacity x y ∧ x = 20 ∧ y = 15 := 
sorry

-- Proving the most cost-effective transportation plan
theorem transportation_plan : 
  ∃ m n cost, (total_transport_cost m n = cost) ∧ most_cost_effective_plan m n cost := 
sorry

end truck_capacities_transportation_plan_l69_69884


namespace union_of_sets_l69_69177

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_sets :
  A ∪ B = {x | 3 < x ∧ x ≤ 10} :=
by
  sorry

end union_of_sets_l69_69177


namespace arrange_6_books_l69_69872

theorem arrange_6_books :
  Nat.factorial 6 = 720 :=
by
  sorry

end arrange_6_books_l69_69872


namespace C_should_pay_correct_amount_l69_69504

def A_oxen_months : ℕ := 10 * 7
def B_oxen_months : ℕ := 12 * 5
def C_oxen_months : ℕ := 15 * 3
def D_oxen_months : ℕ := 20 * 6

def total_rent : ℚ := 225

def C_share_of_rent : ℚ :=
  total_rent * (C_oxen_months : ℚ) / (A_oxen_months + B_oxen_months + C_oxen_months + D_oxen_months)

theorem C_should_pay_correct_amount : C_share_of_rent = 225 * (45 : ℚ) / 295 := by
  sorry

end C_should_pay_correct_amount_l69_69504


namespace inequality_k_l69_69459

variable {R : Type} [LinearOrderedField R] [Nontrivial R]

theorem inequality_k (x y z : R) (k : ℕ) (h : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) 
  (hineq : (1/x) + (1/y) + (1/z) ≥ x + y + z) :
  (1/x^k) + (1/y^k) + (1/z^k) ≥ x^k + y^k + z^k :=
sorry

end inequality_k_l69_69459


namespace greatest_divisor_of_consecutive_product_l69_69861

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69861


namespace polynomial_expression_value_l69_69319

theorem polynomial_expression_value
  (p q r s : ℂ)
  (h1 : p + q + r + s = 0)
  (h2 : p*q + p*r + p*s + q*r + q*s + r*s = -1)
  (h3 : p*q*r + p*q*s + p*r*s + q*r*s = -1)
  (h4 : p*q*r*s = 2) :
  p*(q - r)^2 + q*(r - s)^2 + r*(s - p)^2 + s*(p - q)^2 = -6 :=
by sorry

end polynomial_expression_value_l69_69319


namespace question_condition_l69_69537

def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
  (1 - 2 * x) * (x + 1) < 0 → x > 1 / 2 ∨ x < -1

theorem question_condition
(x : ℝ) : sufficient_but_not_necessary_condition x := sorry

end question_condition_l69_69537


namespace percentage_profit_is_35_l69_69071

-- Define the conditions
def initial_cost_price : ℝ := 100
def markup_percentage : ℝ := 0.5
def discount_percentage : ℝ := 0.1
def marked_price : ℝ := initial_cost_price * (1 + markup_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)

-- Define the statement/proof problem
theorem percentage_profit_is_35 :
  (selling_price - initial_cost_price) / initial_cost_price * 100 = 35 := by 
  sorry

end percentage_profit_is_35_l69_69071


namespace divisor_of_four_consecutive_integers_l69_69758

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69758


namespace four_consecutive_product_divisible_by_12_l69_69788

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69788


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69695

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69695


namespace king_lancelot_seats_38_l69_69916

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69916


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69783

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69783


namespace greatest_divisor_of_consecutive_product_l69_69853

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69853


namespace converse_of_squared_positive_is_negative_l69_69006

theorem converse_of_squared_positive_is_negative (x : ℝ) :
  (∀ x : ℝ, x < 0 → x^2 > 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) := by
sorry

end converse_of_squared_positive_is_negative_l69_69006


namespace adults_in_each_group_l69_69945

theorem adults_in_each_group (A : ℕ) :
  (∃ n : ℕ, n >= 17 ∧ n * 15 = 255) →
  (∃ m : ℕ, m * A = 255 ∧ m >= 17) →
  A = 15 :=
by
  intros h_child_groups h_adult_groups
  -- Use sorry to skip the proof
  sorry

end adults_in_each_group_l69_69945


namespace necessary_and_sufficient_l69_69273

theorem necessary_and_sufficient (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0 → ab < (a + b) / 2 ^ 2) 
  ∧ (ab < (a + b) / 2 ^ 2 → a > 0 ∧ b > 0)) := 
sorry

end necessary_and_sufficient_l69_69273


namespace alia_markers_l69_69890

theorem alia_markers (S A a : ℕ) (h1 : S = 60) (h2 : A = S / 3) (h3 : a = 2 * A) : a = 40 :=
by
  -- Proof omitted
  sorry

end alia_markers_l69_69890


namespace greatest_divisor_four_consecutive_l69_69735

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69735


namespace regions_divided_by_radii_circles_l69_69170

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l69_69170


namespace probability_of_summer_and_autumn_l69_69589

theorem probability_of_summer_and_autumn :
  let stamps := { "Spring Begins", "Summer Begins", "Autumn Equinox", "Great Cold" }
  let draws := (stamps.powerset.filter (fun s => s.card = 2)).toList
  let favorable := draws.count (fun s => "Summer Begins" ∈ s ∧ "Autumn Equinox" ∈ s)
  (favorable : ℚ) / draws.length = 1 / 6 := by
  sorry

end probability_of_summer_and_autumn_l69_69589


namespace order_of_x_y_z_l69_69982

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- Conditions
axiom h1 : 0.9 < x
axiom h2 : x < 1.0
axiom h3 : y = x^x
axiom h4 : z = x^(x^x)

-- Theorem to be proved
theorem order_of_x_y_z (h1 : 0.9 < x) (h2 : x < 1.0) (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y :=
by
  sorry

end order_of_x_y_z_l69_69982


namespace circle_divided_into_regions_l69_69147

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l69_69147


namespace probability_of_same_length_segments_l69_69562

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l69_69562


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69638

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69638


namespace sum_first_n_terms_l69_69972

noncomputable def a_n (n : ℕ) : ℝ :=
if n = 1 then 1 else 2 * a_n (n - 1)

noncomputable def S_n (n : ℕ) : ℝ :=
2 * a_n n - 1

noncomputable def b_n (n : ℕ) : ℝ :=
1 + Real.log (a_n n) / Real.log 2

noncomputable def T_n (n : ℕ) : ℝ :=
∑ k in Finset.range n, 1 / (b_n k * b_n (k + 1))

theorem sum_first_n_terms (n : ℕ) : T_n n = n / (n + 1) := by
  sorry

end sum_first_n_terms_l69_69972


namespace min_value_arith_seq_l69_69409

theorem min_value_arith_seq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + c = 2 * b) :
  (a + c) / b + b / (a + c) ≥ 5 / 2 := 
sorry

end min_value_arith_seq_l69_69409


namespace positive_difference_l69_69205

theorem positive_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : y - x = 9 :=
sorry

end positive_difference_l69_69205


namespace parallelogram_area_l69_69387

theorem parallelogram_area (d : ℝ) (h : ℝ) (α : ℝ) (h_d : d = 30) (h_h : h = 20) : 
  ∃ A : ℝ, A = d * h ∧ A = 600 :=
by
  sorry

end parallelogram_area_l69_69387


namespace circle_divided_into_regions_l69_69148

/-- 
  Given a circle with 16 radii and 10 concentric circles, the total number
  of regions the radii and circles divide the circle into is 176.
-/
theorem circle_divided_into_regions :
  ∀ (radii : ℕ) (concentric_circles : ℕ), 
  radii = 16 → concentric_circles = 10 → 
  let regions := (concentric_circles + 1) * radii
  in regions = 176 :=
by
  intros radii concentric_circles h1 h2
  let regions := (concentric_circles + 1) * radii
  rw [h1, h2]
  have : regions = (10 + 1) * 16, by rw [h1, h2]
  sorry

end circle_divided_into_regions_l69_69148


namespace probability_of_same_length_segments_l69_69561

noncomputable def probability_same_length {S : Finset (Finset ℝ)} 
  (hexagon_sides : Finset ℝ) (longer_diagonals : Finset ℝ) (shorter_diagonals : Finset ℝ)
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) : 
  ℕ := sorry

theorem probability_of_same_length_segments {S : Finset (Finset ℝ)}
  {hexagon_sides longer_diagonals shorter_diagonals : Finset ℝ} 
  (h1 : hexagon_sides.card = 6)
  (h2 : longer_diagonals.card = 6) 
  (h3 : shorter_diagonals.card = 3)
  (hS : S = hexagon_sides ∪ longer_diagonals ∪ shorter_diagonals)
  (hS_length : S.card = 15) :
  probability_same_length hexagon_sides longer_diagonals shorter_diagonals h1 h2 h3 hS hS_length = 33 / 105 := 
begin
  sorry
end

end probability_of_same_length_segments_l69_69561


namespace king_arthur_round_table_seats_l69_69895

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69895


namespace mayoral_election_l69_69480

theorem mayoral_election :
  ∀ (X Y Z : ℕ), (X = Y + (Y / 2)) → (Y = Z - (2 * Z / 5)) → (Z = 25000) → X = 22500 :=
by
  intros X Y Z h1 h2 h3
  -- Proof here, not necessary for the task
  sorry

end mayoral_election_l69_69480


namespace maximize_village_value_l69_69371

theorem maximize_village_value :
  ∃ (x y z : ℕ), 
  x + y + z = 20 ∧ 
  2 * x + 3 * y + 4 * z = 50 ∧ 
  (∀ x' y' z' : ℕ, 
      x' + y' + z' = 20 → 2 * x' + 3 * y' + 4 * z' = 50 → 
      (1.2 * x + 1.5 * y + 1.2 * z : ℝ) ≥ (1.2 * x' + 1.5 * y' + 1.2 * z' : ℝ)) ∧ 
  x = 10 ∧ y = 10 ∧ z = 0 := by 
  sorry

end maximize_village_value_l69_69371


namespace angle_alpha_range_l69_69528

/-- Given point P (tan α, sin α - cos α) is in the first quadrant, 
and 0 ≤ α ≤ 2π, then the range of values for angle α is (π/4, π/2) ∪ (π, 5π/4). -/
theorem angle_alpha_range (α : ℝ) 
  (h0 : 0 ≤ α) (h1 : α ≤ 2 * Real.pi) 
  (h2 : Real.tan α > 0) (h3 : Real.sin α - Real.cos α > 0) : 
  (Real.pi / 4 < α ∧ α < Real.pi / 2) ∨ 
  (Real.pi < α ∧ α < 5 * Real.pi / 4) :=
sorry

end angle_alpha_range_l69_69528


namespace total_quantity_before_adding_water_l69_69880

variable (x : ℚ)
variable (milk water : ℚ)
variable (added_water : ℚ)

-- Mixture contains milk and water in the ratio 3:2
def initial_ratio (milk water : ℚ) : Prop := milk / water = 3 / 2

-- Adding 10 liters of water
def added_amount : ℚ := 10

-- New ratio of milk to water becomes 2:3 after adding 10 liters of water
def new_ratio (milk water : ℚ) (added_water : ℚ) : Prop :=
  milk / (water + added_water) = 2 / 3

theorem total_quantity_before_adding_water
  (h_ratio : initial_ratio milk water)
  (h_added : added_water = 10)
  (h_new_ratio : new_ratio milk water added_water) :
  milk + water = 20 :=
by
  sorry

end total_quantity_before_adding_water_l69_69880


namespace problem_proof_l69_69392

variable (a b c : ℝ)
noncomputable def a_def : ℝ := Real.exp 0.2
noncomputable def b_def : ℝ := Real.sin 1.2
noncomputable def c_def : ℝ := 1 + Real.log 1.2

theorem problem_proof (ha : a = a_def) (hb : b = b_def) (hc : c = c_def) : b < c ∧ c < a :=
by
  have ha_val : a = Real.exp 0.2 := ha
  have hb_val : b = Real.sin 1.2 := hb
  have hc_val : c = 1 + Real.log 1.2 := hc
  sorry

end problem_proof_l69_69392


namespace find_integer_n_l69_69346

theorem find_integer_n :
  ∃ n : ℤ, 
    50 ≤ n ∧ n ≤ 120 ∧ (n % 5 = 0) ∧ (n % 6 = 3) ∧ (n % 7 = 4) ∧ n = 165 :=
by
  sorry

end find_integer_n_l69_69346


namespace greatest_divisor_of_four_consecutive_integers_l69_69829

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69829


namespace division_identity_l69_69259

theorem division_identity : 45 / 0.05 = 900 :=
by
  sorry

end division_identity_l69_69259


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69716

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69716


namespace find_days_A_alone_works_l69_69070

-- Given conditions
def A_is_twice_as_fast_as_B (a b : ℕ) : Prop := a = b / 2
def together_complete_in_12_days (a b : ℕ) : Prop := (1 / b + 1 / a) = 1 / 12

-- We need to prove that A alone can finish the work in 18 days.
def A_alone_in_18_days (a : ℕ) : Prop := a = 18

theorem find_days_A_alone_works :
  ∃ (a b : ℕ), A_is_twice_as_fast_as_B a b ∧ together_complete_in_12_days a b ∧ A_alone_in_18_days a :=
sorry

end find_days_A_alone_works_l69_69070


namespace initial_violet_balloons_l69_69019

-- Defining the conditions
def violet_balloons_given_by_tom : ℕ := 16
def violet_balloons_left_with_tom : ℕ := 14

-- The statement to prove
theorem initial_violet_balloons (initial_balloons : ℕ) :
  initial_balloons = violet_balloons_given_by_tom + violet_balloons_left_with_tom :=
sorry

end initial_violet_balloons_l69_69019


namespace recurring_fraction_sum_l69_69129

theorem recurring_fraction_sum (a b : ℕ) (h : 0.36̅ = ↑a / ↑b) (gcd_ab : Nat.gcd a b = 1) : a + b = 15 :=
sorry

end recurring_fraction_sum_l69_69129


namespace power_set_card_greater_l69_69451

open Set

variables {A : Type*} (α : ℕ) [Fintype A] (hA : Fintype.card A = α)

theorem power_set_card_greater (h : Fintype.card A = α) :
  2 ^ α > α :=
sorry

end power_set_card_greater_l69_69451


namespace ordered_triples_lcm_sum_zero_l69_69381

theorem ordered_triples_lcm_sum_zero :
  ∀ (x y z : ℕ), 
    (0 < x) → 
    (0 < y) → 
    (0 < z) → 
    Nat.lcm x y = 180 →
    Nat.lcm x z = 450 →
    Nat.lcm y z = 600 →
    x + y + z = 120 →
    false := 
by
  intros x y z hx hy hz hxy hxz hyz hs
  sorry

end ordered_triples_lcm_sum_zero_l69_69381


namespace hexagon_same_length_probability_l69_69566

theorem hexagon_same_length_probability :
  let S : Finset (String) := { 
    "side1", "side2", "side3", "side4", "side5", "side6",
    "short_diagonal1", "short_diagonal2", "short_diagonal3", 
    "short_diagonal4", "short_diagonal5", "short_diagonal6",
    "long_diagonal1", "long_diagonal2", "long_diagonal3"
  } in
  let side_count := 6 in
  let short_diagonal_count := 6 in
  let long_diagonal_count := 3 in
  let total_count := side_count + short_diagonal_count + long_diagonal_count in
  let same_length_pairs := 
    (side_count * (side_count - 1) 
     + short_diagonal_count * (short_diagonal_count - 1)
     + long_diagonal_count * (long_diagonal_count - 1)) / 2 in -- number of ways to pick 2 same-length segments
  let total_pairs := (total_count * (total_count - 1)) / 2 in -- total ways to pick any 2 segments
  (same_length_pairs : ℚ) / total_pairs = 11/35 :=
by
  sorry

end hexagon_same_length_probability_l69_69566


namespace function_passes_through_point_l69_69473

theorem function_passes_through_point (a : ℝ) (x y : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (x = 1 ∧ y = 4) ↔ (y = a^(x-1) + 3) :=
sorry

end function_passes_through_point_l69_69473


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69809

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69809


namespace simplify_and_evaluate_expression_l69_69584

theorem simplify_and_evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 2 → -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l69_69584


namespace terminal_side_of_angle_l69_69529

theorem terminal_side_of_angle (θ : Real) (h_cos : Real.cos θ < 0) (h_tan : Real.tan θ > 0) :
  θ ∈ {φ : Real | π < φ ∧ φ < 3 * π / 2} :=
sorry

end terminal_side_of_angle_l69_69529


namespace markdown_calculation_l69_69095

noncomputable def markdown_percentage (P S : ℝ) (h_inc : P = S * 1.1494) : ℝ :=
  1 - (1 / 1.1494)

theorem markdown_calculation (P S : ℝ) (h_sale : S = P * (1 - markdown_percentage P S sorry / 100)) (h_inc : P = S * 1.1494) :
  markdown_percentage P S h_inc = 12.99 := 
sorry

end markdown_calculation_l69_69095


namespace toy_store_revenue_fraction_l69_69236

theorem toy_store_revenue_fraction (N D J : ℝ) 
  (h1 : J = N / 3) 
  (h2 : D = 3.75 * (N + J) / 2) : 
  (N / D) = 2 / 5 :=
by sorry

end toy_store_revenue_fraction_l69_69236


namespace area_of_region_inside_circle_outside_rectangle_l69_69141

theorem area_of_region_inside_circle_outside_rectangle
  (EF FH : ℝ)
  (hEF : EF = 6)
  (hFH : FH = 5)
  (r : ℝ)
  (h_radius : r = (EF^2 + FH^2).sqrt) :
  π * r^2 - EF * FH = 61 * π - 30 :=
by
  sorry

end area_of_region_inside_circle_outside_rectangle_l69_69141


namespace probability_of_black_ball_l69_69422

theorem probability_of_black_ball 
  (p_red : ℝ)
  (p_white : ℝ)
  (h_red : p_red = 0.43)
  (h_white : p_white = 0.27)
  : (1 - p_red - p_white) = 0.3 :=
by 
  sorry

end probability_of_black_ball_l69_69422


namespace greatest_divisor_of_consecutive_product_l69_69852

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69852


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69785

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69785


namespace divisor_of_product_of_four_consecutive_integers_l69_69650

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69650


namespace cosine_product_l69_69486

-- Definitions for the conditions of the problem
variable (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables (circle : Set A) (inscribed_pentagon : Set A)
variables (AB BC CD DE AE : ℝ) (cosB cosACE : ℝ)

-- Conditions
axiom pentagon_inscribed_in_circle : inscribed_pentagon ⊆ circle
axiom AB_eq_3 : AB = 3
axiom BC_eq_3 : BC = 3
axiom CD_eq_3 : CD = 3
axiom DE_eq_3 : DE = 3
axiom AE_eq_2 : AE = 2

-- Theorem statement
theorem cosine_product :
  (1 - cosB) * (1 - cosACE) = (1 / 9) := 
sorry

end cosine_product_l69_69486


namespace total_seats_round_table_l69_69899

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end total_seats_round_table_l69_69899


namespace hexagon_probability_l69_69560

theorem hexagon_probability :
  let S := (6 + 9) in
  let total_segments := 15 in
  let probability_side_to_side := (5 / 14 : ℚ) in
  let probability_diagonal_to_diagonal := (4 / 7 : ℚ) in
  let probability_side_first := (6 / 15 : ℚ) in
  let probability_diagonal_first := (9 / 15 : ℚ) in
  let total_probability := (probability_side_first * probability_side_to_side) +
                            (probability_diagonal_first * probability_diagonal_to_diagonal)
  in
  total_probability = (17 / 35 : ℚ) :=
by 
  sorry

end hexagon_probability_l69_69560


namespace greatest_divisor_of_four_consecutive_integers_l69_69830

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69830


namespace range_of_p_l69_69544

noncomputable def f (x p : ℝ) : ℝ := x - p/x + p/2

theorem range_of_p (p : ℝ) :
  (∀ x : ℝ, 1 < x → (1 + p / x^2) > 0) → p ≥ -1 :=
by
  intro h
  sorry

end range_of_p_l69_69544


namespace dice_probability_l69_69180

theorem dice_probability :
  let prob_roll_less_than_four := 3 / 6
  let prob_roll_even := 3 / 6
  let prob_roll_greater_than_four := 2 / 6
  prob_roll_less_than_four * prob_roll_even * prob_roll_greater_than_four = 1 / 12 :=
by
  sorry

end dice_probability_l69_69180


namespace percentage_markup_l69_69594

/--
The owner of a furniture shop charges his customer a certain percentage more than the cost price.
A customer paid Rs. 3000 for a computer table, and the cost price of the computer table was Rs. 2500.
Prove that the percentage markup on the cost price is 20%.
-/
theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 3000) (h₂ : cost_price = 2500) :
  ((selling_price - cost_price) / cost_price) * 100 = 20 :=
by
  -- proof omitted
  sorry

end percentage_markup_l69_69594


namespace bus_distance_covered_l69_69207

theorem bus_distance_covered (speedTrain speedCar speedBus distanceBus : ℝ) (h1 : speedTrain / speedCar = 16 / 15)
                            (h2 : speedBus = (3 / 4) * speedTrain) (h3 : 450 / 6 = speedCar) (h4 : distanceBus = 8 * speedBus) :
                            distanceBus = 480 :=
by
  sorry

end bus_distance_covered_l69_69207


namespace factor_expression_l69_69956

theorem factor_expression (x : ℝ) : 72 * x ^ 5 - 162 * x ^ 9 = -18 * x ^ 5 * (9 * x ^ 4 - 4) :=
by
  sorry

end factor_expression_l69_69956


namespace wilsons_theorem_l69_69184

theorem wilsons_theorem (p : ℕ) (hp : Nat.Prime p) : (Nat.factorial (p - 1)) % p = p - 1 :=
by
  sorry

end wilsons_theorem_l69_69184


namespace div_product_four_consecutive_integers_l69_69847

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69847


namespace axis_of_symmetry_shifted_cos_l69_69297

noncomputable def shifted_cos_axis_symmetry (x : ℝ) : Prop :=
  ∃ k : ℤ, x = k * (Real.pi / 2) - (Real.pi / 12)

theorem axis_of_symmetry_shifted_cos :
  shifted_cos_axis_symmetry x :=
sorry

end axis_of_symmetry_shifted_cos_l69_69297


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69615

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69615


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69620

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69620


namespace sequence_general_term_formula_l69_69010

-- Definitions based on conditions
def alternating_sign (n : ℕ) : ℤ := (-1) ^ n
def arithmetic_sequence (n : ℕ) : ℤ := 4 * n - 3

-- Definition for the general term formula
def general_term (n : ℕ) : ℤ := alternating_sign n * arithmetic_sequence n

-- Theorem stating that the given sequence's general term formula is a_n = (-1)^n * (4n - 3)
theorem sequence_general_term_formula (n : ℕ) : general_term n = (-1) ^ n * (4 * n - 3) :=
by
  -- Proof logic will go here
  sorry

end sequence_general_term_formula_l69_69010


namespace cone_sphere_ratio_l69_69881

theorem cone_sphere_ratio (r h : ℝ) (h_r_ne_zero : r ≠ 0)
  (h_vol_cone : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := 
by
  sorry

end cone_sphere_ratio_l69_69881


namespace divisor_of_product_of_four_consecutive_integers_l69_69646

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69646


namespace conference_games_l69_69077

theorem conference_games (teams_per_division : ℕ) (divisions : ℕ) 
  (intradivision_games_per_team : ℕ) (interdivision_games_per_team : ℕ) 
  (total_teams : ℕ) (total_games : ℕ) : 
  total_teams = teams_per_division * divisions →
  intradivision_games_per_team = (teams_per_division - 1) * 2 →
  interdivision_games_per_team = teams_per_division →
  total_games = (total_teams * (intradivision_games_per_team + interdivision_games_per_team)) / 2 →
  total_games = 133 :=
by
  intros
  sorry

end conference_games_l69_69077


namespace tolya_is_older_by_either_4_or_22_years_l69_69991

-- Definitions of the problem conditions
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def kolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2013

def tolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2014

-- The problem statement
theorem tolya_is_older_by_either_4_or_22_years (k_birth t_birth : ℕ) 
  (hk : kolya_conditions k_birth) (ht : tolya_conditions t_birth) :
  t_birth - k_birth = 4 ∨ t_birth - k_birth = 22 :=
sorry

end tolya_is_older_by_either_4_or_22_years_l69_69991


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69699

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69699


namespace correct_average_l69_69483

theorem correct_average
  (incorrect_avg : ℝ)
  (incorrect_num correct_num : ℝ)
  (n : ℕ)
  (h1 : incorrect_avg = 16)
  (h2 : incorrect_num = 26)
  (h3 : correct_num = 46)
  (h4 : n = 10) :
  (incorrect_avg * n - incorrect_num + correct_num) / n = 18 :=
sorry

end correct_average_l69_69483


namespace camila_bikes_more_l69_69302

-- Definitions based on conditions
def camila_speed : ℝ := 15
def daniel_speed_initial : ℝ := 15
def daniel_speed_after_3hours : ℝ := 10
def biking_time : ℝ := 6
def time_before_decrease : ℝ := 3
def time_after_decrease : ℝ := biking_time - time_before_decrease

def distance_camila := camila_speed * biking_time
def distance_daniel := (daniel_speed_initial * time_before_decrease) + (daniel_speed_after_3hours * time_after_decrease)

-- The statement to prove: Camila has biked 15 more miles than Daniel
theorem camila_bikes_more : distance_camila - distance_daniel = 15 := 
by
  sorry

end camila_bikes_more_l69_69302


namespace percentage_reduction_is_20_l69_69076

def original_employees : ℝ := 243.75
def reduced_employees : ℝ := 195

theorem percentage_reduction_is_20 :
  (original_employees - reduced_employees) / original_employees * 100 = 20 := 
  sorry

end percentage_reduction_is_20_l69_69076


namespace hall_area_l69_69240

theorem hall_area (L W : ℝ) 
  (h1 : W = (1/2) * L)
  (h2 : L - W = 8) : 
  L * W = 128 := 
  sorry

end hall_area_l69_69240


namespace complex_quadrant_l69_69413

theorem complex_quadrant (i : ℂ) (h_imag : i = Complex.I) :
  let z := (1 + i)⁻¹
  z.re > 0 ∧ z.im < 0 :=
by
  sorry

end complex_quadrant_l69_69413


namespace greatest_divisor_of_four_consecutive_integers_l69_69663

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69663


namespace find_A_from_eq_l69_69217

theorem find_A_from_eq (A : ℕ) (h : 10 - A = 6) : A = 4 :=
by
  sorry

end find_A_from_eq_l69_69217


namespace greatest_divisor_four_consecutive_l69_69734

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69734


namespace div_product_four_consecutive_integers_l69_69844

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69844


namespace engineering_students_pass_percentage_l69_69425

theorem engineering_students_pass_percentage :
  let num_male_students := 120
  let num_female_students := 100
  let perc_male_eng_students := 0.25
  let perc_female_eng_students := 0.20
  let perc_male_eng_pass := 0.20
  let perc_female_eng_pass := 0.25
  
  let num_male_eng_students := num_male_students * perc_male_eng_students
  let num_female_eng_students := num_female_students * perc_female_eng_students
  
  let num_male_eng_pass := num_male_eng_students * perc_male_eng_pass
  let num_female_eng_pass := num_female_eng_students * perc_female_eng_pass
  
  let total_eng_students := num_male_eng_students + num_female_eng_students
  let total_eng_pass := num_male_eng_pass + num_female_eng_pass
  
  (total_eng_pass / total_eng_students) * 100 = 22 :=
by
  sorry

end engineering_students_pass_percentage_l69_69425


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69771

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69771


namespace propositions_using_logical_connectives_l69_69344

-- Define each of the propositions.
def prop1 := "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def prop2 := "Multiples of 10 are definitely multiples of 5."
def prop3 := "A trapezoid is not a rectangle."
def prop4 := "The solutions to the equation x^2 = 1 are x = ± 1."

-- Define logical connectives usage.
def uses_and (s : String) : Prop := 
  s = "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def uses_not (s : String) : Prop := 
  s = "A trapezoid is not a rectangle."
def uses_or (s : String) : Prop := 
  s = "The solutions to the equation x^2 = 1 are x = ± 1."

-- The lean theorem stating the propositions that use logical connectives
theorem propositions_using_logical_connectives :
  (uses_and prop1) ∧ (¬ uses_and prop2) ∧ (uses_not prop3) ∧ (uses_or prop4) := 
by
  sorry

end propositions_using_logical_connectives_l69_69344


namespace sequence_sum_correct_l69_69385

theorem sequence_sum_correct :
  ∀ (r x y : ℝ),
  (x = 128 * r) →
  (y = x * r) →
  (2 * r = 1 / 2) →
  (x + y = 40) :=
by
  intros r x y hx hy hr
  sorry

end sequence_sum_correct_l69_69385


namespace greatest_divisor_of_consecutive_product_l69_69858

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69858


namespace smallest_n_gcd_l69_69354

theorem smallest_n_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 2) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 2) > 1 → m ≥ n) ↔ n = 19 :=
by
  sorry

end smallest_n_gcd_l69_69354


namespace greatest_divisor_of_four_consecutive_integers_l69_69836

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69836


namespace king_lancelot_seats_38_l69_69913

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69913


namespace greatest_divisor_four_consecutive_integers_l69_69729

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69729


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69719

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69719


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69625

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69625


namespace product_of_consecutive_integers_l69_69671

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69671


namespace area_enclosed_by_abs_linear_eq_l69_69021

theorem area_enclosed_by_abs_linear_eq (x y : ℝ) :
  |x| + |3 * y| = 12 → (enclosure_area := 4 * (1 / 2 * 12 * 4)) = 96 := by
  sorry

end area_enclosed_by_abs_linear_eq_l69_69021


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69700

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69700


namespace intersection_of_A_and_B_l69_69278

open Set

def set_A := {x : ℕ | |x| < 3}
def set_B := {x : ℤ | -2 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : (set_A : Set ℤ) ∩ set_B = {0, 1} := by
  sorry

end intersection_of_A_and_B_l69_69278


namespace largest_fraction_l69_69357

theorem largest_fraction (a b c d e : ℚ) (h₀ : a = 3/7) (h₁ : b = 4/9) (h₂ : c = 17/35) 
  (h₃ : d = 100/201) (h₄ : e = 151/301) : 
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end largest_fraction_l69_69357


namespace polynomial_factorization_l69_69098

theorem polynomial_factorization :
  5 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 4 * x^2 = 5 * x^4 + 180 * x^3 + 1431 * x^2 + 4900 * x + 5159 :=
by sorry

end polynomial_factorization_l69_69098


namespace other_root_zero_l69_69400

theorem other_root_zero (b : ℝ) (x : ℝ) (hx_root : x^2 + b * x = 0) (h_x_eq_minus_two : x = -2) : 
  (0 : ℝ) = 0 :=
by
  sorry

end other_root_zero_l69_69400


namespace sum_three_smallest_m_l69_69340

theorem sum_three_smallest_m :
  (∃ a m, 
    (a - 2 + a + a + 2) / 3 = 7 
    ∧ m % 4 = 3 
    ∧ m ≠ 5 ∧ m ≠ 7 ∧ m ≠ 9 
    ∧ (5 + 7 + 9 + m) % 4 = 0 
    ∧ m > 0) 
  → 3 + 11 + 15 = 29 :=
sorry

end sum_three_smallest_m_l69_69340


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69640

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69640


namespace product_of_consecutive_integers_l69_69681

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69681


namespace divisor_of_four_consecutive_integers_l69_69759

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69759


namespace fraction_order_l69_69221

theorem fraction_order:
  let frac1 := (21 : ℚ) / 17
  let frac2 := (22 : ℚ) / 19
  let frac3 := (18 : ℚ) / 15
  let frac4 := (20 : ℚ) / 16
  frac2 < frac3 ∧ frac3 < frac1 ∧ frac1 < frac4 := 
sorry

end fraction_order_l69_69221


namespace checkerboard_cover_max_tiles_l69_69225

theorem checkerboard_cover_max_tiles :
  ∃ k m : ℕ, (15 * 36 = 49 * k + 25 * m) ∧
    ∀ k' m' : ℕ, (15 * 36 = 49 * k' + 25 * m') → (k + m) ≥ (k' + m') :=
begin
  sorry
end

end checkerboard_cover_max_tiles_l69_69225


namespace complex_eq_solution_l69_69391

theorem complex_eq_solution (x y : ℝ) (i : ℂ) (h : (2 * x - 1) + i = y - (3 - y) * i) : 
  x = 5 / 2 ∧ y = 4 :=
  sorry

end complex_eq_solution_l69_69391


namespace probability_even_sum_l69_69138

theorem probability_even_sum :
  let x_set := {1, 2, 3, 4, 5}
  let y_set := {7, 8, 9, 10}
  let even x := x % 2 = 0
  let odd x := x % 2 = 1
  let prob_even_x := 2 / 5
  let prob_odd_x := 3 / 5
  let prob_even_y := 1 / 2
  let prob_odd_y := 1 / 2
  in
  (prob_even_x * prob_even_y + prob_odd_x * prob_odd_y) = 1 / 2 :=
by {
  sorry
}

end probability_even_sum_l69_69138


namespace buyers_cake_and_muffin_l69_69234

theorem buyers_cake_and_muffin (total_buyers cake_buyers muffin_buyers neither_prob : ℕ) :
  total_buyers = 100 →
  cake_buyers = 50 →
  muffin_buyers = 40 →
  neither_prob = 26 →
  (cake_buyers + muffin_buyers - neither_prob) = 74 →
  90 - cake_buyers - muffin_buyers = neither_prob :=
by
  sorry

end buyers_cake_and_muffin_l69_69234


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69605

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69605


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69824

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69824


namespace probability_heads_l69_69874

variable (p : ℝ)
variable (h1 : 0 ≤ p)
variable (h2 : p ≤ 1)
variable (h3 : p * (1 - p) ^ 4 = 0.03125)

theorem probability_heads :
  p = 0.5 :=
sorry

end probability_heads_l69_69874


namespace sum_of_sequences_l69_69227

theorem sum_of_sequences :
  (1 + 11 + 21 + 31 + 41) + (9 + 19 + 29 + 39 + 49) = 250 := 
by 
  sorry

end sum_of_sequences_l69_69227


namespace cylinder_surface_area_l69_69063

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l69_69063


namespace hyperbola_eccentricity_correct_l69_69117

open Real

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
    let PF1 := (12 * a / 5)
    let PF2 := PF1 - 2 * a
    let c := (2 * sqrt 37 * a / 5)
    sqrt (1 + (b^2 / a^2))  -- Assuming the geometric properties hold, the eccentricity should match
-- Lean function to verify the result
def verify_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop :=
    hyperbola_eccentricity a b ha hb = sqrt 37 / 5

-- Statement to be verified
theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : a > 0) (hb : b > 0) :
    verify_eccentricity a b ha hb := sorry

end hyperbola_eccentricity_correct_l69_69117


namespace cylinder_surface_area_l69_69065

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end cylinder_surface_area_l69_69065


namespace fermat_little_theorem_variant_l69_69414

theorem fermat_little_theorem_variant (p : ℕ) (m : ℤ) [hp : Fact (Nat.Prime p)] : 
  (m ^ p - m) % p = 0 :=
sorry

end fermat_little_theorem_variant_l69_69414


namespace max_quad_int_solutions_l69_69521

theorem max_quad_int_solutions :
  ∃ (a b c : ℤ), (∀ n : ℤ, n ∈ {0, 1, 2, -3, 4}) ∧
  ∀ (p : ℤ → ℤ), 
    p(x) = ax^2 + bx + c →
      ∃ n, p(n) = p(n^2) :=
begin
  sorry
end

end max_quad_int_solutions_l69_69521


namespace divisor_of_four_consecutive_integers_l69_69750

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69750


namespace total_number_of_seats_l69_69921

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69921


namespace find_number_l69_69366

-- Define the given conditions and statement as Lean types
theorem find_number (x : ℝ) :
  (0.3 * x > 0.6 * 50 + 30) -> x = 200 :=
by
  -- Proof here
  sorry

end find_number_l69_69366


namespace greatest_divisor_of_four_consecutive_integers_l69_69832

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69832


namespace problem_1_problem_2_l69_69121

open Real

noncomputable def minimum_value (x y z : ℝ) : ℝ :=
  1 / x + 1 / y + 1 / z

theorem problem_1 (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h_sum : x + 2 * y + 3 * z = 1) :
  minimum_value x y z = 6 + 2 * sqrt 2 + 2 * sqrt 3 + 2 * sqrt 6 :=
sorry

theorem problem_2 (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h_sum : x + 2 * y + 3 * z = 1) :
  x^2 + y^2 + z^2 ≥ 1 / 14 :=
sorry

end problem_1_problem_2_l69_69121


namespace totalNameLengths_l69_69174

-- Definitions of the lengths of names
def JonathanNameLength := 8 + 10
def YoungerSisterNameLength := 5 + 10
def OlderBrotherNameLength := 6 + 10
def YoungestSisterNameLength := 4 + 15

-- Statement to prove
theorem totalNameLengths :
  JonathanNameLength + YoungerSisterNameLength + OlderBrotherNameLength + YoungestSisterNameLength = 68 :=
by
  sorry -- no proof required

end totalNameLengths_l69_69174


namespace deepak_age_l69_69946

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 5 / 7)
  (h2 : A + 6 = 36) :
  D = 42 :=
by sorry

end deepak_age_l69_69946


namespace greatest_possible_value_x_y_l69_69501

noncomputable def max_x_y : ℕ :=
  let s1 := 150
  let s2 := 210
  let s3 := 270
  let s4 := 330
  (3 * (s3 + s4) - (s1 + s2 + s3 + s4))

theorem greatest_possible_value_x_y :
  max_x_y = 840 := by
  sorry

end greatest_possible_value_x_y_l69_69501


namespace greatest_divisor_four_consecutive_l69_69743

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69743


namespace circles_ordered_by_radius_l69_69253

def circle_radii_ordered (rA rB rC : ℝ) : Prop :=
  rA < rC ∧ rC < rB

theorem circles_ordered_by_radius :
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  circle_radii_ordered rA rB rC :=
by
  intros
  let rA := 2
  let CB := 10 * Real.pi
  let AC := 16 * Real.pi
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  show circle_radii_ordered rA rB rC
  sorry

end circles_ordered_by_radius_l69_69253


namespace relativ_prime_and_divisible_exists_l69_69446

theorem relativ_prime_and_divisible_exists
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  ∃ r s : ℕ, Nat.gcd r s = 1 ∧ 0 < r ∧ 0 < s ∧ c ∣ (a * r + b * s) :=
by
  sorry

end relativ_prime_and_divisible_exists_l69_69446


namespace fraction_white_surface_area_l69_69877

/-- A 4-inch cube is constructed from 64 smaller cubes, each with 1-inch edges.
   48 of these smaller cubes are colored red and 16 are colored white.
   Prove that if the 4-inch cube is constructed to have the smallest possible white surface area showing,
   the fraction of the white surface area is 1/12. -/
theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let white_cubes := 16
  let exposed_white_surface_area := 8
  (exposed_white_surface_area / total_surface_area) = (1 / 12) := 
  sorry

end fraction_white_surface_area_l69_69877


namespace propositionA_necessary_but_not_sufficient_for_propositionB_l69_69320

-- Definitions for propositions and conditions
def propositionA (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0
def propositionB (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement for the necessary but not sufficient condition
theorem propositionA_necessary_but_not_sufficient_for_propositionB (a : ℝ) :
  (propositionA a) → (¬ propositionB a) ∧ (propositionB a → propositionA a) :=
by
  sorry

end propositionA_necessary_but_not_sufficient_for_propositionB_l69_69320


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69714

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69714


namespace bonus_points_amount_l69_69573

def points_per_10_dollars : ℕ := 50

def beef_price : ℕ := 11
def beef_quantity : ℕ := 3

def fruits_vegetables_price : ℕ := 4
def fruits_vegetables_quantity : ℕ := 8

def spices_price : ℕ := 6
def spices_quantity : ℕ := 3

def other_groceries_total : ℕ := 37

def total_points : ℕ := 850

def total_spent : ℕ :=
  (beef_price * beef_quantity) +
  (fruits_vegetables_price * fruits_vegetables_quantity) +
  (spices_price * spices_quantity) +
  other_groceries_total

def points_from_spending : ℕ :=
  (total_spent / 10) * points_per_10_dollars

theorem bonus_points_amount :
  total_spent > 100 → total_points - points_from_spending = 250 :=
by
  sorry

end bonus_points_amount_l69_69573


namespace swapped_digit_number_l69_69237

theorem swapped_digit_number (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  10 * b + a = new_number :=
sorry

end swapped_digit_number_l69_69237


namespace range_of_S_l69_69134

theorem range_of_S (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) (S : ℝ) (hS : S = 3 * x^2 - 2 * y^2) :
  -2 / 3 < S ∧ S ≤ 3 / 2 :=
sorry

end range_of_S_l69_69134


namespace sequence_term_l69_69276

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  2 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

theorem sequence_term (m n : ℕ) (h : n < m) : 
  let Sn := geometric_sum n
  let Sn_plus_1 := geometric_sum (n + 1)
  Sn - Sn_plus_1 = -(1 / 2 ^ (n - 1)) := sorry

end sequence_term_l69_69276


namespace greatest_divisor_four_consecutive_integers_l69_69721

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69721


namespace find_weight_of_a_l69_69004

theorem find_weight_of_a (A B C D E : ℕ) 
  (h1 : A + B + C = 3 * 84)
  (h2 : A + B + C + D = 4 * 80)
  (h3 : E = D + 3)
  (h4 : B + C + D + E = 4 * 79) : 
  A = 75 := by
  sorry

end find_weight_of_a_l69_69004


namespace sin_alpha_l69_69968

theorem sin_alpha (α : ℝ) (hα : 0 < α ∧ α < π) (hcos : Real.cos (π + α) = 3 / 5) :
  Real.sin α = 4 / 5 :=
sorry

end sin_alpha_l69_69968


namespace solve_for_a_l69_69290

theorem solve_for_a (x a : ℝ) (h : 3 * x + 2 * a = 3) (hx : x = 5) : a = -6 :=
by
  sorry

end solve_for_a_l69_69290


namespace product_of_consecutive_integers_l69_69669

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69669


namespace calculate_bmw_sales_and_revenue_l69_69066

variable (total_cars : ℕ) (percentage_ford percentage_toyota percentage_nissan percentage_audi : ℕ) (avg_price_bmw : ℕ)
variable (h_total_cars : total_cars = 300) (h_percentage_ford : percentage_ford = 10)
variable (h_percentage_toyota : percentage_toyota = 25) (h_percentage_nissan : percentage_nissan = 20)
variable (h_percentage_audi : percentage_audi = 15) (h_avg_price_bmw : avg_price_bmw = 35000)

theorem calculate_bmw_sales_and_revenue :
  let percentage_non_bmw := percentage_ford + percentage_toyota + percentage_nissan + percentage_audi
  let percentage_bmw := 100 - percentage_non_bmw
  let number_bmw := total_cars * percentage_bmw / 100
  let total_revenue := number_bmw * avg_price_bmw
  (number_bmw = 90) ∧ (total_revenue = 3150000) := by
  -- Definitions are taken from conditions and used directly in the theorem statement
  sorry

end calculate_bmw_sales_and_revenue_l69_69066


namespace always_true_inequality_l69_69866

theorem always_true_inequality (a b : ℝ) : a^2 + b^2 ≥ -2 * a * b :=
by
  sorry

end always_true_inequality_l69_69866


namespace price_cashews_l69_69886

noncomputable def price_per_pound_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ) : ℝ := 
  (price_mixed_nuts_per_pound * weight_mixed_nuts - price_peanuts_per_pound * weight_peanuts) / weight_cashews

open Real

theorem price_cashews 
  (price_mixed_nuts_per_pound : ℝ) 
  (weight_mixed_nuts : ℕ) 
  (weight_peanuts : ℕ) 
  (price_peanuts_per_pound : ℝ) 
  (weight_cashews : ℕ)
  (h1 : price_mixed_nuts_per_pound = 2.50) 
  (h2 : weight_mixed_nuts = 100) 
  (h3 : weight_peanuts = 40) 
  (h4 : price_peanuts_per_pound = 3.50) 
  (h5 : weight_cashews = 60) : 
  price_per_pound_cashews price_mixed_nuts_per_pound weight_mixed_nuts weight_peanuts price_peanuts_per_pound weight_cashews = 11 / 6 := by 
  sorry

end price_cashews_l69_69886


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69823

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69823


namespace probability_div_by_3_l69_69223

-- Define the set of prime digits
def prime_digits : set ℕ := {2, 3, 5, 7}

-- Define the set of two-digit numbers with both digits prime
def two_digit_primes : set ℕ := {n | ∃ d1 d2, d1 ∈ prime_digits ∧ d2 ∈ prime_digits ∧ n = 10 * d1 + d2}

-- Define the set of two-digit numbers in two_digit_primes that are divisible by 3
def divisible_by_3 : set ℕ := {n ∈ two_digit_primes | (n % 3 = 0)}

-- Define the total number of two-digit prime numbers
def total_count : ℕ := 16

-- Define the count of numbers divisible by 3
def count_div_3 : ℕ := 5

-- Define the probability
def probability (num favorable total : ℕ) : ℚ := favorable / total

-- The theorem
theorem probability_div_by_3 : probability count_div_3 total_count = 5 / 16 := by
  sorry

end probability_div_by_3_l69_69223


namespace greatest_divisor_four_consecutive_l69_69739

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69739


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69768

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69768


namespace complex_seventh_root_identity_l69_69435

open Complex

theorem complex_seventh_root_identity (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 :=
by
  sorry

end complex_seventh_root_identity_l69_69435


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69709

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69709


namespace total_amount_spent_l69_69255

-- Definitions for the conditions
def cost_magazine : ℝ := 0.85
def cost_pencil : ℝ := 0.50
def coupon_discount : ℝ := 0.35

-- The main theorem to prove
theorem total_amount_spent : cost_magazine + cost_pencil - coupon_discount = 1.00 := by
  sorry

end total_amount_spent_l69_69255


namespace find_selling_price_l69_69230

-- Define the basic parameters
def cost := 80
def s0 := 30
def profit0 := 50
def desired_profit := 2000

-- Additional shirts sold per price reduction
def add_shirts (p : ℕ) := 2 * p

-- Number of shirts sold given selling price x
def num_shirts (x : ℕ) := 290 - 2 * x

-- Profit equation
def profit_equation (x : ℕ) := (x - cost) * num_shirts x = desired_profit

theorem find_selling_price (x : ℕ) :
  (x = 105 ∨ x = 120) ↔ profit_equation x := by
  sorry

end find_selling_price_l69_69230


namespace greatest_divisor_of_four_consecutive_integers_l69_69667

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69667


namespace div_product_four_consecutive_integers_l69_69842

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69842


namespace doodads_for_thingamabobs_l69_69551

-- Definitions for the conditions
def doodads_per_widgets : ℕ := 18
def widgets_per_thingamabobs : ℕ := 11
def widgets_count : ℕ := 5
def thingamabobs_count : ℕ := 4
def target_thingamabobs : ℕ := 80

-- Definition for the final proof statement
theorem doodads_for_thingamabobs : 
    doodads_per_widgets * (target_thingamabobs * widgets_per_thingamabobs / thingamabobs_count / widgets_count) = 792 := 
by
  sorry

end doodads_for_thingamabobs_l69_69551


namespace greatest_divisor_four_consecutive_integers_l69_69724

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69724


namespace total_seats_l69_69933

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69933


namespace geometric_shape_is_sphere_l69_69268

-- Define the spherical coordinate system conditions
def spherical_coordinates (ρ θ φ r : ℝ) : Prop :=
  ρ = r

-- The theorem we want to prove
theorem geometric_shape_is_sphere (ρ θ φ r : ℝ) (h : spherical_coordinates ρ θ φ r) : ∀ (x y z : ℝ), (x^2 + y^2 + z^2 = r^2) :=
by
  sorry

end geometric_shape_is_sphere_l69_69268


namespace sandro_children_l69_69189

variables (sons daughters children : ℕ)

-- Conditions
def has_six_times_daughters (sons daughters : ℕ) : Prop := daughters = 6 * sons
def has_three_sons (sons : ℕ) : Prop := sons = 3

-- Theorem to be proven
theorem sandro_children (h1 : has_six_times_daughters sons daughters) (h2 : has_three_sons sons) : children = 21 :=
by
  -- Definitions from the conditions
  unfold has_six_times_daughters has_three_sons at h1 h2

  -- Skip the proof
  sorry

end sandro_children_l69_69189


namespace greatest_divisor_four_consecutive_l69_69737

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69737


namespace simplify_product_l69_69193

theorem simplify_product (x t : ℕ) : (x^2 * t^3) * (x^3 * t^4) = (x^5) * (t^7) := 
by 
  sorry

end simplify_product_l69_69193


namespace max_quadratic_function_at_intersection_l69_69308

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x * (3 - x)

-- Define function g(x) = 2 * ln x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x

-- Define the circle equation
def circle (x y r : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = r ^ 2

-- Define point P(x0, y0) conditions
def point_P (x0 y0 r : ℝ) : Prop := g x0 = y0 ∧ circle x0 y0 r

theorem max_quadratic_function_at_intersection (x0 y0 : ℝ) (hP: point_P x0 y0 1) :
  ∃ x, f x = 9 / 8 :=
by
  use 3 / 2
  sorry

end max_quadratic_function_at_intersection_l69_69308


namespace sin_480_deg_l69_69464

theorem sin_480_deg : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_deg_l69_69464


namespace divisor_of_product_of_four_consecutive_integers_l69_69644

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69644


namespace area_enclosed_by_graph_l69_69033

theorem area_enclosed_by_graph : 
  (∃ (A : ℝ), A = 96) ↔ (∃ x y : ℝ, abs x + abs(3 * y) = 12) := 
sorry

end area_enclosed_by_graph_l69_69033


namespace daughterAgeThreeYearsFromNow_l69_69054

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end daughterAgeThreeYearsFromNow_l69_69054


namespace part1_part2_l69_69405

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem part1 :
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem part2 :
  ∃ (max_x min_x : ℝ), max_x ∈ Set.Icc (π/12) (π/4) ∧ min_x ∈ Set.Icc (π/12) (π/4) ∧
    f max_x = 7 / 4 ∧ f min_x = (5 + Real.sqrt 3) / 4 ∧
    (max_x = π / 6) ∧ (min_x = π / 12 ∨ min_x = π / 4) :=
by sorry

end part1_part2_l69_69405


namespace divisor_of_four_consecutive_integers_l69_69749

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69749


namespace fraction_sum_l69_69130

theorem fraction_sum (a b : ℕ) (h1 : 0.36 = a / b) (h2: Nat.gcd a b = 1) : a + b = 15 := by
  sorry

end fraction_sum_l69_69130


namespace minimum_doors_to_safety_l69_69017

-- Definitions in Lean 4 based on the conditions provided
def spaceship (corridors : ℕ) : Prop := corridors = 23

def command_closes (N : ℕ) (corridors : ℕ) : Prop := N ≤ corridors

-- Theorem based on the question and conditions
theorem minimum_doors_to_safety (N : ℕ) (corridors : ℕ)
  (h_corridors : spaceship corridors)
  (h_command : command_closes N corridors) :
  N = 22 :=
sorry

end minimum_doors_to_safety_l69_69017


namespace gcd_pow_sub_one_l69_69471

theorem gcd_pow_sub_one (n m : ℕ) (h1 : n = 1005) (h2 : m = 1016) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2047 := by
  rw [h1, h2]
  sorry

end gcd_pow_sub_one_l69_69471


namespace total_number_of_questions_l69_69140

/-
  Given:
    1. There are 20 type A problems.
    2. Type A problems require twice as much time as type B problems.
    3. 32.73 minutes are spent on type A problems.
    4. Total examination time is 3 hours.

  Prove that the total number of questions is 199.
-/

theorem total_number_of_questions
  (type_A_problems : ℕ)
  (type_B_to_A_time_ratio : ℝ)
  (time_spent_on_type_A : ℝ)
  (total_exam_time_hours : ℝ)
  (total_number_of_questions : ℕ)
  (h_type_A_problems : type_A_problems = 20)
  (h_time_ratio : type_B_to_A_time_ratio = 2)
  (h_time_spent_on_type_A : time_spent_on_type_A = 32.73)
  (h_total_exam_time_hours : total_exam_time_hours = 3) :
  total_number_of_questions = 199 := 
sorry

end total_number_of_questions_l69_69140


namespace total_number_of_seats_l69_69926

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69926


namespace more_tails_than_heads_l69_69417

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end more_tails_than_heads_l69_69417


namespace number_of_triangles_in_6x6_grid_l69_69124

def is_valid_point (n : ℕ) (i j : ℕ) : Prop :=
  1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n

def number_of_triangles (n : ℕ) : ℕ :=
  let points := (Finset.range (n + 1)).product (Finset.range (n + 1))
  let valid_points := points.filter (λ p, is_valid_point n p.1 p.2)
  let all_triangles := valid_points.powersetLen 3
  let collinear (a b c : (ℕ × ℕ)) : Prop :=
    (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)
  let non_collinear_triangles := all_triangles.filter (λ t, ¬ collinear t.nthLe! 0 t.nthLe! 1 t.nthLe! 2)
  non_collinear_triangles.card

theorem number_of_triangles_in_6x6_grid : number_of_triangles 6 = 6804 := 
sorry

end number_of_triangles_in_6x6_grid_l69_69124


namespace smallest_odd_angle_in_right_triangle_l69_69453

theorem smallest_odd_angle_in_right_triangle
  (x y : ℤ) (hx1 : even x) (hy1 : odd y) (hx2 : x > y) (ha : x + y = 90) :
  y = 31 :=
by sorry

end smallest_odd_angle_in_right_triangle_l69_69453


namespace number_of_students_l69_69238

theorem number_of_students (n : ℕ) :
  n < 50 ∧ n % 6 = 4 ∧ n % 4 = 2 → n = 10 ∨ n = 22 ∨ n = 34 := by
  -- Proof goes here
  sorry

end number_of_students_l69_69238


namespace ratio_of_times_gina_chooses_to_her_sister_l69_69272

theorem ratio_of_times_gina_chooses_to_her_sister (sister_shows : ℕ) (minutes_per_show : ℕ) (gina_minutes : ℕ) (ratio : ℕ × ℕ) :
  sister_shows = 24 →
  minutes_per_show = 50 →
  gina_minutes = 900 →
  ratio = (900 / Nat.gcd 900 1200, 1200 / Nat.gcd 900 1200) →
  ratio = (3, 4) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_times_gina_chooses_to_her_sister_l69_69272


namespace graph_comparison_l69_69091

theorem graph_comparison :
  (∀ x : ℝ, (x^2 - x + 3) < (x^2 - x + 5)) :=
by
  sorry

end graph_comparison_l69_69091


namespace king_lancelot_seats_38_l69_69918

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69918


namespace greatest_divisor_of_four_consecutive_integers_l69_69657

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69657


namespace factory_hours_per_day_l69_69878

def factory_produces (hours_per_day : ℕ) : Prop :=
  let refrigerators_per_hour := 90
  let coolers_per_hour := 160
  let total_products_per_hour := refrigerators_per_hour + coolers_per_hour
  let total_products_in_5_days := 11250
  total_products_per_hour * (5 * hours_per_day) = total_products_in_5_days

theorem factory_hours_per_day : ∃ h : ℕ, factory_produces h ∧ h = 9 :=
by
  existsi 9
  unfold factory_produces
  sorry

end factory_hours_per_day_l69_69878


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69761

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69761


namespace divisor_of_product_of_four_consecutive_integers_l69_69651

theorem divisor_of_product_of_four_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end divisor_of_product_of_four_consecutive_integers_l69_69651


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69608

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69608


namespace scientific_notation_219400_l69_69049

def scientific_notation (n : ℝ) (m : ℝ) : Prop := n = m * 10^5

theorem scientific_notation_219400 : scientific_notation 219400 2.194 := 
by
  sorry

end scientific_notation_219400_l69_69049


namespace number_of_planting_methods_l69_69523

theorem number_of_planting_methods :
  let vegetables := ["cucumbers", "cabbages", "rape", "flat beans"]
  let plots := ["plot1", "plot2", "plot3"]
  (∀ v ∈ vegetables, v = "cucumbers") →
  (∃! n : ℕ, n = 18)
:= by
  sorry

end number_of_planting_methods_l69_69523


namespace number_of_ways_to_fulfill_order_l69_69948

open Finset Nat

/-- Bill must buy exactly eight donuts from a shop offering five types, 
with at least two of the first type and one of each of the other four types. 
Prove that there are exactly 15 different ways to fulfill this order. -/
theorem number_of_ways_to_fulfill_order : 
  let total_donuts := 8
  let types_of_donuts := 5
  let mandatory_first_type := 2
  let mandatory_each_other_type := 1
  let remaining_donuts := total_donuts - (mandatory_first_type + 4 * mandatory_each_other_type)
  let combinations := (remaining_donuts + types_of_donuts - 1).choose (types_of_donuts - 1)
  combinations = 15 := 
by
  sorry

end number_of_ways_to_fulfill_order_l69_69948


namespace area_of_abs_sum_l69_69025

theorem area_of_abs_sum (x y : ℝ) (h : |x| + |3 * y| = 12) : 
  let area := 96 in
  True :=
begin
  sorry
end

end area_of_abs_sum_l69_69025


namespace product_of_consecutive_integers_l69_69678

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69678


namespace circles_radii_divide_regions_l69_69151

-- Declare the conditions as definitions
def radii_count : ℕ := 16
def circles_count : ℕ := 10

-- State the proof problem
theorem circles_radii_divide_regions (radii : ℕ) (circles : ℕ) (hr : radii = radii_count) (hc : circles = circles_count) : 
  (circles + 1) * radii = 176 := sorry

end circles_radii_divide_regions_l69_69151


namespace ratio_of_segments_l69_69305

theorem ratio_of_segments
  (x y z u v : ℝ)
  (h_triangle : x^2 + y^2 = z^2)
  (h_ratio_legs : 4 * x = 3 * y)
  (h_u : u = x^2 / z)
  (h_v : v = y^2 / z) :
  u / v = 9 / 16 := 
  sorry

end ratio_of_segments_l69_69305


namespace analytic_expression_of_f_l69_69979

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi / 2)

noncomputable def g (α : ℝ) := Real.cos (α - Real.pi / 3)

theorem analytic_expression_of_f :
  (∀ x, f x = Real.cos x) ∧
  (∀ α, α ∈ Set.Icc 0 Real.pi → g α = 1/2 → (α = 0 ∨ α = 2 * Real.pi / 3)) :=
by
  sorry

end analytic_expression_of_f_l69_69979


namespace fraction_eq_l69_69580

theorem fraction_eq : (15.5 / (-0.75) : ℝ) = (-62 / 3) := 
by {
  sorry
}

end fraction_eq_l69_69580


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69803

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69803


namespace optionC_is_correct_l69_69867

theorem optionC_is_correct (x : ℝ) : (x^2)^3 = x^6 :=
by sorry

end optionC_is_correct_l69_69867


namespace total_square_footage_l69_69596

-- Definitions from the problem conditions
def price_per_square_foot : ℝ := 98
def total_property_value : ℝ := 333200

-- The mathematical statement to prove
theorem total_square_footage : (total_property_value / price_per_square_foot) = 3400 :=
by
  -- Proof goes here (skipped with sorry)
  sorry

end total_square_footage_l69_69596


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69710

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69710


namespace tent_ratio_l69_69442

-- Define the relevant variables
variables (N E S C T : ℕ)

-- State the conditions
def conditions : Prop :=
  N = 100 ∧
  E = 2 * N ∧
  S = 200 ∧
  T = 900 ∧
  N + E + S + C = T

-- State the theorem to prove the ratio
theorem tent_ratio (h : conditions N E S C T) : C = 4 * N :=
by sorry

end tent_ratio_l69_69442


namespace expected_steps_unit_interval_l69_69327

noncomputable def expected_steps_to_color_interval : ℝ := 
  -- Placeholder for the function calculating expected steps
  sorry 

theorem expected_steps_unit_interval : expected_steps_to_color_interval = 5 :=
  sorry

end expected_steps_unit_interval_l69_69327


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69773

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69773


namespace negation_universal_statement_l69_69014

theorem negation_universal_statement :
  (¬ (∀ x : ℝ, |x| ≥ 0)) ↔ (∃ x : ℝ, |x| < 0) :=
by sorry

end negation_universal_statement_l69_69014


namespace product_of_consecutive_integers_l69_69674

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69674


namespace greatest_divisor_of_consecutive_product_l69_69863

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69863


namespace total_seats_l69_69931

theorem total_seats (KA_pos : ℕ) (SL_pos : ℕ) (h1 : KA_pos = 10) (h2 : SL_pos = 29) (h3 : SL_pos = KA_pos + (KA_pos * 2 - 1) / 2):
  let total_positions := 2 * (SL_pos - KA_pos - 1) + 2
  total_positions = 38 :=
by
  sorry

end total_seats_l69_69931


namespace four_consecutive_integers_divisible_by_12_l69_69693

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69693


namespace shekar_marks_math_l69_69332

theorem shekar_marks_math (M : ℕ) (science : ℕ) (social_studies : ℕ) (english : ℕ) 
(biology : ℕ) (average : ℕ) (num_subjects : ℕ) 
(h_science : science = 65)
(h_social : social_studies = 82)
(h_english : english = 67)
(h_biology : biology = 55)
(h_average : average = 69)
(h_num_subjects : num_subjects = 5) :
M + science + social_studies + english + biology = average * num_subjects →
M = 76 :=
by
  sorry

end shekar_marks_math_l69_69332


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69697

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69697


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69764

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69764


namespace mary_no_torn_cards_l69_69325

theorem mary_no_torn_cards
  (T : ℕ) -- number of Mary's initial torn baseball cards
  (initial_cards : ℕ := 18) -- initial baseball cards
  (fred_cards : ℕ := 26) -- baseball cards given by Fred
  (bought_cards : ℕ := 40) -- baseball cards bought
  (total_cards : ℕ := 84) -- total baseball cards Mary has now
  (h : initial_cards - T + fred_cards + bought_cards = total_cards)
  : T = 0 :=
by sorry

end mary_no_torn_cards_l69_69325


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69702

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69702


namespace patrick_savings_l69_69183

theorem patrick_savings :
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  saved_money - lent_money = 25 := by
  let bicycle_price := 150
  let saved_money := bicycle_price / 2
  let lent_money := 50
  sorry

end patrick_savings_l69_69183


namespace greatest_divisor_of_product_of_consecutive_integers_l69_69701

theorem greatest_divisor_of_product_of_consecutive_integers :
  ∀ (n : ℕ), ∃ k : ℕ, k ∣ (n*(n+1)*(n+2)*(n+3)) ∧ k = 12 :=
by {
  intro n,
  use 12,
  sorry
}

end greatest_divisor_of_product_of_consecutive_integers_l69_69701


namespace problem1_problem2_l69_69250

-- Problem 1
theorem problem1 (a : ℝ) : 2 * a + 3 * a - 4 * a = a :=
by sorry

-- Problem 2
theorem problem2 : 
  - (1 : ℝ) ^ 2022 + (27 / 4) * (- (1 / 3) - 1) / ((-3) ^ 2) + abs (-1) = -1 :=
by sorry

end problem1_problem2_l69_69250


namespace divisor_of_four_consecutive_integers_l69_69751

theorem divisor_of_four_consecutive_integers (n : ℕ) : 
  ∃ (k : ℕ), k = 12 ∧ (n * (n+1) * (n+2) * (n+3)) % k = 0 :=
by {
  use 12,
  split,
  exact rfl,
  sorry -- This is where the proof details would go.
}

end divisor_of_four_consecutive_integers_l69_69751


namespace possible_starting_cities_l69_69390

open SimpleGraph

noncomputable def cities : Finset String :=
  { "Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan" }

noncomputable def connections : SimpleGraph (Finset String) :=
  ⟨cities, λ u v, (u, v) ∈ {
    ("Saint Petersburg", "Tver"), 
    ("Yaroslavl", "Nizhny Novgorod"), 
    ("Moscow", "Kazan"), 
    ("Nizhny Novgorod", "Kazan"), 
    ("Moscow", "Tver"), 
    ("Moscow", "Nizhny Novgorod")
  }⟩

theorem possible_starting_cities (u : Finset String) :
  u ∈ {"Saint Petersburg", "Yaroslavl"} ↔ 
  ((degree connections u) % 2 = 1) := sorry

end possible_starting_cities_l69_69390


namespace circle_division_l69_69166

theorem circle_division (radii_count : ℕ) (concentric_circles_count : ℕ) :
  radii_count = 16 → concentric_circles_count = 10 → 
  let total_regions := (concentric_circles_count + 1) * radii_count 
  in total_regions = 176 :=
by
  intros h_1 h_2
  simp [h_1, h_2]
  sorry

end circle_division_l69_69166


namespace enclosed_area_abs_x_abs_3y_eq_12_l69_69024

theorem enclosed_area_abs_x_abs_3y_eq_12 : 
  let f (x y : ℝ) := |x| + |3 * y|
  ∃ (A : ℝ), ∀ (x y : ℝ), f x y = 12 → A = 96 := 
sorry

end enclosed_area_abs_x_abs_3y_eq_12_l69_69024


namespace joe_lifting_problem_l69_69481

theorem joe_lifting_problem (x y : ℝ) (h1 : x + y = 900) (h2 : 2 * x = y + 300) : x = 400 :=
sorry

end joe_lifting_problem_l69_69481


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69642

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n → ∃ d : ℕ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
begin
  intro n,
  intro hn,
  use 24,
  split,
  { refl },
  { -- Here we would show that 24 divides the product of n, n+1, n+2, and n+3
    sorry
  }
end

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69642


namespace indeterminate_C_l69_69145

variable (m n C : ℝ)

theorem indeterminate_C (h1 : m = 8 * n + C)
                      (h2 : m + 2 = 8 * (n + 0.25) + C) : 
                      False :=
by
  sorry

end indeterminate_C_l69_69145


namespace simplify_fraction_l69_69194

theorem simplify_fraction (h1 : 90 = 2 * 3^2 * 5) (h2 : 150 = 2 * 3 * 5^2) : (90 / 150 : ℚ) = 3 / 5 := by
  sorry

end simplify_fraction_l69_69194


namespace right_triangle_side_length_l69_69388

theorem right_triangle_side_length (a c b : ℕ) (h1 : a = 3) (h2 : c = 5) (h3 : c^2 = a^2 + b^2) : b = 4 :=
sorry

end right_triangle_side_length_l69_69388


namespace orangeade_price_per_glass_l69_69328

theorem orangeade_price_per_glass (O : ℝ) (W : ℝ) (P : ℝ) (price_1_day : ℝ) 
    (h1 : W = O) (h2 : price_1_day = 0.30) (revenue_equal : 2 * O * price_1_day = 3 * O * P) :
  P = 0.20 :=
by
  sorry

end orangeade_price_per_glass_l69_69328


namespace correct_decision_probability_l69_69944

open ProbabilityTheory

theorem correct_decision_probability :
  (∃ p : ℝ, ∀ (h : 3), Prob (fun i => h.consultant_in_opinion_is_correct) = 0.8) →
  ∑ (i in (Finset.univ : Finset (Fin 3)), if i.independent_majority (0.8)) = 0.896 :=
by
  sorry

end correct_decision_probability_l69_69944


namespace simplify_expression_l69_69334

theorem simplify_expression :
  (1024 ^ (1/5) * 125 ^ (1/3)) = 20 :=
by
  have h1 : 1024 = 2 ^ 10 := by norm_num
  have h2 : 125 = 5 ^ 3 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end simplify_expression_l69_69334


namespace king_arthur_round_table_seats_l69_69898

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69898


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69712

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69712


namespace quotient_is_four_l69_69266

theorem quotient_is_four (dividend : ℕ) (k : ℕ) (h1 : dividend = 16) (h2 : k = 4) : dividend / k = 4 :=
by
  sorry

end quotient_is_four_l69_69266


namespace product_of_consecutive_integers_l69_69673

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69673


namespace four_consecutive_integers_divisible_by_12_l69_69688

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69688


namespace total_seats_at_round_table_l69_69906

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69906


namespace exists_root_in_interval_l69_69316

open Real

theorem exists_root_in_interval 
  (a b c r s : ℝ) 
  (ha : a ≠ 0) 
  (hr : a * r ^ 2 + b * r + c = 0) 
  (hs : -a * s ^ 2 + b * s + c = 0) : 
  ∃ t : ℝ, r < t ∧ t < s ∧ (a / 2) * t ^ 2 + b * t + c = 0 :=
by
  sorry

end exists_root_in_interval_l69_69316


namespace min_value_of_expression_l69_69179

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 4) :
  (9 / x + 1 / y + 25 / z) ≥ 20.25 :=
by 
  sorry

end min_value_of_expression_l69_69179


namespace four_consecutive_product_divisible_by_12_l69_69798

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69798


namespace solve_player_coins_l69_69463

def player_coins (n m k: ℕ) : Prop :=
  ∃ k, 
  (m = k * (n - 1) + 50) ∧ 
  (3 * m = 7 * n * k - 3 * k + 74) ∧ 
  (m = 69)

theorem solve_player_coins (n m k : ℕ) : player_coins n m k :=
by {
  sorry
}

end solve_player_coins_l69_69463


namespace abs_neg_six_l69_69591

theorem abs_neg_six : abs (-6) = 6 := by
  sorry

end abs_neg_six_l69_69591


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69770

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69770


namespace part1_part2_l69_69977

theorem part1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (1 - 4 / (2 * a^0 + a)) = 0) : a = 2 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x : ℝ, (2^x + 1) * (1 - 2 / (2^x + 1)) + k = 0) : k < 1 :=
sorry

end part1_part2_l69_69977


namespace divides_totient_prime_power_minus_one_l69_69574

noncomputable def euler_totient (x : ℕ) : ℕ := Nat.totient x

theorem divides_totient_prime_power_minus_one 
  (p : ℕ) (n : ℕ)
  (hp_prime : Nat.Prime p) :
  n ∣ euler_totient (p^n - 1) :=
by
  sorry

end divides_totient_prime_power_minus_one_l69_69574


namespace tan_sum_formula_l69_69267

theorem tan_sum_formula {A B : ℝ} (hA : A = 55) (hB : B = 65) (h1 : Real.tan (A + B) = Real.tan 120) 
    (h2 : Real.tan 120 = -Real.sqrt 3) :
    Real.tan 55 + Real.tan 65 - Real.sqrt 3 * Real.tan 55 * Real.tan 65 = -Real.sqrt 3 := 
by
  sorry

end tan_sum_formula_l69_69267


namespace cubic_inequality_l69_69999

theorem cubic_inequality (a b : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) (hne : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end cubic_inequality_l69_69999


namespace total_children_l69_69190

theorem total_children (sons daughters : ℕ) (h1 : sons = 3) (h2 : daughters = 6 * sons) : (sons + daughters) = 21 :=
by
  sorry

end total_children_l69_69190


namespace arithmetic_sum_l69_69281

theorem arithmetic_sum :
  ∀ (a : ℕ → ℝ),
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →
  (∃ x : ℝ, ∃ y : ℝ, x^2 - 6 * x - 1 = 0 ∧ y^2 - 6 * y - 1 = 0 ∧ x = a 3 ∧ y = a 15) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  intros a h_arith_seq h_roots
  sorry

end arithmetic_sum_l69_69281


namespace hyperbola_asymptotes_l69_69338

theorem hyperbola_asymptotes :
  ∀ {x y : ℝ},
    (x^2 / 9 - y^2 / 16 = 1) →
    (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end hyperbola_asymptotes_l69_69338


namespace total_seats_at_round_table_l69_69909

-- Define namespace and conditions
namespace KingArthur

variable (n : ℕ) -- Total number of seats

-- Conditions
def king_position : ℕ := 10
def lancelot_position : ℕ := 29
def opposite (a b : ℕ) (n : ℕ) : Prop := (a + (n / 2)) % n = b

-- Proof statement
theorem total_seats_at_round_table : opposite 10 29 n ∧ 29 < n → n = 38 :=
by
  sorry

end KingArthur

end total_seats_at_round_table_l69_69909


namespace second_order_arithmetic_sequence_a30_l69_69271

theorem second_order_arithmetic_sequence_a30 {a : ℕ → ℝ}
  (h₁ : ∀ n, a (n + 1) - a n - (a (n + 2) - a (n + 1)) = 20)
  (h₂ : a 10 = 23)
  (h₃ : a 20 = 23) :
  a 30 = 2023 := 
sorry

end second_order_arithmetic_sequence_a30_l69_69271


namespace total_team_cost_correct_l69_69598

variable (jerseyCost shortsCost socksCost cleatsCost waterBottleCost : ℝ)
variable (numPlayers : ℕ)
variable (discountThreshold discountRate salesTaxRate : ℝ)

noncomputable def totalTeamCost : ℝ :=
  let totalCostPerPlayer := jerseyCost + shortsCost + socksCost + cleatsCost + waterBottleCost
  let totalCost := totalCostPerPlayer * numPlayers
  let discount := if totalCost > discountThreshold then totalCost * discountRate else 0
  let discountedTotal := totalCost - discount
  let tax := discountedTotal * salesTaxRate
  let finalCost := discountedTotal + tax
  finalCost

theorem total_team_cost_correct :
  totalTeamCost 25 15.20 6.80 40 12 25 500 0.10 0.07 = 2383.43 := by
  sorry

end total_team_cost_correct_l69_69598


namespace min_surface_area_l69_69879

/-- Defining the conditions and the problem statement -/
def solid (volume : ℝ) (face1 face2 : ℝ) : Prop := 
  ∃ x y z, x * y * z = volume ∧ (x * y = face1 ∨ y * z = face1 ∨ z * x = face1)
                      ∧ (x * y = face2 ∨ y * z = face2 ∨ z * x = face2)

def juan_solids (face1 face2 face3 face4 face5 face6 : ℝ) : Prop :=
  solid 128 4 32 ∧ solid 128 64 16 ∧ solid 128 8 32

theorem min_surface_area {volume : ℝ} {face1 face2 face3 face4 face5 face6 : ℝ} 
  (h : juan_solids 4 32 64 16 8 32) : 
  ∃ area : ℝ, area = 688 :=
sorry

end min_surface_area_l69_69879


namespace four_consecutive_integers_divisible_by_12_l69_69684

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69684


namespace equations_of_motion_l69_69457

-- Initial conditions and setup
def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 45

-- Questions:
-- 1. Equations of motion for point M
-- 2. Equation of the trajectory of point M
-- 3. Velocity of point M

theorem equations_of_motion (t : ℝ) :
  let xM := 45 * (1 + Real.cos (omega * t))
  let yM := 45 * Real.sin (omega * t)
  xM = 45 * (1 + Real.cos (omega * t)) ∧
  yM = 45 * Real.sin (omega * t) ∧
  ((yM / 45) ^ 2 + ((xM - 45) / 45) ^ 2 = 1) ∧
  let vMx := -450 * Real.sin (omega * t)
  let vMy := 450 * Real.cos (omega * t)
  (vMx = -450 * Real.sin (omega * t)) ∧
  (vMy = 450 * Real.cos (omega * t)) :=
by
  sorry

end equations_of_motion_l69_69457


namespace greatest_divisor_four_consecutive_l69_69736

open Nat

theorem greatest_divisor_four_consecutive (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_four_consecutive_l69_69736


namespace chandra_monsters_l69_69252

def monsters_day_1 : Nat := 2
def monsters_day_2 : Nat := monsters_day_1 * 3
def monsters_day_3 : Nat := monsters_day_2 * 4
def monsters_day_4 : Nat := monsters_day_3 * 5
def monsters_day_5 : Nat := monsters_day_4 * 6

def total_monsters : Nat := monsters_day_1 + monsters_day_2 + monsters_day_3 + monsters_day_4 + monsters_day_5

theorem chandra_monsters : total_monsters = 872 :=
by
  unfold total_monsters
  unfold monsters_day_1
  unfold monsters_day_2
  unfold monsters_day_3
  unfold monsters_day_4
  unfold monsters_day_5
  sorry

end chandra_monsters_l69_69252


namespace no_integer_solution_for_system_l69_69186

theorem no_integer_solution_for_system :
  ¬ ∃ (a b c d : ℤ), 
    (a * b * c * d - a = 1961) ∧ 
    (a * b * c * d - b = 961) ∧ 
    (a * b * c * d - c = 61) ∧ 
    (a * b * c * d - d = 1) :=
by {
  sorry
}

end no_integer_solution_for_system_l69_69186


namespace probability_green_given_not_red_l69_69300

theorem probability_green_given_not_red :
  let total_balls := 20
  let red_balls := 5
  let yellow_balls := 5
  let green_balls := 10
  let non_red_balls := total_balls - red_balls

  let probability_green_given_not_red := (green_balls : ℚ) / (non_red_balls : ℚ)

  probability_green_given_not_red = 2 / 3 :=
by
  sorry

end probability_green_given_not_red_l69_69300


namespace profit_maximization_l69_69499

-- Define the conditions 
variable (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5)

-- Expression for yield ω
noncomputable def yield (x : ℝ) : ℝ := 4 - (3 / (x + 1))

-- Expression for profit function L(x)
noncomputable def profit (x : ℝ) : ℝ := 16 * yield x - x - 2 * x

-- Theorem stating the profit function expression and its maximum
theorem profit_maximization (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) :
  profit x = 64 - 48 / (x + 1) - 3 * x ∧ 
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ 5 → profit x₀ ≤ profit 3) :=
sorry

end profit_maximization_l69_69499


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69610

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69610


namespace product_of_consecutive_integers_l69_69677

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69677


namespace oatmeal_cookies_l69_69443

theorem oatmeal_cookies (total_cookies chocolate_chip_cookies : ℕ)
  (h1 : total_cookies = 6 * 9)
  (h2 : chocolate_chip_cookies = 13) :
  total_cookies - chocolate_chip_cookies = 41 := by
  sorry

end oatmeal_cookies_l69_69443


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69628

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69628


namespace king_lancelot_seats_38_l69_69917

noncomputable def totalSeats (seat_king seat_lancelot : ℕ) : ℕ :=
  if seat_king < seat_lancelot then
    2 * (seat_lancelot - seat_king - 1) + 2
  else
    2 * (seat_king - seat_lancelot - 1) + 2

theorem king_lancelot_seats_38 (seat_king seat_lancelot : ℕ) (h1 : seat_king = 10) (h2 : seat_lancelot = 29) :
  totalSeats seat_king seat_lancelot = 38 := 
  by
    sorry

end king_lancelot_seats_38_l69_69917


namespace problem1_problem2_l69_69248

theorem problem1 : (82 - 15) * (32 + 18) = 3350 :=
by
  sorry

theorem problem2 : (25 + 4) * 75 = 2175 :=
by
  sorry

end problem1_problem2_l69_69248


namespace greatest_divisor_four_consecutive_integers_l69_69728

theorem greatest_divisor_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ (d ∣ (n * (n+1) * (n+2) * (n+3))) :=
begin
  sorry
end

end greatest_divisor_four_consecutive_integers_l69_69728


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69626

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69626


namespace product_of_four_consecutive_integers_divisible_by_12_l69_69801

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l69_69801


namespace find_x2_plus_y2_l69_69975

theorem find_x2_plus_y2 : ∀ (x y : ℝ),
  3 * x + 4 * y = 30 →
  x + 2 * y = 13 →
  x^2 + y^2 = 36.25 :=
by
  intros x y h1 h2
  sorry

end find_x2_plus_y2_l69_69975


namespace john_gives_to_stud_owner_l69_69313

variable (initial_puppies : ℕ) (puppies_given_away : ℕ) (puppies_kept : ℕ) (price_per_puppy : ℕ) (profit : ℕ)

theorem john_gives_to_stud_owner
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = initial_puppies / 2)
  (h3 : puppies_kept = 1)
  (h4 : price_per_puppy = 600)
  (h5 : profit = 1500) :
  let puppies_left_to_sell := initial_puppies - puppies_given_away - puppies_kept
  let total_sales := puppies_left_to_sell * price_per_puppy
  total_sales - profit = 300 :=
by
  intro puppies_left_to_sell
  intro total_sales
  sorry

end john_gives_to_stud_owner_l69_69313


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69708

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69708


namespace cylinder_surface_area_l69_69062

theorem cylinder_surface_area (h : ℝ) (r : ℝ) (h_eq : h = 12) (r_eq : r = 4) : 
  2 * π * r * (r + h) = 128 * π := 
by
  rw [h_eq, r_eq]
  sorry

end cylinder_surface_area_l69_69062


namespace fraction_of_kiwis_l69_69600

theorem fraction_of_kiwis (total_fruits : ℕ) (num_strawberries : ℕ) (h₁ : total_fruits = 78) (h₂ : num_strawberries = 52) :
  (total_fruits - num_strawberries) / total_fruits = 1 / 3 :=
by
  -- proof to be provided, this is just the statement
  sorry

end fraction_of_kiwis_l69_69600


namespace sqrt_sum_inequality_l69_69535

open Real

theorem sqrt_sum_inequality (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ sqrt 21 :=
sorry

end sqrt_sum_inequality_l69_69535


namespace students_taller_than_Yoongi_l69_69478

theorem students_taller_than_Yoongi {n total shorter : ℕ} (h1 : total = 20) (h2 : shorter = 11) : n = 8 :=
by
  sorry

end students_taller_than_Yoongi_l69_69478


namespace product_of_consecutive_integers_l69_69672

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69672


namespace initial_people_employed_l69_69875

-- Definitions from the conditions
def initial_work_days : ℕ := 25
def total_work_days : ℕ := 50
def work_done_percentage : ℕ := 40
def additional_people : ℕ := 30

-- Defining the statement to be proved
theorem initial_people_employed (P : ℕ) 
  (h1 : initial_work_days = 25) 
  (h2 : total_work_days = 50)
  (h3 : work_done_percentage = 40)
  (h4 : additional_people = 30) 
  (work_remaining_percentage := 60) : 
  (P * 25 / 10 = 100) -> (P + 30) * 50 = P * 625 / 10 -> P = 120 :=
by
  sorry

end initial_people_employed_l69_69875


namespace quarters_total_l69_69331

def initial_quarters : ℕ := 21
def additional_quarters : ℕ := 49
def total_quarters : ℕ := initial_quarters + additional_quarters

theorem quarters_total : total_quarters = 70 := by
  sorry

end quarters_total_l69_69331


namespace range_of_y_eq_4_sin_squared_x_minus_2_l69_69460

theorem range_of_y_eq_4_sin_squared_x_minus_2 : 
  (∀ x : ℝ, y = 4 * (Real.sin x)^2 - 2) → 
  (∃ a b : ℝ, ∀ x : ℝ, y ∈ Set.Icc a b ∧ a = -2 ∧ b = 2) :=
sorry

end range_of_y_eq_4_sin_squared_x_minus_2_l69_69460


namespace regions_divided_by_radii_circles_l69_69169

theorem regions_divided_by_radii_circles (n_radii : ℕ) (n_concentric : ℕ)
  (h_radii : n_radii = 16) (h_concentric : n_concentric = 10) :
  let regions := (n_concentric + 1) * n_radii
  in regions = 176 :=
by
  have h1 : regions = (10 + 1) * 16 := by 
    rw [h_radii, h_concentric]
  have h2 : regions = 176 := by
    rw h1
  exact h2

end regions_divided_by_radii_circles_l69_69169


namespace find_cylinder_radius_l69_69500

-- Define the problem conditions
def cone_diameter := 10
def cone_altitude := 12
def cylinder_height_eq_diameter (r: ℚ) := 2 * r

-- Define the cone and cylinder inscribed properties
noncomputable def inscribed_cylinder_radius (r : ℚ) : Prop :=
  (cylinder_height_eq_diameter r) ≤ cone_altitude ∧
  2 * r ≤ cone_diameter ∧
  cone_altitude - cylinder_height_eq_diameter r = (cone_altitude * r) / (cone_diameter / 2)

-- The proof goal
theorem find_cylinder_radius : ∃ r : ℚ, inscribed_cylinder_radius r ∧ r = 30/11 :=
by
  sorry

end find_cylinder_radius_l69_69500


namespace sin_2y_eq_37_40_l69_69969

variable (x y : ℝ)
variable (sin cos : ℝ → ℝ)

axiom sin_def : sin x = 2 * cos y - (5/2) * sin y
axiom cos_def : cos x = 2 * sin y - (5/2) * cos y

theorem sin_2y_eq_37_40 : sin (2 * y) = 37 / 40 := by
  sorry

end sin_2y_eq_37_40_l69_69969


namespace greatest_divisor_of_four_consecutive_integers_l69_69833

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ),
  n > 0 → 
  ∃ k, k = 24 ∧ ∀ m, m ∈ (finset.range 4).map (λ i, n + i) → (m : ℤ) ∣ 24 :=
by
  sorry

end greatest_divisor_of_four_consecutive_integers_l69_69833


namespace total_seats_round_table_l69_69939

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69939


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69784

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ ((n) * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  use 12
  split
  · refl
  · sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69784


namespace sum_of_sequences_l69_69086

-- Definition of the problem conditions
def seq1 := [2, 12, 22, 32, 42]
def seq2 := [10, 20, 30, 40, 50]
def sum_seq1 := 2 + 12 + 22 + 32 + 42
def sum_seq2 := 10 + 20 + 30 + 40 + 50

-- Lean statement of the problem
theorem sum_of_sequences :
  sum_seq1 + sum_seq2 = 260 := by
  sorry

end sum_of_sequences_l69_69086


namespace total_handshakes_l69_69479

variable (n : ℕ) (h : n = 12)

theorem total_handshakes (H : ∀ (b : ℕ), b = n → (n * (n - 1)) / 2 = 66) : 
  (12 * 11) / 2 = 66 := 
by
  sorry

end total_handshakes_l69_69479


namespace parabola_vertex_above_x_axis_l69_69216

theorem parabola_vertex_above_x_axis (k : ℝ) (h : k > 9 / 4) : 
  ∃ y : ℝ, ∀ x : ℝ, y = (x - 3 / 2) ^ 2 + k - 9 / 4 ∧ y > 0 := 
by
  sorry

end parabola_vertex_above_x_axis_l69_69216


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69766

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69766


namespace final_apples_count_l69_69994

-- Define the initial conditions
def initial_apples : Nat := 128

def percent_25 (n : Nat) : Nat := n * 25 / 100

def apples_after_selling_to_jill (n : Nat) : Nat := n - percent_25 n

def apples_after_selling_to_june (n : Nat) : Nat := apples_after_selling_to_jill n - percent_25 (apples_after_selling_to_jill n)

def apples_after_giving_to_teacher (n : Nat) : Nat := apples_after_selling_to_june n - 1

-- The theorem stating the problem to be proved
theorem final_apples_count : apples_after_giving_to_teacher initial_apples = 71 := by
  sorry

end final_apples_count_l69_69994


namespace points_lie_on_circle_l69_69963

theorem points_lie_on_circle (t : ℝ) : 
  let x := Real.cos t
  let y := Real.sin t
  x^2 + y^2 = 1 :=
by
  sorry

end points_lie_on_circle_l69_69963


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69817

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : ℕ, 1 ≤ n →
  ∃ k : ℕ, k = 12 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69817


namespace intersection_x_value_l69_69104

theorem intersection_x_value:
  ∃ x y : ℝ, y = 4 * x - 29 ∧ 3 * x + y = 105 ∧ x = 134 / 7 :=
by
  sorry

end intersection_x_value_l69_69104


namespace log2_15_eq_formula_l69_69530

theorem log2_15_eq_formula (a b : ℝ) (h1 : a = Real.log 6 / Real.log 3) (h2 : b = Real.log 20 / Real.log 5) :
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) :=
by
  sorry

end log2_15_eq_formula_l69_69530


namespace integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l69_69088

open Real

noncomputable def integral_abs_x_plus_2 : ℝ :=
  ∫ x in (-4 : ℝ)..(3 : ℝ), |x + 2|

noncomputable def integral_inv_x_minus_1 : ℝ :=
  ∫ x in (2 : ℝ)..(Real.exp 1 + 1 : ℝ), 1 / (x - 1)

theorem integral_abs_x_plus_2_eq_29_div_2 :
  integral_abs_x_plus_2 = 29 / 2 :=
sorry

theorem integral_inv_x_minus_1_eq_1 :
  integral_inv_x_minus_1 = 1 :=
sorry

end integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l69_69088


namespace four_consecutive_integers_divisible_by_12_l69_69685

theorem four_consecutive_integers_divisible_by_12 (n : ℕ) : 
  12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
sorry

end four_consecutive_integers_divisible_by_12_l69_69685


namespace eval_expr_correct_l69_69105

noncomputable def eval_expr : ℝ :=
  let a := (12:ℝ)^5 * (6:ℝ)^4
  let b := (3:ℝ)^2 * (36:ℝ)^2
  let c := Real.sqrt 9 * Real.log (27:ℝ)
  (a / b) + c

theorem eval_expr_correct : eval_expr = 27657.887510597983 := by
  sorry

end eval_expr_correct_l69_69105


namespace product_of_four_consecutive_integers_divisible_by_twelve_l69_69763

theorem product_of_four_consecutive_integers_divisible_by_twelve :
  ∀ n : ℕ, 12 ∣ (n * (n + 1) * (n + 2) * (n + 3)) := 
by
  sorry

end product_of_four_consecutive_integers_divisible_by_twelve_l69_69763


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69621

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l69_69621


namespace greatest_divisor_of_product_of_four_consecutive_integers_l69_69715

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l69_69715


namespace total_seats_round_table_l69_69938

theorem total_seats_round_table (n : ℕ) (h : n = 38)
  (ka_position : ℕ) (sl_position : ℕ) 
  (h1 : ka_position = 10) 
  (h2 : sl_position = 29) 
  (h3 : (ka_position + n/2) % n = sl_position) : 
  n = 38 :=
by
  -- All steps and solution proof
  sorry

end total_seats_round_table_l69_69938


namespace find_unknown_number_l69_69228

theorem find_unknown_number :
  ∃ (x : ℝ), (786 * x) / 30 = 1938.8 → x = 74 :=
by 
  sorry

end find_unknown_number_l69_69228


namespace div_product_four_consecutive_integers_l69_69846

theorem div_product_four_consecutive_integers (n : ℕ) : 
  ∃ d : ℕ, (∀ (k : ℕ), k ∈ [n, n + 1, n + 2, n + 3] → d ∣ (k * (k + 1) * (k + 2) * (k + 3))) ∧ d = 12 :=
by 
  sorry

end div_product_four_consecutive_integers_l69_69846


namespace four_consecutive_product_divisible_by_12_l69_69790

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l69_69790


namespace geometric_progression_product_l69_69089

variables {n : ℕ} {b q S S' P : ℝ} 

theorem geometric_progression_product (hb : b ≠ 0) (hq : q ≠ 1)
  (hP : P = b^n * q^(n*(n-1)/2))
  (hS : S = b * (1 - q^n) / (1 - q))
  (hS' : S' = (q^n - 1) / (b * (q - 1)))
  : P = (S * S')^(n/2) := 
sorry

end geometric_progression_product_l69_69089


namespace greatest_divisor_of_four_consecutive_integers_l69_69661

theorem greatest_divisor_of_four_consecutive_integers :
  ∀ (n : ℕ), ∃ d : ℕ, d = 12 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intros n
  use 12
  split
  . rfl
  . sorry

end greatest_divisor_of_four_consecutive_integers_l69_69661


namespace average_speed_of_rocket_l69_69359

theorem average_speed_of_rocket
  (ascent_speed : ℕ)
  (ascent_time : ℕ)
  (descent_distance : ℕ)
  (descent_time : ℕ)
  (average_speed : ℕ)
  (h_ascent_speed : ascent_speed = 150)
  (h_ascent_time : ascent_time = 12)
  (h_descent_distance : descent_distance = 600)
  (h_descent_time : descent_time = 3)
  (h_average_speed : average_speed = 160) :
  (ascent_speed * ascent_time + descent_distance) / (ascent_time + descent_time) = average_speed :=
by
  sorry

end average_speed_of_rocket_l69_69359


namespace dismissed_cases_l69_69496

theorem dismissed_cases (total_cases : Int) (X : Int)
  (total_cases_eq : total_cases = 17)
  (remaining_cases_eq : X = (2 * X / 3) + 1 + 4) :
  total_cases - X = 2 :=
by
  -- Placeholder for the proof
  sorry

end dismissed_cases_l69_69496


namespace triangle_bisection_l69_69073

theorem triangle_bisection
    (A B C P Q R S T L : Point)
    (triangle_ABC : Triangle A B C)
    (P_on_AB : PointOnSegment P A B)
    (S_on_AC : PointOnSegment S A C)
    (T_on_BC : PointOnSegment T B C)
    (AP_eq_AS : dist A P = dist A S)
    (BP_eq_BT : dist B P = dist B T)
    (circumcircle_PST : Circle (circumcenter P S T) (circumradius P S T))
    (Q_on_AB' : PointOnCircle Q circumcircle_PST)
    (R_on_BC' : PointOnCircle R circumcircle_PST)
    (L_on_PS_QR : meets_at L (line_through P S) (line_through Q R))
    (C_on_CL : PointOnLine C L) :
    bisects (line_through C L) P Q := 
sorry

end triangle_bisection_l69_69073


namespace min_value_of_z_l69_69094

theorem min_value_of_z : ∃ x : ℝ, ∀ y : ℝ, 5 * x^2 + 20 * x + 25 ≤ 5 * y^2 + 20 * y + 25 :=
by
  sorry

end min_value_of_z_l69_69094


namespace sandro_children_l69_69188

variables (sons daughters children : ℕ)

-- Conditions
def has_six_times_daughters (sons daughters : ℕ) : Prop := daughters = 6 * sons
def has_three_sons (sons : ℕ) : Prop := sons = 3

-- Theorem to be proven
theorem sandro_children (h1 : has_six_times_daughters sons daughters) (h2 : has_three_sons sons) : children = 21 :=
by
  -- Definitions from the conditions
  unfold has_six_times_daughters has_three_sons at h1 h2

  -- Skip the proof
  sorry

end sandro_children_l69_69188


namespace daughter_age_in_3_years_l69_69059

variable (mother_age_now : ℕ) (gap_years : ℕ) (ratio : ℕ)

theorem daughter_age_in_3_years
  (h1 : mother_age_now = 41) 
  (h2 : gap_years = 5)
  (h3 : ratio = 2) :
  let mother_age_then := mother_age_now - gap_years in
  let daughter_age_then := mother_age_then / ratio in
  let daughter_age_now := daughter_age_then + gap_years in
  let daughter_age_in_3_years := daughter_age_now + 3 in
  daughter_age_in_3_years = 26 :=
  by
    sorry

end daughter_age_in_3_years_l69_69059


namespace king_arthur_round_table_seats_l69_69896

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69896


namespace total_number_of_seats_l69_69924

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l69_69924


namespace product_of_consecutive_integers_l69_69679

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l69_69679


namespace total_kayaks_built_by_april_l69_69374

def kayaks_built_february : ℕ := 5
def kayaks_built_next_month (n : ℕ) : ℕ := 3 * n
def kayaks_built_march : ℕ := kayaks_built_next_month kayaks_built_february
def kayaks_built_april : ℕ := kayaks_built_next_month kayaks_built_march

theorem total_kayaks_built_by_april : 
  kayaks_built_february + kayaks_built_march + kayaks_built_april = 65 :=
by
  -- proof goes here
  sorry

end total_kayaks_built_by_april_l69_69374


namespace group4_equations_groupN_equations_find_k_pos_l69_69263

-- Conditions from the problem
def group1_fractions := (1 : ℚ) / 1 + (1 : ℚ) / 3 = 4 / 3
def group1_pythagorean := 4^2 + 3^2 = 5^2

def group2_fractions := (1 : ℚ) / 3 + (1 : ℚ) / 5 = 8 / 15
def group2_pythagorean := 8^2 + 15^2 = 17^2

def group3_fractions := (1 : ℚ) / 5 + (1 : ℚ) / 7 = 12 / 35
def group3_pythagorean := 12^2 + 35^2 = 37^2

-- Proof Statements
theorem group4_equations :
  ((1 : ℚ) / 7 + (1 : ℚ) / 9 = 16 / 63) ∧ (16^2 + 63^2 = 65^2) := 
  sorry

theorem groupN_equations (n : ℕ) :
  ((1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = 4 * n / (4 * n^2 - 1)) ∧
  ((4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2) :=
  sorry

theorem find_k_pos (k : ℕ) : 
  k^2 + 9603^2 = 9605^2 → k = 196 := 
  sorry

end group4_equations_groupN_equations_find_k_pos_l69_69263


namespace saras_sister_ordered_notebooks_l69_69449

theorem saras_sister_ordered_notebooks (x : ℕ) 
  (initial_notebooks : ℕ := 4) 
  (lost_notebooks : ℕ := 2) 
  (current_notebooks : ℕ := 8) :
  initial_notebooks + x - lost_notebooks = current_notebooks → x = 6 :=
by
  intros h
  sorry

end saras_sister_ordered_notebooks_l69_69449


namespace greatest_divisor_of_consecutive_product_l69_69857

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l69_69857


namespace domain_of_ln_x_plus_1_l69_69007

variable (x : ℝ)

theorem domain_of_ln_x_plus_1 : {x : ℝ | x > -1} = {x : ℝ | ∃ y, y = ln(x + 1)} :=
by
  sorry

end domain_of_ln_x_plus_1_l69_69007


namespace king_arthur_round_table_seats_l69_69893

theorem king_arthur_round_table_seats (n : ℕ) (h₁ : n > 1) (h₂ : 10 < 29) (h₃ : (29 - 10) * 2 = n - 2) : 
  n = 38 := 
by
  sorry

end king_arthur_round_table_seats_l69_69893
