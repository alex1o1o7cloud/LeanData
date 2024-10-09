import Mathlib

namespace infinitely_many_n_l2143_214329

theorem infinitely_many_n (p : ℕ) (hp : p.Prime) (hp2 : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ n * 2^n + 1 :=
sorry

end infinitely_many_n_l2143_214329


namespace minimum_triangle_area_l2143_214395

theorem minimum_triangle_area :
  ∀ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (1 / m + 2 / n = 1) → (1 / 2 * m * n) = 4 :=
by
  sorry

end minimum_triangle_area_l2143_214395


namespace scholarship_awards_l2143_214364

theorem scholarship_awards (x : ℕ) (h : 10000 * x + 2000 * (28 - x) = 80000) : x = 3 ∧ (28 - x) = 25 :=
by {
  sorry
}

end scholarship_awards_l2143_214364


namespace circle_symmetry_l2143_214373

theorem circle_symmetry {a : ℝ} (h : a ≠ 0) :
  ∀ {x y : ℝ}, (x^2 + y^2 + 2*a*x - 2*a*y = 0) → (x + y = 0) :=
sorry

end circle_symmetry_l2143_214373


namespace age_sum_l2143_214378

theorem age_sum (my_age : ℕ) (mother_age : ℕ) (h1 : mother_age = 3 * my_age) (h2 : my_age = 10) :
  my_age + mother_age = 40 :=
by 
  -- proof omitted
  sorry

end age_sum_l2143_214378


namespace initial_bottle_caps_l2143_214376

variable (initial_caps added_caps total_caps : ℕ)

theorem initial_bottle_caps 
  (h1 : added_caps = 7) 
  (h2 : total_caps = 14) 
  (h3 : total_caps = initial_caps + added_caps) : 
  initial_caps = 7 := 
by 
  sorry

end initial_bottle_caps_l2143_214376


namespace probability_is_one_third_l2143_214337

noncomputable def probability_four_of_a_kind_or_full_house : ℚ :=
  let total_outcomes := 6
  let probability_triplet_match := 1 / total_outcomes
  let probability_pair_match := 1 / total_outcomes
  probability_triplet_match + probability_pair_match

theorem probability_is_one_third :
  probability_four_of_a_kind_or_full_house = 1 / 3 :=
by
  -- sorry
  trivial

end probability_is_one_third_l2143_214337


namespace find_x_minus_y_l2143_214342

theorem find_x_minus_y (x y n : ℤ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x > y) (h4 : n / 10 < 10 ∧ n / 10 ≥ 1) 
  (h5 : 2 * n = x + y) 
  (h6 : ∃ m : ℤ, m^2 = x * y ∧ m = (n % 10) * 10 + n / 10) 
  : x - y = 66 :=
sorry

end find_x_minus_y_l2143_214342


namespace cameron_list_count_l2143_214382

theorem cameron_list_count :
  let lower := 100
  let upper := 1000
  let step := 20
  let n_min := lower / step
  let n_max := upper / step
  lower % step = 0 ∧ upper % step = 0 →
  upper ≥ lower →
  n_max - n_min + 1 = 46 :=
by
  sorry

end cameron_list_count_l2143_214382


namespace original_cost_price_l2143_214366

theorem original_cost_price (selling_price_friend : ℝ) (gain_percent : ℝ) (loss_percent : ℝ) 
  (final_selling_price : ℝ) : 
  final_selling_price = 54000 → gain_percent = 0.2 → loss_percent = 0.1 → 
  selling_price_friend = (1 - loss_percent) * x → final_selling_price = (1 + gain_percent) * selling_price_friend → 
  x = 50000 :=
by 
  sorry

end original_cost_price_l2143_214366


namespace unique_solution_l2143_214350

def is_prime (n : ℕ) : Prop := Nat.Prime n

def eq_triple (m p q : ℕ) : Prop :=
  2 ^ m * p ^ 2 + 1 = q ^ 5

theorem unique_solution (m p q : ℕ) (h1 : m > 0) (h2 : is_prime p) (h3 : is_prime q) :
  eq_triple m p q ↔ (m, p, q) = (1, 11, 3) := by
  sorry

end unique_solution_l2143_214350


namespace street_tree_fourth_point_l2143_214317

theorem street_tree_fourth_point (a b : ℝ) (h_a : a = 0.35) (h_b : b = 0.37) :
  (a + 4 * ((b - a) / 4)) = b :=
by 
  rw [h_a, h_b]
  sorry

end street_tree_fourth_point_l2143_214317


namespace area_of_inscribed_square_l2143_214318

-- Define the right triangle with segments m and n on the hypotenuse
variables {m n : ℝ}

-- Noncomputable setting for non-constructive aspects
noncomputable def inscribed_square_area (m n : ℝ) : ℝ :=
  (m * n)

-- Theorem stating that the area of the inscribed square is m * n
theorem area_of_inscribed_square (m n : ℝ) : inscribed_square_area m n = m * n :=
by sorry

end area_of_inscribed_square_l2143_214318


namespace A_squared_plus_B_squared_eq_one_l2143_214306

theorem A_squared_plus_B_squared_eq_one
  (A B : ℝ) (h1 : A ≠ B)
  (h2 : ∀ x : ℝ, (A * (B * x ^ 2 + A) ^ 2 + B - (B * (A * x ^ 2 + B) ^ 2 + A)) = B ^ 2 - A ^ 2) :
  A ^ 2 + B ^ 2 = 1 :=
sorry

end A_squared_plus_B_squared_eq_one_l2143_214306


namespace solve_for_x_l2143_214338

theorem solve_for_x (x : ℝ) (h : (1/3 : ℝ) * (x + 8 + 5*x + 3 + 3*x + 4) = 4*x + 1) : x = 4 :=
by {
  sorry
}

end solve_for_x_l2143_214338


namespace corner_coloring_condition_l2143_214302

theorem corner_coloring_condition 
  (n : ℕ) 
  (h1 : n ≥ 5) 
  (board : ℕ → ℕ → Prop) -- board(i, j) = true if cell (i, j) is black, false if white
  (h2 : ∀ i j, board i j = board (i + 1) j → board (i + 2) j = board (i + 1) j → ¬(board i j = board (i + 2) j)) -- row condition
  (h3 : ∀ i j, board i j = board i (j + 1) → board i (j + 2) = board i (j + 1) → ¬(board i j = board i (j + 2))) -- column condition
  (h4 : ∀ i j, board i j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board i j = board (i + 2) (j + 2))) -- diagonal condition
  (h5 : ∀ i j, board (i + 2) j = board (i + 1) (j + 1) → board (i + 2) (j + 2) = board (i + 1) (j + 1) → ¬(board (i + 2) j = board (i + 2) (j + 2))) -- anti-diagonal condition
  : ∀ i j, i + 2 < n ∧ j + 2 < n → ((board i j ∧ board (i + 2) (j + 2)) ∨ (board i (j + 2) ∧ board (i + 2) j)) :=
sorry

end corner_coloring_condition_l2143_214302


namespace no_real_roots_range_l2143_214362

theorem no_real_roots_range (a : ℝ) : (¬ ∃ x : ℝ, x^2 + a * x - 4 * a = 0) ↔ (-16 < a ∧ a < 0) := by
  sorry

end no_real_roots_range_l2143_214362


namespace proportion_solution_l2143_214396

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 4.5 / (7 / 3)) : x = 0.3888888889 :=
by
  sorry

end proportion_solution_l2143_214396


namespace kyoko_bought_three_balls_l2143_214346

theorem kyoko_bought_three_balls
  (cost_per_ball : ℝ)
  (total_paid : ℝ)
  (number_of_balls : ℝ)
  (h_cost_per_ball : cost_per_ball = 1.54)
  (h_total_paid : total_paid = 4.62)
  (h_number_of_balls : number_of_balls = total_paid / cost_per_ball) :
  number_of_balls = 3 := by
  sorry

end kyoko_bought_three_balls_l2143_214346


namespace arccos_one_eq_zero_l2143_214374

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
sorry

end arccos_one_eq_zero_l2143_214374


namespace triangle_inequality_l2143_214316

variable {α β γ a b c: ℝ}

theorem triangle_inequality (h1 : α + β + γ = π)
  (h2 : α > 0) (h3 : β > 0) (h4 : γ > 0)
  (h5 : a > 0) (h6 : b > 0) (h7 : c > 0)
  (h8 : (α > β ∧ a > b) ∨ (α = β ∧ a = b) ∨ (α < β ∧ a < b))
  (h9 : (β > γ ∧ b > c) ∨ (β = γ ∧ b = c) ∨ (β < γ ∧ b < c))
  (h10 : (γ > α ∧ c > a) ∨ (γ = α ∧ c = a) ∨ (γ < α ∧ c < a)) :
  (π / 3) ≤ (a * α + b * β + c * γ) / (a + b + c) ∧
  (a * α + b * β + c * γ) / (a + b + c) < (π / 2) :=
sorry

end triangle_inequality_l2143_214316


namespace eval_64_pow_5_over_6_l2143_214387

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l2143_214387


namespace min_washes_l2143_214332

theorem min_washes (x : ℕ) :
  (1 / 4)^x ≤ 1 / 100 → x ≥ 4 :=
by sorry

end min_washes_l2143_214332


namespace find_base_l2143_214383

theorem find_base (b : ℝ) (h : 2.134 * b^3 < 21000) : b ≤ 21 :=
by
  have h1 : b < (21000 / 2.134) ^ (1 / 3) := sorry
  have h2 : (21000 / 2.134) ^ (1 / 3) < 21.5 := sorry
  have h3 : b ≤ 21 := sorry
  exact h3

end find_base_l2143_214383


namespace original_weight_of_apple_box_l2143_214397

theorem original_weight_of_apple_box:
  ∀ (x : ℕ), (3 * x - 12 = x) → x = 6 :=
by
  intros x h
  sorry

end original_weight_of_apple_box_l2143_214397


namespace unique_solution_triple_l2143_214339

theorem unique_solution_triple {a b c : ℝ} (h₀ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h₁ : a^2 + b^2 + c^2 = 3) (h₂ : (a + b + c) * (a^2 * b + b^2 * c + c^2 * a) = 9) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ c = 1 ∧ b = 1) ∨ (b = 1 ∧ a = 1 ∧ c = 1) ∨ (b = 1 ∧ c = 1 ∧ a = 1) ∨ (c = 1 ∧ a = 1 ∧ b = 1) ∨ (c = 1 ∧ b = 1 ∧ a = 1) :=
sorry

end unique_solution_triple_l2143_214339


namespace sum_of_squares_eq_23456_l2143_214388

theorem sum_of_squares_eq_23456 (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 :=
by sorry

end sum_of_squares_eq_23456_l2143_214388


namespace find_percentage_l2143_214321

theorem find_percentage (P : ℝ) : 
  0.15 * P * (0.5 * 5600) = 126 → P = 0.3 := 
by 
  sorry

end find_percentage_l2143_214321


namespace circle_area_pi_l2143_214359

def circle_eq := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1

theorem circle_area_pi (h : ∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0 → (x + 2) ^ 2 + y ^ 2 = 1) :
  ∃ S : ℝ, S = π :=
by {
  sorry
}

end circle_area_pi_l2143_214359


namespace problem_1_problem_2_l2143_214335

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l2143_214335


namespace hyperbola_condition_l2143_214315

theorem hyperbola_condition (m : ℝ) : 
  (∃ a b : ℝ, a = m + 4 ∧ b = m - 3 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)) ↔ m > 3 :=
sorry

end hyperbola_condition_l2143_214315


namespace eric_less_than_ben_l2143_214369

variables (E B J : ℕ)

theorem eric_less_than_ben
  (hJ : J = 26)
  (hB : B = J - 9)
  (total_money : E + B + J = 50) :
  B - E = 10 :=
sorry

end eric_less_than_ben_l2143_214369


namespace open_spots_level4_correct_l2143_214325

noncomputable def open_spots_level_4 (total_levels : ℕ) (spots_per_level : ℕ) (open_spots_level1 : ℕ) (open_spots_level2 : ℕ) (open_spots_level3 : ℕ) (full_spots_total : ℕ) : ℕ := 
  let total_spots := total_levels * spots_per_level
  let open_spots_total := total_spots - full_spots_total 
  let open_spots_first_three := open_spots_level1 + open_spots_level2 + open_spots_level3
  open_spots_total - open_spots_first_three

theorem open_spots_level4_correct :
  open_spots_level_4 4 100 58 (58 + 2) (58 + 2 + 5) 186 = 31 :=
by
  sorry

end open_spots_level4_correct_l2143_214325


namespace problem_I_problem_II_l2143_214305

/-- Proof problem I: Given f(x) = |x - 1|, prove that the inequality f(x) ≥ 4 - |x - 1| implies x ≥ 3 or x ≤ -1 -/
theorem problem_I (x : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (h2 : f x ≥ 4 - |x - 1|) : x ≥ 3 ∨ x ≤ -1 :=
  sorry

/-- Proof problem II: Given f(x) = |x - 1| and 1/m + 1/(2*n) = 1 (m > 0, n > 0), prove that the minimum value of mn is 2 -/
theorem problem_II (f : ℝ → ℝ) (h1 : ∀ x, f x = |x - 1|) (m n : ℝ) (hm : m > 0) (hn : n > 0) (h2 : 1/m + 1/(2*n) = 1) : m*n ≥ 2 :=
  sorry

end problem_I_problem_II_l2143_214305


namespace compute_fraction_power_l2143_214322

theorem compute_fraction_power :
  9 * (2/3)^4 = 16/9 := 
by
  sorry

end compute_fraction_power_l2143_214322


namespace distance_between_trees_l2143_214300

def yard_length : ℕ := 414
def number_of_trees : ℕ := 24

theorem distance_between_trees : yard_length / (number_of_trees - 1) = 18 := 
by sorry

end distance_between_trees_l2143_214300


namespace solve_equation_l2143_214370

theorem solve_equation (x y : ℝ) (h : 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2):
  y = -x - 2 ∨ y = -2 * x + 1 :=
sorry


end solve_equation_l2143_214370


namespace barn_painting_total_area_l2143_214394

theorem barn_painting_total_area :
  let width := 12
  let length := 15
  let height := 5
  let divider_width := 12
  let divider_height := 5

  let external_wall_area := 2 * (width * height + length * height)
  let dividing_wall_area := 2 * (divider_width * divider_height)
  let ceiling_area := width * length
  let total_area := 2 * external_wall_area + dividing_wall_area + ceiling_area

  total_area = 840 := by
    sorry

end barn_painting_total_area_l2143_214394


namespace angles_with_same_terminal_side_as_15_degree_l2143_214307

def condition1 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 90
def condition2 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 180
def condition3 (β : ℝ) (k : ℤ) : Prop := β = 15 + k * 360
def condition4 (β : ℝ) (k : ℤ) : Prop := β = 15 + 2 * k * 360

def has_same_terminal_side_as_15_degree (β : ℝ) : Prop :=
  ∃ k : ℤ, β = 15 + k * 360

theorem angles_with_same_terminal_side_as_15_degree (β : ℝ) :
  (∃ k : ℤ, condition1 β k)  ∨
  (∃ k : ℤ, condition2 β k)  ∨
  (∃ k : ℤ, condition3 β k)  ∨
  (∃ k : ℤ, condition4 β k) →
  has_same_terminal_side_as_15_degree β :=
by
  sorry

end angles_with_same_terminal_side_as_15_degree_l2143_214307


namespace minimum_distance_l2143_214389

noncomputable def distance (M Q : ℝ × ℝ) : ℝ :=
  ( (M.1 - Q.1) ^ 2 + (M.2 - Q.2) ^ 2 ) ^ (1 / 2)

theorem minimum_distance (M : ℝ × ℝ) :
  ∃ Q : ℝ × ℝ, ( (Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 ) ∧ distance M Q = 1 :=
sorry

end minimum_distance_l2143_214389


namespace original_price_l2143_214354

theorem original_price (P : ℝ) (final_price : ℝ) (percent_increase : ℝ) (h1 : final_price = 450) (h2 : percent_increase = 0.50) : 
  P + percent_increase * P = final_price → P = 300 :=
by
  sorry

end original_price_l2143_214354


namespace second_number_in_pair_l2143_214352

theorem second_number_in_pair (n m : ℕ) (h1 : (n, m) = (57, 58)) (h2 : ∃ (n m : ℕ), n < 1500 ∧ m < 1500 ∧ (n + m) % 5 = 0) : m = 58 :=
by {
  sorry
}

end second_number_in_pair_l2143_214352


namespace find_d_value_l2143_214377

theorem find_d_value (a b : ℚ) (d : ℚ) (h1 : a = 2) (h2 : b = 11) 
  (h3 : ∀ x, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) : 
  d = 53 / 4 :=
sorry

end find_d_value_l2143_214377


namespace total_seats_l2143_214343

-- Define the conditions
variable {S : ℝ} -- Total number of seats in the hall
variable {vacantSeats : ℝ} (h_vacant : vacantSeats = 240) -- Number of vacant seats
variable {filledPercentage : ℝ} (h_filled : filledPercentage = 0.60) -- Percentage of seats filled

-- Total seats in the hall
theorem total_seats (h : 0.40 * S = 240) : S = 600 :=
sorry

end total_seats_l2143_214343


namespace find_n_l2143_214304

-- Define the operation ø
def op (x w : ℕ) : ℕ := (2 ^ x) / (2 ^ w)

-- Prove that n operating with 2 and then 1 equals 8 implies n = 3
theorem find_n (n : ℕ) (H : op (op n 2) 1 = 8) : n = 3 :=
by
  -- Proof will be provided later
  sorry

end find_n_l2143_214304


namespace solve_equation_l2143_214323

def equation_solution (x : ℝ) : Prop :=
  (x^2 + x + 1) / (x + 1) = x + 3

theorem solve_equation :
  ∃ x : ℝ, equation_solution x ∧ x = -2 / 3 :=
by
  sorry

end solve_equation_l2143_214323


namespace Tim_has_52_photos_l2143_214372

theorem Tim_has_52_photos (T : ℕ) (Paul : ℕ) (Total : ℕ) (Tom : ℕ) : 
  (Paul = T + 10) → (Total = Tom + T + Paul) → (Tom = 38) → (Total = 152) → T = 52 :=
by
  intros hPaul hTotal hTom hTotalVal
  -- The proof would go here
  sorry

end Tim_has_52_photos_l2143_214372


namespace complete_square_solution_l2143_214353

theorem complete_square_solution :
  ∀ x : ℝ, ∃ p q : ℝ, (5 * x^2 - 30 * x - 45 = 0) → ((x + p) ^ 2 = q) ∧ (p + q = 15) :=
by
  sorry

end complete_square_solution_l2143_214353


namespace junior_score_is_95_l2143_214361

theorem junior_score_is_95:
  ∀ (n j s : ℕ) (x avg_total avg_seniors : ℕ),
    n = 20 →
    j = n * 15 / 100 →
    s = n * 85 / 100 →
    avg_total = 78 →
    avg_seniors = 75 →
    (j * x + s * avg_seniors) / n = avg_total →
    x = 95 :=
by
  sorry

end junior_score_is_95_l2143_214361


namespace limit_sum_infinite_geometric_series_l2143_214386

noncomputable def infinite_geometric_series_limit (a_1 q : ℝ) :=
  if |q| < 1 then (a_1 / (1 - q)) else 0

theorem limit_sum_infinite_geometric_series :
  infinite_geometric_series_limit 1 (1 / 3) = 3 / 2 :=
by
  sorry

end limit_sum_infinite_geometric_series_l2143_214386


namespace two_digit_number_is_54_l2143_214356

theorem two_digit_number_is_54 
    (n : ℕ) 
    (h1 : 10 ≤ n ∧ n < 100) 
    (h2 : n % 2 = 0) 
    (h3 : ∃ (a b : ℕ), a * b = 20 ∧ 10 * a + b = n) : 
    n = 54 := 
by
  sorry

end two_digit_number_is_54_l2143_214356


namespace initial_average_marks_l2143_214327

theorem initial_average_marks (A : ℝ) (h1 : 25 * A - 50 = 2450) : A = 100 :=
by
  sorry

end initial_average_marks_l2143_214327


namespace find_digits_l2143_214349

theorem find_digits (x y z : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (10 * x + 5) * (3 * 100 + y * 10 + z) = 7850 ↔ (x = 2 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end find_digits_l2143_214349


namespace unique_solution_pairs_count_l2143_214340

theorem unique_solution_pairs_count :
  ∃! (p : ℝ × ℝ), (p.1 + 2 * p.2 = 2 ∧ (|abs p.1 - 2 * abs p.2| = 2) ∧
       ∃! q, (q = (2, 0) ∨ q = (0, 1)) ∧ p = q) := 
sorry

end unique_solution_pairs_count_l2143_214340


namespace max_value_pq_qr_rs_sp_l2143_214320

variable (p q r s : ℕ)

theorem max_value_pq_qr_rs_sp :
  (p = 1 ∨ p = 3 ∨ p = 5 ∨ p = 7) →
  (q = 1 ∨ q = 3 ∨ q = 5 ∨ q = 7) →
  (r = 1 ∨ r = 3 ∨ r = 5 ∨ r = 7) →
  (s = 1 ∨ s = 3 ∨ s = 5 ∨ s = 7) →
  (p ≠ q) →
  (p ≠ r) →
  (p ≠ s) →
  (q ≠ r) →
  (q ≠ s) →
  (r ≠ s) →
  pq + qr + rs + sp ≤ 64 :=
sorry

end max_value_pq_qr_rs_sp_l2143_214320


namespace linear_relationship_correct_profit_160_max_profit_l2143_214399

-- Define the conditions for the problem
def data_points : List (ℝ × ℝ) := [(3.5, 280), (5.5, 120)]

-- The linear function relationship between y and x
def linear_relationship (x : ℝ) : ℝ := -80 * x + 560

-- The equation for profit, given selling price and sales quantity
def profit (x : ℝ) : ℝ := (x - 3) * (linear_relationship x) - 80

-- Prove the relationship y = -80x + 560 from given data points
theorem linear_relationship_correct : 
  ∀ (x y : ℝ), (x, y) ∈ data_points → y = linear_relationship x :=
sorry

-- Prove the selling price x = 4 results in a profit of $160 per day
theorem profit_160 (x : ℝ) (h : profit x = 160) : x = 4 :=
sorry

-- Prove the maximum profit and corresponding selling price
theorem max_profit : 
  ∃ x : ℝ, ∃ w : ℝ, 3.5 ≤ x ∧ x ≤ 5.5 ∧ profit x = w ∧ ∀ y, 3.5 ≤ y ∧ y ≤ 5.5 → profit y ≤ w ∧ w = 240 ∧ x = 5 :=
sorry

end linear_relationship_correct_profit_160_max_profit_l2143_214399


namespace math_problem_l2143_214357

theorem math_problem :
  (Int.ceil ((18: ℚ) / 5 * (-25 / 4)) - Int.floor ((18 / 5) * Int.floor (-25 / 4))) = 4 := 
by
  sorry

end math_problem_l2143_214357


namespace min_value_5x_plus_6y_l2143_214319

theorem min_value_5x_plus_6y (x y : ℝ) (h : 3 * x ^ 2 + 3 * y ^ 2 = 20 * x + 10 * y + 10) : 
  ∃ x y, (5 * x + 6 * y = 122) :=
by
  sorry

end min_value_5x_plus_6y_l2143_214319


namespace polynomial_simplification_l2143_214313

theorem polynomial_simplification (s : ℝ) : (2 * s^2 + 5 * s - 3) - (s^2 + 9 * s - 4) = s^2 - 4 * s + 1 :=
by
  sorry

end polynomial_simplification_l2143_214313


namespace stream_speed_l2143_214341

def boat_speed_still : ℝ := 30
def distance_downstream : ℝ := 80
def distance_upstream : ℝ := 40

theorem stream_speed (v : ℝ) (h : (distance_downstream / (boat_speed_still + v) = distance_upstream / (boat_speed_still - v))) :
  v = 10 :=
sorry

end stream_speed_l2143_214341


namespace ratio_of_ages_l2143_214375

theorem ratio_of_ages (joe_age_now james_age_now : ℕ) (h1 : joe_age_now = james_age_now + 10)
  (h2 : 2 * (joe_age_now + 8) = 3 * (james_age_now + 8)) : 
  (james_age_now + 8) / (joe_age_now + 8) = 2 / 3 := 
by
  sorry

end ratio_of_ages_l2143_214375


namespace packet_a_weight_l2143_214371

theorem packet_a_weight (A B C D E : ℕ) :
  A + B + C = 252 →
  A + B + C + D = 320 →
  E = D + 3 →
  B + C + D + E = 316 →
  A = 75 := by
  sorry

end packet_a_weight_l2143_214371


namespace set_equality_l2143_214324

def P : Set ℝ := { x | x^2 = 1 }

theorem set_equality : P = {-1, 1} :=
by
  sorry

end set_equality_l2143_214324


namespace watermelon_cost_100_l2143_214330

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l2143_214330


namespace find_b_when_a_is_1600_l2143_214314

theorem find_b_when_a_is_1600 :
  ∀ (a b : ℝ), (a * b = 400) ∧ ((2 * a) * b = 600) → (1600 * b = 600) → b = 0.375 :=
by
  intro a b
  intro h
  sorry

end find_b_when_a_is_1600_l2143_214314


namespace abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l2143_214351

theorem abs_x_minus_one_eq_one_minus_x_implies_x_le_one (x : ℝ) (h : |x - 1| = 1 - x) : x ≤ 1 :=
by
  sorry

end abs_x_minus_one_eq_one_minus_x_implies_x_le_one_l2143_214351


namespace inequality_solution_l2143_214301

theorem inequality_solution (x : ℝ) : (x^3 - 10 * x^2 > -25 * x) ↔ (0 < x ∧ x < 5) ∨ (5 < x) := 
sorry

end inequality_solution_l2143_214301


namespace a_formula_b_formula_T_formula_l2143_214347

variable {n : ℕ}

def S (n : ℕ) := 2 * n^2

def a (n : ℕ) : ℕ := 
  if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 2 * (1 / 4 ^ (n - 1))

def c (n : ℕ) : ℕ := (4 * n - 2) / (2 * 4 ^ (n - 1))

def T (n : ℕ) : ℕ := 
  (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5)

theorem a_formula :
  ∀ n, a n = 4 * n - 2 := 
sorry

theorem b_formula :
  ∀ n, b n = 2 / (4 ^ (n - 1)) :=
sorry

theorem T_formula :
  ∀ n, T n = (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5) :=
sorry

end a_formula_b_formula_T_formula_l2143_214347


namespace units_digit_L_L_15_l2143_214333

def Lucas (n : ℕ) : ℕ :=
match n with
| 0 => 2
| 1 => 1
| n + 2 => Lucas n + Lucas (n + 1)

theorem units_digit_L_L_15 : (Lucas (Lucas 15)) % 10 = 7 := by
  sorry

end units_digit_L_L_15_l2143_214333


namespace find_value_of_p_l2143_214393

variable (x y : ℝ)

/-- Given that the hyperbola has the equation x^2 / 4 - y^2 / 12 = 1
    and the eccentricity e = 2, and that the parabola x = 2 * p * y^2 has its focus at (e, 0), 
    prove that the value of the real number p is 1/8. -/
theorem find_value_of_p :
  (∃ (p : ℝ), 
    (∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1) ∧ 
    (∀ (x y : ℝ), x = 2 * p * y^2) ∧
    (2 = 2)) →
    ∃ (p : ℝ), p = 1/8 :=
by 
  sorry

end find_value_of_p_l2143_214393


namespace decreasing_interval_l2143_214348

noncomputable def func (x : ℝ) := 2 * x^3 - 6 * x^2 + 11

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv func x < 0 :=
by
  sorry

end decreasing_interval_l2143_214348


namespace probability_of_next_satisfied_customer_l2143_214311

noncomputable def probability_of_satisfied_customer : ℝ :=
  let p := (0.8 : ℝ)
  let q := (0.15 : ℝ)
  let neg_reviews := (60 : ℝ)
  let pos_reviews := (20 : ℝ)
  p / (p + q) * (q / (q + p))

theorem probability_of_next_satisfied_customer :
  probability_of_satisfied_customer = 0.64 :=
sorry

end probability_of_next_satisfied_customer_l2143_214311


namespace parallelogram_base_l2143_214368

theorem parallelogram_base
  (Area Height Base : ℕ)
  (h_area : Area = 120)
  (h_height : Height = 10)
  (h_area_eq : Area = Base * Height) :
  Base = 12 :=
by
  /- 
    We assume the conditions:
    1. Area = 120
    2. Height = 10
    3. Area = Base * Height 
    Then, we need to prove that Base = 12.
  -/
  sorry

end parallelogram_base_l2143_214368


namespace correct_proposition_l2143_214345

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition :
  ¬ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) ∧
  ¬ (∀ h : ℝ, f (-Real.pi / 6 + h) = f (-Real.pi / 6 - h)) ∧
  (∀ h : ℝ, f (-5 * Real.pi / 12 + h) = f (-5 * Real.pi / 12 - h)) :=
by sorry

end correct_proposition_l2143_214345


namespace union_eq_l2143_214326

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

theorem union_eq : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_eq_l2143_214326


namespace binary_to_decimal_l2143_214308

-- Define the binary number 10011_2
def binary_10011 : ℕ := bit0 (bit1 (bit1 (bit0 (bit1 0))))

-- Define the expected decimal value
def decimal_19 : ℕ := 19

-- State the theorem to convert binary 10011 to decimal
theorem binary_to_decimal :
  binary_10011 = decimal_19 :=
sorry

end binary_to_decimal_l2143_214308


namespace gcd_72_120_168_l2143_214367

theorem gcd_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
by
  sorry

end gcd_72_120_168_l2143_214367


namespace workers_l2143_214379

theorem workers (N C : ℕ) (h1 : N * C = 300000) (h2 : N * (C + 50) = 315000) : N = 300 :=
by
  sorry

end workers_l2143_214379


namespace smallest_possible_value_of_N_l2143_214392

theorem smallest_possible_value_of_N :
  ∀ (a b c d e f : ℕ), a + b + c + d + e + f = 3015 → (0 < a) → (0 < b) → (0 < c) → (0 < d) → (0 < e) → (0 < f) →
  (∃ N : ℕ, N = max (max (max (max (a + b) (b + c)) (c + d)) (d + e)) (e + f) ∧ N = 604) := 
by
  sorry

end smallest_possible_value_of_N_l2143_214392


namespace simplify_expression_l2143_214398

theorem simplify_expression :
  (Real.sqrt 2 * 2 ^ (1 / 2 : ℝ) + 18 / 3 * 3 - 8 ^ (3 / 2 : ℝ)) = (20 - 16 * Real.sqrt 2) :=
by sorry

end simplify_expression_l2143_214398


namespace correct_fraction_statement_l2143_214390

theorem correct_fraction_statement (x : ℝ) :
  (∀ a b : ℝ, (-a) / (-b) = a / b) ∧
  (¬ (∀ a : ℝ, a / 0 = 0)) ∧
  (∀ a b : ℝ, b ≠ 0 → (a * b) / (c * b) = a / c) → 
  ((∃ (a b : ℝ), a = 0 → a / b = 0) ∧ 
   (∀ (a b : ℝ), (a * k) / (b * k) = a / b) ∧ 
   (∀ (a b : ℝ), (-a) / (-b) = a / b) ∧ 
   (x < 1 → (|2 - x| + x) / 2 ≠ 0) 
  -> (∀ (a b : ℝ), (-a) / (-b) = a / b)) :=
by sorry

end correct_fraction_statement_l2143_214390


namespace next_ring_together_l2143_214360

def nextRingTime (libraryInterval : ℕ) (fireStationInterval : ℕ) (hospitalInterval : ℕ) (start : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm libraryInterval fireStationInterval) hospitalInterval + start

theorem next_ring_together : nextRingTime 18 24 30 (8 * 60) = 14 * 60 :=
by
  sorry

end next_ring_together_l2143_214360


namespace school_student_count_l2143_214358

theorem school_student_count (pencils erasers pencils_per_student erasers_per_student students : ℕ) 
    (h1 : pencils = 195) 
    (h2 : erasers = 65) 
    (h3 : pencils_per_student = 3)
    (h4 : erasers_per_student = 1) :
    students = pencils / pencils_per_student ∧ students = erasers / erasers_per_student → students = 65 :=
by
  sorry

end school_student_count_l2143_214358


namespace lines_perpendicular_l2143_214312

noncomputable def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem lines_perpendicular {m : ℝ} :
  is_perpendicular (m + 2) (1 - m) (m - 1) (2 * m + 3) ↔ m = 1 :=
by
  sorry

end lines_perpendicular_l2143_214312


namespace oranges_in_bin_l2143_214303

variable (n₀ n_throw n_new : ℕ)

theorem oranges_in_bin (h₀ : n₀ = 50) (h_throw : n_throw = 40) (h_new : n_new = 24) : 
  n₀ - n_throw + n_new = 34 := 
by 
  sorry

end oranges_in_bin_l2143_214303


namespace polygon_with_given_angle_sums_is_hexagon_l2143_214334

theorem polygon_with_given_angle_sums_is_hexagon
  (n : ℕ)
  (h_interior : (n - 2) * 180 = 2 * 360) :
  n = 6 :=
by
  sorry

end polygon_with_given_angle_sums_is_hexagon_l2143_214334


namespace even_increasing_function_inequality_l2143_214309

theorem even_increasing_function_inequality
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_increasing : ∀ {x₁ x₂ : ℝ}, x₁ < x₂ ∧ x₂ < 0 → f x₁ < f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end even_increasing_function_inequality_l2143_214309


namespace total_cost_l2143_214385

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l2143_214385


namespace nicholas_paid_more_than_kenneth_l2143_214310

def price_per_yard : ℝ := 40
def kenneth_yards : ℝ := 700
def nicholas_multiplier : ℝ := 6
def discount_rate : ℝ := 0.15

def kenneth_total_cost : ℝ := price_per_yard * kenneth_yards
def nicholas_yards : ℝ := nicholas_multiplier * kenneth_yards
def nicholas_original_cost : ℝ := price_per_yard * nicholas_yards
def discount_amount : ℝ := discount_rate * nicholas_original_cost
def nicholas_discounted_cost : ℝ := nicholas_original_cost - discount_amount
def difference_in_cost : ℝ := nicholas_discounted_cost - kenneth_total_cost

theorem nicholas_paid_more_than_kenneth :
  difference_in_cost = 114800 := by
  sorry

end nicholas_paid_more_than_kenneth_l2143_214310


namespace smallest_a_exists_l2143_214365

theorem smallest_a_exists : ∃ a b c : ℕ, 
                          (∀ α β : ℝ, 
                          (α > 0 ∧ α ≤ 1 / 1000) ∧ 
                          (β > 0 ∧ β ≤ 1 / 1000) ∧ 
                          (α + β = -b / a) ∧ 
                          (α * β = c / a) ∧ 
                          (b * b - 4 * a * c > 0)) ∧ 
                          (a = 1001000) := sorry

end smallest_a_exists_l2143_214365


namespace solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l2143_214384

theorem solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c ≤ 0 ↔ x ≤ -1 ∨ x ≥ 3) →
  b = -2*a →
  c = -3*a →
  a < 0 →
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) := 
by 
  intro h_root_set h_b_eq h_c_eq h_a_lt_0 
  sorry

end solution_set_of_cx2_minus_bx_plus_a_lt_0_is_correct_l2143_214384


namespace chess_team_boys_count_l2143_214381

theorem chess_team_boys_count (J S B : ℕ) 
  (h1 : J + S + B = 32) 
  (h2 : (1 / 3 : ℚ) * J + (1 / 2 : ℚ) * S + B = 18) : 
  B = 4 :=
by
  sorry

end chess_team_boys_count_l2143_214381


namespace neither_sufficient_nor_necessary_l2143_214328

-- For given real numbers x and y
-- Prove the statement "at least one of x and y is greater than 1" is not necessary and not sufficient for x^2 + y^2 > 2.
noncomputable def at_least_one_gt_one (x y : ℝ) : Prop := (x > 1) ∨ (y > 1)
def sum_of_squares_gt_two (x y : ℝ) : Prop := x^2 + y^2 > 2

theorem neither_sufficient_nor_necessary (x y : ℝ) :
  ¬(at_least_one_gt_one x y → sum_of_squares_gt_two x y) ∧ ¬(sum_of_squares_gt_two x y → at_least_one_gt_one x y) :=
by
  sorry

end neither_sufficient_nor_necessary_l2143_214328


namespace missing_number_l2143_214336

theorem missing_number 
  (a : ℕ) (b : ℕ) (x : ℕ)
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * x * b) 
  (h3 : b = 147) : 
  x = 3 :=
sorry

end missing_number_l2143_214336


namespace total_percent_decrease_l2143_214331

theorem total_percent_decrease (initial_value first_year_decrease second_year_decrease third_year_decrease : ℝ)
  (h₁ : first_year_decrease = 0.30)
  (h₂ : second_year_decrease = 0.10)
  (h₃ : third_year_decrease = 0.20) :
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let value_after_second_year := value_after_first_year * (1 - second_year_decrease)
  let value_after_third_year := value_after_second_year * (1 - third_year_decrease)
  let total_decrease := initial_value - value_after_third_year
  let total_percent_decrease := (total_decrease / initial_value) * 100
  total_percent_decrease = 49.60 := 
by
  sorry

end total_percent_decrease_l2143_214331


namespace math_problem_l2143_214344

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, a = b * b

theorem math_problem (a m : ℕ) (h1: m = 2992) (h2: a = m^2 + m^2 * (m+1)^2 + (m+1)^2) : is_perfect_square a :=
  sorry

end math_problem_l2143_214344


namespace range_of_x_l2143_214355

theorem range_of_x (x : ℝ) (h : x - 5 ≥ 0) : x ≥ 5 := 
  sorry

end range_of_x_l2143_214355


namespace find_sum_a_b_l2143_214380

theorem find_sum_a_b (a b : ℝ) 
  (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : a + 2 * b = 0 := 
sorry

end find_sum_a_b_l2143_214380


namespace units_digit_division_l2143_214363

theorem units_digit_division (a b c d e denom : ℕ)
  (h30 : a = 30) (h31 : b = 31) (h32 : c = 32) (h33 : d = 33) (h34 : e = 34)
  (h120 : denom = 120) :
  ((a * b * c * d * e) / denom) % 10 = 4 :=
by
  sorry

end units_digit_division_l2143_214363


namespace green_ball_probability_l2143_214391

def prob_green_ball : ℚ :=
  let prob_container := (1 : ℚ) / 3
  let prob_green_I := (4 : ℚ) / 12
  let prob_green_II := (5 : ℚ) / 8
  let prob_green_III := (4 : ℚ) / 8
  prob_container * prob_green_I + prob_container * prob_green_II + prob_container * prob_green_III

theorem green_ball_probability :
  prob_green_ball = 35 / 72 :=
by
  -- Proof steps are omitted as "sorry" is used to skip the proof.
  sorry

end green_ball_probability_l2143_214391
