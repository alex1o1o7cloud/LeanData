import Mathlib

namespace difference_of_roots_l175_175975

theorem difference_of_roots : 
  let a := 6 + 3 * Real.sqrt 5
  let b := 3 + Real.sqrt 5
  let c := 1
  ∃ x1 x2 : ℝ, (a * x1^2 - b * x1 + c = 0) ∧ (a * x2^2 - b * x2 + c = 0) ∧ x1 ≠ x2 
  ∧ x1 > x2 ∧ (x1 - x2) = (Real.sqrt 6 - Real.sqrt 5) / 3 := 
sorry

end difference_of_roots_l175_175975


namespace fraction_ratio_l175_175871

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l175_175871


namespace coin_value_is_630_l175_175178

theorem coin_value_is_630 :
  (∃ x : ℤ, x > 0 ∧ 406 * x = 63000) :=
by {
  sorry
}

end coin_value_is_630_l175_175178


namespace isosceles_trapezoid_area_l175_175941

-- Defining the problem characteristics
variables {a b c d h θ : ℝ}

-- The area formula for an isosceles trapezoid with given bases and height
theorem isosceles_trapezoid_area (h : ℝ) (c d : ℝ) : 
  (1 / 2) * (c + d) * h = (1 / 2) * (c + d) * h := 
sorry

end isosceles_trapezoid_area_l175_175941


namespace ratio_as_percentage_l175_175458

theorem ratio_as_percentage (x : ℝ) (h : (x / 2) / (3 * x / 5) = 3 / 5) : 
  (3 / 5) * 100 = 60 := 
sorry

end ratio_as_percentage_l175_175458


namespace solution_l175_175722

theorem solution
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (H : (1 / a + 1 / b) * (1 / c + 1 / d) + 1 / (a * b) + 1 / (c * d) = 6 / Real.sqrt (a * b * c * d)) :
  (a^2 + a * c + c^2) / (b^2 - b * d + d^2) = 3 :=
sorry

end solution_l175_175722


namespace find_value_l175_175152

theorem find_value (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2 * b)^2 = 25 :=
by
  sorry

end find_value_l175_175152


namespace polynomial_remainder_l175_175476

def f (r : ℝ) : ℝ := r^15 - r + 3

theorem polynomial_remainder :
  f 2 = 32769 := by
  sorry

end polynomial_remainder_l175_175476


namespace paul_money_duration_l175_175582

theorem paul_money_duration (mowing_income weed_eating_income weekly_spending money_last: ℕ) 
    (h1: mowing_income = 44) 
    (h2: weed_eating_income = 28) 
    (h3: weekly_spending = 9) 
    (h4: money_last = 8) 
    : (mowing_income + weed_eating_income) / weekly_spending = money_last := 
by
  sorry

end paul_money_duration_l175_175582


namespace right_triangle_exists_and_r_inscribed_circle_l175_175504

theorem right_triangle_exists_and_r_inscribed_circle (d : ℝ) (hd : d > 0) :
  ∃ (a b c : ℝ), 
    a < b ∧ 
    a^2 + b^2 = c^2 ∧
    b = a + d ∧ 
    c = b + d ∧ 
    (a + b - c) / 2 = d :=
by
  sorry

end right_triangle_exists_and_r_inscribed_circle_l175_175504


namespace value_of_expression_l175_175981

def expr : ℕ :=
  8 + 2 * (3^2)

theorem value_of_expression : expr = 26 :=
  by
  sorry

end value_of_expression_l175_175981


namespace unique_prime_n_l175_175468

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_n (n : ℕ)
  (h1 : isPrime n)
  (h2 : isPrime (n^2 + 10))
  (h3 : isPrime (n^2 - 2))
  (h4 : isPrime (n^3 + 6))
  (h5 : isPrime (n^5 + 36)) : n = 7 :=
by
  sorry

end unique_prime_n_l175_175468


namespace middle_number_divisible_by_4_l175_175553

noncomputable def three_consecutive_cubes_is_cube (x y : ℕ) : Prop :=
  (x-1)^3 + x^3 + (x+1)^3 = y^3

theorem middle_number_divisible_by_4 (x y : ℕ) (h : three_consecutive_cubes_is_cube x y) : 4 ∣ x :=
sorry

end middle_number_divisible_by_4_l175_175553


namespace div_eq_eight_fifths_l175_175947

theorem div_eq_eight_fifths (a b : ℚ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 :=
by
  sorry

end div_eq_eight_fifths_l175_175947


namespace line_through_circle_center_l175_175671

theorem line_through_circle_center (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + y + a = 0 ∧ x^2 + y^2 + 2 * x - 4 * y = 0) ↔ (a = 1) :=
by
  sorry

end line_through_circle_center_l175_175671


namespace sachin_age_l175_175062
-- Import the necessary library

-- Lean statement defining the problem conditions and result
theorem sachin_age :
  ∃ (S R : ℝ), (R = S + 7) ∧ (S / R = 7 / 9) ∧ (S = 24.5) :=
by
  sorry

end sachin_age_l175_175062


namespace probability_three_consecutive_cards_l175_175039

-- Definitions of the conditions
def total_ways_to_draw_three : ℕ := Nat.choose 52 3

def sets_of_consecutive_ranks : ℕ := 10

def ways_to_choose_three_consecutive : ℕ := 64

def favorable_outcomes : ℕ := sets_of_consecutive_ranks * ways_to_choose_three_consecutive

def probability_consecutive_ranks : ℚ := favorable_outcomes / total_ways_to_draw_three

-- The main statement to prove
theorem probability_three_consecutive_cards :
  probability_consecutive_ranks = 32 / 1105 := 
sorry

end probability_three_consecutive_cards_l175_175039


namespace interest_rate_and_years_l175_175400

theorem interest_rate_and_years
    (P : ℝ)
    (n : ℕ)
    (e : ℝ)
    (h1 : P * (e ^ n) * e = P * (e ^ (n + 1)) + 4156.02)
    (h2 : P * (e ^ (n - 1)) = P * (e ^ n) - 3996.12) :
    (e = 1.04) ∧ (P = 60000) ∧ (E = 4/100) ∧ (n = 14) := by
  sorry

end interest_rate_and_years_l175_175400


namespace card_draw_probability_l175_175070

theorem card_draw_probability:
  let hearts := 13
  let diamonds := 13
  let clubs := 13
  let total_cards := 52
  let first_draw_probability := hearts / (total_cards : ℝ)
  let second_draw_probability := diamonds / (total_cards - 1 : ℝ)
  let third_draw_probability := clubs / (total_cards - 2 : ℝ)
  first_draw_probability * second_draw_probability * third_draw_probability = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l175_175070


namespace rose_paid_after_discount_l175_175591

noncomputable def discount_percentage : ℝ := 0.1
noncomputable def original_price : ℝ := 10
noncomputable def discount_amount := discount_percentage * original_price
noncomputable def final_price := original_price - discount_amount

theorem rose_paid_after_discount : final_price = 9 := by
  sorry

end rose_paid_after_discount_l175_175591


namespace rectangle_length_width_difference_l175_175093

theorem rectangle_length_width_difference :
  ∃ (length width : ℕ), (length * width = 864) ∧ (length + width = 60) ∧ (length - width = 12) :=
by
  sorry

end rectangle_length_width_difference_l175_175093


namespace Emily_age_is_23_l175_175293

variable (UncleBob Daniel Emily Zoe : ℕ)

-- Conditions
axiom h1 : UncleBob = 54
axiom h2 : Daniel = UncleBob / 2
axiom h3 : Emily = Daniel - 4
axiom h4 : Emily = 2 * Zoe / 3

-- Question: Prove that Emily's age is 23
theorem Emily_age_is_23 : Emily = 23 :=
by
  sorry

end Emily_age_is_23_l175_175293


namespace packets_of_candy_bought_l175_175575

theorem packets_of_candy_bought
    (candies_per_day_weekday : ℕ)
    (candies_per_day_weekend : ℕ)
    (days_weekday : ℕ)
    (days_weekend : ℕ)
    (weeks : ℕ)
    (candies_per_packet : ℕ)
    (total_candies : ℕ)
    (packets_bought : ℕ) :
    candies_per_day_weekday = 2 →
    candies_per_day_weekend = 1 →
    days_weekday = 5 →
    days_weekend = 2 →
    weeks = 3 →
    candies_per_packet = 18 →
    total_candies = (candies_per_day_weekday * days_weekday + candies_per_day_weekend * days_weekend) * weeks →
    packets_bought = total_candies / candies_per_packet →
    packets_bought = 2 :=
by
  intros
  sorry

end packets_of_candy_bought_l175_175575


namespace total_boxes_stacked_l175_175338

/-- Definitions used in conditions --/
def box_width : ℕ := 1
def box_length : ℕ := 1
def land_width : ℕ := 44
def land_length : ℕ := 35
def first_day_layers : ℕ := 7
def second_day_layers : ℕ := 3

/-- Theorem stating the number of boxes stacked in two days --/
theorem total_boxes_stacked : first_day_layers * (land_width * land_length) + second_day_layers * (land_width * land_length) = 15400 := by
  sorry

end total_boxes_stacked_l175_175338


namespace books_read_l175_175961

-- Given conditions
def chapters_per_book : ℕ := 17
def total_chapters_read : ℕ := 68

-- Statement to prove
theorem books_read : (total_chapters_read / chapters_per_book) = 4 := 
by sorry

end books_read_l175_175961


namespace question_1_question_2_l175_175897

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def vect_a : ℝ × ℝ := (3, 2)
def vect_b : ℝ × ℝ := (-1, 2)
def vect_c : ℝ × ℝ := (4, 1)

theorem question_1 :
  3 • vect_a + vect_b - 2 • vect_c = (0, 6) := 
by
  sorry

theorem question_2 (k : ℝ) : 
  let lhs := (3 + 4 * k) * 2
  let rhs := -5 * (2 + k)
  (lhs = rhs) → k = -16 / 13 := 
by
  sorry

end question_1_question_2_l175_175897


namespace given_conditions_implies_correct_answer_l175_175624

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer_l175_175624


namespace largest_y_coordinate_of_degenerate_ellipse_l175_175206

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + (y + 5)^2 / 16 = 0) → y = -5 :=
by
  intros x y h
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l175_175206


namespace simplify_expression1_simplify_expression2_l175_175096

-- Problem 1
theorem simplify_expression1 (x y : ℤ) :
  (-3) * x + 2 * y - 5 * x - 7 * y = -8 * x - 5 * y :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℤ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 :=
by sorry

end simplify_expression1_simplify_expression2_l175_175096


namespace students_in_class_l175_175832

theorem students_in_class {S : ℕ} 
  (h1 : 20 < S)
  (h2 : S < 30)
  (chess_club_condition : ∃ (n : ℕ), S = 3 * n) 
  (draughts_club_condition : ∃ (m : ℕ), S = 4 * m) : 
  S = 24 := 
sorry

end students_in_class_l175_175832


namespace rectangle_area_unchanged_l175_175198

theorem rectangle_area_unchanged (x y : ℕ) (h1 : x * y = (x + 5/2) * (y - 2/3)) (h2 : x * y = (x - 5/2) * (y + 4/3)) : x * y = 20 :=
by
  sorry

end rectangle_area_unchanged_l175_175198


namespace simplify_fraction_l175_175979

theorem simplify_fraction :
  (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 :=
by
  sorry

end simplify_fraction_l175_175979


namespace minimum_value_of_w_l175_175763

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l175_175763


namespace correct_fraction_order_l175_175495

noncomputable def fraction_ordering : Prop := 
  (16 / 12 < 18 / 13) ∧ (18 / 13 < 21 / 14) ∧ (21 / 14 < 20 / 15)

theorem correct_fraction_order : fraction_ordering := 
by {
  repeat { sorry }
}

end correct_fraction_order_l175_175495


namespace population_factor_proof_l175_175499

-- Define the conditions given in the problem
variables (N x y z : ℕ)

theorem population_factor_proof :
  (N = x^2) ∧ (N + 100 = y^2 + 1) ∧ (N + 200 = z^2) → (7 ∣ N) :=
by sorry

end population_factor_proof_l175_175499


namespace starting_player_wins_by_taking_2_white_first_l175_175323

-- Define initial setup
def initial_blue_balls : ℕ := 15
def initial_white_balls : ℕ := 12

-- Define conditions of the game
def can_take_blue_balls (n : ℕ) : Prop := n % 3 = 0
def can_take_white_balls (n : ℕ) : Prop := n % 2 = 0
def player_win_condition (blue white : ℕ) : Prop := 
  (blue = 0 ∧ white = 0)

-- Define the game strategy to establish and maintain the ratio 3/2
def maintain_ratio (blue white : ℕ) : Prop := blue * 2 = white * 3

-- Prove that the starting player should take 2 white balls first to ensure winning
theorem starting_player_wins_by_taking_2_white_first :
  (can_take_white_balls 2) →
  maintain_ratio initial_blue_balls (initial_white_balls - 2) →
  ∀ (blue white : ℕ), player_win_condition blue white :=
by
  intros h_take_white h_maintain_ratio blue white
  sorry

end starting_player_wins_by_taking_2_white_first_l175_175323


namespace sufficient_condition_for_q_l175_175785

def p (a : ℝ) : Prop := a ≥ 0
def q (a : ℝ) : Prop := a^2 + a ≥ 0

theorem sufficient_condition_for_q (a : ℝ) : p a → q a := by 
  sorry

end sufficient_condition_for_q_l175_175785


namespace cos_A_minus_B_l175_175612

variable {A B : ℝ}

-- Conditions
def cos_conditions (A B : ℝ) : Prop :=
  (Real.cos A + Real.cos B = 1 / 2)

def sin_conditions (A B : ℝ) : Prop :=
  (Real.sin A + Real.sin B = 3 / 2)

-- Mathematically equivalent proof problem
theorem cos_A_minus_B (h1 : cos_conditions A B) (h2 : sin_conditions A B) :
  Real.cos (A - B) = 1 / 4 := 
sorry

end cos_A_minus_B_l175_175612


namespace magic_square_sum_l175_175059

variable {a b c d e : ℕ}

-- Given conditions:
-- It's a magic square and the sums of the numbers in each row, column, and diagonal are equal.
-- Positions and known values specified:
theorem magic_square_sum (h : 15 + 24 = 18 + c ∧ 18 + c = 27 + a ∧ c = 21 ∧ a = 12 ∧ e = 17 ∧ d = 30 ∧ b = 25)
: d + e = 47 :=
by
  -- Sorry used to skip the proof
  sorry

end magic_square_sum_l175_175059


namespace juice_cost_l175_175252

-- Given conditions
def sandwich_cost : ℝ := 0.30
def total_money : ℝ := 2.50
def num_friends : ℕ := 4

-- Cost calculation
def total_sandwich_cost : ℝ := num_friends * sandwich_cost
def remaining_money : ℝ := total_money - total_sandwich_cost

-- The theorem to prove
theorem juice_cost : (remaining_money / num_friends) = 0.325 := by
  sorry

end juice_cost_l175_175252


namespace geometric_sequence_common_ratio_l175_175830

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum_ratio : (a 0 + a 1 + a 2) / a 2 = 7) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l175_175830


namespace man_speed_in_still_water_l175_175017

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water_l175_175017


namespace find_side_length_l175_175985

theorem find_side_length
  (a b c : ℝ) 
  (cosine_diff_angle : ℝ) 
  (h_b : b = 5)
  (h_c : c = 4)
  (h_cosine_diff_angle : cosine_diff_angle = 31 / 32) :
  a = 6 := 
sorry

end find_side_length_l175_175985


namespace perfect_square_value_of_b_l175_175513

theorem perfect_square_value_of_b :
  (∃ b : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + b * b) = (11.98 + b)^2) →
  (∃ b : ℝ, b = 0.02) :=
sorry

end perfect_square_value_of_b_l175_175513


namespace ways_to_draw_balls_eq_total_ways_l175_175726

noncomputable def ways_to_draw_balls (n : Nat) :=
  if h : n = 15 then (15 * 14 * 13 * 12) else 0

noncomputable def valid_combinations : Nat := sorry

noncomputable def total_ways_to_draw : Nat :=
  valid_combinations * 24

theorem ways_to_draw_balls_eq_total_ways :
  ways_to_draw_balls 15 = total_ways_to_draw :=
sorry

end ways_to_draw_balls_eq_total_ways_l175_175726


namespace ratio_of_speeds_l175_175869

noncomputable def speed_ratios (d t_b t : ℚ) : ℚ × ℚ  :=
  let d_b := t_b * t
  let d_a := d - d_b
  let t_h := t / 60
  let s_a := d_a / t_h
  let s_b := t_b
  (s_a / 15, s_b / 15)

theorem ratio_of_speeds
  (d : ℚ) (s_b : ℚ) (t : ℚ)
  (h : d = 88) (h1 : s_b = 90) (h2 : t = 32) :
  speed_ratios d s_b t = (5, 6) :=
  by
  sorry

end ratio_of_speeds_l175_175869


namespace museum_pictures_l175_175728

theorem museum_pictures (P : ℕ) (h1 : ¬ (∃ k, P = 2 * k)) (h2 : ∃ k, P + 1 = 2 * k) : P = 3 := 
by 
  sorry

end museum_pictures_l175_175728


namespace rectangular_field_area_l175_175091

theorem rectangular_field_area :
  ∃ (w l : ℝ), w = l / 3 ∧ 2 * (w + l) = 72 ∧ w * l = 243 :=
by
  sorry

end rectangular_field_area_l175_175091


namespace dot_product_to_linear_form_l175_175581

noncomputable def proof_problem (r a : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := a.1
  let B := a.2
  let C := -m
  (r.1 * a.1 + r.2 * a.2 = m) → (A * r.1 + B * r.2 + C = 0)

-- The theorem statement
theorem dot_product_to_linear_form (r a : ℝ × ℝ) (m : ℝ) :
  proof_problem r a m :=
sorry

end dot_product_to_linear_form_l175_175581


namespace cost_of_building_fence_eq_3944_l175_175024

def area_square : ℕ := 289
def price_per_foot : ℕ := 58

theorem cost_of_building_fence_eq_3944 : 
  let side_length := (area_square : ℝ) ^ (1/2)
  let perimeter := 4 * side_length
  let cost := perimeter * (price_per_foot : ℝ)
  cost = 3944 :=
by
  sorry

end cost_of_building_fence_eq_3944_l175_175024


namespace zero_in_set_zero_l175_175749

-- Define that 0 is an element
def zero_element : Prop := true

-- Define that {0} is a set containing only the element 0
def set_zero : Set ℕ := {0}

-- The main theorem that proves 0 ∈ {0}
theorem zero_in_set_zero (h : zero_element) : 0 ∈ set_zero := 
by sorry

end zero_in_set_zero_l175_175749


namespace cover_large_square_l175_175710

theorem cover_large_square :
  ∃ (small_squares : Fin 8 → Set (ℝ × ℝ)),
    (∀ i, small_squares i = {p : ℝ × ℝ | (p.1 - x_i)^2 + (p.2 - y_i)^2 < (3/2)^2}) ∧
    (∃ (large_square : Set (ℝ × ℝ)),
      large_square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 7 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7} ∧
      large_square ⊆ ⋃ i, small_squares i) :=
sorry

end cover_large_square_l175_175710


namespace apple_percentage_is_23_l175_175434

def total_responses := 70 + 80 + 50 + 30 + 70
def apple_responses := 70

theorem apple_percentage_is_23 :
  (apple_responses : ℝ) / (total_responses : ℝ) * 100 = 23 := 
by
  sorry

end apple_percentage_is_23_l175_175434


namespace smallest_a1_l175_175855

theorem smallest_a1 (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - n) :
  a 1 ≥ 13 / 36 :=
by
  sorry

end smallest_a1_l175_175855


namespace ratio_square_pentagon_l175_175490

theorem ratio_square_pentagon (P_sq P_pent : ℕ) 
  (h_sq : P_sq = 60) (h_pent : P_pent = 60) :
  (P_sq / 4) / (P_pent / 5) = 5 / 4 :=
by 
  sorry

end ratio_square_pentagon_l175_175490


namespace odd_increasing_three_digit_numbers_count_eq_50_l175_175123

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l175_175123


namespace sqrt_nine_eq_three_l175_175584

theorem sqrt_nine_eq_three : Real.sqrt 9 = 3 :=
by
  sorry

end sqrt_nine_eq_three_l175_175584


namespace probability_of_B_l175_175459

variables (A B : Prop)
variables (P : Prop → ℝ) -- Probability Measure

axiom A_and_B : P (A ∧ B) = 0.15
axiom not_A_and_not_B : P (¬A ∧ ¬B) = 0.6

theorem probability_of_B : P B = 0.15 :=
by
  sorry

end probability_of_B_l175_175459


namespace angle_C_is_sixty_l175_175150

variable {A B C D E : Type}
variable {AD BE BC AC : ℝ}
variable {triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A} 
variable (angle_C : ℝ)

-- Given conditions
variable (h_eq : AD * BC = BE * AC)
variable (h_ineq : AC ≠ BC)

-- To prove
theorem angle_C_is_sixty (h_eq : AD * BC = BE * AC) (h_ineq : AC ≠ BC) : angle_C = 60 :=
by
  sorry

end angle_C_is_sixty_l175_175150


namespace number_of_strictly_increasing_sequences_l175_175853

def strictly_increasing_sequences (n : ℕ) : ℕ :=
if n = 0 then 1 else if n = 1 then 1 else strictly_increasing_sequences (n - 1) + strictly_increasing_sequences (n - 2)

theorem number_of_strictly_increasing_sequences :
  strictly_increasing_sequences 12 = 144 :=
by
  sorry

end number_of_strictly_increasing_sequences_l175_175853


namespace cubic_difference_pos_l175_175356

theorem cubic_difference_pos {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cubic_difference_pos_l175_175356


namespace problem_statement_l175_175187

theorem problem_statement (a : ℝ) :
  (∀ x : ℝ, (1/2 < x ∧ x < 2 → ax^2 + 5 * x - 2 > 0)) →
  a = -2 ∧ (∀ x : ℝ, -3 < x ∧ x < (1/2) → ax^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end problem_statement_l175_175187


namespace speed_of_stream_l175_175076

variable (b s : ℝ)

-- Define the conditions from the problem
def downstream_condition := (100 : ℝ) / 4 = b + s
def upstream_condition := (75 : ℝ) / 15 = b - s

theorem speed_of_stream (h1 : downstream_condition b s) (h2: upstream_condition b s) : s = 10 := 
by 
  sorry

end speed_of_stream_l175_175076


namespace find_sum_principal_l175_175176

theorem find_sum_principal (P R : ℝ) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 150 → P = 300 :=
by
  sorry

end find_sum_principal_l175_175176


namespace tan_triple_angle_l175_175917

theorem tan_triple_angle (theta : ℝ) (h : Real.tan theta = 3) : Real.tan (3 * theta) = 9 / 13 :=
by
  -- to be completed
  sorry

end tan_triple_angle_l175_175917


namespace number_of_members_l175_175922

theorem number_of_members (n h : ℕ) (h1 : n * n * h = 362525) : n = 5 :=
sorry

end number_of_members_l175_175922


namespace box_with_20_aluminium_80_plastic_weighs_494_l175_175712

def weight_of_box_with_100_aluminium_balls := 510 -- in grams
def weight_of_box_with_100_plastic_balls := 490 -- in grams
def number_of_aluminium_balls := 100
def number_of_plastic_balls := 100

-- Define the weights per ball type by subtracting the weight of the box
def weight_per_aluminium_ball := (weight_of_box_with_100_aluminium_balls - weight_of_box_with_100_plastic_balls) / number_of_aluminium_balls
def weight_per_plastic_ball := (weight_of_box_with_100_plastic_balls - weight_of_box_with_100_plastic_balls) / number_of_plastic_balls

-- Condition: The weight of the box alone (since it's present in both conditions)
def weight_of_empty_box := weight_of_box_with_100_plastic_balls - (weight_per_plastic_ball * number_of_plastic_balls)

-- Function to compute weight of the box with given number of aluminium and plastic balls
def total_weight (num_al : ℕ) (num_pl : ℕ) : ℕ :=
  weight_of_empty_box + (weight_per_aluminium_ball * num_al) + (weight_per_plastic_ball * num_pl)

-- The theorem to be proven
theorem box_with_20_aluminium_80_plastic_weighs_494 :
  total_weight 20 80 = 494 := sorry

end box_with_20_aluminium_80_plastic_weighs_494_l175_175712


namespace acute_triangle_cannot_divide_into_two_obtuse_l175_175606

def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem acute_triangle_cannot_divide_into_two_obtuse (A B C A1 B1 C1 A2 B2 C2 : ℝ) 
  (h_acute : is_acute_triangle A B C) 
  (h_divide : A + B + C = 180 ∧ A1 + B1 + C1 = 180 ∧ A2 + B2 + C2 = 180)
  (h_sum : A1 + A2 = A ∧ B1 + B2 = B ∧ C1 + C2 = C) :
  ¬ (is_obtuse_triangle A1 B1 C1 ∧ is_obtuse_triangle A2 B2 C2) :=
sorry

end acute_triangle_cannot_divide_into_two_obtuse_l175_175606


namespace main_line_train_probability_l175_175653

noncomputable def probability_catching_main_line (start_main_line start_harbor_line : Nat) (frequency : Nat) : ℝ :=
  if start_main_line % frequency = 0 ∧ start_harbor_line % frequency = 2 then 1 / 2 else 0

theorem main_line_train_probability :
  probability_catching_main_line 0 2 10 = 1 / 2 :=
by
  sorry

end main_line_train_probability_l175_175653


namespace beta_max_success_ratio_l175_175341

-- Define Beta's score conditions
variables (a b c d : ℕ)
def beta_score_conditions :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) ∧
  (a * 25 < b * 9) ∧
  (c * 25 < d * 17) ∧
  (b + d = 600)

-- Define Beta's success ratio
def beta_success_ratio :=
  (a + c) / 600

theorem beta_max_success_ratio :
  beta_score_conditions a b c d →
  beta_success_ratio a c ≤ 407 / 600 :=
sorry

end beta_max_success_ratio_l175_175341


namespace find_triples_l175_175315

theorem find_triples :
  { (a, b, c) : ℕ × ℕ × ℕ | (c-1) * (a * b - b - a) = a + b - 2 } =
  { (2, 1, 0), (1, 2, 0), (3, 4, 2), (4, 3, 2), (1, 0, 2), (0, 1, 2), (2, 4, 3), (4, 2, 3) } :=
by
  sorry

end find_triples_l175_175315


namespace eq_system_correct_l175_175943

theorem eq_system_correct (x y : ℤ) : 
  (7 * x + 7 = y) ∧ (9 * (x - 1) = y) :=
sorry

end eq_system_correct_l175_175943


namespace find_k_l175_175417

-- Identifying conditions from the problem
def point (x : ℝ) : ℝ × ℝ := (x, x^3)  -- A point on the curve y = x^3
def tangent_slope (x : ℝ) : ℝ := 3 * x^2  -- The slope of the tangent to the curve y = x^3 at point (x, x^3)
def tangent_line (x k : ℝ) : ℝ := k * x + 2  -- The given tangent line equation

-- Question as a proof problem
theorem find_k (x : ℝ) (k : ℝ) (h : tangent_line x k = x^3) : k = 3 :=
by
  sorry

end find_k_l175_175417


namespace tim_sarah_age_ratio_l175_175951

theorem tim_sarah_age_ratio :
  ∀ (x : ℕ), ∃ (t s : ℕ),
    t = 23 ∧ s = 11 ∧
    (23 + x) * 2 = (11 + x) * 3 → x = 13 :=
by
  sorry

end tim_sarah_age_ratio_l175_175951


namespace probability_of_winning_at_least_10_rubles_l175_175340

-- Definitions based on conditions
def total_tickets : ℕ := 100
def win_20_rubles_tickets : ℕ := 5
def win_15_rubles_tickets : ℕ := 10
def win_10_rubles_tickets : ℕ := 15
def win_2_rubles_tickets : ℕ := 25
def win_nothing_tickets : ℕ := total_tickets - (win_20_rubles_tickets + win_15_rubles_tickets + win_10_rubles_tickets + win_2_rubles_tickets)

-- Probability calculations
def prob_win_20_rubles : ℚ := win_20_rubles_tickets / total_tickets
def prob_win_15_rubles : ℚ := win_15_rubles_tickets / total_tickets
def prob_win_10_rubles : ℚ := win_10_rubles_tickets / total_tickets

-- Prove the probability of winning at least 10 rubles
theorem probability_of_winning_at_least_10_rubles : 
  prob_win_20_rubles + prob_win_15_rubles + prob_win_10_rubles = 0.30 := by
  sorry

end probability_of_winning_at_least_10_rubles_l175_175340


namespace travel_times_l175_175748

variable (t v1 v2 : ℝ)

def conditions := 
  (v1 * 2 = v2 * t) ∧ 
  (v2 * 4.5 = v1 * t)

theorem travel_times (h : conditions t v1 v2) : 
  t = 3 ∧ 
  (t + 2 = 5) ∧ 
  (t + 4.5 = 7.5) := by
  sorry

end travel_times_l175_175748


namespace circumscribed_triangle_area_relation_l175_175555

theorem circumscribed_triangle_area_relation
    (a b c: ℝ) (h₀: a = 8) (h₁: b = 15) (h₂: c = 17)
    (triangle_area: ℝ) (circle_area: ℝ) (X Y Z: ℝ)
    (hZ: Z > X) (hXY: X < Y)
    (triangle_area_calc: triangle_area = 60)
    (circle_area_calc: circle_area = π * (c / 2)^2) :
    X + Y = Z := by
  sorry

end circumscribed_triangle_area_relation_l175_175555


namespace fraction_zero_implies_x_eq_neg3_l175_175478

theorem fraction_zero_implies_x_eq_neg3 (x : ℝ) (h1 : x ≠ 3) (h2 : (x^2 - 9) / (x - 3) = 0) : x = -3 :=
sorry

end fraction_zero_implies_x_eq_neg3_l175_175478


namespace factorize1_factorize2_factorize3_l175_175950

theorem factorize1 (x : ℝ) : x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2 := 
  sorry

theorem factorize2 (x y : ℝ) : 16 * x^2 - 9 * y^2 = (4 * x - 3 * y) * (4 * x + 3 * y) := 
  sorry

theorem factorize3 (x y : ℝ) : (3 * x + y)^2 - (x - 3 * y) * (3 * x + y) = 2 * (3 * x + y) * (x + 2 * y) := 
  sorry

end factorize1_factorize2_factorize3_l175_175950


namespace percentage_increase_l175_175930

variable (S : ℝ) (P : ℝ)
variable (h1 : S + 0.10 * S = 330)
variable (h2 : S + P * S = 324)

theorem percentage_increase : P = 0.08 := sorry

end percentage_increase_l175_175930


namespace dilution_problem_l175_175255

/-- Samantha needs to add 7.2 ounces of water to achieve a 25% alcohol concentration
given that she starts with 12 ounces of solution containing 40% alcohol. -/
theorem dilution_problem (x : ℝ) : (12 + x) * 0.25 = 4.8 ↔ x = 7.2 :=
by sorry

end dilution_problem_l175_175255


namespace digits_in_2_pow_120_l175_175607

theorem digits_in_2_pow_120 {a b : ℕ} (h : 10^a ≤ 2^200 ∧ 2^200 < 10^b) (ha : a = 60) (hb : b = 61) : 
  ∃ n : ℕ, 10^(n-1) ≤ 2^120 ∧ 2^120 < 10^n ∧ n = 37 :=
by {
  sorry
}

end digits_in_2_pow_120_l175_175607


namespace length_of_diagonal_l175_175619

open Real

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, -a^2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (-a, -a^2)
noncomputable def C (a : ℝ) : ℝ × ℝ := (a, -a^2)
def O : ℝ × ℝ := (0, 0)

noncomputable def is_square (A B O C : ℝ × ℝ) : Prop :=
  dist A B = dist B O ∧ dist B O = dist O C ∧ dist O C = dist C A

theorem length_of_diagonal (a : ℝ) (h_square : is_square (A a) (B a) O (C a)) : 
  dist (A a) (C a) = 2 * abs a :=
sorry

end length_of_diagonal_l175_175619


namespace car_distance_ratio_l175_175927

theorem car_distance_ratio (t : ℝ) (h₁ : t > 0)
    (speed_A speed_B : ℝ)
    (h₂ : speed_A = 70)
    (h₃ : speed_B = 35)
    (ratio : ℝ)
    (h₄ : ratio = 2)
    (h_time : ∀ a b : ℝ, a * t = b * t → a = b) :
  (speed_A * t) / (speed_B * t) = ratio := by
  sorry

end car_distance_ratio_l175_175927


namespace fraction_of_female_attendees_on_time_l175_175766

theorem fraction_of_female_attendees_on_time (A : ℝ)
  (h1 : 3 / 5 * A = M)
  (h2 : 7 / 8 * M = M_on_time)
  (h3 : 0.115 * A = n_A_not_on_time) :
  0.9 * F = (A - M_on_time - n_A_not_on_time)/((2 / 5) * A - n_A_not_on_time) :=
by
  sorry

end fraction_of_female_attendees_on_time_l175_175766


namespace chocolate_cost_in_promotion_l175_175097

/-!
Bernie buys two chocolates every week at a local store, where one chocolate costs $3.
In a different store with a promotion, each chocolate costs some amount and Bernie would save $6 
in three weeks if he bought his chocolates there. Prove that the cost of one chocolate 
in the store with the promotion is $2.
-/

theorem chocolate_cost_in_promotion {n p_local savings : ℕ} (weeks : ℕ) (p_promo : ℕ)
  (h_n : n = 2)
  (h_local : p_local = 3)
  (h_savings : savings = 6)
  (h_weeks : weeks = 3)
  (h_promo : p_promo = (p_local * n * weeks - savings) / (n * weeks)) :
  p_promo = 2 :=
by {
  -- Proof would go here
  sorry
}

end chocolate_cost_in_promotion_l175_175097


namespace student_B_more_stable_than_A_student_B_more_stable_l175_175163

-- Define students A and B.
structure Student :=
  (average_score : ℝ)
  (variance : ℝ)

-- Given data for both students.
def studentA : Student :=
  { average_score := 90, variance := 51 }

def studentB : Student :=
  { average_score := 90, variance := 12 }

-- The theorem that student B has more stable performance than student A.
theorem student_B_more_stable_than_A (A B : Student) (h_avg : A.average_score = B.average_score) :
  A.variance > B.variance → B.variance < A.variance :=
by
  intro h
  linarith

-- Specific instance of the theorem with given data for students A and B.
theorem student_B_more_stable : studentA.variance > studentB.variance → studentB.variance < studentA.variance :=
  student_B_more_stable_than_A studentA studentB rfl

end student_B_more_stable_than_A_student_B_more_stable_l175_175163


namespace find_third_number_l175_175372

theorem find_third_number (x y z : ℝ) 
  (h1 : y = 3 * x - 7)
  (h2 : z = 2 * x + 2)
  (h3 : x + y + z = 168) : z = 60 :=
sorry

end find_third_number_l175_175372


namespace four_digit_numbers_starting_with_1_l175_175848

theorem four_digit_numbers_starting_with_1 
: ∃ n : ℕ, (n = 234) ∧ 
  (∀ (x y z : ℕ), 
    (x ≠ y → x ≠ z → y ≠ z → -- ensuring these constraints
    x ≠ 1 → y ≠ 1 → z = 1 → -- exactly two identical digits which include 1
    (x * 1000 + y * 100 + z * 10 + 1) / 1000 = 1 ∨ (x * 1000 + z * 100 + y * 10 + 1) / 1000 = 1) ∨ 
    (∃ (x y : ℕ),  
    (x ≠ y → x ≠ 1 → y = 1 → 
    (x * 110 + y * 10 + 1) + (x * 11 + y * 10 + 1) + (x * 100 + y * 10 + 1) + (x * 110 + 1) = n))) := sorry

end four_digit_numbers_starting_with_1_l175_175848


namespace greatest_diff_l175_175295

theorem greatest_diff (x y : ℤ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) : y - x = 7 :=
sorry

end greatest_diff_l175_175295


namespace scientific_notation_of_number_l175_175186

theorem scientific_notation_of_number : 15300000000 = 1.53 * (10 : ℝ)^10 := sorry

end scientific_notation_of_number_l175_175186


namespace julio_lost_15_fish_l175_175604

def fish_caught_per_hour : ℕ := 7
def hours_fished : ℕ := 9
def fish_total_without_loss : ℕ := fish_caught_per_hour * hours_fished
def fish_total_actual : ℕ := 48
def fish_lost : ℕ := fish_total_without_loss - fish_total_actual

theorem julio_lost_15_fish : fish_lost = 15 := by
  sorry

end julio_lost_15_fish_l175_175604


namespace ben_final_salary_is_2705_l175_175408

def initial_salary : ℕ := 3000

def salary_after_raise (salary : ℕ) : ℕ :=
  salary * 110 / 100

def salary_after_pay_cut (salary : ℕ) : ℕ :=
  salary * 85 / 100

def final_salary (initial : ℕ) : ℕ :=
  (salary_after_pay_cut (salary_after_raise initial)) - 100

theorem ben_final_salary_is_2705 : final_salary initial_salary = 2705 := 
by 
  sorry

end ben_final_salary_is_2705_l175_175408


namespace find_a_l175_175794

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l175_175794


namespace angle_sum_of_octagon_and_triangle_l175_175599

-- Define the problem setup
def is_interior_angle_of_regular_polygon (n : ℕ) (angle : ℝ) : Prop :=
  angle = 180 * (n - 2) / n

def is_regular_octagon_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 8 angle

def is_equilateral_triangle_angle (angle : ℝ) : Prop :=
  is_interior_angle_of_regular_polygon 3 angle

-- The statement of the problem
theorem angle_sum_of_octagon_and_triangle :
  ∃ angle_ABC angle_ABD : ℝ,
    is_regular_octagon_angle angle_ABC ∧
    is_equilateral_triangle_angle angle_ABD ∧
    angle_ABC + angle_ABD = 195 :=
sorry

end angle_sum_of_octagon_and_triangle_l175_175599


namespace friend_saves_per_week_l175_175824

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l175_175824


namespace cuboid_volume_l175_175727

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 18) (h_height : height = 8) : 
  base_area * height = 144 :=
by
  rw [h_base_area, h_height]
  norm_num

end cuboid_volume_l175_175727


namespace total_problems_l175_175227

-- We define the conditions as provided.
variables (p t : ℕ) -- p and t are positive whole numbers
variables (p_gt_10 : 10 < p) -- p is more than 10

theorem total_problems (p t : ℕ) (p_gt_10 : 10 < p) (h : p * t = (2 * p - 4) * (t - 2)):
  p * t = 60 :=
by
  sorry

end total_problems_l175_175227


namespace transformed_center_coordinates_l175_175399

theorem transformed_center_coordinates (S : (ℝ × ℝ)) (hS : S = (3, -4)) : 
  let reflected_S := (S.1, -S.2)
  let translated_S := (reflected_S.1, reflected_S.2 + 5)
  translated_S = (3, 9) :=
by
  sorry

end transformed_center_coordinates_l175_175399


namespace sufficient_not_necessary_condition_l175_175751

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem sufficient_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  sorry

end sufficient_not_necessary_condition_l175_175751


namespace three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l175_175213

theorem three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3 :
  ∃ x1 x2 x3 : ℕ, ((x1 = 414 ∧ x2 = 444 ∧ x3 = 474) ∧ 
  (∀ n, (100 * 4 + 10 * n + 4 = x1 ∨ 100 * 4 + 10 * n + 4 = x2 ∨ 100 * 4 + 10 * n + 4 = x3) 
  → (100 * 4 + 10 * n + 4) % 3 = 0)) :=
by
  sorry

end three_digit_numbers_with_units_and_hundreds_digit_4_divisible_by_3_l175_175213


namespace max_people_transition_l175_175524

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end max_people_transition_l175_175524


namespace find_triangle_value_l175_175381

variables (triangle q r : ℝ)
variables (h1 : triangle + q = 75) (h2 : triangle + q + r = 138) (h3 : r = q / 3)

theorem find_triangle_value : triangle = -114 :=
by
  sorry

end find_triangle_value_l175_175381


namespace teresa_age_when_michiko_born_l175_175686

theorem teresa_age_when_michiko_born (teresa_current_age morio_current_age morio_age_when_michiko_born : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : morio_age_when_michiko_born = 38) : 
  teresa_current_age - (morio_current_age - morio_age_when_michiko_born) = 26 := 
by 
  sorry

end teresa_age_when_michiko_born_l175_175686


namespace problem_statement_l175_175283

variable (a : ℝ)

theorem problem_statement (h : 5 = a + a⁻¹) : a^4 + (a⁻¹)^4 = 527 := 
by 
  sorry

end problem_statement_l175_175283


namespace yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l175_175369

-- Defining "Yazhong point"
def yazhong (A B M : ℝ) : Prop := abs (M - A) = abs (M - B)

-- Problem 1
theorem yazhong_point_1 {A B M : ℝ} (hA : A = -5) (hB : B = 1) (hM : yazhong A B M) : M = -2 :=
sorry

-- Problem 2
theorem yazhong_point_2 {A B M : ℝ} (hM : M = 2) (hAB : B - A = 9) (h_order : A < B) (hY : yazhong A B M) :
  (A = -5/2) ∧ (B = 13/2) :=
sorry

-- Problem 3 Part ①
theorem yazhong_point_3_part1 (A : ℝ) (B : ℝ) (m : ℤ) 
  (hA : A = -6) (hB_range : -4 ≤ B ∧ B ≤ -2) (hM : yazhong A B m) : 
  m = -5 ∨ m = -4 :=
sorry

-- Problem 3 Part ②
theorem yazhong_point_3_part2 (C D : ℝ) (n : ℤ)
  (hC : C = -4) (hD : D = -2) (hM : yazhong (-6) (C + D + 2 * n) 0) : 
  8 ≤ n ∧ n ≤ 10 :=
sorry

end yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l175_175369


namespace eccentricity_of_ellipse_l175_175529

theorem eccentricity_of_ellipse (a c : ℝ) (h1 : 2 * c = a) : (c / a) = (1 / 2) :=
by
  -- This is where we would write the proof, but we're using sorry to skip the proof steps.
  sorry

end eccentricity_of_ellipse_l175_175529


namespace min_value_of_expression_l175_175973

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 4) :
  (9 / x + 1 / y + 25 / z) ≥ 20.25 :=
by 
  sorry

end min_value_of_expression_l175_175973


namespace intersection_A_B_l175_175743

-- Conditions
def A : Set (ℕ × ℕ) := { (1, 2), (2, 1) }
def B : Set (ℕ × ℕ) := { p | p.fst - p.snd = 1 }

-- Problem statement
theorem intersection_A_B : A ∩ B = { (2, 1) } :=
by
  sorry

end intersection_A_B_l175_175743


namespace sequence_term_25_l175_175182

theorem sequence_term_25 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, n ≥ 2 → a n = (a (n - 1) + a (n + 1)) / 4)
  (h2 : a 1 = 1)
  (h3 : a 9 = 40545) : 
  a 25 = 57424611447841 := 
sorry

end sequence_term_25_l175_175182


namespace binary_operation_result_l175_175963

theorem binary_operation_result :
  let a := 0b1101
  let b := 0b111
  let c := 0b1010
  let d := 0b1001
  a + b - c + d = 0b10011 :=
by {
  sorry
}

end binary_operation_result_l175_175963


namespace product_of_N_l175_175370

theorem product_of_N (M L : ℝ) (N : ℝ) 
  (h1 : M = L + N) 
  (h2 : ∀ M4 L4 : ℝ, M4 = M - 7 → L4 = L + 5 → |M4 - L4| = 4) :
  N = 16 ∨ N = 8 ∧ (16 * 8 = 128) := 
by 
  sorry

end product_of_N_l175_175370


namespace positive_rationals_in_S_l175_175925

variable (S : Set ℚ)

-- Conditions
axiom closed_under_addition (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a + b ∈ S
axiom closed_under_multiplication (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a * b ∈ S
axiom zero_rule : ∀ r : ℚ, r ∈ S ∨ -r ∈ S ∨ r = 0

-- Prove that S is the set of positive rational numbers
theorem positive_rationals_in_S : S = {r : ℚ | 0 < r} :=
by
  sorry

end positive_rationals_in_S_l175_175925


namespace geometric_sequence_a3_l175_175643

theorem geometric_sequence_a3 (a : ℕ → ℝ)
  (h : ∀ n m : ℕ, a (n + m) = a n * a m)
  (pos : ∀ n, 0 < a n)
  (a1 : a 1 = 1)
  (a5 : a 5 = 9) :
  a 3 = 3 := by
  sorry

end geometric_sequence_a3_l175_175643


namespace theta_in_first_quadrant_l175_175006

noncomputable def quadrant_of_theta (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) : ℕ :=
  if 0 < Real.sin theta ∧ 0 < Real.cos theta then 1 else sorry

theorem theta_in_first_quadrant (theta : ℝ) (h1 : Real.sin (Real.pi + theta) < 0) (h2 : Real.cos (Real.pi - theta) < 0) :
  quadrant_of_theta theta h1 h2 = 1 :=
by
  sorry

end theta_in_first_quadrant_l175_175006


namespace optimal_addition_amount_l175_175183

theorem optimal_addition_amount (a b g : ℝ) (h₁ : a = 628) (h₂ : b = 774) (h₃ : g = 718) : 
    b + a - g = 684 :=
by
  sorry

end optimal_addition_amount_l175_175183


namespace cost_of_single_room_l175_175695

theorem cost_of_single_room
  (total_rooms : ℕ)
  (double_rooms : ℕ)
  (cost_double_room : ℕ)
  (revenue_total : ℕ)
  (cost_single_room : ℕ)
  (H1 : total_rooms = 260)
  (H2 : double_rooms = 196)
  (H3 : cost_double_room = 60)
  (H4 : revenue_total = 14000)
  (H5 : revenue_total = (total_rooms - double_rooms) * cost_single_room + double_rooms * cost_double_room)
  : cost_single_room = 35 :=
sorry

end cost_of_single_room_l175_175695


namespace ratio_of_kids_l175_175388

theorem ratio_of_kids (k2004 k2005 k2006 : ℕ) 
  (h2004: k2004 = 60) 
  (h2005: k2005 = k2004 / 2)
  (h2006: k2006 = 20) :
  (k2006 : ℚ) / k2005 = 2 / 3 :=
by
  sorry

end ratio_of_kids_l175_175388


namespace expand_polynomial_l175_175236

variable (x : ℝ)

theorem expand_polynomial :
  2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 :=
by
  sorry

end expand_polynomial_l175_175236


namespace sum_powers_divisible_by_13_l175_175077

-- Statement of the problem in Lean
theorem sum_powers_divisible_by_13 (a b p : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : p = 13) :
  (a^1974 + b^1974) % p = 0 := 
by
  sorry

end sum_powers_divisible_by_13_l175_175077


namespace proof_x_plus_y_equals_30_l175_175887

variable (x y : ℝ) (h_distinct : x ≠ y)
variable (h_det : Matrix.det ![
  ![2, 5, 10],
  ![4, x, y],
  ![4, y, x]
  ] = 0)

theorem proof_x_plus_y_equals_30 :
  x + y = 30 :=
sorry

end proof_x_plus_y_equals_30_l175_175887


namespace positive_integers_expressible_l175_175673

theorem positive_integers_expressible :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (x^2 + y) / (x * y + 1) = 1 ∧
  ∃ (x' y' : ℕ), (x' > 0) ∧ (y' > 0) ∧ (x' ≠ x ∨ y' ≠ y) ∧ (x'^2 + y') / (x' * y' + 1) = 1 :=
by
  sorry

end positive_integers_expressible_l175_175673


namespace cost_first_third_hour_l175_175318

theorem cost_first_third_hour 
  (c : ℝ) 
  (h1 : 0 < c) 
  (h2 : ∀ t : ℝ, t > 1/4 → (t - 1/4) * 12 + c = 31)
  : c = 5 :=
by
  sorry

end cost_first_third_hour_l175_175318


namespace maximize_profit_l175_175872

-- Define constants for purchase and selling prices
def priceA_purchase : ℝ := 16
def priceA_selling : ℝ := 20
def priceB_purchase : ℝ := 20
def priceB_selling : ℝ := 25

-- Define constant for total weight
def total_weight : ℝ := 200

-- Define profit function
def profit (weightA weightB : ℝ) : ℝ :=
  (priceA_selling - priceA_purchase) * weightA + (priceB_selling - priceB_purchase) * weightB

-- Define constraints
def constraint1 (weightA weightB : ℝ) : Prop :=
  weightA + weightB = total_weight

def constraint2 (weightA weightB : ℝ) : Prop :=
  weightA >= 3 * weightB

open Real

-- Define the maximum profit we aim to prove
def max_profit : ℝ := 850

-- The main theorem to prove
theorem maximize_profit : 
  ∃ weightA weightB : ℝ, constraint1 weightA weightB ∧ constraint2 weightA weightB ∧ profit weightA weightB = max_profit :=
by {
  sorry
}

end maximize_profit_l175_175872


namespace watch_A_accurate_l175_175060

variable (T : ℕ) -- Standard time, represented as natural numbers for simplicity
variable (A B : ℕ) -- Watches A and B, also represented as natural numbers
variable (h1 : A = B + 2) -- Watch A is 2 minutes faster than Watch B
variable (h2 : B = T - 2) -- Watch B is 2 minutes slower than the standard time

theorem watch_A_accurate : A = T :=
by
  -- The proof would go here
  sorry

end watch_A_accurate_l175_175060


namespace prod_gcd_lcm_eq_864_l175_175709

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end prod_gcd_lcm_eq_864_l175_175709


namespace abc_zero_iff_quadratic_identities_l175_175620

variable {a b c : ℝ}

theorem abc_zero_iff_quadratic_identities (h : ¬(a = b ∧ b = c ∧ c = a)) : 
  a + b + c = 0 ↔ a^2 + ab + b^2 = b^2 + bc + c^2 ∧ b^2 + bc + c^2 = c^2 + ca + a^2 :=
by
  sorry

end abc_zero_iff_quadratic_identities_l175_175620


namespace find_sum_squares_l175_175385

variables (x y : ℝ)

theorem find_sum_squares (h1 : y + 4 = (x - 2)^2) (h2 : x + 4 = (y - 2)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 15 :=
sorry

end find_sum_squares_l175_175385


namespace find_g_2_l175_175849

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_2
  (H : ∀ (x : ℝ), x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x ^ 2):
  g 2 = 67 / 14 :=
by
  sorry

end find_g_2_l175_175849


namespace no_nonconstant_arithmetic_progression_l175_175181

theorem no_nonconstant_arithmetic_progression (x : ℝ) :
  2 * (2 : ℝ)^(x^2) ≠ (2 : ℝ)^x + (2 : ℝ)^(x^3) :=
sorry

end no_nonconstant_arithmetic_progression_l175_175181


namespace rulers_left_in_drawer_l175_175067

theorem rulers_left_in_drawer (initial_rulers taken_rulers : ℕ) (h1 : initial_rulers = 46) (h2 : taken_rulers = 25) :
  initial_rulers - taken_rulers = 21 :=
by
  sorry

end rulers_left_in_drawer_l175_175067


namespace min_value_of_expr_l175_175598

theorem min_value_of_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = (1 / a) + (1 / b)) :
  ∃ x : ℝ, x = (1 / a) + (2 / b) ∧ x = 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l175_175598


namespace quadratic_sum_is_zero_l175_175724

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end quadratic_sum_is_zero_l175_175724


namespace base4_sum_correct_l175_175288

/-- Define the base-4 numbers as natural numbers. -/
def a := 3 * 4^2 + 1 * 4^1 + 2 * 4^0
def b := 3 * 4^1 + 1 * 4^0
def c := 3 * 4^0

/-- Define their sum in base 10. -/
def sum_base_10 := a + b + c

/-- Define the target sum in base 4 as a natural number. -/
def target := 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0

/-- Prove that the sum of the base-4 numbers equals the target sum in base 4. -/
theorem base4_sum_correct : sum_base_10 = target := by
  sorry

end base4_sum_correct_l175_175288


namespace crackers_per_person_l175_175877

theorem crackers_per_person:
  ∀ (total_crackers friends : ℕ), total_crackers = 36 → friends = 18 → total_crackers / friends = 2 :=
by
  intros total_crackers friends h1 h2
  sorry

end crackers_per_person_l175_175877


namespace olive_charged_10_hours_l175_175634

/-- If Olive charges her phone for 3/5 of the time she charged last night, and that results
    in 12 hours of use, where each hour of charge results in 2 hours of phone usage,
    then the time Olive charged her phone last night was 10 hours. -/
theorem olive_charged_10_hours (x : ℝ) 
  (h1 : 2 * (3 / 5) * x = 12) : 
  x = 10 :=
by
  sorry

end olive_charged_10_hours_l175_175634


namespace fraction_division_correct_l175_175890

theorem fraction_division_correct :
  (2 / 5) / 3 = 2 / 15 :=
by sorry

end fraction_division_correct_l175_175890


namespace evaluate_expression_l175_175518

theorem evaluate_expression : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := 
sorry

end evaluate_expression_l175_175518


namespace flowers_per_vase_l175_175008

theorem flowers_per_vase (carnations roses vases total_flowers flowers_per_vase : ℕ)
  (h1 : carnations = 7)
  (h2 : roses = 47)
  (h3 : vases = 9)
  (h4 : total_flowers = carnations + roses)
  (h5 : flowers_per_vase = total_flowers / vases):
  flowers_per_vase = 6 := 
by {
  sorry
}

end flowers_per_vase_l175_175008


namespace cylinder_volume_eq_pi_over_4_l175_175760

theorem cylinder_volume_eq_pi_over_4
  (r : ℝ)
  (h₀ : r > 0)
  (h₁ : 2 * r = r * 2)
  (h₂ : 4 * π * r^2 = π) : 
  (π * r^2 * (2 * r) = π / 4) :=
by
  sorry

end cylinder_volume_eq_pi_over_4_l175_175760


namespace football_games_per_month_l175_175528

theorem football_games_per_month :
  let total_games := 5491
  let months := 17.0
  total_games / months = 323 := 
by
  let total_games := 5491
  let months := 17.0
  -- This is where the actual computation would happen if we were to provide a proof
  sorry

end football_games_per_month_l175_175528


namespace diagonal_cubes_140_320_360_l175_175075

-- Define the problem parameters 
def length_x : ℕ := 140
def length_y : ℕ := 320
def length_z : ℕ := 360

-- Define the function to calculate the number of unit cubes the internal diagonal passes through.
def num_cubes_diagonal (x y z : ℕ) : ℕ :=
  x + y + z - Nat.gcd x y - Nat.gcd y z - Nat.gcd z x + Nat.gcd (Nat.gcd x y) z

-- The target theorem to be proven
theorem diagonal_cubes_140_320_360 :
  num_cubes_diagonal length_x length_y length_z = 760 :=
by
  sorry

end diagonal_cubes_140_320_360_l175_175075


namespace simplify_expression_l175_175676

theorem simplify_expression (x y : ℝ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 :=
by
  rw [hx, hy]
  -- here we would simplify but leave a hole
  sorry

end simplify_expression_l175_175676


namespace number_of_students_l175_175102

theorem number_of_students 
  (n : ℕ)
  (h1: 108 - 36 = 72)
  (h2: ∀ n > 0, 108 / n - 72 / n = 3) :
  n = 12 :=
sorry

end number_of_students_l175_175102


namespace charge_difference_is_51_l175_175613

-- Define the charges and calculations for print shop X
def print_shop_x_cost (n : ℕ) : ℝ :=
  if n ≤ 50 then n * 1.20 else 50 * 1.20 + (n - 50) * 0.90

-- Define the charges and calculations for print shop Y
def print_shop_y_cost (n : ℕ) : ℝ :=
  10 + n * 1.70

-- Define the difference in charges for 70 copies
def charge_difference : ℝ :=
  print_shop_y_cost 70 - print_shop_x_cost 70

-- The proof statement
theorem charge_difference_is_51 : charge_difference = 51 :=
by
  sorry

end charge_difference_is_51_l175_175613


namespace quadratic_distinct_real_roots_l175_175274

theorem quadratic_distinct_real_roots (k : ℝ) :
  (k > -2 ∧ k ≠ 0) ↔ ( ∃ (a b c : ℝ), a = k ∧ b = -4 ∧ c = -2 ∧ (b^2 - 4 * a * c) > 0) :=
by
  sorry

end quadratic_distinct_real_roots_l175_175274


namespace yearly_feeding_cost_l175_175938

-- Defining the conditions
def num_geckos := 3
def num_iguanas := 2
def num_snakes := 4

def cost_per_snake_per_month := 10
def cost_per_iguana_per_month := 5
def cost_per_gecko_per_month := 15

-- Statement of the proof problem
theorem yearly_feeding_cost : 
  (num_snakes * cost_per_snake_per_month + num_iguanas * cost_per_iguana_per_month + num_geckos * cost_per_gecko_per_month) * 12 = 1140 := 
  by 
    sorry

end yearly_feeding_cost_l175_175938


namespace max_digit_d_for_number_divisible_by_33_l175_175767

theorem max_digit_d_for_number_divisible_by_33 : ∃ d e : ℕ, d ≤ 9 ∧ e ≤ 9 ∧ 8 * 100000 + d * 10000 + 8 * 1000 + 3 * 100 + 3 * 10 + e % 33 = 0 ∧  d = 8 :=
by {
  sorry
}

end max_digit_d_for_number_divisible_by_33_l175_175767


namespace distance_between_trees_l175_175391

theorem distance_between_trees
  (num_trees : ℕ)
  (length_of_yard : ℝ)
  (one_tree_at_each_end : True)
  (h1 : num_trees = 26)
  (h2 : length_of_yard = 400) :
  length_of_yard / (num_trees - 1) = 16 :=
by
  sorry

end distance_between_trees_l175_175391


namespace evaluate_expression_l175_175666

variable (x y : ℝ)

theorem evaluate_expression
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hsum_sq : x^2 + y^2 ≠ 0)
  (hsum : x + y ≠ 0) :
    (x^2 + y^2)⁻¹ * ((x + y)⁻¹ + (x / y)⁻¹) = (1 + y) / ((x^2 + y^2) * (x + y)) :=
sorry

end evaluate_expression_l175_175666


namespace find_subtracted_value_l175_175460

theorem find_subtracted_value (n x : ℕ) (h₁ : n = 36) (h₂ : ((n + 10) * 2 / 2 - x) = 44) : x = 2 :=
by
  sorry

end find_subtracted_value_l175_175460


namespace length_of_faster_train_l175_175208

theorem length_of_faster_train
    (speed_faster : ℕ)
    (speed_slower : ℕ)
    (time_cross : ℕ)
    (h_fast : speed_faster = 72)
    (h_slow : speed_slower = 36)
    (h_time : time_cross = 15) :
    (speed_faster - speed_slower) * (1000 / 3600) * time_cross = 150 := 
by
  sorry

end length_of_faster_train_l175_175208


namespace quadratic_equation_has_root_l175_175013

theorem quadratic_equation_has_root (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + 2 * b * x + c = 0) ∨
             (b * x^2 + 2 * c * x + a = 0) ∨
             (c * x^2 + 2 * a * x + b = 0) :=
sorry

end quadratic_equation_has_root_l175_175013


namespace four_digits_sum_l175_175847

theorem four_digits_sum (A B C D : ℕ) 
  (A_neq_B : A ≠ B) (A_neq_C : A ≠ C) (A_neq_D : A ≠ D) 
  (B_neq_C : B ≠ C) (B_neq_D : B ≠ D) 
  (C_neq_D : C ≠ D)
  (digits_A : A ≤ 9) (digits_B : B ≤ 9) (digits_C : C ≤ 9) (digits_D : D ≤ 9)
  (A_lt_B : A < B) 
  (minimize_fraction : ∃ k : ℕ, (A + B) = k ∧ k ≤ (A + B) ∧ (C + D) ≥ (C + D)) :
  C + D = 17 := 
by
  sorry

end four_digits_sum_l175_175847


namespace compare_08_and_one_eighth_l175_175775

theorem compare_08_and_one_eighth :
  0.8 - (1 / 8 : ℝ) = 0.675 := 
sorry

end compare_08_and_one_eighth_l175_175775


namespace consecutive_sum_to_20_has_one_set_l175_175398

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l175_175398


namespace general_formula_correct_S_k_equals_189_l175_175120

-- Define the arithmetic sequence with initial conditions
def a (n : ℕ) : ℤ :=
  if n = 1 then -11
  else sorry  -- Will be defined by the general formula

-- Given conditions in Lean
def initial_condition (a : ℕ → ℤ) :=
  a 1 = -11 ∧ a 4 + a 6 = -6

-- General formula for the arithmetic sequence to be proven
def general_formula (n : ℕ) : ℤ := 2 * n - 13

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℤ :=
  n^2 - 12 * n

-- Problem 1: Prove the general formula
theorem general_formula_correct : ∀ n : ℕ, initial_condition a → a n = general_formula n :=
by sorry

-- Problem 2: Prove that k = 21 such that S_k = 189
theorem S_k_equals_189 : ∃ k : ℕ, S k = 189 ∧ k = 21 :=
by sorry

end general_formula_correct_S_k_equals_189_l175_175120


namespace range_of_a_l175_175911

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) 
  (h2 : log_base a (a^2 + 1) < log_base a (2 * a))
  (h3 : log_base a (2 * a) < 0) : a ∈ Set.Ioo (0.5) 1 := 
sorry

end range_of_a_l175_175911


namespace ordered_notebooks_amount_l175_175212

def initial_notebooks : ℕ := 10
def ordered_notebooks (x : ℕ) : ℕ := x
def lost_notebooks : ℕ := 2
def current_notebooks : ℕ := 14

theorem ordered_notebooks_amount (x : ℕ) (h : initial_notebooks + ordered_notebooks x - lost_notebooks = current_notebooks) : x = 6 :=
by
  sorry

end ordered_notebooks_amount_l175_175212


namespace math_competition_correct_answers_l175_175055

theorem math_competition_correct_answers (qA qB cA cB : ℕ) 
  (h_total_questions : qA + qB = 10)
  (h_score_A : cA * 5 - (qA - cA) * 2 = 36)
  (h_score_B : cB * 5 - (qB - cB) * 2 = 22) 
  (h_combined_score : cA * 5 - (qA - cA) * 2 + cB * 5 - (qB - cB) * 2 = 58)
  (h_score_difference : cA * 5 - (qA - cA) * 2 - (cB * 5 - (qB - cB) * 2) = 14) : 
  cA = 8 :=
by {
  sorry
}

end math_competition_correct_answers_l175_175055


namespace evaluate_dollar_l175_175112

variable {R : Type} [CommRing R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : 
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) :=
by
  sorry

end evaluate_dollar_l175_175112


namespace gcd_of_3150_and_9800_is_350_l175_175473

-- Definition of the two numbers
def num1 : ℕ := 3150
def num2 : ℕ := 9800

-- The greatest common factor of num1 and num2 is 350
theorem gcd_of_3150_and_9800_is_350 : Nat.gcd num1 num2 = 350 := by
  sorry

end gcd_of_3150_and_9800_is_350_l175_175473


namespace six_pow_2n_plus1_plus_1_div_by_7_l175_175557

theorem six_pow_2n_plus1_plus_1_div_by_7 (n : ℕ) : (6^(2*n+1) + 1) % 7 = 0 := by
  sorry

end six_pow_2n_plus1_plus_1_div_by_7_l175_175557


namespace water_consumed_l175_175567

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end water_consumed_l175_175567


namespace weight_of_daughter_l175_175501

variable (M D G S : ℝ)

theorem weight_of_daughter :
  M + D + G + S = 200 →
  D + G = 60 →
  G = M / 5 →
  S = 2 * D →
  D = 800 / 15 :=
by
  intros h1 h2 h3 h4
  sorry

end weight_of_daughter_l175_175501


namespace problem_l175_175533

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end problem_l175_175533


namespace survey_respondents_l175_175143

theorem survey_respondents
  (X Y Z : ℕ) 
  (h1 : X = 360) 
  (h2 : X * 4 = Y * 9) 
  (h3 : X * 3 = Z * 9) : 
  X + Y + Z = 640 :=
by
  sorry

end survey_respondents_l175_175143


namespace students_joined_l175_175074

theorem students_joined (A X : ℕ) (h1 : 100 * A = 5000) (h2 : (100 + X) * (A - 10) = 5400) :
  X = 35 :=
by
  sorry

end students_joined_l175_175074


namespace no_infinite_arithmetic_progression_l175_175297

open Classical

variable {R : Type*} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem no_infinite_arithmetic_progression
  (f_strict_inc : ∀ x y : R, 0 < x ∧ 0 < y → x < y → f x < f y)
  (f_convex : ∀ x y : R, 0 < x ∧ 0 < y → f ((x + y) / 2) < (f x + f y) / 2) :
  ∀ a : ℕ → R, (∀ n : ℕ, a n = f n) → ¬(∃ d : R, ∀ k : ℕ, a (k + 1) - a k = d) :=
sorry

end no_infinite_arithmetic_progression_l175_175297


namespace problem_statement_l175_175271

theorem problem_statement (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (r s : ℕ)
  (consecutive_primes : Nat.Prime r ∧ Nat.Prime s ∧ (r + 1 = s ∨ s + 1 = r))
  (roots_condition : r + s = p ∧ r * s = 2 * q) :
  (r * s = 2 * q) ∧ (Nat.Prime (p^2 - 2 * q)) ∧ (Nat.Prime (p + 2 * q)) :=
by
  sorry

end problem_statement_l175_175271


namespace mary_total_earnings_l175_175136

-- Define the earnings for each job
def cleaning_earnings (homes_cleaned : ℕ) : ℕ := 46 * homes_cleaned
def babysitting_earnings (days_babysat : ℕ) : ℕ := 35 * days_babysat
def petcare_earnings (days_petcare : ℕ) : ℕ := 60 * days_petcare

-- Define the total earnings
def total_earnings (homes_cleaned days_babysat days_petcare : ℕ) : ℕ :=
  cleaning_earnings homes_cleaned + babysitting_earnings days_babysat + petcare_earnings days_petcare

-- Given values
def homes_cleaned_last_week : ℕ := 4
def days_babysat_last_week : ℕ := 5
def days_petcare_last_week : ℕ := 3

-- Prove the total earnings
theorem mary_total_earnings : total_earnings homes_cleaned_last_week days_babysat_last_week days_petcare_last_week = 539 :=
by
  -- We just state the theorem; the proof is not required
  sorry

end mary_total_earnings_l175_175136


namespace solve_for_x_l175_175675

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := 
by
  sorry

end solve_for_x_l175_175675


namespace pqrs_sum_l175_175012

noncomputable def distinct_real_numbers (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem pqrs_sum (p q r s : ℝ) 
  (h1 : r + s = 12 * p)
  (h2 : r * s = -13 * q)
  (h3 : p + q = 12 * r)
  (h4 : p * q = -13 * s)
  (distinct : distinct_real_numbers p q r s) :
  p + q + r + s = -13 :=
  sorry

end pqrs_sum_l175_175012


namespace factorize_x_squared_minus_sixteen_l175_175949

theorem factorize_x_squared_minus_sixteen (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) :=
by
  sorry

end factorize_x_squared_minus_sixteen_l175_175949


namespace first_day_is_sunday_l175_175429

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end first_day_is_sunday_l175_175429


namespace range_of_a_l175_175942

theorem range_of_a (x a : ℝ) :
  (∀ x : ℝ, x - 1 < 0 ∧ x < a + 3 → x < 1) → a ≥ -2 :=
by
  sorry

end range_of_a_l175_175942


namespace minimum_A_l175_175272

noncomputable def minA : ℝ := (1 + Real.sqrt 2) / 2

theorem minimum_A (x y z w : ℝ) (A : ℝ) 
    (h : xy + 2 * yz + zw ≤ A * (x^2 + y^2 + z^2 + w^2)) :
    A ≥ minA := 
sorry

end minimum_A_l175_175272


namespace simplify_expression_l175_175151

theorem simplify_expression (a : ℝ) (h : a = -2) : 
  (1 - a / (a + 1)) / (1 / (1 - a ^ 2)) = 1 / 3 :=
by
  subst h
  sorry

end simplify_expression_l175_175151


namespace fraction_of_integer_l175_175663

theorem fraction_of_integer :
  (5 / 6) * 30 = 25 :=
by
  sorry

end fraction_of_integer_l175_175663


namespace right_triangle_sets_l175_175992

theorem right_triangle_sets :
  ∃! (a b c : ℕ), 
    ((a = 5 ∧ b = 12 ∧ c = 13) ∧ a * a + b * b = c * c) ∧ 
    ¬(∃ a b c, (a = 3 ∧ b = 4 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 4 ∧ b = 5 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 5 ∧ b = 7 ∧ c = 9) ∧ a * a + b * b = c * c) :=
by {
  --- proof needed
  sorry
}

end right_triangle_sets_l175_175992


namespace prove_nabla_squared_l175_175828

theorem prove_nabla_squared:
  ∃ (odot nabla : ℕ), odot < 20 ∧ nabla < 20 ∧ odot ≠ nabla ∧
  (nabla * nabla * odot = nabla) ∧ (nabla * nabla = 64) :=
by
  sorry

end prove_nabla_squared_l175_175828


namespace dora_rate_correct_l175_175054

noncomputable def betty_rate : ℕ := 10
noncomputable def dora_rate : ℕ := 8
noncomputable def total_time : ℕ := 5
noncomputable def betty_break_time : ℕ := 2
noncomputable def cupcakes_difference : ℕ := 10

theorem dora_rate_correct :
  ∃ D : ℕ, 
  (D = dora_rate) ∧ 
  ((total_time - betty_break_time) * betty_rate = 30) ∧ 
  (total_time * D - 30 = cupcakes_difference) :=
sorry

end dora_rate_correct_l175_175054


namespace initial_deposit_l175_175811

/-- 
A person deposits some money in a bank at an interest rate of 7% per annum (of the original amount). 
After two years, the total amount in the bank is $6384. Prove that the initial amount deposited is $5600.
-/
theorem initial_deposit (P : ℝ) (h : (P + 0.07 * P) + 0.07 * P = 6384) : P = 5600 :=
by
  sorry

end initial_deposit_l175_175811


namespace parallel_lines_l175_175867

theorem parallel_lines (a : ℝ) (h : ∀ x y : ℝ, 2*x - a*y - 1 = 0 → a*x - y = 0) : a = Real.sqrt 2 ∨ a = -Real.sqrt 2 :=
sorry

end parallel_lines_l175_175867


namespace unique_k_solves_eq_l175_175395

theorem unique_k_solves_eq (k : ℕ) (hpos_k : k > 0) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = k * a * b) ↔ k = 2 :=
by
  sorry

end unique_k_solves_eq_l175_175395


namespace number_of_schools_is_23_l175_175955

-- Conditions and definitions
noncomputable def number_of_students_per_school : ℕ := 3
def beth_rank : ℕ := 37
def carla_rank : ℕ := 64

-- Statement of the proof problem
theorem number_of_schools_is_23
  (n : ℕ)
  (h1 : ∀ i < n, ∃ r1 r2 r3: ℕ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h2 : ∀ i < n, ∃ A B C: ℕ, A = (2 * B + 1) ∧ C = A ∧ B = 35 ∧ A < beth_rank ∧ beth_rank < carla_rank):
  n = 23 :=
by
  sorry

end number_of_schools_is_23_l175_175955


namespace mary_total_payment_l175_175364

def fixed_fee : ℕ := 17
def hourly_charge : ℕ := 7
def rental_duration : ℕ := 9
def total_payment (f : ℕ) (h : ℕ) (r : ℕ) : ℕ := f + (h * r)

theorem mary_total_payment:
  total_payment fixed_fee hourly_charge rental_duration = 80 :=
by
  sorry

end mary_total_payment_l175_175364


namespace chord_through_P_midpoint_of_ellipse_has_given_line_l175_175138

-- Define the ellipse
def ellipse (x y : ℝ) := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def pointP := (3, 1)

-- Define the problem statement
theorem chord_through_P_midpoint_of_ellipse_has_given_line:
  ∃ (m : ℝ) (c : ℝ), (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 144 → x + y = m ∧ 3 * x + y = c) → 
  ∃ (A : ℝ) (B : ℝ), ellipse 3 1 ∧ (A * 4 + B * 3 - 15 = 0) := sorry

end chord_through_P_midpoint_of_ellipse_has_given_line_l175_175138


namespace prime_sequence_constant_l175_175415

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Condition: There exists a constant sequence of primes such that the given recurrence relation holds.
theorem prime_sequence_constant (p : ℕ) (k : ℤ) (n : ℕ) 
  (h1 : 1 ≤ n)
  (h2 : ∀ m ≥ 1, is_prime (p + m))
  (h3 : p + k = p + p + k) :
  ∀ m ≥ 1, p + m = p :=
sorry

end prime_sequence_constant_l175_175415


namespace find_b_neg_l175_175795

noncomputable def h (x : ℝ) : ℝ := if x ≤ 0 then -x else 3 * x - 50

theorem find_b_neg (b : ℝ) (h_neg_b : b < 0) : 
  h (h (h 15)) = h (h (h b)) → b = - (55 / 3) :=
by
  sorry

end find_b_neg_l175_175795


namespace fraction_uncovered_l175_175778

def area_rug (length width : ℕ) : ℕ := length * width
def area_square (side : ℕ) : ℕ := side * side

theorem fraction_uncovered 
  (rug_length rug_width floor_area : ℕ)
  (h_rug_length : rug_length = 2)
  (h_rug_width : rug_width = 7)
  (h_floor_area : floor_area = 64)
  : (floor_area - area_rug rug_length rug_width) / floor_area = 25 / 32 := 
sorry

end fraction_uncovered_l175_175778


namespace goods_train_pass_time_l175_175791

theorem goods_train_pass_time 
  (speed_mans_train_kmph : ℝ) (speed_goods_train_kmph : ℝ) (length_goods_train_m : ℝ) :
  speed_mans_train_kmph = 20 → 
  speed_goods_train_kmph = 92 → 
  length_goods_train_m = 280 → 
  abs ((length_goods_train_m / ((speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600)) - 8.99) < 0.01 :=
by
  sorry

end goods_train_pass_time_l175_175791


namespace remainder_2pow33_minus_1_div_9_l175_175148

theorem remainder_2pow33_minus_1_div_9 : (2^33 - 1) % 9 = 7 := 
  sorry

end remainder_2pow33_minus_1_div_9_l175_175148


namespace Rachel_father_age_when_Rachel_is_25_l175_175909

-- Define the problem conditions:
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Prove the age of Rachel's father when she is 25 years old:
theorem Rachel_father_age_when_Rachel_is_25 : 
  Father_age + (25 - Rachel_age) = 60 := by
    sorry

end Rachel_father_age_when_Rachel_is_25_l175_175909


namespace obtuse_triangle_side_range_l175_175821

theorem obtuse_triangle_side_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : (a + 4)^2 > a^2 + (a + 2)^2)
  (h3 : (a + 2)^2 + (a + 4)^2 < a^2) : 
  2 < a ∧ a < 6 := 
sorry

end obtuse_triangle_side_range_l175_175821


namespace maximum_candies_purchase_l175_175426

theorem maximum_candies_purchase (c1 : ℕ) (c4 : ℕ) (c7 : ℕ) (n : ℕ)
    (H_single : c1 = 1)
    (H_pack4  : c4 = 4)
    (H_cost4  : c4 = 3) 
    (H_pack7  : c7 = 7) 
    (H_cost7  : c7 = 4) 
    (H_budget : n = 10) :
    ∃ k : ℕ, k = 16 :=
by
    -- We'll skip the proof since the task requires only the statement
    sorry

end maximum_candies_purchase_l175_175426


namespace eight_disks_area_sum_final_result_l175_175216

theorem eight_disks_area_sum (r : ℝ) (C : ℝ) :
  C = 1 ∧ r = (Real.sqrt 2 + 1) / 2 → 
  8 * (π * (r ^ 2)) = 2 * π * (3 + 2 * Real.sqrt 2) :=
by
  intros h
  sorry

theorem final_result :
  let a := 6
  let b := 4
  let c := 2
  a + b + c = 12 :=
by
  intros
  norm_num

end eight_disks_area_sum_final_result_l175_175216


namespace baby_guppies_l175_175394

theorem baby_guppies (x : ℕ) (h1 : 7 + x + 9 = 52) : x = 36 :=
by
  sorry

end baby_guppies_l175_175394


namespace exists_n_l175_175211

theorem exists_n (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_gcd : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → ¬(2^n ∣ a^k + b^k + c^k) :=
by
  sorry

end exists_n_l175_175211


namespace find_x_l175_175166

theorem find_x
  (x : ℤ)
  (h : 3 * x + 3 * 15 + 3 * 18 + 11 = 152) :
  x = 14 :=
by
  sorry

end find_x_l175_175166


namespace part1_l175_175072

theorem part1 (a b c d : ℤ) (h : a * d - b * c = 1) : Int.gcd (a + b) (c + d) = 1 :=
sorry

end part1_l175_175072


namespace build_wall_time_l175_175454

theorem build_wall_time {d : ℝ} : 
  (15 * 1 + 3 * 2) * 3 = 63 ∧ 
  (25 * 1 + 5 * 2) * d = 63 → 
  d = 1.8 := 
by 
  sorry

end build_wall_time_l175_175454


namespace largest_difference_rounding_l175_175464

variable (A B : ℝ)
variable (estimate_A estimate_B : ℝ)
variable (within_A within_B : ℝ)
variable (diff : ℝ)

axiom est_A : estimate_A = 55000
axiom est_B : estimate_B = 58000
axiom cond_A : within_A = 0.15
axiom cond_B : within_B = 0.10

axiom bounds_A : 46750 ≤ A ∧ A ≤ 63250
axiom bounds_B : 52727 ≤ B ∧ B ≤ 64444

noncomputable def max_possible_difference : ℝ :=
  max (abs (B - A)) (abs (A - B))

theorem largest_difference_rounding :
  max_possible_difference A B = 18000 :=
by
  sorry

end largest_difference_rounding_l175_175464


namespace phi_value_for_unique_symmetry_center_l175_175986

theorem phi_value_for_unique_symmetry_center :
  ∃ (φ : ℝ), (0 < φ ∧ φ < π / 2) ∧
  (φ = π / 12 ∨ φ = π / 6 ∨ φ = π / 3 ∨ φ = 5 * π / 12) ∧
  ((∃ x : ℝ, 2 * x + φ = π ∧ π / 6 < x ∧ x < π / 3) ↔ φ = 5 * π / 12) :=
  sorry

end phi_value_for_unique_symmetry_center_l175_175986


namespace repeating_decimal_fraction_l175_175073

theorem repeating_decimal_fraction (x : ℚ) (h : x = 7.5656) : x = 749 / 99 :=
by
  sorry

end repeating_decimal_fraction_l175_175073


namespace min_value_of_expression_l175_175269

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end min_value_of_expression_l175_175269


namespace number_of_different_ways_to_travel_l175_175158

-- Define the conditions
def number_of_morning_flights : ℕ := 2
def number_of_afternoon_flights : ℕ := 3

-- Assert the question and the answer
theorem number_of_different_ways_to_travel : 
  (number_of_morning_flights * number_of_afternoon_flights) = 6 :=
by
  sorry

end number_of_different_ways_to_travel_l175_175158


namespace determine_b_l175_175996

noncomputable def f (x b : ℝ) : ℝ := x^3 - b * x^2 + 1/2

theorem determine_b (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) → b = 3/2 :=
by
  sorry

end determine_b_l175_175996


namespace problem1_problem2_l175_175876

theorem problem1 : -3 + (-2) * 5 - (-3) = -10 :=
by
  sorry

theorem problem2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 :=
by
  sorry

end problem1_problem2_l175_175876


namespace pradeep_max_marks_l175_175440

-- conditions
variables (M : ℝ)
variable (h1 : 0.40 * M = 220)

-- question and answer
theorem pradeep_max_marks : M = 550 :=
by
  sorry

end pradeep_max_marks_l175_175440


namespace areas_of_shared_parts_l175_175755

-- Define the areas of the non-overlapping parts
def area_non_overlap_1 : ℝ := 68
def area_non_overlap_2 : ℝ := 110
def area_non_overlap_3 : ℝ := 87

-- Define the total area of each circle
def total_area : ℝ := area_non_overlap_2 + area_non_overlap_3 - area_non_overlap_1

-- Define the areas of the shared parts A and B
def area_shared_A : ℝ := total_area - area_non_overlap_2
def area_shared_B : ℝ := total_area - area_non_overlap_3

-- Prove the areas of the shared parts
theorem areas_of_shared_parts :
  area_shared_A = 19 ∧ area_shared_B = 42 :=
by
  sorry

end areas_of_shared_parts_l175_175755


namespace sum_rational_irrational_not_rational_l175_175984

theorem sum_rational_irrational_not_rational (r i : ℚ) (hi : ¬ ∃ q : ℚ, i = q) : ¬ ∃ s : ℚ, r + i = s :=
by
  sorry

end sum_rational_irrational_not_rational_l175_175984


namespace altitude_inequality_not_universally_true_l175_175199

noncomputable def altitudes (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a m_b m_c : ℝ, m_a ≤ m_b ∧ m_b ≤ m_c 

noncomputable def seg_to_orthocenter (a b c : ℝ) (h : a ≥ b ∧ b ≥ c) : Prop :=
  ∃ m_a_star m_b_star m_c_star : ℝ, True

theorem altitude_inequality (a b c m_a m_b m_c : ℝ) 
  (h₀ : a ≥ b) (h₁ : b ≥ c) (h₂ : m_a ≤ m_b) (h₃ : m_b ≤ m_c) :
  (a + m_a ≥ b + m_b) ∧ (b + m_b ≥ c + m_c) :=
by
  sorry

theorem not_universally_true (a b c m_a_star m_b_star m_c_star : ℝ)
  (h₀ : a ≥ b) (h₁ : b ≥ c) :
  ¬(a + m_a_star ≥ b + m_b_star ∧ b + m_b_star ≥ c + m_c_star) :=
by
  sorry

end altitude_inequality_not_universally_true_l175_175199


namespace ratio_of_first_to_second_ball_l175_175477

theorem ratio_of_first_to_second_ball 
  (x y z : ℕ) 
  (h1 : 3 * x = 27) 
  (h2 : y = 18) 
  (h3 : z = 3 * x) : 
  x / y = 1 / 2 := 
sorry

end ratio_of_first_to_second_ball_l175_175477


namespace jane_stick_length_l175_175605

variable (P U S J F : ℕ)
variable (h1 : P = 30)
variable (h2 : U = P - 7)
variable (h3 : U = S / 2)
variable (h4 : F = 2 * 12)
variable (h5 : J = S - F)

theorem jane_stick_length : J = 22 := by
  sorry

end jane_stick_length_l175_175605


namespace aladdin_no_profit_l175_175043

theorem aladdin_no_profit (x : ℕ) :
  (x + 1023000) / 1024 <= x :=
by
  sorry

end aladdin_no_profit_l175_175043


namespace instantaneous_velocity_at_2_l175_175647

-- Define the motion equation
def s (t : ℝ) : ℝ := 3 + t^2

-- State the problem: Prove the instantaneous velocity at t = 2 is 4
theorem instantaneous_velocity_at_2 : (deriv s) 2 = 4 := by
  sorry

end instantaneous_velocity_at_2_l175_175647


namespace circles_externally_tangent_l175_175649

theorem circles_externally_tangent
  (r1 r2 d : ℝ)
  (hr1 : r1 = 2) (hr2 : r2 = 3)
  (hd : d = 5) :
  r1 + r2 = d :=
by
  sorry

end circles_externally_tangent_l175_175649


namespace cyclist_average_speed_l175_175554

theorem cyclist_average_speed (v : ℝ) 
  (h1 : 8 / v + 10 / 8 = 18 / 8.78) : v = 10 :=
by
  sorry

end cyclist_average_speed_l175_175554


namespace cos_C_correct_l175_175279

noncomputable def cos_C (B : ℝ) (AD BD : ℝ) : ℝ :=
  let sinB := Real.sin B
  let angleBAC := (2 : ℝ) * Real.arcsin ((Real.sqrt 3 / 3) * (sinB / 2)) -- derived from bisector property.
  let cosA := (2 : ℝ) * Real.cos angleBAC / 2 - 1
  let sinA := 2 * Real.sin angleBAC / 2 * Real.cos angleBAC / 2
  let cos2thirds := -1 / 2
  let sin2thirds := Real.sqrt 3 / 2
  cos2thirds * cosA + sin2thirds * sinA

theorem cos_C_correct : 
  ∀ (π : ℝ), 
  ∀ (A B C : ℝ),
  B = π / 3 →
  ∀ (AD : ℝ), AD = 3 →
  ∀ (BD : ℝ), BD = 2 →
  cos_C B AD BD = (2 * Real.sqrt 6 - 1) / 6 :=
by
  intros π A B C hB angleBisectorI hAD hBD
  sorry

end cos_C_correct_l175_175279


namespace problem_A_value_l175_175816

theorem problem_A_value (x y A : ℝ) (h : (x + 2 * y) ^ 2 = (x - 2 * y) ^ 2 + A) : A = 8 * x * y :=
by {
    sorry
}

end problem_A_value_l175_175816


namespace num_real_roots_of_eq_l175_175278

theorem num_real_roots_of_eq (x : ℝ) (h : x * |x| - 3 * |x| - 4 = 0) : 
  ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 :=
sorry

end num_real_roots_of_eq_l175_175278


namespace forty_percent_of_number_is_240_l175_175240

-- Define the conditions as assumptions in Lean
variable (N : ℝ)
variable (h1 : (1/4) * (1/3) * (2/5) * N = 20)

-- Prove that 40% of the number N is 240
theorem forty_percent_of_number_is_240 (h1: (1/4) * (1/3) * (2/5) * N = 20) : 0.40 * N = 240 :=
  sorry

end forty_percent_of_number_is_240_l175_175240


namespace find_certain_number_l175_175498

theorem find_certain_number (x : ℕ) (h : 220025 = (x + 445) * (2 * (x - 445)) + 25) : x = 555 :=
sorry

end find_certain_number_l175_175498


namespace sobhas_parents_age_difference_l175_175526

def difference_in_ages (F M : ℕ) : ℕ := F - M

theorem sobhas_parents_age_difference
  (S F M : ℕ)
  (h1 : F = S + 38)
  (h2 : M = S + 32) :
  difference_in_ages F M = 6 := by
  sorry

end sobhas_parents_age_difference_l175_175526


namespace olivia_earnings_this_week_l175_175846

variable (hourly_rate : ℕ) (hours_monday hours_wednesday hours_friday : ℕ)

theorem olivia_earnings_this_week : 
  hourly_rate = 9 → 
  hours_monday = 4 → 
  hours_wednesday = 3 → 
  hours_friday = 6 → 
  (hourly_rate * hours_monday + hourly_rate * hours_wednesday + hourly_rate * hours_friday) = 117 := 
by
  intros
  sorry

end olivia_earnings_this_week_l175_175846


namespace rate_of_current_l175_175777

def downstream_eq (b c : ℝ) : Prop := (b + c) * 4 = 24
def upstream_eq (b c : ℝ) : Prop := (b - c) * 6 = 24

theorem rate_of_current (b c : ℝ) (h1 : downstream_eq b c) (h2 : upstream_eq b c) : c = 1 :=
by sorry

end rate_of_current_l175_175777


namespace workshop_male_workers_l175_175104

variables (F M : ℕ)

theorem workshop_male_workers :
  (M = F + 45) ∧ (M - 5 = 3 * F) → M = 65 :=
by
  intros h
  sorry

end workshop_male_workers_l175_175104


namespace fraction_solution_l175_175541

theorem fraction_solution (a : ℤ) (h : 0 < a ∧ (a : ℚ) / (a + 36) = 775 / 1000) : a = 124 := 
by
  sorry

end fraction_solution_l175_175541


namespace simplify_and_evaluate_l175_175047

theorem simplify_and_evaluate (a : ℚ) (h : a = -1/6) : 
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5 / 2 := by
  rw [h]
  sorry

end simplify_and_evaluate_l175_175047


namespace q_at_2_equals_9_l175_175682

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

-- Define the function q(x)
noncomputable def q (x : ℝ) : ℝ :=
sgn (3 * x - 1) * |3 * x - 1| ^ (1/2) +
3 * sgn (3 * x - 1) * |3 * x - 1| ^ (1/3) +
|3 * x - 1| ^ (1/4)

-- The theorem stating that q(2) equals 9
theorem q_at_2_equals_9 : q 2 = 9 :=
by sorry

end q_at_2_equals_9_l175_175682


namespace next_meeting_time_at_B_l175_175719

-- Definitions of conditions
def perimeter := 800 -- Perimeter of the block in meters
def t1 := 1 -- They meet for the first time after 1 minute
def AB := 100 -- Length of side AB in meters
def BC := 300 -- Length of side BC in meters
def CD := 100 -- Length of side CD in meters
def DA := 300 -- Length of side DA in meters

-- Main theorem statement
theorem next_meeting_time_at_B :
  ∃ t : ℕ, t = 9 ∧ (∃ m1 m2 : ℕ, ((t = m1 * m2 + 1) ∧ m2 = 800 / (t1 * (AB + BC + CD + DA))) ∧ m1 = 9) :=
sorry

end next_meeting_time_at_B_l175_175719


namespace units_digit_of_7_power_19_l175_175071

theorem units_digit_of_7_power_19 : (7^19) % 10 = 3 := by
  sorry

end units_digit_of_7_power_19_l175_175071


namespace solve_for_d_l175_175713

theorem solve_for_d (r s t d c : ℝ)
  (h1 : (t = -r - s))
  (h2 : (c = rs + rt + st))
  (h3 : (t - 1 = -(r + 5) - (s - 4)))
  (h4 : (c = (r + 5) * (s - 4) + (r + 5) * (t - 1) + (s - 4) * (t - 1)))
  (h5 : (d = -r * s * t))
  (h6 : (d + 210 = -(r + 5) * (s - 4) * (t - 1))) :
  d = 240 ∨ d = 420 :=
by
  sorry

end solve_for_d_l175_175713


namespace quadratic_inequality_solution_set_l175_175502

theorem quadratic_inequality_solution_set {x : ℝ} :
  (x^2 + x - 2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end quadratic_inequality_solution_set_l175_175502


namespace coordinates_of_point_P_l175_175716

noncomputable def tangent_slope_4 : Prop :=
  ∀ (x y : ℝ), y = 1 / x → (-1 / (x^2)) = -4 → (x = 1 / 2 ∧ y = 2) ∨ (x = -1 / 2 ∧ y = -2)

theorem coordinates_of_point_P : tangent_slope_4 :=
by sorry

end coordinates_of_point_P_l175_175716


namespace math_problem_l175_175254

theorem math_problem (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = -1) :
  a^2 * b - 2 * a * b + a * b^2 = -1 :=
by
  sorry

end math_problem_l175_175254


namespace smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l175_175868

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) + 1

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem centers_of_symmetry :
  ∀ k : ℤ, ∃ x, x = -Real.pi / 4 + k * Real.pi ∧ f (-x) = f x := sorry

theorem maximum_value :
  ∀ x : ℝ, f x ≤ 2 := sorry

theorem minimum_value :
  ∀ x : ℝ, f x ≥ -1 := sorry

end smallest_positive_period_centers_of_symmetry_maximum_value_minimum_value_l175_175868


namespace increasing_function_of_a_l175_175863

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - (a / 2)) * x + 2

theorem increasing_function_of_a (a : ℝ) : (∀ x y, x < y → f a x ≤ f a y) ↔ 
  (8 / 3 ≤ a ∧ a < 4) :=
sorry

end increasing_function_of_a_l175_175863


namespace power_point_relative_to_circle_l175_175343

noncomputable def circle_power (a b R x1 y1 : ℝ) : ℝ :=
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2

theorem power_point_relative_to_circle (a b R x1 y1 : ℝ) :
  (x1 - a) ^ 2 + (y1 - b) ^ 2 - R ^ 2 = circle_power a b R x1 y1 := by
  unfold circle_power
  sorry

end power_point_relative_to_circle_l175_175343


namespace rewrite_expression_l175_175025

theorem rewrite_expression : ∀ x : ℝ, x^2 + 4 * x + 1 = (x + 2)^2 - 3 :=
by
  intros
  sorry

end rewrite_expression_l175_175025


namespace speed_in_m_per_s_eq_l175_175633

theorem speed_in_m_per_s_eq : (1 : ℝ) / 3.6 = (0.27777 : ℝ) :=
by sorry

end speed_in_m_per_s_eq_l175_175633


namespace min_abs_phi_l175_175956

open Real

theorem min_abs_phi {k : ℤ} :
  ∃ (φ : ℝ), ∀ (k : ℤ), φ = - (5 * π) / 6 + k * π ∧ |φ| = π / 6 := sorry

end min_abs_phi_l175_175956


namespace units_digit_m_squared_plus_3_to_m_l175_175325

theorem units_digit_m_squared_plus_3_to_m (m : ℕ) (h : m = 2021^2 + 3^2021) : (m^2 + 3^m) % 10 = 7 :=
by
  sorry

end units_digit_m_squared_plus_3_to_m_l175_175325


namespace incorrect_mode_l175_175798

theorem incorrect_mode (data : List ℕ) (hdata : data = [1, 2, 4, 3, 5]) : ¬ (∃ mode, mode = 5 ∧ (data.count mode > 1)) :=
by
  sorry

end incorrect_mode_l175_175798


namespace gcd_of_factorials_l175_175352

-- Define factorials
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define 7!
def seven_factorial : ℕ := factorial 7

-- Define (11! / 4!)
def eleven_div_four_factorial : ℕ := factorial 11 / factorial 4

-- GCD function based on prime factorization (though a direct gcd function also exists, we follow the steps)
def prime_factorization_gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Proof statement
theorem gcd_of_factorials : prime_factorization_gcd seven_factorial eleven_div_four_factorial = 5040 := by
  sorry

end gcd_of_factorials_l175_175352


namespace find_a7_l175_175350

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def Sn_for_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a7 (h_arith : arithmetic_sequence a)
  (h_sum_property : Sn_for_arithmetic_sequence a S)
  (h1 : a 2 + a 5 = 4)
  (h2 : S 7 = 21) :
  a 7 = 9 :=
sorry

end find_a7_l175_175350


namespace roots_of_cubic_l175_175661

-- Define the cubic equation having roots 3 and -2
def cubic_eq (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The proof problem statement
theorem roots_of_cubic (a b c d : ℝ) (h₁ : a ≠ 0)
  (h₂ : cubic_eq a b c d 3)
  (h₃ : cubic_eq a b c d (-2)) : 
  (b + c) / a = -7 := 
sorry

end roots_of_cubic_l175_175661


namespace average_sum_problem_l175_175908

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l175_175908


namespace horse_saddle_ratio_l175_175808

theorem horse_saddle_ratio (total_cost : ℕ) (saddle_cost : ℕ) (horse_cost : ℕ) 
  (h_total : total_cost = 5000)
  (h_saddle : saddle_cost = 1000)
  (h_sum : horse_cost + saddle_cost = total_cost) : 
  horse_cost / saddle_cost = 4 :=
by sorry

end horse_saddle_ratio_l175_175808


namespace quadratic_has_two_roots_l175_175496

theorem quadratic_has_two_roots (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 5 * a + b + 2 * c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 := 
  sorry

end quadratic_has_two_roots_l175_175496


namespace range_of_a_l175_175465

-- Definitions derived from conditions
def is_ellipse_with_foci_on_x_axis (a : ℝ) : Prop := a^2 > a + 6 ∧ a + 6 > 0

-- Theorem representing the proof problem
theorem range_of_a (a : ℝ) (h : is_ellipse_with_foci_on_x_axis a) :
  (a > 3) ∨ (-6 < a ∧ a < -2) :=
sorry

end range_of_a_l175_175465


namespace netSalePrice_correct_l175_175351

-- Definitions for item costs and fees
def purchaseCostA : ℝ := 650
def handlingFeeA : ℝ := 0.02 * purchaseCostA
def totalCostA : ℝ := purchaseCostA + handlingFeeA

def purchaseCostB : ℝ := 350
def restockingFeeB : ℝ := 0.03 * purchaseCostB
def totalCostB : ℝ := purchaseCostB + restockingFeeB

def purchaseCostC : ℝ := 400
def transportationFeeC : ℝ := 0.015 * purchaseCostC
def totalCostC : ℝ := purchaseCostC + transportationFeeC

-- Desired profit percentages
def profitPercentageA : ℝ := 0.40
def profitPercentageB : ℝ := 0.25
def profitPercentageC : ℝ := 0.30

-- Net sale prices for achieving the desired profit percentages
def netSalePriceA : ℝ := totalCostA + (profitPercentageA * totalCostA)
def netSalePriceB : ℝ := totalCostB + (profitPercentageB * totalCostB)
def netSalePriceC : ℝ := totalCostC + (profitPercentageC * totalCostC)

-- Expected values
def expectedNetSalePriceA : ℝ := 928.20
def expectedNetSalePriceB : ℝ := 450.63
def expectedNetSalePriceC : ℝ := 527.80

-- Theorem to prove the net sale prices match the expected values
theorem netSalePrice_correct :
  netSalePriceA = expectedNetSalePriceA ∧
  netSalePriceB = expectedNetSalePriceB ∧
  netSalePriceC = expectedNetSalePriceC :=
by
  unfold netSalePriceA netSalePriceB netSalePriceC totalCostA totalCostB totalCostC
         handlingFeeA restockingFeeB transportationFeeC
  sorry

end netSalePrice_correct_l175_175351


namespace range_of_m_l175_175893

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp (2 * x)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1

def exists_x0 (x1 : ℝ) (m : ℝ) : Prop :=
  ∃ (x0 : ℝ), -1 ≤ x0 ∧ x0 ≤ 1 ∧ g m x0 = f x1

theorem range_of_m (m : ℝ) (cond : ∀ (x1 : ℝ), -1 ≤ x1 → x1 ≤ 1 → exists_x0 x1 m) :
  m ∈ Set.Iic (1 - Real.exp 2) ∨ m ∈ Set.Ici (Real.exp 2 - 1) :=
sorry

end range_of_m_l175_175893


namespace propA_propB_relation_l175_175625

variable (x y : ℤ)

theorem propA_propB_relation :
  (x + y ≠ 5 → x ≠ 2 ∨ y ≠ 3) ∧ ¬(x ≠ 2 ∨ y ≠ 3 → x + y ≠ 5) :=
by
  sorry

end propA_propB_relation_l175_175625


namespace Tony_fever_l175_175052

theorem Tony_fever :
  ∀ (normal_temp sickness_increase fever_threshold : ℕ),
    normal_temp = 95 →
    sickness_increase = 10 →
    fever_threshold = 100 →
    (normal_temp + sickness_increase) - fever_threshold = 5 :=
by
  intros normal_temp sickness_increase fever_threshold h1 h2 h3
  sorry

end Tony_fever_l175_175052


namespace simplify_fraction_l175_175773

theorem simplify_fraction (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0) :
  (a^2 + a * b - b^2 + a * c) / (b^2 + b * c - c^2 + b * a) = (a - b) / (b - c) :=
by
  sorry

end simplify_fraction_l175_175773


namespace masha_mushrooms_l175_175092

theorem masha_mushrooms (B1 B2 B3 B4 G1 G2 G3 : ℕ) (total : B1 + B2 + B3 + B4 + G1 + G2 + G3 = 70)
  (girls_distinct : G1 ≠ G2 ∧ G1 ≠ G3 ∧ G2 ≠ G3)
  (boys_threshold : ∀ {A B C D : ℕ}, (A = B1 ∨ A = B2 ∨ A = B3 ∨ A = B4) →
                    (B = B1 ∨ B = B2 ∨ B = B3 ∨ B = B4) →
                    (C = B1 ∨ C = B2 ∨ C = B3 ∨ C = B4) → 
                    (A ≠ B ∧ A ≠ C ∧ B ≠ C) →
                    A + B + C ≥ 43)
  (diff_no_more_than_five_times : ∀ {x y : ℕ}, (x = B1 ∨ x = B2 ∨ x = B3 ∨ x = B4 ∨ x = G1 ∨ x = G2 ∨ x = G3) →
                                  (y = B1 ∨ y = B2 ∨ y = B3 ∨ y = B4 ∨ y = G1 ∨ y = G2 ∨ y = G3) →
                                  x ≠ y → x ≤ 5 * y ∧ y ≤ 5 * x)
  (masha_max_girl : G3 = max G1 (max G2 G3))
  : G3 = 5 :=
sorry

end masha_mushrooms_l175_175092


namespace existence_of_points_on_AC_l175_175135

theorem existence_of_points_on_AC (A B C M : ℝ) (hAB : abs (A - B) = 2) (hBC : abs (B - C) = 1) :
  ((abs (A - M) + abs (B - M) = abs (C - M)) ↔ (M = A - 1) ∨ (M = A + 1)) :=
by
  sorry

end existence_of_points_on_AC_l175_175135


namespace polygon_side_count_eq_six_l175_175692

theorem polygon_side_count_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_side_count_eq_six_l175_175692


namespace expansion_correct_l175_175277

-- Define the polynomials
def poly1 (z : ℤ) : ℤ := 3 * z^2 + 4 * z - 5
def poly2 (z : ℤ) : ℤ := 4 * z^4 - 3 * z^2 + 2

-- Define the expected expanded polynomial
def expanded_poly (z : ℤ) : ℤ := 12 * z^6 + 16 * z^5 - 29 * z^4 - 12 * z^3 + 21 * z^2 + 8 * z - 10

-- The theorem that proves the equivalence of the expanded form
theorem expansion_correct (z : ℤ) : (poly1 z) * (poly2 z) = expanded_poly z := by
  sorry

end expansion_correct_l175_175277


namespace simplified_expression_l175_175632

-- Non-computable context since we are dealing with square roots and division
noncomputable def expr (x : ℝ) : ℝ := ((x / (x - 1)) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem simplified_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : expr x = Real.sqrt 2 / 2 := by
  sorry

end simplified_expression_l175_175632


namespace option_C_equals_a5_l175_175780

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end option_C_equals_a5_l175_175780


namespace rect_side_ratio_square_l175_175670

theorem rect_side_ratio_square (a b d : ℝ) (h1 : b = 2 * a) (h2 : d = a * Real.sqrt 5) : (b / a) ^ 2 = 4 := 
by sorry

end rect_side_ratio_square_l175_175670


namespace students_not_examined_l175_175642

theorem students_not_examined (boys girls examined : ℕ) (h1 : boys = 121) (h2 : girls = 83) (h3 : examined = 150) : 
  (boys + girls - examined = 54) := by
  sorry

end students_not_examined_l175_175642


namespace inequality_proof_l175_175905

theorem inequality_proof (a b : Real) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end inequality_proof_l175_175905


namespace negation_of_universal_proposition_l175_175221

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x ≥ 2 → x^2 ≥ 4)) ↔ (∃ x : ℝ, x ≥ 2 ∧ x^2 < 4) :=
by sorry

end negation_of_universal_proposition_l175_175221


namespace find_starting_number_of_range_l175_175544

theorem find_starting_number_of_range : 
  ∃ (n : ℤ), 
    (∀ k, (0 ≤ k ∧ k < 7) → (n + k * 3 ≤ 31 ∧ n + k * 3 % 3 = 0)) ∧ 
    n + 6 * 3 = 30 - 6 * 3 :=
by
  sorry

end find_starting_number_of_range_l175_175544


namespace James_comics_l175_175962

theorem James_comics (days_in_year : ℕ) (years : ℕ) (writes_every_other_day : ℕ) (no_leap_years : ℕ) 
  (h1 : days_in_year = 365) (h2 : years = 4) (h3 : writes_every_other_day = 2) : 
  (days_in_year * years) / writes_every_other_day = 730 := 
by
  sorry

end James_comics_l175_175962


namespace domain_of_f_l175_175320

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f : {x : ℝ | (x > -1) ∧ (x ≠ 0) ∧ (x ∈ [-3, 3])} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 3} :=
by
  sorry

end domain_of_f_l175_175320


namespace percentage_markup_l175_175031

theorem percentage_markup (selling_price cost_price : ℝ) (h_selling : selling_price = 2000) (h_cost : cost_price = 1250) :
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
  sorry

end percentage_markup_l175_175031


namespace smallest_x_solution_l175_175960

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l175_175960


namespace reeya_average_score_l175_175903

theorem reeya_average_score :
  let scores := [50, 60, 70, 80, 80]
  let sum_scores := scores.sum
  let num_scores := scores.length
  sum_scores / num_scores = 68 :=
by
  sorry

end reeya_average_score_l175_175903


namespace negation_is_correct_l175_175562

-- Define the condition: we have two integers a and b
variables (a b : ℤ)

-- Original proposition: If the sum of two integers is even, then both integers are even.
def original_proposition := (a + b) % 2 = 0 → (a % 2 = 0) ∧ (b % 2 = 0)

-- Negation of the proposition: There exist two integers such that their sum is even and not both are even.
def negation_of_proposition := (a + b) % 2 = 0 ∧ ¬((a % 2 = 0) ∧ (b % 2 = 0))

theorem negation_is_correct :
  ¬ original_proposition a b = negation_of_proposition a b :=
by
  sorry

end negation_is_correct_l175_175562


namespace determinant_inequality_solution_l175_175420

theorem determinant_inequality_solution (a : ℝ) :
  (∀ x : ℝ, (x > -1 → x < (4 / a))) ↔ a = -4 := by
sorry

end determinant_inequality_solution_l175_175420


namespace exponent_power_identity_l175_175916

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end exponent_power_identity_l175_175916


namespace cost_of_1000_pieces_of_gum_l175_175548

theorem cost_of_1000_pieces_of_gum
  (cost_per_piece : ℕ)
  (num_pieces : ℕ)
  (discount_threshold : ℕ)
  (discount_rate : ℚ)
  (conversion_rate : ℕ)
  (h_cost : cost_per_piece = 2)
  (h_pieces : num_pieces = 1000)
  (h_threshold : discount_threshold = 500)
  (h_discount : discount_rate = 0.90)
  (h_conversion : conversion_rate = 100)
  (h_more_than_threshold : num_pieces > discount_threshold) :
  (num_pieces * cost_per_piece * discount_rate) / conversion_rate = 18 := 
sorry

end cost_of_1000_pieces_of_gum_l175_175548


namespace notebook_width_l175_175431

theorem notebook_width
  (circumference : ℕ)
  (length : ℕ)
  (width : ℕ)
  (H1 : circumference = 46)
  (H2 : length = 9)
  (H3 : circumference = 2 * (length + width)) :
  width = 14 :=
by
  sorry -- proof is omitted

end notebook_width_l175_175431


namespace average_probable_weight_l175_175507

-- Definitions based on the conditions
def ArunOpinion (w : ℝ) : Prop := 65 < w ∧ w < 72
def BrotherOpinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def MotherOpinion (w : ℝ) : Prop := w ≤ 68

-- The actual statement we want to prove
theorem average_probable_weight : 
  (∀ (w : ℝ), ArunOpinion w → BrotherOpinion w → MotherOpinion w → 65 < w ∧ w ≤ 68) →
  (65 + 68) / 2 = 66.5 :=
by 
  intros h1
  sorry

end average_probable_weight_l175_175507


namespace tangent_line_equation_l175_175322

/-- Prove that the equation of the tangent line to the curve y = x^3 - 4x^2 + 4 at the point (1,1) is y = -5x + 6 -/
theorem tangent_line_equation (x y : ℝ)
  (h_curve : y = x^3 - 4 * x^2 + 4)
  (h_point : x = 1 ∧ y = 1) :
  y = -5 * x + 6 := by
  sorry

end tangent_line_equation_l175_175322


namespace find_m_l175_175683

noncomputable def geometric_sequence_solution (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) : Prop :=
  (S 3 + S 6 = 2 * S 9) ∧ (a 2 + a 5 = 2 * a m)

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) (h1 : S 3 + S 6 = 2 * S 9)
  (h2 : a 2 + a 5 = 2 * a m) : m = 8 :=
sorry

end find_m_l175_175683


namespace x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l175_175333

theorem x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one {x : ℝ} (h : x + 1 / x = 2) : x^12 = 1 :=
by
  -- The proof will go here, but it is omitted.
  sorry

end x_plus_reciprocal_eq_two_implies_x_pow_12_eq_one_l175_175333


namespace f_2_equals_12_l175_175249

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12_l175_175249


namespace snow_on_second_day_l175_175915

-- Definition of conditions as variables in Lean
def snow_on_first_day := 6 -- in inches
def snow_melted := 2 -- in inches
def additional_snow_fifth_day := 12 -- in inches
def total_snow := 24 -- in inches

-- The variable for snow on the second day
variable (x : ℕ)

-- Proof goal
theorem snow_on_second_day : snow_on_first_day + x - snow_melted + additional_snow_fifth_day = total_snow → x = 8 :=
by
  intros h
  sorry

end snow_on_second_day_l175_175915


namespace card_combinations_l175_175980

noncomputable def valid_card_combinations : List (ℕ × ℕ × ℕ × ℕ) :=
  [(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)]

theorem card_combinations (a b c d : ℕ) (h : a ≤ b ∧ b ≤ c ∧ c ≤ d) :
  (1, 2, 7, 8) ∈ valid_card_combinations ∨ 
  (1, 3, 6, 8) ∈ valid_card_combinations ∨ 
  (1, 4, 5, 8) ∈ valid_card_combinations ∨ 
  (2, 3, 6, 7) ∈ valid_card_combinations ∨ 
  (2, 4, 5, 7) ∈ valid_card_combinations ∨ 
  (3, 4, 5, 6) ∈ valid_card_combinations :=
sorry

end card_combinations_l175_175980


namespace complement_intersection_l175_175882

open Set

noncomputable def U : Set ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def M : Set ℕ := {1, 4}
noncomputable def N : Set ℕ := {2, 3}

theorem complement_intersection :
  ((U \ M) ∩ N) = {2, 3} :=
by
  sorry

end complement_intersection_l175_175882


namespace valid_choice_count_l175_175912

def is_valid_base_7_digit (n : ℕ) : Prop := n < 7
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8
def to_base_10_base_7 (c3 c2 c1 c0 : ℕ) : ℕ := 2401 * c3 + 343 * c2 + 49 * c1 + 7 * c0
def to_base_10_base_8 (d3 d2 d1 d0 : ℕ) : ℕ := 4096 * d3 + 512 * d2 + 64 * d1 + 8 * d0
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem valid_choice_count :
  ∃ (N : ℕ), is_four_digit_number N →
  ∀ (c3 c2 c1 c0 d3 d2 d1 d0 : ℕ),
    is_valid_base_7_digit c3 → is_valid_base_7_digit c2 → is_valid_base_7_digit c1 → is_valid_base_7_digit c0 →
    is_valid_base_8_digit d3 → is_valid_base_8_digit d2 → is_valid_base_8_digit d1 → is_valid_base_8_digit d0 →
    to_base_10_base_7 c3 c2 c1 c0 = N →
    to_base_10_base_8 d3 d2 d1 d0 = N →
    (to_base_10_base_7 c3 c2 c1 c0 + to_base_10_base_8 d3 d2 d1 d0) % 1000 = (2 * N) % 1000 → N = 20 :=
sorry

end valid_choice_count_l175_175912


namespace candle_burning_time_l175_175815

theorem candle_burning_time :
  ∃ t : ℚ, (1 - t / 5) = 3 * (1 - t / 4) ∧ t = 40 / 11 :=
by {
  sorry
}

end candle_burning_time_l175_175815


namespace probability_jqka_is_correct_l175_175511

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end probability_jqka_is_correct_l175_175511


namespace find_other_number_l175_175845

theorem find_other_number (hcf lcm a b: ℕ) (hcf_value: hcf = 12) (lcm_value: lcm = 396) (a_value: a = 36) (gcd_ab: Nat.gcd a b = hcf) (lcm_ab: Nat.lcm a b = lcm) : b = 132 :=
by
  sorry

end find_other_number_l175_175845


namespace danielle_rooms_is_6_l175_175665

def heidi_rooms (danielle_rooms : ℕ) : ℕ := 3 * danielle_rooms
def grant_rooms (heidi_rooms : ℕ) : ℕ := heidi_rooms / 9

theorem danielle_rooms_is_6 (danielle_rooms : ℕ) (h1 : heidi_rooms danielle_rooms = 18) (h2 : grant_rooms (heidi_rooms danielle_rooms) = 2) :
  danielle_rooms = 6 :=
by 
  sorry

end danielle_rooms_is_6_l175_175665


namespace volume_of_cylinder_cut_l175_175618

open Real

noncomputable def cylinder_cut_volume (R α : ℝ) : ℝ :=
  (2 / 3) * R^3 * tan α

theorem volume_of_cylinder_cut (R α : ℝ) :
  cylinder_cut_volume R α = (2 / 3) * R^3 * tan α :=
by
  sorry

end volume_of_cylinder_cut_l175_175618


namespace find_a_minus_b_l175_175974

theorem find_a_minus_b (a b : ℚ) (h_eq : ∀ x : ℚ, (a * (-5 * x + 3) + b) = x - 9) : 
  a - b = 41 / 5 := 
by {
  sorry
}

end find_a_minus_b_l175_175974


namespace arithmetic_contains_geometric_l175_175359

theorem arithmetic_contains_geometric (a d : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) : 
  ∃ b q : ℕ, (b = a) ∧ (q = 1 + d) ∧ (∀ n : ℕ, ∃ k : ℕ, a * (1 + d)^n = a + k * d) :=
by
  sorry

end arithmetic_contains_geometric_l175_175359


namespace find_common_difference_l175_175514

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def first_term (a_n : ℕ → ℕ) := a_n 1 = 1
def common_difference (d : ℕ) := d ≠ 0
def arithmetic_def (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n (n+1) = a_n n + d
def geom_mean_condition (a_n : ℕ → ℕ) := a_n 2 ^ 2 = a_n 1 * a_n 4

-- Proof statement
theorem find_common_difference
  (fa : first_term a_n)
  (cd : common_difference d)
  (ad : arithmetic_def a_n d)
  (gmc : geom_mean_condition a_n) :
  d = 1 := by
  sorry

end find_common_difference_l175_175514


namespace find_a_from_circle_and_chord_l175_175551

theorem find_a_from_circle_and_chord 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0)
  (line_eq : ∀ x y : ℝ, x + y + 2 = 0)
  (chord_length : ∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 + 2*x1 - 2*y1 + a = 0 ∧ x2^2 + y2^2 + 2*x2 - 2*y2 + a = 0 ∧ x1 + y1 + 2 = 0 ∧ x2 + y2 + 2 = 0 → (x1 - x2)^2 + (y1 - y2)^2 = 16) :
  a = -4 :=
by
  sorry

end find_a_from_circle_and_chord_l175_175551


namespace binary_to_decimal_110101_l175_175559

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 53) :=
by
  sorry

end binary_to_decimal_110101_l175_175559


namespace remainder_7_pow_63_mod_8_l175_175382

theorem remainder_7_pow_63_mod_8 : 7^63 % 8 = 7 :=
by sorry

end remainder_7_pow_63_mod_8_l175_175382


namespace arithmetic_sequence_thm_l175_175303

theorem arithmetic_sequence_thm
  (a : ℕ → ℝ)
  (h1 : a 1 + a 4 + a 7 = 48)
  (h2 : a 2 + a 5 + a 8 = 40)
  (d : ℝ)
  (h3 : ∀ n, a (n + 1) = a n + d) :
  a 3 + a 6 + a 9 = 32 :=
by {
  sorry
}

end arithmetic_sequence_thm_l175_175303


namespace minimum_value_of_a_l175_175535

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5 / 2 :=
sorry

end minimum_value_of_a_l175_175535


namespace find_number_mul_l175_175258

theorem find_number_mul (n : ℕ) (h : n * 9999 = 724777430) : n = 72483 :=
by
  sorry

end find_number_mul_l175_175258


namespace max_teams_4_weeks_l175_175149

noncomputable def max_teams_in_tournament (weeks number_teams : ℕ) : ℕ :=
  if h : weeks > 0 then (number_teams * (number_teams - 1)) / (2 * weeks) else 0

theorem max_teams_4_weeks : max_teams_in_tournament 4 7 = 6 := by
  -- Assumptions
  let n := 6
  let teams := 7 * n
  let weeks := 4
  
  -- Define the constraints and checks here
  sorry

end max_teams_4_weeks_l175_175149


namespace non_divisible_l175_175224

theorem non_divisible (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ¬ ∃ k : ℤ, x^2 + y^2 + z^2 = k * 3 * (x * y + y * z + z * x) :=
by sorry

end non_divisible_l175_175224


namespace algebraic_expression_value_l175_175888

-- Given conditions as definitions and assumption
variables (a b : ℝ)
def expression1 (x : ℝ) := 2 * a * x^3 - 3 * b * x + 8
def expression2 := 9 * b - 6 * a + 2

theorem algebraic_expression_value
  (h1 : expression1 (-1) = 18) :
  expression2 = 32 :=
by
  sorry

end algebraic_expression_value_l175_175888


namespace ladder_distance_from_wall_l175_175301

theorem ladder_distance_from_wall (θ : ℝ) (L : ℝ) (d : ℝ) 
  (h_angle : θ = 60) (h_length : L = 19) (h_cos : Real.cos (θ * Real.pi / 180) = 0.5) : 
  d = 9.5 :=
by
  sorry

end ladder_distance_from_wall_l175_175301


namespace subset_exists_l175_175948

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {x + 2, 1}

-- Statement of the theorem
theorem subset_exists (x : ℝ) : B 2 ⊆ A 2 :=
by
  sorry

end subset_exists_l175_175948


namespace wendy_score_l175_175134

def score_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def treasures_second_level : ℕ := 3

theorem wendy_score :
  score_per_treasure * treasures_first_level + score_per_treasure * treasures_second_level = 35 :=
by
  sorry

end wendy_score_l175_175134


namespace smallest_number_of_marbles_l175_175469

theorem smallest_number_of_marbles :
  ∃ N : ℕ, N > 1 ∧ (N % 9 = 1) ∧ (N % 10 = 1) ∧ (N % 11 = 1) ∧ (∀ m : ℕ, m > 1 ∧ (m % 9 = 1) ∧ (m % 10 = 1) ∧ (m % 11 = 1) → N ≤ m) :=
sorry

end smallest_number_of_marbles_l175_175469


namespace mangoes_total_l175_175569

theorem mangoes_total (M A : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : A = 60) :
  A + M = 75 :=
by
  sorry

end mangoes_total_l175_175569


namespace solve_system_of_equations_l175_175479

theorem solve_system_of_equations (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : 1 / (x * y) = x / z + 1)
  (h2 : 1 / (y * z) = y / x + 1)
  (h3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by
  sorry

end solve_system_of_equations_l175_175479


namespace intersection_lines_l175_175245

theorem intersection_lines (c d : ℝ) (h1 : 6 = 2 * 4 + c) (h2 : 6 = 5 * 4 + d) : c + d = -16 := 
by
  sorry

end intersection_lines_l175_175245


namespace compound_interest_rate_l175_175898

theorem compound_interest_rate :
  ∀ (A P : ℝ) (t : ℕ),
  A = 4840.000000000001 ->
  P = 4000 ->
  t = 2 ->
  A = P * (1 + 0.1)^t :=
by
  intros A P t hA hP ht
  rw [hA, hP, ht]
  norm_num
  sorry

end compound_interest_rate_l175_175898


namespace density_ratio_of_large_cube_l175_175121

theorem density_ratio_of_large_cube 
  (V0 m0 : ℝ) (initial_density replacement_density: ℝ)
  (initial_mass final_mass : ℝ) (V_total : ℝ) 
  (h1 : initial_density = m0 / V0)
  (h2 : replacement_density = 2 * initial_density)
  (h3 : initial_mass = 8 * m0)
  (h4 : final_mass = 6 * m0 + 2 * (2 * m0))
  (h5 : V_total = 8 * V0) :
  initial_density / (final_mass / V_total) = 0.8 :=
sorry

end density_ratio_of_large_cube_l175_175121


namespace vertex_h_is_3_l175_175222

open Real

theorem vertex_h_is_3 (a b c : ℝ) (h : ℝ)
    (h_cond : 3 * (a * 3^2 + b * 3 + c) + 6 = 3) : 
    4 * (a * x^2 + b * x + c) = 12 * (x - 3)^2 + 24 → 
    h = 3 := 
by 
sorry

end vertex_h_is_3_l175_175222


namespace min_dSigma_correct_l175_175769

noncomputable def min_dSigma {a r : ℝ} (h : a > r) : ℝ :=
  (a - r) / 2

theorem min_dSigma_correct (a r : ℝ) (h : a > r) :
  min_dSigma h = (a - r) / 2 :=
by 
  unfold min_dSigma
  sorry

end min_dSigma_correct_l175_175769


namespace larger_number_is_28_l175_175026

theorem larger_number_is_28
  (x y : ℕ)
  (h1 : 4 * y = 7 * x)
  (h2 : y - x = 12) : y = 28 :=
sorry

end larger_number_is_28_l175_175026


namespace club_members_remainder_l175_175556

theorem club_members_remainder (N : ℕ) (h1 : 50 < N) (h2 : N < 80)
  (h3 : N % 5 = 0) (h4 : N % 8 = 0 ∨ N % 7 = 0) :
  N % 9 = 6 ∨ N % 9 = 7 := by
  sorry

end club_members_remainder_l175_175556


namespace quadratic_inequality_solution_l175_175069

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : ∀ x, -3 < x ∧ x < 1/2 ↔ cx^2 + bx + a < 0) :
  ∀ x, -1/3 ≤ x ∧ x ≤ 2 ↔ ax^2 + bx + c ≥ 0 :=
sorry

end quadratic_inequality_solution_l175_175069


namespace parabola_focus_coordinates_l175_175436

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ (y₀ : ℝ), (x, y₀) = (1/16, 0)) :=
by
  intro x y hxy
  sorry

end parabola_focus_coordinates_l175_175436


namespace function_monotonicity_l175_175259

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x) / (b^x + c^x) + (b^x) / (a^x + c^x) + (c^x) / (a^x + b^x)

theorem function_monotonicity (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, y ≤ x → x < 0 → f a b c x ≤ f a b c y) :=
by
  sorry

end function_monotonicity_l175_175259


namespace joe_total_paint_used_l175_175016

-- Define the initial amount of paint Joe buys.
def initial_paint : ℕ := 360

-- Define the fraction of paint used during the first week.
def first_week_fraction := 1 / 4

-- Define the fraction of remaining paint used during the second week.
def second_week_fraction := 1 / 2

-- Define the total paint used by Joe in the first week.
def paint_used_first_week := first_week_fraction * initial_paint

-- Define the remaining paint after the first week.
def remaining_paint_after_first_week := initial_paint - paint_used_first_week

-- Define the total paint used by Joe in the second week.
def paint_used_second_week := second_week_fraction * remaining_paint_after_first_week

-- Define the total paint used by Joe.
def total_paint_used := paint_used_first_week + paint_used_second_week

-- The theorem to be proven: the total amount of paint Joe has used is 225 gallons.
theorem joe_total_paint_used : total_paint_used = 225 := by
  sorry

end joe_total_paint_used_l175_175016


namespace place_pawns_distinct_5x5_l175_175041

noncomputable def number_of_ways_place_pawns : ℕ :=
  5 * 4 * 3 * 2 * 1 * 120

theorem place_pawns_distinct_5x5 : number_of_ways_place_pawns = 14400 := by
  sorry

end place_pawns_distinct_5x5_l175_175041


namespace correct_calculation_D_l175_175442

theorem correct_calculation_D (m : ℕ) : 
  (2 * m ^ 3) * (3 * m ^ 2) = 6 * m ^ 5 :=
by
  sorry

end correct_calculation_D_l175_175442


namespace order_of_mnpq_l175_175818

theorem order_of_mnpq 
(m n p q : ℝ) 
(h1 : m < n)
(h2 : p < q)
(h3 : (p - m) * (p - n) < 0)
(h4 : (q - m) * (q - n) < 0) 
: m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_mnpq_l175_175818


namespace factor_expression_l175_175141

variable (a : ℝ)

theorem factor_expression : 37 * a^2 + 111 * a = 37 * a * (a + 3) :=
  sorry

end factor_expression_l175_175141


namespace matrix_vector_combination_l175_175933

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (M : α →ₗ[ℝ] ℝ × ℝ)
variables (u v w : α)
variables (h1 : M u = (-3, 4))
variables (h2 : M v = (2, -7))
variables (h3 : M w = (9, 0))

theorem matrix_vector_combination :
  M (3 • u - 4 • v + 2 • w) = (1, 40) :=
by sorry

end matrix_vector_combination_l175_175933


namespace distance_equals_absolute_value_l175_175639

def distance_from_origin (x : ℝ) : ℝ := abs x

theorem distance_equals_absolute_value (x : ℝ) : distance_from_origin x = abs x :=
by
  sorry

end distance_equals_absolute_value_l175_175639


namespace inequality_proof_l175_175770

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_mul : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 :=
sorry

end inequality_proof_l175_175770


namespace structure_of_S_l175_175965

def set_S (x y : ℝ) : Prop :=
  (5 >= x + 1 ∧ 5 >= y - 5) ∨
  (x + 1 >= 5 ∧ x + 1 >= y - 5) ∨
  (y - 5 >= 5 ∧ y - 5 >= x + 1)

theorem structure_of_S :
  ∃ (a b c : ℝ), set_S x y ↔ (y <= x + 6) ∧ (x <= 4) ∧ (y <= 10) 
:= sorry

end structure_of_S_l175_175965


namespace find_ratio_l175_175376

-- Definition of the function
def f (x : ℝ) (a b: ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

-- Statement to be proved
theorem find_ratio (a b : ℝ) (h1: f 1 a b = 10) (h2 : (3 * 1^2 + 2 * a * 1 + b = 0)) : b = -a / 2 :=
by
  sorry

end find_ratio_l175_175376


namespace time_to_cross_signal_pole_l175_175707

/-- Definitions representing the given conditions --/
def length_of_train : ℕ := 300
def time_to_cross_platform : ℕ := 39
def length_of_platform : ℕ := 350
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform

/-- Main statement to be proven --/
theorem time_to_cross_signal_pole : length_of_train / speed_of_train = 18 := by
  sorry

end time_to_cross_signal_pole_l175_175707


namespace final_answer_l175_175708

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end final_answer_l175_175708


namespace sum_of_smallest_and_largest_l175_175001

theorem sum_of_smallest_and_largest (n : ℕ) (h : Odd n) (b z : ℤ)
  (h_mean : z = b + n - 1 - 2 / (n : ℤ)) :
  ((b - 2) + (b + 2 * (n - 2))) = 2 * z - 4 + 4 / (n : ℤ) :=
by
  sorry

end sum_of_smallest_and_largest_l175_175001


namespace tangent_line_equation_l175_175745

theorem tangent_line_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ∃ (m b : ℝ), y = m * x + b ∧ y = 4 * x - 2 :=
by
  sorry

end tangent_line_equation_l175_175745


namespace eggs_leftover_l175_175184

theorem eggs_leftover :
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  total_eggs % 10 = 0 := by
  let abigail_eggs := 58
  let beatrice_eggs := 35
  let carson_eggs := 27
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  exact Nat.mod_eq_zero_of_dvd (show 10 ∣ total_eggs from by norm_num)

end eggs_leftover_l175_175184


namespace valid_two_digit_numbers_l175_175678

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end valid_two_digit_numbers_l175_175678


namespace quadratic_root_zero_l175_175987

theorem quadratic_root_zero (a : ℝ) : 
  ((a-1) * 0^2 + 0 + a^2 - 1 = 0) 
  → a ≠ 1 
  → a = -1 := 
by
  intro h1 h2
  sorry

end quadratic_root_zero_l175_175987


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l175_175645

-- Define the solutions to the given quadratic equations

theorem solve_eq1 (x : ℝ) : 2 * x ^ 2 - 8 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : x ^ 2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1 :=
by sorry

theorem solve_eq3 (x : ℝ) : 5 * x ^ 2 - 4 * x - 1 = 0 ↔ x = -1 / 5 ∨ x = 1 :=
by sorry

theorem solve_eq4 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l175_175645


namespace money_problem_l175_175446

variable (a b : ℝ)

theorem money_problem (h1 : 4 * a + b = 68) 
                      (h2 : 2 * a - b < 16) 
                      (h3 : a + b > 22) : 
                      a < 14 ∧ b > 12 := 
by 
  sorry

end money_problem_l175_175446


namespace smallest_z_value_l175_175285

-- Definitions: w, x, y, and z as consecutive even positive integers
def consecutive_even_cubes (w x y z : ℤ) : Prop :=
  w % 2 = 0 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  w < x ∧ x < y ∧ y < z ∧
  x = w + 2 ∧ y = x + 2 ∧ z = y + 2

-- Problem statement: Smallest possible value of z
theorem smallest_z_value :
  ∃ w x y z : ℤ, consecutive_even_cubes w x y z ∧ w^3 + x^3 + y^3 = z^3 ∧ z = 12 :=
by
  sorry

end smallest_z_value_l175_175285


namespace distance_walked_north_l175_175595

-- Definition of the problem parameters
def distance_west : ℝ := 10
def total_distance : ℝ := 14.142135623730951

-- The theorem stating the result
theorem distance_walked_north (x : ℝ) (h : distance_west ^ 2 + x ^ 2 = total_distance ^ 2) : x = 10 :=
by sorry

end distance_walked_north_l175_175595


namespace arrange_pencils_l175_175837

-- Definition to express the concept of pencil touching
def pencil_touches (a b : Type) : Prop := sorry

-- Assume we have six pencils represented as 6 distinct variables.
variables (A B C D E F : Type)

-- Main theorem statement
theorem arrange_pencils :
  ∃ (A B C D E F : Type), (pencil_touches A B) ∧ (pencil_touches A C) ∧ 
  (pencil_touches A D) ∧ (pencil_touches A E) ∧ (pencil_touches A F) ∧ 
  (pencil_touches B C) ∧ (pencil_touches B D) ∧ (pencil_touches B E) ∧ 
  (pencil_touches B F) ∧ (pencil_touches C D) ∧ (pencil_touches C E) ∧ 
  (pencil_touches C F) ∧ (pencil_touches D E) ∧ (pencil_touches D F) ∧ 
  (pencil_touches E F) :=
sorry

end arrange_pencils_l175_175837


namespace election_winner_votes_l175_175005

variable (V : ℝ) (winner_votes : ℝ) (winner_margin : ℝ)
variable (condition1 : V > 0)
variable (condition2 : winner_votes = 0.60 * V)
variable (condition3 : winner_margin = 240)

theorem election_winner_votes (h : winner_votes - 0.40 * V = winner_margin) : winner_votes = 720 := by
  sorry

end election_winner_votes_l175_175005


namespace total_salmon_count_l175_175099

def chinook_males := 451228
def chinook_females := 164225
def sockeye_males := 212001
def sockeye_females := 76914
def coho_males := 301008
def coho_females := 111873
def pink_males := 518001
def pink_females := 182945
def chum_males := 230023
def chum_females := 81321

theorem total_salmon_count : 
  chinook_males + chinook_females + 
  sockeye_males + sockeye_females + 
  coho_males + coho_females + 
  pink_males + pink_females + 
  chum_males + chum_females = 2329539 := 
by
  sorry

end total_salmon_count_l175_175099


namespace real_root_of_P_l175_175192

noncomputable def P : ℕ → ℝ → ℝ
| 0, x => 0
| 1, x => x
| n+2, x => x * P (n + 1) x + (1 - x) * P n x

theorem real_root_of_P (n : ℕ) (hn : 1 ≤ n) : ∀ x : ℝ, P n x = 0 → x = 0 := 
by 
  sorry

end real_root_of_P_l175_175192


namespace intersection_conditions_l175_175427

-- Define the conditions
variables (c : ℝ) (k : ℝ) (m : ℝ) (n : ℝ) (p : ℝ)

-- Distance condition
def distance_condition (k : ℝ) (m : ℝ) (n : ℝ) (c : ℝ) : Prop :=
  (abs ((k^2 + 8 * k + c) - (m * k + n)) = 4)

-- Line passing through point (2, 7)
def passes_through_point (m : ℝ) (n : ℝ) : Prop :=
  (7 = 2 * m + n)

-- Definition of discriminants
def discriminant_1 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n - 4))

def discriminant_2 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n + 4))

-- Statement of the problem
theorem intersection_conditions (h₁ : n ≠ 0)
  (h₂ : passes_through_point m n)
  (h₃ : distance_condition k m n c)
  (h₄ : (discriminant_1 m c n = 0 ∨ discriminant_1 m c n < 0))
  (h₅ : (discriminant_2 m c n < 0)) :
  ∃ m n, n = 7 - 2 * m ∧ distance_condition k m n c :=
sorry

end intersection_conditions_l175_175427


namespace acute_angle_at_315_equals_7_5_l175_175020

/-- The degrees in a full circle -/
def fullCircle := 360

/-- The number of hours on a clock -/
def hoursOnClock := 12

/-- The measure in degrees of the acute angle formed by the minute hand and the hour hand at 3:15 -/
def acuteAngleAt315 : ℝ :=
  let degreesPerHour := fullCircle / hoursOnClock
  let hourHandAt3 := degreesPerHour * 3
  let additionalDegrees := (15 / 60) * degreesPerHour
  let hourHandPosition := hourHandAt3 + additionalDegrees
  let minuteHandPosition := (15 / 60) * fullCircle
  abs (hourHandPosition - minuteHandPosition)

theorem acute_angle_at_315_equals_7_5 : acuteAngleAt315 = 7.5 := by
  sorry

end acute_angle_at_315_equals_7_5_l175_175020


namespace pairs_of_different_positives_l175_175636

def W (x : ℕ) : ℕ := x^4 - 3 * x^3 + 5 * x^2 - 9 * x

theorem pairs_of_different_positives (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (hW : W a = W b) : (a, b) = (1, 2) ∨ (a, b) = (2, 1) := 
sorry

end pairs_of_different_positives_l175_175636


namespace necklace_stand_capacity_l175_175455

def necklace_stand_initial := 5
def ring_display_capacity := 30
def ring_display_current := 18
def bracelet_display_capacity := 15
def bracelet_display_current := 8
def cost_per_necklace := 4
def cost_per_ring := 10
def cost_per_bracelet := 5
def total_cost := 183

theorem necklace_stand_capacity : necklace_stand_current + (total_cost - (ring_display_capacity - ring_display_current) * cost_per_ring - (bracelet_display_capacity - bracelet_display_current) * cost_per_bracelet) / cost_per_necklace = 12 :=
by
  sorry

end necklace_stand_capacity_l175_175455


namespace only_triple_l175_175292

theorem only_triple (a b c : ℕ) (h1 : (a * b + 1) % c = 0)
                                (h2 : (a * c + 1) % b = 0)
                                (h3 : (b * c + 1) % a = 0) :
    (a = 1 ∧ b = 1 ∧ c = 1) :=
by
  sorry

end only_triple_l175_175292


namespace max_value_inequality_l175_175522

theorem max_value_inequality (a x₁ x₂ : ℝ) (h_a : a < 0)
  (h_sol : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x₁ < x ∧ x < x₂) :
    x₁ + x₂ + a / (x₁ * x₂) ≤ - 4 * Real.sqrt 3 / 3 := by
  sorry

end max_value_inequality_l175_175522


namespace geometric_sequence_a4_l175_175978

-- Define the geometric sequence and known conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

variables (a : ℕ → ℝ) (q : ℝ)

-- Given conditions:
def a2_eq_4 : Prop := a 2 = 4
def a6_eq_16 : Prop := a 6 = 16

-- The goal is to show a 4 = 8 given the conditions
theorem geometric_sequence_a4 (h_seq : geometric_sequence a q)
  (h_a2 : a2_eq_4 a)
  (h_a6 : a6_eq_16 a) : a 4 = 8 := by
  sorry

end geometric_sequence_a4_l175_175978


namespace ratio_of_capitals_l175_175080

-- Variables for the capitals of Ashok and Pyarelal
variables (A P : ℕ)

-- Given conditions
def total_loss := 670
def pyarelal_loss := 603
def ashok_loss := total_loss - pyarelal_loss

-- Proof statement: the ratio of Ashok's capital to Pyarelal's capital
theorem ratio_of_capitals : ashok_loss * P = total_loss * pyarelal_loss - pyarelal_loss * P → A * pyarelal_loss = P * ashok_loss :=
by
  sorry

end ratio_of_capitals_l175_175080


namespace find_symmetric_point_l175_175539

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane (x y z : ℝ) : ℝ := 
  4 * x + 6 * y + 4 * z - 25

def symmetric_point (M M_prime : Point3D) (plane_eq : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let t : ℝ := (1 / 4)
  let M0 : Point3D := { x := (1 + 4 * t), y := (6 * t), z := (1 + 4 * t) }
  let midpoint_x := (M.x + M_prime.x) / 2
  let midpoint_y := (M.y + M_prime.y) / 2
  let midpoint_z := (M.z + M_prime.z) / 2
  M0.x = midpoint_x ∧ M0.y = midpoint_y ∧ M0.z = midpoint_z ∧
  plane_eq M0.x M0.y M0.z = 0

def M : Point3D := { x := 1, y := 0, z := 1 }

def M_prime : Point3D := { x := 3, y := 3, z := 3 }

theorem find_symmetric_point : symmetric_point M M_prime plane := by
  -- the proof is omitted here
  sorry

end find_symmetric_point_l175_175539


namespace max_students_before_new_year_l175_175757

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l175_175757


namespace miley_total_cost_l175_175732

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost_l175_175732


namespace minimum_oranges_to_profit_l175_175669

/-- 
A boy buys 4 oranges for 12 cents and sells 6 oranges for 25 cents. 
Calculate the minimum number of oranges he needs to sell to make a profit of 150 cents.
--/
theorem minimum_oranges_to_profit (cost_oranges : ℕ) (cost_cents : ℕ)
  (sell_oranges : ℕ) (sell_cents : ℕ) (desired_profit : ℚ) :
  cost_oranges = 4 → cost_cents = 12 →
  sell_oranges = 6 → sell_cents = 25 →
  desired_profit = 150 →
  (∃ n : ℕ, n = 129) :=
by
  sorry

end minimum_oranges_to_profit_l175_175669


namespace abs_diff_less_abs_one_minus_prod_l175_175177

theorem abs_diff_less_abs_one_minus_prod (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end abs_diff_less_abs_one_minus_prod_l175_175177


namespace factor_poly_eq_factored_form_l175_175366

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end factor_poly_eq_factored_form_l175_175366


namespace pascal_triangle_probability_l175_175564

-- Define the probability problem in Lean 4
theorem pascal_triangle_probability :
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  (ones_count + twos_count) / total_elements = 5 / 14 :=
by
  let total_elements := ((20 * (20 + 1)) / 2)
  let ones_count := (1 + 2 * 19)
  let twos_count := (2 * (19 - 2 + 1))
  have h1 : total_elements = 210 := by sorry
  have h2 : ones_count = 39 := by sorry
  have h3 : twos_count = 36 := by sorry
  have h4 : (39 + 36) / 210 = 5 / 14 := by sorry
  exact h4

end pascal_triangle_probability_l175_175564


namespace gcd_poly_l175_175215

theorem gcd_poly (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) : 
  Int.gcd (4 * b ^ 2 + 63 * b + 144) (2 * b + 7) = 1 := 
by 
  sorry

end gcd_poly_l175_175215


namespace m_range_l175_175515

open Real

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem: m must belong to the interval [-4, 5]
theorem m_range (m : ℝ) : (line_eq A.1 A.2 m) → (line_eq B.1 B.2 m) → -4 ≤ m ∧ m ≤ 5 := 
sorry

end m_range_l175_175515


namespace ratio_of_logs_l175_175345

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : Real.log a / Real.log 4 = Real.log b / Real.log 18 ∧ Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_logs_l175_175345


namespace courses_choice_l175_175014

theorem courses_choice (total_courses : ℕ) (chosen_courses : ℕ)
  (h_total_courses : total_courses = 5)
  (h_chosen_courses : chosen_courses = 2) :
  ∃ (ways : ℕ), ways = 60 ∧
    (ways = ((Nat.choose total_courses chosen_courses)^2) - 
            (Nat.choose total_courses chosen_courses) - 
            ((Nat.choose total_courses chosen_courses) * 
             (Nat.choose (total_courses - chosen_courses) chosen_courses))) :=
by
  sorry

end courses_choice_l175_175014


namespace total_money_l175_175294

theorem total_money (John Alice Bob : ℝ) (hJohn : John = 5 / 8) (hAlice : Alice = 7 / 20) (hBob : Bob = 1 / 4) :
  John + Alice + Bob = 1.225 := 
by 
  sorry

end total_money_l175_175294


namespace find_d_l175_175699

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x + 1

theorem find_d (c d : ℝ) (hx : ∀ x, f (g x c) c = 15 * x + d) : d = 8 :=
sorry

end find_d_l175_175699


namespace profit_percentage_l175_175007

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 500) (hSP : SP = 725) : 
  100 * (SP - CP) / CP = 45 :=
by
  sorry

end profit_percentage_l175_175007


namespace find_p_q_l175_175836

noncomputable def cubicFunction (p q : ℝ) (x : ℂ) : ℂ :=
  2 * x^3 + p * x^2 + q * x

theorem find_p_q (p q : ℝ) :
  cubicFunction p q (2 * Complex.I - 3) = 0 ∧ 
  cubicFunction p q (-2 * Complex.I - 3) = 0 → 
  p = 12 ∧ q = 26 :=
by
  sorry

end find_p_q_l175_175836


namespace new_numbers_are_reciprocals_l175_175570

variable {x y : ℝ}

theorem new_numbers_are_reciprocals (h : (1 / x) + (1 / y) = 1) : 
  (x - 1 = 1 / (y - 1)) ∧ (y - 1 = 1 / (x - 1)) := 
by
  sorry

end new_numbers_are_reciprocals_l175_175570


namespace power_function_zeros_l175_175310

theorem power_function_zeros :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = x ^ 3) ∧ (f 2 = 8) ∧ (∀ y : ℝ, (f y - y = 0) ↔ (y = 0 ∨ y = 1 ∨ y = -1)) := by
  sorry

end power_function_zeros_l175_175310


namespace lukas_points_in_5_games_l175_175644

theorem lukas_points_in_5_games (avg_points_per_game : ℕ) (games_played : ℕ) (total_points : ℕ)
  (h_avg : avg_points_per_game = 12) (h_games : games_played = 5) : total_points = 60 :=
by
  sorry

end lukas_points_in_5_games_l175_175644


namespace geometric_sequence_iff_q_neg_one_l175_175801

theorem geometric_sequence_iff_q_neg_one {p q : ℝ} (h1 : p ≠ 0) (h2 : p ≠ 1)
  (S : ℕ → ℝ) (hS : ∀ n, S n = p^n + q) :
  (∃ (a : ℕ → ℝ), (∀ n, a (n+1) = (p - 1) * p^n) ∧ (∀ n, a (n+1) = S (n+1) - S n) ∧
                    (∀ n, a (n+1) / a n = p)) ↔ q = -1 :=
sorry

end geometric_sequence_iff_q_neg_one_l175_175801


namespace estimate_fish_population_l175_175756

theorem estimate_fish_population :
  ∀ (x : ℕ), (1200 / x = 100 / 1000) → x = 12000 := by
  sorry

end estimate_fish_population_l175_175756


namespace sum_of_integers_l175_175800

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 6) (h2 : x * y = 112) (h3 : x > y) : x + y = 22 :=
sorry

end sum_of_integers_l175_175800


namespace chris_initial_donuts_l175_175386

theorem chris_initial_donuts (D : ℝ) (H1 : D * 0.90 - 4 = 23) : D = 30 := 
by
sorry

end chris_initial_donuts_l175_175386


namespace oak_trees_cut_down_l175_175741

   def number_of_cuts (initial: ℕ) (remaining: ℕ) : ℕ :=
     initial - remaining

   theorem oak_trees_cut_down : number_of_cuts 9 7 = 2 :=
   by
     -- Based on the conditions, we start with 9 and after workers finished, there are 7 oak trees.
     -- We calculate the number of trees cut down:
     -- 9 - 7 = 2
     sorry
   
end oak_trees_cut_down_l175_175741


namespace church_path_count_is_321_l175_175494

/-- A person starts at the bottom-left corner of an m x n grid and can only move north, east, or 
    northeast. Prove that the number of distinct paths to the top-right corner is 321 
    for a specific grid size (abstracted parameters included). -/
def distinct_paths_to_church (m n : ℕ) : ℕ :=
  let rec P : ℕ → ℕ → ℕ
    | 0, 0 => 1
    | i + 1, 0 => 1
    | 0, j + 1 => 1
    | i + 1, j + 1 => P i (j + 1) + P (i + 1) j + P i j
  P m n

theorem church_path_count_is_321 : distinct_paths_to_church m n = 321 :=
sorry

end church_path_count_is_321_l175_175494


namespace total_trees_after_planting_l175_175783

def current_trees : ℕ := 7
def trees_planted_today : ℕ := 5
def trees_planted_tomorrow : ℕ := 4

theorem total_trees_after_planting : 
  current_trees + trees_planted_today + trees_planted_tomorrow = 16 :=
by
  sorry

end total_trees_after_planting_l175_175783


namespace find_k_l175_175267

theorem find_k (k : ℝ) : 
  (k - 10) / (-8) = (5 - k) / (-8) → k = 7.5 :=
by
  intro h
  let slope1 := (k - 10) / (-8)
  let slope2 := (5 - k) / (-8)
  have h_eq : slope1 = slope2 := h
  sorry

end find_k_l175_175267


namespace problem1_problem2_real_problem2_complex_problem3_l175_175865

-- Problem 1: Prove that if 2 ∈ A, then {-1, 1/2} ⊆ A
theorem problem1 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : 2 ∈ A) : -1 ∈ A ∧ (1/2) ∈ A := sorry

-- Problem 2: Prove that A cannot be a singleton set for real numbers, but can for complex numbers.
theorem problem2_real (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : ¬(∃ a, A = {a}) := sorry

theorem problem2_complex (A : Set ℂ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : (∃ a, A = {a}) := sorry

-- Problem 3: Prove that 1 - 1/a ∈ A given a ∈ A
theorem problem3 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (a : ℝ) (ha : a ∈ A) : (1 - 1/a) ∈ A := sorry

end problem1_problem2_real_problem2_complex_problem3_l175_175865


namespace average_speed_l175_175317

theorem average_speed (initial final time : ℕ) (h_initial : initial = 2002) (h_final : final = 2332) (h_time : time = 11) : 
  (final - initial) / time = 30 := by
  sorry

end average_speed_l175_175317


namespace number_writing_number_reading_l175_175437

def ten_million_place := 10^7
def hundred_thousand_place := 10^5
def ten_place := 10

def ten_million := 1 * ten_million_place
def three_hundred_thousand := 3 * hundred_thousand_place
def fifty := 5 * ten_place

def constructed_number := ten_million + three_hundred_thousand + fifty

def read_number := "ten million and thirty thousand and fifty"

theorem number_writing : constructed_number = 10300050 := by
  -- Sketch of proof goes here based on place values
  sorry

theorem number_reading : read_number = "ten million and thirty thousand and fifty" := by
  -- Sketch of proof goes here for the reading method
  sorry

end number_writing_number_reading_l175_175437


namespace stratified_sampling_correct_l175_175540

-- Define the total number of students and the ratio of students in grades 10, 11, and 12
def total_students : ℕ := 4000
def ratio_grade10 : ℕ := 32
def ratio_grade11 : ℕ := 33
def ratio_grade12 : ℕ := 35

-- The total sample size
def sample_size : ℕ := 200

-- Define the expected numbers of students drawn from each grade in the sample
def sample_grade10 : ℕ := 64
def sample_grade11 : ℕ := 66
def sample_grade12 : ℕ := 70

-- The theorem to be proved
theorem stratified_sampling_correct :
  (sample_grade10 + sample_grade11 + sample_grade12 = sample_size) ∧
  (sample_grade10 = (ratio_grade10 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade11 = (ratio_grade11 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade12 = (ratio_grade12 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) :=
by
  sorry

end stratified_sampling_correct_l175_175540


namespace Cedar_school_earnings_l175_175049

noncomputable def total_earnings_Cedar_school : ℝ :=
  let total_payment := 774
  let total_student_days := 6 * 4 + 5 * 6 + 3 * 10
  let daily_wage := total_payment / total_student_days
  let Cedar_student_days := 3 * 10
  daily_wage * Cedar_student_days

theorem Cedar_school_earnings :
  total_earnings_Cedar_school = 276.43 :=
by
  sorry

end Cedar_school_earnings_l175_175049


namespace find_number_l175_175856

variable (x : ℝ)

theorem find_number 
  (h1 : 0.20 * x + 0.25 * 60 = 23) :
  x = 40 :=
sorry

end find_number_l175_175856


namespace kolya_start_time_l175_175357

-- Definitions of conditions as per the initial problem statement
def angle_moved_by_minute_hand (x : ℝ) : ℝ := 6 * x
def angle_moved_by_hour_hand (x : ℝ) : ℝ := 30 + 0.5 * x

theorem kolya_start_time (x : ℝ) :
  (angle_moved_by_minute_hand x = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) ∨
  (angle_moved_by_minute_hand x - 180 = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) :=
sorry

end kolya_start_time_l175_175357


namespace bernardo_wins_at_5_l175_175365

theorem bernardo_wins_at_5 :
  (∀ N : ℕ, (16 * N + 900 < 1000) → (920 ≤ 16 * N + 840) → N ≥ 5)
    ∧ (5 < 10 ∧ 16 * 5 + 900 < 1000 ∧ 920 ≤ 16 * 5 + 840) := by
{
  sorry
}

end bernardo_wins_at_5_l175_175365


namespace largest_remainder_a_correct_l175_175688

def largest_remainder_a (n : ℕ) (h : n < 150) : ℕ :=
  (269 % n)

theorem largest_remainder_a_correct : ∃ n < 150, largest_remainder_a n sorry = 133 :=
  sorry

end largest_remainder_a_correct_l175_175688


namespace complete_square_to_d_l175_175920

-- Conditions given in the problem
def quadratic_eq (x : ℝ) : Prop := x^2 + 10 * x + 7 = 0

-- Equivalent Lean 4 statement of the problem
theorem complete_square_to_d (x : ℝ) (c d : ℝ) (h : quadratic_eq x) (hc : c = 5) : (x + c)^2 = d → d = 18 :=
by sorry

end complete_square_to_d_l175_175920


namespace find_k_l175_175107

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, k}
def B : Set ℕ := {2, 5}

-- Given that the union of sets A and B is {1, 2, 3, 5}, prove that k = 3.
theorem find_k (k : ℕ) (h : A k ∪ B = {1, 2, 3, 5}) : k = 3 :=
by
  sorry

end find_k_l175_175107


namespace abs_diff_eq_implies_le_l175_175441

theorem abs_diff_eq_implies_le {x y : ℝ} (h : |x - y| = y - x) : x ≤ y := 
by
  sorry

end abs_diff_eq_implies_le_l175_175441


namespace sum_of_remainders_l175_175534

theorem sum_of_remainders {a b c d e : ℤ} (h1 : a % 13 = 3) (h2 : b % 13 = 5) (h3 : c % 13 = 7) (h4 : d % 13 = 9) (h5 : e % 13 = 11) : 
  ((a + b + c + d + e) % 13) = 9 :=
by
  sorry

end sum_of_remainders_l175_175534


namespace sufficient_condition_for_proposition_l175_175840

theorem sufficient_condition_for_proposition :
  ∀ (a : ℝ), (0 < a ∧ a < 4) → (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) := 
sorry

end sufficient_condition_for_proposition_l175_175840


namespace B_2_2_eq_16_l175_175537

def B : ℕ → ℕ → ℕ
| 0, n       => n + 2
| (m+1), 0   => B m 2
| (m+1), (n+1) => B m (B (m+1) n)

theorem B_2_2_eq_16 : B 2 2 = 16 := by
  sorry

end B_2_2_eq_16_l175_175537


namespace simplest_fraction_is_one_l175_175813

theorem simplest_fraction_is_one :
  ∃ m : ℕ, 
  (∃ k : ℕ, 45 * m = k^2) ∧ 
  (∃ n : ℕ, 56 * m = n^3) → 
  45 * m / 56 * m = 1 := by
  sorry

end simplest_fraction_is_one_l175_175813


namespace tickets_spent_l175_175170

theorem tickets_spent (initial_tickets : ℕ) (tickets_left : ℕ) (tickets_spent : ℕ) 
  (h1 : initial_tickets = 11) (h2 : tickets_left = 8) : tickets_spent = 3 :=
by
  sorry

end tickets_spent_l175_175170


namespace find_larger_number_l175_175759

theorem find_larger_number :
  ∃ x y : ℤ, x + y = 30 ∧ 2 * y - x = 6 ∧ x > y ∧ x = 18 :=
by
  sorry

end find_larger_number_l175_175759


namespace best_fit_of_regression_model_l175_175788

-- Define the context of regression analysis and the coefficient of determination
def regression_analysis : Type := sorry
def coefficient_of_determination (r : regression_analysis) : ℝ := sorry

-- Definitions of each option for clarity in our context
def A (r : regression_analysis) : Prop := sorry -- the linear relationship is stronger
def B (r : regression_analysis) : Prop := sorry -- the linear relationship is weaker
def C (r : regression_analysis) : Prop := sorry -- better fit of the model
def D (r : regression_analysis) : Prop := sorry -- worse fit of the model

-- The formal statement we need to prove
theorem best_fit_of_regression_model (r : regression_analysis) (R2 : ℝ) (h1 : coefficient_of_determination r = R2) (h2 : R2 = 1) : C r :=
by
  sorry

end best_fit_of_regression_model_l175_175788


namespace scientific_notation_population_l175_175179

theorem scientific_notation_population :
    ∃ (a b : ℝ), (b = 5 ∧ 1412.60 * 10 ^ 6 = a * 10 ^ b ∧ a = 1.4126) :=
sorry

end scientific_notation_population_l175_175179


namespace cylinder_volume_l175_175030

theorem cylinder_volume (r h : ℝ) (π : ℝ) 
  (h_pos : 0 < π) 
  (cond1 : 2 * π * r * h = 100 * π) 
  (cond2 : 4 * r^2 + h^2 = 200) : 
  (π * r^2 * h = 250 * π) := 
by 
  sorry

end cylinder_volume_l175_175030


namespace complex_multiplication_l175_175286

variable (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_multiplication :
  i * (2 * i - 1) = -2 - i :=
  sorry

end complex_multiplication_l175_175286


namespace find_range_of_a_l175_175827

-- Definitions
def is_decreasing_function (a : ℝ) : Prop :=
  0 < a ∧ a < 1

def no_real_roots_of_poly (a : ℝ) : Prop :=
  4 * a < 1

def problem_statement (a : ℝ) : Prop :=
  (is_decreasing_function a ∨ no_real_roots_of_poly a) ∧ ¬ (is_decreasing_function a ∧ no_real_roots_of_poly a)

-- Main theorem
theorem find_range_of_a (a : ℝ) : problem_statement a ↔ (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by
  -- Proof omitted
  sorry

end find_range_of_a_l175_175827


namespace Veronica_to_Half_Samir_Ratio_l175_175589

-- Mathematical conditions 
def Samir_stairs : ℕ := 318
def Total_stairs : ℕ := 495
def Half_Samir_stairs : ℚ := Samir_stairs / 2

-- Definition for Veronica's stairs as a multiple of half Samir's stairs
def Veronica_stairs (R: ℚ) : ℚ := R * Half_Samir_stairs

-- Lean statement to prove the ratio
theorem Veronica_to_Half_Samir_Ratio (R : ℚ) (H1 : Veronica_stairs R + Samir_stairs = Total_stairs) : R = 1.1132 := 
by
  sorry

end Veronica_to_Half_Samir_Ratio_l175_175589


namespace infinite_chain_resistance_l175_175368

noncomputable def resistance_of_infinite_chain (R₀ : ℝ) : ℝ :=
  (R₀ * (1 + Real.sqrt 5)) / 2

theorem infinite_chain_resistance : resistance_of_infinite_chain 10 = 5 + 5 * Real.sqrt 5 :=
by
  sorry

end infinite_chain_resistance_l175_175368


namespace number_of_girls_l175_175579

theorem number_of_girls (B G : ℕ) 
  (h1 : B = G + 124) 
  (h2 : B + G = 1250) : G = 563 :=
by
  sorry

end number_of_girls_l175_175579


namespace polygonal_line_exists_l175_175520

theorem polygonal_line_exists (A : Type) (n q : ℕ) (lengths : Fin q → ℝ)
  (yellow_segments : Fin q → (A × A))
  (h_lengths : ∀ i j : Fin q, i < j → lengths i < lengths j)
  (h_yellow_segments_unique : ∀ i j : Fin q, i ≠ j → yellow_segments i ≠ yellow_segments j) :
  ∃ (m : ℕ), m ≥ 2 * q / n :=
sorry

end polygonal_line_exists_l175_175520


namespace parallel_vectors_t_eq_neg1_l175_175207

theorem parallel_vectors_t_eq_neg1 (t : ℝ) :
  let a := (1, -1)
  let b := (t, 1)
  (a.1 + b.1, a.2 + b.2) = (k * (a.1 - b.1), k * (a.2 - b.2)) -> t = -1 :=
by
  sorry

end parallel_vectors_t_eq_neg1_l175_175207


namespace red_car_speed_l175_175945

/-- Dale owns 4 sports cars where:
1. The red car can travel at twice the speed of the green car.
2. The green car can travel at 8 times the speed of the blue car.
3. The blue car can travel at a speed of 80 miles per hour.
We need to determine the speed of the red car. --/
theorem red_car_speed (r g b: ℕ) (h1: r = 2 * g) (h2: g = 8 * b) (h3: b = 80) : 
  r = 1280 :=
by
  sorry

end red_car_speed_l175_175945


namespace problem_statement_l175_175705

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x y : ℝ, f x + f y = f (x + y)) →
  f 3 = 4 →
  f 0 + f (-3) = -4 :=
by
  intros h1 h2
  sorry

end problem_statement_l175_175705


namespace same_parity_iff_exists_c_d_l175_175290

theorem same_parity_iff_exists_c_d (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  (a % 2 = b % 2) ↔ ∃ (c d : ℕ), 0 < c ∧ 0 < d ∧ a^2 + b^2 + c^2 + 1 = d^2 := 
by 
  sorry

end same_parity_iff_exists_c_d_l175_175290


namespace estimate_fish_population_l175_175419

theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) : 
  ∃ N : ℕ, N = m * n / k :=
by
  sorry

end estimate_fish_population_l175_175419


namespace total_spent_l175_175373

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l175_175373


namespace work_together_l175_175010

theorem work_together (W : ℝ) (Dx Dy : ℝ) (hx : Dx = 15) (hy : Dy = 30) : 
  (Dx * Dy) / (Dx + Dy) = 10 := 
by
  sorry

end work_together_l175_175010


namespace hania_age_in_five_years_l175_175993

-- Defining the conditions
variables (H S : ℕ)

-- First condition: Samir's age will be 20 in five years
def condition1 : Prop := S + 5 = 20

-- Second condition: Samir is currently half the age Hania was 10 years ago
def condition2 : Prop := S = (H - 10) / 2

-- The statement to prove: Hania's age in five years will be 45
theorem hania_age_in_five_years (H S : ℕ) (h1 : condition1 S) (h2 : condition2 H S) : H + 5 = 45 :=
sorry

end hania_age_in_five_years_l175_175993


namespace xiao_ming_shopping_l175_175679

theorem xiao_ming_shopping :
  ∃ x : ℕ, x ≤ 16 ∧ 6 * x ≤ 100 ∧ 100 - 6 * x = 28 :=
by
  -- Given that:
  -- 1. x is the same amount spent in each of the six stores.
  -- 2. Total money spent, 6 * x, must be less than or equal to 100.
  -- 3. We seek to prove that Xiao Ming has 28 yuan left.
  sorry

end xiao_ming_shopping_l175_175679


namespace original_radius_of_cylinder_in_inches_l175_175545

theorem original_radius_of_cylinder_in_inches
  (r : ℝ) (h : ℝ) (V : ℝ → ℝ → ℝ → ℝ) 
  (h_increased_radius : V (r + 4) h π = V r (h + 4) π) 
  (h_original_height : h = 3) :
  r = 8 :=
by
  sorry

end original_radius_of_cylinder_in_inches_l175_175545


namespace proof_max_difference_l175_175519

/-- Digits as displayed on the engineering calculator -/
structure Digits :=
  (a b c d e f g h i : ℕ)

-- Possible digits based on broken displays
axiom a_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom b_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom c_values : {x // x = 3 ∨ x = 4 ∨ x = 8 ∨ x = 9}
axiom d_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom e_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom f_values : {x // x = 1 ∨ x = 4 ∨ x = 7}
axiom g_values : {x // x = 4 ∨ x = 5 ∨ x = 9}
axiom h_values : {x // x = 2}
axiom i_values : {x // x = 4 ∨ x = 5 ∨ x = 9}

-- Minuend and subtrahend values
def minuend := 923
def subtrahend := 394

-- Maximum possible value of the difference
def max_difference := 529

theorem proof_max_difference : 
  ∃ (digits : Digits),
    digits.a = 9 ∧ digits.b = 2 ∧ digits.c = 3 ∧
    digits.d = 3 ∧ digits.e = 9 ∧ digits.f = 4 ∧
    digits.g = 5 ∧ digits.h = 2 ∧ digits.i = 9 ∧
    minuend - subtrahend = max_difference :=
by
  sorry

end proof_max_difference_l175_175519


namespace difference_qr_l175_175118

-- Definitions of p, q, r in terms of the common multiplier x
def p (x : ℕ) := 3 * x
def q (x : ℕ) := 7 * x
def r (x : ℕ) := 12 * x

-- Given condition that the difference between p and q's share is 4000
def condition1 (x : ℕ) := q x - p x = 4000

-- Theorem stating that the difference between q and r's share is 5000
theorem difference_qr (x : ℕ) (h : condition1 x) : r x - q x = 5000 :=
by
  -- Proof placeholder
  sorry

end difference_qr_l175_175118


namespace monotonicity_and_range_of_a_l175_175305

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * x + a * Real.log x

theorem monotonicity_and_range_of_a (a : ℝ) (t : ℝ) (ht : t ≥ 1) :
  (∀ x, x > 0 → f x a ≥ f t a - 3) → a ≤ 2 := 
sorry

end monotonicity_and_range_of_a_l175_175305


namespace total_cost_l175_175424

-- Definitions based on the problem's conditions
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 3

def qty_hamburgers : ℕ := 7
def qty_milkshakes : ℕ := 6

-- The proof statement
theorem total_cost :
  (qty_hamburgers * cost_hamburger + qty_milkshakes * cost_milkshake) = 46 :=
by
  sorry

end total_cost_l175_175424


namespace cost_price_is_700_l175_175284

noncomputable def cost_price_was_700 : Prop :=
  ∃ (CP : ℝ),
    (∀ (SP1 SP2 : ℝ),
      SP1 = CP * 0.84 ∧
        SP2 = CP * 1.04 ∧
        SP2 = SP1 + 140) ∧
    CP = 700

theorem cost_price_is_700 : cost_price_was_700 :=
  sorry

end cost_price_is_700_l175_175284


namespace compare_negatives_l175_175226

noncomputable def isNegative (x : ℝ) : Prop := x < 0
noncomputable def absValue (x : ℝ) : ℝ := if x < 0 then -x else x
noncomputable def sqrt14 : ℝ := Real.sqrt 14

theorem compare_negatives : -4 < -Real.sqrt 14 := by
  have h1: Real.sqrt 16 = 4 := by
    sorry
  
  have h2: absValue (-4) = 4 := by
    sorry

  have h3: absValue (-(sqrt14)) = sqrt14 := by
    sorry

  have h4: Real.sqrt 16 > Real.sqrt 14 := by
    sorry

  show -4 < -Real.sqrt 14
  sorry

end compare_negatives_l175_175226


namespace new_socks_bought_l175_175482

theorem new_socks_bought :
  ∀ (original_socks throw_away new_socks total_socks : ℕ),
    original_socks = 28 →
    throw_away = 4 →
    total_socks = 60 →
    total_socks = original_socks - throw_away + new_socks →
    new_socks = 36 :=
by
  intros original_socks throw_away new_socks total_socks h_original h_throw h_total h_eq
  sorry

end new_socks_bought_l175_175482


namespace athena_total_spent_l175_175538

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end athena_total_spent_l175_175538


namespace meaningful_fraction_condition_l175_175805

theorem meaningful_fraction_condition (x : ℝ) : (4 - 2 * x ≠ 0) ↔ (x ≠ 2) :=
by {
  sorry
}

end meaningful_fraction_condition_l175_175805


namespace taxi_ride_cost_is_five_dollars_l175_175457

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def miles_traveled : ℝ := 10.0
def total_cost : ℝ := base_fare + (cost_per_mile * miles_traveled)

theorem taxi_ride_cost_is_five_dollars : total_cost = 5.00 :=
by
  -- proof omitted
  sorry

end taxi_ride_cost_is_five_dollars_l175_175457


namespace cos_alpha_third_quadrant_l175_175287

theorem cos_alpha_third_quadrant (α : ℝ) (hα1 : π < α ∧ α < 3 * π / 2) (hα2 : Real.tan α = 4 / 3) :
  Real.cos α = -3 / 5 :=
sorry

end cos_alpha_third_quadrant_l175_175287


namespace range_of_a_l175_175904

theorem range_of_a (x y a : ℝ) (h1 : 3 * x + y = a + 1) (h2 : x + 3 * y = 3) (h3 : x + y > 5) : a > 16 := 
sorry 

end range_of_a_l175_175904


namespace parabola_condition_max_area_triangle_l175_175037

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l175_175037


namespace geometric_sequence_a_10_l175_175203

noncomputable def geometric_sequence := ℕ → ℝ

def a_3 (a r : ℝ) := a * r^2 = 3
def a_5_equals_8a_7 (a r : ℝ) := a * r^4 = 8 * a * r^6

theorem geometric_sequence_a_10 (a r : ℝ) (seq : geometric_sequence) (h₁ : a_3 a r) (h₂ : a_5_equals_8a_7 a r) :
  seq 10 = a * r^9 := by
  sorry

end geometric_sequence_a_10_l175_175203


namespace present_age_of_B_l175_175273

theorem present_age_of_B :
  ∃ (A B : ℕ), (A + 20 = 2 * (B - 20)) ∧ (A = B + 10) ∧ (B = 70) :=
by
  sorry

end present_age_of_B_l175_175273


namespace ratio_of_administrators_to_teachers_l175_175244

-- Define the conditions
def graduates : ℕ := 50
def parents_per_graduate : ℕ := 2
def teachers : ℕ := 20
def total_chairs : ℕ := 180

-- Calculate intermediate values
def parents : ℕ := graduates * parents_per_graduate
def graduates_and_parents_chairs : ℕ := graduates + parents
def total_graduates_parents_teachers_chairs : ℕ := graduates_and_parents_chairs + teachers
def administrators : ℕ := total_chairs - total_graduates_parents_teachers_chairs

-- Specify the theorem to prove the ratio of administrators to teachers
theorem ratio_of_administrators_to_teachers : administrators / teachers = 1 / 2 :=
by
  -- Proof is omitted; placeholder 'sorry'
  sorry

end ratio_of_administrators_to_teachers_l175_175244


namespace complex_fraction_simplification_l175_175894

theorem complex_fraction_simplification :
  ((10^4 + 324) * (22^4 + 324) * (34^4 + 324) * (46^4 + 324) * (58^4 + 324)) /
  ((4^4 + 324) * (16^4 + 324) * (28^4 + 324) * (40^4 + 324) * (52^4 + 324)) = 373 :=
by
  sorry

end complex_fraction_simplification_l175_175894


namespace saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l175_175655

noncomputable def bread_saving (n_days : ℕ) : ℕ :=
  (1 / 2) * n_days

theorem saving_20_days :
  bread_saving 20 = 10 :=
by
  -- proof steps for bread_saving 20 = 10
  sorry

theorem cost_saving_20_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 20 * cost_per_loaf) = 350 :=
by
  -- proof steps for cost_saving_20_days
  sorry

theorem saving_60_days :
  bread_saving 60 = 30 :=
by
  -- proof steps for bread_saving 60 = 30
  sorry

theorem cost_saving_60_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 60 * cost_per_loaf) = 1050 :=
by
  -- proof steps for cost_saving_60_days
  sorry

end saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l175_175655


namespace total_amount_l175_175128

-- Define p, q, r and their shares
variables (p q r : ℕ)

-- Given conditions translated to Lean definitions
def ratio_pq := (5 * q) = (4 * p)
def ratio_qr := (9 * r) = (10 * q)
def r_share := r = 400

-- Statement to prove
theorem total_amount (hpq : ratio_pq p q) (hqr : ratio_qr q r) (hr : r_share r) :
  (p + q + r) = 1210 :=
by
  sorry

end total_amount_l175_175128


namespace problem_solution_l175_175033

theorem problem_solution (a : ℝ) (h : a = Real.sqrt 5 - 1) :
  2 * a^3 + 7 * a^2 - 2 * a - 12 = 0 :=
by 
  sorry  -- Proof placeholder

end problem_solution_l175_175033


namespace h_at_0_l175_175672

noncomputable def h (x : ℝ) : ℝ := sorry -- the actual polynomial
-- Conditions for h(x)
axiom h_cond1 : h (-2) = -4
axiom h_cond2 : h (1) = -1
axiom h_cond3 : h (-3) = -9
axiom h_cond4 : h (3) = -9
axiom h_cond5 : h (5) = -25

-- Statement of the proof problem
theorem h_at_0 : h (0) = -90 := sorry

end h_at_0_l175_175672


namespace candy_store_spending_l175_175889

variable (weekly_allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)

def remaining_after_arcade (weekly_allowance arcade_fraction : ℝ) : ℝ :=
  weekly_allowance * (1 - arcade_fraction)

def remaining_after_toy_store (remaining_allowance toy_store_fraction : ℝ) : ℝ :=
  remaining_allowance * (1 - toy_store_fraction)

theorem candy_store_spending
  (h1 : weekly_allowance = 3.30)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : toy_store_fraction = 1 / 3) :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance arcade_fraction) toy_store_fraction = 0.88 := 
sorry

end candy_store_spending_l175_175889


namespace intersection_eq_l175_175873

def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem intersection_eq : M ∩ N = {x | 2 < x ∧ x < 3} := 
by
  sorry

end intersection_eq_l175_175873


namespace total_amount_paid_l175_175542

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end total_amount_paid_l175_175542


namespace math_problem_l175_175988

theorem math_problem (a b : ℝ) (h : a * b < 0) : a^2 * |b| - b^2 * |a| + a * b * (|a| - |b|) = 0 :=
sorry

end math_problem_l175_175988


namespace temperature_value_l175_175125

theorem temperature_value (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 221) : t = 105 :=
by
  sorry

end temperature_value_l175_175125


namespace exponential_decreasing_l175_175924

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a+1)^x > (a+1)^y) : -1 < a ∧ a < 0 :=
sorry

end exponential_decreasing_l175_175924


namespace parabola_symmetry_product_l175_175977

theorem parabola_symmetry_product (a p m : ℝ) 
  (hpr1 : a ≠ 0) 
  (hpr2 : p > 0) 
  (hpr3 : ∀ (x₀ y₀ : ℝ), y₀^2 = 2*p*x₀ → (a*(y₀ - m)^2 - 3*(y₀ - m) + 3 = x₀ + m)) :
  a * p * m = -3 := 
sorry

end parabola_symmetry_product_l175_175977


namespace find_n_modulo_conditions_l175_175089

theorem find_n_modulo_conditions :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n % 7 = -3137 % 7 ∧ (n = 1 ∨ n = 8) := sorry

end find_n_modulo_conditions_l175_175089


namespace total_points_scored_l175_175652

theorem total_points_scored :
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  a + b + c + d + e + f + g + h = 54 :=
by
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  sorry

end total_points_scored_l175_175652


namespace find_a_b_c_sum_l175_175674

theorem find_a_b_c_sum (a b c : ℝ) 
  (h_vertex : ∀ x, y = a * x^2 + b * x + c ↔ y = a * (x - 3)^2 + 5)
  (h_passes : a * 1^2 + b * 1 + c = 2) :
  a + b + c = 35 / 4 :=
sorry

end find_a_b_c_sum_l175_175674


namespace equal_cookies_per_person_l175_175621

theorem equal_cookies_per_person 
  (boxes : ℕ) (cookies_per_box : ℕ) (people : ℕ)
  (h1 : boxes = 7) (h2 : cookies_per_box = 10) (h3 : people = 5) :
  (boxes * cookies_per_box) / people = 14 :=
by sorry

end equal_cookies_per_person_l175_175621


namespace not_divisible_1978_1000_l175_175603

theorem not_divisible_1978_1000 (m : ℕ) : ¬ ∃ m : ℕ, (1000^m - 1) ∣ (1978^m - 1) := sorry

end not_divisible_1978_1000_l175_175603


namespace dog_biscuit_cost_l175_175866

open Real

theorem dog_biscuit_cost :
  (∀ (x : ℝ),
    (4 * x + 2) * 7 = 21 →
    x = 1 / 4) :=
by
  intro x h
  sorry

end dog_biscuit_cost_l175_175866


namespace intersection_P_Q_l175_175623

open Set

noncomputable def P : Set ℝ := {1, 2, 3, 4}

noncomputable def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := 
by {
  sorry
}

end intersection_P_Q_l175_175623


namespace least_value_of_a_plus_b_l175_175550

def a_and_b (a b : ℕ) : Prop :=
  (Nat.gcd (a + b) 330 = 1) ∧ 
  (a^a % b^b = 0) ∧ 
  (¬ (a % b = 0))

theorem least_value_of_a_plus_b :
  ∃ (a b : ℕ), a_and_b a b ∧ a + b = 105 :=
sorry

end least_value_of_a_plus_b_l175_175550


namespace length_of_uncovered_side_l175_175347

theorem length_of_uncovered_side (L W : ℕ) (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end length_of_uncovered_side_l175_175347


namespace tangent_line_at_neg1_l175_175040

-- Define the function given in the condition.
def f (x : ℝ) : ℝ := x^2 + 4 * x + 2

-- Define the point of tangency given in the condition.
def point_of_tangency : ℝ × ℝ := (-1, f (-1))

-- Define the derivative of the function.
def derivative_f (x : ℝ) : ℝ := 2 * x + 4

-- The proof statement: the equation of the tangent line at x = -1 is y = 2x + 1
theorem tangent_line_at_neg1 :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = f x → derivative_f (-1) = m ∧ point_of_tangency.fst = -1 ∧ y = m * (x + 1) + b) :=
sorry

end tangent_line_at_neg1_l175_175040


namespace Marcy_spears_l175_175432

def makeSpears (saplings: ℕ) (logs: ℕ) (branches: ℕ) (trunks: ℕ) : ℕ :=
  3 * saplings + 9 * logs + 7 * branches + 15 * trunks

theorem Marcy_spears :
  makeSpears 12 1 6 0 - (3 * 2) + makeSpears 0 4 0 0 - (9 * 4) + makeSpears 0 0 6 1 - (7 * 0) + makeSpears 0 0 0 2 = 81 := by
  sorry

end Marcy_spears_l175_175432


namespace chords_and_circle_l175_175857

theorem chords_and_circle (R : ℝ) (A B C D : ℝ) 
  (hAB : 0 < A - B) (hCD : 0 < C - D) (hR : R > 0) 
  (h_perp : (A - B) * (C - D) = 0) 
  (h_radA : A ^ 2 + B ^ 2 = R ^ 2) 
  (h_radC : C ^ 2 + D ^ 2 = R ^ 2) :
  (A - C)^2 + (B - D)^2 = 4 * R^2 :=
by
  sorry

end chords_and_circle_l175_175857


namespace algebraic_expression_value_l175_175536

theorem algebraic_expression_value (b a c : ℝ) (h₁ : b < a) (h₂ : a < 0) (h₃ : 0 < c) :
  |b| - |b - a| + |c - a| - |a + b| = b + c - a :=
by
  sorry

end algebraic_expression_value_l175_175536


namespace num_perfect_square_factors_l175_175907

def prime_factors_9600 (n : ℕ) : Prop :=
  n = 9600

theorem num_perfect_square_factors (n : ℕ) (h : prime_factors_9600 n) : 
  let cond := h
  (n = 9600) → 9600 = 2^6 * 5^2 * 3^1 → (∃ factors_count: ℕ, factors_count = 8) := by 
  sorry

end num_perfect_square_factors_l175_175907


namespace domain_f_correct_domain_g_correct_l175_175159

noncomputable def domain_f : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ x ≠ 1}

noncomputable def expected_domain_f : Set ℝ :=
  {x | (-1 ≤ x ∧ x < 1) ∨ x > 1}

theorem domain_f_correct :
  domain_f = expected_domain_f :=
by
  sorry

noncomputable def domain_g : Set ℝ :=
  {x | 3 - 4 * x > 0}

noncomputable def expected_domain_g : Set ℝ :=
  {x | x < 3 / 4}

theorem domain_g_correct :
  domain_g = expected_domain_g :=
by
  sorry

end domain_f_correct_domain_g_correct_l175_175159


namespace sin_beta_value_l175_175034

theorem sin_beta_value (a β : ℝ) (ha : 0 < a ∧ a < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hcos_a : Real.cos a = 4 / 5)
  (hcos_a_plus_beta : Real.cos (a + β) = 5 / 13) :
  Real.sin β = 63 / 65 :=
sorry

end sin_beta_value_l175_175034


namespace length_of_AB_l175_175580

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_AB :
  let O := (0, 0)
  let A := (54^(1/3), 0)
  let B := (0, 54^(1/3))
  distance A B = 54^(1/3) * Real.sqrt 2 :=
by
  sorry

end length_of_AB_l175_175580


namespace common_difference_arithmetic_sequence_l175_175782

-- Define the arithmetic sequence properties
variable (S : ℕ → ℕ) -- S represents the sum of the first n terms
variable (a : ℕ → ℕ) -- a represents the terms in the arithmetic sequence
variable (d : ℤ) -- common difference

-- Define the conditions
axiom S2_eq_6 : S 2 = 6
axiom a1_eq_4 : a 1 = 4

-- The problem: show that d = -2
theorem common_difference_arithmetic_sequence :
  (a 2 - a 1 = d) → d = -2 :=
by
  sorry

end common_difference_arithmetic_sequence_l175_175782


namespace base6_to_decimal_l175_175172

theorem base6_to_decimal (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end base6_to_decimal_l175_175172


namespace henry_seashells_l175_175154

theorem henry_seashells (H L : ℕ) (h1 : H + 24 + L = 59) (h2 : H + 24 + (3 * L) / 4 = 53) : H = 11 := by
  sorry

end henry_seashells_l175_175154


namespace interest_rate_second_type_l175_175771

variable (totalInvestment : ℝ) (interestFirstTypeRate : ℝ) (investmentSecondType : ℝ) (totalInterestRate : ℝ) 
variable [Nontrivial ℝ]

theorem interest_rate_second_type :
    totalInvestment = 100000 ∧
    interestFirstTypeRate = 0.09 ∧
    investmentSecondType = 29999.999999999993 ∧
    totalInterestRate = 9 + 3 / 5 →
    (9.6 * totalInvestment - (interestFirstTypeRate * (totalInvestment - investmentSecondType))) / investmentSecondType = 0.11 :=
by
  sorry

end interest_rate_second_type_l175_175771


namespace interest_rate_per_annum_l175_175404
noncomputable def interest_rate_is_10 : ℝ := 10
theorem interest_rate_per_annum (P R : ℝ) : 
  (1200 * ((1 + R / 100)^2 - 1) - 1200 * R * 2 / 100 = 12) → P = 1200 → R = 10 := 
by sorry

end interest_rate_per_annum_l175_175404


namespace owen_sleep_hours_l175_175878

-- Define the time spent by Owen in various activities
def hours_work : ℕ := 6
def hours_chores : ℕ := 7
def total_hours_day : ℕ := 24

-- The proposition to be proven
theorem owen_sleep_hours : (total_hours_day - (hours_work + hours_chores) = 11) := by
  sorry

end owen_sleep_hours_l175_175878


namespace find_x_l175_175841

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 80) : x = 26 :=
sorry

end find_x_l175_175841


namespace sum_of_coefficients_l175_175921

theorem sum_of_coefficients (s : ℕ → ℝ) (a b c : ℝ) : 
  s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 17 ∧ 
  (∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) → 
  a + b + c = 12 := 
by
  sorry

end sum_of_coefficients_l175_175921


namespace pure_imaginary_solution_l175_175725

theorem pure_imaginary_solution (m : ℝ) 
  (h : ∃ m : ℝ, (m^2 + m - 2 = 0) ∧ (m^2 - 1 ≠ 0)) : m = -2 :=
sorry

end pure_imaginary_solution_l175_175725


namespace solve_for_x_l175_175611

theorem solve_for_x (x : ℤ) (h : 24 - 6 = 3 + x) : x = 15 :=
by {
  sorry
}

end solve_for_x_l175_175611


namespace total_rainfall_2007_correct_l175_175130

noncomputable def rainfall_2005 : ℝ := 40.5
noncomputable def rainfall_2006 : ℝ := rainfall_2005 + 3
noncomputable def rainfall_2007 : ℝ := rainfall_2006 + 4
noncomputable def total_rainfall_2007 : ℝ := 12 * rainfall_2007

theorem total_rainfall_2007_correct : total_rainfall_2007 = 570 := 
sorry

end total_rainfall_2007_correct_l175_175130


namespace jeremy_home_to_school_distance_l175_175082

theorem jeremy_home_to_school_distance (v d : ℝ) (h1 : 30 / 60 = 1 / 2) (h2 : 15 / 60 = 1 / 4)
  (h3 : d = v * (1 / 2)) (h4 : d = (v + 12) * (1 / 4)):
  d = 6 :=
by
  -- We assume that the conditions given lead to the distance being 6 miles
  sorry

end jeremy_home_to_school_distance_l175_175082


namespace perpendicular_lines_l175_175810

-- Definitions of conditions
def condition1 (α β γ δ : ℝ) : Prop := α = 90 ∧ α + β = 180 ∧ α + γ = 180 ∧ α + δ = 180
def condition2 (α β γ δ : ℝ) : Prop := α = β ∧ β = γ ∧ γ = δ
def condition3 (α β : ℝ) : Prop := α = β ∧ α + β = 180
def condition4 (α β : ℝ) : Prop := α = β ∧ α + β = 180

-- Main theorem statement
theorem perpendicular_lines (α β γ δ : ℝ) :
  (condition1 α β γ δ ∨ condition2 α β γ δ ∨
   condition3 α β ∨ condition4 α β) → α = 90 :=
by sorry

end perpendicular_lines_l175_175810


namespace find_real_pairs_l175_175018

theorem find_real_pairs (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end find_real_pairs_l175_175018


namespace radius_of_circle_l175_175506

-- Define the problem condition
def diameter_of_circle : ℕ := 14

-- State the problem as a theorem
theorem radius_of_circle (d : ℕ) (hd : d = diameter_of_circle) : d / 2 = 7 := by 
  sorry

end radius_of_circle_l175_175506


namespace negation_of_statement_l175_175336

theorem negation_of_statement (h: ∀ x : ℝ, |x| + x^2 ≥ 0) :
  ¬ (∀ x : ℝ, |x| + x^2 ≥ 0) ↔ ∃ x : ℝ, |x| + x^2 < 0 :=
by
  sorry

end negation_of_statement_l175_175336


namespace no_nat_solutions_no_int_solutions_l175_175814

theorem no_nat_solutions (x y : ℕ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

theorem no_int_solutions (x y : ℤ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

end no_nat_solutions_no_int_solutions_l175_175814


namespace unique_7_tuple_count_l175_175750

theorem unique_7_tuple_count :
  ∃! (x : ℕ → ℝ) (zero_le_x : (∀ i, 0 ≤ i → i ≤ 6 → true)),
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 8 :=
by
  sorry

end unique_7_tuple_count_l175_175750


namespace calculate_expression_l175_175735

theorem calculate_expression :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5 / 4 :=
by
  sorry

end calculate_expression_l175_175735


namespace ball_redistribution_impossible_l175_175196

noncomputable def white_boxes_initial_ball_count := 31
noncomputable def black_boxes_initial_ball_count := 26
noncomputable def white_boxes_new_ball_count := 21
noncomputable def black_boxes_new_ball_count := 16
noncomputable def white_boxes_target_ball_count := 15
noncomputable def black_boxes_target_ball_count := 10

theorem ball_redistribution_impossible
  (initial_white_boxes : ℕ)
  (initial_black_boxes : ℕ)
  (new_white_boxes : ℕ)
  (new_black_boxes : ℕ)
  (total_white_boxes : ℕ)
  (total_black_boxes : ℕ) :
  initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count =
  total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count →
  (new_white_boxes, new_black_boxes) = (total_white_boxes - initial_white_boxes, total_black_boxes - initial_black_boxes) →
  ¬(∃ total_white_boxes total_black_boxes, 
    total_white_boxes * white_boxes_target_ball_count +
    total_black_boxes * black_boxes_target_ball_count =
    initial_white_boxes * white_boxes_initial_ball_count +
    initial_black_boxes * black_boxes_initial_ball_count) :=
by sorry

end ball_redistribution_impossible_l175_175196


namespace calc_expression_l175_175447

theorem calc_expression :
  let a := 3^456
  let b := 9^5 / 9^3
  a - b = 3^456 - 81 :=
by
  let a := 3^456
  let b := 9^5 / 9^3
  sorry

end calc_expression_l175_175447


namespace speed_of_current_l175_175831
  
  theorem speed_of_current (v c : ℝ)
    (h1 : 64 = (v + c) * 8)
    (h2 : 24 = (v - c) * 8) :
    c = 2.5 :=
  by {
    sorry
  }
  
end speed_of_current_l175_175831


namespace min_perimeter_is_676_l175_175326

-- Definitions and conditions based on the problem statement
def equal_perimeter (a b c : ℕ) : Prop :=
  2 * a + 14 * c = 2 * b + 16 * c

def equal_area (a b c : ℕ) : Prop :=
  7 * Real.sqrt (a^2 - 49 * c^2) = 8 * Real.sqrt (b^2 - 64 * c^2)

def base_ratio (b : ℕ) : ℕ := b * 8 / 7

theorem min_perimeter_is_676 :
  ∃ a b c : ℕ, equal_perimeter a b c ∧ equal_area a b c ∧ base_ratio b = a - b ∧ 
  2 * a + 14 * c = 676 :=
sorry

end min_perimeter_is_676_l175_175326


namespace monotonic_intervals_extremum_values_l175_175854

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 8

theorem monotonic_intervals :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, x > 2 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < 2 → deriv f x < 0) := sorry

theorem extremum_values :
  ∃ a b : ℝ, (a = -12) ∧ (b = 15) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≥ b → f x = b) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 3 → f x ≤ a → f x = a) := sorry

end monotonic_intervals_extremum_values_l175_175854


namespace product_of_digits_l175_175165

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 4 = 0) : A * B = 32 ∨ A * B = 36 :=
sorry

end product_of_digits_l175_175165


namespace area_T_l175_175568

variable (T : Set (ℝ × ℝ)) -- T is a region in the plane
variable (A : Matrix (Fin 2) (Fin 2) ℝ) -- A is a 2x2 matrix
variable (detA : ℝ) -- detA is the determinant of A

-- assumptions
axiom area_T : ∃ (area : ℝ), area = 9
axiom matrix_A : A = ![![3, 2], ![-1, 4]]
axiom determinant_A : detA = 14

-- statement to prove
theorem area_T' : ∃ area_T' : ℝ, area_T' = 126 :=
sorry

end area_T_l175_175568


namespace tiffany_lives_next_level_l175_175822

theorem tiffany_lives_next_level (L1 L2 L3 : ℝ)
    (h1 : L1 = 43.0)
    (h2 : L2 = 14.0)
    (h3 : L3 = 84.0) :
    L3 - (L1 + L2) = 27 :=
by
  rw [h1, h2, h3]
  -- The proof is skipped with "sorry"
  sorry

end tiffany_lives_next_level_l175_175822


namespace equilateral_triangle_in_ellipse_l175_175493

def ellipse_equation (x y a b : ℝ) : Prop := 
  ((x - y)^2 / a^2) + ((x + y)^2 / b^2) = 1

theorem equilateral_triangle_in_ellipse 
  {a b x y : ℝ}
  (A B C : ℝ × ℝ)
  (hA : A.1 = 0 ∧ A.2 = b)
  (hBC_parallel : ∃ k : ℝ, B.2 = k * B.1 ∧ C.2 = k * C.1 ∧ k = 1)
  (hF : ∃ F : ℝ × ℝ, F = C)
  (hEllipseA : ellipse_equation A.1 A.2 a b) 
  (hEllipseB : ellipse_equation B.1 B.2 a b)
  (hEllipseC : ellipse_equation C.1 C.2 a b) 
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
  AB / b = 8 / 5 :=
sorry

end equilateral_triangle_in_ellipse_l175_175493


namespace initial_tickets_l175_175103

-- Definitions of the conditions
def ferris_wheel_rides : ℕ := 2
def roller_coaster_rides : ℕ := 3
def log_ride_rides : ℕ := 7

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 5
def log_ride_cost : ℕ := 1

def additional_tickets_needed : ℕ := 6

-- Calculate the total number of tickets needed
def total_tickets_needed : ℕ := 
  (ferris_wheel_rides * ferris_wheel_cost) +
  (roller_coaster_rides * roller_coaster_cost) +
  (log_ride_rides * log_ride_cost)

-- The proof statement
theorem initial_tickets : ∀ (initial_tickets : ℕ), 
  total_tickets_needed - additional_tickets_needed = initial_tickets → 
  initial_tickets = 20 :=
by
  intros initial_tickets h
  sorry

end initial_tickets_l175_175103


namespace arithmetic_mean_of_fractions_l175_175954

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 5
  let b := (5 : ℚ) / 7
  (a + b) / 2 = (23 : ℚ) / 35 := 
by 
  sorry 

end arithmetic_mean_of_fractions_l175_175954


namespace sum_of_remainders_correct_l175_175658

def sum_of_remainders : ℕ :=
  let remainders := [43210 % 37, 54321 % 37, 65432 % 37, 76543 % 37, 87654 % 37, 98765 % 37]
  remainders.sum

theorem sum_of_remainders_correct : sum_of_remainders = 36 :=
by sorry

end sum_of_remainders_correct_l175_175658


namespace total_students_in_class_is_15_l175_175428

noncomputable def choose (n k : ℕ) : ℕ := sorry -- Define a function for combinations
noncomputable def permute (n k : ℕ) : ℕ := sorry -- Define a function for permutations

variables (x m n : ℕ) (hx : choose x 4 = m) (hn : permute x 2 = n) (hratio : m * 2 = n * 13)

theorem total_students_in_class_is_15 : x = 15 :=
sorry

end total_students_in_class_is_15_l175_175428


namespace inverse_100_mod_101_l175_175253

theorem inverse_100_mod_101 : (100 * 100) % 101 = 1 :=
by
  -- Proof can be provided here.
  sorry

end inverse_100_mod_101_l175_175253


namespace derivative_at_one_eq_neg_one_l175_175819

variable {α : Type*} [TopologicalSpace α] {f : ℝ → ℝ}
-- condition: f is differentiable
variable (hf_diff : Differentiable ℝ f)
-- condition: limit condition
variable (h_limit : Tendsto (fun Δx => (f (1 + 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-2)))

-- proof goal: f'(1) = -1
theorem derivative_at_one_eq_neg_one : deriv f 1 = -1 := 
by
  sorry

end derivative_at_one_eq_neg_one_l175_175819


namespace solve_for_r_l175_175684

theorem solve_for_r (r : ℚ) (h : (r + 4) / (r - 3) = (r - 2) / (r + 2)) : r = -2/11 :=
by
  sorry

end solve_for_r_l175_175684


namespace max_f_value_l175_175789

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x / (x^2 + m)

theorem max_f_value (m : ℝ) : 
  (m > 1) ↔ (∀ x : ℝ, f x m < 1) ∧ ¬((∀ x : ℝ, f x m < 1) → (m > 1)) :=
by
  sorry

end max_f_value_l175_175789


namespace tiles_needed_to_cover_floor_l175_175300

/-- 
A floor 10 feet by 15 feet is to be tiled with 3-inch-by-9-inch tiles. 
This theorem verifies that the necessary number of tiles is 800. 
-/
theorem tiles_needed_to_cover_floor
  (floor_length : ℝ)
  (floor_width : ℝ)
  (tile_length_inch : ℝ)
  (tile_width_inch : ℝ)
  (conversion_factor : ℝ)
  (num_tiles : ℕ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 15)
  (h_tile_length_inch : tile_length_inch = 3)
  (h_tile_width_inch : tile_width_inch = 9)
  (h_conversion_factor : conversion_factor = 12)
  (h_num_tiles : num_tiles = 800) :
  (floor_length * floor_width) / ((tile_length_inch / conversion_factor) * (tile_width_inch / conversion_factor)) = num_tiles :=
by
  -- The proof is not included, using sorry to mark this part
  sorry

end tiles_needed_to_cover_floor_l175_175300


namespace union_M_N_l175_175729

def M : Set ℝ := { x | x^2 + 2 * x = 0 }
def N : Set ℝ := { x | x^2 - 2 * x = 0 }

theorem union_M_N : M ∪ N = {0, -2, 2} := by
  sorry

end union_M_N_l175_175729


namespace ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l175_175065

-- Define the basic conditions of the figures
def regular_pentagon (side_length : ℕ) : ℝ := 5 * side_length

-- Define ink length of a figure n
def ink_length (n : ℕ) : ℝ :=
  if n = 1 then regular_pentagon 1 else
  regular_pentagon (n-1) + (3 * (n - 1) + 2)

-- Part (a): Ink length of Figure 4
theorem ink_length_figure_4 : ink_length 4 = 38 := 
  by sorry

-- Part (b): Difference between ink length of Figure 9 and Figure 8
theorem ink_length_difference_9_8 : ink_length 9 - ink_length 8 = 29 :=
  by sorry

-- Part (c): Ink length of Figure 100
theorem ink_length_figure_100 : ink_length 100 = 15350 :=
  by sorry

end ink_length_figure_4_ink_length_difference_9_8_ink_length_figure_100_l175_175065


namespace intersection_of_A_and_B_l175_175422

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l175_175422


namespace man_completes_in_9_days_l175_175022

-- Definitions of the work rates and the conditions given
def M : ℚ := sorry
def W : ℚ := 1 / 6
def B : ℚ := 1 / 18
def combined_rate : ℚ := 1 / 3

-- Statement that the man alone can complete the work in 9 days
theorem man_completes_in_9_days
  (h_combined : M + W + B = combined_rate) : 1 / M = 9 :=
  sorry

end man_completes_in_9_days_l175_175022


namespace problem1_l175_175640

theorem problem1 (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^(n+1) - 9) := 
  sorry

end problem1_l175_175640


namespace odd_n_divides_pow_fact_sub_one_l175_175086

theorem odd_n_divides_pow_fact_sub_one
  {n : ℕ} (hn_pos : n > 0) (hn_odd : n % 2 = 1)
  : n ∣ (2 ^ (Nat.factorial n) - 1) :=
sorry

end odd_n_divides_pow_fact_sub_one_l175_175086


namespace total_amount_received_l175_175617

theorem total_amount_received (P R CI: ℝ) (T: ℕ) 
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (P_eq: P = 2828.80 / 0.1664) 
  (R_eq: R = 8) 
  (T_eq: T = 2) : 
  P + CI = 19828.80 := 
by 
  sorry

end total_amount_received_l175_175617


namespace find_ratio_l175_175164

open Real

-- Definitions and conditions
variables (b1 b2 : ℝ) (F1 F2 : ℝ × ℝ)
noncomputable def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 49) + (Q.2^2 / b1^2) = 1
noncomputable def hyperbola_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 16) - (Q.2^2 / b2^2) = 1
noncomputable def same_foci (Q : ℝ × ℝ) : Prop := true  -- Placeholder: Representing that both shapes have the same foci F1 and F2

-- The main theorem
theorem find_ratio (Q : ℝ × ℝ) (h1 : ellipse_eq b1 Q) (h2 : hyperbola_eq b2 Q) (h3 : same_foci Q) : 
  abs ((dist Q F1) - (dist Q F2)) / ((dist Q F1) + (dist Q F2)) = 4 / 7 := 
sorry

end find_ratio_l175_175164


namespace necessary_and_sufficient_condition_l175_175363

theorem necessary_and_sufficient_condition {a : ℝ} :
    (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end necessary_and_sufficient_condition_l175_175363


namespace find_a_l175_175375

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end find_a_l175_175375


namespace numberOfCows_l175_175173

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

end numberOfCows_l175_175173


namespace compute_abc_l175_175875

theorem compute_abc (a b c : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h₁ : a + b + c = 30) 
  (h₂ : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c + 300/(a * b * c) = 1) : a * b * c = 768 := 
by 
  sorry

end compute_abc_l175_175875


namespace stock_price_rise_l175_175312

theorem stock_price_rise {P : ℝ} (h1 : P > 0)
    (h2007 : P * 1.20 = 1.20 * P)
    (h2008 : 1.20 * P * 0.75 = P * 0.90)
    (hCertainYear : P * 1.17 = P * 0.90 * (1 + 30 / 100)) :
  30 = 30 :=
by sorry

end stock_price_rise_l175_175312


namespace oranges_savings_l175_175900

-- Definitions for the conditions
def liam_oranges : Nat := 40
def liam_price_per_set : Real := 2.50
def oranges_per_set : Nat := 2

def claire_oranges : Nat := 30
def claire_price_per_orange : Real := 1.20

-- Statement of the problem to be proven
theorem oranges_savings : 
  liam_oranges / oranges_per_set * liam_price_per_set + 
  claire_oranges * claire_price_per_orange = 86 := 
by 
  sorry

end oranges_savings_l175_175900


namespace symmetric_points_y_axis_l175_175379

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : (a, 3) = (-2, 3)) (h₂ : (2, b) = (2, 3)) : (a + b) ^ 2015 = 1 := by
  sorry

end symmetric_points_y_axis_l175_175379


namespace probability_all_operating_probability_shutdown_l175_175048

-- Define the events and their probabilities
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Prove that the probability of all three machines operating without supervision is 0.612
theorem probability_all_operating : P_A * P_B * P_C = 0.612 := 
by sorry

-- Prove that the probability of a shutdown is 0.059
theorem probability_shutdown :
    P_A * (1 - P_B) * (1 - P_C) +
    (1 - P_A) * P_B * (1 - P_C) +
    (1 - P_A) * (1 - P_B) * P_C +
    (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059 :=
by sorry

end probability_all_operating_probability_shutdown_l175_175048


namespace xiao_li_more_stable_l175_175959

def average_xiao_li : ℝ := 95
def average_xiao_zhang : ℝ := 95

def variance_xiao_li : ℝ := 0.55
def variance_xiao_zhang : ℝ := 1.35

theorem xiao_li_more_stable : 
  variance_xiao_li < variance_xiao_zhang :=
by
  sorry

end xiao_li_more_stable_l175_175959


namespace room_length_l175_175700

def area_four_walls (L: ℕ) (w: ℕ) (h: ℕ) : ℕ :=
  2 * (L * h) + 2 * (w * h)

def area_door (d_w: ℕ) (d_h: ℕ) : ℕ :=
  d_w * d_h

def area_windows (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  num_windows * (win_w * win_h)

def total_area_to_whitewash (L: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  area_four_walls L w h - area_door d_w d_h - area_windows win_w win_h num_windows

theorem room_length (cost: ℕ) (rate: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) (L: ℕ) :
  cost = rate * total_area_to_whitewash L w h d_w d_h win_w win_h num_windows →
  L = 25 :=
by
  have h1 : total_area_to_whitewash 25 15 12 6 3 4 3 3 = 24 * 25 + 306 := sorry
  have h2 : rate * (24 * 25 + 306) = 5436 := sorry
  sorry

end room_length_l175_175700


namespace arman_age_in_years_l175_175355

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end arman_age_in_years_l175_175355


namespace abc_value_l175_175825

variables (a b c : ℂ)

theorem abc_value :
  (a * b + 4 * b = -16) →
  (b * c + 4 * c = -16) →
  (c * a + 4 * a = -16) →
  a * b * c = 64 :=
by
  intros h1 h2 h3
  sorry

end abc_value_l175_175825


namespace jeff_total_distance_l175_175870

-- Define the conditions as constants
def speed1 : ℝ := 80
def time1 : ℝ := 3

def speed2 : ℝ := 50
def time2 : ℝ := 2

def speed3 : ℝ := 70
def time3 : ℝ := 1

def speed4 : ℝ := 60
def time4 : ℝ := 2

def speed5 : ℝ := 45
def time5 : ℝ := 3

def speed6 : ℝ := 40
def time6 : ℝ := 2

def speed7 : ℝ := 30
def time7 : ℝ := 2.5

-- Define the equation for the total distance traveled
def total_distance : ℝ :=
  speed1 * time1 + 
  speed2 * time2 + 
  speed3 * time3 + 
  speed4 * time4 + 
  speed5 * time5 + 
  speed6 * time6 + 
  speed7 * time7

-- Prove that the total distance is equal to 820 miles
theorem jeff_total_distance : total_distance = 820 := by
  sorry

end jeff_total_distance_l175_175870


namespace right_triangle_sides_l175_175416

theorem right_triangle_sides (x y z : ℕ) (h1 : x + y + z = 30)
    (h2 : x^2 + y^2 + z^2 = 338) (h3 : x^2 + y^2 = z^2) :
    (x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13) :=
by
  sorry

end right_triangle_sides_l175_175416


namespace cannot_form_right_triangle_l175_175348

theorem cannot_form_right_triangle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h1, h2, h3]
  sorry

end cannot_form_right_triangle_l175_175348


namespace capacity_of_each_bag_is_approximately_63_l175_175261

noncomputable def capacity_of_bag (total_sand : ℤ) (num_bags : ℤ) : ℤ :=
  Int.ceil (total_sand / num_bags)

theorem capacity_of_each_bag_is_approximately_63 :
  capacity_of_bag 757 12 = 63 :=
by
  sorry

end capacity_of_each_bag_is_approximately_63_l175_175261


namespace sum_infinite_geometric_series_l175_175302

theorem sum_infinite_geometric_series : 
  let a : ℝ := 2
  let r : ℝ := -5/8
  a / (1 - r) = 16/13 :=
by
  sorry

end sum_infinite_geometric_series_l175_175302


namespace sum_first_10_terms_abs_a_n_l175_175803

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else 3 * n - 7

def abs_a_n (n : ℕ) : ℤ :=
  if n = 1 ∨ n = 2 then -3 * n + 7 else 3 * n - 7

def sum_abs_a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else List.sum (List.map abs_a_n (List.range n))

theorem sum_first_10_terms_abs_a_n : sum_abs_a_n 10 = 105 := 
  sorry

end sum_first_10_terms_abs_a_n_l175_175803


namespace red_stars_eq_35_l175_175015

-- Define the conditions
noncomputable def number_of_total_stars (x : ℕ) : ℕ := x + 20 + 15
noncomputable def red_star_frequency (x : ℕ) : ℚ := x / (number_of_total_stars x : ℚ)

-- Define the theorem statement
theorem red_stars_eq_35 : ∃ x : ℕ, red_star_frequency x = 0.5 ↔ x = 35 := sorry

end red_stars_eq_35_l175_175015


namespace equal_piles_l175_175896

theorem equal_piles (initial_rocks final_piles : ℕ) (moves : ℕ) (total_rocks : ℕ) (rocks_per_pile : ℕ) :
  initial_rocks = 36 →
  final_piles = 7 →
  moves = final_piles - 1 →
  total_rocks = initial_rocks + moves →
  rocks_per_pile = total_rocks / final_piles →
  rocks_per_pile = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_piles_l175_175896


namespace john_bought_notebooks_l175_175003

def pages_per_notebook : ℕ := 40
def pages_per_day : ℕ := 4
def total_days : ℕ := 50

theorem john_bought_notebooks : (pages_per_day * total_days) / pages_per_notebook = 5 :=
by
  sorry

end john_bought_notebooks_l175_175003


namespace plane_through_line_and_point_l175_175023

-- Definitions from the conditions
def line (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 3) / 4 ∧ (x - 1) / 2 = z / (-1)

def pointP1 : ℝ × ℝ × ℝ := (1, 5, 2)

-- Correct answer
def plane_eqn (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 2 * z + 1 = 0

-- The theorem to prove
theorem plane_through_line_and_point (x y z : ℝ) :
  line x y z → plane_eqn x y z := by
  sorry

end plane_through_line_and_point_l175_175023


namespace complement_of_A_in_U_l175_175742

def U : Set ℕ := {1,3,5,7,9}
def A : Set ℕ := {1,9}
def complement_U_A : Set ℕ := {3,5,7}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_in_U_l175_175742


namespace relationship_between_x_and_z_l175_175932

-- Definitions of the given conditions
variable {x y z : ℝ}

-- Statement of the theorem
theorem relationship_between_x_and_z (h1 : x = 1.027 * y) (h2 : y = 0.45 * z) : x = 0.46215 * z :=
by
  sorry

end relationship_between_x_and_z_l175_175932


namespace problem_l175_175497

-- Define the polynomial g(x) with given coefficients
def g (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x^2 + x + 8

-- Define the polynomial f(x) with given coefficients
def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^4 + x^3 + b * x^2 + 50 * x + c

-- Define the conditions
def conditions (a b c r : ℝ) : Prop :=
  ∃ roots : Finset ℝ, (∀ x ∈ roots, g x a = 0) ∧ (∀ x ∈ roots, f x a b c = 0) ∧ (roots.card = 3) ∧
  (8 - r = 50) ∧ (a - r = 1) ∧ (1 - a * r = b) ∧ (-8 * r = c)

-- Define the theorem to be proved
theorem problem (a b c r : ℝ) (h : conditions a b c r) : f 1 a b c = -1333 :=
by sorry

end problem_l175_175497


namespace range_of_m_l175_175651

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x >= (4 + m)) ∧ (x <= 3 * (x - 2) + 4) → (x ≥ 2)) →
  (-3 < m ∧ m <= -2) :=
sorry

end range_of_m_l175_175651


namespace range_of_k_l175_175189

theorem range_of_k 
  (k : ℝ) 
  (line_intersects_hyperbola : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) : 
  -Real.sqrt (15) / 3 < k ∧ k < Real.sqrt (15) / 3 := 
by
  sorry

end range_of_k_l175_175189


namespace jessica_deposit_fraction_l175_175266

theorem jessica_deposit_fraction (init_balance withdraw_amount final_balance : ℝ)
  (withdraw_fraction remaining_fraction deposit_fraction : ℝ) :
  remaining_fraction = withdraw_fraction - (2/5) → 
  init_balance * withdraw_fraction = init_balance - withdraw_amount →
  init_balance * remaining_fraction + deposit_fraction * (init_balance * remaining_fraction) = final_balance →
  init_balance = 500 →
  final_balance = 450 →
  withdraw_amount = 200 →
  remaining_fraction = (3/5) →
  deposit_fraction = 1/2 :=
by
  intros hr hw hrb hb hf hwamount hr_remain
  sorry

end jessica_deposit_fraction_l175_175266


namespace ball_box_distribution_l175_175577

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l175_175577


namespace probability_shaded_is_one_third_l175_175862

-- Define the total number of regions as a constant
def total_regions : ℕ := 12

-- Define the number of shaded regions as a constant
def shaded_regions : ℕ := 4

-- The probability that the tip of a spinner stopping in a shaded region
def probability_shaded : ℚ := shaded_regions / total_regions

-- Main theorem stating the probability calculation is correct
theorem probability_shaded_is_one_third : probability_shaded = 1 / 3 :=
by
  sorry

end probability_shaded_is_one_third_l175_175862


namespace largest_consecutive_odd_number_is_27_l175_175516

theorem largest_consecutive_odd_number_is_27 (a b c : ℤ) 
  (h1: a + b + c = 75)
  (h2: c - a = 6)
  (h3: b = a + 2)
  (h4: c = a + 4) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_is_27_l175_175516


namespace problem_solution_l175_175205

theorem problem_solution :
  (12345 * 5 + 23451 * 4 + 34512 * 3 + 45123 * 2 + 51234 * 1 = 400545) :=
by
  sorry

end problem_solution_l175_175205


namespace sum_remainder_l175_175313

theorem sum_remainder (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 13) 
  (h3 : c % 15 = 9) :
  (a + b + c) % 15 = 3 := 
by
  sorry

end sum_remainder_l175_175313


namespace sum_of_values_l175_175200

theorem sum_of_values (N : ℝ) (R : ℝ) (hN : N ≠ 0) (h_eq : N - 3 / N = R) :
  let N1 := (-R + Real.sqrt (R^2 + 12)) / 2
  let N2 := (-R - Real.sqrt (R^2 + 12)) / 2
  N1 + N2 = R :=
by
  sorry

end sum_of_values_l175_175200


namespace larger_number_l175_175558

theorem larger_number (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 :=
sorry

end larger_number_l175_175558


namespace quadratic_equation_roots_l175_175717

theorem quadratic_equation_roots (a b c : ℝ) : 
  (b ^ 6 > 4 * (a ^ 3) * (c ^ 3)) → (b ^ 10 > 4 * (a ^ 5) * (c ^ 5)) :=
by
  sorry

end quadratic_equation_roots_l175_175717


namespace sequence_a_2016_value_l175_175842

theorem sequence_a_2016_value (a : ℕ → ℕ) 
  (h1 : a 4 = 1)
  (h2 : a 11 = 9)
  (h3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15) :
  a 2016 = 5 :=
sorry

end sequence_a_2016_value_l175_175842


namespace product_grades_probabilities_l175_175204

theorem product_grades_probabilities (P_Q P_S : ℝ) (h1 : P_Q = 0.98) (h2 : P_S = 0.21) :
  P_Q - P_S = 0.77 ∧ 1 - P_Q = 0.02 :=
by
  sorry

end product_grades_probabilities_l175_175204


namespace find_angle_C_l175_175597

variable (A B C : ℝ)
variable (a b c : ℝ)

theorem find_angle_C (hA : A = 39) 
                     (h_condition : (a^2 - b^2)*(a^2 + a*c - b^2) = b^2 * c^2) : 
                     C = 115 :=
sorry

end find_angle_C_l175_175597


namespace perimeter_of_rectangle_l175_175491

theorem perimeter_of_rectangle (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 75) : 2 * l + 2 * b = 40 := 
by 
  sorry

end perimeter_of_rectangle_l175_175491


namespace geometric_series_sum_l175_175646

theorem geometric_series_sum : 
  let a := 6
  let r := - (2 / 5)
  let s := a / (1 - r)
  s = 30 / 7 :=
by
  let a := 6
  let r := -(2 / 5)
  let s := a / (1 - r)
  show s = 30 / 7
  sorry

end geometric_series_sum_l175_175646


namespace donation_student_amount_l175_175517

theorem donation_student_amount (a : ℕ) : 
  let total_amount := 3150
  let teachers_count := 5
  let donation_teachers := teachers_count * a 
  let donation_students := total_amount - donation_teachers
  donation_students = 3150 - 5 * a :=
by
  sorry

end donation_student_amount_l175_175517


namespace largest_n_divisibility_l175_175807

theorem largest_n_divisibility :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧
  (∀ m : ℕ, (m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 :=
by
  sorry

end largest_n_divisibility_l175_175807


namespace price_of_brand_y_pen_l175_175413

-- Definitions based on the conditions
def num_brand_x_pens : ℕ := 8
def price_per_brand_x_pen : ℝ := 4.0
def total_spent : ℝ := 40.0
def total_pens : ℕ := 12

-- price of brand Y that needs to be proven
def price_per_brand_y_pen : ℝ := 2.0

-- Proof statement
theorem price_of_brand_y_pen :
  let num_brand_y_pens := total_pens - num_brand_x_pens
  let spent_on_brand_x_pens := num_brand_x_pens * price_per_brand_x_pen
  let spent_on_brand_y_pens := total_spent - spent_on_brand_x_pens
  spent_on_brand_y_pens / num_brand_y_pens = price_per_brand_y_pen :=
by
  sorry

end price_of_brand_y_pen_l175_175413


namespace average_speeds_l175_175631

theorem average_speeds (x y : ℝ) (h1 : 4 * x + 5 * y = 98) (h2 : 4 * x = 5 * y - 2) : 
  x = 12 ∧ y = 10 :=
by sorry

end average_speeds_l175_175631


namespace determine_xyz_l175_175353

theorem determine_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 
  x * y * z = 23 / 3 := 
by { sorry }

end determine_xyz_l175_175353


namespace eval_expr_eq_zero_l175_175503

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end eval_expr_eq_zero_l175_175503


namespace remaining_dresses_pockets_count_l175_175560

-- Definitions translating each condition in the problem.
def total_dresses : Nat := 24
def dresses_with_pockets : Nat := total_dresses / 2
def dresses_with_two_pockets : Nat := dresses_with_pockets / 3
def total_pockets : Nat := 32

-- Question translated into a proof problem using Lean's logic.
theorem remaining_dresses_pockets_count :
  (total_pockets - (dresses_with_two_pockets * 2)) / (dresses_with_pockets - dresses_with_two_pockets) = 3 := by
  sorry

end remaining_dresses_pockets_count_l175_175560


namespace find_common_difference_l175_175443

variable {a : ℕ → ℤ} 
variable {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def problem_conditions (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 4 = 8 ∧ S 8 = 48

theorem find_common_difference :
  ∃ d, problem_conditions a S d ∧ is_arithmetic_sequence a d ∧ sum_of_first_n_terms a S ∧ d = 2 :=
by
  sorry

end find_common_difference_l175_175443


namespace nth_term_series_l175_175242

def a_n (n : ℕ) : ℝ :=
  if n % 2 = 1 then -4 else 7

theorem nth_term_series (n : ℕ) : a_n n = 1.5 + 5.5 * (-1) ^ n :=
by
  sorry

end nth_term_series_l175_175242


namespace perp_line_eq_l175_175696

theorem perp_line_eq (m : ℝ) (L1 : ∀ (x y : ℝ), m * x - m^2 * y = 1) (P : ℝ × ℝ) (P_def : P = (2, 1)) :
  ∃ d : ℝ, (∀ (x y : ℝ), x + y = d) ∧ P.fst + P.snd = d :=
by
  sorry

end perp_line_eq_l175_175696


namespace solution_set_l175_175433

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

variable {f : ℝ → ℝ}

-- Hypotheses
axiom odd_f : is_odd f
axiom increasing_f : is_increasing f
axiom f_of_neg_three : f (-3) = 0

-- Theorem statement
theorem solution_set (x : ℝ) : (x - 3) * f (x - 3) < 0 ↔ (0 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_l175_175433


namespace min_value_expr_l175_175270

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end min_value_expr_l175_175270


namespace digit_at_1286th_position_l175_175470

def naturally_written_sequence : ℕ → ℕ := sorry

theorem digit_at_1286th_position : naturally_written_sequence 1286 = 3 :=
sorry

end digit_at_1286th_position_l175_175470


namespace highest_score_is_174_l175_175790

theorem highest_score_is_174
  (avg_40_innings : ℝ)
  (highest_exceeds_lowest : ℝ)
  (avg_excl_two : ℝ)
  (total_runs_40 : ℝ)
  (total_runs_38 : ℝ)
  (sum_H_L : ℝ)
  (new_avg_38 : ℝ)
  (H : ℝ)
  (L : ℝ)
  (H_eq_L_plus_172 : H = L + 172)
  (total_runs_40_eq : total_runs_40 = 40 * avg_40_innings)
  (total_runs_38_eq : total_runs_38 = 38 * new_avg_38)
  (sum_H_L_eq : sum_H_L = total_runs_40 - total_runs_38)
  (new_avg_eq : new_avg_38 = avg_40_innings - 2)
  (sum_H_L_val : sum_H_L = 176)
  (avg_40_val : avg_40_innings = 50) :
  H = 174 :=
sorry

end highest_score_is_174_l175_175790


namespace largest_whole_number_solution_for_inequality_l175_175802

theorem largest_whole_number_solution_for_inequality :
  ∀ (x : ℕ), ((1 : ℝ) / 4 + (x : ℝ) / 5 < 2) → x ≤ 23 :=
by sorry

end largest_whole_number_solution_for_inequality_l175_175802


namespace system_of_equations_solve_l175_175576

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l175_175576


namespace fold_string_twice_l175_175044

theorem fold_string_twice (initial_length : ℕ) (half_folds : ℕ) (result_length : ℕ) 
  (h1 : initial_length = 12)
  (h2 : half_folds = 2)
  (h3 : result_length = initial_length / (2 ^ half_folds)) :
  result_length = 3 := 
by
  -- This is where the proof would go
  sorry

end fold_string_twice_l175_175044


namespace gcd_1248_585_l175_175843

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end gcd_1248_585_l175_175843


namespace range_of_m_l175_175201

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, x > 4 ↔ x > m) : m ≤ 4 :=
by {
  -- here we state the necessary assumptions and conclude the theorem
  -- detailed proof steps are not needed, hence sorry is used to skip the proof
  sorry
}

end range_of_m_l175_175201


namespace required_average_for_tickets_l175_175823

theorem required_average_for_tickets 
  (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)
  (num_children : ℝ) (total_score : ℝ) (average_score : ℝ) (S : ℝ)
  (h1 : june_score = 97) (h2 : patty_score = 85) (h3 : josh_score = 100) 
  (h4 : henry_score = 94) (h5 : num_children = 4) 
  (h6 : total_score = june_score + patty_score + josh_score + henry_score)
  (h7 : average_score = total_score / num_children) 
  (h8 : average_score = 94)
  : S ≤ 94 :=
sorry

end required_average_for_tickets_l175_175823


namespace pelican_fish_count_l175_175456

theorem pelican_fish_count 
(P K F : ℕ) 
(h1: K = P + 7) 
(h2: F = 3 * (P + K)) 
(h3: F = P + 86) : P = 13 := 
by 
  sorry

end pelican_fish_count_l175_175456


namespace fraction_order_l175_175475

theorem fraction_order:
  let frac1 := (21 : ℚ) / 17
  let frac2 := (22 : ℚ) / 19
  let frac3 := (18 : ℚ) / 15
  let frac4 := (20 : ℚ) / 16
  frac2 < frac3 ∧ frac3 < frac1 ∧ frac1 < frac4 := 
sorry

end fraction_order_l175_175475


namespace movie_theatre_total_seats_l175_175592

theorem movie_theatre_total_seats (A C : ℕ) 
  (hC : C = 188) 
  (hRevenue : 6 * A + 4 * C = 1124) 
  : A + C = 250 :=
by
  sorry

end movie_theatre_total_seats_l175_175592


namespace number_of_hens_l175_175185

theorem number_of_hens (H C : Nat) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := 
by
  sorry

end number_of_hens_l175_175185


namespace xy_sum_equal_two_or_minus_two_l175_175488

/-- 
Given the conditions |x| = 3, |y| = 5, and xy < 0, prove that x + y = 2 or x + y = -2. 
-/
theorem xy_sum_equal_two_or_minus_two (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) : x + y = 2 ∨ x + y = -2 := 
  sorry

end xy_sum_equal_two_or_minus_two_l175_175488


namespace ferris_wheel_capacity_l175_175140

theorem ferris_wheel_capacity 
  (num_seats : ℕ)
  (people_per_seat : ℕ)
  (h1 : num_seats = 4)
  (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end ferris_wheel_capacity_l175_175140


namespace range_of_x_l175_175147

theorem range_of_x (x : ℝ) (h : (x + 1) ^ 0 = 1) : x ≠ -1 :=
sorry

end range_of_x_l175_175147


namespace angle_complement_supplement_l175_175090

theorem angle_complement_supplement (x : ℝ) (h1 : 90 - x = (1 / 2) * (180 - x)) : x = 90 := by
  sorry

end angle_complement_supplement_l175_175090


namespace find_denomination_l175_175461

def denomination_of_bills (num_tumblers : ℕ) (cost_per_tumbler change num_bills amount_paid bill_denomination : ℤ) : Prop :=
  num_tumblers * cost_per_tumbler + change = amount_paid ∧
  amount_paid = num_bills * bill_denomination

theorem find_denomination :
  denomination_of_bills
    10    -- num_tumblers
    45    -- cost_per_tumbler
    50    -- change
    5     -- num_bills
    500   -- amount_paid
    100   -- bill_denomination
:=
by
  sorry

end find_denomination_l175_175461


namespace find_circle_center_l175_175594

noncomputable def midpoint_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_center (x y : ℝ) : Prop :=
  6 * x - 5 * y = midpoint_line 40 (-20) ∧ 3 * x + 2 * y = 0

theorem find_circle_center : circle_center (20 / 27) (-10 / 9) :=
by
  -- Here would go the proof steps, but we skip it
  sorry

end find_circle_center_l175_175594


namespace range_of_m_l175_175046

theorem range_of_m (a : ℝ) (h : a ≠ 0) (x1 x2 y1 y2 : ℝ) (m : ℝ)
  (hx1 : -2 < x1 ∧ x1 < 0) (hx2 : m < x2 ∧ x2 < m + 1)
  (h_on_parabola_A : y1 = a * x1^2 - 2 * a * x1 - 3)
  (h_on_parabola_B : y2 = a * x2^2 - 2 * a * x2 - 3)
  (h_diff_y : y1 ≠ y2) :
  (0 < m ∧ m ≤ 1) ∨ m ≥ 4 :=
sorry

end range_of_m_l175_175046


namespace remainder_div_19_l175_175423

theorem remainder_div_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
sorry

end remainder_div_19_l175_175423


namespace max_value_of_f_l175_175132

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ x, (f x = 1 / exp 1) ∧ (∀ y, f y ≤ f x) :=
by
  sorry

end max_value_of_f_l175_175132


namespace find_interest_rate_l175_175035

noncomputable def interest_rate (total_investment remaining_investment interest_earned part_interest : ℝ) : ℝ :=
  (interest_earned - part_interest) / remaining_investment

theorem find_interest_rate :
  let total_investment := 9000
  let invested_at_8_percent := 4000
  let total_interest := 770
  let interest_at_8_percent := invested_at_8_percent * 0.08
  let remaining_investment := total_investment - invested_at_8_percent
  let interest_from_remaining := total_interest - interest_at_8_percent
  interest_rate total_investment remaining_investment total_interest interest_at_8_percent = 0.09 :=
by
  sorry

end find_interest_rate_l175_175035


namespace find_x_minus_y_l175_175654

theorem find_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : y^2 = 16) (h3 : x + y > 0) : x - y = 1 ∨ x - y = 9 := 
by sorry

end find_x_minus_y_l175_175654


namespace not_minimum_on_l175_175906

noncomputable def f (x m : ℝ) : ℝ :=
  x * Real.exp x - (m / 2) * x ^ 2 - m * x

theorem not_minimum_on (m : ℝ) : 
  ¬ (∃ x ∈ Set.Icc 1 2, f x m = Real.exp 2 - 2 * m ∧ 
  ∀ y ∈ Set.Icc 1 2, f y m ≥ f x m) :=
sorry

end not_minimum_on_l175_175906


namespace function_satisfies_conditions_l175_175884

theorem function_satisfies_conditions :
  (∃ f : ℤ × ℤ → ℝ,
    (∀ x y z : ℤ, f (x, y) * f (y, z) * f (z, x) = 1) ∧
    (∀ x : ℤ, f (x + 1, x) = 2) ∧
    (∀ x y : ℤ, f (x, y) = 2 ^ (x - y))) :=
by
  sorry

end function_satisfies_conditions_l175_175884


namespace jerry_birthday_games_l175_175282

def jerry_original_games : ℕ := 7
def jerry_total_games_after_birthday : ℕ := 9
def games_jerry_got_for_birthday (original total : ℕ) : ℕ := total - original

theorem jerry_birthday_games :
  games_jerry_got_for_birthday jerry_original_games jerry_total_games_after_birthday = 2 := by
  sorry

end jerry_birthday_games_l175_175282


namespace will_net_calorie_intake_is_600_l175_175257

-- Given conditions translated into Lean definitions and assumptions
def breakfast_calories : ℕ := 900
def jogging_time_minutes : ℕ := 30
def calories_burned_per_minute : ℕ := 10

-- Proof statement in Lean
theorem will_net_calorie_intake_is_600 :
  breakfast_calories - (jogging_time_minutes * calories_burned_per_minute) = 600 :=
by
  sorry

end will_net_calorie_intake_is_600_l175_175257


namespace least_possible_value_of_y_l175_175685

theorem least_possible_value_of_y (x y z : ℤ) (hx : Even x) (hy : Odd y) (hz : Odd z) 
  (h1 : y - x > 5) (h2 : z - x ≥ 9) : y ≥ 7 :=
by {
  -- sorry allows us to skip the proof
  sorry
}

end least_possible_value_of_y_l175_175685


namespace number_of_valid_selections_l175_175098

theorem number_of_valid_selections : 
  ∃ combinations : Finset (Finset ℕ), 
    combinations = {
      {2, 6, 3, 5}, 
      {2, 6, 1, 7}, 
      {2, 4, 1, 5}, 
      {4, 1, 3}, 
      {6, 1, 5}, 
      {4, 6, 3, 7}, 
      {2, 4, 6, 5, 7}
    } ∧ combinations.card = 7 :=
by sorry

end number_of_valid_selections_l175_175098


namespace range_of_a_l175_175218

noncomputable def satisfies_condition (a : ℝ) : Prop :=
∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs ((1 / 2) * x^3 - a * x) ≤ 1

theorem range_of_a :
  {a : ℝ | satisfies_condition a} = {a : ℝ | - (1 / 2) ≤ a ∧ a ≤ (3 / 2)} :=
by
  sorry

end range_of_a_l175_175218


namespace canoe_downstream_speed_l175_175171

-- Definitions based on conditions
def upstream_speed : ℝ := 9  -- upspeed
def stream_speed : ℝ := 1.5  -- vspeed

-- Theorem to prove the downstream speed
theorem canoe_downstream_speed (V_c : ℝ) (V_d : ℝ) :
  (V_c - stream_speed = upstream_speed) →
  (V_d = V_c + stream_speed) →
  V_d = 12 := by 
  intro h1 h2
  sorry

end canoe_downstream_speed_l175_175171


namespace probability_of_perfect_square_sum_l175_175806

def two_dice_probability_of_perfect_square_sum : ℚ :=
  let totalOutcomes := 12 * 12
  let perfectSquareOutcomes := 3 + 8 + 9 -- ways to get sums 4, 9, and 16
  (perfectSquareOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_of_perfect_square_sum :
  two_dice_probability_of_perfect_square_sum = 5 / 36 :=
by
  sorry

end probability_of_perfect_square_sum_l175_175806


namespace range_of_a_plus_b_l175_175523

variable (a b : ℝ)
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (h : a + b + 1/a + 1/b = 5)

theorem range_of_a_plus_b : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end range_of_a_plus_b_l175_175523


namespace sandy_earnings_correct_l175_175452

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l175_175452


namespace power_function_at_point_l175_175410

theorem power_function_at_point (f : ℝ → ℝ) (h : ∃ α, ∀ x, f x = x^α) (hf : f 2 = 4) : f 3 = 9 :=
sorry

end power_function_at_point_l175_175410


namespace orthocenter_ABC_l175_175958

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨5, -1⟩
def B : Point2D := ⟨4, -8⟩
def C : Point2D := ⟨-4, -4⟩

def isOrthocenter (H : Point2D) (A B C : Point2D) : Prop := sorry  -- Define this properly according to the geometric properties in actual formalization.

theorem orthocenter_ABC : ∃ H : Point2D, isOrthocenter H A B C ∧ H = ⟨3, -5⟩ := 
by 
  sorry  -- Proof omitted

end orthocenter_ABC_l175_175958


namespace jack_leftover_money_l175_175812

theorem jack_leftover_money :
  let saved_money_base8 : ℕ := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 : ℕ := 1200
  saved_money_base8 - ticket_cost_base10 = 847 :=
by
  let saved_money_base8 := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 := 1200
  show saved_money_base8 - ticket_cost_base10 = 847
  sorry

end jack_leftover_money_l175_175812


namespace tan_diff_l175_175730

theorem tan_diff (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) : Real.tan (α - β) = 1/3 := 
sorry

end tan_diff_l175_175730


namespace graph_is_empty_l175_175202

theorem graph_is_empty :
  ¬∃ x y : ℝ, 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 :=
by
  -- the proof logic will go here
  sorry

end graph_is_empty_l175_175202


namespace smallest_n_between_76_and_100_l175_175233

theorem smallest_n_between_76_and_100 :
  ∃ (n : ℕ), (n > 1) ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 5 = 1) ∧ (76 < n) ∧ (n < 100) :=
sorry

end smallest_n_between_76_and_100_l175_175233


namespace pages_per_donut_l175_175291

def pages_written (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ) : ℕ :=
  let donuts := total_calories / calories_per_donut
  total_pages / donuts

theorem pages_per_donut (total_pages : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ): 
  total_pages = 12 → calories_per_donut = 150 → total_calories = 900 → pages_written total_pages calories_per_donut total_calories = 2 := by
  intros
  sorry

end pages_per_donut_l175_175291


namespace exist_five_natural_numbers_sum_and_product_equal_ten_l175_175668

theorem exist_five_natural_numbers_sum_and_product_equal_ten : 
  ∃ (n_1 n_2 n_3 n_4 n_5 : ℕ), 
  n_1 + n_2 + n_3 + n_4 + n_5 = 10 ∧ 
  n_1 * n_2 * n_3 * n_4 * n_5 = 10 := 
sorry

end exist_five_natural_numbers_sum_and_product_equal_ten_l175_175668


namespace students_difference_l175_175733

theorem students_difference 
  (C : ℕ → ℕ) 
  (hC1 : C 1 = 24) 
  (hC2 : ∀ n, C n.succ = C n - d)
  (h_total : C 1 + C 2 + C 3 + C 4 + C 5 = 100) :
  d = 2 :=
by sorry

end students_difference_l175_175733


namespace josiah_total_expenditure_l175_175521

noncomputable def cookies_per_day := 2
noncomputable def cost_per_cookie := 16
noncomputable def days_in_march := 31

theorem josiah_total_expenditure :
  (cookies_per_day * days_in_march * cost_per_cookie) = 992 :=
by sorry

end josiah_total_expenditure_l175_175521


namespace sequence_less_than_inverse_l175_175512

-- Define the sequence and conditions given in the problem
variables {a : ℕ → ℝ}
axiom positive_sequence (n : ℕ) : 0 < a n
axiom sequence_inequality (n : ℕ) : a n ^ 2 ≤ a n - a (n + 1)

theorem sequence_less_than_inverse (n : ℕ) : a n < 1 / n := 
sorry

end sequence_less_than_inverse_l175_175512


namespace triangle_area_qin_jiushao_l175_175263

theorem triangle_area_qin_jiushao (a b c : ℝ) (h1: a = 2) (h2: b = 3) (h3: c = Real.sqrt 13) :
  Real.sqrt ((1 / 4) * (a^2 * b^2 - (1 / 4) * (a^2 + b^2 - c^2)^2)) = 3 :=
by
  -- Hypotheses
  rw [h1, h2, h3]
  sorry

end triangle_area_qin_jiushao_l175_175263


namespace teams_in_BIG_M_l175_175449

theorem teams_in_BIG_M (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end teams_in_BIG_M_l175_175449


namespace domain_of_function_l175_175531

theorem domain_of_function :
  { x : ℝ | -2 ≤ x ∧ x < 4 } = { x : ℝ | (x + 2 ≥ 0) ∧ (4 - x > 0) } :=
by
  sorry

end domain_of_function_l175_175531


namespace sum_min_max_x_y_l175_175142

theorem sum_min_max_x_y (x y : ℕ) (h : 6 * x + 7 * y = 2012): 288 + 335 = 623 :=
by
  sorry

end sum_min_max_x_y_l175_175142


namespace find_number_l175_175776

def is_three_digit_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
  n = 100 * x + 10 * y + z ∧ (100 * x + 10 * y + z) / 11 = x^2 + y^2 + z^2

theorem find_number : ∃ n : ℕ, is_three_digit_number n ∧ n = 550 :=
sorry

end find_number_l175_175776


namespace jessica_purchase_cost_l175_175774

noncomputable def c_toy : Real := 10.22
noncomputable def c_cage : Real := 11.73
noncomputable def c_total : Real := c_toy + c_cage

theorem jessica_purchase_cost : c_total = 21.95 :=
by
  sorry

end jessica_purchase_cost_l175_175774


namespace geometric_sequence_second_term_value_l175_175677

theorem geometric_sequence_second_term_value
  (a : ℝ) 
  (r : ℝ) 
  (h1 : 30 * r = a) 
  (h2 : a * r = 7 / 4) 
  (h3 : 0 < a) : 
  a = 7.5 := 
sorry

end geometric_sequence_second_term_value_l175_175677


namespace dvds_bought_online_l175_175000

theorem dvds_bought_online (total_dvds : ℕ) (store_dvds : ℕ) (online_dvds : ℕ) :
  total_dvds = 10 → store_dvds = 8 → online_dvds = total_dvds - store_dvds → online_dvds = 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvds_bought_online_l175_175000


namespace find_green_pepper_weight_l175_175161

variable (weight_red_peppers : ℝ) (total_weight_peppers : ℝ)

theorem find_green_pepper_weight 
    (h1 : weight_red_peppers = 0.33) 
    (h2 : total_weight_peppers = 0.66) 
    : total_weight_peppers - weight_red_peppers = 0.33 := 
by sorry

end find_green_pepper_weight_l175_175161


namespace cube_root_of_64_l175_175032

theorem cube_root_of_64 : ∃ x : ℝ, x^3 = 64 ∧ x = 4 :=
by
  sorry

end cube_root_of_64_l175_175032


namespace min_value_of_expression_l175_175051

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 + x * y = 315) :
  ∃ m : ℝ, m = x^2 + y^2 - x * y ∧ m ≥ 105 :=
by
  sorry

end min_value_of_expression_l175_175051


namespace target_run_correct_l175_175966

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def run_rate_remaining_22_overs : ℝ := 11.363636363636363
def overs_remaining_22 : ℝ := 22

-- Initialize the target run calculation using the given conditions
def runs_first_10_overs := overs_first_10 * run_rate_first_10_overs
def runs_remaining_22_overs := overs_remaining_22 * run_rate_remaining_22_overs
def target_run := runs_first_10_overs + runs_remaining_22_overs 

-- The goal is to prove that the target run is 282
theorem target_run_correct : target_run = 282 := by
  sorry  -- The proof is not required as per the instructions.

end target_run_correct_l175_175966


namespace farmland_acres_l175_175327

theorem farmland_acres (x y : ℝ) 
  (h1 : x + y = 100) 
  (h2 : 300 * x + (500 / 7) * y = 10000) : 
  true :=
sorry

end farmland_acres_l175_175327


namespace trig_expression_equality_l175_175635

theorem trig_expression_equality :
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  2 * tan_60 + tan_45 - 4 * cos_30 = 1 := by
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  sorry

end trig_expression_equality_l175_175635


namespace total_cost_proof_l175_175968

def sandwich_cost : ℝ := 2.49
def soda_cost : ℝ := 1.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 12.46

theorem total_cost_proof : (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = total_cost :=
by
  sorry

end total_cost_proof_l175_175968


namespace find_a_plus_c_l175_175563

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  (b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B) ∧
  (b = 2) ∧
  ((1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 2) / 2)

theorem find_a_plus_c {A B C a b c : ℝ} (h : triangle_ABC A B C a b c) :
  a + c = 4 :=
by
  rcases h with ⟨hc1, hc2, hc3⟩
  sorry

end find_a_plus_c_l175_175563


namespace captain_age_is_24_l175_175113

theorem captain_age_is_24 (C W : ℕ) 
  (hW : W = C + 7)
  (h_total_team_age : 23 * 11 = 253)
  (h_total_9_players_age : 22 * 9 = 198)
  (h_team_age_equation : 253 = 198 + C + W)
  : C = 24 :=
sorry

end captain_age_is_24_l175_175113


namespace oranges_bought_l175_175180

theorem oranges_bought (total_cost : ℝ) 
  (selling_price_per_orange : ℝ) 
  (profit_per_orange : ℝ) 
  (cost_price_per_orange : ℝ) 
  (h1 : total_cost = 12.50)
  (h2 : selling_price_per_orange = 0.60)
  (h3 : profit_per_orange = 0.10)
  (h4 : cost_price_per_orange = selling_price_per_orange - profit_per_orange) :
  (total_cost / cost_price_per_orange) = 25 := 
by
  sorry

end oranges_bought_l175_175180


namespace nicole_answers_correctly_l175_175628

theorem nicole_answers_correctly :
  ∀ (C K N : ℕ), C = 17 → K = C + 8 → N = K - 3 → N = 22 :=
by
  intros C K N hC hK hN
  sorry

end nicole_answers_correctly_l175_175628


namespace blue_line_length_l175_175752

theorem blue_line_length (w b : ℝ) (h1 : w = 7.666666666666667) (h2 : w = b + 4.333333333333333) :
  b = 3.333333333333334 :=
by sorry

end blue_line_length_l175_175752


namespace total_soccer_balls_l175_175480

theorem total_soccer_balls (boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ) 
  (h1 : boxes = 10) (h2 : packages_per_box = 8) (h3 : balls_per_package = 13) : 
  (boxes * packages_per_box * balls_per_package = 1040) :=
by 
  sorry

end total_soccer_balls_l175_175480


namespace older_brother_catches_up_l175_175964

theorem older_brother_catches_up (D : ℝ) (t : ℝ) :
  let vy := D / 25
  let vo := D / 15
  let time := 20
  15 * time = 25 * (time - 8) → (15 * time = 25 * (time - 8) → t = 20)
:= by
  sorry

end older_brother_catches_up_l175_175964


namespace quadratic_roots_range_l175_175451

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x ≠ y ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 := 
sorry

end quadratic_roots_range_l175_175451


namespace cabbage_production_l175_175698

theorem cabbage_production (x y : ℕ) 
  (h1 : y^2 - x^2 = 127) 
  (h2 : y - x = 1) 
  (h3 : 2 * y = 128) : y^2 = 4096 := by
  sorry

end cabbage_production_l175_175698


namespace gallons_bought_l175_175378

variable (total_needed : ℕ) (existing_paint : ℕ) (needed_more : ℕ)

theorem gallons_bought (H : total_needed = 70) (H1 : existing_paint = 36) (H2 : needed_more = 11) : 
  total_needed - existing_paint - needed_more = 23 := 
sorry

end gallons_bought_l175_175378


namespace greatest_value_2q_sub_r_l175_175392

theorem greatest_value_2q_sub_r : 
  ∃ (q r : ℕ), 965 = 22 * q + r ∧ 2 * q - r = 67 := 
by 
  sorry

end greatest_value_2q_sub_r_l175_175392


namespace part_time_employees_l175_175852

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : full_time_employees = 63093) 
  (h3 : total_employees = full_time_employees + part_time_employees) : 
  part_time_employees = 2041 :=
by 
  sorry

end part_time_employees_l175_175852


namespace passing_grade_fraction_l175_175217

variables (students : ℕ) -- total number of students in Mrs. Susna's class

-- Conditions
def fraction_A : ℚ := 1/4
def fraction_B : ℚ := 1/2
def fraction_C : ℚ := 1/8
def fraction_D : ℚ := 1/12
def fraction_F : ℚ := 1/24

-- Prove the fraction of students getting a passing grade (C or higher) is 7/8
theorem passing_grade_fraction : 
  fraction_A + fraction_B + fraction_C = 7/8 :=
by
  sorry

end passing_grade_fraction_l175_175217


namespace mary_total_zoom_time_l175_175939

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end mary_total_zoom_time_l175_175939


namespace random_phenomenon_l175_175994

def is_certain_event (P : Prop) : Prop := ∀ h : P, true

def is_random_event (P : Prop) : Prop := ¬is_certain_event P

def scenario1 : Prop := ∀ pressure temperature : ℝ, (pressure = 101325) → (temperature = 100) → true
-- Under standard atmospheric pressure, water heated to 100°C will boil

def scenario2 : Prop := ∃ time : ℝ, true
-- Encountering a red light at a crossroads (which happens at random times)

def scenario3 (a b : ℝ) : Prop := true
-- For a rectangle with length and width a and b respectively, its area is a * b

def scenario4 : Prop := ∀ a b : ℝ, ∃ x : ℝ, a * x + b = 0
-- A linear equation with real coefficients always has one real root

theorem random_phenomenon : is_random_event scenario2 :=
by
  sorry

end random_phenomenon_l175_175994


namespace sum_of_prime_h_l175_175754

def h (n : ℕ) := n^4 - 380 * n^2 + 600

theorem sum_of_prime_h (S : Finset ℕ) (hS : S = { n | Nat.Prime (h n) }) :
  S.sum h = 0 :=
by
  sorry

end sum_of_prime_h_l175_175754


namespace find_x_l175_175050

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 682.50) : x = 210 :=
by
  sorry

end find_x_l175_175050


namespace length_of_second_train_l175_175117

theorem length_of_second_train (speed1 speed2 : ℝ) (length1 time : ℝ) (h1 : speed1 = 60) (h2 : speed2 = 40) 
  (h3 : length1 = 450) (h4 : time = 26.99784017278618) :
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  let length2 := total_distance - length1
  length2 = 300 :=
by
  sorry

end length_of_second_train_l175_175117


namespace island_perimeter_l175_175969

-- Defining the properties of the island
def width : ℕ := 4
def length : ℕ := 7

-- The main theorem stating the condition to be proved
theorem island_perimeter : 2 * (length + width) = 22 := by
  sorry

end island_perimeter_l175_175969


namespace numeral_diff_local_face_value_l175_175588

theorem numeral_diff_local_face_value (P : ℕ) :
  7 * (10 ^ P - 1) = 693 → P = 2 ∧ (N = 700) :=
by
  intro h
  -- The actual proof is not required hence we insert sorry
  sorry

end numeral_diff_local_face_value_l175_175588


namespace square_side_length_l175_175935

theorem square_side_length(area_sq_cm : ℕ) (h : area_sq_cm = 361) : ∃ side_length : ℕ, side_length ^ 2 = area_sq_cm ∧ side_length = 19 := 
by 
  use 19
  sorry

end square_side_length_l175_175935


namespace area_of_square_not_covered_by_circles_l175_175914

theorem area_of_square_not_covered_by_circles :
  let side : ℝ := 10
  let radius : ℝ := 5
  (side^2 - 4 * (π * radius^2) + 4 * (π * (radius^2) / 2)) = (100 - 50 * π) := 
sorry

end area_of_square_not_covered_by_circles_l175_175914


namespace jason_less_than_jenny_l175_175467

-- Definition of conditions

def grade_Jenny : ℕ := 95
def grade_Bob : ℕ := 35
def grade_Jason : ℕ := 2 * grade_Bob -- Bob's grade is half of Jason's grade

-- The theorem we need to prove
theorem jason_less_than_jenny : grade_Jenny - grade_Jason = 25 :=
by
  sorry

end jason_less_than_jenny_l175_175467


namespace lines_are_coplanar_l175_175094

/- Define the parameterized lines -/
def L1 (s : ℝ) (k : ℝ) : ℝ × ℝ × ℝ := (1 + 2 * s, 4 - k * s, 2 + 2 * k * s)
def L2 (t : ℝ) : ℝ × ℝ × ℝ := (2 + t, 7 + 3 * t, 1 - 2 * t)

/- Prove that k = 0 ensures the lines are coplanar -/
theorem lines_are_coplanar (k : ℝ) : k = 0 ↔ 
  ∃ (s t : ℝ), L1 s k = L2 t :=
by {
  sorry
}

end lines_are_coplanar_l175_175094


namespace base_conversion_addition_correct_l175_175119

theorem base_conversion_addition_correct :
  let A := 10
  let C := 12
  let n13 := 3 * 13^2 + 7 * 13^1 + 6
  let n14 := 4 * 14^2 + A * 14^1 + C
  n13 + n14 = 1540 := by
    let A := 10
    let C := 12
    let n13 := 3 * 13^2 + 7 * 13^1 + 6
    let n14 := 4 * 14^2 + A * 14^1 + C
    let sum := n13 + n14
    have h1 : n13 = 604 := by sorry
    have h2 : n14 = 936 := by sorry
    have h3 : sum = 1540 := by sorry
    exact h3

end base_conversion_addition_correct_l175_175119


namespace paige_team_total_players_l175_175346

theorem paige_team_total_players 
    (total_points : ℕ)
    (paige_points : ℕ)
    (other_points_per_player : ℕ)
    (other_players : ℕ) :
    total_points = paige_points + other_points_per_player * other_players →
    (other_players + 1) = 6 :=
by
  intros h
  sorry

end paige_team_total_players_l175_175346


namespace expected_value_of_difference_is_4_point_5_l175_175238

noncomputable def expected_value_difference : ℚ :=
  (2 * 6 / 56 + 3 * 10 / 56 + 4 * 12 / 56 + 5 * 12 / 56 + 6 * 10 / 56 + 7 * 6 / 56)

theorem expected_value_of_difference_is_4_point_5 :
  expected_value_difference = 4.5 := sorry

end expected_value_of_difference_is_4_point_5_l175_175238


namespace equation_of_plane_l175_175095

/--
The equation of the plane passing through the points (2, -2, 2) and (0, 0, 2),
and which is perpendicular to the plane 2x - y + 4z = 8, is given by:
Ax + By + Cz + D = 0 where A, B, C, D are integers such that A > 0 and gcd(|A|,|B|,|C|,|D|) = 1.
-/
theorem equation_of_plane :
  ∃ (A B C D : ℤ),
    A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
    (∀ x y z : ℤ, A * x + B * y + C * z + D = 0 ↔ x + y = 0) :=
sorry

end equation_of_plane_l175_175095


namespace square_area_l175_175116

theorem square_area (side : ℕ) (h : side = 19) : side * side = 361 := by
  sorry

end square_area_l175_175116


namespace rectangle_diagonal_l175_175241

theorem rectangle_diagonal (k : ℕ) (h1 : 2 * (5 * k + 4 * k) = 72) : 
  (Real.sqrt ((5 * k) ^ 2 + (4 * k) ^ 2)) = Real.sqrt 656 :=
by
  sorry

end rectangle_diagonal_l175_175241


namespace fraction_simplification_l175_175881

theorem fraction_simplification :
  ∃ (p q : ℕ), p = 2021 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (1011 / 1010) - (1010 / 1011) = (p : ℚ) / q := 
sorry

end fraction_simplification_l175_175881


namespace elements_of_set_A_l175_175630

theorem elements_of_set_A (A : Set ℝ) (h₁ : ∀ a : ℝ, a ∈ A → (1 + a) / (1 - a) ∈ A)
(h₂ : -3 ∈ A) : A = {-3, -1/2, 1/3, 2} := by
  sorry

end elements_of_set_A_l175_175630


namespace carpet_size_l175_175299

def length := 5
def width := 2
def area := length * width

theorem carpet_size : area = 10 := by
  sorry

end carpet_size_l175_175299


namespace simplify_expression_l175_175169

theorem simplify_expression (x : ℝ) (h : x = 9) : 
  ((x^9 - 27 * x^6 + 729) / (x^6 - 27) = 730 + 1 / 26) :=
by {
 sorry
}

end simplify_expression_l175_175169


namespace expected_value_is_100_cents_l175_175079

-- Definitions for the values of the coins
def value_quarter : ℕ := 25
def value_half_dollar : ℕ := 50
def value_dollar : ℕ := 100

-- Define the total value of all coins
def total_value : ℕ := 2 * value_quarter + value_half_dollar + value_dollar

-- Probability of heads for a single coin
def p_heads : ℚ := 1 / 2

-- Expected value calculation
def expected_value : ℚ := p_heads * ↑total_value

-- The theorem we need to prove
theorem expected_value_is_100_cents : expected_value = 100 :=
by
  -- This is where the proof would go, but we are omitting it
  sorry

end expected_value_is_100_cents_l175_175079


namespace percentage_of_acid_is_18_18_percent_l175_175648

noncomputable def percentage_of_acid_in_original_mixture
  (a w : ℝ) (h1 : (a + 1) / (a + w + 1) = 1 / 4) (h2 : (a + 1) / (a + w + 2) = 1 / 5) : ℝ :=
  a / (a + w) 

theorem percentage_of_acid_is_18_18_percent :
  ∃ (a w : ℝ), (a + 1) / (a + w + 1) = 1 / 4 ∧ (a + 1) / (a + w + 2) = 1 / 5 ∧ percentage_of_acid_in_original_mixture a w (by sorry) (by sorry) = 18.18 := by
  sorry

end percentage_of_acid_is_18_18_percent_l175_175648


namespace ellipse_parabola_common_point_l175_175438

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔  -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end ellipse_parabola_common_point_l175_175438


namespace find_k_l175_175358

theorem find_k (x y k : ℝ) (hx1 : x - 4 * y + 3 ≤ 0) (hx2 : 3 * x + 5 * y - 25 ≤ 0) (hx3 : x ≥ 1)
  (hmax : ∃ (z : ℝ), z = 12 ∧ z = k * x + y) (hmin : ∃ (z : ℝ), z = 3 ∧ z = k * x + y) :
  k = 2 :=
sorry

end find_k_l175_175358


namespace determine_n_l175_175088

theorem determine_n (n : ℕ) (h1 : 0 < n) 
(h2 : ∃ (sols : Finset (ℕ × ℕ × ℕ)), 
  (∀ (x y z : ℕ), (x, y, z) ∈ sols ↔ 3 * x + 2 * y + z = n ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ sols.card = 55) : 
  n = 36 := 
by 
  sorry 

end determine_n_l175_175088


namespace ab_cd_eq_zero_l175_175940

theorem ab_cd_eq_zero  
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : ad - bc = -1) :
  ab + cd = 0 :=
by
  sorry

end ab_cd_eq_zero_l175_175940


namespace piecewise_function_continuity_l175_175844

theorem piecewise_function_continuity :
  (∀ x, if x > (3 : ℝ) 
        then 2 * (a : ℝ) * x + 4 = (x : ℝ) ^ 2 - 1
        else if x < -1 
        then 3 * (x : ℝ) - (c : ℝ) = (x : ℝ) ^ 2 - 1
        else (x : ℝ) ^ 2 - 1 = (x : ℝ) ^ 2 - 1) →
  a = 2 / 3 →
  c = -3 →
  a + c = -7 / 3 :=
by
  intros h ha hc
  simp [ha, hc]
  sorry

end piecewise_function_continuity_l175_175844


namespace distinct_meals_count_l175_175509

def entries : ℕ := 3
def drinks : ℕ := 3
def desserts : ℕ := 3

theorem distinct_meals_count : entries * drinks * desserts = 27 :=
by
  -- sorry for skipping the proof
  sorry

end distinct_meals_count_l175_175509


namespace smallest_possible_n_l175_175687

-- Definitions needed for the problem
variable (x n : ℕ) (hpos : 0 < x)
variable (m : ℕ) (hm : m = 72)

-- The conditions as already stated
def gcd_cond := Nat.gcd 72 n = x + 8
def lcm_cond := Nat.lcm 72 n = x * (x + 8)

-- The proof statement
theorem smallest_possible_n (h_gcd : gcd_cond x n) (h_lcm : lcm_cond x n) : n = 8 :=
by 
  -- Intuitively outline the proof
  sorry

end smallest_possible_n_l175_175687


namespace evaluate_polynomial_l175_175430

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end evaluate_polynomial_l175_175430


namespace geometric_sequence_sum_reciprocal_ratio_l175_175779

theorem geometric_sequence_sum_reciprocal_ratio
  (a : ℚ) (r : ℚ) (n : ℕ) (S S' : ℚ)
  (h1 : a = 1/4)
  (h2 : r = 2)
  (h3 : S = a * (1 - r^n) / (1 - r))
  (h4 : S' = (1/a) * (1 - (1/r)^n) / (1 - 1/r)) :
  S / S' = 32 :=
sorry

end geometric_sequence_sum_reciprocal_ratio_l175_175779


namespace triangle_side_ratio_range_l175_175337

theorem triangle_side_ratio_range (A B C a b c : ℝ) (h1 : A + 4 * B = 180) (h2 : C = 3 * B) (h3 : 0 < B ∧ B < 45) 
  (h4 : a / b = Real.sin (4 * B) / Real.sin B) : 
  1 < a / b ∧ a / b < 3 := 
sorry

end triangle_side_ratio_range_l175_175337


namespace solution_set_linear_inequalities_l175_175851

theorem solution_set_linear_inequalities (x : ℝ) 
  (h1 : x - 2 > 1) 
  (h2 : x < 4) : 
  3 < x ∧ x < 4 :=
by
  sorry

end solution_set_linear_inequalities_l175_175851


namespace monotonicity_of_f_range_of_a_l175_175248

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.log x - x^2 + a

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f a x ≤ f a (x - 1)) ∧ 
           (a > 0 → ((x < Real.sqrt a → f a x ≤ f a (x + 1)) ∨ 
                     (x > Real.sqrt a → f a x ≥ f a (x - 1))))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 1) := sorry

end monotonicity_of_f_range_of_a_l175_175248


namespace problem1_l175_175861

theorem problem1 (x y : ℝ) (h1 : 2^(x + y) = x + 7) (h2 : x + y = 3) : (x = 1 ∧ y = 2) :=
by
  sorry

end problem1_l175_175861


namespace surveyed_households_count_l175_175786

theorem surveyed_households_count 
  (neither : ℕ) (only_R : ℕ) (both_B : ℕ) (both : ℕ) (h_main : Ξ)
  (H1 : neither = 80)
  (H2 : only_R = 60)
  (H3 : both = 40)
  (H4 : both_B = 3 * both) : 
  neither + only_R + both_B + both = 300 :=
by
  sorry

end surveyed_households_count_l175_175786


namespace total_eggs_l175_175483

theorem total_eggs (students : ℕ) (eggs_per_student : ℕ) (h1 : students = 7) (h2 : eggs_per_student = 8) :
  students * eggs_per_student = 56 :=
by
  sorry

end total_eggs_l175_175483


namespace area_increase_correct_l175_175268

-- Define the dimensions of the rectangular garden
def rect_length : ℕ := 60
def rect_width : ℕ := 20

-- Calculate the area of the rectangular garden
def area_rect : ℕ := rect_length * rect_width

-- Calculate the perimeter of the rectangular garden
def perimeter_rect : ℕ := 2 * (rect_length + rect_width)

-- Calculate the side length of the square garden using the same perimeter
def side_square : ℕ := perimeter_rect / 4

-- Calculate the area of the square garden
def area_square : ℕ := side_square * side_square

-- Calculate the increase in area
def area_increase : ℕ := area_square - area_rect

-- The statement to be proven in Lean 4
theorem area_increase_correct : area_increase = 400 := by
  sorry

end area_increase_correct_l175_175268


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l175_175127

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l175_175127


namespace proof_a_eq_b_pow_n_l175_175223

theorem proof_a_eq_b_pow_n
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n := 
by sorry

end proof_a_eq_b_pow_n_l175_175223


namespace bill_milk_problem_l175_175471

theorem bill_milk_problem 
  (M : ℚ) 
  (sour_cream_milk : ℚ := M / 4)
  (butter_milk : ℚ := M / 4)
  (whole_milk : ℚ := M / 2)
  (sour_cream_gallons : ℚ := sour_cream_milk / 2)
  (butter_gallons : ℚ := butter_milk / 4)
  (butter_revenue : ℚ := butter_gallons * 5)
  (sour_cream_revenue : ℚ := sour_cream_gallons * 6)
  (whole_milk_revenue : ℚ := whole_milk * 3)
  (total_revenue : ℚ := butter_revenue + sour_cream_revenue + whole_milk_revenue)
  (h : total_revenue = 41) :
  M = 16 :=
by
  sorry

end bill_milk_problem_l175_175471


namespace three_pow_y_plus_two_l175_175737

theorem three_pow_y_plus_two (y : ℕ) (h : 3^y = 81) : 3^(y+2) = 729 := sorry

end three_pow_y_plus_two_l175_175737


namespace unique_pair_a_b_l175_175411

open Complex

theorem unique_pair_a_b :
  ∃! (a b : ℂ), a^4 * b^3 = 1 ∧ a^6 * b^7 = 1 := by
  sorry

end unique_pair_a_b_l175_175411


namespace binom_15_4_l175_175701

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end binom_15_4_l175_175701


namespace percentage_comedies_l175_175155

theorem percentage_comedies (a : ℕ) (d c T : ℕ) 
  (h1 : d = 5 * a) 
  (h2 : c = 10 * a) 
  (h3 : T = c + d + a) : 
  (c : ℝ) / T * 100 = 62.5 := 
by 
  sorry

end percentage_comedies_l175_175155


namespace mutually_exclusive_event_is_D_l175_175983

namespace Problem

def event_A (n : ℕ) (defective : ℕ) : Prop := defective ≥ 2
def mutually_exclusive_event (n : ℕ) : Prop := (∀ (defective : ℕ), defective ≤ 1) ↔ (∀ (defective : ℕ), defective ≥ 2 → false)

theorem mutually_exclusive_event_is_D (n : ℕ) : mutually_exclusive_event n := 
by 
  sorry

end Problem

end mutually_exclusive_event_is_D_l175_175983


namespace min_f_triangle_sides_l175_175610

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x ^ 2, Real.sqrt 3)
  let b := (1, Real.sin (2 * x))
  (a.1 * b.1 + a.2 * b.2) - 2

theorem min_f (x : ℝ) (h1 : -Real.pi / 6 ≤ x) (h2 : x ≤ Real.pi / 3) :
  ∃ x₀, f x₀ = -2 ∧ ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x ≥ -2 :=
  sorry

theorem triangle_sides (a b C : ℝ) (h1 : f C = 1) (h2 : C = Real.pi / 6)
  (h3 : 1 = 1) (h4 : a * b = 2 * Real.sqrt 3) (h5 : a > b) :
  a = 2 ∧ b = Real.sqrt 3 :=
  sorry

end min_f_triangle_sides_l175_175610


namespace regular_polygon_sides_l175_175412

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end regular_polygon_sides_l175_175412


namespace wall_length_l175_175028

theorem wall_length (mirror_side length width : ℝ) (h_mirror : mirror_side = 21) (h_width : width = 28) 
  (h_area_relation : (mirror_side * mirror_side) * 2 = width * length) : length = 31.5 :=
by
  -- here you start the proof, but it's not required for the statement
  sorry

end wall_length_l175_175028


namespace sum_of_coefficients_of_poly_l175_175833

-- Define the polynomial
def poly (x y : ℕ) := (2 * x + 3 * y) ^ 12

-- Define the sum of coefficients
def sum_of_coefficients := poly 1 1

-- The theorem stating the result
theorem sum_of_coefficients_of_poly : sum_of_coefficients = 244140625 :=
by
  -- Proof is skipped
  sorry

end sum_of_coefficients_of_poly_l175_175833


namespace radius_of_circumscribed_circle_of_right_triangle_l175_175944

theorem radius_of_circumscribed_circle_of_right_triangle 
  (a b c : ℝ)
  (h_area : (1 / 2) * a * b = 10)
  (h_inradius : (a + b - c) / 2 = 1)
  (h_hypotenuse : c = Real.sqrt (a^2 + b^2)) :
  c / 2 = 4.5 := 
sorry

end radius_of_circumscribed_circle_of_right_triangle_l175_175944


namespace montana_more_than_ohio_l175_175891

-- Define the total number of combinations for Ohio and Montana
def ohio_combinations : ℕ := 26^4 * 10^3
def montana_combinations : ℕ := 26^5 * 10^2

-- The total number of combinations from both states
def ohio_total : ℕ := ohio_combinations
def montana_total : ℕ := montana_combinations

-- Prove the difference
theorem montana_more_than_ohio : montana_total - ohio_total = 731161600 := by
  sorry

end montana_more_than_ohio_l175_175891


namespace inequality_negatives_l175_175835

theorem inequality_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (b / a) < 1 :=
by
  sorry

end inequality_negatives_l175_175835


namespace transform_map_ABCD_to_A_l175_175593

structure Point :=
(x : ℤ)
(y : ℤ)

structure Rectangle :=
(A : Point)
(B : Point)
(C : Point)
(D : Point)

def transform180 (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

def rect_transform180 (rect : Rectangle) : Rectangle :=
  { A := transform180 rect.A,
    B := transform180 rect.B,
    C := transform180 rect.C,
    D := transform180 rect.D }

def ABCD := Rectangle.mk ⟨-3, 2⟩ ⟨-1, 2⟩ ⟨-1, 5⟩ ⟨-3, 5⟩
def A'B'C'D' := Rectangle.mk ⟨3, -2⟩ ⟨1, -2⟩ ⟨1, -5⟩ ⟨3, -5⟩

theorem transform_map_ABCD_to_A'B'C'D' :
  rect_transform180 ABCD = A'B'C'D' :=
by
  -- This is where the proof would go.
  sorry

end transform_map_ABCD_to_A_l175_175593


namespace intervals_of_monotonicity_l175_175543

noncomputable def y (x : ℝ) : ℝ := 2 ^ (x^2 - 2*x + 4)

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x > 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ < y x₂)) ∧
  (∀ x : ℝ, x < 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ > y x₂)) :=
by
  sorry

end intervals_of_monotonicity_l175_175543


namespace convert_point_cylindrical_to_rectangular_l175_175931

noncomputable def cylindrical_to_rectangular_coordinates (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_point_cylindrical_to_rectangular :
  cylindrical_to_rectangular_coordinates 6 (5 * Real.pi / 3) (-3) = (3, -3 * Real.sqrt 3, -3) :=
by
  sorry

end convert_point_cylindrical_to_rectangular_l175_175931


namespace num_ordered_pairs_of_squares_diff_by_144_l175_175209

theorem num_ordered_pairs_of_squares_diff_by_144 :
  ∃ (p : Finset (ℕ × ℕ)), p.card = 4 ∧ ∀ (a b : ℕ), (a, b) ∈ p → a ≥ b ∧ a^2 - b^2 = 144 := by
  sorry

end num_ordered_pairs_of_squares_diff_by_144_l175_175209


namespace min_visible_sum_of_4x4x4_cube_l175_175444

theorem min_visible_sum_of_4x4x4_cube (dice_capacity : ℕ) (opposite_sum : ℕ) (corner_dice edge_dice center_face_dice innermost_dice : ℕ) : 
  dice_capacity = 64 ∧ 
  opposite_sum = 7 ∧ 
  corner_dice = 8 ∧ 
  edge_dice = 24 ∧ 
  center_face_dice = 24 ∧ 
  innermost_dice = 8 → 
  ∃ min_sum, min_sum = 144 := by
  sorry

end min_visible_sum_of_4x4x4_cube_l175_175444


namespace problem_l175_175011

theorem problem (n : ℝ) (h : (n - 2009)^2 + (2008 - n)^2 = 1) : (n - 2009) * (2008 - n) = 0 := 
by
  sorry

end problem_l175_175011


namespace find_least_skilled_painter_l175_175115

-- Define the genders
inductive Gender
| Male
| Female

-- Define the family members
inductive Member
| Grandmother
| Niece
| Nephew
| Granddaughter

-- Define a structure to hold the properties of each family member
structure Properties where
  gender : Gender
  age : Nat
  isTwin : Bool

-- Assume the properties of each family member as given
def grandmother : Properties := { gender := Gender.Female, age := 70, isTwin := false }
def niece : Properties := { gender := Gender.Female, age := 20, isTwin := false }
def nephew : Properties := { gender := Gender.Male, age := 20, isTwin := true }
def granddaughter : Properties := { gender := Gender.Female, age := 20, isTwin := true }

-- Define the best painter
def bestPainter := niece

-- Conditions based on the problem (rephrased to match formalization)
def conditions (least_skilled : Member) : Prop :=
  (bestPainter.gender ≠ (match least_skilled with
                          | Member.Grandmother => grandmother
                          | Member.Niece => niece
                          | Member.Nephew => nephew
                          | Member.Granddaughter => granddaughter ).gender) ∧
  ((match least_skilled with
    | Member.Grandmother => grandmother
    | Member.Niece => niece
    | Member.Nephew => nephew
    | Member.Granddaughter => granddaughter ).isTwin) ∧
  (bestPainter.age = (match least_skilled with
                      | Member.Grandmother => grandmother
                      | Member.Niece => niece
                      | Member.Nephew => nephew
                      | Member.Granddaughter => granddaughter ).age)

-- Statement of the problem
theorem find_least_skilled_painter : ∃ m : Member, conditions m ∧ m = Member.Granddaughter :=
by
  sorry

end find_least_skilled_painter_l175_175115


namespace remainders_equality_l175_175487

open Nat

theorem remainders_equality (P P' D R R' r r': ℕ) 
  (hP : P > P')
  (hP_R : P % D = R)
  (hP'_R' : P' % D = R')
  (hPP' : (P * P') % D = r)
  (hRR' : (R * R') % D = r') : r = r' := 
sorry

end remainders_equality_l175_175487


namespace linear_function_quadrants_l175_175106

theorem linear_function_quadrants
  (k : ℝ) (h₀ : k ≠ 0) (h₁ : ∀ x : ℝ, x > 0 → k*x < 0) :
  (∃ x > 0, 2*x + k > 0) ∧
  (∃ x > 0, 2*x + k < 0) ∧
  (∃ x < 0, 2*x + k < 0) :=
  by
  sorry

end linear_function_quadrants_l175_175106


namespace pet_store_initial_gerbils_l175_175066

-- Define sold gerbils
def sold_gerbils : ℕ := 69

-- Define left gerbils
def left_gerbils : ℕ := 16

-- Define the initial number of gerbils
def initial_gerbils : ℕ := sold_gerbils + left_gerbils

-- State the theorem to be proved
theorem pet_store_initial_gerbils : initial_gerbils = 85 := by
  -- This is where the proof would go
  sorry

end pet_store_initial_gerbils_l175_175066


namespace travel_time_l175_175626

noncomputable def distance (time: ℝ) (rate: ℝ) : ℝ := time * rate

theorem travel_time
  (initial_time: ℝ)
  (initial_speed: ℝ)
  (reduced_speed: ℝ)
  (stopover: ℝ)
  (h1: initial_time = 4)
  (h2: initial_speed = 80)
  (h3: reduced_speed = 50)
  (h4: stopover = 0.5) :
  (distance initial_time initial_speed) / reduced_speed + stopover = 6.9 := 
by
  sorry

end travel_time_l175_175626


namespace range_of_a_l175_175210

theorem range_of_a (a : ℝ) (an bn : ℕ → ℝ)
  (h_an : ∀ n, an n = (-1) ^ (n + 2013) * a)
  (h_bn : ∀ n, bn n = 2 + (-1) ^ (n + 2014) / n)
  (h_condition : ∀ n : ℕ, 1 ≤ n → an n < bn n) :
  -2 ≤ a ∧ a < 1 :=
by
  sorry

end range_of_a_l175_175210


namespace quadratic_roots_quadratic_roots_one_quadratic_roots_two_l175_175309

open scoped Classical

variables {p : Type*} [Field p] {a b c x : p}

theorem quadratic_roots (h_a : a ≠ 0) :
  (¬ ∃ y : p, y^2 = b^2 - 4 * a * c) → ∀ x : p, ¬ a * x^2 + b * x + c = 0 :=
by sorry

theorem quadratic_roots_one (h_a : a ≠ 0) :
  (b^2 - 4 * a * c = 0) → ∃ x : p, a * x^2 + b * x + c = 0 ∧ ∀ y : p, a * y^2 + b * y + c = 0 → y = x :=
by sorry

theorem quadratic_roots_two (h_a : a ≠ 0) :
  (∃ y : p, y^2 = b^2 - 4 * a * c) ∧ (b^2 - 4 * a * c ≠ 0) → ∃ x1 x2 : p, x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
by sorry

end quadratic_roots_quadratic_roots_one_quadratic_roots_two_l175_175309


namespace circle_equation_l175_175038

theorem circle_equation 
  (x y : ℝ)
  (passes_origin : (x, y) = (0, 0))
  (intersects_line : ∃ (x y : ℝ), 2 * x - y + 1 = 0)
  (intersects_circle : ∃ (x y :ℝ), x^2 + y^2 - 2 * x - 15 = 0) : 
  x^2 + y^2 + 28 * x - 15 * y = 0 :=
sorry

end circle_equation_l175_175038


namespace total_pepper_weight_l175_175332

theorem total_pepper_weight :
  let green_peppers := 2.8333333333333335
  let red_peppers := 3.254
  let yellow_peppers := 1.375
  let orange_peppers := 0.567
  (green_peppers + red_peppers + yellow_peppers + orange_peppers) = 8.029333333333333 := 
by
  sorry

end total_pepper_weight_l175_175332


namespace animal_eyes_count_l175_175367

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end animal_eyes_count_l175_175367


namespace max_homework_time_l175_175122

theorem max_homework_time (biology_time history_time geography_time : ℕ) :
    biology_time = 20 ∧ history_time = 2 * biology_time ∧ geography_time = 3 * history_time →
    biology_time + history_time + geography_time = 180 :=
by
    intros
    sorry

end max_homework_time_l175_175122


namespace remainder_when_690_div_170_l175_175972

theorem remainder_when_690_div_170 :
  ∃ r : ℕ, ∃ k l : ℕ, 
    gcd (690 - r) (875 - 25) = 170 ∧
    r = 690 % 170 ∧
    l = 875 / 170 ∧
    r = 10 :=
by 
  sorry

end remainder_when_690_div_170_l175_175972


namespace how_many_fewer_runs_did_E_score_l175_175324

-- Define the conditions
variables (a b c d e : ℕ)
variable (h1 : 5 * 36 = 180)
variable (h2 : d = e + 5)
variable (h3 : e = 20)
variable (h4 : b = d + e)
variable (h5 : b + c = 107)
variable (h6 : a + b + c + d + e = 180)

-- Specification to be proved
theorem how_many_fewer_runs_did_E_score :
  a - e = 8 :=
by {
  sorry
}

end how_many_fewer_runs_did_E_score_l175_175324


namespace complementary_angles_decrease_percent_l175_175380

theorem complementary_angles_decrease_percent
    (a b : ℝ) 
    (h1 : a + b = 90) 
    (h2 : a / b = 3 / 7) 
    (h3 : new_a = a * 1.15) 
    (h4 : new_a + new_b = 90) : 
    (new_b / b * 100) = 93.57 := 
sorry

end complementary_angles_decrease_percent_l175_175380


namespace simplify_expression_l175_175892

theorem simplify_expression :
  (2 + 1 / 2) / (1 - 3 / 4) = 10 :=
by
  sorry

end simplify_expression_l175_175892


namespace roller_coaster_costs_4_l175_175230

-- Definitions from conditions
def tickets_initial: ℕ := 5                     -- Jeanne initially has 5 tickets
def tickets_to_buy: ℕ := 8                      -- Jeanne needs to buy 8 more tickets
def total_tickets_needed: ℕ := tickets_initial + tickets_to_buy -- Total tickets needed
def tickets_ferris_wheel: ℕ := 5                -- Ferris wheel costs 5 tickets
def tickets_total_after_ferris_wheel: ℕ := total_tickets_needed - tickets_ferris_wheel -- Remaining tickets after Ferris wheel

-- Definition to be proved (question = answer)
def cost_roller_coaster_bumper_cars: ℕ := tickets_total_after_ferris_wheel / 2 -- Each of roller coaster and bumper cars cost

-- The theorem that corresponds to the solution
theorem roller_coaster_costs_4 :
  cost_roller_coaster_bumper_cars = 4 :=
by
  sorry

end roller_coaster_costs_4_l175_175230


namespace number_of_boys_l175_175736

theorem number_of_boys (b g : ℕ) (h1: (3/5 : ℚ) * b = (5/6 : ℚ) * g) (h2: b + g = 30)
  (h3: g = (b * 18) / 25): b = 17 := by
  sorry

end number_of_boys_l175_175736


namespace sum_of_squares_remainder_l175_175637

theorem sum_of_squares_remainder (n : ℕ) : 
  ((n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2) % 3 = 2 :=
by
  sorry

end sum_of_squares_remainder_l175_175637


namespace total_selling_price_l175_175565

theorem total_selling_price 
  (n : ℕ) (p : ℕ) (c : ℕ) 
  (h_n : n = 85) (h_p : p = 15) (h_c : c = 85) : 
  (c + p) * n = 8500 :=
by
  sorry

end total_selling_price_l175_175565


namespace ratio_of_female_democrats_l175_175472

theorem ratio_of_female_democrats 
    (M F : ℕ) 
    (H1 : M + F = 990)
    (H2 : M / 4 + 165 = 330) 
    (H3 : 165 = 165) : 
    165 / F = 1 / 2 := 
sorry

end ratio_of_female_democrats_l175_175472


namespace problem_statement_l175_175334

def f (x : Int) : Int :=
  if x > 6 then x^2 - 4
  else if -6 <= x && x <= 6 then 3*x + 2
  else 5

def adjusted_f (x : Int) : Int :=
  let fx := f x
  if x % 3 == 0 then fx + 5 else fx

theorem problem_statement : 
  adjusted_f (-8) + adjusted_f 0 + adjusted_f 9 = 94 :=
by 
  sorry

end problem_statement_l175_175334


namespace total_cans_given_away_l175_175390

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end total_cans_given_away_l175_175390


namespace find_positive_real_solutions_l175_175396

open Real

theorem find_positive_real_solutions 
  (x : ℝ) 
  (h : (1/3 * (4 * x^2 - 2)) = ((x^2 - 60 * x - 15) * (x^2 + 30 * x + 3))) :
  x = 30 + sqrt 917 ∨ x = -15 + (sqrt 8016) / 6 :=
by sorry

end find_positive_real_solutions_l175_175396


namespace range_of_2a_sub_b_l175_175194

theorem range_of_2a_sub_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) : -4 < 2 * a - b ∧ 2 * a - b < 2 :=
by
  sorry

end range_of_2a_sub_b_l175_175194


namespace problem_l175_175601

noncomputable def discriminant (p q : ℝ) : ℝ := p^2 - 4 * q
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (p q : ℝ) (hq : q = -2 * p - 5) :
  (quadratic 1 p (q + 1) 2 = 0) →
  q = -2 * p - 5 ∧
  discriminant p q > 0 ∧
  (discriminant p (q + 1) = 0 → 
    (p = -4 ∧ q = 3 ∧ ∀ x : ℝ, quadratic 1 p q x = 0 ↔ (x = 1 ∨ x = 3))) :=
by
  intro hroot_eq
  sorry

end problem_l175_175601


namespace minValue_at_least_9_minValue_is_9_l175_175706

noncomputable def minValue (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) : ℝ :=
  1 / a + 4 / b + 9 / c

theorem minValue_at_least_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) :
  minValue a b c h_pos h_sum ≥ 9 :=
by
  sorry

theorem minValue_is_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4)
  (h_abc : a = 2/3 ∧ b = 4/3 ∧ c = 2) : minValue a b c h_pos h_sum = 9 :=
by
  sorry

end minValue_at_least_9_minValue_is_9_l175_175706


namespace columbus_discovered_america_in_1492_l175_175339

theorem columbus_discovered_america_in_1492 :
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧
  1 + x + y + z = 16 ∧ y + 1 = 5 * z ∧
  1000 + 100 * x + 10 * y + z = 1492 :=
by
  sorry

end columbus_discovered_america_in_1492_l175_175339


namespace power_of_i_l175_175895

theorem power_of_i : (Complex.I ^ 2018) = -1 := by
  sorry

end power_of_i_l175_175895


namespace john_total_cost_l175_175056

def base_cost : ℤ := 25
def text_cost_per_message : ℤ := 8
def extra_minute_cost_per_minute : ℤ := 15
def international_minute_cost : ℤ := 100

def texts_sent : ℤ := 200
def total_hours : ℤ := 42
def international_minutes : ℤ := 10

-- Calculate the number of extra minutes
def extra_minutes : ℤ := (total_hours - 40) * 60

noncomputable def total_cost : ℤ :=
  base_cost +
  (texts_sent * text_cost_per_message) / 100 +
  (extra_minutes * extra_minute_cost_per_minute) / 100 +
  international_minutes * (international_minute_cost / 100)

theorem john_total_cost :
  total_cost = 69 := by
    sorry

end john_total_cost_l175_175056


namespace boxes_containing_neither_l175_175946

theorem boxes_containing_neither
  (total_boxes : ℕ := 15)
  (boxes_with_markers : ℕ := 9)
  (boxes_with_crayons : ℕ := 5)
  (boxes_with_both : ℕ := 4) :
  (total_boxes - ((boxes_with_markers - boxes_with_both) + (boxes_with_crayons - boxes_with_both) + boxes_with_both)) = 5 := by
  sorry

end boxes_containing_neither_l175_175946


namespace factor_sum_l175_175168

theorem factor_sum (R S : ℝ) (h : ∃ (b c : ℝ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + R*x^2 + S) : R + S = 54 :=
sorry

end factor_sum_l175_175168


namespace cubic_polynomial_coefficients_l175_175999

theorem cubic_polynomial_coefficients (f g : Polynomial ℂ) (b c d : ℂ) :
  f = Polynomial.C 4 + Polynomial.X * (Polynomial.C 3 + Polynomial.X * (Polynomial.C 2 + Polynomial.X)) →
  (∀ x, Polynomial.eval x f = 0 → Polynomial.eval (x^2) g = 0) →
  g = Polynomial.C d + Polynomial.X * (Polynomial.C c + Polynomial.X * (Polynomial.C b + Polynomial.X)) →
  (b, c, d) = (4, -15, -32) :=
by
  intro h1 h2 h3
  sorry

end cubic_polynomial_coefficients_l175_175999


namespace john_task_completion_l175_175702

theorem john_task_completion (J : ℝ) (h : 5 * (1 / J + 1 / 10) + 5 * (1 / J) = 1) : J = 20 :=
by
  sorry

end john_task_completion_l175_175702


namespace area_two_layers_l175_175697

-- Given conditions
variables (A_total A_covered A_three_layers : ℕ)

-- Conditions from the problem
def condition_1 : Prop := A_total = 204
def condition_2 : Prop := A_covered = 140
def condition_3 : Prop := A_three_layers = 20

-- Mathematical equivalent proof problem
theorem area_two_layers (A_total A_covered A_three_layers : ℕ) 
  (h1 : condition_1 A_total) 
  (h2 : condition_2 A_covered) 
  (h3 : condition_3 A_three_layers) : 
  ∃ A_two_layers : ℕ, A_two_layers = 24 :=
by sorry

end area_two_layers_l175_175697


namespace triangle_property_l175_175525

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end triangle_property_l175_175525


namespace find_height_of_cuboid_l175_175926

-- Define the cuboid structure and its surface area formula
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

def surface_area (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Given conditions
def given_cuboid : Cuboid := { length := 12, width := 14, height := 7 }
def given_surface_area : ℝ := 700

-- The theorem to prove
theorem find_height_of_cuboid :
  surface_area given_cuboid = given_surface_area :=
by
  sorry

end find_height_of_cuboid_l175_175926


namespace max_quarters_l175_175174

-- Definitions stating the conditions
def total_money_in_dollars : ℝ := 4.80
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10

-- Theorem statement
theorem max_quarters (q : ℕ) (h1 : total_money_in_dollars = (q * value_of_quarter) + (2 * q * value_of_dime)) : q ≤ 10 :=
by {
  -- Injecting a placeholder to facilitate proof development
  sorry
}

end max_quarters_l175_175174


namespace sum_of_solutions_l175_175225

theorem sum_of_solutions (x y : ℝ) (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : x + y = 2 := 
sorry

end sum_of_solutions_l175_175225


namespace banana_distinct_arrangements_l175_175505

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end banana_distinct_arrangements_l175_175505


namespace degree_of_p_is_unbounded_l175_175401

theorem degree_of_p_is_unbounded (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval (x^2 - 1) = (p.eval x) * (p.eval (-x))) : False :=
sorry

end degree_of_p_is_unbounded_l175_175401


namespace original_number_is_120_l175_175721

theorem original_number_is_120 (N k : ℤ) (hk : N - 33 = 87 * k) : N = 120 :=
by
  have h : N - 33 = 87 * 1 := by sorry
  have N_eq : N = 87 + 33 := by sorry
  have N_val : N = 120 := by sorry
  exact N_val

end original_number_is_120_l175_175721


namespace jessica_saves_l175_175740

-- Define the costs based on the conditions given
def basic_cost : ℕ := 15
def movie_cost : ℕ := 12
def sports_cost : ℕ := movie_cost - 3
def bundle_cost : ℕ := 25

-- Define the total cost when the packages are purchased separately
def separate_cost : ℕ := basic_cost + movie_cost + sports_cost

-- Define the savings when opting for the bundle
def savings : ℕ := separate_cost - bundle_cost

-- The theorem that states the savings are 11 dollars
theorem jessica_saves : savings = 11 :=
by
  sorry

end jessica_saves_l175_175740


namespace pos_int_solutions_l175_175484

theorem pos_int_solutions (x : ℤ) : (3 * x - 4 < 2 * x) → (0 < x) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro h1 h2
  have h3 : x - 4 < 0 := by sorry  -- Step derived from inequality simplification
  have h4 : x < 4 := by sorry     -- Adding 4 to both sides
  sorry                           -- Combine conditions to get the specific solutions

end pos_int_solutions_l175_175484


namespace base_angle_of_isosceles_triangle_l175_175078

theorem base_angle_of_isosceles_triangle (a b c : ℝ) 
  (h₁ : a = 50) (h₂ : a + b + c = 180) (h₃ : a = b ∨ b = c ∨ c = a) : 
  b = 50 ∨ b = 65 :=
by sorry

end base_angle_of_isosceles_triangle_l175_175078


namespace tory_earned_more_than_bert_l175_175997

open Real

noncomputable def bert_day1_earnings : ℝ :=
  let initial_sales := 12 * 18
  let discounted_sales := 3 * (18 - 0.15 * 18)
  let total_sales := initial_sales - 3 * 18 + discounted_sales
  total_sales * 0.95

noncomputable def tory_day1_earnings : ℝ :=
  let initial_sales := 15 * 20
  let discounted_sales := 5 * (20 - 0.10 * 20)
  let total_sales := initial_sales - 5 * 20 + discounted_sales
  total_sales * 0.95

noncomputable def bert_day2_earnings : ℝ :=
  let sales := 10 * 15
  (sales * 0.95) * 1.4

noncomputable def tory_day2_earnings : ℝ :=
  let sales := 8 * 18
  (sales * 0.95) * 1.4

noncomputable def bert_total_earnings : ℝ := bert_day1_earnings + bert_day2_earnings

noncomputable def tory_total_earnings : ℝ := tory_day1_earnings + tory_day2_earnings

noncomputable def earnings_difference : ℝ := tory_total_earnings - bert_total_earnings

theorem tory_earned_more_than_bert :
  earnings_difference = 71.82 := by
  sorry

end tory_earned_more_than_bert_l175_175997


namespace square_area_l175_175377

theorem square_area (x y : ℝ) 
  (h1 : x = 20 ∧ y = 20)
  (h2 : x = 20 ∧ y = 5)
  (h3 : x = x ∧ y = 5)
  (h4 : x = x ∧ y = 20)
  : (∃ a : ℝ, a = 225) :=
sorry

end square_area_l175_175377


namespace bottles_produced_by_10_machines_in_4_minutes_l175_175989

variable (rate_per_machine : ℕ)
variable (total_bottles_per_minute_six_machines : ℕ := 240)
variable (number_of_machines : ℕ := 6)
variable (new_number_of_machines : ℕ := 10)
variable (time_in_minutes : ℕ := 4)

theorem bottles_produced_by_10_machines_in_4_minutes :
  rate_per_machine = total_bottles_per_minute_six_machines / number_of_machines →
  (new_number_of_machines * rate_per_machine * time_in_minutes) = 1600 := 
sorry

end bottles_produced_by_10_machines_in_4_minutes_l175_175989


namespace multiplication_of_fractions_l175_175231

theorem multiplication_of_fractions :
  (77 / 4) * (5 / 2) = 48 + 1 / 8 := 
sorry

end multiplication_of_fractions_l175_175231


namespace order_of_numbers_l175_175133

theorem order_of_numbers (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) : y < -y ∧ -y < -xy ∧ -xy < x :=
by 
  sorry

end order_of_numbers_l175_175133


namespace percentage_increase_is_50_l175_175937

-- Define the conditions
variables {P : ℝ} {x : ℝ}

-- Define the main statement (goal)
theorem percentage_increase_is_50 (h : 0.80 * P + (0.008 * x * P) = 1.20 * P) : x = 50 :=
sorry  -- Skip the proof as per instruction

end percentage_increase_is_50_l175_175937


namespace vectors_not_coplanar_l175_175549

def a : ℝ × ℝ × ℝ := (4, 1, 1)
def b : ℝ × ℝ × ℝ := (-9, -4, -9)
def c : ℝ × ℝ × ℝ := (6, 2, 6)

def scalarTripleProduct (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

theorem vectors_not_coplanar : scalarTripleProduct a b c = -18 := by
  sorry

end vectors_not_coplanar_l175_175549


namespace ganesh_average_speed_l175_175990

variable (D : ℝ) (hD : D > 0)

/-- Ganesh's average speed over the entire journey is 45 km/hr.
    Given:
    - Speed from X to Y is 60 km/hr
    - Speed from Y to X is 36 km/hr
--/
theorem ganesh_average_speed :
  let T1 := D / 60
  let T2 := D / 36
  let total_distance := 2 * D
  let total_time := T1 + T2
  (total_distance / total_time) = 45 :=
by
  sorry

end ganesh_average_speed_l175_175990


namespace unique_solution_implies_relation_l175_175615

theorem unique_solution_implies_relation (a b : ℝ)
    (h : ∃! (x y : ℝ), y = x^2 + a * x + b ∧ x = y^2 + a * y + b) : 
    a^2 = 2 * (a + 2 * b) - 1 :=
by
  sorry

end unique_solution_implies_relation_l175_175615


namespace parabola_translation_l175_175667

theorem parabola_translation :
  ∀ f g : ℝ → ℝ,
    (∀ x, f x = - (x - 1) ^ 2) →
    (∀ x, g x = f (x - 1) + 2) →
    ∀ x, g x = - (x - 2) ^ 2 + 2 :=
by
  -- Add the proof steps here if needed
  sorry

end parabola_translation_l175_175667


namespace find_coordinates_of_P_l175_175190

-- Define the points
def P1 : ℝ × ℝ := (2, -1)
def P2 : ℝ × ℝ := (0, 5)

-- Define the point P
def P : ℝ × ℝ := (-2, 11)

-- Conditions encoded as vector relationships
def vector_P1_P (p : ℝ × ℝ) := (p.1 - P1.1, p.2 - P1.2)
def vector_PP2 (p : ℝ × ℝ) := (P2.1 - p.1, P2.2 - p.2)

-- The hypothesis that | P1P | = 2 * | PP2 |
axiom vector_relation : ∀ (p : ℝ × ℝ), 
  vector_P1_P p = (-2 * (vector_PP2 p).1, -2 * (vector_PP2 p).2) → p = P

theorem find_coordinates_of_P : P = (-2, 11) :=
by
  sorry

end find_coordinates_of_P_l175_175190


namespace negation_of_exists_l175_175883

theorem negation_of_exists {x : ℝ} (h : ∃ x : ℝ, 3^x + x < 0) : ∀ x : ℝ, 3^x + x ≥ 0 :=
sorry

end negation_of_exists_l175_175883


namespace crates_on_third_trip_l175_175758

variable (x : ℕ) -- Denote the number of crates carried on the third trip

-- Conditions
def crate_weight := 1250
def max_weight := 6250
def trip3_weight (x : ℕ) := x * crate_weight

-- The problem statement: Prove that x (the number of crates on the third trip) == 5
theorem crates_on_third_trip : trip3_weight x <= max_weight → x = 5 :=
by
  sorry -- No proof required, just statement

end crates_on_third_trip_l175_175758


namespace S2014_value_l175_175817

variable (S : ℕ → ℤ) -- S_n represents sum of the first n terms of the arithmetic sequence
variable (a1 : ℤ) -- First term of the arithmetic sequence
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Given conditions
variable (h1 : a1 = -2016)
variable (h2 : (S 2016) / 2016 - (S 2010) / 2010 = 6)

-- The proof problem
theorem S2014_value :
  S 2014 = -6042 :=
sorry -- Proof omitted

end S2014_value_l175_175817


namespace solve_system_of_equations_l175_175414

-- Define the given system of equations and conditions
theorem solve_system_of_equations (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : yz / (y + z) = a) 
  (h2 : xz / (x + z) = b) 
  (h3 : xy / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧ 
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧ 
  z = 2 * a * b * c / (a * c + b * c - a * b) := sorry

end solve_system_of_equations_l175_175414


namespace factorization_correct_l175_175063

theorem factorization_correct (x y : ℝ) : 
  x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := 
by
  sorry

end factorization_correct_l175_175063


namespace oranges_apples_ratio_l175_175614

variable (A O P : ℕ)
variable (n : ℚ)
variable (h1 : O = n * A)
variable (h2 : P = 4 * O)
variable (h3 : A = (0.08333333333333333 : ℚ) * P)

theorem oranges_apples_ratio (A O P : ℕ) (n : ℚ) 
  (h1 : O = n * A) (h2 : P = 4 * O) (h3 : A = (0.08333333333333333 : ℚ) * P) : n = 3 := 
by
  sorry

end oranges_apples_ratio_l175_175614


namespace min_expr_value_l175_175738

theorem min_expr_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) :
  (∃ a, a = (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ∧ a ≥ 0) → 
  (∀ (u v : ℝ), u = x + 2 → v = 3 * y + 4 → u * v = 16) →
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 :=
sorry

end min_expr_value_l175_175738


namespace calculate_g_at_5_l175_175137

variable {R : Type} [LinearOrderedField R] (g : R → R)
variable (x : R)

theorem calculate_g_at_5 (h : ∀ x : R, g (3 * x - 4) = 5 * x - 7) : g 5 = 8 :=
by
  sorry

end calculate_g_at_5_l175_175137


namespace largest_y_coordinate_l175_175681

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l175_175681


namespace new_area_eq_1_12_original_area_l175_175998

variable (L W : ℝ)
def increased_length (L : ℝ) : ℝ := 1.40 * L
def decreased_width (W : ℝ) : ℝ := 0.80 * W
def original_area (L W : ℝ) : ℝ := L * W
def new_area (L W : ℝ) : ℝ := (increased_length L) * (decreased_width W)

theorem new_area_eq_1_12_original_area (L W : ℝ) :
  new_area L W = 1.12 * (original_area L W) :=
by
  sorry

end new_area_eq_1_12_original_area_l175_175998


namespace percentage_of_400_equals_100_l175_175574

def part : ℝ := 100
def whole : ℝ := 400

theorem percentage_of_400_equals_100 : (part / whole) * 100 = 25 := by
  sorry

end percentage_of_400_equals_100_l175_175574


namespace gcd_lcm_product_eq_l175_175734

-- Define the numbers
def a : ℕ := 10
def b : ℕ := 15

-- Define the GCD and LCM
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Proposition that needs to be proved
theorem gcd_lcm_product_eq : gcd_ab * lcm_ab = 150 :=
  by
    -- Proof would go here
    sorry

end gcd_lcm_product_eq_l175_175734


namespace expression_non_negative_l175_175970

theorem expression_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b)) + (1 / (b - c)) + (4 / (c - a)) ≥ 0 :=
by
  sorry

end expression_non_negative_l175_175970


namespace annual_interest_rate_is_correct_l175_175587

theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, r = 0.0583 ∧
  (200 * (1 + r)^2 = 224) :=
by
  sorry

end annual_interest_rate_is_correct_l175_175587


namespace contrapositive_correct_l175_175952

-- Conditions and the proposition
def prop1 (a : ℝ) : Prop := a = -1 → a^2 = 1

-- The contrapositive of the proposition
def contrapositive (a : ℝ) : Prop := a^2 ≠ 1 → a ≠ -1

-- The proof problem statement
theorem contrapositive_correct (a : ℝ) : prop1 a ↔ contrapositive a :=
by sorry

end contrapositive_correct_l175_175952


namespace find_abc_l175_175296

theorem find_abc (a b c : ℤ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
    (a, b, c) = (3, 5, 15) ∨ (a, b, c) = (2, 4, 8) :=
by
  sorry

end find_abc_l175_175296


namespace min_height_of_cuboid_l175_175280

theorem min_height_of_cuboid (h : ℝ) (side_len : ℝ) (small_spheres_r : ℝ) (large_sphere_r : ℝ) :
  side_len = 4 → 
  small_spheres_r = 1 → 
  large_sphere_r = 2 → 
  ∃ h_min : ℝ, h_min = 2 + 2 * Real.sqrt 7 ∧ h ≥ h_min := 
by
  sorry

end min_height_of_cuboid_l175_175280


namespace tangent_line_at_origin_l175_175659

noncomputable def curve (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 1

def tangent_line (x₀ y₀ : ℝ) (k : ℝ) (x : ℝ) := y₀ + k * (x - x₀)

theorem tangent_line_at_origin : 
  tangent_line 0 (-1) 3 = λ x => 3 * x - 1 :=
by
  sorry

end tangent_line_at_origin_l175_175659


namespace find_x_prime_l175_175214

theorem find_x_prime (x : ℕ) (h1 : x > 0) (h2 : Prime (x^5 + x + 1)) : x = 1 := sorry

end find_x_prime_l175_175214


namespace area_of_triangles_equal_l175_175053

theorem area_of_triangles_equal {a b c d : ℝ} (h_hyperbola_a : a ≠ 0) (h_hyperbola_b : b ≠ 0) 
    (h_hyperbola_c : c ≠ 0) (h_hyperbola_d : d ≠ 0) (h_parallel : a * b = c * d) :
  (1 / 2) * ((a + c) * (a + c) / (a * c)) = (1 / 2) * ((b + d) * (b + d) / (b * d)) :=
by
  sorry

end area_of_triangles_equal_l175_175053


namespace solution_set_condition_l175_175764

theorem solution_set_condition {a : ℝ} : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ (x ≥ 3)) → a < 3 := 
by 
  intros h
  sorry

end solution_set_condition_l175_175764


namespace trapezoid_perimeter_calc_l175_175084

theorem trapezoid_perimeter_calc 
  (EF GH : ℝ) (d : ℝ)
  (h_parallel : EF = 10) 
  (h_eq : GH = 22) 
  (h_distance : d = 5) 
  (h_parallel_cond : EF = 10 ∧ GH = 22 ∧ d = 5) 
: 32 + 2 * Real.sqrt 61 = (10 : ℝ) + 2 * (Real.sqrt ((12 / 2)^2 + 5^2)) + 22 := 
by {
  -- The proof goes here, but for now it's omitted
  sorry
}

end trapezoid_perimeter_calc_l175_175084


namespace compare_M_N_l175_175711

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2) + 7
  let N := (a - 2) * (a - 3)
  M > N :=
by
  sorry

end compare_M_N_l175_175711


namespace perfect_square_if_integer_l175_175002

theorem perfect_square_if_integer (n : ℤ) (k : ℤ) 
  (h : k = 2 + 2 * Int.sqrt (28 * n^2 + 1)) : ∃ m : ℤ, k = m^2 :=
by 
  sorry

end perfect_square_if_integer_l175_175002


namespace area_of_right_isosceles_triangle_l175_175466

def is_right_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

theorem area_of_right_isosceles_triangle (a b c : ℝ) (h : is_right_isosceles a b c) (h_hypotenuse : c = 10) :
  1/2 * a * b = 25 :=
by
  sorry

end area_of_right_isosceles_triangle_l175_175466


namespace time_for_Q_to_finish_job_alone_l175_175387

theorem time_for_Q_to_finish_job_alone (T_Q : ℝ) 
  (h1 : 0 < T_Q)
  (rate_P : ℝ := 1 / 4) 
  (rate_Q : ℝ := 1 / T_Q)
  (combined_work_rate : ℝ := 3 * (rate_P + rate_Q))
  (remaining_work : ℝ := 0.1) -- 0.4 * rate_P
  (total_work_done : ℝ := 0.9) -- 1 - remaining_work
  (h2 : combined_work_rate = total_work_done) : T_Q = 20 :=
by sorry

end time_for_Q_to_finish_job_alone_l175_175387


namespace find_number_l175_175680

theorem find_number (x : ℝ) : ((1.5 * x) / 7 = 271.07142857142856) → x = 1265 :=
by
  sorry

end find_number_l175_175680


namespace intersection_point_l175_175929

theorem intersection_point :
  (∃ (x y : ℝ), 5 * x - 3 * y = 15 ∧ 4 * x + 2 * y = 14)
  → (∃ (x y : ℝ), x = 3 ∧ y = 1) :=
by
  intro h
  sorry

end intersection_point_l175_175929


namespace tan_beta_value_l175_175761

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan (α + β) = -1) : Real.tan β = 3 :=
by
  sorry

end tan_beta_value_l175_175761


namespace solve_for_b_l175_175629

theorem solve_for_b (a b c : ℝ) (cosC : ℝ) (h_a : a = 3) (h_c : c = 4) (h_cosC : cosC = -1/4) :
    c^2 = a^2 + b^2 - 2 * a * b * cosC → b = 7 / 2 :=
by 
  intro h_cosine_theorem
  sorry

end solve_for_b_l175_175629


namespace find_x_l175_175045

-- Define the conditions
def cherryGum := 25
def grapeGum := 35
def packs (x : ℚ) := x -- Each pack contains exactly x pieces of gum

-- Define the ratios after losing one pack of cherry gum and finding 6 packs of grape gum
def ratioAfterLosingCherryPack (x : ℚ) := (cherryGum - packs x) / grapeGum
def ratioAfterFindingGrapePacks (x : ℚ) := cherryGum / (grapeGum + 6 * packs x)

-- State the theorem to be proved
theorem find_x (x : ℚ) (h : ratioAfterLosingCherryPack x = ratioAfterFindingGrapePacks x) : x = 115 / 6 :=
by
  sorry

end find_x_l175_175045


namespace geometric_series_common_ratio_l175_175228

theorem geometric_series_common_ratio 
  (a : ℝ) (S : ℝ) (h_a : a = 500) (h_S : S = 3000) :
  ∃ r : ℝ, r = 5 / 6 :=
by
  sorry

end geometric_series_common_ratio_l175_175228


namespace solve_system_l175_175042

variable (x y z : ℝ)

theorem solve_system :
  (y + z = 20 - 4 * x) →
  (x + z = -18 - 4 * y) →
  (x + y = 10 - 4 * z) →
  (2 * x + 2 * y + 2 * z = 4) :=
by
  intros h1 h2 h3
  sorry

end solve_system_l175_175042


namespace minimum_value_l175_175004

open Real

theorem minimum_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 9) :
  (x ^ 2 + y ^ 2) / (x + y) + (x ^ 2 + z ^ 2) / (x + z) + (y ^ 2 + z ^ 2) / (y + z) ≥ 9 :=
by sorry

end minimum_value_l175_175004


namespace smallest_possible_value_of_d_l175_175585

noncomputable def smallest_value_of_d : ℝ :=
  2 + Real.sqrt 2

theorem smallest_possible_value_of_d (c d : ℝ) (h1 : 2 < c) (h2 : c < d)
    (triangle_condition1 : ¬ (2 + c > d ∧ 2 + d > c ∧ c + d > 2))
    (triangle_condition2 : ¬ ( (2 / d) + (2 / c) > 2)) : d = smallest_value_of_d :=
  sorry

end smallest_possible_value_of_d_l175_175585


namespace general_term_arithmetic_sequence_l175_175838

theorem general_term_arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) (h_d : d ≠ 0)
  (h1 : a 3 + a 10 = 15)
  (h2 : (a 2 + d) * (a 2 + 10 * d) = (a 2 + 4 * d) * (a 2 + d))
  : ∀ n, a n = n + 1 :=
sorry

end general_term_arithmetic_sequence_l175_175838


namespace quadrilateral_area_is_11_l175_175306

def point := (ℤ × ℤ)

def A : point := (0, 0)
def B : point := (1, 4)
def C : point := (4, 3)
def D : point := (3, 0)

def area_of_quadrilateral (p1 p2 p3 p4 : point) : ℤ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let ⟨x3, y3⟩ := p3
  let ⟨x4, y4⟩ := p4
  (|x1*y2 - y1*x2 + x2*y3 - y2*x3 + x3*y4 - y3*x4 + x4*y1 - y4*x1|) / 2

theorem quadrilateral_area_is_11 : area_of_quadrilateral A B C D = 11 := by 
  sorry

end quadrilateral_area_is_11_l175_175306


namespace find_eccentricity_l175_175850

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

def eccentricity_conic_section (m : ℝ) (e : ℝ) : Prop :=
  (m = 6 → e = (Real.sqrt 30) / 6) ∧
  (m = -6 → e = Real.sqrt 7)

theorem find_eccentricity (m : ℝ) :
  geometric_sequence 4 m 9 →
  eccentricity_conic_section m ((Real.sqrt 30) / 6) ∨
  eccentricity_conic_section m (Real.sqrt 7) :=
by
  sorry

end find_eccentricity_l175_175850


namespace log_expression_value_l175_175885

theorem log_expression_value : 
  (Real.logb 10 (Real.sqrt 2) + Real.logb 10 (Real.sqrt 5) + 2 ^ 0 + (5 ^ (1 / 3)) ^ 2 * Real.sqrt 5 = 13 / 2) := 
by 
  -- The proof is omitted as per the instructions
  sorry

end log_expression_value_l175_175885


namespace B_can_complete_alone_l175_175913

-- Define the given conditions
def A_work_rate := 1 / 20
def total_days := 21
def A_quit_days := 15
def B_completion_days := 30

-- Define the problem statement in Lean
theorem B_can_complete_alone (x : ℝ) (h₁ : A_work_rate = 1 / 20) (h₂ : total_days = 21)
  (h₃ : A_quit_days = 15) (h₄ : (21 - A_quit_days) * (1 / 20 + 1 / x) + A_quit_days * (1 / x) = 1) :
  x = B_completion_days :=
  sorry

end B_can_complete_alone_l175_175913


namespace find_y_l175_175918

open Real

theorem find_y : ∃ y : ℝ, (sqrt ((3 - (-5))^2 + (y - 4)^2) = 12) ∧ (y > 0) ∧ (y = 4 + 4 * sqrt 5) :=
by
  use 4 + 4 * sqrt 5
  -- The proof steps would go here.
  sorry

end find_y_l175_175918


namespace sum_of_two_numbers_l175_175021

theorem sum_of_two_numbers (x y : ℕ) (hxy : x > y) (h1 : x - y = 4) (h2 : x * y = 156) : x + y = 28 :=
by {
  sorry
}

end sum_of_two_numbers_l175_175021


namespace min_value_of_squares_l175_175690

theorem min_value_of_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3 * a * b * c = 8) : 
  ∃ m, m ≥ 4 ∧ ∀ a b c, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → a^2 + b^2 + c^2 ≥ m :=
sorry

end min_value_of_squares_l175_175690


namespace field_trip_count_l175_175361

theorem field_trip_count (vans: ℕ) (buses: ℕ) (people_per_van: ℕ) (people_per_bus: ℕ)
  (hv: vans = 9) (hb: buses = 10) (hpv: people_per_van = 8) (hpb: people_per_bus = 27):
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end field_trip_count_l175_175361


namespace fish_population_estimation_l175_175256

-- Definitions based on conditions
def fish_tagged_day1 : ℕ := 80
def fish_caught_day2 : ℕ := 100
def fish_tagged_day2 : ℕ := 20
def fish_caught_day3 : ℕ := 120
def fish_tagged_day3 : ℕ := 36

-- The average percentage of tagged fish caught on the second and third days
def avg_tag_percentage : ℚ := (20 / 100 + 36 / 120) / 2

-- Statement of the proof problem
theorem fish_population_estimation :
  (avg_tag_percentage * P = fish_tagged_day1) → 
  P = 320 :=
by
  -- Proof goes here
  sorry

end fish_population_estimation_l175_175256


namespace centroid_tetrahedron_l175_175276

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D M : V)

def is_centroid (M A B C D : V) : Prop :=
  M = (1/4:ℝ) • (A + B + C + D)

theorem centroid_tetrahedron (h : is_centroid M A B C D) :
  (M - A) + (M - B) + (M - C) + (M - D) = (0 : V) :=
by {
  sorry
}

end centroid_tetrahedron_l175_175276


namespace h_h3_eq_3568_l175_175450

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l175_175450


namespace floor_add_ceil_eq_five_l175_175157

theorem floor_add_ceil_eq_five (x : ℝ) :
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 5 ↔ 2 < x ∧ x < 3 :=
by sorry

end floor_add_ceil_eq_five_l175_175157


namespace fish_ratio_l175_175311

theorem fish_ratio (B T S Bo : ℕ) 
  (hBilly : B = 10) 
  (hTonyBilly : T = 3 * B) 
  (hSarahTony : S = T + 5) 
  (hBobbySarah : Bo = 2 * S) 
  (hTotalFish : Bo + S + T + B = 145) : 
  T / B = 3 :=
by sorry

end fish_ratio_l175_175311


namespace functional_equality_l175_175111

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equality
  (h1 : ∀ x : ℝ, f x ≤ x)
  (h2 : ∀ x y : ℝ, f (x + y) ≤ f x + f y) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end functional_equality_l175_175111


namespace pen_ratio_l175_175085

theorem pen_ratio 
  (Dorothy_pens Julia_pens Robert_pens : ℕ)
  (pen_cost total_cost : ℚ)
  (h1 : Dorothy_pens = Julia_pens / 2)
  (h2 : Robert_pens = 4)
  (h3 : pen_cost = 1.5)
  (h4 : total_cost = 33)
  (h5 : total_cost / pen_cost = Dorothy_pens + Julia_pens + Robert_pens) :
  (Julia_pens / Robert_pens : ℚ) = 3 :=
  sorry

end pen_ratio_l175_175085


namespace car_drive_distance_l175_175500

-- Define the conditions as constants
def driving_speed : ℕ := 8 -- miles per hour
def driving_hours_before_cool : ℕ := 5 -- hours of constant driving
def cooling_hours : ℕ := 1 -- hours needed for cooling down
def total_time : ℕ := 13 -- hours available

-- Define the calculation for distance driven in cycles
def distance_per_cycle : ℕ := driving_speed * driving_hours_before_cool

-- Calculate the duration of one complete cycle
def cycle_duration : ℕ := driving_hours_before_cool + cooling_hours

-- Theorem statement: the car can drive 88 miles in 13 hours
theorem car_drive_distance : distance_per_cycle * (total_time / cycle_duration) + driving_speed * (total_time % cycle_duration) = 88 :=
by
  sorry

end car_drive_distance_l175_175500


namespace number_of_males_who_listen_l175_175874

theorem number_of_males_who_listen (females_listen : ℕ) (males_dont_listen : ℕ) (total_listen : ℕ) (total_dont_listen : ℕ) (total_females : ℕ) :
  females_listen = 72 →
  males_dont_listen = 88 →
  total_listen = 160 →
  total_dont_listen = 180 →
  (total_females = total_listen + total_dont_listen - (females_listen + males_dont_listen)) →
  (total_females + males_dont_listen + 92 = total_listen + total_dont_listen) →
  total_listen + total_dont_listen = females_listen + males_dont_listen + (total_females - females_listen) + 92 :=
sorry

end number_of_males_who_listen_l175_175874


namespace range_of_k_l175_175167

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x + k^2 - 1 ≤ 0) ↔ (-Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2) :=
by 
  sorry

end range_of_k_l175_175167


namespace sum_series_a_sum_series_b_sum_series_c_l175_175057

-- Part (a)
theorem sum_series_a : (∑' n : ℕ, (1 / 2) ^ (n + 1)) = 1 := by
  --skip proof
  sorry

-- Part (b)
theorem sum_series_b : (∑' n : ℕ, (1 / 3) ^ (n + 1)) = 1/2 := by
  --skip proof
  sorry

-- Part (c)
theorem sum_series_c : (∑' n : ℕ, (1 / 4) ^ (n + 1)) = 1/3 := by
  --skip proof
  sorry

end sum_series_a_sum_series_b_sum_series_c_l175_175057


namespace identify_nearly_regular_polyhedra_l175_175330

structure Polyhedron :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

def nearlyRegularPolyhedra : List Polyhedron :=
  [ 
    ⟨8, 12, 6⟩,   -- Properties of Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Properties of Cuboctahedron
    ⟨32, 60, 30⟩  -- Properties of Dodecahedron-Icosahedron
  ]

theorem identify_nearly_regular_polyhedra :
  nearlyRegularPolyhedra = [
    ⟨8, 12, 6⟩,  -- Tetrahedron-octahedron intersection
    ⟨14, 24, 12⟩, -- Cuboctahedron
    ⟨32, 60, 30⟩  -- Dodecahedron-icosahedron intersection
  ] :=
by
  sorry

end identify_nearly_regular_polyhedra_l175_175330


namespace expression_value_l175_175061

theorem expression_value : (1 * 3 * 5 * 7) / (1^2 + 2^2 + 3^2 + 4^2) = 7 / 2 := by
  sorry

end expression_value_l175_175061


namespace average_rate_of_change_l175_175195

noncomputable def f (x : ℝ) : ℝ :=
  -2 * x^2 + 1

theorem average_rate_of_change : 
  ((f 1 - f 0) / (1 - 0)) = -2 :=
by
  sorry

end average_rate_of_change_l175_175195


namespace find_f_4_l175_175860

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 : (∀ x : ℝ, f (x / 2 - 1) = 2 * x + 3) → f 4 = 23 :=
by
  sorry

end find_f_4_l175_175860


namespace gcd_expression_multiple_of_456_l175_175880

theorem gcd_expression_multiple_of_456 (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) : 
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end gcd_expression_multiple_of_456_l175_175880


namespace order_of_numbers_l175_175934

variables (a b : ℚ)

theorem order_of_numbers (ha_pos : a > 0) (hb_neg : b < 0) (habs : |a| < |b|) :
  b < -a ∧ -a < a ∧ a < -b :=
by { sorry }

end order_of_numbers_l175_175934


namespace train_length_is_150_l175_175281

-- Let length_of_train be the length of the train in meters
def length_of_train (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  (speed_kmh * 1000 / 3600) * time_s

theorem train_length_is_150 (speed_kmh time_s : ℕ) (h_speed : speed_kmh = 180) (h_time : time_s = 3) :
  length_of_train speed_kmh time_s = 150 := by
  sorry

end train_length_is_150_l175_175281


namespace base8_to_base10_conversion_l175_175068

def base8_to_base10 (n : Nat) : Nat := 
  match n with
  | 246 => 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  | _ => 0  -- We define this only for the number 246_8

theorem base8_to_base10_conversion : base8_to_base10 246 = 166 := by 
  sorry

end base8_to_base10_conversion_l175_175068


namespace range_of_m_l175_175262

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x

theorem range_of_m (m : ℝ) : f m > 1 → m < 0 := by
  sorry

end range_of_m_l175_175262


namespace cappuccino_cost_l175_175328

theorem cappuccino_cost 
  (total_order_cost drip_price espresso_price latte_price syrup_price cold_brew_price total_other_cost : ℝ)
  (h1 : total_order_cost = 25)
  (h2 : drip_price = 2 * 2.25)
  (h3 : espresso_price = 3.50)
  (h4 : latte_price = 2 * 4.00)
  (h5 : syrup_price = 0.50)
  (h6 : cold_brew_price = 2 * 2.50)
  (h7 : total_other_cost = drip_price + espresso_price + latte_price + syrup_price + cold_brew_price) :
  total_order_cost - total_other_cost = 3.50 := 
by
  sorry

end cappuccino_cost_l175_175328


namespace option_c_correct_l175_175704

theorem option_c_correct (a b : ℝ) (h : a > b) : 2 + a > 2 + b :=
by sorry

end option_c_correct_l175_175704


namespace direct_proportion_l175_175144

theorem direct_proportion (c f p : ℝ) (h : f ≠ 0 ∧ p = c * f) : ∃ k : ℝ, p / f = k * (f / f) :=
by
  sorry

end direct_proportion_l175_175144


namespace molecular_weight_of_oxygen_part_l175_175229

-- Define the known variables as constants
def atomic_weight_oxygen : ℝ := 16.00
def num_oxygen_atoms : ℕ := 2
def molecular_weight_compound : ℝ := 88.00

-- Define the problem as a theorem
theorem molecular_weight_of_oxygen_part :
  16.00 * 2 = 32.00 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_of_oxygen_part_l175_175229


namespace a_in_M_sufficient_not_necessary_l175_175715

-- Defining the sets M and N
def M := {x : ℝ | x^2 < 3 * x}
def N := {x : ℝ | abs (x - 1) < 2}

-- Stating that a ∈ M is a sufficient but not necessary condition for a ∈ N
theorem a_in_M_sufficient_not_necessary (a : ℝ) (h : a ∈ M) : a ∈ N :=
by sorry

end a_in_M_sufficient_not_necessary_l175_175715


namespace unique_prime_sum_and_diff_l175_175571

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def is_sum_of_two_primes (p : ℕ) : Prop :=
  ∃ q1 q2 : ℕ, is_prime q1 ∧ is_prime q2 ∧ p = q1 + q2

noncomputable def is_diff_of_two_primes (p : ℕ) : Prop :=
  ∃ q3 q4 : ℕ, is_prime q3 ∧ is_prime q4 ∧ q3 > q4 ∧ p = q3 - q4

theorem unique_prime_sum_and_diff :
  ∀ p : ℕ, is_prime p ∧ is_sum_of_two_primes p ∧ is_diff_of_two_primes p ↔ p = 5 := 
by
  sorry

end unique_prime_sum_and_diff_l175_175571


namespace product_4_7_25_l175_175036

theorem product_4_7_25 : 4 * 7 * 25 = 700 :=
by sorry

end product_4_7_25_l175_175036


namespace trig_identity_problem_l175_175622

theorem trig_identity_problem 
  (t m n k : ℕ) 
  (h_rel_prime : Nat.gcd m n = 1) 
  (h_condition1 : (1 + Real.sin t) * (1 + Real.cos t) = 8 / 9) 
  (h_condition2 : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k) 
  (h_pos_int_m : 0 < m) 
  (h_pos_int_n : 0 < n) 
  (h_pos_int_k : 0 < k) :
  k + m + n = 15 := 
sorry

end trig_identity_problem_l175_175622


namespace new_rectangle_area_l175_175781

theorem new_rectangle_area (L W : ℝ) (h : L * W = 300) :
  let L_new := 2 * L
  let W_new := 3 * W
  L_new * W_new = 1800 :=
by
  let L_new := 2 * L
  let W_new := 3 * W
  sorry

end new_rectangle_area_l175_175781


namespace max_sum_arithmetic_sequence_l175_175486

theorem max_sum_arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ) (h1 : (a + 2) ^ 2 = (a + 8) * (a - 2))
  (h2 : ∀ k, S k = (k * (2 * a + (k - 1) * d)) / 2)
  (h3 : 10 = a) (h4 : -2 = d) :
  S 10 = 90 :=
sorry

end max_sum_arithmetic_sequence_l175_175486


namespace find_m_l175_175858

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm 45 m = 180) : m = 72 := 
by 
  sorry

end find_m_l175_175858


namespace project_completion_days_l175_175160

theorem project_completion_days (A_days : ℕ) (B_days : ℕ) (A_alone_days : ℕ) :
  A_days = 20 → B_days = 25 → A_alone_days = 2 → (A_alone_days : ℚ) * (1 / A_days) + (10 : ℚ) * (1 / (A_days * B_days / (A_days + B_days))) = 1 :=
by
  sorry

end project_completion_days_l175_175160


namespace exists_ints_a_b_l175_175058

theorem exists_ints_a_b (n : ℤ) (h : n % 4 ≠ 2) : ∃ a b : ℤ, n + a^2 = b^2 :=
by
  sorry

end exists_ints_a_b_l175_175058


namespace percentage_deposited_to_wife_is_33_l175_175405

-- Definitions based on the conditions
def total_income : ℝ := 800000
def children_distribution_rate : ℝ := 0.20
def number_of_children : ℕ := 3
def donation_rate : ℝ := 0.05
def final_amount : ℝ := 40000

-- We can compute the intermediate values to use them in the final proof
def amount_distributed_to_children : ℝ := total_income * children_distribution_rate * number_of_children
def remaining_after_distribution : ℝ := total_income - amount_distributed_to_children
def donation_amount : ℝ := remaining_after_distribution * donation_rate
def remaining_after_donation : ℝ := remaining_after_distribution - donation_amount
def deposited_to_wife : ℝ := remaining_after_donation - final_amount

-- The statement to prove
theorem percentage_deposited_to_wife_is_33 :
  (deposited_to_wife / total_income) * 100 = 33 := by
  sorry

end percentage_deposited_to_wife_is_33_l175_175405


namespace find_rate_percent_l175_175114

-- Definitions
def simpleInterest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Given conditions
def principal : ℕ := 900
def time : ℕ := 4
def simpleInterestValue : ℕ := 160

-- Rate percent
theorem find_rate_percent : 
  ∃ R : ℕ, simpleInterest principal R time = simpleInterestValue :=
by
  sorry

end find_rate_percent_l175_175114


namespace non_shaded_perimeter_6_l175_175936

theorem non_shaded_perimeter_6 
  (area_shaded : ℝ) (area_large_rect : ℝ) (area_extension : ℝ) (total_area : ℝ)
  (non_shaded_area : ℝ) (perimeter : ℝ) :
  area_shaded = 104 → 
  area_large_rect = 12 * 8 → 
  area_extension = 5 * 2 → 
  total_area = area_large_rect + area_extension → 
  non_shaded_area = total_area - area_shaded → 
  non_shaded_area = 2 → 
  perimeter = 2 * (2 + 1) → 
  perimeter = 6 := 
by 
  sorry

end non_shaded_perimeter_6_l175_175936


namespace unique_cell_distance_50_l175_175982

noncomputable def king_dist (A B: ℤ × ℤ) : ℤ :=
  max (abs (A.1 - B.1)) (abs (A.2 - B.2))

theorem unique_cell_distance_50
  (A B C: ℤ × ℤ)
  (hAB: king_dist A B = 100)
  (hBC: king_dist B C = 100)
  (hCA: king_dist C A = 100) :
  ∃! (X: ℤ × ℤ), king_dist X A = 50 ∧ king_dist X B = 50 ∧ king_dist X C = 50 :=
sorry

end unique_cell_distance_50_l175_175982


namespace additional_tanks_needed_l175_175797

theorem additional_tanks_needed 
    (initial_tanks : ℕ) 
    (initial_capacity_per_tank : ℕ) 
    (total_fish_needed : ℕ) 
    (new_capacity_per_tank : ℕ)
    (h_t1 : initial_tanks = 3)
    (h_t2 : initial_capacity_per_tank = 15)
    (h_t3 : total_fish_needed = 75)
    (h_t4 : new_capacity_per_tank = 10) : 
    (total_fish_needed - initial_tanks * initial_capacity_per_tank) / new_capacity_per_tank = 3 := 
by {
    sorry
}

end additional_tanks_needed_l175_175797


namespace solve_for_w_l175_175153

theorem solve_for_w (w : ℕ) (h : w^2 - 5 * w = 0) (hp : w > 0) : w = 5 :=
sorry

end solve_for_w_l175_175153


namespace range_of_fx_over_x_l175_175590

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

theorem range_of_fx_over_x (odd_f : is_odd f)
                           (increasing_f_pos : is_increasing_on f {x : ℝ | x > 0})
                           (hf1 : f (-1) = 0) :
  {x | f x / x < 0} = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
sorry

end range_of_fx_over_x_l175_175590


namespace first_term_geometric_progression_l175_175804

theorem first_term_geometric_progression (a r : ℝ) 
  (h1 : a / (1 - r) = 6)
  (h2 : a + a * r = 9 / 2) :
  a = 3 ∨ a = 9 := 
sorry -- Proof omitted

end first_term_geometric_progression_l175_175804


namespace pool_fill_time_l175_175561

theorem pool_fill_time
  (faster_pipe_time : ℝ) (slower_pipe_factor : ℝ)
  (H1 : faster_pipe_time = 9) 
  (H2 : slower_pipe_factor = 1.25) : 
  (faster_pipe_time * (1 + slower_pipe_factor) / (faster_pipe_time + faster_pipe_time/slower_pipe_factor)) = 5 :=
by
  sorry

end pool_fill_time_l175_175561


namespace necessary_but_not_sufficient_l175_175739

variable (p q : Prop)

theorem necessary_but_not_sufficient (hp : p) : p ∧ q ↔ p ∧ (p ∧ q → q) :=
  sorry

end necessary_but_not_sufficient_l175_175739


namespace count_whole_numbers_in_interval_l175_175448

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l175_175448


namespace area_enclosed_by_graph_eq_2pi_l175_175609

theorem area_enclosed_by_graph_eq_2pi :
  (∃ (x y : ℝ), x^2 + y^2 = 2 * |x| + 2 * |y| ) →
  ∀ (A : ℝ), A = 2 * Real.pi :=
sorry

end area_enclosed_by_graph_eq_2pi_l175_175609


namespace alice_winning_strategy_l175_175693

theorem alice_winning_strategy (N : ℕ) (hN : N > 0) : 
  (∃! n : ℕ, N = n * n) ↔ (∀ (k : ℕ), ∃ (m : ℕ), m ≠ k ∧ (m ∣ k ∨ k ∣ m)) :=
sorry

end alice_winning_strategy_l175_175693


namespace xy_value_l175_175246

theorem xy_value (x y : ℝ) (h₁ : x + y = 2) (h₂ : x^2 * y^3 + y^2 * x^3 = 32) :
  x * y = 2^(5/3) :=
by
  sorry

end xy_value_l175_175246


namespace percentage_owning_cats_percentage_owning_birds_l175_175463

def total_students : ℕ := 500
def students_owning_cats : ℕ := 80
def students_owning_birds : ℕ := 120

theorem percentage_owning_cats : students_owning_cats * 100 / total_students = 16 := 
by 
  sorry

theorem percentage_owning_birds : students_owning_birds * 100 / total_students = 24 := 
by 
  sorry

end percentage_owning_cats_percentage_owning_birds_l175_175463


namespace courier_total_travel_times_l175_175421

-- Define the conditions
variables (v1 v2 : ℝ) (t : ℝ)
axiom speed_condition_1 : v1 * (t + 16) = (v1 + v2) * t
axiom speed_condition_2 : v2 * (t + 9) = (v1 + v2) * t
axiom time_condition : t = 12

-- Define the total travel times
def total_travel_time_1 : ℝ := t + 16
def total_travel_time_2 : ℝ := t + 9

-- Proof problem statement
theorem courier_total_travel_times :
  total_travel_time_1 = 28 ∧ total_travel_time_2 = 21 :=
by
  sorry

end courier_total_travel_times_l175_175421


namespace necessary_but_not_sufficient_l175_175439

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a ≠ 0) → (ab ≠ 0) ↔ (a ≠ 0) :=
by sorry

end necessary_but_not_sufficient_l175_175439


namespace intersection_M_N_l175_175453

def M (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0
def N (y : ℝ) : Prop := ∃ x : ℝ, y = Real.log x

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {y : ℝ | N y} = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l175_175453


namespace geometric_sequence_a5_l175_175362

theorem geometric_sequence_a5
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_ratio : ∀ n, a (n + 1) = 2 * a n)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := 
sorry

end geometric_sequence_a5_l175_175362


namespace winning_candidate_percentage_l175_175314

def percentage_votes (votes1 votes2 votes3 : ℕ) : ℚ := 
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := max (max votes1 votes2) votes3
  (winning_votes * 100) / total_votes

theorem winning_candidate_percentage :
  percentage_votes 3000 5000 15000 = (15000 * 100) / (3000 + 5000 + 15000) :=
by 
  -- This computation should give us the exact percentage fraction.
  -- Simplifying it would yield the result approximately 65.22%
  -- Proof steps can be provided here.
  sorry

end winning_candidate_percentage_l175_175314


namespace stripe_width_l175_175383

theorem stripe_width (x : ℝ) (h : 60 * x - x^2 = 400) : x = 30 - 5 * Real.sqrt 5 := 
  sorry

end stripe_width_l175_175383


namespace smallest_num_conditions_l175_175409

theorem smallest_num_conditions :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧ n = 11 :=
by
  sorry

end smallest_num_conditions_l175_175409


namespace chip_price_reduction_equation_l175_175792

-- Define initial price
def initial_price : ℝ := 400

-- Define final price after reductions
def final_price : ℝ := 144

-- Define the price reduction percentage
variable (x : ℝ)

-- The equation we need to prove
theorem chip_price_reduction_equation :
  initial_price * (1 - x) ^ 2 = final_price :=
sorry

end chip_price_reduction_equation_l175_175792


namespace max_product_of_two_integers_l175_175191

theorem max_product_of_two_integers (x y : ℤ) (h : x + y = 2024) : 
  x * y ≤ 1024144 := sorry

end max_product_of_two_integers_l175_175191


namespace find_digit_B_l175_175720

def six_digit_number (B : ℕ) : ℕ := 303200 + B

def is_prime_six_digit (B : ℕ) : Prop := Prime (six_digit_number B)

theorem find_digit_B :
  ∃ B : ℕ, (B ≤ 9) ∧ (is_prime_six_digit B) ∧ (B = 9) :=
sorry

end find_digit_B_l175_175720


namespace number_of_blobs_of_glue_is_96_l175_175349

def pyramid_blobs_of_glue : Nat :=
  let layer1 := 4 * (4 - 1) * 2
  let layer2 := 3 * (3 - 1) * 2
  let layer3 := 2 * (2 - 1) * 2
  let between1_and_2 := 3 * 3 * 4
  let between2_and_3 := 2 * 2 * 4
  let between3_and_4 := 4
  layer1 + layer2 + layer3 + between1_and_2 + between2_and_3 + between3_and_4

theorem number_of_blobs_of_glue_is_96 :
  pyramid_blobs_of_glue = 96 :=
by
  sorry

end number_of_blobs_of_glue_is_96_l175_175349


namespace find_m_f_monotonicity_l175_175492

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / x - x ^ m

theorem find_m : ∃ (m : ℝ), f 4 m = -7 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := 2 / x - x

theorem f_monotonicity : ∀ x1 x2 : ℝ, (0 < x2 ∧ x2 < x1) → f x1 1 < f x2 1 := sorry

end find_m_f_monotonicity_l175_175492


namespace tom_charges_per_lawn_l175_175627

theorem tom_charges_per_lawn (gas_cost earnings_from_weeding total_profit lawns_mowed : ℕ) (charge_per_lawn : ℤ) 
  (h1 : gas_cost = 17)
  (h2 : earnings_from_weeding = 10)
  (h3 : total_profit = 29)
  (h4 : lawns_mowed = 3)
  (h5 : total_profit = ((lawns_mowed * charge_per_lawn) + earnings_from_weeding) - gas_cost) :
  charge_per_lawn = 12 := 
by
  sorry

end tom_charges_per_lawn_l175_175627


namespace value_of_c_l175_175902

theorem value_of_c (a b c d w x y z : ℕ) (primes : ∀ p ∈ [w, x, y, z], Prime p)
  (h1 : w < x) (h2 : x < y) (h3 : y < z) 
  (h4 : (w^a) * (x^b) * (y^c) * (z^d) = 660) 
  (h5 : (a + b) - (c + d) = 1) : c = 1 :=
by {
  sorry
}

end value_of_c_l175_175902


namespace greatest_divisor_l175_175307

theorem greatest_divisor (n : ℕ) (h1 : 3461 % n = 23) (h2 : 4783 % n = 41) : n = 2 := by {
  sorry
}

end greatest_divisor_l175_175307


namespace cheryl_used_material_l175_175886

theorem cheryl_used_material
    (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
    (h1 : material1 = 5/9)
    (h2 : material2 = 1/3)
    (h_lf : leftover = 8/24) :
    material1 + material2 - leftover = 5/9 :=
by
  sorry

end cheryl_used_material_l175_175886


namespace hawks_score_l175_175384

theorem hawks_score (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 18) : y = 16 := by
  sorry

end hawks_score_l175_175384


namespace profit_percentage_is_50_l175_175126

noncomputable def cost_of_machine := 11000
noncomputable def repair_cost := 5000
noncomputable def transportation_charges := 1000
noncomputable def selling_price := 25500

noncomputable def total_cost := cost_of_machine + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost) * 100

theorem profit_percentage_is_50 : profit_percentage = 50 := by
  sorry

end profit_percentage_is_50_l175_175126


namespace total_charge_for_3_6_miles_during_peak_hours_l175_175474

-- Define the initial conditions as constants
def initial_fee : ℝ := 2.05
def charge_per_half_mile_first_2_miles : ℝ := 0.45
def charge_per_two_fifth_mile_after_2_miles : ℝ := 0.35
def peak_hour_surcharge : ℝ := 1.50

-- Define the function to calculate the total charge
noncomputable def total_charge (total_distance : ℝ) (is_peak_hour : Bool) : ℝ :=
  let first_2_miles_charge := if total_distance > 2 then 4 * charge_per_half_mile_first_2_miles else (total_distance / 0.5) * charge_per_half_mile_first_2_miles
  let remaining_distance := if total_distance > 2 then total_distance - 2 else 0
  let after_2_miles_charge := if total_distance > 2 then (remaining_distance / (2 / 5)) * charge_per_two_fifth_mile_after_2_miles else 0
  let surcharge := if is_peak_hour then peak_hour_surcharge else 0
  initial_fee + first_2_miles_charge + after_2_miles_charge + surcharge

-- Prove that total charge of 3.6 miles during peak hours is 6.75
theorem total_charge_for_3_6_miles_during_peak_hours : total_charge 3.6 true = 6.75 := by
  sorry

end total_charge_for_3_6_miles_during_peak_hours_l175_175474


namespace number_of_real_roots_l175_175762

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 19) ^ x + (5 / 19) ^ x + (11 / 19) ^ x

noncomputable def g (x : ℝ) : ℝ := sqrt (x - 1)

theorem number_of_real_roots : ∃! x : ℝ, 1 ≤ x ∧ f x = g x :=
by
  sorry

end number_of_real_roots_l175_175762


namespace count_integers_congruent_to_7_mod_13_l175_175839

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l175_175839


namespace point_on_graph_l175_175765

theorem point_on_graph (g : ℝ → ℝ) (h : g 8 = 10) :
  ∃ x y : ℝ, 3 * y = g (3 * x - 1) + 3 ∧ x = 3 ∧ y = 13 / 3 ∧ x + y = 22 / 3 :=
by
  sorry

end point_on_graph_l175_175765


namespace total_travel_time_in_minutes_l175_175901

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l175_175901


namespace solve_root_equation_l175_175329

noncomputable def sqrt4 (x : ℝ) : ℝ := x^(1/4)

theorem solve_root_equation (x : ℝ) :
  sqrt4 (43 - 2 * x) + sqrt4 (39 + 2 * x) = 4 ↔ x = 21 ∨ x = -13.5 :=
by
  sorry

end solve_root_equation_l175_175329


namespace mul_mod_eq_l175_175532

theorem mul_mod_eq :
  (66 * 77 * 88) % 25 = 16 :=
by 
  sorry

end mul_mod_eq_l175_175532


namespace find_last_number_l175_175746

theorem find_last_number
  (A B C D : ℝ)
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11) :
  D = 4 :=
by
  sorry

end find_last_number_l175_175746


namespace additional_distance_l175_175508

theorem additional_distance (distance_speed_10 : ℝ) (speed1 speed2 time1 time2 distance actual_distance additional_distance : ℝ)
  (h1 : actual_distance = distance_speed_10)
  (h2 : time1 = distance_speed_10 / speed1)
  (h3 : time1 = 5)
  (h4 : speed1 = 10)
  (h5 : time2 = actual_distance / speed2)
  (h6 : speed2 = 14)
  (h7 : distance = speed2 * time1)
  (h8 : distance = 70)
  : additional_distance = distance - actual_distance
  := by
  sorry

end additional_distance_l175_175508


namespace quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l175_175371

-- Proof Problem (1)
theorem quadratic_inequality_roots_a_eq_neg1
  (a : ℝ)
  (h : ∀ x, (-1 < x ∧ x < 3) → ax^2 - 2 * a * x + 3 > 0) :
  a = -1 :=
sorry

-- Proof Problem (2)
theorem quadratic_inequality_for_all_real_a_range
  (a : ℝ)
  (h : ∀ x, ax^2 - 2 * a * x + 3 > 0) :
  0 ≤ a ∧ a < 3 :=
sorry

end quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l175_175371


namespace tip_percentage_l175_175809

theorem tip_percentage
  (original_bill : ℝ)
  (shared_per_person : ℝ)
  (num_people : ℕ)
  (total_shared : ℝ)
  (tip_percent : ℝ)
  (h1 : original_bill = 139.0)
  (h2 : shared_per_person = 50.97)
  (h3 : num_people = 3)
  (h4 : total_shared = shared_per_person * num_people)
  (h5 : total_shared - original_bill = 13.91) :
  tip_percent = 13.91 / 139.0 * 100 := 
sorry

end tip_percentage_l175_175809


namespace smallest_k_power_l175_175834

theorem smallest_k_power (k : ℕ) (hk : ∀ m : ℕ, m < 14 → 7^m ≤ 4^19) : 7^14 > 4^19 :=
sorry

end smallest_k_power_l175_175834


namespace black_squares_count_l175_175316

def checkerboard_size : Nat := 32
def total_squares : Nat := checkerboard_size * checkerboard_size
def black_squares (n : Nat) : Nat := n / 2

theorem black_squares_count : black_squares total_squares = 512 := by
  let n := total_squares
  show black_squares n = 512
  sorry

end black_squares_count_l175_175316


namespace probability_four_vertices_same_plane_proof_l175_175641

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end probability_four_vertices_same_plane_proof_l175_175641


namespace quadratic_minimum_l175_175723

-- Define the constants p and q as positive real numbers
variables (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 3 * x^2 + p * x + q

-- Assertion to prove: the function f reaches its minimum at x = -p / 6
theorem quadratic_minimum : 
  ∃ x : ℝ, x = -p / 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
sorry

end quadratic_minimum_l175_175723


namespace proof1_proof2_proof3_proof4_l175_175864

-- Define variables.
variable (m n x y z : ℝ)

-- Prove the expressions equalities.
theorem proof1 : (m + 2 * n) - (m - 2 * n) = 4 * n := sorry
theorem proof2 : 2 * (x - 3) - (-x + 4) = 3 * x - 10 := sorry
theorem proof3 : 2 * x - 3 * (x - 2 * y + 3 * x) + 2 * (3 * x - 3 * y + 2 * z) = -4 * x + 4 * z := sorry
theorem proof4 : 8 * m^2 - (4 * m^2 - 2 * m - 4 * (2 * m^2 - 5 * m)) = 12 * m^2 - 18 * m := sorry

end proof1_proof2_proof3_proof4_l175_175864


namespace power_multiplication_result_l175_175919

theorem power_multiplication_result :
  ( (8 / 9)^3 * (1 / 3)^3 * (2 / 5)^3 = (4096 / 2460375) ) :=
by
  sorry

end power_multiplication_result_l175_175919


namespace square_area_divided_into_rectangles_l175_175510

theorem square_area_divided_into_rectangles (l w : ℝ) 
  (h1 : 2 * (l + w) = 120)
  (h2 : l = 5 * w) :
  (5 * w * w)^2 = 2500 := 
by {
  -- Sorry placeholder for proof
  sorry
}

end square_area_divided_into_rectangles_l175_175510


namespace second_horse_revolutions_l175_175660

-- Define the parameters and conditions:
def r₁ : ℝ := 30  -- Distance of the first horse from the center
def revolutions₁ : ℕ := 15  -- Number of revolutions by the first horse
def r₂ : ℝ := 5  -- Distance of the second horse from the center

-- Define the statement to prove:
theorem second_horse_revolutions : r₂ * (↑revolutions₁ * r₁⁻¹) * (↑revolutions₁) = 90 := 
by sorry

end second_horse_revolutions_l175_175660


namespace three_zeros_implies_a_lt_neg3_l175_175264

noncomputable def f (a x : ℝ) := x^3 + a * x + 2

theorem three_zeros_implies_a_lt_neg3 (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) →
  a < -3 :=
by
  sorry

end three_zeros_implies_a_lt_neg3_l175_175264


namespace expression_range_l175_175731

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end expression_range_l175_175731


namespace num_candidates_l175_175234

theorem num_candidates (n : ℕ) (h : n * (n - 1) = 30) : n = 6 :=
sorry

end num_candidates_l175_175234


namespace digit_expression_equals_2021_l175_175995

theorem digit_expression_equals_2021 :
  ∃ (f : ℕ → ℕ), 
  (f 0 = 0 ∧
   f 1 = 1 ∧
   f 2 = 2 ∧
   f 3 = 3 ∧
   f 4 = 4 ∧
   f 5 = 5 ∧
   f 6 = 6 ∧
   f 7 = 7 ∧
   f 8 = 8 ∧
   f 9 = 9 ∧
   43 * (8 * 5 + 7) + 0 * 1 * 2 * 6 * 9 = 2021) :=
sorry

end digit_expression_equals_2021_l175_175995


namespace ratio_of_areas_l175_175374

noncomputable def area_ratio (a : ℝ) : ℝ :=
  let side_triangle : ℝ := a
  let area_triangle : ℝ := (1 / 2) * side_triangle * side_triangle
  let height_rhombus : ℝ := side_triangle * Real.sin (Real.pi / 3)
  let area_rhombus : ℝ := height_rhombus * side_triangle
  area_rhombus / area_triangle

theorem ratio_of_areas (a : ℝ) (h : a > 0) : area_ratio a = 3 := by
  -- The proof would be here
  sorry

end ratio_of_areas_l175_175374


namespace sector_area_l175_175714

noncomputable def area_of_sector (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) : ℝ :=
  1 / 2 * arc_length * radius

theorem sector_area (R : ℝ)
  (arc_length : ℝ) (central_angle : ℝ)
  (h_arc : arc_length = 4 * Real.pi)
  (h_angle : central_angle = Real.pi / 3)
  (h_radius : arc_length = central_angle * R) :
  area_of_sector arc_length central_angle 12 = 24 * Real.pi :=
by
  -- Proof skipped
  sorry

#check sector_area

end sector_area_l175_175714


namespace cyclist_waits_15_minutes_l175_175243

-- Definitions
def hiker_rate := 7 -- miles per hour
def cyclist_rate := 28 -- miles per hour
def wait_time := 15 / 60 -- hours, as the cyclist waits 15 minutes, converted to hours

-- The statement to be proven
theorem cyclist_waits_15_minutes :
  ∃ t : ℝ, t = 15 / 60 ∧
  (∀ d : ℝ, d = (hiker_rate * wait_time) →
            d = (cyclist_rate * t - hiker_rate * t)) :=
by
  sorry

end cyclist_waits_15_minutes_l175_175243


namespace negation_correct_l175_175250

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct_l175_175250


namespace heat_of_neutralization_combination_l175_175552

-- Define instruments
inductive Instrument
| Balance
| MeasuringCylinder
| Beaker
| Burette
| Thermometer
| TestTube
| AlcoholLamp

def correct_combination : List Instrument :=
  [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer]

theorem heat_of_neutralization_combination :
  correct_combination = [Instrument.MeasuringCylinder, Instrument.Beaker, Instrument.Thermometer] :=
sorry

end heat_of_neutralization_combination_l175_175552


namespace find_x_l175_175796

def f (x : ℝ) : ℝ := 2 * x - 3 -- Definition of the function f

def c : ℝ := 11 -- Definition of the constant c

theorem find_x : 
  ∃ x : ℝ, 2 * f x - c = f (x - 2) ↔ x = 5 :=
by 
  sorry

end find_x_l175_175796


namespace fastest_route_time_l175_175546

theorem fastest_route_time (d1 d2 : ℕ) (s1 s2 : ℕ) (h1 : d1 = 1500) (h2 : d2 = 750) (h3 : s1 = 75) (h4 : s2 = 25) :
  min (d1 / s1) (d2 / s2) = 20 := by
  sorry

end fastest_route_time_l175_175546


namespace tenth_term_arithmetic_sequence_l175_175799

def a : ℚ := 2 / 3
def d : ℚ := 2 / 3

theorem tenth_term_arithmetic_sequence : 
  let a := 2 / 3
  let d := 2 / 3
  let n := 10
  a + (n - 1) * d = 20 / 3 := by
  sorry

end tenth_term_arithmetic_sequence_l175_175799


namespace optimal_messenger_strategy_l175_175110

theorem optimal_messenger_strategy (p : ℝ) (hp : 0 < p ∧ p < 1) :
  (p < 1/3 → ∃ n : ℕ, n = 4 ∧ ∀ (k : ℕ), k = 10) ∧ 
  (1/3 ≤ p → ∃ n : ℕ, n = 2 ∧ ∀ (m : ℕ), m = 20) :=
by
  sorry

end optimal_messenger_strategy_l175_175110


namespace cost_of_snake_toy_l175_175879

-- Given conditions
def cost_of_cage : ℝ := 14.54
def dollar_bill_found : ℝ := 1.00
def total_cost : ℝ := 26.30

-- Theorem to find the cost of the snake toy
theorem cost_of_snake_toy : 
  (total_cost + dollar_bill_found - cost_of_cage) = 12.76 := 
  by sorry

end cost_of_snake_toy_l175_175879


namespace green_pill_cost_l175_175105

-- Define the conditions 
variables (pinkCost greenCost : ℝ)
variable (totalCost : ℝ := 819) -- total cost for three weeks
variable (days : ℝ := 21) -- number of days in three weeks

-- Establish relationships between pink and green pill costs
axiom greenIsMore : greenCost = pinkCost + 1
axiom dailyCost : 2 * greenCost + pinkCost = 39

-- Define the theorem to prove the cost of one green pill
theorem green_pill_cost : greenCost = 40/3 :=
by
  -- Proof would go here, but is omitted for now.
  sorry

end green_pill_cost_l175_175105


namespace mod_x_squared_l175_175747

theorem mod_x_squared :
  (∃ x : ℤ, 5 * x ≡ 9 [ZMOD 26] ∧ 4 * x ≡ 15 [ZMOD 26]) →
  ∃ y : ℤ, y ≡ 10 [ZMOD 26] :=
by
  intro h
  rcases h with ⟨x, h₁, h₂⟩
  exists x^2
  sorry

end mod_x_squared_l175_175747


namespace B_investment_is_72000_l175_175239

noncomputable def A_investment : ℝ := 27000
noncomputable def C_investment : ℝ := 81000
noncomputable def C_profit : ℝ := 36000
noncomputable def total_profit : ℝ := 80000

noncomputable def B_investment : ℝ :=
  let total_investment := (C_investment * total_profit) / C_profit
  total_investment - A_investment - C_investment

theorem B_investment_is_72000 :
  B_investment = 72000 :=
by
  sorry

end B_investment_is_72000_l175_175239


namespace sqrt_range_l175_175146

theorem sqrt_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_range_l175_175146


namespace total_dogs_l175_175527

def number_of_boxes : ℕ := 15
def dogs_per_box : ℕ := 8

theorem total_dogs : number_of_boxes * dogs_per_box = 120 := by
  sorry

end total_dogs_l175_175527


namespace triangle_angle_bisectors_l175_175081

theorem triangle_angle_bisectors {a b c : ℝ} (ht : (a = 2 ∧ b = 3 ∧ c < 5)) : 
  (∃ h_a h_b h_c : ℝ, h_a + h_b > h_c ∧ h_a + h_c > h_b ∧ h_b + h_c > h_a) →
  ¬ (∃ ell_a ell_b ell_c : ℝ, ell_a + ell_b > ell_c ∧ ell_a + ell_c > ell_b ∧ ell_b + ell_c > ell_a) :=
by
  sorry

end triangle_angle_bisectors_l175_175081


namespace calculate_original_lemon_price_l175_175344

variable (p_lemon_old p_lemon_new p_grape_old p_grape_new : ℝ)
variable (num_lemons num_grapes revenue : ℝ)

theorem calculate_original_lemon_price :
  ∀ (L : ℝ),
  -- conditions
  p_lemon_old = L ∧
  p_lemon_new = L + 4 ∧
  p_grape_old = 7 ∧
  p_grape_new = 9 ∧
  num_lemons = 80 ∧
  num_grapes = 140 ∧
  revenue = 2220 ->
  -- proof that the original price is 8
  p_lemon_old = 8 :=
by
  intros L h
  have h1 : p_lemon_new = L + 4 := h.2.1
  have h2 : p_grape_old = 7 := h.2.2.1
  have h3 : p_grape_new = 9 := h.2.2.2.1
  have h4 : num_lemons = 80 := h.2.2.2.2.1
  have h5 : num_grapes = 140 := h.2.2.2.2.2.1
  have h6 : revenue = 2220 := h.2.2.2.2.2.2
  sorry

end calculate_original_lemon_price_l175_175344


namespace order_of_xyz_l175_175664

variable (a b c d : ℝ)

noncomputable def x : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
noncomputable def y : ℝ := Real.sqrt (a * c) + Real.sqrt (b * d)
noncomputable def z : ℝ := Real.sqrt (a * d) + Real.sqrt (b * c)

theorem order_of_xyz (h₁ : a > b) (h₂ : b > c) (h₃ : c > d) (h₄ : d > 0) : x a b c d > y a b c d ∧ y a b c d > z a b c d :=
by
  sorry

end order_of_xyz_l175_175664


namespace jessica_withdraw_fraq_l175_175418

theorem jessica_withdraw_fraq {B : ℝ} (h : B - 200 + (1 / 2) * (B - 200) = 450) :
  (200 / B) = 2 / 5 := by
  sorry

end jessica_withdraw_fraq_l175_175418


namespace min_value_ineq_l175_175342

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) : 
  (∀ z : ℝ, z = (4 / x + 1 / y) → z ≥ 9) :=
by
  sorry

end min_value_ineq_l175_175342


namespace set_equality_l175_175608

theorem set_equality (a : ℤ) : 
  {z : ℤ | ∃ x : ℤ, (x - a = z ∧ a - 1 ≤ x ∧ x ≤ a + 1)} = {-1, 0, 1} :=
by {
  sorry
}

end set_equality_l175_175608


namespace books_left_l175_175859

namespace PaulBooksExample

-- Defining the initial conditions as given in the problem
def initial_books : ℕ := 134
def books_given : ℕ := 39
def books_sold : ℕ := 27

-- Proving that the final number of books Paul has is 68
theorem books_left : initial_books - (books_given + books_sold) = 68 := by
  sorry

end PaulBooksExample

end books_left_l175_175859


namespace imaginary_part_of_z_l175_175360

open Complex

-- Define the context
variables (z : ℂ) (a b : ℂ)

-- Define the condition
def condition := (1 - 2*I) * z = 5 * I

-- Lean 4 statement to prove the imaginary part of z 
theorem imaginary_part_of_z (h : condition z) : z.im = 1 :=
sorry

end imaginary_part_of_z_l175_175360


namespace eggs_for_husband_is_correct_l175_175308

-- Define the conditions
def eggs_per_child : Nat := 2
def num_children : Nat := 4
def eggs_for_herself : Nat := 2
def total_eggs_per_year : Nat := 3380
def days_per_week : Nat := 5
def weeks_per_year : Nat := 52

-- Define the total number of eggs Lisa makes for her husband per year
def eggs_for_husband : Nat :=
  total_eggs_per_year - 
  (num_children * eggs_per_child + eggs_for_herself) * (days_per_week * weeks_per_year)

-- Prove the main statement
theorem eggs_for_husband_is_correct : eggs_for_husband = 780 := by
  sorry

end eggs_for_husband_is_correct_l175_175308


namespace S_2012_value_l175_175638

-- Define the first term of the arithmetic sequence
def a1 : ℤ := -2012

-- Define the common difference
def d : ℤ := 2

-- Define the sequence a_n
def a (n : ℕ) : ℤ := a1 + d * (n - 1)

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℤ := n * (a1 + a n) / 2

-- Formalize the given problem as a Lean statement
theorem S_2012_value : S 2012 = -2012 :=
by 
{
  -- The proof is omitted as requested
  sorry
}

end S_2012_value_l175_175638


namespace original_salary_condition_l175_175691

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end original_salary_condition_l175_175691


namespace positive_number_square_sum_eq_210_l175_175108

theorem positive_number_square_sum_eq_210 (n : ℕ) (h : n^2 + n = 210) : n = 14 :=
sorry

end positive_number_square_sum_eq_210_l175_175108


namespace sin_C_eq_sin_A_minus_B_eq_l175_175656

open Real

-- Problem 1
theorem sin_C_eq (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) :
  sin C = (sqrt 3 + 3 * sqrt 2) / 6 :=
sorry

-- Problem 2
theorem sin_A_minus_B_eq (A B C : ℝ) (a b c : ℝ)
  (h_cosC : cos C = 2 / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) 
  (hA_acute : 0 < A ∧ A < π / 2)
  (hB_acute : 0 < B ∧ B < π / 2) :
  sin (A - B) = -sqrt 5 / 3 :=
sorry

end sin_C_eq_sin_A_minus_B_eq_l175_175656


namespace geometry_problem_l175_175616

-- Definitions for points and segments based on given conditions
variables {O A B C D E F G : Type} [Inhabited O] [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F] [Inhabited G]

-- Lengths of segments based on given conditions
variables (DE EG : ℝ)
variable (BG : ℝ)

-- Given lengths
def given_lengths : Prop :=
  DE = 5 ∧ EG = 3

-- Goal to prove
def goal : Prop :=
  BG = 12

-- The theorem combining conditions and the goal
theorem geometry_problem (h : given_lengths DE EG) : goal BG :=
  sorry

end geometry_problem_l175_175616


namespace probability_X_equals_Y_l175_175435

noncomputable def prob_X_equals_Y : ℚ :=
  let count_intersections : ℚ := 15
  let total_possibilities : ℚ := 15 * 15
  count_intersections / total_possibilities

theorem probability_X_equals_Y :
  (∀ (x y : ℝ), -15 * Real.pi ≤ x ∧ x ≤ 15 * Real.pi ∧ -15 * Real.pi ≤ y ∧ y ≤ 15 * Real.pi →
    (Real.cos (Real.cos x) = Real.cos (Real.cos y)) →
    prob_X_equals_Y = 1/15) :=
sorry

end probability_X_equals_Y_l175_175435


namespace count_two_digit_decimals_between_0_40_and_0_50_l175_175403

theorem count_two_digit_decimals_between_0_40_and_0_50 : 
  ∃ (n : ℕ), n = 9 ∧ ∀ x : ℝ, 0.40 < x ∧ x < 0.50 → (exists d : ℕ, (1 ≤ d ∧ d ≤ 9 ∧ x = 0.4 + d * 0.01)) :=
by
  sorry

end count_two_digit_decimals_between_0_40_and_0_50_l175_175403


namespace luisa_trip_l175_175586

noncomputable def additional_miles (d1: ℝ) (s1: ℝ) (s2: ℝ) (desired_avg_speed: ℝ) : ℝ := 
  let t1 := d1 / s1
  let t := (d1 * (desired_avg_speed - s1)) / (s2 * (s1 - desired_avg_speed))
  s2 * t

theorem luisa_trip :
  additional_miles 18 36 60 45 = 18 :=
by
  sorry

end luisa_trip_l175_175586


namespace find_relationship_l175_175657

theorem find_relationship (n m : ℕ) (a : ℚ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (n > m ↔ (1 / n < a)) → m = ⌊1 / a⌋ :=
sorry

end find_relationship_l175_175657


namespace num_sets_B_l175_175485

open Set

def A : Set ℕ := {1, 3}

theorem num_sets_B :
  ∃ (B : ℕ → Set ℕ), (∀ b, B b ∪ A = {1, 3, 5}) ∧ (∃ s t u v, B s = {5} ∧
                                                   B t = {1, 5} ∧
                                                   B u = {3, 5} ∧
                                                   B v = {1, 3, 5} ∧ 
                                                   s ≠ t ∧ s ≠ u ∧ s ≠ v ∧
                                                   t ≠ u ∧ t ≠ v ∧
                                                   u ≠ v) :=
sorry

end num_sets_B_l175_175485


namespace brick_height_l175_175899

theorem brick_height (h : ℝ) : 
    let wall_length := 900
    let wall_width := 600
    let wall_height := 22.5
    let num_bricks := 7200
    let brick_length := 25
    let brick_width := 11.25
    wall_length * wall_width * wall_height = num_bricks * (brick_length * brick_width * h) -> 
    h = 67.5 := 
by
  intros
  sorry

end brick_height_l175_175899


namespace point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l175_175298

theorem point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb
  (x1 x2 : ℝ) : 
  (x1 * x2 / 4 = -1) ↔ ((x1 / 2) * (x2 / 2) = -1) :=
by sorry

end point_on_line_l_is_necessary_and_sufficient_for_pa_perpendicular_pb_l175_175298


namespace janet_dresses_pockets_l175_175462

theorem janet_dresses_pockets :
  ∀ (x : ℕ), (∀ (dresses_with_pockets remaining_dresses total_pockets : ℕ),
  dresses_with_pockets = 24 / 2 →
  total_pockets = 32 →
  remaining_dresses = dresses_with_pockets - dresses_with_pockets / 3 →
  (dresses_with_pockets / 3) * x + remaining_dresses * 3 = total_pockets →
  x = 2) :=
by
  intros x dresses_with_pockets remaining_dresses total_pockets h1 h2 h3 h4
  sorry

end janet_dresses_pockets_l175_175462


namespace xiao_gang_steps_l175_175087

theorem xiao_gang_steps (x : ℕ) (H1 : 9000 / x = 13500 / (x + 15)) : x = 30 :=
by
  sorry

end xiao_gang_steps_l175_175087


namespace chris_newspapers_l175_175820

theorem chris_newspapers (C L : ℕ) 
  (h1 : L = C + 23) 
  (h2 : C + L = 65) : 
  C = 21 := 
by 
  sorry

end chris_newspapers_l175_175820


namespace difference_of_numbers_l175_175265

/-- Given two natural numbers a and 10a whose sum is 23,320,
prove that the difference between them is 19,080. -/
theorem difference_of_numbers (a : ℕ) (h : a + 10 * a = 23320) : 10 * a - a = 19080 := by
  sorry

end difference_of_numbers_l175_175265


namespace malcolm_initial_white_lights_l175_175991

theorem malcolm_initial_white_lights :
  let red_lights := 12
  let blue_lights := 3 * red_lights
  let green_lights := 6
  let bought_lights := red_lights + blue_lights + green_lights
  let remaining_lights := 5
  let total_needed_lights := bought_lights + remaining_lights
  W = total_needed_lights :=
by
  sorry

end malcolm_initial_white_lights_l175_175991


namespace inequality_true_l175_175139

-- Define the conditions
variables (a b : ℝ) (h : a < b) (hb_neg : b < 0)

-- State the theorem to be proved
theorem inequality_true (ha : a < b) (hb : b < 0) : (|a| / |b| > 1) :=
sorry

end inequality_true_l175_175139


namespace segment_outside_spheres_l175_175793

noncomputable def fraction_outside_spheres (α : ℝ) : ℝ :=
  (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)

theorem segment_outside_spheres (R α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < 2 * Real.pi) :
  fraction_outside_spheres α = (1 - Real.cos (α / 2)^2) / (1 + (Real.cos (α / 2))^2) :=
  by sorry

end segment_outside_spheres_l175_175793


namespace probability_blue_or_green_face_l175_175530

def cube_faces: ℕ := 6
def blue_faces: ℕ := 3
def red_faces: ℕ := 2
def green_faces: ℕ := 1

theorem probability_blue_or_green_face (h1: blue_faces + red_faces + green_faces = cube_faces):
  (3 + 1) / 6 = 2 / 3 :=
by
  sorry

end probability_blue_or_green_face_l175_175530


namespace total_seedlings_transferred_l175_175923

-- Define the number of seedlings planted on the first day
def seedlings_day_1 : ℕ := 200

-- Define the number of seedlings planted on the second day
def seedlings_day_2 : ℕ := 2 * seedlings_day_1

-- Define the total number of seedlings planted on both days
def total_seedlings : ℕ := seedlings_day_1 + seedlings_day_2

-- The theorem statement
theorem total_seedlings_transferred : total_seedlings = 600 := by
  -- The proof goes here
  sorry

end total_seedlings_transferred_l175_175923


namespace circle_radius_l175_175321

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2*x = 0) → ∃ r, r = 1 :=
by sorry

end circle_radius_l175_175321


namespace reciprocal_of_2022_l175_175402

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l175_175402


namespace minimum_dot_product_l175_175124

-- Definitions of points A and B
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 0)

-- Definition of condition that P lies on the line x - y + 1 = 0
def onLineP (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Definition of dot product between vectors PA and PB
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Lean 4 theorem statement
theorem minimum_dot_product (P : ℝ × ℝ) (hP : onLineP P) : 
  dotProduct P pointA pointB = 0 := 
sorry

end minimum_dot_product_l175_175124


namespace trains_clear_each_other_in_12_seconds_l175_175304

noncomputable def length_train1 : ℕ := 137
noncomputable def length_train2 : ℕ := 163
noncomputable def speed_train1_kmph : ℕ := 42
noncomputable def speed_train2_kmph : ℕ := 48

noncomputable def kmph_to_mps (v : ℕ) : ℚ := v * (5 / 18)
noncomputable def total_distance : ℕ := length_train1 + length_train2
noncomputable def relative_speed_kmph : ℕ := speed_train1_kmph + speed_train2_kmph
noncomputable def relative_speed_mps : ℚ := kmph_to_mps relative_speed_kmph

theorem trains_clear_each_other_in_12_seconds :
  (total_distance : ℚ) / relative_speed_mps = 12 := by
  sorry

end trains_clear_each_other_in_12_seconds_l175_175304


namespace gcd_204_85_l175_175406

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l175_175406


namespace Vasya_and_Petya_no_mistake_exists_l175_175566

def is_prime (n : ℕ) : Prop := ∃ d, d > 1 ∧ d < n ∧ n % d = 0

theorem Vasya_and_Petya_no_mistake_exists :
  ∃ x : ℝ, (∃ p : ℕ, is_prime p ∧ 10 * x = p) ∧ 
           (∃ q : ℕ, is_prime q ∧ 15 * x = q) :=
sorry

end Vasya_and_Petya_no_mistake_exists_l175_175566


namespace probability_of_three_5s_in_eight_rolls_l175_175768

-- Conditions
def total_outcomes : ℕ := 6 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3

-- The probability that the number 5 appears exactly three times in eight rolls of a fair die
theorem probability_of_three_5s_in_eight_rolls :
  (favorable_outcomes / total_outcomes : ℚ) = (56 / 1679616 : ℚ) :=
by
  sorry

end probability_of_three_5s_in_eight_rolls_l175_175768


namespace sum_of_three_numbers_l175_175219

theorem sum_of_three_numbers {a b c : ℝ} (h₁ : a ≤ b ∧ b ≤ c) (h₂ : b = 10)
  (h₃ : (a + b + c) / 3 = a + 20) (h₄ : (a + b + c) / 3 = c - 25) :
  a + b + c = 45 :=
by
  sorry

end sum_of_three_numbers_l175_175219


namespace average_first_6_numbers_l175_175129

theorem average_first_6_numbers (A : ℕ) (h1 : (13 * 9) = (6 * A + 45 + 6 * 7)) : A = 5 :=
by 
  -- h1 : 117 = (6 * A + 45 + 42),
  -- solving for the value of A by performing algebraic operations will prove it.
  sorry

end average_first_6_numbers_l175_175129


namespace option_C_is_always_odd_l175_175109

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem option_C_is_always_odd (k : ℤ) : is_odd (2007 + 2 * k ^ 2) :=
sorry

end option_C_is_always_odd_l175_175109


namespace sum_of_longest_altitudes_l175_175197

theorem sum_of_longest_altitudes (a b c : ℝ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) (h₄ : a^2 + b^2 = c^2) :
  a + b = 14 :=
by
  sorry

end sum_of_longest_altitudes_l175_175197


namespace original_price_of_racket_l175_175829

theorem original_price_of_racket (P : ℝ) (h : (3 / 2) * P = 90) : P = 60 :=
sorry

end original_price_of_racket_l175_175829


namespace proportionality_problem_l175_175572

noncomputable def find_x (z w : ℝ) (k : ℝ) : ℝ :=
  k / (z^(3/2) * w^2)

theorem proportionality_problem :
  ∃ k : ℝ, 
    (find_x 16 2 k = 5) ∧
    (find_x 64 4 k = 5 / 32) :=
by
  sorry

end proportionality_problem_l175_175572


namespace exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l175_175407

theorem exists_n_such_that_an_is_cube_and_bn_is_fifth_power
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), n ≥ 1 ∧ (∃ k : ℤ, a * n = k^3) ∧ (∃ l : ℤ, b * n = l^5) := 
by
  sorry

end exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l175_175407


namespace average_of_remaining_two_l175_175650

theorem average_of_remaining_two
  (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg_2_1 : (a + b) / 2 = 4.2)
  (h_avg_2_2 : (c + d) / 2 = 3.85) : 
  ((e + f) / 2) = 3.8 :=
by
  sorry

end average_of_remaining_two_l175_175650


namespace GCF_of_LCMs_l175_175019

def GCF : ℕ → ℕ → ℕ := Nat.gcd
def LCM : ℕ → ℕ → ℕ := Nat.lcm

theorem GCF_of_LCMs :
  GCF (LCM 9 21) (LCM 10 15) = 3 :=
by
  sorry

end GCF_of_LCMs_l175_175019


namespace symmetrical_circle_proof_l175_175156

open Real

-- Definition of the original circle equation
def original_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Defining the symmetrical circle equation to be proven
def symmetrical_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 5

theorem symmetrical_circle_proof :
  ∀ x y : ℝ, original_circle x y ↔ symmetrical_circle x y :=
by sorry

end symmetrical_circle_proof_l175_175156


namespace katelyn_sandwiches_difference_l175_175784

theorem katelyn_sandwiches_difference :
  ∃ (K : ℕ), K - 49 = 47 ∧ (49 + K + K / 4 = 169) := 
sorry

end katelyn_sandwiches_difference_l175_175784


namespace area_of_rectangle_at_stage_4_l175_175393

def area_at_stage (n : ℕ) : ℕ :=
  let square_area := 16
  let initial_squares := 2
  let common_difference := 2
  let total_squares := initial_squares + common_difference * (n - 1)
  total_squares * square_area

theorem area_of_rectangle_at_stage_4 :
  area_at_stage 4 = 128 :=
by
  -- computation and transformations are omitted
  sorry

end area_of_rectangle_at_stage_4_l175_175393


namespace kristy_gave_to_brother_l175_175600

def total_cookies : Nat := 22
def kristy_ate : Nat := 2
def first_friend_took : Nat := 3
def second_friend_took : Nat := 5
def third_friend_took : Nat := 5
def cookies_left : Nat := 6

theorem kristy_gave_to_brother :
  kristy_ate + first_friend_took + second_friend_took + third_friend_took = 15 ∧
  total_cookies - cookies_left - (kristy_ate + first_friend_took + second_friend_took + third_friend_took) = 1 :=
by
  sorry

end kristy_gave_to_brother_l175_175600


namespace snakes_in_breeding_ball_l175_175547

theorem snakes_in_breeding_ball (x : ℕ) (h : 3 * x + 12 = 36) : x = 8 :=
by sorry

end snakes_in_breeding_ball_l175_175547


namespace scientific_notation_of_361000000_l175_175718

theorem scientific_notation_of_361000000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (361000000 = a * 10^n) ∧ (a = 3.61) ∧ (n = 8) :=
sorry

end scientific_notation_of_361000000_l175_175718


namespace total_jellybeans_l175_175100

def nephews := 3
def nieces := 2
def jellybeans_per_child := 14
def children := nephews + nieces

theorem total_jellybeans : children * jellybeans_per_child = 70 := by
  sorry

end total_jellybeans_l175_175100


namespace pieces_of_fudge_l175_175957

def pan_length : ℝ := 27.5
def pan_width : ℝ := 17.5
def pan_height : ℝ := 2.5
def cube_side : ℝ := 2.3

def volume (l w h : ℝ) : ℝ := l * w * h

def V_pan : ℝ := volume pan_length pan_width pan_height
def V_cube : ℝ := volume cube_side cube_side cube_side

theorem pieces_of_fudge : ⌊V_pan / V_cube⌋ = 98 := by
  -- calculation can be filled in here in the actual proof
  sorry

end pieces_of_fudge_l175_175957


namespace coloring_satisfies_conditions_l175_175235

-- Define lattice points as points with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the color function
def color (p : LatticePoint) : ℕ :=
  if (p.x % 2 = 0) ∧ (p.y % 2 = 1) then 0 -- Black
  else if (p.x % 2 = 1) ∧ (p.y % 2 = 0) then 1 -- White
  else 2 -- Red

-- Define condition (1)
def infinite_lines_with_color (c : ℕ) : Prop :=
  ∀ k : ℤ, ∃ p : LatticePoint, color p = c ∧ p.x = k

-- Define condition (2)
def parallelogram_exists (A B C : LatticePoint) (wc rc bc : ℕ) : Prop :=
  (color A = wc) ∧ (color B = rc) ∧ (color C = bc) →
  ∃ D : LatticePoint, color D = rc ∧ D.x = C.x + (A.x - B.x) ∧ D.y = C.y + (A.y - B.y)

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : ℕ, ∃ p : LatticePoint, infinite_lines_with_color c) ∧
  (∀ A B C : LatticePoint, ∃ wc rc bc : ℕ, parallelogram_exists A B C wc rc bc) :=
sorry

end coloring_satisfies_conditions_l175_175235


namespace div_n_by_8_eq_2_8089_l175_175583

theorem div_n_by_8_eq_2_8089
  (n : ℕ)
  (h : n = 16^2023) :
  n / 8 = 2^8089 := by
  sorry

end div_n_by_8_eq_2_8089_l175_175583


namespace find_a_l175_175971

def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_a (a : ℝ) : 
  are_perpendicular a (a + 2) → 
  a = -1 :=
by
  intro h
  unfold are_perpendicular at h
  have h_eq : a * (a + 2) = -1 := h
  have eq_zero : a * a + 2 * a + 1 = 0 := by linarith
  sorry

end find_a_l175_175971


namespace perimeter_of_tangents_triangle_l175_175703

theorem perimeter_of_tangents_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
    (4 * a * Real.sqrt (a * b)) / (a - b) = 4 * a * (Real.sqrt (a * b) / (a - b)) := 
sorry

end perimeter_of_tangents_triangle_l175_175703


namespace vertical_asymptote_at_neg_two_over_three_l175_175489

theorem vertical_asymptote_at_neg_two_over_three : 
  ∃ x : ℝ, 6 * x + 4 = 0 ∧ x = -2 / 3 := 
by
  use -2 / 3
  sorry

end vertical_asymptote_at_neg_two_over_three_l175_175489


namespace f_2009_l175_175101

def f (x : ℝ) : ℝ := x^3 -- initial definition for x in [-1, 1]

axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom symmetric_around_1 : ∀ x : ℝ, f (1 + x) = f (1 - x)
axiom f_cubed : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3

theorem f_2009 : f 2009 = 1 := by {
  -- The body of the theorem will be filled with proof steps
  sorry
}

end f_2009_l175_175101


namespace min_value_exprB_four_min_value_exprC_four_l175_175232

noncomputable def exprB (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def exprC (x : ℝ) : ℝ := 1 / (Real.sin x)^2 + 1 / (Real.cos x)^2

theorem min_value_exprB_four : ∃ x : ℝ, exprB x = 4 := sorry

theorem min_value_exprC_four : ∃ x : ℝ, exprC x = 4 := sorry

end min_value_exprB_four_min_value_exprC_four_l175_175232


namespace axisymmetric_triangle_is_isosceles_l175_175009

-- Define a triangle and its properties
structure Triangle :=
  (a b c : ℝ) -- Triangle sides as real numbers
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)

def is_axisymmetric (T : Triangle) : Prop :=
  -- Here define what it means for a triangle to be axisymmetric
  -- This is often represented as having at least two sides equal
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

def is_isosceles (T : Triangle) : Prop :=
  -- Definition of an isosceles triangle
  (T.a = T.b ∨ T.b = T.c ∨ T.c = T.a)

-- The theorem to be proven
theorem axisymmetric_triangle_is_isosceles (T : Triangle) (h : is_axisymmetric T) : is_isosceles T :=
by {
  -- Proof would go here
  sorry
}

end axisymmetric_triangle_is_isosceles_l175_175009


namespace simplify_sin_formula_l175_175425

theorem simplify_sin_formula : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := by
  -- Conditions and values used in the proof
  sorry

end simplify_sin_formula_l175_175425


namespace find_salary_of_Thomas_l175_175826

-- Declare the variables representing the salaries of Raj, Roshan, and Thomas
variables (R S T : ℝ)

-- Given conditions as definitions
def avg_salary_Raj_Roshan : Prop := (R + S) / 2 = 4000
def avg_salary_Raj_Roshan_Thomas : Prop := (R + S + T) / 3 = 5000

-- Stating the theorem
theorem find_salary_of_Thomas
  (h1 : avg_salary_Raj_Roshan R S)
  (h2 : avg_salary_Raj_Roshan_Thomas R S T) : T = 7000 :=
by
  sorry

end find_salary_of_Thomas_l175_175826


namespace average_age_of_all_l175_175193

theorem average_age_of_all (students parents : ℕ) (student_avg parent_avg : ℚ) 
  (h_students: students = 40) 
  (h_student_avg: student_avg = 12) 
  (h_parents: parents = 60) 
  (h_parent_avg: parent_avg = 36)
  : (students * student_avg + parents * parent_avg) / (students + parents) = 26.4 :=
by
  sorry

end average_age_of_all_l175_175193


namespace total_steps_five_days_l175_175481

def steps_monday : ℕ := 150 + 170
def steps_tuesday : ℕ := 140 + 170
def steps_wednesday : ℕ := 160 + 210 + 25
def steps_thursday : ℕ := 150 + 140 + 30 + 15
def steps_friday : ℕ := 180 + 200 + 20

theorem total_steps_five_days :
  steps_monday + steps_tuesday + steps_wednesday + steps_thursday + steps_friday = 1760 :=
by
  have h1 : steps_monday = 320 := rfl
  have h2 : steps_tuesday = 310 := rfl
  have h3 : steps_wednesday = 395 := rfl
  have h4 : steps_thursday = 335 := rfl
  have h5 : steps_friday = 400 := rfl
  show 320 + 310 + 395 + 335 + 400 = 1760
  sorry

end total_steps_five_days_l175_175481


namespace smallest_square_perimeter_l175_175744

theorem smallest_square_perimeter (P_largest : ℕ) (units_apart : ℕ) (num_squares : ℕ) (H1 : P_largest = 96) (H2 : units_apart = 1) (H3 : num_squares = 8) : 
  ∃ P_smallest : ℕ, P_smallest = 40 := by
  sorry

end smallest_square_perimeter_l175_175744


namespace dustin_reads_more_pages_l175_175027

theorem dustin_reads_more_pages (dustin_rate_per_hour : ℕ) (sam_rate_per_hour : ℕ) : 
  (dustin_rate_per_hour = 75) → (sam_rate_per_hour = 24) → 
  (dustin_rate_per_hour * 40 / 60 - sam_rate_per_hour * 40 / 60 = 34) :=
by
  sorry

end dustin_reads_more_pages_l175_175027


namespace rewrite_expression_and_compute_l175_175145

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l175_175145


namespace walkway_area_l175_175578

theorem walkway_area (l w : ℕ) (walkway_width : ℕ) (total_length total_width pool_area walkway_area : ℕ)
  (hl : l = 20) 
  (hw : w = 8)
  (hww : walkway_width = 1)
  (htl : total_length = l + 2 * walkway_width)
  (htw : total_width = w + 2 * walkway_width)
  (hpa : pool_area = l * w)
  (hta : (total_length * total_width) = pool_area + walkway_area) :
  walkway_area = 60 := 
  sorry

end walkway_area_l175_175578


namespace sphere_has_circular_views_l175_175967

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l175_175967


namespace carpet_breadth_l175_175602

theorem carpet_breadth
  (b : ℝ)
  (h1 : ∀ b, ∃ l, l = 1.44 * b)
  (h2 : 4082.4 = 45 * ((1.40 * l) * (1.25 * b)))
  : b = 6.08 :=
by
  sorry

end carpet_breadth_l175_175602


namespace pizza_area_increase_l175_175260

theorem pizza_area_increase (A1 A2 r1 r2 : ℝ) (r1_eq : r1 = 7) (r2_eq : r2 = 5) (A1_eq : A1 = Real.pi * r1^2) (A2_eq : A2 = Real.pi * r2^2) :
  ((A1 - A2) / A2) * 100 = 96 := by
  sorry

end pizza_area_increase_l175_175260


namespace arithmetic_sequence_first_term_l175_175247

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_terms_int : ∀ n, ∃ k : ℤ, a n = k) 
  (ha20 : a 20 = 205) : a 1 = 91 :=
sorry

end arithmetic_sequence_first_term_l175_175247


namespace shortest_chord_through_point_l175_175175

theorem shortest_chord_through_point 
  (P : ℝ × ℝ) (hx : P = (2, 1))
  (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4 → (x, y) ∈ {p : ℝ × ℝ | (p.fst - 1)^2 + p.snd^2 = 4}) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ a * (P.1) + b * (P.2) + c = 0 := 
by
  -- proof skipped
  sorry

end shortest_chord_through_point_l175_175175


namespace inequality_always_holds_l175_175029

theorem inequality_always_holds
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = (1 - 2^x) / (1 + 2^x))
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_ineq : f (2 * a + b) + f (4 - 3 * b) > 0)
  : b - a > 2 :=
sorry

end inequality_always_holds_l175_175029


namespace log_addition_closed_l175_175910

def is_log_of_nat (n : ℝ) : Prop := ∃ k : ℕ, k > 0 ∧ n = Real.log k

theorem log_addition_closed (a b : ℝ) (ha : is_log_of_nat a) (hb : is_log_of_nat b) : is_log_of_nat (a + b) :=
by
  sorry

end log_addition_closed_l175_175910


namespace jerry_earnings_per_task_l175_175220

theorem jerry_earnings_per_task :
  ∀ (task_hours : ℕ) (daily_hours : ℕ) (days_per_week : ℕ) (total_earnings : ℕ),
    task_hours = 2 →
    daily_hours = 10 →
    days_per_week = 5 →
    total_earnings = 1400 →
    total_earnings / ((daily_hours / task_hours) * days_per_week) = 56 :=
by
  intros task_hours daily_hours days_per_week total_earnings
  intros h_task_hours h_daily_hours h_days_per_week h_total_earnings
  sorry

end jerry_earnings_per_task_l175_175220


namespace problem1_problem2_l175_175596

-- Definitions based on the given conditions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := a^2 + a * b - 1

-- Statement for problem (1)
theorem problem1 (a b : ℝ) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 4 * a^2 + 5 * a * b - 2 * a - 3 :=
by sorry

-- Statement for problem (2)
theorem problem2 (a b : ℝ) (h : ∀ a, A a b - 2 * B a b = k) : 
  b = 2 :=
by sorry

end problem1_problem2_l175_175596


namespace problem_l175_175251

def seq (a : ℕ → ℝ) := a 0 = 1 / 2 ∧ ∀ n > 0, a n = a (n - 1) + (1 / n^2) * (a (n - 1))^2

theorem problem (a : ℕ → ℝ) (n : ℕ) (h_seq : seq a) (h_n_pos : n > 0) :
  (1 / a (n - 1) - 1 / a n < 1 / n^2) ∧
  (∀ n > 0, a n < n) ∧
  (∀ n > 0, 1 / a n < 5 / 6 + 1 / (n + 1)) :=
by
  sorry

end problem_l175_175251


namespace geometric_sum_five_terms_l175_175131

theorem geometric_sum_five_terms (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_a2a4 : a 1 * a 3 = 16)
  (h_ratio : (a 3 + a 4 + a 7) / (a 0 + a 1 + a 4) = 8) :
  S 5 = 31 :=
sorry

end geometric_sum_five_terms_l175_175131


namespace probability_A_and_B_same_last_hour_l175_175335
open Classical

-- Define the problem conditions
def attraction_count : ℕ := 6
def total_scenarios : ℕ := attraction_count * attraction_count
def favorable_scenarios : ℕ := attraction_count

-- Define the probability calculation
def probability_same_attraction : ℚ := favorable_scenarios / total_scenarios

-- The proof problem statement
theorem probability_A_and_B_same_last_hour : 
  probability_same_attraction = 1 / 6 :=
sorry

end probability_A_and_B_same_last_hour_l175_175335


namespace both_hit_exactly_one_hits_at_least_one_hits_l175_175064

noncomputable def prob_A : ℝ := 0.8
noncomputable def prob_B : ℝ := 0.9

theorem both_hit : prob_A * prob_B = 0.72 := by
  sorry

theorem exactly_one_hits : prob_A * (1 - prob_B) + (1 - prob_A) * prob_B = 0.26 := by
  sorry

theorem at_least_one_hits : 1 - (1 - prob_A) * (1 - prob_B) = 0.98 := by
  sorry

end both_hit_exactly_one_hits_at_least_one_hits_l175_175064


namespace proof_l175_175237

noncomputable def proof_problem (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem proof (
  a b c : ℝ
) (h1 : (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 = 3 * (a^3 - b^3) * (b^3 - c^3) * (c^3 - a^3))
  (h2 : (a - b)^3 + (b - c)^3 + (c - a)^3 = 3 * (a - b) * (b - c) * (c - a)) :
  proof_problem a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end proof_l175_175237


namespace next_year_multiple_of_6_8_9_l175_175319

theorem next_year_multiple_of_6_8_9 (n : ℕ) (h₀ : n = 2016) (h₁ : n % 6 = 0) (h₂ : n % 8 = 0) (h₃ : n % 9 = 0) : ∃ m > n, m % 6 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 ∧ m = 2088 :=
by
  sorry

end next_year_multiple_of_6_8_9_l175_175319


namespace texas_california_plate_diff_l175_175573

def california_plates := 26^3 * 10^3
def texas_plates := 26^3 * 10^4
def plates_difference := texas_plates - california_plates

theorem texas_california_plate_diff :
  plates_difference = 158184000 :=
by sorry

end texas_california_plate_diff_l175_175573


namespace quotient_when_divided_by_8_l175_175772

theorem quotient_when_divided_by_8
  (n : ℕ)
  (h1 : n = 12 * 7 + 5)
  : (n / 8) = 11 :=
by
  -- the proof is omitted
  sorry

end quotient_when_divided_by_8_l175_175772


namespace base_three_to_decimal_l175_175397

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end base_three_to_decimal_l175_175397


namespace newspaper_pages_l175_175787

theorem newspaper_pages (p : ℕ) (h₁ : p >= 21) (h₂ : 8•2 - 1 ≤ p) (h₃ : p ≤ 8•3) : p = 28 :=
sorry

end newspaper_pages_l175_175787


namespace molecular_weight_is_correct_l175_175976

-- Define the masses of the individual isotopes
def H1 : ℕ := 1
def H2 : ℕ := 2
def O : ℕ := 16
def C : ℕ := 13
def N : ℕ := 15
def S : ℕ := 33

-- Define the molecular weight calculation
def molecular_weight : ℕ := (2 * H1) + H2 + O + C + N + S

-- The goal is to prove that the calculated molecular weight is 81
theorem molecular_weight_is_correct : molecular_weight = 81 :=
by 
  sorry

end molecular_weight_is_correct_l175_175976


namespace remainder_sand_amount_l175_175928

def total_sand : ℝ := 2548726
def bag_capacity : ℝ := 85741.2
def full_bags : ℝ := 29
def not_full_bag_sand : ℝ := 62231.2

theorem remainder_sand_amount :
  total_sand - (full_bags * bag_capacity) = not_full_bag_sand :=
by
  sorry

end remainder_sand_amount_l175_175928


namespace f_monotonic_intervals_g_greater_than_4_3_l175_175662

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := f x - Real.log x

theorem f_monotonic_intervals :
  (∀ x < -1, ∀ y < -1, x < y → f x > f y) ∧ 
  (∀ x > -1, ∀ y > -1, x < y → f x < f y) :=
sorry

theorem g_greater_than_4_3 (x : ℝ) (h : x > 0) : g x > (4 / 3) :=
sorry

end f_monotonic_intervals_g_greater_than_4_3_l175_175662


namespace train_length_proof_l175_175389

noncomputable def speed_km_per_hr : ℝ := 108
noncomputable def time_seconds : ℝ := 9
noncomputable def length_of_train : ℝ := 270
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

theorem train_length_proof : 
  (speed_km_per_hr * (km_to_m / hr_to_s) * time_seconds) = length_of_train :=
  by
  sorry

end train_length_proof_l175_175389


namespace ab_finish_job_in_15_days_l175_175275

theorem ab_finish_job_in_15_days (A B C : ℝ) (h1 : A + B + C = 1/12) (h2 : C = 1/60) : 1 / (A + B) = 15 := 
by
  sorry

end ab_finish_job_in_15_days_l175_175275


namespace prime_divisor_form_l175_175354

theorem prime_divisor_form (n : ℕ) (q : ℕ) (hq : (2^(2^n) + 1) % q = 0) (prime_q : Nat.Prime q) :
  ∃ k : ℕ, q = 2^(n+1) * k + 1 :=
sorry

end prime_divisor_form_l175_175354


namespace five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l175_175289

-- Define what "5 PM" and "10 PM" mean in hours
def five_pm: ℕ := 17
def ten_pm: ℕ := 22

-- Define function for converting from PM to 24-hour time
def pm_to_hours (n: ℕ): ℕ := n + 12

-- Define the times in minutes for comparison
def time_16_40: ℕ := 16 * 60 + 40
def time_17_20: ℕ := 17 * 60 + 20

-- Define the differences in minutes
def minutes_passed (start end_: ℕ): ℕ := end_ - start

-- Prove the equivalences
theorem five_pm_is_seventeen_hours: pm_to_hours 5 = five_pm := by 
  unfold pm_to_hours
  unfold five_pm
  rfl

theorem ten_pm_is_twenty_two_hours: pm_to_hours 10 = ten_pm := by 
  unfold pm_to_hours
  unfold ten_pm
  rfl

theorem time_difference_is_forty_minutes: minutes_passed time_16_40 time_17_20 = 40 := by 
  unfold time_16_40
  unfold time_17_20
  unfold minutes_passed
  rfl

#check five_pm_is_seventeen_hours
#check ten_pm_is_twenty_two_hours
#check time_difference_is_forty_minutes

end five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l175_175289


namespace new_solution_is_45_percent_liquid_x_l175_175162

-- Define initial conditions
def solution_y_initial_weight := 8.0 -- kilograms
def percent_liquid_x := 0.30
def percent_water := 0.70
def evaporated_water_weight := 4.0 -- kilograms
def added_solution_y_weight := 4.0 -- kilograms

-- Define the relevant quantities
def liquid_x_initial := solution_y_initial_weight * percent_liquid_x
def water_initial := solution_y_initial_weight * percent_water
def remaining_water_after_evaporation := water_initial - evaporated_water_weight

def liquid_x_after_evaporation := liquid_x_initial 
def water_after_evaporation := remaining_water_after_evaporation

def added_liquid_x := added_solution_y_weight * percent_liquid_x
def added_water := added_solution_y_weight * percent_water

def total_liquid_x := liquid_x_after_evaporation + added_liquid_x
def total_water := water_after_evaporation + added_water

def total_new_solution_weight := total_liquid_x + total_water

def new_solution_percent_liquid_x := (total_liquid_x / total_new_solution_weight) * 100

-- The theorem we want to prove
theorem new_solution_is_45_percent_liquid_x : new_solution_percent_liquid_x = 45 := by
  sorry

end new_solution_is_45_percent_liquid_x_l175_175162


namespace sum_of_numbers_l175_175753

theorem sum_of_numbers (a b c : ℝ) (h_ratio : a / 1 = b / 2 ∧ b / 2 = c / 3) (h_sum_squares : a^2 + b^2 + c^2 = 2744) : 
  a + b + c = 84 := 
sorry

end sum_of_numbers_l175_175753


namespace abs_diff_roots_eq_3_l175_175953

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l175_175953


namespace no_more_beverages_needed_l175_175694

namespace HydrationPlan

def daily_water_need := 9
def daily_juice_need := 5
def daily_soda_need := 3
def days := 60

def total_water_needed := daily_water_need * days
def total_juice_needed := daily_juice_need * days
def total_soda_needed := daily_soda_need * days

def water_already_have := 617
def juice_already_have := 350
def soda_already_have := 215

theorem no_more_beverages_needed :
  (water_already_have >= total_water_needed) ∧ 
  (juice_already_have >= total_juice_needed) ∧ 
  (soda_already_have >= total_soda_needed) :=
by 
  -- proof goes here
  sorry

end HydrationPlan

end no_more_beverages_needed_l175_175694


namespace find_a5_l175_175188

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a5 (a : ℕ → ℝ) (h_seq : geometric_sequence a) (h_a2 : a 2 = 2) (h_a8 : a 8 = 32) :
  a 5 = 8 :=
by
  sorry

end find_a5_l175_175188


namespace triangle_count_l175_175445

theorem triangle_count (a b c : ℕ) (hb : b = 2008) (hab : a ≤ b) (hbc : b ≤ c) (ht : a + b > c) : 
  ∃ n, n = 2017036 :=
by
  sorry

end triangle_count_l175_175445


namespace largest_consecutive_sum_is_nine_l175_175083

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l175_175083


namespace expand_product_l175_175689

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9 * x + 18 := 
by sorry

end expand_product_l175_175689


namespace ratio_volumes_l175_175331

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end ratio_volumes_l175_175331
