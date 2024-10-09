import Mathlib

namespace smallest_integer_condition_l1817_181707

def is_not_prime (n : Nat) : Prop := ¬ Nat.Prime n

def is_not_square (n : Nat) : Prop :=
  ∀ m : Nat, m * m ≠ n

def has_no_prime_factor_less_than (n k : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p < k → ¬ (p ∣ n)

theorem smallest_integer_condition :
  ∃ n : Nat, n > 0 ∧ is_not_prime n ∧ is_not_square n ∧ has_no_prime_factor_less_than n 70 ∧ n = 5183 :=
by {
  sorry
}

end smallest_integer_condition_l1817_181707


namespace Frank_days_to_finish_book_l1817_181788

theorem Frank_days_to_finish_book (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 22) (h2 : total_pages = 12518) : total_pages / pages_per_day = 569 := by
  sorry

end Frank_days_to_finish_book_l1817_181788


namespace root_reciprocals_identity_l1817_181787

noncomputable def cubic_roots (a b c : ℝ) : Prop :=
  (a + b + c = 12) ∧ (a * b + b * c + c * a = 20) ∧ (a * b * c = -5)

theorem root_reciprocals_identity (a b c : ℝ) (h : cubic_roots a b c) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 20.8 :=
by
  sorry

end root_reciprocals_identity_l1817_181787


namespace initial_number_of_eggs_l1817_181783

theorem initial_number_of_eggs (eggs_taken harry_eggs eggs_left initial_eggs : ℕ)
    (h1 : harry_eggs = 5)
    (h2 : eggs_left = 42)
    (h3 : initial_eggs = eggs_left + harry_eggs) : 
    initial_eggs = 47 := by
  sorry

end initial_number_of_eggs_l1817_181783


namespace chocolate_bars_in_large_box_l1817_181742

theorem chocolate_bars_in_large_box : 
  let small_boxes := 19 
  let bars_per_small_box := 25 
  let total_bars := small_boxes * bars_per_small_box 
  total_bars = 475 := by 
  -- declarations and assumptions
  let small_boxes : ℕ := 19 
  let bars_per_small_box : ℕ := 25 
  let total_bars : ℕ := small_boxes * bars_per_small_box 
  sorry

end chocolate_bars_in_large_box_l1817_181742


namespace geometric_sequence_common_ratio_l1817_181718

-- Define a sequence as a list of real numbers
def seq : List ℚ := [8, -20, 50, -125]

-- Define the common ratio of a geometric sequence
def common_ratio (l : List ℚ) : ℚ := l.head! / l.tail!.head!

-- The theorem to prove the common ratio is -5/2
theorem geometric_sequence_common_ratio :
  common_ratio seq = -5 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l1817_181718


namespace cos_double_angle_l1817_181708

theorem cos_double_angle (y0 : ℝ) (h : (1 / 3)^2 + y0^2 = 1) : 
  Real.cos (2 * Real.arccos (1 / 3)) = -7 / 9 := 
by
  sorry

end cos_double_angle_l1817_181708


namespace find_p_l1817_181758

theorem find_p (p : ℝ) (h1 : (1/2) * 15 * (3 + 15) - ((1/2) * 3 * (15 - p) + (1/2) * 15 * p) = 40) : 
  p = 12.0833 :=
by sorry

end find_p_l1817_181758


namespace original_square_side_length_l1817_181747

theorem original_square_side_length (a : ℕ) (initial_thickness final_thickness : ℕ) (side_length_reduction_factor thickness_doubling_factor : ℕ) (s : ℕ) :
  a = 3 →
  final_thickness = 16 →
  initial_thickness = 1 →
  side_length_reduction_factor = 16 →
  thickness_doubling_factor = 16 →
  s * s = side_length_reduction_factor * a * a →
  s = 12 :=
by
  intros ha hfinal_thickness hin_initial_thickness hside_length_reduction_factor hthickness_doubling_factor h_area_equiv
  sorry

end original_square_side_length_l1817_181747


namespace M_intersect_P_l1817_181751

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }
noncomputable def P : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

theorem M_intersect_P :
  M ∩ P = { y | y ≥ 1 } :=
sorry

end M_intersect_P_l1817_181751


namespace complex_quadrant_l1817_181754

open Complex

theorem complex_quadrant 
  (z : ℂ) 
  (h : (1 - I) ^ 2 / z = 1 + I) :
  z = -1 - I :=
by
  sorry

end complex_quadrant_l1817_181754


namespace find_cos_alpha_l1817_181768

theorem find_cos_alpha (α : ℝ) (h : (1 - Real.cos α) / Real.sin α = 3) : Real.cos α = -4/5 :=
by
  sorry

end find_cos_alpha_l1817_181768


namespace Ivan_walk_time_l1817_181766

variables (u v : ℝ) (T t : ℝ)

-- Define the conditions
def condition1 : Prop := T = 10 * v / u
def condition2 : Prop := T + 70 = t
def condition3 : Prop := v * t = u * T + v * (t - T + 70)

-- Problem statement: Given the conditions, prove T = 80
theorem Ivan_walk_time (h1 : condition1 u v T) (h2 : condition2 T t) (h3 : condition3 u v T t) : 
  T = 80 := by
  sorry

end Ivan_walk_time_l1817_181766


namespace full_price_tickets_revenue_l1817_181700

theorem full_price_tickets_revenue (f h d p : ℕ) 
  (h1 : f + h + d = 200) 
  (h2 : f * p + h * (p / 2) + d * (2 * p) = 5000) 
  (h3 : p = 50) : 
  f * p = 4500 :=
by
  sorry

end full_price_tickets_revenue_l1817_181700


namespace determinant_in_terms_of_roots_l1817_181782

noncomputable def determinant_3x3 (a b c : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - 1) - 1 * (1 + c) + (1 + b) * 1

theorem determinant_in_terms_of_roots (a b c s p q : ℝ)
  (h1 : a + b + c = -s)
  (h2 : a * b + a * c + b * c = p)
  (h3 : a * b * c = -q) :
  determinant_3x3 a b c = -q + p - s :=
by
  sorry

end determinant_in_terms_of_roots_l1817_181782


namespace central_angle_of_sector_l1817_181716

theorem central_angle_of_sector (r l θ : ℝ) 
  (h1 : 2 * r + l = 8) 
  (h2 : (1 / 2) * l * r = 4) 
  (h3 : θ = l / r) : θ = 2 := 
sorry

end central_angle_of_sector_l1817_181716


namespace inequality_proof_l1817_181735

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b + b * c + c * a = 1) :
  (1 / (a^2 + 1)) + (1 / (b^2 + 1)) + (1 / (c^2 + 1)) ≤ 9 / 4 :=
by
  sorry

end inequality_proof_l1817_181735


namespace smallest_lcm_of_4digit_multiples_of_5_l1817_181796

theorem smallest_lcm_of_4digit_multiples_of_5 :
  ∃ m n : ℕ, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (1000 ≤ n) ∧ (n ≤ 9999) ∧ (Nat.gcd m n = 5) ∧ (Nat.lcm m n = 201000) := 
sorry

end smallest_lcm_of_4digit_multiples_of_5_l1817_181796


namespace fraction_bounds_l1817_181759

theorem fraction_bounds (n : ℕ) (h : 0 < n) : (1 : ℚ) / 2 ≤ n / (n + 1 : ℚ) ∧ n / (n + 1 : ℚ) < 1 :=
by
  sorry

end fraction_bounds_l1817_181759


namespace cost_of_second_batch_l1817_181765

theorem cost_of_second_batch
  (C_1 C_2 : ℕ)
  (quantity_ratio cost_increase: ℕ) 
  (H1 : C_1 = 3000) 
  (H2 : C_2 = 9600) 
  (H3 : quantity_ratio = 3) 
  (H4 : cost_increase = 1)
  : (∃ x : ℕ, C_1 / x = C_2 / (x + cost_increase) / quantity_ratio) ∧ 
    (C_2 / (C_1 / 15 + cost_increase) / 3 = 16) :=
by
  sorry

end cost_of_second_batch_l1817_181765


namespace shampoo_duration_l1817_181704

theorem shampoo_duration
  (rose_shampoo : ℚ := 1/3)
  (jasmine_shampoo : ℚ := 1/4)
  (daily_usage : ℚ := 1/12) :
  (rose_shampoo + jasmine_shampoo) / daily_usage = 7 := 
by
  sorry

end shampoo_duration_l1817_181704


namespace neither_sufficient_nor_necessary_condition_l1817_181737

theorem neither_sufficient_nor_necessary_condition
  (a1 b1 c1 a2 b2 c2 : ℝ) (h1 : a1 ≠ 0) (h2 : b1 ≠ 0) (h3 : c1 ≠ 0)
  (h4 : a2 ≠ 0) (h5 : b2 ≠ 0) (h6 : c2 ≠ 0) :
  (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) ↔
  ¬(∀ x, a1 * x^2 + b1 * x + c1 > 0 ↔ a2 * x^2 + b2 * x + c2 > 0) :=
sorry

end neither_sufficient_nor_necessary_condition_l1817_181737


namespace chris_current_age_l1817_181777

def praveens_age_after_10_years (P : ℝ) : ℝ := P + 10
def praveens_age_3_years_back (P : ℝ) : ℝ := P - 3

def praveens_age_condition (P : ℝ) : Prop :=
  praveens_age_after_10_years P = 3 * praveens_age_3_years_back P

def chris_age (P : ℝ) : ℝ := (P - 4) - 2

theorem chris_current_age (P : ℝ) (h₁ : praveens_age_condition P) :
  chris_age P = 3.5 :=
sorry

end chris_current_age_l1817_181777


namespace sphere_volume_l1817_181778

theorem sphere_volume (r : ℝ) (h1 : 4 * π * r^2 = 256 * π) : 
  (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l1817_181778


namespace definite_integral_l1817_181725

open Real

theorem definite_integral : ∫ x in (0 : ℝ)..(π / 2), (x + sin x) = π^2 / 8 + 1 :=
by
  sorry

end definite_integral_l1817_181725


namespace complex_multiplication_l1817_181720

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (hi : i^2 = -1) : 
  ((3 : ℂ) - 4 * i) * ((-7 : ℂ) + 6 * i) = (3 : ℂ) + 46 * i := 
  by
    sorry

end complex_multiplication_l1817_181720


namespace a4_minus_b4_l1817_181705

theorem a4_minus_b4 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : a^4 - b^4 = -1 := by
  sorry

end a4_minus_b4_l1817_181705


namespace leah_total_coin_value_l1817_181738

variable (p n : ℕ) -- Let p be the number of pennies and n be the number of nickels

-- Leah has 15 coins consisting of pennies and nickels
axiom coin_count : p + n = 15

-- If she had three more nickels, she would have twice as many pennies as nickels
axiom conditional_equation : p = 2 * (n + 3)

-- We want to prove that the total value of Leah's coins in cents is 27
theorem leah_total_coin_value : 5 * n + p = 27 := by
  sorry

end leah_total_coin_value_l1817_181738


namespace freds_change_l1817_181709

theorem freds_change (ticket_cost : ℝ) (num_tickets : ℕ) (borrowed_movie_cost : ℝ) (total_paid : ℝ) 
  (h_ticket_cost : ticket_cost = 5.92) 
  (h_num_tickets : num_tickets = 2) 
  (h_borrowed_movie_cost : borrowed_movie_cost = 6.79) 
  (h_total_paid : total_paid = 20) : 
  total_paid - (num_tickets * ticket_cost + borrowed_movie_cost) = 1.37 := 
by 
  sorry

end freds_change_l1817_181709


namespace cos_seven_pi_over_four_proof_l1817_181786

def cos_seven_pi_over_four : Prop := (Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2)

theorem cos_seven_pi_over_four_proof : cos_seven_pi_over_four :=
by
  sorry

end cos_seven_pi_over_four_proof_l1817_181786


namespace initial_men_in_hostel_l1817_181762

theorem initial_men_in_hostel (x : ℕ) (h1 : 36 * x = 45 * (x - 50)) : x = 250 := 
  sorry

end initial_men_in_hostel_l1817_181762


namespace geometric_sequence_sum_l1817_181734

theorem geometric_sequence_sum (a₁ q : ℝ) (h1 : q ≠ 1)
    (hS2 : (a₁ * (1 - q^2)) / (1 - q) = 1)
    (hS4 : (a₁ * (1 - q^4)) / (1 - q) = 3) :
    (a₁ * (1 - q^8)) / (1 - q) = 15 := 
by
  sorry

end geometric_sequence_sum_l1817_181734


namespace farmer_land_area_l1817_181728

theorem farmer_land_area
  (A : ℝ)
  (h1 : A / 3 + A / 4 + A / 5 + 26 = A) : A = 120 :=
sorry

end farmer_land_area_l1817_181728


namespace exchange_candies_l1817_181724

-- Define the problem conditions and calculate the required values
def chocolates := 7
def caramels := 9
def exchange := 5

-- Combinatorial function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem exchange_candies (h1 : chocolates = 7) (h2 : caramels = 9) (h3 : exchange = 5) :
  binomial chocolates exchange * binomial caramels exchange = 2646 := by
  sorry

end exchange_candies_l1817_181724


namespace percent_decrease_l1817_181773

theorem percent_decrease (original_price sale_price : ℝ) 
  (h_original: original_price = 100) 
  (h_sale: sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 :=
by
  sorry

end percent_decrease_l1817_181773


namespace cube_expression_l1817_181789

theorem cube_expression (a : ℝ) (h : (a + 1/a)^2 = 5) : a^3 + 1/a^3 = 2 * Real.sqrt 5 :=
by
  sorry

end cube_expression_l1817_181789


namespace find_floors_l1817_181776

theorem find_floors (a b : ℕ) 
  (h1 : 3 * a + 4 * b = 25)
  (h2 : 2 * a + 3 * b = 18) : 
  a = 3 ∧ b = 4 := 
sorry

end find_floors_l1817_181776


namespace smallest_integer_a_l1817_181752

theorem smallest_integer_a (a : ℤ) (b : ℤ) (h1 : a < 21) (h2 : 20 ≤ b) (h3 : b < 31) (h4 : (a : ℝ) / b < 2 / 3) : 13 < a :=
sorry

end smallest_integer_a_l1817_181752


namespace negation_cube_of_every_odd_is_odd_l1817_181732

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def cube (n : ℤ) : ℤ := n * n * n

def cube_of_odd_is_odd (n : ℤ) : Prop := odd n → odd (cube n)

theorem negation_cube_of_every_odd_is_odd :
  ¬ (∀ n : ℤ, odd n → odd (cube n)) ↔ ∃ n : ℤ, odd n ∧ ¬ odd (cube n) :=
sorry

end negation_cube_of_every_odd_is_odd_l1817_181732


namespace share_difference_l1817_181785

theorem share_difference 
  (S : ℝ) -- Total sum of money
  (A B C D : ℝ) -- Shares of a, b, c, d respectively
  (h_proportion : A = 5 / 14 * S)
  (h_proportion : B = 2 / 14 * S)
  (h_proportion : C = 4 / 14 * S)
  (h_proportion : D = 3 / 14 * S)
  (h_d_share : D = 1500) :
  C - D = 500 :=
sorry

end share_difference_l1817_181785


namespace problem_arith_seq_l1817_181743

variables {a : ℕ → ℝ} (d : ℝ)
def is_arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem problem_arith_seq (h_arith : is_arithmetic_sequence a) 
  (h_condition : a 1 + a 6 + a 11 = 3) 
  : a 3 + a 9 = 2 :=
sorry

end problem_arith_seq_l1817_181743


namespace spending_ratio_l1817_181726

theorem spending_ratio 
  (lisa_tshirts : Real)
  (lisa_jeans : Real)
  (lisa_coats : Real)
  (carly_tshirts : Real)
  (carly_jeans : Real)
  (carly_coats : Real)
  (total_spent : Real)
  (hl1 : lisa_tshirts = 40)
  (hl2 : lisa_jeans = lisa_tshirts / 2)
  (hl3 : lisa_coats = 2 * lisa_tshirts)
  (hc1 : carly_tshirts = lisa_tshirts / 4)
  (hc2 : carly_coats = lisa_coats / 4)
  (htotal : total_spent = lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats)
  (h_total_spent_val : total_spent = 230) :
  carly_jeans = 3 * lisa_jeans :=
by
  -- Placeholder for theorem's proof
  sorry

end spending_ratio_l1817_181726


namespace max_playground_area_l1817_181756

/-- Mara is setting up a fence around a rectangular playground with given constraints.
    We aim to prove that the maximum area the fence can enclose is 10000 square feet. --/
theorem max_playground_area (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 400) 
  (h2 : l ≥ 100) 
  (h3 : w ≥ 50) : 
  l * w ≤ 10000 :=
sorry

end max_playground_area_l1817_181756


namespace perpendicular_vectors_X_value_l1817_181715

open Real

-- Define vectors a and b, and their perpendicularity condition
def vector_a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def vector_b : ℝ × ℝ := (1, 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The theorem statement
theorem perpendicular_vectors_X_value (x : ℝ) 
  (h : dot_product (vector_a x) vector_b = 0) : 
    x = -2 / 3 :=
by sorry

end perpendicular_vectors_X_value_l1817_181715


namespace total_animals_sighted_l1817_181794

theorem total_animals_sighted (lions_saturday elephants_saturday buffaloes_sunday leopards_sunday rhinos_monday warthogs_monday : ℕ)
(hlions_saturday : lions_saturday = 3)
(helephants_saturday : elephants_saturday = 2)
(hbuffaloes_sunday : buffaloes_sunday = 2)
(hleopards_sunday : leopards_sunday = 5)
(hrhinos_monday : rhinos_monday = 5)
(hwarthogs_monday : warthogs_monday = 3) :
  lions_saturday + elephants_saturday + buffaloes_sunday + leopards_sunday + rhinos_monday + warthogs_monday = 20 :=
by
  -- This is where the proof will be, but we are skipping the proof here.
  sorry

end total_animals_sighted_l1817_181794


namespace digits_right_of_decimal_l1817_181760

theorem digits_right_of_decimal : 
  ∃ n : ℕ, (3^6 : ℚ) / ((6^4 : ℚ) * 625) = 9 * 10^(-4 : ℤ) ∧ n = 4 := 
by 
  sorry

end digits_right_of_decimal_l1817_181760


namespace sum_of_two_numbers_l1817_181780

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y = 12) (h2 : 1 / x = 3 * (1 / y)) : x + y = 8 :=
by
  sorry

end sum_of_two_numbers_l1817_181780


namespace side_length_of_square_l1817_181781

theorem side_length_of_square (s : ℝ) (h : s^2 = 100) : s = 10 := 
sorry

end side_length_of_square_l1817_181781


namespace smallest_positive_integer_exists_l1817_181710

theorem smallest_positive_integer_exists :
  ∃ (n : ℕ), n > 0 ∧ (∃ (k m : ℕ), n = 5 * k + 3 ∧ n = 12 * m) ∧ n = 48 :=
by
  sorry

end smallest_positive_integer_exists_l1817_181710


namespace elena_novel_pages_l1817_181745

theorem elena_novel_pages
  (days_vacation : ℕ)
  (pages_first_two_days : ℕ)
  (pages_next_three_days : ℕ)
  (pages_last_day : ℕ)
  (h1 : days_vacation = 6)
  (h2 : pages_first_two_days = 2 * 42)
  (h3 : pages_next_three_days = 3 * 35)
  (h4 : pages_last_day = 15) :
  pages_first_two_days + pages_next_three_days + pages_last_day = 204 := by
  sorry

end elena_novel_pages_l1817_181745


namespace max_value_x_y_squared_l1817_181772

theorem max_value_x_y_squared (x y : ℝ) (h : 3 * (x^3 + y^3) = x + y^2) : x + y^2 ≤ 1/3 :=
sorry

end max_value_x_y_squared_l1817_181772


namespace intersection_P_Q_l1817_181722

def P : Set ℤ := { x | -4 ≤ x ∧ x ≤ 2 }

def Q : Set ℤ := { x | -3 < x ∧ x < 1 }

theorem intersection_P_Q : P ∩ Q = {-2, -1, 0} :=
by
  sorry

end intersection_P_Q_l1817_181722


namespace mandy_gets_15_pieces_l1817_181757

def initial_pieces : ℕ := 75
def michael_takes (pieces : ℕ) : ℕ := pieces / 3
def paige_takes (pieces : ℕ) : ℕ := (pieces - michael_takes pieces) / 2
def ben_takes (pieces : ℕ) : ℕ := 2 * (pieces - michael_takes pieces - paige_takes pieces) / 5
def mandy_takes (pieces : ℕ) : ℕ := pieces - michael_takes pieces - paige_takes pieces - ben_takes pieces

theorem mandy_gets_15_pieces :
  mandy_takes initial_pieces = 15 :=
by
  sorry

end mandy_gets_15_pieces_l1817_181757


namespace value_of_x2_plus_9y2_l1817_181769

theorem value_of_x2_plus_9y2 (x y : ℝ) (h1 : x + 3 * y = 9) (h2 : x * y = -15) : x^2 + 9 * y^2 = 171 :=
sorry

end value_of_x2_plus_9y2_l1817_181769


namespace solution_set_of_inequality_l1817_181713

open Set Real

theorem solution_set_of_inequality :
  {x : ℝ | sqrt (x + 3) > 3 - x} = {x : ℝ | 1 < x} ∪ {x : ℝ | x ≥ 3} := by
  sorry

end solution_set_of_inequality_l1817_181713


namespace seashells_at_end_of_month_l1817_181770

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l1817_181770


namespace find_prices_maximize_profit_l1817_181703

-- Definition of conditions
def sales_eq1 (m n : ℝ) : Prop := 150 * m + 100 * n = 1450
def sales_eq2 (m n : ℝ) : Prop := 200 * m + 50 * n = 1100

def profit_function (x : ℕ) : ℝ := -2 * x + 1500
def range_x (x : ℕ) : Prop := 375 ≤ x ∧ x ≤ 500

-- Theorem to prove the unit prices
theorem find_prices : ∃ m n : ℝ, sales_eq1 m n ∧ sales_eq2 m n ∧ m = 3 ∧ n = 10 := 
sorry

-- Theorem to prove the profit function and maximum profit
theorem maximize_profit : ∃ (x : ℕ) (W : ℝ), range_x x ∧ W = profit_function x ∧ W = 750 :=
sorry

end find_prices_maximize_profit_l1817_181703


namespace tom_initial_money_l1817_181748

-- Defining the given values
def super_nintendo_value : ℝ := 150
def store_percentage : ℝ := 0.80
def nes_price : ℝ := 160
def game_value : ℝ := 30
def change_received : ℝ := 10

-- Calculate the credit received for the Super Nintendo
def credit_received := store_percentage * super_nintendo_value

-- Calculate the remaining amount Tom needs to pay for the NES after using the credit
def remaining_amount := nes_price - credit_received

-- Calculate the total amount Tom needs to pay, including the game value
def total_amount_needed := remaining_amount + game_value

-- Proving that the initial money Tom gave is $80
theorem tom_initial_money : total_amount_needed + change_received = 80 :=
by
    sorry

end tom_initial_money_l1817_181748


namespace exists_triangle_free_not_4_colorable_l1817_181727

/-- Define a graph as a structure with vertices and edges. -/
structure Graph (V : Type*) :=
  (adj : V → V → Prop)
  (symm : ∀ x y, adj x y → adj y x)
  (irreflexive : ∀ x, ¬adj x x)

/-- A definition of triangle-free graph. -/
def triangle_free {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c : V), G.adj a b → G.adj b c → G.adj c a → false

/-- A definition that a graph cannot be k-colored. -/
def not_k_colorable {V : Type*} (G : Graph V) (k : ℕ) : Prop :=
  ¬∃ (f : V → ℕ), (∀ (v : V), f v < k) ∧ (∀ (v w : V), G.adj v w → f v ≠ f w)

/-- There exists a triangle-free graph that is not 4-colorable. -/
theorem exists_triangle_free_not_4_colorable : ∃ (V : Type*) (G : Graph V), triangle_free G ∧ not_k_colorable G 4 := 
sorry

end exists_triangle_free_not_4_colorable_l1817_181727


namespace perfume_price_reduction_l1817_181763

theorem perfume_price_reduction : 
  let original_price := 1200
  let increased_price := original_price * (1 + 0.10)
  let final_price := increased_price * (1 - 0.15)
  original_price - final_price = 78 := 
by
  sorry

end perfume_price_reduction_l1817_181763


namespace abc_sum_l1817_181774

def f (x : Int) (a b c : Nat) : Int :=
  if x > 0 then a * x + 4
  else if x = 0 then a * b
  else b * x + c

theorem abc_sum :
  ∃ a b c : Nat, 
  f 3 a b c = 7 ∧ 
  f 0 a b c = 6 ∧ 
  f (-3) a b c = -15 ∧ 
  a + b + c = 10 :=
by
  sorry

end abc_sum_l1817_181774


namespace largest_angle_in_triangle_l1817_181717

theorem largest_angle_in_triangle (A B C : ℝ) 
  (h_sum : A + B = 126) 
  (h_diff : B = A + 40) 
  (h_triangle : A + B + C = 180) : max A (max B C) = 83 := 
by
  sorry

end largest_angle_in_triangle_l1817_181717


namespace correct_answer_l1817_181739

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem correct_answer : P ∩ Q ⊆ P := by
  sorry

end correct_answer_l1817_181739


namespace apple_count_l1817_181744

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l1817_181744


namespace lucas_fraction_of_money_left_l1817_181706

theorem lucas_fraction_of_money_left (m p n : ℝ) (h1 : (1 / 4) * m = (1 / 2) * n * p) :
  (m - n * p) / m = 1 / 2 :=
by 
  -- Sorry is used to denote that we are skipping the proof
  sorry

end lucas_fraction_of_money_left_l1817_181706


namespace sum_eq_expected_l1817_181701

noncomputable def complex_sum : Complex :=
  12 * Complex.exp (Complex.I * 3 * Real.pi / 13) + 12 * Complex.exp (Complex.I * 6 * Real.pi / 13)

noncomputable def expected_value : Complex :=
  24 * Real.cos (Real.pi / 13) * Complex.exp (Complex.I * 9 * Real.pi / 26)

theorem sum_eq_expected :
  complex_sum = expected_value :=
by
  sorry

end sum_eq_expected_l1817_181701


namespace total_balloons_l1817_181711

-- Define the number of balloons Alyssa, Sandy, and Sally have.
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Theorem stating that the total number of balloons is 104.
theorem total_balloons : alyssa_balloons + sandy_balloons + sally_balloons = 104 :=
by
  -- Proof is omitted for the purpose of this task.
  sorry

end total_balloons_l1817_181711


namespace number_square_roots_l1817_181779

theorem number_square_roots (a x : ℤ) (h1 : x = (2 * a + 3) ^ 2) (h2 : x = (a - 18) ^ 2) : x = 169 :=
by 
  sorry

end number_square_roots_l1817_181779


namespace least_non_lucky_multiple_of_7_correct_l1817_181784

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

def least_non_lucky_multiple_of_7 : ℕ :=
  14

theorem least_non_lucky_multiple_of_7_correct : 
  ¬ is_lucky 14 ∧ ∀ m, m < 14 → m % 7 = 0 → ¬ ¬ is_lucky m :=
by
  sorry

end least_non_lucky_multiple_of_7_correct_l1817_181784


namespace no_intersection_points_l1817_181733

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

-- State the theorem
theorem no_intersection_points : ∀ x y : ℝ, f1 x = y ∧ f2 x = y → false := by
  sorry

end no_intersection_points_l1817_181733


namespace solve_equation_l1817_181755

theorem solve_equation (x : ℝ) (h : x ≠ -2) : (x^2 + x + 1) / (x + 2) = x + 1 → x = -1 / 2 := 
by
  intro h1
  sorry

end solve_equation_l1817_181755


namespace proof_problem_l1817_181723

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∀ x, (a * x^2 + b * x + 2 > 0) ↔ (x ∈ Set.Ioo (-1/2 : ℝ) (1/3 : ℝ))) 

theorem proof_problem (a b : ℝ) (h : problem_statement a b) : a + b = -14 :=
sorry

end proof_problem_l1817_181723


namespace intersection_M_N_l1817_181740

noncomputable def set_M : Set ℚ := {α | ∃ k : ℤ, α = k * 90 - 36}
noncomputable def set_N : Set ℚ := {α | -180 < α ∧ α < 180}

theorem intersection_M_N : set_M ∩ set_N = {-36, 54, 144, -126} := by
  sorry

end intersection_M_N_l1817_181740


namespace min_value_quadratic_l1817_181791

theorem min_value_quadratic :
  ∃ (x y : ℝ), (∀ (a b : ℝ), (3*a^2 + 4*a*b + 2*b^2 - 6*a - 8*b + 6 ≥ 0)) ∧ 
  (3*x^2 + 4*x*y + 2*y^2 - 6*x - 8*y + 6 = 0) := 
sorry

end min_value_quadratic_l1817_181791


namespace expand_polynomial_expression_l1817_181798

theorem expand_polynomial_expression (x : ℝ) : 
  (x + 6) * (x + 8) * (x - 3) = x^3 + 11 * x^2 + 6 * x - 144 :=
by
  sorry

end expand_polynomial_expression_l1817_181798


namespace compute_g3_l1817_181795

def g (x : ℤ) : ℤ := 7 * x - 3

theorem compute_g3: g (g (g 3)) = 858 :=
by
  sorry

end compute_g3_l1817_181795


namespace tim_income_percentage_less_than_juan_l1817_181741

variables (M T J : ℝ)

theorem tim_income_percentage_less_than_juan 
  (h1 : M = 1.60 * T)
  (h2 : M = 0.80 * J) : 
  100 - 100 * (T / J) = 50 :=
by
  sorry

end tim_income_percentage_less_than_juan_l1817_181741


namespace arrangement_is_correct_l1817_181767

-- Definition of adjacency in a 3x3 matrix.
def adjacent (i j k l : Nat) : Prop := 
  (i = k ∧ j = l + 1) ∨ (i = k ∧ j = l - 1) ∨ -- horizontal adjacency
  (i = k + 1 ∧ j = l) ∨ (i = k - 1 ∧ j = l) ∨ -- vertical adjacency
  (i = k + 1 ∧ j = l + 1) ∨ (i = k - 1 ∧ j = l - 1) ∨ -- diagonal adjacency
  (i = k + 1 ∧ j = l - 1) ∨ (i = k - 1 ∧ j = l + 1)   -- diagonal adjacency

-- Definition to check if two numbers share a common divisor greater than 1.
def coprime (x y : Nat) : Prop := Nat.gcd x y = 1

-- The arrangement of the numbers in the 3x3 grid.
def grid := 
  [ [8, 9, 10],
    [5, 7, 11],
    [6, 13, 12] ]

-- Ensure adjacents numbers are coprime
def correctArrangement :=
  (coprime grid[0][0] grid[0][1]) ∧ (coprime grid[0][1] grid[0][2]) ∧
  (coprime grid[1][0] grid[1][1]) ∧ (coprime grid[1][1] grid[1][2]) ∧
  (coprime grid[2][0] grid[2][1]) ∧ (coprime grid[2][1] grid[2][2]) ∧
  (adjacent 0 0 1 1 → coprime grid[0][0] grid[1][1]) ∧
  (adjacent 0 2 1 1 → coprime grid[0][2] grid[1][1]) ∧
  (adjacent 1 0 2 1 → coprime grid[1][0] grid[2][1]) ∧
  (adjacent 1 2 2 1 → coprime grid[1][2] grid[2][1]) ∧
  (coprime grid[0][1] grid[1][0]) ∧ (coprime grid[0][1] grid[1][2]) ∧
  (coprime grid[2][1] grid[1][0]) ∧ (coprime grid[2][1] grid[1][2])

-- Statement to be proven
theorem arrangement_is_correct : correctArrangement := 
  sorry

end arrangement_is_correct_l1817_181767


namespace squirrels_acorns_l1817_181731

theorem squirrels_acorns (squirrels : ℕ) (total_collected : ℕ) (acorns_needed_per_squirrel : ℕ) (total_needed : ℕ) (acorns_still_needed : ℕ) : 
  squirrels = 5 → 
  total_collected = 575 → 
  acorns_needed_per_squirrel = 130 → 
  total_needed = squirrels * acorns_needed_per_squirrel →
  acorns_still_needed = total_needed - total_collected →
  acorns_still_needed / squirrels = 15 :=
by
  sorry

end squirrels_acorns_l1817_181731


namespace total_money_raised_l1817_181730

def tickets_sold : ℕ := 25
def ticket_price : ℕ := 2
def num_15_donations : ℕ := 2
def donation_15_amount : ℕ := 15
def donation_20_amount : ℕ := 20

theorem total_money_raised : 
  tickets_sold * ticket_price + num_15_donations * donation_15_amount + donation_20_amount = 100 := 
by sorry

end total_money_raised_l1817_181730


namespace number_of_terms_before_4_appears_l1817_181797

-- Define the parameters of the arithmetic sequence
def first_term : ℤ := 100
def common_difference : ℤ := -4
def nth_term (n : ℕ) : ℤ := first_term + common_difference * (n - 1)

-- Problem: Prove that the number of terms before the number 4 appears in this sequence is 24.
theorem number_of_terms_before_4_appears :
  ∃ n : ℕ, nth_term n = 4 ∧ n - 1 = 24 := 
by
  sorry

end number_of_terms_before_4_appears_l1817_181797


namespace shirt_price_after_discount_l1817_181771

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end shirt_price_after_discount_l1817_181771


namespace intersection_points_l1817_181790

noncomputable def y1 := 2*((7 + Real.sqrt 61)/2)^2 - 3*((7 + Real.sqrt 61)/2) + 1
noncomputable def y2 := 2*((7 - Real.sqrt 61)/2)^2 - 3*((7 - Real.sqrt 61)/2) + 1

theorem intersection_points :
  ∃ (x y : ℝ), (y = 2*x^2 - 3*x + 1) ∧ (y = x^2 + 4*x + 4) ∧
                ((x = (7 + Real.sqrt 61)/2 ∧ y = y1) ∨
                 (x = (7 - Real.sqrt 61)/2 ∧ y = y2)) :=
by
  sorry

end intersection_points_l1817_181790


namespace combined_payment_is_correct_l1817_181702

-- Define the conditions for discounts
def discount_scheme (amount : ℕ) : ℕ :=
  if amount ≤ 100 then amount
  else if amount ≤ 300 then (amount * 90) / 100
  else (amount * 80) / 100

-- Given conditions for Wang Bo's purchases
def first_purchase := 80
def second_purchase_with_discount_applied := 252

-- Two possible original amounts for the second purchase
def possible_second_purchases : Set ℕ :=
  { x | discount_scheme x = second_purchase_with_discount_applied }

-- Total amount to be considered for combined buys with discounts
def total_amount_paid := {x + first_purchase | x ∈ possible_second_purchases}

-- discount applied on the combined amount
def discount_applied_amount (combined : ℕ) : ℕ :=
  discount_scheme combined

-- Prove the combined amount is either 288 or 316
theorem combined_payment_is_correct :
  ∃ combined ∈ total_amount_paid, discount_applied_amount combined = 288 ∨ discount_applied_amount combined = 316 :=
sorry

end combined_payment_is_correct_l1817_181702


namespace triangle_leg_length_l1817_181775

theorem triangle_leg_length (perimeter_square : ℝ)
                            (base_triangle : ℝ)
                            (area_equality : ∃ (side_square : ℝ) (height_triangle : ℝ),
                                4 * side_square = perimeter_square ∧
                                side_square * side_square = (1/2) * base_triangle * height_triangle)
                            : ∃ (y : ℝ), y = 22.5 :=
by
  -- Placeholder proof
  sorry

end triangle_leg_length_l1817_181775


namespace sum_last_two_digits_l1817_181764

theorem sum_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) :
  (a ^ 30 + b ^ 30) % 100 = 0 := 
by
  sorry

end sum_last_two_digits_l1817_181764


namespace words_per_page_l1817_181736

theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p % 221 = 207) : p = 100 :=
sorry

end words_per_page_l1817_181736


namespace fifth_term_arithmetic_seq_l1817_181719

theorem fifth_term_arithmetic_seq (a d : ℤ) 
  (h10th : a + 9 * d = 23) 
  (h11th : a + 10 * d = 26) 
  : a + 4 * d = 8 :=
sorry

end fifth_term_arithmetic_seq_l1817_181719


namespace student_correct_answers_l1817_181712

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 79) : C = 93 :=
by
  sorry

end student_correct_answers_l1817_181712


namespace problem_1_problem_2_l1817_181793

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem problem_1 (x : ℝ) : (g x ≥ abs (x - 1)) ↔ (x ≥ 2/3) :=
by
  sorry

theorem problem_2 (c : ℝ) : (∀ x, abs (g x) - c ≥ abs (x - 1)) → (c ≤ -1/2) :=
by
  sorry

end problem_1_problem_2_l1817_181793


namespace triangle_perimeter_l1817_181750

theorem triangle_perimeter
  (a b : ℕ) (c : ℕ) 
  (h_side1 : a = 3)
  (h_side2 : b = 4)
  (h_third_side : c^2 - 13 * c + 40 = 0)
  (h_valid_triangle : (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
  a + b + c = 12 :=
by {
  sorry
}

end triangle_perimeter_l1817_181750


namespace thursday_loaves_baked_l1817_181749

theorem thursday_loaves_baked (wednesday friday saturday sunday monday : ℕ) (p1 : wednesday = 5) (p2 : friday = 10) (p3 : saturday = 14) (p4 : sunday = 19) (p5 : monday = 25) : 
  ∃ thursday : ℕ, thursday = 11 := 
by 
  sorry

end thursday_loaves_baked_l1817_181749


namespace sum_and_product_of_roots_l1817_181799

-- Define the equation in terms of |x|
def equation (x : ℝ) : ℝ := |x|^3 - |x|^2 - 6 * |x| + 8

-- Lean statement to prove the sum and product of the roots
theorem sum_and_product_of_roots :
  (∀ x, equation x = 0 → (∃ L : List ℝ, L.sum = 0 ∧ L.prod = 16 ∧ ∀ y ∈ L, equation y = 0)) := 
sorry

end sum_and_product_of_roots_l1817_181799


namespace expected_worth_of_coin_flip_l1817_181729

theorem expected_worth_of_coin_flip :
  let p_heads := 2 / 3
  let p_tails := 1 / 3
  let gain_heads := 5
  let loss_tails := -9
  (p_heads * gain_heads) + (p_tails * loss_tails) = 1 / 3 :=
by
  -- Proof will be here
  sorry

end expected_worth_of_coin_flip_l1817_181729


namespace eq_pow_four_l1817_181714

theorem eq_pow_four (a b : ℝ) (h : a = b + 1) : a^4 = b^4 → a = 1/2 ∧ b = -1/2 :=
by
  sorry

end eq_pow_four_l1817_181714


namespace polynomial_identity_solution_l1817_181761

theorem polynomial_identity_solution (P : Polynomial ℝ) :
  (∀ x : ℝ, x * P.eval (x - 1) = (x - 2) * P.eval x) ↔ (∃ a : ℝ, P = Polynomial.C a * (Polynomial.X ^ 2 - Polynomial.X)) :=
by
  sorry

end polynomial_identity_solution_l1817_181761


namespace share_of_B_l1817_181753

noncomputable def B_share (B_investment A_investment C_investment D_investment total_profit : ℝ) : ℝ :=
  (B_investment / (A_investment + B_investment + C_investment + D_investment)) * total_profit

theorem share_of_B (B_investment total_profit : ℝ) (hA : A_investment = 3 * B_investment) 
  (hC : C_investment = (3 / 2) * B_investment) 
  (hD : D_investment = (3 / 2) * B_investment) 
  (h_profit : total_profit = 19900) :
  B_share B_investment A_investment C_investment D_investment total_profit = 2842.86 :=
by
  rw [B_share, hA, hC, hD, h_profit]
  sorry

end share_of_B_l1817_181753


namespace find_smallest_angle_l1817_181746

theorem find_smallest_angle (x : ℝ) (h1 : Real.tan (2 * x) + Real.tan (3 * x) = 1) :
  x = 9 * Real.pi / 180 :=
by
  sorry

end find_smallest_angle_l1817_181746


namespace solution_system_equations_l1817_181792

theorem solution_system_equations :
  ∀ (x y : ℝ) (k n : ℤ),
    (4 * (Real.cos x) ^ 2 - 4 * Real.cos x * (Real.cos (6 * x)) ^ 2 + (Real.cos (6 * x)) ^ 2 = 0) ∧
    (Real.sin x = Real.cos y) →
    (∃ k n : ℤ, (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = (Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = (Real.pi / 3) + 2 * Real.pi * k ∧ y = -(Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = (5 * Real.pi / 6) + 2 * Real.pi * n) ∨
                 (x = -(Real.pi / 3) + 2 * Real.pi * k ∧ y = -(5 * Real.pi / 6) + 2 * Real.pi * n)) :=
by
  sorry

end solution_system_equations_l1817_181792


namespace minimum_of_a_plus_b_l1817_181721

theorem minimum_of_a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 4/b = 1) : a + b ≥ 9 :=
by sorry

end minimum_of_a_plus_b_l1817_181721
