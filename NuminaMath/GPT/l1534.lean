import Mathlib

namespace NUMINAMATH_GPT_quadruple_perimeter_l1534_153473

variable (s : ℝ) -- side length of the original square
variable (x : ℝ) -- perimeter of the original square
variable (P_new : ℝ) -- new perimeter after side length is quadrupled

theorem quadruple_perimeter (h1 : x = 4 * s) (h2 : P_new = 4 * (4 * s)) : P_new = 4 * x := 
by sorry

end NUMINAMATH_GPT_quadruple_perimeter_l1534_153473


namespace NUMINAMATH_GPT_cake_pieces_l1534_153497

theorem cake_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) 
  (pan_dim : pan_length = 24 ∧ pan_width = 15) 
  (piece_dim : piece_length = 3 ∧ piece_width = 2) : 
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
sorry

end NUMINAMATH_GPT_cake_pieces_l1534_153497


namespace NUMINAMATH_GPT_ball_hits_ground_at_time_l1534_153493

theorem ball_hits_ground_at_time :
  ∀ (t : ℝ), (-18 * t^2 + 30 * t + 60 = 0) ↔ (t = (5 + Real.sqrt 145) / 6) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_at_time_l1534_153493


namespace NUMINAMATH_GPT_problem_statement_l1534_153405

-- Define the functions
def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x - 5

-- Define the main theorem statement
theorem problem_statement : f (g (-2)) = 81 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1534_153405


namespace NUMINAMATH_GPT_quadratic_one_root_greater_than_two_other_less_than_two_l1534_153452

theorem quadratic_one_root_greater_than_two_other_less_than_two (m : ℝ) :
  (∀ x y : ℝ, x^2 + (2 * m - 3) * x + m - 150 = 0 ∧ x > 2 ∧ y < 2) →
  m > 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_one_root_greater_than_two_other_less_than_two_l1534_153452


namespace NUMINAMATH_GPT_xyz_neg_of_ineq_l1534_153411

variables {x y z : ℝ}

theorem xyz_neg_of_ineq
  (h1 : 2 * x - y < 0)
  (h2 : 3 * y - 2 * z < 0)
  (h3 : 4 * z - 3 * x < 0) :
  x < 0 ∧ y < 0 ∧ z < 0 :=
sorry

end NUMINAMATH_GPT_xyz_neg_of_ineq_l1534_153411


namespace NUMINAMATH_GPT_tracy_dog_food_l1534_153482

theorem tracy_dog_food
(f : ℕ) (c : ℝ) (m : ℕ) (d : ℕ)
(hf : f = 4) (hc : c = 2.25) (hm : m = 3) (hd : d = 2) :
  (f * c / m) / d = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_tracy_dog_food_l1534_153482


namespace NUMINAMATH_GPT_fraction_not_collapsing_l1534_153494

variable (total_homes : ℕ)
variable (termite_ridden_fraction collapsing_fraction : ℚ)
variable (h : termite_ridden_fraction = 1 / 3)
variable (c : collapsing_fraction = 7 / 10)

theorem fraction_not_collapsing : 
  (termite_ridden_fraction - (termite_ridden_fraction * collapsing_fraction)) = 1 / 10 := 
by 
  rw [h, c]
  sorry

end NUMINAMATH_GPT_fraction_not_collapsing_l1534_153494


namespace NUMINAMATH_GPT_multiple_of_age_is_3_l1534_153416

def current_age : ℕ := 9
def age_six_years_ago : ℕ := 3
def age_multiple (current : ℕ) (previous : ℕ) : ℕ := current / previous

theorem multiple_of_age_is_3 : age_multiple current_age age_six_years_ago = 3 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_age_is_3_l1534_153416


namespace NUMINAMATH_GPT_ratio_pr_l1534_153483

variable (p q r s : ℚ)

def ratio_pq (p q : ℚ) : Prop := p / q = 5 / 4
def ratio_rs (r s : ℚ) : Prop := r / s = 4 / 3
def ratio_sq (s q : ℚ) : Prop := s / q = 1 / 5

theorem ratio_pr (hpq : ratio_pq p q) (hrs : ratio_rs r s) (hsq : ratio_sq s q) : p / r = 75 / 16 := by
  sorry

end NUMINAMATH_GPT_ratio_pr_l1534_153483


namespace NUMINAMATH_GPT_invitation_methods_l1534_153459

-- Definitions
def num_ways_invite_6_out_of_10 : ℕ := Nat.choose 10 6
def num_ways_both_A_and_B : ℕ := Nat.choose 8 4

-- Theorem statement
theorem invitation_methods : num_ways_invite_6_out_of_10 - num_ways_both_A_and_B = 140 :=
by
  -- Proof should be provided here
  sorry

end NUMINAMATH_GPT_invitation_methods_l1534_153459


namespace NUMINAMATH_GPT_beetle_speed_l1534_153429

theorem beetle_speed
  (distance_ant : ℝ )
  (time_minutes : ℝ)
  (distance_beetle : ℝ) 
  (distance_percent_less : ℝ)
  (time_hours : ℝ)
  (beetle_speed_kmh : ℝ)
  (h1 : distance_ant = 600)
  (h2 : time_minutes = 10)
  (h3 : time_hours = time_minutes / 60)
  (h4 : distance_percent_less = 0.25)
  (h5 : distance_beetle = distance_ant * (1 - distance_percent_less))
  (h6 : beetle_speed_kmh = distance_beetle / time_hours) : 
  beetle_speed_kmh = 2.7 :=
by 
  sorry

end NUMINAMATH_GPT_beetle_speed_l1534_153429


namespace NUMINAMATH_GPT_Yan_distance_ratio_l1534_153487

theorem Yan_distance_ratio (d x : ℝ) (v : ℝ) (h1 : d > 0) (h2 : x > 0) (h3 : x < d)
  (h4 : 7 * (d - x) = x + d) : 
  x / (d - x) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_Yan_distance_ratio_l1534_153487


namespace NUMINAMATH_GPT_problem_solution_l1534_153422

theorem problem_solution (x y : ℝ) (h₁ : x + Real.cos y = 2010) (h₂ : x + 2010 * Real.sin y = 2011) (h₃ : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := 
sorry

end NUMINAMATH_GPT_problem_solution_l1534_153422


namespace NUMINAMATH_GPT_maximize_garden_area_l1534_153437

def optimal_dimensions_area : Prop :=
  let l := 100
  let w := 60
  let area := 6000
  (2 * l) + (2 * w) = 320 ∧ l >= 100 ∧ (l * w) = area

theorem maximize_garden_area : optimal_dimensions_area := by
  sorry

end NUMINAMATH_GPT_maximize_garden_area_l1534_153437


namespace NUMINAMATH_GPT_f_max_a_zero_f_zero_range_l1534_153489

-- Part 1: Proving the maximum value when a = 0
theorem f_max_a_zero : ∀ (f : ℝ → ℝ) (x : ℝ),
  (f x = (-1 / x) - Real.log x) ∧ x > 0 → 
  ∃ x_max : ℝ, f x_max = -1 ∧ 
  (∀ x > 0, f x ≤ -1) := 
sorry

-- Part 2: Proving the range of a for exactly one zero of f(x)
theorem f_zero_range (a : ℝ) : (0 < a) → 
  ∀ (f : ℝ → ℝ) (x : ℝ), 
  (f x = a * x - 1 / x - (a + 1) * Real.log x) ∧ x > 0 →
  ∃! (x_zero : ℝ), f x_zero = 0 :=
sorry

end NUMINAMATH_GPT_f_max_a_zero_f_zero_range_l1534_153489


namespace NUMINAMATH_GPT_baseball_league_games_l1534_153413

theorem baseball_league_games (n m : ℕ) (h : 3 * n + 4 * m = 76) (h1 : n > 2 * m) (h2 : m > 4) : n = 16 :=
by 
  sorry

end NUMINAMATH_GPT_baseball_league_games_l1534_153413


namespace NUMINAMATH_GPT_correct_calculation_l1534_153406

theorem correct_calculation (x : ℕ) (h : x + 10 = 21) : x * 10 = 110 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1534_153406


namespace NUMINAMATH_GPT_problem_X_plus_Y_l1534_153434

def num_five_digit_even_numbers : Nat := 45000
def num_five_digit_multiples_of_7 : Nat := 12857
def X := num_five_digit_even_numbers
def Y := num_five_digit_multiples_of_7

theorem problem_X_plus_Y : X + Y = 57857 :=
by
  sorry

end NUMINAMATH_GPT_problem_X_plus_Y_l1534_153434


namespace NUMINAMATH_GPT_percentage_increase_l1534_153427

theorem percentage_increase (x y P : ℚ)
  (h1 : x = 0.9 * y)
  (h2 : x = 123.75)
  (h3 : y = 125 + 1.25 * P) : 
  P = 10 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_increase_l1534_153427


namespace NUMINAMATH_GPT_second_person_avg_pages_per_day_l1534_153458

theorem second_person_avg_pages_per_day
  (summer_days : ℕ) 
  (deshaun_books : ℕ) 
  (avg_pages_per_book : ℕ) 
  (percentage_read_by_second_person : ℚ) :
  -- Given conditions
  summer_days = 80 →
  deshaun_books = 60 →
  avg_pages_per_book = 320 →
  percentage_read_by_second_person = 0.75 →
  -- Prove
  (percentage_read_by_second_person * (deshaun_books * avg_pages_per_book) / summer_days) = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_second_person_avg_pages_per_day_l1534_153458


namespace NUMINAMATH_GPT_combinations_eight_choose_three_l1534_153469

theorem combinations_eight_choose_three : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_GPT_combinations_eight_choose_three_l1534_153469


namespace NUMINAMATH_GPT_brenda_age_l1534_153414

theorem brenda_age
  (A B J : ℕ)
  (h1 : A = 4 * B)
  (h2 : J = B + 9)
  (h3 : A = J)
  : B = 3 :=
by 
  sorry

end NUMINAMATH_GPT_brenda_age_l1534_153414


namespace NUMINAMATH_GPT_problem_1_problem_2_l1534_153474

open Real
open Set

noncomputable def y (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem problem_1 :
  { x : ℝ | y x = 1 ∧ sin x ≠ -1 } = { x | ∃ (k : ℤ), x = 2 * k * π + (π / 2) } :=
by
  sorry

theorem problem_2 : 
  ∃ x, y x = 1 ∧ ∀ x', y x' ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1534_153474


namespace NUMINAMATH_GPT_magic_coin_l1534_153479

theorem magic_coin (m n : ℕ) (h_m_prime: Nat.gcd m n = 1)
  (h_prob : (m : ℚ) / n = 1 / 158760): m + n = 158761 := by
  sorry

end NUMINAMATH_GPT_magic_coin_l1534_153479


namespace NUMINAMATH_GPT_find_F_16_l1534_153445

noncomputable def F : ℝ → ℝ := sorry

lemma F_condition_1 : ∀ x, (x + 4) ≠ 0 ∧ (x + 2) ≠ 0 → (F (4 * x) / F (x + 4) = 16 - (64 * x + 64) / (x^2 + 6 * x + 8)) := sorry

lemma F_condition_2 : F 8 = 33 := sorry

theorem find_F_16 : F 16 = 136 :=
by
  have h1 := F_condition_1
  have h2 := F_condition_2
  sorry

end NUMINAMATH_GPT_find_F_16_l1534_153445


namespace NUMINAMATH_GPT_students_like_all_three_l1534_153417

variables (N : ℕ) (r : ℚ) (j : ℚ) (o : ℕ) (n : ℕ)

-- Number of students in the class
def num_students := N = 40

-- Fraction of students who like Rock
def fraction_rock := r = 1/4

-- Fraction of students who like Jazz
def fraction_jazz := j = 1/5

-- Number of students who like other genres
def num_other_genres := o = 8

-- Number of students who do not like any of the three genres
def num_no_genres := n = 6

---- Proof theorem
theorem students_like_all_three
  (h1 : num_students N)
  (h2 : fraction_rock r)
  (h3 : fraction_jazz j)
  (h4 : num_other_genres o)
  (h5 : num_no_genres n) :
  ∃ z : ℕ, z = 2 := 
sorry

end NUMINAMATH_GPT_students_like_all_three_l1534_153417


namespace NUMINAMATH_GPT_milburg_population_l1534_153425

-- Define the number of grown-ups and children in Milburg
def grown_ups : ℕ := 5256
def children : ℕ := 2987

-- The total population is defined as the sum of grown-ups and children
def total_population : ℕ := grown_ups + children

-- Goal: Prove that the total population in Milburg is 8243
theorem milburg_population : total_population = 8243 := 
by {
  -- the proof should be here, but we use sorry to skip it
  sorry
}

end NUMINAMATH_GPT_milburg_population_l1534_153425


namespace NUMINAMATH_GPT_negative_remainder_l1534_153424

theorem negative_remainder (a : ℤ) (h : a % 1999 = 1) : (-a) % 1999 = 1998 :=
by
  sorry

end NUMINAMATH_GPT_negative_remainder_l1534_153424


namespace NUMINAMATH_GPT_compute_expr_l1534_153496

theorem compute_expr {x : ℝ} (h : x = 5) : (x^6 - 2 * x^3 + 1) / (x^3 - 1) = 124 :=
by
  sorry

end NUMINAMATH_GPT_compute_expr_l1534_153496


namespace NUMINAMATH_GPT_factorize_expression_l1534_153464

variable {x y : ℝ}

theorem factorize_expression :
  3 * x^2 - 27 * y^2 = 3 * (x + 3 * y) * (x - 3 * y) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1534_153464


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1534_153481

theorem solution_set_of_inequality :
  {x : ℝ | 1 / x < 1 / 3} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1534_153481


namespace NUMINAMATH_GPT_solve_quadratic_l1534_153463

   theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 5 * x^2 + 8 * x - 24 = 0) : x = 6 / 5 :=
   sorry
   
end NUMINAMATH_GPT_solve_quadratic_l1534_153463


namespace NUMINAMATH_GPT_ratio_of_investments_l1534_153453

theorem ratio_of_investments (I B_profit total_profit : ℝ) (x : ℝ)
  (h1 : B_profit = 4000) (h2 : total_profit = 28000) (h3 : I * (2 * B_profit / 4000 - 1) = total_profit - B_profit) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_investments_l1534_153453


namespace NUMINAMATH_GPT_learning_machine_price_reduction_l1534_153435

theorem learning_machine_price_reduction (x : ℝ) (h1 : 2000 * (1 - x) * (1 - x) = 1280) : 2000 * (1 - x)^2 = 1280 :=
by
  sorry

end NUMINAMATH_GPT_learning_machine_price_reduction_l1534_153435


namespace NUMINAMATH_GPT_percentage_of_respondents_l1534_153470

variables {X Y : ℝ}
variable (h₁ : 23 <= 100 - X)

theorem percentage_of_respondents 
  (h₁ : 0 ≤ X) 
  (h₂ : X ≤ 100) 
  (h₃ : 0 ≤ 23) 
  (h₄ : 23 ≤ 23) : 
  Y = 100 - X := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_respondents_l1534_153470


namespace NUMINAMATH_GPT_find_x_ceil_mul_l1534_153498

theorem find_x_ceil_mul (x : ℝ) (h : ⌈x⌉ * x = 75) : x = 8.333 := by
  sorry

end NUMINAMATH_GPT_find_x_ceil_mul_l1534_153498


namespace NUMINAMATH_GPT_AlbertTookAwayCandies_l1534_153465

-- Define the parameters and conditions given in the problem
def PatriciaStartCandies : ℕ := 76
def PatriciaEndCandies : ℕ := 71

-- Define the statement that proves the number of candies Albert took away
theorem AlbertTookAwayCandies :
  PatriciaStartCandies - PatriciaEndCandies = 5 := by
  sorry

end NUMINAMATH_GPT_AlbertTookAwayCandies_l1534_153465


namespace NUMINAMATH_GPT_Area_S_inequality_l1534_153456

def S (t : ℝ) (x y : ℝ) : Prop :=
  let T := Real.sin (Real.pi * t)
  |x - T| + |y - T| ≤ T

theorem Area_S_inequality (t : ℝ) :
  let T := Real.sin (Real.pi * t)
  0 ≤ 2 * T^2 := by
  sorry

end NUMINAMATH_GPT_Area_S_inequality_l1534_153456


namespace NUMINAMATH_GPT_remaining_kids_l1534_153467

def initial_kids : Float := 22.0
def kids_who_went_home : Float := 14.0

theorem remaining_kids : initial_kids - kids_who_went_home = 8.0 :=
by 
  sorry

end NUMINAMATH_GPT_remaining_kids_l1534_153467


namespace NUMINAMATH_GPT_max_k_mono_incr_binom_l1534_153475

theorem max_k_mono_incr_binom :
  ∀ (k : ℕ), (k ≤ 11) → 
  (∀ i j : ℕ, 1 ≤ i → i < j → j ≤ k → (Nat.choose 10 (i - 1) < Nat.choose 10 (j - 1))) →
  k = 6 :=
by sorry

end NUMINAMATH_GPT_max_k_mono_incr_binom_l1534_153475


namespace NUMINAMATH_GPT_cos_lt_sin3_div_x3_l1534_153492

open Real

theorem cos_lt_sin3_div_x3 (x : ℝ) (h1 : 0 < x) (h2 : x < pi / 2) : 
  cos x < (sin x / x) ^ 3 := 
  sorry

end NUMINAMATH_GPT_cos_lt_sin3_div_x3_l1534_153492


namespace NUMINAMATH_GPT_max_area_house_l1534_153476

def price_colored := 450
def price_composite := 200
def cost_limit := 32000

def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

theorem max_area_house : 
  ∃ (x y S : ℝ), 
    (S = x * y) ∧ 
    (material_cost x y ≤ cost_limit) ∧ 
    (0 < S ∧ S ≤ 100) ∧ 
    (S = 100 → x = 20 / 3) := 
by
  sorry

end NUMINAMATH_GPT_max_area_house_l1534_153476


namespace NUMINAMATH_GPT_polynomial_negativity_l1534_153486

theorem polynomial_negativity (a x : ℝ) (h₀ : 0 < x) (h₁ : x < a) (h₂ : 0 < a) : 
  (a - x)^6 - 3 * a * (a - x)^5 + (5 / 2) * a^2 * (a - x)^4 - (1 / 2) * a^4 * (a - x)^2 < 0 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_negativity_l1534_153486


namespace NUMINAMATH_GPT_vermont_clicked_ads_l1534_153403

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end NUMINAMATH_GPT_vermont_clicked_ads_l1534_153403


namespace NUMINAMATH_GPT_segment_area_formula_l1534_153420
noncomputable def area_of_segment (r a : ℝ) : ℝ :=
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2)

theorem segment_area_formula (r a : ℝ) : area_of_segment r a =
  r^2 * Real.arcsin (a / (2 * r)) - (a / 4) * Real.sqrt (4 * r^2 - a^2) :=
sorry

end NUMINAMATH_GPT_segment_area_formula_l1534_153420


namespace NUMINAMATH_GPT_intersection_M_N_eq_M_l1534_153441

-- Definitions of M and N
def M : Set ℝ := { x : ℝ | x^2 - x < 0 }
def N : Set ℝ := { x : ℝ | abs x < 2 }

-- Proof statement
theorem intersection_M_N_eq_M : M ∩ N = M := 
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_M_l1534_153441


namespace NUMINAMATH_GPT_correct_average_weight_l1534_153446

theorem correct_average_weight 
  (n : ℕ) 
  (w_avg : ℝ) 
  (W_init : ℝ)
  (d1 : ℝ)
  (d2 : ℝ)
  (d3 : ℝ)
  (W_adj : ℝ)
  (w_corr : ℝ)
  (h1 : n = 30)
  (h2 : w_avg = 58.4)
  (h3 : W_init = n * w_avg)
  (h4 : d1 = 62 - 56)
  (h5 : d2 = 59 - 65)
  (h6 : d3 = 54 - 50)
  (h7 : W_adj = W_init + d1 + d2 + d3)
  (h8 : w_corr = W_adj / n) :
  w_corr = 58.5 := 
sorry

end NUMINAMATH_GPT_correct_average_weight_l1534_153446


namespace NUMINAMATH_GPT_percent_students_with_pets_l1534_153488

theorem percent_students_with_pets 
  (total_students : ℕ) (students_with_cats : ℕ) (students_with_dogs : ℕ) (students_with_both : ℕ) (h_total : total_students = 500)
  (h_cats : students_with_cats = 150) (h_dogs : students_with_dogs = 100) (h_both : students_with_both = 40) :
  (students_with_cats + students_with_dogs - students_with_both) * 100 / total_students = 42 := 
by
  sorry

end NUMINAMATH_GPT_percent_students_with_pets_l1534_153488


namespace NUMINAMATH_GPT_sum_of_interior_angles_hexagon_l1534_153409

theorem sum_of_interior_angles_hexagon : 
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 = 720 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_hexagon_l1534_153409


namespace NUMINAMATH_GPT_eggs_total_l1534_153455

-- Definitions based on the conditions
def breakfast_eggs : Nat := 2
def lunch_eggs : Nat := 3
def dinner_eggs : Nat := 1

-- Theorem statement
theorem eggs_total : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  -- This part will be filled in with proof steps, but it's omitted here
  sorry

end NUMINAMATH_GPT_eggs_total_l1534_153455


namespace NUMINAMATH_GPT_calculate_rows_l1534_153439

-- Definitions based on conditions
def totalPecanPies : ℕ := 16
def totalApplePies : ℕ := 14
def piesPerRow : ℕ := 5

-- The goal is to prove the total rows of pies
theorem calculate_rows : (totalPecanPies + totalApplePies) / piesPerRow = 6 := by
  sorry

end NUMINAMATH_GPT_calculate_rows_l1534_153439


namespace NUMINAMATH_GPT_simplify_expr_l1534_153471

theorem simplify_expr (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1534_153471


namespace NUMINAMATH_GPT_equilateral_triangle_sum_l1534_153468

theorem equilateral_triangle_sum (a u v w : ℝ)
  (h1: u^2 + v^2 = w^2):
  w^2 + Real.sqrt 3 * u * v = a^2 := 
sorry

end NUMINAMATH_GPT_equilateral_triangle_sum_l1534_153468


namespace NUMINAMATH_GPT_find_two_numbers_l1534_153499

theorem find_two_numbers :
  ∃ (x y : ℝ), 
  (2 * (x + y) = x^2 - y^2 ∧ 2 * (x + y) = (x * y) / 4 - 56) ∧ 
  ((x = 26 ∧ y = 24) ∨ (x = -8 ∧ y = -10)) := 
sorry

end NUMINAMATH_GPT_find_two_numbers_l1534_153499


namespace NUMINAMATH_GPT_value_of_m_l1534_153412

theorem value_of_m
  (m : ℤ)
  (h1 : ∃ p : ℕ → ℝ, p 4 = 1/3 ∧ p 1 = -(m + 4) ∧ p 0 = -11 ∧ (∀ (n : ℕ), (n ≠ 4 ∧ n ≠ 1 ∧ n ≠ 0) → p n = 0) ∧ 1 ≤ p 4 + p 1 + p 0) :
  m = 4 :=
  sorry

end NUMINAMATH_GPT_value_of_m_l1534_153412


namespace NUMINAMATH_GPT_min_value_of_expression_l1534_153423

theorem min_value_of_expression (a : ℝ) (h₀ : a > 0)
  (x₁ x₂ : ℝ)
  (h₁ : x₁ + x₂ = 4 * a)
  (h₂ : x₁ * x₂ = a * a) :
  x₁ + x₂ + a / (x₁ * x₂) = 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1534_153423


namespace NUMINAMATH_GPT_dwarfs_truthful_count_l1534_153484

theorem dwarfs_truthful_count : ∃ (x y : ℕ), x + y = 10 ∧ x + 2 * y = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_dwarfs_truthful_count_l1534_153484


namespace NUMINAMATH_GPT_ratio_AB_CD_lengths_AB_CD_l1534_153450

-- Given conditions as definitions
def ABD_triangle (A B D : Point) : Prop := true  -- In quadrilateral ABCD, a diagonal BD is drawn
def BCD_triangle (B C D : Point) : Prop := true  -- Circles are inscribed in triangles ABD and BCD
def Line_through_B_center_AM_M (A B D M : Point) (AM MD : ℚ) : Prop :=
  (AM = 8/5) ∧ (MD = 12/5)
def Line_through_D_center_BN_N (B C D N : Point) (BN NC : ℚ) : Prop :=
  (BN = 30/11) ∧ (NC = 25/11)

-- Mathematically equivalent proof problems
theorem ratio_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB / CD = 4 / 5 :=
by
  sorry

theorem lengths_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB + CD = 9 ∧
  AB - CD = -1 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_AB_CD_lengths_AB_CD_l1534_153450


namespace NUMINAMATH_GPT_chloe_points_first_round_l1534_153480

theorem chloe_points_first_round 
  (P : ℕ)
  (second_round_points : ℕ := 50)
  (lost_points : ℕ := 4)
  (total_points : ℕ := 86)
  (h : P + second_round_points - lost_points = total_points) : 
  P = 40 := 
by 
  sorry

end NUMINAMATH_GPT_chloe_points_first_round_l1534_153480


namespace NUMINAMATH_GPT_factorization_identity_l1534_153448

variable (a b : ℝ)

theorem factorization_identity : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end NUMINAMATH_GPT_factorization_identity_l1534_153448


namespace NUMINAMATH_GPT_dropouts_correct_l1534_153457

/-- Definition for initial racers, racers joining after 20 minutes, and racers at finish line. -/
def initial_racers : ℕ := 50
def joining_racers : ℕ := 30
def finishers : ℕ := 130

/-- Total racers after initial join and doubling. -/
def total_racers : ℕ := (initial_racers + joining_racers) * 2

/-- The number of people who dropped out before finishing the race. -/
def dropped_out : ℕ := total_racers - finishers

/-- Proof statement to show the number of people who dropped out before finishing is 30. -/
theorem dropouts_correct : dropped_out = 30 := by
  sorry

end NUMINAMATH_GPT_dropouts_correct_l1534_153457


namespace NUMINAMATH_GPT_find_x_value_l1534_153432

theorem find_x_value (x : ℝ) (h : (7 / (x - 2) + x / (2 - x) = 4)) : x = 3 :=
sorry

end NUMINAMATH_GPT_find_x_value_l1534_153432


namespace NUMINAMATH_GPT_productivity_increase_l1534_153472

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : (7/8) * b * (1 + x / 100) = 1.05 * b)

theorem productivity_increase (x : ℝ) : x = 20 := sorry

end NUMINAMATH_GPT_productivity_increase_l1534_153472


namespace NUMINAMATH_GPT_lattice_points_on_hyperbola_l1534_153466

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end NUMINAMATH_GPT_lattice_points_on_hyperbola_l1534_153466


namespace NUMINAMATH_GPT_three_digit_sum_27_l1534_153401

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_sum_27_l1534_153401


namespace NUMINAMATH_GPT_chips_in_bag_l1534_153443

theorem chips_in_bag :
  let initial_chips := 5
  let additional_chips := 5
  let daily_chips := 10
  let total_days := 10
  let first_day_chips := initial_chips + additional_chips
  let remaining_days := total_days - 1
  (first_day_chips + remaining_days * daily_chips) = 100 :=
by
  sorry

end NUMINAMATH_GPT_chips_in_bag_l1534_153443


namespace NUMINAMATH_GPT_solve_for_r_l1534_153451

theorem solve_for_r (r s : ℚ) (h : (2 * (r - 45)) / 3 = (3 * s - 2 * r) / 4) (s_val : s = 20) :
  r = 270 / 7 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_r_l1534_153451


namespace NUMINAMATH_GPT_evaluate_expression_at_2_l1534_153444

noncomputable def replace_and_evaluate (x : ℝ) : ℝ :=
  (3 * x - 2) / (-x + 6)

theorem evaluate_expression_at_2 :
  replace_and_evaluate 2 = -2 :=
by
  -- evaluation and computation would go here, skipped with sorry
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_2_l1534_153444


namespace NUMINAMATH_GPT_maximum_n_for_positive_S_l1534_153478

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem maximum_n_for_positive_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (S : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (d_neg : d < 0)
  (S4_eq_S8 : S 4 = S 8)
  (h1 : is_arithmetic_sequence a d)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n, ∀ m, m ≤ n → S m > 0 ∧ ∀ k, k > n → S k ≤ 0 ∧ n = 11 :=
sorry

end NUMINAMATH_GPT_maximum_n_for_positive_S_l1534_153478


namespace NUMINAMATH_GPT_passengers_remaining_l1534_153490

theorem passengers_remaining :
  let initial_passengers := 64
  let reduction_factor := (2 / 3)
  ∀ (n : ℕ), n = 4 → initial_passengers * reduction_factor^n = 1024 / 81 := by
sorry

end NUMINAMATH_GPT_passengers_remaining_l1534_153490


namespace NUMINAMATH_GPT_fraction_of_students_who_walk_home_l1534_153495

theorem fraction_of_students_who_walk_home (bus auto bikes scooters : ℚ) 
  (hbus : bus = 2/5) (hauto : auto = 1/5) 
  (hbikes : bikes = 1/10) (hscooters : scooters = 1/10) : 
  1 - (bus + auto + bikes + scooters) = 1/5 :=
by 
  rw [hbus, hauto, hbikes, hscooters]
  sorry

end NUMINAMATH_GPT_fraction_of_students_who_walk_home_l1534_153495


namespace NUMINAMATH_GPT_part1_part2_part3_l1534_153426

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
axiom sequence_def (n : ℕ) : a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

-- Part (1)
axiom a1_def : a 1 = 1
axiom a2_def : a 2 = 2
theorem part1 : a 4 = 1 ∨ a 4 = 3 ∨ a 4 = 5 :=
  sorry

-- Part (2)
axiom has_max (M : ℕ) : ∀ n, a n ≤ M
theorem part2 : ∃ n, a n = 0 :=
  sorry

-- Part (3)
axiom positive_seq : ∀ n, a n > 0
theorem part3 : ¬∃ M : ℝ, ∀ n, a n ≤ M :=
  sorry

end NUMINAMATH_GPT_part1_part2_part3_l1534_153426


namespace NUMINAMATH_GPT_divisor_and_remainder_correct_l1534_153442

theorem divisor_and_remainder_correct:
  ∃ d r : ℕ, d ≠ 0 ∧ 1270 = 74 * d + r ∧ r = 12 ∧ d = 17 :=
by
  sorry

end NUMINAMATH_GPT_divisor_and_remainder_correct_l1534_153442


namespace NUMINAMATH_GPT_number_is_100_l1534_153440

theorem number_is_100 (n : ℕ) 
  (hquot : n / 11 = 9) 
  (hrem : n % 11 = 1) : 
  n = 100 := 
by 
  sorry

end NUMINAMATH_GPT_number_is_100_l1534_153440


namespace NUMINAMATH_GPT_consecutive_numbers_square_sum_l1534_153454

theorem consecutive_numbers_square_sum (n : ℕ) (a b : ℕ) (h1 : 2 * n + 1 = 144169^2)
  (h2 : a = 72084) (h3 : b = a + 1) : a^2 + b^2 = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_numbers_square_sum_l1534_153454


namespace NUMINAMATH_GPT_original_amount_in_cookie_jar_l1534_153485

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end NUMINAMATH_GPT_original_amount_in_cookie_jar_l1534_153485


namespace NUMINAMATH_GPT_third_player_games_l1534_153460

theorem third_player_games (p1 p2 p3 : ℕ) (h1 : p1 = 21) (h2 : p2 = 10)
  (total_games : p1 = p2 + p3) : p3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_third_player_games_l1534_153460


namespace NUMINAMATH_GPT_abs_nested_expression_l1534_153418

theorem abs_nested_expression (x : ℝ) (h : x = 2023) : 
  abs (abs (abs x - x) - abs x) - x = 0 :=
by
  subst h
  sorry

end NUMINAMATH_GPT_abs_nested_expression_l1534_153418


namespace NUMINAMATH_GPT_parity_of_expression_l1534_153431

theorem parity_of_expression (e m : ℕ) (he : (∃ k : ℕ, e = 2 * k)) : Odd (e ^ 2 + 3 ^ m) :=
  sorry

end NUMINAMATH_GPT_parity_of_expression_l1534_153431


namespace NUMINAMATH_GPT_truncated_cone_radius_l1534_153410

theorem truncated_cone_radius (R: ℝ) (l: ℝ) (h: 0 < l)
  (h1 : ∃ (r: ℝ), r = (R + 5) / 2 ∧ (5 + r) = (1 / 2) * (R + r))
  : R = 25 :=
sorry

end NUMINAMATH_GPT_truncated_cone_radius_l1534_153410


namespace NUMINAMATH_GPT_overlapping_area_l1534_153433

def area_of_overlap (g1 g2 : Grid) : ℝ :=
  -- Dummy implementation to ensure code compiles
  6.0

structure Grid :=
  (size : ℝ) (arrow_direction : Direction)

inductive Direction
| North
| West

theorem overlapping_area (g1 g2 : Grid) 
  (h1 : g1.size = 4) 
  (h2 : g2.size = 4) 
  (d1 : g1.arrow_direction = Direction.North) 
  (d2 : g2.arrow_direction = Direction.West) 
  : area_of_overlap g1 g2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_overlapping_area_l1534_153433


namespace NUMINAMATH_GPT_value_of_expression_l1534_153430

theorem value_of_expression (x : ℝ) (h : |x| = x + 2) : 19 * x ^ 99 + 3 * x + 27 = 5 :=
by
  have h1: x ≥ -2 := sorry
  have h2: x = -1 := sorry
  sorry

end NUMINAMATH_GPT_value_of_expression_l1534_153430


namespace NUMINAMATH_GPT_value_of_c_l1534_153400

theorem value_of_c
    (x y c : ℝ)
    (h1 : 3 * x - 5 * y = 5)
    (h2 : x / (x + y) = c)
    (h3 : x - y = 2.999999999999999) :
    c = 0.7142857142857142 :=
by
    sorry

end NUMINAMATH_GPT_value_of_c_l1534_153400


namespace NUMINAMATH_GPT_a_2016_is_1_l1534_153449

noncomputable def seq_a (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * b n

theorem a_2016_is_1 (a b : ℕ → ℝ)
  (h1 : a 1 = 1)
  (hb : seq_a a b)
  (h3 : b 1008 = 1) :
  a 2016 = 1 :=
sorry

end NUMINAMATH_GPT_a_2016_is_1_l1534_153449


namespace NUMINAMATH_GPT_total_marks_more_than_physics_l1534_153491

-- Definitions of variables for marks in different subjects
variables (P C M : ℕ)

-- Conditions provided in the problem
def total_marks_condition (P : ℕ) (C : ℕ) (M : ℕ) : Prop := P + C + M > P
def average_chemistry_math_marks (C : ℕ) (M : ℕ) : Prop := (C + M) / 2 = 55

-- The main proof statement: Proving the difference in total marks and physics marks
theorem total_marks_more_than_physics 
    (h1 : total_marks_condition P C M)
    (h2 : average_chemistry_math_marks C M) :
  (P + C + M) - P = 110 := 
sorry

end NUMINAMATH_GPT_total_marks_more_than_physics_l1534_153491


namespace NUMINAMATH_GPT_andrew_start_age_l1534_153404

-- Define the conditions
def annual_donation : ℕ := 7
def current_age : ℕ := 29
def total_donation : ℕ := 133

-- The theorem to prove
theorem andrew_start_age : (total_donation / annual_donation) = (current_age - 10) :=
by
  sorry

end NUMINAMATH_GPT_andrew_start_age_l1534_153404


namespace NUMINAMATH_GPT_part_a_part_b_l1534_153402

-- Part (a): Number of ways to distribute 20 identical balls into 6 boxes so that no box is empty
theorem part_a:
  ∃ (n : ℕ), n = Nat.choose 19 5 :=
sorry

-- Part (b): Number of ways to distribute 20 identical balls into 6 boxes if some boxes can be empty
theorem part_b:
  ∃ (n : ℕ), n = Nat.choose 25 5 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1534_153402


namespace NUMINAMATH_GPT_people_joined_group_l1534_153436

theorem people_joined_group (x y : ℕ) (h1 : 1430 = 22 * x) (h2 : 1430 = 13 * (x + y)) : y = 45 := 
by 
  -- This is just the statement, so we add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_people_joined_group_l1534_153436


namespace NUMINAMATH_GPT_geometric_series_sum_l1534_153438

open Real

theorem geometric_series_sum :
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  S = -716637955 / 16777216 :=
by
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  have h : S = -716637955 / 16777216 := sorry
  exact h

end NUMINAMATH_GPT_geometric_series_sum_l1534_153438


namespace NUMINAMATH_GPT_octagon_diag_20_algebraic_expr_positive_l1534_153419

def octagon_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octagon_diag_20 : octagon_diagonals 8 = 20 := by
  -- Formula for diagonals is used here
  sorry

theorem algebraic_expr_positive (x : ℝ) : 2 * x^2 - 2 * x + 1 > 0 := by
  -- Complete the square to show it's always positive
  sorry

end NUMINAMATH_GPT_octagon_diag_20_algebraic_expr_positive_l1534_153419


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1534_153428

theorem neither_necessary_nor_sufficient (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ¬(∀ a b, (a > b → (1 / a < 1 / b)) ∧ ((1 / a < 1 / b) → a > b)) := sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l1534_153428


namespace NUMINAMATH_GPT_odd_and_increasing_f1_odd_and_increasing_f2_l1534_153421

-- Define the functions
def f1 (x : ℝ) : ℝ := x * |x|
def f2 (x : ℝ) : ℝ := x^3

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Define the increasing function property
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → f x1 < f x2

-- Lean statement to prove
theorem odd_and_increasing_f1 : is_odd f1 ∧ is_increasing f1 := by
  sorry

theorem odd_and_increasing_f2 : is_odd f2 ∧ is_increasing f2 := by
  sorry

end NUMINAMATH_GPT_odd_and_increasing_f1_odd_and_increasing_f2_l1534_153421


namespace NUMINAMATH_GPT_circle_placement_in_rectangle_l1534_153415

theorem circle_placement_in_rectangle
  (L W : ℝ) (n : ℕ) (side_length diameter : ℝ)
  (h_dim : L = 20) (w_dim : W = 25)
  (h_squares : n = 120) (h_side_length : side_length = 1)
  (h_diameter : diameter = 1) :
  ∃ (x y : ℝ) (circle_radius : ℝ), 
    circle_radius = diameter / 2 ∧
    0 ≤ x ∧ x + diameter / 2 ≤ L ∧ 
    0 ≤ y ∧ y + diameter / 2 ≤ W ∧ 
    ∀ (i : ℕ) (hx : i < n) (sx sy : ℝ),
      0 ≤ sx ∧ sx + side_length ≤ L ∧
      0 ≤ sy ∧ sy + side_length ≤ W ∧
      dist (x, y) (sx + side_length / 2, sy + side_length / 2) ≥ diameter / 2 := 
sorry

end NUMINAMATH_GPT_circle_placement_in_rectangle_l1534_153415


namespace NUMINAMATH_GPT_inverse_proportion_relationship_l1534_153461

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_relationship (h1 : x1 < 0) (h2 : 0 < x2) 
  (hy1 : y1 = 3 / x1) (hy2 : y2 = 3 / x2) : y1 < 0 ∧ 0 < y2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_relationship_l1534_153461


namespace NUMINAMATH_GPT_contradiction_in_triangle_l1534_153408

theorem contradiction_in_triangle (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (sum_angles : A + B + C = 180) : false :=
by
  sorry

end NUMINAMATH_GPT_contradiction_in_triangle_l1534_153408


namespace NUMINAMATH_GPT_projectile_max_height_l1534_153462

theorem projectile_max_height :
  ∀ (t : ℝ), -12 * t^2 + 72 * t + 45 ≤ 153 :=
by
  sorry

end NUMINAMATH_GPT_projectile_max_height_l1534_153462


namespace NUMINAMATH_GPT_greatest_four_digit_multiple_of_17_l1534_153447

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end NUMINAMATH_GPT_greatest_four_digit_multiple_of_17_l1534_153447


namespace NUMINAMATH_GPT_carrie_mom_money_l1534_153407

theorem carrie_mom_money :
  ∀ (sweater_cost t_shirt_cost shoes_cost left_money total_money : ℕ),
  sweater_cost = 24 →
  t_shirt_cost = 6 →
  shoes_cost = 11 →
  left_money = 50 →
  total_money = sweater_cost + t_shirt_cost + shoes_cost + left_money →
  total_money = 91 :=
sorry

end NUMINAMATH_GPT_carrie_mom_money_l1534_153407


namespace NUMINAMATH_GPT_desired_alcohol_percentage_is_18_l1534_153477

noncomputable def final_alcohol_percentage (volume_x volume_y : ℕ) (percentage_x percentage_y : ℚ) : ℚ :=
  let total_volume := (volume_x + volume_y)
  let total_alcohol := (percentage_x * volume_x + percentage_y * volume_y)
  total_alcohol / total_volume * 100

theorem desired_alcohol_percentage_is_18 : 
  final_alcohol_percentage 300 200 0.10 0.30 = 18 := 
  sorry

end NUMINAMATH_GPT_desired_alcohol_percentage_is_18_l1534_153477
