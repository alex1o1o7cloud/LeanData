import Mathlib

namespace sin_180_degrees_l76_76404

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l76_76404


namespace infinite_cube_volume_sum_l76_76121

noncomputable def sum_of_volumes_of_infinite_cubes (a : ℝ) : ℝ :=
  ∑' n, (((a / (3 ^ n))^3))

theorem infinite_cube_volume_sum (a : ℝ) : sum_of_volumes_of_infinite_cubes a = (27 / 26) * a^3 :=
sorry

end infinite_cube_volume_sum_l76_76121


namespace casey_marathon_time_l76_76263

theorem casey_marathon_time (C : ℝ) (h : (C + (4 / 3) * C) / 2 = 7) : C = 10.5 :=
by
  sorry

end casey_marathon_time_l76_76263


namespace f_at_47_l76_76359

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_equation : ∀ x : ℝ, f (x - 1) + f (x + 1) = 0
axiom f_interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem f_at_47 : f 47 = -1 := by
  sorry

end f_at_47_l76_76359


namespace domain_of_f_2x_minus_1_l76_76381

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) →
  ∀ x, (1 / 2) ≤ x ∧ x ≤ (3 / 2) → ∃ y, f y = (2 * x - 1) :=
by
  intros h x hx
  sorry

end domain_of_f_2x_minus_1_l76_76381


namespace matrix_det_is_neg16_l76_76318

def matrix := Matrix (Fin 2) (Fin 2) ℤ
def given_matrix : matrix := ![![ -7, 5], ![6, -2]]

theorem matrix_det_is_neg16 : Matrix.det given_matrix = -16 := 
by
  sorry

end matrix_det_is_neg16_l76_76318


namespace min_value_abc_l76_76698

theorem min_value_abc : 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^b % 10 = 4) ∧ (b^c % 10 = 2) ∧ (c^a % 10 = 9) ∧ 
    (a + b + c = 17) :=
  by {
    sorry
  }

end min_value_abc_l76_76698


namespace tan_alpha_calc_l76_76466

theorem tan_alpha_calc (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
by sorry

end tan_alpha_calc_l76_76466


namespace sum_of_cubes_is_zero_l76_76128

theorem sum_of_cubes_is_zero 
  (a b : ℝ) 
  (h1 : a + b = 0) 
  (h2 : a * b = -1) : 
  a^3 + b^3 = 0 := by
  sorry

end sum_of_cubes_is_zero_l76_76128


namespace point_P_inside_circle_l76_76836

theorem point_P_inside_circle
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (e : ℝ)
  (h4 : e = 1 / 2)
  (x1 x2 : ℝ)
  (hx1 : a * x1 ^ 2 + b * x1 - c = 0)
  (hx2 : a * x2 ^ 2 + b * x2 - c = 0) :
  x1 ^ 2 + x2 ^ 2 < 2 :=
by
  sorry

end point_P_inside_circle_l76_76836


namespace equation_of_circle_C_equation_of_line_l_l76_76545

-- Condition: The center of the circle lies on the line y = x + 1.
def center_on_line (a b : ℝ) : Prop :=
  b = a + 1

-- Condition: The circle is tangent to the x-axis.
def tangent_to_x_axis (a b r : ℝ) : Prop :=
  r = b

-- Condition: Point P(-5, -2) lies on the circle.
def point_on_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Condition: Point Q(-4, -5) lies outside the circle.
def point_outside_circle (a b r x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 > r^2

-- Proof (1): Find the equation of the circle.
theorem equation_of_circle_C :
  ∃ (a b r : ℝ), center_on_line a b ∧ tangent_to_x_axis a b r ∧ point_on_circle a b r (-5) (-2) ∧ point_outside_circle a b r (-4) (-5) ∧ (∀ x y, (x - a)^2 + (y - b)^2 = r^2 ↔ (x + 3)^2 + (y + 2)^2 = 4) :=
sorry

-- Proof (2): Find the equation of the line l.
theorem equation_of_line_l (a b r : ℝ) (ha : center_on_line a b) (hb : tangent_to_x_axis a b r) (hc : point_on_circle a b r (-5) (-2)) (hd : point_outside_circle a b r (-4) (-5)) :
  ∃ (k : ℝ), ∀ x y, ((k = 0 ∧ x = -2) ∨ (k ≠ 0 ∧ y + 4 = -3/4 * (x + 2))) ↔ ((x = -2) ∨ (3 * x + 4 * y + 22 = 0)) :=
sorry

end equation_of_circle_C_equation_of_line_l_l76_76545


namespace find_S25_l76_76328

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end find_S25_l76_76328


namespace all_points_equal_l76_76165

-- Define the problem conditions and variables
variable (P : Type) -- points in the plane
variable [MetricSpace P] -- the plane is a metric space
variable (f : P → ℝ) -- assignment of numbers to points
variable (incenter : P → P → P → P) -- calculates incenter of a nondegenerate triangle

-- Condition: the value at the incenter of a triangle is the arithmetic mean of the values at the vertices
axiom incenter_mean_property : ∀ (A B C : P), 
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  f (incenter A B C) = (f A + f B + f C) / 3

-- The theorem to be proved
theorem all_points_equal : ∀ x y : P, f x = f y :=
by
  sorry

end all_points_equal_l76_76165


namespace foci_equality_ellipse_hyperbola_l76_76913

theorem foci_equality_ellipse_hyperbola (m : ℝ) (h : m > 0) 
  (hl: ∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (4 - m^2)) 
  (hh: ∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (m^2 + 2)) : 
  m = 1 :=
by {
  sorry
}

end foci_equality_ellipse_hyperbola_l76_76913


namespace minimum_percentage_increase_mean_l76_76842

def mean (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

theorem minimum_percentage_increase_mean (F : List ℤ) (p1 p2 : ℤ) (F' : List ℤ)
  (hF : F = [ -4, -1, 0, 6, 9 ])
  (hp1 : p1 = 2) (hp2 : p2 = 3)
  (hF' : F' = [p1, p2, 0, 6, 9])
  : (mean F' - mean F) / mean F * 100 = 100 := 
sorry

end minimum_percentage_increase_mean_l76_76842


namespace find_g3_l76_76721

variable (g : ℝ → ℝ)

axiom condition_g :
  ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 2 * x

theorem find_g3 : g 3 = 9 / 2 :=
  by
    sorry

end find_g3_l76_76721


namespace girls_more_than_boys_l76_76845

-- Defining the conditions
def ratio_boys_girls : Nat := 3 / 4
def total_students : Nat := 42

-- Defining the hypothesis based on conditions
theorem girls_more_than_boys : (total_students * ratio_boys_girls) / (3 + 4) * (4 - 3) = 6 := by
  sorry

end girls_more_than_boys_l76_76845


namespace expand_product_l76_76795

-- Define the expressions (x + 3)(x + 8) and x^2 + 11x + 24
def expr1 (x : ℝ) : ℝ := (x + 3) * (x + 8)
def expr2 (x : ℝ) : ℝ := x^2 + 11 * x + 24

-- Prove that the two expressions are equal
theorem expand_product (x : ℝ) : expr1 x = expr2 x := by
  sorry

end expand_product_l76_76795


namespace trajectory_sufficient_not_necessary_l76_76467

-- Define for any point P if its trajectory is y = |x|
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2 = abs P.1

-- Define for any point P if its distances to the coordinate axes are equal
def equal_distances (P : ℝ × ℝ) : Prop :=
  abs P.1 = abs P.2

-- The main statement: prove that the trajectory is a sufficient but not necessary condition for equal_distances
theorem trajectory_sufficient_not_necessary (P : ℝ × ℝ) :
  trajectory P → equal_distances P ∧ ¬(equal_distances P → trajectory P) := 
sorry

end trajectory_sufficient_not_necessary_l76_76467


namespace divisible_by_pow3_l76_76379

-- Define the digit sequence function
def num_with_digits (a n : Nat) : Nat :=
  a * ((10 ^ (3 ^ n) - 1) / 9)

-- Main theorem statement
theorem divisible_by_pow3 (a n : Nat) (h_pos : 0 < n) : (num_with_digits a n) % (3 ^ n) = 0 := 
by
  sorry

end divisible_by_pow3_l76_76379


namespace multiplication_vs_subtraction_difference_l76_76209

variable (x : ℕ)
variable (h : x = 10)

theorem multiplication_vs_subtraction_difference :
  3 * x - (26 - x) = 14 := by
  sorry

end multiplication_vs_subtraction_difference_l76_76209


namespace park_area_calculation_l76_76351

noncomputable def width_of_park := Real.sqrt (9000000 / 65)
noncomputable def length_of_park := 8 * width_of_park

def actual_area_of_park (w l : ℝ) : ℝ := w * l

theorem park_area_calculation :
  let w := width_of_park
  let l := length_of_park
  actual_area_of_park w l = 1107746.48 :=
by
  -- Calculations from solution are provided here directly as conditions and definitions
  sorry

end park_area_calculation_l76_76351


namespace average_age_of_town_l76_76850

-- Definitions based on conditions
def ratio_of_women_to_men (nw nm : ℕ) : Prop := nw * 8 = nm * 9

def young_men (nm : ℕ) (n_young_men : ℕ) (average_age_young : ℕ) : Prop :=
  n_young_men = 40 ∧ average_age_young = 25

def remaining_men_average_age (nm n_young_men : ℕ) (average_age_remaining : ℕ) : Prop :=
  average_age_remaining = 35

def women_average_age (average_age_women : ℕ) : Prop :=
  average_age_women = 30

-- Complete problem statement we need to prove
theorem average_age_of_town (nw nm : ℕ) (total_avg_age : ℕ) :
  ratio_of_women_to_men nw nm →
  young_men nm 40 25 →
  remaining_men_average_age nm 40 35 →
  women_average_age 30 →
  total_avg_age = 32 * 17 + 6 :=
sorry

end average_age_of_town_l76_76850


namespace intersection_A_B_l76_76283

def A := { x : ℝ | -1 < x ∧ x ≤ 3 }
def B := { x : ℝ | 0 < x ∧ x < 10 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 3 } :=
  by sorry

end intersection_A_B_l76_76283


namespace albert_mary_age_ratio_l76_76517

theorem albert_mary_age_ratio
  (A M B : ℕ)
  (h1 : A = 4 * B)
  (h2 : M = A - 14)
  (h3 : B = 7)
  :
  A / M = 2 := 
by sorry

end albert_mary_age_ratio_l76_76517


namespace three_digit_number_multiple_of_eleven_l76_76180

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end three_digit_number_multiple_of_eleven_l76_76180


namespace four_digit_swap_square_l76_76078

theorem four_digit_swap_square (a b : ℤ) (N M : ℤ) : 
  N = 1111 * a + 123 ∧ 
  M = 1111 * a + 1023 ∧ 
  M = b ^ 2 → 
  N = 3456 := 
by sorry

end four_digit_swap_square_l76_76078


namespace product_evaluation_l76_76069

theorem product_evaluation (a b c : ℕ) (h : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) :
  6 * 15 * 2 = 4 := by
  sorry

end product_evaluation_l76_76069


namespace square_of_radius_l76_76077

-- Definitions based on conditions
def ER := 24
def RF := 31
def GS := 40
def SH := 29

-- The goal is to find square of radius r such that r^2 = 841
theorem square_of_radius (r : ℝ) :
  let R := ER
  let F := RF
  let G := GS
  let S := SH
  (∀ r : ℝ, (R + F) * (G + S) = r^2) → r^2 = 841 :=
sorry

end square_of_radius_l76_76077


namespace Dave_has_more_money_than_Derek_l76_76942

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l76_76942


namespace mul_exponent_property_l76_76350

variable (m : ℕ)  -- Assuming m is a natural number for simplicity

theorem mul_exponent_property : m^2 * m^3 = m^5 := 
by {
  sorry
}

end mul_exponent_property_l76_76350


namespace max_pages_l76_76524

/-- Prove that the maximum number of pages the book has is 208 -/
theorem max_pages (pages: ℕ) (h1: pages ≥ 16 * 12 + 1) (h2: pages ≤ 13 * 16) 
(h3: pages ≥ 20 * 10 + 1) (h4: pages ≤ 11 * 20) : 
  pages ≤ 208 :=
by
  -- proof to be filled in
  sorry

end max_pages_l76_76524


namespace distance_to_nearest_town_l76_76781

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l76_76781


namespace sqrt_nat_or_irrational_l76_76145

theorem sqrt_nat_or_irrational {n : ℕ} : 
  (∃ m : ℕ, m^2 = n) ∨ (¬ ∃ q r : ℕ, r ≠ 0 ∧ (q^2 = n * r^2 ∧ r * r ≠ n * n)) :=
sorry

end sqrt_nat_or_irrational_l76_76145


namespace price_of_most_expensive_book_l76_76159

-- Define the conditions
def number_of_books := 41
def price_increment := 3

-- Define the price of the n-th book as a function of the price of the first book
def price (c : ℕ) (n : ℕ) : ℕ := c + price_increment * (n - 1)

-- Define a theorem stating the result
theorem price_of_most_expensive_book (c : ℕ) :
  c = 30 → price c number_of_books = 150 :=
by {
  sorry
}

end price_of_most_expensive_book_l76_76159


namespace Ron_needs_to_drink_80_percent_l76_76305

theorem Ron_needs_to_drink_80_percent 
  (volume_each : ℕ)
  (volume_intelligence : ℕ)
  (volume_beauty : ℕ)
  (volume_strength : ℕ)
  (volume_second_pitcher : ℕ)
  (effective_volume : ℕ)
  (volume_intelligence_left : ℕ)
  (volume_beauty_left : ℕ)
  (volume_strength_left : ℕ)
  (total_volume : ℕ)
  (Ron_needs : ℕ)
  (intelligence_condition : effective_volume = 30)
  (initial_volumes : volume_each = 300)
  (first_drink : volume_intelligence = volume_each / 2)
  (mix_before_second_drink : volume_second_pitcher = volume_intelligence + volume_beauty)
  (Hermione_drink : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left)
  (Harry_drink : volume_strength_left = volume_each / 2)
  (second_mix : volume_second_pitcher = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (final_mix : volume_second_pitcher / 2 = volume_intelligence_left + volume_beauty_left + volume_strength_left)
  (Ron_needs_condition : Ron_needs = effective_volume / volume_intelligence_left * 100)
  : Ron_needs = 80 := sorry

end Ron_needs_to_drink_80_percent_l76_76305


namespace both_games_players_l76_76476

theorem both_games_players (kabadi_players kho_kho_only total_players both_games : ℕ)
  (h_kabadi : kabadi_players = 10)
  (h_kho_kho_only : kho_kho_only = 15)
  (h_total : total_players = 25)
  (h_equation : kabadi_players + kho_kho_only + both_games = total_players) :
  both_games = 0 :=
by
  -- question == answer given conditions
  sorry

end both_games_players_l76_76476


namespace winner_percentage_l76_76280

theorem winner_percentage (votes_winner : ℕ) (votes_difference : ℕ) (total_votes : ℕ) 
  (h1 : votes_winner = 1044) 
  (h2 : votes_difference = 288) 
  (h3 : total_votes = votes_winner + (votes_winner - votes_difference)) :
  (votes_winner * 100) / total_votes = 58 :=
by
  sorry

end winner_percentage_l76_76280


namespace smallest_class_size_l76_76605

theorem smallest_class_size (n : ℕ) (x : ℕ) (h1 : n > 50) (h2 : n = 4 * x + 2) : n = 54 :=
by
  sorry

end smallest_class_size_l76_76605


namespace no_consecutive_positive_integers_with_no_real_solutions_l76_76464

theorem no_consecutive_positive_integers_with_no_real_solutions :
  ∀ b c : ℕ, (c = b + 1) → (b^2 - 4 * c < 0) → (c^2 - 4 * b < 0) → false :=
by
  intro b c
  sorry

end no_consecutive_positive_integers_with_no_real_solutions_l76_76464


namespace solve_equation_l76_76061

theorem solve_equation : ∀ x : ℝ, (x - (x + 2) / 2 = (2 * x - 1) / 3 - 1) → (x = 2) :=
by
  intros x h
  sorry

end solve_equation_l76_76061


namespace smallest_N_value_proof_l76_76198

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l76_76198


namespace price_of_each_toy_l76_76079

variables (T : ℝ)

-- Given conditions
def total_cost (T : ℝ) : ℝ := 3 * T + 2 * 5 + 5 * 6

theorem price_of_each_toy :
  total_cost T = 70 → T = 10 :=
sorry

end price_of_each_toy_l76_76079


namespace Seokhyung_drank_the_most_l76_76544

-- Define the conditions
def Mina_Amount := 0.6
def Seokhyung_Amount := 1.5
def Songhwa_Amount := Seokhyung_Amount - 0.6

-- Statement to prove that Seokhyung drank the most cola
theorem Seokhyung_drank_the_most : Seokhyung_Amount > Mina_Amount ∧ Seokhyung_Amount > Songhwa_Amount :=
by
  -- Proof skipped
  sorry

end Seokhyung_drank_the_most_l76_76544


namespace total_spent_l76_76095

theorem total_spent (cost_per_deck : ℕ) (decks_frank : ℕ) (decks_friend : ℕ) (total : ℕ) : 
  cost_per_deck = 7 → 
  decks_frank = 3 → 
  decks_friend = 2 → 
  total = (decks_frank * cost_per_deck) + (decks_friend * cost_per_deck) → 
  total = 35 :=
by
  sorry

end total_spent_l76_76095


namespace garden_area_l76_76189

/-- A rectangular garden is 350 cm long and 50 cm wide. Determine its area in square meters. -/
theorem garden_area (length_cm width_cm : ℝ) (h_length : length_cm = 350) (h_width : width_cm = 50) : (length_cm / 100) * (width_cm / 100) = 1.75 :=
by
  sorry

end garden_area_l76_76189


namespace polynomials_exist_l76_76717

theorem polynomials_exist (p : ℕ) (hp : Nat.Prime p) :
  ∃ (P Q : Polynomial ℤ),
  ¬(Polynomial.degree P = 0) ∧ ¬(Polynomial.degree Q = 0) ∧
  (∀ n, (Polynomial.coeff (P * Q) n).natAbs % p =
    if n = 0 then 1
    else if n = 4 then 1
    else if n = 2 then p - 2
    else 0) :=
sorry

end polynomials_exist_l76_76717


namespace find_width_of_jordan_rectangle_l76_76565

theorem find_width_of_jordan_rectangle (width : ℕ) (h1 : 12 * 15 = 9 * width) : width = 20 :=
by
  sorry

end find_width_of_jordan_rectangle_l76_76565


namespace range_of_a_l76_76158

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := 
by 
  sorry

end range_of_a_l76_76158


namespace faster_train_speed_l76_76141

theorem faster_train_speed (v : ℝ) (h_total_length : 100 + 100 = 200) 
  (h_cross_time : 8 = 8) (h_speeds : 3 * v = 200 / 8) : 2 * v = 50 / 3 :=
sorry

end faster_train_speed_l76_76141


namespace roger_steps_to_minutes_l76_76879

theorem roger_steps_to_minutes (h1 : ∃ t: ℕ, t = 30 ∧ ∃ s: ℕ, s = 2000)
                               (h2 : ∃ g: ℕ, g = 10000) :
  ∃ m: ℕ, m = 150 :=
by 
  sorry

end roger_steps_to_minutes_l76_76879


namespace square_form_l76_76131

theorem square_form (m n : ℤ) : 
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 :=
by
  let x := (2 * m^2 + n^2)
  let y := x^2
  let k := 2 * m * n
  let l := 2 * m^2 - n^2
  use k, l
  sorry

end square_form_l76_76131


namespace number_machine_output_l76_76119

def machine (x : ℕ) : ℕ := x + 15 - 6

theorem number_machine_output : machine 68 = 77 := by
  sorry

end number_machine_output_l76_76119


namespace repay_loan_with_interest_l76_76130

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l76_76130


namespace jason_attended_games_l76_76274

-- Define the conditions as given in the problem
def games_planned_this_month : ℕ := 11
def games_planned_last_month : ℕ := 17
def games_missed : ℕ := 16

-- Define the total number of games planned
def games_planned_total : ℕ := games_planned_this_month + games_planned_last_month

-- Define the number of games attended
def games_attended : ℕ := games_planned_total - games_missed

-- Prove that Jason attended 12 games
theorem jason_attended_games : games_attended = 12 := by
  -- The proof is omitted, but the theorem statement is required
  sorry

end jason_attended_games_l76_76274


namespace lines_parallel_if_perpendicular_to_same_plane_l76_76269

variable {Plane Line : Type}
variable {α β γ : Plane}
variable {m n : Line}

-- Define perpendicularity and parallelism as axioms for simplicity
axiom perp (L : Line) (P : Plane) : Prop
axiom parallel (L1 L2 : Line) : Prop

-- Assume conditions for the theorem
variables (h1 : perp m α) (h2 : perp n α)

-- The theorem proving the required relationship
theorem lines_parallel_if_perpendicular_to_same_plane : parallel m n := 
by
  sorry

end lines_parallel_if_perpendicular_to_same_plane_l76_76269


namespace circle_passes_through_fixed_point_l76_76001

theorem circle_passes_through_fixed_point (a : ℝ) (ha : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ (x y : ℝ), (x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0) → (x, y) = P :=
sorry

end circle_passes_through_fixed_point_l76_76001


namespace inequality_proof_l76_76125

theorem inequality_proof (a b c : ℝ) (h : a > b) : a / (c ^ 2 + 1) > b / (c ^ 2 + 1) :=
by
  sorry

end inequality_proof_l76_76125


namespace probability_longer_piece_at_least_x_squared_l76_76511

noncomputable def probability_longer_piece (x : ℝ) : ℝ :=
  if x = 0 then 1 else (2 / (x^2 + 1))

theorem probability_longer_piece_at_least_x_squared (x : ℝ) :
  probability_longer_piece x = (2 / (x^2 + 1)) :=
sorry

end probability_longer_piece_at_least_x_squared_l76_76511


namespace percentage_increase_l76_76444

theorem percentage_increase (P Q : ℝ)
  (price_decreased : ∀ P', P' = 0.80 * P)
  (revenue_increased : ∀ R R', R = P * Q ∧ R' = 1.28000000000000025 * R)
  : ∃ Q', Q' = 1.6000000000000003125 * Q :=
by
  sorry

end percentage_increase_l76_76444


namespace correct_average_weight_l76_76166

theorem correct_average_weight (n : ℕ) (incorrect_avg_weight : ℝ) (initial_avg_weight : ℝ)
  (misread_weight correct_weight : ℝ) (boys_count : ℕ) :
  incorrect_avg_weight = 58.4 →
  n = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  boys_count = n →
  initial_avg_weight = (incorrect_avg_weight * n + (correct_weight - misread_weight)) / boys_count →
  initial_avg_weight = 58.85 :=
by
  intro h1 h2 h3 h4 h5 h_avg
  sorry

end correct_average_weight_l76_76166


namespace prove_ineq_l76_76304

-- Define the quadratic equation
def quadratic_eqn (a b x : ℝ) : Prop :=
  3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0

-- Define the root relation
def root_relation (x1 x2 : ℝ) : Prop :=
  x1 * (x1 + 1) + x2 * (x2 + 1) = (x1 + 1) * (x2 + 1)

-- State the theorem
theorem prove_ineq (a b : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eqn a b x1 ∧ quadratic_eqn a b x2 ∧ root_relation x1 x2) →
  (a + b)^2 ≤ 4 :=
by
  sorry

end prove_ineq_l76_76304


namespace percentage_decrease_l76_76154

theorem percentage_decrease (x : ℝ) (h : x > 0) : ∃ p : ℝ, p = 0.20 ∧ ((1.25 * x) * (1 - p) = x) :=
by
  sorry

end percentage_decrease_l76_76154


namespace roots_polynomial_d_l76_76664

theorem roots_polynomial_d (c d u v : ℝ) (ru rpush rv rpush2 : ℝ) :
    (u + v + ru = 0) ∧ (u+3 + v-2 + rpush2 = 0) ∧
    (d + 153 = -(u + 3) * (v - 2) * (ru)) ∧ (d + 153 = s) ∧ (s = -(u + 3) * (v - 2) * (rpush2 - 1)) →
    d = 0 :=
by
  sorry

end roots_polynomial_d_l76_76664


namespace quadratic_discriminant_correct_l76_76382

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant_correct :
  discriminant 5 (5 + 1/2) (-1/2) = 161 / 4 :=
by
  -- let's prove the equality directly
  sorry

end quadratic_discriminant_correct_l76_76382


namespace range_of_given_function_l76_76204

noncomputable def given_function (x : ℝ) : ℝ :=
  abs (Real.sin x) / (Real.sin x) + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_given_function : Set.range given_function = {-1, 3} :=
by
  sorry

end range_of_given_function_l76_76204


namespace minimum_value_8m_n_l76_76625

noncomputable def min_value (m n : ℝ) : ℝ :=
  8 * m + n

theorem minimum_value_8m_n (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (8 / n) = 4) : 
  min_value m n = 8 :=
sorry

end minimum_value_8m_n_l76_76625


namespace clock_correction_l76_76692

def gain_per_day : ℚ := 13 / 4
def hours_per_day : ℕ := 24
def days_passed : ℕ := 9
def extra_hours : ℕ := 8
def total_hours : ℕ := days_passed * hours_per_day + extra_hours
def gain_per_hour : ℚ := gain_per_day / hours_per_day
def total_gain : ℚ := total_hours * gain_per_hour
def required_correction : ℚ := 30.33

theorem clock_correction :
  total_gain = required_correction :=
  by sorry

end clock_correction_l76_76692


namespace lucky_larry_l76_76322

theorem lucky_larry (a b c d e k : ℤ) 
    (h1 : a = 2) 
    (h2 : b = 3) 
    (h3 : c = 4) 
    (h4 : d = 5)
    (h5 : a - b - c - d + e = 2 - (b - (c - (d + e)))) 
    (h6 : k * 2 = e) : 
    k = 2 := by
  sorry

end lucky_larry_l76_76322


namespace variance_is_stability_measure_l76_76816

def stability_measure (yields : Fin 10 → ℝ) : Prop :=
  let mean := (yields 0 + yields 1 + yields 2 + yields 3 + yields 4 + yields 5 + yields 6 + yields 7 + yields 8 + yields 9) / 10
  let variance := 
    ((yields 0 - mean)^2 + (yields 1 - mean)^2 + (yields 2 - mean)^2 + (yields 3 - mean)^2 + 
     (yields 4 - mean)^2 + (yields 5 - mean)^2 + (yields 6 - mean)^2 + (yields 7 - mean)^2 + 
     (yields 8 - mean)^2 + (yields 9 - mean)^2) / 10
  true -- just a placeholder, would normally state that this is the appropriate measure

theorem variance_is_stability_measure (yields : Fin 10 → ℝ) : stability_measure yields :=
by 
  sorry

end variance_is_stability_measure_l76_76816


namespace value_of_y_l76_76417

theorem value_of_y : 
  ∀ y : ℚ, y = (2010^2 - 2010 + 1 : ℚ) / 2010 → y = (2009 + 1 / 2010 : ℚ) := by
  sorry

end value_of_y_l76_76417


namespace geometric_series_common_ratio_l76_76259

theorem geometric_series_common_ratio (a S r : ℝ) 
  (hS : S = a / (1 - r)) 
  (h_modified : (a * r^2) / (1 - r) = S / 16) : 
  r = 1/4 ∨ r = -1/4 :=
by
  sorry

end geometric_series_common_ratio_l76_76259


namespace john_pushups_less_l76_76655

theorem john_pushups_less (zachary david john : ℕ) 
  (h1 : zachary = 19)
  (h2 : david = zachary + 39)
  (h3 : david = 58)
  (h4 : john < david) : 
  david - john = 0 :=
sorry

end john_pushups_less_l76_76655


namespace inverse_variation_l76_76529

theorem inverse_variation (w : ℝ) (h1 : ∃ (c : ℝ), ∀ (x : ℝ), x^4 * w^(1/4) = c)
  (h2 : (3 : ℝ)^4 * (16 : ℝ)^(1/4) = (6 : ℝ)^4 * w^(1/4)) : 
  w = 1 / 4096 :=
by
  sorry

end inverse_variation_l76_76529


namespace min_value_expression_l76_76647

theorem min_value_expression (x y : ℝ) : (x^2 * y - 1)^2 + (x + y - 1)^2 ≥ 1 :=
sorry

end min_value_expression_l76_76647


namespace discounted_price_correct_l76_76000

noncomputable def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (discount / 100 * original_price)

theorem discounted_price_correct :
  discounted_price 800 30 = 560 :=
by
  -- Correctness of the discounted price calculation
  sorry

end discounted_price_correct_l76_76000


namespace unique_solution_for_a_l76_76916

theorem unique_solution_for_a (a : ℝ) :
  (∃! x : ℝ, 2 ^ |2 * x - 2| - a * Real.cos (1 - x) = 0) ↔ a = 1 :=
sorry

end unique_solution_for_a_l76_76916


namespace florist_bouquets_is_36_l76_76233

noncomputable def florist_bouquets : Prop :=
  let r := 125
  let y := 125
  let o := 125
  let p := 125
  let rk := 45
  let yk := 61
  let ok := 30
  let pk := 40
  let initial_flowers := r + y + o + p
  let total_killed := rk + yk + ok + pk
  let remaining_flowers := initial_flowers - total_killed
  let flowers_per_bouquet := 9
  let bouquets := remaining_flowers / flowers_per_bouquet
  bouquets = 36

theorem florist_bouquets_is_36 : florist_bouquets :=
  by
    sorry

end florist_bouquets_is_36_l76_76233


namespace x_is_48_percent_of_z_l76_76755

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end x_is_48_percent_of_z_l76_76755


namespace units_digit_13_pow_2003_l76_76915

theorem units_digit_13_pow_2003 : (13 ^ 2003) % 10 = 7 := by
  sorry

end units_digit_13_pow_2003_l76_76915


namespace find_range_t_l76_76809

noncomputable def f (x t : ℝ) : ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_range_t (f : ℝ → ℝ → ℝ)
  (h : ∀ t : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ t = x₁ - 6 ∧ f x₂ t = x₂ - 6 ∧ f x₃ t = x₃ - 6)) :
  ∀ t : ℝ, 1 < t ∧ t ≤ 2 := sorry

end find_range_t_l76_76809


namespace base_h_addition_eq_l76_76170

theorem base_h_addition_eq (h : ℕ) :
  let n1 := 7 * h^3 + 3 * h^2 + 6 * h + 4
  let n2 := 8 * h^3 + 4 * h^2 + 2 * h + 1
  let sum := 1 * h^4 + 7 * h^3 + 2 * h^2 + 8 * h + 5
  n1 + n2 = sum → h = 8 :=
by
  intros n1 n2 sum h_eq
  sorry

end base_h_addition_eq_l76_76170


namespace calc_3a2b_times_neg_a_squared_l76_76746

variables {a b : ℝ}

theorem calc_3a2b_times_neg_a_squared : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b :=
by
  sorry

end calc_3a2b_times_neg_a_squared_l76_76746


namespace solution_set_inequality_l76_76702

theorem solution_set_inequality (x : ℝ) : (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end solution_set_inequality_l76_76702


namespace fraction_sum_l76_76332

theorem fraction_sum :
  (1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 : ℚ) = -5 / 6 :=
by sorry

end fraction_sum_l76_76332


namespace abs_eq_neg_self_iff_l76_76981

theorem abs_eq_neg_self_iff (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by
  -- skipping proof with sorry
  sorry

end abs_eq_neg_self_iff_l76_76981


namespace Tina_independent_work_hours_l76_76425

-- Defining conditions as Lean constants
def Tina_work_rate := 1 / 12
def Ann_work_rate := 1 / 9
def Ann_work_hours := 3

-- Declaring the theorem to be proven
theorem Tina_independent_work_hours : 
  (Ann_work_hours * Ann_work_rate = 1/3) →
  ((1 : ℚ) - (Ann_work_hours * Ann_work_rate)) / Tina_work_rate = 8 :=
by {
  sorry
}

end Tina_independent_work_hours_l76_76425


namespace jessica_routes_count_l76_76200

def line := Type

def valid_route_count (p q r s t u : line) : ℕ := 9 + 36 + 36

theorem jessica_routes_count (p q r s t u : line) :
  valid_route_count p q r s t u = 81 :=
by
  sorry

end jessica_routes_count_l76_76200


namespace previous_salary_is_40_l76_76320

-- Define the conditions
def new_salary : ℕ := 80
def percentage_increase : ℕ := 100

-- Proven goal: John's previous salary before the raise
def previous_salary : ℕ := new_salary / 2

theorem previous_salary_is_40 : previous_salary = 40 := 
by
  -- Proof steps would go here
  sorry

end previous_salary_is_40_l76_76320


namespace optimal_discount_sequence_saves_more_l76_76122

theorem optimal_discount_sequence_saves_more :
  (let initial_price := 30
   let flat_discount := 5
   let percent_discount := 0.25
   let first_seq_price := ((initial_price - flat_discount) * (1 - percent_discount))
   let second_seq_price := ((initial_price * (1 - percent_discount)) - flat_discount)
   first_seq_price - second_seq_price = 1.25) :=
by
  sorry

end optimal_discount_sequence_saves_more_l76_76122


namespace eggs_left_l76_76658

theorem eggs_left (x : ℕ) : (47 - 5 - x) = (42 - x) :=
  by
  sorry

end eggs_left_l76_76658


namespace sum_of_possible_x_l76_76905

theorem sum_of_possible_x 
  (x : ℝ)
  (squareSide : ℝ) 
  (rectangleLength : ℝ) 
  (rectangleWidth : ℝ) 
  (areaCondition : (rectangleLength * rectangleWidth) = 3 * (squareSide ^ 2)) : 
  6 + 6.5 = 12.5 := 
by 
  sorry

end sum_of_possible_x_l76_76905


namespace triangle_cosines_identity_l76_76265

theorem triangle_cosines_identity 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) / a) + 
  (c^2 * Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) / b) + 
  (a^2 * Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / c) = 
  (a^4 + b^4 + c^4) / (2 * a * b * c) :=
by
  sorry

end triangle_cosines_identity_l76_76265


namespace sin_half_pi_plus_A_l76_76221

theorem sin_half_pi_plus_A (A : Real) (h : Real.cos (Real.pi + A) = -1 / 2) :
  Real.sin (Real.pi / 2 + A) = 1 / 2 := by
  sorry

end sin_half_pi_plus_A_l76_76221


namespace present_age_of_son_l76_76447

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 29) (h2 : M + 2 = 2 * (S + 2)) : S = 27 :=
sorry

end present_age_of_son_l76_76447


namespace find_M_l76_76653

theorem find_M : ∃ M : ℕ, M > 0 ∧ 18 ^ 2 * 45 ^ 2 = 15 ^ 2 * M ^ 2 ∧ M = 54 := by
  use 54
  sorry

end find_M_l76_76653


namespace ratio_of_bronze_to_silver_l76_76688

def total_gold_coins := 3500
def num_chests := 5
def total_silver_coins := 500
def coins_per_chest := 1000

-- Definitions based on the conditions to be used in the proof
def gold_coins_per_chest := total_gold_coins / num_chests
def silver_coins_per_chest := total_silver_coins / num_chests
def bronze_coins_per_chest := coins_per_chest - gold_coins_per_chest - silver_coins_per_chest
def bronze_to_silver_ratio := bronze_coins_per_chest / silver_coins_per_chest

theorem ratio_of_bronze_to_silver : bronze_to_silver_ratio = 2 := 
by
  sorry

end ratio_of_bronze_to_silver_l76_76688


namespace find_piles_l76_76055

theorem find_piles :
  ∃ N : ℕ, 
  (1000 < N ∧ N < 2000) ∧ 
  (N % 2 = 1) ∧ (N % 3 = 1) ∧ (N % 4 = 1) ∧ 
  (N % 5 = 1) ∧ (N % 6 = 1) ∧ (N % 7 = 1) ∧ (N % 8 = 1) ∧ 
  (∃ p : ℕ, p = 41 ∧ p > 1 ∧ p < N ∧ N % p = 0) :=
sorry

end find_piles_l76_76055


namespace average_birds_seen_l76_76750

def MarcusBirds : Nat := 7
def HumphreyBirds : Nat := 11
def DarrelBirds : Nat := 9
def IsabellaBirds : Nat := 15

def totalBirds : Nat := MarcusBirds + HumphreyBirds + DarrelBirds + IsabellaBirds
def numberOfIndividuals : Nat := 4

theorem average_birds_seen : (totalBirds / numberOfIndividuals : Real) = 10.5 := 
by
  -- Proof skipped
  sorry

end average_birds_seen_l76_76750


namespace ratio_of_cubes_l76_76945

/-- A cubical block of metal weighs 7 pounds. Another cube of the same metal, with sides of a certain ratio longer, weighs 56 pounds. Prove that the ratio of the side length of the second cube to the first cube is 2:1. --/
theorem ratio_of_cubes (s r : ℝ) (weight1 weight2 : ℝ)
  (h1 : weight1 = 7) (h2 : weight2 = 56)
  (h_vol1 : weight1 = s^3)
  (h_vol2 : weight2 = (r * s)^3) :
  r = 2 := 
sorry

end ratio_of_cubes_l76_76945


namespace masha_nonnegative_l76_76472

theorem masha_nonnegative (a b c d : ℝ) (h1 : a + b = c * d) (h2 : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := 
by
  -- Proof is omitted
  sorry

end masha_nonnegative_l76_76472


namespace chocolate_chips_per_family_member_l76_76722

def total_family_members : ℕ := 4
def batches_choco_chip : ℕ := 3
def batches_double_choco_chip : ℕ := 2
def batches_white_choco_chip : ℕ := 1
def cookies_per_batch_choco_chip : ℕ := 12
def cookies_per_batch_double_choco_chip : ℕ := 10
def cookies_per_batch_white_choco_chip : ℕ := 15
def choco_chips_per_cookie_choco_chip : ℕ := 2
def choco_chips_per_cookie_double_choco_chip : ℕ := 4
def choco_chips_per_cookie_white_choco_chip : ℕ := 3

theorem chocolate_chips_per_family_member :
  (batches_choco_chip * cookies_per_batch_choco_chip * choco_chips_per_cookie_choco_chip +
   batches_double_choco_chip * cookies_per_batch_double_choco_chip * choco_chips_per_cookie_double_choco_chip +
   batches_white_choco_chip * cookies_per_batch_white_choco_chip * choco_chips_per_cookie_white_choco_chip) / 
   total_family_members = 49 :=
by
  sorry

end chocolate_chips_per_family_member_l76_76722


namespace range_of_S_l76_76241

theorem range_of_S (x y : ℝ) (h : 2 * x^2 + 3 * y^2 = 1) (S : ℝ) (hS : S = 3 * x^2 - 2 * y^2) :
  -2 / 3 < S ∧ S ≤ 3 / 2 :=
sorry

end range_of_S_l76_76241


namespace count_shapes_in_figure_l76_76093

-- Definitions based on the conditions
def firstLayerTriangles : Nat := 3
def secondLayerSquares : Nat := 2
def thirdLayerLargeTriangle : Nat := 1
def totalSmallTriangles := firstLayerTriangles
def totalLargeTriangles := thirdLayerLargeTriangle
def totalTriangles := totalSmallTriangles + totalLargeTriangles
def totalSquares := secondLayerSquares

-- Lean 4 statement to prove the problem
theorem count_shapes_in_figure : totalTriangles = 4 ∧ totalSquares = 2 :=
by {
  -- The proof is not required, so we use sorry to skip it.
  sorry
}

end count_shapes_in_figure_l76_76093


namespace max_area_of_fenced_rectangle_l76_76648

theorem max_area_of_fenced_rectangle (x y : ℝ) (h : x + y = 30) : x * y ≤ 225 :=
by {
  sorry
}

end max_area_of_fenced_rectangle_l76_76648


namespace powers_of_two_div7_l76_76203

theorem powers_of_two_div7 (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := sorry

end powers_of_two_div7_l76_76203


namespace volume_tetrahedron_375sqrt2_l76_76373

noncomputable def tetrahedronVolume (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle_ABC_BCD : ℝ) : ℝ :=
  let h_BCD := (2 * area_BCD) / BC
  let h_D_ABD := h_BCD * Real.sin angle_ABC_BCD
  (1 / 3) * area_ABC * h_D_ABD

theorem volume_tetrahedron_375sqrt2 :
  tetrahedronVolume 150 90 12 (Real.pi / 4) = 375 * Real.sqrt 2 := by
  sorry

end volume_tetrahedron_375sqrt2_l76_76373


namespace no_solution_l76_76817

theorem no_solution (x : ℝ) (h₁ : x ≠ -1/3) (h₂ : x ≠ -4/5) :
  (2 * x - 4) / (3 * x + 1) ≠ (2 * x - 10) / (5 * x + 4) := 
sorry

end no_solution_l76_76817


namespace factorial_fraction_simplification_l76_76738

theorem factorial_fraction_simplification : 
  (4 * (Nat.factorial 6) + 24 * (Nat.factorial 5)) / (Nat.factorial 7) = 8 / 7 :=
by
  sorry

end factorial_fraction_simplification_l76_76738


namespace scientific_notation_of_3100000_l76_76005

theorem scientific_notation_of_3100000 :
  ∃ (a : ℝ) (n : ℤ), 3100000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.1 ∧ n = 6 :=
  sorry

end scientific_notation_of_3100000_l76_76005


namespace emily_cards_l76_76025

theorem emily_cards (initial_cards : ℕ) (total_cards : ℕ) (given_cards : ℕ) 
  (h1 : initial_cards = 63) (h2 : total_cards = 70) 
  (h3 : total_cards = initial_cards + given_cards) : 
  given_cards = 7 := 
by 
  sorry

end emily_cards_l76_76025


namespace number_of_shoes_outside_library_l76_76452

-- Define the conditions
def number_of_people : ℕ := 10
def shoes_per_person : ℕ := 2

-- Define the proof that the number of shoes kept outside the library is 20.
theorem number_of_shoes_outside_library : number_of_people * shoes_per_person = 20 :=
by
  -- Proof left as sorry because the proof steps are not required
  sorry

end number_of_shoes_outside_library_l76_76452


namespace speed_of_stream_l76_76008

theorem speed_of_stream (b s : ℕ) 
  (h1 : b + s = 42) 
  (h2 : b - s = 24) :
  s = 9 := by sorry

end speed_of_stream_l76_76008


namespace evaluate_expression_l76_76019

theorem evaluate_expression : (1 / (1 - 1 / (3 + 1 / 4))) = (13 / 9) :=
by
  sorry

end evaluate_expression_l76_76019


namespace sum_a_b_eq_34_over_3_l76_76355

theorem sum_a_b_eq_34_over_3 (a b: ℚ)
  (h1 : 2 * a + 5 * b = 43)
  (h2 : 8 * a + 2 * b = 50) :
  a + b = 34 / 3 :=
sorry

end sum_a_b_eq_34_over_3_l76_76355


namespace males_in_sample_l76_76067

theorem males_in_sample (total_employees female_employees sample_size : ℕ) 
  (h1 : total_employees = 300)
  (h2 : female_employees = 160)
  (h3 : sample_size = 15)
  (h4 : (female_employees * sample_size) / total_employees = 8) :
  sample_size - ((female_employees * sample_size) / total_employees) = 7 :=
by
  sorry

end males_in_sample_l76_76067


namespace min_AP_squared_sum_value_l76_76201

-- Definitions based on given problem conditions
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 4
def D : ℝ := 7
def E : ℝ := 15

def distance_squared (x y : ℝ) : ℝ := (x - y)^2

noncomputable def min_AP_squared_sum (r : ℝ) : ℝ :=
  r^2 + distance_squared r B + distance_squared r C + distance_squared r D + distance_squared r E

theorem min_AP_squared_sum_value : ∃ (r : ℝ), (min_AP_squared_sum r) = 137.2 :=
by
  existsi 5.6
  sorry

end min_AP_squared_sum_value_l76_76201


namespace inverse_proportion_y_relation_l76_76187

theorem inverse_proportion_y_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (hA : y₁ = -4 / x₁) 
  (hB : y₂ = -4 / x₂)
  (h₁ : x₁ < 0) 
  (h₂ : 0 < x₂) : 
  y₁ > y₂ := 
sorry

end inverse_proportion_y_relation_l76_76187


namespace inequality_solution_range_l76_76971

variable (a : ℝ)

def f (x : ℝ) := 2 * x^2 - 8 * x - 4

theorem inequality_solution_range :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ f x - a > 0) ↔ a < -4 := 
by
  sorry

end inequality_solution_range_l76_76971


namespace calculate_interest_rate_l76_76550

variables (A : ℝ) (R : ℝ)

-- Conditions as definitions in Lean 4
def compound_interest_condition (A : ℝ) (R : ℝ) : Prop :=
  (A * (1 + R)^20 = 4 * A)

-- Theorem statement
theorem calculate_interest_rate (A : ℝ) (R : ℝ) (h : compound_interest_condition A R) : 
  R = (4)^(1/20) - 1 := 
sorry

end calculate_interest_rate_l76_76550


namespace f_bound_l76_76998

theorem f_bound (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1) 
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) : ∀ x : ℝ, |f x| ≤ 2 + x^2 :=
by
  sorry

end f_bound_l76_76998


namespace Erik_ate_pie_l76_76073

theorem Erik_ate_pie (Frank_ate Erik_ate more_than: ℝ) (h1: Frank_ate = 0.3333333333333333)
(h2: more_than = 0.3333333333333333)
(h3: Erik_ate = Frank_ate + more_than) : Erik_ate = 0.6666666666666666 :=
by
  sorry

end Erik_ate_pie_l76_76073


namespace abs_neg_three_halves_l76_76449

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end abs_neg_three_halves_l76_76449


namespace complex_number_quadrant_l76_76633

def i_squared : ℂ := -1

def z (i : ℂ) : ℂ := (-2 + i) * i^5

def in_quadrant_III (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem complex_number_quadrant 
  (i : ℂ) (hi : i^2 = -1) (z_val : z i = (-2 + i) * i^5) :
  in_quadrant_III (z i) :=
sorry

end complex_number_quadrant_l76_76633


namespace painter_total_fence_painted_l76_76224

theorem painter_total_fence_painted : 
  ∀ (L T W Th F : ℕ), 
  (T = W) → (W = Th) → 
  (L = T / 2) → 
  (F = 2 * T * (6 / 8)) → 
  (F = L + 300) → 
  (L + T + W + Th + F = 1500) :=
by
  sorry

end painter_total_fence_painted_l76_76224


namespace weigh_grain_with_inaccurate_scales_l76_76652

theorem weigh_grain_with_inaccurate_scales
  (inaccurate_scales : ℕ → ℕ → Prop)
  (correct_weight : ℕ)
  (bag_of_grain : ℕ → Prop)
  (balanced : ∀ a b : ℕ, inaccurate_scales a b → a = b := sorry)
  : ∃ grain_weight : ℕ, bag_of_grain grain_weight ∧ grain_weight = correct_weight :=
sorry

end weigh_grain_with_inaccurate_scales_l76_76652


namespace max_slope_tangent_eqn_l76_76483

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem max_slope_tangent_eqn (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
    (∃ m b, m = Real.sqrt 2 ∧ b = -Real.sqrt 2 * (Real.pi / 4) ∧ 
    (∀ y, y = m * x + b)) :=
sorry

end max_slope_tangent_eqn_l76_76483


namespace number_of_non_congruent_triangles_perimeter_18_l76_76589

theorem number_of_non_congruent_triangles_perimeter_18 : 
  {n : ℕ // n = 9} := 
sorry

end number_of_non_congruent_triangles_perimeter_18_l76_76589


namespace sum_of_fractions_removal_l76_76858

theorem sum_of_fractions_removal :
  (1 / 3 + 1 / 6 + 1 / 9 + 1 / 12 + 1 / 15 + 1 / 18 + 1 / 21) 
  - (1 / 12 + 1 / 21) = 3 / 4 := 
 by sorry

end sum_of_fractions_removal_l76_76858


namespace combined_average_age_l76_76480

theorem combined_average_age 
    (avgA : ℕ → ℕ → ℕ) -- defines the average function
    (avgA_cond : avgA 6 240 = 40) 
    (avgB : ℕ → ℕ → ℕ)
    (avgB_cond : avgB 4 100 = 25) 
    (combined_total_age : ℕ := 340) 
    (total_people : ℕ := 10) : avgA (total_people) (combined_total_age) = 34 := 
by
  sorry

end combined_average_age_l76_76480


namespace mark_age_l76_76020

-- Definitions based on the conditions in the problem
variables (M J P : ℕ)  -- Current ages of Mark, John, and their parents respectively

-- Condition definitions
def condition1 : Prop := J = M - 10
def condition2 : Prop := P = 5 * J
def condition3 : Prop := P - 22 = M

-- The theorem to prove the correct answer
theorem mark_age : condition1 M J ∧ condition2 J P ∧ condition3 P M → M = 18 := by
  sorry

end mark_age_l76_76020


namespace find_original_price_l76_76126

-- Define the conditions for the problem
def original_price (P : ℝ) : Prop :=
  0.90 * P = 1620

-- Prove the original price P
theorem find_original_price (P : ℝ) (h : original_price P) : P = 1800 :=
by
  -- The proof goes here
  sorry

end find_original_price_l76_76126


namespace intersection_S_T_eq_T_l76_76358

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l76_76358


namespace integer_satisfying_conditions_l76_76094

theorem integer_satisfying_conditions :
  {a : ℤ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = {1, 11, 16, 36, 46, 51, 71, 81, 86} :=
by
  sorry

end integer_satisfying_conditions_l76_76094


namespace tv_horizontal_length_l76_76339

-- Conditions
def is_rectangular_tv (width height : ℝ) : Prop :=
width / height = 9 / 12

def diagonal_is (d : ℝ) : Prop :=
d = 32

-- Theorem to prove
theorem tv_horizontal_length (width height diagonal : ℝ) 
(h1 : is_rectangular_tv width height) 
(h2 : diagonal_is diagonal) : 
width = 25.6 := by 
sorry

end tv_horizontal_length_l76_76339


namespace ellipse_hyperbola_tangent_l76_76875

variable {x y m : ℝ}

theorem ellipse_hyperbola_tangent (h : ∃ x y, x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y + 1)^2 = 1) : m = 2 := 
by 
  sorry

end ellipse_hyperbola_tangent_l76_76875


namespace coefficients_divisible_by_seven_l76_76035

theorem coefficients_divisible_by_seven {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  a % 7 = 0 ∧ b % 7 = 0 ∧ c % 7 = 0 ∧ d % 7 = 0 ∧ e % 7 = 0 := 
  sorry

end coefficients_divisible_by_seven_l76_76035


namespace isosceles_trapezoid_height_l76_76060

theorem isosceles_trapezoid_height (S h : ℝ) (h_nonneg : 0 ≤ h) 
  (diag_perpendicular : S = (1 / 2) * h^2) : h = Real.sqrt S :=
by
  sorry

end isosceles_trapezoid_height_l76_76060


namespace abs_neg_three_eq_three_l76_76222

theorem abs_neg_three_eq_three : abs (-3) = 3 :=
by sorry

end abs_neg_three_eq_three_l76_76222


namespace probability_20_correct_l76_76411

noncomputable def probability_sum_20_dodecahedral : ℚ :=
  let num_faces := 12
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_20_correct : probability_sum_20_dodecahedral = 5 / 144 := 
by 
  sorry

end probability_20_correct_l76_76411


namespace find_distance_l76_76002

-- Definitions based on given conditions
def speed : ℝ := 40 -- in km/hr
def time : ℝ := 6 -- in hours

-- Theorem statement
theorem find_distance (speed : ℝ) (time : ℝ) : speed = 40 → time = 6 → speed * time = 240 :=
by
  intros h1 h2
  rw [h1, h2]
  -- skipping the proof with sorry
  sorry

end find_distance_l76_76002


namespace students_meet_time_l76_76925

theorem students_meet_time :
  ∀ (distance rate1 rate2 : ℝ),
    distance = 350 ∧ rate1 = 1.6 ∧ rate2 = 1.9 →
    distance / (rate1 + rate2) = 100 := by
  sorry

end students_meet_time_l76_76925


namespace factorial_divisibility_l76_76813

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : a ≥ 2 * b + 1 :=
sorry

end factorial_divisibility_l76_76813


namespace charles_finishes_in_11_days_l76_76207

theorem charles_finishes_in_11_days : 
  ∀ (total_pages : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) (pages_thu : ℕ) 
    (does_not_read_on_weekend : Prop),
  total_pages = 96 →
  pages_mon = 7 →
  pages_tue = 12 →
  pages_wed = 10 →
  pages_thu = 6 →
  does_not_read_on_weekend →
  ∃ days_to_finish : ℕ, days_to_finish = 11 :=
by
  intros
  sorry

end charles_finishes_in_11_days_l76_76207


namespace min_major_axis_l76_76039

theorem min_major_axis (a b c : ℝ) (h1 : b * c = 1) (h2 : a = Real.sqrt (b^2 + c^2)) : 2 * a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_l76_76039


namespace cost_price_for_one_meter_l76_76160

variable (meters_sold : Nat) (selling_price : Nat) (loss_per_meter : Nat) (total_cost_price : Nat)
variable (cost_price_per_meter : Rat)

theorem cost_price_for_one_meter (h1 : meters_sold = 200)
                                  (h2 : selling_price = 12000)
                                  (h3 : loss_per_meter = 12)
                                  (h4 : total_cost_price = selling_price + loss_per_meter * meters_sold)
                                  (h5 : cost_price_per_meter = total_cost_price / meters_sold) :
  cost_price_per_meter = 72 := by
  sorry

end cost_price_for_one_meter_l76_76160


namespace probability_of_three_primes_out_of_six_l76_76291

-- Define the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

-- Given six 12-sided fair dice
def total_dice : ℕ := 6
def sides : ℕ := 12

-- Probability of rolling a prime number on one die
def prime_probability : ℚ := 5 / 12

-- Probability of rolling a non-prime number on one die
def non_prime_probability : ℚ := 7 / 12

-- Number of ways to choose 3 dice from 6 to show a prime number
def combination (n k : ℕ) : ℕ := n.choose k
def choose_3_out_of_6 : ℕ := combination total_dice 3

-- Combined probability for exactly 3 primes and 3 non-primes
def combined_probability : ℚ :=
  (prime_probability ^ 3) * (non_prime_probability ^ 3)

-- Total probability
def total_probability : ℚ :=
  choose_3_out_of_6 * combined_probability

-- Main theorem statement
theorem probability_of_three_primes_out_of_six :
  total_probability = 857500 / 5177712 :=
by
  sorry

end probability_of_three_primes_out_of_six_l76_76291


namespace find_difference_l76_76288

theorem find_difference (x y : ℚ) (h₁ : x + y = 520) (h₂ : x / y = 3 / 4) : y - x = 520 / 7 :=
by
  sorry

end find_difference_l76_76288


namespace negation_equivalent_statement_l76_76306

theorem negation_equivalent_statement (x y : ℝ) :
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0)) :=
sorry

end negation_equivalent_statement_l76_76306


namespace value_of_k_l76_76362

noncomputable def find_k (x1 x2 : ℝ) (k : ℝ) : Prop :=
  (2 * x1^2 + k * x1 - 2 = 0) ∧ (2 * x2^2 + k * x2 - 2 = 0) ∧ ((x1 - 2) * (x2 - 2) = 10)

theorem value_of_k (x1 x2 : ℝ) (k : ℝ) (h : find_k x1 x2 k) : k = 7 :=
sorry

end value_of_k_l76_76362


namespace problem_statement_l76_76445

theorem problem_statement (a b : ℝ) (h1 : 1/a < 1/b) (h2 : 1/b < 0) :
  (a + b < a * b) ∧ ¬(a^2 > b^2) ∧ ¬(a < b) ∧ (b/a + a/b > 2) := by
  sorry

end problem_statement_l76_76445


namespace odds_against_y_winning_l76_76896

/- 
   Define the conditions: 
   odds_w: odds against W winning is 4:1
   odds_x: odds against X winning is 5:3
-/
def odds_w : ℚ := 4 / 1
def odds_x : ℚ := 5 / 3

/- 
   Calculate the odds against Y winning 
-/
theorem odds_against_y_winning : 
  (4 / (4 + 1)) + (5 / (5 + 3)) < 1 ∧
  (1 - ((4 / (4 + 1)) + (5 / (5 + 3)))) = 17 / 40 ∧
  ((1 - (17 / 40)) / (17 / 40)) = 23 / 17 := by
  sorry

end odds_against_y_winning_l76_76896


namespace equation_of_tangent_circle_l76_76725

-- Define the point and conditional tangency
def center : ℝ × ℝ := (5, 4)
def tangent_to_x_axis : Prop := true -- Placeholder for the tangency condition, which is encoded in our reasoning

-- Define the proof statement
theorem equation_of_tangent_circle :
  (∀ (x y : ℝ), tangent_to_x_axis → 
  (center = (5, 4)) → 
  ((x - 5) ^ 2 + (y - 4) ^ 2 = 16)) := 
sorry

end equation_of_tangent_circle_l76_76725


namespace no_integer_solution_l76_76827

theorem no_integer_solution :
  ¬(∃ x : ℤ, 7 - 3 * (x^2 - 2) > 19) :=
by
  sorry

end no_integer_solution_l76_76827


namespace growth_rate_yield_per_acre_l76_76576

theorem growth_rate_yield_per_acre (x : ℝ) (a_i y_i y_f : ℝ) (h1 : a_i = 5) (h2 : y_i = 10000) (h3 : y_f = 30000) 
  (h4 : y_f = 5 * (1 + 2 * x) * (y_i / a_i) * (1 + x)) : x = 0.5 := 
by
  -- Insert the proof here
  sorry

end growth_rate_yield_per_acre_l76_76576


namespace youngest_sibling_age_l76_76386

theorem youngest_sibling_age
  (Y : ℕ)
  (h1 : Y + (Y + 3) + (Y + 6) + (Y + 7) = 120) :
  Y = 26 :=
by
  -- proof steps would be here 
  sorry

end youngest_sibling_age_l76_76386


namespace x_minus_y_solution_l76_76412

theorem x_minus_y_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end x_minus_y_solution_l76_76412


namespace joes_fast_food_cost_l76_76572

noncomputable def cost_of_sandwich (n : ℕ) : ℝ := n * 4
noncomputable def cost_of_soda (m : ℕ) : ℝ := m * 1.50
noncomputable def total_cost (n m : ℕ) : ℝ :=
  if n >= 10 then cost_of_sandwich n - 5 + cost_of_soda m else cost_of_sandwich n + cost_of_soda m

theorem joes_fast_food_cost :
  total_cost 10 6 = 44 := by
  sorry

end joes_fast_food_cost_l76_76572


namespace positive_integer_with_four_smallest_divisors_is_130_l76_76273

theorem positive_integer_with_four_smallest_divisors_is_130:
  ∃ n : ℕ, ∀ p1 p2 p3 p4 : ℕ, 
    n = p1^2 + p2^2 + p3^2 + p4^2 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
    ∀ p : ℕ, p ∣ n → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4) → 
    n = 130 :=
  by
  sorry

end positive_integer_with_four_smallest_divisors_is_130_l76_76273


namespace initial_logs_l76_76324

theorem initial_logs (x : ℕ) (h1 : x - 3 - 3 - 3 + 2 + 2 + 2 = 3) : x = 6 := by
  sorry

end initial_logs_l76_76324


namespace solve_for_y_l76_76622

theorem solve_for_y (x y : ℝ) (h : 2 * y - 4 * x + 5 = 0) : y = 2 * x - 2.5 :=
sorry

end solve_for_y_l76_76622


namespace find_smaller_integer_l76_76673

theorem find_smaller_integer
  (x y : ℤ)
  (h1 : x + y = 30)
  (h2 : 2 * y = 5 * x - 10) :
  x = 10 :=
by
  -- proof would go here
  sorry

end find_smaller_integer_l76_76673


namespace probability_reach_2C_l76_76723

noncomputable def f (x C : ℝ) : ℝ :=
  x / (2 * C)

theorem probability_reach_2C (x C : ℝ) (hC : 0 < C) (hx : 0 < x ∧ x < 2 * C) :
  f x C = x / (2 * C) := 
by
  sorry

end probability_reach_2C_l76_76723


namespace determine_a_for_unique_solution_of_quadratic_l76_76645

theorem determine_a_for_unique_solution_of_quadratic :
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0} = {0, 2} :=
sorry

end determine_a_for_unique_solution_of_quadratic_l76_76645


namespace area_of_square_l76_76361

theorem area_of_square 
  (a : ℝ)
  (h : 4 * a = 28) :
  a^2 = 49 :=
sorry

end area_of_square_l76_76361


namespace floor_value_correct_l76_76508

def calc_floor_value : ℤ :=
  let a := (15 : ℚ) / 8
  let b := a^2
  let c := (225 : ℚ) / 64
  let d := 4
  let e := (19 : ℚ) / 5
  let f := d + e
  ⌊f⌋

theorem floor_value_correct : calc_floor_value = 7 := by
  sorry

end floor_value_correct_l76_76508


namespace find_C_line_MN_l76_76040

def point := (ℝ × ℝ)

-- Given points A and B
def A : point := (5, -2)
def B : point := (7, 3)

-- Conditions: M is the midpoint of AC and is on the y-axis
def M_on_y_axis (M : point) (A C : point) : Prop :=
  M.1 = 0 ∧ M.2 = (A.2 + C.2) / 2

-- Conditions: N is the midpoint of BC and is on the x-axis
def N_on_x_axis (N : point) (B C : point) : Prop :=
  N.1 = (B.1 + C.1) / 2 ∧ N.2 = 0

-- Coordinates of point C
theorem find_C (C : point)
  (M : point) (N : point)
  (hM : M_on_y_axis M A C)
  (hN : N_on_x_axis N B C) : C = (-5, -8) := sorry

-- Equation of line MN
theorem line_MN (M N : point)
  (MN_eq : M_on_y_axis M A (-5, -8) ∧ N_on_x_axis N B (-5, -8)) :
   ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ ((y = M.2) ∧ (x = M.1)) ∨ ((y = N.2) ∧ (x = N.1))) ∧ m = (3/2) ∧ b = 0 := sorry

end find_C_line_MN_l76_76040


namespace abs_difference_21st_term_l76_76920

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 - 14 * (n - 1)

theorem abs_difference_21st_term :
  |sequence_C 21 - sequence_D 21| = 520 := by
  sorry

end abs_difference_21st_term_l76_76920


namespace min_value_sum_l76_76787

def positive_real (x : ℝ) : Prop := x > 0

theorem min_value_sum (x y : ℝ) (hx : positive_real x) (hy : positive_real y)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 6) : x + y ≥ 20 :=
sorry

end min_value_sum_l76_76787


namespace car_distance_l76_76393

noncomputable def distance_covered (S : ℝ) (T : ℝ) (new_speed : ℝ) : ℝ :=
  S * T

theorem car_distance (S : ℝ) (T : ℝ) (new_time : ℝ) (new_speed : ℝ)
  (h1 : T = 12)
  (h2 : new_time = (3/4) * T)
  (h3 : new_speed = 60)
  (h4 : distance_covered new_speed new_time = 540) :
    distance_covered S T = 540 :=
by
  sorry

end car_distance_l76_76393


namespace remainder_52_l76_76268

theorem remainder_52 (x y : ℕ) (k m : ℤ)
  (h₁ : x = 246 * k + 37)
  (h₂ : y = 357 * m + 53) :
  (x + y + 97) % 123 = 52 := by
  sorry

end remainder_52_l76_76268


namespace find_opposite_pair_l76_76824

def is_opposite (x y : ℤ) : Prop := x = -y

theorem find_opposite_pair :
  ¬is_opposite 4 4 ∧ ¬is_opposite 2 2 ∧ ¬is_opposite (-8) (-8) ∧ is_opposite 4 (-4) := 
by
  sorry

end find_opposite_pair_l76_76824


namespace evaluate_f_at_3_over_4_l76_76678

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ := (1 - y) / y

theorem evaluate_f_at_3_over_4 (h : g (x : ℝ) = 1 - x^2) (x_ne_zero : x ≠ 0) :
  f (3 / 4) = 3 :=
by
  sorry

end evaluate_f_at_3_over_4_l76_76678


namespace john_total_expense_l76_76500

-- Define variables
variables (M D : ℝ)

-- Define the conditions
axiom cond1 : M = 20 * D
axiom cond2 : M = 24 * (D - 3)

-- State the theorem to prove
theorem john_total_expense : M = 360 :=
by
  -- Add the proof steps here
  sorry

end john_total_expense_l76_76500


namespace number_of_bowls_l76_76247

theorem number_of_bowls (n : ℕ) (h1 : 12 * 8 = 96) (h2 : ∀ t : ℕ, t = 6 * n -> t = 96) : n = 16 := by
  sorry

end number_of_bowls_l76_76247


namespace evaluate_expression_l76_76052

theorem evaluate_expression (x y : ℚ) (hx : x = 4 / 3) (hy : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 40 :=
by
  rw [hx, hy]
  sorry

end evaluate_expression_l76_76052


namespace triangular_faces_area_of_pyramid_l76_76015

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l76_76015


namespace uncolored_area_of_rectangle_l76_76372

theorem uncolored_area_of_rectangle :
  let width := 30
  let length := 50
  let radius := width / 4
  let rectangle_area := width * length
  let circle_area := π * (radius ^ 2)
  let total_circles_area := 4 * circle_area
  rectangle_area - total_circles_area = 1500 - 225 * π := by
  sorry

end uncolored_area_of_rectangle_l76_76372


namespace remaining_liquid_weight_l76_76765

theorem remaining_liquid_weight 
  (liqX_content : ℝ := 0.20)
  (water_content : ℝ := 0.80)
  (initial_solution : ℝ := 8)
  (evaporated_water : ℝ := 2)
  (added_solution : ℝ := 2)
  (new_solution_fraction : ℝ := 0.25) :
  ∃ (remaining_liquid : ℝ), remaining_liquid = 6 := 
by
  -- Skip the proof to ensure the statement is built successfully
  sorry

end remaining_liquid_weight_l76_76765


namespace diamonds_balance_emerald_l76_76230

theorem diamonds_balance_emerald (D E : ℝ) (h1 : 9 * D = 4 * E) (h2 : 9 * D + E = 4 * E) : 3 * D = E := by
  sorry

end diamonds_balance_emerald_l76_76230


namespace ratio_blue_gill_to_bass_l76_76289

theorem ratio_blue_gill_to_bass (bass trout blue_gill : ℕ) 
  (h1 : bass = 32)
  (h2 : trout = bass / 4)
  (h3 : bass + trout + blue_gill = 104) 
: blue_gill / bass = 2 := 
sorry

end ratio_blue_gill_to_bass_l76_76289


namespace molecular_weight_of_N2O5_is_correct_l76_76720

-- Definitions for atomic weights
def atomic_weight_N : ℚ := 14.01
def atomic_weight_O : ℚ := 16.00

-- Define the molecular weight calculation for N2O5
def molecular_weight_N2O5 : ℚ := (2 * atomic_weight_N) + (5 * atomic_weight_O)

-- The theorem to prove
theorem molecular_weight_of_N2O5_is_correct : molecular_weight_N2O5 = 108.02 := by
  -- Proof here
  sorry

end molecular_weight_of_N2O5_is_correct_l76_76720


namespace tidal_power_station_location_l76_76808

-- Define the conditions
def tidal_power_plants : ℕ := 9
def first_bidirectional_plant := 1980
def significant_bidirectional_plant_location : String := "Jiangxia"
def largest_bidirectional_plant : Prop := true

-- Assumptions based on conditions
axiom china_has_9_tidal_power_plants : tidal_power_plants = 9
axiom first_bidirectional_in_1980 : (first_bidirectional_plant = 1980) -> significant_bidirectional_plant_location = "Jiangxia"
axiom largest_bidirectional_in_world : largest_bidirectional_plant

-- Definition of the problem
theorem tidal_power_station_location : significant_bidirectional_plant_location = "Jiangxia" :=
by
  sorry

end tidal_power_station_location_l76_76808


namespace sequence_formula_l76_76533

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 3
  | 2 => 6
  | 3 => 10
  | _ => sorry  -- The pattern is more general

theorem sequence_formula (n : ℕ) : a n = (n * (n + 1)) / 2 := 
  sorry

end sequence_formula_l76_76533


namespace net_change_salary_l76_76135

/-- Given an initial salary S and a series of percentage changes:
    20% increase, 10% decrease, 15% increase, and 5% decrease,
    prove that the net change in salary is 17.99%. -/
theorem net_change_salary (S : ℝ) :
  (1.20 * 0.90 * 1.15 * 0.95 - 1) * S = 0.1799 * S :=
sorry

end net_change_salary_l76_76135


namespace area_of_square_land_l76_76295

-- Define the problem conditions
variable (A P : ℕ)

-- Define the main theorem statement: proving area A given the conditions
theorem area_of_square_land (h₁ : 5 * A = 10 * P + 45) (h₂ : P = 36) : A = 81 := by
  sorry

end area_of_square_land_l76_76295


namespace desired_average_sale_is_5600_l76_76401

-- Define the sales for five consecutive months
def sale1 : ℕ := 5266
def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029

-- Define the required sale for the sixth month
def sale6 : ℕ := 4937

-- Calculate total sales for the first five months
def total_five_months := sale1 + sale2 + sale3 + sale4 + sale5

-- Calculate total sales for six months
def total_six_months := total_five_months + sale6

-- Calculate the desired average sale for six months
def desired_average := total_six_months / 6

-- The theorem statement: desired average sale for the six months
theorem desired_average_sale_is_5600 : desired_average = 5600 :=
by
  sorry

end desired_average_sale_is_5600_l76_76401


namespace minimum_m_n_1978_l76_76610

-- Define the conditions given in the problem
variables (m n : ℕ) (h1 : n > m) (h2 : m > 1)
-- Define the condition that the last three digits of 1978^m and 1978^n are identical
def same_last_three_digits (a b : ℕ) : Prop :=
  (a % 1000 = b % 1000)

-- Define the problem statement: under the conditions, prove that m + n = 106 when minimized
theorem minimum_m_n_1978 (h : same_last_three_digits (1978^m) (1978^n)) : m + n = 106 :=
sorry   -- Proof will be provided here

end minimum_m_n_1978_l76_76610


namespace find_triples_l76_76275

theorem find_triples (a b c : ℝ) : 
  a + b + c = 14 ∧ a^2 + b^2 + c^2 = 84 ∧ a^3 + b^3 + c^3 = 584 ↔ (a = 4 ∧ b = 2 ∧ c = 8) ∨ (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 2 ∧ c = 4) :=
by
  sorry

end find_triples_l76_76275


namespace S_11_is_22_l76_76933

-- Definitions and conditions
variable (a_1 d : ℤ) -- first term and common difference of the arithmetic sequence
noncomputable def S (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- The given condition
variable (h : S a_1 d 8 - S a_1 d 3 = 10)

-- The proof goal
theorem S_11_is_22 : S a_1 d 11 = 22 :=
by
  sorry

end S_11_is_22_l76_76933


namespace problem_l76_76188

noncomputable def y := 2 + Real.sqrt 3

theorem problem (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : y = c + Real.sqrt d)
  (hy_eq : y^2 + 2*y + 2/y + 1/y^2 = 20) : c + d = 5 :=
  sorry

end problem_l76_76188


namespace grims_groks_zeets_l76_76059

variable {T : Type}
variable (Groks Zeets Grims Snarks : Set T)

-- Given conditions as definitions in Lean 4
variable (h1 : Groks ⊆ Zeets)
variable (h2 : Grims ⊆ Zeets)
variable (h3 : Snarks ⊆ Groks)
variable (h4 : Grims ⊆ Snarks)

-- The statement to be proved
theorem grims_groks_zeets : Grims ⊆ Groks ∧ Grims ⊆ Zeets := by
  sorry

end grims_groks_zeets_l76_76059


namespace min_value_y1_y2_sq_l76_76007

theorem min_value_y1_y2_sq (k : ℝ) (y1 y2 : ℝ) :
  ∃ y1 y2, y1 + y2 = 4 / k ∧ y1 * y2 = -4 ∧ y1^2 + y2^2 = 8 :=
sorry

end min_value_y1_y2_sq_l76_76007


namespace smallest_number_among_bases_l76_76537

noncomputable def convert_base_9 (n : ℕ) : ℕ :=
match n with
| 85 => 8 * 9 + 5
| _ => 0

noncomputable def convert_base_4 (n : ℕ) : ℕ :=
match n with
| 1000 => 1 * 4^3
| _ => 0

noncomputable def convert_base_2 (n : ℕ) : ℕ :=
match n with
| 111111 => 1 * 2^6 - 1
| _ => 0

theorem smallest_number_among_bases:
  min (min (convert_base_9 85) (convert_base_4 1000)) (convert_base_2 111111) = convert_base_2 111111 :=
by {
  sorry
}

end smallest_number_among_bases_l76_76537


namespace find_fraction_l76_76560

theorem find_fraction (n d : ℕ) (h1 : n / (d + 1) = 1 / 2) (h2 : (n + 1) / d = 1) : n / d = 2 / 3 := 
by 
  sorry

end find_fraction_l76_76560


namespace money_left_for_lunch_and_snacks_l76_76773

-- Definitions according to the conditions
def ticket_cost_per_person : ℝ := 5
def bus_fare_one_way_per_person : ℝ := 1.50
def total_budget : ℝ := 40
def number_of_people : ℝ := 2

-- The proposition to be proved
theorem money_left_for_lunch_and_snacks : 
  let total_zoo_cost := ticket_cost_per_person * number_of_people
  let total_bus_fare := bus_fare_one_way_per_person * number_of_people * 2
  let total_expense := total_zoo_cost + total_bus_fare
  total_budget - total_expense = 24 :=
by
  sorry

end money_left_for_lunch_and_snacks_l76_76773


namespace Q_subset_P_l76_76167

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l76_76167


namespace vector_combination_l76_76814

-- Definitions of the given vectors and condition of parallelism
def vec_a : (ℝ × ℝ) := (1, -2)
def vec_b (m : ℝ) : (ℝ × ℝ) := (2, m)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 - a.2 * b.1 = 0

-- Goal to prove
theorem vector_combination :
  ∀ m : ℝ, are_parallel vec_a (vec_b m) → 3 * vec_a.1 + 2 * (vec_b m).1 = 7 ∧ 3 * vec_a.2 + 2 * (vec_b m).2 = -14 :=
by
  intros m h_par
  sorry

end vector_combination_l76_76814


namespace area_of_combined_rectangle_l76_76103

theorem area_of_combined_rectangle
  (short_side : ℝ) (num_small_rectangles : ℕ) (total_area : ℝ)
  (h1 : num_small_rectangles = 4)
  (h2 : short_side = 7)
  (h3 : total_area = (3 * short_side + short_side) * (2 * short_side)) :
  total_area = 392 := by
  sorry

end area_of_combined_rectangle_l76_76103


namespace bins_of_vegetables_l76_76369

-- Define the conditions
def total_bins : ℝ := 0.75
def bins_of_soup : ℝ := 0.12
def bins_of_pasta : ℝ := 0.5

-- Define the statement to be proved
theorem bins_of_vegetables :
  total_bins = bins_of_soup + (0.13) + bins_of_pasta := 
sorry

end bins_of_vegetables_l76_76369


namespace inequality_sol_range_t_l76_76249

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem inequality_sol : {x : ℝ | f x > 2} = {x : ℝ | x < -5} ∪ {x : ℝ | 1 < x} :=
sorry

theorem range_t (t : ℝ) : (∀ x : ℝ, f x ≥ t^2 - 11/2 * t) ↔ (1/2 ≤ t ∧ t ≤ 5) :=
sorry

end inequality_sol_range_t_l76_76249


namespace find_smallest_A_divisible_by_51_l76_76776

theorem find_smallest_A_divisible_by_51 :
  ∃ (x y : ℕ), (A = 1100 * x + 11 * y) ∧ 
    (0 ≤ x) ∧ (x ≤ 9) ∧ 
    (0 ≤ y) ∧ (y ≤ 9) ∧ 
    (A % 51 = 0) ∧ 
    (A = 1122) :=
sorry

end find_smallest_A_divisible_by_51_l76_76776


namespace raine_change_l76_76683

noncomputable def price_bracelet : ℝ := 15
noncomputable def price_necklace : ℝ := 10
noncomputable def price_mug : ℝ := 20
noncomputable def price_keychain : ℝ := 5

noncomputable def quantity_bracelet : ℕ := 3
noncomputable def quantity_necklace : ℕ := 2
noncomputable def quantity_mug : ℕ := 1
noncomputable def quantity_keychain : ℕ := 4

noncomputable def discount_rate : ℝ := 0.12

noncomputable def amount_given : ℝ := 100

-- The total cost before discount
noncomputable def total_before_discount : ℝ := 
  quantity_bracelet * price_bracelet + 
  quantity_necklace * price_necklace + 
  quantity_mug * price_mug + 
  quantity_keychain * price_keychain

-- The discount amount
noncomputable def discount_amount : ℝ := total_before_discount * discount_rate

-- The final amount Raine has to pay after discount
noncomputable def final_amount : ℝ := total_before_discount - discount_amount

-- The change Raine gets back
noncomputable def change : ℝ := amount_given - final_amount

theorem raine_change : change = 7.60 := 
by sorry

end raine_change_l76_76683


namespace quadratic_roots_solve_equation_l76_76489

theorem quadratic_roots (a b c : ℝ) (x1 x2 : ℝ) (h : a ≠ 0)
  (root_eq : x1 = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
            ∧ x2 = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a))
  (h_eq : a*x^2 + b*x + c = 0) :
  ∀ x, a*x^2 + b*x + c = 0 → x = x1 ∨ x = x2 :=
by
  sorry -- Proof not given

theorem solve_equation (x : ℝ) :
  7*x*(5*x + 2) = 6*(5*x + 2) ↔ x = -2 / 5 ∨ x = 6 / 7 :=
by
  sorry -- Proof not given

end quadratic_roots_solve_equation_l76_76489


namespace roots_cubed_l76_76199

noncomputable def q (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + b^2 - c^2
noncomputable def p (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * (b^2 + 3 * c^2) * x + (b^2 - c^2)^3 
def x1 (b c : ℝ) := b + c
def x2 (b c : ℝ) := b - c

theorem roots_cubed (b c : ℝ) :
  (q b c (x1 b c) = 0 ∧ q b c (x2 b c) = 0) →
  (p b c ((x1 b c)^3) = 0 ∧ p b c ((x2 b c)^3) = 0) :=
by
  sorry

end roots_cubed_l76_76199


namespace lily_of_the_valley_bushes_needed_l76_76718

theorem lily_of_the_valley_bushes_needed 
  (r l : ℕ) (h_radius : r = 20) (h_length : l = 400) : 
  l / (2 * r) = 10 := 
by 
  sorry

end lily_of_the_valley_bushes_needed_l76_76718


namespace employee_pays_correct_amount_l76_76952

theorem employee_pays_correct_amount
    (wholesale_cost : ℝ)
    (retail_markup : ℝ)
    (employee_discount : ℝ)
    (weekend_discount : ℝ)
    (sales_tax : ℝ)
    (final_price : ℝ) :
    wholesale_cost = 200 →
    retail_markup = 0.20 →
    employee_discount = 0.05 →
    weekend_discount = 0.10 →
    sales_tax = 0.08 →
    final_price = 221.62 :=
by
  intros h0 h1 h2 h3 h4
  sorry

end employee_pays_correct_amount_l76_76952


namespace sum_of_reciprocal_transformed_roots_l76_76151

theorem sum_of_reciprocal_transformed_roots :
  ∀ (a b c : ℝ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    -1 < a ∧ a < 1 ∧
    -1 < b ∧ b < 1 ∧
    -1 < c ∧ c < 1 ∧
    (45 * a ^ 3 - 70 * a ^ 2 + 28 * a - 2 = 0) ∧
    (45 * b ^ 3 - 70 * b ^ 2 + 28 * b - 2 = 0) ∧
    (45 * c ^ 3 - 70 * c ^ 2 + 28 * c - 2 = 0)
  → (1 - a)⁻¹ + (1 - b)⁻¹ + (1 - c)⁻¹ = 13 / 9 := 
by 
  sorry

end sum_of_reciprocal_transformed_roots_l76_76151


namespace find_q_l76_76184

variable (x : ℝ)

def f (x : ℝ) := (5 * x^4 + 15 * x^3 + 30 * x^2 + 10 * x + 10)
def g (x : ℝ) := (2 * x^6 + 4 * x^4 + 10 * x^2)
def q (x : ℝ) := (-2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)

theorem find_q :
  (∀ x, q x + g x = f x) ↔ (∀ x, q x = -2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)
:= sorry

end find_q_l76_76184


namespace factor_expression_l76_76759

theorem factor_expression (x y z : ℝ) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 ≠ 0 →
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / ((x - y)^3 + (y - z)^3 + (z - x)^3) =
    (x + y) * (y + z) * (z + x) :=
by
  intro h
  sorry

end factor_expression_l76_76759


namespace shadow_area_correct_l76_76667

noncomputable def shadow_area (R : ℝ) : ℝ := 3 * Real.pi * R^2

theorem shadow_area_correct (R r d R' : ℝ)
  (h1 : r = (Real.sqrt 3) * R / 2)
  (h2 : d = (3 * R) / 2)
  (h3 : R' = ((3 * R * r) / d)) :
  shadow_area R = Real.pi * R' ^ 2 :=
by
  sorry

end shadow_area_correct_l76_76667


namespace temperature_problem_l76_76492

theorem temperature_problem
  (M L N : ℝ)
  (h1 : M = L + N)
  (h2 : M - 9 = M - 9)
  (h3 : L + 5 = L + 5)
  (h4 : abs (M - 9 - (L + 5)) = 1) :
  (N = 15 ∨ N = 13) → (N = 15 ∧ N = 13 → 15 * 13 = 195) :=
by
  sorry

end temperature_problem_l76_76492


namespace tuesday_more_than_monday_l76_76566

variable (M T W Th x : ℕ)

-- Conditions
def monday_dinners : M = 40 := by sorry
def tuesday_dinners : T = M + x := by sorry
def wednesday_dinners : W = T / 2 := by sorry
def thursday_dinners : Th = W + 3 := by sorry
def total_dinners : M + T + W + Th = 203 := by sorry

-- Proof problem: How many more dinners were sold on Tuesday than on Monday?
theorem tuesday_more_than_monday : x = 32 :=
by
  sorry

end tuesday_more_than_monday_l76_76566


namespace lisa_children_l76_76539

theorem lisa_children (C : ℕ) 
  (h1 : 5 * 52 = 260)
  (h2 : (2 * C + 3 + 2) * 260 = 3380) : 
  C = 4 := 
by
  sorry

end lisa_children_l76_76539


namespace count_multiples_4_6_10_less_300_l76_76734

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l76_76734


namespace sum_of_first_twelve_terms_l76_76624

section ArithmeticSequence

variables (a : ℕ → ℚ) (d : ℚ) (a₁ : ℚ)

-- General definition of the nth term in arithmetic progression
def arithmetic_term (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Given conditions in the problem
axiom fifth_term : arithmetic_term a₁ d 5 = 1
axiom seventeenth_term : arithmetic_term a₁ d 17 = 18

-- Define the sum of the first n terms in arithmetic sequence
def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- Statement of the proof problem
theorem sum_of_first_twelve_terms : 
  sum_arithmetic_sequence a₁ d 12 = 37.5 := 
sorry

end ArithmeticSequence

end sum_of_first_twelve_terms_l76_76624


namespace greatest_power_of_two_factor_l76_76292

theorem greatest_power_of_two_factor (n : ℕ) (h : n = 1000) :
  ∃ k, 2^k ∣ 10^n + 4^(n/2) ∧ k = 1003 :=
by {
  sorry
}

end greatest_power_of_two_factor_l76_76292


namespace jenna_filter_change_15th_is_March_l76_76531

def month_of_nth_change (startMonth interval n : ℕ) : ℕ :=
  ((interval * (n - 1)) % 12 + startMonth) % 12

theorem jenna_filter_change_15th_is_March :
  month_of_nth_change 1 7 15 = 3 := 
  sorry

end jenna_filter_change_15th_is_March_l76_76531


namespace same_type_l76_76948

variable (X Y : Prop) 

-- Definition of witnesses A and B based on their statements
def witness_A (A : Prop) := A ↔ (X → Y)
def witness_B (B : Prop) := B ↔ (¬X ∨ Y)

-- Proposition stating that A and B must be of the same type
theorem same_type (A B : Prop) (HA : witness_A X Y A) (HB : witness_B X Y B) : 
  (A = B) := 
sorry

end same_type_l76_76948


namespace randy_total_trees_l76_76902

def mango_trees : ℕ := 60
def coconut_trees : ℕ := mango_trees / 2 - 5
def total_trees (mangos coconuts : ℕ) : ℕ := mangos + coconuts

theorem randy_total_trees : total_trees mango_trees coconut_trees = 85 :=
by
  sorry

end randy_total_trees_l76_76902


namespace difference_english_math_l76_76571

/-- There are 30 students who pass in English and 20 students who pass in Math. -/
axiom passes_in_english : ℕ
axiom passes_in_math : ℕ
axiom both_subjects : ℕ
axiom only_english : ℕ
axiom only_math : ℕ

/-- Definitions based on the problem conditions -/
axiom number_passes_in_english : only_english + both_subjects = 30
axiom number_passes_in_math : only_math + both_subjects = 20

/-- The difference between the number of students who pass only in English
    and the number of students who pass only in Math is 10. -/
theorem difference_english_math : only_english - only_math = 10 :=
by
  sorry

end difference_english_math_l76_76571


namespace correct_operation_l76_76967

-- Define the conditions as hypotheses
variable (a : ℝ)

-- A: \(a^2 \cdot a = a^3\)
def condition_A : Prop := a^2 * a = a^3

-- B: \((a^3)^3 = a^6\)
def condition_B : Prop := (a^3)^3 = a^6

-- C: \(a^3 + a^3 = a^5\)
def condition_C : Prop := a^3 + a^3 = a^5

-- D: \(a^6 \div a^2 = a^3\)
def condition_D : Prop := a^6 / a^2 = a^3

-- Proof that only condition A is correct:
theorem correct_operation : condition_A a ∧ ¬condition_B a ∧ ¬condition_C a ∧ ¬condition_D a :=
by
  sorry  -- Actual proofs would go here

end correct_operation_l76_76967


namespace temperature_conversion_l76_76994

theorem temperature_conversion (C F F_new C_new : ℚ) 
  (h_formula : C = (5/9) * (F - 32))
  (h_C : C = 30)
  (h_F_new : F_new = F + 15)
  (h_F : F = 86)
: C_new = (5/9) * (F_new - 32) ↔ C_new = 38.33 := 
by 
  sorry

end temperature_conversion_l76_76994


namespace min_length_intersection_l76_76616

theorem min_length_intersection
  (m n : ℝ)
  (hM0 : 0 ≤ m)
  (hM1 : m + 3/4 ≤ 1)
  (hN0 : n - 1/3 ≥ 0)
  (hN1 : n ≤ 1) :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧
  x = ((m + 3/4) + (n - 1/3)) - 1 :=
sorry

end min_length_intersection_l76_76616


namespace inequality_proof_l76_76505

theorem inequality_proof (a b c d : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) 
  (h_sum : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_proof_l76_76505


namespace solve_equation1_solve_equation2_l76_76402

def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

theorem solve_equation1 :
  (equation1 (-1) ∨ equation1 (1 / 3)) ∧ 
  (∀ x, equation1 x → x = -1 ∨ x = 1 / 3) :=
sorry

theorem solve_equation2 :
  (equation2 1 ∨ equation2 (-4)) ∧ 
  (∀ x, equation2 x → x = 1 ∨ x = -4) :=
sorry

end solve_equation1_solve_equation2_l76_76402


namespace age_ratio_correct_l76_76219

noncomputable def RahulDeepakAgeRatio : Prop :=
  let R := 20
  let D := 8
  R / D = 5 / 2

theorem age_ratio_correct (R D : ℕ) (h1 : R + 6 = 26) (h2 : D = 8) : RahulDeepakAgeRatio :=
by
  -- Proof omitted
  sorry

end age_ratio_correct_l76_76219


namespace problem1_problem2_problem3_l76_76840

-- Definition of sets A, B, and U
def A : Set ℤ := {1, 2, 3, 4, 5}
def B : Set ℤ := {-1, 1, 2, 3}
def U : Set ℤ := {x | -1 ≤ x ∧ x < 6}

-- The complement of B in U
def C_U (B : Set ℤ) : Set ℤ := {x ∈ U | x ∉ B}

-- Problem statements
theorem problem1 : A ∩ B = {1, 2, 3} := by sorry
theorem problem2 : A ∪ B = {-1, 1, 2, 3, 4, 5} := by sorry
theorem problem3 : (C_U B) ∩ A = {4, 5} := by sorry

end problem1_problem2_problem3_l76_76840


namespace Lizzie_group_number_l76_76397

theorem Lizzie_group_number (x : ℕ) (h1 : x + (x + 17) = 91) : x + 17 = 54 :=
by
  sorry

end Lizzie_group_number_l76_76397


namespace find_a_plus_b_l76_76389

theorem find_a_plus_b (a b : ℝ) (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 0)
                      (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 16) : a + b = -16 :=
by sorry

end find_a_plus_b_l76_76389


namespace solution_xy_l76_76254

noncomputable def find_xy (x y : ℚ) : Prop :=
  (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1 / 3

theorem solution_xy :
  find_xy (10 + 1 / 3) (10 + 2 / 3) :=
by
  sorry

end solution_xy_l76_76254


namespace cube_volume_from_surface_area_l76_76429

theorem cube_volume_from_surface_area (A : ℝ) (h : A = 54) :
  ∃ V : ℝ, V = 27 := by
  sorry

end cube_volume_from_surface_area_l76_76429


namespace equation_has_at_least_two_distinct_roots_l76_76954

theorem equation_has_at_least_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a^2 * (x1 - 2) + a * (39 - 20 * x1) + 20 = 0 ∧ a^2 * (x2 - 2) + a * (39 - 20 * x2) + 20 = 0) ↔ a = 20 :=
by
  sorry

end equation_has_at_least_two_distinct_roots_l76_76954


namespace solve_fractional_equation_l76_76990

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -5) : 
    (2 * x / (x - 1)) - 1 = 4 / (1 - x) → x = -5 := 
by
  sorry

end solve_fractional_equation_l76_76990


namespace divisor_of_1053_added_with_5_is_2_l76_76979

theorem divisor_of_1053_added_with_5_is_2 :
  ∃ d : ℕ, d > 1 ∧ ∀ (x : ℝ), x = 5.000000000000043 → (1053 + x) % d = 0 → d = 2 :=
by
  sorry

end divisor_of_1053_added_with_5_is_2_l76_76979


namespace bananas_proof_l76_76366

noncomputable def number_of_bananas (total_oranges : ℕ) (total_fruits_percent_good : ℝ) 
  (percent_rotten_oranges : ℝ) (percent_rotten_bananas : ℝ) : ℕ := 448

theorem bananas_proof :
  let total_oranges := 600
  let percent_rotten_oranges := 0.15
  let percent_rotten_bananas := 0.08
  let total_fruits_percent_good := 0.878
  
  number_of_bananas total_oranges total_fruits_percent_good percent_rotten_oranges percent_rotten_bananas = 448 :=
by
  sorry

end bananas_proof_l76_76366


namespace carl_candy_bars_l76_76054

/-- 
Carl earns $0.75 every week for taking out his neighbor's trash. 
Carl buys a candy bar every time he earns $0.50. 
After four weeks, Carl will be able to buy 6 candy bars.
-/
theorem carl_candy_bars :
  (0.75 * 4) / 0.50 = 6 := 
  by
    sorry

end carl_candy_bars_l76_76054


namespace sum_of_numbers_l76_76950

theorem sum_of_numbers (x y : ℕ) (h1 : x * y = 9375) (h2 : y / x = 15) : x + y = 400 :=
by
  sorry

end sum_of_numbers_l76_76950


namespace find_p_q_sum_l76_76315

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l76_76315


namespace even_quadruple_composition_l76_76939

variable {α : Type*} [AddGroup α]

-- Definition of an odd function
def is_odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

theorem even_quadruple_composition {f : α → α} 
  (hf_odd : is_odd_function f) : 
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
by
  sorry

end even_quadruple_composition_l76_76939


namespace largest_adjacent_to_1_number_of_good_cells_l76_76830

def table_width := 51
def table_height := 3
def total_cells := 153

-- Conditions
def condition_1_present (n : ℕ) : Prop := n ∈ Finset.range (total_cells + 1)
def condition_2_bottom_left : Prop := (1 = 1)
def condition_3_adjacent (a b : ℕ) : Prop := 
  (a = b + 1) ∨ 
  (a + 1 = b) ∧ 
  (condition_1_present a) ∧ 
  (condition_1_present b)

-- Part (a): Largest number adjacent to cell containing 1 is 152.
theorem largest_adjacent_to_1 : ∃ b, b = 152 ∧ condition_3_adjacent 1 b :=
by sorry

-- Part (b): Number of good cells that can contain the number 153 is 76.
theorem number_of_good_cells : ∃ count, count = 76 ∧ 
  ∀ (i : ℕ) (j: ℕ), (i, j) ∈ (Finset.range table_height).product (Finset.range table_width) →
  condition_1_present 153 ∧
  (i = table_height - 1 ∨ j = 0 ∨ j = table_width - 1 ∨ j ∈ (Finset.range (table_width - 2)).erase 1) →
  (condition_3_adjacent (i*table_width + j) 153) :=
by sorry

end largest_adjacent_to_1_number_of_good_cells_l76_76830


namespace problem_statement_l76_76974

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

theorem problem_statement (a : ℝ) :
  f (Real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a :=
by
  sorry

end problem_statement_l76_76974


namespace solve_quadratic_l76_76985

theorem solve_quadratic (x : ℝ) (h : x^2 - 2 * x - 3 = 0) : x = 3 ∨ x = -1 := 
sorry

end solve_quadratic_l76_76985


namespace fathers_age_more_than_4_times_son_l76_76756

-- Let F (Father's age) be 44 and S (Son's age) be 10 as given by solving the equations
def X_years_more_than_4_times_son_age (F S X : ℕ) : Prop :=
  F = 4 * S + X ∧ F + 4 = 2 * (S + 4) + 20

theorem fathers_age_more_than_4_times_son (F S X : ℕ) (h1 : F = 44) (h2 : F = 4 * S + X) (h3 : F + 4 = 2 * (S + 4) + 20) :
  X = 4 :=
by
  -- The proof would go here
  sorry

end fathers_age_more_than_4_times_son_l76_76756


namespace some_number_is_l76_76352

theorem some_number_is (x some_number : ℤ) (h1 : x = 4) (h2 : 5 * x + 3 = 10 * x - some_number) : some_number = 17 := by
  sorry

end some_number_is_l76_76352


namespace total_number_of_books_ways_to_select_books_l76_76284

def first_layer_books : ℕ := 6
def second_layer_books : ℕ := 5
def third_layer_books : ℕ := 4

theorem total_number_of_books : first_layer_books + second_layer_books + third_layer_books = 15 := by
  sorry

theorem ways_to_select_books : first_layer_books * second_layer_books * third_layer_books = 120 := by
  sorry

end total_number_of_books_ways_to_select_books_l76_76284


namespace no_real_solution_implies_a_range_l76_76329

noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 4 * x + a^2

theorem no_real_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, quadratic a x ≤ 0 → false) ↔ a < -2 ∨ a > 2 := 
sorry

end no_real_solution_implies_a_range_l76_76329


namespace min_ratio_of_integers_l76_76632

theorem min_ratio_of_integers (x y : ℕ) (hx : 50 < x) (hy : 50 < y) (h_mean : x + y = 130) : 
  x = 51 → y = 79 → x / y = 51 / 79 := by
  sorry

end min_ratio_of_integers_l76_76632


namespace triangle_area_l76_76987

theorem triangle_area {x y : ℝ} :

  (∀ a:ℝ, y = a ↔ a = x) ∧
  (∀ b:ℝ, y = -b ↔ b = x) ∧
  ( y = 10 )
  → 1 / 2 * abs (10 - (-10)) * 10 = 100 :=
by
  sorry

end triangle_area_l76_76987


namespace greg_rolls_more_ones_than_fives_l76_76380

def probability_more_ones_than_fives (n : ℕ) : ℚ :=
  if n = 6 then 695 / 1944 else 0

theorem greg_rolls_more_ones_than_fives :
  probability_more_ones_than_fives 6 = 695 / 1944 :=
by sorry

end greg_rolls_more_ones_than_fives_l76_76380


namespace even_function_f_l76_76481

noncomputable def f (x : ℝ) : ℝ := if 0 < x ∧ x < 10 then Real.log x else 0

theorem even_function_f (x : ℝ) (h : f (-x) = f x) (h1 : ∀ x, 0 < x ∧ x < 10 → f x = Real.log x) :
  f (-Real.exp 1) + f (Real.exp 2) = 3 := by
  sorry

end even_function_f_l76_76481


namespace expand_and_count_nonzero_terms_l76_76901

theorem expand_and_count_nonzero_terms (x : ℝ) : 
  (x-3)*(3*x^2-2*x+6) + 2*(x^3 + x^2 - 4*x) = 5*x^3 - 9*x^2 + 4*x - 18 ∧ 
  (5 ≠ 0 ∧ -9 ≠ 0 ∧ 4 ≠ 0 ∧ -18 ≠ 0) :=
sorry

end expand_and_count_nonzero_terms_l76_76901


namespace Iain_pennies_problem_l76_76549

theorem Iain_pennies_problem :
  ∀ (P : ℝ), 200 - 30 = 170 →
             170 - (P / 100) * 170 = 136 →
             P = 20 :=
by
  intros P h1 h2
  sorry

end Iain_pennies_problem_l76_76549


namespace max_non_colored_cubes_l76_76715

open Nat

-- Define the conditions
def isRectangularPrism (length width height volume : ℕ) := length * width * height = volume

-- The theorem stating the equivalent math proof problem
theorem max_non_colored_cubes (length width height : ℕ) (h₁ : isRectangularPrism length width height 1024) :
(length > 2 ∧ width > 2 ∧ height > 2) → (length - 2) * (width - 2) * (height - 2) = 504 := by
  sorry

end max_non_colored_cubes_l76_76715


namespace pages_revised_once_l76_76637

-- Definitions
def total_pages : ℕ := 200
def pages_revised_twice : ℕ := 20
def total_cost : ℕ := 1360
def cost_first_time : ℕ := 5
def cost_revision : ℕ := 3

theorem pages_revised_once (x : ℕ) (h1 : total_cost = 1000 + 3 * x + 120) : x = 80 := by
  sorry

end pages_revised_once_l76_76637


namespace proof_problem_l76_76475

-- Definitions coming from the conditions
def num_large_divisions := 12
def num_small_divisions_per_large := 5
def seconds_per_small_division := 1
def seconds_per_large_division := num_small_divisions_per_large * seconds_per_small_division
def start_position := 5
def end_position := 9
def divisions_moved := end_position - start_position
def total_seconds_actual := divisions_moved * seconds_per_large_division
def total_seconds_claimed := 4

-- The theorem stating the false claim
theorem proof_problem : total_seconds_actual ≠ total_seconds_claimed :=
by {
  -- We skip the actual proof as instructed
  sorry
}

end proof_problem_l76_76475


namespace greatest_divisor_l76_76833

theorem greatest_divisor (n : ℕ) (h1 : 1428 % n = 9) (h2 : 2206 % n = 13) : n = 129 :=
sorry

end greatest_divisor_l76_76833


namespace evaluate_complex_power_expression_l76_76630

theorem evaluate_complex_power_expression : (i : ℂ)^23 + ((i : ℂ)^105 * (i : ℂ)^17) = -i - 1 := by
  sorry

end evaluate_complex_power_expression_l76_76630


namespace colorful_family_children_count_l76_76127

theorem colorful_family_children_count 
    (B W S x : ℕ)
    (h1 : B = W) (h2 : W = S)
    (h3 : (B - x) + W = 10)
    (h4 : W + (S + x) = 18) :
    B + W + S = 21 :=
by
  sorry

end colorful_family_children_count_l76_76127


namespace max_ratio_of_two_digit_numbers_with_mean_55_l76_76006

theorem max_ratio_of_two_digit_numbers_with_mean_55 (x y : ℕ) (h1 : 10 ≤ x) (h2 : x ≤ 99) (h3 : 10 ≤ y) (h4 : y ≤ 99) (h5 : (x + y) / 2 = 55) : x / y ≤ 9 :=
sorry

end max_ratio_of_two_digit_numbers_with_mean_55_l76_76006


namespace square_difference_example_l76_76153

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l76_76153


namespace tangent_line_at_0_l76_76613

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x + Real.sin x

theorem tangent_line_at_0 :
  ∃ (m b : ℝ), ∀ (x : ℝ), f 0 = 1 ∧ (f' : ℝ → ℝ) 0 = 1 ∧ (f' x = Real.exp x + 2 * x - 1 + Real.cos x) ∧ 
  (m = 1) ∧ (b = (m * 0 + 1)) ∧ (∀ x : ℝ, y = m * x + b) :=
by
  sorry

end tangent_line_at_0_l76_76613


namespace positive_three_digit_integers_divisible_by_12_and_7_l76_76768

theorem positive_three_digit_integers_divisible_by_12_and_7 : 
  ∃ n : ℕ, n = 11 ∧ ∀ k : ℕ, (k ∣ 12) ∧ (k ∣ 7) ∧ (100 ≤ k) ∧ (k < 1000) :=
by
  sorry

end positive_three_digit_integers_divisible_by_12_and_7_l76_76768


namespace fraction_expression_eq_l76_76056

theorem fraction_expression_eq (x y : ℕ) (hx : x = 4) (hy : y = 5) : 
  ((1 / y) + (1 / x)) / (1 / x) = 9 / 5 :=
by
  rw [hx, hy]
  sorry

end fraction_expression_eq_l76_76056


namespace correct_articles_l76_76639

-- Definitions based on conditions provided in the problem
def sentence := "Traveling in ____ outer space is quite ____ exciting experience."
def first_blank_article := "no article"
def second_blank_article := "an"

-- Statement of the proof problem
theorem correct_articles : 
  (first_blank_article = "no article" ∧ second_blank_article = "an") :=
by
  sorry

end correct_articles_l76_76639


namespace xiaohua_final_score_l76_76440

-- Definitions for conditions
def education_score : ℝ := 9
def experience_score : ℝ := 7
def work_attitude_score : ℝ := 8
def weight_education : ℝ := 1
def weight_experience : ℝ := 2
def weight_attitude : ℝ := 2

-- Computation of the final score
noncomputable def final_score : ℝ :=
  education_score * (weight_education / (weight_education + weight_experience + weight_attitude)) +
  experience_score * (weight_experience / (weight_education + weight_experience + weight_attitude)) +
  work_attitude_score * (weight_attitude / (weight_education + weight_experience + weight_attitude))

-- The statement we want to prove
theorem xiaohua_final_score :
  final_score = 7.8 :=
sorry

end xiaohua_final_score_l76_76440


namespace lawn_width_is_60_l76_76592

theorem lawn_width_is_60
  (length : ℕ)
  (width : ℕ)
  (road_width : ℕ)
  (cost_per_sq_meter : ℕ)
  (total_cost : ℕ)
  (area_of_lawn : ℕ)
  (total_area_of_roads : ℕ)
  (intersection_area : ℕ)
  (area_cost_relation : total_area_of_roads * cost_per_sq_meter = total_cost)
  (intersection_included : (road_width * length + road_width * width - intersection_area) = total_area_of_roads)
  (length_eq : length = 80)
  (road_width_eq : road_width = 10)
  (cost_eq : cost_per_sq_meter = 2)
  (total_cost_eq : total_cost = 2600)
  (intersection_area_eq : intersection_area = road_width * road_width)
  : width = 60 :=
by
  sorry

end lawn_width_is_60_l76_76592


namespace sum_of_undefined_values_l76_76215

theorem sum_of_undefined_values (y : ℝ) :
  (y^2 - 7 * y + 12 = 0) → y = 3 ∨ y = 4 → (3 + 4 = 7) :=
by
  intro hy
  intro hy'
  sorry

end sum_of_undefined_values_l76_76215


namespace smallest_sum_of_squares_l76_76887

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := 
  sorry

end smallest_sum_of_squares_l76_76887


namespace circle_area_l76_76640

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : π * r^2 = 3 / 2 :=
by
  -- We leave this place for computations and derivations.
  sorry

end circle_area_l76_76640


namespace find_shorter_parallel_side_l76_76737

variable (x : ℝ) (a : ℝ) (b : ℝ) (h : ℝ)

def is_trapezium_area (a b h : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * (a + b) * h

theorem find_shorter_parallel_side
  (h28 : a = 28)
  (h15 : h = 15)
  (hArea : area = 345)
  (hIsTrapezium : is_trapezium_area a b h area):
  b = 18 := 
sorry

end find_shorter_parallel_side_l76_76737


namespace tens_digit_of_7_pow_35_l76_76066

theorem tens_digit_of_7_pow_35 : 
  (7 ^ 35) % 100 / 10 % 10 = 4 :=
by
  sorry

end tens_digit_of_7_pow_35_l76_76066


namespace find_d_l76_76956

theorem find_d (a b c d : ℤ) (h_poly : ∃ s1 s2 s3 s4 : ℤ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧ 
  ( ∀ x, (Polynomial.eval x (Polynomial.C d + Polynomial.X * Polynomial.C c + Polynomial.X^2 * Polynomial.C b + Polynomial.X^3 * Polynomial.C a + Polynomial.X^4)) =
    (x + s1) * (x + s2) * (x + s3) * (x + s4) ) ) 
  (h_sum : a + b + c + d = 2013) : d = 0 :=
by
  sorry

end find_d_l76_76956


namespace part1_part2_part3_l76_76620

def climbing_function_1_example (x : ℝ) : Prop :=
  ∃ a : ℝ, a^2 = -8 / a

theorem part1 (x : ℝ) : climbing_function_1_example x ↔ (x = -2) := sorry

def climbing_function_2_example (m : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = m*a + m) ∧ ∀ d: ℝ, ((d^2 = m*d + m) → d = a)

theorem part2 (m : ℝ) : (m = -4) ∧ climbing_function_2_example m := sorry

def climbing_function_3_example (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : Prop :=
  ∃ a1 a2 : ℝ, ((a1 + a2 = n/(1-m)) ∧ (a1*a2 = 1/(m-1)) ∧ (|a1 - a2| = p)) ∧ 
  (∀ x : ℝ, (m * x^2 + n * x + 1) ≥ q) 

theorem part3 (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : climbing_function_3_example m n p q h1 h2 ↔ (0 < q) ∧ (q ≤ 4/11) := sorry

end part1_part2_part3_l76_76620


namespace range_of_m_l76_76603

theorem range_of_m (f g : ℝ → ℝ) (h1 : ∃ m : ℝ, ∀ x : ℝ, f x = m * (x - m) * (x + m + 3))
  (h2 : ∀ x : ℝ, g x = 2 ^ x - 4)
  (h3 : ∀ x : ℝ, f x < 0 ∨ g x < 0) :
  ∃ m : ℝ, -5 < m ∧ m < 0 :=
sorry

end range_of_m_l76_76603


namespace inverse_matrix_l76_76400

theorem inverse_matrix
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (H : A * B = ![![1, 2], ![0, 6]]) :
  A⁻¹ = ![![-1, 0], ![0, 2]] :=
sorry

end inverse_matrix_l76_76400


namespace annies_initial_amount_l76_76745

theorem annies_initial_amount :
  let hamburger_cost := 4
  let cheeseburger_cost := 5
  let french_fries_cost := 3
  let milkshake_cost := 5
  let smoothie_cost := 6
  let people_count := 8
  let burger_discount := 1
  let milkshake_discount := 2
  let smoothie_discount_buy2_get1free := 6
  let sales_tax := 0.08
  let tip_rate := 0.15
  let max_single_person_cost := cheeseburger_cost + french_fries_cost + smoothie_cost
  let total_cost := people_count * max_single_person_cost
  let total_burger_discount := people_count * burger_discount
  let total_milkshake_discount := 4 * milkshake_discount
  let total_smoothie_discount := smoothie_discount_buy2_get1free
  let total_discount := total_burger_discount + total_milkshake_discount + total_smoothie_discount
  let discounted_cost := total_cost - total_discount
  let tax_amount := discounted_cost * sales_tax
  let subtotal_with_tax := discounted_cost + tax_amount
  let original_total_cost := people_count * max_single_person_cost
  let tip_amount := original_total_cost * tip_rate
  let final_amount := subtotal_with_tax + tip_amount
  let annie_has_left := 30
  let annies_initial_money := final_amount + annie_has_left
  annies_initial_money = 144 :=
by
  sorry

end annies_initial_amount_l76_76745


namespace laptop_price_l76_76261

theorem laptop_price (cost upfront : ℝ) (upfront_percentage : ℝ) (upfront_eq : upfront = 240) (upfront_percentage_eq : upfront_percentage = 20) : 
  cost = 1200 :=
by
  sorry

end laptop_price_l76_76261


namespace graph_transformation_point_l76_76964

theorem graph_transformation_point {f : ℝ → ℝ} (h : f 1 = 0) : f (0 + 1) + 1 = 1 :=
by
  sorry

end graph_transformation_point_l76_76964


namespace largest_among_four_numbers_l76_76733

theorem largest_among_four_numbers
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : a + b = 1) :
  b > max (max (1/2) (2 * a * b)) (a^2 + b^2) := 
sorry

end largest_among_four_numbers_l76_76733


namespace power_of_power_eq_512_l76_76323

theorem power_of_power_eq_512 : (2^3)^3 = 512 := by
  sorry

end power_of_power_eq_512_l76_76323


namespace find_p_and_q_l76_76977

theorem find_p_and_q :
  (∀ p q: ℝ, (∃ x : ℝ, x^2 + p * x + q = 0 ∧ q * x^2 + p * x + 1 = 0) ∧ (-2) ^ 2 + p * (-2) + q = 0 ∧ p ≠ 0 ∧ q ≠ 0 → 
    (p, q) = (1, -2) ∨ (p, q) = (3, 2) ∨ (p, q) = (5/2, 1)) :=
sorry

end find_p_and_q_l76_76977


namespace count_invitations_l76_76383

theorem count_invitations (teachers : Finset ℕ) (A B : ℕ) (hA : A ∈ teachers) (hB : B ∈ teachers) (h_size : teachers.card = 10):
  ∃ (ways : ℕ), ways = 140 ∧ ∀ (S : Finset ℕ), S.card = 6 → ((A ∈ S ∧ B ∉ S) ∨ (A ∉ S ∧ B ∈ S) ∨ (A ∉ S ∧ B ∉ S)) ↔ ways = 140 := 
sorry

end count_invitations_l76_76383


namespace solve_equation_1_solve_equation_2_l76_76731

theorem solve_equation_1 (x : ℝ) : x^2 - 7 * x = 0 ↔ (x = 0 ∨ x = 7) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 6 * x + 1 = 0 ↔ (x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) :=
by sorry

end solve_equation_1_solve_equation_2_l76_76731


namespace water_added_l76_76242

theorem water_added (x : ℝ) (salt_initial_percentage : ℝ) (salt_final_percentage : ℝ) 
   (evap_fraction : ℝ) (salt_added : ℝ) (W : ℝ) 
   (hx : x = 150) (h_initial_salt : salt_initial_percentage = 0.2) 
   (h_final_salt : salt_final_percentage = 1 / 3) 
   (h_evap_fraction : evap_fraction = 1 / 4) 
   (h_salt_added : salt_added = 20) : 
  W = 37.5 :=
by
  sorry

end water_added_l76_76242


namespace jimmy_fill_bucket_time_l76_76202

-- Definitions based on conditions
def pool_volume : ℕ := 84
def bucket_volume : ℕ := 2
def total_time_minutes : ℕ := 14
def total_time_seconds : ℕ := total_time_minutes * 60
def trips : ℕ := pool_volume / bucket_volume

-- Theorem statement
theorem jimmy_fill_bucket_time : (total_time_seconds / trips) = 20 := by
  sorry

end jimmy_fill_bucket_time_l76_76202


namespace zog_words_count_l76_76148

-- Defining the number of letters in the Zoggian alphabet
def num_letters : ℕ := 6

-- Function to calculate the number of words with n letters
def words_with_n_letters (n : ℕ) : ℕ := num_letters ^ n

-- Definition to calculate the total number of words with at most 4 letters
def total_words : ℕ :=
  (words_with_n_letters 1) +
  (words_with_n_letters 2) +
  (words_with_n_letters 3) +
  (words_with_n_letters 4)

-- Theorem statement
theorem zog_words_count : total_words = 1554 := by
  sorry

end zog_words_count_l76_76148


namespace dogs_for_sale_l76_76470

variable (D : ℕ)
def number_of_cats := D / 2
def number_of_birds := 2 * D
def number_of_fish := 3 * D
def total_animals := D + number_of_cats D + number_of_birds D + number_of_fish D

theorem dogs_for_sale (h : total_animals D = 39) : D = 6 :=
by
  sorry

end dogs_for_sale_l76_76470


namespace unique_solution_range_l76_76626
-- import relevant libraries

-- define the functions
def f (a x : ℝ) : ℝ := 2 * a * x ^ 3 + 3
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 2

-- state and prove the main theorem (statement only)
theorem unique_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = g x ∧ ∀ y : ℝ, y > 0 → f a y = g y → y = x) ↔ a ∈ Set.Iio (-1) :=
sorry

end unique_solution_range_l76_76626


namespace find_percentage_l76_76870

theorem find_percentage (P : ℕ) (h1 : P * 64 = 320 * 10) : P = 5 := 
  by
  sorry

end find_percentage_l76_76870


namespace y_at_40_l76_76988

def y_at_x (x : ℤ) : ℤ :=
  3 * x + 4

theorem y_at_40 : y_at_x 40 = 124 :=
by {
  sorry
}

end y_at_40_l76_76988


namespace calculate_expression_l76_76253

theorem calculate_expression : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculate_expression_l76_76253


namespace sachin_age_l76_76825

theorem sachin_age {Sachin_age Rahul_age : ℕ} (h1 : Sachin_age + 14 = Rahul_age) (h2 : Sachin_age * 9 = Rahul_age * 7) : Sachin_age = 49 := by
sorry

end sachin_age_l76_76825


namespace average_reading_time_l76_76855

theorem average_reading_time (t_Emery t_Serena : ℕ) (h1 : t_Emery = 20) (h2 : t_Serena = 5 * t_Emery) : 
  (t_Emery + t_Serena) / 2 = 60 := 
by
  sorry

end average_reading_time_l76_76855


namespace portrait_in_silver_box_l76_76003

theorem portrait_in_silver_box
  (gold_box : Prop)
  (silver_box : Prop)
  (lead_box : Prop)
  (p : Prop) (q : Prop) (r : Prop)
  (h1 : p ↔ gold_box)
  (h2 : q ↔ ¬silver_box)
  (h3 : r ↔ ¬gold_box)
  (h4 : (p ∨ q ∨ r) ∧ ¬(p ∧ q) ∧ ¬(q ∧ r) ∧ ¬(r ∧ p)) :
  silver_box :=
sorry

end portrait_in_silver_box_l76_76003


namespace plane_through_A_perpendicular_to_BC_l76_76421

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between_points (P Q : Point3D) : Point3D :=
  { x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z }

def plane_eq (n : Point3D) (P : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - P.x) + n.y * (y - P.y) + n.z * (z - P.z)

def A := Point3D.mk 0 (-2) 8
def B := Point3D.mk 4 3 2
def C := Point3D.mk 1 4 3

def n := vector_between_points B C
def plane := plane_eq n A

theorem plane_through_A_perpendicular_to_BC :
  ∀ x y z : ℝ, plane x y z = 0 ↔ -3 * x + y + z - 6 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l76_76421


namespace range_of_a_l76_76173

variable {a : ℝ}

-- Proposition p: The solution set of the inequality x^2 - (a+1)x + 1 ≤ 0 is empty
def prop_p (a : ℝ) : Prop := (a + 1) ^ 2 - 4 < 0 

-- Proposition q: The function f(x) = (a+1)^x is increasing within its domain
def prop_q (a : ℝ) : Prop := a > 0 

-- The combined conditions
def combined_conditions (a : ℝ) : Prop := (prop_p a) ∨ (prop_q a) ∧ ¬(prop_p a ∧ prop_q a)

-- The range of values for a
theorem range_of_a (h : combined_conditions a) : -3 < a ∧ a ≤ 0 ∨ a ≥ 1 :=
  sorry

end range_of_a_l76_76173


namespace simplify_fraction_l76_76749

noncomputable def sin_15 := Real.sin (15 * Real.pi / 180)
noncomputable def cos_15 := Real.cos (15 * Real.pi / 180)
noncomputable def angle_15 := 15 * Real.pi / 180

theorem simplify_fraction : (1 / sin_15 - 1 / cos_15 = 2 * Real.sqrt 2) :=
by
  sorry

end simplify_fraction_l76_76749


namespace range_of_m_l76_76921

noncomputable def f (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x

def tangents_condition (m : ℝ) : Prop := ∃ x : ℝ, (-3 * x^2 + 12 * x - 9) * (x + 1) + m = -x^3 + 6 * x^2 - 9 * x

theorem range_of_m (m : ℝ) : tangents_condition m → -11 < m ∧ m < 16 :=
sorry

end range_of_m_l76_76921


namespace systematic_sampling_correct_l76_76919

-- Define the conditions for the problem
def num_employees : ℕ := 840
def num_selected : ℕ := 42
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Define systematic sampling interval
def sampling_interval := num_employees / num_selected

-- Define the length of the given interval
def interval_length := interval_end - interval_start + 1

-- The theorem to prove
theorem systematic_sampling_correct :
  (interval_length / sampling_interval) = 12 := sorry

end systematic_sampling_correct_l76_76919


namespace find_smaller_number_l76_76174

variable (x y : ℕ)

theorem find_smaller_number (h1 : ∃ k : ℕ, x = 2 * k ∧ y = 5 * k) (h2 : x + y = 21) : x = 6 :=
by
  sorry

end find_smaller_number_l76_76174


namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l76_76553

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l76_76553


namespace additive_inverse_commutativity_l76_76641

section
  variable {R : Type} [Ring R] (h : ∀ x : R, x ^ 2 = x)

  theorem additive_inverse (x : R) : -x = x := by
    sorry

  theorem commutativity (x y : R) : x * y = y * x := by
    sorry
end

end additive_inverse_commutativity_l76_76641


namespace contrapositive_statement_l76_76863

theorem contrapositive_statement {a b : ℤ} :
  (∀ a b : ℤ, (a % 2 = 1 ∧ b % 2 = 1) → (a + b) % 2 = 0) →
  (∀ a b : ℤ, ¬((a + b) % 2 = 0) → ¬(a % 2 = 1 ∧ b % 2 = 1)) :=
by 
  intros h a b
  sorry

end contrapositive_statement_l76_76863


namespace optimal_play_probability_Reimu_l76_76012

noncomputable def probability_Reimu_wins : ℚ :=
  5 / 16

theorem optimal_play_probability_Reimu :
  probability_Reimu_wins = 5 / 16 := 
by
  sorry

end optimal_play_probability_Reimu_l76_76012


namespace division_problem_solution_l76_76331

theorem division_problem_solution (x : ℝ) (h : (2.25 / x) * 12 = 9) : x = 3 :=
sorry

end division_problem_solution_l76_76331


namespace quadratic_minimum_value_proof_l76_76897

-- Define the quadratic function and its properties
def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the condition that the coefficient of the squared term is positive
def coefficient_positive : Prop := (2 : ℝ) > 0

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : Prop := h = 3

-- Define the minimum value of the quadratic function
def minimum_value (y_min : ℝ) : Prop := ∀ x : ℝ, y_min ≤ quadratic_function x 

-- Define the correct answer choice
def correct_answer : Prop := minimum_value 2

-- The theorem stating the proof problem
theorem quadratic_minimum_value_proof :
  coefficient_positive ∧ axis_of_symmetry 3 → correct_answer :=
sorry

end quadratic_minimum_value_proof_l76_76897


namespace integer_solutions_conditions_even_l76_76778

theorem integer_solutions_conditions_even (n : ℕ) (x : ℕ → ℤ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 
    x i ^ 2 + x ((i % n) + 1) ^ 2 + 50 = 16 * x i + 12 * x ((i % n) + 1) ) → 
  n % 2 = 0 :=
by 
sorry

end integer_solutions_conditions_even_l76_76778


namespace clubs_students_equal_l76_76587

theorem clubs_students_equal
  (C E : ℕ)
  (h1 : ∃ N, N = 3 * C)
  (h2 : ∃ N, N = 3 * E) :
  C = E :=
by
  sorry

end clubs_students_equal_l76_76587


namespace additional_time_needed_l76_76751

theorem additional_time_needed (total_parts apprentice_first_phase remaining_parts apprentice_rate master_rate combined_rate : ℕ)
  (h1 : total_parts = 500)
  (h2 : apprentice_first_phase = 45)
  (h3 : remaining_parts = total_parts - apprentice_first_phase)
  (h4 : apprentice_rate = 15)
  (h5 : master_rate = 20)
  (h6 : combined_rate = apprentice_rate + master_rate) :
  remaining_parts / combined_rate = 13 := 
by {
  sorry
}

end additional_time_needed_l76_76751


namespace frank_reads_pages_per_day_l76_76528

-- Define the conditions and problem statement
def total_pages : ℕ := 450
def total_chapters : ℕ := 41
def total_days : ℕ := 30

-- The derived value we need to prove
def pages_per_day : ℕ := total_pages / total_days

-- The theorem to prove
theorem frank_reads_pages_per_day : pages_per_day = 15 :=
  by
  -- Proof goes here
  sorry

end frank_reads_pages_per_day_l76_76528


namespace clock_hands_straight_twenty_four_hours_l76_76116

noncomputable def hands_straight_per_day : ℕ :=
  2 * 22

theorem clock_hands_straight_twenty_four_hours :
  hands_straight_per_day = 44 :=
by
  sorry

end clock_hands_straight_twenty_four_hours_l76_76116


namespace equation_1_solution_equation_2_solution_l76_76965

theorem equation_1_solution (x : ℝ) : (x-1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4 := 
by 
  sorry

theorem equation_2_solution (x : ℝ) : 3 * x * (x - 2) = x -2 ↔ x = 2 ∨ x = 1/3 := 
by 
  sorry

end equation_1_solution_equation_2_solution_l76_76965


namespace more_calories_per_dollar_l76_76403

-- The conditions given in the problem as definitions
def price_burritos : ℕ := 6
def price_burgers : ℕ := 8
def calories_per_burrito : ℕ := 120
def calories_per_burger : ℕ := 400
def num_burritos : ℕ := 10
def num_burgers : ℕ := 5

-- The theorem stating the mathematically equivalent proof problem
theorem more_calories_per_dollar : 
  (num_burgers * calories_per_burger / price_burgers) - (num_burritos * calories_per_burrito / price_burritos) = 50 :=
by
  sorry

end more_calories_per_dollar_l76_76403


namespace ab_cd_eq_neg190_over_9_l76_76766

theorem ab_cd_eq_neg190_over_9 (a b c d : ℝ)
  (h1 : a + b + c = 3)
  (h2 : a + b + d = -2)
  (h3 : a + c + d = 8)
  (h4 : b + c + d = -1) :
  a * b + c * d = -190 / 9 :=
by
  sorry

end ab_cd_eq_neg190_over_9_l76_76766


namespace triangle_angle_proof_l76_76420

theorem triangle_angle_proof (α β γ : ℝ) (hα : α > 60) (hβ : β > 60) (hγ : γ > 60) (h_sum : α + β + γ = 180) : false :=
by
  sorry

end triangle_angle_proof_l76_76420


namespace simplify_fraction_l76_76458

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end simplify_fraction_l76_76458


namespace relationship_between_y_l76_76968

theorem relationship_between_y (y1 y2 y3 : ℝ)
  (hA : y1 = 3 / -5)
  (hB : y2 = 3 / -3)
  (hC : y3 = 3 / 2) : y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_between_y_l76_76968


namespace find_p_l76_76521

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2 * p * x
def quadrilateral_is_rectangle (A B C D : ℝ × ℝ) : Prop := 
  A.1 = C.1 ∧ B.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2

theorem find_p (A B C D : ℝ × ℝ) (p : ℝ) (h1 : ∃ x y, circle_eq x y ∧ parabola_eq p x y) 
  (h2 : ∃ x y, circle_eq x y ∧ x = 0) 
  (h3 : quadrilateral_is_rectangle A B C D) 
  (h4 : 0 < p) : 
  p = 2 := 
sorry

end find_p_l76_76521


namespace math_problem_l76_76774

theorem math_problem (a b c : ℚ) 
  (h1 : a * (-2) = 1)
  (h2 : |b + 2| = 5)
  (h3 : c = 5 - 6) :
  4 * a - b + 3 * c = -8 ∨ 4 * a - b + 3 * c = 2 :=
by
  sorry

end math_problem_l76_76774


namespace algebraic_expression_l76_76162

variable (m n x y : ℤ)

theorem algebraic_expression (h1 : x = m) (h2 : y = n) (h3 : x - y = 2) : n - m = -2 := 
by
  sorry

end algebraic_expression_l76_76162


namespace weight_of_11_25m_rod_l76_76714

noncomputable def weight_per_meter (total_weight : ℝ) (length : ℝ) : ℝ :=
  total_weight / length

def weight_of_rod (weight_per_length : ℝ) (length : ℝ) : ℝ :=
  weight_per_length * length

theorem weight_of_11_25m_rod :
  let total_weight_8m := 30.4
  let length_8m := 8.0
  let length_11_25m := 11.25
  let weight_per_length := weight_per_meter total_weight_8m length_8m
  weight_of_rod weight_per_length length_11_25m = 42.75 :=
by sorry

end weight_of_11_25m_rod_l76_76714


namespace find_a_l76_76762

-- Given conditions and definitions
def circle_eq (x y : ℝ) : Prop := (x^2 + y^2 - 2*x - 2*y + 1 = 0)
def line_eq (x y a : ℝ) : Prop := (x - 2*y + a = 0)
def chord_length (r : ℝ) : ℝ := 2 * r

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, circle_eq x y) → 
  (∀ x y : ℝ, line_eq x y a) → 
  (∃ x y : ℝ, (x = 1 ∧ y = 1) ∧ (line_eq x y a ∧ chord_length 1 = 2)) → 
  a = 1 := by sorry

end find_a_l76_76762


namespace intersection_points_lie_on_ellipse_l76_76846

theorem intersection_points_lie_on_ellipse (s : ℝ) : 
  ∃ (x y : ℝ), (2 * s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) ∧ (x^2 / 16 + y^2 / 9 = 1) :=
sorry

end intersection_points_lie_on_ellipse_l76_76846


namespace plum_cost_l76_76108

theorem plum_cost
  (total_fruits : ℕ)
  (total_cost : ℕ)
  (peach_cost : ℕ)
  (plums_bought : ℕ)
  (peaches_bought : ℕ)
  (P : ℕ) :
  total_fruits = 32 →
  total_cost = 52 →
  peach_cost = 1 →
  plums_bought = 20 →
  peaches_bought = total_fruits - plums_bought →
  total_cost = 20 * P + peaches_bought * peach_cost →
  P = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end plum_cost_l76_76108


namespace math_and_english_scores_sum_l76_76803

theorem math_and_english_scores_sum (M E : ℕ) (total_score : ℕ) :
  (∀ (H : ℕ), H = (50 + M + E) / 3 → 
   50 + M + E + H = total_score) → 
   total_score = 248 → 
   M + E = 136 :=
by
  intros h1 h2;
  sorry

end math_and_english_scores_sum_l76_76803


namespace determine_base_l76_76267

theorem determine_base (x : ℕ) (h : 2 * x^3 + x + 6 = x^3 + 2 * x + 342) : x = 7 := 
sorry

end determine_base_l76_76267


namespace simplify_expression_l76_76250

theorem simplify_expression :
  5 * (18 / 7) * (21 / -45) = -6 / 5 := 
sorry

end simplify_expression_l76_76250


namespace trajectory_of_point_l76_76885

/-- 
  Given points A and B on the coordinate plane, with |AB|=2, 
  and a moving point P such that the sum of the distances from P
  to points A and B is constantly 2, the trajectory of point P 
  is the line segment AB. 
-/
theorem trajectory_of_point (A B P : ℝ × ℝ) 
  (h_AB : dist A B = 2) 
  (h_sum : dist P A + dist P B = 2) :
  P ∈ segment ℝ A B :=
sorry

end trajectory_of_point_l76_76885


namespace find_a7_l76_76434

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (n : ℕ)

-- Condition 1: The sequence {a_n} is geometric with all positive terms.
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- Condition 2: a₄ * a₁₀ = 16
axiom geo_seq_condition : is_geometric_sequence a r ∧ a 4 * a 10 = 16

-- The goal to prove
theorem find_a7 : (is_geometric_sequence a r ∧ a 4 * a 10 = 16) → a 7 = 4 :=
by {
  sorry
}

end find_a7_l76_76434


namespace man_age_twice_son_age_l76_76844

theorem man_age_twice_son_age (S M X : ℕ) (h1 : S = 28) (h2 : M = S + 30) (h3 : M + X = 2 * (S + X)) : X = 2 :=
by
  sorry

end man_age_twice_son_age_l76_76844


namespace range_of_x_l76_76909

theorem range_of_x (x : ℝ) : (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_x_l76_76909


namespace approximate_number_of_fish_l76_76681

/-
  In a pond, 50 fish were tagged and returned. 
  Later, in another catch of 50 fish, 2 were tagged. 
  Assuming the proportion of tagged fish in the second catch approximates that of the pond,
  prove that the total number of fish in the pond is approximately 1250.
-/

theorem approximate_number_of_fish (N : ℕ) 
  (tagged_in_pond : ℕ := 50) 
  (total_in_second_catch : ℕ := 50) 
  (tagged_in_second_catch : ℕ := 2) 
  (proportion_approx : tagged_in_second_catch / total_in_second_catch = tagged_in_pond / N) :
  N = 1250 :=
by
  sorry

end approximate_number_of_fish_l76_76681


namespace pow_fraction_eq_l76_76034

theorem pow_fraction_eq : (4:ℕ) = 2^2 ∧ (8:ℕ) = 2^3 → (4^800 / 8^400 = 2^400) :=
by
  -- proof steps should go here, but they are omitted as per the instruction
  sorry

end pow_fraction_eq_l76_76034


namespace price_per_litre_mixed_oil_l76_76775

-- Define the given conditions
def cost_oil1 : ℝ := 100 * 45
def cost_oil2 : ℝ := 30 * 57.50
def cost_oil3 : ℝ := 20 * 72
def total_cost : ℝ := cost_oil1 + cost_oil2 + cost_oil3
def total_volume : ℝ := 100 + 30 + 20

-- Define the statement to be proved
theorem price_per_litre_mixed_oil : (total_cost / total_volume) = 51.10 :=
by
  sorry

end price_per_litre_mixed_oil_l76_76775


namespace geometric_sequence_problem_l76_76133

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2)
  (h2 : a 1 + a 3 + a 5 = 14) (h_seq : geometric_sequence a) :
  (1 / a 1) + (1 / a 3) + (1 / a 5) = 7 / 8 := sorry

end geometric_sequence_problem_l76_76133


namespace exists_cube_number_divisible_by_six_in_range_l76_76378

theorem exists_cube_number_divisible_by_six_in_range :
  ∃ (y : ℕ), y > 50 ∧ y < 350 ∧ (∃ (n : ℕ), y = n^3) ∧ y % 6 = 0 :=
by 
  use 216
  sorry

end exists_cube_number_divisible_by_six_in_range_l76_76378


namespace smallest_c_minus_a_l76_76062

theorem smallest_c_minus_a (a b c : ℕ) (h1 : a * b * c = 720) (h2 : a < b) (h3 : b < c) : c - a ≥ 24 :=
sorry

end smallest_c_minus_a_l76_76062


namespace sqrt_of_26244_div_by_100_l76_76940

theorem sqrt_of_26244_div_by_100 (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 :=
sorry

end sqrt_of_26244_div_by_100_l76_76940


namespace M_inter_N_eq_l76_76448

def set_M (x : ℝ) : Prop := x^2 - 3 * x < 0
def set_N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4

def M := { x : ℝ | set_M x }
def N := { x : ℝ | set_N x }

theorem M_inter_N_eq : M ∩ N = { x | 1 ≤ x ∧ x < 3 } :=
by sorry

end M_inter_N_eq_l76_76448


namespace wuyang_volleyball_team_members_l76_76606

theorem wuyang_volleyball_team_members :
  (Finset.filter Nat.Prime (Finset.range 50)).card = 15 :=
by
  sorry

end wuyang_volleyball_team_members_l76_76606


namespace no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l76_76117

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n - 2) ^ 2 + (n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2

def is_divisible_by (a b : ℤ) : Prop := b ≠ 0 ∧ a % b = 0

theorem no_member_of_T_divisible_by_9_but_some_member_divisible_by_4 :
  ¬ (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 9) ∧
  (∃ n : ℤ, is_divisible_by (sum_of_squares_of_four_consecutive_integers n) 4) :=
by 
  sorry

end no_member_of_T_divisible_by_9_but_some_member_divisible_by_4_l76_76117


namespace union_of_A_and_B_l76_76866

def setA : Set ℝ := {x | (x + 1) * (x - 2) < 0}
def setB : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B : setA ∪ setB = {x | -1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end union_of_A_and_B_l76_76866


namespace inequality_am_gm_l76_76137

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3 * a * b * c :=
sorry

end inequality_am_gm_l76_76137


namespace ellipse_eq_l76_76346

theorem ellipse_eq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a^2 - b^2 = 4)
  (h4 : ∃ (line_eq : ℝ → ℝ), ∀ (x : ℝ), line_eq x = 3 * x + 7)
  (h5 : ∃ (mid_y : ℝ), mid_y = 1 ∧ ∃ (x1 y1 x2 y2 : ℝ), 
    ((y1 = 3 * x1 + 7) ∧ (y2 = 3 * x2 + 7)) ∧ 
    (y1 + y2) / 2 = mid_y): 
  (∀ x y : ℝ, (y^2 / (a^2 - 4) + x^2 / b^2 = 1) ↔ 
  (x^2 / 8 + y^2 / 12 = 1)) :=
by { sorry }

end ellipse_eq_l76_76346


namespace highest_value_of_a_l76_76454

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def highest_a : Nat :=
  7

theorem highest_value_of_a (a : Nat) 
  (last_three_digits := a * 100 + 53)
  (number := 4 * 10^8 + 3 * 10^7 + 7 * 10^6 + 5 * 10^5 + 2 * 10^4 + a * 10^3 + 5 * 10^2 + 3 * 10^1 + 9) :
  (∃ a, last_three_digits % 8 = 0 ∧ sum_of_digits number % 9 = 0 ∧ number % 12 = 0 ∧ a <= 9) → a = highest_a :=
by
  intros
  sorry

end highest_value_of_a_l76_76454


namespace divisor_is_twelve_l76_76984

theorem divisor_is_twelve (d : ℕ) (h : 64 = 5 * d + 4) : d = 12 := 
sorry

end divisor_is_twelve_l76_76984


namespace james_fraction_of_pizza_slices_l76_76443

theorem james_fraction_of_pizza_slices :
  (2 * 6 = 12) ∧ (8 / 12 = 2 / 3) :=
by
  sorry

end james_fraction_of_pizza_slices_l76_76443


namespace B_pow_2048_l76_76663

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2048 :
  B ^ 2048 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end B_pow_2048_l76_76663


namespace ewan_sequence_has_113_l76_76014

def sequence_term (n : ℕ) : ℤ := 11 * n - 8

theorem ewan_sequence_has_113 : ∃ n : ℕ, sequence_term n = 113 := by
  sorry

end ewan_sequence_has_113_l76_76014


namespace binomial_divisible_by_prime_l76_76172

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end binomial_divisible_by_prime_l76_76172


namespace sum_of_fractions_eq_five_fourteen_l76_76286

theorem sum_of_fractions_eq_five_fourteen :
  (1 : ℚ) / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) = 5 / 14 := 
by
  sorry

end sum_of_fractions_eq_five_fourteen_l76_76286


namespace gcd_of_three_numbers_l76_76262

theorem gcd_of_three_numbers (a b c : ℕ) (h1 : a = 15378) (h2 : b = 21333) (h3 : c = 48906) :
  Nat.gcd (Nat.gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  sorry

end gcd_of_three_numbers_l76_76262


namespace square_area_from_diagonal_l76_76769

theorem square_area_from_diagonal (d : ℝ) (hd : d = 3.8) : 
  ∃ (A : ℝ), A = 7.22 ∧ (∀ s : ℝ, d^2 = 2 * (s^2) → A = s^2) :=
by
  sorry

end square_area_from_diagonal_l76_76769


namespace future_years_l76_76649

theorem future_years (P A F : ℝ) (Y : ℝ) 
  (h1 : P = 50)
  (h2 : P = 1.25 * A)
  (h3 : P = 5 / 6 * F)
  (h4 : A + 10 + Y = F) : 
  Y = 10 := sorry

end future_years_l76_76649


namespace symmetric_points_l76_76399

variable (a b : ℝ)

def condition_1 := a - 1 = 2
def condition_2 := 5 = -(b - 1)

theorem symmetric_points (h1 : condition_1 a) (h2 : condition_2 b) :
  (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_points_l76_76399


namespace revenue_fraction_l76_76308

variable (N D J : ℝ)
variable (h1 : J = 1 / 5 * N)
variable (h2 : D = 4.166666666666666 * (N + J) / 2)

theorem revenue_fraction (h1 : J = 1 / 5 * N) (h2 : D = 4.166666666666666 * (N + J) / 2) : N / D = 2 / 5 :=
by
  sorry

end revenue_fraction_l76_76308


namespace number_of_ways_to_write_528_as_sum_of_consecutive_integers_l76_76090

theorem number_of_ways_to_write_528_as_sum_of_consecutive_integers : 
  ∃ (n : ℕ), (2 ≤ n ∧ ∃ k : ℕ, n * (2 * k + n - 1) = 1056) ∧ n = 15 :=
by
  sorry

end number_of_ways_to_write_528_as_sum_of_consecutive_integers_l76_76090


namespace notebook_pre_tax_cost_eq_l76_76257

theorem notebook_pre_tax_cost_eq :
  (∃ (n c X : ℝ), n + c = 3 ∧ n = 2 + c ∧ 1.1 * X = 3.3 ∧ X = n + c → n = 2.5) :=
by
  sorry

end notebook_pre_tax_cost_eq_l76_76257


namespace alternating_intersections_l76_76938

theorem alternating_intersections (n : ℕ)
  (roads : Fin n → ℝ → ℝ) -- Roads are functions from reals to reals
  (h_straight : ∀ (i : Fin n), ∃ (a b : ℝ), ∀ x, roads i x = a * x + b) 
  (h_intersect : ∀ (i j : Fin n), i ≠ j → ∃ x, roads i x = roads j x)
  (h_two_roads : ∀ (x y : ℝ), ∃! (i j : Fin n), i ≠ j ∧ roads i x = roads j y) :
  ∃ (design : ∀ (i : Fin n), ℝ → Prop), 
  -- ensuring alternation, road 'i' alternates crossings with other roads 
  (∀ (i : Fin n) (x y : ℝ), 
    roads i x = roads i y → (design i x ↔ ¬design i y)) := sorry

end alternating_intersections_l76_76938


namespace alice_age_2005_l76_76671

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005_l76_76671


namespace negativity_of_c_plus_b_l76_76100

variable (a b c : ℝ)

def isWithinBounds : Prop := (1 < a ∧ a < 2) ∧ (0 < b ∧ b < 1) ∧ (-2 < c ∧ c < -1)

theorem negativity_of_c_plus_b (h : isWithinBounds a b c) : c + b < 0 :=
sorry

end negativity_of_c_plus_b_l76_76100


namespace commodity_x_increase_rate_l76_76992

variable (x_increase : ℕ) -- annual increase in cents of commodity X
variable (y_increase : ℕ := 20) -- annual increase in cents of commodity Y
variable (x_2001_price : ℤ := 420) -- price of commodity X in cents in 2001
variable (y_2001_price : ℤ := 440) -- price of commodity Y in cents in 2001
variable (year_difference : ℕ := 2010 - 2001) -- difference in years between 2010 and 2001
variable (x_y_diff_2010 : ℕ := 70) -- cents by which X is more expensive than Y in 2010

theorem commodity_x_increase_rate :
  x_increase * year_difference = (x_2001_price + x_increase * year_difference) - (y_2001_price + y_increase * year_difference) + x_y_diff_2010 := by
  sorry

end commodity_x_increase_rate_l76_76992


namespace bead_game_solution_l76_76923

-- Define the main theorem, stating the solution is valid for r = (b + 1) / b
theorem bead_game_solution {r : ℚ} (h : r > 1) (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 1010) :
  r = (b + 1) / b ∧ (∀ k : ℕ, k ≤ 2021 → True) := by
  sorry

end bead_game_solution_l76_76923


namespace arithmetic_sequence_n_l76_76800

theorem arithmetic_sequence_n 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3_plus_a5 : a 3 + a 5 = 14)
  (Sn_eq_100 : S n = 100) :
  n = 10 :=
sorry

end arithmetic_sequence_n_l76_76800


namespace vector_operation_result_l76_76171

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result_l76_76171


namespace inverse_proportion_k_value_l76_76010

theorem inverse_proportion_k_value (k : ℝ) (h₁ : k ≠ 0) (h₂ : (2, -1) ∈ {p : ℝ × ℝ | ∃ (k' : ℝ), k' = k ∧ p.snd = k' / p.fst}) :
  k = -2 := 
by
  sorry

end inverse_proportion_k_value_l76_76010


namespace total_weight_on_scale_l76_76510

def weight_blue_ball : ℝ := 6
def weight_brown_ball : ℝ := 3.12

theorem total_weight_on_scale :
  weight_blue_ball + weight_brown_ball = 9.12 :=
by sorry

end total_weight_on_scale_l76_76510


namespace pow_mult_rule_l76_76788

variable (x : ℝ)

theorem pow_mult_rule : (x^3) * (x^2) = x^5 :=
by sorry

end pow_mult_rule_l76_76788


namespace geometric_solution_l76_76972

theorem geometric_solution (x y : ℝ) (h : x^2 + 2 * y^2 - 10 * x + 12 * y + 43 = 0) : x = 5 ∧ y = -3 := 
  by sorry

end geometric_solution_l76_76972


namespace washing_machines_total_pounds_l76_76456

theorem washing_machines_total_pounds (pounds_per_machine_per_day : ℕ) (number_of_machines : ℕ)
  (h1 : pounds_per_machine_per_day = 28) (h2 : number_of_machines = 8) :
  number_of_machines * pounds_per_machine_per_day = 224 :=
by
  sorry

end washing_machines_total_pounds_l76_76456


namespace tony_total_puzzle_time_l76_76801

def warm_up_puzzle_time : ℕ := 10
def number_of_puzzles : ℕ := 2
def multiplier : ℕ := 3
def time_per_puzzle : ℕ := warm_up_puzzle_time * multiplier
def total_time : ℕ := warm_up_puzzle_time + number_of_puzzles * time_per_puzzle

theorem tony_total_puzzle_time : total_time = 70 := 
by
  sorry

end tony_total_puzzle_time_l76_76801


namespace sum_of_20th_and_30th_triangular_numbers_l76_76709

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_20th_and_30th_triangular_numbers :
  triangular_number 20 + triangular_number 30 = 675 :=
by
  sorry

end sum_of_20th_and_30th_triangular_numbers_l76_76709


namespace number_of_shelves_l76_76546

/-- Adam could fit 11 action figures on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- Adam's shelves could hold a total of 44 action figures -/
def total_action_figures_on_shelves : ℕ := 44

/-- Prove the number of shelves in Adam's room -/
theorem number_of_shelves:
  total_action_figures_on_shelves / action_figures_per_shelf = 4 := 
by {
    sorry
}

end number_of_shelves_l76_76546


namespace f_is_even_f_range_l76_76490

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (|x| + 2) / (1 - |x|)

-- Prove that f(x) is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

-- Prove the range of f(x) is (-∞, -1) ∪ [2, +∞)
theorem f_range : ∀ y : ℝ, ∃ x : ℝ, y = f x ↔ y ≥ 2 ∨ y < -1 := by
  sorry

end f_is_even_f_range_l76_76490


namespace rational_solutions_exist_l76_76290

theorem rational_solutions_exist (x p q : ℚ) (h : p^2 - x * q^2 = 1) :
  ∃ (a b : ℤ), p = (a^2 + x * b^2) / (a^2 - x * b^2) ∧ q = (2 * a * b) / (a^2 - x * b^2) :=
by
  sorry

end rational_solutions_exist_l76_76290


namespace remainder_of_division_l76_76996

theorem remainder_of_division (x : ℕ) (r : ℕ) :
  1584 - x = 1335 ∧ 1584 = 6 * x + r → r = 90 := by
  sorry

end remainder_of_division_l76_76996


namespace move_point_right_l76_76231

theorem move_point_right (x y : ℝ) (h₁ : x = 1) (h₂ : y = 1) (dx : ℝ) (h₃ : dx = 2) : (x + dx, y) = (3, 1) :=
by
  rw [h₁, h₂, h₃]
  simp
  sorry

end move_point_right_l76_76231


namespace An_integer_and_parity_l76_76898

theorem An_integer_and_parity (k : Nat) (h : k > 0) : 
  ∀ n ≥ 1, ∃ A : Nat, 
   (A = 1 ∨ (∀ A' : Nat, A' = ( (A * n + 2 * (n+1) ^ (2 * k)) / (n+2)))) 
  ∧ (A % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) := 
by 
  sorry

end An_integer_and_parity_l76_76898


namespace largest_number_with_two_moves_l76_76101

theorem largest_number_with_two_moves (n : Nat) (matches_limit : Nat) (initial_number : Nat)
  (h_n : initial_number = 1405) (h_limit: matches_limit = 2) : n = 7705 :=
by
  sorry

end largest_number_with_two_moves_l76_76101


namespace find_symmetric_point_l76_76460

structure Point := (x : Int) (y : Int)

def translate_right (p : Point) (n : Int) : Point :=
  { x := p.x + n, y := p.y }

def symmetric_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem find_symmetric_point : 
  ∀ (A B C : Point),
  A = ⟨-1, 2⟩ →
  B = translate_right A 2 →
  C = symmetric_x_axis B →
  C = ⟨1, -2⟩ :=
by
  intros A B C hA hB hC
  sorry

end find_symmetric_point_l76_76460


namespace triangle_angle_sum_l76_76638

theorem triangle_angle_sum (x : ℝ) :
  let a := 40
  let b := 60
  let sum_of_angles := 180
  a + b + x = sum_of_angles → x = 80 :=
by
  intros
  sorry

end triangle_angle_sum_l76_76638


namespace read_time_proof_l76_76260

noncomputable def read_time_problem : Prop :=
  ∃ (x y : ℕ), 
    x > 0 ∧
    y = 480 / x ∧
    (y - 5) = 480 / (x + 16) ∧
    y = 15

theorem read_time_proof : read_time_problem := 
sorry

end read_time_proof_l76_76260


namespace circle_radius_eq_five_l76_76889

theorem circle_radius_eq_five : 
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 8 * y = 0) → (∃ r : ℝ, ((x - 3)^2 + (y + 4)^2 = r^2) ∧ r = 5) :=
by
  sorry

end circle_radius_eq_five_l76_76889


namespace justin_and_tim_play_same_game_210_times_l76_76893

def number_of_games_with_justin_and_tim : ℕ :=
  have num_players : ℕ := 12
  have game_size : ℕ := 6
  have justin_and_tim_fixed : ℕ := 2
  have remaining_players : ℕ := num_players - justin_and_tim_fixed
  have players_to_choose : ℕ := game_size - justin_and_tim_fixed
  Nat.choose remaining_players players_to_choose

theorem justin_and_tim_play_same_game_210_times :
  number_of_games_with_justin_and_tim = 210 :=
by sorry

end justin_and_tim_play_same_game_210_times_l76_76893


namespace intersection_M_P_l76_76877

def M : Set ℝ := {0, 1, 2, 3}
def P : Set ℝ := {x | 0 ≤ x ∧ x < 2}

theorem intersection_M_P : M ∩ P = {0, 1} := 
by
  -- You can fill in the proof here
  sorry

end intersection_M_P_l76_76877


namespace additional_grassy_area_l76_76365

theorem additional_grassy_area (r1 r2 : ℝ) (r1_pos : r1 = 10) (r2_pos : r2 = 35) : 
  let A1 := π * r1^2
  let A2 := π * r2^2
  (A2 - A1) = 1125 * π :=
by 
  sorry

end additional_grassy_area_l76_76365


namespace cos_double_angle_l76_76680

theorem cos_double_angle (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
sorry

end cos_double_angle_l76_76680


namespace ribbon_per_gift_l76_76161

theorem ribbon_per_gift
  (total_ribbon : ℕ)
  (number_of_gifts : ℕ)
  (ribbon_left : ℕ)
  (used_ribbon := total_ribbon - ribbon_left)
  (ribbon_per_gift := used_ribbon / number_of_gifts)
  (h_total : total_ribbon = 18)
  (h_gifts : number_of_gifts = 6)
  (h_left : ribbon_left = 6) :
  ribbon_per_gift = 2 := by
  sorry

end ribbon_per_gift_l76_76161


namespace unique_x1_exists_l76_76753

theorem unique_x1_exists (x : ℕ → ℝ) :
  (∀ n : ℕ+, x (n+1) = x n * (x n + 1 / n)) →
  ∃! (x1 : ℝ), (∀ n : ℕ+, 0 < x n ∧ x n < x (n+1) ∧ x (n+1) < 1) :=
sorry

end unique_x1_exists_l76_76753


namespace inequality_le_one_equality_case_l76_76707

open Real

theorem inequality_le_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) :=
sorry

theorem equality_case (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_le_one_equality_case_l76_76707


namespace imaginary_part_of_z_l76_76513

-- Define the complex number z
def z : Complex := Complex.mk 3 (-4)

-- State the proof goal
theorem imaginary_part_of_z : z.im = -4 :=
by
  sorry

end imaginary_part_of_z_l76_76513


namespace additional_pots_last_hour_l76_76314

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l76_76314


namespace factorize_difference_of_squares_factorize_cubic_l76_76046

-- Problem 1: Prove that 4x^2 - 36 = 4(x + 3)(x - 3)
theorem factorize_difference_of_squares (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := 
  sorry

-- Problem 2: Prove that x^3 - 2x^2y + xy^2 = x(x - y)^2
theorem factorize_cubic (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
  sorry

end factorize_difference_of_squares_factorize_cubic_l76_76046


namespace hemisphere_containers_needed_l76_76278

theorem hemisphere_containers_needed 
  (total_volume : ℕ) (volume_per_hemisphere : ℕ) 
  (h₁ : total_volume = 11780) 
  (h₂ : volume_per_hemisphere = 4) : 
  total_volume / volume_per_hemisphere = 2945 := 
by
  sorry

end hemisphere_containers_needed_l76_76278


namespace greatest_divisor_of_product_of_four_consecutive_integers_l76_76183

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end greatest_divisor_of_product_of_four_consecutive_integers_l76_76183


namespace fifth_term_sum_of_powers_of_4_l76_76144

theorem fifth_term_sum_of_powers_of_4 :
  (4^0 + 4^1 + 4^2 + 4^3 + 4^4) = 341 := 
by
  sorry

end fifth_term_sum_of_powers_of_4_l76_76144


namespace trig_identity_l76_76251

variable (α : ℝ)
variable (h : Real.sin α = 3 / 5)

theorem trig_identity : Real.sin (Real.pi / 2 + 2 * α) = 7 / 25 :=
by
  sorry

end trig_identity_l76_76251


namespace coat_price_reduction_l76_76009

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 :=
by {
  sorry -- This is where the proof would go
}

end coat_price_reduction_l76_76009


namespace arithmetic_sequence_problem_l76_76526

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (a1 : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :=
  ∀ n, a n = a1 + n * d

-- Given condition
variable (h1 : a 3 + a 4 + a 5 = 36)

-- The goal is to prove that a 0 + a 8 = 24
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (a1 : ℝ) (d : ℝ) :
  arithmetic_sequence a a1 d →
  a 3 + a 4 + a 5 = 36 →
  a 0 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_problem_l76_76526


namespace expression_evaluation_l76_76495

-- Definitions of the expressions
def expr (x y : ℤ) : ℤ :=
  ((x - 2 * y) ^ 2 + (3 * x - y) * (3 * x + y) - 3 * y ^ 2) / (-2 * x)

-- Proof that the expression evaluates to -11 when x = 1 and y = -3
theorem expression_evaluation : expr 1 (-3) = -11 :=
by
  -- Declarations
  let x := 1
  let y := -3
  -- The core calculation
  show expr x y = -11
  sorry

end expression_evaluation_l76_76495


namespace basket_A_apples_count_l76_76558

-- Conditions
def total_baskets : ℕ := 5
def avg_fruits_per_basket : ℕ := 25
def fruits_in_B : ℕ := 30
def fruits_in_C : ℕ := 20
def fruits_in_D : ℕ := 25
def fruits_in_E : ℕ := 35

-- Calculation of total number of fruits
def total_fruits : ℕ := total_baskets * avg_fruits_per_basket
def other_baskets_fruits : ℕ := fruits_in_B + fruits_in_C + fruits_in_D + fruits_in_E

-- Question and Proof Goal
theorem basket_A_apples_count : total_fruits - other_baskets_fruits = 15 := by
  sorry

end basket_A_apples_count_l76_76558


namespace right_triangle_hypotenuse_equals_area_l76_76946

/-- Given a right triangle where the hypotenuse is equal to the area, 
    show that the scaling factor x satisfies the equation. -/
theorem right_triangle_hypotenuse_equals_area 
  (m n x : ℝ) (h_hyp: (m^2 + n^2) * x = mn * (m^2 - n^2) * x^2) :
  x = (m^2 + n^2) / (mn * (m^2 - n^2)) := 
by
  sorry

end right_triangle_hypotenuse_equals_area_l76_76946


namespace company_ordered_weight_of_stone_l76_76326

theorem company_ordered_weight_of_stone :
  let weight_concrete := 0.16666666666666666
  let weight_bricks := 0.16666666666666666
  let total_material := 0.8333333333333334
  let weight_stone := total_material - (weight_concrete + weight_bricks)
  weight_stone = 0.5 :=
by
  sorry

end company_ordered_weight_of_stone_l76_76326


namespace odd_n_divides_3n_plus_1_is_1_l76_76868

theorem odd_n_divides_3n_plus_1_is_1 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) (h3 : n ∣ 3^n + 1) : n = 1 :=
sorry

end odd_n_divides_3n_plus_1_is_1_l76_76868


namespace total_cost_is_17_l76_76615

def taco_shells_cost : ℝ := 5
def bell_pepper_cost_per_unit : ℝ := 1.5
def bell_pepper_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2

def total_spent : ℝ :=
  taco_shells_cost + (bell_pepper_cost_per_unit * bell_pepper_quantity) + (meat_cost_per_pound * meat_quantity)

theorem total_cost_is_17 : total_spent = 17 := 
  by sorry

end total_cost_is_17_l76_76615


namespace second_largest_of_five_consecutive_is_19_l76_76536

theorem second_largest_of_five_consecutive_is_19 (n : ℕ) 
  (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 90): 
  n + 3 = 19 :=
by sorry

end second_largest_of_five_consecutive_is_19_l76_76536


namespace value_of_x_in_equation_l76_76997

theorem value_of_x_in_equation : 
  (∀ x : ℕ, 8 ^ 17 + 8 ^ 17 + 8 ^ 17 + 8 ^ 17 = 2 ^ x → x = 53) := 
by 
  sorry

end value_of_x_in_equation_l76_76997


namespace min_value_of_T_l76_76828

noncomputable def T (x p : ℝ) : ℝ := |x - p| + |x - 15| + |x - (15 + p)|

theorem min_value_of_T (p : ℝ) (hp : 0 < p ∧ p < 15) :
  ∃ x, p ≤ x ∧ x ≤ 15 ∧ T x p = 15 :=
sorry

end min_value_of_T_l76_76828


namespace range_of_a_l76_76839

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a * x + 5 < 0) ↔ (a < -2 * Real.sqrt 5 ∨ a > 2 * Real.sqrt 5) := 
by 
  sorry

end range_of_a_l76_76839


namespace bob_pays_more_than_samantha_l76_76272

theorem bob_pays_more_than_samantha
  (total_slices : ℕ := 12)
  (cost_plain_pizza : ℝ := 12)
  (cost_olives : ℝ := 3)
  (slices_one_third_pizza : ℕ := total_slices / 3)
  (total_cost : ℝ := cost_plain_pizza + cost_olives)
  (cost_per_slice : ℝ := total_cost / total_slices)
  (bob_slices_total : ℕ := slices_one_third_pizza + 3)
  (samantha_slices_total : ℕ := total_slices - bob_slices_total)
  (bob_total_cost : ℝ := bob_slices_total * cost_per_slice)
  (samantha_total_cost : ℝ := samantha_slices_total * cost_per_slice) :
  bob_total_cost - samantha_total_cost = 2.5 :=
by
  sorry

end bob_pays_more_than_samantha_l76_76272


namespace principal_amount_borrowed_l76_76831

theorem principal_amount_borrowed (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (R_eq : R = 12) (T_eq : T = 3) (SI_eq : SI = 7200) :
  (SI = (P * R * T) / 100) → P = 20000 :=
by sorry

end principal_amount_borrowed_l76_76831


namespace price_reduction_l76_76152

theorem price_reduction (C : ℝ) (h1 : C > 0) :
  let first_discounted_price := 0.7 * C
  let final_discounted_price := 0.8 * first_discounted_price
  let reduction := 1 - final_discounted_price / C
  reduction = 0.44 :=
by
  sorry

end price_reduction_l76_76152


namespace Flynn_tv_minutes_weekday_l76_76794

theorem Flynn_tv_minutes_weekday :
  ∀ (tv_hours_per_weekend : ℕ)
    (tv_hours_per_year : ℕ)
    (weeks_per_year : ℕ) 
    (weekdays_per_week : ℕ),
  tv_hours_per_weekend = 2 →
  tv_hours_per_year = 234 →
  weeks_per_year = 52 →
  weekdays_per_week = 5 →
  (tv_hours_per_year - (tv_hours_per_weekend * weeks_per_year)) / (weekdays_per_week * weeks_per_year) * 60
  = 30 :=
by
  intros tv_hours_per_weekend tv_hours_per_year weeks_per_year weekdays_per_week
        h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end Flynn_tv_minutes_weekday_l76_76794


namespace opposite_numbers_abs_l76_76375

theorem opposite_numbers_abs (a b : ℤ) (h : a + b = 0) : |a - 2014 + b| = 2014 :=
by
  -- proof here
  sorry

end opposite_numbers_abs_l76_76375


namespace asha_remaining_money_l76_76790

theorem asha_remaining_money :
  let brother := 20
  let father := 40
  let mother := 30
  let granny := 70
  let savings := 100
  let total_money := brother + father + mother + granny + savings
  let spent := (3 / 4) * total_money
  let remaining := total_money - spent
  remaining = 65 :=
by
  sorry

end asha_remaining_money_l76_76790


namespace chris_pounds_of_nuts_l76_76586

theorem chris_pounds_of_nuts :
  ∀ (R : ℝ) (x : ℝ),
  (∃ (N : ℝ), N = 4 * R) →
  (∃ (total_mixture_cost : ℝ), total_mixture_cost = 3 * R + 4 * R * x) →
  (3 * R = 0.15789473684210525 * total_mixture_cost) →
  x = 4 :=
by
  intros R x hN htotal_mixture_cost hRA
  sorry

end chris_pounds_of_nuts_l76_76586


namespace A_not_divisible_by_B_l76_76136

variable (A B : ℕ)
variable (h1 : A ≠ B)
variable (h2 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))
variable (h3 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))

theorem A_not_divisible_by_B : ¬ (A % B = 0) :=
sorry

end A_not_divisible_by_B_l76_76136


namespace range_of_m_l76_76856

theorem range_of_m (m : ℝ) (x : ℝ) :
  (|1 - (x - 1) / 2| ≤ 3) →
  (x^2 - 2 * x + 1 - m^2 ≤ 0) →
  (m > 0) →
  (∃ (q_is_necessary_but_not_sufficient_for_p : Prop), q_is_necessary_but_not_sufficient_for_p →
  (m ≥ 8)) :=
by
  sorry

end range_of_m_l76_76856


namespace overlap_region_area_l76_76591

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1/2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

noncomputable def overlap_area : ℝ := 
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  let P1 : ℝ × ℝ := (2, 2);
  let P2 : ℝ × ℝ := (4, 2);
  let P3 : ℝ × ℝ := (3, 3);
  let P4 : ℝ × ℝ := (2, 3);
  1/2 * abs (P1.1 * (P2.2 - P4.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P4.2 - P2.2) + P4.1 * (P1.2 - P3.2))

theorem overlap_region_area :
  let A : ℝ × ℝ := (0, 0);
  let B : ℝ × ℝ := (6, 2);
  let C : ℝ × ℝ := (2, 6);
  let D : ℝ × ℝ := (6, 6);
  let E : ℝ × ℝ := (0, 2);
  let F : ℝ × ℝ := (2, 0);
  triangle_area A B C > 0 →
  triangle_area D E F > 0 →
  overlap_area = 0.5 :=
by { sorry }

end overlap_region_area_l76_76591


namespace find_a_l76_76754

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 < a^2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}
def C : Set ℝ := {x | 1 < x ∧ x < 2}

theorem find_a (a : ℝ) (h : A a ∩ B = C) : a = 2 ∨ a = -2 := by
  sorry

end find_a_l76_76754


namespace fifth_digit_is_one_l76_76179

def self_descriptive_seven_digit_number (A B C D E F G : ℕ) : Prop :=
  A = 3 ∧ B = 2 ∧ C = 2 ∧ D = 1 ∧ E = 1 ∧ [A, B, C, D, E, F, G].count 0 = A ∧
  [A, B, C, D, E, F, G].count 1 = B ∧ [A, B, C, D, E, F, G].count 2 = C ∧
  [A, B, C, D, E, F, G].count 3 = D ∧ [A, B, C, D, E, F, G].count 4 = E

theorem fifth_digit_is_one
  (A B C D E F G : ℕ) (h : self_descriptive_seven_digit_number A B C D E F G) : E = 1 := by
  sorry

end fifth_digit_is_one_l76_76179


namespace real_solution_exists_l76_76432

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end real_solution_exists_l76_76432


namespace vasya_hits_ship_l76_76847

theorem vasya_hits_ship (board_size : ℕ) (ship_length : ℕ) (shots : ℕ) : 
  board_size = 10 ∧ ship_length = 4 ∧ shots = 24 → ∃ strategy : Fin board_size × Fin board_size → Prop, 
  (∀ pos, strategy pos → pos.1 * board_size + pos.2 < shots) ∧ 
  ∀ (ship_pos : Fin board_size × Fin board_size) (horizontal : Bool), 
  ∃ shot_pos, strategy shot_pos ∧ 
  (if horizontal then 
    ship_pos.1 = shot_pos.1 ∧ ship_pos.2 ≤ shot_pos.2 ∧ shot_pos.2 < ship_pos.2 + ship_length 
  else 
    ship_pos.2 = shot_pos.2 ∧ ship_pos.1 ≤ shot_pos.1 ∧ shot_pos.1 < ship_pos.1 + ship_length) :=
sorry

end vasya_hits_ship_l76_76847


namespace sum_modified_midpoint_coordinates_l76_76092

theorem sum_modified_midpoint_coordinates :
  let p1 : (ℝ × ℝ) := (10, 3)
  let p2 : (ℝ × ℝ) := (-4, 7)
  let midpoint : (ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let modified_x := 2 * midpoint.1 
  (modified_x + midpoint.2) = 11 := by
  sorry

end sum_modified_midpoint_coordinates_l76_76092


namespace triangle_is_isosceles_l76_76821

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (triangle : Type)

noncomputable def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) (triangle : Type) : Prop :=
  c = 2 * a * Real.cos B → A = B ∨ B = C ∨ C = A

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) (triangle : Type) (h : c = 2 * a * Real.cos B) :
  is_isosceles_triangle A B C a b c triangle :=
sorry

end triangle_is_isosceles_l76_76821


namespace recycling_points_l76_76838

theorem recycling_points (chloe_recycled : ℤ) (friends_recycled : ℤ) (points_per_pound : ℤ) :
  chloe_recycled = 28 ∧ friends_recycled = 2 ∧ points_per_pound = 6 → (chloe_recycled + friends_recycled) / points_per_pound = 5 :=
by
  sorry

end recycling_points_l76_76838


namespace largest_divisor_l76_76820

theorem largest_divisor (n : ℕ) (hn : n > 0) (h : 360 ∣ n^3) :
  ∃ w : ℕ, w > 0 ∧ w ∣ n ∧ ∀ d : ℕ, (d > 0 ∧ d ∣ n) → d ≤ 30 := 
sorry

end largest_divisor_l76_76820


namespace distance_with_tide_60_min_l76_76578

variable (v_m v_t : ℝ)

axiom man_with_tide : (v_m + v_t) = 5
axiom man_against_tide : (v_m - v_t) = 4

theorem distance_with_tide_60_min : (v_m + v_t) = 5 := by
  sorry

end distance_with_tide_60_min_l76_76578


namespace largest_possible_difference_l76_76235

theorem largest_possible_difference 
  (weight_A weight_B weight_C : ℝ)
  (hA : 24.9 ≤ weight_A ∧ weight_A ≤ 25.1)
  (hB : 24.8 ≤ weight_B ∧ weight_B ≤ 25.2)
  (hC : 24.7 ≤ weight_C ∧ weight_C ≤ 25.3) :
  ∃ w1 w2 : ℝ, (w1 = weight_C ∧ w2 = weight_C ∧ abs (w1 - w2) = 0.6) :=
by
  sorry

end largest_possible_difference_l76_76235


namespace three_not_divide_thirtyone_l76_76782

theorem three_not_divide_thirtyone : ¬ ∃ q : ℤ, 31 = 3 * q := sorry

end three_not_divide_thirtyone_l76_76782


namespace divisibility_1989_l76_76071

theorem divisibility_1989 (n : ℕ) (h1 : n ≥ 3) :
  1989 ∣ n^(n^(n^n)) - n^(n^n) :=
sorry

end divisibility_1989_l76_76071


namespace find_speed_of_man_in_still_water_l76_76047

def speed_of_man_in_still_water (t1 t2 d1 d2: ℝ) (v_m v_s: ℝ) : Prop :=
  d1 / t1 = v_m + v_s ∧ d2 / t2 = v_m - v_s

theorem find_speed_of_man_in_still_water :
  ∃ v_m : ℝ, ∃ v_s : ℝ, speed_of_man_in_still_water 2 2 16 10 v_m v_s ∧ v_m = 6.5 :=
by
  sorry

end find_speed_of_man_in_still_water_l76_76047


namespace solve_floor_sum_eq_125_l76_76506

def floorSum (x : ℕ) : ℕ :=
  (x - 1) * x * (4 * x + 1) / 6

theorem solve_floor_sum_eq_125 (x : ℕ) (h_pos : 0 < x) : floorSum x = 125 → x = 6 := by
  sorry

end solve_floor_sum_eq_125_l76_76506


namespace quadratic_trinomial_prime_l76_76584

theorem quadratic_trinomial_prime (p x : ℤ) (hp : p > 1) (hx : 0 ≤ x ∧ x < p)
  (h_prime : Prime (x^2 - x + p)) : x = 0 ∨ x = 1 :=
by
  sorry

end quadratic_trinomial_prime_l76_76584


namespace determine_a_l76_76342

noncomputable def f (a x : ℝ) : ℝ := a^2 * x^2 - 2 * a * x + 1 

theorem determine_a (a : ℝ) (h : ¬ (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0)) : a > 1 :=
sorry

end determine_a_l76_76342


namespace circle_area_l76_76107

/-
Circle A has a diameter equal to the radius of circle B.
The area of circle A is 16π square units.
Prove the area of circle B is 64π square units.
-/

theorem circle_area (rA dA rB : ℝ) (h1 : dA = 2 * rA) (h2 : rB = dA) (h3 : π * rA ^ 2 = 16 * π) : π * rB ^ 2 = 64 * π :=
by
  sorry

end circle_area_l76_76107


namespace simplify_expression_l76_76590

theorem simplify_expression (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 :=
by
  sorry

end simplify_expression_l76_76590


namespace calculate_minutes_worked_today_l76_76748

-- Define the conditions
def production_rate := 6 -- shirts per minute
def total_shirts_today := 72 

-- The statement to prove
theorem calculate_minutes_worked_today :
  total_shirts_today / production_rate = 12 := 
by
  sorry

end calculate_minutes_worked_today_l76_76748


namespace unique_solutions_of_system_l76_76585

def system_of_equations (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

theorem unique_solutions_of_system :
  ∀ (x y : ℝ), system_of_equations x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end unique_solutions_of_system_l76_76585


namespace total_games_played_l76_76600

theorem total_games_played (jerry_wins dave_wins ken_wins : ℕ)
  (h1 : jerry_wins = 7)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : ken_wins = dave_wins + 5) :
  jerry_wins + dave_wins + ken_wins = 32 :=
by
  sorry

end total_games_played_l76_76600


namespace cost_price_of_computer_table_l76_76357

theorem cost_price_of_computer_table (S : ℝ) (C : ℝ) (h1 : 1.80 * C = S) (h2 : S = 3500) : C = 1944.44 :=
by
  sorry

end cost_price_of_computer_table_l76_76357


namespace exponential_inequality_l76_76744

-- Define the conditions for the problem
variables {x y a : ℝ}
axiom h1 : x > y
axiom h2 : y > 1
axiom h3 : 0 < a
axiom h4 : a < 1

-- State the problem to be proved
theorem exponential_inequality (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < 1) : a ^ x < a ^ y :=
sorry

end exponential_inequality_l76_76744


namespace fencing_rate_l76_76072

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let circumference := Real.pi * d
  total_cost / circumference

theorem fencing_rate (diameter cost : ℝ) (h₀ : diameter = 34) (h₁ : cost = 213.63) :
  rate_per_meter diameter cost = 2 := by
  sorry

end fencing_rate_l76_76072


namespace ants_meet_distance_is_half_total_l76_76799

-- Definitions given in the problem
structure Tile :=
  (width : ℤ)
  (length : ℤ)

structure Ant :=
  (start_position : String)

-- Conditions from the problem
def tile : Tile := ⟨4, 6⟩
def maricota : Ant := ⟨"M"⟩
def nandinha : Ant := ⟨"N"⟩
def total_lengths := 14
def total_widths := 12

noncomputable
def calculate_total_distance (total_lengths : ℤ) (total_widths : ℤ) (tile : Tile) := 
  (total_lengths * tile.length) + (total_widths * tile.width)

-- Question stated as a theorem
theorem ants_meet_distance_is_half_total :
  calculate_total_distance total_lengths total_widths tile = 132 →
  (calculate_total_distance total_lengths total_widths tile) / 2 = 66 :=
by
  intro h
  sorry

end ants_meet_distance_is_half_total_l76_76799


namespace train_speed_in_km_hr_l76_76666

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 9.679225661947045
noncomputable def distance_covered : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := distance_covered / crossing_time
noncomputable def speed_km_hr : ℝ := speed_m_s * 3.6

theorem train_speed_in_km_hr : speed_km_hr = 90.0216 := by
  sorry

end train_speed_in_km_hr_l76_76666


namespace tom_age_ratio_l76_76548

variable (T M : ℕ)
variable (h1 : T = T) -- Tom's age is equal to the sum of the ages of his four children
variable (h2 : T - M = 3 * (T - 4 * M)) -- M years ago, Tom's age was three times the sum of his children's ages then

theorem tom_age_ratio : (T / M) = 11 / 2 := 
by
  sorry

end tom_age_ratio_l76_76548


namespace real_solution_unique_l76_76729

theorem real_solution_unique (x : ℝ) (h : x^4 + (2 - x)^4 + 2 * x = 34) : x = 0 :=
sorry

end real_solution_unique_l76_76729


namespace slope_of_parallel_line_l76_76146

theorem slope_of_parallel_line (m : ℚ) (b : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 * y = 21 → y = (5 / 3) * x + b) →
  m = 5 / 3 :=
by
  intros hyp
  sorry

end slope_of_parallel_line_l76_76146


namespace children_group_size_l76_76829

theorem children_group_size (x : ℕ) (h1 : 255 % 17 = 0) (h2: ∃ n : ℕ, n * 17 = 255) 
                            (h3 : ∀ a c, a = c → a = 255 → c = 255 → x = 17) : 
                            (255 / x = 15) → x = 17 :=
by
  sorry

end children_group_size_l76_76829


namespace two_cos_45_eq_sqrt_two_l76_76953

theorem two_cos_45_eq_sqrt_two
  (h1 : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2) :
  2 * Real.cos (Real.pi / 4) = Real.sqrt 2 :=
sorry

end two_cos_45_eq_sqrt_two_l76_76953


namespace Maya_takes_longer_l76_76477

-- Define the constants according to the conditions
def Xavier_reading_speed : ℕ := 120
def Maya_reading_speed : ℕ := 60
def novel_pages : ℕ := 360
def minutes_per_hour : ℕ := 60

-- Define the times it takes for Xavier and Maya to read the novel
def Xavier_time : ℕ := novel_pages / Xavier_reading_speed
def Maya_time : ℕ := novel_pages / Maya_reading_speed

-- Define the time difference in hours and then in minutes
def time_difference_hours : ℕ := Maya_time - Xavier_time
def time_difference_minutes : ℕ := time_difference_hours * minutes_per_hour

-- The statement to prove
theorem Maya_takes_longer :
  time_difference_minutes = 180 :=
by
  sorry

end Maya_takes_longer_l76_76477


namespace sum_of_all_possible_values_of_g7_l76_76911

def f (x : ℝ) : ℝ := x ^ 2 - 6 * x + 14
def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g7 :
  let x1 := 3 + Real.sqrt 2;
  let x2 := 3 - Real.sqrt 2;
  let g1 := g x1;
  let g2 := g x2;
  g (f 7) = g1 + g2 := by
  sorry

end sum_of_all_possible_values_of_g7_l76_76911


namespace range_of_m_l76_76810

theorem range_of_m (m : ℝ) : 
  (¬ (∀ x : ℝ, x^2 + m * x + 1 = 0 → x > 0) → m ≥ -2) :=
by
  sorry

end range_of_m_l76_76810


namespace cantor_length_formula_l76_76031

noncomputable def cantor_length : ℕ → ℚ
| 0 => 1
| (n+1) => 2/3 * cantor_length n

theorem cantor_length_formula (n : ℕ) : cantor_length n = (2/3 : ℚ)^(n-1) :=
  sorry

end cantor_length_formula_l76_76031


namespace problem_c_l76_76112

noncomputable def M (a b : ℝ) := (a^4 + b^4) * (a^2 + b^2)
noncomputable def N (a b : ℝ) := (a^3 + b^3) ^ 2

theorem problem_c (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_neq : a ≠ b) : M a b > N a b := 
by
  -- Proof goes here
  sorry

end problem_c_l76_76112


namespace salary_january_l76_76435

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8300)
  (h3 : May = 6500) :
  J = 5300 :=
by
  sorry

end salary_january_l76_76435


namespace ordered_pair_A_B_l76_76098

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 6
noncomputable def linear_function (x : ℝ) : ℝ := -2 / 3 * x + 2

noncomputable def points_intersect (x1 x2 x3 y1 y2 y3 : ℝ) : Prop :=
  cubic_function x1 = y1 ∧ cubic_function x2 = y2 ∧ cubic_function x3 = y3 ∧
  2 * x1 + 3 * y1 = 6 ∧ 2 * x2 + 3 * y2 = 6 ∧ 2 * x3 + 3 * y3 = 6

theorem ordered_pair_A_B (x1 x2 x3 y1 y2 y3 A B : ℝ)
  (h_intersect : points_intersect x1 x2 x3 y1 y2 y3) 
  (h_sum_x : x1 + x2 + x3 = A)
  (h_sum_y : y1 + y2 + y3 = B) :
  (A, B) = (2, 14 / 3) :=
by {
  sorry
}

end ordered_pair_A_B_l76_76098


namespace cups_of_flour_put_in_l76_76962

-- Conditions
def recipeSugar : ℕ := 3
def recipeFlour : ℕ := 10
def neededMoreFlourThanSugar : ℕ := 5

-- Question: How many cups of flour did she put in?
-- Answer: 5 cups of flour
theorem cups_of_flour_put_in : (recipeSugar + neededMoreFlourThanSugar = recipeFlour) → recipeFlour - neededMoreFlourThanSugar = 5 := 
by
  intros h
  sorry

end cups_of_flour_put_in_l76_76962


namespace ratio_of_groups_l76_76644

variable (x : ℚ)

-- The total number of people in the calligraphy group
def calligraphy_group (x : ℚ) := x + (2 / 7) * x

-- The total number of people in the recitation group
def recitation_group (x : ℚ) := x + (1 / 5) * x

theorem ratio_of_groups (x : ℚ) (hx : x ≠ 0) : 
    (calligraphy_group x) / (recitation_group x) = (3 : ℚ) / (4 : ℚ) := by
  sorry

end ratio_of_groups_l76_76644


namespace problem1_problem2_l76_76501

open Real

/-- Problem 1: Simplify trigonometric expression. -/
theorem problem1 : 
  (sqrt (1 - 2 * sin (10 * pi / 180) * cos (10 * pi / 180)) /
  (sin (170 * pi / 180) - sqrt (1 - sin (170 * pi / 180)^2))) = -1 :=
sorry

/-- Problem 2: Given tan(θ) = 2, find the value.
  Required to prove: 2 + sin(θ) * cos(θ) - cos(θ)^2 equals 11/5 -/
theorem problem2 (θ : ℝ) (h : tan θ = 2) :
  2 + sin θ * cos θ - cos θ^2 = 11 / 5 :=
sorry

end problem1_problem2_l76_76501


namespace solution_l76_76367

-- Define the conditions
def equation (x : ℝ) : Prop :=
  (x / 15) = (15 / x)

theorem solution (x : ℝ) : equation x → x = 15 ∨ x = -15 :=
by
  intros h
  -- The proof would go here.
  sorry

end solution_l76_76367


namespace line_of_intersecting_circles_l76_76519

theorem line_of_intersecting_circles
  (A B : ℝ × ℝ)
  (hAB1 : A.1^2 + A.2^2 + 4 * A.1 - 4 * A.2 = 0)
  (hAB2 : B.1^2 + B.2^2 + 4 * B.1 - 4 * B.2 = 0)
  (hAB3 : A.1^2 + A.2^2 + 2 * A.1 - 12 = 0)
  (hAB4 : B.1^2 + B.2^2 + 2 * B.1 - 12 = 0) :
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧ a * B.1 + b * B.2 + c = 0 ∧
                  a = 1 ∧ b = -2 ∧ c = 6 :=
sorry

end line_of_intersecting_circles_l76_76519


namespace tariffs_impact_but_no_timeframe_l76_76181

noncomputable def cost_of_wine_today : ℝ := 20.00
noncomputable def increase_percentage : ℝ := 0.25
noncomputable def bottles_count : ℕ := 5
noncomputable def price_increase_for_bottles : ℝ := 25.00

theorem tariffs_impact_but_no_timeframe :
  ¬ ∃ (t : ℝ), (cost_of_wine_today * (1 + increase_percentage) - cost_of_wine_today) * bottles_count = price_increase_for_bottles →
  (t = sorry) :=
by 
  sorry

end tariffs_impact_but_no_timeframe_l76_76181


namespace sum_proper_divisors_81_l76_76679

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end sum_proper_divisors_81_l76_76679


namespace B_gain_correct_l76_76636

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def gain_of_B : ℝ :=
  let principal : ℝ := 3150
  let interest_rate_A_to_B : ℝ := 0.08
  let annual_compound : ℕ := 1
  let time_A_to_B : ℝ := 3

  let interest_rate_B_to_C : ℝ := 0.125
  let semiannual_compound : ℕ := 2
  let time_B_to_C : ℝ := 2.5

  let amount_A_to_B := compound_interest principal interest_rate_A_to_B annual_compound time_A_to_B
  let amount_B_to_C := compound_interest principal interest_rate_B_to_C semiannual_compound time_B_to_C

  amount_B_to_C - amount_A_to_B

theorem B_gain_correct : gain_of_B = 282.32 :=
  sorry

end B_gain_correct_l76_76636


namespace prime_divides_30_l76_76895

theorem prime_divides_30 (p : ℕ) (h_prime : Prime p) (h_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
  sorry

end prime_divides_30_l76_76895


namespace rolls_remaining_to_sell_l76_76784

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l76_76784


namespace unattainable_y_value_l76_76563

theorem unattainable_y_value (y : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : ¬ (y = -1 / 3) :=
by {
  -- The proof is omitted for now. 
  -- We're only constructing the outline with necessary imports and conditions.
  sorry
}

end unattainable_y_value_l76_76563


namespace min_value_expression_l76_76832

theorem min_value_expression (x : ℝ) (hx : x > 0) : 9 * x + 1 / x^3 ≥ 10 :=
sorry

end min_value_expression_l76_76832


namespace area_ratio_none_of_these_l76_76608

theorem area_ratio_none_of_these (h r a : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) (a_pos : 0 < a) (h_square_a_square : h^2 > a^2) :
  ¬ (∃ ratio, ratio = (π * r / (h + r)) ∨
               ratio = (π * r^2 / (a + h)) ∨
               ratio = (π * a * r / (h + 2 * r)) ∨
               ratio = (π * r / (a + r))) :=
by sorry

end area_ratio_none_of_these_l76_76608


namespace unique_intersection_point_l76_76694

theorem unique_intersection_point (c : ℝ) :
  (∀ x : ℝ, (|x - 20| + |x + 18| = x + c) → (x = 18 - 2 \/ x = 38 - x \/ x = 2 - 3 * x)) →
  c = 18 :=
by
  sorry

end unique_intersection_point_l76_76694


namespace geometric_sequence_sum_l76_76048

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a1 : a 1 = 3)
  (h_sum : a 1 + a 3 + a 5 = 21) : 
  a 3 + a 5 + a 7 = 42 :=
sorry

end geometric_sequence_sum_l76_76048


namespace exponential_graph_passes_through_point_l76_76446

variable (a : ℝ) (hx1 : a > 0) (hx2 : a ≠ 1)

theorem exponential_graph_passes_through_point :
  ∃ y : ℝ, (y = a^0 + 1) ∧ (y = 2) :=
sorry

end exponential_graph_passes_through_point_l76_76446


namespace geom_seq_sum_of_terms_l76_76623

theorem geom_seq_sum_of_terms
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geometric: ∀ n, a (n + 1) = a n * q)
  (h_q : q = 2)
  (h_sum : a 0 + a 1 + a 2 = 21)
  (h_pos : ∀ n, a n > 0) :
  a 2 + a 3 + a 4 = 84 :=
by
  sorry

end geom_seq_sum_of_terms_l76_76623


namespace cake_stand_cost_calculation_l76_76672

-- Define the constants given in the problem
def flour_cost : ℕ := 5
def money_given : ℕ := 43
def change_received : ℕ := 10

-- Define the cost of the cake stand based on the problem's conditions
def cake_stand_cost : ℕ := (money_given - change_received) - flour_cost

-- The theorem we want to prove
theorem cake_stand_cost_calculation : cake_stand_cost = 28 :=
by
  sorry

end cake_stand_cost_calculation_l76_76672


namespace diamond_value_l76_76884

variable {a b : ℤ}

-- Define the operation diamond following the given condition.
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Define the conditions given in the problem.
axiom h1 : a + b = 10
axiom h2 : a * b = 24

-- State the target theorem.
theorem diamond_value : diamond a b = 5 / 12 :=
by
  sorry

end diamond_value_l76_76884


namespace relationship_of_inequalities_l76_76087

theorem relationship_of_inequalities (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a > b) → (a^2 > b^2)) ∧ 
  ¬ (∀ a b : ℝ, (a^2 > b^2) → (a > b)) := 
by 
  sorry

end relationship_of_inequalities_l76_76087


namespace cost_of_pen_l76_76582

theorem cost_of_pen :
  ∃ p q : ℚ, (3 * p + 4 * q = 264) ∧ (4 * p + 2 * q = 230) ∧ (p = 39.2) :=
by
  sorry

end cost_of_pen_l76_76582


namespace find_a_and_tangent_point_l76_76285

noncomputable def tangent_line_and_curve (a : ℚ) (P : ℚ × ℚ) : Prop :=
  ∃ (x₀ : ℚ), (P = (x₀, x₀ + a)) ∧ (P = (x₀, x₀^3 - x₀^2 + 1)) ∧ (3*x₀^2 - 2*x₀ = 1)

theorem find_a_and_tangent_point :
  ∃ (a : ℚ) (P : ℚ × ℚ), tangent_line_and_curve a P ∧ a = 32/27 ∧ P = (-1/3, 23/27) :=
sorry

end find_a_and_tangent_point_l76_76285


namespace weight_of_2019_is_correct_l76_76439

-- Declare the conditions as definitions to be used in Lean 4
def stick_weight : Real := 0.5
def digit_to_sticks (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- other digits aren't considered in this problem

-- Calculate the total weight of the number 2019
def weight_of_2019 : Real :=
  (digit_to_sticks 2 + digit_to_sticks 0 + digit_to_sticks 1 + digit_to_sticks 9) * stick_weight

-- Statement to prove the weight of the number 2019
theorem weight_of_2019_is_correct : weight_of_2019 = 9.5 := by
  sorry

end weight_of_2019_is_correct_l76_76439


namespace jenna_reading_pages_l76_76949

theorem jenna_reading_pages :
  ∀ (total_pages goal_pages flight_pages busy_days total_days reading_days : ℕ),
    total_days = 30 →
    busy_days = 4 →
    flight_pages = 100 →
    goal_pages = 600 →
    reading_days = total_days - busy_days - 1 →
    (goal_pages - flight_pages) / reading_days = 20 :=
by
  intros total_pages goal_pages flight_pages busy_days total_days reading_days
  sorry

end jenna_reading_pages_l76_76949


namespace remainder_of_power_mod_l76_76422

theorem remainder_of_power_mod (a b n : ℕ) (h_prime : Nat.Prime n) (h_a_not_div : ¬ (n ∣ a)) :
  a ^ b % n = 82 :=
by
  have : n = 379 := sorry
  have : a = 6 := sorry
  have : b = 97 := sorry
  sorry

end remainder_of_power_mod_l76_76422


namespace range_of_a_l76_76208

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_of_a_l76_76208


namespace not_all_inequalities_true_l76_76102

theorem not_all_inequalities_true (a b c : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 0 < b ∧ b < 1) (h₂ : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
  sorry

end not_all_inequalities_true_l76_76102


namespace isosceles_triangle_smallest_angle_l76_76081

def is_isosceles (angle_A angle_B angle_C : ℝ) : Prop := 
(angle_A = angle_B) ∨ (angle_B = angle_C) ∨ (angle_C = angle_A)

theorem isosceles_triangle_smallest_angle
  (angle_A angle_B angle_C : ℝ)
  (h_isosceles : is_isosceles angle_A angle_B angle_C)
  (h_angle_162 : angle_A = 162) :
  angle_B = 9 ∧ angle_C = 9 ∨ angle_A = 9 ∧ (angle_B = 9 ∨ angle_C = 9) :=
by
  sorry

end isosceles_triangle_smallest_angle_l76_76081


namespace differential_savings_l76_76244

-- Defining conditions given in the problem
def initial_tax_rate : ℝ := 0.45
def new_tax_rate : ℝ := 0.30
def annual_income : ℝ := 48000

-- Statement of the theorem to prove the differential savings
theorem differential_savings : (annual_income * initial_tax_rate) - (annual_income * new_tax_rate) = 7200 := by
  sorry  -- providing the proof is not required

end differential_savings_l76_76244


namespace fraction_simplify_l76_76348

theorem fraction_simplify :
  (3 + 9 - 27 + 81 - 243 + 729) / (9 + 27 - 81 + 243 - 729 + 2187) = 1 / 3 :=
by
  sorry

end fraction_simplify_l76_76348


namespace typing_problem_l76_76654

theorem typing_problem (a b m n : ℕ) (h1 : 60 = a * b) (h2 : 540 = 75 * n) (h3 : n = 3 * m) :
  a = 25 :=
by {
  -- sorry placeholder where the proof would go
  sorry
}

end typing_problem_l76_76654


namespace evaluate_expression_at_values_l76_76886

theorem evaluate_expression_at_values :
  let x := 2
  let y := -1
  let z := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
    sorry

end evaluate_expression_at_values_l76_76886


namespace shpuntik_can_form_triangle_l76_76805

theorem shpuntik_can_form_triangle 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (hx : x1 + x2 + x3 = 1)
  (hy : y1 + y2 + y3 = 1)
  (infeasibility_vintik : x1 ≥ x2 + x3) :
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a < b + c ∧ b < a + c ∧ c < a + b :=
sorry

end shpuntik_can_form_triangle_l76_76805


namespace min_value_polynomial_l76_76175

theorem min_value_polynomial (a b : ℝ) : 
  ∃ c, (∀ a b, c ≤ a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) ∧
       (∀ a b, a = -1 ∧ b = -1 → c = a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) :=
sorry

end min_value_polynomial_l76_76175


namespace lowest_temperature_in_january_2023_l76_76599

theorem lowest_temperature_in_january_2023 
  (T_Beijing T_Shanghai T_Shenzhen T_Jilin : ℝ)
  (h_Beijing : T_Beijing = -5)
  (h_Shanghai : T_Shanghai = 6)
  (h_Shenzhen : T_Shenzhen = 19)
  (h_Jilin : T_Jilin = -22) :
  T_Jilin < T_Beijing ∧ T_Jilin < T_Shanghai ∧ T_Jilin < T_Shenzhen :=
by
  sorry

end lowest_temperature_in_january_2023_l76_76599


namespace inequality_division_l76_76325

variable (m n : ℝ)

theorem inequality_division (h : m > n) : (m / 4) > (n / 4) :=
sorry

end inequality_division_l76_76325


namespace vasya_days_without_purchase_l76_76419

variables (x y z w : ℕ)

-- Given conditions as assumptions
def total_days : Prop := x + y + z + w = 15
def total_marshmallows : Prop := 9 * x + 4 * z = 30
def total_meat_pies : Prop := 2 * y + z = 9

-- Prove w = 7
theorem vasya_days_without_purchase (h1 : total_days x y z w) 
                                     (h2 : total_marshmallows x z) 
                                     (h3 : total_meat_pies y z) : 
  w = 7 :=
by
  -- Code placeholder to satisfy the theorem's syntax
  sorry

end vasya_days_without_purchase_l76_76419


namespace persons_in_boat_l76_76735

theorem persons_in_boat (W1 W2 new_person_weight : ℝ) (n : ℕ)
  (hW1 : W1 = 55)
  (h_new_person : new_person_weight = 50)
  (hW2 : W2 = W1 - 5) :
  (n * W1 + new_person_weight) / (n + 1) = W2 → false :=
by
  intros h_eq
  sorry

end persons_in_boat_l76_76735


namespace find_divisor_l76_76451

theorem find_divisor 
    (x : ℕ) 
    (h : 83 = 9 * x + 2) : 
    x = 9 := 
  sorry

end find_divisor_l76_76451


namespace mean_temperature_l76_76312

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem mean_temperature (temps : List ℝ) (length_temps_10 : temps.length = 10)
    (temps_vals : temps = [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]) : 
    mean temps = 88.2 := by
  sorry

end mean_temperature_l76_76312


namespace profit_percent_300_l76_76155

theorem profit_percent_300 (SP : ℝ) (CP : ℝ) (h : CP = 0.25 * SP) : ((SP - CP) / CP) * 100 = 300 :=
by
  sorry

end profit_percent_300_l76_76155


namespace buses_needed_40_buses_needed_30_l76_76194

-- Define the number of students
def number_of_students : ℕ := 186

-- Define the function to calculate minimum buses needed
def min_buses_needed (n : ℕ) : ℕ := (number_of_students + n - 1) / n

-- Theorem statements for the specific cases
theorem buses_needed_40 : min_buses_needed 40 = 5 := 
by 
  sorry

theorem buses_needed_30 : min_buses_needed 30 = 7 := 
by 
  sorry

end buses_needed_40_buses_needed_30_l76_76194


namespace parallel_lines_l76_76686

theorem parallel_lines (a : ℝ) : (2 * a = a * (a + 4)) → a = -2 :=
by
  intro h
  sorry

end parallel_lines_l76_76686


namespace problem_part_I_problem_part_II_l76_76392

-- Define the problem and the proof requirements in Lean 4
theorem problem_part_I (a b c : ℝ) (A B C : ℝ) (sinB_nonneg : 0 ≤ Real.sin B) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) 
(h_a : a = 2) (h_b : b = 2) : 
Real.cos B = 1/4 :=
sorry

theorem problem_part_II (a b c : ℝ) (A B C : ℝ) (h_B : B = π / 2) 
(h_a : a = Real.sqrt 2) 
(sinB_squared : Real.sin B ^ 2 = 2 * Real.sin A * Real.sin C) :
1/2 * a * c = 1 :=
sorry

end problem_part_I_problem_part_II_l76_76392


namespace measure_of_angle_C_maximum_area_of_triangle_l76_76321

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

theorem measure_of_angle_C :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 :=
by
  intros A B C a b c h
  sorry

theorem maximum_area_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 →
  1 / 2 * a * b * Real.sin C = (Real.sqrt 2 / 2 + 1 / 2) :=
by
  intros A B C a b c h hC
  sorry

end measure_of_angle_C_maximum_area_of_triangle_l76_76321


namespace velocity_division_l76_76651

/--
Given a trapezoidal velocity-time graph with bases V and U,
determine the velocity W that divides the area under the graph into
two regions such that the areas are in the ratio 1:k.
-/
theorem velocity_division (V U k : ℝ) (h_k : k ≠ -1) : 
  ∃ W : ℝ, W = (V^2 + k * U^2) / (k + 1) :=
by
  sorry

end velocity_division_l76_76651


namespace prove_percent_liquid_X_in_new_solution_l76_76255

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end prove_percent_liquid_X_in_new_solution_l76_76255


namespace problem_l76_76195

theorem problem (a : ℝ) (h : a^2 - 5 * a - 1 = 0) : 3 * a^2 - 15 * a = 3 :=
by
  sorry

end problem_l76_76195


namespace income_ratio_l76_76811

theorem income_ratio (I1 I2 E1 E2 : ℕ) (h1 : I1 = 5000) (h2 : E1 / E2 = 3 / 2) (h3 : I1 - E1 = 2000) (h4 : I2 - E2 = 2000) : I1 / I2 = 5 / 4 :=
by
  /- Proof omitted -/
  sorry

end income_ratio_l76_76811


namespace two_abs_inequality_l76_76398

theorem two_abs_inequality (x y : ℝ) :
  2 * abs (x + y) ≤ abs x + abs y ↔ 
  (x ≥ 0 ∧ -3 * x ≤ y ∧ y ≤ -x / 3) ∨ 
  (x < 0 ∧ -x / 3 ≤ y ∧ y ≤ -3 * x) :=
by
  sorry

end two_abs_inequality_l76_76398


namespace min_value_of_expression_l76_76106

noncomputable def f (m : ℝ) : ℝ :=
  let x1 := -m - (m^2 + 3 * m - 2)
  let x2 := -2 * m - x1
  x1 * (x2 + x1) + x2^2

theorem min_value_of_expression :
  ∃ m : ℝ, f m = 3 * (m - 1/2)^2 + 5/4 ∧ f m ≥ f (1/2) := by
  sorry

end min_value_of_expression_l76_76106


namespace algae_coverage_day_21_l76_76186

-- Let "algae_coverage n" denote the percentage of lake covered by algae on day n.
noncomputable def algaeCoverage : ℕ → ℝ
| 0 => 1 -- initial state on day 0 taken as baseline (can be adjusted accordingly)
| (n+1) => 2 * algaeCoverage n

-- Define the problem statement
theorem algae_coverage_day_21 :
  algaeCoverage 24 = 100 → algaeCoverage 21 = 12.5 :=
by
  sorry

end algae_coverage_day_21_l76_76186


namespace parallelogram_construction_l76_76711

theorem parallelogram_construction 
  (α : ℝ) (hα : 0 ≤ α ∧ α < 180)
  (A B : (ℝ × ℝ))
  (in_angle : (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ α ∧ 
               ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ α))
  (C D : (ℝ × ℝ)) :
  ∃ O : (ℝ × ℝ), 
    O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) :=
sorry

end parallelogram_construction_l76_76711


namespace arithmetic_sequence_a9_l76_76873

noncomputable def S (n : ℕ) (a₁ aₙ : ℝ) : ℝ := (n * (a₁ + aₙ)) / 2

theorem arithmetic_sequence_a9 (a₁ a₁₇ : ℝ) (h1 : S 17 a₁ a₁₇ = 102) : (a₁ + a₁₇) / 2 = 6 :=
by
  sorry

end arithmetic_sequence_a9_l76_76873


namespace min_value_fraction_l76_76854

noncomputable section

open Real

theorem min_value_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 4) : 
  ∃ t : ℝ, (∀ x' y' : ℝ, (x' > 0 ∧ y' > 0 ∧ x' + 2 * y' = 4) → (2 / x' + 1 / y') ≥ t) ∧ t = 2 :=
by
  sorry

end min_value_fraction_l76_76854


namespace product_possible_values_l76_76914

theorem product_possible_values (N L M M_5: ℤ) :
  M = L + N → 
  M_5 = M - 8 → 
  ∃ L_5, L_5 = L + 5 ∧ |M_5 - L_5| = 6 →
  N = 19 ∨ N = 7 → 19 * 7 = 133 :=
by {
  sorry
}

end product_possible_values_l76_76914


namespace find_number_l76_76538

theorem find_number (x : ℤ) (h : 33 + 3 * x = 48) : x = 5 :=
by
  sorry

end find_number_l76_76538


namespace bicycle_cost_price_l76_76408

-- Definitions of conditions
def profit_22_5_percent (x : ℝ) : ℝ := 1.225 * x
def loss_14_3_percent (x : ℝ) : ℝ := 0.857 * x
def profit_32_4_percent (x : ℝ) : ℝ := 1.324 * x
def loss_7_8_percent (x : ℝ) : ℝ := 0.922 * x
def discount_5_percent (x : ℝ) : ℝ := 0.95 * x
def tax_6_percent (x : ℝ) : ℝ := 1.06 * x

theorem bicycle_cost_price (CP_A : ℝ) (TP_E : ℝ) (h : TP_E = 295.88) : 
  CP_A = 295.88 / 1.29058890594 :=
by
  sorry

end bicycle_cost_price_l76_76408


namespace ratio_of_women_to_men_l76_76509

theorem ratio_of_women_to_men (M W : ℕ) 
  (h1 : M + W = 72) 
  (h2 : M - 16 = W + 8) : 
  W / M = 1 / 2 :=
sorry

end ratio_of_women_to_men_l76_76509


namespace evaluate_expression_l76_76771

/- The mathematical statement to prove:

Evaluate the expression 2/10 + 4/20 + 6/30, then multiply the result by 3
and show that it equals to 9/5.
-/

theorem evaluate_expression : 
  (2 / 10 + 4 / 20 + 6 / 30) * 3 = 9 / 5 := 
by 
  sorry

end evaluate_expression_l76_76771


namespace z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l76_76026

open Complex

-- Problem definitions
def z (m : ℝ) : ℂ := (2 + I) * m^2 - 2 * (1 - I)

-- Prove that for all m in ℝ, z is imaginary
theorem z_is_imaginary (m : ℝ) : ∃ a : ℝ, z m = a * I :=
  sorry

-- Prove that z is purely imaginary iff m = ±1
theorem z_is_purely_imaginary_iff (m : ℝ) : (∃ b : ℝ, z m = b * I ∧ b ≠ 0) ↔ (m = 1 ∨ m = -1) :=
  sorry

-- Prove that z is on the angle bisector iff m = 0
theorem z_on_angle_bisector_iff (m : ℝ) : (z m).re = -((z m).im) ↔ (m = 0) :=
  sorry

end z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l76_76026


namespace red_other_side_probability_is_one_l76_76018

/-- Definitions from the problem conditions --/
def total_cards : ℕ := 10
def green_both_sides : ℕ := 5
def green_red_sides : ℕ := 2
def red_both_sides : ℕ := 3
def red_faces : ℕ := 6 -- 3 cards × 2 sides each

/-- The theorem proves the probability is 1 that the other side is red given that one side seen is red --/
theorem red_other_side_probability_is_one
  (h_total_cards : total_cards = 10)
  (h_green_both : green_both_sides = 5)
  (h_green_red : green_red_sides = 2)
  (h_red_both : red_both_sides = 3)
  (h_red_faces : red_faces = 6) :
  1 = (red_faces / red_faces) :=
by
  -- Write the proof steps here
  sorry

end red_other_side_probability_is_one_l76_76018


namespace find_a10_of_arithmetic_sequence_l76_76139

theorem find_a10_of_arithmetic_sequence (a : ℕ → ℚ)
  (h_seq : ∀ n : ℕ, ∃ d : ℚ, ∀ m : ℕ, a (n + m + 1) = a (n + m) + d)
  (h_a1 : a 1 = 1)
  (h_a4 : a 4 = 4) :
  a 10 = -4 / 5 :=
sorry

end find_a10_of_arithmetic_sequence_l76_76139


namespace find_f_100_l76_76271

theorem find_f_100 (f : ℝ → ℝ) (k : ℝ) (h_nonzero : k ≠ 0) 
(h_func : ∀ x y : ℝ, 0 < x → 0 < y → k * (x * f y - y * f x) = f (x / y)) : 
f 100 = 0 := 
by
  sorry

end find_f_100_l76_76271


namespace leaves_decrease_by_four_fold_l76_76196

theorem leaves_decrease_by_four_fold (x y : ℝ) (h1 : y ≤ x / 4) : 
  9 * y ≤ (9 * x) / 4 := by 
  sorry

end leaves_decrease_by_four_fold_l76_76196


namespace problem_l76_76075

variable (a : Int)
variable (h : -a = 1)

theorem problem : 3 * a - 2 = -5 :=
by
  -- Proof will go here
  sorry

end problem_l76_76075


namespace common_points_line_circle_l76_76676

theorem common_points_line_circle (a b : ℝ) :
    (∃ x y : ℝ, x / a + y / b = 1 ∧ x^2 + y^2 = 1) →
    (1 / (a * a) + 1 / (b * b) ≥ 1) :=
by
  sorry

end common_points_line_circle_l76_76676


namespace polynomial_coefficients_sum_l76_76239

theorem polynomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ), 
  (∀ x : ℚ, (3 * x - 2)^9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 +
                            a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + 
                            a_7 * x^7 + a_8 * x^8 + a_9 * x^9) →
  (a_0 = -512) →
  ((a_0 + a_1 * (1/3) + a_2 * (1/3)^2 + a_3 * (1/3)^3 + 
    a_4 * (1/3)^4 + a_5 * (1/3)^5 + a_6 * (1/3)^6 + 
    a_7 * (1/3)^7 + a_8 * (1/3)^8 + a_9 * (1/3)^9) = -1) →
  (a_1 / 3 + a_2 / 3^2 + a_3 / 3^3 + a_4 / 3^4 + a_5 / 3^5 + 
   a_6 / 3^6 + a_7 / 3^7 + a_8 / 3^8 + a_9 / 3^9 = 511) :=
by 
  -- The proof would go here
  sorry

end polynomial_coefficients_sum_l76_76239


namespace anne_cleaning_time_l76_76696

theorem anne_cleaning_time :
  ∃ (A B : ℝ), (4 * (B + A) = 1) ∧ (3 * (B + 2 * A) = 1) ∧ (1 / A = 12) :=
by
  sorry

end anne_cleaning_time_l76_76696


namespace total_area_of_room_l76_76370

theorem total_area_of_room : 
  let length_rect := 8 
  let width_rect := 6 
  let base_triangle := 6 
  let height_triangle := 3 
  let area_rect := length_rect * width_rect 
  let area_triangle := (1 / 2 : ℝ) * base_triangle * height_triangle 
  let total_area := area_rect + area_triangle 
  total_area = 57 := 
by 
  sorry

end total_area_of_room_l76_76370


namespace expression_evaluation_l76_76431

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end expression_evaluation_l76_76431


namespace equations_have_same_solution_l76_76436

theorem equations_have_same_solution (x c : ℝ) 
  (h1 : 3 * x + 9 = 0) (h2 : c * x + 15 = 3) : c = 4 :=
by
  sorry

end equations_have_same_solution_l76_76436


namespace problem_equiv_l76_76169

variable (a b c d e f : ℝ)

theorem problem_equiv :
  a * b * c = 65 → 
  b * c * d = 65 → 
  c * d * e = 1000 → 
  d * e * f = 250 → 
  (a * f) / (c * d) = 1 / 4 := 
by 
  intros h1 h2 h3 h4
  sorry

end problem_equiv_l76_76169


namespace cost_function_discrete_points_l76_76471

def cost (n : ℕ) : ℕ :=
  if n <= 10 then 20 * n
  else if n <= 25 then 18 * n
  else 0

theorem cost_function_discrete_points :
  (∀ n, 1 ≤ n ∧ n ≤ 25 → ∃ y, cost n = y) ∧
  (∀ m n, 1 ≤ m ∧ m ≤ 25 ∧ 1 ≤ n ∧ n ≤ 25 ∧ m ≠ n → cost m ≠ cost n) :=
sorry

end cost_function_discrete_points_l76_76471


namespace starting_player_ensures_non_trivial_solution_l76_76527

theorem starting_player_ensures_non_trivial_solution :
  ∀ (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℚ), 
    ∃ (x y z : ℚ), 
    ((a1 * x + b1 * y + c1 * z = 0) ∧ 
     (a2 * x + b2 * y + c2 * z = 0) ∧ 
     (a3 * x + b3 * y + c3 * z = 0)) 
    ∧ ((a1 * (b2 * c3 - b3 * c2) - b1 * (a2 * c3 - a3 * c2) + c1 * (a2 * b3 - a3 * b2) = 0) ∧ 
         (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) :=
by
  intros a1 b1 c1 a2 b2 c2 a3 b3 c3
  sorry

end starting_player_ensures_non_trivial_solution_l76_76527


namespace range_of_k_l76_76302

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) → (k < 1 ∨ k > 3) := 
by 
  sorry

end range_of_k_l76_76302


namespace percentage_of_useful_items_l76_76924

theorem percentage_of_useful_items
  (junk_percentage : ℚ)
  (useful_items junk_items total_items : ℕ)
  (h1 : junk_percentage = 0.70)
  (h2 : useful_items = 8)
  (h3 : junk_items = 28)
  (h4 : junk_percentage * total_items = junk_items) :
  (useful_items : ℚ) / (total_items : ℚ) * 100 = 20 :=
sorry

end percentage_of_useful_items_l76_76924


namespace complex_magnitude_sixth_power_l76_76867

noncomputable def z := (2 : ℂ) + (2 * Real.sqrt 3) * Complex.I

theorem complex_magnitude_sixth_power :
  Complex.abs (z^6) = 4096 := 
by
  sorry

end complex_magnitude_sixth_power_l76_76867


namespace initial_hours_per_day_l76_76932

/-- 
Given:
1. 18 men working a certain number of hours per day dig 30 meters deep.
2. To dig to a depth of 50 meters, working 6 hours per day, 22 extra men should be put to work (total of 40 men).

Prove:
The initial 18 men were working \(\frac{200}{9}\) hours per day.
-/
theorem initial_hours_per_day 
  (h : ℚ)
  (work_done_18_men : 18 * h * 30 = 40 * 6 * 50) :
  h = 200 / 9 :=
by
  sorry

end initial_hours_per_day_l76_76932


namespace expression_is_integer_iff_divisible_l76_76414

theorem expression_is_integer_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, n = m * (k + 2) ↔ (∃ C : ℤ, (3 * n - 4 * k + 2) / (k + 2) * C = (3 * n - 4 * k + 2) / (k + 2)) :=
sorry

end expression_is_integer_iff_divisible_l76_76414


namespace quadratic_has_two_equal_real_roots_l76_76388

theorem quadratic_has_two_equal_real_roots : ∃ c : ℝ, ∀ x : ℝ, (x^2 - 6*x + c = 0 ↔ (x = 3)) :=
by
  sorry

end quadratic_has_two_equal_real_roots_l76_76388


namespace complete_laps_l76_76665

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l76_76665


namespace marie_profit_l76_76245

-- Define constants and conditions
def loaves_baked : ℕ := 60
def morning_price : ℝ := 3.00
def discount : ℝ := 0.25
def afternoon_price : ℝ := morning_price * (1 - discount)
def cost_per_loaf : ℝ := 1.00
def donated_loaves : ℕ := 5

-- Define the number of loaves sold and revenue
def morning_loaves : ℕ := loaves_baked / 3
def morning_revenue : ℝ := morning_loaves * morning_price

def remaining_after_morning : ℕ := loaves_baked - morning_loaves
def afternoon_loaves : ℕ := remaining_after_morning / 2
def afternoon_revenue : ℝ := afternoon_loaves * afternoon_price

def remaining_after_afternoon : ℕ := remaining_after_morning - afternoon_loaves
def unsold_loaves : ℕ := remaining_after_afternoon - donated_loaves

-- Define the total revenue and cost
def total_revenue : ℝ := morning_revenue + afternoon_revenue
def total_cost : ℝ := loaves_baked * cost_per_loaf

-- Define the profit
def profit : ℝ := total_revenue - total_cost

-- State the proof problem
theorem marie_profit : profit = 45 := by
  sorry

end marie_profit_l76_76245


namespace prove_values_of_a_l76_76741

-- Definitions of the conditions
def condition_1 (a x y : ℝ) : Prop := (x * y)^(1/3) = a^(a^2)
def condition_2 (a x y : ℝ) : Prop := (Real.log x / Real.log a * Real.log y / Real.log a) + (Real.log y / Real.log a * Real.log x / Real.log a) = 3 * a^3

-- The proof problem
theorem prove_values_of_a (a x y : ℝ) (h1 : condition_1 a x y) (h2 : condition_2 a x y) : a > 0 ∧ a ≤ 2/3 :=
sorry

end prove_values_of_a_l76_76741


namespace tod_driving_time_l76_76410
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l76_76410


namespace area_times_breadth_l76_76559

theorem area_times_breadth (b l A : ℕ) (h1 : b = 11) (h2 : l - b = 10) (h3 : A = l * b) : A / b = 21 := 
by
  sorry

end area_times_breadth_l76_76559


namespace right_triangle_relation_l76_76237

theorem right_triangle_relation (a b c x : ℝ)
  (h : c^2 = a^2 + b^2)
  (altitude : a * b = c * x) :
  (1 / x^2) = (1 / a^2) + (1 / b^2) :=
sorry

end right_triangle_relation_l76_76237


namespace product_mod_m_l76_76646

-- Define the constants
def a : ℕ := 2345
def b : ℕ := 1554
def m : ℕ := 700

-- Definitions derived from the conditions
def a_mod_m : ℕ := a % m
def b_mod_m : ℕ := b % m

-- The proof problem
theorem product_mod_m (a b m : ℕ) (h1 : a % m = 245) (h2 : b % m = 154) :
  (a * b) % m = 630 := by sorry

end product_mod_m_l76_76646


namespace andy_wrong_questions_l76_76761

theorem andy_wrong_questions
  (a b c d : ℕ)
  (h1 : a + b = c + d + 6)
  (h2 : a + d = b + c + 4)
  (h3 : c = 10) :
  a = 15 :=
by
  sorry

end andy_wrong_questions_l76_76761


namespace no_couples_next_to_each_other_l76_76691

def factorial (n: Nat): Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (m n p q: Nat): Nat :=
  factorial m - n * factorial (m - 1) + p * factorial (m - 2) - q * factorial (m - 3)

theorem no_couples_next_to_each_other :
  arrangements 7 8 24 32 + 16 * factorial 3 = 1488 :=
by
  -- Here we state that the calculation of special arrangements equals 1488.
  sorry

end no_couples_next_to_each_other_l76_76691


namespace parabola_directrix_l76_76033

theorem parabola_directrix (y : ℝ) (x : ℝ) (h : y = 8 * x^2) : 
  y = -1 / 32 :=
sorry

end parabola_directrix_l76_76033


namespace sequence_is_constant_l76_76826

noncomputable def sequence_condition (a : ℕ → ℝ) :=
  a 1 = 1 ∧ ∀ m n : ℕ, m > 0 → n > 0 → |a n - a m| ≤ 2 * m * n / (m ^ 2 + n ^ 2)

theorem sequence_is_constant (a : ℕ → ℝ) 
  (h : sequence_condition a) :
  ∀ n : ℕ, n > 0 → a n = 1 :=
by
  sorry

end sequence_is_constant_l76_76826


namespace value_of_m_l76_76871

theorem value_of_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 8) (h2 : x1 = 3 * x2) : m = 12 :=
by
  -- Proof will be provided here
  sorry

end value_of_m_l76_76871


namespace bakery_profit_l76_76478

noncomputable def revenue_per_piece : ℝ := 4
noncomputable def pieces_per_pie : ℕ := 3
noncomputable def pies_per_hour : ℕ := 12
noncomputable def cost_per_pie : ℝ := 0.5

theorem bakery_profit (pieces_per_pie_pos : 0 < pieces_per_pie) 
                      (pies_per_hour_pos : 0 < pies_per_hour) 
                      (cost_per_pie_pos : 0 < cost_per_pie) :
  pies_per_hour * (pieces_per_pie * revenue_per_piece) - (pies_per_hour * cost_per_pie) = 138 := 
sorry

end bakery_profit_l76_76478


namespace union_complement_equals_set_l76_76301

universe u

variable {I A B : Set ℕ}

def universal_set : Set ℕ := {0, 1, 2, 3, 4}
def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {2, 3, 4}
def complement_B : Set ℕ := { x ∈ universal_set | x ∉ set_B }

theorem union_complement_equals_set :
  set_A ∪ complement_B = {0, 1, 2} := by
  sorry

end union_complement_equals_set_l76_76301


namespace Brian_Frodo_ratio_l76_76264

-- Definitions from the conditions
def Lily_tennis_balls : Int := 3
def Frodo_tennis_balls : Int := Lily_tennis_balls + 8
def Brian_tennis_balls : Int := 22

-- The proof statement
theorem Brian_Frodo_ratio :
  Brian_tennis_balls / Frodo_tennis_balls = 2 := by
  sorry

end Brian_Frodo_ratio_l76_76264


namespace find_acute_angle_l76_76958

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, 2)
noncomputable def vector_b (α : ℝ) : ℝ × ℝ := (3, 4 * Real.sin α)
def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem find_acute_angle (α : ℝ) (h : are_parallel (vector_a α) (vector_b α)) (h_acute : 0 < α ∧ α < π / 2) : 
  α = π / 4 :=
by
  sorry

end find_acute_angle_l76_76958


namespace triangle_properties_l76_76792

theorem triangle_properties :
  (∀ (α β γ : ℝ), α + β + γ = 180 → 
    (α = β ∨ α = γ ∨ β = γ ∨ 
     (α = 60 ∧ β = 60 ∧ γ = 60) ∨
     ¬(α = 90 ∧ β = 90))) :=
by
  -- Placeholder for the actual proof, ensuring the theorem can build
  intros α β γ h₁
  sorry

end triangle_properties_l76_76792


namespace number_of_B_students_l76_76807

/- Define the assumptions of the problem -/
variable (x : ℝ)  -- the number of students who earn a B

/- Express the number of students getting each grade in terms of x -/
def number_of_A (x : ℝ) := 0.6 * x
def number_of_C (x : ℝ) := 1.3 * x
def number_of_D (x : ℝ) := 0.8 * x
def total_students (x : ℝ) := number_of_A x + x + number_of_C x + number_of_D x

/- Prove that x = 14 for the total number of students being 50 -/
theorem number_of_B_students : total_students x = 50 → x = 14 :=
by 
  sorry

end number_of_B_students_l76_76807


namespace common_factor_l76_76336

theorem common_factor (x y a b : ℤ) : 
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) :=
by {
  sorry
}

end common_factor_l76_76336


namespace exam_maximum_marks_l76_76978

theorem exam_maximum_marks :
  (∃ M S E : ℕ, 
    (90 + 20 = 40 * M / 100) ∧ 
    (110 + 35 = 35 * S / 100) ∧ 
    (80 + 10 = 30 * E / 100) ∧ 
    M = 275 ∧ 
    S = 414 ∧ 
    E = 300) :=
by
  sorry

end exam_maximum_marks_l76_76978


namespace gcd_1617_1225_gcd_2023_111_gcd_589_6479_l76_76057

theorem gcd_1617_1225 : Nat.gcd 1617 1225 = 49 :=
by
  sorry

theorem gcd_2023_111 : Nat.gcd 2023 111 = 1 :=
by
  sorry

theorem gcd_589_6479 : Nat.gcd 589 6479 = 589 :=
by
  sorry

end gcd_1617_1225_gcd_2023_111_gcd_589_6479_l76_76057


namespace probability_is_zero_l76_76423

noncomputable def probability_same_number (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : ℝ :=
  0

theorem probability_is_zero (b d : ℕ) (h_b : b < 150) (h_d : d < 150)
    (h_b_multiple: b % 15 = 0) (h_d_multiple: d % 20 = 0) (h_square: b * b = b ∨ d * d = d) : 
    probability_same_number b d h_b h_d h_b_multiple h_d_multiple h_square = 0 :=
  sorry

end probability_is_zero_l76_76423


namespace smallest_n_for_at_least_64_candies_l76_76650

theorem smallest_n_for_at_least_64_candies :
  ∃ n : ℕ, (n > 0) ∧ (n * (n + 1) / 2 ≥ 64) ∧ (∀ m : ℕ, (m > 0) ∧ (m * (m + 1) / 2 ≥ 64) → n ≤ m) := 
sorry

end smallest_n_for_at_least_64_candies_l76_76650


namespace find_a_b_l76_76980

-- Conditions defining the solution sets A and B
def A : Set ℝ := { x | -1 < x ∧ x < 3 }
def B : Set ℝ := { x | -3 < x ∧ x < 2 }

-- The solution set of the inequality x^2 + ax + b < 0 is the intersection A∩B
def C : Set ℝ := A ∩ B

-- Proving that there exist values of a and b such that the solution set C corresponds to the inequality x^2 + ax + b < 0
theorem find_a_b : ∃ a b : ℝ, (∀ x : ℝ, C x ↔ x^2 + a*x + b < 0) ∧ a + b = -3 := 
by 
  sorry

end find_a_b_l76_76980


namespace simplify_evaluate_expr_l76_76299

theorem simplify_evaluate_expr (x y : ℚ) (h₁ : x = -1) (h₂ : y = -1 / 2) :
  (4 * x * y + (2 * x^2 + 5 * x * y - y^2) - 2 * (x^2 + 3 * x * y)) = 5 / 4 :=
by
  rw [h₁, h₂]
  -- Here we would include the specific algebra steps to convert the LHS to 5/4.
  sorry

end simplify_evaluate_expr_l76_76299


namespace sum_of_numbers_ge_1_1_l76_76109

theorem sum_of_numbers_ge_1_1 :
  let numbers := [1.4, 0.9, 1.2, 0.5, 1.3]
  let threshold := 1.1
  let filtered_numbers := numbers.filter (fun x => x >= threshold)
  let sum_filtered := filtered_numbers.sum
  sum_filtered = 3.9 :=
by {
  sorry
}

end sum_of_numbers_ge_1_1_l76_76109


namespace arrange_magnitudes_l76_76427

theorem arrange_magnitudes (x : ℝ) (h1 : 0.85 < x) (h2 : x < 1.1)
  (y : ℝ := x + Real.sin x) (z : ℝ := x ^ (x ^ x)) : x < y ∧ y < z := 
sorry

end arrange_magnitudes_l76_76427


namespace paco_initial_sweet_cookies_l76_76028

theorem paco_initial_sweet_cookies (S : ℕ) (h1 : S - 15 = 7) : S = 22 :=
by
  sorry

end paco_initial_sweet_cookies_l76_76028


namespace largest_value_among_l76_76621

theorem largest_value_among (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hneq : a ≠ b) :
  max (a + b) (max (2 * Real.sqrt (a * b)) ((a^2 + b^2) / (2 * a * b))) = a + b :=
sorry

end largest_value_among_l76_76621


namespace probability_ace_king_queen_same_suit_l76_76294

theorem probability_ace_king_queen_same_suit :
  let total_probability := (1 : ℝ) / 52 * (1 : ℝ) / 51 * (1 : ℝ) / 50
  total_probability = (1 : ℝ) / 132600 :=
by
  sorry

end probability_ace_king_queen_same_suit_l76_76294


namespace meeting_at_centroid_l76_76463

theorem meeting_at_centroid :
  let A := (2, 9)
  let B := (-3, -4)
  let C := (6, -1)
  let centroid := ((2 - 3 + 6) / 3, (9 - 4 - 1) / 3)
  centroid = (5 / 3, 4 / 3) := sorry

end meeting_at_centroid_l76_76463


namespace car_drive_time_60_kmh_l76_76182

theorem car_drive_time_60_kmh
  (t : ℝ)
  (avg_speed : ℝ := 80)
  (dist_speed_60 : ℝ := 60 * t)
  (time_speed_90 : ℝ := 2 / 3)
  (dist_speed_90 : ℝ := 90 * time_speed_90)
  (total_distance : ℝ := dist_speed_60 + dist_speed_90)
  (total_time : ℝ := t + time_speed_90)
  (avg_speed_eq : avg_speed = total_distance / total_time) :
  t = 1 / 3 := 
sorry

end car_drive_time_60_kmh_l76_76182


namespace sum_of_money_l76_76752

theorem sum_of_money (J C P : ℕ) 
  (h1 : P = 60)
  (h2 : P = 3 * J)
  (h3 : C + 7 = 2 * J) : 
  J + P + C = 113 := 
by
  sorry

end sum_of_money_l76_76752


namespace mother_to_father_age_ratio_l76_76300

def DarcieAge : ℕ := 4
def FatherAge : ℕ := 30
def MotherAge : ℕ := DarcieAge * 6

theorem mother_to_father_age_ratio :
  (MotherAge : ℚ) / (FatherAge : ℚ) = (4 / 5) := by
  sorry

end mother_to_father_age_ratio_l76_76300


namespace selling_price_correct_l76_76220

theorem selling_price_correct (C P_rate : ℝ) (hC : C = 50) (hP_rate : P_rate = 0.40) : 
  C + (P_rate * C) = 70 :=
by
  sorry

end selling_price_correct_l76_76220


namespace smallest_n_integer_l76_76995

theorem smallest_n_integer (m n : ℕ) (s : ℝ) (h_m : m = (n + s)^4) (h_n_pos : 0 < n) (h_s_range : 0 < s ∧ s < 1 / 2000) : n = 8 := 
by
  sorry

end smallest_n_integer_l76_76995


namespace felix_brother_lifting_capacity_is_600_l76_76387

-- Define the conditions
def felix_lifting_capacity (felix_weight : ℝ) : ℝ := 1.5 * felix_weight
def felix_brother_weight (felix_weight : ℝ) : ℝ := 2 * felix_weight
def felix_brother_lifting_capacity (brother_weight : ℝ) : ℝ := 3 * brother_weight
def felix_actual_lifting_capacity : ℝ := 150

-- Define the proof problem
theorem felix_brother_lifting_capacity_is_600 :
  ∃ felix_weight : ℝ,
    felix_lifting_capacity felix_weight = felix_actual_lifting_capacity ∧
    felix_brother_lifting_capacity (felix_brother_weight felix_weight) = 600 :=
by
  sorry

end felix_brother_lifting_capacity_is_600_l76_76387


namespace maximize_expression_l76_76206

-- Given the condition
theorem maximize_expression (x y : ℝ) (h : x + y = 1) : (x^3 + 1) * (y^3 + 1) ≤ (1)^3 + 1 * (0)^3 + 1 * (0)^3 + 1 :=
sorry

end maximize_expression_l76_76206


namespace fourth_is_20_fewer_than_third_l76_76236

-- Definitions of the number of road signs at each intersection
def first_intersection := 40
def second_intersection := first_intersection + first_intersection / 4
def third_intersection := 2 * second_intersection
def total_signs := 270
def fourth_intersection := total_signs - (first_intersection + second_intersection + third_intersection)

-- Proving the fourth intersection has 20 fewer signs than the third intersection
theorem fourth_is_20_fewer_than_third : third_intersection - fourth_intersection = 20 :=
by
  -- This is a placeholder for the proof
  sorry

end fourth_is_20_fewer_than_third_l76_76236


namespace paintable_sum_l76_76406

theorem paintable_sum :
  ∃ (h t u v : ℕ), h > 0 ∧ t > 0 ∧ u > 0 ∧ v > 0 ∧
  (∀ k, k % h = 1 ∨ k % t = 2 ∨ k % u = 3 ∨ k % v = 4) ∧
  (∀ k k', k ≠ k' → (k % h ≠ k' % h ∧ k % t ≠ k' % t ∧ k % u ≠ k' % u ∧ k % v ≠ k' % v)) ∧
  1000 * h + 100 * t + 10 * u + v = 4536 :=
by
  sorry

end paintable_sum_l76_76406


namespace factor_81_minus_4y4_l76_76785

theorem factor_81_minus_4y4 (y : ℝ) : 81 - 4 * y^4 = (9 + 2 * y^2) * (9 - 2 * y^2) := by 
    sorry

end factor_81_minus_4y4_l76_76785


namespace sum_of_midpoint_coordinates_l76_76851

theorem sum_of_midpoint_coordinates 
  (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : (x1, y1, z1) = (2, 3, 4)) 
  (h2 : (x2, y2, z2) = (8, 15, 12)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 + (z1 + z2) / 2 = 22 := 
by
  sorry

end sum_of_midpoint_coordinates_l76_76851


namespace inequality_proof_l76_76843

variable (u v w : ℝ)

theorem inequality_proof (h1 : u > 0) (h2 : v > 0) (h3 : w > 0) (h4 : u + v + w + Real.sqrt (u * v * w) = 4) :
    Real.sqrt (u * v / w) + Real.sqrt (v * w / u) + Real.sqrt (w * u / v) ≥ u + v + w := 
  sorry

end inequality_proof_l76_76843


namespace village_foods_sales_l76_76628

-- Definitions based on conditions
def customer_count : Nat := 500
def lettuce_per_customer : Nat := 2
def tomato_per_customer : Nat := 4
def price_per_lettuce : Nat := 1
def price_per_tomato : Nat := 1 / 2 -- Note: Handling decimal requires careful type choice

-- Main statement to prove
theorem village_foods_sales : 
  customer_count * (lettuce_per_customer * price_per_lettuce + tomato_per_customer * price_per_tomato) = 2000 := 
by
  sorry

end village_foods_sales_l76_76628


namespace yanna_gave_100_l76_76872

/--
Yanna buys 10 shirts at $5 each and 3 pairs of sandals at $3 each, 
and she receives $41 in change. Prove that she gave $100.
-/
theorem yanna_gave_100 :
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  total_cost + change = 100 :=
by
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  show total_cost + change = 100
  sorry

end yanna_gave_100_l76_76872


namespace minimum_value_is_1297_l76_76882

noncomputable def find_minimum_value (a b c n : ℕ) : ℕ :=
  if (a + b ≠ b + c) ∧ (b + c ≠ c + a) ∧ (a + b ≠ c + a) ∧
     ((a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
      (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
      (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) then
    a^2 + b^2 + c^2
  else
    0

theorem minimum_value_is_1297 (a b c n : ℕ) :
  a ≠ b → b ≠ c → c ≠ a → (∃ a b c n, (a + b = n^2 ∧ b + c = (n + 1)^2 ∧ c + a = (n + 2)^2) ∨
                                  (a + b = (n + 1)^2 ∧ b + c = (n + 2)^2 ∧ c + a = n^2) ∨
                                  (a + b = (n + 2)^2 ∧ b + c = n^2 ∧ c + a = (n + 1)^2)) →
  (∃ a b c, a^2 + b^2 + c^2 = 1297) :=
by sorry

end minimum_value_is_1297_l76_76882


namespace yuna_candy_days_l76_76573

theorem yuna_candy_days (total_candies : ℕ) (daily_candies_week : ℕ) (days_week : ℕ) (remaining_candies : ℕ) (daily_candies_future : ℕ) :
  total_candies = 60 →
  daily_candies_week = 6 →
  days_week = 7 →
  remaining_candies = total_candies - (daily_candies_week * days_week) →
  daily_candies_future = 3 →
  remaining_candies / daily_candies_future = 6 :=
by
  intros h_total h_daily_week h_days_week h_remaining h_daily_future
  sorry

end yuna_candy_days_l76_76573


namespace arielle_age_l76_76596

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l76_76596


namespace arithmetic_progression_exists_l76_76516

theorem arithmetic_progression_exists (a_1 a_2 a_3 a_4 : ℕ) (d : ℕ) :
  a_2 = a_1 + d →
  a_3 = a_1 + 2 * d →
  a_4 = a_1 + 3 * d →
  a_1 * a_2 * a_3 = 6 →
  a_1 * a_2 * a_3 * a_4 = 24 →
  a_1 = 1 ∧ a_2 = 2 ∧ a_3 = 3 ∧ a_4 = 4 :=
by
  sorry

end arithmetic_progression_exists_l76_76516


namespace kelly_baking_powder_l76_76024

variable (current_supply : ℝ) (additional_supply : ℝ)

theorem kelly_baking_powder (h1 : current_supply = 0.3)
                            (h2 : additional_supply = 0.1) :
                            current_supply + additional_supply = 0.4 := 
by
  sorry

end kelly_baking_powder_l76_76024


namespace smallest_four_digits_valid_remainder_l76_76493

def isFourDigit (x : ℕ) : Prop := 1000 ≤ x ∧ x ≤ 9999 

def validRemainder (x : ℕ) : Prop := 
  ∀ k ∈ [2, 3, 4, 5, 6], x % k = 1

theorem smallest_four_digits_valid_remainder :
  ∃ x1 x2 x3 x4 : ℕ,
    isFourDigit x1 ∧ validRemainder x1 ∧
    isFourDigit x2 ∧ validRemainder x2 ∧
    isFourDigit x3 ∧ validRemainder x3 ∧
    isFourDigit x4 ∧ validRemainder x4 ∧
    x1 = 1021 ∧ x2 = 1081 ∧ x3 = 1141 ∧ x4 = 1201 := 
sorry

end smallest_four_digits_valid_remainder_l76_76493


namespace categorize_numbers_l76_76281

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers_l76_76281


namespace pugs_working_together_l76_76080

theorem pugs_working_together (P : ℕ) (H1 : P * 45 = 15 * 12) : P = 4 :=
by {
  sorry
}

end pugs_working_together_l76_76080


namespace interval_contains_integer_l76_76371

theorem interval_contains_integer (a : ℝ) : 
  (∃ n : ℤ, (3 * a < n) ∧ (n < 5 * a - 2)) ↔ (1.2 < a ∧ a < 4 / 3) ∨ (7 / 5 < a) :=
by sorry

end interval_contains_integer_l76_76371


namespace base9_num_digits_2500_l76_76395

theorem base9_num_digits_2500 : 
  ∀ (n : ℕ), (9^1 = 9) → (9^2 = 81) → (9^3 = 729) → (9^4 = 6561) → n = 4 := by
  sorry

end base9_num_digits_2500_l76_76395


namespace prove_equal_values_l76_76583

theorem prove_equal_values :
  (-2: ℝ)^3 = -(2: ℝ)^3 :=
by sorry

end prove_equal_values_l76_76583


namespace arithmetic_sequence_terms_l76_76716

theorem arithmetic_sequence_terms (a d n : ℕ) 
  (h_sum_first_3 : 3 * a + 3 * d = 34)
  (h_sum_last_3 : 3 * a + 3 * d * (n - 1) = 146)
  (h_sum_all : n * (2 * a + (n - 1) * d) = 2 * 390) : 
  n = 13 :=
by
  sorry

end arithmetic_sequence_terms_l76_76716


namespace print_time_is_fifteen_l76_76732

noncomputable def time_to_print (total_pages rate : ℕ) := 
  (total_pages : ℚ) / rate

theorem print_time_is_fifteen :
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  round time = 15 := by
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  have time_val : time = (350 : ℚ) / 24 := by rfl
  let rounded_time := round time
  have rounded_time_val : rounded_time = 15 := by sorry
  exact rounded_time_val

end print_time_is_fifteen_l76_76732


namespace possible_values_of_sum_l76_76311

theorem possible_values_of_sum
  (p q r : ℝ)
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_system : q = p * (4 - p) ∧ r = q * (4 - q) ∧ p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end possible_values_of_sum_l76_76311


namespace derivative_of_y_l76_76643

noncomputable def y (x : ℝ) : ℝ :=
  -1/4 * Real.arcsin ((5 + 3 * Real.cosh x) / (3 + 5 * Real.cosh x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (3 + 5 * Real.cosh x) :=
sorry

end derivative_of_y_l76_76643


namespace can_form_triangle_l76_76099

theorem can_form_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 10) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rw [h1, h2, h3]
  repeat {sorry}

end can_form_triangle_l76_76099


namespace points_in_quadrants_l76_76114

theorem points_in_quadrants (x y : ℝ) (h_line : 4 * x + 7 * y = 28)
  (h_equidistant : |x| = |y|) : 
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
sorry

end points_in_quadrants_l76_76114


namespace max_chord_length_l76_76835

noncomputable def family_of_curves (θ x y : ℝ) := 
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) := 2 * x = y

theorem max_chord_length :
  (∀ (θ : ℝ), ∀ (x y : ℝ), family_of_curves θ x y → line x y) → 
  ∃ (L : ℝ), L = 8 * Real.sqrt 5 :=
by
  sorry

end max_chord_length_l76_76835


namespace option_C_correct_l76_76929

theorem option_C_correct {a : ℝ} : a^2 * a^3 = a^5 := by
  -- Proof to be filled
  sorry

end option_C_correct_l76_76929


namespace universal_proposition_is_B_l76_76689

theorem universal_proposition_is_B :
  (∀ n : ℤ, (2 * n % 2 = 0)) = True :=
sorry

end universal_proposition_is_B_l76_76689


namespace find_small_pack_size_l76_76617

-- Define the conditions of the problem
def soymilk_sold_in_packs (pack_size : ℕ) : Prop :=
  pack_size = 2 ∨ ∃ L : ℕ, pack_size = L

def cartons_bought (total_cartons : ℕ) (large_pack_size : ℕ) (num_large_packs : ℕ) (small_pack_size : ℕ) : Prop :=
  total_cartons = num_large_packs * large_pack_size + small_pack_size

-- The problem statement as a Lean theorem
theorem find_small_pack_size (total_cartons : ℕ) (num_large_packs : ℕ) (large_pack_size : ℕ) :
  soymilk_sold_in_packs 2 →
  soymilk_sold_in_packs large_pack_size →
  cartons_bought total_cartons large_pack_size num_large_packs 2 →
  total_cartons = 17 →
  num_large_packs = 3 →
  large_pack_size = 5 →
  ∃ S : ℕ, soymilk_sold_in_packs S ∧ S = 2 :=
by
  sorry

end find_small_pack_size_l76_76617


namespace unique_intersection_point_l76_76303

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point : ∃ a : ℝ, f a = a ∧ f a = -1 ∧ f a = f⁻¹ a :=
by 
  sorry

end unique_intersection_point_l76_76303


namespace goods_train_length_is_280_meters_l76_76699

def speed_of_man_train_kmph : ℝ := 80
def speed_of_goods_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 9

theorem goods_train_length_is_280_meters :
  let relative_speed_kmph := speed_of_man_train_kmph + speed_of_goods_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let length_of_goods_train := relative_speed_mps * time_to_pass_seconds
  abs (length_of_goods_train - 280) < 1 :=
by
  -- skipping the proof
  sorry

end goods_train_length_is_280_meters_l76_76699


namespace calculate_group5_students_l76_76852

variable (total_students : ℕ) (freq_group1 : ℕ) (sum_freq_group2_3 : ℝ) (freq_group4 : ℝ)

theorem calculate_group5_students
  (h1 : total_students = 50)
  (h2 : freq_group1 = 7)
  (h3 : sum_freq_group2_3 = 0.46)
  (h4 : freq_group4 = 0.2) :
  (total_students * (1 - (freq_group1 / total_students + sum_freq_group2_3 + freq_group4)) = 10) :=
by
  sorry

end calculate_group5_students_l76_76852


namespace longer_diagonal_eq_l76_76205

variable (a b : ℝ)
variable (h_cd : CD = a) (h_bc : BC = b) (h_diag : AC = a) (h_ad : AD = 2 * b)

theorem longer_diagonal_eq (CD BC AC AD BD : ℝ) (h_cd : CD = a)
  (h_bc : BC = b) (h_diag : AC = CD) (h_ad : AD = 2 * b) :
  BD = Real.sqrt (a^2 + 3 * b^2) :=
sorry

end longer_diagonal_eq_l76_76205


namespace find_r_fourth_l76_76693

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l76_76693


namespace per_capita_income_ratio_l76_76631

theorem per_capita_income_ratio
  (PL_10 PZ_10 PL_now PZ_now : ℝ)
  (h1 : PZ_10 = 0.4 * PL_10)
  (h2 : PZ_now = 0.8 * PL_now)
  (h3 : PL_now = 3 * PL_10) :
  PZ_now / PZ_10 = 6 := by
  -- Proof to be filled
  sorry

end per_capita_income_ratio_l76_76631


namespace circles_tangent_l76_76935

theorem circles_tangent
  (rA rB rC rD rF : ℝ) (rE : ℚ) (m n : ℕ)
  (m_n_rel_prime : Int.gcd m n = 1)
  (rA_pos : 0 < rA) (rB_pos : 0 < rB)
  (rC_pos : 0 < rC) (rD_pos : 0 < rD)
  (rF_pos : 0 < rF)
  (inscribed_triangle_in_A : True)  -- Triangle T is inscribed in circle A
  (B_tangent_A : True)  -- Circle B is internally tangent to circle A
  (C_tangent_A : True)  -- Circle C is internally tangent to circle A
  (D_tangent_A : True)  -- Circle D is internally tangent to circle A
  (B_externally_tangent_E : True)  -- Circle B is externally tangent to circle E
  (C_externally_tangent_E : True)  -- Circle C is externally tangent to circle E
  (D_externally_tangent_E : True)  -- Circle D is externally tangent to circle E
  (F_tangent_A : True)  -- Circle F is internally tangent to circle A at midpoint of side opposite to B's tangency
  (F_externally_tangent_E : True)  -- Circle F is externally tangent to circle E
  (rA_eq : rA = 12) (rB_eq : rB = 5)
  (rC_eq : rC = 3) (rD_eq : rD = 2)
  (rF_eq : rF = 1)
  (rE_eq : rE = m / n)
  : m + n = 23 :=
by
  sorry

end circles_tangent_l76_76935


namespace nth_number_eq_l76_76890

noncomputable def nth_number (n : Nat) : ℚ := n / (n^2 + 1)

theorem nth_number_eq (n : Nat) : nth_number n = n / (n^2 + 1) :=
by
  sorry

end nth_number_eq_l76_76890


namespace line_intersects_parabola_at_vertex_l76_76659

theorem line_intersects_parabola_at_vertex :
  ∃ (a : ℝ), (∀ x : ℝ, -x + a = x^2 + a^2) ↔ a = 0 ∨ a = 1 :=
by
  sorry

end line_intersects_parabola_at_vertex_l76_76659


namespace calculate_final_number_l76_76567

theorem calculate_final_number (initial increment times : ℕ) (h₀ : initial = 540) (h₁ : increment = 10) (h₂ : times = 6) : initial + increment * times = 600 :=
by
  sorry

end calculate_final_number_l76_76567


namespace lily_distance_from_start_l76_76063

open Real

def north_south_net := 40 - 10 -- 30 meters south
def east_west_net := 30 - 15 -- 15 meters east

theorem lily_distance_from_start : 
  ∀ (north_south : ℝ) (east_west : ℝ), 
    north_south = north_south_net → 
    east_west = east_west_net → 
    distance = Real.sqrt ((north_south * north_south) + (east_west * east_west)) → 
    distance = 15 * Real.sqrt 5 :=
by
  intros
  sorry

end lily_distance_from_start_l76_76063


namespace polygon_sides_l76_76210

theorem polygon_sides (h : ∀ (n : ℕ), 360 / n = 36) : 10 = 10 := by
  sorry

end polygon_sides_l76_76210


namespace range_of_a_l76_76298

def P (x : ℝ) : Prop := x^2 ≤ 1

def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : ∀ x, (P x ∨ x = a) ↔ P x) : P a :=
by
  sorry

end range_of_a_l76_76298


namespace no_such_integers_and_function_l76_76227

theorem no_such_integers_and_function (f : ℝ → ℝ) (m n : ℤ) (h1 : ∀ x, f (f x) = 2 * f x - x - 2) (h2 : (m : ℝ) ≤ (n : ℝ) ∧ f m = n) : False :=
sorry

end no_such_integers_and_function_l76_76227


namespace ping_pong_ball_probability_l76_76547

noncomputable def multiple_of_6_9_or_both_probability : ℚ :=
  let total_numbers := 72
  let multiples_of_6 := 12
  let multiples_of_9 := 8
  let multiples_of_both := 4
  (multiples_of_6 + multiples_of_9 - multiples_of_both) / total_numbers

theorem ping_pong_ball_probability :
  multiple_of_6_9_or_both_probability = 2 / 9 :=
by
  sorry

end ping_pong_ball_probability_l76_76547


namespace cyclic_sum_inequality_l76_76770

theorem cyclic_sum_inequality (x y z a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ 3 / (a + b) :=
  sorry

end cyclic_sum_inequality_l76_76770


namespace cylinder_volume_ratio_l76_76226

theorem cylinder_volume_ratio (a b : ℕ) (h_dim : (a, b) = (9, 12)) :
  let r₁ := (a : ℝ) / (2 * Real.pi)
  let h₁ := (↑b : ℝ)
  let V₁ := (Real.pi * r₁^2 * h₁)
  let r₂ := (b : ℝ) / (2 * Real.pi)
  let h₂ := (↑a : ℝ)
  let V₂ := (Real.pi * r₂^2 * h₂)
  (if V₂ > V₁ then V₂ / V₁ else V₁ / V₂) = (16 / 3) :=
by {
  sorry
}

end cylinder_volume_ratio_l76_76226


namespace cube_difference_l76_76335

theorem cube_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) :
  a^3 - b^3 = 108 :=
sorry

end cube_difference_l76_76335


namespace average_salary_l76_76091

theorem average_salary (total_workers technicians other_workers technicians_avg_salary other_workers_avg_salary total_salary : ℝ)
  (h_workers : total_workers = 21)
  (h_technicians : technicians = 7)
  (h_other_workers : other_workers = total_workers - technicians)
  (h_technicians_avg_salary : technicians_avg_salary = 12000)
  (h_other_workers_avg_salary : other_workers_avg_salary = 6000)
  (h_total_technicians_salary : total_salary = (technicians * technicians_avg_salary + other_workers * other_workers_avg_salary))
  (h_total_other_salary : total_salary = 168000) :
  total_salary / total_workers = 8000 := by
    sorry

end average_salary_l76_76091


namespace simplify_expression_l76_76819

variable {R : Type} [LinearOrderedField R]

theorem simplify_expression (x y z : R) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h_sum : x + y + z = 3) :
  (1 / (y^2 + z^2 - x^2) + 1 / (x^2 + z^2 - y^2) + 1 / (x^2 + y^2 - z^2)) =
    3 / (-9 + 6 * y + 6 * z - 2 * y * z) :=
  sorry

end simplify_expression_l76_76819


namespace sum_of_roots_quadratic_l76_76050

theorem sum_of_roots_quadratic :
  ∀ (a b : ℝ), (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) :=
by
  intro a b
  intros
  sorry

end sum_of_roots_quadratic_l76_76050


namespace max_modulus_z_i_l76_76223

open Complex

theorem max_modulus_z_i (z : ℂ) (hz : abs z = 2) : ∃ z₂ : ℂ, abs z₂ = 2 ∧ abs (z₂ - I) = 3 :=
sorry

end max_modulus_z_i_l76_76223


namespace find_m_for_parallel_lines_l76_76823

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0 → 
  -((2 : ℝ) / (m + 1)) = -(m / 3)) → (m = 2 ∨ m = -3) :=
by
  sorry

end find_m_for_parallel_lines_l76_76823


namespace ordering_of_xyz_l76_76088

theorem ordering_of_xyz :
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  z < y ∧ y < x :=
by
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  sorry

end ordering_of_xyz_l76_76088


namespace min_boys_needed_l76_76687

theorem min_boys_needed
  (T : ℕ) -- total apples
  (n : ℕ) -- total number of boys
  (x : ℕ) -- number of boys collecting 20 apples each
  (y : ℕ) -- number of boys collecting 20% of total apples each
  (h1 : n = x + y)
  (h2 : T = 20 * x + Nat.div (T * 20 * y) 100)
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : n ≥ 2 :=
sorry

end min_boys_needed_l76_76687


namespace intersection_of_circle_and_line_l76_76327

theorem intersection_of_circle_and_line 
  (α : ℝ) 
  (x y : ℝ)
  (h1 : x = Real.cos α) 
  (h2 : y = 1 + Real.sin α) 
  (h3 : y = 1) :
  (x, y) = (1, 1) :=
by
  sorry

end intersection_of_circle_and_line_l76_76327


namespace tate_initial_tickets_l76_76772

theorem tate_initial_tickets (T : ℕ) (h1 : T + 2 + (T + 2)/2 = 51) : T = 32 := 
by
  sorry

end tate_initial_tickets_l76_76772


namespace sqrt_frac_meaningful_l76_76457

theorem sqrt_frac_meaningful (x : ℝ) (h : 1 / (x - 1) > 0) : x > 1 :=
sorry

end sqrt_frac_meaningful_l76_76457


namespace parts_processed_per_hour_before_innovation_l76_76337

variable (x : ℝ) (h : 1500 / x - 1500 / (2.5 * x) = 18)

theorem parts_processed_per_hour_before_innovation : x = 50 :=
by
  sorry

end parts_processed_per_hour_before_innovation_l76_76337


namespace roque_bike_time_l76_76834

-- Definitions of conditions
def roque_walk_time_per_trip : ℕ := 2
def roque_walk_trips_per_week : ℕ := 3
def roque_bike_trips_per_week : ℕ := 2
def total_commuting_time_per_week : ℕ := 16

-- Statement of the problem to prove
theorem roque_bike_time (B : ℕ) :
  (roque_walk_time_per_trip * 2 * roque_walk_trips_per_week + roque_bike_trips_per_week * 2 * B = total_commuting_time_per_week) → 
  B = 1 :=
by
  sorry

end roque_bike_time_l76_76834


namespace domain_of_sqrt_sum_l76_76013

theorem domain_of_sqrt_sum (x : ℝ) : (1 ≤ x ∧ x ≤ 3) ↔ (x - 1 ≥ 0 ∧ 3 - x ≥ 0) := by
  sorry

end domain_of_sqrt_sum_l76_76013


namespace radius_of_circle_l76_76363

theorem radius_of_circle (x y : ℝ) : (x^2 + y^2 - 8*x = 0) → (∃ r, r = 4) :=
by
  intro h
  sorry

end radius_of_circle_l76_76363


namespace rainfall_wednesday_correct_l76_76540

def monday_rainfall : ℝ := 0.9
def tuesday_rainfall : ℝ := monday_rainfall - 0.7
def wednesday_rainfall : ℝ := 2 * (monday_rainfall + tuesday_rainfall)

theorem rainfall_wednesday_correct : wednesday_rainfall = 2.2 := by
sorry

end rainfall_wednesday_correct_l76_76540


namespace orangeade_ratio_l76_76157

theorem orangeade_ratio (O W : ℝ) (price1 price2 : ℝ) (revenue1 revenue2 : ℝ)
  (h1 : price1 = 0.30) (h2 : price2 = 0.20)
  (h3 : revenue1 = revenue2)
  (glasses1 glasses2 : ℝ)
  (V : ℝ) :
  glasses1 = (O + W) / V → glasses2 = (O + 2 * W) / V →
  revenue1 = glasses1 * price1 → revenue2 = glasses2 * price2 →
  (O + W) * price1 = (O + 2 * W) * price2 → O / W = 1 :=
by sorry

end orangeade_ratio_l76_76157


namespace find_m_l76_76096

theorem find_m (m : ℕ) (h₁ : 256 = 4^4) : (256 : ℝ)^(1/4) = (4 : ℝ)^m ↔ m = 1 :=
by
  sorry

end find_m_l76_76096


namespace relationship_among_a_b_c_l76_76786

noncomputable def a : ℝ := 3 ^ Real.cos (Real.pi / 6)
noncomputable def b : ℝ := Real.log (Real.sin (Real.pi / 6)) / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log (Real.tan (Real.pi / 6)) / Real.log 2

theorem relationship_among_a_b_c : a > b ∧ b > c := 
by
  sorry

end relationship_among_a_b_c_l76_76786


namespace bob_can_order_199_sandwiches_l76_76455

-- Define the types of bread, meat, and cheese
def number_of_bread : ℕ := 5
def number_of_meat : ℕ := 7
def number_of_cheese : ℕ := 6

-- Define the forbidden combinations
def forbidden_turkey_swiss : ℕ := number_of_bread -- 5
def forbidden_rye_roastbeef : ℕ := number_of_cheese -- 6

-- Calculate the total sandwiches and subtract forbidden combinations
def total_sandwiches : ℕ := number_of_bread * number_of_meat * number_of_cheese
def forbidden_sandwiches : ℕ := forbidden_turkey_swiss + forbidden_rye_roastbeef

def sandwiches_bob_can_order : ℕ := total_sandwiches - forbidden_sandwiches

theorem bob_can_order_199_sandwiches :
  sandwiches_bob_can_order = 199 :=
by
  -- The calculation steps are encapsulated in definitions and are considered done
  sorry

end bob_can_order_199_sandwiches_l76_76455


namespace maximum_tangency_circles_l76_76704

/-- Points \( P_1, P_2, \ldots, P_n \) are in the plane
    Real numbers \( r_1, r_2, \ldots, r_n \) are such that the distance between \( P_i \) and \( P_j \) is \( r_i + r_j \) for \( i \ne j \).
    -/
theorem maximum_tangency_circles (n : ℕ) (P : Fin n → ℝ × ℝ) (r : Fin n → ℝ)
  (h : ∀ i j : Fin n, i ≠ j → dist (P i) (P j) = r i + r j) : n ≤ 4 :=
sorry

end maximum_tangency_circles_l76_76704


namespace inequality_of_ab_l76_76552

theorem inequality_of_ab (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b ∧ b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_of_ab_l76_76552


namespace acute_triangle_inequality_l76_76385

theorem acute_triangle_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = Real.pi)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.sin A + 1 / Real.sin B + 1 / Real.sin C) ≤
    Real.pi * (1 / A + 1 / B + 1 / C) :=
sorry

end acute_triangle_inequality_l76_76385


namespace scientific_notation_of_42_trillion_l76_76191

theorem scientific_notation_of_42_trillion : (42.1 * 10^12) = 4.21 * 10^13 :=
by
  sorry

end scientific_notation_of_42_trillion_l76_76191


namespace calculate_expression_l76_76660

theorem calculate_expression : (Real.sqrt 4) + abs (3 - Real.pi) + (1 / 3)⁻¹ = 2 + Real.pi :=
by 
  sorry

end calculate_expression_l76_76660


namespace percentage_of_men_l76_76462

variable (M W : ℝ)
variable (h1 : M + W = 100)
variable (h2 : 0.20 * W + 0.70 * M = 40)

theorem percentage_of_men : M = 40 :=
by
  sorry

end percentage_of_men_l76_76462


namespace workbooks_needed_l76_76143

theorem workbooks_needed (classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) (total_workbooks : ℕ) :
  classes = 25 → workbooks_per_class = 144 → spare_workbooks = 80 → total_workbooks = 25 * 144 + 80 → 
  total_workbooks = classes * workbooks_per_class + spare_workbooks :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end workbooks_needed_l76_76143


namespace average_children_in_families_with_children_l76_76178

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l76_76178


namespace length_of_bridge_is_correct_l76_76739

noncomputable def length_of_bridge (length_of_train : ℕ) (time_in_seconds : ℕ) (speed_in_kmph : ℝ) : ℝ :=
  let speed_in_mps := speed_in_kmph * (1000 / 3600)
  time_in_seconds * speed_in_mps - length_of_train

theorem length_of_bridge_is_correct :
  length_of_bridge 150 40 42.3 = 320 := by
  sorry

end length_of_bridge_is_correct_l76_76739


namespace sandy_total_spent_on_clothes_l76_76726

theorem sandy_total_spent_on_clothes :
  let shorts := 13.99
  let shirt := 12.14 
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := 
by
  sorry

end sandy_total_spent_on_clothes_l76_76726


namespace multiplication_addition_l76_76065

theorem multiplication_addition :
  23 * 37 + 16 = 867 :=
by
  sorry

end multiplication_addition_l76_76065


namespace complex_pow_difference_l76_76044

theorem complex_pow_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 12 - (1 - i) ^ 12 = 0 :=
  sorry

end complex_pow_difference_l76_76044


namespace sin_minus_cos_value_l76_76656

open Real

noncomputable def tan_alpha := sqrt 3
noncomputable def alpha_condition (α : ℝ) := π < α ∧ α < (3 / 2) * π

theorem sin_minus_cos_value (α : ℝ) (h1 : tan α = tan_alpha) (h2 : alpha_condition α) : 
  sin α - cos α = -((sqrt 3) - 1) / 2 := 
by 
  sorry

end sin_minus_cos_value_l76_76656


namespace cricket_target_runs_l76_76570

-- Define the conditions
def first_20_overs_run_rate : ℝ := 4.2
def remaining_30_overs_run_rate : ℝ := 8
def overs_20 : ℤ := 20
def overs_30 : ℤ := 30

-- State the proof problem
theorem cricket_target_runs : 
  (first_20_overs_run_rate * (overs_20 : ℝ)) + (remaining_30_overs_run_rate * (overs_30 : ℝ)) = 324 :=
by
  sorry

end cricket_target_runs_l76_76570


namespace neg_proposition_equiv_l76_76798

theorem neg_proposition_equiv :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) :=
by
  sorry

end neg_proposition_equiv_l76_76798


namespace find_f_of_2_l76_76869

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x) = (1 + x) / x) : f 2 = 3 :=
sorry

end find_f_of_2_l76_76869


namespace molecular_weight_proof_l76_76504

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_H : Float := 1.01

def molecular_weight_AlOH3 : Float :=
  (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

def moles : Float := 7.0

def molecular_weight_7_moles_AlOH3 : Float :=
  moles * molecular_weight_AlOH3

theorem molecular_weight_proof : molecular_weight_7_moles_AlOH3 = 546.07 :=
by
  /- Here we calculate the molecular weight of Al(OH)3 and multiply it by 7.
     molecular_weight_AlOH3 = (1 * 26.98) + (3 * 16.00) + (3 * 1.01) = 78.01
     molecular_weight_7_moles_AlOH3 = 7 * 78.01 = 546.07 -/
  sorry

end molecular_weight_proof_l76_76504


namespace converse_even_sum_l76_76627

def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem converse_even_sum (a b : Int) :
  (is_even a ∧ is_even b → is_even (a + b)) →
  (is_even (a + b) → is_even a ∧ is_even b) :=
by
  sorry

end converse_even_sum_l76_76627


namespace m_is_perfect_square_l76_76561

theorem m_is_perfect_square (n : ℕ) (m : ℤ) (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1) ∧ Int.sqrt (44 * n^2 + 1) * Int.sqrt (44 * n^2 + 1) = 44 * n^2 + 1) :
  ∃ k : ℕ, m = k^2 :=
by
  sorry

end m_is_perfect_square_l76_76561


namespace trigonometric_identity_l76_76706

theorem trigonometric_identity (alpha : ℝ) (h : Real.tan alpha = 2 * Real.tan (π / 5)) :
  (Real.cos (alpha - 3 * π / 10) / Real.sin (alpha - π / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l76_76706


namespace fourth_person_height_l76_76874

theorem fourth_person_height 
  (h : ℝ)
  (height_average : (h + (h + 2) + (h + 4) + (h + 10)) / 4 = 79)
  : h + 10 = 85 := 
by
  sorry

end fourth_person_height_l76_76874


namespace find_x_rational_l76_76597

theorem find_x_rational (x : ℝ) (h1 : ∃ (a : ℚ), x + Real.sqrt 3 = a)
  (h2 : ∃ (b : ℚ), x^2 + Real.sqrt 3 = b) :
  x = (1 / 2 : ℝ) - Real.sqrt 3 :=
sorry

end find_x_rational_l76_76597


namespace total_votes_l76_76674

theorem total_votes (V : ℝ) (C R : ℝ) 
  (hC : C = 0.10 * V)
  (hR1 : R = 0.10 * V + 16000)
  (hR2 : R = 0.90 * V) :
  V = 20000 :=
by
  sorry

end total_votes_l76_76674


namespace maximum_consecutive_positive_integers_sum_500_l76_76309

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l76_76309


namespace marbles_percentage_l76_76579

def solid_color_other_than_yellow (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) : ℚ :=
  solid_color_percent - solid_yellow_percent

theorem marbles_percentage (total_marbles : ℕ) (solid_color_percent solid_yellow_percent : ℚ) :
  solid_color_percent = 90 / 100 →
  solid_yellow_percent = 5 / 100 →
  solid_color_other_than_yellow total_marbles solid_color_percent solid_yellow_percent = 85 / 100 :=
by
  intro h1 h2
  rw [h1, h2]
  norm_num
  sorry

end marbles_percentage_l76_76579


namespace cube_root_of_unity_identity_l76_76989

theorem cube_root_of_unity_identity (ω : ℂ) (hω3: ω^3 = 1) (hω_ne_1 : ω ≠ 1) (hunit : ω^2 + ω + 1 = 0) :
  (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
by
  sorry

end cube_root_of_unity_identity_l76_76989


namespace mutually_exclusive_not_opposite_l76_76246

universe u

-- Define the colors and people involved
inductive Color
| black
| red
| white

inductive Person 
| A
| B
| C

-- Define a function that distributes the cards amongst the people
def distributes (cards : List Color) (people : List Person) : People -> Color :=
  sorry

-- Define events as propositions
def A_gets_red (d : Person -> Color) : Prop :=
  d Person.A = Color.red

def B_gets_red (d : Person -> Color) : Prop :=
  d Person.B = Color.red

-- The main theorem stating the problem
theorem mutually_exclusive_not_opposite 
  (d : Person -> Color)
  (h : A_gets_red d → ¬ B_gets_red d) : 
  ¬ ( ∀ (p : Prop), A_gets_red d ↔ p ) → B_gets_red d :=
sorry

end mutually_exclusive_not_opposite_l76_76246


namespace cylinder_radius_and_volume_l76_76317

theorem cylinder_radius_and_volume
  (h : ℝ) (surface_area : ℝ) :
  h = 8 ∧ surface_area = 130 * Real.pi →
  ∃ (r : ℝ) (V : ℝ), r = 5 ∧ V = 200 * Real.pi := by
  sorry

end cylinder_radius_and_volume_l76_76317


namespace Kim_nail_polishes_l76_76802

-- Define the conditions
variable (K : ℕ)
def Heidi_nail_polishes (K : ℕ) : ℕ := K + 5
def Karen_nail_polishes (K : ℕ) : ℕ := K - 4

-- The main statement to prove
theorem Kim_nail_polishes (K : ℕ) (H : Heidi_nail_polishes K + Karen_nail_polishes K = 25) : K = 12 := by
  sorry

end Kim_nail_polishes_l76_76802


namespace max_vx_minus_yz_l76_76193

-- Define the set A
def A : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

-- Define the conditions
variables (v w x y z : ℤ)
#check v ∈ A -- v belongs to set A
#check w ∈ A -- w belongs to set A
#check x ∈ A -- x belongs to set A
#check y ∈ A -- y belongs to set A
#check z ∈ A -- z belongs to set A

-- vw = x
axiom vw_eq_x : v * w = x

-- w ≠ 0
axiom w_ne_zero : w ≠ 0

-- The target problem
theorem max_vx_minus_yz : ∃ v w x y z : ℤ, v ∈ A ∧ w ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ v * w = x ∧ w ≠ 0 ∧ (v * x - y * z) = 150 := by
  sorry

end max_vx_minus_yz_l76_76193


namespace jake_delay_l76_76123

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end jake_delay_l76_76123


namespace shape_volume_to_surface_area_ratio_l76_76229

/-- 
Define the volume and surface area of our specific shape with given conditions:
1. Five unit cubes in a straight line.
2. An additional cube on top of the second cube.
3. Another cube beneath the fourth cube.

Prove that the ratio of the volume to the surface area is \( \frac{1}{4} \).
-/
theorem shape_volume_to_surface_area_ratio :
  let volume := 7
  let surface_area := 28
  volume / surface_area = 1 / 4 :=
by
  sorry

end shape_volume_to_surface_area_ratio_l76_76229


namespace no_such_triples_l76_76405

theorem no_such_triples 
  (a b c : ℕ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : ¬ ∃ k, k ∣ a + c ∧ k ∣ b + c ∧ k ∣ a + b) 
  (h₃ : c^2 ∣ a + b) 
  (h₄ : b^2 ∣ a + c) 
  (h₅ : a^2 ∣ b + c) : 
  false :=
sorry

end no_such_triples_l76_76405


namespace dan_balloons_correct_l76_76453

-- Define the initial conditions
def sam_initial_balloons : Float := 46.0
def sam_given_fred_balloons : Float := 10.0
def total_balloons : Float := 52.0

-- Calculate Sam's remaining balloons
def sam_current_balloons : Float := sam_initial_balloons - sam_given_fred_balloons

-- Define the target: Dan's balloons
def dan_balloons := total_balloons - sam_current_balloons

-- Statement to prove
theorem dan_balloons_correct : dan_balloons = 16.0 := sorry

end dan_balloons_correct_l76_76453


namespace age_ratio_l76_76557

/-- Given that Sandy's age after 6 years will be 30 years,
    and Molly's current age is 18 years, 
    prove that the current ratio of Sandy's age to Molly's age is 4:3. -/
theorem age_ratio (M S : ℕ) 
  (h1 : M = 18) 
  (h2 : S + 6 = 30) : 
  S / gcd S M = 4 ∧ M / gcd S M = 3 :=
by
  sorry

end age_ratio_l76_76557


namespace find_a_l76_76218

theorem find_a (a : ℕ) (h : a * 2 * 2^3 = 2^6) : a = 4 := 
by 
  sorry

end find_a_l76_76218


namespace arithmetic_sequence_sum_l76_76338

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_sum
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l76_76338


namespace points_on_circle_l76_76569

theorem points_on_circle (t : ℝ) : 
  ( (2 - 3 * t^2) / (2 + t^2) )^2 + ( 3 * t / (2 + t^2) )^2 = 1 := 
by 
  sorry

end points_on_circle_l76_76569


namespace bicycle_profit_theorem_l76_76864

def bicycle_profit_problem : Prop :=
  let CP_A : ℝ := 120
  let SP_C : ℝ := 225
  let profit_percentage_B : ℝ := 0.25
  -- intermediate calculations
  let CP_B : ℝ := SP_C / (1 + profit_percentage_B)
  let SP_A : ℝ := CP_B
  let Profit_A : ℝ := SP_A - CP_A
  let Profit_Percentage_A : ℝ := (Profit_A / CP_A) * 100
  -- final statement to prove
  Profit_Percentage_A = 50

theorem bicycle_profit_theorem : bicycle_profit_problem := 
by
  sorry

end bicycle_profit_theorem_l76_76864


namespace find_a_l76_76502

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (1 - x) - Real.log (1 + x) + a

theorem find_a 
  (M : ℝ) (N : ℝ) (a : ℝ)
  (h1 : M = f a (-1/2))
  (h2 : N = f a (1/2))
  (h3 : M + N = 1) :
  a = 1 / 2 := 
sorry

end find_a_l76_76502


namespace problem_statement_l76_76459

-- Definitions corresponding to the given condition
noncomputable def sum_to_n (n : ℕ) : ℤ := (n * (n + 1)) / 2
noncomputable def alternating_sum_to_n (n : ℕ) : ℤ := if n % 2 = 0 then -(n / 2) else (n / 2 + 1)

-- Lean statement for the problem
theorem problem_statement :
  (alternating_sum_to_n 2022) * (sum_to_n 2023 - 1) - (alternating_sum_to_n 2023) * (sum_to_n 2022 - 1) = 2023 :=
sorry

end problem_statement_l76_76459


namespace find_multiple_of_number_l76_76038

theorem find_multiple_of_number (n : ℝ) (m : ℝ) (h1 : n ≠ 0) (h2 : n = 9) (h3 : (n + n^2) / 2 = m * n) : m = 5 :=
sorry

end find_multiple_of_number_l76_76038


namespace total_dog_legs_l76_76409

theorem total_dog_legs (total_animals cats dogs: ℕ) (h1: total_animals = 300) 
  (h2: cats = 2 / 3 * total_animals) 
  (h3: dogs = 1 / 3 * total_animals): (dogs * 4) = 400 :=
by
  sorry

end total_dog_legs_l76_76409


namespace geometric_sequence_20_sum_is_2_pow_20_sub_1_l76_76675

def geometric_sequence_sum_condition (a : ℕ → ℕ) (q : ℕ) : Prop :=
  (a 1 * q + 2 * a 1 = 4) ∧ (a 1 ^ 2 * q ^ 4 = a 1 * q ^ 4)

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) : ℕ :=
  (a 1 * (1 - q ^ 20)) / (1 - q)

theorem geometric_sequence_20_sum_is_2_pow_20_sub_1 (a : ℕ → ℕ) (q : ℕ) 
  (h : geometric_sequence_sum_condition a q) : 
  geometric_sequence_sum a q =  2 ^ 20 - 1 := 
sorry

end geometric_sequence_20_sum_is_2_pow_20_sub_1_l76_76675


namespace waiter_tables_l76_76082

/-
Problem:
A waiter had 22 customers in his section.
14 of them left.
The remaining customers were seated at tables with 4 people per table.
Prove the number of tables is 2.
-/

theorem waiter_tables:
  ∃ (tables : ℤ), 
    (∀ (customers_initial customers_remaining people_per_table tables_calculated : ℤ), 
      customers_initial = 22 →
      customers_remaining = customers_initial - 14 →
      people_per_table = 4 →
      tables_calculated = customers_remaining / people_per_table →
      tables = tables_calculated) →
    tables = 2 :=
by
  sorry

end waiter_tables_l76_76082


namespace larger_number_hcf_lcm_l76_76258

theorem larger_number_hcf_lcm (a b : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 20) 
  (h_factor1 : factor1 = 13) 
  (h_factor2 : factor2 = 14) 
  (h_ab_hcf : Nat.gcd a b = hcf)
  (h_ab_lcm : Nat.lcm a b = hcf * factor1 * factor2) :
  max a b = 280 :=
by 
  sorry

end larger_number_hcf_lcm_l76_76258


namespace cookie_distribution_l76_76360

theorem cookie_distribution : 
  ∀ (n c T : ℕ), n = 6 → c = 4 → T = n * c → T = 24 :=
by 
  intros n c T h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cookie_distribution_l76_76360


namespace garden_area_l76_76105

theorem garden_area 
  (property_width : ℕ)
  (property_length : ℕ)
  (garden_width_ratio : ℚ)
  (garden_length_ratio : ℚ)
  (width_ratio_eq : garden_width_ratio = (1 : ℚ) / 8)
  (length_ratio_eq : garden_length_ratio = (1 : ℚ) / 10)
  (property_width_eq : property_width = 1000)
  (property_length_eq : property_length = 2250) :
  (property_width * garden_width_ratio * property_length * garden_length_ratio = 28125) :=
  sorry

end garden_area_l76_76105


namespace rolling_green_probability_l76_76017

/-- A cube with 5 green faces and 1 yellow face. -/
structure ColoredCube :=
  (green_faces : ℕ)
  (yellow_face : ℕ)
  (total_faces : ℕ)

def example_cube : ColoredCube :=
  { green_faces := 5, yellow_face := 1, total_faces := 6 }

/-- The probability of rolling a green face on a given cube. -/
def probability_of_rolling_green (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

theorem rolling_green_probability :
  probability_of_rolling_green example_cube = 5 / 6 :=
by simp [probability_of_rolling_green, example_cube]

end rolling_green_probability_l76_76017


namespace percent_voters_for_candidate_A_l76_76767

theorem percent_voters_for_candidate_A (d r i u p_d p_r p_i p_u : ℝ) 
  (hd : d = 0.45) (hr : r = 0.30) (hi : i = 0.20) (hu : u = 0.05)
  (hp_d : p_d = 0.75) (hp_r : p_r = 0.25) (hp_i : p_i = 0.50) (hp_u : p_u = 0.50) :
  d * p_d + r * p_r + i * p_i + u * p_u = 0.5375 :=
by
  sorry

end percent_voters_for_candidate_A_l76_76767


namespace yacht_capacity_l76_76791

theorem yacht_capacity :
  ∀ (x y : ℕ), (3 * x + 2 * y = 68) → (2 * x + 3 * y = 57) → (3 * x + 6 * y = 96) :=
by
  intros x y h1 h2
  sorry

end yacht_capacity_l76_76791


namespace walking_speed_is_correct_l76_76364

-- Define the conditions
def time_in_minutes : ℝ := 10
def distance_in_meters : ℝ := 1666.6666666666665
def speed_in_km_per_hr : ℝ := 2.777777777777775

-- Define the theorem to prove
theorem walking_speed_is_correct :
  (distance_in_meters / time_in_minutes) * 60 / 1000 = speed_in_km_per_hr :=
sorry

end walking_speed_is_correct_l76_76364


namespace Jason_cards_l76_76036

theorem Jason_cards (initial_cards : ℕ) (cards_bought : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 3) (h2 : cards_bought = 2) : remaining_cards = 1 :=
by
  sorry

end Jason_cards_l76_76036


namespace tournament_game_count_l76_76779

/-- In a tournament with 25 players where each player plays 4 games against each other,
prove that the total number of games played is 1200. -/
theorem tournament_game_count : 
  let n := 25
  let games_per_pair := 4
  let total_games := (n * (n - 1) / 2) * games_per_pair
  total_games = 1200 :=
by
  -- Definitions based on the conditions
  let n := 25
  let games_per_pair := 4

  -- Calculating the total number of games
  let total_games := (n * (n - 1) / 2) * games_per_pair

  -- This is the main goal to prove
  have h : total_games = 1200 := sorry
  exact h

end tournament_game_count_l76_76779


namespace find_f91_plus_fm91_l76_76943

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^6 + b * x^4 - c * x^2 + 3

theorem find_f91_plus_fm91 (a b c : ℝ) (h : f 91 a b c = 1) : f 91 a b c + f (-91) a b c = 2 := by
  sorry

end find_f91_plus_fm91_l76_76943


namespace count_3_digit_numbers_divisible_by_5_l76_76113

theorem count_3_digit_numbers_divisible_by_5 :
  let a := 100
  let l := 995
  let d := 5
  let n := (l - a) / d + 1
  n = 180 :=
by
  sorry

end count_3_digit_numbers_divisible_by_5_l76_76113


namespace planned_daily_catch_l76_76374

theorem planned_daily_catch (x y : ℝ) 
  (h1 : x * y = 1800)
  (h2 : (x / 3) * (y - 20) + ((2 * x / 3) - 1) * (y + 20) = 1800) :
  y = 100 :=
by
  sorry

end planned_daily_catch_l76_76374


namespace sec_neg_450_undefined_l76_76555

theorem sec_neg_450_undefined : ¬ ∃ x, x = 1 / Real.cos (-450 * Real.pi / 180) :=
by
  -- Proof skipped using 'sorry'
  sorry

end sec_neg_450_undefined_l76_76555


namespace vector_sum_magnitude_eq_2_or_5_l76_76535

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := 3
def equal_angles (θ : ℝ) := θ = 120 ∨ θ = 0

theorem vector_sum_magnitude_eq_2_or_5
  (a_mag : ℝ := a)
  (b_mag : ℝ := b)
  (c_mag : ℝ := c)
  (θ : ℝ)
  (Hθ : equal_angles θ) :
  (|a_mag| = 1) ∧ (|b_mag| = 1) ∧ (|c_mag| = 3) →
  (|a_mag + b_mag + c_mag| = 2 ∨ |a_mag + b_mag + c_mag| = 5) :=
by
  sorry

end vector_sum_magnitude_eq_2_or_5_l76_76535


namespace largest_angle_smallest_angle_middle_angle_l76_76297

-- Definitions for angles of a triangle in degrees
variable (α β γ : ℝ)
variable (h_sum : α + β + γ = 180)

-- Largest angle condition
theorem largest_angle (h1 : α ≥ β) (h2 : α ≥ γ) : (60 ≤ α ∧ α < 180) :=
  sorry

-- Smallest angle condition
theorem smallest_angle (h1 : α ≤ β) (h2 : α ≤ γ) : (0 < α ∧ α ≤ 60) :=
  sorry

-- Middle angle condition
theorem middle_angle (h1 : α > β ∧ α < γ ∨ α < β ∧ α > γ) : (0 < α ∧ α < 90) :=
  sorry

end largest_angle_smallest_angle_middle_angle_l76_76297


namespace find_integer_tuples_l76_76695

theorem find_integer_tuples (a b c x y z : ℤ) :
  a + b + c = x * y * z →
  x + y + z = a * b * c →
  a ≥ b → b ≥ c → c ≥ 1 →
  x ≥ y → y ≥ z → z ≥ 1 →
  (a, b, c, x, y, z) = (2, 2, 2, 6, 1, 1) ∨
  (a, b, c, x, y, z) = (5, 2, 1, 8, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 3, 1, 7, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 2, 1, 6, 2, 1) :=
by
  sorry

end find_integer_tuples_l76_76695


namespace equation_one_solution_equation_two_solution_l76_76494

theorem equation_one_solution (x : ℝ) : ((x + 3) ^ 2 - 9 = 0) ↔ (x = 0 ∨ x = -6) := by
  sorry

theorem equation_two_solution (x : ℝ) : (x ^ 2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  sorry

end equation_one_solution_equation_two_solution_l76_76494


namespace initial_mixtureA_amount_l76_76070

-- Condition 1: Mixture A is 20% oil and 80% material B by weight.
def oil_content (x : ℝ) : ℝ := 0.20 * x
def materialB_content (x : ℝ) : ℝ := 0.80 * x

-- Condition 2: 2 more kilograms of oil are added to a certain amount of mixture A
def oil_added := 2

-- Condition 3: 6 kilograms of mixture A must be added to make a 70% material B in the new mixture.
def mixture_added := 6

-- The total weight of the new mixture
def total_weight (x : ℝ) : ℝ := x + mixture_added + oil_added

-- The total amount of material B in the new mixture
def total_materialB (x : ℝ) : ℝ := 0.80 * x + 0.80 * mixture_added

-- The new mixture is supposed to be 70% material B.
def is_70_percent_materialB (x : ℝ) : Prop := total_materialB x = 0.70 * total_weight x

-- Proving x == 8 given the conditions
theorem initial_mixtureA_amount : ∃ x : ℝ, is_70_percent_materialB x ∧ x = 8 :=
by
  sorry

end initial_mixtureA_amount_l76_76070


namespace sum_of_consecutive_integers_l76_76413

theorem sum_of_consecutive_integers (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + 1 = b) (h4 : b + 1 = c) (h5 : a * b * c = 336) : a + b + c = 21 :=
sorry

end sum_of_consecutive_integers_l76_76413


namespace largest_of_four_numbers_l76_76023

variables {x y z w : ℕ}

theorem largest_of_four_numbers
  (h1 : x + y + z = 180)
  (h2 : x + y + w = 197)
  (h3 : x + z + w = 208)
  (h4 : y + z + w = 222) :
  max x (max y (max z w)) = 89 :=
sorry

end largest_of_four_numbers_l76_76023


namespace angle_A_range_l76_76551

open Real

theorem angle_A_range (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) (h3 : 0 < A ∧ A < π) : 
  π / 2 < A ∧ A < 3 * π / 4 :=
by
  sorry

end angle_A_range_l76_76551


namespace smallest_x_satisfying_abs_eq_l76_76661

theorem smallest_x_satisfying_abs_eq (x : ℝ) 
  (h : |2 * x^2 + 3 * x - 1| = 33) : 
  x = (-3 - Real.sqrt 281) / 4 := 
sorry

end smallest_x_satisfying_abs_eq_l76_76661


namespace problem_final_value_l76_76601

theorem problem_final_value (x y z : ℝ) (hz : z ≠ 0) 
  (h1 : 3 * x - 2 * y - 2 * z = 0) 
  (h2 : x - 4 * y + 8 * z = 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := 
by 
  sorry

end problem_final_value_l76_76601


namespace jake_buys_packages_l76_76438

theorem jake_buys_packages:
  ∀ (pkg_weight cost_per_pound total_paid : ℕ),
    pkg_weight = 2 →
    cost_per_pound = 4 →
    total_paid = 24 →
    (total_paid / (pkg_weight * cost_per_pound)) = 3 :=
by
  intros pkg_weight cost_per_pound total_paid hw_cp ht
  sorry

end jake_buys_packages_l76_76438


namespace Q_subset_P_l76_76906

def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | x^2 < 4 }

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l76_76906


namespace monomials_exponents_l76_76908

theorem monomials_exponents (m n : ℕ) 
  (h₁ : 3 * x ^ 5 * y ^ m + -2 * x ^ n * y ^ 7 = 0) : m - n = 2 := 
by
  sorry

end monomials_exponents_l76_76908


namespace equal_expense_sharing_l76_76282

variables (O L B : ℝ)

theorem equal_expense_sharing (h1 : O < L) (h2 : O < B) : 
    (L + B - 2 * O) / 6 = (O + L + B) / 3 - O :=
by
    sorry

end equal_expense_sharing_l76_76282


namespace parabola_reflection_translation_l76_76530

open Real

noncomputable def f (a b c x : ℝ) : ℝ := a * (x - 4)^2 + b * (x - 4) + c
noncomputable def g (a b c x : ℝ) : ℝ := -a * (x + 4)^2 - b * (x + 4) - c
noncomputable def fg_x (a b c x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation (a b c x : ℝ) (ha : a ≠ 0) :
  fg_x a b c x = -16 * a * x :=
by
  sorry

end parabola_reflection_translation_l76_76530


namespace complex_number_a_eq_1_l76_76918

theorem complex_number_a_eq_1 
  (a : ℝ) 
  (h : ∃ b : ℝ, (a - b * I) / (1 + I) = 0 + b * I) : 
  a = 1 := 
sorry

end complex_number_a_eq_1_l76_76918


namespace number_of_schools_l76_76507

theorem number_of_schools (total_students d : ℕ) (S : ℕ) (ellen frank : ℕ) (d_median : total_students = 2 * d - 1)
    (d_highest : ellen < d) (ellen_position : ellen = 29) (frank_position : frank = 50) (team_size : ∀ S, total_students = 3 * S) : 
    S = 19 := 
by 
  sorry

end number_of_schools_l76_76507


namespace line_through_point_equal_intercepts_l76_76961

theorem line_through_point_equal_intercepts (x y a b : ℝ) :
  ∀ (x y : ℝ), 
    (x - 1) = a → 
    (y - 2) = b →
    (a = -1 ∨ a = 2) → 
    ((x + y - 3 = 0) ∨ (2 * x - y = 0)) := by
  sorry

end line_through_point_equal_intercepts_l76_76961


namespace doughnut_cost_l76_76344

theorem doughnut_cost:
  ∃ (D C : ℝ), 
    3 * D + 4 * C = 4.91 ∧ 
    5 * D + 6 * C = 7.59 ∧ 
    D = 0.45 :=
by
  sorry

end doughnut_cost_l76_76344


namespace john_money_left_l76_76384

-- Given definitions
def drink_cost (q : ℝ) := q
def small_pizza_cost (q : ℝ) := q
def large_pizza_cost (q : ℝ) := 4 * q
def initial_amount := 50

-- Problem statement
theorem john_money_left (q : ℝ) : initial_amount - (4 * drink_cost q + 2 * small_pizza_cost q + large_pizza_cost q) = 50 - 10 * q :=
by
  sorry

end john_money_left_l76_76384


namespace hypotenuse_length_l76_76518

-- Let a and b be the lengths of the non-hypotenuse sides of a right triangle.
-- We are given that a = 6 and b = 8, and we need to prove that the hypotenuse c is 10.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c ^ 2 = a ^ 2 + b ^ 2) : c = 10 :=
by
  -- The proof goes here.
  sorry

end hypotenuse_length_l76_76518


namespace contingency_fund_amount_l76_76969

theorem contingency_fund_amount :
  ∀ (donation : ℝ),
  (1/3 * donation + 1/2 * donation + 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2  * donation)))) →
  (donation = 240) → (donation - (1/3 * donation + 1/2 * donation) - 1/4 * (donation - (1/3 * donation + 1/2 * donation)) = 30) :=
by
    intro donation h1 h2
    sorry

end contingency_fund_amount_l76_76969


namespace find_integer_n_l76_76728

def s : List ℤ := [8, 11, 12, 14, 15]

theorem find_integer_n (n : ℤ) (h : (s.sum + n) / (s.length + 1) = (25 / 100) * (s.sum / s.length) + (s.sum / s.length)) : n = 30 := by
  sorry

end find_integer_n_l76_76728


namespace negation_of_existence_l76_76562

variable (Triangle : Type) (has_circumcircle : Triangle → Prop)

theorem negation_of_existence :
  ¬ (∃ t : Triangle, ¬ has_circumcircle t) ↔ ∀ t : Triangle, has_circumcircle t :=
by sorry

end negation_of_existence_l76_76562


namespace number_of_pizza_varieties_l76_76415

-- Definitions for the problem conditions
def number_of_flavors : Nat := 8
def toppings : List String := ["C", "M", "O", "J", "L"]

-- Function to count valid combinations of toppings
def valid_combinations (n : Nat) : Nat :=
  match n with
  | 1 => 5
  | 2 => 10 - 1 -- Subtracting the invalid combination (O, J)
  | 3 => 10 - 3 -- Subtracting the 3 invalid combinations containing (O, J)
  | _ => 0

def total_topping_combinations : Nat :=
  valid_combinations 1 + valid_combinations 2 + valid_combinations 3

-- The final proof stating the number of pizza varieties
theorem number_of_pizza_varieties : total_topping_combinations * number_of_flavors = 168 := by
  -- Calculation steps can be inserted here, we use sorry for now
  sorry

end number_of_pizza_varieties_l76_76415


namespace system_of_equations_has_no_solution_l76_76685

theorem system_of_equations_has_no_solution
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y + z = 10)
  (h2 : 6 * x - 8 * y + 2 * z = 16)
  (h3 : x + y - z = 3) :
  false :=
by 
  sorry

end system_of_equations_has_no_solution_l76_76685


namespace brandon_textbooks_weight_l76_76575

-- Define the weights of Jon's textbooks
def jon_textbooks : List ℕ := [2, 8, 5, 9]

-- Define the weight ratio between Jon's and Brandon's textbooks
def weight_ratio : ℕ := 3

-- Define the total weight of Jon's textbooks
def weight_jon : ℕ := jon_textbooks.sum

-- Define the weight of Brandon's textbooks to be proven
def weight_brandon : ℕ := weight_jon / weight_ratio

-- The theorem to be proven
theorem brandon_textbooks_weight : weight_brandon = 8 :=
by sorry

end brandon_textbooks_weight_l76_76575


namespace symmetric_point_yoz_l76_76512

theorem symmetric_point_yoz (x y z : ℝ) (hx : x = 2) (hy : y = 3) (hz : z = 4) :
  (-x, y, z) = (-2, 3, 4) :=
by
  -- The proof is skipped
  sorry

end symmetric_point_yoz_l76_76512


namespace minimum_value_expression_l76_76115

theorem minimum_value_expression (x y : ℝ) : ∃ (m : ℝ), ∀ x y : ℝ, x^2 + 3 * x * y + y^2 ≥ m ∧ m = 0 :=
by
  use 0
  sorry

end minimum_value_expression_l76_76115


namespace solution_set_l76_76266

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution_set (x : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_def : ∀ x : ℝ, x >= 0 → f x = x^2 - 4 * x) :
    f (x + 2) < 5 ↔ -7 < x ∧ x < 3 :=
sorry

end solution_set_l76_76266


namespace find_a_l76_76042

theorem find_a (a: ℕ) : (2000 + 100 * a + 17) % 19 = 0 ↔ a = 7 :=
by
  sorry

end find_a_l76_76042


namespace algebra_expression_evaluation_l76_76975

theorem algebra_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 5) : -2 * a^2 - 4 * a + 5 = -7 :=
by
  sorry

end algebra_expression_evaluation_l76_76975


namespace remainder_division_l76_76134

def f (x : ℝ) : ℝ := x^3 - 4 * x + 7

theorem remainder_division (x : ℝ) : f 3 = 22 := by
  sorry

end remainder_division_l76_76134


namespace find_length_DY_l76_76951

noncomputable def length_DY : Real :=
    let AE := 2
    let AY := 4 * AE
    let DY  := Real.sqrt (66 + Real.sqrt 5)
    DY

theorem find_length_DY : length_DY = Real.sqrt (66 + Real.sqrt 5) := 
  by
    sorry

end find_length_DY_l76_76951


namespace combinations_sum_l76_76319
open Nat

theorem combinations_sum : 
  let d := [1, 2, 3, 4]
  let count_combinations (n : Nat) := factorial n
  count_combinations 1 + count_combinations 2 + count_combinations 3 + count_combinations 4 = 64 :=
  by
    sorry

end combinations_sum_l76_76319


namespace area_of_triangle_ABC_l76_76747

/--
Given a triangle \(ABC\) with points \(D\) and \(E\) on sides \(BC\) and \(AC\) respectively,
where \(BD = 4\), \(DE = 2\), \(EC = 6\), and \(BF = FC = 3\),
proves that the area of triangle \( \triangle ABC \) is \( 18\sqrt{3} \).
-/
theorem area_of_triangle_ABC :
  ∀ (ABC D E : Type) (BD DE EC BF FC : ℝ),
    BD = 4 → DE = 2 → EC = 6 → BF = 3 → FC = 3 → 
    ∃ area, area = 18 * Real.sqrt 3 :=
by
  intros ABC D E BD DE EC BF FC hBD hDE hEC hBF hFC
  sorry

end area_of_triangle_ABC_l76_76747


namespace gravel_cost_l76_76330

-- Definitions of conditions
def lawn_length : ℝ := 70
def lawn_breadth : ℝ := 30
def road_width : ℝ := 5
def gravel_cost_per_sqm : ℝ := 4

-- Theorem statement
theorem gravel_cost : (lawn_length * road_width + lawn_breadth * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by
  -- Definitions used in the problem
  let area_first_road := lawn_length * road_width
  let area_second_road := lawn_breadth * road_width
  let area_intersection := road_width * road_width

  -- Total area to be graveled
  let total_area_to_be_graveled := area_first_road + area_second_road - area_intersection

  -- Calculate the cost
  let cost := total_area_to_be_graveled * gravel_cost_per_sqm

  show cost = 1900
  sorry

end gravel_cost_l76_76330


namespace faster_train_passes_slower_l76_76424

theorem faster_train_passes_slower (v_fast v_slow : ℝ) (length_fast : ℝ) 
  (hv_fast : v_fast = 50) (hv_slow : v_slow = 32) (hl_length_fast : length_fast = 75) :
  ∃ t : ℝ, t = 15 := 
by
  sorry

end faster_train_passes_slower_l76_76424


namespace infinite_k_values_l76_76084

theorem infinite_k_values (k : ℕ) : (∃ k, ∀ (a b c : ℕ),
  (a = 64 ∧ b ≥ 0 ∧ c = 0 ∧ k = 2^a * 3^b * 5^c) ↔
  Nat.lcm (Nat.lcm (2^8) (2^24 * 3^12)) k = 2^64) →
  ∃ (b : ℕ), true :=
by
  sorry

end infinite_k_values_l76_76084


namespace power_function_passes_through_point_l76_76983

theorem power_function_passes_through_point (a : ℝ) : (2 ^ a = Real.sqrt 2) → (a = 1 / 2) :=
  by
  intro h
  sorry

end power_function_passes_through_point_l76_76983


namespace find_a22_l76_76345

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l76_76345


namespace regression_line_equation_l76_76607

-- Define the conditions in the problem
def slope_of_regression_line : ℝ := 1.23
def center_of_sample_points : ℝ × ℝ := (4, 5)

-- The proof problem to show that the equation of the regression line is y = 1.23x + 0.08
theorem regression_line_equation :
  ∃ b : ℝ, (∀ x y : ℝ, (y = slope_of_regression_line * x + b) 
  → (4, 5) = (x, y)) → b = 0.08 :=
sorry

end regression_line_equation_l76_76607


namespace algebraic_expression_value_l76_76657

noncomputable def a : ℝ := Real.sqrt 6 + 1
noncomputable def b : ℝ := Real.sqrt 6 - 1

theorem algebraic_expression_value :
  a^2 + a * b = 12 + 2 * Real.sqrt 6 :=
sorry

end algebraic_expression_value_l76_76657


namespace expr_value_l76_76574

-- Define the given expression
def expr : ℕ := 11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3

-- Assert the proof goal
theorem expr_value : expr = 21 := by
  sorry

end expr_value_l76_76574


namespace min_sum_of_factors_of_2310_l76_76853

theorem min_sum_of_factors_of_2310 : ∃ a b c : ℕ, a * b * c = 2310 ∧ a + b + c = 52 :=
by
  sorry

end min_sum_of_factors_of_2310_l76_76853


namespace possible_values_of_m_l76_76614

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem possible_values_of_m (m : ℝ) : (∀ x, S x m → P x) ↔ (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

end possible_values_of_m_l76_76614


namespace fraction_sum_is_one_l76_76740

theorem fraction_sum_is_one
    (a b c d w x y z : ℝ)
    (h1 : 17 * w + b * x + c * y + d * z = 0)
    (h2 : a * w + 29 * x + c * y + d * z = 0)
    (h3 : a * w + b * x + 37 * y + d * z = 0)
    (h4 : a * w + b * x + c * y + 53 * z = 0)
    (a_ne_17 : a ≠ 17)
    (b_ne_29 : b ≠ 29)
    (c_ne_37 : c ≠ 37)
    (wxyz_nonzero : w ≠ 0 ∨ x ≠ 0 ∨ y ≠ 0) :
    (a / (a - 17)) + (b / (b - 29)) + (c / (c - 37)) + (d / (d - 53)) = 1 := 
sorry

end fraction_sum_is_one_l76_76740


namespace lcm_852_1491_l76_76602

theorem lcm_852_1491 : Nat.lcm 852 1491 = 5961 := by
  sorry

end lcm_852_1491_l76_76602


namespace find_n_from_remainders_l76_76818

theorem find_n_from_remainders (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := 
by 
  -- proof needed here
  sorry

end find_n_from_remainders_l76_76818


namespace frank_more_miles_than_jim_in_an_hour_l76_76713

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end frank_more_miles_than_jim_in_an_hour_l76_76713


namespace fewer_noodles_than_pirates_l76_76899

theorem fewer_noodles_than_pirates 
  (P : ℕ) (N : ℕ) (h1 : P = 45) (h2 : N + P = 83) : P - N = 7 := by 
  sorry

end fewer_noodles_than_pirates_l76_76899


namespace solve_system_l76_76523

theorem solve_system (x y : ℝ) (h1 : 4 * x - y = 2) (h2 : 3 * x - 2 * y = -1) : x - y = -1 := 
by
  sorry

end solve_system_l76_76523


namespace evaluate_expression_l76_76045

theorem evaluate_expression (a : ℕ) (h : a = 2) : a^3 * a^4 = 128 := 
by
  sorry

end evaluate_expression_l76_76045


namespace travel_time_l76_76104

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end travel_time_l76_76104


namespace unique_digit_sum_is_21_l76_76468

theorem unique_digit_sum_is_21
  (Y E M T : ℕ)
  (YE ME : ℕ)
  (HT0 : YE = 10 * Y + E)
  (HT1 : ME = 10 * M + E)
  (H1 : YE * ME = 999)
  (H2 : Y ≠ E)
  (H3 : Y ≠ M)
  (H4 : Y ≠ T)
  (H5 : E ≠ M)
  (H6 : E ≠ T)
  (H7 : M ≠ T)
  (H8 : Y < 10)
  (H9 : E < 10)
  (H10 : M < 10)
  (H11 : T < 10) :
  Y + E + M + T = 21 :=
sorry

end unique_digit_sum_is_21_l76_76468


namespace fish_population_estimate_l76_76496

theorem fish_population_estimate :
  (∀ (x : ℕ),
    ∃ (m n k : ℕ), 
      m = 30 ∧
      k = 2 ∧
      n = 30 ∧
      ((k : ℚ) / n = m / x) → x = 450) :=
by
  sorry

end fish_population_estimate_l76_76496


namespace area_of_T_l76_76396

open Complex Real

noncomputable def omega := -1 / 2 + (1 / 2) * Complex.I * Real.sqrt 3
noncomputable def omega2 := -1 / 2 - (1 / 2) * Complex.I * Real.sqrt 3

def inT (z : ℂ) (a b c : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 ∧
  0 ≤ b ∧ b ≤ 1 ∧
  0 ≤ c ∧ c ≤ 1 ∧
  z = a + b * omega + c * omega2

theorem area_of_T : ∃ A : ℝ, A = 2 * Real.sqrt 3 :=
sorry

end area_of_T_l76_76396


namespace miles_driven_before_gas_stop_l76_76903

def total_distance : ℕ := 78
def distance_left : ℕ := 46

theorem miles_driven_before_gas_stop : total_distance - distance_left = 32 := by
  sorry

end miles_driven_before_gas_stop_l76_76903


namespace x_squared_minus_y_squared_l76_76333

theorem x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 9 / 16)
  (h2 : x - y = 5 / 16) :
  x^2 - y^2 = 45 / 256 :=
by
  sorry

end x_squared_minus_y_squared_l76_76333


namespace dig_days_l76_76097

theorem dig_days (m1 m2 : ℕ) (d1 d2 : ℚ) (k : ℚ) 
  (h1 : m1 * d1 = k) (h2 : m2 * d2 = k) : 
  m1 = 30 ∧ d1 = 6 ∧ m2 = 40 → d2 = 4.5 := 
by sorry

end dig_days_l76_76097


namespace maximize_total_profit_maximize_average_annual_profit_l76_76618

-- Define the profit function
def total_profit (x : ℤ) : ℤ := -x^2 + 18*x - 36

-- Define the average annual profit function
def average_annual_profit (x : ℤ) : ℤ :=
  let y := total_profit x
  y / x

-- Prove the maximum total profit
theorem maximize_total_profit : 
  ∃ x : ℤ, (total_profit x = 45) ∧ (x = 9) := 
  sorry

-- Prove the maximum average annual profit
theorem maximize_average_annual_profit : 
  ∃ x : ℤ, (average_annual_profit x = 6) ∧ (x = 6) :=
  sorry

end maximize_total_profit_maximize_average_annual_profit_l76_76618


namespace find_a_and_b_l76_76677

theorem find_a_and_b (a b : ℝ) 
  (curve : ∀ x : ℝ, y = x^2 + a * x + b) 
  (tangent : ∀ x : ℝ, y - b = a * x) 
  (tangent_line : ∀ x y : ℝ, x + y = 1) :
  a = -1 ∧ b = 1 := 
by 
  sorry

end find_a_and_b_l76_76677


namespace solve_for_y_l76_76904

theorem solve_for_y (y : ℤ) (h : 7 - y = 10) : y = -3 :=
sorry

end solve_for_y_l76_76904


namespace function_zero_solution_l76_76163

-- Define the statement of the problem
theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → ∀ y : ℝ, f (x ^ 2 + y) ≥ (1 / x + 1) * f y) →
  (∀ x : ℝ, f x = 0) :=
by
  -- The proof of this theorem will be inserted here.
  sorry

end function_zero_solution_l76_76163


namespace factor_expression_l76_76296

theorem factor_expression (x : ℝ) : 
  (21 * x ^ 4 + 90 * x ^ 3 + 40 * x - 10) - (7 * x ^ 4 + 6 * x ^ 3 + 8 * x - 6) = 
  2 * x * (7 * x ^ 3 + 42 * x ^ 2 + 16) - 4 :=
by sorry

end factor_expression_l76_76296


namespace gcd_689_1021_l76_76941

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 :=
by sorry

end gcd_689_1021_l76_76941


namespace area_of_circle_l76_76743

theorem area_of_circle (x y : ℝ) :
  (x^2 + y^2 - 8*x - 6*y = -9) → 
  (∃ (R : ℝ), (x - 4)^2 + (y - 3)^2 = R^2 ∧ π * R^2 = 16 * π) :=
by
  sorry

end area_of_circle_l76_76743


namespace total_stones_is_odd_l76_76719

variable (d : ℕ) (total_distance : ℕ)

theorem total_stones_is_odd (h1 : d = 10) (h2 : total_distance = 4800) :
  ∃ (N : ℕ), N % 2 = 1 ∧ total_distance = ((N - 1) * 2 * d) :=
by
  -- Let's denote the number of stones as N
  -- Given dx = 10 and total distance as 4800, we want to show that N is odd and 
  -- satisfies the equation: total_distance = ((N - 1) * 2 * d)
  sorry

end total_stones_is_odd_l76_76719


namespace initial_earning_members_l76_76894

theorem initial_earning_members (n : ℕ) (h1 : (n * 735) - ((n - 1) * 650) = 905) : n = 3 := by
  sorry

end initial_earning_members_l76_76894


namespace acute_angle_inequality_l76_76310

theorem acute_angle_inequality (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := 
sorry

end acute_angle_inequality_l76_76310


namespace triangle_area_l76_76085

theorem triangle_area (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * (a * b) = 24 := by
  sorry

end triangle_area_l76_76085


namespace circle_area_l76_76349

theorem circle_area (r : ℝ) (h : 5 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 5 := 
by
  sorry -- Proof is not required, placeholder for the actual proof

end circle_area_l76_76349


namespace fraction_planted_of_field_is_correct_l76_76797

/-- Given a right triangle with legs 5 units and 12 units, and a small unplanted square S
at the right-angle vertex such that the shortest distance from S to the hypotenuse is 3 units,
prove that the fraction of the field that is planted is 52761/857430. -/
theorem fraction_planted_of_field_is_correct :
  let area_triangle := (5 * 12) / 2
  let area_square := (180 / 169) ^ 2
  let area_planted := area_triangle - area_square
  let fraction_planted := area_planted / area_triangle
  fraction_planted = 52761 / 857430 :=
sorry

end fraction_planted_of_field_is_correct_l76_76797


namespace probability_of_two_hearts_and_three_diff_suits_l76_76703

def prob_two_hearts_and_three_diff_suits (n : ℕ) : ℚ :=
  if n = 5 then 135 / 1024 else 0

theorem probability_of_two_hearts_and_three_diff_suits :
  prob_two_hearts_and_three_diff_suits 5 = 135 / 1024 :=
by
  sorry

end probability_of_two_hearts_and_three_diff_suits_l76_76703


namespace dot_product_square_ABCD_l76_76118

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

def square_ABCD : Prop :=
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨2, 0⟩
  let C : Point := ⟨2, 2⟩
  let D : Point := ⟨0, 2⟩
  let E : Point := ⟨1, 0⟩  -- E is the midpoint of AB
  let EC := vector E C
  let ED := vector E D
  dot_product EC ED = 3

theorem dot_product_square_ABCD : square_ABCD := by
  sorry

end dot_product_square_ABCD_l76_76118


namespace average_age_in_club_l76_76491

theorem average_age_in_club (women men children : ℕ) 
    (avg_age_women avg_age_men avg_age_children : ℤ)
    (hw : women = 12) (hm : men = 18) (hc : children = 20)
    (haw : avg_age_women = 32) (ham : avg_age_men = 36) (hac : avg_age_children = 10) :
    (12 * 32 + 18 * 36 + 20 * 10) / (12 + 18 + 20) = 24 := by
  sorry

end average_age_in_club_l76_76491


namespace cube_root_simplification_l76_76912

theorem cube_root_simplification {a b : ℕ} (h : (a * b^(1/3) : ℝ) = (2450 : ℝ)^(1/3)) 
  (a_pos : 0 < a) (b_pos : 0 < b) (h_smallest : ∀ b', 0 < b' → (∃ a', (a' * b'^(1/3) : ℝ) = (2450 : ℝ)^(1/3) → b ≤ b')) :
  a + b = 37 := 
sorry

end cube_root_simplification_l76_76912


namespace exists_fifth_degree_polynomial_l76_76418

noncomputable def p (x : ℝ) : ℝ :=
  12.4 * (x^5 - 1.38 * x^3 + 0.38 * x)

theorem exists_fifth_degree_polynomial :
  (∃ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 ∧ -1 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ 
    p x1 = 1 ∧ p x2 = -1 ∧ p (-1) = 0 ∧ p 1 = 0) :=
  sorry

end exists_fifth_degree_polynomial_l76_76418


namespace base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l76_76450

theorem base_area_of_cone_with_slant_height_10_and_semi_lateral_surface :
  (l = 10) → (l = 2 * r) → (A = 25 * π) :=
  by
  intros l_eq_ten l_eq_two_r
  have r_is_five : r = 5 := by sorry
  have A_is_25pi : A = 25 * π := by sorry
  exact A_is_25pi

end base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l76_76450


namespace yang_hui_problem_l76_76503

theorem yang_hui_problem (x : ℝ) :
  x * (x + 12) = 864 :=
sorry

end yang_hui_problem_l76_76503


namespace rate_of_interest_l76_76684

-- Define the conditions
def P : ℝ := 1200
def SI : ℝ := 432
def T (R : ℝ) : ℝ := R

-- Define the statement to be proven
theorem rate_of_interest (R : ℝ) (h : SI = (P * R * T R) / 100) : R = 6 :=
by sorry

end rate_of_interest_l76_76684


namespace meal_arrangement_exactly_two_correct_l76_76595

noncomputable def meal_arrangement_count : ℕ :=
  let total_people := 13
  let meal_types := ["B", "B", "B", "B", "C", "C", "C", "F", "F", "F", "V", "V", "V"]
  let choose_2_people := (total_people.choose 2)
  let derangement_7 := 1854  -- Derangement of BBCCCVVV
  let derangement_9 := 133496  -- Derangement of BBCCFFFVV
  choose_2_people * (derangement_7 + derangement_9)

theorem meal_arrangement_exactly_two_correct : meal_arrangement_count = 10482600 := by
  sorry

end meal_arrangement_exactly_two_correct_l76_76595


namespace relationship_a_b_l76_76390

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 ∨ x ≥ 1 then 0 else -x^2 - x + 2

theorem relationship_a_b (a b : ℝ) (h_pos : a > 0) :
  (∀ x : ℝ, a * x + b = g x) → (2 * a < b ∧ b < (a + 1)^2 / 4 + 2 ∧ 0 < a ∧ a < 3) :=
sorry

end relationship_a_b_l76_76390


namespace necklaces_sold_correct_l76_76862

-- Define the given constants and conditions
def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earring_price : ℕ := 10
def ensemble_price : ℕ := 45
def bracelets_sold : ℕ := 10
def earrings_sold : ℕ := 20
def ensembles_sold : ℕ := 2
def total_revenue : ℕ := 565

-- Define the equation to calculate the total revenue
def total_revenue_calculation (N : ℕ) : ℕ :=
  (necklace_price * N) + (bracelet_price * bracelets_sold) + (earring_price * earrings_sold) + (ensemble_price * ensembles_sold)

-- Define the proof problem
theorem necklaces_sold_correct : 
  ∃ N : ℕ, total_revenue_calculation N = total_revenue ∧ N = 5 := by
  sorry

end necklaces_sold_correct_l76_76862


namespace ratio_of_votes_l76_76880

theorem ratio_of_votes (up_votes down_votes : ℕ) (h_up : up_votes = 18) (h_down : down_votes = 4) : (up_votes / Nat.gcd up_votes down_votes) = 9 ∧ (down_votes / Nat.gcd up_votes down_votes) = 2 :=
by
  sorry

end ratio_of_votes_l76_76880


namespace ordered_pair_of_positive_integers_l76_76742

theorem ordered_pair_of_positive_integers :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x^y + 4 = y^x) ∧ (3 * x^y = y^x + 10) ∧ (x = 7 ∧ y = 1) :=
by
  sorry

end ordered_pair_of_positive_integers_l76_76742


namespace initial_red_marbles_l76_76598

theorem initial_red_marbles
    (r g : ℕ)
    (h1 : 3 * r = 5 * g)
    (h2 : 2 * (r - 15) = g + 18) :
    r = 34 := by
  sorry

end initial_red_marbles_l76_76598


namespace part_I_solution_set_part_II_solution_range_l76_76697

-- Part I: Defining the function and proving the solution set for m = 3
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part_I_solution_set (x : ℝ) :
  (f x 3 ≥ 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
sorry

-- Part II: Proving the range of values for m such that f(x) ≥ 8 for any real number x
theorem part_II_solution_range (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
sorry

end part_I_solution_set_part_II_solution_range_l76_76697


namespace task_force_combinations_l76_76498

theorem task_force_combinations :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end task_force_combinations_l76_76498


namespace toucan_count_l76_76029

theorem toucan_count :
  (2 + 1 = 3) :=
by simp [add_comm]

end toucan_count_l76_76029


namespace algebraic_expression_value_l76_76465

variable (x : ℝ)

theorem algebraic_expression_value (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by
  -- This is where the detailed proof would go, but we are skipping it with sorry.
  sorry

end algebraic_expression_value_l76_76465


namespace find_ax5_by5_l76_76293

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 :=
sorry

end find_ax5_by5_l76_76293


namespace volume_increase_is_79_4_percent_l76_76110

noncomputable def original_volume (L B H : ℝ) : ℝ := L * B * H

noncomputable def new_volume (L B H : ℝ) : ℝ :=
  (L * 1.15) * (B * 1.30) * (H * 1.20)

noncomputable def volume_increase (L B H : ℝ) : ℝ :=
  new_volume L B H - original_volume L B H

theorem volume_increase_is_79_4_percent (L B H : ℝ) :
  volume_increase L B H = 0.794 * original_volume L B H := by
  sorry

end volume_increase_is_79_4_percent_l76_76110


namespace remainder_when_2x_div_8_is_1_l76_76487

theorem remainder_when_2x_div_8_is_1 (x y : ℤ) 
  (h1 : x = 11 * y + 4)
  (h2 : ∃ r : ℤ, 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) : ∃ r : ℤ, r = 1 :=
by
  sorry

end remainder_when_2x_div_8_is_1_l76_76487


namespace max_trig_expression_l76_76216

open Real

theorem max_trig_expression (x y z : ℝ) :
  (sin (2 * x) + sin y + sin (3 * z)) * (cos (2 * x) + cos y + cos (3 * z)) ≤ 4.5 := sorry

end max_trig_expression_l76_76216


namespace simplify_fraction_l76_76479

theorem simplify_fraction : (1 / (2 + (2/3))) = (3 / 8) :=
by
  sorry

end simplify_fraction_l76_76479


namespace find_original_number_l76_76758

theorem find_original_number (x : ℝ) 
  (h1 : x * 16 = 3408) 
  (h2 : 1.6 * 21.3 = 34.080000000000005) : 
  x = 213 :=
sorry

end find_original_number_l76_76758


namespace sin_cos_value_sin_plus_cos_value_l76_76252

noncomputable def given_condition (θ : ℝ) : Prop := 
  (Real.tan θ + 1 / Real.tan θ = 2)

theorem sin_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ * Real.cos θ = 1 / 2 :=
sorry

theorem sin_plus_cos_value (θ : ℝ) (h : given_condition θ) : 
  Real.sin θ + Real.cos θ = Real.sqrt 2 ∨ Real.sin θ + Real.cos θ = -Real.sqrt 2 :=
sorry

end sin_cos_value_sin_plus_cos_value_l76_76252


namespace range_of_k_l76_76243

theorem range_of_k (k : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧ 
   (x₁^3 - 3*x₁ = k ∧ x₂^3 - 3*x₂ = k ∧ x₃^3 - 3*x₃ = k)) ↔ (-2 < k ∧ k < 2) :=
sorry

end range_of_k_l76_76243


namespace range_of_x_l76_76966

noncomputable def y (x : ℝ) : ℝ := (x + 2) / (x - 1)

theorem range_of_x : ∀ x : ℝ, (y x ≠ 0) → x ≠ 1 := by
  intro x h
  sorry

end range_of_x_l76_76966


namespace lowest_positive_integer_divisible_by_primes_between_10_and_50_l76_76461

def primes_10_to_50 : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def lcm_list (lst : List ℕ) : ℕ :=
lst.foldr Nat.lcm 1

theorem lowest_positive_integer_divisible_by_primes_between_10_and_50 :
  lcm_list primes_10_to_50 = 614889782588491410 :=
by
  sorry

end lowest_positive_integer_divisible_by_primes_between_10_and_50_l76_76461


namespace cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l76_76564

theorem cos_beta_of_tan_alpha_and_sin_alpha_plus_beta 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 3) (h_sin_alpha_beta : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 10 / 10 := 
sorry

end cos_beta_of_tan_alpha_and_sin_alpha_plus_beta_l76_76564


namespace number_of_correct_answers_l76_76708

theorem number_of_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = 58 :=
sorry

end number_of_correct_answers_l76_76708


namespace evaluate_expression_l76_76016

theorem evaluate_expression : 8 - 5 * (9 - (4 - 2)^2) * 2 = -42 := by
  sorry

end evaluate_expression_l76_76016


namespace bricks_required_l76_76662

-- Definitions
def courtyard_length : ℕ := 20  -- in meters
def courtyard_breadth : ℕ := 16  -- in meters
def brick_length : ℕ := 20  -- in centimeters
def brick_breadth : ℕ := 10  -- in centimeters

-- Statement to prove
theorem bricks_required :
  ((courtyard_length * 100) * (courtyard_breadth * 100)) / (brick_length * brick_breadth) = 16000 :=
sorry

end bricks_required_l76_76662


namespace find_a_l76_76138

noncomputable def exists_nonconstant_function (a : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 ≠ f x2) ∧ 
  (∀ x : ℝ, f (a * x) = a^2 * f x) ∧
  (∀ x : ℝ, f (f x) = a * f x)

theorem find_a :
  ∀ (a : ℝ), exists_nonconstant_function a → (a = 0 ∨ a = 1) :=
by
  sorry

end find_a_l76_76138


namespace number_of_teams_l76_76430

-- Define the necessary conditions and variables
variable (n : ℕ)
variable (num_games : ℕ)

-- Define the condition that each team plays each other team exactly once 
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The main theorem to prove
theorem number_of_teams (h : total_games n = 91) : n = 14 :=
sorry

end number_of_teams_l76_76430


namespace domain_of_f_l76_76960

noncomputable def f (x k : ℝ) := (3 * x ^ 2 + 4 * x - 7) / (-7 * x ^ 2 + 4 * x + k)

theorem domain_of_f {x k : ℝ} (h : k < -4/7): ∀ x, -7 * x ^ 2 + 4 * x + k ≠ 0 :=
by 
  intro x
  sorry

end domain_of_f_l76_76960


namespace find_number_l76_76030

theorem find_number (x : ℤ) :
  45 - (x - (37 - (15 - 18))) = 57 → x = 28 :=
by
  sorry

end find_number_l76_76030


namespace revenue_fall_percentage_l76_76783

theorem revenue_fall_percentage:
  let oldRevenue := 72.0
  let newRevenue := 48.0
  (oldRevenue - newRevenue) / oldRevenue * 100 = 33.33 :=
by
  let oldRevenue := 72.0
  let newRevenue := 48.0
  sorry

end revenue_fall_percentage_l76_76783


namespace geometric_sequence_100th_term_l76_76780

theorem geometric_sequence_100th_term :
  ∀ (a₁ a₂ : ℤ) (r : ℤ), a₁ = 5 → a₂ = -15 → r = a₂ / a₁ → 
  (a₁ * r ^ 99 = -5 * 3 ^ 99) :=
by
  intros a₁ a₂ r ha₁ ha₂ hr
  sorry

end geometric_sequence_100th_term_l76_76780


namespace average_mark_of_first_class_is_40_l76_76340

open Classical

noncomputable def average_mark_first_class (n1 n2 : ℕ) (m2 : ℕ) (a : ℚ) : ℚ :=
  let x := (a * (n1 + n2) - n2 * m2) / n1
  x

theorem average_mark_of_first_class_is_40 : average_mark_first_class 30 50 90 71.25 = 40 := by
  sorry

end average_mark_of_first_class_is_40_l76_76340


namespace two_p_plus_q_l76_76926

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = 19 / 7 * q :=
by {
  sorry
}

end two_p_plus_q_l76_76926


namespace parabola_properties_l76_76849

-- Definitions of the conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def point_A (a b c : ℝ) : Prop := parabola a b c (-1) = 0
def point_B (a b c m : ℝ) : Prop := parabola a b c m = 0
def opens_downwards (a : ℝ) : Prop := a < 0
def valid_m (m : ℝ) : Prop := 1 < m ∧ m < 2

-- Conclusion ①
def conclusion_1 (a b : ℝ) : Prop := b > 0

-- Conclusion ②
def conclusion_2 (a c : ℝ) : Prop := 3 * a + 2 * c < 0

-- Conclusion ③
def conclusion_3 (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  x1 < x2 ∧ x1 + x2 > 1 ∧ parabola a b c x1 = y1 ∧ parabola a b c x2 = y2 → y1 > y2

-- Conclusion ④
def conclusion_4 (a b c : ℝ) : Prop :=
  a ≤ -1 → ∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 1) ∧ (a * x2^2 + b * x2 + c = 1) ∧ (x1 ≠ x2)

-- The theorem to prove
theorem parabola_properties (a b c m : ℝ) :
  (opens_downwards a) →
  (point_A a b c) →
  (point_B a b c m) →
  (valid_m m) →
  (conclusion_1 a b) ∧ (conclusion_2 a c → false) ∧ (∀ x1 x2 y1 y2, conclusion_3 a b c x1 x2 y1 y2) ∧ (conclusion_4 a b c) :=
by
  sorry

end parabola_properties_l76_76849


namespace simplify_fraction_l76_76604

theorem simplify_fraction (a b c : ℕ) (h1 : 222 = 2 * 111) (h2 : 999 = 3 * 333) (h3 : 111 = 3 * 37) :
  (222 / 999 * 111) = 74 :=
by
  sorry

end simplify_fraction_l76_76604


namespace problem_solution_l76_76837

def x : ℤ := -2 + 3
def y : ℤ := abs (-5)
def z : ℤ := 4 * (-1/4)

theorem problem_solution : x + y + z = 5 := 
by
  -- Definitions based on the problem statement
  have h1 : x = -2 + 3 := rfl
  have h2 : y = abs (-5) := rfl
  have h3 : z = 4 * (-1/4) := rfl
  
  -- Exact result required to be proved. Adding placeholder for steps.
  sorry

end problem_solution_l76_76837


namespace math_problem_l76_76848

theorem math_problem (x y : ℝ) (h1 : x + Real.sin y = 2023) (h2 : x + 2023 * Real.cos y = 2022) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2022 + Real.pi / 2 :=
sorry

end math_problem_l76_76848


namespace stock_percentage_l76_76074

theorem stock_percentage (investment income : ℝ) (investment total : ℝ) (P : ℝ) : 
  (income = 3800) → (total = 15200) → (income = (total * P) / 100) → P = 25 :=
by
  intros h1 h2 h3
  sorry

end stock_percentage_l76_76074


namespace fish_ratio_l76_76815

variables (O R B : ℕ)
variables (h1 : O = B + 25)
variables (h2 : B = 75)
variables (h3 : (O + B + R) / 3 = 75)

theorem fish_ratio : R / O = 1 / 2 :=
sorry

end fish_ratio_l76_76815


namespace calc_pow_expression_l76_76577

theorem calc_pow_expression : (27^3 * 9^2) / 3^15 = 1 / 9 := 
by sorry

end calc_pow_expression_l76_76577


namespace natalia_crates_l76_76441

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end natalia_crates_l76_76441


namespace fraction_simplification_l76_76129

theorem fraction_simplification : (98 / 210 : ℚ) = 7 / 15 := 
by 
  sorry

end fraction_simplification_l76_76129


namespace team_E_speed_l76_76635

noncomputable def average_speed_team_E (d t_E t_A v_A v_E : ℝ) : Prop :=
  d = 300 ∧
  t_A = t_E - 3 ∧
  v_A = v_E + 5 ∧
  d = v_E * t_E ∧
  d = v_A * t_A →
  v_E = 20

theorem team_E_speed : ∃ (v_E : ℝ), average_speed_team_E 300 t_E (t_E - 3) (v_E + 5) v_E :=
by
  sorry

end team_E_speed_l76_76635


namespace problem_l76_76176

-- Define sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y <= -1 }

-- Define set C as a function of a
def C (a : ℝ) : Set ℝ := { x | x < -a / 2 }

-- The statement of the problem: if B ⊆ C, then a < 2
theorem problem (a : ℝ) : (B ⊆ C a) → a < 2 :=
by sorry

end problem_l76_76176


namespace superhero_speed_l76_76276

def convert_speed (speed_mph : ℕ) (mile_to_km : ℚ) : ℚ :=
  let speed_kmh := (speed_mph : ℚ) * (1 / mile_to_km)
  speed_kmh / 60

theorem superhero_speed :
  convert_speed 36000 (6 / 10) = 1000 :=
by sorry

end superhero_speed_l76_76276


namespace rectangle_area_l76_76213

open Classical

noncomputable def point := {x : ℝ × ℝ // x.1 >= 0 ∧ x.2 >= 0}

structure Triangle :=
  (X Y Z : point)

structure Rectangle :=
  (P Q R S : point)

def height_from (t : Triangle) : ℝ :=
  8

def xz_length (t : Triangle) : ℝ :=
  15

def ps_on_xz (r : Rectangle) (t : Triangle) : Prop :=
  r.S.val.1 = r.P.val.1 ∧ r.S.val.1 = t.X.val.1 ∧ r.S.val.2 = 0 ∧ r.P.val.2 = 0

def pq_is_one_third_ps (r : Rectangle) : Prop :=
  dist r.P.1 r.Q.1 = (1/3) * dist r.P.1 r.S.1

theorem rectangle_area : ∀ (R : Rectangle) (T : Triangle),
  height_from T = 8 → xz_length T = 15 → ps_on_xz R T → pq_is_one_third_ps R →
  (dist R.P.1 R.Q.1) * (dist R.P.1 R.S.1) = 4800/169 :=
by
  intros
  sorry

end rectangle_area_l76_76213


namespace terminating_decimal_expansion_of_7_over_72_l76_76944

theorem terminating_decimal_expansion_of_7_over_72 : (7 / 72) = 0.175 := 
sorry

end terminating_decimal_expansion_of_7_over_72_l76_76944


namespace buoy_min_force_l76_76356

-- Define the problem in Lean
variables (M : ℝ) (ax : ℝ) (T_star : ℝ) (a : ℝ) (F_current : ℝ)
-- Conditions
variables (h_horizontal_component : T_star * Real.sin a = F_current)
          (h_zero_net_force : M * ax = 0)

theorem buoy_min_force (h_horizontal_component : T_star * Real.sin a = F_current) : 
  F_current = 400 := 
sorry

end buoy_min_force_l76_76356


namespace max_possible_b_l76_76973

theorem max_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 12 :=
by sorry

end max_possible_b_l76_76973


namespace tank_capacity_l76_76354

theorem tank_capacity (T : ℕ) (h1 : T > 0) 
    (h2 : (2 * T) / 5 + 15 + 20 = T - 25) : 
    T = 100 := 
  by 
    sorry

end tank_capacity_l76_76354


namespace radius_increase_rate_l76_76793

theorem radius_increase_rate (r : ℝ) (u : ℝ)
  (h : r = 20) (dS_dt : ℝ) (h_dS_dt : dS_dt = 10 * Real.pi) :
  u = 1 / 4 :=
by
  have S := Real.pi * r^2
  have dS_dt_eq : dS_dt = 2 * Real.pi * r * u := sorry
  rw [h_dS_dt, h] at dS_dt_eq
  exact sorry

end radius_increase_rate_l76_76793


namespace ab_value_l76_76124

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry

end ab_value_l76_76124


namespace resulting_curve_eq_l76_76279

def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

def transformed_curve (x y: ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), 
    is_on_circle x0 y0 ∧ 
    x = x0 ∧ 
    y = 4 * y0

theorem resulting_curve_eq : ∀ (x y : ℝ), transformed_curve x y → (x^2 / 9 + y^2 / 144 = 1) :=
by
  intros x y h
  sorry

end resulting_curve_eq_l76_76279


namespace residue_neg_437_mod_13_l76_76917

theorem residue_neg_437_mod_13 : (-437) % 13 = 5 :=
by
  sorry

end residue_neg_437_mod_13_l76_76917


namespace contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l76_76556

variable {A B : Prop}

def contrary (A : Prop) : Prop := A ∧ ¬A
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B)

theorem contrary_implies_mutually_exclusive (A : Prop) : contrary A → mutually_exclusive A (¬A) :=
by sorry

theorem contrary_sufficient_but_not_necessary (A B : Prop) :
  (∃ (A : Prop), contrary A) → mutually_exclusive A B →
  (∃ (A : Prop), contrary A ∧ mutually_exclusive A B) :=
by sorry

end contrary_implies_mutually_exclusive_contrary_sufficient_but_not_necessary_l76_76556


namespace values_of_n_for_replaced_constant_l76_76541

theorem values_of_n_for_replaced_constant (n : ℤ) (x : ℤ) :
  (∀ n : ℤ, 4 * n + x > 1 ∧ 4 * n + x < 60) → x = 8 → 
  (∀ n : ℤ, 4 * n + 8 > 1 ∧ 4 * n + 8 < 60) :=
by
  sorry

end values_of_n_for_replaced_constant_l76_76541


namespace max_10a_3b_15c_l76_76865

theorem max_10a_3b_15c (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) : 
  10 * a + 3 * b + 15 * c ≤ (Real.sqrt 337) / 6 := 
sorry

end max_10a_3b_15c_l76_76865


namespace initial_amount_of_liquid_A_l76_76083

theorem initial_amount_of_liquid_A (A B : ℕ) (x : ℕ) (h1 : 4 * x = A) (h2 : x = B) (h3 : 4 * x + x = 5 * x)
    (h4 : 4 * x - 8 = 3 * (x + 8) / 2) : A = 16 :=
  by
  sorry

end initial_amount_of_liquid_A_l76_76083


namespace articles_produced_l76_76064

theorem articles_produced (x y z w : ℕ) :
  (x ≠ 0) → (y ≠ 0) → (z ≠ 0) → (w ≠ 0) →
  ((x * x * x * (1 / x^2) = x) →
  y * z * w * (1 / x^2) = y * z * w / x^2) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end articles_produced_l76_76064


namespace find_a9_l76_76963

variable (a : ℕ → ℤ)
variable (h1 : a 2 = -3)
variable (h2 : a 3 = -5)
variable (d : ℤ := a 3 - a 2)

theorem find_a9 : a 9 = -17 :=
by
  sorry

end find_a9_l76_76963


namespace complement_A_is_interval_l76_76580

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def compl_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_is_interval : compl_U_A = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_is_interval_l76_76580


namespace sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l76_76407

theorem sin_C_eq_sqrt14_div_8 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  sinC = Real.sqrt 14 / 8 := 
by
  -- Proof is omitted
  sorry

theorem area_triangle_eq_sqrt7_div_4 (b c : ℝ) (cosB : ℝ) (h1 : b = Real.sqrt 2) (h2 : c = 1) (h3 : cosB = 3 / 4) : 
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := c * sinB / b
  let cosC := Real.sqrt (1 - sinC^2)
  let sinA := sinB * cosC + cosB * sinC
  let area := 1 / 2 * b * c * sinA
  area = Real.sqrt 7 / 4 := 
by
  -- Proof is omitted
  sorry

end sin_C_eq_sqrt14_div_8_area_triangle_eq_sqrt7_div_4_l76_76407


namespace sqrt_expression_non_negative_l76_76334

theorem sqrt_expression_non_negative (x : ℝ) : 4 + 2 * x ≥ 0 ↔ x ≥ -2 :=
by sorry

end sqrt_expression_non_negative_l76_76334


namespace parabola_midpoint_length_squared_l76_76111

theorem parabola_midpoint_length_squared :
  ∀ (A B : ℝ × ℝ), 
  (∃ (x y : ℝ), A = (x, 3*x^2 + 4*x + 2) ∧ B = (-x, -(3*x^2 + 4*x + 2)) ∧ ((A.1 + B.1) / 2 = 0) ∧ ((A.2 + B.2) / 2 = 0)) →
  dist A B^2 = 8 :=
by
  sorry

end parabola_midpoint_length_squared_l76_76111


namespace remaining_number_is_divisible_by_divisor_l76_76806

def initial_number : ℕ := 427398
def subtracted_number : ℕ := 8
def remaining_number : ℕ := initial_number - subtracted_number
def divisor : ℕ := 10

theorem remaining_number_is_divisible_by_divisor :
  remaining_number % divisor = 0 :=
by {
  sorry
}

end remaining_number_is_divisible_by_divisor_l76_76806


namespace range_of_alpha_l76_76700

variable {x : ℝ}

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_alpha (x : ℝ) (α : ℝ) (h : α = Real.arctan (3*x^2 - 1)) :
  α ∈ Set.Ico 0 (Real.pi / 2) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
sorry

end range_of_alpha_l76_76700


namespace arithmetic_problem_l76_76796

theorem arithmetic_problem : 72 * 1313 - 32 * 1313 = 52520 := by
  sorry

end arithmetic_problem_l76_76796


namespace find_distance_walker_l76_76777

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  (d = x * t) ∧
  (d = (x + 1) * (3 / 4) * t) ∧
  (d = (x - 1) * (t + 3))

theorem find_distance_walker (x t d : ℝ) (h : distance_walked x t d) : d = 18 := 
sorry

end find_distance_walker_l76_76777


namespace maria_chairs_l76_76514

variable (C : ℕ) -- Number of chairs Maria bought
variable (tables : ℕ := 2) -- Number of tables Maria bought is 2
variable (time_per_furniture : ℕ := 8) -- Time spent on each piece of furniture in minutes
variable (total_time : ℕ := 32) -- Total time spent assembling furniture

theorem maria_chairs :
  (time_per_furniture * C + time_per_furniture * tables = total_time) → C = 2 :=
by
  intro h
  sorry

end maria_chairs_l76_76514


namespace describe_T_correctly_l76_76287

def T (x y : ℝ) : Prop :=
(x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2)

theorem describe_T_correctly :
  (∀ x y : ℝ, T x y ↔
    ((x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2))) :=
by
  sorry

end describe_T_correctly_l76_76287


namespace sam_total_pennies_l76_76473

theorem sam_total_pennies : 
  ∀ (initial_pennies found_pennies total_pennies : ℕ),
  initial_pennies = 98 → 
  found_pennies = 93 → 
  total_pennies = initial_pennies + found_pennies → 
  total_pennies = 191 := by
  intros
  sorry

end sam_total_pennies_l76_76473


namespace pairs_satisfying_condition_l76_76140

theorem pairs_satisfying_condition :
  (∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 1000 ∧ 1 ≤ y ∧ y ≤ 1000 ∧ (x^2 + y^2) % 7 = 0) → 
  (∃ n : ℕ, n = 20164) :=
sorry

end pairs_satisfying_condition_l76_76140


namespace sequence_general_term_l76_76957

theorem sequence_general_term {a : ℕ → ℕ} (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 1, a (n + 1) = a n + 2) :
  ∀ n : ℕ, n ≥ 1 → a n = 2 * n - 1 :=
by
  -- skip the proof with sorry
  sorry

end sequence_general_term_l76_76957


namespace unique_value_of_n_l76_76976

theorem unique_value_of_n
  (n t : ℕ) (h1 : t ≠ 0)
  (h2 : 15 * t + (n - 20) * t / 3 = (n * t) / 2) :
  n = 50 :=
by sorry

end unique_value_of_n_l76_76976


namespace number_of_people_l76_76217

-- Define the total number of candy bars
def total_candy_bars : ℝ := 5.0

-- Define the amount of candy each person gets
def candy_per_person : ℝ := 1.66666666699999

-- Define a theorem to state that dividing the total candy bars by candy per person gives 3 people
theorem number_of_people : total_candy_bars / candy_per_person = 3 :=
  by
  -- Proof omitted
  sorry

end number_of_people_l76_76217


namespace problem_solutions_l76_76049

theorem problem_solutions (a b c : ℝ) (h : ∀ x, ax^2 + bx + c ≤ 0 ↔ x ≤ -4 ∨ x ≥ 3) :
  (a + b + c > 0) ∧ (∀ x, bx + c > 0 ↔ x < 12) :=
by
  -- The following proof steps are not needed as per the instructions provided
  sorry

end problem_solutions_l76_76049


namespace arithmetic_expression_eval_l76_76485

theorem arithmetic_expression_eval : 
  (1000 * 0.09999) / 10 * 999 = 998001 := 
by 
  sorry

end arithmetic_expression_eval_l76_76485


namespace dark_squares_exceed_light_squares_by_one_l76_76212

theorem dark_squares_exceed_light_squares_by_one :
  let dark_squares := 25
  let light_squares := 24
  dark_squares - light_squares = 1 :=
by
  sorry

end dark_squares_exceed_light_squares_by_one_l76_76212


namespace correct_average_is_19_l76_76353

-- Definitions
def incorrect_avg : ℕ := 16
def num_values : ℕ := 10
def incorrect_reading : ℕ := 25
def correct_reading : ℕ := 55

-- Theorem to prove
theorem correct_average_is_19 :
  ((incorrect_avg * num_values - incorrect_reading + correct_reading) / num_values) = 19 :=
by
  sorry

end correct_average_is_19_l76_76353


namespace proof_p_and_q_true_l76_76027

open Real

def p : Prop := ∃ x : ℝ, x - 2 > log x
def q : Prop := ∀ x : ℝ, exp x > x

theorem proof_p_and_q_true : p ∧ q :=
by
  -- Assume you have already proven that p and q are true separately
  sorry

end proof_p_and_q_true_l76_76027


namespace evaluate_expression_l76_76482

theorem evaluate_expression : 
  3 * (-4) - ((5 * (-5)) * (-2)) + 6 = -56 := 
by 
  sorry

end evaluate_expression_l76_76482


namespace probability_all_red_is_correct_l76_76032

def total_marbles (R W B : Nat) : Nat := R + W + B

def first_red_probability (R W B : Nat) : Rat := R / total_marbles R W B
def second_red_probability (R W B : Nat) : Rat := (R - 1) / (total_marbles R W B - 1)
def third_red_probability (R W B : Nat) : Rat := (R - 2) / (total_marbles R W B - 2)

def all_red_probability (R W B : Nat) : Rat := 
  first_red_probability R W B * 
  second_red_probability R W B * 
  third_red_probability R W B

theorem probability_all_red_is_correct 
  (R W B : Nat) (hR : R = 5) (hW : W = 6) (hB : B = 7) :
  all_red_probability R W B = 5 / 408 := by
  sorry

end probability_all_red_is_correct_l76_76032


namespace sine_thirteen_pi_over_six_l76_76076

theorem sine_thirteen_pi_over_six : Real.sin ((13 * Real.pi) / 6) = 1 / 2 := by
  sorry

end sine_thirteen_pi_over_six_l76_76076


namespace pow_equation_sum_l76_76568

theorem pow_equation_sum (x y : ℕ) (hx : 2 ^ 11 * 6 ^ 5 = 4 ^ x * 3 ^ y) : x + y = 13 :=
  sorry

end pow_equation_sum_l76_76568


namespace sheena_completes_in_37_weeks_l76_76515

-- Definitions based on the conditions
def hours_per_dress : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_cycle : List Nat := [5, 3, 6, 4]
def finalize_hours : Nat := 10

-- The total hours needed to sew all dresses
def total_dress_hours : Nat := hours_per_dress.sum

-- The total hours needed including finalizing hours
def total_hours : Nat := total_dress_hours + finalize_hours

-- Total hours sewed in each 4-week cycle
def hours_per_cycle : Nat := hours_cycle.sum

-- Total number of weeks it will take to complete all dresses
def weeks_needed : Nat := 4 * ((total_hours + hours_per_cycle - 1) / hours_per_cycle)
def additional_weeks : Nat := if total_hours % hours_per_cycle == 0 then 0 else 1

theorem sheena_completes_in_37_weeks : weeks_needed + additional_weeks = 37 := by
  sorry

end sheena_completes_in_37_weeks_l76_76515


namespace monotonic_decreasing_interval_l76_76593

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def decreasing_interval (a b : ℝ) := 
  ∀ x : ℝ, a < x ∧ x < b → deriv f x < 0

theorem monotonic_decreasing_interval : decreasing_interval 0 1 :=
sorry

end monotonic_decreasing_interval_l76_76593


namespace valid_x_for_sqrt_l76_76682

theorem valid_x_for_sqrt (x : ℝ) (hx : x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) : x ≥ 2 ↔ x = 3 := 
sorry

end valid_x_for_sqrt_l76_76682


namespace relation_between_x_and_y_l76_76416

open Real

noncomputable def x (t : ℝ) : ℝ := t^(1 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 1))

theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : (y t)^(x t) = (x t)^(y t) :=
by sorry

end relation_between_x_and_y_l76_76416


namespace square_side_length_range_l76_76542

theorem square_side_length_range (a : ℝ) (h : a^2 = 30) : 5.4 < a ∧ a < 5.5 :=
sorry

end square_side_length_range_l76_76542


namespace geometric_series_sum_l76_76520

theorem geometric_series_sum :
  ∑' n : ℕ, (2 : ℝ) * (1 / 4) ^ n = 8 / 3 := by
  sorry

end geometric_series_sum_l76_76520


namespace Gunther_free_time_left_l76_76764

def vacuuming_time := 45
def dusting_time := 60
def folding_laundry_time := 25
def mopping_time := 30
def cleaning_bathroom_time := 40
def wiping_windows_time := 15
def brushing_cats_time := 4 * 5
def washing_dishes_time := 20
def first_tasks_total_time := 2 * 60 + 30
def available_free_time := 5 * 60

theorem Gunther_free_time_left : 
  (available_free_time - 
   (vacuuming_time + dusting_time + folding_laundry_time + 
    mopping_time + cleaning_bathroom_time + 
    wiping_windows_time + brushing_cats_time + 
    washing_dishes_time) = 45) := 
by 
  sorry

end Gunther_free_time_left_l76_76764


namespace range_of_z_l76_76021

theorem range_of_z (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : -2 < b) (h4 : b < -1) :
  5 < 2 * a - b ∧ 2 * a - b < 8 :=
by
  sorry

end range_of_z_l76_76021


namespace volcano_ash_height_l76_76841

theorem volcano_ash_height (r d : ℝ) (h : r = 2700) (h₁ : 2 * r = 18 * d) : d = 300 :=
by
  sorry

end volcano_ash_height_l76_76841


namespace min_value_of_n_l76_76486

theorem min_value_of_n : 
  ∃ (n : ℕ), (∃ r : ℕ, 4 * n - 7 * r = 0) ∧ n = 7 := 
sorry

end min_value_of_n_l76_76486


namespace class_sizes_l76_76634

theorem class_sizes
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (garcia_students : ℕ)
  (smith_students : ℕ)
  (h1 : finley_students = 24)
  (h2 : johnson_students = 10 + finley_students / 2)
  (h3 : garcia_students = 2 * johnson_students)
  (h4 : smith_students = finley_students / 3) :
  finley_students = 24 ∧ johnson_students = 22 ∧ garcia_students = 44 ∧ smith_students = 8 :=
by
  sorry

end class_sizes_l76_76634


namespace part1_solve_inequality_part2_range_of_a_l76_76022

def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1_solve_inequality (x : ℝ) (h : -2 < x ∧ x < -2/3) :
    f x 1 > 1 :=
by
  sorry

theorem part2_range_of_a (h : ∀ x, 2 ≤ x ∧ x ≤ 3 → f x (a : ℝ) > 0) :
    -5/2 < a ∧ a < -2 :=
by
  sorry

end part1_solve_inequality_part2_range_of_a_l76_76022


namespace ratio_cost_to_marked_l76_76760

variable (m : ℝ)

def marked_price (m : ℝ) := m

def selling_price (m : ℝ) : ℝ := 0.75 * m

def cost_price (m : ℝ) : ℝ := 0.60 * selling_price m

theorem ratio_cost_to_marked (m : ℝ) : 
  cost_price m / marked_price m = 0.45 := 
by
  sorry

end ratio_cost_to_marked_l76_76760


namespace find_f_2017_l76_76488

noncomputable def f (x : ℤ) (a α b β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2017
(x : ℤ)
(a α b β : ℝ)
(h : f 4 a α b β = 3) :
f 2017 a α b β = -3 := 
sorry

end find_f_2017_l76_76488


namespace correct_equation_l76_76727

-- Define the initial deposit
def initial_deposit : ℝ := 2500

-- Define the total amount after one year with interest tax deducted
def total_amount : ℝ := 2650

-- Define the annual interest rate
variable (x : ℝ)

-- Define the interest tax rate
def interest_tax_rate : ℝ := 0.20

-- Define the equation for the total amount after one year considering the tax
theorem correct_equation :
  initial_deposit * (1 + (1 - interest_tax_rate) * x) = total_amount :=
sorry

end correct_equation_l76_76727


namespace solve_inequality_l76_76629

theorem solve_inequality :
  {x : ℝ | 0 ≤ x ∧ x ≤ 1 } = {x : ℝ | x * (x - 1) ≤ 0} :=
by sorry

end solve_inequality_l76_76629


namespace joker_probability_l76_76891

-- Definition of the problem parameters according to the conditions
def total_cards := 54
def jokers := 2

-- Calculate the probability
def probability (favorable : Nat) (total : Nat) : ℚ :=
  favorable / total

-- State the theorem that we want to prove
theorem joker_probability : probability jokers total_cards = 1 / 27 := by
  sorry

end joker_probability_l76_76891


namespace quadratic_function_integer_values_not_imply_integer_coefficients_l76_76804

theorem quadratic_function_integer_values_not_imply_integer_coefficients :
  ∃ (a b c : ℚ), (∀ x : ℤ, ∃ y : ℤ, (a * (x : ℚ)^2 + b * (x : ℚ) + c = (y : ℚ))) ∧
    (¬ (∃ (a_int b_int c_int : ℤ), a = (a_int : ℚ) ∧ b = (b_int : ℚ) ∧ c = (c_int : ℚ))) :=
by
  sorry

end quadratic_function_integer_values_not_imply_integer_coefficients_l76_76804


namespace arithmetic_sequence_sum_ratio_l76_76248

variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Definition of arithmetic sequence sum
def arithmeticSum (n : ℕ) : ℚ :=
  (n / 2) * (a 1 + a n)

-- Given condition
axiom condition : (a 6) / (a 5) = 9 / 11

theorem arithmetic_sequence_sum_ratio :
  (S 11) / (S 9) = 1 :=
by
  sorry

end arithmetic_sequence_sum_ratio_l76_76248


namespace vector_t_solution_l76_76959

theorem vector_t_solution (t : ℝ) :
  ∃ t, (∃ (AB AC BC : ℝ × ℝ), 
         AB = (t, 1) ∧ AC = (2, 2) ∧ BC = (2 - t, 1) ∧ 
         (AC.1 - AB.1) * AC.1 + (AC.2 - AB.2) * AC.2 = 0 ) → 
         t = 3 :=
by {
  sorry -- proof content omitted as per instructions
}

end vector_t_solution_l76_76959


namespace product_of_2020_numbers_even_l76_76234

theorem product_of_2020_numbers_even (a : ℕ → ℕ) 
  (h : (Finset.sum (Finset.range 2020) a) % 2 = 1) : 
  (Finset.prod (Finset.range 2020) a) % 2 = 0 :=
sorry

end product_of_2020_numbers_even_l76_76234


namespace plates_are_multiple_of_eleven_l76_76469

theorem plates_are_multiple_of_eleven
    (P : ℕ)    -- Number of plates
    (S : ℕ := 33)    -- Number of spoons
    (g : ℕ := 11)    -- Greatest number of groups
    (hS : S % g = 0)    -- Condition: All spoons can be divided into these groups evenly
    (hP : ∀ (k : ℕ), P = k * g) : ∃ x : ℕ, P = 11 * x :=
by
  sorry

end plates_are_multiple_of_eleven_l76_76469


namespace wrench_turns_bolt_l76_76581

theorem wrench_turns_bolt (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (Real.sqrt 3 / Real.sqrt 2 < b / a) ∧ (b / a ≤ 3 - Real.sqrt 3) :=
sorry

end wrench_turns_bolt_l76_76581


namespace correct_operation_l76_76883

theorem correct_operation (x y : ℝ) : (-x - y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 :=
sorry

end correct_operation_l76_76883


namespace sum_of_six_consecutive_odd_numbers_l76_76724

theorem sum_of_six_consecutive_odd_numbers (a b c d e f : ℕ) 
  (ha : 135135 = a * b * c * d * e * f)
  (hb : a < b) (hc : b < c) (hd : c < d) (he : d < e) (hf : e < f)
  (hzero : a % 2 = 1) (hone : b % 2 = 1) (htwo : c % 2 = 1) 
  (hthree : d % 2 = 1) (hfour : e % 2 = 1) (hfive : f % 2 = 1) :
  a + b + c + d + e + f = 48 := by
  sorry

end sum_of_six_consecutive_odd_numbers_l76_76724


namespace boat_distance_against_stream_l76_76669

-- Define the conditions
variable (v_s : ℝ)
variable (speed_still_water : ℝ := 9)
variable (distance_downstream : ℝ := 13)

-- Assert the given condition
axiom condition : speed_still_water + v_s = distance_downstream

-- Prove the required distance against the stream
theorem boat_distance_against_stream : (speed_still_water - (distance_downstream - speed_still_water)) = 5 :=
by
  sorry

end boat_distance_against_stream_l76_76669


namespace kenya_peanuts_count_l76_76474

def peanuts_jose : ℕ := 85
def diff_kenya_jose : ℕ := 48
def peanuts_kenya : ℕ := peanuts_jose + diff_kenya_jose

theorem kenya_peanuts_count : peanuts_kenya = 133 := 
by
  -- proof goes here
  sorry

end kenya_peanuts_count_l76_76474


namespace system_of_equations_solution_l76_76860

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x - 2 * y = 1)
  (h2 : 3 * x + 4 * y = 23) :
  x = 5 ∧ y = 2 :=
sorry

end system_of_equations_solution_l76_76860


namespace rectangle_area_3650_l76_76619

variables (L B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := L - B = 23
def condition2 : Prop := 2 * (L + B) = 246

-- Prove that the area of the rectangle is 3650 m² given the conditions
theorem rectangle_area_3650 (h1 : condition1 L B) (h2 : condition2 L B) : L * B = 3650 := by
  sorry

end rectangle_area_3650_l76_76619


namespace find_percentage_decrease_l76_76120

noncomputable def initialPrice : ℝ := 100
noncomputable def priceAfterJanuary : ℝ := initialPrice * 1.30
noncomputable def priceAfterFebruary : ℝ := priceAfterJanuary * 0.85
noncomputable def priceAfterMarch : ℝ := priceAfterFebruary * 1.10

theorem find_percentage_decrease :
  ∃ (y : ℝ), (priceAfterMarch * (1 - y / 100) = initialPrice) ∧ abs (y - 18) < 1 := 
sorry

end find_percentage_decrease_l76_76120


namespace time_for_D_to_complete_job_l76_76609

-- Definitions for conditions
def A_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 4

-- We need to find D_rate
def D_rate : ℚ := combined_rate - A_rate

-- Now we state the theorem
theorem time_for_D_to_complete_job :
  D_rate = 1 / 12 :=
by
  /-
  We want to show that given the conditions:
  1. A_rate = 1 / 6
  2. A_rate + D_rate = 1 / 4
  it results in D_rate = 1 / 12.
  -/
  sorry

end time_for_D_to_complete_job_l76_76609


namespace equation_of_symmetric_line_l76_76228

theorem equation_of_symmetric_line
  (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  (∀ x : ℝ, ∃ y : ℝ, (x = a * y + b)) → (∀ x : ℝ, ∃ y : ℝ, (y = (1/a) * x - (b/a))) :=
by
  sorry

end equation_of_symmetric_line_l76_76228


namespace minimum_a_l76_76185

noncomputable def func (t a : ℝ) := 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5

theorem minimum_a (a : ℝ) (h: ∀ t ≥ 0, func t a ≥ 24) :
  a = 2 * Real.sqrt ((24 / 7) ^ 7) :=
sorry

end minimum_a_l76_76185


namespace area_of_field_with_tomatoes_l76_76190

theorem area_of_field_with_tomatoes :
  let length := 3.6
  let width := 2.5 * length
  let total_area := length * width
  let area_with_tomatoes := total_area / 2
  area_with_tomatoes = 16.2 :=
by
  sorry

end area_of_field_with_tomatoes_l76_76190


namespace min_value_fraction_expression_l76_76930

theorem min_value_fraction_expression {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := 
by
  sorry

end min_value_fraction_expression_l76_76930


namespace max_students_distribute_eq_pens_pencils_l76_76053

theorem max_students_distribute_eq_pens_pencils (n_pens n_pencils n : ℕ) (h_pens : n_pens = 890) (h_pencils : n_pencils = 630) :
  (∀ k : ℕ, k > n → (n_pens % k ≠ 0 ∨ n_pencils % k ≠ 0)) → (n = Nat.gcd n_pens n_pencils) := by
  sorry

end max_students_distribute_eq_pens_pencils_l76_76053


namespace back_seat_tickets_sold_l76_76484

variable (M B : ℕ)

theorem back_seat_tickets_sold:
  M + B = 20000 ∧ 55 * M + 45 * B = 955000 → B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l76_76484


namespace weight_of_rod_l76_76936

theorem weight_of_rod (length1 length2 weight1 weight2 weight_per_meter : ℝ)
  (h1 : length1 = 6) (h2 : weight1 = 22.8) (h3 : length2 = 11.25)
  (h4 : weight_per_meter = weight1 / length1) :
  weight2 = weight_per_meter * length2 :=
by
  -- The proof would go here
  sorry

end weight_of_rod_l76_76936


namespace arithmetic_sequence_k_value_l76_76937

theorem arithmetic_sequence_k_value (a : ℕ → ℤ) (S: ℕ → ℤ)
    (h1 : ∀ n, S (n + 1) = S n + a (n + 1))
    (h2 : S 11 = S 4)
    (h3 : a 1 = 1)
    (h4 : ∃ k, a k + a 4 = 0) :
    ∃ k, k = 12 :=
by 
  sorry

end arithmetic_sequence_k_value_l76_76937


namespace solution_l76_76907

theorem solution (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) 
  : x * y * z = 8 := 
by sorry

end solution_l76_76907


namespace find_a_l76_76991

theorem find_a (a : ℝ) (x : ℝ) (h₀ : a > 0) (h₁ : x > 0)
  (h₂ : a * Real.sqrt x = Real.log (Real.sqrt x))
  (h₃ : (a / (2 * Real.sqrt x)) = (1 / (2 * x))) : a = Real.exp (-1) :=
by
  sorry

end find_a_l76_76991


namespace amount_of_brown_paint_l76_76668

-- Definition of the conditions
def white_paint : ℕ := 20
def green_paint : ℕ := 15
def total_paint : ℕ := 69

-- Theorem statement for the amount of brown paint
theorem amount_of_brown_paint : (total_paint - (white_paint + green_paint)) = 34 :=
by
  sorry

end amount_of_brown_paint_l76_76668


namespace f_g_x_eq_l76_76934

noncomputable def f (x : ℝ) : ℝ := (x * (x + 1)) / 3
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem f_g_x_eq (x : ℝ) : f (g x) = (x^2 + 7*x + 12) / 3 := by
  sorry

end f_g_x_eq_l76_76934


namespace maximum_monthly_profit_l76_76922

-- Let's set up our conditions

def selling_price := 25
def monthly_profit := 120
def cost_price := 20
def selling_price_threshold := 32
def relationship (x n : ℝ) := -10 * x + n

-- Define the value of n
def value_of_n : ℝ := 370

-- Profit function
def profit_function (x n : ℝ) : ℝ := (x - cost_price) * (relationship x n)

-- Define the condition for maximum profit where the selling price should be higher than 32
def max_profit_condition (n : ℝ) (x : ℝ) := x > selling_price_threshold

-- Define what the maximum profit should be
def max_profit := 160

-- The main theorem to be proven
theorem maximum_monthly_profit :
  (relationship selling_price value_of_n = monthly_profit) →
  max_profit_condition value_of_n 32 →
  profit_function 32 value_of_n = max_profit :=
by sorry

end maximum_monthly_profit_l76_76922


namespace difference_30th_28th_triangular_l76_76086

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_30th_28th_triangular :
  triangular_number 30 - triangular_number 28 = 59 :=
by
  sorry

end difference_30th_28th_triangular_l76_76086


namespace evaluate_expression_l76_76543

theorem evaluate_expression (x : ℕ) (h : x = 3) : 5^3 - 2^x * 3 + 4^2 = 117 :=
by
  rw [h]
  sorry

end evaluate_expression_l76_76543


namespace width_of_room_l76_76670

-- Define the givens
def length_of_room : ℝ := 5.5
def total_cost : ℝ := 20625
def rate_per_sq_meter : ℝ := 1000

-- Define the required proof statement
theorem width_of_room : (total_cost / rate_per_sq_meter) / length_of_room = 3.75 :=
by
  sorry

end width_of_room_l76_76670


namespace find_x_minus_y_l76_76037

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end find_x_minus_y_l76_76037


namespace find_monotonic_function_l76_76993

-- Define Jensen's functional equation property
def jensens_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

-- Define monotonicity property
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The main theorem stating the equivalence
theorem find_monotonic_function (f : ℝ → ℝ) (h₁ : jensens_eq f) (h₂ : monotonic f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := 
sorry

end find_monotonic_function_l76_76993


namespace smallest_b_theorem_l76_76343

open Real

noncomputable def smallest_b (a b c: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) : Prop :=
  b = 5

theorem smallest_b_theorem (a b c: ℝ) (r: ℝ) (h1: b > 0) (h2: a = b / r) (h3: c = b * r) (h4: a * b * c = 125) :
  smallest_b a b c h1 h2 h3 h4 :=
by {
  sorry
}

end smallest_b_theorem_l76_76343


namespace no_nat_solution_l76_76947

theorem no_nat_solution (x y z : ℕ) : ¬ (x^3 + 2 * y^3 = 4 * z^3) :=
sorry

end no_nat_solution_l76_76947


namespace find_x1_l76_76970

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3) : 
  x1 = 4 / 5 := 
  sorry

end find_x1_l76_76970


namespace derivative_at_one_max_value_l76_76313

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Prove that f'(1) = 0
theorem derivative_at_one : deriv f 1 = 0 :=
by sorry

-- Prove that the maximum value of f(x) is 2
theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 2 :=
by sorry

end derivative_at_one_max_value_l76_76313


namespace find_k_l76_76307

theorem find_k (k : ℝ) (h_line : ∀ x y : ℝ, 3 * x + 5 * y + k = 0)
    (h_sum_intercepts : - (k / 3) - (k / 5) = 16) : k = -30 := by
  sorry

end find_k_l76_76307


namespace total_spending_l76_76928

-- Conditions
def pop_spending : ℕ := 15
def crackle_spending : ℕ := 3 * pop_spending
def snap_spending : ℕ := 2 * crackle_spending

-- Theorem stating the total spending
theorem total_spending : snap_spending + crackle_spending + pop_spending = 150 :=
by
  sorry

end total_spending_l76_76928


namespace part_a_part_b_part_c_l76_76982

-- Part (a)
theorem part_a : 
  ∃ n : ℕ, n = 2023066 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (b)
theorem part_b : 
  ∃ n : ℕ, n = 1006 ∧ (∃ x y z : ℕ, x + y + z = 2013 ∧ x = y ∧ x > 0 ∧ y > 0 ∧ z > 0) :=
sorry

-- Part (c)
theorem part_c : 
  ∃ (x y z : ℕ), (x + y + z = 2013 ∧ (x * y * z = 671 * 671 * 671)) :=
sorry

end part_a_part_b_part_c_l76_76982


namespace minimum_value_expr_l76_76051

theorem minimum_value_expr (x y : ℝ) : 
  ∃ (a b : ℝ), 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 = 2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ∧ 
  2 * (a - 3)^2 + 3 * (b + 1)^2 + 4 ≥ 4 :=
by 
  sorry

end minimum_value_expr_l76_76051


namespace reciprocal_of_sum_l76_76041

theorem reciprocal_of_sum :
  (1 / ((3 : ℚ) / 4 + (5 : ℚ) / 6)) = (12 / 19) :=
by
  sorry

end reciprocal_of_sum_l76_76041


namespace rabbits_and_raccoons_l76_76142

variable (b_r t_r x : ℕ)

theorem rabbits_and_raccoons : 
  2 * b_r = x ∧ 3 * t_r = x ∧ b_r = t_r + 3 → x = 18 := 
by
  sorry

end rabbits_and_raccoons_l76_76142


namespace mary_potatoes_l76_76859

theorem mary_potatoes (original new_except : ℕ) (h₁ : original = 25) (h₂ : new_except = 7) :
  original + new_except = 32 := by
  sorry

end mary_potatoes_l76_76859


namespace max_value_of_xy_expression_l76_76611

theorem max_value_of_xy_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y < 60) : 
  xy * (60 - 4 * x - 3 * y) ≤ 2000 / 3 := 
sorry

end max_value_of_xy_expression_l76_76611


namespace solution_set_inequality_l76_76214

theorem solution_set_inequality (t : ℝ) (ht : 0 < t ∧ t < 1) :
  {x : ℝ | x^2 - (t + t⁻¹) * x + 1 < 0} = {x : ℝ | t < x ∧ x < t⁻¹} :=
sorry

end solution_set_inequality_l76_76214


namespace combined_selling_price_l76_76004

theorem combined_selling_price :
  let cost_price_A := 180
  let profit_percent_A := 0.15
  let cost_price_B := 220
  let profit_percent_B := 0.20
  let cost_price_C := 130
  let profit_percent_C := 0.25
  let selling_price_A := cost_price_A * (1 + profit_percent_A)
  let selling_price_B := cost_price_B * (1 + profit_percent_B)
  let selling_price_C := cost_price_C * (1 + profit_percent_C)
  selling_price_A + selling_price_B + selling_price_C = 633.50 := by
  sorry

end combined_selling_price_l76_76004


namespace sum_g_h_l76_76256

theorem sum_g_h (d g h : ℝ) 
  (h1 : (8 * d^2 - 4 * d + g) * (4 * d^2 + h * d + 7) = 32 * d^4 + (4 * h - 16) * d^3 - (14 * d^2 - 28 * d - 56)) :
  g + h = -8 :=
sorry

end sum_g_h_l76_76256


namespace students_and_confucius_same_arrival_time_l76_76168

noncomputable def speed_of_students_walking (x : ℝ) : ℝ := x

noncomputable def speed_of_bullock_cart (x : ℝ) : ℝ := 1.5 * x

noncomputable def time_for_students_to_school (x : ℝ) : ℝ := 30 / x

noncomputable def time_for_confucius_to_school (x : ℝ) : ℝ := 30 / (1.5 * x) + 1

theorem students_and_confucius_same_arrival_time (x : ℝ) (h1 : 0 < x) :
  30 / x = 30 / (1.5 * x) + 1 :=
by
  sorry

end students_and_confucius_same_arrival_time_l76_76168


namespace subway_train_speed_l76_76164

theorem subway_train_speed (s : ℕ) (h1 : 0 ≤ s ∧ s ≤ 7) (h2 : s^2 + 2*s = 63) : s = 7 :=
by
  sorry

end subway_train_speed_l76_76164


namespace joe_used_paint_total_l76_76426

theorem joe_used_paint_total :
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  total_first_airport + total_second_airport = 415 :=
by
  let first_airport_paint := 360
  let second_airport_paint := 600
  let first_week_first_airport := (1/4 : ℝ) * first_airport_paint
  let remaining_first_airport := first_airport_paint - first_week_first_airport
  let second_week_first_airport := (1/6 : ℝ) * remaining_first_airport
  let total_first_airport := first_week_first_airport + second_week_first_airport
  let first_week_second_airport := (1/3 : ℝ) * second_airport_paint
  let remaining_second_airport := second_airport_paint - first_week_second_airport
  let second_week_second_airport := (1/5 : ℝ) * remaining_second_airport
  let total_second_airport := first_week_second_airport + second_week_second_airport
  show total_first_airport + total_second_airport = 415
  sorry

end joe_used_paint_total_l76_76426


namespace circles_exceeding_n_squared_l76_76428

noncomputable def num_circles (n : ℕ) : ℕ :=
  if n >= 8 then 
    5 * n + 4 * (n - 1)
  else 
    n * n

theorem circles_exceeding_n_squared (n : ℕ) (hn : n ≥ 8) : num_circles n > n^2 := 
by {
  sorry
}

end circles_exceeding_n_squared_l76_76428


namespace mat_weaves_problem_l76_76900

theorem mat_weaves_problem (S1 S2: ℕ) (days1 days2: ℕ) (mats1 mats2: ℕ) (H1: S1 = 1)
    (H2: S2 = 8) (H3: days1 = 4) (H4: days2 = 8) (H5: mats1 = 4) (H6: mats2 = 16) 
    (rate_consistency: (mats1 / days1) = (mats2 / days2 / S2)): S1 = 4 := 
by
  sorry

end mat_weaves_problem_l76_76900


namespace cost_of_acai_berry_juice_l76_76910

theorem cost_of_acai_berry_juice (cost_per_litre_cocktail : ℝ)
                                 (cost_per_litre_fruit_juice : ℝ)
                                 (litres_fruit_juice : ℝ)
                                 (litres_acai_juice : ℝ)
                                 (total_cost_cocktail : ℝ)
                                 (cost_per_litre_acai : ℝ) :
  cost_per_litre_cocktail = 1399.45 →
  cost_per_litre_fruit_juice = 262.85 →
  litres_fruit_juice = 34 →
  litres_acai_juice = 22.666666666666668 →
  total_cost_cocktail = (34 + 22.666666666666668) * 1399.45 →
  (litres_fruit_juice * cost_per_litre_fruit_juice + litres_acai_juice * cost_per_litre_acai) = total_cost_cocktail →
  cost_per_litre_acai = 3106.66666666666666 :=
by
  intros
  sorry

end cost_of_acai_berry_juice_l76_76910


namespace find_m_l76_76238

theorem find_m (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 < 2) ∧ (m^2 + m = 2) → m = -2 :=
by
  sorry

end find_m_l76_76238


namespace probability_roll_2_four_times_in_five_rolls_l76_76892

theorem probability_roll_2_four_times_in_five_rolls :
  (∃ (prob_roll_2 : ℚ) (prob_not_roll_2 : ℚ), 
   prob_roll_2 = 1/6 ∧ prob_not_roll_2 = 5/6 ∧ 
   (5 * prob_roll_2^4 * prob_not_roll_2 = 5/72)) :=
sorry

end probability_roll_2_four_times_in_five_rolls_l76_76892


namespace problem_l76_76068

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}
def C : Set ℕ := {1, 3}

theorem problem : A ∩ (U \ B) = C := by
  sorry

end problem_l76_76068


namespace hurricane_damage_in_euros_l76_76211

-- Define the conditions
def usd_damage : ℝ := 45000000  -- Damage in US dollars
def exchange_rate : ℝ := 0.9    -- Exchange rate from US dollars to Euros

-- Define the target value in Euros
def eur_damage : ℝ := 40500000  -- Expected damage in Euros

-- The theorem to prove
theorem hurricane_damage_in_euros :
  usd_damage * exchange_rate = eur_damage :=
by
  sorry

end hurricane_damage_in_euros_l76_76211


namespace days_in_month_l76_76316

theorem days_in_month 
  (S : ℕ) (D : ℕ) (h1 : 150 * S + 120 * D = (S + D) * 125) (h2 : S = 5) :
  S + D = 30 :=
by
  sorry

end days_in_month_l76_76316


namespace values_of_x_l76_76857

theorem values_of_x (x : ℤ) :
  (∃ t : ℤ, x = 105 * t + 22) ∨ (∃ t : ℤ, x = 105 * t + 37) ↔ 
  (5 * x^3 - x + 17) % 15 = 0 ∧ (2 * x^2 + x - 3) % 7 = 0 :=
by {
  sorry
}

end values_of_x_l76_76857


namespace find_FC_l76_76888

theorem find_FC (DC : ℝ) (CB : ℝ) (AB AD ED FC : ℝ) 
  (h1 : DC = 9) 
  (h2 : CB = 10) 
  (h3 : AB = (1/3) * AD) 
  (h4 : ED = (3/4) * AD) 
  (h5 : FC = 14.625) : FC = 14.625 :=
by sorry

end find_FC_l76_76888


namespace max_mineral_value_l76_76433

/-- Jane discovers three types of minerals with given weights and values:
6-pound mineral chunks worth $16 each,
3-pound mineral chunks worth $9 each,
and 2-pound mineral chunks worth $3 each. 
There are at least 30 of each type available.
She can haul a maximum of 21 pounds in her cart.
Prove that the maximum value, in dollars, that Jane can transport is $63. -/
theorem max_mineral_value : 
  ∃ (value : ℕ), (∀ (x y z : ℕ), 6 * x + 3 * y + 2 * z ≤ 21 → 
    (x ≤ 30 ∧ y ≤ 30 ∧ z ≤ 30) → value ≥ 16 * x + 9 * y + 3 * z) ∧ value = 63 :=
by sorry

end max_mineral_value_l76_76433


namespace cost_per_mile_first_plan_l76_76822

theorem cost_per_mile_first_plan 
  (initial_fee : ℝ) (cost_per_mile_first : ℝ) (cost_per_mile_second : ℝ) (miles : ℝ)
  (h_first : initial_fee = 65)
  (h_cost_second : cost_per_mile_second = 0.60)
  (h_miles : miles = 325)
  (h_equal_cost : initial_fee + miles * cost_per_mile_first = miles * cost_per_mile_second) :
  cost_per_mile_first = 0.40 :=
by
  sorry

end cost_per_mile_first_plan_l76_76822


namespace sqrt_one_div_four_is_one_div_two_l76_76270

theorem sqrt_one_div_four_is_one_div_two : Real.sqrt (1 / 4) = 1 / 2 :=
by
  sorry

end sqrt_one_div_four_is_one_div_two_l76_76270


namespace function_evaluation_l76_76197

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2 * x :=
by
  sorry

end function_evaluation_l76_76197


namespace blue_marbles_in_bag_l76_76043

theorem blue_marbles_in_bag
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (prob_red_white : ℚ)
  (number_red_marbles: red_marbles = 9) 
  (total_marbles_eq: total_marbles = 30) 
  (prob_red_white_eq: prob_red_white = 5/6): 
  ∃ (blue_marbles : ℕ), blue_marbles = 5 :=
by
  have W := 16        -- This is from (9 + W)/30 = 5/6 which gives W = 16
  let B := total_marbles - red_marbles - W
  use B
  have h : B = 30 - 9 - 16 := by
    -- Remaining calculations
    sorry
  exact h

end blue_marbles_in_bag_l76_76043


namespace younger_brother_age_l76_76710

variable (x y : ℕ)

theorem younger_brother_age :
  x + y = 46 →
  y = x / 3 + 10 →
  y = 19 :=
by
  intros h1 h2
  sorry

end younger_brother_age_l76_76710


namespace distance_to_Rock_Mist_Mountains_l76_76225

theorem distance_to_Rock_Mist_Mountains (d_Sky_Falls : ℕ) (multiplier : ℕ) (d_Rock_Mist : ℕ) :
  d_Sky_Falls = 8 → multiplier = 50 → d_Rock_Mist = d_Sky_Falls * multiplier → d_Rock_Mist = 400 :=
by 
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end distance_to_Rock_Mist_Mountains_l76_76225


namespace rectangle_area_expression_l76_76240

theorem rectangle_area_expression {d x : ℝ} (h : d^2 = 29 * x^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = (10 / 29) :=
by {
 sorry
}

end rectangle_area_expression_l76_76240


namespace harkamal_paid_amount_l76_76594

variable (grapesQuantity : ℕ)
variable (grapesRate : ℕ)
variable (mangoesQuantity : ℕ)
variable (mangoesRate : ℕ)

theorem harkamal_paid_amount (h1 : grapesQuantity = 8) (h2 : grapesRate = 70) (h3 : mangoesQuantity = 9) (h4 : mangoesRate = 45) :
  (grapesQuantity * grapesRate + mangoesQuantity * mangoesRate) = 965 := by
  sorry

end harkamal_paid_amount_l76_76594


namespace an_expression_l76_76058

-- Given conditions
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n

-- The statement to be proved
theorem an_expression (a : ℕ → ℕ) (n : ℕ) (h_Sn : ∀ n, Sn a n = 2 * a n - n) :
  a n = 2^n - 1 :=
sorry

end an_expression_l76_76058


namespace puppies_start_count_l76_76437

theorem puppies_start_count (x : ℕ) (given_away : ℕ) (left : ℕ) (h1 : given_away = 7) (h2 : left = 5) (h3 : x = given_away + left) : x = 12 :=
by
  rw [h1, h2] at h3
  exact h3

end puppies_start_count_l76_76437


namespace quadratic_minimum_eq_one_l76_76554

variable (p q : ℝ)

theorem quadratic_minimum_eq_one (hq : q = 1 + p^2 / 18) : 
  ∃ x : ℝ, 3 * x^2 + p * x + q = 1 :=
by
  sorry

end quadratic_minimum_eq_one_l76_76554


namespace locus_of_M_l76_76757

/-- Define the coordinates of points A and B, and given point M(x, y) with the 
    condition x ≠ ±1, ensure the equation of the locus of point M -/
theorem locus_of_M (x y : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) 
  (h3 : (y / (x + 1)) + (y / (x - 1)) = 2) : x^2 - x * y - 1 = 0 := 
sorry

end locus_of_M_l76_76757


namespace find_number_l76_76147

theorem find_number (x : ℤ) (h : 2 * x - 8 = -12) : x = -2 :=
by
  sorry

end find_number_l76_76147


namespace problems_completed_l76_76525

theorem problems_completed (p t : ℕ) (h1 : p ≥ 15) (h2 : p * t = (2 * p - 10) * (t - 1)) : p * t = 60 := sorry

end problems_completed_l76_76525


namespace ratio_of_perimeters_is_one_l76_76011

-- Definitions based on the given conditions
def original_rectangle : ℝ × ℝ := (6, 8)
def folded_rectangle : ℝ × ℝ := (3, 8)
def small_rectangle : ℝ × ℝ := (3, 4)
def large_rectangle : ℝ × ℝ := (3, 4)

-- The perimeter function for a rectangle given its dimensions (length, width)
def perimeter (r : ℝ × ℝ) : ℝ := 2 * (r.1 + r.2)

-- The main theorem to prove
theorem ratio_of_perimeters_is_one : 
  perimeter small_rectangle / perimeter large_rectangle = 1 :=
by
  sorry

end ratio_of_perimeters_is_one_l76_76011


namespace any_nat_as_difference_or_element_l76_76789

noncomputable def seq (q : ℕ → ℕ) : Prop :=
∀ n, q n < 2 * n

theorem any_nat_as_difference_or_element (q : ℕ → ℕ) (h_seq : seq q) (m : ℕ) :
  (∃ k, q k = m) ∨ (∃ k l, q l - q k = m) :=
sorry

end any_nat_as_difference_or_element_l76_76789


namespace team_incorrect_answers_l76_76376

theorem team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) 
  (ofelia_correct : ℕ) :
  total_questions = 35 → riley_mistakes = 3 → 
  ofelia_correct = ((total_questions - riley_mistakes) / 2 + 5) → 
  riley_mistakes + (total_questions - ofelia_correct) = 17 :=
by
  intro h1 h2 h3
  sorry

end team_incorrect_answers_l76_76376


namespace same_side_of_line_l76_76881

theorem same_side_of_line (a : ℝ) :
    let point1 := (3, -1)
    let point2 := (-4, -3)
    let line_eq (x y : ℝ) := 3 * x - 2 * y + a
    (line_eq point1.1 point1.2) * (line_eq point2.1 point2.2) > 0 ↔
        (a < -11 ∨ a > 6) := sorry

end same_side_of_line_l76_76881


namespace problem_solution_l76_76377

variables (p q : Prop)

theorem problem_solution (h1 : ¬ (p ∧ q)) (h2 : p ∨ q) : ¬ p ∨ ¬ q := by
  sorry

end problem_solution_l76_76377


namespace cricket_team_matches_played_in_august_l76_76876

theorem cricket_team_matches_played_in_august
    (M : ℕ)
    (h1 : ∃ W : ℕ, W = 24 * M / 100)
    (h2 : ∃ W : ℕ, W + 70 = 52 * (M + 70) / 100) :
    M = 120 :=
sorry

end cricket_team_matches_played_in_august_l76_76876


namespace cross_section_area_ratio_correct_l76_76705

variable (α : ℝ)
noncomputable def cross_section_area_ratio : ℝ := 2 * (Real.cos α)

theorem cross_section_area_ratio_correct (α : ℝ) : 
  cross_section_area_ratio α = 2 * Real.cos α :=
by
  unfold cross_section_area_ratio
  sorry

end cross_section_area_ratio_correct_l76_76705


namespace inverse_proportion_decreases_l76_76690

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end inverse_proportion_decreases_l76_76690


namespace negation_of_existence_l76_76522

theorem negation_of_existence (p : Prop) (h : ∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) : 
  ¬ (∃ (c : ℝ), c > 0 ∧ (∃ (x : ℝ), x^2 - x + c = 0)) ↔ 
  ∀ (c : ℝ), c > 0 → ¬ (∃ (x : ℝ), x^2 - x + c = 0) :=
by 
  sorry

end negation_of_existence_l76_76522


namespace mary_score_unique_l76_76927

theorem mary_score_unique (c w : ℕ) (s : ℕ) (h_score_formula : s = 35 + 4 * c - w)
  (h_limit : c + w ≤ 35) (h_greater_90 : s > 90) :
  (∀ s' > 90, s' ≠ s → ¬ ∃ c' w', s' = 35 + 4 * c' - w' ∧ c' + w' ≤ 35) → s = 91 :=
by
  sorry

end mary_score_unique_l76_76927


namespace acres_used_for_corn_l76_76999

theorem acres_used_for_corn (total_land : ℕ) (ratio_beans : ℕ) (ratio_wheat : ℕ) (ratio_corn : ℕ)
  (total_ratio_parts : ℕ) (one_part_size : ℕ) :
  total_land = 1034 →
  ratio_beans = 5 →
  ratio_wheat = 2 →
  ratio_corn = 4 →
  total_ratio_parts = ratio_beans + ratio_wheat + ratio_corn →
  one_part_size = total_land / total_ratio_parts →
  ratio_corn * one_part_size = 376 :=
by
  intros
  sorry

end acres_used_for_corn_l76_76999


namespace minimum_workers_needed_to_make_profit_l76_76642

-- Given conditions
def fixed_maintenance_fee : ℝ := 550
def setup_cost : ℝ := 200
def wage_per_hour : ℝ := 18
def widgets_per_worker_per_hour : ℝ := 6
def sell_price_per_widget : ℝ := 3.5
def work_hours_per_day : ℝ := 8

-- Definitions derived from conditions
def daily_wage_per_worker := wage_per_hour * work_hours_per_day
def daily_revenue_per_worker := widgets_per_worker_per_hour * work_hours_per_day * sell_price_per_widget
def total_daily_cost (n : ℝ) := fixed_maintenance_fee + setup_cost + n * daily_wage_per_worker

-- Prove that the number of workers needed to make a profit is at least 32
theorem minimum_workers_needed_to_make_profit (n : ℕ) (h : (total_daily_cost (n : ℝ)) < n * daily_revenue_per_worker) :
  n ≥ 32 := by
  -- We fill the sorry for proof to pass Lean check
  sorry

end minimum_workers_needed_to_make_profit_l76_76642


namespace polar_to_cartesian_l76_76861

theorem polar_to_cartesian (r θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 6) :
  (r * Real.cos θ, r * Real.sin θ) = (Real.sqrt 3, 1) :=
by
  rw [h_r, h_θ]
  have h_cos : Real.cos (π / 6) = Real.sqrt 3 / 2 := sorry -- This identity can be used from trigonometric property.
  have h_sin : Real.sin (π / 6) = 1 / 2 := sorry -- This identity can be used from trigonometric property.
  rw [h_cos, h_sin]
  -- some algebraic steps to simplifiy left sides to (Real.sqrt 3, 1) should follow here. using multiplication and commmutaivity properties mainly.
  sorry

end polar_to_cartesian_l76_76861


namespace smallest_number_groups_l76_76368

theorem smallest_number_groups (x : ℕ) (h₁ : x % 18 = 0) (h₂ : x % 45 = 0) : x = 90 :=
sorry

end smallest_number_groups_l76_76368


namespace henri_total_miles_l76_76347

noncomputable def g_total : ℕ := 315 * 3
noncomputable def h_total : ℕ := g_total + 305

theorem henri_total_miles : h_total = 1250 :=
by
  -- proof goes here
  sorry

end henri_total_miles_l76_76347


namespace equalize_marbles_condition_l76_76986

variables (D : ℝ)
noncomputable def marble_distribution := 
    let C := 1.25 * D
    let B := 1.4375 * D
    let A := 1.725 * D
    let total := A + B + C + D
    let equal := total / 4
    let move_from_A := (A - equal) / A * 100
    let move_from_B := (B - equal) / B * 100
    let add_to_C := (equal - C) / C * 100
    let add_to_D := (equal - D) / D * 100
    (move_from_A, move_from_B, add_to_C, add_to_D)

theorem equalize_marbles_condition :
    marble_distribution D = (21.56, 5.87, 8.25, 35.31) := sorry

end equalize_marbles_condition_l76_76986


namespace number_of_10_digit_numbers_divisible_by_66667_l76_76931

def ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 : ℕ := 33

theorem number_of_10_digit_numbers_divisible_by_66667 :
  ∃ n : ℕ, n = ten_digit_numbers_composed_of_3_4_5_6_divisible_by_66667 :=
by
  sorry

end number_of_10_digit_numbers_divisible_by_66667_l76_76931


namespace sophomores_in_seminar_l76_76730

theorem sophomores_in_seminar (P Q x y : ℕ)
  (h1 : P + Q = 50)
  (h2 : x = y)
  (h3 : x = (1 / 5 : ℚ) * P)
  (h4 : y = (1 / 4 : ℚ) * Q) :
  P = 22 :=
by
  sorry

end sophomores_in_seminar_l76_76730


namespace parabola_distance_l76_76177

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l76_76177


namespace undefined_expression_iff_l76_76497

theorem undefined_expression_iff (x : ℝ) :
  (x^2 - 24 * x + 144 = 0) ↔ (x = 12) := 
sorry

end undefined_expression_iff_l76_76497


namespace cost_formula_l76_76812

def cost (P : ℕ) : ℕ :=
  if P ≤ 5 then 5 * P + 10 else 5 * P + 5

theorem cost_formula (P : ℕ) : 
  cost P = (if P ≤ 5 then 5 * P + 10 else 5 * P + 5) :=
by 
  sorry

end cost_formula_l76_76812


namespace find_xyz_l76_76277

variables (A B C B₁ A₁ C₁ : Type)
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B] [AddCommGroup C] [Module ℝ C]

def AC1 (AB BC CC₁ : A) (x y z : ℝ) : A :=
  x • AB + 2 • y • BC + 3 • z • CC₁

theorem find_xyz (AB BC CC₁ AC1 : A)
  (h1 : AC1 = AB + BC + CC₁)
  (h2 : AC1 = x • AB + 2 • y • BC + 3 • z • CC₁) :
  x + y + z = 11 / 6 :=
sorry

end find_xyz_l76_76277


namespace fractions_product_l76_76534

theorem fractions_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := 
by 
  sorry

end fractions_product_l76_76534


namespace total_weight_of_onions_l76_76878

def weight_per_bag : ℕ := 50
def bags_per_trip : ℕ := 10
def trips : ℕ := 20

theorem total_weight_of_onions : bags_per_trip * weight_per_bag * trips = 10000 := by
  sorry

end total_weight_of_onions_l76_76878


namespace statement_a_statement_b_statement_c_statement_d_l76_76149

open Real

-- Statement A (incorrect)
theorem statement_a (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a*c > b*d) := sorry

-- Statement B (correct)
theorem statement_b (a b : ℝ) (h1 : b < a) (h2 : a < 0) : (1 / a < 1 / b) := sorry

-- Statement C (incorrect)
theorem statement_c (a b : ℝ) (h : 1 / (a^2) < 1 / (b^2)) : ¬ (a > abs b) := sorry

-- Statement D (correct)
theorem statement_d (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) : (a + m) / (b + m) > a / b := sorry

end statement_a_statement_b_statement_c_statement_d_l76_76149


namespace mean_height_basketball_team_l76_76341

def heights : List ℕ :=
  [58, 59, 60, 62, 63, 65, 65, 68, 70, 71, 71, 72, 76, 76, 78, 79, 79]

def mean_height (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_height_basketball_team :
  mean_height heights = 70 := by
  sorry

end mean_height_basketball_team_l76_76341


namespace batsman_average_after_25th_innings_l76_76192

theorem batsman_average_after_25th_innings (A : ℝ) (runs_25th : ℝ) (increase : ℝ) (not_out_innings : ℕ) 
    (total_innings : ℕ) (average_increase_condition : 24 * A + runs_25th = 25 * (A + increase)) :       
    runs_25th = 150 ∧ increase = 3 ∧ not_out_innings = 3 ∧ total_innings = 25 → 
    ∃ avg : ℝ, avg = 88.64 := by 
  sorry

end batsman_average_after_25th_innings_l76_76192


namespace man_speed_proof_l76_76955

noncomputable def train_length : ℝ := 150 
noncomputable def crossing_time : ℝ := 6 
noncomputable def train_speed_kmph : ℝ := 84.99280057595394 
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

noncomputable def relative_speed_mps : ℝ := train_length / crossing_time
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps
noncomputable def man_speed_kmph : ℝ := man_speed_mps * (3600 / 1000)

theorem man_speed_proof : man_speed_kmph = 5.007198224048459 := by 
  sorry

end man_speed_proof_l76_76955


namespace odd_function_solution_l76_76132

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_solution (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, x > 0 → f x = x^3 + x + 1) :
  ∀ x : ℝ, x < 0 → f x = x^3 + x - 1 :=
by
  sorry

end odd_function_solution_l76_76132


namespace solid_circles_count_2006_l76_76612

def series_of_circles (n : ℕ) : List Char :=
  if n ≤ 0 then []
  else if n % 5 == 0 then '●' :: series_of_circles (n - 1)
  else '○' :: series_of_circles (n - 1)

def count_solid_circles (l : List Char) : ℕ :=
  l.count '●'

theorem solid_circles_count_2006 : count_solid_circles (series_of_circles 2006) = 61 := 
by
  sorry

end solid_circles_count_2006_l76_76612


namespace number_of_proper_subsets_l76_76232

theorem number_of_proper_subsets (S : Finset ℕ) (h : S = {1, 2, 3, 4}) : S.powerset.card - 1 = 15 := by
  sorry

end number_of_proper_subsets_l76_76232


namespace research_question_correct_survey_method_correct_l76_76089

-- Define the conditions.
def total_students : Nat := 400
def sampled_students : Nat := 80

-- Define the research question.
def research_question : String := "To understand the vision conditions of 400 eighth-grade students in a certain school."

-- Define the survey method.
def survey_method : String := "A sampling survey method was used."

-- Prove the research_question matches the expected question given the conditions.
theorem research_question_correct :
  research_question = "To understand the vision conditions of 400 eighth-grade students in a certain school" := by
  sorry

-- Prove the survey method used matches the expected method given the conditions.
theorem survey_method_correct :
  survey_method = "A sampling survey method was used" := by
  sorry

end research_question_correct_survey_method_correct_l76_76089


namespace factor_polynomial_l76_76156

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end factor_polynomial_l76_76156


namespace trig_identity_proof_l76_76394

theorem trig_identity_proof
  (h1: Float.sin 50 = Float.cos 40)
  (h2: Float.tan 45 = 1)
  (h3: Float.tan 10 = Float.sin 10 / Float.cos 10)
  (h4: Float.sin 80 = Float.cos 10) :
  Float.sin 50 * (Float.tan 45 + Float.sqrt 3 * Float.tan 10) = 1 :=
by
  sorry

end trig_identity_proof_l76_76394


namespace complement_intersection_l76_76499

open Set

theorem complement_intersection
  (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = univ) 
  (hA : A = { x : ℝ | x ≤ -2 }) 
  (hB : B = { x : ℝ | x < 1 }) :
  (U \ A) ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l76_76499


namespace order_of_nums_l76_76588

variable (a b : ℝ)

theorem order_of_nums (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := 
sorry

end order_of_nums_l76_76588


namespace fifth_equation_l76_76150

noncomputable def equation_1 : Prop := 2 * 1 = 2
noncomputable def equation_2 : Prop := 2 ^ 2 * 1 * 3 = 3 * 4
noncomputable def equation_3 : Prop := 2 ^ 3 * 1 * 3 * 5 = 4 * 5 * 6

theorem fifth_equation
  (h1 : equation_1)
  (h2 : equation_2)
  (h3 : equation_3) :
  2 ^ 5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
by {
  sorry
}

end fifth_equation_l76_76150


namespace radii_of_cylinder_and_cone_are_equal_l76_76701

theorem radii_of_cylinder_and_cone_are_equal
  (h : ℝ)
  (r : ℝ)
  (V_cylinder : ℝ := π * r^2 * h)
  (V_cone : ℝ := (1/3) * π * r^2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
  r = r :=
by
  sorry

end radii_of_cylinder_and_cone_are_equal_l76_76701


namespace oranges_left_to_sell_today_l76_76391

theorem oranges_left_to_sell_today (initial_dozen : Nat)
    (reserved_fraction1 reserved_fraction2 sold_fraction eaten_fraction : ℚ)
    (rotten_oranges : Nat) 
    (h1 : initial_dozen = 7)
    (h2 : reserved_fraction1 = 1/4)
    (h3 : reserved_fraction2 = 1/6)
    (h4 : sold_fraction = 3/7)
    (h5 : eaten_fraction = 1/10)
    (h6 : rotten_oranges = 4) : 
    let total_oranges := initial_dozen * 12
    let reserved1 := total_oranges * reserved_fraction1
    let reserved2 := total_oranges * reserved_fraction2
    let remaining_after_reservation := total_oranges - reserved1 - reserved2
    let sold_yesterday := remaining_after_reservation * sold_fraction
    let remaining_after_sale := remaining_after_reservation - sold_yesterday
    let eaten_by_birds := remaining_after_sale * eaten_fraction
    let remaining_after_birds := remaining_after_sale - eaten_by_birds
    let final_remaining := remaining_after_birds - rotten_oranges
    final_remaining = 22 :=
by
    sorry

end oranges_left_to_sell_today_l76_76391


namespace shopkeeper_percentage_gain_l76_76532

theorem shopkeeper_percentage_gain (false_weight true_weight : ℝ) 
    (h_false_weight : false_weight = 930)
    (h_true_weight : true_weight = 1000) : 
    (true_weight - false_weight) / false_weight * 100 = 7.53 := 
by
  rw [h_false_weight, h_true_weight]
  sorry

end shopkeeper_percentage_gain_l76_76532


namespace range_of_x_l76_76712

variable (f : ℝ → ℝ)

def even_function :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

def f_value_at_2 := f 2 = 0

theorem range_of_x (h1 : even_function f) (h2 : monotonically_decreasing f) (h3 : f_value_at_2 f) :
  { x : ℝ | f (x - 1) > 0 } = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end range_of_x_l76_76712


namespace total_pokemon_cards_l76_76736

-- Definitions based on the problem statement

def dozen_to_cards (dozen : ℝ) : ℝ :=
  dozen * 12

def melanie_cards : ℝ :=
  dozen_to_cards 7.5

def benny_cards : ℝ :=
  dozen_to_cards 9

def sandy_cards : ℝ :=
  dozen_to_cards 5.2

def jessica_cards : ℝ :=
  dozen_to_cards 12.8

def total_cards : ℝ :=
  melanie_cards + benny_cards + sandy_cards + jessica_cards

theorem total_pokemon_cards : total_cards = 414 := 
  by sorry

end total_pokemon_cards_l76_76736


namespace hexagon_perimeter_l76_76763

def side_length : ℝ := 4
def number_of_sides : ℕ := 6

theorem hexagon_perimeter :
  6 * side_length = 24 := by
    sorry

end hexagon_perimeter_l76_76763


namespace quadratic_function_example_l76_76442

theorem quadratic_function_example : ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x^2 + b * x + c = 0) ↔ (x = 1 ∨ x = 5)) ∧ 
  (a * 3^2 + b * 3 + c = 8) ∧ 
  (a = -2 ∧ b = 12 ∧ c = -10) :=
by
  sorry

end quadratic_function_example_l76_76442
