import Mathlib

namespace modular_inverse_5_mod_19_l219_219016

theorem modular_inverse_5_mod_19 : ∃ a : ℕ, 0 ≤ a ∧ a < 19 ∧ (5 * a) % 19 = 1 :=
by {
  use 23,
  split,
  { exact nat.le_refl 23, },
  split,
  { norm_num, },
  { norm_num, },
  sorry
}

end modular_inverse_5_mod_19_l219_219016


namespace region_area_l219_219191

theorem region_area (A B : ℝ × ℝ) (hAB : dist A B = 1) :
  let area := (4 * real.pi - 3 * real.sqrt 3) / 18 in
  ∃ P : ℝ × ℝ, ∃ θ : ℝ, θ > 120 ∧ (abs (angle A P B) = θ) → 
  region.area = area := 
by
  sorry

end region_area_l219_219191


namespace smallest_increase_between_Q11_Q12_l219_219974

noncomputable def smallest_percent_increase_pair : Nat × Nat :=
  let values := [100, 300, 600, 900, 1200, 1700, 2700, 5000, 8000, 12000, 16000, 21000, 27000, 34000, 50000]
  let percent_increase (i j : Nat) : Float := 
    (values[j] - values[i]) / values[i].toFloat * 100
  List.argmin (List.range (values.length - 1)) (λ i, percent_increase i (i+1)) = (10, 11)

theorem smallest_increase_between_Q11_Q12 : smallest_percent_increase_pair = (10, 11) :=
  sorry

end smallest_increase_between_Q11_Q12_l219_219974


namespace tetrahedron_circumscribed_sphere_radius_l219_219578

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end tetrahedron_circumscribed_sphere_radius_l219_219578


namespace xiao_ming_water_usage_ge_8_l219_219783

def min_monthly_water_usage (x : ℝ) : Prop :=
  ∀ (c : ℝ), c ≥ 15 →
    (c = if x ≤ 5 then x * 1.8 else (5 * 1.8 + (x - 5) * 2)) →
      x ≥ 8

theorem xiao_ming_water_usage_ge_8 : ∃ x : ℝ, min_monthly_water_usage x :=
  sorry

end xiao_ming_water_usage_ge_8_l219_219783


namespace complement_A_in_U_l219_219080

noncomputable def U : Set ℕ := {0, 1, 2}
noncomputable def A : Set ℕ := {x | x^2 - x = 0}
noncomputable def complement_U (A : Set ℕ) : Set ℕ := U \ A

theorem complement_A_in_U : 
  complement_U {x | x^2 - x = 0} = {2} := 
sorry

end complement_A_in_U_l219_219080


namespace train_speed_is_60_kmph_l219_219425

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end train_speed_is_60_kmph_l219_219425


namespace subsets_with_at_least_four_adjacent_chairs_l219_219723

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219723


namespace tournament_total_players_l219_219968

noncomputable def total_players_in_tournament : ℕ := 24

theorem tournament_total_players (n : ℕ) : 
  let total_players := n + 12 in
  let total_games := (total_players * (total_players - 1)) / 2 in
  let low_players_games := (12 * (12 - 1)) / 2 = 66 in
  let low_players_points := low_players_games * 2 = 132 in
  let high_players_points := (n * (n - 1)) + 132 in
  let total_points := high_players_points = total_games in
  n = 12 → total_players = 24 :=
by 
  intros n;
  let total_players = n + 12;
  let total_games = (total_players * (total_players - 1)) / 2;
  let low_players_games = (12 * (12 - 1)) / 2;
  let low_players_points = low_players_games * 2;
  let high_players_points = (n * (n - 1)) + 132;
  let total_points = high_players_points = total_games;
  assume n_eq_12 : n = 12;
  rw n_eq_12 at ⊢ total_players;
  exact total_players = 24,
  sorry


end tournament_total_players_l219_219968


namespace subsets_with_at_least_four_adjacent_chairs_l219_219761

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219761


namespace tangent_circle_parallel_l219_219662

-- Define the convex quadrilateral and its properties.
variables (A B C D : Point) (circle1 circle2 : Circle)
variable (O1 : Point)
variables (tangent1 tangent2 : Line)
variable [ConvexQuadrilateral ABCD]
variable [TangentLine tangent1 circle1]
variable [TangentLine tangent2 circle2]

-- Define the conditions of tangency and parallelism.
axiom CD_tangent_to_circle1 : TangentToCircle CD circle1
axiom AB_tangent_to_circle2 : TangentToCircle AB circle2 ↔ Parallel BC AD

-- Prove the equivalence statement.
theorem tangent_circle_parallel :
  TangentToCircle AB circle2 ↔ Parallel BC AD :=
by
  sorry

end tangent_circle_parallel_l219_219662


namespace problem_statement_l219_219468

def spadesuit (a b : ℕ) : ℕ :=
  abs (a - b)

theorem problem_statement : (spadesuit 2 3) + (spadesuit 5 10) = 6 := 
by
  sorry

end problem_statement_l219_219468


namespace probability_at_most_one_red_ball_l219_219031

variable (bag : Finset ({w : ℕ // w < 6}))
variable (w3 : {w : ℕ // w < 6}) (r2 : {r : ℕ // r < 6})
variable (draw_3 : Finset ({ball : {w : ℕ // w < 6} // ball ∈ bag}) )

def white_ball : {w : ℕ // w < 6} := sorry
def red_ball : {r : ℕ // r < 6} := sorry

axiom h_bag : 3 ≤ bag.card
axiom h_white_balls : ∀ w ∈ bag, (draw_3.card = 3 → draw_3.filter (λ x, x = ⟨white_ball, sorry⟩).card = 3 - r2.card)
axiom h_red_balls : ∀ r ∈ bag, (draw_3.card = 3 → draw_3.filter (λ x, x = ⟨red_ball, sorry⟩).card = 2)

theorem probability_at_most_one_red_ball :
  (3 / 5) * (2 / 4) * (2 / 3) = 7 / 10 :=
sorry

end probability_at_most_one_red_ball_l219_219031


namespace max_AN_dot_AM_l219_219190

-- Define the data for the square and points
def square_side_length : ℝ := 2
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, square_side_length)
def C : ℝ × ℝ := (square_side_length, square_side_length)
def D : ℝ × ℝ := (square_side_length, 0)

-- Define moving point M inside or on the boundary of the square ABCD
variable (M : ℝ × ℝ)
hypothesis hM : M.fst ≥ 0 ∧ M.fst ≤ square_side_length ∧ M.snd ≥ 0 ∧ M.snd ≤ square_side_length

-- Define the midpoint N of side BC
def N : ℝ × ℝ := ((B.fst + C.fst) / 2, (B.snd + C.snd) / 2)

-- Define the vector operations
def vec (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.fst - P.fst, Q.snd - P.snd)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.fst * v.fst + u.snd * v.snd

-- Define the vectors AN and AM
def AN := vec A N
def AM := vec A M

-- State and prove the proposition
theorem max_AN_dot_AM : dot_product AN AM ≤ 6 := sorry

end max_AN_dot_AM_l219_219190


namespace num_of_correct_statements_l219_219789

-- Define propositions p and q
variables (p q : Prop)

-- Define the four statements
def stmt1 := p ∧ q
def stmt2 := p ∧ ¬ q
def stmt3 := ¬ p ∧ q
def stmt4 := ¬ p ∧ ¬ q

-- Define the negation of p ∧ q
def neg_p_and_q := ¬ (p ∧ q)

-- Proof statement: There are exactly 3 statements concluding ¬ (p ∧ q)
theorem num_of_correct_statements : 
  (if stmt1 then 1 else 0) + (if stmt2 then 1 else 0) + (if stmt3 then 1 else 0) + (if stmt4 then 1 else 0) = 3 := sorry

end num_of_correct_statements_l219_219789


namespace probability_five_digit_palindrome_div_by_11_l219_219815

noncomputable
def five_digit_palindrome_div_by_11_probability : ℚ :=
  let total_palindromes := 900
  let valid_palindromes := 80
  valid_palindromes / total_palindromes

theorem probability_five_digit_palindrome_div_by_11 :
  five_digit_palindrome_div_by_11_probability = 2 / 25 := by
  sorry

end probability_five_digit_palindrome_div_by_11_l219_219815


namespace fraction_value_l219_219487

def x : ℚ := 4 / 7
def y : ℚ := 8 / 11

theorem fraction_value : (7 * x + 11 * y) / (49 * x * y) = 231 / 56 := by
  sorry

end fraction_value_l219_219487


namespace ratio_of_volumes_l219_219046

theorem ratio_of_volumes (r : ℝ) (π : ℝ) (V1 V2 : ℝ) 
  (h1 : V2 = (4 / 3) * π * r^3) 
  (h2 : V1 = 2 * π * r^3) : 
  V1 / V2 = 3 / 2 :=
by
  sorry

end ratio_of_volumes_l219_219046


namespace inequality_holds_for_k_2_l219_219863

theorem inequality_holds_for_k_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := 
by 
  sorry

end inequality_holds_for_k_2_l219_219863


namespace sequence_formula_sum_cn_l219_219892

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n : ℕ, n > 0 → S n = a n + n^2 - 1) :
  ∀ n : ℕ, n > 0 → a n = 2 * n + 1 :=
sorry

theorem sum_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℝ) (T : ℕ → ℝ)
  (hS : ∀ n : ℕ, n > 0 → S n = a n + n^2 - 1)
  (ha : ∀ n : ℕ, n > 0 → a n = 2 * n + 1)
  (hc : ∀ n : ℕ, n > 0 → c n = (2 * (real.to_fract (ñ) / a n)) mod 1)
  (hT : T 1 = 0 ∧ ∀ n : ℕ, n > 1 → T n = ∑ i in range n, c i) :
  ∀ n : ℕ, n > 0 → T n = (5 * n^2 + 3 * n - 8) / (4 * n^2 + 12 * n + 8) :=
sorry

end sequence_formula_sum_cn_l219_219892


namespace sum_prime_factors_143_l219_219310

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219310


namespace hyperbola_eccentricity_identity_l219_219500

theorem hyperbola_eccentricity_identity
  (a b : ℝ) (A B C : ℝ × ℝ)
  (h_hyperbola_eq : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h_foci_A : A = (-(sqrt (a^2 + b^2)), 0))
  (h_foci_C : C = (sqrt (a^2 + b^2), 0))
  (h_vertex_B : B ∈ { p : ℝ × ℝ | ∃ x y, x^2 / a^2 - y^2 / b^2 = 1 })
  (e : ℝ)
  (h_eccentricity : e = sqrt (1 + b^2 / a^2)) :
  (abs (Real.sin (fst A) - Real.sin (snd C)) / Real.sin (snd B) = 1 / e) := sorry

end hyperbola_eccentricity_identity_l219_219500


namespace sum_even_102_to_200_l219_219700

def sum_even_integers (m n : ℕ) : ℕ :=
  sum (list.map (λ k, 2 * k) (list.range' m (n - m + 1)))

theorem sum_even_102_to_200 :
  sum_even_integers (102 / 2) (200 / 2) = 7550 := 
sorry

end sum_even_102_to_200_l219_219700


namespace spherical_distance_equiv_l219_219904

noncomputable def earth_radius : ℝ := sorry -- Earth's radius R
noncomputable def arc_length : ℝ := (Real.sqrt 2 / 4) * Real.pi * earth_radius
noncomputable def latitude_radius : ℝ := earth_radius / (Real.sqrt 2)

theorem spherical_distance_equiv (R : ℝ)
  (h1 : Locations A and B are at latitude 45°N)
  (h2 : arc_length = (Real.sqrt 2 / 4) * Real.pi * R) :
  let α : ℝ := (Real.sqrt 2 / 4) * Real.pi in
  let AB := R * α in
  AB = R * (Real.pi * Real.sqrt 2 / 4) := 
by
  -- Define the variables
  let r := R / Real.sqrt 2
  have h_arc_length : arc_length = (Real.sqrt 2 / 4) * Real.pi * R, from h2
  let α := arc_length / r
  have α_simp : α = (Real.sqrt 2 / 4) * Real.pi, by sorry
  let AB := R * α
  have AB_simp : AB = R * (Real.pi * Real.sqrt 2 / 4), by sorry
  exact AB_simp

end spherical_distance_equiv_l219_219904


namespace sum_divisors_eq_l219_219631

theorem sum_divisors_eq (n : ℕ) (h : n = nat.factorial 1990) (d : ℕ → ℕ) (hd : ∀ i, d i ∣ n) :
    (∀ i, d i ∈ finset.univ.image (λ x, x * n / x) → ∑ i, (d i / real.sqrt n) = ∑ i, (real.sqrt n / d i)) :=
by 
  intros h0
  sorry

end sum_divisors_eq_l219_219631


namespace Serezha_puts_more_berries_l219_219852

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end Serezha_puts_more_berries_l219_219852


namespace garage_sale_items_count_l219_219788

theorem garage_sale_items_count (n_high n_low: ℕ) :
  n_high = 17 ∧ n_low = 24 → total_items = 40 :=
by
  let n_high: ℕ := 17
  let n_low: ℕ := 24
  let total_items: ℕ := (n_high - 1) + (n_low - 1) + 1
  sorry

end garage_sale_items_count_l219_219788


namespace sum_of_prime_factors_of_143_l219_219308

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219308


namespace smallest_x_for_multiple_of_600_and_1152_l219_219282

theorem smallest_x_for_multiple_of_600_and_1152 : ∃ (x : ℕ), x > 0 ∧ (600 * x) % 1152 = 0 ∧ ∀ y : ℕ, (y > 0 ∧ (600 * y) % 1152 = 0) → x ≤ y :=
by
  use 48
  split
  case h_left => exact Nat.zero_lt_succ _
  case h_right => sorry

end smallest_x_for_multiple_of_600_and_1152_l219_219282


namespace max_balls_in_cube_l219_219775

theorem max_balls_in_cube :
  let V_cube := 6^3 in
  let r := 2 in
  let V_ball := (4/3) * π * r^3 in
  ⌊ V_cube / V_ball ⌋ = 6 :=
by
  let V_cube := 6^3
  let r := 2
  let V_ball := (4/3) * π * r^3
  show ⌊ V_cube / V_ball ⌋ = 6
  sorry

end max_balls_in_cube_l219_219775


namespace collision_probability_l219_219252

noncomputable def probability_of_collision 
  (arrival_time_A : ℝ) (arrival_time_B : ℝ)
  (clearance_time : ℝ) 
  (uniform_dist_A : probabilistic_distribution_uniform ℝ 0 330 arrival_time_A) 
  (uniform_dist_B : probabilistic_distribution_uniform ℝ 30 210 arrival_time_B) 
  (independence : independent arrival_time_A arrival_time_B) : ℝ :=
  have collision_area : ℝ := 56025 / 59400 in -- area calculations 
  collision_area

theorem collision_probability :
  probability_of_collision 0 330 45 uniform_dist_A uniform_dist_B independence = 13 / 48 :=
  sorry

end collision_probability_l219_219252


namespace sum_prime_factors_of_143_l219_219354

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219354


namespace polar_equation_of_curve_slope_of_line_l219_219596

open Real

-- Definitions for the parametric equations of curve C and the polar system
def curve_parametric (α : ℝ) : ℝ × ℝ := 
  (2 + sqrt 3 * cos α, sqrt 3 * sin α)

-- Conditions provided in the problem
def polar_condition (θ : ℝ) : Prop := 
  ∃ ρ : ℝ, ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_intersection_condition (θ : ℝ) : Prop :=
  let ρ1 := 2 * cos θ + sqrt (3 - 4 * cos^2 θ),
      ρ2 := 2 * cos θ - sqrt (3 - 4 * cos^2 θ) in
  abs ρ1 + abs ρ2 = 2 * sqrt 3

-- The theorems we want to prove
theorem polar_equation_of_curve : 
  ∀ α, 
  0 ≤ α → α ≤ 2 * π → 
  ∃ θ : ℝ, polar_condition θ := 
by
  sorry

theorem slope_of_line : 
  ∀ θ, 
  line_intersection_condition θ →
  θ = π / 6 ∨ θ = 5 * π / 6 → 
  tan θ = (sqrt 3) / 3 ∨ tan θ = - (sqrt 3) / 3 :=
by
  sorry

end polar_equation_of_curve_slope_of_line_l219_219596


namespace solve_root_value_l219_219155

noncomputable theory

theorem solve_root_value :
  ∃ a b c : ℝ, a < b ∧ b < c ∧ (2016 * a^3 - 4 * a + 3 / real.sqrt 2016 = 0) ∧
                        (2016 * b^3 - 4 * b + 3 / real.sqrt 2016 = 0) ∧
                        (2016 * c^3 - 4 * c + 3 / real.sqrt 2016 = 0) ∧
                        (-1 / (a * b^2 * c) = 1354752) :=
by
  sorry

end solve_root_value_l219_219155


namespace cost_of_first_box_card_l219_219214

variable (x : ℝ)

/-- Conditions -/
def first_box_cost := x
def second_box_cost := 1.75
def cards_bought := 6
def total_cost := 18

/-- Statement to Prove -/
theorem cost_of_first_box_card : 6 * first_box_cost + 6 * second_box_cost = total_cost → 
  first_box_cost = 1.25 := 
by
  intro h
  sorry

end cost_of_first_box_card_l219_219214


namespace total_students_l219_219967

theorem total_students (initial_candies leftover_candies girls boys : ℕ) (h1 : initial_candies = 484)
  (h2 : leftover_candies = 4) (h3 : boys = girls + 3) (h4 : (2 * girls + boys) * (2 * girls + boys) = initial_candies - leftover_candies) :
  2 * girls + boys = 43 :=
  sorry

end total_students_l219_219967


namespace minimum_length_diagonal_BD_l219_219634

theorem minimum_length_diagonal_BD
  (A B C D I : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_eq1 : dist B C = 2)
  (h_eq2 : dist C D = 2)
  (h_incenter : Incenter I A B D)
  (h_AI : dist A I = 2) :
  ∃(BD_min : ℝ), BD_min = 2 * Real.sqrt 3 ∧ ∀(BD : ℝ), dist B D = BD → BD ≥ BD_min :=
sorry

end minimum_length_diagonal_BD_l219_219634


namespace spherical_plane_formation_l219_219588

theorem spherical_plane_formation {ρ θ φ : ℝ} (hθ : θ = π / 4) : 
  ∃ x y z : ℝ, (spherical_to_cartesian ρ θ φ = (x, y, z)) ∧ (x = y) :=
sorry

end spherical_plane_formation_l219_219588


namespace minimum_travel_time_correct_l219_219151

-- Definitions for conditions
def AC (t : ℝ) : ℝ := 10 * t
def AD (t : ℝ) : ℝ := 5 * t
def DC_length (t : ℝ) : ℝ := 10 * t - 5 * t
def E_meeting_time (t : ℝ) : ℝ := t + (10 * t - 5 * t) / 20

-- Positions on the segment DC
def DE (t : ℝ) : ℝ := 5 * (t / 4)
def EC (t : ℝ) : ℝ := 15 * (t / 4)
def CB (t : ℝ) : ℝ := 20 - 10 * t
def BE (t : ℝ) : ℝ := (20 - (25 * t / 4))

-- Trip end times
def T1 (t : ℝ) : ℝ := 2 + (5 * t / 8)
def T2 (t : ℝ) : ℝ := 4 - t

-- The maximum travel time function
def T (t : ℝ) : ℝ := max (T1 t) (T2 t)

-- The critical point
def critical_point : ℝ := 16 / 13

-- Minimum travel time at the critical point
noncomputable def minimum_travel_time : ℝ := 36 / 13

-- Proof statement
theorem minimum_travel_time_correct (t : ℝ) (h1 : AC t = 10 * t) (h2 : AD t = 5 * t)
    (h3 : DC_length t = 5 * t) (h4 : E_meeting_time t = t + (10 * t - 5 * t) / 20)
    (h5 : DE t = 5 * (t / 4)) (h6 : EC t = 15 * (t / 4)) (h7 : CB t = 20 - 10 * t)
    (h8 : BE t = (20 - (25 * t / 4))) (h9 : T1 t = 2 + (5 * t / 8))
    (h10 : T2 t = 4 - t) (h11 : T t = max (T1 t) (T2 t))
    (h12 : t = critical_point) : T t = minimum_travel_time := by
  sorry

end minimum_travel_time_correct_l219_219151


namespace geometric_seq_a5_value_l219_219105

theorem geometric_seq_a5_value (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) = a n * r) (h_pos : ∀ n, a n > 0) (h_a1a9 : a 1 * a 9 = 10) : a 5 = sqrt 10 := 
by
  sorry

end geometric_seq_a5_value_l219_219105


namespace spherical_coords_conversion_l219_219117

open Real

theorem spherical_coords_conversion :
  ∀ (ρ θ φ : ℝ),
    ρ > 0 → θ = 5 * π / 6 → φ = 9 * π / 5 → (0 ≤ θ ∧ θ < 2 * π) ∧ (0 ≤ φ ∧ φ ≤ π) →
    (∃ θ' φ', ρ = 5 ∧ θ' = 11 * π / 6 ∧ φ' = π / 5) :=
by {
  intros _ _ _ hρ hθ hφ _,
  use [11 * π / 6, π / 5],
  refine ⟨rfl, rfl, rfl⟩
}

end spherical_coords_conversion_l219_219117


namespace weight_of_pecans_l219_219404

theorem weight_of_pecans (total_weight_of_nuts almonds_weight pecans_weight : ℝ)
  (h1 : total_weight_of_nuts = 0.52)
  (h2 : almonds_weight = 0.14)
  (h3 : pecans_weight = total_weight_of_nuts - almonds_weight) :
  pecans_weight = 0.38 :=
  by
    sorry

end weight_of_pecans_l219_219404


namespace complex_number_sum_problem_l219_219268

theorem complex_number_sum_problem 
    (a b c d e f g h : ℂ)
    (h_b : b = 2)
    (h_g : g = -(a + c + e))
    (h_sum : (a + complex.I * b) + (c + complex.I * d) + (e + complex.I * f) + (g + complex.I * h) = complex.I * 3) :
    d + f + h = 1 :=
sorry

end complex_number_sum_problem_l219_219268


namespace sum_prime_factors_143_l219_219315

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219315


namespace circumcenter_not_lattice_point_l219_219577

noncomputable theory
open real euclidean_geometry

def lattice_point (p : ℤ × ℤ) := p.1 ∈ set.univ ∧ p.2 ∈ set.univ

structure triangle :=
(A B C : ℤ × ℤ)
(IsLattice : lattice_point A ∧ lattice_point B ∧ lattice_point C)
(AreaMinimal : ∀ (A' B' C' : ℤ × ℤ), 
  (∃ k : ℝ, k > 0 ∧ triangle_similar (A, B, C) (A', B', C')) →
  (triangle_area (A, B, C) ≤ triangle_area (A', B', C')))

def circumcenter (t : triangle) : ℝ × ℝ :=
sorry -- Assume we have a way to calculate the circumcenter

theorem circumcenter_not_lattice_point (t : triangle) :
  ¬ lattice_point (int.floor (circumcenter t).1, int.floor (circumcenter t).2) :=
sorry

end circumcenter_not_lattice_point_l219_219577


namespace impossible_circle_arrangement_l219_219133

theorem impossible_circle_arrangement :
  ¬ ∃ (arrangement : List ℕ), arrangement.length = 2017 ∧ (∀ (i : ℕ), 
    (17 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*])) ∨ 
    (21 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*]))) :=
sorry

end impossible_circle_arrangement_l219_219133


namespace find_a_value_l219_219478

theorem find_a_value :
  ∃ a : ℝ, a = 17 * π / 6 ∧ (∀ f : ℝ → ℝ, f = λ x => sin (sqrt (a^2 - x^2 - 2*x - 1)) →
  (finset.univ.filter (λ x => f x = 0.5)).card = 7) :=
begin
  sorry
end

end find_a_value_l219_219478


namespace train_passes_man_in_12_seconds_l219_219823

-- Definitions of the given conditions
def train_length : ℝ := 200 -- meters
def train_speed : ℝ := 68 * 1000 / 3600 -- converting kmph to m/s
def man_speed : ℝ := 8 * 1000 / 3600 -- converting kmph to m/s
def relative_speed : ℝ := train_speed - man_speed

-- Statement of the theorem to prove
theorem train_passes_man_in_12_seconds : (train_length / relative_speed) ≈ 12 := 
by sorry

end train_passes_man_in_12_seconds_l219_219823


namespace prob_exceeds_2100_l219_219148

open ProbabilityTheory MeasureTheory

noncomputable def normal_dist (μ σ : ℝ) : Measure ℝ := {
  to_fun := λ s, ∫ x in s, PDF (normalPDF μ σ) x,
  zero := by sorry,
  add := by sorry
}

theorem prob_exceeds_2100 :
  ∀ (ξ : Measure ℝ), 
    (ξ = normal_dist 2000 100) → 
    (∫⁻ x, ↑(if x > 2100 then 1 else 0) ∂ξ) = 0.1587 :=
by sorry

end prob_exceeds_2100_l219_219148


namespace log_function_fixed_point_l219_219682

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_function_fixed_point (a : ℝ) (ha1 : 0 < a) (ha2 : a ≠ 1) :
  let y := log_a a (2 - 1) + 2 in y = 2 :=
by
  sorry

end log_function_fixed_point_l219_219682


namespace cubes_difference_l219_219498

theorem cubes_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := 
sorry

end cubes_difference_l219_219498


namespace intersection_cases_l219_219535

noncomputable def function_y_domain (a : ℝ) : set ℝ :=
  { x | x ≥ 3 ∨ x ≤ -3 }

def set_B (a : ℝ) : set ℝ := { x | x < a }

theorem intersection_cases (a : ℝ) :
  let A := function_y_domain a in
  let B := set_B a in
  (a ≤ -3 → A ∩ B = { x | x < a }) ∧
  (-3 < a ∧ a ≤ 3 → A ∩ B = { x | x ≤ -3 }) ∧
  (a > 3 → A ∩ B = { x | x ≤ -3 ∨ 3 ≤ x ∧ x < a }) :=
by
  sorry

end intersection_cases_l219_219535


namespace prob_m_gt_n_l219_219511

theorem prob_m_gt_n :
  let A : Finset ℕ := {2, 4, 6, 8, 10}
  let B : Finset ℕ := {1, 3, 5, 7, 9}
  (A.card = 5) → (B.card = 5) →
  (∃ (m : ℕ) (hm : m ∈ A) (n : ℕ) (hn : n ∈ B), m > n) →
  (15 / 25 : ℝ) = 0.6 :=
by
  intros A B hA hB hex
  sorry

end prob_m_gt_n_l219_219511


namespace sequence_2005th_term_is_133_l219_219218

def sum_of_cubes_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum (λ x, x ^ 3)

def sequence (a : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := sum_of_cubes_of_digits (sequence n)

theorem sequence_2005th_term_is_133 : sequence 2005 2005 = 133 := 
  sorry

end sequence_2005th_term_is_133_l219_219218


namespace circle_equation_bisected_and_tangent_l219_219371

theorem circle_equation_bisected_and_tangent :
  (∃ x0 y0 r : ℝ, x0 = y0 ∧ (x0 + y0 - 2 * r) = 0 ∧ (∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2 → (x - 1)^2 + (y - 1)^2 = 2)) := sorry

end circle_equation_bisected_and_tangent_l219_219371


namespace same_number_of_acquaintances_l219_219573

theorem same_number_of_acquaintances (n : ℕ) (knows : fin n → fin n → Prop) :
  ∃ (i j : fin n), i ≠ j ∧ (finset.card (finset.filter (knows i) (finset.univ : finset (fin n))))
               = (finset.card (finset.filter (knows j) (finset.univ : finset (fin n)))) :=
by {
  sorry
}

end same_number_of_acquaintances_l219_219573


namespace harmonic_inequality_harmonic_step_increase_l219_219275

noncomputable def harmonic_sum (n : ℕ) : ℝ := ∑ i in (finset.range n).image (λ k, 1/(2^k))

theorem harmonic_inequality (n : ℕ) (hn : 0 < n) : harmonic_sum n > n / 2 :=
by sorry

theorem harmonic_step_increase (k : ℕ) :
  harmonic_sum (k + 1) = harmonic_sum k + (∑ i in (finset.range (2^k)).image (λ j, 1/(2^j))) :=
by sorry

end harmonic_inequality_harmonic_step_increase_l219_219275


namespace inequality_solution_set_l219_219008

theorem inequality_solution_set :
  { x : ℝ | (x^2 / ((x - 5)^2) ≥ 0) } = set.Ioo (-∞ : ℝ) 5 ∪ set.Ioo 5 ∞ :=
by
  sorry

end inequality_solution_set_l219_219008


namespace problem_a_l219_219455

theorem problem_a (a : ℝ) : 
  ∫∫ (λ (x y : ℝ), x^2 * y) in (region_eq (λ x, 0) (λ x, sqrt (2 * a * x - x^2))) = (4 / 5) * |a|^5 := 
sorry

end problem_a_l219_219455


namespace circle_chairs_subsets_count_l219_219739

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219739


namespace lily_disproves_tom_claim_l219_219489

-- Define the cards and the claim
inductive Card
| A : Card
| R : Card
| Circle : Card
| Square : Card
| Triangle : Card

def has_consonant (c : Card) : Prop :=
  match c with
  | Card.R => true
  | _ => false

def has_triangle (c : Card) : Card → Prop :=
  fun c' =>
    match c with
    | Card.R => c' = Card.Triangle
    | _ => true

def tom_claim (c : Card) (c' : Card) : Prop :=
  has_consonant c → has_triangle c c'

-- Proof problem statement:
theorem lily_disproves_tom_claim (c : Card) (c' : Card) : c = Card.R → ¬ has_triangle c c' → ¬ tom_claim c c' :=
by
  intros
  sorry

end lily_disproves_tom_claim_l219_219489


namespace sum_prime_factors_143_l219_219321

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219321


namespace decreasing_hyperbola_l219_219027

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end decreasing_hyperbola_l219_219027


namespace sum_of_areas_of_circles_l219_219263

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l219_219263


namespace sum_of_angles_l219_219902

theorem sum_of_angles (θ φ : ℝ) 
  (h1 : θ > 0) (h2 : θ < π / 2) 
  (h3 : φ > 0) (h4 : φ < π / 2)
  (h5 : tan θ = 1 / 3) 
  (h6 : sin φ = 1 / 3) : 
  θ + 2 * φ = π / 4 :=
begin
  sorry
end

end sum_of_angles_l219_219902


namespace sphere_surface_area_ratio_l219_219093

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio_l219_219093


namespace good_or_bad_of_prime_divides_l219_219019

-- Define the conditions
variables (k n n' : ℕ)
variables (h1 : k ≥ 2) (h2 : n ≥ k) (h3 : n' ≥ k)
variables (prime_divides : ∀ p, prime p → p ≤ k → (p ∣ n ↔ p ∣ n'))

-- Define what it means for a number to be good or bad
def is_good (m : ℕ) : Prop := ∃ strategy : ℕ → Prop, strategy m

-- Prove that either both n and n' are good or both are bad
theorem good_or_bad_of_prime_divides :
  (is_good n ∧ is_good n') ∨ (¬is_good n ∧ ¬is_good n') :=
sorry

end good_or_bad_of_prime_divides_l219_219019


namespace least_number_to_add_problem_statement_l219_219779

theorem least_number_to_add (n : ℕ) (divisor : ℕ) (remainder : ℕ) :
  (n % divisor = remainder) ∧ (remainder ≠ 0) → divisor - remainder = 1 :=
by
  intros h,
  sorry

theorem problem_statement : least_number_to_add 1057 23 22 :=
by
  sorry

end least_number_to_add_problem_statement_l219_219779


namespace geo_seq_product_l219_219970

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n * a m / a 1

theorem geo_seq_product
  {a : ℕ → ℝ}
  (h_pos : ∀ n, a n > 0)
  (h_seq : geometric_sequence a)
  (h_roots : ∃ x y, (x*x - 10 * x + 16 = 0) ∧ (y*y - 10 * y + 16 = 0) ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geo_seq_product_l219_219970


namespace sum_prime_factors_143_l219_219357

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219357


namespace cross_product_result_l219_219944

variable (v k : ℝ^3)

theorem cross_product_result :
  v × (v + k) = ⟨3, -1, 2⟩ →
  (2 • v + k) × (2 • v + 3 • k) = ⟨36, -12, 24⟩ := by
  sorry

end cross_product_result_l219_219944


namespace triangle_AC_5_sqrt_3_l219_219599

theorem triangle_AC_5_sqrt_3 
  (A B C : ℝ)
  (BC AC : ℝ)
  (h1 : 2 * Real.sin (A - B) + Real.cos (B + C) = 2)
  (h2 : BC = 5) :
  AC = 5 * Real.sqrt 3 :=
  sorry

end triangle_AC_5_sqrt_3_l219_219599


namespace impossible_tiling_8x8_impossible_tiling_8x8_missing_top_left_l219_219793

-- Definitions of the checkerboard and triminos
def checkerboard (n : ℕ) := fin n × fin n

def trimino := fin 3 → checkerboard 3

-- Conditions for tiling
def can_tile (board : set (fin 8 × fin 8)) (pieces : set (fin 3 × fin 1) := exists 
(t : trimino → fin n → checkerboard n), board = ⋃ x in pieces, t x)

-- The Lean 4 statement for part (i) and (ii)
theorem impossible_tiling_8x8 : 
  ¬ can_tile (set.univ : set (checkerboard 8)) :=
sorry

theorem impossible_tiling_8x8_missing_top_left :
 (¬ can_tile ({p : checkerboard 8 | p ≠ ⟨0, 0⟩} : set (checkerboard 8))) := 
sorry

end impossible_tiling_8x8_impossible_tiling_8x8_missing_top_left_l219_219793


namespace find_b_l219_219543

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end find_b_l219_219543


namespace anna_reading_time_l219_219447

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l219_219447


namespace daragh_favorites_taken_out_l219_219466

noncomputable def favorite_bears (initial_bears : ℕ) (eden_initial : ℕ) (eden_current : ℕ) (sisters : ℕ) : ℕ :=
  let eden_new_bears := eden_current - eden_initial in
  let each_sister_gets := eden_new_bears in
  let total_given := each_sister_gets * sisters in
  initial_bears - total_given

theorem daragh_favorites_taken_out : 
  favorite_bears 20 10 14 3 = 8 := 
by 
  -- This line ensures the code can build successfully.
  sorry

end daragh_favorites_taken_out_l219_219466


namespace find_lambda_l219_219925

theorem find_lambda (a b c : ℝ × ℝ) (λ : ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : 2 • a + b = (4, 2)) 
  (h3 : c = (1, λ)) 
  (h4 : ∃ k : ℝ, b = k • c) : λ = -1 :=
by
  sorry

end find_lambda_l219_219925


namespace subsets_with_at_least_four_adjacent_chairs_l219_219728

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219728


namespace part1_part2_l219_219524

-- Definitions provided as conditions
variable {n : ℕ}
variable {S_n : ℕ → ℝ}
variable {a_n : ℕ → ℝ}
variable {T_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}

-- Provided condition equations
def condition1 (n : ℕ) : ℝ := (λ n, S_n n)
def condition2 (n : ℕ) : Prop := (2 / 3) * S_n n = a_n n - (2 / 3) * n - 2

-- To be proven: the sequence {a_n + 1} is geometric
def is_geometric_sequence (f : ℕ → ℝ) : Prop := 
  ∃ r : ℝ, ∀ n : ℕ, f (n + 1) = r * f n

-- Proof (1): a_n + 1 forms a geometric sequence
theorem part1 : 
  (∀ n, condition2 n) → is_geometric_sequence (λ n, a_n n + 1) :=
by
  intros h
  sorry

-- Define b_n based on the sequence a_n
def b (n : ℕ) : ℝ := 1 / (a_n n + 2)

-- Sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b i

-- Proof (2): Sum of sequence b_n is less than 1/6
theorem part2 : 
  (∀ n, condition2 n) → (∀ n, T n < 1 / 6) :=
by
  intros h
  sorry

end part1_part2_l219_219524


namespace irrational_of_sqrt_3_l219_219367

noncomputable def is_irritational (x : ℝ) : Prop :=
  ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)

theorem irrational_of_sqrt_3 :
  is_irritational 0 = false ∧
  is_irritational 3.14 = false ∧
  is_irritational (-1) = false ∧
  is_irritational (Real.sqrt 3) = true := 
by
  -- Proof omitted
  sorry

end irrational_of_sqrt_3_l219_219367


namespace shaded_fraction_l219_219988

/-- 
Given a parallelogram containing two identical regular hexagons,
each hexagon being divisible into 6 equilateral triangles,
with the non-shaded area corresponding to 12 equilateral triangles,
and the total area of the parallelogram corresponding to 24 equilateral triangles,
prove that the fraction of the area of the parallelogram that is shaded is 1/2.
-/
theorem shaded_fraction (parallelogram_area hexagon_area non_shaded_area : ℕ)
  (h1 : hexagon_area = 6)
  (h2 : non_shaded_area = 2 * hexagon_area)
  (h3 : parallelogram_area = 24) :
  (parallelogram_area - non_shaded_area) = parallelogram_area / 2 := 
begin
  sorry
end

end shaded_fraction_l219_219988


namespace impossible_circle_arrangement_l219_219134

theorem impossible_circle_arrangement :
  ¬ ∃ (arrangement : List ℕ), arrangement.length = 2017 ∧ (∀ (i : ℕ), 
    (17 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*])) ∨ 
    (21 : ℤ) ∣ (arrangement.nth_le i (by simp [*]) - arrangement.nth_le ((i + 1) % 2017) (by simp [*]))) :=
sorry

end impossible_circle_arrangement_l219_219134


namespace expected_rainfall_7_days_l219_219430

/-- Probability definitions for each weather condition -/
def p_sunny : ℝ := 0.30
def p_rain_5_inches : ℝ := 0.20
def p_rain_7_inches : ℝ := 0.25
def p_cloudy : ℝ := 0.25
def p_no_rain : ℝ := p_sunny + p_cloudy

/-- Expected rainfall per day -/
def E_daily_rainfall : ℝ := (p_no_rain * 0) + (p_rain_5_inches * 5) + (p_rain_7_inches * 7)

/-- Expected rainfall over 7 days -/
def E_total_rainfall : ℝ := 7 * E_daily_rainfall

/-- Theorem: Prove the expected value of the total number of inches of rain over seven days -/
theorem expected_rainfall_7_days : E_total_rainfall = 19.3 := by
  have p_no_rain_eq : p_no_rain = 0.55 := by calc
    p_no_rain = p_sunny + p_cloudy : rfl
    ... = 0.30 + 0.25 : rfl
    ... = 0.55 : by norm_num

  have E_daily_rainfall_eq : E_daily_rainfall = 2.75 := by calc
    E_daily_rainfall = (p_no_rain * 0) + (p_rain_5_inches * 5) + (p_rain_7_inches * 7) : rfl
    ... = (0.55 * 0) + (0.20 * 5) + (0.25 * 7) : by rw [p_no_rain_eq]
    ... = 0 + 1 + 1.75 : by norm_num
    ... = 2.75 : by norm_num

  calc E_total_rainfall = 7 * E_daily_rainfall : rfl
    ... = 7 * 2.75 : by rw [E_daily_rainfall_eq]
    ... = 19.25 : by norm_num
    ... = 19.3 : by norm_num 

#align expected_rainfall_7_days expected_rainfall_7_days

end expected_rainfall_7_days_l219_219430


namespace runner_time_second_half_l219_219376

theorem runner_time_second_half (v : ℝ) (h1 : 20 / v + 4 = 40 / v) : 40 / v = 8 :=
by
  sorry

end runner_time_second_half_l219_219376


namespace max_neg_integers_eq_0_l219_219483

theorem max_neg_integers_eq_0 (a b c d : ℤ) (h : 5^a + 5^b = 2^c + 2^d + 2) : 
  (a < 0) → (b < 0) → (c < 0) → (d < 0) → false := 
sorry

end max_neg_integers_eq_0_l219_219483


namespace parabola_focus_hyperbola_vertex_distance_to_asymptote_l219_219012

-- Proving the coordinate of the focus for the given parabola
theorem parabola_focus (a : ℝ) (ha : a ≠ 0) : 
  let focus_x := 1 / (4 * a), focus_y := 0
  in (focus_x, focus_y) = (1 / (4 * a), 0) :=
by
  sorry

-- Proving the distance from the vertex to the asymptote for the given hyperbola
theorem hyperbola_vertex_distance_to_asymptote : 
  let vertex_x := sqrt 12, vertex_y := 0
  in dist_to_asymptote vertex_x vertex_y (line_equation 1 (sqrt 3) 0) = sqrt 3 :=
by
  sorry


end parabola_focus_hyperbola_vertex_distance_to_asymptote_l219_219012


namespace path_count_A_to_B_l219_219552

/-- 
Let A, B, C, D, E, F, and G be distinct points in the plane,
placed as shown in the problem figure. Given the condition that the
path must travel along the segments of the figure and cannot revisit
any point, the number of continuous paths from A to B is 16. 
-/
theorem path_count_A_to_B : 
  let points := {A, B, C, D, E, F, G}
  ∃ (paths : finset (list point)), 
    (∀ p ∈ paths, p.head = A ∧ p.last = B) ∧
    (∀ p ∈ paths, ∀ v ∈ p, v ∈ points) ∧
    (∀ p ∈ paths, ∀ (v₁ v₂ : point), v₁ ∈ p → v₂ ∈ p → v₁ ≠ v₂) ∧
    paths.card = 16 := sorry

end path_count_A_to_B_l219_219552


namespace locus_of_tangent_circle_centers_l219_219227

theorem locus_of_tangent_circle_centers :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 9}
  ∃ L, (L = ⋃ (p : ℝ × ℝ) (hp : tangent_to_both p C1 C2), {p}),
  is_hyperbola_and_line L :=
by sorry

end locus_of_tangent_circle_centers_l219_219227


namespace number_of_subsets_with_four_adj_chairs_l219_219716

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219716


namespace sum_prime_factors_of_143_l219_219349

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219349


namespace work_completion_time_l219_219379

-- Definitions for work rates
def work_rate_B : ℚ := 1 / 7
def work_rate_A : ℚ := 1 / 10

-- Statement to prove
theorem work_completion_time (W : ℚ) : 
  (1 / work_rate_A + 1 / work_rate_B) = 70 / 17 := 
by 
  sorry

end work_completion_time_l219_219379


namespace polar_circle_l219_219595

/-- In the polar coordinate system, the equation of a circle with its center
at (1, π/4) and a radius of 1 is ρ = 2cos(θ - π/4). -/
theorem polar_circle (ρ θ : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = π / 4 ∧ (ρ = 2 * cos (θ - y))) :=
sorry

end polar_circle_l219_219595


namespace hexagon_diagonals_parallel_l219_219272

theorem hexagon_diagonals_parallel {A B C A1 B1 C1 O : Type*} 
  [inhabited A] [inhabited B] [inhabited C] [inhabited A1] [inhabited B1] [inhabited C1] [inhabited O]
  (circle : set Type*) (is_inscribed : ∀ T,  set Type → Prop)
  (T1 T2 : set Type*) :
  is_inscribed T1 circle → 
  is_inscribed T2 circle →
  (A1 ∈ T2) → (B1 ∈ T2) → (C1 ∈ T2) →
  (A1 = midpoint circle B C) → (B1 = midpoint circle C A) → (C1 = midpoint circle A B) →
  ∃ O,  concylic A B C A1 B1 C1 ∧ hexagon_diagonals_intersect_at_point ⟨A, B, C, A1, B1, C1⟩ O ∧ hexagon_diagonals_parallel_to_sides ⟨A, B, C, A1, B1, C1⟩ T1 :=
sorry

end hexagon_diagonals_parallel_l219_219272


namespace total_light_path_length_l219_219623

def point := (ℝ × ℝ × ℝ)

def A : point := (0, 0, 0)
def E : point := (0, 0, 10)
def F : point := (0, 10, 10)
def H : point := (0, 10, 0)
def D : point := (10, 10, 0)
def P : point := (0, 3, 4)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

noncomputable def path_length : ℝ := distance A P + distance P D

theorem total_light_path_length :
  path_length = 5 + real.sqrt 165 :=
begin
  sorry
end

end total_light_path_length_l219_219623


namespace nadine_spent_money_l219_219184

theorem nadine_spent_money (table_cost : ℕ) (chair_cost : ℕ) (num_chairs : ℕ) 
    (h_table_cost : table_cost = 34) 
    (h_chair_cost : chair_cost = 11) 
    (h_num_chairs : num_chairs = 2) : 
    table_cost + num_chairs * chair_cost = 56 :=
by
  sorry

end nadine_spent_money_l219_219184


namespace common_root_divisibility_l219_219509

variables (a b c : ℤ)

theorem common_root_divisibility 
  (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) 
  : 3 ∣ (a + b + 2 * c) :=
sorry

end common_root_divisibility_l219_219509


namespace prime_divisor_condition_l219_219787

theorem prime_divisor_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : q ∣ 2^p - 1) : p ∣ q - 1 :=
  sorry

end prime_divisor_condition_l219_219787


namespace total_volume_is_10_l219_219802

noncomputable def total_volume_of_final_mixture (V : ℝ) : ℝ :=
  2.5 + V

theorem total_volume_is_10 :
  ∃ (V : ℝ), 
  (0.30 * 2.5 + 0.50 * V = 0.45 * (2.5 + V)) ∧ 
  total_volume_of_final_mixture V = 10 :=
by
  sorry

end total_volume_is_10_l219_219802


namespace route_bound_l219_219123

theorem route_bound (m n : ℕ) (f : ℕ → ℕ → ℕ) (A B : ℕ × ℕ) 
             (roads_EW roads_NS : ℕ) 
             (no_revisit : ∀ (r : list (ℕ × ℕ)), r.nodup) :
  roads_EW = n + 1 → roads_NS = m + 1 → 
  A = (0, 0) → B = (m, n) → f m n ≤ 2 ^ (m * n) :=
by
  sorry

end route_bound_l219_219123


namespace find_sum_of_f_powers_of_2_l219_219038

noncomputable def f : ℝ → ℝ
| x := 4 * (Real.log x / Real.log 3) * Real.log 3 / Real.log 2 + 233

theorem find_sum_of_f_powers_of_2 :
  f 2 + f 4 + f 8 + f 16 + f 32 + f 64 + f 128 + f 256 = 2008 :=
by
  sorry

end find_sum_of_f_powers_of_2_l219_219038


namespace measure_of_angle_C_l219_219639

theorem measure_of_angle_C (p q : Line) (A B C : Angle)
  (parallel_p_q : p ∥ q)
  (measure_A_eq_1_div_10_measure_B : measure A = (1 / 10) * measure B)
  (supplementary_B_C: measure B + measure C = 180) :
  measure C = 180 / 11 :=
sorry

end measure_of_angle_C_l219_219639


namespace book_arrangement_ways_l219_219581

open Nat

theorem book_arrangement_ways : 
  let m := 4  -- Number of math books
  let h := 6  -- Number of history books
  -- Number of ways to place a math book on both ends:
  let ways_ends := m * (m - 1)  -- Choices for the left end and right end
  -- Number of ways to arrange the remaining books:
  let ways_entities := 2!  -- Arrangements of the remaining entities
  -- Number of ways to arrange history books within the block:
  let arrange_history := factorial h
  -- Total arrangements
  let total_ways := ways_ends * ways_entities * arrange_history
  total_ways = 17280 := sorry

end book_arrangement_ways_l219_219581


namespace solve_for_t_l219_219204

theorem solve_for_t (t : ℝ)
  (h : 5 * 5^t + real.sqrt(25 * 25^t) = 150) : t = real.logb 5 15 :=
sorry

end solve_for_t_l219_219204


namespace anna_reading_time_l219_219442

theorem anna_reading_time:
  (∀ n : ℕ, n ∈ (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1) → True) →
  (let chapters_read := (Finset.range 31).filter (λ x, ¬ (∃ k : ℕ, k * 3 + 3 = x + 1)).card,
  reading_time := chapters_read * 20,
  hours := reading_time / 60 in
  hours = 7) :=
by
  intros
  let chapters_read := (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1).card
  have h1 : chapters_read = 21 := by sorry
  let reading_time := chapters_read * 20
  have h2 : reading_time = 420 := by sorry
  let hours := reading_time / 60
  have h3 : hours = 7 := by sorry
  exact h3

end anna_reading_time_l219_219442


namespace find_a22_l219_219248

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l219_219248


namespace even_number_of_diagonals_intersected_l219_219567

theorem even_number_of_diagonals_intersected
  (polygon : convex_polygon 2009)
  (line : affine_line ℝ)
  (h_intersects : intersects polygon line)
  (h_no_vertex : ∀ v ∈ polygon.vertices, ¬ line.contains v) :
  ∃ k : ℕ, intersects_diagonals polygon line = 2 * k := 
sorry

end even_number_of_diagonals_intersected_l219_219567


namespace part1_part2_l219_219917

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  Real.log x + 0.5 * m * x^2 - 2 

def perpendicular_slope_condition (m : ℝ) : Prop := 
  let k := (1 / 1 + m)
  k = -1 / 2

def inequality_condition (m : ℝ) : Prop := 
  ∀ x > 0, 
  Real.log x - 0.5 * m * x^2 + (1 - m) * x + 1 ≤ 0

theorem part1 : perpendicular_slope_condition (-3/2) :=
  sorry

theorem part2 : ∃ m : ℤ, m ≥ 2 ∧ inequality_condition m :=
  sorry

end part1_part2_l219_219917


namespace katrina_cookies_left_l219_219609

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l219_219609


namespace sum_prime_factors_143_is_24_l219_219289

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219289


namespace trigonometric_identity_proof_l219_219497

variable (α : Real)

theorem trigonometric_identity_proof (h1 : Real.tan α = 4 / 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7 / 5 :=
by
  sorry

end trigonometric_identity_proof_l219_219497


namespace diagonal_section_area_l219_219225

-- Define the variables and assumptions
variables (a : ℝ) (α : ℝ)

-- Define the theorem statement
theorem diagonal_section_area 
  (h1 : a > 0) 
  (h2 : 0 < α ∧ α < π / 2) :
  let cos_alpha := Real.cos α in
  let term1 := cos_alpha^2 in
  let term2 := -2 * (Real.cos (2 * α)) in
  2 * a^2 * term1 * sqrt term2 = 2 * a^2 * cos^2 (α / 2) * sqrt (-2 * Real.cos (2 * α)) :=
sorry

end diagonal_section_area_l219_219225


namespace boy_girl_pairs_l219_219800

theorem boy_girl_pairs (X : ℕ) (boys groups : ℕ) :
  boys = 15 →
  groups = 5 →
  1.5 * (boys - groups) = 20 - groups →
  let boy_pairs := boys - groups in
  let girl_pairs := 20 - groups in
  boy_pairs = 10 →
  girl_pairs = 15 →
  (15 + 20 - (boy_pairs + girl_pairs)) = 10 :=
by
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h4, h5],
  sorry

end boy_girl_pairs_l219_219800


namespace tank_fraction_after_adding_water_l219_219936

noncomputable def fraction_of_tank_full 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  : ℚ :=
(initial_fraction * total_capacity + additional_water) / total_capacity

theorem tank_fraction_after_adding_water 
  (initial_fraction : ℚ) 
  (additional_water : ℚ) 
  (total_capacity : ℚ) 
  (h_initial : initial_fraction = 3 / 4) 
  (h_addition : additional_water = 4) 
  (h_capacity : total_capacity = 32) 
: fraction_of_tank_full initial_fraction additional_water total_capacity = 7 / 8 :=
by
  sorry

end tank_fraction_after_adding_water_l219_219936


namespace girl_attendance_l219_219450

theorem girl_attendance (g b : ℕ) (h1 : g + b = 1500) (h2 : (3 / 4 : ℚ) * g + (1 / 3 : ℚ) * b = 900) :
  (3 / 4 : ℚ) * g = 720 :=
by
  sorry

end girl_attendance_l219_219450


namespace equation_system_equiv_l219_219830

theorem equation_system_equiv :
  ∀ x : ℝ, 
  ((x^2 + x + 1) * (3 * x + 4) * (-7 * x + 2) * (2 * x - real.sqrt 5) * (-12 * x - 16) = 0 ↔ 
  (3 * x + 4 = 0 ∨ -7 * x + 2 = 0 ∨ 2 * x - real.sqrt 5 = 0 ∨ -12 * x - 16 = 0)) := 
by
  sorry

end equation_system_equiv_l219_219830


namespace area_of_triangle_l219_219960

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : b = 2) (h₂ : c = 2 * Real.sqrt 2) (h₃ : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - B - C) = Real.sqrt 3 + 1 := 
by
  sorry

end area_of_triangle_l219_219960


namespace circle_areas_sum_l219_219258

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l219_219258


namespace find_60th_pair_l219_219536

-- Given conditions
def sequence_term (n k : ℕ) : ℕ × ℕ :=
  (k, n - k)

def sum_pairs_count (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Lemma stating the proposition corresponding to the problem.
theorem find_60th_pair :
  ∃ p : ℕ × ℕ, (∃ n : ℕ, n = 11 ∧ p = (5, 7)) ∧ sum_pairs_count 10 < 60 ∧ 60 ≤ sum_pairs_count 11 := 
begin
  sorry
end

end find_60th_pair_l219_219536


namespace sequence_of_digits_l219_219555

-- Define the domain of digits and the conditions
def is_even (n : Nat) : Prop := n % 2 = 0
def is_odd (n : Nat) : Prop := n % 2 = 1

theorem sequence_of_digits :
  ∃ (seq : Fin 8 → Nat), 
    (∀ i, seq i < 10) ∧
    is_even (seq 0) ∧
    (∀ i, 0 < i → seq i % 2 ≠ seq (i - 1) % 2) ∧
    (5 * 5 ^ 7 = 390625) :=
begin
  -- Proof skipped
  sorry
end

end sequence_of_digits_l219_219555


namespace problem_part1_problem_part2_l219_219531

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 / a^2
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := f x a - (3 * b * x / a^2) + 3

theorem problem_part1 (a b : ℝ) (hx : a^2 ≠ 0)
  (h : (∂ (g x a b) ∂x).eval 1 = 0) :
  g x a b = x^3 - 3x + 3 := sorry

theorem problem_part2 (a b m : ℝ)
  (hx : a^2 ≠ 0)
  (h_deriv : ∀ x ∈ [-1, 1], (∂ (g x a b) ∂x).eval x ≥ 0)
  (h_ineq : ∀ x ∈ [-1, 1], b^2 - m * b + 4 ≥ g x a b) :
  ∀ m, m ≥ 3 := sorry

end problem_part1_problem_part2_l219_219531


namespace sum_of_coefficients_l219_219207

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
  let expr := (10 * d - 3 + 16 * d^2) + (4 * d + 7) in
  let a := 14 in
  let b := 4 in
  let c := 16 in
  (a + b + c) = 34 :=
by 
  let expr := (10 * d - 3 + 16 * d^2) + (4 * d + 7)
  let a := 14
  let b := 4
  let c := 16
  -- Proof is omitted.
  sorry

end sum_of_coefficients_l219_219207


namespace number_of_clients_l219_219377

theorem number_of_clients (num_cars num_selections_per_car num_cars_per_client total_selections num_clients : ℕ)
  (h1 : num_cars = 15)
  (h2 : num_selections_per_car = 3)
  (h3 : num_cars_per_client = 3)
  (h4 : total_selections = num_cars * num_selections_per_car)
  (h5 : num_clients = total_selections / num_cars_per_client) :
  num_clients = 15 := 
by
  sorry

end number_of_clients_l219_219377


namespace factorial_divides_l219_219163

theorem factorial_divides {m n : ℕ} : m! * n! * (m + n)! ∣ (2 * m)! * (2 * n)! :=
sorry

end factorial_divides_l219_219163


namespace rational_resistance_circuit_l219_219022

theorem rational_resistance_circuit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ circuit : Type, (circuit → ℝ) ∧
  (∀ u : circuit, u = 1) ∧
  (total_resistance circuit = (a : ℝ) / (b : ℝ)) :=
sorry

end rational_resistance_circuit_l219_219022


namespace arbelos_equals_circle_area_l219_219033

variable {r1 r2 : ℝ}

-- Definitions
def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

def arbelos_area (r1 r2 : ℝ) : ℝ :=
  semicircle_area (r1 + r2) - (semicircle_area r1 + semicircle_area r2)

def circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  Real.pi * r^2

-- Given
axiom perpendicular_from_D_to_AC (D A C B : Point) : Perpendicular D A C B
axiom semicircle_ABC (A B C D : Point) : Semicircle A B C
axiom semicircle_AMD (A M D : Point) : Semicircle A M D
axiom semicircle_DNC (D N C : Point) : Semicircle D N C

-- Proof statement
theorem arbelos_equals_circle_area (D A C B : Point) :
  perpendicular_from_D_to_AC D A C B →
  semicircle_ABC A B C D →
  semicircle_AMD A M D →
  semicircle_DNC D N C →
  arbelos_area r1 r2 = circle_area (2 * Real.sqrt (r1 * r2)) := by
  sorry

end arbelos_equals_circle_area_l219_219033


namespace range_of_a_l219_219078

open Set

def A : Set ℝ := { x | x ≤ 1 ∨ x ≥ 3 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 1 }

theorem range_of_a (a : ℝ) (h : B a ⊆ A) :
  a ≤ 0 ∨ a ≥ 3 :=
by 
  sorry

end range_of_a_l219_219078


namespace range_of_a_l219_219907

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (x + y + a = 0) ∧ (x^2 + y^2 = 1) ∧ 
  (|vector.OA + vector.OB| ≥ |vector.AB|)) →
  (a ∈ (-√2, -1] ∪ [1, √2)) :=
sorry

end range_of_a_l219_219907


namespace sum_prime_factors_of_143_l219_219339

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219339


namespace num_extreme_points_f_l219_219040

def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

def domain := set.Icc 0 (2011 * Real.pi)

theorem num_extreme_points_f :
  ∃ (n : ℕ), (∀ (x : ℝ), x ∈ domain → (f' x = 0 → (n = 1005))) :=
sorry

end num_extreme_points_f_l219_219040


namespace obtain_x_squared_obtain_xy_l219_219602

theorem obtain_x_squared (x y : ℝ) (hx : x ≠ 1) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x^2 :=
by
  sorry

theorem obtain_xy (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x * y :=
by
  sorry

end obtain_x_squared_obtain_xy_l219_219602


namespace imaginary_part_of_z_l219_219520

-- Given definitions:
def i : ℂ := complex.I
def z : ℂ := (1 - i) / (3 - i)

-- Theorem statement to prove the imaginary part of z is -1/5
theorem imaginary_part_of_z : complex.im z = -1 / 5 := sorry

end imaginary_part_of_z_l219_219520


namespace cos_identity_l219_219055

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 - α) = sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) = - (sqrt 3 / 3) := by
  sorry

end cos_identity_l219_219055


namespace carpet_size_l219_219999

theorem carpet_size (width length : ℕ) (h_width : width = 2) (h_length : length = 5) : width * length = 10 := by
  calc
    width * length = 2 * 5 : by rw [h_width, h_length]
    ...            = 10     : by norm_num


end carpet_size_l219_219999


namespace problem_1_problem_2_l219_219540

-- Definition of sets A and B as in the problem's conditions
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | x > 2 ∨ x < -2}
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Prove that A ∩ B is as described
theorem problem_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} := by
  sorry

-- Prove that a ≥ 6 given the conditions in the problem
theorem problem_2 (a : ℝ) : (A ⊆ C a) → a ≥ 6 := by
  sorry

end problem_1_problem_2_l219_219540


namespace problem_statement_problem_statement_2_l219_219541

noncomputable def A (m : ℝ) : Set ℝ := {x | x > 2^m}
noncomputable def B : Set ℝ := {x | -4 < x - 4 ∧ x - 4 < 4}

theorem problem_statement (m : ℝ) (h1 : m = 2) :
  (A m ∪ B = {x | x > 0}) ∧ (A m ∩ B = {x | 4 < x ∧ x < 8}) :=
by sorry

theorem problem_statement_2 (m : ℝ) (h2 : A m ⊆ {x | x ≤ 0 ∨ 8 ≤ x}) :
  3 ≤ m :=
by sorry

end problem_statement_problem_statement_2_l219_219541


namespace andy_older_than_rahim_l219_219963

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end andy_older_than_rahim_l219_219963


namespace new_average_age_l219_219674

theorem new_average_age (avg_age_students : ℕ) (num_students : ℕ) (teacher_age : ℕ) :
  (avg_age_students * num_students + teacher_age) / (num_students + 1) = 16 :=
by
  -- conditions
  let avg_age_students := 15
  let num_students := 10
  let teacher_age := 26
  -- proposition to prove
  calc
    (avg_age_students * num_students + teacher_age) / (num_students + 1)
      = (15 * 10 + 26) / (10 + 1) : by rw [←nat.add_mul, ←nat.add_assoc] sorry
      ... = 176 / 11 : by rw [←nat.one_mul, nat.mul_div_cancel _] sorry
      ... = 16 : by norm_num

end new_average_age_l219_219674


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219754

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219754


namespace anna_reading_time_l219_219445

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l219_219445


namespace sum_prime_factors_143_l219_219316

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219316


namespace velocity_of_current_l219_219811

theorem velocity_of_current (v : ℝ) 
  (row_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h_row_speed : row_speed = 5)
  (h_distance : distance = 2.4)
  (h_total_time : total_time = 1)
  (h_equation : distance / (row_speed + v) + distance / (row_speed - v) = total_time) :
  v = 1 :=
sorry

end velocity_of_current_l219_219811


namespace triangle_area_question_l219_219271

open Real

noncomputable def triangle_area (a b c : ℝ) (R T S : ℝ × ℝ) : ℝ :=
  let x := (b - a) / 2
  let y := sqrt (17 ^ 2 - ((b - a) / 2) ^ 2)
  let RU := 120 / 17
  let SU := (17 / 2) - (56 / 17)
  (SU * y / 2) * sqrt 7 / 2

theorem triangle_area_question :
  ∀ (PR QR : ℝ) (TR : ℝ × ℝ), PR = 8 → QR = 15 → (triangle_area 0 17 TR).nat_abs = 1300 :=
by
  intros PR QR TR h1 h2
  have h3 : (17 : ℝ) = sqrt (PR ^ 2 + QR ^ 2) by
    simp [PR, QR, h1, h2]
    norm_num
  have h4 : PR ^ 2 + QR ^ 2 = 17 ^ 2 := by
    simp [PR, QR, h1, h2]
    norm_num
  have area_expression : (triangle_area PR QR TR).nat_abs = 1300 := by
    sorry  -- Placeholder for the actual proof implementation
  exact area_expression

end triangle_area_question_l219_219271


namespace percentage_enclosed_by_hexagons_is_50_l219_219411

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

noncomputable def total_tiling_unit_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * square_area s

noncomputable def percentage_enclosed_by_hexagons (s : ℝ) : ℝ :=
  (hexagon_area s / total_tiling_unit_area s) * 100

theorem percentage_enclosed_by_hexagons_is_50 (s : ℝ) : percentage_enclosed_by_hexagons s = 50 := by
  sorry

end percentage_enclosed_by_hexagons_is_50_l219_219411


namespace compare_stability_with_variance_l219_219108

theorem compare_stability_with_variance (scores_A scores_B : Fin 5 → ℝ) 
(h_avg : (∑ i, scores_A i) / 5 = (∑ i, scores_B i) / 5) :
  (∑ i, (scores_A i - (∑ i, scores_A i) / 5)^2) / 5 = (∑ i, (scores_B i - (∑ i, scores_B i) / 5)^2) / 5 → 
  ∑ i, (scores_A i - ∑ i, scores_A i / 5)^2 = ∑ i, (scores_B i - ∑ i, scores_B i / 5)^2 :=
sorry

end compare_stability_with_variance_l219_219108


namespace max_monthly_donation_l219_219389

def charity_per_day (k : ℕ) : ℕ := k / 3

def total_working_hours (L : ℕ) (days : ℕ) : ℕ := 3 * L * days

def total_rest_hours (k_per_day : ℕ) (days : ℕ) : ℕ := k_per_day * days

noncomputable def monthly_donation (k_total : ℕ) : ℕ := k_total / 3

theorem max_monthly_donation 
  (cost_per_hour : ℕ) (investment_income : ℕ) (monthly_expenses : ℕ)
  (working_days : ℕ) (sleep_hours : ℕ) (rest_hours_per_day : ℕ) 
  (L : ℕ) (k_per_day : ℕ)
  (daily_income := cost_per_hour * L) 
  (max_L := (24 - sleep_hours - 2 * L - rest_hours_per_day) / 3)
  (total_income := investment_income + (cost_per_hour * max_L * working_days))
  (total_expenses := monthly_expenses + charity_per_day(k_per_day) * working_days) :
  monthly_donation (total_rest_hours k_per_day working_days) = 70 :=
by
  sorry

end max_monthly_donation_l219_219389


namespace contrapositive_x_squared_l219_219676

theorem contrapositive_x_squared :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := 
sorry

end contrapositive_x_squared_l219_219676


namespace ratio_third_median_altitude_l219_219111

noncomputable theory
open_locale classical

variables {A B C A' B' M : Type*}
variable [is_scalar_tower A B C]
variables (triangle : triangle A B C)
variables [scalene_triangle triangle]
variables (m_A' : median A A') (m_B' : median B B')
variables (h_B : altitude B) (h_C : altitude C)

def conditions := 
  (m_A'.length = h_B.length) ∧ 
  (m_B'.length = h_C.length)

theorem ratio_third_median_altitude (h : conditions triangle m_A' m_B' h_B h_C) :
  ratio_third_median_altitude = 7 / 2 :=
sorry

end ratio_third_median_altitude_l219_219111


namespace percentage_of_girls_l219_219969

theorem percentage_of_girls (B G : ℕ) 
  (h : G + (0.5 * B) = 1.5 * (0.5 * B)) : 
  (G : ℝ) / (B + G) * 100 = 20 := by
  sorry

end percentage_of_girls_l219_219969


namespace sum_of_x_values_l219_219486

theorem sum_of_x_values (x : ℝ) (cos : ℝ → ℝ) :
  (120 < x ∧ x < 220) →
  (cos (4 * x))^3 + (cos (6 * x))^3 > 9 * (cos (5 * x))^3 * (cos x)^3 →
  let valid_x := [135, 126, 162, 198, 150, 180] in
  ∑ x in valid_x, x = 951 := 
sorry

end sum_of_x_values_l219_219486


namespace proof_of_problem_l219_219765

theorem proof_of_problem (a b : ℝ) (h1 : a > b) (h2 : a * b = a / b) : b = 1 ∧ 0 < a :=
by
  sorry

end proof_of_problem_l219_219765


namespace chairs_adjacent_subsets_l219_219734

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219734


namespace complex_modulus_power_theorem_l219_219484

noncomputable def complex_modulus_power : ℂ :=
  let z : ℂ := 2 + 3 * Real.sqrt 2 * Complex.i
  in |z^4|

theorem complex_modulus_power_theorem : complex_modulus_power = 484 := by
  sorry

end complex_modulus_power_theorem_l219_219484


namespace find_angle_A_l219_219598

-- Given triangle with vertices A, B, C and point E meeting the conditions
variables {A B C E : Type} [IsTriangle A B C]

-- Conditions
variable (h1 : AB = AC) -- AB = AC
variable (h2 : AngleBisector B A C E) -- E is the angle bisector of ∠B
variable (h3 : BC = BE + EA) -- BC = BE + EA

-- Theorem to prove
theorem find_angle_A (h1 : AB = AC) (h2 : AngleBisector B A C E) (h3 : BC = BE + EA) : 
  Angle A = 100 :=
by 
  sorry

end find_angle_A_l219_219598


namespace sea_lions_count_l219_219378

theorem sea_lions_count (S P : ℕ) (h1 : 11 * S = 4 * P) (h2 : P = S + 84) : S = 48 := 
by {
  sorry
}

end sea_lions_count_l219_219378


namespace car_trip_proof_l219_219270

def initial_oil_quantity (y : ℕ → ℕ) : Prop :=
  y 0 = 50

def consumption_rate (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = y (t - 1) - 5

def relationship_between_y_and_t (y : ℕ → ℕ) : Prop :=
  ∀ t, y t = 50 - 5 * t

def oil_left_after_8_hours (y : ℕ → ℕ) : Prop :=
  y 8 = 10

theorem car_trip_proof (y : ℕ → ℕ) :
  initial_oil_quantity y ∧ consumption_rate y ∧ relationship_between_y_and_t y ∧ oil_left_after_8_hours y :=
by
  -- the proof goes here
  sorry

end car_trip_proof_l219_219270


namespace max_soap_boxes_l219_219373

theorem max_soap_boxes (l_carton w_carton h_carton l_soap w_soap h_soap : ℝ) 
  (H_carton : l_carton = 30 ∧ w_carton = 42 ∧ h_carton = 60)
  (H_soap : l_soap = 7 ∧ w_soap = 6 ∧ h_soap = 5) :
  let V_carton := l_carton * w_carton * h_carton,
      V_soap_box := l_soap * w_soap * h_soap in
  V_carton / V_soap_box = 360 :=
by
  sorry

end max_soap_boxes_l219_219373


namespace origin_outside_circle_l219_219912

theorem origin_outside_circle {a : ℝ} (ha : 0 < a ∧ a < 1) :
    let circle_eq := λ x y, x^2 + y^2 + 2*a*x + 2*y + (a - 1)^2 in
    ¬ circle_eq 0 0 = 0 :=
by
-- math proof
sorry

end origin_outside_circle_l219_219912


namespace stratified_sampling_correct_l219_219407

theorem stratified_sampling_correct : 
  ∃ (n_f n_s n_j : ℕ), 
    n_f + n_s + n_j = 45 ∧ 
    n_f = 15 ∧ 
    n_s = 10 ∧ 
    n_j = 20 :=
by
  existsi 15, 10, 20
  simp
  sorry

end stratified_sampling_correct_l219_219407


namespace S_eq_T_l219_219952

-- Define the sets S and T
def S : Set ℤ := {x | ∃ n : ℕ, x = 3 * n + 1}
def T : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 2}

-- Prove that S = T
theorem S_eq_T : S = T := 
by {
  sorry
}

end S_eq_T_l219_219952


namespace find_BF_l219_219993

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

theorem find_BF {A B C D E F: ℝ × ℝ}
  (hD: D = midpoint A B) 
  (hE: E = midpoint B C) 
  (hF: F = midpoint C A) 
  (hAB: dist A B = 10) 
  (hCD: dist C D = 9) 
  (h_perp: ∃ (t₀ t₁: ℝ), (C = (1 - t₀) • D + t₀ • E) ∧ (C - D) ⬝ (A - E) = 0) :
  dist B F = 3 * Real.sqrt 13 :=
begin
  sorry
end

end find_BF_l219_219993


namespace sum_of_prime_factors_of_143_l219_219305

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219305


namespace minimum_jars_to_fill_large_pack_without_excess_l219_219421

-- Define the given conditions
def standard_jar_weight : ℕ := 140
def large_pack_weight : ℕ := 2000

-- Define the proof problem statement
theorem minimum_jars_to_fill_large_pack_without_excess : 
  ∃ n : ℕ, n = 15 ∧ n * standard_jar_weight >= large_pack_weight ∧ (n - 1) * standard_jar_weight < large_pack_weight :=
begin
  sorry -- proof is omitted
end

end minimum_jars_to_fill_large_pack_without_excess_l219_219421


namespace ellipse_problem_l219_219530

noncomputable def point_coordinates (x y b : ℝ) : Prop :=
  x = 1 ∧ y = 1 ∧ (4 * x^2 = 4) ∧ (4 * b^2 / (4 + b^2) = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 - b^2)) / a

theorem ellipse_problem (b : ℝ) (h₁ : 4 * b^2 / (4 + b^2) = 1) :
  ∃ x y, point_coordinates x y b 
  ∧ eccentricity 2 b = Real.sqrt 6 / 3 := 
by 
  sorry

end ellipse_problem_l219_219530


namespace sqrt_inequality_l219_219044

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  sqrt (a / (a + 3 * b)) + sqrt (b / (b + 3 * a)) ≥ 1 := 
by
  sorry

end sqrt_inequality_l219_219044


namespace sum_prime_factors_143_l219_219356

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219356


namespace equidistant_point_x_axis_l219_219770

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end equidistant_point_x_axis_l219_219770


namespace sin_squared_alpha_plus_pi_over_4_l219_219496

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h : sin (2 * α) = 1 / 4) : 
  sin^2 (α + π / 4) = 5 / 8 := by
  sorry

end sin_squared_alpha_plus_pi_over_4_l219_219496


namespace three_distinct_real_solutions_l219_219953

theorem three_distinct_real_solutions (b c : ℝ):
  (∀ x : ℝ, x^2 + b * |x| + c = 0 → x = 0) ∧ (∃! x : ℝ, x^2 + b * |x| + c = 0) →
  b < 0 ∧ c = 0 :=
by {
  sorry
}

end three_distinct_real_solutions_l219_219953


namespace number_of_subsets_with_four_adj_chairs_l219_219718

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219718


namespace sum_even_integers_102_to_200_l219_219704

theorem sum_even_integers_102_to_200 : 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 1 102), 2 * k) = 2550 →
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) + 1250 = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100) + ∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 1300 → 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 200), 2 * i) = 1250 :=
begin
  sorry
end

end sum_even_integers_102_to_200_l219_219704


namespace proof_problem_l219_219381

-- Define the conditions: n is a positive integer and (n(n + 1) / 3) is a square
def problem_condition (n : ℕ) : Prop :=
  ∃ m : ℕ, n > 0 ∧ (n * (n + 1)) = 3 * m^2

-- Define the proof problem: given the condition, n is a multiple of 3, n+1 and n/3 are squares
theorem proof_problem (n : ℕ) (h : problem_condition n) : 
  (∃ a : ℕ, n = 3 * a^2) ∧ 
  (∃ b : ℕ, n + 1 = b^2) ∧ 
  (∃ c : ℕ, n = 3 * c^2) :=
sorry

end proof_problem_l219_219381


namespace pure_imaginary_condition_l219_219882

variable {a b c d : ℝ}
def z1 := a + b * complex.i
def z2 := c + d * complex.i

theorem pure_imaginary_condition (ha : z1 + z2 = (0 : ℝ) + (b + d) * complex.i) : a + c = 0 ∧ b + d ≠ 0 :=
sorry

end pure_imaginary_condition_l219_219882


namespace law_I_false_law_II_true_law_III_true_l219_219621

namespace AveragedOperation

def averaged (a b : ℝ) : ℝ := (a + b) / 2

theorem law_I_false (x y z : ℝ) : x @ (y - z) ≠ (x @ y) - (x @ z) :=
begin
    have h_lhs : x @ (y - z) = (x + (y - z)) / 2 := by refl,
    have h_rhs : (x @ y) - (x @ z) = (x + y) / 2 - (x + z) / 2 := by refl,
    simp [averaged, h_lhs, h_rhs],
    intro h,
    have : x + y - z ≠ y - z,
    {
        from (λ hxy, by linarith [hxy])
    },
    contradiction,
end

theorem law_II_true (x y z : ℝ) : x - (y @ z) = (x - y) @ (x - z) :=
begin
    have h_lhs : x - (y @ z) = x - ((y + z) / 2) := by refl,
    have h_rhs : (x - y) @ (x - z) = ((x - y) + (x - z)) / 2 := by refl,
    simp [averaged, h_lhs, h_rhs, (sub_add_sub_cancel x y z), (add_comm x x)],
end

theorem law_III_true (x y z : ℝ) : x @ (y @ z) = (x @ y) @ (x @ z) :=
begin
    have h_lhs : x @ (y @ z) = (x + (y + z) / 2) / 2 := by refl,
    have h_rhs : (x @ y) @ (x @ z) = ((x + y) / 2 + (x + z) / 2) / 2 := by refl,
    simp [averaged, h_lhs, h_rhs],
    field_simp,
    ring,
end

end AveragedOperation

end law_I_false_law_II_true_law_III_true_l219_219621


namespace negation_of_P_is_exists_ge_1_l219_219036

theorem negation_of_P_is_exists_ge_1 :
  let P := ∀ x : ℤ, x < 1
  ¬P ↔ ∃ x : ℤ, x ≥ 1 := by
  sorry

end negation_of_P_is_exists_ge_1_l219_219036


namespace vec_parallel_l219_219924

-- Define the vectors and the condition of parallel
variables {x : ℝ}

def a : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (2, -3)

-- Theorem to prove that x = -2/3 when vectors a and b are parallel
theorem vec_parallel (h : (∃ k : ℝ, a = (k * 2, k * -3))) : x = -2 / 3 :=
sorry

end vec_parallel_l219_219924


namespace solve_equation_l219_219205

theorem solve_equation :
  ∀ x : ℝ, x ≠ 2 → (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 5 ↔ x = -2 :=
by
  intros x h_x_ne_2
  split
  {
    intro h_eqn
    sorry
  }
  {
    intro h_x_eq_minus_2
    sorry
  }

end solve_equation_l219_219205


namespace division_result_l219_219380

theorem division_result : 3486 / 189 = 18.444444444444443 := 
by sorry

end division_result_l219_219380


namespace john_annual_profit_l219_219144

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l219_219144


namespace max_intersections_of_lines_l219_219640

theorem max_intersections_of_lines :
  (∀ n : ℕ, 1 ≤ n → n ≤ 25 → parallel (L (4 * n)) ∧
            ((1 ≤ 4 * n - 3 ∧ 4 * n - 3 ≤ 100) → they_pass_through_point (L (4 * n - 3)))) →
  max_intersections 100 = 4351 :=
sorry

end max_intersections_of_lines_l219_219640


namespace samantha_birth_year_l219_219680

theorem samantha_birth_year 
  (first_amc8 : ℕ)
  (amc8_annual : ∀ n : ℕ, n ≥ first_amc8)
  (seventh_amc8 : ℕ)
  (samantha_age : ℕ)
  (samantha_birth_year : ℕ)
  (move_year : ℕ)
  (h1 : first_amc8 = 1983)
  (h2 : seventh_amc8 = first_amc8 + 6)
  (h3 : seventh_amc8 = 1989)
  (h4 : samantha_age = 14)
  (h5 : samantha_birth_year = seventh_amc8 - samantha_age)
  (h6 : move_year = seventh_amc8 - 3) :
  samantha_birth_year = 1975 :=
sorry

end samantha_birth_year_l219_219680


namespace sin_double_alpha_l219_219880

theorem sin_double_alpha (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end sin_double_alpha_l219_219880


namespace angle_C_eq_60_l219_219479

-- Definitions for the triangle and the conditions
variable {A B C : Type*}
variable [IsTriangle A B C] -- Assuming some property or structure defining a triangle

-- Definitions for circumcircle and orthocenter
variable R (O M : Point) -- Circumcircle radius and points
variable [IsCircumcenter O A B C] -- O is the circumcenter of triangle ABC
variable [IsOrthocenter M A B C] -- M is the orthocenter of triangle ABC
variable [EqDist C M C O] -- Distance from vertex C to orthocenter M equals the radius of circumcircle O

-- The angle to be proven
theorem angle_C_eq_60 (r : ℝ) (O M : Point) [IsCircumcenter O A B C] [IsOrthocenter M A B C] [EqDist C M r] :
  ∠A B C = 60 :=
sorry

end angle_C_eq_60_l219_219479


namespace n_times_2pow_nplus1_plus_1_is_square_l219_219877

theorem n_times_2pow_nplus1_plus_1_is_square (n : ℕ) (h : 0 < n) :
  ∃ m : ℤ, n * 2 ^ (n + 1) + 1 = m * m ↔ n = 3 := 
by
  sorry

end n_times_2pow_nplus1_plus_1_is_square_l219_219877


namespace marbles_solution_l219_219561

noncomputable def marbles_problem : Prop :=
  ∃ (x : ℤ),
    let my_marbles := 16 - x,
        brother_marbles := (16 - x) / 2,
        friend_marbles := 3 * (16 - x),
        total_marbles := my_marbles + brother_marbles + friend_marbles
    in  my_marbles = 2 * brother_marbles ∧
        friend_marbles = 3 * my_marbles ∧
        total_marbles = 63 ∧
        x = 2

theorem marbles_solution : marbles_problem :=
by
  sorry

end marbles_solution_l219_219561


namespace alice_distance_from_start_l219_219435

-- Definitions for each condition from part a)
def westMeters := 30
def northMeters := 50
def eastMeters := 80
def southMeters := 20

-- Using these definitions to state the final theorem
theorem alice_distance_from_start :
  let northSouth := northMeters - southMeters,
      eastWest := eastMeters - westMeters,
      distance := Math.sqrt (northSouth^2 + eastWest^2)
  in distance = Math.sqrt 3400 := 
by
  sorry

end alice_distance_from_start_l219_219435


namespace find_unknown_number_l219_219673

theorem find_unknown_number : 
  ∃ x : ℝ, (10 + 30 + 50) / 3 = ((20 + 40 + x) / 3) + 8 ∧ x = 6 :=
by {
  existsi 6,
  split,
  {
    calc (10 + 30 + 50) / 3
      = 90 / 3 : by norm_num
      ... = 30 : by norm_num
      ... = ((20 + 40 + 6) / 3) + 8 : by norm_num
  },
  {
    refl,
  }
}

end find_unknown_number_l219_219673


namespace part_one_part_two_l219_219533

-- Defining the function and its first derivative
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Part (Ⅰ)
theorem part_one (a b : ℝ)
  (H1 : f' a b 3 = 24)
  (H2 : f' a b 1 = 0) :
  a = 1 ∧ b = -3 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f' 1 (-3) x ≤ 0) :=
sorry

-- Part (Ⅱ)
theorem part_two (b : ℝ)
  (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 3 * x^2 + b ≤ 0) :
  b ≤ -3 :=
sorry

end part_one_part_two_l219_219533


namespace exist_n_disjoint_triangles_l219_219149

noncomputable def area {X} [measurable_space X] (s : set X) [measurable_space (set X)] : X → ℝ := sorry

def disjoint (s1 s2 : set ℝ) : Prop := ∀ x, x ∈ s1 → x ∉ s2

def lies_within (s1 s2 : set ℝ) : Prop := s1 ⊆ s2

theorem exist_n_disjoint_triangles (ABCD : set ℝ) (n : ℕ) (h1 : n ≥ 2)
  (h2 : convex ABCD)
  (h3 : ∀ i j, i ≠ j → disjoint T_i T_j)
  (h4 : ∀ i, lies_within T_i ABCD)
  (h5 : ∑ i in finset.range n, area (T_i) ≥ (4 * n) / (4 * n + 1) * area (ABCD)) :
  ∃ (T_1 T_2 ... T_n : set ℝ), sorry :=
sorry

end exist_n_disjoint_triangles_l219_219149


namespace find_abs_diff_l219_219527

-- Define the angle \alpha and the points A and B with their respective coordinates
variables (α : ℝ) (a b : ℝ)

-- Conditions
def condition_1 : Prop := α / π ∈ set.Icc 0 1
def condition_2 : Prop := ∃ A B : ℝ × ℝ, A = (1, a) ∧ B = (2, b) ∧ A.2 / A.1 = B.2 / B.1
def condition_3 : Prop := cos (2 * α) = 2 / 3
def condition_4 : Prop := a = b / 2

-- Statement
theorem find_abs_diff (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4) :
  |a - b| = 3 * real.sqrt 5 / 5 :=
sorry

end find_abs_diff_l219_219527


namespace katrina_cookies_left_l219_219606

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l219_219606


namespace seq_a22_l219_219242

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l219_219242


namespace natalie_height_l219_219185

variable (height_Natalie height_Harpreet height_Jiayin : ℝ)
variable (h1 : height_Natalie = height_Harpreet)
variable (h2 : height_Jiayin = 161)
variable (h3 : (height_Natalie + height_Harpreet + height_Jiayin) / 3 = 171)

theorem natalie_height : height_Natalie = 176 :=
by 
  sorry

end natalie_height_l219_219185


namespace circle_chairs_subsets_count_l219_219741

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219741


namespace number_of_4_digit_mountain_numbers_l219_219278

def is_middle_max {d1 d2 d3 d4 : ℕ} : Prop :=
  d2 > d1 ∧ d3 > d1 ∧ d2 > d4 ∧ d3 > d4 ∧ d2 = d3

def is_middle_max_odd {d1 d2 d3 d4 : ℕ} : Prop :=
  d2 > d1 ∧ d2 > d3 ∧ d2 > d4

def is_middle_max_zero {d1 d2 d3 d4 : ℕ} : Prop :=
  d2 > d1 ∧ d3 = 0 ∧ d2 > d4

def is_mountain_number (n : ℕ) : Prop :=
  let d1 := n / 1000,
      d2 := (n / 100) % 10,
      d3 := (n / 10) % 10,
      d4 := n % 10
  in  (is_middle_max d1 d2 d3 d4 ∨ is_middle_max_odd d1 d2 d3 d4 ∨ is_middle_max_zero d1 d2 d3 d4) ∧ d1 ≠ 0 ∧ d2 ≠ 0

def count_mountain_numbers : ℕ :=
  Nat.card { n // 1000 ≤ n ∧ n < 10000 ∧ is_mountain_number n }

theorem number_of_4_digit_mountain_numbers : count_mountain_numbers = 184 :=
  sorry

end number_of_4_digit_mountain_numbers_l219_219278


namespace calculate_first_worker_time_l219_219431

theorem calculate_first_worker_time
    (T : ℝ)
    (h : 1/T + 1/4 = 1/2.2222222222222223) :
    T = 5 := sorry

end calculate_first_worker_time_l219_219431


namespace circle_areas_sum_l219_219260

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l219_219260


namespace trigonometric_identity_l219_219658

theorem trigonometric_identity (α : ℝ) (h1 : 0 < α) (h2 : α < (π / 4)) :
  (Real.tan(2 * α) / Real.tan α) = 1 + (1 / Real.cos(2 * α)) :=
sorry

end trigonometric_identity_l219_219658


namespace intersection_of_sets_l219_219077

open Set

def A : Set ℝ := { x | 3 * x + 1 > 0 }
def B : Set ℝ := { x | abs (x - 1) < 2 }

theorem intersection_of_sets : A ∩ B = { x | -1/3 < x ∧ x < 3 } := 
sorry

end intersection_of_sets_l219_219077


namespace vasya_cannot_repaint_pentagon_l219_219103

theorem vasya_cannot_repaint_pentagon :
  ∀ (P : Type) [fintype P], 
  (convex_polygon P 5) → 
  (∀ p : P, blue p) → 
  ¬(∃ n : ℕ, ∀ m ≤ n, perform_operation m P) → 
  (∃ p : P, red p) :=
begin
  sorry
end

end vasya_cannot_repaint_pentagon_l219_219103


namespace remove_highest_and_lowest_mean_variance_l219_219576

theorem remove_highest_and_lowest_mean_variance 
    (scores : List ℕ)
    (h_scores : scores = [90, 89, 90, 95, 93, 94, 93])
    : let remaining_scores := (scores.erase 95).erase 89 in 
      let mean := remaining_scores.sum / remaining_scores.length in
      let variance := 
        (remaining_scores.map (λ x => (x - mean)^2)).sum / remaining_scores.length in 
      mean = 92 ∧ variance = 2.8 :=
by
  sorry

end remove_highest_and_lowest_mean_variance_l219_219576


namespace sum_prime_factors_143_is_24_l219_219290

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219290


namespace odd_function_f_l219_219521

noncomputable def f : ℝ → ℝ
| x => if hx : x ≥ 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x * (1 - x)) :
  ∀ x : ℝ, x ≤ 0 → f x = x * (1 + x) := by
  intro x hx
  sorry

end odd_function_f_l219_219521


namespace probability_of_5_defective_books_l219_219821

open Real

noncomputable def probability_of_exactly_5_defective_books 
  (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
let λ := n * p in
(λ ^ k * exp (-λ)) / (Nat.factorial k)

theorem probability_of_5_defective_books 
  (h_n : 100000 = 100000)
  (h_p : 0.0001 = 0.0001)
  (h_k : 5 = 5) :
  probability_of_exactly_5_defective_books 100000 0.0001 5 = 0.0375 :=
by
  sorry

end probability_of_5_defective_books_l219_219821


namespace sum_of_prime_factors_143_l219_219297

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219297


namespace area_of_square_inscribed_in_circle_l219_219372

-- Define the side length of the equilateral triangle
def side_length_triangle : ℝ := 6

-- Define the radius of the circle inscribed in an equilateral triangle
def radius_circle (s : ℝ) : ℝ := s / (2 * Real.sqrt 3)

-- Calculate the radius for the given side length
def r : ℝ := radius_circle side_length_triangle

-- Diameter of the circle is twice the radius
def diameter_circle (r : ℝ) : ℝ := 2 * r

-- Define the diagonal of the square, which equals the diameter of the circle
def diagonal_square (d : ℝ) : ℝ := d

-- Define the side length of the square in terms of its diagonal
def side_length_square (d : ℝ) : ℝ := d / Real.sqrt 2

-- Calculate the side length of the square for the given diameter
def s : ℝ := side_length_square (diameter_circle r)

-- Define the area of the square
def area_square (s : ℝ) : ℝ := s ^ 2

-- Calculate the area for our side length
def area_of_inscribed_square : ℝ := area_square s

-- Define the expected area as approximately 5.9988 cm² for comparison
def expected_area : ℝ := 5.9988

-- The theorem to prove
theorem area_of_square_inscribed_in_circle :
  abs (area_of_inscribed_square - expected_area) < 0.001 := sorry

end area_of_square_inscribed_in_circle_l219_219372


namespace abc_value_l219_219413

theorem abc_value (a b c : ℝ) 
  (h0 : (a * (0 : ℝ)^2 + b * (0 : ℝ) + c) = 7) 
  (h1 : (a * (1 : ℝ)^2 + b * (1 : ℝ) + c) = 4) : 
  a + b + 2 * c = 11 :=
by sorry

end abc_value_l219_219413


namespace spider_crawl_distance_l219_219418

/-- Define the given conditions -/
def radius : ℝ := 45
def diameter : ℝ := 2 * radius
def third_leg : ℝ := 70
def second_leg : ℝ := real.sqrt (diameter^2 - third_leg^2)
def total_distance : ℝ := diameter + third_leg + second_leg

/-- State the theorem to prove the total distance -/
theorem spider_crawl_distance : total_distance ≈ 216.57 := by
  -- The proof is omitted, indicated by sorry
  sorry

end spider_crawl_distance_l219_219418


namespace pieces_per_plant_yield_l219_219406

theorem pieces_per_plant_yield 
  (rows : ℕ) (plants_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : rows = 30) (h2 : plants_per_row = 10) (h3 : total_harvest = 6000) : 
  (total_harvest / (rows * plants_per_row) = 20) :=
by
  -- Insert math proof here.
  sorry

end pieces_per_plant_yield_l219_219406


namespace eq1_solution_eq2_solution_l219_219838


-- Theorem for the first equation (4(x + 1)^2 - 25 = 0)
theorem eq1_solution (x : ℝ) : (4 * (x + 1)^2 - 25 = 0) ↔ (x = 3 / 2 ∨ x = -7 / 2) :=
by
  sorry

-- Theorem for the second equation ((x + 10)^3 = -125)
theorem eq2_solution (x : ℝ) : ((x + 10)^3 = -125) ↔ (x = -15) :=
by
  sorry

end eq1_solution_eq2_solution_l219_219838


namespace subsets_with_at_least_four_adjacent_chairs_l219_219762

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219762


namespace solution_fraction_replaced_l219_219403

-- Define the conditions
variables (x : ℝ) (initial_quantity remaining_quantity replaced_quantity final_quantity : ℝ)
variable (correct_x : ℝ := 0.5)

-- The given conditions translated
def initial_concentration : ℝ := 0.45
def replaced_concentration : ℝ := 0.25
def final_concentration : ℝ := 0.35
def total_quantity : ℝ := 1

-- The definitions from conditions
def remaining_fraction := (1 - x)
def replaced_fraction := x

-- Amount of pure substance
def pure_substance_in_initial := initial_concentration * total_quantity
def pure_substance_in_remaining := initial_concentration * remaining_fraction
def pure_substance_in_replaced := replaced_concentration * replaced_fraction

-- Total pure substance in the final solution
def pure_substance_final := final_concentration * total_quantity

-- The main statement that needs to be proved
theorem solution_fraction_replaced :
    pure_substance_in_remaining + pure_substance_in_replaced = pure_substance_final →
    x = correct_x :=
by
  sorry

end solution_fraction_replaced_l219_219403


namespace subsets_with_at_least_four_adjacent_chairs_l219_219763

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219763


namespace original_number_of_men_l219_219374

theorem original_number_of_men (x : ℕ) 
  (h1 : 17 * x = 21 * (x - 8)) : x = 42 := 
by {
   -- proof steps can be filled in here
   sorry
}

end original_number_of_men_l219_219374


namespace set_all_equal_one_l219_219620

theorem set_all_equal_one (n : ℕ) (A : finset ℕ) (k : ℕ) :
  3 ≤ n →
  A = finset.range (n + 1) \ {0} →
  (∀ i j ∈ A, (i + j ∈ A ∧ (i - j).nat_abs ∈ A)) →
  (∀ s : finset ℕ, (∀ x ∈ s, x = k) → s = A) →
  k = 1 :=
by
  intros h₁ h₂ h₃ h₄,
  sorry

end set_all_equal_one_l219_219620


namespace length_BD_l219_219715

-- Define the conditions
variable (A B C D E : Type)
variable [Point A] [Point B] [Point C] [Point D] [Point E]
variable (triangle_ABC : Triangle A B C)
variable (right_isosceles_ABC : isRightIsosceles triangle_ABC AB BC)
variable (midpoint_D_BC : Midpoint D B C)
variable (DE_15_units : Length DE = 15)
variable (perpendicular_bisector_AC : PerpendicularBisector DE AC)

-- Prove that BD = 5*sqrt(3)
theorem length_BD : Length BD = 5 * sqrt 3 :=
by sorry

end length_BD_l219_219715


namespace sum_prime_factors_143_l219_219317

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219317


namespace number_of_divisors_of_n_cubed_l219_219812

theorem number_of_divisors_of_n_cubed (n : ℕ) (h : ∃ p : ℕ, nat.prime p ∧ n = p ^ 3) : nat.num_divisors (n ^ 3) = 10 :=
by sorry

end number_of_divisors_of_n_cubed_l219_219812


namespace approx_change_in_y_l219_219276

-- Definition of the function
def y (x : ℝ) : ℝ := x^3 - 7 * x^2 + 80

-- Derivative of the function, calculated manually
def y_prime (x : ℝ) : ℝ := 3 * x^2 - 14 * x

-- The change in x
def delta_x : ℝ := 0.01

-- The given value of x
def x_initial : ℝ := 5

-- To be proved: the approximate change in y
theorem approx_change_in_y : (y_prime x_initial) * delta_x = 0.05 :=
by
  -- Imported and recognized theorem verifications skipped
  sorry

end approx_change_in_y_l219_219276


namespace monochromatic_rectangle_l219_219692

theorem monochromatic_rectangle (n : ℕ) (coloring : ℕ × ℕ → Fin n) :
  ∃ (a b c d : ℕ × ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end monochromatic_rectangle_l219_219692


namespace Cody_birthday_money_l219_219840

variable (x : ℤ) -- Define the variable "x" as an integer

theorem Cody_birthday_money :
  ∃ x, (x = 9) ∧ (45 + x - 19 = 35) :=
begin
  use 9,
  split,
  refl, -- x = 9
  calc
    45 + 9 - 19 = 35 : by norm_num,
end

end Cody_birthday_money_l219_219840


namespace complement_intersect_integers_l219_219624

open set

def M : set ℝ := {x | x ≤ 1 ∨ x > 3}
def Z : set ℤ := {x | true}

theorem complement_intersect_integers :
  ((C ℝ M) ∩ Z) = ({2, 3} : set ℤ) :=
sorry

end complement_intersect_integers_l219_219624


namespace impossible_to_cover_10x10_with_25_4x1_tiles_l219_219996

theorem impossible_to_cover_10x10_with_25_4x1_tiles : 
  ∀ (board : fin 10 × fin 10 → ℕ) (tile : fin 4 → fin 1 → ℕ),
  (∀ (t : fin 25) (x : fin 10) (y : fin 10), 
    (tile x y = 1 → board (x, y) = 1)) →
  ¬ ∃ (config : fin 25 → fin 10 × fin 10),
    ∀ (c : fin 25), ∀ (i : fin 4), ∀ (j : fin 1),
      board (config c) + tile i j = 1 :=
by
  sorry

end impossible_to_cover_10x10_with_25_4x1_tiles_l219_219996


namespace period_of_sin_x_plus_cos_2x_l219_219850

theorem period_of_sin_x_plus_cos_2x :
  ∃ T > 0, ∀ x, (sin (x + T) + cos (2 * (x + T))) = (sin x + cos (2 * x)) ↔ T = π :=
by
  sorry

end period_of_sin_x_plus_cos_2x_l219_219850


namespace ratio_of_areas_l219_219791

variables {α : Type} [real_field α]

/-- Given an acute triangle ABC with AB > AC, where BC = a, CA = b, and AB = c,
    I is the incenter, I1 is the excenter opposite vertex A, and L is the midpoint of BC.
    The line LI intersects AC at point M, and the line I1L intersects AB at point N.
    Then, the ratio of the areas of triangle MN and triangle ABC is a(c - b)/ (a + c - b)^2. -/
theorem ratio_of_areas (a b c : α)
    (h1 : AB > AC)
    (I : Type α) (I1 : Type α) (L : Type α)
    (M : α) (N : α)
    (h2 : midpoint BC L)
    (h3 : line LI intersects AC M)
    (h4 : line I1L intersects AB N)
    : (area (triangle MN))/(area (triangle ABC)) = a * (c - b) / (a + c - b)^2 :=
begin
  sorry
end

end ratio_of_areas_l219_219791


namespace cistern_emptying_time_l219_219405

theorem cistern_emptying_time :
  (∀ (R L : ℝ), 
    (R * 10 = 1) ∧ 
    ((R - L) * 12 = 1) 
    → (1 / L = 60)) :=
begin
  sorry
end

end cistern_emptying_time_l219_219405


namespace part1_part2_l219_219843

theorem part1 : ( (1 / 2 : ℝ) ^ (-1 : ℝ) - real.sqrt 3 * real.cos (real.pi / 6) + (2014 - real.pi) ^ 0) = (3 / 2 : ℝ) := sorry

theorem part2 (a : ℝ) : a * (a + 1) - (a + 1) * (a - 1) = a + 1 := sorry

end part1_part2_l219_219843


namespace circle_chairs_subsets_count_l219_219738

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219738


namespace sum_of_powers_divisible_by_m_l219_219490

open Nat

-- Definition of Euler's totient function
def euler_totient (n : ℕ) : ℕ :=
  (finset.range n).filter (λ k, gcd k n = 1).card

-- Main theorem statement
theorem sum_of_powers_divisible_by_m (n : ℕ) (k : ℕ) (m : ℕ) 
  (h1 : n ≥ 2)
  (h2 : m = euler_totient n) 
  (h3 : ∀ p : ℕ, prime p → p ∣ m → p ∣ n) :
  let S := (finset.range n).filter (λ a, gcd a n = 1) in
  (S.sum (λ a, a ^ k)) % m = 0 :=
by
  sorry

end sum_of_powers_divisible_by_m_l219_219490


namespace binary_of_19_is_10011_l219_219005

noncomputable def binary_representation_of_nineteen : string := "10011"

theorem binary_of_19_is_10011 : binary_representation_of_nineteen = Int.toString 2 19 :=
by
  sorry

end binary_of_19_is_10011_l219_219005


namespace sum_prime_factors_of_143_l219_219353

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219353


namespace Karen_baked_50_cookies_l219_219146

def Karen_kept_cookies : ℕ := 10
def Karen_grandparents_cookies : ℕ := 8
def people_in_class : ℕ := 16
def cookies_per_person : ℕ := 2

theorem Karen_baked_50_cookies :
  Karen_kept_cookies + Karen_grandparents_cookies + (people_in_class * cookies_per_person) = 50 :=
by 
  sorry

end Karen_baked_50_cookies_l219_219146


namespace sufficient_condition_parallel_l219_219503

-- Define lines a and b, and planes alpha and beta
variables {a b : Line} {α β : Plane}

-- Define the given conditions
def condition_D (a b : Line) (α : Plane) :=
  (a ∥ b) ∧ (b ∥ α) ∧ (¬(a ⊆ α))

-- State the theorem that these conditions imply a parallel to alpha
theorem sufficient_condition_parallel (a b : Line) (α : Plane) :
  condition_D a b α → (a ∥ α) :=
by
  sorry

end sufficient_condition_parallel_l219_219503


namespace find_x_l219_219983

theorem find_x (Ω : ℕ) (□ : ℕ) (△ : ℕ) (x : ℕ)
  (h1 : Ω ≠ □) (h2 : Ω ≠ △) (h3 : □ ≠ △)
  (col_sum1 : Ω + □ + Ω = 22)
  (col_sum2 : △ + △ + △ = 12)
  (col_sum3 : □ + △ + □ = 20)
  (row_sum2 : Ω + □ + △ = 19)
  (row_sum3 : □ + △ + □ = 15) :
  x = Ω + △ + □ :=
by 
  -- proof here
  sorry

end find_x_l219_219983


namespace kolya_cannot_ensure_win_l219_219147

theorem kolya_cannot_ensure_win :
  ∀ (initial_pile : ℕ), initial_pile = 31 →
  (∀ stones, stones > 1 → ∃ a b : ℕ, a + b = stones ∧ a > 0 ∧ b > 0) →
  (∀ moves : list ℕ, (∀ stones ∈ moves, stones = 1) → length moves > 0 → false) :=
by
  intros initial_pile h_pile split_condition win_condition
  sorry

end kolya_cannot_ensure_win_l219_219147


namespace sum_even_integers_102_to_200_l219_219699

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l219_219699


namespace knight_tour_impossible_5x5_l219_219603

theorem knight_tour_impossible_5x5 :
  ∀ (start : (ℕ × ℕ)), start.1 % 2 = start.2 % 2 → ¬(∃ (f : Fin 25 → ℕ × ℕ), 
  (∀ i, f i.1 = start → i = 0) ∧ 
  (∀ i j, i < j → f i ≠ f j) ∧ 
  (∀ i, i < 24 → (f (i + 1), f i) ∈ {(x, y) | abs (x.1 - y.1) = 2 ∧ abs (x.2 - y.2) = 1 ∨ abs (x.1 - y.1) = 1 ∧ abs (x.2 - y.2) = 2}) ∧ 
  (f 24 = f 0)) :=
by 
  intro start h
  sorry

end knight_tour_impossible_5x5_l219_219603


namespace number_of_possible_values_of_a_l219_219193

theorem number_of_possible_values_of_a :
  ∃ (a b c d : ℕ), a > b ∧ b > c ∧ c > d ∧ 
                    a + b + c + d = 2014 ∧ 
                    a^2 - b^2 + c^2 - d^2 = 2014 ∧ 
                    (Set.to_finset {a : ℕ | ∃ b c d : ℕ,
                      a > b ∧ b > c ∧ c > d ∧ 
                      a + b + c + d = 2014 ∧ 
                      a^2 - b^2 + c^2 - d^2 = 2014}).card = 502 := 
sorry

end number_of_possible_values_of_a_l219_219193


namespace solve_for_x_l219_219667

variable {x y : ℝ}

theorem solve_for_x (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : x = 3 / 2 := by
  sorry

end solve_for_x_l219_219667


namespace sum_prime_factors_of_143_l219_219343

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219343


namespace subsets_with_at_least_four_adjacent_chairs_l219_219760

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219760


namespace find_m_l219_219942

theorem find_m (a b m : ℝ) :
  (∀ x : ℝ, (x^2 - b * x + b^2) / (a * x^2 - b^2) = (m - 1) / (m + 1) → (∀ y : ℝ, x = y ∧ x = -y)) →
  c = b^2 →
  m = (a - 1) / (a + 1) :=
by
  sorry

end find_m_l219_219942


namespace sum_prime_factors_143_is_24_l219_219291

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219291


namespace sum_first_five_b_terms_l219_219894

noncomputable def a : ℕ → ℕ
| 1 := 3
| 2 := 6
| (n + 1) := a n + 3

noncomputable def b (n : ℕ) : ℕ := a (2 * n)

theorem sum_first_five_b_terms :
  (b 1 + b 2 + b 3 + b 4 + b 5) = 90 := 
by
  sorry

end sum_first_five_b_terms_l219_219894


namespace cats_weigh_more_by_5_kg_l219_219930

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l219_219930


namespace arithmetic_sequence_a6_eq_1_l219_219119

theorem arithmetic_sequence_a6_eq_1
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : S 11 = 11)
  (h2 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h3 : ∃ d, ∀ n, a n = a 1 + (n - 1) * d) :
  a 6 = 1 :=
by
  sorry

end arithmetic_sequence_a6_eq_1_l219_219119


namespace katrina_cookies_left_l219_219610

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l219_219610


namespace sin_160_eq_sin_20_l219_219394

theorem sin_160_eq_sin_20 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180) :=
by
  sorry

end sin_160_eq_sin_20_l219_219394


namespace intersection_of_sets_l219_219542

def setA : Set ℝ := {x | (x^2 - x - 2 < 0)}
def setB : Set ℝ := {y | ∃ x ≤ 0, y = 3^x}

theorem intersection_of_sets : (setA ∩ setB) = {z | 0 < z ∧ z ≤ 1} :=
sorry

end intersection_of_sets_l219_219542


namespace line_intersects_even_number_of_diagonals_l219_219570

-- Define the convex polygon with 2009 vertices
structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (convex : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → 
              let A := vertices i, B := vertices j, C := vertices k in 
              ( (A.1 - B.1) * (C.2 - B.2) - (A.2 - B.2) * (C.1 - B.1) ) ≥ 0)

-- Definition of an intersection
def intersects (l : ℝ × ℝ → Prop) (P : ConvexPolygon 2009) : Finset (Fin 2009 × Fin 2009) :=
  { e | ∃ t, l (P.vertices e.1) ∧ l (P.vertices e.2) }

-- Main theorem
theorem line_intersects_even_number_of_diagonals (P : ConvexPolygon 2009) 
  (l : ℝ × ℝ → Prop) (inter_limits : ∀ v, ∃ t, ¬l (P.vertices v)) : 
  (intersects l P).card % 2 = 0 := 
sorry

end line_intersects_even_number_of_diagonals_l219_219570


namespace converted_land_eqn_l219_219429

theorem converted_land_eqn (forest_land dry_land converted_dry_land : ℝ)
  (h1 : forest_land = 108)
  (h2 : dry_land = 54)
  (h3 : converted_dry_land = x) :
  (dry_land - converted_dry_land = 0.2 * (forest_land + converted_dry_land)) :=
by
  simp [h1, h2, h3]
  sorry

end converted_land_eqn_l219_219429


namespace intersection_complement_l219_219079

open Set

variable (U : Finset ℕ) (M : Finset ℕ) (N : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2, 3})
variable (hN : N = {3, 4, 5})

theorem intersection_complement :
  M ∩ (U \ N) = {1, 2} :=
by
  rw [hU, hM, hN]
  simp
  sorry

end intersection_complement_l219_219079


namespace sum_prime_factors_of_143_l219_219348

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219348


namespace factor_expression_l219_219475

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end factor_expression_l219_219475


namespace rhombus_diagonal_distance_l219_219506

open Real

theorem rhombus_diagonal_distance (a θ: ℝ) (h1: 0 < a) (h2: π / 3 ≤ θ ∧ θ ≤ 2 * π / 3) :
  let Δ₁ := 2 * a * cos (π / 6), Δ₁_half := Δ₁ / 2,
      halfθ := θ / 2, max_dist := Δ₁_half * cos (π / 6)
  in
      max_dist = (3 / 4 * a) :=
by
  sorry

end rhombus_diagonal_distance_l219_219506


namespace exists_element_star_eq_self_l219_219661

variables {T : Type*} [Fintype T] (star : T → T → T)
variables [IsAssociative T star] [IsCommutative T star]

-- The theorem to state the problem
theorem exists_element_star_eq_self: ∃ a ∈ (univ : finset T), ∀ b ∈ (univ : finset T), star a b = a :=
sorry

end exists_element_star_eq_self_l219_219661


namespace find_n_divisibility_l219_219000

theorem find_n_divisibility :
  ∃ n : ℕ, n < 10 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 11 = 0 ∧ (6 * 10000 + n * 1000 + 2 * 100 + 7 * 10 + 2) % 5 = 0 :=
by
  use 3
  sorry

end find_n_divisibility_l219_219000


namespace sum_prime_factors_143_is_24_l219_219284

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219284


namespace find_angle_B_max_perimeter_l219_219961

variables {A B C a b c : ℝ}
variables {pi : ℝ} [fact (real.pi = pi)]

-- Given condition for part (1)
theorem find_angle_B (h : 3 * real.cos B = 2 * real.sin (pi / 3 + A) * real.sin (pi / 3 - A) + 2 * (real.sin A) ^ 2) :
  B = pi / 3 :=
sorry

-- Given conditions and aim for part (2)
theorem max_perimeter (b_val : b = 2 * real.sqrt 3)
  (h1 : a / real.sin A = 4) (h2 : c / real.sin C = 4) :
  ∃ (P : ℝ), P = 6 * real.sqrt 3 :=
sorry

end find_angle_B_max_perimeter_l219_219961


namespace sticker_cost_l219_219137

theorem sticker_cost :
  let P := 4 in                   -- number of packs
  let S := 30 in                  -- stickers per pack
  let J := 6 in                   -- James's payment
  let total_cost := J * 2 in      -- total payment for stickers
  let total_stickers := P * S in  -- total number of stickers
  (total_cost / total_stickers) = 0.10 :=  -- cost per sticker
by
  sorry

end sticker_cost_l219_219137


namespace sum_even_integers_102_to_200_l219_219698

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l219_219698


namespace factor_of_lcm_l219_219685

theorem factor_of_lcm (A B hcf : ℕ) (h_gcd : Nat.gcd A B = hcf) (hcf_eq : hcf = 16) (A_eq : A = 224) :
  ∃ X : ℕ, X = 14 := by
  sorry

end factor_of_lcm_l219_219685


namespace compare_stability_with_variance_l219_219109

theorem compare_stability_with_variance (scores_A scores_B : Fin 5 → ℝ) 
(h_avg : (∑ i, scores_A i) / 5 = (∑ i, scores_B i) / 5) :
  (∑ i, (scores_A i - (∑ i, scores_A i) / 5)^2) / 5 = (∑ i, (scores_B i - (∑ i, scores_B i) / 5)^2) / 5 → 
  ∑ i, (scores_A i - ∑ i, scores_A i / 5)^2 = ∑ i, (scores_B i - ∑ i, scores_B i / 5)^2 :=
sorry

end compare_stability_with_variance_l219_219109


namespace cot_sum_eq_zero_l219_219219

theorem cot_sum_eq_zero
(f : ℝ → ℝ)
(n : ℕ)
(hn : 1 < n)
(b c : ℝ)
(hbc : b ≠ c)
(B C : Fin n → ℝ)
(h_f_b : ∀ i, f (B i) = b)
(h_f_c : ∀ i, f (C i) = c)
(P : ℝ)
(hP : ∀ i, C (Fin.last n) < P) :
(∑ i : Fin n, Real.cot (Real.angle (B i) (C i) P)) = 0 := sorry

end cot_sum_eq_zero_l219_219219


namespace remainder_when_divided_by_x_plus_2_l219_219365

variable (D E F : ℝ)

def q (x : ℝ) := D * x^4 + E * x^2 + F * x + 7

theorem remainder_when_divided_by_x_plus_2 :
  q D E F (-2) = 21 - 2 * F :=
by
  have hq2 : q D E F 2 = 21 := sorry
  sorry

end remainder_when_divided_by_x_plus_2_l219_219365


namespace seats_in_hall_l219_219973

theorem seats_in_hall (S : ℕ) (h1 : 0.45 * S + 330 = S) : S = 600 :=
by
  -- problem conditions and variable definitions
  have h2 : 0.55 * S = 330 :=
    by linarith [h1]
  -- solve for S
  have h3 : S = 330 / 0.55 :=
    by field_simp [h2, mul_comm]
  -- cast division result to appropriate type
  have h4 : (S : ℝ) = 330 / 0.55 :=
    by assumption
  -- final proof using cast
  exact eq_of_nat_cast_eq_nat_cast h4

end seats_in_hall_l219_219973


namespace subsets_with_at_least_four_adjacent_chairs_l219_219764

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219764


namespace prime_divides_sigma_prime_minus_one_l219_219084

open Nat

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def sigma (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d ∣ n).sum

theorem prime_divides_sigma_prime_minus_one (p : ℕ) (hp : is_prime p) :
  p ∣ sigma (p - 1) → p = 3 :=
begin
  sorry
end

end prime_divides_sigma_prime_minus_one_l219_219084


namespace compare_stability_l219_219107

-- Specify the context of scores for students A and B.
variables (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℝ)

-- Define the average scores for both sets
def avg_A : ℝ := (a1 + a2 + a3 + a4 + a5) / 5
def avg_B : ℝ := (b1 + b2 + b3 + b4 + b5) / 5

-- Specify the condition that the average scores for both sets are equal.
axiom avg_equal : avg_A = avg_B

-- Define variance for both sets.
def var_A : ℝ := ((a1 - avg_A)^2 + (a2 - avg_A)^2 + (a3 - avg_A)^2 + (a4 - avg_A)^2 + (a5 - avg_A)^2) / 5
def var_B : ℝ := ((b1 - avg_B)^2 + (b2 - avg_B)^2 + (b3 - avg_B)^2 + (b4 - avg_B)^2 + (b5 - avg_B)^2) / 5

-- The proof problem: To determine which performance is more stable, compare the variances var_A and var_B.
theorem compare_stability : var_A = var_B → avg_equal → sorry

end compare_stability_l219_219107


namespace anna_reading_time_l219_219443

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l219_219443


namespace sum_prime_factors_143_l219_219319

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219319


namespace sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l219_219083

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (4, -2)

def perp_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sufficient_not_necessary_condition :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) ↔ (m = 1 ∨ m = -3) :=
by
  sorry

theorem m_eq_1_sufficient :
  (m = 1) → perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) :=
by
  sorry

theorem m_eq_1_not_necessary :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) → (m = 1 ∨ m = -3) :=
by
  sorry

end sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l219_219083


namespace correct_inequality_l219_219095

-- Define the function and its properties
def f (x : ℝ) (b c : ℝ) : ℝ := -x^2 + b * x + c

-- Define the property related to the axis of symmetry
def axis_of_symmetry (b : ℝ) : Prop := ∀ (x : ℝ), f (2) b c = f (2 - x) b c

-- Statement to prove
theorem correct_inequality (b c : ℝ) (h : axis_of_symmetry b) : 
  f 4 b c < f 1 b c ∧ f 1 b c < f 2 b c :=
sorry

end correct_inequality_l219_219095


namespace find_a22_l219_219246

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l219_219246


namespace largest_cubed_side_length_l219_219249

/--
Theorem: Given a cone with slant height \( l \) and angle \( \theta \) between slant height and base,
the side length \( a \) of the largest cube that can be inscribed in the cone,
where 4 vertices of the cube are on the base and the other 4 vertices on the lateral surface,
is given by:
   a = (√2 * l * sin θ) / (tan θ + √2)
-/
theorem largest_cubed_side_length (l θ : ℝ) (hθ_ne_zero : θ ≠ 0) (hθ_ne_pi_over_2 : θ ≠ Real.pi / 2) :
  ∃ a : ℝ, a = (Real.sqrt 2 * l * Real.sin θ) / (Real.tan θ + Real.sqrt 2) :=
by
  use (Real.sqrt 2 * l * Real.sin θ) / (Real.tan θ + Real.sqrt 2)
  sorry

end largest_cubed_side_length_l219_219249


namespace cos_sum_to_product_form_l219_219679

theorem cos_sum_to_product_form (x : ℝ) :
  let a := 4
  let b := 6
  let c := 3
  let d := 1
  ∃ a b c d : ℕ, a = 4 ∧ b = 6 ∧ c = 3 ∧ d = 1 ∧
  (cos (2 * x) + cos (4 * x) + cos (8 * x) + cos (10 * x)) = a * (cos (b * x)) * (cos (c * x)) * (cos (d * x))
:=
begin
  sorry
end

end cos_sum_to_product_form_l219_219679


namespace inverse_sum_l219_219683

noncomputable def f : ℝ → ℝ
| x => if x < 5 then x - 3 else real.sqrt (x + 1)

noncomputable def f_inv : ℝ → ℝ
| y => if y < 2 then y + 3 else y^2 - 1

theorem inverse_sum :
  f_inv (-6) + f_inv (-5) + f_inv (-4) + f_inv (-3) + f_inv (-2) + f_inv (-1) + f_inv 0 + f_inv 1 +
  f_inv 2 + f_inv 3 + f_inv 4 + f_inv 5 = 54 :=
by
  sorry

end inverse_sum_l219_219683


namespace geometric_sum_1500_terms_l219_219251

theorem geometric_sum_1500_terms (a r : ℝ) 
  (h₁ : (∑ i in finset.range 500, a * r^i) = 400) 
  (h₂ : (∑ i in finset.range 1000, a * r^i) = 720) : 
  (∑ i in finset.range 1500, a * r^i) = 976 :=
sorry

end geometric_sum_1500_terms_l219_219251


namespace daughters_dress_probability_l219_219786

theorem daughters_dress_probability : (2 / 6 : ℚ) = 1 / 3 := 
        by
        calc
        (2 / 6 : ℚ) = 1 / 3 : by squeeze_simp
        -- here, squeeze_simp or norm_num would help eliminate the fraction.
        
#check daughters_dress_probability -- This should check type correctness

end daughters_dress_probability_l219_219786


namespace quadratic_graph_passes_through_l219_219221

theorem quadratic_graph_passes_through (a b c n : ℝ) 
  (h1 : ∀ x : ℝ, x = 2 → y = ax^2 + bx + c = 4)
  (h2 : y = ax^2 + bx + c ∧ y = -16 ∧ x = 0)
  (h3 : y = ax^2 + bx + c ∧ x = 5) :
  n = -41 := 
sorry

end quadratic_graph_passes_through_l219_219221


namespace subsets_with_at_least_four_adjacent_chairs_l219_219729

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219729


namespace sum_even_integers_102_to_200_l219_219703

theorem sum_even_integers_102_to_200 : 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 1 102), 2 * k) = 2550 →
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) + 1250 = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100) + ∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 1300 → 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 200), 2 * i) = 1250 :=
begin
  sorry
end

end sum_even_integers_102_to_200_l219_219703


namespace even_goals_more_likely_l219_219809

theorem even_goals_more_likely (p₁ : ℝ) (q₁ : ℝ) 
  (h₁ : q₁ = 1 - p₁)
  (independent_halves : (p₁ * p₁ + q₁ * q₁) > (2 * p₁ * q₁)) :
  (p₁ * p₁ + q₁ * q₁) > (1 - (p₁ * p₁ + q₁ * q₁)) :=
by
  sorry

end even_goals_more_likely_l219_219809


namespace construct_triangle_l219_219464

-- Define the given heights and median
variable (h_a m_a h_b : ℝ)

-- Problem statement: Constructing a triangle with the given measurements results in two possible triangles
theorem construct_triangle (h_a m_a h_b : ℝ) : 
  ∃ ABC : Type,
    (∃ (A B C : ABC) (M : ABC),
      height A B C = h_a ∧
      median A B C = m_a ∧
      height B A C = h_b ∧
        -- Indicate the construction results in two possible triangles
      (two_possible_triangles A B C M)) := sorry

end construct_triangle_l219_219464


namespace angle_diff_60_degrees_l219_219508

open EuclideanGeometry

/-- Given an acute ABC triangle with circumcenter O.
Line AO intersects BC at point D. E and F are circumcenters of ABD and ACD respectively.
If AB > AC and EF = BC, then angle C - angle B equals 60 degrees. -/
theorem angle_diff_60_degrees 
  (A B C O D E F : Point)
  (hAcute : is_acute_triangle A B C)
  (hCircumcenter : circumcenter A B C O)
  (hIntersection : line_intersects_line (line_through A O) (line_through B C) D)
  (hCircumcenter_ABD : circumcenter A B D E)
  (hCircumcenter_ACD : circumcenter A C D F)
  (hAB_gt_AC : distance A B > distance A C)
  (hEF_eq_BC : distance E F = distance B C) :
  ∠C - ∠B = 60 :=
sorry

end angle_diff_60_degrees_l219_219508


namespace find_a_l219_219562

theorem find_a 
  (a b c : ℚ) 
  (h1 : b = 4 * a) 
  (h2 : b = 15 - 4 * a - c) 
  (h3 : c = a + 2) : 
  a = 13 / 9 := 
by 
  sorry

end find_a_l219_219562


namespace cos_angle_PXS_l219_219677

theorem cos_angle_PXS (P Q R S X : Type) [Rectangle P Q R S] (hPS : PS = 10) (hRS : RS = 24)
  (hInt : diagonals_intersect P Q R S X) : 
  cos (angle P X S) = 119 / 169 := 
sorry

end cos_angle_PXS_l219_219677


namespace locus_of_tangent_circle_centers_l219_219226

theorem locus_of_tangent_circle_centers :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let C2 := {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 9}
  ∃ L, (L = ⋃ (p : ℝ × ℝ) (hp : tangent_to_both p C1 C2), {p}),
  is_hyperbola_and_line L :=
by sorry

end locus_of_tangent_circle_centers_l219_219226


namespace mary_saw_total_snakes_l219_219177

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l219_219177


namespace sum_of_prime_factors_of_143_l219_219301

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219301


namespace coordinates_of_AC_l219_219035

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p1.x - p2.x,
    y := p1.y - p2.y,
    z := p1.z - p2.z }

def scalar_mult (k : ℝ) (v : Point3D) : Point3D :=
  { x := k * v.x,
    y := k * v.y,
    z := k * v.z }

noncomputable def A : Point3D := { x := 1, y := 2, z := 3 }
noncomputable def B : Point3D := { x := 4, y := 5, z := 9 }

theorem coordinates_of_AC : vector_sub B A = { x := 3, y := 3, z := 6 } →
  scalar_mult (1 / 3) (vector_sub B A) = { x := 1, y := 1, z := 2 } :=
by
  sorry

end coordinates_of_AC_l219_219035


namespace cost_difference_l219_219182

def TMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 50
  let additional_line_cost := 16
  let discount := 0.1
  let data_charge := 3
  let monthly_cost_before_discount := base_cost + (additional_line_cost * (num_lines - 2))
  let total_monthly_cost := monthly_cost_before_discount + (data_charge * num_lines)
  (total_monthly_cost * (1 - discount)) * 12

def MMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 45
  let additional_line_cost := 14
  let activation_fee := 20
  let monthly_cost := base_cost + (additional_line_cost * (num_lines - 2))
  (monthly_cost * 12) + (activation_fee * num_lines)

theorem cost_difference (num_lines : ℕ) (h : num_lines = 5) :
  TMobile_cost num_lines - MMobile_cost num_lines = 76.40 :=
  sorry

end cost_difference_l219_219182


namespace paths_mat8_l219_219122

-- Define variables
def grid := [
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"],
  ["A", "T", "A", "T", "A"],
  ["M", "A", "M", "A", "M"]
]

def is_adjacent (x1 y1 x2 y2 : Nat): Bool :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 = y2 - 1)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 = x2 - 1))

def count_paths (grid: List (List String)): Nat :=
  -- implementation to count number of paths
  4 * 4 * 2

theorem paths_mat8 (grid: List (List String)): count_paths grid = 32 := by
  sorry

end paths_mat8_l219_219122


namespace conditions_for_unique_solution_l219_219875

noncomputable def is_solution (n p x y z : ℕ) : Prop :=
x + p * y = n ∧ x + y = p^z

def unique_positive_integer_solution (n p : ℕ) : Prop :=
∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ is_solution n p x y z

theorem conditions_for_unique_solution {n p : ℕ} :
  (1 < p) ∧ ((n - 1) % (p - 1) = 0) ∧ ∀ k : ℕ, n ≠ p^k ↔ unique_positive_integer_solution n p :=
sorry

end conditions_for_unique_solution_l219_219875


namespace bound_g_l219_219867

noncomputable def f : ℝ → ℝ := sorry

def g (x y : ℝ) : ℝ :=
  (f (x + y) - f x) / (f x - f (x - y))

theorem bound_g (h_inc : ∀ x y, x ≤ y → f x ≤ f y)
  (h_cond : ∀ y, (0 < y → 1/2 < g 0 y ∧ g 0 y < 2) 
                 ∧ ∀ x y, (x ≠ 0 → 0 < y ∧ y ≤ |x| → 1/2 < g x y ∧ g x y < 2)) :
  ∀ x y, (0 < y → 1/14 < g x y ∧ g x y < 14) :=
        sorry

end bound_g_l219_219867


namespace sum_prime_factors_143_l219_219313

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219313


namespace cats_weight_more_than_puppies_l219_219929

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l219_219929


namespace train_speed_is_60_kmph_l219_219426

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end train_speed_is_60_kmph_l219_219426


namespace katrina_cookies_left_l219_219612

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l219_219612


namespace triplet_count_l219_219422

-- Defining the conditions
def isValidTriplet (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 2015 ∧ a + b + c = 2018

-- Proving the number of valid triplets (a, b, c) is 338352
theorem triplet_count : 
  let validTriplets := { (a, b, c) | 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 2015 ∧ a + b + c = 2018 }
  Finset.card validTriplets = 338352 := 
sorry

end triplet_count_l219_219422


namespace sum_of_prime_factors_of_143_l219_219304

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219304


namespace expansion_properties_l219_219494

noncomputable def binom (n k : ℕ) : ℕ :=
  if k ≤ n then nat.choose n k else 0

theorem expansion_properties :
  let n := 7 in
  let x : ℝ := 1 in
  let sum_coeffs := (∑ r in finset.range (n + 1), (2 ^ r * binom n r)) in
  let sum_binom := (∑ r in finset.range (n + 1), binom n r) in
  sum_coeffs = 2187 ∧
  sum_binom = 128 ∧
  {r : ℕ | r ∈ finset.range (n + 1) ∧ (r % 2 = 0)} = {0, 2, 4, 6} ∧
  all (λ r, (2 ^ r * binom n r * x ^ (r / 2)) ∈ {1, 84, 560, 448})
    {r : ℕ | r ∈ finset.range (n + 1) ∧ (r % 2 = 0)} :=
by
  sorry

end expansion_properties_l219_219494


namespace songs_downloaded_later_l219_219614

-- Definition that each song has a size of 5 MB
def song_size : ℕ := 5

-- Definition that the new songs will occupy 140 MB of memory space
def total_new_song_memory : ℕ := 140

-- Prove that the number of songs Kira downloaded later on that day is 28
theorem songs_downloaded_later (x : ℕ) (h : song_size * x = total_new_song_memory) : x = 28 :=
by
  sorry

end songs_downloaded_later_l219_219614


namespace binomial_coefficients_sum_l219_219593

noncomputable def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem binomial_coefficients_sum : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := 
by
  sorry

end binomial_coefficients_sum_l219_219593


namespace find_m_l219_219072

theorem find_m (m : ℝ) : (∀ x y : ℝ, (x = 1 ∧ y = -2) → (4 * x - m * y + 12 = 0)) → m = -8 :=
by
  intro h
  specialize h 1 -2
  simp at h
  sorry

end find_m_l219_219072


namespace number_of_cooks_l219_219451

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end number_of_cooks_l219_219451


namespace fish_given_l219_219183

theorem fish_given (initial_fish new_fish given_fish : ℕ) 
  (h₁ : initial_fish = 22) 
  (h₂ : new_fish = 69) 
  (h : new_fish = initial_fish + given_fish) : 
  given_fish = 47 :=
by
  rw [h₁, h₂] at h
  exact Nat.add_right_cancel h

end fish_given_l219_219183


namespace arc_length_parametric_curve_l219_219454

noncomputable def curve_x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def curve_y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

theorem arc_length_parametric_curve : 
  ∫ (t : ℝ) in (π/2)..π, Real.sqrt ((Deriv curve_x t)^2 + (Deriv curve_y t)^2) = 2 * (Real.exp π - Real.exp (π/2)) :=
by
  sorry

end arc_length_parametric_curve_l219_219454


namespace find_phi_l219_219626

theorem find_phi :
  ∃ s : ℝ, s > 0 ∧ ∀ z : ℂ, (z^8 - z^7 + z^5 - z^4 + z^3 - z + 1 = 0) → (0 < z.im) ∧ (0 <= z.re ∧ z.re < 360) →
    (z = s * (complex.exp (308.57 * real.pi / 180))) :=
sorry

end find_phi_l219_219626


namespace triangle_ABC_angle_45_degrees_l219_219994

variables (A B C M N : Type) 
variable [MetricSpace A]
variable [MetricSpace B]
variable [MetricSpace C]
variable [MetricSpace M]
variable [MetricSpace N]

-- Definitions based on the problem
def is_midpoint (X Y Z : Type) [MetricSpace X] [MetricSpace Y] [MetricSpace Z] := sorry
def are_parallel (L1 L2 : Type) [MetricSpace L1] [MetricSpace L2] := sorry
def is_orthocenter (O : Type) [MetricSpace O] := sorry
def find_angle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] := sorry

-- Specific conditions from the problem
axiom midpoint_M : is_midpoint A C M
axiom midpoint_N : is_midpoint B C N
axiom intersection_medians_orthocenter : is_orthocenter (find_angle A M N)

-- The statement we need to prove
theorem triangle_ABC_angle_45_degrees : find_angle A B C = 45 := 
sorry

end triangle_ABC_angle_45_degrees_l219_219994


namespace sum_prime_factors_143_l219_219359

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219359


namespace value_makes_expression_undefined_l219_219491

theorem value_makes_expression_undefined (a : ℝ) : 
    (a^2 - 9 * a + 20 = 0) ↔ (a = 4 ∨ a = 5) :=
by
  sorry

end value_makes_expression_undefined_l219_219491


namespace minimum_value_of_f_l219_219499

-- Define the function f(x) as the integral of (2t - 4) from 0 to x
noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, (2 * t - 4)

-- Define the interval [1, 3]
def interval : set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }

-- Theorem stating the minimum value of f(x) in the interval [1, 3] is -4
theorem minimum_value_of_f : ∀ x ∈ interval, f x ≥ -4 ∧ (∃ y ∈ interval, f y = -4) :=
sorry

end minimum_value_of_f_l219_219499


namespace each_child_plays_30_minutes_l219_219669

theorem each_child_plays_30_minutes 
  (num_children : ℕ) 
  (pairs_playing : ℕ) 
  (total_game_time : ℕ) 
  (equal_playtime : bool) 
  (h1 : num_children = 6) 
  (h2 : pairs_playing = 2) 
  (h3 : total_game_time = 90) 
  (h4 : equal_playtime = tt) : 
  ∃ t : ℕ, t = 30 := 
by 
  sorry

end each_child_plays_30_minutes_l219_219669


namespace magnitude_a_add_b_is_sqrt_10_l219_219042

-- Define the variables and vectors
variables (x y : ℝ)
def a : EuclideanSpace ℝ (Fin 2) := ![x, 1]
def b : EuclideanSpace ℝ (Fin 2) := ![1, y]
def c : EuclideanSpace ℝ (Fin 2) := ![2, -4]

-- Conditions
def a_perp_c : Prop := a ⬝ c = 0
def b_parallel_c : Prop := ∃ k : ℝ, b = k • c

-- Quantity to prove
def magnitude_a_add_b : ℝ := ∥a + b∥

theorem magnitude_a_add_b_is_sqrt_10 
  (h1 : a_perp_c x)
  (h2 : b_parallel_c y) :
  magnitude_a_add_b x y = sqrt 10 :=
by sorry

end magnitude_a_add_b_is_sqrt_10_l219_219042


namespace num_chemistry_books_l219_219709

noncomputable theory

def num_biology_books : ℕ := 15
def total_ways : ℕ := 2940

def comb (n : ℕ) (k : ℕ) : ℕ := 
  n.choose k

def ways_biology : ℕ :=
  comb num_biology_books 2

theorem num_chemistry_books (C : ℕ) : 
  (C.choose 2 * ways_biology = total_ways) → C = 8 :=
by
  intro h
  sorry

end num_chemistry_books_l219_219709


namespace impossible_arrangement_l219_219131

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end impossible_arrangement_l219_219131


namespace decreasing_hyperbola_l219_219028

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end decreasing_hyperbola_l219_219028


namespace cell_shapes_in_square_l219_219398

theorem cell_shapes_in_square :
  ∃ x y : ℕ, 4 * x + 5 * y = 36 ∧ x = 4 ∧ y = 4 :=
by
  existsi 4
  existsi 4
  split
  calc
    4 * 4 + 5 * 4 = 16 + 20 := by rfl
    _ = 36 := by rfl
  split <;> rfl
  sorry

end cell_shapes_in_square_l219_219398


namespace number_of_crosswalks_per_intersection_l219_219781

theorem number_of_crosswalks_per_intersection 
  (num_intersections : Nat) 
  (total_lines : Nat) 
  (lines_per_crosswalk : Nat) 
  (h1 : num_intersections = 5) 
  (h2 : total_lines = 400) 
  (h3 : lines_per_crosswalk = 20) :
  (total_lines / lines_per_crosswalk) / num_intersections = 4 :=
by
  -- Proof steps can be inserted here
  sorry

end number_of_crosswalks_per_intersection_l219_219781


namespace shortest_distance_ln_x_y_eq_x_l219_219918

theorem shortest_distance_ln_x_y_eq_x :
  let g := fun (x : ℝ) => Real.log x in
  let line := fun (x y : ℝ) => y = x in
  ∃ P : ℝ × ℝ, P ∈ (λ x : ℝ, (x, g x)) ∧ 
                ∃ d : ℝ, d = Real.sqrt 2 / 2 ∧ 
                ∀ Q ∈ (λ x : ℝ, (x, g x)), Real.dist Q (⟨Q.1, Q.1⟩ : ℝ × ℝ) ≥ d :=
begin
  sorry
end

end shortest_distance_ln_x_y_eq_x_l219_219918


namespace at_least_one_vertex_in_or_on_M_l219_219824

-- Definitions of T, M, T', and P
variables (T T' : Triangle) (M : Polygon) (P : Point)

-- Conditions
axiom convex (polygon : Polygon) : Prop
axiom point_symmetric (polygon : Polygon) : Prop

-- T is a triangle contained inside M
axiom triangle_T : Triangle
axiom triangle_contained_in_polygon : Triangle → Polygon → Prop
axiom triangle_T_condition : triangle_contained_in_polygon T M

-- M is a point-symmetrical polygon
axiom polygon_M : Polygon
axiom point_symmetric_condition : point_symmetric M

-- P is a point inside the triangle T
axiom point_P : Point
axiom point_inside_triangle : Point → Triangle → Prop
axiom point_P_condition : point_inside_triangle P T

-- T' is the reflection of T at point P
axiom reflection : Triangle → Point → Triangle
axiom triangle_T'_condition : T' = reflection T P

-- Statement to be proven: at least one of the vertices of T' lies inside or on the boundary of M
theorem at_least_one_vertex_in_or_on_M :
  ∃ (v : Point), (v ∈ T'.vertices) ∧ (v ∈ M.boundary ∨ v ∈ M.interior) :=
sorry

end at_least_one_vertex_in_or_on_M_l219_219824


namespace find_theta_l219_219538

open Real

theorem find_theta (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2)
  (h1 : dist (cos θ, sin θ) ({p | p.1 * sin θ + p.2 * cos θ - 1 = 0}) = 1 / 2) :
  θ = π / 12 ∨ θ = 5 * π / 12 := sorry

end find_theta_l219_219538


namespace mary_saw_total_snakes_l219_219176

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l219_219176


namespace find_b_in_triangle_ABC_l219_219564

noncomputable def cos_rule_b (a c B : ℝ) (h1 : a = 2) (h2 : c = 2 * real.sqrt 3) (h3 : B = real.pi / 6) : ℝ :=
real.sqrt (a^2 + c^2 - 2 * a * c * real.cos B)

theorem find_b_in_triangle_ABC (a c B b : ℝ)
  (h1 : a = 2) (h2 : c = 2 * real.sqrt 3) (h3 : B = real.pi / 6) 
  (h4 : b = cos_rule_b a c B h1 h2 h3) : 
  b = 2 :=
sorry

end find_b_in_triangle_ABC_l219_219564


namespace sum_prime_factors_of_143_l219_219345

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219345


namespace find_a22_l219_219245

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l219_219245


namespace difference_is_1343_l219_219216

-- Define the larger number L and the relationship with the smaller number S.
def L : ℕ := 1608
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Define the relationship: L = 6S + 15
def relationship (S : ℕ) : Prop := L = quotient * S + remainder

-- The theorem we want to prove: The difference between the larger and smaller number is 1343
theorem difference_is_1343 (S : ℕ) (h_rel : relationship S) : L - S = 1343 :=
by
  sorry

end difference_is_1343_l219_219216


namespace max_area_triangle_OAB_l219_219654

theorem max_area_triangle_OAB :
  let secondHandSpeed := 6 -- degrees per second
  let minuteHandSpeed := 0.1 -- degrees per second
  let t := 90 / (secondHandSpeed - minuteHandSpeed)
  t = 15 / 59 := by
  let secondHandSpeed := 6
  let minuteHandSpeed := 0.1
  have h : (6 - 0.1) * (15 / 59) = 90 := by
    sorry
  exact h

end max_area_triangle_OAB_l219_219654


namespace gwen_stocks_worth_1350_l219_219926

def initial_bonus : ℝ := 900
def invested_in_stock_A : ℝ := initial_bonus / 3
def invested_in_stock_B : ℝ := initial_bonus / 3
def invested_in_stock_C : ℝ := initial_bonus / 3

def value_stock_A_end_year : ℝ := invested_in_stock_A * 2
def value_stock_B_end_year : ℝ := invested_in_stock_B * 2
def value_stock_C_end_year : ℝ := invested_in_stock_C / 2

def total_value_end_year : ℝ := value_stock_A_end_year + value_stock_B_end_year + value_stock_C_end_year

theorem gwen_stocks_worth_1350 :
  total_value_end_year = 1350 := by
  -- Proof will go here
  sorry

end gwen_stocks_worth_1350_l219_219926


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219745

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219745


namespace circle_areas_sum_l219_219261

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l219_219261


namespace locus_of_centers_tangent_circles_l219_219228

-- Definition of circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 9

-- Distance between centers
def distance_centers : ℝ := 4

theorem locus_of_centers_tangent_circles : 
    (∃ h : (x y : ℝ), (circle1 x y) ∧ (circle2 x y)) → 
    ((∃ x y : ℝ, (4 * (x-4)^2 - y^2 = 0)) ∨ 
     (∃ x : ℝ, (x = 2))) :=
sorry

end locus_of_centers_tangent_circles_l219_219228


namespace sally_pokemon_cards_l219_219666

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end sally_pokemon_cards_l219_219666


namespace product_roots_of_unity_l219_219002

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.i / 13)

theorem product_roots_of_unity :
  (∏ k in Finset.range 12, (3 - omega^k)) = 885735 :=
  sorry

end product_roots_of_unity_l219_219002


namespace find_n_and_rational_terms_l219_219169

theorem find_n_and_rational_terms (x : ℝ) (n : ℕ) (M : ℝ) (N : ℕ) :
  M = (3*x + sqrt(x))^n ∧ M - N = 240 → n = 4 ∧ 
  (∃ T : list (ℝ × ℕ), T = [((81 : ℝ), 4), ((54 : ℝ), 3), ((1 : ℝ), 2)]) :=
by 
  intros h
  sorry

end find_n_and_rational_terms_l219_219169


namespace sum_prime_factors_143_l219_219363

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219363


namespace comparison_theorem_l219_219034

open Real

noncomputable def comparison (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) : Prop :=
  let a := log (sin x)
  let b := sin x
  let c := exp (sin x)
  a < b ∧ b < c

theorem comparison_theorem (x : ℝ) (h : 0 < x ∧ x < π / 2) : comparison x h.1 h.2 :=
by { sorry }

end comparison_theorem_l219_219034


namespace find_x_l219_219795

def op (a b : ℤ) : ℤ := -2 * a + b

theorem find_x (x : ℤ) (h : op x (-5) = 3) : x = -4 :=
by
  sorry

end find_x_l219_219795


namespace ellipse_properties_l219_219896

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties
  {a b : ℝ} (h1 : a > b) (h2 : b > 0) 
  (h3 : (√2) / 2 = (real.sqrtMode (-a^2 + c^2)) / a)
  (h4 : ∃ (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ), angle F1 M F2 = 90 
        ∧ area_of_triangle F1 M F2 = 1) :
  (∃ x y : ℝ, ellipse_equation a b x y) ∧ 
  (∀ (A D : ℝ × ℝ) (B : ℝ × ℝ), 
   let k1 := slope B D,
   let k2 := slope (0, 0) A in 
   k1 * k2 = - (1 / 6)) := sorry

end ellipse_properties_l219_219896


namespace complete_square_l219_219670

theorem complete_square {x : ℝ} (h : x^2 + 10 * x - 3 = 0) : (x + 5)^2 = 28 :=
sorry

end complete_square_l219_219670


namespace license_plate_combination_count_l219_219086

theorem license_plate_combination_count :
  let letters := 26 in
  let odd_digits := 5 in
  let even_digits := 5 in
  let multiple_of_3_digits := 4 in
  (letters ^ 3) * odd_digits * even_digits * multiple_of_3_digits = 878800 :=
by 
  let letters := 26
  let odd_digits := 5
  let even_digits := 5
  let multiple_of_3_digits := 4
  show (letters ^ 3) * odd_digits * even_digits * multiple_of_3_digits = 878800
  sorry

end license_plate_combination_count_l219_219086


namespace area_is_12_5_l219_219991

-- Define the triangle XYZ
structure Triangle := 
  (X Y Z : Type) 
  (XZ YZ : ℝ) 
  (angleX angleY angleZ : ℝ)

-- Provided conditions in the problem
def triangleXYZ : Triangle := {
  X := ℝ, 
  Y := ℝ, 
  Z := ℝ, 
  XZ := 5,
  YZ := 5,
  angleX := 45,
  angleY := 45,
  angleZ := 90
}

-- Lean statement to prove the area of triangle XYZ
theorem area_is_12_5 (t : Triangle) 
  (h1 : t.angleZ = 90)
  (h2 : t.angleX = 45)
  (h3 : t.angleY = 45)
  (h4 : t.XZ = 5)
  (h5 : t.YZ = 5) : 
  (1/2 * t.XZ * t.YZ) = 12.5 :=
sorry

end area_is_12_5_l219_219991


namespace four_letter_arrangements_count_l219_219085

theorem four_letter_arrangements_count : 
  (∃! arrangements : Finset (List Char),
     (∀ l ∈ arrangements, l.length = 4) ∧
     ('D' = l.head) ∧
     ('A' ∈ l.tail ∧ 'B' ∈ l.tail) ∧
     (l.nodup)) ->
  arrangements.card = 30 :=
sorry

end four_letter_arrangements_count_l219_219085


namespace zero_function_of_sum_condition_l219_219672

open Real

theorem zero_function_of_sum_condition (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b)
  (h2 : ∀ x ∈ Ioo a b, f x = 0)
  (h3 : ∀ (p : ℕ) (hp : Nat.Prime p) (y : ℝ),
    (∑ k in Finset.range p, f (y + k / p)) = 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end zero_function_of_sum_condition_l219_219672


namespace find_value_of_6b_l219_219090

theorem find_value_of_6b (a b : ℝ) (h1 : 10 * a = 20) (h2 : 120 * a * b = 800) : 6 * b = 20 :=
by
  sorry

end find_value_of_6b_l219_219090


namespace katrina_cookies_left_l219_219607

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l219_219607


namespace exists_distinct_permutations_divisible_l219_219900

open Finset

noncomputable def S (c : ℕ → ℤ) (a : Perm (Fin n)) := 
  ∑ i in range n, c i * (a i)

theorem exists_distinct_permutations_divisible (c : ℕ → ℤ) (h : Odd n) :
  ∃ (a b : Perm (Fin n)), a ≠ b ∧ (S c a - S c b) % (nat.factorial n) = 0 :=
begin
  sorry
end

end exists_distinct_permutations_divisible_l219_219900


namespace range_a_l219_219903

-- Define the function y
def y (a x : ℝ) : ℝ := log a (2 - a * x)

-- Define the condition for y being decreasing on the interval [0, 1]
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x ≤ y → f y ≤ f x

-- The interval [0, 1]
def interval : set ℝ := set.Icc 0 1

-- The main theorem to prove
theorem range_a (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing_on (y a) interval := sorry

end range_a_l219_219903


namespace ordered_pairs_satisfy_equation_l219_219554

theorem ordered_pairs_satisfy_equation :
  (∃ (a : ℝ) (b : ℤ), a > 0 ∧ 3 ≤ b ∧ b ≤ 203 ∧ (Real.log a / Real.log b) ^ 2021 = Real.log (a ^ 2021) / Real.log b) :=
sorry

end ordered_pairs_satisfy_equation_l219_219554


namespace sum_prime_factors_143_l219_219320

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219320


namespace number_of_non_representable_l219_219619

theorem number_of_non_representable :
  ∀ (a b : ℕ), Nat.gcd a b = 1 →
  (∃ n : ℕ, ¬ ∃ x y : ℕ, n = a * x + b * y) :=
sorry

end number_of_non_representable_l219_219619


namespace sum_of_areas_of_circles_l219_219265

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l219_219265


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219753

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219753


namespace maximum_term_value_a_perpendicular_to_b_l219_219549

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def term_sequence (n : ℕ) : ℕ := n

def term_ratio (n : ℕ) : ℚ := term_sequence n / (term_sequence (n + 1) * term_sequence (n + 4))

theorem maximum_term_value : ∀ (n : ℕ) (positive_n : n > 0), a_perpendicular_to_b (2, -n) (sum_of_first_n_terms n, n + 1) → term_ratio n ≤ 1 / 9 :=
by sorry

-- Helper theorem to express the orthogonality condition
theorem a_perpendicular_to_b {real : Type} [linear_ordered_field real] {a b : real} (vec_a : real × real) (vec_b : real × real) : 
  (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 0) ↔ (2 * sum_of_first_n_terms (vec_a.2.to_nat) - vec_a.2 * (vec_b.2) = 0) :=
by sorry

end maximum_term_value_a_perpendicular_to_b_l219_219549


namespace sum_prime_factors_143_l219_219311

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219311


namespace ribbon_unused_length_l219_219604

theorem ribbon_unused_length (total_length : ℕ) (parts : ℕ) (used_parts : ℕ) (length_each_part : ℕ) (used_length : ℕ) (total_length_eq : total_length = 30) (parts_eq : parts = 6) (used_parts_eq : used_parts = 4) (length_each_part_eq : length_each_part = total_length / parts) (used_length_eq : used_length = used_parts * length_each_part) : (total_length - used_length) = 10 := 
by
  rw [total_length_eq, parts_eq, used_parts_eq, length_each_part_eq, used_length_eq]
  sorry

end ribbon_unused_length_l219_219604


namespace sum_prime_factors_143_is_24_l219_219285

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219285


namespace sum_of_squares_of_roots_eq_14_l219_219911

theorem sum_of_squares_of_roots_eq_14 {α β γ : ℝ}
  (h1: ∀ x: ℝ, (x^3 - 6*x^2 + 11*x - 6 = 0) → (x = α ∨ x = β ∨ x = γ)) :
  α^2 + β^2 + γ^2 = 14 :=
by
  sorry

end sum_of_squares_of_roots_eq_14_l219_219911


namespace brocard_theorem_l219_219630

/-- The Brocard theorem statement in Lean 4: Given an inscribed quadrilateral, 
let \( P, Q, \) and \( M \) be the points of intersection of the diagonals 
and the extensions of opposite sides, respectively. We need to prove that 
the orthocenter of triangle \( P Q M \) coincides with the circumcenter of the quadrilateral. -/
theorem brocard_theorem
    (P Q M O : Type*)
    [metric_space P]
    [metric_space Q]
    [metric_space M]
    [metric_space O]
    (is_circumcenter : P → Q → M → O → Prop)
    (is_orthocenter : P → Q → M → O → Prop)
    (h1 : is_circumcenter P Q M O)
    (h2 : is_orthocenter P Q M O) : 
  O = P ∧ Q = M :=
by
  sorry

end brocard_theorem_l219_219630


namespace sum_even_integers_102_to_200_l219_219705

theorem sum_even_integers_102_to_200 : 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 1 102), 2 * k) = 2550 →
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) + 1250 = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100) + ∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 1300 → 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 200), 2 * i) = 1250 :=
begin
  sorry
end

end sum_even_integers_102_to_200_l219_219705


namespace kata_first_move_equivalence_l219_219629

def KP : Set (ℕ × ℕ × ℕ) := {p | ∃ x y z, p = (x, y, z) ∧ x ∈ {1,2,3,4} ∧ y ∈ {1,2,3,4} ∧ z ∈ {1,2,3,4}}

def f (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 1
  | 3 => 4
  | 4 => 3
  | _ => 0 -- This is a safeguard for non-valid inputs, though in the conditions, n is always in {1,2,3,4}

def transform (p : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  match p with
  | (x, y, z) => (f x, f y, f z)

theorem kata_first_move_equivalence :
  let p1 := (1, 1, 2)
  let p2 := (2, 2, 1)
  transform p1 = p2 ∧ transform p2 = p1 →
  (game_result_when_kata_starts_at p1 = game_result_when_kata_starts_at p2) :=
by
  intros p1 p2 h_trans_eq
  sorry

end kata_first_move_equivalence_l219_219629


namespace product_of_sequence_l219_219859

theorem product_of_sequence :
  ∏ n in finset.range 98, (n * (n + 3)) / ((n + 1) * (n + 1)) = (101 / 198) :=
by
  sorry

end product_of_sequence_l219_219859


namespace sum_b_five_l219_219065

noncomputable def a : ℕ → ℤ
| n := 2 * n - 1

noncomputable def b : ℕ → ℤ
| 1 := 2
| n + 1 := (1 - 2 * (n + 1)) * (2 ^ (n + 1))

noncomputable def S : ℕ → ℤ
| 0     := 0
| (n + 1) := S n + b (n + 1)

theorem sum_b_five : S 5 = -450 :=
by sorry

end sum_b_five_l219_219065


namespace number_of_ways_to_make_78_rubles_l219_219584

theorem number_of_ways_to_make_78_rubles : ∃ n, n = 5 ∧ ∃ x y : ℕ, 78 = 5 * x + 3 * y := sorry

end number_of_ways_to_make_78_rubles_l219_219584


namespace add_octal_numbers_base8_eq_l219_219433

theorem add_octal_numbers_base8_eq (a b : ℕ) (h1 : a = 5) (h2 : b = 017) : 
  (a + b) = 24_8 :=
by 
  sorry

end add_octal_numbers_base8_eq_l219_219433


namespace find_a_l219_219954

theorem find_a (f : ℕ → ℕ) (a : ℕ) 
  (h1 : ∀ x : ℕ, f (x + 1) = x) 
  (h2 : f a = 8) : a = 9 :=
sorry

end find_a_l219_219954


namespace votes_for_Crow_l219_219114

theorem votes_for_Crow 
  (J : ℕ)
  (P V K : ℕ)
  (ε1 ε2 ε3 ε4 : ℤ)
  (h₁ : P + V = 15 + ε1)
  (h₂ : V + K = 18 + ε2)
  (h₃ : K + P = 20 + ε3)
  (h₄ : P + V + K = 59 + ε4)
  (bound₁ : |ε1| ≤ 13)
  (bound₂ : |ε2| ≤ 13)
  (bound₃ : |ε3| ≤ 13)
  (bound₄ : |ε4| ≤ 13)
  : V = 13 :=
sorry

end votes_for_Crow_l219_219114


namespace sum_of_prime_factors_143_l219_219294

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219294


namespace john_annual_profit_l219_219143

-- Definitions of monthly incomes
def TenantA_income : ℕ := 350
def TenantB_income : ℕ := 400
def TenantC_income : ℕ := 450

-- Total monthly income
def total_monthly_income : ℕ := TenantA_income + TenantB_income + TenantC_income

-- Definitions of monthly expenses
def rent_expense : ℕ := 900
def utilities_expense : ℕ := 100
def maintenance_fee : ℕ := 50

-- Total monthly expenses
def total_monthly_expense : ℕ := rent_expense + utilities_expense + maintenance_fee

-- Monthly profit
def monthly_profit : ℕ := total_monthly_income - total_monthly_expense

-- Annual profit
def annual_profit : ℕ := monthly_profit * 12

theorem john_annual_profit :
  annual_profit = 1800 := by
  -- The proof is omitted, but the statement asserts that John makes an annual profit of $1800.
  sorry

end john_annual_profit_l219_219143


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219749

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219749


namespace measure_angle_C_area_of_triangle_l219_219099

open Real
open Tactic

variable (A B C a b c : ℝ)

-- Defining the conditions
def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = π ∧
  2 * sin (A + B) / 2 ^ 2 = sin C + 1

-- Part I: Proving the measure of angle C
theorem measure_angle_C (h : triangle_ABC A B C) : C = π / 4 :=
sorry

-- Part II: Proving the area given additional conditions
theorem area_of_triangle
  (h : triangle_ABC A B C)
  (ha : a = sqrt 2)
  (hb : b = sqrt 2)
  (hc : c = 1)
  (hC : C = π / 4) : 
  Real.sqrt((s := (a + b + c) / 2) * ((s - a) * (s - b) * (s - c) * (s - C)) / 2 = 1 / 2 :=
sorry


end measure_angle_C_area_of_triangle_l219_219099


namespace h_is_decreasing_intervals_l219_219869

noncomputable def f (x : ℝ) := if x >= 1 then x - 2 else 0
noncomputable def g (x : ℝ) := if x <= 2 then -2 * x + 3 else 0

noncomputable def h (x : ℝ) :=
  if x >= 1 ∧ x <= 2 then f x * g x
  else if x >= 1 then f x
  else if x <= 2 then g x
  else 0

theorem h_is_decreasing_intervals :
  (∀ x1 x2 : ℝ, x1 < x2 → x1 < 1 → h x1 > h x2) ∧
  (∀ x1 x2 : ℝ, x1 < x2 → x1 ≥ 7 / 4 → x2 ≤ 2 → h x1 ≥ h x2) :=
by
  sorry

end h_is_decreasing_intervals_l219_219869


namespace cats_weight_more_than_puppies_l219_219927

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l219_219927


namespace fraction_of_total_amount_l219_219382

theorem fraction_of_total_amount (p q r : ℕ) (h1 : p + q + r = 4000) (h2 : r = 1600) :
  r / (p + q + r) = 2 / 5 :=
by
  sorry

end fraction_of_total_amount_l219_219382


namespace no_perfect_power_l219_219200

theorem no_perfect_power (n m : ℕ) (hn : 0 < n) (hm : 1 < m) : 102 ^ 1991 + 103 ^ 1991 ≠ n ^ m := 
sorry

end no_perfect_power_l219_219200


namespace count_valid_pairs_l219_219492

theorem count_valid_pairs : 
    let lst := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6] in 
    let valid_pairs := [(x, y) | x ∈ lst, y ∈ lst, y > x, x + y = 3] in
    valid_pairs.length = 5 :=
by 
  let lst := [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
  let valid_pairs := [(x, y) | x ∈ lst, y ∈ lst, y > x, x + y = 3]
  show valid_pairs.length = 5
  valid_pairs.length = 5


end count_valid_pairs_l219_219492


namespace seating_arrangement_ways_l219_219981

/-- The number of ways to seat four people in a row of six chairs -/
theorem seating_arrangement_ways :
  ∃ n : ℕ, n = 6 * 5 * 4 * 3 ∧ n = 360 :=
begin
  use 360,
  simp,
  sorry, -- Proof is not required here
end

end seating_arrangement_ways_l219_219981


namespace angle_AOC_eq_180_degrees_l219_219987

-- Define angles and perpendicularity conditions
noncomputable def Angle (x y : Point) := sorry

axiom perp_AO_DO : ∀ A O D : Point, Perp (OA : Vector) (OD : Vector)
axiom perp_BO_CO : ∀ B O C : Point, Perp (OB : Vector) (OC : Vector)
axiom angle_AOC_4times_BOD : ∀ A O C B D : Point, Angle A O C = 4 * Angle B O D

-- Define the points A, O, B, C, D.
variables (A O B C D : Point)

-- The statement we need to prove
theorem angle_AOC_eq_180_degrees : Angle A O C = 180 :=
by
  sorry

end angle_AOC_eq_180_degrees_l219_219987


namespace problem_1_problem_2_l219_219277

noncomputable def k_value (θ₀ θ₁ θ : ℝ) (t : ℝ) :=
  real.log ((θ₁ - θ₀) / (θ - θ₀)) / -t

theorem problem_1 : 
  let θ₀ := 20 
      θ₁ := 98 
      θ := 71.2 
      t := 1 
  in abs (k_value θ₀ θ₁ θ t - 0.029) < 0.001 := 
by 
  sorry

noncomputable def room_temp (k θ₁ θ : ℝ) (t : ℝ) :=
  θ₁ + (θ - θ₁) * real.exp (k * t)

theorem problem_2 :
  let k := 0.01 
      θ₁ := 100 
      θ := 40 
      t := 2.5 
  in abs (room_temp k θ₁ θ t - 20.0) < 0.1 := 
by 
  sorry

end problem_1_problem_2_l219_219277


namespace sum_of_prime_factors_of_143_l219_219335

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219335


namespace no_quadratic_polynomials_with_given_zeros_l219_219855

theorem no_quadratic_polynomials_with_given_zeros :
  ¬ ∃ (P Q : ℝ[X]), degree P = 2 ∧ degree Q = 2 ∧ 
  (∀ (x : ℝ), P.eval (Q.eval x) = 0 ↔ x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7) :=
sorry

end no_quadratic_polynomials_with_given_zeros_l219_219855


namespace facemasks_per_box_l219_219550

theorem facemasks_per_box (x : ℝ) :
  (3 * x * 0.50) - 15 = 15 → x = 20 :=
by
  intros h
  sorry

end facemasks_per_box_l219_219550


namespace find_triples_l219_219021

theorem find_triples (m n x y : ℕ) (a : ℕ) (hpos_m : m > 0) (hpos_n : n > 0) (hpos_x : x > 0) (hpos_y : y > 0)
  (hm_even : m % 2 = 0) (h_gcd : Nat.gcd m n = 1) (h_eq : (x^2 + y^2)^m = (x * y)^n) :
  ∃ a : ℕ, (n, x, y) = (m + 1, 2^a, 2^a) :=
begin
  sorry
end

end find_triples_l219_219021


namespace find_a_l219_219638

-- Let \( a \) and \( k \) be real numbers
variables {a k : ℝ}

-- The conditions of the problem
def condition1 := |18 * a - 18 * k - 153| = 1005
def condition2 := |18 * a ^ 2 - 18 * k - 153| = 865

-- The theorem to be proven
theorem find_a (h1 : condition1) (h2 : condition2) : a = - 7 / 3 :=
by
  sorry

end find_a_l219_219638


namespace ratio_third_median_altitude_l219_219110

noncomputable theory
open_locale classical

variables {A B C A' B' M : Type*}
variable [is_scalar_tower A B C]
variables (triangle : triangle A B C)
variables [scalene_triangle triangle]
variables (m_A' : median A A') (m_B' : median B B')
variables (h_B : altitude B) (h_C : altitude C)

def conditions := 
  (m_A'.length = h_B.length) ∧ 
  (m_B'.length = h_C.length)

theorem ratio_third_median_altitude (h : conditions triangle m_A' m_B' h_B h_C) :
  ratio_third_median_altitude = 7 / 2 :=
sorry

end ratio_third_median_altitude_l219_219110


namespace intersection_A_B_l219_219053

def A : Set ℝ := { x | x^2 - 2*x < 0 }
def B : Set ℝ := { x | |x| > 1 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end intersection_A_B_l219_219053


namespace sum_of_prime_factors_of_143_l219_219331

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219331


namespace sum_prime_factors_of_143_l219_219337

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219337


namespace tangent_line_equation_at_one_two_l219_219678

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 / x

theorem tangent_line_equation_at_one_two :
  let slope := Deriv f 1
  in let tangent_line := (λ x y : ℝ, y - 2 = slope * (x - 1))
  in tangent_line 1 2 = 0 ∧ 
     ∀ x y : ℝ, tangent_line x y = 0 ↔ x + y - 3 = 0 :=
by
  sorry

end tangent_line_equation_at_one_two_l219_219678


namespace sum_prime_factors_143_is_24_l219_219288

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219288


namespace rectangle_of_abcd_l219_219115

variables (V : Type*) [inner_product_space ℝ V]
variables (A B C D : V)

-- Given conditions
def ab_eq_dc : Prop :=
  (B - A) = (D - C)

def ad_ab_eq_ad_plus_ab : Prop :=
  ‖(A - D) - (A - B)‖ = ‖(A - D) + (A - B)‖

-- To Prove
theorem rectangle_of_abcd (h1 : ab_eq_dc V A B C D) (h2 : ad_ab_eq_ad_plus_ab V A B C D) : 
  inner_product_space.is_orthogonal (A - D) (A - B) :=
by sorry

end rectangle_of_abcd_l219_219115


namespace subsets_with_at_least_four_adjacent_chairs_l219_219758

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219758


namespace find_m_existence_l219_219890

theorem find_m_existence
  (α : ℝ)
  (m : ℝ)
  (hm_alpha : P (m, m+1) ∃ ∧ cos α = 3/5)
  (cos_definition : cos α = m / sqrt (m^2 + (m + 1)^2)) :
  ∃ m, m = 3 :=
by
  sorry

end find_m_existence_l219_219890


namespace fraction_simplification_l219_219860

theorem fraction_simplification :
  (3 / 7 + 5 / 8 + 2 / 9) / (5 / 12 + 1 / 4) = 643 / 336 :=
by
  sorry

end fraction_simplification_l219_219860


namespace last_score_is_80_l219_219648

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end last_score_is_80_l219_219648


namespace f_inv_f_inv_15_l219_219671

def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inv (x : ℝ) : ℝ := (x - 6) / 3

theorem f_inv_f_inv_15 : f_inv (f_inv 15) = -1 :=
by
  sorry

end f_inv_f_inv_15_l219_219671


namespace triangle_equilateral_l219_219128

theorem triangle_equilateral (a b c : ℝ) (C : ℝ) 
  (h1 : C = 60) (h2 : c^2 = a * b) : a = b ∧ b = c := by
  -- We prefer using degrees for angles and ensuring consitency
  have h3 : 2 * c^2 = 2 * a * b, from congr_arg (λ x, 2 * x) h2
  
  -- Using Law of Cosines directly
  have h4: c^2 = a^2 + b^2 - 2 * a * b * (Real.cos (C * Real.pi / 180)), from sorry,

  -- Based on the given angle C = 60 degrees, cos(60 degrees) = 1/2
  have cos_60_deg : Real.cos (60 * Real.pi / 180) = 1/2, from sorry,
  
  -- Substitute the value of cos 60 degrees in the equation
  have h5 : c^2 = a^2 + b^2 - a * b, from sorry,

  -- Given c^2 = a * b
  have h6 : a * b = a^2 + b^2 = a*b, from sorry,

  -- Rearranging to prove a = b = c
  have h7 : 0 = a^2 - a * b +  b^2, from sorry,
  have h8 : (a - b)^2 = 0, from sorry,
  have h9 : a = b, from sorry,
  have h10 : 2 * c^2 = 2 * a * b, from congr_arg (λ x, 2 * x) h2,
  have h11 : 2 * a = 2 * b, from congr_arg (λ x, 2 * x) h9,
  have h12 : c = b, from sorry
  have h13 : a = b ∧ b = c, from sorry
  apply h13

end triangle_equilateral_l219_219128


namespace find_k_l219_219657

noncomputable theory

-- Definition of the inverse proportion function
def inverseProportionFunction (k : ℝ) (x : ℝ) : ℝ := k / x

-- Definitions of the given conditions
def point_P := (-1, 3)

-- The main theorem statement
theorem find_k (k : ℝ) : inverseProportionFunction k (-1) = 3 → k = -3 :=
by
  intro h
  sorry

end find_k_l219_219657


namespace jeans_price_difference_l219_219785

variable (x : Real)

theorem jeans_price_difference
  (hx : 0 < x) -- Assuming x > 0 for a positive cost
  (r := 1.40 * x)
  (c := 1.30 * r) :
  c = 1.82 * x :=
by
  sorry

end jeans_price_difference_l219_219785


namespace sum_of_prime_factors_143_l219_219298

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219298


namespace triangle_is_obtuse_l219_219992

variables {A B C : ℝ}

def is_triangle_ABC : Prop := A + B + C = π

def condition : Prop := sin A * sin B < cos A * cos B

def is_obtuse_triangle_ABC : Prop := π / 2 < C ∧ C < π

theorem triangle_is_obtuse (htriangle : is_triangle_ABC) (hcondition : condition) : is_obtuse_triangle_ABC :=
sorry

end triangle_is_obtuse_l219_219992


namespace no_13_cities_with_5_roads_each_l219_219995

theorem no_13_cities_with_5_roads_each :
  ¬ ∃ (G : SimpleGraph (Fin 13)), (∀ v, G.degree v = 5) :=
by
  sorry

end no_13_cities_with_5_roads_each_l219_219995


namespace value_of_x_l219_219778

theorem value_of_x (x : ℝ) (h : (sqrt x) ^ 4 = 256) : x = 16 :=
by sorry

end value_of_x_l219_219778


namespace count_divisible_digits_l219_219872

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end count_divisible_digits_l219_219872


namespace expression_eq_one_l219_219459

variable {b x : ℝ}
variable (hb : b ≠ 0)

def expression (b x : ℝ) : ℝ :=
  ((b / (b - x)) - (x / (b + x))) / ((b / (b + x)) + (x / (b - x)))

theorem expression_eq_one (hb : b ≠ 0) (hx : x ≠ b) (hx_neg : x ≠ -b) :
  expression b x = 1 := sorry

end expression_eq_one_l219_219459


namespace expression_equals_one_l219_219660

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end expression_equals_one_l219_219660


namespace find_a22_l219_219237

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l219_219237


namespace sum_prime_factors_143_l219_219324

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219324


namespace sum_of_circle_areas_l219_219254

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l219_219254


namespace first_number_in_a10_l219_219187

-- Define a function that captures the sequence of the first number in each sum 'a_n'.
def first_in_an (n : ℕ) : ℕ :=
  1 + 2 * (n * (n - 1)) / 2 

-- State the theorem we want to prove
theorem first_number_in_a10 : first_in_an 10 = 91 := 
  sorry

end first_number_in_a10_l219_219187


namespace cats_weigh_more_by_5_kg_l219_219931

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l219_219931


namespace find_a22_l219_219247

-- Definitions and conditions
noncomputable def seq (n : ℕ) : ℝ := if n = 0 then 0 else sorry

axiom seq_conditions
  (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) : True

theorem find_a22 (a : ℕ → ℝ)
  (h1 : ∀ n ∈ Finset.range 99, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 :=
sorry

end find_a22_l219_219247


namespace value_of_x_l219_219559

theorem value_of_x (x : ℝ) (hx_pos : 0 < x) (hx_eq : x^2 = 1024) : x = 32 := 
by
  sorry

end value_of_x_l219_219559


namespace tangent_function_range_l219_219412

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

theorem tangent_function_range {a : ℝ} :
  (∃ (m : ℝ), 4 * m^3 - 3 * a * m^2 + 6 = 0) ↔ a > 2 * Real.sqrt 33 :=
sorry -- proof omitted

end tangent_function_range_l219_219412


namespace mark_walk_distance_l219_219032

def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem mark_walk_distance :
  let north := 30
  let west := 80
  let south := 20
  let east := 30
  let net_y := north - south
  let net_x := west - east
  let D := distance 0 0 net_x net_y
  D = 51 :=
by
  let north := 30
  let west := 80
  let south := 20
  let east := 30
  let net_y := north - south
  let net_x := west - east
  have h_net_y : net_y = 10 := rfl
  have h_net_x : net_x = 50 := rfl
  let D := distance 0 0 net_x net_y
  have h_distance : D = real.sqrt (10^2 + 50^2) := by rw [distance, h_net_y, h_net_x]; norm_num
  have h_distance_sqrt : real.sqrt (10^2 + 50^2) = real.sqrt 2600 := by rw [nat.pow_two, nat.pow_two]; norm_num
  sorry

end mark_walk_distance_l219_219032


namespace power_mod_result_l219_219836

theorem power_mod_result :
  (47 ^ 1235 - 22 ^ 1235) % 8 = 7 := by
  sorry

end power_mod_result_l219_219836


namespace AKLB_is_rhombus_l219_219656

-- Definitions based on conditions
variables {A B C D K L : Type}
variables [is_perpendicular A CD]
variables [is_perpendicular B CD]
variables [is_intersection_point K BD]
variables [is_intersection_point L AC]

-- Statement of the problem
theorem AKLB_is_rhombus : is_rhombus AKLB := 
sorry

end AKLB_is_rhombus_l219_219656


namespace determinant_tan_matrix_l219_219472

theorem determinant_tan_matrix (B C : ℝ) (h : B + C = 3 * π / 4) :
  Matrix.det ![
    ![Real.tan (π / 4), 1, 1],
    ![1, Real.tan B, 1],
    ![1, 1, Real.tan C]
  ] = 1 :=
by
  sorry

end determinant_tan_matrix_l219_219472


namespace find_integer_k_l219_219888

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - ((Real.sin x) + (Real.cos x))^2 - 2

theorem find_integer_k :
  (∃ x1 x2 : ℝ, x1 ∈ set.Icc 0 (3 * Real.pi / 4) ∧ x2 ∈ set.Icc 0 (3 * Real.pi / 4) ∧ f x1 < (-3) ∧ (-3) < f x2) :=
sorry

end find_integer_k_l219_219888


namespace Andy_is_1_year_older_l219_219965

variable Rahim_current_age : ℕ
variable Rahim_age_in_5_years : ℕ := Rahim_current_age + 5
variable Andy_age_in_5_years : ℕ := 2 * Rahim_current_age
variable Andy_current_age : ℕ := Andy_age_in_5_years - 5

theorem Andy_is_1_year_older :
  Rahim_current_age = 6 → Andy_current_age = Rahim_current_age + 1 :=
by
  sorry

end Andy_is_1_year_older_l219_219965


namespace sum_of_prime_factors_143_l219_219299

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219299


namespace winning_candidate_percentage_correct_voter_turnout_percentage_correct_l219_219104

-- Define the conditions
def votes_candidate1 : ℕ := 4136
def votes_candidate2 : ℕ := 7636
def votes_candidate3 : ℕ := 11628
def votes_candidate4 : ℕ := 9822
def votes_candidate5 : ℕ := 8705
def invalid_votes : ℕ := 325
def total_registered_voters : ℕ := 50000

-- Define the total votes calculated
def total_valid_votes : ℕ := votes_candidate1 + votes_candidate2 + votes_candidate3 + votes_candidate4 + votes_candidate5
def total_votes_cast : ℕ := total_valid_votes + invalid_votes

-- Define the winning candidate's votes
def winning_candidate_votes : ℕ := votes_candidate3

-- Define the expected results
def percentage_votes_winning_candidate : ℚ := (winning_candidate_votes : ℚ) / (total_valid_votes : ℚ) * 100
def turnout_percentage : ℚ := (total_votes_cast : ℚ) / (total_registered_voters : ℚ) * 100

-- Theorem statements
theorem winning_candidate_percentage_correct : percentage_votes_winning_candidate ≈ 27.73 :=
sorry

theorem voter_turnout_percentage_correct : turnout_percentage ≈ 84.504 :=
sorry

end winning_candidate_percentage_correct_voter_turnout_percentage_correct_l219_219104


namespace ratio_PQ_EF_l219_219663

/-- Define the rectangle ABCD and the relevant points E, F, G in a coordinate system, 
    and state the desired proof -/
theorem ratio_PQ_EF :
  ∃ (A B C D E F G P Q : ℝ × ℝ),
    -- Points Definition
    A = (0, 6) ∧
    B = (8, 6) ∧
    C = (8, 0) ∧
    D = (0, 0) ∧
    E = (6, 6) ∧
    F = (1, 0) ∧
    G = (8, 3) ∧
    -- Define lines and intersection points P and Q
    let AG_slope := (G.2 - A.2) / (G.1 - A.1),
        AC_slope := (C.2 - A.2) / (C.1 - A.1),
        EF_slope := (F.2 - E.2) / (F.1 - E.1),
        y_ag := A.2 + AG_slope * (8 - A.1),
        y_ac := A.2 + AC_slope * (8 - A.1),
        y_ef := E.2 + EF_slope * (1 - E.1),
        P_x := (6 - y_ac) / (-AC_slope + EF_slope),
        Q_x := (6 - y_ag) / (-AG_slope + EF_slope),
        P := (P_x, y_ac + AC_slope * P_x),
        Q := (Q_x, y_ag + AG_slope * Q_x),  
        EF_length := Real.sqrt ((E.2 - F.2)^2 + (E.1 - F.1)^2),
        PQ_distance := Real.abs (P.1 - Q.1),
        ratio := PQ_distance / EF_length
    -- Desired equivalence conclusion
    in ratio = 244 / (93 * Real.sqrt 61) :=
    sorry

end ratio_PQ_EF_l219_219663


namespace apple_pies_needed_l219_219645

theorem apple_pies_needed 
  (peach_pies blueberry_pies : ℕ) 
  (peach_cost apple_cost blueberry_cost : ℕ) 
  (total_spent : ℕ) :
  peach_pies = 5 → 
  blueberry_pies = 3 → 
  peach_cost = 2 → 
  apple_cost = 1 → 
  blueberry_cost = 1 → 
  total_spent = 51 →
  let peach_pie_cost := 3 * peach_cost
  let blueberry_pie_cost := 3 * blueberry_cost
  let apple_pie_cost := 3 * apple_cost
  let total_peach_cost := peach_pies * peach_pie_cost
  let total_blueberry_cost := blueberry_pies * blueberry_pie_cost
  let remaining_money := total_spent - total_peach_cost - total_blueberry_cost
  let apple_pies := remaining_money / apple_pie_cost
  apple_pies = 4 := by {
    intros h_peach h_blueberry h_peach_cost h_apple_cost h_blueberry_cost h_total_spent,
    dsimp at *,
    rw [h_peach, h_blueberry, h_peach_cost, h_apple_cost, h_blueberry_cost],
    let peach_pie_cost := 3 * 2,
    let blueberry_pie_cost := 3 * 1,
    let apple_pie_cost := 3 * 1,
    let total_peach_cost := 5 * peach_pie_cost,
    let total_blueberry_cost := 3 * blueberry_pie_cost,
    let remaining_money := 51 - total_peach_cost - total_blueberry_cost,
    let apple_pies := remaining_money / apple_pie_cost,
    calc
      apple_pies = (51 - (5 * (3 * 2)) - (3 * (3 * 1))) / (3 * 1) : by rfl
      ... = (51 - 30 - 9) / 3 : by norm_num
      ... = 12 / 3 : by norm_num
      ... = 4 : by norm_num
  }

end apple_pies_needed_l219_219645


namespace total_order_cost_l219_219857

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l219_219857


namespace image_center_of_circle_l219_219408

-- Definitions and conditions:
variable {P : Type*} -- Type for our plane
variable {Circle : Type*} -- Type for the circle in 3D space
variable (proj : Point → P) -- Parallel projection from 3D space to plane

-- Assuming the following conditions:
-- 1. The preservation of the midpoint under parallel projection.
axiom preserve_midpoint (A B : Point) : proj ((A + B) / 2) = (proj A + proj B) / 2

-- 2. The preservation of parallelism under parallel projection.
axiom preserve_parallel (A B C D : Point) :
  (A - B) ∥ (C - D) → (proj A - proj B) ∥ (proj C - proj D)

-- Main theorem to prove:
theorem image_center_of_circle {circle : Circle} (O : Point) (radius : Real) :
  let diam := some (diameter circle) in  -- Assuming some diameter of the circle
  let P := diam.1 in
  let Q := diam.2 in
  let Pproj := proj P in
  let Qproj := proj Q in
  let Oproj := proj O in
  Oproj = (Pproj + Qproj) / 2 :=
sorry -- Proof goes here

end image_center_of_circle_l219_219408


namespace right_triangle_distance_midpoint_l219_219587

noncomputable def distance_from_F_to_midpoint_DE
  (D E F : ℝ × ℝ)
  (right_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C) 
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)
  : ℝ :=
  if hD : (D.1 - E.1)^2 + (D.2 - E.2)^2 = DE^2 then
    if hF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = DF^2 then
      if hDE : DE = 15 then
        (15 / 2) --distance from F to midpoint of DE
      else
        0 -- This will never be executed since DE = 15 is a given condition
    else
      0 -- This will never be executed since DF = 9 is a given condition
  else
    0 -- This will never be executed since EF = 12 is a given condition

theorem right_triangle_distance_midpoint
  (D E F : ℝ × ℝ)
  (h_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C)
  (hDE : (D.1 - E.1)^2 + (D.2 - E.2)^2 = 15^2)
  (hDF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9^2)
  (hEF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 12^2) :
  distance_from_F_to_midpoint_DE D E F h_triangle 15 9 12 = 7.5 :=
by sorry

end right_triangle_distance_midpoint_l219_219587


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219752

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219752


namespace part_I_part_II_l219_219168

noncomputable def f (x a : ℝ) := exp x - a * x + a
noncomputable def f' (x a : ℝ) := exp x - a

theorem part_I (h : ∀ x : ℝ, f x a = 0) : a = exp 2 ∨ a > exp 2 := sorry

theorem part_II (x₁ x₂ a : ℝ) (h₁ : x₁ < x₂) (h₂ : exp x₁ - a * x₁ + a = 0) (h₃ : exp x₂ - a * x₂ + a = 0) (h₄ : a > exp 2) :
  f' (sqrt (x₁ * x₂)) a < 0 := sorry

end part_I_part_II_l219_219168


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219757

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219757


namespace sum_prime_factors_143_l219_219318

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219318


namespace hershel_fish_ratio_l219_219937

theorem hershel_fish_ratio (initial_betta : ℕ) (initial_goldfish : ℕ) 
(bexley_betta_factor : ℚ) (bexley_goldfish_factor : ℚ) 
(final_fish : ℕ) (betta_brought : ℕ := bexley_betta_factor * initial_betta) 
(goldfish_brought : ℕ := bexley_goldfish_factor * initial_goldfish) : 
  initial_betta = 10 →
  initial_goldfish = 15 →
  bexley_betta_factor = 2/5 →
  bexley_goldfish_factor = 1/3 →
  final_fish = 17 →
  (initial_betta + betta_brought + initial_goldfish + goldfish_brought - final_fish) / 
  (initial_betta + betta_brought + initial_goldfish + goldfish_brought) = 1/2 :=
by
  intros h_initial_betta h_initial_goldfish h_bexley_betta_factor h_bexley_goldfish_factor h_final_fish
  have h_total_betta := show betta_brought = 4 by sorry
  have h_total_goldfish := show goldfish_brought = 5 by sorry
  have h_total_fish := show initial_betta + betta_brought + initial_goldfish + goldfish_brought = 34 by sorry
  have h_gifted_fish := show initial_betta + betta_brought + initial_goldfish + goldfish_brought - final_fish = 17 by sorry
  sorry

end hershel_fish_ratio_l219_219937


namespace sum_prime_factors_143_l219_219314

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219314


namespace last_score_is_80_l219_219649

def is_integer (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ k : ℤ, n = k * d

def average_condition (scores : List ℤ) : Prop :=
  (∀ i in List.range scores.length, is_integer (scores.take i |>.sum) (i + 1))

theorem last_score_is_80 (scores : List ℤ) (h : scores = [71, 76, 80, 82, 91]) :
  average_condition (insert_nth 80 scores) :=
by
  sorry

end last_score_is_80_l219_219649


namespace equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l219_219387

-- Definition of volumes for rectangular parallelepipeds
def volume_rect_parallelepiped (a b c: ℝ) : ℝ := a * b * c

-- Definition of volumes for prisms
def volume_prism (base_area height: ℝ) : ℝ := base_area * height

-- Definition of decomposability of rectangular parallelepipeds
def decomposable_rect_parallelepipeds (a1 b1 c1 a2 b2 c2: ℝ) : Prop :=
  (volume_rect_parallelepiped a1 b1 c1) = (volume_rect_parallelepiped a2 b2 c2)

-- Lean statement for part (a)
theorem equal_volume_rect_parallelepipeds_decomposable (a1 b1 c1 a2 b2 c2: ℝ) (h: decomposable_rect_parallelepipeds a1 b1 c1 a2 b2 c2) :
  True := sorry

-- Definition of decomposability of prisms
def decomposable_prisms (base_area1 height1 base_area2 height2: ℝ) : Prop :=
  (volume_prism base_area1 height1) = (volume_prism base_area2 height2)

-- Lean statement for part (b)
theorem equal_volume_prisms_decomposable (base_area1 height1 base_area2 height2: ℝ) (h: decomposable_prisms base_area1 height1 base_area2 height2) :
  True := sorry

end equal_volume_rect_parallelepipeds_decomposable_equal_volume_prisms_decomposable_l219_219387


namespace common_difference_of_arithmetic_sequence_l219_219120

theorem common_difference_of_arithmetic_sequence
  (a_n : ℕ → ℚ)
  (a_7 : a_n 7 = 8)
  (S_7 : ∑ i in finset.range 7, a_n i = 42) :
  ∃ d : ℚ, d = 2 / 3 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l219_219120


namespace total_spent_l219_219023

theorem total_spent (B D : ℝ) (h1 : D = 0.7 * B) (h2 : B = D + 15) : B + D = 85 :=
sorry

end total_spent_l219_219023


namespace sum_of_prime_factors_143_l219_219292

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219292


namespace locus_of_centers_tangent_circles_l219_219229

-- Definition of circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 9

-- Distance between centers
def distance_centers : ℝ := 4

theorem locus_of_centers_tangent_circles : 
    (∃ h : (x y : ℝ), (circle1 x y) ∧ (circle2 x y)) → 
    ((∃ x y : ℝ, (4 * (x-4)^2 - y^2 = 0)) ∨ 
     (∃ x : ℝ, (x = 2))) :=
sorry

end locus_of_centers_tangent_circles_l219_219229


namespace sum_prime_factors_of_143_l219_219342

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219342


namespace angle_C_eq_2pi_over_3_l219_219514

-- Define the given vectors.
def vector_m (C : ℝ) : ℝ × ℝ := (2 * Real.cos C - 1, -2)
def vector_n (C : ℝ) : ℝ × ℝ := (Real.cos C, Real.cos C + 1)

-- Define the dot product of two vectors.
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition stating that m is perpendicular to n
def perp_condition (C : ℝ) : Prop := dot_product (vector_m C) (vector_n C) = 0

-- Statement to prove
theorem angle_C_eq_2pi_over_3 (C : ℝ) (h_angle_int : C ∈ Set.Ioo 0 Real.pi)
    (h_perp : perp_condition C) : C = 2 * Real.pi / 3 := by
  sorry

end angle_C_eq_2pi_over_3_l219_219514


namespace exist_point_M_l219_219846

open EuclideanGeometry

-- Definitions of points, lines, and conditions from the problem.
variable {A B C : Point} 
variable (Δ : Triangle A B C)

-- Assumption that the triangle is acute-angled
def acute_angled (Δ : Triangle A B C) : Prop := acute ∆.angleA ∧ acute Δ.angleB ∧ acute Δ.angleC

-- Definition of the point P where perpendiculars from B to AC and from C to AB meet
def point_P := 
  meet_lines (altitude Δ B (line_segment A C)) (altitude Δ C (line_segment A B))

-- The existence of point M on side BC such that the mentioned conditions hold
theorem exist_point_M (h_acute : acute_angled Δ) : 
  ∃ M : Point, M ∈ line_segment B C ∧
  let p := point_P Δ
  homothety_centered A p M ∧
  is_parallel (line_through (foot_perpendicular M (line_segment A B)) (foot_perpendicular M (line_segment A C))) (line_segment B C) :=
begin
  sorry
end

end exist_point_M_l219_219846


namespace rational_sum_eq_one_l219_219094

theorem rational_sum_eq_one (a b : ℚ) (h : |3 - a| + (b + 2)^2 = 0) : a + b = 1 := 
by
  sorry

end rational_sum_eq_one_l219_219094


namespace bridge_length_l219_219423

theorem bridge_length
  (train_length : ℝ) (train_speed_kmph : ℝ) (cross_time : ℝ) :
  train_length = 110 →
  train_speed_kmph = 36 →
  cross_time = 24.198064154867613 →
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_speed_mps * cross_time in
  total_distance - train_length = 131.98064154867613 :=
by
  intros h1 h2 h3
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * cross_time
  have ht : train_speed_mps = 10 := by
    rw [h2, ←mul_div_assoc, mul_comm 1000, div_eq_mul_one_div, mul_assoc, div_self, ←one_mul, div_one,
      mul_comm, mul_comm 36, one_div, one_mul, one_comm_heat]
  have td : total_distance = train_speed_mps * 24.198064154867613 := rfl
  calc
    total_distance - train_length
      = (train_speed_mps * cross_time) - train_length : by rw td
  ... = (10 * 24.198064154867613) - 110 : by rw [h1, h3, ht]
  ... = 241.98064154867613 - 110 : rfl
  ... = 131.98064154867613 : rfl

end bridge_length_l219_219423


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219747

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219747


namespace isosceles_triangle_sides_l219_219822

theorem isosceles_triangle_sides (length_rope : ℝ) (one_side : ℝ) (a b : ℝ) :
  length_rope = 18 ∧ one_side = 5 ∧ a + a + one_side = length_rope ∧ b = one_side ∨ b + b + one_side = length_rope -> (a = 6.5 ∨ a = 5) ∧ (b = 6.5 ∨ b = 5) :=
by
  sorry

end isosceles_triangle_sides_l219_219822


namespace task_completion_days_l219_219797

theorem task_completion_days (a b c: ℕ) :
  (b = a + 6) → (c = b + 3) → 
  (3 / a + 4 / b = 9 / c) →
  a = 18 ∧ b = 24 ∧ c = 27 :=
  by
  sorry

end task_completion_days_l219_219797


namespace quadratic_residue_solution_l219_219009

theorem quadratic_residue_solution (a b c : ℕ) (h : ∀ (p : ℕ) [Fact p.prime], ∀ (n : ℕ), n^2 % p = n % p → (a * n^2 + b * n + c) % p = (a * n^2 + b * n + c) % p) :
    ∃ (d e : ℤ), a = d^2 ∧ b = 2 * d * e ∧ c = e^2 :=
by
  sorry

end quadratic_residue_solution_l219_219009


namespace r_s_t_u_bounds_l219_219513

theorem r_s_t_u_bounds (r s t u : ℝ) 
  (H1: 5 * r + 4 * s + 3 * t + 6 * u = 100)
  (H2: r ≥ s)
  (H3: s ≥ t)
  (H4: t ≥ u)
  (H5: u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := 
sorry

end r_s_t_u_bounds_l219_219513


namespace testing_methods_count_l219_219710

theorem testing_methods_count :
  let total_products := 7
  let defective_products := 4
  let non_defective_products := 3
  (total_products = 7 ∧ defective_products = 4 ∧ non_defective_products = 3) →
  (number_of_testing_methods := 1080) 
  ∃ (t1 t2 t3 t4 t5 t6 t7 : {0, 1}), 
    -- 1 represents defective and 0 non-defective products
    t4 = 1 ∧ 
    (t1 + t2 + t3 = 2) ∧ 
    (t5 + t6 + t7 = 1) →
    number_of_testing_methods = 1080 :=
by
  sorry

end testing_methods_count_l219_219710


namespace color_2013_2014_red_and_2_7_blue_l219_219471

def color (n m : ℕ) : Prop := 
  ∃ c : string, (c = "red" ∨ c = "blue") ∧
  ((n = 1 ∧ c = "red") ∨
   ((n ≠ 1 ∧ ∃ n' m', (n = n' + 1 ∧ m = m') ∨ (m = m' + 1 ∧ n = n')) ∧ 
    ((n ≠ 1 ∧ c = "blue") ∨ (n = 1 ∧ c = "red")) ∧ 
    (n ≠ 1 ∧ ∃ r : ℕ, (1 = r ∨ m = r) ∧ c ≠ "red")))

theorem color_2013_2014_red_and_2_7_blue :
  color 2013 2014 = "red" ∧ color 2 7 = "blue" :=
by
  sorry

end color_2013_2014_red_and_2_7_blue_l219_219471


namespace distance_from_M_to_plane_alpha_l219_219505

noncomputable def point : Type := (ℝ × ℝ × ℝ)

def A : point := (1, 2, 0)
def B : point := (-2, 0, 1)
def C : point := (0, 2, 2)
def M : point := (-1, 2, 3)

def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def dot_product (v1 v2 : point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : point) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def distance_point_to_plane (M A B C : point) : ℝ :=
  let AM := vector_sub M A
  let AB := vector_sub B A
  let AC := vector_sub C A
  let n := (2, -5/2, 1) -- normal vector manually given as per the problem's solved system
  in real.abs (dot_product AM n) / (norm n)

theorem distance_from_M_to_plane_alpha :
  distance_point_to_plane M A B C = 2 * real.sqrt 5 / 15 :=
by sorry

end distance_from_M_to_plane_alpha_l219_219505


namespace sum_prime_factors_143_l219_219325

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219325


namespace log_decreasing_range_l219_219914

theorem log_decreasing_range (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : log a 2 > log a 3) : a ∈ set.Ioo 0 1 :=
sorry

end log_decreasing_range_l219_219914


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219746

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219746


namespace tetradecagon_edge_length_correct_l219_219563

-- Define the parameters of the problem
def regular_tetradecagon_perimeter (n : ℕ := 14) : ℕ := 154

-- Define the length of one edge
def edge_length (P : ℕ) (n : ℕ) : ℕ := P / n

-- State the theorem
theorem tetradecagon_edge_length_correct :
  edge_length (regular_tetradecagon_perimeter 14) 14 = 11 := by
  sorry

end tetradecagon_edge_length_correct_l219_219563


namespace milkshake_cost_l219_219139

-- Definitions for the given conditions
def cheeseburger_cost := 3
def cheese_fries_cost := 8
def jim_money := 20
def cousin_money := 10
def combined_money := jim_money + cousin_money
def percent_spent := 0.80
def total_spent := percent_spent * combined_money

-- Statement to prove that the cost of one milkshake (M) is $5
theorem milkshake_cost :
  ∃ M : ℝ, 2 * M + 2 * cheeseburger_cost + cheese_fries_cost = total_spent ∧ M = 5 :=
by
  -- Defining variables
  let cheeseburger_cost := 3
  let cheese_fries_cost := 8
  let jim_money := 20
  let cousin_money := 10
  let combined_money := jim_money + cousin_money
  let percent_spent := 0.80
  let total_spent := percent_spent * combined_money

  -- The main equation we need to solve
  let eq1 := 2 * cheeseburger_cost + cheese_fries_cost + 2 * M

  -- Hypothesis
  have h : eq1 = total_spent :=
    by 
      sorry

  -- Concluding the proof (to be filled with more steps if needed)
  existsi 5
  split
  • exact h
  • sorry


end milkshake_cost_l219_219139


namespace football_game_attendance_l219_219828

theorem football_game_attendance 
  (x y : ℕ) 
  (h1 : x + y = 280) 
  (h2 : 0.60 * x + 0.25 * y = 140) : 
  x = 200 :=
by 
  sorry

end football_game_attendance_l219_219828


namespace match_duration_l219_219782

theorem match_duration (goals_per_15min : ℕ) (total_goals : ℕ) (duration_goal_avg : ℕ) :
  goals_per_15min = 2 → total_goals = 16 → duration_goal_avg = 15 →
  (total_goals / goals_per_15min) * duration_goal_avg = 120 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end match_duration_l219_219782


namespace sum_of_prime_factors_of_143_l219_219303

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219303


namespace solve_quadratic_inequality_l219_219250

theorem solve_quadratic_inequality (x : ℝ) : (-x^2 - 2 * x + 3 < 0) ↔ (x < -3 ∨ x > 1) := 
sorry

end solve_quadratic_inequality_l219_219250


namespace equation_represents_hyperbola_l219_219469

def equation : Prop := ∀ (x y : ℝ), x^2 - 4y^2 + 6x - 8 = 0 → 
(∃ (a b : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧
∃ (c : ℝ), (x + 3)^2 / a - 4 * y^2 / b = c ∧ c ≠ 0 ∧ c > 0 ∧ a > 0 ∧ b < 0))

theorem equation_represents_hyperbola : equation := by
  sorry

end equation_represents_hyperbola_l219_219469


namespace weight_of_person_replaced_l219_219971

def initial_total_weight (W : ℝ) : ℝ := W
def new_person_weight : ℝ := 137
def average_increase : ℝ := 7.2
def group_size : ℕ := 10

theorem weight_of_person_replaced 
(W : ℝ) 
(weight_replaced : ℝ) 
(h1 : (W / group_size) + average_increase = (W - weight_replaced + new_person_weight) / group_size) : 
weight_replaced = 65 := 
sorry

end weight_of_person_replaced_l219_219971


namespace totalNutsInCar_l219_219644

-- Definitions based on the conditions
def busySquirrelNutsPerDay : Nat := 30
def busySquirrelDays : Nat := 35
def numberOfBusySquirrels : Nat := 2

def lazySquirrelNutsPerDay : Nat := 20
def lazySquirrelDays : Nat := 40
def numberOfLazySquirrels : Nat := 3

def sleepySquirrelNutsPerDay : Nat := 10
def sleepySquirrelDays : Nat := 45
def numberOfSleepySquirrels : Nat := 1

-- Calculate the total number of nuts stored by each type of squirrels
def totalNutsStoredByBusySquirrels : Nat := numberOfBusySquirrels * (busySquirrelNutsPerDay * busySquirrelDays)
def totalNutsStoredByLazySquirrels : Nat := numberOfLazySquirrels * (lazySquirrelNutsPerDay * lazySquirrelDays)
def totalNutsStoredBySleepySquirrel : Nat := numberOfSleepySquirrels * (sleepySquirrelNutsPerDay * sleepySquirrelDays)

-- The final theorem to prove
theorem totalNutsInCar : totalNutsStoredByBusySquirrels + totalNutsStoredByLazySquirrels + totalNutsStoredBySleepySquirrel = 4950 := by
  sorry

end totalNutsInCar_l219_219644


namespace cats_weigh_more_than_puppies_l219_219934

theorem cats_weigh_more_than_puppies :
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  cats_weight - puppies_weight = 5 :=
by
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  show cats_weight - puppies_weight = 5 from sorry

end cats_weigh_more_than_puppies_l219_219934


namespace circle_areas_sum_l219_219259

theorem circle_areas_sum {r s t : ℝ}
  (h1 : r + s = 5)
  (h2 : s + t = 12)
  (h3 : r + t = 13) :
  (Real.pi * r ^ 2 + Real.pi * s ^ 2 + Real.pi * t ^ 2) = 81 * Real.pi :=
by sorry

end circle_areas_sum_l219_219259


namespace median_of_data_set_is_2_l219_219689

theorem median_of_data_set_is_2 : 
  let data := [0, 1, 1, 4, 3, 3] in
  (data.sorted!!(data.length / 2 - 1) + data.sorted!!(data.length / 2)) / 2 = 2 :=
by
  let data := [0, 1, 1, 4, 3, 3]
  sorry

end median_of_data_set_is_2_l219_219689


namespace cats_weigh_more_by_5_kg_l219_219932

def puppies_weight (num_puppies : ℕ) (weight_per_puppy : ℝ) : ℝ :=
  num_puppies * weight_per_puppy

def cats_weight (num_cats : ℕ) (weight_per_cat : ℝ) : ℝ :=
  num_cats * weight_per_cat

theorem cats_weigh_more_by_5_kg :
  puppies_weight 4 7.5  = 30 ∧ cats_weight 14 2.5 = 35 → (cats_weight 14 2.5 - puppies_weight 4 7.5 = 5) := 
by
  intro h
  sorry

end cats_weigh_more_by_5_kg_l219_219932


namespace tangent_point_coordinates_l219_219523

variables (x t : ℝ)

def curve_y (x : ℝ) := Real.exp x

def line_through_point (t : ℝ) : x → ℝ :=
  λ x, Real.exp t * (x - t) + Real.exp t

theorem tangent_point_coordinates :
  (∃ t : ℝ, line_through_point t (-1) = 0 ∧ curve_y t = Real.exp t ∧ (t = 0 → t=0)) →
  (0, 1) = (0, curve_y 0) :=
by
  sorry

end tangent_point_coordinates_l219_219523


namespace quad_area_is_correct_l219_219817

noncomputable def area_of_quad : ℝ :=
  let A := (1, 4)
  let B := (1, 1)
  let C := (3, 1)
  let D := (2008, 2009)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    0.5 * ((Q.1 - P.1) * (R.2 - P.2) - (Q.2 - P.2) * (R.1 - P.1)).abs
  area_triangle A B D + area_triangle B C D

theorem quad_area_is_correct : area_of_quad = 5018.5 := 
by
  -- Proof is omitted
  sorry

end quad_area_is_correct_l219_219817


namespace correct_statements_l219_219780

theorem correct_statements :
  ( -- Statement 1 is correct
    (∀ (l : Type) (A : l) (B : l), A ≠ B → ∃! (P : l), P ∥ A ∧ P ∋ B)
      -- Statement 2 is incorrect
    ∧ ¬ (∀ (A B : ℝ) (C : Prop), (A + B = 180 → C = (A + B = 90)))
      -- Statement 3 is correct
    ∧ (∀ (A B : ℝ), A = B → ∃ C : l, C ≠ (A ∠ B))
      -- Statement 4 is incorrect
    ∧ ¬ (∀ (l1 l2 l3 : Line), l1 ≠ l2 → (l1 ∪ l3 = l2 ∪ l3 → ∀ A B C, ∠ A B C = ∠ l1 l2))
      -- Statement 5 is correct
    ∧ (∀ (a b t : Line), (∃ A B, A ∈ a ∧ B ∈ b ∧ A ∠ B = (∠ a b t)) → a ∥ b)
  ) :=
sorry

end correct_statements_l219_219780


namespace rainstorm_ratio_l219_219713

theorem rainstorm_ratio (x : ℝ) :
  (4 : ℕ) = 4 ∧ -- The rainstorm lasted 4 days
  (72 : ℝ) = 6 * 12 ∧ -- The area can hold 72 inches of rain (6 feet)
  (3 : ℝ) = 3 ∧ -- It can drain out 3 inches of rain per day
  (10 : ℝ) = 10 ∧ -- It rained 10 inches on the first day
  let rain_second_day := 10 * x in -- The second day it rained 10x inches
  let rain_third_day := 1.5 * rain_second_day in -- On the third day, it rained 50% more than the second day
  let rain_drain := 3 * 3 in -- Total drain capacity over the first three days
  let rain_fourth_day := (21 : ℝ) in -- Minimum rain on the fourth day is 21 inches
  (10 + rain_second_day + rain_third_day + rain_fourth_day = 72 + rain_drain) → -- Total rain and drain conditions
  (x = 2) := -- The ratio of the rain on the second day to the rain on the first day is 2:1
by
  sorry

end rainstorm_ratio_l219_219713


namespace committee_count_is_252_l219_219582

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l219_219582


namespace sum_of_cubes_of_consecutive_integers_l219_219866

theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : (n-1)^2 + n^2 + (n+1)^2 = 8450) : 
  (n-1)^3 + n^3 + (n+1)^3 = 446949 := 
sorry

end sum_of_cubes_of_consecutive_integers_l219_219866


namespace distinct_sequences_equal_six_l219_219938

theorem distinct_sequences_equal_six :
  let letters := ['E', 'Q', 'U', 'A', 'L', 'S']
  let sequences := {seq : List Char // seq.head = 'E' ∧ seq.drop 1.head = 'Q' ∧ seq.reverse.head = 'S' ∧ (List.dedup seq).length = seq.length}
  set.to_finset sequences.card = 6 :=
sorry

end distinct_sequences_equal_six_l219_219938


namespace sin_alpha_value_l219_219516

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.cos (2 * α) = 3 / 7) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α < 0) : 
  Real.sin α = sqrt 14 / 7 :=
by
  sorry

end sin_alpha_value_l219_219516


namespace sum_prime_factors_143_l219_219327

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219327


namespace males_listen_l219_219706

theorem males_listen (males_dont_listen females_listen total_listen total_dont_listen : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listen = 180)
  (h4 : total_dont_listen = 120) :
  ∃ m, m = 105 :=
by {
  sorry
}

end males_listen_l219_219706


namespace triangle_AB_length_l219_219129

variable {A B C : Type}
variables {a b : ℝ}  -- lengths of sides BC and AC

-- condition: AC = b and BC = a
variable (AC_eq_b : ∥A - C∥ = b)
variable (BC_eq_a : ∥B - C∥ = a)

-- condition: The medians drawn to AC and BC intersect at a right angle.
variable (medians_right_angle : 
  let M := (A + C) / 2 
  let N := (B + C) / 2 in
  ∥A + B - C - N - M∥ = ∥0∥)

-- theorem: length of AB
theorem triangle_AB_length (AC_eq_b : ∥A - C∥ = b) (BC_eq_a : ∥B - C∥ = a) (medians_right_angle : 
  let M := (A + C) / 2 
  let N := (B + C) / 2 in
  ∥A + B - C - N - M∥ = ∥0∥) : 
  ∥A - B∥ = Real.sqrt ((a^2 + b^2) / 5) :=
sorry

end triangle_AB_length_l219_219129


namespace find_angle_between_vectors_l219_219059

variables (a b : ℝ^3)
variables (theta : ℝ)

-- Given conditions
def condition1 : Prop := ‖a‖ = 1
def condition2 : Prop := ‖b‖ = 2
def condition3 : Prop := (a + b) ⬝ (2 • a - b) = -1

-- Prove the angle between vectors a and b
theorem find_angle_between_vectors
  (h1 : condition1 a)
  (h2 : condition2 b)
  (h3 : condition3 a b)
  : theta = π / 3 :=
sorry

end find_angle_between_vectors_l219_219059


namespace monotonic_subsequence_exists_l219_219048

theorem monotonic_subsequence_exists (n : ℕ) (a : Fin ((2^n : ℕ) + 1) → ℕ)
  (h : ∀ k : Fin (2^n + 1), a k ≤ k.val) : 
  ∃ (b : Fin (n + 2) → Fin (2^n + 1)),
    (∀ i j : Fin (n + 2), i ≤ j → b i ≤ b j) ∧
    (∀ i j : Fin (n + 2), i < j → a (b i) ≤ a ( b j)) :=
by
  sorry

end monotonic_subsequence_exists_l219_219048


namespace min_reciprocal_sum_l219_219158

theorem min_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy_sum : x + y = 12) (hxy_neq : x ≠ y) : 
  ∃ c : ℝ, c = 1 / 3 ∧ (1 / x + 1 / y ≥ c) :=
sorry

end min_reciprocal_sum_l219_219158


namespace sum_prime_factors_of_143_l219_219346

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219346


namespace coordinates_in_new_basis_l219_219910

variables (V : Type*) [AddCommGroup V] [Module ℚ V]
variables (a b c p : V)
variables (h_basis1 : LinearIndependent ℚ ![a, b, c])
variables (h_basis2 : LinearIndependent ℚ ![a - b, a + b, c])
variables (h_p : p = 4 • a + 2 • b - 1 • c)

theorem coordinates_in_new_basis :
  ∃ (x y z : ℚ), p = x • (a - b) + y • (a + b) + z • c ∧ x = 1 ∧ y = 3 ∧ z = -1 :=
begin
  sorry
end

end coordinates_in_new_basis_l219_219910


namespace arithmetic_sequence_r_value_l219_219591

theorem arithmetic_sequence_r_value :
  ∀ (p q s : ℝ), 
  let seq := [12, p, q, r, s, 47],
  let avg := (12 + 47) / 2 in 
  (seq[3] = avg) → r = 29.5 :=
begin
  intro p,
  intro q,
  intro s,
  let seq := [12, p, q, r, s, 47],
  let avg := (12 + 47) / 2,
  intro h_avg,
  have h_correct : r = avg := h_avg,
  simp [h_correct, avg],
  sorry
end

end arithmetic_sequence_r_value_l219_219591


namespace circle_chairs_subsets_count_l219_219737

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219737


namespace problem_l219_219098

def f1 (x: ℝ) : ℝ := -x^2 + 3 * x

def f2 (x: ℝ) : ℝ := x^2 + x + 1

def f3 (x: ℝ) : ℝ := x + 1 / (x + 1)

def f4 (x: ℝ) : ℝ :=
  if (0 < x ∧ x <= 1) then x
  else if (1 < x ∧ x <= 2) then 1 / x
  else 0

theorem problem (x : ℝ) (h : 0 < x ∧ x <= 2) :
  (f1(x) > f1(0) ∨ f4(x) > f4(0)) ∧ ¬(f2(x) > f2(0) ∧ f3(x) > f3(0)) :=
by
  sorry

end problem_l219_219098


namespace fair_bet_l219_219652

def hit_probabilities : List (ℕ × ℚ) := [(10, 1/5), (9, 1/4), (8, 1/6), (7, 1/8), (6, 1/10)]
def other_hit_probability : ℚ := 1/12
def below_six_payment : ℚ := 1.7
def x : ℚ := 96

noncomputable def total_probability_of_hits : ℚ :=
  (1/5) + (1/4) + (1/6) + (1/8) + (1/10) + 6 * (1/12)

theorem fair_bet : 
  let miss_probability := 1 - total_probability_of_hits in
  let expected_value := 
    (1/5 * 10) + (1/4 * 9) + (1/6 * 8) + (1/8 * 7) + (1/10 * 6) + 6 * (1/12 * below_six_payment) - (miss_probability * x) in
  miss_probability = 3/40 ∧ expected_value = 0 :=
by 
  sorry

end fair_bet_l219_219652


namespace AC_equals_CE_l219_219622

noncomputable def geometric_problem (A B C D E : Type) [linear_ordered_field A] [metric_space A]
  [normed_group B] [normed_space ℝ B] [inheritance.linear_ordered_field B]
  [complete_space B] [normed_Euclidean_space B] :=
  let T := affine.mk (finset.empty : finset A) in
  exists (RightAngledTriangleABC : T)
    (pointD : T)
    (perpendicularDE : T) in
  (RightAngledTriangleABC.isRightAngleAtC) ∧
  (C ∠ ABC = 90°) ∧
  (segmentAC.contains D) ∧
  (segmentCD.isEqualTo segmentBC) ∧
  (segmentDE.isPerpendicularTo segmentAB) ∧
  (segmentDE.intersects BC E) ∧
  (segmentAC.equals segmentCE)

axiom RightAngledTriangleABC : A → A → Prop
axiom pointD : A → Prop
axiom perpendicularDE : A → Prop
axiom contains : A → A → Prop
axiom isEqualTo : A → A → Prop
axiom isPerpendicularTo : A → A → Prop
axiom intersects : A → A → A → Prop
axiom equals : A → A → Prop

theorem AC_equals_CE : geometric_problem A B C D E := sorry

end AC_equals_CE_l219_219622


namespace arithmetic_sequence_S_2015_l219_219049

theorem arithmetic_sequence_S_2015 : ∀ {a_1 a_5 : ℤ} (h1 : a_1 = tan 225) (h2 : a_5 = 13 * a_1), 
  S_2015 (-1)^2015 a = -3022 :=
by
  sorry

end arithmetic_sequence_S_2015_l219_219049


namespace probability_A_not_losing_l219_219366

theorem probability_A_not_losing (P_draw P_win : ℚ) (h1 : P_draw = 1/2) (h2 : P_win = 1/3) : 
  P_draw + P_win = 5/6 :=
by
  sorry

end probability_A_not_losing_l219_219366


namespace sum_of_prime_factors_of_143_l219_219302

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219302


namespace junior_score_l219_219566

theorem junior_score (total_students : ℕ) (juniors_percentage : ℝ) (seniors_percentage : ℝ)
  (class_average : ℝ) (senior_average : ℝ) (juniors_same_score : Prop) 
  (h1 : juniors_percentage = 0.2) (h2 : seniors_percentage = 0.8)
  (h3 : class_average = 85) (h4 : senior_average = 84) : 
  ∃ junior_score : ℝ, juniors_same_score → junior_score = 89 :=
by
  sorry

end junior_score_l219_219566


namespace distance_between_skew_lines_is_sqrt_6_l219_219547

noncomputable def distance_between_lines (l m : Line) (A B C : Point) (D E F : Point) (AB BC : ℝ) 
  (AD BE CF : ℝ) : ℝ :=
  if h₁ : AB = BC
     ∧ (∃ (AD_BE_CF_eqs : AD = sqrt 15 ∧ BE = 7/2 ∧ CF = sqrt 10),
  distance l m = sqrt 6
else sorry

theorem distance_between_skew_lines_is_sqrt_6
  (l m : Line) (A B C : Point) (D E F : Point)
  (AB BC : ℝ) (AD BE CF : ℝ)
  (H1 : AB = BC)
  (H2 : AD = sqrt 15)
  (H3 : BE = 7/2)
  (H4 : CF = sqrt 10) :
  distance_between_lines l m A B C D E F AB BC AD BE CF = sqrt 6 := sorry

end distance_between_skew_lines_is_sqrt_6_l219_219547


namespace each_person_pays_l219_219997

def cost_first_tier : ℝ := 100
def cost_second_tier : ℝ := 75
def cost_third_tier : ℝ := cost_second_tier / 2
def cost_fourth_tier : ℝ := cost_third_tier * 1.25

def total_cost : ℝ := cost_first_tier + cost_second_tier + (cost_third_tier * 1.5) + (cost_fourth_tier * 2)
def cost_per_person : ℝ := total_cost / 4

theorem each_person_pays : cost_per_person = 81.25 :=
by
  sorry

end each_person_pays_l219_219997


namespace find_a_n_find_sum_first_n_terms_l219_219908

variable {a_n : ℕ → ℕ} -- Define the sequence as a function from ℕ to ℕ

-- Define the conditions given in the problem.
def arithmetic_sequence (a_n : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a_n = a_1 + (n - 1) * d

def sum_first_n_terms (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) : Prop :=
  ∀ n, S_n n = n * (a_n 1 + a_n n) / 2

-- The given conditions
def conditions : Prop :=
  ∃ a_n S_n, arithmetic_sequence a_n ∧ a_n 3 = 6 ∧ S_n 3 = 12

-- The goals to prove
theorem find_a_n (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :
  conditions →
  ∀ n, a_n n = 2 * n :=
sorry

theorem find_sum_first_n_terms (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :
  conditions →
  ∀ n, S_n n = n * (n + 1) :=
sorry

end find_a_n_find_sum_first_n_terms_l219_219908


namespace B_work_rate_l219_219400

theorem B_work_rate :
  let A := (1 : ℝ) / 8
  let C := (1 : ℝ) / 4.8
  (A + B + C = 1 / 2) → (B = 1 / 6) :=
by
  intro h
  let A : ℝ := 1 / 8
  let C : ℝ := 1 / 4.8
  let B : ℝ := 1 / 6
  sorry

end B_work_rate_l219_219400


namespace anna_reading_time_l219_219441

theorem anna_reading_time:
  (∀ n : ℕ, n ∈ (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1) → True) →
  (let chapters_read := (Finset.range 31).filter (λ x, ¬ (∃ k : ℕ, k * 3 + 3 = x + 1)).card,
  reading_time := chapters_read * 20,
  hours := reading_time / 60 in
  hours = 7) :=
by
  intros
  let chapters_read := (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1).card
  have h1 : chapters_read = 21 := by sorry
  let reading_time := chapters_read * 20
  have h2 : reading_time = 420 := by sorry
  let hours := reading_time / 60
  have h3 : hours = 7 := by sorry
  exact h3

end anna_reading_time_l219_219441


namespace minimum_a_condition_l219_219039

def piecewise_function (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

theorem minimum_a_condition :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, piecewise_function (x + 4) ≥ piecewise_function (2 * 4 - x)) →
  (∀ a : ℝ, (∀ x ∈ Set.Icc (-2 : ℝ) 2, piecewise_function (x + a) ≥ piecewise_function (2 * a - x)) → a ≥ 4) :=
begin
  -- the rest of the proof would go here
  sorry
end

end minimum_a_condition_l219_219039


namespace proof_problem_l219_219537

theorem proof_problem (t : ℝ) (ρ θ : ℝ) : 
  (∀ t, x = 2 + t ∧ y = t) ∧ (ρ^2 * cos θ ^ 2 + 9 * ρ^2 * sin θ ^ 2 = 9) 
  ∧ ∃ (A B E : ℝ × ℝ), 
      (A ∈ line_C1 ∧ A ∈ ellipse_C2) ∧ 
      (B ∈ line_C1 ∧ B ∈ ellipse_C2) ∧ 
      (E ∈ line_C1 ∧ E.x = 2 ∧ E.y = 0) → 
  |EA + EB| = (6 * sqrt 3) / 5 :=
sorry

end proof_problem_l219_219537


namespace sum_prime_factors_143_l219_219361

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219361


namespace midpoint_arc_equidistant_l219_219253

-- Definition of a triangle inscribed in a circle with given constraints
theorem midpoint_arc_equidistant (A B C M N K P Q S : Point)
  (Ω : Circle) (inscribed : InscribedTriangle Ω A B C)
  (hAB_gt_BC : dist A B > dist B C)
  (hAM_eq_CN : dist A M = dist C N)
  (hMN_AC_intersect : Intersect (Line M N) (Line A C) K)
  (hP_incenter_AMK : IncenterTriangle P (Triangle A M K))
  (hQ_excenter_CNK : ExcenterTriangle Q (Triangle C N K))
  (hS_arc_midpoint : ArcMidpoint S A B C Ω) :
  dist S P = dist S Q := 
by sorry

end midpoint_arc_equidistant_l219_219253


namespace max_value_of_f_l219_219688

noncomputable def f (x : ℝ) : ℝ :=
  (1 + Real.cos x)^8 + (1 - Real.cos x)^8

theorem max_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f y ≤ 256 :=
begin
  use 0,  -- starting point for simplification, this is a placeholder
  intro y,
  sorry
end

end max_value_of_f_l219_219688


namespace expression_for_u16_l219_219463

noncomputable def sequence (a : ℝ) (u : ℕ → ℝ) : Prop :=
  u 1 = 2 * a ∧ ∀ n, u (n + 1) = - (2 * a) / (u n + 2 * a)

theorem expression_for_u16 (a : ℝ) (u : ℕ → ℝ) (h : sequence a u) (ha : a > 0) : u 16 = 2 * a :=
sorry

end expression_for_u16_l219_219463


namespace total_games_in_season_l219_219419

variables {teams_per_division : ℕ} (num_divisions : ℕ) [fact (num_divisions = 2)]
variables (games_within_division games_across_division : ℕ)
variables (total_teams : ℕ) [fact (total_teams = teams_per_division * num_divisions)]
variables [fact (teams_per_division = 8)]
variables [fact (games_within_division = 3)]
variables [fact (games_across_division = 1)]

/- The statement -/
theorem total_games_in_season
  (num_divisions = 2)
  (teams_per_division = 8)
  (games_within_division = 3)
  (games_across_division = 1)
  (total_teams = 16) :
  let total_games_for_each_team := (teams_per_division - 1) * games_within_division + teams_per_division in
  let calculated_total := total_teams * total_games_for_each_team / 2 in
  calculated_total = 232 :=
by
  sorry

end total_games_in_season_l219_219419


namespace sum_of_prime_factors_of_143_l219_219336

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219336


namespace event_X_eq_6_characterization_l219_219804

-- Definitions based on the problem statement
def bag : set ℕ := {1, 2, 3, 4, 5, 6}
def draw_subset (s : set ℕ) (n : ℕ) : Prop := s ⊆ bag ∧ s.card = n

-- Definition of the highest number in a subset
def highest (s : set ℕ) := s.sup id

-- Event X = 6 means the highest number drawn is 6
def event_X_eq_6 (s : set ℕ) : Prop := highest(s) = 6

-- The condition described in the solution
def condition_to_prove (s : set ℕ) : Prop :=
  ∃ (t : set ℕ), t = s ∧ 6 ∈ t ∧ (t.erase 6) ⊆ {1, 2, 3, 4, 5} ∧ (t.erase 6).card = 2

-- The proof problem statement
theorem event_X_eq_6_characterization (s : set ℕ) (h1 : draw_subset s 3) :
  event_X_eq_6 s ↔ condition_to_prove s :=
by {
  sorry
}

end event_X_eq_6_characterization_l219_219804


namespace line_circle_intersection_l219_219686

-- Define the condition for the line equation given constants a and b
def line (a b : ℝ) : set (ℝ × ℝ) := 
  { p | ∃ x y, p = (x, y) ∧ (a + 2 * b) * x + (b - a) * y + ( a - b ) = 0 }

-- Define the condition for the circle equation given constant m
def circle (m : ℝ) : set (ℝ × ℝ) := 
  { p | ∃ x y, p = (x, y) ∧ x^2 + y^2 = m }

-- Prove that the line and circle intersect if and only if m >= 1/2.
theorem line_circle_intersection (a b m : ℝ) : 
  (∃ x y, ((a + 2 * b) * x + (b - a) * y + (a - b) = 0 ∧ x^2 + y^2 = m)) ↔ m ≥ (1 / 2) := 
sorry

end line_circle_intersection_l219_219686


namespace bug_tetrahedron_prob_l219_219399

/--
 A tetrahedron has 4 vertices and 6 edges. A bug starts at one vertex 
 and moves along the edges. At each vertex, it randomly chooses one of 
 the two edges it has not come from, with equal probability. Prove that 
 the probability that the bug will have visited every vertex exactly 
 once after three moves is 1/2.
-/
theorem bug_tetrahedron_prob :
  let vertices := 4 in 
  let edges := 6 in
  let start_vertex := 1 in 
  let moves := 3 in 
  let favorable_paths := 24 in
  let possible_paths := 48 in
  (favorable_paths / possible_paths : ℚ) = 1 / 2 :=
by
  sorry

end bug_tetrahedron_prob_l219_219399


namespace sum_of_valid_integers_l219_219958

def inequality_system (x a : ℝ) : Prop :=
  (5 * x - a) / 3 - x < 3 ∧ 3 * x < 2 * x + 1

def fractional_equation (y a: ℝ) : Prop :=
  y > 0 ∧ y ≠ 1 ∧ (3 * y + a) / (y - 1) - 1 = 2 * a / (1 - y)

noncomputable def valid_integers (a : ℝ) : Prop :=
  ∃ y, fractional_equation y a ∧ ∃ x, inequality_system x a ∧ x < 1

theorem sum_of_valid_integers :
  (∑ a in (finset.range 15).filter (λ a, valid_integers (-(a:ℝ) - 1)), -((a:ℝ) + 1)) = -15 :=
by sorry

end sum_of_valid_integers_l219_219958


namespace larry_gave_86_candies_l219_219829

def start : ℕ := 5
def end : ℕ := 91
def candies_from_Larry : ℕ := end - start

theorem larry_gave_86_candies : candies_from_Larry = 86 := by
  -- This is where the proof would go, but we leave it as a sorry for now.
  sorry

end larry_gave_86_candies_l219_219829


namespace vertex_on_y_axis_passes_through_fixed_points_root_conditions_l219_219919

-- Definitions
def quadratic_func (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x - 4

-- (1) Prove a = -1 given the vertex of quadratic_func is on the y-axis.
theorem vertex_on_y_axis (a : ℝ) : quadratic_func a x = 0 → a = -1 :=
sorry

-- (2) Prove that the function passes through the points (0, -4) and (1, -5) for any value of a.
theorem passes_through_fixed_points (a : ℝ) :
  quadratic_func a 0 = -4 ∧ quadratic_func a 1 = -5 :=
sorry

-- (3) Prove integer value of a = 2 given root conditions.
theorem root_conditions (a : ℝ) :
  (∃ a ∈ Ico (-1 : ℝ) 0, quadratic_func a a = 0) ∧ (∃ a ∈ Ico (2 : ℝ) 3, quadratic_func a a = 0) → a = 2 :=
sorry

end vertex_on_y_axis_passes_through_fixed_points_root_conditions_l219_219919


namespace desired_interest_rate_l219_219409

def face_value : Real := 52
def dividend_rate : Real := 0.09
def market_value : Real := 39

theorem desired_interest_rate : (dividend_rate * face_value / market_value) * 100 = 12 := by
  sorry

end desired_interest_rate_l219_219409


namespace points_on_parabola_order_l219_219881

variable {m : ℝ} (h : m < -2)
def y1 := (m - 1)^2 + 2 * (m - 1)
def y2 := m^2 + 2 * m
def y3 := (m + 1)^2 + 2 * (m + 1)

theorem points_on_parabola_order : y3 < y2 < y1 :=
by
  sorry

end points_on_parabola_order_l219_219881


namespace no_arithmetic_progression_l219_219199

theorem no_arithmetic_progression {n k : ℕ} (hn : 0 < n) (hk : 0 < k) (hkn : k ≤ n) :
  ¬(∃ d : ℤ, ∀ i : ℕ, i ∈ Finset.range 3 → (binom n (k + (i + 1)) : ℤ) - (binom n (k + i) : ℤ) = d) := by
  sorry

end no_arithmetic_progression_l219_219199


namespace total_snakes_count_l219_219175

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l219_219175


namespace students_voted_for_meat_l219_219975

theorem students_voted_for_meat (total_votes veggies_votes : ℕ) (h_total: total_votes = 672) (h_veggies: veggies_votes = 337) :
  total_votes - veggies_votes = 335 := 
by
  -- Proof steps go here
  sorry

end students_voted_for_meat_l219_219975


namespace probability_three_numbers_divides_l219_219714

open Finset

noncomputable def probability_divides (s : Finset ℕ) : ℚ :=
let combs := s.powerset.filter (λ t, t.card = 3) in
let valid_combs := combs.filter (λ t, ∃ a b, a ∈ t ∧ b ∈ t ∧ a ≠ b ∧ a ∣ b) in
(valid_combs.card : ℚ) / combs.card

theorem probability_three_numbers_divides :
  probability_divides ({1, 2, 3, 4, 6} : Finset ℕ) = 9 / 10 :=
by 
s sorry

end probability_three_numbers_divides_l219_219714


namespace exists_x0_f_eq_neg_f_l219_219206

noncomputable def f : ℝ → ℝ := sorry

axiom f_twice_continuously_differentiable :
  ∀ x : ℝ, differentiable ℝ (deriv (deriv f))

axiom f_bound :
  ∀ x : ℝ, |f x| ≤ 1

axiom f_initial_condition :
  (f 0)^2 + (deriv f 0)^2 = 4

theorem exists_x0_f_eq_neg_f'' :
  ∃ (x0 : ℝ), f x0 + deriv (deriv f) x0 = 0 :=
by
  sorry

end exists_x0_f_eq_neg_f_l219_219206


namespace four_digit_number_count_l219_219230

theorem four_digit_number_count : 
  let digits := {0, 1, 2} in
  (∀ n : ℕ, n ∈ digits → n < 10) ∧ 
  (∃ d : ℕ, d ∈ digits ∧ ∀ n : ℕ, (n ∈ digits \ {d} → n ≠ d)) ∧ 
  (∀ a b c : ℕ, a ∈ digits ∧ b ∈ digits ∧ c ∈ digits → a ≠ 0) →
  (card {x : fin 10 | (∃ a b c d: ℕ, 
    (a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits) ∧ 
    (x = 1000 * a + 100 * b + 10 * c + d) ∧ 
    ((a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d) ∧ 
    ((a, b, c, d).nodup) )})) = 20 :=
by
  sorry

end four_digit_number_count_l219_219230


namespace max_intersections_l219_219273

theorem max_intersections (n1 n2 k : ℕ) 
  (h1 : n1 ≤ n2)
  (h2 : k ≤ n1) : 
  ∃ max_intersections : ℕ, 
  max_intersections = k * n2 :=
by
  sorry

end max_intersections_l219_219273


namespace max_donation_proof_l219_219391

noncomputable def max_donation (tutoring_cost_per_hour : ℕ) (investment_income : ℕ)
                              (working_days_per_month : ℕ) (sleep_hours : ℕ) 
                              (monthly_expenses : ℕ) 
                              (donation_per_rest_hour : ℕ) : ℕ :=
  let L_i := (16 - k) / 3 in  -- Work hours per day
  let total_hours_rest := λ k * working_days_per_month in -- Total rest hours per month
  let total_donation := total_hours_rest / donation_per_rest_hour in -- Max donation per month
  total_donation

theorem max_donation_proof : max_donation 3 14 21 8 70 3 = 70 := by
  sorry

end max_donation_proof_l219_219391


namespace sum_of_prime_factors_of_143_l219_219334

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219334


namespace chairs_adjacent_subsets_l219_219732

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219732


namespace trajectory_equation_l219_219504

noncomputable def point_trajectory (x y : ℝ) (x0 y0 : ℝ) : Prop :=
  x0 = -((sqrt 2 + 1) / sqrt 2) * x ∧
  y0 = (sqrt 2 + 1) * y ∧
  x0^2 + y0^2 = (sqrt 2 + 1)^2

theorem trajectory_equation (x y : ℝ) (x0 y0 : ℝ) :
  point_trajectory x y x0 y0 → (x^2 / 2 + y^2 = 1) :=
by
  intros h
  sorry

end trajectory_equation_l219_219504


namespace altitude_small_cone_l219_219810

noncomputable def frustum_altitude := 30
noncomputable def lower_base_area := 324 * Real.pi
noncomputable def upper_base_area := 36 * Real.pi

theorem altitude_small_cone:
    (H : ℝ) (x : ℝ) (r_l : ℝ) (r_u : ℝ)
    (hl: lower_base_area = Real.pi * r_l^2)
    (hu: upper_base_area = Real.pi * r_u^2)
    (frac_height_ratio : r_u / r_l = 1 / 3)
    (H_rel: frustum_altitude = 2 / 3 * H) :
    x = 1 / 3 * H :=
begin
    sorry
end

end altitude_small_cone_l219_219810


namespace max_value_S_n_l219_219121

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- The conditions given in the problem
def conditions (d : ℤ) :=
  a_n 3 + a_n 10 = 5 ∧
  a_n 7 = 1 ∧
  (∀ n, a_n n = a_n 1 + (n - 1) * d) ∧
  S_n = λ n, n * a_n 1 + (n * (n - 1) / 2) * d

-- Proving the maximum value of Sn
theorem max_value_S_n : ∀ d : ℤ, conditions a_n S_n d → ∃ n, S_n n = 70 :=
by
  intro d h
  have h1 := h.1
  have h2 := h.2
  have h3 := h.3
  have h4 := h.4
  -- We need to solve using the given conditions to find S_7 = 70
  sorry

end max_value_S_n_l219_219121


namespace ice_cream_eaten_on_friday_l219_219434

theorem ice_cream_eaten_on_friday
  (x : ℝ) -- the amount eaten on Friday night
  (saturday_night : ℝ) -- the amount eaten on Saturday night
  (total : ℝ) -- the total amount eaten
  
  (h1 : saturday_night = 0.25)
  (h2 : total = 3.5)
  (h3 : x + saturday_night = total) : x = 3.25 :=
by
  sorry

end ice_cream_eaten_on_friday_l219_219434


namespace john_annual_profit_is_1800_l219_219141

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l219_219141


namespace findKForCollinearABD_l219_219921

variable (e1 e2 : ℝ → ℝ → ℝ)
variable (A B D : ℝ → ℝ → ℝ)
variable (k : ℝ)

-- Conditions
def AB := e1 1 0 - k * e2 0 1
def CB := e1 2 0 - e2 1 0
def CD := e1 3 0 - e2 3 0
def BD := CD - CB

-- Collinearity of AB and BD
def collinear (A B D : (ℝ → ℝ → ℝ)) : Prop :=
  ∃ λ : ℝ, AB = λ • BD

-- Proof problem
theorem findKForCollinearABD (k : ℝ) :
  collinear A B D ↔ k = 2 :=
by
  sorry

end findKForCollinearABD_l219_219921


namespace general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l219_219544

-- Defines the sequences and properties given in the problem
def sequences (a_n b_n S_n T_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ S_n 2 = 4 ∧ 
  (∀ n : ℕ, 3 * S_n (n + 1) = 2 * S_n n + S_n (n + 2) + a_n n)

-- (1) Prove the general formula for {a_n}
theorem general_formula_for_a_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- (2) If {b_n} is an arithmetic sequence and ∀n ∈ ℕ, S_n > T_n, prove a_n > b_n
theorem a_n_greater_than_b_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (arithmetic_b : ∃ d: ℕ, ∀ n: ℕ, b_n n = b_n 0 + n * d)
  (Sn_greater_Tn : ∀ (n : ℕ), S_n n > T_n n) :
  ∀ n : ℕ, a_n n > b_n n :=
sorry

-- (3) If {b_n} is a geometric sequence, find n such that (a_n + 2 * T_n) / (b_n + 2 * S_n) = a_k
theorem find_n_in_geometric_sequence
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (geometric_b : ∃ r: ℕ, ∀ n: ℕ, b_n n = b_n 0 * r^n)
  (b1_eq_1 : b_n 1 = 1)
  (b2_eq_3 : b_n 2 = 3)
  (k : ℕ) :
  ∃ n : ℕ, (a_n n + 2 * T_n n) / (b_n n + 2 * S_n n) = a_n k := 
sorry

end general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l219_219544


namespace seq_a22_l219_219244

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l219_219244


namespace evaluate_propositions_l219_219116

variables {ℝ : Type*} [inner_product_space ℝ ℝ]

-- Definitions of vectors a and b
variables (a b : ℝ)

-- Original proposition: if the dot product is zero, then l is parallel to alpha
def original_proposition : Prop := (inner a b = 0) → (a ≠ 0 → ¬∃ k : ℝ, a = k * b)

-- Converse of the original proposition: if l is parallel to alpha, then the dot product is zero
def converse_proposition : Prop := (∃ k : ℝ, a = k * b) → (inner a b = 0)

-- Statement encoding the correctness of the propositions
theorem evaluate_propositions : (¬original_proposition) ∧ converse_proposition :=
by
  sorry

end evaluate_propositions_l219_219116


namespace complement_union_A_B_l219_219920

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 5}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}
def U : Set ℝ := A ∪ B
def R : Set ℝ := univ

theorem complement_union_A_B : (R \ U) = {x | -2 < x ∧ x ≤ -1} :=
by
  sorry

end complement_union_A_B_l219_219920


namespace number_of_ways_to_play_ads_l219_219209

-- Definitions for the problem
def n_ads : ℕ := 5
def n_commercial_ads : ℕ := 3
def n_olympic_ads : ℕ := 2

-- Conditions for the problem
def last_ad_is_olympic (seq : list ℕ) : Prop :=
  seq.length = n_ads ∧ seq.last' = some n_olympic_ads

def olympic_ads_not_consecutive (seq : list ℕ) : Prop :=
  ∀ i, 0 < i < seq.length - 1 → (seq.get i = some n_olympic_ads → seq.get (i + 1) ≠ some n_olympic_ads)

-- The final proof statement
theorem number_of_ways_to_play_ads : 
  ∃ (seq : list ℕ), seq.length = n_ads ∧ 
    last_ad_is_olympic seq ∧ 
    olympic_ads_not_consecutive seq ∧ 
    (number_of_possible_ways seq = 36) :=
sorry

end number_of_ways_to_play_ads_l219_219209


namespace correct_choice_l219_219210

theorem correct_choice
  (options : List String)
  (correct : String)
  (is_correct : correct = "that") :
  "The English spoken in the United States is only slightly different from ____ spoken in England." = 
  "The English spoken in the United States is only slightly different from that spoken in England." :=
by
  sorry

end correct_choice_l219_219210


namespace quadratic_function_properties_l219_219462

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem quadratic_function_properties :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f x ≤ f y) ∧
  (∃ x : ℝ, x = 1.5 ∧ ∀ y : ℝ, f x ≤ f y) :=
by
  sorry

end quadratic_function_properties_l219_219462


namespace sum_of_areas_of_circles_l219_219264

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l219_219264


namespace solution_set_l219_219062

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (h1 : ∀ x ∈ ℝ, 2 * f' x > f x)

theorem solution_set (x : ℝ) (hx : e ^ ((x - 1) / 2) * f x < f (2 * x - 1)) : x > 1 :=
sorry

end solution_set_l219_219062


namespace subsets_with_at_least_four_adjacent_chairs_l219_219726

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219726


namespace arithmetic_sequence_problem_l219_219986

variable {α : Type*} [LinearOrderedRing α]

theorem arithmetic_sequence_problem
  (a : ℕ → α)
  (h : ∀ n, a (n + 1) = a n + (a 1 - a 0))
  (h_seq : a 5 + a 6 + a 7 + a 8 + a 9 = 450) :
  a 3 + a 11 = 180 :=
sorry

end arithmetic_sequence_problem_l219_219986


namespace modulus_of_complex_eq_i_l219_219167

theorem modulus_of_complex_eq_i (z : ℂ) (h : (1 + z) / (1 - z) = complex.I) : complex.abs z = 1 :=
sorry

end modulus_of_complex_eq_i_l219_219167


namespace simplify_fraction_result_l219_219203

theorem simplify_fraction_result : (130 / 16900) * 65 = 1 / 2 :=
by sorry

end simplify_fraction_result_l219_219203


namespace diagonal_lt_half_perimeter_l219_219194

theorem diagonal_lt_half_perimeter (AB BC CD DA AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0) 
  (h_triangle1 : AC < AB + BC) (h_triangle2 : AC < AD + DC) :
  AC < (AB + BC + CD + DA) / 2 :=
by {
  sorry
}

end diagonal_lt_half_perimeter_l219_219194


namespace max_X_placement_l219_219845

-- Define the condition for a valid placement of X's
def valid_placement (X : ℕ × ℕ → Prop) : Prop :=
  ∀ (r1 r2 r3 r4 c1 c2 c3 c4 : ℕ),
  (r1 < 3 ∧ r2 < 3 ∧ r3 < 3 ∧ r4 < 3 ∧
   c1 < 5 ∧ c2 < 5 ∧ c3 < 5 ∧ c4 < 5) →
  (X (r1, c1) ∧ X (r2, c2) ∧ X (r3, c3) ∧ X (r4, c4)) →
  ((r1 ≠ r2 ∨ r2 ≠ r3 ∨ r3 ≠ r4) ∧
   (c1 ≠ c2 ∨ c2 ≠ c3 ∨ c3 ≠ c4) ∧
   (r1 - c1 ≠ r2 - c2 ∨ r2 - c2 ≠ r3 - c3 ∨ r3 - c3 ≠ r4 - c4) ∧
   (r1 + c1 ≠ r2 + c2 ∨ r2 + c2 ≠ r3 + c3 ∨ r3 + c3 ≠ r4 + c4))

-- Define the maximum number of X's such that no four align
def max_X (X : ℕ × ℕ → Prop) : ℕ :=
  ⟦Σ (i : fin 3) (j : fin 5), if X (i, j) then 1 else 0⟧

theorem max_X_placement (X : ℕ × ℕ → Prop) :
  valid_placement X → max_X X ≤ 9 := 
by
  sorry

end max_X_placement_l219_219845


namespace projections_distance_eq6_l219_219546

-- Definitions based on given conditions
variable (P Q : Type) [MetricSpace P] [MetricSpace Q]

-- Given two parallel lines
variable (line1 line2 : Set P)
variable (M : P) -- Point M which is 3 units away from one of the lines
variable (dist_lines : ℝ) (dist_M : ℝ)
variable (circle : Set P) (center O: P) (radius : ℝ)

-- Given conditions encoded
axiom parallel_lines (lines_parallel : Parallel line1 line2) 
axiom lines_distance (lines_15units : dist_lines = 15)
axiom point_M_props (M_condition16: dist M line1 = 3)

axiom circle_tangent (tangent_to_lines : Tangent circle line1 ∧ Tangent circle line2)
axiom circle_throughM (circle_through_M : M ∈ circle)
axiom circle_center_radius (circle_center_dist : center ∈ circle ∧ radius = 15 / 2)

-- Proof statement
theorem projections_distance_eq6 : 
  distance_proj_between_center_M = 6 :=
by
  sorry

end projections_distance_eq6_l219_219546


namespace pi_minus_five_floor_value_l219_219223

noncomputable def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem pi_minus_five_floor_value :
  greatest_integer_function (Real.pi - 5) = -2 :=
by
  -- The proof is omitted
  sorry

end pi_minus_five_floor_value_l219_219223


namespace count_divisible_digits_l219_219873

def is_divisible (a b : ℕ) : Prop := b ≠ 0 ∧ a % b = 0

theorem count_divisible_digits :
  ∃! (s : Finset ℕ), s = {n | n ∈ Finset.range 10 ∧ n ≠ 0 ∧ is_divisible (25 * n) n} ∧ (Finset.card s = 3) := 
by
  sorry

end count_divisible_digits_l219_219873


namespace z_2021_value_l219_219528

noncomputable def z_sequence : ℕ → ℂ
| 0       := ⟨ (real.sqrt 3) / 2, 0 ⟩
| (n + 1) := conj (z_sequence n) * (1 + z_sequence n * complex.I)

theorem z_2021_value :
  z_sequence 2020 = ⟨(real.sqrt 3) / 2, (1 / 2 + 1 / 2 ^ 2020)⟩ :=
sorry

end z_2021_value_l219_219528


namespace complex_number_solution_l219_219887

open Complex

noncomputable def sqrt {α : Type*} [linear_ordered_semiring α] (n : α) : α := classical.some (exists_sqrt n)

theorem complex_number_solution (z : ℂ) (h : (1 + I) * z = abs (√3 + I)) : z = 1 - I := 
by sorry

end complex_number_solution_l219_219887


namespace value_of_f_g3_l219_219945

def g (x : ℝ) : ℝ := 4 * x - 5
def f (x : ℝ) : ℝ := 6 * x + 11

theorem value_of_f_g3 : f (g 3) = 53 := by
  sorry

end value_of_f_g3_l219_219945


namespace compute_expression_l219_219456

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end compute_expression_l219_219456


namespace angle_equality_l219_219045

-- Definitions for quadrilateral, intersection points, projection, and angles
def convex_quadrilateral (A B C D : Type) := sorry
def intersection (X Y : Type) : Type := sorry
def projection (P line : Type) : Type := sorry
def angle (X Y Z : Type) : ℝ := sorry

-- Given conditions
variables (A B C D E F P O : Type)
variables [convex_quadrilateral A B C D]
variables [intersection (line_through A B) (line_through C D) = E]
variables [intersection (line_through A D) (line_through B C) = F]
variables [intersection (line_through A C) (line_through B D) = P]
variables [projection P (line_through E F) = O]

-- Goal statement
theorem angle_equality : angle B O C = angle A O D :=
by
  sorry

end angle_equality_l219_219045


namespace necessary_but_not_sufficient_l219_219628

variable (p q : Prop)

theorem necessary_but_not_sufficient (h : ¬p → q) (h1 : ¬ (q → ¬p)) : ¬q → p := 
by
  sorry

end necessary_but_not_sufficient_l219_219628


namespace train_fraction_speed_l219_219424

theorem train_fraction_speed :
  let T := 50.000000000000014
  let delay := 10
  let fraction := (T / (T + delay))
  fraction = 5 / 6 :=
by
  let T := 50.000000000000014
  let delay := 10
  let fraction := (T / (T + delay))
  show fraction = 5 / 6
  sorry

end train_fraction_speed_l219_219424


namespace sum_prime_factors_of_143_l219_219341

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219341


namespace coefficient_x4_term_l219_219011

theorem coefficient_x4_term {α : Type*} [CommRing α] :
  (binomial 5 4 : ℕ) + (binomial 6 4 : ℕ) + (binomial 7 4 : ℕ) = 55 :=
by
  sorry

end coefficient_x4_term_l219_219011


namespace sum_prime_factors_143_is_24_l219_219283

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219283


namespace total_snakes_count_l219_219173

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l219_219173


namespace find_c_l219_219220

def is_midpoint (p1 p2 mid : ℝ × ℝ) : Prop :=
(mid.1 = (p1.1 + p2.1) / 2) ∧ (mid.2 = (p1.2 + p2.2) / 2)

def is_perpendicular_bisector (line : ℝ → ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop := 
∃ mid : ℝ × ℝ, 
is_midpoint p1 p2 mid ∧ line mid.1 mid.2 = 0

theorem find_c (c : ℝ) : 
is_perpendicular_bisector (λ x y => 3 * x - y - c) (2, 4) (6, 8) → c = 6 :=
by
  sorry

end find_c_l219_219220


namespace triangle_area_proof_l219_219101

noncomputable def area_of_triangle (X Y Z : Type) [is_triangle X Y Z] (U V K : Type)
  [is_median X U] [is_median Y V] [is_centroid K] (h_intersect : right_angle XU YV)
  (XU_len : XU = 12) (YV_len : YV = 18) : ℝ :=
  have XK : ℝ := (2 / 3) * XU_len,
  have YK : ℝ := (2 / 3) * YV_len,
  have area_XYK : ℝ := (1 / 2) * XK * YK,
  have area_XYZ : ℝ := 3 * area_XYK,
  area_XYZ

theorem triangle_area_proof (X Y Z U V K : Type)
  [is_triangle X Y Z] [is_median X U] [is_median Y V] [is_centroid K]
  (h_intersect : right_angle XU YV) (XU_len : XU = 12) (YV_len : YV = 18) :
  area_of_triangle X Y Z U V K h_intersect XU_len YV_len = 144 :=
sorry

end triangle_area_proof_l219_219101


namespace diameter_of_large_circle_l219_219001

theorem diameter_of_large_circle :
  ∃ D : ℝ, (∀ (r : ℝ) (n : ℕ) (θ : ℝ), r = 4 ∧ n = 8 ∧ θ = 45 ∧ D = 2 * (4 * tan (θ / 2) + 4)) → abs (D - 11.312) < 0.001 :=
by
  sorry

end diameter_of_large_circle_l219_219001


namespace total_cuts_length_eq_60_l219_219820

noncomputable def total_length_of_cuts (side_length : ℝ) (num_rectangles : ℕ) : ℝ :=
  if side_length = 36 ∧ num_rectangles = 3 then 60 else 0

theorem total_cuts_length_eq_60 :
  ∀ (side_length : ℝ) (num_rectangles : ℕ),
    side_length = 36 ∧ num_rectangles = 3 →
    total_length_of_cuts side_length num_rectangles = 60 := by
  intros
  simp [total_length_of_cuts]
  sorry

end total_cuts_length_eq_60_l219_219820


namespace gold_percentage_in_fourth_bar_l219_219792

-- Conditions given in the problem
def weight_total : ℕ := 1700
def gold_total_percentage : ℕ := 56
def bar1_weight : ℕ := 300
def bar1_gold_percentage : ℕ := 96
def bar2_weight : ℕ := 200
def bar2_gold_percentage : ℕ := 86
def bar3_weight : ℕ := 400
def bar3_gold_percentage : ℕ := 64

-- This needs to be proven
theorem gold_percentage_in_fourth_bar :
  let bar4_weight := weight_total - (bar1_weight + bar2_weight + bar3_weight) in
  let bar4_gold := λ x : ℕ, x in
  (300 * 96 + 200 * 86 + 400 * 64 + bar4_weight * bar4_gold) / weight_total = 56 → 
  bar4_gold = 29.5 :=
by
  let bar4_weight := weight_total - (bar1_weight + bar2_weight + bar3_weight)
  let bar1_gold := bar1_weight * bar1_gold_percentage / 100
  let bar2_gold := bar2_weight * bar2_gold_percentage / 100
  let bar3_gold := bar3_weight * bar3_gold_percentage / 100
  let gold_total := 1700 * gold_total_percentage / 100
  have h : bar1_gold + bar2_gold + bar3_gold + bar4_weight * 29.5 / 100 = gold_total := sorry
  sorry

end gold_percentage_in_fourth_bar_l219_219792


namespace inequality_proof_l219_219633

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) : 
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := 
sorry

end inequality_proof_l219_219633


namespace sum_of_roots_l219_219232

theorem sum_of_roots :
  let φs := (List.range 7).map (λ k, (225 + 360 * k) / 7 : ℝ)
  0 ≤ φs.min' (by simp) ∧ φs.max' (by simp) < 360 → (φs.sum = 1305) :=
by
  sorry

end sum_of_roots_l219_219232


namespace sum_of_prime_factors_of_143_l219_219332

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219332


namespace rate_of_grapes_l219_219551

theorem rate_of_grapes (G : ℝ) (H : 8 * G + 9 * 50 = 1010) : G = 70 := by
  sorry

end rate_of_grapes_l219_219551


namespace problem_statement_l219_219428

def count_ways : Nat :=
  let volunteers := (1 to 30).toList
  let chosen_three := [6, 15, 24]
  let remaining := volunteers.filter (λ n => ¬ n ∈ chosen_three)
  let selected_above_25 := remaining.filter (λ n => n > 25)
  let selected_below_6 := remaining.filter (λ n => n < 6)
  let ways_above_25 := Nat.choose selected_above_25.length 3 
  let ways_below_6 := Nat.choose selected_below_6.length 3 
  let total_ways := ways_above_25 + ways_below_6
  total_ways * 2  -- considering the 2 arrangements into the Jiangxi Hall and the Guangdian Hall

theorem problem_statement : count_ways = 60 := by
  sorry

end problem_statement_l219_219428


namespace sereja_picked_more_berries_l219_219854

theorem sereja_picked_more_berries (total_berries : ℕ) (sereja_rate : ℕ) (dima_rate : ℕ)
  (sereja_pattern_input : ℕ) (sereja_pattern_eat : ℕ)
  (dima_pattern_input : ℕ) (dima_pattern_eat : ℕ)
  (rate_relation : sereja_rate = 2 * dima_rate)
  (total_berries_collected : sereja_rate + dima_rate = total_berries) :
  let sereja_collected := total_berries_collected * (sereja_pattern_input / (sereja_pattern_input + sereja_pattern_eat)),
      dima_collected := total_berries_collected * (dima_pattern_input / (dima_pattern_input + dima_pattern_eat)) in
  sereja_collected > dima_collected ∧ sereja_collected - dima_collected = 50 :=
by {
  sorry
}

end sereja_picked_more_berries_l219_219854


namespace partial_fraction_product_l219_219691

theorem partial_fraction_product (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x^2 - 10 * x + 24 ≠ 0 →
            (x^2 - 25) / (x^3 - 3 * x^2 - 10 * x + 24) = A / (x - 2) + B / (x + 3) + C / (x - 4)) →
  A = 1 ∧ B = 1 ∧ C = 1 →
  A * B * C = 1 := by
  sorry

end partial_fraction_product_l219_219691


namespace max_donation_proof_l219_219390

noncomputable def max_donation (tutoring_cost_per_hour : ℕ) (investment_income : ℕ)
                              (working_days_per_month : ℕ) (sleep_hours : ℕ) 
                              (monthly_expenses : ℕ) 
                              (donation_per_rest_hour : ℕ) : ℕ :=
  let L_i := (16 - k) / 3 in  -- Work hours per day
  let total_hours_rest := λ k * working_days_per_month in -- Total rest hours per month
  let total_donation := total_hours_rest / donation_per_rest_hour in -- Max donation per month
  total_donation

theorem max_donation_proof : max_donation 3 14 21 8 70 3 = 70 := by
  sorry

end max_donation_proof_l219_219390


namespace geometric_sequence_value_l219_219693

theorem geometric_sequence_value (a : ℝ) (h_pos : 0 < a) 
    (h_geom1 : ∃ r, 25 * r = a)
    (h_geom2 : ∃ r, a * r = 7 / 9) : 
    a = 5 * Real.sqrt 7 / 3 :=
by
  sorry

end geometric_sequence_value_l219_219693


namespace anna_reading_time_l219_219440

theorem anna_reading_time:
  (∀ n : ℕ, n ∈ (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1) → True) →
  (let chapters_read := (Finset.range 31).filter (λ x, ¬ (∃ k : ℕ, k * 3 + 3 = x + 1)).card,
  reading_time := chapters_read * 20,
  hours := reading_time / 60 in
  hours = 7) :=
by
  intros
  let chapters_read := (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1).card
  have h1 : chapters_read = 21 := by sorry
  let reading_time := chapters_read * 20
  have h2 : reading_time = 420 := by sorry
  let hours := reading_time / 60
  have h3 : hours = 7 := by sorry
  exact h3

end anna_reading_time_l219_219440


namespace cats_weight_more_than_puppies_l219_219928

theorem cats_weight_more_than_puppies :
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  (num_cats * weight_per_cat) - (num_puppies * weight_per_puppy) = 5 :=
by 
  let num_puppies := 4
  let weight_per_puppy := 7.5
  let num_cats := 14
  let weight_per_cat := 2.5
  sorry

end cats_weight_more_than_puppies_l219_219928


namespace inequality_proof_l219_219884

variable {n : ℕ} (h_n : n ≥ 2)
variable (x : Fin n → ℝ) 
variable (h_x_pos : ∀ i, 0 < x i)
variable (h_sum_eq_one : (Finset.univ.sum x) = 1)

theorem inequality_proof 
  (h_n : n ≥ 2) 
  (h_x_pos : ∀ i, 0 < x i) 
  (h_sum_eq_one : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ i, (x i) / (Real.sqrt (1 - (x i))))) 
  ≥ (Finset.univ.sum (λ i, Real.sqrt (x i)) / (Real.sqrt (n - 1))) := 
  sorry

end inequality_proof_l219_219884


namespace total_snakes_l219_219179

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l219_219179


namespace class_total_cost_is_correct_l219_219397

-- Define the conditions
def num_students := 30
def num_teachers := 4
def student_ticket_price := 8
def teacher_ticket_price := 12
def group_discount := 0.20
def bus_rental_cost := 150
def meal_cost_per_person := 10

-- Calculate if group discount applies
def group_size := num_students + num_teachers
def qualify_for_discount := group_size >= 25

-- Calculate ticket costs before discount
def total_cost_tickets_before_discount := (num_students * student_ticket_price) + (num_teachers * teacher_ticket_price)

-- Apply discount if applicable
def discount_amount := if qualify_for_discount then group_discount * total_cost_tickets_before_discount else 0
def total_cost_tickets_after_discount := total_cost_tickets_before_discount - discount_amount

-- Calculate meal costs
def total_meal_cost := meal_cost_per_person * group_size

-- Calculate total cost
def total_cost := total_cost_tickets_after_discount + bus_rental_cost + total_meal_cost

-- Prove that the total cost is $720.40
theorem class_total_cost_is_correct : total_cost = 720.40 := 
  by 
    -- skipping the proof
    sorry

end class_total_cost_is_correct_l219_219397


namespace sum_prime_factors_of_143_l219_219344

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219344


namespace positional_relationship_perpendicular_l219_219037

theorem positional_relationship_perpendicular 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h : b * Real.sin A - a * Real.sin B = 0) :
  (∀ x y : ℝ, (x * Real.sin A + a * y + c = 0) ↔ (b * x - y * Real.sin B + Real.sin C = 0)) :=
sorry

end positional_relationship_perpendicular_l219_219037


namespace impossible_arrangement_l219_219132

theorem impossible_arrangement (s : Finset ℕ) (h₁ : s = Finset.range 2018 \ {0})
  (h₂ : ∀ a ∈ s, ∀ b ∈ s, a ≠ b ∧ (b = a + 17 ∨ b = a + 21 ∨ b = a - 17 ∨ b = a - 21)) : False :=
by
  sorry

end impossible_arrangement_l219_219132


namespace sequence_property_l219_219893

theorem sequence_property
  (a : ℕ+ → ℝ)
  (k : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2)
  (h4 : a 4 = 64)
  (h_rec : ∀ n : ℕ+, a n * a (n + 2) = k * (a (n + 1))^2) :
  k = 2 ∧ ∀ n : ℕ+, a n = 2^((n * (n - 1)) / 2) := by
  sorry

end sequence_property_l219_219893


namespace total_time_to_complete_work_l219_219171

noncomputable def mahesh_work_rate (W : ℕ) := W / 40
noncomputable def mahesh_work_done_in_20_days (W : ℕ) := 20 * (mahesh_work_rate W)
noncomputable def remaining_work (W : ℕ) := W - (mahesh_work_done_in_20_days W)
noncomputable def rajesh_work_rate (W : ℕ) := (remaining_work W) / 30

theorem total_time_to_complete_work (W : ℕ) :
    (mahesh_work_rate W) + (rajesh_work_rate W) = W / 24 →
    (mahesh_work_done_in_20_days W) = W / 2 →
    (remaining_work W) = W / 2 →
    (rajesh_work_rate W) = W / 60 →
    20 + 30 = 50 :=
by 
  intros _ _ _ _
  sorry

end total_time_to_complete_work_l219_219171


namespace rectangle_area_eq_six_l219_219217

-- Define the areas of the small squares
def smallSquareArea : ℝ := 1

-- Define the number of small squares
def numberOfSmallSquares : ℤ := 2

-- Define the area of the larger square
def largeSquareArea : ℝ := (2 ^ 2)

-- Define the area of rectangle ABCD
def areaRectangleABCD : ℝ :=
  (numberOfSmallSquares * smallSquareArea) + largeSquareArea

-- The theorem we want to prove
theorem rectangle_area_eq_six :
  areaRectangleABCD = 6 := by sorry

end rectangle_area_eq_six_l219_219217


namespace percentage_increase_l219_219136

variable (T : ℕ) (total_time : ℕ)

theorem percentage_increase (h1 : T = 4) (h2 : total_time = 10) : 
  ∃ P : ℕ, (T + P / 100 * T = total_time - T) → P = 50 := 
by 
  sorry

end percentage_increase_l219_219136


namespace cone_base_circumference_l219_219806

theorem cone_base_circumference (r : ℝ) (h_r : r = 6) : 
  let original_circumference := 2 * Real.pi * r in
  let sector_angle := 180 / 360 in
  let base_circumference := sector_angle * original_circumference in
  base_circumference = 6 * Real.pi := 
by 
  sorry  -- Proof to be filled in later

end cone_base_circumference_l219_219806


namespace length_on_ninth_day_l219_219589

-- Define relevant variables and conditions.
variables (a1 d : ℕ)

-- Define conditions as hypotheses.
def problem_conditions : Prop :=
  (7 * a1 + 21 * d = 28) ∧ 
  (a1 + d + a1 + 4 * d + a1 + 7 * d = 15)

theorem length_on_ninth_day (h : problem_conditions a1 d) : (a1 + 8 * d = 9) :=
  sorry

end length_on_ninth_day_l219_219589


namespace cannot_complete_task_l219_219370

theorem cannot_complete_task (grid : fin 6 × fin 6 → ℕ) 
  (strips : fin 11 → fin 3 × fin 1 → ℕ) (corner : fin 3 → fin 3 → ℕ) :
  (∀ i j, grid (i, j) ∈ {1, 2, 3}) ∧
  (sum (λ i, sum (λ j, grid (i, j))) = 72) ∧
  (∀ s, sum (λ j, strips s (j, 0)) % 3 = 0) ∧
  (sum (λ i, sum (λ j, corner i j)) % 3 ≠ 0) →
  ¬ (∃ (position_strips : fin 11 → (fin 6 × fin 6 → bool))
     (position_corner : fin 6 × fin 6 → bool),
     (∀ s, sum (λ i, position_strips s (i, 0)) = 3) ∧
     (sum (λ i, sum (λ j, position_corner (i, j))) = 3) ∧
     (∀ i j, position_strips i j ∨ position_corner i j → grid (i, j) = strips i j ∨ grid (i, j) = corner i j)) :=
by 
  intro h,
  sorry

end cannot_complete_task_l219_219370


namespace circle_equation_tangent_line_l219_219886

-- Let assumptions be defined here
variables (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C]
variables (A B : C)
variables (l : C → Prop)
variables (center : C)

-- Conditions
def on_line (c : C) : Prop := l c
def passes_through (c : C) (p : C) : Prop := dist c p = dist c B

-- Statements
theorem circle_equation (h_line : on_line center)
                        (h_A : passes_through center A)
                        (h_B : passes_through center B) :
    ∃ (x y : ℝ), (x + 3)^2 + (y + 2)^2 = 25 := sorry

theorem tangent_line (h_A : passes_through center A)
                     (h_tangent_point : A) :
    ∃ (x y : ℝ), 4 * x + 3 * y - 7 = 0 := sorry

end circle_equation_tangent_line_l219_219886


namespace sum_even_102_to_200_l219_219701

def sum_even_integers (m n : ℕ) : ℕ :=
  sum (list.map (λ k, 2 * k) (list.range' m (n - m + 1)))

theorem sum_even_102_to_200 :
  sum_even_integers (102 / 2) (200 / 2) = 7550 := 
sorry

end sum_even_102_to_200_l219_219701


namespace range_of_b_l219_219070

theorem range_of_b (b c : ℝ) :
  (∀ x ∈ set.Iio 1, (differentiable_at ℝ (λ x : ℝ, x^2 + b*x + c) x) ∧
          (derivative (λ x : ℝ, x^2 + b*x + c) x ≤ 0)) →
  b ≤ -2 := by
  sorry

end range_of_b_l219_219070


namespace standard_01_sequences_count_14_l219_219848

def is_standard_01_sequence (a : List ℕ) : Prop :=
  (∀ n : ℕ, n < a.length → (a.take n).count '0' ≥ (a.take n).count '1') ∧ 
  (a.count '0' = a.count '1') ∧ 
  (a.length % 2 = 0)

def count_standard_01_sequences (m : ℕ) : ℕ :=
  (List.range (2 * m)).filter (λ a, is_standard_01_sequence a).length

theorem standard_01_sequences_count_14 (m : ℕ) (hm : m = 4) :
  count_standard_01_sequences m = 14 := 
by 
  rw [hm]
  sorry

end standard_01_sequences_count_14_l219_219848


namespace sequence_property_l219_219236

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l219_219236


namespace sum_of_permutations_divisible_by_9_l219_219790

theorem sum_of_permutations_divisible_by_9:
  let digits := [1, 2, 3, 4, 5, 6, 7] in
  let factorial := 5040 in
  (∑ n in (Finset.permutations digits), n ) % 9 = 0 :=
sorry

end sum_of_permutations_divisible_by_9_l219_219790


namespace round_0_689_to_two_decimal_places_l219_219664

def round_to_two_decimal_places (x : ℝ) : ℝ :=
  (Real.floor ((x * 100.0 + 0.5)) / 100)

theorem round_0_689_to_two_decimal_places :
  round_to_two_decimal_places 0.689 = 0.69 :=
by
  sorry

end round_0_689_to_two_decimal_places_l219_219664


namespace count_integer_A_l219_219868

def A (n : ℕ) : ℝ :=
  if n < 2 then 0 else ∑ i in Ico 1 (n + 1), (i * floor (sqrt (i)))

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem count_integer_A : finset.card {n ∈ finset.Icc 2 2000 | is_integer (A n)} = 1315 := sorry

end count_integer_A_l219_219868


namespace chairs_adjacent_subsets_l219_219730

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219730


namespace bounces_to_threshold_l219_219417

-- State the parameters of the problem
def initial_height : ℝ := 320
def rebound_ratio : ℝ := 3 / 4
def threshold_height : ℝ := 40

-- Define the main theorem
theorem bounces_to_threshold (b : ℕ) : 
  (b > (real.log (threshold_height / initial_height) / real.log rebound_ratio)).ceil = 25 :=
by
  -- Using the definitions and arithmetic, we can asserts the condition
  let lhs := (real.log (threshold_height / initial_height) / real.log rebound_ratio).ceil
  have h1 : lhs = 25 := sorry
  exact h1

end bounces_to_threshold_l219_219417


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219756

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219756


namespace equilateral_triangle_with_midpoints_l219_219798

theorem equilateral_triangle_with_midpoints (A B C P Q : Type)
  [EquilateralTriangle A B C]
  (midpoint : is_midpoint P A C)
  (hAC : length A C = length A P)
  (hAP : length A P = length P Q)
  (hPQ : length P Q = length Q B)
  (hQB : length Q B = length B C) :
  measure_angle B = 60 :=
by
  sorry

end equilateral_triangle_with_midpoints_l219_219798


namespace sum_prime_factors_of_143_l219_219352

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219352


namespace total_snakes_l219_219180

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l219_219180


namespace rootiful_contains_all_integers_l219_219767

def is_rootiful (S : Set ℤ) : Prop :=
  ∀ (n : ℕ) (a : Fin (n+1) → ℤ), (∀ x : ℤ, (∑ i in Finset.range (n+1), a ⟨i, Nat.lt_succ_iff.mpr (Fin.is_lt ⟨i, Nat.lt_succ_self n⟩)⟩ * x^i = 0) → x ∈ S)

def contains_form (S : Set ℤ) : Prop :=
  ∀ a b : ℕ, 0 < a → 0 < b → 2^a - 2^b ∈ S

theorem rootiful_contains_all_integers (S : Set ℤ) (h1 : is_rootiful S) (h2 : contains_form S) : S = Set.univ :=
by
  sorry

end rootiful_contains_all_integers_l219_219767


namespace angle_XPY_eq_angle_XQY_add_angle_XRY_l219_219617

-- Definitions based on given conditions
variable (A B C D P X Y Q R : Point)
variable (parallelogram_abcd : Parallelogram A B C D)
variable (angle_A_obtuse : Angle A > 90)
variable (P_on_BD : OnSegment P B D)
variable (circle_with_center_P : Circle P A)
variable (cuts_AD_at_Y : CutsLineAt circle_with_center_P A D A Y)
variable (cuts_AB_at_X : CutsLineAt circle_with_center_P A B A X)
variable (AP_intersects_BC_at_Q : IntersectsLineAt A P B C Q)
variable (AP_intersects_CD_at_R : IntersectsLineAt A P C D R)

-- The theorem to be proven
theorem angle_XPY_eq_angle_XQY_add_angle_XRY 
  (h1 : ∀ X Y P, Center P X Y → ∠ XP Y)
  (h2 : ∀ X Q Y R, InscribedAngle X Q Y R) :
  ∠ XP Y = ∠ XQ Y + ∠ XR Y :=
sorry -- Proof placeholder

end angle_XPY_eq_angle_XQY_add_angle_XRY_l219_219617


namespace golden_rod_total_weight_l219_219590

theorem golden_rod_total_weight:
  (∀ n ∈ {1, 2, 3, 4, 5}, ∃ w : ℝ, (w = 4 - (n - 1) * d)) →
  (n_foot_end : ∃ (d : ℝ), (d = (4 - 2) / 4)) →
  (∀ i, 1 ≤ i ∧ i ≤ 5 → ∃ x : ℝ, (x = 2 + (i - 1) * d)) →
  Σ x in {1, 2, 3, 4, 5}, x = 15 := 
by
  sorry

end golden_rod_total_weight_l219_219590


namespace boys_count_l219_219579

variables (B G : ℕ)

-- Conditions
def girls_more_than_boys := G = B + 400
def sixty_percent_total := 0.60 * (B + G) = 960

-- Theorem statement: prove the number of boys is 600 given the conditions
theorem boys_count (h1 : girls_more_than_boys B G) (h2 : sixty_percent_total B G) :
  B = 600 :=
sorry

end boys_count_l219_219579


namespace largest_mass_of_one_fish_l219_219998

theorem largest_mass_of_one_fish (total_fish : ℕ)
  (min_mass : ℝ)
  (total_mass_first_three : ℝ)
  (average_mass_condition : (total_mass_first_three / 3 = M / total_fish))
  (M : ℝ)
  (total_remaining_18 : ℝ)
  (max_mass_of_one_fish : ℝ) :
  total_fish = 21 →
  min_mass = 0.2 →
  total_mass_first_three = 1.5 →
  M = total_fish * (total_mass_first_three / 3) →
  total_remaining_18 = M - total_mass_first_three →
  max_mass_of_one_fish = total_remaining_18 - (total_fish - 4) * min_mass →
  max_mass_of_one_fish = 5.6 :=
by {
  intro h_total_fish h_min_mass h_total_mass_first_three h_M h_total_remaining_18 h_max_mass_of_one_fish,
  sorry
}

end largest_mass_of_one_fish_l219_219998


namespace time_to_pass_l219_219384
-- Import the Mathlib library

-- Define the lengths of the trains
def length_train1 := 150 -- meters
def length_train2 := 150 -- meters

-- Define the speeds of the trains in km/h
def speed_train1_kmh := 95 -- km/h
def speed_train2_kmh := 85 -- km/h

-- Convert speeds to m/s
def speed_train1_ms := (speed_train1_kmh * 1000) / 3600 -- meters per second
def speed_train2_ms := (speed_train2_kmh * 1000) / 3600 -- meters per second

-- Calculate the relative speed in m/s (since they move in opposite directions, the relative speed is additive)
def relative_speed_ms := speed_train1_ms + speed_train2_ms -- meters per second

-- Calculate the total distance to be covered (sum of the lengths of the trains)
def total_length := length_train1 + length_train2 -- meters

-- State the theorem: the time taken for the trains to pass each other
theorem time_to_pass :
  total_length / relative_speed_ms = 6 := by
  sorry

end time_to_pass_l219_219384


namespace trig_identity_l219_219393

theorem trig_identity (a b : ℝ) (ha : a = 145) (hb : b = 35) :
  sin (real.to_radians a) * cos (real.to_radians b) = (1 / 2) * sin (real.to_radians (2 * b)) :=
begin
  sorry
end

end trig_identity_l219_219393


namespace total_profit_l219_219827

theorem total_profit (A_invest B_invest C_invest A_share : ℝ) (h1 : A_invest = 6300) (h2 : B_invest = 4200) (h3 : C_invest = 10500) (h4 : A_share = 3660) : 
  let ratio : ℝ := 3 / 10 in  
  let P := A_share / ratio in 
  P = 12200 := 
by {
  rw [h1, h2, h3, h4],
  have ratio_def : ratio = 3 / 10 := rfl,
  have P_def : P = A_share / ratio := rfl,
  rw [ratio_def, P_def, h4],
  norm_num,
  sorry
}

end total_profit_l219_219827


namespace inscribed_circle_area_l219_219891

/-- Defining the inscribed circle problem and its area. -/
theorem inscribed_circle_area (l : ℝ) (h₁ : 90 = 90) (h₂ : true) : 
  ∃ r : ℝ, (r = (2 * (Real.sqrt 2 - 1) * l / Real.pi)) ∧ ((Real.pi * r ^ 2) = (12 - 8 * Real.sqrt 2) * l ^ 2 / Real.pi) :=
  sorry

end inscribed_circle_area_l219_219891


namespace find_a22_l219_219240

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l219_219240


namespace standard_price_of_strawberries_l219_219833

-- Define the conditions
def entrance_fee : ℝ := 4
def total_paid : ℝ := 128
def total_picked : ℝ := 7
def num_people : ℕ := 3
def total_entrance_fee : ℝ := num_people * entrance_fee
def cost_without_entrance : ℝ := total_paid - total_entrance_fee
def price_per_pound : ℝ := cost_without_entrance / total_picked

-- Problem: Prove the standard price of a pound of strawberries
theorem standard_price_of_strawberries : price_per_pound = 16.57 := 
by sorry

end standard_price_of_strawberries_l219_219833


namespace largest_possible_product_l219_219274

theorem largest_possible_product :
  ∃ (a b c d e f : ℕ), 
    (a ≠ 0) ∧ 
    (e ≠ 0) ∧ 
    {a, b, c, d, e, f} = {1, 3, 5, 8, 9, 0} ∧ 
    1000 * a + 100 * b + 10 * c + d > 999 ∧ 
    10 * e + f > 9 ∧ 
    (1000 * a + 100 * b + 10 * c + d) * (10 * e + f) = 760240 :=
sorry

end largest_possible_product_l219_219274


namespace count_valid_n_l219_219870

theorem count_valid_n : 
  let valid_n := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ (250 + n) % n = 0} 
  in valid_n.to_finset.card = 3 :=
by
  sorry

end count_valid_n_l219_219870


namespace product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l219_219198

theorem product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers (n : ℤ) :
  let T := (n - 1) * n * (n + 1) * (n + 2)
  let M := n * (n + 1)
  T = (M - 2) * M :=
by
  -- proof here
  sorry

end product_of_four_consecutive_integers_is_product_of_two_consecutive_even_numbers_l219_219198


namespace solve_for_x2_minus_y2_minus_z2_l219_219959

theorem solve_for_x2_minus_y2_minus_z2
  (x y z : ℝ)
  (h1 : x + y + z = 12)
  (h2 : x - y = 4)
  (h3 : y + z = 7) :
  x^2 - y^2 - z^2 = -12 :=
by
  sorry

end solve_for_x2_minus_y2_minus_z2_l219_219959


namespace train_time_to_pass_bridge_l219_219427

def length_train : ℝ := 310  -- Length of the train in meters
def speed_train_kmph : ℝ := 45  -- Speed of the train in km/h
def length_bridge : ℝ := 140  -- Length of the bridge in meters

def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)  -- Conversion from km/h to m/s
def total_distance (train_length bridge_length : ℝ) : ℝ := train_length + bridge_length  -- Total distance to cover

theorem train_time_to_pass_bridge :
  let speed_train_mps := kmph_to_mps speed_train_kmph in
  let distance := total_distance length_train length_bridge in
  (distance / speed_train_mps) = 36 :=
by
  sorry

end train_time_to_pass_bridge_l219_219427


namespace subsets_with_at_least_four_adjacent_chairs_l219_219725

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219725


namespace find_xy_l219_219482

noncomputable def is_solution (x y : ℤ) : Prop :=
  even x ∧ log 2 (x/7 + y/6) = log 2 (x/7) + log 2 (y/6)

theorem find_xy : ∃ (x y : ℤ), is_solution x y ∧ ((x, y) = (8, 48) ∨ (x, y) = (10, 20) ∨ (x, y) = (14, 12) ∨ (x, y) = (28, 8)) :=
by {
  sorry
}

end find_xy_l219_219482


namespace find_m_if_f_is_odd_l219_219096

variable (f : ℝ → ℝ)
variable (m n : ℝ)

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem find_m_if_f_is_odd (h : is_odd_function (λ x : ℝ, x^3 + 3 * m * x^2 + n * x + m^2)) : m = 0 :=
by
  sorry

end find_m_if_f_is_odd_l219_219096


namespace car_mileage_correct_l219_219808

noncomputable def car_mileage (reading : Nat) : Nat :=
  -- Function to convert base-8 representation with skipped digits 3 and 4 to decimal
  let convert_base8_to_dec (n : Nat) : Nat :=
    let digits := [0, 1, 2, 5, 6, 7, 8, 9] -- Representation of base-8 like system
    let str_repr := repr n
    let base8_digits := str_repr.toList.map (λ c, digits.indexOf (c.toNat - '0'.toNat))
    base8_digits.foldl (λ acc d, 8 * acc + d) 0
  in convert_base8_to_dec reading

-- Odometer reading condition
def odometer_reading : Nat :=
  3006

-- Mathematically equivalent proof problem statement
theorem car_mileage_correct : car_mileage odometer_reading = 1030 :=
  by
  -- The actual proof steps would go here
  sorry

end car_mileage_correct_l219_219808


namespace sum_prime_factors_143_is_24_l219_219287

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219287


namespace minimum_value_7a_4b_l219_219063

noncomputable def original_cond (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4

theorem minimum_value_7a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  original_cond a b ha hb → 7 * a + 4 * b = 9 / 4 :=
by
  sorry

end minimum_value_7a_4b_l219_219063


namespace sum_of_prime_factors_of_143_l219_219307

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219307


namespace undefined_values_of_fraction_l219_219876

theorem undefined_values_of_fraction (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_values_of_fraction_l219_219876


namespace range_cos_2alpha_cos_2beta_l219_219056

variable (α β : ℝ)
variable (h : Real.sin α + Real.cos β = 3 / 2)

theorem range_cos_2alpha_cos_2beta :
  -3/2 ≤ Real.cos (2 * α) + Real.cos (2 * β) ∧ Real.cos (2 * α) + Real.cos (2 * β) ≤ 3/2 :=
sorry

end range_cos_2alpha_cos_2beta_l219_219056


namespace part1_solution_part2_solution_l219_219369

-- Conditions
variables (x y : ℕ) -- Let x be the number of parcels each person sorts manually per hour,
                     -- y be the number of machines needed

def machine_efficiency : ℕ := 20 * x
def time_machines (parcels : ℕ) (machines : ℕ) : ℕ := parcels / (machines * machine_efficiency x)
def time_people (parcels : ℕ) (people : ℕ) : ℕ := parcels / (people * x)
def parcels_per_day : ℕ := 100000

-- Problem 1: Find x
axiom problem1 : (time_people 6000 20) - (time_machines 6000 5) = 4

-- Problem 2: Find y to sort 100000 parcels in a day with machines working 16 hours/day
axiom problem2 : 16 * machine_efficiency x * y ≥ parcels_per_day

-- Correct answers:
theorem part1_solution : x = 60 := by sorry
theorem part2_solution : y = 6 := by sorry

end part1_solution_part2_solution_l219_219369


namespace sum_of_prime_factors_of_143_l219_219309

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219309


namespace system_real_solutions_l219_219018

theorem system_real_solutions (a b c : ℝ) :
  (∃ x : ℝ, 
    a * x^2 + b * x + c = 0 ∧ 
    b * x^2 + c * x + a = 0 ∧ 
    c * x^2 + a * x + b = 0) ↔ 
  a + b + c = 0 :=
sorry

end system_real_solutions_l219_219018


namespace anna_reading_time_l219_219448

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l219_219448


namespace mean_of_added_numbers_l219_219212

theorem mean_of_added_numbers (mean_seven : ℝ) (mean_ten : ℝ) (x y z : ℝ)
    (h1 : mean_seven = 40)
    (h2 : mean_ten = 55) :
    (mean_seven * 7 + x + y + z) / 10 = mean_ten → (x + y + z) / 3 = 90 :=
by sorry

end mean_of_added_numbers_l219_219212


namespace box_problem_l219_219818

noncomputable def box_height (m n : ℕ) (h_rel_prime : Nat.coprime m n) : ℚ := m / n

theorem box_problem (m n : ℕ) (h_rel_prime : Nat.coprime m n)
  (h_area : let h := box_height m n h_rel_prime in
   let w := 12 in
   let l := 16 in
   let s₁ := (w^2 + l^2) / 4 in
   let s₂ := (w^2 + h^2) / 4 in
   let s₃ := (l^2 + h^2) / 4 in
   let area_triangle := Real.sqrt ((s₁ + s₂ + s₃)^2 - 2 * (s₁^2 + s₂^2 + s₃^2)) / 4 in
   area_triangle = 30) 
  : m + n = 41 := 
sorry

end box_problem_l219_219818


namespace real_values_b_non_real_roots_l219_219092

theorem real_values_b_non_real_roots (b : ℝ) : 
  (∀ (a : ℝ) (c : ℝ), a = 1 ∧ c = 16 ∧ (b^2 - 4 * a * c < 0) → b ∈ Ioo (-8 : ℝ) (8 : ℝ)) := 
by
  intro a c h
  cases h with ha hc
  cases hc with hc hd
  rw [ha, hc] at hd
  have h1 : b^2 < 64 := by linarith
  split
  { linarith }
  { linarith }

end real_values_b_non_real_roots_l219_219092


namespace count_valid_n_l219_219871

theorem count_valid_n : 
  let valid_n := {n : ℕ | 1 ≤ n ∧ n ≤ 9 ∧ (250 + n) % n = 0} 
  in valid_n.to_finset.card = 3 :=
by
  sorry

end count_valid_n_l219_219871


namespace simplify_expression_l219_219493

theorem simplify_expression : ((1 + 2 + 3 + 4 + 5 + 6) / 3 + (3 * 5 + 12) / 4) = 13.75 :=
by
-- Proof steps would go here, but we replace them with 'sorry' for now.
sorry

end simplify_expression_l219_219493


namespace andy_older_than_rahim_l219_219964

-- Define Rahim's current age
def Rahim_current_age : ℕ := 6

-- Define Andy's age in 5 years
def Andy_age_in_5_years : ℕ := 2 * Rahim_current_age

-- Define Andy's current age
def Andy_current_age : ℕ := Andy_age_in_5_years - 5

-- Define the difference in age between Andy and Rahim right now
def age_difference : ℕ := Andy_current_age - Rahim_current_age

-- Theorem stating the age difference between Andy and Rahim right now is 1 year
theorem andy_older_than_rahim : age_difference = 1 :=
by
  -- Proof is skipped
  sorry

end andy_older_than_rahim_l219_219964


namespace hyperbola_decreasing_l219_219026

-- Defining the variables and inequalities
variables (x : ℝ) (m : ℝ)
hypothesis (hx : x > 0)

-- The statement we need to prove
theorem hyperbola_decreasing (hx : x > 0) : (∀ x, x > 0 → (λ (x : ℝ), (1 - m) / x) x > (λ (x' : ℝ), (1 - m) / (x + x')) x) ↔ m < 1 :=
begin
  sorry
end

end hyperbola_decreasing_l219_219026


namespace probability_of_last_yellow_marble_l219_219834

-- Define the conditions and the bags
def BagA := {white := 5, black := 5}
def BagB := {yellow := 8, blue := 6}
def BagC := {yellow := 3, blue := 7}
def BagD := {green := 4, red := 6}

-- Probability calculations
def probability_last_marble_yellow : ℚ :=
  let p_white := BagA.white.to_rat / (BagA.white + BagA.black)
  let p_black := BagA.black.to_rat / (BagA.white + BagA.black)
  
  let p_yellow_from_B := BagB.yellow.to_rat / (BagB.yellow + BagB.blue)
  let p_blue_from_B := BagB.blue.to_rat / (BagB.yellow + BagB.blue)
  let p_yellow_from_C := BagC.yellow.to_rat / (BagC.yellow + BagC.blue)
  
  let p_yellow_from_B_after_blue := p_blue_from_B * p_yellow_from_B
  
  let prob_yellow :=
    p_white * p_yellow_from_B +
    p_white * p_yellow_from_B_after_blue +
    p_black * p_yellow_from_C
  prob_yellow

theorem probability_of_last_yellow_marble :
  probability_last_marble_yellow = 73 / 140 :=
by
  -- sorry will be replaced with the full proof.
  sorry

end probability_of_last_yellow_marble_l219_219834


namespace seq_a22_l219_219241

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l219_219241


namespace quadrilateral_MNPQ_is_rectangle_l219_219081

structure Point :=
  (x : ℝ)
  (y : ℝ)

def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

def is_rectangle (M N P Q : Point) : Prop :=
  let k_MN := slope M N
  let k_PQ := slope P Q
  let k_MQ := slope M Q
  let k_PN := slope P N
  k_MN = k_PQ ∧
  k_MQ = k_PN ∧
  k_MN * k_MQ = -1

theorem quadrilateral_MNPQ_is_rectangle :
  let M := Point.mk 1 1
  let N := Point.mk 3 (-1)
  let P := Point.mk 4 0
  let Q := Point.mk 2 2
  is_rectangle M N P Q :=
by
  let M := Point.mk 1 1
  let N := Point.mk 3 (-1)
  let P := Point.mk 4 0
  let Q := Point.mk 2 2
  sorry

end quadrilateral_MNPQ_is_rectangle_l219_219081


namespace failed_in_english_l219_219978

/- Lean definitions and statement -/

def total_percentage := 100
def failed_H := 32
def failed_H_and_E := 12
def passed_H_or_E := 24

theorem failed_in_english (total_percentage failed_H failed_H_and_E passed_H_or_E : ℕ) (h1 : total_percentage = 100) (h2 : failed_H = 32) (h3 : failed_H_and_E = 12) (h4 : passed_H_or_E = 24) :
  total_percentage - (failed_H + (total_percentage - passed_H_or_E - failed_H_and_E)) = 56 :=
by sorry

end failed_in_english_l219_219978


namespace total_order_cost_l219_219856

theorem total_order_cost :
  let c := 2 * 30
  let w := 9 * 15
  let s := 50
  c + w + s = 245 := 
by
  linarith

end total_order_cost_l219_219856


namespace convex_combination_le_l219_219161

-- Define the function f and the interval I as conditions
variable (f : ℝ → ℝ)
variable (I : Set ℝ)
variable (h_convex : ∀ x₁ x₂ ∈ I, ∀ t ∈ Icc (0 : ℝ) 1, f((1 - t) * x₁ + t * x₂) ≤ (1 - t) * f(x₁) + t * f(x₂))

-- Given variables
variable (n : ℕ) (n_ge_two : n ≥ 2)
variable (x : Fin n → ℝ) (x_in_I : ∀ i, x i ∈ I)
variable (p : Fin n → ℝ) (p_nonneg : ∀ i, 0 ≤ p i) (p_sum_one : ∑ i in Finset.univ, p i = 1)

-- The theorem statement
theorem convex_combination_le :
  f (∑ i in Finset.univ, p i * x i) ≤ ∑ i in Finset.univ, p i * f (x i) := 
sorry

end convex_combination_le_l219_219161


namespace min_value_QS_ST_TR_l219_219100

noncomputable def minimum_path_length (P Q R S T R' : Point) (PQ PR : ℝ) (anglePQR : ℝ)
  (hPQ : PQ = 8) (hPR : PR = 12) (hAnglePQR : anglePQR = 30) (hPQ_S : S ∈ line_segment P Q)
  (hPR_T : T ∈ line_segment P R) (hReflection : R' = reflect R line_segment P Q)
  (hCollinear : collinear {Q, S, T, R'}) : ℝ := 
  QS + ST + TR

theorem min_value_QS_ST_TR (P Q R S T R' : Point) (PQ PR : ℝ) (anglePQR : ℝ)
  (hPQ : PQ = 8) (hPR : PR = 12) (hAnglePQR : anglePQR = 30) 
  (hPQ_S : S ∈ line_segment P Q) (hPR_T : T ∈ line_segment P R) 
  (hReflection : R' = reflect R line_segment P Q) 
  (hCollinear : collinear {Q, S, T, R'}) :
  minimum_path_length P Q R S T R' PQ PR anglePQR hPQ hPR hAnglePQR hPQ_S hPR_T hReflection hCollinear = sqrt (208 + 96 * sqrt 3) := sorry

end min_value_QS_ST_TR_l219_219100


namespace gcd_n4_plus_16_n_plus_3_eq_1_l219_219020

theorem gcd_n4_plus_16_n_plus_3_eq_1 (n : ℕ) (h : n > 16) : gcd (n^4 + 16) (n + 3) = 1 := 
sorry

end gcd_n4_plus_16_n_plus_3_eq_1_l219_219020


namespace angle_terminal_side_on_non_negative_y_axis_l219_219525

theorem angle_terminal_side_on_non_negative_y_axis (P : ℝ × ℝ) (α : ℝ) (hP : P = (0, 3)) :
  α = some_angle_with_terminal_side_on_non_negative_y_axis := by
  sorry

end angle_terminal_side_on_non_negative_y_axis_l219_219525


namespace circle_chairs_subsets_count_l219_219743

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219743


namespace parallel_vectors_cosine_identity_l219_219082

-- Defining the problem in Lean 4

theorem parallel_vectors_cosine_identity :
  ∀ α : ℝ, (∃ k : ℝ, (1 / 3, Real.tan α) = (k * Real.cos α, k)) →
  Real.cos (Real.pi / 2 + α) = -1 / 3 :=
by
  sorry

end parallel_vectors_cosine_identity_l219_219082


namespace correct_propositions_l219_219437

theorem correct_propositions :
  let p1 := ∃ x0 > 0, log x0 > x0 - 1
  let p2 := ∀ x : ℝ, x^2 - x + 1 > 0
  let p3 := ∃ x0 > 0, log (1 / x0) > - x0 + 1
  let p4 := ∀ x : ℝ, (0 < x) → ((1 / 2) ^ x > real.log x / real.log 2)
  (¬ p1) ∧ p2 ∧ p3 ∧ (¬ p4) :=
begin
  -- conditions p1, p2, p3, p4 are declared
  -- we are to prove the combination ¬p1 ∧ p2 ∧ p3 ∧ ¬p4
  sorry
end

end correct_propositions_l219_219437


namespace find_x_l219_219801

variable (x : ℝ)

theorem find_x (h : 0.60 * x = (1/3) * x + 110) : x = 412.5 :=
sorry

end find_x_l219_219801


namespace smallest_integer_with_remainder_3_l219_219776

/-- The smallest positive integer greater than 1 that leaves a remainder of 3 when divided by each of 4, 5, 6, 7, and 8 is 843. -/
theorem smallest_integer_with_remainder_3 : 
  ∃ m : ℕ, m > 1 ∧ (∀ n ∈ ({4, 5, 6, 7, 8} : Finset ℕ), (m % n = 3)) ∧ m = 843 :=
begin
  sorry
end

end smallest_integer_with_remainder_3_l219_219776


namespace smallest_positive_period_of_f_max_min_values_of_f_intervals_of_monotonicity_of_f_symmetry_of_f_l219_219068

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (cos x)^2 + (1/2) * sin (2 * x)

theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
  sorry

theorem max_min_values_of_f : ∃ (max min : ℝ), (∀ x ∈ Icc (-π/6) (π/4), f x ≤ max) ∧
  (∀ x ∈ Icc (-π/6) (π/4), f x ≥ min) ∧ 
  max = 1 + sqrt 3 / 2 ∧ min = sqrt 3 / 2 :=
  sorry

theorem intervals_of_monotonicity_of_f : 
  (∀ k ∈ ℤ, ∀ x ∈ Icc ((k : ℝ) * π - 5 * π / 12) ((k : ℝ) * π + π / 12), f.deriv x > 0) ∧
  (∀ k ∈ ℤ, ∀ x ∈ Icc ((k : ℝ) * π + π / 12) ((k : ℝ) * π + 7 * π / 12), f.deriv x < 0) :=
  sorry

theorem symmetry_of_f :
  (∀ k ∈ ℤ, ∃ (axis_x : ℝ), axis_x = (1/2 : ℝ) * (k : ℝ) * π + π / 12 ∧ f axis_x = f ((axis_x - (1/2 : ℝ) * π))) ∧
  (∀ k ∈ ℤ, ∃ (center_x : ℝ), center_x = (1/2 : ℝ) * (k : ℝ) * π - π / 6 ∧ f center_x = f (center_x + π)) :=
  sorry

end smallest_positive_period_of_f_max_min_values_of_f_intervals_of_monotonicity_of_f_symmetry_of_f_l219_219068


namespace project_within_budget_l219_219420

def area : ℝ := 3136
def cost_per_meter : ℝ := 1.10
def gate_width : ℝ := 1
def gate_weight : ℝ := 25
def cost_per_kg : ℝ := 350
def labor_charge_per_day : ℝ := 1500
def days : ℝ := 2
def budget : ℝ := 10000

noncomputable def total_cost : ℝ :=
  let side_length := real.sqrt area
  let perimeter := 4 * side_length
  let barbed_wire_length := perimeter - 2 * gate_width
  let cost_barbed_wire := barbed_wire_length * cost_per_meter
  let total_iron_weight := 2 * gate_weight
  let cost_iron_gates := total_iron_weight * cost_per_kg
  let labor_charges := days * labor_charge_per_day
  cost_barbed_wire + cost_iron_gates + labor_charges

theorem project_within_budget : total_cost <= budget := by
  sorry

end project_within_budget_l219_219420


namespace right_triangle_AC_length_l219_219125

theorem right_triangle_AC_length
  (ABC : Triangle)
  (right_angle_A : ABC.rightAngle A)
  (altitude_AH : ABC.altitude A H)
  (circle_AH : ∃ c : Circle, c.passesThrough A ∧ c.passesThrough H)
  (AX : 5)
  (AY : 6)
  (AB : 9) :
  AC = 13.5 :=
begin
  -- sorry
  sorry
end

end right_triangle_AC_length_l219_219125


namespace sum_prime_factors_143_l219_219360

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219360


namespace range_sub_x_is_interval_l219_219684

theorem range_sub_x_is_interval (f : ℝ → ℝ) 
  (h1 : ∀ x, -3 ≤ x ∧ x ≤ 3 → f(x) = x)
  (h2 : ∀ x ∈ ℚ, x ∉ {-2, -1, 0, 1, 2, 3} → f(x) = x) : 
  set.range (λ x, f x - x) = {y : ℝ | -1 < y ∧ y ≤ 0} :=
sorry

end range_sub_x_is_interval_l219_219684


namespace distinct_finite_subsets_or_rational_lt_one_l219_219897

theorem distinct_finite_subsets_or_rational_lt_one (S : set ℕ) (hS : ∀ s ∈ S, 0 < s) :
  (∃ (F G : finset ℕ), F ≠ G ∧ F ⊆ S ∧ G ⊆ S ∧ (F.sum (λ x, 1 / (x : ℝ)) = G.sum (λ x, 1 / (x : ℝ)))) ∨
  (∃ (r : ℚ), r > 0 ∧ r < 1 ∧ ∀ (F : finset ℕ), F ⊆ S → (F.sum (λ x, 1 / (x : ℝ)) : ℚ) ≠ r) := sorry

end distinct_finite_subsets_or_rational_lt_one_l219_219897


namespace number_of_integers_divisible_by_18_or_21_but_not_both_l219_219088

theorem number_of_integers_divisible_by_18_or_21_but_not_both :
  let num_less_2019_div_by_18 := 112
  let num_less_2019_div_by_21 := 96
  let num_less_2019_div_by_both := 16
  num_less_2019_div_by_18 + num_less_2019_div_by_21 - 2 * num_less_2019_div_by_both = 176 :=
by
  sorry

end number_of_integers_divisible_by_18_or_21_but_not_both_l219_219088


namespace M_plus_N_eq_64_l219_219152

theorem M_plus_N_eq_64 : 
  let S := {1, 2, 3, 4, 6}
  let permutations := Multiset.permutations S
  let products := permutations.map (λ l, l[0] * l[1] + l[1] * l[2] + l[2] * l[3] + l[3] * l[4] + l[4] * l[0])
  let maxM := Multiset.max products
  let countN := products.count maxM
  maxM + countN = 64 := 
by
  sorry

end M_plus_N_eq_64_l219_219152


namespace trivia_team_points_l219_219825

theorem trivia_team_points : 
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  (member1_points + member2_points + member3_points + member4_points + member5_points + member6_points + member7_points + member8_points) = 76 :=
by
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  sorry

end trivia_team_points_l219_219825


namespace John_total_weekly_consumption_l219_219140

/-
  Prove that John's total weekly consumption of water, milk, and juice in quarts is 49.25 quarts, 
  given the specified conditions on his daily and periodic consumption.
-/

def John_consumption_problem (gallons_per_day : ℝ) (pints_every_other_day : ℝ) (ounces_every_third_day : ℝ) 
  (quarts_per_gallon : ℝ) (quarts_per_pint : ℝ) (quarts_per_ounce : ℝ) : ℝ :=
  let water_per_day := gallons_per_day * quarts_per_gallon
  let water_per_week := water_per_day * 7
  let milk_per_other_day := pints_every_other_day * quarts_per_pint
  let milk_per_week := milk_per_other_day * 4 -- assuming he drinks milk 4 times a week
  let juice_per_third_day := ounces_every_third_day * quarts_per_ounce
  let juice_per_week := juice_per_third_day * 2 -- assuming he drinks juice 2 times a week
  water_per_week + milk_per_week + juice_per_week

theorem John_total_weekly_consumption :
  John_consumption_problem 1.5 3 20 4 (1/2) (1/32) = 49.25 :=
by
  sorry

end John_total_weekly_consumption_l219_219140


namespace Q_gets_less_than_P_l219_219091

theorem Q_gets_less_than_P (x : Real) (hx : x > 0) (hP : P = 1.25 * x): 
  Q = P * 0.8 := 
sorry

end Q_gets_less_than_P_l219_219091


namespace calculate_expression_l219_219837

theorem calculate_expression : (235 - 2 * 3 * 5) * 7 / 5 = 287 := 
by
  sorry

end calculate_expression_l219_219837


namespace sequence_property_l219_219233

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l219_219233


namespace count_distinct_values_l219_219162

def g (x : ℝ) : ℝ :=
  if x ≥ -3 then x^2 - 3
  else if -6 ≤ x then x + 4
  else -2 * x

theorem count_distinct_values (x_vals : finset ℝ) (h : ∀ x ∈ x_vals, g (g x) = 4 ) :
  x_vals.card = 4 :=
proof
  sorry

end count_distinct_values_l219_219162


namespace equidistant_point_x_axis_l219_219771

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end equidistant_point_x_axis_l219_219771


namespace f_at_pi_l219_219915

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + Real.pi / 6)

theorem f_at_pi (ω : ℝ) (h1 : ω > 0)
  (h2 : ∀ x y, Real.pi < x ∧ x < y ∧ y < 2 * Real.pi → f(ω)(x) ≥ f(ω)(y))
  (h3 : ∀ x y, 2 * Real.pi < x ∧ x < y ∧ y < 3 * Real.pi → f(ω)(x) ≤ f(ω)(y)) :
  f(ω)(Real.pi) = 1 :=
sorry

end f_at_pi_l219_219915


namespace sequence_solution_l219_219074

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, n > 0 → (n + 1) * a (n + 1) = a n + n

theorem sequence_solution (a : ℕ → ℝ) (h : sequence a) :
  ∀ n : ℕ, n > 0 → a n = 1 / (n.factorial) + 1 :=
begin
  sorry
end

end sequence_solution_l219_219074


namespace probability_one_white_one_black_two_touches_l219_219565

def probability_white_ball : ℚ := 7 / 10
def probability_black_ball : ℚ := 3 / 10

theorem probability_one_white_one_black_two_touches :
  (probability_white_ball * probability_black_ball) + (probability_black_ball * probability_white_ball) = (7 / 10) * (3 / 10) + (3 / 10) * (7 / 10) :=
by
  -- The proof is omitted here.
  sorry

end probability_one_white_one_black_two_touches_l219_219565


namespace relationship_among_abc_l219_219519

-- Define the function f(x) given the conditions
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Math.log x / Math.log 10
  else if x ≤ 0 then f (-x)
  else f (x - 2 * Float.floor (x / 2))

-- Define values a, b, c using the function f
def a : ℝ := f (6 / 5)
def b : ℝ := f (3 / 2)
def c : ℝ := f (5 / 2)

-- Prove the relationship among a, b, and c
theorem relationship_among_abc : c < a ∧ a < b := sorry

end relationship_among_abc_l219_219519


namespace tangent_line_through_A_l219_219529

-- Definitions of the curve and the point
def curve (x : ℝ) : ℝ := x^3 + 4
def point_A := (1 : ℝ, 5 : ℝ)

-- Tangent line equations to be proven
def tangent_line_equation_1 (x y : ℝ) : Prop := 3*x - y - 2 = 0
def tangent_line_equation_2 (x y : ℝ) : Prop := 3*x - 4*y + 17 = 0

-- The statement of the theorem in Lean
theorem tangent_line_through_A :
  tangent_line_equation_1 1 5 ∨ tangent_line_equation_2 1 5 :=
sorry

end tangent_line_through_A_l219_219529


namespace y_n_squared_eq_three_x_n_squared_plus_one_l219_219832

open Int

def x_sequence : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 4 * x_sequence (n+1) - x_sequence n

def y_sequence : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := 4 * y_sequence (n+1) - y_sequence n

theorem y_n_squared_eq_three_x_n_squared_plus_one : ∀ n : ℕ, y_sequence n ^ 2 = 3 * (x_sequence n ^ 2) + 1 := by
  sorry

end y_n_squared_eq_three_x_n_squared_plus_one_l219_219832


namespace decreasing_f_on_interval_range_of_y_l219_219196

-- Part 1: Proving that f(x) = x + 2/x is decreasing on (0, sqrt(2))
theorem decreasing_f_on_interval : 
  ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < real.sqrt 2) ∧ (0 < x2 ∧ x2 < real.sqrt 2) ∧ x1 < x2 → (x1 + 2/x1) > (x2 + 2/x2) :=
by sorry

-- Part 2: Determining the range of y = (2(x^2 + x)) / (x - 1) for 2 ≤ x < 4
theorem range_of_y : 
  ∀ y : ℝ, ∃ x : ℝ, 2 ≤ x ∧ x < 4 ∧ y = (2 * (x^2 + x)) / (x - 1) 
  ↔ y ∈ set.Icc (2 * (real.sqrt 2 + 2) + 6) (40 / 3) :=
by sorry

end decreasing_f_on_interval_range_of_y_l219_219196


namespace angle_AOB_eq_pi_over_two_l219_219905

variables {V : Type*} [inner_product_space ℝ V] {O A B C : V}

-- Hypotheses
axiom circumcircle_radius_one : ∥O - A∥ = 1 ∧ ∥O - B∥ = 1 ∧ ∥O - C∥ = 1
axiom vector_sum_zero : 3 • (O - A) + 4 • (O - B) + 5 • (O - C) = 0

-- The theorem to prove
theorem angle_AOB_eq_pi_over_two : ∠(O - A) (O - B) = real.pi / 2 :=
sorry

end angle_AOB_eq_pi_over_two_l219_219905


namespace sum_of_prime_factors_143_l219_219295

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219295


namespace number_of_female_students_l219_219383

theorem number_of_female_students
    (F : ℕ)  -- Number of female students
    (avg_all : ℝ)  -- Average score for all students
    (avg_male : ℝ)  -- Average score for male students
    (avg_female : ℝ)  -- Average score for female students
    (num_male : ℕ)  -- Number of male students
    (h_avg_all : avg_all = 90)
    (h_avg_male : avg_male = 82)
    (h_avg_female : avg_female = 92)
    (h_num_male : num_male = 8)
    (h_avg : avg_all * (num_male + F) = avg_male * num_male + avg_female * F) :
  F = 32 :=
by
  sorry

end number_of_female_students_l219_219383


namespace charming_number_unique_l219_219467

def is_charming (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = 2 * a + b^3

theorem charming_number_unique : ∃! n, 10 ≤ n ∧ n ≤ 99 ∧ is_charming n := by
  sorry

end charming_number_unique_l219_219467


namespace gardener_can_ensure_majestic_trees_l219_219580

-- Defining the conditions of the problem
def is_majestic (height : ℕ) := height ≥ 10^6
def grid_size : ℕ := 2022

-- Predicted number of majestic trees
def K : ℚ := (5 / 9) * (grid_size ^ 2)

-- The Lean statement to prove
theorem gardener_can_ensure_majestic_trees :
  ∃ K ≥ ((5 : ℚ) / 9) * (grid_size ^ 2), ∀ moves : list (list (ℕ × ℕ)), -- sequence of chosen squares,
    -- add conditions or implementation as needed
    sorry

end gardener_can_ensure_majestic_trees_l219_219580


namespace smallest_palindrome_greater_than_10_proof_l219_219844

-- Helper functions to check if a number is a palindrome in a given base
def to_base (b n : ℕ) : List ℕ :=
  if h : b > 1 then
    let rec to_base_aux (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc else to_base_aux (n / b) ((n % b) :: acc)
    to_base_aux n []
  else
    []

def is_palindrome {α : Type _} [DecidableEq α] (l : List α) : Bool :=
  l = l.reverse

def is_palindrome_base (b n : ℕ) : Bool :=
  is_palindrome (to_base b n)

noncomputable def smallest_palindrome_greater_than_10 : ℕ :=
  if h : ∃ n > 10, is_palindrome_base 2 n ∧ is_palindrome_base 3 n then
    Nat.find h
  else
    0

theorem smallest_palindrome_greater_than_10_proof :
  smallest_palindrome_greater_than_10 = 585 :=
by
  -- Proof is omitted
  sorry

end smallest_palindrome_greater_than_10_proof_l219_219844


namespace sum_of_roots_f_zero_l219_219165

def f (x : ℝ) : ℝ :=
if x ≤ 2 then -x^2 - 4*x - 4 else x^2 / 4 - 1

theorem sum_of_roots_f_zero : 
  (∑ x in {x : ℝ | f x = 0}.to_finset, x) = -2 :=
sorry

end sum_of_roots_f_zero_l219_219165


namespace committee_count_is_252_l219_219583

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l219_219583


namespace minimum_value_of_f_on_interval_l219_219532

noncomputable def f (a x : ℝ) := Real.log x + a * x

theorem minimum_value_of_f_on_interval (a : ℝ) (h : a < 0) :
  ( ( -Real.log 2 ≤ a ∧ a < 0 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ a ) ∧
    ( a < -Real.log 2 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ (Real.log 2 + 2 * a) )
  ) :=
by
  sorry

end minimum_value_of_f_on_interval_l219_219532


namespace compare_sqrt_expression_l219_219842

theorem compare_sqrt_expression : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := 
sorry

end compare_sqrt_expression_l219_219842


namespace hyperbola_vertices_distance_l219_219013

noncomputable def distance_between_vertices : ℝ :=
  2 * Real.sqrt 7.5

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), 4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0 →
  distance_between_vertices = 2 * Real.sqrt 7.5 :=
by sorry

end hyperbola_vertices_distance_l219_219013


namespace find_angle_B_max_area_triangle_l219_219600

noncomputable def given_conditions (a b c : ℝ) (A B C : ℝ) := a / (cos C * sin B) = (b * cos C + c * sin B) / (sin B * cos C)

theorem find_angle_B {a b c A B C : ℝ} (h : given_conditions a b c A B C) : 
  B = π / 4 :=
sorry

theorem max_area_triangle {a b c A B C : ℝ} (h : given_conditions a b c A B C) (hb : b = sqrt 2) (hB : B = π / 4) :
  let S : ℝ := (1 / 2) * a * c * sin B
  in S ≤ (sqrt 2 + 1) / 2 :=
sorry

end find_angle_B_max_area_triangle_l219_219600


namespace sequence_property_l219_219235

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l219_219235


namespace range_of_a_l219_219518

noncomputable def p (a : ℝ) : Prop :=
  let A : ℝ × ℝ × ℝ := (-2, -2 * a, 7)
  let B : ℝ × ℝ × ℝ := (a + 1, a + 4, 2)
  let distance := real.sqrt (((a + 3)^2 + (3 * a + 4)^2 + (-5)^2))
  distance < 3 * real.sqrt 10

noncomputable def q (a : ℝ) : Prop :=
  let M : ℝ × ℝ := (a^2 / 4, a)
  let distance := real.sqrt (((a^2 / 4 + 1 - 0)^2 + (a - 0)^2))
  distance > 2

-- Now we need to prove that the range of values for \(a\) such that p(a) is true and q(a) is false, is [-2, 1].
theorem range_of_a : {a : ℝ | p a ∧ ¬q a} = set.Icc (-2 : ℝ) (1 : ℝ) := 
sorry

end range_of_a_l219_219518


namespace probability_no_neighbouring_same_color_l219_219030

-- Given conditions
def red_beads : ℕ := 4
def white_beads : ℕ := 2
def blue_beads : ℕ := 2
def total_beads : ℕ := red_beads + white_beads + blue_beads

-- Total permutations
def total_orderings : ℕ := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- Probability calculation proof
theorem probability_no_neighbouring_same_color : (30 / 420 : ℚ) = (1 / 14 : ℚ) :=
by
  -- proof steps
  sorry

end probability_no_neighbouring_same_color_l219_219030


namespace no_integer_solutions_l219_219195

theorem no_integer_solutions (k l n : ℤ) (hlk : k > l) (hl0 : l ≥ 0) : 4 ^ k - 4 ^ l ≠ 10 ^ n := by
  sorry

end no_integer_solutions_l219_219195


namespace units_digit_n_l219_219865

theorem units_digit_n (m n : ℕ) (h1 : m * n = 14^5) (h2 : m % 10 = 8) : n % 10 = 3 :=
sorry

end units_digit_n_l219_219865


namespace average_of_values_l219_219849

theorem average_of_values (z : ℝ) : 
  (0 + 1 + 2 + 4 + 8 + 32 : ℝ) * z / (6 : ℝ) = 47 * z / 6 :=
by
  sorry

end average_of_values_l219_219849


namespace minimum_value_parabola_l219_219159

theorem minimum_value_parabola (p : ℝ) (hp : 0 < p) (x_A y_A x_B y_B : ℝ) 
    (hA : y_A^2 = 2 * p * x_A) (hB : y_B^2 = 2 * p * x_B):
    ∃ A B : ℝ, (y_A ≠ y_B) → 
    (let OA := (x_A + x_B)^2 + (y_A + y_B)^2
         AB := (x_A - x_B)^2 + (y_A - y_B)^2 
     in OA - AB = -4 * p ^ 2) :=
begin
  sorry
end

end minimum_value_parabola_l219_219159


namespace katrina_cookies_left_l219_219611

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l219_219611


namespace inequality_350_l219_219659

theorem inequality_350 (a b c d : ℝ) : 
  (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 :=
by
  sorry

end inequality_350_l219_219659


namespace ab_perpendicular_cd_l219_219922

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assuming points are members of a metric space and distances are calculated using the distance function
variables (a b c d : A)

-- Given condition
def given_condition : Prop := 
  dist a c ^ 2 + dist b d ^ 2 = dist a d ^ 2 + dist b c ^ 2

-- Statement that needs to be proven
theorem ab_perpendicular_cd (h : given_condition a b c d) : dist a b * dist c d = 0 :=
sorry

end ab_perpendicular_cd_l219_219922


namespace mary_saw_total_snakes_l219_219178

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l219_219178


namespace equidistant_x_coordinate_l219_219769

open Real

-- Definitions for points C and D
def C : ℝ × ℝ := (-3, 0)
def D : ℝ × ℝ := (0, 5)

-- Definition for the distance function on the plane
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The x-coordinate of the point that is equidistant from C and D
theorem equidistant_x_coordinate : ∃ x : ℝ, dist (x, 0) C = dist (x, 0) D ∧ x = 8/3 :=
by
  let x := 8/3
  have h1 : dist (x, 0) C = sqrt ((-3 - x)^2 + 0^2),
    simp only [C],
  have h2 : dist (x, 0) D = sqrt ((0 - x)^2 + (-5)^2),
    simp only [D],
  use x,
  split,
  {
    rw [dist, dist, h1, h2],
    sorry -- Proof steps omitted
  },
  {
    refl,
  }

end equidistant_x_coordinate_l219_219769


namespace range_of_a_l219_219064

-- Definitions
def domain_f : Set ℝ := {x : ℝ | x ≤ -4 ∨ x ≥ 4}
def range_g (a : ℝ) : Set ℝ := {y : ℝ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ y = x^2 - 2*x + a}

-- Theorem to prove the range of values for a
theorem range_of_a :
  (∀ x : ℝ, x ∈ domain_f ∨ (∃ y : ℝ, ∃ a : ℝ, y ∈ range_g a ∧ x = y)) ↔ (-4 ≤ a ∧ a ≤ -3) :=
sorry

end range_of_a_l219_219064


namespace sin_double_angle_l219_219597

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α such that its terminal side passes through point P
noncomputable def α : ℝ := sorry -- The exact definition of α is not needed for this statement

-- Define r as the distance from the origin to the point P
noncomputable def r : ℝ := Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

-- Define sin(α) and cos(α)
noncomputable def sin_α : ℝ := P.2 / r
noncomputable def cos_α : ℝ := P.1 / r

-- The proof statement
theorem sin_double_angle : 2 * sin_α * cos_α = -4 / 5 := by
  sorry

end sin_double_angle_l219_219597


namespace translate_and_find_quadrant_l219_219984

def point_translated_quadrant (P : ℝ × ℝ) (dy dx : ℝ) : ℕ :=
  let P' := (P.1 + dx, P.2 - dy)
  if P'.1 > 0 ∧ P'.2 < 0 then 4 else 0

theorem translate_and_find_quadrant : 
  point_translated_quadrant (-2, 3) 4 5 = 4 := 
sorry

end translate_and_find_quadrant_l219_219984


namespace quadratic_ineq_l219_219632

variable (n : ℕ)
variable (m : Fin n → ℝ)
variable (a b c : Fin n → ℝ)

axiom mj_pos {j : Fin n} : 0 < m j
axiom abc_order {j : Fin n} : a j ≤ b j ∧ b j ≤ c j
axiom a_ascending {i j : Fin n} (hij : i ≤ j) : a i ≤ a j
axiom b_ascending {i j : Fin n} (hij : i ≤ j) : b i ≤ b j
axiom c_ascending {i j : Fin n} (hij : i ≤ j) : c i ≤ c j
axiom strict_abc {j : Fin n} : a j < b j ∧ b j < c j

theorem quadratic_ineq (n : ℕ) (m : Fin n → ℝ) (a b c : Fin n → ℝ)
  (mj_pos : ∀ j, 0 < m j)
  (abc_order : ∀ j, a j ≤ b j ∧ b j ≤ c j)
  (a_ascending : ∀ i j, i ≤ j → a i ≤ a j)
  (b_ascending : ∀ i j, i ≤ j → b i ≤ b j)
  (c_ascending : ∀ i j, i ≤ j → c i ≤ c j)
  (strict_abc : ∀ j, a j < b j ∧ b j < c j) :
  (∑ j, m j * (a j + b j + c j))^2 > 
  3 * (∑ j, m j) * (∑ j, m j * (a j * b j + b j * c j + c j * a j)) :=
sorry

end quadratic_ineq_l219_219632


namespace max_distance_curve_to_line_product_distances_to_M_l219_219395

section CoordinateSystemAndParametricEquations

variables {α : ℝ}

def curve (α : ℝ) : ℝ × ℝ := (√3 * Real.cos α, Real.sin α)

def line (x y : ℝ) : Prop := x - y - 6 = 0

def line₁ (x y : ℝ) : Prop := x + y = -1

def point_M := (-1 : ℝ, 0 : ℝ)

theorem max_distance_curve_to_line :
  ∃ (P : ℝ × ℝ), P = curve (5 * Real.pi / 6) ∧ ∀ α, 
  (let P' := curve α in dist P' (line P'.1 P'.2) ≤ 4 * √2) ∧
  dist (curve (5 * Real.pi / 6)) (line (curve (5 * Real.pi / 6)).1 (curve (5 * Real.pi / 6)).2) = 4 * √2 :=
sorry

theorem product_distances_to_M :
  ∃ A B : ℝ × ℝ, A ∈ curve ∧ B ∈ curve ∧ 
  line₁ A.1 A.2 ∧ line₁ B.1 B.2 ∧
  (let dA := dist point_M A in let dB := dist point_M B in dA * dB = 2) :=
sorry

end CoordinateSystemAndParametricEquations

end max_distance_curve_to_line_product_distances_to_M_l219_219395


namespace standard_deviation_range_l219_219213

theorem standard_deviation_range 
  (average_age : ℕ) 
  (std_dev : ℕ) 
  (num_ages : ℕ) 
  (h_average : average_age = 10) 
  (h_std_dev : std_dev = 8) 
  (h_num_ages : num_ages = 17) : 
  ∃ k : ℕ, (10 + k * 8 - (10 - k * 8) + 1 = 17) ∧ k = 1 :=
by {
  use 1,
  split,
  { rw [h_average, h_std_dev],
    norm_num },
  { norm_num }
}

end standard_deviation_range_l219_219213


namespace prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l219_219189

noncomputable def prob_A_wins_game := 2 / 3
noncomputable def prob_B_wins_game := 1 / 3

/-- The probability that the match ends after two games with player A's victory is 4/9. -/
theorem prob_match_ends_two_games_A_wins :
  prob_A_wins_game * prob_A_wins_game = 4 / 9 := by
  sorry

/-- The probability that the match ends exactly after four games is 20/81. -/
theorem prob_match_ends_four_games :
  2 * prob_A_wins_game * prob_B_wins_game * (prob_A_wins_game^2 + prob_B_wins_game^2) = 20 / 81 := by
  sorry

/-- The probability that player A wins the match overall is 74/81. -/
theorem prob_A_wins_overall :
  (prob_A_wins_game^2 + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game^2
  + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game * prob_B_wins_game) / (prob_A_wins_game + prob_B_wins_game) = 74 / 81 := by
  sorry

end prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l219_219189


namespace angle_EAF_eq_arctan_three_halves_k_l219_219231

theorem angle_EAF_eq_arctan_three_halves_k
  (ABCD : Type)
  (x y : ℝ) -- width and length of the rectangle
  (k : ℝ)
  (E F : ℝ × ℝ) -- midpoints of BC and CD respectively
  (hE : E = (y / 2, 0))
  (hF : F = (x, x / 2))
  (ratio_area_diagonal_square : (x * y) / (x ^ 2 + y ^ 2) = k) :
  ∠EAF = Real.arctan (3 * k / 2) := 
  by sorry

end angle_EAF_eq_arctan_three_halves_k_l219_219231


namespace largest_d_for_g_of_minus5_l219_219015

theorem largest_d_for_g_of_minus5 (d : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + d = -5) → d ≤ -4 :=
by
-- Proof steps will be inserted here
sorry

end largest_d_for_g_of_minus5_l219_219015


namespace lamp_turn_off_ways_l219_219267

theorem lamp_turn_off_ways : 
  ∃ (ways : ℕ), ways = 10 ∧
  (∃ (n : ℕ) (m : ℕ), 
    n = 6 ∧  -- 6 lamps in a row
    m = 2 ∧  -- turn off 2 of them
    ways = Nat.choose (n - m + 1) m) := -- 2 adjacent lamps cannot be turned off
by
  -- Proof will be provided here.
  sorry

end lamp_turn_off_ways_l219_219267


namespace rental_cost_equal_mileage_l219_219665

theorem rental_cost_equal_mileage :
  ∃ m : ℝ, 
    (21.95 + 0.19 * m = 18.95 + 0.21 * m) ∧ 
    m = 150 :=
by
  sorry

end rental_cost_equal_mileage_l219_219665


namespace count_valid_three_digit_numbers_l219_219939

def odd_digits : Set ℕ := {1, 3, 7, 9}

def is_not_divisible_by_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

def valid_3_digit_number (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  n >= 100 ∧ n < 1000 ∧
  (∀ d ∈ digits, d ∈ odd_digits) ∧
  (∀ i, i < 3 → is_not_divisible_by_5 (digits.erase_nth i).sum)

theorem count_valid_three_digit_numbers : {n : ℕ | valid_3_digit_number n}.to_finset.card = 64 :=
by
  sorry

end count_valid_three_digit_numbers_l219_219939


namespace car_miles_per_tankful_city_l219_219401

theorem car_miles_per_tankful_city :
  forall (miles_per_tankful_highway miles_per_gallon_city : ℕ),
  miles_per_tankful_highway = 462 →
  miles_per_gallon_city = 8 →
  (miles_per_gallon_city + 3) * (miles_per_tankful_highway / (miles_per_gallon_city + 3)) = 336 := 
by
  -- variables and conditions here
  intros miles_per_tankful_highway miles_per_gallon_city h1 h2,
  sorry -- placeholder for the proof

end car_miles_per_tankful_city_l219_219401


namespace total_snakes_count_l219_219174

-- Define the basic conditions
def breedingBalls : Nat := 3
def snakesPerBall : Nat := 8
def pairsOfSnakes : Nat := 6
def snakesPerPair : Nat := 2

-- Define the total number of snakes
theorem total_snakes_count : 
  (breedingBalls * snakesPerBall) + (pairsOfSnakes * snakesPerPair) = 36 := 
by 
  -- we skip the proof with sorry
  sorry

end total_snakes_count_l219_219174


namespace brownie_pieces_count_l219_219641

def area := (24: ℤ) * 30
def piece_area := (3: ℤ) * 4

theorem brownie_pieces_count : (area / piece_area) = 60 :=
by
  -- calculation is as per the given conditions
  have h_area : area = 720 := by norm_num
  have h_piece_area : piece_area = 12 := by norm_num
  have h_div : area / piece_area = 60 := by norm_num [h_area, h_piece_area]
  exact h_div

end brownie_pieces_count_l219_219641


namespace positive_integer_multiples_of_231_form_8j_minus_8i_l219_219087

theorem positive_integer_multiples_of_231_form_8j_minus_8i :
  ∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 99 → 231 ∣ (8^j - 8^i) → 
  ∑ (k : ℕ) in finset.range 17, 94 - 6 * k = 784 := 
by sorry

end positive_integer_multiples_of_231_form_8j_minus_8i_l219_219087


namespace value_of_x_l219_219364

theorem value_of_x : (2015^2 + 2015 - 1) / (2015 : ℝ) = 2016 - 1 / 2015 := 
  sorry

end value_of_x_l219_219364


namespace sum_divisible_by_120_l219_219712

-- Define the properties of the 120-digit number and permutations
theorem sum_divisible_by_120 
  (A : Fin 12 → ℕ) 
  (B : ℕ) 
  (S : ℕ := (∑ (i : Fin 120), A i) * 10 ^ 108 + 120 * B) :
  S % 120 = 0 := 
sorry

end sum_divisible_by_120_l219_219712


namespace gcd_combination_l219_219222

theorem gcd_combination (a b d : ℕ) (h : d = Nat.gcd a b) : 
  Nat.gcd (5 * a + 3 * b) (13 * a + 8 * b) = d := 
by
  sorry

end gcd_combination_l219_219222


namespace projection_matrix_inverse_l219_219153

def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let denom := a^2 + b^2
  (1 / denom) • (Matrix.vec_cons
    (Matrix.vec_cons (a^2) (Matrix.vec_cons (a*b) Matrix.vec_empty))
    (Matrix.vec_cons (a*b) (Matrix.vec_cons (b^2) Matrix.vec_empty)))

noncomputable def P : Matrix (Fin 2) (Fin 2) ℝ := projection_matrix 4 (-5)

theorem projection_matrix_inverse : ∀ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 → 
  let P := projection_matrix a b 
  let P_inv := if a^2 + b^2 = 0 then 0 else Matrix.inv P 
  P_inv = 0 :=
by
  intros a b h
  let P := projection_matrix a b
  have h_det : det P = 0 := sorry
  have h_inv : Matrix.inv P = 0 := sorry
  exact h_inv

end projection_matrix_inverse_l219_219153


namespace arithmetic_sequence_general_term_sum_sequence_proof_l219_219517

theorem arithmetic_sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d > 0)
  (h3 : a1 * (a1 + 3 * d) = 22)
  (h4 : 4 * a1 + 6 * d = 26) :
  ∀ n, a_n n = 3 * n - 1 := sorry

theorem sum_sequence_proof (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 3 * n - 1)
  (h2 : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)))
  (h3 : ∀ n, T_n n = (Finset.range n).sum b_n)
  (n : ℕ) :
  T_n n < 1 / 6 := sorry

end arithmetic_sequence_general_term_sum_sequence_proof_l219_219517


namespace binary_representation_of_19_l219_219003

theorem binary_representation_of_19 : nat.to_digits 2 19 = [1, 0, 0, 1, 1] :=
by
  sorry

end binary_representation_of_19_l219_219003


namespace hyperbola_decreasing_l219_219025

-- Defining the variables and inequalities
variables (x : ℝ) (m : ℝ)
hypothesis (hx : x > 0)

-- The statement we need to prove
theorem hyperbola_decreasing (hx : x > 0) : (∀ x, x > 0 → (λ (x : ℝ), (1 - m) / x) x > (λ (x' : ℝ), (1 - m) / (x + x')) x) ↔ m < 1 :=
begin
  sorry
end

end hyperbola_decreasing_l219_219025


namespace count_refined_visible_factor_numbers_l219_219814

def is_digit (n : ℕ) : Prop := n >= 0 ∧ n < 10

def is_unique_digits (n : ℕ) : Prop :=
  ∀ d1 d2, d1 ≠ d2 → d1 ∈ nat.digits 10 n → d2 ∈ nat.digits 10 n → (d1 ≠ d2)

def sum_digits (n : ℕ) : ℕ := (nat.digits 10 n).sum

def is_refined_visible_factor_number (n : ℕ) : Prop :=
  n ∈ (100..199) ∧
  is_unique_digits n ∧
  ∀ d ∈ nat.digits 10 n, d ≠ 0 ∧ n % d = 0 ∧ n % sum_digits n = 0

theorem count_refined_visible_factor_numbers : 
  cardinality {n | is_refined_visible_factor_number n} = 8 := 
sorry

end count_refined_visible_factor_numbers_l219_219814


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219750

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219750


namespace coefficient_x8_in_expansion_l219_219480

theorem coefficient_x8_in_expansion :
  (∀ (x : ℂ), 
    ((∑ (k1 k2 k3 : ℕ) in (finset.Ico 0 6).product (finset.Ico 0 6).product (finset.Ico 0 6), 
        k1 + k2 + k3 = 5 ∧ k2 + 2 * k3 = 8 →
        finset.card ((finset.range (k1 + k2 + k3 + 1)).attach) * 
        (binomial (k1 + k2 + k3) k1) * 
        (binomial (k1 + k2 + k3 - k1) k2) * 
        (1 : ℂ) ^ k1 * 
        (3 * x) ^ k2 * 
        (x ^ 2) ^ k3) = 90) :=
begin
  sorry
end

end coefficient_x8_in_expansion_l219_219480


namespace sum_prime_factors_of_143_l219_219338

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219338


namespace sum_prime_factors_143_l219_219326

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219326


namespace part_a_l219_219556

def f_X (X : Set (ℝ × ℝ)) (n : ℕ) : ℝ :=
  sorry  -- Placeholder for the largest possible area function

theorem part_a (X : Set (ℝ × ℝ)) (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m + 1) + f_X X (n - 1) :=
sorry

end part_a_l219_219556


namespace time_to_cross_platform_l219_219803

theorem time_to_cross_platform (train_length : ℝ) (platform_length : ℝ) (signal_time : ℝ) (given_time : ℝ) 
(train_length_eq : train_length = 300) (platform_length_eq : platform_length = 333.33) 
(signal_time_eq : signal_time = 18) : 
given_time = 38 :=
by
  -- Calculating speed of the train v = distance / time
  let speed := train_length / signal_time
  have speed_eq : speed = 300 / 18, from sorry -- Plugging the given values
  -- Calculating total distance to cross the platform.
  let distance := train_length + platform_length
  have distance_eq : distance = 633.33, from sorry -- Summing lengths of train and platform
  -- Calculating time to cross the platform t = distance / speed
  let time := distance / speed
  have time_eq : time = 633.33 / (300 / 18), from sorry -- Calculating time
  -- Check if calculated time is approximately 38 seconds
  have final_time : time ≈ given_time, from sorry -- Need an approximation check
  sorry

end time_to_cross_platform_l219_219803


namespace crayons_count_l219_219642

theorem crayons_count (l b f : ℕ) (h1 : l = b / 2) (h2 : b = 3 * f) (h3 : l = 27) : f = 18 :=
by
  sorry

end crayons_count_l219_219642


namespace odd_coefficients_count_l219_219616

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  let m := (Nat.toDigits 2 n).count 1
  2^m

theorem odd_coefficients_count (n : ℕ) (h : 0 < n) :
  let u_n (x : ℕ) := (x^2 + x + 1)^n in
  let m := (Nat.toDigits 2 n).count 1 in
  number_of_odd_coefficients n = 2^m :=
sorry

end odd_coefficients_count_l219_219616


namespace valid_pairings_l219_219839

-- Definitions based on the problem conditions
def bowls := {red, blue, yellow, green, purple}
def glasses := {red, blue, yellow, green}
def cannot_pair_same_color (bowl glass : String) : Prop := bowl ≠ glass 

-- The main theorem stating the problem and answer
theorem valid_pairings : 
  (∑ b in bowls.to_finset, ∑ g in glasses.to_finset, if cannot_pair_same_color b g then 1 else 0) = 16 :=
by
  sorry

end valid_pairings_l219_219839


namespace sum_of_areas_of_circles_l219_219262

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end sum_of_areas_of_circles_l219_219262


namespace find_k_inv_h_5_l219_219449

variable (h k : ℝ → ℝ)
variable (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)
hypothesis h_inv_def : ∀ x, h_inv (k x) = 3 * x - 4

theorem find_k_inv_h_5 : k_inv (h 5) = 3 :=
by
  sorry

end find_k_inv_h_5_l219_219449


namespace last_score_is_80_l219_219650

def is_integer (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ k : ℤ, n = k * d

def average_condition (scores : List ℤ) : Prop :=
  (∀ i in List.range scores.length, is_integer (scores.take i |>.sum) (i + 1))

theorem last_score_is_80 (scores : List ℤ) (h : scores = [71, 76, 80, 82, 91]) :
  average_condition (insert_nth 80 scores) :=
by
  sorry

end last_score_is_80_l219_219650


namespace triangle_inequality_l219_219202

variable {a b c S n : ℝ}

theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(habc : a + b > c) (habc' : a + c > b) (habc'' : b + c > a)
(hS : 2 * S = a + b + c) (hn : n ≥ 1) :
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ ((2 / 3)^(n - 2)) * S^(n - 1) :=
by
  sorry

end triangle_inequality_l219_219202


namespace sum_f_1_to_2017_l219_219913

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 3 * x)

theorem sum_f_1_to_2017 : 
  (∑ n in Finset.range 2017, f (n + 1)) = 1 / 2 :=
sorry

end sum_f_1_to_2017_l219_219913


namespace anna_reading_time_l219_219444

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l219_219444


namespace continuous_and_equal_of_conditions_l219_219150

noncomputable theory
open Classical

variables {f g : ℝ → ℝ}

-- Hypotheses
def inf_condition (a : ℝ) : Prop :=
  g(a) = ⨅ (x : ℝ) (hx : x > a), f(x)

def sup_condition (a : ℝ) : Prop :=
  f(a) = ⨆ (x : ℝ) (hx : x < a), g(x)

def darboux_property (f : ℝ → ℝ) : Prop :=
  ∀ (a b : ℝ) (h : a < b) (y : ℝ) (hya : f a ≤ y) (hyb : y ≤ f b), ∃ c ∈ set.Ioo a b, f c = y

-- The proof statement
theorem continuous_and_equal_of_conditions
  (hf_darboux : darboux_property f)
  (inf_cond : ∀ a : ℝ, inf_condition a)
  (sup_cond : ∀ a : ℝ, sup_condition a) :
  ∀ x : ℝ, f x = g x ∧ continuous f ∧ continuous g :=
sorry

end continuous_and_equal_of_conditions_l219_219150


namespace rate_of_current_l219_219375

variable (c t : ℝ)

-- Assume the man can row 4.2 km/hr in still water
def rowing_speed_still_water : ℝ := 4.2

--Assume it takes him twice as long to row upstream as to row downstream
def time_relation (downstream_time : ℝ) : Prop := 
  let upstream_time := 2 * downstream_time in
  rowing_speed_still_water * downstream_time = (rowing_speed_still_water - c) * upstream_time

theorem rate_of_current (H : time_relation t) : c = 1.4 := by
  sorry

end rate_of_current_l219_219375


namespace moles_of_MgCO3_formed_l219_219485

theorem moles_of_MgCO3_formed 
  (moles_MgO : ℕ) (moles_CO2 : ℕ)
  (h_eq : moles_MgO = 3 ∧ moles_CO2 = 3)
  (balanced_eq : ∀ n : ℕ, n * MgO + n * CO2 = n * MgCO3) : 
  moles_MgCO3 = 3 :=
by
  sorry

end moles_of_MgCO3_formed_l219_219485


namespace median_of_64_consecutive_integers_l219_219695

theorem median_of_64_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 64) (h2 : S = 8^4) :
  S / n = 64 :=
by
  -- to skip the proof
  sorry

end median_of_64_consecutive_integers_l219_219695


namespace race_course_length_l219_219784

variable (v d : ℝ)

theorem race_course_length (h1 : 4 * v > 0) (h2 : ∀ t : ℝ, t > 0 → (d / (4 * v)) = ((d - 72) / v)) : d = 96 := by
  sorry

end race_course_length_l219_219784


namespace jerry_needs_money_l219_219138

theorem jerry_needs_money 
  (current_count : ℕ) (total_needed : ℕ) (cost_per_action_figure : ℕ)
  (h1 : current_count = 7) 
  (h2 : total_needed = 16) 
  (h3 : cost_per_action_figure = 8) :
  (total_needed - current_count) * cost_per_action_figure = 72 :=
by sorry

end jerry_needs_money_l219_219138


namespace last_digit_2_to_2010_l219_219186

theorem last_digit_2_to_2010 : (2 ^ 2010) % 10 = 4 := 
by
  -- proofs and lemmas go here
  sorry

end last_digit_2_to_2010_l219_219186


namespace sereja_picked_more_berries_l219_219853

theorem sereja_picked_more_berries (total_berries : ℕ) (sereja_rate : ℕ) (dima_rate : ℕ)
  (sereja_pattern_input : ℕ) (sereja_pattern_eat : ℕ)
  (dima_pattern_input : ℕ) (dima_pattern_eat : ℕ)
  (rate_relation : sereja_rate = 2 * dima_rate)
  (total_berries_collected : sereja_rate + dima_rate = total_berries) :
  let sereja_collected := total_berries_collected * (sereja_pattern_input / (sereja_pattern_input + sereja_pattern_eat)),
      dima_collected := total_berries_collected * (dima_pattern_input / (dima_pattern_input + dima_pattern_eat)) in
  sereja_collected > dima_collected ∧ sereja_collected - dima_collected = 50 :=
by {
  sorry
}

end sereja_picked_more_berries_l219_219853


namespace total_snakes_l219_219181

/-
  Problem: Mary sees three breeding balls with 8 snakes each and 6 additional pairs of snakes.
           How many snakes did she see total?
  Conditions:
    - There are 3 breeding balls.
    - Each breeding ball has 8 snakes.
    - There are 6 additional pairs of snakes.
  Answer: 36 snakes
-/

theorem total_snakes (balls : ℕ) (snakes_per_ball : ℕ) (pairs : ℕ) (snakes_per_pair : ℕ) :
    balls = 3 → snakes_per_ball = 8 → pairs = 6 → snakes_per_pair = 2 →
    (balls * snakes_per_ball) + (pairs * snakes_per_pair) = 36 :=
  by 
    intros hb hspb hp hsp
    sorry

end total_snakes_l219_219181


namespace function_for_negative_x_l219_219906

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem function_for_negative_x {f : ℝ → ℝ} :
  odd_function f → given_function f → ∀ x, x < 0 → f x = x * (1 + x) :=
by
  intros h1 h2
  sorry

end function_for_negative_x_l219_219906


namespace time_period_principal_1000_amount_1120_interest_5_l219_219017

-- Definitions based on the conditions
def principal : ℝ := 1000
def amount : ℝ := 1120
def interest_rate : ℝ := 0.05

-- Lean 4 statement asserting the time period
theorem time_period_principal_1000_amount_1120_interest_5
  (P : ℝ) (A : ℝ) (r : ℝ) (T : ℝ) 
  (hP : P = principal)
  (hA : A = amount)
  (hr : r = interest_rate) :
  (A - P) * 100 / (P * r * 100) = 2.4 :=
by 
  -- The proof is filled in by 'sorry'
  sorry

end time_period_principal_1000_amount_1120_interest_5_l219_219017


namespace common_factor_polynomials_l219_219166

theorem common_factor_polynomials (a : ℝ) :
  (∀ p : ℝ, p ≠ 0 ∧ 
           (p^3 - p - a = 0) ∧ 
           (p^2 + p - a = 0)) → 
  (a = 0 ∨ a = 10 ∨ a = -2) := by
  sorry

end common_factor_polynomials_l219_219166


namespace sum_of_circle_areas_l219_219256

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l219_219256


namespace SplitWinnings_l219_219794

noncomputable def IstvanInitialContribution : ℕ := 5000 * 20
noncomputable def IstvanSecondPeriodContribution : ℕ := (5000 + 4000) * 30
noncomputable def IstvanThirdPeriodContribution : ℕ := (5000 + 4000 - 2500) * 40
noncomputable def IstvanTotalContribution : ℕ := IstvanInitialContribution + IstvanSecondPeriodContribution + IstvanThirdPeriodContribution

noncomputable def KalmanContribution : ℕ := 4000 * 70
noncomputable def LaszloContribution : ℕ := 2500 * 40
noncomputable def MiklosContributionAdjustment : ℕ := 2000 * 90

noncomputable def IstvanExpectedShare : ℕ := IstvanTotalContribution * 12 / 100
noncomputable def KalmanExpectedShare : ℕ := KalmanContribution * 12 / 100
noncomputable def LaszloExpectedShare : ℕ := LaszloContribution * 12 / 100
noncomputable def MiklosExpectedShare : ℕ := MiklosContributionAdjustment * 12 / 100

noncomputable def IstvanActualShare : ℕ := IstvanExpectedShare * 7 / 8
noncomputable def KalmanActualShare : ℕ := (KalmanExpectedShare - MiklosExpectedShare) * 7 / 8
noncomputable def LaszloActualShare : ℕ := LaszloExpectedShare * 7 / 8
noncomputable def MiklosActualShare : ℕ := MiklosExpectedShare * 7 / 8

theorem SplitWinnings :
  IstvanActualShare = 54600 ∧ KalmanActualShare = 7800 ∧ LaszloActualShare = 10500 ∧ MiklosActualShare = 18900 :=
by
  sorry

end SplitWinnings_l219_219794


namespace number_of_regions_divided_by_lines_l219_219197

theorem number_of_regions_divided_by_lines (n : ℕ)
  (λP : fin (2 * n) → ℕ) :
  let f := 1 + n + ∑ i in finset.range 2*n, (λP i - 1),
  let unbounded_regions := 2 * n in
  (f = 1 + n + ∑ i in finset.range 2*n, (λP i - 1)) ∧ (unbounded_regions = 2 * n) :=
sorry

end number_of_regions_divided_by_lines_l219_219197


namespace arithmetic_sequence_odd_function_always_positive_l219_219637

theorem arithmetic_sequence_odd_function_always_positive
    (f : ℝ → ℝ) (a : ℕ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_monotone_geq_0 : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
    (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
    (h_a3_neg : a 3 < 0) :
    f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 := by
    sorry

end arithmetic_sequence_odd_function_always_positive_l219_219637


namespace sin_add_pi_div_four_l219_219060

theorem sin_add_pi_div_four (θ : ℝ) (h1 : θ ∈ Ioo (π / 2) π) (h2 : tan (θ - π / 4) = -4 / 3) :
  sin (θ + π / 4) = -3 / 5 :=
sorry

end sin_add_pi_div_four_l219_219060


namespace number_of_subsets_with_four_adj_chairs_l219_219721

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219721


namespace golden_section_length_MP_l219_219955

theorem golden_section_length_MP 
  (MN : ℝ) (hMN : MN = 2) 
  (P : ℝ → ℝ → Prop) (hP : ∀ M N, P M N → M / (M - N) = (M - N) / N) :
  ∃ x : ℝ, x = (√5 - 1) ∧ MN / x = x / (MN - x) :=
by
  use (√5 - 1)
  split
  · exact rfl
  · sorry

end golden_section_length_MP_l219_219955


namespace ellipse_eccentricity_l219_219438

theorem ellipse_eccentricity :
  (∀ x y : ℝ, x^2 / 4 + y^2 / 9 = 1) →
  let a := 3 in
  let b := 2 in
  let c := Real.sqrt (a^2 - b^2) in
  let e := c / a in
  e = Real.sqrt 5 / 3 :=
sorry

end ellipse_eccentricity_l219_219438


namespace sets_equality_if_finite_union_l219_219512

open Set

variable {A B C : Set ℝ}

theorem sets_equality_if_finite_union 
  (h_inf1 : ∃ᶠ k in filter.at_top, A ⊆ {x + k * y | x ∈ B, y ∈ C})
  (h_inf2 : ∃ᶠ k in filter.at_top, B ⊆ {x + k * y | x ∈ C, y ∈ A})
  (h_inf3 : ∃ᶠ k in filter.at_top, C ⊆ {x + k * y | x ∈ A, y ∈ B})
  (h_fin : Finite (A ∪ B ∪ C)) : 
  A = B ∧ B = C := 
sorry

end sets_equality_if_finite_union_l219_219512


namespace log_x3y2_value_l219_219943

open Real

noncomputable def log_identity (x y : ℝ) : Prop :=
  log (x * y^4) = 1 ∧ log (x^3 * y) = 1

theorem log_x3y2_value (x y : ℝ) (h : log_identity x y) : log (x^3 * y^2) = 13 / 11 :=
  by
  sorry

end log_x3y2_value_l219_219943


namespace prove_given_expression_l219_219651

theorem prove_given_expression :
  75 * 222 + 76 * 225 - 25 * 14 * 15 - 25 * 15 * 16 = 302 := by
suffices : ∀ n, n * (n + 1) + (n + 1) * (n + 2) = 2 * (n + 1) ^ 2
from sorry
sorry

end prove_given_expression_l219_219651


namespace sum_of_prime_factors_of_143_l219_219330

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219330


namespace units_digit_of_odd_integers_product_l219_219777

theorem units_digit_of_odd_integers_product :
  let odd_integers := {n | (n % 2 = 1) ∧ (20 < n) ∧ (n < 200)} in
  (∏ n in odd_integers, n) % 10 = 5 := 
sorry

end units_digit_of_odd_integers_product_l219_219777


namespace volunteer_arrangement_count_l219_219879

theorem volunteer_arrangement_count :
  ∃ (V : Finset ℕ) (C : Finset (Finset ℕ)), 
  V.card = 5 ∧ 
  ∀ s ∈ C, s.card = 3 ∧ 
  (∀ A B s ∈ C, A ≠ B) ∧
  ∃ n : ℕ, n = 162 :=
by
  sorry

end volunteer_arrangement_count_l219_219879


namespace find_a_for_symmetry_l219_219069

theorem find_a_for_symmetry :
  ∃ a : ℝ, (∀ x : ℝ, a * Real.sin x + Real.cos (x + π / 6) = 
                    a * Real.sin (π / 3 - x) + Real.cos (π / 3 - x + π / 6)) 
           ↔ a = 2 :=
by
  sorry

end find_a_for_symmetry_l219_219069


namespace compare_stability_l219_219106

-- Specify the context of scores for students A and B.
variables (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℝ)

-- Define the average scores for both sets
def avg_A : ℝ := (a1 + a2 + a3 + a4 + a5) / 5
def avg_B : ℝ := (b1 + b2 + b3 + b4 + b5) / 5

-- Specify the condition that the average scores for both sets are equal.
axiom avg_equal : avg_A = avg_B

-- Define variance for both sets.
def var_A : ℝ := ((a1 - avg_A)^2 + (a2 - avg_A)^2 + (a3 - avg_A)^2 + (a4 - avg_A)^2 + (a5 - avg_A)^2) / 5
def var_B : ℝ := ((b1 - avg_B)^2 + (b2 - avg_B)^2 + (b3 - avg_B)^2 + (b4 - avg_B)^2 + (b5 - avg_B)^2) / 5

-- The proof problem: To determine which performance is more stable, compare the variances var_A and var_B.
theorem compare_stability : var_A = var_B → avg_equal → sorry

end compare_stability_l219_219106


namespace lowest_die_exactly_3_prob_l219_219950

noncomputable def fair_die_prob_at_least (n : ℕ) : ℚ :=
  if h : 1 ≤ n ∧ n ≤ 6 then (6 - n + 1) / 6 else 0

noncomputable def prob_lowest_die_exactly_3 : ℚ :=
  let p_at_least_3 := fair_die_prob_at_least 3
  let p_at_least_4 := fair_die_prob_at_least 4
  (p_at_least_3 ^ 4) - (p_at_least_4 ^ 4)

theorem lowest_die_exactly_3_prob :
  prob_lowest_die_exactly_3 = 175 / 1296 := by
  sorry

end lowest_die_exactly_3_prob_l219_219950


namespace base_conversion_subtraction_l219_219474

/-- Definition of base conversion from base 7 and base 5 to base 10. -/
def convert_base_7_to_10 (n : Nat) : Nat :=
  match n with
  | 52103 => 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def convert_base_5_to_10 (n : Nat) : Nat :=
  match n with
  | 43120 => 4 * 5^4 + 3 * 5^3 + 1 * 5^2 + 2 * 5^1 + 0 * 5^0
  | _ => 0

theorem base_conversion_subtraction : 
  convert_base_7_to_10 52103 - convert_base_5_to_10 43120 = 9833 :=
by
  -- The proof goes here
  sorry

end base_conversion_subtraction_l219_219474


namespace A_fraction_of_capital_subscribed_l219_219826

-- Let X denote the total capital
variable (X : ℝ)

-- Conditions
def B_share := (1/4) * X
def C_share := (1/5) * X
def A_profit := 830
def total_profit := 2490

-- Statement to prove
theorem A_fraction_of_capital_subscribed : (A_profit / total_profit) = 83 / 249 := by
  sorry

end A_fraction_of_capital_subscribed_l219_219826


namespace distance_from_origin_to_line_AB_is_sqrt6_div_3_l219_219522

open Real

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

def left_focus : Point := ⟨-1, 0⟩

def line_through_focus (t : ℝ) (p : Point) : Prop :=
  p.x = t * p.y - 1

def origin : Point := ⟨0, 0⟩

def perpendicular (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

noncomputable def distance (O : Point) (A B : Point) : ℝ :=
  let a := A.y - B.y
  let b := B.x - A.x
  let c := A.x * B.y - A.y * B.x
  abs (a * O.x + b * O.y + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line_AB_is_sqrt6_div_3 
  (A B : Point)
  (hA_on_ellipse : ellipse A)
  (hB_on_ellipse : ellipse B)
  (h_line_through_focus : ∃ t : ℝ, line_through_focus t A ∧ line_through_focus t B)
  (h_perpendicular : perpendicular A B) :
  distance origin A B = sqrt 6 / 3 := sorry

end distance_from_origin_to_line_AB_is_sqrt6_div_3_l219_219522


namespace income_increase_l219_219962

variable (a : ℝ)

theorem income_increase (h : ∃ a : ℝ, a > 0):
  a * 1.142 = a * 1 + a * 0.142 :=
by
  sorry

end income_increase_l219_219962


namespace compute_cos_l219_219526

noncomputable def angle1 (A C B : ℝ) : Prop := A + C = 2 * B
noncomputable def angle2 (A C B : ℝ) : Prop := 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B

theorem compute_cos (A B C : ℝ) (h1 : angle1 A C B) (h2 : angle2 A C B) : 
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 :=
sorry

end compute_cos_l219_219526


namespace sum_prime_factors_143_l219_219358

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219358


namespace systematic_sampling_number_l219_219976

theorem systematic_sampling_number (students : ℕ) (sample_size : ℕ) (sample : set ℕ) (num1 num2 num3 num4 : ℕ)
  (h_students : students = 56)
  (h_sample_size : sample_size = 4)
  (h_sample : sample = {num1, num2, num3, num4})
  (h_num1 : num1 = 5)
  (h_num2 : num2 = 33)
  (h_num3 : num3 = 47)
  (h_interval : (students / sample_size) = 14)
  (h_num_within_interval : ∀ x ∈ sample, ∃ k : ℕ, x = num1 + k * 14 ∨ x = num2 + k * 14 ∨ x = num3 + k * 14) :
  num4 = 19 :=
by
  sorry

end systematic_sampling_number_l219_219976


namespace mul_inv_mod_35_l219_219476

theorem mul_inv_mod_35 : (8 * 22) % 35 = 1 := 
  sorry

end mul_inv_mod_35_l219_219476


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219751

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219751


namespace sum_prime_factors_of_143_l219_219347

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219347


namespace anna_reading_time_l219_219446

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l219_219446


namespace perimeter_triangle_PST_l219_219592

-- Let P, Q, R be points such that PQ = 19, QR = 18, and PR = 17
variables {P Q R S T U : Type} [metric_space P] [metric_space Q] [metric_space R]
variables [metric_space S] [metric_space T] [metric_space U]
variable (PQ QR PR : ℝ)

-- Setup the distances between the points per the given conditions
def distance_pq : ℝ := 19
def distance_qr : ℝ := 18
def distance_pr : ℝ := 17

-- Define the conditions given in the problem
variables (QS SU UT : ℝ)

-- Conditions Q S = S U and U T = T R
axiom QS_eq_SU : QS = SU
axiom UT_eq_TR : UT = TR

-- Additional distances as variables
variable PT : ℝ
variable ST : ℝ

-- Define the distances in terms of fundamental points
def PS := distance_pq - QS
def PT := distance_pr - UT
def ST := QS + UT

-- Define the perimeter of triangle PST
def perimeter_PST := PS + ST + PT

-- The Lean statement stating the goal
theorem perimeter_triangle_PST (PQ QR PR QS SU UT : ℝ) 
(h1 : PQ = 19) (h2 : QR = 18) (h3 : PR = 17) 
(h4 : QS = SU) (h5 : UT = TR) : 
perimeter_PST P Q R S T U PQ QR PR = 36 :=
by
  sorry

end perimeter_triangle_PST_l219_219592


namespace Serezha_puts_more_berries_l219_219851

theorem Serezha_puts_more_berries (berries : ℕ) 
    (Serezha_puts : ℕ) (Serezha_eats : ℕ)
    (Dima_puts : ℕ) (Dima_eats : ℕ)
    (Serezha_rate : ℕ) (Dima_rate : ℕ)
    (total_berries : berries = 450)
    (Serezha_pattern : Serezha_puts = 1 ∧ Serezha_eats = 1)
    (Dima_pattern : Dima_puts = 2 ∧ Dima_eats = 1)
    (Serezha_faster : Serezha_rate = 2 * Dima_rate) : 
    ∃ (Serezha_in_basket : ℕ) (Dima_in_basket : ℕ),
      Serezha_in_basket > Dima_in_basket ∧ Serezha_in_basket - Dima_in_basket = 50 :=
by
  sorry -- Skip the proof

end Serezha_puts_more_berries_l219_219851


namespace factorization_analysis_l219_219007

variable (a b c : ℝ)

theorem factorization_analysis : a^2 - 2 * a * b + b^2 - c^2 = (a - b + c) * (a - b - c) := 
sorry

end factorization_analysis_l219_219007


namespace differentiate_exp_sin_l219_219946

theorem differentiate_exp_sin (x : ℝ) : 
  let y := exp x + sin x in 
  (deriv (λ x, y)) = exp x + cos x := 
by 
  sorry

end differentiate_exp_sin_l219_219946


namespace green_papayas_left_l219_219708

/-- 
  Problem: Prove that the number of green papayas left on the tree is 8,
  given the conditions:
  1. Initially, there are 14 green papayas.
  2. On Friday, 2 papayas turn yellow.
  3. On Sunday, twice as many papayas as on Friday turn yellow.
-/
theorem green_papayas_left (initial_papayas : ℕ) (friday_yellow : ℕ) (sunday_yellow_multiplier : ℕ) : 
  initial_papayas = 14 → 
  friday_yellow = 2 → 
  sunday_yellow_multiplier = 2 → 
  initial_papayas - friday_yellow - (friday_yellow * sunday_yellow_multiplier) = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end green_papayas_left_l219_219708


namespace seq_a22_l219_219243

def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0) ∧
  (a 10 = 10)

theorem seq_a22 : ∀ (a : ℕ → ℝ), seq a → a 22 = 10 :=
by
  intros a h,
  have h1 := h.1,
  have h99 := h.2.1,
  have h100 := h.2.2.1,
  have h_eq := h.2.2.2,
  sorry

end seq_a22_l219_219243


namespace find_vector_OB_l219_219118

def A : ℝ × ℝ := (0, 1)

def rotation (θ : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * real.cos θ - p.2 * real.sin θ, p.1 * real.sin θ + p.2 * real.cos θ)

theorem find_vector_OB :
  let O : ℝ × ℝ := (0, 0)
  let B := rotation (60 * π / 180) A
  B = (-real.sqrt 3 / 2, 1 / 2) :=
by
  sorry

end find_vector_OB_l219_219118


namespace real_solution_count_l219_219862

noncomputable def f (x : ℝ) : ℝ :=
  (1/(x - 1)) + (2/(x - 2)) + (3/(x - 3)) + (4/(x - 4)) + 
  (5/(x - 5)) + (6/(x - 6)) + (7/(x - 7)) + (8/(x - 8)) + 
  (9/(x - 9)) + (10/(x - 10))

theorem real_solution_count : ∃ n : ℕ, n = 11 ∧ 
  ∃ x : ℝ, f x = x :=
sorry

end real_solution_count_l219_219862


namespace min_distance_AB_l219_219124

-- Define points A and B on their respective curves
def C1 (x y : ℝ) : Prop := (x - 3) ^ 2 + (y - 4) ^ 2 = 1
def C2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 = 1

-- Define the problem: The minimum distance between points on curves C1 and C2
theorem min_distance_AB : 
  let A := {P : ℝ × ℝ // C1 P.1 P.2},
      B := {P : ℝ × ℝ // C2 P.1 P.2} in
  ∃ (a : A) (b : B), dist (a : ℝ × ℝ) (b : ℝ × ℝ) = 3 :=
by
  sorry

end min_distance_AB_l219_219124


namespace triangle_area_calculation_l219_219772

variables (A B C : Type) [AddGroup A] [AddGroup B] [AddGroup C]

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * (abs ((fst A * (snd B - snd C)) + (fst B * (snd C - snd A)) + (fst C * (snd A - snd B))))

theorem triangle_area_calculation : triangle_area (0, 0) (0, 4) (6, 10) = 12 :=
by
  sorry

end triangle_area_calculation_l219_219772


namespace katrina_cookies_left_l219_219613

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l219_219613


namespace problem_statement_l219_219043

noncomputable def f (x : ℝ) : ℝ := 
  real.sqrt (real.sin x ^ 4 + 4 * real.cos x ^ 2) - 
  real.sqrt (real.cos x ^ 4 + 4 * real.sin x ^ 2)

theorem problem_statement : 
  f (π / 12) = (real.sqrt 3) / 2 := 
by
  sorry

end problem_statement_l219_219043


namespace sum_prime_factors_143_l219_219322

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219322


namespace ellipse_equation_simplification_l219_219668

theorem ellipse_equation_simplification (x y : ℝ) :
  (sqrt ((x-2)^2 + y^2) + sqrt ((x+2)^2 + y^2) = 10) →
  (x^2 / 25 + y^2 / 21 = 1) :=
by
  sorry

end ellipse_equation_simplification_l219_219668


namespace subsets_with_at_least_four_adjacent_chairs_l219_219724

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219724


namespace find_k_perpendicular_l219_219548

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (2, -3)

-- Define a function for the vector k * a - 2 * b
def vec_expression (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - 2 * vec_b.1, k * vec_a.2 - 2 * vec_b.2)

-- Prove that if the dot product of vec_expression k and vec_a is zero, then k = -1
theorem find_k_perpendicular (k : ℝ) :
  ((vec_expression k).1 * vec_a.1 + (vec_expression k).2 * vec_a.2 = 0) → k = -1 :=
by
  sorry

end find_k_perpendicular_l219_219548


namespace sum_non_negative_integers_less_than_5_l219_219696

-- Conditions
def isNonNegative (n : ℤ) : Prop := n ≥ 0

def absLessThan5 (n : ℤ) : Prop := |n| < 5

def S : Set ℤ := {n | isNonNegative n ∧ absLessThan5 n}

-- Proof statement
theorem sum_non_negative_integers_less_than_5 : (∑ x in S, x) = 10 :=
by sorry

end sum_non_negative_integers_less_than_5_l219_219696


namespace emma_needs_six_trips_l219_219858

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r^3

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

def number_of_trips (volume_bucket volume_barrel : ℝ) : ℝ :=
  volume_barrel / volume_bucket

theorem emma_needs_six_trips :
  let barrel_radius := 8
  let barrel_height := 20
  let bucket_radius := 7
  let volume_bucket := volume_hemisphere bucket_radius
  let volume_barrel := volume_cylinder barrel_radius barrel_height
  Real.ceil (number_of_trips volume_bucket volume_barrel) = 6 :=
by
  sorry

end emma_needs_six_trips_l219_219858


namespace sum_of_angles_l219_219058

theorem sum_of_angles (α β : ℝ) (h1 : tan α = (-5 + sqrt(5^2 - 4 * 6 * 1))/(2 * 6) ∨ tan α = (-5 - sqrt(5^2 - 4 * 6 * 1))/(2 * 6))
  (h2 : tan β = (-5 + sqrt(5^2 - 4 * 6 * 1))/(2 * 6) ∨ tan β = (-5 - sqrt(5^2 - 4 * 6 * 1))/(2 * 6))
  (h3 : 0 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π) :
  α + β = π / 4 ∨ α + β = 5 * π / 4 :=
sorry

end sum_of_angles_l219_219058


namespace candy_shipment_l219_219211

open Nat

theorem candy_shipment (choc_cases lollipop_cases : ℕ)
  (h_ratio : ∀ (n : ℕ), choc_cases = 3 * n ∧ lollipop_cases = 2 * n) :
  choc_cases = 36 → lollipop_cases = 48 →
  ∃ (gummy_cases total_cases : ℕ), 
    (gummy_cases = (choc_cases / 3) * 1 ∧ total_cases = choc_cases + lollipop_cases + gummy_cases ∧ 
    gummy_cases = 24 ∧ total_cases = 108) := 
by
  intros h_choc h_lollipop
  let n := 12
  have h1 : choc_cases = 3 * n := by rw [h_choc]; exact rfl
  have h2 : lollipop_cases = 2 * n := by rw [h_lollipop]; exact rfl
  let gummy_cases := n * 1
  let total_cases := choc_cases + lollipop_cases + gummy_cases
  use [gummy_cases, total_cases]
  repeat3 (apply And.intro)
  · exact rfl
  · exact add_assoc _ _ _
  · exact rfl
  · exact rfl
  sorry 

end candy_shipment_l219_219211


namespace big_al_bananas_l219_219453

theorem big_al_bananas (total_bananas : ℕ) (a : ℕ)
  (h : total_bananas = 150)
  (h1 : a + (a + 7) + (a + 14) + (a + 21) + (a + 28) = total_bananas) :
  a + 14 = 30 :=
by
  -- Using the given conditions to prove the statement
  sorry

end big_al_bananas_l219_219453


namespace problem_statement_l219_219636

noncomputable def complex_seq_product (z : ℂ) := 
  (∑ k in Finset.range 10, z^(2 * (k + 1))) * (∑ k in Finset.range 10, (1 / z)^(2 * (k + 1)))

theorem problem_statement : 
  let z : ℂ := (1 - complex.i) / real.sqrt 2 in 
  complex_seq_product z = 25 := 
by
  sorry

end problem_statement_l219_219636


namespace hyperbolic_curve_hyperbola_identity_l219_219024

noncomputable def hyperbolic_curve (t : ℝ) : ℝ × ℝ :=
  (2 * Real.cosh(t), 4 * Real.sinh(t))

theorem hyperbolic_curve_hyperbola_identity (t : ℝ) :
  let x := 2 * Real.cosh(t)
  let y := 4 * Real.sinh(t)
  (x^2 / 4) - (y^2 / 16) = 1 :=
by {
  let x := 2 * Real.cosh(t),
  let y := 4 * Real.sinh(t),
  sorry
}

end hyperbolic_curve_hyperbola_identity_l219_219024


namespace line_intersects_even_number_of_diagonals_l219_219569

-- Define the convex polygon with 2009 vertices
structure ConvexPolygon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (convex : ∀ (i j k : Fin n), i ≠ j → j ≠ k → k ≠ i → 
              let A := vertices i, B := vertices j, C := vertices k in 
              ( (A.1 - B.1) * (C.2 - B.2) - (A.2 - B.2) * (C.1 - B.1) ) ≥ 0)

-- Definition of an intersection
def intersects (l : ℝ × ℝ → Prop) (P : ConvexPolygon 2009) : Finset (Fin 2009 × Fin 2009) :=
  { e | ∃ t, l (P.vertices e.1) ∧ l (P.vertices e.2) }

-- Main theorem
theorem line_intersects_even_number_of_diagonals (P : ConvexPolygon 2009) 
  (l : ℝ × ℝ → Prop) (inter_limits : ∀ v, ∃ t, ¬l (P.vertices v)) : 
  (intersects l P).card % 2 = 0 := 
sorry

end line_intersects_even_number_of_diagonals_l219_219569


namespace value_of_100a_10b_c_l219_219076

def is_correct (a b c : ℕ) : Prop :=
  (a ≠ 3 ∧ b ≠ 3 ∧ c = 0 ∨
   a ≠ 3 ∧ b = 3 ∧ c ≠ 0 ∨
   a = 3 ∧ b ≠ 3 ∧ c ≠ 0) ∧ 
  ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 0) ∧ (a ≠ 3 ∧ b = 3 ∧ c ≠ 0) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 0))

theorem value_of_100a_10b_c (a b c : ℕ) (h₀ : {a, b, c} = {0, 1, 3}) (h₁ : is_correct a b c) :
  100 * a + 10 * b + c = 301 := 
sorry

end value_of_100a_10b_c_l219_219076


namespace printing_company_proportion_l219_219414

theorem printing_company_proportion (x y : ℕ) :
  (28*x + 42*y) / (28*x) = 5/3 → x / y = 9 / 4 := by
  sorry

end printing_company_proportion_l219_219414


namespace max_diagonal_sum_l219_219224

/-- The integers 1, 2, ..., 64 are written on an 8x8 chess board such that for each 
    1 ≤ i < 64, the integers i and i+1 are in squares that share an edge.
    Prove that the largest possible sum that can appear along one of the diagonals
    of the board cannot exceed 432. -/
theorem max_diagonal_sum (board : Fin 8 → Fin 8 → Nat) 
  (h_integers : ∀ i j, 1 ≤ board i j ∧ board i j ≤ 64)
  (h_consecutive : ∀ i j, ∀ i', ∀ j', board i j + 1 = board i' j' → 
    (i = i' ∧ (j = j' + 1 ∨ j = j' - 1)) ∨ 
    (j = j' ∧ (i = i' + 1 ∨ i = i' - 1))) :
  (∑ i, board i i) ≤ 432 :=
sorry

end max_diagonal_sum_l219_219224


namespace band_ratio_requirement_l219_219643

noncomputable def calc_students (flutes: ℕ) (clarinets: ℕ) (trumpets: ℕ) (pianists: ℕ) (drummers: ℕ) : ℕ :=
  let ratio := 5:3:6:2:4 in
  let x := max (16 / 5) (max (15 / 3) (max (20 / 6) (max (2 / 2) (12 / 4)))) in
  5 * x + 3 * x + 6 * x + 2 * x + 4 * x

theorem band_ratio_requirement : calc_students 20 30 60 20 16 = 100 := sorry

end band_ratio_requirement_l219_219643


namespace equidistant_x_coordinate_l219_219768

open Real

-- Definitions for points C and D
def C : ℝ × ℝ := (-3, 0)
def D : ℝ × ℝ := (0, 5)

-- Definition for the distance function on the plane
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The x-coordinate of the point that is equidistant from C and D
theorem equidistant_x_coordinate : ∃ x : ℝ, dist (x, 0) C = dist (x, 0) D ∧ x = 8/3 :=
by
  let x := 8/3
  have h1 : dist (x, 0) C = sqrt ((-3 - x)^2 + 0^2),
    simp only [C],
  have h2 : dist (x, 0) D = sqrt ((0 - x)^2 + (-5)^2),
    simp only [D],
  use x,
  split,
  {
    rw [dist, dist, h1, h2],
    sorry -- Proof steps omitted
  },
  {
    refl,
  }

end equidistant_x_coordinate_l219_219768


namespace recorded_line_length_l219_219807

theorem recorded_line_length :
  let initial_radius := 29 / 2
  let final_radius := 11.5 / 2
  let total_time := 24.5
  let revolutions_per_minute := 100
  let angular_speed := 2 * Real.pi * revolutions_per_minute
  let radius_decrease_rate := (initial_radius - final_radius) / total_time
  let current_radius (t : Real) := initial_radius - radius_decrease_rate * t
  let tangential_speed (t : Real) := angular_speed * current_radius t
  let radial_speed := radius_decrease_rate
  let total_speed (t : Real) := Real.sqrt (tangential_speed t ^ 2 + radial_speed ^ 2)
  let integral_expression := ∫ (t : Real) in 0 .. total_time, total_speed t
  integral_expression = 155862.265789099 := 
sorry

end recorded_line_length_l219_219807


namespace M_inter_N_is_empty_l219_219923

noncomputable def M (t : ℝ) (h₁ : t ≠ -1) (h₂ : t ≠ 0) : ℂ :=
⟨t / (1 + t), (1 + t) / t⟩

noncomputable def N (t : ℝ) (h : |t| ≤ 1) : ℂ :=
⟨√2 * √(1 - t^2), √2 * t⟩

theorem M_inter_N_is_empty : 
  ∀ (z : ℂ), 
  (∃ t₁ : ℝ, t₁ ≠ -1 ∧ t₁ ≠ 0 ∧ z = M t₁ (by sorry) (by sorry)) ∧ 
  (∃ t₂ : ℝ, |t₂| ≤ 1 ∧ z = N t₂ (by sorry)) → 
  False :=
begin
  sorry
end

end M_inter_N_is_empty_l219_219923


namespace circle_tangent_to_circumscribed_circles_l219_219618

theorem circle_tangent_to_circumscribed_circles
  (A B C D O : Point)
  (h_parallelogram : parallelogram A B C D)
  (h_angles : ∠ A O B + ∠ C O D = ∠ B O C + ∠ A O D) :
  ∃ k : Circle,
    is_tangent k (circumcircle A O B) ∧
    is_tangent k (circumcircle B O C) ∧
    is_tangent k (circumcircle C O D) ∧
    is_tangent k (circumcircle D O A) :=
sorry

end circle_tangent_to_circumscribed_circles_l219_219618


namespace circumscribed_circle_diameter_l219_219951

/-- Define the given side of the triangle (a) and the opposite angle (A) --/
def a := 15
def A := 45

/-- Define the diameter of the circumscribed circle using the extended law of sines --/
def D := a / Real.sin (A * Real.pi / 180)

/-- Prove that the diameter is 15 * sqrt(2) inches. --/
theorem circumscribed_circle_diameter :
  D = 15 * Real.sqrt 2 :=
by
  sorry

end circumscribed_circle_diameter_l219_219951


namespace a_8_is_36_l219_219126

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 1) := a n + (n + 1)

theorem a_8_is_36 : a 8 = 36 := by
  sorry

end a_8_is_36_l219_219126


namespace triangle_side_range_l219_219980

theorem triangle_side_range (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) (h₃ : a = 2) (h₄ : b = 3) (h₅ : 2 * 2 + 3 * 3 - c * c < 0) :
  sqrt 13 < c ∧ c < 5 :=
by
  sorry

end triangle_side_range_l219_219980


namespace point_X_exists_l219_219545

noncomputable def construct_point_X {l1 l2 : Line} {A B E : Point} (a : ℝ) (hA : ¬Collinear A l1) (hB : ¬Collinear B l1) (hE : E ∈ l2) : Point :=
  sorry

theorem point_X_exists {l1 l2 : Line} {A B E : Point} (a : ℝ) (hA : ¬Collinear A l1) (hB : ¬Collinear B l1) (hE : E ∈ l2) :
  ∃ X : Point, (X ∈ l1) ∧ (let AX_intersect_l2 := IntersectLine X l2 in let BX_intersect_l2 := IntersectLine B l2 in
  Segment_length (AX_intersect_l2, BX_intersect_l2) = a ∧ Bisection_Point (AX_intersect_l2, BX_intersect_l2) = E) :=
begin
  sorry
end

end point_X_exists_l219_219545


namespace message_hours_needed_l219_219816

-- Define the sequence and the condition
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem message_hours_needed : ∃ n : ℕ, S n > 55 ∧ n = 5 := by
  sorry

end message_hours_needed_l219_219816


namespace sum_prime_factors_143_l219_219355

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219355


namespace sum_of_circle_areas_l219_219257

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l219_219257


namespace last_score_is_80_l219_219647

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end last_score_is_80_l219_219647


namespace S_10_is_65_l219_219625

variable (a_1 d : ℤ)
variable (S : ℤ → ℤ)

-- Define the arithmetic sequence conditions
def a_3 : ℤ := a_1 + 2 * d
def S_n (n : ℤ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a_3_is_4 : a_3 = 4
axiom S_9_minus_S_6_is_27 : S 9 - S 6 = 27

-- The target statement to be proven
theorem S_10_is_65 : S 10 = 65 :=
by
  sorry

end S_10_is_65_l219_219625


namespace smallest_interpolating_polynomial_degree_l219_219047

noncomputable def find_smallest_k (n : ℕ) (hn : 3 ≤ n) : ℕ :=
  ⌊n / 2⌋

-- Hypothetical theorem stating the required proof problem
theorem smallest_interpolating_polynomial_degree (n : ℕ) (hn : 3 ≤ n)
  (points : Fin n → ℝ × ℝ) (h_no_collinear : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ¬ collinear ((points i).1, (points i).2) ((points j).1, (points j).2) ((points k).1, (points k).2))
  (c : Fin n → ℝ) :
  ∃ (P : ℝ → ℝ → ℝ), (degree P ≤ find_smallest_k n hn) ∧
    (∀ i : Fin n, P (points i).fst (points i).snd = c i) :=
sorry

private def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ a * A.1 + b * A.2 + c = 0 ∧
    a * B.1 + b * B.2 + c = 0 ∧ a * C.1 + b * C.2 + c = 0

end smallest_interpolating_polynomial_degree_l219_219047


namespace tan_half_angle_is_two_l219_219901

-- Define the setup
variables (α : ℝ) (H1 : α ∈ Icc (π/2) π) (H2 : 3 * Real.sin α + 4 * Real.cos α = 0)

-- Define the main theorem
theorem tan_half_angle_is_two : Real.tan (α / 2) = 2 :=
sorry

end tan_half_angle_is_two_l219_219901


namespace exists_line_intersects_four_circles_l219_219130

theorem exists_line_intersects_four_circles (side_length : ℝ) (circumferences : list ℝ)
  (h_side_length : side_length = 1)
  (h_circumferences_sum : circumferences.sum = 10) :
  ∃ (l : ℝ → ℝ → Prop), ∃ (c1 c2 c3 c4 : ℝ × ℝ × ℝ), l c1.1 c1.2 ∧ l c2.1 c2.2 ∧ l c3.1 c3.2 ∧ l c4.1 c4.2 :=
by {
  sorry
}

end exists_line_intersects_four_circles_l219_219130


namespace max_sin_cos_ratio_is_l219_219948

noncomputable def max_sin_cos_ratio 
  (α β γ : ℝ) 
  (h1 : α > 0) 
  (h2 : α < π / 2) 
  (h3 : β > 0) 
  (h4 : β < π / 2) 
  (h5 : γ > 0) 
  (h6 : γ < π / 2) 
  (h7 : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  ℝ :=
sorry

theorem max_sin_cos_ratio_is 
  (α β γ : ℝ) 
  (h1 : α > 0) 
  (h2 : α < π / 2) 
  (h3 : β > 0) 
  (h4 : β < π / 2) 
  (h5 : γ > 0) 
  (h6 : γ < π / 2) 
  (h7 : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  max_sin_cos_ratio α β γ h1 h2 h3 h4 h5 h6 h7 = (sqrt 2) / 2 :=
sorry

end max_sin_cos_ratio_is_l219_219948


namespace probability_at_least_one_train_present_l219_219208

-- Define the condition for Train 1 arrival and wait time
noncomputable def train1_arrival : Prop := ∀ t ∈ set.Icc (0 : ℝ) 60, t + 15 ∈ set.Icc (0 : ℝ) 75

-- Define the condition for Train 2 arrival and wait time
noncomputable def train2_arrival : Prop := ∀ t ∈ set.Icc (30 : ℝ) 90, t + 25 ∈ set.Icc (30 : ℝ) 115

-- Define the condition for passenger arrival time
noncomputable def passenger_arrival : set.Icc (0 : ℝ) 90 := set.Icc (0 : ℝ) 90

theorem probability_at_least_one_train_present : 
  (∃ t ∈ passenger_arrival, train1_arrival t ∨ train2_arrival t) → 
  ∃ p : ℚ, p = 775 / 810 := 
sorry

end probability_at_least_one_train_present_l219_219208


namespace carrots_picked_next_day_l219_219172

theorem carrots_picked_next_day :
  ∀ (initial_picked thrown_out additional_picked total : ℕ),
    initial_picked = 48 →
    thrown_out = 11 →
    total = 52 →
    additional_picked = total - (initial_picked - thrown_out) →
    additional_picked = 15 :=
by
  intros initial_picked thrown_out additional_picked total h_ip h_to h_total h_ap
  sorry

end carrots_picked_next_day_l219_219172


namespace find_ordered_triple_l219_219127

variables {A B C P D E : Type} [AddCommGroup A] [AddCommGroup B]
          [AddCommGroup C] [AddCommGroup P] [AddCommGroup D] [AddCommGroup E]
          [VectorSpace ℝ A] [VectorSpace ℝ B] [VectorSpace ℝ C]
          [VectorSpace ℝ P] [VectorSpace ℝ D] [VectorSpace ℝ E]
          {a b c p : ℝ}

noncomputable def vector_D (B C : Type) [VectorSpace ℝ B] [VectorSpace ℝ C] (b c : ℝ) : B := 
  (3/2 : ℝ) • c - (1/2 : ℝ) • b

noncomputable def vector_E (A C : Type) [VectorSpace ℝ A] [VectorSpace ℝ C] (a c : ℝ) : A :=
  (3/8 : ℝ) • a + (5/8 : ℝ) • c

noncomputable def vector_P (A B C : Type) [VectorSpace ℝ A] [VectorSpace ℝ B] [VectorSpace ℝ C] (a b c : ℝ) : A :=
  (9/19 : ℝ) • a - (5/19 : ℝ) • b + (15/19 : ℝ) • c

theorem find_ordered_triple
  (BD_DC : ∀ (b d : Type) [VectorSpace ℝ b] [VectorSpace ℝ d], b.d_ratio = 3/1)
  (AE_EC : ∀ (a e : Type) [VectorSpace ℝ a] [VectorSpace ℝ e], a.e_ratio = 5/3)
  (x y z : ℝ)
  (h : x + y + z = 1) :
  vector_P A B C a b c = (9/19 : ℝ) • A - (5/19 : ℝ) • B + (15/19 : ℝ) • C :=
sorry

end find_ordered_triple_l219_219127


namespace race_length_126_l219_219102

-- Definitions
def length_of_race (L : ℕ) (S J : ℚ) (T : ℕ) : Prop :=
  S = L / 13 ∧ J = L / 18 ∧ (S ≠ J → S * 13 = J * 13 + 35) ∧ T = 13

theorem race_length_126 :
  ∃ (L : ℕ), 
  (∃ (S J : ℚ), 
    (∃ (T : ℕ), 
      length_of_race L S J T)) → 
  L = 126 :=
by 
  -- Variables
  let L := 126;
  let S := (L : ℚ) / 13;
  let J := (L : ℚ) / 18;
  let T := 13;
  -- Constraints
  have h1 : S = L / 13 := rfl,
  have h2 : J = L / 18 := rfl,
  have h3 : S ≠ J → S * 13 = J * 13 + 35 := sorry,
  have h4 : T = 13 := rfl,
  -- Conclusion
  exact ⟨L, ⟨S, ⟨J, ⟨T, ⟨h1, h2, h3, h4⟩⟩⟩⟩⟩,
  sorry

end race_length_126_l219_219102


namespace octagon_reflected_arcs_area_l219_219416

theorem octagon_reflected_arcs_area :
  let s := 2
  let θ := 45
  let r := 2 / Real.sqrt (2 - Real.sqrt (2))
  let sector_area := θ / 360 * Real.pi * r^2
  let total_arc_area := 8 * sector_area
  let circle_area := Real.pi * r^2
  let bounded_region_area := 8 * (circle_area - 2 * Real.sqrt (2) * 1 / 2)
  bounded_region_area = (16 * Real.sqrt 2 / 3 - Real.pi)
:= sorry

end octagon_reflected_arcs_area_l219_219416


namespace no_b_satisfies_l219_219051

theorem no_b_satisfies (b : ℝ) : ¬ (2 * 1 - b * (-2) + 1 ≤ 0 ∧ 2 * (-1) - b * 2 + 1 ≤ 0) :=
by
  sorry

end no_b_satisfies_l219_219051


namespace chairs_adjacent_subsets_l219_219735

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219735


namespace max_area_OPQ_l219_219067

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) := 
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem max_area_OPQ :
  ∀ (a b c : ℝ) (l1 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop),
  a^2 = 4 →
  b^2 = 1 →
  c = real.sqrt 3 →
  a > b ∧ b > 0 →
  ∃ (x y : ℝ), ellipse_equation a b x y ∧ x = 1 ∧ y = -real.sqrt 3 / 2 →
  l1 = λ p, p.1 - real.sqrt 3 * p.2 + 4 = 0 →
  l = λ p, p.2 = -real.sqrt 3 * p.1 + m →
  (∀ (P Q : ℝ × ℝ), 
    P ≠ Q ∧ 
    ellipse_equation a b P.1 P.2 ∧ 
    ellipse_equation a b Q.1 Q.2 ∧ 
    l P ∧ 
    l Q →
    let x1 := P.1,
        y1 := P.2,
        x2 := Q.1,
        y2 := Q.2,
        distance_PQ := real.sqrt((x1 - x2)^2 + (y1 - y2)^2),
        d := m / 2,
        area_OPQ := 0.5 * d * distance_PQ in
    area_OPQ ≤ 1) :=
sorry

end max_area_OPQ_l219_219067


namespace problem_I_5_example_problem_II_problem_III_l219_219949

-- E sequence definition
def is_E_sequence (seq : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k → k < n → abs (seq (k + 1) - seq k) = 1

-- Sum of sequence
def S (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, seq (i + 1)

-- Problem (I)
theorem problem_I_5_example (seq : ℕ → ℤ) (h1 : seq 1 = 0) (h3 : seq 3 = 0) :
  is_E_sequence seq 5 → seq 5 = 0 ∧ (seq 2 = 1 ∨ seq 2 = -1) :=
sorry

-- Problem (II)
theorem problem_II (seq : ℕ → ℤ) (h1 : seq 1 = 12) :
  is_E_sequence seq 2000 → (∀ k, 1 ≤ k → k < 2000 → seq (k + 1) > seq k) ↔ seq 2000 = 2011 :=
sorry

-- Problem (III)
theorem problem_III (seq : ℕ → ℤ) (h1 : seq 1 = 4) :
  is_E_sequence seq n → S seq n = 0 →  n ≥ 9 :=
sorry

end problem_I_5_example_problem_II_problem_III_l219_219949


namespace mouse_lives_count_l219_219805

-- Define the basic conditions
def catLives : ℕ := 9
def dogLives : ℕ := catLives - 3
def mouseLives : ℕ := dogLives + 7

-- The main theorem to prove
theorem mouse_lives_count : mouseLives = 13 :=
by
  -- proof steps go here
  sorry

end mouse_lives_count_l219_219805


namespace sum_prime_factors_of_143_l219_219350

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219350


namespace graph_shift_l219_219269

-- Definition of the functions
def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) * cos (x + π / 6)
def g (x : ℝ) : ℝ := sin (2 * x)

-- Statement of the problem
theorem graph_shift :
  ∀ x, f x = g (x + π / 6) := by
  sorry

end graph_shift_l219_219269


namespace examination_student_count_l219_219979

variable {T : ℕ} -- T is the total number of students

-- Define the two conditions
def passing_rate := 0.35
def failed_students := 455
def failing_rate := 1 - passing_rate

-- The theorem we want to prove
theorem examination_student_count (h1 : failing_rate * T = failed_students) : T = 700 := 
by
  sorry

end examination_student_count_l219_219979


namespace GDP_2005_l219_219432

noncomputable def initial_GDP : ℝ := 9593.3
noncomputable def growth_rate : ℝ := 0.073
def years : ℕ := 4

theorem GDP_2005 :
  (initial_GDP * (1 + growth_rate)^years) = 12725.4 := by
  have h := 9593.3 * (1 + 0.073)^4
  change h = 12725.4
  sorry

end GDP_2005_l219_219432


namespace circle_chairs_subsets_count_l219_219740

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219740


namespace domain_of_g_l219_219774

noncomputable def g (x : ℝ) : ℝ := log 3 (log 4 (log 5 (log 6 x)))

theorem domain_of_g (x : ℝ) : (x > 7776) ↔ ∃ y₁ y₂ y₃ y₄, y₁ = log 6 x ∧ y₂ = log 5 y₁ ∧ y₃ = log 4 y₂ ∧ y₄ = log 3 y₃ ∧ y₁ > 1 ∧ y₂ > 1 ∧ y₃ > 1 ∧ y₄ > 0 :=
by
  sorry

end domain_of_g_l219_219774


namespace abcd_eq_eleven_l219_219557

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Conditions on a, b, c, d
axiom cond_a : a = Real.sqrt (4 + Real.sqrt (5 + a))
axiom cond_b : b = Real.sqrt (4 - Real.sqrt (5 + b))
axiom cond_c : c = Real.sqrt (4 + Real.sqrt (5 - c))
axiom cond_d : d = Real.sqrt (4 - Real.sqrt (5 - d))

-- Theorem to prove
theorem abcd_eq_eleven : a * b * c * d = 11 :=
by
  sorry

end abcd_eq_eleven_l219_219557


namespace expansion_of_one_minus_x_to_ten_l219_219883

theorem expansion_of_one_minus_x_to_ten :
  (∃ (a : ℕ → ℤ), 
    (1 - x)^10 = ∑ i in range (11), a i * x^i ∧
    a 0 = 1 ∧
    (∑ i in range (10), a (i + 1)) = -1 ∧
    a 9 = -10 ∧
    (a 2 + a 4 + a 6 + a 8 + a 10) = 511) :=
sorry

end expansion_of_one_minus_x_to_ten_l219_219883


namespace compare_squares_and_powers_l219_219841

theorem compare_squares_and_powers (n : ℕ) (h : n ≥ 3) : (n + 1) ^ 2 < 3 ^ n :=
by skipproof

end compare_squares_and_powers_l219_219841


namespace find_integers_l219_219481

theorem find_integers (n : ℤ) : (6 ∣ (n - 4)) ∧ (10 ∣ (n - 8)) ↔ (n % 30 = 28) :=
by
  sorry

end find_integers_l219_219481


namespace percentage_chromium_first_alloy_l219_219585

theorem percentage_chromium_first_alloy 
  (x : ℝ) (w1 w2 : ℝ) (p2 p_new : ℝ) 
  (h1 : w1 = 10) 
  (h2 : w2 = 30) 
  (h3 : p2 = 0.08)
  (h4 : p_new = 0.09):
  ((x / 100) * w1 + p2 * w2) = p_new * (w1 + w2) → x = 12 :=
by
  sorry

end percentage_chromium_first_alloy_l219_219585


namespace sin_alpha_plus_pi_div_4_l219_219501

open Real

theorem sin_alpha_plus_pi_div_4 
  {α β : ℝ} 
  (h1 : α ∈ Ioo (3 * π / 4) π) 
  (h2 : β ∈ Ioo (3 * π / 4) π)
  (h3 : cos (α + β) = 4 / 5) 
  (h4 : cos (β - π / 4) = - 5 / 13) 
  : sin (α + π / 4) = - 33 / 65 :=
by
  sorry

end sin_alpha_plus_pi_div_4_l219_219501


namespace units_digit_of_m_squared_plus_two_to_m_is_7_l219_219156

theorem units_digit_of_m_squared_plus_two_to_m_is_7 :
  let m := 2021^2 + 3^2021 in (m^2 + 2^m) % 10 = 7 := by
let m := 2021^2 + 3^2021
sorry

end units_digit_of_m_squared_plus_two_to_m_is_7_l219_219156


namespace quaternary_to_decimal_l219_219847

theorem quaternary_to_decimal (n : ℕ) (h : n = 1010) : 
  let digits := [1, 0, 1, 0]
  let base := 4
  (digits.head * base^3 + digits.tail.head! * base^2 + digits.tail.tail.head! * base^1 + digits.tail.tail.tail.head!) = 68 := 
by sorry

end quaternary_to_decimal_l219_219847


namespace range_of_m_l219_219558

-- Assume the necessary conditions as given in the problem
variables (x y z m : ℝ)

-- Given conditions
def cond1 : Prop := 6 * x = 3 * y + 12 ∧ 6 * x = 2 * z
def cond2 : Prop := y ≥ 0
def cond3 : Prop := z ≤ 9
def cond4 : m = 2 * x + y - 3 * z

-- Prove the range of m
theorem range_of_m (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) :
  -19 ≤ m ∧ m ≤ -14 :=
sorry

end range_of_m_l219_219558


namespace paper_pattern_after_unfolding_l219_219819

-- Define the number of layers after folding the square paper four times
def folded_layers (initial_layers : ℕ) : ℕ :=
  initial_layers * 2 ^ 4

-- Define the number of quarter-circles removed based on the layers
def quarter_circles_removed (layers : ℕ) : ℕ :=
  layers

-- Define the number of complete circles from the quarter circles
def complete_circles (quarter_circles : ℕ) : ℕ :=
  quarter_circles / 4

-- The main theorem that we need to prove
theorem paper_pattern_after_unfolding :
  (complete_circles (quarter_circles_removed (folded_layers 1)) = 4) :=
by
  sorry

end paper_pattern_after_unfolding_l219_219819


namespace sum_of_prime_factors_143_l219_219296

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219296


namespace third_median_to_third_altitude_ratio_l219_219113

-- Definitions based on the problem conditions.
def scalene_triangle (A B C : Point) : Prop :=
  ¬collinear A B C ∧ ∀ (AB BC CA : Segment), AB ≠ BC ∧ BC ≠ CA ∧ AB ≠ CA

def is_median (A B C M : Point) : Prop :=
  midpoint M B C ∧ B ≠ C ∧ line_segment A M ∈ {X | ∃ (P Q : Point), midpoint P Q X}

def is_altitude (A B C P : Point) : Prop :=
  line_segment B C ⟂ line_segment A P ∧ P ∈ line_segment B C

def ratio_of_third_median_to_third_altitude 
  (A B C A' B' C' : Point)
  (hm : scalene_triangle A B C)
  (hma1 : is_median A A' B C ∧ is_altitude B A' B C)
  (hma2 : is_median B B' A C ∧ is_altitude C B' A C)
  : ℚ :=
7 / 2

-- Statement that the ratio of the third median to the third altitude is 7:2.
theorem third_median_to_third_altitude_ratio 
  (A B C G A' B' C' : Point)
  (H_triangle : scalene_triangle A B C)
  (H_medians_equal_altitudes : is_median A A' B C ∧ is_altitude B A' B C ∧ is_median B B' A C ∧ is_altitude C B' A C)
  : ratio_of_third_median_to_third_altitude A B C A' B' C' H_triangle H_medians_equal_altitudes = 7 / 2 :=
sorry

end third_median_to_third_altitude_ratio_l219_219113


namespace train_crosses_platform_in_50_seconds_l219_219396

theorem train_crosses_platform_in_50_seconds
  (length_train : ℝ) (time_signal : ℝ) (length_platform : ℝ)
  (h_train : length_train = 200)
  (h_signal : time_signal = 42)
  (h_platform : length_platform = 38.0952380952381) :
  (length_train + length_platform) / (length_train / time_signal) = 50 := 
by {
  have h_speed : (length_train / time_signal) = 4.76190476190476 :=
    by ring,
  have h_distance : (length_train + length_platform) = 238.0952380952381 :=
    by ring,
  calc
    (length_train + length_platform) / (length_train / time_signal)
        = 238.0952380952381 / 4.76190476190476 : by rw [h_distance, h_speed]
    ... = 50 : by ring
}

end train_crosses_platform_in_50_seconds_l219_219396


namespace range_of_independent_variable_l219_219594

theorem range_of_independent_variable:
  ∀ (x : ℝ), (x - 2 ≠ 0 ∧ x + 1 ≥ 0) ↔ (x ≥ -1 ∧ x ≠ 2) :=
by
  intro x
  split
  · intro h
    cases h with h₁ h₂
    constructor
    · exact h₂
    · exact ne_of_lt (lt_of_sub_ne_zero h₁)
  · intro h
    cases h with h₁ h₂
    constructor
    · intro h₃
      exact h₂ h₃
    · exact h₁

end range_of_independent_variable_l219_219594


namespace find_alpha_beta_l219_219885

-- Define the conditions of the problem
variables (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π)
variable (h_eq : ∀ x : ℝ, cos (x + α) + sin (x + β) + sqrt 2 * cos x = 0)

-- State the required proof as a theorem
theorem find_alpha_beta : α = 3 * π / 4 ∧ β = 7 * π / 4 :=
by
  sorry

end find_alpha_beta_l219_219885


namespace even_number_of_diagonals_intersected_l219_219568

theorem even_number_of_diagonals_intersected
  (polygon : convex_polygon 2009)
  (line : affine_line ℝ)
  (h_intersects : intersects polygon line)
  (h_no_vertex : ∀ v ∈ polygon.vertices, ¬ line.contains v) :
  ∃ k : ℕ, intersects_diagonals polygon line = 2 * k := 
sorry

end even_number_of_diagonals_intersected_l219_219568


namespace no_such_abc_l219_219831

theorem no_such_abc :
  ¬ ∃ (a b c : ℕ+),
    (∃ k1 : ℕ, a ^ 2 * b * c + 2 = k1 ^ 2) ∧
    (∃ k2 : ℕ, b ^ 2 * c * a + 2 = k2 ^ 2) ∧
    (∃ k3 : ℕ, c ^ 2 * a * b + 2 = k3 ^ 2) := 
sorry

end no_such_abc_l219_219831


namespace evaporation_rate_is_200_ml_per_hour_l219_219135

-- Definitions based on the given conditions
def faucet_drip_rate : ℕ := 40 -- ml per minute
def running_time : ℕ := 9 -- hours
def dumped_water : ℕ := 12000 -- ml (converted from liters)
def water_left : ℕ := 7800 -- ml

-- Alias for total water dripped in running_time
noncomputable def total_dripped_water : ℕ := faucet_drip_rate * 60 * running_time

-- Total water that should have been in the bathtub without evaporation
noncomputable def total_without_evaporation : ℕ := total_dripped_water - dumped_water

-- Water evaporated
noncomputable def evaporated_water : ℕ := total_without_evaporation - water_left

-- Evaporation rate in ml/hour
noncomputable def evaporation_rate : ℕ := evaporated_water / running_time

-- The goal theorem statement
theorem evaporation_rate_is_200_ml_per_hour : evaporation_rate = 200 := by
  -- proof here
  sorry

end evaporation_rate_is_200_ml_per_hour_l219_219135


namespace sum_f_values_l219_219502

theorem sum_f_values :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = -f (x + 3 / 2)) ∧
    (∀ x, f (-3 / 4 - x) = -f (-3 / 4 + x)) ∧
    f (-1) = 1 ∧
    f 0 = -2 ∧
    (∑ k in (finset.range 2008).map (λ n, n.succ), f k) = 1 :=
begin
  sorry
end

end sum_f_values_l219_219502


namespace chairs_adjacent_subsets_l219_219736

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219736


namespace select_3_representatives_l219_219201

theorem select_3_representatives (boys girls : ℕ) (h_boys : boys = 4) (h_girls : girls = 4) :
  (∑ g in {2, 3}, (Nat.choose girls g) * (Nat.choose boys (3 - g))) = 28 :=
by
  sorry

end select_3_representatives_l219_219201


namespace sum_of_prime_factors_of_143_l219_219333

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219333


namespace solve_for_x_l219_219488

theorem solve_for_x : ∃ x : ℝ, (15^2 * 8^x / 256 = 450) ∧ x = 3 :=
by
  have h₁ : (15 : ℝ)^2 = 225 := by norm_num
  have h₂ : (256 : ℝ) = 2^8 := by norm_num
  have h₃ : 450 / 225 = 2 := by norm_num
  have h₄ : 225 * 8^3 / 256 = 450 := by sorry
  use 3
  split
  · exact h₄
  · norm_num

end solve_for_x_l219_219488


namespace sum_largest_and_smallest_3digit_with_2_and_7_l219_219864

theorem sum_largest_and_smallest_3digit_with_2_and_7 :
  let largest := 297
      smallest := 207
  in largest + smallest = 504 := by
  sorry

end sum_largest_and_smallest_3digit_with_2_and_7_l219_219864


namespace hyperbola_eccentricity_l219_219014

theorem hyperbola_eccentricity :
  let a := Real.sqrt 3
  let b := Real.sqrt 6
  let c := 3
  ∃ e : ℝ, e = Real.sqrt 3 ∧ e = c / a :=
by
  let a := Real.sqrt 3
  let b := Real.sqrt 6
  let c := 3
  use Real.sqrt 3
  split
  · rfl
  · have h : c / a = Real.sqrt 3 := by
      calc
        c / a = 3 / Real.sqrt 3 : by rfl
        ...   = Real.sqrt 3     : by norm_num
    exact h

end hyperbola_eccentricity_l219_219014


namespace sum_prime_factors_of_143_l219_219340

theorem sum_prime_factors_of_143 : 
  let primes := {p : ℕ | p.prime ∧ p ∣ 143} in
  ∑ p in primes, p = 24 := 
by
  sorry

end sum_prime_factors_of_143_l219_219340


namespace katrina_cookies_left_l219_219608

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l219_219608


namespace find_r_plus_s_l219_219157

-- Definitions of the circle equations in terms of centers and radii
def u1 : (ℝ × ℝ) × ℝ := ((-4, 9), 10)
def u2 : (ℝ × ℝ) × ℝ := ((4, 9), 2)

-- Definition for the condition involving b where the line y = bx
def condition (b : ℝ) : Prop :=
  (b ^ 2 = 61 / 24)

-- The proof statement
theorem find_r_plus_s (b : ℝ) (r s : ℕ) (h1 : Nat.coprime r s) (h2 : n^2 = r / s ) (h3 : condition b) : r + s = 85 :=
  sorry

end find_r_plus_s_l219_219157


namespace sum_even_102_to_200_l219_219702

def sum_even_integers (m n : ℕ) : ℕ :=
  sum (list.map (λ k, 2 * k) (list.range' m (n - m + 1)))

theorem sum_even_102_to_200 :
  sum_even_integers (102 / 2) (200 / 2) = 7550 := 
sorry

end sum_even_102_to_200_l219_219702


namespace ratio_of_trap_to_square_is_five_eighths_l219_219982

-- Definition of the square and midpoints
structure Square (s : ℝ) :=
(A B C D M N P : Point)
(is_square : is_square A B C D s)
(is_midpoint_M : is_midpoint A B s (M))
(is_midpoint_N : is_midpoint B C s (N))
(is_midpoint_P : is_midpoint C D s (P))

noncomputable def ratio_of_areas (s : ℝ) (sq : Square s) : ℝ :=
(trapezoid_AMNP_area sq / square_ABCD_area sq)

theorem ratio_of_trap_to_square_is_five_eighths (s : ℝ) (sq : Square s) :
  ratio_of_areas s sq = 5 / 8 := 
sorry

end ratio_of_trap_to_square_is_five_eighths_l219_219982


namespace perfect_square_product_l219_219690

theorem perfect_square_product :
  (∃ (pairs : list (ℕ × ℕ)), 
    (∀ (p ∈ pairs), 
       (p.fst + p.snd = 2021 ∧ 7 ≤ p.fst ∧ p.snd ≤ 2014) ∨
       (p.fst + p.snd = 6 ∧ p.fst ∈ {1, 2, 3} ∧ p.snd ∈ {4, 5, 6}) ∨
       (p.fst + p.snd = 9 ∧ (p.fst, p.snd) = (3, 6))) ∧
    (pairs.length = 1007)) → 
  ∃ (k : ℕ), 
    ∏ (p : ℕ × ℕ) in (list.to_finset pairs), (p.fst + p.snd) = k^2 :=
sorry

end perfect_square_product_l219_219690


namespace binary_of_19_is_10011_l219_219006

noncomputable def binary_representation_of_nineteen : string := "10011"

theorem binary_of_19_is_10011 : binary_representation_of_nineteen = Int.toString 2 19 :=
by
  sorry

end binary_of_19_is_10011_l219_219006


namespace crossing_horizontal_asymptote_l219_219029

-- Define the function g(x)
def g (x : ℝ) := (5 * x^2 - 8 * x - 10) / (x^2 - 6 * x + 9)

-- Prove that g(x) = 5 when x = 5/2
theorem crossing_horizontal_asymptote : g (5 / 2) = 5 := by
  sorry

end crossing_horizontal_asymptote_l219_219029


namespace average_of_first_21_multiples_of_5_l219_219280

theorem average_of_first_21_multiples_of_5 : (∑ i in finset.range 21, 5 * (i + 1)) / 21 = 55 := sorry

end average_of_first_21_multiples_of_5_l219_219280


namespace find_value_correct_l219_219164

-- Definitions for the given conditions
def equation1 (a b : ℚ) : Prop := 3 * a - b = 8
def equation2 (a b : ℚ) : Prop := 4 * b + 7 * a = 13

-- Definition for the question
def find_value (a b : ℚ) : ℚ := 2 * a + b

-- Statement of the proof
theorem find_value_correct (a b : ℚ) (h1 : equation1 a b) (h2 : equation2 a b) : find_value a b = 73 / 19 := 
by 
  sorry

end find_value_correct_l219_219164


namespace point_in_fourth_quadrant_l219_219985

def point (x y : ℝ) := (x, y)
def x_positive (p : ℝ × ℝ) : Prop := p.1 > 0
def y_negative (p : ℝ × ℝ) : Prop := p.2 < 0
def in_fourth_quadrant (p : ℝ × ℝ) : Prop := x_positive p ∧ y_negative p

theorem point_in_fourth_quadrant : in_fourth_quadrant (2, -4) :=
by
  -- The proof states that (2, -4) is in the fourth quadrant.
  sorry

end point_in_fourth_quadrant_l219_219985


namespace complex_number_identity_l219_219066

theorem complex_number_identity (z : ℂ) (h : z = 1 + (1 : ℂ) * I) : z^2 + z = 1 + 3 * I := 
sorry

end complex_number_identity_l219_219066


namespace average_multiples_of_10_l219_219385

theorem average_multiples_of_10 : 
  (∑ k in (filter (λ x, x % 10 = 0) (range 161)), k) / ((filter (λ x, x % 10 = 0) (range 161)).length) = 85 := 
by
  sorry

end average_multiples_of_10_l219_219385


namespace sequence_property_l219_219234

theorem sequence_property (a : ℕ → ℝ) (h1 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n+1) + 2021 * a (n+2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := 
sorry

end sequence_property_l219_219234


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219755

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219755


namespace range_of_k_l219_219510

theorem range_of_k :
  ∃ k, (1 <= k) ∧ (k <= 3) → (∀ A B P, A = (1,0) ∧ B = (3,0) ∧ P ∈ line (y = kx + 1) → PA ⊥ PB)
→ (-4/3 ≤ k ∧ k ≤ 0) :=
by
  sorry

end range_of_k_l219_219510


namespace cats_weigh_more_than_puppies_l219_219935

theorem cats_weigh_more_than_puppies :
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  cats_weight - puppies_weight = 5 :=
by
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  show cats_weight - puppies_weight = 5 from sorry

end cats_weigh_more_than_puppies_l219_219935


namespace equivalence_of_X_conditions_l219_219154

theorem equivalence_of_X_conditions {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
  {ξ X : Ω → ℝ}
  (h_ind : Independent ξ X)
  (h_dist_ξ : P {ω | ξ ω = 1} = 1 / 2 ∧ P {ω | ξ ω = -1} = 1 / 2) :
  (X ∼ᵈ (-X)) ↔ (X ∼ᵈ (ξ * X)) ↔ (X ∼ᵈ (ξ * |X|)) :=
sorry

end equivalence_of_X_conditions_l219_219154


namespace rational_number_condition_l219_219170

namespace ProofProblem

-- Defining necessary assumptions and conditions
variables (a c b : ℕ) (h_pos_a : a > 0) (h_pos_c : c > 0) (h_pos_b : b > 0)
variable  (h_div : b ∣ ac - 1)
variable  {r : ℚ} (h_r_pos : r > 0) (h_r_lt_one : r < 1)

-- Define the gcd of integers as used in the solution steps
def gcd (x y : ℕ) : ℕ := Nat.gcd x y

-- Define the set A(r) as described in the problem
noncomputable def A (r : ℚ) : Set ℚ :=
  {m (r - ac) + n * ab | m n : ℤ}

-- Main theorem statement
theorem rational_number_condition :
  (∀ r, (r > 0 ∧ r < 1) → (∃ r = a / (a + b), (∀ m n : ℤ, (m * (r - a * c) + n * (b * a))  / (gcd (m * (p - (q * a * c))) (n * (q * (a * b)))) ) ≥ (a * b / (a+b)))) :=
sorry

end ProofProblem

end rational_number_condition_l219_219170


namespace circle_chairs_subsets_count_l219_219742

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l219_219742


namespace geom_seq_206th_term_l219_219279

theorem geom_seq_206th_term (a r : ℤ) 
  (h1 : a = 4) 
  (h2 : a * r = -12) : 
  let nth_term := a * r ^ (205) in
  nth_term = -4 * 3 ^ 204 := by
  sorry

end geom_seq_206th_term_l219_219279


namespace rowing_speed_downstream_l219_219410

theorem rowing_speed_downstream (V_u V_s V_d : ℝ) (h1 : V_u = 10) (h2 : V_s = 15)
  (h3 : V_s = (V_u + V_d) / 2) : V_d = 20 := by
  sorry

end rowing_speed_downstream_l219_219410


namespace maximum_area_of_sector_l219_219061

variables {r θ : ℝ}

-- Define the conditions
def circumference_eq : Prop := r * θ + 2 * r = 40
def area (r θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Problem: Given that the circumference of a sector is 40, prove that the maximum area is 100.
theorem maximum_area_of_sector : circumference_eq → (∃ (max_area : ℝ), max_area = 100) :=
by
  assume h : circumference_eq,
  -- Sorry is used here to skip the proof:
  sorry

end maximum_area_of_sector_l219_219061


namespace coefficient_of_x9_in_expansion_l219_219010

theorem coefficient_of_x9_in_expansion :
  let general_term (r : ℕ) := (-1)^r * (1 / 2^r) * Nat.choose 9 r * (x : ℝ)^(18 - 3 * r),
      r_value := 6,
      term_at_r6 := general_term r_value
  in term_at_r6 = (21 / 16) * x^9 := 
by 
  simp [general_term, r_value]; 
  sorry

end coefficient_of_x9_in_expansion_l219_219010


namespace notebooks_problem_l219_219766

variable (a b c : ℕ)

theorem notebooks_problem (h1 : a + 6 = b + c) (h2 : b + 10 = a + c) : c = 8 :=
  sorry

end notebooks_problem_l219_219766


namespace triangle_sine_identity_triangle_cosine_identity_l219_219601

variables {A B C a b c : ℝ} (R : ℝ)

-- Law of Sines assumption
axiom law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

theorem triangle_sine_identity : 
    (a * sin ((B - C) / 2) / sin (A / 2) + 
     b * sin ((C - A) / 2) / sin (B / 2) + 
     c * sin ((A - B) / 2) / sin (C / 2)) = 0 :=
sorry 

theorem triangle_cosine_identity :
    (a * sin ((B - C) / 2) / cos (A / 2) + 
     b * sin ((C - A) / 2) / cos (B / 2) + 
     c * sin ((A - B) / 2) / cos (C / 2)) = 0 :=
sorry

end triangle_sine_identity_triangle_cosine_identity_l219_219601


namespace fraction_relation_l219_219160

-- Definitions for arithmetic sequences and their sums
noncomputable def a_n (a₁ d₁ n : ℕ) := a₁ + (n - 1) * d₁
noncomputable def b_n (b₁ d₂ n : ℕ) := b₁ + (n - 1) * d₂

noncomputable def A_n (a₁ d₁ n : ℕ) := n * a₁ + n * (n - 1) * d₁ / 2
noncomputable def B_n (b₁ d₂ n : ℕ) := n * b₁ + n * (n - 1) * d₂ / 2

-- Theorem statement
theorem fraction_relation (a₁ d₁ b₁ d₂ : ℕ) (h : ∀ n : ℕ, B_n a₁ d₁ n ≠ 0 → A_n a₁ d₁ n / B_n b₁ d₂ n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b_n b₁ d₂ n ≠ 0 → a_n a₁ d₁ n / b_n b₁ d₂ n = (4 * n - 3) / (6 * n - 2) :=
sorry

end fraction_relation_l219_219160


namespace correct_sequences_and_sum_l219_219895

-- Arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Sum of the first n terms, S_n
def S (n : ℕ) : ℕ := n * (n + 1) / 2

-- Conditions given in the problem
def a2_eq_2 : a(2) = 2 := rfl
def S5_eq_15 : S(5) = 15 := rfl

-- Sequence {b_n}
def b : ℕ → ℝ
| 1 => 1 / 2
| n + 1 => (1 + 1 / a(n)) * b(n) / 2

-- Sum of the sequence {b_n}, T_n
def T (n : ℕ) : ℝ := (List.range n).map b |>.sum

-- Sequence {c_n}
def c (n : ℕ) : ℝ := (2 - T(n)) / (4 * S(n))

-- Theorem to prove the correctness
theorem correct_sequences_and_sum (n : ℕ) : 
  (∀ n, a n = n) ∧ 
  (∀ n, b n = n / 2^n) ∧
  (∑ k in range n, c (k + 1)) < (1 / 2) :=
by
  -- The proof would go here
  sorry


end correct_sequences_and_sum_l219_219895


namespace chairs_adjacent_subsets_l219_219731

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219731


namespace sum_prime_factors_143_l219_219362

open Nat

theorem sum_prime_factors_143 : (11 + 13) = 24 :=
by
  have h1 : Prime 11 := by sorry
  have h2 : Prime 13 := by sorry
  have h3 : 143 = 11 * 13 := by sorry
  exact add_eq_of_eq h3 (11 + 13) 24 sorry

end sum_prime_factors_143_l219_219362


namespace calculate_f3_times_l219_219615

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4 * n + 2

theorem calculate_f3_times : f (f (f 3)) = 170 := by
  sorry

end calculate_f3_times_l219_219615


namespace collinear_points_count_l219_219436

theorem collinear_points_count :
  let vertices := 8
  let edge_midpoints := 12
  let face_centers := 6
  let cube_center := 1
  let total_points := vertices + edge_midpoints + face_centers + cube_center
  total_points = 27 → (number_of_collinear_sets total_points = 49) :=
by
  intro h
  sorry

end collinear_points_count_l219_219436


namespace sum_prime_factors_143_is_24_l219_219286

def is_not_divisible (n k : ℕ) : Prop := ¬ (n % k = 0)

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

noncomputable def prime_factors_sum_143 : ℕ :=
  if is_not_divisible 143 2 ∧
     is_not_divisible 143 3 ∧
     is_not_divisible 143 5 ∧
     is_not_divisible 143 7 ∧
     (143 % 11 = 0) ∧
     (143 / 11 = 13) ∧
     is_prime 11 ∧
     is_prime 13 then 11 + 13 else 0

theorem sum_prime_factors_143_is_24 :
  prime_factors_sum_143 = 24 :=
by
  sorry

end sum_prime_factors_143_is_24_l219_219286


namespace proof_geom_seq_sum_sequence_b_l219_219899

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧ 
  a 1 * a 4 = 8 ∧ 
  a 2 + a 3 = 6

noncomputable def sequence_b (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (finset.range n).sum (λ i, (2 * (i + 1) - 1) * a (i + 1) / b (i + 1)) = n

theorem proof_geom_seq (a : ℕ → ℝ) 
  (h_geom_seq : geometric_sequence a) : 
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1) :=
sorry

theorem sum_sequence_b (a b : ℕ → ℝ)
  (h_geom_seq : geometric_sequence a) 
  (h_seq_b : sequence_b a b) :
  ∀ n : ℕ, n > 0 → (finset.range n).sum (λ i, b (i + 1)) = (2 * n - 3) * 2 ^ n + 3 :=
sorry

end proof_geom_seq_sum_sequence_b_l219_219899


namespace correct_flowerpot_calculation_l219_219452

-- Define the perimeter of the rectangular plaza
def perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

-- The number of flower pots given a specific spacing
def num_flowerpots (perimeter spacing : ℕ) : ℕ :=
  perimeter / spacing

-- Given condition values
def length := 500
def width := 300
def current_spacing := 2.5
def new_spacing := 2
def corners := 4

-- Current and new number of flowerpots
def current_flowerpots := num_flowerpots (perimeter length width) current_spacing
def new_flowerpots := num_flowerpots (perimeter length width) new_spacing

-- Number of additional flowerpots needed
def additional_flowerpots : ℕ :=
  new_flowerpots - current_flowerpots

-- Proving the correct answers
theorem correct_flowerpot_calculation :
  additional_flowerpots = 160 ∧ corners = 160 :=
by
  sorry

end correct_flowerpot_calculation_l219_219452


namespace volume_of_inscribed_sphere_l219_219956

noncomputable def volume_of_tetrahedron (R : ℝ) (S1 S2 S3 S4 : ℝ) : ℝ :=
  R * (S1 + S2 + S3 + S4)

theorem volume_of_inscribed_sphere (R S1 S2 S3 S4 V : ℝ) :
  V = R * (S1 + S2 + S3 + S4) :=
sorry

end volume_of_inscribed_sphere_l219_219956


namespace average_test_score_fifty_percent_l219_219560

-- Given conditions
def percent1 : ℝ := 15
def avg1 : ℝ := 100
def percent2 : ℝ := 50
def avg3 : ℝ := 63
def overall_average : ℝ := 76.05

-- Intermediate calculations based on given conditions
def total_percent : ℝ := 100
def percent3: ℝ := total_percent - percent1 - percent2
def sum_of_weights: ℝ := overall_average * total_percent

-- Expected average of the group that is 50% of the class
theorem average_test_score_fifty_percent (X: ℝ) :
  sum_of_weights = percent1 * avg1 + percent2 * X + percent3 * avg3 → X = 78 := by
  sorry

end average_test_score_fifty_percent_l219_219560


namespace cats_weigh_more_than_puppies_l219_219933

theorem cats_weigh_more_than_puppies :
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  cats_weight - puppies_weight = 5 :=
by
  let puppies_weight := 4 * 7.5
  let cats_weight := 14 * 2.5
  show cats_weight - puppies_weight = 5 from sorry

end cats_weigh_more_than_puppies_l219_219933


namespace fraction_simplification_l219_219460

theorem fraction_simplification (a b c : ℝ) :
  (4 * a^2 + 2 * c^2 - 4 * b^2 - 8 * b * c) / (3 * a^2 + 6 * a * c - 3 * c^2 - 6 * a * b) =
  (4 / 3) * ((a - 2 * b + c) * (a - c)) / ((a - b + c) * (a - b - c)) :=
by
  sorry

end fraction_simplification_l219_219460


namespace subsets_with_at_least_four_adjacent_chairs_l219_219759

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219759


namespace sum_of_prime_factors_of_143_l219_219306

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_of_143 :
  let pfs : List ℕ := [11, 13] in
  (∀ p ∈ pfs, is_prime p) → pfs.sum = 24 → pfs.product = 143  :=
by
  sorry

end sum_of_prime_factors_of_143_l219_219306


namespace difference_is_two_l219_219653

def students_scores : List (ℝ × ℝ) := [(0.2, 60), (0.4, 75), (0.25, 85), (0.15, 95)]

def median_score (scores : List (ℝ × ℝ)) : ℝ :=
  scores.nth 1 |> Option.getD 0 *.snd -- 75 is the median as it's the second interval in the cumulative distribution

def mean_score (scores : List (ℝ × ℝ)) : ℝ :=
  scores.foldl (λ acc (p, s), acc + p * s) 0

noncomputable def difference_mean_median (scores : List (ℝ × ℝ)) : ℝ :=
  |mean_score scores - median_score scores|

theorem difference_is_two :
  difference_mean_median students_scores = 2 :=
sorry

end difference_is_two_l219_219653


namespace area_KLMN_l219_219192

theorem area_KLMN (A B C D K L M N : Point) 
  (h_ab : is_square A B C D) 
  (h_area_abcd : area A B C D = 25)
  (h_AK: dist A K = dist K C)
  (h_BL: dist B L = dist L C)
  (h_CM: dist C M = dist M D)
  (h_DN: dist D N = dist N A)
  (h_right_akb : is_right_triangle A K B)
  (h_right_blc : is_right_triangle B L C)
  (h_right_cmd : is_right_triangle C M D)
  (h_right_dna : is_right_triangle D N A) :
  area K L M N = 6.25 :=
sorry

end area_KLMN_l219_219192


namespace base6_calculation_l219_219655

-- Define the base 6 numbers and their operations
def base6 := ℕ

def to_base10 (n : base6) : ℕ :=
  -- Define function to convert a base 6 number to base 10
  sorry

def from_base10 (n : ℕ) : base6 :=
  -- Define function to convert a base 10 number to base 6
  sorry

-- Define base 6 numbers in base 10 form for easy calculation
def b6_15 : base6 := 11 -- 1 * 6 + 5
def b6_4 : base6 := 4 -- 4
def b6_20 : base6 := 12 -- 2 * 6 + 0
def b6_31 : base6 := 19 -- 3 * 6 + 1

-- Define the operations in base 6
def b6_sub (a b : base6) : base6 :=
  from_base10 (to_base10 a - to_base10 b)

def b6_add (a b : base6) : base6 :=
  from_base10 (to_base10 a + to_base10 b)

-- The proof statement:
theorem base6_calculation : b6_add (b6_sub b6_15 b6_4) b6_20 = b6_31 :=
by sorry

end base6_calculation_l219_219655


namespace probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l219_219835
-- Import all necessary libraries

-- Define the conditions as variables
variable (n k : ℕ) (p q : ℚ)
variable (dice_divisible_by_3_prob : ℚ)
variable (dice_not_divisible_by_3_prob : ℚ)

-- Assign values based on the problem statement
noncomputable def cond_replicate_n_fair_12_sided_dice := n = 7
noncomputable def cond_exactly_k_divisible_by_3 := k = 3
noncomputable def cond_prob_divisible_by_3 := dice_divisible_by_3_prob = 1 / 3
noncomputable def cond_prob_not_divisible_by_3 := dice_not_divisible_by_3_prob = 2 / 3

-- The theorem statement with the final answer incorporated
theorem probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice :
  cond_replicate_n_fair_12_sided_dice n →
  cond_exactly_k_divisible_by_3 k →
  cond_prob_divisible_by_3 dice_divisible_by_3_prob →
  cond_prob_not_divisible_by_3 dice_not_divisible_by_3_prob →
  p = (35 : ℚ) * ((1 / 3) ^ 3) * ((2 / 3) ^ 4) →
  q = (560 / 2187 : ℚ) →
  p = q :=
by
  intros
  sorry

end probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l219_219835


namespace max_monthly_donation_l219_219388

def charity_per_day (k : ℕ) : ℕ := k / 3

def total_working_hours (L : ℕ) (days : ℕ) : ℕ := 3 * L * days

def total_rest_hours (k_per_day : ℕ) (days : ℕ) : ℕ := k_per_day * days

noncomputable def monthly_donation (k_total : ℕ) : ℕ := k_total / 3

theorem max_monthly_donation 
  (cost_per_hour : ℕ) (investment_income : ℕ) (monthly_expenses : ℕ)
  (working_days : ℕ) (sleep_hours : ℕ) (rest_hours_per_day : ℕ) 
  (L : ℕ) (k_per_day : ℕ)
  (daily_income := cost_per_hour * L) 
  (max_L := (24 - sleep_hours - 2 * L - rest_hours_per_day) / 3)
  (total_income := investment_income + (cost_per_hour * max_L * working_days))
  (total_expenses := monthly_expenses + charity_per_day(k_per_day) * working_days) :
  monthly_donation (total_rest_hours k_per_day working_days) = 70 :=
by
  sorry

end max_monthly_donation_l219_219388


namespace scientific_notation_of_0_00003_l219_219392

theorem scientific_notation_of_0_00003 :
  0.00003 = 3 * 10^(-5) :=
sorry

end scientific_notation_of_0_00003_l219_219392


namespace find_triples_l219_219861

-- Definitions of the problem conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def satisfies_equation (a b p : ℕ) : Prop := a^p = factorial b + p

-- The main theorem statement based on the problem conditions
theorem find_triples :
  (satisfies_equation 2 2 2 ∧ is_prime 2) ∧
  (satisfies_equation 3 4 3 ∧ is_prime 3) ∧
  (∀ (a b p : ℕ), (satisfies_equation a b p ∧ is_prime p) → (a, b, p) = (2, 2, 2) ∨ (a, b, p) = (3, 4, 3)) :=
by
  -- Proof to be filled
  sorry

end find_triples_l219_219861


namespace find_a4_l219_219889

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (T_7 : ℝ)

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom common_ratio_ne_one : q ≠ 1
axiom product_first_seven_terms : (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) = 128

-- Goal
theorem find_a4 : a 4 = 2 :=
sorry

end find_a4_l219_219889


namespace sum_prime_factors_143_l219_219323

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ 143 = p * q ∧ p + q = 24 :=
begin
  use 13,
  use 11,
  repeat { split },
  { exact nat.prime_of_four_divisors 13 (by norm_num) },
  { exact nat.prime_of_four_divisors 11 (by norm_num) },
  { norm_num },
  { norm_num }
end

end sum_prime_factors_143_l219_219323


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219748

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219748


namespace like_terms_exponent_l219_219940

theorem like_terms_exponent (m n : ℤ) (h₁ : n = 2) (h₂ : m = 1) : m - n = -1 :=
by
  sorry

end like_terms_exponent_l219_219940


namespace intercepts_sum_eq_eight_l219_219461

def parabola_x_y (x y : ℝ) := x = 3 * y^2 - 9 * y + 5

theorem intercepts_sum_eq_eight :
  ∃ (a b c : ℝ), parabola_x_y a 0 ∧ parabola_x_y 0 b ∧ parabola_x_y 0 c ∧ a + b + c = 8 :=
sorry

end intercepts_sum_eq_eight_l219_219461


namespace probability_even_sum_two_cards_l219_219796

theorem probability_even_sum_two_cards :
  let cards := [1, 2, 3, 4, 5]
  let total_pairs := Finset.card (Finset.powersetLen 2 (Finset.range 6))
  let even_sum_pairs := Finset.card (Finset.filter (λ s, (s.sum Mod 2 = 0)) (Finset.powersetLen 2 (Finset.range 6)))
  (even_sum_pairs : ℚ) / total_pairs = 2 / 5 := 
by
  sorry

end probability_even_sum_two_cards_l219_219796


namespace max_min_terms_in_sequence_l219_219075

noncomputable def sequence (n : ℕ) : ℝ :=
  (2 / 3)^(n - 1) * ((2 / 3)^(n - 1) - 1)

theorem max_min_terms_in_sequence:
  sequence 1 = 0 ∧ sequence 3 = -(20 / 81) ∧ 
  (∀ n : ℕ, n ≠ 1 → sequence n ≤ 0) ∧ 
  (∀ n : ℕ, n > 3 → sequence n > sequence (n + 1)) :=
by
  sorry

end max_min_terms_in_sequence_l219_219075


namespace subsets_with_at_least_four_adjacent_chairs_l219_219727

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l219_219727


namespace certain_number_l219_219089

theorem certain_number (N : ℝ) (k : ℝ) 
  (h1 : (1 / 2) ^ 22 * N ^ k = 1 / 18 ^ 22) 
  (h2 : k = 11) 
  : N = 81 := 
by
  sorry

end certain_number_l219_219089


namespace log_base_a_of_b_l219_219041

noncomputable def imaginary_unit := Complex.i
def z := (2 - imaginary_unit) * (1 + imaginary_unit)^2
def real_part_z := 2  -- given in conditions
def imaginary_part_z := 4  -- given in conditions

theorem log_base_a_of_b : Real.logBase real_part_z imaginary_part_z = 2 := by
  sorry

end log_base_a_of_b_l219_219041


namespace find_f_2_l219_219681

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2 (f : ℝ → ℝ)
  (h : ∀ x, x ≠ 0 → f x - 3 * f (1 / x) = 3^x)
  : f 2 = -9/8 - 3 * real.sqrt 3 / 8 := sorry

end find_f_2_l219_219681


namespace sum_prime_factors_143_l219_219312

theorem sum_prime_factors_143 : 
  ∃ p q : ℕ, prime p ∧ prime q ∧ 143 = p * q ∧ p + q = 24 := 
by
  let p := 11
  let q := 13
  have h1 : 143 = p * q := by norm_num
  have h2 : prime p := by norm_num
  have h3 : prime q := by norm_num
  have h4 : p + q = 24 := by norm_num
  exact ⟨p, q, h2, h3, h1, h4⟩  

end sum_prime_factors_143_l219_219312


namespace minimum_length_diagonal_BD_l219_219635

theorem minimum_length_diagonal_BD
  (A B C D I : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_eq1 : dist B C = 2)
  (h_eq2 : dist C D = 2)
  (h_incenter : Incenter I A B D)
  (h_AI : dist A I = 2) :
  ∃(BD_min : ℝ), BD_min = 2 * Real.sqrt 3 ∧ ∀(BD : ℝ), dist B D = BD → BD ≥ BD_min :=
sorry

end minimum_length_diagonal_BD_l219_219635


namespace find_tangent_line_equation_monotonic_intervals_range_of_a_l219_219534

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a * Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + (1 + a) / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := - (1 + a) / x
def e := Real.exp 1

theorem find_tangent_line_equation (a x y : ℝ) : a = 2 → x = 1 → f x a = y → x + y - 2 = 0 := by
  sorry

theorem monotonic_intervals (a x : ℝ) :
  (a > -1 → (∀ (x : ℝ), 0 < x ∧ x < a + 1 → h x a < h (a + 1) a) ∧ 
  ∀ (x : ℝ), x > a + 1 → h x a > h (a + 1) a) ∧ 
  (a ≤ -1 → (∀ (x : ℝ), 0 < x → h x a > 0)) := by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ e ∧ f x₀ a ≤ g x₀ a) ↔ (a ≥ (e^2 + 1) / (e - 1) ∨ a ≤ -2) := by
  sorry

end find_tangent_line_equation_monotonic_intervals_range_of_a_l219_219534


namespace even_factors_count_l219_219553

theorem even_factors_count (n : ℕ) (a b c : ℕ) :
  n = 2^2 * 3^1 * 7^2 →
  (∀ a b c, (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 1) ∧ (0 ≤ c ∧ c ≤ 2) →
    ∃ k, k = 2^a * 3^b * 7^c ∧ k ∣ n ∧ 1 ≤ a) →
  finset.card {k | ∃ a b c, k = 2^a * 3^b * 7^c ∧ k ∣ n ∧ 1 ≤ a} = 12 := 
by 
  sorry

end even_factors_count_l219_219553


namespace check_stdEquation_check_eccentricity_check_slopeRange_l219_219050

/-- Definition of the ellipse in the problem and the given conditions. -/
structure EllipseProblem :=
  (a b c : ℝ)
  (h : a > b ∧ b > 0)
  (h_foci : |2 * c| = 6) -- |F1F2| = 2c
  (h_perimeter : 2 * a + 2 * c = 16)
  (h_equation : ∀ x y: ℝ, x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def stdEquation := 
  ∃ e: EllipseProblem, (e.a = 5) ∧ (e.b ^ 2 = 16) ∧ (∀ x y, x^2 / 25 + y^2 / 16 = 1)

noncomputable def eccentricity := 
  ∃ e: EllipseProblem, (e.a ^ 2 = 12) ∧ (e.b ^ 2 = 3) ∧ (e.c = e.a / 2) ∧ (e.a / e.b = 2 / 3) ∧ (e.h_perimeter = 1 / 2)

noncomputable def slopeRange :=
  ∃ e: EllipseProblem, 
  (e.a ^ 2 = 12) ∧ 
  (e.b ^ 2 = 3) ∧ 
  (e.h_perimeter = ∃ k_1: ℝ, -2 < k_1 ∧ k_1 < -1 ∧ ∀ k_2: ℝ, k_2 = -1 / (4 * k_1) ∧ (1 / 8 < k_2 ∧ k_2 < 1 / 4))

-- Statements
theorem check_stdEquation : stdEquation :=
sorry

theorem check_eccentricity : eccentricity :=
sorry

theorem check_slopeRange : slopeRange :=
sorry

end check_stdEquation_check_eccentricity_check_slopeRange_l219_219050


namespace katrina_cookies_left_l219_219605

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l219_219605


namespace min_total_cost_l219_219465

-- Defining the variables involved
variables (x y z : ℝ)
variables (h : ℝ := 1) (V : ℝ := 4)
def base_cost (x y : ℝ) : ℝ := 200 * (x * y)
def side_cost (x y : ℝ) (h : ℝ) : ℝ := 100 * (2 * (x + y)) * h
def total_cost (x y h : ℝ) : ℝ := base_cost x y + side_cost x y h

-- The condition that volume is 4 m^3
theorem min_total_cost : 
  (∀ x y, x * y = V) → 
  ∃ x y, total_cost x y h = 1600 :=
by
  sorry

end min_total_cost_l219_219465


namespace find_a22_l219_219239

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l219_219239


namespace area_of_TURS_l219_219990

theorem area_of_TURS (PQ RS : ℕ) (h1 : PQ = 10) (h2 : RS = 30) (altitude : ℕ) (h3 : altitude = 18) 
  (T U PR QS : ℝ) (hT : T = (PQ + PR) / 2) (hU : U = (QS + RS) / 2) : 
  let TU := (PQ + RS) / 2 in
  let h_TURS := altitude / 2 in
  let b1 := TU in
  let b2 := RS in
  1/2 * (b1 + b2) * h_TURS = 225 := 
sorry

end area_of_TURS_l219_219990


namespace cos_add_sin_eq_neg_7_over_13_l219_219909

noncomputable def cos_add_sin (x y r : ℤ) (hx : x = -12) (hy : y = 5) (hr : r = 13) : ℚ :=
  let cosθ := (x : ℚ) / (r : ℚ)
  let sinθ := (y : ℚ) / (r : ℚ)
  cosθ + sinθ

theorem cos_add_sin_eq_neg_7_over_13 :
  cos_add_sin (-12) 5 13 (-12).rfl (5).rfl (13).rfl = -7/13 :=
by
  sorry

end cos_add_sin_eq_neg_7_over_13_l219_219909


namespace correct_statements_l219_219898

noncomputable def complex_norm (z : ℂ) := Complex.abs z

theorem correct_statements (z z1 z2 : ℂ) (conj_z : ℂ) :
  conj_z = Complex.conj z →
  (z * conj_z = complex_norm z ^ 2) ∧
  (complex_norm (z1 * z2) = complex_norm z1 * complex_norm z2) ∧
  (complex_norm (z - 1) = 1 → ∃ z' : ℂ, complex_norm (z' + 1) = 1) := 
by {
  sorry
}

end correct_statements_l219_219898


namespace sum_of_circle_areas_l219_219255

theorem sum_of_circle_areas (r s t : ℝ) (h1 : r + s = 5) (h2 : s + t = 12) (h3 : r + t = 13) :
  real.pi * (r^2 + s^2 + t^2) = 113 * real.pi :=
by
  sorry

end sum_of_circle_areas_l219_219255


namespace findAsymptoteSlopes_l219_219515

-- Definitions of the hyperbola and its properties
def isHyperbola (x y b : ℝ) :=
  x^2 / 4 - y^2 / b^2 = 1

def isFocusPoint (F₁ F₂ : ℝ × ℝ) := 
  F₁.1 = -2 ∧ F₁.2 = 0 ∧ F₂.1 = 2 ∧ F₂.2 = 0

def onHyperbola (P : ℝ × ℝ) (b : ℝ) :=
  isHyperbola P.1 P.2 b

def angle120 (F₁ P F₂ : ℝ × ℝ) :=
  ∠P F₁ F₂ = 120

def formsArithmeticSeq (m n : ℝ) :=
  m - n = 4

-- Problem statement
theorem findAsymptoteSlopes (F₁ F₂ P : ℝ × ℝ) (b : ℝ) :
  isFocusPoint F₁ F₂ →
  onHyperbola P b →
  angle120 F₁ P F₂ →
  formsArithmeticSeq |PF₁| |PF₂| →
  (b = 3 * Real.sqrt 5 → 
  ∃ k: ℝ, (k = 3 * Real.sqrt 5 / 2 ∨ k = -3 * Real.sqrt 5 / 2)) := 
sorry

end findAsymptoteSlopes_l219_219515


namespace sum_of_prime_factors_143_l219_219300

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219300


namespace original_square_area_ratio_l219_219957

theorem original_square_area_ratio (s : ℝ) (sqrt5_pos : 0 < Real.sqrt 5) :
  let A_original := s^2 in
  let A_resultant := (s * Real.sqrt 5)^2 in
  A_original / A_resultant = 1 / 5 :=
by
  sorry

end original_square_area_ratio_l219_219957


namespace number_of_subsets_with_four_adj_chairs_l219_219719

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219719


namespace log2_yx_value_l219_219947

theorem log2_yx_value (x y : ℝ) (h : (8 * y - 1)^2 + |x - 16 * y| = 0) : log 2 (y ^ x) = -6 :=
by
  sorry

end log2_yx_value_l219_219947


namespace well_defined_set_example_l219_219368

/-- 
  The problem is to prove that out of the given descriptions, only the set of all natural numbers less than 10 
  forms a well-defined set.
-/
theorem well_defined_set_example :
  (∃ s : set ℕ, s = {n | n < 10}) ∧ 
  (¬ (∃ s : set (Class × ℕ), s = {student | student.1.height > 175})) ∧ 
  (¬ (∃ s : set ℝ, s = {x | abs (x - 1) < ε})) ∧ 
  (¬ (∃ s : set (Class × Personality), s = {student | student.1.outgoing = true})) :=
by
  sorry

end well_defined_set_example_l219_219368


namespace circle_radius_l219_219402

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 180 * π) : r = 10 := 
by
  sorry

end circle_radius_l219_219402


namespace triangular_pyramid_volume_l219_219458

-- Define the points A, B, C
def A : Point := (0, 0)
def B : Point := (30, 0)
def C : Point := (10, 20)

-- Define the midpoints D, E, F
def D : Point := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
def E : Point := ((C.1 + A.1) / 2, (C.2 + A.2) / 2)
def F : Point := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Question and conditions in Lean 4 statement
theorem triangular_pyramid_volume (A B C D E F : Point) (hD : D = (20, 10)) (hE : E = (5, 10)) (hF : F = (15, 0)) :
  volume_of_pyramid A B C D E F = 111.17 := by
    -- The proof goes here.
    sorry

end triangular_pyramid_volume_l219_219458


namespace circle_symmetric_line_a_value_l219_219675

theorem circle_symmetric_line_a_value :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, (x, y) = (-1, 2)) →
  (∀ x y : ℝ, ax + y + 1 = 0) →
  a = 3 :=
by
  sorry

end circle_symmetric_line_a_value_l219_219675


namespace sum_of_prime_factors_of_143_l219_219328

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219328


namespace number_of_divisors_of_n_cubed_l219_219813

theorem number_of_divisors_of_n_cubed (n : ℕ) (h : ∃ p : ℕ, nat.prime p ∧ n = p ^ 3) : nat.num_divisors (n ^ 3) = 10 :=
by sorry

end number_of_divisors_of_n_cubed_l219_219813


namespace batsman_stats_l219_219572

theorem batsman_stats (total_runs_16_inns : ℕ → ℝ) (total_boundaries : ℕ → ℕ) (innings_played : ℕ)
(score_17th_inning : ℕ) (increase_in_average : ℝ) :
  innings_played = 17 →
  score_17th_inning = 84 →
  increase_in_average = 2.5 →
  total_boundaries 16 = 60 →
  total_boundaries 17 = 72 →
  (∃ (A B : ℝ), A = total_runs_16_inns 16 / 16 ∧
                (total_runs_16_inns 17 = total_runs_16_inns 16 + score_17th_inning) ∧ 
                (total_runs_16_inns 17 / innings_played = A + increase_in_average) ∧ 
                B = total_boundaries 17 / innings_played ∧ 
                A + increase_in_average = 44 ∧ 
                B ≈ 4.235) :=
by
  intros h_inn_played h_score_17th h_inc_avg h_boundaries_16 h_boundaries_17
  use (total_runs_16_inns 16 / 16), (total_boundaries 17 / innings_played)
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  split
  { sorry }
  sorry

end batsman_stats_l219_219572


namespace number_of_subsets_with_four_adj_chairs_l219_219720

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219720


namespace number_of_subsets_with_four_adj_chairs_l219_219722

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219722


namespace line_equation_l219_219687

theorem line_equation (a : ℝ) (ha : a < 3) :
  ∃ l : ℝ → ℝ, (∀ x y, (x^2 + y^2 + 2*x - 4*y + a = 0 → (0,1) = ((x + xa)/2, (y + ya)/2)) → 
              (∀ x, l x = x + 1)) :=
begin
  sorry
end

end line_equation_l219_219687


namespace rectangular_field_diagonal_l219_219188

noncomputable def field_diagonal_length (a b : ℕ) : ℕ :=
  (a^2 + b^2) ^ (1 / 2 : ℚ)

theorem rectangular_field_diagonal :
  ∀ (a : ℕ) (area : ℕ), a = 15 → area = 120 → field_diagonal_length 15 8 = 17 :=
by
  intros a area h1 h2
  have b : ℕ := 8
  rw [h1, h2]
  sorry

end rectangular_field_diagonal_l219_219188


namespace fox_jeans_price_l219_219878

theorem fox_jeans_price (F : ℝ) (P : ℝ) 
  (pony_price : P = 18) 
  (total_savings : 3 * F * 0.08 + 2 * P * 0.14 = 8.64)
  (total_discount_rate : 0.08 + 0.14 = 0.22)
  (pony_discount_rate : 0.14 = 13.999999999999993 / 100) 
  : F = 15 :=
by
  sorry

end fox_jeans_price_l219_219878


namespace sphere_surface_area_l219_219052

open EuclideanGeometry

-- Necessary constants and definitions
constant A B C O : EuclideanSpace
constant angle_BAC : Real := 2 * Real.pi / 3
constant BC : Real := 4 * Real.sqrt 3
constant dist_center_plane_O : Real := 3
constant R : Real := 5
constant surface_area : Real := 4 * Real.pi * R ^ 2

-- Conditions as properties
axiom point_on_sphere_A : is_on_sphere A O R
axiom point_on_sphere_B : is_on_sphere B O R
axiom point_on_sphere_C : is_on_sphere C O R
axiom given_angle_BAC : angle (B, A, C) = angle_BAC
axiom given_BC : distance B C = BC
axiom given_distance_to_plane : distance_to_plane O (A, B, C) = dist_center_plane_O

-- Statement to be proved
theorem sphere_surface_area : surface_area = 100 * Real.pi :=
by
  sorry

end sphere_surface_area_l219_219052


namespace smallest_n_sqrt_12n_integer_l219_219057

theorem smallest_n_sqrt_12n_integer : ∃ n : ℕ, (n > 0) ∧ (∃ k : ℕ, 12 * n = k^2) ∧ n = 3 := by
  sorry

end smallest_n_sqrt_12n_integer_l219_219057


namespace two_transfers_not_sufficient_three_transfers_sufficient_l219_219711

-- Define initial candy counts in each bag
def initial_candies : List ℕ := [7, 8, 9, 11, 15]

-- Define the target candy count in each bag
def target_candies : ℕ := 10

-- Prove that you cannot equalize the candies with two transfers
theorem two_transfers_not_sufficient (bags : List ℕ) (target : ℕ) : 
  bags = initial_candies → target = target_candies → 
  ¬ (∃ transfers : List (ℕ × ℕ × ℕ), transfers.length = 2 ∧ balanced_candies bags target transfers) :=
by 
  sorry

-- Prove that you can equalize the candies with three transfers
theorem three_transfers_sufficient (bags : List ℕ) (target : ℕ) : 
  bags = initial_candies → target = target_candies → 
  ∃ transfers : List (ℕ × ℕ × ℕ), transfers.length = 3 ∧ balanced_candies bags target transfers :=
by 
  sorry

-- Helper function to determine if candies can be balanced with given transfers
noncomputable def balanced_candies : List ℕ → ℕ → List (ℕ × ℕ × ℕ) → Prop :=
λ bags target transfers, sorry  -- Definition will simulate the transfer process and check equality

end two_transfers_not_sufficient_three_transfers_sufficient_l219_219711


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l219_219744

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l219_219744


namespace problem_I_problem_II_l219_219916

-- Define the function f
def f (x : ℝ) (a b : ℝ) := 2 * x + a * x^2 + b * cos x

-- Define the conditions and statements
theorem problem_I (a b : ℝ)
  (hx : f (π / 2) a b = 3 * π / 4)
  (h_deriv : deriv (λ x, f x a b) (π / 2) = 0) :
  a = -1 / π ∧ b = 1 ∧ ∀ x, 0 ≤ x ∧ x ≤ π / 2 → deriv (λ x, f x (-1 / π) 1) x > 0 :=
sorry

theorem problem_II (x1 x2 : ℝ)
  (hx1 : 0 < x1) (hx2 : x1 < x2) (hx3 : x2 < π)
  (hf : f x1 (-1 / π) 1 = f x2 (-1 / π) 1) :
  deriv (λ x, f x (-1 / π) 1) ((x1 + x2) / 2) < 0 :=
sorry

end problem_I_problem_II_l219_219916


namespace value_at_4_l219_219073

-- Define the power function f
def f (x : ℝ) (α : ℝ) := x ^ α

-- Specify the given condition
variable (α : ℝ)
axiom passes_through_point : f 2 α = Real.sqrt 2

-- The theorem to prove
theorem value_at_4 : f 4 (1/2) = 2 :=
by
  have h : α = 1/2 := sorry
  rw [h]
  sorry

end value_at_4_l219_219073


namespace point_inside_circle_l219_219575

theorem point_inside_circle 
    (P O : Type) 
    (radius : ℝ)
    (PO : ℝ)
    (h_radius : radius = 5)
    (h_PO : PO = 4) : 
    PO < radius :=
by {
    rw [h_radius, h_PO],
    norm_num,
    sorry
}

end point_inside_circle_l219_219575


namespace chairs_adjacent_subsets_l219_219733

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l219_219733


namespace least_subtraction_divisibility_l219_219386

theorem least_subtraction_divisibility :
  ∃ k : ℕ, 427398 - k = 14 * n ∧ k = 6 :=
by
  use 6
  sorry

end least_subtraction_divisibility_l219_219386


namespace sum_prime_factors_of_143_l219_219351

theorem sum_prime_factors_of_143 :
  let is_prime (n : ℕ) := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0 in
  ∃ (a b : ℕ), is_prime a ∧ is_prime b ∧ 143 = a * b ∧ a ≠ b ∧ (a + b = 24) :=
by
  sorry

end sum_prime_factors_of_143_l219_219351


namespace range_of_k_for_real_roots_l219_219874

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 :=
by
  sorry

end range_of_k_for_real_roots_l219_219874


namespace false_propositions_l219_219941

noncomputable theory

-- Definitions
variables {P : Type} [point : P]
variables {l m : line} (h_skew : skew l m)

-- Propositions
def parallel_to_both (P : point) (l m : line) : Prop :=
  ∃ k, through P k ∧ parallel k l ∧ parallel k m

def perpendicular_to_both (P : point) (l m : line) : Prop :=
  ∃ k, through P k ∧ perpendicular k l ∧ perpendicular k m

def intersects_both (P : point) (l m : line) : Prop :=
  ∃ k, through P k ∧ intersects k l ∧ intersects k m

def skew_to_both (P : point) (l m : line) : Prop :=
  ∃ k, through P k ∧ skew k l ∧ skew k m

-- Proof statement
theorem false_propositions (P : point) (l m : line) (h_skew : skew l m) :
  ¬ (parallel_to_both P l m) ∧ ¬ (intersects_both P l m) ∧ ∃ k, through P k ∧ skew k l ∧ skew k m :=
begin
  sorry -- Here the proof would go
end

end false_propositions_l219_219941


namespace john_annual_profit_is_1800_l219_219142

def tenant_A_monthly_payment : ℕ := 350
def tenant_B_monthly_payment : ℕ := 400
def tenant_C_monthly_payment : ℕ := 450
def john_monthly_rent : ℕ := 900
def utility_cost : ℕ := 100
def maintenance_fee : ℕ := 50

noncomputable def annual_profit : ℕ :=
  let total_monthly_income := tenant_A_monthly_payment + tenant_B_monthly_payment + tenant_C_monthly_payment
  let total_monthly_expenses := john_monthly_rent + utility_cost + maintenance_fee
  let monthly_profit := total_monthly_income - total_monthly_expenses
  monthly_profit * 12

theorem john_annual_profit_is_1800 : annual_profit = 1800 := by
  sorry

end john_annual_profit_is_1800_l219_219142


namespace alpha_beta_purchase_ways_l219_219439

-- Definitions matching the conditions
def oreo_flavors : ℕ := 5
def milk_flavors : ℕ := 3

def products (alpha_oreos : ℕ) (beta_products : ℕ) : ℕ :=
  -- Alpha's choices
  if alpha_oreos > oreo_flavors then 0
  else 
    let alpha_ways := Nat.choose oreo_flavors alpha_oreos in
    let total_products := oreo_flavors + milk_flavors in
    
    -- Beta's choices
    let beta_ways := Nat.partitions_pieces_with_repeats total_products beta_products in
    alpha_ways * beta_ways

-- Theorem matching the question and correct answer
theorem alpha_beta_purchase_ways :
  products 3 0 + products 2 1 + products 1 2 + products 0 3 = 342 :=
begin
  sorry
end

end alpha_beta_purchase_ways_l219_219439


namespace rectangle_area_288_l219_219415

/-- A rectangle contains eight circles arranged in a 2x4 grid. Each circle has a radius of 3 inches.
    We are asked to prove that the area of the rectangle is 288 square inches. --/
noncomputable def circle_radius : ℝ := 3
noncomputable def circles_per_width : ℕ := 2
noncomputable def circles_per_length : ℕ := 4
noncomputable def circle_diameter : ℝ := 2 * circle_radius
noncomputable def rectangle_width : ℝ := circles_per_width * circle_diameter
noncomputable def rectangle_length : ℝ := circles_per_length * circle_diameter
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

theorem rectangle_area_288 :
  rectangle_area = 288 :=
by
  -- Proof of the area will be filled in here.
  sorry

end rectangle_area_288_l219_219415


namespace cos_2alpha_sin_alpha_plus_beta_l219_219495

noncomputable def alpha : ℝ := sorry
noncomputable def beta : ℝ := sorry

axiom alpha_beta_in_interval : (0 < alpha) ∧ (alpha < real.pi) ∧ (0 < beta) ∧ (beta < real.pi)
axiom cos_alpha_eq : real.cos alpha = 4 / 5
axiom sin_alpha_minus_beta_eq : real.sin (alpha - beta) = 5 / 13

theorem cos_2alpha : real.cos (2 * alpha) = 7 / 25 :=
by { sorry }

theorem sin_alpha_plus_beta : real.sin (alpha + beta) = 253 / 325 :=
by { sorry }

end cos_2alpha_sin_alpha_plus_beta_l219_219495


namespace size_of_angle_C_max_value_of_a_add_b_l219_219977

variable (A B C a b c : ℝ)
variable (h₀ : 0 < A ∧ A < π / 2)
variable (h₁ : 0 < B ∧ B < π / 2)
variable (h₂ : 0 < C ∧ C < π / 2)
variable (h₃ : a = 2 * c * sin A / sqrt 3)
variable (h₄ : a * a + b * b - 2 * a * b * cos (π / 3) = c * c)

theorem size_of_angle_C (h₅: a ≠ 0):
  C = π / 3 :=
by sorry

theorem max_value_of_a_add_b (h₆: c = 2):
  a + b ≤ 4 :=
by sorry

end size_of_angle_C_max_value_of_a_add_b_l219_219977


namespace coeff_x6_in_expansion_l219_219773

theorem coeff_x6_in_expansion : 
  (Polynomial.coeff ((1 - 3 * Polynomial.X ^ 3) ^ 7 : Polynomial ℤ) 6) = 189 :=
by
  sorry

end coeff_x6_in_expansion_l219_219773


namespace concurrency_and_concurrency_point_on_Gamma_l219_219054

-- Let ABC be a triangle and T a point inside of it
variables {A B C T A1 B1 C1 A2 B2 C2 : Type*}
variable {Γ : circle}

-- Given conditions
variables (h1 : T ∈ interior (triangle A B C))
          (h2 : symmetric_to T with_respect_to BC = A1)
          (h3 : symmetric_to T with_respect_to CA = B1)
          (h4 : symmetric_to T with_respect_to AB = C1)
          (h5 : circumcircle (triangle A1 B1 C1) = Γ)
          (h6 : second_intersection (line A1 T) Γ = A2)
          (h7 : second_intersection (line B1 T) Γ = B2)
          (h8 : second_intersection (line C1 T) Γ = C2)

-- Prove the concurrency and the point of concurrency lies on Γ
theorem concurrency_and_concurrency_point_on_Gamma :
  concurrent (line (A A2)) (line (B B2)) (line (C C2)) ∧ concurrency_point ∈ Γ :=
sorry

end concurrency_and_concurrency_point_on_Gamma_l219_219054


namespace max_negatives_l219_219266

theorem max_negatives (a : Fin 101 → ℤ) (h : ∀ i : Fin 101, a i > a ((i + 1) % 101) * a ((i + 2) % 101)) : 
    ∃k : ℕ, k ≤ 101 ∧ (∀l : ℕ, l ≤ 101 → (∀i in Finset.range l, a i < 0) ∧ l = 67) :=
sorry

end max_negatives_l219_219266


namespace find_a22_l219_219238

variable (a : ℕ → ℝ)
variable (h : ∀ n, 1 ≤ n ∧ n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
variable (h99 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
variable (h100 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
variable (h10 : a 10 = 10)

theorem find_a22 : a 22 = 10 := sorry

end find_a22_l219_219238


namespace sin_B_right_triangle_l219_219586

theorem sin_B_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (h : ∃ (P Q R : A), 
       ∃ (a b c : ℝ),
       (b = 12) ∧ 
       (c = 20) ∧ 
       b^2 + c^2 = a^2 ∧ 
       has_angle Q 90) : 
       ∃ θ : ℝ, (b/a = sin θ) ∧ (θ = (3*sqrt(34))/34) :=
by sorry

end sin_B_right_triangle_l219_219586


namespace third_median_to_third_altitude_ratio_l219_219112

-- Definitions based on the problem conditions.
def scalene_triangle (A B C : Point) : Prop :=
  ¬collinear A B C ∧ ∀ (AB BC CA : Segment), AB ≠ BC ∧ BC ≠ CA ∧ AB ≠ CA

def is_median (A B C M : Point) : Prop :=
  midpoint M B C ∧ B ≠ C ∧ line_segment A M ∈ {X | ∃ (P Q : Point), midpoint P Q X}

def is_altitude (A B C P : Point) : Prop :=
  line_segment B C ⟂ line_segment A P ∧ P ∈ line_segment B C

def ratio_of_third_median_to_third_altitude 
  (A B C A' B' C' : Point)
  (hm : scalene_triangle A B C)
  (hma1 : is_median A A' B C ∧ is_altitude B A' B C)
  (hma2 : is_median B B' A C ∧ is_altitude C B' A C)
  : ℚ :=
7 / 2

-- Statement that the ratio of the third median to the third altitude is 7:2.
theorem third_median_to_third_altitude_ratio 
  (A B C G A' B' C' : Point)
  (H_triangle : scalene_triangle A B C)
  (H_medians_equal_altitudes : is_median A A' B C ∧ is_altitude B A' B C ∧ is_median B B' A C ∧ is_altitude C B' A C)
  : ratio_of_third_median_to_third_altitude A B C A' B' C' H_triangle H_medians_equal_altitudes = 7 / 2 :=
sorry

end third_median_to_third_altitude_ratio_l219_219112


namespace max_possible_area_l219_219972

-- Define the radii of the four circles
def r1 : ℝ := 2
def r2 : ℝ := 4
def r3 : ℝ := 6
def r4 : ℝ := 8

-- Circle areas based on given radii
def area (r : ℝ) : ℝ := π * r^2

-- Define the total region area for the circles
def region_area := area r1 + area r2 + area r3 + area r4

-- Given conditions: circles are tangent to two perpendicular lines intersecting at a point, each extending in different quadrants
def tangent_to_lines (r : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x = r ∨ x = -r ∨ y = r ∨ y = -r)

theorem max_possible_area (S : Type) (region_in_S : region_area ≤ S) :
  region_area = 120 * π :=
by
  sorry

end max_possible_area_l219_219972


namespace dog_roaming_area_l219_219646

-- Definition of the problem conditions
def leash_length : ℝ := 1 -- Length of each leash in meters.
def distance_between_people : ℝ := 1 -- The constant distance between Monsieur and Madame Dubois in meters.

-- The expected result is approximately 1.228 m^2. 
-- The exact mathematical expression for the area:
def expected_dog_area : ℝ := (2 * Real.pi / 3) - (Real.sqrt 3 / 2)

-- The Lean theorem to be proved
theorem dog_roaming_area : 
  (leash_length = 1) → 
  (distance_between_people = 1) → 
  let actual_area := expected_dog_area in
  actual_area ≈ 1.228 :=
by
  intros h1 h2
  unfold expected_dog_area
  sorry -- Proof to be completed

end dog_roaming_area_l219_219646


namespace greatest_prime_factor_of_expression_l219_219281

theorem greatest_prime_factor_of_expression 
    (e : ℕ) (h : e = 5^5 + 12^3) 
    (h_eq : 5^5 + 12^3 = 4853) :
    ∃ p : ℕ, p = 19 ∧ prime p ∧ (∀ q : ℕ, prime q → q ∣ e → q ≤ p) :=
by
  -- We state the given conditions
  have h_expr := h_eq,
  rw h_eq at h, 
  
  -- State the prime factors found in solution steps and their primality
  have h_prime_3 : prime 3, from prime_three,
  have h_prime_5 : prime 5, from prime_five,
  have h_prime_17 : prime 17, from prime_seventeen,
  have h_prime_19 : prime 19, from prime_nineteen,
  
  -- By checking h, e = 4853 and e's prime factors are 3, 5, 17, and 19,
  -- we confirm that 19 is the greatest prime factor
  use 19,
  split,
  { refl, },
  split,
  { exact h_prime_19, },
  intros q h_prime_q h_q_div_e,
  interval_cases q;
  sorry

end greatest_prime_factor_of_expression_l219_219281


namespace max_principals_in_period_l219_219470

-- Define the conditions
def principal_term_length : Nat := 3
def total_period : Nat := 15

-- Statement of the theorem
theorem max_principals_in_period : ∃ n : Nat, n = 5 ∧ (total_period ≤ n * principal_term_length) :=
by
  -- Providing the correct answer directly
  exists 5
  apply And.intro
  . rfl
  . sorry

end max_principals_in_period_l219_219470


namespace parabola_intercepts_min_max_l219_219071

noncomputable def f (θ : ℝ) : ℝ :=
  -((Real.sin θ + 2) ^ 2) + 7

theorem parabola_intercepts_min_max :
  ∀ θ : ℝ, (-(Real.sin θ + 2) ^ 2 + 7 ≥ -2) ∧ (-(Real.sin θ + 2) ^ 2 + 7 ≤ 6) :=
by
  intro θ
  split
  -- Proof for minimum value
  sorry
  -- Proof for maximum value
  sorry

end parabola_intercepts_min_max_l219_219071


namespace all_div_by_25_form_no_div_by_35_l219_219477

noncomputable def exists_div_by_25 (M : ℕ) : Prop :=
∃ (M N : ℕ) (n : ℕ), M = 6 * 10 ^ (n - 1) + N ∧ M = 25 * N ∧ 4 * N = 10 ^ (n - 1)

theorem all_div_by_25_form :
  ∀ M, exists_div_by_25 M → (∃ k : ℕ, M = 625 * 10 ^ k) :=
by
  intro M
  intro h
  sorry

noncomputable def not_exists_div_by_35 (M : ℕ) : Prop :=
∀ (M N : ℕ) (n : ℕ), M ≠ 6 * 10 ^ (n - 1) + N ∨ M ≠ 35 * N

theorem no_div_by_35 :
  ∀ M, not_exists_div_by_35 M :=
by
  intro M
  intro h
  sorry

end all_div_by_25_form_no_div_by_35_l219_219477


namespace area_TURS_l219_219989

variables (P Q R S T U : Type)
variables {PQ RS : ℝ} {altitude_PQRS : ℝ}

-- Conditions
def is_trapezoid (PQRS : Type) (P Q R S : PQRS) := 
  ∃ (PQ RS: ℝ), PQ = 10 ∧ RS = 25 ∧ (PQ // RS ∥ PQRS)

-- Midpoints
def midpoint (X Y M : Type) := 
  ∃ (mx my : ℝ), mx = (X + Y) / 2 ∧ my = (X + Y) / 2

axiom T_midpoint : midpoint P S T
axiom U_midpoint : midpoint Q R U

-- Altitude
def altitude (PQRS : Type) := altitude_PQRS = 15

-- Area of trapezoid
def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  height * (base1 + base2) / 2

theorem area_TURS : 
  is_trapezoid PQRS P Q R S ∧ altitude PQRS ∧ T_midpoint P S T ∧ U_midpoint Q R U →
  trapezoid_area 17.5 25 7.5 = 159.375 :=
by sorry

end area_TURS_l219_219989


namespace quadrilateral_ratio_l219_219571

variables (E F G H A B C D E1 F1 G1 H1 : Type)
variables [nonempty E] [nonempty F] [nonempty G] [nonempty H] [nonempty A] [nonempty B] [nonempty C] [nonempty D] [nonempty E1] [nonempty F1] [nonempty G1] [nonempty H1]

-- Definitions for the segments and ratios
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ)
variables (λ : ℝ)
variables (EF_par_E1F1 FG_par_F1G1 GH_par_G1H1 HE_par_H1E1 : Prop)
variables (quadrilateral_1 : AE / EB * BF / FC * CG / GD * DH / HA = 1)

-- Problem statement: Prove that F1C / CG1 = λ given the conditions
theorem quadrilateral_ratio :
  E1F1_par_EF -> F1G1_par_FG -> G1H1_par_GH -> H1E1_par_HE -> 
  (E1A / AH1 = λ) -> 
  (F1C / CG1 = λ) := 
by sorry

end quadrilateral_ratio_l219_219571


namespace sum_of_prime_factors_of_143_l219_219329

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end sum_of_prime_factors_of_143_l219_219329


namespace number_of_subsets_with_four_adj_chairs_l219_219717

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l219_219717


namespace ordered_pairs_1944_l219_219694

theorem ordered_pairs_1944 :
  ∃ n : ℕ, (∀ x y : ℕ, (x * y = 1944 ↔ x > 0 ∧ y > 0)) → n = 24 :=
by
  sorry

end ordered_pairs_1944_l219_219694


namespace sum_of_prime_factors_143_l219_219293

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_of_prime_factors_143 : 
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 143 = p1 * p2 ∧ p1 + p2 = 24 := by
  sorry

end sum_of_prime_factors_143_l219_219293


namespace quadratic_transforms_l219_219539

theorem quadratic_transforms (p q : ℝ) :
  ∀ (x : ℝ), (x^2 + p * x + q = 0) →
  (x^2 - p * x + q = 0) ∧ (q * x^2 + p * x + 1 = 0) :=
begin
  sorry
end

end quadratic_transforms_l219_219539


namespace parametric_equation_of_line_polar_to_cartesian_eq_l219_219799

-- Definitions of the conditions
def inclination_angle : ℝ := π / 6
def point_P : ℝ × ℝ := (1, 1)
def polar_equation (θ : ℝ) : ℝ := 10 * cos (π / 3 - θ)

-- Lean statements of the proof problems
-- Parametric equation of the line
theorem parametric_equation_of_line (t : ℝ) :
  let a := sqrt 3 / 2
      b := 1 / 2 in
  (1 + a * t, 1 + b * t) = (
    (point_P.1 + a * t),
    (point_P.2 + b * t)
  ) :=
sorry

-- Conversion of polar equation to Cartesian equation
theorem polar_to_cartesian_eq :
  ∀ θ ρ,
  ρ = polar_equation θ →
  ρ^2 = 5 * ρ * cos θ + 5 * sqrt 3 * ρ * sin θ →
  x^2 + y^2 - 5 * x - 5 * sqrt 3 * y = 0 :=
sorry

end parametric_equation_of_line_polar_to_cartesian_eq_l219_219799


namespace Andy_is_1_year_older_l219_219966

variable Rahim_current_age : ℕ
variable Rahim_age_in_5_years : ℕ := Rahim_current_age + 5
variable Andy_age_in_5_years : ℕ := 2 * Rahim_current_age
variable Andy_current_age : ℕ := Andy_age_in_5_years - 5

theorem Andy_is_1_year_older :
  Rahim_current_age = 6 → Andy_current_age = Rahim_current_age + 1 :=
by
  sorry

end Andy_is_1_year_older_l219_219966


namespace num_rel_prime_up_to_2014_10000_eq_4648_l219_219457

-- Definitions for relevant mathematical functions and values.
def is_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def euler_totient (n : ℕ) : ℕ :=
  (List.range n).filter (λ k, is_relatively_prime k n).length

def num_rel_prime_up_to (n m : ℕ) : ℕ :=
  (List.range (m + 1)).filter (λ k, is_relatively_prime k n).length

-- Conditions given in the problem.
def n : ℕ := 10000
def m : ℕ := 2014

-- Ensures that the statement uses Definitions and Conditions from the problem.
theorem num_rel_prime_up_to_2014_10000_eq_4648 :
  num_rel_prime_up_to m n = 4648 :=
  by
    -- Proof skipped, as not required
    sorry

end num_rel_prime_up_to_2014_10000_eq_4648_l219_219457


namespace polynomial_expansion_l219_219473

def P (x : ℝ) : ℝ := 1 + x^2 + 2 * x - x^4
def Q (x : ℝ) : ℝ := 3 - x^3 + 2 * x^2 - 5 * x

theorem polynomial_expansion (x : ℝ) : (P(x) * Q(x)) = x^7 - 2 * x^6 + 4 * x^5 - 3 * x^4 - 2 * x^3 - 4 * x^2 + x + 3 :=
by
  sorry

end polynomial_expansion_l219_219473


namespace binary_representation_of_19_l219_219004

theorem binary_representation_of_19 : nat.to_digits 2 19 = [1, 0, 0, 1, 1] :=
by
  sorry

end binary_representation_of_19_l219_219004


namespace smallest_coins_taken_by_blind_pew_l219_219574

variable (num_chests num_boxes_per_chest num_coins_per_box num_locks_available : ℕ)

-- Conditions
def chests_in_trunk : ℕ := 5
def boxes_per_chest : ℕ := 4
def coins_per_box : ℕ := 10
def locks_available : ℕ := 9

-- Proof statement
theorem smallest_coins_taken_by_blind_pew :
  (num_chests = chests_in_trunk) →
  (num_boxes_per_chest = boxes_per_chest) →
  (num_coins_per_box = coins_per_box) →
  (num_locks_available = locks_available) →
  (∃ min_coins : ℕ, min_coins = 30) :=
by
  intro h1 h2 h3 h4
  use 30
  sorry

end smallest_coins_taken_by_blind_pew_l219_219574


namespace constant_term_binomial_expansion_l219_219215

theorem constant_term_binomial_expansion (n : ℕ) (hn : n = 6) :
  (2 : ℤ) * (x : ℝ) - (1 : ℤ) / (2 : ℝ) / (x : ℝ) ^ n == -20 := by
  sorry

end constant_term_binomial_expansion_l219_219215


namespace find_a100_l219_219507

def sequence_a (n : ℕ) (S : ℕ → ℚ) : ℚ := 
  if n = 1 then 3 else (3 * (S n) ^ 2) / (3 * (S n) - 2)

def sequence_S (a : ℕ → ℚ) : ℕ → ℚ
| 0       := 0
| (n + 1) := a (n + 1) + sequence_S a n

theorem find_a100 :
  let a := sequence_a in
  let S := sequence_S a in
  a 100 = -9 / 84668 := 
sorry

end find_a100_l219_219507


namespace uniform_mixtures_possible_l219_219707

theorem uniform_mixtures_possible (n : ℕ) (h_n_pos : n > 0) :
  ∃ (f : Fin (n+1) → Fin n → ℝ), 
  (∀ i, ∑ j, f i j = 1) ∧ 
  (∀ j, ∑ i, f i j = 1) ∧ 
  (∃ k, ∀ j, f k j = 0) := 
sorry

end uniform_mixtures_possible_l219_219707


namespace megan_savings_days_l219_219145

theorem megan_savings_days :
  let josiah_saving_rate : ℝ := 0.25
  let josiah_days : ℕ := 24
  let josiah_total := josiah_saving_rate * josiah_days

  let leah_saving_rate : ℝ := 0.5
  let leah_days : ℕ := 20
  let leah_total := leah_saving_rate * leah_days

  let total_savings : ℝ := 28.0
  let josiah_leah_total := josiah_total + leah_total
  let megan_total := total_savings - josiah_leah_total

  let megan_saving_rate := 2 * leah_saving_rate
  let megan_days := megan_total / megan_saving_rate
  
  megan_days = 12 :=
by
  sorry

end megan_savings_days_l219_219145


namespace sum_even_integers_102_to_200_l219_219697

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l219_219697


namespace perpendicular_lines_l219_219097

theorem perpendicular_lines (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0) ∧ (∀ x y : ℝ, 2 * x + m * y - 6 = 0) → m = -1 :=
by
  sorry

end perpendicular_lines_l219_219097


namespace equation_solutions_count_l219_219627

theorem equation_solutions_count (n : ℕ) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 2 * x + 3 * y + z + x^2 = n) →
  (n = 32 ∨ n = 33) :=
sorry

end equation_solutions_count_l219_219627
