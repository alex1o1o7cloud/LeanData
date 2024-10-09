import Mathlib

namespace sum_of_three_pentagons_l673_67382

variable (x y : ℚ)

axiom eq1 : 3 * x + 2 * y = 27
axiom eq2 : 2 * x + 3 * y = 25

theorem sum_of_three_pentagons : 3 * y = 63 / 5 := 
by {
  sorry -- No need to provide proof steps
}

end sum_of_three_pentagons_l673_67382


namespace simplify_expression_solve_fractional_eq_l673_67340

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l673_67340


namespace wine_age_problem_l673_67377

theorem wine_age_problem
  (C F T B Bo : ℕ)
  (h1 : F = 3 * C)
  (h2 : C = 4 * T)
  (h3 : B = (1 / 2 : ℝ) * T)
  (h4 : Bo = 2 * F)
  (h5 : C = 40) :
  F = 120 ∧ T = 10 ∧ B = 5 ∧ Bo = 240 := 
  by
    sorry

end wine_age_problem_l673_67377


namespace motorists_with_tickets_l673_67387

section SpeedingTickets

variables
  (total_motorists : ℕ)
  (percent_speeding : ℝ) -- percent_speeding is 25% (given)
  (percent_not_ticketed : ℝ) -- percent_not_ticketed is 60% (given)

noncomputable def percent_ticketed : ℝ :=
  let speeding_motorists := percent_speeding * total_motorists / 100
  let ticketed_motorists := speeding_motorists * ((100 - percent_not_ticketed) / 100)
  ticketed_motorists / total_motorists * 100

theorem motorists_with_tickets (total_motorists : ℕ) 
  (h1 : percent_speeding = 25)
  (h2 : percent_not_ticketed = 60) :
  percent_ticketed total_motorists percent_speeding percent_not_ticketed = 10 := 
by
  unfold percent_ticketed
  rw [h1, h2]
  sorry

end SpeedingTickets

end motorists_with_tickets_l673_67387


namespace total_selling_price_correct_l673_67366

-- Definitions of initial purchase prices in different currencies
def init_price_eur : ℕ := 600
def init_price_gbp : ℕ := 450
def init_price_usd : ℕ := 750

-- Definitions of initial exchange rates
def init_exchange_rate_eur_to_usd : ℝ := 1.1
def init_exchange_rate_gbp_to_usd : ℝ := 1.3

-- Definitions of profit percentages for each article
def profit_percent_eur : ℝ := 0.08
def profit_percent_gbp : ℝ := 0.1
def profit_percent_usd : ℝ := 0.15

-- Definitions of new exchange rates at the time of selling
def new_exchange_rate_eur_to_usd : ℝ := 1.15
def new_exchange_rate_gbp_to_usd : ℝ := 1.25

-- Calculation of purchase prices in USD
def purchase_price_in_usd₁ : ℝ := init_price_eur * init_exchange_rate_eur_to_usd
def purchase_price_in_usd₂ : ℝ := init_price_gbp * init_exchange_rate_gbp_to_usd
def purchase_price_in_usd₃ : ℝ := init_price_usd

-- Calculation of selling prices including profit in USD
def selling_price_in_usd₁ : ℝ := (init_price_eur + (init_price_eur * profit_percent_eur)) * new_exchange_rate_eur_to_usd
def selling_price_in_usd₂ : ℝ := (init_price_gbp + (init_price_gbp * profit_percent_gbp)) * new_exchange_rate_gbp_to_usd
def selling_price_in_usd₃ : ℝ := init_price_usd * (1 + profit_percent_usd)

-- Total selling price in USD
def total_selling_price_in_usd : ℝ :=
  selling_price_in_usd₁ + selling_price_in_usd₂ + selling_price_in_usd₃

-- Proof goal: total selling price should equal 2225.85 USD
theorem total_selling_price_correct :
  total_selling_price_in_usd = 2225.85 :=
by
  sorry

end total_selling_price_correct_l673_67366


namespace ages_correct_l673_67312

-- Definitions of the given conditions
def john_age : ℕ := 42
def tim_age : ℕ := 79
def james_age : ℕ := 30
def lisa_age : ℚ := 54.5
def kate_age : ℕ := 34
def michael_age : ℚ := 61.5
def anna_age : ℚ := 54.5

-- Mathematically equivalent proof problem
theorem ages_correct :
  (james_age = 30) ∧
  (lisa_age = 54.5) ∧
  (kate_age = 34) ∧
  (michael_age = 61.5) ∧
  (anna_age = 54.5) :=
by {
  sorry  -- Proof to be filled in
}

end ages_correct_l673_67312


namespace radius_ratio_l673_67349

noncomputable def ratio_of_radii (V1 V2 : ℝ) (R : ℝ) : ℝ := 
  (V2 / V1)^(1/3) * R 

theorem radius_ratio (V1 V2 : ℝ) (π : ℝ) (R r : ℝ) :
  V1 = 450 * π → 
  V2 = 36 * π → 
  (4 / 3) * π * R^3 = V1 →
  (4 / 3) * π * r^3 = V2 →
  r / R = 1 / (12.5)^(1/3) :=
by {
  sorry
}

end radius_ratio_l673_67349


namespace nobel_prize_laureates_at_workshop_l673_67394

theorem nobel_prize_laureates_at_workshop :
  ∃ (T W W_and_N N_no_W X N : ℕ), 
    T = 50 ∧ 
    W = 31 ∧ 
    W_and_N = 16 ∧ 
    (N_no_W = X + 3) ∧ 
    (T - W = 19) ∧ 
    (N_no_W + X = 19) ∧ 
    (N = W_and_N + N_no_W) ∧ 
    N = 27 :=
by
  sorry

end nobel_prize_laureates_at_workshop_l673_67394


namespace cube_sum_decomposition_l673_67331

theorem cube_sum_decomposition : 
  (∃ (a b c d e : ℤ), (1000 * x^3 + 27) = (a * x + b) * (c * x^2 + d * x + e) ∧ (a + b + c + d + e = 92)) :=
by
  sorry

end cube_sum_decomposition_l673_67331


namespace system_of_equations_solution_l673_67348

theorem system_of_equations_solution (x y z : ℕ) :
  x + y + z = 6 ∧ xy + yz + zx = 11 ∧ xyz = 6 ↔
  (x, y, z) = (1, 2, 3) ∨ (x, y, z) = (1, 3, 2) ∨ 
  (x, y, z) = (2, 1, 3) ∨ (x, y, z) = (2, 3, 1) ∨ 
  (x, y, z) = (3, 1, 2) ∨ (x, y, z) = (3, 2, 1) := by
  sorry

end system_of_equations_solution_l673_67348


namespace evaluate_i_powers_sum_l673_67388

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l673_67388


namespace ellipse_equation_fixed_point_l673_67317

/-- Given an ellipse with equation x^2 / a^2 + y^2 / b^2 = 1 where a > b > 0 and eccentricity e = 1/2,
    prove that the equation of the ellipse is x^2 / 4 + y^2 / 3 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a^2 = b^2 + (a / 2)^2) :
  (∀ x y : ℝ, (x^2 / 4 + y^2 / 3 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :=
by sorry

/-- Given an ellipse with equation x^2 / 4 + y^2 / 3 = 1,
    if a line l: y = kx + m intersects the ellipse at two points A and B (which are not the left and right vertices),
    and a circle passing through the right vertex of the ellipse has AB as its diameter,
    prove that the line passes through a fixed point and find its coordinates -/
theorem fixed_point (k m : ℝ) :
  (∃ x y, (x = 2 / 7 ∧ y = 0)) :=
by sorry

end ellipse_equation_fixed_point_l673_67317


namespace unique_nonzero_b_l673_67396

variable (a b m n : ℝ)
variable (h_ne : m ≠ n)
variable (h_m_nonzero : m ≠ 0)
variable (h_n_nonzero : n ≠ 0)

theorem unique_nonzero_b (h : (a * m + b * n + m)^2 - (a * m + b * n + n)^2 = (m - n)^2) : 
  a = 0 ∧ b = -1 :=
sorry

end unique_nonzero_b_l673_67396


namespace range_of_a_l673_67371

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

-- The mathematical statement to be proven in Lean
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, ∃ m M : ℝ, m = (f a x) ∧ M = (f a y) ∧ (∀ z : ℝ, f a z ≥ m) ∧ (∀ z : ℝ, f a z ≤ M)) ↔ 
  (a < -3 ∨ a > 6) :=
sorry

end range_of_a_l673_67371


namespace john_total_skateboarded_miles_l673_67342

-- Definitions
def distance_skateboard_to_park := 16
def distance_walk := 8
def distance_bike := 6
def distance_skateboard_home := distance_skateboard_to_park

-- Statement to prove
theorem john_total_skateboarded_miles : 
  distance_skateboard_to_park + distance_skateboard_home = 32 := 
by
  sorry

end john_total_skateboarded_miles_l673_67342


namespace expression_evaluation_l673_67395

theorem expression_evaluation : 
  (2^10 * 3^3) / (6 * 2^5) = 144 :=
by 
  sorry

end expression_evaluation_l673_67395


namespace scientific_notation_of_30067_l673_67367

theorem scientific_notation_of_30067 : ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 30067 = a * 10^n := by
  use 3.0067
  use 4
  sorry

end scientific_notation_of_30067_l673_67367


namespace train_length_l673_67315

theorem train_length (v_kmh : ℝ) (p_len : ℝ) (t_sec : ℝ) (l_train : ℝ) 
  (h_v : v_kmh = 72) (h_p : p_len = 250) (h_t : t_sec = 26) :
  l_train = 270 :=
by
  sorry

end train_length_l673_67315


namespace percentage_increase_l673_67318

theorem percentage_increase (X Y Z : ℝ) (h1 : X = 1.25 * Y) (h2 : Z = 100) (h3 : X + Y + Z = 370) :
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l673_67318


namespace jackson_total_souvenirs_l673_67373

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end jackson_total_souvenirs_l673_67373


namespace carnival_days_l673_67346

theorem carnival_days (d : ℕ) (h : 50 * d + 3 * (50 * d) - 30 * d - 75 = 895) : d = 5 :=
by
  sorry

end carnival_days_l673_67346


namespace front_view_length_l673_67375

-- Define the conditions of the problem
variables (d_body : ℝ) (d_side : ℝ) (d_top : ℝ)
variables (d_front : ℝ)

-- The given conditions
def conditions :=
  d_body = 5 * Real.sqrt 2 ∧
  d_side = 5 ∧
  d_top = Real.sqrt 34

-- The theorem to be proved
theorem front_view_length : 
  conditions d_body d_side d_top →
  d_front = Real.sqrt 41 :=
sorry

end front_view_length_l673_67375


namespace volume_of_prism_in_cubic_feet_l673_67383

theorem volume_of_prism_in_cubic_feet:
  let length_yd := 1
  let width_yd := 2
  let height_yd := 3
  let yard_to_feet := 3
  let length_ft := length_yd * yard_to_feet
  let width_ft := width_yd * yard_to_feet
  let height_ft := height_yd * yard_to_feet
  let volume := length_ft * width_ft * height_ft
  volume = 162 := by
  sorry

end volume_of_prism_in_cubic_feet_l673_67383


namespace part_A_part_B_l673_67347

-- Definitions for the setup
variables (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0)

-- Part (A): Specific distance 5d
theorem part_A (d : ℝ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = 25 * d^2 ∧ |y - d| = 5 * d → 
  (x = 3 * d ∧ y = -4 * d) ∨ (x = -3 * d ∧ y = -4 * d)) :=
sorry

-- Part (B): General distance nd
theorem part_B (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d → ∃ x y, (x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d)) :=
sorry

end part_A_part_B_l673_67347


namespace product_ab_zero_l673_67380

theorem product_ab_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end product_ab_zero_l673_67380


namespace proctoring_arrangements_l673_67303

/-- Consider 4 teachers A, B, C, D each teaching their respective classes a, b, c, d.
    Each teacher must not proctor their own class.
    Prove that there are exactly 9 ways to arrange the proctoring as required. -/
theorem proctoring_arrangements : 
  ∃ (arrangements : Finset ((Fin 4) → (Fin 4))), 
    (∀ (f : (Fin 4) → (Fin 4)), f ∈ arrangements → ∀ i : Fin 4, f i ≠ i) 
    ∧ arrangements.card = 9 :=
sorry

end proctoring_arrangements_l673_67303


namespace number_one_seventh_equals_five_l673_67360

theorem number_one_seventh_equals_five (n : ℕ) (h : n / 7 = 5) : n = 35 :=
sorry

end number_one_seventh_equals_five_l673_67360


namespace inequality_of_positive_numbers_l673_67350

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_of_positive_numbers_l673_67350


namespace ellipse_hyperbola_foci_l673_67332

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l673_67332


namespace initial_number_of_men_l673_67316

theorem initial_number_of_men (M A : ℕ) 
  (h1 : ((M * A) - 22 + 42 = M * (A + 2))) : M = 10 :=
by
  sorry

end initial_number_of_men_l673_67316


namespace min_value_of_a_k_l673_67341

-- Define the conditions for our proof in Lean

-- a_n is a positive arithmetic sequence
def is_positive_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ m, a (m + 1) = a m + d

-- Given inequality condition for the sequence
def inequality_condition (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

-- Prove the minimum value of a_k
theorem min_value_of_a_k (a : ℕ → ℝ) (k : ℕ) (h_arith : is_positive_arithmetic_seq a) (h_ineq : inequality_condition a k) :
  a k = 9 / 2 :=
sorry

end min_value_of_a_k_l673_67341


namespace r_plus_s_value_l673_67392

theorem r_plus_s_value :
  (∃ (r s : ℝ) (line_intercepts : ∀ x y, y = -1/2 * x + 8 ∧ ((x = 16 ∧ y = 0) ∨ (x = 0 ∧ y = 8))), 
    s = -1/2 * r + 8 ∧ (16 * 8 / 2) = 2 * (16 * s / 2) ∧ r + s = 12) :=
sorry

end r_plus_s_value_l673_67392


namespace third_discount_is_five_percent_l673_67357

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end third_discount_is_five_percent_l673_67357


namespace weight_of_second_piece_l673_67356

-- Given conditions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

def weight (density : ℚ) (area : ℕ) : ℚ := density * area

-- Given dimensions and weight of the first piece
def length1 : ℕ := 4
def width1 : ℕ := 3
def area1 : ℕ := area length1 width1
def weight1 : ℚ := 18

-- Given dimensions of the second piece
def length2 : ℕ := 6
def width2 : ℕ := 4
def area2 : ℕ := area length2 width2

-- Uniform density implies a proportional relationship between area and weight
def density1 : ℚ := weight1 / area1

-- The main theorem to prove
theorem weight_of_second_piece :
  weight density1 area2 = 36 :=
by
  -- use sorry to skip the proof
  sorry

end weight_of_second_piece_l673_67356


namespace larry_expression_correct_l673_67389

theorem larry_expression_correct (a b c d e : ℤ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 2) (h₄ : d = 5) :
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 :=
by
  sorry

end larry_expression_correct_l673_67389


namespace computation_result_l673_67335

theorem computation_result :
  let a := -6
  let b := 25
  let c := -39
  let d := 40
  9 * a + 3 * b + 6 * c + d = -173 := by
  sorry

end computation_result_l673_67335


namespace total_cookies_prepared_l673_67324

-- Definition of conditions
def cookies_per_guest : ℕ := 19
def number_of_guests : ℕ := 2

-- Theorem statement
theorem total_cookies_prepared : (cookies_per_guest * number_of_guests) = 38 :=
by
  sorry

end total_cookies_prepared_l673_67324


namespace sqrt_expression_l673_67374

theorem sqrt_expression (x : ℝ) : 2 - x ≥ 0 ↔ x ≤ 2 := sorry

end sqrt_expression_l673_67374


namespace graph_single_point_l673_67361

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l673_67361


namespace set_union_example_l673_67307

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem set_union_example : M ∪ N = {1, 2, 3, 4} := by
  sorry

end set_union_example_l673_67307


namespace find_C_l673_67336

theorem find_C (A B C D : ℕ) (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_eq : 4000 + 100 * A + 50 + B + (1000 * C + 200 + 10 * D + 7) = 7070) : C = 2 :=
sorry

end find_C_l673_67336


namespace seashell_count_l673_67362

variable (initial_seashells additional_seashells total_seashells : ℕ)

theorem seashell_count (h1 : initial_seashells = 19) (h2 : additional_seashells = 6) : 
  total_seashells = initial_seashells + additional_seashells → total_seashells = 25 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end seashell_count_l673_67362


namespace solid_id_views_not_cylinder_l673_67319

theorem solid_id_views_not_cylinder :
  ∀ (solid : Type),
  (∃ (shape1 shape2 shape3 : solid),
    shape1 = shape2 ∧ shape2 = shape3) →
  solid ≠ cylinder :=
by 
  sorry

end solid_id_views_not_cylinder_l673_67319


namespace robbers_divide_and_choose_l673_67386

/-- A model of dividing loot between two robbers who do not trust each other -/
def divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) : Prop :=
  ∀ (B : ℕ → ℕ), B (max P1 P2) ≥ B P1 ∧ B (max P1 P2) ≥ B P2

theorem robbers_divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) :
  divide_and_choose P1 P2 A :=
sorry

end robbers_divide_and_choose_l673_67386


namespace rayden_has_more_birds_l673_67329

-- Definitions based on given conditions
def ducks_lily := 20
def geese_lily := 10
def chickens_lily := 5
def pigeons_lily := 30

def ducks_rayden := 3 * ducks_lily
def geese_rayden := 4 * geese_lily
def chickens_rayden := 5 * chickens_lily
def pigeons_rayden := pigeons_lily / 2

def more_ducks := ducks_rayden - ducks_lily
def more_geese := geese_rayden - geese_lily
def more_chickens := chickens_rayden - chickens_lily
def fewer_pigeons := pigeons_rayden - pigeons_lily

def total_more_birds := more_ducks + more_geese + more_chickens - fewer_pigeons

-- Statement to prove that Rayden has 75 more birds in total than Lily
theorem rayden_has_more_birds : total_more_birds = 75 := by
    sorry

end rayden_has_more_birds_l673_67329


namespace other_car_speed_l673_67337

-- Definitions of the conditions
def red_car_speed : ℕ := 30
def initial_gap : ℕ := 20
def overtaking_time : ℕ := 1

-- Assertion of what needs to be proved
theorem other_car_speed : (initial_gap + red_car_speed * overtaking_time) = 50 :=
  sorry

end other_car_speed_l673_67337


namespace area_of_triangle_ABC_l673_67351

theorem area_of_triangle_ABC (AB CD : ℝ) (height : ℝ) (h1 : CD = 3 * AB) (h2 : AB * height + CD * height = 48) :
  (1/2) * AB * height = 6 :=
by
  have trapezoid_area : AB * height + CD * height = 48 := h2
  have length_relation : CD = 3 * AB := h1
  have area_triangle_ABC := 6
  sorry

end area_of_triangle_ABC_l673_67351


namespace solve_for_n_l673_67354

theorem solve_for_n :
  ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 :=
by
  sorry

end solve_for_n_l673_67354


namespace solve_for_x_l673_67333

theorem solve_for_x (x : ℚ) :
  (4 * x - 12) / 3 = (3 * x + 6) / 5 → 
  x = 78 / 11 :=
sorry

end solve_for_x_l673_67333


namespace perfect_square_factors_450_l673_67358

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l673_67358


namespace greatest_number_divisible_by_11_and_3_l673_67369

namespace GreatestNumberDivisibility

theorem greatest_number_divisible_by_11_and_3 : 
  ∃ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (2 * A - 2 * B + C) % 11 = 0 ∧ 
    (2 * A + 2 * C + B) % 3 = 0 ∧
    (10000 * A + 1000 * C + 100 * C + 10 * B + A) = 95695 :=
by
  -- The proof here is omitted.
  sorry

end GreatestNumberDivisibility

end greatest_number_divisible_by_11_and_3_l673_67369


namespace quadratic_has_real_root_l673_67308

theorem quadratic_has_real_root (a b : ℝ) : 
  (¬(∀ x : ℝ, x^2 + a * x + b ≠ 0)) → (∃ x : ℝ, x^2 + a * x + b = 0) := 
by
  intro h
  sorry

end quadratic_has_real_root_l673_67308


namespace rhombus_diagonal_l673_67352

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 20) (h2 : area = 170) :
  (area = (d1 * d2) / 2) → d2 = 17 :=
by
  sorry

end rhombus_diagonal_l673_67352


namespace average_of_eight_twelve_and_N_is_12_l673_67323

theorem average_of_eight_twelve_and_N_is_12 (N : ℝ) (hN : 11 < N ∧ N < 19) : (8 + 12 + N) / 3 = 12 :=
by
  -- Place the complete proof step here
  sorry

end average_of_eight_twelve_and_N_is_12_l673_67323


namespace fraction_green_after_tripling_l673_67372

theorem fraction_green_after_tripling 
  (x : ℕ)
  (h₁ : ∃ x, 0 < x) -- Total number of marbles is a positive integer
  (h₂ : ∀ g y, g + y = x ∧ g = 1/4 * x ∧ y = 3/4 * x) -- Initial distribution
  (h₃ : ∀ y : ℕ, g' = 3 * g ∧ y' = y) -- Triple the green marbles, yellow stays the same
  : (g' / (g' + y')) = 1/2 := 
sorry

end fraction_green_after_tripling_l673_67372


namespace complex_number_imaginary_axis_l673_67322

theorem complex_number_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) → (a = 0 ∨ a = 2) :=
by
  sorry

end complex_number_imaginary_axis_l673_67322


namespace remainder_when_divided_by_17_l673_67359

theorem remainder_when_divided_by_17 (N : ℤ) (k : ℤ) 
  (h : N = 221 * k + 43) : N % 17 = 9 := 
by
  sorry

end remainder_when_divided_by_17_l673_67359


namespace hypotenuse_of_right_triangle_l673_67309

theorem hypotenuse_of_right_triangle (a b : ℕ) (h : ℕ)
  (h1 : a = 15) (h2 : b = 36) (right_triangle : a^2 + b^2 = h^2) : h = 39 :=
by
  sorry

end hypotenuse_of_right_triangle_l673_67309


namespace question_implies_answer_l673_67302

theorem question_implies_answer (x y : ℝ) (h : y^2 - x^2 < x) :
  (x ≥ 0 ∨ x ≤ -1) ∧ (-Real.sqrt (x^2 + x) < y ∧ y < Real.sqrt (x^2 + x)) :=
sorry

end question_implies_answer_l673_67302


namespace sin_double_angle_l673_67325

theorem sin_double_angle (x : ℝ) (h : Real.sin (x - π / 4) = 3 / 5) : Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end sin_double_angle_l673_67325


namespace paper_cranes_l673_67313

theorem paper_cranes (B C A : ℕ) (h1 : A + B + C = 1000)
  (h2 : A = 3 * B - 100)
  (h3 : C = A - 67) : A = 443 := by
  sorry

end paper_cranes_l673_67313


namespace grains_on_11th_more_than_1_to_9_l673_67339

theorem grains_on_11th_more_than_1_to_9 : 
  let grains_on_square (k : ℕ) := 3 ^ k
  let sum_first_n_squares (n : ℕ) := (3 * (3 ^ n - 1) / (3 - 1))
  grains_on_square 11 - sum_first_n_squares 9 = 147624 :=
by
  sorry

end grains_on_11th_more_than_1_to_9_l673_67339


namespace solve_fraction_eq_l673_67344

theorem solve_fraction_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : (1 / (x - 1) = 3 / (x - 3)) ↔ x = 0 :=
by {
  sorry
}

end solve_fraction_eq_l673_67344


namespace multiple_of_sandy_age_l673_67376

theorem multiple_of_sandy_age
    (k_age : ℕ)
    (e : ℕ) 
    (s_current_age : ℕ) 
    (h1: k_age = 10) 
    (h2: e = 340) 
    (h3: s_current_age + 2 = 3 * (k_age + 2)) :
  e / s_current_age = 10 :=
by
  sorry

end multiple_of_sandy_age_l673_67376


namespace grape_juice_percentage_after_addition_l673_67304

def initial_mixture_volume : ℝ := 40
def initial_grape_juice_percentage : ℝ := 0.10
def added_grape_juice_volume : ℝ := 10

theorem grape_juice_percentage_after_addition :
  ((initial_mixture_volume * initial_grape_juice_percentage + added_grape_juice_volume) /
  (initial_mixture_volume + added_grape_juice_volume)) * 100 = 28 :=
by 
  sorry

end grape_juice_percentage_after_addition_l673_67304


namespace sin_neg_60_eq_neg_sqrt_3_div_2_l673_67384

theorem sin_neg_60_eq_neg_sqrt_3_div_2 : 
  Real.sin (-π / 3) = - (Real.sqrt 3) / 2 := 
by
  sorry

end sin_neg_60_eq_neg_sqrt_3_div_2_l673_67384


namespace range_of_t_l673_67353

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 := 
by
  sorry

end range_of_t_l673_67353


namespace find_pots_l673_67327

def num_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def total_cost := 390

theorem find_pots (x y : ℕ) (h1 : x + y = num_pots) (h2 : cost_green_lily * x + cost_spider_plant * y = total_cost) :
  x = 38 ∧ y = 8 :=
by
  sorry

end find_pots_l673_67327


namespace max_distance_proof_l673_67364

def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline_gallons : ℝ := 21
def maximum_distance : ℝ := highway_mpg * gasoline_gallons

theorem max_distance_proof : maximum_distance = 256.2 := by
  sorry

end max_distance_proof_l673_67364


namespace required_run_rate_equivalence_l673_67363

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.5
def overs_first_phase : ℝ := 10
def total_target_runs : ℝ := 350
def remaining_overs : ℝ := 35
def total_overs : ℝ := 45

-- Define the already scored runs
def runs_scored_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_phase

-- Define the required runs for the remaining overs
def runs_needed : ℝ := total_target_runs - runs_scored_first_10_overs

-- Theorem stating the required run rate in the remaining 35 overs
theorem required_run_rate_equivalence :
  runs_needed / remaining_overs = 9 :=
by
  sorry

end required_run_rate_equivalence_l673_67363


namespace solve_system_eqns_l673_67310

noncomputable def eq1 (x y z : ℚ) : Prop := x^2 + 2 * y * z = x
noncomputable def eq2 (x y z : ℚ) : Prop := y^2 + 2 * z * x = y
noncomputable def eq3 (x y z : ℚ) : Prop := z^2 + 2 * x * y = z

theorem solve_system_eqns (x y z : ℚ) :
  (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) ↔
  ((x, y, z) = (0, 0, 0) ∨
   (x, y, z) = (1/3, 1/3, 1/3) ∨
   (x, y, z) = (1, 0, 0) ∨
   (x, y, z) = (0, 1, 0) ∨
   (x, y, z) = (0, 0, 1) ∨
   (x, y, z) = (2/3, -1/3, -1/3) ∨
   (x, y, z) = (-1/3, 2/3, -1/3) ∨
   (x, y, z) = (-1/3, -1/3, 2/3)) :=
by sorry

end solve_system_eqns_l673_67310


namespace initial_water_percentage_l673_67334

theorem initial_water_percentage (W : ℕ) (V1 V2 V3 W3 : ℕ) (h1 : V1 = 10) (h2 : V2 = 15) (h3 : V3 = V1 + V2) (h4 : V3 = 25) (h5 : W3 = 2) (h6 : (W * V1) / 100 = (W3 * V3) / 100) : W = 5 :=
by
  sorry

end initial_water_percentage_l673_67334


namespace find_dividend_l673_67326

-- Define the given conditions
def quotient : ℝ := 0.0012000000000000001
def divisor : ℝ := 17

-- State the problem: Prove that the dividend is the product of the quotient and the divisor
theorem find_dividend (q : ℝ) (d : ℝ) (hq : q = 0.0012000000000000001) (hd : d = 17) : 
  q * d = 0.0204000000000000027 :=
sorry

end find_dividend_l673_67326


namespace smallest_integer_y_l673_67381

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end smallest_integer_y_l673_67381


namespace triangle_area_is_correct_l673_67328

noncomputable def triangle_area (a b c B : ℝ) : ℝ := 
  0.5 * a * c * Real.sin B

theorem triangle_area_is_correct :
  let a := Real.sqrt 2
  let c := Real.sqrt 2
  let b := Real.sqrt 6
  let B := 2 * Real.pi / 3 -- 120 degrees in radians
  triangle_area a b c B = Real.sqrt 3 / 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end triangle_area_is_correct_l673_67328


namespace number_of_technicians_l673_67330

theorem number_of_technicians
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (avg_salary_techs : ℝ)
  (avg_salary_rest : ℝ)
  (num_techs num_rest : ℕ)
  (h_total_workers : total_workers = 56)
  (h_avg_salary_all : avg_salary_all = 6750)
  (h_avg_salary_techs : avg_salary_techs = 12000)
  (h_avg_salary_rest : avg_salary_rest = 6000)
  (h_eq_workers : num_techs + num_rest = total_workers)
  (h_eq_salaries : (num_techs * avg_salary_techs + num_rest * avg_salary_rest) = total_workers * avg_salary_all) :
  num_techs = 7 := sorry

end number_of_technicians_l673_67330


namespace students_taking_art_l673_67338

def total_students : ℕ := 500
def students_taking_music : ℕ := 20
def students_taking_both : ℕ := 10
def students_taking_neither : ℕ := 470

theorem students_taking_art :
  ∃ (A : ℕ), A = 20 ∧ total_students = 
             (students_taking_music - students_taking_both) + (A - students_taking_both) + students_taking_both + students_taking_neither :=
by
  sorry

end students_taking_art_l673_67338


namespace factorial_ends_with_base_8_zeroes_l673_67301

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def highestPowerOfFactorInFactorial (n p : ℕ) : ℕ :=
  if p = 1 then n else
  Nat.div (n - 1) (p - 1)

theorem factorial_ends_with_base_8_zeroes (n : ℕ) : 
  highestPowerOfFactorInFactorial 15 8 = 3 := 
sorry

end factorial_ends_with_base_8_zeroes_l673_67301


namespace tony_combined_lift_weight_l673_67355

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l673_67355


namespace find_13x2_22xy_13y2_l673_67306

variable (x y : ℝ)

theorem find_13x2_22xy_13y2 
  (h1 : 3 * x + 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) 
: 13 * x^2 + 22 * x * y + 13 * y^2 = 184 := 
sorry

end find_13x2_22xy_13y2_l673_67306


namespace animals_on_stump_l673_67343

def possible_n_values (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 15

theorem animals_on_stump (n : ℕ) (h1 : n ≥ 3) (h2 : n ≤ 20)
  (h3 : 11 ≥ (n + 1) / 3) (h4 : 9 ≥ n - (n + 1) / 3) : possible_n_values n :=
by {
  sorry
}

end animals_on_stump_l673_67343


namespace arithmetic_sequence_common_difference_l673_67314

theorem arithmetic_sequence_common_difference :
  ∃ d : ℝ, (∀ (a_n : ℕ → ℝ), a_n 1 = 3 ∧ a_n 3 = 7 ∧ (∀ n, a_n n = 3 + (n - 1) * d)) → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l673_67314


namespace box_volume_l673_67311

theorem box_volume (l w h : ℝ)
  (A1 : l * w = 30)
  (A2 : w * h = 20)
  (A3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end box_volume_l673_67311


namespace total_order_cost_is_correct_l673_67391

noncomputable def totalOrderCost : ℝ :=
  let costGeography := 35 * 10.5
  let costEnglish := 35 * 7.5
  let costMath := 20 * 12.0
  let costScience := 30 * 9.5
  let costHistory := 25 * 11.25
  let costArt := 15 * 6.75
  let discount c := c * 0.10
  let netGeography := if 35 >= 30 then costGeography - discount costGeography else costGeography
  let netEnglish := if 35 >= 30 then costEnglish - discount costEnglish else costEnglish
  let netScience := if 30 >= 30 then costScience - discount costScience else costScience
  let netMath := costMath
  let netHistory := costHistory
  let netArt := costArt
  netGeography + netEnglish + netMath + netScience + netHistory + netArt

theorem total_order_cost_is_correct : totalOrderCost = 1446.00 := by
  sorry

end total_order_cost_is_correct_l673_67391


namespace possible_values_of_n_l673_67399

theorem possible_values_of_n (n : ℕ) (h1 : 0 < n)
  (h2 : 12 * n^3 = n^4 + 11 * n^2) :
  n = 1 ∨ n = 11 :=
sorry

end possible_values_of_n_l673_67399


namespace original_number_of_employees_l673_67300

theorem original_number_of_employees (x : ℕ) 
  (h1 : 0.77 * (x : ℝ) = 328) : x = 427 :=
sorry

end original_number_of_employees_l673_67300


namespace greatest_distance_between_centers_l673_67390

-- Define the conditions
noncomputable def circle_radius : ℝ := 4
noncomputable def rectangle_length : ℝ := 20
noncomputable def rectangle_width : ℝ := 16

-- Define the centers of the circles
noncomputable def circle_center1 : ℝ × ℝ := (4, circle_radius)
noncomputable def circle_center2 : ℝ × ℝ := (rectangle_length - 4, circle_radius)

-- Calculate the greatest possible distance
noncomputable def distance : ℝ := Real.sqrt ((8 ^ 2) + (rectangle_width ^ 2))

-- Statement to prove
theorem greatest_distance_between_centers :
  distance = 8 * Real.sqrt 5 :=
  sorry

end greatest_distance_between_centers_l673_67390


namespace total_money_spent_l673_67321

def total_cost (blades_cost : Nat) (string_cost : Nat) : Nat :=
  blades_cost + string_cost

theorem total_money_spent 
  (num_blades : Nat)
  (cost_per_blade : Nat)
  (string_cost : Nat)
  (h1 : num_blades = 4)
  (h2 : cost_per_blade = 8)
  (h3 : string_cost = 7) :
  total_cost (num_blades * cost_per_blade) string_cost = 39 :=
by
  sorry

end total_money_spent_l673_67321


namespace problem_y_eq_l673_67385

theorem problem_y_eq (y : ℝ) (h : y^3 - 3*y = 9) : y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
  sorry

end problem_y_eq_l673_67385


namespace sum_of_cubes_eq_96_over_7_l673_67393

-- Define the conditions from the problem
variables (a r : ℝ)
axiom condition_sum : a / (1 - r) = 2
axiom condition_sum_squares : a^2 / (1 - r^2) = 6

-- Define the correct answer that we expect to prove
theorem sum_of_cubes_eq_96_over_7 :
  a^3 / (1 - r^3) = 96 / 7 :=
sorry

end sum_of_cubes_eq_96_over_7_l673_67393


namespace ratio_closest_to_one_l673_67378

-- Define the entrance fee for adults and children.
def adult_fee : ℕ := 20
def child_fee : ℕ := 15

-- Define the total collected amount.
def total_collected : ℕ := 2400

-- Define the number of adults and children.
variables (a c : ℕ)

-- The main theorem to prove:
theorem ratio_closest_to_one 
  (h1 : a > 0) -- at least one adult
  (h2 : c > 0) -- at least one child
  (h3 : adult_fee * a + child_fee * c = total_collected) : 
  a / (c : ℚ) = 69 / 68 := 
sorry

end ratio_closest_to_one_l673_67378


namespace sum_odd_integers_correct_l673_67397

def sum_odd_integers_from_13_to_41 : ℕ := 
  let a := 13
  let l := 41
  let n := 15
  n * (a + l) / 2

theorem sum_odd_integers_correct : sum_odd_integers_from_13_to_41 = 405 :=
  by sorry

end sum_odd_integers_correct_l673_67397


namespace no_integer_roots_l673_67370

  theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 4 * x + 24 ≠ 0 :=
  by
    sorry
  
end no_integer_roots_l673_67370


namespace meeting_time_and_location_l673_67398

/-- Define the initial conditions -/
def start_time : ℕ := 8 -- 8:00 AM
def city_distance : ℕ := 12 -- 12 kilometers
def pedestrian_speed : ℚ := 6 -- 6 km/h
def cyclist_speed : ℚ := 18 -- 18 km/h

/-- Define the conditions for meeting time and location -/
theorem meeting_time_and_location :
  ∃ (meet_time : ℕ) (meet_distance : ℚ),
    meet_time = 9 * 60 + 15 ∧   -- 9:15 AM in minutes
    meet_distance = 4.5 :=      -- 4.5 kilometers
sorry

end meeting_time_and_location_l673_67398


namespace food_duration_l673_67345

theorem food_duration (mom_meals_per_day : ℕ) (mom_cups_per_meal : ℚ)
                      (puppy_count : ℕ) (puppy_meals_per_day : ℕ) (puppy_cups_per_meal : ℚ)
                      (total_food : ℚ)
                      (H_mom : mom_meals_per_day = 3) 
                      (H_mom_cups : mom_cups_per_meal = 3/2)
                      (H_puppies : puppy_count = 5) 
                      (H_puppy_meals : puppy_meals_per_day = 2) 
                      (H_puppy_cups : puppy_cups_per_meal = 1/2) 
                      (H_total_food : total_food = 57) : 
  (total_food / ((mom_meals_per_day * mom_cups_per_meal) + (puppy_count * puppy_meals_per_day * puppy_cups_per_meal))) = 6 := 
by
  sorry

end food_duration_l673_67345


namespace sum_of_digits_of_N_plus_2021_is_10_l673_67379

-- The condition that N is the smallest positive integer whose digits add to 41.
def smallest_integer_with_digit_sum_41 (N : ℕ) : Prop :=
  (N > 0) ∧ ((N.digits 10).sum = 41)

-- The Lean 4 statement to prove the problem.
theorem sum_of_digits_of_N_plus_2021_is_10 :
  ∃ N : ℕ, smallest_integer_with_digit_sum_41 N ∧ ((N + 2021).digits 10).sum = 10 :=
by
  -- The proof would go here
  sorry

end sum_of_digits_of_N_plus_2021_is_10_l673_67379


namespace part1_part2_l673_67320

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1 (m : ℝ) : (∃ x, deriv f x = 2 ∧ f x = 2 * x + m) → m = -Real.exp 1 :=
sorry

theorem part2 : ∀ x > 0, -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x) :=
sorry

end part1_part2_l673_67320


namespace value_of_expression_l673_67368

theorem value_of_expression : (4 * 3) + 2 = 14 := by
  sorry

end value_of_expression_l673_67368


namespace smallest_three_digit_divisible_by_4_and_5_l673_67365

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 4 = 0) ∧ (m % 5 = 0) → m ≥ n →
n = 100 :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l673_67365


namespace value_of_w_l673_67305

theorem value_of_w (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := 
sorry

end value_of_w_l673_67305
