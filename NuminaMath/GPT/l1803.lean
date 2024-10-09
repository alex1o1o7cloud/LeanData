import Mathlib

namespace book_cost_l1803_180316

theorem book_cost (n_5 n_3 : ℕ) (N : ℕ) :
  (N = n_5 + n_3) ∧ (N > 10) ∧ (N < 20) ∧ (5 * n_5 = 3 * n_3) →  5 * n_5 = 30 := 
sorry

end book_cost_l1803_180316


namespace speed_against_current_l1803_180342

theorem speed_against_current (V_curr : ℝ) (V_man : ℝ) (V_curr_val : V_curr = 3.2) (V_man_with_curr : V_man = 15) :
    V_man - V_curr = 8.6 := 
by 
  rw [V_curr_val, V_man_with_curr]
  norm_num
  sorry

end speed_against_current_l1803_180342


namespace zero_of_function_is_not_intersection_l1803_180345

noncomputable def is_function_zero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

theorem zero_of_function_is_not_intersection (f : ℝ → ℝ) :
  ¬ (∀ x : ℝ, is_function_zero f x ↔ (f x = 0 ∧ x ∈ {x | f x = 0})) :=
by
  sorry

end zero_of_function_is_not_intersection_l1803_180345


namespace remainder_of_product_mod_12_l1803_180335

-- Define the given constants
def a := 1125
def b := 1127
def c := 1129
def d := 12

-- State the conditions as Lean hypotheses
lemma mod_eq_1125 : a % d = 9 := by sorry
lemma mod_eq_1127 : b % d = 11 := by sorry
lemma mod_eq_1129 : c % d = 1 := by sorry

-- Define the theorem to prove
theorem remainder_of_product_mod_12 : (a * b * c) % d = 3 := by
  -- Use the conditions stated above to prove the theorem
  sorry

end remainder_of_product_mod_12_l1803_180335


namespace point_b_in_third_quadrant_l1803_180364

-- Definitions of the points with their coordinates
def PointA : ℝ × ℝ := (2, 3)
def PointB : ℝ × ℝ := (-1, -4)
def PointC : ℝ × ℝ := (-4, 1)
def PointD : ℝ × ℝ := (5, -3)

-- Definition of a point being in the third quadrant
def inThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- The main Theorem to prove that PointB is in the third quadrant
theorem point_b_in_third_quadrant : inThirdQuadrant PointB :=
by sorry

end point_b_in_third_quadrant_l1803_180364


namespace proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l1803_180306

variable (a b : ℝ)

-- Condition
def condition : Prop :=
  (1 / a) - (1 / b) = 1 / (a + b)

-- Proof statement
theorem proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab (h : condition a b) :
  (1 / a^2) - (1 / b^2) = 1 / (a * b) :=
sorry

end proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l1803_180306


namespace ratio_add_b_l1803_180328

theorem ratio_add_b (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 :=
by
  sorry

end ratio_add_b_l1803_180328


namespace number_of_lines_passing_through_four_points_l1803_180347

-- Defining the three-dimensional points and conditions
structure Point3D where
  x : ℕ
  y : ℕ
  z : ℕ
  h1 : 1 ≤ x ∧ x ≤ 5
  h2 : 1 ≤ y ∧ y ≤ 5
  h3 : 1 ≤ z ∧ z ≤ 5

-- Define a valid line passing through four distinct points (Readonly accessors for the conditions)
def valid_line (p1 p2 p3 p4 : Point3D) : Prop := 
  sorry -- Define conditions for points to be collinear and distinct

-- Main theorem statement
theorem number_of_lines_passing_through_four_points : 
  ∃ (lines : ℕ), lines = 150 :=
sorry

end number_of_lines_passing_through_four_points_l1803_180347


namespace geometric_series_sum_eq_l1803_180326

-- Given conditions
def a : ℚ := 1 / 2
def r : ℚ := 1 / 2
def n : ℕ := 5

-- Define the geometric series sum formula
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- The main theorem to prove
theorem geometric_series_sum_eq : geometric_series_sum a r n = 31 / 32 := by
  sorry

end geometric_series_sum_eq_l1803_180326


namespace travel_time_total_l1803_180325

theorem travel_time_total (dist1 dist2 dist3 speed1 speed2 speed3 : ℝ)
  (h_dist1 : dist1 = 50) (h_dist2 : dist2 = 100) (h_dist3 : dist3 = 150)
  (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_speed3 : speed3 = 120) :
  dist1 / speed1 + dist2 / speed2 + dist3 / speed3 = 3.5 :=
by
  sorry

end travel_time_total_l1803_180325


namespace union_M_N_l1803_180387

namespace MyMath

def M : Set ℝ := {x | x^2 = 1}
def N : Set ℝ := {1, 2}

theorem union_M_N : M ∪ N = {-1, 1, 2} := sorry

end MyMath

end union_M_N_l1803_180387


namespace determine_a_l1803_180379

theorem determine_a (a : ℝ) 
  (h1 : (a - 1) * (0:ℝ)^2 + 0 + a^2 - 1 = 0)
  (h2 : a - 1 ≠ 0) : 
  a = -1 := 
sorry

end determine_a_l1803_180379


namespace trapezoid_area_difference_l1803_180361

def trapezoid_area (base1 base2 height : ℝ) : ℝ :=
  0.5 * (base1 + base2) * height

def combined_area (base1 base2 height : ℝ) : ℝ :=
  2 * trapezoid_area base1 base2 height

theorem trapezoid_area_difference :
  let combined_area1 := combined_area 11 19 10
  let combined_area2 := combined_area 9.5 11 8
  combined_area1 - combined_area2 = 136 :=
by
  let combined_area1 := combined_area 11 19 10 
  let combined_area2 := combined_area 9.5 11 8 
  show combined_area1 - combined_area2 = 136
  sorry

end trapezoid_area_difference_l1803_180361


namespace greatest_2q_minus_r_l1803_180302

theorem greatest_2q_minus_r :
  ∃ (q r : ℕ), 1027 = 21 * q + r ∧ q > 0 ∧ r > 0 ∧ 2 * q - r = 77 :=
by
  sorry

end greatest_2q_minus_r_l1803_180302


namespace prime_addition_fraction_equivalence_l1803_180336

theorem prime_addition_fraction_equivalence : 
  ∃ n : ℕ, Prime n ∧ (4 + n) * 8 = (7 + n) * 7 ∧ n = 17 := 
sorry

end prime_addition_fraction_equivalence_l1803_180336


namespace markese_earnings_16_l1803_180318

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l1803_180318


namespace range_of_f_is_0_2_3_l1803_180301

def f (x : ℤ) : ℤ := x + 1
def S : Set ℤ := {-1, 1, 2}

theorem range_of_f_is_0_2_3 : Set.image f S = {0, 2, 3} := by
  sorry

end range_of_f_is_0_2_3_l1803_180301


namespace coin_toss_probability_l1803_180303

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  binomial_probability 3 2 0.5 = 0.375 :=
by
  sorry

end coin_toss_probability_l1803_180303


namespace people_in_room_proof_l1803_180357

-- Definitions corresponding to the problem conditions
def people_in_room (total_people : ℕ) : ℕ := total_people
def seated_people (total_people : ℕ) : ℕ := (3 * total_people / 5)
def total_chairs (total_people : ℕ) : ℕ := (3 * (5 * people_in_room total_people) / 2 / 5 + 8)
def empty_chairs : ℕ := 8
def occupied_chairs (total_people : ℕ) : ℕ := (2 * total_chairs total_people / 3)

-- Proving that there are 27 people in the room
theorem people_in_room_proof (total_chairs : ℕ) :
  (seated_people 27 = 2 * total_chairs / 3) ∧ 
  (8 = total_chairs - 2 * total_chairs / 3) → 
  people_in_room 27 = 27 :=
by
  sorry

end people_in_room_proof_l1803_180357


namespace inequality_solution_l1803_180308

noncomputable def solution_set (x : ℝ) : Prop := 
  (x < -1) ∨ (x > 3)

theorem inequality_solution :
  { x : ℝ | (3 - x) / (x + 1) < 0 } = { x : ℝ | solution_set x } :=
by
  sorry

end inequality_solution_l1803_180308


namespace slices_per_pizza_l1803_180367

def number_of_people : ℕ := 18
def slices_per_person : ℕ := 3
def number_of_pizzas : ℕ := 6
def total_slices : ℕ := number_of_people * slices_per_person

theorem slices_per_pizza : total_slices / number_of_pizzas = 9 :=
by
  -- proof steps would go here
  sorry

end slices_per_pizza_l1803_180367


namespace sufficient_but_not_necessary_condition_l1803_180353

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x >= 3) → (x^2 - 2*x - 3 >= 0) ∧ ¬((x^2 - 2*x - 3 >= 0) → (x >= 3)) := by
  sorry

end sufficient_but_not_necessary_condition_l1803_180353


namespace min_am_hm_l1803_180322

theorem min_am_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1/a + 1/b) ≥ 4 :=
by sorry

end min_am_hm_l1803_180322


namespace ryegrass_percentage_l1803_180304

theorem ryegrass_percentage (x_ryegrass_percent : ℝ) (y_ryegrass_percent : ℝ) (mixture_x_percent : ℝ)
  (hx : x_ryegrass_percent = 0.40)
  (hy : y_ryegrass_percent = 0.25)
  (hmx : mixture_x_percent = 0.8667) :
  (x_ryegrass_percent * mixture_x_percent + y_ryegrass_percent * (1 - mixture_x_percent)) * 100 = 38 :=
by
  sorry

end ryegrass_percentage_l1803_180304


namespace calculate_f2_f_l1803_180351

variable {f : ℝ → ℝ}

-- Definition of the conditions
def tangent_line_at_x2 (f : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ → ℝ), (∀ x, L x = -x + 1) ∧ (∀ x, f x = L x + (f x - L 2))

theorem calculate_f2_f'2 (h : tangent_line_at_x2 f) :
  f 2 + deriv f 2 = -2 :=
sorry

end calculate_f2_f_l1803_180351


namespace cone_sections_equal_surface_area_l1803_180368

theorem cone_sections_equal_surface_area {m r : ℝ} (h_r_pos : r > 0) (h_m_pos : m > 0) :
  ∃ (m1 m2 : ℝ), 
  (m1 = m / Real.sqrt 3) ∧ 
  (m2 = m / 3 * Real.sqrt 6) :=
sorry

end cone_sections_equal_surface_area_l1803_180368


namespace initial_customers_l1803_180349

theorem initial_customers (tables : ℕ) (people_per_table : ℕ) (customers_left : ℕ) (h1 : tables = 5) (h2 : people_per_table = 9) (h3 : customers_left = 17) :
  tables * people_per_table + customers_left = 62 :=
by
  sorry

end initial_customers_l1803_180349


namespace solve_system_eqns_l1803_180374

theorem solve_system_eqns (x y z : ℝ) (h1 : x^3 + y^3 + z^3 = 8)
  (h2 : x^2 + y^2 + z^2 = 22)
  (h3 : 1/x + 1/y + 1/z + z/(x * y) = 0) :
  (x = 3 ∧ y = 2 ∧ z = -3) ∨ (x = -3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = -3) ∨ (x = 2 ∧ y = -3 ∧ z = 3) :=
by
  sorry

end solve_system_eqns_l1803_180374


namespace spent_on_puzzle_l1803_180348

-- Defining all given conditions
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def final_amount : ℕ := 1

-- Define the total money before spending on the puzzle
def total_before_puzzle := initial_money + saved_money - spent_on_comic

-- Prove that the amount spent on the puzzle is $18
theorem spent_on_puzzle : (total_before_puzzle - final_amount) = 18 := 
by {
  sorry
}

end spent_on_puzzle_l1803_180348


namespace max_value_f_compare_magnitude_l1803_180323

open Real

def f (x : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- 1. Prove that the maximum value of f(x) is 2.
theorem max_value_f : ∃ x : ℝ, f x = 2 :=
sorry

-- 2. Given the condition, prove 2m + n > 2.
theorem compare_magnitude (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : (1 / m) + (1 / (2 * n)) = 2) : 
  2 * m + n > 2 :=
sorry

end max_value_f_compare_magnitude_l1803_180323


namespace Monroe_spiders_l1803_180378

theorem Monroe_spiders (S : ℕ) (h1 : 12 * 6 + S * 8 = 136) : S = 8 :=
by
  sorry

end Monroe_spiders_l1803_180378


namespace gain_in_meters_l1803_180317

noncomputable def cost_price : ℝ := sorry
noncomputable def selling_price : ℝ := 1.5 * cost_price
noncomputable def total_cost_price : ℝ := 30 * cost_price
noncomputable def total_selling_price : ℝ := 30 * selling_price
noncomputable def gain : ℝ := total_selling_price - total_cost_price

theorem gain_in_meters (S C : ℝ) (h_S : S = 1.5 * C) (h_gain : gain = 15 * C) :
  15 * C / S = 10 := by
  sorry

end gain_in_meters_l1803_180317


namespace find_tan_half_angle_l1803_180358

variable {α : Real} (h₁ : Real.sin α = -24 / 25) (h₂ : α ∈ Set.Ioo (π:ℝ) (3 * π / 2))

theorem find_tan_half_angle : Real.tan (α / 2) = -4 / 3 :=
sorry

end find_tan_half_angle_l1803_180358


namespace exists_constant_a_l1803_180309

theorem exists_constant_a (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : (m : ℝ) / n < Real.sqrt 7) :
  ∃ (a : ℝ), a > 1 ∧ (7 - (m^2 : ℝ) / (n^2 : ℝ) ≥ a / (n^2 : ℝ)) ∧ a = 3 :=
by
  sorry

end exists_constant_a_l1803_180309


namespace sequence_square_terms_l1803_180329

theorem sequence_square_terms (k : ℤ) (y : ℕ → ℤ) 
  (h1 : y 1 = 1)
  (h2 : y 2 = 1)
  (h3 : ∀ n ≥ 1, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) :
  (∀ n, ∃ m : ℤ, y n = m ^ 2) ↔ k = 3 :=
by sorry

end sequence_square_terms_l1803_180329


namespace equilateral_triangles_formed_l1803_180338

theorem equilateral_triangles_formed :
  ∀ k : ℤ, -8 ≤ k ∧ k ≤ 8 →
  (∃ triangles : ℕ, triangles = 426) :=
by sorry

end equilateral_triangles_formed_l1803_180338


namespace represent_1917_as_sum_diff_of_squares_l1803_180393

theorem represent_1917_as_sum_diff_of_squares : ∃ a b c : ℤ, 1917 = a^2 - b^2 + c^2 :=
by
  use 480, 478, 1
  sorry

end represent_1917_as_sum_diff_of_squares_l1803_180393


namespace find_B_l1803_180363

theorem find_B (A B : ℕ) (h1 : Prime A) (h2 : Prime B) (h3 : A > 0) (h4 : B > 0) 
  (h5 : 1 / A - 1 / B = 192 / (2005^2 - 2004^2)) : B = 211 :=
sorry

end find_B_l1803_180363


namespace binomial_16_4_l1803_180390

theorem binomial_16_4 : Nat.choose 16 4 = 1820 :=
  sorry

end binomial_16_4_l1803_180390


namespace brass_players_10_l1803_180366

theorem brass_players_10:
  ∀ (brass woodwind percussion : ℕ),
    brass + woodwind + percussion = 110 →
    percussion = 4 * woodwind →
    woodwind = 2 * brass →
    brass = 10 :=
by
  intros brass woodwind percussion h1 h2 h3
  sorry

end brass_players_10_l1803_180366


namespace balloons_division_correct_l1803_180310

def number_of_balloons_per_school (yellow blue more_black num_schools: ℕ) : ℕ :=
  let black := yellow + more_black
  let total := yellow + blue + black
  total / num_schools

theorem balloons_division_correct :
  number_of_balloons_per_school 3414 5238 1762 15 = 921 := 
by
  sorry

end balloons_division_correct_l1803_180310


namespace find_x_l1803_180333

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) (h₁ : log x 16 = log 4 256) : x = 2 := by
  sorry

end find_x_l1803_180333


namespace hare_wins_l1803_180394

def hare_wins_race : Prop :=
  let hare_speed := 10
  let hare_run_time := 30
  let hare_nap_time := 30
  let tortoise_speed := 4
  let tortoise_delay := 10
  let total_race_time := 60
  let hare_distance := hare_speed * hare_run_time
  let tortoise_total_time := total_race_time - tortoise_delay
  let tortoise_distance := tortoise_speed * tortoise_total_time
  hare_distance > tortoise_distance

theorem hare_wins : hare_wins_race := by
  -- Proof here
  sorry

end hare_wins_l1803_180394


namespace integer_solution_range_l1803_180355

theorem integer_solution_range {m : ℝ} : 
  (∀ x : ℤ, -1 ≤ x → x < m → (x = -1 ∨ x = 0)) ↔ (0 < m ∧ m ≤ 1) :=
by 
  sorry

end integer_solution_range_l1803_180355


namespace total_paintings_is_correct_l1803_180339

-- Definitions for Philip's schedule and starting number of paintings
def philip_paintings_monday_and_tuesday := 3
def philip_paintings_wednesday := 2
def philip_paintings_thursday_and_friday := 5
def philip_initial_paintings := 20

-- Definitions for Amelia's schedule and starting number of paintings
def amelia_paintings_every_day := 2
def amelia_initial_paintings := 45

-- Calculation of total paintings after 5 weeks
def philip_weekly_paintings := 
  (2 * philip_paintings_monday_and_tuesday) + 
  philip_paintings_wednesday + 
  (2 * philip_paintings_thursday_and_friday)

def amelia_weekly_paintings := 
  7 * amelia_paintings_every_day

def total_paintings_after_5_weeks := 5 * philip_weekly_paintings + philip_initial_paintings + 5 * amelia_weekly_paintings + amelia_initial_paintings

-- Proof statement
theorem total_paintings_is_correct :
  total_paintings_after_5_weeks = 225 :=
  by sorry

end total_paintings_is_correct_l1803_180339


namespace parabola_vertex_l1803_180389

theorem parabola_vertex {a b c : ℝ} (h₁ : ∃ b c, ∀ x, a * x^2 + b * x + c = a * (x + 3)^2) (h₂ : a * (2 + 3)^2 = -50) : a = -2 :=
by
  sorry

end parabola_vertex_l1803_180389


namespace no_positive_a_b_for_all_primes_l1803_180380

theorem no_positive_a_b_for_all_primes :
  ∀ (a b : ℕ), 0 < a → 0 < b → ∃ (p q : ℕ), p > 1000 ∧ q > 1000 ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ ¬Prime (a * p + b * q) :=
by
  sorry

end no_positive_a_b_for_all_primes_l1803_180380


namespace gcd_10293_29384_l1803_180314

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end gcd_10293_29384_l1803_180314


namespace cos_diff_trigonometric_identity_l1803_180344

-- Problem 1
theorem cos_diff :
  (Real.cos (25 * Real.pi / 180) * Real.cos (35 * Real.pi / 180) - 
   Real.cos (65 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)) = 
  1/2 :=
sorry

-- Problem 2
theorem trigonometric_identity (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 0) :
  (Real.cos (2 * θ) - Real.sin (2 * θ)) / (1 + (Real.cos θ)^2) = 5/6 :=
sorry

end cos_diff_trigonometric_identity_l1803_180344


namespace decimals_between_6_1_and_6_4_are_not_two_l1803_180332

-- Definitions from the conditions in a)
def is_between (x : ℝ) (a b : ℝ) : Prop := a < x ∧ x < b

-- The main theorem statement
theorem decimals_between_6_1_and_6_4_are_not_two :
  ∀ x, is_between x 6.1 6.4 → false :=
by
  sorry

end decimals_between_6_1_and_6_4_are_not_two_l1803_180332


namespace find_maximum_k_l1803_180341

theorem find_maximum_k {k : ℝ} 
  (h_eq : ∀ x, x^2 + k * x + 8 = 0)
  (h_roots_diff : ∀ x₁ x₂, x₁ - x₂ = 10) :
  k = 2 * Real.sqrt 33 := 
sorry

end find_maximum_k_l1803_180341


namespace lemonade_third_intermission_l1803_180399

theorem lemonade_third_intermission (a b c T : ℝ) (h1 : a = 0.25) (h2 : b = 0.42) (h3 : T = 0.92) (h4 : T = a + b + c) : c = 0.25 :=
by
  sorry

end lemonade_third_intermission_l1803_180399


namespace reduction_in_consumption_l1803_180340

def rate_last_month : ℝ := 16
def rate_current : ℝ := 20
def initial_consumption (X : ℝ) : ℝ := X

theorem reduction_in_consumption (X : ℝ) : initial_consumption X - (initial_consumption X * rate_last_month / rate_current) = initial_consumption X * 0.2 :=
by
  sorry

end reduction_in_consumption_l1803_180340


namespace problem_statement_l1803_180373

-- Define the roots of the quadratic as r and s
variables (r s : ℝ)

-- Given conditions
def root_condition (r s : ℝ) := (r + s = 2 * Real.sqrt 6) ∧ (r * s = 3)

theorem problem_statement (h : root_condition r s) : r^8 + s^8 = 93474 :=
sorry

end problem_statement_l1803_180373


namespace ratio_third_to_second_year_l1803_180359

-- Define the yearly production of the apple tree
def first_year_production : Nat := 40
def second_year_production : Nat := 2 * first_year_production + 8
def total_production_three_years : Nat := 194
def third_year_production : Nat := total_production_three_years - (first_year_production + second_year_production)

-- Define the ratio calculation
def ratio (a b : Nat) : (Nat × Nat) := 
  let gcd_ab := Nat.gcd a b 
  (a / gcd_ab, b / gcd_ab)

-- Prove the ratio of the third year's production to the second year's production
theorem ratio_third_to_second_year : 
  ratio third_year_production second_year_production = (3, 4) :=
  sorry

end ratio_third_to_second_year_l1803_180359


namespace lily_sees_leo_l1803_180315

theorem lily_sees_leo : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
  d₁ = 0.75 → 
  d₂ = 0.75 → 
  v₁ = 15 → 
  v₂ = 9 → 
  (d₁ + d₂) / (v₁ - v₂) * 60 = 15 :=
by 
  intros d₁ d₂ v₁ v₂ h₁ h₂ h₃ h₄
  -- skipping the proof with sorry
  sorry

end lily_sees_leo_l1803_180315


namespace sequence_infinite_integers_l1803_180392

theorem sequence_infinite_integers (x : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : x 1 = x1) 
  (h2 : x 2 = x2) 
  (h3 : ∀ n ≥ 3, x n = x (n - 2) * x (n - 1) / (2 * x (n - 2) - x (n - 1))) : 
  (∃ k : ℤ, x1 = k ∧ x2 = k) ↔ (∀ n, ∃ m : ℤ, x n = m) :=
sorry

end sequence_infinite_integers_l1803_180392


namespace identify_7_real_coins_l1803_180337

theorem identify_7_real_coins (coins : Fin 63 → ℝ) (fakes : Finset (Fin 63)) (h_fakes_count : fakes.card = 7) (real_weight fake_weight : ℝ)
  (h_weights : ∀ i, i ∉ fakes → coins i = real_weight) (h_fake_weights : ∀ i, i ∈ fakes → coins i = fake_weight) (h_lighter : fake_weight < real_weight) :
  ∃ real_coins : Finset (Fin 63), real_coins.card = 7 ∧ (∀ i, i ∈ real_coins → coins i = real_weight) :=
sorry

end identify_7_real_coins_l1803_180337


namespace cos_sequence_next_coeff_sum_eq_28_l1803_180397

theorem cos_sequence_next_coeff_sum_eq_28 (α : ℝ) :
  let u := 2 * Real.cos α
  2 * Real.cos (8 * α) = u ^ 8 - 8 * u ^ 6 + 20 * u ^ 4 - 16 * u ^ 2 + 2 → 
  8 + (-8) + 6 + 20 + 2 = 28 :=
by intros u; sorry

end cos_sequence_next_coeff_sum_eq_28_l1803_180397


namespace cos_double_angle_l1803_180362

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_double_angle_l1803_180362


namespace students_distribute_l1803_180320

theorem students_distribute (x y : ℕ) (h₁ : x + y = 4200)
        (h₂ : x * 108 / 100 + y * 111 / 100 = 4620) :
    x = 1400 ∧ y = 2800 :=
by
  sorry

end students_distribute_l1803_180320


namespace total_paths_A_to_D_l1803_180312

-- Given conditions
def paths_from_A_to_B := 2
def paths_from_B_to_C := 2
def paths_from_C_to_D := 2
def direct_path_A_to_C := 1
def direct_path_B_to_D := 1

-- Proof statement
theorem total_paths_A_to_D : 
  paths_from_A_to_B * paths_from_B_to_C * paths_from_C_to_D + 
  direct_path_A_to_C * paths_from_C_to_D + 
  paths_from_A_to_B * direct_path_B_to_D = 12 := 
  by
    sorry

end total_paths_A_to_D_l1803_180312


namespace rachel_milk_amount_l1803_180377

theorem rachel_milk_amount : 
  let don_milk := (3 : ℚ) / 7
  let rachel_fraction := 4 / 5
  let rachel_milk := rachel_fraction * don_milk
  rachel_milk = 12 / 35 :=
by sorry

end rachel_milk_amount_l1803_180377


namespace minimum_soldiers_to_add_l1803_180376

theorem minimum_soldiers_to_add 
  (N : ℕ)
  (h1 : N % 7 = 2)
  (h2 : N % 12 = 2) : 
  (84 - N % 84) = 82 := 
by 
  sorry

end minimum_soldiers_to_add_l1803_180376


namespace doctors_to_lawyers_ratio_l1803_180319

theorem doctors_to_lawyers_ratio
  (d l : ℕ)
  (h1 : (40 * d + 55 * l) / (d + l) = 45)
  (h2 : d + l = 20) :
  d / l = 2 :=
by sorry

end doctors_to_lawyers_ratio_l1803_180319


namespace circumscribed_circle_radius_l1803_180356

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (R : ℝ)
  (h1 : b = 6) (h2 : c = 2) (h3 : A = π / 3) :
  R = (2 * Real.sqrt 21) / 3 :=
by
  sorry

end circumscribed_circle_radius_l1803_180356


namespace vincent_books_l1803_180346

theorem vincent_books (x : ℕ) (h1 : 10 + 3 + x = 13 + x)
                      (h2 : 16 * (13 + x) = 224) : x = 1 :=
by sorry

end vincent_books_l1803_180346


namespace time_to_cross_pole_l1803_180360

-- Setting up the definitions
def speed_kmh : ℤ := 72
def length_m : ℤ := 180

-- Conversion function from km/hr to m/s
def convert_speed (v : ℤ) : ℚ :=
  v * (1000 : ℚ) / 3600

-- Given conditions in mathematics
def speed_ms : ℚ := convert_speed speed_kmh

-- Desired proposition
theorem time_to_cross_pole : 
  length_m / speed_ms = 9 := 
by
  -- Temporarily skipping the proof
  sorry

end time_to_cross_pole_l1803_180360


namespace tax_liability_difference_l1803_180383

theorem tax_liability_difference : 
  let annual_income := 150000
  let old_tax_rate := 0.45
  let new_tax_rate_1 := 0.30
  let new_tax_rate_2 := 0.35
  let new_tax_rate_3 := 0.40
  let mortgage_interest := 10000
  let old_tax_liability := annual_income * old_tax_rate
  let taxable_income_new := annual_income - mortgage_interest
  let new_tax_liability := 
    if taxable_income_new <= 50000 then 
      taxable_income_new * new_tax_rate_1
    else if taxable_income_new <= 100000 then 
      50000 * new_tax_rate_1 + (taxable_income_new - 50000) * new_tax_rate_2
    else 
      50000 * new_tax_rate_1 + 50000 * new_tax_rate_2 + (taxable_income_new - 100000) * new_tax_rate_3
  let tax_liability_difference := old_tax_liability - new_tax_liability
  tax_liability_difference = 19000 := 
by
  sorry

end tax_liability_difference_l1803_180383


namespace quadratic_inequality_has_real_solutions_l1803_180386

theorem quadratic_inequality_has_real_solutions (c : ℝ) (h : 0 < c) : 
  (∃ x : ℝ, x^2 - 6 * x + c < 0) ↔ (0 < c ∧ c < 9) :=
sorry

end quadratic_inequality_has_real_solutions_l1803_180386


namespace simple_interest_rate_l1803_180382

theorem simple_interest_rate
  (A5 A8 : ℝ) (years_between : ℝ := 3) (I3 : ℝ) (annual_interest : ℝ)
  (P : ℝ) (R : ℝ)
  (h1 : A5 = 9800) -- Amount after 5 years is Rs. 9800
  (h2 : A8 = 12005) -- Amount after 8 years is Rs. 12005
  (h3 : I3 = A8 - A5) -- Interest for 3 years
  (h4 : annual_interest = I3 / years_between) -- Annual interest
  (h5 : P = 9800) -- Principal amount after 5 years
  (h6 : R = (annual_interest * 100) / P) -- Rate of interest formula revised
  : R = 7.5 := 
sorry

end simple_interest_rate_l1803_180382


namespace polygon_has_five_sides_l1803_180369

theorem polygon_has_five_sides (angle : ℝ) (h : angle = 108) :
  (∃ n : ℕ, n > 2 ∧ (180 - angle) * n = 360) ↔ n = 5 := 
by
  sorry

end polygon_has_five_sides_l1803_180369


namespace mean_of_all_students_l1803_180343

theorem mean_of_all_students (M A : ℕ) (m a : ℕ) (hM : M = 88) (hA : A = 68) (hRatio : m * 5 = 2 * a) : 
  (176 * a + 340 * a) / (7 * a) = 74 :=
by sorry

end mean_of_all_students_l1803_180343


namespace laptop_repair_cost_l1803_180354

theorem laptop_repair_cost
  (price_phone_repair : ℝ)
  (price_computer_repair : ℝ)
  (price_laptop_repair : ℝ)
  (condition1 : price_phone_repair = 11)
  (condition2 : price_computer_repair = 18)
  (condition3 : 5 * price_phone_repair + 2 * price_laptop_repair + 2 * price_computer_repair = 121) :
  price_laptop_repair = 15 :=
by
  sorry

end laptop_repair_cost_l1803_180354


namespace Expected_and_Variance_l1803_180372

variables (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

def P (xi : ℕ) : ℝ := 
  if xi = 0 then p else if xi = 1 then 1 - p else 0

def E_xi : ℝ := 0 * P p 0 + 1 * P p 1

def D_xi : ℝ := (0 - E_xi p)^2 * P p 0 + (1 - E_xi p)^2 * P p 1

theorem Expected_and_Variance :
  (E_xi p = 1 - p) ∧ (D_xi p = p * (1 - p)) :=
sorry

end Expected_and_Variance_l1803_180372


namespace sum_of_factors_l1803_180385

theorem sum_of_factors (W F c : ℕ) (hW_gt_20: W > 20) (hF_gt_20: F > 20) (product_eq : W * F = 770) (sum_eq : W + F = c) :
  c = 57 :=
by sorry

end sum_of_factors_l1803_180385


namespace inequality_proof_l1803_180321

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)) + (bc / (b + c)) + (ca / (c + a)) ≤ (3 * (ab + bc + ca)) / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l1803_180321


namespace evaluate_expression_l1803_180313

theorem evaluate_expression : (5^2 - 4^2)^3 = 729 :=
by
  sorry

end evaluate_expression_l1803_180313


namespace Razorback_tshirt_problem_l1803_180381

theorem Razorback_tshirt_problem
  (A T : ℕ)
  (h1 : A + T = 186)
  (h2 : 78 * T = 1092) :
  A = 172 := by
  sorry

end Razorback_tshirt_problem_l1803_180381


namespace shortest_path_from_vertex_to_center_of_non_adjacent_face_l1803_180350

noncomputable def shortest_path_on_cube (edge_length : ℝ) : ℝ :=
  edge_length + (edge_length * Real.sqrt 2 / 2)

theorem shortest_path_from_vertex_to_center_of_non_adjacent_face :
  shortest_path_on_cube 1 = 1 + Real.sqrt 2 / 2 :=
by
  sorry

end shortest_path_from_vertex_to_center_of_non_adjacent_face_l1803_180350


namespace perfect_squares_of_k_l1803_180375

theorem perfect_squares_of_k (k : ℕ) (h : ∃ (a : ℕ), k * (k + 1) = 3 * a^2) : 
  ∃ (m n : ℕ), k = 3 * m^2 ∧ k + 1 = n^2 := 
sorry

end perfect_squares_of_k_l1803_180375


namespace relationship_a_b_c_l1803_180324

noncomputable def distinct_positive_numbers (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem relationship_a_b_c (a b c : ℝ) (h1 : distinct_positive_numbers a b c) (h2 : a^2 + c^2 = 2 * b * c) : b > a ∧ a > c :=
by
  sorry

end relationship_a_b_c_l1803_180324


namespace intersection_of_asymptotes_l1803_180327

noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

theorem intersection_of_asymptotes : f 3 = 1 :=
by sorry

end intersection_of_asymptotes_l1803_180327


namespace quadratic_m_leq_9_l1803_180365

-- Define the quadratic equation
def quadratic_eq_has_real_roots (a b c : ℝ) : Prop := 
  b^2 - 4*a*c ≥ 0

-- Define the specific property we need to prove
theorem quadratic_m_leq_9 (m : ℝ) : (quadratic_eq_has_real_roots 1 (-6) m) → (m ≤ 9) := 
by
  sorry

end quadratic_m_leq_9_l1803_180365


namespace product_of_three_consecutive_natural_numbers_divisible_by_six_l1803_180307

theorem product_of_three_consecutive_natural_numbers_divisible_by_six (n : ℕ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
by
  sorry

end product_of_three_consecutive_natural_numbers_divisible_by_six_l1803_180307


namespace find_y_when_x_is_twelve_l1803_180391

variables (x y k : ℝ)

theorem find_y_when_x_is_twelve
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = 12) :
  y = 56.25 :=
sorry

end find_y_when_x_is_twelve_l1803_180391


namespace find_f_2015_l1803_180396

noncomputable def f : ℝ → ℝ :=
sorry

lemma f_period : ∀ x : ℝ, f (x + 8) = f x :=
sorry

axiom f_func_eq : ∀ x : ℝ, f (x + 2) = (1 + f x) / (1 - f x)

axiom f_initial : f 1 = 1 / 4

theorem find_f_2015 : f 2015 = -3 / 5 :=
sorry

end find_f_2015_l1803_180396


namespace quadratic_inequality_solution_l1803_180300

theorem quadratic_inequality_solution :
  { m : ℝ // ∀ x : ℝ, m * x^2 - 6 * m * x + 5 * m + 1 > 0 } = { m : ℝ // 0 ≤ m ∧ m < 1/4 } :=
sorry

end quadratic_inequality_solution_l1803_180300


namespace b_41_mod_49_l1803_180352

noncomputable def b (n : ℕ) : ℕ :=
  6 ^ n + 8 ^ n

theorem b_41_mod_49 : b 41 % 49 = 35 := by
  sorry

end b_41_mod_49_l1803_180352


namespace problem_f3_is_neg2_l1803_180398

theorem problem_f3_is_neg2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x, f (1 + x) = -f (1 - x)) (h3 : f 1 = 2) : f 3 = -2 :=
sorry

end problem_f3_is_neg2_l1803_180398


namespace Matt_buys_10_key_chains_l1803_180330

theorem Matt_buys_10_key_chains
  (cost_per_keychain_in_pack_of_10 : ℝ)
  (cost_per_keychain_in_pack_of_4 : ℝ)
  (number_of_keychains : ℝ)
  (savings : ℝ)
  (h1 : cost_per_keychain_in_pack_of_10 = 2)
  (h2 : cost_per_keychain_in_pack_of_4 = 3)
  (h3 : savings = 20)
  (h4 : 3 * number_of_keychains - 2 * number_of_keychains = savings) :
  number_of_keychains = 10 := 
by
  sorry

end Matt_buys_10_key_chains_l1803_180330


namespace probability_tile_from_ANGLE_l1803_180395

def letters_in_ALGEBRA : List Char := ['A', 'L', 'G', 'E', 'B', 'R', 'A']
def letters_in_ANGLE : List Char := ['A', 'N', 'G', 'L', 'E']
def count_matching_letters (letters: List Char) (target: List Char) : Nat :=
  letters.foldr (fun l acc => if l ∈ target then acc + 1 else acc) 0

theorem probability_tile_from_ANGLE :
  (count_matching_letters letters_in_ALGEBRA letters_in_ANGLE : ℚ) / (letters_in_ALGEBRA.length : ℚ) = 5 / 7 :=
by
  sorry

end probability_tile_from_ANGLE_l1803_180395


namespace tangent_line_circle_l1803_180334

theorem tangent_line_circle {m : ℝ} (tangent : ∀ x y : ℝ, x + y + m = 0 → x^2 + y^2 = m → false) : m = 2 :=
sorry

end tangent_line_circle_l1803_180334


namespace sams_weight_l1803_180384

  theorem sams_weight (j s : ℝ) (h1 : j + s = 240) (h2 : s - j = j / 3) : s = 2880 / 21 :=
  by
    sorry
  
end sams_weight_l1803_180384


namespace largest_number_among_given_l1803_180305

theorem largest_number_among_given (
  A B C D E : ℝ
) (hA : A = 0.936)
  (hB : B = 0.9358)
  (hC : C = 0.9361)
  (hD : D = 0.935)
  (hE : E = 0.921):
  C = max A (max B (max C (max D E))) :=
by
  sorry

end largest_number_among_given_l1803_180305


namespace wrapping_paper_per_present_l1803_180331

theorem wrapping_paper_per_present :
  let sum_paper := 1 / 2
  let num_presents := 5
  (sum_paper / num_presents) = 1 / 10 := by
  sorry

end wrapping_paper_per_present_l1803_180331


namespace Minjeong_family_juice_consumption_l1803_180388

theorem Minjeong_family_juice_consumption :
  (∀ (amount_per_time : ℝ) (times_per_day : ℕ) (days_per_week : ℕ),
  amount_per_time = 0.2 → times_per_day = 3 → days_per_week = 7 → 
  amount_per_time * times_per_day * days_per_week = 4.2) :=
by
  intros amount_per_time times_per_day days_per_week h1 h2 h3
  rw [h1, h2, h3]
  norm_num

end Minjeong_family_juice_consumption_l1803_180388


namespace correct_option_a_l1803_180311

theorem correct_option_a (x y a b : ℝ) : 3 * x - 2 * x = x :=
by sorry

end correct_option_a_l1803_180311


namespace integer_root_sum_abs_l1803_180370

theorem integer_root_sum_abs :
  ∃ a b c m : ℤ, 
    (a + b + c = 0 ∧ ab + bc + ca = -2023 ∧ |a| + |b| + |c| = 94) := sorry

end integer_root_sum_abs_l1803_180370


namespace cos_sum_seventh_root_of_unity_l1803_180371

theorem cos_sum_seventh_root_of_unity (z : ℂ) (α : ℝ) 
  (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : ∃ k : ℤ, α = (2 * k * π) / 7 ) :
  (Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)) = -1 / 2 :=
by 
  sorry

end cos_sum_seventh_root_of_unity_l1803_180371
