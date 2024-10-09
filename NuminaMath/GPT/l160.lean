import Mathlib

namespace eunji_initial_money_l160_16076

-- Define the conditions
def snack_cost : ℕ := 350
def allowance : ℕ := 800
def money_left_after_pencil : ℕ := 550

-- Define what needs to be proven
theorem eunji_initial_money (initial_money : ℕ) :
  initial_money - snack_cost + allowance = money_left_after_pencil * 2 →
  initial_money = 650 :=
by
  sorry

end eunji_initial_money_l160_16076


namespace solve_equation_l160_16053

open Real

theorem solve_equation (t : ℝ) :
  ¬cos t = 0 ∧ ¬cos (2 * t) = 0 → 
  (tan (2 * t) / (cos t)^2 - tan t / (cos (2 * t))^2 = 0 ↔ 
    (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = π * ↑n + π / 6) ∨ (∃ n : ℤ, t = π * ↑n - π / 6)) :=
by
  intros h
  sorry

end solve_equation_l160_16053


namespace apple_street_length_l160_16046

theorem apple_street_length :
  ∀ (n : ℕ) (d : ℕ), 
    (n = 15) → (d = 200) → 
    (∃ l : ℝ, (l = ((n + 1) * d) / 1000) ∧ l = 3.2) :=
by
  intros
  sorry

end apple_street_length_l160_16046


namespace find_x_l160_16005

theorem find_x (x y z w : ℕ) (h1 : x = y + 8) (h2 : y = z + 15) (h3 : z = w + 25) (h4 : w = 90) : x = 138 :=
by
  sorry

end find_x_l160_16005


namespace max_students_seated_l160_16035

-- Define the number of seats in the i-th row
def seats_in_row (i : ℕ) : ℕ := 10 + 2 * i

-- Define the maximum number of students that can be seated in the i-th row
def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

-- Sum the maximum number of students for all 25 rows
def total_max_students : ℕ := (Finset.range 25).sum max_students_in_row

-- The theorem statement
theorem max_students_seated : total_max_students = 450 := by
  sorry

end max_students_seated_l160_16035


namespace simplest_quadratic_radicals_l160_16079

theorem simplest_quadratic_radicals (a : ℝ) :
  (3 * a - 8 ≥ 0) ∧ (17 - 2 * a ≥ 0) → a = 5 :=
by
  intro h
  sorry

end simplest_quadratic_radicals_l160_16079


namespace linda_savings_l160_16081

theorem linda_savings :
  ∀ (S : ℝ), (5 / 6 * S + 500 = S) → S = 3000 :=
by
  intros S h
  sorry

end linda_savings_l160_16081


namespace pushkin_family_pension_l160_16032

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end pushkin_family_pension_l160_16032


namespace final_temperature_correct_l160_16057

-- Define the initial conditions
def initial_temperature : ℝ := 12
def decrease_per_hour : ℝ := 5
def time_duration : ℕ := 4

-- Define the expected final temperature
def expected_final_temperature : ℝ := -8

-- The theorem to prove that the final temperature after a given time is as expected
theorem final_temperature_correct :
  initial_temperature + (-decrease_per_hour * time_duration) = expected_final_temperature :=
by
  sorry

end final_temperature_correct_l160_16057


namespace maximize_S_l160_16001

noncomputable def a (n: ℕ) : ℝ := 24 - 2 * n

noncomputable def S (n: ℕ) : ℝ := -n^2 + 23 * n

theorem maximize_S (n : ℕ) : 
  (n = 11 ∨ n = 12) → ∀ m : ℕ, m ≠ 11 ∧ m ≠ 12 → S m ≤ S n :=
sorry

end maximize_S_l160_16001


namespace num_students_in_second_class_l160_16071

theorem num_students_in_second_class 
  (avg1 : ℕ) (num1 : ℕ) (avg2 : ℕ) (overall_avg : ℕ) (n : ℕ) :
  avg1 = 50 → num1 = 30 → avg2 = 60 → overall_avg = 5625 → 
  (num1 * avg1 + n * avg2) = (num1 + n) * overall_avg → n = 50 :=
by sorry

end num_students_in_second_class_l160_16071


namespace least_k_square_divisible_by_240_l160_16099

theorem least_k_square_divisible_by_240 (k : ℕ) (h : ∃ m : ℕ, k ^ 2 = 240 * m) : k ≥ 60 :=
by
  sorry

end least_k_square_divisible_by_240_l160_16099


namespace solution_of_equations_l160_16056

variables (x y z w : ℤ)

def system_of_equations :=
  x + y + z + w = 20 ∧
  y + 2 * z - 3 * w = 28 ∧
  x - 2 * y + z = 36 ∧
  -7 * x - y + 5 * z + 3 * w = 84

theorem solution_of_equations (x y z w : ℤ) :
  system_of_equations x y z w → (x, y, z, w) = (4, -6, 20, 2) :=
by sorry

end solution_of_equations_l160_16056


namespace number_of_yellow_balls_l160_16062

theorem number_of_yellow_balls (x : ℕ) (h : (6 : ℝ) / (6 + x) = 0.3) : x = 14 :=
by
  sorry

end number_of_yellow_balls_l160_16062


namespace min_AB_CD_value_l160_16041

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def AB_CD (AC BD CB : vector) : ℝ :=
  let AB := (CB.1 + AC.1, CB.2 + AC.2)
  let CD := (CB.1 + BD.1, CB.2 + BD.2)
  dot_product AB CD

theorem min_AB_CD_value : ∀ (AC BD : vector), AC = (1, 2) → BD = (-2, 2) → 
  ∃ CB : vector, AB_CD AC BD CB = -9 / 4 :=
by
  intros AC BD hAC hBD
  sorry

end min_AB_CD_value_l160_16041


namespace infinite_series_computation_l160_16008

theorem infinite_series_computation : 
  ∑' k : ℕ, (8^k) / ((2^k - 1) * (2^(k + 1) - 1)) = 4 :=
by
  sorry

end infinite_series_computation_l160_16008


namespace divide_plane_into_regions_l160_16092

theorem divide_plane_into_regions (n : ℕ) (h₁ : n < 199) (h₂ : ∃ (k : ℕ), k = 99):
  n = 100 ∨ n = 198 :=
sorry

end divide_plane_into_regions_l160_16092


namespace solve_quadratic_l160_16038

noncomputable def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic : ∀ x : ℝ, quadratic_roots 1 (-4) (-5) x ↔ (x = -1 ∨ x = 5) :=
by
  intro x
  rw [quadratic_roots]
  sorry

end solve_quadratic_l160_16038


namespace find_x_that_satisfies_f_l160_16036

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem find_x_that_satisfies_f (α : ℝ) (x : ℝ) (h : power_function α (-2) = -1/8) : 
  power_function α x = 27 → x = 1/3 :=
  by
  sorry

end find_x_that_satisfies_f_l160_16036


namespace problem_a_b_n_geq_1_l160_16031

theorem problem_a_b_n_geq_1 (a b n : ℕ) (h1 : a > b) (h2 : b > 1) (h3 : Odd b) (h4 : n > 0)
  (h5 : b^n ∣ a^n - 1) : a^b > 3^n / n := 
by 
  sorry

end problem_a_b_n_geq_1_l160_16031


namespace difference_of_cubes_l160_16059

theorem difference_of_cubes (x y : ℕ) (h1 : x = y + 3) (h2 : x + y = 5) : x^3 - y^3 = 63 :=
by sorry

end difference_of_cubes_l160_16059


namespace multiples_of_4_count_l160_16003

theorem multiples_of_4_count (a b : ℕ) (h₁ : a = 100) (h₂ : b = 400) :
  ∃ n : ℕ, n = 75 ∧ ∀ k : ℕ, (k >= a ∧ k <= b ∧ k % 4 = 0) ↔ (k / 4 - 25 ≥ 1 ∧ k / 4 - 25 ≤ n) :=
sorry

end multiples_of_4_count_l160_16003


namespace least_subtracted_divisible_l160_16010

theorem least_subtracted_divisible :
  ∃ k, (5264 - 11) = 17 * k :=
by
  sorry

end least_subtracted_divisible_l160_16010


namespace right_angled_triangle_side_length_l160_16060

theorem right_angled_triangle_side_length :
  ∃ c : ℕ, (c = 5) ∧ (3^2 + 4^2 = c^2) ∧ (c = 4 + 1) := by
  sorry

end right_angled_triangle_side_length_l160_16060


namespace find_hypotenuse_l160_16069

-- Let a, b be the legs of the right triangle, c be the hypotenuse.
-- Let h be the altitude to the hypotenuse and r be the radius of the inscribed circle.
variable (a b c h r : ℝ)

-- Assume conditions of a right-angled triangle
def right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Given the altitude to the hypotenuse
def altitude (h c : ℝ) : Prop :=
  ∃ a b : ℝ, right_angled a b c ∧ h = a * b / c

-- Given the radius of the inscribed circle
def inscribed_radius (r a b c : ℝ) : Prop :=
  r = (a + b - c) / 2

-- The proof problem statement
theorem find_hypotenuse (a b c h r : ℝ) 
  (h_right_angled : right_angled a b c)
  (h_altitude : altitude h c)
  (h_inscribed_radius : inscribed_radius r a b c) : 
  c = 2 * r^2 / (h - 2 * r) :=
  sorry

end find_hypotenuse_l160_16069


namespace find_third_root_l160_16034

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end find_third_root_l160_16034


namespace geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l160_16040

theorem geometric_sequence_general_term_and_arithmetic_sequence_max_sum :
  (∃ a_n : ℕ → ℕ, ∃ b_n : ℕ → ℤ, ∃ T_n : ℕ → ℤ,
    (∀ n, a_n n = 2^(n-1)) ∧
    (a_n 1 + a_n 2 = 3) ∧
    (b_n 2 = a_n 3) ∧
    (b_n 3 = -b_n 5) ∧
    (∀ n, T_n n = n * (b_n 1 + b_n n) / 2) ∧
    (T_n 3 = 12) ∧
    (T_n 4 = 12)) :=
by
  sorry

end geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l160_16040


namespace distance_to_origin_l160_16082

theorem distance_to_origin (a : ℝ) (h: |a| = 5) : 3 - a = -2 ∨ 3 - a = 8 :=
sorry

end distance_to_origin_l160_16082


namespace orchestra_members_l160_16067

theorem orchestra_members (n : ℕ) (h₀ : 100 ≤ n) (h₁ : n ≤ 300)
    (h₂ : n % 4 = 3) (h₃ : n % 5 = 1) (h₄ : n % 7 = 5) : n = 231 := by
  sorry

end orchestra_members_l160_16067


namespace card_game_probability_l160_16098

theorem card_game_probability :
  let A_wins := 4;  -- number of heads needed for A to win all cards
  let B_wins := 4;  -- number of tails needed for B to win all cards
  let total_flips := 5;  -- exactly 5 flips
  (Nat.choose total_flips 1 + Nat.choose total_flips 1) / (2^total_flips) = 5 / 16 :=
by
  sorry

end card_game_probability_l160_16098


namespace solve_garden_width_l160_16020

noncomputable def garden_width_problem (w l : ℕ) :=
  (w + l = 30) ∧ (w * l = 200) ∧ (l = w + 8) → w = 11

theorem solve_garden_width (w l : ℕ) : garden_width_problem w l :=
by
  intro h
  -- Omitting the actual proof
  sorry

end solve_garden_width_l160_16020


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l160_16029

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l160_16029


namespace rhombus_area_fraction_l160_16023

theorem rhombus_area_fraction :
  let grid_area := 36
  let vertices := [(2, 2), (4, 2), (3, 3), (3, 1)]
  let rhombus_area := 2
  rhombus_area / grid_area = 1 / 18 :=
by
  sorry

end rhombus_area_fraction_l160_16023


namespace sandbag_weight_l160_16061

theorem sandbag_weight (s : ℝ) (f : ℝ) (h : ℝ) : 
  f = 0.75 ∧ s = 450 ∧ h = 0.65 → f * s + h * (f * s) = 556.875 :=
by
  intro hfs
  sorry

end sandbag_weight_l160_16061


namespace garden_breadth_l160_16013

theorem garden_breadth (P L B : ℕ) (h₁ : P = 950) (h₂ : L = 375) (h₃ : P = 2 * (L + B)) : B = 100 := by
  sorry

end garden_breadth_l160_16013


namespace single_ticket_cost_l160_16049

/-- Define the conditions: sales total, attendee count, number of couple tickets, and cost of couple tickets. -/
def total_sales : ℤ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16
def cost_of_couple_ticket : ℤ := 35

/-- Define the derived conditions: people covered by couple tickets, single tickets sold, and sales from couple tickets. -/
def people_covered_by_couple_tickets : ℕ := couple_tickets_sold * 2
def single_tickets_sold : ℕ := total_attendees - people_covered_by_couple_tickets
def sales_from_couple_tickets : ℤ := couple_tickets_sold * cost_of_couple_ticket

/-- Define the core equation that ties single ticket sales to the total sales. -/
def core_equation (x : ℤ) : Bool := 
  sales_from_couple_tickets + single_tickets_sold * x = total_sales

-- Finally, the statement that needs to be proved.
theorem single_ticket_cost :
  ∃ x : ℤ, core_equation x ∧ x = 18 := by
  sorry

end single_ticket_cost_l160_16049


namespace find_b_l160_16089

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a = b + 2 ∧ 
  b = 2 * c ∧ 
  a + b + c = 42

theorem find_b (a b c : ℕ) (h : conditions a b c) : b = 16 := 
sorry

end find_b_l160_16089


namespace simplify_expression_l160_16024

theorem simplify_expression : 
  ((3 + 4 + 5 + 6 + 7) / 3 + (3 * 6 + 9)^2 / 9) = 268 / 3 := 
by 
  sorry

end simplify_expression_l160_16024


namespace geoboard_quadrilaterals_l160_16006

-- Definitions of the quadrilaterals as required by the conditions of the problem.
def quadrilateral_area (quad : Type) : ℝ := sorry
def quadrilateral_perimeter (quad : Type) : ℝ := sorry

-- Declaration of Quadrilateral I and II on a geoboard.
def quadrilateral_i : Type := sorry
def quadrilateral_ii : Type := sorry

-- The proof problem statement.
theorem geoboard_quadrilaterals :
  quadrilateral_area quadrilateral_i = quadrilateral_area quadrilateral_ii ∧
  quadrilateral_perimeter quadrilateral_i < quadrilateral_perimeter quadrilateral_ii := by
  sorry

end geoboard_quadrilaterals_l160_16006


namespace fruits_eaten_l160_16066

theorem fruits_eaten (initial_cherries initial_strawberries initial_blueberries left_cherries left_strawberries left_blueberries : ℕ)
  (h1 : initial_cherries = 16) (h2 : initial_strawberries = 10) (h3 : initial_blueberries = 20)
  (h4 : left_cherries = 6) (h5 : left_strawberries = 8) (h6 : left_blueberries = 15) :
  (initial_cherries - left_cherries) + (initial_strawberries - left_strawberries) + (initial_blueberries - left_blueberries) = 17 := 
by
  sorry

end fruits_eaten_l160_16066


namespace smallest_integer_l160_16094

-- Define a function to calculate the LCM of a list of numbers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

-- List of divisors
def divisors : List ℕ := [4, 5, 6, 7, 8, 9, 10]

-- Calculating the required integer
noncomputable def required_integer : ℕ := lcm_list divisors + 1

-- The proof statement
theorem smallest_integer : required_integer = 2521 :=
  by 
  sorry

end smallest_integer_l160_16094


namespace y_coord_vertex_of_parabola_l160_16012

-- Define the quadratic equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Statement to prove
theorem y_coord_vertex_of_parabola : ∃ (x : ℝ), parabola x = 2 * (x + 4)^2 - 3 := sorry

end y_coord_vertex_of_parabola_l160_16012


namespace max_tied_teams_l160_16033

theorem max_tied_teams (n : ℕ) (h_n : n = 8) (tournament : Fin n → Fin n → Prop)
  (h_symmetric : ∀ i j, tournament i j ↔ tournament j i)
  (h_antisymmetric : ∀ i j, tournament i j → ¬ tournament j i)
  (h_total : ∀ i j, i ≠ j → tournament i j ∨ tournament j i) :
  ∃ (k : ℕ), k = 7 ∧ ∀ (wins : Fin n → ℕ), 
  (∀ i, wins i = 4 → ∃! j, i ≠ j ∧ tournament i j) → True :=
by sorry

end max_tied_teams_l160_16033


namespace problem1_l160_16044

theorem problem1 (x : ℝ) (n : ℕ) (h : x^n = 2) : (3 * x^n)^2 - 4 * (x^2)^n = 20 :=
by
  sorry

end problem1_l160_16044


namespace treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l160_16074

namespace PirateTreasure

-- Given conditions
def num_pirates_excl_captain := 100
def max_coins := 1000
def remaining_coins_99_pirates := 51
def remaining_coins_77_pirates := 29

-- Problem Part (a): Prove the number of coins in treasure
theorem treasure_contains_645_coins : 
  ∃ (N : ℕ), N < max_coins ∧ (N % 99 = remaining_coins_99_pirates ∧ N % 77 = remaining_coins_77_pirates) ∧ N = 645 :=
  sorry

-- Problem Part (b): Prove the number of pirates Barbaroxa should choose
theorem max_leftover_coins_when_choosing_93_pirates :
  ∃ (n : ℕ), n ≤ num_pirates_excl_captain ∧ (∀ k, k ≤ num_pirates_excl_captain → (645 % k) ≤ (645 % k) ∧ n = 93) :=
  sorry

end PirateTreasure

end treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l160_16074


namespace smallest_of_three_numbers_l160_16091

theorem smallest_of_three_numbers : ∀ (a b c : ℕ), (a = 5) → (b = 8) → (c = 4) → min (min a b) c = 4 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end smallest_of_three_numbers_l160_16091


namespace bottle_t_capsules_l160_16016

theorem bottle_t_capsules 
  (num_capsules_r : ℕ)
  (cost_r : ℝ)
  (cost_t : ℝ)
  (cost_per_capsule_difference : ℝ)
  (h1 : num_capsules_r = 250)
  (h2 : cost_r = 6.25)
  (h3 : cost_t = 3.00)
  (h4 : cost_per_capsule_difference = 0.005) :
  ∃ (num_capsules_t : ℕ), num_capsules_t = 150 := 
by
  sorry

end bottle_t_capsules_l160_16016


namespace reciprocal_of_minus_one_over_2023_l160_16055

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l160_16055


namespace sine_tangent_not_possible_1_sine_tangent_not_possible_2_l160_16051

theorem sine_tangent_not_possible_1 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.27413 ∧ Real.tan θ = 0.25719) :=
sorry

theorem sine_tangent_not_possible_2 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.25719 ∧ Real.tan θ = 0.27413) :=
sorry

end sine_tangent_not_possible_1_sine_tangent_not_possible_2_l160_16051


namespace find_x_plus_y_l160_16093

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3000)
  (h2 : x + 3000 * Real.sin y = 2999) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
  sorry

end find_x_plus_y_l160_16093


namespace total_lives_remaining_l160_16064

theorem total_lives_remaining (initial_players quit_players : Nat) 
  (lives_3_players lives_4_players lives_2_players bonus_lives : Nat)
  (h1 : initial_players = 16)
  (h2 : quit_players = 7)
  (h3 : lives_3_players = 10)
  (h4 : lives_4_players = 8)
  (h5 : lives_2_players = 6)
  (h6 : bonus_lives = 4)
  (remaining_players : Nat)
  (h7 : remaining_players = initial_players - quit_players)
  (lives_before_bonus : Nat)
  (h8 : lives_before_bonus = 3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players)
  (bonus_total : Nat)
  (h9 : bonus_total = remaining_players * bonus_lives) :
  3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players + remaining_players * bonus_lives = 110 :=
by
  sorry

end total_lives_remaining_l160_16064


namespace find_m_l160_16007

theorem find_m (m : ℕ) (h : 10^(m-1) < 2^512 ∧ 2^512 < 10^m): 
  m = 155 :=
sorry

end find_m_l160_16007


namespace supervisors_per_bus_l160_16009

theorem supervisors_per_bus (total_supervisors : ℕ) (total_buses : ℕ) (H1 : total_supervisors = 21) (H2 : total_buses = 7) : (total_supervisors / total_buses = 3) :=
by
  sorry

end supervisors_per_bus_l160_16009


namespace solve_equation_l160_16043

theorem solve_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ -1) : (x / (x + 1) = 1 + (1 / x)) ↔ (x = -1 / 2) :=
by
  sorry

end solve_equation_l160_16043


namespace tree_height_fraction_l160_16004

theorem tree_height_fraction :
  ∀ (initial_height growth_per_year : ℝ),
  initial_height = 4 ∧ growth_per_year = 0.5 →
  ((initial_height + 6 * growth_per_year) - (initial_height + 4 * growth_per_year)) / (initial_height + 4 * growth_per_year) = 1 / 6 :=
by
  intros initial_height growth_per_year h
  rcases h with ⟨h1, h2⟩
  sorry

end tree_height_fraction_l160_16004


namespace systematic_sampling_student_number_l160_16088

theorem systematic_sampling_student_number 
  (total_students : ℕ)
  (sample_size : ℕ)
  (interval_between_numbers : ℕ)
  (student_17_in_sample : ∃ n, 17 = n ∧ n ≤ total_students ∧ n % interval_between_numbers = 5)
  : ∃ m, m = 41 ∧ m ≤ total_students ∧ m % interval_between_numbers = 5 := 
sorry

end systematic_sampling_student_number_l160_16088


namespace douglas_votes_in_Y_is_46_l160_16097

variable (V : ℝ)
variable (P : ℝ)

def percentage_won_in_Y :=
  let total_voters_X := 2 * V
  let total_voters_Y := V
  let votes_in_X := 0.64 * total_voters_X
  let votes_in_Y := P / 100 * total_voters_Y
  let total_votes := 1.28 * V + (P / 100 * V)
  let combined_voters := 3 * V
  let combined_votes_percentage := 0.58 * combined_voters
  P = 46

theorem douglas_votes_in_Y_is_46
  (V_pos : V > 0)
  (H : 1.28 * V + (P / 100 * V) = 0.58 * 3 * V) :
  percentage_won_in_Y V P := by
  sorry

end douglas_votes_in_Y_is_46_l160_16097


namespace Jack_Income_Ratio_l160_16027

noncomputable def Ernie_current_income (x : ℕ) : ℕ :=
  (4 / 5) * x

noncomputable def Jack_current_income (combined_income Ernie_current_income : ℕ) : ℕ :=
  combined_income - Ernie_current_income

theorem Jack_Income_Ratio (Ernie_previous_income combined_income : ℕ) (h₁ : Ernie_previous_income = 6000) (h₂ : combined_income = 16800) :
  let Ernie_current := Ernie_current_income Ernie_previous_income
  let Jack_current := Jack_current_income combined_income Ernie_current
  (Jack_current / Ernie_previous_income) = 2 := by
  sorry

end Jack_Income_Ratio_l160_16027


namespace cosine_eq_one_fifth_l160_16073

theorem cosine_eq_one_fifth {α : ℝ} 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := 
sorry

end cosine_eq_one_fifth_l160_16073


namespace rectangle_diagonal_ratio_l160_16065

theorem rectangle_diagonal_ratio (s : ℝ) :
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  D / d = Real.sqrt 5 :=
by
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  sorry

end rectangle_diagonal_ratio_l160_16065


namespace percentage_calculation_l160_16077

-- Definitions based on conditions
def x : ℕ := 5200
def p1 : ℚ := 0.50
def p2 : ℚ := 0.30
def p3 : ℚ := 0.15

-- The theorem stating the desired proof
theorem percentage_calculation : p3 * (p2 * (p1 * x)) = 117 := by
  sorry

end percentage_calculation_l160_16077


namespace find_inequality_solution_set_l160_16075

noncomputable def inequality_solution_set : Set ℝ :=
  { x | (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < (1 / 4) }

theorem find_inequality_solution_set :
  inequality_solution_set = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end find_inequality_solution_set_l160_16075


namespace triangle_formation_inequalities_l160_16068

theorem triangle_formation_inequalities (a b c d : ℝ)
  (h_abc_pos : 0 < a)
  (h_bcd_pos : 0 < b)
  (h_cde_pos : 0 < c)
  (h_def_pos : 0 < d)
  (tri_ineq_1 : a + b + c > d)
  (tri_ineq_2 : b + c + d > a)
  (tri_ineq_3 : a + d > b + c) :
  (a < (b + c + d) / 2) ∧ (b + c < a + d) ∧ (¬ (c + d < b / 2)) :=
by 
  sorry

end triangle_formation_inequalities_l160_16068


namespace total_amount_l160_16096

variable (A B C : ℕ)
variable (h1 : C = 495)
variable (h2 : (A - 10) * 18 = (B - 20) * 11)
variable (h3 : (B - 20) * 24 = (C - 15) * 18)

theorem total_amount (A B C : ℕ) (h1 : C = 495)
  (h2 : (A - 10) * 18 = (B - 20) * 11)
  (h3 : (B - 20) * 24 = (C - 15) * 18) :
  A + B + C = 1105 :=
sorry

end total_amount_l160_16096


namespace subtraction_of_bases_l160_16037

def base8_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 7^2 + ((n % 100) / 10) * 7^1 + (n % 10) * 7^0

theorem subtraction_of_bases :
  base8_to_base10 343 - base7_to_base10 265 = 82 :=
by
  sorry

end subtraction_of_bases_l160_16037


namespace maurice_earnings_l160_16063

theorem maurice_earnings (bonus_per_10_tasks : ℕ → ℕ) (num_tasks : ℕ) (total_earnings : ℕ) :
  (∀ n, n * (bonus_per_10_tasks n) = 6 * n) →
  num_tasks = 30 →
  total_earnings = 78 →
  bonus_per_10_tasks num_tasks / 10 = 3 →
  (total_earnings - (bonus_per_10_tasks num_tasks / 10) * 6) / num_tasks = 2 :=
by
  intros h_bonus h_num_tasks h_total_earnings h_bonus_count
  sorry

end maurice_earnings_l160_16063


namespace agnes_flight_cost_l160_16030

theorem agnes_flight_cost
  (booking_fee : ℝ) (cost_per_km : ℝ) (distance_XY : ℝ)
  (h1 : booking_fee = 120)
  (h2 : cost_per_km = 0.12)
  (h3 : distance_XY = 4500) :
  booking_fee + cost_per_km * distance_XY = 660 := 
by
  sorry

end agnes_flight_cost_l160_16030


namespace average_visitors_per_day_in_november_l160_16028
-- Import the entire Mathlib library for necessary definitions and operations.

-- Define the average visitors per different days of the week.
def sunday_visitors := 510
def monday_visitors := 240
def tuesday_visitors := 240
def wednesday_visitors := 300
def thursday_visitors := 300
def friday_visitors := 200
def saturday_visitors := 200

-- Define the counts of each type of day in November.
def sundays := 5
def mondays := 4
def tuesdays := 4
def wednesdays := 4
def thursdays := 4
def fridays := 4
def saturdays := 4

-- Define the number of days in November.
def days_in_november := 30

-- State the theorem to prove the average number of visitors per day.
theorem average_visitors_per_day_in_november : 
  (5 * sunday_visitors + 
   4 * monday_visitors + 
   4 * tuesday_visitors + 
   4 * wednesday_visitors + 
   4 * thursday_visitors + 
   4 * friday_visitors + 
   4 * saturday_visitors) / days_in_november = 282 :=
by
  sorry

end average_visitors_per_day_in_november_l160_16028


namespace inequality_proof_l160_16000

theorem inequality_proof 
  (x1 x2 y1 y2 z1 z2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2)
  (hxy1 : x1 * y1 > z1 ^ 2)
  (hxy2 : x2 * y2 > z2 ^ 2) :
  8 / ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2) ≤
  1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2) :=
sorry

end inequality_proof_l160_16000


namespace brick_surface_area_l160_16011

theorem brick_surface_area (l w h : ℝ) (hl : l = 10) (hw : w = 4) (hh : h = 3) : 
  2 * (l * w + l * h + w * h) = 164 := 
by
  sorry

end brick_surface_area_l160_16011


namespace time_to_cross_bridge_l160_16072

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (speed_conversion_factor : ℝ) (time_conversion_factor : ℝ) (expected_time : ℝ) :
  speed_km_hr = 5 →
  length_m = 1250 →
  speed_conversion_factor = 1000 →
  time_conversion_factor = 60 →
  expected_time = length_m / (speed_km_hr * (speed_conversion_factor / time_conversion_factor)) →
  expected_time = 15 :=
by
  intros
  sorry

end time_to_cross_bridge_l160_16072


namespace percentage_increase_in_expenses_l160_16087

variable (a b c : ℝ)

theorem percentage_increase_in_expenses :
  (10 / 100 * a + 30 / 100 * b + 20 / 100 * c) / (a + b + c) =
  (10 * a + 30 * b + 20 * c) / (100 * (a + b + c)) :=
by
  sorry

end percentage_increase_in_expenses_l160_16087


namespace necessary_and_sufficient_condition_l160_16045

variable (p q : Prop)

theorem necessary_and_sufficient_condition (hp : p) (hq : q) : ¬p ∨ ¬q = False :=
by {
    -- You are requested to fill out the proof here.
    sorry
}

end necessary_and_sufficient_condition_l160_16045


namespace find_x_plus_inv_x_l160_16083

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l160_16083


namespace simplify_expression_as_single_fraction_l160_16048

variable (d : ℚ)

theorem simplify_expression_as_single_fraction :
  (5 + 4*d)/9 + 3 = (32 + 4*d)/9 := 
by
  sorry

end simplify_expression_as_single_fraction_l160_16048


namespace unique_pairs_of_socks_l160_16002

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end unique_pairs_of_socks_l160_16002


namespace geometric_sequence_sum_5_l160_16090

theorem geometric_sequence_sum_5 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q, ∀ n, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  (a 1 * (1 - (2:ℝ)^5) / (1 - (2:ℝ))) = 31 := 
by
  sorry

end geometric_sequence_sum_5_l160_16090


namespace milk_leftover_l160_16084

theorem milk_leftover 
  (total_milk : ℕ := 24)
  (kids_percent : ℝ := 0.80)
  (cooking_percent : ℝ := 0.60)
  (neighbor_percent : ℝ := 0.25)
  (husband_percent : ℝ := 0.06) :
  let milk_after_kids := total_milk * (1 - kids_percent)
  let milk_after_cooking := milk_after_kids * (1 - cooking_percent)
  let milk_after_neighbor := milk_after_cooking * (1 - neighbor_percent)
  let milk_after_husband := milk_after_neighbor * (1 - husband_percent)
  milk_after_husband = 1.3536 :=
by 
  -- skip the proof for simplicity
  sorry

end milk_leftover_l160_16084


namespace smallest_b_satisfying_inequality_l160_16080

theorem smallest_b_satisfying_inequality : ∀ b : ℝ, (b^2 - 16 * b + 55) ≥ 0 ↔ b ≤ 5 ∨ b ≥ 11 := sorry

end smallest_b_satisfying_inequality_l160_16080


namespace cost_per_bar_l160_16039

variable (months_in_year : ℕ := 12)
variable (months_per_bar_of_soap : ℕ := 2)
variable (total_cost_for_year : ℕ := 48)

theorem cost_per_bar (h1 : months_per_bar_of_soap > 0)
                     (h2 : total_cost_for_year > 0) : 
    (total_cost_for_year / (months_in_year / months_per_bar_of_soap)) = 8 := 
by
  sorry

end cost_per_bar_l160_16039


namespace new_ratio_after_2_years_l160_16014

-- Definitions based on conditions
variable (A : ℕ) -- Current age of a
variable (B : ℕ) -- Current age of b

-- Conditions
def ratio_a_b := A / B = 5 / 3
def current_age_b := B = 6

-- Theorem: New ratio after 2 years is 3:2
theorem new_ratio_after_2_years (h1 : ratio_a_b A B) (h2 : current_age_b B) : (A + 2) / (B + 2) = 3 / 2 := by
  sorry

end new_ratio_after_2_years_l160_16014


namespace complex_inequality_l160_16022

open Complex

noncomputable def condition (a b c : ℂ) := a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0

theorem complex_inequality (a b c : ℂ) (h : condition a b c) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := 
sorry

end complex_inequality_l160_16022


namespace count_board_configurations_l160_16052

-- Define the 3x3 board as a type with 9 positions
inductive Position 
| top_left | top_center | top_right
| middle_left | center | middle_right
| bottom_left | bottom_center | bottom_right

-- Define an enum for players' moves
inductive Mark
| X | O | Empty

-- Define a board as a mapping from positions to marks
def Board : Type := Position → Mark

-- Define the win condition for Carl
def win_condition (b : Board) : Prop := 
(b Position.center = Mark.O) ∧ 
((b Position.top_left = Mark.O ∧ b Position.top_center = Mark.O) ∨ 
(b Position.middle_left = Mark.O ∧ b Position.middle_right = Mark.O) ∨ 
(b Position.bottom_left = Mark.O ∧ b Position.bottom_center = Mark.O))

-- Define the condition for a filled board
def filled_board (b : Board) : Prop :=
∀ p : Position, b p ≠ Mark.Empty

-- The proof problem to show the total number of configurations is 30
theorem count_board_configurations : 
  ∃ (n : ℕ), n = 30 ∧
  (∃ b : Board, win_condition b ∧ filled_board b) := 
sorry

end count_board_configurations_l160_16052


namespace find_f_neg_2017_l160_16078

-- Define f as given in the problem
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the given problem condition
def condition (a b : ℝ) : Prop :=
  f a b 2017 = 10

-- The main problem statement proving the solution
theorem find_f_neg_2017 (a b : ℝ) (h : condition a b) : f a b (-2017) = -14 :=
by
  -- We state this theorem and provide a sorry to skip the proof
  sorry

end find_f_neg_2017_l160_16078


namespace linear_eq_solution_l160_16026

theorem linear_eq_solution (m x : ℝ) (h : |m| = 1) (h1: 1 - m ≠ 0):
  x = -(1/2) :=
sorry

end linear_eq_solution_l160_16026


namespace lateral_surface_area_of_cone_l160_16018

open Real

theorem lateral_surface_area_of_cone
  (SA : ℝ) (SB : ℝ)
  (cos_angle_SA_SB : ℝ) (angle_SA_base : ℝ)
  (area_SAB : ℝ) :
  cos_angle_SA_SB = 7 / 8 →
  angle_SA_base = π / 4 →
  area_SAB = 5 * sqrt 15 →
  SA = 4 * sqrt 5 →
  SB = SA →
  (1/2) * (sqrt 2 / 2 * SA) * (2 * π * SA) = 40 * sqrt 2 * π :=
sorry

end lateral_surface_area_of_cone_l160_16018


namespace area_of_fourth_square_l160_16021

theorem area_of_fourth_square (AB BC AC CD AD : ℝ) (h_sum_ABC : AB^2 + 25 = 50)
  (h_sum_ACD : 50 + 49 = AD^2) : AD^2 = 99 :=
by
  sorry

end area_of_fourth_square_l160_16021


namespace eat_jar_together_time_l160_16054

-- Define the rate of the child
def child_rate := 1 / 6

-- Define the rate of Karlson who eats twice as fast as the child
def karlson_rate := 2 * child_rate

-- Define the combined rate when both eat together
def combined_rate := child_rate + karlson_rate

-- Prove that the time taken together to eat one jar is 2 minutes
theorem eat_jar_together_time : (1 / combined_rate) = 2 :=
by
  -- Add the proof steps here
  sorry

end eat_jar_together_time_l160_16054


namespace function_relationship_l160_16086

theorem function_relationship (f : ℝ → ℝ)
  (h₁ : ∀ x, f (x + 1) = f (-x + 1))
  (h₂ : ∀ x, x ≥ 1 → f x = (1 / 2) ^ x - 1) :
  f (2 / 3) > f (3 / 2) ∧ f (3 / 2) > f (1 / 3) :=
by sorry

end function_relationship_l160_16086


namespace arithmetic_geom_seq_a5_l160_16085

theorem arithmetic_geom_seq_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) (q : ℝ)
  (a1 : a 1 = 1)
  (S8 : s 8 = 17 * s 4) :
  a 5 = 16 :=
sorry

end arithmetic_geom_seq_a5_l160_16085


namespace quadratic_root_l160_16042

theorem quadratic_root (k : ℝ) (h : ∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) : k = -1 :=
sorry

end quadratic_root_l160_16042


namespace average_of_remaining_two_numbers_l160_16017

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.95)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 :=
sorry

end average_of_remaining_two_numbers_l160_16017


namespace ryan_learning_hours_l160_16047

theorem ryan_learning_hours (H_E : ℕ) (H_C : ℕ) (h1 : H_E = 6) (h2 : H_C = 2) : H_E - H_C = 4 := by
  sorry

end ryan_learning_hours_l160_16047


namespace question_a_question_b_l160_16050

-- Definitions
def isSolutionA (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 7

def isSolutionB (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 25

-- Statements
theorem question_a (a b : ℤ) : isSolutionA a b ↔ (a, b) ∈ [(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)] :=
sorry

theorem question_b (a b : ℤ) : isSolutionB a b ↔ (a, b) ∈ [(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)] :=
sorry

end question_a_question_b_l160_16050


namespace larger_pile_toys_l160_16058

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l160_16058


namespace cube_side_length_equals_six_l160_16025

theorem cube_side_length_equals_six {s : ℝ} (h : 6 * s ^ 2 = s ^ 3) : s = 6 :=
by
  sorry

end cube_side_length_equals_six_l160_16025


namespace gcd_of_987654_and_123456_l160_16070

theorem gcd_of_987654_and_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end gcd_of_987654_and_123456_l160_16070


namespace absent_children_l160_16015

theorem absent_children (total_children bananas_per_child_if_present bananas_per_child_if_absent children_present absent_children : ℕ) 
  (H1 : total_children = 740)
  (H2 : bananas_per_child_if_present = 2)
  (H3 : bananas_per_child_if_absent = 4)
  (H4 : children_present * bananas_per_child_if_absent = total_children * bananas_per_child_if_present)
  (H5 : children_present = total_children - absent_children) : 
  absent_children = 370 :=
sorry

end absent_children_l160_16015


namespace problem_statement_l160_16019

-- Define the universal set
def U : Set ℕ := {x | x ≤ 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {4, 5, 6}

-- Define the complement of A with respect to U
def complement_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- Define the intersection of the complement of A and B
def intersect_complement_A_B : Set ℕ := {x | x ∈ complement_A ∧ x ∈ B}

-- Theorem statement to be proven
theorem problem_statement : intersect_complement_A_B = {4, 6} :=
by
  sorry

end problem_statement_l160_16019


namespace triangle_inequality_l160_16095

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end triangle_inequality_l160_16095
