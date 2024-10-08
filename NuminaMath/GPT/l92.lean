import Mathlib

namespace chess_tournament_games_l92_92440

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 2 * n * (n - 1) = 1200 :=
by
  sorry

end chess_tournament_games_l92_92440


namespace solve_inequality_l92_92720

theorem solve_inequality (x : ℝ) : (|x - 3| + |x - 5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end solve_inequality_l92_92720


namespace xiao_hong_mistake_l92_92009

theorem xiao_hong_mistake (a : ℕ) (h : 31 - a = 12) : 31 + a = 50 :=
by
  sorry

end xiao_hong_mistake_l92_92009


namespace smallest_possible_value_m_l92_92053

theorem smallest_possible_value_m (r y b : ℕ) (h : 16 * r = 18 * y ∧ 18 * y = 20 * b) : 
  ∃ m : ℕ, 30 * m = 16 * r ∧ 30 * m = 720 ∧ m = 24 :=
by {
  sorry
}

end smallest_possible_value_m_l92_92053


namespace betty_total_blue_and_green_beads_l92_92924

theorem betty_total_blue_and_green_beads (r b g : ℕ) (h1 : 5 * b = 3 * r) (h2 : 5 * g = 2 * r) (h3 : r = 50) : b + g = 50 :=
by
  sorry

end betty_total_blue_and_green_beads_l92_92924


namespace graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l92_92637

open Real

def function_y (a x : ℝ) : ℝ := (4 * a + 2) * x^2 + (9 - 6 * a) * x - 4 * a + 4

theorem graph_t_intersects_x_axis (a : ℝ) : ∃ x : ℝ, function_y a x = 0 :=
by sorry

theorem exists_integer_a_with_integer_points_on_x_axis_intersection :
  ∃ (a : ℤ), 
  (∀ x : ℝ, (function_y a x = 0) → ∃ (x_int : ℤ), x = x_int) ∧ 
  (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1) :=
by sorry

end graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l92_92637


namespace solve_quadratic_equation_l92_92669

theorem solve_quadratic_equation (x : ℝ) :
  (x^2 - 2 * x - 5 = 0) ↔ (x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6) := 
sorry

end solve_quadratic_equation_l92_92669


namespace find_y_l92_92800

theorem find_y : (12 : ℝ)^3 * (2 : ℝ)^4 / 432 = 5184 → (2 : ℝ) = 2 :=
by
  intro h
  sorry

end find_y_l92_92800


namespace sum_to_12_of_7_chosen_l92_92863

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end sum_to_12_of_7_chosen_l92_92863


namespace min_overlap_l92_92443

noncomputable def drinks_coffee := 0.60
noncomputable def drinks_tea := 0.50
noncomputable def drinks_neither := 0.10
noncomputable def drinks_either := 1 - drinks_neither
noncomputable def total_overlap := drinks_coffee + drinks_tea - drinks_either

theorem min_overlap (hcoffee : drinks_coffee = 0.60) (htea : drinks_tea = 0.50) (hneither : drinks_neither = 0.10) :
  total_overlap = 0.20 :=
by
  sorry

end min_overlap_l92_92443


namespace arithmetic_sequence_sum_is_right_l92_92225

noncomputable def arithmetic_sequence_sum : ℤ :=
  let a1 := 1
  let d := -2
  let a2 := a1 + d
  let a3 := a1 + 2 * d
  let a6 := a1 + 5 * d
  let S6 := 6 * a1 + (6 * (6-1)) / 2 * d
  S6

theorem arithmetic_sequence_sum_is_right {d : ℤ} (h₀ : d ≠ 0) 
(h₁ : (a1 + 2 * d) ^ 2 = (a1 + d) * (a1 + 5 * d)) :
  arithmetic_sequence_sum = -24 := by
  sorry

end arithmetic_sequence_sum_is_right_l92_92225


namespace students_not_taking_math_or_physics_l92_92496

theorem students_not_taking_math_or_physics (total_students math_students phys_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 75)
  (h3 : phys_students = 50)
  (h4 : both_students = 15) :
  total_students - (math_students + phys_students - both_students) = 10 :=
by
  sorry

end students_not_taking_math_or_physics_l92_92496


namespace movement_down_l92_92776

def point := (ℤ × ℤ)

theorem movement_down (C D : point) (hC : C = (1, 2)) (hD : D = (1, -1)) :
  D = (C.1, C.2 - 3) :=
by
  sorry

end movement_down_l92_92776


namespace unique_sum_of_three_distinct_positive_perfect_squares_l92_92838

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_positive_perfect_squares_that_sum_to (a b c sum : ℕ) : Prop :=
  is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c ∧
  a < b ∧ b < c ∧ a + b + c = sum

theorem unique_sum_of_three_distinct_positive_perfect_squares :
  (∃ a b c : ℕ, distinct_positive_perfect_squares_that_sum_to a b c 100) ∧
  (∀ a1 b1 c1 a2 b2 c2 : ℕ,
    distinct_positive_perfect_squares_that_sum_to a1 b1 c1 100 ∧
    distinct_positive_perfect_squares_that_sum_to a2 b2 c2 100 →
    (a1 = a2 ∧ b1 = b2 ∧ c1 = c2)) :=
by
  sorry

end unique_sum_of_three_distinct_positive_perfect_squares_l92_92838


namespace john_spent_l92_92500

/-- John bought 9.25 meters of cloth at a cost price of $44 per meter.
    Prove that the total amount John spent on the cloth is $407. -/
theorem john_spent :
  let length_of_cloth := 9.25
  let cost_per_meter := 44
  let total_cost := length_of_cloth * cost_per_meter
  total_cost = 407 := by
  sorry

end john_spent_l92_92500


namespace b_investment_l92_92431

theorem b_investment (a_investment : ℝ) (c_investment : ℝ) (total_profit : ℝ) (a_share_profit : ℝ) (b_investment : ℝ) : a_investment = 6300 → c_investment = 10500 → total_profit = 14200 → a_share_profit = 4260 → b_investment = 4220 :=
by
  intro h_a h_c h_total h_a_share
  have h1 : 6300 / (6300 + 4220 + 10500) = 4260 / 14200 := sorry
  have h2 : 6300 * 14200 = 4260 * (6300 + 4220 + 10500) := sorry
  have h3 : b_investment = 4220 := sorry
  exact h3

end b_investment_l92_92431


namespace second_train_speed_l92_92674

theorem second_train_speed (d : ℝ) (s₁ : ℝ) (t₁ : ℝ) (t₂ : ℝ) (meet_time : ℝ) (total_distance : ℝ) :
  d = 110 ∧ s₁ = 20 ∧ t₁ = 3 ∧ t₂ = 2 ∧ meet_time = 10 ∧ total_distance = d →
  60 + 2 * (total_distance - 60) / 2 = 110 →
  (total_distance - 60) / 2 = 25 :=
by
  intro h1 h2
  sorry

end second_train_speed_l92_92674


namespace smallest_five_digit_perfect_square_and_cube_l92_92544

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ n = 15625 :=
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l92_92544


namespace T_is_x_plus_3_to_the_4_l92_92269

variable (x : ℝ)

def T : ℝ := (x + 2)^4 + 4 * (x + 2)^3 + 6 * (x + 2)^2 + 4 * (x + 2) + 1

theorem T_is_x_plus_3_to_the_4 : T x = (x + 3)^4 := by
  -- Proof would go here
  sorry

end T_is_x_plus_3_to_the_4_l92_92269


namespace subtraction_division_l92_92834

theorem subtraction_division : 3550 - (1002 / 20.04) = 3499.9501 := by
  sorry

end subtraction_division_l92_92834


namespace unique_pair_exists_l92_92079

theorem unique_pair_exists :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  (a + b + (Nat.gcd a b)^2 = Nat.lcm a b) ∧
  (Nat.lcm a b = 2 * Nat.lcm (a - 1) b) ∧
  (a, b) = (6, 15) :=
sorry

end unique_pair_exists_l92_92079


namespace real_number_c_l92_92253

theorem real_number_c (x1 x2 c : ℝ) (h_eqn : x1 + x2 = -1) (h_prod : x1 * x2 = c) (h_cond : x1^2 * x2 + x2^2 * x1 = 3) : c = -3 :=
by sorry

end real_number_c_l92_92253


namespace smallest_five_consecutive_even_sum_320_l92_92878

theorem smallest_five_consecutive_even_sum_320 : ∃ (a b c d e : ℤ), a + b + c + d + e = 320 ∧ (∀ i j : ℤ, (i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e) → (j = a ∨ j = b ∨ j = c ∨ j = d ∨ j = e) → (i = j + 2 ∨ i = j - 2 ∨ i = j)) ∧ (a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∧ a = 60 :=
by
  sorry

end smallest_five_consecutive_even_sum_320_l92_92878


namespace ball_cost_l92_92913

theorem ball_cost (C x y : ℝ)
  (H1 :  x = 1/3 * (C/2 + y + 5) )
  (H2 :  y = 1/4 * (C/2 + x + 5) )
  (H3 :  C/2 + x + y + 5 = C ) : C = 20 := 
by
  sorry

end ball_cost_l92_92913


namespace audi_crossing_intersection_between_17_and_18_l92_92086

-- Given conditions:
-- Two cars, an Audi and a BMW, are moving along two intersecting roads at equal constant speeds.
-- At both 17:00 and 18:00, the BMW was twice as far from the intersection as the Audi.
-- Let the distance of Audi from the intersection at 17:00 be x and BMW's distance be 2x.
-- Both vehicles travel at a constant speed v.

noncomputable def car_position (initial_distance : ℝ) (velocity : ℝ) (time_elapsed : ℝ) : ℝ :=
  initial_distance + velocity * time_elapsed

theorem audi_crossing_intersection_between_17_and_18 (x v : ℝ) :
  ∃ t : ℝ, (t = 15 ∨ t = 45) ∧
    car_position x (-v) (t/60) = 0 ∧ car_position (2 * x) (-v) (t/60) = 2 * car_position x (-v) (1 - t/60) :=
sorry

end audi_crossing_intersection_between_17_and_18_l92_92086


namespace modulus_complex_number_l92_92566

theorem modulus_complex_number (i : ℂ) (h : i = Complex.I) : 
  Complex.abs (1 / (i - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end modulus_complex_number_l92_92566


namespace range_of_AB_l92_92286

variable (AB BC AC : ℝ)
variable (θ : ℝ)
variable (B : ℝ)

-- Conditions
axiom angle_condition : θ = 150
axiom length_condition : AC = 2

-- Theorem to prove
theorem range_of_AB (h_θ : θ = 150) (h_AC : AC = 2) : (0 < AB) ∧ (AB ≤ 4) :=
sorry

end range_of_AB_l92_92286


namespace range_of_k_l92_92264

theorem range_of_k 
  (h : ∀ x : ℝ, x^2 + 2 * k * x - (k - 2) > 0) : -2 < k ∧ k < 1 := 
sorry

end range_of_k_l92_92264


namespace total_legs_correct_l92_92406

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end total_legs_correct_l92_92406


namespace top_card_is_11_l92_92060

-- Define the initial configuration of cards
def initial_array : List (List Nat) := [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12],
  [13, 14, 15, 16, 17, 18]
]

-- Perform the described sequence of folds
def fold1 (arr : List (List Nat)) : List (List Nat) := [
  [3, 4, 5, 6],
  [9, 10, 11, 12],
  [15, 16, 17, 18],
  [1, 2],
  [7, 8],
  [13, 14]
]

def fold2 (arr : List (List Nat)) : List (List Nat) := [
  [5, 6],
  [11, 12],
  [17, 18],
  [3, 4, 1, 2],
  [9, 10, 7, 8],
  [15, 16, 13, 14]
]

def fold3 (arr : List (List Nat)) : List (List Nat) := [
  [11, 12, 7, 8],
  [17, 18, 13, 14],
  [5, 6, 1, 2],
  [9, 10, 3, 4],
  [15, 16, 9, 10]
]

-- Define the final array after all the folds
def final_array := fold3 (fold2 (fold1 initial_array))

-- Statement to be proven
theorem top_card_is_11 : (final_array.head!.head!) = 11 := 
  by
    sorry -- Proof to be filled in

end top_card_is_11_l92_92060


namespace smallest_d_l92_92629

theorem smallest_d (d : ℕ) (h : 3150 * d = k ^ 2) : d = 14 :=
by
  -- assuming the condition: 3150 = 2 * 3 * 5^2 * 7
  have h_factorization : 3150 = 2 * 3 * 5^2 * 7 := by sorry
  -- based on the computation and verification, the smallest d that satisfies the condition is 14
  sorry

end smallest_d_l92_92629


namespace books_ratio_l92_92337

-- Definitions based on the conditions
def Alyssa_books : Nat := 36
def Nancy_books : Nat := 252

-- Statement to prove
theorem books_ratio :
  (Nancy_books / Alyssa_books) = 7 := 
sorry

end books_ratio_l92_92337


namespace autumn_sales_l92_92023

theorem autumn_sales (T : ℝ) (spring summer winter autumn : ℝ) 
    (h1 : spring = 3)
    (h2 : summer = 6)
    (h3 : winter = 5)
    (h4 : T = (3 / 0.2)) :
    autumn = 1 :=
by 
  -- Proof goes here
  sorry

end autumn_sales_l92_92023


namespace four_distinct_numbers_are_prime_l92_92516

-- Lean 4 statement proving the conditions
theorem four_distinct_numbers_are_prime : 
  ∃ (a b c d : ℕ), 
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧ 
    (Prime (a * b + c * d)) ∧ 
    (Prime (a * c + b * d)) ∧ 
    (Prime (a * d + b * c)) := 
sorry

end four_distinct_numbers_are_prime_l92_92516


namespace sequence_term_l92_92205

theorem sequence_term (S : ℕ → ℕ) (h : ∀ (n : ℕ), S n = 5 * n + 2 * n^2) (r : ℕ) : 
  (S r - S (r - 1) = 4 * r + 3) :=
by {
  sorry
}

end sequence_term_l92_92205


namespace y_intercept_line_l92_92804

theorem y_intercept_line : ∀ y : ℝ, (∃ x : ℝ, x = 0 ∧ x - 3 * y - 1 = 0) → y = -1/3 :=
by
  intro y
  intro h
  sorry

end y_intercept_line_l92_92804


namespace smallest_period_pi_max_value_min_value_l92_92338

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

open Real

theorem smallest_period_pi : ∀ x, f (x + π) = f x := by
  unfold f
  intros
  sorry

theorem max_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 + sqrt 2 := by
  unfold f
  intros
  sorry

theorem min_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 0 := by
  unfold f
  intros
  sorry

end smallest_period_pi_max_value_min_value_l92_92338


namespace daniel_age_is_correct_l92_92926

open Nat

-- Define Uncle Ben's age
def uncleBenAge : ℕ := 50

-- Define Edward's age as two-thirds of Uncle Ben's age
def edwardAge : ℚ := (2 / 3) * uncleBenAge

-- Define that Daniel is 7 years younger than Edward
def danielAge : ℚ := edwardAge - 7

-- Assert that Daniel's age is 79/3 years old
theorem daniel_age_is_correct : danielAge = 79 / 3 := by
  sorry

end daniel_age_is_correct_l92_92926


namespace remainder_of_f_div_x_minus_2_is_48_l92_92728

-- Define the polynomial f(x)
noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 8 * x^3 + 25 * x^2 - 14 * x - 40

-- State the theorem to prove that the remainder of f(x) when divided by x - 2 is 48
theorem remainder_of_f_div_x_minus_2_is_48 : f 2 = 48 :=
by sorry

end remainder_of_f_div_x_minus_2_is_48_l92_92728


namespace probability_two_dice_same_l92_92559

def fair_dice_probability (dice : ℕ) (sides : ℕ) : ℚ :=
  1 - ((sides.factorial / (sides - dice).factorial) / sides^dice)

theorem probability_two_dice_same (dice : ℕ) (sides : ℕ) (h1 : dice = 5) (h2 : sides = 10) :
  fair_dice_probability dice sides = 1744 / 2500 := by
  sorry

end probability_two_dice_same_l92_92559


namespace relationship_y1_y2_y3_l92_92184

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l92_92184


namespace rectangular_coordinates_from_polar_l92_92310

theorem rectangular_coordinates_from_polar (x y r θ : ℝ) (h1 : r * Real.cos θ = x) (h2 : r * Real.sin θ = y) :
    r = 10 ∧ θ = Real.arctan (6 / 8) ∧ (2 * r, 3 * θ) = (20, 3 * Real.arctan (6 / 8)) →
    (20 * Real.cos (3 * Real.arctan (6 / 8)), 20 * Real.sin (3 * Real.arctan (6 / 8))) = (-7.04, 18.72) :=
by
  intros
  -- We need to prove that the statement holds
  sorry

end rectangular_coordinates_from_polar_l92_92310


namespace parabola_equation_l92_92460

theorem parabola_equation (h1: ∃ k, ∀ x y : ℝ, (x, y) = (4, -2) → y^2 = k * x) 
                          (h2: ∃ m, ∀ x y : ℝ, (x, y) = (4, -2) → x^2 = -2 * m * y) :
                          (y : ℝ)^2 = x ∨ (x : ℝ)^2 = -8 * y :=
by 
  sorry

end parabola_equation_l92_92460


namespace number_of_solutions_l92_92333

def f (x : ℝ) : ℝ := |1 - 2 * x|

theorem number_of_solutions :
  (∃ n : ℕ, n = 8 ∧ ∀ x ∈ [0,1], f (f (f x)) = (1 / 2) * x) :=
sorry

end number_of_solutions_l92_92333


namespace passengers_on_third_plane_l92_92731

theorem passengers_on_third_plane (
  P : ℕ
) (h1 : 600 - 2 * 50 = 500) -- Speed of the first plane
  (h2 : 600 - 2 * 60 = 480) -- Speed of the second plane
  (h_avg : (500 + 480 + (600 - 2 * P)) / 3 = 500) -- Average speed condition
  : P = 40 := by sorry

end passengers_on_third_plane_l92_92731


namespace find_m_l92_92706

theorem find_m (m : ℕ) (hm_pos : m > 0)
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : m = 60 := sorry

end find_m_l92_92706


namespace complement_of_M_with_respect_to_U_l92_92207

namespace Complements

open Set

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_with_respect_to_U :
  U \ M = {5, -6} :=
by
  sorry

end Complements

end complement_of_M_with_respect_to_U_l92_92207


namespace evaluate_exponents_l92_92825

theorem evaluate_exponents :
  (5 ^ 0.4) * (5 ^ 0.6) * (5 ^ 0.2) * (5 ^ 0.3) * (5 ^ 0.5) = 25 := 
by
  sorry

end evaluate_exponents_l92_92825


namespace symmetric_point_xOz_l92_92393

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l92_92393


namespace percent_increase_sales_l92_92591

-- Define constants for sales
def sales_last_year : ℕ := 320
def sales_this_year : ℕ := 480

-- Define the percent increase formula
def percent_increase (old_value new_value : ℕ) : ℚ :=
  ((new_value - old_value) / old_value) * 100

-- Prove the percent increase from last year to this year is 50%
theorem percent_increase_sales : percent_increase sales_last_year sales_this_year = 50 := by
  sorry

end percent_increase_sales_l92_92591


namespace line_equation_l92_92330

-- Define the structure of a point
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the projection condition
def projection_condition (P : Point) (l : ℤ → ℤ → Prop) : Prop :=
  l P.x P.y ∧ ∀ (Q : Point), l Q.x Q.y → (Q.x ^ 2 + Q.y ^ 2) ≥ (P.x ^ 2 + P.y ^ 2)

-- Define the point P(-2, 1)
def P : Point := ⟨ -2, 1 ⟩

-- Define line l
def line_l (x y : ℤ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem line_equation :
  projection_condition P line_l → ∀ (x y : ℤ), line_l x y ↔ 2 * x - y + 5 = 0 :=
by
  sorry

end line_equation_l92_92330


namespace inequality_implies_double_l92_92349

-- Define the condition
variables {x y : ℝ}

theorem inequality_implies_double (h : x < y) : 2 * x < 2 * y :=
  sorry

end inequality_implies_double_l92_92349


namespace single_elimination_game_count_l92_92332

theorem single_elimination_game_count (n : Nat) (h : n = 23) : n - 1 = 22 :=
by
  sorry

end single_elimination_game_count_l92_92332


namespace modulus_of_complex_l92_92524

open Complex

theorem modulus_of_complex : ∀ (z : ℂ), z = 3 - 2 * I → Complex.abs z = Real.sqrt 13 :=
by
  intro z
  intro h
  rw [h]
  simp [Complex.abs]
  sorry

end modulus_of_complex_l92_92524


namespace four_digit_multiples_of_5_count_l92_92341

-- Define the range of four-digit numbers
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a multiple of 5
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

-- Define the required proof problem
theorem four_digit_multiples_of_5_count : 
  ∃ n : ℕ, (n = 1800) ∧ (∀ k : ℕ, is_four_digit k ∧ is_multiple_of_five k → n = 1800) :=
sorry

end four_digit_multiples_of_5_count_l92_92341


namespace sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l92_92780

theorem sum_of_consecutive_integers_a_lt_sqrt3_lt_b 
  (a b : ℤ) (h1 : a < b) (h2 : ∀ x : ℤ, x ≤ a → x < b) (h3 : a < Real.sqrt 3) (h4 : Real.sqrt 3 < b) : 
  a + b = 3 :=
by
  sorry

end sum_of_consecutive_integers_a_lt_sqrt3_lt_b_l92_92780


namespace dot_product_parallel_vectors_l92_92963

variable (x : ℝ)
def a : ℝ × ℝ := (x, x - 1)
def b : ℝ × ℝ := (1, 2)
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem dot_product_parallel_vectors
  (h_parallel : are_parallel (a x) b)
  (h_x : x = -1) :
  (a x).1 * (b).1 + (a x).2 * (b).2 = -5 :=
by
  sorry

end dot_product_parallel_vectors_l92_92963


namespace unique_valid_configuration_l92_92039

-- Define the conditions: a rectangular array of chairs organized in rows and columns such that
-- each row contains the same number of chairs as every other row, each column contains the
-- same number of chairs as every other column, with at least two chairs in every row and column.
def valid_array_configuration (rows cols : ℕ) : Prop :=
  2 ≤ rows ∧ 2 ≤ cols ∧ rows * cols = 49

-- The theorem statement: determine how many valid arrays are possible given the conditions.
theorem unique_valid_configuration : ∃! (rows cols : ℕ), valid_array_configuration rows cols :=
sorry

end unique_valid_configuration_l92_92039


namespace remainder_of_concatenated_number_l92_92578

def concatenated_number : ℕ :=
  -- Definition of the concatenated number
  -- That is 123456789101112...4344
  -- For simplicity, we'll just assign it directly
  1234567891011121314151617181920212223242526272829303132333435363738394041424344

theorem remainder_of_concatenated_number :
  concatenated_number % 45 = 9 :=
sorry

end remainder_of_concatenated_number_l92_92578


namespace magic_shop_purchase_l92_92174

theorem magic_shop_purchase :
  let deck_price := 7
  let frank_decks := 3
  let friend_decks := 2
  let discount_rate := 0.1
  let tax_rate := 0.05
  let total_cost := (frank_decks + friend_decks) * deck_price
  let discount := discount_rate * total_cost
  let discounted_total := total_cost - discount
  let sales_tax := tax_rate * discounted_total
  let rounded_sales_tax := (sales_tax * 100).round / 100
  let final_amount := discounted_total + rounded_sales_tax
  final_amount = 33.08 :=
by
  sorry

end magic_shop_purchase_l92_92174


namespace A_completes_work_in_18_days_l92_92820

-- Define the conditions
def efficiency_A_twice_B (A B : ℕ → ℕ) : Prop := ∀ w, A w = 2 * B w
def same_work_time (A B C D : ℕ → ℕ) : Prop := 
  ∀ w t, A w + B w = C w + D w ∧ C t = 1 / 20 ∧ D t = 1 / 30

-- Define the key quantity to be proven
theorem A_completes_work_in_18_days (A B C D : ℕ → ℕ) 
  (h1 : efficiency_A_twice_B A B) 
  (h2 : same_work_time A B C D) : 
  ∀ w, A w = 1 / 18 :=
sorry

end A_completes_work_in_18_days_l92_92820


namespace BobsFruitDrinkCost_l92_92876

theorem BobsFruitDrinkCost 
  (AndySpent : ℕ)
  (BobSpent : ℕ)
  (AndySodaCost : ℕ)
  (AndyHamburgerCost : ℕ)
  (BobSandwichCost : ℕ)
  (FruitDrinkCost : ℕ) :
  AndySpent = 5 ∧ AndySodaCost = 1 ∧ AndyHamburgerCost = 2 ∧ 
  AndySpent = BobSpent ∧ 
  BobSandwichCost = 3 ∧ 
  FruitDrinkCost = BobSpent - BobSandwichCost →
  FruitDrinkCost = 2 := by
  sorry

end BobsFruitDrinkCost_l92_92876


namespace solutions_count_l92_92545

noncomputable def number_of_solutions (x y z : ℚ) : ℕ :=
if (x^2 - y * z = 1) ∧ (y^2 - x * z = 1) ∧ (z^2 - x * y = 1)
then 6
else 0

theorem solutions_count : number_of_solutions x y z = 6 :=
sorry

end solutions_count_l92_92545


namespace fish_caught_in_second_catch_l92_92217

theorem fish_caught_in_second_catch
  (tagged_fish_released : Int)
  (tagged_fish_in_second_catch : Int)
  (total_fish_in_pond : Int)
  (C : Int)
  (h_tagged_fish_count : tagged_fish_released = 60)
  (h_tagged_in_second_catch : tagged_fish_in_second_catch = 2)
  (h_total_fish : total_fish_in_pond = 1800) :
  C = 60 :=
by
  sorry

end fish_caught_in_second_catch_l92_92217


namespace train_speed_l92_92594

theorem train_speed :
  let train_length := 200 -- in meters
  let platform_length := 175.03 -- in meters
  let time_taken := 25 -- in seconds
  let total_distance := train_length + platform_length -- total distance in meters
  let speed_mps := total_distance / time_taken -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed to kilometers per hour
  speed_kmph = 54.00432 := sorry

end train_speed_l92_92594


namespace percentage_calculation_l92_92069

/-- If x % of 375 equals 5.4375, then x % equals 1.45 %. -/
theorem percentage_calculation (x : ℝ) (h : x / 100 * 375 = 5.4375) : x = 1.45 := 
sorry

end percentage_calculation_l92_92069


namespace pencils_per_box_l92_92414

theorem pencils_per_box (total_pencils : ℝ) (num_boxes : ℝ) (pencils_per_box : ℝ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4.0) 
  (h3 : pencils_per_box = total_pencils / num_boxes) : 
  pencils_per_box = 648 :=
by
  sorry

end pencils_per_box_l92_92414


namespace length_of_each_train_l92_92880

noncomputable def length_of_train : ℝ := 
  let speed_fast := 46 -- in km/hr
  let speed_slow := 36 -- in km/hr
  let relative_speed := speed_fast - speed_slow -- 10 km/hr
  let relative_speed_km_per_sec := relative_speed / 3600.0 -- converting to km/sec
  let time_sec := 18.0 -- time in seconds
  let distance_km := relative_speed_km_per_sec * time_sec -- calculates distance in km
  distance_km * 1000.0 -- converts to meters

theorem length_of_each_train : length_of_train = 50 :=
  by
    sorry

end length_of_each_train_l92_92880


namespace average_speed_correct_l92_92593

noncomputable def average_speed (d v_up v_down : ℝ) : ℝ :=
  let t_up := d / v_up
  let t_down := d / v_down
  let total_distance := 2 * d
  let total_time := t_up + t_down
  total_distance / total_time

theorem average_speed_correct :
  average_speed 0.2 24 36 = 28.8 := by {
  sorry
}

end average_speed_correct_l92_92593


namespace height_of_fifth_tree_l92_92159

theorem height_of_fifth_tree 
  (h₁ : tallest_tree = 108) 
  (h₂ : second_tallest_tree = 54 - 6) 
  (h₃ : third_tallest_tree = second_tallest_tree / 4) 
  (h₄ : fourth_shortest_tree = (second_tallest_tree + third_tallest_tree) - 2) 
  (h₅ : fifth_tree = 0.75 * (tallest_tree + second_tallest_tree + third_tallest_tree + fourth_shortest_tree)) : 
  fifth_tree = 169.5 :=
by
  sorry

end height_of_fifth_tree_l92_92159


namespace locus_of_right_angle_vertex_l92_92815

variables {x y : ℝ}

/-- Given points M(-2,0) and N(2,0), if P(x,y) is the right-angled vertex of
  a right-angled triangle with MN as its hypotenuse, then the locus equation
  of P is given by x^2 + y^2 = 4 with the condition x ≠ ±2. -/
theorem locus_of_right_angle_vertex (h : x ≠ 2 ∧ x ≠ -2) :
  x^2 + y^2 = 4 :=
sorry

end locus_of_right_angle_vertex_l92_92815


namespace tangent_parallel_x_axis_coordinates_l92_92678

theorem tangent_parallel_x_axis_coordinates :
  ∃ (x y : ℝ), (y = x^2 - 3 * x) ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) :=
by
  use (3 / 2)
  use (-9 / 4)
  sorry

end tangent_parallel_x_axis_coordinates_l92_92678


namespace slower_speed_l92_92204

theorem slower_speed (x : ℝ) :
  (5 * (24 / x) = 24 + 6) → x = 4 := 
by
  intro h
  sorry

end slower_speed_l92_92204


namespace correct_percentage_is_500_over_7_l92_92856

-- Given conditions
variable (x : ℕ)
def total_questions : ℕ := 7 * x
def missed_questions : ℕ := 2 * x

-- Definition of the fraction and percentage calculation
def correct_fraction : ℚ := (total_questions x - missed_questions x : ℕ) / total_questions x
def correct_percentage : ℚ := correct_fraction x * 100

-- The theorem to prove
theorem correct_percentage_is_500_over_7 : correct_percentage x = 500 / 7 :=
by
  -- Proof goes here
  sorry

end correct_percentage_is_500_over_7_l92_92856


namespace carpenter_needs_80_woodblocks_l92_92620

-- Define the number of logs the carpenter currently has
def existing_logs : ℕ := 8

-- Define the number of woodblocks each log can produce
def woodblocks_per_log : ℕ := 5

-- Define the number of additional logs needed
def additional_logs : ℕ := 8

-- Calculate the total number of woodblocks needed
def total_woodblocks_needed : ℕ := 
  (existing_logs * woodblocks_per_log) + (additional_logs * woodblocks_per_log)

-- Prove that the total number of woodblocks needed is 80
theorem carpenter_needs_80_woodblocks : total_woodblocks_needed = 80 := by
  sorry

end carpenter_needs_80_woodblocks_l92_92620


namespace tiles_needed_l92_92646

theorem tiles_needed (A_classroom : ℝ) (side_length_tile : ℝ) (H_classroom : A_classroom = 56) (H_side_length : side_length_tile = 0.4) :
  A_classroom / (side_length_tile * side_length_tile) = 350 :=
by
  sorry

end tiles_needed_l92_92646


namespace units_digit_of_subtraction_is_seven_l92_92657

theorem units_digit_of_subtraction_is_seven (a b c: ℕ) (h1: a = c + 3) (h2: b = 2 * c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  result % 10 = 7 :=
by
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  sorry

end units_digit_of_subtraction_is_seven_l92_92657


namespace relationship_between_y_values_l92_92182

-- Define the quadratic function given the constraints
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + abs b * x + c

-- Define the points (x1, y1), (x2, y2), (x3, y3)
def x1 := -14 / 3
def x2 := 5 / 2
def x3 := 3

def y1 (a b c : ℝ) : ℝ := quadratic_function a b c x1
def y2 (a b c : ℝ) : ℝ := quadratic_function a b c x2
def y3 (a b c : ℝ) : ℝ := quadratic_function a b c x3

theorem relationship_between_y_values 
  (a b c : ℝ) (h1 : - (abs b) / (2 * a) = -1) 
  (y1_value : ℝ := y1 a b c) 
  (y2_value : ℝ := y2 a b c) 
  (y3_value : ℝ := y3 a b c) : 
  y2_value < y1_value ∧ y1_value < y3_value := 
by 
  sorry

end relationship_between_y_values_l92_92182


namespace minimum_work_to_remove_cube_l92_92202

namespace CubeBuoyancy

def edge_length (ℓ : ℝ) := ℓ = 0.30 -- in meters
def wood_density (ρ : ℝ) := ρ = 750  -- in kg/m^3
def water_density (ρ₀ : ℝ) := ρ₀ = 1000 -- in kg/m^3

theorem minimum_work_to_remove_cube 
  {ℓ ρ ρ₀ : ℝ} 
  (h₁ : edge_length ℓ)
  (h₂ : wood_density ρ)
  (h₃ : water_density ρ₀) : 
  ∃ W : ℝ, W = 22.8 := 
sorry

end CubeBuoyancy

end minimum_work_to_remove_cube_l92_92202


namespace tank_inflow_rate_l92_92015

/-- 
  Tanks A and B have the same capacity of 20 liters. Tank A has
  an inflow rate of 2 liters per hour and takes 5 hours longer to
  fill than tank B. Show that the inflow rate in tank B is 4 liters 
  per hour.
-/
theorem tank_inflow_rate (capacity : ℕ) (rate_A : ℕ) (extra_time : ℕ) (rate_B : ℕ) 
  (h1 : capacity = 20) (h2 : rate_A = 2) (h3 : extra_time = 5) (h4 : capacity / rate_A = (capacity / rate_B) + extra_time) :
  rate_B = 4 :=
sorry

end tank_inflow_rate_l92_92015


namespace number_of_ants_proof_l92_92078

-- Define the conditions
def width_ft := 500
def length_ft := 600
def ants_per_sq_inch := 4
def inches_per_foot := 12

-- Define the calculation to get the number of ants
def number_of_ants (width_ft : ℕ) (length_ft : ℕ) (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :=
  let width_inch := width_ft * inches_per_foot
  let length_inch := length_ft * inches_per_foot
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

-- Prove the number of ants is approximately 173 million
theorem number_of_ants_proof :
  number_of_ants width_ft length_ft ants_per_sq_inch inches_per_foot = 172800000 :=
by
  sorry

end number_of_ants_proof_l92_92078


namespace unequal_numbers_l92_92912

theorem unequal_numbers {k : ℚ} (h : 3 * (1 : ℚ) + 7 * (1 : ℚ) + 2 * k = 0) (d : (7^2 : ℚ) - 4 * 3 * 2 * k = 0) : 
    (3 : ℚ) ≠ (7 : ℚ) ∧ (3 : ℚ) ≠ k ∧ (7 : ℚ) ≠ k :=
by
  -- adding sorry for skipping proof
  sorry

end unequal_numbers_l92_92912


namespace find_S20_l92_92090

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def S : ℕ → ℝ := sorry

axiom a_nonzero (n : ℕ) : a_seq n ≠ 0
axiom a1_eq : a_seq 1 = 1
axiom Sn_eq (n : ℕ) : S n = (a_seq n * a_seq (n + 1)) / 2

theorem find_S20 : S 20 = 210 := sorry

end find_S20_l92_92090


namespace infinite_series_evaluates_to_12_l92_92370

noncomputable def infinite_series : ℝ :=
  ∑' k, (k^3) / (3^k)

theorem infinite_series_evaluates_to_12 :
  infinite_series = 12 :=
by
  sorry

end infinite_series_evaluates_to_12_l92_92370


namespace find_s_l92_92778

theorem find_s (s t : ℚ) (h1 : 8 * s + 6 * t = 120) (h2 : s = t - 3) : s = 51 / 7 := by
  sorry

end find_s_l92_92778


namespace right_angle_triangle_probability_l92_92401

def vertex_count : ℕ := 16
def ways_to_choose_3_points : ℕ := Nat.choose vertex_count 3
def number_of_rectangles : ℕ := 36
def right_angle_triangles_per_rectangle : ℕ := 4
def total_right_angle_triangles : ℕ := number_of_rectangles * right_angle_triangles_per_rectangle
def probability_right_angle_triangle : ℚ := total_right_angle_triangles / ways_to_choose_3_points

theorem right_angle_triangle_probability :
  probability_right_angle_triangle = (9 / 35 : ℚ) := by
  sorry

end right_angle_triangle_probability_l92_92401


namespace well_depth_l92_92249

theorem well_depth :
  (∃ t₁ t₂ : ℝ, t₁ + t₂ = 9.5 ∧ 20 * t₁ ^ 2 = d ∧ t₂ = d / 1000 ∧ d = 1332.25) :=
by
  sorry

end well_depth_l92_92249


namespace compare_cubics_l92_92417

variable {a b : ℝ}

theorem compare_cubics (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end compare_cubics_l92_92417


namespace f_2_value_l92_92886

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_2_value :
  (f a b (-2)) = 2 → (f a b 2) = -10 :=
by
  intro h
  -- Provide the solution steps here, starting with simplifying the equation. Sorry for now
  sorry

end f_2_value_l92_92886


namespace large_circuit_longer_l92_92361

theorem large_circuit_longer :
  ∀ (small_circuit_length large_circuit_length : ℕ),
  ∀ (laps_jana laps_father : ℕ),
  laps_jana = 3 →
  laps_father = 4 →
  (laps_father * large_circuit_length = 2 * (laps_jana * small_circuit_length)) →
  small_circuit_length = 400 →
  large_circuit_length - small_circuit_length = 200 :=
by
  intros small_circuit_length large_circuit_length laps_jana laps_father
  intros h_jana_laps h_father_laps h_distance h_small_length
  sorry

end large_circuit_longer_l92_92361


namespace divisors_form_60k_l92_92095

-- Define the conditions in Lean
def is_positive_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def satisfies_conditions (n a b c : ℕ) : Prop :=
  is_positive_divisor n a ∧
  is_positive_divisor n b ∧
  is_positive_divisor n c ∧
  a > b ∧ b > c ∧
  is_positive_divisor n (a^2 - b^2) ∧
  is_positive_divisor n (b^2 - c^2) ∧
  is_positive_divisor n (a^2 - c^2)

-- State the theorem to be proven in Lean
theorem divisors_form_60k (n : ℕ) (a b c : ℕ) (h1 : satisfies_conditions n a b c) : 
  ∃ k : ℕ, n = 60 * k :=
sorry

end divisors_form_60k_l92_92095


namespace estimate_total_balls_l92_92553

theorem estimate_total_balls (red_balls : ℕ) (frequency : ℝ) (total_balls : ℕ) 
  (h_red : red_balls = 12) (h_freq : frequency = 0.6) 
  (h_eq : (red_balls : ℝ) / total_balls = frequency) : 
  total_balls = 20 :=
by
  sorry

end estimate_total_balls_l92_92553


namespace find_a_l92_92024

theorem find_a 
  (x y z a : ℤ)
  (h1 : z + a = -2)
  (h2 : y + z = 1)
  (h3 : x + y = 0) : 
  a = -2 := 
  by 
    sorry

end find_a_l92_92024


namespace tangent_line_at_x_2_increasing_on_1_to_infinity_l92_92351

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Subpart I
theorem tangent_line_at_x_2 (a b : ℝ) :
  (a / 2 + 2 = 1) ∧ (2 + a * Real.log 2 = 2 + b) → (a = -2 ∧ b = -2 * Real.log 2) :=
by
  sorry

-- Subpart II
theorem increasing_on_1_to_infinity (a : ℝ) :
  (∀ x > 1, (x + a / x) ≥ 0) → (a ≥ -1) :=
by
  sorry

end tangent_line_at_x_2_increasing_on_1_to_infinity_l92_92351


namespace find_ordered_pair_l92_92517

theorem find_ordered_pair :
  ∃ x y : ℚ, 
  (x + 2 * y = (7 - x) + (7 - 2 * y)) ∧
  (3 * x - 2 * y = (x + 2) - (2 * y + 2)) ∧
  x = 0 ∧ 
  y = 7 / 2 :=
by
  sorry

end find_ordered_pair_l92_92517


namespace more_trees_in_ahmeds_orchard_l92_92897

-- Given conditions
def ahmed_orange_trees : ℕ := 8
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees
def ahmed_total_trees : ℕ := ahmed_orange_trees + ahmed_apple_trees
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Statement to be proven
theorem more_trees_in_ahmeds_orchard : ahmed_total_trees - hassan_total_trees = 9 :=
by
  sorry

end more_trees_in_ahmeds_orchard_l92_92897


namespace black_to_brown_ratio_l92_92735

-- Definitions of the given conditions
def total_shoes : ℕ := 66
def brown_shoes : ℕ := 22
def black_shoes : ℕ := total_shoes - brown_shoes

-- Lean 4 problem statement: Prove the ratio of black shoes to brown shoes is 2:1
theorem black_to_brown_ratio :
  (black_shoes / Nat.gcd black_shoes brown_shoes) = 2 ∧ (brown_shoes / Nat.gcd black_shoes brown_shoes) = 1 := by
sorry

end black_to_brown_ratio_l92_92735


namespace more_white_birds_than_grey_l92_92474

def num_grey_birds_in_cage : ℕ := 40
def num_remaining_birds : ℕ := 66

def num_grey_birds_freed : ℕ := num_grey_birds_in_cage / 2
def num_grey_birds_left_in_cage : ℕ := num_grey_birds_in_cage - num_grey_birds_freed
def num_white_birds : ℕ := num_remaining_birds - num_grey_birds_left_in_cage

theorem more_white_birds_than_grey : num_white_birds - num_grey_birds_in_cage = 6 := by
  sorry

end more_white_birds_than_grey_l92_92474


namespace projection_correct_l92_92424

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := ⟨-1, 3, -4⟩

def projection_yOz_plane (P : Point3D) : Point3D :=
  ⟨0, P.y, P.z⟩

theorem projection_correct :
  projection_yOz_plane P = ⟨0, 3, -4⟩ :=
by
  -- The theorem proof is omitted.
  sorry

end projection_correct_l92_92424


namespace simple_interest_rate_l92_92769

theorem simple_interest_rate (P A : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ) :
  P = 800 → A = 950 → T = 5 → SI = A - P → SI = (P * R * T) / 100 → R = 3.75 :=
  by
  intros hP hA hT hSI h_formula
  sorry

end simple_interest_rate_l92_92769


namespace gcd_840_1764_l92_92889

def a : ℕ := 840
def b : ℕ := 1764

theorem gcd_840_1764 : Nat.gcd a b = 84 := by
  -- Proof omitted
  sorry

end gcd_840_1764_l92_92889


namespace mushrooms_collected_l92_92120

theorem mushrooms_collected (x1 x2 x3 x4 : ℕ) 
  (h1 : x1 + x2 = 7) 
  (h2 : x1 + x3 = 9)
  (h3 : x2 + x3 = 10) : x1 = 3 ∧ x2 = 4 ∧ x3 = 6 ∧ x4 = 7 :=
by
  sorry

end mushrooms_collected_l92_92120


namespace arithmetic_sequence_first_term_l92_92676

theorem arithmetic_sequence_first_term (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 500) 
  (h2 : 30 * (2 * a + 179 * d) = 2900) : 
  a = -34 / 3 := 
sorry

end arithmetic_sequence_first_term_l92_92676


namespace factory_sample_capacity_l92_92400

theorem factory_sample_capacity (n : ℕ) (a_ratio b_ratio c_ratio : ℕ) 
  (total_ratio : a_ratio + b_ratio + c_ratio = 10) (a_sample : ℕ)
  (h : a_sample = 16) (h_ratio : a_ratio = 2) :
  n = 80 :=
by
  -- sample calculations proof would normally be here
  sorry

end factory_sample_capacity_l92_92400


namespace initial_fish_count_l92_92075

-- Definitions based on the given conditions
def Fish_given : ℝ := 22.0
def Fish_now : ℝ := 25.0

-- The goal is to prove the initial number of fish Mrs. Sheridan had.
theorem initial_fish_count : (Fish_given + Fish_now) = 47.0 := by
  sorry

end initial_fish_count_l92_92075


namespace car_speed_problem_l92_92190

theorem car_speed_problem (S1 S2 : ℝ) (T : ℝ) (avg_speed : ℝ) (H1 : S1 = 70) (H2 : T = 2) (H3 : avg_speed = 80) :
  S2 = 90 :=
by
  have avg_speed_eq : avg_speed = (S1 + S2) / T := sorry
  have h : S2 = 90 := sorry
  exact h

end car_speed_problem_l92_92190


namespace solve_for_x_l92_92081

theorem solve_for_x : ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by
  sorry

end solve_for_x_l92_92081


namespace multiply_24_99_l92_92868

theorem multiply_24_99 : 24 * 99 = 2376 :=
by
  sorry

end multiply_24_99_l92_92868


namespace lines_parallel_l92_92785

theorem lines_parallel :
  ∀ (x y : ℝ), (x - y + 2 = 0) ∧ (x - y + 1 = 0) → False :=
by
  intros x y h
  sorry

end lines_parallel_l92_92785


namespace find_x_given_conditions_l92_92821

variable (x y z : ℝ)

theorem find_x_given_conditions
  (h1: x * y / (x + y) = 4)
  (h2: x * z / (x + z) = 9)
  (h3: y * z / (y + z) = 16)
  (h_pos: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_distinct: x ≠ y ∧ x ≠ z ∧ y ≠ z) :
  x = 384/21 :=
sorry

end find_x_given_conditions_l92_92821


namespace no_such_function_exists_l92_92375

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f^[n] n = n + 1 :=
by
  sorry

end no_such_function_exists_l92_92375


namespace smallest_m_l92_92073

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  abs (3 * fractional_part x - 1.5)

theorem smallest_m (m : ℤ) (h1 : ∀ x : ℝ, m^2 * f (x * f x) = x → True) : ∃ m, m = 8 :=
by
  have h2 : ∀ m : ℤ, (∃ (s : ℕ), s ≥ 1008 ∧ (m^2 * abs (3 * fractional_part (s * abs (1.5 - 3 * (fractional_part s) )) - 1.5) = s)) → m = 8
  {
    sorry
  }
  sorry

end smallest_m_l92_92073


namespace students_neither_play_l92_92784

theorem students_neither_play (total_students football_players tennis_players both_players neither_players : ℕ)
  (h1 : total_students = 40)
  (h2 : football_players = 26)
  (h3 : tennis_players = 20)
  (h4 : both_players = 17)
  (h5 : neither_players = total_students - (football_players + tennis_players - both_players)) :
  neither_players = 11 :=
by
  sorry

end students_neither_play_l92_92784


namespace decompose_96_l92_92209

theorem decompose_96 (x y : ℤ) (h1 : x * y = 96) (h2 : x^2 + y^2 = 208) :
  (x = 8 ∧ y = 12) ∨ (x = 12 ∧ y = 8) ∨ (x = -8 ∧ y = -12) ∨ (x = -12 ∧ y = -8) := by
  sorry

end decompose_96_l92_92209


namespace fraction_equality_x_eq_neg1_l92_92806

theorem fraction_equality_x_eq_neg1 (x : ℝ) (h : (5 + x) / (7 + x) = (3 + x) / (4 + x)) : x = -1 := by
  sorry

end fraction_equality_x_eq_neg1_l92_92806


namespace find_common_ratio_l92_92792

-- We need to state that q is the common ratio of the geometric sequence

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first three terms for the geometric sequence
def S_3 (a : ℕ → ℝ) := a 0 + a 1 + a 2

-- State the Lean 4 declaration of the proof problem
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : (S_3 a) / (a 2) = 3) :
  q = 1 := 
sorry

end find_common_ratio_l92_92792


namespace min_value_expr_least_is_nine_l92_92109

noncomputable def minimum_value_expression (a b c d : ℝ) : ℝ :=
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2

theorem min_value_expr_least_is_nine (a b c d : ℝ)
  (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  minimum_value_expression a b c d = 9 := 
sorry

end min_value_expr_least_is_nine_l92_92109


namespace no_perfect_square_in_range_l92_92498

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end no_perfect_square_in_range_l92_92498


namespace additional_profit_is_80000_l92_92455

-- Define the construction cost of a regular house
def construction_cost_regular (C : ℝ) : ℝ := C

-- Define the construction cost of the special house
def construction_cost_special (C : ℝ) : ℝ := C + 200000

-- Define the selling price of a regular house
def selling_price_regular : ℝ := 350000

-- Define the selling price of the special house
def selling_price_special : ℝ := 1.8 * 350000

-- Define the profit from selling a regular house
def profit_regular (C : ℝ) : ℝ := selling_price_regular - (construction_cost_regular C)

-- Define the profit from selling the special house
def profit_special (C : ℝ) : ℝ := selling_price_special - (construction_cost_special C)

-- Define the additional profit made by building and selling the special house compared to a regular house
def additional_profit (C : ℝ) : ℝ := (profit_special C) - (profit_regular C)

-- Theorem to prove the additional profit is $80,000
theorem additional_profit_is_80000 (C : ℝ) : additional_profit C = 80000 :=
sorry

end additional_profit_is_80000_l92_92455


namespace price_of_72_cans_l92_92908

def regular_price_per_can : ℝ := 0.60
def discount_percentage : ℝ := 0.20
def total_price : ℝ := 34.56

theorem price_of_72_cans (discounted_price_per_can : ℝ) (number_of_cans : ℕ)
  (H1 : discounted_price_per_can = regular_price_per_can - (discount_percentage * regular_price_per_can))
  (H2 : number_of_cans = total_price / discounted_price_per_can) :
  total_price = number_of_cans * discounted_price_per_can := by
  sorry

end price_of_72_cans_l92_92908


namespace sam_dad_gave_39_nickels_l92_92284

-- Define the initial conditions
def initial_pennies : ℕ := 49
def initial_nickels : ℕ := 24
def given_quarters : ℕ := 31
def dad_given_nickels : ℕ := 63 - initial_nickels

-- Statement to prove
theorem sam_dad_gave_39_nickels 
    (total_nickels_after : ℕ) 
    (initial_nickels : ℕ) 
    (final_nickels : ℕ := total_nickels_after - initial_nickels) : 
    final_nickels = 39 :=
sorry

end sam_dad_gave_39_nickels_l92_92284


namespace initial_bees_l92_92011

variable (B : ℕ)

theorem initial_bees (h : B + 10 = 26) : B = 16 :=
by sorry

end initial_bees_l92_92011


namespace root_in_interval_iff_a_outside_range_l92_92911

theorem root_in_interval_iff_a_outside_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) 1 ∧ a * x + 1 = 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end root_in_interval_iff_a_outside_range_l92_92911


namespace find_m_l92_92420

open Real

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (sin x ^ 4 + cos x ^ 4) + m * (sin x + cos x) ^ 4

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x m ≤ 5) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x m = 5) :=
sorry

end find_m_l92_92420


namespace simplify_polynomial_l92_92554

theorem simplify_polynomial (x : ℝ) :
  (14 * x ^ 12 + 8 * x ^ 9 + 3 * x ^ 8) + (2 * x ^ 14 - x ^ 12 + 2 * x ^ 9 + 5 * x ^ 5 + 7 * x ^ 2 + 6) =
  2 * x ^ 14 + 13 * x ^ 12 + 10 * x ^ 9 + 3 * x ^ 8 + 5 * x ^ 5 + 7 * x ^ 2 + 6 :=
by
  sorry

end simplify_polynomial_l92_92554


namespace probability_genuine_coins_given_weight_condition_l92_92486

/--
Given the following conditions:
- Ten counterfeit coins of equal weight are mixed with 20 genuine coins.
- The weight of a counterfeit coin is different from the weight of a genuine coin.
- Two pairs of coins are selected randomly without replacement from the 30 coins. 

Prove that the probability that all 4 selected coins are genuine, given that the combined weight
of the first pair is equal to the combined weight of the second pair, is 5440/5481.
-/
theorem probability_genuine_coins_given_weight_condition :
  let num_coins := 30
  let num_genuine := 20
  let num_counterfeit := 10
  let pairs_selected := 2
  let pairs_remaining := num_coins - pairs_selected * 2
  let P := (num_genuine / num_coins) * ((num_genuine - 1) / (num_coins - 1)) * ((num_genuine - 2) / pairs_remaining) * ((num_genuine - 3) / (pairs_remaining - 1))
  let event_A_given_B := P / (7 / 16)
  event_A_given_B = 5440 / 5481 := 
sorry

end probability_genuine_coins_given_weight_condition_l92_92486


namespace rectangle_area_ratio_l92_92392

theorem rectangle_area_ratio (l b : ℕ) (h1 : l = b + 10) (h2 : b = 8) : (l * b) / b = 18 := by
  sorry

end rectangle_area_ratio_l92_92392


namespace circle_radius_5_l92_92612

-- The circle equation given
def circle_eq (x y : ℝ) (c : ℝ) : Prop :=
  x^2 + 4 * x + y^2 + 8 * y + c = 0

-- The radius condition given
def radius_condition : Prop :=
  5 = (25 : ℝ).sqrt

-- The final proof statement
theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, circle_eq x y c) → radius_condition → c = -5 := 
by
  sorry

end circle_radius_5_l92_92612


namespace property_value_at_beginning_l92_92026

theorem property_value_at_beginning 
  (r : ℝ) (v3 : ℝ) (V : ℝ) (rate : ℝ) (years : ℕ) 
  (h_rate : rate = 6.25 / 100) 
  (h_years : years = 3) 
  (h_v3 : v3 = 21093) 
  (h_r : r = 1 - rate) 
  (h_V : V * r ^ years = v3) 
  : V = 25656.25 :=
by
  sorry

end property_value_at_beginning_l92_92026


namespace root_analysis_l92_92445

noncomputable def root1 (a : ℝ) : ℝ :=
2 * a + 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def root2 (a : ℝ) : ℝ :=
2 * a - 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def derivedRoot (a : ℝ) : ℝ :=
(3 * a - 2) / a

theorem root_analysis (a : ℝ) (ha : a > 0) :
( (2/3 ≤ a ∧ a < 1) ∨ (2 < a) → (root1 a ≥ 0 ∧ root2 a ≥ 0)) ∧
( 0 < a ∧ a < 2/3 → (derivedRoot a < 0 ∧ root1 a ≥ 0)) :=
sorry

end root_analysis_l92_92445


namespace solve_quadratic_l92_92685

theorem solve_quadratic (x : ℝ) (h : x^2 - 4 = 0) : x = 2 ∨ x = -2 :=
by sorry

end solve_quadratic_l92_92685


namespace perpendicular_lines_l92_92625

theorem perpendicular_lines (a : ℝ) 
  (h1 : (3 : ℝ) * y + (2 : ℝ) * x - 6 = 0) 
  (h2 : (4 : ℝ) * y + a * x - 5 = 0) : 
  a = -6 :=
sorry

end perpendicular_lines_l92_92625


namespace find_k_value_l92_92210

theorem find_k_value (k : ℝ) : 
  (-x ^ 2 - (k + 11) * x - 8 = -( (x - 2) * (x - 4) ) ) → k = -17 := 
by 
  sorry

end find_k_value_l92_92210


namespace packages_per_truck_l92_92352

theorem packages_per_truck (total_packages : ℕ) (number_of_trucks : ℕ) (h1 : total_packages = 490) (h2 : number_of_trucks = 7) :
  (total_packages / number_of_trucks) = 70 := by
  sorry

end packages_per_truck_l92_92352


namespace odd_function_properties_l92_92161

def f : ℝ → ℝ := sorry

theorem odd_function_properties 
  (H1 : ∀ x, f (-x) = -f x) -- f is odd
  (H2 : ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → f x ≤ f y) -- f is increasing on [1, 3]
  (H3 : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≥ 7) -- f has a minimum value of 7 on [1, 3]
  : (∀ x y, -3 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) -- f is increasing on [-3, -1]
    ∧ (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ -7) -- f has a maximum value of -7 on [-3, -1]
:= sorry

end odd_function_properties_l92_92161


namespace perpendicular_condition_parallel_condition_opposite_direction_l92_92506

/-- Conditions definitions --/
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)

def k_vector_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
def vector_a_minus_3b : ℝ × ℝ := (10, -4)

/-- Problem 1: Prove the perpendicular condition --/
theorem perpendicular_condition (k : ℝ) : (k_vector_a_plus_b k).fst * vector_a_minus_3b.fst + (k_vector_a_plus_b k).snd * vector_a_minus_3b.snd = 0 → k = 19 :=
by
  sorry

/-- Problem 2: Prove the parallel condition --/
theorem parallel_condition (k : ℝ) : (-(k - 3) / 10 = (2 * k + 2) / (-4)) → k = -1/3 :=
by
  sorry

/-- Determine if the vectors are in opposite directions --/
theorem opposite_direction (k : ℝ) (hk : k = -1/3) : k_vector_a_plus_b k = (-(1/3):ℝ) • vector_a_minus_3b :=
by
  sorry

end perpendicular_condition_parallel_condition_opposite_direction_l92_92506


namespace calc_expression_l92_92468

def r (θ : ℚ) : ℚ := 1 / (1 + θ)
def s (θ : ℚ) : ℚ := θ + 1

theorem calc_expression : s (r (s (r (s (r 2))))) = 24 / 17 :=
by 
  sorry

end calc_expression_l92_92468


namespace inequality_a5_b5_c5_l92_92590

theorem inequality_a5_b5_c5 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^5 + b^5 + c^5 ≥ a^3 * b * c + a * b^3 * c + a * b * c^3 :=
by
  sorry

end inequality_a5_b5_c5_l92_92590


namespace power_of_same_base_power_of_different_base_l92_92164

theorem power_of_same_base (a n : ℕ) (h : ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m) :
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ a^n = (a^k)^m :=
  sorry

theorem power_of_different_base (a n : ℕ) : ∃ (b m : ℕ), a^n = b^m :=
  sorry

end power_of_same_base_power_of_different_base_l92_92164


namespace tom_purchased_8_kg_of_apples_l92_92609

noncomputable def number_of_apples_purchased (price_per_kg_apple : ℤ) (price_per_kg_mango : ℤ) (kg_mangoes : ℤ) (total_paid : ℤ) : ℤ :=
  let total_cost_mangoes := price_per_kg_mango * kg_mangoes
  total_paid - total_cost_mangoes / price_per_kg_apple

theorem tom_purchased_8_kg_of_apples : 
  number_of_apples_purchased 70 65 9 1145 = 8 := 
by {
  -- Expand the definitions and simplify
  sorry
}

end tom_purchased_8_kg_of_apples_l92_92609


namespace ball_height_intersect_l92_92568

noncomputable def ball_height (h : ℝ) (t₁ t₂ : ℝ) (h₁ h₂ : ℝ → ℝ) : Prop :=
  (∀ t, h₁ t = h₂ (t - 1) ↔ t = t₁) ∧
  (h₁ t₁ = h ∧ h₂ t₁ = h) ∧ 
  (∀ t, h₂ (t - 1) = h₁ t) ∧ 
  (h₁ (1.1) = h ∧ h₂ (1.1) = h)

theorem ball_height_intersect (h : ℝ)
  (h₁ h₂ : ℝ → ℝ)
  (h_max : ∀ t₁ t₂, ball_height h t₁ t₂ h₁ h₂) :
  (∃ t₁, t₁ = 1.6) :=
sorry

end ball_height_intersect_l92_92568


namespace exists_mn_coprime_l92_92958

theorem exists_mn_coprime (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gcd : Int.gcd a b = 1) :
  ∃ (m n : ℕ), 1 ≤ m ∧ 1 ≤ n ∧ (a^m + b^n) % (a * b) = 1 % (a * b) :=
sorry

end exists_mn_coprime_l92_92958


namespace find_x_from_exponential_eq_l92_92839

theorem find_x_from_exponential_eq (x : ℕ) (h : 3^x + 3^x + 3^x + 3^x = 6561) : x = 6 := 
sorry

end find_x_from_exponential_eq_l92_92839


namespace avg_height_first_30_girls_l92_92103

theorem avg_height_first_30_girls (H : ℝ)
  (h1 : ∀ x : ℝ, 30 * x + 10 * 156 = 40 * 159) :
  H = 160 :=
by sorry

end avg_height_first_30_girls_l92_92103


namespace weight_of_B_l92_92661

-- Definitions for the weights of A, B, and C
variable (A B C : ℝ)

-- Conditions given in the problem
def condition1 := (A + B + C) / 3 = 45
def condition2 := (A + B) / 2 = 40
def condition3 := (B + C) / 2 = 43

-- The theorem to prove that B = 31 under the given conditions
theorem weight_of_B : condition1 A B C → condition2 A B → condition3 B C → B = 31 := by
  intros
  sorry

end weight_of_B_l92_92661


namespace fraction_stamp_collection_l92_92565

theorem fraction_stamp_collection (sold_amount total_value : ℝ) (sold_for : sold_amount = 28) (total : total_value = 49) : sold_amount / total_value = 4 / 7 :=
by
  sorry

end fraction_stamp_collection_l92_92565


namespace sequence_a7_l92_92917

/-- 
  Given a sequence {a_n} such that a_1 + a_{2n-1} = 4n - 6, 
  prove that a_7 = 11 
-/
theorem sequence_a7 (a : ℕ → ℤ)
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : a 7 = 11 :=
by
  sorry

end sequence_a7_l92_92917


namespace trigonometric_identity_l92_92294

theorem trigonometric_identity (x : ℝ) : 
  x = Real.pi / 4 → (1 + Real.sin (x + Real.pi / 4) - Real.cos (x + Real.pi / 4)) / 
                          (1 + Real.sin (x + Real.pi / 4) + Real.cos (x + Real.pi / 4)) = 1 :=
by 
  sorry

end trigonometric_identity_l92_92294


namespace find_sale_in_third_month_l92_92050

def sale_in_first_month := 5700
def sale_in_second_month := 8550
def sale_in_fourth_month := 3850
def sale_in_fifth_month := 14045
def average_sale := 7800
def num_months := 5
def total_sales := average_sale * num_months

theorem find_sale_in_third_month (X : ℕ) 
  (H : total_sales = sale_in_first_month + sale_in_second_month + X + sale_in_fourth_month + sale_in_fifth_month) :
  X = 9455 :=
by
  sorry

end find_sale_in_third_month_l92_92050


namespace initial_investment_calculation_l92_92721

-- Define the conditions
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 2
def A : ℝ := 6050.000000000001
def one : ℝ := 1

-- The goal is to prove that the initial principal P is 5000 under these conditions
theorem initial_investment_calculation (P : ℝ) : P = 5000 :=
by
  have interest_compounded : ℝ := (one + r / n) ^ (n * t)
  have total_amount : ℝ := P * interest_compounded
  sorry

end initial_investment_calculation_l92_92721


namespace C_neither_necessary_nor_sufficient_for_A_l92_92419

theorem C_neither_necessary_nor_sufficient_for_A 
  (A B C : Prop) 
  (h1 : B → C)
  (h2 : B → A) : 
  ¬(A → C) ∧ ¬(C → A) :=
by
  sorry

end C_neither_necessary_nor_sufficient_for_A_l92_92419


namespace simplify_expression_l92_92192

variable (x : ℝ)

theorem simplify_expression : 2 * (1 - (2 * (1 - (1 + (2 - (3 * x)))))) = -10 + 12 * x := 
  sorry

end simplify_expression_l92_92192


namespace sum_series_75_to_99_l92_92888

theorem sum_series_75_to_99 : 
  let a := 75
  let l := 99
  let n := l - a + 1
  let s := n * (a + l) / 2
  s = 2175 :=
by
  sorry

end sum_series_75_to_99_l92_92888


namespace ball_hits_ground_at_10_over_7_l92_92326

def ball_hits_ground (t : ℚ) : Prop :=
  -4.9 * t^2 + 3.5 * t + 5 = 0

theorem ball_hits_ground_at_10_over_7 : ball_hits_ground (10 / 7) :=
by
  sorry

end ball_hits_ground_at_10_over_7_l92_92326


namespace product_of_fractions_l92_92130

theorem product_of_fractions :
  (2 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) * (5 / 6 : ℚ) * (6 / 7 : ℚ) * (7 / 8 : ℚ) = 1 / 4 :=
by
  sorry

end product_of_fractions_l92_92130


namespace find_b_l92_92741

noncomputable def Q (x d b e : ℝ) : ℝ := x^3 + d*x^2 + b*x + e

theorem find_b (d b e : ℝ) (h1 : -d / 3 = -e) (h2 : -e = 1 + d + b + e) (h3 : e = 6) : b = -31 :=
by sorry

end find_b_l92_92741


namespace smallest_rectangles_to_cover_square_l92_92990

theorem smallest_rectangles_to_cover_square :
  ∃ n : ℕ, 
    (∃ a : ℕ, a = 3 * 4) ∧
    (∃ k : ℕ, k = lcm 3 4) ∧
    (∃ s : ℕ, s = k * k) ∧
    (s / a = n) ∧
    n = 12 :=
by
  sorry

end smallest_rectangles_to_cover_square_l92_92990


namespace dividend_is_correct_l92_92945

-- Definitions of the given conditions.
def divisor : ℕ := 17
def quotient : ℕ := 4
def remainder : ℕ := 8

-- Define the dividend using the given formula.
def dividend : ℕ := (divisor * quotient) + remainder

-- The theorem to prove.
theorem dividend_is_correct : dividend = 76 := by
  -- The following line contains a placeholder for the actual proof.
  sorry

end dividend_is_correct_l92_92945


namespace min_value_fractions_l92_92336

open Real

theorem min_value_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) :
  3 ≤ (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a)) :=
sorry

end min_value_fractions_l92_92336


namespace teresa_marks_ratio_l92_92608

theorem teresa_marks_ratio (science music social_studies total_marks physics_ratio : ℝ) 
  (h_science : science = 70)
  (h_music : music = 80)
  (h_social_studies : social_studies = 85)
  (h_total_marks : total_marks = 275)
  (h_physics : science + music + social_studies + physics_ratio * music = total_marks) :
  physics_ratio = 1 / 2 :=
by
  subst h_science
  subst h_music
  subst h_social_studies
  subst h_total_marks
  have : 70 + 80 + 85 + physics_ratio * 80 = 275 := h_physics
  linarith

end teresa_marks_ratio_l92_92608


namespace min_chord_length_m_l92_92803

-- Definition of the circle and the line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 6 * y + 4 = 0
def line_eq (m x y : ℝ) : Prop := m * x - y + 1 = 0

-- Theorem statement: value of m that minimizes the length of the chord
theorem min_chord_length_m (m : ℝ) : m = 1 ↔
  ∃ x y : ℝ, circle_eq x y ∧ line_eq m x y := sorry

end min_chord_length_m_l92_92803


namespace male_students_in_grade_l92_92677

-- Define the total number of students and the number of students in the sample
def total_students : ℕ := 1200
def sample_students : ℕ := 30

-- Define the number of female students in the sample
def female_students_sample : ℕ := 14

-- Calculate the number of male students in the sample
def male_students_sample := sample_students - female_students_sample

-- State the main theorem
theorem male_students_in_grade :
  (male_students_sample : ℕ) * total_students / sample_students = 640 :=
by
  -- placeholder for calculations based on provided conditions
  sorry

end male_students_in_grade_l92_92677


namespace minister_can_organize_traffic_l92_92059

-- Definition of cities and roads
structure City (α : Type) :=
(road : α → α → Prop)

-- Defining the Minister's goal
def organize_traffic {α : Type} (c : City α) (num_days : ℕ) : Prop :=
∀ x y : α, c.road x y → num_days ≤ 214

theorem minister_can_organize_traffic :
  ∃ (c : City ℕ) (num_days : ℕ), (num_days ≤ 214 ∧ organize_traffic c num_days) :=
by {
  sorry
}

end minister_can_organize_traffic_l92_92059


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l92_92324

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l92_92324


namespace find_divisor_l92_92266

theorem find_divisor (q r : ℤ) : ∃ d : ℤ, 151 = d * q + r ∧ q = 11 ∧ r = -4 → d = 14 :=
by
  intros
  sorry

end find_divisor_l92_92266


namespace set_B_listing_method_l92_92745

variable (A : Set ℕ) (B : Set ℕ)

theorem set_B_listing_method (hA : A = {1, 2, 3}) (hB : B = {x | x ∈ A}) :
  B = {1, 2, 3} :=
  by
    sorry

end set_B_listing_method_l92_92745


namespace Ariel_current_age_l92_92452

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end Ariel_current_age_l92_92452


namespace coalsBurnedEveryTwentyMinutes_l92_92789

-- Definitions based on the conditions
def totalGrillingTime : Int := 240
def coalsPerBag : Int := 60
def numberOfBags : Int := 3
def grillingInterval : Int := 20

-- Derived definitions based on conditions
def totalCoals : Int := numberOfBags * coalsPerBag
def numberOfIntervals : Int := totalGrillingTime / grillingInterval

-- The Lean theorem we want to prove
theorem coalsBurnedEveryTwentyMinutes : (totalCoals / numberOfIntervals) = 15 := by
  sorry

end coalsBurnedEveryTwentyMinutes_l92_92789


namespace intersection_of_sets_example_l92_92812

theorem intersection_of_sets_example :
  let M := { x : ℝ | 0 < x ∧ x < 4 }
  let N := { x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5 }
  let expected := { x : ℝ | 1 / 3 ≤ x ∧ x < 4 }
  (M ∩ N) = expected :=
by
  sorry

end intersection_of_sets_example_l92_92812


namespace find_ordered_pair_l92_92365

theorem find_ordered_pair (s m : ℚ) :
  (∃ t : ℚ, (5 * s - 7 = 2) ∧ 
           ((∃ (t1 : ℚ), (x = s + 3 * t1) ∧  (y = 2 + m * t1)) 
           → (x = 24 / 5) → (y = 5))) →
  (s = 9 / 5 ∧ m = 3) :=
by
  sorry

end find_ordered_pair_l92_92365


namespace range_of_a_l92_92623

theorem range_of_a (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono : ∀ ⦃a b⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (h_cond : ∀ a, f a < f (2 * a - 1) → a > 1) :
  ∀ a, f a < f (2 * a - 1) → 1 < a := 
sorry

end range_of_a_l92_92623


namespace who_received_q_first_round_l92_92770

-- Define the variables and conditions
variables (p q r : ℕ) (A B C : ℕ → ℕ) (n : ℕ)

-- Conditions
axiom h1 : 0 < p
axiom h2 : p < q
axiom h3 : q < r
axiom h4 : n ≥ 3
axiom h5 : A n = 20
axiom h6 : B n = 10
axiom h7 : C n = 9
axiom h8 : ∀ k, k > 0 → (B k = r → B (k-1) ≠ r)
axiom h9 : p + q + r = 13

-- Theorem to prove
theorem who_received_q_first_round : C 1 = q :=
sorry

end who_received_q_first_round_l92_92770


namespace sum_of_reciprocals_of_numbers_l92_92603

theorem sum_of_reciprocals_of_numbers (x y : ℕ) (h_sum : x + y = 45) (h_hcf : Nat.gcd x y = 3)
    (h_lcm : Nat.lcm x y = 100) : 1/x + 1/y = 3/20 := 
by 
  sorry

end sum_of_reciprocals_of_numbers_l92_92603


namespace find_radius_of_large_circle_l92_92430

noncomputable def radius_of_large_circle (r : ℝ) : Prop :=
  let r_A := 3
  let r_B := 2
  let d := 6
  (r - r_A)^2 + (r - r_B)^2 + 2 * (r - r_A) * (r - r_B) = d^2 ∧
  r = (5 + Real.sqrt 33) / 2

theorem find_radius_of_large_circle : ∃ (r : ℝ), radius_of_large_circle r :=
by {
  sorry
}

end find_radius_of_large_circle_l92_92430


namespace rank_from_last_l92_92858

theorem rank_from_last (total_students : ℕ) (rank_from_top : ℕ) (rank_from_last : ℕ) : 
  total_students = 35 → 
  rank_from_top = 14 → 
  rank_from_last = (total_students - rank_from_top + 1) → 
  rank_from_last = 22 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end rank_from_last_l92_92858


namespace electricity_bill_written_as_decimal_l92_92604

-- Definitions as conditions
def number : ℝ := 71.08

-- Proof statement
theorem electricity_bill_written_as_decimal : number = 71.08 :=
by sorry

end electricity_bill_written_as_decimal_l92_92604


namespace initial_cookies_l92_92529

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end initial_cookies_l92_92529


namespace gcd_and_sum_of_1729_and_867_l92_92329

-- Given numbers
def a := 1729
def b := 867

-- Define the problem statement
theorem gcd_and_sum_of_1729_and_867 : Nat.gcd a b = 1 ∧ a + b = 2596 := by
  sorry

end gcd_and_sum_of_1729_and_867_l92_92329


namespace corrections_needed_l92_92860

-- Define the corrected statements
def corrected_statements : List String :=
  ["A = 50", "B = A", "x = 1", "y = 2", "z = 3", "INPUT“How old are you?”;x",
   "INPUT x", "PRINT“A+B=”;C", "PRINT“Good-bye!”"]

-- Define the function to check if the statement is correctly formatted
def is_corrected (statement : String) : Prop :=
  statement ∈ corrected_statements

-- Lean theorem statement to prove each original incorrect statement should be correctly formatted
theorem corrections_needed (s : String) (incorrect : s ∈ ["A = B = 50", "x = 1, y = 2, z = 3", 
  "INPUT“How old are you”x", "INPUT, x", "PRINT A+B=;C", "PRINT Good-bye!"]) :
  ∃ t : String, is_corrected t :=
by 
  sorry

end corrections_needed_l92_92860


namespace david_english_marks_l92_92611

theorem david_english_marks :
  let Mathematics := 45
  let Physics := 72
  let Chemistry := 77
  let Biology := 75
  let AverageMarks := 68.2
  let TotalSubjects := 5
  let TotalMarks := AverageMarks * TotalSubjects
  let MarksInEnglish := TotalMarks - (Mathematics + Physics + Chemistry + Biology)
  MarksInEnglish = 72 :=
by
  sorry

end david_english_marks_l92_92611


namespace fifty_percent_of_2002_is_1001_l92_92773

theorem fifty_percent_of_2002_is_1001 :
  (1 / 2) * 2002 = 1001 :=
sorry

end fifty_percent_of_2002_is_1001_l92_92773


namespace parabola_ellipse_tangency_l92_92327

theorem parabola_ellipse_tangency :
  ∃ (a b : ℝ), (∀ x y, y = x^2 - 5 → (x^2 / a) + (y^2 / b) = 1) →
               (∃ x, y = x^2 - 5 ∧ (x^2 / a) + ((x^2 - 5)^2 / b) = 1) ∧
               a = 1/10 ∧ b = 1 :=
by
  sorry

end parabola_ellipse_tangency_l92_92327


namespace D_coin_count_l92_92810

def A_coin_count : ℕ := 21
def B_coin_count := A_coin_count - 9
def C_coin_count := B_coin_count + 17
def sum_A_B := A_coin_count + B_coin_count
def sum_C_D := sum_A_B + 5

theorem D_coin_count :
  ∃ D : ℕ, sum_C_D - C_coin_count = D :=
sorry

end D_coin_count_l92_92810


namespace rhombus_area_l92_92983

theorem rhombus_area (s d1 d2 : ℝ)
  (h1 : s = Real.sqrt 113)
  (h2 : abs (d1 - d2) = 8)
  (h3 : s^2 = (d1 / 2)^2 + (d2 / 2)^2) :
  (d1 * d2) / 2 = 194 := by
  sorry

end rhombus_area_l92_92983


namespace prism_coloring_1995_prism_coloring_1996_l92_92168

def prism_coloring_possible (n : ℕ) : Prop :=
  ∃ (color : ℕ → ℕ),
    (∀ i, 1 ≤ color i ∧ color i ≤ 3) ∧ -- Each color is within bounds
    (∀ i, color i ≠ color ((i + 1) % n)) ∧ -- Colors on each face must be different
    (n % 3 = 0 ∨ n ≠ 1996) -- Condition for coloring

theorem prism_coloring_1995 : prism_coloring_possible 1995 :=
sorry

theorem prism_coloring_1996 : ¬prism_coloring_possible 1996 :=
sorry

end prism_coloring_1995_prism_coloring_1996_l92_92168


namespace jessica_total_monthly_payment_l92_92428

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end jessica_total_monthly_payment_l92_92428


namespace perimeter_with_new_tiles_l92_92121

theorem perimeter_with_new_tiles (p_original : ℕ) (num_original_tiles : ℕ) (num_new_tiles : ℕ)
  (h1 : p_original = 16)
  (h2 : num_original_tiles = 9)
  (h3 : num_new_tiles = 3) :
  ∃ p_new : ℕ, p_new = 17 :=
by
  sorry

end perimeter_with_new_tiles_l92_92121


namespace arccos_one_eq_zero_l92_92746

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l92_92746


namespace monotonicity_of_f_inequality_of_f_l92_92293

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

theorem monotonicity_of_f {a : ℝ}:
(a ≥ 0 → ∀ x y : ℝ, 0 < x ∧ x < y → f x a ≤ f y a) ∧
(a < 0 → ∀ x y : ℝ, 0 < x ∧ x < y ∧ x ≥ -1 + Real.sqrt (1 - 2 * a) → f x a ≤ f y a 
∨ 0 < x ∧ x < -1 + Real.sqrt (1 - 2 * a) → f x a ≥ f y a) := sorry

theorem inequality_of_f {a : ℝ} (h : t ≥ 1) :
(f (2*t-1) a ≥ 2 * f t a - 3) ↔ (a ≤ 2) := sorry

end monotonicity_of_f_inequality_of_f_l92_92293


namespace contingency_table_confidence_l92_92458

theorem contingency_table_confidence (k_squared : ℝ) (h1 : k_squared = 4.013) : 
  confidence_99 :=
  sorry

end contingency_table_confidence_l92_92458


namespace problem1_problem2_l92_92563

-- Definitions of the conditions
def periodic_func (f: ℝ → ℝ) (a: ℝ) (x: ℝ) : Prop :=
(∀ x, f (x + 3) = f x) ∧ 
(∀ x, -2 ≤ x ∧ x < 0 → f x = x + a) ∧ 
(∀ x, 0 ≤ x ∧ x < 1 → f x = (1/2)^x)

-- 1. Prove f(13/2) = sqrt(2)/2
theorem problem1 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) : f (13/2) = (Real.sqrt 2) / 2 := 
sorry

-- 2. Prove that if f(x) has a minimum value but no maximum value, then 1 < a ≤ 5/2
theorem problem2 (f: ℝ → ℝ) (a: ℝ) (h: periodic_func f a x) (hmin: ∃ m, ∀ x, f x ≥ m) (hmax: ¬∃ M, ∀ x, f x ≤ M) : 1 < a ∧ a ≤ 5/2 :=
sorry

end problem1_problem2_l92_92563


namespace total_cost_is_correct_l92_92555

noncomputable def total_cost_of_gifts : ℝ :=
  let polo_shirts := 3 * 26
  let necklaces := 2 * 83
  let computer_game := 90
  let socks := 4 * 7
  let books := 3 * 15
  let scarves := 2 * 22
  let mugs := 5 * 8
  let sneakers := 65

  let cost_before_discounts := polo_shirts + necklaces + computer_game + socks + books + scarves + mugs + sneakers

  let discount_books := 0.10 * books
  let discount_sneakers := 0.15 * sneakers
  let cost_after_discounts := cost_before_discounts - discount_books - discount_sneakers

  let sales_tax := 0.065 * cost_after_discounts
  let cost_after_tax := cost_after_discounts + sales_tax

  let final_cost := cost_after_tax - 12

  final_cost

theorem total_cost_is_correct :
  total_cost_of_gifts = 564.96 := by
sorry

end total_cost_is_correct_l92_92555


namespace smallest_positive_integer_l92_92052

theorem smallest_positive_integer
    (n : ℕ)
    (h : ∀ (a : Fin n → ℤ), ∃ (i j : Fin n), i ≠ j ∧ (2009 ∣ (a i + a j) ∨ 2009 ∣ (a i - a j))) : n = 1006 := by
  -- Proof is required here
  sorry

end smallest_positive_integer_l92_92052


namespace find_rosy_age_l92_92936

-- Definitions and conditions
def rosy_current_age (R : ℕ) : Prop :=
  ∃ D : ℕ,
    (D = R + 18) ∧ -- David is 18 years older than Rosy
    (D + 6 = 2 * (R + 6)) -- In 6 years, David will be twice as old as Rosy

-- Proof statement: Rosy's current age is 12
theorem find_rosy_age : rosy_current_age 12 :=
  sorry

end find_rosy_age_l92_92936


namespace polynomial_coefficient_sum_l92_92493

theorem polynomial_coefficient_sum :
  let p := (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6)
  let q := 4 * x^4 + 10 * x^3 + x^2 + 15 * x - 18
  p = q →
  (4 + 10 + 1 + 15 - 18 = 12) :=
by
  intro p_eq_q
  sorry

end polynomial_coefficient_sum_l92_92493


namespace units_digit_of_expression_l92_92743

noncomputable def C : ℝ := 7 + Real.sqrt 50
noncomputable def D : ℝ := 7 - Real.sqrt 50

theorem units_digit_of_expression (C D : ℝ) (hC : C = 7 + Real.sqrt 50) (hD : D = 7 - Real.sqrt 50) : 
  ((C ^ 21 + D ^ 21) % 10) = 4 :=
  sorry

end units_digit_of_expression_l92_92743


namespace domain_of_sqrt_cos_function_l92_92331

theorem domain_of_sqrt_cos_function:
  (∀ k : ℤ, ∀ x : ℝ, 2 * Real.cos x + 1 ≥ 0 ↔ x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3)) :=
by
  sorry

end domain_of_sqrt_cos_function_l92_92331


namespace value_of_M_l92_92396

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 4025) : M = 5635 :=
sorry

end value_of_M_l92_92396


namespace solve_y_percentage_l92_92795

noncomputable def y_percentage (x y : ℝ) : ℝ :=
  100 * y / x

theorem solve_y_percentage (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) :
  y_percentage x y = 300 / 17 :=
by
  sorry

end solve_y_percentage_l92_92795


namespace katie_bead_necklaces_l92_92434

theorem katie_bead_necklaces (B : ℕ) (gemstone_necklaces : ℕ := 3) (cost_each_necklace : ℕ := 3) (total_earnings : ℕ := 21) :
  gemstone_necklaces * cost_each_necklace + B * cost_each_necklace = total_earnings → B = 4 :=
by
  intro h
  sorry

end katie_bead_necklaces_l92_92434


namespace max_n_is_4024_l92_92226

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) : ℕ :=
  4024

theorem max_n_is_4024 (a : ℕ → ℝ) (d : ℝ) (h1 : d < 0) (h2 : a 1 > 0) (h3 : a 2013 * (a 2012 + a 2013) < 0) :
  max_n_for_positive_sum a d h1 h2 h3 = 4024 :=
by
  sorry

end max_n_is_4024_l92_92226


namespace cost_per_amulet_is_30_l92_92713

variable (days_sold : ℕ := 2)
variable (amulets_per_day : ℕ := 25)
variable (price_per_amulet : ℕ := 40)
variable (faire_percentage : ℕ := 10)
variable (profit : ℕ := 300)

def total_amulets_sold := days_sold * amulets_per_day
def total_revenue := total_amulets_sold * price_per_amulet
def faire_cut := total_revenue * faire_percentage / 100
def revenue_after_faire := total_revenue - faire_cut
def total_cost := revenue_after_faire - profit
def cost_per_amulet := total_cost / total_amulets_sold

theorem cost_per_amulet_is_30 : cost_per_amulet = 30 := by
  sorry

end cost_per_amulet_is_30_l92_92713


namespace analogical_reasoning_correct_l92_92255

theorem analogical_reasoning_correct (a b c : ℝ) (hc : c ≠ 0) : (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c :=
by
  sorry

end analogical_reasoning_correct_l92_92255


namespace platform_length_is_correct_l92_92028

noncomputable def length_of_platform (T : ℕ) (t_p t_s : ℕ) : ℕ :=
  let speed_of_train := T / t_s
  let distance_when_crossing_platform := speed_of_train * t_p
  distance_when_crossing_platform - T

theorem platform_length_is_correct :
  ∀ (T t_p t_s : ℕ),
  T = 300 → t_p = 33 → t_s = 18 →
  length_of_platform T t_p t_s = 250 :=
by
  intros T t_p t_s hT ht_p ht_s
  simp [length_of_platform, hT, ht_p, ht_s]
  sorry

end platform_length_is_correct_l92_92028


namespace max_sum_of_four_numbers_l92_92102

theorem max_sum_of_four_numbers : 
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ (2 * a + 3 * b + 2 * c + 3 * d = 2017) ∧ 
    (a + b + c + d = 806) :=
by
  sorry

end max_sum_of_four_numbers_l92_92102


namespace sum_of_intercepts_eq_16_l92_92262

noncomputable def line_eq (x y : ℝ) : Prop :=
  y + 3 = -3 * (x - 5)

def x_intercept : ℝ := 4
def y_intercept : ℝ := 12

theorem sum_of_intercepts_eq_16 : 
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept) → x_intercept + y_intercept = 16 :=
by
  intros h
  sorry

end sum_of_intercepts_eq_16_l92_92262


namespace positive_real_solutions_unique_l92_92193

theorem positive_real_solutions_unique (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
(h : (a^2 - b * d) / (b + 2 * c + d) + (b^2 - c * a) / (c + 2 * d + a) + (c^2 - d * b) / (d + 2 * a + b) + (d^2 - a * c) / (a + 2 * b + c) = 0) : 
a = b ∧ b = c ∧ c = d :=
sorry

end positive_real_solutions_unique_l92_92193


namespace positive_integers_N_segment_condition_l92_92523

theorem positive_integers_N_segment_condition (N : ℕ) (x : ℕ) (n : ℕ)
  (h1 : 10 ≤ N ∧ N ≤ 10^20)
  (h2 : N = x * (10^n - 1) / 9) (h3 : 1 ≤ n ∧ n ≤ 20) : 
  N + 1 = (x + 1) * (9 + 1)^n ∧ x < 10 :=
by {
  sorry
}

end positive_integers_N_segment_condition_l92_92523


namespace service_center_milepost_l92_92213

theorem service_center_milepost :
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  service_center = 120 :=
by
  let mp4 := 50
  let mp12 := 190
  let service_center := mp4 + (mp12 - mp4) / 2
  sorry

end service_center_milepost_l92_92213


namespace distance_between_consecutive_trees_l92_92105

-- Define the conditions as separate definitions
def num_trees : ℕ := 57
def yard_length : ℝ := 720
def spaces_between_trees := num_trees - 1

-- Define the target statement to prove
theorem distance_between_consecutive_trees :
  yard_length / spaces_between_trees = 12.857142857 := sorry

end distance_between_consecutive_trees_l92_92105


namespace exists_univariate_polynomial_l92_92850

def polynomial_in_three_vars (P : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ,
  P x y z = P x y (x * y - z) ∧
  P x y z = P x (z * x - y) z ∧
  P x y z = P (y * z - x) y z

theorem exists_univariate_polynomial (P : ℝ → ℝ → ℝ → ℝ) (h : polynomial_in_three_vars P) :
  ∃ F : ℝ → ℝ, ∀ x y z : ℝ, P x y z = F (x^2 + y^2 + z^2 - x * y * z) :=
sorry

end exists_univariate_polynomial_l92_92850


namespace regression_passes_through_none_l92_92793

theorem regression_passes_through_none (b a x y : ℝ) (h₀ : (0, 0) ≠ (0*b + a, 0))
                                     (h₁ : (x, 0) ≠ (x*b + a, 0))
                                     (h₂ : (x, y) ≠ (x*b + a, y)) : 
                                     ¬ ((0, 0) = (0*b + a, 0) ∨ (x, 0) = (x*b + a, 0) ∨ (x, y) = (x*b + a, y)) :=
by sorry

end regression_passes_through_none_l92_92793


namespace find_fg_of_3_l92_92099

def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x^2 + 4 * x - 5

theorem find_fg_of_3 : f (g 3) = 31 := by
  sorry

end find_fg_of_3_l92_92099


namespace positive_integer_root_k_l92_92549

theorem positive_integer_root_k (k : ℕ) :
  (∃ x : ℕ, x > 0 ∧ x * x - 34 * x + 34 * k - 1 = 0) ↔ k = 1 :=
by
  sorry

end positive_integer_root_k_l92_92549


namespace four_p_minus_three_is_perfect_square_l92_92215

theorem four_p_minus_three_is_perfect_square 
  {n p : ℕ} (hn : 1 < n) (hp : 1 < p) (hp_prime : Prime p) 
  (h1 : n ∣ (p - 1)) (h2 : p ∣ (n^3 - 1)) :
  ∃ k : ℕ, 4 * p - 3 = k ^ 2 := 
by 
  sorry

end four_p_minus_three_is_perfect_square_l92_92215


namespace speed_of_stream_l92_92940

theorem speed_of_stream (c v : ℝ) (h1 : c - v = 8) (h2 : c + v = 12) : v = 2 :=
by {
  -- proof will go here
  sorry
}

end speed_of_stream_l92_92940


namespace rate_per_meter_l92_92200

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (rate_per_meter : ℝ) (h_d : d = 30)
    (h_total_cost : total_cost = 188.49555921538757) :
    rate_per_meter = 2 :=
by
  sorry

end rate_per_meter_l92_92200


namespace horses_legs_problem_l92_92304

theorem horses_legs_problem 
    (m h a b : ℕ) 
    (h_eq : h = m) 
    (men_to_A : m = 3 * a) 
    (men_to_B : m = 4 * b) 
    (total_legs : 2 * m + 4 * (h / 2) + 3 * a + 4 * b = 200) : 
    h = 25 :=
  sorry

end horses_legs_problem_l92_92304


namespace dollars_saved_is_correct_l92_92449

noncomputable def blender_in_store_price : ℝ := 120
noncomputable def juicer_in_store_price : ℝ := 80
noncomputable def blender_tv_price : ℝ := 4 * 28 + 12
noncomputable def total_in_store_price_with_discount : ℝ := (blender_in_store_price + juicer_in_store_price) * 0.90
noncomputable def dollars_saved : ℝ := total_in_store_price_with_discount - blender_tv_price

theorem dollars_saved_is_correct :
  dollars_saved = 56 := by
  sorry

end dollars_saved_is_correct_l92_92449


namespace remainder_of_sum_l92_92302

theorem remainder_of_sum : 
  let a := 21160
  let b := 21162
  let c := 21164
  let d := 21166
  let e := 21168
  let f := 21170
  (a + b + c + d + e + f) % 12 = 6 :=
by
  sorry

end remainder_of_sum_l92_92302


namespace problem_solution_l92_92483

/-- 
Assume we have points A, B, C, D, and E as defined in the problem with the following properties:
- Triangle ABC has a right angle at C
- AC = 4
- BC = 3
- Triangle ABD has a right angle at A
- AD = 15
- Points C and D are on opposite sides of line AB
- The line through D parallel to AC meets CB extended at E.

Prove that the ratio DE/DB simplifies to 57/80 where p = 57 and q = 80, making p + q = 137.
-/
theorem problem_solution :
  ∃ (p q : ℕ), gcd p q = 1 ∧ (∃ D E : ℝ, DE/DB = p/q ∧ p + q = 137) :=
by
  sorry

end problem_solution_l92_92483


namespace find_the_number_l92_92110

theorem find_the_number (x : ℝ) (h : 8 * x + 64 = 336) : x = 34 :=
by
  sorry

end find_the_number_l92_92110


namespace discount_is_five_l92_92509
-- Importing the needed Lean Math library

-- Defining the problem conditions
def costPrice : ℝ := 100
def profit_percent_with_discount : ℝ := 0.2
def profit_percent_without_discount : ℝ := 0.25

-- Calculating the respective selling prices
def sellingPrice_with_discount := costPrice * (1 + profit_percent_with_discount)
def sellingPrice_without_discount := costPrice * (1 + profit_percent_without_discount)

-- Calculating the discount 
def calculated_discount := sellingPrice_without_discount - sellingPrice_with_discount

-- Proving that the discount is $5
theorem discount_is_five : calculated_discount = 5 := by
  -- Proof omitted
  sorry

end discount_is_five_l92_92509


namespace smallest_integer_l92_92211

theorem smallest_integer (k : ℕ) (n : ℕ) (h936 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : 2^5 ∣ 936 * k)
  (h3 : 3^3 ∣ 936 * k)
  (h4 : 12^2 ∣ 936 * k) : 
  k = 36 :=
by {
  sorry
}

end smallest_integer_l92_92211


namespace flower_options_l92_92340

theorem flower_options (x y : ℕ) : 2 * x + 3 * y = 20 → ∃ x1 y1 x2 y2 x3 y3, 
  (2 * x1 + 3 * y1 = 20) ∧ (2 * x2 + 3 * y2 = 20) ∧ (2 * x3 + 3 * y3 = 20) ∧ 
  (((x1, y1) ≠ (x2, y2)) ∧ ((x2, y2) ≠ (x3, y3)) ∧ ((x1, y1) ≠ (x3, y3))) ∧ 
  ((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3)) :=
sorry

end flower_options_l92_92340


namespace polynomial_divisible_by_squared_root_l92_92201

noncomputable def f (a1 a2 a3 a4 x : ℝ) : ℝ := 
  x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4

noncomputable def f_prime (a1 a2 a3 a4 x : ℝ) : ℝ := 
  4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_squared_root 
  (a1 a2 a3 a4 x0 : ℝ) 
  (h1 : f a1 a2 a3 a4 x0 = 0) 
  (h2 : f_prime a1 a2 a3 a4 x0 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x, f a1 a2 a3 a4 x = (x - x0)^2 * g x := 
sorry

end polynomial_divisible_by_squared_root_l92_92201


namespace probability_at_least_one_white_ball_stall_owner_monthly_earning_l92_92386

noncomputable def prob_at_least_one_white_ball : ℚ :=
1 - (3 / 10)

theorem probability_at_least_one_white_ball : prob_at_least_one_white_ball = 9 / 10 :=
sorry

noncomputable def expected_monthly_earnings (daily_draws : ℕ) (days_in_month : ℕ) : ℤ :=
(days_in_month * (90 * 1 - 10 * 5))

theorem stall_owner_monthly_earning (daily_draws : ℕ) (days_in_month : ℕ) :
  daily_draws = 100 → days_in_month = 30 →
  expected_monthly_earnings daily_draws days_in_month = 1200 :=
sorry

end probability_at_least_one_white_ball_stall_owner_monthly_earning_l92_92386


namespace paving_cost_correct_l92_92355

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 400
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost_correct :
  cost = 8250 := by
  sorry

end paving_cost_correct_l92_92355


namespace ratio_A_B_l92_92758

theorem ratio_A_B (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : 5 * C = 8 * B) : A / B = 2 / 3 := 
by sorry

end ratio_A_B_l92_92758


namespace find_x_plus_y_l92_92831

theorem find_x_plus_y :
  ∀ (x y : ℝ), (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 :=
by
  intros x y h
  sorry

end find_x_plus_y_l92_92831


namespace farmer_ploughing_problem_l92_92019

theorem farmer_ploughing_problem (A D : ℕ) (h1 : A = 120 * D) (h2 : A - 40 = 85 * (D + 2)) : 
  A = 720 ∧ D = 6 :=
by
  sorry

end farmer_ploughing_problem_l92_92019


namespace accelerations_l92_92670

open Real

namespace Problem

variables (m M g : ℝ) (a1 a2 : ℝ)

theorem accelerations (mass_condition : 4 * m + M ≠ 0):
  (a1 = 2 * ((2 * m + M) * g) / (4 * m + M)) ∧
  (a2 = ((2 * m + M) * g) / (4 * m + M)) :=
sorry

end Problem

end accelerations_l92_92670


namespace cooper_savings_l92_92100

theorem cooper_savings :
  let daily_savings := 34
  let days_in_year := 365
  daily_savings * days_in_year = 12410 :=
by
  sorry

end cooper_savings_l92_92100


namespace perimeter_eq_28_l92_92313

theorem perimeter_eq_28 (PQ QR TS TU : ℝ) (h2 : PQ = 4) (h3 : QR = 4) 
(h5 : TS = 8) (h7 : TU = 4) : 
PQ + QR + TS + TS - TU + TU + TU = 28 := by
  sorry

end perimeter_eq_28_l92_92313


namespace find_fraction_l92_92155

variable (N : ℕ) (F : ℚ)
theorem find_fraction (h1 : N = 90) (h2 : 3 + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = F * N) : F = 1 / 15 :=
sorry

end find_fraction_l92_92155


namespace eqn_intersecting_straight_lines_l92_92573

theorem eqn_intersecting_straight_lines (x y : ℝ) : 
  x^2 - y^2 = 0 → (y = x ∨ y = -x) :=
by
  intros h
  sorry

end eqn_intersecting_straight_lines_l92_92573


namespace brownies_count_l92_92969

variable (total_people : Nat) (pieces_per_person : Nat) (cookies : Nat) (candy : Nat) (brownies : Nat)

def total_dessert_needed : Nat := total_people * pieces_per_person

def total_pieces_have : Nat := cookies + candy

def total_brownies_needed : Nat := total_dessert_needed total_people pieces_per_person - total_pieces_have cookies candy

theorem brownies_count (h1 : total_people = 7)
                       (h2 : pieces_per_person = 18)
                       (h3 : cookies = 42)
                       (h4 : candy = 63) :
                       total_brownies_needed total_people pieces_per_person cookies candy = 21 :=
by
  rw [h1, h2, h3, h4]
  sorry

end brownies_count_l92_92969


namespace mean_goals_is_correct_l92_92503

theorem mean_goals_is_correct :
  let goals5 := 5
  let players5 := 4
  let goals6 := 6
  let players6 := 3
  let goals7 := 7
  let players7 := 2
  let goals8 := 8
  let players8 := 1
  let total_goals := goals5 * players5 + goals6 * players6 + goals7 * players7 + goals8 * players8
  let total_players := players5 + players6 + players7 + players8
  (total_goals / total_players : ℝ) = 6 :=
by
  -- The proof is omitted.
  sorry

end mean_goals_is_correct_l92_92503


namespace banker_discount_calculation_l92_92772

-- Define the future value function with given interest rates and periods.
def face_value (PV : ℝ) : ℝ :=
  (PV * (1 + 0.10) ^ 4) * (1 + 0.12) ^ 4

-- Define the true discount as the difference between the future value and the present value.
def true_discount (PV : ℝ) : ℝ :=
  face_value PV - PV

-- Given conditions
def banker_gain : ℝ := 900

-- Define the banker's discount.
def banker_discount (PV : ℝ) : ℝ :=
  banker_gain + true_discount PV

-- The proof statement to prove the relationship.
theorem banker_discount_calculation (PV : ℝ) :
  banker_discount PV = banker_gain + (face_value PV - PV) := by
  sorry

end banker_discount_calculation_l92_92772


namespace correct_expression_for_representatives_l92_92807

/-- Definition for the number of representatives y given the class size x
    and the conditions that follow. -/
def elect_representatives (x : ℕ) : ℕ :=
  if 6 < x % 10 then (x + 3) / 10 else x / 10

theorem correct_expression_for_representatives (x : ℕ) :
  elect_representatives x = (x + 3) / 10 :=
by
  sorry

end correct_expression_for_representatives_l92_92807


namespace light_ray_total_distance_l92_92543

theorem light_ray_total_distance 
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (M_eq : M = (2, 1))
  (N_eq : N = (4, 5)) :
  dist M N = 2 * Real.sqrt 10 := 
sorry

end light_ray_total_distance_l92_92543


namespace algebraic_expression_value_l92_92153

theorem algebraic_expression_value (x : ℝ) 
  (h : 2 * x^2 + 3 * x + 7 = 8) : 
  4 * x^2 + 6 * x - 9 = -7 := 
by 
  sorry

end algebraic_expression_value_l92_92153


namespace rotation_test_l92_92144

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℝ) : Point ℝ :=
  Point.mk p.y (-p.x)

def A : Point ℝ := ⟨2, 3⟩
def B : Point ℝ := ⟨3, -2⟩

theorem rotation_test : rotate_90_clockwise A = B :=
by
  sorry

end rotation_test_l92_92144


namespace fence_remaining_l92_92925

noncomputable def totalFence : Float := 150.0
noncomputable def ben_whitewashed : Float := 20.0

-- Remaining fence after Ben's contribution
noncomputable def remaining_after_ben : Float := totalFence - ben_whitewashed

noncomputable def billy_fraction : Float := 1.0 / 5.0
noncomputable def billy_whitewashed : Float := billy_fraction * remaining_after_ben

-- Remaining fence after Billy's contribution
noncomputable def remaining_after_billy : Float := remaining_after_ben - billy_whitewashed

noncomputable def johnny_fraction : Float := 1.0 / 3.0
noncomputable def johnny_whitewashed : Float := johnny_fraction * remaining_after_billy

-- Remaining fence after Johnny's contribution
noncomputable def remaining_after_johnny : Float := remaining_after_billy - johnny_whitewashed

noncomputable def timmy_percentage : Float := 15.0 / 100.0
noncomputable def timmy_whitewashed : Float := timmy_percentage * remaining_after_johnny

-- Remaining fence after Timmy's contribution
noncomputable def remaining_after_timmy : Float := remaining_after_johnny - timmy_whitewashed

noncomputable def alice_fraction : Float := 1.0 / 8.0
noncomputable def alice_whitewashed : Float := alice_fraction * remaining_after_timmy

-- Remaining fence after Alice's contribution
noncomputable def remaining_fence : Float := remaining_after_timmy - alice_whitewashed

theorem fence_remaining : remaining_fence = 51.56 :=
by
    -- Placeholder for actual proof
    sorry

end fence_remaining_l92_92925


namespace largest_operation_result_is_div_l92_92002

noncomputable def max_operation_result : ℚ :=
  max (max (-1 + (-1 / 2)) (-1 - (-1 / 2)))
      (max (-1 * (-1 / 2)) (-1 / (-1 / 2)))

theorem largest_operation_result_is_div :
  max_operation_result = 2 := by
  sorry

end largest_operation_result_is_div_l92_92002


namespace squares_of_roots_equation_l92_92642

theorem squares_of_roots_equation (a b x : ℂ) 
  (h : ab * x^2 - (a + b) * x + 1 = 0) : 
  a^2 * b^2 * x^2 - (a^2 + b^2) * x + 1 = 0 :=
sorry

end squares_of_roots_equation_l92_92642


namespace div_relation_l92_92857

variables (a b c : ℚ)

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 1 / 2) : c / a = 2 / 3 :=
by
  -- proof to be filled in
  sorry

end div_relation_l92_92857


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l92_92687

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l92_92687


namespace treaty_of_versailles_original_day_l92_92972

-- Define the problem in Lean terms
def treatySignedDay : Nat -> Nat -> String
| 1919, 6 => "Saturday"
| _, _ => "Unknown"

-- Theorem statement
theorem treaty_of_versailles_original_day :
  treatySignedDay 1919 6 = "Saturday" :=
sorry

end treaty_of_versailles_original_day_l92_92972


namespace A_more_than_B_l92_92311

noncomputable def proportion := (5, 3, 2, 3)
def C_share := 1000
def parts := 2
noncomputable def part_value := C_share / parts
noncomputable def A_share := part_value * 5
noncomputable def B_share := part_value * 3

theorem A_more_than_B : A_share - B_share = 1000 := by
  sorry

end A_more_than_B_l92_92311


namespace expression_value_zero_l92_92281

theorem expression_value_zero (a b c : ℝ) (h1 : a^2 + b = b^2 + c) (h2 : b^2 + c = c^2 + a) (h3 : c^2 + a = a^2 + b) :
  a * (a^2 - b^2) + b * (b^2 - c^2) + c * (c^2 - a^2) = 0 :=
by
  sorry

end expression_value_zero_l92_92281


namespace average_marks_is_25_l92_92402

variable (M P C : ℕ)

def average_math_chemistry (M C : ℕ) : ℕ :=
  (M + C) / 2

theorem average_marks_is_25 (M P C : ℕ) 
  (h₁ : M + P = 30)
  (h₂ : C = P + 20) : 
  average_math_chemistry M C = 25 :=
by
  sorry

end average_marks_is_25_l92_92402


namespace matching_red_pair_probability_l92_92490

def total_socks := 8
def red_socks := 4
def blue_socks := 2
def green_socks := 2

noncomputable def total_pairs := Nat.choose total_socks 2
noncomputable def red_pairs := Nat.choose red_socks 2
noncomputable def blue_pairs := Nat.choose blue_socks 2
noncomputable def green_pairs := Nat.choose green_socks 2
noncomputable def total_matching_pairs := red_pairs + blue_pairs + green_pairs
noncomputable def probability_red := (red_pairs : ℚ) / total_matching_pairs

theorem matching_red_pair_probability : probability_red = 3 / 4 :=
  by sorry

end matching_red_pair_probability_l92_92490


namespace compute_fractions_product_l92_92965

theorem compute_fractions_product :
  (2 * (2^4 - 1) / (2 * (2^4 + 1))) *
  (2 * (3^4 - 1) / (2 * (3^4 + 1))) *
  (2 * (4^4 - 1) / (2 * (4^4 + 1))) *
  (2 * (5^4 - 1) / (2 * (5^4 + 1))) *
  (2 * (6^4 - 1) / (2 * (6^4 + 1))) *
  (2 * (7^4 - 1) / (2 * (7^4 + 1)))
  = 4400 / 135 := by
sorry

end compute_fractions_product_l92_92965


namespace max_length_of_third_side_of_triangle_l92_92922

noncomputable def max_third_side_length (D E F : ℝ) (a b : ℝ) : ℝ :=
  let c_square := a^2 + b^2 - 2 * a * b * Real.cos (90 * Real.pi / 180)
  Real.sqrt c_square

theorem max_length_of_third_side_of_triangle (D E F : ℝ) (a b : ℝ) (h₁ : Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1)
    (h₂ : a = 8) (h₃ : b = 15) : 
    max_third_side_length D E F a b = 17 := 
by
  sorry

end max_length_of_third_side_of_triangle_l92_92922


namespace intersection_M_N_l92_92964

def M (x : ℝ) : Prop := x^2 ≥ x

def N (x : ℝ) (y : ℝ) : Prop := y = 3^x + 1

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | ∃ y : ℝ, N x y ∧ y > 1} = {x : ℝ | x > 1} :=
by {
  sorry
}

end intersection_M_N_l92_92964


namespace num_six_year_olds_l92_92252

theorem num_six_year_olds (x : ℕ) 
  (h3 : 13 = 13) 
  (h4 : 20 = 20) 
  (h5 : 15 = 15) 
  (h_sum1 : 13 + 20 = 33) 
  (h_sum2 : 15 + x = 15 + x) 
  (h_avg : 2 * 35 = 70) 
  (h_total : 33 + (15 + x) = 70) : 
  x = 22 :=
by
  sorry

end num_six_year_olds_l92_92252


namespace binom_18_6_mul_smallest_prime_gt_10_eq_80080_l92_92505

theorem binom_18_6_mul_smallest_prime_gt_10_eq_80080 :
  (Nat.choose 18 6) * 11 = 80080 := sorry

end binom_18_6_mul_smallest_prime_gt_10_eq_80080_l92_92505


namespace cube_volume_and_diagonal_l92_92272

theorem cube_volume_and_diagonal (A : ℝ) (s : ℝ) (V : ℝ) (d : ℝ) 
  (h1 : A = 864)
  (h2 : 6 * s^2 = A)
  (h3 : V = s^3)
  (h4 : d = s * Real.sqrt 3) :
  V = 1728 ∧ d = 12 * Real.sqrt 3 :=
by 
  sorry

end cube_volume_and_diagonal_l92_92272


namespace find_a_l92_92536

variable (a x : ℝ)

noncomputable def curve1 (x : ℝ) := x + Real.log x
noncomputable def curve2 (a x : ℝ) := a * x^2 + (a + 2) * x + 1

theorem find_a : (curve1 1 = 1 ∧ curve1 1 = curve2 a 1) → a = 8 :=
by
  sorry

end find_a_l92_92536


namespace exists_nat_numbers_satisfying_sum_l92_92173

theorem exists_nat_numbers_satisfying_sum :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 :=
sorry

end exists_nat_numbers_satisfying_sum_l92_92173


namespace num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l92_92125

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_tens_and_units_is_ten (n : ℕ) : Prop :=
  (n / 10 % 10) + (n % 10) = 10

theorem num_even_three_digit_numbers_with_sum_of_tens_and_units_10 : 
  ∃! (N : ℕ), (N = 36) ∧ 
               (∀ n : ℕ, is_three_digit n → is_even n → sum_of_tens_and_units_is_ten n →
                         n = 36) := 
sorry

end num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l92_92125


namespace smallest_repeating_block_length_of_7_over_13_l92_92181

theorem smallest_repeating_block_length_of_7_over_13 : 
  ∀ k, (∃ a b, 7 / 13 = a + (b / 10^k)) → k = 6 := 
sorry

end smallest_repeating_block_length_of_7_over_13_l92_92181


namespace max_gcd_coprime_l92_92070

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l92_92070


namespace gcd_117_182_l92_92038

theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by 
  sorry

end gcd_117_182_l92_92038


namespace divide_segment_l92_92308

theorem divide_segment (a : ℝ) (n : ℕ) (h : 0 < n) : 
  ∃ P : ℝ, P = a / (n + 1) ∧ P > 0 :=
by
  sorry

end divide_segment_l92_92308


namespace product_of_terms_form_l92_92667

theorem product_of_terms_form 
  (a b c d : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ p q : ℝ, 
    (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 
    ∧ 0 ≤ p 
    ∧ 0 ≤ q := 
by
  let p := a * c + 5 * b * d
  let q := a * d + b * c
  use p, q
  sorry

end product_of_terms_form_l92_92667


namespace JimAgeInXYears_l92_92018

-- Definitions based on conditions
def TomCurrentAge := 37
def JimsAge7YearsAgo := 5 + (TomCurrentAge - 7) / 2

-- We introduce a variable X to represent the number of years into the future.
variable (X : ℕ)

-- Lean 4 statement to prove that Jim will be 27 + X years old in X years from now.
theorem JimAgeInXYears : JimsAge7YearsAgo + 7 + X = 27 + X := 
by
  sorry

end JimAgeInXYears_l92_92018


namespace possible_values_of_a1_l92_92230

def sequence_satisfies_conditions (a : ℕ → ℕ) : Prop :=
  (∀ n ≥ 1, a n ≤ a (n + 1) ∧ a (n + 1) ≤ a n + 5) ∧
  (∀ n ≥ 1, n ∣ a n)

theorem possible_values_of_a1 (a : ℕ → ℕ) :
  sequence_satisfies_conditions a → ∃ k ≤ 26, a 1 = k :=
by
  sorry

end possible_values_of_a1_l92_92230


namespace total_investment_amount_l92_92937

-- Define the initial conditions
def amountAt8Percent : ℝ := 3000
def interestAt8Percent (amount : ℝ) : ℝ := amount * 0.08
def interestAt10Percent (amount : ℝ) : ℝ := amount * 0.10
def totalAmount (x y : ℝ) : ℝ := x + y

-- State the theorem
theorem total_investment_amount : 
    let x := 2400
    totalAmount amountAt8Percent x = 5400 :=
by
  sorry

end total_investment_amount_l92_92937


namespace peanut_price_is_correct_l92_92115

noncomputable def price_per_pound_of_peanuts : ℝ := 
  let total_weight := 100
  let mixed_price_per_pound := 2.5
  let cashew_weight := 60
  let cashew_price_per_pound := 4
  let peanut_weight := total_weight - cashew_weight
  let total_revenue := total_weight * mixed_price_per_pound
  let cashew_cost := cashew_weight * cashew_price_per_pound
  let peanut_cost := total_revenue - cashew_cost
  peanut_cost / peanut_weight

theorem peanut_price_is_correct :
  price_per_pound_of_peanuts = 0.25 := 
by sorry

end peanut_price_is_correct_l92_92115


namespace monkey_climb_ladder_l92_92066

theorem monkey_climb_ladder (n : ℕ) 
  (h1 : ∀ k, (k % 18 = 0 → (k - 18 + 10) % 26 = 8))
  (h2 : ∀ m, (m % 10 = 0 → (m - 10 + 18) % 26 = 18))
  (h3 : ∀ l, (l % 18 = 0 ∧ l % 10 = 0 → l = 0 ∨ l = 26)):
  n = 26 :=
by
  sorry

end monkey_climb_ladder_l92_92066


namespace equipment_total_cost_l92_92583

def cost_jersey : ℝ := 25
def cost_shorts : ℝ := 15.20
def cost_socks : ℝ := 6.80
def cost_cleats : ℝ := 40
def cost_water_bottle : ℝ := 12
def cost_one_player := cost_jersey + cost_shorts + cost_socks + cost_cleats + cost_water_bottle
def num_players : ℕ := 25
def total_cost_for_team : ℝ := cost_one_player * num_players

theorem equipment_total_cost :
  total_cost_for_team = 2475 := by
  sorry

end equipment_total_cost_l92_92583


namespace find_cos_C_l92_92003

noncomputable def cos_C_eq (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : Prop :=
  Real.cos C = 7 / 25

theorem find_cos_C (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) :
  cos_C_eq A B C a b c h1 h2 :=
sorry

end find_cos_C_l92_92003


namespace area_three_layers_is_nine_l92_92206

-- Define the areas as natural numbers
variable (P Q R S T U V : ℕ)

-- Define the combined area of the rugs
def combined_area_rugs := P + Q + R + 2 * (S + T + U) + 3 * V = 90

-- Define the total area covered by the floor
def total_area_floor := P + Q + R + S + T + U + V = 60

-- Define the area covered by exactly two layers of rug
def area_two_layers := S + T + U = 12

-- Define the area covered by exactly three layers of rug
def area_three_layers := V

-- Prove the area covered by exactly three layers of rug is 9
theorem area_three_layers_is_nine
  (h1 : combined_area_rugs P Q R S T U V)
  (h2 : total_area_floor P Q R S T U V)
  (h3 : area_two_layers S T U) :
  area_three_layers V = 9 := by
  sorry

end area_three_layers_is_nine_l92_92206


namespace linear_system_sum_l92_92832

theorem linear_system_sum (x y : ℝ) 
  (h1: x - y = 2) 
  (h2: y = 2): 
  x + y = 6 := 
sorry

end linear_system_sum_l92_92832


namespace find_set_B_l92_92649

open Set

variable (U : Finset ℕ) (A B : Finset ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7})
variable (h1 : (U \ (A ∪ B)) = {1, 3})
variable (h2 : A ∩ (U \ B) = {2, 5})

theorem find_set_B : B = {4, 6, 7} := by
  sorry

end find_set_B_l92_92649


namespace three_children_meet_l92_92001

theorem three_children_meet 
  (children : Finset ℕ)
  (visited_times : ℕ → ℕ)
  (meet_at_stand : ℕ → ℕ → Prop)
  (h_children_count : children.card = 7)
  (h_visited_times : ∀ c ∈ children, visited_times c = 3)
  (h_meet_pairwise : ∀ (c1 c2 : ℕ), c1 ∈ children → c2 ∈ children → c1 ≠ c2 → meet_at_stand c1 c2) :
  ∃ (t : ℕ), ∃ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
  c1 ∈ children ∧ c2 ∈ children ∧ c3 ∈ children ∧ 
  meet_at_stand c1 t ∧ meet_at_stand c2 t ∧ meet_at_stand c3 t := 
sorry

end three_children_meet_l92_92001


namespace bad_iff_prime_l92_92663

def a_n (n : ℕ) : ℕ := (2 * n)^2 + 1

def is_bad (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a_n n = a^2 + b^2

theorem bad_iff_prime (n : ℕ) : is_bad n ↔ Nat.Prime (a_n n) :=
by
  sorry

end bad_iff_prime_l92_92663


namespace point_in_third_quadrant_l92_92407

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : 
  (-b < 0) ∧ (a < 0) ∧ (-b > a) :=
by
  sorry

end point_in_third_quadrant_l92_92407


namespace quadratic_roots_l92_92567

theorem quadratic_roots (m : ℝ) (h1 : m > 4) :
  (∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0 ∧ (m-5) * y^2 - 2 * (m + 2) * y + m = 0)
  ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0)
  ∨ (¬((∃ x y : ℝ, x ≠ y ∧ (m-5) * x^2 - 2 * (m + 2) * x + m = 0) ∨ (m = 5 ∧ ∃ x : ℝ, (m-5) * x^2 - 2 * (m + 2) * x + m = 0))) :=
by
  sorry

end quadratic_roots_l92_92567


namespace train_speed_l92_92949

theorem train_speed (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_man_km_hr : ℝ) 
  (h_length : length_of_train = 420)
  (h_time : time_to_cross = 62.99496040316775)
  (h_man_speed : speed_of_man_km_hr = 6) :
  ∃ speed_of_train_km_hr : ℝ, speed_of_train_km_hr = 30 :=
by
  sorry

end train_speed_l92_92949


namespace mult_xy_eq_200_over_3_l92_92231

def hash_op (a b : ℚ) : ℚ := a + a / b

def x : ℚ := hash_op 8 3

def y : ℚ := hash_op 5 4

theorem mult_xy_eq_200_over_3 : x * y = 200 / 3 := 
by 
  -- lean uses real division operator, and hash_op must remain rational
  sorry

end mult_xy_eq_200_over_3_l92_92231


namespace correct_pythagorean_triple_l92_92177

def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem correct_pythagorean_triple :
  (is_pythagorean_triple 1 2 3 = false) ∧ 
  (is_pythagorean_triple 4 5 6 = false) ∧ 
  (is_pythagorean_triple 6 8 9 = false) ∧ 
  (is_pythagorean_triple 7 24 25 = true) :=
by
  sorry

end correct_pythagorean_triple_l92_92177


namespace calculate_height_l92_92679

def base_length : ℝ := 2 -- in cm
def base_width : ℝ := 5 -- in cm
def volume : ℝ := 30 -- in cm^3

theorem calculate_height: base_length * base_width * 3 = volume :=
by
  -- base_length * base_width = 10
  -- 10 * 3 = 30
  sorry

end calculate_height_l92_92679


namespace total_weight_of_rings_l92_92006

-- Conditions
def weight_orange : ℝ := 0.08333333333333333
def weight_purple : ℝ := 0.3333333333333333
def weight_white : ℝ := 0.4166666666666667

-- Goal
theorem total_weight_of_rings : weight_orange + weight_purple + weight_white = 0.8333333333333333 := by
  sorry

end total_weight_of_rings_l92_92006


namespace hannah_trip_time_ratio_l92_92477

theorem hannah_trip_time_ratio 
  (u : ℝ) -- Speed on the first trip in miles per hour.
  (u_pos : u > 0) -- Speed should be positive.
  (t1 t2 : ℝ) -- Time taken for the first and second trip respectively.
  (h_t1 : t1 = 30 / u) -- Time for the first trip.
  (h_t2 : t2 = 150 / (4 * u)) -- Time for the second trip.
  : t2 / t1 = 1.25 := by
  sorry

end hannah_trip_time_ratio_l92_92477


namespace log_conditions_l92_92671

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log_conditions (m n : ℝ) (h₁ : log_base m 9 < log_base n 9)
  (h₂ : log_base n 9 < 0) : 0 < m ∧ m < n ∧ n < 1 :=
sorry

end log_conditions_l92_92671


namespace find_unknown_number_l92_92624

theorem find_unknown_number (y : ℝ) (h : 25 / y = 80 / 100) : y = 31.25 :=
sorry

end find_unknown_number_l92_92624


namespace cost_per_mile_l92_92993

def miles_per_week : ℕ := 3 * 50 + 4 * 100
def weeks_per_year : ℕ := 52
def miles_per_year : ℕ := miles_per_week * weeks_per_year
def weekly_fee : ℕ := 100
def yearly_total_fee : ℕ := 7800
def yearly_weekly_fees : ℕ := 52 * weekly_fee
def yearly_mile_fees := yearly_total_fee - yearly_weekly_fees
def pay_per_mile := yearly_mile_fees / miles_per_year

theorem cost_per_mile : pay_per_mile = 909 / 10000 := by
  -- proof will be added here
  sorry

end cost_per_mile_l92_92993


namespace find_sum_of_digits_l92_92569

theorem find_sum_of_digits (a b c d : ℕ) 
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : a = 1)
  (h3 : 1000 * a + 100 * b + 10 * c + d - (100 * b + 10 * c + d) < 100)
  : a + b + c + d = 2 := 
sorry

end find_sum_of_digits_l92_92569


namespace parallel_vectors_l92_92380

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (-2, k)

theorem parallel_vectors (k : ℝ) (h : (1, 4) = c k) : k = -8 :=
sorry

end parallel_vectors_l92_92380


namespace extreme_values_l92_92497

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 3

theorem extreme_values (a b : ℝ) : 
  (f a b (-1) = 10) ∧ (f a b 2 = -17) →
  (6 * (-1)^2 + 2 * a * (-1) + b = 0) ∧ (6 * 2^2 + 2 * (a * 2) + b = 0) →
  a = -3 ∧ b = -12 :=
by 
  sorry

end extreme_values_l92_92497


namespace product_of_solutions_l92_92388

theorem product_of_solutions : 
  ∀ x : ℝ, 5 = -2 * x^2 + 6 * x → (∃ α β : ℝ, (α ≠ β ∧ (α * β = 5 / 2))) :=
by
  sorry

end product_of_solutions_l92_92388


namespace combined_mpg_l92_92855

def ray_mpg := 50
def tom_mpg := 20
def ray_miles := 100
def tom_miles := 200

theorem combined_mpg : 
  let ray_gallons := ray_miles / ray_mpg
  let tom_gallons := tom_miles / tom_mpg
  let total_gallons := ray_gallons + tom_gallons
  let total_miles := ray_miles + tom_miles
  total_miles / total_gallons = 25 :=
by
  sorry

end combined_mpg_l92_92855


namespace ramsey_six_vertices_monochromatic_quadrilateral_l92_92650

theorem ramsey_six_vertices_monochromatic_quadrilateral :
  ∀ (V : Type) (E : V → V → Prop), (∀ x y : V, x ≠ y → E x y ∨ ¬ E x y) →
  ∃ (u v w x : V), u ≠ v ∧ v ≠ w ∧ w ≠ x ∧ x ≠ u ∧ (E u v = E v w ∧ E v w = E w x ∧ E w x = E x u) :=
by sorry

end ramsey_six_vertices_monochromatic_quadrilateral_l92_92650


namespace race_distance_l92_92828

theorem race_distance (d x y z : ℝ) 
  (h1: d / x = (d - 25) / y)
  (h2: d / y = (d - 15) / z)
  (h3: d / x = (d - 35) / z) :
  d = 75 :=
sorry

end race_distance_l92_92828


namespace paula_aunt_gave_her_total_money_l92_92683

theorem paula_aunt_gave_her_total_money :
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  total_spent + money_left = 109 :=
by
  let shirt_price := 11
  let pants_price := 13
  let shirts_bought := 2
  let money_left := 74
  let total_spent := shirts_bought * shirt_price + pants_price
  show total_spent + money_left = 109
  sorry

end paula_aunt_gave_her_total_money_l92_92683


namespace original_price_per_kg_l92_92448

theorem original_price_per_kg (P : ℝ) (S : ℝ) (reduced_price : ℝ := 0.8 * P) (total_cost : ℝ := 400) (extra_salt : ℝ := 10) :
  S * P = total_cost ∧ (S + extra_salt) * reduced_price = total_cost → P = 10 :=
by
  intros
  sorry

end original_price_per_kg_l92_92448


namespace tilde_tilde_tilde_47_l92_92237

def tilde (N : ℝ) : ℝ := 0.4 * N + 2

theorem tilde_tilde_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := 
by
  sorry

end tilde_tilde_tilde_47_l92_92237


namespace roots_expression_value_l92_92488

theorem roots_expression_value {m n : ℝ} (h₁ : m^2 - 3 * m - 2 = 0) (h₂ : n^2 - 3 * n - 2 = 0) : 
  (7 * m^2 - 21 * m - 3) * (3 * n^2 - 9 * n + 5) = 121 := 
by 
  sorry

end roots_expression_value_l92_92488


namespace hidden_dots_are_32_l92_92738

theorem hidden_dots_are_32 
  (visible_faces : List ℕ)
  (h_visible : visible_faces = [1, 2, 3, 4, 4, 5, 6, 6])
  (num_dice : ℕ)
  (h_num_dice : num_dice = 3)
  (faces_per_die : List ℕ)
  (h_faces_per_die : faces_per_die = [1, 2, 3, 4, 5, 6]) :
  63 - visible_faces.sum = 32 := by
  sorry

end hidden_dots_are_32_l92_92738


namespace int_999_column_is_C_l92_92076

def column_of_int (n : ℕ) : String :=
  let m := n - 2
  match (m / 7 % 2, m % 7) with
  | (0, 0) => "A"
  | (0, 1) => "B"
  | (0, 2) => "C"
  | (0, 3) => "D"
  | (0, 4) => "E"
  | (0, 5) => "F"
  | (0, 6) => "G"
  | (1, 0) => "G"
  | (1, 1) => "F"
  | (1, 2) => "E"
  | (1, 3) => "D"
  | (1, 4) => "C"
  | (1, 5) => "B"
  | (1, 6) => "A"
  | _      => "Invalid"

theorem int_999_column_is_C : column_of_int 999 = "C" := by
  sorry

end int_999_column_is_C_l92_92076


namespace cost_of_new_game_l92_92580

theorem cost_of_new_game (initial_money : ℕ) (money_left : ℕ) (toy_cost : ℕ) (toy_count : ℕ)
  (h_initial : initial_money = 68) (h_toy_cost : toy_cost = 7) (h_toy_count : toy_count = 3) 
  (h_money_left : money_left = toy_count * toy_cost) :
  initial_money - money_left = 47 :=
by {
  sorry
}

end cost_of_new_game_l92_92580


namespace units_digit_n_l92_92404

theorem units_digit_n (m n : ℕ) (h₁ : m * n = 14^8) (hm : m % 10 = 6) : n % 10 = 1 :=
sorry

end units_digit_n_l92_92404


namespace minimize_squares_in_rectangle_l92_92999

theorem minimize_squares_in_rectangle (w h : ℕ) (hw : w = 63) (hh : h = 42) : 
  ∃ s : ℕ, s = Nat.gcd w h ∧ s = 21 :=
by
  sorry

end minimize_squares_in_rectangle_l92_92999


namespace length_of_each_train_l92_92630

theorem length_of_each_train (L : ℝ) (s1 : ℝ) (s2 : ℝ) (t : ℝ)
    (h1 : s1 = 46) (h2 : s2 = 36) (h3 : t = 144) (h4 : 2 * L = ((s1 - s2) * (5 / 18)) * t) :
    L = 200 := 
sorry

end length_of_each_train_l92_92630


namespace hexagon_side_count_l92_92753

noncomputable def convex_hexagon_sides (a b perimeter : ℕ) : ℕ := 
  if a ≠ b then 6 - (perimeter - (6 * b)) else 0

theorem hexagon_side_count (G H I J K L : ℕ)
  (a b : ℕ)
  (p : ℕ)
  (dist_a : a = 7)
  (dist_b : b = 8)
  (perimeter : p = 46)
  (cond : GHIJKL = [a, b, X, Y, Z, W] ∧ ∀ x ∈ [X, Y, Z, W], x = a ∨ x = b)
  : convex_hexagon_sides a b p = 4 :=
by 
  sorry

end hexagon_side_count_l92_92753


namespace sin_theta_value_l92_92511

theorem sin_theta_value (a : ℝ) (h : a ≠ 0) (h_tan : Real.tan θ = -a) (h_point : P = (a, -1)) : Real.sin θ = -Real.sqrt 2 / 2 :=
sorry

end sin_theta_value_l92_92511


namespace tree_height_increase_l92_92700

-- Definitions given in the conditions
def h0 : ℝ := 4
def h (t : ℕ) (x : ℝ) : ℝ := h0 + t * x

-- Proof statement
theorem tree_height_increase (x : ℝ) :
  h 6 x = (4 / 3) * h 4 x + h 4 x → x = 2 :=
by
  intro h6_eq
  rw [h, h] at h6_eq
  norm_num at h6_eq
  sorry

end tree_height_increase_l92_92700


namespace proof_f_g_f3_l92_92025

def f (x: ℤ) : ℤ := 2*x + 5
def g (x: ℤ) : ℤ := 5*x + 2

theorem proof_f_g_f3 :
  f (g (f 3)) = 119 := by
  sorry

end proof_f_g_f3_l92_92025


namespace butterfat_milk_mixing_l92_92123

theorem butterfat_milk_mixing :
  ∀ (x : ℝ), 
  (0.35 * x + 0.10 * 12 = 0.20 * (x + 12)) → x = 8 :=
by
  intro x
  intro h
  sorry

end butterfat_milk_mixing_l92_92123


namespace total_widgets_sold_after_20_days_l92_92000

-- Definition of the arithmetic sequence
def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 * n - 1

-- Sum of the first n terms of the sequence
def sum_of_widgets_sold (n : ℕ) : ℕ :=
  n * (widgets_sold_on_day 1 + widgets_sold_on_day n) / 2

-- Prove that the total widgets sold after 20 days is 400
theorem total_widgets_sold_after_20_days : sum_of_widgets_sold 20 = 400 :=
by
  sorry

end total_widgets_sold_after_20_days_l92_92000


namespace total_books_count_l92_92579

theorem total_books_count (total_cost : ℕ) (math_book_cost : ℕ) (history_book_cost : ℕ) 
    (math_books_count : ℕ) (history_books_count : ℕ) (total_books : ℕ) :
    total_cost = 390 ∧ math_book_cost = 4 ∧ history_book_cost = 5 ∧ 
    math_books_count = 10 ∧ total_books = math_books_count + history_books_count ∧ 
    total_cost = (math_book_cost * math_books_count) + (history_book_cost * history_books_count) →
    total_books = 80 := by
  sorry

end total_books_count_l92_92579


namespace solve_for_x_l92_92802

-- Definitions of the conditions
def condition (x : ℚ) : Prop :=
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 2 * x - 24)

-- Statement of the theorem
theorem solve_for_x (x : ℚ) (h : condition x) : x = -5 / 4 :=
by 
  sorry

end solve_for_x_l92_92802


namespace rain_at_least_one_day_l92_92712

-- Define the probabilities
def P_A1 : ℝ := 0.30
def P_A2 : ℝ := 0.40
def P_A2_given_A1 : ℝ := 0.70

-- Define complementary probabilities
def P_not_A1 : ℝ := 1 - P_A1
def P_not_A2 : ℝ := 1 - P_A2
def P_not_A2_given_A1 : ℝ := 1 - P_A2_given_A1

-- Calculate probabilities of no rain on both days under different conditions
def P_no_rain_both_days_if_no_rain_first : ℝ := P_not_A1 * P_not_A2
def P_no_rain_both_days_if_rain_first : ℝ := P_A1 * P_not_A2_given_A1

-- Total probability of no rain on both days
def P_no_rain_both_days : ℝ := P_no_rain_both_days_if_no_rain_first + P_no_rain_both_days_if_rain_first

-- Probability of rain on at least one of the two days
def P_rain_one_or_more_days : ℝ := 1 - P_no_rain_both_days

-- Expressing the result as a percentage
def result_percentage : ℝ := P_rain_one_or_more_days * 100

-- Theorem statement
theorem rain_at_least_one_day : result_percentage = 49 := by
  -- We skip the proof
  sorry

end rain_at_least_one_day_l92_92712


namespace find_a_and_b_l92_92391

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the curve equation
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 3 * x^2 + a

-- Main theorem to prove a = -1 and b = 3 given tangency conditions
theorem find_a_and_b 
  (k : ℝ) (a b : ℝ) (tangent_point : ℝ × ℝ)
  (h_tangent : tangent_point = (1, 3))
  (h_line : line k tangent_point.1 = tangent_point.2)
  (h_curve : curve a b tangent_point.1 = tangent_point.2)
  (h_slope : curve_derivative a tangent_point.1 = k) : 
  a = -1 ∧ b = 3 := 
by
  sorry

end find_a_and_b_l92_92391


namespace sin_60_equiv_l92_92265

theorem sin_60_equiv : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := 
by
  sorry

end sin_60_equiv_l92_92265


namespace john_heroes_on_large_sheets_front_l92_92454

noncomputable def num_pictures_on_large_sheets_front : ℕ :=
  let total_pictures := 20
  let minutes_spent := 75 - 5
  let average_time_per_picture := 5
  let front_pictures := total_pictures / 2
  let x := front_pictures / 3
  2 * x

theorem john_heroes_on_large_sheets_front : num_pictures_on_large_sheets_front = 6 :=
by
  sorry

end john_heroes_on_large_sheets_front_l92_92454


namespace min_value_arithmetic_seq_l92_92171

theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h_arith_seq : ∀ n, a n ≤ a (n + 1)) (h_pos : ∀ n, a n > 0) (h_cond : a 1 + a 2017 = 2) :
  ∃ (min_value : ℝ), min_value = 2 ∧ (∀ (x y : ℝ), x + y = 2 → x > 0 → y > 0 → x + y / (x * y) = 2) :=
  sorry

end min_value_arithmetic_seq_l92_92171


namespace distinct_arrangements_balloon_l92_92656

noncomputable def totalPermutations (n nl no : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial nl * Nat.factorial no)

theorem distinct_arrangements_balloon :
  totalPermutations 7 2 2 = 1260 := by 
  sorry

end distinct_arrangements_balloon_l92_92656


namespace dad_strawberry_weight_l92_92952

theorem dad_strawberry_weight :
  ∀ (T L M D : ℕ), T = 36 → L = 8 → M = 12 → (D = T - L - M) → D = 16 :=
by
  intros T L M D hT hL hM hD
  rw [hT, hL, hM] at hD
  exact hD

end dad_strawberry_weight_l92_92952


namespace simplify_expression_l92_92501

variable (a b : ℝ)

theorem simplify_expression (a b : ℝ) :
  (6 * a^5 * b^2) / (3 * a^3 * b^2) + ((2 * a * b^3)^2) / ((-b^2)^3) = -2 * a^2 :=
by 
  sorry

end simplify_expression_l92_92501


namespace tens_digit_of_23_pow_2023_l92_92507

theorem tens_digit_of_23_pow_2023 : (23 ^ 2023 % 100 / 10) = 6 :=
by
  sorry

end tens_digit_of_23_pow_2023_l92_92507


namespace bran_tuition_fee_l92_92345

theorem bran_tuition_fee (P : ℝ) (S : ℝ) (M : ℕ) (R : ℝ) (T : ℝ) 
  (h1 : P = 15) (h2 : S = 0.30) (h3 : M = 3) (h4 : R = 18) 
  (h5 : 0.70 * T - (M * P) = R) : T = 90 :=
by
  sorry

end bran_tuition_fee_l92_92345


namespace max_prime_difference_l92_92899

theorem max_prime_difference (a b c d : ℕ) 
  (p1 : Prime a) (p2 : Prime b) (p3 : Prime c) (p4 : Prime d)
  (p5 : Prime (a + b + c + 18 + d)) (p6 : Prime (a + b + c + 18 - d))
  (p7 : Prime (b + c)) (p8 : Prime (c + d))
  (h1 : a + b + c = 2010) (h2 : a ≠ 3) (h3 : b ≠ 3) (h4 : c ≠ 3) (h5 : d ≠ 3) (h6 : d ≤ 50)
  (distinct_primes : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ (a + b + c + 18 + d)
                    ∧ a ≠ (a + b + c + 18 - d) ∧ a ≠ (b + c) ∧ a ≠ (c + d)
                    ∧ b ≠ c ∧ b ≠ d ∧ b ≠ (a + b + c + 18 + d)
                    ∧ b ≠ (a + b + c + 18 - d) ∧ b ≠ (b + c) ∧ b ≠ (c + d)
                    ∧ c ≠ d ∧ c ≠ (a + b + c + 18 + d)
                    ∧ c ≠ (a + b + c + 18 - d) ∧ c ≠ (b + c) ∧ c ≠ (c + d)
                    ∧ d ≠ (a + b + c + 18 + d) ∧ d ≠ (a + b + c + 18 - d)
                    ∧ d ≠ (b + c) ∧ d ≠ (c + d)
                    ∧ (a + b + c + 18 + d) ≠ (a + b + c + 18 - d)
                    ∧ (a + b + c + 18 + d) ≠ (b + c) ∧ (a + b + c + 18 + d) ≠ (c + d)
                    ∧ (a + b + c + 18 - d) ≠ (b + c) ∧ (a + b + c + 18 - d) ≠ (c + d)
                    ∧ (b + c) ≠ (c + d)) :
  ∃ max_diff : ℕ, max_diff = 2067 := sorry

end max_prime_difference_l92_92899


namespace prob_A_second_day_is_correct_l92_92974

-- Definitions for the problem conditions
def prob_first_day_A : ℝ := 0.5
def prob_A_given_A : ℝ := 0.6
def prob_first_day_B : ℝ := 0.5
def prob_A_given_B : ℝ := 0.5

-- Calculate the probability of going to A on the second day
def prob_A_second_day : ℝ :=
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B

-- The theorem statement
theorem prob_A_second_day_is_correct : 
  prob_A_second_day = 0.55 :=
by
  unfold prob_A_second_day prob_first_day_A prob_A_given_A prob_first_day_B prob_A_given_B
  sorry

end prob_A_second_day_is_correct_l92_92974


namespace find_third_number_l92_92325

theorem find_third_number (x : ℝ) (third_number : ℝ) : 
  0.6 / 0.96 = third_number / 8 → x = 0.96 → third_number = 5 :=
by
  intro h1 h2
  sorry

end find_third_number_l92_92325


namespace total_pies_eaten_l92_92884

variable (Adam Bill Sierra : ℕ)

axiom condition1 : Adam = Bill + 3
axiom condition2 : Sierra = 2 * Bill
axiom condition3 : Sierra = 12

theorem total_pies_eaten : Adam + Bill + Sierra = 27 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_pies_eaten_l92_92884


namespace octal_to_decimal_conversion_l92_92239

theorem octal_to_decimal_conversion : 
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  7 * d8^0 + 6 * d8^1 + 3 * d8^2 = 247 := 
by
  let d8 := 8
  let f := fun (x: Nat) (y: Nat) => x * d8 ^ y
  sorry

end octal_to_decimal_conversion_l92_92239


namespace solution_set_for_inequality_l92_92158

theorem solution_set_for_inequality :
  {x : ℝ | (1 / (x - 1) ≥ -1)} = {x : ℝ | x ≤ 0 ∨ x > 1} :=
by
  sorry

end solution_set_for_inequality_l92_92158


namespace jill_arrives_earlier_by_30_minutes_l92_92218

theorem jill_arrives_earlier_by_30_minutes :
  ∀ (d : ℕ) (v_jill v_jack : ℕ),
  d = 2 →
  v_jill = 12 →
  v_jack = 3 →
  ((d / v_jack) * 60 - (d / v_jill) * 60) = 30 :=
by
  intros d v_jill v_jack hd hvjill hvjack
  sorry

end jill_arrives_earlier_by_30_minutes_l92_92218


namespace simplify_and_evaluate_l92_92680

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2) :
  ( (1 + x) / (1 - x) / (x - (2 * x / (1 - x))) = - (Real.sqrt 2 + 2) / 2) :=
by
  rw [h]
  simp
  sorry

end simplify_and_evaluate_l92_92680


namespace percentage_increase_l92_92705

def initialProductivity := 120
def totalArea := 1440
def daysInitialProductivity := 2
def daysAheadOfSchedule := 2

theorem percentage_increase :
  let originalDays := totalArea / initialProductivity
  let daysWithIncrease := originalDays - daysAheadOfSchedule
  let daysWithNewProductivity := daysWithIncrease - daysInitialProductivity
  let remainingArea := totalArea - (daysInitialProductivity * initialProductivity)
  let newProductivity := remainingArea / daysWithNewProductivity
  let increase := ((newProductivity - initialProductivity) / initialProductivity) * 100
  increase = 25 :=
by
  sorry

end percentage_increase_l92_92705


namespace margo_donation_l92_92616

variable (M J : ℤ)

theorem margo_donation (h1: J = 4700) (h2: (|J - M| / 2) = 200) : M = 4300 :=
sorry

end margo_donation_l92_92616


namespace probability_of_4_rainy_days_out_of_6_l92_92921

noncomputable def probability_of_rain_on_given_day : ℝ := 0.5

noncomputable def probability_of_rain_on_exactly_k_days (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem probability_of_4_rainy_days_out_of_6 :
  probability_of_rain_on_exactly_k_days 6 4 probability_of_rain_on_given_day = 0.234375 :=
by
  sorry

end probability_of_4_rainy_days_out_of_6_l92_92921


namespace minimum_distance_l92_92323

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

theorem minimum_distance :
  ∃ (A B : ℝ × ℝ), circle_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ dist A B = 1 / 2 :=
sorry

end minimum_distance_l92_92323


namespace sin2θ_over_1pluscos2θ_eq_sqrt3_l92_92931

theorem sin2θ_over_1pluscos2θ_eq_sqrt3 {θ : ℝ} (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin (2 * θ)) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 :=
sorry

end sin2θ_over_1pluscos2θ_eq_sqrt3_l92_92931


namespace arithmetic_and_geometric_mean_l92_92902

theorem arithmetic_and_geometric_mean (a b : ℝ) (h1 : a + b = 40) (h2 : a * b = 100) : a^2 + b^2 = 1400 := by
  sorry

end arithmetic_and_geometric_mean_l92_92902


namespace compute_f_2_neg3_neg1_l92_92684

def f (p q r : ℤ) : ℚ := (r + p : ℚ) / (r - q + 1 : ℚ)

theorem compute_f_2_neg3_neg1 : f 2 (-3) (-1) = 1 / 3 := 
by
  sorry

end compute_f_2_neg3_neg1_l92_92684


namespace positive_integer_triples_l92_92246

theorem positive_integer_triples (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b ∣ (a + 1) ∧ c ∣ (b + 1) ∧ a ∣ (c + 1)) ↔ (a = 1 ∧ b = 1 ∧ c = 1 ∨
  a = 3 ∧ b = 4 ∧ c = 5 ∨ a = 4 ∧ b = 5 ∧ c = 3 ∨ a = 5 ∧ b = 3 ∧ c = 4) :=
by
  sorry

end positive_integer_triples_l92_92246


namespace locus_of_points_is_straight_line_l92_92628

theorem locus_of_points_is_straight_line 
  (a R1 R2 : ℝ) 
  (h_nonzero_a : a ≠ 0)
  (h_positive_R1 : R1 > 0)
  (h_positive_R2 : R2 > 0) :
  ∃ x : ℝ, ∀ (y : ℝ),
  ((x + a)^2 + y^2 - R1^2 = (x - a)^2 + y^2 - R2^2) ↔ 
  x = (R1^2 - R2^2) / (4 * a) :=
by
  sorry

end locus_of_points_is_straight_line_l92_92628


namespace greatest_y_value_l92_92722

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : y ≤ 1 :=
sorry

end greatest_y_value_l92_92722


namespace binary_addition_l92_92248

theorem binary_addition :
  let num1 := 0b111111111
  let num2 := 0b101010101
  num1 + num2 = 852 := by
  sorry

end binary_addition_l92_92248


namespace probability_three_white_balls_l92_92994

def total_balls := 11
def white_balls := 5
def black_balls := 6
def balls_drawn := 5
def white_balls_drawn := 3
def black_balls_drawn := 2

theorem probability_three_white_balls :
  let total_outcomes := Nat.choose total_balls balls_drawn
  let favorable_outcomes := (Nat.choose white_balls white_balls_drawn) * (Nat.choose black_balls black_balls_drawn)
  (favorable_outcomes : ℚ) / total_outcomes = 25 / 77 :=
by
  sorry

end probability_three_white_balls_l92_92994


namespace train_crossing_pole_time_l92_92537

theorem train_crossing_pole_time :
  ∀ (length_of_train : ℝ) (speed_km_per_hr : ℝ) (t : ℝ),
    length_of_train = 45 →
    speed_km_per_hr = 108 →
    t = 1.5 →
    t = length_of_train / (speed_km_per_hr * 1000 / 3600) := 
  sorry

end train_crossing_pole_time_l92_92537


namespace solutions_of_equation_l92_92442

theorem solutions_of_equation :
  ∀ x : ℝ, x * (x - 3) = x - 3 ↔ x = 1 ∨ x = 3 :=
by sorry

end solutions_of_equation_l92_92442


namespace negation_of_exists_l92_92673

theorem negation_of_exists (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x + 1 < 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≥ 0) :=
by sorry

end negation_of_exists_l92_92673


namespace equivalent_proof_problem_l92_92285

noncomputable def perimeter_inner_polygon (pentagon_perimeter : ℕ) : ℕ :=
  let side_length := pentagon_perimeter / 5
  let inner_polygon_sides := 10
  inner_polygon_sides * side_length

theorem equivalent_proof_problem :
  perimeter_inner_polygon 65 = 130 :=
by
  sorry

end equivalent_proof_problem_l92_92285


namespace find_y_z_l92_92691

def abs_diff (x y : ℝ) := abs (x - y)

noncomputable def seq_stabilize (x y z : ℝ) (n : ℕ) : Prop :=
  let x1 := abs_diff x y 
  let y1 := abs_diff y z 
  let z1 := abs_diff z x
  ∃ k : ℕ, k ≥ n ∧ abs_diff x1 y1 = x ∧ abs_diff y1 z1 = y ∧ abs_diff z1 x1 = z

theorem find_y_z (x y z : ℝ) (hx : x = 1) (hstab : ∃ n : ℕ, seq_stabilize x y z n) : y = 0 ∧ z = 0 :=
sorry

end find_y_z_l92_92691


namespace patsy_deviled_eggs_l92_92664

-- Definitions based on given problem conditions
def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def total_appetizers_needed : ℕ := appetizers_per_guest * guests
def pigs_in_blanket : ℕ := 2
def kebabs : ℕ := 2
def additional_appetizers_needed (already_planned : ℕ) : ℕ := 8 + already_planned
def already_planned_appetizers : ℕ := pigs_in_blanket + kebabs
def total_appetizers_planned : ℕ := additional_appetizers_needed already_planned_appetizers

-- The proof problem statement
theorem patsy_deviled_eggs : total_appetizers_needed = total_appetizers_planned * 12 → 
                            total_appetizers_planned = already_planned_appetizers + 8 →
                            (total_appetizers_planned - already_planned_appetizers) = 8 :=
by
  sorry

end patsy_deviled_eggs_l92_92664


namespace burrito_calories_l92_92058

theorem burrito_calories :
  ∀ (C : ℕ), 
  (10 * C = 6 * (250 - 50)) →
  C = 120 :=
by
  intros C h
  sorry

end burrito_calories_l92_92058


namespace circle_condition_iff_l92_92068

-- Given a condition a < 2, we need to show it is a necessary and sufficient condition
-- for the equation x^2 + y^2 - 2x + 2y + a = 0 to represent a circle.

theorem circle_condition_iff (a : ℝ) :
  (∃ (x y : ℝ), (x - 1) ^ 2 + (y + 1) ^ 2 = 2 - a) ↔ (a < 2) :=
sorry

end circle_condition_iff_l92_92068


namespace find_N_value_l92_92872

variable (a b N : ℚ)
variable (h1 : a + 2 * b = N)
variable (h2 : a * b = 4)
variable (h3 : 2 / a + 1 / b = 1.5)

theorem find_N_value : N = 6 :=
by
  sorry

end find_N_value_l92_92872


namespace sally_earnings_proof_l92_92191

def sally_last_month_earnings : ℝ := 1000
def raise_percentage : ℝ := 0.10
def sally_this_month_earnings := sally_last_month_earnings * (1 + raise_percentage)
def sally_total_two_months_earnings := sally_last_month_earnings + sally_this_month_earnings

theorem sally_earnings_proof :
  sally_total_two_months_earnings = 2100 :=
by
  sorry

end sally_earnings_proof_l92_92191


namespace at_least_two_equal_elements_l92_92223

open Function

theorem at_least_two_equal_elements :
  ∀ (k : Fin 10 → Fin 10),
    (∀ i j : Fin 10, i ≠ j → k i ≠ k j) → False :=
by
  intros k h
  sorry

end at_least_two_equal_elements_l92_92223


namespace smallest_number_condition_l92_92405

theorem smallest_number_condition
  (x : ℕ)
  (h1 : (x - 24) % 5 = 0)
  (h2 : (x - 24) % 10 = 0)
  (h3 : (x - 24) % 15 = 0)
  (h4 : (x - 24) / 30 = 84)
  : x = 2544 := 
sorry

end smallest_number_condition_l92_92405


namespace Johnson_potatoes_left_l92_92233

noncomputable def Gina_potatoes : ℝ := 93.5
noncomputable def Tom_potatoes : ℝ := 3.2 * Gina_potatoes
noncomputable def Anne_potatoes : ℝ := (2/3) * Tom_potatoes
noncomputable def Jack_potatoes : ℝ := (1/7) * (Gina_potatoes + Anne_potatoes)
noncomputable def Total_given_away : ℝ := Gina_potatoes + Tom_potatoes + Anne_potatoes + Jack_potatoes
noncomputable def Initial_potatoes : ℝ := 1250
noncomputable def Potatoes_left : ℝ := Initial_potatoes - Total_given_away

theorem Johnson_potatoes_left : Potatoes_left = 615.98 := 
  by
    sorry

end Johnson_potatoes_left_l92_92233


namespace largest_a_for_integer_solution_l92_92988

noncomputable def largest_integer_a : ℤ := 11

theorem largest_a_for_integer_solution :
  ∃ (x : ℤ), ∃ (a : ℤ), 
  (∃ (a : ℤ), a ≤ largest_integer_a) ∧
  (a = largest_integer_a → (
    (x^2 - (a + 7) * x + 7 * a)^3 = -3^3)) := 
by 
  sorry

end largest_a_for_integer_solution_l92_92988


namespace large_jars_count_l92_92534

theorem large_jars_count (S L : ℕ) (h1 : S + L = 100) (h2 : S = 62) (h3 : 3 * S + 5 * L = 376) : L = 38 :=
by
  sorry

end large_jars_count_l92_92534


namespace remainder_of_number_mod_1000_l92_92263

-- Definitions according to the conditions
def num_increasing_8_digit_numbers_with_zero : ℕ := Nat.choose 17 8

-- The main statement to be proved
theorem remainder_of_number_mod_1000 : 
  (num_increasing_8_digit_numbers_with_zero % 1000) = 310 :=
by
  sorry

end remainder_of_number_mod_1000_l92_92263


namespace bookshelf_prices_purchasing_plans_l92_92992

/-
We are given the following conditions:
1. 3 * x + 2 * y = 1020
2. 4 * x + 3 * y = 1440

From these conditions, we need to prove that:
1. Price of type A bookshelf (x) is 180 yuan.
2. Price of type B bookshelf (y) is 240 yuan.

Given further conditions:
1. The school plans to purchase a total of 20 bookshelves.
2. Type B bookshelves not less than type A bookshelves.
3. Maximum budget of 4320 yuan.

We need to prove that the following plans are valid:
1. 8 type A bookshelves, 12 type B bookshelves.
2. 9 type A bookshelves, 11 type B bookshelves.
3. 10 type A bookshelves, 10 type B bookshelves.
-/

theorem bookshelf_prices (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 1020) 
  (h2 : 4 * x + 3 * y = 1440) : 
  x = 180 ∧ y = 240 :=
by sorry

theorem purchasing_plans (m : ℕ) 
  (h3 : 8 ≤ m ∧ m ≤ 10) 
  (h4 : 180 * m + 240 * (20 - m) ≤ 4320) 
  (h5 : 20 - m ≥ m) : 
  m = 8 ∨ m = 9 ∨ m = 10 :=
by sorry

end bookshelf_prices_purchasing_plans_l92_92992


namespace real_roots_in_intervals_l92_92747

theorem real_roots_in_intervals (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x1 x2 : ℝ, (x1 = a / 3 ∨ x1 = -2 * b / 3) ∧ (x2 = a / 3 ∨ x2 = -2 * b / 3) ∧ x1 ≠ x2 ∧
  (a / 3 ≤ x1 ∧ x1 ≤ 2 * a / 3) ∧ (-2 * b / 3 ≤ x2 ∧ x2 ≤ -b / 3) ∧
  (x1 > 0 ∧ x2 < 0) ∧ (1 / x1 + 1 / (x1 - a) + 1 / (x1 + b) = 0) ∧
  (1 / x2 + 1 / (x2 - a) + 1 / (x2 + b) = 0) :=
sorry

end real_roots_in_intervals_l92_92747


namespace complex_real_number_l92_92435

-- Definition of the complex number z
def z (a : ℝ) : ℂ := (a^2 + 2011) + (a - 1) * Complex.I

-- The proof problem statement
theorem complex_real_number (a : ℝ) (h : z a = (a^2 + 2011 : ℂ)) : a = 1 :=
by
  sorry

end complex_real_number_l92_92435


namespace sum_of_cubes_of_roots_eq_1_l92_92271

theorem sum_of_cubes_of_roots_eq_1 (a : ℝ) (x1 x2 : ℝ) :
  (x1^2 + a * x1 + a + 1 = 0) → 
  (x2^2 + a * x2 + a + 1 = 0) → 
  (x1 + x2 = -a) → 
  (x1 * x2 = a + 1) → 
  (x1^3 + x2^3 = 1) → 
  a = -1 :=
sorry

end sum_of_cubes_of_roots_eq_1_l92_92271


namespace complex_ratio_proof_l92_92562

noncomputable def complex_ratio (x y : ℂ) : ℂ :=
  ((x^6 + y^6) / (x^6 - y^6)) - ((x^6 - y^6) / (x^6 + y^6))

theorem complex_ratio_proof (x y : ℂ) (h : ((x - y) / (x + y)) - ((x + y) / (x - y)) = 2) :
  complex_ratio x y = L :=
  sorry

end complex_ratio_proof_l92_92562


namespace value_of_a_plus_one_l92_92220

theorem value_of_a_plus_one (a : ℤ) (h : |a| = 3) : a + 1 = 4 ∨ a + 1 = -2 :=
by
  sorry

end value_of_a_plus_one_l92_92220


namespace percent_of_x_l92_92178

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25 - x / 10 + x / 5) = (16 / 100) * x := by
  sorry

end percent_of_x_l92_92178


namespace max_value_of_f_l92_92134

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 ∧ (f 0 = Real.sin 1 + 1) :=
by
  intro x
  sorry

end max_value_of_f_l92_92134


namespace sarah_annual_income_l92_92287

theorem sarah_annual_income (q : ℝ) (I T : ℝ)
    (h1 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) 
    (h2 : T = 0.01 * (q + 0.5) * I) : 
    I = 36000 := by
  sorry

end sarah_annual_income_l92_92287


namespace radius_of_base_circle_of_cone_l92_92854

theorem radius_of_base_circle_of_cone (θ : ℝ) (r_sector : ℝ) (L : ℝ) (C : ℝ) (r_base : ℝ) :
  θ = 120 ∧ r_sector = 6 ∧ L = (θ / 360) * 2 * Real.pi * r_sector ∧ C = L ∧ C = 2 * Real.pi * r_base → r_base = 2 := by
  sorry

end radius_of_base_circle_of_cone_l92_92854


namespace calculate_expression_l92_92754

theorem calculate_expression : 4 * 6 * 8 + 24 / 4 - 10 = 188 := by
  sorry

end calculate_expression_l92_92754


namespace percentage_of_carnations_is_44_percent_l92_92742

noncomputable def total_flowers : ℕ := sorry
def pink_percentage : ℚ := 2 / 5
def red_percentage : ℚ := 2 / 5
def yellow_percentage : ℚ := 1 / 5
def pink_roses_fraction : ℚ := 2 / 5
def red_carnations_fraction : ℚ := 1 / 2

theorem percentage_of_carnations_is_44_percent
  (F : ℕ)
  (h_pink : pink_percentage * F = 2 / 5 * F)
  (h_red : red_percentage * F = 2 / 5 * F)
  (h_yellow : yellow_percentage * F = 1 / 5 * F)
  (h_pink_roses : pink_roses_fraction * (pink_percentage * F) = 2 / 25 * F)
  (h_red_carnations : red_carnations_fraction * (red_percentage * F) = 1 / 5 * F) :
  ((6 / 25 * F + 5 / 25 * F) / F) * 100 = 44 := sorry

end percentage_of_carnations_is_44_percent_l92_92742


namespace polynomial_condition_l92_92167

theorem polynomial_condition {P : Polynomial ℝ} :
  (∀ (a b c : ℝ), a * b + b * c + c * a = 0 → P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = 2 * P.eval (a + b + c)) →
    ∃ α β : ℝ, P = Polynomial.C α * Polynomial.X^4 + Polynomial.C β * Polynomial.X^2 :=
by
  intro h
  sorry

end polynomial_condition_l92_92167


namespace models_kirsty_can_buy_l92_92796

def original_price : ℝ := 0.45
def saved_for_models : ℝ := 30 * original_price
def new_price : ℝ := 0.50

theorem models_kirsty_can_buy :
  saved_for_models / new_price = 27 :=
sorry

end models_kirsty_can_buy_l92_92796


namespace max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l92_92064

noncomputable def A_excircle_area_ratio (α : Real) (s : Real) : Real :=
  0.5 * Real.sin α

theorem max_A_excircle_area_ratio (α : Real) (s : Real) : (A_excircle_area_ratio α s) ≤ 0.5 :=
by
  sorry

theorem max_A_excircle_area_ratio_eq (s : Real) : 
  (A_excircle_area_ratio (Real.pi / 2) s) = 0.5 :=
by
  sorry

end max_A_excircle_area_ratio_max_A_excircle_area_ratio_eq_l92_92064


namespace f_at_neg_one_l92_92152

def f : ℝ → ℝ := sorry

theorem f_at_neg_one :
  (∀ x : ℝ, f (x / (1 + x)) = x) →
  f (-1) = -1 / 2 :=
by
  intro h
  -- proof omitted for clarity
  sorry

end f_at_neg_one_l92_92152


namespace three_fifths_difference_products_l92_92491

theorem three_fifths_difference_products :
  (3 / 5) * ((7 * 9) - (4 * 3)) = 153 / 5 :=
by
  sorry

end three_fifths_difference_products_l92_92491


namespace find_b2_a2_a1_l92_92767

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def geometric_sequence (b : ℕ → ℝ) : Prop :=
∀ n : ℕ, b (n + 1) / b n = b 1 / b 0

theorem find_b2_a2_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a1 : a 0 = a₁) (h_a2 : a 2 = a₂)
  (h_b2 : b 2 = b₂) :
  b₂ * (a₂ - a₁) = 6 ∨ b₂ * (a₂ - a₁) = -6 :=
by
  sorry

end find_b2_a2_a1_l92_92767


namespace inequality_abc_equality_condition_l92_92198

theorem inequality_abc (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) ≥ 12 :=
sorry

theorem equality_condition (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_abc_equality_condition_l92_92198


namespace problem_solution_eq_l92_92991

theorem problem_solution_eq : 
  { x : ℝ | (x ^ 2 - 9) / (x ^ 2 - 1) > 0 } = { x : ℝ | x > 3 ∨ x < -3 } :=
by
  sorry

end problem_solution_eq_l92_92991


namespace convert_13_to_binary_l92_92346

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem convert_13_to_binary : decimal_to_binary 13 = [1, 1, 0, 1] :=
  by
    sorry -- Proof to be provided

end convert_13_to_binary_l92_92346


namespace candle_height_problem_l92_92943

/-- Define the height functions of the two candles. -/
def h1 (t : ℚ) : ℚ := 1 - t / 5
def h2 (t : ℚ) : ℚ := 1 - t / 4

/-- The main theorem stating the time t when the first candle is three times the height of the second candle. -/
theorem candle_height_problem : 
  (∀ t : ℚ, h1 t = 3 * h2 t) → t = (40 : ℚ) / 11 :=
by
  sorry

end candle_height_problem_l92_92943


namespace factorization_correct_l92_92390

theorem factorization_correct (x y : ℝ) :
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2) * (x^2 - 3*y + 2) :=
by
  sorry

end factorization_correct_l92_92390


namespace effective_annual_rate_of_interest_l92_92822

theorem effective_annual_rate_of_interest 
  (i : ℝ) (n : ℕ) (h_i : i = 0.10) (h_n : n = 2) : 
  (1 + i / n)^n - 1 = 0.1025 :=
by
  sorry

end effective_annual_rate_of_interest_l92_92822


namespace geometric_sequence_a3_l92_92541

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a3 
  (a : ℕ → ℝ) (h1 : a 1 = -2) (h5 : a 5 = -8)
  (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) : 
  a 3 = -4 :=
sorry

end geometric_sequence_a3_l92_92541


namespace focus_of_parabola_l92_92054

theorem focus_of_parabola (x y : ℝ) : (y^2 = 4 * x) → (x = 2 ∧ y = 0) :=
by
  sorry

end focus_of_parabola_l92_92054


namespace graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l92_92586

theorem graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines :
  ∀ x y : ℝ, (x^2 - y^2 = 0) ↔ (y = x ∨ y = -x) := 
by
  sorry

end graph_of_x2_minus_y2_eq_0_is_two_intersecting_lines_l92_92586


namespace reduced_price_per_dozen_apples_l92_92156

variables (P R : ℝ) 

theorem reduced_price_per_dozen_apples (h₁ : R = 0.70 * P) 
  (h₂ : (30 / P + 54) * R = 30) :
  12 * R = 2 := 
sorry

end reduced_price_per_dozen_apples_l92_92156


namespace intersecting_graphs_l92_92373

theorem intersecting_graphs (a b c d : ℝ) (h₁ : (3, 6) = (3, -|3 - a| + b))
  (h₂ : (9, 2) = (9, -|9 - a| + b))
  (h₃ : (3, 6) = (3, |3 - c| + d))
  (h₄ : (9, 2) = (9, |9 - c| + d)) : 
  a + c = 12 := 
sorry

end intersecting_graphs_l92_92373


namespace weight_of_A_l92_92864

theorem weight_of_A (A B C D E : ℝ) 
  (h1 : (A + B + C) / 3 = 84) 
  (h2 : (A + B + C + D) / 4 = 80) 
  (h3 : (B + C + D + E) / 4 = 79) 
  (h4 : E = D + 7): 
  A = 79 := by
  have h5 : A + B + C = 252 := by
    linarith [h1]
  have h6 : A + B + C + D = 320 := by
    linarith [h2]
  have h7 : B + C + D + E = 316 := by
    linarith [h3]
  have hD : D = 68 := by
    linarith [h5, h6]
  have hE : E = 75 := by
    linarith [hD, h4]
  have hBC : B + C = 252 - A := by
    linarith [h5]
  have : 252 - A + 68 + 75 = 316 := by
    linarith [h7, hBC, hD, hE]
  linarith

end weight_of_A_l92_92864


namespace compare_exponents_l92_92354

theorem compare_exponents (n : ℕ) (hn : n > 8) :
  let a := Real.sqrt n
  let b := Real.sqrt (n + 1)
  a^b > b^a :=
sorry

end compare_exponents_l92_92354


namespace greatest_possible_x_l92_92029

theorem greatest_possible_x (x : ℕ) (h : x^3 < 15) : x ≤ 2 := by
  sorry

end greatest_possible_x_l92_92029


namespace bakery_batches_per_day_l92_92122

-- Definitions for the given problem's conditions
def baguettes_per_batch := 48
def baguettes_sold_batch1 := 37
def baguettes_sold_batch2 := 52
def baguettes_sold_batch3 := 49
def baguettes_left := 6

-- Theorem stating the number of batches made
theorem bakery_batches_per_day : 
  (baguettes_sold_batch1 + baguettes_sold_batch2 + baguettes_sold_batch3 + baguettes_left) / baguettes_per_batch = 3 :=
by 
  sorry

end bakery_batches_per_day_l92_92122


namespace complement_U_P_l92_92297

theorem complement_U_P :
  let U := {y : ℝ | y ≠ 0 }
  let P := {y : ℝ | 0 < y ∧ y < 1/2}
  let complement_U_P := {y : ℝ | y ∈ U ∧ y ∉ P}
  (complement_U_P = {y : ℝ | y < 0} ∪ {y : ℝ | y > 1/2}) :=
by
  sorry

end complement_U_P_l92_92297


namespace price_of_baseball_bat_l92_92296

theorem price_of_baseball_bat 
  (price_A : ℕ) (price_B : ℕ) (price_bat : ℕ) 
  (hA : price_A = 10 * 29)
  (hB : price_B = 14 * (25 / 10))
  (h0 : price_A = price_B + price_bat + 237) :
  price_bat = 18 :=
by
  sorry

end price_of_baseball_bat_l92_92296


namespace inequality_2_inequality_1_9_l92_92531

variables {a : ℕ → ℝ}

-- Conditions
def non_negative (a : ℕ → ℝ) : Prop := ∀ n, a n ≥ 0
def boundary_zero (a : ℕ → ℝ) : Prop := a 1 = 0 ∧ a 9 = 0
def non_zero_interior (a : ℕ → ℝ) : Prop := ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a i ≠ 0

-- Proof problems
theorem inequality_2 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 2 * a i := sorry

theorem inequality_1_9 (a : ℕ → ℝ) (h1 : non_negative a) (h2 : boundary_zero a) (h3 : non_zero_interior a) :
  ∃ i, 2 ≤ i ∧ i ≤ 8 ∧ a (i - 1) + a (i + 1) < 1.9 * a i := sorry

end inequality_2_inequality_1_9_l92_92531


namespace time_to_pass_l92_92874
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

end time_to_pass_l92_92874


namespace largest_of_five_consecutive_integers_with_product_15120_is_9_l92_92247

theorem largest_of_five_consecutive_integers_with_product_15120_is_9 :
  ∃ (a b c d e : ℤ), a * b * c * d * e = 15120 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e = 9 :=
sorry

end largest_of_five_consecutive_integers_with_product_15120_is_9_l92_92247


namespace exam_max_incorrect_answers_l92_92139

theorem exam_max_incorrect_answers :
  ∀ (c w b : ℕ),
  (c + w + b = 30) →
  (4 * c - w ≥ 85) → 
  (c ≥ 22) →
  (w ≤ 3) :=
by
  intros c w b h1 h2 h3
  sorry

end exam_max_incorrect_answers_l92_92139


namespace number_of_sheep_total_number_of_animals_l92_92288

theorem number_of_sheep (ratio_sh_horse : 5 / 7 * horses = sheep) 
    (horse_food_per_day : horses * 230 = 12880) :
    sheep = 40 :=
by
  -- These are all the given conditions
  sorry

theorem total_number_of_animals (sheep : ℕ) (horses : ℕ)
    (H1 : sheep = 40) (H2 : horses = 56) :
    sheep + horses = 96 :=
by
  -- Given conditions for the total number of animals on the farm
  sorry

end number_of_sheep_total_number_of_animals_l92_92288


namespace problem_1_problem_2_l92_92283

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l92_92283


namespace triangle_angle_not_greater_than_60_l92_92798

theorem triangle_angle_not_greater_than_60 (A B C : ℝ) (h1 : A + B + C = 180) :
  A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
sorry -- proof by contradiction to be implemented here

end triangle_angle_not_greater_than_60_l92_92798


namespace sequence_geometric_and_general_formula_find_minimum_n_l92_92437

theorem sequence_geometric_and_general_formula 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) :=
sorry

theorem find_minimum_n 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (b T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n)
  (h2 : ∀ n : ℕ, b n = (2 * n + 1) * a n + (2 * n + 1))
  (h3 : T 0 = 0)
  (h4 : ∀ n : ℕ, T (n + 1) = T n + b (n + 1)) :
  ∃ n : ℕ, n ≥ 1 ∧ (T n - 2) / (2 * n - 1) > 2010 :=
sorry

end sequence_geometric_and_general_formula_find_minimum_n_l92_92437


namespace dima_and_serezha_meet_time_l92_92760

-- Define the conditions and the main theorem to be proven.
theorem dima_and_serezha_meet_time :
  let dima_run_time := 15 / 60.0 -- Dima runs for 15 minutes
  let dima_run_speed := 6.0 -- Dima's running speed is 6 km/h
  let serezha_boat_speed := 20.0 -- Serezha's boat speed is 20 km/h
  let serezha_boat_time := 30 / 60.0 -- Serezha's boat time is 30 minutes
  let common_run_speed := 6.0 -- Both run at 6 km/h towards each other
  let distance_to_meet := dima_run_speed * dima_run_time -- Distance Dima runs along the shore
  let total_time := distance_to_meet / (common_run_speed + common_run_speed) -- Time until they meet after parting
  total_time = 7.5 / 60.0 := -- 7.5 minutes converted to hours
sorry

end dima_and_serezha_meet_time_l92_92760


namespace combined_cost_is_450_l92_92550

-- Given conditions
def bench_cost : ℕ := 150
def table_cost : ℕ := 2 * bench_cost

-- The statement we want to prove
theorem combined_cost_is_450 : bench_cost + table_cost = 450 :=
by
  sorry

end combined_cost_is_450_l92_92550


namespace second_batch_students_l92_92429

theorem second_batch_students :
  ∃ x : ℕ,
    (40 * 45 + x * 55 + 60 * 65 : ℝ) / (40 + x + 60) = 56.333333333333336 ∧
    x = 50 :=
by
  use 50
  sorry

end second_batch_students_l92_92429


namespace exist_divisible_n_and_n1_l92_92484

theorem exist_divisible_n_and_n1 (d : ℕ) (hd : 0 < d) :
  ∃ (n n1 : ℕ), n % d = 0 ∧ n1 % d = 0 ∧ n ≠ n1 ∧
  (∃ (k a b c : ℕ), b ≠ 0 ∧ n = 10^k * (10 * a + b) + c ∧ n1 = 10^k * a + c) :=
by
  sorry

end exist_divisible_n_and_n1_l92_92484


namespace units_digit_of_3_pow_4_l92_92251

theorem units_digit_of_3_pow_4 : (3^4 % 10) = 1 :=
by
  sorry

end units_digit_of_3_pow_4_l92_92251


namespace remaining_pictures_l92_92061

theorem remaining_pictures (k m : ℕ) (d1 := 9 * k + 4) (d2 := 9 * m + 6) :
  (d1 * d2) % 9 = 6 → 9 - (d1 * d2 % 9) = 3 :=
by
  intro h
  sorry

end remaining_pictures_l92_92061


namespace number_of_black_balls_l92_92589

variable (T : ℝ)
variable (red_balls : ℝ := 21)
variable (prop_red : ℝ := 0.42)
variable (prop_white : ℝ := 0.28)
variable (white_balls : ℝ := 0.28 * T)

noncomputable def total_balls : ℝ := red_balls / prop_red

theorem number_of_black_balls :
  T = total_balls → 
  ∃ black_balls : ℝ, black_balls = total_balls - red_balls - white_balls ∧ black_balls = 15 := 
by
  intro hT
  let black_balls := total_balls - red_balls - white_balls
  use black_balls
  simp [total_balls]
  sorry

end number_of_black_balls_l92_92589


namespace number_of_thrown_out_carrots_l92_92494

-- Definitions from the conditions
def initial_carrots : ℕ := 48
def picked_next_day : ℕ := 42
def total_carrots : ℕ := 45

-- Proposition stating the problem
theorem number_of_thrown_out_carrots (x : ℕ) : initial_carrots - x + picked_next_day = total_carrots → x = 45 :=
by
  sorry

end number_of_thrown_out_carrots_l92_92494


namespace problem1_l92_92236

theorem problem1 (a b : ℝ) (ha : a > 2) (hb : b > 2) :
  (a - 2) * (b - 2) = 2 :=
sorry

end problem1_l92_92236


namespace series_sum_equals_three_fourths_l92_92846

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end series_sum_equals_three_fourths_l92_92846


namespace skaters_total_hours_l92_92932

-- Define the practice hours based on the conditions
def hannah_weekend_hours := 8
def hannah_weekday_extra_hours := 17
def sarah_weekday_hours := 12
def sarah_weekend_hours := 6
def emma_weekday_hour_multiplier := 2
def emma_weekend_hour_extra := 5

-- Hannah's total hours
def hannah_weekday_hours := hannah_weekend_hours + hannah_weekday_extra_hours
def hannah_total_hours := hannah_weekend_hours + hannah_weekday_hours

-- Sarah's total hours
def sarah_total_hours := sarah_weekday_hours + sarah_weekend_hours

-- Emma's total hours
def emma_weekday_hours := emma_weekday_hour_multiplier * sarah_weekday_hours
def emma_weekend_hours := sarah_weekend_hours + emma_weekend_hour_extra
def emma_total_hours := emma_weekday_hours + emma_weekend_hours

-- Total hours for all three skaters combined
def total_hours := hannah_total_hours + sarah_total_hours + emma_total_hours

-- Lean statement version only, no proof required
theorem skaters_total_hours : total_hours = 86 := by
  sorry

end skaters_total_hours_l92_92932


namespace ludwig_weekly_salary_is_55_l92_92933

noncomputable def daily_salary : ℝ := 10
noncomputable def full_days : ℕ := 4
noncomputable def half_days : ℕ := 3
noncomputable def half_day_salary := daily_salary / 2

theorem ludwig_weekly_salary_is_55 :
  (full_days * daily_salary + half_days * half_day_salary = 55) := by
  sorry

end ludwig_weekly_salary_is_55_l92_92933


namespace determinant_condition_l92_92914

variable (p q r s : ℝ)

theorem determinant_condition (h: p * s - q * r = 5) :
  p * (5 * r + 4 * s) - r * (5 * p + 4 * q) = 20 :=
by
  sorry

end determinant_condition_l92_92914


namespace proposition_true_l92_92403

theorem proposition_true (x y : ℝ) : x + 2 * y ≠ 5 → (x ≠ 1 ∨ y ≠ 2) :=
by
  sorry

end proposition_true_l92_92403


namespace carrots_remaining_l92_92344

theorem carrots_remaining 
  (total_carrots : ℕ)
  (weight_20_carrots : ℕ)
  (removed_carrots : ℕ)
  (avg_weight_remaining : ℕ)
  (avg_weight_removed : ℕ)
  (h1 : total_carrots = 20)
  (h2 : weight_20_carrots = 3640)
  (h3 : removed_carrots = 4)
  (h4 : avg_weight_remaining = 180)
  (h5 : avg_weight_removed = 190) :
  total_carrots - removed_carrots = 16 :=
by 
  -- h1 : 20 carrots in total
  -- h2 : total weight of 20 carrots is 3640 grams
  -- h3 : 4 carrots are removed
  -- h4 : average weight of remaining carrots is 180 grams
  -- h5 : average weight of removed carrots is 190 grams
  sorry

end carrots_remaining_l92_92344


namespace num_of_valid_m_vals_l92_92145

theorem num_of_valid_m_vals : 
  (∀ m x : ℤ, (x + m ≤ 4 ∧ (x / 2 - (x - 1) / 4 > 1 → x > 3 → ∃ (c : ℚ), (x + 1)/4 > 1 )) ∧
  (∃ (x : ℤ), (x + m ≤ 4 ∧ (x > 3) ∧ (m < 1 ∧ m > -4)) ∧ 
  ∃ a b : ℚ, x^2 + a * x + b = 0) → 
  (∃ (count m : ℤ), count = 2)) :=
sorry

end num_of_valid_m_vals_l92_92145


namespace oranges_in_bin_after_changes_l92_92891

def initial_oranges := 31
def thrown_away_oranges := 9
def new_oranges := 38

theorem oranges_in_bin_after_changes : 
  initial_oranges - thrown_away_oranges + new_oranges = 60 := by
  sorry

end oranges_in_bin_after_changes_l92_92891


namespace min_value_fraction_sum_l92_92318

theorem min_value_fraction_sum (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_sum_l92_92318


namespace abe_age_sum_is_31_l92_92903

-- Define the present age of Abe
def abe_present_age : ℕ := 19

-- Define Abe's age 7 years ago
def abe_age_7_years_ago : ℕ := abe_present_age - 7

-- Define the sum of Abe's present age and his age 7 years ago
def abe_age_sum : ℕ := abe_present_age + abe_age_7_years_ago

-- Prove that the sum is 31
theorem abe_age_sum_is_31 : abe_age_sum = 31 := 
by 
  sorry

end abe_age_sum_is_31_l92_92903


namespace mean_of_added_numbers_l92_92071

theorem mean_of_added_numbers (mean_seven : ℝ) (mean_ten : ℝ) (x y z : ℝ)
    (h1 : mean_seven = 40)
    (h2 : mean_ten = 55) :
    (mean_seven * 7 + x + y + z) / 10 = mean_ten → (x + y + z) / 3 = 90 :=
by sorry

end mean_of_added_numbers_l92_92071


namespace problem_statement_l92_92551

variable (a b c d x : ℕ)

theorem problem_statement
  (h1 : a + b = x)
  (h2 : b + c = 9)
  (h3 : c + d = 3)
  (h4 : a + d = 6) :
  x = 12 :=
by
  sorry

end problem_statement_l92_92551


namespace radiator_water_fraction_l92_92896

theorem radiator_water_fraction :
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  fraction_remaining_per_replacement^4 = 81 / 256 := by
  let initial_volume := 20
  let replacement_volume := 5
  let fraction_remaining_per_replacement := (initial_volume - replacement_volume) / initial_volume
  sorry

end radiator_water_fraction_l92_92896


namespace john_spends_40_dollars_l92_92208

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end john_spends_40_dollars_l92_92208


namespace total_ticket_cost_is_correct_l92_92909

-- Definitions based on the conditions provided
def child_ticket_cost : ℝ := 4.25
def adult_ticket_cost : ℝ := child_ticket_cost + 3.50
def senior_ticket_cost : ℝ := adult_ticket_cost - 1.75

def number_adult_tickets : ℕ := 2
def number_child_tickets : ℕ := 4
def number_senior_tickets : ℕ := 1

def total_ticket_cost_before_discount : ℝ := 
  number_adult_tickets * adult_ticket_cost + 
  number_child_tickets * child_ticket_cost + 
  number_senior_tickets * senior_ticket_cost

def total_tickets : ℕ := number_adult_tickets + number_child_tickets + number_senior_tickets
def discount : ℝ := if total_tickets >= 5 then 3.0 else 0.0

def total_ticket_cost_after_discount : ℝ := total_ticket_cost_before_discount - discount

-- The proof statement: proving the total ticket cost after the discount is $35.50
theorem total_ticket_cost_is_correct : total_ticket_cost_after_discount = 35.50 := by
  -- Note: The exact solution is omitted and replaced with sorry to denote where the proof would be.
  sorry

end total_ticket_cost_is_correct_l92_92909


namespace find_x_l92_92607

theorem find_x (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) (h3 : a^3 = 21 * x * 15 * b) : x = 25 :=
by
  -- This is where the proof would go
  sorry

end find_x_l92_92607


namespace football_players_count_l92_92799

def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def softball_players : ℕ := 13
def total_players : ℕ := 59

theorem football_players_count :
  total_players - (cricket_players + hockey_players + softball_players) = 18 :=
by 
  sorry

end football_players_count_l92_92799


namespace original_population_l92_92660

theorem original_population (p: ℝ) :
  (p + 1500) * 0.85 = p - 45 -> p = 8800 :=
by
  sorry

end original_population_l92_92660


namespace two_pow_n_plus_one_divisible_by_three_l92_92814

-- defining what it means to be an odd number
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- stating the main theorem in Lean
theorem two_pow_n_plus_one_divisible_by_three (n : ℕ) (h_pos : 0 < n) : (2^n + 1) % 3 = 0 ↔ is_odd n :=
by sorry

end two_pow_n_plus_one_divisible_by_three_l92_92814


namespace eval_operations_l92_92636

def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

theorem eval_operations : star (star 6 8) (hash 3 5) = 26 := by
  sorry

end eval_operations_l92_92636


namespace common_remainder_proof_l92_92140

def least_subtracted := 6
def original_number := 1439
def reduced_number := original_number - least_subtracted
def divisors := [5, 11, 13]
def common_remainder := 3

theorem common_remainder_proof :
  ∀ d ∈ divisors, reduced_number % d = common_remainder := by
  sorry

end common_remainder_proof_l92_92140


namespace circle_chord_area_l92_92244

noncomputable def part_circle_area_between_chords (R : ℝ) : ℝ :=
  (R^2 * (Real.pi + Real.sqrt 3)) / 2

theorem circle_chord_area (R : ℝ) :
  ∀ (a₃ a₆ : ℝ),
    a₃ = Real.sqrt 3 * R →
    a₆ = R →
    part_circle_area_between_chords R = (R^2 * (Real.pi + Real.sqrt 3)) / 2 :=
by
  intros a₃ a₆ h₁ h₂
  sorry

end circle_chord_area_l92_92244


namespace sufficient_but_not_necessary_l92_92788

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4) ∧ ¬((a + b > 4 ∧ a * b > 4) → (a > 2 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l92_92788


namespace max_discarded_grapes_l92_92441

theorem max_discarded_grapes (n : ℕ) : ∃ r, r < 8 ∧ n % 8 = r ∧ r = 7 :=
by
  sorry

end max_discarded_grapes_l92_92441


namespace shaded_region_area_l92_92372

theorem shaded_region_area
  (R r : ℝ)
  (h : r^2 = R^2 - 2500)
  : π * (R^2 - r^2) = 2500 * π :=
by
  sorry

end shaded_region_area_l92_92372


namespace distinct_pairwise_products_l92_92871

theorem distinct_pairwise_products
  (n a b c d : ℕ) (h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_bounds: n^2 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2) :
  (a * b ≠ a * c ∧ a * b ≠ a * d ∧ a * b ≠ b * c ∧ a * b ≠ b * d ∧ a * b ≠ c * d) ∧
  (a * c ≠ a * d ∧ a * c ≠ b * c ∧ a * c ≠ b * d ∧ a * c ≠ c * d) ∧
  (a * d ≠ b * c ∧ a * d ≠ b * d ∧ a * d ≠ c * d) ∧
  (b * c ≠ b * d ∧ b * c ≠ c * d) ∧
  (b * d ≠ c * d) :=
sorry

end distinct_pairwise_products_l92_92871


namespace largest_base4_is_largest_l92_92571

theorem largest_base4_is_largest 
  (n1 : ℕ) (n2 : ℕ) (n3 : ℕ) (n4 : ℕ)
  (h1 : n1 = 31) (h2 : n2 = 52) (h3 : n3 = 54) (h4 : n4 = 46) :
  n3 = Nat.max (Nat.max n1 n2) (Nat.max n3 n4) :=
by
  sorry

end largest_base4_is_largest_l92_92571


namespace total_parts_in_order_l92_92526

theorem total_parts_in_order (total_cost : ℕ) (cost_20 : ℕ) (cost_50 : ℕ) (num_50_dollar_parts : ℕ) (num_20_dollar_parts : ℕ) :
  total_cost = 2380 → cost_20 = 20 → cost_50 = 50 → num_50_dollar_parts = 40 → (total_cost = num_50_dollar_parts * cost_50 + num_20_dollar_parts * cost_20) → (num_50_dollar_parts + num_20_dollar_parts = 59) :=
by
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end total_parts_in_order_l92_92526


namespace ratio_of_wins_l92_92849

-- Definitions based on conditions
def W1 : ℕ := 15  -- Number of wins before first loss
def L : ℕ := 2    -- Total number of losses
def W2 : ℕ := 30 - W1  -- Calculate W2 based on W1 and total wins being 28 more than losses

-- Theorem statement: Prove the ratio of wins after her first loss to wins before her first loss is 1:1
theorem ratio_of_wins (h : W1 = 15 ∧ L = 2) : W2 / W1 = 1 := by
  sorry

end ratio_of_wins_l92_92849


namespace not_divisible_by_121_l92_92312

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 12)) :=
sorry

end not_divisible_by_121_l92_92312


namespace seating_arrangements_l92_92950

/-
Given:
1. There are 8 students.
2. Four different classes: (1), (2), (3), and (4).
3. Each class has 2 students.
4. There are 2 cars, Car A and Car B, each with a capacity for 4 students.
5. The two students from Class (1) (twin sisters) must ride in the same car.

Prove:
The total number of ways to seat the students such that exactly 2 students from the same class are in Car A is 24.
-/

theorem seating_arrangements : 
  ∃ (arrangements : ℕ), arrangements = 24 :=
sorry

end seating_arrangements_l92_92950


namespace find_n_l92_92119

variable (a b c n : ℤ)
variable (h1 : a + b + c = 100)
variable (h2 : a + b / 2 = 40)

theorem find_n : n = a - c := by
  sorry

end find_n_l92_92119


namespace find_number_l92_92648

theorem find_number (x : ℤ) (N : ℤ) (h1 : 3 * x = (N - x) + 18) (hx : x = 11) : N = 26 :=
by
  sorry

end find_number_l92_92648


namespace sculpture_cost_in_CNY_l92_92873

theorem sculpture_cost_in_CNY (USD_to_NAD USD_to_CNY cost_NAD : ℝ) :
  USD_to_NAD = 8 → USD_to_CNY = 5 → cost_NAD = 160 → (cost_NAD * (1 / USD_to_NAD) * USD_to_CNY) = 100 :=
by
  intros h1 h2 h3
  sorry

end sculpture_cost_in_CNY_l92_92873


namespace correct_negation_of_exactly_one_even_l92_92466

-- Define a predicate to check if a natural number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a predicate to check if a natural number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem statement in Lean
theorem correct_negation_of_exactly_one_even (a b c : ℕ) :
  ¬ ( (is_even a ∧ is_odd b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_even b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_odd b ∧ is_even c) ) ↔ 
  ( (is_odd a ∧ is_odd b ∧ is_odd c) ∨ 
    (is_even a ∧ is_even b ∧ is_even c) ) :=
by 
  sorry

end correct_negation_of_exactly_one_even_l92_92466


namespace QT_value_l92_92910

noncomputable def find_QT (PQ RS PT : ℝ) : ℝ :=
  let tan_gamma := (RS / PQ)
  let QT := (RS / tan_gamma) - PT
  QT

theorem QT_value :
  let PQ := 45
  let RS := 75
  let PT := 15
  find_QT PQ RS PT = 210 := by
  sorry

end QT_value_l92_92910


namespace simplify_exponent_expression_l92_92188

theorem simplify_exponent_expression (n : ℕ) :
  (3^(n+4) - 3 * 3^n) / (3 * 3^(n+3)) = 26 / 9 := by
  sorry

end simplify_exponent_expression_l92_92188


namespace quadratic_real_roots_l92_92385

theorem quadratic_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ (x^2 - m * x + (m - 1) = 0) ∧ (y^2 - m * y + (m - 1) = 0) 
  ∨ ∃ z : ℝ, (z^2 - m * z + (m - 1) = 0) := 
sorry

end quadratic_real_roots_l92_92385


namespace range_of_k_l92_92243

theorem range_of_k (k : ℝ) :
  (∃ a b c : ℝ, (a = 1) ∧ (b = -1) ∧ (c = -k) ∧ (b^2 - 4 * a * c > 0)) ↔ k > -1 / 4 :=
by
  sorry

end range_of_k_l92_92243


namespace find_b_l92_92257

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3 * x^2
noncomputable def tangent_line (x b : ℝ) : ℝ := -3 * x + b

theorem find_b
  (b : ℝ)
  (h : ∃ x : ℝ, curve x = tangent_line x b ∧ deriv curve x = -3) :
  b = 1 :=
by
  sorry

end find_b_l92_92257


namespace smallest_value_a_b_l92_92136

theorem smallest_value_a_b (a b : ℕ) (h : 2^6 * 3^9 = a^b) : a > 0 ∧ b > 0 ∧ (a + b = 111) :=
by
  sorry

end smallest_value_a_b_l92_92136


namespace cubic_and_quintic_values_l92_92696

theorem cubic_and_quintic_values (a : ℝ) (h : (a + 1/a)^2 = 11) : 
    (a^3 + 1/a^3 = 8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = 71 * Real.sqrt 11) ∨ 
    (a^3 + 1/a^3 = -8 * Real.sqrt 11 ∧ a^5 + 1/a^5 = -71 * Real.sqrt 11) :=
by
  sorry

end cubic_and_quintic_values_l92_92696


namespace length_of_PS_l92_92085

noncomputable def triangle_segments : ℝ := 
  let PR := 15
  let ratio_PS_SR := 3 / 4
  let total_length := 15
  let SR := total_length / (1 + ratio_PS_SR)
  let PS := ratio_PS_SR * SR
  PS

theorem length_of_PS :
  triangle_segments = 45 / 7 :=
by
  sorry

end length_of_PS_l92_92085


namespace smallest_n_satisfying_conditions_l92_92904

variable (n : ℕ)
variable (h1 : 100 ≤ n ∧ n < 1000)
variable (h2 : (n + 7) % 6 = 0)
variable (h3 : (n - 5) % 9 = 0)

theorem smallest_n_satisfying_conditions : n = 113 := by
  sorry

end smallest_n_satisfying_conditions_l92_92904


namespace sum_of_integers_75_to_95_l92_92273

def arithmeticSumOfIntegers (a l : ℕ) : ℕ :=
  let n := l - a + 1
  n / 2 * (a + l)

theorem sum_of_integers_75_to_95 : arithmeticSumOfIntegers 75 95 = 1785 :=
  by
  sorry

end sum_of_integers_75_to_95_l92_92273


namespace problem_equivalence_l92_92971

noncomputable def f (a b x : ℝ) : ℝ := a ^ x + b

theorem problem_equivalence (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : f a b 0 = -2) (h4 : f a b 2 = 0) :
    a = Real.sqrt 3 ∧ b = -3 ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 4, (-8 / 3 : ℝ) ≤ f a b x ∧ f a b x ≤ 6) :=
sorry

end problem_equivalence_l92_92971


namespace b7_value_l92_92328

theorem b7_value (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h₀a : a 0 = 3) (h₀b : b 0 = 4)
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 / b n)
  (h₂ : ∀ n, b (n + 1) = b n ^ 2 / a n) :
  b 7 = 4 ^ 730 / 3 ^ 1093 :=
by
  sorry

end b7_value_l92_92328


namespace ratio_preference_l92_92124

-- Definitions based on conditions
def total_respondents : ℕ := 180
def preferred_brand_x : ℕ := 150
def preferred_brand_y : ℕ := total_respondents - preferred_brand_x

-- Theorem statement to prove the ratio of preferences
theorem ratio_preference : preferred_brand_x / preferred_brand_y = 5 := by
  sorry

end ratio_preference_l92_92124


namespace paintings_total_l92_92757

def june_paintings : ℕ := 2
def july_paintings : ℕ := 2 * june_paintings
def august_paintings : ℕ := 3 * july_paintings
def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem paintings_total : total_paintings = 18 :=
by {
  sorry
}

end paintings_total_l92_92757


namespace cube_division_l92_92371

theorem cube_division (n : ℕ) (hn1 : 6 ≤ n) (hn2 : n % 2 = 0) : 
  ∃ m : ℕ, (n = 2 * m) ∧ (∀ a : ℕ, ∀ b : ℕ, ∀ c: ℕ, a = m^3 - (m - 1)^3 + 1 → b = 3 * m * (m - 1) + 2 → a = b) :=
by
  sorry

end cube_division_l92_92371


namespace f_5_eq_25sqrt5_l92_92765

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : Continuous f
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_2 : f 2 = 5

theorem f_5_eq_25sqrt5 : f 5 = 25 * Real.sqrt 5 := by
  sorry

end f_5_eq_25sqrt5_l92_92765


namespace min_value_condition_l92_92088

variable (a b : ℝ)

theorem min_value_condition (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 2) :
    (1 / a^2) + (1 / b^2) = 9 / 2 :=
sorry

end min_value_condition_l92_92088


namespace geometric_sequence_increasing_neither_sufficient_nor_necessary_l92_92228

-- Definitions based on the conditions
def is_geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop := ∀ n, a (n + 1) = a n * q
def is_increasing_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) > a n

-- Define the main theorem according to the problem statement
theorem geometric_sequence_increasing_neither_sufficient_nor_necessary (a : ℕ → ℝ) (a1 q : ℝ) 
  (h_geom : is_geometric_sequence a a1 q) :
  ¬ ( ( (∀ (h : a1 * q > 0), is_increasing_sequence a) ∨ 
        (∀ (h : is_increasing_sequence a), a1 * q > 0) ) ) :=
sorry

end geometric_sequence_increasing_neither_sufficient_nor_necessary_l92_92228


namespace water_depth_in_cylindrical_tub_l92_92245

theorem water_depth_in_cylindrical_tub
  (tub_diameter : ℝ) (tub_depth : ℝ) (pail_angle : ℝ)
  (h_diam : tub_diameter = 40)
  (h_depth : tub_depth = 50)
  (h_angle : pail_angle = 45) :
  ∃ water_depth : ℝ, water_depth = 30 :=
by
  sorry

end water_depth_in_cylindrical_tub_l92_92245


namespace solution_l92_92736

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)

axiom periodic_f : ∀ x : ℝ, f (x - 3) = - f x

axiom increasing_f_on_interval : ∀ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ x1 ≠ x2) → (f x1 - f x2) / (x1 - x2) > 0

theorem solution : f 49 < f 64 ∧ f 64 < f 81 :=
by
  sorry

end solution_l92_92736


namespace gcd_polynomial_l92_92097

theorem gcd_polynomial (b : ℤ) (h : 1820 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_polynomial_l92_92097


namespace fraction_division_l92_92981

theorem fraction_division: 
  ((3 + 1 / 2) / 7) / (5 / 3) = 3 / 10 := 
by 
  sorry

end fraction_division_l92_92981


namespace shipment_cost_l92_92518

-- Define the conditions
def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def shipping_cost_per_crate : ℝ := 1.5
def surcharge_per_crate : ℝ := 0.5
def flat_fee : ℝ := 10

-- Define the question as a theorem
theorem shipment_cost : 
  let crates := total_weight / weight_per_crate
  let cost_per_crate := shipping_cost_per_crate + surcharge_per_crate
  let total_cost_crates := crates * cost_per_crate
  let total_cost := total_cost_crates + flat_fee
  total_cost = 46 := by
  -- Proof omitted
  sorry

end shipment_cost_l92_92518


namespace abs_eq_k_solution_l92_92175

theorem abs_eq_k_solution (k : ℝ) (h : k > 4014) :
  {x : ℝ | |x - 2007| + |x + 2007| = k} = (Set.Iio (-2007)) ∪ (Set.Ioi (2007)) :=
by
  sorry

end abs_eq_k_solution_l92_92175


namespace stickers_total_l92_92877

theorem stickers_total (yesterday_packs : ℕ) (increment_packs : ℕ) (today_packs : ℕ) (total_packs : ℕ) :
  yesterday_packs = 15 → increment_packs = 10 → today_packs = yesterday_packs + increment_packs → total_packs = yesterday_packs + today_packs → total_packs = 40 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h1, h3] at h4
  exact h4

end stickers_total_l92_92877


namespace a_values_in_terms_of_x_l92_92726

open Real

-- Definitions for conditions
variables (a b x y : ℝ)
variables (h1 : a^3 - b^3 = 27 * x^3)
variables (h2 : a - b = y)
variables (h3 : y = 2 * x)

-- Theorem to prove
theorem a_values_in_terms_of_x : 
  (a = x + 5 * x / sqrt 6) ∨ (a = x - 5 * x / sqrt 6) :=
sorry

end a_values_in_terms_of_x_l92_92726


namespace domain_g_l92_92203

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 14 * x - 3)

theorem domain_g :
  {x : ℝ | -8 * x^2 + 14 * x - 3 ≥ 0} = { x : ℝ | x ≤ 1 / 4 ∨ x ≥ 3 / 2 } :=
by
  sorry

end domain_g_l92_92203


namespace max_area_quad_l92_92426

noncomputable def MaxAreaABCD : ℝ :=
  let x : ℝ := 3
  let θ : ℝ := Real.pi / 2
  let φ : ℝ := Real.pi
  let area_ABC := (1/2) * x * 3 * Real.sin θ
  let area_BCD := (1/2) * 3 * 5 * Real.sin (φ - θ)
  area_ABC + area_BCD

theorem max_area_quad (x : ℝ) (h : x > 0)
  (BC_eq_3 : True)
  (CD_eq_5 : True)
  (centroids_form_isosceles : True) :
  MaxAreaABCD = 12 := by
  sorry

end max_area_quad_l92_92426


namespace magic_8_ball_probability_l92_92462

theorem magic_8_ball_probability :
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  (Nat.choose 7 3) * (p^3) * (q^4) = 590625 / 2097152 :=
by
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  sorry

end magic_8_ball_probability_l92_92462


namespace average_distance_per_day_l92_92513

def monday_distance : ℝ := 4.2
def tuesday_distance : ℝ := 3.8
def wednesday_distance : ℝ := 3.6
def thursday_distance : ℝ := 4.4
def number_of_days : ℕ := 4

theorem average_distance_per_day :
  (monday_distance + tuesday_distance + wednesday_distance + thursday_distance) / number_of_days = 4 :=
by
  sorry

end average_distance_per_day_l92_92513


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l92_92112

variable {a b : ℝ}

theorem correct_calculation : a ^ 3 * a = a ^ 4 := 
by
  sorry

theorem incorrect_calculation_A : a ^ 3 + a ^ 3 ≠ 2 * a ^ 6 := 
by
  sorry

theorem incorrect_calculation_B : (a ^ 3) ^ 3 ≠ a ^ 6 :=
by
  sorry

theorem incorrect_calculation_D : (a - b) ^ 2 ≠ a ^ 2 - b ^ 2 :=
by
  sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l92_92112


namespace octal_subtraction_correct_l92_92801

-- Define the octal numbers
def octal752 : ℕ := 7 * 8^2 + 5 * 8^1 + 2 * 8^0
def octal364 : ℕ := 3 * 8^2 + 6 * 8^1 + 4 * 8^0
def octal376 : ℕ := 3 * 8^2 + 7 * 8^1 + 6 * 8^0

-- Prove the octal number subtraction
theorem octal_subtraction_correct : octal752 - octal364 = octal376 := by
  sorry

end octal_subtraction_correct_l92_92801


namespace probability_is_five_eleven_l92_92104

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := n.choose k

-- Define the number of favorable outcomes for same letter and same color
def favorable_same_letter : ℕ := 4 * comb 3 2
def favorable_same_color : ℕ := 3 * comb 4 2

-- Total number of favorable outcomes
def total_favorable : ℕ := favorable_same_letter + favorable_same_color

-- Total number of ways to draw 2 cards from 12
def total_ways : ℕ := comb total_cards 2

-- Probability of drawing a winning pair
def probability_winning_pair : ℚ := total_favorable / total_ways

theorem probability_is_five_eleven : probability_winning_pair = 5 / 11 :=
by
  sorry

end probability_is_five_eleven_l92_92104


namespace center_of_hyperbola_l92_92701

theorem center_of_hyperbola :
  (∃ h k : ℝ, ∀ x y : ℝ, (3*y + 3)^2 / 49 - (2*x - 5)^2 / 9 = 1 ↔ x = h ∧ y = k) → 
  h = 5 / 2 ∧ k = -1 :=
by
  sorry

end center_of_hyperbola_l92_92701


namespace right_triangle_exists_l92_92870

theorem right_triangle_exists (a b c d : ℕ) (h1 : ab = cd) (h2 : a + b = c - d) : 
  ∃ (x y z : ℕ), x * y / 2 = ab ∧ x^2 + y^2 = z^2 :=
sorry

end right_triangle_exists_l92_92870


namespace percentage_less_than_l92_92172

theorem percentage_less_than (x y : ℝ) (h : x = 8 * y) : ((x - y) / x) * 100 = 87.5 := 
by sorry

end percentage_less_than_l92_92172


namespace polynomial_sum_l92_92399

def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end polynomial_sum_l92_92399


namespace equal_probabilities_hearts_clubs_l92_92939

/-- Define the total number of cards in a standard deck including two Jokers -/
def total_cards := 52 + 2

/-- Define the counts of specific card types -/
def num_jokers := 2
def num_spades := 13
def num_tens := 4
def num_hearts := 13
def num_clubs := 13

/-- Define the probabilities of drawing specific card types -/
def prob_joker := num_jokers / total_cards
def prob_spade := num_spades / total_cards
def prob_ten := num_tens / total_cards
def prob_heart := num_hearts / total_cards
def prob_club := num_clubs / total_cards

theorem equal_probabilities_hearts_clubs :
  prob_heart = prob_club :=
by
  sorry

end equal_probabilities_hearts_clubs_l92_92939


namespace low_income_households_sampled_l92_92224

def total_households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sampled_high_income_households := 25

theorem low_income_households_sampled :
  (sampled_high_income_households / high_income_households) * low_income_households = 19 := by
  sorry

end low_income_households_sampled_l92_92224


namespace foci_distance_of_hyperbola_l92_92574

theorem foci_distance_of_hyperbola :
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  2 * c = 2 * Real.sqrt 34 :=
by
  let a_sq := 25
  let b_sq := 9
  let c := Real.sqrt (a_sq + b_sq)
  sorry

end foci_distance_of_hyperbola_l92_92574


namespace ratio_of_intercepts_l92_92774

theorem ratio_of_intercepts
  (u v : ℚ)
  (h1 : 2 = 5 * u)
  (h2 : 3 = -7 * v) :
  u / v = -14 / 15 :=
by
  sorry

end ratio_of_intercepts_l92_92774


namespace function_is_even_with_period_pi_div_2_l92_92946

noncomputable def f (x : ℝ) : ℝ := (1 + Real.cos (2 * x)) * (Real.sin x) ^ 2

theorem function_is_even_with_period_pi_div_2 : 
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x : ℝ, f (x + (π / 2)) = f x) :=
by
  sorry

end function_is_even_with_period_pi_div_2_l92_92946


namespace quotient_base5_l92_92852

theorem quotient_base5 (a b quotient : ℕ) 
  (ha : a = 2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 1) 
  (hb : b = 2 * 5^1 + 3) 
  (hquotient : quotient = 1 * 5^2 + 0 * 5^1 + 3) :
  a / b = quotient :=
by sorry

end quotient_base5_l92_92852


namespace function_values_l92_92186

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem function_values (a b : ℝ) (h1 : f 1 a b = 2) (h2 : a = 2) : f 2 a b = 4 := by
  sorry

end function_values_l92_92186


namespace chord_length_of_tangent_l92_92005

theorem chord_length_of_tangent (R r : ℝ) (h : R^2 - r^2 = 25) : ∃ c : ℝ, c = 10 :=
by
  sorry

end chord_length_of_tangent_l92_92005


namespace bird_average_l92_92665

theorem bird_average (a b c : ℤ) (h1 : a = 7) (h2 : b = 11) (h3 : c = 9) :
  (a + b + c) / 3 = 9 :=
by
  sorry

end bird_average_l92_92665


namespace necessary_not_sufficient_l92_92362

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end necessary_not_sufficient_l92_92362


namespace lesser_fraction_l92_92997

theorem lesser_fraction (x y : ℚ) (h1 : x + y = 14 / 15) (h2 : x * y = 1 / 10) : min x y = 1 / 5 :=
sorry

end lesser_fraction_l92_92997


namespace total_marbles_l92_92241

theorem total_marbles (p y u : ℕ) :
  y + u = 10 →
  p + u = 12 →
  p + y = 6 →
  p + y + u = 14 :=
by
  intros h1 h2 h3
  sorry

end total_marbles_l92_92241


namespace find_b_l92_92020

theorem find_b (b : ℚ) : (∃ x y : ℚ, x = 3 ∧ y = -5 ∧ (b * x - (b + 2) * y = b - 3)) → b = -13 / 7 :=
sorry

end find_b_l92_92020


namespace correct_choice_l92_92947

def proposition_p : Prop := ∀ (x : ℝ), 2^x > x^2
def proposition_q : Prop := ∃ (x_0 : ℝ), x_0 - 2 > 0

theorem correct_choice : ¬proposition_p ∧ proposition_q :=
by
  sorry

end correct_choice_l92_92947


namespace min_gloves_proof_l92_92227

-- Let n represent the number of participants
def n : Nat := 63

-- Let g represent the number of gloves per participant
def g : Nat := 2

-- The minimum number of gloves required
def min_gloves : Nat := n * g

theorem min_gloves_proof : min_gloves = 126 :=
by 
  -- Placeholder for the proof
  sorry

end min_gloves_proof_l92_92227


namespace rhomboid_toothpicks_l92_92761

/-- 
Given:
- The rhomboid consists of two sections, each similar to half of a large equilateral triangle split along its height.
- The longest diagonal of the rhomboid contains 987 small equilateral triangles.
- The effective fact that each small equilateral triangle contributes on average 1.5 toothpicks due to shared sides.

Prove:
- The number of toothpicks required to construct the rhomboid is 1463598.
-/

-- Defining the number of small triangles along the base of the rhomboid
def base_triangles : ℕ := 987

-- Calculating the number of triangles in one section of the rhomboid
def triangles_in_section : ℕ := (base_triangles * (base_triangles + 1)) / 2

-- Calculating the total number of triangles in the rhomboid
def total_triangles : ℕ := 2 * triangles_in_section

-- Given the effective sides per triangle contributing to toothpicks is on average 1.5
def avg_sides_per_triangle : ℚ := 1.5

-- Calculating the total number of toothpicks required
def total_toothpicks : ℚ := avg_sides_per_triangle * total_triangles

theorem rhomboid_toothpicks (h : base_triangles = 987) : total_toothpicks = 1463598 := by
  sorry

end rhomboid_toothpicks_l92_92761


namespace role_assignment_l92_92479

theorem role_assignment (m w : ℕ) (m_roles w_roles e_roles : ℕ) 
  (hm : m = 5) (hw : w = 6) (hm_roles : m_roles = 2) (hw_roles : w_roles = 2) (he_roles : e_roles = 2) :
  ∃ (total_assignments : ℕ), total_assignments = 25200 :=
by
  sorry

end role_assignment_l92_92479


namespace max_consecutive_sum_l92_92056

theorem max_consecutive_sum (n : ℕ) : 
  (∀ (n : ℕ), (n*(n + 1))/2 ≤ 400 → n ≤ 27) ∧ ((27*(27 + 1))/2 ≤ 400) :=
by
  sorry

end max_consecutive_sum_l92_92056


namespace vector_c_condition_l92_92317

variables (a b c : ℝ × ℝ)

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * w.1, k * w.2)

theorem vector_c_condition (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, -3)) 
  (hc : c = (7 / 2, -7 / 4)) :
  is_perpendicular c a ∧ is_parallel b (a - c) :=
sorry

end vector_c_condition_l92_92317


namespace strawberry_cake_cost_proof_l92_92051

-- Define the constants
def chocolate_cakes : ℕ := 3
def price_per_chocolate_cake : ℕ := 12
def total_bill : ℕ := 168
def number_of_strawberry_cakes : ℕ := 6

-- Define the calculation for the total cost of chocolate cakes
def total_cost_of_chocolate_cakes : ℕ := chocolate_cakes * price_per_chocolate_cake

-- Define the remaining cost for strawberry cakes
def remaining_cost : ℕ := total_bill - total_cost_of_chocolate_cakes

-- Prove the cost per strawberry cake
def cost_per_strawberry_cake : ℕ := remaining_cost / number_of_strawberry_cakes

theorem strawberry_cake_cost_proof : cost_per_strawberry_cake = 22 := by
  -- We skip the proof here. Detailed proof steps would go in the place of sorry
  sorry

end strawberry_cake_cost_proof_l92_92051


namespace cost_of_building_fence_l92_92176

-- Define the conditions
def area_of_circle := 289 -- Area in square feet
def price_per_foot := 58  -- Price in rupees per foot

-- Define the equations used in the problem
noncomputable def radius := Real.sqrt (area_of_circle / Real.pi)
noncomputable def circumference := 2 * Real.pi * radius
noncomputable def cost := circumference * price_per_foot

-- The statement to prove
theorem cost_of_building_fence : cost = 1972 :=
  sorry

end cost_of_building_fence_l92_92176


namespace gift_boxes_in_3_days_l92_92036
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l92_92036


namespace find_X_d_minus_Y_d_l92_92654

def digits_in_base_d (X Y d : ℕ) : Prop :=
  2 * d * X + X + Y = d^2 + 8 * d + 2 

theorem find_X_d_minus_Y_d (d X Y : ℕ) (h1 : digits_in_base_d X Y d) (h2 : d > 8) : X - Y = d - 8 :=
by 
  sorry

end find_X_d_minus_Y_d_l92_92654


namespace percent_of_x_is_z_l92_92481

variable {x y z : ℝ}

theorem percent_of_x_is_z 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z / x = 1.2 := 
sorry

end percent_of_x_is_z_l92_92481


namespace infinite_series_sum_l92_92113

theorem infinite_series_sum (a r : ℝ) (h₀ : -1 < r) (h₁ : r < 1) :
    (∑' n, if (n % 2 = 0) then a * r^(n/2) else a^2 * r^((n+1)/2)) = (a * (1 + a * r))/(1 - r^2) :=
by
  sorry

end infinite_series_sum_l92_92113


namespace count_numbers_1000_to_5000_l92_92334

def countFourDigitNumbersInRange (lower upper : ℕ) : ℕ :=
  if lower <= upper then upper - lower + 1 else 0

theorem count_numbers_1000_to_5000 : countFourDigitNumbersInRange 1000 5000 = 4001 :=
by
  sorry

end count_numbers_1000_to_5000_l92_92334


namespace car_a_distance_behind_car_b_l92_92319

theorem car_a_distance_behind_car_b :
  ∃ D : ℝ, D = 40 ∧ 
    (∀ (t : ℝ), t = 4 →
    ((58 - 50) * t + 8) = D + 8)
  := by
  sorry

end car_a_distance_behind_car_b_l92_92319


namespace students_in_hollow_square_are_160_l92_92962

-- Define the problem conditions
def hollow_square_formation (outer_layer : ℕ) (inner_layer : ℕ) : Prop :=
  outer_layer = 52 ∧ inner_layer = 28

-- Define the total number of students in the group based on the given condition
def total_students (n : ℕ) : Prop := n = 160

-- Prove that the total number of students is 160 given the hollow square formation conditions
theorem students_in_hollow_square_are_160 : ∀ (outer_layer inner_layer : ℕ),
  hollow_square_formation outer_layer inner_layer → total_students 160 :=
by
  intros outer_layer inner_layer h
  sorry

end students_in_hollow_square_are_160_l92_92962


namespace max_divisors_with_remainder_10_l92_92185

theorem max_divisors_with_remainder_10 (m : ℕ) :
  (m > 0) → (∀ k, (2008 % k = 10) ↔ k < m) → m = 11 :=
by
  sorry

end max_divisors_with_remainder_10_l92_92185


namespace number_of_ways_to_choose_chairs_l92_92219

def choose_chairs_equivalent (chairs : Nat) (students : Nat) (professors : Nat) : Nat :=
  let positions := (chairs - 2)  -- exclude first and last chair
  Nat.choose positions professors * Nat.factorial professors

theorem number_of_ways_to_choose_chairs : choose_chairs_equivalent 10 5 4 = 1680 :=
by
  -- The positions for professors are available from chairs 2 through 9 which are 8 positions.
  /- Calculation for choosing 4 positions out of these 8:
     C(8,4) * 4! = 70 * 24 = 1680 -/
  sorry

end number_of_ways_to_choose_chairs_l92_92219


namespace distance_between_B_and_D_l92_92847

theorem distance_between_B_and_D (a b c d : ℝ) (h1 : |2 * a - 3 * c| = 1) (h2 : |2 * b - 3 * c| = 1) (h3 : |(2/3) * (d - a)| = 1) (h4 : a ≠ b) :
  |d - b| = 0.5 ∨ |d - b| = 2.5 :=
by
  sorry

end distance_between_B_and_D_l92_92847


namespace arithmetic_sequence_first_term_l92_92920

theorem arithmetic_sequence_first_term
  (a : ℕ) -- First term of the arithmetic sequence
  (d : ℕ := 3) -- Common difference, given as 3
  (n : ℕ := 20) -- Number of terms, given as 20
  (S : ℕ := 650) -- Sum of the sequence, given as 650
  (h : S = (n / 2) * (2 * a + (n - 1) * d)) : a = 4 := 
by
  sorry

end arithmetic_sequence_first_term_l92_92920


namespace intersection_M_N_l92_92816

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := 3^x - 2

def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | x < 1} :=
by sorry

end intersection_M_N_l92_92816


namespace temperature_on_friday_l92_92717

variables {M T W Th F : ℝ}

theorem temperature_on_friday
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 41) :
  F = 33 :=
  sorry

end temperature_on_friday_l92_92717


namespace triangle_exterior_angle_bisectors_l92_92279

theorem triangle_exterior_angle_bisectors 
  (α β γ α1 β1 γ1 : ℝ) 
  (h₁ : α = (β / 2 + γ / 2)) 
  (h₂ : β = (γ / 2 + α / 2)) 
  (h₃ : γ = (α / 2 + β / 2)) :
  α = 180 - 2 * α1 ∧
  β = 180 - 2 * β1 ∧
  γ = 180 - 2 * γ1 := by
  sorry

end triangle_exterior_angle_bisectors_l92_92279


namespace point_on_hyperbola_l92_92369

theorem point_on_hyperbola (x y : ℝ) (h_eqn : y = -4 / x) (h_point : x = -2 ∧ y = 2) : x * y = -4 := 
by
  intros
  sorry

end point_on_hyperbola_l92_92369


namespace ordered_pair_exists_l92_92633

theorem ordered_pair_exists :
  ∃ p q : ℝ, 
  (3 + 8 * p = 2 - 3 * q) ∧ (-4 - 6 * p = -3 + 4 * q) ∧ (p = -1/14) ∧ (q = -1/7) :=
by
  sorry

end ordered_pair_exists_l92_92633


namespace elliot_storeroom_blocks_l92_92975

def storeroom_volume (length: ℕ) (width: ℕ) (height: ℕ) : ℕ :=
  length * width * height

def inner_volume (length: ℕ) (width: ℕ) (height: ℕ) (thickness: ℕ) : ℕ :=
  (length - 2 * thickness) * (width - 2 * thickness) * (height - thickness)

def blocks_needed (outer_volume: ℕ) (inner_volume: ℕ) : ℕ :=
  outer_volume - inner_volume

theorem elliot_storeroom_blocks :
  let length := 15
  let width := 12
  let height := 8
  let thickness := 2
  let outer_volume := storeroom_volume length width height
  let inner_volume := inner_volume length width height thickness
  let required_blocks := blocks_needed outer_volume inner_volume
  required_blocks = 912 :=
by {
  -- Definitions and calculations as per conditions
  sorry
}

end elliot_storeroom_blocks_l92_92975


namespace max_expression_value_l92_92127

open Real

theorem max_expression_value (a b d x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : (x₁^4 - a * x₁^3 + b * x₁^2 - a * x₁ + d = 0))
  (h2 : (x₂^4 - a * x₂^3 + b * x₂^2 - a * x₂ + d = 0))
  (h3 : (x₃^4 - a * x₃^3 + b * x₃^2 - a * x₃ + d = 0))
  (h4 : (x₄^4 - a * x₄^3 + b * x₄^2 - a * x₄ + d = 0))
  (h5 : (1 / 2 ≤ x₁ ∧ x₁ ≤ 2))
  (h6 : (1 / 2 ≤ x₂ ∧ x₂ ≤ 2))
  (h7 : (1 / 2 ≤ x₃ ∧ x₃ ≤ 2))
  (h8 : (1 / 2 ≤ x₄ ∧ x₄ ≤ 2)) :
  ∃ (M : ℝ), M = 5 / 4 ∧
  (∀ (y₁ y₂ y₃ y₄ : ℝ),
    (y₁^4 - a * y₁^3 + b * y₁^2 - a * y₁ + d = 0) →
    (y₂^4 - a * y₂^3 + b * y₂^2 - a * y₂ + d = 0) →
    (y₃^4 - a * y₃^3 + b * y₃^2 - a * y₃ + d = 0) →
    (y₄^4 - a * y₄^3 + b * y₄^2 - a * y₄ + d = 0) →
    (1 / 2 ≤ y₁ ∧ y₁ ≤ 2) →
    (1 / 2 ≤ y₂ ∧ y₂ ≤ 2) →
    (1 / 2 ≤ y₃ ∧ y₃ ≤ 2) →
    (1 / 2 ≤ y₄ ∧ y₄ ≤ 2) →
    (y = (y₁ + y₂) * (y₁ + y₃) * y₄ / ((y₄ + y₂) * (y₄ + y₃) * y₁)) →
    y ≤ M) := 
sorry

end max_expression_value_l92_92127


namespace verify_A_l92_92290

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![62 / 7, -9 / 7], ![2 / 7, 17 / 7]]

theorem verify_A :
  matrix_A.mulVec ![1, 3] = ![5, 7] ∧
  matrix_A.mulVec ![-2, 1] = ![-19, 3] :=
by
  sorry

end verify_A_l92_92290


namespace sum_increased_consecutive_integers_product_990_l92_92818

theorem sum_increased_consecutive_integers_product_990 
  (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : a * b * c = 990) :
  (a + 2) + (b + 2) + (c + 2) = 36 :=
sorry

end sum_increased_consecutive_integers_product_990_l92_92818


namespace smallest_integer_n_l92_92433

theorem smallest_integer_n (n : ℤ) (h : n^2 - 9 * n + 20 > 0) : n ≥ 6 := 
sorry

end smallest_integer_n_l92_92433


namespace restore_temperature_time_l92_92289

theorem restore_temperature_time :
  let rate_increase := 8 -- degrees per hour
  let duration_increase := 3 -- hours
  let rate_decrease := 4 -- degrees per hour
  let total_increase := rate_increase * duration_increase
  let time := total_increase / rate_decrease
  time = 6 := 
by
  sorry

end restore_temperature_time_l92_92289


namespace no_magpies_left_l92_92585

theorem no_magpies_left (initial_magpies killed_magpies : ℕ) (fly_away : Prop):
  initial_magpies = 40 → killed_magpies = 6 → fly_away → ∀ M : ℕ, M = 0 :=
by
  intro h0 h1 h2
  sorry

end no_magpies_left_l92_92585


namespace count_arithmetic_sequence_l92_92546

theorem count_arithmetic_sequence :
  let a1 := 2.5
  let an := 68.5
  let d := 6.0
  let offset := 0.5
  let adjusted_a1 := a1 + offset
  let adjusted_an := an + offset
  let n := (adjusted_an - adjusted_a1) / d + 1
  n = 12 :=
by {
  sorry
}

end count_arithmetic_sequence_l92_92546


namespace smallest_b_value_l92_92572

noncomputable def smallest_b (a b : ℝ) : ℝ :=
if a > 2 ∧ 2 < a ∧ a < b 
   ∧ (2 + a ≤ b) 
   ∧ ((1 / a) + (1 / b) ≤ 1 / 2) 
then b else 0

theorem smallest_b_value : ∀ (a b : ℝ), 
  (2 < a) → (a < b) → (2 + a ≤ b) → 
  ((1 / a) + (1 / b) ≤ 1 / 2) → 
  b = 3 + Real.sqrt 5 := sorry

end smallest_b_value_l92_92572


namespace diff_roots_eq_sqrt_2p2_add_2p_sub_2_l92_92432

theorem diff_roots_eq_sqrt_2p2_add_2p_sub_2 (p : ℝ) :
  let a := 1
  let b := -2 * p
  let c := p^2 - p + 1
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let r1 := (-b + sqrt_discriminant) / (2 * a)
  let r2 := (-b - sqrt_discriminant) / (2 * a)
  r1 - r2 = Real.sqrt (2*p^2 + 2*p - 2) :=
by
  sorry

end diff_roots_eq_sqrt_2p2_add_2p_sub_2_l92_92432


namespace sum_of_fractions_eq_two_l92_92631

theorem sum_of_fractions_eq_two : 
  (1 / 2) + (2 / 4) + (4 / 8) + (8 / 16) = 2 :=
by sorry

end sum_of_fractions_eq_two_l92_92631


namespace distance_focus_directrix_l92_92044

theorem distance_focus_directrix (y x : ℝ) (h : y^2 = 2 * x) : x = 1 := 
by 
  sorry

end distance_focus_directrix_l92_92044


namespace three_digit_number_exists_l92_92626

theorem three_digit_number_exists : 
  ∃ (x y z : ℕ), 
  (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧ 
  (100 * x + 10 * z + y + 1 = 2 * (100 * y + 10 * z + x)) ∧ 
  (100 * x + 10 * z + y = 793) :=
by
  sorry

end three_digit_number_exists_l92_92626


namespace simplify_fraction_l92_92759

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l92_92759


namespace area_shaded_region_l92_92183

theorem area_shaded_region :
  let r_s := 3   -- Radius of the smaller circle
  let r_l := 3 * r_s  -- Radius of the larger circle
  let A_l := π * r_l^2  -- Area of the larger circle
  let A_s := π * r_s^2  -- Area of the smaller circle
  A_l - A_s = 72 * π := 
by
  sorry

end area_shaded_region_l92_92183


namespace div_fraction_l92_92478

/-- The result of dividing 3/7 by 2 1/2 equals 6/35 -/
theorem div_fraction : (3/7) / (2 + 1/2) = 6/35 :=
by 
  sorry

end div_fraction_l92_92478


namespace cassie_water_bottle_ounces_l92_92605

-- Define the given quantities
def cups_per_day : ℕ := 12
def ounces_per_cup : ℕ := 8
def refills_per_day : ℕ := 6

-- Define the total ounces of water Cassie drinks per day
def total_ounces_per_day := cups_per_day * ounces_per_cup

-- Define the ounces her water bottle holds
def ounces_per_bottle := total_ounces_per_day / refills_per_day

-- Prove the statement
theorem cassie_water_bottle_ounces : 
  ounces_per_bottle = 16 := by 
  sorry

end cassie_water_bottle_ounces_l92_92605


namespace grade_point_average_one_third_l92_92137

theorem grade_point_average_one_third :
  ∃ (x : ℝ), 55 = (1/3) * x + (2/3) * 60 ∧ x = 45 :=
by
  sorry

end grade_point_average_one_third_l92_92137


namespace eval_expr_l92_92091

variable {x y : ℝ}

theorem eval_expr (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^4 + 1) / x^2) * ((y^4 + 1) / y^2) - ((x^4 - 1) / y^2) * ((y^4 - 1) / x^2) = (2 * x^2) / (y^2) + (2 * y^2) / (x^2) := by
  sorry

end eval_expr_l92_92091


namespace maximize_revenue_l92_92394

noncomputable def revenue (p : ℝ) : ℝ := 100 * p - 4 * p^2

theorem maximize_revenue : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 20 ∧ (∀ q : ℝ, 0 ≤ q ∧ q ≤ 20 → revenue q ≤ revenue p) ∧ p = 12.5 := by
  sorry

end maximize_revenue_l92_92394


namespace point_in_third_quadrant_l92_92409

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l92_92409


namespace leon_total_payment_l92_92359

-- Define the constants based on the problem conditions
def cost_toy_organizer : ℝ := 78
def num_toy_organizers : ℝ := 3
def cost_gaming_chair : ℝ := 83
def num_gaming_chairs : ℝ := 2
def delivery_fee_rate : ℝ := 0.05

-- Calculate the cost for each category and the total cost
def total_cost_toy_organizers : ℝ := num_toy_organizers * cost_toy_organizer
def total_cost_gaming_chairs : ℝ := num_gaming_chairs * cost_gaming_chair
def total_sales : ℝ := total_cost_toy_organizers + total_cost_gaming_chairs
def delivery_fee : ℝ := delivery_fee_rate * total_sales
def total_amount_paid : ℝ := total_sales + delivery_fee

-- State the theorem for the total amount Leon has to pay
theorem leon_total_payment :
  total_amount_paid = 420 := by
  sorry

end leon_total_payment_l92_92359


namespace max_initial_jars_l92_92935

theorem max_initial_jars (w_B w_C a : ℤ) (h1 : w_C = 13 * w_B) (h2 : w_C - a = 8 * (w_B + a)) : 
  ∃ (n : ℤ), n ≤ 23 ∧ ∀ (k : ℤ), w_B = 9 * k ∧ w_C = 117 * k := 
  by 
  sorry

end max_initial_jars_l92_92935


namespace range_of_a1_l92_92602

theorem range_of_a1 (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 1 / (2 - a n)) 
  (h_pos : ∀ n, a (n + 1) > a n) : a 1 < 1 := 
sorry

end range_of_a1_l92_92602


namespace journey_time_difference_l92_92151

theorem journey_time_difference :
  let t1 := (100:ℝ) / 60
  let t2 := (400:ℝ) / 40
  let T1 := t1 + t2
  let T2 := (500:ℝ) / 50
  let difference := (T1 - T2) * 60
  abs (difference - 100) < 0.01 :=
by
  sorry

end journey_time_difference_l92_92151


namespace shaded_area_correct_l92_92242

-- Conditions
def side_length_square := 40
def triangle1_base := 15
def triangle1_height := 15
def triangle2_base := 15
def triangle2_height := 15

-- Calculation
def square_area := side_length_square * side_length_square
def triangle1_area := 1 / 2 * triangle1_base * triangle1_height
def triangle2_area := 1 / 2 * triangle2_base * triangle2_height
def total_triangle_area := triangle1_area + triangle2_area
def shaded_region_area := square_area - total_triangle_area

-- Theorem to prove
theorem shaded_area_correct : shaded_region_area = 1375 := by
  sorry

end shaded_area_correct_l92_92242


namespace largest_square_side_length_l92_92737

theorem largest_square_side_length (smallest_square_side next_square_side : ℕ) (h1 : smallest_square_side = 1) 
(h2 : next_square_side = smallest_square_side + 6) :
  ∃ x : ℕ, x = 7 :=
by
  existsi 7
  sorry

end largest_square_side_length_l92_92737


namespace find_L_l92_92439

-- Conditions definitions
def initial_marbles := 57
def marbles_won_second_game := 25
def final_marbles := 64

-- Definition of L
def L := initial_marbles - 18

theorem find_L (L : ℕ) (H1 : initial_marbles = 57) (H2 : marbles_won_second_game = 25) (H3 : final_marbles = 64) : 
(initial_marbles - L) + marbles_won_second_game = final_marbles -> 
L = 18 :=
by
  sorry

end find_L_l92_92439


namespace minimize_feed_costs_l92_92610

theorem minimize_feed_costs 
  (x y : ℝ)
  (h1: 5 * x + 3 * y ≥ 30)
  (h2: 2.5 * x + 3 * y ≥ 22.5)
  (h3: x ≥ 0)
  (h4: y ≥ 0)
  : (x = 3 ∧ y = 5) ∧ (x + y = 8) := 
sorry

end minimize_feed_costs_l92_92610


namespace quadratic_equation_with_distinct_roots_l92_92826

theorem quadratic_equation_with_distinct_roots 
  (a p q b α : ℝ) 
  (hα1 : α ≠ 0) 
  (h_quad1 : α^2 + a * α + b = 0) 
  (h_quad2 : α^2 + p * α + q = 0) : 
  ∃ x : ℝ, x^2 - (b + q) * (a - p) / (q - b) * x + b * q * (a - p)^2 / (q - b)^2 = 0 :=
by
  sorry

end quadratic_equation_with_distinct_roots_l92_92826


namespace scientific_notation_of_18500000_l92_92686

-- Definition of scientific notation function
def scientific_notation (n : ℕ) : string := sorry

-- Problem statement
theorem scientific_notation_of_18500000 : 
  scientific_notation 18500000 = "1.85 × 10^7" :=
sorry

end scientific_notation_of_18500000_l92_92686


namespace polynomial_form_l92_92037

noncomputable def polynomial_solution (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))

theorem polynomial_form :
  ∀ (P : ℝ → ℝ), polynomial_solution P ↔ ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x^2 + b * x^4 :=
by 
  sorry

end polynomial_form_l92_92037


namespace Anton_thought_of_729_l92_92187

def is_digit_match (a b : ℕ) (pos : ℕ) : Prop :=
  ((a / (10 ^ pos)) % 10) = ((b / (10 ^ pos)) % 10)

theorem Anton_thought_of_729 :
  ∃ n : ℕ, n < 1000 ∧
  (is_digit_match n 109 0 ∧ ¬is_digit_match n 109 1 ∧ ¬is_digit_match n 109 2) ∧
  (¬is_digit_match n 704 0 ∧ is_digit_match n 704 1 ∧ ¬is_digit_match n 704 2) ∧
  (¬is_digit_match n 124 0 ∧ ¬is_digit_match n 124 1 ∧ is_digit_match n 124 2) ∧
  n = 729 :=
sorry

end Anton_thought_of_729_l92_92187


namespace reciprocal_inequality_l92_92533

theorem reciprocal_inequality {a b c : ℝ} (hab : a < b) (hbc : b < c) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  (1 / a) < (1 / b) :=
sorry

end reciprocal_inequality_l92_92533


namespace rectangle_area_l92_92797

theorem rectangle_area (b l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 112) : l * b = 588 := by
  sorry

end rectangle_area_l92_92797


namespace colleen_paid_more_l92_92556

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l92_92556


namespace value_of_MN_l92_92363

theorem value_of_MN (M N : ℝ) (log : ℝ → ℝ → ℝ)
    (h1 : log (M ^ 2) N = log N (M ^ 2))
    (h2 : M ≠ N)
    (h3 : M * N > 0)
    (h4 : M ≠ 1)
    (h5 : N ≠ 1) :
    M * N = N^(1/2) :=
  sorry

end value_of_MN_l92_92363


namespace manufacturing_department_percentage_l92_92142

theorem manufacturing_department_percentage (total_degrees mfg_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : mfg_degrees = 162) : (mfg_degrees / total_degrees) * 100 = 45 :=
by 
  sorry

end manufacturing_department_percentage_l92_92142


namespace root_sum_of_reciprocals_l92_92300

theorem root_sum_of_reciprocals {m : ℝ} :
  (∃ (a b : ℝ), a ≠ b ∧ (a + b) = 2 * (m + 1) ∧ (a * b) = m^2 + 2 ∧ (1/a + 1/b) = 1) →
  m = 2 :=
by sorry

end root_sum_of_reciprocals_l92_92300


namespace min_distance_between_curves_l92_92887

noncomputable def distance_between_intersections : ℝ :=
  let f (x : ℝ) := (2 * x + 1) - (x + Real.log x)
  let f' (x : ℝ) := 1 - 1 / x
  let minimum_distance :=
    if hs : 1 < 1 then 2 else
    if hs : 1 > 1 then 2 else
    2
  minimum_distance

theorem min_distance_between_curves : distance_between_intersections = 2 :=
by
  sorry

end min_distance_between_curves_l92_92887


namespace solve_for_x_l92_92961

theorem solve_for_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y)
  (h2 : y = x^2) :
  x = (-1 + Real.sqrt 55) / 3 := 
by
  sorry

end solve_for_x_l92_92961


namespace all_terms_perfect_squares_l92_92647

def seq_x : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => 14 * seq_x (n + 1) - seq_x n - 4

theorem all_terms_perfect_squares : ∀ n, ∃ k, seq_x n = k^2 :=
by
  sorry

end all_terms_perfect_squares_l92_92647


namespace find_angle_B_l92_92067

theorem find_angle_B 
  (A B : ℝ)
  (h1 : B + A = 90)
  (h2 : B = 4 * A) : 
  B = 144 :=
by
  sorry

end find_angle_B_l92_92067


namespace inequality_for_natural_n_l92_92869

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_for_natural_n_l92_92869


namespace net_change_is_12_l92_92146

-- Definitions based on the conditions of the problem

def initial_investment : ℝ := 100
def first_year_increase_percentage : ℝ := 0.60
def second_year_decrease_percentage : ℝ := 0.30

-- Calculate the wealth at the end of the first year
def end_of_first_year_wealth : ℝ := initial_investment * (1 + first_year_increase_percentage)

-- Calculate the wealth at the end of the second year
def end_of_second_year_wealth : ℝ := end_of_first_year_wealth * (1 - second_year_decrease_percentage)

-- Calculate the net change
def net_change : ℝ := end_of_second_year_wealth - initial_investment

-- The target theorem to prove
theorem net_change_is_12 : net_change = 12 := by
  sorry

end net_change_is_12_l92_92146


namespace a_share_is_1400_l92_92980

-- Definitions for the conditions
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def share_B : ℕ := 2200

-- Definition for the ratios
def ratio_A : ℚ := investment_A / 1000
def ratio_B : ℚ := investment_B / 1000
def ratio_C : ℚ := investment_C / 1000

-- Sum of ratios
def sum_ratios : ℚ := ratio_A + ratio_B + ratio_C

-- Total profit P can be deduced from B's share
def total_profit : ℚ := share_B * sum_ratios / ratio_B

-- Goal: Prove that A's share is $1400
def share_A : ℚ := ratio_A * total_profit / sum_ratios

theorem a_share_is_1400 : share_A = 1400 :=
sorry

end a_share_is_1400_l92_92980


namespace value_of_a_plus_b_l92_92875

variables (a b c d x : ℕ)

theorem value_of_a_plus_b : (b + c = 9) → (c + d = 3) → (a + d = 8) → (a + b = x) → x = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_a_plus_b_l92_92875


namespace sumNats_l92_92733

-- Define the set of natural numbers between 29 and 31 inclusive
def NatRange : List ℕ := [29, 30, 31]

-- Define the condition that checks the elements in the range
def isValidNumbers (n : ℕ) : Prop := n ≤ 31 ∧ n > 28

-- Check if all numbers in NatRange are valid
def allValidNumbers : Prop := ∀ n, n ∈ NatRange → isValidNumbers n

-- Define the sum function for the list
def sumList (lst : List ℕ) : ℕ := lst.foldr (.+.) 0

-- The main theorem
theorem sumNats : (allValidNumbers → (sumList NatRange) = 90) :=
by
  sorry

end sumNats_l92_92733


namespace james_sells_boxes_l92_92901

theorem james_sells_boxes (profit_per_candy_bar : ℝ) (total_profit : ℝ) 
                          (candy_bars_per_box : ℕ) (x : ℕ)
                          (h1 : profit_per_candy_bar = 1.5 - 1)
                          (h2 : total_profit = 25)
                          (h3 : candy_bars_per_box = 10) 
                          (h4 : total_profit = (x * candy_bars_per_box) * profit_per_candy_bar) :
                          x = 5 :=
by
  sorry

end james_sells_boxes_l92_92901


namespace average_speed_entire_journey_l92_92179

-- Define the average speed for the journey from x to y
def speed_xy := 60

-- Define the average speed for the journey from y to x
def speed_yx := 30

-- Definition for the distance (D) (it's an abstract value, so we don't need to specify)
variable (D : ℝ) (hD : D > 0)

-- Theorem stating that the average speed for the entire journey is 40 km/hr
theorem average_speed_entire_journey : 
  2 * D / ((D / speed_xy) + (D / speed_yx)) = 40 := 
by 
  sorry

end average_speed_entire_journey_l92_92179


namespace geometric_progression_l92_92250

theorem geometric_progression :
  ∃ (b1 q : ℚ), 
    (b1 * q * (q^2 - 1) = -45/32) ∧ 
    (b1 * q^3 * (q^2 - 1) = -45/512) ∧ 
    ((b1 = 6 ∧ q = 1/4) ∨ (b1 = -6 ∧ q = -1/4)) :=
by
  sorry

end geometric_progression_l92_92250


namespace product_of_5_consecutive_numbers_not_square_l92_92711

-- Define what it means for a product to be a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- The main theorem stating the problem
theorem product_of_5_consecutive_numbers_not_square :
  ∀ (a : ℕ), 0 < a → ¬ is_perfect_square (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
by 
  sorry

end product_of_5_consecutive_numbers_not_square_l92_92711


namespace iterate_F_l92_92096

def F (x : ℝ) : ℝ := x^3 + 3*x^2 + 3*x

theorem iterate_F (x : ℝ) : (Nat.iterate F 2017 x) = (x + 1)^(3^2017) - 1 :=
by
  sorry

end iterate_F_l92_92096


namespace two_digit_square_difference_l92_92592

-- Define the problem in Lean
theorem two_digit_square_difference :
  ∃ (X Y : ℕ), (10 ≤ X ∧ X ≤ 99) ∧ (10 ≤ Y ∧ Y ≤ 99) ∧ (X > Y) ∧
  (∃ (t : ℕ), (1 ≤ t ∧ t ≤ 9) ∧ (X^2 - Y^2 = 100 * t)) :=
sorry

end two_digit_square_difference_l92_92592


namespace correct_proposition_l92_92895

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Conditions
axiom perpendicular (m : Line) (α : Plane) : Prop
axiom parallel (n : Line) (α : Plane) : Prop

-- Specific conditions given
axiom m_perp_α : perpendicular m α
axiom n_par_α : parallel n α

-- Statement to prove
theorem correct_proposition : perpendicular m n := sorry

end correct_proposition_l92_92895


namespace op_7_3_eq_70_l92_92504

noncomputable def op (x y : ℝ) : ℝ := sorry

axiom ax1 : ∀ x : ℝ, op x 0 = x
axiom ax2 : ∀ x y : ℝ, op x y = op y x
axiom ax3 : ∀ x y : ℝ, op (x + 1) y = (op x y) + y + 2

theorem op_7_3_eq_70 : op 7 3 = 70 := by
  sorry

end op_7_3_eq_70_l92_92504


namespace area_enclosed_by_region_l92_92163

theorem area_enclosed_by_region : ∀ (x y : ℝ), (x^2 + y^2 - 8*x + 6*y = -9) → (π * (4 ^ 2) = 16 * π) :=
by
  intro x y h
  sorry

end area_enclosed_by_region_l92_92163


namespace fraction_members_absent_l92_92976

variable (p : ℕ) -- Number of persons in the office
variable (W : ℝ) -- Total work amount
variable (x : ℝ) -- Fraction of members absent

theorem fraction_members_absent (h : W / (p * (1 - x)) = W / p + W / (6 * p)) : x = 1 / 7 :=
by
  sorry

end fraction_members_absent_l92_92976


namespace units_digit_37_pow_37_l92_92751

theorem units_digit_37_pow_37: (37^37) % 10 = 7 :=
by sorry

end units_digit_37_pow_37_l92_92751


namespace smallest_solution_for_quartic_eq_l92_92512

theorem smallest_solution_for_quartic_eq :
  let f (x : ℝ) := x^4 - 40*x^2 + 144
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y :=
sorry

end smallest_solution_for_quartic_eq_l92_92512


namespace best_purchase_option_l92_92725

-- Define the prices and discount conditions for each store
def technik_city_price_before_discount : ℝ := 2000 + 4000
def technomarket_price_before_discount : ℝ := 1500 + 4800

def technik_city_discount : ℝ := technik_city_price_before_discount * 0.10
def technomarket_bonus : ℝ := technomarket_price_before_discount * 0.20

def technik_city_final_price : ℝ := technik_city_price_before_discount - technik_city_discount
def technomarket_final_price : ℝ := technomarket_price_before_discount

-- The theorem stating the ultimate proof problem
theorem best_purchase_option : technik_city_final_price < technomarket_final_price :=
by
  -- Replace 'sorry' with the actual proof if required
  sorry

end best_purchase_option_l92_92725


namespace wind_velocity_l92_92934

theorem wind_velocity (P A V : ℝ) (k : ℝ := 1/200) :
  (P = k * A * V^2) →
  (P = 2) → (A = 1) → (V = 20) →
  ∀ (P' A' : ℝ), P' = 128 → A' = 4 → ∃ V' : ℝ, V'^2 = 6400 :=
by
  intros h1 h2 h3 h4 P' A' h5 h6
  use 80
  linarith

end wind_velocity_l92_92934


namespace find_roots_of_g_l92_92900

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - a*x - b
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := b*x^2 - a*x - 1

theorem find_roots_of_g :
  (∀ a b : ℝ, f 2 a b = 0 ∧ f 3 a b = 0 → ∃ (x1 x2 : ℝ), g x1 a b = 0 ∧ g x2 a b = 0 ∧
    (x1 = -1/2 ∨ x1 = -1/3) ∧ (x2 = -1/2 ∨ x2 = -1/3) ∧ x1 ≠ x2) :=
by
  sorry

end find_roots_of_g_l92_92900


namespace negation_of_exists_leq_zero_l92_92527

theorem negation_of_exists_leq_zero (x : ℝ) : ¬(∃ x ≥ 1, 2^x ≤ 0) ↔ ∀ x ≥ 1, 2^x > 0 :=
by
  sorry

end negation_of_exists_leq_zero_l92_92527


namespace cyclic_sum_nonneg_l92_92411

theorem cyclic_sum_nonneg 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (k : ℝ) (hk1 : 0 ≤ k) (hk2 : k < 2) :
  (a^2 - b * c) / (b^2 + c^2 + k * a^2)
  + (b^2 - c * a) / (c^2 + a^2 + k * b^2)
  + (c^2 - a * b) / (a^2 + b^2 + k * c^2) ≥ 0 :=
sorry

end cyclic_sum_nonneg_l92_92411


namespace units_digit_G_1000_l92_92465

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G_1000 : (G 1000) % 10 = 4 :=
  sorry

end units_digit_G_1000_l92_92465


namespace x_proportionality_find_x_value_l92_92729

theorem x_proportionality (m n : ℝ) (x z : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h3 : x = 4) (h4 : z = 8) :
  ∃ k, ∀ z : ℝ, x = k / z^8 := 
sorry

theorem find_x_value (m n : ℝ) (k : ℝ) (h1 : ∀ y, x = m * y^4) (h2 : ∀ z, y = n / z^2) (h5 : k = 67108864) :
  ∀ z, (z = 32 → x = 1 / 16) :=
sorry

end x_proportionality_find_x_value_l92_92729


namespace james_tylenol_intake_per_day_l92_92881

variable (hours_in_day : ℕ := 24) 
variable (tablets_per_dose : ℕ := 2) 
variable (mg_per_tablet : ℕ := 375)
variable (hours_per_dose : ℕ := 6)

theorem james_tylenol_intake_per_day :
  (tablets_per_dose * mg_per_tablet) * (hours_in_day / hours_per_dose) = 3000 := by
  sorry

end james_tylenol_intake_per_day_l92_92881


namespace value_of_ab_l92_92699

theorem value_of_ab (a b : ℤ) (h1 : ∀ x : ℤ, -1 < x ∧ x < 1 → (2 * x < a + 1) ∧ (x > 2 * b + 3)) :
  (a + 1) * (b - 1) = -6 :=
by
  sorry

end value_of_ab_l92_92699


namespace books_fill_shelf_l92_92530

theorem books_fill_shelf
  (A H S M E : ℕ)
  (h1 : A ≠ H) (h2 : S ≠ M) (h3 : M ≠ H) (h4 : E > 0)
  (Eq1 : A > 0) (Eq2 : H > 0) (Eq3 : S > 0) (Eq4 : M > 0)
  (h5 : A ≠ S) (h6 : E ≠ A) (h7 : E ≠ H) (h8 : E ≠ S) (h9 : E ≠ M) :
  E = (A * M - S * H) / (M - H) :=
by
  sorry

end books_fill_shelf_l92_92530


namespace vincent_total_packs_l92_92547

noncomputable def total_packs (yesterday today_addition: ℕ) : ℕ :=
  let today := yesterday + today_addition
  yesterday + today

theorem vincent_total_packs
  (yesterday_packs : ℕ)
  (today_addition: ℕ)
  (hyesterday: yesterday_packs = 15)
  (htoday_addition: today_addition = 10) :
  total_packs yesterday_packs today_addition = 40 :=
by
  rw [hyesterday, htoday_addition]
  unfold total_packs
  -- at this point it simplifies to 15 + (15 + 10) = 40
  sorry

end vincent_total_packs_l92_92547


namespace complement_intersection_l92_92033

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {3, 4, 5}) :
  U \ (A ∩ B) = {1, 2, 4, 5} :=
by
  sorry

end complement_intersection_l92_92033


namespace base_b_sum_correct_l92_92867

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct_l92_92867


namespace trajectory_of_G_l92_92692

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop :=
  9 * x^2 / 4 + 3 * y^2 = 1

-- State the theorem
theorem trajectory_of_G (P G : ℝ × ℝ) (hP : ellipse P.1 P.2) (hG_relation : ∃ k : ℝ, k = 2 ∧ P = (3 * G.1, 3 * G.2)) :
  trajectory G.1 G.2 :=
by
  sorry

end trajectory_of_G_l92_92692


namespace lucas_mod_prime_zero_l92_92276

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0 => 1       -- Note that in the mathematical problem L_1 is given as 1. Therefore we adjust for 0-based index in programming.
| 1 => 3
| (n + 2) => lucas n + lucas (n + 1)

-- Main theorem statement
theorem lucas_mod_prime_zero (p : ℕ) (hp : Nat.Prime p) : (lucas p - 1) % p = 0 := by
  sorry

end lucas_mod_prime_zero_l92_92276


namespace vector_computation_equiv_l92_92907

variables (u v w : ℤ × ℤ)

def vector_expr (u v w : ℤ × ℤ) :=
  2 • u + 4 • v - 3 • w

theorem vector_computation_equiv :
  u = (3, -5) →
  v = (-1, 6) →
  w = (2, -4) →
  vector_expr u v w = (-4, 26) :=
by
  intros hu hv hw
  rw [hu, hv, hw]
  dsimp [vector_expr]
  -- The actual proof goes here, but we use 'sorry' to skip it.
  sorry

end vector_computation_equiv_l92_92907


namespace john_max_correct_answers_l92_92982

theorem john_max_correct_answers 
  (c w b : ℕ) -- define c, w, b as natural numbers
  (h1 : c + w + b = 30) -- condition 1: total questions
  (h2 : 4 * c - 3 * w = 36) -- condition 2: scoring equation
  : c ≤ 12 := -- statement to prove
sorry

end john_max_correct_answers_l92_92982


namespace single_rooms_booked_l92_92197

noncomputable def hotel_problem (S D : ℕ) : Prop :=
  S + D = 260 ∧ 35 * S + 60 * D = 14000

theorem single_rooms_booked (S D : ℕ) (h : hotel_problem S D) : S = 64 :=
by
  sorry

end single_rooms_booked_l92_92197


namespace slope_of_l_l92_92588

noncomputable def C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 4 * Real.sin θ)
noncomputable def l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

theorem slope_of_l
  (α θ₁ θ₂ t₁ t₂ : ℝ)
  (h_midpoint : (C θ₁).fst + (C θ₂).fst = 1 + (t₁ + t₂) * Real.cos α ∧ 
                (C θ₁).snd + (C θ₂).snd = 2 + (t₁ + t₂) * Real.sin α) :
  Real.tan α = -2 :=
by
  sorry

end slope_of_l_l92_92588


namespace contradiction_proof_l92_92575

theorem contradiction_proof (x y : ℝ) (h1 : x + y < 2) (h2 : 1 < x) (h3 : 1 < y) : false := 
by 
  sorry

end contradiction_proof_l92_92575


namespace find_number_l92_92560

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 := 
sorry

end find_number_l92_92560


namespace vectors_parallel_opposite_directions_l92_92129

theorem vectors_parallel_opposite_directions
  (a b : ℝ × ℝ)
  (h₁ : a = (-1, 2))
  (h₂ : b = (2, -4)) :
  b = (-2 : ℝ) • a ∧ b = -2 • a :=
by
  sorry

end vectors_parallel_opposite_directions_l92_92129


namespace smallest_integer_with_eight_factors_l92_92690

theorem smallest_integer_with_eight_factors : ∃ N : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) ∧ ∀ M : ℕ, (∀ p q : ℕ, N = p ^ 1 * q ^ 3 ∨ N = p ^ 3 * q ^ 1) → N ≤ M :=
sorry -- Proof not provided.

end smallest_integer_with_eight_factors_l92_92690


namespace ratio_x_y_l92_92495

theorem ratio_x_y (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1 / 2) : 
  x / y = 3 / (6 * x - 1) := 
sorry

end ratio_x_y_l92_92495


namespace cos_double_angle_l92_92938

open Real

theorem cos_double_angle {α β : ℝ} (h1 : sin α = sqrt 5 / 5)
                         (h2 : sin (α - β) = - sqrt 10 / 10)
                         (h3 : 0 < α ∧ α < π / 2)
                         (h4 : 0 < β ∧ β < π / 2) :
  cos (2 * β) = 0 :=
  sorry

end cos_double_angle_l92_92938


namespace total_handshakes_l92_92275

theorem total_handshakes (n : ℕ) (h : n = 10) : ∃ k, k = (n * (n - 1)) / 2 ∧ k = 45 :=
by {
  sorry
}

end total_handshakes_l92_92275


namespace sqrt_ratio_simplify_l92_92548

theorem sqrt_ratio_simplify :
  ( (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 ) :=
by
  let sqrt27 := Real.sqrt 27
  let sqrt243 := Real.sqrt 243
  let sqrt75 := Real.sqrt 75
  have h_sqrt27 : sqrt27 = Real.sqrt (3^2 * 3) := by sorry
  have h_sqrt243 : sqrt243 = Real.sqrt (3^5) := by sorry
  have h_sqrt75 : sqrt75 = Real.sqrt (3 * 5^2) := by sorry
  have h_simplified : (sqrt27 + sqrt243) / sqrt75 = 12 / 5 := by sorry
  exact h_simplified

end sqrt_ratio_simplify_l92_92548


namespace primes_solution_l92_92322

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end primes_solution_l92_92322


namespace B_C_work_days_l92_92851

noncomputable def days_for_B_and_C {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) : ℝ :=
  30 / 7

theorem B_C_work_days {A B C : ℝ} (hA : A = 1 / 10) (hA_B : A + B = 1 / 5) (hA_B_C : A + B + C = 1 / 3) :
  days_for_B_and_C hA hA_B hA_B_C = 30 / 7 :=
sorry

end B_C_work_days_l92_92851


namespace bead_bracelet_problem_l92_92923

-- Define the condition Bead A and Bead B are always next to each other
def adjacent (A B : ℕ) (l : List ℕ) : Prop :=
  ∃ (l1 l2 : List ℕ), l = l1 ++ A :: B :: l2 ∨ l = l1 ++ B :: A :: l2

-- Define the context and translate the problem
def bracelet_arrangements (n : ℕ) : ℕ :=
  if n = 8 then 720 else 0

theorem bead_bracelet_problem : bracelet_arrangements 8 = 720 :=
by {
  -- Place proof here
  sorry 
}

end bead_bracelet_problem_l92_92923


namespace ammonium_iodide_molecular_weight_l92_92953

theorem ammonium_iodide_molecular_weight :
  let N := 14.01
  let H := 1.008
  let I := 126.90
  let NH4I_weight := (1 * N) + (4 * H) + (1 * I)
  NH4I_weight = 144.942 :=
by
  -- The proof will go here
  sorry

end ammonium_iodide_molecular_weight_l92_92953


namespace number_of_real_solutions_is_one_l92_92027

noncomputable def num_real_solutions (a b c d : ℝ) : ℕ :=
  let x := Real.sin (a + b + c)
  let y := Real.sin (b + c + d)
  let z := Real.sin (c + d + a)
  let w := Real.sin (d + a + b)
  if (a + b + c + d) % 360 = 0 then 1 else 0

theorem number_of_real_solutions_is_one (a b c d : ℝ) (h : (a + b + c + d) % 360 = 0) :
  num_real_solutions a b c d = 1 :=
by
  sorry

end number_of_real_solutions_is_one_l92_92027


namespace part_a_l92_92150

theorem part_a (x y : ℝ) : x^2 - 2*y^2 = -((x + 2*y)^2 - 2*(x + y)^2) :=
sorry

end part_a_l92_92150


namespace find_a_l92_92421

-- Define the conditions
def parabola_equation (a : ℝ) (x : ℝ) : ℝ := a * x^2
def axis_of_symmetry : ℝ := -2

-- The main theorem: proving the value of a
theorem find_a (a : ℝ) : (axis_of_symmetry = - (1 / (4 * a))) → a = 1/8 :=
by
  intro h
  sorry

end find_a_l92_92421


namespace sequence_a_n_a5_eq_21_l92_92004

theorem sequence_a_n_a5_eq_21 
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  a 5 = 21 :=
by
  sorry

end sequence_a_n_a5_eq_21_l92_92004


namespace binary_equals_octal_l92_92035

-- Define that 1001101 in binary is a specific integer
def binary_value : ℕ := 0b1001101

-- Define that 115 in octal is a specific integer
def octal_value : ℕ := 0o115

-- State the theorem we need to prove
theorem binary_equals_octal : binary_value = octal_value :=
  by sorry

end binary_equals_octal_l92_92035


namespace ratio_of_volumes_l92_92475

theorem ratio_of_volumes (C D : ℚ) (h1: C = (3/4) * C) (h2: D = (5/8) * D) : C / D = 5 / 6 :=
sorry

end ratio_of_volumes_l92_92475


namespace system_solution_unique_l92_92084

theorem system_solution_unique (w x y z : ℝ) (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := 
sorry

end system_solution_unique_l92_92084


namespace mersenne_primes_less_than_1000_l92_92763

open Nat

-- Definitions and Conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

-- Theorem Statement
theorem mersenne_primes_less_than_1000 : {p : ℕ | is_mersenne_prime p ∧ p < 1000} = {3, 7, 31, 127} :=
by
  sorry

end mersenne_primes_less_than_1000_l92_92763


namespace pages_per_day_l92_92093

-- Define the given conditions
def total_pages : ℕ := 957
def total_days : ℕ := 47

-- State the theorem based on the conditions and the required proof
theorem pages_per_day (p : ℕ) (d : ℕ) (h1 : p = total_pages) (h2 : d = total_days) :
  p / d = 20 := by
  sorry

end pages_per_day_l92_92093


namespace temperature_conversion_l92_92235

theorem temperature_conversion (F : ℝ) (C : ℝ) : 
  F = 95 → 
  C = (F - 32) * 5 / 9 → 
  C = 35 := by
  intro hF hC
  sorry

end temperature_conversion_l92_92235


namespace emily_annual_income_l92_92194

variables {q I : ℝ}

theorem emily_annual_income (h1 : (0.01 * q * 30000 + 0.01 * (q + 3) * (I - 30000)) = ((q + 0.75) * 0.01 * I)) : 
  I = 40000 := 
by
  sorry

end emily_annual_income_l92_92194


namespace yogurt_amount_l92_92655

namespace SmoothieProblem

def strawberries := 0.2 -- cups
def orange_juice := 0.2 -- cups
def total_ingredients := 0.5 -- cups

def yogurt_used := total_ingredients - (strawberries + orange_juice)

theorem yogurt_amount : yogurt_used = 0.1 :=
by
  unfold yogurt_used strawberries orange_juice total_ingredients
  norm_num
  sorry  -- Proof can be filled in as needed

end SmoothieProblem

end yogurt_amount_l92_92655


namespace election_win_percentage_l92_92387

theorem election_win_percentage (total_votes : ℕ) (james_percentage : ℝ) (additional_votes_needed : ℕ) (votes_needed_to_win_percentage : ℝ) :
    total_votes = 2000 →
    james_percentage = 0.005 →
    additional_votes_needed = 991 →
    votes_needed_to_win_percentage = (1001 / 2000) * 100 →
    votes_needed_to_win_percentage > 50.05 :=
by
  intros h_total_votes h_james_percentage h_additional_votes_needed h_votes_needed_to_win_percentage
  sorry

end election_win_percentage_l92_92387


namespace sqrt_sum_ineq_l92_92898

open Real

theorem sqrt_sum_ineq (a b c d : ℝ) (h : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0)
  (h4 : a + b + c + d = 4) : 
  sqrt (a + b + c) + sqrt (b + c + d) + sqrt (c + d + a) + sqrt (d + a + b) ≥ 6 :=
sorry

end sqrt_sum_ineq_l92_92898


namespace rational_function_eq_l92_92062

theorem rational_function_eq (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 :=
by sorry

end rational_function_eq_l92_92062


namespace not_possible_to_construct_l92_92021

/-- The frame consists of 54 unit segments. -/
def frame_consists_of_54_units : Prop := sorry

/-- Each part of the construction set consists of three unit segments. -/
def part_is_three_units : Prop := sorry

/-- Each vertex of a cube is shared by three edges. -/
def vertex_shares_three_edges : Prop := sorry

/-- Six segments emerge from the center of the cube. -/
def center_has_six_segments : Prop := sorry

/-- It is not possible to construct the frame with exactly 18 parts. -/
theorem not_possible_to_construct
  (h1 : frame_consists_of_54_units)
  (h2 : part_is_three_units)
  (h3 : vertex_shares_three_edges)
  (h4 : center_has_six_segments) : 
  ¬ ∃ (parts : ℕ), parts = 18 :=
sorry

end not_possible_to_construct_l92_92021


namespace parabola_focus_distance_area_l92_92456

theorem parabola_focus_distance_area (p : ℝ) (hp : p > 0)
  (A : ℝ × ℝ) (hA : A.2^2 = 2 * p * A.1)
  (hDist : A.1 + p / 2 = 2 * A.1)
  (hArea : 1/2 * (p / 2) * |A.2| = 1) :
  p = 2 :=
sorry

end parabola_focus_distance_area_l92_92456


namespace rational_sign_product_l92_92697

theorem rational_sign_product (a b c : ℚ) (h : |a| / a + |b| / b + |c| / c = 1) : abc / |abc| = -1 := 
by
  -- Proof to be provided
  sorry

end rational_sign_product_l92_92697


namespace overall_profit_is_600_l92_92502

def grinder_cp := 15000
def mobile_cp := 10000
def laptop_cp := 20000
def camera_cp := 12000

def grinder_loss_percent := 4 / 100
def mobile_profit_percent := 10 / 100
def laptop_loss_percent := 8 / 100
def camera_profit_percent := 15 / 100

def grinder_sp := grinder_cp * (1 - grinder_loss_percent)
def mobile_sp := mobile_cp * (1 + mobile_profit_percent)
def laptop_sp := laptop_cp * (1 - laptop_loss_percent)
def camera_sp := camera_cp * (1 + camera_profit_percent)

def total_cp := grinder_cp + mobile_cp + laptop_cp + camera_cp
def total_sp := grinder_sp + mobile_sp + laptop_sp + camera_sp

def overall_profit_or_loss := total_sp - total_cp

theorem overall_profit_is_600 : overall_profit_or_loss = 600 := by
  sorry

end overall_profit_is_600_l92_92502


namespace avg_nested_l92_92034

def avg (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem avg_nested {x y z : ℕ} :
  avg (avg 2 3 1) (avg 4 1 0) 5 = 26 / 9 :=
by
  sorry

end avg_nested_l92_92034


namespace optimal_discount_savings_l92_92643

theorem optimal_discount_savings : 
  let total_amount := 15000
  let discount1 := 0.30
  let discount2 := 0.15
  let single_discount := 0.40
  let two_successive_discounts := total_amount * (1 - discount1) * (1 - discount2)
  let one_single_discount := total_amount * (1 - single_discount)
  one_single_discount - two_successive_discounts = 75 :=
by
  sorry

end optimal_discount_savings_l92_92643


namespace peach_ratios_and_percentages_l92_92012

def red_peaches : ℕ := 8
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6
def orange_peaches : ℕ := 4
def total_peaches : ℕ := red_peaches + yellow_peaches + green_peaches + orange_peaches

theorem peach_ratios_and_percentages :
  ((green_peaches : ℚ) / total_peaches = 3 / 16) ∧
  ((green_peaches : ℚ) / total_peaches * 100 = 18.75) ∧
  ((yellow_peaches : ℚ) / total_peaches = 7 / 16) ∧
  ((yellow_peaches : ℚ) / total_peaches * 100 = 43.75) :=
by {
  sorry
}

end peach_ratios_and_percentages_l92_92012


namespace spherical_triangle_area_correct_l92_92098

noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ :=
  R^2 * (α + β + γ - Real.pi)

theorem spherical_triangle_area_correct (R α β γ : ℝ) :
  spherical_triangle_area R α β γ = R^2 * (α + β + γ - Real.pi) := by
  sorry

end spherical_triangle_area_correct_l92_92098


namespace incorrect_neg_p_l92_92381

theorem incorrect_neg_p (p : ∀ x : ℝ, x ≥ 1) : ¬ (∀ x : ℝ, x < 1) :=
sorry

end incorrect_neg_p_l92_92381


namespace solve_for_m_l92_92382

-- Define the conditions as hypotheses
def hyperbola_equation (x y : Real) (m : Real) : Prop :=
  (x^2)/(m+9) + (y^2)/9 = 1

def eccentricity (e : Real) (a b : Real) : Prop :=
  e = 2 ∧ e^2 = 1 + (b^2)/(a^2)

-- Prove that m = -36 given the conditions
theorem solve_for_m (m : Real) (h : hyperbola_equation x y m) (h_ecc : eccentricity 2 3 (Real.sqrt (-(m+9)))) :
  m = -36 :=
sorry

end solve_for_m_l92_92382


namespace monthly_rent_calculation_l92_92343

-- Definitions based on the problem conditions
def investment_amount : ℝ := 20000
def desired_annual_return_rate : ℝ := 0.06
def annual_property_taxes : ℝ := 650
def maintenance_percentage : ℝ := 0.15

-- Theorem stating the mathematically equivalent problem
theorem monthly_rent_calculation : 
  let required_annual_return := desired_annual_return_rate * investment_amount
  let total_annual_earnings := required_annual_return + annual_property_taxes
  let monthly_earnings_target := total_annual_earnings / 12
  let monthly_rent := monthly_earnings_target / (1 - maintenance_percentage)
  monthly_rent = 181.38 :=
by
  sorry

end monthly_rent_calculation_l92_92343


namespace cost_of_popsicle_sticks_l92_92046

theorem cost_of_popsicle_sticks
  (total_money : ℕ)
  (cost_of_molds : ℕ)
  (cost_per_bottle : ℕ)
  (popsicles_per_bottle : ℕ)
  (sticks_used : ℕ)
  (sticks_left : ℕ)
  (number_of_sticks : ℕ)
  (remaining_money : ℕ) :
  total_money = 10 →
  cost_of_molds = 3 →
  cost_per_bottle = 2 →
  popsicles_per_bottle = 20 →
  sticks_left = 40 →
  number_of_sticks = 100 →
  remaining_money = total_money - cost_of_molds - (sticks_used / popsicles_per_bottle * cost_per_bottle) →
  sticks_used = number_of_sticks - sticks_left →
  remaining_money = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end cost_of_popsicle_sticks_l92_92046


namespace equal_side_length_is_4_or_10_l92_92752

-- Define the conditions
def isosceles_triangle (base_length equal_side_length : ℝ) :=
  base_length = 7 ∧
  (equal_side_length > base_length ∧ equal_side_length - base_length = 3) ∨
  (equal_side_length < base_length ∧ base_length - equal_side_length = 3)

-- Lean 4 statement to prove
theorem equal_side_length_is_4_or_10 (base_length equal_side_length : ℝ) 
  (h : isosceles_triangle base_length equal_side_length) : 
  equal_side_length = 4 ∨ equal_side_length = 10 :=
by 
  sorry

end equal_side_length_is_4_or_10_l92_92752


namespace sequence_arithmetic_progression_l92_92444

theorem sequence_arithmetic_progression (b : ℕ → ℕ) (b1_eq : b 1 = 1) (recurrence : ∀ n, b (n + 2) = b (n + 1) * b n + 1) : b 2 = 1 ↔ 
  ∃ d : ℕ, ∀ n, b (n + 1) - b n = d :=
sorry

end sequence_arithmetic_progression_l92_92444


namespace infinite_series_sum_l92_92734

noncomputable def partial_sum (n : ℕ) : ℚ := (2 * n - 1) / (n * (n + 1) * (n + 2))

theorem infinite_series_sum : (∑' n, partial_sum (n + 1)) = 3 / 4 :=
by
  sorry

end infinite_series_sum_l92_92734


namespace dentist_cleaning_cost_l92_92492

theorem dentist_cleaning_cost
  (F: ℕ)
  (C: ℕ)
  (B: ℕ)
  (tooth_extraction_cost: ℕ)
  (HC1: F = 120)
  (HC2: B = 5 * F)
  (HC3: tooth_extraction_cost = 290)
  (HC4: B = C + 2 * F + tooth_extraction_cost) :
  C = 70 :=
by
  sorry

end dentist_cleaning_cost_l92_92492


namespace exists_infinite_diff_but_not_sum_of_kth_powers_l92_92196

theorem exists_infinite_diff_but_not_sum_of_kth_powers (k : ℕ) (hk : k > 1) :
  ∃ (infinitely_many x : ℕ), (∃ (a b : ℕ), x = a^k - b^k) ∧ ¬ (∃ (c d : ℕ), x = c^k + d^k) :=
  sorry

end exists_infinite_diff_but_not_sum_of_kth_powers_l92_92196


namespace difference_in_squares_l92_92519

noncomputable def radius_of_circle (x y h R : ℝ) : Prop :=
  5 * x^2 - 4 * x * h + h^2 = R^2 ∧ 5 * y^2 + 4 * y * h + h^2 = R^2

theorem difference_in_squares (x y h R : ℝ) (h_radius : radius_of_circle x y h R) :
  2 * x - 2 * y = (8/5 : ℝ) * h :=
by
  sorry

end difference_in_squares_l92_92519


namespace x_gt_zero_sufficient_but_not_necessary_l92_92552

theorem x_gt_zero_sufficient_but_not_necessary (x : ℝ): 
  (x > 0 → x ≠ 0) ∧ (x ≠ 0 → ¬ (x > 0)) → 
  ((x > 0 ↔ x ≠ 0) = false) :=
by
  intro h
  sorry

end x_gt_zero_sufficient_but_not_necessary_l92_92552


namespace find_multiple_of_q_l92_92595

theorem find_multiple_of_q
  (q : ℕ)
  (x : ℕ := 55 + 2 * q)
  (y : ℕ)
  (m : ℕ)
  (h1 : y = m * q + 41)
  (h2 : x = y)
  (h3 : q = 7) : m = 4 :=
by
  sorry

end find_multiple_of_q_l92_92595


namespace roots_of_quadratic_eq_l92_92809

theorem roots_of_quadratic_eq {x y : ℝ} (h1 : x + y = 10) (h2 : (x - y) * (x + y) = 48) : 
    ∃ a b c : ℝ, (a ≠ 0) ∧ (x^2 - a*x + b = 0) ∧ (y^2 - a*y + b = 0) ∧ b = 19.24 := 
by
  sorry

end roots_of_quadratic_eq_l92_92809


namespace words_memorized_on_fourth_day_l92_92986

-- Definitions for the conditions
def first_three_days_words (k : ℕ) : ℕ := 3 * k
def last_four_days_words (k : ℕ) : ℕ := 4 * k
def fourth_day_words (k : ℕ) (a : ℕ) : ℕ := a
def last_three_days_words (k : ℕ) (a : ℕ) : ℕ := last_four_days_words k - a

-- Problem Statement
theorem words_memorized_on_fourth_day {k a : ℕ} (h1 : first_three_days_words k + last_four_days_words k > 100)
    (h2 : first_three_days_words k * 6 = 5 * (4 * k - a))
    (h3 : 21 * (2 * k / 3) = 100) : 
    a = 10 :=
by 
  sorry

end words_memorized_on_fourth_day_l92_92986


namespace triangle_inequality_l92_92941

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a / (b + c) + b / (c + a) + c / (a + b) ≤ 2 :=
sorry

end triangle_inequality_l92_92941


namespace triangle_side_ratios_l92_92212

theorem triangle_side_ratios
    (A B C : ℝ) (a b c : ℝ)
    (h1 : 2 * b * Real.sin (2 * A) = a * Real.sin B)
    (h2 : c = 2 * b) :
    a / b = 2 :=
by
  sorry

end triangle_side_ratios_l92_92212


namespace calculate_fraction_l92_92744

theorem calculate_fraction: (1 / (2 + 1 / (3 + 1 / 4))) = 13 / 30 := by
  sorry

end calculate_fraction_l92_92744


namespace problem_statement_l92_92928

-- Definition of the conditions
variables {a : ℝ} (h₀ : a > 0) (h₁ : a ≠ 1)

-- The Lean 4 statement for the problem
theorem problem_statement (h : 0 < a ∧ a < 1) : 
  (∀ x y : ℝ, x < y → a^x > a^y) → 
  (∀ x : ℝ, (2 - a) * x^3 > 0) ∧ 
  (∀ x : ℝ, (2 - a) * x^3 > 0 → 0 < a ∧ a < 2 ∧ (∀ x y : ℝ, x < y → a^x > a^y) → False) :=
by
  intros
  sorry

end problem_statement_l92_92928


namespace part_a_part_b_l92_92989

variable (p : ℕ)
variable (h1 : prime p)
variable (h2 : p > 3)

theorem part_a : (p + 1) % 4 = 0 ∨ (p - 1) % 4 = 0 :=
sorry

theorem part_b : ¬ ((p + 1) % 5 = 0 ∨ (p - 1) % 5 = 0) :=
sorry

end part_a_part_b_l92_92989


namespace stickers_total_l92_92436

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end stickers_total_l92_92436


namespace smallest_positive_integer_in_linear_combination_l92_92379

theorem smallest_positive_integer_in_linear_combination :
  ∃ m n : ℤ, 2016 * m + 43200 * n = 24 :=
by
  sorry

end smallest_positive_integer_in_linear_combination_l92_92379


namespace team_e_speed_l92_92525

-- Definitions and conditions
variables (v t : ℝ)
def distance_team_e := 300 = v * t
def distance_team_a := 300 = (v + 5) * (t - 3)

-- The theorem statement: Prove that given the conditions, Team E's speed is 20 mph
theorem team_e_speed (h1 : distance_team_e v t) (h2 : distance_team_a v t) : v = 20 :=
by
  sorry -- proof steps are omitted as requested

end team_e_speed_l92_92525


namespace melanie_batches_l92_92859

theorem melanie_batches (total_brownies_given: ℕ)
                        (brownies_per_batch: ℕ)
                        (fraction_bake_sale: ℚ)
                        (fraction_container: ℚ)
                        (remaining_brownies_given: ℕ) :
                        brownies_per_batch = 20 →
                        fraction_bake_sale = 3/4 →
                        fraction_container = 3/5 →
                        total_brownies_given = 20 →
                        (remaining_brownies_given / (brownies_per_batch * (1 - fraction_bake_sale) * (1 - fraction_container))) = 10 :=
by
  sorry

end melanie_batches_l92_92859


namespace sam_gave_fraction_l92_92558

/-- Given that Mary bought 1500 stickers and shared them between Susan, Andrew, 
and Sam in the ratio 1:1:3. After Sam gave some stickers to Andrew, Andrew now 
has 900 stickers. Prove that the fraction of Sam's stickers given to Andrew is 2/3. -/
theorem sam_gave_fraction (total_stickers : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
    (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (final_B : ℕ) (given_stickers : ℕ) :
    total_stickers = 1500 → ratio_A = 1 → ratio_B = 1 → ratio_C = 3 →
    initial_A = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_B = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_C = 3 * (total_stickers / (ratio_A + ratio_B + ratio_C)) →
    final_B = 900 →
    initial_B + given_stickers = final_B →
    given_stickers / initial_C = 2 / 3 :=
by
  intros
  sorry

end sam_gave_fraction_l92_92558


namespace minimum_perimeter_area_l92_92698

-- Define the focus point F of the parabola and point A
def F : ℝ × ℝ := (1, 0)  -- Focus for the parabola y² = 4x is (1, 0)
def A : ℝ × ℝ := (5, 4)

-- Parabola definition as a set of points (x, y) such that y² = 4x
def is_on_parabola (B : ℝ × ℝ) : Prop := B.2 * B.2 = 4 * B.1

-- The area of triangle ABF
def triangle_area (A B F : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 - B.1) * (A.2 - F.2) - (A.1 - F.1) * (A.2 - B.2))

-- Statement: The area of ∆ABF is 2 when the perimeter of ∆ABF is minimum
theorem minimum_perimeter_area (B : ℝ × ℝ) (hB : is_on_parabola B) 
  (hA_B_perimeter_min : ∀ (C : ℝ × ℝ), is_on_parabola C → 
                        (dist A C + dist C F ≥ dist A B + dist B F)) : 
  triangle_area A B F = 2 := 
sorry

end minimum_perimeter_area_l92_92698


namespace triangle_area_is_12_5_l92_92861

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨5, 0⟩
def N : Point := ⟨0, 5⟩
noncomputable def P (x y : ℝ) (h : x + y = 8) : Point := ⟨x, y⟩

noncomputable def area_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

theorem triangle_area_is_12_5 (x y : ℝ) (h : x + y = 8) :
  area_triangle M N (P x y h) = 12.5 :=
sorry

end triangle_area_is_12_5_l92_92861


namespace initial_integer_l92_92261

theorem initial_integer (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_integer_l92_92261


namespace largest_circle_radius_l92_92601

noncomputable def largest_inscribed_circle_radius (AB BC CD DA : ℝ) : ℝ :=
  let s := (AB + BC + CD + DA) / 2
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))
  A / s

theorem largest_circle_radius {AB BC CD DA : ℝ} (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 13)
  : largest_inscribed_circle_radius AB BC CD DA = 3 * Real.sqrt 245 / 10 :=
by
  simp [largest_inscribed_circle_radius, hAB, hBC, hCD, hDA]
  sorry

end largest_circle_radius_l92_92601


namespace range_of_a_l92_92672

theorem range_of_a (x y a : ℝ) (h1 : x < y) (h2 : (a - 3) * x > (a - 3) * y) : a < 3 :=
sorry

end range_of_a_l92_92672


namespace polynomial_root_divisibility_l92_92693

noncomputable def p (x : ℤ) (a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

theorem polynomial_root_divisibility (a b c : ℤ) (h : ∃ u v : ℤ, p 0 a b c = (u * v * u * v)) :
  2 * (p (-1) a b c) ∣ (p 1 a b c + p (-1) a b c - 2 * (1 + p 0 a b c)) :=
sorry

end polynomial_root_divisibility_l92_92693


namespace percentage_shaded_is_14_29_l92_92485

noncomputable def side_length : ℝ := 20
noncomputable def rect_length : ℝ := 35
noncomputable def rect_width : ℝ := side_length
noncomputable def rect_area : ℝ := rect_length * rect_width
noncomputable def overlap_length : ℝ := 2 * side_length - rect_length
noncomputable def overlap_area : ℝ := overlap_length * side_length
noncomputable def shaded_percentage : ℝ := (overlap_area / rect_area) * 100

theorem percentage_shaded_is_14_29 :
  shaded_percentage = 14.29 :=
sorry

end percentage_shaded_is_14_29_l92_92485


namespace scientific_notation_of_393000_l92_92101

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end scientific_notation_of_393000_l92_92101


namespace negative_expressions_l92_92944

-- Define the approximated values for P, Q, R, S, and T
def P : ℝ := 3.5
def Q : ℝ := 1.1
def R : ℝ := -0.1
def S : ℝ := 0.9
def T : ℝ := 1.5

-- State the theorem to be proved
theorem negative_expressions : 
  (R / (P * Q) < 0) ∧ ((S + T) / R < 0) :=
by
  sorry

end negative_expressions_l92_92944


namespace rectangle_area_ratio_is_three_l92_92114

variables {a b : ℝ}

-- Rectangle ABCD with midpoint F on CD, BC = 3 * BE
def rectangle_midpoint_condition (CD_length : ℝ) (BC_length : ℝ) (BE_length : ℝ) (F_midpoint : Prop) :=
  F_midpoint ∧ BC_length = 3 * BE_length

-- Areas and the ratio
def area_rectangle (CD_length BC_length : ℝ) : ℝ :=
  CD_length * BC_length

def area_shaded (a b : ℝ) : ℝ :=
  2 * a * b

theorem rectangle_area_ratio_is_three (h : rectangle_midpoint_condition (2 * a) (3 * b) b (F_midpoint := True)) :
  area_rectangle (2 * a) (3 * b) = 3 * area_shaded a b :=
by
  unfold rectangle_midpoint_condition at h
  unfold area_rectangle area_shaded
  rw [←mul_assoc, ←mul_assoc]
  sorry

end rectangle_area_ratio_is_three_l92_92114


namespace largest_angle_of_obtuse_isosceles_triangle_l92_92658

variables (X Y Z : ℝ)

def is_triangle (X Y Z : ℝ) : Prop := X + Y + Z = 180
def is_isosceles_triangle (X Y : ℝ) : Prop := X = Y
def is_obtuse_triangle (X Y Z : ℝ) : Prop := X > 90 ∨ Y > 90 ∨ Z > 90

theorem largest_angle_of_obtuse_isosceles_triangle
  (X Y Z : ℝ)
  (h1 : is_triangle X Y Z)
  (h2 : is_isosceles_triangle X Y)
  (h3 : X = 30)
  (h4 : is_obtuse_triangle X Y Z) :
  Z = 120 :=
sorry

end largest_angle_of_obtuse_isosceles_triangle_l92_92658


namespace percentage_error_in_side_l92_92166

theorem percentage_error_in_side {S S' : ℝ}
  (hs : S > 0)
  (hs' : S' > S)
  (h_area_error : (S'^2 - S^2) / S^2 * 100 = 90.44) :
  ((S' - S) / S * 100) = 38 :=
by
  sorry

end percentage_error_in_side_l92_92166


namespace find_f_2011_l92_92853

def f: ℝ → ℝ :=
sorry

axiom f_periodicity (x : ℝ) : f (x + 3) = -f x
axiom f_initial_value : f 4 = -2

theorem find_f_2011 : f 2011 = 2 :=
by
  sorry

end find_f_2011_l92_92853


namespace banana_orange_equivalence_l92_92995

/-- Given that 3/4 of 12 bananas are worth 9 oranges,
    prove that 1/3 of 9 bananas are worth 3 oranges. -/
theorem banana_orange_equivalence :
  (3 / 4) * 12 = 9 → (1 / 3) * 9 = 3 :=
by
  intro h
  have h1 : (9 : ℝ) = 9 := by sorry -- This is from the provided condition
  have h2 : 1 * 9 = 1 * 9 := by sorry -- Deducing from h1: 9 = 9
  have h3 : 9 = 9 := by sorry -- concluding 9 bananas = 9 oranges
  have h4 : (1 / 3) * 9 = 3 := by sorry -- 1/3 of 9
  exact h4

end banana_orange_equivalence_l92_92995


namespace ratio_dislikes_to_likes_l92_92315

theorem ratio_dislikes_to_likes 
  (D : ℕ) 
  (h1 : D + 1000 = 2600) 
  (h2 : 3000 > 0) : 
  D / 3000 = 8 / 15 :=
by sorry

end ratio_dislikes_to_likes_l92_92315


namespace binom_20_5_l92_92539

-- Definition of the binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Problem statement
theorem binom_20_5 : binomial_coefficient 20 5 = 7752 := 
by {
  -- Proof goes here
  sorry
}

end binom_20_5_l92_92539


namespace smallest_number_is_3_l92_92389

theorem smallest_number_is_3 (a b c : ℝ) (h1 : (a + b + c) / 3 = 7) (h2 : a = 9 ∨ b = 9 ∨ c = 9) : min (min a b) c = 3 := 
sorry

end smallest_number_is_3_l92_92389


namespace roots_of_quadratic_l92_92376

theorem roots_of_quadratic (x : ℝ) : x^2 + x = 0 ↔ (x = 0 ∨ x = -1) :=
by sorry

end roots_of_quadratic_l92_92376


namespace probability_neither_red_blue_purple_l92_92082

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_neither_red_blue_purple :
  (total_balls - (red_balls + blue_balls + purple_balls)) / total_balls = 35 / 48 := 
by 
  /- Proof details are not necessary -/
  sorry

end probability_neither_red_blue_purple_l92_92082


namespace find_a_for_inequality_l92_92843

theorem find_a_for_inequality (a : ℚ) :
  (∀ x : ℚ, (ax / (x - 1)) < 1 ↔ (x < 1 ∨ x > 2)) → a = 1/2 :=
by
  sorry

end find_a_for_inequality_l92_92843


namespace Teresa_current_age_l92_92160

-- Definitions of the conditions
def Morio_current_age := 71
def Morio_age_when_Michiko_born := 38
def Teresa_age_when_Michiko_born := 26

-- Definition of Michiko's current age
def Michiko_current_age := Morio_current_age - Morio_age_when_Michiko_born

-- The Theorem statement
theorem Teresa_current_age : Teresa_age_when_Michiko_born + Michiko_current_age = 59 :=
by
  -- Skip the proof
  sorry

end Teresa_current_age_l92_92160


namespace arithmetic_sequence_a4_possible_values_l92_92398

theorem arithmetic_sequence_a4_possible_values (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 * a 5 = 9)
  (h3 : a 2 = 3) : 
  a 4 = 3 ∨ a 4 = 7 := 
by 
  sorry

end arithmetic_sequence_a4_possible_values_l92_92398


namespace initial_decaf_percentage_l92_92652

theorem initial_decaf_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 100) 
  (h3 : (x / 100 * 400) + 60 = 220) :
  x = 40 :=
by sorry

end initial_decaf_percentage_l92_92652


namespace system_solution_conditions_l92_92836

theorem system_solution_conditions (α1 α2 α3 α4 : ℝ) :
  (α1 = α4 ∨ α2 = α3) ↔ 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 + x2 = α1 * α2 ∧
    x1 + x3 = α1 * α3 ∧
    x1 + x4 = α1 * α4 ∧
    x2 + x3 = α2 * α3 ∧
    x2 + x4 = α2 * α4 ∧
    x3 + x4 = α3 * α4 ∧
    x1 = x2 ∧
    x2 = x3 ∧
    x1 = α2^2 / 2 ∧
    x3 = α2^2 / 2 ∧
    x4 = α2 * α4 - (α2^2 / 2) ) :=
by sorry

end system_solution_conditions_l92_92836


namespace min_C2_D2_at_36_l92_92074

noncomputable def min_value_C2_D2 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : ℝ :=
  let C := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
  let D := (Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3))
  C^2 - D^2

theorem min_C2_D2_at_36 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : 
  min_value_C2_D2 x y z hx hy hz = 36 :=
sorry

end min_C2_D2_at_36_l92_92074


namespace proof_S5_l92_92459

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a1, ∀ n, a (n + 1) = a1 * q ^ (n + 1)

theorem proof_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ) : 
  (geometric_sequence a) → 
  (a 2 * a 5 = 2 * a 3) → 
  ((a 4 + 2 * a 7) / 2 = 5 / 4) → 
  (S 5 = a1 * (1 - (1 / 2) ^ 5) / (1 - 1 / 2)) → 
  S 5 = 31 := 
by sorry

end proof_S5_l92_92459


namespace number_of_people_l92_92984

theorem number_of_people
  (x y : ℕ)
  (h1 : x + y = 28)
  (h2 : 2 * x + 4 * y = 92) :
  x = 10 :=
by
  sorry

end number_of_people_l92_92984


namespace factorable_polynomial_with_integer_coeffs_l92_92634

theorem factorable_polynomial_with_integer_coeffs (m : ℤ) : 
  ∃ A B C D E F : ℤ, 
  (A * D = 1) ∧ (B * E = 0) ∧ (A * E + B * D = 5) ∧ 
  (A * F + C * D = 1) ∧ (B * F + C * E = 2 * m) ∧ (C * F = -10) ↔ m = 5 := sorry

end factorable_polynomial_with_integer_coeffs_l92_92634


namespace price_of_mixture_l92_92476

theorem price_of_mixture (P1 P2 P3 : ℝ) (h1 : P1 = 126) (h2 : P2 = 135) (h3 : P3 = 175.5) : 
  (P1 + P2 + 2 * P3) / 4 = 153 :=
by 
  -- Main goal is to show (126 + 135 + 2 * 175.5) / 4 = 153
  sorry

end price_of_mixture_l92_92476


namespace circle_intersection_area_l92_92260

theorem circle_intersection_area
  (r : ℝ)
  (θ : ℝ)
  (a b c : ℝ)
  (h_r : r = 5)
  (h_θ : θ = π / 2)
  (h_expr : a * Real.sqrt b + c * π = 5 * 5 * π / 2 - 5 * 5 * Real.sqrt 3 / 2 ) :
  a + b + c = -9.5 :=
by
  sorry

end circle_intersection_area_l92_92260


namespace doughnut_price_l92_92229

theorem doughnut_price
  (K C B : ℕ)
  (h1: K = 4 * C + 5)
  (h2: K = 5 * C - 6)
  (h3: K = 2 * C + 3 * B) :
  B = 9 := 
sorry

end doughnut_price_l92_92229


namespace area_of_smaller_circle_l92_92170

noncomputable def radius_large_circle (x : ℝ) : ℝ := 2 * x
noncomputable def radius_small_circle (y : ℝ) : ℝ := y

theorem area_of_smaller_circle 
(pa ab : ℝ)
(r : ℝ)
(area : ℝ) 
(h1 : pa = 5) 
(h2 : ab = 5) 
(h3 : radius_large_circle r = 2 * radius_small_circle r)
(h4 : 2 * radius_small_circle r + radius_large_circle r = 10)
(h5 : area = Real.pi * (radius_small_circle r)^2) 
: area = 6.25 * Real.pi :=
by
  sorry

end area_of_smaller_circle_l92_92170


namespace smallest_n_l92_92489

theorem smallest_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n = 3 * k) (h3 : ∃ m : ℕ, 3 * n = 5 * m) : n = 15 :=
sorry

end smallest_n_l92_92489


namespace train_crossing_platform_l92_92845

/-- Given a train crosses a 100 m platform in 15 seconds, and the length of the train is 350 m,
    prove that the train takes 20 seconds to cross a second platform of length 250 m. -/
theorem train_crossing_platform (dist1 dist2 l_t t1 t2 : ℝ) (h1 : dist1 = 100) (h2 : dist2 = 250) (h3 : l_t = 350) (h4 : t1 = 15) :
  t2 = 20 :=
sorry

end train_crossing_platform_l92_92845


namespace range_a_l92_92675

theorem range_a (a : ℝ) : (∀ x, x > 0 → x^2 - a * x + 1 > 0) → -2 < a ∧ a < 2 := by
  sorry

end range_a_l92_92675


namespace num_of_cows_is_7_l92_92438

variables (C H : ℕ)

-- Define the conditions
def cow_legs : ℕ := 4 * C
def chicken_legs : ℕ := 2 * H
def cow_heads : ℕ := C
def chicken_heads : ℕ := H

def total_legs : ℕ := cow_legs C + chicken_legs H
def total_heads : ℕ := cow_heads C + chicken_heads H
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 14

-- The theorem to be proved
theorem num_of_cows_is_7 (h : legs_condition C H) : C = 7 :=
by sorry

end num_of_cows_is_7_l92_92438


namespace solve_for_x_l92_92378

theorem solve_for_x (x : ℚ) (h : (x - 3) / (x + 2) + (3 * x - 9) / (x - 3) = 2) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l92_92378


namespace room_area_in_square_meters_l92_92138

theorem room_area_in_square_meters :
  ∀ (length_ft width_ft : ℝ), 
  (length_ft = 15) → 
  (width_ft = 8) → 
  (1 / 9 * 0.836127 = 0.092903) → 
  (length_ft * width_ft * 0.092903 = 11.14836) :=
by
  intros length_ft width_ft h_length h_width h_conversion
  -- sorry to skip the proof steps.
  sorry

end room_area_in_square_meters_l92_92138


namespace no_such_f_exists_l92_92342

theorem no_such_f_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x : ℝ), f (f x) = x^2 - 2 := by
  sorry

end no_such_f_exists_l92_92342


namespace alpha_beta_sum_l92_92450

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102 * x + 2021) / (x^2 + 89 * x - 3960)) : α + β = 176 := by
  sorry

end alpha_beta_sum_l92_92450


namespace initial_people_employed_l92_92515

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

end initial_people_employed_l92_92515


namespace election_winner_won_by_votes_l92_92106

theorem election_winner_won_by_votes (V : ℝ) (winner_votes : ℝ) (loser_votes : ℝ)
    (h1 : winner_votes = 0.62 * V)
    (h2 : winner_votes = 930)
    (h3 : loser_votes = 0.38 * V)
    : winner_votes - loser_votes = 360 := 
  sorry

end election_winner_won_by_votes_l92_92106


namespace expand_polynomial_product_l92_92783

variable (x : ℝ)

def P (x : ℝ) : ℝ := 5 * x ^ 2 + 3 * x - 4
def Q (x : ℝ) : ℝ := 6 * x ^ 3 + 2 * x ^ 2 - x + 7

theorem expand_polynomial_product :
  (P x) * (Q x) = 30 * x ^ 5 + 28 * x ^ 4 - 23 * x ^ 3 + 24 * x ^ 2 + 25 * x - 28 :=
by
  sorry

end expand_polynomial_product_l92_92783


namespace divisor_of_number_l92_92368

theorem divisor_of_number : 
  ∃ D, 
    let x := 75 
    let R' := 7 
    let Q := R' + 8 
    x = D * Q + 0 :=
by
  sorry

end divisor_of_number_l92_92368


namespace first_digit_power_l92_92621

theorem first_digit_power (n : ℕ) (h : ∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) :
  (∃ k' : ℕ, 1 * 10^k' ≤ 5^n ∧ 5^n < 2 * 10^k') :=
sorry

end first_digit_power_l92_92621


namespace trisect_chord_exists_l92_92622

noncomputable def distance (O P : Point) : ℝ := sorry
def trisect (P : Point) (A B : Point) : Prop := 2 * (distance A P) = distance P B

-- Main theorem based on the given conditions and conclusions
theorem trisect_chord_exists (O P : Point) (r : ℝ) (hP_in_circle : distance O P < r) :
  (∃ A B : Point, trisect P A B) ↔ 
  (distance O P > r / 3 ∨ distance O P = r / 3) :=
by
  sorry

end trisect_chord_exists_l92_92622


namespace school_minimum_payment_l92_92615

noncomputable def individual_ticket_price : ℝ := 6
noncomputable def group_ticket_price : ℝ := 40
noncomputable def discount : ℝ := 0.9
noncomputable def students : ℕ := 1258

-- Define the minimum amount the school should pay
noncomputable def minimum_amount := 4536

theorem school_minimum_payment :
  (students / 10 : ℝ) * group_ticket_price * discount + 
  (students % 10) * individual_ticket_price * discount = minimum_amount := sorry

end school_minimum_payment_l92_92615


namespace ab_a4_b4_divisible_by_30_l92_92418

theorem ab_a4_b4_divisible_by_30 (a b : Int) : 30 ∣ a * b * (a^4 - b^4) := 
by
  sorry

end ab_a4_b4_divisible_by_30_l92_92418


namespace curve_is_line_l92_92358

def curve_theta (theta : ℝ) : Prop :=
  theta = Real.pi / 4

theorem curve_is_line : curve_theta θ → (curve_type = "line") :=
by
  intros h
  cases h
  -- This is where the proof would go, but we'll use a placeholder for now.
  -- The essence of the proof will show that all points making an angle of π/4 with the x-axis lie on a line.
  exact sorry

end curve_is_line_l92_92358


namespace billy_gaming_percentage_l92_92779

-- Define the conditions
def free_time_per_day := 8
def days_in_weekend := 2
def total_free_time := free_time_per_day * days_in_weekend
def books_read := 3
def pages_per_book := 80
def reading_rate := 60 -- pages per hour
def total_pages_read := books_read * pages_per_book
def reading_time := total_pages_read / reading_rate
def gaming_time := total_free_time - reading_time
def gaming_percentage := (gaming_time / total_free_time) * 100

-- State the theorem
theorem billy_gaming_percentage : gaming_percentage = 75 := by
  sorry

end billy_gaming_percentage_l92_92779


namespace total_movies_correct_l92_92704

def num_movies_Screen1 : Nat := 3
def num_movies_Screen2 : Nat := 4
def num_movies_Screen3 : Nat := 2
def num_movies_Screen4 : Nat := 3
def num_movies_Screen5 : Nat := 5
def num_movies_Screen6 : Nat := 2

def total_movies : Nat :=
  num_movies_Screen1 + num_movies_Screen2 + num_movies_Screen3 + num_movies_Screen4 + num_movies_Screen5 + num_movies_Screen6

theorem total_movies_correct :
  total_movies = 19 :=
by 
  sorry

end total_movies_correct_l92_92704


namespace distance_from_origin_l92_92540

theorem distance_from_origin (A : ℝ) (h : |A - 0| = 4) : A = 4 ∨ A = -4 :=
by {
  sorry
}

end distance_from_origin_l92_92540


namespace parallel_vectors_solution_l92_92157

theorem parallel_vectors_solution 
  (x : ℝ) 
  (a : ℝ × ℝ := (-1, 3)) 
  (b : ℝ × ℝ := (x, 1)) 
  (h : ∃ k : ℝ, a = k • b) :
  x = -1 / 3 :=
by
  sorry

end parallel_vectors_solution_l92_92157


namespace tim_bought_two_appetizers_l92_92819

-- Definitions of the conditions.
def total_spending : ℝ := 50
def portion_spent_on_entrees : ℝ := 0.80
def entree_cost : ℝ := total_spending * portion_spent_on_entrees
def appetizer_cost : ℝ := 5
def appetizer_spending : ℝ := total_spending - entree_cost

-- The statement to prove: that Tim bought 2 appetizers.
theorem tim_bought_two_appetizers :
  appetizer_spending / appetizer_cost = 2 := 
by
  sorry

end tim_bought_two_appetizers_l92_92819


namespace length_BF_l92_92347

-- Define the geometrical configuration
structure Point :=
  (x : ℝ) (y : ℝ)

def A := Point.mk 0 0
def B := Point.mk 6 4.8
def C := Point.mk 12 0
def D := Point.mk 3 (-6)
def E := Point.mk 3 0
def F := Point.mk 6 0

-- Define given conditions
def AE := (3 : ℝ)
def CE := (9 : ℝ)
def DE := (6 : ℝ)
def AC := AE + CE

theorem length_BF : (BF = (72 / 7 : ℝ)) :=
by
  sorry

end length_BF_l92_92347


namespace no_solution_abs_val_l92_92256

theorem no_solution_abs_val (x : ℝ) : ¬(∃ x : ℝ, |5 * x| + 7 = 0) :=
sorry

end no_solution_abs_val_l92_92256


namespace missed_questions_l92_92270

-- Define variables
variables (a b c T : ℕ) (X Y Z : ℝ)
variables (h1 : a + b + c = T) 
          (h2 : 0 ≤ X ∧ X ≤ 100) 
          (h3 : 0 ≤ Y ∧ Y ≤ 100) 
          (h4 : 0 ≤ Z ∧ Z ≤ 100) 
          (h5 : 6 * (a * (100 - X) / 500 + 2 * b * (100 - Y) / 500 + 3 * c * (100 - Z) / 500) = 216)

-- Define the theorem
theorem missed_questions : 5 * (a * (100 - X) / 500 + b * (100 - Y) / 500 + c * (100 - Z) / 500) = 180 :=
by sorry

end missed_questions_l92_92270


namespace arithmetic_evaluation_l92_92410

theorem arithmetic_evaluation :
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end arithmetic_evaluation_l92_92410


namespace total_cans_in_display_l92_92653

-- Definitions and conditions
def first_term : ℕ := 30
def second_term : ℕ := 27
def nth_term : ℕ := 3
def common_difference : ℕ := second_term - first_term

-- Statement of the problem
theorem total_cans_in_display : 
  ∃ (n : ℕ), nth_term = first_term + (n - 1) * common_difference ∧
  (2 * 165 = n * (first_term + nth_term)) :=
by
  sorry

end total_cans_in_display_l92_92653


namespace jack_morning_emails_l92_92423

theorem jack_morning_emails (x : ℕ) (aft_mails eve_mails total_morn_eve : ℕ) (h1: aft_mails = 4) (h2: eve_mails = 8) (h3: total_morn_eve = 11) :
  x = total_morn_eve - eve_mails :=
by 
  sorry

end jack_morning_emails_l92_92423


namespace number_of_paths_l92_92472

theorem number_of_paths (r u : ℕ) (h_r : r = 5) (h_u : u = 4) : 
  (Nat.choose (r + u) u) = 126 :=
by
  -- The proof is omitted, as requested.
  sorry

end number_of_paths_l92_92472


namespace exp_pos_for_all_x_l92_92258

theorem exp_pos_for_all_x (h : ¬ ∃ x_0 : ℝ, Real.exp x_0 ≤ 0) : ∀ x : ℝ, Real.exp x > 0 :=
by
  sorry

end exp_pos_for_all_x_l92_92258


namespace find_p_value_l92_92662

theorem find_p_value (D E F : ℚ) (α β : ℚ)
  (h₁: D ≠ 0) 
  (h₂: E^2 - 4*D*F ≥ 0) 
  (hαβ: D * (α^2 + β^2) + E * (α + β) + 2*F = 2*D^2 - E^2) :
  ∃ p : ℚ, (p = (2*D*F - E^2 - 2*D^2) / D^2) :=
sorry

end find_p_value_l92_92662


namespace intersection_M_N_l92_92730

def M : Set ℝ := { x | |x - 2| ≤ 1 }
def N : Set ℝ := { x | x^2 - x - 6 ≥ 0 }

theorem intersection_M_N : M ∩ N = {3} := by
  sorry

end intersection_M_N_l92_92730


namespace total_time_in_pool_is_29_minutes_l92_92666

noncomputable def calculate_total_time_in_pool : ℝ :=
  let jerry := 3             -- Jerry's time in minutes
  let elaine := 2 * jerry    -- Elaine's time in minutes
  let george := elaine / 3    -- George's time in minutes
  let susan := 150 / 60      -- Susan's time in minutes
  let puddy := elaine / 2    -- Puddy's time in minutes
  let frank := elaine / 2    -- Frank's time in minutes
  let estelle := 0.1 * 60    -- Estelle's time in minutes
  let total_excluding_newman := jerry + elaine + george + susan + puddy + frank + estelle
  let newman := total_excluding_newman / 7   -- Newman's average time
  total_excluding_newman + newman

theorem total_time_in_pool_is_29_minutes : 
  calculate_total_time_in_pool = 29 :=
by
  sorry

end total_time_in_pool_is_29_minutes_l92_92666


namespace three_digit_integers_congruent_to_2_mod_4_l92_92829

theorem three_digit_integers_congruent_to_2_mod_4 : 
    ∃ n, n = 225 ∧ ∀ x, (100 ≤ x ∧ x ≤ 999 ∧ x % 4 = 2) ↔ (∃ m, 25 ≤ m ∧ m ≤ 249 ∧ x = 4 * m + 2) := by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l92_92829


namespace extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l92_92480

theorem extremum_implies_derivative_zero {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_extremum : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :
  deriv f x₀ = 0 :=
sorry

theorem derivative_zero_not_implies_extremum {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_deriv_zero : deriv f x₀ = 0) :
  ¬ (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :=
sorry

end extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l92_92480


namespace probability_within_two_units_of_origin_correct_l92_92453

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let square_area := 36
  let circle_area := 4 * Real.pi
  circle_area / square_area

theorem probability_within_two_units_of_origin_correct :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_two_units_of_origin_correct_l92_92453


namespace total_people_after_four_years_l92_92708

-- Define initial conditions
def initial_total_people : Nat := 9
def board_members : Nat := 3
def regular_members_initial : Nat := initial_total_people - board_members
def years : Nat := 4

-- Define the function for regular members over the years
def regular_members (n : Nat) : Nat :=
  if n = 0 then 
    regular_members_initial
  else 
    2 * regular_members (n - 1)

theorem total_people_after_four_years :
  regular_members years = 96 := 
sorry

end total_people_after_four_years_l92_92708


namespace union_M_N_intersection_complementM_N_l92_92461

open Set  -- Open the Set namespace for convenient notation.

noncomputable def funcDomain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def setN : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def complementFuncDomain : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

theorem union_M_N :
  (funcDomain ∪ setN) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  sorry

theorem intersection_complementM_N :
  (complementFuncDomain ∩ setN) = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end union_M_N_intersection_complementM_N_l92_92461


namespace log_equation_solution_l92_92892

theorem log_equation_solution (x : ℝ) (h : Real.log x + Real.log (x + 4) = Real.log (2 * x + 8)) : x = 2 :=
sorry

end log_equation_solution_l92_92892


namespace fraction_eval_l92_92422

theorem fraction_eval : 1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 89 :=
by
  sorry

end fraction_eval_l92_92422


namespace american_summits_more_water_l92_92348

-- Definitions based on the conditions
def FosterFarmsChickens := 45
def AmericanSummitsWater := 2 * FosterFarmsChickens
def HormelChickens := 3 * FosterFarmsChickens
def BoudinButchersChickens := HormelChickens / 3
def TotalItems := 375
def ItemsByFourCompanies := FosterFarmsChickens + AmericanSummitsWater + HormelChickens + BoudinButchersChickens
def DelMonteWater := TotalItems - ItemsByFourCompanies
def WaterDifference := AmericanSummitsWater - DelMonteWater

theorem american_summits_more_water : WaterDifference = 30 := by
  sorry

end american_summits_more_water_l92_92348


namespace number_of_cakes_l92_92514

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l92_92514


namespace dice_sum_not_22_l92_92339

theorem dice_sum_not_22 (a b c d e : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 6) (h₁ : 1 ≤ b ∧ b ≤ 6)
  (h₂ : 1 ≤ c ∧ c ≤ 6) (h₃ : 1 ≤ d ∧ d ≤ 6) (h₄ : 1 ≤ e ∧ e ≤ 6) 
  (h₅ : a * b * c * d * e = 432) : a + b + c + d + e ≠ 22 :=
sorry

end dice_sum_not_22_l92_92339


namespace stability_of_scores_requires_variance_l92_92022

-- Define the conditions
variable (scores : List ℝ)

-- Define the main theorem
theorem stability_of_scores_requires_variance : True :=
  sorry

end stability_of_scores_requires_variance_l92_92022


namespace amit_work_days_l92_92645

theorem amit_work_days (x : ℕ) (h : 2 * (1 / x : ℚ) + 16 * (1 / 20 : ℚ) = 1) : x = 10 :=
by {
  sorry
}

end amit_work_days_l92_92645


namespace odd_function_a_b_l92_92689

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1/(1-x))) + b

theorem odd_function_a_b (a b : ℝ) :
  (forall x : ℝ, x ≠ 1 → a + 1/(1-x) ≠ 0 → f a b x = -f a b (-x)) ∧
  (forall x : ℝ, x ≠ 1 + 1/a) → a = -1/2 ∧ b = Real.log 2 :=
by sorry

end odd_function_a_b_l92_92689


namespace perpendicular_lines_a_value_l92_92929

theorem perpendicular_lines_a_value :
  ∀ (a : ℝ), (∀ x y : ℝ, 2 * x - y = 0) -> (∀ x y : ℝ, a * x - 2 * y - 1 = 0) ->    
  (∀ m1 m2 : ℝ, m1 = 2 -> m2 = a / 2 -> m1 * m2 = -1) -> a = -1 :=
sorry

end perpendicular_lines_a_value_l92_92929


namespace car_2_speed_proof_l92_92043

noncomputable def car_1_speed : ℝ := 30
noncomputable def car_1_start_time : ℝ := 9
noncomputable def car_2_start_delay : ℝ := 10 / 60
noncomputable def catch_up_time : ℝ := 10.5
noncomputable def car_2_start_time : ℝ := car_1_start_time + car_2_start_delay
noncomputable def travel_duration : ℝ := catch_up_time - car_2_start_time
noncomputable def car_1_head_start_distance : ℝ := car_1_speed * car_2_start_delay
noncomputable def car_1_travel_distance : ℝ := car_1_speed * travel_duration
noncomputable def total_distance : ℝ := car_1_head_start_distance + car_1_travel_distance
noncomputable def car_2_speed : ℝ := total_distance / travel_duration

theorem car_2_speed_proof : car_2_speed = 33.75 := 
by 
  sorry

end car_2_speed_proof_l92_92043


namespace cone_diameter_base_l92_92234

theorem cone_diameter_base 
  (r l : ℝ) 
  (h_semicircle : l = 2 * r) 
  (h_surface_area : π * r ^ 2 + π * r * l = 3 * π) 
  : 2 * r = 2 :=
by
  sorry

end cone_diameter_base_l92_92234


namespace mean_of_five_integers_l92_92998

theorem mean_of_five_integers
  (p q r s t : ℤ)
  (h1 : (p + q + r) / 3 = 9)
  (h2 : (s + t) / 2 = 14) :
  (p + q + r + s + t) / 5 = 11 :=
by
  sorry

end mean_of_five_integers_l92_92998


namespace solve_inequality_l92_92057

theorem solve_inequality (a : ℝ) :
  (a < 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1 - a) ∨
  (a > 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ 1 - a < x ∧ x < a) ∨
  (a = 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ false) :=
sorry

end solve_inequality_l92_92057


namespace radius_for_visibility_l92_92959

theorem radius_for_visibility (r : ℝ) (h₁ : r > 0)
  (h₂ : ∃ o : ℝ, ∀ (s : ℝ), s = 3 → o = 0):
  (∃ p : ℝ, p = 1/3) ∧ (r = 3.6) :=
sorry

end radius_for_visibility_l92_92959


namespace sqrt_of_square_eq_seven_l92_92766

theorem sqrt_of_square_eq_seven (x : ℝ) (h : x^2 = 7) : x = Real.sqrt 7 ∨ x = -Real.sqrt 7 :=
sorry

end sqrt_of_square_eq_seven_l92_92766


namespace proof_problem_l92_92469

variable {a b x y : ℝ}

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem proof_problem : dollar ((x + y) ^ 2) (y ^ 2 + x ^ 2) = 4 * x ^ 2 * y ^ 2 := by
  sorry

end proof_problem_l92_92469


namespace cubes_sum_to_91_l92_92848

theorem cubes_sum_to_91
  (a b : ℤ)
  (h : a^3 + b^3 = 91) : a * b = 12 :=
sorry

end cubes_sum_to_91_l92_92848


namespace shaded_area_l92_92893

-- Defining the conditions
def total_area_of_grid : ℕ := 38
def base_of_triangle : ℕ := 12
def height_of_triangle : ℕ := 4

-- Using the formula for the area of a right triangle
def area_of_unshaded_triangle : ℕ := (base_of_triangle * height_of_triangle) / 2

-- The goal: Prove the area of the shaded region
theorem shaded_area : total_area_of_grid - area_of_unshaded_triangle = 14 :=
by
  sorry

end shaded_area_l92_92893


namespace weight_of_grapes_l92_92894

theorem weight_of_grapes :
  ∀ (weight_of_fruits weight_of_apples weight_of_oranges weight_of_strawberries weight_of_grapes : ℕ),
  weight_of_fruits = 10 →
  weight_of_apples = 3 →
  weight_of_oranges = 1 →
  weight_of_strawberries = 3 →
  weight_of_fruits = weight_of_apples + weight_of_oranges + weight_of_strawberries + weight_of_grapes →
  weight_of_grapes = 3 :=
by
  intros
  sorry

end weight_of_grapes_l92_92894


namespace det_of_matrix_M_l92_92996

open Matrix

def M : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![2, -4, 4], 
    ![0, 6, -2], 
    ![5, -3, 2]]

theorem det_of_matrix_M : Matrix.det M = -68 :=
by
  sorry

end det_of_matrix_M_l92_92996


namespace average_monthly_bill_l92_92613

-- Definitions based on conditions
def first_4_months_average := 30
def last_2_months_average := 24
def first_4_months_total := 4 * first_4_months_average
def last_2_months_total := 2 * last_2_months_average
def total_spent := first_4_months_total + last_2_months_total
def total_months := 6

-- The theorem statement
theorem average_monthly_bill : total_spent / total_months = 28 := by
  sorry

end average_monthly_bill_l92_92613


namespace tim_scored_sum_first_8_even_numbers_l92_92482

-- Define the first 8 even numbers.
def first_8_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16]

-- Define the sum of those numbers.
def sum_first_8_even_numbers : ℕ := List.sum first_8_even_numbers

-- The theorem stating the problem.
theorem tim_scored_sum_first_8_even_numbers : sum_first_8_even_numbers = 72 := by
  sorry

end tim_scored_sum_first_8_even_numbers_l92_92482


namespace find_integer_pairs_l92_92775

theorem find_integer_pairs :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (m n : ℤ), (m, n) ∈ S ↔ mn ≤ 0 ∧ m^3 + n^3 - 37 * m * n = 343) ∧ S.card = 9 :=
sorry

end find_integer_pairs_l92_92775


namespace max_value_6a_3b_10c_l92_92314

theorem max_value_6a_3b_10c (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 25 * c ^ 2 = 1) : 
  6 * a + 3 * b + 10 * c ≤ (Real.sqrt 41) / 2 :=
sorry

end max_value_6a_3b_10c_l92_92314


namespace intersection_of_sets_l92_92614

noncomputable def A : Set ℤ := {x | x^2 - 1 = 0}
def B : Set ℤ := {-1, 2, 5}

theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end intersection_of_sets_l92_92614


namespace proof_PQ_expression_l92_92985

theorem proof_PQ_expression (P Q : ℝ) (h1 : P^2 - P * Q = 1) (h2 : 4 * P * Q - 3 * Q^2 = 2) : 
  P^2 + 3 * P * Q - 3 * Q^2 = 3 :=
by
  sorry

end proof_PQ_expression_l92_92985


namespace gcd_of_expression_l92_92016

noncomputable def gcd_expression (a b c d : ℤ) : ℤ :=
  (a - b) * (b - c) * (c - d) * (d - a) * (b - d) * (a - c)

theorem gcd_of_expression : 
  ∀ (a b c d : ℤ), ∃ (k : ℤ), gcd_expression a b c d = 12 * k :=
sorry

end gcd_of_expression_l92_92016


namespace total_pink_crayons_l92_92582

-- Define the conditions
def Mara_crayons : ℕ := 40
def Mara_pink_percent : ℕ := 10
def Luna_crayons : ℕ := 50
def Luna_pink_percent : ℕ := 20

-- Define the proof problem statement
theorem total_pink_crayons : 
  (Mara_crayons * Mara_pink_percent / 100) + (Luna_crayons * Luna_pink_percent / 100) = 14 := 
by sorry

end total_pink_crayons_l92_92582


namespace infinite_series_value_l92_92031

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 4 * n^2 + 8 * n + 8) / (3^n * (n^3 + 5)) = 1 / 2 :=
by sorry

end infinite_series_value_l92_92031


namespace number_of_hens_l92_92195

theorem number_of_hens (H C G : ℕ) 
  (h1 : H + C + G = 120) 
  (h2 : 2 * H + 4 * C + 4 * G = 348) : 
  H = 66 := 
by 
  sorry

end number_of_hens_l92_92195


namespace remaining_paint_fraction_l92_92274

theorem remaining_paint_fraction (x : ℝ) (h : 1.2 * x = 1 / 2) : (1 / 2) - x = 1 / 12 :=
by 
  sorry

end remaining_paint_fraction_l92_92274


namespace award_medals_at_most_one_canadian_l92_92384

/-- Definition of conditions -/
def sprinter_count := 10 -- Total number of sprinters
def canadian_sprinter_count := 4 -- Number of Canadian sprinters
def medals := ["Gold", "Silver", "Bronze"] -- Types of medals

/-- Definition stating the requirement of the problem -/
def atMostOneCanadianMedal (total_sprinters : Nat) (canadian_sprinters : Nat) 
    (medal_types : List String) : Bool := 
  if total_sprinters = sprinter_count ∧ canadian_sprinters = canadian_sprinter_count ∧ medal_types = medals then
    true
  else
    false

/-- Statement to prove the number of ways to award the medals -/
theorem award_medals_at_most_one_canadian :
  (atMostOneCanadianMedal sprinter_count canadian_sprinter_count medals) →
  ∃ (ways : Nat), ways = 480 :=
by
  sorry

end award_medals_at_most_one_canadian_l92_92384


namespace range_of_a_l92_92111

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end range_of_a_l92_92111


namespace find_a_pure_imaginary_l92_92412

theorem find_a_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = (0 : ℝ) + I) :
  (∃ b : ℝ, a - (17 / (4 - i)) = (0 + b*I)) → a = 4 :=
by
  sorry

end find_a_pure_imaginary_l92_92412


namespace subtraction_of_decimals_l92_92557

theorem subtraction_of_decimals : 58.3 - 0.45 = 57.85 := by
  sorry

end subtraction_of_decimals_l92_92557


namespace avg_salary_supervisors_l92_92013

-- Definitions based on the conditions of the problem
def total_workers : Nat := 48
def supervisors : Nat := 6
def laborers : Nat := 42
def avg_salary_total : Real := 1250
def avg_salary_laborers : Real := 950

-- Given the above conditions, we need to prove the average salary of the supervisors.
theorem avg_salary_supervisors :
  (supervisors * (supervisors * total_workers * avg_salary_total - laborers * avg_salary_laborers) / supervisors) = 3350 :=
by
  sorry

end avg_salary_supervisors_l92_92013


namespace sqrt_identity_l92_92063

def condition1 (α : ℝ) : Prop := 
  ∃ P : ℝ × ℝ, P = (Real.sin 2, Real.cos 2) ∧ Real.sin α = Real.cos 2

def condition2 (P : ℝ × ℝ) : Prop := 
  P.1 ^ 2 + P.2 ^ 2 = 1

theorem sqrt_identity (α : ℝ) (P : ℝ × ℝ) 
  (h₁ : condition1 α) (h₂ : condition2 P) : 
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by 
  sorry

end sqrt_identity_l92_92063


namespace train_combined_distance_l92_92360

/-- Prove that the combined distance covered by three trains is 3480 km,
    given their respective speeds and travel times. -/
theorem train_combined_distance : 
  let speed_A := 150 -- Speed of Train A in km/h
  let time_A := 8     -- Time Train A travels in hours
  let speed_B := 180 -- Speed of Train B in km/h
  let time_B := 6     -- Time Train B travels in hours
  let speed_C := 120 -- Speed of Train C in km/h
  let time_C := 10    -- Time Train C travels in hours
  let distance_A := speed_A * time_A -- Distance covered by Train A
  let distance_B := speed_B * time_B -- Distance covered by Train B
  let distance_C := speed_C * time_C -- Distance covered by Train C
  let combined_distance := distance_A + distance_B + distance_C -- Combined distance covered by all trains
  combined_distance = 3480 :=
by
  sorry

end train_combined_distance_l92_92360


namespace quadratic_inequality_solution_l92_92564

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 3 * x - 18 < 0} = {x : ℝ | -3 < x ∧ x < 6} :=
by
  sorry

end quadratic_inequality_solution_l92_92564


namespace triangle_area_l92_92651

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end triangle_area_l92_92651


namespace place_mat_length_l92_92879

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l92_92879


namespace range_of_m_l92_92447

variable (x y m : ℝ)

def system_of_eq1 := 2 * x + y = -4 * m + 5
def system_of_eq2 := x + 2 * y = m + 4
def inequality1 := x - y > -6
def inequality2 := x + y < 8

theorem range_of_m:
  system_of_eq1 x y m → 
  system_of_eq2 x y m → 
  inequality1 x y → 
  inequality2 x y → 
  -5 < m ∧ m < 7/5 :=
by 
  intros h1 h2 h3 h4
  sorry

end range_of_m_l92_92447


namespace lower_rent_amount_l92_92659

-- Define the conditions and proof goal
variable (T R : ℕ)
variable (L : ℕ)

-- Condition 1: Total rent is $1000
def total_rent (T R : ℕ) (L : ℕ) := 60 * R + L * (T - R)

-- Condition 2: Reduction by 20% when 10 rooms are swapped
def reduced_rent (T R : ℕ) (L : ℕ) := 60 * (R - 10) + L * (T - R + 10)

-- Proof that the lower rent amount is $40 given the conditions
theorem lower_rent_amount (h1 : total_rent T R L = 1000)
                         (h2 : reduced_rent T R L = 800) : L = 40 :=
by
  sorry

end lower_rent_amount_l92_92659


namespace common_factor_is_n_plus_1_l92_92638

def polynomial1 (n : ℕ) : ℕ := n^2 - 1
def polynomial2 (n : ℕ) : ℕ := n^2 + n

theorem common_factor_is_n_plus_1 (n : ℕ) : 
  ∃ (d : ℕ), d ∣ polynomial1 n ∧ d ∣ polynomial2 n ∧ d = n + 1 := by
  sorry

end common_factor_is_n_plus_1_l92_92638


namespace erasers_given_l92_92030

theorem erasers_given (initial final : ℕ) (h1 : initial = 8) (h2 : final = 11) : (final - initial = 3) :=
by
  sorry

end erasers_given_l92_92030


namespace abs_ac_bd_leq_one_l92_92782

theorem abs_ac_bd_leq_one {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : |a * c + b * d| ≤ 1 :=
by
  sorry

end abs_ac_bd_leq_one_l92_92782


namespace eccentricity_of_ellipse_l92_92116

open Real

theorem eccentricity_of_ellipse 
  (O B F : ℝ × ℝ)
  (a b : ℝ) 
  (h_a_gt_b: a > b)
  (h_b_gt_0: b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_OB_eq_OF : dist O B = dist O F)
  (O_is_origin : O = (0,0))
  (B_is_upper_vertex : B = (0, b))
  (F_is_right_focus : F = (c, 0) ∧ c = Real.sqrt (a^2 - b^2)) :
 (c / a = sqrt 2 / 2)
:=
sorry

end eccentricity_of_ellipse_l92_92116


namespace remainder_when_divided_by_100_l92_92606

-- Define the given m
def m : ℕ := 76^2006 - 76

-- State the theorem
theorem remainder_when_divided_by_100 : m % 100 = 0 :=
by
  sorry

end remainder_when_divided_by_100_l92_92606


namespace min_value_of_a_l92_92350

theorem min_value_of_a :
  ∀ (x y : ℝ), |x| + |y| ≤ 1 → (|2 * x - 3 * y + 3 / 2| + |y - 1| + |2 * y - x - 3| ≤ 23 / 2) :=
by
  intros x y h
  sorry

end min_value_of_a_l92_92350


namespace b_power_a_equals_nine_l92_92561

theorem b_power_a_equals_nine (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : b^a = 9 := by
  sorry

end b_power_a_equals_nine_l92_92561


namespace total_simple_interest_is_correct_l92_92584

noncomputable def principal : ℝ := 15041.875
noncomputable def rate : ℝ := 8
noncomputable def time : ℝ := 5
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

theorem total_simple_interest_is_correct :
  simple_interest principal rate time = 6016.75 := 
sorry

end total_simple_interest_is_correct_l92_92584


namespace min_value_x_plus_y_l92_92951

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y_l92_92951


namespace remainder_8347_div_9_l92_92520
-- Import all necessary Mathlib modules

-- Define the problem and conditions
theorem remainder_8347_div_9 : (8347 % 9) = 4 :=
by
  -- To ensure the code builds successfully and contains a placeholder for the proof
  sorry

end remainder_8347_div_9_l92_92520


namespace maximum_value_of_expression_l92_92833

noncomputable def maxValue (x y z : ℝ) : ℝ :=
(x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2)

theorem maximum_value_of_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  maxValue x y z ≤ 243 / 16 :=
sorry

end maximum_value_of_expression_l92_92833


namespace pyramid_volume_l92_92397

def area_SAB : ℝ := 9
def area_SBC : ℝ := 9
def area_SCD : ℝ := 27
def area_SDA : ℝ := 27
def area_ABCD : ℝ := 36
def dihedral_angle_equal := ∀ (α β γ δ: ℝ), α = β ∧ β = γ ∧ γ = δ

theorem pyramid_volume (h_eq_dihedral : dihedral_angle_equal)
  (area_conditions : area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27)
  (area_quadrilateral : area_ABCD = 36) :
  (1 / 3 * area_ABCD * 4.5) = 54 :=
sorry

end pyramid_volume_l92_92397


namespace find_missing_edge_l92_92587

-- Define the known parameters
def volume : ℕ := 80
def edge1 : ℕ := 2
def edge3 : ℕ := 8

-- Define the missing edge
def missing_edge : ℕ := 5

-- State the problem
theorem find_missing_edge (volume : ℕ) (edge1 : ℕ) (edge3 : ℕ) (missing_edge : ℕ) :
  volume = edge1 * missing_edge * edge3 →
  missing_edge = 5 :=
by
  sorry

end find_missing_edge_l92_92587


namespace M_identically_zero_l92_92107

noncomputable def M (x y : ℝ) : ℝ := sorry

theorem M_identically_zero (a : ℝ) (h1 : a > 1) (h2 : ∀ x, M x (a^x) = 0) : ∀ x y, M x y = 0 :=
sorry

end M_identically_zero_l92_92107


namespace find_a_prove_f_pos_l92_92682

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.log x + (1 / 2) * x

theorem find_a (a x0 : ℝ) (hx0 : x0 > 0) (h_tangent : (x0 - a) * Real.log x0 + (1 / 2) * x0 = (1 / 2) * x0 ∧ Real.log x0 - a / x0 + 3 / 2 = 1 / 2) :
  a = 1 :=
sorry

theorem prove_f_pos (a : ℝ) (h_range : 1 / (2 * Real.exp 1) < a ∧ a < 2 * Real.sqrt (Real.exp 1)) (x : ℝ) (hx : x > 0) :
  f x a > 0 :=
sorry

end find_a_prove_f_pos_l92_92682


namespace mail_difference_eq_15_l92_92367

variable (Monday Tuesday Wednesday Thursday : ℕ)
variable (total : ℕ)

theorem mail_difference_eq_15
  (h1 : Monday = 65)
  (h2 : Tuesday = Monday + 10)
  (h3 : Wednesday = Tuesday - 5)
  (h4 : total = 295)
  (h5 : total = Monday + Tuesday + Wednesday + Thursday) :
  Thursday - Wednesday = 15 := 
  by
  sorry

end mail_difference_eq_15_l92_92367


namespace work_completion_days_l92_92740

-- Define the work rates
def john_work_rate : ℚ := 1/8
def rose_work_rate : ℚ := 1/16
def dave_work_rate : ℚ := 1/12

-- Define the combined work rate
def combined_work_rate : ℚ := john_work_rate + rose_work_rate + dave_work_rate

-- Define the required number of days to complete the work together
def days_to_complete_work : ℚ := 1 / combined_work_rate

-- Prove that the total number of days required to complete the work is 48/13
theorem work_completion_days : days_to_complete_work = 48 / 13 :=
by 
  -- Here is where the actual proof would be, but it is not needed as per instructions
  sorry

end work_completion_days_l92_92740


namespace find_s_at_3_l92_92948

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := y^2 - (y + 12)

theorem find_s_at_3 : s 3 = -6 :=
by
  sorry

end find_s_at_3_l92_92948


namespace butternut_wood_figurines_l92_92710

theorem butternut_wood_figurines (B : ℕ) (basswood_blocks : ℕ) (aspen_blocks : ℕ) (butternut_blocks : ℕ) 
  (basswood_figurines_per_block : ℕ) (aspen_figurines_per_block : ℕ) (total_figurines : ℕ) 
  (h_basswood_blocks : basswood_blocks = 15)
  (h_aspen_blocks : aspen_blocks = 20)
  (h_butternut_blocks : butternut_blocks = 20)
  (h_basswood_figurines_per_block : basswood_figurines_per_block = 3)
  (h_aspen_figurines_per_block : aspen_figurines_per_block = 2 * basswood_figurines_per_block)
  (h_total_figurines : total_figurines = 245) :
  B = 4 :=
by
  -- Definitions based on the given conditions
  let basswood_figurines := basswood_blocks * basswood_figurines_per_block
  let aspen_figurines := aspen_blocks * aspen_figurines_per_block
  let figurines_from_butternut := total_figurines - basswood_figurines - aspen_figurines
  -- Calculate the number of figurines per block of butternut wood
  let butternut_figurines_per_block := figurines_from_butternut / butternut_blocks
  -- The objective is to prove that the number of figurines per block of butternut wood is 4
  exact sorry

end butternut_wood_figurines_l92_92710


namespace circle_equation_l92_92791

-- Define the circle's equation as a predicate
def is_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Given conditions, defining the known center and passing point
def center_x : ℝ := 2
def center_y : ℝ := -3
def point_M_x : ℝ := -1
def point_M_y : ℝ := 1

-- Prove that the circle with the given conditions has the correct equation
theorem circle_equation :
  is_circle x y center_x center_y 5 ↔ 
  ∀ x y : ℝ, (x - center_x)^2 + (y + center_y)^2 = 25 := sorry

end circle_equation_l92_92791


namespace total_cost_correct_l92_92108

def cost_first_day : Nat := 4 + 5 + 3 + 2
def cost_second_day : Nat := 5 + 6 + 4
def total_cost : Nat := cost_first_day + cost_second_day

theorem total_cost_correct : total_cost = 29 := by
  sorry

end total_cost_correct_l92_92108


namespace odd_function_period_2pi_l92_92377

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem odd_function_period_2pi (x : ℝ) : 
  f (-x) = -f (x) ∧ 
  ∃ p > 0, p = 2 * Real.pi ∧ ∀ x, f (x + p) = f (x) := 
by
  sorry

end odd_function_period_2pi_l92_92377


namespace sarah_class_choices_l92_92135

-- Conditions 
def total_classes : ℕ := 10
def choose_classes : ℕ := 4
def specific_classes : ℕ := 2

-- Statement
theorem sarah_class_choices : 
  ∃ (n : ℕ), n = Nat.choose (total_classes - specific_classes) 3 ∧ n = 56 :=
by 
  sorry

end sarah_class_choices_l92_92135


namespace convert_degrees_to_radians_l92_92882

theorem convert_degrees_to_radians (θ : ℝ) (h : θ = -630) : θ * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end convert_degrees_to_radians_l92_92882


namespace length_of_platform_l92_92942

theorem length_of_platform (length_train speed_train time_crossing speed_train_mps distance_train_cross : ℝ)
  (h1 : length_train = 120)
  (h2 : speed_train = 60)
  (h3 : time_crossing = 20)
  (h4 : speed_train_mps = 16.67)
  (h5 : distance_train_cross = speed_train_mps * time_crossing):
  (distance_train_cross = length_train + 213.4) :=
by
  sorry

end length_of_platform_l92_92942


namespace q_can_be_true_or_false_l92_92408

-- Define the propositions p and q
variables (p q : Prop)

-- The assumptions given in the problem
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ p

-- The statement we want to prove
theorem q_can_be_true_or_false : ∀ q, q ∨ ¬ q :=
by
  intro q
  exact em q -- Use the principle of excluded middle

end q_can_be_true_or_false_l92_92408


namespace count_multiples_of_5_l92_92463

theorem count_multiples_of_5 (a b : ℕ) (h₁ : 50 ≤ a) (h₂ : a ≤ 300) (h₃ : 50 ≤ b) (h₄ : b ≤ 300) (h₅ : a % 5 = 0) (h₆ : b % 5 = 0) 
  (h₇ : ∀ n : ℕ, 50 ≤ n ∧ n ≤ 300 → n % 5 = 0 → a ≤ n ∧ n ≤ b) :
  b = a + 48 * 5 → (b - a) / 5 + 1 = 49 :=
by
  sorry

end count_multiples_of_5_l92_92463


namespace Meena_cookies_left_l92_92842

def cookies_initial := 5 * 12
def cookies_sold_to_teacher := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

def cookies_left := cookies_initial - cookies_sold_to_teacher - cookies_bought_by_brock - cookies_bought_by_katy

theorem Meena_cookies_left : cookies_left = 15 := 
by 
  -- steps to be proven here
  sorry

end Meena_cookies_left_l92_92842


namespace add_to_1_eq_62_l92_92240

theorem add_to_1_eq_62 :
  let y := 5 * 12 / (180 / 3)
  ∃ x, y + x = 62 ∧ x = 61 :=
by
  sorry

end add_to_1_eq_62_l92_92240


namespace triangle_area_30_l92_92794

theorem triangle_area_30 (h : ∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ a = 5 ∧ c = 13 ∧ b > 0) : 
  ∃ area : ℝ, area = 1 / 2 * 5 * (b : ℝ) ∧ area = 30 :=
by
  sorry

end triangle_area_30_l92_92794


namespace ratio_d_s_proof_l92_92072

noncomputable def ratio_d_s (n : ℕ) (s d : ℝ) : ℝ :=
  d / s

theorem ratio_d_s_proof : ∀ (n : ℕ) (s d : ℝ), 
  (n = 30) → 
  ((n ^ 2 * s ^ 2) / (n * s + 2 * n * d) ^ 2 = 0.81) → 
  ratio_d_s n s d = 1 / 18 :=
by
  intros n s d h_n h_area
  sorry

end ratio_d_s_proof_l92_92072


namespace remainder_sum_1_to_12_div_9_l92_92714

-- Define the sum of the first n natural numbers
def sum_natural (n : Nat) : Nat := n * (n + 1) / 2

-- Define the sum of the numbers from 1 to 12
def sum_1_to_12 := sum_natural 12

-- Define the remainder function
def remainder (a b : Nat) : Nat := a % b

-- Prove that the remainder when the sum of the numbers from 1 to 12 is divided by 9 is 6
theorem remainder_sum_1_to_12_div_9 : remainder sum_1_to_12 9 = 6 := by
  sorry

end remainder_sum_1_to_12_div_9_l92_92714


namespace freshmen_and_sophomores_without_pet_l92_92366

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end freshmen_and_sophomores_without_pet_l92_92366


namespace symmetric_about_one_symmetric_about_two_l92_92632

-- Part 1
theorem symmetric_about_one (rational_num_x : ℚ) (rational_num_r : ℚ) 
(h1 : 3 - 1 = 1 - rational_num_x) (hr1 : r = 3 - 1): 
  rational_num_x = -1 ∧ rational_num_r = 2 := 
by
  sorry

-- Part 2
theorem symmetric_about_two (a b : ℚ) (symmetric_radius : ℚ) 
(h2 : (a + b) / 2 = 2) (condition : |a| = 2 * |b|) : 
  symmetric_radius = 2 / 3 ∨ symmetric_radius = 6 := 
by
  sorry

end symmetric_about_one_symmetric_about_two_l92_92632


namespace salary_increase_after_five_years_l92_92141

theorem salary_increase_after_five_years (S : ℝ) : 
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  percent_increase = 76.23 :=
by
  let final_salary := S * (1.12)^5
  let increase := final_salary - S
  let percent_increase := (increase / S) * 100
  sorry

end salary_increase_after_five_years_l92_92141


namespace sqrt_approximation_l92_92528

theorem sqrt_approximation :
  (2^2 < 5) ∧ (5 < 3^2) ∧ 
  (2.2^2 < 5) ∧ (5 < 2.3^2) ∧ 
  (2.23^2 < 5) ∧ (5 < 2.24^2) ∧ 
  (2.236^2 < 5) ∧ (5 < 2.237^2) →
  (Float.ceil (Float.sqrt 5 * 100) / 100) = 2.24 := 
by
  intro h
  sorry

end sqrt_approximation_l92_92528


namespace polynomial_value_at_two_l92_92221

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_two : f 2 = 243 := by
  -- Proof steps go here
  sorry

end polynomial_value_at_two_l92_92221


namespace large_planter_holds_seeds_l92_92618

theorem large_planter_holds_seeds (total_seeds : ℕ) (small_planter_capacity : ℕ) (num_small_planters : ℕ) (num_large_planters : ℕ) 
  (h1 : total_seeds = 200)
  (h2 : small_planter_capacity = 4)
  (h3 : num_small_planters = 30)
  (h4 : num_large_planters = 4) : 
  (total_seeds - num_small_planters * small_planter_capacity) / num_large_planters = 20 := by
  sorry

end large_planter_holds_seeds_l92_92618


namespace no_hot_dogs_l92_92916

def hamburgers_initial := 9.0
def hamburgers_additional := 3.0
def hamburgers_total := 12.0

theorem no_hot_dogs (h1 : hamburgers_initial + hamburgers_additional = hamburgers_total) : 0 = 0 :=
by
  sorry

end no_hot_dogs_l92_92916


namespace sales_professionals_count_l92_92599

theorem sales_professionals_count :
  (∀ (C : ℕ) (MC : ℕ) (M : ℕ), C = 500 → MC = 10 → M = 5 → C / M / MC = 10) :=
by
  intros C MC M hC hMC hM
  sorry

end sales_professionals_count_l92_92599


namespace sum_of_digits_2_1989_and_5_1989_l92_92824

theorem sum_of_digits_2_1989_and_5_1989 
  (m n : ℕ) 
  (h1 : 10^(m-1) < 2^1989 ∧ 2^1989 < 10^m) 
  (h2 : 10^(n-1) < 5^1989 ∧ 5^1989 < 10^n) 
  (h3 : 2^1989 * 5^1989 = 10^1989) : 
  m + n = 1990 := 
sorry

end sum_of_digits_2_1989_and_5_1989_l92_92824


namespace donation_total_correct_l92_92305

noncomputable def total_donation (t : ℝ) (y : ℝ) (x : ℝ) : ℝ :=
  t + t + x
  
theorem donation_total_correct (t : ℝ) (y : ℝ) (x : ℝ)
  (h1 : t = 570.00) (h2 : y = 140.00) (h3 : t = x + y) : total_donation t y x = 1570.00 :=
by
  sorry

end donation_total_correct_l92_92305


namespace percentage_third_year_students_l92_92918

-- Define the conditions as given in the problem
variables (T : ℝ) (T_3 : ℝ) (S_2 : ℝ)

-- Conditions
def cond1 : Prop := S_2 = 0.10 * T
def cond2 : Prop := (0.10 * T) / (T - T_3) = 1 / 7

-- Define the proof goal
theorem percentage_third_year_students (h1 : cond1 T S_2) (h2 : cond2 T T_3) : T_3 = 0.30 * T :=
sorry

end percentage_third_year_students_l92_92918


namespace last_digit_of_2_pow_2018_l92_92750

-- Definition of the cyclic pattern
def last_digit_cycle : List ℕ := [2, 4, 8, 6]

-- Function to find the last digit of 2^n using the cycle
def last_digit_of_power_of_two (n : ℕ) : ℕ :=
  last_digit_cycle.get! ((n % 4) - 1)

-- Main theorem statement
theorem last_digit_of_2_pow_2018 : last_digit_of_power_of_two 2018 = 4 :=
by
  -- The proof part is omitted
  sorry

end last_digit_of_2_pow_2018_l92_92750


namespace coordinates_in_second_quadrant_l92_92042

section 
variable (x y : ℝ)
variable (hx : x = -7)
variable (hy : y = 4)
variable (quadrant : x < 0 ∧ y > 0)
variable (distance_x : |y| = 4)
variable (distance_y : |x| = 7)

theorem coordinates_in_second_quadrant :
  (x, y) = (-7, 4) := by
  sorry
end

end coordinates_in_second_quadrant_l92_92042


namespace sequence_10th_term_l92_92216

theorem sequence_10th_term (a : ℕ → ℝ) 
  (h_initial : a 1 = 1) 
  (h_recursive : ∀ n, a (n + 1) = 2 * a n / (a n + 2)) : 
  a 10 = 2 / 11 :=
sorry

end sequence_10th_term_l92_92216


namespace max_profit_l92_92169

variables (x y : ℕ)

def steel_constraint := 10 * x + 70 * y ≤ 700
def non_ferrous_constraint := 23 * x + 40 * y ≤ 642
def non_negativity := x ≥ 0 ∧ y ≥ 0
def profit := 80 * x + 100 * y

theorem max_profit (h₁ : steel_constraint x y)
                   (h₂ : non_ferrous_constraint x y)
                   (h₃ : non_negativity x y):
  profit x y = 2180 := 
sorry

end max_profit_l92_92169


namespace quadrangular_prism_volume_l92_92522

theorem quadrangular_prism_volume
  (perimeter : ℝ)
  (side_length : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (H1 : perimeter = 32)
  (H2 : side_length = perimeter / 4)
  (H3 : height = side_length)
  (H4 : volume = side_length * side_length * height) :
  volume = 512 := by
    sorry

end quadrangular_prism_volume_l92_92522


namespace arithmetic_sequence_sum_l92_92238

noncomputable def sum_first_ten_terms (a d : ℕ) : ℕ :=
  (10 / 2) * (2 * a + (10 - 1) * d)

theorem arithmetic_sequence_sum 
  (a d : ℕ) 
  (h1 : a + 2 * d = 8) 
  (h2 : a + 5 * d = 14) :
  sum_first_ten_terms a d = 130 :=
by
  sorry

end arithmetic_sequence_sum_l92_92238


namespace function_decreases_iff_l92_92817

theorem function_decreases_iff (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (m - 3) * x1 + 4 > (m - 3) * x2 + 4) ↔ m < 3 :=
by
  sorry

end function_decreases_iff_l92_92817


namespace common_ratio_geometric_sequence_l92_92154

theorem common_ratio_geometric_sequence (n : ℕ) :
  ∃ q : ℕ, (∀ k : ℕ, q = 4^(2*k+3) / 4^(2*k+1)) ∧ q = 16 :=
by
  use 16
  sorry

end common_ratio_geometric_sequence_l92_92154


namespace sum_of_ages_l92_92919

variable (M E : ℝ)
variable (h1 : M = E + 9)
variable (h2 : M + 5 = 3 * (E - 3))

theorem sum_of_ages : M + E = 32 :=
by
  sorry

end sum_of_ages_l92_92919


namespace value_of_A_l92_92977

-- Definitions for values in the factor tree, ensuring each condition is respected.
def D : ℕ := 3 * 2 * 2
def E : ℕ := 5 * 2
def B : ℕ := 3 * D
def C : ℕ := 5 * E
def A : ℕ := B * C

-- Assertion of the correct value for A
theorem value_of_A : A = 1800 := by
  -- Mathematical equivalence proof problem placeholder
  sorry

end value_of_A_l92_92977


namespace average_difference_l92_92457

-- Definitions for the conditions
def set1 : List ℕ := [20, 40, 60]
def set2 : List ℕ := [10, 60, 35]

-- Function to compute the average of a list of numbers
def average (lst : List ℕ) : ℚ :=
  lst.sum / lst.length

-- The main theorem to prove the difference between the averages is 5
theorem average_difference : average set1 - average set2 = 5 := by
  sorry

end average_difference_l92_92457


namespace rice_in_each_container_l92_92008

-- Given conditions from the problem
def total_weight_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- A theorem that each container has 25 ounces of rice given the conditions
theorem rice_in_each_container (h : total_weight_pounds * pounds_to_ounces / num_containers = 25) : True :=
  sorry

end rice_in_each_container_l92_92008


namespace product_range_l92_92835

theorem product_range (m b : ℚ) (h₀ : m = 3 / 4) (h₁ : b = 6 / 5) : 0 < m * b ∧ m * b < 1 :=
by
  sorry

end product_range_l92_92835


namespace quadratic_root_product_l92_92955

theorem quadratic_root_product (a b : ℝ) (m p r : ℝ)
  (h1 : a * b = 3)
  (h2 : ∀ x, x^2 - mx + 3 = 0 → x = a ∨ x = b)
  (h3 : ∀ x, x^2 - px + r = 0 → x = a + 2 / b ∨ x = b + 2 / a) :
  r = 25 / 3 := by
  sorry

end quadratic_root_product_l92_92955


namespace symmetrical_point_with_respect_to_x_axis_l92_92446

-- Define the point P with coordinates (-2, -1)
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the given point
def P : Point := { x := -2, y := -1 }

-- Define the symmetry with respect to the x-axis
def symmetry_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

-- Verify the symmetrical point
theorem symmetrical_point_with_respect_to_x_axis :
  symmetry_x_axis P = { x := -2, y := 1 } :=
by
  -- Skip the proof
  sorry

end symmetrical_point_with_respect_to_x_axis_l92_92446


namespace alyssa_ate_limes_l92_92306

def mikes_limes : ℝ := 32.0
def limes_left : ℝ := 7.0

theorem alyssa_ate_limes : mikes_limes - limes_left = 25.0 := by
  sorry

end alyssa_ate_limes_l92_92306


namespace parabola_intersection_value_l92_92688

theorem parabola_intersection_value (a : ℝ) (h : a^2 - a - 1 = 0) : a^2 - a + 2014 = 2015 :=
by
  sorry

end parabola_intersection_value_l92_92688


namespace find_principal_amount_l92_92885

theorem find_principal_amount (P r : ℝ) (h1 : 720 = P * (1 + 2 * r)) (h2 : 1020 = P * (1 + 7 * r)) : P = 600 :=
by sorry

end find_principal_amount_l92_92885


namespace necessary_and_sufficient_condition_for_extreme_value_l92_92131

-- Defining the function f(x) = ax^3 + x + 1
def f (a x : ℝ) : ℝ := a * x^3 + x + 1

-- Defining the condition for f to have an extreme value
def has_extreme_value (a : ℝ) : Prop := ∃ x : ℝ, deriv (f a) x = 0

-- Stating the problem
theorem necessary_and_sufficient_condition_for_extreme_value (a : ℝ) :
  has_extreme_value a ↔ a < 0 := by
  sorry

end necessary_and_sufficient_condition_for_extreme_value_l92_92131


namespace fixed_point_l92_92499

noncomputable def f (a : ℝ) (x : ℝ) := a^(x - 2) - 3

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 2 = -2 :=
by
  sorry

end fixed_point_l92_92499


namespace sample_size_of_survey_l92_92805

theorem sample_size_of_survey (total_students : ℕ) (analyzed_students : ℕ)
  (h1 : total_students = 4000) (h2 : analyzed_students = 500) :
  analyzed_students = 500 :=
by
  sorry

end sample_size_of_survey_l92_92805


namespace inequality_transformation_l92_92837

variable {a b : ℝ}

theorem inequality_transformation (h : a < b) : -a / 3 > -b / 3 :=
  sorry

end inequality_transformation_l92_92837


namespace work_done_in_five_days_l92_92702

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 11
def work_rate_B : ℚ := 1 / 5
def work_rate_C : ℚ := 1 / 55

-- Define the work done in a cycle of 2 days
def work_one_cycle : ℚ := (work_rate_A + work_rate_B) + (work_rate_A + work_rate_C)

-- The total work needed to be done is 1
def total_work : ℚ := 1

-- The number of days in a cycle of 2 days
def days_per_cycle : ℕ := 2

-- Proving that the work will be done in exactly 5 days
theorem work_done_in_five_days :
  ∃ n : ℕ, n = 5 →
  n * (work_rate_A + work_rate_B) + (n-1) * (work_rate_A + work_rate_C) = total_work :=
by
  -- Sorry to skip the detailed proof steps
  sorry

end work_done_in_five_days_l92_92702


namespace smallest_possible_number_of_students_l92_92823

theorem smallest_possible_number_of_students :
  ∃ n : ℕ, (n % 200 = 0) ∧ (∀ m : ℕ, (m < n → 
    75 * m ≤ 100 * n) ∧
    (∃ a b c : ℕ, a = m / 4 ∧ b = a / 10 ∧ 
    ∃ y z : ℕ, y = 3 * z ∧ (y + z - b = a) ∧ y * c = n / 4)) :=
by
  sorry

end smallest_possible_number_of_students_l92_92823


namespace sin_sum_triangle_inequality_l92_92695

theorem sin_sum_triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_triangle_inequality_l92_92695


namespace cube_fraction_inequality_l92_92277

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l92_92277


namespace range_of_m_l92_92768

variable (p q : Prop)
variable (m : ℝ)
variable (hp : (∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (1 - m) = 1) → (0 < m ∧ m < 1/3)))
variable (hq : (m^2 - 15 * m < 0))

theorem range_of_m (h_not_p_and_q : ¬ (p ∧ q)) (h_p_or_q : p ∨ q) :
  (1/3 ≤ m ∧ m < 15) :=
sorry

end range_of_m_l92_92768


namespace ratio_of_spots_to_wrinkles_l92_92915

-- Definitions
def E : ℕ := 3
def W : ℕ := 3 * E
def S : ℕ := E + W - 69

-- Theorem
theorem ratio_of_spots_to_wrinkles : S / W = 7 :=
by
  sorry

end ratio_of_spots_to_wrinkles_l92_92915


namespace find_angle_x_l92_92535

-- Define the angles and parallel lines conditions
def parallel_lines (k l : Prop) (angle1 : Real) (angle2 : Real) : Prop :=
  k ∧ l ∧ angle1 = 30 ∧ angle2 = 90

-- Statement of the problem in Lean syntax
theorem find_angle_x (k l : Prop) (angle1 angle2 : Real) (x : Real) : 
  parallel_lines k l angle1 angle2 → x = 150 :=
by
  -- Assuming conditions are given, prove x = 150
  sorry

end find_angle_x_l92_92535


namespace trains_cross_time_l92_92383

noncomputable def time_to_cross_trains : ℝ :=
  200 / (89.992800575953935 * (1000 / 3600))

theorem trains_cross_time :
  abs (time_to_cross_trains - 8) < 1e-7 :=
by
  sorry

end trains_cross_time_l92_92383


namespace charge_y1_charge_y2_cost_effective_range_call_duration_difference_l92_92254

def y1 (x : ℕ) : ℝ :=
  if x ≤ 600 then 30 else 0.1 * x - 30

def y2 (x : ℕ) : ℝ :=
  if x ≤ 1200 then 50 else 0.1 * x - 70

theorem charge_y1 (x : ℕ) :
  (x ≤ 600 → y1 x = 30) ∧ (x > 600 → y1 x = 0.1 * x - 30) :=
by sorry

theorem charge_y2 (x : ℕ) :
  (x ≤ 1200 → y2 x = 50) ∧ (x > 1200 → y2 x = 0.1 * x - 70) :=
by sorry

theorem cost_effective_range (x : ℕ) :
  (0 ≤ x) ∧ (x < 800) → y1 x < y2 x :=
by sorry

noncomputable def call_time_xiaoming : ℕ := 1300
noncomputable def call_time_xiaohua : ℕ := 900

theorem call_duration_difference :
  call_time_xiaoming = call_time_xiaohua + 400 :=
by sorry

end charge_y1_charge_y2_cost_effective_range_call_duration_difference_l92_92254


namespace myrtle_eggs_after_collection_l92_92301

def henA_eggs_per_day : ℕ := 3
def henB_eggs_per_day : ℕ := 4
def henC_eggs_per_day : ℕ := 2
def henD_eggs_per_day : ℕ := 5
def henE_eggs_per_day : ℕ := 3

def days_gone : ℕ := 12
def eggs_taken_by_neighbor : ℕ := 32

def eggs_dropped_day1 : ℕ := 3
def eggs_dropped_day2 : ℕ := 5
def eggs_dropped_day3 : ℕ := 2

theorem myrtle_eggs_after_collection :
  let total_eggs :=
    (henA_eggs_per_day * days_gone) +
    (henB_eggs_per_day * days_gone) +
    (henC_eggs_per_day * days_gone) +
    (henD_eggs_per_day * days_gone) +
    (henE_eggs_per_day * days_gone)
  let remaining_eggs_after_neighbor := total_eggs - eggs_taken_by_neighbor
  let total_dropped_eggs := eggs_dropped_day1 + eggs_dropped_day2 + eggs_dropped_day3
  let eggs_after_drops := remaining_eggs_after_neighbor - total_dropped_eggs
  eggs_after_drops = 162 := 
by 
  sorry

end myrtle_eggs_after_collection_l92_92301


namespace m_intersects_at_least_one_of_a_or_b_l92_92844

-- Definitions based on given conditions
variables {Plane : Type} {Line : Type} (α β : Plane) (a b m : Line)

-- Assume necessary conditions
axiom skew_lines (a b : Line) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom plane_intersection_is_line (p1 p2 : Plane) : Line
axiom intersects (l1 l2 : Line) : Prop

-- Given conditions
variables
  (h1 : skew_lines a b)               -- a and b are skew lines
  (h2 : line_in_plane a α)            -- a is contained in plane α
  (h3 : line_in_plane b β)            -- b is contained in plane β
  (h4 : plane_intersection_is_line α β = m)  -- α ∩ β = m

-- The theorem to prove the correct answer
theorem m_intersects_at_least_one_of_a_or_b :
  intersects m a ∨ intersects m b :=
sorry -- proof to be provided

end m_intersects_at_least_one_of_a_or_b_l92_92844


namespace z_in_second_quadrant_l92_92956

open Complex

-- Given the condition
def satisfies_eqn (z : ℂ) : Prop := z * (1 - I) = 4 * I

-- Define the second quadrant condition
def in_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : satisfies_eqn z) : in_second_quadrant z :=
  sorry

end z_in_second_quadrant_l92_92956


namespace unattainable_y_l92_92048

theorem unattainable_y (x : ℝ) (h : x ≠ -5/4) : ¬∃ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3 / 4 :=
by
  sorry

end unattainable_y_l92_92048


namespace solve_trig_problem_l92_92128

open Real

theorem solve_trig_problem (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = 1) :
  x = 0 ∨ x = π / 2 := sorry

end solve_trig_problem_l92_92128


namespace motorcyclist_speed_before_delay_l92_92222

/-- Given conditions and question:
1. The motorcyclist was delayed by 0.4 hours.
2. After the delay, the motorcyclist increased his speed by 10 km/h.
3. The motorcyclist made up for the lost time over a stretch of 80 km.
-/
theorem motorcyclist_speed_before_delay :
  ∃ x : ℝ, (80 / x - 0.4 = 80 / (x + 10)) ∧ x = 40 :=
sorry

end motorcyclist_speed_before_delay_l92_92222


namespace perpendicular_vectors_x_value_l92_92718

theorem perpendicular_vectors_x_value 
  (x : ℝ) (a b : ℝ × ℝ) (hₐ : a = (1, -2)) (hᵦ : b = (3, x)) (h_perpendicular : a.1 * b.1 + a.2 * b.2 = 0) : 
  x = 3 / 2 :=
by
  -- The proof is not required, hence we use 'sorry'
  sorry

end perpendicular_vectors_x_value_l92_92718


namespace negation_of_P_is_non_P_l92_92199

open Real

/-- Proposition P: For any x in the real numbers, sin(x) <= 1 -/
def P : Prop := ∀ x : ℝ, sin x ≤ 1

/-- Negation of P: There exists x in the real numbers such that sin(x) >= 1 -/
def non_P : Prop := ∃ x : ℝ, sin x ≥ 1

theorem negation_of_P_is_non_P : ¬P ↔ non_P :=
by 
  sorry

end negation_of_P_is_non_P_l92_92199


namespace ab_value_l92_92707

theorem ab_value (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 := by
  sorry

end ab_value_l92_92707


namespace TotalMarks_l92_92055

def AmayaMarks (Arts Maths Music SocialStudies : ℕ) : Prop :=
  Maths = Arts - 20 ∧
  Maths = (9 * Arts) / 10 ∧
  Music = 70 ∧
  Music + 10 = SocialStudies

theorem TotalMarks (Arts Maths Music SocialStudies : ℕ) : 
  AmayaMarks Arts Maths Music SocialStudies → 
  (Arts + Maths + Music + SocialStudies = 530) :=
by
  sorry

end TotalMarks_l92_92055


namespace infinitely_many_divisors_l92_92755

theorem infinitely_many_divisors (a : ℕ) : ∃ᶠ n in at_top, n ∣ a ^ (n - a + 1) - 1 :=
sorry

end infinitely_many_divisors_l92_92755


namespace roots_ratio_sum_eq_six_l92_92732

theorem roots_ratio_sum_eq_six (x1 x2 : ℝ) (h1 : 2 * x1^2 - 4 * x1 + 1 = 0) (h2 : 2 * x2^2 - 4 * x2 + 1 = 0) :
  (x1 / x2) + (x2 / x1) = 6 :=
sorry

end roots_ratio_sum_eq_six_l92_92732


namespace find_cost_price_l92_92644

variable (CP : ℝ)

def SP1 : ℝ := 0.80 * CP
def SP2 : ℝ := 1.06 * CP

axiom cond1 : SP2 - SP1 = 520

theorem find_cost_price : CP = 2000 :=
by
  sorry

end find_cost_price_l92_92644


namespace problem_statement_l92_92464

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end problem_statement_l92_92464


namespace time_per_bone_l92_92973

theorem time_per_bone (total_hours : ℕ) (total_bones : ℕ) (h1 : total_hours = 1030) (h2 : total_bones = 206) :
  (total_hours / total_bones = 5) :=
by {
  sorry
}

end time_per_bone_l92_92973


namespace min_students_l92_92307

theorem min_students (b g : ℕ) (hb : (3 / 5 : ℚ) * b = (5 / 6 : ℚ) * g) :
  b + g = 43 :=
sorry

end min_students_l92_92307


namespace a_minus_b_is_perfect_square_l92_92416
-- Import necessary libraries

-- Define the problem in Lean
theorem a_minus_b_is_perfect_square (a b c : ℕ) (h1: Nat.gcd a (Nat.gcd b c) = 1) 
    (h2: (ab : ℚ) / (a - b) = c) : ∃ k : ℕ, a - b = k * k :=
by
  sorry

end a_minus_b_is_perfect_square_l92_92416


namespace range_of_a_in_fourth_quadrant_l92_92259

-- Define the fourth quadrant condition
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Define the point P(a+1, a-1) and state the theorem
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  in_fourth_quadrant (a + 1) (a - 1) → -1 < a ∧ a < 1 :=
by
  intro h
  have h1 : a + 1 > 0 := h.1
  have h2 : a - 1 < 0 := h.2
  have h3 : a > -1 := by linarith
  have h4 : a < 1 := by linarith
  exact ⟨h3, h4⟩

end range_of_a_in_fourth_quadrant_l92_92259


namespace max_product_is_2331_l92_92532

open Nat

noncomputable def max_product (a b : ℕ) : ℕ :=
  if a + b = 100 ∧ a % 5 = 2 ∧ b % 6 = 3 then a * b else 0

theorem max_product_is_2331 (a b : ℕ) (h_sum : a + b = 100) (h_mod_a : a % 5 = 2) (h_mod_b : b % 6 = 3) :
  max_product a b = 2331 :=
  sorry

end max_product_is_2331_l92_92532


namespace empty_one_container_l92_92635

theorem empty_one_container (a b c : ℕ) :
  ∃ a' b' c', (a' = 0 ∨ b' = 0 ∨ c' = 0) ∧
    (a' = a ∧ b' = b ∧ c' = c ∨
     (a' ≤ a ∧ b' ≤ b ∧ c' ≤ c ∧ (a + b + c = a' + b' + c')) ∧
     (∀ i j, i ≠ j → (i = 1 ∨ i = 2 ∨ i = 3) →
              (j = 1 ∨ j = 2 ∨ j = 3) →
              (if i = 1 then (if j = 2 then a' = a - a ∨ a' = a else (if j = 3 then a' = a - a ∨ a' = a else false))
               else if i = 2 then (if j = 1 then b' = b - b ∨ b' = b else (if j = 3 then b' = b - b ∨ b' = b else false))
               else (if j = 1 then c' = c - c ∨ c' = c else (if j = 2 then c' = c - c ∨ c' = c else false))))) :=
by
  sorry

end empty_one_container_l92_92635


namespace product_of_solutions_abs_eq_l92_92214

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x - 5| + 4 = 7) : x * (if x = 8 then 2 else 8) = 16 :=
by {
  sorry
}

end product_of_solutions_abs_eq_l92_92214


namespace kelly_total_apples_l92_92353

variable (initial_apples : ℕ) (additional_apples : ℕ)

theorem kelly_total_apples (h1 : initial_apples = 56) (h2 : additional_apples = 49) :
  initial_apples + additional_apples = 105 :=
by
  sorry

end kelly_total_apples_l92_92353


namespace range_of_a_l92_92092

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) → ((x - a) ^ 2 < 1)) ↔ (1 ≤ a ∧ a ≤ 2) :=
by 
  sorry

end range_of_a_l92_92092


namespace joey_average_speed_l92_92148

noncomputable def average_speed_of_round_trip (distance_out : ℝ) (time_out : ℝ) (speed_return : ℝ) : ℝ :=
  let distance_return := distance_out
  let total_distance := distance_out + distance_return
  let time_return := distance_return / speed_return
  let total_time := time_out + time_return
  total_distance / total_time

theorem joey_average_speed :
  average_speed_of_round_trip 2 1 6.000000000000002 = 3 := by
  sorry

end joey_average_speed_l92_92148


namespace speed_ratio_l92_92374

-- Define the speeds of A and B
variables (v_A v_B : ℝ)

-- Assume the conditions of the problem
axiom h1 : 200 / v_A = 400 / v_B

-- Prove the ratio of the speeds
theorem speed_ratio : v_A / v_B = 1 / 2 :=
by
  sorry

end speed_ratio_l92_92374


namespace total_weight_correct_l92_92291

-- Conditions for the weights of different types of candies
def frank_chocolate_weight : ℝ := 3
def gwen_chocolate_weight : ℝ := 2
def frank_gummy_bears_weight : ℝ := 2
def gwen_gummy_bears_weight : ℝ := 2.5
def frank_caramels_weight : ℝ := 1
def gwen_caramels_weight : ℝ := 1
def frank_hard_candy_weight : ℝ := 4
def gwen_hard_candy_weight : ℝ := 1.5

-- Combined weights of each type of candy
def chocolate_weight : ℝ := frank_chocolate_weight + gwen_chocolate_weight
def gummy_bears_weight : ℝ := frank_gummy_bears_weight + gwen_gummy_bears_weight
def caramels_weight : ℝ := frank_caramels_weight + gwen_caramels_weight
def hard_candy_weight : ℝ := frank_hard_candy_weight + gwen_hard_candy_weight

-- Total weight of the Halloween candy haul
def total_halloween_weight : ℝ := 
  chocolate_weight +
  gummy_bears_weight +
  caramels_weight +
  hard_candy_weight

-- Theorem to prove the total weight is 17 pounds
theorem total_weight_correct : total_halloween_weight = 17 := by
  sorry

end total_weight_correct_l92_92291


namespace largest_additional_license_plates_l92_92930

theorem largest_additional_license_plates :
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  new_total - original_total = 40 :=
by
  let original_first_set := 5
  let original_second_set := 3
  let original_third_set := 4
  let original_total := original_first_set * original_second_set * original_third_set

  let new_set_case1 := original_first_set * (original_second_set + 2) * original_third_set
  let new_set_case2 := original_first_set * (original_second_set + 1) * (original_third_set + 1)

  let new_total := max new_set_case1 new_set_case2

  sorry

end largest_additional_license_plates_l92_92930


namespace set_proof_l92_92617

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem set_proof :
  (U \ A) ∩ (U \ B) = {4, 8} := by
  sorry

end set_proof_l92_92617


namespace divisible_digit_B_l92_92126

-- Define the digit type as natural numbers within the range 0 to 9.
def digit := {n : ℕ // n <= 9}

-- Define what it means for a number to be even.
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be divisible by 3.
def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define our problem in Lean as properties of the digit B.
theorem divisible_digit_B (B : digit) (h_even : even B.1) (h_div_by_3 : divisible_by_3 (14 + B.1)) : B.1 = 4 :=
sorry

end divisible_digit_B_l92_92126


namespace totalKidsInLawrenceCounty_l92_92094

-- Constants representing the number of kids in each category
def kidsGoToCamp : ℕ := 629424
def kidsStayHome : ℕ := 268627

-- Statement of the total number of kids in Lawrence county
theorem totalKidsInLawrenceCounty : kidsGoToCamp + kidsStayHome = 898051 := by
  sorry

end totalKidsInLawrenceCounty_l92_92094


namespace ratio_a3_a2_l92_92890

theorem ratio_a3_a2 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_3 / a_2 = -2 :=
sorry

end ratio_a3_a2_l92_92890


namespace problem_statement_l92_92118

theorem problem_statement : 1103^2 - 1097^2 - 1101^2 + 1099^2 = 8800 := by
  sorry

end problem_statement_l92_92118


namespace max_subway_riders_l92_92978

theorem max_subway_riders:
  ∃ (P F : ℕ), P + F = 251 ∧ (1 / 11) * P + (1 / 13) * F = 22 := sorry

end max_subway_riders_l92_92978


namespace gcd_lcm_sum_ge_sum_l92_92014

theorem gcd_lcm_sum_ge_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  Nat.gcd a b + Nat.lcm a b ≥ a + b := 
sorry

end gcd_lcm_sum_ge_sum_l92_92014


namespace find_rate_percent_l92_92841

theorem find_rate_percent (SI P T : ℝ) (h : SI = (P * R * T) / 100) (H_SI : SI = 250) 
  (H_P : P = 1500) (H_T : T = 5) : R = 250 / 75 := by
  sorry

end find_rate_percent_l92_92841


namespace pentagon_rectangle_ratio_l92_92032

theorem pentagon_rectangle_ratio :
  let p : ℝ := 60  -- Perimeter of both the pentagon and the rectangle
  let length_side_pentagon : ℝ := 12
  let w : ℝ := 10
  p / 5 = length_side_pentagon ∧ p/6 = w ∧ length_side_pentagon / w = 6/5 :=
sorry

end pentagon_rectangle_ratio_l92_92032


namespace fifth_equation_l92_92295

-- Define the conditions
def condition1 : Prop := 2^1 * 1 = 2
def condition2 : Prop := 2^2 * 1 * 3 = 3 * 4
def condition3 : Prop := 2^3 * 1 * 3 * 5 = 4 * 5 * 6

-- The statement to prove
theorem fifth_equation (h1 : condition1) (h2 : condition2) (h3 : condition3) : 
  2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 :=
sorry

end fifth_equation_l92_92295


namespace equal_integers_l92_92364

theorem equal_integers (a b : ℕ)
  (h : ∀ n : ℕ, n > 0 → a > 0 → b > 0 → (a^n + n) ∣ (b^n + n)) : a = b := 
sorry

end equal_integers_l92_92364


namespace A_can_give_C_start_l92_92133

def canGiveStart (total_distance start_A_B start_B_C start_A_C : ℝ) :=
  (total_distance - start_A_B) / total_distance * (total_distance - start_B_C) / total_distance = 
  (total_distance - start_A_C) / total_distance

theorem A_can_give_C_start :
  canGiveStart 1000 70 139.7849462365591 200 :=
by
  sorry

end A_can_give_C_start_l92_92133


namespace expression_behavior_l92_92723

theorem expression_behavior (x : ℝ) (h1 : -3 < x) (h2 : x < 2) :
  ¬∃ m, ∀ y : ℝ, (h3 : -3 < y) → (h4 : y < 2) → (x ≠ 1) → (y ≠ 1) → 
    (m <= (y^2 - 3*y + 3) / (y - 1)) ∧ 
    (m >= (y^2 - 3*y + 3) / (y - 1)) :=
sorry

end expression_behavior_l92_92723


namespace fraction_girls_at_meet_l92_92640

-- Define the conditions of the problem
def numStudentsMaplewood : ℕ := 300
def ratioBoysGirlsMaplewood : ℕ × ℕ := (3, 2)
def numStudentsRiverview : ℕ := 240
def ratioBoysGirlsRiverview : ℕ × ℕ := (3, 5)

-- Define the combined number of students and number of girls
def totalStudentsMaplewood := numStudentsMaplewood
def totalStudentsRiverview := numStudentsRiverview

def numGirlsMaplewood : ℕ :=
  let (b, g) := ratioBoysGirlsMaplewood
  (totalStudentsMaplewood * g) / (b + g)

def numGirlsRiverview : ℕ :=
  let (b, g) := ratioBoysGirlsRiverview
  (totalStudentsRiverview * g) / (b + g)

def totalGirls := numGirlsMaplewood + numGirlsRiverview
def totalStudents := totalStudentsMaplewood + totalStudentsRiverview

-- Formalize the actual proof statement
theorem fraction_girls_at_meet : 
  (totalGirls : ℚ) / totalStudents = 1 / 2 := by
  sorry

end fraction_girls_at_meet_l92_92640


namespace train_cross_time_l92_92764

noncomputable def train_length : ℝ := 317.5
noncomputable def train_speed_kph : ℝ := 153.3
noncomputable def convert_speed_to_mps (speed_kph : ℝ) : ℝ :=
  (speed_kph * 1000) / 3600

noncomputable def train_speed_mps : ℝ := convert_speed_to_mps train_speed_kph
noncomputable def time_to_cross_pole (length : ℝ) (speed : ℝ) : ℝ :=
  length / speed

theorem train_cross_time :
  time_to_cross_pole train_length train_speed_mps = 7.456 :=
by 
  -- This is where the proof would go
  sorry

end train_cross_time_l92_92764


namespace countDivisorsOf72Pow8_l92_92581

-- Definitions of conditions in Lean 4
def isPerfectSquare (a b : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0
def isPerfectCube (a b : ℕ) : Prop := a % 3 = 0 ∧ b % 3 = 0
def isPerfectSixthPower (a b : ℕ) : Prop := a % 6 = 0 ∧ b % 6 = 0

def countPerfectSquares : ℕ := 13 * 9
def countPerfectCubes : ℕ := 9 * 6
def countPerfectSixthPowers : ℕ := 5 * 3

-- The proof problem to prove the number of such divisors is 156
theorem countDivisorsOf72Pow8:
  (countPerfectSquares + countPerfectCubes - countPerfectSixthPowers) = 156 :=
by
  sorry

end countDivisorsOf72Pow8_l92_92581


namespace pets_beds_calculation_l92_92570

theorem pets_beds_calculation
  (initial_beds : ℕ)
  (additional_beds : ℕ)
  (total_pets : ℕ)
  (H1 : initial_beds = 12)
  (H2 : additional_beds = 8)
  (H3 : total_pets = 10) :
  (initial_beds + additional_beds) / total_pets = 2 := 
by 
  sorry

end pets_beds_calculation_l92_92570


namespace betty_initial_marbles_l92_92298

theorem betty_initial_marbles (B : ℝ) (h1 : 0.40 * B = 24) : B = 60 :=
by
  sorry

end betty_initial_marbles_l92_92298


namespace sqrt_square_eq_14_l92_92041

theorem sqrt_square_eq_14 : Real.sqrt (14 ^ 2) = 14 :=
by
  sorry

end sqrt_square_eq_14_l92_92041


namespace margaret_mean_score_l92_92967

theorem margaret_mean_score : 
  let all_scores_sum := 832
  let cyprian_scores_count := 5
  let margaret_scores_count := 4
  let cyprian_mean_score := 92
  let cyprian_scores_sum := cyprian_scores_count * cyprian_mean_score
  (all_scores_sum - cyprian_scores_sum) / margaret_scores_count = 93 := by
  sorry

end margaret_mean_score_l92_92967


namespace longest_sticks_triangle_shortest_sticks_not_triangle_l92_92619

-- Define the lengths of the six sticks in descending order
variables {a1 a2 a3 a4 a5 a6 : ℝ}

-- Assuming the conditions
axiom h1 : a1 ≥ a2
axiom h2 : a2 ≥ a3
axiom h3 : a3 ≥ a4
axiom h4 : a4 ≥ a5
axiom h5 : a5 ≥ a6
axiom h6 : a1 + a2 > a3

-- Proof problem 1: It is always possible to form a triangle from the three longest sticks.
theorem longest_sticks_triangle : a1 < a2 + a3 := by sorry

-- Assuming an additional condition for proof problem 2
axiom two_triangles_formed : ∃ b1 b2 b3 b4 b5 b6: ℝ, 
  ((b1 + b2 > b3 ∧ b1 + b3 > b2 ∧ b2 + b3 > b1) ∧
   (b4 + b5 > b6 ∧ b4 + b6 > b5 ∧ b5 + b6 > b4 ∧ 
    a1 = b1 ∧ a2 = b2 ∧ a3 = b3 ∧ a4 = b4 ∧ a5 = b5 ∧ a6 = b6))

-- Proof problem 2: It is not always possible to form a triangle from the three shortest sticks.
theorem shortest_sticks_not_triangle : ¬(a4 < a5 + a6 ∧ a5 < a4 + a6 ∧ a6 < a4 + a5) := by sorry

end longest_sticks_triangle_shortest_sticks_not_triangle_l92_92619


namespace m_greater_than_p_l92_92316

theorem m_greater_than_p (p m n : ℕ) (pp : Nat.Prime p) (pos_m : m > 0) (pos_n : n > 0) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_greater_than_p_l92_92316


namespace find_a9_l92_92639

variable (a : ℕ → ℝ)  -- Define a sequence a_n.

-- Define the conditions for the arithmetic sequence.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

variables (h_arith_seq : is_arithmetic_sequence a)
          (h_a3 : a 3 = 8)   -- Condition a_3 = 8
          (h_a6 : a 6 = 5)   -- Condition a_6 = 5 

-- State the theorem.
theorem find_a9 : a 9 = 2 := by
  sorry

end find_a9_l92_92639


namespace weight_of_new_man_l92_92808

theorem weight_of_new_man (avg_increase : ℝ) (num_oarsmen : ℕ) (old_weight : ℝ) (weight_increase : ℝ) 
  (h1 : avg_increase = 1.8) (h2 : num_oarsmen = 10) (h3 : old_weight = 53) (h4 : weight_increase = num_oarsmen * avg_increase) :
  ∃ W : ℝ, W = old_weight + weight_increase :=
by
  sorry

end weight_of_new_man_l92_92808


namespace gcd_20586_58768_l92_92970

theorem gcd_20586_58768 : Int.gcd 20586 58768 = 2 := by
  sorry

end gcd_20586_58768_l92_92970


namespace largest_increase_is_2007_2008_l92_92425

-- Define the number of students each year
def students_2005 : ℕ := 50
def students_2006 : ℕ := 55
def students_2007 : ℕ := 60
def students_2008 : ℕ := 70
def students_2009 : ℕ := 72
def students_2010 : ℕ := 80

-- Define the percentage increase function
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old) : ℚ) / old * 100

-- Define percentage increases for each pair of consecutive years
def increase_2005_2006 := percentage_increase students_2005 students_2006
def increase_2006_2007 := percentage_increase students_2006 students_2007
def increase_2007_2008 := percentage_increase students_2007 students_2008
def increase_2008_2009 := percentage_increase students_2008 students_2009
def increase_2009_2010 := percentage_increase students_2009 students_2010

-- State the theorem
theorem largest_increase_is_2007_2008 :
  (max (max increase_2005_2006 (max increase_2006_2007 increase_2008_2009))
       increase_2009_2010) < increase_2007_2008 := 
by
  -- Add proof steps if necessary.
  sorry

end largest_increase_is_2007_2008_l92_92425


namespace house_construction_days_l92_92047

theorem house_construction_days
  (D : ℕ) -- number of planned days to build the house
  (Hwork_done : 1000 + 200 * (D - 10) = 100 * (D + 90)) : 
  D = 110 :=
sorry

end house_construction_days_l92_92047


namespace fraction_of_second_year_students_not_declared_major_l92_92309

theorem fraction_of_second_year_students_not_declared_major (T : ℕ) :
  (1 / 2 : ℝ) * (1 - (1 / 3 * (1 / 5))) = 7 / 15 :=
by
  sorry

end fraction_of_second_year_students_not_declared_major_l92_92309


namespace jane_journey_duration_l92_92267

noncomputable def hours_to_seconds (h : ℕ) : ℕ := h * 3600 + 30

theorem jane_journey_duration :
  ∃ (start_time end_time : ℕ), 
    (start_time > 10 * 3600) ∧ (start_time < 11 * 3600) ∧
    (end_time > 17 * 3600) ∧ (end_time < 18 * 3600) ∧
    end_time - start_time = hours_to_seconds 7 :=
by sorry

end jane_journey_duration_l92_92267


namespace hyperbola_standard_equation_equation_of_line_L_l92_92987

open Real

noncomputable def hyperbola (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

noncomputable def focus_on_y_axis := ∃ c : ℝ, c = 2

noncomputable def asymptote (x y : ℝ) : Prop := 
  y = sqrt 3 / 3 * x ∨ y = - sqrt 3 / 3 * x

noncomputable def point_A := (1, 1 / 2)

noncomputable def line_L (x y : ℝ) : Prop :=
  4 * x - 6 * y - 1 = 0

theorem hyperbola_standard_equation :
  ∃ (x y: ℝ), hyperbola x y :=
sorry

theorem equation_of_line_L :
  ∀ (x y : ℝ), point_A = (1, 1 / 2) ∧ line_L x y :=
sorry

end hyperbola_standard_equation_equation_of_line_L_l92_92987


namespace stock_investment_net_increase_l92_92045

theorem stock_investment_net_increase :
  ∀ (initial_investment : ℝ)
    (increase_first_year : ℝ)
    (decrease_second_year : ℝ)
    (increase_third_year : ℝ),
  initial_investment = 100 → 
  increase_first_year = 0.60 → 
  decrease_second_year = 0.30 → 
  increase_third_year = 0.20 → 
  ((initial_investment * (1 + increase_first_year)) * (1 - decrease_second_year)) * (1 + increase_third_year) - initial_investment = 34.40 :=
by 
  intros initial_investment increase_first_year decrease_second_year increase_third_year 
  intros h_initial_investment h_increase_first_year h_decrease_second_year h_increase_third_year 
  rw [h_initial_investment, h_increase_first_year, h_decrease_second_year, h_increase_third_year]
  sorry

end stock_investment_net_increase_l92_92045


namespace length_of_AP_l92_92787

variables {x : ℝ} (M B C P A : Point) (circle : Circle)
  (BC AB MP : Line)

-- Definitions of conditions
def is_midpoint_of_arc (M B C : Point) (circle : Circle) : Prop := sorry
def is_perpendicular (MP AB : Line) (P : Point) : Prop := sorry
def chord_length (BC : Line) (length : ℝ) : Prop := sorry
def segment_length (BP : Line) (length : ℝ) : Prop := sorry

-- Prove statement
theorem length_of_AP
  (h1 : is_midpoint_of_arc M B C circle)
  (h2 : is_perpendicular MP AB P)
  (h3 : chord_length BC (2 * x))
  (h4 : segment_length BP (3 * x)) :
  ∃AP : Line, segment_length AP (2 * x) :=
sorry

end length_of_AP_l92_92787


namespace garden_length_l92_92709

theorem garden_length (w l : ℝ) (h1: l = 2 * w) (h2 : 2 * l + 2 * w = 180) : l = 60 := 
by
  sorry

end garden_length_l92_92709


namespace find_a3_l92_92087

theorem find_a3 (a : ℕ → ℕ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, (1 + 2 * a (n + 1)) = (1 + 2 * a n) + 1) : a 3 = 3 :=
by
  -- This is where the proof would go, but we'll leave it as sorry for now.
  sorry

end find_a3_l92_92087


namespace part1_part2_l92_92487

-- Definitions for problem conditions and questions

/-- 
Let p and q be two distinct prime numbers greater than 5. 
Show that if p divides 5^q - 2^q then q divides p - 1.
-/
theorem part1 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div : p ∣ 5^q - 2^q) : q ∣ p - 1 :=
by sorry

/-- 
Let p and q be two distinct prime numbers greater than 5.
Deduce that pq does not divide (5^p - 2^p)(5^q - 2^q).
-/
theorem part2 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div_q_p1 : q ∣ p - 1)
  (h_div_p_q1 : p ∣ q - 1) : ¬(pq : ℕ) ∣ (5^p - 2^p) * (5^q - 2^q) :=
by sorry

end part1_part2_l92_92487


namespace inequality_solution_set_nonempty_l92_92719

-- Define the statement
theorem inequality_solution_set_nonempty (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) ↔ m > 2 :=
by
  sorry

end inequality_solution_set_nonempty_l92_92719


namespace valid_q_values_l92_92356

theorem valid_q_values (q : ℕ) (h : q > 0) :
  q = 3 ∨ q = 4 ∨ q = 9 ∨ q = 28 ↔ ((5 * q + 40) / (3 * q - 8)) * (3 * q - 8) = 5 * q + 40 :=
by
  sorry

end valid_q_values_l92_92356


namespace hypotenuse_length_l92_92668

theorem hypotenuse_length (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a^2 + b^2 + c^2 = 1800) : 
  c = 30 :=
sorry

end hypotenuse_length_l92_92668


namespace total_rainfall_l92_92724

theorem total_rainfall :
  let monday := 0.12962962962962962
  let tuesday := 0.35185185185185186
  let wednesday := 0.09259259259259259
  let thursday := 0.25925925925925924
  let friday := 0.48148148148148145
  let saturday := 0.2222222222222222
  let sunday := 0.4444444444444444
  (monday + tuesday + wednesday + thursday + friday + saturday + sunday) = 1.9814814814814815 :=
by
  -- proof to be filled here
  sorry

end total_rainfall_l92_92724


namespace find_number_l92_92748

theorem find_number (x : ℤ) 
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 := 
by 
  sorry

end find_number_l92_92748


namespace problem_statement_l92_92089

noncomputable def solveProblem : ℝ :=
  let a := 2
  let b := -3
  let c := 1
  a + b + c

-- The theorem statement to ensure a + b + c equals 0
theorem problem_statement : solveProblem = 0 := by
  sorry

end problem_statement_l92_92089


namespace geometric_sequence_seventh_term_l92_92866

theorem geometric_sequence_seventh_term
  (a r : ℝ)
  (h1 : a * r^4 = 16)
  (h2 : a * r^10 = 4) :
  a * r^6 = 4 * (2^(2/3)) :=
by
  sorry

end geometric_sequence_seventh_term_l92_92866


namespace value_of_expression_l92_92065

theorem value_of_expression : 3 ^ (0 ^ (2 ^ 11)) + ((3 ^ 0) ^ 2) ^ 11 = 2 := by
  sorry

end value_of_expression_l92_92065


namespace gain_percent_is_approx_30_11_l92_92576

-- Definitions for cost price (CP) and selling price (SP)
def CP : ℕ := 930
def SP : ℕ := 1210

-- Definition for gain percent
noncomputable def gain_percent : ℚ :=
  ((SP - CP : ℚ) / CP) * 100

-- Statement to prove the gain percent is approximately 30.11%
theorem gain_percent_is_approx_30_11 :
  abs (gain_percent - 30.11) < 0.01 := by
  sorry

end gain_percent_is_approx_30_11_l92_92576


namespace trigonometric_identity_l92_92781

-- Define variables
variables (α : ℝ) (hα : α ∈ Ioc 0 π) (h_tan : Real.tan α = 2)

-- The Lean statement
theorem trigonometric_identity :
  Real.cos (5 * Real.pi / 2 + 2 * α) = -4 / 5 :=
sorry

end trigonometric_identity_l92_92781


namespace man_l92_92749

-- Lean 4 statement
theorem man's_speed_against_stream (speed_with_stream : ℝ) (speed_still_water : ℝ) 
(h1 : speed_with_stream = 16) (h2 : speed_still_water = 4) : 
  |speed_still_water - (speed_with_stream - speed_still_water)| = 8 :=
by
  -- Dummy proof since only statement is required
  sorry

end man_l92_92749


namespace Carlos_earnings_l92_92149

theorem Carlos_earnings :
  ∃ (wage : ℝ), 
  (18 * wage) = (12 * wage + 36) ∧ 
  wage = 36 / 6 ∧ 
  (12 * wage + 18 * wage) = 180 :=
by
  sorry

end Carlos_earnings_l92_92149


namespace quadratic_solution_difference_l92_92694

theorem quadratic_solution_difference : 
  ∃ a b : ℝ, (a^2 - 12 * a + 20 = 0) ∧ (b^2 - 12 * b + 20 = 0) ∧ (a > b) ∧ (a - b = 8) :=
by
  sorry

end quadratic_solution_difference_l92_92694


namespace fundraiser_total_money_l92_92600

def fundraiser_money : ℝ :=
  let brownies_students := 70
  let brownies_each := 20
  let brownies_price := 1.50
  let cookies_students := 40
  let cookies_each := 30
  let cookies_price := 2.25
  let donuts_students := 35
  let donuts_each := 18
  let donuts_price := 3.00
  let cupcakes_students := 25
  let cupcakes_each := 12
  let cupcakes_price := 2.50
  let total_brownies := brownies_students * brownies_each
  let total_cookies := cookies_students * cookies_each
  let total_donuts := donuts_students * donuts_each
  let total_cupcakes := cupcakes_students * cupcakes_each
  let money_brownies := total_brownies * brownies_price
  let money_cookies := total_cookies * cookies_price
  let money_donuts := total_donuts * donuts_price
  let money_cupcakes := total_cupcakes * cupcakes_price
  money_brownies + money_cookies + money_donuts + money_cupcakes

theorem fundraiser_total_money : fundraiser_money = 7440 := sorry

end fundraiser_total_money_l92_92600


namespace compare_log_inequalities_l92_92415

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem compare_log_inequalities (a x1 x2 : ℝ) 
  (ha_pos : a > 0) (ha_neq_one : a ≠ 1) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (a > 1 → 1 / 2 * (f a x1 + f a x2) ≤ f a ((x1 + x2) / 2)) ∧
  (0 < a ∧ a < 1 → 1 / 2 * (f a x1 + f a x2) ≥ f a ((x1 + x2) / 2)) :=
by { sorry }

end compare_log_inequalities_l92_92415


namespace halfway_fraction_l92_92320

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end halfway_fraction_l92_92320


namespace triceratops_count_l92_92510

theorem triceratops_count (r t : ℕ) 
  (h_legs : 4 * r + 4 * t = 48) 
  (h_horns : 2 * r + 3 * t = 31) : 
  t = 7 := 
by 
  hint

/- The given conditions are:
1. Each rhinoceros has 2 horns.
2. Each triceratops has 3 horns.
3. Each animal has 4 legs.
4. There is a total of 31 horns.
5. There is a total of 48 legs.

Using these conditions and the equations derived from them, we need to prove that the number of triceratopses (t) is 7.
-/

end triceratops_count_l92_92510


namespace sweets_distribution_l92_92957

theorem sweets_distribution (S X : ℕ) (h1 : S = 112 * X) (h2 : S = 80 * (X + 6)) :
  X = 15 := 
by
  sorry

end sweets_distribution_l92_92957


namespace area_of_given_triangle_l92_92083

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : 
  triangle_area (1, 1) (7, 1) (5, 3) = 6 :=
by
  -- the proof should go here
  sorry

end area_of_given_triangle_l92_92083


namespace unit_digit_power3_58_l92_92596

theorem unit_digit_power3_58 : (3 ^ 58) % 10 = 9 := by
  -- proof steps will be provided here
  sorry

end unit_digit_power3_58_l92_92596


namespace find_value_of_a_l92_92827

-- Let a, b, and c be different numbers from {1, 2, 4}
def a_b_c_valid (a b c : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
  (a = 1 ∨ a = 2 ∨ a = 4) ∧ 
  (b = 1 ∨ b = 2 ∨ b = 4) ∧ 
  (c = 1 ∨ c = 2 ∨ c = 4)

-- The condition that (a / 2) / (b / c) equals 4 when evaluated
def expr_eq_four (a b c : ℕ) : Prop :=
  (a / 2 : ℚ) / (b / c : ℚ) = 4

-- Given the above conditions, prove that the value of 'a' is 4
theorem find_value_of_a (a b c : ℕ) (h_valid : a_b_c_valid a b c) (h_expr : expr_eq_four a b c) : a = 4 := 
  sorry

end find_value_of_a_l92_92827


namespace combined_tickets_l92_92395

-- Definitions for the initial conditions
def stuffedTigerPrice : ℝ := 43
def keychainPrice : ℝ := 5.5
def discount1 : ℝ := 0.20 * stuffedTigerPrice
def discountedTigerPrice : ℝ := stuffedTigerPrice - discount1
def ticketsLeftDave : ℝ := 55
def spentDave : ℝ := discountedTigerPrice + keychainPrice
def initialTicketsDave : ℝ := spentDave + ticketsLeftDave

def dinoToyPrice : ℝ := 65
def discount2 : ℝ := 0.15 * dinoToyPrice
def discountedDinoToyPrice : ℝ := dinoToyPrice - discount2
def ticketsLeftAlex : ℝ := 42
def spentAlex : ℝ := discountedDinoToyPrice
def initialTicketsAlex : ℝ := spentAlex + ticketsLeftAlex

-- Lean statement proving the combined number of tickets at the start
theorem combined_tickets {dave_alex_combined : ℝ} 
    (h1 : dave_alex_combined = initialTicketsDave + initialTicketsAlex) : 
    dave_alex_combined = 192.15 := 
by 
    -- Placeholder for the actual proof
    sorry

end combined_tickets_l92_92395


namespace mario_meet_speed_l92_92786

noncomputable def Mario_average_speed (x : ℝ) : ℝ :=
  let t1 := x / 5
  let t2 := x / 3
  let t3 := x / 4
  let t4 := x / 10
  let T := t1 + t2 + t3 + t4
  let d_mario := 1.5 * x
  d_mario / T

theorem mario_meet_speed : ∀ (x : ℝ), x > 0 → Mario_average_speed x = 90 / 53 :=
by
  intros
  rw [Mario_average_speed]
  -- You can insert calculations similar to those in the provided solution
  sorry

end mario_meet_speed_l92_92786


namespace factor_expression_l92_92627

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l92_92627


namespace xiao_ming_reading_plan_l92_92508

-- Define the number of pages in the book
def total_pages : Nat := 72

-- Define the total number of days to finish the book
def total_days : Nat := 10

-- Define the number of pages read per day for the first two days
def pages_first_two_days : Nat := 5

-- Define the variable x to represent the number of pages read per day for the remaining days
variable (x : Nat)

-- Define the inequality representing the reading plan
def reading_inequality (x : Nat) : Prop :=
  10 + 8 * x ≥ total_pages

-- The statement to be proved
theorem xiao_ming_reading_plan (x : Nat) : reading_inequality x := sorry

end xiao_ming_reading_plan_l92_92508


namespace solve_for_x_l92_92180

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 8) * x = 12) : x = 168 := by
  sorry

end solve_for_x_l92_92180


namespace inclination_line_eq_l92_92966

theorem inclination_line_eq (l : ℝ → ℝ) (h1 : ∃ x, l x = 2 ∧ ∃ y, l y = 2) (h2 : ∃ θ, θ = 135) :
  ∃ a b c, a = 1 ∧ b = 1 ∧ c = -4 ∧ ∀ x y, y = l x → a * x + b * y + c = 0 :=
by 
  sorry

end inclination_line_eq_l92_92966


namespace integer_roots_of_polynomial_l92_92147

theorem integer_roots_of_polynomial :
  {x : ℤ | x^3 - 4*x^2 - 14*x + 24 = 0} = {-4, -3, 3} := by
  sorry

end integer_roots_of_polynomial_l92_92147


namespace dot_product_is_4_l92_92727

-- Define vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, -1)

-- Define the condition that a is parallel to (a + b)
def is_parallel (u v : ℝ × ℝ) : Prop := 
  (u.1 * v.2 - u.2 * v.1) = 0

theorem dot_product_is_4 (x : ℝ) (h_parallel : is_parallel (a x) (a x + b)) : 
  (a x).1 * b.1 + (a x).2 * b.2 = 4 :=
sorry

end dot_product_is_4_l92_92727


namespace isosceles_triangle_has_perimeter_22_l92_92278

noncomputable def isosceles_triangle_perimeter (a b : ℕ) : ℕ :=
if a + a > b ∧ a + b > a ∧ b + b > a then a + a + b else 0

theorem isosceles_triangle_has_perimeter_22 :
  isosceles_triangle_perimeter 9 4 = 22 :=
by 
  -- Add a note for clarity; this will be completed via 'sorry'
  -- Prove that with side lengths 9 and 4 (with 9 being the equal sides),
  -- they form a valid triangle and its perimeter is 22
  sorry

end isosceles_triangle_has_perimeter_22_l92_92278


namespace simplify_tan_cot_60_l92_92080

theorem simplify_tan_cot_60 :
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  (tan60^3 + cot60^3) / (tan60 + cot60) = 7 / 3 :=
by
  let tan60 := Real.sqrt 3
  let cot60 := 1 / Real.sqrt 3
  sorry

end simplify_tan_cot_60_l92_92080


namespace largest_digit_M_divisible_by_six_l92_92771

theorem largest_digit_M_divisible_by_six :
  (∃ M : ℕ, M ≤ 9 ∧ (45670 + M) % 6 = 0 ∧ ∀ m : ℕ, m ≤ M → (45670 + m) % 6 ≠ 0) :=
sorry

end largest_digit_M_divisible_by_six_l92_92771


namespace loss_percentage_on_first_book_l92_92715

variable (C1 C2 SP L : ℝ)
variable (total_cost : ℝ := 540)
variable (C1_value : ℝ := 315)
variable (gain_percentage : ℝ := 0.19)
variable (common_selling_price : ℝ := 267.75)

theorem loss_percentage_on_first_book :
  C1 = C1_value →
  C2 = total_cost - C1 →
  SP = 1.19 * C2 →
  SP = C1 - (L / 100 * C1) →
  L = 15 :=
sorry

end loss_percentage_on_first_book_l92_92715


namespace correct_operation_A_l92_92473

-- Definitions for the problem
def division_rule (a : ℝ) (m n : ℕ) : Prop := a^m / a^n = a^(m - n)
def multiplication_rule (a : ℝ) (m n : ℕ) : Prop := a^m * a^n = a^(m + n)
def power_rule (a : ℝ) (m n : ℕ) : Prop := (a^m)^n = a^(m * n)
def addition_like_terms_rule (a : ℝ) (m : ℕ) : Prop := a^m + a^m = 2 * a^m

-- The theorem to prove
theorem correct_operation_A (a : ℝ) : division_rule a 4 2 :=
by {
  sorry
}

end correct_operation_A_l92_92473


namespace problems_per_hour_l92_92790

def num_math_problems : ℝ := 17.0
def num_spelling_problems : ℝ := 15.0
def total_hours : ℝ := 4.0

theorem problems_per_hour :
  (num_math_problems + num_spelling_problems) / total_hours = 8.0 := by
  sorry

end problems_per_hour_l92_92790


namespace valentine_count_initial_l92_92927

def valentines_given : ℕ := 42
def valentines_left : ℕ := 16
def valentines_initial := valentines_given + valentines_left

theorem valentine_count_initial :
  valentines_initial = 58 :=
by
  sorry

end valentine_count_initial_l92_92927


namespace inverse_value_at_2_l92_92968

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := x / (1 - 2 * x)

theorem inverse_value_at_2 :
  f_inv 2 = -2/3 := by
  sorry

end inverse_value_at_2_l92_92968


namespace find_c_l92_92905

theorem find_c (x : ℝ) (c : ℝ) (h1: 3 * x + 6 = 0) (h2: c * x + 15 = 3) : c = 6 := 
by
  sorry

end find_c_l92_92905


namespace average_age_of_team_l92_92681

def total_age (A : ℕ) (N : ℕ) := A * N
def wicket_keeper_age (A : ℕ) := A + 3
def remaining_players_age (A : ℕ) (N : ℕ) (W : ℕ) := (total_age A N) - (A + W)

theorem average_age_of_team
  (A : ℕ)
  (N : ℕ)
  (H1 : N = 11)
  (H2 : A = 28)
  (W : ℕ)
  (H3 : W = wicket_keeper_age A)
  (H4 : (wicket_keeper_age A) = A + 3)
  : (remaining_players_age A N W) / (N - 2) = A - 1 :=
by
  rw [H1, H2, H3, H4]; sorry

end average_age_of_team_l92_92681


namespace length_of_MN_l92_92232

theorem length_of_MN (A B C D K L M N : Type) 
  (h1 : A → B → C → D → Prop) -- Condition for rectangle ABCD
  (h2 : K → L → Prop) -- Condition for circle intersecting AB at K and L
  (h3 : M → N → Prop) -- Condition for circle intersecting CD at M and N
  (AK KL DN : ℝ)
  (h4 : AK = 10)
  (h5 : KL = 17)
  (h6 : DN = 7) :
  ∃ MN : ℝ, MN = 23 := 
sorry

end length_of_MN_l92_92232


namespace standard_equation_line_BC_fixed_point_l92_92010

section EllipseProof

open Real

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions from the problem
axiom a_gt_b_gt_0 : ∀ (a b : ℝ), a > b → b > 0
axiom passes_through_point : ∀ (a b x y : ℝ), ellipse a b x y → (x = 1 ∧ y = sqrt 2 / 2)
axiom has_eccentricity : ∀ (a b c : ℝ), c / a = sqrt 2 / 2 → c^2 = a^2 - b^2 → b = 1

-- The standard equation of the ellipse
theorem standard_equation (a b : ℝ) (x y : ℝ) :
  a = sqrt 2 → b = 1 → ellipse a b x y → ellipse (sqrt 2) 1 x y :=
sorry

-- Prove that BC always passes through a fixed point
theorem line_BC_fixed_point (a b x1 x2 y1 y2 : ℝ) :
  a = sqrt 2 → b = 1 → 
  ellipse a b x1 y1 → ellipse a b x2 y2 →
  y1 = -y2 → x1 ≠ x2 → (-1, 0) = (-1, 0) →
  ∃ (k : ℝ) (x : ℝ), x = -2 ∧ y = 0 :=
sorry

end EllipseProof

end standard_equation_line_BC_fixed_point_l92_92010


namespace final_salt_concentration_is_25_l92_92862

-- Define the initial conditions
def original_solution_weight : ℝ := 100
def original_salt_concentration : ℝ := 0.10
def added_salt_weight : ℝ := 20

-- Define the amount of salt in the original solution
def original_salt_weight := original_solution_weight * original_salt_concentration

-- Define the total amount of salt after adding pure salt
def total_salt_weight := original_salt_weight + added_salt_weight

-- Define the total weight of the new solution
def new_solution_weight := original_solution_weight + added_salt_weight

-- Define the final salt concentration
noncomputable def final_salt_concentration := (total_salt_weight / new_solution_weight) * 100

-- Prove the final salt concentration equals 25%
theorem final_salt_concentration_is_25 : final_salt_concentration = 25 :=
by
  sorry

end final_salt_concentration_is_25_l92_92862


namespace birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l92_92280

theorem birds_percentage_not_hawks_paddyfield_warblers_kingfishers
  (total_birds : ℕ)
  (hawks_percentage : ℝ := 0.3)
  (paddyfield_warblers_percentage : ℝ := 0.4)
  (kingfishers_ratio : ℝ := 0.25) :
  (35 : ℝ) = 100 * ( total_birds - (hawks_percentage * total_birds) 
                     - (paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) 
                     - (kingfishers_ratio * paddyfield_warblers_percentage * (total_birds - (hawks_percentage * total_birds))) )
                / total_birds :=
by
  sorry

end birds_percentage_not_hawks_paddyfield_warblers_kingfishers_l92_92280


namespace rationalize_denominator_l92_92292

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
  (1 / (Real.sqrt 5 + Real.sqrt 2 + Real.sqrt 11)) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
  A + B + C + D + E + F = 136 := 
sorry

end rationalize_denominator_l92_92292


namespace solve_equation_l92_92598

theorem solve_equation (x : ℝ) : 
  (9 - 3 * x) * (3 ^ x) - (x - 2) * (x ^ 2 - 5 * x + 6) = 0 ↔ x = 3 :=
by sorry

end solve_equation_l92_92598


namespace rectangular_solid_surface_area_l92_92739

theorem rectangular_solid_surface_area
  (length : ℕ) (width : ℕ) (depth : ℕ)
  (h_length : length = 9) (h_width : width = 8) (h_depth : depth = 5) :
  2 * (length * width + width * depth + length * depth) = 314 := 
  by
  sorry

end rectangular_solid_surface_area_l92_92739


namespace tenth_term_is_correct_l92_92077

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l92_92077


namespace janice_homework_time_l92_92321

variable (H : ℝ)
variable (cleaning_room walk_dog take_trash : ℝ)

-- Conditions from the problem translated directly
def cleaning_room_time : cleaning_room = H / 2 := sorry
def walk_dog_time : walk_dog = H + 5 := sorry
def take_trash_time : take_trash = H / 6 := sorry
def total_time_before_movie : 35 + (H + cleaning_room + walk_dog + take_trash) = 120 := sorry

-- The main theorem to prove
theorem janice_homework_time (H : ℝ)
        (cleaning_room : ℝ := H / 2)
        (walk_dog : ℝ := H + 5)
        (take_trash : ℝ := H / 6) :
    H + cleaning_room + walk_dog + take_trash + 35 = 120 → H = 30 :=
by
  sorry

end janice_homework_time_l92_92321


namespace not_all_ten_segments_form_triangle_l92_92282

theorem not_all_ten_segments_form_triangle :
  ∃ (segments : Fin 10 → ℕ), ∀ i j k : Fin 10, i < j → j < k → segments i + segments j ≤ segments k := 
sorry

end not_all_ten_segments_form_triangle_l92_92282


namespace angle_A_is_equilateral_l92_92716

namespace TriangleProof

variables {A B C : ℝ} {a b c : ℝ}

-- Given condition (a+b+c)(a-b-c) + 3bc = 0
def condition1 (a b c : ℝ) : Prop := (a + b + c) * (a - b - c) + 3 * b * c = 0

-- Given condition a = 2c * cos B
def condition2 (a c B : ℝ) : Prop := a = 2 * c * Real.cos B

-- Prove that if (a+b+c)(a-b-c) + 3bc = 0, then A = π / 3
theorem angle_A (h1 : condition1 a b c) : A = Real.pi / 3 :=
sorry

-- Prove that if a = 2c * cos B and A = π / 3, then ∆ ABC is an equilateral triangle
theorem is_equilateral (h2 : condition2 a c B) (hA : A = Real.pi / 3) : 
  b = c ∧ a = b ∧ B = C :=
sorry

end TriangleProof

end angle_A_is_equilateral_l92_92716


namespace blue_faces_cube_l92_92470

theorem blue_faces_cube (n : ℕ) (h1 : n > 0) (h2 : (6 * n^2) = 1 / 3 * 6 * n^3) : n = 3 :=
by
  -- we only need the statement for now; the proof is omitted.
  sorry

end blue_faces_cube_l92_92470


namespace correct_statements_l92_92703

def problem_statements :=
  [ "The negation of the statement 'There exists an x ∈ ℝ such that x^2 - 3x + 3 = 0' is true.",
    "The statement '-1/2 < x < 0' is a necessary but not sufficient condition for '2x^2 - 5x - 3 < 0'.",
    "The negation of the statement 'If xy = 0, then at least one of x or y is equal to 0' is true.",
    "The curves x^2/25 + y^2/9 = 1 and x^2/(25 − k) + y^2/(9 − k) = 1 (9 < k < 25) share the same foci.",
    "There exists a unique line that passes through the point (1,3) and is tangent to the parabola y^2 = 4x."
  ]

theorem correct_statements :
  (∀ x : ℝ, ¬(x^2 - 3 * x + 3 = 0)) ∧ 
  ¬ (¬-1/2 < x ∧ x < 0 → 2 * x^2 - 5*x - 3 < 0) ∧ 
  (∀ x y : ℝ, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧ 
  (∀ k : ℝ, 9 < k ∧ k < 25 → ∀ x y : ℝ, (x^2 / (25 - k) + y^2 / (9 - k) = 1) → (x^2 / 25 + y^2 / 9 = 1) → (x ≠ 0 ∨ y ≠ 0)) ∧ 
  ¬ (∃ l : ℝ, ∀ pt : ℝ × ℝ, pt = (1, 3) → ∀ y : ℝ, y^2 = 4 * pt.1 → y = 2 * pt.2)
:= 
  sorry

end correct_statements_l92_92703


namespace find_n_l92_92413

def C (k : ℕ) : ℕ :=
  if k = 1 then 0
  else (Nat.factors k).eraseDup.foldr (· + ·) 0

theorem find_n (n : ℕ) : 
  (∀ n, (C (2 ^ n + 1) = C n) ↔ n = 3) := 
by
  sorry

end find_n_l92_92413


namespace sum_of_reciprocals_l92_92467

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end sum_of_reciprocals_l92_92467


namespace find_n_l92_92979

theorem find_n (n : ℕ) (hn : (n - 2) * (n - 3) / 12 = 14 / 3) : n = 10 := by
  sorry

end find_n_l92_92979


namespace solution_positive_iff_k_range_l92_92542

theorem solution_positive_iff_k_range (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x ≠ 2 ∧ (k / (2 * x - 4) - 1 = x / (x - 2))) ↔ (k > -4 ∧ k ≠ 4) := 
sorry

end solution_positive_iff_k_range_l92_92542


namespace last_digit_101_pow_100_l92_92960

theorem last_digit_101_pow_100 :
  (101^100) % 10 = 1 :=
by
  sorry

end last_digit_101_pow_100_l92_92960


namespace max_sigma_squared_l92_92189

theorem max_sigma_squared (c d : ℝ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_c_ge_d : c ≥ d)
    (h : ∃ x y : ℝ, 0 ≤ x ∧ x < c ∧ 0 ≤ y ∧ y < d ∧ 
      c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x) ^ 2 + (d - y) ^ 2) : 
    σ^2 = 4 / 3 := by
  sorry

end max_sigma_squared_l92_92189


namespace response_activity_solutions_l92_92427

theorem response_activity_solutions (x y z : ℕ) :
  5 * x + 4 * y + 3 * z = 15 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 0 ∧ y = 3 ∧ z = 1) :=
by
  sorry

end response_activity_solutions_l92_92427


namespace money_left_after_spending_l92_92471

def initial_money : ℕ := 24
def doris_spent : ℕ := 6
def martha_spent : ℕ := doris_spent / 2
def total_spent : ℕ := doris_spent + martha_spent
def money_left := initial_money - total_spent

theorem money_left_after_spending : money_left = 15 := by
  sorry

end money_left_after_spending_l92_92471


namespace expression_for_x_expression_for_y_l92_92811

variables {A B C : ℝ}

-- Conditions: A, B, and C are positive numbers with A > B > C > 0
axiom h1 : A > 0
axiom h2 : B > 0
axiom h3 : C > 0
axiom h4 : A > B
axiom h5 : B > C

-- A is x% greater than B
variables {x : ℝ}
axiom h6 : A = (1 + x / 100) * B

-- A is y% greater than C
variables {y : ℝ}
axiom h7 : A = (1 + y / 100) * C

-- Proving the expressions for x and y
theorem expression_for_x : x = 100 * ((A - B) / B) :=
sorry

theorem expression_for_y : y = 100 * ((A - C) / C) :=
sorry

end expression_for_x_expression_for_y_l92_92811


namespace trapezoid_sides_and_height_l92_92756

def trapezoid_base_height (a h A: ℝ) :=
  (h = (2 * a + 3) / 2) ∧
  (A = a^2 + 3 * a + 9 / 4) ∧
  (A = 2 * a^2 - 7.75)

theorem trapezoid_sides_and_height :
  ∃ (a b h : ℝ), (b = a + 3) ∧
  trapezoid_base_height a h 7.75 ∧
  a = 5 ∧ b = 8 ∧ h = 6.5 :=
by
  sorry

end trapezoid_sides_and_height_l92_92756


namespace necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l92_92840

variable (x y : ℝ)

theorem necessary_but_not_sufficient (hx : x < y ∧ y < 0) : x^2 > y^2 :=
sorry

theorem not_sufficient (hx : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
sorry

-- Optional: Combining the two to create a combined theorem statement
theorem x2_gt_y2_iff_x_lt_y_lt_0 : (∀ x y : ℝ, x < y ∧ y < 0 → x^2 > y^2) ∧ (∃ x y : ℝ, x^2 > y^2 ∧ ¬ (x < y ∧ y < 0)) :=
sorry

end necessary_but_not_sufficient_not_sufficient_x2_gt_y2_iff_x_lt_y_lt_0_l92_92840


namespace area_of_curves_l92_92906

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0:ℝ)..1, (Real.sqrt x - x^2)

theorem area_of_curves :
  enclosed_area = 1 / 3 :=
sorry

end area_of_curves_l92_92906


namespace number_of_nickels_l92_92299

variable (n : Nat) -- number of nickels

def value_of_nickels := n * 5 -- value of nickels n in cents
def total_value :=
    2 * 100 +   -- 2 one-dollar bills
    1 * 500 +   -- 1 five-dollar bill
    13 * 25 +   -- 13 quarters
    20 * 10 +   -- 20 dimes
    35 * 1 +    -- 35 pennies
    value_of_nickels n

theorem number_of_nickels :
    total_value n = 1300 ↔ n = 8 :=
by sorry

end number_of_nickels_l92_92299


namespace last_three_digits_of_16_pow_128_l92_92017

theorem last_three_digits_of_16_pow_128 : (16 ^ 128) % 1000 = 721 := 
by
  sorry

end last_three_digits_of_16_pow_128_l92_92017


namespace decrease_in_B_share_l92_92143

theorem decrease_in_B_share (a b c : ℝ) (x : ℝ) 
  (h1 : c = 495)
  (h2 : a + b + c = 1010)
  (h3 : (a - 25) / 3 = (b - x) / 2)
  (h4 : (a - 25) / 3 = (c - 15) / 5) :
  x = 10 :=
by
  sorry

end decrease_in_B_share_l92_92143


namespace find_breadth_of_wall_l92_92049

theorem find_breadth_of_wall
  (b h l V : ℝ)
  (h1 : V = 12.8)
  (h2 : h = 5 * b)
  (h3 : l = 8 * h) :
  b = 0.4 :=
by
  sorry

end find_breadth_of_wall_l92_92049


namespace roger_money_in_january_l92_92303

theorem roger_money_in_january (x : ℝ) (h : (x - 20) + 46 = 71) : x = 45 :=
sorry

end roger_money_in_january_l92_92303


namespace total_players_l92_92132

def num_teams : Nat := 35
def players_per_team : Nat := 23

theorem total_players :
  num_teams * players_per_team = 805 :=
by
  sorry

end total_players_l92_92132


namespace acrobats_count_l92_92762

theorem acrobats_count (a g : ℕ) 
  (h1 : 2 * a + 4 * g = 32) 
  (h2 : a + g = 10) : 
  a = 4 := by
  -- Proof omitted
  sorry

end acrobats_count_l92_92762


namespace percentage_pure_acid_l92_92641

theorem percentage_pure_acid (volume_pure_acid total_volume: ℝ) (h1 : volume_pure_acid = 1.4) (h2 : total_volume = 4) : 
  (volume_pure_acid / total_volume) * 100 = 35 := 
by
  -- Given metric volumes of pure acid and total solution, we need to prove the percentage 
  -- Here, we assert the conditions and conclude the result
  sorry

end percentage_pure_acid_l92_92641


namespace evaluate_expression_l92_92597

theorem evaluate_expression :
  let a := 12
  let b := 14
  let c := 18
  (144 * ((1:ℝ)/b - (1:ℝ)/c) + 196 * ((1:ℝ)/c - (1:ℝ)/a) + 324 * ((1:ℝ)/a - (1:ℝ)/b)) /
  (a * ((1:ℝ)/b - (1:ℝ)/c) + b * ((1:ℝ)/c - (1:ℝ)/a) + c * ((1:ℝ)/a - (1:ℝ)/b)) = a + b + c := by
  sorry

end evaluate_expression_l92_92597


namespace number_of_people_in_each_van_l92_92268

theorem number_of_people_in_each_van (x : ℕ) 
  (h1 : 6 * x + 8 * 18 = 180) : x = 6 :=
by sorry

end number_of_people_in_each_van_l92_92268


namespace nth_derivative_correct_l92_92521

noncomputable def y (x : ℝ) : ℝ :=
  Real.sin (3 * x + 1) + Real.cos (5 * x)

noncomputable def n_th_derivative (n : ℕ) (x : ℝ) : ℝ :=
  3^n * Real.sin ((3 * Real.pi / 2) * n + 3 * x + 1) + 5^n * Real.cos ((3 * Real.pi / 2) * n + 5 * x)

theorem nth_derivative_correct (x : ℝ) (n : ℕ) :
  derivative^[n] y x = n_th_derivative n x :=
by
  sorry

end nth_derivative_correct_l92_92521


namespace cost_of_toilet_paper_roll_l92_92954

-- Definitions of the problem's conditions
def num_toilet_paper_rolls : Nat := 10
def num_paper_towel_rolls : Nat := 7
def num_tissue_boxes : Nat := 3

def cost_per_paper_towel : Real := 2
def cost_per_tissue_box : Real := 2

def total_cost : Real := 35

-- The function to prove
def cost_per_toilet_paper_roll (x : Real) :=
  num_toilet_paper_rolls * x + 
  num_paper_towel_rolls * cost_per_paper_towel + 
  num_tissue_boxes * cost_per_tissue_box = total_cost

-- Statement to prove
theorem cost_of_toilet_paper_roll : 
  cost_per_toilet_paper_roll 1.5 := 
by
  simp [num_toilet_paper_rolls, num_paper_towel_rolls, num_tissue_boxes, cost_per_paper_towel, cost_per_tissue_box, total_cost]
  sorry

end cost_of_toilet_paper_roll_l92_92954


namespace evaluate_expression_l92_92451

theorem evaluate_expression : 8 * ((1 : ℚ) / 3)^3 - 1 = -19 / 27 := by
  sorry

end evaluate_expression_l92_92451


namespace sum_first_six_terms_l92_92777

-- Define the conditions given in the problem
def a3 := 7
def a4 := 11
def a5 := 15

-- Define the common difference
def d := a4 - a3 -- 4

-- Define the first term
def a1 := a3 - 2 * d -- -1

-- Define the sum of the first six terms of the arithmetic sequence
def S6 := (6 / 2) * (2 * a1 + (6 - 1) * d) -- 54

-- The theorem we want to prove
theorem sum_first_six_terms : S6 = 54 := by
  sorry

end sum_first_six_terms_l92_92777


namespace first_number_in_expression_l92_92577

theorem first_number_in_expression (a b c d e : ℝ)
  (h_expr : (a * b * c) / d + e = 2229) :
  a = 26.3 :=
  sorry

end first_number_in_expression_l92_92577


namespace minimum_cost_is_correct_l92_92813

noncomputable def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def flower_cost_per_sqft (flower : String) : ℝ :=
  match flower with
  | "Marigold" => 1.00
  | "Sunflower" => 1.75
  | "Tulip" => 1.25
  | "Orchid" => 2.75
  | "Iris" => 3.25
  | _ => 0.00

def min_garden_cost : ℝ :=
  let areas := [rectangular_area 5 2, rectangular_area 7 3, rectangular_area 5 5, rectangular_area 2 4, rectangular_area 5 4]
  let costs := [flower_cost_per_sqft "Orchid" * 8, 
                flower_cost_per_sqft "Iris" * 10, 
                flower_cost_per_sqft "Sunflower" * 20, 
                flower_cost_per_sqft "Tulip" * 21, 
                flower_cost_per_sqft "Marigold" * 25]
  costs.sum

theorem minimum_cost_is_correct :
  min_garden_cost = 140.75 :=
  by
    -- Proof omitted
    sorry

end minimum_cost_is_correct_l92_92813


namespace time_differences_l92_92830

def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 8 -- minutes per mile
def lila_speed := 7 -- minutes per mile
def race_distance := 12 -- miles

noncomputable def malcolm_time := malcolm_speed * race_distance
noncomputable def joshua_time := joshua_speed * race_distance
noncomputable def lila_time := lila_speed * race_distance

theorem time_differences :
  joshua_time - malcolm_time = 24 ∧
  lila_time - malcolm_time = 12 :=
by
  sorry

end time_differences_l92_92830


namespace proportion_equivalence_l92_92538

variable {x y : ℝ}

theorem proportion_equivalence (h : 3 * x = 5 * y) (hy : y ≠ 0) : 
  x / 5 = y / 3 :=
by
  -- Proof goes here
  sorry

end proportion_equivalence_l92_92538


namespace find_value_l92_92007

variables {p q s u : ℚ}

theorem find_value
  (h1 : p / q = 5 / 6)
  (h2 : s / u = 7 / 15) :
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 :=
sorry

end find_value_l92_92007


namespace right_triangle_integral_sides_parity_l92_92165

theorem right_triangle_integral_sides_parity 
  (a b c : ℕ) 
  (h : a^2 + b^2 = c^2) 
  (ha : a % 2 = 1 ∨ a % 2 = 0) 
  (hb : b % 2 = 1 ∨ b % 2 = 0) 
  (hc : c % 2 = 1 ∨ c % 2 = 0) : 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) := 
sorry

end right_triangle_integral_sides_parity_l92_92165


namespace cube_volume_l92_92865

theorem cube_volume (d : ℝ) (h : d = 6 * Real.sqrt 2) : 
  ∃ v : ℝ, v = 48 * Real.sqrt 6 := by
  let s := d / Real.sqrt 3
  let volume := s ^ 3
  use volume
  /- Proof of the volume calculation is omitted. -/
  sorry

end cube_volume_l92_92865


namespace additional_coins_needed_l92_92883

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end additional_coins_needed_l92_92883


namespace multiply_by_12_correct_result_l92_92357

theorem multiply_by_12_correct_result (x : ℕ) (h : x / 14 = 42) : x * 12 = 7056 :=
by
  sorry

end multiply_by_12_correct_result_l92_92357


namespace dolly_dresses_shipment_l92_92040

variable (T : ℕ)

/-- Given that 70% of the total number of Dolly Dresses in the shipment is equal to 140,
    prove that the total number of Dolly Dresses in the shipment is 200. -/
theorem dolly_dresses_shipment (h : (7 * T) / 10 = 140) : T = 200 :=
sorry

end dolly_dresses_shipment_l92_92040


namespace find_m_l92_92162

-- Define the points M and N and the normal vector n
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def M (m : ℝ) : Point3D := { x := m, y := -2, z := 1 }
def N (m : ℝ) : Point3D := { x := 0, y := m, z := 3 }
def n : Point3D := { x := 3, y := 1, z := 2 }

-- Define the dot product
def dot_product (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

-- Define the vector MN
def MN (m : ℝ) : Point3D := { x := -(m), y := m + 2, z := 2 }

-- Prove the dot product condition is zero implies m = 3
theorem find_m (m : ℝ) (h : dot_product n (MN m) = 0) : m = 3 :=
by
  sorry

end find_m_l92_92162


namespace smallest_integer_solution_l92_92117

open Int

theorem smallest_integer_solution :
  ∃ x : ℤ, (⌊ (x : ℚ) / 8 ⌋ - ⌊ (x : ℚ) / 40 ⌋ + ⌊ (x : ℚ) / 240 ⌋ = 210) ∧ x = 2016 :=
by
  sorry

end smallest_integer_solution_l92_92117


namespace fraction_expression_equiv_l92_92335

theorem fraction_expression_equiv:
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := 
by 
  sorry

end fraction_expression_equiv_l92_92335
