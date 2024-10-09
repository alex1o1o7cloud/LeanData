import Mathlib

namespace pencils_purchased_l2080_208012

variable (P : ℕ)

theorem pencils_purchased (misplaced broke found bought left : ℕ) (h1 : misplaced = 7) (h2 : broke = 3) (h3 : found = 4) (h4 : bought = 2) (h5 : left = 16) :
  P - misplaced - broke + found + bought = left → P = 22 :=
by
  intros h
  have h_eq : P - 7 - 3 + 4 + 2 = 16 := by
    rw [h1, h2, h3, h4, h5] at h; exact h
  sorry

end pencils_purchased_l2080_208012


namespace find_a_if_f_is_odd_l2080_208078

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a_if_f_is_odd (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = - f a x) → a = 4 :=
by
  sorry

end find_a_if_f_is_odd_l2080_208078


namespace complement_of_A_l2080_208027

def A : Set ℝ := { x | x^2 - x ≥ 0 }
def R_complement_A : Set ℝ := { x | 0 < x ∧ x < 1 }

theorem complement_of_A :
  ∀ x : ℝ, x ∈ R_complement_A ↔ x ∉ A :=
sorry

end complement_of_A_l2080_208027


namespace complement_fraction_irreducible_l2080_208075

theorem complement_fraction_irreducible (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.gcd (b - a) b = 1 :=
sorry

end complement_fraction_irreducible_l2080_208075


namespace team_total_points_l2080_208038

theorem team_total_points : 
  ∀ (Tobee Jay Sean : ℕ),
  (Tobee = 4) →
  (Jay = Tobee + 6) →
  (Sean = Tobee + Jay - 2) →
  (Tobee + Jay + Sean = 26) :=
by
  intros Tobee Jay Sean h1 h2 h3
  rw [h1, h2, h3]
  sorry

end team_total_points_l2080_208038


namespace four_digit_num_exists_l2080_208036

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem four_digit_num_exists :
  ∃ (n : ℕ), (is_two_digit (n / 100)) ∧ (is_two_digit (n % 100)) ∧
  ((n / 100) + (n % 100))^2 = 100 * (n / 100) + (n % 100) :=
by
  sorry

end four_digit_num_exists_l2080_208036


namespace perimeter_of_square_with_area_36_l2080_208079

theorem perimeter_of_square_with_area_36 : 
  ∀ (A : ℝ), A = 36 → (∃ P : ℝ, P = 24 ∧ (∃ s : ℝ, s^2 = A ∧ P = 4 * s)) :=
by
  sorry

end perimeter_of_square_with_area_36_l2080_208079


namespace find_m_minus_n_l2080_208011

theorem find_m_minus_n (m n : ℤ) (h1 : |m| = 14) (h2 : |n| = 23) (h3 : m + n > 0) : m - n = -9 ∨ m - n = -37 := 
sorry

end find_m_minus_n_l2080_208011


namespace part1_part2_l2080_208023

noncomputable def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

theorem part1 (a : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → f x ≤ a) ↔ a ≥ 4 :=
sorry

theorem part2 : {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ -4 < x ∧ x < 1} :=
sorry

end part1_part2_l2080_208023


namespace red_or_blue_probability_is_half_l2080_208028

-- Define the number of each type of marble
def num_red_marbles : ℕ := 3
def num_blue_marbles : ℕ := 2
def num_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ := num_red_marbles + num_blue_marbles + num_yellow_marbles

-- Define the number of marbles that are either red or blue
def num_red_or_blue_marbles : ℕ := num_red_marbles + num_blue_marbles

-- Define the probability of drawing a red or blue marble
def probability_red_or_blue : ℚ := num_red_or_blue_marbles / total_marbles

-- Theorem stating the probability is 0.5
theorem red_or_blue_probability_is_half : probability_red_or_blue = 0.5 := by
  sorry

end red_or_blue_probability_is_half_l2080_208028


namespace gcd_38_23_is_1_l2080_208056

theorem gcd_38_23_is_1 : Nat.gcd 38 23 = 1 := by
  sorry

end gcd_38_23_is_1_l2080_208056


namespace probability_red_chips_drawn_first_l2080_208058

def probability_all_red_drawn (total_chips : Nat) (red_chips : Nat) (green_chips : Nat) : ℚ :=
  let total_arrangements := Nat.choose total_chips green_chips
  let favorable_arrangements := Nat.choose (total_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem probability_red_chips_drawn_first :
  probability_all_red_drawn 9 5 4 = 4 / 9 :=
by
  sorry

end probability_red_chips_drawn_first_l2080_208058


namespace non_neg_ints_less_than_pi_l2080_208082

-- Define the condition: non-negative integers with absolute value less than π
def condition (x : ℕ) : Prop := |(x : ℝ)| < Real.pi

-- Prove that the set satisfying the condition is {0, 1, 2, 3}
theorem non_neg_ints_less_than_pi :
  {x : ℕ | condition x} = {0, 1, 2, 3} := by
  sorry

end non_neg_ints_less_than_pi_l2080_208082


namespace total_and_average_games_l2080_208081

def football_games_per_month : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games_per_month : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games_per_month : List Nat := [17, 21, 14, 32, 22, 27]

def total_games (games_per_month : List Nat) : Nat :=
  List.sum games_per_month

def average_games (total : Nat) (months : Nat) : Nat :=
  total / months

theorem total_and_average_games :
  total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month = 486
  ∧ average_games (total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month) 6 = 81 :=
by
  sorry

end total_and_average_games_l2080_208081


namespace shaded_area_is_correct_l2080_208002

-- Definitions based on the conditions
def is_square (s : ℝ) (area : ℝ) : Prop := s * s = area
def rect_area (l w : ℝ) : ℝ := l * w

variables (s : ℝ) (area_s : ℝ) (rect1_l rect1_w rect2_l rect2_w : ℝ)

-- Given conditions
def square := is_square s area_s
def rect1 := rect_area rect1_l rect1_w
def rect2 := rect_area rect2_l rect2_w

-- Problem statement: Prove the area of the shaded region
theorem shaded_area_is_correct
  (s: ℝ)
  (rect1_l rect1_w rect2_l rect2_w : ℝ)
  (h_square: is_square s 16)
  (h_rect1: rect_area rect1_l rect1_w = 6)
  (h_rect2: rect_area rect2_l rect2_w = 2) :
  (16 - (6 + 2) = 8) := 
  sorry

end shaded_area_is_correct_l2080_208002


namespace scientific_notation_correct_l2080_208096

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l2080_208096


namespace demokhar_lifespan_l2080_208021

-- Definitions based on the conditions
def boy_fraction := 1 / 4
def young_man_fraction := 1 / 5
def adult_man_fraction := 1 / 3
def old_man_years := 13

-- Statement without proof
theorem demokhar_lifespan :
  ∀ (x : ℕ), (boy_fraction * x) + (young_man_fraction * x) + (adult_man_fraction * x) + old_man_years = x → x = 60 :=
by
  sorry

end demokhar_lifespan_l2080_208021


namespace circle_eq_of_hyperbola_focus_eccentricity_l2080_208043

theorem circle_eq_of_hyperbola_focus_eccentricity :
  ∀ (x y : ℝ), ((y^2 - (x^2 / 3) = 1) → (x^2 + (y-2)^2 = 4)) := by
  intro x y
  intro hyp_eq
  sorry

end circle_eq_of_hyperbola_focus_eccentricity_l2080_208043


namespace problem_l2080_208015

def count_numbers_with_more_ones_than_zeros (n : ℕ) : ℕ :=
  -- function that counts numbers less than or equal to 'n'
  -- whose binary representation has more '1's than '0's
  sorry

theorem problem (M := count_numbers_with_more_ones_than_zeros 1500) : 
  M % 1000 = 884 :=
sorry

end problem_l2080_208015


namespace number_of_boys_l2080_208024

theorem number_of_boys (n : ℕ) (handshakes : ℕ) (h_handshakes : handshakes = n * (n - 1) / 2) (h_total : handshakes = 55) : n = 11 := by
  sorry

end number_of_boys_l2080_208024


namespace min_moves_to_reassemble_l2080_208029

theorem min_moves_to_reassemble (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, (∀ pieces, pieces = n - 1) ∧ pieces = 1 → move_count = n - 1 :=
by
  sorry

end min_moves_to_reassemble_l2080_208029


namespace sum_of_circle_areas_l2080_208014

theorem sum_of_circle_areas (a b c: ℝ)
  (h1: a + b = 6)
  (h2: b + c = 8)
  (h3: a + c = 10) :
  π * a^2 + π * b^2 + π * c^2 = 56 * π := 
by
  sorry

end sum_of_circle_areas_l2080_208014


namespace smallest_n_gcd_l2080_208005

theorem smallest_n_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 2) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 2) > 1 → m ≥ n) ↔ n = 19 :=
by
  sorry

end smallest_n_gcd_l2080_208005


namespace solution1_solution2_l2080_208017

-- Define the first problem
def equation1 (x : ℝ) : Prop :=
  (x + 1) / 3 - 1 = (x - 1) / 2

-- Prove that x = -1 is the solution to the first problem
theorem solution1 : equation1 (-1) := 
by 
  sorry

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  x - y = 1 ∧ 3 * x + y = 7

-- Prove that x = 2 and y = 1 are the solutions to the system of equations
theorem solution2 : system_of_equations 2 1 :=
by 
  sorry

end solution1_solution2_l2080_208017


namespace prime_quadruples_l2080_208098

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_quadruples {p₁ p₂ p₃ p₄ : ℕ} (prime_p₁ : is_prime p₁) (prime_p₂ : is_prime p₂) (prime_p₃ : is_prime p₃) (prime_p₄ : is_prime p₄)
  (h1 : p₁ < p₂) (h2 : p₂ < p₃) (h3 : p₃ < p₄) (eq_condition : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  (p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
  (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
  (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29) :=
sorry

end prime_quadruples_l2080_208098


namespace total_fence_poles_needed_l2080_208030

def number_of_poles_per_side := 27

theorem total_fence_poles_needed (n : ℕ) (h : n = number_of_poles_per_side) : 
  4 * n - 4 = 104 :=
by sorry

end total_fence_poles_needed_l2080_208030


namespace technician_round_trip_percentage_l2080_208050

theorem technician_round_trip_percentage
  (D : ℝ) 
  (H1 : D > 0) -- Assume D is positive
  (H2 : true) -- The technician completes the drive to the center
  (H3 : true) -- The technician completes 20% of the drive from the center
  : (1.20 * D / (2 * D)) * 100 = 60 := 
by
  simp [H1, H2, H3]
  sorry

end technician_round_trip_percentage_l2080_208050


namespace sin_identity_l2080_208097

variable (α : ℝ)
axiom alpha_def : α = Real.pi / 7

theorem sin_identity : (Real.sin (3 * α)) ^ 2 - (Real.sin α) ^ 2 = Real.sin (2 * α) * Real.sin (3 * α) := 
by 
  sorry

end sin_identity_l2080_208097


namespace movie_ticket_percentage_decrease_l2080_208055

theorem movie_ticket_percentage_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100) 
  (h2 : new_price = 80) :
  ((old_price - new_price) / old_price) * 100 = 20 := 
by
  sorry

end movie_ticket_percentage_decrease_l2080_208055


namespace rationalize_denominator_l2080_208091

theorem rationalize_denominator :
  (3 : ℝ) / Real.sqrt 48 = Real.sqrt 3 / 4 :=
by
  sorry

end rationalize_denominator_l2080_208091


namespace justin_current_age_l2080_208020

theorem justin_current_age
  (angelina_older : ∀ (j a : ℕ), a = j + 4)
  (angelina_future_age : ∀ (a : ℕ), a + 5 = 40) :
  ∃ (justin_current_age : ℕ), justin_current_age = 31 := 
by
  sorry

end justin_current_age_l2080_208020


namespace min_correct_answers_l2080_208073

theorem min_correct_answers (x : ℕ) (hx : 10 * x - 5 * (30 - x) > 90) : x ≥ 17 :=
by {
  -- calculations and solution steps go here.
  sorry
}

end min_correct_answers_l2080_208073


namespace calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l2080_208083

-- Define the necessary probability events and conditions.
variable {p : ℝ} (calc_action : ℕ → ℝ)

-- Condition: initially, the display shows 0.
def initial_display : ℕ := 0

-- Events for part (a): addition only, randomly chosen numbers from 0 to 9.
def random_addition_event (n : ℕ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Events for part (b): both addition and multiplication allowed.
def random_operation_event (n : ℕ) : Prop := (n % 2 = 0 ∧ n % 2 = 1) ∨ -- addition
                                               (n ≠ 0 ∧ n % 2 = 1 ∧ (n/2) % 2 = 1) -- multiplication

-- Statements to be proved based on above definitions.
theorem calc_addition_even_odd_probability :
  calc_action 0 = 1 / 2 → random_addition_event initial_display := sorry

theorem calc_addition_multiplication_even_probability :
  calc_action (initial_display + 1) > 1 / 2 → random_operation_event (initial_display + 1) := sorry

end calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l2080_208083


namespace cost_price_one_meter_l2080_208037

theorem cost_price_one_meter (selling_price : ℤ) (total_meters : ℤ) (profit_per_meter : ℤ) 
  (h1 : selling_price = 6788) (h2 : total_meters = 78) (h3 : profit_per_meter = 29) : 
  (selling_price - (profit_per_meter * total_meters)) / total_meters = 58 := 
by 
  sorry

end cost_price_one_meter_l2080_208037


namespace solve_x_plus_y_l2080_208067

variable {x y : ℚ} -- Declare x and y as rational numbers

theorem solve_x_plus_y
  (h1: (1 / x) + (1 / y) = 1)
  (h2: (1 / x) - (1 / y) = 5) :
  x + y = -1 / 6 :=
sorry

end solve_x_plus_y_l2080_208067


namespace find_x_l2080_208064

def f (x: ℝ) : ℝ := 3 * x - 5

theorem find_x (x : ℝ) (h : 2 * (f x) - 10 = f (x - 2)) : x = 3 :=
by
  sorry

end find_x_l2080_208064


namespace limit_sum_perimeters_areas_of_isosceles_triangles_l2080_208000

theorem limit_sum_perimeters_areas_of_isosceles_triangles (b s h : ℝ) : 
  ∃ P A : ℝ, 
    (P = 2*(b + 2*s)) ∧ 
    (A = (2/3)*b*h) :=
  sorry

end limit_sum_perimeters_areas_of_isosceles_triangles_l2080_208000


namespace math_club_team_selection_l2080_208022

open scoped BigOperators

def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem math_club_team_selection :
  (comb 7 2 * comb 9 4) + 
  (comb 7 3 * comb 9 3) +
  (comb 7 4 * comb 9 2) +
  (comb 7 5 * comb 9 1) +
  (comb 7 6 * comb 9 0) = 7042 := 
sorry

end math_club_team_selection_l2080_208022


namespace find_side_c_l2080_208099

theorem find_side_c (a C S : ℝ) (ha : a = 3) (hC : C = 120) (hS : S = (15 * Real.sqrt 3) / 4) : 
  ∃ (c : ℝ), c = 7 :=
by
  sorry

end find_side_c_l2080_208099


namespace factorize_polynomial_l2080_208057

theorem factorize_polynomial (m : ℤ) : 4 * m^2 - 16 = 4 * (m + 2) * (m - 2) := by
  sorry

end factorize_polynomial_l2080_208057


namespace football_games_this_year_l2080_208094

theorem football_games_this_year 
  (total_games : ℕ) 
  (games_last_year : ℕ) 
  (games_this_year : ℕ) 
  (h1 : total_games = 9) 
  (h2 : games_last_year = 5) 
  (h3 : total_games = games_last_year + games_this_year) : 
  games_this_year = 4 := 
sorry

end football_games_this_year_l2080_208094


namespace quadratic_root_range_specific_m_value_l2080_208095

theorem quadratic_root_range (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1^2 - 2 * (1 - m) * x1 + m^2 = 0 ∧ x2^2 - 2 * (1 - m) * x2 + m^2 = 0 ↔ m ≤ 1/2 :=
by
  sorry

theorem specific_m_value (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2 * (1 - m) * x1 + m^2 = 0)
  (h2 : x2^2 - 2 * (1 - m) * x2 + m^2 = 0) (h3 : x1^2 + 12 * m + x2^2 = 10) : 
  m = -3 :=
by
  sorry

end quadratic_root_range_specific_m_value_l2080_208095


namespace symmetric_point_of_A_l2080_208092

theorem symmetric_point_of_A (a b : ℝ) 
  (h1 : 2 * a - 4 * b + 9 = 0) 
  (h2 : ∃ t : ℝ, (a, b) = (1 - 4 * t, 4 + 2 * t)) : 
  (a, b) = (1, 4) :=
sorry

end symmetric_point_of_A_l2080_208092


namespace difference_highest_lowest_score_l2080_208084

-- Definitions based on conditions
def total_innings : ℕ := 46
def avg_innings : ℕ := 61
def highest_score : ℕ := 202
def avg_excl_highest_lowest : ℕ := 58
def innings_excl_highest_lowest : ℕ := 44

-- Calculated total runs
def total_runs : ℕ := total_innings * avg_innings
def total_runs_excl_highest_lowest : ℕ := innings_excl_highest_lowest * avg_excl_highest_lowest
def sum_of_highest_lowest : ℕ := total_runs - total_runs_excl_highest_lowest
def lowest_score : ℕ := sum_of_highest_lowest - highest_score

theorem difference_highest_lowest_score 
  (h1: total_runs = total_innings * avg_innings)
  (h2: avg_excl_highest_lowest * innings_excl_highest_lowest = total_runs_excl_highest_lowest)
  (h3: sum_of_highest_lowest = total_runs - total_runs_excl_highest_lowest)
  (h4: highest_score = 202)
  (h5: lowest_score = sum_of_highest_lowest - highest_score)
  : highest_score - lowest_score = 150 :=
by
  -- We only need to state the theorem, so we can skip the proof.
  -- The exact statements of conditions and calculations imply the result.
  sorry

end difference_highest_lowest_score_l2080_208084


namespace cone_base_diameter_l2080_208086

theorem cone_base_diameter
  (h_cone : ℝ) (r_sphere : ℝ) (waste_percentage : ℝ) (d : ℝ) :
  h_cone = 9 → r_sphere = 9 → waste_percentage = 0.75 → 
  (V_cone = 1/3 * π * (d/2)^2 * h_cone) →
  (V_sphere = 4/3 * π * r_sphere^3) →
  (V_cone = (1 - waste_percentage) * V_sphere) →
  d = 9 :=
by
  intros h_cond r_cond waste_cond v_cone_eq v_sphere_eq v_cone_sphere_eq
  sorry

end cone_base_diameter_l2080_208086


namespace cyclist_wait_time_l2080_208001

theorem cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ) (catch_up_time : ℝ) 
  (hiker_speed_eq : hiker_speed = 4) 
  (cyclist_speed_eq : cyclist_speed = 12) 
  (wait_time_eq : wait_time = 5 / 60) 
  (catch_up_time_eq : catch_up_time = (2 / 3) / (1 / 15)) 
  : catch_up_time * 60 = 10 := 
by 
  sorry

end cyclist_wait_time_l2080_208001


namespace calculate_expression_l2080_208040

theorem calculate_expression : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end calculate_expression_l2080_208040


namespace sum_of_babies_ages_in_five_years_l2080_208066

-- Given Definitions
def lioness_age := 12
def hyena_age := lioness_age / 2
def lioness_baby_age := lioness_age / 2
def hyena_baby_age := hyena_age / 2

-- The declaration of the statement to be proven
theorem sum_of_babies_ages_in_five_years : (lioness_baby_age + 5) + (hyena_baby_age + 5) = 19 :=
by 
  sorry 

end sum_of_babies_ages_in_five_years_l2080_208066


namespace smallest_class_size_l2080_208063

theorem smallest_class_size (N : ℕ) (G : ℕ) (h1: 0.25 < (G : ℝ) / N) (h2: (G : ℝ) / N < 0.30) : N = 7 := 
sorry

end smallest_class_size_l2080_208063


namespace sum_of_cubes_l2080_208042

theorem sum_of_cubes (p q r : ℝ) (h1 : p + q + r = 7) (h2 : p * q + p * r + q * r = 10) (h3 : p * q * r = -20) :
  p^3 + q^3 + r^3 = 181 :=
by
  sorry

end sum_of_cubes_l2080_208042


namespace find_q_l2080_208035

theorem find_q (p q : ℝ) (p_gt : p > 1) (q_gt : q > 1) (h1 : 1/p + 1/q = 1) (h2 : p*q = 9) :
  q = (9 + 3 * Real.sqrt 5) / 2 :=
by sorry

end find_q_l2080_208035


namespace solve_equation_l2080_208053

theorem solve_equation :
  ∀ x : ℝ, (-x^2 = (2*x + 4) / (x + 2)) ↔ (x = -2 ∨ x = -1) :=
by
  intro x
  -- the proof steps would go here
  sorry

end solve_equation_l2080_208053


namespace max_value_x_plus_y_l2080_208069

theorem max_value_x_plus_y :
  ∃ x y : ℝ, 5 * x + 3 * y ≤ 10 ∧ 3 * x + 5 * y = 15 ∧ x + y = 47 / 16 :=
by
  sorry

end max_value_x_plus_y_l2080_208069


namespace polynomial_divisor_l2080_208051

theorem polynomial_divisor (f : Polynomial ℂ) (n : ℕ) (h : (X - 1) ∣ (f.comp (X ^ n))) : (X ^ n - 1) ∣ (f.comp (X ^ n)) :=
sorry

end polynomial_divisor_l2080_208051


namespace number_of_eggplant_packets_l2080_208018

-- Defining the problem conditions in Lean 4
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def sunflower_packets := 6
def total_plants := 116

-- Our goal is to prove the number of eggplant seed packets Shyne bought
theorem number_of_eggplant_packets : ∃ E : ℕ, E * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants ∧ E = 4 :=
sorry

end number_of_eggplant_packets_l2080_208018


namespace calc_num_articles_l2080_208085

-- Definitions based on the conditions
def cost_price (C : ℝ) : ℝ := C
def selling_price (C : ℝ) : ℝ := 1.10000000000000004 * C
def num_articles (n : ℝ) (C : ℝ) (S : ℝ) : Prop := 55 * C = n * S

-- Proof Statement
theorem calc_num_articles (C : ℝ) : ∃ n : ℝ, num_articles n C (selling_price C) ∧ n = 50 :=
by sorry

end calc_num_articles_l2080_208085


namespace inequality_solution_l2080_208039

theorem inequality_solution (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 :=
by 
  sorry

end inequality_solution_l2080_208039


namespace ellipse_eccentricity_range_l2080_208068

theorem ellipse_eccentricity_range (a b : ℝ) (h : a > b) (h_b : b > 0) : 
  ∃ e : ℝ, (e = (Real.sqrt (a^2 - b^2)) / a) ∧ (e > 1/2 ∧ e < 1) :=
by
  sorry

end ellipse_eccentricity_range_l2080_208068


namespace find_b_l2080_208004

def h (x : ℝ) : ℝ := 4 * x - 5

theorem find_b (b : ℝ) (h_b : h b = 1) : b = 3 / 2 :=
by
  sorry

end find_b_l2080_208004


namespace diamond_and_face_card_probability_l2080_208070

noncomputable def probability_first_diamond_second_face_card : ℚ :=
  let total_cards := 52
  let total_faces := 12
  let diamond_faces := 3
  let diamond_non_faces := 10
  (9/52) * (12/51) + (3/52) * (11/51)

theorem diamond_and_face_card_probability :
  probability_first_diamond_second_face_card = 47 / 884 := 
by {
  sorry
}

end diamond_and_face_card_probability_l2080_208070


namespace largest_valid_four_digit_number_l2080_208090

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l2080_208090


namespace quadratic_root_square_of_another_l2080_208062

theorem quadratic_root_square_of_another (a : ℚ) :
  (∃ x y : ℚ, x^2 - (15/4) * x + a^3 = 0 ∧ (x = y^2 ∨ y = x^2) ∧ (x*y = a^3)) →
  (a = 3/2 ∨ a = -5/2) :=
sorry

end quadratic_root_square_of_another_l2080_208062


namespace complex_expression_evaluation_l2080_208061

-- Conditions
def i : ℂ := Complex.I -- Representing the imaginary unit i

-- Defining the inverse of a complex number
noncomputable def complex_inv (z : ℂ) := 1 / z

-- Proof statement
theorem complex_expression_evaluation :
  (i - complex_inv i + 3)⁻¹ = (3 - 2 * i) / 13 := by
sorry

end complex_expression_evaluation_l2080_208061


namespace min_sum_distances_to_corners_of_rectangle_center_l2080_208088

theorem min_sum_distances_to_corners_of_rectangle_center (P A B C D : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 0))
  (hC : C = (1, 1))
  (hD : D = (0, 1))
  (hP_center : P = (0.5, 0.5)) :
  ∀ Q, (dist Q A + dist Q B + dist Q C + dist Q D) ≥ (dist P A + dist P B + dist P C + dist P D) := 
sorry

end min_sum_distances_to_corners_of_rectangle_center_l2080_208088


namespace weight_of_tin_of_cookies_l2080_208074

def weight_of_bag_of_chips := 20 -- in ounces
def weight_jasmine_carries := 336 -- converting 21 pounds to ounces
def bags_jasmine_buys := 6
def tins_multiplier := 4

theorem weight_of_tin_of_cookies 
  (weight_of_bag_of_chips : ℕ := weight_of_bag_of_chips)
  (weight_jasmine_carries : ℕ := weight_jasmine_carries)
  (bags_jasmine_buys : ℕ := bags_jasmine_buys)
  (tins_multiplier : ℕ := tins_multiplier) : 
  ℕ :=
  let total_weight_bags := bags_jasmine_buys * weight_of_bag_of_chips
  let total_weight_cookies := weight_jasmine_carries - total_weight_bags
  let num_of_tins := bags_jasmine_buys * tins_multiplier
  total_weight_cookies / num_of_tins

example : weight_of_tin_of_cookies weight_of_bag_of_chips weight_jasmine_carries bags_jasmine_buys tins_multiplier = 9 :=
by sorry

end weight_of_tin_of_cookies_l2080_208074


namespace spencer_total_distance_l2080_208003

-- Define the individual segments of Spencer's travel
def walk1 : ℝ := 1.2
def bike1 : ℝ := 1.8
def bus1 : ℝ := 3
def walk2 : ℝ := 0.4
def walk3 : ℝ := 0.6
def bike2 : ℝ := 2
def walk4 : ℝ := 1.5

-- Define the conversion factors
def bike_to_walk_conversion : ℝ := 0.5
def bus_to_walk_conversion : ℝ := 0.8

-- Calculate the total walking distance
def total_walking_distance : ℝ := walk1 + walk2 + walk3 + walk4

-- Calculate the total biking distance as walking equivalent
def total_biking_distance_as_walking : ℝ := (bike1 + bike2) * bike_to_walk_conversion

-- Calculate the total bus distance as walking equivalent
def total_bus_distance_as_walking : ℝ := bus1 * bus_to_walk_conversion

-- Define the total walking equivalent distance
def total_distance : ℝ := total_walking_distance + total_biking_distance_as_walking + total_bus_distance_as_walking

-- Theorem stating the total distance covered is 8 miles
theorem spencer_total_distance : total_distance = 8 := by
  unfold total_distance
  unfold total_walking_distance
  unfold total_biking_distance_as_walking
  unfold total_bus_distance_as_walking
  norm_num
  sorry

end spencer_total_distance_l2080_208003


namespace scout_earnings_weekend_l2080_208093

-- Define the conditions
def base_pay_per_hour : ℝ := 10.00
def saturday_hours : ℝ := 6
def saturday_customers : ℝ := 5
def saturday_tip_per_customer : ℝ := 5.00
def sunday_hours : ℝ := 8
def sunday_customers_with_3_tip : ℝ := 5
def sunday_customers_with_7_tip : ℝ := 5
def sunday_tip_3_per_customer : ℝ := 3.00
def sunday_tip_7_per_customer : ℝ := 7.00
def overtime_multiplier : ℝ := 1.5

-- Statement to prove earnings for the weekend is $255.00
theorem scout_earnings_weekend : 
  (base_pay_per_hour * saturday_hours + saturday_customers * saturday_tip_per_customer) +
  (base_pay_per_hour * overtime_multiplier * sunday_hours + 
   sunday_customers_with_3_tip * sunday_tip_3_per_customer +
   sunday_customers_with_7_tip * sunday_tip_7_per_customer) = 255 :=
by
  sorry

end scout_earnings_weekend_l2080_208093


namespace ed_more_marbles_l2080_208006

-- Define variables for initial number of marbles
variables {E D : ℕ}

-- Ed had some more marbles than Doug initially.
-- Doug lost 8 of his marbles at the playground.
-- Now Ed has 30 more marbles than Doug.
theorem ed_more_marbles (h : E = (D - 8) + 30) : E - D = 22 :=
by
  sorry

end ed_more_marbles_l2080_208006


namespace kyle_money_left_l2080_208072

-- Define variables and conditions
variables (d k : ℕ)
variables (has_kyle : k = 3 * d - 12) (has_dave : d = 46)

-- State the theorem to prove 
theorem kyle_money_left (d k : ℕ) (has_kyle : k = 3 * d - 12) (has_dave : d = 46) :
  k - k / 3 = 84 :=
by
  -- Sorry to complete the proof block
  sorry

end kyle_money_left_l2080_208072


namespace division_of_field_l2080_208065

theorem division_of_field :
  (∀ (hectares : ℕ) (parts : ℕ), hectares = 5 ∧ parts = 8 →
  (1 / parts = 1 / 8) ∧ (hectares / parts = 5 / 8)) :=
by
  sorry


end division_of_field_l2080_208065


namespace fraction_meaningful_if_not_neg_two_l2080_208016

theorem fraction_meaningful_if_not_neg_two {a : ℝ} : (a + 2 ≠ 0) ↔ (a ≠ -2) :=
by sorry

end fraction_meaningful_if_not_neg_two_l2080_208016


namespace problem_statement_l2080_208077

theorem problem_statement (a : ℤ)
  (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a) ^ 2 + (2004 - a) ^ 2 = 4014 :=
sorry

end problem_statement_l2080_208077


namespace not_p_and_q_equiv_not_p_or_not_q_l2080_208044

variable (p q : Prop)

theorem not_p_and_q_equiv_not_p_or_not_q (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end not_p_and_q_equiv_not_p_or_not_q_l2080_208044


namespace factor_expression_l2080_208045

variable (x : ℝ)

-- Mathematically define the expression e
def e : ℝ := 4 * x * (x + 2) + 10 * (x + 2) + 2 * (x + 2)

-- State that e is equivalent to the factored form
theorem factor_expression : e x = (x + 2) * (4 * x + 12) :=
by
  sorry

end factor_expression_l2080_208045


namespace measure_of_angle_C_l2080_208046

theorem measure_of_angle_C
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := 
sorry

end measure_of_angle_C_l2080_208046


namespace cyclist_downhill_speed_l2080_208013

noncomputable def downhill_speed (d uphill_speed avg_speed : ℝ) : ℝ :=
  let downhill_speed := (2 * d * uphill_speed) / (avg_speed * d - uphill_speed * 2)
  -- We want to prove
  downhill_speed

theorem cyclist_downhill_speed :
  downhill_speed 150 25 35 = 58.33 :=
by
  -- Proof omitted
  sorry

end cyclist_downhill_speed_l2080_208013


namespace proof_problem_l2080_208048

theorem proof_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c * d = 1) : 
  (a * c + b * d) * (a * d + b * c) ≥ 1 := 
by 
  sorry

end proof_problem_l2080_208048


namespace evaluate_expression_l2080_208019

-- Define the given numbers as real numbers
def x : ℝ := 175.56
def y : ℝ := 54321
def z : ℝ := 36947
def w : ℝ := 1521

-- State the theorem to be proved
theorem evaluate_expression : (x / y) * (z / w) = 0.07845 :=
by 
  -- We skip the proof here
  sorry

end evaluate_expression_l2080_208019


namespace solve_fractional_equation_l2080_208049

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ -1) :
  (2 / x = 3 / (x + 1)) → (x = 2) :=
by
  -- Proof will be filled in here
  sorry

end solve_fractional_equation_l2080_208049


namespace inequality_proof_l2080_208080

theorem inequality_proof (x y : ℝ) (h : |x - 2 * y| = 5) : x^2 + y^2 ≥ 5 := 
  sorry

end inequality_proof_l2080_208080


namespace finite_operations_invariant_final_set_l2080_208059

theorem finite_operations (n : ℕ) (a : Fin n → ℕ) :
  ∃ N : ℕ, ∀ k, k > N → ((∃ i j, i ≠ j ∧ ¬ (a i ∣ a j ∨ a j ∣ a i)) → False) :=
sorry

theorem invariant_final_set (n : ℕ) (a : Fin n → ℕ) :
  ∃ b : Fin n → ℕ, (∀ i, ∃ j, b i = a j) ∧ ∀ (c : Fin n → ℕ), (∀ i, ∃ j, c i = a j) → c = b :=
sorry

end finite_operations_invariant_final_set_l2080_208059


namespace find_N_l2080_208032

theorem find_N (x y : ℕ) (N : ℕ) (h1 : N = x * (x + 9)) (h2 : N = y * (y + 6)) : 
  N = 112 :=
  sorry

end find_N_l2080_208032


namespace valid_three_digit_numbers_count_l2080_208033

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end valid_three_digit_numbers_count_l2080_208033


namespace francie_has_3_dollars_remaining_l2080_208089

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l2080_208089


namespace desks_per_row_calc_l2080_208007

theorem desks_per_row_calc :
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  (total_desks / 4 = 6) :=
by
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  show total_desks / 4 = 6
  sorry

end desks_per_row_calc_l2080_208007


namespace symmetric_points_on_ellipse_are_m_in_range_l2080_208071

open Real

theorem symmetric_points_on_ellipse_are_m_in_range (m : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 ^ 2) / 4 + (A.2 ^ 2) / 3 = 1 ∧ 
                   (B.1 ^ 2) / 4 + (B.2 ^ 2) / 3 = 1 ∧ 
                   ∃ x0 y0 : ℝ, y0 = 4 * x0 + m ∧ x0 = (A.1 + B.1) / 2 ∧ y0 = (A.2 + B.2) / 2) 
  ↔ -2 * sqrt 13 / 13 < m ∧ m < 2 * sqrt 13 / 13 := 
 sorry

end symmetric_points_on_ellipse_are_m_in_range_l2080_208071


namespace spent_on_board_game_l2080_208047

theorem spent_on_board_game (b : ℕ)
  (h1 : 4 * 7 = 28)
  (h2 : b + 28 = 30) : 
  b = 2 := 
sorry

end spent_on_board_game_l2080_208047


namespace initial_money_amount_l2080_208041

theorem initial_money_amount (M : ℝ)
  (h_clothes : M * (1 / 3) = c)
  (h_food : (M - c) * (1 / 5) = f)
  (h_travel : (M - c - f) * (1 / 4) = t)
  (h_remaining : M - c - f - t = 600) : M = 1500 := by
  sorry

end initial_money_amount_l2080_208041


namespace expression_evaluation_l2080_208087

-- Using the given conditions
def a : ℕ := 3
def b : ℕ := a^2 + 2 * a + 5
def c : ℕ := b^2 - 14 * b + 45

-- We need to assume that none of the denominators are zero.
lemma non_zero_denominators : (a + 1 ≠ 0) ∧ (b - 3 ≠ 0) ∧ (c + 7 ≠ 0) :=
  by {
    -- Proof goes here
  sorry }

theorem expression_evaluation :
  (a = 3) →
  ((a^2 + 2*a + 5) = b) →
  ((b^2 - 14*b + 45) = c) →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (↑(a + 3) / ↑(a + 1) * ↑(b - 1) / ↑(b - 3) * ↑(c + 9) / ↑(c + 7) = 4923 / 2924) :=
  by {
    -- Proof goes here
  sorry }

end expression_evaluation_l2080_208087


namespace solution_set_of_inequality_l2080_208034

theorem solution_set_of_inequality (a b x : ℝ) (h1 : 0 < a) (h2 : b = 2 * a) : ax > b ↔ x > -2 :=
by sorry

end solution_set_of_inequality_l2080_208034


namespace alice_age_multiple_sum_l2080_208010

theorem alice_age_multiple_sum (B : ℕ) (C : ℕ := 3) (A : ℕ := B + 2) (next_multiple_age : ℕ := A + (3 - (A % 3))) :
  B % C = 0 ∧ A = B + 2 ∧ C = 3 → 
  (next_multiple_age % 3 = 0 ∧
   (next_multiple_age / 10) + (next_multiple_age % 10) = 6) := 
by
  intros h
  sorry

end alice_age_multiple_sum_l2080_208010


namespace total_earnings_l2080_208009

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l2080_208009


namespace roger_earned_correct_amount_l2080_208025

def small_lawn_rate : ℕ := 9
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def initial_small_lawns : ℕ := 5
def initial_medium_lawns : ℕ := 4
def initial_large_lawns : ℕ := 5

def forgot_small_lawns : ℕ := 2
def forgot_medium_lawns : ℕ := 3
def forgot_large_lawns : ℕ := 3

def actual_small_lawns := initial_small_lawns - forgot_small_lawns
def actual_medium_lawns := initial_medium_lawns - forgot_medium_lawns
def actual_large_lawns := initial_large_lawns - forgot_large_lawns

def money_earned_small := actual_small_lawns * small_lawn_rate
def money_earned_medium := actual_medium_lawns * medium_lawn_rate
def money_earned_large := actual_large_lawns * large_lawn_rate

def total_money_earned := money_earned_small + money_earned_medium + money_earned_large

theorem roger_earned_correct_amount : total_money_earned = 69 := by
  sorry

end roger_earned_correct_amount_l2080_208025


namespace triangle_side_length_l2080_208031

theorem triangle_side_length (a b c : ℝ) (B : ℝ) (ha : a = 2) (hB : B = 60) (hc : c = 3) :
  b = Real.sqrt 7 :=
by
  sorry

end triangle_side_length_l2080_208031


namespace problem1_problem2_l2080_208060

-- Define the total number of balls for clarity
def total_red_balls : ℕ := 4
def total_white_balls : ℕ := 6
def total_balls_drawn : ℕ := 4

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := n.choose k

-- Problem 1: Prove that the number of ways to draw 4 balls that include both colors is 194
theorem problem1 :
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) +
  (binom total_red_balls 1 * binom total_white_balls 3) = 194 :=
  sorry

-- Problem 2: Prove that the number of ways to draw 4 balls where the number of red balls is at least the number of white balls is 115
theorem problem2 :
  (binom total_red_balls 4 * binom total_white_balls 0) +
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) = 115 :=
  sorry

end problem1_problem2_l2080_208060


namespace ab_value_l2080_208076

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5 / 8) : ab = (Real.sqrt 3) / 4 :=
by
  sorry

end ab_value_l2080_208076


namespace sub_numbers_correct_l2080_208026

theorem sub_numbers_correct : 
  (500.50 - 123.45 - 55 : ℝ) = 322.05 := by 
-- The proof can be filled in here
sorry

end sub_numbers_correct_l2080_208026


namespace find_triples_l2080_208008

-- Define the conditions in Lean 4
def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the math proof problem
theorem find_triples (m n p : ℕ) (hp : is_prime p) (hm : is_positive_integer m) (hn : is_positive_integer n) : 
  p^n + 3600 = m^2 ↔ (m = 61 ∧ n = 2 ∧ p = 11) ∨ (m = 65 ∧ n = 4 ∧ p = 5) ∨ (m = 68 ∧ n = 10 ∧ p = 2) :=
by
  sorry

end find_triples_l2080_208008


namespace inequality_proof_l2080_208054

variable (a b c d : ℝ)

theorem inequality_proof
  (h_pos: 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1)
  (h_product: a * b * c * d = (1 - a) * (1 - b) * (1 - c) * (1 - d)) : 
  (a + b + c + d) - (a + c) * (b + d) ≥ 1 :=
by
  sorry

end inequality_proof_l2080_208054


namespace longest_tape_l2080_208052

theorem longest_tape (r b y : ℚ) (h₀ : r = 11 / 6) (h₁ : b = 7 / 4) (h₂ : y = 13 / 8) : r > b ∧ r > y :=
by 
  sorry

end longest_tape_l2080_208052
