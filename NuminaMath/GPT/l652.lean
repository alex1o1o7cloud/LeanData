import Mathlib

namespace NUMINAMATH_GPT_polynomial_pair_solution_l652_65275

-- We define the problem in terms of polynomials over real numbers
open Polynomial

theorem polynomial_pair_solution (P Q : ℝ[X]) :
  (∀ x y : ℝ, P.eval (x + Q.eval y) = Q.eval (x + P.eval y)) →
  (P = Q ∨ (∃ a b : ℝ, P = X + C a ∧ Q = X + C b)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_pair_solution_l652_65275


namespace NUMINAMATH_GPT_probability_event_proof_l652_65282

noncomputable def probability_event_occur (deck_size : ℕ) (num_queens : ℕ) (num_jacks : ℕ) (num_reds : ℕ) : ℚ :=
  let prob_two_queens := (num_queens / deck_size) * ((num_queens - 1) / (deck_size - 1))
  let prob_at_least_one_jack := 
    (num_jacks / deck_size) * ((deck_size - num_jacks) / (deck_size - 1)) +
    ((deck_size - num_jacks) / deck_size) * (num_jacks / (deck_size - 1)) +
    (num_jacks / deck_size) * ((num_jacks - 1) / (deck_size - 1))
  let prob_both_red := (num_reds / deck_size) * ((num_reds - 1) / (deck_size - 1))
  prob_two_queens + prob_at_least_one_jack + prob_both_red

theorem probability_event_proof :
  probability_event_occur 52 4 4 26 = 89 / 221 :=
by
  sorry

end NUMINAMATH_GPT_probability_event_proof_l652_65282


namespace NUMINAMATH_GPT_num_O_atoms_l652_65215

def compound_molecular_weight : ℕ := 62
def atomic_weight_H : ℕ := 1
def atomic_weight_C : ℕ := 12
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_C_atoms : ℕ := 1

theorem num_O_atoms (H_weight : ℕ := num_H_atoms * atomic_weight_H)
                    (C_weight : ℕ := num_C_atoms * atomic_weight_C)
                    (total_weight : ℕ := compound_molecular_weight)
                    (O_weight := atomic_weight_O) : 
    (total_weight - (H_weight + C_weight)) / O_weight = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_O_atoms_l652_65215


namespace NUMINAMATH_GPT_sum_abcd_l652_65239

variables (a b c d : ℚ)

theorem sum_abcd :
  3 * a + 4 * b + 6 * c + 8 * d = 48 →
  4 * (d + c) = b →
  4 * b + 2 * c = a →
  c + 1 = d →
  a + b + c + d = 513 / 37 :=
by
sorry

end NUMINAMATH_GPT_sum_abcd_l652_65239


namespace NUMINAMATH_GPT_ratio_of_areas_l652_65218

theorem ratio_of_areas (s : ℝ) (h1 : s > 0) : 
  let small_square_area := s^2
  let total_small_squares_area := 4 * s^2
  let large_square_side_length := 4 * s
  let large_square_area := (4 * s)^2
  total_small_squares_area / large_square_area = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l652_65218


namespace NUMINAMATH_GPT_pages_copied_for_15_dollars_l652_65245

theorem pages_copied_for_15_dollars
  (cost_per_page : ℕ)
  (dollar_to_cents : ℕ)
  (dollars_available : ℕ)
  (convert_to_cents : dollar_to_cents = 100)
  (cost_per_page_eq : cost_per_page = 3)
  (dollars_available_eq : dollars_available = 15) :
  (dollars_available * dollar_to_cents) / cost_per_page = 500 := by
  -- Convert the dollar amount to cents
  -- Calculate the number of pages that can be copied
  sorry

end NUMINAMATH_GPT_pages_copied_for_15_dollars_l652_65245


namespace NUMINAMATH_GPT_inequality_solution_l652_65256

theorem inequality_solution (x : ℝ) :
  ((2 / (x - 1)) - (3 / (x - 3)) + (2 / (x - 4)) - (2 / (x - 5)) < (1 / 15)) ↔
  (x < -1 ∨ (1 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (7 < x ∧ x < 8)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l652_65256


namespace NUMINAMATH_GPT_car_rental_cost_eq_800_l652_65291

-- Define the number of people
def num_people : ℕ := 8

-- Define the cost of the Airbnb rental
def airbnb_cost : ℕ := 3200

-- Define each person's share
def share_per_person : ℕ := 500

-- Define the total contribution of all people
def total_contribution : ℕ := num_people * share_per_person

-- Define the car rental cost
def car_rental_cost : ℕ := total_contribution - airbnb_cost

-- State the theorem to be proved
theorem car_rental_cost_eq_800 : car_rental_cost = 800 :=
  by sorry

end NUMINAMATH_GPT_car_rental_cost_eq_800_l652_65291


namespace NUMINAMATH_GPT_solve_poly_l652_65263

open Real

-- Define the condition as a hypothesis
def prob_condition (x : ℝ) : Prop :=
  arctan (1 / x) + arctan (1 / (x^5)) = π / 6

-- Define the statement to be proven that x satisfies the polynomial equation
theorem solve_poly (x : ℝ) (h : prob_condition x) :
  x^6 - sqrt 3 * x^5 - sqrt 3 * x - 1 = 0 :=
sorry

end NUMINAMATH_GPT_solve_poly_l652_65263


namespace NUMINAMATH_GPT_value_of_expr_l652_65287

theorem value_of_expr (x : ℤ) (h : x = 3) : (2 * x + 6) ^ 2 = 144 := by
  sorry

end NUMINAMATH_GPT_value_of_expr_l652_65287


namespace NUMINAMATH_GPT_typist_original_salary_l652_65241

theorem typist_original_salary (S : ℝ) (h : (1.12 * 0.93 * 1.15 * 0.90 * S = 5204.21)) : S = 5504.00 :=
sorry

end NUMINAMATH_GPT_typist_original_salary_l652_65241


namespace NUMINAMATH_GPT_algebraic_expression_transformation_l652_65265

theorem algebraic_expression_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 + 4 * x + 3 = (x - 1)^2 + a * (x - 1) + b) → (a + b = 14) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_algebraic_expression_transformation_l652_65265


namespace NUMINAMATH_GPT_minimum_sum_of_areas_l652_65224

theorem minimum_sum_of_areas (x y : ℝ) (hx : x + y = 16) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) : 
  (x ^ 2 / 16 + y ^ 2 / 16) / 4 ≥ 8 :=
  sorry

end NUMINAMATH_GPT_minimum_sum_of_areas_l652_65224


namespace NUMINAMATH_GPT_xiaoliang_steps_l652_65225

/-- 
  Xiaoping lives on the fifth floor and climbs 80 steps to get home every day.
  Xiaoliang lives on the fourth floor.
  Prove that the number of steps Xiaoliang has to climb is 60.
-/
theorem xiaoliang_steps (steps_per_floor : ℕ) (h_xiaoping : 4 * steps_per_floor = 80) : 3 * steps_per_floor = 60 :=
by {
  -- The proof is intentionally left out
  sorry
}

end NUMINAMATH_GPT_xiaoliang_steps_l652_65225


namespace NUMINAMATH_GPT_min_value_quadratic_expr_l652_65259

theorem min_value_quadratic_expr (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : a > 0) 
  (h2 : x₁ ≠ x₂) 
  (h3 : x₁^2 - 4*a*x₁ + 3*a^2 < 0) 
  (h4 : x₂^2 - 4*a*x₂ + 3*a^2 < 0)
  (h5 : x₁ + x₂ = 4*a)
  (h6 : x₁ * x₂ = 3*a^2) : 
  x₁ + x₂ + a / (x₁ * x₂) = 4 * a + 1 / (3 * a) := 
sorry

end NUMINAMATH_GPT_min_value_quadratic_expr_l652_65259


namespace NUMINAMATH_GPT_emilee_earns_25_l652_65206

-- Define the conditions
def earns_together (jermaine terrence emilee : ℕ) : Prop := 
  jermaine + terrence + emilee = 90

def jermaine_more (jermaine terrence : ℕ) : Prop :=
  jermaine = terrence + 5

def terrence_earning : ℕ := 30

-- The goal: Prove Emilee earns 25 dollars
theorem emilee_earns_25 (jermaine terrence emilee : ℕ) (h1 : earns_together jermaine terrence emilee) 
  (h2 : jermaine_more jermaine terrence) (h3 : terrence = terrence_earning) : 
  emilee = 25 := 
sorry

end NUMINAMATH_GPT_emilee_earns_25_l652_65206


namespace NUMINAMATH_GPT_sum_cubes_mod_l652_65249

theorem sum_cubes_mod (n : ℕ) : (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3 + 9^3 + 10^3) % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_sum_cubes_mod_l652_65249


namespace NUMINAMATH_GPT_weekend_weekday_ratio_l652_65214

-- Defining the basic constants and conditions
def weekday_episodes : ℕ := 8
def total_episodes_in_week : ℕ := 88

-- Defining the main theorem
theorem weekend_weekday_ratio : (2 * (total_episodes_in_week - 5 * weekday_episodes)) / weekday_episodes = 3 :=
by
  sorry

end NUMINAMATH_GPT_weekend_weekday_ratio_l652_65214


namespace NUMINAMATH_GPT_new_ratio_milk_to_water_l652_65294

def total_volume : ℕ := 100
def initial_milk_ratio : ℚ := 3
def initial_water_ratio : ℚ := 2
def additional_water : ℕ := 48

def new_milk_volume := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume
def new_water_volume := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * total_volume + additional_water

theorem new_ratio_milk_to_water :
  new_milk_volume / (new_water_volume : ℚ) = 15 / 22 :=
by
  sorry

end NUMINAMATH_GPT_new_ratio_milk_to_water_l652_65294


namespace NUMINAMATH_GPT_area_of_quadrilateral_ABCD_l652_65279

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem area_of_quadrilateral_ABCD :
  let AB := 15 * sqrt 2
  let BE := 15 * sqrt 2
  let BC := 7.5 * sqrt 2
  let CE := 7.5 * sqrt 6
  let CD := 7.5 * sqrt 2
  let DE := 7.5 * sqrt 6
  (1/2 * AB * BE) + (1/2 * BC * CE) + (1/2 * CD * DE) = 225 + 112.5 * sqrt 12 :=
by
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_ABCD_l652_65279


namespace NUMINAMATH_GPT_relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l652_65226

-- Define the relationship between Q and t
def remaining_power (t : ℕ) : ℕ := 80 - 15 * t

-- Question 1: Prove relationship between Q and t
theorem relationship_between_Q_and_t : ∀ t : ℕ, remaining_power t = 80 - 15 * t :=
by sorry

-- Question 2: Prove remaining power after 5 hours
theorem remaining_power_after_5_hours : remaining_power 5 = 5 :=
by sorry

-- Question 3: Prove distance the car can travel with 40 kW·h remaining power
theorem distance_with_40_power 
  (remaining_power : ℕ := (80 - 15 * t)) 
  (t := 8 / 3)
  (speed : ℕ := 90) : (90 * (8 / 3)) = 240 :=
by sorry

end NUMINAMATH_GPT_relationship_between_Q_and_t_remaining_power_after_5_hours_distance_with_40_power_l652_65226


namespace NUMINAMATH_GPT_inverse_proportion_point_l652_65273

theorem inverse_proportion_point (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : y1 = k / x1) 
  (h2 : x1 = -2) 
  (h3 : y1 = 3)
  (h4 : x2 = 2) :
  y2 = -3 := 
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_inverse_proportion_point_l652_65273


namespace NUMINAMATH_GPT_sum_in_range_l652_65205

open Real

def mix1 := 3 + 3/8
def mix2 := 4 + 2/5
def mix3 := 6 + 1/11
def mixed_sum := mix1 + mix2 + mix3

theorem sum_in_range : mixed_sum > 13 ∧ mixed_sum < 14 :=
by
  -- Since we are just providing the statement, we leave the proof as a placeholder.
  sorry

end NUMINAMATH_GPT_sum_in_range_l652_65205


namespace NUMINAMATH_GPT_surface_area_correct_l652_65276

def radius_hemisphere : ℝ := 9
def height_cone : ℝ := 12
def radius_cone_base : ℝ := 9

noncomputable def total_surface_area : ℝ := 
  let base_area : ℝ := radius_hemisphere^2 * Real.pi
  let curved_area_hemisphere : ℝ := 2 * radius_hemisphere^2 * Real.pi
  let slant_height_cone : ℝ := Real.sqrt (radius_cone_base^2 + height_cone^2)
  let lateral_area_cone : ℝ := radius_cone_base * slant_height_cone * Real.pi
  base_area + curved_area_hemisphere + lateral_area_cone

theorem surface_area_correct : total_surface_area = 378 * Real.pi := by
  sorry

end NUMINAMATH_GPT_surface_area_correct_l652_65276


namespace NUMINAMATH_GPT_number_of_times_each_player_plays_l652_65217

def players : ℕ := 7
def total_games : ℕ := 42

theorem number_of_times_each_player_plays (x : ℕ) 
  (H1 : 42 = (players * (players - 1) * x) / 2) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_times_each_player_plays_l652_65217


namespace NUMINAMATH_GPT_alice_wins_l652_65229

noncomputable def game_condition (r : ℝ) (f : ℕ → ℝ) : Prop :=
∀ n, 0 ≤ f n ∧ f n ≤ 1

theorem alice_wins (r : ℝ) (f : ℕ → ℝ) (hf : game_condition r f) :
  r ≤ 3 → (∃ x : ℕ → ℝ, game_condition 3 x ∧ (abs (x 0 - x 1) + abs (x 2 - x 3) + abs (x 4 - x 5) ≥ r)) :=
by
  sorry

end NUMINAMATH_GPT_alice_wins_l652_65229


namespace NUMINAMATH_GPT_expand_polynomial_l652_65270

noncomputable def p (x : ℝ) : ℝ := 7 * x ^ 2 + 5
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 3 + 2 * x + 1

theorem expand_polynomial (x : ℝ) : 
  (p x) * (q x) = 21 * x ^ 5 + 29 * x ^ 3 + 7 * x ^ 2 + 10 * x + 5 := 
by sorry

end NUMINAMATH_GPT_expand_polynomial_l652_65270


namespace NUMINAMATH_GPT_rooms_in_house_l652_65293

-- define the number of paintings
def total_paintings : ℕ := 32

-- define the number of paintings per room
def paintings_per_room : ℕ := 8

-- define the number of rooms
def number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ := total_paintings / paintings_per_room

-- state the theorem
theorem rooms_in_house : number_of_rooms total_paintings paintings_per_room = 4 :=
by sorry

end NUMINAMATH_GPT_rooms_in_house_l652_65293


namespace NUMINAMATH_GPT_find_digit_A_l652_65258

open Nat

theorem find_digit_A :
  let n := 52
  let k := 13
  let number_of_hands := choose n k
  number_of_hands = 635013587600 → 0 = 0 := by
  suffices h: 635013587600 = 635013587600 by
    simp [h]
  sorry

end NUMINAMATH_GPT_find_digit_A_l652_65258


namespace NUMINAMATH_GPT_squared_distance_focus_product_tangents_l652_65271

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end NUMINAMATH_GPT_squared_distance_focus_product_tangents_l652_65271


namespace NUMINAMATH_GPT_triangle_angles_l652_65244

theorem triangle_angles (α β : ℝ) (A B C : ℝ) (hA : A = 2) (hB : B = 3) (hC : C = 4) :
  2 * α + 3 * β = 180 :=
sorry

end NUMINAMATH_GPT_triangle_angles_l652_65244


namespace NUMINAMATH_GPT_darwin_final_money_l652_65222

def initial_amount : ℕ := 600
def spent_on_gas (initial : ℕ) : ℕ := initial * 1 / 3
def remaining_after_gas (initial spent_gas : ℕ) : ℕ := initial - spent_gas
def spent_on_food (remaining : ℕ) : ℕ := remaining * 1 / 4
def final_amount (remaining spent_food : ℕ) : ℕ := remaining - spent_food

theorem darwin_final_money :
  final_amount (remaining_after_gas initial_amount (spent_on_gas initial_amount)) (spent_on_food (remaining_after_gas initial_amount (spent_on_gas initial_amount))) = 300 :=
by
  sorry

end NUMINAMATH_GPT_darwin_final_money_l652_65222


namespace NUMINAMATH_GPT_yellow_ball_percentage_l652_65290

theorem yellow_ball_percentage
  (yellow_balls : ℕ)
  (brown_balls : ℕ)
  (blue_balls : ℕ)
  (green_balls : ℕ)
  (total_balls : ℕ := yellow_balls + brown_balls + blue_balls + green_balls)
  (h_yellow : yellow_balls = 75)
  (h_brown : brown_balls = 120)
  (h_blue : blue_balls = 45)
  (h_green : green_balls = 60) :
  (yellow_balls * 100) / total_balls = 25 := 
by
  sorry

end NUMINAMATH_GPT_yellow_ball_percentage_l652_65290


namespace NUMINAMATH_GPT_probability_palindrome_divisible_by_11_is_zero_l652_65268

-- Define the three-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Define the divisibility condition
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Prove that the probability is zero
theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ n, is_palindrome n ∧ is_divisible_by_11 n) →
  (0 : ℕ) = 0 := by
  sorry

end NUMINAMATH_GPT_probability_palindrome_divisible_by_11_is_zero_l652_65268


namespace NUMINAMATH_GPT_disproves_proposition_b_l652_65299

-- Definition and condition of complementary angles
def angles_complementary (angle1 angle2: ℝ) : Prop := angle1 + angle2 = 180

-- Proposition to disprove
def disprove (angle1 angle2: ℝ) : Prop := ¬ ((angle1 < 90 ∧ angle2 > 90 ∧ angle2 < 180) ∨ (angle2 < 90 ∧ angle1 > 90 ∧ angle1 < 180))

-- Definition of angles in sets
def set_a := (120, 60)
def set_b := (95.1, 84.9)
def set_c := (30, 60)
def set_d := (90, 90)

-- Statement to prove
theorem disproves_proposition_b : 
  (angles_complementary 95.1 84.9) ∧ (disprove 95.1 84.9) :=
by
  sorry

end NUMINAMATH_GPT_disproves_proposition_b_l652_65299


namespace NUMINAMATH_GPT_alicia_stickers_l652_65242

theorem alicia_stickers :
  ∃ S : ℕ, S > 2 ∧
  (S % 5 = 2) ∧ (S % 11 = 2) ∧ (S % 13 = 2) ∧
  S = 717 :=
sorry

end NUMINAMATH_GPT_alicia_stickers_l652_65242


namespace NUMINAMATH_GPT_relationship_abc_l652_65233

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.exp (-Real.pi)
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- proofs would be added here
  sorry

end NUMINAMATH_GPT_relationship_abc_l652_65233


namespace NUMINAMATH_GPT_solve_for_y_l652_65203

theorem solve_for_y (x y : ℝ) (h : 4 * x - y = 3) : y = 4 * x - 3 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l652_65203


namespace NUMINAMATH_GPT_total_students_is_37_l652_65212

-- Let b be the number of blue swim caps 
-- Let r be the number of red swim caps
variables (b r : ℕ)

-- The number of blue swim caps according to the male sports commissioner
def condition1 : Prop := b = 4 * r + 1

-- The number of blue swim caps according to the female sports commissioner
def condition2 : Prop := b = r + 24

-- The total number of students in the 3rd grade
def total_students : ℕ := b + r

theorem total_students_is_37 (h1 : condition1 b r) (h2 : condition2 b r) : total_students b r = 37 :=
by sorry

end NUMINAMATH_GPT_total_students_is_37_l652_65212


namespace NUMINAMATH_GPT_percentage_alcohol_in_first_vessel_is_zero_l652_65266

theorem percentage_alcohol_in_first_vessel_is_zero (x : ℝ) :
  ∀ (alcohol_first_vessel total_vessel_capacity first_vessel_capacity second_vessel_capacity concentration_mixture : ℝ),
  first_vessel_capacity = 2 →
  (∃ xpercent, alcohol_first_vessel = (first_vessel_capacity * xpercent / 100)) →
  second_vessel_capacity = 6 →
  (∃ ypercent, ypercent = 40 ∧ alcohol_first_vessel + 2.4 = concentration_mixture * (total_vessel_capacity/8) * 8) →
  concentration_mixture = 0.3 →
  0 = x := sorry

end NUMINAMATH_GPT_percentage_alcohol_in_first_vessel_is_zero_l652_65266


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l652_65252

-- Definitions and conditions
def monomial_degree_condition (a : ℝ) : Prop := 2 + (1 + a) = 5

-- Proof goals
theorem problem_1 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = 9 := sorry
theorem problem_2 (a : ℝ) (h : monomial_degree_condition a) : (a + 1) * (a^2 - a + 1) = 9 := sorry
theorem problem_3 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = (a + 1) * (a^2 - a + 1) := sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l652_65252


namespace NUMINAMATH_GPT_number_of_divisors_3465_l652_65234

def prime_factors_3465 : Prop := 3465 = 3^2 * 5 * 7^2

theorem number_of_divisors_3465 (h : prime_factors_3465) : Nat.totient 3465 = 18 :=
  sorry

end NUMINAMATH_GPT_number_of_divisors_3465_l652_65234


namespace NUMINAMATH_GPT_find_x2_minus_x1_l652_65257

theorem find_x2_minus_x1 (a x1 x2 d e : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x1 ≠ x2) (h_e : e = -d * x1)
  (h_y1 : ∀ x, y1 = a * (x - x1) * (x - x2)) (h_y2 : ∀ x, y2 = d * x + e)
  (h_intersect : ∀ x, y = a * (x - x1) * (x - x2) + (d * x + e)) 
  (h_single_point : ∀ x, y = a * (x - x1)^2) :
  x2 - x1 = d / a :=
sorry

end NUMINAMATH_GPT_find_x2_minus_x1_l652_65257


namespace NUMINAMATH_GPT_ways_to_make_change_l652_65247

theorem ways_to_make_change : ∃ ways : ℕ, ways = 60 ∧ (∀ (p n d q : ℕ), p + 5 * n + 10 * d + 25 * q = 55 → True) := 
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_ways_to_make_change_l652_65247


namespace NUMINAMATH_GPT_sin_x_cos_x_value_l652_65255

theorem sin_x_cos_x_value (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : Real.sin x * Real.cos x = 4 / 17 :=
  sorry

end NUMINAMATH_GPT_sin_x_cos_x_value_l652_65255


namespace NUMINAMATH_GPT_sum_reciprocal_squares_roots_l652_65219

-- Define the polynomial P(X) = X^3 - 3X - 1
noncomputable def P (X : ℂ) : ℂ := X^3 - 3 * X - 1

-- Define the roots of the polynomial
variables (r1 r2 r3 : ℂ)

-- State that r1, r2, and r3 are roots of the polynomial
variable (hroots : P r1 = 0 ∧ P r2 = 0 ∧ P r3 = 0)

-- Vieta's formulas conditions for the polynomial P
variable (hvieta : r1 + r2 + r3 = 0 ∧ r1 * r2 + r1 * r3 + r2 * r3 = -3 ∧ r1 * r2 * r3 = 1)

-- The sum of the reciprocals of the squares of the roots
theorem sum_reciprocal_squares_roots : (1 / r1^2) + (1 / r2^2) + (1 / r3^2) = 9 := 
sorry

end NUMINAMATH_GPT_sum_reciprocal_squares_roots_l652_65219


namespace NUMINAMATH_GPT_map_width_l652_65211

theorem map_width (length : ℝ) (area : ℝ) (h1 : length = 2) (h2 : area = 20) : ∃ (width : ℝ), width = 10 :=
by
  sorry

end NUMINAMATH_GPT_map_width_l652_65211


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l652_65261

theorem remainder_when_divided_by_9 (z : ℤ) (k : ℤ) (h : z + 3 = 9 * k) :
  z % 9 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l652_65261


namespace NUMINAMATH_GPT_trigonometric_identity_l652_65296

theorem trigonometric_identity (α : ℝ) (h : Real.tan (Real.pi + α) = 2) :
  4 * Real.sin α * Real.cos α + 3 * (Real.cos α) ^ 2 = 11 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l652_65296


namespace NUMINAMATH_GPT_condition_for_a_b_complex_l652_65235

theorem condition_for_a_b_complex (a b : ℂ) (h1 : a ≠ 0) (h2 : 2 * a + b ≠ 0) :
  (2 * a + b) / a = b / (2 * a + b) → 
  (∃ z : ℂ, a = z ∨ b = z) ∨ 
  ((∃ z1 : ℂ, a = z1) ∧ (∃ z2 : ℂ, b = z2)) :=
sorry

end NUMINAMATH_GPT_condition_for_a_b_complex_l652_65235


namespace NUMINAMATH_GPT_video_time_per_week_l652_65267

-- Define the basic conditions
def short_video_length : ℕ := 2
def multiplier : ℕ := 6
def long_video_length : ℕ := multiplier * short_video_length
def short_videos_per_day : ℕ := 2
def long_videos_per_day : ℕ := 1
def days_in_week : ℕ := 7

-- Calculate daily and weekly video release time
def daily_video_time : ℕ := (short_videos_per_day * short_video_length) + (long_videos_per_day * long_video_length)
def weekly_video_time : ℕ := daily_video_time * days_in_week

-- Main theorem to prove
theorem video_time_per_week : weekly_video_time = 112 := by
    sorry

end NUMINAMATH_GPT_video_time_per_week_l652_65267


namespace NUMINAMATH_GPT_no_non_trivial_solution_l652_65223

theorem no_non_trivial_solution (a b c : ℤ) (h : a^2 = 2 * b^2 + 3 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end NUMINAMATH_GPT_no_non_trivial_solution_l652_65223


namespace NUMINAMATH_GPT_simplify_expression_l652_65281

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l652_65281


namespace NUMINAMATH_GPT_distance_between_A_and_B_l652_65221

theorem distance_between_A_and_B 
  (d : ℕ) -- The distance we want to prove
  (ha : ∀ (t : ℕ), d = 700 * t)
  (hb : ∀ (t : ℕ), d + 400 = 2100 * t) :
  d = 1700 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l652_65221


namespace NUMINAMATH_GPT_layla_earnings_l652_65283

-- Define the hourly rates for each family
def rate_donaldson : ℕ := 15
def rate_merck : ℕ := 18
def rate_hille : ℕ := 20
def rate_johnson : ℕ := 22
def rate_ramos : ℕ := 25

-- Define the hours Layla worked for each family
def hours_donaldson : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def hours_johnson : ℕ := 4
def hours_ramos : ℕ := 2

-- Calculate the earnings for each family
def earnings_donaldson : ℕ := rate_donaldson * hours_donaldson
def earnings_merck : ℕ := rate_merck * hours_merck
def earnings_hille : ℕ := rate_hille * hours_hille
def earnings_johnson : ℕ := rate_johnson * hours_johnson
def earnings_ramos : ℕ := rate_ramos * hours_ramos

-- Calculate total earnings
def total_earnings : ℕ :=
  earnings_donaldson + earnings_merck + earnings_hille + earnings_johnson + earnings_ramos

-- The assertion that Layla's total earnings are $411
theorem layla_earnings : total_earnings = 411 := by
  sorry

end NUMINAMATH_GPT_layla_earnings_l652_65283


namespace NUMINAMATH_GPT_range_of_a_l652_65284

noncomputable def f (a x : ℝ) : ℝ := 
  if x < 1 then a^x else (a-3)*x + 4*a

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  0 < a ∧ a ≤ 3/4 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l652_65284


namespace NUMINAMATH_GPT_juniper_bones_proof_l652_65295

-- Define the conditions
def juniper_original_bones : ℕ := 4
def bones_given_by_master : ℕ := juniper_original_bones
def bones_stolen_by_neighbor : ℕ := 2

-- Define the final number of bones Juniper has
def juniper_remaining_bones : ℕ := juniper_original_bones + bones_given_by_master - bones_stolen_by_neighbor

-- State the theorem to prove the given answer
theorem juniper_bones_proof : juniper_remaining_bones = 6 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_juniper_bones_proof_l652_65295


namespace NUMINAMATH_GPT_marilyn_initial_bottle_caps_l652_65278

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end NUMINAMATH_GPT_marilyn_initial_bottle_caps_l652_65278


namespace NUMINAMATH_GPT_notebook_cost_l652_65200

theorem notebook_cost (total_spent ruler_cost pencil_count pencil_cost: ℕ)
  (h1 : total_spent = 74)
  (h2 : ruler_cost = 18)
  (h3 : pencil_count = 3)
  (h4 : pencil_cost = 7) :
  total_spent - (ruler_cost + pencil_count * pencil_cost) = 35 := 
by 
  sorry

end NUMINAMATH_GPT_notebook_cost_l652_65200


namespace NUMINAMATH_GPT_P_has_common_root_l652_65238

def P (x : ℝ) (p : ℝ) (q : ℝ) : ℝ := x^2 + p * x + q

theorem P_has_common_root (p q : ℝ) (t : ℝ) (h : P t p q = 0) :
  P 0 p q * P 1 p q = 0 :=
by
  sorry

end NUMINAMATH_GPT_P_has_common_root_l652_65238


namespace NUMINAMATH_GPT_cost_per_pound_correct_l652_65213

noncomputable def cost_per_pound_of_coffee (initial_amount spent_amount pounds_of_coffee : ℕ) : ℚ :=
  (initial_amount - spent_amount) / pounds_of_coffee

theorem cost_per_pound_correct :
  let initial_amount := 70
  let amount_left    := 35.68
  let pounds_of_coffee := 4
  (initial_amount - amount_left) / pounds_of_coffee = 8.58 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_pound_correct_l652_65213


namespace NUMINAMATH_GPT_area_of_the_region_l652_65280

noncomputable def region_area (C D : ℝ×ℝ) (rC rD : ℝ) (y : ℝ) : ℝ :=
  let rect_area := (D.1 - C.1) * y
  let sector_areaC := (1 / 2) * Real.pi * rC^2
  let sector_areaD := (1 / 2) * Real.pi * rD^2
  rect_area - (sector_areaC + sector_areaD)

theorem area_of_the_region :
  region_area (3, 5) (10, 5) 3 5 5 = 35 - 17 * Real.pi := by
  sorry

end NUMINAMATH_GPT_area_of_the_region_l652_65280


namespace NUMINAMATH_GPT_circumscribed_circle_radius_of_rectangle_l652_65260

theorem circumscribed_circle_radius_of_rectangle 
  (a b : ℝ) 
  (h1: a = 1) 
  (angle_between_diagonals : ℝ) 
  (h2: angle_between_diagonals = 60) : 
  ∃ R, R = 1 :=
by 
  sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_of_rectangle_l652_65260


namespace NUMINAMATH_GPT_point_B_position_l652_65246

/-- Given points A and B on the same number line, with A at -2 and B 5 units away from A, prove 
    that B can be either -7 or 3. -/
theorem point_B_position (A B : ℤ) (hA : A = -2) (hB : (B = A + 5) ∨ (B = A - 5)) : 
  B = 3 ∨ B = -7 :=
sorry

end NUMINAMATH_GPT_point_B_position_l652_65246


namespace NUMINAMATH_GPT_find_number_l652_65231

theorem find_number (x : ℝ) : 4 * x - 23 = 33 → x = 14 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_number_l652_65231


namespace NUMINAMATH_GPT_factor_theorem_l652_65274

theorem factor_theorem (h : ℤ) : (∀ m : ℤ, (m - 8) ∣ (m^2 - h * m - 24) ↔ h = 5) :=
  sorry

end NUMINAMATH_GPT_factor_theorem_l652_65274


namespace NUMINAMATH_GPT_same_function_l652_65254

noncomputable def f (x : ℝ) : ℝ := x
noncomputable def g (t : ℝ) : ℝ := (t^3 + t) / (t^2 + 1)

theorem same_function : ∀ x : ℝ, f x = g x :=
by sorry

end NUMINAMATH_GPT_same_function_l652_65254


namespace NUMINAMATH_GPT_isosceles_triangle_length_l652_65220

variable (a b : ℝ)

theorem isosceles_triangle_length (h1 : 2 * a + 3 = 16) (h2 : a != 3) : a = 6.5 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_length_l652_65220


namespace NUMINAMATH_GPT_curve_transformation_l652_65298

def matrix_transform (a : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (0 * x + 1 * y, a * x + 0 * y)

def curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 1

def transformed_curve_eq (x y : ℝ) : Prop :=
  x ^ 2 + (y ^ 2) / 4 = 1

theorem curve_transformation (a : ℝ) 
  (h₁ : matrix_transform a 2 (-2) = (-2, 4))
  (h₂ : ∀ x y, curve_eq x y → transformed_curve_eq (matrix_transform a x y).fst (matrix_transform a x y).snd) :
  a = 2 ∧ ∀ x y, curve_eq x y → transformed_curve_eq (0 * x + 1 * y) (2 * x + 0 * y) :=
by
  sorry

end NUMINAMATH_GPT_curve_transformation_l652_65298


namespace NUMINAMATH_GPT_selling_price_correct_l652_65248

noncomputable def cost_price : ℝ := 2800
noncomputable def loss_percentage : ℝ := 25
noncomputable def loss_amount (cost_price loss_percentage : ℝ) : ℝ := (loss_percentage / 100) * cost_price
noncomputable def selling_price (cost_price loss_amount : ℝ) : ℝ := cost_price - loss_amount

theorem selling_price_correct : 
  selling_price cost_price (loss_amount cost_price loss_percentage) = 2100 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l652_65248


namespace NUMINAMATH_GPT_volume_inside_sphere_outside_cylinder_l652_65216

noncomputable def volumeDifference (r_cylinder base_radius_sphere : ℝ) :=
  let height := 4 * Real.sqrt 5
  let V_sphere := (4/3) * Real.pi * base_radius_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * height
  V_sphere - V_cylinder

theorem volume_inside_sphere_outside_cylinder
  (base_radius_sphere r_cylinder : ℝ) (h_base_radius_sphere : base_radius_sphere = 6) (h_r_cylinder : r_cylinder = 4) :
  volumeDifference r_cylinder base_radius_sphere = (288 - 64 * Real.sqrt 5) * Real.pi := by
  sorry

end NUMINAMATH_GPT_volume_inside_sphere_outside_cylinder_l652_65216


namespace NUMINAMATH_GPT_Diana_friends_count_l652_65209

theorem Diana_friends_count (totalErasers : ℕ) (erasersPerFriend : ℕ) 
  (h1: totalErasers = 3840) (h2: erasersPerFriend = 80) : 
  totalErasers / erasersPerFriend = 48 := 
by 
  sorry

end NUMINAMATH_GPT_Diana_friends_count_l652_65209


namespace NUMINAMATH_GPT_factorial_fraction_is_integer_l652_65230

open Nat

theorem factorial_fraction_is_integer (m n : ℕ) : 
  ↑((factorial (2 * m)) * (factorial (2 * n))) % (factorial m * factorial n * factorial (m + n)) = 0 := sorry

end NUMINAMATH_GPT_factorial_fraction_is_integer_l652_65230


namespace NUMINAMATH_GPT_tigers_count_l652_65251

theorem tigers_count (T C : ℝ) 
  (h1 : 12 + T + C = 39) 
  (h2 : C = 0.5 * (12 + T)) : 
  T = 14 := by
  sorry

end NUMINAMATH_GPT_tigers_count_l652_65251


namespace NUMINAMATH_GPT_sum_cotangents_equal_l652_65236

theorem sum_cotangents_equal (a b c S m_a m_b m_c S' : ℝ) (cot_A cot_B cot_C cot_A' cot_B' cot_C' : ℝ)
  (h1 : cot_A + cot_B + cot_C = (a^2 + b^2 + c^2) / (4 * S))
  (h2 : m_a^2 + m_b^2 + m_c^2 = 3 * (a^2 + b^2 + c^2) / 4)
  (h3 : S' = 3 * S / 4)
  (h4 : cot_A' + cot_B' + cot_C' = (m_a^2 + m_b^2 + m_c^2) / (4 * S')) :
  cot_A + cot_B + cot_C = cot_A' + cot_B' + cot_C' :=
by
  -- Proof is needed, but omitted here
  sorry

end NUMINAMATH_GPT_sum_cotangents_equal_l652_65236


namespace NUMINAMATH_GPT_cube_dihedral_angle_is_60_degrees_l652_65210

-- Define the cube and related geometrical features
structure Point := (x y z : ℝ)
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : Point)
  (is_cube : true) -- Placeholder for cube properties

-- Define the function to calculate dihedral angle measure
noncomputable def dihedral_angle_measure (cube: Cube) : ℝ := sorry

-- The theorem statement
theorem cube_dihedral_angle_is_60_degrees (cube : Cube) : dihedral_angle_measure cube = 60 :=
by sorry

end NUMINAMATH_GPT_cube_dihedral_angle_is_60_degrees_l652_65210


namespace NUMINAMATH_GPT_problem_1_problem_2_l652_65262

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Problem I
theorem problem_1 (x : ℝ) : (f x 1 ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
by sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -3) : a = 6 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l652_65262


namespace NUMINAMATH_GPT_conference_duration_is_960_l652_65269

-- The problem statement definition
def conference_sessions_duration_in_minutes (day1_hours : ℕ) (day1_minutes : ℕ) (day2_hours : ℕ) (day2_minutes : ℕ) : ℕ :=
  (day1_hours * 60 + day1_minutes) + (day2_hours * 60 + day2_minutes)

-- The theorem we want to prove given the above conditions
theorem conference_duration_is_960 :
  conference_sessions_duration_in_minutes 7 15 8 45 = 960 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_conference_duration_is_960_l652_65269


namespace NUMINAMATH_GPT_remainder_2753_div_98_l652_65204

theorem remainder_2753_div_98 : (2753 % 98) = 9 := 
by sorry

end NUMINAMATH_GPT_remainder_2753_div_98_l652_65204


namespace NUMINAMATH_GPT_henri_drove_farther_l652_65240

theorem henri_drove_farther (gervais_avg_miles_per_day : ℕ) (gervais_days : ℕ) (henri_total_miles : ℕ)
  (h1 : gervais_avg_miles_per_day = 315) (h2 : gervais_days = 3) (h3 : henri_total_miles = 1250) :
  (henri_total_miles - (gervais_avg_miles_per_day * gervais_days) = 305) :=
by
  -- Here we would provide the proof, but we are omitting it as requested
  sorry

end NUMINAMATH_GPT_henri_drove_farther_l652_65240


namespace NUMINAMATH_GPT_sqrt_5sq_4six_eq_320_l652_65277

theorem sqrt_5sq_4six_eq_320 : Real.sqrt (5^2 * 4^6) = 320 :=
by sorry

end NUMINAMATH_GPT_sqrt_5sq_4six_eq_320_l652_65277


namespace NUMINAMATH_GPT_polynomial_roots_sum_reciprocal_l652_65237

open Polynomial

theorem polynomial_roots_sum_reciprocal (a b c : ℝ) (h : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
    (40 * a^3 - 70 * a^2 + 32 * a - 3 = 0) ∧
    (40 * b^3 - 70 * b^2 + 32 * b - 3 = 0) ∧
    (40 * c^3 - 70 * c^2 + 32 * c - 3 = 0) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_roots_sum_reciprocal_l652_65237


namespace NUMINAMATH_GPT_length_of_uncovered_side_l652_65250

-- Define the conditions of the problem
def area_condition (L W : ℝ) : Prop := L * W = 210
def fencing_condition (L W : ℝ) : Prop := L + 2 * W = 41

-- Define the proof statement
theorem length_of_uncovered_side (L W : ℝ) (h_area : area_condition L W) (h_fence : fencing_condition L W) : 
  L = 21 :=
  sorry

end NUMINAMATH_GPT_length_of_uncovered_side_l652_65250


namespace NUMINAMATH_GPT_pythagorean_triple_divisible_by_60_l652_65227

theorem pythagorean_triple_divisible_by_60 
  (a b c : ℕ) (h : a * a + b * b = c * c) : 60 ∣ (a * b * c) :=
sorry

end NUMINAMATH_GPT_pythagorean_triple_divisible_by_60_l652_65227


namespace NUMINAMATH_GPT_evaluate_expression_l652_65208

theorem evaluate_expression :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
sorry

end NUMINAMATH_GPT_evaluate_expression_l652_65208


namespace NUMINAMATH_GPT_carol_total_peanuts_l652_65207

open Nat

-- Define the conditions
def peanuts_from_tree : Nat := 48
def peanuts_from_ground : Nat := 178
def bags_of_peanuts : Nat := 3
def peanuts_per_bag : Nat := 250

-- Define the total number of peanuts Carol has to prove it equals 976
def total_peanuts (peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag : Nat) : Nat :=
  peanuts_from_tree + peanuts_from_ground + (bags_of_peanuts * peanuts_per_bag)

theorem carol_total_peanuts : total_peanuts peanuts_from_tree peanuts_from_ground bags_of_peanuts peanuts_per_bag = 976 :=
  by
    -- proof goes here
    sorry

end NUMINAMATH_GPT_carol_total_peanuts_l652_65207


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l652_65232

theorem simplify_and_evaluate_expression (a b : ℤ) (h_a : a = 2) (h_b : b = -1) : 
  2 * (-a^2 + 2 * a * b) - 3 * (a * b - a^2) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l652_65232


namespace NUMINAMATH_GPT_positive_divisors_multiple_of_15_l652_65286

theorem positive_divisors_multiple_of_15 (a b c : ℕ) (n : ℕ) (divisor : ℕ) (h_factorization : n = 6480)
  (h_prime_factorization : n = 2^4 * 3^4 * 5^1)
  (h_divisor : divisor = 2^a * 3^b * 5^c)
  (h_a_range : 0 ≤ a ∧ a ≤ 4)
  (h_b_range : 1 ≤ b ∧ b ≤ 4)
  (h_c_range : 1 ≤ c ∧ c ≤ 1) : sorry :=
sorry

end NUMINAMATH_GPT_positive_divisors_multiple_of_15_l652_65286


namespace NUMINAMATH_GPT_angle_bisector_divides_longest_side_l652_65264

theorem angle_bisector_divides_longest_side :
  ∀ (a b c : ℕ) (p q : ℕ), a = 12 → b = 15 → c = 18 →
  p + q = c → p * b = q * a → p = 8 ∧ q = 10 :=
by
  intros a b c p q ha hb hc hpq hprop
  rw [ha, hb, hc] at *
  sorry

end NUMINAMATH_GPT_angle_bisector_divides_longest_side_l652_65264


namespace NUMINAMATH_GPT_max_f_when_a_minus_1_range_of_a_l652_65202

noncomputable section

-- Definitions of the functions given in the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
def g (a : ℝ) (x : ℝ) : ℝ := x * f a x
def h (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - (2 * a - 1) * x + (a - 1)

-- Statement (1): Proving the maximum value of f(x) when a = -1
theorem max_f_when_a_minus_1 : 
  (∀ x : ℝ, f (-1) x ≤ f (-1) 1) :=
sorry

-- Statement (2): Proving the range of a when g(x) ≤ h(x) for x ≥ 1
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → g a x ≤ h a x) → (1 ≤ a) :=
sorry

end NUMINAMATH_GPT_max_f_when_a_minus_1_range_of_a_l652_65202


namespace NUMINAMATH_GPT_max_value_of_expression_l652_65289

open Real

theorem max_value_of_expression
  (x y : ℝ)
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : x^2 - 2 * x * y + 3 * y^2 = 10) 
  : x^2 + 2 * x * y + 3 * y^2 ≤ 10 * (45 + 42 * sqrt 3) := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l652_65289


namespace NUMINAMATH_GPT_algebraic_expression_value_l652_65285

theorem algebraic_expression_value (a b c : ℝ) (h : (∀ x : ℝ, (x - 1) * (x + 2) = a * x^2 + b * x + c)) :
  4 * a - 2 * b + c = 0 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l652_65285


namespace NUMINAMATH_GPT_count_positive_integers_l652_65228

theorem count_positive_integers (n : ℤ) : 
  (130 * n) ^ 50 > (n : ℤ) ^ 100 ∧ (n : ℤ) ^ 100 > 2 ^ 200 → 
  ∃ k : ℕ, k = 125 := sorry

end NUMINAMATH_GPT_count_positive_integers_l652_65228


namespace NUMINAMATH_GPT_original_cube_volume_l652_65288

theorem original_cube_volume
  (a : ℝ)
  (h : (a + 2) * (a - 1) * a = a^3 + 14) :
  a^3 = 64 :=
by
  sorry

end NUMINAMATH_GPT_original_cube_volume_l652_65288


namespace NUMINAMATH_GPT_polynomial_divisible_by_seven_l652_65292

-- Define the theorem
theorem polynomial_divisible_by_seven (n : ℤ) : 7 ∣ (n + 7)^2 - n^2 :=
by sorry

end NUMINAMATH_GPT_polynomial_divisible_by_seven_l652_65292


namespace NUMINAMATH_GPT_avg_age_team_proof_l652_65272

-- Defining the known constants
def members : ℕ := 15
def avg_age_team : ℕ := 28
def captain_age : ℕ := avg_age_team + 4
def remaining_players : ℕ := members - 2
def avg_age_remaining : ℕ := avg_age_team - 2

-- Stating the problem to prove the average age remains 28
theorem avg_age_team_proof (W : ℕ) :
  28 = avg_age_team ∧
  members = 15 ∧
  captain_age = avg_age_team + 4 ∧
  remaining_players = members - 2 ∧
  avg_age_remaining = avg_age_team - 2 ∧
  28 * 15 = 26 * 13 + captain_age + W :=
sorry

end NUMINAMATH_GPT_avg_age_team_proof_l652_65272


namespace NUMINAMATH_GPT_man_speed_l652_65201

theorem man_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h1 : distance = 12)
  (h2 : time_minutes = 72)
  (h3 : time_hours = time_minutes / 60)
  (h4 : speed = distance / time_hours) : speed = 10 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_l652_65201


namespace NUMINAMATH_GPT_range_of_m_l652_65253

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0) ∨ ((1 / 2) * m > 1) ↔ ((m > 4) ∧ ¬(∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0)) :=
sorry

end NUMINAMATH_GPT_range_of_m_l652_65253


namespace NUMINAMATH_GPT_speed_in_still_water_l652_65297

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) (h₁ : upstream_speed = 20) (h₂ : downstream_speed = 60) :
  (upstream_speed + downstream_speed) / 2 = 40 := by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l652_65297


namespace NUMINAMATH_GPT_geometric_arithmetic_seq_unique_ratio_l652_65243

variable (d : ℚ) (q : ℚ) (k : ℤ)
variable (h_d_nonzero : d ≠ 0)
variable (h_q_pos : 0 < q) (h_q_lt_one : q < 1)
variable (h_integer : 14 / (1 + q + q^2) = k)

theorem geometric_arithmetic_seq_unique_ratio :
  q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_seq_unique_ratio_l652_65243
