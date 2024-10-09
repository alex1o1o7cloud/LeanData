import Mathlib

namespace find_coefficients_l464_46491

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients_l464_46491


namespace arithmetic_sum_S9_l464_46495

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variable (S : ℕ → ℝ) -- Define the sum of the first n terms
variable (d : ℝ) -- Define the common difference
variable (a_1 : ℝ) -- Define the first term of the sequence

-- Assume the arithmetic sequence properties
axiom arith_seq_def : ∀ n, a (n + 1) = a_1 + n * d

-- Define the sum of the first n terms
axiom sum_first_n_terms : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Given condition
axiom given_condition : a 1 + a 7 = 15 - a 4

theorem arithmetic_sum_S9 : S 9 = 45 :=
by
  -- Proof omitted
  sorry

end arithmetic_sum_S9_l464_46495


namespace max_students_l464_46461

theorem max_students (pens pencils : ℕ) (h_pens : pens = 1340) (h_pencils : pencils = 1280) : Nat.gcd pens pencils = 20 := by
    sorry

end max_students_l464_46461


namespace isosceles_triangle_of_condition_l464_46465

theorem isosceles_triangle_of_condition (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 2 * b * Real.cos C)
  (h2 : A + B + C = Real.pi) :
  (B = C) ∨ (A = C) ∨ (A = B) := 
sorry

end isosceles_triangle_of_condition_l464_46465


namespace polygon_sides_eq_six_l464_46429

theorem polygon_sides_eq_six (n : ℕ) (h : 3 * n - (n * (n - 3)) / 2 = 6) : n = 6 := 
sorry

end polygon_sides_eq_six_l464_46429


namespace problem_l464_46471

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l464_46471


namespace geometric_progression_positions_l464_46435

theorem geometric_progression_positions (u1 q : ℝ) (m n p : ℕ)
  (h27 : 27 = u1 * q ^ (m - 1))
  (h8 : 8 = u1 * q ^ (n - 1))
  (h12 : 12 = u1 * q ^ (p - 1)) :
  m = 3 * p - 2 * n :=
sorry

end geometric_progression_positions_l464_46435


namespace evaluate_expression_l464_46493

-- Define the base and the exponents
def base : ℝ := 64
def exponent1 : ℝ := 0.125
def exponent2 : ℝ := 0.375
def combined_result : ℝ := 8

-- Statement of the problem
theorem evaluate_expression : (base^exponent1) * (base^exponent2) = combined_result := 
by 
  sorry

end evaluate_expression_l464_46493


namespace calculate_expression_l464_46458

theorem calculate_expression : (Real.sqrt 8 + Real.sqrt (1 / 2)) * Real.sqrt 32 = 20 := by
  sorry

end calculate_expression_l464_46458


namespace initial_mean_l464_46457

theorem initial_mean (M : ℝ) (h1 : 50 * (36.5 : ℝ) - 23 = 50 * (36.04 : ℝ) + 23)
: M = 36.04 :=
by
  sorry

end initial_mean_l464_46457


namespace solve_for_x_l464_46498

variables {A B C m n x : ℝ}

-- Existing conditions
def A_rate_condition : A = (B + C) / m := sorry
def B_rate_condition : B = (C + A) / n := sorry
def C_rate_condition : C = (A + B) / x := sorry

-- The theorem to be proven
theorem solve_for_x (A_rate_condition : A = (B + C) / m)
                    (B_rate_condition : B = (C + A) / n)
                    (C_rate_condition : C = (A + B) / x)
                    : x = (2 + m + n) / (m * n - 1) := by
  sorry

end solve_for_x_l464_46498


namespace probability_A_level_l464_46455

theorem probability_A_level (p_B : ℝ) (p_C : ℝ) (h_B : p_B = 0.03) (h_C : p_C = 0.01) : 
  (1 - (p_B + p_C)) = 0.96 :=
by
  -- Proof is omitted
  sorry

end probability_A_level_l464_46455


namespace books_sold_correct_l464_46483

-- Definitions of the conditions
def initial_books : ℕ := 33
def remaining_books : ℕ := 7
def books_sold : ℕ := initial_books - remaining_books

-- The statement to be proven (with proof omitted)
theorem books_sold_correct : books_sold = 26 := by
  -- Proof omitted
  sorry

end books_sold_correct_l464_46483


namespace arithmetic_sequence_sum_l464_46460

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 = 2) (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l464_46460


namespace intersection_of_sets_l464_46426

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 0 < x }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by sorry

end intersection_of_sets_l464_46426


namespace equation_has_one_real_root_l464_46402

noncomputable def f (x : ℝ) : ℝ :=
  (3 / 11)^x + (5 / 11)^x + (7 / 11)^x - 1

theorem equation_has_one_real_root :
  ∃! x : ℝ, f x = 0 := sorry

end equation_has_one_real_root_l464_46402


namespace complete_the_square_l464_46442

theorem complete_the_square (x : ℝ) :
  (x^2 + 8*x + 9 = 0) ↔ ((x + 4)^2 = 7) :=
sorry

end complete_the_square_l464_46442


namespace manage_committee_combination_l464_46407

theorem manage_committee_combination : (Nat.choose 20 3) = 1140 := by
  sorry

end manage_committee_combination_l464_46407


namespace cakes_remaining_l464_46453

theorem cakes_remaining (initial_cakes sold_cakes remaining_cakes: ℕ) (h₀ : initial_cakes = 167) (h₁ : sold_cakes = 108) (h₂ : remaining_cakes = initial_cakes - sold_cakes) : remaining_cakes = 59 :=
by
  rw [h₀, h₁] at h₂
  exact h₂

end cakes_remaining_l464_46453


namespace positive_quadratic_if_and_only_if_l464_46487

variable (a : ℝ)
def p (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem positive_quadratic_if_and_only_if (h : ∀ x : ℝ, p a x > 0) : a > 1 := sorry

end positive_quadratic_if_and_only_if_l464_46487


namespace number_of_yellow_balls_l464_46469

-- Definitions based on conditions
def number_of_red_balls : ℕ := 10
def probability_red_ball := (1 : ℚ) / 3

-- Theorem stating the number of yellow balls
theorem number_of_yellow_balls :
  ∃ (y : ℕ), (number_of_red_balls : ℚ) / (number_of_red_balls + y) = probability_red_ball ∧ y = 20 :=
by
  sorry

end number_of_yellow_balls_l464_46469


namespace carson_air_per_pump_l464_46440

-- Define the conditions
def total_air_needed : ℝ := 2 * 500 + 0.6 * 500 + 0.3 * 500

def total_pumps : ℕ := 29

-- Proof problem statement
theorem carson_air_per_pump : total_air_needed / total_pumps = 50 := by
  sorry

end carson_air_per_pump_l464_46440


namespace max_full_box_cards_l464_46400

-- Given conditions
def total_cards : ℕ := 94
def unfilled_box_cards : ℕ := 6

-- Define the number of cards that are evenly distributed into full boxes
def evenly_distributed_cards : ℕ := total_cards - unfilled_box_cards

-- Prove that the maximum number of cards a full box can hold is 22
theorem max_full_box_cards (h : evenly_distributed_cards = 88) : ∃ x : ℕ, evenly_distributed_cards % x = 0 ∧ x = 22 :=
by 
  -- Proof goes here
  sorry

end max_full_box_cards_l464_46400


namespace isosceles_triangle_area_l464_46405

-- Define the conditions for the isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop := a = b ∨ b = c ∨ a = c 

-- Define the side lengths
def side_length_1 : ℝ := 15
def side_length_2 : ℝ := 15
def side_length_3 : ℝ := 24

-- State the theorem
theorem isosceles_triangle_area :
  is_isosceles_triangle side_length_1 side_length_2 side_length_3 →
  side_length_1 = 15 →
  side_length_2 = 15 →
  side_length_3 = 24 →
  ∃ A : ℝ, (A = (1 / 2) * 24 * 9) ∧ A = 108 :=
sorry

end isosceles_triangle_area_l464_46405


namespace max_pens_min_pens_l464_46423

def pen_prices : List ℕ := [2, 3, 4]
def total_money : ℕ := 31

/-- Given the conditions of the problem, prove the maximum number of pens -/
theorem max_pens  (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 14 := by
  sorry

/-- Given the conditions of the problem, prove the minimum number of pens -/
theorem min_pens (hx : 31 = total_money) 
  (ha : pen_prices = [2, 3, 4])
  (at_least_one : ∀ p ∈ pen_prices, 1 ≤ p) :
  exists n : ℕ, n = 9 := by
  sorry

end max_pens_min_pens_l464_46423


namespace revolutions_same_distance_l464_46476

theorem revolutions_same_distance (r R : ℝ) (revs_30 : ℝ) (dist_30 dist_10 : ℝ)
  (h_radius: r = 10) (H_radius: R = 30) (h_revs_30: revs_30 = 15) 
  (H_dist_30: dist_30 = 2 * Real.pi * R * revs_30) 
  (H_dist_10: dist_10 = 2 * Real.pi * r * 45) :
  dist_30 = dist_10 :=
by {
  sorry
}

end revolutions_same_distance_l464_46476


namespace range_of_heights_l464_46438

theorem range_of_heights (max_height min_height : ℝ) (h_max : max_height = 175) (h_min : min_height = 100) :
  (max_height - min_height) = 75 :=
by
  -- Defer proof
  sorry

end range_of_heights_l464_46438


namespace range_of_a_minus_b_l464_46431

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 2) (h₃ : -2 < b) (h₄ : b < 1) :
  -2 < a - b ∧ a - b < 4 :=
by
  sorry

end range_of_a_minus_b_l464_46431


namespace total_points_scored_l464_46452

theorem total_points_scored (layla_score nahima_score : ℕ)
  (h1 : layla_score = 70)
  (h2 : layla_score = nahima_score + 28) :
  layla_score + nahima_score = 112 :=
by
  sorry

end total_points_scored_l464_46452


namespace speeds_of_bus_and_car_l464_46412

theorem speeds_of_bus_and_car
  (d t : ℝ) (v1 v2 : ℝ)
  (h1 : 1.5 * v1 + 1.5 * v2 = d)
  (h2 : 2.5 * v1 + 1 * v2 = d) :
  v1 = 40 ∧ v2 = 80 :=
by sorry

end speeds_of_bus_and_car_l464_46412


namespace parabola_intersection_radius_sqr_l464_46494

theorem parabola_intersection_radius_sqr {x y : ℝ} :
  (y = (x - 2)^2) →
  (x - 3 = (y + 2)^2) →
  ∃ r, r^2 = 9 / 2 :=
by
  intros h1 h2
  sorry

end parabola_intersection_radius_sqr_l464_46494


namespace total_classic_books_l464_46418

-- Definitions for the conditions
def authors := 6
def books_per_author := 33

-- Statement of the math proof problem
theorem total_classic_books : authors * books_per_author = 198 := by
  sorry  -- Proof to be filled in

end total_classic_books_l464_46418


namespace cos_half_angle_l464_46434

open Real

theorem cos_half_angle (α : ℝ) (h_sin : sin α = (4 / 9) * sqrt 2) (h_obtuse : π / 2 < α ∧ α < π) :
  cos (α / 2) = 1 / 3 :=
by
  sorry

end cos_half_angle_l464_46434


namespace boys_count_in_dance_class_l464_46403

theorem boys_count_in_dance_class
  (total_students : ℕ) 
  (ratio_girls_to_boys : ℕ) 
  (ratio_boys_to_girls: ℕ)
  (total_students_eq : total_students = 35)
  (ratio_eq : ratio_girls_to_boys = 3 ∧ ratio_boys_to_girls = 4) : 
  ∃ boys : ℕ, boys = 20 :=
by
  let k := total_students / (ratio_girls_to_boys + ratio_boys_to_girls)
  have girls := ratio_girls_to_boys * k
  have boys := ratio_boys_to_girls * k
  use boys
  sorry

end boys_count_in_dance_class_l464_46403


namespace ball_hits_ground_approx_time_l464_46470

noncomputable def ball_hits_ground_time (t : ℝ) : ℝ :=
-6 * t^2 - 12 * t + 60

theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, |t - 2.32| < 0.01 ∧ ball_hits_ground_time t = 0 :=
sorry

end ball_hits_ground_approx_time_l464_46470


namespace primes_in_arithmetic_sequence_l464_46447

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_in_arithmetic_sequence (p : ℕ) :
  is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  intro h
  sorry

end primes_in_arithmetic_sequence_l464_46447


namespace apples_distribution_l464_46448

variable (p b t : ℕ)

theorem apples_distribution (p_eq : p = 40) (b_eq : b = p + 8) (t_eq : t = (3 * b) / 8) :
  t = 18 := by
  sorry

end apples_distribution_l464_46448


namespace trigonometric_product_identity_l464_46425

theorem trigonometric_product_identity : 
  let cos_40 : Real := Real.cos (Real.pi * 40 / 180)
  let sin_40 : Real := Real.sin (Real.pi * 40 / 180)
  let cos_50 : Real := Real.cos (Real.pi * 50 / 180)
  let sin_50 : Real := Real.sin (Real.pi * 50 / 180)
  (sin_50 = cos_40) → (cos_50 = sin_40) →
  (1 - cos_40⁻¹) * (1 + sin_50⁻¹) * (1 - sin_40⁻¹) * (1 + cos_50⁻¹) = 1 := by
  sorry

end trigonometric_product_identity_l464_46425


namespace ratio_boys_to_girls_l464_46422

-- Define the given conditions
def G : ℕ := 300
def T : ℕ := 780

-- State the proposition to be proven
theorem ratio_boys_to_girls (B : ℕ) (h : B + G = T) : B / G = 8 / 5 :=
by
  -- Proof placeholder
  sorry

end ratio_boys_to_girls_l464_46422


namespace exists_xy_for_cube_difference_l464_46419

theorem exists_xy_for_cube_difference (a : ℕ) (h : 0 < a) :
  ∃ x y : ℤ, x^2 - y^2 = a^3 :=
sorry

end exists_xy_for_cube_difference_l464_46419


namespace simplify_expression_l464_46404

variable (a b : ℝ)

theorem simplify_expression :
  (a^3 - b^3) / (a * b) - (ab - b^2) / (ab - a^3) = (a^2 + ab + b^2) / b :=
by
  sorry

end simplify_expression_l464_46404


namespace find_a_l464_46449

-- Define the function f based on the given conditions
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then 2^x - a * x else -2^(-x) - a * x

-- Define the fact that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = -f (-x)

-- State the main theorem that needs to be proven
theorem find_a (a : ℝ) :
  (is_odd_function (f a)) ∧ (f a 2 = 2) → a = -9 / 8 :=
by
  sorry

end find_a_l464_46449


namespace convex_polyhedron_same_number_of_sides_l464_46401

theorem convex_polyhedron_same_number_of_sides {N : ℕ} (hN : N ≥ 4): 
  ∃ (f1 f2 : ℕ), (f1 >= 3 ∧ f1 < N ∧ f2 >= 3 ∧ f2 < N) ∧ f1 = f2 :=
by
  sorry

end convex_polyhedron_same_number_of_sides_l464_46401


namespace second_number_is_30_l464_46492

theorem second_number_is_30 
  (A B C : ℝ)
  (h1 : A + B + C = 98)
  (h2 : A / B = 2 / 3)
  (h3 : B / C = 5 / 8) : 
  B = 30 :=
by
  sorry

end second_number_is_30_l464_46492


namespace kaylin_age_32_l464_46474

-- Defining the ages of the individuals as variables
variables (Kaylin Sarah Eli Freyja Alfred Olivia : ℝ)

-- Defining the given conditions
def conditions : Prop := 
  (Kaylin = Sarah - 5) ∧
  (Sarah = 2 * Eli) ∧
  (Eli = Freyja + 9) ∧
  (Freyja = 2.5 * Alfred) ∧
  (Alfred = (3/4) * Olivia) ∧
  (Freyja = 9.5)

-- Main statement to prove
theorem kaylin_age_32 (h : conditions Kaylin Sarah Eli Freyja Alfred Olivia) : Kaylin = 32 :=
by
  sorry

end kaylin_age_32_l464_46474


namespace dealership_truck_sales_l464_46414

theorem dealership_truck_sales (SUVs Trucks : ℕ) (h1 : SUVs = 45) (h2 : 3 * Trucks = 5 * SUVs) : Trucks = 75 :=
by
  sorry

end dealership_truck_sales_l464_46414


namespace inequality_proof_l464_46450

theorem inequality_proof (m n : ℝ) (h₁ : m > 0) (h₂ : n > 0) (h₃ : m + n = 1) :
  (m + 1/m) * (n + 1/n) ≥ 25/4 :=
by
  sorry

end inequality_proof_l464_46450


namespace find_third_number_l464_46428

theorem find_third_number (x : ℕ) (h : (6 + 16 + x) / 3 = 13) : x = 17 :=
by
  sorry

end find_third_number_l464_46428


namespace solve_for_y_l464_46463

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem solve_for_y (y : ℝ) : star 2 y = 10 → y = 0 := by
  intro h
  sorry

end solve_for_y_l464_46463


namespace min_max_values_l464_46472

noncomputable def expression (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  ( (x₁ ^ 2 / x₂) + (x₂ ^ 2 / x₃) + (x₃ ^ 2 / x₄) + (x₄ ^ 2 / x₁) ) /
  ( x₁ + x₂ + x₃ + x₄ )

theorem min_max_values
  (a b : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (h₀ : 0 < a) (h₁ : a < b)
  (h₂ : a ≤ x₁) (h₃ : x₁ ≤ b)
  (h₄ : a ≤ x₂) (h₅ : x₂ ≤ b)
  (h₆ : a ≤ x₃) (h₇ : x₃ ≤ b)
  (h₈ : a ≤ x₄) (h₉ : x₄ ≤ b) :
  expression x₁ x₂ x₃ x₄ ≥ 1 / b ∧ expression x₁ x₂ x₃ x₄ ≤ 1 / a :=
  sorry

end min_max_values_l464_46472


namespace Sam_has_correct_amount_of_dimes_l464_46489

-- Definitions for initial values and transactions
def initial_dimes := 9
def dimes_from_dad := 7
def dimes_taken_by_mom := 3
def sets_from_sister := 4
def dimes_per_set := 2

-- Definition of the total dimes Sam has now
def total_dimes_now : Nat :=
  initial_dimes + dimes_from_dad - dimes_taken_by_mom + (sets_from_sister * dimes_per_set)

-- Proof statement
theorem Sam_has_correct_amount_of_dimes : total_dimes_now = 21 := by
  sorry

end Sam_has_correct_amount_of_dimes_l464_46489


namespace edge_length_of_cube_l464_46497

theorem edge_length_of_cube (total_cubes : ℕ) (box_edge_length_m : ℝ) (box_edge_length_cm : ℝ) 
  (conversion_factor : ℝ) (edge_length_cm : ℝ) : 
  total_cubes = 8 ∧ box_edge_length_m = 1 ∧ box_edge_length_cm = box_edge_length_m * conversion_factor ∧ conversion_factor = 100 ∧ 
  edge_length_cm = box_edge_length_cm / 2 ↔ edge_length_cm = 50 := 
by 
  sorry

end edge_length_of_cube_l464_46497


namespace bruce_money_left_to_buy_more_clothes_l464_46441

def calculate_remaining_money 
  (amount_given : ℝ) 
  (shirt_price : ℝ) (num_shirts : ℕ)
  (pants_price : ℝ)
  (sock_price : ℝ) (num_socks : ℕ)
  (belt_original_price : ℝ) (belt_discount : ℝ)
  (total_discount : ℝ) : ℝ := 
let shirts_cost := shirt_price * num_shirts
let socks_cost := sock_price * num_socks
let belt_price := belt_original_price * (1 - belt_discount)
let total_cost := shirts_cost + pants_price + socks_cost + belt_price
let discount_cost := total_cost * total_discount
let final_cost := total_cost - discount_cost
amount_given - final_cost

theorem bruce_money_left_to_buy_more_clothes 
  : calculate_remaining_money 71 5 5 26 3 2 12 0.25 0.10 = 11.60 := 
by
  sorry

end bruce_money_left_to_buy_more_clothes_l464_46441


namespace remaining_soup_feeds_adults_l464_46443

theorem remaining_soup_feeds_adults (C A k c : ℕ) 
    (hC : C= 10) 
    (hA : A = 5) 
    (hk : k = 8) 
    (hc : c = 20) : k - c / C * 10 * A = 30 := sorry

end remaining_soup_feeds_adults_l464_46443


namespace angle_same_terminal_side_l464_46490

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, -330 = k * 360 + 30 :=
by
  use -1
  sorry

end angle_same_terminal_side_l464_46490


namespace total_candies_l464_46468

def candies_in_boxes (num_boxes: Nat) (pieces_per_box: Nat) : Nat :=
  num_boxes * pieces_per_box

theorem total_candies :
  candies_in_boxes 3 6 + candies_in_boxes 5 8 + candies_in_boxes 4 10 = 98 := by
  sorry

end total_candies_l464_46468


namespace center_of_circle_l464_46446

-- Define the circle in polar coordinates
def circle_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center of the circle in polar coordinates
def center_polar (ρ θ : ℝ) : Prop := (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- The theorem states that the center of the given circle in polar coordinates is (1, π/2) or (1, 3π/2)
theorem center_of_circle : ∃ (ρ θ : ℝ), circle_polar ρ θ → center_polar ρ θ :=
by
  -- The center of the circle given the condition in polar coordinate system is (1, π/2) or (1, 3π/2)
  sorry

end center_of_circle_l464_46446


namespace living_space_increase_l464_46475

theorem living_space_increase (a b x : ℝ) (h₁ : a = 10) (h₂ : b = 12.1) : a * (1 + x) ^ 2 = b :=
sorry

end living_space_increase_l464_46475


namespace simplify_product_l464_46439

theorem simplify_product (x t : ℕ) : (x^2 * t^3) * (x^3 * t^4) = (x^5) * (t^7) := 
by 
  sorry

end simplify_product_l464_46439


namespace intersection_point_of_lines_l464_46486

theorem intersection_point_of_lines : 
  ∃ (x y : ℝ), (x - 4 * y - 1 = 0) ∧ (2 * x + y - 2 = 0) ∧ (x = 1) ∧ (y = 0) :=
by
  sorry

end intersection_point_of_lines_l464_46486


namespace repetitive_decimals_subtraction_correct_l464_46411

noncomputable def repetitive_decimals_subtraction : Prop :=
  let a : ℚ := 4567 / 9999
  let b : ℚ := 1234 / 9999
  let c : ℚ := 2345 / 9999
  a - b - c = 988 / 9999

theorem repetitive_decimals_subtraction_correct : repetitive_decimals_subtraction :=
  by sorry

end repetitive_decimals_subtraction_correct_l464_46411


namespace range_of_a_l464_46485

-- Defining the function f(x) = x^2 + 2ax - 1
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - 1

-- Conditions: x1, x2 ∈ [1, +∞) and x1 < x2
variables (x1 x2 a : ℝ)
variables (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2)

-- Statement of the proof problem:
theorem range_of_a (hf_ineq : x2 * f x1 a - x1 * f x2 a < a * (x1 - x2)) : a ≤ 2 :=
sorry

end range_of_a_l464_46485


namespace exists_real_k_l464_46480

theorem exists_real_k (c : Fin 1998 → ℕ)
  (h1 : 0 ≤ c 1)
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → m + n < 1998 → c m + c n ≤ c (m + n) ∧ c (m + n) ≤ c m + c n + 1) :
  ∃ k : ℝ, ∀ n : Fin 1998, 1 ≤ n → c n = Int.floor (n * k) :=
by
  sorry

end exists_real_k_l464_46480


namespace total_cards_correct_l464_46479

-- Define the number of dozens each person has
def dozens_per_person : Nat := 9

-- Define the number of cards per dozen
def cards_per_dozen : Nat := 12

-- Define the number of people
def num_people : Nat := 4

-- Define the total number of Pokemon cards in all
def total_cards : Nat := dozens_per_person * cards_per_dozen * num_people

-- The statement to be proved
theorem total_cards_correct : total_cards = 432 := 
by 
  -- Proof omitted as requested
  sorry

end total_cards_correct_l464_46479


namespace geometric_sequence_sum_l464_46417

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (a1 : a 1 = 3)
  (a4 : a 4 = 24)
  (h_geo : ∃ q : ℝ, ∀ n : ℕ, a n = 3 * q^(n - 1)) :
  a 3 + a 4 + a 5 = 84 :=
by
  sorry

end geometric_sequence_sum_l464_46417


namespace find_a_and_b_l464_46464

theorem find_a_and_b :
  ∃ a b : ℝ, 
    (∀ x : ℝ, (x^3 + 3*x^2 + 2*x > 0) ↔ (x > 0 ∨ -2 < x ∧ x < -1)) ∧
    (∀ x : ℝ, (x^2 + a*x + b ≤ 0) ↔ (-2 < x ∧ x ≤ 0 ∨ 0 < x ∧ x ≤ 2)) ∧ 
    a = -1 ∧ b = -2 := 
  sorry

end find_a_and_b_l464_46464


namespace play_number_of_children_l464_46415

theorem play_number_of_children (A C : ℕ) (ticket_price_adult : ℕ) (ticket_price_child : ℕ)
    (total_people : ℕ) (total_money : ℕ)
    (h1 : ticket_price_adult = 8)
    (h2 : ticket_price_child = 1)
    (h3 : total_people = 22)
    (h4 : total_money = 50)
    (h5 : A + C = total_people)
    (h6 : ticket_price_adult * A + ticket_price_child * C = total_money) :
    C = 18 := sorry

end play_number_of_children_l464_46415


namespace volume_tetrahedron_constant_l464_46408

theorem volume_tetrahedron_constant (m n h : ℝ) (ϕ : ℝ) :
  ∃ V : ℝ, V = (1 / 6) * m * n * h * Real.sin ϕ :=
by
  sorry

end volume_tetrahedron_constant_l464_46408


namespace cos_alpha_value_l464_46406

theorem cos_alpha_value (α β γ: ℝ) (h1: β = 2 * α) (h2: γ = 4 * α)
 (h3: 2 * (Real.sin β) = (Real.sin α + Real.sin γ)) : Real.cos α = -1/2 := 
by
  sorry

end cos_alpha_value_l464_46406


namespace jensen_meetings_percentage_l464_46496

theorem jensen_meetings_percentage :
  ∃ (first second third total_work_day total_meeting_time : ℕ),
    total_work_day = 600 ∧
    first = 35 ∧
    second = 2 * first ∧
    third = first + second ∧
    total_meeting_time = first + second + third ∧
    (total_meeting_time * 100) / total_work_day = 35 := sorry

end jensen_meetings_percentage_l464_46496


namespace sum_of_other_endpoint_l464_46436

theorem sum_of_other_endpoint (x y : ℝ) :
  (10, -6) = ((x + 12) / 2, (y + 4) / 2) → x + y = -8 :=
by
  sorry

end sum_of_other_endpoint_l464_46436


namespace initial_investment_l464_46416

theorem initial_investment (P r : ℝ) 
  (h1 : 600 = P * (1 + 0.02 * r)) 
  (h2 : 850 = P * (1 + 0.07 * r)) : 
  P = 500 :=
sorry

end initial_investment_l464_46416


namespace complement_union_complement_intersection_l464_46462

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem complement_union (A B : Set ℝ) :
  (A ∪ B)ᶜ = { x : ℝ | x ≤ 2 ∨ x ≥ 10 } :=
by
  sorry

theorem complement_intersection (A B : Set ℝ) :
  (Aᶜ ∩ B) = { x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } :=
by
  sorry

end complement_union_complement_intersection_l464_46462


namespace remainder_zero_l464_46444

theorem remainder_zero (x : ℂ) 
  (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) : 
  x^55 + x^44 + x^33 + x^22 + x^11 + 1 = 0 := 
by 
  sorry

end remainder_zero_l464_46444


namespace total_hours_for_songs_l464_46482

def total_hours_worked_per_day := 10
def total_days_per_song := 10
def number_of_songs := 3

theorem total_hours_for_songs :
  total_hours_worked_per_day * total_days_per_song * number_of_songs = 300 :=
by
  sorry

end total_hours_for_songs_l464_46482


namespace find_A_l464_46409

def hash_relation (A B : ℕ) : ℕ := A^2 + B^2

theorem find_A (A : ℕ) (h1 : hash_relation A 7 = 218) : A = 13 := 
by sorry

end find_A_l464_46409


namespace james_weekly_earnings_l464_46499

def rate_per_hour : ℕ := 20
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 4

def daily_earnings : ℕ := rate_per_hour * hours_per_day
def weekly_earnings : ℕ := daily_earnings * days_per_week

theorem james_weekly_earnings : weekly_earnings = 640 := sorry

end james_weekly_earnings_l464_46499


namespace initial_number_of_persons_l464_46488

-- Define the conditions and the goal
def weight_increase_due_to_new_person : ℝ := 102 - 75
def average_weight_increase (n : ℝ) : ℝ := 4.5 * n

theorem initial_number_of_persons (n : ℝ) (h1 : average_weight_increase n = weight_increase_due_to_new_person) : n = 6 :=
by
  -- Skip the proof with sorry
  sorry

end initial_number_of_persons_l464_46488


namespace find_n_cubes_l464_46410

theorem find_n_cubes (n : ℕ) (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h1 : 837 + n = y^3) (h2 : 837 - n = x^3) : n = 494 :=
by {
  sorry
}

end find_n_cubes_l464_46410


namespace no_integer_solutions_l464_46420

theorem no_integer_solutions :
  ¬ (∃ a b : ℤ, 3 * a^2 = b^2 + 1) :=
by 
  sorry

end no_integer_solutions_l464_46420


namespace tomatoes_sold_to_mr_wilson_l464_46454

theorem tomatoes_sold_to_mr_wilson :
  let T := 245.5
  let S_m := 125.5
  let N := 42
  let S_w := T - S_m - N
  S_w = 78 := 
by
  sorry

end tomatoes_sold_to_mr_wilson_l464_46454


namespace sum_of_digits_divisible_by_9_l464_46424

theorem sum_of_digits_divisible_by_9 (N : ℕ) (a b c : ℕ) (hN : N < 10^1962)
  (h1 : N % 9 = 0)
  (ha : a = (N.digits 10).sum)
  (hb : b = (a.digits 10).sum)
  (hc : c = (b.digits 10).sum) :
  c = 9 :=
sorry

end sum_of_digits_divisible_by_9_l464_46424


namespace scientific_notation_of_22nm_l464_46427

theorem scientific_notation_of_22nm (h : 22 * 10^(-9) = 0.000000022) : 0.000000022 = 2.2 * 10^(-8) :=
sorry

end scientific_notation_of_22nm_l464_46427


namespace person_B_completion_time_l464_46473

variables {A B : ℝ} (H : A + B = 1/6 ∧ (A + 10 * B = 1/6))

theorem person_B_completion_time :
    (1 / (1 - 2 * (A + B)) / B = 15) :=
by
  sorry

end person_B_completion_time_l464_46473


namespace zan_guo_gets_one_deer_l464_46481

noncomputable def a1 : ℚ := 5 / 3
noncomputable def sum_of_sequence (a1 : ℚ) (d : ℚ) : ℚ := 5 * a1 + (5 * 4 / 2) * d
noncomputable def d : ℚ := -1 / 3
noncomputable def a3 (a1 : ℚ) (d : ℚ) : ℚ := a1 + 2 * d

theorem zan_guo_gets_one_deer :
  a3 a1 d = 1 := by
  sorry

end zan_guo_gets_one_deer_l464_46481


namespace greatest_three_digit_multiple_of_17_is_986_l464_46437

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_multiple_of_17 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 17 * k

def greatest_three_digit_multiple_of_17 : ℕ :=
  986

theorem greatest_three_digit_multiple_of_17_is_986 :
  ∀ n : ℕ, is_three_digit_number n → is_multiple_of_17 n → n ≤ greatest_three_digit_multiple_of_17 :=
by
  sorry

end greatest_three_digit_multiple_of_17_is_986_l464_46437


namespace factorization_pq_difference_l464_46456

theorem factorization_pq_difference :
  ∃ (p q : ℤ), 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q) ∧ p - q = 36 := by
-- Given the conditions in the problem,
-- We assume ∃ integers p and q such that (5x + p)(5x + q) = 25x² - 135x - 150 and derive the difference p - q = 36.
  sorry

end factorization_pq_difference_l464_46456


namespace product_of_areas_eq_square_of_volume_l464_46459

variable (x y z : ℝ)

def area_xy : ℝ := x * y
def area_yz : ℝ := y * z
def area_zx : ℝ := z * x

theorem product_of_areas_eq_square_of_volume :
  (area_xy x y) * (area_yz y z) * (area_zx z x) = (x * y * z) ^ 2 :=
by
  sorry

end product_of_areas_eq_square_of_volume_l464_46459


namespace intersection_S_T_l464_46478

def S : Set ℝ := { x | (x - 2) * (x - 3) >= 0 }
def T : Set ℝ := { x | x > 0 }

theorem intersection_S_T :
  S ∩ T = { x | (0 < x ∧ x <= 2) ∨ (x >= 3) } := by
  sorry

end intersection_S_T_l464_46478


namespace fixed_point_always_on_line_l464_46467

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line_l464_46467


namespace probability_first_or_second_l464_46430

/-- Define the events and their probabilities --/
def prob_hit_first_sector : ℝ := 0.4
def prob_hit_second_sector : ℝ := 0.3
def prob_hit_first_or_second : ℝ := 0.7

/-- The proof that these probabilities add up as mutually exclusive events --/
theorem probability_first_or_second (P_A : ℝ) (P_B : ℝ) (P_A_or_B : ℝ) (hP_A : P_A = prob_hit_first_sector) (hP_B : P_B = prob_hit_second_sector) (hP_A_or_B : P_A_or_B = prob_hit_first_or_second) :
  P_A_or_B = P_A + P_B := 
  by
    rw [hP_A, hP_B, hP_A_or_B]
    sorry

end probability_first_or_second_l464_46430


namespace cookie_combinations_l464_46413

theorem cookie_combinations (total_cookies kinds : Nat) (at_least_one : kinds > 0 ∧ ∀ k : Nat, k < kinds → k > 0) : 
  (total_cookies = 8 ∧ kinds = 4) → 
  (∃ comb : Nat, comb = 34) := 
by 
  -- insert proof here 
  sorry

end cookie_combinations_l464_46413


namespace a_4_eq_28_l464_46432

def Sn (n : ℕ) : ℕ := 4 * n^2

def a_n (n : ℕ) : ℕ := Sn n - Sn (n - 1)

theorem a_4_eq_28 : a_n 4 = 28 :=
by
  sorry

end a_4_eq_28_l464_46432


namespace investmentAmounts_l464_46484

variable (totalInvestment : ℝ) (bonds stocks mutualFunds : ℝ)

-- Given conditions
def conditions := 
  totalInvestment = 210000 ∧
  stocks = 2 * bonds ∧
  mutualFunds = 4 * stocks ∧
  bonds + stocks + mutualFunds = totalInvestment

-- Prove the investments
theorem investmentAmounts (h : conditions totalInvestment bonds stocks mutualFunds) :
  bonds = 19090.91 ∧ stocks = 38181.82 ∧ mutualFunds = 152727.27 :=
sorry

end investmentAmounts_l464_46484


namespace triangle_inequality_property_l464_46451

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  (a * b * c) / (4 * Real.sqrt (A * B * C))

noncomputable def inradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  Real.sqrt (A * B * C) * perimeter a b c

theorem triangle_inequality_property (a b c A B C : ℝ)
  (h₁ : ∀ {x}, x > 0)
  (h₂ : A ≠ B)
  (h₃ : B ≠ C)
  (h₄ : C ≠ A) :
  ¬ (perimeter a b c ≤ circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c > circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c / 6 < circumradius a b c A B C + inradius a b c A B C ∨ 
  circumradius a b c A B C + inradius a b c A B C < 6 * perimeter a b c) :=
sorry

end triangle_inequality_property_l464_46451


namespace simplify_expression_l464_46433

-- Defining each term in the sum
def term1  := 1 / ((1 / 3) ^ 1)
def term2  := 1 / ((1 / 3) ^ 2)
def term3  := 1 / ((1 / 3) ^ 3)

-- Sum of the terms
def terms_sum := term1 + term2 + term3

-- The simplification proof statement
theorem simplify_expression : 1 / terms_sum = 1 / 39 :=
by sorry

end simplify_expression_l464_46433


namespace angle_RBC_10_degrees_l464_46477

noncomputable def compute_angle_RBC (angle_BRA angle_BAC angle_ABC : ℝ) : ℝ :=
  let angle_RBA := 180 - angle_BRA - angle_BAC
  angle_RBA - angle_ABC

theorem angle_RBC_10_degrees :
  ∀ (angle_BRA angle_BAC angle_ABC : ℝ), 
    angle_BRA = 72 → angle_BAC = 43 → angle_ABC = 55 → 
    compute_angle_RBC angle_BRA angle_BAC angle_ABC = 10 :=
by
  intros
  unfold compute_angle_RBC
  sorry

end angle_RBC_10_degrees_l464_46477


namespace sin_product_l464_46445

theorem sin_product (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.sin (π / 2 - α) = 2 / 5 :=
by
  -- proof shorter placeholder
  sorry

end sin_product_l464_46445


namespace smallest_prime_linear_pair_l464_46466

def is_prime (n : ℕ) : Prop := ¬(∃ k > 1, k < n ∧ k ∣ n)

theorem smallest_prime_linear_pair :
  ∃ a b : ℕ, is_prime a ∧ is_prime b ∧ a + b = 180 ∧ a > b ∧ b = 7 := 
by
  sorry

end smallest_prime_linear_pair_l464_46466


namespace product_of_x_y_l464_46421

theorem product_of_x_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) : x * y = 72 :=
by
  sorry

end product_of_x_y_l464_46421
