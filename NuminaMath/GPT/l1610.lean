import Mathlib

namespace johns_profit_l1610_161057

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end johns_profit_l1610_161057


namespace domain_of_f_l1610_161078

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l1610_161078


namespace find_positive_n_l1610_161089

theorem find_positive_n (n x : ℝ) (h : 16 * x ^ 2 + n * x + 4 = 0) : n = 16 :=
by
  sorry

end find_positive_n_l1610_161089


namespace find_rate_percent_l1610_161001

theorem find_rate_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : 2420 = P * (1 + r / 100)^2) 
  (h2 : 3025 = P * (1 + r / 100)^3) : 
  r = 25 :=
by
  sorry

end find_rate_percent_l1610_161001


namespace bicycle_helmet_savings_l1610_161092

theorem bicycle_helmet_savings :
  let bicycle_regular_price := 320
  let bicycle_discount := 0.2
  let helmet_regular_price := 80
  let helmet_discount := 0.1
  let bicycle_savings := bicycle_regular_price * bicycle_discount
  let helmet_savings := helmet_regular_price * helmet_discount
  let total_savings := bicycle_savings + helmet_savings
  let total_regular_price := bicycle_regular_price + helmet_regular_price
  let percentage_savings := (total_savings / total_regular_price) * 100
  percentage_savings = 18 := 
by sorry

end bicycle_helmet_savings_l1610_161092


namespace opposite_of_pi_l1610_161085

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end opposite_of_pi_l1610_161085


namespace f_no_zero_point_l1610_161088

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem f_no_zero_point (x : ℝ) (h : x > 0) : f x ≠ 0 :=
by 
  sorry

end f_no_zero_point_l1610_161088


namespace first_bakery_sacks_per_week_l1610_161065

theorem first_bakery_sacks_per_week (x : ℕ) 
    (H1 : 4 * x + 4 * 4 + 4 * 12 = 72) : x = 2 :=
by 
  -- we will provide the proof here if needed
  sorry

end first_bakery_sacks_per_week_l1610_161065


namespace remainder_div_13_l1610_161039

theorem remainder_div_13 {k : ℤ} (N : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 :=
by
  sorry

end remainder_div_13_l1610_161039


namespace jogging_problem_l1610_161037

theorem jogging_problem (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : ¬ ∃ p : ℕ, Prime p ∧ p^2 ∣ z) : 
  (x - y * Real.sqrt z) = 60 - 30 * Real.sqrt 2 → x + y + z = 92 :=
by
  intro h5
  have h6 : (60 - (60 - 30 * Real.sqrt 2))^2 = 1800 :=
    by sorry
  sorry

end jogging_problem_l1610_161037


namespace operation_8_to_cube_root_16_l1610_161002

theorem operation_8_to_cube_root_16 : ∃ (x : ℕ), x = 8 ∧ (x * x = (Nat.sqrt 16)^3) :=
by
  sorry

end operation_8_to_cube_root_16_l1610_161002


namespace solution_set_inequality_l1610_161091

theorem solution_set_inequality (x : ℝ) : 
  (∃ x, (x-1)/((x^2) - x - 30) > 0) ↔ (x > -5 ∧ x < 1) ∨ (x > 6) :=
by
  sorry

end solution_set_inequality_l1610_161091


namespace part1_part2_l1610_161024

def f (x a b : ℝ) : ℝ := |x - a| + |x - b|

theorem part1 (a b c x : ℝ) (h1 : |a - b| > c) : f x a b > c :=
  by sorry

theorem part2 (a : ℝ) (h1 : ∃ (x : ℝ), f x a 1 < 2 - |a - 2|) : 1/2 < a ∧ a < 5/2 :=
  by sorry

end part1_part2_l1610_161024


namespace stratified_sampling_l1610_161046

theorem stratified_sampling (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : (5 : ℚ) / 10 = 150 / n) : n = 300 :=
by
  sorry

end stratified_sampling_l1610_161046


namespace vector_problem_l1610_161009

noncomputable def t_value : ℝ :=
  (-5 - Real.sqrt 13) / 2

theorem vector_problem 
  (t : ℝ)
  (a : ℝ × ℝ := (1, 1))
  (b : ℝ × ℝ := (2, t))
  (h : Real.sqrt ((1 - 2)^2 + (1 - t)^2) = (1 * 2 + 1 * t)) :
  t = t_value := 
sorry

end vector_problem_l1610_161009


namespace translated_coordinates_of_B_l1610_161080

-- Definitions and conditions
def pointA : ℝ × ℝ := (-2, 3)

def translate_right (x : ℝ) (units : ℝ) : ℝ := x + units
def translate_down (y : ℝ) (units : ℝ) : ℝ := y - units

-- Theorem statement
theorem translated_coordinates_of_B :
  let Bx := translate_right (-2) 3
  let By := translate_down 3 5
  (Bx, By) = (1, -2) :=
by
  -- This is where the proof would go, but we're using sorry to skip the proof steps.
  sorry

end translated_coordinates_of_B_l1610_161080


namespace number_of_machines_l1610_161034

def machine_problem : Prop :=
  ∃ (m : ℕ), (6 * 42) = 6 * 36 ∧ m = 7

theorem number_of_machines : machine_problem :=
  sorry

end number_of_machines_l1610_161034


namespace find_custom_operator_result_l1610_161023

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l1610_161023


namespace non_fiction_vs_fiction_diff_l1610_161084

def total_books : Nat := 35
def fiction_books : Nat := 5
def picture_books : Nat := 11
def autobiography_books : Nat := 2 * fiction_books

def accounted_books : Nat := fiction_books + autobiography_books + picture_books
def non_fiction_books : Nat := total_books - accounted_books

theorem non_fiction_vs_fiction_diff :
  non_fiction_books - fiction_books = 4 := by 
  sorry

end non_fiction_vs_fiction_diff_l1610_161084


namespace cost_price_per_meter_l1610_161026

namespace ClothCost

theorem cost_price_per_meter (selling_price_total : ℝ) (meters_sold : ℕ) (loss_per_meter : ℝ) : 
  selling_price_total = 18000 → 
  meters_sold = 300 → 
  loss_per_meter = 5 →
  (selling_price_total / meters_sold) + loss_per_meter = 65 := 
by
  intros hsp hms hloss
  sorry

end ClothCost

end cost_price_per_meter_l1610_161026


namespace arithmetic_geometric_sequence_l1610_161048

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_common_diff : d = 2) (h_geom : a 2 ^ 2 = a 1 * a 5) : 
  a 2 = 3 :=
by
  sorry

end arithmetic_geometric_sequence_l1610_161048


namespace sum_remainders_eq_two_l1610_161049

theorem sum_remainders_eq_two (a b c : ℤ) (h_a : a % 24 = 10) (h_b : b % 24 = 4) (h_c : c % 24 = 12) :
  (a + b + c) % 24 = 2 :=
by
  sorry

end sum_remainders_eq_two_l1610_161049


namespace partial_fraction_decomposition_l1610_161038

theorem partial_fraction_decomposition (x : ℝ) :
  (5 * x - 3) / (x^2 - 5 * x - 14) = (32 / 9) / (x - 7) + (13 / 9) / (x + 2) := by
  sorry

end partial_fraction_decomposition_l1610_161038


namespace distance_from_Asheville_to_Darlington_l1610_161041

theorem distance_from_Asheville_to_Darlington (BC AC BD AD : ℝ) 
(h0 : BC = 12) 
(h1 : BC = (1/3) * AC) 
(h2 : BC = (1/4) * BD) :
AD = 72 :=
sorry

end distance_from_Asheville_to_Darlington_l1610_161041


namespace new_average_is_21_l1610_161014

def initial_number_of_students : ℕ := 30
def late_students : ℕ := 4
def initial_jumping_students : ℕ := initial_number_of_students - late_students
def initial_average_score : ℕ := 20
def late_student_scores : List ℕ := [26, 27, 28, 29]
def total_jumps_initial_students : ℕ := initial_jumping_students * initial_average_score
def total_jumps_late_students : ℕ := late_student_scores.sum
def total_jumps_all_students : ℕ := total_jumps_initial_students + total_jumps_late_students
def new_average_score : ℕ := total_jumps_all_students / initial_number_of_students

theorem new_average_is_21 :
  new_average_score = 21 :=
sorry

end new_average_is_21_l1610_161014


namespace intersection_points_are_integers_l1610_161060

theorem intersection_points_are_integers :
  ∀ (a b : Fin 2021 → ℕ), Function.Injective a → Function.Injective b →
  ∀ i j, i ≠ j → 
  ∃ x : ℤ, (∃ y : ℚ, y = (a i : ℚ) / (x + (b i : ℚ))) ∧ 
           (∃ y : ℚ, y = (a j : ℚ) / (x + (b j : ℚ))) := 
sorry

end intersection_points_are_integers_l1610_161060


namespace tan_sum_l1610_161079

-- Define the conditions as local variables
variables {α β : ℝ} (h₁ : Real.tan α = -2) (h₂ : Real.tan β = 5)

-- The statement to prove
theorem tan_sum : Real.tan (α + β) = 3 / 11 :=
by 
  -- Proof goes here, using 'sorry' as placeholder
  sorry

end tan_sum_l1610_161079


namespace matrix_unique_solution_l1610_161073

-- Definitions for the conditions given in the problem
def vec_i : Fin 3 → ℤ := ![1, 0, 0]
def vec_j : Fin 3 → ℤ := ![0, 1, 0]
def vec_k : Fin 3 → ℤ := ![0, 0, 1]

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![5, -3, 8],
  ![4, 6, -2],
  ![-9, 0, 5]
]

-- Define the target vectors
def target_i : Fin 3 → ℤ := ![5, 4, -9]
def target_j : Fin 3 → ℤ := ![-3, 6, 0]
def target_k : Fin 3 → ℤ := ![8, -2, 5]

-- The statement of the proof
theorem matrix_unique_solution : 
  (matrix_M.mulVec vec_i = target_i) ∧
  (matrix_M.mulVec vec_j = target_j) ∧
  (matrix_M.mulVec vec_k = target_k) :=
  by {
    sorry
  }

end matrix_unique_solution_l1610_161073


namespace expected_adjacent_black_pairs_60_cards_l1610_161018

noncomputable def expected_adjacent_black_pairs 
(deck_size : ℕ) (black_cards : ℕ) (red_cards : ℕ) : ℚ :=
  if h : deck_size = black_cards + red_cards 
  then (black_cards:ℚ) * (black_cards - 1) / (deck_size - 1) 
  else 0

theorem expected_adjacent_black_pairs_60_cards :
  expected_adjacent_black_pairs 60 36 24 = 1260 / 59 := by
  sorry

end expected_adjacent_black_pairs_60_cards_l1610_161018


namespace total_time_required_l1610_161004

noncomputable def walking_speed_flat : ℝ := 4
noncomputable def walking_speed_uphill : ℝ := walking_speed_flat * 0.8

noncomputable def running_speed_flat : ℝ := 8
noncomputable def running_speed_uphill : ℝ := running_speed_flat * 0.7

noncomputable def distance_walked_uphill : ℝ := 2
noncomputable def distance_run_uphill : ℝ := 1
noncomputable def distance_run_flat : ℝ := 1

noncomputable def time_walk_uphill := distance_walked_uphill / walking_speed_uphill
noncomputable def time_run_uphill := distance_run_uphill / running_speed_uphill
noncomputable def time_run_flat := distance_run_flat / running_speed_flat

noncomputable def total_time := time_walk_uphill + time_run_uphill + time_run_flat

theorem total_time_required :
  total_time = 0.9286 := by
  sorry

end total_time_required_l1610_161004


namespace friends_belong_special_team_l1610_161076

-- Define a type for students
universe u
variable {Student : Type u}

-- Assume a friendship relation among students
variable (friend : Student → Student → Prop)

-- Assume the conditions as given in the problem
variable (S : Student → Set (Set Student))
variable (students : Set Student)
variable (S_non_empty : ∀ v : Student, S v ≠ ∅)
variable (friendship_condition : 
  ∀ u v : Student, friend u v → 
    (∃ w : Student, S u ∩ S v ⊇ S w))
variable (special_team : ∀ (T : Set Student),
  (∃ v ∈ T, ∀ w : Student, w ∈ T → friend v w) ↔
  (∃ v ∈ T, ∀ w : Student, friend v w → w ∈ T))

-- Prove that any two friends belong to some special team
theorem friends_belong_special_team :
  ∀ u v : Student, friend u v → 
    (∃ T : Set Student, T ∈ S u ∩ S v ∧ 
      (∃ w ∈ T, ∀ x : Student, friend w x → x ∈ T)) :=
by
  sorry  -- Proof omitted


end friends_belong_special_team_l1610_161076


namespace variance_transformed_list_l1610_161061

noncomputable def stddev (xs : List ℝ) : ℝ := sorry
noncomputable def variance (xs : List ℝ) : ℝ := sorry

theorem variance_transformed_list :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℝ),
  stddev [a_1, a_2, a_3, a_4, a_5] = 2 →
  variance [3 * a_1 - 2, 3 * a_2 - 2, 3 * a_3 - 2, 3 * a_4 - 2, 3 * a_5 - 2] = 36 :=
by
  intros
  sorry

end variance_transformed_list_l1610_161061


namespace prove_fraction_identity_l1610_161010

theorem prove_fraction_identity (x y : ℂ) (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 := 
by 
  sorry

end prove_fraction_identity_l1610_161010


namespace john_total_distance_l1610_161051

theorem john_total_distance :
  let s₁ : ℝ := 45       -- Speed for the first part (mph)
  let t₁ : ℝ := 2        -- Time for the first part (hours)
  let s₂ : ℝ := 50       -- Speed for the second part (mph)
  let t₂ : ℝ := 3        -- Time for the second part (hours)
  let d₁ : ℝ := s₁ * t₁ -- Distance for the first part
  let d₂ : ℝ := s₂ * t₂ -- Distance for the second part
  d₁ + d₂ = 240          -- Total distance
:= by
  sorry

end john_total_distance_l1610_161051


namespace pet_store_cages_l1610_161012

theorem pet_store_cages (init_puppies sold_puppies puppies_per_cage : ℕ)
  (h1 : init_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (init_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l1610_161012


namespace greatest_possible_y_l1610_161077

theorem greatest_possible_y
  (x y : ℤ)
  (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 :=
sorry

end greatest_possible_y_l1610_161077


namespace find_ordered_pair_l1610_161000

theorem find_ordered_pair : ∃ (x y : ℚ), 
  3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ 
  x = 57 / 31 ∧ y = 195 / 62 :=
by {
  sorry
}

end find_ordered_pair_l1610_161000


namespace opposite_of_pi_eq_neg_pi_l1610_161028

theorem opposite_of_pi_eq_neg_pi (π : Real) (h : π = Real.pi) : -π = -Real.pi :=
by sorry

end opposite_of_pi_eq_neg_pi_l1610_161028


namespace elise_initial_money_l1610_161006

theorem elise_initial_money :
  ∃ (X : ℤ), X + 13 - 2 - 18 = 1 ∧ X = 8 :=
by
  sorry

end elise_initial_money_l1610_161006


namespace first_investment_percentage_l1610_161097

theorem first_investment_percentage :
  let total_inheritance := 4000
  let invested_6_5 := 1800
  let interest_rate_6_5 := 0.065
  let total_interest := 227
  let remaining_investment := total_inheritance - invested_6_5
  let interest_from_6_5 := invested_6_5 * interest_rate_6_5
  let interest_from_remaining := total_interest - interest_from_6_5
  let P := interest_from_remaining / remaining_investment
  P = 0.05 :=
by 
  sorry

end first_investment_percentage_l1610_161097


namespace expand_and_simplify_l1610_161007

theorem expand_and_simplify (x : ℝ) : 
  (2 * x + 6) * (x + 10) = 2 * x^2 + 26 * x + 60 :=
sorry

end expand_and_simplify_l1610_161007


namespace susan_more_cats_than_bob_l1610_161050

-- Given problem: Initial and transaction conditions
def susan_initial_cats : ℕ := 21
def bob_initial_cats : ℕ := 3
def susan_additional_cats : ℕ := 5
def bob_additional_cats : ℕ := 7
def susan_gives_bob_cats : ℕ := 4

-- Declaration to find the difference between Susan's and Bob's cats
def final_susan_cats (initial : ℕ) (additional : ℕ) (given : ℕ) : ℕ := initial + additional - given
def final_bob_cats (initial : ℕ) (additional : ℕ) (received : ℕ) : ℕ := initial + additional + received

-- The proof statement which we need to show
theorem susan_more_cats_than_bob : 
  final_susan_cats susan_initial_cats susan_additional_cats susan_gives_bob_cats - 
  final_bob_cats bob_initial_cats bob_additional_cats susan_gives_bob_cats = 8 := by
  sorry

end susan_more_cats_than_bob_l1610_161050


namespace farm_area_l1610_161068

theorem farm_area (length width area : ℝ) 
  (h1 : length = 0.6) 
  (h2 : width = 3 * length) 
  (h3 : area = length * width) : 
  area = 1.08 := 
by 
  sorry

end farm_area_l1610_161068


namespace coefficient_of_quadratic_polynomial_l1610_161043

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_of_quadratic_polynomial (a b c : ℝ) (h : a > 0) :
  |f a b c 1| = 2 ∧ |f a b c 2| = 2 ∧ |f a b c 3| = 2 →
  (a = 4 ∧ b = -16 ∧ c = 14) ∨ (a = 2 ∧ b = -6 ∧ c = 2) ∨ (a = 2 ∧ b = -10 ∧ c = 10) :=
by
  sorry

end coefficient_of_quadratic_polynomial_l1610_161043


namespace fraction_product_l1610_161019

theorem fraction_product (a b c d e : ℝ) (h1 : a = 1/2) (h2 : b = 1/3) (h3 : c = 1/4) (h4 : d = 1/6) (h5 : e = 144) :
  a * b * c * d * e = 1 := 
by
  -- Given the conditions h1 to h5, we aim to prove the product is 1
  sorry

end fraction_product_l1610_161019


namespace min_value_geq_4_plus_2sqrt2_l1610_161013

theorem min_value_geq_4_plus_2sqrt2
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 1)
  (h4: a + b = 1) :
  ( ( (a^2 + 1) / (a * b) - 2 ) * c + (Real.sqrt 2) / (c - 1) ) ≥ (4 + 2 * (Real.sqrt 2)) :=
sorry

end min_value_geq_4_plus_2sqrt2_l1610_161013


namespace students_both_courses_l1610_161070

-- Definitions from conditions
def total_students : ℕ := 87
def students_french : ℕ := 41
def students_german : ℕ := 22
def students_neither : ℕ := 33

-- The statement we need to prove
theorem students_both_courses : (students_french + students_german - 9 + students_neither = total_students) → (9 = 96 - total_students) :=
by
  -- The proof would go here, but we leave it as sorry for now
  sorry

end students_both_courses_l1610_161070


namespace pool_capacity_l1610_161055

theorem pool_capacity (hose_rate leak_rate : ℝ) (fill_time : ℝ) (net_rate := hose_rate - leak_rate) (total_water := net_rate * fill_time) :
  hose_rate = 1.6 → 
  leak_rate = 0.1 → 
  fill_time = 40 → 
  total_water = 60 := by
  intros
  sorry

end pool_capacity_l1610_161055


namespace radius_of_sphere_l1610_161074

-- Define the conditions.
def radius_wire : ℝ := 8
def length_wire : ℝ := 36

-- Given the volume of the metallic sphere is equal to the volume of the wire,
-- Prove that the radius of the sphere is 12 cm.
theorem radius_of_sphere (r_wire : ℝ) (h_wire : ℝ) (r_sphere : ℝ) : 
    r_wire = radius_wire → h_wire = length_wire →
    (π * r_wire^2 * h_wire = (4/3) * π * r_sphere^3) → 
    r_sphere = 12 :=
by
  intros h₁ h₂ h₃
  -- Add proof steps here.
  sorry

end radius_of_sphere_l1610_161074


namespace segment_length_l1610_161045
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length_l1610_161045


namespace min_distance_line_curve_l1610_161072

/-- 
  Given line l with parametric equations:
    x = 1 + t * cos α,
    y = t * sin α,
  and curve C with the polar equation:
    ρ * sin^2 θ = 4 * cos θ,
  prove:
    1. The Cartesian coordinate equation of C is y^2 = 4x.
    2. The minimum value of the distance |AB|, where line l intersects curve C, is 4.
-/
theorem min_distance_line_curve {t α θ ρ x y : ℝ} 
  (h_line_x: x = 1 + t * Real.cos α)
  (h_line_y: y = t * Real.sin α)
  (h_curve_polar: ρ * (Real.sin θ)^2 = 4 * Real.cos θ)
  (h_alpha_range: 0 < α ∧ α < Real.pi) : 
  (∀ {x y}, y^2 = 4 * x) ∧ (min_value_of_AB = 4) :=
sorry

end min_distance_line_curve_l1610_161072


namespace min_k_intersects_circle_l1610_161075

def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 4
def line_eq (x y k : ℝ) := k * x - y - 2 * k = 0

theorem min_k_intersects_circle :
  (∀ k : ℝ, (∃ x y : ℝ, circle_eq x y ∧ line_eq x y k) → k ≥ - (Real.sqrt 3) / 3) :=
sorry

end min_k_intersects_circle_l1610_161075


namespace fraction_pow_zero_l1610_161025

theorem fraction_pow_zero :
  let a := 7632148
  let b := -172836429
  (a / b ≠ 0) → (a / b)^0 = 1 := by
  sorry

end fraction_pow_zero_l1610_161025


namespace expression_value_l1610_161098

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 :=
by
  sorry

end expression_value_l1610_161098


namespace slope_of_tangent_line_at_1_1_l1610_161044

theorem slope_of_tangent_line_at_1_1 : 
  ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * x^2) ∧ (f' 1 = 3) :=
by
  sorry

end slope_of_tangent_line_at_1_1_l1610_161044


namespace center_of_circle_l1610_161066

theorem center_of_circle (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (10, 5)) :
    (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 1 :=
by
  sorry

end center_of_circle_l1610_161066


namespace function_above_x_axis_l1610_161062

noncomputable def quadratic_function (a x : ℝ) := (a^2 - 3 * a + 2) * x^2 + (a - 1) * x + 2

theorem function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x > 0) ↔ (a > 15 / 7 ∨ a ≤ 1) :=
by {
  sorry
}

end function_above_x_axis_l1610_161062


namespace find_number_l1610_161035

theorem find_number (x : ℕ) (h : 3 * x = 33) : x = 11 :=
sorry

end find_number_l1610_161035


namespace find_square_sum_l1610_161022

theorem find_square_sum (x y : ℝ) (h1 : 2 * x * (x + y) = 54) (h2 : 3 * y * (x + y) = 81) : (x + y) ^ 2 = 135 :=
sorry

end find_square_sum_l1610_161022


namespace arithmetic_sequence_sum_l1610_161083

/-
The sum of the first 20 terms of the arithmetic sequence 8, 5, 2, ... is -410.
-/

theorem arithmetic_sequence_sum :
  let a : ℤ := 8
  let d : ℤ := -3
  let n : ℤ := 20
  let S_n : ℤ := n * a + (d * n * (n - 1)) / 2
  S_n = -410 := by
  sorry

end arithmetic_sequence_sum_l1610_161083


namespace find_xy_solution_l1610_161082

theorem find_xy_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) 
    (h : 3^x + x^4 = y.factorial + 2019) : 
    (x = 6 ∧ y = 3) :=
by {
  sorry
}

end find_xy_solution_l1610_161082


namespace wall_cost_equal_l1610_161086

theorem wall_cost_equal (A B C : ℝ) (d_1 d_2 : ℝ) (h1 : A = B) (h2 : B = C) : d_1 = d_2 :=
by
  -- sorry is used to skip the proof
  sorry

end wall_cost_equal_l1610_161086


namespace find_a7_a8_a9_l1610_161071

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l1610_161071


namespace correct_sampling_methods_l1610_161064

-- Define conditions for the sampling problems
structure SamplingProblem where
  scenario: String
  samplingMethod: String

-- Define the three scenarios
def firstScenario : SamplingProblem :=
  { scenario := "Draw 5 bottles from 15 bottles of drinks for food hygiene inspection", samplingMethod := "Simple random sampling" }

def secondScenario : SamplingProblem :=
  { scenario := "Sample 20 staff members from 240 staff members in a middle school", samplingMethod := "Stratified sampling" }

def thirdScenario : SamplingProblem :=
  { scenario := "Select 25 audience members from a full science and technology report hall", samplingMethod := "Systematic sampling" }

-- Main theorem combining all conditions and proving the correct answer
theorem correct_sampling_methods :
  (firstScenario.samplingMethod = "Simple random sampling") ∧
  (secondScenario.samplingMethod = "Stratified sampling") ∧
  (thirdScenario.samplingMethod = "Systematic sampling") :=
by
  sorry -- Proof is omitted

end correct_sampling_methods_l1610_161064


namespace ending_number_divisible_by_9_l1610_161067

theorem ending_number_divisible_by_9 (E : ℕ) 
  (h1 : ∀ n, 10 ≤ n → n ≤ E → n % 9 = 0 → ∃ m ≥ 1, n = 18 + 9 * (m - 1)) 
  (h2 : (E - 18) / 9 + 1 = 111110) : 
  E = 999999 :=
by
  sorry

end ending_number_divisible_by_9_l1610_161067


namespace johnson_potatoes_l1610_161008

/-- Given that Johnson has a sack of 300 potatoes, 
    gives some to Gina, twice that amount to Tom, and 
    one-third of the amount given to Tom to Anne,
    and has 47 potatoes left, we prove that 
    Johnson gave Gina 69 potatoes. -/
theorem johnson_potatoes : 
  ∃ G : ℕ, 
  ∀ (Gina Tom Anne total : ℕ), 
    total = 300 ∧ 
    total - (Gina + Tom + Anne) = 47 ∧ 
    Tom = 2 * Gina ∧ 
    Anne = (1 / 3 : ℚ) * Tom ∧ 
    (Gina + Tom + (Anne : ℕ)) = (11 / 3 : ℚ) * Gina ∧ 
    (Gina + Tom + Anne) = 253 
    ∧ total = Gina + Tom + Anne + 47 
    → Gina = 69 := sorry


end johnson_potatoes_l1610_161008


namespace ashley_family_spending_l1610_161099

theorem ashley_family_spending:
  let child_ticket := 4.25
  let adult_ticket := child_ticket + 3.50
  let senior_ticket := adult_ticket - 1.75
  let morning_discount := 0.10
  let total_morning_tickets := 2 * adult_ticket + 4 * child_ticket + senior_ticket
  let morning_tickets_after_discount := total_morning_tickets * (1 - morning_discount)
  let buy_2_get_1_free_discount := child_ticket
  let discount_for_5_or_more := 4.00
  let total_tickets_after_vouchers := morning_tickets_after_discount - buy_2_get_1_free_discount - discount_for_5_or_more
  let popcorn := 5.25
  let soda := 3.50
  let candy := 4.00
  let concession_total := 3 * popcorn + 2 * soda + candy
  let concession_discount := concession_total * 0.10
  let concession_after_discount := concession_total - concession_discount
  let final_total := total_tickets_after_vouchers + concession_after_discount
  final_total = 50.47 := by
  sorry

end ashley_family_spending_l1610_161099


namespace sequence_elements_are_prime_l1610_161021

variable {a : ℕ → ℕ} {p : ℕ → ℕ}

def increasing_seq (f : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → f i < f j

def divisible_by_prime (a p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n) ∧ p n ∣ a n

def satisfies_condition (a p : ℕ → ℕ) : Prop :=
  ∀ n k, a n - a k = p n - p k

theorem sequence_elements_are_prime (h1 : increasing_seq a) 
    (h2 : divisible_by_prime a p) 
    (h3 : satisfies_condition a p) :
    ∀ n, Prime (a n) :=
by 
  sorry

end sequence_elements_are_prime_l1610_161021


namespace lowest_height_l1610_161096

noncomputable def length_A : ℝ := 2.4
noncomputable def length_B : ℝ := 3.2
noncomputable def length_C : ℝ := 2.8

noncomputable def height_Eunji : ℝ := 8 * length_A
noncomputable def height_Namjoon : ℝ := 4 * length_B
noncomputable def height_Hoseok : ℝ := 5 * length_C

theorem lowest_height :
  height_Namjoon = 12.8 ∧ 
  height_Namjoon < height_Eunji ∧ 
  height_Namjoon < height_Hoseok :=
by
  sorry

end lowest_height_l1610_161096


namespace rotate_90deg_l1610_161030

def Shape := Type

structure Figure :=
(triangle : Shape)
(circle : Shape)
(square : Shape)
(pentagon : Shape)

def rotated_position (fig : Figure) : Figure :=
{ triangle := fig.circle,
  circle := fig.square,
  square := fig.pentagon,
  pentagon := fig.triangle }

theorem rotate_90deg (fig : Figure) :
  rotated_position fig = { triangle := fig.circle,
                           circle := fig.square,
                           square := fig.pentagon,
                           pentagon := fig.triangle } :=
by {
  sorry
}

end rotate_90deg_l1610_161030


namespace solution_set_of_inequality_l1610_161063

theorem solution_set_of_inequality (x : ℝ) : x^2 > x ↔ x < 0 ∨ 1 < x := 
by
  sorry

end solution_set_of_inequality_l1610_161063


namespace arithmetic_sequence_diff_l1610_161053

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition for the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop := 
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Definition of the common difference
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The proof problem statement in Lean 4
theorem arithmetic_sequence_diff (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a → condition a → common_difference a d → a 7 - a 8 = -d :=
by
  intros _ _ _
  -- Proof will be conducted here
  sorry

end arithmetic_sequence_diff_l1610_161053


namespace power_cycle_i_l1610_161017

theorem power_cycle_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^23 + i^75 = -2 * i :=
by
  sorry

end power_cycle_i_l1610_161017


namespace green_marble_prob_l1610_161058

-- Problem constants
def total_marbles : ℕ := 84
def prob_white : ℚ := 1 / 4
def prob_red_or_blue : ℚ := 0.4642857142857143

-- Defining the individual variables for the counts
variable (W R B G : ℕ)

-- Conditions
axiom total_marbles_eq : W + R + B + G = total_marbles
axiom prob_white_eq : (W : ℚ) / total_marbles = prob_white
axiom prob_red_or_blue_eq : (R + B : ℚ) / total_marbles = prob_red_or_blue

-- Proving the probability of drawing a green marble
theorem green_marble_prob :
  (G : ℚ) / total_marbles = 2 / 7 :=
by
  sorry  -- Proof is not required and thus omitted

end green_marble_prob_l1610_161058


namespace calc1_calc2_calc3_l1610_161003

theorem calc1 : 1 - 2 + 3 - 4 + 5 = 3 := by sorry
theorem calc2 : - (4 / 7) / (8 / 49) = - (7 / 2) := by sorry
theorem calc3 : ((1 / 2) - (3 / 5) + (2 / 3)) * (-15) = - (17 / 2) := by sorry

end calc1_calc2_calc3_l1610_161003


namespace fishbowl_count_l1610_161005

def number_of_fishbowls (total_fish : ℕ) (fish_per_bowl : ℕ) : ℕ :=
  total_fish / fish_per_bowl

theorem fishbowl_count (h1 : 23 > 0) (h2 : 6003 % 23 = 0) :
  number_of_fishbowls 6003 23 = 261 :=
by
  -- proof goes here
  sorry

end fishbowl_count_l1610_161005


namespace exists_arithmetic_progression_with_sum_zero_l1610_161032

theorem exists_arithmetic_progression_with_sum_zero : 
  ∃ (a d : Int) (n : Int), n > 0 ∧ (n * (2 * a + (n - 1) * d)) = 0 :=
by 
  sorry

end exists_arithmetic_progression_with_sum_zero_l1610_161032


namespace cards_left_l1610_161029

variable (initialCards : ℕ) (givenCards : ℕ) (remainingCards : ℕ)

def JasonInitialCards := 13
def CardsGivenAway := 9

theorem cards_left : initialCards = JasonInitialCards → givenCards = CardsGivenAway → remainingCards = initialCards - givenCards → remainingCards = 4 :=
by
  intros
  subst_vars
  sorry

end cards_left_l1610_161029


namespace small_triangle_count_l1610_161087

theorem small_triangle_count (n : ℕ) (h : n = 2009) : (2 * n + 1) = 4019 := 
by {
    sorry
}

end small_triangle_count_l1610_161087


namespace sum_of_ages_l1610_161040

-- Definitions from the problem conditions
def Maria_age : ℕ := 14
def age_difference_between_Jose_and_Maria : ℕ := 12
def Jose_age : ℕ := Maria_age + age_difference_between_Jose_and_Maria

-- To be proven: sum of their ages is 40
theorem sum_of_ages : Maria_age + Jose_age = 40 :=
by
  -- skip the proof
  sorry

end sum_of_ages_l1610_161040


namespace monotonically_decreasing_when_a_half_l1610_161095

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - a * x)

theorem monotonically_decreasing_when_a_half :
  ∀ x : ℝ, 0 < x → (f x (1 / 2)) ≤ 0 :=
by
  sorry

end monotonically_decreasing_when_a_half_l1610_161095


namespace aquarium_length_l1610_161081

theorem aquarium_length {L : ℝ} (W H : ℝ) (final_volume : ℝ)
  (hW : W = 6) (hH : H = 3) (h_final_volume : final_volume = 54)
  (h_volume_relation : final_volume = 3 * (1/4 * L * W * H)) :
  L = 4 := by
  -- Mathematically translate the problem given conditions and resulting in L = 4.
  sorry

end aquarium_length_l1610_161081


namespace jake_more_peaches_than_jill_l1610_161036

theorem jake_more_peaches_than_jill :
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  jake_peaches - jill_peaches = 3 :=
by
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  sorry

end jake_more_peaches_than_jill_l1610_161036


namespace rectangular_coords_of_neg_theta_l1610_161020

theorem rectangular_coords_of_neg_theta 
  (x y z : ℝ) 
  (rho theta phi : ℝ)
  (hx : x = 8)
  (hy : y = 6)
  (hz : z = -3)
  (h_rho : rho = Real.sqrt (x^2 + y^2 + z^2))
  (h_cos_phi : Real.cos phi = z / rho)
  (h_sin_phi : Real.sin phi = Real.sqrt (1 - (Real.cos phi)^2))
  (h_tan_theta : Real.tan theta = y / x) :
  (rho * Real.sin phi * Real.cos (-theta), rho * Real.sin phi * Real.sin (-theta), rho * Real.cos phi) = (8, -6, -3) := 
  sorry

end rectangular_coords_of_neg_theta_l1610_161020


namespace units_digit_two_pow_2010_l1610_161047

-- Conditions from part a)
def two_power_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case will not occur due to modulo operation

-- Question translated to a proof problem
theorem units_digit_two_pow_2010 : (two_power_units_digit 2010) = 4 :=
by 
  -- Proof would go here
  sorry

end units_digit_two_pow_2010_l1610_161047


namespace ratio_of_area_l1610_161027

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l1610_161027


namespace books_remaining_in_library_l1610_161090

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l1610_161090


namespace O_is_incenter_l1610_161094

variable {n : ℕ}
variable (A : Fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions
def inside_convex_ngon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_acute (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_inequality (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry

-- This is the statement that we need to prove.
theorem O_is_incenter 
  (h1 : inside_convex_ngon O A)
  (h2 : angles_acute O A) 
  (h3 : angles_inequality O A) 
: sorry := sorry

end O_is_incenter_l1610_161094


namespace share_y_is_18_l1610_161054

-- Definitions from conditions
def total_amount := 70
def ratio_x := 100
def ratio_y := 45
def ratio_z := 30
def total_ratio := ratio_x + ratio_y + ratio_z
def part_value := total_amount / total_ratio
def share_y := ratio_y * part_value

-- Statement to be proved
theorem share_y_is_18 : share_y = 18 :=
by
  -- Placeholder for the proof
  sorry

end share_y_is_18_l1610_161054


namespace no_integer_solutions_l1610_161031

theorem no_integer_solutions (x y : ℤ) : x^3 + 4 * x^2 + x ≠ 18 * y^3 + 18 * y^2 + 6 * y + 3 := 
by 
  sorry

end no_integer_solutions_l1610_161031


namespace regular_polygon_sides_l1610_161016

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end regular_polygon_sides_l1610_161016


namespace room_length_l1610_161093

theorem room_length
  (width : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (h_width : width = 4)
  (h_cost_rate : cost_rate = 850)
  (h_total_cost : total_cost = 18700) :
  ∃ L : ℝ, L = 5.5 ∧ total_cost = cost_rate * (L * width) :=
by
  sorry

end room_length_l1610_161093


namespace pentagon_perimeter_l1610_161059

-- Problem statement: Given an irregular pentagon with specified side lengths,
-- prove that its perimeter is equal to 52.9 cm.

theorem pentagon_perimeter 
  (a b c d e : ℝ)
  (h1 : a = 5.2)
  (h2 : b = 10.3)
  (h3 : c = 15.8)
  (h4 : d = 8.7)
  (h5 : e = 12.9) 
  : a + b + c + d + e = 52.9 := 
by
  sorry

end pentagon_perimeter_l1610_161059


namespace max_length_cos_theta_l1610_161042

def domain (x y : ℝ) : Prop := (x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ (Real.sqrt 2 / 3))

theorem max_length_cos_theta :
  (∃ x y : ℝ, domain x y ∧ ∀ θ : ℝ, (0 < θ ∧ θ < (Real.pi / 2)) → θ = Real.arctan (Real.sqrt 2) → 
  (Real.cos θ = Real.sqrt 3 / 3)) := sorry

end max_length_cos_theta_l1610_161042


namespace identity_completion_factorize_polynomial_equilateral_triangle_l1610_161011

-- Statement 1: Prove that a^3 - b^3 + a^2 b - ab^2 = (a - b)(a + b)^2 
theorem identity_completion (a b : ℝ) : a^3 - b^3 + a^2 * b - a * b^2 = (a - b) * (a + b)^2 :=
sorry

-- Statement 2: Prove that 4x^2 - 2x - y^2 - y = (2x + y)(2x - y - 1)
theorem factorize_polynomial (x y : ℝ) : 4 * x^2 - 2 * x - y^2 - y = (2 * x + y) * (2 * x - y - 1) :=
sorry

-- Statement 3: Given a^2 + b^2 + 2c^2 - 2ac - 2bc = 0, Prove that triangle ABC is equilateral
theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + 2 * c^2 - 2 * a * c - 2 * b * c = 0) : a = b ∧ b = c :=
sorry

end identity_completion_factorize_polynomial_equilateral_triangle_l1610_161011


namespace largest_multiple_of_8_less_than_neg_63_l1610_161015

theorem largest_multiple_of_8_less_than_neg_63 : 
  ∃ n : ℤ, (n < -63) ∧ (∃ k : ℤ, n = 8 * k) ∧ (∀ m : ℤ, (m < -63) ∧ (∃ l : ℤ, m = 8 * l) → m ≤ n) :=
sorry

end largest_multiple_of_8_less_than_neg_63_l1610_161015


namespace rational_eq_reciprocal_l1610_161033

theorem rational_eq_reciprocal (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 :=
by {
  sorry
}

end rational_eq_reciprocal_l1610_161033


namespace volume_of_mixture_removed_replaced_l1610_161069

noncomputable def volume_removed (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ): ℝ :=
  (1 - final_concentration / initial_milk) * initial_mixture

theorem volume_of_mixture_removed_replaced (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ) (V: ℝ):
  initial_mixture = 100 →
  initial_milk = 36 →
  final_concentration = 9 →
  V = 50 →
  volume_removed initial_mixture initial_milk final_concentration = V :=
by
  intros h1 h2 h3 h4
  have h5 : initial_mixture = 100 := h1
  have h6 : initial_milk = 36 := h2
  have h7 : final_concentration = 9 := h3
  rw [h5, h6, h7]
  sorry

end volume_of_mixture_removed_replaced_l1610_161069


namespace arithmetic_geometric_sequence_a4_value_l1610_161052

theorem arithmetic_geometric_sequence_a4_value 
  (a : ℕ → ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) : 
  a 4 = 1 := 
sorry

end arithmetic_geometric_sequence_a4_value_l1610_161052


namespace problem_solution_l1610_161056

theorem problem_solution :
  (∀ (p q : ℚ), 
    (∀ (x : ℚ), (x + 3 * p) * (x^2 - x + (1 / 3) * q) = x^3 + (3 * p - 1) * x^2 + ((1 / 3) * q - 3 * p) * x + p * q) →
    (3 * p - 1 = 0) →
    ((1 / 3) * q - 3 * p = 0) →
    p = 1 / 3 ∧ q = 3)
  ∧ ((1 / 3) ^ 2020 * 3 ^ 2021 = 3) :=
by
  sorry

end problem_solution_l1610_161056
