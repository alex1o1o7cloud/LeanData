import Mathlib

namespace range_of_a_max_area_of_triangle_l650_65027

variable (p a : ℝ) (h : p > 0)

def parabola_eq (x y : ℝ) := y ^ 2 = 2 * p * x
def line_eq (x y : ℝ) := y = x - a
def intersects_parabola (A B : ℝ × ℝ) := parabola_eq p A.fst A.snd ∧ line_eq a A.fst A.snd ∧ parabola_eq p B.fst B.snd ∧ line_eq a B.fst B.snd
def ab_length_le_2p (A B : ℝ × ℝ) := (Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) ≤ 2 * p)

theorem range_of_a
  (A B : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B) :
  - p / 2 < a ∧ a ≤ - p / 4 := sorry

theorem max_area_of_triangle
  (A B : ℝ × ℝ) (N : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B)
  (h_N : N.snd = 0) :
  ∃ (S : ℝ), S = Real.sqrt 2 * p^2 := sorry

end range_of_a_max_area_of_triangle_l650_65027


namespace sqrt_224_between_14_and_15_l650_65016

theorem sqrt_224_between_14_and_15 : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end sqrt_224_between_14_and_15_l650_65016


namespace arithmetic_sequence_sum_l650_65038

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l650_65038


namespace count_integers_in_interval_l650_65014

theorem count_integers_in_interval : 
  ∃ (k : ℤ), k = 46 ∧ 
  (∀ n : ℤ, -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ) → (-13 ≤ n ∧ n ≤ 32)) ∧ 
  (∀ n : ℤ, -13 ≤ n ∧ n ≤ 32 → -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ)) :=
sorry

end count_integers_in_interval_l650_65014


namespace initial_workers_count_l650_65033

theorem initial_workers_count (W : ℕ) 
  (h1 : W * 30 = W * 30) 
  (h2 : W * 15 = (W - 5) * 20)
  (h3 : W > 5) 
  : W = 20 :=
by {
  sorry
}

end initial_workers_count_l650_65033


namespace grass_field_width_l650_65065

theorem grass_field_width (w : ℝ) (length_field : ℝ) (path_width : ℝ) (area_path : ℝ) :
  length_field = 85 → path_width = 2.5 → area_path = 1450 →
  (90 * (w + path_width * 2) - length_field * w = area_path) → w = 200 :=
by
  intros h_length_field h_path_width h_area_path h_eq
  sorry

end grass_field_width_l650_65065


namespace triangle_DEF_angle_l650_65021

noncomputable def one_angle_of_triangle_DEF (x : ℝ) : ℝ :=
  let arc_DE := 2 * x + 40
  let arc_EF := 3 * x + 50
  let arc_FD := 4 * x - 30
  if (arc_DE + arc_EF + arc_FD = 360)
  then (1 / 2) * arc_EF
  else 0

theorem triangle_DEF_angle (x : ℝ) (h : 2 * x + 40 + 3 * x + 50 + 4 * x - 30 = 360) :
  one_angle_of_triangle_DEF x = 75 :=
by sorry

end triangle_DEF_angle_l650_65021


namespace find_f_l650_65058

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f (x : ℝ) :
  (∀ t : ℝ, t = (1 - x) / (1 + x) → f t = (1 - x^2) / (1 + x^2)) →
  f x = (2 * x) / (1 + x^2) :=
by
  intros h
  specialize h ((1 - x) / (1 + x))
  specialize h rfl
  exact sorry

end find_f_l650_65058


namespace woman_speed_still_water_l650_65012

theorem woman_speed_still_water (v_w v_c : ℝ) 
    (h1 : 120 = (v_w + v_c) * 10)
    (h2 : 24 = (v_w - v_c) * 14) : 
    v_w = 48 / 7 :=
by {
  sorry
}

end woman_speed_still_water_l650_65012


namespace no_integer_solution_for_triples_l650_65088

theorem no_integer_solution_for_triples :
  ∀ (x y z : ℤ),
    x^2 - 2*x*y + 3*y^2 - z^2 = 17 →
    -x^2 + 4*y*z + z^2 = 28 →
    x^2 + 2*x*y + 5*z^2 = 42 →
    false :=
by
  intros x y z h1 h2 h3
  sorry

end no_integer_solution_for_triples_l650_65088


namespace metal_relative_atomic_mass_is_24_l650_65024

noncomputable def relative_atomic_mass (metal_mass : ℝ) (hcl_mass_percent : ℝ) (hcl_total_mass : ℝ) (mol_mass_hcl : ℝ) : ℝ :=
  let moles_hcl := (hcl_total_mass * hcl_mass_percent / 100) / mol_mass_hcl
  let maximum_molar_mass := metal_mass / (moles_hcl / 2)
  let minimum_molar_mass := metal_mass / (moles_hcl / 2)
  if 20 < maximum_molar_mass ∧ maximum_molar_mass < 28 then
    24
  else
    0

theorem metal_relative_atomic_mass_is_24
  (metal_mass_1 : ℝ)
  (metal_mass_2 : ℝ)
  (hcl_mass_percent : ℝ)
  (hcl_total_mass : ℝ)
  (mol_mass_hcl : ℝ)
  (moles_used_1 : ℝ)
  (moles_used_2 : ℝ)
  (excess : Bool)
  (complete : Bool) :
  relative_atomic_mass 3.5 18.25 50 36.5 = 24 :=
by
  sorry

end metal_relative_atomic_mass_is_24_l650_65024


namespace tournament_total_players_l650_65011

theorem tournament_total_players (n : ℕ) (total_points : ℕ) (total_games : ℕ) (half_points : ℕ → ℕ) :
  (∀ k, half_points k * 2 = total_points) ∧ total_points = total_games ∧
  total_points = n * (n + 11) + 132 ∧
  total_games = (n + 12) * (n + 11) / 2 →
  n + 12 = 24 :=
by
  sorry

end tournament_total_players_l650_65011


namespace min_a2_b2_l650_65008

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) :
  a^2 + b^2 ≥ 4 / 5 :=
sorry

end min_a2_b2_l650_65008


namespace probability_correct_l650_65019

-- Define the set of segment lengths
def segment_lengths : List ℕ := [1, 3, 5, 7, 9]

-- Define the triangle inequality condition
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of favorable outcomes, i.e., sets that can form a triangle
def favorable_sets : List (ℕ × ℕ × ℕ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

-- Define the total number of ways to select three segments out of five
def total_combinations : ℕ :=
  10

-- Define the number of favorable sets
def number_of_favorable_sets : ℕ :=
  favorable_sets.length

-- Calculate the probability of selecting three segments that form a triangle
def probability_of_triangle : ℚ :=
  number_of_favorable_sets / total_combinations

-- The theorem to prove
theorem probability_correct : probability_of_triangle = 3 / 10 :=
  by {
    -- Placeholder for the proof
    sorry
  }

end probability_correct_l650_65019


namespace circumference_of_jogging_track_l650_65061

noncomputable def trackCircumference (Deepak_speed : ℝ) (Wife_speed : ℝ) (meet_time_minutes : ℝ) : ℝ :=
  let relative_speed := Deepak_speed + Wife_speed
  let meet_time_hours := meet_time_minutes / 60
  relative_speed * meet_time_hours

theorem circumference_of_jogging_track :
  trackCircumference 20 17 37 = 1369 / 60 :=
by
  sorry

end circumference_of_jogging_track_l650_65061


namespace units_digit_problem_l650_65006

open BigOperators

-- Define relevant constants
def A : ℤ := 21
noncomputable def B : ℤ := 14 -- since B = sqrt(196) = 14

-- Define the terms
noncomputable def term1 : ℤ := (A + B) ^ 20
noncomputable def term2 : ℤ := (A - B) ^ 20

-- Statement of the theorem
theorem units_digit_problem :
  ((term1 - term2) % 10) = 4 := 
sorry

end units_digit_problem_l650_65006


namespace values_of_a_and_b_l650_65055

def is_root (a b x : ℝ) : Prop := x^2 - 2*a*x + b = 0

noncomputable def A : Set ℝ := {-1, 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | is_root a b x}

theorem values_of_a_and_b (a b : ℝ) (h_nonempty : Set.Nonempty (B a b)) (h_union : A ∪ B a b = A) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1) :=
sorry

end values_of_a_and_b_l650_65055


namespace gilled_mushrooms_count_l650_65091

def mushrooms_problem (G S : ℕ) : Prop :=
  (S = 9 * G) ∧ (G + S = 30) → (G = 3)

-- The theorem statement corresponding to the problem
theorem gilled_mushrooms_count (G S : ℕ) : mushrooms_problem G S :=
by {
  sorry
}

end gilled_mushrooms_count_l650_65091


namespace calc_fraction_l650_65020
-- Import necessary libraries

-- Define the necessary fractions and the given expression
def expr := (5 / 6) * (1 / (7 / 8 - 3 / 4))

-- State the theorem
theorem calc_fraction : expr = 20 / 3 := 
by
  sorry

end calc_fraction_l650_65020


namespace rotated_parabola_eq_l650_65087

theorem rotated_parabola_eq :
  ∀ x y : ℝ, y = x^2 → ∃ y' x' : ℝ, (y' = (-x':ℝ)^2) := sorry

end rotated_parabola_eq_l650_65087


namespace range_of_x_coordinate_of_Q_l650_65074

def Point := ℝ × ℝ

def parabola (P : Point) : Prop :=
  P.2 = P.1 ^ 2

def vector (P Q : Point) : Point :=
  (Q.1 - P.1, Q.2 - P.2)

def dot_product (u v : Point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (P Q R : Point) : Prop :=
  dot_product (vector P Q) (vector P R) = 0

theorem range_of_x_coordinate_of_Q:
  ∀ (A P Q: Point), 
    A = (-1, 1) →
    parabola P →
    parabola Q →
    perpendicular P A Q →
    (Q.1 ≤ -3 ∨ Q.1 ≥ 1) :=
by
  intros A P Q hA hParabP hParabQ hPerp
  sorry

end range_of_x_coordinate_of_Q_l650_65074


namespace lee_charged_per_action_figure_l650_65003

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end lee_charged_per_action_figure_l650_65003


namespace combinations_count_l650_65059

theorem combinations_count:
  let valid_a (a: ℕ) := a < 1000 ∧ a % 29 = 7
  let valid_b (b: ℕ) := b < 1000 ∧ b % 47 = 22
  let valid_c (c: ℕ) (a b: ℕ) := c < 1000 ∧ c = (a + b) % 23 
  ∃ (a b c: ℕ), valid_a a ∧ valid_b b ∧ valid_c c a b :=
  sorry

end combinations_count_l650_65059


namespace ratio_frogs_to_dogs_l650_65036

variable (D C F : ℕ)

-- Define the conditions as given in the problem statement
def cats_eq_dogs_implied : Prop := C = Nat.div (4 * D) 5
def frogs : Prop := F = 160
def total_animals : Prop := D + C + F = 304

-- Define the statement to be proved
theorem ratio_frogs_to_dogs (h1 : cats_eq_dogs_implied D C) (h2 : frogs F) (h3 : total_animals D C F) : F / D = 2 := by
  sorry

end ratio_frogs_to_dogs_l650_65036


namespace average_speed_round_trip_l650_65045

noncomputable def distance_AB : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

theorem average_speed_round_trip :
  (2 * distance_AB * speed_AB * speed_BA) / (distance_AB * (speed_AB + speed_BA)) = 34 := 
  by 
    sorry

end average_speed_round_trip_l650_65045


namespace age_difference_l650_65043

variable {A B C : ℕ}

-- Definition of conditions
def condition1 (A B C : ℕ) : Prop := A + B > B + C
def condition2 (A C : ℕ) : Prop := C = A - 16

-- The theorem stating the math problem
theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A C) :
  (A + B) - (B + C) = 16 := by
  sorry

end age_difference_l650_65043


namespace probability_of_both_selected_l650_65018

theorem probability_of_both_selected :
  let pX := 1 / 5
  let pY := 2 / 7
  (pX * pY) = 2 / 35 :=
by
  let pX := 1 / 5
  let pY := 2 / 7
  show (pX * pY) = 2 / 35
  sorry

end probability_of_both_selected_l650_65018


namespace triangle_inequality_difference_l650_65040

theorem triangle_inequality_difference :
  ∀ (x : ℤ), (x + 8 > 3) → (x + 3 > 8) → (8 + 3 > x) →
  ( 10 - 6 = 4 ) :=
by sorry

end triangle_inequality_difference_l650_65040


namespace final_weight_is_200_l650_65096

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l650_65096


namespace swiss_slices_correct_l650_65075

-- Define the variables and conditions
variables (S : ℕ) (cheddar_slices : ℕ := 12) (total_cheddar_slices : ℕ := 84) (total_swiss_slices : ℕ := 84)

-- Define the statement to be proved
theorem swiss_slices_correct (H : total_cheddar_slices = total_swiss_slices) : S = 12 :=
sorry

end swiss_slices_correct_l650_65075


namespace average_age_of_students_l650_65093

variable (A : ℕ) -- We define A as a natural number representing average age

-- Define the conditions
def num_students : ℕ := 32
def staff_age : ℕ := 49
def new_average_age := A + 1

-- Definition of total age including the staff
def total_age_with_staff := 33 * new_average_age

-- Original condition stated as an equality
def condition : Prop := num_students * A + staff_age = total_age_with_staff

-- Theorem statement asserting that the average age A is 16 given the condition
theorem average_age_of_students : condition A → A = 16 :=
by sorry

end average_age_of_students_l650_65093


namespace december_19th_day_l650_65009

theorem december_19th_day (december_has_31_days : true)
  (december_1st_is_monday : true)
  (day_of_week : ℕ → ℕ) :
  day_of_week 19 = 5 :=
sorry

end december_19th_day_l650_65009


namespace committee_selection_l650_65067

-- Definitions based on the conditions
def num_people := 12
def num_women := 7
def num_men := 5
def committee_size := 5
def min_women := 2

-- Binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Required number of ways to form the committee
def num_ways_5_person_committee_with_at_least_2_women : ℕ :=
  binom num_women min_women * binom (num_people - min_women) (committee_size - min_women)

-- Statement to be proven
theorem committee_selection : num_ways_5_person_committee_with_at_least_2_women = 2520 :=
by
  sorry

end committee_selection_l650_65067


namespace evaluate_expression_l650_65060

theorem evaluate_expression : 
  (900 * 900) / ((306 * 306) - (294 * 294)) = 112.5 := by
  sorry

end evaluate_expression_l650_65060


namespace median_computation_l650_65042

noncomputable def length_of_median (A B C A1 P Q R : ℝ) : Prop :=
  let AB := 10
  let AC := 6
  let BC := Real.sqrt (AB^2 - AC^2)
  let A1C := 24 / 7
  let A1B := 32 / 7
  let QR := Real.sqrt (A1B^2 - A1C^2)
  let median_length := QR / 2
  median_length = 4 * Real.sqrt 7 / 7

theorem median_computation (A B C A1 P Q R : ℝ) :
  length_of_median A B C A1 P Q R := by
  sorry

end median_computation_l650_65042


namespace p_minus_q_eq_16_sqrt_2_l650_65002

theorem p_minus_q_eq_16_sqrt_2 (p q : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 28 * x - 84 → x = p ∨ x = q)
  (h_distinct : p ≠ q) (h_p_gt_q : p > q) : p - q = 16 * Real.sqrt 2 :=
sorry

end p_minus_q_eq_16_sqrt_2_l650_65002


namespace solve_equation_nat_numbers_l650_65037

theorem solve_equation_nat_numbers (a b : ℕ) (h : (a, b) = (11, 170) ∨ (a, b) = (22, 158) ∨ (a, b) = (33, 146) ∨
                                    (a, b) = (44, 134) ∨ (a, b) = (55, 122) ∨ (a, b) = (66, 110) ∨
                                    (a, b) = (77, 98) ∨ (a, b) = (88, 86) ∨ (a, b) = (99, 74) ∨
                                    (a, b) = (110, 62) ∨ (a, b) = (121, 50) ∨ (a, b) = (132, 38) ∨
                                    (a, b) = (143, 26) ∨ (a, b) = (154, 14) ∨ (a, b) = (165, 2)) :
  12 * a + 11 * b = 2002 :=
by
  sorry

end solve_equation_nat_numbers_l650_65037


namespace min_value_correct_l650_65084

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  Real.sqrt ((a^2 + 2 * b^2) * (4 * a^2 + b^2)) / (a * b)

theorem min_value_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value a b ha hb ≥ 3 :=
sorry

end min_value_correct_l650_65084


namespace num_5_digit_numbers_is_six_l650_65081

-- Define that we have the digits 2, 45, and 68
def digits : List Nat := [2, 45, 68]

-- Function to generate all permutations of given digits
def permute : List Nat → List (List Nat)
| [] => [[]]
| (x::xs) =>
  List.join (List.map (λ ys =>
    List.map (λ zs => x :: zs) (permute xs)) (permute xs))

-- Calculate the number of distinct 5-digit numbers
def numberOf5DigitNumbers : Int := 
  (permute digits).length

-- Theorem to prove the number of distinct 5-digit numbers formed
theorem num_5_digit_numbers_is_six : numberOf5DigitNumbers = 6 := by
  sorry

end num_5_digit_numbers_is_six_l650_65081


namespace probability_red_red_red_l650_65076

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end probability_red_red_red_l650_65076


namespace value_op_and_add_10_l650_65070

def op_and (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem value_op_and_add_10 : op_and 8 5 + 10 = 49 :=
by
  sorry

end value_op_and_add_10_l650_65070


namespace tangent_and_parallel_l650_65083

noncomputable def parabola1 (x : ℝ) (b1 c1 : ℝ) : ℝ := -x^2 + b1 * x + c1
noncomputable def parabola2 (x : ℝ) (b2 c2 : ℝ) : ℝ := -x^2 + b2 * x + c2
noncomputable def parabola3 (x : ℝ) (b3 c3 : ℝ) : ℝ := x^2 + b3 * x + c3

theorem tangent_and_parallel (b1 b2 b3 c1 c2 c3 : ℝ) :
  (b3 - b1)^2 = 8 * (c3 - c1) → (b3 - b2)^2 = 8 * (c3 - c2) →
  ((b2^2 - b1^2 + 2 * b3 * (b2 - b1)) / (4 * (b2 - b1))) = 
  ((4 * (c1 - c2) - 2 * b3 * (b1 - b2)) / (2 * (b2 - b1))) :=
by
  intros h1 h2
  sorry

end tangent_and_parallel_l650_65083


namespace point_P_coordinates_l650_65034

theorem point_P_coordinates :
  ∃ (x y : ℝ), (y = (x^3 - 10 * x + 3)) ∧ (x < 0) ∧ (3 * x^2 - 10 = 2) ∧ (x = -2 ∧ y = 15) := by
sorry

end point_P_coordinates_l650_65034


namespace part1_part2_l650_65013

-- Define the function y in Lean
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part (1)
theorem part1 (x : ℝ) : y (1/2) x < 0 ↔ -1 < x ∧ x < 2 :=
  sorry

-- Part (2)
theorem part2 (x m : ℝ) : y m x < (1 - m) * x - 1 ↔ 
  (m = 0 → x > 0) ∧ 
  (m > 0 → 0 < x ∧ x < 1 / m) ∧ 
  (m < 0 → x < 1 / m ∨ x > 0) :=
  sorry

end part1_part2_l650_65013


namespace total_tires_parking_lot_l650_65098

-- Definitions for each condition in a)
def four_wheel_drive_cars := 30
def motorcycles := 20
def six_wheel_trucks := 10
def bicycles := 5
def unicycles := 3
def baby_strollers := 2

def extra_roof_tires := 4
def flat_bike_tires_removed := 3
def extra_unicycle_wheel := 1

def tires_per_car := 4 + 1
def tires_per_motorcycle := 2 + 2
def tires_per_truck := 6 + 1
def tires_per_bicycle := 2
def tires_per_unicycle := 1
def tires_per_stroller := 4

-- Define total tires calculation
def total_tires (four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
                 extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel : ℕ) :=
  (four_wheel_drive_cars * tires_per_car + extra_roof_tires) +
  (motorcycles * tires_per_motorcycle) +
  (six_wheel_trucks * tires_per_truck) +
  (bicycles * tires_per_bicycle - flat_bike_tires_removed) +
  (unicycles * tires_per_unicycle + extra_unicycle_wheel) +
  (baby_strollers * tires_per_stroller)

-- The Lean statement for the proof problem
theorem total_tires_parking_lot : 
  total_tires four_wheel_drive_cars motorcycles six_wheel_trucks bicycles unicycles baby_strollers 
              extra_roof_tires flat_bike_tires_removed extra_unicycle_wheel = 323 :=
by 
  sorry

end total_tires_parking_lot_l650_65098


namespace modulus_Z_l650_65071

theorem modulus_Z (Z : ℂ) (h : Z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : Complex.abs Z = 2 := 
sorry

end modulus_Z_l650_65071


namespace increasing_function_l650_65049

theorem increasing_function (k b : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b < (2 * k + 1) * x2 + b) ↔ k > -1/2 := 
by
  sorry

end increasing_function_l650_65049


namespace calculate_expression_l650_65032

theorem calculate_expression (h₁ : x = 7 / 8) (h₂ : y = 5 / 6) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4 * x - 6 * y) / (60 * x * y) = -6 / 175 := 
sorry

end calculate_expression_l650_65032


namespace part1_part2_l650_65082

-- Given conditions for part (Ⅰ)
variables {a_n : ℕ → ℝ} {S_n : ℕ → ℝ}

-- The general formula for the sequence {a_n}
theorem part1 (a3_eq : a_n 3 = 1 / 8)
  (arith_seq : S_n 2 + 1 / 16 = 2 * S_n 3 - S_n 4) :
  ∀ n, a_n n = (1 / 2)^n := sorry

-- Given conditions for part (Ⅱ)
variables {b_n : ℕ → ℝ} {T_n : ℕ → ℝ}

-- The sum of the first n terms of the sequence {b_n}
theorem part2 (h_general : ∀ n, a_n n = (1 / 2)^n)
  (b_formula : ∀ n, b_n n = a_n n * (Real.log (a_n n) / Real.log (1 / 2))) :
  ∀ n, T_n n = 2 - (n + 2) / 2^n := sorry

end part1_part2_l650_65082


namespace company_production_n_l650_65077

theorem company_production_n (n : ℕ) (P : ℕ) 
  (h1 : P = n * 50) 
  (h2 : (P + 90) / (n + 1) = 58) : n = 4 := by 
  sorry

end company_production_n_l650_65077


namespace correct_pronoun_possessive_l650_65079

theorem correct_pronoun_possessive : 
  (∃ (pronoun : String), 
    pronoun = "whose" ∧ 
    pronoun = "whose" ∨ pronoun = "who" ∨ pronoun = "that" ∨ pronoun = "which") := 
by
  -- the proof would go here
  sorry

end correct_pronoun_possessive_l650_65079


namespace sum_of_fractions_l650_65053

theorem sum_of_fractions :
  (2 / 8) + (4 / 8) + (6 / 8) + (8 / 8) + (10 / 8) + 
  (12 / 8) + (14 / 8) + (16 / 8) + (18 / 8) + (20 / 8) = 13.75 :=
by sorry

end sum_of_fractions_l650_65053


namespace slope_of_CD_l650_65064

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 2 * y + 40 = 0

-- Theorem statement
theorem slope_of_CD :
  ∃ C D : ℝ × ℝ,
    (circle1 C.1 C.2) ∧ (circle2 C.1 C.2) ∧ (circle1 D.1 D.2) ∧ (circle2 D.1 D.2) ∧
    (∃ m : ℝ, m = -2 / 3) := 
  sorry

end slope_of_CD_l650_65064


namespace find_triangle_altitude_l650_65051

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l650_65051


namespace alyssa_turnips_l650_65072

theorem alyssa_turnips (k a t: ℕ) (h1: k = 6) (h2: t = 15) (h3: t = k + a) : a = 9 := 
by
  -- proof goes here
  sorry

end alyssa_turnips_l650_65072


namespace jessica_cut_r_l650_65046

variable (r_i r_t r_c : ℕ)

theorem jessica_cut_r : r_i = 7 → r_g = 59 → r_t = 20 → r_c = r_t - r_i → r_c = 13 :=
by
  intros h_i h_g h_t h_c
  have h1 : r_i = 7 := h_i
  have h2 : r_t = 20 := h_t
  have h3 : r_c = r_t - r_i := h_c
  have h_correct : r_c = 13
  · sorry
  exact h_correct

end jessica_cut_r_l650_65046


namespace chimney_problem_l650_65041

variable (x : ℕ) -- number of bricks in the chimney
variable (t : ℕ)
variables (brenda_hours brandon_hours : ℕ)

def brenda_rate := x / brenda_hours
def brandon_rate := x / brandon_hours
def combined_rate := (brenda_rate + brandon_rate - 15) * t

theorem chimney_problem (h1 : brenda_hours = 9)
    (h2 : brandon_hours = 12)
    (h3 : t = 6)
    (h4 : combined_rate = x) : x = 540 := sorry

end chimney_problem_l650_65041


namespace temperature_on_tuesday_l650_65031

variable (T W Th F : ℝ)

-- Conditions
axiom H1 : (T + W + Th) / 3 = 42
axiom H2 : (W + Th + F) / 3 = 44
axiom H3 : F = 43

-- Proof statement
theorem temperature_on_tuesday : T = 37 :=
by
  -- This would be the place to fill in the proof using H1, H2, and H3
  sorry

end temperature_on_tuesday_l650_65031


namespace correct_option_c_l650_65068

variable (a b c : ℝ)

def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom symmetry_axis : -b / (2 * a) = 1

theorem correct_option_c (h : b = -2 * a) : c > 2 * b :=
 sorry

end correct_option_c_l650_65068


namespace max_regions_7_dots_l650_65023

-- Definitions based on conditions provided.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def R (n : ℕ) : ℕ := 1 + binom n 2 + binom n 4

-- The goal is to state the proposition that the maximum number of regions created by joining 7 dots on a circle is 57.
theorem max_regions_7_dots : R 7 = 57 :=
by
  -- The proof is to be filled in here
  sorry

end max_regions_7_dots_l650_65023


namespace regular_polygon_sides_l650_65069

theorem regular_polygon_sides (n : ℕ) (h : 0 < n) (h_angle : (n - 2) * 180 = 144 * n) :
  n = 10 :=
sorry

end regular_polygon_sides_l650_65069


namespace probability_of_spade_or_king_in_two_draws_l650_65066

def total_cards : ℕ := 52
def spades_count : ℕ := 13
def kings_count : ℕ := 4
def king_of_spades_count : ℕ := 1
def spades_or_kings_count : ℕ := spades_count + kings_count - king_of_spades_count
def probability_not_spade_or_king : ℚ := (total_cards - spades_or_kings_count) / total_cards
def probability_both_not_spade_or_king : ℚ := probability_not_spade_or_king^2
def probability_at_least_one_spade_or_king : ℚ := 1 - probability_both_not_spade_or_king

theorem probability_of_spade_or_king_in_two_draws :
  probability_at_least_one_spade_or_king = 88 / 169 :=
sorry

end probability_of_spade_or_king_in_two_draws_l650_65066


namespace problem_1_problem_2_l650_65092

noncomputable def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x + a < 0}

theorem problem_1 (a : ℝ) :
  a = -2 →
  A ∩ B a = {x | (1 / 2 : ℝ) ≤ x ∧ x < 2} :=
by
  intro ha
  sorry

theorem problem_2 (a : ℝ) :
  (A ∩ B a) = A → a < -3 :=
by
  intro h
  sorry

end problem_1_problem_2_l650_65092


namespace find_h_l650_65001

noncomputable def y1 (x h j : ℝ) := 4 * (x - h) ^ 2 + j
noncomputable def y2 (x h k : ℝ) := 3 * (x - h) ^ 2 + k

theorem find_h (h j k : ℝ)
  (C1 : y1 0 h j = 2024)
  (C2 : y2 0 h k = 2025)
  (H1 : y1 x h j = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 506)
  (H2 : y2 x h k = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 675) :
  h = 22.5 :=
sorry

end find_h_l650_65001


namespace f_2008th_derivative_at_0_l650_65086

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4))^6 + (Real.cos (x / 4))^6

theorem f_2008th_derivative_at_0 : (deriv^[2008] f) 0 = 3 / 8 :=
sorry

end f_2008th_derivative_at_0_l650_65086


namespace max_area_inscribed_octagon_l650_65022

theorem max_area_inscribed_octagon
  (R : ℝ)
  (s : ℝ)
  (a b : ℝ)
  (h1 : s^2 = 5)
  (h2 : (a * b) = 4)
  (h3 : (s * Real.sqrt 2) = (2*R))
  (h4 : (Real.sqrt (a^2 + b^2)) = 2 * R) :
  ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  sorry

end max_area_inscribed_octagon_l650_65022


namespace sum_of_four_consecutive_integers_prime_factor_l650_65094

theorem sum_of_four_consecutive_integers_prime_factor (n : ℤ) : ∃ p : ℤ, Prime p ∧ p = 2 ∧ ∀ n : ℤ, p ∣ ((n - 1) + n + (n + 1) + (n + 2)) := 
by 
  sorry

end sum_of_four_consecutive_integers_prime_factor_l650_65094


namespace arithmetic_result_l650_65056

/-- Define the constants involved in the arithmetic operation. -/
def a : ℕ := 999999999999
def b : ℕ := 888888888888
def c : ℕ := 111111111111

/-- The theorem stating that the given arithmetic operation results in the expected answer. -/
theorem arithmetic_result :
  a - b + c = 222222222222 :=
by
  sorry

end arithmetic_result_l650_65056


namespace sprouted_percentage_l650_65080

-- Define the initial conditions
def cherryPits := 80
def saplingsSold := 6
def saplingsLeft := 14

-- Define the calculation of the total saplings that sprouted
def totalSaplingsSprouted := saplingsSold + saplingsLeft

-- Define the percentage calculation
def percentageSprouted := (totalSaplingsSprouted / cherryPits) * 100

-- The theorem to be proved
theorem sprouted_percentage : percentageSprouted = 25 := by
  sorry

end sprouted_percentage_l650_65080


namespace sandwich_is_not_condiments_l650_65057

theorem sandwich_is_not_condiments (sandwich_weight condiments_weight : ℕ)
  (h1 : sandwich_weight = 150)
  (h2 : condiments_weight = 45) :
  (sandwich_weight - condiments_weight) / sandwich_weight * 100 = 70 := 
by sorry

end sandwich_is_not_condiments_l650_65057


namespace paint_area_is_correct_l650_65017

-- Define the dimensions of the wall, window, and door
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def door_height : ℕ := 1
def door_length : ℕ := 7

-- Calculate area
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def door_area : ℕ := door_height * door_length

-- Calculate area to be painted
def area_to_be_painted : ℕ := wall_area - window_area - door_area

-- The theorem statement
theorem paint_area_is_correct : area_to_be_painted = 128 := 
by
  -- The proof would go here (omitted)
  sorry

end paint_area_is_correct_l650_65017


namespace minimal_rope_cost_l650_65028

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end minimal_rope_cost_l650_65028


namespace negation_of_proposition_l650_65000

variable (a b : ℝ)

theorem negation_of_proposition :
  (¬ (a * b = 0 → a = 0 ∨ b = 0)) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_of_proposition_l650_65000


namespace largest_k_for_sum_of_integers_l650_65048

theorem largest_k_for_sum_of_integers (k : ℕ) (n : ℕ) (h1 : 3^12 = k * n + k * (k + 1) / 2) 
  (h2 : k ∣ 2 * 3^12) (h3 : k < 1031) : k ≤ 486 :=
by 
  sorry -- The proof is skipped here, only the statement is required 

end largest_k_for_sum_of_integers_l650_65048


namespace angle_sum_around_point_l650_65044

theorem angle_sum_around_point (y : ℝ) (h : 170 + y + y = 360) : y = 95 := 
sorry

end angle_sum_around_point_l650_65044


namespace chocolate_squares_remaining_l650_65004

theorem chocolate_squares_remaining (m : ℕ) : m * 6 - 21 = 45 :=
by
  sorry

end chocolate_squares_remaining_l650_65004


namespace trirectangular_tetrahedron_max_volume_l650_65095

noncomputable def max_volume_trirectangular_tetrahedron (S : ℝ) : ℝ :=
  S^3 * (Real.sqrt 2 - 1)^3 / 162

theorem trirectangular_tetrahedron_max_volume
  (a b c : ℝ) (H : a > 0 ∧ b > 0 ∧ c > 0)
  (S : ℝ)
  (edge_sum :
    S = a + b + c + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2))
  : ∃ V, V = max_volume_trirectangular_tetrahedron S :=
by
  sorry

end trirectangular_tetrahedron_max_volume_l650_65095


namespace positive_integers_square_less_than_three_times_l650_65090

theorem positive_integers_square_less_than_three_times (n : ℕ) (hn : 0 < n) (ineq : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
by sorry

end positive_integers_square_less_than_three_times_l650_65090


namespace acute_triangle_angles_l650_65029

theorem acute_triangle_angles (α β γ : ℕ) (h1 : α ≥ β) (h2 : β ≥ γ) (h3 : α = 5 * γ) (h4 : α + β + γ = 180) :
  (α = 85 ∧ β = 78 ∧ γ = 17) :=
sorry

end acute_triangle_angles_l650_65029


namespace alpha_plus_2beta_l650_65047

noncomputable def sin_square (θ : ℝ) := (Real.sin θ)^2
noncomputable def sin_double (θ : ℝ) := Real.sin (2 * θ)

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
(hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : 3 * sin_square α + 2 * sin_square β = 1)
(h2 : 3 * sin_double α - 2 * sin_double β = 0) : 
α + 2 * β = 5 * Real.pi / 6 :=
by
  sorry

end alpha_plus_2beta_l650_65047


namespace sum_of_diffs_is_10_l650_65085

-- Define the number of fruits each person has
def Sharon_plums : ℕ := 7
def Allan_plums : ℕ := 10
def Dave_oranges : ℕ := 12

-- Define the differences in the number of fruits
def diff_Sharon_Allan : ℕ := Allan_plums - Sharon_plums
def diff_Sharon_Dave : ℕ := Dave_oranges - Sharon_plums
def diff_Allan_Dave : ℕ := Dave_oranges - Allan_plums

-- Define the sum of these differences
def sum_of_diffs : ℕ := diff_Sharon_Allan + diff_Sharon_Dave + diff_Allan_Dave

-- State the theorem to be proved
theorem sum_of_diffs_is_10 : sum_of_diffs = 10 := by
  sorry

end sum_of_diffs_is_10_l650_65085


namespace unique_solution_a_l650_65015

theorem unique_solution_a (a : ℚ) : 
  (∃ x : ℚ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0 ∧ 
  ∀ y : ℚ, (y ≠ x → (a^2 - 1) * y^2 + (a + 1) * y + 1 ≠ 0)) ↔ a = 1 ∨ a = 5/3 := 
sorry

end unique_solution_a_l650_65015


namespace initial_blue_balls_l650_65099

-- Define the problem conditions
variable (R B : ℕ) -- Number of red balls and blue balls originally in the box.

-- Condition 1: Blue balls are 17 more than red balls
axiom h1 : B = R + 17

-- Condition 2: Ball addition and removal scenario
noncomputable def total_balls_after_changes : ℕ :=
  (B + 57) + (R + 18) - 44

-- Condition 3: Total balls after all changes is 502
axiom h2 : total_balls_after_changes R B = 502

-- We need to prove the initial number of blue balls
theorem initial_blue_balls : B = 244 :=
by
  sorry

end initial_blue_balls_l650_65099


namespace unpainted_cubes_l650_65035

theorem unpainted_cubes (n : ℕ) (cubes_per_face : ℕ) (faces : ℕ) (total_cubes : ℕ) (painted_cubes : ℕ) :
  n = 6 → cubes_per_face = 4 → faces = 6 → total_cubes = 216 → painted_cubes = 24 → 
  total_cubes - painted_cubes = 192 := by
  intros
  sorry

end unpainted_cubes_l650_65035


namespace vector_scalar_sub_l650_65010

def a : ℝ × ℝ := (3, -9)
def b : ℝ × ℝ := (2, -8)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_scalar_sub:
  scalar1 • a - scalar2 • b = (6, -12) := by
  sorry

end vector_scalar_sub_l650_65010


namespace rectangle_area_l650_65063

theorem rectangle_area (a b : ℕ) 
  (h1 : 2 * (a + b) = 16)
  (h2 : a^2 + b^2 - 2 * a * b - 4 = 0) :
  a * b = 30 :=
by
  sorry

end rectangle_area_l650_65063


namespace polynomial_divisibility_l650_65007

-- Define the polynomial f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + m * x - 16

-- Prove that f(x) is divisible by x-2 if and only if m=8
theorem polynomial_divisibility (m : ℝ) :
  (∀ (x : ℝ), (x - 2) ∣ f x m) ↔ m = 8 := 
by
  sorry

end polynomial_divisibility_l650_65007


namespace percentage_scientists_born_in_june_l650_65025

theorem percentage_scientists_born_in_june :
  (18 / 200 * 100) = 9 :=
by sorry

end percentage_scientists_born_in_june_l650_65025


namespace earthquake_energy_multiple_l650_65050

theorem earthquake_energy_multiple (E : ℕ → ℝ) (n9 n7 : ℕ)
  (h1 : E n9 = 10 ^ n9) 
  (h2 : E n7 = 10 ^ n7) 
  (hn9 : n9 = 9) 
  (hn7 : n7 = 7) : 
  E n9 / E n7 = 100 := 
by 
  sorry

end earthquake_energy_multiple_l650_65050


namespace number_of_racks_l650_65005

theorem number_of_racks (cds_per_rack total_cds : ℕ) (h1 : cds_per_rack = 8) (h2 : total_cds = 32) :
  total_cds / cds_per_rack = 4 :=
by
  -- actual proof goes here
  sorry

end number_of_racks_l650_65005


namespace boat_fuel_cost_per_hour_l650_65078

variable (earnings_per_photo : ℕ)
variable (shark_frequency_minutes : ℕ)
variable (hunting_hours : ℕ)
variable (expected_profit : ℕ)

def cost_of_fuel_per_hour (earnings_per_photo shark_frequency_minutes hunting_hours expected_profit : ℕ) : ℕ :=
  sorry

theorem boat_fuel_cost_per_hour
  (h₁ : earnings_per_photo = 15)
  (h₂ : shark_frequency_minutes = 10)
  (h₃ : hunting_hours = 5)
  (h₄ : expected_profit = 200) :
  cost_of_fuel_per_hour earnings_per_photo shark_frequency_minutes hunting_hours expected_profit = 50 :=
  sorry

end boat_fuel_cost_per_hour_l650_65078


namespace basic_spatial_data_source_l650_65026

def source_of_basic_spatial_data (s : String) : Prop :=
  s = "Detailed data provided by high-resolution satellite remote sensing technology" ∨
  s = "Data from various databases provided by high-speed networks" ∨
  s = "Various data collected and organized through the information highway" ∨
  s = "Various spatial exchange data provided by GIS"

theorem basic_spatial_data_source :
  source_of_basic_spatial_data "Data from various databases provided by high-speed networks" :=
sorry

end basic_spatial_data_source_l650_65026


namespace ratio_girls_to_boys_l650_65097

-- Define the number of students and conditions
def num_students : ℕ := 25
def girls_more_than_boys : ℕ := 3

-- Define the variables
variables (g b : ℕ)

-- Define the conditions
def total_students := g + b = num_students
def girls_boys_relationship := b = g - girls_more_than_boys

-- Lean theorem statement
theorem ratio_girls_to_boys (g b : ℕ) (h1 : total_students g b) (h2 : girls_boys_relationship g b) : (g : ℚ) / b = 14 / 11 :=
sorry

end ratio_girls_to_boys_l650_65097


namespace chloe_profit_l650_65030

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end chloe_profit_l650_65030


namespace value_of_u_when_m_is_3_l650_65039

theorem value_of_u_when_m_is_3 :
  ∀ (u t m : ℕ), (t = 3^m + m) → (u = 4^t - 3 * t) → m = 3 → u = 4^30 - 90 :=
by
  intros u t m ht hu hm
  sorry

end value_of_u_when_m_is_3_l650_65039


namespace circle_area_in_sq_cm_l650_65062

theorem circle_area_in_sq_cm (diameter_meters : ℝ) (h : diameter_meters = 5) : 
  let radius_meters := diameter_meters / 2
  let area_square_meters := π * radius_meters^2
  let area_square_cm := area_square_meters * 10000
  area_square_cm = 62500 * π :=
by
  sorry

end circle_area_in_sq_cm_l650_65062


namespace tangent_lines_create_regions_l650_65054

theorem tangent_lines_create_regions (n : ℕ) (h : n = 26) : ∃ k, k = 68 :=
by
  have h1 : ∃ k, k = 68 := ⟨68, rfl⟩
  exact h1

end tangent_lines_create_regions_l650_65054


namespace find_g_neg2_l650_65052

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h_even_f : even_function f)
variables (h_g_def : ∀ x, g x = f x + x^3)
variables (h_g_2 : g 2 = 10)

-- Statement to prove
theorem find_g_neg2 : g (-2) = -6 :=
sorry

end find_g_neg2_l650_65052


namespace find_R_l650_65089

theorem find_R (a b Q R : ℕ) (ha_prime : Prime a) (hb_prime : Prime b) (h_distinct : a ≠ b)
  (h1 : a^2 - a * Q + R = 0) (h2 : b^2 - b * Q + R = 0) : R = 6 :=
sorry

end find_R_l650_65089


namespace karen_average_speed_correct_l650_65073

def karen_time_duration : ℚ := (22 : ℚ) / 3
def karen_distance : ℚ := 230

def karen_average_speed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem karen_average_speed_correct :
  karen_average_speed karen_distance karen_time_duration = (31 + 4/11 : ℚ) :=
by
  sorry

end karen_average_speed_correct_l650_65073
