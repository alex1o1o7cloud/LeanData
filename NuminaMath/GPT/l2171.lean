import Mathlib

namespace fraction_of_grid_covered_by_triangle_l2171_217159

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))|

noncomputable def area_of_grid : ℝ := 7 * 6

noncomputable def fraction_covered : ℝ :=
  area_of_triangle (-1, 2) (3, 5) (2, 2) / area_of_grid

theorem fraction_of_grid_covered_by_triangle : fraction_covered = (3 / 28) :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l2171_217159


namespace unique_y_for_star_l2171_217153

def star (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y - 3

theorem unique_y_for_star : (∀ y : ℝ, star 4 y = 17 → y = 0) ∧ (∃! y : ℝ, star 4 y = 17) := by
  sorry

end unique_y_for_star_l2171_217153


namespace second_player_win_strategy_l2171_217181

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end second_player_win_strategy_l2171_217181


namespace age_of_student_who_left_l2171_217171

variables
  (avg_age_students : ℝ)
  (num_students_before : ℕ)
  (num_students_after : ℕ)
  (age_teacher : ℝ)
  (new_avg_age_class : ℝ)

theorem age_of_student_who_left
  (h1 : avg_age_students = 14)
  (h2 : num_students_before = 45)
  (h3 : num_students_after = 44)
  (h4 : age_teacher = 45)
  (h5 : new_avg_age_class = 14.66)
: ∃ (age_student_left : ℝ), abs (age_student_left - 15.3) < 0.1 :=
sorry

end age_of_student_who_left_l2171_217171


namespace suff_not_necc_condition_l2171_217103

theorem suff_not_necc_condition (x : ℝ) : (x=2) → ((x-2) * (x+5) = 0) ∧ ¬((x-2) * (x+5) = 0 → x=2) :=
by {
  sorry
}

end suff_not_necc_condition_l2171_217103


namespace find_initial_jellybeans_l2171_217185

-- Definitions of the initial conditions
def jellybeans_initial (x : ℝ) (days : ℕ) (remaining : ℝ) := 
  days = 4 ∧ remaining = 48 ∧ (0.7 ^ days) * x = remaining

-- The theorem to prove
theorem find_initial_jellybeans (x : ℝ) : 
  jellybeans_initial x 4 48 → x = 200 :=
sorry

end find_initial_jellybeans_l2171_217185


namespace plates_used_l2171_217150

def plates_per_course : ℕ := 2
def courses_breakfast : ℕ := 2
def courses_lunch : ℕ := 2
def courses_dinner : ℕ := 3
def courses_late_snack : ℕ := 3
def courses_per_day : ℕ := courses_breakfast + courses_lunch + courses_dinner + courses_late_snack
def plates_per_day : ℕ := courses_per_day * plates_per_course

def parents_and_siblings_stay : ℕ := 6
def grandparents_stay : ℕ := 4
def cousins_stay : ℕ := 3

def parents_and_siblings_count : ℕ := 5
def grandparents_count : ℕ := 2
def cousins_count : ℕ := 4

def plates_parents_and_siblings : ℕ := parents_and_siblings_count * plates_per_day * parents_and_siblings_stay
def plates_grandparents : ℕ := grandparents_count * plates_per_day * grandparents_stay
def plates_cousins : ℕ := cousins_count * plates_per_day * cousins_stay

def total_plates_used : ℕ := plates_parents_and_siblings + plates_grandparents + plates_cousins

theorem plates_used (expected : ℕ) : total_plates_used = expected :=
by
  sorry

end plates_used_l2171_217150


namespace length_of_plot_l2171_217125

theorem length_of_plot (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) 
  (h1 : cost_per_meter = 26.50) 
  (h2 : total_cost = 5300)
  (h3 : breadth + 20 = 60) :
  2 * ((breadth + 20) + breadth) = total_cost / cost_per_meter := 
by
  sorry

end length_of_plot_l2171_217125


namespace angle_B_measure_l2171_217196

open Real EuclideanGeometry Classical

noncomputable def measure_angle_B (A C : ℝ) : ℝ := 180 - (180 - A - C)

theorem angle_B_measure
  (l m : ℝ → ℝ → Prop) -- parallel lines l and m (can be interpreted as propositions for simplicity)
  (h_parallel : ∀ x y, l x y → m x y → x = y) -- Lines l and m are parallel
  (A C : ℝ)
  (hA : A = 120)
  (hC : C = 70) :
  measure_angle_B A C = 130 := 
by
  sorry

end angle_B_measure_l2171_217196


namespace y_coordinate_equidistant_l2171_217105

theorem y_coordinate_equidistant : ∃ y : ℝ, (∀ A B : ℝ × ℝ, A = (-3, 0) → B = (-2, 5) → dist (0, y) A = dist (0, y) B) ∧ y = 2 :=
by
  sorry

end y_coordinate_equidistant_l2171_217105


namespace perpendicular_lines_l2171_217173

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y - a = 0) → (a * x - (2 * a - 3) * y - 1 = 0) → 
    (∀ x y : ℝ, ( -1 / a ) * ( -a / (2 * a - 3)) = 1 )) → a = 3 := 
by
  sorry

end perpendicular_lines_l2171_217173


namespace next_term_in_geometric_sequence_l2171_217141

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end next_term_in_geometric_sequence_l2171_217141


namespace gcd_gx_x_l2171_217186

noncomputable def g (x : ℕ) := (5 * x + 3) * (11 * x + 2) * (6 * x + 7) * (3 * x + 8)

theorem gcd_gx_x {x : ℕ} (hx : 36000 ∣ x) : Nat.gcd (g x) x = 144 := by
  sorry

end gcd_gx_x_l2171_217186


namespace gcd_solutions_l2171_217148

theorem gcd_solutions (x m n p: ℤ) (h_eq: x * (4 * x - 5) = 7) (h_gcd: Int.gcd m (Int.gcd n p) = 1)
  (h_form: ∃ x1 x2: ℤ, x1 = (m + Int.sqrt n) / p ∧ x2 = (m - Int.sqrt n) / p) : m + n + p = 150 :=
by
  have disc_eq : 25 + 112 = 137 :=
    by norm_num
  sorry

end gcd_solutions_l2171_217148


namespace prime_factorization_min_x_l2171_217122

-- Define the conditions
variable (x y : ℕ) (a b e f : ℕ)

-- Given conditions: x and y are positive integers, and 5x^7 = 13y^11
axiom condition1 : 0 < x ∧ 0 < y
axiom condition2 : 5 * x^7 = 13 * y^11

-- Prove the mathematical equivalence
theorem prime_factorization_min_x (a b e f : ℕ) 
    (hx : 5 * x^7 = 13 * y^11)
    (h_prime : a = 13 ∧ b = 5 ∧ e = 6 ∧ f = 1) :
    a + b + e + f = 25 :=
sorry

end prime_factorization_min_x_l2171_217122


namespace find_missing_values_l2171_217177

theorem find_missing_values :
  (∃ x y : ℕ, 4 / 5 = 20 / x ∧ 4 / 5 = y / 20 ∧ 4 / 5 = 80 / 100) →
  (x = 25 ∧ y = 16 ∧ 4 / 5 = 80 / 100) :=
by
  sorry

end find_missing_values_l2171_217177


namespace points_for_victory_l2171_217131

theorem points_for_victory (V : ℕ) :
  (∃ (played total_games : ℕ) (points_after_games : ℕ) (remaining_games : ℕ) (needed_points : ℕ) 
     (draw_points defeat_points : ℕ) (minimum_wins : ℕ), 
     played = 5 ∧
     total_games = 20 ∧ 
     points_after_games = 12 ∧
     remaining_games = total_games - played ∧
     needed_points = 40 - points_after_games ∧
     draw_points = 1 ∧
     defeat_points = 0 ∧
     minimum_wins = 7 ∧
     7 * V ≥ needed_points ∧
     remaining_games = total_games - played ∧
     needed_points = 28) → V = 4 :=
sorry

end points_for_victory_l2171_217131


namespace range_of_a_l2171_217166

theorem range_of_a
  (a : ℝ)
  (h : ∀ (x : ℝ), 1 < x ∧ x < 4 → x^2 - 3 * x - 2 - a > 0) :
  a < 2 :=
sorry

end range_of_a_l2171_217166


namespace find_list_price_l2171_217197

noncomputable def list_price (x : ℝ) (alice_price_diff bob_price_diff : ℝ) (alice_comm_fraction bob_comm_fraction : ℝ) : Prop :=
  alice_comm_fraction * (x - alice_price_diff) = bob_comm_fraction * (x - bob_price_diff)

theorem find_list_price : list_price 40 15 25 0.15 0.25 :=
by
  sorry

end find_list_price_l2171_217197


namespace john_total_payment_in_month_l2171_217108

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l2171_217108


namespace other_number_is_29_l2171_217190

theorem other_number_is_29
    (k : ℕ)
    (some_number : ℕ)
    (h1 : k = 2)
    (h2 : (5 + k) * (5 - k) = some_number - 2^3) :
    some_number = 29 :=
by
  sorry

end other_number_is_29_l2171_217190


namespace total_servings_daily_l2171_217101

def cost_per_serving : ℕ := 14
def price_A : ℕ := 20
def price_B : ℕ := 18
def total_revenue : ℕ := 1120
def total_profit : ℕ := 280

theorem total_servings_daily (x y : ℕ) (h1 : price_A * x + price_B * y = total_revenue)
                             (h2 : (price_A - cost_per_serving) * x + (price_B - cost_per_serving) * y = total_profit) :
                             x + y = 60 := sorry

end total_servings_daily_l2171_217101


namespace numbers_less_than_reciprocal_l2171_217155

theorem numbers_less_than_reciprocal :
  (1 / 3 < 3) ∧ (1 / 2 < 2) ∧ ¬(1 < 1) ∧ ¬(2 < 1 / 2) ∧ ¬(3 < 1 / 3) :=
by
  sorry

end numbers_less_than_reciprocal_l2171_217155


namespace visitors_correct_l2171_217129

def visitors_that_day : ℕ := 92
def visitors_previous_day : ℕ := 419
def total_visitors_before_that_day : ℕ := 522
def visitors_two_days_before : ℕ := total_visitors_before_that_day - visitors_previous_day - visitors_that_day

theorem visitors_correct : visitors_two_days_before = 11 := by
  -- Sorry, proof to be filled in
  sorry

end visitors_correct_l2171_217129


namespace gcd_187_119_base5_l2171_217124

theorem gcd_187_119_base5 :
  ∃ b : Nat, Nat.gcd 187 119 = 17 ∧ 17 = 3 * 5 + 2 ∧ 3 = 0 * 5 + 3 ∧ b = 3 * 10 + 2 := by
  sorry

end gcd_187_119_base5_l2171_217124


namespace product_mod_9_l2171_217167

theorem product_mod_9 (a b c : ℕ) (h1 : a % 6 = 2) (h2 : b % 7 = 3) (h3 : c % 8 = 4) : (a * b * c) % 9 = 6 :=
by
  sorry

end product_mod_9_l2171_217167


namespace dice_sum_probability_15_l2171_217198
open Nat

theorem dice_sum_probability_15 (n : ℕ) (h : n = 3432) : 
  ∃ d1 d2 d3 d4 d5 d6 d7 d8 : ℕ,
  (1 ≤ d1 ∧ d1 ≤ 6) ∧ (1 ≤ d2 ∧ d2 ≤ 6) ∧ 
  (1 ≤ d3 ∧ d3 ≤ 6) ∧ (1 ≤ d4 ∧ d4 ≤ 6) ∧ 
  (1 ≤ d5 ∧ d5 ≤ 6) ∧ (1 ≤ d6 ∧ d6 ≤ 6) ∧ 
  (1 ≤ d7 ∧ d7 ≤ 6) ∧ (1 ≤ d8 ∧ d8 ≤ 6) ∧ 
  (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 = 15) :=
by
  sorry

end dice_sum_probability_15_l2171_217198


namespace gcd_6051_10085_l2171_217145

theorem gcd_6051_10085 : Nat.gcd 6051 10085 = 2017 := by
  sorry

end gcd_6051_10085_l2171_217145


namespace max_value_on_ellipse_l2171_217117

theorem max_value_on_ellipse (b : ℝ) (hb : b > 0) :
  ∃ (M : ℝ), 
    (∀ (x y : ℝ), (x^2 / 4 + y^2 / b^2 = 1) → x^2 + 2 * y ≤ M) ∧
    ((b ≤ 4 → M = b^2 / 4 + 4) ∧ (b > 4 → M = 2 * b)) :=
  sorry

end max_value_on_ellipse_l2171_217117


namespace exists_midpoint_with_integer_coordinates_l2171_217111

theorem exists_midpoint_with_integer_coordinates (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ ((points i).1 + (points j).1) % 2 = 0 ∧ ((points i).2 + (points j).2) % 2 = 0 :=
by
  sorry

end exists_midpoint_with_integer_coordinates_l2171_217111


namespace new_perimeter_after_adding_tiles_l2171_217144

-- Define the original condition as per the problem statement
def original_T_shape (n : ℕ) : Prop :=
  n = 6

def original_perimeter (p : ℕ) : Prop :=
  p = 12

-- Define hypothesis required to add three more tiles while sharing a side with existing tiles
def add_three_tiles_with_shared_side (original_tiles : ℕ) (new_tiles_added : ℕ) : Prop :=
  original_tiles + new_tiles_added = 9

-- Prove the new perimeter after adding three tiles to the original T-shaped figure
theorem new_perimeter_after_adding_tiles
  (n : ℕ) (p : ℕ) (new_tiles : ℕ) (new_p : ℕ)
  (h1 : original_T_shape n)
  (h2 : original_perimeter p)
  (h3 : add_three_tiles_with_shared_side n new_tiles)
  : new_p = 16 :=
sorry

end new_perimeter_after_adding_tiles_l2171_217144


namespace crayon_boxes_needed_l2171_217175

theorem crayon_boxes_needed (total_crayons : ℕ) (crayons_per_box : ℕ) (h1 : total_crayons = 80) (h2 : crayons_per_box = 8) : (total_crayons / crayons_per_box) = 10 :=
by
  sorry

end crayon_boxes_needed_l2171_217175


namespace probability_not_exceed_60W_l2171_217107

noncomputable def total_bulbs : ℕ := 250
noncomputable def bulbs_100W : ℕ := 100
noncomputable def bulbs_60W : ℕ := 50
noncomputable def bulbs_25W : ℕ := 50
noncomputable def bulbs_15W : ℕ := 50

noncomputable def probability_of_event (event : ℕ) (total : ℕ) : ℝ := 
  event / total

noncomputable def P_A : ℝ := probability_of_event bulbs_60W total_bulbs
noncomputable def P_B : ℝ := probability_of_event bulbs_25W total_bulbs
noncomputable def P_C : ℝ := probability_of_event bulbs_15W total_bulbs
noncomputable def P_D : ℝ := probability_of_event bulbs_100W total_bulbs

theorem probability_not_exceed_60W : 
  P_A + P_B + P_C = 3 / 5 :=
by
  sorry

end probability_not_exceed_60W_l2171_217107


namespace steps_per_flight_l2171_217135

-- Define the problem conditions
def jack_flights_up := 3
def jack_flights_down := 6
def steps_height_inches := 8
def jack_height_change_feet := 24

-- Convert the height change to inches
def jack_height_change_inches := jack_height_change_feet * 12

-- Calculate the net flights down
def net_flights_down := jack_flights_down - jack_flights_up

-- Calculate total height change in inches for net flights
def total_height_change_inches := net_flights_down * jack_height_change_inches

-- Calculate the number of steps in each flight
def number_of_steps_per_flight :=
  total_height_change_inches / (steps_height_inches * net_flights_down)

theorem steps_per_flight :
  number_of_steps_per_flight = 108 :=
sorry

end steps_per_flight_l2171_217135


namespace cosine_double_angle_l2171_217113

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l2171_217113


namespace isPossible_l2171_217106

structure Person where
  firstName : String
  patronymic : String
  surname : String

def conditions (people : List Person) : Prop :=
  people.length = 4 ∧
  ∀ p1 p2 p3 : Person, 
    p1 ∈ people → p2 ∈ people → p3 ∈ people →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    p1.firstName ≠ p2.firstName ∨ p2.firstName ≠ p3.firstName ∨ p1.firstName ≠ p3.firstName ∧
    p1.patronymic ≠ p2.patronymic ∨ p2.patronymic ≠ p3.patronymic ∨ p1.patronymic ≠ p3.patronymic ∧
    p1.surname ≠ p2.surname ∨ p2.surname ≠ p3.surname ∨ p1.surname ≠ p3.surname ∧
  ∀ p1 p2 : Person, 
    p1 ∈ people → p2 ∈ people →
    p1 ≠ p2 →
    p1.firstName = p2.firstName ∨ p1.patronymic = p2.patronymic ∨ p1.surname = p2.surname

theorem isPossible : ∃ people : List Person, conditions people := by
  sorry

end isPossible_l2171_217106


namespace freshmen_minus_sophomores_eq_24_l2171_217152

def total_students := 800
def percent_juniors := 27 / 100
def percent_not_sophomores := 75 / 100
def number_seniors := 160

def number_juniors := percent_juniors * total_students
def number_not_sophomores := percent_not_sophomores * total_students
def number_sophomores := total_students - number_not_sophomores
def number_freshmen := total_students - (number_juniors + number_sophomores + number_seniors)

theorem freshmen_minus_sophomores_eq_24 :
  number_freshmen - number_sophomores = 24 :=
sorry

end freshmen_minus_sophomores_eq_24_l2171_217152


namespace number_of_australians_l2171_217170

-- Conditions are given here as definitions
def total_people : ℕ := 49
def number_americans : ℕ := 16
def number_chinese : ℕ := 22

-- Goal is to prove the number of Australians is 11
theorem number_of_australians : total_people - (number_americans + number_chinese) = 11 := by
  sorry

end number_of_australians_l2171_217170


namespace compare_abc_l2171_217192

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem compare_abc : a > c ∧ c > b := by
  unfold a b c
  -- Proof goes here
  sorry

end compare_abc_l2171_217192


namespace abc_sum_16_l2171_217139

theorem abc_sum_16 (a b c : ℕ) (h1 : a ≥ 4) (h2 : b ≥ 4) (h3 : c ≥ 4) (h4 : a ≠ b ∨ b ≠ c ∨ a ≠ c)
  (h5 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by
  sorry

end abc_sum_16_l2171_217139


namespace new_average_l2171_217134

variable (avg9 : ℝ) (score10 : ℝ) (n : ℕ)
variable (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9)

theorem new_average (h : avg9 = 80) (h10 : score10 = 100) (n9 : n = 9) :
  ((n * avg9 + score10) / (n + 1)) = 82 :=
by
  rw [h, h10, n9]
  sorry

end new_average_l2171_217134


namespace positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l2171_217189

theorem positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5 : 
  ∃ (x : ℕ), (x = 594) ∧ (18 ∣ x) ∧ (24 ≤ Real.sqrt (x) ∧ Real.sqrt (x) ≤ 24.5) := 
by 
  sorry

end positive_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l2171_217189


namespace area_OPA_l2171_217194

variable (x : ℝ)

def y (x : ℝ) : ℝ := -x + 6

def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, y x)

def area_triangle (O A P : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.fst * P.snd + P.fst * O.snd + O.fst * A.snd - A.snd * P.fst - P.snd * O.fst - O.snd * A.fst)

theorem area_OPA : 0 < x ∧ x < 6 → area_triangle O A (P x) = 12 - 2 * x := by
  -- proof to be provided here
  sorry


end area_OPA_l2171_217194


namespace correct_product_l2171_217138

-- Definitions for conditions
def reversed_product (a b : ℕ) : Prop :=
  let reversed_a := (a % 10) * 10 + (a / 10)
  reversed_a * b = 204

theorem correct_product (a b : ℕ) (h : reversed_product a b) : a * b = 357 := 
by
  sorry

end correct_product_l2171_217138


namespace dot_product_not_sufficient_nor_necessary_for_parallel_l2171_217188

open Real

-- Definitions for plane vectors \overrightarrow{a} and \overrightarrow{b}
variables (a b : ℝ × ℝ)

-- Dot product definition for two plane vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Parallelism condition for plane vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k • v2) ∨ v2 = (k • v1)

-- Statement to be proved
theorem dot_product_not_sufficient_nor_necessary_for_parallel :
  ¬ (∀ a b : ℝ × ℝ, (dot_product a b > 0) ↔ (parallel a b)) :=
sorry

end dot_product_not_sufficient_nor_necessary_for_parallel_l2171_217188


namespace expand_and_simplify_l2171_217120

variable (y : ℝ)

theorem expand_and_simplify :
  -2 * (5 * y^3 - 4 * y^2 + 3 * y - 6) = -10 * y^3 + 8 * y^2 - 6 * y + 12 :=
  sorry

end expand_and_simplify_l2171_217120


namespace min_distance_eq_5_l2171_217137

-- Define the conditions
def condition1 (a b : ℝ) : Prop := b = 4 * Real.log a - a^2
def condition2 (c d : ℝ) : Prop := d = 2 * c + 2

-- Define the function to prove the minimum value
def minValue (a b c d : ℝ) : ℝ := (a - c)^2 + (b - d)^2

-- The main theorem statement
theorem min_distance_eq_5 (a b c d : ℝ) (ha : a > 0) (h1: condition1 a b) (h2: condition2 c d) : 
  ∃ a c b d, minValue a b c d = 5 := 
sorry

end min_distance_eq_5_l2171_217137


namespace income_of_m_l2171_217163

theorem income_of_m (M N O : ℝ)
  (h1 : (M + N) / 2 = 5050)
  (h2 : (N + O) / 2 = 6250)
  (h3 : (M + O) / 2 = 5200) :
  M = 4000 :=
by
  -- sorry is used to skip the actual proof.
  sorry

end income_of_m_l2171_217163


namespace sum_of_digits_base8_product_l2171_217116

theorem sum_of_digits_base8_product
  (a b : ℕ)
  (a_base8 : a = 3 * 8^1 + 4 * 8^0)
  (b_base8 : b = 2 * 8^1 + 2 * 8^0)
  (product : ℕ := a * b)
  (product_base8 : ℕ := (product / 64) * 8^2 + ((product / 8) % 8) * 8^1 + (product % 8)) :
  ((product_base8 / 8^2) + ((product_base8 / 8) % 8) + (product_base8 % 8)) = 1 * 8^1 + 6 * 8^0 :=
sorry

end sum_of_digits_base8_product_l2171_217116


namespace find_ordered_pairs_l2171_217132

theorem find_ordered_pairs (x y : ℝ) :
  x^2 * y = 3 ∧ x + x * y = 4 → (x, y) = (1, 3) ∨ (x, y) = (3, 1 / 3) :=
sorry

end find_ordered_pairs_l2171_217132


namespace expanded_polynomial_correct_l2171_217142

noncomputable def polynomial_product (x : ℚ) : ℚ :=
  (2 * x^3 - 3 * x + 1) * (x^2 + 4 * x + 3)

theorem expanded_polynomial_correct (x : ℚ) : 
  polynomial_product x = 2 * x^5 + 8 * x^4 + 3 * x^3 - 11 * x^2 - 5 * x + 3 := 
by
  sorry

end expanded_polynomial_correct_l2171_217142


namespace mark_more_hours_than_kate_l2171_217102

-- Definitions for the problem
variable (K : ℕ)  -- K is the number of hours charged by Kate
variable (P : ℕ)  -- P is the number of hours charged by Pat
variable (M : ℕ)  -- M is the number of hours charged by Mark

-- Conditions
def total_hours := K + P + M = 216
def pat_kate_relation := P = 2 * K
def pat_mark_relation := P = (1 / 3) * M

-- The statement to be proved
theorem mark_more_hours_than_kate (K P M : ℕ) (h1 : total_hours K P M)
  (h2 : pat_kate_relation K P) (h3 : pat_mark_relation P M) :
  (M - K = 120) :=
by
  sorry

end mark_more_hours_than_kate_l2171_217102


namespace weight_of_new_person_l2171_217127

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → old_weight = 60 → 
  new_weight = old_weight + num_persons * avg_increase → new_weight = 80 :=
  by
    intros
    sorry

end weight_of_new_person_l2171_217127


namespace quadratic_transformation_l2171_217182

theorem quadratic_transformation :
  ∀ (x : ℝ), (x^2 + 6*x - 2 = 0) → ((x + 3)^2 = 11) :=
by
  intros x h
  sorry

end quadratic_transformation_l2171_217182


namespace range_of_a_l2171_217184

section
  variable {x a : ℝ}

  -- Define set A
  def setA : Set ℝ := { x | x^2 - 4*x + 3 < 0 }

  -- Define set B
  def setB (a : ℝ) : Set ℝ := 
    { x | (2*x + a ≤ 0) ∧ (x^2 - 2*(a + 7)*x + 5 ≤ 0)}

  -- The proof problem statement
  theorem range_of_a (a : ℝ) : 
    (setA ⊆ setB a) ↔ (-4 ≤ a ∧ a ≤ -2) :=
  sorry
end

end range_of_a_l2171_217184


namespace isosceles_trapezoid_AC_length_l2171_217110

noncomputable def length_of_AC (AB AD BC CD AC : ℝ) :=
  AB = 30 ∧ AD = 15 ∧ BC = 15 ∧ CD = 12 → AC = 23.32

theorem isosceles_trapezoid_AC_length :
  length_of_AC 30 15 15 12 23.32 := by
  sorry

end isosceles_trapezoid_AC_length_l2171_217110


namespace line_inclination_angle_l2171_217156

theorem line_inclination_angle (θ : ℝ) : 
  (∃ θ : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → θ = 3 * π / 4) := sorry

end line_inclination_angle_l2171_217156


namespace minimum_bench_sections_l2171_217165

theorem minimum_bench_sections (N : ℕ) (hN : 8 * N = 12 * N) : N = 3 :=
sorry

end minimum_bench_sections_l2171_217165


namespace shadow_length_correct_l2171_217174

theorem shadow_length_correct :
  let light_source := (0, 16)
  let disc_center := (6, 10)
  let radius := 2
  let m := 4
  let n := 17
  let length_form := m * Real.sqrt n
  length_form = 4 * Real.sqrt 17 :=
by
  sorry

end shadow_length_correct_l2171_217174


namespace anna_current_age_l2171_217162

theorem anna_current_age (A : ℕ) (Clara_now : ℕ) (years_ago : ℕ) (Clara_age_ago : ℕ) 
    (H1 : Clara_now = 80) 
    (H2 : years_ago = 41) 
    (H3 : Clara_age_ago = Clara_now - years_ago) 
    (H4 : Clara_age_ago = 3 * (A - years_ago)) : 
    A = 54 :=
by
  sorry

end anna_current_age_l2171_217162


namespace relationship_between_abc_l2171_217123

noncomputable def a : Real := Real.sqrt 1.2
noncomputable def b : Real := Real.exp 0.1
noncomputable def c : Real := 1 + Real.log 1.1

theorem relationship_between_abc : b > a ∧ a > c :=
by {
  -- a = sqrt(1.2)
  -- b = exp(0.1)
  -- c = 1 + log(1.1)
  -- We need to prove: b > a > c
  sorry
}

end relationship_between_abc_l2171_217123


namespace total_surface_area_of_new_solid_l2171_217118

-- Define the heights of the pieces using the given conditions
def height_A := 1 / 4
def height_B := 1 / 5
def height_C := 1 / 6
def height_D := 1 / 7
def height_E := 1 / 8
def height_F := 1 - (height_A + height_B + height_C + height_D + height_E)

-- Assembling the pieces back in reverse order (F to A), encapsulate the total surface area calculation
theorem total_surface_area_of_new_solid : 
  (2 * (1 : ℝ)) + (2 * (1 * 1 : ℝ)) + (2 * (1 * 1 : ℝ)) = 6 :=
by
  sorry

end total_surface_area_of_new_solid_l2171_217118


namespace total_high_sulfur_samples_l2171_217100

-- Define the conditions as given in the problem
def total_samples : ℕ := 143
def heavy_oil_freq : ℚ := 2 / 11
def light_low_sulfur_freq : ℚ := 7 / 13
def no_low_sulfur_in_heavy_oil : Prop := ∀ (x : ℕ), (x / total_samples = heavy_oil_freq) → false

-- Define total high-sulfur samples
def num_heavy_oil := heavy_oil_freq * total_samples
def num_light_oil := total_samples - num_heavy_oil
def num_light_low_sulfur_oil := light_low_sulfur_freq * num_light_oil
def num_light_high_sulfur_oil := num_light_oil - num_light_low_sulfur_oil

-- Now state that we need to prove the total number of high-sulfur samples
theorem total_high_sulfur_samples : num_light_high_sulfur_oil + num_heavy_oil = 80 :=
by
  sorry

end total_high_sulfur_samples_l2171_217100


namespace rectangle_perimeter_l2171_217195

theorem rectangle_perimeter : 
  ∃ (x y a b : ℝ), 
  (x * y = 2016) ∧ 
  (a * b = 2016) ∧ 
  (x^2 + y^2 = 4 * (a^2 - b^2)) → 
  2 * (x + y) = 8 * Real.sqrt 1008 :=
sorry

end rectangle_perimeter_l2171_217195


namespace original_strength_of_class_l2171_217149

-- Definitions from the problem conditions
def average_age_original (x : ℕ) : ℕ := 40 * x
def total_students (x : ℕ) : ℕ := x + 17
def total_age_new_students : ℕ := 17 * 32
def new_average_age : ℕ := 36

-- Lean statement to prove that the original strength of the class is 17.
theorem original_strength_of_class :
  ∃ x : ℕ, average_age_original x + total_age_new_students = total_students x * new_average_age ∧ x = 17 :=
by
  sorry

end original_strength_of_class_l2171_217149


namespace landscape_breadth_l2171_217128

theorem landscape_breadth (L B : ℕ)
  (h1 : B = 6 * L)
  (h2 : 4200 = (1 / 7 : ℚ) * 6 * L^2) :
  B = 420 := 
  sorry

end landscape_breadth_l2171_217128


namespace discarded_marble_weight_l2171_217169

-- Define the initial weight of the marble block and the weights of the statues
def initial_weight : ℕ := 80
def weight_statue_1 : ℕ := 10
def weight_statue_2 : ℕ := 18
def weight_statue_3 : ℕ := 15
def weight_statue_4 : ℕ := 15

-- The proof statement: the discarded weight of marble is 22 pounds.
theorem discarded_marble_weight :
  initial_weight - (weight_statue_1 + weight_statue_2 + weight_statue_3 + weight_statue_4) = 22 :=
by
  sorry

end discarded_marble_weight_l2171_217169


namespace complement_union_l2171_217143

variable (U : Set ℤ)
variable (A : Set ℤ)
variable (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} :=
sorry

end complement_union_l2171_217143


namespace units_digit_of_sum_64_8_75_8_is_1_l2171_217178

def units_digit_in_base_8_sum (a b : ℕ) : ℕ :=
  (a + b) % 8

theorem units_digit_of_sum_64_8_75_8_is_1 :
  units_digit_in_base_8_sum 0o64 0o75 = 1 :=
sorry

end units_digit_of_sum_64_8_75_8_is_1_l2171_217178


namespace factor_expression_1_factor_expression_2_l2171_217183

theorem factor_expression_1 (a b c : ℝ) : a^2 + 2 * a * b + b^2 + a * c + b * c = (a + b) * (a + b + c) :=
  sorry

theorem factor_expression_2 (a x y : ℝ) : 4 * a^2 - x^2 + 4 * x * y - 4 * y^2 = (2 * a + x - 2 * y) * (2 * a - x + 2 * y) :=
  sorry

end factor_expression_1_factor_expression_2_l2171_217183


namespace polynomial_divisibility_by_5_l2171_217158

theorem polynomial_divisibility_by_5
  (a b c d : ℤ)
  (divisible : ∀ x : ℤ, 5 ∣ (a * x ^ 3 + b * x ^ 2 + c * x + d)) :
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d :=
sorry

end polynomial_divisibility_by_5_l2171_217158


namespace find_triangle_C_coordinates_find_triangle_area_l2171_217133

noncomputable def triangle_C_coordinates (A B : (ℝ × ℝ)) (median_eq altitude_eq : (ℝ × ℝ × ℝ)) : Prop :=
  ∃ C : ℝ × ℝ, C = (3, 1) ∧
    let A := (1,2)
    let B := (3, 4)
    let median_eq := (2, 1, -7)
    let altitude_eq := (2, -1, -2)
    true

noncomputable def triangle_area (A B C : (ℝ × ℝ)) : Prop :=
  ∃ S : ℝ, S = 3 ∧
    let A := (1,2)
    let B := (3, 4)
    let C := (3, 1)
    true

theorem find_triangle_C_coordinates : triangle_C_coordinates (1,2) (3,4) (2, 1, -7) (2, -1, -2) :=
by { sorry }

theorem find_triangle_area : triangle_area (1,2) (3,4) (3,1) :=
by { sorry }

end find_triangle_C_coordinates_find_triangle_area_l2171_217133


namespace cost_of_game_l2171_217104

theorem cost_of_game
  (number_of_ice_creams : ℕ) 
  (price_per_ice_cream : ℕ)
  (total_sold : number_of_ice_creams = 24)
  (price : price_per_ice_cream = 5) :
  (number_of_ice_creams * price_per_ice_cream) / 2 = 60 :=
by
  sorry

end cost_of_game_l2171_217104


namespace joey_more_fish_than_peter_l2171_217199

-- Define the conditions
variables (A P J : ℕ)

-- Condition that Ali's fish weight is twice that of Peter's
def ali_double_peter (A P : ℕ) : Prop := A = 2 * P

-- Condition that Ali caught 12 kg of fish
def ali_caught_12 (A : ℕ) : Prop := A = 12

-- Condition that the total weight of the fish is 25 kg
def total_weight (A P J : ℕ) : Prop := A + P + J = 25

-- Prove that Joey caught 1 kg more fish than Peter
theorem joey_more_fish_than_peter (A P J : ℕ) :
  ali_double_peter A P → ali_caught_12 A → total_weight A P J → J = 1 :=
by 
  intro h1 h2 h3
  sorry

end joey_more_fish_than_peter_l2171_217199


namespace songs_today_is_14_l2171_217187

-- Define the number of songs Jeremy listened to yesterday
def songs_yesterday (x : ℕ) : ℕ := x

-- Define the number of songs Jeremy listened to today
def songs_today (x : ℕ) : ℕ := x + 5

-- Given conditions
def total_songs (x : ℕ) : Prop := songs_yesterday x + songs_today x = 23

-- Prove the number of songs Jeremy listened to today
theorem songs_today_is_14 : ∃ x: ℕ, total_songs x ∧ songs_today x = 14 :=
by {
  sorry
}

end songs_today_is_14_l2171_217187


namespace total_cost_l2171_217146

-- Given conditions
def pen_cost : ℕ := 4
def briefcase_cost : ℕ := 5 * pen_cost

-- Theorem stating the total cost Marcel paid for both items
theorem total_cost (pen_cost briefcase_cost : ℕ) (h_pen: pen_cost = 4) (h_briefcase: briefcase_cost = 5 * pen_cost) :
  pen_cost + briefcase_cost = 24 := by
  sorry

end total_cost_l2171_217146


namespace expression_value_l2171_217164

def α : ℝ := 60
def β : ℝ := 20
def AB : ℝ := 1

noncomputable def γ : ℝ := 180 - (α + β)

noncomputable def AC : ℝ := AB * (Real.sin γ / Real.sin β)
noncomputable def BC : ℝ := (Real.sin α / Real.sin γ) * AB

theorem expression_value : (1 / AC - BC) = 2 := by
  sorry

end expression_value_l2171_217164


namespace quadratic_function_n_neg_l2171_217115

theorem quadratic_function_n_neg (n : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + n = 0 → x > 0) → n < 0 :=
by
  sorry

end quadratic_function_n_neg_l2171_217115


namespace find_a_and_other_root_l2171_217126

-- Define the quadratic equation with a
def quadratic_eq (a x : ℝ) : ℝ := (a + 1) * x^2 + x - 1

-- Define the conditions where -1 is a root
def condition (a : ℝ) : Prop := quadratic_eq a (-1) = 0

theorem find_a_and_other_root (a : ℝ) :
  condition a → 
  (a = 1 ∧ ∃ x : ℝ, x ≠ -1 ∧ quadratic_eq 1 x = 0 ∧ x = 1 / 2) :=
by
  intro h
  sorry

end find_a_and_other_root_l2171_217126


namespace even_num_Z_tetrominoes_l2171_217121

-- Definitions based on the conditions of the problem
def is_tiled_with_S_tetrominoes (P : Type) : Prop := sorry
def tiling_uses_S_Z_tetrominoes (P : Type) : Prop := sorry
def num_Z_tetrominoes (P : Type) : ℕ := sorry

-- The theorem statement
theorem even_num_Z_tetrominoes (P : Type) 
  (hTiledWithS : is_tiled_with_S_tetrominoes P) 
  (hTilingWithSZ : tiling_uses_S_Z_tetrominoes P) : num_Z_tetrominoes P % 2 = 0 :=
sorry

end even_num_Z_tetrominoes_l2171_217121


namespace compute_c_plus_d_l2171_217147

theorem compute_c_plus_d (c d : ℝ) 
  (h1 : c^3 - 18 * c^2 + 25 * c - 75 = 0) 
  (h2 : 9 * d^3 - 72 * d^2 - 345 * d + 3060 = 0) : 
  c + d = 10 := 
sorry

end compute_c_plus_d_l2171_217147


namespace susan_spending_ratio_l2171_217176

theorem susan_spending_ratio (initial_amount clothes_spent books_left books_spent left_after_clothes gcd_ratio : ℤ)
  (h1 : initial_amount = 600)
  (h2 : clothes_spent = initial_amount / 2)
  (h3 : left_after_clothes = initial_amount - clothes_spent)
  (h4 : books_left = 150)
  (h5 : books_spent = left_after_clothes - books_left)
  (h6 : gcd books_spent left_after_clothes = 150)
  (h7 : books_spent / gcd_ratio = 1)
  (h8 : left_after_clothes / gcd_ratio = 2) :
  books_spent / gcd books_spent left_after_clothes = 1 ∧ left_after_clothes / gcd books_spent left_after_clothes = 2 :=
sorry

end susan_spending_ratio_l2171_217176


namespace sheila_hourly_wage_l2171_217157

-- Sheila works 8 hours per day on Monday, Wednesday, and Friday
-- Sheila works 6 hours per day on Tuesday and Thursday
-- Sheila does not work on Saturday and Sunday
-- Sheila earns $288 per week

def hours_worked (monday_wednesday_friday_hours : Nat) (tuesday_thursday_hours : Nat) : Nat :=
  (monday_wednesday_friday_hours * 3) + (tuesday_thursday_hours * 2)

def weekly_earnings : Nat := 288
def total_hours_worked : Nat := hours_worked 8 6
def hourly_wage : Nat := weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage = 8 := by
  -- Proof (omitted)
  sorry

end sheila_hourly_wage_l2171_217157


namespace right_triangle_construction_condition_l2171_217119

theorem right_triangle_construction_condition (A B C : Point) (b d : ℝ) :
  AC = b → AC + BC - AB = d → b > d :=
by
  intro h1 h2
  sorry

end right_triangle_construction_condition_l2171_217119


namespace expand_product_l2171_217136

-- Define the problem
theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 :=
by
  sorry

end expand_product_l2171_217136


namespace candy_problem_l2171_217151

theorem candy_problem (a : ℕ) (h₁ : a % 10 = 6) (h₂ : a % 15 = 11) (h₃ : 200 ≤ a) (h₄ : a ≤ 250) :
  a = 206 ∨ a = 236 :=
sorry

end candy_problem_l2171_217151


namespace inverse_of_5_mod_35_l2171_217179

theorem inverse_of_5_mod_35 : (5 * 28) % 35 = 1 :=
by
  sorry

end inverse_of_5_mod_35_l2171_217179


namespace modified_pyramid_volume_l2171_217160

theorem modified_pyramid_volume (s h : ℝ) (V : ℝ) 
  (hV : V = 1/3 * s^2 * h) (hV_eq : V = 72) :
  (1/3) * (3 * s)^2 * (2 * h) = 1296 := by
  sorry

end modified_pyramid_volume_l2171_217160


namespace tangent_circles_distance_l2171_217180

-- Define the radii of the circles.
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 2

-- Define the condition that the circles are tangent.
def tangent (r1 r2 d : ℝ) : Prop :=
  d = r1 + r2 ∨ d = r1 - r2

-- State the theorem.
theorem tangent_circles_distance (d : ℝ) :
  tangent radius_O1 radius_O2 d → (d = 1 ∨ d = 5) :=
by
  sorry

end tangent_circles_distance_l2171_217180


namespace minimum_value_of_2a5_a4_l2171_217140

variable {a : ℕ → ℝ} {q : ℝ}

-- Defining that the given sequence is geometric, i.e., a_{n+1} = a_n * q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

-- The condition given in the problem is
def condition (a : ℕ → ℝ) : Prop :=
2 * a 4 + a 3 - 2 * a 2 - a 1 = 8

-- The sequence is positive
def positive_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n > 0

theorem minimum_value_of_2a5_a4 (h_geom : is_geometric_sequence a q) (h_cond : condition a) (h_pos : positive_sequence a) (h_q : q > 0) :
  2 * a 5 + a 4 = 12 * Real.sqrt 3 :=
sorry

end minimum_value_of_2a5_a4_l2171_217140


namespace unique_integer_m_l2171_217193

theorem unique_integer_m :
  ∃! (m : ℤ), m - ⌊m / (2005 : ℝ)⌋ = 2005 :=
by
  --- Here belongs the proof part, but we leave it with a sorry
  sorry

end unique_integer_m_l2171_217193


namespace train_speed_kmh_l2171_217191

def man_speed_kmh : ℝ := 3 -- The man's speed in km/h
def train_length_m : ℝ := 110 -- The train's length in meters
def passing_time_s : ℝ := 12 -- Time taken to pass the man in seconds

noncomputable def man_speed_ms : ℝ := (man_speed_kmh * 1000) / 3600 -- Convert man's speed to m/s

theorem train_speed_kmh :
  (110 / 12) - (5 / 6) * (3600 / 1000) = 30 := by
  -- Omitted steps will go here
  sorry

end train_speed_kmh_l2171_217191


namespace minimum_possible_n_l2171_217172

theorem minimum_possible_n (n p : ℕ) (h1: p > 0) (h2: 15 * n - 45 = 105) : n = 10 :=
sorry

end minimum_possible_n_l2171_217172


namespace geom_seq_a4_a5_a6_value_l2171_217109

theorem geom_seq_a4_a5_a6_value (a : ℕ → ℝ) (h_geom : ∃ r, 0 < r ∧ ∀ n, a (n + 1) = r * a n)
  (h_roots : ∃ x y, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 9 = y) :
  a 4 * a 5 * a 6 = 64 :=
by
  sorry

end geom_seq_a4_a5_a6_value_l2171_217109


namespace factorize_quartic_l2171_217154

-- Specify that p and q are real numbers (ℝ)
variables {p q : ℝ}

-- Statement: For any real numbers p and q, the polynomial x^4 + p x^2 + q can always be factored into two quadratic polynomials.
theorem factorize_quartic (p q : ℝ) : 
  ∃ a b c d e f : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + p * x^2 + q :=
sorry

end factorize_quartic_l2171_217154


namespace solve_quadratic1_solve_quadratic2_l2171_217114

theorem solve_quadratic1 :
  (∀ x, x^2 + x - 4 = 0 → x = ( -1 + Real.sqrt 17 ) / 2 ∨ x = ( -1 - Real.sqrt 17 ) / 2) := sorry

theorem solve_quadratic2 :
  (∀ x, (2*x + 1)^2 + 15 = 8*(2*x + 1) → x = 1 ∨ x = 2) := sorry

end solve_quadratic1_solve_quadratic2_l2171_217114


namespace find_a10_l2171_217130

variable {G : Type*} [LinearOrderedField G]
variable (a : ℕ → G)

-- Conditions
def geometric_sequence (a : ℕ → G) (r : G) := ∀ n, a (n + 1) = r * a n
def positive_terms (a : ℕ → G) := ∀ n, 0 < a n
def specific_condition (a : ℕ → G) := a 3 * a 11 = 16

theorem find_a10
  (h_geom : geometric_sequence a 2)
  (h_pos : positive_terms a)
  (h_cond : specific_condition a) :
  a 10 = 32 := by
  sorry

end find_a10_l2171_217130


namespace factor_expression_l2171_217168

theorem factor_expression (x : ℕ) : 63 * x + 54 = 9 * (7 * x + 6) :=
by
  sorry

end factor_expression_l2171_217168


namespace find_base_of_exponent_l2171_217161

theorem find_base_of_exponent
  (x : ℝ)
  (h1 : 4 ^ (2 * x + 2) = (some_number : ℝ) ^ (3 * x - 1))
  (x_eq : x = 1) :
  some_number = 16 := 
by
  -- proof steps would go here
  sorry

end find_base_of_exponent_l2171_217161


namespace true_discount_correct_l2171_217112

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  (BD * FV) / (BD + FV)

theorem true_discount_correct :
  true_discount 270 54 = 45 :=
by
  sorry

end true_discount_correct_l2171_217112
