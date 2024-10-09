import Mathlib

namespace find_fraction_l2240_224093

theorem find_fraction : 
  ∀ (x : ℚ), (120 - x * 125 = 45) → x = 3 / 5 :=
by
  intro x
  intro h
  sorry

end find_fraction_l2240_224093


namespace original_average_of_15_numbers_l2240_224049

theorem original_average_of_15_numbers (A : ℝ) (h1 : 15 * A + 15 * 12 = 52 * 15) :
  A = 40 :=
sorry

end original_average_of_15_numbers_l2240_224049


namespace sets_relationship_l2240_224086

def set_M : Set ℝ := {x | x^2 - 2 * x > 0}
def set_N : Set ℝ := {x | x > 3}

theorem sets_relationship : set_M ∩ set_N = set_N := by
  sorry

end sets_relationship_l2240_224086


namespace find_m_l2240_224054

-- Definitions for the conditions
def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_of_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = a 1 * (1 - (a n / a 1)) / (1 - (a 2 / a 1))

def arithmetic_sequence (S3 S9 S6 : ℝ) :=
  2 * S9 = S3 + S6

def condition_3 (a : ℕ → ℝ) (m : ℕ) :=
  a 2 + a 5 = 2 * a m

-- Lean 4 statement that requires proof
theorem find_m 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ) 
  (geom_seq : geometric_sequence a a1 q)
  (sum_geom_seq : sum_of_geometric_sequence S a)
  (arith_seq : arithmetic_sequence (S 3) (S 9) (S 6))
  (cond3 : condition_3 a 8) : 
  8 = 8 := 
sorry

end find_m_l2240_224054


namespace church_distance_l2240_224066

def distance_to_church (speed : ℕ) (hourly_rate : ℕ) (flat_fee : ℕ) (total_paid : ℕ) : ℕ :=
  let hours := (total_paid - flat_fee) / hourly_rate
  hours * speed

theorem church_distance :
  distance_to_church 10 30 20 80 = 20 :=
by
  sorry

end church_distance_l2240_224066


namespace maximum_possible_value_of_x_l2240_224097

-- Define the conditions and the question
def ten_teams_playing_each_other_once (number_of_teams : ℕ) : Prop :=
  number_of_teams = 10

def points_system (win_points draw_points loss_points : ℕ) : Prop :=
  win_points = 3 ∧ draw_points = 1 ∧ loss_points = 0

def max_points_per_team (x : ℕ) : Prop :=
  x = 13

-- The theorem to be proved: maximum possible value of x given the conditions
theorem maximum_possible_value_of_x :
  ∀ (number_of_teams win_points draw_points loss_points x : ℕ),
    ten_teams_playing_each_other_once number_of_teams →
    points_system win_points draw_points loss_points →
    max_points_per_team x :=
  sorry

end maximum_possible_value_of_x_l2240_224097


namespace find_cost_price_l2240_224072

theorem find_cost_price (C : ℝ) (SP : ℝ) (M : ℝ) (h1 : SP = 1.25 * C) (h2 : 0.90 * M = SP) (h3 : SP = 65.97) : 
  C = 52.776 :=
by
  sorry

end find_cost_price_l2240_224072


namespace value_of_s_for_g_neg_1_eq_0_l2240_224006

def g (x s : ℝ) := 3 * x^5 - 2 * x^3 + x^2 - 4 * x + s

theorem value_of_s_for_g_neg_1_eq_0 (s : ℝ) : g (-1) s = 0 ↔ s = -4 :=
by
  sorry

end value_of_s_for_g_neg_1_eq_0_l2240_224006


namespace average_rate_of_change_is_4_l2240_224068

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_is_4_l2240_224068


namespace zander_stickers_l2240_224039

/-- Zander starts with 100 stickers, Andrew receives 1/5 of Zander's total, 
    and Bill receives 3/10 of the remaining stickers. Prove that the total 
    number of stickers given to Andrew and Bill is 44. -/
theorem zander_stickers :
  let total_stickers := 100
  let andrew_fraction := 1 / 5
  let remaining_stickers := total_stickers - (total_stickers * andrew_fraction)
  let bill_fraction := 3 / 10
  (total_stickers * andrew_fraction) + (remaining_stickers * bill_fraction) = 44 := 
by
  sorry

end zander_stickers_l2240_224039


namespace remainder_when_divided_by_20_l2240_224000

theorem remainder_when_divided_by_20 (n : ℕ) : (4 * 6^n + 5^(n-1)) % 20 = 9 := 
by
  sorry

end remainder_when_divided_by_20_l2240_224000


namespace part1_part2_l2240_224077

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Part (1): Given m = 4, prove A ∪ B = {x | -2 ≤ x ∧ x ≤ 7}
theorem part1 : A ∪ B 4 = {x | -2 ≤ x ∧ x ≤ 7} :=
by
  sorry

-- Part (2): Given B ⊆ A, prove m ∈ (-∞, 3]
theorem part2 {m : ℝ} (h : B m ⊆ A) : m ∈ Set.Iic 3 :=
by
  sorry

end part1_part2_l2240_224077


namespace toy_cost_price_l2240_224023

theorem toy_cost_price (x : ℝ) (h : 1.5 * x * 0.8 - x = 20) : x = 100 := 
sorry

end toy_cost_price_l2240_224023


namespace prize_expectation_l2240_224028

theorem prize_expectation :
  let total_people := 100
  let envelope_percentage := 0.4
  let grand_prize_prob := 0.1
  let second_prize_prob := 0.2
  let consolation_prize_prob := 0.3
  let people_with_envelopes := total_people * envelope_percentage
  let grand_prize_winners := people_with_envelopes * grand_prize_prob
  let second_prize_winners := people_with_envelopes * second_prize_prob
  let consolation_prize_winners := people_with_envelopes * consolation_prize_prob
  let empty_envelopes := people_with_envelopes - (grand_prize_winners + second_prize_winners + consolation_prize_winners)
  grand_prize_winners = 4 ∧
  second_prize_winners = 8 ∧
  consolation_prize_winners = 12 ∧
  empty_envelopes = 16 := by
  sorry

end prize_expectation_l2240_224028


namespace inequality_holds_l2240_224027

theorem inequality_holds (x : ℝ) : x + 2 < x + 3 := 
by {
    sorry
}

end inequality_holds_l2240_224027


namespace length_of_living_room_l2240_224065

theorem length_of_living_room (L : ℝ) (width : ℝ) (border_width : ℝ) (border_area : ℝ) 
  (h1 : width = 10)
  (h2 : border_width = 2)
  (h3 : border_area = 72) :
  L = 12 :=
by
  sorry

end length_of_living_room_l2240_224065


namespace no_solution_exists_l2240_224071

open Nat

theorem no_solution_exists : ¬ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 ^ x + 3 ^ y - 5 ^ z = 2 * 11 :=
by
  sorry

end no_solution_exists_l2240_224071


namespace M_subset_N_l2240_224058

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end M_subset_N_l2240_224058


namespace quadratic_root_reciprocal_l2240_224043

theorem quadratic_root_reciprocal (p q r s : ℝ) 
    (h1 : ∃ a : ℝ, a^2 + p * a + q = 0 ∧ (1 / a)^2 + r * (1 / a) + s = 0) :
    (p * s - r) * (q * r - p) = (q * s - 1)^2 :=
by
  sorry

end quadratic_root_reciprocal_l2240_224043


namespace number_of_tiles_l2240_224035

theorem number_of_tiles (w l : ℕ) (h1 : 2 * w + 2 * l - 4 = (w * l - (2 * w + 2 * l - 4)))
  (h2 : w > 0) (h3 : l > 0) : w * l = 48 ∨ w * l = 60 :=
by
  sorry

end number_of_tiles_l2240_224035


namespace sum_of_first_12_terms_geometric_sequence_l2240_224074

variable {α : Type*} [Field α]

def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (Finset.range n).sum a

theorem sum_of_first_12_terms_geometric_sequence
  (a : ℕ → α)
  (h_geo : geometric_sequence a)
  (h_sum1 : sum_first_n_terms a 3 = 4)
  (h_sum2 : sum_first_n_terms a 6 - sum_first_n_terms a 3 = 8) :
  sum_first_n_terms a 12 = 60 := 
sorry

end sum_of_first_12_terms_geometric_sequence_l2240_224074


namespace sum_of_roots_eq_14_l2240_224031

theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, (x - 7)^2 = 16 → (x = 11 ∨ x = 3)) →
  (∀ x1 x2 : ℝ, (x1 - 7)^2 = 16 ∧ (x2 - 7)^2 = 16 ∧ x1 ≠ x2 → x1 + x2 = 14) :=
by
  intros h x1 x2 h_comb
  sorry

end sum_of_roots_eq_14_l2240_224031


namespace Q_after_move_up_4_units_l2240_224026

-- Define the initial coordinates.
def Q_initial : (ℤ × ℤ) := (-4, -6)

-- Define the transformation - moving up 4 units.
def move_up (P : ℤ × ℤ) (units : ℤ) : (ℤ × ℤ) := (P.1, P.2 + units)

-- State the theorem to be proved.
theorem Q_after_move_up_4_units : move_up Q_initial 4 = (-4, -2) :=
by 
  sorry

end Q_after_move_up_4_units_l2240_224026


namespace sin_cos_identity_proof_l2240_224032

noncomputable def solution : ℝ := Real.sin (Real.pi / 6) * Real.cos (Real.pi / 12) + Real.cos (Real.pi / 6) * Real.sin (Real.pi / 12)

theorem sin_cos_identity_proof : solution = Real.sqrt 2 / 2 := by
  sorry

end sin_cos_identity_proof_l2240_224032


namespace green_and_yellow_peaches_total_is_correct_l2240_224057

-- Define the number of red, yellow, and green peaches
def red_peaches : ℕ := 5
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6

-- Definition of the total number of green and yellow peaches
def total_green_and_yellow_peaches : ℕ := green_peaches + yellow_peaches

-- Theorem stating that the total number of green and yellow peaches is 20
theorem green_and_yellow_peaches_total_is_correct : total_green_and_yellow_peaches = 20 :=
by 
  sorry

end green_and_yellow_peaches_total_is_correct_l2240_224057


namespace largest_sphere_radius_l2240_224092

-- Define the conditions
def inner_radius : ℝ := 3
def outer_radius : ℝ := 7
def circle_center_x := 5
def circle_center_z := 2
def circle_radius := 2

-- Define the question into a statement
noncomputable def radius_of_largest_sphere : ℝ :=
  (29 : ℝ) / 4

-- Prove the required radius given the conditions
theorem largest_sphere_radius:
  ∀ (r : ℝ),
  r = radius_of_largest_sphere → r * r = inner_radius * inner_radius + (circle_center_x * circle_center_x + (r - circle_center_z) * (r - circle_center_z))
:=
by
  sorry

end largest_sphere_radius_l2240_224092


namespace total_cost_div_selling_price_eq_23_div_13_l2240_224081

-- Conditions from part (a)
def pencil_count := 140
def pen_count := 90
def eraser_count := 60

def loss_pencils := 70
def loss_pens := 30
def loss_erasers := 20

def pen_cost (P : ℝ) := P
def pencil_cost (P : ℝ) := 2 * P
def eraser_cost (P : ℝ) := 1.5 * P

def total_cost (P : ℝ) :=
  pencil_count * pencil_cost P +
  pen_count * pen_cost P +
  eraser_count * eraser_cost P

def loss (P : ℝ) :=
  loss_pencils * pencil_cost P +
  loss_pens * pen_cost P +
  loss_erasers * eraser_cost P

def selling_price (P : ℝ) :=
  total_cost P - loss P

-- Statement to be proved: the total cost is 23/13 times the selling price.
theorem total_cost_div_selling_price_eq_23_div_13 (P : ℝ) :
  total_cost P / selling_price P = 23 / 13 := by
  sorry

end total_cost_div_selling_price_eq_23_div_13_l2240_224081


namespace arun_borrowed_amount_l2240_224096

theorem arun_borrowed_amount :
  ∃ P : ℝ, 
    (P * 0.08 * 4 + P * 0.10 * 6 + P * 0.12 * 5 = 12160) → P = 8000 :=
sorry

end arun_borrowed_amount_l2240_224096


namespace ride_cost_l2240_224080

theorem ride_cost (joe_age_over_18 : Prop)
                   (joe_brother_age : Nat)
                   (joe_entrance_fee : ℝ)
                   (brother_entrance_fee : ℝ)
                   (total_spending : ℝ)
                   (rides_per_person : Nat)
                   (total_persons : Nat)
                   (total_entrance_fee : ℝ)
                   (amount_spent_on_rides : ℝ)
                   (total_rides : Nat) :
  joe_entrance_fee = 6 →
  brother_entrance_fee = 5 →
  total_spending = 20.5 →
  rides_per_person = 3 →
  total_persons = 3 →
  total_entrance_fee = 16 →
  amount_spent_on_rides = (total_spending - total_entrance_fee) →
  total_rides = (rides_per_person * total_persons) →
  (amount_spent_on_rides / total_rides) = 0.50 :=
by
  sorry

end ride_cost_l2240_224080


namespace arc_length_condition_l2240_224079

open Real

noncomputable def hyperbola_eq (a b x y: ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem arc_length_condition (a b r: ℝ) (h1: hyperbola_eq a b 2 1) (h2: r > 0)
  (h3: ∃ x y, x^2 + y^2 = r^2 ∧ hyperbola_eq a b x y) :
  r > 2 * sqrt 2 :=
sorry

end arc_length_condition_l2240_224079


namespace eval_expr_x_eq_3_y_eq_4_l2240_224001

theorem eval_expr_x_eq_3_y_eq_4 : 
  ∀ (x y : ℕ), x = 3 → y = 4 → 5 * x^y + 6 * y^x + x * y = 801 := 
by 
  intros x y hx hy 
  rw [hx, hy]
  -- Proof omitted
  sorry

end eval_expr_x_eq_3_y_eq_4_l2240_224001


namespace sum_of_xy_l2240_224064

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l2240_224064


namespace perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l2240_224008

theorem perfect_squares_multiple_of_72 (N : ℕ) : 
  (N^2 < 1000000) ∧ (N^2 % 72 = 0) ↔ N ≤ 996 :=
sorry

theorem number_of_perfect_squares_multiple_of_72 : 
  ∃ upper_bound : ℕ, upper_bound = 83 ∧ ∀ n : ℕ, (n < 1000000) → (n % 144 = 0) → n ≤ (12 * upper_bound) :=
sorry

end perfect_squares_multiple_of_72_number_of_perfect_squares_multiple_of_72_l2240_224008


namespace length_of_platform_l2240_224007

-- Definitions for conditions
def train_length : ℕ := 300
def time_cross_platform : ℕ := 39
def time_cross_signal : ℕ := 12

-- Speed calculation
def train_speed := train_length / time_cross_signal

-- Total distance calculation while crossing the platform
def total_distance := train_speed * time_cross_platform

-- Length of the platform
def platform_length : ℕ := total_distance - train_length

-- Theorem stating the length of the platform
theorem length_of_platform :
  platform_length = 675 := by
  sorry

end length_of_platform_l2240_224007


namespace intersection_complement_l2240_224014

def A : Set ℝ := {x | 0 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_complement :
  A ∩ (compl B) = {x | 0 < x ∧ x < 2} := by
  sorry

end intersection_complement_l2240_224014


namespace perpendicular_lines_b_value_l2240_224018

theorem perpendicular_lines_b_value 
  (b : ℝ) 
  (line1 : ∀ x y : ℝ, x + 3 * y + 5 = 0 → True) 
  (line2 : ∀ x y : ℝ, b * x + 3 * y + 5 = 0 → True)
  (perpendicular_condition : (-1 / 3) * (-b / 3) = -1) : 
  b = -9 := 
sorry

end perpendicular_lines_b_value_l2240_224018


namespace lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l2240_224084

-- Define a polynomial and conditions for divisibility by 7
def poly_deg_6 (a b c d e f g x : ℤ) : ℤ :=
  a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g

-- Theorem for divisibility by 7
theorem lowest_degree_for_divisibility_by_7 : 
  (∀ x : ℤ, poly_deg_6 a b c d e f g x % 7 = 0) → false :=
sorry

-- Define a polynomial and conditions for divisibility by 12
def poly_deg_3 (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

-- Theorem for divisibility by 12
theorem lowest_degree_for_divisibility_by_12 : 
  (∀ x : ℤ, poly_deg_3 a b c d x % 12 = 0) → false :=
sorry

end lowest_degree_for_divisibility_by_7_lowest_degree_for_divisibility_by_12_l2240_224084


namespace find_g_neg1_l2240_224040

-- Define that f(x) is an odd function
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Given conditions
variables (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_eq : ∀ x : ℝ, f x = g x + x^2)
variable (h_g1 : g 1 = 1)

-- The statement to prove
theorem find_g_neg1 : g (-1) = -3 :=
sorry

end find_g_neg1_l2240_224040


namespace initial_pages_l2240_224088

variable (P : ℕ)
variable (h : 20 * P - 20 = 220)

theorem initial_pages (h : 20 * P - 20 = 220) : P = 12 := by
  sorry

end initial_pages_l2240_224088


namespace new_salary_l2240_224038

theorem new_salary (increase : ℝ) (percent_increase : ℝ) (S_new : ℝ) :
  increase = 25000 → percent_increase = 38.46153846153846 → S_new = 90000 :=
by
  sorry

end new_salary_l2240_224038


namespace each_sibling_gets_13_pencils_l2240_224041

theorem each_sibling_gets_13_pencils (colored_pencils black_pencils kept_pencils siblings : ℕ) 
  (h1 : colored_pencils = 14)
  (h2 : black_pencils = 35)
  (h3 : kept_pencils = 10)
  (h4 : siblings = 3) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by
  sorry

end each_sibling_gets_13_pencils_l2240_224041


namespace even_function_a_value_l2240_224048

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) → a = 1 :=
by
  sorry

end even_function_a_value_l2240_224048


namespace solution_quad_ineq_l2240_224013

noncomputable def quadratic_inequality_solution_set :=
  {x : ℝ | (x > -1) ∧ (x < 3) ∧ (x ≠ 2)}

theorem solution_quad_ineq (x : ℝ) :
  ((x^2 - 2*x - 3)*(x^2 - 4*x + 4) < 0) ↔ x ∈ quadratic_inequality_solution_set :=
by sorry

end solution_quad_ineq_l2240_224013


namespace product_is_square_of_24975_l2240_224067

theorem product_is_square_of_24975 : (500 * 49.95 * 4.995 * 5000 : ℝ) = (24975 : ℝ) ^ 2 :=
by {
  sorry
}

end product_is_square_of_24975_l2240_224067


namespace total_flowers_l2240_224099

theorem total_flowers (pots: ℕ) (flowers_per_pot: ℕ) (h_pots: pots = 2150) (h_flowers_per_pot: flowers_per_pot = 128) :
    pots * flowers_per_pot = 275200 :=
by 
    sorry

end total_flowers_l2240_224099


namespace red_jellybeans_count_l2240_224015

theorem red_jellybeans_count (total_jellybeans : ℕ)
  (blue_jellybeans : ℕ)
  (purple_jellybeans : ℕ)
  (orange_jellybeans : ℕ)
  (H1 : total_jellybeans = 200)
  (H2 : blue_jellybeans = 14)
  (H3 : purple_jellybeans = 26)
  (H4 : orange_jellybeans = 40) :
  total_jellybeans - (blue_jellybeans + purple_jellybeans + orange_jellybeans) = 120 :=
by sorry

end red_jellybeans_count_l2240_224015


namespace retail_store_paid_40_percent_more_l2240_224069

variables (C R : ℝ)

-- Condition: The customer price is 96% more than manufacturing cost
def customer_price_from_manufacturing (C : ℝ) : ℝ := 1.96 * C

-- Condition: The customer price is 40% more than the retailer price
def customer_price_from_retail (R : ℝ) : ℝ := 1.40 * R

-- Theorem to be proved
theorem retail_store_paid_40_percent_more (C R : ℝ) 
  (h_customer_price : customer_price_from_manufacturing C = customer_price_from_retail R) :
  (R - C) / C = 0.40 :=
by
  sorry

end retail_store_paid_40_percent_more_l2240_224069


namespace correct_operation_l2240_224090

theorem correct_operation (x : ℝ) : (-x^3)^2 = x^6 :=
by sorry

end correct_operation_l2240_224090


namespace integer_triangle_answer_l2240_224063

def integer_triangle_condition :=
∀ a r : ℕ, (1 ≤ a ∧ a ≤ 19) → 
(a = 12) → (r = 3) → 
(r = 96 / (20 + a))

theorem integer_triangle_answer : 
  integer_triangle_condition := 
by
  sorry

end integer_triangle_answer_l2240_224063


namespace gcd_expression_l2240_224003

theorem gcd_expression (a : ℤ) (k : ℤ) (h1 : a = k * 1171) (h2 : k % 2 = 1) (prime_1171 : Prime 1171) : 
  Int.gcd (3 * a^2 + 35 * a + 77) (a + 15) = 1 :=
by
  sorry

end gcd_expression_l2240_224003


namespace problem_l2240_224021

theorem problem (n : ℕ) (p : ℕ) (a b c : ℤ)
  (hn : 0 < n)
  (hp : Nat.Prime p)
  (h_eq : a^n + p * b = b^n + p * c)
  (h_eq2 : b^n + p * c = c^n + p * a) :
  a = b ∧ b = c := 
sorry

end problem_l2240_224021


namespace floral_arrangement_carnations_percentage_l2240_224044

theorem floral_arrangement_carnations_percentage :
  ∀ (F : ℕ),
  (1 / 4) * (7 / 10) * F + (2 / 3) * (3 / 10) * F = (29 / 40) * F :=
by
  sorry

end floral_arrangement_carnations_percentage_l2240_224044


namespace cost_of_pencil_l2240_224089

theorem cost_of_pencil (s n c : ℕ) (h_majority : s > 15) (h_pencils : n > 1) (h_cost : c > n)
  (h_total_cost : s * c * n = 1771) : c = 11 :=
sorry

end cost_of_pencil_l2240_224089


namespace complement_intersection_example_l2240_224051

open Set

variable (U A B : Set ℕ)

def C_U (A : Set ℕ) (U : Set ℕ) : Set ℕ := U \ A

theorem complement_intersection_example 
  (hU : U = {0, 1, 2, 3})
  (hA : A = {0, 1})
  (hB : B = {1, 2, 3}) :
  (C_U A U) ∩ B = {2, 3} :=
by
  sorry

end complement_intersection_example_l2240_224051


namespace necessary_condition_for_inequality_l2240_224076

theorem necessary_condition_for_inequality (m : ℝ) :
  (∀ x : ℝ, (x^2 - 3 * x + 2 < 0) → (x > m)) ∧ (∃ x : ℝ, (x > m) ∧ ¬(x^2 - 3 * x + 2 < 0)) → m ≤ 1 := 
by
  sorry

end necessary_condition_for_inequality_l2240_224076


namespace cube_dimension_l2240_224052

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l2240_224052


namespace value_of_expression_l2240_224047

open Polynomial

theorem value_of_expression (a b : ℚ) (h1 : (3 : ℚ) * a ^ 2 + 9 * a - 21 = 0) (h2 : (3 : ℚ) * b ^ 2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (2 * b - 2) = -4 :=
by sorry

end value_of_expression_l2240_224047


namespace find_new_songs_l2240_224045

-- Definitions for the conditions
def initial_songs : ℕ := 6
def deleted_songs : ℕ := 3
def final_songs : ℕ := 23

-- The number of new songs added
def new_songs_added : ℕ := 20

-- Statement of the proof problem
theorem find_new_songs (n d f x : ℕ) (h1 : n = initial_songs) (h2 : d = deleted_songs) (h3 : f = final_songs) : f = n - d + x → x = new_songs_added :=
by
  intros h4
  sorry

end find_new_songs_l2240_224045


namespace fraction_to_decimal_subtraction_l2240_224082

theorem fraction_to_decimal_subtraction 
    (h : (3 : ℚ) / 40 = 0.075) : 
    0.075 - 0.005 = 0.070 := 
by 
    sorry

end fraction_to_decimal_subtraction_l2240_224082


namespace polynomial_min_value_P_l2240_224025

theorem polynomial_min_value_P (a b : ℝ) (h_root_pos : ∀ x, a * x^3 - x^2 + b * x - 1 = 0 → 0 < x) :
    (∀ x : ℝ, a * x^3 - x^2 + b * x - 1 = 0 → x > 0) →
    ∃ P : ℝ, P = 12 * Real.sqrt 3 :=
sorry

end polynomial_min_value_P_l2240_224025


namespace no_three_digit_numbers_with_sum_27_are_even_l2240_224004

-- We define a 3-digit number and its conditions based on digit-sum and even properties
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem no_three_digit_numbers_with_sum_27_are_even :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ is_even n :=
by sorry

end no_three_digit_numbers_with_sum_27_are_even_l2240_224004


namespace find_additional_speed_l2240_224091

noncomputable def speed_initial : ℝ := 55
noncomputable def t_initial : ℝ := 4
noncomputable def speed_total : ℝ := 60
noncomputable def t_total : ℝ := 6

theorem find_additional_speed :
  let distance_initial := speed_initial * t_initial
  let distance_total := speed_total * t_total
  let t_additional := t_total - t_initial
  let distance_additional := distance_total - distance_initial
  let speed_additional := distance_additional / t_additional
  speed_additional = 70 :=
by
  sorry

end find_additional_speed_l2240_224091


namespace cookies_baked_total_l2240_224010

   -- Definitions based on the problem conditions
   def cookies_yesterday : ℕ := 435
   def cookies_this_morning : ℕ := 139

   -- The theorem we want to prove
   theorem cookies_baked_total : cookies_yesterday + cookies_this_morning = 574 :=
   by sorry
   
end cookies_baked_total_l2240_224010


namespace max_mn_value_min_4m_square_n_square_l2240_224050

variable {m n : ℝ}
variable (h_cond1 : m > 0)
variable (h_cond2 : n > 0)
variable (h_eq : 2 * m + n = 1)

theorem max_mn_value : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ m * n = 1/8) := 
  sorry

theorem min_4m_square_n_square : (∃ m n : ℝ, m > 0 ∧ n > 0 ∧ 2 * m + n = 1 ∧ 4 * m^2 + n^2 = 1/2) := 
  sorry

end max_mn_value_min_4m_square_n_square_l2240_224050


namespace percentage_of_green_ducks_l2240_224061

theorem percentage_of_green_ducks (ducks_small_pond ducks_large_pond : ℕ) 
  (green_fraction_small_pond green_fraction_large_pond : ℚ) 
  (h1 : ducks_small_pond = 20) 
  (h2 : ducks_large_pond = 80) 
  (h3 : green_fraction_small_pond = 0.20) 
  (h4 : green_fraction_large_pond = 0.15) :
  let total_ducks := ducks_small_pond + ducks_large_pond
  let green_ducks := (green_fraction_small_pond * ducks_small_pond) + 
                     (green_fraction_large_pond * ducks_large_pond)
  (green_ducks / total_ducks) * 100 = 16 := 
by 
  sorry

end percentage_of_green_ducks_l2240_224061


namespace milk_production_l2240_224083

theorem milk_production 
  (initial_cows : ℕ)
  (initial_milk : ℕ)
  (initial_days : ℕ)
  (max_milk_per_cow_per_day : ℕ)
  (available_cows : ℕ)
  (days : ℕ)
  (H_initial : initial_cows = 10)
  (H_initial_milk : initial_milk = 40)
  (H_initial_days : initial_days = 5)
  (H_max_milk : max_milk_per_cow_per_day = 2)
  (H_available_cows : available_cows = 15)
  (H_days : days = 8) :
  available_cows * initial_milk / (initial_cows * initial_days) * days = 96 := 
by 
  sorry

end milk_production_l2240_224083


namespace annual_decrease_rate_l2240_224030

theorem annual_decrease_rate (r : ℝ) 
  (h1 : 15000 * (1 - r / 100)^2 = 9600) : 
  r = 20 := 
sorry

end annual_decrease_rate_l2240_224030


namespace distinct_ordered_pairs_count_l2240_224062

theorem distinct_ordered_pairs_count : 
  ∃ (n : ℕ), (∀ (a b : ℕ), a + b = 50 → 0 ≤ a ∧ 0 ≤ b) ∧ n = 51 :=
by
  sorry

end distinct_ordered_pairs_count_l2240_224062


namespace train_speed_is_36_kph_l2240_224078

noncomputable def speed_of_train (length_train length_bridge time_to_pass : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_mps := total_distance / time_to_pass
  let speed_kph := speed_mps * 3600 / 1000
  speed_kph

theorem train_speed_is_36_kph :
  speed_of_train 360 140 50 = 36 :=
by
  sorry

end train_speed_is_36_kph_l2240_224078


namespace height_of_spheres_l2240_224029

theorem height_of_spheres (R r : ℝ) (h : ℝ) :
  0 < r ∧ r < R → h = R - Real.sqrt ((3 * R^2 - 6 * R * r - r^2) / 3) :=
by
  intros h0
  sorry

end height_of_spheres_l2240_224029


namespace volume_rectangular_box_l2240_224085

variables {l w h : ℝ}

theorem volume_rectangular_box (h1 : l * w = 30) (h2 : w * h = 20) (h3 : l * h = 12) :
  l * w * h = 60 :=
sorry

end volume_rectangular_box_l2240_224085


namespace group_size_l2240_224019

def total_blocks : ℕ := 820
def num_groups : ℕ := 82

theorem group_size :
  total_blocks / num_groups = 10 := 
by 
  sorry

end group_size_l2240_224019


namespace total_stoppage_time_per_hour_l2240_224042

variables (speed_ex_stoppages_1 speed_in_stoppages_1 : ℕ)
variables (speed_ex_stoppages_2 speed_in_stoppages_2 : ℕ)
variables (speed_ex_stoppages_3 speed_in_stoppages_3 : ℕ)

-- Definitions of the speeds given in the problem's conditions.
def speed_bus_1_ex_stoppages := 54
def speed_bus_1_in_stoppages := 36
def speed_bus_2_ex_stoppages := 60
def speed_bus_2_in_stoppages := 40
def speed_bus_3_ex_stoppages := 72
def speed_bus_3_in_stoppages := 48

-- The main theorem to be proved.
theorem total_stoppage_time_per_hour :
  ((1 - speed_bus_1_in_stoppages / speed_bus_1_ex_stoppages : ℚ)
   + (1 - speed_bus_2_in_stoppages / speed_bus_2_ex_stoppages : ℚ)
   + (1 - speed_bus_3_in_stoppages / speed_bus_3_ex_stoppages : ℚ)) = 1 := by
  sorry

end total_stoppage_time_per_hour_l2240_224042


namespace matrix_power_B150_l2240_224070

open Matrix

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

-- Prove that B^150 = I
theorem matrix_power_B150 : 
  (B ^ 150 = (1 : Matrix (Fin 3) (Fin 3) ℝ)) :=
by
  sorry

end matrix_power_B150_l2240_224070


namespace first_person_amount_l2240_224009

theorem first_person_amount (A B C : ℕ) (h1 : A = 28) (h2 : B = 72) (h3 : C = 98) (h4 : A + B + C = 198) (h5 : 99 ≤ max (A + B) (B + C) / 2) : 
  A = 28 :=
by
  -- placeholder for proof
  sorry

end first_person_amount_l2240_224009


namespace factorization_correct_l2240_224060

theorem factorization_correct :
  ∀ (y : ℝ), (y^2 - 1 = (y + 1) * (y - 1)) :=
by
  intro y
  sorry

end factorization_correct_l2240_224060


namespace expression_not_defined_l2240_224056

theorem expression_not_defined (x : ℝ) : 
  (x^2 - 21 * x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by
sorry

end expression_not_defined_l2240_224056


namespace combined_savings_after_5_years_l2240_224036

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + (r / n)) ^ (n * t)

theorem combined_savings_after_5_years :
  let P1 := 600
  let r1 := 0.10
  let n1 := 12
  let t := 5
  let P2 := 400
  let r2 := 0.08
  let n2 := 4
  compound_interest P1 r1 n1 t + compound_interest P2 r2 n2 t = 1554.998 :=
by
  sorry

end combined_savings_after_5_years_l2240_224036


namespace contrapositive_l2240_224020

variable (k : ℝ)

theorem contrapositive (h : ¬∃ x : ℝ, x^2 - x - k = 0) : k ≤ 0 :=
sorry

end contrapositive_l2240_224020


namespace swimming_speed_l2240_224011

theorem swimming_speed (v_m v_s : ℝ) 
  (h1 : v_m + v_s = 6)
  (h2 : v_m - v_s = 8) : 
  v_m = 7 :=
by
  sorry

end swimming_speed_l2240_224011


namespace event_distance_l2240_224037

noncomputable def distance_to_event (cost_per_mile : ℝ) (days : ℕ) (rides_per_day : ℕ) (total_cost : ℝ) : ℝ :=
  total_cost / (days * rides_per_day * cost_per_mile)

theorem event_distance 
  (cost_per_mile : ℝ)
  (days : ℕ)
  (rides_per_day : ℕ)
  (total_cost : ℝ)
  (h1 : cost_per_mile = 2.5)
  (h2 : days = 7)
  (h3 : rides_per_day = 2)
  (h4 : total_cost = 7000) : 
  distance_to_event cost_per_mile days rides_per_day total_cost = 200 :=
by {
  sorry
}

end event_distance_l2240_224037


namespace cubic_of_m_eq_4_l2240_224053

theorem cubic_of_m_eq_4 (m : ℕ) (h : 3 ^ m = 81) : m ^ 3 = 64 := 
by
  sorry

end cubic_of_m_eq_4_l2240_224053


namespace inequality_add_six_l2240_224073

theorem inequality_add_six (x y : ℝ) (h : x < y) : x + 6 < y + 6 :=
sorry

end inequality_add_six_l2240_224073


namespace polynomial_value_l2240_224017

noncomputable def p (x : ℝ) : ℝ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) + 24 * x

theorem polynomial_value :
  (p 1 = 24) ∧ (p 2 = 48) ∧ (p 3 = 72) ∧ (p 4 = 96) →
  p 0 + p 5 = 168 := 
by
  sorry

end polynomial_value_l2240_224017


namespace minimum_value_of_function_l2240_224033

theorem minimum_value_of_function :
  ∀ x : ℝ, (x > -2) → (x + (16 / (x + 2)) ≥ 6) :=
by
  intro x hx
  sorry

end minimum_value_of_function_l2240_224033


namespace solve_10_arithmetic_in_1_minute_l2240_224094

-- Define the times required for each task
def time_math_class : Nat := 40 -- in minutes
def time_walk_kilometer : Nat := 20 -- in minutes
def time_solve_arithmetic : Nat := 1 -- in minutes

-- The question: Which task can be completed in 1 minute?
def task_completed_in_1_minute : Nat := 1

theorem solve_10_arithmetic_in_1_minute :
  time_solve_arithmetic = task_completed_in_1_minute :=
by
  sorry

end solve_10_arithmetic_in_1_minute_l2240_224094


namespace joe_initial_tests_count_l2240_224034

theorem joe_initial_tests_count (n S : ℕ) (h1 : S = 45 * n) (h2 : S - 30 = 50 * (n - 1)) : n = 4 := by
  sorry

end joe_initial_tests_count_l2240_224034


namespace range_of_a2_l2240_224055

theorem range_of_a2 (a : ℕ → ℝ) (S : ℕ → ℝ) (a2 : ℝ) (a3 a6 : ℝ) (h1: 3 * a3 = a6 + 4) (h2 : S 5 < 10) :
  a2 < 2 := 
sorry

end range_of_a2_l2240_224055


namespace part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l2240_224022

variable {x a : ℝ}

theorem part1_solution (h1 : a > 1 / 3) (h2 : (a * x - 1) / (x ^ 2 - 1) = 0) : x = 3 := by
  sorry

theorem part2_solution_1 (h1 : -1 < a) (h2 : a < 0) : {x | x < (1 / a) ∨ (-1 < x ∧ x < 1)} := by
  sorry

theorem part2_solution_2 (h1 : a = -1) : {x | x < 1 ∧ x ≠ -1} := by
  sorry

theorem part2_solution_3 (h1 : a < -1) : {x | x < -1 ∨ (1 / a < x ∧ x < 1)} := by
  sorry

end part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l2240_224022


namespace find_common_difference_l2240_224024

variable {α : Type*} [LinearOrderedField α]

-- Define the properties of the arithmetic sequence
def arithmetic_sum (a1 d : α) (n : ℕ) : α := n * a1 + (n * (n - 1) * d) / 2

variables (a1 d : α) -- First term and common difference of the arithmetic sequence (to be found)
variable (S : ℕ → α) -- Sum of the first n terms of the arithmetic sequence

-- Conditions given in the problem
axiom sum_3_eq_6 : S 3 = 6
axiom term_3_eq_4 : a1 + 2 * d = 4

-- The question translated into a theorem statement that the common difference is 2
theorem find_common_difference : d = 2 :=
by
  sorry

end find_common_difference_l2240_224024


namespace solve_inequality_l2240_224075

theorem solve_inequality (x : ℝ) (h : |2 * x + 6| < 10) : -8 < x ∧ x < 2 :=
sorry

end solve_inequality_l2240_224075


namespace two_digit_num_square_ends_in_self_l2240_224059

theorem two_digit_num_square_ends_in_self {x : ℕ} (hx : 10 ≤ x ∧ x < 100) (hx0 : x % 10 ≠ 0) : 
  (x * x % 100 = x) ↔ (x = 25 ∨ x = 76) :=
sorry

end two_digit_num_square_ends_in_self_l2240_224059


namespace bea_glasses_sold_is_10_l2240_224016

variable (B : ℕ)
variable (earnings_bea earnings_dawn : ℕ)

def bea_price_per_glass := 25
def dawn_price_per_glass := 28
def dawn_glasses_sold := 8
def earnings_diff := 26

def bea_earnings := bea_price_per_glass * B
def dawn_earnings := dawn_price_per_glass * dawn_glasses_sold

def bea_earnings_greater := bea_earnings = dawn_earnings + earnings_diff

theorem bea_glasses_sold_is_10 (h : bea_earnings_greater) : B = 10 :=
by sorry

end bea_glasses_sold_is_10_l2240_224016


namespace total_minutes_to_finish_album_l2240_224046

variable (initial_songs : ℕ) (additional_songs : ℕ) (duration : ℕ)

theorem total_minutes_to_finish_album 
  (h1: initial_songs = 25) 
  (h2: additional_songs = 10) 
  (h3: duration = 3) :
  (initial_songs + additional_songs) * duration = 105 :=
sorry

end total_minutes_to_finish_album_l2240_224046


namespace equation_one_solutions_equation_two_solutions_l2240_224098

theorem equation_one_solutions (x : ℝ) : x^2 + 2 * x - 8 = 0 ↔ x = -4 ∨ x = 2 := 
by {
  sorry
}

theorem equation_two_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 := 
by {
  sorry
}

end equation_one_solutions_equation_two_solutions_l2240_224098


namespace product_of_two_numbers_l2240_224005

theorem product_of_two_numbers
  (x y : ℝ)
  (h_diff : x - y ≠ 0)
  (h1 : x + y = 5 * (x - y))
  (h2 : x * y = 15 * (x - y)) :
  x * y = 37.5 :=
by
  sorry

end product_of_two_numbers_l2240_224005


namespace total_amount_lent_l2240_224002

theorem total_amount_lent (A T : ℝ) (hA : A = 15008) (hInterest : 0.08 * A + 0.10 * (T - A) = 850) : 
  T = 11501.6 :=
by
  sorry

end total_amount_lent_l2240_224002


namespace solve_inequality_l2240_224012

def inequality_solution :=
  {x : ℝ // x < -3 ∨ x > -6/5}

theorem solve_inequality (x : ℝ) : 
  |2*x - 4| - |3*x + 9| < 1 → x < -3 ∨ x > -6/5 :=
by
  sorry

end solve_inequality_l2240_224012


namespace probability_sum_greater_than_five_l2240_224087

theorem probability_sum_greater_than_five (dice_outcomes : List (ℕ × ℕ)) (h: dice_outcomes = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,2), (4,1), (5,1), (2,4)] ++ 
                              [(1,5), (2,6), (3,3), (3,4), (3,5), (3,6), (4,2), (4,3), (4,4), (4,5), (4,6), 
                               (5,2), (5,3), (5,4), (5,5), (5,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)]) :
  p_greater_5 = 2 / 3 := 
by
  sorry

end probability_sum_greater_than_five_l2240_224087


namespace statement_A_statement_B_statement_D_l2240_224095

theorem statement_A (x : ℝ) (hx : x > 1) : 
  ∃(y : ℝ), y = 3 * x + 1 / (x - 1) ∧ y = 2 * Real.sqrt 3 + 3 := 
  sorry

theorem statement_B (x y : ℝ) (hx : x > -1) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  ∃(z : ℝ), z = 1 / (x + 1) + 2 / y ∧ z = 9 / 2 := 
  sorry

theorem statement_D (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃(k : ℝ), k = (x^2 + y^2 + z^2) / (3 * x * y + 4 * y * z) ∧ k = 2 / 5 := 
  sorry

end statement_A_statement_B_statement_D_l2240_224095
