import Mathlib

namespace NUMINAMATH_GPT_smoking_lung_disease_confidence_l1719_171917

/-- Prove that given the conditions, the correct statement is C:
   If it is concluded from the statistic that there is a 95% confidence 
   that smoking is related to lung disease, then there is a 5% chance of
   making a wrong judgment. -/
theorem smoking_lung_disease_confidence 
  (P Q : Prop) 
  (confidence_level : ℝ) 
  (h_conf : confidence_level = 0.95) 
  (h_PQ : P → (Q → true)) :
  ¬Q → (confidence_level = 1 - 0.05) :=
by
  sorry

end NUMINAMATH_GPT_smoking_lung_disease_confidence_l1719_171917


namespace NUMINAMATH_GPT_area_difference_l1719_171974

theorem area_difference (r1 d2 : ℝ) (h1 : r1 = 30) (h2 : d2 = 15) : 
  π * r1^2 - π * (d2 / 2)^2 = 843.75 * π :=
by
  sorry

end NUMINAMATH_GPT_area_difference_l1719_171974


namespace NUMINAMATH_GPT_inequality_satisfied_equality_condition_l1719_171925

theorem inequality_satisfied (x y : ℝ) : x^2 + y^2 + 1 ≥ 2 * (x * y - x + y) :=
sorry

theorem equality_condition (x y : ℝ) : (x^2 + y^2 + 1 = 2 * (x * y - x + y)) ↔ (x = y - 1) :=
sorry

end NUMINAMATH_GPT_inequality_satisfied_equality_condition_l1719_171925


namespace NUMINAMATH_GPT_circle_condition_l1719_171935

theorem circle_condition (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 4 * m * x - 2 * y + 5 * m = 0) ↔ (m < 1 / 4 ∨ m > 1) :=
sorry

end NUMINAMATH_GPT_circle_condition_l1719_171935


namespace NUMINAMATH_GPT_weight_per_linear_foot_l1719_171900

theorem weight_per_linear_foot 
  (length_of_log : ℕ) 
  (cut_length : ℕ) 
  (piece_weight : ℕ) 
  (h1 : length_of_log = 20) 
  (h2 : cut_length = length_of_log / 2) 
  (h3 : piece_weight = 1500) 
  (h4 : length_of_log / 2 = 10) 
  : piece_weight / cut_length = 150 := 
  by 
  sorry

end NUMINAMATH_GPT_weight_per_linear_foot_l1719_171900


namespace NUMINAMATH_GPT_mutually_exclusive_but_not_complementary_l1719_171905

-- Definitions for the problem conditions
inductive Card
| red | black | white | blue

inductive Person
| A | B | C | D

open Card Person

-- The statement of the proof
theorem mutually_exclusive_but_not_complementary : 
  (∃ (f : Person → Card), (f A = red) ∧ (f B ≠ red)) ∧ (∃ (f : Person → Card), (f B = red) ∧ (f A ≠ red)) :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_but_not_complementary_l1719_171905


namespace NUMINAMATH_GPT_evaluate_fraction_l1719_171936

theorem evaluate_fraction : 
  ( (20 - 19) + (18 - 17) + (16 - 15) + (14 - 13) + (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1) ) 
  / 
  ( (1 - 2) + (3 - 4) + (5 - 6) + (7 - 8) + (9 - 10) + (11 - 12) + 13 ) 
  = (10 / 7) := 
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1719_171936


namespace NUMINAMATH_GPT_cricket_run_rate_l1719_171926

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target_runs : ℝ) (first_overs : ℝ) (remaining_overs : ℝ):
  run_rate_first_10_overs = 6.2 → 
  target_runs = 282 →
  first_overs = 10 →
  remaining_overs = 40 →
  (target_runs - run_rate_first_10_overs * first_overs) / remaining_overs = 5.5 :=
by
  intros h1 h2 h3 h4
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l1719_171926


namespace NUMINAMATH_GPT_opposite_of_2023_l1719_171928

theorem opposite_of_2023 : -2023 = -2023 := by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l1719_171928


namespace NUMINAMATH_GPT_sum_of_coordinates_of_A_l1719_171990

open Real

theorem sum_of_coordinates_of_A (A B C : ℝ × ℝ) (h1 : B = (2, 8)) (h2 : C = (5, 2))
  (h3 : ∃ (k : ℝ), A = ((2 * (B.1:ℝ) + C.1) / 3, (2 * (B.2:ℝ) + C.2) / 3) ∧ k = 1/3) :
  A.1 + A.2 = 9 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_A_l1719_171990


namespace NUMINAMATH_GPT_missing_coin_value_l1719_171975

-- Definitions based on the conditions
def value_of_dime := 10 -- Value of 1 dime in cents
def value_of_nickel := 5 -- Value of 1 nickel in cents
def num_dimes := 1
def num_nickels := 2
def total_value_found := 45 -- Total value found in cents

-- Statement to prove the missing coin's value
theorem missing_coin_value : 
  (total_value_found - (num_dimes * value_of_dime + num_nickels * value_of_nickel)) = 25 := 
by
  sorry

end NUMINAMATH_GPT_missing_coin_value_l1719_171975


namespace NUMINAMATH_GPT_outfit_count_l1719_171950

section OutfitProblem

-- Define the number of each type of shirts, pants, and hats
def num_red_shirts : ℕ := 7
def num_blue_shirts : ℕ := 5
def num_green_shirts : ℕ := 8

def num_pants : ℕ := 10

def num_green_hats : ℕ := 10
def num_red_hats : ℕ := 6
def num_blue_hats : ℕ := 7

-- The main theorem to prove the number of outfits where shirt and hat are not the same color
theorem outfit_count : 
  (num_red_shirts * num_pants * (num_green_hats + num_blue_hats) +
  num_blue_shirts * num_pants * (num_green_hats + num_red_hats) +
  num_green_shirts * num_pants * (num_red_hats + num_blue_hats)) = 3030 :=
  sorry

end OutfitProblem

end NUMINAMATH_GPT_outfit_count_l1719_171950


namespace NUMINAMATH_GPT_cookies_on_third_plate_l1719_171971

theorem cookies_on_third_plate :
  ∀ (a5 a7 a14 a19 a25 : ℕ),
  (a5 = 5) ∧ (a7 = 7) ∧ (a14 = 14) ∧ (a19 = 19) ∧ (a25 = 25) →
  ∃ (a12 : ℕ), a12 = 12 :=
by
  sorry

end NUMINAMATH_GPT_cookies_on_third_plate_l1719_171971


namespace NUMINAMATH_GPT_total_mile_times_l1719_171984

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end NUMINAMATH_GPT_total_mile_times_l1719_171984


namespace NUMINAMATH_GPT_problem_remainder_P2017_mod_1000_l1719_171907

def P (x : ℤ) : ℤ := x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem problem_remainder_P2017_mod_1000 :
  (P 2017) % 1000 = 167 :=
by
  -- this proof examines \( P(2017) \) modulo 1000
  sorry

end NUMINAMATH_GPT_problem_remainder_P2017_mod_1000_l1719_171907


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1719_171957

theorem geometric_sequence_common_ratio
  (S : ℕ → ℝ)
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : S 1 = a 1)
  (h2 : S 2 = a 1 + a 1 * q)
  (h3 : a 2 = a 1 * q)
  (h4 : a 3 = a 1 * q^2)
  (h5 : 3 * S 2 = a 3 - 2)
  (h6 : 3 * S 1 = a 2 - 2) :
  q = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1719_171957


namespace NUMINAMATH_GPT_solution_set_f_pos_l1719_171927

open Set Function

variables (f : ℝ → ℝ)
variables (h_even : ∀ x : ℝ, f (-x) = f x)
variables (h_diff : ∀ x ≠ 0, DifferentiableAt ℝ f x)
variables (h_pos : ∀ x : ℝ, x > 0 → f x + x * (f' x) > 0)
variables (h_at_2 : f 2 = 0)

theorem solution_set_f_pos :
  {x : ℝ | f x > 0} = (Iio (-2)) ∪ (Ioi 2) :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_f_pos_l1719_171927


namespace NUMINAMATH_GPT_fraction_sum_l1719_171915

theorem fraction_sum : (1 / 3 : ℚ) + (2 / 7) + (3 / 8) = 167 / 168 := by
  sorry

end NUMINAMATH_GPT_fraction_sum_l1719_171915


namespace NUMINAMATH_GPT_find_radius_of_base_of_cone_l1719_171959

noncomputable def radius_of_cone (CSA : ℝ) (l : ℝ) : ℝ :=
  CSA / (Real.pi * l)

theorem find_radius_of_base_of_cone :
  radius_of_cone 527.7875658030853 14 = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_radius_of_base_of_cone_l1719_171959


namespace NUMINAMATH_GPT_remaining_money_after_purchases_l1719_171934

def initial_amount : ℝ := 100
def bread_cost : ℝ := 4
def candy_cost : ℝ := 3
def cereal_cost : ℝ := 6
def fruit_percentage : ℝ := 0.2
def milk_cost_each : ℝ := 4.50
def turkey_fraction : ℝ := 0.25

-- Calculate total spent on initial purchases
def initial_spent : ℝ := bread_cost + (2 * candy_cost) + cereal_cost

-- Remaining amount after initial purchases
def remaining_after_initial : ℝ := initial_amount - initial_spent

-- Spend 20% on fruits
def spent_on_fruits : ℝ := fruit_percentage * remaining_after_initial
def remaining_after_fruits : ℝ := remaining_after_initial - spent_on_fruits

-- Spend on two gallons of milk
def spent_on_milk : ℝ := 2 * milk_cost_each
def remaining_after_milk : ℝ := remaining_after_fruits - spent_on_milk

-- Spend 1/4 on turkey
def spent_on_turkey : ℝ := turkey_fraction * remaining_after_milk
def final_remaining : ℝ := remaining_after_milk - spent_on_turkey

theorem remaining_money_after_purchases : final_remaining = 43.65 := by
  sorry

end NUMINAMATH_GPT_remaining_money_after_purchases_l1719_171934


namespace NUMINAMATH_GPT_negate_proposition_l1719_171998

theorem negate_proposition :
    (¬ ∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1719_171998


namespace NUMINAMATH_GPT_minute_hand_angle_backward_l1719_171903

theorem minute_hand_angle_backward (backward_minutes : ℝ) (h : backward_minutes = 10) :
  (backward_minutes / 60) * (2 * Real.pi) = Real.pi / 3 := by
  sorry

end NUMINAMATH_GPT_minute_hand_angle_backward_l1719_171903


namespace NUMINAMATH_GPT_max_value_l1719_171920

open Real

/-- Given vectors a, b, and c, and real numbers m and n such that m * a + n * b = c,
prove that the maximum value for (m - 3)^2 + n^2 is 16. --/
theorem max_value
  (α : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ)
  (m n : ℝ)
  (ha : a = (1, 1))
  (hb : b = (1, -1))
  (hc : c = (sqrt 2 * cos α, sqrt 2 * sin α))
  (h : m * a.1 + n * b.1 = c.1 ∧ m * a.2 + n * b.2 = c.2) :
  (m - 3)^2 + n^2 ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_max_value_l1719_171920


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1719_171931

theorem rectangle_diagonal_length (p : ℝ) (r_lw : ℝ) (l w d : ℝ) 
    (h_p : p = 84) 
    (h_ratio : r_lw = 5 / 2) 
    (h_l : l = 5 * (p / 2) / 7) 
    (h_w : w = 2 * (p / 2) / 7) 
    (h_d : d = Real.sqrt (l ^ 2 + w ^ 2)) :
  d = 2 * Real.sqrt 261 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1719_171931


namespace NUMINAMATH_GPT_possible_values_of_t_l1719_171943

theorem possible_values_of_t
  (theta : ℝ) 
  (x y t : ℝ) :
  x = Real.cos theta →
  y = Real.sin theta →
  t = (Real.sin theta) ^ 2 + (Real.cos theta) ^ 2 →
  x^2 + y^2 = 1 →
  t = 1 := by
  sorry

end NUMINAMATH_GPT_possible_values_of_t_l1719_171943


namespace NUMINAMATH_GPT_product_of_two_numbers_l1719_171973

theorem product_of_two_numbers (x y : ℚ) 
  (h1 : x + y = 8 * (x - y)) 
  (h2 : x * y = 15 * (x - y)) : 
  x * y = 100 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1719_171973


namespace NUMINAMATH_GPT_cost_effectiveness_order_l1719_171958

variables {cS cM cL qS qM qL : ℝ}
variables (h1 : cM = 2 * cS)
variables (h2 : qM = 0.7 * qL)
variables (h3 : qL = 3 * qS)
variables (h4 : cL = 1.2 * cM)

theorem cost_effectiveness_order :
  (cL / qL <= cM / qM) ∧ (cM / qM <= cS / qS) :=
by
  sorry

end NUMINAMATH_GPT_cost_effectiveness_order_l1719_171958


namespace NUMINAMATH_GPT_simplify_expression_l1719_171904

theorem simplify_expression (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1719_171904


namespace NUMINAMATH_GPT_average_age_decrease_l1719_171960

theorem average_age_decrease (N : ℕ) (T : ℝ) 
  (h1 : T = 40 * N) 
  (h2 : ∀ new_average_age : ℝ, (T + 12 * 34) / (N + 12) = new_average_age → new_average_age = 34) :
  ∃ decrease : ℝ, decrease = 6 :=
by
  sorry

end NUMINAMATH_GPT_average_age_decrease_l1719_171960


namespace NUMINAMATH_GPT_necklace_length_l1719_171965

-- Given conditions as definitions in Lean
def num_pieces : ℕ := 16
def piece_length : ℝ := 10.4
def overlap_length : ℝ := 3.5
def effective_length : ℝ := piece_length - overlap_length
def total_length : ℝ := effective_length * num_pieces

-- The theorem to prove
theorem necklace_length :
  total_length = 110.4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_necklace_length_l1719_171965


namespace NUMINAMATH_GPT_find_t_plus_a3_l1719_171952

noncomputable def geometric_sequence_sum (n : ℕ) (t : ℤ) : ℤ :=
  3 ^ n + t

noncomputable def a_1 (t : ℤ) : ℤ :=
  geometric_sequence_sum 1 t

noncomputable def a_2 (t : ℤ) : ℤ :=
  geometric_sequence_sum 2 t - geometric_sequence_sum 1 t

noncomputable def a_3 (t : ℤ) : ℤ :=
  geometric_sequence_sum 3 t - geometric_sequence_sum 2 t

theorem find_t_plus_a3 (t : ℤ) : t + a_3 t = 17 :=
sorry

end NUMINAMATH_GPT_find_t_plus_a3_l1719_171952


namespace NUMINAMATH_GPT_abs_diff_eq_seven_l1719_171968

theorem abs_diff_eq_seven (m n : ℤ) (h1 : |m| = 5) (h2 : |n| = 2) (h3 : m * n < 0) : |m - n| = 7 := 
sorry

end NUMINAMATH_GPT_abs_diff_eq_seven_l1719_171968


namespace NUMINAMATH_GPT_initial_seashells_l1719_171910

-- Definitions based on the problem conditions
def gave_to_joan : ℕ := 6
def left_with_jessica : ℕ := 2

-- Theorem statement to prove the number of seashells initially found by Jessica
theorem initial_seashells : gave_to_joan + left_with_jessica = 8 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_initial_seashells_l1719_171910


namespace NUMINAMATH_GPT_harry_to_sue_nuts_ratio_l1719_171923

-- Definitions based on conditions
def sue_nuts : ℕ := 48
def bill_nuts (harry_nuts : ℕ) : ℕ := 6 * harry_nuts
def total_nuts (harry_nuts : ℕ) : ℕ := bill_nuts harry_nuts + harry_nuts

-- Proving the ratio
theorem harry_to_sue_nuts_ratio (H : ℕ) (h1 : sue_nuts = 48) (h2 : bill_nuts H + H = 672) : H / sue_nuts = 2 :=
by
  sorry

end NUMINAMATH_GPT_harry_to_sue_nuts_ratio_l1719_171923


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l1719_171980

-- Define the number of axles given the conditions
def num_axles (total_wheels rear_axle_wheels front_axle_wheels : ℕ) : ℕ :=
  let rear_axles := (total_wheels - front_axle_wheels) / rear_axle_wheels
  rear_axles + 1

-- Define the toll calculation given the number of axles
def toll (axles : ℕ) : ℝ :=
  1.50 + 0.50 * (axles - 2)

-- Constants specific to the problem
def total_wheels : ℕ := 18
def rear_axle_wheels : ℕ := 4
def front_axle_wheels : ℕ := 2

-- Calculate the number of axles for the given truck
def truck_axles : ℕ := num_axles total_wheels rear_axle_wheels front_axle_wheels

-- The actual statement to prove
theorem toll_for_18_wheel_truck : toll truck_axles = 3.00 :=
  by
    -- proof will go here
    sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l1719_171980


namespace NUMINAMATH_GPT_solution_set_l1719_171999

noncomputable def f : ℝ → ℝ := sorry

-- The function f is defined to be odd.
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- The function f is increasing on (-∞, 0).
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y

-- Given f(2) = 0
axiom f_at_2 : f 2 = 0

-- Prove the solution set for x f(x + 1) < 0
theorem solution_set : { x : ℝ | x * f (x + 1) < 0 } = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1719_171999


namespace NUMINAMATH_GPT_solve_CD_l1719_171986

noncomputable def find_CD : Prop :=
  ∃ C D : ℝ, (C = 11 ∧ D = 0) ∧ (∀ x : ℝ, x ≠ -4 ∧ x ≠ 12 → 
    (7 * x - 3) / ((x + 4) * (x - 12)) = C / (x + 4) + D / (x - 12))

theorem solve_CD : find_CD :=
sorry

end NUMINAMATH_GPT_solve_CD_l1719_171986


namespace NUMINAMATH_GPT_xyz_abs_eq_one_l1719_171916

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end NUMINAMATH_GPT_xyz_abs_eq_one_l1719_171916


namespace NUMINAMATH_GPT_broadway_show_total_amount_collected_l1719_171909

theorem broadway_show_total_amount_collected (num_adults num_children : ℕ) 
  (adult_ticket_price child_ticket_ratio : ℕ) 
  (child_ticket_price : ℕ) 
  (h1 : num_adults = 400) 
  (h2 : num_children = 200) 
  (h3 : adult_ticket_price = 32) 
  (h4 : child_ticket_ratio = 2) 
  (h5 : adult_ticket_price = child_ticket_ratio * child_ticket_price) : 
  num_adults * adult_ticket_price + num_children * child_ticket_price = 16000 := 
  by 
    sorry

end NUMINAMATH_GPT_broadway_show_total_amount_collected_l1719_171909


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1719_171967

/-- 
Prove that the perimeter of an isosceles triangle with sides 6 cm and 8 cm, 
and an area of 12 cm², is 20 cm.
--/
theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (S : ℝ) (h3 : S = 12) :
  a ≠ b →
  a = c ∨ b = c →
  ∃ P : ℝ, P = 20 := sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1719_171967


namespace NUMINAMATH_GPT_time_to_travel_A_to_C_is_6_l1719_171996

-- Assume the existence of a real number t representing the time taken
-- Assume constant speed r for the river current and p for the power boat relative to the river.
variables (t r p : ℝ)

-- Conditions
axiom condition1 : p > 0
axiom condition2 : r > 0
axiom condition3 : t * (1.5 * (p + r)) + (p - r) * (12 - t) = 12 * r

-- Define the time taken for the power boat to travel from A to C
def time_from_A_to_C : ℝ := t

-- The proof problem: Prove time_from_A_to_C = 6 under the given conditions
theorem time_to_travel_A_to_C_is_6 : time_from_A_to_C = 6 := by
  sorry

end NUMINAMATH_GPT_time_to_travel_A_to_C_is_6_l1719_171996


namespace NUMINAMATH_GPT_smallest_sum_is_S5_l1719_171940

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Definitions of arithmetic sequence sum
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
axiom h1 : a 3 + a 8 > 0
axiom h2 : S 9 < 0

-- Statements relating terms and sums in arithmetic sequence
axiom h3 : ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem smallest_sum_is_S5 (seq_a : arithmetic_sequence a) : S 5 ≤ S 1 ∧ S 5 ≤ S 2 ∧ S 5 ≤ S 3 ∧ S 5 ≤ S 4 ∧ S 5 ≤ S 6 ∧ S 5 ≤ S 7 ∧ S 5 ≤ S 8 ∧ S 5 ≤ S 9 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_sum_is_S5_l1719_171940


namespace NUMINAMATH_GPT_cond_prob_B_given_A_l1719_171946

-- Definitions based on the conditions
def eventA := {n : ℕ | n > 4 ∧ n ≤ 6}
def eventB := {k : ℕ × ℕ | (k.1 + k.2) = 7}

-- Probability of event A
def probA := (2 : ℚ) / 6

-- Joint probability of events A and B
def probAB := (1 : ℚ) / (6 * 6)

-- Conditional probability P(B|A)
def cond_prob := probAB / probA

-- The final statement to prove
theorem cond_prob_B_given_A : cond_prob = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_cond_prob_B_given_A_l1719_171946


namespace NUMINAMATH_GPT_binary_111_to_decimal_l1719_171981

-- Define a function to convert binary list to decimal
def binaryToDecimal (bin : List ℕ) : ℕ :=
  bin.reverse.enumFrom 0 |>.foldl (λ acc ⟨i, b⟩ => acc + b * (2 ^ i)) 0

-- Assert the equivalence between the binary number [1, 1, 1] and its decimal representation 7
theorem binary_111_to_decimal : binaryToDecimal [1, 1, 1] = 7 :=
  by
  sorry

end NUMINAMATH_GPT_binary_111_to_decimal_l1719_171981


namespace NUMINAMATH_GPT_part1_part2_l1719_171961

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem part1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1719_171961


namespace NUMINAMATH_GPT_generalized_inequality_l1719_171972

theorem generalized_inequality (n k : ℕ) (h1 : 3 ≤ n) (h2 : 1 ≤ k ∧ k ≤ n) : 
  2^n + 5^n > 2^(n - k) * 5^k + 2^k * 5^(n - k) := 
by 
  sorry

end NUMINAMATH_GPT_generalized_inequality_l1719_171972


namespace NUMINAMATH_GPT_distinct_bead_arrangements_l1719_171983

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n-1)

theorem distinct_bead_arrangements : factorial 8 / (8 * 2) = 2520 := 
  by sorry

end NUMINAMATH_GPT_distinct_bead_arrangements_l1719_171983


namespace NUMINAMATH_GPT_unique_solution_values_l1719_171985

theorem unique_solution_values (a : ℝ) :
  (∃! x : ℝ, a * x^2 - x + 1 = 0) ↔ (a = 0 ∨ a = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_values_l1719_171985


namespace NUMINAMATH_GPT_trigonometric_identity_l1719_171924

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1719_171924


namespace NUMINAMATH_GPT_simplify_sum_of_square_roots_l1719_171987

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_sum_of_square_roots_l1719_171987


namespace NUMINAMATH_GPT_range_of_a_l1719_171914

theorem range_of_a (a : ℝ) :
  ¬ (∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1719_171914


namespace NUMINAMATH_GPT_customer_pays_correct_amount_l1719_171976

def wholesale_price : ℝ := 4
def markup : ℝ := 0.25
def discount : ℝ := 0.05

def retail_price : ℝ := wholesale_price * (1 + markup)
def discount_amount : ℝ := retail_price * discount
def customer_price : ℝ := retail_price - discount_amount

theorem customer_pays_correct_amount : customer_price = 4.75 := by
  -- proof steps would go here, but we are skipping them as instructed
  sorry

end NUMINAMATH_GPT_customer_pays_correct_amount_l1719_171976


namespace NUMINAMATH_GPT_probability_of_sum_23_l1719_171989

def is_valid_time (h m : ℕ) : Prop :=
  0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def sum_of_time_digits (h m : ℕ) : ℕ :=
  sum_of_digits h + sum_of_digits m

theorem probability_of_sum_23 :
  (∃ h m, is_valid_time h m ∧ sum_of_time_digits h m = 23) →
  (4 / 1440 : ℚ) = (1 / 360 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_sum_23_l1719_171989


namespace NUMINAMATH_GPT_real_solutions_l1719_171962

-- Given the condition (equation)
def quadratic_equation (x y : ℝ) : Prop :=
  x^2 + 2 * x * Real.sin (x * y) + 1 = 0

-- The main theorem statement proving the solutions for x and y
theorem real_solutions (x y : ℝ) (k : ℤ) :
  quadratic_equation x y ↔
  (x = 1 ∧ (y = (Real.pi / 2 + 2 * k * Real.pi) ∨ y = (-Real.pi / 2 + 2 * k * Real.pi))) ∨
  (x = -1 ∧ (y = (-Real.pi / 2 + 2 * k * Real.pi) ∨ y = (Real.pi / 2 + 2 * k * Real.pi))) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_l1719_171962


namespace NUMINAMATH_GPT_solve_inequality_l1719_171978

theorem solve_inequality (x : ℝ) : 3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1719_171978


namespace NUMINAMATH_GPT_mrs_hilt_apples_l1719_171921

theorem mrs_hilt_apples (hours : ℕ := 3) (rate : ℕ := 5) : 
  (rate * hours) = 15 := 
by sorry

end NUMINAMATH_GPT_mrs_hilt_apples_l1719_171921


namespace NUMINAMATH_GPT_triangle_PQR_min_perimeter_l1719_171995

theorem triangle_PQR_min_perimeter (PQ PR QR : ℕ) (QJ : ℕ) 
  (hPQ_PR : PQ = PR) (hQJ_10 : QJ = 10) (h_pos_QR : 0 < QR) :
  QR * 2 + PQ * 2 = 96 :=
  sorry

end NUMINAMATH_GPT_triangle_PQR_min_perimeter_l1719_171995


namespace NUMINAMATH_GPT_max_value_in_interval_l1719_171988

theorem max_value_in_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → x^4 - 2 * x^2 + 5 ≤ 13 :=
by
  sorry

end NUMINAMATH_GPT_max_value_in_interval_l1719_171988


namespace NUMINAMATH_GPT_cubic_roots_a_b_third_root_l1719_171949

theorem cubic_roots_a_b_third_root (a b : ℝ) :
  (∀ x, x^3 + a * x^2 + b * x + 6 = 0 → (x = 2 ∨ x = 3 ∨ x = -1)) →
  a = -4 ∧ b = 1 :=
by
  intro h
  -- We're skipping the proof steps and focusing on definite the goal
  sorry

end NUMINAMATH_GPT_cubic_roots_a_b_third_root_l1719_171949


namespace NUMINAMATH_GPT_point_outside_circle_l1719_171991

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a*x + b*y = 1 ∧ x^2 + y^2 = 1)) : a^2 + b^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_point_outside_circle_l1719_171991


namespace NUMINAMATH_GPT_probability_of_double_tile_is_one_fourth_l1719_171992

noncomputable def probability_double_tile : ℚ :=
  let total_pairs := (7 * 7) / 2
  let double_pairs := 7
  double_pairs / total_pairs

theorem probability_of_double_tile_is_one_fourth :
  probability_double_tile = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_double_tile_is_one_fourth_l1719_171992


namespace NUMINAMATH_GPT_total_road_length_l1719_171913

theorem total_road_length (L : ℚ) : (1/3) * L + (2/5) * (2/3) * L = 135 → L = 225 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_total_road_length_l1719_171913


namespace NUMINAMATH_GPT_intersection_A_B_l1719_171982

def A : Set ℝ := { y | ∃ x : ℝ, y = |x| }
def B : Set ℝ := { y | ∃ x : ℝ, y = 1 - 2*x - x^2 }

theorem intersection_A_B :
  A ∩ B = { y | 0 ≤ y ∧ y ≤ 2 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1719_171982


namespace NUMINAMATH_GPT_unit_digit_15_pow_100_l1719_171977

theorem unit_digit_15_pow_100 : ((15^100) % 10) = 5 := 
by sorry

end NUMINAMATH_GPT_unit_digit_15_pow_100_l1719_171977


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1719_171963

theorem sufficient_not_necessary (a b : ℝ) :
  (a = -1 ∧ b = 2 → a * b = -2) ∧ (a * b = -2 → ¬(a = -1 ∧ b = 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1719_171963


namespace NUMINAMATH_GPT_number_of_diagonals_in_hexagon_l1719_171942

-- Define the number of sides of the hexagon
def sides_of_hexagon : ℕ := 6

-- Define the formula for the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The theorem we want to prove
theorem number_of_diagonals_in_hexagon : number_of_diagonals sides_of_hexagon = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_in_hexagon_l1719_171942


namespace NUMINAMATH_GPT_competition_result_l1719_171906

variables (Olya Oleg Pasha : ℕ)

theorem competition_result 
  (h1 : Olya ≠ 1 → Olya ≠ 3 → False)
  (h2 : (Oleg = 1 ∨ Oleg = 3) → Olya = 3)
  (h3 : (Oleg ≠ 1 → (Olya = 2 ∨ Olya = 3)))
  (h4 : Olya ≠ 1 ∧ Oleg ≠ 2 ∧ Pasha ≠ 3) :
  Oleg = 1 ∧ Pasha = 2 ∧ Olya = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_competition_result_l1719_171906


namespace NUMINAMATH_GPT_x_cubed_plus_square_plus_lin_plus_a_l1719_171933

theorem x_cubed_plus_square_plus_lin_plus_a (a b x : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b :=
by {
  sorry
}

end NUMINAMATH_GPT_x_cubed_plus_square_plus_lin_plus_a_l1719_171933


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1719_171970

theorem max_value_of_quadratic : 
  ∃ x : ℝ, (∃ M : ℝ, ∀ y : ℝ, (-3 * y^2 + 15 * y + 9 <= M)) ∧ M = 111 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1719_171970


namespace NUMINAMATH_GPT_total_distance_is_correct_l1719_171941

noncomputable def boat_speed : ℝ := 20 -- boat speed in still water (km/hr)
noncomputable def current_speed_first : ℝ := 5 -- current speed for the first 6 minutes (km/hr)
noncomputable def current_speed_second : ℝ := 8 -- current speed for the next 6 minutes (km/hr)
noncomputable def current_speed_third : ℝ := 3 -- current speed for the last 6 minutes (km/hr)
noncomputable def time_in_hours : ℝ := 6 / 60 -- 6 minutes in hours (0.1 hours)

noncomputable def total_distance_downstream := 
  (boat_speed + current_speed_first) * time_in_hours +
  (boat_speed + current_speed_second) * time_in_hours +
  (boat_speed + current_speed_third) * time_in_hours

theorem total_distance_is_correct : total_distance_downstream = 7.6 :=
  by 
  sorry

end NUMINAMATH_GPT_total_distance_is_correct_l1719_171941


namespace NUMINAMATH_GPT_smallest_x_l1719_171902

theorem smallest_x (x : ℕ) (M : ℕ) (h : 1800 * x = M^3) :
  x = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l1719_171902


namespace NUMINAMATH_GPT_min_transport_cost_l1719_171937

-- Definitions based on conditions
def total_washing_machines : ℕ := 100
def typeA_max_count : ℕ := 4
def typeB_max_count : ℕ := 8
def typeA_cost : ℕ := 400
def typeA_capacity : ℕ := 20
def typeB_cost : ℕ := 300
def typeB_capacity : ℕ := 10

-- Minimum transportation cost calculation
def min_transportation_cost : ℕ :=
  let typeA_trucks_used := min typeA_max_count (total_washing_machines / typeA_capacity)
  let remaining_washing_machines := total_washing_machines - typeA_trucks_used * typeA_capacity
  let typeB_trucks_used := min typeB_max_count (remaining_washing_machines / typeB_capacity)
  typeA_trucks_used * typeA_cost + typeB_trucks_used * typeB_cost

-- Lean 4 statement to prove the minimum transportation cost
theorem min_transport_cost : min_transportation_cost = 2200 := by
  sorry

end NUMINAMATH_GPT_min_transport_cost_l1719_171937


namespace NUMINAMATH_GPT_sector_max_area_l1719_171912

-- Define the problem conditions
variables (α : ℝ) (R : ℝ)
variables (h_perimeter : 2 * R + R * α = 40)
variables (h_positive_radius : 0 < R)

-- State the theorem
theorem sector_max_area (h_alpha : α = 2) : 
  1/2 * α * (40 - 2 * R) * R = 100 := 
sorry

end NUMINAMATH_GPT_sector_max_area_l1719_171912


namespace NUMINAMATH_GPT_shaded_area_l1719_171948

theorem shaded_area (PR PV PQ QR : ℝ) (hPR : PR = 20) (hPV : PV = 12) (hPQ_QR : PQ + QR = PR) :
  PR * PV - 1 / 2 * 12 * PR = 120 :=
by
  -- Definitions used earlier
  have h_area_rectangle : PR * PV = 240 := by
    rw [hPR, hPV]
    norm_num
  have h_half_total_unshaded : (1 / 2) * 12 * PR = 120 := by
    rw [hPR]
    norm_num
  rw [h_area_rectangle, h_half_total_unshaded]
  norm_num

end NUMINAMATH_GPT_shaded_area_l1719_171948


namespace NUMINAMATH_GPT_relationship_a_b_c_l1719_171922

theorem relationship_a_b_c (x y a b c : ℝ) (h1 : x + y = a)
  (h2 : x^2 + y^2 = b) (h3 : x^3 + y^3 = c) : a^3 - 3*a*b + 2*c = 0 := by
  sorry

end NUMINAMATH_GPT_relationship_a_b_c_l1719_171922


namespace NUMINAMATH_GPT_pet_store_satisfaction_l1719_171944

theorem pet_store_satisfaction :
  let puppies := 15
  let kittens := 6
  let hamsters := 8
  let friends := 3
  puppies * kittens * hamsters * friends.factorial = 4320 := by
  sorry

end NUMINAMATH_GPT_pet_store_satisfaction_l1719_171944


namespace NUMINAMATH_GPT_find_common_difference_l1719_171901

theorem find_common_difference (AB BC AC : ℕ) (x y z d : ℕ) 
  (h1 : AB = 300) (h2 : BC = 350) (h3 : AC = 400) 
  (hx : x = (2 * d) / 5) (hy : y = (7 * d) / 15) (hz : z = (8 * d) / 15) 
  (h_sum : x + y + z = 750) : 
  d = 536 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_common_difference_l1719_171901


namespace NUMINAMATH_GPT_compute_operation_l1719_171908

def operation_and (x : ℝ) := 10 - x
def operation_and_prefix (x : ℝ) := x - 10

theorem compute_operation (x : ℝ) : operation_and_prefix (operation_and 15) = -15 :=
by
  sorry

end NUMINAMATH_GPT_compute_operation_l1719_171908


namespace NUMINAMATH_GPT_intersection_M_N_l1719_171938

-- Define the sets M and N according to the conditions given in the problem
def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

-- State the theorem to prove the intersection of M and N
theorem intersection_M_N : (M ∩ N) = {0, 1} := 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1719_171938


namespace NUMINAMATH_GPT_quadratic_solution_unique_l1719_171956

noncomputable def solve_quad_eq (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) : ℝ :=
-2 / 3

theorem quadratic_solution_unique (a : ℝ) (h : a ≠ 0) (h_uniq : (36^2 - 4 * a * 12) = 0) :
  (∃! x : ℝ, a * x^2 + 36 * x + 12 = 0) ∧ (solve_quad_eq a h h_uniq) = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_unique_l1719_171956


namespace NUMINAMATH_GPT_min_max_F_l1719_171918

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

def F (x y : ℝ) : ℝ := x^2 + y^2

theorem min_max_F (x y : ℝ) (h1 : f (y^2 - 6 * y + 11) + f (x^2 - 8 * x + 10) ≤ 0) (h2 : y ≥ 3) :
  ∃ (min_val max_val : ℝ), min_val = 13 ∧ max_val = 49 ∧
    min_val ≤ F x y ∧ F x y ≤ max_val :=
sorry

end NUMINAMATH_GPT_min_max_F_l1719_171918


namespace NUMINAMATH_GPT_camera_filter_kit_savings_l1719_171930

variable (kit_price : ℝ) (single_prices : List ℝ)
variable (correct_saving_amount : ℝ)

theorem camera_filter_kit_savings
    (h1 : kit_price = 145.75)
    (h2 : single_prices = [3 * 9.50, 2 * 15.30, 1 * 20.75, 2 * 25.80])
    (h3 : correct_saving_amount = -14.30) :
    (single_prices.sum - kit_price = correct_saving_amount) :=
by
  sorry

end NUMINAMATH_GPT_camera_filter_kit_savings_l1719_171930


namespace NUMINAMATH_GPT_train_length_proof_l1719_171919

-- Define the conditions
def train_speed_kmph := 72
def platform_length_m := 290
def crossing_time_s := 26

-- Conversion factor
def kmph_to_mps := 5 / 18

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * kmph_to_mps

-- Distance covered by train while crossing the platform (in meters)
def distance_covered := train_speed_mps * crossing_time_s

-- Length of the train (in meters)
def train_length := distance_covered - platform_length_m

-- The theorem to be proved
theorem train_length_proof : train_length = 230 :=
by 
  -- proof would be placed here 
  sorry

end NUMINAMATH_GPT_train_length_proof_l1719_171919


namespace NUMINAMATH_GPT_negation_of_universal_statement_l1719_171939

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔ ∃ a : ℝ, ∀ x : ℝ, ¬(x > 0 ∧ a * x^2 - 3 * x - a = 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_statement_l1719_171939


namespace NUMINAMATH_GPT_pears_left_l1719_171997

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) (total_pears : ℕ) (pears_left : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) 
  (h4 : total_pears = jason_pears + keith_pears) 
  (h5 : pears_left = total_pears - mike_ate) 
  : pears_left = 81 :=
by
  sorry

end NUMINAMATH_GPT_pears_left_l1719_171997


namespace NUMINAMATH_GPT_ab_condition_l1719_171911

theorem ab_condition (a b : ℝ) : ¬((a + b > 1 → a^2 + b^2 > 1) ∧ (a^2 + b^2 > 1 → a + b > 1)) :=
by {
  -- This proof problem states that the condition "a + b > 1" is neither sufficient nor necessary for "a^2 + b^2 > 1".
  sorry
}

end NUMINAMATH_GPT_ab_condition_l1719_171911


namespace NUMINAMATH_GPT_bicyclist_speed_first_100_km_l1719_171953

theorem bicyclist_speed_first_100_km (v : ℝ) :
  (16 = 400 / ((100 / v) + 20)) →
  v = 20 :=
by
  sorry

end NUMINAMATH_GPT_bicyclist_speed_first_100_km_l1719_171953


namespace NUMINAMATH_GPT_swap_equality_l1719_171994

theorem swap_equality {a1 b1 a2 b2 : ℝ} 
  (h1 : a1^2 + b1^2 = 1)
  (h2 : a2^2 + b2^2 = 1)
  (h3 : a1 * a2 + b1 * b2 = 0) :
  b1 = a2 ∨ b1 = -a2 :=
by sorry

end NUMINAMATH_GPT_swap_equality_l1719_171994


namespace NUMINAMATH_GPT_samantha_interest_l1719_171979

-- Definitions based on problem conditions
def P : ℝ := 2000
def r : ℝ := 0.08
def n : ℕ := 5

-- Compound interest calculation
noncomputable def A : ℝ := P * (1 + r) ^ n
noncomputable def Interest : ℝ := A - P

-- Theorem statement with Lean 4
theorem samantha_interest : Interest = 938.656 := 
by 
  sorry

end NUMINAMATH_GPT_samantha_interest_l1719_171979


namespace NUMINAMATH_GPT_solve_for_a_l1719_171966

theorem solve_for_a (a : ℚ) (h : a + a / 3 = 8 / 3) : a = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1719_171966


namespace NUMINAMATH_GPT_test_completion_days_l1719_171951

theorem test_completion_days :
  let barbara_days := 10
  let edward_days := 9
  let abhinav_days := 11
  let alex_days := 12
  let barbara_rate := 1 / barbara_days
  let edward_rate := 1 / edward_days
  let abhinav_rate := 1 / abhinav_days
  let alex_rate := 1 / alex_days
  let one_cycle_work := barbara_rate + edward_rate + abhinav_rate + alex_rate
  let cycles_needed := (1 : ℚ) / one_cycle_work
  Nat.ceil cycles_needed = 3 :=
by
  sorry

end NUMINAMATH_GPT_test_completion_days_l1719_171951


namespace NUMINAMATH_GPT_poodle_terrier_bark_ratio_l1719_171955

theorem poodle_terrier_bark_ratio :
  ∀ (P T : ℕ),
  (T = 12) →
  (P = 24) →
  (P / T = 2) :=
by intros P T hT hP
   sorry

end NUMINAMATH_GPT_poodle_terrier_bark_ratio_l1719_171955


namespace NUMINAMATH_GPT_number_of_pairings_l1719_171964

-- Definitions for conditions.
def bowls : Finset String := {"red", "blue", "yellow", "green"}
def glasses : Finset String := {"red", "blue", "yellow", "green"}

-- The theorem statement
theorem number_of_pairings : bowls.card * glasses.card = 16 := by
  sorry

end NUMINAMATH_GPT_number_of_pairings_l1719_171964


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1719_171932

theorem eccentricity_of_hyperbola
  (a b c e : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c = a * e)
  (h4 : c^2 = a^2 + b^2)
  (h5 : ∀ B : ℝ × ℝ, B = (0, b))
  (h6 : ∀ F : ℝ × ℝ, F = (c, 0))
  (h7 : ∀ m_FB m_asymptote : ℝ, m_FB * m_asymptote = -1 → (m_FB = -b / c) ∧ (m_asymptote = b / a)) :
  e = (1 + Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1719_171932


namespace NUMINAMATH_GPT_cube_volume_and_diagonal_from_surface_area_l1719_171947

theorem cube_volume_and_diagonal_from_surface_area
    (A : ℝ) (h : A = 150) :
    ∃ (V : ℝ) (d : ℝ), V = 125 ∧ d = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_and_diagonal_from_surface_area_l1719_171947


namespace NUMINAMATH_GPT_exponents_product_as_cube_l1719_171929

theorem exponents_product_as_cube :
  (3^12 * 3^3) = 243^3 :=
sorry

end NUMINAMATH_GPT_exponents_product_as_cube_l1719_171929


namespace NUMINAMATH_GPT_chemistry_textbook_weight_l1719_171954

theorem chemistry_textbook_weight (G C : ℝ) 
  (h1 : G = 0.625) 
  (h2 : C = G + 6.5) : 
  C = 7.125 := 
by 
  sorry

end NUMINAMATH_GPT_chemistry_textbook_weight_l1719_171954


namespace NUMINAMATH_GPT_correct_calculation_l1719_171993

-- Definitions of the conditions
def condition_A (a : ℝ) : Prop := a^2 + a^2 = a^4
def condition_B (a : ℝ) : Prop := 3 * a^2 + 2 * a^2 = 5 * a^2
def condition_C (a : ℝ) : Prop := a^4 - a^2 = a^2
def condition_D (a : ℝ) : Prop := 3 * a^2 - 2 * a^2 = 1

-- The theorem statement
theorem correct_calculation (a : ℝ) : condition_B a := by 
sorry

end NUMINAMATH_GPT_correct_calculation_l1719_171993


namespace NUMINAMATH_GPT_units_digit_is_seven_l1719_171969

-- Defining the structure of the three-digit number and its properties
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

def four_times_original (a b c : ℕ) : ℕ := 4 * original_number a b c
def subtract_reversed (a b c : ℕ) : ℕ := four_times_original a b c - reversed_number a b c

-- Theorem statement: Given the condition, what is the units digit of the result?
theorem units_digit_is_seven (a b c : ℕ) (h : a = c + 3) : (subtract_reversed a b c) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_is_seven_l1719_171969


namespace NUMINAMATH_GPT_squares_overlap_ratio_l1719_171945

theorem squares_overlap_ratio (a b : ℝ) (h1 : 0.52 * a^2 = a^2 - (a^2 - 0.52 * a^2))
                             (h2 : 0.73 * b^2 = b^2 - (b^2 - 0.73 * b^2)) :
                             a / b = 3 / 4 := by
sorry

end NUMINAMATH_GPT_squares_overlap_ratio_l1719_171945
