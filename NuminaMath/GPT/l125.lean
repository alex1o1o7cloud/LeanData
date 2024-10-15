import Mathlib

namespace NUMINAMATH_GPT_girl_travel_distance_l125_12596

def speed : ℝ := 6 -- meters per second
def time : ℕ := 16 -- seconds

def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem girl_travel_distance : distance speed time = 96 :=
by 
  unfold distance
  sorry

end NUMINAMATH_GPT_girl_travel_distance_l125_12596


namespace NUMINAMATH_GPT_cost_per_tree_l125_12507

theorem cost_per_tree
    (initial_temperature : ℝ := 80)
    (final_temperature : ℝ := 78.2)
    (total_cost : ℝ := 108)
    (temperature_drop_per_tree : ℝ := 0.1) :
    total_cost / ((initial_temperature - final_temperature) / temperature_drop_per_tree) = 6 :=
by sorry

end NUMINAMATH_GPT_cost_per_tree_l125_12507


namespace NUMINAMATH_GPT_parabola_sum_is_neg_fourteen_l125_12553

noncomputable def parabola_sum (a b c : ℝ) : ℝ := a + b + c

theorem parabola_sum_is_neg_fourteen :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = -(x + 3)^2 + 2) ∧
    ((-1)^2 = a * (-1 + 3)^2 + 6) ∧ 
    ((-3)^2 = a * (-3 + 3)^2 + 2) ∧
    (parabola_sum a b c = -14) :=
sorry

end NUMINAMATH_GPT_parabola_sum_is_neg_fourteen_l125_12553


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l125_12597

theorem number_of_sides_of_polygon (n : ℕ) (h : 3 * (n * (n - 3) / 2) - n = 21) : n = 6 :=
by sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l125_12597


namespace NUMINAMATH_GPT_polar_bear_trout_l125_12588

/-
Question: How many buckets of trout does the polar bear eat daily?
Conditions:
  1. The polar bear eats some amount of trout and 0.4 bucket of salmon daily.
  2. The polar bear eats a total of 0.6 buckets of fish daily.
Answer: 0.2 buckets of trout daily.
-/

theorem polar_bear_trout (trout salmon total : ℝ) 
  (h1 : salmon = 0.4)
  (h2 : total = 0.6)
  (h3 : trout + salmon = total) :
  trout = 0.2 :=
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_polar_bear_trout_l125_12588


namespace NUMINAMATH_GPT_innings_question_l125_12554

theorem innings_question (n : ℕ) (runs_in_inning : ℕ) (avg_increase : ℕ) (new_avg : ℕ) 
  (h_runs_in_inning : runs_in_inning = 88) 
  (h_avg_increase : avg_increase = 3) 
  (h_new_avg : new_avg = 40)
  (h_eq : 37 * n + runs_in_inning = new_avg * (n + 1)): n + 1 = 17 :=
by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_innings_question_l125_12554


namespace NUMINAMATH_GPT_prime_divides_expression_l125_12573

theorem prime_divides_expression (p : ℕ) (hp : Nat.Prime p) : ∃ n : ℕ, p ∣ (2^n + 3^n + 6^n - 1) := 
by
  sorry

end NUMINAMATH_GPT_prime_divides_expression_l125_12573


namespace NUMINAMATH_GPT_parallel_lines_slope_l125_12500

theorem parallel_lines_slope (a : ℝ) :
  (∀ (x y : ℝ), x + a * y + 6 = 0 ∧ (a - 2) * x + 3 * y + 2 * a = 0 → (1 / (a - 2) = a / 3)) →
  a = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_parallel_lines_slope_l125_12500


namespace NUMINAMATH_GPT_compound_interest_rate_l125_12540

theorem compound_interest_rate :
  ∃ r : ℝ, (1000 * (1 + r)^3 = 1331.0000000000005) ∧ r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l125_12540


namespace NUMINAMATH_GPT_sqrt_mul_sqrt_l125_12591

theorem sqrt_mul_sqrt (h1 : Real.sqrt 25 = 5) : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_mul_sqrt_l125_12591


namespace NUMINAMATH_GPT_quadratic_sum_of_b_and_c_l125_12563

theorem quadratic_sum_of_b_and_c :
  ∃ b c : ℝ, (∀ x : ℝ, x^2 - 20 * x + 36 = (x + b)^2 + c) ∧ b + c = -74 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_of_b_and_c_l125_12563


namespace NUMINAMATH_GPT_hash_four_times_l125_12577

noncomputable def hash (N : ℝ) : ℝ := 0.6 * N + 2

theorem hash_four_times (N : ℝ) : hash (hash (hash (hash N))) = 11.8688 :=
  sorry

end NUMINAMATH_GPT_hash_four_times_l125_12577


namespace NUMINAMATH_GPT_fraction_strawberries_remaining_l125_12543

theorem fraction_strawberries_remaining 
  (baskets : ℕ)
  (strawberries_per_basket : ℕ)
  (hedgehogs : ℕ)
  (strawberries_per_hedgehog : ℕ)
  (h1 : baskets = 3)
  (h2 : strawberries_per_basket = 900)
  (h3 : hedgehogs = 2)
  (h4 : strawberries_per_hedgehog = 1050) :
  (baskets * strawberries_per_basket - hedgehogs * strawberries_per_hedgehog) / (baskets * strawberries_per_basket) = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_strawberries_remaining_l125_12543


namespace NUMINAMATH_GPT_find_number_l125_12505

noncomputable def number_with_point_one_percent (x : ℝ) : Prop :=
  0.1 * x / 100 = 12.356

theorem find_number :
  ∃ x : ℝ, number_with_point_one_percent x ∧ x = 12356 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l125_12505


namespace NUMINAMATH_GPT_correct_operation_l125_12559

variable (a b : ℝ)

theorem correct_operation (h1 : a^2 + a^3 ≠ a^5)
                          (h2 : (-a^2)^3 ≠ a^6)
                          (h3 : -2*a^3*b / (a*b) ≠ -2*a^2*b) :
                          a^2 * a^3 = a^5 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l125_12559


namespace NUMINAMATH_GPT_return_trip_time_l125_12531

variables (d p w : ℝ)
-- Condition 1: The outbound trip against the wind took 120 minutes.
axiom h1 : d = 120 * (p - w)
-- Condition 2: The return trip with the wind took 15 minutes less than it would in still air.
axiom h2 : d / (p + w) = d / p - 15

-- Translate the conclusion that needs to be proven in Lean 4
theorem return_trip_time (h1 : d = 120 * (p - w)) (h2 : d / (p + w) = d / p - 15) : (d / (p + w) = 15) ∨ (d / (p + w) = 85) :=
sorry

end NUMINAMATH_GPT_return_trip_time_l125_12531


namespace NUMINAMATH_GPT_cube_rolling_impossible_l125_12595

-- Definitions
def paintedCube : Type := sorry   -- Define a painted black-and-white cube.
def chessboard : Type := sorry    -- Define the chessboard.
def roll (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the rolling over the board visiting each square exactly once.
def matchColors (c : paintedCube) (b : chessboard) : Prop := sorry   -- Define the condition that colors match on contact.

-- Theorem
theorem cube_rolling_impossible (c : paintedCube) (b : chessboard)
  (h1 : roll c b) : ¬ matchColors c b := sorry

end NUMINAMATH_GPT_cube_rolling_impossible_l125_12595


namespace NUMINAMATH_GPT_avg_of_first_5_multiples_of_5_l125_12580

theorem avg_of_first_5_multiples_of_5 : (5 + 10 + 15 + 20 + 25) / 5 = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_avg_of_first_5_multiples_of_5_l125_12580


namespace NUMINAMATH_GPT_largest_real_number_l125_12575

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 8 / 9) : x = 63 / 8 := sorry

end NUMINAMATH_GPT_largest_real_number_l125_12575


namespace NUMINAMATH_GPT_trinomial_identity_l125_12561

theorem trinomial_identity :
  let a := 23
  let b := 15
  let c := 7
  (a + b + c)^2 - (a^2 + b^2 + c^2) = 1222 :=
by
  let a := 23
  let b := 15
  let c := 7
  sorry

end NUMINAMATH_GPT_trinomial_identity_l125_12561


namespace NUMINAMATH_GPT_remainder_division_lemma_l125_12502

theorem remainder_division_lemma (j : ℕ) (hj : 0 < j) (hmod : 132 % (j^2) = 12) : 250 % j = 0 :=
sorry

end NUMINAMATH_GPT_remainder_division_lemma_l125_12502


namespace NUMINAMATH_GPT_max_a_value_l125_12589

theorem max_a_value : 
  (∀ (x : ℝ), (x - 1) * x - (a - 2) * (a + 1) ≥ 1) → a ≤ 3 / 2 :=
sorry

end NUMINAMATH_GPT_max_a_value_l125_12589


namespace NUMINAMATH_GPT_ferry_distance_l125_12550

theorem ferry_distance 
  (x : ℝ)
  (v_w : ℝ := 3)  -- speed of water flow in km/h
  (t_downstream : ℝ := 5)  -- time taken to travel downstream in hours
  (t_upstream : ℝ := 7)  -- time taken to travel upstream in hours
  (eqn : x / t_downstream - v_w = x / t_upstream + v_w) :
  x = 105 :=
sorry

end NUMINAMATH_GPT_ferry_distance_l125_12550


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l125_12557

variable (a : ℕ → ℝ) (d : ℝ) (m : ℕ)

noncomputable def a_seq := ∀ n, a n = a 1 + (n - 1) * d

theorem arithmetic_sequence_problem
  (h1 : a 1 = 0)
  (h2 : d ≠ 0)
  (h3 : a m = a 1 + a 2 + a 3 + a 4 + a 5) :
  m = 11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l125_12557


namespace NUMINAMATH_GPT_find_x2_plus_y2_l125_12508

theorem find_x2_plus_y2 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : x * y + x + y = 90) 
  (h2 : x^2 * y + x * y^2 = 1122) : 
  x^2 + y^2 = 1044 :=
sorry

end NUMINAMATH_GPT_find_x2_plus_y2_l125_12508


namespace NUMINAMATH_GPT_product_eq_neg_one_l125_12513

theorem product_eq_neg_one (m b : ℚ) (hm : m = -2 / 3) (hb : b = 3 / 2) : m * b = -1 :=
by
  rw [hm, hb]
  sorry

end NUMINAMATH_GPT_product_eq_neg_one_l125_12513


namespace NUMINAMATH_GPT_perimeter_of_square_l125_12592

-- Defining the context and proving the equivalence.
theorem perimeter_of_square (x y : ℕ) (h : Nat.gcd x y = 3) (area : ℕ) :
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  perimeter = 24 * Real.sqrt 5 :=
by
  let lcm_xy := Nat.lcm x y
  let side_length := Real.sqrt (20 * lcm_xy)
  let perimeter := 4 * side_length
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l125_12592


namespace NUMINAMATH_GPT_temperature_difference_l125_12585

theorem temperature_difference (T_high T_low : ℝ) (h1 : T_high = 8) (h2 : T_low = -2) : T_high - T_low = 10 :=
by
  sorry

end NUMINAMATH_GPT_temperature_difference_l125_12585


namespace NUMINAMATH_GPT_same_cost_for_same_sheets_l125_12539

def John's_Photo_World_cost (x : ℕ) : ℝ := 2.75 * x + 125
def Sam's_Picture_Emporium_cost (x : ℕ) : ℝ := 1.50 * x + 140

theorem same_cost_for_same_sheets :
  ∃ (x : ℕ), John's_Photo_World_cost x = Sam's_Picture_Emporium_cost x ∧ x = 12 :=
by
  sorry

end NUMINAMATH_GPT_same_cost_for_same_sheets_l125_12539


namespace NUMINAMATH_GPT_sum_of_solutions_l125_12521

theorem sum_of_solutions (y1 y2 : ℝ) (h1 : y1 + 16 / y1 = 12) (h2 : y2 + 16 / y2 = 12) : 
  y1 + y2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l125_12521


namespace NUMINAMATH_GPT_maximal_n_is_k_minus_1_l125_12547

section
variable (k : ℕ) (n : ℕ)
variable (cards : Finset ℕ)
variable (red : List ℕ) (blue : List (List ℕ))

-- Conditions
axiom h_k_pos : k > 1
axiom h_card_count : cards = Finset.range (2 * n + 1)
axiom h_initial_red : red = (List.range' 1 (2 * n)).reverse
axiom h_initial_blue : blue.length = k

-- Question translated to a goal
theorem maximal_n_is_k_minus_1 (h : ∀ (n' : ℕ), n' ≤ (k - 1)) : n = k - 1 :=
sorry
end

end NUMINAMATH_GPT_maximal_n_is_k_minus_1_l125_12547


namespace NUMINAMATH_GPT_neg_P_is_univ_l125_12594

noncomputable def P : Prop :=
  ∃ x0 : ℝ, x0^2 + 2 * x0 + 2 ≤ 0

theorem neg_P_is_univ :
  ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_neg_P_is_univ_l125_12594


namespace NUMINAMATH_GPT_alice_questions_wrong_l125_12525

theorem alice_questions_wrong (a b c d : ℝ) 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 3) 
  (h3 : c = 7) : 
  a = 8.5 := 
by
  sorry

end NUMINAMATH_GPT_alice_questions_wrong_l125_12525


namespace NUMINAMATH_GPT_ratio_eq_neg_1009_l125_12541

theorem ratio_eq_neg_1009 (p q : ℝ) (h : (1 / p + 1 / q) / (1 / p - 1 / q) = 1009) : (p + q) / (p - q) = -1009 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_eq_neg_1009_l125_12541


namespace NUMINAMATH_GPT_x_y_sum_vals_l125_12587

theorem x_y_sum_vals (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 6) (h3 : x > y) : x + y = -3 ∨ x + y = -9 := 
by
  sorry

end NUMINAMATH_GPT_x_y_sum_vals_l125_12587


namespace NUMINAMATH_GPT_bus_people_next_pickup_point_l125_12524

theorem bus_people_next_pickup_point (bus_capacity : ℕ) (fraction_first_pickup : ℚ) (cannot_board : ℕ)
  (h1 : bus_capacity = 80)
  (h2 : fraction_first_pickup = 3 / 5)
  (h3 : cannot_board = 18) : 
  ∃ people_next_pickup : ℕ, people_next_pickup = 50 :=
by
  sorry

end NUMINAMATH_GPT_bus_people_next_pickup_point_l125_12524


namespace NUMINAMATH_GPT_cube_points_l125_12512

theorem cube_points (A B C D E F : ℕ) 
  (h1 : A + B = 13)
  (h2 : C + D = 13)
  (h3 : E + F = 13)
  (h4 : A + C + E = 16)
  (h5 : B + D + E = 24) :
  F = 6 :=
by
  sorry  -- Proof to be filled in by the user

end NUMINAMATH_GPT_cube_points_l125_12512


namespace NUMINAMATH_GPT_average_sales_is_104_l125_12528

-- Define the sales data for the months January to May
def january_sales : ℕ := 150
def february_sales : ℕ := 90
def march_sales : ℕ := 60
def april_sales : ℕ := 140
def may_sales : ℕ := 100
def may_discount : ℕ := 20

-- Define the adjusted sales for May after applying the discount
def adjusted_may_sales : ℕ := may_sales - (may_sales * may_discount / 100)

-- Define the total sales from January to May
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + adjusted_may_sales

-- Define the number of months
def number_of_months : ℕ := 5

-- Define the average sales per month
def average_sales_per_month : ℕ := total_sales / number_of_months

-- Prove that the average sales per month is equal to 104
theorem average_sales_is_104 : average_sales_per_month = 104 := by
  -- Here, we'd write the proof, but we'll leave it as 'sorry' for now
  sorry

end NUMINAMATH_GPT_average_sales_is_104_l125_12528


namespace NUMINAMATH_GPT_sally_quarters_total_l125_12526

/--
Sally originally had 760 quarters. She received 418 more quarters. 
Prove that the total number of quarters Sally has now is 1178.
-/
theorem sally_quarters_total : 
  let original_quarters := 760
  let additional_quarters := 418
  original_quarters + additional_quarters = 1178 :=
by
  let original_quarters := 760
  let additional_quarters := 418
  show original_quarters + additional_quarters = 1178
  sorry

end NUMINAMATH_GPT_sally_quarters_total_l125_12526


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l125_12523

theorem isosceles_triangle_perimeter (a b : ℕ) (h_a : a = 8 ∨ a = 9) (h_b : b = 8 ∨ b = 9) 
(h_iso : a = a) (h_tri_ineq : a + a > b ∧ a + b > a ∧ b + a > a) :
  a + a + b = 25 ∨ a + a + b = 26 := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l125_12523


namespace NUMINAMATH_GPT_truncated_cone_volume_l125_12527

theorem truncated_cone_volume 
  (V_initial : ℝ)
  (r_ratio : ℝ)
  (V_final : ℝ)
  (r_ratio_eq : r_ratio = 1 / 2)
  (V_initial_eq : V_initial = 1) :
  V_final = 7 / 8 :=
  sorry

end NUMINAMATH_GPT_truncated_cone_volume_l125_12527


namespace NUMINAMATH_GPT_value_of_k_for_square_of_binomial_l125_12566

theorem value_of_k_for_square_of_binomial (a k : ℝ) : (x : ℝ) → x^2 - 14 * x + k = (x - a)^2 → k = 49 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_value_of_k_for_square_of_binomial_l125_12566


namespace NUMINAMATH_GPT_third_angle_in_triangle_sum_of_angles_in_triangle_l125_12583

theorem third_angle_in_triangle (a b : ℝ) (h₁ : a = 50) (h₂ : b = 80) : 180 - a - b = 50 :=
by
  rw [h₁, h₂]
  norm_num

-- Adding this to demonstrate the constraint of the problem: Sum of angles in a triangle is 180°
theorem sum_of_angles_in_triangle (a b c : ℝ) (h₁: a + b + c = 180) : true :=
by
  trivial

end NUMINAMATH_GPT_third_angle_in_triangle_sum_of_angles_in_triangle_l125_12583


namespace NUMINAMATH_GPT_discriminant_positive_l125_12529

theorem discriminant_positive
  (a b c : ℝ)
  (h : (a + b + c) * c < 0) : b^2 - 4 * a * c > 0 :=
sorry

end NUMINAMATH_GPT_discriminant_positive_l125_12529


namespace NUMINAMATH_GPT_parabola_point_dot_product_eq_neg4_l125_12570

-- Definition of the parabola
def is_parabola_point (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

-- Definition of the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Coordinates of origin
def origin : ℝ × ℝ := (0, 0)

-- Vector from origin to point A
def vector_OA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, A.2)

-- Vector from point A to the focus
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - A.1, focus.2 - A.2)

-- Theorem statement
theorem parabola_point_dot_product_eq_neg4 (A : ℝ × ℝ) 
  (hA : is_parabola_point A) 
  (h_dot : dot_product (vector_OA A) (vector_AF A) = -4) :
  A = (1, 2) ∨ A = (1, -2) :=
sorry

end NUMINAMATH_GPT_parabola_point_dot_product_eq_neg4_l125_12570


namespace NUMINAMATH_GPT_packets_in_box_l125_12506

theorem packets_in_box 
  (coffees_per_day : ℕ) 
  (packets_per_coffee : ℕ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (days : ℕ) 
  (P : ℕ) 
  (h_coffees_per_day : coffees_per_day = 2)
  (h_packets_per_coffee : packets_per_coffee = 1)
  (h_cost_per_box : cost_per_box = 4)
  (h_total_cost : total_cost = 24)
  (h_days : days = 90)
  : P = 30 := 
by
  sorry

end NUMINAMATH_GPT_packets_in_box_l125_12506


namespace NUMINAMATH_GPT_length_base_bc_l125_12530

theorem length_base_bc {A B C D : Type} [Inhabited A]
  (AB AC : ℕ)
  (BD : ℕ → ℕ → ℕ → ℕ) -- function for the median on AC
  (perimeter1 perimeter2 : ℕ)
  (h1 : AB = AC)
  (h2 : perimeter1 = 24 ∨ perimeter2 = 30)
  (AD CD : ℕ) :
  (AD = CD ∧ (∃ ab ad cd, ab + ad = perimeter1 ∧ cd + ad = perimeter2 ∧ ((AB = 2 * AD ∧ BC = 30 - CD) ∨ (AB = 2 * AD ∧ BC = 24 - CD)))) →
  (BC = 22 ∨ BC = 14) := 
sorry

end NUMINAMATH_GPT_length_base_bc_l125_12530


namespace NUMINAMATH_GPT_find_cos2α_l125_12522

noncomputable def cos2α (tanα : ℚ) : ℚ :=
  (1 - tanα^2) / (1 + tanα^2)

theorem find_cos2α (h : tanα = (3 / 4)) : cos2α tanα = (7 / 25) :=
by
  rw [cos2α, h]
  -- here the simplification steps would be performed
  sorry

end NUMINAMATH_GPT_find_cos2α_l125_12522


namespace NUMINAMATH_GPT_problem_l125_12564

theorem problem
  (x y : ℝ)
  (h₁ : x - 2 * y = -5)
  (h₂ : x * y = -2) :
  2 * x^2 * y - 4 * x * y^2 = 20 := 
by
  sorry

end NUMINAMATH_GPT_problem_l125_12564


namespace NUMINAMATH_GPT_sad_girls_count_l125_12598

-- Given definitions
def total_children : ℕ := 60
def happy_children : ℕ := 30
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def boys_neither_happy_nor_sad : ℕ := 10

-- Intermediate definitions
def sad_boys : ℕ := boys - happy_boys - boys_neither_happy_nor_sad
def sad_girls : ℕ := sad_children - sad_boys

-- Theorem to prove that the number of sad girls is 4
theorem sad_girls_count : sad_girls = 4 := by
  sorry

end NUMINAMATH_GPT_sad_girls_count_l125_12598


namespace NUMINAMATH_GPT_c_work_rate_l125_12568

variable {W : ℝ} -- Denoting the work by W
variable {a_rate : ℝ} -- Work rate of a
variable {b_rate : ℝ} -- Work rate of b
variable {c_rate : ℝ} -- Work rate of c
variable {combined_rate : ℝ} -- Combined work rate of a, b, and c

theorem c_work_rate (W a_rate b_rate c_rate combined_rate : ℝ)
  (h1 : a_rate = W / 12)
  (h2 : b_rate = W / 24)
  (h3 : combined_rate = W / 4)
  (h4 : combined_rate = a_rate + b_rate + c_rate) :
  c_rate = W / 4.5 :=
by
  sorry

end NUMINAMATH_GPT_c_work_rate_l125_12568


namespace NUMINAMATH_GPT_tangent_line_through_origin_l125_12516

theorem tangent_line_through_origin (x : ℝ) (h₁ : 0 < x) (h₂ : ∀ x, ∃ y, y = 2 * Real.log x) (h₃ : ∀ x, y = 2 * Real.log x) :
  x = Real.exp 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_through_origin_l125_12516


namespace NUMINAMATH_GPT_isabel_camera_pics_l125_12533

-- Conditions
def phone_pics := 2
def albums := 3
def pics_per_album := 2

-- Define the total pictures and camera pictures
def total_pics := albums * pics_per_album
def camera_pics := total_pics - phone_pics

theorem isabel_camera_pics : camera_pics = 4 :=
by
  -- The goal is translated from the correct answer in step b)
  sorry

end NUMINAMATH_GPT_isabel_camera_pics_l125_12533


namespace NUMINAMATH_GPT_muffin_count_l125_12593

theorem muffin_count (doughnuts cookies muffins : ℕ) (h1 : doughnuts = 50) (h2 : cookies = (3 * doughnuts) / 5) (h3 : muffins = (1 * doughnuts) / 5) : muffins = 10 :=
by sorry

end NUMINAMATH_GPT_muffin_count_l125_12593


namespace NUMINAMATH_GPT_possible_values_2a_b_l125_12571

theorem possible_values_2a_b (a b x y z : ℕ) (h1: a^x = 1994^z) (h2: b^y = 1994^z) (h3: 1/x + 1/y = 1/z) : 
  (2 * a + b = 1001) ∨ (2 * a + b = 1996) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_2a_b_l125_12571


namespace NUMINAMATH_GPT_a1d1_a2d2_a3d3_eq_neg1_l125_12581

theorem a1d1_a2d2_a3d3_eq_neg1 (a1 a2 a3 d1 d2 d3 : ℝ) (h : ∀ x : ℝ, 
  x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 + 1)) : 
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end NUMINAMATH_GPT_a1d1_a2d2_a3d3_eq_neg1_l125_12581


namespace NUMINAMATH_GPT_balls_in_boxes_ways_l125_12582

theorem balls_in_boxes_ways : ∃ (ways : ℕ), ways = 56 :=
by
  let n := 5
  let m := 4
  let ways := 56
  sorry

end NUMINAMATH_GPT_balls_in_boxes_ways_l125_12582


namespace NUMINAMATH_GPT_john_spent_fraction_on_snacks_l125_12537

theorem john_spent_fraction_on_snacks (x : ℚ) :
  (∀ (x : ℚ), (1 - x) * 20 - (3 / 4) * (1 - x) * 20 = 4) → (x = 1 / 5) :=
by sorry

end NUMINAMATH_GPT_john_spent_fraction_on_snacks_l125_12537


namespace NUMINAMATH_GPT_journey_total_distance_l125_12558

theorem journey_total_distance (D : ℝ) 
  (h1 : (D / 3) / 21 + (D / 3) / 14 + (D / 3) / 6 = 12) : 
  D = 126 :=
sorry

end NUMINAMATH_GPT_journey_total_distance_l125_12558


namespace NUMINAMATH_GPT_intersection_complement_l125_12535

open Set

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
variable (hA : A = {2, 3, 5, 6})
variable (hB : B = {1, 3, 4, 6, 7})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 5} :=
sorry

end NUMINAMATH_GPT_intersection_complement_l125_12535


namespace NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l125_12549

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 2) = a (n + 1) + d

theorem find_n_in_arithmetic_sequence (x : ℝ) (n : ℕ) (b : ℕ → ℝ)
  (h1 : b 1 = Real.exp x) 
  (h2 : b 2 = x) 
  (h3 : is_arithmetic_sequence b) : 
  b n = 1 + Real.exp x ↔ n = (1 + x) / (x - Real.exp x) :=
sorry

end NUMINAMATH_GPT_find_n_in_arithmetic_sequence_l125_12549


namespace NUMINAMATH_GPT_find_second_number_l125_12534

theorem find_second_number 
  (k : ℕ)
  (h_k_is_1 : k = 1)
  (h_div_1657 : ∃ q1 : ℕ, 1657 = k * q1 + 10)
  (h_div_x : ∃ q2 : ℕ, ∀ x : ℕ, x = k * q2 + 7 → x = 1655) 
: ∃ x : ℕ, x = 1655 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l125_12534


namespace NUMINAMATH_GPT_complex_number_eq_l125_12548

theorem complex_number_eq (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 :=
sorry

end NUMINAMATH_GPT_complex_number_eq_l125_12548


namespace NUMINAMATH_GPT_each_girl_gets_2_dollars_after_debt_l125_12538

variable (Lulu_saved : ℕ)
variable (Nora_saved : ℕ)
variable (Tamara_saved : ℕ)
variable (debt : ℕ)
variable (remaining : ℕ)
variable (each_girl_share : ℕ)

-- Conditions
axiom Lulu_saved_cond : Lulu_saved = 6
axiom Nora_saved_cond : Nora_saved = 5 * Lulu_saved
axiom Nora_Tamara_relation : Nora_saved = 3 * Tamara_saved
axiom debt_cond : debt = 40

-- Question == Answer to prove
theorem each_girl_gets_2_dollars_after_debt (total_saved : ℕ) (remaining: ℕ) (each_girl_share: ℕ) :
  total_saved = Tamara_saved + Nora_saved + Lulu_saved →
  remaining = total_saved - debt →
  each_girl_share = remaining / 3 →
  each_girl_share = 2 := 
sorry

end NUMINAMATH_GPT_each_girl_gets_2_dollars_after_debt_l125_12538


namespace NUMINAMATH_GPT_find_r4_l125_12517

-- Definitions of the problem conditions
variable (r1 r2 r3 r4 r5 r6 r7 : ℝ)
-- Given radius of the smallest circle
axiom smallest_circle : r1 = 6
-- Given radius of the largest circle
axiom largest_circle : r7 = 24
-- Given that radii of circles form a geometric sequence
axiom geometric_sequence : r2 = r1 * (r7 / r1)^(1/6) ∧ 
                            r3 = r1 * (r7 / r1)^(2/6) ∧
                            r4 = r1 * (r7 / r1)^(3/6) ∧
                            r5 = r1 * (r7 / r1)^(4/6) ∧
                            r6 = r1 * (r7 / r1)^(5/6)

-- Statement to prove
theorem find_r4 : r4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_r4_l125_12517


namespace NUMINAMATH_GPT_marble_problem_l125_12503

theorem marble_problem
  (h1 : ∀ x : ℕ, x > 0 → (x + 2) * ((220 / x) - 1) = 220) :
  ∃ x : ℕ, x > 0 ∧ (x + 2) * ((220 / ↑x) - 1) = 220 ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_marble_problem_l125_12503


namespace NUMINAMATH_GPT_symmetry_axes_condition_l125_12545

/-- Define the property of having axes of symmetry for a geometric figure -/
def has_symmetry_axes (bounded : Bool) (two_parallel_axes : Bool) : Prop :=
  if bounded then 
    ¬ two_parallel_axes 
  else 
    true

/-- Main theorem stating the condition on symmetry axes for bounded and unbounded geometric figures -/
theorem symmetry_axes_condition (bounded : Bool) : 
  ∃ two_parallel_axes : Bool, has_symmetry_axes bounded two_parallel_axes :=
by
  -- The proof itself is not necessary as per the problem statement
  sorry

end NUMINAMATH_GPT_symmetry_axes_condition_l125_12545


namespace NUMINAMATH_GPT_M_minus_N_l125_12590

theorem M_minus_N (a b c d : ℕ) (h1 : a + b = 20) (h2 : a + c = 24) (h3 : a + d = 22) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) : 
  let M := 2 * b + 26
  let N := 2 * 1 + 26
  (M - N) = 36 :=
by
  sorry

end NUMINAMATH_GPT_M_minus_N_l125_12590


namespace NUMINAMATH_GPT_new_average_of_remaining_students_l125_12579

theorem new_average_of_remaining_students 
  (avg_initial_score : ℝ)
  (num_initial_students : ℕ)
  (dropped_score : ℝ)
  (num_remaining_students : ℕ)
  (new_avg_score : ℝ) 
  (h_avg : avg_initial_score = 62.5)
  (h_num_initial : num_initial_students = 16)
  (h_dropped : dropped_score = 55)
  (h_num_remaining : num_remaining_students = 15)
  (h_new_avg : new_avg_score = 63) :
  let total_initial_score := avg_initial_score * num_initial_students
  let total_remaining_score := total_initial_score - dropped_score
  let calculated_new_avg := total_remaining_score / num_remaining_students
  calculated_new_avg = new_avg_score := 
by
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_new_average_of_remaining_students_l125_12579


namespace NUMINAMATH_GPT_jason_steps_is_8_l125_12509

-- Definition of the problem conditions
def nancy_steps (jason_steps : ℕ) := 3 * jason_steps -- Nancy steps 3 times as often as Jason

def together_steps (jason_steps nancy_steps : ℕ) := jason_steps + nancy_steps -- Total steps

-- Lean statement of the problem to prove
theorem jason_steps_is_8 (J : ℕ) (h₁ : together_steps J (nancy_steps J) = 32) : J = 8 :=
sorry

end NUMINAMATH_GPT_jason_steps_is_8_l125_12509


namespace NUMINAMATH_GPT_candy_problem_l125_12578

theorem candy_problem (
  a : ℤ
) : (a % 10 = 6) →
    (a % 15 = 11) →
    (200 ≤ a ∧ a ≤ 250) →
    (a = 206 ∨ a = 236) :=
sorry

end NUMINAMATH_GPT_candy_problem_l125_12578


namespace NUMINAMATH_GPT_polynomial_sum_l125_12518

def f (x : ℝ) : ℝ := -4 * x^3 - 3 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + x - 7
def h (x : ℝ) : ℝ := 3 * x^3 + 6 * x^2 + 3 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = x^3 - 2 * x^2 + 6 * x - 10 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l125_12518


namespace NUMINAMATH_GPT_average_increase_l125_12576

theorem average_increase (x : ℝ) (y : ℝ) (h : y = 0.245 * x + 0.321) : 
  ∀ x_increase : ℝ, x_increase = 1 → (0.245 * (x + x_increase) + 0.321) - (0.245 * x + 0.321) = 0.245 :=
by
  intro x_increase
  intro hx
  rw [hx]
  simp
  sorry

end NUMINAMATH_GPT_average_increase_l125_12576


namespace NUMINAMATH_GPT_total_cartons_used_l125_12552

theorem total_cartons_used (x : ℕ) (y : ℕ) (h1 : y = 24) (h2 : 2 * x + 3 * y = 100) : x + y = 38 :=
sorry

end NUMINAMATH_GPT_total_cartons_used_l125_12552


namespace NUMINAMATH_GPT_ship_lighthouse_distance_l125_12514

-- Definitions for conditions
def speed : ℝ := 15 -- speed of the ship in km/h
def time : ℝ := 4  -- time the ship sails eastward in hours
def angle_A : ℝ := 60 -- angle at point A in degrees
def angle_C : ℝ := 30 -- angle at point C in degrees

-- Main theorem statement
theorem ship_lighthouse_distance (d_A_C : ℝ) (d_C_B : ℝ) : d_A_C = speed * time → d_C_B = 60 := 
by sorry

end NUMINAMATH_GPT_ship_lighthouse_distance_l125_12514


namespace NUMINAMATH_GPT_prove_arithmetic_sequence_l125_12510

def arithmetic_sequence (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2 * x + 3
| n => sorry

theorem prove_arithmetic_sequence {x : ℝ} (a : ℕ → ℝ)
  (h_terms : a 0 = x - 1 ∧ a 1 = x + 1 ∧ a 2 = 2 * x + 3)
  (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0)) :
  x = 0 ∧ ∀ n, a n = 2 * n - 3 :=
by
  sorry

end NUMINAMATH_GPT_prove_arithmetic_sequence_l125_12510


namespace NUMINAMATH_GPT_solve_inequality_system_l125_12515

theorem solve_inequality_system (y : ℝ) :
  (2 * (y + 1) < 5 * y - 7) ∧ ((y + 2) / 2 < 5) ↔ (3 < y) ∧ (y < 8) := 
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l125_12515


namespace NUMINAMATH_GPT_talia_drives_total_distance_l125_12546

-- Define the distances for each leg of the trip
def distance_house_to_park : ℕ := 5
def distance_park_to_store : ℕ := 3
def distance_store_to_friend : ℕ := 6
def distance_friend_to_house : ℕ := 4

-- Define the total distance Talia drives
def total_distance := distance_house_to_park + distance_park_to_store + distance_store_to_friend + distance_friend_to_house

-- Prove that the total distance is 18 miles
theorem talia_drives_total_distance : total_distance = 18 := by
  sorry

end NUMINAMATH_GPT_talia_drives_total_distance_l125_12546


namespace NUMINAMATH_GPT_length_of_one_side_l125_12511

-- Definitions according to the conditions
def perimeter (nonagon : Type) : ℝ := 171
def sides (nonagon : Type) : ℕ := 9

-- Math proof problem to prove
theorem length_of_one_side (nonagon : Type) : perimeter nonagon / sides nonagon = 19 :=
by
  sorry

end NUMINAMATH_GPT_length_of_one_side_l125_12511


namespace NUMINAMATH_GPT_prove_A_plus_B_l125_12584

variable (A B : ℝ)

theorem prove_A_plus_B (h : ∀ x : ℝ, x ≠ 2 → (A / (x - 2) + B * (x + 3) = (-5 * x^2 + 20 * x + 34) / (x - 2))) : A + B = 9 := by
  sorry

end NUMINAMATH_GPT_prove_A_plus_B_l125_12584


namespace NUMINAMATH_GPT_subset_A_B_l125_12520

def A : Set ℝ := { x | x^2 - 3 * x + 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem subset_A_B : A ⊆ B := sorry

end NUMINAMATH_GPT_subset_A_B_l125_12520


namespace NUMINAMATH_GPT_g_g_g_g_2_eq_16_l125_12556

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem g_g_g_g_2_eq_16 : g (g (g (g 2))) = 16 := by
  sorry

end NUMINAMATH_GPT_g_g_g_g_2_eq_16_l125_12556


namespace NUMINAMATH_GPT_range_of_a_l125_12542

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 + 2 * a * x + 2 < 0) ↔ 0 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l125_12542


namespace NUMINAMATH_GPT_john_average_speed_l125_12551

noncomputable def time_uphill : ℝ := 45 / 60 -- 45 minutes converted to hours
noncomputable def distance_uphill : ℝ := 2   -- 2 km

noncomputable def time_downhill : ℝ := 15 / 60 -- 15 minutes converted to hours
noncomputable def distance_downhill : ℝ := 2   -- 2 km

noncomputable def total_distance : ℝ := distance_uphill + distance_downhill
noncomputable def total_time : ℝ := time_uphill + time_downhill

theorem john_average_speed : total_distance / total_time = 4 :=
by
  have h1 : total_distance = 4 := by sorry
  have h2 : total_time = 1 := by sorry
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_john_average_speed_l125_12551


namespace NUMINAMATH_GPT_eval_difference_of_squares_l125_12501

theorem eval_difference_of_squares :
  (81^2 - 49^2 = 4160) :=
by
  -- Since the exact mathematical content is established in a formal context, 
  -- we omit the detailed proof steps.
  sorry

end NUMINAMATH_GPT_eval_difference_of_squares_l125_12501


namespace NUMINAMATH_GPT_Alex_is_26_l125_12565

-- Define the ages as integers
variable (Alex Jose Zack Inez : ℤ)

-- Conditions of the problem
variable (h1 : Alex = Jose + 6)
variable (h2 : Zack = Inez + 5)
variable (h3 : Inez = 18)
variable (h4 : Jose = Zack - 3)

-- Theorem we need to prove
theorem Alex_is_26 (h1: Alex = Jose + 6) (h2 : Zack = Inez + 5) (h3 : Inez = 18) (h4 : Jose = Zack - 3) : Alex = 26 :=
by
  sorry

end NUMINAMATH_GPT_Alex_is_26_l125_12565


namespace NUMINAMATH_GPT_savings_example_l125_12560

def window_cost : ℕ → ℕ := λ n => n * 120

def discount_windows (n : ℕ) : ℕ := (n / 6) * 2 + n

def effective_cost (needed : ℕ) : ℕ := 
  let free_windows := (needed / 8) * 2
  (needed - free_windows) * 120

def combined_cost (n m : ℕ) : ℕ :=
  effective_cost (n + m)

def separate_cost (needed1 needed2 : ℕ) : ℕ :=
  effective_cost needed1 + effective_cost needed2

def savings_if_combined (n m : ℕ) : ℕ :=
  separate_cost n m - combined_cost n m

theorem savings_example : savings_if_combined 12 9 = 360 := by
  sorry

end NUMINAMATH_GPT_savings_example_l125_12560


namespace NUMINAMATH_GPT_kit_time_to_ticket_window_l125_12536

theorem kit_time_to_ticket_window 
  (rate : ℝ)
  (remaining_distance : ℝ)
  (yard_to_feet_conv : ℝ)
  (new_rate : rate = 90 / 30)
  (remaining_distance_in_feet : remaining_distance = 100 * yard_to_feet_conv)
  (yard_to_feet_conv_val : yard_to_feet_conv = 3) :
  (remaining_distance / rate = 100) := 
by 
  simp [new_rate, remaining_distance_in_feet, yard_to_feet_conv_val]
  sorry

end NUMINAMATH_GPT_kit_time_to_ticket_window_l125_12536


namespace NUMINAMATH_GPT_carrots_cost_l125_12532

/-
Define the problem conditions and parameters.
-/
def num_third_grade_classes := 5
def students_per_third_grade_class := 30
def num_fourth_grade_classes := 4
def students_per_fourth_grade_class := 28
def num_fifth_grade_classes := 4
def students_per_fifth_grade_class := 27

def cost_per_hamburger : ℝ := 2.10
def cost_per_cookie : ℝ := 0.20
def total_lunch_cost : ℝ := 1036

/-
Calculate the total number of students.
-/
def total_students : ℕ :=
  (num_third_grade_classes * students_per_third_grade_class) +
  (num_fourth_grade_classes * students_per_fourth_grade_class) +
  (num_fifth_grade_classes * students_per_fifth_grade_class)

/-
Calculate the cost of hamburgers and cookies.
-/
def hamburgers_cost : ℝ := total_students * cost_per_hamburger
def cookies_cost : ℝ := total_students * cost_per_cookie
def total_hamburgers_and_cookies_cost : ℝ := hamburgers_cost + cookies_cost

/-
State the proof problem: How much do the carrots cost?
-/
theorem carrots_cost : total_lunch_cost - total_hamburgers_and_cookies_cost = 185 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_carrots_cost_l125_12532


namespace NUMINAMATH_GPT_probability_different_colors_l125_12519

theorem probability_different_colors :
  let total_chips := 18
  let blue_chips := 7
  let red_chips := 6
  let yellow_chips := 5
  let prob_first_blue := blue_chips / total_chips
  let prob_first_red := red_chips / total_chips
  let prob_first_yellow := yellow_chips / total_chips
  let prob_second_not_blue := (red_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_red := (blue_chips + yellow_chips) / (total_chips - 1)
  let prob_second_not_yellow := (blue_chips + red_chips) / (total_chips - 1)
  (
    prob_first_blue * prob_second_not_blue +
    prob_first_red * prob_second_not_red +
    prob_first_yellow * prob_second_not_yellow
  ) = 122 / 153 :=
by sorry

end NUMINAMATH_GPT_probability_different_colors_l125_12519


namespace NUMINAMATH_GPT_find_R_l125_12504

theorem find_R (a b : ℝ) (Q R : ℝ) (hQ : Q = 4)
  (h1 : 1/a + 1/b = Q/(a + b))
  (h2 : a/b + b/a = R) : R = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_R_l125_12504


namespace NUMINAMATH_GPT_class_speeds_relationship_l125_12586

theorem class_speeds_relationship (x : ℝ) (hx : 0 < x) :
    (15 / (1.2 * x)) = ((15 / x) - (1 / 2)) :=
sorry

end NUMINAMATH_GPT_class_speeds_relationship_l125_12586


namespace NUMINAMATH_GPT_sara_change_l125_12562

-- Define the costs of individual items
def cost_book_1 : ℝ := 5.5
def cost_book_2 : ℝ := 6.5
def cost_notebook : ℝ := 3
def cost_bookmarks : ℝ := 2

-- Define the discounts and taxes
def discount_books : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Define the payment amount
def amount_given : ℝ := 20

-- Calculate the total cost, discount, and final amount
def discounted_book_cost := (cost_book_1 + cost_book_2) * (1 - discount_books)
def subtotal := discounted_book_cost + cost_notebook + cost_bookmarks
def total_with_tax := subtotal * (1 + sales_tax)
def change := amount_given - total_with_tax

-- State the theorem
theorem sara_change : change = 3.41 := by
  sorry

end NUMINAMATH_GPT_sara_change_l125_12562


namespace NUMINAMATH_GPT_k_for_circle_radius_7_l125_12569

theorem k_for_circle_radius_7 (k : ℝ) :
  (∃ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0) →
  (∃ x y : ℝ, (x + 4)^2 + (y + 2)^2 = 49) →
  k = 29 :=
by
  sorry

end NUMINAMATH_GPT_k_for_circle_radius_7_l125_12569


namespace NUMINAMATH_GPT_sin_sum_of_acute_l125_12572

open Real

theorem sin_sum_of_acute (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  sin (α + β) ≤ sin α + sin β := 
by
  sorry

end NUMINAMATH_GPT_sin_sum_of_acute_l125_12572


namespace NUMINAMATH_GPT_percentage_failed_in_english_l125_12567

theorem percentage_failed_in_english (total_students : ℕ) (hindi_failed : ℕ) (both_failed : ℕ) (both_passed : ℕ) 
  (H1 : hindi_failed = total_students * 25 / 100)
  (H2 : both_failed = total_students * 25 / 100)
  (H3 : both_passed = total_students * 50 / 100)
  : (total_students * 50 / 100) = (total_students * 75 / 100) + (both_failed) - both_passed
:= sorry

end NUMINAMATH_GPT_percentage_failed_in_english_l125_12567


namespace NUMINAMATH_GPT_jackson_star_fish_count_l125_12599

def total_starfish_per_spiral_shell (hermit_crabs : ℕ) (shells_per_crab : ℕ) (total_souvenirs : ℕ) : ℕ :=
  (total_souvenirs - (hermit_crabs + hermit_crabs * shells_per_crab)) / (hermit_crabs * shells_per_crab)

theorem jackson_star_fish_count :
  total_starfish_per_spiral_shell 45 3 450 = 2 :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_jackson_star_fish_count_l125_12599


namespace NUMINAMATH_GPT_Nicole_fish_tanks_l125_12555

-- Definition to express the conditions
def first_tank_water := 8 -- gallons
def second_tank_difference := 2 -- fewer gallons than first tanks
def num_first_tanks := 2
def num_second_tanks := 2
def total_water_four_weeks := 112 -- gallons
def weeks := 4

-- Calculate the total water per week
def water_per_week := (num_first_tanks * first_tank_water) + (num_second_tanks * (first_tank_water - second_tank_difference))

-- Calculate the total number of tanks
def total_tanks := num_first_tanks + num_second_tanks

-- Proof statement
theorem Nicole_fish_tanks : total_water_four_weeks / water_per_week = weeks → total_tanks = 4 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Nicole_fish_tanks_l125_12555


namespace NUMINAMATH_GPT_abs_a_gt_b_l125_12544

theorem abs_a_gt_b (a b : ℝ) (h : a > b) : |a| > b :=
sorry

end NUMINAMATH_GPT_abs_a_gt_b_l125_12544


namespace NUMINAMATH_GPT_plane_intercept_equation_l125_12574

-- Define the conditions in Lean 4
variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

-- State the main theorem
theorem plane_intercept_equation :
  ∃ (p : ℝ → ℝ → ℝ → ℝ), (∀ x y z, p x y z = x / a + y / b + z / c) :=
sorry

end NUMINAMATH_GPT_plane_intercept_equation_l125_12574
