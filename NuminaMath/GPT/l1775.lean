import Mathlib

namespace problem_statement_l1775_177563

def op (x y : ℕ) : ℕ := x^2 + 2*y

theorem problem_statement (a : ℕ) : op a (op a a) = 3*a^2 + 4*a := 
by sorry

end problem_statement_l1775_177563


namespace range_of_m_l1775_177565

noncomputable def A (x : ℝ) : ℝ := x^2 - (3/2) * x + 1

def in_interval (x : ℝ) : Prop := (3/4 ≤ x) ∧ (x ≤ 2)

def B (y : ℝ) (m : ℝ) : Prop := y ≥ 1 - m^2

theorem range_of_m (m : ℝ) :
  (∀ x, in_interval x → B (A x) m) ↔ (m ≤ - (3/4) ∨ m ≥ (3/4)) := 
sorry

end range_of_m_l1775_177565


namespace arithmetic_seq_infinitely_many_squares_l1775_177514

theorem arithmetic_seq_infinitely_many_squares 
  (a d : ℕ) 
  (h : ∃ (n y : ℕ), a + n * d = y^2) : 
  ∃ (m : ℕ), ∀ k : ℕ, ∃ n' y' : ℕ, a + n' * d = y'^2 :=
by sorry

end arithmetic_seq_infinitely_many_squares_l1775_177514


namespace range_of_a_l1775_177500

def is_monotonically_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, (0 < x) → (x < y) → (f x ≤ f y)

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem range_of_a (a : ℝ) : 
  is_monotonically_increasing (f a) a → a ≤ 2 :=
sorry

end range_of_a_l1775_177500


namespace probability_of_BEI3_is_zero_l1775_177516

def isVowelOrDigit (s : Char) : Prop :=
  (s ∈ ['A', 'E', 'I', 'O', 'U']) ∨ (s.isDigit)

def isNonVowel (s : Char) : Prop :=
  s ∈ ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']

def isHexDigit (s : Char) : Prop :=
  s.isDigit ∨ s ∈ ['A', 'B', 'C', 'D', 'E', 'F']

noncomputable def numPossiblePlates : Nat :=
  13 * 21 * 20 * 16

theorem probability_of_BEI3_is_zero :
    ∃ (totalPlates : Nat), 
    (totalPlates = numPossiblePlates) ∧
    ¬(isVowelOrDigit 'B') →
    (1 : ℚ) / (totalPlates : ℚ) = 0 :=
by
  sorry

end probability_of_BEI3_is_zero_l1775_177516


namespace kernel_red_given_popped_l1775_177596

def prob_red_given_popped (P_red : ℚ) (P_green : ℚ) 
                           (P_popped_given_red : ℚ) (P_popped_given_green : ℚ) : ℚ :=
  let P_red_popped := P_red * P_popped_given_red
  let P_green_popped := P_green * P_popped_given_green
  let P_popped := P_red_popped + P_green_popped
  P_red_popped / P_popped

theorem kernel_red_given_popped : prob_red_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end kernel_red_given_popped_l1775_177596


namespace average_last_30_l1775_177546

theorem average_last_30 (avg_first_40 : ℝ) 
  (avg_all_70 : ℝ) 
  (sum_first_40 : ℝ := 40 * avg_first_40)
  (sum_all_70 : ℝ := 70 * avg_all_70) 
  (total_results: ℕ := 70):
  (30 : ℝ) * (40: ℝ) + (30: ℝ) * (40: ℝ) = 70 * 34.285714285714285 :=
by
  sorry

end average_last_30_l1775_177546


namespace circles_radius_difference_l1775_177520

variable (s : ℝ)

theorem circles_radius_difference (h : (π * (2*s)^2) / (π * s^2) = 4) : (2 * s - s) = s :=
by
  sorry

end circles_radius_difference_l1775_177520


namespace average_price_of_remaining_packets_l1775_177511

variables (initial_avg_price : ℕ) (initial_packets : ℕ) (returned_packets : ℕ) (returned_avg_price : ℕ)

def total_initial_cost := initial_avg_price * initial_packets
def total_returned_cost := returned_avg_price * returned_packets
def remaining_packets := initial_packets - returned_packets
def total_remaining_cost := total_initial_cost initial_avg_price initial_packets - total_returned_cost returned_avg_price returned_packets
def remaining_avg_price := total_remaining_cost initial_avg_price initial_packets returned_avg_price returned_packets / remaining_packets initial_packets returned_packets

theorem average_price_of_remaining_packets :
  initial_avg_price = 20 →
  initial_packets = 5 →
  returned_packets = 2 →
  returned_avg_price = 32 →
  remaining_avg_price initial_avg_price initial_packets returned_avg_price returned_packets = 12
:=
by
  intros h1 h2 h3 h4
  rw [remaining_avg_price, total_remaining_cost, total_initial_cost, total_returned_cost]
  norm_num [h1, h2, h3, h4]
  sorry

end average_price_of_remaining_packets_l1775_177511


namespace rectangle_area_l1775_177592

theorem rectangle_area (ABCD : Type*) (small_square : ℕ) (shaded_squares : ℕ) (side_length : ℕ) 
  (shaded_area : ℕ) (width : ℕ) (height : ℕ)
  (H1 : shaded_squares = 3) 
  (H2 : side_length = 2)
  (H3 : shaded_area = side_length * side_length)
  (H4 : width = 6)
  (H5 : height = 4)
  : (width * height) = 24 :=
by
  sorry

end rectangle_area_l1775_177592


namespace union_correct_l1775_177552

variable (x : ℝ)
def A := {x | -2 < x ∧ x < 1}
def B := {x | 0 < x ∧ x < 3}
def unionSet := {x | -2 < x ∧ x < 3}

theorem union_correct : ( {x | -2 < x ∧ x < 1} ∪ {x | 0 < x ∧ x < 3} ) = {x | -2 < x ∧ x < 3} := by
  sorry

end union_correct_l1775_177552


namespace simplify_sqrt_expression_l1775_177528

theorem simplify_sqrt_expression :
  (3 * (Real.sqrt (4 * 3)) - 2 * (Real.sqrt (1 / 3)) +
     Real.sqrt (16 * 3)) / (2 * Real.sqrt 3) = 14 / 3 := by
sorry

end simplify_sqrt_expression_l1775_177528


namespace second_frog_hops_eq_18_l1775_177505

-- Define the given conditions
variables (x : ℕ) (h3 : ℕ)

def second_frog_hops := 2 * h3
def first_frog_hops := 4 * second_frog_hops
def total_hops := h3 + second_frog_hops + first_frog_hops

-- The proof goal
theorem second_frog_hops_eq_18 (H : total_hops = 99) : second_frog_hops = 18 :=
by
  sorry

end second_frog_hops_eq_18_l1775_177505


namespace averagePricePerBook_l1775_177507

-- Define the prices and quantities from the first store
def firstStoreFictionBooks : ℕ := 25
def firstStoreFictionPrice : ℝ := 20
def firstStoreNonFictionBooks : ℕ := 15
def firstStoreNonFictionPrice : ℝ := 30
def firstStoreChildrenBooks : ℕ := 20
def firstStoreChildrenPrice : ℝ := 8

-- Define the prices and quantities from the second store
def secondStoreFictionBooks : ℕ := 10
def secondStoreFictionPrice : ℝ := 18
def secondStoreNonFictionBooks : ℕ := 20
def secondStoreNonFictionPrice : ℝ := 25
def secondStoreChildrenBooks : ℕ := 30
def secondStoreChildrenPrice : ℝ := 5

-- Definition of total books from first and second store
def totalBooks : ℕ :=
  firstStoreFictionBooks + firstStoreNonFictionBooks + firstStoreChildrenBooks +
  secondStoreFictionBooks + secondStoreNonFictionBooks + secondStoreChildrenBooks

-- Definition of the total cost from first and second store
def totalCost : ℝ :=
  (firstStoreFictionBooks * firstStoreFictionPrice) +
  (firstStoreNonFictionBooks * firstStoreNonFictionPrice) +
  (firstStoreChildrenBooks * firstStoreChildrenPrice) +
  (secondStoreFictionBooks * secondStoreFictionPrice) +
  (secondStoreNonFictionBooks * secondStoreNonFictionPrice) +
  (secondStoreChildrenBooks * secondStoreChildrenPrice)

-- Theorem: average price per book
theorem averagePricePerBook : (totalCost / totalBooks : ℝ) = 16.17 := by
  sorry

end averagePricePerBook_l1775_177507


namespace intersection_A_B_l1775_177585

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}
def C : Set ℤ := {x | x ∈ A ∧ (x : ℝ) ∈ B}

theorem intersection_A_B : C = {0, 1, 2} := 
by
  sorry

end intersection_A_B_l1775_177585


namespace equation1_sol_equation2_sol_equation3_sol_l1775_177582

theorem equation1_sol (x : ℝ) : 9 * x^2 - (x - 1)^2 = 0 ↔ (x = -0.5 ∨ x = 0.25) :=
sorry

theorem equation2_sol (x : ℝ) : (x * (x - 3) = 10) ↔ (x = 5 ∨ x = -2) :=
sorry

theorem equation3_sol (x : ℝ) : (x + 3)^2 = 2 * x + 5 ↔ (x = -2) :=
sorry

end equation1_sol_equation2_sol_equation3_sol_l1775_177582


namespace weight_of_square_piece_l1775_177568

open Real

theorem weight_of_square_piece 
  (uniform_density : Prop)
  (side_length_triangle side_length_square : ℝ)
  (weight_triangle : ℝ)
  (ht : side_length_triangle = 6)
  (hs : side_length_square = 6)
  (wt : weight_triangle = 48) :
  ∃ weight_square : ℝ, weight_square = 27.7 :=
by
  sorry

end weight_of_square_piece_l1775_177568


namespace problem_maximum_marks_l1775_177591

theorem problem_maximum_marks (M : ℝ) (h : 0.92 * M = 184) : M = 200 :=
sorry

end problem_maximum_marks_l1775_177591


namespace minimum_value_of_expression_l1775_177541

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) : 6 * x + 1 / x ^ 6 ≥ 7 :=
sorry

end minimum_value_of_expression_l1775_177541


namespace union_of_M_and_N_l1775_177598

def M : Set ℝ := {x | x^2 - 6 * x + 5 = 0}
def N : Set ℝ := {x | x^2 - 5 * x = 0}

theorem union_of_M_and_N : M ∪ N = {0, 1, 5} := by
  sorry

end union_of_M_and_N_l1775_177598


namespace exist_positive_real_x_l1775_177539

theorem exist_positive_real_x (x : ℝ) (hx1 : 0 < x) (hx2 : Nat.floor x * x = 90) : x = 10 := 
sorry

end exist_positive_real_x_l1775_177539


namespace division_of_powers_of_ten_l1775_177515

theorem division_of_powers_of_ten : 10^8 / (2 * 10^6) = 50 := by 
  sorry

end division_of_powers_of_ten_l1775_177515


namespace Adam_total_balls_l1775_177556

def number_of_red_balls := 20
def number_of_blue_balls := 10
def number_of_orange_balls := 5
def number_of_pink_balls := 3 * number_of_orange_balls

def total_number_of_balls := 
  number_of_red_balls + number_of_blue_balls + number_of_pink_balls + number_of_orange_balls

theorem Adam_total_balls : total_number_of_balls = 50 := by
  sorry

end Adam_total_balls_l1775_177556


namespace max_lg_value_l1775_177584

noncomputable def max_lg_product (x y : ℝ) (hx: x > 1) (hy: y > 1) (hxy: Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) : ℝ :=
  4

theorem max_lg_value (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  max_lg_product x y hx hy hxy = 4 := 
by
  unfold max_lg_product
  sorry

end max_lg_value_l1775_177584


namespace calculation_1_calculation_2_calculation_3_calculation_4_l1775_177562

theorem calculation_1 : -3 - (-4) = 1 :=
by sorry

theorem calculation_2 : -1/3 + (-4/3) = -5/3 :=
by sorry

theorem calculation_3 : (-2) * (-3) * (-5) = -30 :=
by sorry

theorem calculation_4 : 15 / 4 * (-1/4) = -15/16 :=
by sorry

end calculation_1_calculation_2_calculation_3_calculation_4_l1775_177562


namespace ice_cubes_total_l1775_177527

theorem ice_cubes_total (initial_cubes made_cubes : ℕ) (h_initial : initial_cubes = 2) (h_made : made_cubes = 7) : initial_cubes + made_cubes = 9 :=
by
  sorry

end ice_cubes_total_l1775_177527


namespace sum_of_roots_eq_14_l1775_177555

-- Define the condition as a hypothesis
def equation (x : ℝ) := (x - 7) ^ 2 = 16

-- Define the statement that needs to be proved
theorem sum_of_roots_eq_14 : 
  (∀ x : ℝ, equation x → x = 11 ∨ x = 3) → 11 + 3 = 14 :=
by 
sorry

end sum_of_roots_eq_14_l1775_177555


namespace increasing_interval_iff_l1775_177599

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 3 * x

def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 → f a x₁ < f a x₂

theorem increasing_interval_iff (a : ℝ) (h : a ≠ 0) :
  is_increasing a ↔ a ∈ Set.Ioo (-(5/4)) 0 ∪ Set.Ioi 0 :=
sorry

end increasing_interval_iff_l1775_177599


namespace exact_time_now_l1775_177531

noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * (t + 4)
noncomputable def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t - 2) + 270
noncomputable def is_opposite (x y : ℝ) : Prop := |x - y| = 180

theorem exact_time_now (t : ℝ) (h1 : 0 ≤ t) (h2 : t < 60)
  (h3 : is_opposite (minute_hand_position t) (hour_hand_position t)) :
  t = 591/50 :=
by
  sorry

end exact_time_now_l1775_177531


namespace solve_inverse_function_l1775_177576

-- Define the given functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 1
def g (x : ℝ) : ℝ := x^4 - x^3 + 4*x^2 + 8*x + 8
def h (x : ℝ) : ℝ := x + 1

-- State the mathematical equivalent proof problem
theorem solve_inverse_function (x : ℝ) :
  f ⁻¹' {g x} = {y | h y = x + 1} ↔
  (x = (3 + Real.sqrt 5) / 2) ∨ (x = (3 - Real.sqrt 5) / 2) :=
sorry -- Proof is omitted

end solve_inverse_function_l1775_177576


namespace largest_pack_size_of_markers_l1775_177569

theorem largest_pack_size_of_markers (markers_John markers_Alex : ℕ) (h_John : markers_John = 36) (h_Alex : markers_Alex = 60) : 
  ∃ (n : ℕ), (∀ (x : ℕ), (∀ (y : ℕ), (x * n = markers_John ∧ y * n = markers_Alex) → n ≤ 12) ∧ (12 * x = markers_John ∨ 12 * y = markers_Alex)) :=
by 
  sorry

end largest_pack_size_of_markers_l1775_177569


namespace solve_equation_l1775_177502

theorem solve_equation (x : ℝ) : 
  (3 * x + 2) * (x + 3) = x + 3 ↔ (x = -3 ∨ x = -1/3) :=
by sorry

end solve_equation_l1775_177502


namespace weekly_rental_fee_percentage_l1775_177533

theorem weekly_rental_fee_percentage
  (camera_value : ℕ)
  (rental_period_weeks : ℕ)
  (friend_percentage : ℚ)
  (john_paid : ℕ)
  (percentage : ℚ)
  (total_rental_fee : ℚ)
  (weekly_rental_fee : ℚ)
  (P : ℚ)
  (camera_value_pos : camera_value = 5000)
  (rental_period_weeks_pos : rental_period_weeks = 4)
  (friend_percentage_pos : friend_percentage = 0.40)
  (john_paid_pos : john_paid = 1200)
  (percentage_pos : percentage = 1 - friend_percentage)
  (total_rental_fee_calc : total_rental_fee = john_paid / percentage)
  (weekly_rental_fee_calc : weekly_rental_fee = total_rental_fee / rental_period_weeks)
  (weekly_rental_fee_equation : weekly_rental_fee = P * camera_value)
  (P_calc : P = weekly_rental_fee / camera_value) :
  P * 100 = 10 := 
by 
  sorry

end weekly_rental_fee_percentage_l1775_177533


namespace min_k_value_l1775_177549

variable (p q r s k : ℕ)

/-- Prove the smallest value of k for which p, q, r, and s are positive integers and 
    satisfy the given equations is 77
-/
theorem min_k_value (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (eq1 : p + 2 * q + 3 * r + 4 * s = k)
  (eq2 : 4 * p = 3 * q)
  (eq3 : 4 * p = 2 * r)
  (eq4 : 4 * p = s) : k = 77 :=
sorry

end min_k_value_l1775_177549


namespace pool_capacity_percentage_l1775_177501

theorem pool_capacity_percentage :
  let width := 60 
  let length := 150 
  let depth := 10 
  let drain_rate := 60 
  let time := 1200 
  let total_volume := width * length * depth
  let water_removed := drain_rate * time
  let capacity_percentage := (water_removed / total_volume : ℚ) * 100
  capacity_percentage = 80 := by
  sorry

end pool_capacity_percentage_l1775_177501


namespace find_angle_B_l1775_177594

noncomputable def angle_B (a b c : ℝ) (B C : ℝ) : Prop :=
b = 2 * Real.sqrt 3 ∧ c = 2 ∧ C = Real.pi / 6 ∧
(Real.sin B = (b * Real.sin C) / c ∧ b > c → (B = Real.pi / 3 ∨ B = 2 * Real.pi / 3))

theorem find_angle_B :
  ∃ (B : ℝ), angle_B 1 (2 * Real.sqrt 3) 2 B (Real.pi / 6) :=
by
  sorry

end find_angle_B_l1775_177594


namespace insulation_cost_l1775_177574

def tank_length : ℕ := 4
def tank_width : ℕ := 5
def tank_height : ℕ := 2
def cost_per_sqft : ℕ := 20

def surface_area (L W H : ℕ) : ℕ := 2 * (L * W + L * H + W * H)
def total_cost (SA cost_per_sqft : ℕ) : ℕ := SA * cost_per_sqft

theorem insulation_cost : 
  total_cost (surface_area tank_length tank_width tank_height) cost_per_sqft = 1520 :=
by
  sorry

end insulation_cost_l1775_177574


namespace total_students_in_class_l1775_177523

theorem total_students_in_class 
  (avg_age_all : ℝ)
  (num_students1 : ℕ) (avg_age1 : ℝ)
  (num_students2 : ℕ) (avg_age2 : ℝ)
  (age_student17 : ℕ)
  (total_students : ℕ) :
  avg_age_all = 17 →
  num_students1 = 5 →
  avg_age1 = 14 →
  num_students2 = 9 →
  avg_age2 = 16 →
  age_student17 = 75 →
  total_students = num_students1 + num_students2 + 1 →
  total_students = 17 :=
by
  intro h_avg_all h_num1 h_avg1 h_num2 h_avg2 h_age17 h_total
  -- Additional proof steps would go here
  sorry

end total_students_in_class_l1775_177523


namespace students_sign_up_ways_l1775_177573

theorem students_sign_up_ways :
  let students := 4
  let choices_per_student := 3
  (choices_per_student ^ students) = 3^4 :=
by
  sorry

end students_sign_up_ways_l1775_177573


namespace least_five_digit_integer_congruent_3_mod_17_l1775_177558

theorem least_five_digit_integer_congruent_3_mod_17 : 
  ∃ n, n ≥ 10000 ∧ n % 17 = 3 ∧ ∀ m, (m ≥ 10000 ∧ m % 17 = 3) → n ≤ m := 
sorry

end least_five_digit_integer_congruent_3_mod_17_l1775_177558


namespace evaluate_expression_l1775_177553

theorem evaluate_expression (x : ℤ) (h : x = 4) : 3 * x + 5 = 17 :=
by
  sorry

end evaluate_expression_l1775_177553


namespace cost_of_first_20_kgs_l1775_177587

theorem cost_of_first_20_kgs 
  (l m n : ℕ) 
  (hl1 : 30 * l +  3 * m = 333) 
  (hl2 : 30 * l +  6 * m = 366) 
  (hl3 : 30 * l + 15 * m = 465) 
  (hl4 : 30 * l + 20 * m = 525) 
  : 20 * l = 200 :=
by
  sorry

end cost_of_first_20_kgs_l1775_177587


namespace cold_brew_cost_l1775_177551

theorem cold_brew_cost :
  let drip_coffee_cost := 2.25
  let espresso_cost := 3.50
  let latte_cost := 4.00
  let vanilla_syrup_cost := 0.50
  let cappuccino_cost := 3.50
  let total_order_cost := 25.00
  let drip_coffee_total := 2 * drip_coffee_cost
  let lattes_total := 2 * latte_cost
  let known_costs := drip_coffee_total + espresso_cost + lattes_total + vanilla_syrup_cost + cappuccino_cost
  total_order_cost - known_costs = 5.00 →
  5.00 / 2 = 2.50 := by sorry

end cold_brew_cost_l1775_177551


namespace negation_exists_to_forall_l1775_177575

theorem negation_exists_to_forall :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by
  sorry

end negation_exists_to_forall_l1775_177575


namespace abc_inequality_l1775_177513

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  a * b + b * c + c * a ≥ a * Real.sqrt (b * c) + b * Real.sqrt (a * c) + c * Real.sqrt (a * b) :=
sorry

end abc_inequality_l1775_177513


namespace find_x_and_C_l1775_177554

def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {1, 2 - x}

theorem find_x_and_C (x : ℝ) (C : Set ℝ) :
  B x ⊆ A x → B (-2) ∪ C = A (-2) → x = -2 ∧ C = {3} :=
by
  sorry

end find_x_and_C_l1775_177554


namespace matthew_crackers_left_l1775_177543

-- Definition of the conditions:
def initial_crackers := 23
def friends := 2
def crackers_eaten_per_friend := 6

-- Calculate the number of crackers Matthew has left:
def crackers_left (total_crackers : ℕ) (num_friends : ℕ) (eaten_per_friend : ℕ) : ℕ :=
  let crackers_given := (total_crackers - total_crackers % num_friends)
  let kept_by_matthew := total_crackers % num_friends
  let remaining_with_friends := (crackers_given / num_friends - eaten_per_friend) * num_friends
  kept_by_matthew + remaining_with_friends
  
-- Theorem to prove:
theorem matthew_crackers_left : crackers_left initial_crackers friends crackers_eaten_per_friend = 11 := by
  sorry

end matthew_crackers_left_l1775_177543


namespace max_handshakes_25_people_l1775_177578

-- Define the number of people attending the conference.
def num_people : ℕ := 25

-- Define the combinatorial formula to calculate the maximum number of handshakes.
def max_handshakes (n : ℕ) : ℕ := n.choose 2

-- State the theorem that we need to prove.
theorem max_handshakes_25_people : max_handshakes num_people = 300 :=
by
  -- Proof will be filled in later
  sorry

end max_handshakes_25_people_l1775_177578


namespace scientific_notation_correct_l1775_177588

-- The given number
def given_number : ℕ := 9000000000

-- The correct answer in scientific notation
def correct_sci_not : ℕ := 9 * (10 ^ 9)

-- The theorem to prove
theorem scientific_notation_correct :
  given_number = correct_sci_not :=
by
  sorry

end scientific_notation_correct_l1775_177588


namespace observations_count_l1775_177508

theorem observations_count (n : ℕ) 
  (original_mean : ℚ) (wrong_value_corrected : ℚ) (corrected_mean : ℚ)
  (h1 : original_mean = 36)
  (h2 : wrong_value_corrected = 1)
  (h3 : corrected_mean = 36.02) :
  n = 50 :=
by
  sorry

end observations_count_l1775_177508


namespace find_c_l1775_177506

def is_midpoint (p1 p2 mid : ℝ × ℝ) : Prop :=
(mid.1 = (p1.1 + p2.1) / 2) ∧ (mid.2 = (p1.2 + p2.2) / 2)

def is_perpendicular_bisector (line : ℝ → ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop := 
∃ mid : ℝ × ℝ, 
is_midpoint p1 p2 mid ∧ line mid.1 mid.2 = 0

theorem find_c (c : ℝ) : 
is_perpendicular_bisector (λ x y => 3 * x - y - c) (2, 4) (6, 8) → c = 6 :=
by
  sorry

end find_c_l1775_177506


namespace interior_edges_sum_l1775_177572

theorem interior_edges_sum (frame_width area outer_length : ℝ) (h1 : frame_width = 2) (h2 : area = 30)
  (h3 : outer_length = 7) : 
  2 * (outer_length - 2 * frame_width) + 2 * ((area / outer_length - 4)) = 7 := 
by
  sorry

end interior_edges_sum_l1775_177572


namespace jar_last_days_l1775_177517

theorem jar_last_days :
  let serving_size := 0.5 -- each serving is 0.5 ounces
  let daily_servings := 3  -- James uses 3 servings every day
  let quart_ounces := 32   -- 1 quart = 32 ounces
  let jar_size := quart_ounces - 2 -- container is 2 ounces less than 1 quart
  let daily_consumption := daily_servings * serving_size
  let number_of_days := jar_size / daily_consumption
  number_of_days = 20 := by
  sorry

end jar_last_days_l1775_177517


namespace joe_spends_50_per_month_l1775_177561

variable (X : ℕ) -- amount Joe spends per month

theorem joe_spends_50_per_month :
  let initial_amount := 240
  let resale_value := 30
  let months := 12
  let final_amount := 0 -- this means he runs out of money
  (initial_amount = months * X - months * resale_value) →
  X = 50 := 
by
  intros
  sorry

end joe_spends_50_per_month_l1775_177561


namespace triangle_area_range_l1775_177536

theorem triangle_area_range (A B C : ℝ) (a b c : ℝ) 
  (h1 : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h2 : a = 3) :
  0 < (1 / 2) * b * c * Real.sin A ∧ 
  (1 / 2) * b * c * Real.sin A ≤ (9 * Real.sqrt 3) / 4 := 
  sorry

end triangle_area_range_l1775_177536


namespace solution_to_exponential_equation_l1775_177526

theorem solution_to_exponential_equation :
  ∃ x : ℕ, (8^12 + 8^12 + 8^12 = 2^x) ∧ x = 38 :=
by
  sorry

end solution_to_exponential_equation_l1775_177526


namespace min_points_to_guarantee_win_l1775_177571

theorem min_points_to_guarantee_win (P Q R S: ℕ) (bonus: ℕ) :
    (P = 6 ∨ P = 4 ∨ P = 2) ∧ (Q = 6 ∨ Q = 4 ∨ Q = 2) ∧ 
    (R = 6 ∨ R = 4 ∨ R = 2) ∧ (S = 6 ∨ S = 4 ∨ S = 2) →
    (bonus = 3 ↔ ((P = 6 ∧ Q = 4 ∧ R = 2) ∨ (P = 6 ∧ Q = 2 ∧ R = 4) ∨ 
                   (P = 4 ∧ Q = 6 ∧ R = 2) ∨ (P = 4 ∧ Q = 2 ∧ R = 6) ∨ 
                   (P = 2 ∧ Q = 6 ∧ R = 4) ∨ (P = 2 ∧ Q = 4 ∧ R = 6))) →
    (P + Q + R + S + bonus ≥ 24) :=
by sorry

end min_points_to_guarantee_win_l1775_177571


namespace missing_fraction_is_two_l1775_177545

theorem missing_fraction_is_two :
  (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + (-5/6) + 2 = 0.8333333333333334 := by
  sorry

end missing_fraction_is_two_l1775_177545


namespace convex_quadrilateral_inequality_l1775_177534

variable (a b c d : ℝ) -- lengths of sides of quadrilateral
variable (S : ℝ) -- Area of the quadrilateral

-- Given condition: a, b, c, d are lengths of the sides of a convex quadrilateral
def is_convex_quadrilateral (a b c d : ℝ) (S : ℝ) : Prop :=
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4

theorem convex_quadrilateral_inequality (a b c d : ℝ) (S : ℝ) 
  (h : is_convex_quadrilateral a b c d S) : 
  S ≤ (a^2 + b^2 + c^2 + d^2) / 4 := 
by
  sorry

end convex_quadrilateral_inequality_l1775_177534


namespace hemisphere_surface_area_ratio_l1775_177559

theorem hemisphere_surface_area_ratio 
  (r : ℝ) (sphere_surface_area : ℝ) (hemisphere_surface_area : ℝ) 
  (eq1 : sphere_surface_area = 4 * π * r^2) 
  (eq2 : hemisphere_surface_area = 3 * π * r^2) : 
  hemisphere_surface_area / sphere_surface_area = 3 / 4 :=
by sorry

end hemisphere_surface_area_ratio_l1775_177559


namespace functional_equation_solution_l1775_177595

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) → (∀ x : ℝ, f x = 0) :=
by
  sorry

end functional_equation_solution_l1775_177595


namespace above_265_is_234_l1775_177522

namespace PyramidArray

-- Definition of the pyramid structure and identifying important properties
def is_number_in_pyramid (n : ℕ) : Prop :=
  ∃ k : ℕ, (k^2 - (k - 1)^2) / 2 ≥ n ∧ (k^2 - (k - 1)^2) / 2 < n + (2 * k - 1)

def row_start (k : ℕ) : ℕ :=
  (k - 1)^2 + 1

def row_end (k : ℕ) : ℕ :=
  k^2

def number_above (n : ℕ) (r : ℕ) : ℕ :=
  row_start r + ((n - row_start (r + 1)) % (2 * (r + 1) - 1))

theorem above_265_is_234 : 
  (number_above 265 16) = 234 := 
sorry

end PyramidArray

end above_265_is_234_l1775_177522


namespace problem_quadratic_has_real_root_l1775_177586

theorem problem_quadratic_has_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end problem_quadratic_has_real_root_l1775_177586


namespace students_watching_l1775_177547

theorem students_watching (b g : ℕ) (h : b + g = 33) : (2 / 3 : ℚ) * b + (2 / 3 : ℚ) * g = 22 := by
  sorry

end students_watching_l1775_177547


namespace sum_arithmetic_sequence_terms_l1775_177550

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 1 - a 0)

theorem sum_arithmetic_sequence_terms (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a) 
  (h₅ : a 5 = 8) :
  a 2 + a 4 + a 5 + a 9 = 32 :=
by
  sorry

end sum_arithmetic_sequence_terms_l1775_177550


namespace combined_flock_size_after_5_years_l1775_177566

noncomputable def initial_flock_size : ℕ := 100
noncomputable def ducks_killed_per_year : ℕ := 20
noncomputable def ducks_born_per_year : ℕ := 30
noncomputable def years_passed : ℕ := 5
noncomputable def other_flock_size : ℕ := 150

theorem combined_flock_size_after_5_years
  (init_size : ℕ := initial_flock_size)
  (killed_per_year : ℕ := ducks_killed_per_year)
  (born_per_year : ℕ := ducks_born_per_year)
  (years : ℕ := years_passed)
  (other_size : ℕ := other_flock_size) :
  init_size + (years * (born_per_year - killed_per_year)) + other_size = 300 := by
  -- The formal proof would go here.
  sorry

end combined_flock_size_after_5_years_l1775_177566


namespace suitable_graph_for_air_composition_is_pie_chart_l1775_177518

/-- The most suitable type of graph to visually represent the percentage 
of each component in the air is a pie chart, based on the given conditions. -/
theorem suitable_graph_for_air_composition_is_pie_chart 
  (bar_graph : Prop)
  (line_graph : Prop)
  (pie_chart : Prop)
  (histogram : Prop)
  (H1 : bar_graph → comparing_quantities)
  (H2 : line_graph → display_data_over_time)
  (H3 : pie_chart → show_proportions_of_whole)
  (H4 : histogram → show_distribution_of_dataset) 
  : suitable_graph_to_represent_percentage = pie_chart :=
sorry

end suitable_graph_for_air_composition_is_pie_chart_l1775_177518


namespace smallest_three_digit_plus_one_multiple_l1775_177544

theorem smallest_three_digit_plus_one_multiple (x : ℕ) : 
  (421 = x) →
  (x ≥ 100 ∧ x < 1000) ∧ 
  ∃ k : ℕ, x = k * Nat.lcm (Nat.lcm 3 4) * Nat.lcm 5 7 + 1 :=
by
  sorry

end smallest_three_digit_plus_one_multiple_l1775_177544


namespace monotonic_intervals_max_min_values_on_interval_l1775_177510

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end monotonic_intervals_max_min_values_on_interval_l1775_177510


namespace drum_y_capacity_filled_l1775_177590

-- Definitions of the initial conditions
def capacity_of_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def capacity_of_drum_Y (C : ℝ) (two_c_y : ℝ) := two_c_y = 2 * C
def oil_in_drum_X (C : ℝ) (half_full_x : ℝ) := half_full_x = 1 / 2 * C
def oil_in_drum_Y (C : ℝ) (four_fifth_c_y : ℝ) := four_fifth_c_y = 4 / 5 * C

-- Theorem to prove the capacity filled in drum Y after pouring all oil from X
theorem drum_y_capacity_filled {C : ℝ} (hx : 1/2 * C = 1 / 2 * C) (hy : 2 * C = 2 * C) (ox : 1/2 * C = 1 / 2 * C) (oy : 4/5 * 2 * C = 4 / 5 * C) :
  ( (1/2 * C + 4/5 * C) / (2 * C) ) = 13 / 20 :=
by
  sorry

end drum_y_capacity_filled_l1775_177590


namespace prime_sq_mod_12_l1775_177532

theorem prime_sq_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) : (p * p) % 12 = 1 := by
  sorry

end prime_sq_mod_12_l1775_177532


namespace bob_age_is_eleven_l1775_177540

/-- 
Susan, Arthur, Tom, and Bob are siblings. Arthur is 2 years older than Susan, 
Tom is 3 years younger than Bob. Susan is 15 years old, 
and the total age of all four family members is 51 years. 
This theorem states that Bob is 11 years old.
-/

theorem bob_age_is_eleven
  (S A T B : ℕ)
  (h1 : A = S + 2)
  (h2 : T = B - 3)
  (h3 : S = 15)
  (h4 : S + A + T + B = 51) : 
  B = 11 :=
  sorry

end bob_age_is_eleven_l1775_177540


namespace correct_option_is_A_l1775_177504

-- Define the conditions
def chromosome_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 2
  else if phase = "metaphase" then 2
  else if phase = "anaphase" then if is_meiosis then 2 else 4
  else if phase = "telophase" then if is_meiosis then 1 else 2
  else 0

def dna_counts (phase : String) (is_meiosis : Bool) : Nat :=
  if phase = "prophase" then 4
  else if phase = "metaphase" then 4
  else if phase = "anaphase" then 4
  else if phase = "telophase" then 2
  else 0

def chromosome_behavior (phase : String) (is_meiosis : Bool) : String :=
  if is_meiosis && phase = "prophase" then "synapsis"
  else if is_meiosis && phase = "metaphase" then "tetrad formation"
  else if is_meiosis && phase = "anaphase" then "separation"
  else if is_meiosis && phase = "telophase" then "recombination"
  else "no special behavior"

-- Problem statement in terms of a Lean theorem
theorem correct_option_is_A :
  ∀ (phase : String),
  (chromosome_counts phase false = chromosome_counts phase true ∧
   chromosome_behavior phase false ≠ chromosome_behavior phase true ∧
   dna_counts phase false ≠ dna_counts phase true) →
  "A" = "A" :=
by 
  intro phase 
  simp only [imp_self]
  sorry

end correct_option_is_A_l1775_177504


namespace problem1_solution_set_problem2_range_of_a_l1775_177580

-- Define the functions
def f (x a : ℝ) : ℝ := |2 * x - 1| + |2 * x + a|
def g (x : ℝ) : ℝ := x + 3

-- Problem 1: Proving the solution set when a = -2
theorem problem1_solution_set (x : ℝ) : (f x (-2) < g x) ↔ (0 < x ∧ x < 2) :=
  sorry

-- Problem 2: Proving the range of a
theorem problem2_range_of_a (a : ℝ) : 
  (a > -1) ∧ (∀ x, (x ∈ Set.Icc (-a/2) (1/2) → f x a ≤ g x)) ↔ a ∈ Set.Ioo (-1) (4/3) ∨ a = 4/3 :=
  sorry

end problem1_solution_set_problem2_range_of_a_l1775_177580


namespace bus_ticket_problem_l1775_177564

variables (x y : ℕ)

theorem bus_ticket_problem (h1 : x + y = 99) (h2 : 2 * x + 3 * y = 280) : x = 17 ∧ y = 82 :=
by
  sorry

end bus_ticket_problem_l1775_177564


namespace factorize_expression_l1775_177529

-- Variables used in the expression
variables (m n : ℤ)

-- The expression to be factored
def expr := 4 * m^3 * n - 16 * m * n^3

-- The desired factorized form of the expression
def factored := 4 * m * n * (m + 2 * n) * (m - 2 * n)

-- The proof problem statement
theorem factorize_expression : expr m n = factored m n :=
by sorry

end factorize_expression_l1775_177529


namespace cost_of_each_candy_bar_l1775_177577

theorem cost_of_each_candy_bar
  (p_chips : ℝ)
  (total_cost : ℝ)
  (num_students : ℕ)
  (num_chips_per_student : ℕ)
  (num_candy_bars_per_student : ℕ)
  (h1 : p_chips = 0.50)
  (h2 : total_cost = 15)
  (h3 : num_students = 5)
  (h4 : num_chips_per_student = 2)
  (h5 : num_candy_bars_per_student = 1) :
  ∃ C : ℝ, C = 2 := 
by 
  sorry

end cost_of_each_candy_bar_l1775_177577


namespace meaningful_fraction_l1775_177519

theorem meaningful_fraction (x : ℝ) : (x ≠ 5) ↔ (∃ y : ℝ, y = 1 / (x - 5)) :=
by
  sorry

end meaningful_fraction_l1775_177519


namespace min_square_sum_l1775_177579

theorem min_square_sum (a b m n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 15 * a + 16 * b = m * m) (h4 : 16 * a - 15 * b = n * n) : 481 ≤ min (m * m) (n * n) :=
sorry

end min_square_sum_l1775_177579


namespace radius_circle_B_l1775_177524

theorem radius_circle_B (rA rB rD : ℝ) 
  (hA : rA = 2) (hD : rD = 2 * rA) (h_tangent : (rA + rB) ^ 2 = rD ^ 2) : 
  rB = 2 :=
by
  sorry

end radius_circle_B_l1775_177524


namespace division_number_l1775_177589

-- Definitions from conditions
def D : Nat := 3
def Q : Nat := 4
def R : Nat := 3

-- Theorem statement
theorem division_number : ∃ N : Nat, N = D * Q + R ∧ N = 15 :=
by
  sorry

end division_number_l1775_177589


namespace sugar_for_recipe_l1775_177509

theorem sugar_for_recipe (sugar_frosting sugar_cake : ℝ) (h1 : sugar_frosting = 0.6) (h2 : sugar_cake = 0.2) :
  sugar_frosting + sugar_cake = 0.8 :=
by
  sorry

end sugar_for_recipe_l1775_177509


namespace xy_sum_of_squares_l1775_177535

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 22) : x^2 + y^2 = 404 := by
  sorry

end xy_sum_of_squares_l1775_177535


namespace part1_zero_of_f_part2_a_range_l1775_177503

-- Define the given function f
def f (x a b : ℝ) : ℝ := (x - a) * |x| + b

-- Define the problem statement for Part 1
theorem part1_zero_of_f :
  ∀ (x : ℝ),
    f x 2 3 = 0 ↔ x = -1 := 
by
  sorry

-- Define the problem statement for Part 2
theorem part2_a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f x a (-2) < 0) ↔ a > -1 :=
by
  sorry

end part1_zero_of_f_part2_a_range_l1775_177503


namespace total_feet_is_correct_l1775_177593

-- definitions according to conditions
def number_of_heads := 46
def number_of_hens := 24
def number_of_cows := number_of_heads - number_of_hens
def hen_feet := 2
def cow_feet := 4
def total_hen_feet := number_of_hens * hen_feet
def total_cow_feet := number_of_cows * cow_feet
def total_feet := total_hen_feet + total_cow_feet

-- proof statement with sorry
theorem total_feet_is_correct : total_feet = 136 :=
by
  sorry

end total_feet_is_correct_l1775_177593


namespace correct_system_of_equations_l1775_177537

theorem correct_system_of_equations (x y : ℕ) (h1 : x + y = 145) (h2 : 10 * x + 12 * y = 1580) :
  (x + y = 145) ∧ (10 * x + 12 * y = 1580) :=
by
  sorry

end correct_system_of_equations_l1775_177537


namespace round_2741836_to_nearest_integer_l1775_177521

theorem round_2741836_to_nearest_integer :
  (2741836.4928375).round = 2741836 := 
by
  -- Explanation that 0.4928375 < 0.5 leading to rounding down
  sorry

end round_2741836_to_nearest_integer_l1775_177521


namespace x_plus_2y_equals_5_l1775_177548

theorem x_plus_2y_equals_5 (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : (x + y) / 3 = 1.222222222222222) : x + 2 * y = 5 := 
by sorry

end x_plus_2y_equals_5_l1775_177548


namespace remainder_of_polynomial_l1775_177538

-- Define the polynomial
def P (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

-- State the theorem
theorem remainder_of_polynomial (x : ℝ) : P 3 = 50 := sorry

end remainder_of_polynomial_l1775_177538


namespace sum_of_areas_of_sixteen_disks_l1775_177525

theorem sum_of_areas_of_sixteen_disks :
  let r := 1 - (2:ℝ).sqrt
  let area_one_disk := r^2 * Real.pi
  let total_area := 16 * area_one_disk
  total_area = Real.pi * (48 - 32 * (2:ℝ).sqrt) :=
by
  sorry

end sum_of_areas_of_sixteen_disks_l1775_177525


namespace linear_function_passing_origin_l1775_177583

theorem linear_function_passing_origin (m : ℝ) :
  (∃ (y x : ℝ), y = -2 * x + (m - 5) ∧ y = 0 ∧ x = 0) → m = 5 :=
by
  sorry

end linear_function_passing_origin_l1775_177583


namespace probability_of_pulling_blue_ball_l1775_177567

def given_conditions (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ) :=
  total_balls = 15 ∧ initial_blue_balls = 7 ∧ blue_balls_removed = 3

theorem probability_of_pulling_blue_ball
  (total_balls : ℕ) (initial_blue_balls : ℕ) (blue_balls_removed : ℕ)
  (hc : given_conditions total_balls initial_blue_balls blue_balls_removed) :
  ((initial_blue_balls - blue_balls_removed) / (total_balls - blue_balls_removed) : ℚ) = 1 / 3 :=
by
  sorry

end probability_of_pulling_blue_ball_l1775_177567


namespace counting_numbers_dividing_56_greater_than_2_l1775_177557

theorem counting_numbers_dividing_56_greater_than_2 :
  (∃ (A : Finset ℕ), A = {n ∈ (Finset.range 57) | n > 2 ∧ 56 % n = 0} ∧ A.card = 5) :=
sorry

end counting_numbers_dividing_56_greater_than_2_l1775_177557


namespace Sasha_earnings_proof_l1775_177530

def Monday_hours : ℕ := 90  -- 1.5 hours * 60 minutes/hour
def Tuesday_minutes : ℕ := 75  -- 1 hour * 60 minutes/hour + 15 minutes
def Wednesday_minutes : ℕ := 115  -- 11:10 AM - 9:15 AM
def Thursday_minutes : ℕ := 45

def total_minutes_worked : ℕ := Monday_hours + Tuesday_minutes + Wednesday_minutes + Thursday_minutes

def hourly_rate : ℚ := 4.50
def total_hours : ℚ := total_minutes_worked / 60

def weekly_earnings : ℚ := total_hours * hourly_rate

theorem Sasha_earnings_proof : weekly_earnings = 24 := by
  sorry

end Sasha_earnings_proof_l1775_177530


namespace functional_expression_and_range_l1775_177560

-- We define the main problem conditions and prove the required statements based on those conditions
theorem functional_expression_and_range (x y : ℝ) (h1 : ∃ k : ℝ, (y + 2) = k * (4 - x) ∧ k ≠ 0)
                                        (h2 : x = 3 → y = 1) :
                                        (y = -3 * x + 10) ∧ ( -2 < y ∧ y < 1 → 3 < x ∧ x < 4) :=
by
  sorry

end functional_expression_and_range_l1775_177560


namespace petya_vasya_common_result_l1775_177570

theorem petya_vasya_common_result (a b : ℝ) (h1 : b ≠ 0) (h2 : a/b = (a + b)/(2 * a)) (h3 : a/b ≠ 1) : 
  a/b = -1/2 :=
by 
  sorry

end petya_vasya_common_result_l1775_177570


namespace arcsin_double_angle_identity_l1775_177581

open Real

theorem arcsin_double_angle_identity (x θ : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (h₃ : arcsin x = θ) (h₄ : -π / 2 ≤ θ) (h₅ : θ ≤ -π / 4) :
    arcsin (2 * x * sqrt (1 - x^2)) = -(π + 2 * θ) := by
  sorry

end arcsin_double_angle_identity_l1775_177581


namespace tank_capacity_l1775_177512

theorem tank_capacity (C : ℕ) 
  (leak_rate : C / 4 = C / 4)               -- Condition: Leak rate is C/4 litres per hour
  (inlet_rate : 6 * 60 = 360)                -- Condition: Inlet rate is 360 litres per hour
  (net_emptying_rate : C / 12 = (360 - C / 4))  -- Condition: Net emptying rate for 12 hours
  : C = 1080 := 
by 
  -- Conditions imply that C = 1080 
  sorry

end tank_capacity_l1775_177512


namespace calculate_exponent_product_l1775_177542

theorem calculate_exponent_product :
  (2^0.5) * (2^0.3) * (2^0.2) * (2^0.1) * (2^0.9) = 4 :=
by
  sorry

end calculate_exponent_product_l1775_177542


namespace children_got_off_l1775_177597

theorem children_got_off {x : ℕ} 
  (initial_children : ℕ := 22)
  (children_got_on : ℕ := 40)
  (children_left : ℕ := 2)
  (equation : initial_children + children_got_on - x = children_left) :
  x = 60 :=
sorry

end children_got_off_l1775_177597
