import Mathlib

namespace NUMINAMATH_GPT_value_of_a7_l1354_135414

theorem value_of_a7 (a : ℕ → ℤ) (h1 : a 1 = 0) (h2 : ∀ n, a (n + 2) - a n = 2) : a 7 = 6 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_value_of_a7_l1354_135414


namespace NUMINAMATH_GPT_crayons_at_the_end_of_thursday_l1354_135458

-- Definitions for each day's changes
def monday_crayons : ℕ := 7
def tuesday_crayons (initial : ℕ) := initial + 3
def wednesday_crayons (initial : ℕ) := initial - 5 + 4
def thursday_crayons (initial : ℕ) := initial + 6 - 2

-- Proof statement to show the number of crayons at the end of Thursday
theorem crayons_at_the_end_of_thursday : thursday_crayons (wednesday_crayons (tuesday_crayons monday_crayons)) = 13 :=
by
  sorry

end NUMINAMATH_GPT_crayons_at_the_end_of_thursday_l1354_135458


namespace NUMINAMATH_GPT_perfect_square_expression_l1354_135466

theorem perfect_square_expression (p : ℝ) (h : p = 0.28) : 
  (12.86 * 12.86 + 12.86 * p + 0.14 * 0.14) = (12.86 + 0.14) * (12.86 + 0.14) :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l1354_135466


namespace NUMINAMATH_GPT_min_area_after_fold_l1354_135494

theorem min_area_after_fold (A : ℝ) (h_A : A = 1) (c : ℝ) (h_c : 0 ≤ c ∧ c ≤ 1) : 
  ∃ (m : ℝ), m = min_area ∧ m = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_area_after_fold_l1354_135494


namespace NUMINAMATH_GPT_simple_interest_rate_l1354_135490

variables (P R T SI : ℝ)

theorem simple_interest_rate :
  T = 10 →
  SI = (2 / 5) * P →
  SI = (P * R * T) / 100 →
  R = 4 :=
by
  intros hT hSI hFormula
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1354_135490


namespace NUMINAMATH_GPT_intersection_property_l1354_135475

theorem intersection_property (x_0 : ℝ) (h1 : x_0 > 0) (h2 : -x_0 = Real.tan x_0) :
  (x_0^2 + 1) * (Real.cos (2 * x_0) + 1) = 2 :=
sorry

end NUMINAMATH_GPT_intersection_property_l1354_135475


namespace NUMINAMATH_GPT_number_of_solutions_l1354_135484

theorem number_of_solutions :
  ∃ (solutions : Finset (ℝ × ℝ)), 
  (∀ (x y : ℝ), (x, y) ∈ solutions ↔ (x + 2 * y = 2 ∧ abs (abs x - 2 * abs y) = 1)) ∧ 
  solutions.card = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l1354_135484


namespace NUMINAMATH_GPT_probability_A_not_losing_l1354_135477

theorem probability_A_not_losing (P_draw : ℚ) (P_win_A : ℚ) (h1 : P_draw = 1/2) (h2 : P_win_A = 1/3) : 
  P_draw + P_win_A = 5/6 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_probability_A_not_losing_l1354_135477


namespace NUMINAMATH_GPT_train_pass_platform_time_l1354_135472

theorem train_pass_platform_time :
  ∀ (length_train length_platform speed_time_cross_tree speed_train pass_time : ℕ), 
  length_train = 1200 →
  length_platform = 300 →
  speed_time_cross_tree = 120 →
  speed_train = length_train / speed_time_cross_tree →
  pass_time = (length_train + length_platform) / speed_train →
  pass_time = 150 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_pass_platform_time_l1354_135472


namespace NUMINAMATH_GPT_rate_of_fuel_consumption_l1354_135437

-- Define the necessary conditions
def total_fuel : ℝ := 100
def total_hours : ℝ := 175

-- Prove the rate of fuel consumption per hour
theorem rate_of_fuel_consumption : (total_fuel / total_hours) = 100 / 175 := 
by 
  sorry

end NUMINAMATH_GPT_rate_of_fuel_consumption_l1354_135437


namespace NUMINAMATH_GPT_problem1_problem2_l1354_135423

theorem problem1 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : 0 < t ∧ t < 1) :
  x^t - (x-1)^t < (x-2)^t - (x-3)^t :=
sorry

theorem problem2 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : t > 1) :
  x^t - (x-1)^t > (x-2)^t - (x-3)^t :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1354_135423


namespace NUMINAMATH_GPT_compute_ab_l1354_135471

namespace MathProof

variable {a b : ℝ}

theorem compute_ab (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := 
by
  sorry

end MathProof

end NUMINAMATH_GPT_compute_ab_l1354_135471


namespace NUMINAMATH_GPT_bus_rent_proof_l1354_135495

theorem bus_rent_proof (r1 r2 : ℝ) (r1_rent_eq : r1 + 2 * r2 = 2800) (r2_mult : r2 = 1.25 * r1) :
  r1 = 800 ∧ r2 = 1000 := 
by
  sorry

end NUMINAMATH_GPT_bus_rent_proof_l1354_135495


namespace NUMINAMATH_GPT_minimal_surface_area_l1354_135459

-- Definitions based on the conditions in the problem.
def unit_cube (a b c : ℕ) : Prop := a * b * c = 25
def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + a * c + b * c)

-- The proof problem statement.
theorem minimal_surface_area : ∃ (a b c : ℕ), unit_cube a b c ∧ surface_area a b c = 54 := 
sorry

end NUMINAMATH_GPT_minimal_surface_area_l1354_135459


namespace NUMINAMATH_GPT_fewer_seats_on_right_side_l1354_135427

-- Definitions based on the conditions
def left_seats := 15
def seats_per_seat := 3
def back_seat_capacity := 8
def total_capacity := 89

-- Statement to prove the problem
theorem fewer_seats_on_right_side : left_seats - (total_capacity - back_seat_capacity - (left_seats * seats_per_seat)) / seats_per_seat = 3 := 
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_fewer_seats_on_right_side_l1354_135427


namespace NUMINAMATH_GPT_multiples_of_3_ending_number_l1354_135476

theorem multiples_of_3_ending_number :
  ∃ n, ∃ k, k = 93 ∧ (∀ m, 81 + 3 * m = n → 0 ≤ m ∧ m < k) ∧ n = 357 := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_3_ending_number_l1354_135476


namespace NUMINAMATH_GPT_geometric_sequence_seventh_term_l1354_135410

-- Define the initial conditions
def geometric_sequence_first_term := 3
def geometric_sequence_fifth_term (r : ℝ) := geometric_sequence_first_term * r^4 = 243

-- Statement for the seventh term problem
theorem geometric_sequence_seventh_term (r : ℝ) 
  (h1 : geometric_sequence_first_term = 3) 
  (h2 : geometric_sequence_fifth_term r) : 
  3 * r^6 = 2187 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_seventh_term_l1354_135410


namespace NUMINAMATH_GPT_problem_f_g_comp_sum_l1354_135449

-- Define the functions
def f (x : ℚ) : ℚ := (4 * x^2 + 6 * x + 9) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

-- Define the statement we want to prove
theorem problem_f_g_comp_sum (x : ℚ) (h : x = 2) : f (g x) + g (f x) = 36 / 5 := by
  sorry

end NUMINAMATH_GPT_problem_f_g_comp_sum_l1354_135449


namespace NUMINAMATH_GPT_mutually_exclusive_B_C_l1354_135425

-- Define the events A, B, C
def event_A (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∨ x 2 = false)
def event_B (x y : ℕ → Bool) : Prop := x 1 = false ∧ x 2 = false
def event_C (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∧ x 2 = false)

-- Prove that event B and event C are mutually exclusive
theorem mutually_exclusive_B_C (x y : ℕ → Bool) :
  (event_B x y ∧ event_C x y) ↔ false := sorry

end NUMINAMATH_GPT_mutually_exclusive_B_C_l1354_135425


namespace NUMINAMATH_GPT_initial_candy_bobby_l1354_135489

-- Definitions given conditions
def initial_candy (x : ℕ) : Prop :=
  (x + 42 = 70)

-- Theorem statement
theorem initial_candy_bobby : ∃ x : ℕ, initial_candy x ∧ x = 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_candy_bobby_l1354_135489


namespace NUMINAMATH_GPT_water_parts_in_solution_l1354_135436

def lemonade_syrup_parts : ℝ := 7
def target_percentage : ℝ := 0.30
def adjusted_parts : ℝ := 2.1428571428571423

-- Original equation: L = 0.30 * (L + W)
-- Substitute L = 7 for the particular instance.
-- Therefore, 7 = 0.30 * (7 + W)

theorem water_parts_in_solution (W : ℝ) : 
  (7 = 0.30 * (7 + W)) → 
  W = 16.333333333333332 := 
by
  sorry

end NUMINAMATH_GPT_water_parts_in_solution_l1354_135436


namespace NUMINAMATH_GPT_shorter_piece_length_l1354_135445

theorem shorter_piece_length : ∃ (x : ℕ), (x + (x + 2) = 30) ∧ x = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_shorter_piece_length_l1354_135445


namespace NUMINAMATH_GPT_max_ab_bc_ca_l1354_135426

theorem max_ab_bc_ca (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 3) :
  ab + bc + ca ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_ab_bc_ca_l1354_135426


namespace NUMINAMATH_GPT_find_x_l1354_135462

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1354_135462


namespace NUMINAMATH_GPT_arithmetic_sequence_l1354_135478

theorem arithmetic_sequence (S : ℕ → ℕ) (h : ∀ n, S n = 3 * n * n) :
  (∃ a d : ℕ, ∀ n : ℕ, S n - S (n - 1) = a + (n - 1) * d) ∧
  (∀ n, S n - S (n - 1) = 6 * n - 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_l1354_135478


namespace NUMINAMATH_GPT_greatest_difference_l1354_135461

theorem greatest_difference (n m : ℕ) (hn : 1023 = 17 * n + m) (hn_pos : 0 < n) (hm_pos : 0 < m) : n - m = 57 :=
sorry

end NUMINAMATH_GPT_greatest_difference_l1354_135461


namespace NUMINAMATH_GPT_sum_gt_product_iff_l1354_135416

theorem sum_gt_product_iff (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m + n > m * n ↔ m = 1 ∨ n = 1 :=
sorry

end NUMINAMATH_GPT_sum_gt_product_iff_l1354_135416


namespace NUMINAMATH_GPT_find_roots_l1354_135440

theorem find_roots (x : ℝ) :
  5 * x^4 - 28 * x^3 + 46 * x^2 - 28 * x + 5 = 0 → x = 3.2 ∨ x = 0.8 ∨ x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_roots_l1354_135440


namespace NUMINAMATH_GPT_revenue_and_empty_seats_l1354_135420

-- Define seating and ticket prices
def seats_A : ℕ := 90
def seats_B : ℕ := 70
def seats_C : ℕ := 50
def VIP_seats : ℕ := 10

def ticket_A : ℕ := 15
def ticket_B : ℕ := 10
def ticket_C : ℕ := 5
def VIP_ticket : ℕ := 25

-- Define discounts
def discount : ℤ := 20

-- Define actual occupancy
def adults_A : ℕ := 35
def children_A : ℕ := 15
def adults_B : ℕ := 20
def seniors_B : ℕ := 5
def adults_C : ℕ := 10
def veterans_C : ℕ := 5
def VIP_occupied : ℕ := 10

-- Concession sales
def hot_dogs_sold : ℕ := 50
def hot_dog_price : ℕ := 4
def soft_drinks_sold : ℕ := 75
def soft_drink_price : ℕ := 2

-- Define the total revenue and empty seats calculation
theorem revenue_and_empty_seats :
  let revenue_from_tickets := (adults_A * ticket_A + children_A * ticket_A * (100 - discount) / 100 +
                               adults_B * ticket_B + seniors_B * ticket_B * (100 - discount) / 100 +
                               adults_C * ticket_C + veterans_C * ticket_C * (100 - discount) / 100 +
                               VIP_occupied * VIP_ticket)
  let revenue_from_concessions := (hot_dogs_sold * hot_dog_price + soft_drinks_sold * soft_drink_price)
  let total_revenue := revenue_from_tickets + revenue_from_concessions
  let empty_seats_A := seats_A - (adults_A + children_A)
  let empty_seats_B := seats_B - (adults_B + seniors_B)
  let empty_seats_C := seats_C - (adults_C + veterans_C)
  let empty_VIP_seats := VIP_seats - VIP_occupied
  total_revenue = 1615 ∧ empty_seats_A = 40 ∧ empty_seats_B = 45 ∧ empty_seats_C = 35 ∧ empty_VIP_seats = 0 := by
  sorry

end NUMINAMATH_GPT_revenue_and_empty_seats_l1354_135420


namespace NUMINAMATH_GPT_geom_seq_ratio_l1354_135442

variable {a_1 r : ℚ}
variable {S : ℕ → ℚ}

-- The sum of the first n terms of a geometric sequence
def geom_sum (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * (1 - r^n) / (1 - r)

-- Given conditions
axiom Sn_def : ∀ n, S n = geom_sum a_1 r n
axiom condition : S 10 / S 5 = 1 / 2

-- Theorem to prove
theorem geom_seq_ratio (h : r ≠ 1) : S 15 / S 5 = 3 / 4 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_geom_seq_ratio_l1354_135442


namespace NUMINAMATH_GPT_proof1_proof2a_proof2b_l1354_135493

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ)

-- Given conditions for Question 1
def condition1 := (a = 3 * Real.cos C ∧ b = 1)

-- Proof statement for Question 1
theorem proof1 : condition1 a b C → Real.tan C = 2 * Real.tan B :=
by sorry

-- Given conditions for Question 2a
def condition2a := (S = 1 / 2 * a * b * Real.sin C ∧ S = 1 / 2 * 3 * Real.cos C * 1 * Real.sin C)

-- Proof statement for Question 2a
theorem proof2a : condition2a a b C S → Real.cos (2 * B) = 3 / 5 :=
by sorry

-- Given conditions for Question 2b
def condition2b := (c = Real.sqrt 10 / 2)

-- Proof statement for Question 2b
theorem proof2b : condition1 a b C → condition2b c → Real.cos (2 * B) = 3 / 5 :=
by sorry

end NUMINAMATH_GPT_proof1_proof2a_proof2b_l1354_135493


namespace NUMINAMATH_GPT_rahul_share_l1354_135463

theorem rahul_share :
  let total_payment := 370
  let bonus := 30
  let remaining_payment := total_payment - bonus
  let rahul_work_per_day := 1 / 3
  let rajesh_work_per_day := 1 / 2
  let ramesh_work_per_day := 1 / 4
  
  let total_work_per_day := rahul_work_per_day + rajesh_work_per_day + ramesh_work_per_day
  let rahul_share_of_work := rahul_work_per_day / total_work_per_day
  let rahul_payment := rahul_share_of_work * remaining_payment

  rahul_payment = 80 :=
by {
  sorry
}

end NUMINAMATH_GPT_rahul_share_l1354_135463


namespace NUMINAMATH_GPT_moles_HBr_formed_l1354_135401

theorem moles_HBr_formed 
    (moles_CH4 : ℝ) (moles_Br2 : ℝ) (reaction : ℝ) : 
    moles_CH4 = 1 ∧ moles_Br2 = 1 → reaction = 1 :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_moles_HBr_formed_l1354_135401


namespace NUMINAMATH_GPT_union_M_N_eq_M_l1354_135451

-- Define set M
def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Define set N
def N : Set ℝ := { y | ∃ x : ℝ, y = Real.log (x - 1) }

-- Statement to prove that M ∪ N = M
theorem union_M_N_eq_M : M ∪ N = M := by
  sorry

end NUMINAMATH_GPT_union_M_N_eq_M_l1354_135451


namespace NUMINAMATH_GPT_area_of_field_l1354_135464

theorem area_of_field (L W A : ℕ) (h₁ : L = 20) (h₂ : L + 2 * W = 80) : A = 600 :=
by
  sorry

end NUMINAMATH_GPT_area_of_field_l1354_135464


namespace NUMINAMATH_GPT_kim_hard_correct_l1354_135452

-- Definitions
def points_per_easy := 2
def points_per_average := 3
def points_per_hard := 5
def easy_correct := 6
def average_correct := 2
def total_points := 38

-- Kim's correct answers in the hard round is 4
theorem kim_hard_correct : (total_points - (easy_correct * points_per_easy + average_correct * points_per_average)) / points_per_hard = 4 :=
by
  sorry

end NUMINAMATH_GPT_kim_hard_correct_l1354_135452


namespace NUMINAMATH_GPT_trig_identity_l1354_135411

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  (Real.sin x + 2 * Real.cos x = 1 / 2) ∨ (Real.sin x + 2 * Real.cos x = 83 / 29) := sorry

end NUMINAMATH_GPT_trig_identity_l1354_135411


namespace NUMINAMATH_GPT_four_digit_numbers_gt_3000_l1354_135487

theorem four_digit_numbers_gt_3000 (d1 d2 d3 d4 : ℕ) (h_digits : (d1, d2, d3, d4) = (2, 0, 5, 5)) (h_distinct_4digit : (d1 * 1000 + d2 * 100 + d3 * 10 + d4) > 3000) :
  ∃ count, count = 3 := sorry

end NUMINAMATH_GPT_four_digit_numbers_gt_3000_l1354_135487


namespace NUMINAMATH_GPT_total_ticket_cost_is_14_l1354_135467

-- Definitions of the ticket costs
def ticket_cost_hat : ℕ := 2
def ticket_cost_stuffed_animal : ℕ := 10
def ticket_cost_yoyo : ℕ := 2

-- Definition of the total ticket cost
def total_ticket_cost : ℕ := ticket_cost_hat + ticket_cost_stuffed_animal + ticket_cost_yoyo

-- Theorem stating the total ticket cost is 14
theorem total_ticket_cost_is_14 : total_ticket_cost = 14 := by
  -- Proof would go here, but sorry is used to skip it
  sorry

end NUMINAMATH_GPT_total_ticket_cost_is_14_l1354_135467


namespace NUMINAMATH_GPT_negative_square_inequality_l1354_135439

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_negative_square_inequality_l1354_135439


namespace NUMINAMATH_GPT_smallest_b_perfect_fourth_power_l1354_135424

theorem smallest_b_perfect_fourth_power:
  ∃ b : ℕ, (∀ n : ℕ, 5 * n = (7 * b^2 + 7 * b + 7) → ∃ x : ℕ, n = x^4) 
  ∧ b = 41 :=
sorry

end NUMINAMATH_GPT_smallest_b_perfect_fourth_power_l1354_135424


namespace NUMINAMATH_GPT_john_gym_hours_l1354_135496

theorem john_gym_hours :
  (2 * (1 + 1/3)) + (2 * (1 + 1/2)) + (1.5 + 3/4) = 7.92 :=
by
  sorry

end NUMINAMATH_GPT_john_gym_hours_l1354_135496


namespace NUMINAMATH_GPT_y_gt_1_l1354_135406

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end NUMINAMATH_GPT_y_gt_1_l1354_135406


namespace NUMINAMATH_GPT_abc_not_less_than_two_l1354_135444

theorem abc_not_less_than_two (a b c : ℝ) (h : a + b + c = 6) : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

end NUMINAMATH_GPT_abc_not_less_than_two_l1354_135444


namespace NUMINAMATH_GPT_train_passing_through_tunnel_l1354_135447

theorem train_passing_through_tunnel :
  let train_length : ℝ := 300
  let tunnel_length : ℝ := 1200
  let speed_in_kmh : ℝ := 54
  let speed_in_mps : ℝ := speed_in_kmh * (1000 / 3600)
  let total_distance : ℝ := train_length + tunnel_length
  let time : ℝ := total_distance / speed_in_mps
  time = 100 :=
by
  sorry

end NUMINAMATH_GPT_train_passing_through_tunnel_l1354_135447


namespace NUMINAMATH_GPT_mindy_tax_rate_proof_l1354_135404

noncomputable def mindy_tax_rate (M r : ℝ) : Prop :=
  let Mork_tax := 0.10 * M
  let Mindy_income := 3 * M
  let Mindy_tax := r * Mindy_income
  let Combined_tax_rate := 0.175
  let Combined_tax := Combined_tax_rate * (M + Mindy_income)
  Mork_tax + Mindy_tax = Combined_tax

theorem mindy_tax_rate_proof (M r : ℝ) 
  (h1 : Mork_tax_rate = 0.10) 
  (h2 : mindy_income = 3 * M) 
  (h3 : combined_tax_rate = 0.175) : 
  r = 0.20 := 
sorry

end NUMINAMATH_GPT_mindy_tax_rate_proof_l1354_135404


namespace NUMINAMATH_GPT_abs_a1_plus_abs_a2_to_abs_a6_l1354_135417

theorem abs_a1_plus_abs_a2_to_abs_a6 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ)
  (h : (2 - x) ^ 6 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6) :
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 :=
sorry

end NUMINAMATH_GPT_abs_a1_plus_abs_a2_to_abs_a6_l1354_135417


namespace NUMINAMATH_GPT_B_div_A_75_l1354_135457

noncomputable def find_ratio (A B : ℝ) (x : ℝ) :=
  (A / (x + 3) + B / (x * (x - 9)) = (x^2 - 3*x + 15) / (x * (x + 3) * (x - 9)))

theorem B_div_A_75 :
  ∀ (A B : ℝ), (∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → find_ratio A B x) → 
  B/A = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_B_div_A_75_l1354_135457


namespace NUMINAMATH_GPT_find_length_of_sheet_l1354_135497

noncomputable section

-- Axioms regarding the conditions
def width_of_sheet : ℝ := 36       -- The width of the metallic sheet is 36 meters
def side_of_square : ℝ := 7        -- The side length of the square cut off from each corner is 7 meters
def volume_of_box : ℝ := 5236      -- The volume of the resulting box is 5236 cubic meters

-- Define the length of the metallic sheet as L
def length_of_sheet (L : ℝ) : Prop :=
  let new_length := L - 2 * side_of_square
  let new_width := width_of_sheet - 2 * side_of_square
  let height := side_of_square
  volume_of_box = new_length * new_width * height

-- The condition to prove
theorem find_length_of_sheet : ∃ L : ℝ, length_of_sheet L ∧ L = 48 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_sheet_l1354_135497


namespace NUMINAMATH_GPT_find_triples_l1354_135435

theorem find_triples (a b c : ℝ) 
  (h1 : a = (b + c) ^ 2) 
  (h2 : b = (a + c) ^ 2) 
  (h3 : c = (a + b) ^ 2) : 
  (a = 0 ∧ b = 0 ∧ c = 0) 
  ∨ 
  (a = 1/4 ∧ b = 1/4 ∧ c = 1/4) :=
  sorry

end NUMINAMATH_GPT_find_triples_l1354_135435


namespace NUMINAMATH_GPT_initial_men_employed_l1354_135446

theorem initial_men_employed (M : ℕ) 
  (h1 : ∀ m d, m * d = 2 * 10)
  (h2 : ∀ m t, (m + 30) * t = 10 * 30) : 
  M = 75 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_employed_l1354_135446


namespace NUMINAMATH_GPT_cos_theta_sub_pi_div_3_value_l1354_135433

open Real

noncomputable def problem_statement (θ : ℝ) : Prop :=
  sin (3 * π - θ) = (sqrt 5 / 2) * sin (π / 2 + θ)

theorem cos_theta_sub_pi_div_3_value (θ : ℝ) (hθ : problem_statement θ) :
  cos (θ - π / 3) = 1 / 3 + sqrt 15 / 6 ∨ cos (θ - π / 3) = - (1 / 3 + sqrt 15 / 6) :=
sorry

end NUMINAMATH_GPT_cos_theta_sub_pi_div_3_value_l1354_135433


namespace NUMINAMATH_GPT_harriet_travel_time_l1354_135429

theorem harriet_travel_time (D : ℝ) (h : (D / 90 + D / 160 = 5)) : (D / 90) * 60 = 192 := 
by sorry

end NUMINAMATH_GPT_harriet_travel_time_l1354_135429


namespace NUMINAMATH_GPT_non_adjacent_divisibility_l1354_135450

theorem non_adjacent_divisibility (a : Fin 7 → ℕ) (h : ∀ i, a i ∣ a ((i + 1) % 7) ∨ a ((i + 1) % 7) ∣ a i) :
  ∃ i j : Fin 7, i ≠ j ∧ (¬(i + 1)%7 = j) ∧ (a i ∣ a j ∨ a j ∣ a i) :=
by
  sorry

end NUMINAMATH_GPT_non_adjacent_divisibility_l1354_135450


namespace NUMINAMATH_GPT_ratio_of_X_to_Y_l1354_135491

theorem ratio_of_X_to_Y (total_respondents : ℕ) (preferred_X : ℕ)
    (h_total : total_respondents = 250)
    (h_X : preferred_X = 200) :
    preferred_X / (total_respondents - preferred_X) = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_of_X_to_Y_l1354_135491


namespace NUMINAMATH_GPT_find_principal_amount_l1354_135412

theorem find_principal_amount 
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)
  (hA : A = 3087) (hr : r = 0.05) (hn : n = 1) (ht : t = 2)
  (hcomp : A = P * (1 + r / n)^(n * t)) :
  P = 2800 := 
  by sorry

end NUMINAMATH_GPT_find_principal_amount_l1354_135412


namespace NUMINAMATH_GPT_combination_recurrence_l1354_135438

variable {n r : ℕ}
variable (C : ℕ → ℕ → ℕ)

theorem combination_recurrence (hn : n > 0) (hr : r > 0) (h : n > r)
  (h2 : ∀ (k : ℕ), k = 1 → C 2 1 = C 1 1 + C 1) 
  (h3 : ∀ (k : ℕ), k = 1 → C 3 1 = C 2 1 + C 2) 
  (h4 : ∀ (k : ℕ), k = 2 → C 3 2 = C 2 2 + C 2 1)
  (h5 : ∀ (k : ℕ), k = 1 → C 4 1 = C 3 1 + C 3) 
  (h6 : ∀ (k : ℕ), k = 2 → C 4 2 = C 3 2 + C 3 1)
  (h7 : ∀ (k : ℕ), k = 3 → C 4 3 = C 3 3 + C 3 2)
  (h8 : ∀ n r : ℕ, (n > r) → C n r = C (n-1) r + C (n-1) (r-1)) :
  C n r = C (n-1) r + C (n-1) (r-1) :=
sorry

end NUMINAMATH_GPT_combination_recurrence_l1354_135438


namespace NUMINAMATH_GPT_water_tank_capacity_l1354_135468

theorem water_tank_capacity (C : ℝ) :
  (0.40 * C - 0.25 * C = 36) → C = 240 :=
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l1354_135468


namespace NUMINAMATH_GPT_min_value_expr_l1354_135482

theorem min_value_expr (x y z w : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) (hw : -2 < w ∧ w < 2) :
  2 ≤ (1 / ((1 - x) * (1 - y) * (1 - z) * (1 - w / 2)) + 1 / ((1 + x) * (1 + y) * (1 + z) * (1 + w / 2))) :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1354_135482


namespace NUMINAMATH_GPT_factorization_of_x_squared_minus_4_l1354_135419

theorem factorization_of_x_squared_minus_4 (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_x_squared_minus_4_l1354_135419


namespace NUMINAMATH_GPT_option_C_is_correct_l1354_135479

-- Define the conditions as propositions
def condition_A := |-2| = 2
def condition_B := (-1)^2 = 1
def condition_C := -7 + 3 = -4
def condition_D := 6 / (-2) = -3

-- The statement that option C is correct
theorem option_C_is_correct : condition_C := by
  sorry

end NUMINAMATH_GPT_option_C_is_correct_l1354_135479


namespace NUMINAMATH_GPT_div120_l1354_135408

theorem div120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end NUMINAMATH_GPT_div120_l1354_135408


namespace NUMINAMATH_GPT_range_of_b_l1354_135474

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) → (5 < b ∧ b < 7) :=
sorry

end NUMINAMATH_GPT_range_of_b_l1354_135474


namespace NUMINAMATH_GPT_victory_circle_count_l1354_135402

   -- Define the conditions
   def num_runners : ℕ := 8
   def num_medals : ℕ := 5
   def medals : List String := ["gold", "silver", "bronze", "titanium", "copper"]
   
   -- Define the scenarios
   def scenario1 : ℕ := 2 * 6 -- 2! * 3!
   def scenario2 : ℕ := 6 * 2 -- 3! * 2!
   def scenario3 : ℕ := 2 * 2 * 1 -- 2! * 2! * 1!

   -- Calculate the total number of victory circles
   def total_victory_circles : ℕ := scenario1 + scenario2 + scenario3

   theorem victory_circle_count : total_victory_circles = 28 := by
     sorry
   
end NUMINAMATH_GPT_victory_circle_count_l1354_135402


namespace NUMINAMATH_GPT_earnings_difference_l1354_135430

theorem earnings_difference :
  let lower_tasks := 400
  let lower_rate := 0.25
  let higher_tasks := 5
  let higher_rate := 2.00
  let lower_earnings := lower_tasks * lower_rate
  let higher_earnings := higher_tasks * higher_rate
  lower_earnings - higher_earnings = 90 := by
  sorry

end NUMINAMATH_GPT_earnings_difference_l1354_135430


namespace NUMINAMATH_GPT_odometer_trip_l1354_135453

variables (d e f : ℕ) (x : ℕ)

-- Define the conditions
def start_odometer (d e f : ℕ) : ℕ := 100 * d + 10 * e + f
def end_odometer (d e f : ℕ) : ℕ := 100 * f + 10 * e + d
def distance_travelled (x : ℕ) : ℕ := 65 * x
def valid_trip (d e f x : ℕ) : Prop := 
  d ≥ 1 ∧ d + e + f ≤ 9 ∧ 
  end_odometer d e f - start_odometer d e f = distance_travelled x

-- The final statement to prove
theorem odometer_trip (h : valid_trip d e f x) : d^2 + e^2 + f^2 = 41 := 
sorry

end NUMINAMATH_GPT_odometer_trip_l1354_135453


namespace NUMINAMATH_GPT_binary_subtraction_l1354_135415

theorem binary_subtraction : ∀ (x y : ℕ), x = 0b11011 → y = 0b101 → x - y = 0b10110 :=
by
  sorry

end NUMINAMATH_GPT_binary_subtraction_l1354_135415


namespace NUMINAMATH_GPT_area_of_quadrilateral_EFGM_l1354_135418

noncomputable def area_ABMJ := 1.8 -- Given area of quadrilateral ABMJ

-- Conditions described in a more abstract fashion:
def is_perpendicular (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of each adjacent pairs being perpendicular
  sorry

def is_congruent (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of all sides except AL and GF being congruent
  sorry

def are_segments_intersecting (B G E L : Point) (M : Point) : Prop :=
  -- Description of segments BG and EL intersecting at point M
  sorry

def area_ratio (tri1 tri2 : Finset Triangle) : ℝ :=
  -- Function that returns the ratio of areas covered by the triangles
  sorry

theorem area_of_quadrilateral_EFGM 
  (A B C D E F G H I J K L M : Point)
  (h1 : is_perpendicular A B C D E F G H I J K L)
  (h2 : is_congruent A B C D E F G H I J K L)
  (h3 : are_segments_intersecting B G E L M)
  : 7 / 3 * area_ABMJ = 4.2 :=
by
  -- Proof of the theorem that area EFGM == 4.2 using the conditions
  sorry

end NUMINAMATH_GPT_area_of_quadrilateral_EFGM_l1354_135418


namespace NUMINAMATH_GPT_partiallyFilledBoxes_l1354_135470

/-- Define the number of cards Joe collected -/
def numPokemonCards : Nat := 65
def numMagicCards : Nat := 55
def numYuGiOhCards : Nat := 40

/-- Define the number of cards each full box can hold -/
def pokemonBoxCapacity : Nat := 8
def magicBoxCapacity : Nat := 10
def yuGiOhBoxCapacity : Nat := 12

/-- Define the partially filled boxes for each type -/
def pokemonPartialBox : Nat := numPokemonCards % pokemonBoxCapacity
def magicPartialBox : Nat := numMagicCards % magicBoxCapacity
def yuGiOhPartialBox : Nat := numYuGiOhCards % yuGiOhBoxCapacity

/-- Theorem to prove number of cards in each partially filled box -/
theorem partiallyFilledBoxes :
  pokemonPartialBox = 1 ∧
  magicPartialBox = 5 ∧
  yuGiOhPartialBox = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_partiallyFilledBoxes_l1354_135470


namespace NUMINAMATH_GPT_cos_A_zero_l1354_135413

theorem cos_A_zero (A : ℝ) (h : Real.tan A + (1 / Real.tan A) + 2 / (Real.cos A) = 4) : Real.cos A = 0 :=
sorry

end NUMINAMATH_GPT_cos_A_zero_l1354_135413


namespace NUMINAMATH_GPT_sin_pi_over_six_l1354_135422

theorem sin_pi_over_six : Real.sin (Real.pi / 6) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_sin_pi_over_six_l1354_135422


namespace NUMINAMATH_GPT_FGH_supermarkets_total_l1354_135456

theorem FGH_supermarkets_total 
  (us_supermarkets : ℕ)
  (ca_supermarkets : ℕ)
  (h1 : us_supermarkets = 41)
  (h2 : us_supermarkets = ca_supermarkets + 22) :
  us_supermarkets + ca_supermarkets = 60 :=
by
  sorry

end NUMINAMATH_GPT_FGH_supermarkets_total_l1354_135456


namespace NUMINAMATH_GPT_incorrect_calculation_l1354_135473

theorem incorrect_calculation (a : ℝ) : (2 * a) ^ 3 ≠ 6 * a ^ 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_calculation_l1354_135473


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1354_135485

theorem sum_of_two_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 9) (h2 : (1 / x) = 4 * (1 / y)) : x + y = 15 / 2 :=
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1354_135485


namespace NUMINAMATH_GPT_completing_the_square_l1354_135448

theorem completing_the_square (x : ℝ) (h : x^2 - 6 * x + 7 = 0) : (x - 3)^2 - 2 = 0 := 
by sorry

end NUMINAMATH_GPT_completing_the_square_l1354_135448


namespace NUMINAMATH_GPT_ratio_of_segments_l1354_135434

theorem ratio_of_segments (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_segments_l1354_135434


namespace NUMINAMATH_GPT_problem1_problem2_l1354_135465

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + b|
noncomputable def g (x a b : ℝ) : ℝ := -x^2 - a*x - b

-- Problem 1: Prove that a + b = 3
theorem problem1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : ∀ x, f x a b ≤ 3) : a + b = 3 := 
sorry

-- Problem 2: Prove that 1/2 < a < 3
theorem problem2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 3) 
  (h₃ : ∀ x, x ≥ a → g x a b < f x a b) : 1/2 < a ∧ a < 3 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1354_135465


namespace NUMINAMATH_GPT_total_dogs_at_center_l1354_135498

structure PawsitiveTrainingCenter :=
  (sit : Nat)
  (stay : Nat)
  (fetch : Nat)
  (roll_over : Nat)
  (sit_stay : Nat)
  (sit_fetch : Nat)
  (sit_roll_over : Nat)
  (stay_fetch : Nat)
  (stay_roll_over : Nat)
  (fetch_roll_over : Nat)
  (sit_stay_fetch : Nat)
  (sit_stay_roll_over : Nat)
  (sit_fetch_roll_over : Nat)
  (stay_fetch_roll_over : Nat)
  (all_four : Nat)
  (none : Nat)

def PawsitiveTrainingCenter.total_dogs (p : PawsitiveTrainingCenter) : Nat :=
  p.sit + p.stay + p.fetch + p.roll_over
  - p.sit_stay - p.sit_fetch - p.sit_roll_over - p.stay_fetch - p.stay_roll_over - p.fetch_roll_over
  + p.sit_stay_fetch + p.sit_stay_roll_over + p.sit_fetch_roll_over + p.stay_fetch_roll_over
  - p.all_four + p.none

theorem total_dogs_at_center (p : PawsitiveTrainingCenter) (h : 
  p.sit = 60 ∧
  p.stay = 35 ∧
  p.fetch = 45 ∧
  p.roll_over = 40 ∧
  p.sit_stay = 20 ∧
  p.sit_fetch = 15 ∧
  p.sit_roll_over = 10 ∧
  p.stay_fetch = 5 ∧
  p.stay_roll_over = 8 ∧
  p.fetch_roll_over = 6 ∧
  p.sit_stay_fetch = 4 ∧
  p.sit_stay_roll_over = 3 ∧
  p.sit_fetch_roll_over = 2 ∧
  p.stay_fetch_roll_over = 1 ∧
  p.all_four = 2 ∧
  p.none = 12
) : PawsitiveTrainingCenter.total_dogs p = 135 := by
  sorry

end NUMINAMATH_GPT_total_dogs_at_center_l1354_135498


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l1354_135400

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → c = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l1354_135400


namespace NUMINAMATH_GPT_solve_y_l1354_135499

theorem solve_y : ∀ y : ℚ, (9 * y^2 + 8 * y - 2 = 0) ∧ (27 * y^2 + 62 * y - 8 = 0) → y = 1 / 9 :=
by
  intro y h
  cases h
  sorry

end NUMINAMATH_GPT_solve_y_l1354_135499


namespace NUMINAMATH_GPT_integer_solution_unique_l1354_135480

theorem integer_solution_unique (x y z : ℤ) : x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end NUMINAMATH_GPT_integer_solution_unique_l1354_135480


namespace NUMINAMATH_GPT_factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l1354_135405

-- Problem 1: Prove equivalence for factorizing -2a^2 + 4a.
theorem factorize_problem1 (a : ℝ) : -2 * a^2 + 4 * a = -2 * a * (a - 2) := 
by sorry

-- Problem 2: Prove equivalence for factorizing 4x^3 y - 9xy^3.
theorem factorize_problem2 (x y : ℝ) : 4 * x^3 * y - 9 * x * y^3 = x * y * (2 * x + 3 * y) * (2 * x - 3 * y) := 
by sorry

-- Problem 3: Prove equivalence for factorizing 4x^2 - 12x + 9.
theorem factorize_problem3 (x : ℝ) : 4 * x^2 - 12 * x + 9 = (2 * x - 3)^2 := 
by sorry

-- Problem 4: Prove equivalence for factorizing (a+b)^2 - 6(a+b) + 9.
theorem factorize_problem4 (a b : ℝ) : (a + b)^2 - 6 * (a + b) + 9 = (a + b - 3)^2 := 
by sorry

end NUMINAMATH_GPT_factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l1354_135405


namespace NUMINAMATH_GPT_find_x_for_equation_l1354_135460

def f (x : ℝ) : ℝ := 2 * x - 3

theorem find_x_for_equation : (2 * f x - 21 = f (x - 4)) ↔ (x = 8) :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_equation_l1354_135460


namespace NUMINAMATH_GPT_triangle_has_angle_45_l1354_135454

theorem triangle_has_angle_45
  (A B C : ℝ)
  (h1 : A + B + C = 180)
  (h2 : B + C = 3 * A) :
  A = 45 :=
by
  sorry

end NUMINAMATH_GPT_triangle_has_angle_45_l1354_135454


namespace NUMINAMATH_GPT_money_total_l1354_135488

theorem money_total (s j m : ℝ) (h1 : 3 * s = 80) (h2 : j / 2 = 70) (h3 : 2.5 * m = 100) :
  s + j + m = 206.67 :=
sorry

end NUMINAMATH_GPT_money_total_l1354_135488


namespace NUMINAMATH_GPT_cubes_not_touching_foil_l1354_135409

-- Define the variables for length, width, height, and total cubes
variables (l w h : ℕ)

-- Conditions extracted from the problem
def width_is_twice_length : Prop := w = 2 * l
def width_is_twice_height : Prop := w = 2 * h
def foil_covered_prism_width : Prop := w + 2 = 10

-- The proof statement
theorem cubes_not_touching_foil (l w h : ℕ) 
  (h1 : width_is_twice_length l w) 
  (h2 : width_is_twice_height w h) 
  (h3 : foil_covered_prism_width w) : 
  l * w * h = 128 := 
by sorry

end NUMINAMATH_GPT_cubes_not_touching_foil_l1354_135409


namespace NUMINAMATH_GPT_range_of_a_l1354_135421

noncomputable def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0

noncomputable def q (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0

def sufficient_but_not_necessary_condition (a : ℝ) : Prop :=
  ∀ x, p x → q x a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : sufficient_but_not_necessary_condition a) :
  9 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l1354_135421


namespace NUMINAMATH_GPT_stickers_total_l1354_135407

def karl_stickers : ℕ := 25
def ryan_stickers : ℕ := karl_stickers + 20
def ben_stickers : ℕ := ryan_stickers - 10
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem stickers_total : total_stickers = 105 := by
  sorry

end NUMINAMATH_GPT_stickers_total_l1354_135407


namespace NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l1354_135481

def unique_letters_in_mathematics : List Char := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']

def number_of_unique_letters_in_word : ℕ := unique_letters_in_mathematics.length

def total_letters_in_alphabet : ℕ := 26

theorem probability_of_picking_letter_in_mathematics :
  (number_of_unique_letters_in_word : ℚ) / total_letters_in_alphabet = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_picking_letter_in_mathematics_l1354_135481


namespace NUMINAMATH_GPT_part1_part2_l1354_135428

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part1 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a ≥ 0)) ↔ (0 < a ∧ a ≤ 2) := sorry

theorem part2 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (x - 1) * f x a ≥ 0) ↔ (0 < a ∧ a ≤ 2) := sorry

end NUMINAMATH_GPT_part1_part2_l1354_135428


namespace NUMINAMATH_GPT_seven_in_M_l1354_135469

-- Define the universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- Define the set M complement with respect to U
def compl_U_M : Set ℕ := {1, 3, 5}

-- Define the set M
def M : Set ℕ := U \ compl_U_M

-- Prove that 7 is an element of M
theorem seven_in_M : 7 ∈ M :=
by {
  sorry
}

end NUMINAMATH_GPT_seven_in_M_l1354_135469


namespace NUMINAMATH_GPT_interest_rate_proof_l1354_135492
noncomputable def interest_rate_B (P : ℝ) (rA : ℝ) (t : ℝ) (gain_B : ℝ) : ℝ := 
  (P * rA * t + gain_B) / (P * t)

theorem interest_rate_proof
  (P : ℝ := 3500)
  (rA : ℝ := 0.10)
  (t : ℝ := 3)
  (gain_B : ℝ := 210) :
  interest_rate_B P rA t gain_B = 0.12 :=
sorry

end NUMINAMATH_GPT_interest_rate_proof_l1354_135492


namespace NUMINAMATH_GPT_problem1_l1354_135486

theorem problem1 (k : ℝ) : (∃ x : ℝ, k*x^2 + (2*k + 1)*x + (k - 1) = 0) → k ≥ -1/8 := 
sorry

end NUMINAMATH_GPT_problem1_l1354_135486


namespace NUMINAMATH_GPT_solve_for_y_l1354_135441

noncomputable def find_angle_y : Prop :=
  let AB_CD_are_straight_lines : Prop := True
  let angle_AXB : ℕ := 70
  let angle_BXD : ℕ := 40
  let angle_CYX : ℕ := 100
  let angle_YXZ := 180 - angle_AXB - angle_BXD
  let angle_XYZ := 180 - angle_CYX
  let y := 180 - angle_YXZ - angle_XYZ
  y = 30

theorem solve_for_y : find_angle_y :=
by
  trivial

end NUMINAMATH_GPT_solve_for_y_l1354_135441


namespace NUMINAMATH_GPT_find_x_l1354_135455

theorem find_x (x : ℕ) (h1 : x ≥ 10) (h2 : x > 8) : x = 9 := by
  sorry

end NUMINAMATH_GPT_find_x_l1354_135455


namespace NUMINAMATH_GPT_min_empty_squares_eq_nine_l1354_135443

-- Definition of the problem conditions
def chessboard_size : ℕ := 9
def total_squares : ℕ := chessboard_size * chessboard_size
def number_of_white_squares : ℕ := 4 * chessboard_size
def number_of_black_squares : ℕ := 5 * chessboard_size
def minimum_number_of_empty_squares : ℕ := number_of_black_squares - number_of_white_squares

-- Theorem to prove minimum number of empty squares
theorem min_empty_squares_eq_nine :
  minimum_number_of_empty_squares = 9 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_min_empty_squares_eq_nine_l1354_135443


namespace NUMINAMATH_GPT_quadratic_range_and_value_l1354_135431

theorem quadratic_range_and_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0)) →
  k ≤ 5 / 4 ∧ (∀ x1 x2 : ℝ, (x1^2 + (2 * k - 1) * x1 + k^2 - 1 = 0) ∧
  (x2^2 + (2 * k - 1) * x2 + k^2 - 1 = 0) ∧ (x1^2 + x2^2 = 16 + x1 * x2)) → k = -2 :=
by sorry

end NUMINAMATH_GPT_quadratic_range_and_value_l1354_135431


namespace NUMINAMATH_GPT_percentage_of_water_in_mixture_is_17_14_l1354_135432

def Liquid_A_water_percentage : ℝ := 0.10
def Liquid_B_water_percentage : ℝ := 0.15
def Liquid_C_water_percentage : ℝ := 0.25
def Liquid_D_water_percentage : ℝ := 0.35

def parts_A : ℝ := 3
def parts_B : ℝ := 2
def parts_C : ℝ := 1
def parts_D : ℝ := 1

def part_unit : ℝ := 100

noncomputable def total_units : ℝ := 
  parts_A * part_unit + parts_B * part_unit + parts_C * part_unit + parts_D * part_unit

noncomputable def total_water_units : ℝ :=
  parts_A * part_unit * Liquid_A_water_percentage +
  parts_B * part_unit * Liquid_B_water_percentage +
  parts_C * part_unit * Liquid_C_water_percentage +
  parts_D * part_unit * Liquid_D_water_percentage

noncomputable def percentage_water : ℝ := (total_water_units / total_units) * 100

theorem percentage_of_water_in_mixture_is_17_14 :
  percentage_water = 17.14 := sorry

end NUMINAMATH_GPT_percentage_of_water_in_mixture_is_17_14_l1354_135432


namespace NUMINAMATH_GPT_sector_arc_length_120_degrees_radius_3_l1354_135483

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem sector_arc_length_120_degrees_radius_3 :
  arc_length 120 3 = 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_arc_length_120_degrees_radius_3_l1354_135483


namespace NUMINAMATH_GPT_inverse_r_l1354_135403

def p (x: ℝ) : ℝ := 4 * x + 5
def q (x: ℝ) : ℝ := 3 * x - 4
def r (x: ℝ) : ℝ := p (q x)

theorem inverse_r (x : ℝ) : r⁻¹ x = (x + 11) / 12 :=
sorry

end NUMINAMATH_GPT_inverse_r_l1354_135403
