import Mathlib

namespace NUMINAMATH_GPT_total_lemonade_poured_l1286_128685

-- Define the amounts of lemonade served during each intermission.
def first_intermission : ℝ := 0.25
def second_intermission : ℝ := 0.42
def third_intermission : ℝ := 0.25

-- State the theorem that the total amount of lemonade poured is 0.92 pitchers.
theorem total_lemonade_poured : first_intermission + second_intermission + third_intermission = 0.92 :=
by
  -- Placeholders to skip the proof.
  sorry

end NUMINAMATH_GPT_total_lemonade_poured_l1286_128685


namespace NUMINAMATH_GPT_wickets_before_last_match_l1286_128603

theorem wickets_before_last_match (R W : ℕ) 
  (initial_average : ℝ) (runs_last_match wickets_last_match : ℕ) (average_decrease : ℝ)
  (h_initial_avg : initial_average = 12.4)
  (h_last_match_runs : runs_last_match = 26)
  (h_last_match_wickets : wickets_last_match = 5)
  (h_avg_decrease : average_decrease = 0.4)
  (h_initial_runs_eq : R = initial_average * W)
  (h_new_average : (R + runs_last_match) / (W + wickets_last_match) = initial_average - average_decrease) :
  W = 85 :=
by
  sorry

end NUMINAMATH_GPT_wickets_before_last_match_l1286_128603


namespace NUMINAMATH_GPT_harmonic_mean_ordered_pairs_l1286_128618

theorem harmonic_mean_ordered_pairs :
  ∃ n : ℕ, n = 23 ∧ ∀ (a b : ℕ), 
    0 < a ∧ 0 < b ∧ a < b ∧ (2 * a * b = 2 ^ 24 * (a + b)) → n = 23 :=
by sorry

end NUMINAMATH_GPT_harmonic_mean_ordered_pairs_l1286_128618


namespace NUMINAMATH_GPT_fifth_pile_magazines_l1286_128637

theorem fifth_pile_magazines :
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  fifth_pile = 13 :=
by
  let first_pile := 3
  let second_pile := first_pile + 1
  let third_pile := second_pile + 2
  let fourth_pile := third_pile + 3
  let fifth_pile := fourth_pile + (3 + 1)
  show fifth_pile = 13
  sorry

end NUMINAMATH_GPT_fifth_pile_magazines_l1286_128637


namespace NUMINAMATH_GPT_additional_carpet_needed_l1286_128636

-- Define the given conditions as part of the hypothesis:
def carpetArea : ℕ := 18
def roomLength : ℕ := 4
def roomWidth : ℕ := 20

-- The theorem we want to prove:
theorem additional_carpet_needed : (roomLength * roomWidth - carpetArea) = 62 := by
  sorry

end NUMINAMATH_GPT_additional_carpet_needed_l1286_128636


namespace NUMINAMATH_GPT_common_difference_is_3_l1286_128658

theorem common_difference_is_3 (a : ℕ → ℤ) (d : ℤ) (h1 : a 2 = 4) (h2 : 1 + a 3 = 5 + d)
  (h3 : a 6 = 4 + 4 * d) (h4 : 4 + a 10 = 8 + 8 * d) :
  (5 + d) * (8 + 8 * d) = (4 + 4 * d) ^ 2 → d = 3 := 
by
  intros hg
  sorry

end NUMINAMATH_GPT_common_difference_is_3_l1286_128658


namespace NUMINAMATH_GPT_percentage_wax_left_eq_10_l1286_128674

def total_amount_wax : ℕ := 
  let wax20 := 5 * 20
  let wax5 := 5 * 5
  let wax1 := 25 * 1
  wax20 + wax5 + wax1

def wax_used_for_new_candles : ℕ := 
  3 * 5

def percentage_wax_used (total_wax : ℕ) (wax_used : ℕ) : ℕ := 
  (wax_used * 100) / total_wax

theorem percentage_wax_left_eq_10 :
  percentage_wax_used total_amount_wax wax_used_for_new_candles = 10 :=
by
  sorry

end NUMINAMATH_GPT_percentage_wax_left_eq_10_l1286_128674


namespace NUMINAMATH_GPT_opera_house_earnings_l1286_128628

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end NUMINAMATH_GPT_opera_house_earnings_l1286_128628


namespace NUMINAMATH_GPT_possible_values_of_m_l1286_128608

theorem possible_values_of_m
  (m : ℕ)
  (h1 : ∃ (m' : ℕ), m = m' ∧ 0 < m)            -- m is a positive integer
  (h2 : 2 * (m - 1) + 3 * (m + 2) > 4 * (m - 5))    -- AB + AC > BC
  (h3 : 2 * (m - 1) + 4 * (m + 5) > 3 * (m + 2))    -- AB + BC > AC
  (h4 : 3 * (m + 2) + 4 * (m + 5) > 2 * (m - 1))    -- AC + BC > AB
  (h5 : 3 * (m + 2) > 2 * (m - 1))                  -- AC > AB
  (h6 : 4 * (m + 5) > 3 * (m + 2))                  -- BC > AC
  : m ≥ 7 := 
sorry

end NUMINAMATH_GPT_possible_values_of_m_l1286_128608


namespace NUMINAMATH_GPT_least_four_digit_divisible_by_15_25_40_75_is_1200_l1286_128663

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

def divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

def divisible_by_40 (n : ℕ) : Prop :=
  n % 40 = 0

def divisible_by_75 (n : ℕ) : Prop :=
  n % 75 = 0

theorem least_four_digit_divisible_by_15_25_40_75_is_1200 :
  ∃ n : ℕ, is_four_digit n ∧ divisible_by_15 n ∧ divisible_by_25 n ∧ divisible_by_40 n ∧ divisible_by_75 n ∧
  (∀ m : ℕ, is_four_digit m ∧ divisible_by_15 m ∧ divisible_by_25 m ∧ divisible_by_40 m ∧ divisible_by_75 m → n ≤ m) ∧
  n = 1200 := 
sorry

end NUMINAMATH_GPT_least_four_digit_divisible_by_15_25_40_75_is_1200_l1286_128663


namespace NUMINAMATH_GPT_solve_system_l1286_128621

def x : ℚ := 2.7 / 13
def y : ℚ := 1.0769

theorem solve_system :
  (∃ (x' y' : ℚ), 4 * x' - 3 * y' = -2.4 ∧ 5 * x' + 6 * y' = 7.5) ↔
  (x = 2.7 / 13 ∧ y = 1.0769) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1286_128621


namespace NUMINAMATH_GPT_rowing_time_from_A_to_B_and_back_l1286_128613

-- Define the problem parameters and conditions
def rowing_speed_still_water : ℝ := 5
def distance_AB : ℝ := 12
def stream_speed : ℝ := 1

-- Define the problem to prove
theorem rowing_time_from_A_to_B_and_back :
  let downstream_speed := rowing_speed_still_water + stream_speed
  let upstream_speed := rowing_speed_still_water - stream_speed
  let time_downstream := distance_AB / downstream_speed
  let time_upstream := distance_AB / upstream_speed
  let total_time := time_downstream + time_upstream
  total_time = 5 :=
by
  sorry

end NUMINAMATH_GPT_rowing_time_from_A_to_B_and_back_l1286_128613


namespace NUMINAMATH_GPT_largest_n_fact_product_of_four_consecutive_integers_l1286_128665

theorem largest_n_fact_product_of_four_consecutive_integers :
  ∀ (n : ℕ), (∃ x : ℕ, n.factorial = x * (x + 1) * (x + 2) * (x + 3)) → n ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_fact_product_of_four_consecutive_integers_l1286_128665


namespace NUMINAMATH_GPT_max_operations_l1286_128625

def arithmetic_mean (a b : ℕ) := (a + b) / 2

theorem max_operations (b : ℕ) (hb : b < 2002) (heven : (2002 + b) % 2 = 0) :
  ∃ n, n = 10 ∧ (2002 - b) / 2^n = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_operations_l1286_128625


namespace NUMINAMATH_GPT_intersection_A_B_l1286_128686

def set_A : Set ℕ := {x | x^2 - 2 * x = 0}
def set_B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : set_A ∩ set_B = {0, 2} := 
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1286_128686


namespace NUMINAMATH_GPT_seashells_solution_l1286_128630

def seashells_problem (T : ℕ) : Prop :=
  T + 13 = 50 → T = 37

theorem seashells_solution : seashells_problem 37 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_seashells_solution_l1286_128630


namespace NUMINAMATH_GPT_book_collection_example_l1286_128670

theorem book_collection_example :
  ∃ (P C B : ℕ), 
    (P : ℚ) / C = 3 / 2 ∧ 
    (C : ℚ) / B = 4 / 3 ∧ 
    P + C + B = 3002 ∧ 
    P + C + B > 3000 :=
by
  sorry

end NUMINAMATH_GPT_book_collection_example_l1286_128670


namespace NUMINAMATH_GPT_rooks_same_distance_l1286_128667

theorem rooks_same_distance (rooks : Fin 8 → (ℕ × ℕ)) 
    (h_non_attacking : ∀ i j, i ≠ j → Prod.fst (rooks i) ≠ Prod.fst (rooks j) ∧ Prod.snd (rooks i) ≠ Prod.snd (rooks j)) 
    : ∃ i j k l, i ≠ j ∧ k ≠ l ∧ (Prod.fst (rooks i) - Prod.fst (rooks k))^2 + (Prod.snd (rooks i) - Prod.snd (rooks k))^2 = (Prod.fst (rooks j) - Prod.fst (rooks l))^2 + (Prod.snd (rooks j) - Prod.snd (rooks l))^2 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rooks_same_distance_l1286_128667


namespace NUMINAMATH_GPT_max_value_of_XYZ_XY_YZ_ZX_l1286_128620

theorem max_value_of_XYZ_XY_YZ_ZX (X Y Z : ℕ) (h : X + Y + Z = 15) : 
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 := 
sorry

end NUMINAMATH_GPT_max_value_of_XYZ_XY_YZ_ZX_l1286_128620


namespace NUMINAMATH_GPT_completing_the_square_l1286_128611

theorem completing_the_square (x : ℝ) :
  4 * x^2 - 2 * x - 1 = 0 → (x - 1/4)^2 = 5/16 := 
by
  sorry

end NUMINAMATH_GPT_completing_the_square_l1286_128611


namespace NUMINAMATH_GPT_cruzs_marbles_l1286_128624

theorem cruzs_marbles (Atticus Jensen Cruz : ℕ) 
  (h1 : 3 * (Atticus + Jensen + Cruz) = 60) 
  (h2 : Atticus = Jensen / 2) 
  (h3 : Atticus = 4) : 
  Cruz = 8 := 
sorry

end NUMINAMATH_GPT_cruzs_marbles_l1286_128624


namespace NUMINAMATH_GPT_three_squares_not_divisible_by_three_l1286_128642

theorem three_squares_not_divisible_by_three 
  (N : ℕ) (a b c : ℤ) 
  (h₁ : N = 9 * (a^2 + b^2 + c^2)) :
  ∃ x y z : ℤ, N = x^2 + y^2 + z^2 ∧ ¬ (3 ∣ x) ∧ ¬ (3 ∣ y) ∧ ¬ (3 ∣ z) := 
sorry

end NUMINAMATH_GPT_three_squares_not_divisible_by_three_l1286_128642


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1286_128646

theorem sum_of_a_and_b (a b : ℝ) (h1 : abs a = 5) (h2 : b = -2) (h3 : a * b > 0) : a + b = -7 := by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1286_128646


namespace NUMINAMATH_GPT_part_I_part_II_l1286_128666

noncomputable def curve_M (theta : ℝ) : ℝ := 4 * Real.cos theta

noncomputable def line_l (t m alpha : ℝ) : ℝ × ℝ :=
  let x := m + t * Real.cos alpha
  let y := t * Real.sin alpha
  (x, y)

theorem part_I (varphi : ℝ) :
  let OB := curve_M (varphi + π / 4)
  let OC := curve_M (varphi - π / 4)
  let OA := curve_M varphi
  OB + OC = Real.sqrt 2 * OA := by
  sorry

theorem part_II (m alpha : ℝ) :
  let varphi := π / 12
  let B := (1, Real.sqrt 3)
  let C := (3, -Real.sqrt 3)
  exists t1 t2, line_l t1 m alpha = B ∧ line_l t2 m alpha = C :=
  have hα : alpha = 2 * π / 3 := by sorry
  have hm : m = 2 := by sorry
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1286_128666


namespace NUMINAMATH_GPT_avg_one_sixth_one_fourth_l1286_128678

theorem avg_one_sixth_one_fourth : (1 / 6 + 1 / 4) / 2 = 5 / 24 := by
  sorry

end NUMINAMATH_GPT_avg_one_sixth_one_fourth_l1286_128678


namespace NUMINAMATH_GPT_initial_total_balls_l1286_128639

theorem initial_total_balls (B T : Nat) (h1 : B = 9) (h2 : ∀ (n : Nat), (T - 5) * 1/5 = 4) :
  T = 25 := sorry

end NUMINAMATH_GPT_initial_total_balls_l1286_128639


namespace NUMINAMATH_GPT_exists_f_with_f3_eq_9_forall_f_f3_le_9_l1286_128622

-- Define the real-valued function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_real : ∀ x : ℝ, true)  -- f is real-valued and defined for all real numbers
variable (f_mul : ∀ x y : ℝ, f (x * y) = f x * f y)  -- f(xy) = f(x)f(y)
variable (f_add : ∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y))  -- f(x+y) ≤ 2(f(x) + f(y))
variable (f_2 : f 2 = 4)  -- f(2) = 4

-- Part a
theorem exists_f_with_f3_eq_9 : ∃ f : ℝ → ℝ, (∀ x : ℝ, true) ∧ 
                              (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
                              (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) ∧ 
                              (f 2 = 4) ∧ 
                              (f 3 = 9) := 
sorry

-- Part b
theorem forall_f_f3_le_9 : ∀ f : ℝ → ℝ, 
                        (∀ x : ℝ, true) → 
                        (∀ x y : ℝ, f (x * y) = f x * f y) → 
                        (∀ x y : ℝ, f (x + y) ≤ 2 * (f x + f y)) → 
                        (f 2 = 4) → 
                        (f 3 ≤ 9) := 
sorry

end NUMINAMATH_GPT_exists_f_with_f3_eq_9_forall_f_f3_le_9_l1286_128622


namespace NUMINAMATH_GPT_poly_div_l1286_128605

theorem poly_div (A B : ℂ) :
  (∀ x : ℂ, x^3 + x^2 + 1 = 0 → x^202 + A * x + B = 0) → A + B = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_poly_div_l1286_128605


namespace NUMINAMATH_GPT_parking_lot_cars_l1286_128672

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end NUMINAMATH_GPT_parking_lot_cars_l1286_128672


namespace NUMINAMATH_GPT_sample_size_eq_100_l1286_128631

variables (frequency : ℕ) (frequency_rate : ℚ)

theorem sample_size_eq_100 (h1 : frequency = 50) (h2 : frequency_rate = 0.5) :
  frequency / frequency_rate = 100 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_eq_100_l1286_128631


namespace NUMINAMATH_GPT_factorize_expression_l1286_128638

theorem factorize_expression (x : ℝ) : -2 * x^2 + 2 * x - (1 / 2) = -2 * (x - (1 / 2))^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1286_128638


namespace NUMINAMATH_GPT_bicycle_distance_l1286_128692

theorem bicycle_distance (b t : ℝ) (h : t ≠ 0) :
  let rate := (b / 2) / t / 3
  let total_seconds := 5 * 60
  rate * total_seconds = 50 * b / t := by
    sorry

end NUMINAMATH_GPT_bicycle_distance_l1286_128692


namespace NUMINAMATH_GPT_value_of_smaller_denom_l1286_128635

-- We are setting up the conditions given in the problem.
variables (x : ℕ) -- The value of the smaller denomination bill.

-- Condition 1: She has 4 bills of denomination x.
def value_smaller_denomination : ℕ := 4 * x

-- Condition 2: She has 8 bills of $10 denomination.
def value_ten_bills : ℕ := 8 * 10

-- Condition 3: The total value of the bills is $100.
def total_value : ℕ := 100

-- Prove that x = 5 using the given conditions.
theorem value_of_smaller_denom : value_smaller_denomination x + value_ten_bills = total_value → x = 5 :=
by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_value_of_smaller_denom_l1286_128635


namespace NUMINAMATH_GPT_base_k_sum_l1286_128679

theorem base_k_sum (k : ℕ) (t : ℕ) (h1 : (k + 3) * (k + 4) * (k + 7) = 4 * k^3 + 7 * k^2 + 3 * k + 5)
    (h2 : t = (k + 3) + (k + 4) + (k + 7)) :
    t = 50 := sorry

end NUMINAMATH_GPT_base_k_sum_l1286_128679


namespace NUMINAMATH_GPT_fraction_operation_l1286_128617

theorem fraction_operation : (3 / 5 - 1 / 10 + 2 / 15 = 19 / 30) :=
by
  sorry

end NUMINAMATH_GPT_fraction_operation_l1286_128617


namespace NUMINAMATH_GPT_intersection_points_hyperbola_l1286_128683

theorem intersection_points_hyperbola (t : ℝ) :
  ∃ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 4 = 0) ∧ 
  (x^2 / 4 - y^2 / (9 / 16) = 1) :=
sorry

end NUMINAMATH_GPT_intersection_points_hyperbola_l1286_128683


namespace NUMINAMATH_GPT_fred_found_43_seashells_l1286_128615

-- Define the conditions
def tom_seashells : ℕ := 15
def additional_seashells : ℕ := 28

-- Define Fred's total seashells based on the conditions
def fred_seashells : ℕ := tom_seashells + additional_seashells

-- The theorem to prove that Fred found 43 seashells
theorem fred_found_43_seashells : fred_seashells = 43 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fred_found_43_seashells_l1286_128615


namespace NUMINAMATH_GPT_fewest_four_dollar_frisbees_l1286_128673

-- Definitions based on the conditions
variables (x y : ℕ) -- The numbers of $3 and $4 frisbees, respectively.
def total_frisbees (x y : ℕ) : Prop := x + y = 60
def total_receipts (x y : ℕ) : Prop := 3 * x + 4 * y = 204

-- The statement to prove
theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : total_frisbees x y) (h2 : total_receipts x y) : y = 24 :=
sorry

end NUMINAMATH_GPT_fewest_four_dollar_frisbees_l1286_128673


namespace NUMINAMATH_GPT_quadratic_z_and_u_l1286_128662

variables (a b c α β γ : ℝ)
variable (d : ℝ)
variable (δ : ℝ)
variables (x₁ x₂ y₁ y₂ z₁ z₂ u₁ u₂ : ℝ)

-- Given conditions
variable (h_nonzero : a * α ≠ 0)
variable (h_discriminant1 : b^2 - 4 * a * c ≥ 0)
variable (h_discriminant2 : β^2 - 4 * α * γ ≥ 0)
variable (hx_roots_order : x₁ ≤ x₂)
variable (hy_roots_order : y₁ ≤ y₂)
variable (h_eq_discriminant1 : b^2 - 4 * a * c = d^2)
variable (h_eq_discriminant2 : β^2 - 4 * α * γ = δ^2)

-- Translate into mathematical constraints for the roots
variable (hx1 : x₁ = (-b - d) / (2 * a))
variable (hx2 : x₂ = (-b + d) / (2 * a))
variable (hy1 : y₁ = (-β - δ) / (2 * α))
variable (hy2 : y₂ = (-β + δ) / (2 * α))

-- Variables for polynomial equations roots
axiom h_z1 : z₁ = x₁ + y₁
axiom h_z2 : z₂ = x₂ + y₂
axiom h_u1 : u₁ = x₁ + y₂
axiom h_u2 : u₂ = x₂ + y₁

theorem quadratic_z_and_u :
  (2 * a * α) * z₂ * z₂ + 2 * (a * β + α * b) * z₁ + (2 * a * γ + 2 * α * c + b * β - d * δ) = 0 ∧
  (2 * a * α) * u₂ * u₂ + 2 * (a * β + α * b) * u₁ + (2 * a * γ + 2 * α * c + b * β + d * δ) = 0 := sorry

end NUMINAMATH_GPT_quadratic_z_and_u_l1286_128662


namespace NUMINAMATH_GPT_micheal_item_count_l1286_128601

theorem micheal_item_count : ∃ a b c : ℕ, a + b + c = 50 ∧ 60 * a + 500 * b + 400 * c = 10000 ∧ a = 30 :=
  by
    sorry

end NUMINAMATH_GPT_micheal_item_count_l1286_128601


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1286_128640

theorem sufficient_not_necessary (x y : ℝ) : (x > |y|) → (x > y ∧ ¬ (x > y → x > |y|)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1286_128640


namespace NUMINAMATH_GPT_one_prime_p_10_14_l1286_128649

theorem one_prime_p_10_14 :
  ∃! (p : ℕ), Prime p ∧ Prime (p + 10) ∧ Prime (p + 14) :=
sorry

end NUMINAMATH_GPT_one_prime_p_10_14_l1286_128649


namespace NUMINAMATH_GPT_ellipse_hyperbola_tangent_l1286_128660

theorem ellipse_hyperbola_tangent (m : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 = 9 ∧ x^2 - m * (y - 2)^2 = 4) →
  m = 45 / 31 :=
by sorry

end NUMINAMATH_GPT_ellipse_hyperbola_tangent_l1286_128660


namespace NUMINAMATH_GPT_find_sum_of_coefficients_l1286_128696

theorem find_sum_of_coefficients : 
  (∃ m n p : ℕ, 
    (n.gcd p = 1) ∧ 
    m + 36 = 72 ∧
    n + 33*3 = 103 ∧ 
    p = 3 ∧ 
    (72 + 33 * ℼ + (8 * (1/8 * (4 * π / 3))) + 36) = m + n * π / p) → 
  m + n + p = 430 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_sum_of_coefficients_l1286_128696


namespace NUMINAMATH_GPT_exponent_comparison_l1286_128668

theorem exponent_comparison : 1.7 ^ 0.3 > 0.9 ^ 11 := 
by sorry

end NUMINAMATH_GPT_exponent_comparison_l1286_128668


namespace NUMINAMATH_GPT_number_of_correct_statements_l1286_128690

theorem number_of_correct_statements (a : ℚ) : 
  (¬ (a < 0 → -a < 0) ∧ ¬ (|a| > 0) ∧ ¬ ((a < 0 ∨ -a < 0) ∧ ¬ (a = 0))) 
  → 0 = 0 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_correct_statements_l1286_128690


namespace NUMINAMATH_GPT_range_of_k_l1286_128619

theorem range_of_k (k : ℝ) :
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0 ↔ k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1286_128619


namespace NUMINAMATH_GPT_johns_raise_percentage_increase_l1286_128627

def initial_earnings : ℚ := 65
def new_earnings : ℚ := 70
def percentage_increase (initial new : ℚ) : ℚ := ((new - initial) / initial) * 100

theorem johns_raise_percentage_increase : percentage_increase initial_earnings new_earnings = 7.692307692 :=
by
  sorry

end NUMINAMATH_GPT_johns_raise_percentage_increase_l1286_128627


namespace NUMINAMATH_GPT_tax_deduction_cents_l1286_128643

def bob_hourly_wage : ℝ := 25
def tax_rate : ℝ := 0.025

theorem tax_deduction_cents :
  (bob_hourly_wage * 100 * tax_rate) = 62.5 :=
by
  -- This is the statement that needs to be proven.
  sorry

end NUMINAMATH_GPT_tax_deduction_cents_l1286_128643


namespace NUMINAMATH_GPT_train_passing_time_l1286_128661

noncomputable def length_of_train : ℝ := 450
noncomputable def speed_kmh : ℝ := 80
noncomputable def length_of_station : ℝ := 300
noncomputable def speed_m_per_s : ℝ := speed_kmh * 1000 / 3600 -- Convert km/hour to m/second
noncomputable def total_distance : ℝ := length_of_train + length_of_station
noncomputable def passing_time : ℝ := total_distance / speed_m_per_s

theorem train_passing_time : abs (passing_time - 33.75) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_train_passing_time_l1286_128661


namespace NUMINAMATH_GPT_frozenFruitSold_l1286_128681

variable (totalFruit : ℕ) (freshFruit : ℕ)

-- Define the condition that the total fruit sold is 9792 pounds
def totalFruitSold := totalFruit = 9792

-- Define the condition that the fresh fruit sold is 6279 pounds
def freshFruitSold := freshFruit = 6279

-- Define the question as a Lean statement
theorem frozenFruitSold
  (h1 : totalFruitSold totalFruit)
  (h2 : freshFruitSold freshFruit) :
  totalFruit - freshFruit = 3513 := by
  sorry

end NUMINAMATH_GPT_frozenFruitSold_l1286_128681


namespace NUMINAMATH_GPT_distinct_solutions_subtract_eight_l1286_128654

noncomputable def f (x : ℝ) : ℝ := (6 * x - 18) / (x^2 + 2 * x - 15)
noncomputable def equation := ∀ x, f x = x + 3

noncomputable def r_solutions (r s : ℝ) := (r > s) ∧ (f r = r + 3) ∧ (f s = s + 3)

theorem distinct_solutions_subtract_eight
  (r s : ℝ) (h : r_solutions r s) : r - s = 8 :=
sorry

end NUMINAMATH_GPT_distinct_solutions_subtract_eight_l1286_128654


namespace NUMINAMATH_GPT_hyperbola_equation_l1286_128641

theorem hyperbola_equation (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h_hyperbola : ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) 
  (h_focus : ∃ (p : ℝ × ℝ), p = (1, 0))
  (h_line_passing_focus : ∀ y, ∃ (m c : ℝ), y = -b * y + c)
  (h_parallel : ∀ x y : ℝ, b/a = -b)
  (h_perpendicular : ∀ x y : ℝ, b/a * (-b) = -1) : 
  ∀ x y : ℝ, x^2 - y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l1286_128641


namespace NUMINAMATH_GPT_train_speed_l1286_128688

def train_length : ℝ := 1000  -- train length in meters
def time_to_cross_pole : ℝ := 200  -- time to cross the pole in seconds

theorem train_speed : train_length / time_to_cross_pole = 5 := by
  sorry

end NUMINAMATH_GPT_train_speed_l1286_128688


namespace NUMINAMATH_GPT_min_abs_E_value_l1286_128652

theorem min_abs_E_value (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 10) : |E| = 9 :=
sorry

end NUMINAMATH_GPT_min_abs_E_value_l1286_128652


namespace NUMINAMATH_GPT_quad_form_b_c_sum_l1286_128684

theorem quad_form_b_c_sum :
  ∃ (b c : ℝ), (b + c = -10) ∧ (∀ x : ℝ, x^2 - 20 * x + 100 = (x + b)^2 + c) :=
by
  sorry

end NUMINAMATH_GPT_quad_form_b_c_sum_l1286_128684


namespace NUMINAMATH_GPT_typing_speed_equation_l1286_128632

theorem typing_speed_equation (x : ℕ) (h_pos : x > 0) :
  120 / x = 180 / (x + 6) :=
sorry

end NUMINAMATH_GPT_typing_speed_equation_l1286_128632


namespace NUMINAMATH_GPT_barium_oxide_amount_l1286_128693

theorem barium_oxide_amount (BaO H2O BaOH₂ : ℕ) 
  (reaction : BaO + H2O = BaOH₂) 
  (molar_ratio : BaOH₂ = BaO) 
  (required_BaOH₂ : BaOH₂ = 2) :
  BaO = 2 :=
by 
  sorry

end NUMINAMATH_GPT_barium_oxide_amount_l1286_128693


namespace NUMINAMATH_GPT_y_intercept_of_line_l1286_128634

theorem y_intercept_of_line (m x y b : ℝ) (h_slope : m = 4) (h_point : (x, y) = (199, 800)) (h_line : y = m * x + b) :
    b = 4 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1286_128634


namespace NUMINAMATH_GPT_find_angle_C_find_area_of_triangle_l1286_128651

variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides of the triangle

-- Proof 1: Prove \(C = \frac{\pi}{3}\) given \(a \cos B \cos C + b \cos A \cos C = \frac{c}{2}\).

theorem find_angle_C 
  (h : a * Real.cos B * Real.cos C + b * Real.cos A * Real.cos C = c / 2) : C = π / 3 :=
sorry

-- Proof 2: Prove the area of triangle \(ABC = \frac{3\sqrt{3}}{2}\) given \(c = \sqrt{7}\), \(a + b = 5\), and \(C = \frac{\pi}{3}\).

theorem find_area_of_triangle 
  (h1 : c = Real.sqrt 7) (h2 : a + b = 5) (h3 : C = π / 3) : 
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_area_of_triangle_l1286_128651


namespace NUMINAMATH_GPT_koschei_never_equal_l1286_128645

-- Define the problem setup 
def coins_at_vertices (n1 n2 n3 n4 n5 n6 : ℕ) : Prop := 
  ∃ k : ℕ, n1 = k ∧ n2 = k ∧ n3 = k ∧ n4 = k ∧ n5 = k ∧ n6 = k

-- Define the operation condition
def operation_condition (n1 n2 n3 n4 n5 n6 : ℕ) : Prop :=
  ∃ x : ℕ, (n1 - x = x ∧ n2 + 6 * x = x) ∨ (n2 - x = x ∧ n3 + 6 * x = x) ∨ 
  (n3 - x = x ∧ n4 + 6 * x = x) ∨ (n4 - x = x ∧ n5 + 6 * x = x) ∨ 
  (n5 - x = x ∧ n6 + 6 * x = x) ∨ (n6 - x = x ∧ n1 + 6 * x = x)

-- The main theorem 
theorem koschei_never_equal (n1 n2 n3 n4 n5 n6 : ℕ) : 
  (∃ x : ℕ, coins_at_vertices n1 n2 n3 n4 n5 n6) → False :=
by
  sorry

end NUMINAMATH_GPT_koschei_never_equal_l1286_128645


namespace NUMINAMATH_GPT_find_greatest_and_second_greatest_problem_solution_l1286_128671

theorem find_greatest_and_second_greatest
  (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : (a > b) ∧ (b > c) ∧ (c > d) :=
by 
  sorry

def greatest_and_second_greatest_eq (x1 x2 : ℝ) : Prop :=
  x1 = 4 ^ (1 / 4) ∧ x2 = 5 ^ (1 / 5)

theorem problem_solution (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : greatest_and_second_greatest_eq a b :=
by 
  sorry

end NUMINAMATH_GPT_find_greatest_and_second_greatest_problem_solution_l1286_128671


namespace NUMINAMATH_GPT_find_number_l1286_128606

theorem find_number (n : ℕ) : (n / 2) + 5 = 15 → n = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_number_l1286_128606


namespace NUMINAMATH_GPT_total_photos_l1286_128698

def n_pages1 : ℕ := 12
def photos_per_page1 : ℕ := 2
def n_pages2 : ℕ := 9
def photos_per_page2 : ℕ := 3

theorem total_photos : n_pages1 * photos_per_page1 + n_pages2 * photos_per_page2 = 51 := 
by 
  sorry

end NUMINAMATH_GPT_total_photos_l1286_128698


namespace NUMINAMATH_GPT_tetrahedron_volume_formula_l1286_128602

-- Definitions used directly in the conditions
variable (a b d : ℝ) (φ : ℝ)

-- Tetrahedron volume formula theorem statement
theorem tetrahedron_volume_formula 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b) 
  (hd_pos : 0 < d) 
  (hφ_pos : 0 < φ) 
  (hφ_le_pi : φ ≤ Real.pi) :
  (∀ V : ℝ, V = 1 / 6 * a * b * d * Real.sin φ) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_formula_l1286_128602


namespace NUMINAMATH_GPT_hall_of_mirrors_l1286_128626

theorem hall_of_mirrors (h : ℝ) 
    (condition1 : 2 * (30 * h) + (20 * h) = 960) :
  h = 12 :=
by
  sorry

end NUMINAMATH_GPT_hall_of_mirrors_l1286_128626


namespace NUMINAMATH_GPT_harriet_speed_l1286_128669

/-- Harriet drove back from B-town to A-ville at a constant speed of 145 km/hr.
    The entire trip took 5 hours, and it took Harriet 2.9 hours to drive from A-ville to B-town.
    Prove that Harriet's speed while driving from A-ville to B-town was 105 km/hr. -/
theorem harriet_speed (v_return : ℝ) (T_total : ℝ) (t_AB : ℝ) (v_AB : ℝ) :
  v_return = 145 →
  T_total = 5 →
  t_AB = 2.9 →
  v_AB = 105 :=
by
  intros
  sorry

end NUMINAMATH_GPT_harriet_speed_l1286_128669


namespace NUMINAMATH_GPT_cost_price_of_ball_l1286_128607

variable (C : ℝ)

theorem cost_price_of_ball (h : 15 * C - 720 = 5 * C) : C = 72 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_ball_l1286_128607


namespace NUMINAMATH_GPT_arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l1286_128677

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two (x : ℝ) (hx1 : -1 ≤ x) (hx2 : x ≤ 1) :
  (Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_arcsin_cos_arcsin_plus_arccos_sin_arccos_eq_pi_div_two_l1286_128677


namespace NUMINAMATH_GPT_thread_length_l1286_128657

theorem thread_length (initial_length : ℝ) (fraction : ℝ) (additional_length : ℝ) (total_length : ℝ) 
  (h1 : initial_length = 12) 
  (h2 : fraction = 3 / 4) 
  (h3 : additional_length = initial_length * fraction)
  (h4 : total_length = initial_length + additional_length) : 
  total_length = 21 := 
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_thread_length_l1286_128657


namespace NUMINAMATH_GPT_apples_remaining_in_each_basket_l1286_128656

-- Definition of conditions
def total_apples : ℕ := 128
def number_of_baskets : ℕ := 8
def apples_taken_per_basket : ℕ := 7

-- Definition of the problem
theorem apples_remaining_in_each_basket :
  (total_apples / number_of_baskets) - apples_taken_per_basket = 9 := 
by 
  sorry

end NUMINAMATH_GPT_apples_remaining_in_each_basket_l1286_128656


namespace NUMINAMATH_GPT_sum_product_le_four_l1286_128653

theorem sum_product_le_four (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) : 
  a * b + b * c + c * d + d * a ≤ 4 := 
sorry

end NUMINAMATH_GPT_sum_product_le_four_l1286_128653


namespace NUMINAMATH_GPT_problem_min_a2_area_l1286_128664

noncomputable def area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  0.5 * b * c * Real.sin A

noncomputable def min_a2_area (a b c : ℝ) (A B C : ℝ): ℝ := 
  let S := area a b c A B C
  a^2 / S

theorem problem_min_a2_area :
  ∀ (a b c A B C : ℝ), 
    a > 0 → b > 0 → c > 0 → 
    A + B + C = Real.pi →
    a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C →
    b * Real.cos C + c * Real.cos B = 3 * a * Real.cos A →
    min_a2_area a b c A B C ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_min_a2_area_l1286_128664


namespace NUMINAMATH_GPT_baby_whales_on_second_trip_l1286_128694

def iwishmael_whales_problem : Prop :=
  let male1 := 28
  let female1 := 2 * male1
  let male3 := male1 / 2
  let female3 := female1
  let total_whales := 178
  let total_without_babies := (male1 + female1) + (male3 + female3)
  total_whales - total_without_babies = 24

theorem baby_whales_on_second_trip : iwishmael_whales_problem :=
  by
  sorry

end NUMINAMATH_GPT_baby_whales_on_second_trip_l1286_128694


namespace NUMINAMATH_GPT_factors_of_1320_l1286_128691

theorem factors_of_1320 : ∃ n : ℕ, n = 24 ∧ ∃ (a b c d : ℕ),
  1320 = 2^a * 3^b * 5^c * 11^d ∧ (a = 0 ∨ a = 1 ∨ a = 2) ∧ (b = 0 ∨ b = 1) ∧ (c = 0 ∨ c = 1) ∧ (d = 0 ∨ d = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_factors_of_1320_l1286_128691


namespace NUMINAMATH_GPT_pet_food_total_weight_l1286_128648

theorem pet_food_total_weight:
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3 -- pounds
  let dog_food_bags := 4 
  let weight_per_dog_food_bag := 5 -- pounds
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2 -- pounds
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  total_weight_ounces = 624 :=
by
  let cat_food_bags := 3
  let weight_per_cat_food_bag := 3
  let dog_food_bags := 4
  let weight_per_dog_food_bag := 5
  let bird_food_bags := 5
  let weight_per_bird_food_bag := 2
  let total_weight_pounds := (cat_food_bags * weight_per_cat_food_bag) + (dog_food_bags * weight_per_dog_food_bag) + (bird_food_bags * weight_per_bird_food_bag)
  let total_weight_ounces := total_weight_pounds * 16
  show total_weight_ounces = 624
  sorry

end NUMINAMATH_GPT_pet_food_total_weight_l1286_128648


namespace NUMINAMATH_GPT_tom_average_speed_l1286_128682

theorem tom_average_speed 
  (karen_speed : ℕ) (tom_distance : ℕ) (karen_advantage : ℕ) (delay : ℚ)
  (h1 : karen_speed = 60)
  (h2 : tom_distance = 24)
  (h3 : karen_advantage = 4)
  (h4 : delay = 4/60) :
  ∃ (v : ℚ), v = 45 := by
  sorry

end NUMINAMATH_GPT_tom_average_speed_l1286_128682


namespace NUMINAMATH_GPT_parabola_intersection_min_y1_y2_sqr_l1286_128616

theorem parabola_intersection_min_y1_y2_sqr :
  ∀ (x1 x2 y1 y2 : ℝ)
    (h1 : y1 ^ 2 = 4 * x1)
    (h2 : y2 ^ 2 = 4 * x2)
    (h3 : (∃ k : ℝ, x1 = 4 ∧ y1 = k * (4 - 4)) ∨ x1 = 4 ∧ y1 ≠ x2),
    ∃ m : ℝ, (y1^2 + y2^2) = m ∧ m = 32 := 
sorry

end NUMINAMATH_GPT_parabola_intersection_min_y1_y2_sqr_l1286_128616


namespace NUMINAMATH_GPT_lawrence_worked_hours_l1286_128633

-- Let h_M, h_T, h_F be the hours worked on Monday, Tuesday, and Friday respectively
-- Let h_W be the hours worked on Wednesday (h_W = 5.5)
-- Let h_R be the hours worked on Thursday (h_R = 5.5)
-- Let total hours worked in 5 days be 25
-- Prove that h_M + h_T + h_F = 14

theorem lawrence_worked_hours :
  ∀ (h_M h_T h_F : ℝ), h_W = 5.5 → h_R = 5.5 → (5 * 5 = 25) → 
  h_M + h_T + h_F + h_W + h_R = 25 → h_M + h_T + h_F = 14 :=
by
  intros h_M h_T h_F h_W h_R h_total h_sum
  sorry

end NUMINAMATH_GPT_lawrence_worked_hours_l1286_128633


namespace NUMINAMATH_GPT_find_income_l1286_128612

-- Define the conditions
def income_and_expenditure (income expenditure : ℕ) : Prop :=
  5 * expenditure = 3 * income

def savings (income expenditure : ℕ) (saving : ℕ) : Prop :=
  income - expenditure = saving

-- State the theorem
theorem find_income (expenditure : ℕ) (saving : ℕ) (h1 : income_and_expenditure 5 3) (h2 : savings (5 * expenditure) (3 * expenditure) saving) :
  5 * expenditure = 10000 :=
by
  -- Use the provided hint or conditions
  sorry

end NUMINAMATH_GPT_find_income_l1286_128612


namespace NUMINAMATH_GPT_problem_solution_l1286_128647

noncomputable def question (x y z : ℝ) : Prop := 
  (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) → 
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) )

theorem problem_solution (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  ((x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∨ 
   (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ∨ 
   (z + x)/(z^2 + z*x + x^2) = (x + y)/(x^2 + x*y + y^2)) →
  ( (x + y)/(x^2 + x*y + y^2) = (y + z)/(y^2 + y*z + z^2) ∧ 
    (y + z)/(y^2 + y*z + z^2) = (z + x)/(z^2 + z*x + x^2) ) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1286_128647


namespace NUMINAMATH_GPT_greatest_b_l1286_128699

theorem greatest_b (b : ℤ) (h : ∀ x : ℝ, x^2 + b * x + 20 ≠ -6) : b = 10 := sorry

end NUMINAMATH_GPT_greatest_b_l1286_128699


namespace NUMINAMATH_GPT_ratio_of_metals_l1286_128614

theorem ratio_of_metals (G C S : ℝ) (h1 : 11 * G + 5 * C + 7 * S = 9 * (G + C + S)) : 
  G / C = 1 / 2 ∧ G / S = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_metals_l1286_128614


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1286_128610

-- Define the sets M and N with the given conditions
def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem that the intersection of M and N is as described
theorem intersection_of_M_and_N : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 1} :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1286_128610


namespace NUMINAMATH_GPT_ratio_larger_to_smaller_l1286_128676

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
  a / b

theorem ratio_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : ratio_of_numbers a b = 9 / 5 := 
  sorry

end NUMINAMATH_GPT_ratio_larger_to_smaller_l1286_128676


namespace NUMINAMATH_GPT_percentage_increase_l1286_128689

noncomputable def price_increase (d new_price : ℝ) : ℝ :=
  ((new_price - d) / d) * 100

theorem percentage_increase 
  (d new_price : ℝ)
  (h1 : 2 * d = 585)
  (h2 : new_price = 351) :
  price_increase d new_price = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1286_128689


namespace NUMINAMATH_GPT_greater_expected_area_vasya_l1286_128655

noncomputable def expected_area_vasya : ℚ :=
  (1/6) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

noncomputable def expected_area_asya : ℚ :=
  ((1/6) * (1 + 2 + 3 + 4 + 5 + 6)) * ((1/6) * (1 + 2 + 3 + 4 + 5 + 6))

theorem greater_expected_area_vasya : expected_area_vasya > expected_area_asya :=
  by
  -- We've provided the expected area values as definitions
  -- expected_area_vasya = 91/6
  -- vs. expected_area_asya = 12.25 = (21/6)^2 = 441/36 = 12.25
  sorry

end NUMINAMATH_GPT_greater_expected_area_vasya_l1286_128655


namespace NUMINAMATH_GPT_ratio_of_areas_of_squares_l1286_128650

theorem ratio_of_areas_of_squares (side_C side_D : ℕ) 
  (hC : side_C = 48) (hD : side_D = 60) : 
  (side_C^2 : ℚ)/(side_D^2 : ℚ) = 16/25 :=
by
  -- sorry, proof omitted
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_squares_l1286_128650


namespace NUMINAMATH_GPT_find_solutions_l1286_128629

theorem find_solutions :
  {x : ℝ | 1 / (x^2 + 12 * x - 9) + 1 / (x^2 + 3 * x - 9) + 1 / (x^2 - 14 * x - 9) = 0} = {1, -9, 3, -3} :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_l1286_128629


namespace NUMINAMATH_GPT_math_proof_problem_l1286_128659

/-- Given three real numbers a, b, and c such that a ≥ b ≥ 1 ≥ c ≥ 0 and a + b + c = 3.

Part (a): Prove that 2 ≤ ab + bc + ca ≤ 3.
Part (b): Prove that (24 / (a^3 + b^3 + c^3)) + (25 / (ab + bc + ca)) ≥ 14.
--/
theorem math_proof_problem (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ 1) (h3 : 1 ≥ c)
  (h4 : c ≥ 0) (h5 : a + b + c = 3) :
  (2 ≤ a * b + b * c + c * a ∧ a * b + b * c + c * a ≤ 3) ∧ 
  (24 / (a^3 + b^3 + c^3) + 25 / (a * b + b * c + c * a) ≥ 14) 
  :=
by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1286_128659


namespace NUMINAMATH_GPT_number_of_semesters_l1286_128623

-- Define the given conditions
def units_per_semester : ℕ := 20
def cost_per_unit : ℕ := 50
def total_cost : ℕ := 2000

-- Define the cost per semester using the conditions
def cost_per_semester := units_per_semester * cost_per_unit

-- Prove the number of semesters is 2 given the conditions
theorem number_of_semesters : total_cost / cost_per_semester = 2 := by
  -- Add a placeholder "sorry" to skip the actual proof
  sorry

end NUMINAMATH_GPT_number_of_semesters_l1286_128623


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1286_128697

theorem value_of_a_plus_b (a b : ℝ) (h : |a - 2| = -(b + 5)^2) : a + b = -3 :=
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1286_128697


namespace NUMINAMATH_GPT_part1_part2_l1286_128600

noncomputable def f (x a : ℝ) := |x - a|

theorem part1 (a m : ℝ) :
  (∀ x, f x a ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
by
  sorry

theorem part2 (t x : ℝ) (h_t : 0 ≤ t ∧ t < 2) :
  f x 2 + t ≥ f (x + 2) 2 ↔ x ≤ (t + 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1286_128600


namespace NUMINAMATH_GPT_cost_of_machines_max_type_A_machines_l1286_128644

-- Defining the cost equations for type A and type B machines
theorem cost_of_machines (x y : ℝ) (h1 : 3 * x + 2 * y = 31) (h2 : x - y = 2) : x = 7 ∧ y = 5 :=
sorry

-- Defining the budget constraint and computing the maximum number of type A machines purchasable
theorem max_type_A_machines (m : ℕ) (h : 7 * m + 5 * (6 - m) ≤ 34) : m ≤ 2 :=
sorry

end NUMINAMATH_GPT_cost_of_machines_max_type_A_machines_l1286_128644


namespace NUMINAMATH_GPT_combination_5_3_eq_10_l1286_128604

-- Define the combination function according to its formula
noncomputable def combination (n k : ℕ) : ℕ :=
  (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem stating the required result
theorem combination_5_3_eq_10 : combination 5 3 = 10 := by
  sorry

end NUMINAMATH_GPT_combination_5_3_eq_10_l1286_128604


namespace NUMINAMATH_GPT_value_of_x_l1286_128695

theorem value_of_x : 
  ∀ (x y z : ℕ), 
  (x = y / 3) ∧ 
  (y = z / 6) ∧ 
  (z = 72) → 
  x = 4 :=
by
  intros x y z h
  have h1 : y = z / 6 := h.2.1
  have h2 : z = 72 := h.2.2
  have h3 : x = y / 3 := h.1
  sorry

end NUMINAMATH_GPT_value_of_x_l1286_128695


namespace NUMINAMATH_GPT_total_people_transport_l1286_128680

-- Define the conditions
def boatA_trips_day1 := 7
def boatB_trips_day1 := 5
def boatA_capacity := 20
def boatB_capacity := 15
def boatA_trips_day2 := 5
def boatB_trips_day2 := 6

-- Define the theorem statement
theorem total_people_transport :
  (boatA_trips_day1 * boatA_capacity + boatB_trips_day1 * boatB_capacity) +
  (boatA_trips_day2 * boatA_capacity + boatB_trips_day2 * boatB_capacity)
  = 405 := 
  by
  sorry

end NUMINAMATH_GPT_total_people_transport_l1286_128680


namespace NUMINAMATH_GPT_carbonated_water_solution_l1286_128609

variable (V V_1 V_2 : ℝ)
variable (C2 : ℝ)

def carbonated_water_percent (V V1 V2 C2 : ℝ) : Prop :=
  0.8 * V1 + C2 * V2 = 0.6 * V

theorem carbonated_water_solution :
  ∀ (V : ℝ),
  (V1 = 0.1999999999999997 * V) →
  (V2 = 0.8000000000000003 * V) →
  carbonated_water_percent V V1 V2 C2 →
  C2 = 0.55 :=
by
  intros V V1_eq V2_eq carbonated_eq
  sorry

end NUMINAMATH_GPT_carbonated_water_solution_l1286_128609


namespace NUMINAMATH_GPT_polynomial_divisible_iff_l1286_128687

theorem polynomial_divisible_iff (a b : ℚ) : 
  ((a + b) * 1^5 + (a * b) * 1^2 + 1 = 0) ∧ 
  ((a + b) * 2^5 + (a * b) * 2^2 + 1 = 0) ↔ 
  (a = -1 ∧ b = 31/28) ∨ (a = 31/28 ∧ b = -1) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_divisible_iff_l1286_128687


namespace NUMINAMATH_GPT_segments_can_form_triangle_l1286_128675

noncomputable def can_form_triangle (a b c : ℝ) : Prop :=
  a + b + c = 2 ∧ a + b > 1 ∧ a + c > b ∧ b + c > a

theorem segments_can_form_triangle (a b c : ℝ) (h : a + b + c = 2) : (a + b > 1) ↔ (a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  sorry

end NUMINAMATH_GPT_segments_can_form_triangle_l1286_128675
