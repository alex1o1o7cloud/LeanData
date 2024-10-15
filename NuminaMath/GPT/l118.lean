import Mathlib

namespace NUMINAMATH_GPT_rotary_club_extra_omelets_l118_11861

theorem rotary_club_extra_omelets
  (small_children_tickets : ℕ)
  (older_children_tickets : ℕ)
  (adult_tickets : ℕ)
  (senior_tickets : ℕ)
  (eggs_total : ℕ)
  (omelet_for_small_child : ℝ)
  (omelet_for_older_child : ℝ)
  (omelet_for_adult : ℝ)
  (omelet_for_senior : ℝ)
  (eggs_per_omelet : ℕ)
  (extra_omelets : ℕ) :
  small_children_tickets = 53 →
  older_children_tickets = 35 →
  adult_tickets = 75 →
  senior_tickets = 37 →
  eggs_total = 584 →
  omelet_for_small_child = 0.5 →
  omelet_for_older_child = 1 →
  omelet_for_adult = 2 →
  omelet_for_senior = 1.5 →
  eggs_per_omelet = 2 →
  extra_omelets = (eggs_total - (2 * (small_children_tickets * omelet_for_small_child +
                                      older_children_tickets * omelet_for_older_child +
                                      adult_tickets * omelet_for_adult +
                                      senior_tickets * omelet_for_senior))) / eggs_per_omelet →
  extra_omelets = 25 :=
by
  intros hsmo_hold hsoc_hold hat_hold hsnt_hold htot_hold
        hosm_hold hocc_hold hact_hold hsen_hold hepom_hold hres_hold
  sorry

end NUMINAMATH_GPT_rotary_club_extra_omelets_l118_11861


namespace NUMINAMATH_GPT_sid_fraction_left_l118_11885

noncomputable def fraction_left (original total_spent remaining additional : ℝ) : ℝ :=
  (remaining - additional) / original

theorem sid_fraction_left 
  (original : ℝ := 48) 
  (spent_computer : ℝ := 12) 
  (spent_snacks : ℝ := 8) 
  (remaining : ℝ := 28) 
  (additional : ℝ := 4) :
  fraction_left original (spent_computer + spent_snacks) remaining additional = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sid_fraction_left_l118_11885


namespace NUMINAMATH_GPT_problem_statement_l118_11859

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f (x + 2016) = f (-x + 2016))
    (h2 : ∀ x1 x2 : ℝ, 2016 ≤ x1 ∧ 2016 ≤ x2 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
    f 2019 < f 2014 ∧ f 2014 < f 2017 :=
sorry

end NUMINAMATH_GPT_problem_statement_l118_11859


namespace NUMINAMATH_GPT_perfect_square_trinomial_l118_11862

theorem perfect_square_trinomial (a b c : ℤ) (f : ℤ → ℤ) (h : ∀ x : ℤ, f x = a * x^2 + b * x + c) :
  ∃ d e : ℤ, ∀ x : ℤ, f x = (d * x + e) ^ 2 :=
sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l118_11862


namespace NUMINAMATH_GPT_log_8_4000_l118_11897

theorem log_8_4000 : ∃ (n : ℤ), 8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 4000 ∧ 4000 < 4096 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_log_8_4000_l118_11897


namespace NUMINAMATH_GPT_min_value_fraction_l118_11858

theorem min_value_fraction (x : ℝ) (h : x > 6) : 
  (∃ x_min, x_min = 12 ∧ (∀ x > 6, (x * x) / (x - 6) ≥ 18) ∧ (x * x) / (x - 6) = 18) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l118_11858


namespace NUMINAMATH_GPT_tiffany_bags_found_day_after_next_day_l118_11863

noncomputable def tiffany_start : Nat := 10
noncomputable def tiffany_next_day : Nat := 3
noncomputable def tiffany_total : Nat := 20
noncomputable def tiffany_day_after_next_day : Nat := 20 - (tiffany_start + tiffany_next_day)

theorem tiffany_bags_found_day_after_next_day : tiffany_day_after_next_day = 7 := by
  sorry

end NUMINAMATH_GPT_tiffany_bags_found_day_after_next_day_l118_11863


namespace NUMINAMATH_GPT_ducks_in_the_marsh_l118_11803

-- Define the conditions
def number_of_geese : ℕ := 58
def total_number_of_birds : ℕ := 95
def number_of_ducks : ℕ := total_number_of_birds - number_of_geese

-- Prove the conclusion
theorem ducks_in_the_marsh : number_of_ducks = 37 := by
  -- subtraction to find number_of_ducks
  sorry

end NUMINAMATH_GPT_ducks_in_the_marsh_l118_11803


namespace NUMINAMATH_GPT_derivative_f_cos2x_l118_11839

variable {f : ℝ → ℝ} {x : ℝ}

theorem derivative_f_cos2x :
  f (Real.cos (2 * x)) = 1 - 2 * (Real.sin x) ^ 2 →
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end NUMINAMATH_GPT_derivative_f_cos2x_l118_11839


namespace NUMINAMATH_GPT_perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l118_11807

def square_land (A P D : ℝ) :=
  (5 * A = 10 * P + 45) ∧
  (3 * D = 2 * P + 10)

theorem perimeter_of_square_land_is_36 (A P D : ℝ) (h1 : 5 * A = 10 * P + 45) (h2 : 3 * D = 2 * P + 10) :
  P = 36 :=
sorry

theorem diagonal_of_square_land_is_27_33 (A P D : ℝ) (h1 : P = 36) (h2 : 3 * D = 2 * P + 10) :
  D = 82 / 3 :=
sorry

end NUMINAMATH_GPT_perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l118_11807


namespace NUMINAMATH_GPT_find_son_age_l118_11872

variable {S F : ℕ}

theorem find_son_age (h1 : F = S + 35) (h2 : F + 2 = 2 * (S + 2)) : S = 33 :=
sorry

end NUMINAMATH_GPT_find_son_age_l118_11872


namespace NUMINAMATH_GPT_parabola_equation_l118_11868

theorem parabola_equation
  (axis_of_symmetry : ∀ x y : ℝ, x = 1)
  (focus : ∀ x y : ℝ, x = -1 ∧ y = 0) :
  ∀ y x : ℝ, y^2 = -4*x := 
sorry

end NUMINAMATH_GPT_parabola_equation_l118_11868


namespace NUMINAMATH_GPT_final_price_correct_l118_11818

noncomputable def final_price_per_litre : Real :=
  let cost_1 := 70 * 43 * (1 - 0.15)
  let cost_2 := 50 * 51 * (1 + 0.10)
  let cost_3 := 15 * 60 * (1 - 0.08)
  let cost_4 := 25 * 62 * (1 + 0.12)
  let cost_5 := 40 * 67 * (1 - 0.05)
  let cost_6 := 10 * 75 * (1 - 0.18)
  let total_cost := cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6
  let total_volume := 70 + 50 + 15 + 25 + 40 + 10
  total_cost / total_volume

theorem final_price_correct : final_price_per_litre = 52.80 := by
  sorry

end NUMINAMATH_GPT_final_price_correct_l118_11818


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l118_11880

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l118_11880


namespace NUMINAMATH_GPT_prove_sum_is_12_l118_11814

theorem prove_sum_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := 
by 
  sorry

end NUMINAMATH_GPT_prove_sum_is_12_l118_11814


namespace NUMINAMATH_GPT_miles_to_burger_restaurant_l118_11879

-- Definitions and conditions
def miles_per_gallon : ℕ := 19
def gallons_of_gas : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_friend_house : ℕ := 4
def miles_to_home : ℕ := 11
def total_gas_distance := miles_per_gallon * gallons_of_gas
def total_known_distances := miles_to_school + miles_to_softball_park + miles_to_friend_house + miles_to_home

-- Problem statement to prove
theorem miles_to_burger_restaurant :
  ∃ (miles_to_burger_restaurant : ℕ), 
  total_gas_distance = total_known_distances + miles_to_burger_restaurant ∧ miles_to_burger_restaurant = 2 := 
by
  sorry

end NUMINAMATH_GPT_miles_to_burger_restaurant_l118_11879


namespace NUMINAMATH_GPT_find_general_term_find_sum_of_b_l118_11801

variables {n : ℕ} (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Given conditions
axiom a5 : a 5 = 10
axiom S7 : S 7 = 56

-- Definition of S (Sum of first n terms of an arithmetic sequence)
def S_def (a : ℕ → ℕ) (n : ℕ) : ℕ := n * (a 1 + a n) / 2

-- Definition of the arithmetic sequence
def a_arith_seq (n : ℕ) : ℕ := 2 * n

-- Assuming the axiom for the arithmetic sequence sum
axiom S_is_arith : S 7 = S_def a 7

theorem find_general_term : a = a_arith_seq := 
by sorry

-- Sequence b
def b (n : ℕ) : ℕ := 2 + 9 ^ n

-- Sum of first n terms of sequence b
def T (n : ℕ) : ℕ := (Finset.range n).sum b

-- Prove T_n formula
theorem find_sum_of_b : ∀ n, T n = 2 * n + 9 / 8 * (9 ^ n - 1) :=
by sorry

end NUMINAMATH_GPT_find_general_term_find_sum_of_b_l118_11801


namespace NUMINAMATH_GPT_algebraic_expression_value_l118_11834

variable {a b c : ℝ}

theorem algebraic_expression_value
  (h1 : (a + b) * (b + c) * (c + a) = 0)
  (h2 : a * b * c < 0) :
  (a / |a|) + (b / |b|) + (c / |c|) = 1 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l118_11834


namespace NUMINAMATH_GPT_find_fraction_eq_l118_11869

theorem find_fraction_eq 
  {x : ℚ} 
  (h : x / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_eq_l118_11869


namespace NUMINAMATH_GPT_truck_tank_capacity_l118_11865

-- Definitions based on conditions
def truck_tank (T : ℝ) : Prop := true
def car_tank : Prop := true
def truck_half_full (T : ℝ) : Prop := true
def car_third_full : Prop := true
def add_fuel (T : ℝ) : Prop := T / 2 + 8 = 18

-- Theorem statement
theorem truck_tank_capacity (T : ℝ) (ht : truck_tank T) (hc : car_tank) 
  (ht_half : truck_half_full T) (hc_third : car_third_full) (hf_add : add_fuel T) : T = 20 :=
  sorry

end NUMINAMATH_GPT_truck_tank_capacity_l118_11865


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l118_11828

theorem point_in_fourth_quadrant (m : ℝ) (h : m < 0) : (-m + 1 > 0 ∧ -1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l118_11828


namespace NUMINAMATH_GPT_fraction_painted_red_l118_11875

theorem fraction_painted_red :
  let matilda_section := (1:ℚ) / 2 -- Matilda's half section
  let ellie_section := (1:ℚ) / 2    -- Ellie's half section
  let matilda_painted := matilda_section / 2 -- Matilda's painted fraction
  let ellie_painted := ellie_section / 3    -- Ellie's painted fraction
  (matilda_painted + ellie_painted) = 5 / 12 := 
by
  sorry

end NUMINAMATH_GPT_fraction_painted_red_l118_11875


namespace NUMINAMATH_GPT_trig_expression_eval_l118_11804

open Real

-- Declare the main theorem
theorem trig_expression_eval (θ : ℝ) (k : ℤ) 
  (h : sin (θ + k * π) = -2 * cos (θ + k * π)) :
  (4 * sin θ - 2 * cos θ) / (5 * cos θ + 3 * sin θ) = 10 :=
  sorry

end NUMINAMATH_GPT_trig_expression_eval_l118_11804


namespace NUMINAMATH_GPT_number_of_people_needed_to_lift_car_l118_11896

-- Define the conditions as Lean definitions
def twice_as_many_people_to_lift_truck (C T : ℕ) : Prop :=
  T = 2 * C

def people_needed_for_cars_and_trucks (C T total_people : ℕ) : Prop :=
  60 = 6 * C + 3 * T

-- Define the theorem statement using the conditions
theorem number_of_people_needed_to_lift_car :
  ∃ C, (∃ T, twice_as_many_people_to_lift_truck C T) ∧ people_needed_for_cars_and_trucks C T 60 ∧ C = 5 :=
sorry

end NUMINAMATH_GPT_number_of_people_needed_to_lift_car_l118_11896


namespace NUMINAMATH_GPT_ratio_of_w_to_y_l118_11867

theorem ratio_of_w_to_y (w x y z : ℚ)
  (h1 : w / x = 5 / 4)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 4) :
  w / y = 10 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_w_to_y_l118_11867


namespace NUMINAMATH_GPT_sophia_finished_more_pages_l118_11882

noncomputable def length_of_book : ℝ := 89.99999999999999

noncomputable def total_pages : ℕ := 90  -- Considering the practical purpose

noncomputable def finished_pages : ℕ := total_pages * 2 / 3

noncomputable def remaining_pages : ℕ := total_pages - finished_pages

theorem sophia_finished_more_pages :
  finished_pages - remaining_pages = 30 := 
  by
    -- Use sorry here as placeholder for the proof
    sorry

end NUMINAMATH_GPT_sophia_finished_more_pages_l118_11882


namespace NUMINAMATH_GPT_largest_even_whole_number_l118_11806

theorem largest_even_whole_number (x : ℕ) (h1 : 9 * x < 150) (h2 : x % 2 = 0) : x ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_largest_even_whole_number_l118_11806


namespace NUMINAMATH_GPT_minimum_knights_l118_11841

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end NUMINAMATH_GPT_minimum_knights_l118_11841


namespace NUMINAMATH_GPT_unique_integral_solution_l118_11802

noncomputable def positiveInt (x : ℤ) : Prop := x > 0

theorem unique_integral_solution (m n : ℤ) (hm : positiveInt m) (hn : positiveInt n) (unique_sol : ∃! (x y : ℤ), x + y^2 = m ∧ x^2 + y = n) : 
  ∃ (k : ℕ), m - n = 2^k ∨ m - n = -2^k :=
sorry

end NUMINAMATH_GPT_unique_integral_solution_l118_11802


namespace NUMINAMATH_GPT_caffeine_in_cup_l118_11895

-- Definitions based on the conditions
def caffeine_goal : ℕ := 200
def excess_caffeine : ℕ := 40
def total_cups : ℕ := 3

-- The statement proving that the amount of caffeine in a cup is 80 mg given the conditions.
theorem caffeine_in_cup : (3 * (80 : ℕ)) = (caffeine_goal + excess_caffeine) := by
  -- Plug in the value and simplify
  simp [caffeine_goal, excess_caffeine]

end NUMINAMATH_GPT_caffeine_in_cup_l118_11895


namespace NUMINAMATH_GPT_ascending_function_k_ge_2_l118_11805

open Real

def is_ascending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem ascending_function_k_ge_2 :
  ∀ (k : ℝ), (∀ x : ℝ, x ≥ -1 → (x + k) ^ 2 ≥ x ^ 2) → k ≥ 2 :=
by
  intros k h
  sorry

end NUMINAMATH_GPT_ascending_function_k_ge_2_l118_11805


namespace NUMINAMATH_GPT_part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l118_11886

-- Definitions for the sequences
def first_row (n : ℕ) : ℤ := (-3) ^ n
def second_row (n : ℕ) : ℤ := (-3) ^ n - 3
def third_row (n : ℕ) : ℤ := -((-3) ^ n) - 1

-- Statement for part 1
theorem part1_fifth_numbers:
  first_row 5 = -243 ∧ second_row 5 = -246 ∧ third_row 5 = 242 := sorry

-- Statement for part 2
theorem part2_three_adjacent_sum :
  ∃ n : ℕ, first_row (n-1) + first_row n + first_row (n+1) = -1701 ∧
           first_row (n-1) = -243 ∧ first_row n = 729 ∧ first_row (n+1) = -2187 := sorry

-- Statement for part 3
def sum_nth (n : ℕ) : ℤ := first_row n + second_row n + third_row n
theorem part3_difference_largest_smallest (n : ℕ) (m : ℤ) (hn : sum_nth n = m) :
  (∃ diff, (n % 2 = 1 → diff = -2 * m - 6) ∧ (n % 2 = 0 → diff = 2 * m + 9)) := sorry

end NUMINAMATH_GPT_part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l118_11886


namespace NUMINAMATH_GPT_intersection_point_l118_11827

theorem intersection_point (a b d x y : ℝ) (h1 : a = b + d) (h2 : a * x + b * y = b + 2 * d) :
    (x, y) = (-1, 1) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l118_11827


namespace NUMINAMATH_GPT_trip_early_movie_savings_l118_11870

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end NUMINAMATH_GPT_trip_early_movie_savings_l118_11870


namespace NUMINAMATH_GPT_negation_equiv_l118_11873

open Nat

theorem negation_equiv (P : Prop) :
  (¬ (∃ n : ℕ, (n! * n!) > (2^n))) ↔ (∀ n : ℕ, (n! * n!) ≤ (2^n)) :=
by
  sorry

end NUMINAMATH_GPT_negation_equiv_l118_11873


namespace NUMINAMATH_GPT_drum_oil_capacity_l118_11830

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) 
  (hX : DrumX_Oil = 0.5 * C) 
  (hY : DrumY_Cap = 2 * C) 
  (hY_filled : Y + 0.5 * C = 0.65 * (2 * C)) :
  Y = 0.8 * C :=
by
  sorry

end NUMINAMATH_GPT_drum_oil_capacity_l118_11830


namespace NUMINAMATH_GPT_seq_inequality_l118_11846

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 3 ∧ a 3 = 6 ∧ ∀ n, n > 3 → a n = 3 * a (n - 1) - a (n - 2) - 2 * a (n - 3)

theorem seq_inequality (a : ℕ → ℕ) (h : seq a) : ∀ n, n > 3 → a n > 3 * 2 ^ (n - 2) :=
  sorry

end NUMINAMATH_GPT_seq_inequality_l118_11846


namespace NUMINAMATH_GPT_find_2023rd_letter_l118_11854

def seq : List Char := ['A', 'B', 'C', 'D', 'D', 'C', 'B', 'A']

theorem find_2023rd_letter : seq.get! ((2023 % seq.length) - 1) = 'B' :=
by
  sorry

end NUMINAMATH_GPT_find_2023rd_letter_l118_11854


namespace NUMINAMATH_GPT_Nickel_ate_3_chocolates_l118_11845

-- Definitions of the conditions
def Robert_chocolates : ℕ := 12
def extra_chocolates : ℕ := 9
def Nickel_chocolates (N : ℕ) : Prop := Robert_chocolates = N + extra_chocolates

-- The proof goal
theorem Nickel_ate_3_chocolates : ∃ N : ℕ, Nickel_chocolates N ∧ N = 3 :=
by
  sorry

end NUMINAMATH_GPT_Nickel_ate_3_chocolates_l118_11845


namespace NUMINAMATH_GPT_min_abs_sum_l118_11812

theorem min_abs_sum : ∃ x : ℝ, (|x + 1| + |x + 2| + |x + 6|) = 5 :=
sorry

end NUMINAMATH_GPT_min_abs_sum_l118_11812


namespace NUMINAMATH_GPT_probability_event_a_without_replacement_independence_of_events_with_replacement_l118_11836

open ProbabilityTheory MeasureTheory Set

-- Definitions corresponding to the conditions
def BallLabeled (i : ℕ) : Prop := i ∈ Finset.range 10

def EventA (second_ball : ℕ) : Prop := second_ball = 2

def EventB (first_ball second_ball : ℕ) (m : ℕ) : Prop := first_ball + second_ball = m

-- First Part: Probability without replacement
theorem probability_event_a_without_replacement :
  ∃ P_A : ℝ, P_A = 1 / 10 := sorry

-- Second Part: Independence with replacement
theorem independence_of_events_with_replacement (m : ℕ) :
  (EventA 2 → (∀ first_ball : ℕ, BallLabeled first_ball → EventB first_ball 2 m) ↔ m = 9) := sorry

end NUMINAMATH_GPT_probability_event_a_without_replacement_independence_of_events_with_replacement_l118_11836


namespace NUMINAMATH_GPT_integer_solutions_l118_11815

theorem integer_solutions :
  { (x, y) : ℤ × ℤ | x^2 = 1 + 4 * y^3 * (y + 2) } = {(1, 0), (1, -2), (-1, 0), (-1, -2)} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l118_11815


namespace NUMINAMATH_GPT_initial_shells_l118_11866

theorem initial_shells (x : ℕ) (h : x + 23 = 28) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_shells_l118_11866


namespace NUMINAMATH_GPT_solution_interval_l118_11823

def check_solution (b : ℝ) (x : ℝ) : ℝ :=
  x^2 - b * x - 5

theorem solution_interval (b x : ℝ) :
  (check_solution b (-2) = 5) ∧
  (check_solution b (-1) = -1) ∧
  (check_solution b (4) = -1) ∧
  (check_solution b (5) = 5) →
  (∃ x, -2 < x ∧ x < -1 ∧ check_solution b x = 0) ∨
  (∃ x, 4 < x ∧ x < 5 ∧ check_solution b x = 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_l118_11823


namespace NUMINAMATH_GPT_simplify_cosine_tangent_product_of_cosines_l118_11878

-- Problem 1
theorem simplify_cosine_tangent :
  Real.cos 40 * (1 + Real.sqrt 3 * Real.tan 10) = 1 :=
sorry

-- Problem 2
theorem product_of_cosines :
  (Real.cos (2 * Real.pi / 7)) * (Real.cos (4 * Real.pi / 7)) * (Real.cos (6 * Real.pi / 7)) = 1 / 8 :=
sorry

end NUMINAMATH_GPT_simplify_cosine_tangent_product_of_cosines_l118_11878


namespace NUMINAMATH_GPT_prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l118_11891

-- Problem 1: Proof of sin(C - B) = 1 given the trigonometric identity
theorem prove_sin_c_minus_b_eq_one
  (A B C : ℝ)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  : Real.sin (C - B) = 1 := 
sorry

-- Problem 2: Proof of CD/BC given the ratios AB:AD:AC and the trigonometric identity
theorem prove_cd_div_bc_eq
  (A B C : ℝ)
  (AB AD AC BC CD : ℝ)
  (h_ratio : AB / AD = Real.sqrt 3 / Real.sqrt 2)
  (h_ratio_2 : AB / AC = Real.sqrt 3 / 1)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  (h_D_on_BC : True) -- Placeholder for D lies on BC condition
  : CD / BC = (Real.sqrt 5 - 1) / 2 := 
sorry

end NUMINAMATH_GPT_prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l118_11891


namespace NUMINAMATH_GPT_division_problem_l118_11810

theorem division_problem :
  0.045 / 0.0075 = 6 :=
sorry

end NUMINAMATH_GPT_division_problem_l118_11810


namespace NUMINAMATH_GPT_max_c_l118_11819

theorem max_c (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2 * y^2 ≤ c * x * (y - x)) 
  → c ≤ 2 * Real.sqrt 2 - 4 := 
by
  sorry

end NUMINAMATH_GPT_max_c_l118_11819


namespace NUMINAMATH_GPT_overall_percent_decrease_l118_11848

theorem overall_percent_decrease (trouser_price_italy : ℝ) (jacket_price_italy : ℝ) 
(trouser_price_uk : ℝ) (trouser_discount_uk : ℝ) (jacket_price_uk : ℝ) 
(jacket_discount_uk : ℝ) (exchange_rate : ℝ) 
(h1 : trouser_price_italy = 200) (h2 : jacket_price_italy = 150) 
(h3 : trouser_price_uk = 150) (h4 : trouser_discount_uk = 0.20) 
(h5 : jacket_price_uk = 120) (h6 : jacket_discount_uk = 0.30) 
(h7 : exchange_rate = 0.85) : 
((trouser_price_italy + jacket_price_italy) - 
 ((trouser_price_uk * (1 - trouser_discount_uk) / exchange_rate) + 
 (jacket_price_uk * (1 - jacket_discount_uk) / exchange_rate))) / 
 (trouser_price_italy + jacket_price_italy) * 100 = 31.43 := 
by 
  sorry

end NUMINAMATH_GPT_overall_percent_decrease_l118_11848


namespace NUMINAMATH_GPT_c_10_eq_3_pow_89_l118_11850

section sequence
  open Nat

  -- Define the sequence c
  def c : ℕ → ℕ
  | 0     => 3  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 9
  | (n+2) => c n.succ * c n

  -- Define the auxiliary sequence d
  def d : ℕ → ℕ
  | 0     => 1  -- Note: Typically Lean sequences start from 0, not 1
  | 1     => 2
  | (n+2) => d n.succ + d n

  -- The theorem we need to prove
  theorem c_10_eq_3_pow_89 : c 9 = 3 ^ d 9 :=    -- Note: c_{10} in the original problem is c(9) in Lean
  sorry   -- Proof omitted
end sequence

end NUMINAMATH_GPT_c_10_eq_3_pow_89_l118_11850


namespace NUMINAMATH_GPT_syllogism_correct_l118_11890

theorem syllogism_correct 
  (natnum : ℕ → Prop) 
  (intnum : ℤ → Prop) 
  (is_natnum  : natnum 4) 
  (natnum_to_intnum : ∀ n, natnum n → intnum n) : intnum 4 :=
by
  sorry

end NUMINAMATH_GPT_syllogism_correct_l118_11890


namespace NUMINAMATH_GPT_abs_diff_of_solutions_eq_5_point_5_l118_11842

theorem abs_diff_of_solutions_eq_5_point_5 (x y : ℝ)
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.7)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 8.2) :
  |x - y| = 5.5 :=
sorry

end NUMINAMATH_GPT_abs_diff_of_solutions_eq_5_point_5_l118_11842


namespace NUMINAMATH_GPT_p_squared_plus_one_over_p_squared_plus_six_l118_11811

theorem p_squared_plus_one_over_p_squared_plus_six (p : ℝ) (h : p + 1/p = 10) : p^2 + 1/p^2 + 6 = 104 :=
by {
  sorry
}

end NUMINAMATH_GPT_p_squared_plus_one_over_p_squared_plus_six_l118_11811


namespace NUMINAMATH_GPT_power_of_fraction_to_decimal_l118_11853

theorem power_of_fraction_to_decimal : ∃ x : ℕ, (1 / 9 : ℚ) ^ x = 1 / 81 ∧ x = 2 :=
by
  use 2
  simp
  sorry

end NUMINAMATH_GPT_power_of_fraction_to_decimal_l118_11853


namespace NUMINAMATH_GPT_sum_of_angles_x_y_l118_11831

theorem sum_of_angles_x_y :
  let num_arcs := 15
  let angle_per_arc := 360 / num_arcs
  let central_angle_x := 3 * angle_per_arc
  let central_angle_y := 5 * angle_per_arc
  let inscribed_angle (central_angle : ℝ) := central_angle / 2
  let angle_x := inscribed_angle central_angle_x
  let angle_y := inscribed_angle central_angle_y
  angle_x + angle_y = 96 := 
  sorry

end NUMINAMATH_GPT_sum_of_angles_x_y_l118_11831


namespace NUMINAMATH_GPT_monotone_range_of_f_l118_11860

theorem monotone_range_of_f {f : ℝ → ℝ} (a : ℝ) 
  (h : ∀ x y : ℝ, 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x ≤ y → f x ≤ f y) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_monotone_range_of_f_l118_11860


namespace NUMINAMATH_GPT_shopkeeper_discount_problem_l118_11838

theorem shopkeeper_discount_problem (CP SP_with_discount SP_without_discount Discount : ℝ)
  (h1 : SP_with_discount = CP + 0.273 * CP)
  (h2 : SP_without_discount = CP + 0.34 * CP) :
  Discount = SP_without_discount - SP_with_discount →
  (Discount / SP_without_discount) * 100 = 5 := 
sorry

end NUMINAMATH_GPT_shopkeeper_discount_problem_l118_11838


namespace NUMINAMATH_GPT_payment_to_N_l118_11808

variable (x : ℝ)

/-- Conditions stating the total payment and the relationship between M and N's payment --/
axiom total_payment : x + 1.20 * x = 550

/-- Statement to prove the amount paid to N per week --/
theorem payment_to_N : x = 250 :=
by
  sorry

end NUMINAMATH_GPT_payment_to_N_l118_11808


namespace NUMINAMATH_GPT_joshua_share_is_30_l118_11833

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end NUMINAMATH_GPT_joshua_share_is_30_l118_11833


namespace NUMINAMATH_GPT_circle_center_and_radius_l118_11843

theorem circle_center_and_radius:
  ∀ x y : ℝ, 
  (x + 1) ^ 2 + (y - 3) ^ 2 = 36 
  → ∃ C : (ℝ × ℝ), C = (-1, 3) ∧ ∃ r : ℝ, r = 6 := sorry

end NUMINAMATH_GPT_circle_center_and_radius_l118_11843


namespace NUMINAMATH_GPT_parabola_shifts_down_decrease_c_real_roots_l118_11813

-- The parabolic function and conditions
variables {a b c k : ℝ}

-- Assumption that a is positive
axiom ha : a > 0

-- Parabola shifts down when constant term c is decreased
theorem parabola_shifts_down (c : ℝ) (k : ℝ) (hk : k > 0) :
  ∀ x, (a * x^2 + b * x + (c - k)) = (a * x^2 + b * x + c) - k :=
by sorry

-- Discriminant of quadratic equation ax^2 + bx + c = 0
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- If the discriminant is negative, decreasing c can result in real roots
theorem decrease_c_real_roots (b c : ℝ) (hb : b^2 < 4 * a * c) (k : ℝ) (hk : k > 0) :
  discriminant a b (c - k) ≥ 0 :=
by sorry

end NUMINAMATH_GPT_parabola_shifts_down_decrease_c_real_roots_l118_11813


namespace NUMINAMATH_GPT_S_equals_x4_l118_11829

-- Define the expression for S
def S (x : ℝ) : ℝ := (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * x - 3

-- State the theorem to be proved
theorem S_equals_x4 (x : ℝ) : S x = x^4 :=
by
  sorry

end NUMINAMATH_GPT_S_equals_x4_l118_11829


namespace NUMINAMATH_GPT_haleigh_cats_l118_11889

open Nat

def total_pairs := 14
def dog_leggings := 4
def legging_per_animal := 1

theorem haleigh_cats : ∀ (dogs cats : ℕ), 
  dogs = 4 → 
  total_pairs = dogs * legging_per_animal + cats * legging_per_animal → 
  cats = 10 :=
by
  intros dogs cats h1 h2
  sorry

end NUMINAMATH_GPT_haleigh_cats_l118_11889


namespace NUMINAMATH_GPT_original_average_rent_l118_11881

theorem original_average_rent
    (A : ℝ) -- original average rent per person
    (h1 : 4 * A + 200 = 3400) -- condition derived from the rent problem
    : A = 800 := 
sorry

end NUMINAMATH_GPT_original_average_rent_l118_11881


namespace NUMINAMATH_GPT_num_license_plates_l118_11835

-- Let's state the number of letters in the alphabet, vowels, consonants, and digits.
def num_letters : ℕ := 26
def num_vowels : ℕ := 5  -- A, E, I, O, U and Y is not a vowel
def num_consonants : ℕ := 21  -- Remaining letters including Y
def num_digits : ℕ := 10  -- 0 through 9

-- Prove the number of five-character license plates
theorem num_license_plates : 
  (num_consonants * num_consonants * num_vowels * num_vowels * num_digits) = 110250 :=
  by 
  sorry

end NUMINAMATH_GPT_num_license_plates_l118_11835


namespace NUMINAMATH_GPT_find_a_range_l118_11800

noncomputable def f (x : ℝ) := (x - 1) / Real.exp x

noncomputable def condition_holds (a : ℝ) : Prop :=
∀ t ∈ (Set.Icc (1/2 : ℝ) 2), f t > t

theorem find_a_range (a : ℝ) (h : condition_holds a) : a > Real.exp 2 + 1/2 := sorry

end NUMINAMATH_GPT_find_a_range_l118_11800


namespace NUMINAMATH_GPT_exists_integer_a_l118_11844

theorem exists_integer_a (p : ℕ) (hp : p ≥ 5) [Fact (Nat.Prime p)] : 
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ p^2 ∣ a^(p-1) - 1) ∧ (¬ p^2 ∣ (a+1)^(p-1) - 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_integer_a_l118_11844


namespace NUMINAMATH_GPT_n_times_2pow_nplus1_plus_1_is_square_l118_11821

theorem n_times_2pow_nplus1_plus_1_is_square (n : ℕ) (h : 0 < n) :
  ∃ m : ℤ, n * 2 ^ (n + 1) + 1 = m * m ↔ n = 3 := 
by
  sorry

end NUMINAMATH_GPT_n_times_2pow_nplus1_plus_1_is_square_l118_11821


namespace NUMINAMATH_GPT_remainder_modulo_seven_l118_11837

theorem remainder_modulo_seven (n : ℕ)
  (h₁ : n^2 % 7 = 1)
  (h₂ : n^3 % 7 = 6) :
  n % 7 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_modulo_seven_l118_11837


namespace NUMINAMATH_GPT_total_pies_sold_l118_11899

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end NUMINAMATH_GPT_total_pies_sold_l118_11899


namespace NUMINAMATH_GPT_angle_B_is_30_degrees_l118_11887

variable (a b : ℝ)
variable (A B : ℝ)

axiom a_value : a = 2 * Real.sqrt 3
axiom b_value : b = Real.sqrt 6
axiom A_value : A = Real.pi / 4

theorem angle_B_is_30_degrees (h1 : a = 2 * Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = Real.pi / 4) : B = Real.pi / 6 :=
  sorry

end NUMINAMATH_GPT_angle_B_is_30_degrees_l118_11887


namespace NUMINAMATH_GPT_tangent_of_curve_at_point_l118_11894

def curve (x : ℝ) : ℝ := x^3 - 4 * x

def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

theorem tangent_of_curve_at_point : 
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ tangent_line x y) :=
sorry

end NUMINAMATH_GPT_tangent_of_curve_at_point_l118_11894


namespace NUMINAMATH_GPT_square_area_from_circle_area_l118_11856

variable (square_area : ℝ) (circle_area : ℝ)

theorem square_area_from_circle_area 
  (h1 : circle_area = 9 * Real.pi) 
  (h2 : square_area = (2 * Real.sqrt (circle_area / Real.pi))^2) : 
  square_area = 36 := 
by
  sorry

end NUMINAMATH_GPT_square_area_from_circle_area_l118_11856


namespace NUMINAMATH_GPT_museum_college_students_income_l118_11877

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end NUMINAMATH_GPT_museum_college_students_income_l118_11877


namespace NUMINAMATH_GPT_percentage_of_x_l118_11855

theorem percentage_of_x (x : ℝ) (h : x > 0) : ((x / 5 + x / 25) / x) * 100 = 24 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_x_l118_11855


namespace NUMINAMATH_GPT_sets_relationship_l118_11876

variables {U : Type*} (A B C : Set U)

theorem sets_relationship (h1 : A ∩ B = C) (h2 : B ∩ C = A) : A = C ∧ ∃ B, A ⊆ B := by
  sorry

end NUMINAMATH_GPT_sets_relationship_l118_11876


namespace NUMINAMATH_GPT_oak_trees_initial_count_l118_11851

theorem oak_trees_initial_count (x : ℕ) (cut_down : ℕ) (remaining : ℕ) (h_cut : cut_down = 2) (h_remaining : remaining = 7)
  (h_equation : (x - cut_down) = remaining) : x = 9 := by
  -- We are given that cut_down = 2
  -- and remaining = 7
  -- and we need to show that the initial count x = 9
  sorry

end NUMINAMATH_GPT_oak_trees_initial_count_l118_11851


namespace NUMINAMATH_GPT_proof_ab_greater_ac_l118_11847

theorem proof_ab_greater_ac (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  a * b > a * c :=
by sorry

end NUMINAMATH_GPT_proof_ab_greater_ac_l118_11847


namespace NUMINAMATH_GPT_max_gcd_expression_l118_11893

theorem max_gcd_expression (n : ℕ) (h1 : n > 0) (h2 : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_expression_l118_11893


namespace NUMINAMATH_GPT_polar_line_eq_l118_11857

theorem polar_line_eq (ρ θ : ℝ) : (ρ * Real.cos θ = 1) ↔ (ρ = Real.cos θ ∨ ρ = Real.sin θ ∨ 1 / Real.cos θ = ρ) := by
  sorry

end NUMINAMATH_GPT_polar_line_eq_l118_11857


namespace NUMINAMATH_GPT_max_ab_l118_11832

theorem max_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
sorry

end NUMINAMATH_GPT_max_ab_l118_11832


namespace NUMINAMATH_GPT_g_value_at_50_l118_11826

noncomputable def g : ℝ → ℝ :=
sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - y ^ 2 * g x = g (x / y)) :
  g 50 = 0 :=
by
  sorry

end NUMINAMATH_GPT_g_value_at_50_l118_11826


namespace NUMINAMATH_GPT_min_value_fraction_l118_11840

theorem min_value_fraction (x : ℝ) (h : x > 0) : ∃ y, y = 4 ∧ (∀ z, z = (x + 5) / Real.sqrt (x + 1) → y ≤ z) := sorry

end NUMINAMATH_GPT_min_value_fraction_l118_11840


namespace NUMINAMATH_GPT_value_of_a3_l118_11871

theorem value_of_a3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (a : ℝ) (h₀ : (1 + x) * (a - x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) 
(h₁ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) : 
a₃ = -5 :=
sorry

end NUMINAMATH_GPT_value_of_a3_l118_11871


namespace NUMINAMATH_GPT_least_value_y_l118_11809

theorem least_value_y
  (h : ∀ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 → -3 ≤ y) : 
  ∃ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 ∧ y = -3 :=
by
  sorry

end NUMINAMATH_GPT_least_value_y_l118_11809


namespace NUMINAMATH_GPT_digit_150_of_1_over_13_is_3_l118_11888

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end NUMINAMATH_GPT_digit_150_of_1_over_13_is_3_l118_11888


namespace NUMINAMATH_GPT_sandwich_cost_is_5_l118_11884

-- We define the variables and conditions first
def total_people := 4
def sandwiches := 4
def fruit_salads := 4
def sodas := 8
def snack_bags := 3

def fruit_salad_cost_per_unit := 3
def soda_cost_per_unit := 2
def snack_bag_cost_per_unit := 4
def total_cost := 60

-- We now define the calculations based on the given conditions
def total_fruit_salad_cost := fruit_salads * fruit_salad_cost_per_unit
def total_soda_cost := sodas * soda_cost_per_unit
def total_snack_bag_cost := snack_bags * snack_bag_cost_per_unit
def other_items_cost := total_fruit_salad_cost + total_soda_cost + total_snack_bag_cost
def remaining_budget := total_cost - other_items_cost
def sandwich_cost := remaining_budget / sandwiches

-- The final proof problem statement in Lean 4
theorem sandwich_cost_is_5 : sandwich_cost = 5 := by
  sorry

end NUMINAMATH_GPT_sandwich_cost_is_5_l118_11884


namespace NUMINAMATH_GPT_eliza_height_is_68_l118_11864

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end NUMINAMATH_GPT_eliza_height_is_68_l118_11864


namespace NUMINAMATH_GPT_percentage_error_in_area_l118_11849

noncomputable def side_with_error (s : ℝ) : ℝ := 1.04 * s

noncomputable def actual_area (s : ℝ) : ℝ := s ^ 2

noncomputable def calculated_area (s : ℝ) : ℝ := (side_with_error s) ^ 2

noncomputable def percentage_error (actual : ℝ) (calculated : ℝ) : ℝ :=
  ((calculated - actual) / actual) * 100

theorem percentage_error_in_area (s : ℝ) :
  percentage_error (actual_area s) (calculated_area s) = 8.16 := by
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l118_11849


namespace NUMINAMATH_GPT_points_on_circle_l118_11883

theorem points_on_circle (t : ℝ) : ∃ x y : ℝ, x = Real.cos t ∧ y = Real.sin t ∧ x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_points_on_circle_l118_11883


namespace NUMINAMATH_GPT_solution_to_inequality_l118_11892

theorem solution_to_inequality : 
  ∀ x : ℝ, (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solution_to_inequality_l118_11892


namespace NUMINAMATH_GPT_certain_number_l118_11898

theorem certain_number (x y : ℕ) (h₁ : x = 14) (h₂ : 2^x - 2^(x - 2) = 3 * 2^y) : y = 12 :=
  by
  sorry

end NUMINAMATH_GPT_certain_number_l118_11898


namespace NUMINAMATH_GPT_oliver_baths_per_week_l118_11852

-- Define all the conditions given in the problem
def bucket_capacity : ℕ := 120
def num_buckets_to_fill_tub : ℕ := 14
def num_buckets_removed : ℕ := 3
def weekly_water_usage : ℕ := 9240

-- Calculate total water to fill bathtub, water removed, water used per bath, and baths per week
def total_tub_capacity : ℕ := num_buckets_to_fill_tub * bucket_capacity
def water_removed : ℕ := num_buckets_removed * bucket_capacity
def water_per_bath : ℕ := total_tub_capacity - water_removed
def baths_per_week : ℕ := weekly_water_usage / water_per_bath

theorem oliver_baths_per_week : baths_per_week = 7 := by
  sorry

end NUMINAMATH_GPT_oliver_baths_per_week_l118_11852


namespace NUMINAMATH_GPT_first_year_after_2020_with_digit_sum_18_l118_11824

theorem first_year_after_2020_with_digit_sum_18 : 
  ∃ (y : ℕ), y > 2020 ∧ (∃ a b c : ℕ, (2 + a + b + c = 18 ∧ y = 2000 + 100 * a + 10 * b + c)) ∧ y = 2799 := 
sorry

end NUMINAMATH_GPT_first_year_after_2020_with_digit_sum_18_l118_11824


namespace NUMINAMATH_GPT_no_real_roots_iff_no_positive_discriminant_l118_11816

noncomputable def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c

theorem no_real_roots_iff_no_positive_discriminant (m : ℝ) 
  (h : discriminant m (-2*(m+2)) (m+5) < 0) : 
  (discriminant (m-5) (-2*(m+2)) m < 0 ∨ discriminant (m-5) (-2*(m+2)) m > 0 ∨ m - 5 = 0) :=
by 
  sorry

end NUMINAMATH_GPT_no_real_roots_iff_no_positive_discriminant_l118_11816


namespace NUMINAMATH_GPT_total_lives_l118_11825

theorem total_lives (initial_friends : ℕ) (initial_lives_per_friend : ℕ) (additional_players : ℕ) (lives_per_new_player : ℕ) :
  initial_friends = 7 →
  initial_lives_per_friend = 7 →
  additional_players = 2 →
  lives_per_new_player = 7 →
  (initial_friends * initial_lives_per_friend + additional_players * lives_per_new_player) = 63 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_lives_l118_11825


namespace NUMINAMATH_GPT_total_amount_spent_l118_11874

variable (you friend : ℝ)

theorem total_amount_spent (h1 : friend = you + 3) (h2 : friend = 7) : 
  you + friend = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l118_11874


namespace NUMINAMATH_GPT_students_passed_both_tests_l118_11822

theorem students_passed_both_tests
    (total_students : ℕ)
    (passed_long_jump : ℕ)
    (passed_shot_put : ℕ)
    (failed_both : ℕ)
    (h_total : total_students = 50)
    (h_long_jump : passed_long_jump = 40)
    (h_shot_put : passed_shot_put = 31)
    (h_failed_both : failed_both = 4) : 
    (total_students - failed_both = passed_long_jump + passed_shot_put - 25) :=
by 
  sorry

end NUMINAMATH_GPT_students_passed_both_tests_l118_11822


namespace NUMINAMATH_GPT_water_evaporation_problem_l118_11820

theorem water_evaporation_problem 
  (W : ℝ) 
  (evaporation_rate : ℝ := 0.01) 
  (evaporation_days : ℝ := 20) 
  (total_evaporation : ℝ := evaporation_rate * evaporation_days) 
  (evaporation_percentage : ℝ := 0.02) 
  (evaporation_amount : ℝ := evaporation_percentage * W) :
  evaporation_amount = total_evaporation → W = 10 :=
by
  sorry

end NUMINAMATH_GPT_water_evaporation_problem_l118_11820


namespace NUMINAMATH_GPT_total_cost_paid_l118_11817

-- Definition of the given conditions
def number_of_DVDs : ℕ := 4
def cost_per_DVD : ℝ := 1.2

-- The theorem to be proven
theorem total_cost_paid : number_of_DVDs * cost_per_DVD = 4.8 := by
  sorry

end NUMINAMATH_GPT_total_cost_paid_l118_11817
