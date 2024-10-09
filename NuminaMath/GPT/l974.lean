import Mathlib

namespace correct_weights_l974_97455

def weight (item : String) : Nat :=
  match item with
  | "Banana" => 140
  | "Pear" => 120
  | "Melon" => 1500
  | "Tomato" => 150
  | "Apple" => 170
  | _ => 0

theorem correct_weights :
  weight "Banana" = 140 ∧
  weight "Pear" = 120 ∧
  weight "Melon" = 1500 ∧
  weight "Tomato" = 150 ∧
  weight "Apple" = 170 ∧
  (weight "Melon" > weight "Pear") ∧
  (weight "Melon" < weight "Tomato") :=
by
  sorry

end correct_weights_l974_97455


namespace f_1988_eq_1988_l974_97417

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_f_eq_add (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : f (f m + f n) = m + n

theorem f_1988_eq_1988 : f 1988 = 1988 := 
by
  sorry

end f_1988_eq_1988_l974_97417


namespace problem1_problem2_l974_97429

-- Define the sets P and Q
def set_P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def set_Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Problem (1): P ∩ Q = Q implies a ∈ (-1/2, 2)
theorem problem1 (a : ℝ) : (set_Q a) ⊆ set_P → -1/2 < a ∧ a < 2 :=
by 
  sorry

-- Problem (2): P ∩ Q = ∅ implies a ∈ (-∞, -3/2] ∪ [3, ∞)
theorem problem2 (a : ℝ) : (set_Q a) ∩ set_P = ∅ → a ≤ -3/2 ∨ a ≥ 3 :=
by 
  sorry

end problem1_problem2_l974_97429


namespace gasoline_used_by_car_l974_97412

noncomputable def total_gasoline_used (gasoline_per_km : ℝ) (duration_hours : ℝ) (speed_kmh : ℝ) : ℝ :=
  gasoline_per_km * duration_hours * speed_kmh

theorem gasoline_used_by_car :
  total_gasoline_used 0.14 (2 + 0.5) 93.6 = 32.76 := sorry

end gasoline_used_by_car_l974_97412


namespace value_of_x_l974_97434

theorem value_of_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end value_of_x_l974_97434


namespace mn_plus_one_unequal_pos_integers_l974_97449

theorem mn_plus_one_unequal_pos_integers (m n : ℕ) 
  (S : Finset ℕ) (h_card : S.card = m * n + 1) :
  (∃ (b : Fin (m + 1) → ℕ), (∀ i j : Fin (m + 1), i ≠ j → ¬(b i ∣ b j)) ∧ (∀ i : Fin (m + 1), b i ∈ S)) ∨ 
  (∃ (a : Fin (n + 1) → ℕ), (∀ i : Fin n, a i ∣ a (i + 1)) ∧ (∀ i : Fin (n + 1), a i ∈ S)) :=
sorry

end mn_plus_one_unequal_pos_integers_l974_97449


namespace regular_price_of_one_tire_l974_97421

theorem regular_price_of_one_tire
  (x : ℝ) -- Define the variable \( x \) as the regular price of one tire
  (h1 : 3 * x + 10 = 250) -- Set up the equation based on the condition

  : x = 80 := 
sorry

end regular_price_of_one_tire_l974_97421


namespace find_a_l974_97409

-- Define the sets A and B based on the conditions
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, a ^ 2 + 1, 2 * a - 1}

-- Statement: Prove that a = -1 satisfies the condition A ∩ B = {-3}
theorem find_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by
  sorry

end find_a_l974_97409


namespace janet_wait_time_l974_97472

theorem janet_wait_time
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (janet_time : ℝ)
  (sister_time : ℝ) :
  janet_speed = 30 →
  sister_speed = 12 →
  lake_width = 60 →
  janet_time = lake_width / janet_speed →
  sister_time = lake_width / sister_speed →
  (sister_time - janet_time = 3) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end janet_wait_time_l974_97472


namespace selling_price_l974_97497

theorem selling_price (cost_price profit_percentage : ℝ) (h1 : cost_price = 90) (h2 : profit_percentage = 100) : 
    cost_price + (profit_percentage * cost_price / 100) = 180 :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end selling_price_l974_97497


namespace number_of_bowls_l974_97441

theorem number_of_bowls (n : ℕ) :
  (∀ (b : ℕ), b > 0) →
  (∀ (a : ℕ), ∃ (k : ℕ), true) →
  (8 * 12 = 96) →
  (6 * n = 96) →
  n = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end number_of_bowls_l974_97441


namespace max_groups_l974_97432

-- Define the conditions
def valid_eq (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ (3 * a + b = 13)

-- The proof problem: No need for the proof body, just statement
theorem max_groups : ∃! (l : List (ℕ × ℕ)), (∀ ab ∈ l, valid_eq ab.fst ab.snd) ∧ l.length = 3 := sorry

end max_groups_l974_97432


namespace find_positive_integers_n_satisfying_equation_l974_97481

theorem find_positive_integers_n_satisfying_equation :
  ∀ x y z : ℕ,
  x > 0 → y > 0 → z > 0 →
  (x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) →
  (n = 1 ∨ n = 3) :=
by
  sorry

end find_positive_integers_n_satisfying_equation_l974_97481


namespace gcd_seq_finitely_many_values_l974_97467

def gcd_seq_finite_vals (A B : ℕ) (x : ℕ → ℕ) : Prop :=
  (∀ n ≥ 2, x (n + 1) = A * Nat.gcd (x n) (x (n-1)) + B) →
  ∃ N : ℕ, ∀ m n, m ≥ N → n ≥ N → x m = x n

theorem gcd_seq_finitely_many_values (A B : ℕ) (x : ℕ → ℕ) :
  gcd_seq_finite_vals A B x :=
by
  intros h
  sorry

end gcd_seq_finitely_many_values_l974_97467


namespace number_of_combinations_l974_97446

-- Define the binomial coefficient (combinations) function
def C (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Our main theorem statement
theorem number_of_combinations (n k m : ℕ) (h1 : 1 ≤ n) (h2 : m > 1) :
  let valid_combinations := C (n - (k - 1) * (m - 1)) k;
  let invalid_combinations := n - (k - 1) * m;
  valid_combinations - invalid_combinations = 
  C (n - (k - 1) * (m - 1)) k - (n - (k - 1) * m) := by
  let valid_combinations := C (n - (k - 1) * (m - 1)) k
  let invalid_combinations := n - (k - 1) * m
  sorry

end number_of_combinations_l974_97446


namespace outOfPocketCost_l974_97406

noncomputable def visitCost : ℝ := 300
noncomputable def castCost : ℝ := 200
noncomputable def insuranceCoverage : ℝ := 0.60

theorem outOfPocketCost : (visitCost + castCost - (visitCost + castCost) * insuranceCoverage) = 200 := by
  sorry

end outOfPocketCost_l974_97406


namespace inequality_pqr_l974_97468

theorem inequality_pqr (p q r : ℝ) (n : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p * q * r = 1) :
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 :=
sorry

end inequality_pqr_l974_97468


namespace probability_sum_of_two_draws_is_three_l974_97405

theorem probability_sum_of_two_draws_is_three :
  let outcomes := [(1, 1), (1, 2), (2, 1), (2, 2)]
  let favorable := [(1, 2), (2, 1)]
  (favorable.length : ℚ) / (outcomes.length : ℚ) = 1 / 2 :=
by
  sorry

end probability_sum_of_two_draws_is_three_l974_97405


namespace equation1_solution_equation2_solution_l974_97433

theorem equation1_solution : ∀ x : ℚ, x - 0.4 * x = 120 → x = 200 := by
  sorry

theorem equation2_solution : ∀ x : ℚ, 5 * x - 5/6 = 5/4 → x = 5/12 := by
  sorry

end equation1_solution_equation2_solution_l974_97433


namespace lines_perpendicular_if_one_perpendicular_and_one_parallel_l974_97451

def Line : Type := sorry  -- Define the type representing lines
def Plane : Type := sorry  -- Define the type representing planes

def is_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry  -- Definition for a line being perpendicular to a plane
def is_parallel_to_plane (b : Line) (α : Plane) : Prop := sorry  -- Definition for a line being parallel to a plane
def is_perpendicular (a b : Line) : Prop := sorry  -- Definition for a line being perpendicular to another line

theorem lines_perpendicular_if_one_perpendicular_and_one_parallel 
  (a b : Line) (α : Plane) 
  (h1 : is_perpendicular_to_plane a α) 
  (h2 : is_parallel_to_plane b α) : 
  is_perpendicular a b := 
sorry

end lines_perpendicular_if_one_perpendicular_and_one_parallel_l974_97451


namespace find_temperature_l974_97460

theorem find_temperature 
  (temps : List ℤ)
  (h_len : temps.length = 8)
  (h_mean : (temps.sum / 8 : ℝ) = -0.5)
  (h_temps : temps = [-6, -3, x, -6, 2, 4, 3, 0]) : 
  x = 2 :=
by 
  sorry

end find_temperature_l974_97460


namespace lopez_family_seating_arrangement_count_l974_97465

def lopez_family_seating_arrangements : Nat := 2 * 4 * 6

theorem lopez_family_seating_arrangement_count : lopez_family_seating_arrangements = 48 :=
by 
    sorry

end lopez_family_seating_arrangement_count_l974_97465


namespace moles_of_KOH_combined_l974_97413

theorem moles_of_KOH_combined 
  (moles_NH4Cl : ℕ)
  (moles_KCl : ℕ)
  (balanced_reaction : ℕ → ℕ → ℕ)
  (h_NH4Cl : moles_NH4Cl = 3)
  (h_KCl : moles_KCl = 3)
  (reaction_ratio : ∀ n, balanced_reaction n n = n) :
  balanced_reaction moles_NH4Cl moles_KCl = 3 * balanced_reaction 1 1 := 
by
  sorry

end moles_of_KOH_combined_l974_97413


namespace solve_quadratic_eq_l974_97431

theorem solve_quadratic_eq (x : ℝ) (h : (x + 5) ^ 2 = 16) : x = -1 ∨ x = -9 :=
sorry

end solve_quadratic_eq_l974_97431


namespace bears_total_l974_97420

-- Define the number of each type of bear
def brown_bears : ℕ := 15
def white_bears : ℕ := 24
def black_bears : ℕ := 27
def polar_bears : ℕ := 12
def grizzly_bears : ℕ := 18

-- Define the total number of bears
def total_bears : ℕ := brown_bears + white_bears + black_bears + polar_bears + grizzly_bears

-- The theorem stating the total number of bears is 96
theorem bears_total : total_bears = 96 :=
by
  -- The proof is omitted here
  sorry

end bears_total_l974_97420


namespace fly_least_distance_l974_97478

noncomputable def least_distance_fly_crawled (radius height dist_start dist_end : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let slant_height := Real.sqrt (radius^2 + height^2)
  let angle := circumference / slant_height
  let half_angle := angle / 2
  let start_x := dist_start
  let end_x := dist_end * Real.cos half_angle
  let end_y := dist_end * Real.sin half_angle
  Real.sqrt ((end_x - start_x)^2 + end_y^2)

theorem fly_least_distance : least_distance_fly_crawled 500 (300 * Real.sqrt 3) 150 (450 * Real.sqrt 2) = 486.396 := by
  sorry

end fly_least_distance_l974_97478


namespace solution_set_l974_97419

variable {f : ℝ → ℝ}

-- Define that f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define that f is decreasing on positive reals
def decreasing_on_pos_reals (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- Given conditions
axiom f_odd : odd_function f
axiom f_decreasing : decreasing_on_pos_reals f
axiom f_at_two_zero : f 2 = 0

-- Main theorem statement
theorem solution_set : { x : ℝ | (x - 1) * f (x - 1) > 0 } = { x | x < -1 } ∪ { x | x > 3 } :=
sorry

end solution_set_l974_97419


namespace lottery_probability_correct_l974_97486

noncomputable def probability_winning_lottery : ℚ :=
  let starBall_probability := 1 / 30
  let combinations (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  let magicBalls_probability := 1 / (combinations 49 6)
  starBall_probability * magicBalls_probability

theorem lottery_probability_correct :
  probability_winning_lottery = 1 / 419514480 := by
  sorry

end lottery_probability_correct_l974_97486


namespace time_for_a_to_complete_one_round_l974_97401

theorem time_for_a_to_complete_one_round (T_a T_b : ℝ) 
  (h1 : 4 * T_a = 3 * T_b)
  (h2 : T_b = T_a + 10) : 
  T_a = 30 := by
  sorry

end time_for_a_to_complete_one_round_l974_97401


namespace melanie_attended_games_l974_97430

-- Define the total number of football games and the number of games missed by Melanie.
def total_games := 7
def missed_games := 4

-- Define what we need to prove: the number of games attended by Melanie.
theorem melanie_attended_games : total_games - missed_games = 3 := 
by
  sorry

end melanie_attended_games_l974_97430


namespace determine_n_l974_97407

open Function

noncomputable def coeff_3 (n : ℕ) : ℕ :=
  2^(n-2) * Nat.choose n 2

noncomputable def coeff_4 (n : ℕ) : ℕ :=
  2^(n-3) * Nat.choose n 3

theorem determine_n (n : ℕ) (b3_eq_2b4 : coeff_3 n = 2 * coeff_4 n) : n = 5 :=
  sorry

end determine_n_l974_97407


namespace simplify_and_evaluate_expression_l974_97484

variable (a : ℚ)

theorem simplify_and_evaluate_expression (h : a = -1/3) : 
  (a + 1) * (a - 1) - a * (a + 3) = 0 := 
by
  sorry

end simplify_and_evaluate_expression_l974_97484


namespace angle_F_measure_l974_97443

theorem angle_F_measure (D E F : ℝ) (h₁ : D = 80) (h₂ : E = 2 * F + 24) (h₃ : D + E + F = 180) : F = 76 / 3 :=
by
  sorry

end angle_F_measure_l974_97443


namespace subtraction_addition_example_l974_97463

theorem subtraction_addition_example :
  1500000000000 - 877888888888 + 123456789012 = 745567900124 :=
by
  sorry

end subtraction_addition_example_l974_97463


namespace solve_integer_divisibility_l974_97445

theorem solve_integer_divisibility :
  {n : ℕ | n < 589 ∧ 589 ∣ (n^2 + n + 1)} = {49, 216, 315, 482} :=
by
  sorry

end solve_integer_divisibility_l974_97445


namespace polio_cases_in_1990_l974_97425

theorem polio_cases_in_1990 (c_1970 c_2000 : ℕ) (T : ℕ) (linear_decrease : ∀ t, c_1970 - (c_2000 * t) / T > 0):
  (c_1970 = 300000) → (c_2000 = 600) → (T = 30) → ∃ c_1990, c_1990 = 100400 :=
by
  intros
  sorry

end polio_cases_in_1990_l974_97425


namespace books_left_over_l974_97493

theorem books_left_over (boxes : ℕ) (books_per_box_initial : ℕ) (books_per_box_new: ℕ) (total_books : ℕ) :
  boxes = 1500 →
  books_per_box_initial = 45 →
  books_per_box_new = 47 →
  total_books = boxes * books_per_box_initial →
  (total_books % books_per_box_new) = 8 :=
by intros; sorry

end books_left_over_l974_97493


namespace equivalent_annual_rate_8_percent_quarterly_is_8_24_l974_97480

noncomputable def quarterly_interest_rate (annual_rate : ℚ) := annual_rate / 4

noncomputable def growth_factor (interest_rate : ℚ) := 1 + interest_rate / 100

noncomputable def annual_growth_factor_from_quarterly (quarterly_factor : ℚ) := quarterly_factor ^ 4

noncomputable def equivalent_annual_interest_rate (annual_growth_factor : ℚ) := 
  ((annual_growth_factor - 1) * 100)

theorem equivalent_annual_rate_8_percent_quarterly_is_8_24 :
  let quarter_rate := quarterly_interest_rate 8
  let quarterly_factor := growth_factor quarter_rate
  let annual_factor := annual_growth_factor_from_quarterly quarterly_factor
  equivalent_annual_interest_rate annual_factor = 8.24 := by
  sorry

end equivalent_annual_rate_8_percent_quarterly_is_8_24_l974_97480


namespace fouad_age_l974_97408

theorem fouad_age (F : ℕ) (Ahmed_current_age : ℕ) (H : Ahmed_current_age = 11) (H2 : F + 4 = 2 * Ahmed_current_age) : F = 18 :=
by
  -- We do not need to write the proof steps, just a placeholder.
  sorry

end fouad_age_l974_97408


namespace attendance_rate_correct_l974_97414

def total_students : ℕ := 50
def students_on_leave : ℕ := 2
def given_attendance_rate : ℝ := 96

theorem attendance_rate_correct :
  ((total_students - students_on_leave) / total_students * 100 : ℝ) = given_attendance_rate := sorry

end attendance_rate_correct_l974_97414


namespace exists_linear_function_l974_97400

-- Define the properties of the function f
def is_contraction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f x - f y| ≤ |x - y|

-- Define the property of an arithmetic progression
def is_arith_seq (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ d : ℝ, ∀ n : ℕ, (f^[n] x) = x + n * d

-- Main theorem to prove
theorem exists_linear_function (f : ℝ → ℝ) (h1 : is_contraction f) (h2 : is_arith_seq f) : ∃ a : ℝ, ∀ x : ℝ, f x = x + a :=
sorry

end exists_linear_function_l974_97400


namespace minimum_sum_sequence_l974_97437

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 49

noncomputable def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

theorem minimum_sum_sequence : ∃ n : ℕ, S_n n = (n - 24) * (n - 24) - 24 * 24 ∧ (∀ m : ℕ, S_n m ≥ S_n n) ∧ n = 24 := 
by {
  sorry -- Proof omitted
}

end minimum_sum_sequence_l974_97437


namespace b5b9_l974_97474

-- Assuming the sequences are indexed from natural numbers starting at 1
-- a_n is an arithmetic sequence with common difference d
-- b_n is a geometric sequence
-- Given conditions
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry
def d : ℝ := sorry
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) - a n = d
axiom d_nonzero : d ≠ 0
axiom condition_arith : 2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0
axiom geometric_seq : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1
axiom b7_equals_a7 : b 7 = a 7

-- To prove
theorem b5b9 : b 5 * b 9 = 16 :=
by
  sorry

end b5b9_l974_97474


namespace find_positive_real_solution_l974_97485

theorem find_positive_real_solution (x : ℝ) (h : 0 < x) :
  (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 75 * x - 15) * (x ^ 2 + 40 * x + 8) →
  x = (75 + Real.sqrt (75 ^ 2 + 4 * 13)) / 2 ∨ x = (-40 + Real.sqrt (40 ^ 2 - 4 * 7)) / 2 :=
by
  sorry

end find_positive_real_solution_l974_97485


namespace proof_y_solves_diff_eqn_l974_97494

noncomputable def y (x : ℝ) : ℝ := Real.exp (2 * x)

theorem proof_y_solves_diff_eqn : ∀ x : ℝ, (deriv^[3] y x) - 8 * y x = 0 := by
  sorry

end proof_y_solves_diff_eqn_l974_97494


namespace ned_defuse_time_l974_97418

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l974_97418


namespace temperature_on_Friday_l974_97416

-- Define the temperatures for each day
variables (M T W Th F : ℕ)

-- Declare the given conditions as assumptions
axiom cond1 : (M + T + W + Th) / 4 = 48
axiom cond2 : (T + W + Th + F) / 4 = 46
axiom cond3 : M = 40

-- State the theorem
theorem temperature_on_Friday : F = 32 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end temperature_on_Friday_l974_97416


namespace range_of_x_l974_97415

def y_function (x : ℝ) : ℝ := x

def y_translated (x : ℝ) : ℝ := x + 2

theorem range_of_x {x : ℝ} (h : y_translated x > 0) : x > -2 := 
by {
  sorry
}

end range_of_x_l974_97415


namespace possible_values_of_cubes_l974_97490

noncomputable def matrix_N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![y, z, x], ![z, x, y]]

def related_conditions (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ) : Prop :=
  N^2 = -1 ∧ x * y * z = -1

theorem possible_values_of_cubes (x y z : ℂ) (N : Matrix (Fin 3) (Fin 3) ℂ)
  (hc1 : matrix_N x y z = N) (hc2 : related_conditions x y z N) :
  ∃ w : ℂ, w = x^3 + y^3 + z^3 ∧ (w = -3 + Complex.I ∨ w = -3 - Complex.I) :=
by
  sorry

end possible_values_of_cubes_l974_97490


namespace find_product_of_two_numbers_l974_97456

theorem find_product_of_two_numbers (a b : ℚ) (h1 : a + b = 7) (h2 : a - b = 2) : 
  a * b = 11 + 1/4 := 
by 
  sorry

end find_product_of_two_numbers_l974_97456


namespace fraction_of_A_or_B_l974_97483

def fraction_A : ℝ := 0.7
def fraction_B : ℝ := 0.2

theorem fraction_of_A_or_B : fraction_A + fraction_B = 0.9 := 
by
  sorry

end fraction_of_A_or_B_l974_97483


namespace grasshopper_jump_l974_97411

-- Definitions for the distances jumped
variables (G F M : ℕ)

-- Conditions given in the problem
def condition1 : Prop := G = F + 19
def condition2 : Prop := M = F - 12
def condition3 : Prop := M = 8

-- The theorem statement
theorem grasshopper_jump : condition1 G F ∧ condition2 F M ∧ condition3 M → G = 39 :=
by
  sorry

end grasshopper_jump_l974_97411


namespace journey_speed_l974_97424

theorem journey_speed (v : ℚ) 
  (equal_distance : ∀ {d}, (d = 0.22) → ((0.66 / 3) = d))
  (total_distance : ∀ {d}, (d = 660 / 1000) → (660 / 1000 = 0.66))
  (total_time : ∀ {t} , (t = 11 / 60) → (11 / 60 = t)): 
  (0.22 / 2 + 0.22 / v + 0.22 / 6 = 11 / 60) → v = 1.2 := 
by 
  sorry

end journey_speed_l974_97424


namespace inequality_abc_l974_97495

theorem inequality_abc (a b c : ℝ) 
  (habc : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ ab + bc + ca = 1) :
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2) :=
sorry

end inequality_abc_l974_97495


namespace max_pawns_l974_97452

def chessboard : Type := ℕ × ℕ -- Define a chessboard as a grid of positions (1,1) to (8,8)
def e4 : chessboard := (5, 4) -- Define the position e4
def symmetric_wrt_e4 (p1 p2 : chessboard) : Prop :=
  p1.1 + p2.1 = 10 ∧ p1.2 + p2.2 = 8 -- Symmetry condition relative to e4

def placed_on (pos : chessboard) : Prop := sorry -- placeholder for placement condition

theorem max_pawns (no_e4 : ¬ placed_on e4)
  (no_symmetric_pairs : ∀ p1 p2, symmetric_wrt_e4 p1 p2 → ¬ (placed_on p1 ∧ placed_on p2)) :
  ∃ max_pawns : ℕ, max_pawns = 39 :=
sorry

end max_pawns_l974_97452


namespace ten_year_old_dog_is_64_human_years_l974_97444

namespace DogYears

-- Definition of the conditions
def first_year_in_human_years : ℕ := 15
def second_year_in_human_years : ℕ := 9
def subsequent_year_in_human_years : ℕ := 5

-- Definition of the total human years for a 10-year-old dog.
def dog_age_in_human_years (dog_age : ℕ) : ℕ :=
  if dog_age = 1 then first_year_in_human_years
  else if dog_age = 2 then first_year_in_human_years + second_year_in_human_years
  else first_year_in_human_years + second_year_in_human_years + (dog_age - 2) * subsequent_year_in_human_years

-- The statement to prove
theorem ten_year_old_dog_is_64_human_years : dog_age_in_human_years 10 = 64 :=
  by
    sorry

end DogYears

end ten_year_old_dog_is_64_human_years_l974_97444


namespace tomatoes_difference_is_50_l974_97488

variable (yesterday_tomatoes today_tomatoes total_tomatoes : ℕ)

theorem tomatoes_difference_is_50 
  (h1 : yesterday_tomatoes = 120)
  (h2 : total_tomatoes = 290)
  (h3 : total_tomatoes = today_tomatoes + yesterday_tomatoes) :
  today_tomatoes - yesterday_tomatoes = 50 := sorry

end tomatoes_difference_is_50_l974_97488


namespace find_x_l974_97439

theorem find_x (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_x_l974_97439


namespace new_tv_width_l974_97498

-- Define the conditions
def first_tv_width := 24
def first_tv_height := 16
def first_tv_cost := 672
def new_tv_height := 32
def new_tv_cost := 1152
def cost_difference := 1

-- Define the question as a theorem
theorem new_tv_width : 
  let first_tv_area := first_tv_width * first_tv_height
  let first_tv_cost_per_sq_inch := first_tv_cost / first_tv_area
  let new_tv_cost_per_sq_inch := first_tv_cost_per_sq_inch - cost_difference
  let new_tv_area := new_tv_cost / new_tv_cost_per_sq_inch
  let new_tv_width := new_tv_area / new_tv_height
  new_tv_width = 48 :=
by
  -- Here, we would normally provide the proof steps, but we insert sorry as required.
  sorry

end new_tv_width_l974_97498


namespace sum_of_coordinates_l974_97482

noncomputable def g : ℝ → ℝ := sorry
noncomputable def h (x : ℝ) : ℝ := (g x) ^ 2

theorem sum_of_coordinates : g 3 = 6 → (3 + h 3 = 39) := by
  intro hg3
  have : h 3 = (g 3) ^ 2 := by rfl
  rw [hg3] at this
  rw [this]
  exact sorry

end sum_of_coordinates_l974_97482


namespace preston_charges_5_dollars_l974_97435

def cost_per_sandwich (x : Real) : Prop :=
  let number_of_sandwiches := 18
  let delivery_fee := 20
  let tip_percentage := 0.10
  let total_received := 121
  let total_cost := number_of_sandwiches * x + delivery_fee
  let tip := tip_percentage * total_cost
  let final_amount := total_cost + tip
  final_amount = total_received

theorem preston_charges_5_dollars :
  ∀ x : Real, cost_per_sandwich x → x = 5 :=
by
  intros x h
  sorry

end preston_charges_5_dollars_l974_97435


namespace ball_fall_time_l974_97473

theorem ball_fall_time (h g : ℝ) (t : ℝ) : 
  h = 20 → g = 10 → h + 20 * (t - 2) - 5 * ((t - 2) ^ 2) = t * (20 - 10 * (t - 2)) → 
  t = Real.sqrt 8 := 
by
  intros h_eq g_eq motion_eq
  sorry

end ball_fall_time_l974_97473


namespace line_does_not_pass_through_third_quadrant_l974_97487

theorem line_does_not_pass_through_third_quadrant (k : ℝ) :
  (∀ x : ℝ, ¬ (x > 0 ∧ (-3 * x + k) < 0)) ∧ (∀ x : ℝ, ¬ (x < 0 ∧ (-3 * x + k) > 0)) → k ≥ 0 :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l974_97487


namespace general_term_min_sum_Sn_l974_97499

-- (I) Prove the general term formula for the arithmetic sequence
theorem general_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10) 
  (geometric_cond : (a 2 + 10) * (a 4 + 6) = (a 3 + 8) ^ 2) : 
  ∃ n : ℕ, a n = 2 * n - 12 :=
by
  sorry

-- (II) Prove the minimum value of the sum of the first n terms
theorem min_sum_Sn (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10)
  (general_term : ∀ n, a n = 2 * n - 12) : 
  ∃ n, S n = n * n - 11 * n ∧ S n = -30 :=
by
  sorry

end general_term_min_sum_Sn_l974_97499


namespace hypotenuse_length_l974_97466

theorem hypotenuse_length {a b c : ℝ} (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
  sorry

end hypotenuse_length_l974_97466


namespace greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l974_97442

theorem greatest_possible_sum_of_two_consecutive_integers_product_lt_1000 : 
  ∃ n : ℤ, (n * (n + 1) < 1000) ∧ (n + (n + 1) = 63) :=
sorry

end greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l974_97442


namespace set_equality_l974_97464

open Set

variable (A : Set ℕ)

theorem set_equality (h1 : {1, 3} ⊆ A) (h2 : {1, 3} ∪ A = {1, 3, 5}) : A = {1, 3, 5} :=
sorry

end set_equality_l974_97464


namespace circle_center_radius_l974_97475

def circle_equation (x y : ℝ) : Prop := x^2 + 4 * x + y^2 - 6 * y - 12 = 0

theorem circle_center_radius :
  ∃ (h k r : ℝ), (circle_equation (x : ℝ) (y: ℝ) -> (x + h)^2 + (y + k)^2 = r^2) ∧ h = -2 ∧ k = 3 ∧ r = 5 :=
sorry

end circle_center_radius_l974_97475


namespace f_eq_l974_97470

noncomputable def a (n : ℕ) : ℚ := 1 / ((n + 1) ^ 2)

noncomputable def f : ℕ → ℚ
| 0     => 1
| (n+1) => f n * (1 - a (n+1))

theorem f_eq : ∀ n : ℕ, f n = (n + 2) / (2 * (n + 1)) :=
by
  sorry

end f_eq_l974_97470


namespace monica_milk_l974_97423

theorem monica_milk (don_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) (h_don : don_milk = 3 / 4)
  (h_rachel : rachel_fraction = 1 / 2) (h_monica : monica_fraction = 1 / 3) :
  monica_fraction * (rachel_fraction * don_milk) = 1 / 8 :=
by
  sorry

end monica_milk_l974_97423


namespace intersect_condition_l974_97458

theorem intersect_condition (m : ℕ) (h : m ≠ 0) : 
  (∃ x y : ℝ, (3 * x - 2 * y = 0) ∧ ((x - m)^2 + y^2 = 1)) → m = 1 :=
by 
  sorry

end intersect_condition_l974_97458


namespace convert_to_scientific_notation_l974_97462

theorem convert_to_scientific_notation :
  40.25 * 10^9 = 4.025 * 10^9 :=
by
  -- Sorry is used here to skip the proof
  sorry

end convert_to_scientific_notation_l974_97462


namespace num_arithmetic_sequences_l974_97450

-- Definitions of the arithmetic sequence conditions
def is_arithmetic_sequence (a d n : ℕ) : Prop :=
  0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ 
  (∃ k : ℕ, k = 97 ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * k ^ 2)) 

-- Prove that there are exactly 4 such sequences
theorem num_arithmetic_sequences : 
  ∃ (n : ℕ) (a d : ℕ), 
  is_arithmetic_sequence a d n ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * 97^2) ∧ (
    (n = 97 ∧ ((a = 97 ∧ d = 0) ∨ (a = 49 ∧ d = 1) ∨ (a = 1 ∧ d = 2))) ∨
    (n = 97^2 ∧ a = 1 ∧ d = 0)
  ) :=
sorry

end num_arithmetic_sequences_l974_97450


namespace race_duration_l974_97410

theorem race_duration 
  (lap_distance : ℕ) (laps : ℕ)
  (award_per_hundred_meters : ℝ) (earn_rate_per_minute : ℝ)
  (total_distance : ℕ) (total_award : ℝ) (duration : ℝ) :
  lap_distance = 100 →
  laps = 24 →
  award_per_hundred_meters = 3.5 →
  earn_rate_per_minute = 7 →
  total_distance = lap_distance * laps →
  total_award = (total_distance / 100) * award_per_hundred_meters →
  duration = total_award / earn_rate_per_minute →
  duration = 12 := 
by 
  intros;
  sorry

end race_duration_l974_97410


namespace complex_multiplication_l974_97422

-- Define the imaginary unit i
def i := Complex.I

-- Define the theorem we need to prove
theorem complex_multiplication : 
  (3 - 7 * i) * (-6 + 2 * i) = -4 + 48 * i := 
by 
  -- Proof is omitted
  sorry

end complex_multiplication_l974_97422


namespace linear_function_mask_l974_97496

theorem linear_function_mask (x : ℝ) : ∃ k, k = 0.9 ∧ ∀ x, y = k * x :=
by
  sorry

end linear_function_mask_l974_97496


namespace area_of_triangle_l974_97403

theorem area_of_triangle (s1 s2 s3 : ℕ) (h1 : s1^2 = 36) (h2 : s2^2 = 64) (h3 : s3^2 = 100) (h4 : s1^2 + s2^2 = s3^2) :
  (1 / 2 : ℚ) * s1 * s2 = 24 := by
  sorry

end area_of_triangle_l974_97403


namespace point_value_of_other_questions_is_4_l974_97491

theorem point_value_of_other_questions_is_4
  (total_points : ℕ)
  (total_questions : ℕ)
  (points_from_2_point_questions : ℕ)
  (other_questions : ℕ)
  (points_each_2_point_question : ℕ)
  (points_from_2_point_questions_calc : ℕ)
  (remaining_points : ℕ)
  (point_value_of_other_type : ℕ)
  : total_points = 100 →
    total_questions = 40 →
    points_each_2_point_question = 2 →
    other_questions = 10 →
    points_from_2_point_questions = 30 →
    points_from_2_point_questions_calc = points_each_2_point_question * points_from_2_point_questions →
    remaining_points = total_points - points_from_2_point_questions_calc →
    remaining_points = other_questions * point_value_of_other_type →
    point_value_of_other_type = 4 := by
  sorry

end point_value_of_other_questions_is_4_l974_97491


namespace total_amount_collected_in_paise_total_amount_collected_in_rupees_l974_97454

-- Definitions and conditions
def num_members : ℕ := 96
def contribution_per_member : ℕ := 96
def total_paise_collected : ℕ := num_members * contribution_per_member
def total_rupees_collected : ℚ := total_paise_collected / 100

-- Theorem stating the total amount collected
theorem total_amount_collected_in_paise :
  total_paise_collected = 9216 := by sorry

theorem total_amount_collected_in_rupees :
  total_rupees_collected = 92.16 := by sorry

end total_amount_collected_in_paise_total_amount_collected_in_rupees_l974_97454


namespace value_of_k_l974_97427

theorem value_of_k (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p/q = 3/2 ∧ p + q = -10 ∧ p * q = k) → k = 24 :=
by 
  sorry

end value_of_k_l974_97427


namespace missing_root_l974_97471

theorem missing_root (p q r : ℝ) 
  (h : p * (q - r) ≠ 0 ∧ q * (r - p) ≠ 0 ∧ r * (p - q) ≠ 0 ∧ 
       p * (q - r) * (-1)^2 + q * (r - p) * (-1) + r * (p - q) = 0) : 
  ∃ x : ℝ, x ≠ -1 ∧ 
  p * (q - r) * x^2 + q * (r - p) * x + r * (p - q) = 0 ∧ 
  x = - (r * (p - q) / (p * (q - r))) :=
sorry

end missing_root_l974_97471


namespace gcd_2023_2048_l974_97476

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l974_97476


namespace smallest_positive_multiple_of_17_with_condition_l974_97447

theorem smallest_positive_multiple_of_17_with_condition :
  ∃ k : ℕ, k > 0 ∧ (k % 17 = 0) ∧ (k - 3) % 101 = 0 ∧ k = 306 :=
by
  sorry

end smallest_positive_multiple_of_17_with_condition_l974_97447


namespace books_read_so_far_l974_97469

/-- There are 22 different books in the 'crazy silly school' series -/
def total_books : Nat := 22

/-- You still have to read 10 more books -/
def books_left_to_read : Nat := 10

theorem books_read_so_far :
  total_books - books_left_to_read = 12 :=
by
  sorry

end books_read_so_far_l974_97469


namespace determine_b_l974_97461

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 1 / (3 * x + b)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_b (b : ℝ) :
    (∀ x : ℝ, f_inv (f x b) = x) ↔ b = -3 :=
by
  sorry

end determine_b_l974_97461


namespace sin_lg_roots_l974_97479

theorem sin_lg_roots (f : ℝ → ℝ) (g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x) (h₂ : ∀ x, g x = Real.log x)
  (domain : ∀ x, x > 0 → x < 10) (h₃ : ∀ x, f x ≤ 1 ∧ g x ≤ 1) :
  ∃ x1 x2 x3, (0 < x1 ∧ x1 < 10) ∧ (f x1 = g x1) ∧
               (0 < x2 ∧ x2 < 10) ∧ (f x2 = g x2) ∧
               (0 < x3 ∧ x3 < 10) ∧ (f x3 = g x3) ∧
               x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 :=
by
  sorry

end sin_lg_roots_l974_97479


namespace mr_bird_speed_to_work_l974_97426

theorem mr_bird_speed_to_work (
  d t : ℝ
) (h1 : d = 45 * (t + 4 / 60)) 
  (h2 : d = 55 * (t - 2 / 60))
  (h3 : t = 29 / 60)
  (d_eq : d = 24.75) :
  (24.75 / (29 / 60)) = 51.207 := 
sorry

end mr_bird_speed_to_work_l974_97426


namespace obtuse_triangle_count_l974_97457

-- Definitions based on conditions
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def is_arithmetic_sequence (a b c : ℕ) : Prop :=
  2 * b = a + c

def is_obtuse_triangle (a b c : ℕ) : Prop :=
  a * a + b * b < c * c ∨ b * b + c * c < a * a ∨ c * c + a * a < b * b

-- Main conjecture to prove
theorem obtuse_triangle_count :
  ∃ (n : ℕ), n = 157 ∧
    ∀ (a b c : ℕ), 
      a <= 50 ∧ b <= 50 ∧ c <= 50 ∧ 
      is_arithmetic_sequence a b c ∧ 
      is_triangle a b c ∧ 
      is_obtuse_triangle a b c → 
    true := sorry

end obtuse_triangle_count_l974_97457


namespace perfect_square_solution_l974_97477

theorem perfect_square_solution (m n : ℕ) (p : ℕ) [hp : Fact (Nat.Prime p)] :
  (∃ k : ℕ, (5 ^ m + 2 ^ n * p) / (5 ^ m - 2 ^ n * p) = k ^ 2)
  ↔ (m = 1 ∧ n = 1 ∧ p = 2 ∨ m = 3 ∧ n = 2 ∧ p = 3 ∨ m = 2 ∧ n = 2 ∧ p = 5) :=
by
  sorry

end perfect_square_solution_l974_97477


namespace square_floor_tile_count_l974_97438

theorem square_floor_tile_count (n : ℕ) (h1 : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_floor_tile_count_l974_97438


namespace exists_four_distinct_indices_l974_97404

theorem exists_four_distinct_indices
  (a : Fin 5 → ℝ)
  (h : ∀ i, 0 < a i) :
  ∃ i j k l : (Fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < 1 / 2 :=
by
  sorry

end exists_four_distinct_indices_l974_97404


namespace students_present_each_day_l974_97448
open BigOperators

namespace Absenteeism

def absenteeism_rate : ℕ → ℝ 
| 0 => 14
| n+1 => absenteeism_rate n + 2

def present_rate (n : ℕ) : ℝ := 100 - absenteeism_rate n

theorem students_present_each_day :
  present_rate 0 = 86 ∧
  present_rate 1 = 84 ∧
  present_rate 2 = 82 ∧
  present_rate 3 = 80 ∧
  present_rate 4 = 78 := 
by
  -- Placeholder for the proof steps
  sorry

end Absenteeism

end students_present_each_day_l974_97448


namespace count_square_free_integers_l974_97440

def square_free_in_range_2_to_199 : Nat :=
  91

theorem count_square_free_integers :
  ∃ n : Nat, n = 91 ∧
  ∀ m : Nat, 2 ≤ m ∧ m < 200 →
  (∀ k : Nat, k^2 ∣ m → k^2 = 1) :=
by
  -- The proof will be filled here
  sorry

end count_square_free_integers_l974_97440


namespace cost_price_of_computer_table_l974_97436

/-- The owner of a furniture shop charges 20% more than the cost price. 
    Given that the customer paid Rs. 3000 for the computer table, 
    prove that the cost price of the computer table was Rs. 2500. -/
theorem cost_price_of_computer_table (CP SP : ℝ) (h1 : SP = CP + 0.20 * CP) (h2 : SP = 3000) : CP = 2500 :=
by {
  sorry
}

end cost_price_of_computer_table_l974_97436


namespace symmetric_coords_l974_97459

-- Define the initial point and the line equation
def initial_point : ℝ × ℝ := (-1, 1)
def line_eq (x y : ℝ) : Prop := x - y - 1 = 0

-- Define what it means for one point to be symmetric to another point with respect to a line
def symmetric_point (p q : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), line_eq m p.1 ∧ line_eq m q.1 ∧ 
             p.1 + q.1 = 2 * m ∧
             p.2 + q.2 = 2 * m

-- The theorem we want to prove
theorem symmetric_coords : ∃ (symmetric : ℝ × ℝ), symmetric_point initial_point symmetric ∧ symmetric = (2, -2) :=
sorry

end symmetric_coords_l974_97459


namespace cyclist_problem_l974_97428

theorem cyclist_problem (MP NP : ℝ) (h1 : NP = MP + 30) (h2 : ∀ t : ℝ, t*MP = 10*t) 
  (h3 : ∀ t : ℝ, t*NP = 10*t) 
  (h4 : ∀ t : ℝ, t*MP = 42 → t*(MP + 30) = t*42 - 1/3) : 
  MP = 180 := 
sorry

end cyclist_problem_l974_97428


namespace train_length_l974_97489

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 60) (h2 : time_sec = 9) : length_m = 150 := by
  sorry

end train_length_l974_97489


namespace min_value_a_2b_l974_97402

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 3 / b = 1) :
  a + 2 * b = 7 + 2 * Real.sqrt 6 :=
sorry

end min_value_a_2b_l974_97402


namespace find_bc_l974_97453

theorem find_bc (b c : ℤ) (h : ∀ x : ℝ, x^2 + (b : ℝ) * x + (c : ℝ) = 0 ↔ x = 1 ∨ x = 2) :
  b = -3 ∧ c = 2 := by
  sorry

end find_bc_l974_97453


namespace minimum_period_f_l974_97492

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x / 2 + Real.pi / 4)

theorem minimum_period_f :
  ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) :=
sorry

end minimum_period_f_l974_97492
