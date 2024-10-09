import Mathlib

namespace distance_between_closest_points_of_circles_l1080_108056

theorem distance_between_closest_points_of_circles :
  let circle1_center : ℝ × ℝ := (3, 3)
  let circle2_center : ℝ × ℝ := (20, 15)
  let circle1_radius : ℝ := 3
  let circle2_radius : ℝ := 15
  let distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (15 - 3)^2)
  distance_between_centers - (circle1_radius + circle2_radius) = 2.81 :=
by {
  sorry
}

end distance_between_closest_points_of_circles_l1080_108056


namespace g_increasing_on_minus_infty_one_l1080_108073

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def f_inv (x : ℝ) : ℝ := (x + 1) / (x - 1)
noncomputable def g (x : ℝ) : ℝ := 1 + (2 * x) / (1 - x)

theorem g_increasing_on_minus_infty_one : (∀ x y : ℝ, x < y → x < 1 → y ≤ 1 → g x < g y) :=
sorry

end g_increasing_on_minus_infty_one_l1080_108073


namespace table_chair_price_l1080_108067

theorem table_chair_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : T = 84) : T + C = 96 :=
sorry

end table_chair_price_l1080_108067


namespace number_in_tens_place_is_7_l1080_108040

theorem number_in_tens_place_is_7
  (digits : Finset ℕ)
  (a b c : ℕ)
  (h1 : digits = {7, 5, 2})
  (h2 : 100 * a + 10 * b + c > 530)
  (h3 : 100 * a + 10 * b + c < 710)
  (h4 : a ∈ digits)
  (h5 : b ∈ digits)
  (h6 : c ∈ digits)
  (h7 : ∀ x ∈ digits, x ≠ a → x ≠ b → x ≠ c) :
  b = 7 := sorry

end number_in_tens_place_is_7_l1080_108040


namespace incorrect_statement_l1080_108075

theorem incorrect_statement (a : ℝ) (x : ℝ) (h : a > 1) :
  ¬((x = 0 → a^x = 1) ∧
    (x = 1 → a^x = a) ∧
    (x = -1 → a^x = 1/a) ∧
    (x < 0 → 0 < a^x ∧ ∀ ε > 0, ∃ x' < x, a^x' < ε)) :=
sorry

end incorrect_statement_l1080_108075


namespace mary_saves_in_five_months_l1080_108086

def washing_earnings : ℕ := 20
def walking_earnings : ℕ := 40
def monthly_earnings : ℕ := washing_earnings + walking_earnings
def savings_rate : ℕ := 2
def monthly_savings : ℕ := monthly_earnings / savings_rate
def total_savings_target : ℕ := 150

theorem mary_saves_in_five_months :
  total_savings_target / monthly_savings = 5 :=
by
  sorry

end mary_saves_in_five_months_l1080_108086


namespace correct_statements_truth_of_statements_l1080_108049

-- Define basic properties related to factor and divisor
def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = d * k

-- Given conditions as definitions
def condition_A : Prop := is_factor 4 100
def condition_B1 : Prop := is_divisor 19 133
def condition_B2 : Prop := ¬ is_divisor 19 51
def condition_C1 : Prop := is_divisor 30 90
def condition_C2 : Prop := ¬ is_divisor 30 53
def condition_D1 : Prop := is_divisor 7 21
def condition_D2 : Prop := ¬ is_divisor 7 49
def condition_E : Prop := is_factor 10 200

-- Statement that needs to be proved
theorem correct_statements : 
  (condition_A ∧ 
  (condition_B1 ∧ condition_B2) ∧ 
  condition_E) :=
by sorry -- proof to be inserted

-- Equivalent Lean 4 statement with all conditions encapsulated
theorem truth_of_statements :
  (is_factor 4 100) ∧ 
  (is_divisor 19 133 ∧ ¬ is_divisor 19 51) ∧ 
  is_factor 10 200 :=
by sorry -- proof to be inserted

end correct_statements_truth_of_statements_l1080_108049


namespace jane_weekly_pages_l1080_108084

-- Define the daily reading amounts
def monday_wednesday_morning_pages : ℕ := 5
def monday_wednesday_evening_pages : ℕ := 10
def tuesday_thursday_morning_pages : ℕ := 7
def tuesday_thursday_evening_pages : ℕ := 8
def friday_morning_pages : ℕ := 10
def friday_evening_pages : ℕ := 15
def weekend_morning_pages : ℕ := 12
def weekend_evening_pages : ℕ := 20

-- Define the number of days
def monday_wednesday_days : ℕ := 2
def tuesday_thursday_days : ℕ := 2
def friday_days : ℕ := 1
def weekend_days : ℕ := 2

-- Function to calculate weekly pages
def weekly_pages :=
  (monday_wednesday_days * (monday_wednesday_morning_pages + monday_wednesday_evening_pages)) +
  (tuesday_thursday_days * (tuesday_thursday_morning_pages + tuesday_thursday_evening_pages)) +
  (friday_days * (friday_morning_pages + friday_evening_pages)) +
  (weekend_days * (weekend_morning_pages + weekend_evening_pages))

-- Proof statement
theorem jane_weekly_pages : weekly_pages = 149 := by
  unfold weekly_pages
  norm_num
  sorry

end jane_weekly_pages_l1080_108084


namespace min_red_hair_students_l1080_108087

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l1080_108087


namespace largest_number_using_digits_l1080_108004

theorem largest_number_using_digits (d1 d2 d3 : ℕ) (h1 : d1 = 7) (h2 : d2 = 1) (h3 : d3 = 0) : 
  ∃ n : ℕ, (n = 710) ∧ (∀ m : ℕ, (m = d1 * 100 + d2 * 10 + d3) ∨ (m = d1 * 100 + d3 * 10 + d2) ∨ (m = d2 * 100 + d1 * 10 + d3) ∨ 
  (m = d2 * 100 + d3 * 10 + d1) ∨ (m = d3 * 100 + d1 * 10 + d2) ∨ (m = d3 * 100 + d2 * 10 + d1) → n ≥ m) := 
by
  sorry

end largest_number_using_digits_l1080_108004


namespace sum_of_integers_is_28_l1080_108079

theorem sum_of_integers_is_28 (m n p q : ℕ) (hmnpq_diff : m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q)
  (hm_pos : 0 < m) (hn_pos : 0 < n) (hp_pos : 0 < p) (hq_pos : 0 < q)
  (h_prod : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 :=
by
  sorry

end sum_of_integers_is_28_l1080_108079


namespace pond_sustain_capacity_l1080_108039

-- Defining the initial number of frogs
def initial_frogs : ℕ := 5

-- Defining the number of tadpoles
def number_of_tadpoles (frogs: ℕ) : ℕ := 3 * frogs

-- Defining the number of matured tadpoles (those that survive to become frogs)
def matured_tadpoles (tadpoles: ℕ) : ℕ := (2 * tadpoles) / 3

-- Defining the total number of frogs after tadpoles mature
def total_frogs_after_mature (initial_frogs: ℕ) (matured_tadpoles: ℕ) : ℕ :=
  initial_frogs + matured_tadpoles

-- Defining the number of frogs that need to find a new pond
def frogs_to_leave : ℕ := 7

-- Defining the number of frogs the pond can sustain
def frogs_pond_can_sustain (total_frogs: ℕ) (frogs_to_leave: ℕ) : ℕ :=
  total_frogs - frogs_to_leave

-- The main theorem stating the number of frogs the pond can sustain given the conditions
theorem pond_sustain_capacity : frogs_pond_can_sustain
  (total_frogs_after_mature initial_frogs (matured_tadpoles (number_of_tadpoles initial_frogs)))
  frogs_to_leave = 8 := by
  -- proof goes here
  sorry

end pond_sustain_capacity_l1080_108039


namespace initial_skittles_geq_16_l1080_108001

variable (S : ℕ) -- S represents the total number of Skittles Lillian had initially
variable (L : ℕ) -- L represents the number of Skittles Lillian kept as leftovers

theorem initial_skittles_geq_16 (h1 : S = 8 * 2 + L) : S ≥ 16 :=
by
  sorry

end initial_skittles_geq_16_l1080_108001


namespace area_enclosed_by_sin_l1080_108092

/-- The area of the figure enclosed by the curve y = sin(x), the lines x = -π/3, x = π/2, and the x-axis is 3/2. -/
theorem area_enclosed_by_sin (x y : ℝ) (h : y = Real.sin x) (a b : ℝ) 
(h1 : a = -Real.pi / 3) (h2 : b = Real.pi / 2) :
  ∫ x in a..b, |Real.sin x| = 3 / 2 := 
sorry

end area_enclosed_by_sin_l1080_108092


namespace compare_abc_l1080_108006

noncomputable def a : ℝ := 2^(1/2)
noncomputable def b : ℝ := 3^(1/3)
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > a ∧ a > c :=
by
  sorry

end compare_abc_l1080_108006


namespace rings_sold_l1080_108042

theorem rings_sold (R : ℕ) : 
  ∀ (num_necklaces total_sales necklace_price ring_price : ℕ),
  num_necklaces = 4 →
  total_sales = 80 →
  necklace_price = 12 →
  ring_price = 4 →
  num_necklaces * necklace_price + R * ring_price = total_sales →
  R = 8 := 
by 
  intros num_necklaces total_sales necklace_price ring_price h1 h2 h3 h4 h5
  sorry

end rings_sold_l1080_108042


namespace range_of_a_l1080_108057

variable (a : ℝ)
def A := Set.Ico (-2 : ℝ) 4
def B := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (h : B a ⊆ A) : 0 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l1080_108057


namespace B_pow_5_eq_rB_plus_sI_l1080_108012

def B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 4, 5]

def I : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 0, 1]

theorem B_pow_5_eq_rB_plus_sI : 
  ∃ (r s : ℤ), r = 1169 ∧ s = -204 ∧ B^5 = r • B + s • I := 
by
  use 1169
  use -204
  sorry

end B_pow_5_eq_rB_plus_sI_l1080_108012


namespace sum_of_squares_of_roots_eq_226_l1080_108010

theorem sum_of_squares_of_roots_eq_226 (s_1 s_2 : ℝ) (h_eq : ∀ x, x^2 - 16 * x + 15 = 0 → (x = s_1 ∨ x = s_2)) :
  s_1^2 + s_2^2 = 226 := by
  sorry

end sum_of_squares_of_roots_eq_226_l1080_108010


namespace women_in_room_l1080_108080

theorem women_in_room (M W : ℕ) 
  (h1 : 9 * M = 7 * W) 
  (h2 : M + 5 = 23) : 
  3 * (W - 4) = 57 :=
by
  sorry

end women_in_room_l1080_108080


namespace common_ratio_of_geometric_seq_l1080_108019

variable {a : ℕ → ℚ} -- The sequence
variable {d : ℚ} -- Common difference

-- Assuming the arithmetic and geometric sequence properties
def is_arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_seq (a1 a4 a5 : ℚ) (q : ℚ) : Prop :=
  a4 = a1 * q ∧ a5 = a4 * q

theorem common_ratio_of_geometric_seq (h_arith: is_arithmetic_seq a d) (h_nonzero_d : d ≠ 0)
  (h_geometric: is_geometric_seq (a 1) (a 4) (a 5) (1 / 3)) : (a 4 / a 1) = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_seq_l1080_108019


namespace inequality_with_means_l1080_108081

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l1080_108081


namespace algebraic_expression_evaluation_l1080_108011

open Real

noncomputable def x : ℝ := 2 - sqrt 3

theorem algebraic_expression_evaluation :
  (7 + 4 * sqrt 3) * x^2 - (2 + sqrt 3) * x + sqrt 3 = 2 + sqrt 3 :=
by
  sorry

end algebraic_expression_evaluation_l1080_108011


namespace min_alpha_beta_l1080_108076

theorem min_alpha_beta (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1)
  (alpha : ℝ := a + 1 / a) (beta : ℝ := b + 1 / b) :
  alpha + beta ≥ 10 := by
  sorry

end min_alpha_beta_l1080_108076


namespace smaller_cube_edge_length_l1080_108013

theorem smaller_cube_edge_length (x : ℝ) 
    (original_edge_length : ℝ := 7)
    (increase_percentage : ℝ := 600) 
    (original_surface_area_formula : ℝ := 6 * original_edge_length^2)
    (new_surface_area_formula : ℝ := (1 + increase_percentage / 100) * original_surface_area_formula) :
  ∃ x : ℝ, 6 * x^2 * (original_edge_length ^ 3 / x ^ 3) = new_surface_area_formula → x = 1 := by
  sorry

end smaller_cube_edge_length_l1080_108013


namespace no_two_right_angles_in_triangle_l1080_108091

theorem no_two_right_angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 90) (h3 : B = 90): false :=
by
  -- we assume A = 90 and B = 90,
  -- then A + B + C > 180, which contradicts h1,
  sorry
  
example : (3 = 3) := by sorry  -- Given the context of the multiple-choice problem.

end no_two_right_angles_in_triangle_l1080_108091


namespace work_together_l1080_108005

theorem work_together (A_rate B_rate : ℝ) (hA : A_rate = 1 / 9) (hB : B_rate = 1 / 18) : (1 / (A_rate + B_rate) = 6) :=
by
  -- we only need to write the statement, proof is not required.
  sorry

end work_together_l1080_108005


namespace find_x_l1080_108051

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x^2 + 18 * x * y = x^3 + 3 * x^2 * y + 6 * x) : 
  x = 3 :=
sorry

end find_x_l1080_108051


namespace sets_relation_l1080_108027

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def M : Set ℚ := {x | ∃ (m : ℤ), x = m + 1/6}
def S : Set ℚ := {x | ∃ (s : ℤ), x = s/2 - 1/3}
def P : Set ℚ := {x | ∃ (p : ℤ), x = p/2 + 1/6}

theorem sets_relation : M ⊆ S ∧ S = P := by
  sorry

end sets_relation_l1080_108027


namespace smallest_possible_value_l1080_108090

theorem smallest_possible_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h1 : a^2 ≥ 12 * b)
  (h2 : 9 * b^2 ≥ 4 * a) : a + b = 7 :=
sorry

end smallest_possible_value_l1080_108090


namespace pentagon_angle_sum_l1080_108060

theorem pentagon_angle_sum
  (a b c d : ℝ) (Q : ℝ)
  (sum_angles : 180 * (5 - 2) = 540)
  (given_angles : a = 130 ∧ b = 80 ∧ c = 105 ∧ d = 110) :
  Q = 540 - (a + b + c + d) := by
  sorry

end pentagon_angle_sum_l1080_108060


namespace P_lt_Q_l1080_108077

noncomputable def P (a : ℝ) : ℝ := (Real.sqrt (a + 41)) - (Real.sqrt (a + 40))
noncomputable def Q (a : ℝ) : ℝ := (Real.sqrt (a + 39)) - (Real.sqrt (a + 38))

theorem P_lt_Q (a : ℝ) (h : a > -38) : P a < Q a := by sorry

end P_lt_Q_l1080_108077


namespace plane_ratio_l1080_108085

section

variables (D B T P : ℕ)

-- Given conditions
axiom total_distance : D = 1800
axiom distance_by_bus : B = 720
axiom distance_by_train : T = (2 * B) / 3

-- Prove the ratio of the distance traveled by plane to the whole trip
theorem plane_ratio :
  D = 1800 →
  B = 720 →
  T = (2 * B) / 3 →
  P = D - (T + B) →
  P / D = 1 / 3 := by
  intros h1 h2 h3 h4
  sorry

end

end plane_ratio_l1080_108085


namespace alison_money_l1080_108034

theorem alison_money (k b br bt al : ℝ) 
  (h1 : al = 1/2 * bt) 
  (h2 : bt = 4 * br) 
  (h3 : br = 2 * k) 
  (h4 : k = 1000) : 
  al = 4000 := 
by 
  sorry

end alison_money_l1080_108034


namespace fibonacci_series_sum_l1080_108064

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Theorem to prove that the infinite series sum is 2
theorem fibonacci_series_sum : (∑' n : ℕ, (fib n : ℝ) / (2 ^ n : ℝ)) = 2 :=
sorry

end fibonacci_series_sum_l1080_108064


namespace evaluate_expression_l1080_108095

theorem evaluate_expression : 2^(Real.log 5 / Real.log 2) + Real.log 25 / Real.log 5 = 7 := by
  sorry

end evaluate_expression_l1080_108095


namespace find_n_l1080_108082

theorem find_n (x y m n : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) 
  (h1 : 100 * y + x = (x + y) * m) (h2 : 100 * x + y = (x + y) * n) : n = 101 - m :=
by
  sorry

end find_n_l1080_108082


namespace yardage_lost_due_to_sacks_l1080_108098

theorem yardage_lost_due_to_sacks 
  (throws : ℕ)
  (percent_no_throw : ℝ)
  (half_sack_prob : ℕ)
  (sack_pattern : ℕ → ℕ)
  (correct_answer : ℕ) : 
  throws = 80 →
  percent_no_throw = 0.30 →
  (∀ (n: ℕ), half_sack_prob = n/2) →
  (sack_pattern 1 = 3 ∧ sack_pattern 2 = 5 ∧ ∀ n, n > 2 → sack_pattern n = sack_pattern (n - 1) + 2) →
  correct_answer = 168 :=
by
  sorry

end yardage_lost_due_to_sacks_l1080_108098


namespace find_b_l1080_108026

theorem find_b (a : ℝ) (A : ℝ) (B : ℝ) (b : ℝ)
  (ha : a = 5) 
  (hA : A = Real.pi / 6) 
  (htanB : Real.tan B = 3 / 4)
  (hsinB : Real.sin B = 3 / 5):
  b = 6 := 
by 
  sorry

end find_b_l1080_108026


namespace problem_solution_l1080_108044

theorem problem_solution (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end problem_solution_l1080_108044


namespace am_gm_inequality_example_l1080_108062

theorem am_gm_inequality_example (x1 x2 x3 : ℝ)
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h_sum1 : x1 + x2 + x3 = 1) :
  (x2^2 / x1) + (x3^2 / x2) + (x1^2 / x3) ≥ 1 :=
by
  sorry

end am_gm_inequality_example_l1080_108062


namespace handshake_count_l1080_108035

theorem handshake_count (n : ℕ) (m : ℕ) (couples : ℕ) (people : ℕ) 
  (h1 : couples = 15) 
  (h2 : people = 2 * couples)
  (h3 : people = 30)
  (h4 : n = couples) 
  (h5 : m = people / 2)
  (h6 : ∀ i : ℕ, i < m → ∀ j : ℕ, j < m → i ≠ j → i * j + i ≠ n 
    ∧ j * i + j ≠ n) 
  : n * (n - 1) / 2 + (2 * n - 2) * n = 315 :=
by
  sorry

end handshake_count_l1080_108035


namespace correct_transformation_l1080_108055

variable (a b : ℝ)
variable (h₀ : a ≠ 0)
variable (h₁ : b ≠ 0)
variable (h₂ : a / 2 = b / 3)

theorem correct_transformation : 3 / b = 2 / a :=
by
  sorry

end correct_transformation_l1080_108055


namespace distance_traveled_on_fifth_day_equals_12_li_l1080_108088

theorem distance_traveled_on_fifth_day_equals_12_li:
  ∀ {a_1 : ℝ},
    (a_1 * ((1 - (1 / 2) ^ 6) / (1 - 1 / 2)) = 378) →
    (a_1 * (1 / 2) ^ 4 = 12) :=
by
  intros a_1 h
  sorry

end distance_traveled_on_fifth_day_equals_12_li_l1080_108088


namespace tom_ratio_is_three_fourths_l1080_108014

-- Define the years for the different programs
def bs_years : ℕ := 3
def phd_years : ℕ := 5
def tom_years : ℕ := 6
def normal_years : ℕ := bs_years + phd_years

-- Define the ratio of Tom's time to the normal time
def ratio : ℚ := tom_years / normal_years

theorem tom_ratio_is_three_fourths :
  ratio = 3 / 4 :=
by
  unfold ratio normal_years bs_years phd_years tom_years
  -- continued proof steps would go here
  sorry

end tom_ratio_is_three_fourths_l1080_108014


namespace rowing_speed_in_still_water_l1080_108020

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.4) (t : ℝ)
  (h2 : (v + c) * t = (v - c) * (2 * t)) : 
  v = 4.2 :=
by
  sorry

end rowing_speed_in_still_water_l1080_108020


namespace cats_new_total_weight_l1080_108047

noncomputable def total_weight (weights : List ℚ) : ℚ :=
  weights.sum

noncomputable def remove_min_max_weight (weights : List ℚ) : ℚ :=
  let min_weight := weights.minimum?.getD 0
  let max_weight := weights.maximum?.getD 0
  weights.sum - min_weight - max_weight

theorem cats_new_total_weight :
  let weights := [3.5, 7.2, 4.8, 6, 5.5, 9, 4]
  remove_min_max_weight weights = 27.5 := by
  sorry

end cats_new_total_weight_l1080_108047


namespace find_m_from_parallel_vectors_l1080_108069

variables (m : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

-- The condition that vectors a and b are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Given that a and b are parallel, prove that m = -3/2
theorem find_m_from_parallel_vectors
  (h : vectors_parallel (1, m) (2, -3)) :
  m = -3 / 2 :=
sorry

end find_m_from_parallel_vectors_l1080_108069


namespace magician_assistant_trick_l1080_108031

/-- A coin can be either heads or tails. -/
inductive Coin
| heads : Coin
| tails : Coin

/-- Given a cyclic arrangement of 11 coins, there exists at least one pair of adjacent coins with the same face. -/
theorem magician_assistant_trick (coins : Fin 11 → Coin) : 
  ∃ i : Fin 11, coins i = coins (i + 1) := 
by
  sorry

end magician_assistant_trick_l1080_108031


namespace janice_total_cost_is_correct_l1080_108003

def cost_of_items (cost_juices : ℕ) (juices : ℕ) (cost_sandwiches : ℕ) (sandwiches : ℕ) (cost_pastries : ℕ) (pastries : ℕ) (cost_salad : ℕ) (discount_salad : ℕ) : ℕ :=
  let one_sandwich := cost_sandwiches / sandwiches
  let one_juice := cost_juices / juices
  let total_pastries := pastries * cost_pastries
  let discounted_salad := cost_salad - (cost_salad * discount_salad / 100)
  one_sandwich + one_juice + total_pastries + discounted_salad

-- Conditions
def cost_juices := 10
def juices := 5
def cost_sandwiches := 6
def sandwiches := 2
def cost_pastries := 4
def pastries := 2
def cost_salad := 8
def discount_salad := 20

-- Expected Total Cost
def expected_total_cost := 1940 -- in cents to avoid float numbers

theorem janice_total_cost_is_correct : 
  cost_of_items cost_juices juices cost_sandwiches sandwiches cost_pastries pastries cost_salad discount_salad = expected_total_cost :=
by
  simp [cost_of_items, cost_juices, juices, cost_sandwiches, sandwiches, cost_pastries, pastries, cost_salad, discount_salad]
  norm_num
  sorry

end janice_total_cost_is_correct_l1080_108003


namespace sam_total_yellow_marbles_l1080_108033

def sam_original_yellow_marbles : Float := 86.0
def sam_yellow_marbles_given_by_joan : Float := 25.0

theorem sam_total_yellow_marbles : sam_original_yellow_marbles + sam_yellow_marbles_given_by_joan = 111.0 := by
  sorry

end sam_total_yellow_marbles_l1080_108033


namespace right_triangle_equation_l1080_108036

-- Let a, b, and c be the sides of a right triangle with a^2 + b^2 = c^2
variables (a b c : ℕ)
-- Define the semiperimeter
def semiperimeter (a b c : ℕ) : ℕ := (a + b + c) / 2
-- Define the radius of the inscribed circle
def inscribed_radius (a b c : ℕ) : ℚ := (a * b) / (2 * semiperimeter a b c)
-- State the theorem to prove
theorem right_triangle_equation : 
    ∀ a b c : ℕ, a^2 + b^2 = c^2 → semiperimeter a b c + inscribed_radius a b c = a + b := by
  sorry

end right_triangle_equation_l1080_108036


namespace roots_condition_l1080_108021

theorem roots_condition (r1 r2 p : ℝ) (h_eq : ∀ x : ℝ, x^2 + p * x + 12 = 0 → (x = r1 ∨ x = r2))
(h_distinct : r1 ≠ r2) (h_vieta1 : r1 + r2 = -p) (h_vieta2 : r1 * r2 = 12) : 
|r1| > 3 ∨ |r2| > 3 :=
by
  sorry

end roots_condition_l1080_108021


namespace sara_walking_distance_l1080_108078

noncomputable def circle_area := 616
noncomputable def pi_estimate := (22: ℚ) / 7
noncomputable def extra_distance := 3

theorem sara_walking_distance (r : ℚ) (radius_pos : 0 < r) : 
  pi_estimate * r^2 = circle_area →
  2 * pi_estimate * r + extra_distance = 91 :=
by
  intros h
  sorry

end sara_walking_distance_l1080_108078


namespace x_y_square_sum_l1080_108043

theorem x_y_square_sum (x y : ℝ) (h1 : x - y = -1) (h2 : x * y = 1 / 2) : x^2 + y^2 = 2 := 
by 
  sorry

end x_y_square_sum_l1080_108043


namespace base_conversion_b_l1080_108083

-- Define the problem in Lean
theorem base_conversion_b (b : ℕ) : 
  (b^2 + 2 * b - 16 = 0) → b = 4 := 
by
  intro h
  sorry

end base_conversion_b_l1080_108083


namespace range_of_a_l1080_108008

theorem range_of_a (a : ℝ) : 
  (∀ x, (x ≤ 1 ∨ x ≥ 3) ↔ ((a ≤ x ∧ x ≤ a + 1) → (x ≤ 1 ∨ x ≥ 3))) → 
  (a ≤ 0 ∨ a ≥ 3) :=
by
  sorry

end range_of_a_l1080_108008


namespace sum_of_midpoint_coordinates_l1080_108089

theorem sum_of_midpoint_coordinates : 
  let (x1, y1) := (4, 7)
  let (x2, y2) := (10, 19)
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 20 := sorry

end sum_of_midpoint_coordinates_l1080_108089


namespace base_length_of_isosceles_triangle_l1080_108068

-- Definitions for the problem
def isosceles_triangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] :=
  ∃ (AB BC : ℝ), AB = BC

-- The problem to prove
theorem base_length_of_isosceles_triangle
  {A B C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
  (AB BC : ℝ) (AC x : ℝ)
  (height_base : ℝ) (height_side : ℝ) 
  (h1 : AB = BC)
  (h2 : height_base = 10)
  (h3 : height_side = 12)
  (h4 : AC = x)
  (h5 : ∀ AE BD : ℝ, AE = height_side → BD = height_base) :
  x = 15 := by sorry

end base_length_of_isosceles_triangle_l1080_108068


namespace paper_pattern_after_unfolding_l1080_108063

-- Define the number of layers after folding the square paper four times
def folded_layers (initial_layers : ℕ) : ℕ :=
  initial_layers * 2 ^ 4

-- Define the number of quarter-circles removed based on the layers
def quarter_circles_removed (layers : ℕ) : ℕ :=
  layers

-- Define the number of complete circles from the quarter circles
def complete_circles (quarter_circles : ℕ) : ℕ :=
  quarter_circles / 4

-- The main theorem that we need to prove
theorem paper_pattern_after_unfolding :
  (complete_circles (quarter_circles_removed (folded_layers 1)) = 4) :=
by
  sorry

end paper_pattern_after_unfolding_l1080_108063


namespace range_of_a_l1080_108046

noncomputable def A (x : ℝ) : Prop := (3 * x) / (x + 1) ≤ 2
noncomputable def B (x a : ℝ) : Prop := a - 2 < x ∧ x < 2 * a + 1

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, A x ↔ B x a) ↔ (1 / 2 < a ∧ a ≤ 1) := by
sorry

end range_of_a_l1080_108046


namespace solve_equation_simplify_expression_l1080_108070

-- Part 1: Solving the equation
theorem solve_equation (x : ℝ) : 9 * (x - 3) ^ 2 - 121 = 0 ↔ x = 20 / 3 ∨ x = -2 / 3 :=
by 
    sorry

-- Part 2: Simplifying the expression
theorem simplify_expression (x y : ℝ) : (x - 2 * y) * (x ^ 2 + 2 * x * y + 4 * y ^ 2) = x ^ 3 - 8 * y ^ 3 :=
by 
    sorry

end solve_equation_simplify_expression_l1080_108070


namespace find_f_of_neg3_l1080_108065

noncomputable def f : ℚ → ℚ := sorry 

theorem find_f_of_neg3 (h : ∀ (x : ℚ) (hx : x ≠ 0), 5 * f (x⁻¹) + 3 * (f x) * x⁻¹ = 2 * x^2) :
  f (-3) = -891 / 22 :=
sorry

end find_f_of_neg3_l1080_108065


namespace simplify_fraction_l1080_108022

theorem simplify_fraction (a : ℕ) (h : a = 3) : (10 * a ^ 3) / (55 * a ^ 2) = 6 / 11 :=
by sorry

end simplify_fraction_l1080_108022


namespace solve_for_x_l1080_108071

theorem solve_for_x : 
  ∀ x : ℝ, 
    (x ≠ 2) ∧ (4 * x^2 + 3 * x + 1) / (x - 2) = 4 * x + 5 → 
    x = -11 / 6 :=
by
  intro x
  intro h 
  sorry

end solve_for_x_l1080_108071


namespace least_distance_between_ticks_l1080_108096

theorem least_distance_between_ticks :
  ∃ z : ℝ, ∀ (a b : ℤ), (a / 5 ≠ b / 7) → abs (a / 5 - b / 7) = (1 / 35) := 
sorry

end least_distance_between_ticks_l1080_108096


namespace total_trees_cut_down_l1080_108037

-- Definitions based on conditions in the problem
def trees_per_day_james : ℕ := 20
def days_with_just_james : ℕ := 2
def total_trees_by_james := trees_per_day_james * days_with_just_james

def brothers : ℕ := 2
def days_with_brothers : ℕ := 3
def trees_per_day_brothers := (20 * (100 - 20)) / 100 -- 20% fewer than James
def trees_per_day_total := brothers * trees_per_day_brothers + trees_per_day_james

def total_trees_with_brothers := trees_per_day_total * days_with_brothers

-- The statement to be proved
theorem total_trees_cut_down : total_trees_by_james + total_trees_with_brothers = 136 := by
  sorry

end total_trees_cut_down_l1080_108037


namespace two_thirds_greater_l1080_108025

theorem two_thirds_greater :
  let epsilon : ℚ := (2 : ℚ) / (3 * 10^8)
  let decimal_part : ℚ := 66666666 / 10^8
  (2 / 3) - decimal_part = epsilon := by
  sorry

end two_thirds_greater_l1080_108025


namespace brad_has_9_green_balloons_l1080_108029

theorem brad_has_9_green_balloons (total_balloons red_balloons : ℕ) (h_total : total_balloons = 17) (h_red : red_balloons = 8) : total_balloons - red_balloons = 9 :=
by {
  sorry
}

end brad_has_9_green_balloons_l1080_108029


namespace find_z_l1080_108099

variable (x y z : ℝ)

-- Define x, y as given in the problem statement
def x_def : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3) := by
  sorry

def y_def : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3) := by
  sorry

-- Define the equation relating z to x and y
def z_eq : 192 * z = x^4 + y^4 + (x + y)^4 := by 
  sorry

-- Theorem stating the value of z
theorem find_z (h1 : x = (Real.sqrt 7 + Real.sqrt 3) / (Real.sqrt 7 - Real.sqrt 3))
               (h2 : y = (Real.sqrt 7 - Real.sqrt 3) / (Real.sqrt 7 + Real.sqrt 3))
               (h3 : 192 * z = x^4 + y^4 + (x + y)^4) :
  z = 6 := by 
  sorry

end find_z_l1080_108099


namespace proof_op_nabla_l1080_108023

def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem proof_op_nabla :
  op_nabla (op_nabla (1/2) (1/3)) (1/4) = 9 / 11 := by
  sorry

end proof_op_nabla_l1080_108023


namespace vacation_cost_split_l1080_108059

theorem vacation_cost_split (t d : ℕ) 
  (h_total : 105 + 125 + 175 = 405)
  (h_split : 405 / 3 = 135)
  (h_t : t = 135 - 105)
  (h_d : d = 135 - 125) : 
  t - d = 20 := by
  sorry

end vacation_cost_split_l1080_108059


namespace shopkeeper_loss_l1080_108066

theorem shopkeeper_loss
    (total_stock : ℝ)
    (stock_sold_profit_percent : ℝ)
    (stock_profit_percent : ℝ)
    (stock_sold_loss_percent : ℝ)
    (stock_loss_percent : ℝ) :
    total_stock = 12500 →
    stock_sold_profit_percent = 0.20 →
    stock_profit_percent = 0.10 →
    stock_sold_loss_percent = 0.80 →
    stock_loss_percent = 0.05 →
    ∃ loss_amount, loss_amount = 250 :=
by
  sorry

end shopkeeper_loss_l1080_108066


namespace prime_difference_fourth_powers_is_not_prime_l1080_108016

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_difference_fourth_powers_is_not_prime (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p > q) : 
  ¬ is_prime (p^4 - q^4) :=
sorry

end prime_difference_fourth_powers_is_not_prime_l1080_108016


namespace tangent_line_slope_l1080_108024

theorem tangent_line_slope (k : ℝ) :
  (∃ m : ℝ, (m^3 - m^2 + m = k * m) ∧ (k = 3 * m^2 - 2 * m + 1)) →
  (k = 1 ∨ k = 3 / 4) :=
by
  -- Proof goes here
  sorry

end tangent_line_slope_l1080_108024


namespace toms_age_ratio_l1080_108072

variable (T N : ℕ)

def toms_age_condition : Prop :=
  T = 3 * (T - 4 * N) + N

theorem toms_age_ratio (h : toms_age_condition T N) : T / N = 11 / 2 :=
by sorry

end toms_age_ratio_l1080_108072


namespace min_colors_needed_l1080_108052

theorem min_colors_needed (n : ℕ) : 
  (n + (n * (n - 1)) / 2 ≥ 12) → (n = 5) :=
by
  sorry

end min_colors_needed_l1080_108052


namespace relay_team_order_count_l1080_108018

def num_ways_to_order_relay (total_members : Nat) (jordan_lap : Nat) : Nat :=
  if jordan_lap = total_members then (total_members - 1).factorial else 0

theorem relay_team_order_count : num_ways_to_order_relay 5 5 = 24 :=
by
  -- the proof would go here
  sorry

end relay_team_order_count_l1080_108018


namespace crystal_barrette_sets_l1080_108061

-- Definitional and situational context
def cost_of_barrette : ℕ := 3
def cost_of_comb : ℕ := 1
def kristine_total_cost : ℕ := 4
def total_spent : ℕ := 14

-- The Lean 4 theorem statement to prove that Crystal bought 3 sets of barrettes
theorem crystal_barrette_sets (x : ℕ) 
  (kristine_cost : kristine_total_cost = cost_of_barrette + cost_of_comb + 1)
  (total_cost_eq : kristine_total_cost + (x * cost_of_barrette + cost_of_comb) = total_spent) 
  : x = 3 := 
sorry

end crystal_barrette_sets_l1080_108061


namespace find_integer_l1080_108053

theorem find_integer (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : Real.cos (n * Real.pi / 180) = Real.sin (312 * Real.pi / 180)) :
  n = 42 :=
by
  sorry

end find_integer_l1080_108053


namespace positive_number_y_l1080_108000

theorem positive_number_y (y : ℕ) (h1 : y > 0) (h2 : y^2 / 100 = 9) : y = 30 :=
by
  sorry

end positive_number_y_l1080_108000


namespace maximum_ab_minimum_frac_minimum_exp_l1080_108054

variable {a b : ℝ}

theorem maximum_ab (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  ab <= 1/8 :=
sorry

theorem minimum_frac (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2/a + 1/b >= 8 :=
sorry

theorem minimum_exp (h1: a > 0) (h2: b > 0) (h3: a + 2 * b = 1) : 
  2^a + 4^b >= 2 * Real.sqrt 2 :=
sorry

end maximum_ab_minimum_frac_minimum_exp_l1080_108054


namespace second_discount_percentage_l1080_108094

theorem second_discount_percentage (x : ℝ) :
  9356.725146198829 * 0.8 * (1 - x / 100) * 0.95 = 6400 → x = 10 :=
by
  sorry

end second_discount_percentage_l1080_108094


namespace speed_of_man_l1080_108030

theorem speed_of_man 
  (L : ℝ) 
  (V_t : ℝ) 
  (T : ℝ) 
  (conversion_factor : ℝ) 
  (kmph_to_mps : ℝ → ℝ)
  (final_conversion : ℝ → ℝ) 
  (relative_speed : ℝ) 
  (Vm : ℝ) : Prop := 
L = 220 ∧ V_t = 59 ∧ T = 12 ∧ 
conversion_factor = 1000 / 3600 ∧ 
kmph_to_mps V_t = V_t * conversion_factor ∧ 
relative_speed = L / T ∧ 
Vm = relative_speed - (kmph_to_mps V_t) ∧ 
final_conversion Vm = Vm * 3.6 ∧ 
final_conversion Vm = 6.984

end speed_of_man_l1080_108030


namespace seeds_per_bed_l1080_108007

theorem seeds_per_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 60) (h2 : flower_beds = 6) : total_seeds / flower_beds = 10 := by
  sorry

end seeds_per_bed_l1080_108007


namespace find_f_3_l1080_108058

def f (x : ℝ) : ℝ := x^2 + 4 * x + 8

theorem find_f_3 : f 3 = 29 := by
  sorry

end find_f_3_l1080_108058


namespace probability_of_exactly_one_instrument_l1080_108050

-- Definitions
def total_people : ℕ := 800
def fraction_play_at_least_one_instrument : ℚ := 2 / 5
def num_play_two_or_more_instruments : ℕ := 96

-- Calculation
def num_play_at_least_one_instrument := fraction_play_at_least_one_instrument * total_people
def num_play_exactly_one_instrument := num_play_at_least_one_instrument - num_play_two_or_more_instruments

-- Probability calculation
def probability_play_exactly_one_instrument := num_play_exactly_one_instrument / total_people

-- Proof statement
theorem probability_of_exactly_one_instrument :
  probability_play_exactly_one_instrument = 0.28 := by
  sorry

end probability_of_exactly_one_instrument_l1080_108050


namespace group_total_cost_l1080_108074

noncomputable def total_cost
  (num_people : Nat) 
  (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem group_total_cost (num_people := 15) (cost_per_person := 900) :
  total_cost num_people cost_per_person = 13500 :=
by
  sorry

end group_total_cost_l1080_108074


namespace swimming_class_attendance_l1080_108032

def total_students : ℕ := 1000
def chess_ratio : ℝ := 0.25
def swimming_ratio : ℝ := 0.50

def chess_students := chess_ratio * total_students
def swimming_students := swimming_ratio * chess_students

theorem swimming_class_attendance :
  swimming_students = 125 :=
by
  sorry

end swimming_class_attendance_l1080_108032


namespace maximum_x1_x2_x3_l1080_108009

theorem maximum_x1_x2_x3 :
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
  x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
  x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
  x1 + x2 + x3 ≤ 61 := 
by sorry

end maximum_x1_x2_x3_l1080_108009


namespace solution_set_of_inequality_l1080_108028

theorem solution_set_of_inequality (a : ℝ) (h1 : 2 * a - 3 < 0) (h2 : 1 - a < 0) : 1 < a ∧ a < 3 / 2 :=
by
  sorry

end solution_set_of_inequality_l1080_108028


namespace weight_of_new_person_l1080_108041

-- Definitions for the conditions given.

-- Average weight increase
def avg_weight_increase : ℝ := 2.5

-- Number of persons
def num_persons : ℕ := 8

-- Weight of the person being replaced
def weight_replaced : ℝ := 65

-- Total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Statement to prove the weight of the new person
theorem weight_of_new_person : 
  ∃ (W_new : ℝ), W_new = weight_replaced + total_weight_increase :=
sorry

end weight_of_new_person_l1080_108041


namespace intersection_of_M_and_P_l1080_108097

def M : Set ℝ := { x | x^2 = x }
def P : Set ℝ := { x | |x - 1| = 1 }

theorem intersection_of_M_and_P : M ∩ P = {0} := by
  sorry

end intersection_of_M_and_P_l1080_108097


namespace find_q_l1080_108045

-- Given polynomial Q(x) with coefficients p, q, d
variables {p q d : ℝ}

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 + p * x^2 + q * x + d

-- Assume the conditions of the problem
theorem find_q (h1 : d = 5)                   -- y-intercept is 5
    (h2 : (-p / 3) = -d)                    -- mean of zeros = product of zeros
    (h3 : (-p / 3) = 1 + p + q + d)          -- mean of zeros = sum of coefficients
    : q = -26 := 
    sorry

end find_q_l1080_108045


namespace find_y_when_z_is_three_l1080_108015

theorem find_y_when_z_is_three
  (k : ℝ) (y z : ℝ)
  (h1 : y = 3)
  (h2 : z = 1)
  (h3 : y ^ 4 * z ^ 2 = k)
  (hc : z = 3) :
  y ^ 4 = 9 :=
sorry

end find_y_when_z_is_three_l1080_108015


namespace collinear_vectors_l1080_108017

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)

theorem collinear_vectors
  {a b m n : V}
  (h1 : m = a + b)
  (h2 : n = 2 • a + 2 • b)
  (h3 : not_collinear a b) :
  ∃ k : ℝ, k ≠ 0 ∧ n = k • m :=
by
  sorry

end collinear_vectors_l1080_108017


namespace find_numbers_l1080_108093

theorem find_numbers (a b : ℝ) (h1 : a * b = 5) (h2 : a + b = 8) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  (a = 4 + Real.sqrt 11 ∧ b = 4 - Real.sqrt 11) ∨ (a = 4 - Real.sqrt 11 ∧ b = 4 + Real.sqrt 11) :=
sorry

end find_numbers_l1080_108093


namespace find_ab_l1080_108038

theorem find_ab (a b : ℕ) (h1 : 1 <= a) (h2 : a < 10) (h3 : 0 <= b) (h4 : b < 10) (h5 : 66 * ((1 : ℝ) + ((10 * a + b : ℕ) / 100) - (↑(10 * a + b) / 99)) = 0.5) : 10 * a + b = 75 :=
by
  sorry

end find_ab_l1080_108038


namespace find_acid_percentage_l1080_108048

theorem find_acid_percentage (P : ℕ) (x : ℕ) (h1 : 4 + x = 20) 
  (h2 : x = 20 - 4) 
  (h3 : (P : ℝ)/100 * 4 + 0.75 * 16 = 0.72 * 20) : P = 60 :=
by
  sorry

end find_acid_percentage_l1080_108048


namespace cheese_cookie_packs_l1080_108002

def packs_per_box (P : ℕ) : Prop :=
  let cartons := 12
  let boxes_per_carton := 12
  let total_boxes := cartons * boxes_per_carton
  let total_cost := 1440
  let box_cost := total_cost / total_boxes
  let pack_cost := 1
  P = box_cost / pack_cost

theorem cheese_cookie_packs : packs_per_box 10 := by
  sorry

end cheese_cookie_packs_l1080_108002
