import Mathlib

namespace lyle_payment_l2259_225972

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end lyle_payment_l2259_225972


namespace number_of_candies_in_a_packet_l2259_225924

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l2259_225924


namespace instrument_failure_probability_l2259_225953

noncomputable def probability_of_instrument_not_working (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

theorem instrument_failure_probability (m : ℕ) (P : ℝ) :
  0 ≤ P → P ≤ 1 → probability_of_instrument_not_working m P = 1 - (1 - P)^m :=
by
  intros _ _
  sorry

end instrument_failure_probability_l2259_225953


namespace condition_neither_sufficient_nor_necessary_l2259_225958

theorem condition_neither_sufficient_nor_necessary (p q : Prop) :
  (¬ (p ∧ q)) → (p ∨ q) → False :=
by sorry

end condition_neither_sufficient_nor_necessary_l2259_225958


namespace exists_function_l2259_225907

theorem exists_function {n : ℕ} (hn : n ≥ 3) (S : Finset ℤ) (hS : S.card = n) :
  ∃ f : Fin (n) → S, 
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∀ i j k : Fin n, i < j ∧ j < k → 2 * (f j : ℤ) ≠ (f i : ℤ) + (f k : ℤ)) :=
by
  sorry

end exists_function_l2259_225907


namespace balance_proof_l2259_225948

variables (a b c : ℝ)

theorem balance_proof (h1 : 4 * a + 2 * b = 12 * c) (h2 : 2 * a = b + 3 * c) : 3 * b = 4.5 * c :=
sorry

end balance_proof_l2259_225948


namespace closest_integer_to_cube_root_of_150_l2259_225966

theorem closest_integer_to_cube_root_of_150 : 
  ∃ (n : ℤ), ∀ m : ℤ, abs (150 - 5 ^ 3) < abs (150 - m ^ 3) → n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l2259_225966


namespace geometric_seq_condition_l2259_225931

/-- In a geometric sequence with common ratio q, sum of the first n terms S_n.
  Given q > 0, show that it is a necessary condition for {S_n} to be an increasing sequence,
  but not a sufficient condition. -/
theorem geometric_seq_condition (a1 q : ℝ) (S : ℕ → ℝ)
  (hS : ∀ n, S n = a1 * (1 - q^n) / (1 - q))
  (h1 : q > 0) : 
  (∀ n, S n < S (n + 1)) ↔ a1 > 0 :=
sorry

end geometric_seq_condition_l2259_225931


namespace ratio_john_maya_age_l2259_225903

theorem ratio_john_maya_age :
  ∀ (john drew maya peter jacob : ℕ),
  -- Conditions:
  john = 30 ∧
  drew = maya + 5 ∧
  peter = drew + 4 ∧
  jacob = 11 ∧
  jacob + 2 = (peter + 2) / 2 →
  -- Conclusion:
  john / gcd john maya = 2 ∧ maya / gcd john maya = 1 :=
by
  sorry

end ratio_john_maya_age_l2259_225903


namespace find_c_l2259_225976

theorem find_c :
  ∃ c : ℝ, 0 < c ∧ ∀ line : ℝ, (∃ x y : ℝ, (x = 1 ∧ y = c) ∧ (x*x + y*y - 2*x - 2*y - 7 = 0)) ∧ (line = 1*x + 0 + y*c - 0) :=
sorry

end find_c_l2259_225976


namespace round_table_vip_arrangements_l2259_225975

-- Define the conditions
def number_of_people : ℕ := 10
def vip_seats : ℕ := 2

noncomputable def number_of_arrangements : ℕ :=
  let total_arrangements := Nat.factorial number_of_people
  let vip_choices := Nat.choose number_of_people vip_seats
  let remaining_arrangements := Nat.factorial (number_of_people - vip_seats)
  vip_choices * remaining_arrangements

-- Theorem stating the result
theorem round_table_vip_arrangements : number_of_arrangements = 1814400 := by
  sorry

end round_table_vip_arrangements_l2259_225975


namespace find_D_l2259_225955

theorem find_D (A D : ℝ) (h1 : D + A = 5) (h2 : D - A = -3) : D = 1 :=
by
  sorry

end find_D_l2259_225955


namespace total_surfers_calculation_l2259_225957

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l2259_225957


namespace travel_speed_l2259_225964

theorem travel_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 160) (h_time : time = 8) :
  ∃ speed : ℕ, speed = distance / time ∧ speed = 20 :=
by
  sorry

end travel_speed_l2259_225964


namespace gcd_of_products_l2259_225913

theorem gcd_of_products (a b a' b' d d' : ℕ) (h1 : Nat.gcd a b = d) (h2 : Nat.gcd a' b' = d') (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a') (hb' : 0 < b') :
  Nat.gcd (Nat.gcd (aa') (ab')) (Nat.gcd (ba') (bb')) = d * d' := 
sorry

end gcd_of_products_l2259_225913


namespace evaluate_g_g_g_25_l2259_225941

def g (x : ℤ) : ℤ :=
  if x < 10 then x^2 - 9 else x - 20

theorem evaluate_g_g_g_25 : g (g (g 25)) = -4 := by
  sorry

end evaluate_g_g_g_25_l2259_225941


namespace karthik_weight_average_l2259_225997

noncomputable def average_probable_weight_of_karthik (weight : ℝ) : Prop :=
  (55 < weight ∧ weight < 62) ∧
  (50 < weight ∧ weight < 60) ∧
  (weight ≤ 58) →
  weight = 56.5

theorem karthik_weight_average :
  ∀ weight : ℝ, average_probable_weight_of_karthik weight :=
by
  sorry

end karthik_weight_average_l2259_225997


namespace sequence_value_at_5_l2259_225909

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 / 3 ∧ ∀ n, 1 < n → a n = (-1) ^ n * 2 * a (n - 1)

theorem sequence_value_at_5 (a : ℕ → ℚ) (h : seq a) : a 5 = 16 / 3 :=
by 
  sorry

end sequence_value_at_5_l2259_225909


namespace calc_value_l2259_225983

def f (x : ℤ) : ℤ := x^2 + 5 * x + 4
def g (x : ℤ) : ℤ := 2 * x - 3

theorem calc_value :
  f (g (-3)) - 2 * g (f 2) = -26 := by
  sorry

end calc_value_l2259_225983


namespace cassandra_collected_pennies_l2259_225938

theorem cassandra_collected_pennies 
(C : ℕ) 
(h1 : ∀ J : ℕ,  J = C - 276) 
(h2 : ∀ J : ℕ, C + J = 9724) 
: C = 5000 := 
by
  sorry

end cassandra_collected_pennies_l2259_225938


namespace vic_max_marks_l2259_225969

theorem vic_max_marks (M : ℝ) (h : 0.92 * M = 368) : M = 400 := 
sorry

end vic_max_marks_l2259_225969


namespace evaluate_ratio_l2259_225946

theorem evaluate_ratio : (2^2003 * 3^2002) / (6^2002) = 2 := 
by {
  sorry
}

end evaluate_ratio_l2259_225946


namespace necessary_but_not_sufficient_l2259_225960

theorem necessary_but_not_sufficient (x : Real)
  (p : Prop := x < 1) 
  (q : Prop := x^2 + x - 2 < 0) 
  : p -> (q <-> x > -2 ∧ x < 1) ∧ (q -> p) → ¬ (p -> q) ∧ (x > -2 -> p) :=
by
  sorry

end necessary_but_not_sufficient_l2259_225960


namespace problem_proof_l2259_225927

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- The conditions
def even_function_on_ℝ := ∀ x : ℝ, f x = f (-x)
def f_at_0_is_2 := f 0 = 2
def odd_after_translation := ∀ x : ℝ, f (x - 1) = -f (-x - 1)

-- Prove the required condition
theorem problem_proof (h1 : even_function_on_ℝ f) (h2 : f_at_0_is_2 f) (h3 : odd_after_translation f) :
    f 1 + f 3 + f 5 + f 7 + f 9 = 0 :=
by
  sorry

end problem_proof_l2259_225927


namespace tank_capacity_l2259_225919

theorem tank_capacity (C : ℝ) :
  (3/4 * C - 0.4 * (3/4 * C) + 0.3 * (3/4 * C - 0.4 * (3/4 * C))) = 4680 → C = 8000 :=
by
  sorry

end tank_capacity_l2259_225919


namespace calculate_total_students_l2259_225987

-- Define the conditions and state the theorem
theorem calculate_total_students (perc_bio : ℝ) (num_not_bio : ℝ) (perc_not_bio : ℝ) (T : ℝ) :
  perc_bio = 0.475 →
  num_not_bio = 462 →
  perc_not_bio = 1 - perc_bio →
  perc_not_bio * T = num_not_bio →
  T = 880 :=
by
  intros
  -- proof will be here
  sorry

end calculate_total_students_l2259_225987


namespace min_value_of_expression_l2259_225910

theorem min_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x = a^2 + b^2 + (1 / (a + b)^2) + (1 / (a * b)) ∧ x = Real.sqrt 10 :=
sorry

end min_value_of_expression_l2259_225910


namespace sin_double_angle_identity_l2259_225988

theorem sin_double_angle_identity 
  (α : ℝ) 
  (h₁ : α ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h₂ : Real.sin α = 1 / 5) : 
  Real.sin (2 * α) = - (4 * Real.sqrt 6) / 25 :=
by
  sorry

end sin_double_angle_identity_l2259_225988


namespace percentage_of_exceedance_l2259_225970

theorem percentage_of_exceedance (x p : ℝ) (h : x = (p / 100) * x + 52.8) (hx : x = 60) : p = 12 :=
by 
  sorry

end percentage_of_exceedance_l2259_225970


namespace min_value_expression_l2259_225999

theorem min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π / 2 → 
    (3 * Real.sin θ' + 4 / Real.cos θ' + 2 * Real.sqrt 3 * Real.tan θ') ≥ 9 * Real.sqrt 3) ∧ 
    (3 * Real.sin θ + 4 / Real.cos θ + 2 * Real.sqrt 3 * Real.tan θ = 9 * Real.sqrt 3) :=
by
  sorry

end min_value_expression_l2259_225999


namespace quadratic_vertex_problem_l2259_225934

/-- 
    Given a quadratic equation y = ax^2 + bx + c, where (2, -3) 
    is the vertex of the parabola and it passes through (0, 1), 
    prove that a - b + c = 6. 
-/
theorem quadratic_vertex_problem 
    (a b c : ℤ)
    (h : ∀ x : ℝ, y = a * (x - 2)^2 - 3)
    (h_point : y = 1)
    (h_passes_through_origin : y = a * (0 - 2)^2 - 3) :
    a - b + c = 6 :=
sorry

end quadratic_vertex_problem_l2259_225934


namespace certain_number_exists_l2259_225917

theorem certain_number_exists
  (N : ℕ) 
  (hN : ∀ x, x < N → x % 2 = 1 → ∃ k m, k = 5 * m ∧ x = k ∧ m % 2 = 1) :
  N = 76 := by
  sorry

end certain_number_exists_l2259_225917


namespace lathes_equal_parts_processed_15_minutes_l2259_225990

variable (efficiencyA efficiencyB efficiencyC : ℝ)
variable (timeA timeB timeC : ℕ)

/-- Lathe A starts 10 minutes before lathe C -/
def start_time_relation_1 : Prop := timeA + 10 = timeC

/-- Lathe C starts 5 minutes before lathe B -/
def start_time_relation_2 : Prop := timeC + 5 = timeB

/-- After lathe B has been working for 10 minutes, B and C process the same number of parts -/
def parts_processed_relation_1 (efficiencyB efficiencyC : ℝ) : Prop :=
  10 * efficiencyB = (10 + 5) * efficiencyC

/-- After lathe C has been working for 30 minutes, A and C process the same number of parts -/
def parts_processed_relation_2 (efficiencyA efficiencyC : ℝ) : Prop :=
  (30 + 10) * efficiencyA = 30 * efficiencyC

/-- How many minutes after lathe B starts will it have processed the same number of standard parts as lathe A? -/
theorem lathes_equal_parts_processed_15_minutes
  (h₁ : start_time_relation_1 timeA timeC)
  (h₂ : start_time_relation_2 timeC timeB)
  (h₃ : parts_processed_relation_1 efficiencyB efficiencyC)
  (h₄ : parts_processed_relation_2 efficiencyA efficiencyC) :
  ∃ t : ℕ, (t = 15) ∧ ( (timeB + t) * efficiencyB = (timeA + (timeB + t - timeA)) * efficiencyA ) := sorry

end lathes_equal_parts_processed_15_minutes_l2259_225990


namespace algebraic_expression_correct_l2259_225965

theorem algebraic_expression_correct (x y : ℝ) :
  (x - y)^2 - (x^2 - y^2) = (x - y)^2 - (x^2 - y^2) :=
by
  sorry

end algebraic_expression_correct_l2259_225965


namespace simplify_expression_l2259_225915

theorem simplify_expression : 
  (2 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7) = 
  Real.sqrt 6 + 2 * Real.sqrt 2 - Real.sqrt 14 :=
sorry

end simplify_expression_l2259_225915


namespace multiples_7_not_14_l2259_225963

theorem multiples_7_not_14 (n : ℕ) : (n < 500) → (n % 7 = 0) → (n % 14 ≠ 0) → ∃ k, (k = 36) :=
by {
  sorry
}

end multiples_7_not_14_l2259_225963


namespace gumballs_difference_l2259_225921

theorem gumballs_difference :
  ∃ (x_min x_max : ℕ), 
    19 ≤ (16 + 12 + x_min) / 3 ∧ (16 + 12 + x_min) / 3 ≤ 25 ∧
    19 ≤ (16 + 12 + x_max) / 3 ∧ (16 + 12 + x_max) / 3 ≤ 25 ∧
    (x_max - x_min = 18) :=
by
  sorry

end gumballs_difference_l2259_225921


namespace difference_of_quarters_l2259_225929

variables (n d q : ℕ)

theorem difference_of_quarters :
  (n + d + q = 150) ∧ (5 * n + 10 * d + 25 * q = 1425) →
  (∃ qmin qmax : ℕ, q = qmax - qmin ∧ qmax - qmin = 30) :=
by
  sorry

end difference_of_quarters_l2259_225929


namespace incorrect_statement_about_GIS_l2259_225954

def statement_A := "GIS can provide information for geographic decision-making"
def statement_B := "GIS are computer systems specifically designed to process geographic spatial data"
def statement_C := "Urban management is one of the earliest and most effective fields of GIS application"
def statement_D := "GIS's main functions include data collection, data analysis, decision-making applications, etc."

def correct_answer := statement_B

theorem incorrect_statement_about_GIS:
  correct_answer = statement_B := 
sorry

end incorrect_statement_about_GIS_l2259_225954


namespace max_snowmen_l2259_225956

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end max_snowmen_l2259_225956


namespace intersection_sum_l2259_225992

noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1) ^ 2 / 3

theorem intersection_sum :
  ∃ a b : ℝ, f a = f (a - 4) ∧ b = f a ∧ a + b = 16 / 3 :=
sorry

end intersection_sum_l2259_225992


namespace gabi_final_prices_l2259_225980

theorem gabi_final_prices (x y : ℝ) (hx : 0.8 * x = 1.2 * y) (hl : (x - 0.8 * x) + (y - 1.2 * y) = 10) :
  x = 30 ∧ y = 20 := sorry

end gabi_final_prices_l2259_225980


namespace term_61_is_201_l2259_225995

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a5 : ℤ)

-- Define the general formula for the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a5 + (n - 5) * d

-- Given variables and conditions:
axiom h1 : a5 = 33
axiom h2 : d = 3

theorem term_61_is_201 :
  arithmetic_sequence a5 d 61 = 201 :=
by
  -- proof here
  sorry

end term_61_is_201_l2259_225995


namespace place_value_ratio_l2259_225950

def number : ℝ := 90347.6208
def place_value_0 : ℝ := 10000 -- tens of thousands
def place_value_6 : ℝ := 0.1 -- tenths

theorem place_value_ratio : 
  place_value_0 / place_value_6 = 100000 := by 
    sorry

end place_value_ratio_l2259_225950


namespace quadratic_trinomial_neg_values_l2259_225996

theorem quadratic_trinomial_neg_values (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by
sorry

end quadratic_trinomial_neg_values_l2259_225996


namespace part1_part2_l2259_225962

open Real

noncomputable def a_value := 2 * sqrt 2

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  x + y - 4 = 0

noncomputable def point_on_line (ρ θ : ℝ) :=
  ρ * cos (θ - π / 4) = a_value

noncomputable def curve_param_eqns (θ : ℝ) : (ℝ × ℝ) :=
  (sqrt 3 * cos θ, sin θ)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / sqrt 2

theorem part1 (P : ℝ × ℝ) (ρ θ : ℝ) : 
  P = (4, π / 2) ∧ point_on_line ρ θ → 
  a_value = 2 * sqrt 2 ∧ line_cartesian_eqn 4 (4 * tan (π / 4)) :=
sorry

theorem part2 :
  (∀ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) ≤ 3 * sqrt 2) ∧
  (∃ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) = 3 * sqrt 2) :=
sorry

end part1_part2_l2259_225962


namespace f_even_l2259_225949

-- Define E_x^n as specified
def E_x (n : ℕ) (x : ℝ) : ℝ := List.prod (List.map (λ i => x + i) (List.range n))

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * E_x 5 (x - 2)

-- Define the statement to prove f(x) is even
theorem f_even (x : ℝ) : f x = f (-x) := by
  sorry

end f_even_l2259_225949


namespace solution_set_of_inequality_l2259_225901

theorem solution_set_of_inequality
  (a b : ℝ)
  (x y : ℝ)
  (h1 : a * (-2) + b = 3)
  (h2 : a * (-1) + b = 2)
  :  -x + 1 < 0 ↔ x > 1 :=
by 
  -- Proof goes here
  sorry

end solution_set_of_inequality_l2259_225901


namespace not_possible_linear_poly_conditions_l2259_225902

theorem not_possible_linear_poly_conditions (a b : ℝ):
    ¬ (abs (b - 1) < 1 ∧ abs (a + b - 3) < 1 ∧ abs (2 * a + b - 9) < 1) := 
by
    sorry

end not_possible_linear_poly_conditions_l2259_225902


namespace part1_part2_l2259_225912

def A (x : ℝ) : Prop := x < -2 ∨ x > 3
def B (a : ℝ) (x : ℝ) : Prop := 1 - a < x ∧ x < a + 3

theorem part1 (x : ℝ) : (¬A x ∨ B 1 x) ↔ -2 ≤ x ∧ x < 4 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, ¬(A x ∧ B a x)) ↔ a ≤ 0 :=
by
  sorry

end part1_part2_l2259_225912


namespace speed_of_boat_in_still_water_12_l2259_225932

theorem speed_of_boat_in_still_water_12 (d b c : ℝ) (h1 : d = (b - c) * 5) (h2 : d = (b + c) * 3) (hb : b = 12) : b = 12 :=
by
  sorry

end speed_of_boat_in_still_water_12_l2259_225932


namespace rectangular_prism_inequality_l2259_225977

variable {a b c l : ℝ}

theorem rectangular_prism_inequality (h_diag : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
sorry

end rectangular_prism_inequality_l2259_225977


namespace sale_price_tea_correct_l2259_225922

noncomputable def sale_price_of_mixed_tea (weight1 weight2 price1 price2 profit_percentage : ℝ) : ℝ :=
let total_cost := weight1 * price1 + weight2 * price2
let total_weight := weight1 + weight2
let cost_price_per_kg := total_cost / total_weight
let profit_per_kg := profit_percentage * cost_price_per_kg
let sale_price_per_kg := cost_price_per_kg + profit_per_kg
sale_price_per_kg

theorem sale_price_tea_correct :
  sale_price_of_mixed_tea 80 20 15 20 0.20 = 19.2 :=
  by
  sorry

end sale_price_tea_correct_l2259_225922


namespace invalid_votes_l2259_225926

theorem invalid_votes (W L total_polls : ℕ) 
  (h1 : total_polls = 90830) 
  (h2 : L = 9 * W / 11) 
  (h3 : W = L + 9000)
  (h4 : 100 * (W + L) = 90000) : 
  total_polls - (W + L) = 830 := 
sorry

end invalid_votes_l2259_225926


namespace functional_equation_identity_l2259_225971

def f : ℝ → ℝ := sorry

theorem functional_equation_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) : 
  ∀ y : ℝ, f y = y :=
sorry

end functional_equation_identity_l2259_225971


namespace product_pqr_l2259_225940

/-- Mathematical problem statement -/
theorem product_pqr (p q r : ℤ) (hp: p ≠ 0) (hq: q ≠ 0) (hr: r ≠ 0)
  (h1 : p + q + r = 36)
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 540 / (p * q * r) = 1) :
  p * q * r = 864 :=
sorry

end product_pqr_l2259_225940


namespace original_average_is_6_2_l2259_225933

theorem original_average_is_6_2 (n : ℕ) (S : ℚ) (h1 : 6.2 = S / n) (h2 : 6.6 = (S + 4) / n) :
  6.2 = S / n :=
by
  sorry

end original_average_is_6_2_l2259_225933


namespace total_amount_paid_l2259_225951

def original_price : ℝ := 20
def discount_rate : ℝ := 0.5
def number_of_tshirts : ℕ := 6

theorem total_amount_paid : 
  (number_of_tshirts : ℝ) * (original_price * discount_rate) = 60 := by
  sorry

end total_amount_paid_l2259_225951


namespace pages_used_l2259_225943

variable (n o c : ℕ)

theorem pages_used (h_n : n = 3) (h_o : o = 13) (h_c : c = 8) :
  (n + o) / c = 2 :=
  by
    sorry

end pages_used_l2259_225943


namespace range_of_a_l2259_225985

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (π / 6 < x) → (x < π / 3) → (f a x) ≤ (f a (x + ε))) → 2 ≤ a :=
by
  sorry

end range_of_a_l2259_225985


namespace problem_l2259_225911

open Real

noncomputable def f (x : ℝ) : ℝ := exp (2 * x) + 2 * cos x - 4

theorem problem (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * π) : 
  ∀ a b : ℝ, (0 ≤ a ∧ a ≤ 2 * π) → (0 ≤ b ∧ b ≤ 2 * π) → a ≤ b → f a ≤ f b := 
sorry

end problem_l2259_225911


namespace minimum_value_m_sq_plus_n_sq_l2259_225900

theorem minimum_value_m_sq_plus_n_sq :
  ∃ (m n : ℝ), (m ≠ 0) ∧ (∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ (m * x^2 + (2 * n + 1) * x - m - 2) = 0) ∧
  (m^2 + n^2) = 0.01 :=
by
  sorry

end minimum_value_m_sq_plus_n_sq_l2259_225900


namespace decompose_96_l2259_225942

theorem decompose_96 (a b : ℤ) (h1 : a * b = 96) (h2 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) ∨ (a = -8 ∧ b = -12) ∨ (a = -12 ∧ b = -8) :=
by
  sorry

end decompose_96_l2259_225942


namespace min_lamps_l2259_225989

theorem min_lamps (n p : ℕ) (h1: p > 0) (h_total_profit : 3 * (3 * p / 4 / n) + (n - 3) * (p / n + 10) - p = 100) : n = 13 :=
by
  sorry

end min_lamps_l2259_225989


namespace kenneth_distance_past_finish_l2259_225945

noncomputable def distance_past_finish_line (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) : ℕ :=
  let biff_time := race_distance / biff_speed
  let kenneth_distance := kenneth_speed * biff_time
  kenneth_distance - race_distance

theorem kenneth_distance_past_finish (race_distance : ℕ) (biff_speed : ℕ) (kenneth_speed : ℕ) (finish_line_distance : ℕ) : 
  race_distance = 500 ->
  biff_speed = 50 -> 
  kenneth_speed = 51 ->
  finish_line_distance = 10 ->
  distance_past_finish_line race_distance biff_speed kenneth_speed = finish_line_distance := by
  sorry

end kenneth_distance_past_finish_l2259_225945


namespace point_transformation_correct_l2259_225979

-- Define the rectangular coordinate system O-xyz
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the point in the original coordinate system
def originalPoint : Point3D := { x := 1, y := -2, z := 3 }

-- Define the transformation function for the yOz plane
def transformToYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

-- Define the expected transformed point
def transformedPoint : Point3D := { x := -1, y := -2, z := 3 }

-- State the theorem to be proved
theorem point_transformation_correct :
  transformToYOzPlane originalPoint = transformedPoint :=
by
  sorry

end point_transformation_correct_l2259_225979


namespace father_l2259_225925

theorem father's_age_at_middle_son_birth (a b c F : ℕ) 
  (h1 : a = b + c) 
  (h2 : F * a * b * c = 27090) : 
  F - b = 34 :=
by sorry

end father_l2259_225925


namespace seq_a8_value_l2259_225908

theorem seq_a8_value 
  (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) 
  (a7_eq : a 7 = 120) 
  : a 8 = 194 :=
sorry

end seq_a8_value_l2259_225908


namespace range_of_independent_variable_l2259_225920

theorem range_of_independent_variable (x : ℝ) : 
  (y = 3 / (x + 2)) → (x ≠ -2) :=
by
  -- suppose the function y = 3 / (x + 2) is given
  -- we need to prove x ≠ -2 for the function to be defined
  sorry

end range_of_independent_variable_l2259_225920


namespace rectangles_in_5x5_grid_l2259_225994

theorem rectangles_in_5x5_grid : 
  ∃ n : ℕ, n = 100 ∧ (∀ (grid : Fin 6 → Fin 6 → Prop), 
  (∃ (vlines hlines : Finset (Fin 6)),
   (vlines.card = 2 ∧ hlines.card = 2) ∧
   n = (vlines.card.choose 2) * (hlines.card.choose 2))) :=
by
  sorry

end rectangles_in_5x5_grid_l2259_225994


namespace restaurant_total_earnings_l2259_225981

noncomputable def restaurant_earnings (weekdays weekends : ℕ) (weekday_earnings : ℝ) 
    (weekend_min_earnings weekend_max_earnings discount special_event_earnings : ℝ) : ℝ :=
  let num_mondays := weekdays / 5 
  let weekday_earnings_with_discount := weekday_earnings - (weekday_earnings * discount)
  let earnings_mondays := num_mondays * weekday_earnings_with_discount
  let earnings_other_weekdays := (weekdays - num_mondays) * weekday_earnings
  let average_weekend_earnings := (weekend_min_earnings + weekend_max_earnings) / 2
  let total_weekday_earnings := earnings_mondays + earnings_other_weekdays
  let total_weekend_earnings := 2 * weekends * average_weekend_earnings
  total_weekday_earnings + total_weekend_earnings + special_event_earnings

theorem restaurant_total_earnings 
  (weekdays weekends : ℕ)
  (weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings total_earnings : ℝ)
  (h_weekdays : weekdays = 22)
  (h_weekends : weekends = 8)
  (h_weekday_earnings : weekday_earnings = 600)
  (h_weekend_min_earnings : weekend_min_earnings = 1000)
  (h_weekend_max_earnings : weekend_max_earnings = 1500)
  (h_discount : discount = 0.1)
  (h_special_event_earnings : special_event_earnings = 500)
  (h_total_earnings : total_earnings = 33460) :
  restaurant_earnings weekdays weekends weekday_earnings weekend_min_earnings weekend_max_earnings discount special_event_earnings = total_earnings := 
by
  sorry

end restaurant_total_earnings_l2259_225981


namespace gcd_cubed_and_sum_l2259_225991

theorem gcd_cubed_and_sum (n : ℕ) (h_pos : 0 < n) (h_gt_square : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := 
sorry

end gcd_cubed_and_sum_l2259_225991


namespace tan_sub_eq_minus_2sqrt3_l2259_225967

theorem tan_sub_eq_minus_2sqrt3 
  (h1 : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3)
  (h2 : Real.tan (5 * Real.pi / 12) = 2 + Real.sqrt 3) : 
  Real.tan (Real.pi / 12) - Real.tan (5 * Real.pi / 12) = -2 * Real.sqrt 3 :=
by
  sorry

end tan_sub_eq_minus_2sqrt3_l2259_225967


namespace combined_salaries_l2259_225986

-- Define the variables and constants corresponding to the conditions
variable (A B D E C : ℝ)
variable (avg_salary : ℝ)
variable (num_individuals : ℕ)

-- Given conditions translated into Lean definitions 
def salary_C : ℝ := 15000
def average_salary : ℝ := 8800
def number_of_individuals : ℕ := 5

-- Define the statement to prove
theorem combined_salaries (h1 : C = salary_C) (h2 : avg_salary = average_salary) (h3 : num_individuals = number_of_individuals) : 
  A + B + D + E = avg_salary * num_individuals - salary_C := 
by 
  -- Here the proof would involve calculating the total salary and subtracting C's salary
  sorry

end combined_salaries_l2259_225986


namespace Kolya_Homework_Problem_l2259_225904

-- Given conditions as definitions
def squaresToDigits (x : ℕ) (a b : ℕ) : Prop := x^2 = 10 * a + b
def doubledToDigits (x : ℕ) (a b : ℕ) : Prop := 2 * x = 10 * b + a

-- The main theorem statement
theorem Kolya_Homework_Problem :
  ∃ (x a b : ℕ), squaresToDigits x a b ∧ doubledToDigits x a b ∧ x = 9 ∧ x^2 = 81 :=
by
  -- proof skipped
  sorry

end Kolya_Homework_Problem_l2259_225904


namespace value_of_y_l2259_225961

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 8) (h2 : x = 2) : y = 3 / 2 :=
by
  sorry

end value_of_y_l2259_225961


namespace cos_210_eq_neg_sqrt3_div_2_l2259_225905

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_eq_neg_sqrt3_div_2_l2259_225905


namespace max_band_members_l2259_225916

theorem max_band_members (r x m : ℕ) (h1 : m < 150) (h2 : r * x + 3 = m) (h3 : (r - 3) * (x + 2) = m) : m = 147 := by
  sorry

end max_band_members_l2259_225916


namespace option_D_correct_l2259_225944

-- Defining the types for lines and planes
variables {Line Plane : Type}

-- Defining what's needed for perpendicularity and parallelism
variables (perp : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (parallel : Line → Line → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Main theorem statement
theorem option_D_correct (a b : Line) (α β : Plane) :
  perp a α → subset b β → parallel a b → perp_planes α β :=
by
  sorry

end option_D_correct_l2259_225944


namespace isaiah_types_more_words_than_micah_l2259_225959

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end isaiah_types_more_words_than_micah_l2259_225959


namespace cos_3theta_value_l2259_225998

open Real

noncomputable def cos_3theta (theta : ℝ) : ℝ := 4 * (cos theta)^3 - 3 * (cos theta)

theorem cos_3theta_value (theta : ℝ) (h : cos theta = 1 / 3) : cos_3theta theta = - 23 / 27 :=
by
  sorry

end cos_3theta_value_l2259_225998


namespace oranges_for_juice_l2259_225914

-- Define conditions
def total_oranges : ℝ := 7 -- in million tons
def export_percentage : ℝ := 0.25
def juice_percentage : ℝ := 0.60

-- Define the mathematical problem
theorem oranges_for_juice : 
  (total_oranges * (1 - export_percentage) * juice_percentage) = 3.2 :=
by
  sorry

end oranges_for_juice_l2259_225914


namespace toys_in_stock_l2259_225982

theorem toys_in_stock (sold_first_week sold_second_week toys_left toys_initial: ℕ) :
  sold_first_week = 38 → 
  sold_second_week = 26 → 
  toys_left = 19 → 
  toys_initial = sold_first_week + sold_second_week + toys_left → 
  toys_initial = 83 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end toys_in_stock_l2259_225982


namespace range_of_a_l2259_225918

theorem range_of_a (a : ℝ) : (1 < a) → 
  ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ 
  (1 / (x1 + 2) = a * |x1| ∧ 1 / (x2 + 2) = a * |x2| ∧ 1 / (x3 + 2) = a * |x3|) :=
sorry

end range_of_a_l2259_225918


namespace scale_model_height_l2259_225937

theorem scale_model_height (real_height : ℕ) (scale_ratio : ℕ) (h_real : real_height = 1454) (h_scale : scale_ratio = 50) : 
⌊(real_height : ℝ) / scale_ratio + 0.5⌋ = 29 :=
by
  rw [h_real, h_scale]
  norm_num
  sorry

end scale_model_height_l2259_225937


namespace R_transformed_is_R_l2259_225939

-- Define the initial coordinates of the rectangle PQRS
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (6, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (3, 1)

-- Define the reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the translation down by 2 units
def translate_down_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 - 2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the translation up by 2 units
def translate_up_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 2)

-- Define the transformation to find R''
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up_2 (reflect_y_neg_x (translate_down_2 (reflect_x p)))

-- Prove that the result of transforming R is (-3, -4)
theorem R_transformed_is_R'' : transform R = (-3, -4) :=
  by sorry

end R_transformed_is_R_l2259_225939


namespace ratio_of_sleep_l2259_225952

theorem ratio_of_sleep (connor_sleep : ℝ) (luke_extra : ℝ) (puppy_sleep : ℝ) 
    (h1 : connor_sleep = 6)
    (h2 : luke_extra = 2)
    (h3 : puppy_sleep = 16) :
    puppy_sleep / (connor_sleep + luke_extra) = 2 := 
by 
  sorry

end ratio_of_sleep_l2259_225952


namespace convert_2e_15pi_i4_to_rectangular_form_l2259_225973

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  let θ := (15 * Real.pi) / 4
  let θ' := θ - 2 * Real.pi
  2 * Complex.exp (θ' * Complex.I)

theorem convert_2e_15pi_i4_to_rectangular_form :
  convert_to_rectangular_form (2 * Complex.exp ((15 * Real.pi) / 4 * Complex.I)) = (Real.sqrt 2 - Complex.I * Real.sqrt 2) :=
  sorry

end convert_2e_15pi_i4_to_rectangular_form_l2259_225973


namespace function_zero_solution_l2259_225930

def floor (x : ℝ) : ℤ := sorry -- Define floor function properly.

theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = (-1) ^ (floor y) * f x + (-1) ^ (floor x) * f y) →
  (∀ x : ℝ, f x = 0) := 
by
  -- Proof goes here
  sorry

end function_zero_solution_l2259_225930


namespace tan_135_eq_neg1_l2259_225947

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg1_l2259_225947


namespace percentage_salt_solution_l2259_225993

-- Definitions
def P : ℝ := 60
def ounces_added := 40
def initial_solution_ounces := 40
def initial_solution_percentage := 0.20
def final_solution_percentage := 0.40
def final_solution_ounces := 80

-- Lean Statement
theorem percentage_salt_solution (P : ℝ) :
  (8 + 0.01 * P * ounces_added) = 0.40 * final_solution_ounces → P = 60 := 
by
  sorry

end percentage_salt_solution_l2259_225993


namespace value_of_expression_l2259_225928

theorem value_of_expression (x y : ℝ) (h1 : x = -2) (h2 : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 :=
by
  sorry

end value_of_expression_l2259_225928


namespace evaluate_expression_l2259_225936

theorem evaluate_expression (x y : ℝ) (P Q : ℝ) 
  (hP : P = x^2 + y^2) 
  (hQ : Q = x - y) : 
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by 
  -- Insert proof here
  sorry

end evaluate_expression_l2259_225936


namespace part1_l2259_225906

noncomputable def f (x a : ℝ) : ℝ := x - a * Real.log x

theorem part1 (x : ℝ) (hx : x > 0) : 
  f x 1 ≥ 1 :=
sorry

end part1_l2259_225906


namespace quadratic_solution_l2259_225923

theorem quadratic_solution : 
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ (x = 3 ∨ x = -1) :=
by {
  sorry
}

end quadratic_solution_l2259_225923


namespace total_handshakes_l2259_225974

def people := 40
def groupA := 25
def groupB := 15
def knownByGroupB (x : ℕ) : ℕ := 5
def interactionsWithinGroupB : ℕ := 105
def interactionsBetweenGroups : ℕ := 75

theorem total_handshakes : (groupB * knownByGroupB 0) + interactionsWithinGroupB = 180 :=
by
  sorry

end total_handshakes_l2259_225974


namespace number_of_blueberries_l2259_225935

def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blackberries : ℕ := total_berries / 3
def blueberries : ℕ := total_berries - (raspberries + blackberries)

theorem number_of_blueberries :
  blueberries = 7 :=
by
  sorry

end number_of_blueberries_l2259_225935


namespace value_of_m_l2259_225968

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l2259_225968


namespace intersection_of_M_and_N_l2259_225978

open Set

noncomputable def M := {x : ℝ | ∃ y:ℝ, y = Real.log (2 - x)}
noncomputable def N := {x : ℝ | x^2 - 3*x - 4 ≤ 0 }
noncomputable def I := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = I := 
  sorry

end intersection_of_M_and_N_l2259_225978


namespace f_plus_g_eq_l2259_225984

variables {R : Type*} [CommRing R]

-- Define the odd function f
def f (x : R) : R := sorry

-- Define the even function g
def g (x : R) : R := sorry

-- Define that f is odd and g is even
axiom f_odd (x : R) : f (-x) = -f x
axiom g_even (x : R) : g (-x) = g x

-- Define the given equation
axiom f_minus_g_eq (x : R) : f x - g x = x ^ 2 + 9 * x + 12

-- Statement of the goal
theorem f_plus_g_eq (x : R) : f x + g x = -x ^ 2 + 9 * x - 12 := by
  sorry

end f_plus_g_eq_l2259_225984
