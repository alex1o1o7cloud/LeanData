import Mathlib

namespace gertrude_fleas_l2169_216986

variables (G M O : ℕ)

def fleas_maud := M = 5 * O
def fleas_olive := O = G / 2
def total_fleas := G + M + O = 40

theorem gertrude_fleas
  (h_maud : fleas_maud M O)
  (h_olive : fleas_olive G O)
  (h_total : total_fleas G M O) :
  G = 10 :=
sorry

end gertrude_fleas_l2169_216986


namespace domain_of_h_l2169_216966

def domain_f : Set ℝ := {x | -10 ≤ x ∧ x ≤ 3}

def h_dom := {x | -3 * x ∈ domain_f}

theorem domain_of_h :
  h_dom = {x | x ≥ 10 / 3} :=
by
  sorry

end domain_of_h_l2169_216966


namespace coffee_amount_l2169_216900

theorem coffee_amount (total_mass : ℕ) (coffee_ratio : ℕ) (milk_ratio : ℕ) (h_total_mass : total_mass = 4400) (h_coffee_ratio : coffee_ratio = 2) (h_milk_ratio : milk_ratio = 9) : 
  total_mass * coffee_ratio / (coffee_ratio + milk_ratio) = 800 :=
by
  -- Placeholder for the proof
  sorry

end coffee_amount_l2169_216900


namespace total_spent_is_correct_l2169_216930

def cost_of_lunch : ℝ := 60.50
def tip_percentage : ℝ := 0.20
def tip_amount : ℝ := cost_of_lunch * tip_percentage
def total_amount_spent : ℝ := cost_of_lunch + tip_amount

theorem total_spent_is_correct : total_amount_spent = 72.60 := by
  -- placeholder for the proof
  sorry

end total_spent_is_correct_l2169_216930


namespace solution_set_quadratic_inequality_l2169_216909

theorem solution_set_quadratic_inequality :
  { x : ℝ | x^2 + 3 * x - 4 < 0 } = { x : ℝ | -4 < x ∧ x < 1 } :=
sorry

end solution_set_quadratic_inequality_l2169_216909


namespace percentage_reduction_is_58_perc_l2169_216953

-- Define the conditions
def initial_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.7 * P
def increased_price (P : ℝ) : ℝ := 1.2 * (discount_price P)
def clearance_price (P : ℝ) : ℝ := 0.5 * (increased_price P)

-- The statement of the proof problem
theorem percentage_reduction_is_58_perc (P : ℝ) (h : P > 0) :
  (1 - (clearance_price P / initial_price P)) * 100 = 58 :=
by
  -- Proof omitted
  sorry

end percentage_reduction_is_58_perc_l2169_216953


namespace comb_eq_l2169_216969

theorem comb_eq {n : ℕ} (h : Nat.choose 18 n = Nat.choose 18 2) : n = 2 ∨ n = 16 :=
by
  sorry

end comb_eq_l2169_216969


namespace geometric_seq_sum_four_and_five_l2169_216963

noncomputable def geom_seq (a₁ q : ℝ) (n : ℕ) := a₁ * q^(n-1)

theorem geometric_seq_sum_four_and_five :
  (∀ n, geom_seq a₁ q n > 0) →
  geom_seq a₁ q 3 = 4 →
  geom_seq a₁ q 6 = 1 / 2 →
  geom_seq a₁ q 4 + geom_seq a₁ q 5 = 3 :=
by
  sorry

end geometric_seq_sum_four_and_five_l2169_216963


namespace smallest_n_not_divisible_by_10_l2169_216916

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l2169_216916


namespace thirty_percent_of_x_l2169_216946

noncomputable def x : ℝ := 160 / 0.40

theorem thirty_percent_of_x (h : 0.40 * x = 160) : 0.30 * x = 120 :=
sorry

end thirty_percent_of_x_l2169_216946


namespace distance_EC_l2169_216956

-- Define the points and given distances as conditions
structure Points :=
  (A B C D E : Type)

-- Distances between points
variables {Points : Type}
variables (dAB dBC dCD dDE dEA dEC : ℝ)
variables [Nonempty Points]

-- Specify conditions: distances in kilometers
def distances_given (dAB dBC dCD dDE dEA : ℝ) : Prop :=
  dAB = 30 ∧ dBC = 80 ∧ dCD = 236 ∧ dDE = 86 ∧ dEA = 40

-- Main theorem: prove that the distance from E to C is 63.4 km
theorem distance_EC (h : distances_given 30 80 236 86 40) : dEC = 63.4 :=
sorry

end distance_EC_l2169_216956


namespace cat_mouse_positions_after_247_moves_l2169_216982

-- Definitions for Positions:
inductive Position
| TopLeft
| TopRight
| BottomRight
| BottomLeft
| TopMiddle
| RightMiddle
| BottomMiddle
| LeftMiddle

open Position

-- Function to calculate position of the cat
def cat_position (n : ℕ) : Position :=
  match n % 4 with
  | 0 => TopLeft
  | 1 => TopRight
  | 2 => BottomRight
  | 3 => BottomLeft
  | _ => TopLeft   -- This case is impossible since n % 4 is in {0, 1, 2, 3}

-- Function to calculate position of the mouse
def mouse_position (n : ℕ) : Position :=
  match n % 8 with
  | 0 => TopMiddle
  | 1 => TopRight
  | 2 => RightMiddle
  | 3 => BottomRight
  | 4 => BottomMiddle
  | 5 => BottomLeft
  | 6 => LeftMiddle
  | 7 => TopLeft
  | _ => TopMiddle -- This case is impossible since n % 8 is in {0, 1, .., 7}

-- Target theorem
theorem cat_mouse_positions_after_247_moves :
  cat_position 247 = BottomRight ∧ mouse_position 247 = LeftMiddle :=
by
  sorry

end cat_mouse_positions_after_247_moves_l2169_216982


namespace N_eq_P_l2169_216907

def N : Set ℝ := {x | ∃ n : ℤ, x = (n : ℝ) / 2 - 1 / 3}
def P : Set ℝ := {x | ∃ p : ℤ, x = (p : ℝ) / 2 + 1 / 6}

theorem N_eq_P : N = P :=
  sorry

end N_eq_P_l2169_216907


namespace maximum_mass_difference_l2169_216904

theorem maximum_mass_difference (m1 m2 : ℝ) (h1 : 19.7 ≤ m1 ∧ m1 ≤ 20.3) (h2 : 19.7 ≤ m2 ∧ m2 ≤ 20.3) :
  abs (m1 - m2) ≤ 0.6 :=
by
  sorry

end maximum_mass_difference_l2169_216904


namespace flowers_per_bug_l2169_216952

theorem flowers_per_bug (bugs : ℝ) (flowers : ℝ) (h_bugs : bugs = 2.0) (h_flowers : flowers = 3.0) :
  flowers / bugs = 1.5 :=
by
  sorry

end flowers_per_bug_l2169_216952


namespace economical_refuel_l2169_216994

theorem economical_refuel (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  (x + y) / 2 > (2 * x * y) / (x + y) :=
sorry -- Proof omitted

end economical_refuel_l2169_216994


namespace frustum_radius_l2169_216996

theorem frustum_radius (C1 C2 l: ℝ) (S_lateral: ℝ) (r: ℝ) :
  (C1 = 2 * r * π) ∧ (C2 = 6 * r * π) ∧ (l = 3) ∧ (S_lateral = 84 * π) → (r = 7) :=
by
  sorry

end frustum_radius_l2169_216996


namespace range_of_a_l2169_216942

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) :
  (0 < a ∧ 4 - 4 * a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l2169_216942


namespace evaluate_f_l2169_216950

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 4 * x

theorem evaluate_f (h : f 3 - f (-3) = 672) : True :=
by
  sorry

end evaluate_f_l2169_216950


namespace ratio_volumes_l2169_216936

theorem ratio_volumes (hA rA hB rB : ℝ) (hA_def : hA = 30) (rA_def : rA = 15) (hB_def : hB = rA) (rB_def : rB = 2 * hA) :
    (1 / 3 * Real.pi * rA^2 * hA) / (1 / 3 * Real.pi * rB^2 * hB) = 1 / 24 :=
by
  -- skipping the proof
  sorry

end ratio_volumes_l2169_216936


namespace combined_population_port_perry_lazy_harbor_l2169_216960

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l2169_216960


namespace parallel_vectors_solution_l2169_216972

noncomputable def vector_a : (ℝ × ℝ) := (1, 2)
noncomputable def vector_b (x : ℝ) : (ℝ × ℝ) := (x, -4)

def vectors_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_solution (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = -2 :=
sorry

end parallel_vectors_solution_l2169_216972


namespace find_function_p_t_additional_hours_l2169_216943

variable (p0 : ℝ) (t k : ℝ)

-- Given condition: initial concentration decreased by 1/5 after one hour
axiom filtration_condition_1 : (p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)))
axiom filtration_condition_2 : (p0 * ((4 : ℝ) / 5) = p0 * (Real.exp (-k)))

-- Problem 1: Find the function p(t)
theorem find_function_p_t : ∃ k, ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t)) := by
  sorry

-- Problem 2: Find the additional hours of filtration needed
theorem additional_hours (h : ∀ t, p0 * ((4 : ℝ) / 5) ^ t = p0 * (Real.exp (-k * t))) :
  ∀ t, p0 * ((4 : ℝ) / 5) ^ t ≤ (p0 / 1000) → t ≥ 30 := by
  sorry

end find_function_p_t_additional_hours_l2169_216943


namespace problem_statement_l2169_216980

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

def n : ℕ := sorry  -- n is the number of possible values of g(3)
def s : ℝ := sorry  -- s is the sum of all possible values of g(3)

theorem problem_statement : n * s = 0 := sorry

end problem_statement_l2169_216980


namespace g_six_l2169_216906

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eq (x y : ℝ) : g (x - y) = g x * g y
axiom g_nonzero (x : ℝ) : g x ≠ 0
axiom g_double (x : ℝ) : g (2 * x) = g x ^ 2
axiom g_value : g 6 = 1

theorem g_six : g 6 = 1 := by
  exact g_value

end g_six_l2169_216906


namespace derivatives_at_zero_l2169_216981

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem derivatives_at_zero :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv (deriv f) 0 = -4 ∧
  deriv (deriv (deriv f)) 0 = 0 ∧
  deriv (deriv (deriv (deriv f))) 0 = 16 :=
by
  sorry

end derivatives_at_zero_l2169_216981


namespace units_digit_of_1583_pow_1246_l2169_216961

theorem units_digit_of_1583_pow_1246 : 
  (1583^1246) % 10 = 9 := 
sorry

end units_digit_of_1583_pow_1246_l2169_216961


namespace Kyle_fish_count_l2169_216949

def Carla_fish := 8
def Total_fish := 36
def Kyle_fish := (Total_fish - Carla_fish) / 2

theorem Kyle_fish_count : Kyle_fish = 14 :=
by
  -- This proof will be provided later
  sorry

end Kyle_fish_count_l2169_216949


namespace rent_budget_l2169_216917

variables (food_per_week : ℝ) (weekly_food_budget : ℝ) (video_streaming : ℝ)
          (cell_phone : ℝ) (savings : ℝ) (rent : ℝ)
          (total_spending : ℝ)

-- Conditions
def food_budget := food_per_week * 4 = weekly_food_budget
def video_streaming_budget := video_streaming = 30
def cell_phone_budget := cell_phone = 50
def savings_budget := savings = 0.1 * total_spending
def savings_amount := savings = 198

-- Prove
theorem rent_budget (h1 : food_budget food_per_week weekly_food_budget)
                    (h2 : video_streaming_budget video_streaming)
                    (h3 : cell_phone_budget cell_phone)
                    (h4 : savings_budget savings total_spending)
                    (h5 : savings_amount savings) :
  rent = 1500 :=
sorry

end rent_budget_l2169_216917


namespace range_F_l2169_216937

-- Define the function and its critical points
def F (x : ℝ) : ℝ := |2 * x + 4| - |x - 2|

theorem range_F : ∀ y : ℝ, y ∈ Set.range F ↔ -4 ≤ y := by
  sorry

end range_F_l2169_216937


namespace pencils_remaining_l2169_216914

variable (initial_pencils : ℝ) (pencils_given : ℝ)

theorem pencils_remaining (h1 : initial_pencils = 56.0) 
                          (h2 : pencils_given = 9.5) 
                          : initial_pencils - pencils_given = 46.5 :=
by 
  sorry

end pencils_remaining_l2169_216914


namespace allocation_ways_l2169_216928

/-- Defining the number of different balls and boxes -/
def num_balls : ℕ := 4
def num_boxes : ℕ := 3

/-- The theorem asserting the number of ways to place the balls into the boxes -/
theorem allocation_ways : (num_boxes ^ num_balls) = 81 := by
  sorry

end allocation_ways_l2169_216928


namespace albert_wins_strategy_l2169_216983

theorem albert_wins_strategy (n : ℕ) (h : n = 1999) : 
  ∃ strategy : (ℕ → ℕ), (∀ tokens : ℕ, tokens = n → tokens > 1 → 
  (∃ next_tokens : ℕ, next_tokens < tokens ∧ next_tokens ≥ 1 ∧ next_tokens ≥ tokens / 2) → 
  (∃ k, tokens = 2^k + 1) → strategy n = true) :=
sorry

end albert_wins_strategy_l2169_216983


namespace find_divisor_of_115_l2169_216976

theorem find_divisor_of_115 (x : ℤ) (N : ℤ)
  (hN : N = 115)
  (h1 : N % 38 = 1)
  (h2 : N % x = 1) :
  x = 57 :=
by
  sorry

end find_divisor_of_115_l2169_216976


namespace aston_found_pages_l2169_216959

-- Given conditions
def pages_per_comic := 25
def initial_untorn_comics := 5
def total_comics_now := 11

-- The number of pages Aston found on the floor
theorem aston_found_pages :
  (total_comics_now - initial_untorn_comics) * pages_per_comic = 150 := 
by
  sorry

end aston_found_pages_l2169_216959


namespace geometric_sequence_ratio_l2169_216947
-- Lean 4 Code

noncomputable def a_n (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q ^ n

theorem geometric_sequence_ratio
  (a : ℝ)
  (q : ℝ)
  (h_pos : a > 0)
  (h_q_neq_1 : q ≠ 1)
  (h_arith_seq : 2 * a_n a q 4 = a_n a q 2 + a_n a q 5)
  : (a_n a q 2 + a_n a q 3) / (a_n a q 3 + a_n a q 4) = (Real.sqrt 5 - 1) / 2 :=
by {
  sorry
}

end geometric_sequence_ratio_l2169_216947


namespace units_digit_expression_l2169_216951

lemma units_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 := sorry

lemma units_digit_5_pow_2024 : (5 ^ 2024) % 10 = 5 := sorry

lemma units_digit_11_pow_2025 : (11 ^ 2025) % 10 = 1 := sorry

theorem units_digit_expression : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by 
  have h1 := units_digit_2_pow_2023
  have h2 := units_digit_5_pow_2024
  have h3 := units_digit_11_pow_2025
  sorry

end units_digit_expression_l2169_216951


namespace min_value_expression_l2169_216923

theorem min_value_expression : ∃ x y : ℝ, (xy-2)^2 + (x^2 + y^2) = 4 :=
by
  sorry

end min_value_expression_l2169_216923


namespace exponent_division_l2169_216908

theorem exponent_division :
  (1000 ^ 7) / (10 ^ 17) = 10 ^ 4 := 
  sorry

end exponent_division_l2169_216908


namespace power_of_fraction_l2169_216992

theorem power_of_fraction :
  (3 / 4) ^ 5 = 243 / 1024 :=
by sorry

end power_of_fraction_l2169_216992


namespace proof_time_lent_to_C_l2169_216939

theorem proof_time_lent_to_C :
  let P_B := 5000
  let R := 0.1
  let T_B := 2
  let Total_Interest := 2200
  let P_C := 3000
  let I_B := P_B * R * T_B
  let I_C := Total_Interest - I_B
  let T_C := I_C / (P_C * R)
  T_C = 4 :=
by
  sorry

end proof_time_lent_to_C_l2169_216939


namespace john_plays_periods_l2169_216920

theorem john_plays_periods
  (PointsPer4Minutes : ℕ := 7)
  (PeriodDurationMinutes : ℕ := 12)
  (TotalPoints : ℕ := 42) :
  (TotalPoints / PointsPer4Minutes) / (PeriodDurationMinutes / 4) = 2 := by
  sorry

end john_plays_periods_l2169_216920


namespace not_power_of_two_l2169_216964

theorem not_power_of_two (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  ¬ ∃ k : ℕ, (36 * m + n) * (m + 36 * n) = 2 ^ k :=
sorry

end not_power_of_two_l2169_216964


namespace books_written_l2169_216932

variable (Z F : ℕ)

theorem books_written (h1 : Z = 60) (h2 : Z = 4 * F) : Z + F = 75 := by
  sorry

end books_written_l2169_216932


namespace repeating_decimals_product_fraction_l2169_216989

theorem repeating_decimals_product_fraction : 
  let x := 1 / 33
  let y := 9 / 11
  x * y = 9 / 363 := 
by
  sorry

end repeating_decimals_product_fraction_l2169_216989


namespace orthocentric_tetrahedron_edge_tangent_iff_l2169_216958

structure Tetrahedron :=
(V : Type*)
(a b c d e f : V)
(is_orthocentric : Prop)
(has_edge_tangent_sphere : Prop)
(face_equilateral : Prop)
(edges_converging_equal : Prop)

variable (T : Tetrahedron)

noncomputable def edge_tangent_iff_equilateral_edges_converging_equal : Prop :=
T.has_edge_tangent_sphere ↔ (T.face_equilateral ∧ T.edges_converging_equal)

-- Now create the theorem statement
theorem orthocentric_tetrahedron_edge_tangent_iff :
  T.is_orthocentric →
  (∀ a d b e c f p r : ℝ, 
    a + d = b + e ∧ b + e = c + f ∧ a^2 + d^2 = b^2 + e^2 ∧ b^2 + e^2 = c^2 + f^2 ) → 
    edge_tangent_iff_equilateral_edges_converging_equal T := 
by
  intros
  unfold edge_tangent_iff_equilateral_edges_converging_equal
  sorry

end orthocentric_tetrahedron_edge_tangent_iff_l2169_216958


namespace correct_multiple_l2169_216991

theorem correct_multiple (n : ℝ) (m : ℝ) (h1 : n = 6) (h2 : m * n - 6 = 2 * n) : m * n = 18 :=
by
  sorry

end correct_multiple_l2169_216991


namespace product_of_binomials_l2169_216965

theorem product_of_binomials :
  (2*x^2 + 3*x - 4) * (x + 6) = 2*x^3 + 15*x^2 + 14*x - 24 :=
by {
  sorry
}

end product_of_binomials_l2169_216965


namespace minimum_time_to_finish_route_l2169_216926

-- Step (a): Defining conditions and necessary terms
def points : Nat := 12
def segments_between_points : ℕ := 17
def time_per_segment : ℕ := 10 -- in minutes
def total_time_in_minutes : ℕ := segments_between_points * time_per_segment -- Total time in minutes

-- Step (c): Proving the question == answer given conditions
theorem minimum_time_to_finish_route (K : ℕ) : K = 4 :=
by
  have time_in_hours : ℕ := total_time_in_minutes / 60
  have minimum_time : ℕ := 4
  sorry -- proof needed

end minimum_time_to_finish_route_l2169_216926


namespace friend_spent_l2169_216901

theorem friend_spent (x you friend total: ℝ) (h1 : total = you + friend) (h2 : friend = you + 3) (h3 : total = 11) : friend = 7 := by
  sorry

end friend_spent_l2169_216901


namespace store_shelves_l2169_216927

theorem store_shelves (initial_books sold_books books_per_shelf : ℕ) 
    (h_initial: initial_books = 27)
    (h_sold: sold_books = 6)
    (h_per_shelf: books_per_shelf = 7) :
    (initial_books - sold_books) / books_per_shelf = 3 := by
  sorry

end store_shelves_l2169_216927


namespace prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l2169_216918

noncomputable def probability_A_exactly_2_hits :=
  let p_A := 1/2
  let trials := 3
  (trials.choose 2) * (p_A ^ 2) * ((1 - p_A) ^ (trials - 2))

noncomputable def probability_B_at_least_2_hits :=
  let p_B := 2/3
  let trials := 3
  (trials.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ (trials - 2)) + (trials.choose 3) * (p_B ^ 3)

noncomputable def probability_B_exactly_2_more_hits_A :=
  let p_A := 1/2
  let p_B := 2/3
  let trials := 3
  let B_2_A_0 := (trials.choose 2) * (p_B ^ 2) * (1 - p_B) * (trials.choose 0) * (p_A ^ 0) * ((1 - p_A) ^ trials)
  let B_3_A_1 := (trials.choose 3) * (p_B ^ 3) * (trials.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (trials - 1))
  B_2_A_0 + B_3_A_1

theorem prove_A_exactly_2_hits : probability_A_exactly_2_hits = 3/8 := sorry
theorem prove_B_at_least_2_hits : probability_B_at_least_2_hits = 20/27 := sorry
theorem prove_B_exactly_2_more_hits_A : probability_B_exactly_2_more_hits_A = 1/6 := sorry

end prove_A_exactly_2_hits_prove_B_at_least_2_hits_prove_B_exactly_2_more_hits_A_l2169_216918


namespace Tod_speed_is_25_mph_l2169_216973

-- Definitions of the conditions
def miles_north : ℕ := 55
def miles_west : ℕ := 95
def hours_driven : ℕ := 6

-- The total distance travelled
def total_distance : ℕ := miles_north + miles_west

-- The speed calculation, dividing total distance by hours driven
def speed : ℕ := total_distance / hours_driven

-- The theorem to prove
theorem Tod_speed_is_25_mph : speed = 25 :=
by
  -- Proof of the theorem will be filled here, but for now using sorry
  sorry

end Tod_speed_is_25_mph_l2169_216973


namespace find_a_not_perfect_square_l2169_216935

theorem find_a_not_perfect_square :
  {a : ℕ | ∀ n : ℕ, n > 0 → ¬(∃ k : ℕ, n * (n + a) = k * k)} = {1, 2, 4} :=
sorry

end find_a_not_perfect_square_l2169_216935


namespace sample_size_l2169_216999

-- Definitions for the conditions
def ratio_A : Nat := 2
def ratio_B : Nat := 3
def ratio_C : Nat := 4
def stratified_sample_size : Nat := 9 -- Total parts in the ratio sum
def products_A_sample : Nat := 18 -- Sample contains 18 Type A products

-- We need to tie these conditions together and prove the size of the sample n
theorem sample_size (n : Nat) (ratio_A ratio_B ratio_C stratified_sample_size products_A_sample : Nat) :
  ratio_A = 2 → ratio_B = 3 → ratio_C = 4 → stratified_sample_size = 9 → products_A_sample = 18 → n = 81 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof body here
  sorry -- Placeholder for the proof

end sample_size_l2169_216999


namespace sum_coefficients_l2169_216910

theorem sum_coefficients (a : ℤ) (f : ℤ → ℤ) :
  f x = (1 - 2 * x)^7 ∧ a_0 = f 0 ∧ a_1_plus_a_7 = f 1 - f 0 
→ a_1_plus_a_7 = -2 :=
by sorry

end sum_coefficients_l2169_216910


namespace drawings_on_last_page_l2169_216915

theorem drawings_on_last_page :
  let n_notebooks := 10 
  let p_pages := 50
  let d_original := 5
  let d_new := 8
  let total_drawings := n_notebooks * p_pages * d_original
  let total_pages_new := total_drawings / d_new
  let filled_complete_pages := 6 * p_pages
  let drawings_on_last_page := total_drawings - filled_complete_pages * d_new - 40 * d_new
  drawings_on_last_page == 4 :=
  sorry

end drawings_on_last_page_l2169_216915


namespace find_A_l2169_216924

def is_valid_A (A : ℕ) : Prop :=
  A = 1 ∨ A = 2 ∨ A = 4 ∨ A = 7 ∨ A = 9

def number (A : ℕ) : ℕ :=
  3 * 100000 + 0 * 10000 + 5 * 1000 + 2 * 100 + 0 * 10 + A

theorem find_A (A : ℕ) (h_valid_A : is_valid_A A) : A = 1 ↔ Nat.Prime (number A) :=
by
  sorry

end find_A_l2169_216924


namespace am_gm_inequality_l2169_216929

theorem am_gm_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_inequality_l2169_216929


namespace CD_expression_l2169_216905

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C A1 B1 C1 D : V)
variables (a b c : V)

-- Given conditions
axiom AB_eq_a : A - B = a
axiom AC_eq_b : A - C = b
axiom AA1_eq_c : A - A1 = c
axiom midpoint_D : D = (1/2) • (B1 + C1)

-- We need to show
theorem CD_expression : C - D = (1/2) • a - (1/2) • b + c :=
sorry

end CD_expression_l2169_216905


namespace inequality_system_solution_l2169_216912

theorem inequality_system_solution (x : ℝ) :
  (x + 7) / 3 ≤ x + 3 ∧ 2 * (x + 1) < x + 3 ↔ -1 ≤ x ∧ x < 1 :=
by
  sorry

end inequality_system_solution_l2169_216912


namespace space_needed_between_apple_trees_l2169_216970

-- Definitions based on conditions
def apple_tree_width : ℕ := 10
def peach_tree_width : ℕ := 12
def space_between_peach_trees : ℕ := 15
def total_space : ℕ := 71
def number_of_apple_trees : ℕ := 2
def number_of_peach_trees : ℕ := 2

-- Lean 4 theorem statement
theorem space_needed_between_apple_trees :
  (total_space 
   - (number_of_peach_trees * peach_tree_width + space_between_peach_trees))
  - (number_of_apple_trees * apple_tree_width) 
  = 12 := by
  sorry

end space_needed_between_apple_trees_l2169_216970


namespace rowing_distance_l2169_216995

noncomputable def effective_speed_with_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed + current_speed

noncomputable def effective_speed_against_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed - current_speed

noncomputable def distance (speed time : ℕ) : ℕ :=
  speed * time

theorem rowing_distance (rowing_speed current_speed total_time : ℕ) 
  (hrowing_speed : rowing_speed = 10)
  (hcurrent_speed : current_speed = 2)
  (htotal_time : total_time = 30) : 
  (distance 8 18) = 144 := 
by
  sorry

end rowing_distance_l2169_216995


namespace max_S_value_max_S_value_achievable_l2169_216990

theorem max_S_value (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) ≤ 8 / 27 :=
sorry

theorem max_S_value_achievable :
  ∃ (x y z w : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) = 8 / 27 :=
sorry

end max_S_value_max_S_value_achievable_l2169_216990


namespace roots_magnitudes_less_than_one_l2169_216941

theorem roots_magnitudes_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + A * r + B = 0))
  (h2 : ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + C * r + D = 0)) :
  ∀ (r : ℝ), |r| < 1 → r ≠ 0 → (r ^ 2 + (1 / 2 * (A + C)) * r + (1 / 2 * (B + D)) = 0) :=
by
  sorry

end roots_magnitudes_less_than_one_l2169_216941


namespace units_digit_expression_l2169_216944

theorem units_digit_expression :
  ((2 * 21 * 2019 + 2^5) - 4^3) % 10 = 6 := 
sorry

end units_digit_expression_l2169_216944


namespace minimum_tetrahedra_partition_l2169_216931

-- Definitions for the problem conditions
def cube_faces : ℕ := 6
def tetrahedron_faces : ℕ := 4

def face_constraint (cube_faces : ℕ) (tetrahedral_faces : ℕ) : Prop :=
  cube_faces * 2 = 12

def volume_constraint (cube_volume : ℝ) (tetrahedron_volume : ℝ) : Prop :=
  tetrahedron_volume < cube_volume / 6

-- Main proof statement
theorem minimum_tetrahedra_partition (cube_faces tetrahedron_faces : ℕ) (cube_volume tetrahedron_volume : ℝ) :
  face_constraint cube_faces tetrahedron_faces →
  volume_constraint cube_volume tetrahedron_volume →
  5 ≤ cube_faces * 2 / 3 :=
  sorry

end minimum_tetrahedra_partition_l2169_216931


namespace rower_trip_time_to_Big_Rock_l2169_216975

noncomputable def row_trip_time (rowing_speed_in_still_water : ℝ) (river_speed : ℝ) (distance_to_destination : ℝ) : ℝ :=
  let speed_upstream := rowing_speed_in_still_water - river_speed
  let speed_downstream := rowing_speed_in_still_water + river_speed
  let time_upstream := distance_to_destination / speed_upstream
  let time_downstream := distance_to_destination / speed_downstream
  time_upstream + time_downstream

theorem rower_trip_time_to_Big_Rock :
  row_trip_time 7 2 3.2142857142857144 = 1 :=
by
  sorry

end rower_trip_time_to_Big_Rock_l2169_216975


namespace cylinder_water_depth_l2169_216985

theorem cylinder_water_depth 
  (height radius : ℝ)
  (h_ge_zero : height ≥ 0)
  (r_ge_zero : radius ≥ 0)
  (total_height : height = 1200)
  (total_radius : radius = 100)
  (above_water_vol : 1 / 3 * π * radius^2 * height = 1 / 3 * π * radius^2 * 1200) :
  height - 800 = 400 :=
by
  -- Use provided constraints and logical reasoning on structures
  sorry

end cylinder_water_depth_l2169_216985


namespace total_candidates_l2169_216978

def average_marks_all_candidates : ℕ := 35
def average_marks_passed_candidates : ℕ := 39
def average_marks_failed_candidates : ℕ := 15
def passed_candidates : ℕ := 100

theorem total_candidates (T : ℕ) (F : ℕ) 
  (h1 : 35 * T = 39 * passed_candidates + 15 * F)
  (h2 : T = passed_candidates + F) : T = 120 := 
  sorry

end total_candidates_l2169_216978


namespace second_quadrant_set_l2169_216987

-- Define the set P of points in the second quadrant
def P : Set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 }

-- Statement of the problem: Prove that this definition accurately describes the set of all points in the second quadrant
theorem second_quadrant_set :
  P = { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } :=
by
  sorry

end second_quadrant_set_l2169_216987


namespace sodas_total_l2169_216971

def morning_sodas : ℕ := 77
def afternoon_sodas : ℕ := 19
def total_sodas : ℕ := morning_sodas + afternoon_sodas

theorem sodas_total :
  total_sodas = 96 :=
by
  sorry

end sodas_total_l2169_216971


namespace smallest_unrepresentable_integer_l2169_216962

theorem smallest_unrepresentable_integer :
  ∃ n : ℕ, (∀ a b c d : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → 
  n ≠ (2^a - 2^b) / (2^c - 2^d)) ∧ n = 11 :=
by
  sorry

end smallest_unrepresentable_integer_l2169_216962


namespace subset_interval_l2169_216988

theorem subset_interval (a : ℝ) : 
  (∀ x : ℝ, (-a-1 < x ∧ x < -a+1 → -3 < x ∧ x < 1)) ↔ (0 ≤ a ∧ a ≤ 2) := 
by
  sorry

end subset_interval_l2169_216988


namespace find_largest_n_l2169_216902

theorem find_largest_n 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (x y : ℕ)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_arith_a : ∀ n : ℕ, a n = 1 + (n - 1) * x)
  (h_arith_b : ∀ n : ℕ, b n = 1 + (n - 1) * y)
  (h_order : x ≤ y)
  (h_product : ∃ n : ℕ, a n * b n = 4021) :
  ∃ n : ℕ, a n * b n = 4021 ∧ n ≤ 11 := 
by
  sorry

end find_largest_n_l2169_216902


namespace gcd_pow_minus_one_l2169_216955

theorem gcd_pow_minus_one (n m : ℕ) (hn : n = 1030) (hm : m = 1040) :
  Nat.gcd (2^n - 1) (2^m - 1) = 1023 := 
by
  sorry

end gcd_pow_minus_one_l2169_216955


namespace find_m_l2169_216933

theorem find_m (S : ℕ → ℝ) (m : ℕ) (h1 : S m = -2) (h2 : S (m+1) = 0) (h3 : S (m+2) = 3) : m = 4 :=
by
  sorry

end find_m_l2169_216933


namespace first_chapter_length_l2169_216922

theorem first_chapter_length (total_pages : ℕ) (second_chapter_pages : ℕ) (third_chapter_pages : ℕ)
  (h : total_pages = 125) (h2 : second_chapter_pages = 35) (h3 : third_chapter_pages  = 24) :
  total_pages - second_chapter_pages - third_chapter_pages = 66 :=
by
  -- Construct the proof using the provided conditions
  sorry

end first_chapter_length_l2169_216922


namespace total_distance_between_alice_bob_l2169_216957

-- Define the constants for Alice's and Bob's speeds and the time duration in terms of conditions.
def alice_speed := 1 / 12  -- miles per minute
def bob_speed := 3 / 20    -- miles per minute
def time_duration := 120   -- minutes

-- Statement: Prove that the total distance between Alice and Bob after 2 hours is 28 miles.
theorem total_distance_between_alice_bob : (alice_speed * time_duration) + (bob_speed * time_duration) = 28 :=
by
  sorry

end total_distance_between_alice_bob_l2169_216957


namespace umar_age_is_10_l2169_216903

-- Define Ali's age
def Ali_age := 8

-- Define the age difference between Ali and Yusaf
def age_difference := 3

-- Define Yusaf's age based on the conditions
def Yusaf_age := Ali_age - age_difference

-- Define Umar's age which is twice Yusaf's age
def Umar_age := 2 * Yusaf_age

-- Prove that Umar's age is 10
theorem umar_age_is_10 : Umar_age = 10 :=
by
  -- Proof is skipped
  sorry

end umar_age_is_10_l2169_216903


namespace find_first_number_l2169_216938

theorem find_first_number (x : ℕ) (h1 : x + 35 = 62) : x = 27 := by
  sorry

end find_first_number_l2169_216938


namespace attendance_difference_is_85_l2169_216954

def saturday_attendance : ℕ := 80
def monday_attendance : ℕ := saturday_attendance - 20
def wednesday_attendance : ℕ := monday_attendance + 50
def friday_attendance : ℕ := saturday_attendance + monday_attendance
def thursday_attendance : ℕ := 45
def expected_audience : ℕ := 350

def total_attendance : ℕ := 
  saturday_attendance + 
  monday_attendance + 
  wednesday_attendance + 
  friday_attendance + 
  thursday_attendance

def more_people_attended_than_expected : ℕ :=
  total_attendance - expected_audience

theorem attendance_difference_is_85 : more_people_attended_than_expected = 85 := 
by
  unfold more_people_attended_than_expected
  unfold total_attendance
  unfold saturday_attendance
  unfold monday_attendance
  unfold wednesday_attendance
  unfold friday_attendance
  unfold thursday_attendance
  unfold expected_audience
  exact sorry

end attendance_difference_is_85_l2169_216954


namespace unique_not_in_range_of_g_l2169_216945

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem unique_not_in_range_of_g (p q r s : ℝ) (hps_qr_zero : p * s + q * r = 0) 
  (hpr_rs_zero : p * r + r * s = 0) (hg3 : g p q r s 3 = 3) 
  (hg81 : g p q r s 81 = 81) (h_involution : ∀ x ≠ (-s / r), g p q r s (g p q r s x) = x) :
  ∀ x : ℝ, x ≠ 42 :=
sorry

end unique_not_in_range_of_g_l2169_216945


namespace incorrect_option_A_l2169_216921

theorem incorrect_option_A (x y : ℝ) :
  ¬(5 * x + y / 2 = (5 * x + y) / 2) :=
by sorry

end incorrect_option_A_l2169_216921


namespace area_transformed_region_l2169_216913

-- Define the transformation matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 1], ![4, 3]]

-- Define the area of region T
def area_T := 6

-- The statement we want to prove: the area of T' is 30.
theorem area_transformed_region :
  let det := matrix.det
  area_T * det = 30 :=
by
  sorry

end area_transformed_region_l2169_216913


namespace quadratic_same_roots_abs_l2169_216977

theorem quadratic_same_roots_abs (d e : ℤ) : 
  (∀ x : ℤ, |x - 8| = 3 ↔ x = 11 ∨ x = 5) →
  (∀ x : ℤ, x^2 + d * x + e = 0 ↔ x = 11 ∨ x = 5) →
  (d, e) = (-16, 55) :=
by
  intro h₁ h₂
  have h₃ : ∀ x : ℤ, x^2 - 16 * x + 55 = 0 ↔ x = 11 ∨ x = 5 := sorry
  sorry

end quadratic_same_roots_abs_l2169_216977


namespace pencils_per_row_l2169_216997

-- Define the conditions as parameters
variables (total_pencils : Int) (rows : Int) 

-- State the proof problem using the conditions and the correct answer
theorem pencils_per_row (h₁ : total_pencils = 12) (h₂ : rows = 3) : total_pencils / rows = 4 := 
by 
  sorry

end pencils_per_row_l2169_216997


namespace cubicsum_eq_neg36_l2169_216940

noncomputable def roots (p q r : ℝ) := 
  ∃ l : ℝ, (p^3 - 12) / p = l ∧ (q^3 - 12) / q = l ∧ (r^3 - 12) / r = l

theorem cubicsum_eq_neg36 {p q r : ℝ} (h : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hl : roots p q r) :
  p^3 + q^3 + r^3 = -36 :=
sorry

end cubicsum_eq_neg36_l2169_216940


namespace find_height_of_box_l2169_216984

-- Given the conditions
variables (h l w : ℝ)
variables (V : ℝ)

-- Conditions as definitions in Lean
def length_eq_height (h : ℝ) : ℝ := 3 * h
def length_eq_width (w : ℝ) : ℝ := 4 * w
def volume_eq (h l w : ℝ) : ℝ := l * w * h

-- The proof problem: Prove height of the box is 12 given the conditions
theorem find_height_of_box : 
  (∃ h l w, l = 3 * h ∧ l = 4 * w ∧ l * w * h = 3888) → h = 12 :=
by
  sorry

end find_height_of_box_l2169_216984


namespace remaining_speed_l2169_216979

theorem remaining_speed (D : ℝ) (V : ℝ) 
  (h1 : 0.35 * D / 35 + 0.65 * D / V = D / 50) : V = 32.5 :=
by sorry

end remaining_speed_l2169_216979


namespace orange_pear_difference_l2169_216919

theorem orange_pear_difference :
  let O1 := 37
  let O2 := 10
  let O3 := 2 * O2
  let P1 := 30
  let P2 := 3 * P1
  let P3 := P2 + 4
  (O1 + O2 + O3 - (P1 + P2 + P3)) = -147 := 
by
  sorry

end orange_pear_difference_l2169_216919


namespace tan_y_l2169_216948

theorem tan_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hy : 0 < y ∧ y < π / 2)
  (hsiny : Real.sin y = 2 * a * b / (a^2 + b^2)) :
  Real.tan y = 2 * a * b / (a^2 - b^2) :=
sorry

end tan_y_l2169_216948


namespace rectangle_area_l2169_216967

theorem rectangle_area (l w : ℝ) (h₁ : (2 * l + 2 * w) = 46) (h₂ : (l^2 + w^2) = 289) : l * w = 120 :=
by
  sorry

end rectangle_area_l2169_216967


namespace small_to_large_circle_ratio_l2169_216993

theorem small_to_large_circle_ratio (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 5 * π * a^2) :
  a / b = 1 / Real.sqrt 6 :=
by
  sorry

end small_to_large_circle_ratio_l2169_216993


namespace figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l2169_216925

-- Given conditions:
def initial_squares : ℕ := 3
def initial_perimeter : ℕ := 8
def squares_per_step : ℕ := 2
def perimeter_per_step : ℕ := 4

-- Statement proving Figure 8 has 17 squares
theorem figure8_squares : 3 + 2 * (8 - 1) = 17 := by sorry

-- Statement proving Figure 12 has a perimeter of 52 cm
theorem figure12_perimeter : 8 + 4 * (12 - 1) = 52 := by sorry

-- Statement proving no positive integer C yields perimeter of 38 cm
theorem no_figure_C : ¬∃ C : ℕ, 8 + 4 * (C - 1) = 38 := by sorry
  
-- Statement proving closest D giving the ratio for perimeter between Figure 29 and Figure D
theorem figure29_figureD_ratio : (8 + 4 * (29 - 1)) * 11 = 4 * (8 + 4 * (81 - 1)) := by sorry

end figure8_squares_figure12_perimeter_no_figure_C_figure29_figureD_ratio_l2169_216925


namespace trapezoid_area_eq_15_l2169_216974

theorem trapezoid_area_eq_15 :
  let line1 := fun (x : ℝ) => 2 * x
  let line2 := fun (x : ℝ) => 8
  let line3 := fun (x : ℝ) => 2
  let y_axis := fun (y : ℝ) => 0
  let intersection_points := [
    (4, 8),   -- Intersection of line1 and line2
    (1, 2),   -- Intersection of line1 and line3
    (0, 8),   -- Intersection of y_axis and line2
    (0, 2)    -- Intersection of y_axis and line3
  ]
  let base1 := (4 - 0 : ℝ)  -- Length of top base 
  let base2 := (1 - 0 : ℝ)  -- Length of bottom base
  let height := (8 - 2 : ℝ) -- Vertical distance between line2 and line3
  (0.5 * (base1 + base2) * height = 15.0) := by
  sorry

end trapezoid_area_eq_15_l2169_216974


namespace expressionEquals243_l2169_216911

noncomputable def calculateExpression : ℕ :=
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 *
  (1 / 19683) * 59049

theorem expressionEquals243 : calculateExpression = 243 := by
  sorry

end expressionEquals243_l2169_216911


namespace compute_pow_l2169_216934

theorem compute_pow (i : ℂ) (h : i^2 = -1) : (1 - i)^6 = 8 * i := by
  sorry

end compute_pow_l2169_216934


namespace find_pictures_museum_l2169_216968

-- Define the given conditions
def pictures_zoo : Nat := 24
def pictures_deleted : Nat := 14
def pictures_remaining : Nat := 22

-- Define the target: the number of pictures taken at the museum
def pictures_museum : Nat := 12

-- State the goal to be proved
theorem find_pictures_museum :
  pictures_zoo + pictures_museum - pictures_deleted = pictures_remaining :=
sorry

end find_pictures_museum_l2169_216968


namespace number_of_boys_in_school_l2169_216998

theorem number_of_boys_in_school (total_students : ℕ) (sample_size : ℕ) 
(number_diff : ℕ) (ratio_boys_sample_girls_sample : ℚ) : 
total_students = 1200 → sample_size = 200 → number_diff = 10 →
ratio_boys_sample_girls_sample = 105 / 95 →
∃ (boys_in_school : ℕ), boys_in_school = 630 := by 
  sorry

end number_of_boys_in_school_l2169_216998
