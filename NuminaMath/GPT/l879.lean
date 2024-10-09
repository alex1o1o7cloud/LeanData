import Mathlib

namespace vec_same_direction_l879_87979

theorem vec_same_direction (k : ℝ) : (k = 2) ↔ ∃ m : ℝ, m > 0 ∧ (k, 2) = (m * 1, m * 1) :=
by
  sorry

end vec_same_direction_l879_87979


namespace solve_quadratic_solve_inequality_system_l879_87924

theorem solve_quadratic :
  ∀ x : ℝ, x^2 - 6 * x + 5 = 0 ↔ x = 1 ∨ x = 5 :=
sorry

theorem solve_inequality_system :
  ∀ x : ℝ, (x + 3 > 0 ∧ 2 * (x + 1) < 4) ↔ (-3 < x ∧ x < 1) :=
sorry

end solve_quadratic_solve_inequality_system_l879_87924


namespace line_parabola_midpoint_l879_87917

theorem line_parabola_midpoint (a b : ℝ) 
  (r s : ℝ) 
  (intersects_parabola : ∀ x, x = r ∨ x = s → ax + b = x^2)
  (midpoint_cond : (r + s) / 2 = 5 ∧ (r^2 + s^2) / 2 = 101) :
  a + b = -41 :=
sorry

end line_parabola_midpoint_l879_87917


namespace average_speed_of_bus_trip_l879_87944

theorem average_speed_of_bus_trip
  (v : ℝ)
  (distance : ℝ)
  (time_difference : ℝ)
  (speed_increment : ℝ)
  (original_time : ℝ := distance / v)
  (faster_time : ℝ := distance / (v + speed_increment))
  (h1 : distance = 360)
  (h2 : time_difference = 1)
  (h3 : speed_increment = 5)
  (h4 : original_time - time_difference = faster_time) :
  v = 40 :=
by
  sorry

end average_speed_of_bus_trip_l879_87944


namespace present_price_after_discount_l879_87965

theorem present_price_after_discount :
  ∀ (P : ℝ), (∀ x : ℝ, (3 * x = P - 0.20 * P) ∧ (x = (P / 3) - 4)) → P = 60 → 0.80 * P = 48 :=
by
  intros P hP h60
  sorry

end present_price_after_discount_l879_87965


namespace compare_x_y_l879_87942

variable (a b : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (a_ne_b : a ≠ b)

noncomputable def x : ℝ := (Real.sqrt a + Real.sqrt b) / Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt (a + b)

theorem compare_x_y : y a b > x a b := sorry

end compare_x_y_l879_87942


namespace value_of_h_l879_87938

theorem value_of_h (h : ℝ) : (∃ x : ℝ, x^3 + h * x - 14 = 0 ∧ x = 3) → h = -13/3 :=
by
  sorry

end value_of_h_l879_87938


namespace g_zero_l879_87961

variable (f g h : Polynomial ℤ) -- Assume f, g, h are polynomials over the integers

-- Condition: h(x) = f(x) * g(x)
axiom h_def : h = f * g

-- Condition: The constant term of f(x) is 2
axiom f_const : f.coeff 0 = 2

-- Condition: The constant term of h(x) is -6
axiom h_const : h.coeff 0 = -6

-- Proof statement that g(0) = -3
theorem g_zero : g.coeff 0 = -3 := by
  sorry

end g_zero_l879_87961


namespace intersection_N_complement_M_l879_87997

def U : Set ℝ := Set.univ
def M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}
def CU_M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | (1 - x) / (x - 3) > 0}

theorem intersection_N_complement_M :
  N ∩ CU_M = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end intersection_N_complement_M_l879_87997


namespace percentage_error_is_94_l879_87975

theorem percentage_error_is_94 (x : ℝ) (hx : 0 < x) :
  let correct_result := 4 * x
  let error_result := x / 4
  let error := |correct_result - error_result|
  let percentage_error := (error / correct_result) * 100
  percentage_error = 93.75 := by
    sorry

end percentage_error_is_94_l879_87975


namespace choosing_ways_president_vp_committee_l879_87929

theorem choosing_ways_president_vp_committee :
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  total_choices = 2520 := by
  let n := 10
  let president_choices := n
  let vp_choices := n - 1
  let committee_choices := (n - 2) * (n - 3) / 2
  let total_choices := president_choices * vp_choices * committee_choices
  have : total_choices = 2520 := by
    sorry
  exact this

end choosing_ways_president_vp_committee_l879_87929


namespace total_cost_eq_898_80_l879_87900

theorem total_cost_eq_898_80 (M R F : ℝ)
  (h1 : 10 * M = 24 * R)
  (h2 : 6 * F = 2 * R)
  (h3 : F = 21) :
  4 * M + 3 * R + 5 * F = 898.80 :=
by
  sorry

end total_cost_eq_898_80_l879_87900


namespace initial_participants_l879_87947

theorem initial_participants (p : ℕ) (h1 : 0.6 * p = 0.6 * (p : ℝ)) (h2 : ∀ (n : ℕ), n = 4 * m → 30 = (2 / 5) * n * (1 / 4)) :
  p = 300 :=
by sorry

end initial_participants_l879_87947


namespace sunzi_oranges_l879_87914

theorem sunzi_oranges :
  ∃ (a : ℕ), ( 5 * a + 10 * 3 = 60 ) ∧ ( ∀ n, n = 0 → a = 6 ) :=
by
  sorry

end sunzi_oranges_l879_87914


namespace cannot_lie_on_line_l879_87969

open Real

theorem cannot_lie_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  (0, -2023) ≠ (0, b) :=
by
  sorry

end cannot_lie_on_line_l879_87969


namespace train_pass_bridge_in_56_seconds_l879_87939

noncomputable def time_for_train_to_pass_bridge 
(length_of_train : ℕ) (speed_of_train_kmh : ℕ) (length_of_bridge : ℕ) : ℕ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  total_distance / speed_of_train_ms

theorem train_pass_bridge_in_56_seconds :
  time_for_train_to_pass_bridge 560 45 140 = 56 := by
  sorry

end train_pass_bridge_in_56_seconds_l879_87939


namespace a4_equals_8_l879_87954

variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {n : ℕ}

-- Defining the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a (n + 1) = a n * r

-- Given conditions as hypotheses
variable (h_geometric : geometric_sequence a r)
variable (h_root_2 : a 2 * a 6 = 64)
variable (h_roots_eq : ∀ x, x^2 - 34 * x + 64 = 0 → (x = a 2 ∨ x = a 6))

-- The statement to prove
theorem a4_equals_8 : a 4 = 8 :=
by
  sorry

end a4_equals_8_l879_87954


namespace solve_inequality_min_value_F_l879_87991

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x + 1)
def m := 3    -- Arbitrary constant, m + n = 7 implies n = 4
def n := 4

-- First statement: Solve the inequality f(x) ≥ (m + n)x
theorem solve_inequality (x : ℝ) : f x ≥ (m + n) * x ↔ x ≤ 0 := by
  sorry

noncomputable def F (x y : ℝ) : ℝ := max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n))

-- Second statement: Find the minimum value of F
theorem min_value_F (x y : ℝ) : (F x y) ≥ 1 ∧ (∃ x y, (F x y) = 1) := by
  sorry

end solve_inequality_min_value_F_l879_87991


namespace system_solution_l879_87906

noncomputable def x1 : ℝ := 55 / Real.sqrt 91
noncomputable def y1 : ℝ := 18 / Real.sqrt 91
noncomputable def x2 : ℝ := -55 / Real.sqrt 91
noncomputable def y2 : ℝ := -18 / Real.sqrt 91

theorem system_solution (x y : ℝ) (h1 : x^2 = 4 * y^2 + 19) (h2 : x * y + 2 * y^2 = 18) :
  (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) :=
sorry

end system_solution_l879_87906


namespace prob_blue_lower_than_yellow_l879_87943

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  3^(-k : ℤ)

noncomputable def prob_same_bin : ℝ :=
  ∑' k, 3^(-2*k : ℤ)

theorem prob_blue_lower_than_yellow :
  (1 - prob_same_bin) / 2 = 7 / 16 :=
by
  -- proof goes here
  sorry

end prob_blue_lower_than_yellow_l879_87943


namespace point_in_fourth_quadrant_l879_87951

theorem point_in_fourth_quadrant (θ : ℝ) (h : -1 < Real.cos θ ∧ Real.cos θ < 0) :
    ∃ (x y : ℝ), x = Real.sin (Real.cos θ) ∧ y = Real.cos (Real.cos θ) ∧ x < 0 ∧ y > 0 :=
by
  sorry

end point_in_fourth_quadrant_l879_87951


namespace mrs_hilt_money_l879_87923

-- Definitions and given conditions
def cost_of_pencil := 5  -- in cents
def number_of_pencils := 10

-- The theorem we need to prove
theorem mrs_hilt_money : cost_of_pencil * number_of_pencils = 50 := by
  sorry

end mrs_hilt_money_l879_87923


namespace problem_solution_l879_87957

variable (x y : ℝ)

-- Conditions
axiom h1 : x ≠ 0
axiom h2 : y ≠ 0
axiom h3 : (4 * x - 3 * y) / (x + 4 * y) = 3

-- Goal
theorem problem_solution : (x - 4 * y) / (4 * x + 3 * y) = 11 / 63 :=
by
  sorry

end problem_solution_l879_87957


namespace find_f_neg_one_l879_87915

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)

theorem find_f_neg_one (k : ℝ) (h : ∀ (x : ℝ), f k (-x) = -f k x) : f k (-1) = 2 :=
sorry

end find_f_neg_one_l879_87915


namespace binom_two_formula_l879_87998

def binom (n k : ℕ) : ℕ :=
  n.choose k

-- Formalizing the conditions
variable (n : ℕ)
variable (h : n ≥ 2)

-- Stating the problem mathematically in Lean
theorem binom_two_formula :
  binom n 2 = n * (n - 1) / 2 := by
  sorry

end binom_two_formula_l879_87998


namespace evaluate_expression_l879_87962

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l879_87962


namespace solve_equation_l879_87912

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end solve_equation_l879_87912


namespace inequality_has_real_solution_l879_87911

variable {f : ℝ → ℝ}

theorem inequality_has_real_solution (h : ∃ x : ℝ, f x > 0) : 
    (∃ x : ℝ, f x > 0) :=
by
  sorry

end inequality_has_real_solution_l879_87911


namespace eventually_one_student_answers_yes_l879_87989

-- Conditions and Definitions
variable (a b r₁ r₂ : ℕ)
variable (h₁ : r₁ ≠ r₂)   -- r₁ and r₂ are distinct
variable (h₂ : r₁ = a + b ∨ r₂ = a + b) -- One of r₁ or r₂ is the sum a + b
variable (h₃ : a > 0) -- a is a positive integer
variable (h₄ : b > 0) -- b is a positive integer

theorem eventually_one_student_answers_yes (a b r₁ r₂ : ℕ) (h₁ : r₁ ≠ r₂) (h₂ : r₁ = a + b ∨ r₂ = a + b) (h₃ : a > 0) (h₄ : b > 0) :
  ∃ n : ℕ, (∃ c : ℕ, (r₁ = c + b ∨ r₂ = c + b) ∧ (c = a ∨ c ≤ r₁ ∨ c ≤ r₂)) ∨ 
  (∃ c : ℕ, (r₁ = a + c ∨ r₂ = a + c) ∧ (c = b ∨ c ≤ r₁ ∨ c ≤ r₂)) :=
sorry

end eventually_one_student_answers_yes_l879_87989


namespace common_ratio_geometric_progression_l879_87928

theorem common_ratio_geometric_progression (r : ℝ) (a : ℝ) (h : a > 0) (h_r : r > 0) (h_eq : ∀ (n : ℕ), a * r^(n-1) = a * r^n + a * r^(n+1) + a * r^(n+2)) : r^3 + r^2 + r - 1 = 0 := 
by sorry

end common_ratio_geometric_progression_l879_87928


namespace parabola_point_ordinate_l879_87916

-- The definition of the problem as a Lean 4 statement
theorem parabola_point_ordinate (a : ℝ) (x₀ y₀ : ℝ) 
  (h₀ : 0 < a)
  (h₁ : x₀^2 = (1 / a) * y₀)
  (h₂ : dist (0, 1 / (4 * a)) (0, -1 / (4 * a)) = 1)
  (h₃ : dist (x₀, y₀) (0, 1 / (4 * a)) = 5) :
  y₀ = 9 / 2 := 
sorry

end parabola_point_ordinate_l879_87916


namespace printer_task_total_pages_l879_87981

theorem printer_task_total_pages
  (A B : ℕ)
  (h1 : 1 / A + 1 / B = 1 / 24)
  (h2 : 1 / A = 1 / 60)
  (h3 : B = A + 6) :
  60 * A = 720 := by
  sorry

end printer_task_total_pages_l879_87981


namespace nebraska_license_plate_increase_l879_87985

open Nat

theorem nebraska_license_plate_increase :
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  new_plates / old_plates = 260 :=
by
  -- Definitions based on conditions
  let old_plates : ℕ := 26 * 10^3
  let new_plates : ℕ := 26^2 * 10^4
  -- Assertion to prove
  show new_plates / old_plates = 260
  sorry

end nebraska_license_plate_increase_l879_87985


namespace deer_meat_distribution_l879_87926

theorem deer_meat_distribution (a d : ℕ) (H1 : a = 100) :
  ∀ (Dafu Bugeng Zanbao Shangzao Gongshe : ℕ),
    Dafu = a - 2 * d →
    Bugeng = a - d →
    Zanbao = a →
    Shangzao = a + d →
    Gongshe = a + 2 * d →
    Dafu + Bugeng + Zanbao + Shangzao + Gongshe = 500 →
    Bugeng + Zanbao + Shangzao = 300 :=
by
  intros Dafu Bugeng Zanbao Shangzao Gongshe hDafu hBugeng hZanbao hShangzao hGongshe hSum
  sorry

end deer_meat_distribution_l879_87926


namespace tree_height_increase_fraction_l879_87993

theorem tree_height_increase_fraction :
  ∀ (initial_height annual_increase : ℝ) (additional_years₄ additional_years₆ : ℕ),
    initial_height = 4 →
    annual_increase = 0.4 →
    additional_years₄ = 4 →
    additional_years₆ = 6 →
    ((initial_height + annual_increase * additional_years₆) - (initial_height + annual_increase * additional_years₄)) / (initial_height + annual_increase * additional_years₄) = 1 / 7 :=
by
  sorry

end tree_height_increase_fraction_l879_87993


namespace value_of_fraction_l879_87945

theorem value_of_fraction (x y : ℝ) (h : 1 / x - 1 / y = 2) : (x + x * y - y) / (x - x * y - y) = 1 / 3 :=
by
  sorry

end value_of_fraction_l879_87945


namespace factor_x_minus_1_l879_87949

theorem factor_x_minus_1 (P Q R S : Polynomial ℂ) : 
  (P.eval 1 = 0) → 
  (P.eval (x^5) + x * Q.eval (x^5) + x^2 * R.eval (x^5) 
  = (x^4 + x^3 + x^2 + x + 1) * S.eval (x)) :=
sorry

end factor_x_minus_1_l879_87949


namespace boxes_calculation_proof_l879_87952

variable (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_box : ℕ)
variable (total_eggs : ℕ := baskets * eggs_per_basket)
variable (boxes_needed : ℕ := total_eggs / eggs_per_box)

theorem boxes_calculation_proof :
  baskets = 21 →
  eggs_per_basket = 48 →
  eggs_per_box = 28 →
  boxes_needed = 36 :=
by
  intros
  sorry

end boxes_calculation_proof_l879_87952


namespace min_sum_of_squares_l879_87968

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3 * b + 5 * c + 7 * d = 14) : 
  a^2 + b^2 + c^2 + d^2 ≥ 7 / 3 :=
sorry

end min_sum_of_squares_l879_87968


namespace student_score_max_marks_l879_87930

theorem student_score_max_marks (M : ℝ)
  (pass_threshold : ℝ := 0.60 * M)
  (student_marks : ℝ := 80)
  (fail_by : ℝ := 40)
  (required_passing_score : ℝ := student_marks + fail_by) :
  pass_threshold = required_passing_score → M = 200 := 
by
  sorry

end student_score_max_marks_l879_87930


namespace cost_large_bulb_l879_87920

def small_bulbs : Nat := 3
def cost_small_bulb : Nat := 8
def total_budget : Nat := 60
def amount_left : Nat := 24

theorem cost_large_bulb (cost_large_bulb : Nat) :
  total_budget - amount_left - small_bulbs * cost_small_bulb = cost_large_bulb →
  cost_large_bulb = 12 := by
  sorry

end cost_large_bulb_l879_87920


namespace find_x_y_l879_87973

theorem find_x_y (x y : ℝ)
  (h1 : (x - 1) ^ 2003 + 2002 * (x - 1) = -1)
  (h2 : (y - 2) ^ 2003 + 2002 * (y - 2) = 1) :
  x + y = 3 :=
sorry

end find_x_y_l879_87973


namespace f_is_monotonic_l879_87971

variable (f : ℝ → ℝ)

theorem f_is_monotonic (h : ∀ a b x : ℝ, a < x ∧ x < b → min (f a) (f b) < f x ∧ f x < max (f a) (f b)) :
  (∀ x y : ℝ, x ≤ y → f x <= f y) ∨ (∀ x y : ℝ, x ≤ y → f x >= f y) :=
sorry

end f_is_monotonic_l879_87971


namespace probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l879_87996

def TianJi_top {α : Type} [LinearOrder α] (a1 a2 : α) (b1 : α) : Prop :=
  a2 < b1 ∧ b1 < a1

def TianJi_middle {α : Type} [LinearOrder α] (a3 a2 : α) (b2 : α) : Prop :=
  a3 < b2 ∧ b2 < a2

def TianJi_bottom {α : Type} [LinearOrder α] (a3 : α) (b3 : α) : Prop :=
  b3 < a3

def without_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning without knowing the order
  1 / 6

theorem probability_without_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  without_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 6 :=
sorry

def with_order_knowledge_probability (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : ℚ :=
  -- Formula for the probability of Tian Ji winning with specific group knowledge
  1 / 2

theorem probability_with_order_knowledge_correct (a1 a2 a3 b1 b2 b3 : ℕ) 
  (h_top : TianJi_top a1 a2 b1) 
  (h_middle : TianJi_middle a3 a2 b2) 
  (h_bottom : TianJi_bottom a3 b3) : 
  with_order_knowledge_probability a1 a2 a3 b1 b2 b3 h_top h_middle h_bottom = 1 / 2 :=
sorry

end probability_without_order_knowledge_correct_probability_with_order_knowledge_correct_l879_87996


namespace problem_statement_l879_87966

theorem problem_statement (x y : ℤ) (k : ℤ) (h : 4 * x - y = 3 * k) : 9 ∣ 4 * x^2 + 7 * x * y - 2 * y^2 :=
by
  sorry

end problem_statement_l879_87966


namespace gcd_2_pow_2018_2_pow_2029_l879_87946

theorem gcd_2_pow_2018_2_pow_2029 : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2047 :=
by
  sorry

end gcd_2_pow_2018_2_pow_2029_l879_87946


namespace range_of_x_l879_87927

theorem range_of_x (a : ℝ) (x : ℝ) (h₁ : a = 1) (h₂ : (x - a) * (x - 3 * a) < 0) (h₃ : 2 < x ∧ x ≤ 3) : 2 < x ∧ x < 3 :=
by sorry

end range_of_x_l879_87927


namespace find_x_value_l879_87936

theorem find_x_value :
  ∃ x : ℝ, (75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734) ∧ (x = 37.03) :=
by {
  sorry
}

end find_x_value_l879_87936


namespace sum_mod_9_l879_87959

theorem sum_mod_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  sorry

end sum_mod_9_l879_87959


namespace data_set_average_l879_87935

theorem data_set_average (a : ℝ) (h : (2 + 3 + 3 + 4 + a) / 5 = 3) : a = 3 := 
sorry

end data_set_average_l879_87935


namespace circle_radius_of_diameter_l879_87980

theorem circle_radius_of_diameter (d : ℝ) (h : d = 22) : d / 2 = 11 :=
by
  sorry

end circle_radius_of_diameter_l879_87980


namespace gain_percent_is_80_l879_87909

theorem gain_percent_is_80 (C S : ℝ) (h : 81 * C = 45 * S) : ((S - C) / C) * 100 = 80 :=
by
  sorry

end gain_percent_is_80_l879_87909


namespace angle_B_is_pi_over_3_l879_87992

theorem angle_B_is_pi_over_3
  (A B C : ℝ) (a b c : ℝ)
  (h_triangle : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (h_sin_ratios : ∃ k > 0, a = 5*k ∧ b = 7*k ∧ c = 8*k) :
  B = π / 3 := 
by
  sorry

end angle_B_is_pi_over_3_l879_87992


namespace all_acute_angles_in_first_quadrant_l879_87941

def terminal_side_same (θ₁ θ₂ : ℝ) : Prop := 
  ∃ (k : ℤ), θ₁ = θ₂ + 360 * k

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def first_quadrant_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem all_acute_angles_in_first_quadrant :
  ∀ θ : ℝ, acute_angle θ → first_quadrant_angle θ :=
by
  intros θ h
  exact h

end all_acute_angles_in_first_quadrant_l879_87941


namespace huahuan_initial_cards_l879_87937

theorem huahuan_initial_cards
  (a b c : ℕ) -- let a, b, c be the initial number of cards Huahuan, Yingying, and Nini have
  (total : a + b + c = 2712)
  (condition_after_50_rounds : ∃ d, b = a + d ∧ c = a + 2 * d) -- after 50 rounds, form an arithmetic sequence
  : a = 754 := sorry

end huahuan_initial_cards_l879_87937


namespace range_of_g_l879_87967

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by
  sorry

end range_of_g_l879_87967


namespace simplify_expression_l879_87988

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (4 * z^2 + 2 * z - 5) = 8 - 9 * z^2 - 2 * z :=
by
  sorry

end simplify_expression_l879_87988


namespace polar_to_cartesian_2_pi_over_6_l879_87904

theorem polar_to_cartesian_2_pi_over_6 :
  let r : ℝ := 2
  let θ : ℝ := (Real.pi / 6)
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (Real.sqrt 3, 1) := by
    -- Initialize the constants and their values
    let r := 2
    let θ := Real.pi / 6
    let x := r * Real.cos θ
    let y := r * Real.sin θ
    -- Placeholder for the actual proof
    sorry

end polar_to_cartesian_2_pi_over_6_l879_87904


namespace sufficient_condition_for_inequality_l879_87908

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → a ≥ 5 :=
by
  sorry

end sufficient_condition_for_inequality_l879_87908


namespace center_of_tangent_circle_lies_on_hyperbola_l879_87907

open Real

def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y + 24 = 0

noncomputable def locus_of_center : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), ∀ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 → 
    dist P (x1, y1) = r + 2 ∧ dist P (x2, y2) = r + 1}

theorem center_of_tangent_circle_lies_on_hyperbola :
  ∀ P : ℝ × ℝ, P ∈ locus_of_center → ∃ (a b : ℝ) (F1 F2 : ℝ × ℝ), ∀ Q : ℝ × ℝ,
    dist Q F1 - dist Q F2 = 1 ∧ 
    dist F1 F2 = 5 ∧
    P ∈ {Q | dist Q F1 - dist Q F2 = 1} :=
sorry

end center_of_tangent_circle_lies_on_hyperbola_l879_87907


namespace thirteenth_result_is_128_l879_87931

theorem thirteenth_result_is_128 
  (avg_all : ℕ → ℕ → ℕ) (avg_first : ℕ → ℕ → ℕ) (avg_last : ℕ → ℕ → ℕ) :
  avg_all 25 20 = (avg_first 12 14) + (avg_last 12 17) + 128 :=
by
  sorry

end thirteenth_result_is_128_l879_87931


namespace expressions_positive_l879_87976

-- Definitions based on given conditions
def A := 2.5
def B := -0.8
def C := -2.2
def D := 1.1
def E := -3.1

-- The Lean statement to prove the necessary expressions are positive numbers.

theorem expressions_positive :
  (B + C) / E = 0.97 ∧
  B * D - A * C = 4.62 ∧
  C / (A * B) = 1.1 :=
by
  -- Assuming given conditions and steps to prove the theorem.
  sorry

end expressions_positive_l879_87976


namespace arithmetic_geometric_sequence_relation_l879_87999

theorem arithmetic_geometric_sequence_relation 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (hA : ∀ n: ℕ, a (n + 1) - a n = a 1) 
  (hG : ∀ n: ℕ, b (n + 1) / b n = b 1) 
  (h1 : a 1 = b 1) 
  (h11 : a 11 = b 11) 
  (h_pos : 0 < a 1 ∧ 0 < a 11 ∧ 0 < b 11 ∧ 0 < b 1) :
  a 6 ≥ b 6 := sorry

end arithmetic_geometric_sequence_relation_l879_87999


namespace point_on_x_axis_l879_87932

theorem point_on_x_axis (m : ℝ) (P : ℝ × ℝ) (hP : P = (m + 3, m - 1)) (hx : P.2 = 0) :
  P = (4, 0) :=
by
  sorry

end point_on_x_axis_l879_87932


namespace flowers_per_pot_l879_87940

def total_gardens : ℕ := 10
def pots_per_garden : ℕ := 544
def total_flowers : ℕ := 174080

theorem flowers_per_pot  :
  (total_flowers / (total_gardens * pots_per_garden)) = 32 :=
by
  -- Here would be the place to provide the proof, but we use sorry for now
  sorry

end flowers_per_pot_l879_87940


namespace max_profit_l879_87978

-- Definitions based on conditions from the problem
def L1 (x : ℕ) : ℤ := -5 * (x : ℤ)^2 + 900 * (x : ℤ) - 16000
def L2 (x : ℕ) : ℤ := 300 * (x : ℤ) - 2000
def total_vehicles := 110
def total_profit (x : ℕ) : ℤ := L1 x + L2 (total_vehicles - x)

-- Statement of the problem
theorem max_profit :
  ∃ x y : ℕ, x + y = 110 ∧ x ≥ 0 ∧ y ≥ 0 ∧
  (L1 x + L2 y = 33000 ∧
   (∀ z w : ℕ, z + w = 110 ∧ z ≥ 0 ∧ w ≥ 0 → L1 z + L2 w ≤ 33000)) :=
sorry

end max_profit_l879_87978


namespace domain_of_f_l879_87990

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

theorem domain_of_f :
  {x : ℝ | f x = Real.log (x^2 - x)} = {x : ℝ | x < 0 ∨ x > 1} :=
sorry

end domain_of_f_l879_87990


namespace B_is_empty_l879_87905

def A : Set ℤ := {0}
def B : Set ℤ := {x | x > 8 ∧ x < 5}
def C : Set ℕ := {x | x - 1 = 0}
def D : Set ℤ := {x | x > 4}

theorem B_is_empty : B = ∅ := by
  sorry

end B_is_empty_l879_87905


namespace statement_B_not_true_l879_87901

def diamondsuit (x y : ℝ) : ℝ := 2 * |(x - y)| + 1

theorem statement_B_not_true : ¬ (∀ x y : ℝ, 3 * diamondsuit x y = 3 * diamondsuit (2 * x) (2 * y)) :=
sorry

end statement_B_not_true_l879_87901


namespace cars_on_river_road_l879_87948

theorem cars_on_river_road (B C : ℕ) (h_ratio : B / C = 1 / 3) (h_fewer : C = B + 40) : C = 60 :=
sorry

end cars_on_river_road_l879_87948


namespace unique_solution_xy_l879_87970

theorem unique_solution_xy
  (x y : ℕ)
  (h1 : (x^3 + y) % (x^2 + y^2) = 0)
  (h2 : (y^3 + x) % (x^2 + y^2) = 0) :
  x = 1 ∧ y = 1 := sorry

end unique_solution_xy_l879_87970


namespace Brenda_new_lead_l879_87950

noncomputable def Brenda_initial_lead : ℤ := 22
noncomputable def Brenda_play_points : ℤ := 15
noncomputable def David_play_points : ℤ := 32

theorem Brenda_new_lead : 
  Brenda_initial_lead + Brenda_play_points - David_play_points = 5 := 
by
  sorry

end Brenda_new_lead_l879_87950


namespace square_area_l879_87994

theorem square_area (s : ℝ) (h : s = 12) : s * s = 144 :=
by
  rw [h]
  norm_num

end square_area_l879_87994


namespace manager_salary_l879_87963

theorem manager_salary (avg_salary_50 : ℕ) (num_employees : ℕ) (increment_new_avg : ℕ)
  (new_avg_salary : ℕ) (total_old_salary : ℕ) (total_new_salary : ℕ) (M : ℕ) :
  avg_salary_50 = 2000 →
  num_employees = 50 →
  increment_new_avg = 250 →
  new_avg_salary = avg_salary_50 + increment_new_avg →
  total_old_salary = num_employees * avg_salary_50 →
  total_new_salary = (num_employees + 1) * new_avg_salary →
  M = total_new_salary - total_old_salary →
  M = 14750 :=
by {
  sorry
}

end manager_salary_l879_87963


namespace find_positive_integers_unique_solution_l879_87958

theorem find_positive_integers_unique_solution :
  ∃ x r p n : ℕ,  
  0 < x ∧ 0 < r ∧ 0 < n ∧  Nat.Prime p ∧ 
  r > 1 ∧ n > 1 ∧ x^r - 1 = p^n ∧ 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := 
    sorry

end find_positive_integers_unique_solution_l879_87958


namespace talkingBirds_count_l879_87960

-- Define the conditions
def totalBirds : ℕ := 77
def nonTalkingBirds : ℕ := 13
def talkingBirds (T : ℕ) : Prop := T + nonTalkingBirds = totalBirds

-- Statement to prove
theorem talkingBirds_count : ∃ T, talkingBirds T ∧ T = 64 :=
by
  -- Proof will go here
  sorry

end talkingBirds_count_l879_87960


namespace max_area_rectangle_l879_87983

theorem max_area_rectangle (P : ℝ) (x : ℝ) (h1 : P = 40) (h2 : 6 * x = P) : 
  2 * (x ^ 2) = 800 / 9 :=
by
  sorry

end max_area_rectangle_l879_87983


namespace isosceles_triangle_base_length_l879_87933

theorem isosceles_triangle_base_length (P B : ℕ) (hP : P = 13) (hB : B = 3) :
    ∃ S : ℕ, S ≠ 3 ∧ S = 3 :=
by
    sorry

end isosceles_triangle_base_length_l879_87933


namespace percentage_cleared_all_sections_l879_87918

def total_candidates : ℝ := 1200
def cleared_none : ℝ := 0.05 * total_candidates
def cleared_one_section : ℝ := 0.25 * total_candidates
def cleared_four_sections : ℝ := 0.20 * total_candidates
def cleared_two_sections : ℝ := 0.245 * total_candidates
def cleared_three_sections : ℝ := 300

-- Let x be the percentage of candidates who cleared all sections
def cleared_all_sections (x: ℝ) : Prop :=
  let total_cleared := (cleared_none + 
                        cleared_one_section + 
                        cleared_four_sections + 
                        cleared_two_sections + 
                        cleared_three_sections + 
                        x * total_candidates / 100)
  total_cleared = total_candidates

theorem percentage_cleared_all_sections :
  ∃ x, cleared_all_sections x ∧ x = 0.5 :=
by
  sorry

end percentage_cleared_all_sections_l879_87918


namespace relationship_among_abc_l879_87956

noncomputable def a := Real.log 2 / Real.log (1/5)
noncomputable def b := 3 ^ (3/5)
noncomputable def c := 4 ^ (1/5)

theorem relationship_among_abc : a < c ∧ c < b := 
by
  sorry

end relationship_among_abc_l879_87956


namespace age_problem_l879_87902

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- State the conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 22) : b = 8 :=
by
  sorry

end age_problem_l879_87902


namespace symmetrical_ring_of_polygons_l879_87972

theorem symmetrical_ring_of_polygons (m n : ℕ) (hn : n ≥ 7) (hm : m ≥ 3) 
  (condition1 : ∀ p1 p2 : ℕ, p1 ≠ p2 → n = 1) 
  (condition2 : ∀ p : ℕ, p * (n - 2) = 4) 
  (condition3 : ∀ p : ℕ, 2 * m - (n - 2) = 4) :
  ∃ k, (k = 6) :=
by
  -- This block is only a placeholder. The actual proof would go here.
  sorry

end symmetrical_ring_of_polygons_l879_87972


namespace alex_original_seat_l879_87974

-- We define a type for seats
inductive Seat where
  | s1 | s2 | s3 | s4 | s5 | s6
  deriving DecidableEq, Inhabited

open Seat

-- Define the initial conditions and movements
def initial_seats : (Fin 6 → Seat) := ![s1, s2, s3, s4, s5, s6]

def move_bella (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

def move_coral (s : Seat) : Seat :=
  match s with
  | s1 => s6  -- two seats left from s1 wraps around to s6
  | s2 => s1
  | s3 => s2
  | s4 => s3
  | s5 => s4
  | s6 => s5

-- Dan and Eve switch seats among themselves
def switch_dan_eve (s : Seat) : Seat :=
  match s with
  | s3 => s4
  | s4 => s3
  | _ => s  -- all other positions remain the same

def move_finn (s : Seat) : Seat :=
  match s with
  | s1 => s2
  | s2 => s3
  | s3 => s4
  | s4 => s5
  | s5 => s6
  | s6 => s1  -- invalid movement for the problem context, can handle separately

-- Define the final seat for Alex
def alex_final_seat : Seat := s6  -- Alex returns to one end seat

-- Define a theorem for the proof of Alex's original seat being Seat.s1
theorem alex_original_seat :
  ∃ (original_seat : Seat), original_seat = s1 :=
  sorry

end alex_original_seat_l879_87974


namespace simplify_trig_identity_l879_87925

open Real

theorem simplify_trig_identity (x y : ℝ) :
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = sin y ^ 2 := 
sorry

end simplify_trig_identity_l879_87925


namespace determine_set_A_l879_87955

-- Define the function f as described
def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n - 1)

-- Define the set A
def A (n : ℕ) : Set ℕ :=
  { x | (Nat.iterate (f n) n x) = x }

-- State the theorem
theorem determine_set_A (n : ℕ) (hn : n > 0) :
    A n = { x | 1 ≤ x ∧ x ≤ 2^n } :=
sorry

end determine_set_A_l879_87955


namespace find_m_plus_n_l879_87953

-- Define the number of ways Blair and Corey can draw the remaining cards
def num_ways_blair_and_corey_draw : ℕ := Nat.choose 50 2

-- Define the function q(a) as given in the problem
noncomputable def q (a : ℕ) : ℚ :=
  (Nat.choose (42 - a) 2 + Nat.choose (a - 1) 2) / num_ways_blair_and_corey_draw

-- Define the problem statement to find the minimum value of a for which q(a) >= 1/2
noncomputable def minimum_a : ℤ :=
  if q 7 >= 1/2 then 7 else 36 -- According to the solution, these are the points of interest

-- The final statement to be proved
theorem find_m_plus_n : minimum_a = 7 ∨ minimum_a = 36 :=
  sorry

end find_m_plus_n_l879_87953


namespace intersection_points_of_circle_and_line_l879_87903

theorem intersection_points_of_circle_and_line :
  (∃ y, (4, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25}) → 
  ∃ s : Finset (ℝ × ℝ), s.card = 2 ∧ ∀ p ∈ s, (p.1 = 4 ∧ (p.1 ^ 2 + p.2 ^ 2 = 25)) :=
by
  sorry

end intersection_points_of_circle_and_line_l879_87903


namespace polynomial_identity_l879_87913

theorem polynomial_identity
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)
  (h : (2*x + 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) :
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 729)
  ∧ (a_1 + a_3 + a_5 = 364)
  ∧ (a_2 + a_4 = 300) := sorry

end polynomial_identity_l879_87913


namespace geom_seq_seventh_term_l879_87995

theorem geom_seq_seventh_term (a r : ℝ) (n : ℕ) (h1 : a = 2) (h2 : r^8 * a = 32) :
  a * r^6 = 128 :=
by
  sorry

end geom_seq_seventh_term_l879_87995


namespace opposite_of_5_is_neg5_l879_87964

def opposite (n x : ℤ) := n + x = 0

theorem opposite_of_5_is_neg5 : opposite 5 (-5) :=
by
  sorry

end opposite_of_5_is_neg5_l879_87964


namespace total_distance_journey_l879_87984

def miles_driven : ℕ := 384
def miles_remaining : ℕ := 816

theorem total_distance_journey :
  miles_driven + miles_remaining = 1200 :=
by
  sorry

end total_distance_journey_l879_87984


namespace comparison_of_a_and_c_l879_87987

variable {α : Type _} [LinearOrderedField α]

theorem comparison_of_a_and_c (a b c : α) (h1 : a > b) (h2 : (a - b) * (b - c) * (c - a) > 0) : a > c :=
sorry

end comparison_of_a_and_c_l879_87987


namespace max_tied_teams_round_robin_l879_87922

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l879_87922


namespace side_length_range_l879_87910

-- Define the inscribed circle diameter condition
def inscribed_circle_diameter (d : ℝ) (cir_diameter : ℝ) := cir_diameter = 1

-- Define inscribed square side condition
def inscribed_square_side (d side : ℝ) :=
  ∃ (triangle_ABC : Type) (AB AC BC : triangle_ABC → ℝ), 
    side = d ∧
    side < 1

-- Define the main theorem: The side length of the inscribed square lies within given bounds
theorem side_length_range (d : ℝ) :
  inscribed_circle_diameter d 1 → inscribed_square_side d d → (4/5) ≤ d ∧ d < 1 :=
by
  intros h1 h2
  sorry

end side_length_range_l879_87910


namespace quadratic_vertex_coordinates_l879_87934

theorem quadratic_vertex_coordinates (x y : ℝ) (h : y = 2 * x^2 - 4 * x + 5) : (x, y) = (1, 3) :=
sorry

end quadratic_vertex_coordinates_l879_87934


namespace exists_number_between_70_and_80_with_gcd_10_l879_87986

theorem exists_number_between_70_and_80_with_gcd_10 :
  ∃ n : ℕ, 70 ≤ n ∧ n ≤ 80 ∧ Nat.gcd 30 n = 10 :=
sorry

end exists_number_between_70_and_80_with_gcd_10_l879_87986


namespace rectangle_ratio_l879_87982

theorem rectangle_ratio (s y x : ℝ) 
  (inner_square_area outer_square_area : ℝ) 
  (h1 : inner_square_area = s^2)
  (h2 : outer_square_area = 9 * inner_square_area)
  (h3 : outer_square_area = (3 * s)^2)
  (h4 : s + 2 * y = 3 * s)
  (h5 : x + y = 3 * s)
  : x / y = 2 := 
by
  -- Proof steps will go here
  sorry

end rectangle_ratio_l879_87982


namespace find_n_for_k_eq_1_l879_87921

theorem find_n_for_k_eq_1 (n : ℤ) (h : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 1)) : n = 5 := 
by 
  sorry

end find_n_for_k_eq_1_l879_87921


namespace pages_to_read_l879_87919

variable (E P_Science P_Civics P_Chinese Total : ℕ)
variable (h_Science : P_Science = 16)
variable (h_Civics : P_Civics = 8)
variable (h_Chinese : P_Chinese = 12)
variable (h_Total : Total = 14)

theorem pages_to_read :
  (E / 4) + (P_Science / 4) + (P_Civics / 4) + (P_Chinese / 4) = Total → 
  E = 20 := by
  sorry

end pages_to_read_l879_87919


namespace geom_seq_sum_eqn_l879_87977

theorem geom_seq_sum_eqn (n : ℕ) (a : ℚ) (r : ℚ) (S_n : ℚ) : 
  a = 1/3 → r = 1/3 → S_n = 80/243 → S_n = a * (1 - r^n) / (1 - r) → n = 5 :=
by
  intros ha hr hSn hSum
  sorry

end geom_seq_sum_eqn_l879_87977
