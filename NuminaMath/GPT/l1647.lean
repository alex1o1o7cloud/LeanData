import Mathlib

namespace mul_72519_9999_eq_725117481_l1647_164787

theorem mul_72519_9999_eq_725117481 : 72519 * 9999 = 725117481 := by
  sorry

end mul_72519_9999_eq_725117481_l1647_164787


namespace beads_left_in_container_l1647_164722

theorem beads_left_in_container 
  (initial_beads green brown red total_beads taken_beads remaining_beads : Nat) 
  (h1 : green = 1) (h2 : brown = 2) (h3 : red = 3) 
  (h4 : total_beads = green + brown + red)
  (h5 : taken_beads = 2) 
  (h6 : remaining_beads = total_beads - taken_beads) : 
  remaining_beads = 4 := 
by
  sorry

end beads_left_in_container_l1647_164722


namespace max_gold_coins_l1647_164715

theorem max_gold_coins (n : ℕ) (k : ℕ) (H1 : n = 13 * k + 3) (H2 : n < 150) : n ≤ 146 := 
by
  sorry

end max_gold_coins_l1647_164715


namespace gcd_3_1200_1_3_1210_1_l1647_164755

theorem gcd_3_1200_1_3_1210_1 : 
  Int.gcd (3^1200 - 1) (3^1210 - 1) = 59048 := 
by 
  sorry

end gcd_3_1200_1_3_1210_1_l1647_164755


namespace max_chords_l1647_164707

noncomputable def max_closed_chords (n : ℕ) (h : n ≥ 3) : ℕ :=
  n

/-- Given an integer number n ≥ 3 and n distinct points on a circle, labeled 1 through n,
prove that the maximum number of closed chords [ij], i ≠ j, having pairwise non-empty intersections is n. -/
theorem max_chords {n : ℕ} (h : n ≥ 3) :
  max_closed_chords n h = n := 
sorry

end max_chords_l1647_164707


namespace number_of_connections_l1647_164766

theorem number_of_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) :
    (n * m) / 2 = 60 := by
  -- Since each switch is connected to 4 others,
  -- and each connection is counted twice, 
  -- the number of unique connections is 60.
  sorry

end number_of_connections_l1647_164766


namespace problem1_problem2_l1647_164726

theorem problem1 : (82 - 15) * (32 + 18) = 3350 :=
by
  sorry

theorem problem2 : (25 + 4) * 75 = 2175 :=
by
  sorry

end problem1_problem2_l1647_164726


namespace parallelogram_not_symmetrical_l1647_164795

def is_symmetrical (shape : String) : Prop :=
  shape = "Circle" ∨ shape = "Rectangle" ∨ shape = "Isosceles Trapezoid"

theorem parallelogram_not_symmetrical : ¬ is_symmetrical "Parallelogram" :=
by
  sorry

end parallelogram_not_symmetrical_l1647_164795


namespace problem_solution_l1647_164781

theorem problem_solution (x : ℝ) : 
  (x < -2 ∨ (-2 < x ∧ x ≤ 0) ∨ (0 < x ∧ x < 2) ∨ (2 ≤ x ∧ x < (15 - Real.sqrt 257) / 8) ∨ ((15 + Real.sqrt 257) / 8 < x)) ↔ 
  (x^2 - 1) / (x + 2) ≥ 3 / (x - 2) + 7 / 4 := sorry

end problem_solution_l1647_164781


namespace bucket_full_weight_l1647_164709

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end bucket_full_weight_l1647_164709


namespace arithmetic_square_root_problem_l1647_164728

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end arithmetic_square_root_problem_l1647_164728


namespace hyperbola_m_range_l1647_164788

-- Given conditions
def is_hyperbola_equation (m : ℝ) : Prop :=
  ∃ x y : ℝ, (4 - m) ≠ 0 ∧ (2 + m) ≠ 0 ∧ x^2 / (4 - m) - y^2 / (2 + m) = 1

-- Prove the range of m is -2 < m < 4
theorem hyperbola_m_range (m : ℝ) : is_hyperbola_equation m → (-2 < m ∧ m < 4) :=
by
  sorry

end hyperbola_m_range_l1647_164788


namespace inequality_solution_l1647_164732

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2 ↔ (0 < x ∧ x ≤ 0.5) ∨ (6 ≤ x) :=
by { sorry }

end inequality_solution_l1647_164732


namespace percent_voters_for_A_l1647_164753

-- Definitions from conditions
def total_voters : ℕ := 100
def percent_democrats : ℝ := 0.70
def percent_republicans : ℝ := 0.30
def percent_dems_for_A : ℝ := 0.80
def percent_reps_for_A : ℝ := 0.30

-- Calculations based on definitions
def num_democrats := total_voters * percent_democrats
def num_republicans := total_voters * percent_republicans
def dems_for_A := num_democrats * percent_dems_for_A
def reps_for_A := num_republicans * percent_reps_for_A
def total_for_A := dems_for_A + reps_for_A

-- Proof problem statement
theorem percent_voters_for_A : (total_for_A / total_voters) * 100 = 65 :=
by
  sorry

end percent_voters_for_A_l1647_164753


namespace increasing_log_condition_range_of_a_l1647_164717

noncomputable def t (x a : ℝ) := x^2 - a*x + 3*a

theorem increasing_log_condition :
  (∀ x ≥ 2, 2 * x - a ≥ 0) ∧ a > -4 ∧ a ≤ 4 →
  ∀ x ≥ 2, x^2 - a*x + 3*a > 0 :=
by
  sorry

theorem range_of_a
  (h1 : ∀ x ≥ 2, 2 * x - a ≥ 0)
  (h2 : 4 - 2 * a + 3 * a > 0)
  (h3 : ∀ x ≥ 2, t x a > 0)
  : a > -4 ∧ a ≤ 4 :=
by
  sorry

end increasing_log_condition_range_of_a_l1647_164717


namespace evaluate_expression_l1647_164727

theorem evaluate_expression : (164^2 - 148^2) / 16 = 312 := 
by 
  sorry

end evaluate_expression_l1647_164727


namespace normal_vector_proof_l1647_164705

-- Define the 3D vector type
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a specific normal vector n
def n : Vector3D := ⟨1, -2, 2⟩

-- Define the vector v we need to prove is a normal vector of the same plane
def v : Vector3D := ⟨2, -4, 4⟩

-- Define the statement (without the proof)
theorem normal_vector_proof : v = ⟨2 * n.x, 2 * n.y, 2 * n.z⟩ :=
by
  sorry

end normal_vector_proof_l1647_164705


namespace minimum_value_is_1_l1647_164796

def minimum_value_expression (x y : ℝ) : ℝ :=
  x^2 + y^2 - 8*x + 6*y + 26

theorem minimum_value_is_1 (x y : ℝ) (h : x ≥ 4) : 
  minimum_value_expression x y ≥ 1 :=
by {
  sorry
}

end minimum_value_is_1_l1647_164796


namespace marys_remaining_money_l1647_164702

def drinks_cost (p : ℝ) := 4 * p
def medium_pizzas_cost (p : ℝ) := 3 * (3 * p)
def large_pizzas_cost (p : ℝ) := 2 * (5 * p)
def total_initial_money := 50

theorem marys_remaining_money (p : ℝ) : 
  total_initial_money - (drinks_cost p + medium_pizzas_cost p + large_pizzas_cost p) = 50 - 23 * p :=
by
  sorry

end marys_remaining_money_l1647_164702


namespace arithmetic_sequence_common_difference_l1647_164762

theorem arithmetic_sequence_common_difference 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : a 2 = a 1 + d)
  (h4 : a 3 = a 1 + 2 * d)
  (h5 : 2 * S 3 = 3 * S 2 + 6) : d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l1647_164762


namespace distance_between_points_on_line_l1647_164776

theorem distance_between_points_on_line 
  (p q r s : ℝ)
  (line_eq : q = 2 * p + 3) 
  (s_eq : s = 2 * r + 6) :
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) :=
sorry

end distance_between_points_on_line_l1647_164776


namespace total_fish_bought_l1647_164770

theorem total_fish_bought (gold_fish blue_fish : Nat) (h1 : gold_fish = 15) (h2 : blue_fish = 7) : gold_fish + blue_fish = 22 := by
  sorry

end total_fish_bought_l1647_164770


namespace intersection_of_A_and_B_l1647_164758

-- Definitions from conditions
def A : Set ℤ := {x | x - 1 ≥ 0}
def B : Set ℤ := {0, 1, 2}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1647_164758


namespace root_of_equation_in_interval_l1647_164754

theorem root_of_equation_in_interval :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 2^x = 2 - x := 
sorry

end root_of_equation_in_interval_l1647_164754


namespace num_rectangular_arrays_with_48_chairs_l1647_164799

theorem num_rectangular_arrays_with_48_chairs : 
  ∃ n, (∀ (r c : ℕ), 2 ≤ r ∧ 2 ≤ c ∧ r * c = 48 → (n = 8 ∨ n = 0)) ∧ (n = 8) :=
by 
  sorry

end num_rectangular_arrays_with_48_chairs_l1647_164799


namespace james_meditation_sessions_l1647_164737

theorem james_meditation_sessions (minutes_per_session : ℕ) (hours_per_week : ℕ) (days_per_week : ℕ) (h1 : minutes_per_session = 30) (h2 : hours_per_week = 7) (h3 : days_per_week = 7) : 
  (hours_per_week * 60 / days_per_week / minutes_per_session) = 2 := 
by 
  sorry

end james_meditation_sessions_l1647_164737


namespace solve_for_y_l1647_164794

theorem solve_for_y (x y : ℝ) (h1 : x * y = 25) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 5 / 6 := 
by
  sorry

end solve_for_y_l1647_164794


namespace student_history_score_l1647_164718

theorem student_history_score 
  (math : ℕ) 
  (third : ℕ) 
  (average : ℕ) 
  (H : ℕ) 
  (h_math : math = 74)
  (h_third : third = 67)
  (h_avg : average = 75)
  (h_overall_avg : (math + third + H) / 3 = average) : 
  H = 84 :=
by
  sorry

end student_history_score_l1647_164718


namespace eq_condition_implies_inequality_l1647_164772

theorem eq_condition_implies_inequality (a : ℝ) (h_neg_root : 2 * a - 4 < 0) : (a - 3) * (a - 4) > 0 :=
by {
  sorry
}

end eq_condition_implies_inequality_l1647_164772


namespace interior_angle_regular_octagon_l1647_164773

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l1647_164773


namespace ratio_shorter_to_longer_l1647_164785

theorem ratio_shorter_to_longer (x y : ℝ) (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = y / 3) : x / y = 5 / 12 :=
sorry

end ratio_shorter_to_longer_l1647_164785


namespace number_of_games_X_l1647_164720

variable (x : ℕ) -- Total number of games played by team X
variable (y : ℕ) -- Wins by team Y
variable (ly : ℕ) -- Losses by team Y
variable (dy : ℕ) -- Draws by team Y
variable (wx : ℕ) -- Wins by team X
variable (lx : ℕ) -- Losses by team X
variable (dx : ℕ) -- Draws by team X

axiom wins_ratio_X : wx = 3 * x / 4
axiom wins_ratio_Y : y = 2 * (x + 12) / 3
axiom wins_difference : y = wx + 4
axiom losses_difference : ly = lx + 5
axiom draws_difference : dy = dx + 3
axiom eq_losses_draws : lx + dx = (x - wx)

theorem number_of_games_X : x = 48 :=
by
  sorry

end number_of_games_X_l1647_164720


namespace max_profit_price_l1647_164735

-- Define the conditions
def hotel_rooms : ℕ := 50
def base_price : ℕ := 180
def price_increase : ℕ := 10
def expense_per_room : ℕ := 20

-- Define the price as a function of x
def room_price (x : ℕ) : ℕ := base_price + price_increase * x

-- Define the number of occupied rooms as a function of x
def occupied_rooms (x : ℕ) : ℕ := hotel_rooms - x

-- Define the profit function
def profit (x : ℕ) : ℕ := (room_price x - expense_per_room) * occupied_rooms x

-- The statement to be proven:
theorem max_profit_price : ∃ (x : ℕ), room_price x = 350 ∧ ∀ y : ℕ, profit y ≤ profit x :=
by
  sorry

end max_profit_price_l1647_164735


namespace fly_total_distance_l1647_164798

-- Definitions and conditions
def cyclist_speed : ℝ := 10 -- speed of each cyclist in miles per hour
def initial_distance : ℝ := 50 -- initial distance between the cyclists in miles
def fly_speed : ℝ := 15 -- speed of the fly in miles per hour

-- Statement to prove
theorem fly_total_distance : 
  (cyclist_speed * 2 * initial_distance / (cyclist_speed + cyclist_speed) / fly_speed * fly_speed) = 37.5 :=
by
  -- sorry is used here to skip the proof
  sorry

end fly_total_distance_l1647_164798


namespace each_friend_pays_6413_l1647_164712

noncomputable def amount_each_friend_pays (total_bill : ℝ) (friends : ℕ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let bill_after_first_coupon := total_bill * (1 - first_discount)
  let bill_after_second_coupon := bill_after_first_coupon * (1 - second_discount)
  bill_after_second_coupon / friends

theorem each_friend_pays_6413 :
  amount_each_friend_pays 600 8 0.10 0.05 = 64.13 :=
by
  sorry

end each_friend_pays_6413_l1647_164712


namespace mod_equiv_1043_36_mod_equiv_1_10_l1647_164760

open Int

-- Define the integers involved
def a : ℤ := -1043
def m1 : ℕ := 36
def m2 : ℕ := 10

-- Theorems to prove modulo equivalence
theorem mod_equiv_1043_36 : a % m1 = 1 := by
  sorry

theorem mod_equiv_1_10 : 1 % m2 = 1 := by
  sorry

end mod_equiv_1043_36_mod_equiv_1_10_l1647_164760


namespace average_six_consecutive_integers_starting_with_d_l1647_164741

theorem average_six_consecutive_integers_starting_with_d (c : ℝ) (d : ℝ)
  (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5 :=
by
  sorry -- Proof to be completed

end average_six_consecutive_integers_starting_with_d_l1647_164741


namespace Liliane_more_soda_than_Alice_l1647_164761

variable (J : ℝ) -- Represents the amount of soda Jacqueline has

-- Conditions: Representing the amounts for Benjamin, Liliane, and Alice
def B := 1.75 * J
def L := 1.60 * J
def A := 1.30 * J

-- Question: Proving the relationship in percentage terms between the amounts Liliane and Alice have
theorem Liliane_more_soda_than_Alice :
  (L - A) / A * 100 = 23 := 
by sorry

end Liliane_more_soda_than_Alice_l1647_164761


namespace mark_leftover_amount_l1647_164711

-- Definitions
def raise_percentage : ℝ := 0.05
def old_hourly_wage : ℝ := 40
def hours_per_week : ℝ := 8 * 5
def old_weekly_expenses : ℝ := 600
def new_expense : ℝ := 100

-- Calculate new hourly wage
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)

-- Calculate weekly earnings at the new wage
def weekly_earnings : ℝ := new_hourly_wage * hours_per_week

-- Calculate new total weekly expenses
def total_weekly_expenses : ℝ := old_weekly_expenses + new_expense

-- Calculate leftover amount
def leftover_per_week : ℝ := weekly_earnings - total_weekly_expenses

theorem mark_leftover_amount : leftover_per_week = 980 := by
  sorry

end mark_leftover_amount_l1647_164711


namespace S9_is_45_l1647_164765

-- Define the required sequence and conditions
variable {a : ℕ → ℝ} -- a function that gives us the arithmetic sequence
variable {S : ℕ → ℝ} -- a function that gives us the sum of the first n terms of the sequence

-- Define the condition that a_2 + a_8 = 10
axiom a2_a8_condition : a 2 + a 8 = 10

-- Define the arithmetic property of the sequence
axiom arithmetic_property (n m : ℕ) : a (n + m) = a n + a m

-- Define the sum formula for the first n terms of an arithmetic sequence
axiom sum_formula (n : ℕ) : S n = (n / 2) * (a 1 + a n)

-- The main theorem to prove
theorem S9_is_45 : S 9 = 45 :=
by
  -- Here would go the proof, but it is omitted
  sorry

end S9_is_45_l1647_164765


namespace hcf_lcm_fraction_l1647_164769

theorem hcf_lcm_fraction (m n : ℕ) (HCF : Nat.gcd m n = 6) (LCM : Nat.lcm m n = 210) (sum_mn : m + n = 72) : 
  (1 / m : ℚ) + (1 / n : ℚ) = 2 / 35 :=
by
  sorry

end hcf_lcm_fraction_l1647_164769


namespace trig_identity_simplify_l1647_164700

-- Define the problem in Lean 4
theorem trig_identity_simplify (α : Real) : (Real.sin (α - Real.pi / 2) * Real.tan (Real.pi - α)) = Real.sin α :=
by
  sorry

end trig_identity_simplify_l1647_164700


namespace topological_sort_possible_l1647_164745
-- Import the necessary library

-- Definition of simple, directed, and acyclic graph (DAG)
structure SimpleDirectedAcyclicGraph (V : Type*) :=
  (E : V → V → Prop)
  (acyclic : ∀ v : V, ¬(E v v)) -- no loops
  (simple : ∀ (u v : V), (E u v) → ¬(E v u)) -- no bidirectional edges
  (directional : ∀ (u v w : V), E u v → E v w → E u w) -- directional transitivity

-- Existence of topological sort definition
def topological_sort_exists {V : Type*} (G : SimpleDirectedAcyclicGraph V) : Prop :=
  ∃ (numbering : V → ℕ), ∀ (u v : V), (G.E u v) → (numbering u > numbering v)

-- Theorem statement
theorem topological_sort_possible (V : Type*) (G : SimpleDirectedAcyclicGraph V) : topological_sort_exists G :=
  sorry

end topological_sort_possible_l1647_164745


namespace product_is_solution_quotient_is_solution_l1647_164739

-- Definitions and conditions from the problem statement
variable (a b c d : ℤ)

-- The conditions
axiom h1 : a^2 - 5 * b^2 = 1
axiom h2 : c^2 - 5 * d^2 = 1

-- Lean 4 statement for the first part: the product
theorem product_is_solution :
  ∃ (m n : ℤ), ((m + n * (5:ℚ)) = (a + b * (5:ℚ)) * (c + d * (5:ℚ))) ∧ (m^2 - 5 * n^2 = 1) :=
sorry

-- Lean 4 statement for the second part: the quotient
theorem quotient_is_solution :
  ∃ (p q : ℤ), ((p + q * (5:ℚ)) = (a + b * (5:ℚ)) / (c + d * (5:ℚ))) ∧ (p^2 - 5 * q^2 = 1) :=
sorry

end product_is_solution_quotient_is_solution_l1647_164739


namespace sprint_time_l1647_164730

def speed (Mark : Type) : ℝ := 6.0
def distance (Mark : Type) : ℝ := 144.0

theorem sprint_time (Mark : Type) : (distance Mark) / (speed Mark) = 24 := by
  sorry

end sprint_time_l1647_164730


namespace find_m_l1647_164751

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 25) + (p.2 ^ 2 / 16) = 1}
noncomputable def hyperbola (m : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / m) - (p.2 ^ 2 / 5) = 1}

theorem find_m (m : ℝ) (h1 : ∃ f : ℝ × ℝ, f ∈ ellipse ∧ f ∈ hyperbola m) : m = 4 := by
  sorry

end find_m_l1647_164751


namespace triangle_formation_l1647_164774

theorem triangle_formation (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c^2 = a^2 + b^2 + a * b) : 
  a + b > c ∧ a + c > b ∧ c + (a + b) > a :=
by
  sorry

end triangle_formation_l1647_164774


namespace determine_BD_l1647_164738

def quadrilateral (AB BC CD DA BD : ℕ) : Prop :=
AB = 6 ∧ BC = 15 ∧ CD = 8 ∧ DA = 12 ∧ (7 < BD ∧ BD < 18)

theorem determine_BD : ∃ BD : ℕ, quadrilateral 6 15 8 12 BD ∧ 8 ≤ BD ∧ BD ≤ 17 :=
by
  sorry

end determine_BD_l1647_164738


namespace sequence_problem_l1647_164706

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n - 1) = a 1 - a 0

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b n * b (n - 1) = b 1 * b 0

theorem sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : a 0 = -9) (ha1 : a 3 = -1) (ha_seq : arithmetic_sequence a)
  (hb : b 0 = -9) (hb4 : b 4 = -1) (hb_seq : geometric_sequence b) :
  b 2 * (a 2 - a 1) = -8 :=
sorry

end sequence_problem_l1647_164706


namespace head_start_fraction_of_length_l1647_164724

-- Define the necessary variables and assumptions.
variables (Va Vb L H : ℝ)

-- Given conditions
def condition_speed_relation : Prop := Va = (22 / 19) * Vb
def condition_dead_heat : Prop := (L / Va) = ((L - H) / Vb)

-- The statement to be proven
theorem head_start_fraction_of_length (h_speed_relation: condition_speed_relation Va Vb) (h_dead_heat: condition_dead_heat L Va H Vb) : 
  H = (3 / 22) * L :=
sorry

end head_start_fraction_of_length_l1647_164724


namespace decoded_word_is_correct_l1647_164725

-- Assume that we have a way to represent figures and encoded words
structure Figure1
structure Figure2

-- Assume the existence of a key that maps arrow patterns to letters
def decode (f1 : Figure1) (f2 : Figure2) : String := sorry

theorem decoded_word_is_correct (f1 : Figure1) (f2 : Figure2) :
  decode f1 f2 = "КОМПЬЮТЕР" :=
by
  sorry

end decoded_word_is_correct_l1647_164725


namespace cost_of_antibiotics_for_a_week_l1647_164729

noncomputable def antibiotic_cost : ℕ := 3
def doses_per_day : ℕ := 3
def days_in_week : ℕ := 7

theorem cost_of_antibiotics_for_a_week : doses_per_day * days_in_week * antibiotic_cost = 63 :=
by
  sorry

end cost_of_antibiotics_for_a_week_l1647_164729


namespace original_number_exists_l1647_164710

theorem original_number_exists : 
  ∃ (t o : ℕ), (10 * t + o = 74) ∧ (t = o * o - 9) ∧ (10 * o + t = 10 * t + o - 27) :=
by
  sorry

end original_number_exists_l1647_164710


namespace leo_peeled_potatoes_l1647_164792

noncomputable def lucy_rate : ℝ := 4
noncomputable def leo_rate : ℝ := 6
noncomputable def total_potatoes : ℝ := 60
noncomputable def lucy_time_alone : ℝ := 6
noncomputable def total_potatoes_left : ℝ := total_potatoes - lucy_rate * lucy_time_alone
noncomputable def combined_rate : ℝ := lucy_rate + leo_rate
noncomputable def combined_time : ℝ := total_potatoes_left / combined_rate
noncomputable def leo_potatoes : ℝ := combined_time * leo_rate

theorem leo_peeled_potatoes :
  leo_potatoes = 22 :=
by
  sorry

end leo_peeled_potatoes_l1647_164792


namespace quadratic_inequality_ab_l1647_164786

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set -1 < x < 1/3,
    prove that ab = 6. -/
theorem quadratic_inequality_ab (a b : ℝ) (h1 : ∀ x, -1 < x ∧ x < 1 / 3 → a * x ^ 2 + b * x + 1 > 0):
  a * b = 6 := 
sorry

end quadratic_inequality_ab_l1647_164786


namespace temperature_on_tuesday_l1647_164790

variable (T W Th F : ℕ)

-- Conditions
def cond1 : Prop := (T + W + Th) / 3 = 32
def cond2 : Prop := (W + Th + F) / 3 = 34
def cond3 : Prop := F = 44

-- Theorem statement
theorem temperature_on_tuesday : cond1 T W Th → cond2 W Th F → cond3 F → T = 38 :=
by
  sorry

end temperature_on_tuesday_l1647_164790


namespace required_run_rate_is_correct_l1647_164723

open Nat

noncomputable def requiredRunRate (initialRunRate : ℝ) (initialOvers : ℕ) (targetRuns : ℕ) (totalOvers : ℕ) : ℝ :=
  let runsScored := initialRunRate * initialOvers
  let runsNeeded := targetRuns - runsScored
  let remainingOvers := totalOvers - initialOvers
  runsNeeded / (remainingOvers : ℝ)

theorem required_run_rate_is_correct :
  (requiredRunRate 3.6 10 282 50 = 6.15) :=
by
  sorry

end required_run_rate_is_correct_l1647_164723


namespace blue_pill_cost_l1647_164744

theorem blue_pill_cost :
  ∃ (y : ℝ), (∀ (d : ℝ), d = 45) ∧
  (∀ (b : ℝ) (r : ℝ), b = y ∧ r = y - 2) ∧
  ((21 : ℝ) * 45 = 945) ∧
  (b + r = 45) ∧
  y = 23.5 := 
by
  sorry

end blue_pill_cost_l1647_164744


namespace solve_for_a_b_and_extrema_l1647_164708

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end solve_for_a_b_and_extrema_l1647_164708


namespace reeya_fifth_score_l1647_164759

theorem reeya_fifth_score
  (s1 s2 s3 s4 avg: ℝ)
  (h1: s1 = 65)
  (h2: s2 = 67)
  (h3: s3 = 76)
  (h4: s4 = 82)
  (h_avg: avg = 75) :
  ∃ s5, s1 + s2 + s3 + s4 + s5 = 5 * avg ∧ s5 = 85 :=
by
  use 85
  sorry

end reeya_fifth_score_l1647_164759


namespace third_number_is_32_l1647_164764

theorem third_number_is_32 (A B C : ℕ) 
  (hA : A = 24) (hB : B = 36) 
  (hHCF : Nat.gcd (Nat.gcd A B) C = 32) 
  (hLCM : Nat.lcm (Nat.lcm A B) C = 1248) : 
  C = 32 := 
sorry

end third_number_is_32_l1647_164764


namespace third_box_weight_l1647_164778

def box1_height := 1 -- inches
def box1_width := 2 -- inches
def box1_length := 4 -- inches
def box1_weight := 30 -- grams

def box2_height := 3 * box1_height
def box2_width := 2 * box1_width
def box2_length := box1_length

def box3_height := box2_height
def box3_width := box2_width / 2
def box3_length := box2_length

def volume (height : ℕ) (width : ℕ) (length : ℕ) : ℕ := height * width * length

def weight (box1_weight : ℕ) (box1_volume : ℕ) (box3_volume : ℕ) : ℕ := 
  box3_volume / box1_volume * box1_weight

theorem third_box_weight :
  weight box1_weight (volume box1_height box1_width box1_length) 
  (volume box3_height box3_width box3_length) = 90 :=
by
  sorry

end third_box_weight_l1647_164778


namespace percentage_conveyance_l1647_164768

def percentage_on_food := 40 / 100
def percentage_on_rent := 20 / 100
def percentage_on_entertainment := 10 / 100
def salary := 12500
def savings := 2500

def total_percentage_spent := percentage_on_food + percentage_on_rent + percentage_on_entertainment
def total_spent := salary - savings
def amount_spent_on_conveyance := total_spent - (salary * total_percentage_spent)
def percentage_spent_on_conveyance := (amount_spent_on_conveyance / salary) * 100

theorem percentage_conveyance : percentage_spent_on_conveyance = 10 :=
by sorry

end percentage_conveyance_l1647_164768


namespace phone_cost_l1647_164731

theorem phone_cost (C : ℝ) (h1 : 0.40 * C + 780 = C) : C = 1300 := by
  sorry

end phone_cost_l1647_164731


namespace coefficient_x4_in_expansion_sum_l1647_164767

theorem coefficient_x4_in_expansion_sum :
  (Nat.choose 5 4 + Nat.choose 6 4 + Nat.choose 7 4 = 55) :=
by
  sorry

end coefficient_x4_in_expansion_sum_l1647_164767


namespace melanie_dimes_l1647_164721

variable (initial_dimes : ℕ) -- initial dimes Melanie had
variable (dimes_from_dad : ℕ) -- dimes given by dad
variable (dimes_to_mother : ℕ) -- dimes given to mother

def final_dimes (initial_dimes dimes_from_dad dimes_to_mother : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad - dimes_to_mother

theorem melanie_dimes :
  initial_dimes = 7 →
  dimes_from_dad = 8 →
  dimes_to_mother = 4 →
  final_dimes initial_dimes dimes_from_dad dimes_to_mother = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end melanie_dimes_l1647_164721


namespace joan_remaining_balloons_l1647_164743

def initial_balloons : ℕ := 9
def lost_balloons : ℕ := 2
def remaining_balloons : ℕ := initial_balloons - lost_balloons

theorem joan_remaining_balloons : remaining_balloons = 7 := by
  sorry

end joan_remaining_balloons_l1647_164743


namespace football_sampling_l1647_164757

theorem football_sampling :
  ∀ (total_members football_members basketball_members volleyball_members total_sample : ℕ),
  total_members = 120 →
  football_members = 40 →
  basketball_members = 60 →
  volleyball_members = 20 →
  total_sample = 24 →
  (total_sample * football_members / (football_members + basketball_members + volleyball_members) = 8) :=
by 
  intros total_members football_members basketball_members volleyball_members total_sample h_total_members h_football_members h_basketball_members h_volleyball_members h_total_sample
  sorry

end football_sampling_l1647_164757


namespace intersecting_lines_at_point_find_b_plus_m_l1647_164782

theorem intersecting_lines_at_point_find_b_plus_m :
  ∀ (m b : ℝ),
  (12 = m * 4 + 2) →
  (12 = -2 * 4 + b) →
  (b + m = 22.5) :=
by
  intros m b h1 h2
  sorry

end intersecting_lines_at_point_find_b_plus_m_l1647_164782


namespace sum_lent_is_1500_l1647_164777

/--
A person lent a certain sum of money at 4% per annum at simple interest.
In 4 years, the interest amounted to Rs. 1260 less than the sum lent.
Prove that the sum lent was Rs. 1500.
-/
theorem sum_lent_is_1500
  (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)
  (h1 : r = 4) (h2 : t = 4)
  (h3 : I = P - 1260)
  (h4 : I = P * r * t / 100):
  P = 1500 :=
by
  sorry

end sum_lent_is_1500_l1647_164777


namespace number_of_buses_l1647_164771

theorem number_of_buses (total_people : ℕ) (bus_capacity : ℕ) (h1 : total_people = 1230) (h2 : bus_capacity = 48) : 
  Nat.ceil (total_people / bus_capacity : ℝ) = 26 := 
by 
  unfold Nat.ceil 
  sorry

end number_of_buses_l1647_164771


namespace probability_of_all_red_is_correct_l1647_164746

noncomputable def probability_of_all_red_drawn : ℚ :=
  let total_ways := (Nat.choose 10 5)   -- Total ways to choose 5 balls from 10
  let red_ways := (Nat.choose 5 5)      -- Ways to choose all 5 red balls
  red_ways / total_ways

theorem probability_of_all_red_is_correct :
  probability_of_all_red_drawn = 1 / 252 := by
  sorry

end probability_of_all_red_is_correct_l1647_164746


namespace puppies_adopted_per_day_l1647_164783

theorem puppies_adopted_per_day 
    (initial_puppies : ℕ) 
    (additional_puppies : ℕ) 
    (total_days : ℕ) 
    (total_puppies : ℕ)
    (H1 : initial_puppies = 5) 
    (H2 : additional_puppies = 35) 
    (H3 : total_days = 5) 
    (H4 : total_puppies = initial_puppies + additional_puppies) : 
    total_puppies / total_days = 8 := by
  sorry

end puppies_adopted_per_day_l1647_164783


namespace notebook_pen_cost_correct_l1647_164789

noncomputable def notebook_pen_cost : Prop :=
  ∃ (x y : ℝ), 
  3 * x + 2 * y = 7.40 ∧ 
  2 * x + 5 * y = 9.75 ∧ 
  (x + 3 * y) = 5.53

theorem notebook_pen_cost_correct : notebook_pen_cost :=
sorry

end notebook_pen_cost_correct_l1647_164789


namespace evaluate_f_at_2_l1647_164797

def f (x : ℝ) : ℝ := x^2 - x

theorem evaluate_f_at_2 : f 2 = 2 := by
  sorry

end evaluate_f_at_2_l1647_164797


namespace intersection_M_N_l1647_164748

open Set

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end intersection_M_N_l1647_164748


namespace cannot_tile_10x10_board_l1647_164716

-- Define the tiling board problem
def typeA_piece (i j : ℕ) : Prop := 
  ((i ≤ 98) ∧ (j ≤ 98) ∧ (i % 2 = 0) ∧ (j % 2 = 0))

def typeB_piece (i j : ℕ) : Prop := 
  ((i + 2 < 10) ∧ (j + 2 < 10))

def typeC_piece (i j : ℕ) : Prop := 
  ((i % 4 = 0 ∨ i % 4 = 2) ∧ (j % 4 = 0 ∨ j % 4 = 2))

-- Main theorem statement
theorem cannot_tile_10x10_board : 
  ¬ (∃ f : Fin 25 → Fin 10 × Fin 10, 
    (∀ k : Fin 25, typeA_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeB_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeC_piece (f k).1 (f k).2)) :=
sorry

end cannot_tile_10x10_board_l1647_164716


namespace sufficient_but_not_necessary_l1647_164752

theorem sufficient_but_not_necessary (x y : ℝ) (h₁ : x = 2) (h₂ : y = -1) :
    (x + y - 1 = 0) ∧ ¬ ∀ x y, (x + y - 1 = 0) → (x = 2 ∧ y = -1) :=
  by
  sorry

end sufficient_but_not_necessary_l1647_164752


namespace find_pq_cube_l1647_164763

theorem find_pq_cube (p q : ℝ) (h1 : p + q = 5) (h2 : p * q = 3) : (p + q) ^ 3 = 125 := 
by
  -- This is where the proof would go
  sorry

end find_pq_cube_l1647_164763


namespace therapists_next_meeting_day_l1647_164742

theorem therapists_next_meeting_day : Nat.lcm (Nat.lcm 5 2) (Nat.lcm 9 3) = 90 := by
  -- Given that Alex works every 5 days,
  -- Brice works every 2 days,
  -- Emma works every 9 days,
  -- and Fiona works every 3 days, we need to show that the LCM of these numbers is 90.
  sorry

end therapists_next_meeting_day_l1647_164742


namespace combined_rate_of_three_cars_l1647_164784

theorem combined_rate_of_three_cars
  (m : ℕ)
  (ray_avg : ℕ)
  (tom_avg : ℕ)
  (alice_avg : ℕ)
  (h1 : ray_avg = 30)
  (h2 : tom_avg = 15)
  (h3 : alice_avg = 20) :
  let total_distance := 3 * m
  let total_gasoline := m / ray_avg + m / tom_avg + m / alice_avg
  (total_distance / total_gasoline) = 20 := 
by
  sorry

end combined_rate_of_three_cars_l1647_164784


namespace abs_diff_ps_pds_eq_31_100_l1647_164791

-- Defining the conditions
def num_red : ℕ := 500
def num_black : ℕ := 700
def num_blue : ℕ := 800
def total_marbles : ℕ := num_red + num_black + num_blue
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculating P_s and P_d
def ways_same_color : ℕ := choose num_red 2 + choose num_black 2 + choose num_blue 2
def total_ways : ℕ := choose total_marbles 2
def P_s : ℚ := ways_same_color / total_ways

def ways_different_color : ℕ := num_red * num_black + num_red * num_blue + num_black * num_blue
def P_d : ℚ := ways_different_color / total_ways

-- Proving the statement
theorem abs_diff_ps_pds_eq_31_100 : |P_s - P_d| = (31 : ℚ) / 100 := by
  sorry

end abs_diff_ps_pds_eq_31_100_l1647_164791


namespace value_of_b_l1647_164704

theorem value_of_b (a b c : ℤ) : 
  (∃ d : ℤ, a = 17 + d ∧ b = 17 + 2 * d ∧ c = 17 + 3 * d ∧ 41 = 17 + 4 * d) → b = 29 :=
by
  intros h
  sorry


end value_of_b_l1647_164704


namespace infinite_series_sum_l1647_164701

noncomputable def sum_geometric_series (a b : ℝ) (h : ∑' n : ℕ, a / b ^ (n + 1) = 3) : ℝ :=
  ∑' n : ℕ, a / b ^ (n + 1)

theorem infinite_series_sum (a b c : ℝ) (h : sum_geometric_series a b (by sorry) = 3) :
  ∑' n : ℕ, (c * a) / (a + b) ^ (n + 1) = 3 * c / 4 :=
sorry

end infinite_series_sum_l1647_164701


namespace average_age_is_27_l1647_164750

variables (a b c : ℕ)

def average_age_of_a_and_c (a c : ℕ) := (a + c) / 2

def age_of_b := 23

def average_age_of_a_b_and_c (a b c : ℕ) := (a + b + c) / 3

theorem average_age_is_27 (h1 : average_age_of_a_and_c a c = 29) (h2 : b = age_of_b) :
  average_age_of_a_b_and_c a b c = 27 := by
  sorry

end average_age_is_27_l1647_164750


namespace solve_system_l1647_164756

theorem solve_system :
  ∃ x y : ℚ, (4 * x - 7 * y = -20) ∧ (9 * x + 3 * y = -21) ∧ (x = -69 / 25) ∧ (y = 32 / 25) := by
  sorry

end solve_system_l1647_164756


namespace radius_of_smaller_molds_l1647_164740

noncomputable def hemisphereVolume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  ∀ (R r : ℝ), R = 2 ∧ (64 * hemisphereVolume r) = hemisphereVolume R → r = 1 / 2 :=
by
  intros R r h
  sorry

end radius_of_smaller_molds_l1647_164740


namespace tangency_condition_l1647_164713

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6
def hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

-- The theorem statement based on the question and correct answer:
theorem tangency_condition (n : ℝ) (x y : ℝ) : 
  ellipse x y → hyperbola x y n → n = -6 :=
sorry

end tangency_condition_l1647_164713


namespace log_geometric_sequence_l1647_164733

theorem log_geometric_sequence :
  ∀ (a : ℕ → ℝ), (∀ n, 0 < a n) → (∃ r : ℝ, ∀ n, a (n + 1) = a n * r) →
  a 2 * a 18 = 16 → Real.logb 2 (a 10) = 2 :=
by
  intros a h_positive h_geometric h_condition
  sorry

end log_geometric_sequence_l1647_164733


namespace angle_measure_l1647_164719

theorem angle_measure (x : ℝ) (h1 : (180 - x) = 3*x - 2) : x = 45.5 :=
by
  sorry

end angle_measure_l1647_164719


namespace tim_total_money_raised_l1647_164734

-- Definitions based on conditions
def maxDonation : ℤ := 1200
def numMaxDonors : ℤ := 500
def numHalfDonors : ℤ := 3 * numMaxDonors
def halfDonation : ℤ := maxDonation / 2
def totalPercent : ℚ := 0.4

def totalDonationFromMaxDonors : ℤ := numMaxDonors * maxDonation
def totalDonationFromHalfDonors : ℤ := numHalfDonors * halfDonation
def totalDonation : ℤ := totalDonationFromMaxDonors + totalDonationFromHalfDonors

-- Proposition that Tim's total money raised is $3,750,000
theorem tim_total_money_raised : (totalDonation : ℚ) / totalPercent = 3750000 := by
  -- Verified in the proof steps
  sorry

end tim_total_money_raised_l1647_164734


namespace correct_inequality_relation_l1647_164775

theorem correct_inequality_relation :
  ¬(∀ (a b c : ℝ), a > b ↔ a * (c^2) > b * (c^2)) ∧
  ¬(∀ (a b : ℝ), a > b → (1/a) < (1/b)) ∧
  ¬(∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d → a/d > b/c) ∧
  (∀ (a b c : ℝ), a > b ∧ b > 1 ∧ c < 0 → a^c < b^c) := sorry

end correct_inequality_relation_l1647_164775


namespace foci_distance_of_hyperbola_l1647_164749

theorem foci_distance_of_hyperbola :
  ∀ (x y : ℝ), (x^2 / 32 - y^2 / 8 = 1) → 2 * (Real.sqrt (32 + 8)) = 4 * Real.sqrt 10 :=
by
  intros x y h
  sorry

end foci_distance_of_hyperbola_l1647_164749


namespace combined_cost_of_items_is_221_l1647_164779

def wallet_cost : ℕ := 22
def purse_cost : ℕ := 4 * wallet_cost - 3
def shoes_cost : ℕ := wallet_cost + purse_cost + 7
def combined_cost : ℕ := wallet_cost + purse_cost + shoes_cost

theorem combined_cost_of_items_is_221 : combined_cost = 221 := by
  sorry

end combined_cost_of_items_is_221_l1647_164779


namespace simplify_and_evaluate_expr_find_ab_l1647_164793

theorem simplify_and_evaluate_expr (x y : ℝ) (hx : x = 0.5) (hy : y = -1) :
  (x - 5 * y) * (-x - 5 * y) - (-x + 5 * y)^2 = -5.5 :=
by
  rw [hx, hy]
  sorry

theorem find_ab (a b : ℝ) (h : a^2 - 2 * a + b^2 + 4 * b + 5 = 0) :
  (a + b) ^ 2013 = -1 :=
by
  sorry

end simplify_and_evaluate_expr_find_ab_l1647_164793


namespace maci_school_supplies_cost_l1647_164736

theorem maci_school_supplies_cost :
  let blue_pen_cost := 0.10
  let red_pen_cost := 2 * blue_pen_cost
  let pencil_cost := red_pen_cost / 2
  let notebook_cost := 10 * blue_pen_cost
  let blue_pen_count := 10
  let red_pen_count := 15
  let pencil_count := 5
  let notebook_count := 3
  let total_pen_count := blue_pen_count + red_pen_count
  let total_cost_before_discount := 
      blue_pen_count * blue_pen_cost + 
      red_pen_count * red_pen_cost + 
      pencil_count * pencil_cost + 
      notebook_count * notebook_cost
  let pen_discount_rate := if total_pen_count > 12 then 0.10 else 0
  let notebook_discount_rate := if notebook_count > 4 then 0.20 else 0
  let pen_discount := pen_discount_rate * (blue_pen_count * blue_pen_cost + red_pen_count * red_pen_cost)
  let total_cost_after_discount := 
      total_cost_before_discount - pen_discount
  total_cost_after_discount = 7.10 :=
by
  sorry

end maci_school_supplies_cost_l1647_164736


namespace unique_solution_l1647_164780

theorem unique_solution (x y z : ℝ) 
  (h : x^2 + 2*x + y^2 + 4*y + z^2 + 6*z = -14) : 
  x = -1 ∧ y = -2 ∧ z = -3 :=
by
  -- entering main proof section
  sorry

end unique_solution_l1647_164780


namespace card_at_position_52_l1647_164703

def cards_order : List String := ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

theorem card_at_position_52 : cards_order[(52 % 13)] = "A" :=
by
  -- proof will be added here
  sorry

end card_at_position_52_l1647_164703


namespace difference_in_amount_paid_l1647_164714

variable (P Q : ℝ)

theorem difference_in_amount_paid (hP : P > 0) (hQ : Q > 0) :
  (1.10 * P * 0.80 * Q - P * Q) = -0.12 * (P * Q) := 
by 
  sorry

end difference_in_amount_paid_l1647_164714


namespace part_one_part_two_l1647_164747

def universal_set : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

noncomputable def C_R_A : Set ℝ := { x | x < 1 ∨ x ≥ 7 }
noncomputable def C_R_A_union_B : Set ℝ := C_R_A ∪ B

theorem part_one : C_R_A_union_B = { x | x < 1 ∨ x > 2 } :=
sorry

theorem part_two (a : ℝ) (h : A ⊆ C a) : a ≥ 7 :=
sorry

end part_one_part_two_l1647_164747
