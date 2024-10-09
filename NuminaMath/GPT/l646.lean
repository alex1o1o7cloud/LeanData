import Mathlib

namespace range_of_a_increasing_function_l646_64622

noncomputable def f (x a : ℝ) := x^3 + a * x + 1 / x

noncomputable def f' (x a : ℝ) := 3 * x^2 - 1 / x^2 + a

theorem range_of_a_increasing_function (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → f' x a ≥ 0) ↔ a ≥ 13 / 4 := 
sorry

end range_of_a_increasing_function_l646_64622


namespace smallest_angle_of_trapezoid_l646_64698

theorem smallest_angle_of_trapezoid (a d : ℝ) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) = 360) → 
  (a + 3 * d = 150) → 
  a = 15 :=
by
  sorry

end smallest_angle_of_trapezoid_l646_64698


namespace time_to_cover_length_correct_l646_64626

-- Given conditions
def speed_escalator := 20 -- ft/sec
def length_escalator := 210 -- feet
def speed_person := 4 -- ft/sec

-- Time is distance divided by speed
def time_to_cover_length : ℚ :=
  length_escalator / (speed_escalator + speed_person)

theorem time_to_cover_length_correct :
  time_to_cover_length = 8.75 := by
  sorry

end time_to_cover_length_correct_l646_64626


namespace min_value_reciprocal_sum_l646_64678

theorem min_value_reciprocal_sum (m n : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) :
  1 / m + 1 / n ≥ 4 :=
sorry

end min_value_reciprocal_sum_l646_64678


namespace least_number_of_teams_l646_64672

theorem least_number_of_teams
  (total_athletes : ℕ)
  (max_team_size : ℕ)
  (h_total : total_athletes = 30)
  (h_max : max_team_size = 12) :
  ∃ (number_of_teams : ℕ) (team_size : ℕ),
    number_of_teams * team_size = total_athletes ∧
    team_size ≤ max_team_size ∧
    number_of_teams = 3 :=
by
  sorry

end least_number_of_teams_l646_64672


namespace slower_plane_speed_l646_64601

-- Let's define the initial conditions and state the theorem in Lean 4
theorem slower_plane_speed 
    (x : ℕ) -- speed of the slower plane
    (h1 : x + 2*x = 900) : -- based on the total distance after 3 hours
    x = 300 :=
by
    -- Proof goes here
    sorry

end slower_plane_speed_l646_64601


namespace choose_four_socks_from_seven_l646_64660

theorem choose_four_socks_from_seven : (Nat.choose 7 4) = 35 :=
by
  sorry

end choose_four_socks_from_seven_l646_64660


namespace determine_function_f_l646_64691

noncomputable def f (c x : ℝ) : ℝ := c ^ (1 / Real.log x)

theorem determine_function_f (f : ℝ → ℝ) (c : ℝ) (Hc : c > 1) :
  (∀ x, 1 < x → 1 < f x) →
  (∀ (x y : ℝ) (u v : ℝ), 1 < x → 1 < y → 0 < u → 0 < v →
    f (x ^ 4 * y ^ v) ≤ (f x) ^ (1 / (4 * u)) * (f y) ^ (1 / (4 * v))) →
  (∀ x : ℝ, 1 < x → f x = c ^ (1 / Real.log x)) :=
by
  sorry

end determine_function_f_l646_64691


namespace find_divisor_l646_64627

def positive_integer := {e : ℕ // e > 0}

theorem find_divisor (d : ℕ) :
  (∃ e : positive_integer, (e.val % 13 = 2)) →
  (∃ n : ℕ, n < 180 ∧ n % d = 5 ∧ ∀ m < 180, m % d = 5 → m = n) →
  d = 175 :=
by
  sorry

end find_divisor_l646_64627


namespace new_area_after_increasing_length_and_width_l646_64611

theorem new_area_after_increasing_length_and_width
  (L W : ℝ)
  (hA : L * W = 450)
  (hL' : 1.2 * L = L')
  (hW' : 1.3 * W = W') :
  (1.2 * L) * (1.3 * W) = 702 :=
by sorry

end new_area_after_increasing_length_and_width_l646_64611


namespace batsman_new_average_l646_64632

variable (A : ℝ) -- Assume that A is the average before the 17th inning
variable (score : ℝ) -- The score in the 17th inning
variable (new_average : ℝ) -- The new average after the 17th inning

-- The conditions
axiom H1 : score = 85
axiom H2 : new_average = A + 3

-- The statement to prove
theorem batsman_new_average : 
    new_average = 37 :=
by 
  sorry

end batsman_new_average_l646_64632


namespace bob_has_17_pennies_l646_64670

-- Definitions based on the problem conditions
variable (a b : ℕ)
def condition1 : Prop := b + 1 = 4 * (a - 1)
def condition2 : Prop := b - 2 = 2 * (a + 2)

-- The main statement to be proven
theorem bob_has_17_pennies (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 17 :=
by
  sorry

end bob_has_17_pennies_l646_64670


namespace maximum_expression_value_l646_64690

theorem maximum_expression_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 - 2 * x * y + 3 * y^2 = 9) :
  x^2 + 2 * x * y + 3 * y^2 ≤ 33 :=
sorry

end maximum_expression_value_l646_64690


namespace income_increase_l646_64655

variable (a : ℝ)

theorem income_increase (h : ∃ a : ℝ, a > 0):
  a * 1.142 = a * 1 + a * 0.142 :=
by
  sorry

end income_increase_l646_64655


namespace segments_after_cuts_l646_64641

-- Definitions from the conditions
def cuts : ℕ := 10

-- Mathematically equivalent proof statement
theorem segments_after_cuts : (cuts + 1 = 11) :=
by sorry

end segments_after_cuts_l646_64641


namespace geometric_sequence_property_l646_64629

open Classical

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_property :
  ∃ (a : ℕ → ℝ) (q : ℝ), q < 0 ∧ geometric_sequence a q ∧
    a 1 = 1 - a 0 ∧ a 3 = 4 - a 2 ∧ a 3 + a 4 = -8 :=
by
  sorry

end geometric_sequence_property_l646_64629


namespace find_n_l646_64692

theorem find_n (n : ℕ) (h₁ : 3 * n + 4 = 13) : n = 3 :=
by 
  sorry

end find_n_l646_64692


namespace exists_zero_in_interval_l646_64608

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem exists_zero_in_interval : ∃ c ∈ Set.Ioo 0 (1/2 : ℝ), f c = 0 := by
  -- proof to be filled in
  sorry

end exists_zero_in_interval_l646_64608


namespace solve_inequality_l646_64636

theorem solve_inequality (a x : ℝ) :
  (a = 1/2 → (x ≠ 1/2 → (x - a) * (x + a - 1) > 0)) ∧
  (a < 1/2 → ((x > (1 - a) ∨ x < a) → (x - a) * (x + a - 1) > 0)) ∧
  (a > 1/2 → ((x > a ∨ x < (1 - a)) → (x - a) * (x + a - 1) > 0)) :=
by
  sorry

end solve_inequality_l646_64636


namespace election_votes_total_l646_64681

-- Definitions representing the conditions
def CandidateAVotes (V : ℕ) := 45 * V / 100
def CandidateBVotes (V : ℕ) := 35 * V / 100
def CandidateCVotes (V : ℕ) := 20 * V / 100

-- Main theorem statement
theorem election_votes_total (V : ℕ) (h1: CandidateAVotes V = 45 * V / 100) (h2: CandidateBVotes V = 35 * V / 100) (h3: CandidateCVotes V = 20 * V / 100)
  (h4: CandidateAVotes V - CandidateBVotes V = 1800) : V = 18000 :=
  sorry

end election_votes_total_l646_64681


namespace chess_tournament_participants_l646_64689

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 378) : n = 28 :=
sorry

end chess_tournament_participants_l646_64689


namespace max_correct_answers_l646_64620

-- Definitions based on the conditions
def total_problems : ℕ := 12
def points_per_correct : ℕ := 6
def points_per_incorrect : ℕ := 3
def max_score : ℤ := 37 -- Final score, using ℤ to handle potential negatives in deducting points

-- The statement to prove
theorem max_correct_answers :
  ∃ (c w : ℕ), c + w = total_problems ∧ points_per_correct * c - points_per_incorrect * (total_problems - c) = max_score ∧ c = 8 :=
by
  sorry

end max_correct_answers_l646_64620


namespace base8_operations_l646_64682

def add_base8 (a b : ℕ) : ℕ :=
  let sum := (a + b) % 8
  sum

def subtract_base8 (a b : ℕ) : ℕ :=
  let diff := (a + 8 - b) % 8
  diff

def step1 := add_base8 672 156
def step2 := subtract_base8 step1 213

theorem base8_operations :
  step2 = 0645 :=
by
  sorry

end base8_operations_l646_64682


namespace figurine_cost_is_one_l646_64663

-- Definitions from the conditions
def cost_per_tv : ℕ := 50
def num_tvs : ℕ := 5
def num_figurines : ℕ := 10
def total_spent : ℕ := 260

-- The price of a single figurine
def cost_per_figurine (total_spent num_tvs cost_per_tv num_figurines : ℕ) : ℕ :=
  (total_spent - num_tvs * cost_per_tv) / num_figurines

-- The theorem statement
theorem figurine_cost_is_one : cost_per_figurine total_spent num_tvs cost_per_tv num_figurines = 1 :=
by
  sorry

end figurine_cost_is_one_l646_64663


namespace abc_value_l646_64683

variables (a b c d e f : ℝ)
variables (h1 : b * c * d = 65)
variables (h2 : c * d * e = 750)
variables (h3 : d * e * f = 250)
variables (h4 : (a * f) / (c * d) = 0.6666666666666666)

theorem abc_value : a * b * c = 130 :=
by { sorry }

end abc_value_l646_64683


namespace number_of_distinct_intersection_points_l646_64677

theorem number_of_distinct_intersection_points :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
  let line := {p : ℝ × ℝ | p.1 = 4}
  let intersection_points := circle ∩ line
  ∃! p : ℝ × ℝ, p ∈ intersection_points :=
by
  sorry

end number_of_distinct_intersection_points_l646_64677


namespace solve_m_n_l646_64618

theorem solve_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 :=
sorry

end solve_m_n_l646_64618


namespace equal_sums_arithmetic_sequences_l646_64603

-- Define the arithmetic sequences and their sums
def s₁ (n : ℕ) : ℕ := n * (5 * n + 13) / 2
def s₂ (n : ℕ) : ℕ := n * (3 * n + 37) / 2

-- State the theorem: for given n != 0, prove s₁ n = s₂ n implies n = 12
theorem equal_sums_arithmetic_sequences (n : ℕ) (h : n ≠ 0) : 
  s₁ n = s₂ n → n = 12 :=
by
  sorry

end equal_sums_arithmetic_sequences_l646_64603


namespace even_function_maximum_l646_64602

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

noncomputable def has_maximum_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x : ℝ, a ≤ x ∧ x ≤ b ∧ ∀ y : ℝ, a ≤ y ∧ y ≤ b → f y ≤ f x

theorem even_function_maximum 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_max_1_7 : has_maximum_on_interval f 1 7) :
  has_maximum_on_interval f (-7) (-1) :=
sorry

end even_function_maximum_l646_64602


namespace number_of_soccer_balls_in_first_set_l646_64646

noncomputable def cost_of_soccer_ball : ℕ := 50
noncomputable def first_cost_condition (F c : ℕ) : Prop := 3 * F + c = 155
noncomputable def second_cost_condition (F : ℕ) : Prop := 2 * F + 3 * cost_of_soccer_ball = 220

theorem number_of_soccer_balls_in_first_set (F : ℕ) :
  (first_cost_condition F 50) ∧ (second_cost_condition F) → 1 = 1 :=
by
  sorry

end number_of_soccer_balls_in_first_set_l646_64646


namespace distance_to_station_is_6_l646_64654

noncomputable def distance_man_walks (walking_speed1 walking_speed2 time_diff: ℝ) : ℝ :=
  let D := (time_diff * walking_speed1 * walking_speed2) / (walking_speed1 - walking_speed2)
  D

theorem distance_to_station_is_6 :
  distance_man_walks 5 6 (12 / 60) = 6 :=
by
  sorry

end distance_to_station_is_6_l646_64654


namespace solution_eq1_solution_eq2_l646_64667

-- Definitions corresponding to the conditions of the problem.
def eq1 (x : ℝ) : Prop := 16 * x^2 = 49
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 64

-- Statements for the proof problem.
theorem solution_eq1 (x : ℝ) : eq1 x → (x = 7 / 4 ∨ x = - (7 / 4)) :=
by
  intro h
  sorry

theorem solution_eq2 (x : ℝ) : eq2 x → (x = 10 ∨ x = -6) :=
by
  intro h
  sorry

end solution_eq1_solution_eq2_l646_64667


namespace find_f_10_l646_64686

def f : ℕ → ℚ := sorry
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = f x / (1 + f x)
axiom f_initial : f 1 = 1

theorem find_f_10 : f 10 = 1 / 10 :=
by
  sorry

end find_f_10_l646_64686


namespace employee_payment_l646_64665

theorem employee_payment
  (A B C : ℝ)
  (h_total : A + B + C = 1500)
  (h_A : A = 1.5 * B)
  (h_C : C = 0.8 * B) :
  A = 682 ∧ B = 454 ∧ C = 364 := by
  sorry

end employee_payment_l646_64665


namespace find_value_l646_64675

variables (x1 x2 y1 y2 : ℝ)

def condition1 := x1 ^ 2 + 5 * x2 ^ 2 = 10
def condition2 := x2 * y1 - x1 * y2 = 5
def condition3 := x1 * y1 + 5 * x2 * y2 = Real.sqrt 105

theorem find_value (h1 : condition1 x1 x2) (h2 : condition2 x1 x2 y1 y2) (h3 : condition3 x1 x2 y1 y2) :
  y1 ^ 2 + 5 * y2 ^ 2 = 23 :=
sorry

end find_value_l646_64675


namespace total_cost_is_2160_l646_64642

variables (x y z : ℝ)

-- Conditions
def cond1 : Prop := x = 0.45 * y
def cond2 : Prop := y = 0.8 * z
def cond3 : Prop := z = x + 640

-- Goal
def total_cost := x + y + z

theorem total_cost_is_2160 (x y z : ℝ) (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 x z) :
  total_cost x y z = 2160 :=
by
  sorry

end total_cost_is_2160_l646_64642


namespace fantasy_gala_handshakes_l646_64625

theorem fantasy_gala_handshakes
    (gremlins imps : ℕ)
    (gremlin_handshakes : ℕ)
    (imp_handshakes : ℕ)
    (imp_gremlin_handshakes : ℕ)
    (total_handshakes : ℕ)
    (h1 : gremlins = 30)
    (h2 : imps = 20)
    (h3 : gremlin_handshakes = (30 * 29) / 2)
    (h4 : imp_handshakes = (20 * 5) / 2)
    (h5 : imp_gremlin_handshakes = 20 * 30)
    (h6 : total_handshakes = gremlin_handshakes + imp_handshakes + imp_gremlin_handshakes) :
    total_handshakes = 1085 := by
    sorry

end fantasy_gala_handshakes_l646_64625


namespace Zoe_given_card_6_l646_64650

-- Define the cards and friends
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def friends : List String := ["Eliza", "Miguel", "Naomi", "Ivan", "Zoe"]

-- Define scores 
def scores (name : String) : ℕ :=
  match name with
  | "Eliza"  => 15
  | "Miguel" => 11
  | "Naomi"  => 9
  | "Ivan"   => 13
  | "Zoe"    => 10
  | _ => 0

-- Each friend is given a pair of cards
def cardAssignments (name : String) : List (ℕ × ℕ) :=
  match name with
  | "Eliza"  => [(6,9), (7,8), (5,10), (4,11), (3,12)]
  | "Miguel" => [(1,10), (2,9), (3,8), (4,7), (5,6)]
  | "Naomi"  => [(1,8), (2,7), (3,6), (4,5)]
  | "Ivan"   => [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
  | "Zoe"    => [(1,9), (2,8), (3,7), (4,6)]
  | _ => []

-- The proof statement
theorem Zoe_given_card_6 : ∃ c1 c2, (c1, c2) ∈ cardAssignments "Zoe" ∧ (c1 = 6 ∨ c2 = 6)
:= by
  sorry -- Proof omitted as per the instructions

end Zoe_given_card_6_l646_64650


namespace largest_value_after_2001_presses_l646_64615

noncomputable def max_value_after_presses (n : ℕ) : ℝ :=
if n = 0 then 1 else sorry -- Placeholder for the actual function definition

theorem largest_value_after_2001_presses :
  max_value_after_presses 2001 = 1 :=
sorry

end largest_value_after_2001_presses_l646_64615


namespace parallelogram_sides_l646_64699

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 3 * x + 6 = 15) 
  (h2 : 10 * y - 2 = 12) :
  x + y = 4.4 := 
sorry

end parallelogram_sides_l646_64699


namespace major_axis_length_of_ellipse_l646_64680

theorem major_axis_length_of_ellipse :
  ∀ {y x : ℝ},
  (y^2 / 25 + x^2 / 15 = 1) → 
  2 * Real.sqrt 25 = 10 :=
by
  intro y x h
  sorry

end major_axis_length_of_ellipse_l646_64680


namespace cube_inequality_of_greater_l646_64623

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l646_64623


namespace value_of_a_plus_b_l646_64668

-- Define the main problem conditions
variables (a b : ℝ)

-- State the problem in Lean
theorem value_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = 3) (h3 : |a - b| = - (a - b)) :
  a + b = 5 ∨ a + b = 1 :=
sorry

end value_of_a_plus_b_l646_64668


namespace students_in_either_but_not_both_l646_64617

-- Definitions and conditions
def both : ℕ := 18
def geom : ℕ := 35
def only_stats : ℕ := 16

-- Correct answer to prove
def total_not_both : ℕ := geom - both + only_stats

theorem students_in_either_but_not_both : total_not_both = 33 := by
  sorry

end students_in_either_but_not_both_l646_64617


namespace range_of_a_l646_64606

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - a / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem range_of_a (e : ℝ) (a : ℝ) (H : ∀ x ∈ Set.Icc 1 e, f x ≥ g x a) :
  -2 ≤ a ∧ a ≤ (2 * e) / (e - 1) :=
by
  sorry

end range_of_a_l646_64606


namespace heather_aprons_l646_64614

theorem heather_aprons :
  ∀ (total sewn already_sewn sewn_today half_remaining tomorrow_sew : ℕ),
    total = 150 →
    already_sewn = 13 →
    sewn_today = 3 * already_sewn →
    sewn = already_sewn + sewn_today →
    remaining = total - sewn →
    half_remaining = remaining / 2 →
    tomorrow_sew = half_remaining →
    tomorrow_sew = 49 := 
by 
  -- The proof is left as an exercise.
  sorry

end heather_aprons_l646_64614


namespace increasing_function_in_interval_l646_64645

noncomputable def y₁ (x : ℝ) : ℝ := abs (x + 1)
noncomputable def y₂ (x : ℝ) : ℝ := 3 - x
noncomputable def y₃ (x : ℝ) : ℝ := 1 / x
noncomputable def y₄ (x : ℝ) : ℝ := -x^2 + 4

theorem increasing_function_in_interval : ∀ x, (0 < x ∧ x < 1) → 
  y₁ x > y₁ (x - 0.1) ∧ y₂ x < y₂ (x - 0.1) ∧ y₃ x < y₃ (x - 0.1) ∧ y₄ x < y₄ (x - 0.1) :=
by {
  sorry
}

end increasing_function_in_interval_l646_64645


namespace function_values_at_mean_l646_64674

noncomputable def f (x : ℝ) : ℝ := x^2 - 10 * x + 16

theorem function_values_at_mean (x₁ x₂ : ℝ) (h₁ : x₁ = 8) (h₂ : x₂ = 2) :
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  f x' = -9 ∧ f x'' = -8 := by
  let x' := (x₁ + x₂) / 2
  let x'' := Real.sqrt (x₁ * x₂)
  have hx' : x' = 5 := sorry
  have hx'' : x'' = 4 := sorry
  have hf_x' : f x' = -9 := sorry
  have hf_x'' : f x'' = -8 := sorry
  exact ⟨hf_x', hf_x''⟩

end function_values_at_mean_l646_64674


namespace Linda_sold_7_tees_l646_64652

variables (T : ℕ)
variables (jeans_price tees_price total_money_from_jeans total_money total_money_from_tees : ℕ)
variables (jeans_sold : ℕ)

def tees_sold :=
  jeans_price = 11 ∧ tees_price = 8 ∧ jeans_sold = 4 ∧
  total_money = 100 ∧ total_money_from_jeans = jeans_sold * jeans_price ∧
  total_money_from_tees = total_money - total_money_from_jeans ∧
  T = total_money_from_tees / tees_price
  
theorem Linda_sold_7_tees (h : tees_sold T jeans_price tees_price total_money_from_jeans total_money total_money_from_tees jeans_sold) : T = 7 :=
by
  sorry

end Linda_sold_7_tees_l646_64652


namespace natural_eq_rational_exists_diff_l646_64624

-- Part (a)
theorem natural_eq (x y : ℕ) (h : x^3 + y = y^3 + x) : x = y := 
by sorry

-- Part (b)
theorem rational_exists_diff (x y : ℚ) (h : x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) : ∃ (x y : ℚ), x ≠ y ∧ x^3 + y = y^3 + x := 
by sorry

end natural_eq_rational_exists_diff_l646_64624


namespace minimum_students_for_200_candies_l646_64621

theorem minimum_students_for_200_candies (candies : ℕ) (students : ℕ) (h_candies : candies = 200) : students = 21 :=
by
  sorry

end minimum_students_for_200_candies_l646_64621


namespace pool_houses_count_l646_64647

-- Definitions based on conditions
def total_houses : ℕ := 65
def num_garage : ℕ := 50
def num_both : ℕ := 35
def num_neither : ℕ := 10
def num_pool : ℕ := total_houses - num_garage - num_neither + num_both

theorem pool_houses_count :
  num_pool = 40 := by
  -- Simplified form of the problem expressed in Lean 4 theorem statement.
  sorry

end pool_houses_count_l646_64647


namespace hyperbolas_same_asymptotes_l646_64639

-- Define the given hyperbolas
def hyperbola1 (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
def hyperbola2 (x y M : ℝ) : Prop := (y^2 / 25) - (x^2 / M) = 1

-- The main theorem statement
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, hyperbola1 x y → hyperbola2 x y M) ↔ M = 225/16 :=
by
  sorry

end hyperbolas_same_asymptotes_l646_64639


namespace arrange_in_order_l646_64604

noncomputable def x1 : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def x2 : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def x3 : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))
noncomputable def x4 : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))

theorem arrange_in_order : 
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 := 
by 
  sorry

end arrange_in_order_l646_64604


namespace equation_root_a_plus_b_l646_64635

theorem equation_root_a_plus_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b ≥ 0) 
(h_root : (∃ x : ℝ, x > 0 ∧ x^3 - x^2 + 18 * x - 320 = 0 ∧ x = Real.sqrt a - ↑b)) : 
a + b = 25 := by
  sorry

end equation_root_a_plus_b_l646_64635


namespace athlete_B_more_stable_l646_64653

variable (average_scores_A average_scores_B : ℝ)
variable (s_A_squared s_B_squared : ℝ)

theorem athlete_B_more_stable
  (h_avg : average_scores_A = average_scores_B)
  (h_var_A : s_A_squared = 1.43)
  (h_var_B : s_B_squared = 0.82) :
  s_A_squared > s_B_squared :=
by 
  rw [h_var_A, h_var_B]
  sorry

end athlete_B_more_stable_l646_64653


namespace mass_ratio_speed_ratio_l646_64610

variable {m1 m2 : ℝ} -- masses of the two balls
variable {V0 V : ℝ} -- velocities before and after collision
variable (h1 : V = 4 * V0) -- speed of m2 is four times that of m1 after collision

theorem mass_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                   (h3 : m1 * V0 = m1 * V + 4 * m2 * V) :
  m2 / m1 = 1 / 2 := sorry

theorem speed_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                    (h3 : m1 * V0 = m1 * V + 4 * m2 * V)
                    (h4 : m2 / m1 = 1 / 2) :
  V0 / V = 3 := sorry

end mass_ratio_speed_ratio_l646_64610


namespace lucy_crayons_correct_l646_64662

-- Define the number of crayons Willy has.
def willyCrayons : ℕ := 5092

-- Define the number of extra crayons Willy has compared to Lucy.
def extraCrayons : ℕ := 1121

-- Define the number of crayons Lucy has.
def lucyCrayons : ℕ := willyCrayons - extraCrayons

-- Statement to prove
theorem lucy_crayons_correct : lucyCrayons = 3971 := 
by
  -- The proof is omitted as per instructions
  sorry

end lucy_crayons_correct_l646_64662


namespace spends_at_arcade_each_weekend_l646_64693

def vanessa_savings : ℕ := 20
def parents_weekly_allowance : ℕ := 30
def dress_cost : ℕ := 80
def weeks : ℕ := 3

theorem spends_at_arcade_each_weekend (arcade_weekend_expense : ℕ) :
  (vanessa_savings + weeks * parents_weekly_allowance - dress_cost = weeks * parents_weekly_allowance - arcade_weekend_expense * weeks) →
  arcade_weekend_expense = 30 :=
by
  intro h
  sorry

end spends_at_arcade_each_weekend_l646_64693


namespace men_in_first_scenario_l646_64600

theorem men_in_first_scenario 
  (M : ℕ) 
  (daily_hours_first weekly_earning_first daily_hours_second weekly_earning_second : ℝ) 
  (number_of_men_second : ℕ)
  (days_per_week : ℕ := 7) 
  (h1 : M * daily_hours_first * days_per_week = weekly_earning_first)
  (h2 : number_of_men_second * daily_hours_second * days_per_week = weekly_earning_second) 
  (h1_value : daily_hours_first = 10) 
  (w1_value : weekly_earning_first = 1400) 
  (h2_value : daily_hours_second = 6) 
  (w2_value : weekly_earning_second = 1890)
  (second_scenario_men : number_of_men_second = 9) : 
  M = 4 :=
by
  sorry

end men_in_first_scenario_l646_64600


namespace motorcycles_count_l646_64661

/-- 
Prove that the number of motorcycles in the parking lot is 28 given the conditions:
1. Each car has 5 wheels (including one spare).
2. Each motorcycle has 2 wheels.
3. Each tricycle has 3 wheels.
4. There are 19 cars in the parking lot.
5. There are 11 tricycles in the parking lot.
6. Altogether all vehicles have 184 wheels.
-/
theorem motorcycles_count 
  (cars := 19) 
  (tricycles := 11) 
  (total_wheels := 184) 
  (wheels_per_car := 5) 
  (wheels_per_tricycle := 3) 
  (wheels_per_motorcycle := 2) :
  (184 - (19 * 5 + 11 * 3)) / 2 = 28 :=
by 
  sorry

end motorcycles_count_l646_64661


namespace checkerboard_corner_sum_is_164_l646_64612

def checkerboard_sum_corners : ℕ :=
  let top_left := 1
  let top_right := 9
  let bottom_left := 73
  let bottom_right := 81
  top_left + top_right + bottom_left + bottom_right

theorem checkerboard_corner_sum_is_164 :
  checkerboard_sum_corners = 164 :=
by
  sorry

end checkerboard_corner_sum_is_164_l646_64612


namespace find_principal_l646_64628

/-- Given that the simple interest SI is Rs. 90, the rate R is 3.5 percent, and the time T is 4 years,
prove that the principal P is approximately Rs. 642.86 using the simple interest formula. -/
theorem find_principal
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 90) (h2 : R = 3.5) (h3 : T = 4) 
  : P = 90 * 100 / (3.5 * 4) :=
by
  sorry

end find_principal_l646_64628


namespace smallest_cut_length_l646_64659

theorem smallest_cut_length (x : ℕ) (h₁ : 9 ≥ x) (h₂ : 12 ≥ x) (h₃ : 15 ≥ x)
  (h₄ : x ≥ 6) (h₅ : x ≥ 12) (h₆ : x ≥ 18) : x = 6 :=
by
  sorry

end smallest_cut_length_l646_64659


namespace mult_mod_7_zero_l646_64649

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end mult_mod_7_zero_l646_64649


namespace existence_of_same_remainder_mod_36_l646_64609

theorem existence_of_same_remainder_mod_36
  (a : Fin 7 → ℕ) :
  ∃ (i j k l : Fin 7), i < j ∧ k < l ∧ (a i)^2 + (a j)^2 % 36 = (a k)^2 + (a l)^2 % 36 := by
  sorry

end existence_of_same_remainder_mod_36_l646_64609


namespace ellipse_eccentricity_half_l646_64684

-- Definitions and assumptions
variable (a b c e : ℝ)
variable (h₁ : a = 2 * c)
variable (h₂ : b = sqrt 3 * c)
variable (eccentricity_def : e = c / a)

-- Theorem statement
theorem ellipse_eccentricity_half : e = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_half_l646_64684


namespace smallest_positive_integer_between_101_and_200_l646_64695

theorem smallest_positive_integer_between_101_and_200 :
  ∃ n : ℕ, n > 1 ∧ n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1 ∧ 101 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_positive_integer_between_101_and_200_l646_64695


namespace solve_for_x_l646_64666

open Real

-- Define the condition and the target result
def target (x : ℝ) : Prop :=
  sqrt (9 + sqrt (16 + 3 * x)) + sqrt (3 + sqrt (4 + x)) = 3 + 3 * sqrt 2

theorem solve_for_x (x : ℝ) (h : target x) : x = 8 * sqrt 2 / 3 :=
  sorry

end solve_for_x_l646_64666


namespace exists_polynomials_Q_R_l646_64630

theorem exists_polynomials_Q_R (P : Polynomial ℝ) (hP : ∀ x > 0, P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ a, 0 ≤ a → ∀ b, 0 ≤ b → Q.coeff a ≥ 0 ∧ R.coeff b ≥ 0) ∧ ∀ x > 0, P.eval x = (Q.eval x) / (R.eval x) := 
by
  sorry

end exists_polynomials_Q_R_l646_64630


namespace find_number_l646_64697

theorem find_number (x : ℝ) (h : (x / 4) + 9 = 15) : x = 24 :=
by
  sorry

end find_number_l646_64697


namespace Liza_rent_l646_64637

theorem Liza_rent :
  (800 - R + 1500 - 117 - 100 - 70 = 1563) -> R = 450 :=
by
  intros h
  sorry

end Liza_rent_l646_64637


namespace at_least_one_inequality_holds_l646_64687

theorem at_least_one_inequality_holds
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l646_64687


namespace game_winning_starting_numbers_count_l646_64607

theorem game_winning_starting_numbers_count : 
  ∃ win_count : ℕ, (win_count = 6) ∧ 
                  ∀ n : ℕ, (1 ≤ n ∧ n < 10) → 
                  (n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) ↔ 
                  ((∃ m, (2 * n ≤ m ∧ m ≤ 3 * n) ∧ m < 2007)  → 
                   (∃ k, (2 * m ≤ k ∧ k ≤ 3 * m) ∧ k ≥ 2007) = false) := 
sorry

end game_winning_starting_numbers_count_l646_64607


namespace roots_of_quadratic_l646_64633

theorem roots_of_quadratic :
  ∃ m n : ℝ, (∀ x : ℝ, x^2 - 4 * x - 1 = 0 → (x = m ∨ x = n)) ∧
            (m + n = 4) ∧
            (m * n = -1) ∧
            (m + n - m * n = 5) :=
by
  sorry

end roots_of_quadratic_l646_64633


namespace compute_four_at_seven_l646_64616

def operation (a b : ℤ) : ℤ :=
  5 * a - 2 * b

theorem compute_four_at_seven : operation 4 7 = 6 :=
by
  sorry

end compute_four_at_seven_l646_64616


namespace problem_statement_l646_64619

theorem problem_statement (a b c : ℤ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 3) 
  (h3 : |c| = 6) 
  (h4 : |a + b| = - (a + b)) 
  (h5 : |a + c| = a + c) : 
  a - b + c = -2 ∨ a - b + c = 4 :=
sorry

end problem_statement_l646_64619


namespace sin_960_eq_sqrt3_over_2_neg_l646_64644

-- Conditions
axiom sine_periodic : ∀ θ, Real.sin (θ + 360 * Real.pi / 180) = Real.sin θ

-- Theorem to prove
theorem sin_960_eq_sqrt3_over_2_neg : Real.sin (960 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  -- skipping the proof
  sorry

end sin_960_eq_sqrt3_over_2_neg_l646_64644


namespace remainder_of_sum_l646_64638

theorem remainder_of_sum (a b : ℤ) (k m : ℤ)
  (h1 : a = 84 * k + 78)
  (h2 : b = 120 * m + 114) :
  (a + b) % 42 = 24 :=
  sorry

end remainder_of_sum_l646_64638


namespace value_of_expression_l646_64658

theorem value_of_expression (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 53) :
  x^3 - y^3 - 2 * (x + y) + 10 = 2011 :=
sorry

end value_of_expression_l646_64658


namespace range_of_values_for_k_l646_64648

theorem range_of_values_for_k (k : ℝ) (h : k ≠ 0) :
  (1 : ℝ) ∈ { x : ℝ | k^2 * x^2 - 6 * k * x + 8 ≥ 0 } ↔ (k ≥ 4 ∨ k ≤ 2) := 
by
  -- proof 
  sorry

end range_of_values_for_k_l646_64648


namespace BoatCrafters_total_canoes_l646_64605

def canoe_production (n : ℕ) : ℕ :=
  if n = 0 then 5 else 3 * canoe_production (n-1) - 1

theorem BoatCrafters_total_canoes : 
  (canoe_production 0 - 1) + (canoe_production 1 - 1) + (canoe_production 2 - 1) + (canoe_production 3 - 1) = 196 := 
by
  sorry

end BoatCrafters_total_canoes_l646_64605


namespace product_of_two_numbers_l646_64643

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h1 : x - y = 2) 
  (h2 : x + y = 8 * (x - y)) 
  (h3 : x * y = 40 * (x - y)) 
  : x * y = 63 := 
by 
  sorry

end product_of_two_numbers_l646_64643


namespace rate_of_interest_l646_64651

variable (P SI T R : ℝ)
variable (hP : P = 400)
variable (hSI : SI = 160)
variable (hT : T = 2)

theorem rate_of_interest :
  (SI = (P * R * T) / 100) → R = 20 :=
by
  intro h
  have h1 : P = 400 := hP
  have h2 : SI = 160 := hSI
  have h3 : T = 2 := hT
  sorry

end rate_of_interest_l646_64651


namespace find_multiplier_l646_64631

-- Define the numbers and the equation based on the conditions
def n : ℝ := 3.0
def m : ℝ := 7

-- State the problem in Lean 4
theorem find_multiplier : m * n = 3 * n + 12 := by
  -- Specific steps skipped; only structure is needed
  sorry

end find_multiplier_l646_64631


namespace only_ten_perfect_square_l646_64640

theorem only_ten_perfect_square (n : ℤ) :
  ∃ k : ℤ, n^4 + 6 * n^3 + 11 * n^2 + 3 * n + 31 = k^2 ↔ n = 10 :=
by
  sorry

end only_ten_perfect_square_l646_64640


namespace ratio_of_milk_and_water_l646_64694

theorem ratio_of_milk_and_water (x y : ℝ) (hx : 9 * x = 9 * y) : 
  let total_milk := (7 * x + 8 * y)
  let total_water := (2 * x + y)
  (total_milk / total_water) = 5 :=
by
  sorry

end ratio_of_milk_and_water_l646_64694


namespace quadratic_relationship_l646_64696

theorem quadratic_relationship (a b c : ℝ) (α : ℝ) (h₁ : α + α^2 = -b / a) (h₂ : α^3 = c / a) : b^2 = 3 * a * c + c^2 :=
by
  sorry

end quadratic_relationship_l646_64696


namespace medicine_supply_duration_l646_64634

theorem medicine_supply_duration
  (pills_per_three_days : ℚ := 1 / 3)
  (total_pills : ℕ := 60)
  (days_per_month : ℕ := 30) :
  (((total_pills : ℚ) * ( 3 / pills_per_three_days)) / days_per_month) = 18 := sorry

end medicine_supply_duration_l646_64634


namespace minimum_travel_time_l646_64671

structure TravelSetup where
  distance_ab : ℝ
  number_of_people : ℕ
  number_of_bicycles : ℕ
  speed_cyclist : ℝ
  speed_pedestrian : ℝ
  unattended_rule : Prop

theorem minimum_travel_time (setup : TravelSetup) : setup.distance_ab = 45 → 
                                                    setup.number_of_people = 3 → 
                                                    setup.number_of_bicycles = 2 → 
                                                    setup.speed_cyclist = 15 → 
                                                    setup.speed_pedestrian = 5 → 
                                                    setup.unattended_rule → 
                                                    ∃ t : ℝ, t = 3 := 
by
  intros
  sorry

end minimum_travel_time_l646_64671


namespace dealer_gross_profit_l646_64664

theorem dealer_gross_profit (purchase_price : ℝ) (markup_rate : ℝ) (selling_price : ℝ) (gross_profit : ℝ) 
  (purchase_price_cond : purchase_price = 150)
  (markup_rate_cond : markup_rate = 0.25)
  (selling_price_eq : selling_price = purchase_price + (markup_rate * selling_price))
  (gross_profit_eq : gross_profit = selling_price - purchase_price) : 
  gross_profit = 50 :=
by
  sorry

end dealer_gross_profit_l646_64664


namespace find_a_and_b_l646_64656

open Function

theorem find_a_and_b (a b : ℚ) (k : ℚ)  (hA : (6 : ℚ) = k * (-3))
    (hB : (a : ℚ) = k * 2)
    (hC : (-1 : ℚ) = k * b) : 
    a = -4 ∧ b = 1 / 2 :=
by
  sorry

end find_a_and_b_l646_64656


namespace astroid_area_l646_64669

-- Definitions coming from the conditions
noncomputable def x (t : ℝ) := 4 * (Real.cos t)^3
noncomputable def y (t : ℝ) := 4 * (Real.sin t)^3

-- The theorem stating the area of the astroid
theorem astroid_area : (∫ t in (0 : ℝ)..(Real.pi / 2), y t * (deriv x t)) * 4 = 24 * Real.pi :=
by
  sorry

end astroid_area_l646_64669


namespace consecutive_grouping_probability_l646_64657

theorem consecutive_grouping_probability :
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements = 1 / 4620 :=
by
  let green_factorial := Nat.factorial 4
  let orange_factorial := Nat.factorial 3
  let blue_factorial := Nat.factorial 5
  let block_arrangements := Nat.factorial 3
  let total_arrangements := Nat.factorial 12
  have h : (block_arrangements * green_factorial * orange_factorial * blue_factorial) = 103680 := sorry
  have h1 : (total_arrangements) = 479001600 := sorry
  calc
    (block_arrangements * green_factorial * orange_factorial * blue_factorial) / total_arrangements
    _ = 103680 / 479001600 := by rw [h, h1]
    _ = 1 / 4620 := sorry

end consecutive_grouping_probability_l646_64657


namespace trapezoid_inscribed_circles_radii_l646_64685

open Real

variables (a b m n : ℝ)
noncomputable def r := (a * sqrt b) / (sqrt a + sqrt b)
noncomputable def R := (b * sqrt a) / (sqrt a + sqrt b)

theorem trapezoid_inscribed_circles_radii
  (h : a < b)
  (hM : m = sqrt (a * b))
  (hN : m = sqrt (a * b)) :
  (r a b = (a * sqrt b) / (sqrt a + sqrt b)) ∧
  (R a b = (b * sqrt a) / (sqrt a + sqrt b)) :=
by
  sorry

end trapezoid_inscribed_circles_radii_l646_64685


namespace field_dimension_area_l646_64688

theorem field_dimension_area (m : ℝ) : (3 * m + 8) * (m - 3) = 120 → m = 7 :=
by
  sorry

end field_dimension_area_l646_64688


namespace unique_n_value_l646_64673

def is_n_table (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) : Prop :=
  ∃ i j, 
    (∀ k : Fin n, A i j ≥ A i k) ∧   -- Max in its row
    (∀ k : Fin n, A i j ≤ A k j)     -- Min in its column

theorem unique_n_value 
  {n : ℕ} (h : 2 ≤ n) 
  (A : Matrix (Fin n) (Fin n) ℕ) 
  (hA : ∀ i j, A i j ∈ Finset.range (n^2)) -- Each number appears exactly once
  (hn : is_n_table n A) : 
  ∃! a, ∃ i j, A i j = a ∧ 
           (∀ k : Fin n, a ≥ A i k) ∧ 
           (∀ k : Fin n, a ≤ A k j) := 
sorry

end unique_n_value_l646_64673


namespace triangle_side_ratio_eq_one_l646_64679

theorem triangle_side_ratio_eq_one
    (a b c C : ℝ)
    (h1 : a = 2 * b * Real.cos C)
    (cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
    (b / c = 1) := 
by 
    sorry

end triangle_side_ratio_eq_one_l646_64679


namespace shuttle_speed_l646_64613

theorem shuttle_speed (v : ℕ) (h : v = 9) : v * 3600 = 32400 :=
by
  sorry

end shuttle_speed_l646_64613


namespace raviraj_cycle_distance_l646_64676

theorem raviraj_cycle_distance :
  ∃ (d : ℝ), d = Real.sqrt ((425: ℝ)^2 + (200: ℝ)^2) ∧ d = 470 := 
by
  sorry

end raviraj_cycle_distance_l646_64676
