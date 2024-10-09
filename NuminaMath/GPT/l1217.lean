import Mathlib

namespace min_orange_chips_l1217_121767

theorem min_orange_chips (p g o : ℕ)
    (h1: g ≥ (1 / 3) * p)
    (h2: g ≤ (1 / 4) * o)
    (h3: p + g ≥ 75) : o = 76 :=
    sorry

end min_orange_chips_l1217_121767


namespace clips_ratio_l1217_121706

def clips (April May: Nat) : Prop :=
  April = 48 ∧ April + May = 72 → (48 / (72 - 48)) = 2

theorem clips_ratio : clips 48 (72 - 48) :=
by
  sorry

end clips_ratio_l1217_121706


namespace intersection_P_Q_l1217_121702

-- Definitions and Conditions
variable (P Q : Set ℕ)
noncomputable def f (t : ℕ) : ℕ := t ^ 2
axiom hQ : Q = {1, 4}

-- Theorem to Prove
theorem intersection_P_Q (P : Set ℕ) (Q : Set ℕ) (hQ : Q = {1, 4})
  (hf : ∀ t ∈ P, f t ∈ Q) : P ∩ Q = {1} ∨ P ∩ Q = ∅ :=
sorry

end intersection_P_Q_l1217_121702


namespace no_graph_for_equation_l1217_121751

theorem no_graph_for_equation (x y : ℝ) : 
  ¬ ∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := 
by 
  sorry

end no_graph_for_equation_l1217_121751


namespace maximum_value_of_3m_4n_l1217_121732

noncomputable def max_value (m n : ℕ) : ℕ :=
  3 * m + 4 * n

theorem maximum_value_of_3m_4n 
  (m n : ℕ) 
  (h_even : ∀ i, i < m → (2 * (i + 1)) > 0) 
  (h_odd : ∀ j, j < n → (2 * j + 1) > 0)
  (h_sum : m * (m + 1) + n^2 ≤ 1987) 
  (h_odd_n : n % 2 = 1) :
  max_value m n ≤ 221 := 
sorry

end maximum_value_of_3m_4n_l1217_121732


namespace addition_problem_l1217_121757

theorem addition_problem (x y S : ℕ) 
    (h1 : x = S - 2000)
    (h2 : S = y + 6) :
    x = 6 ∧ y = 2000 ∧ S = 2006 :=
by
  -- The proof will go here
  sorry

end addition_problem_l1217_121757


namespace polygon_sides_l1217_121746

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 > 2970) :
  n = 19 :=
by
  sorry

end polygon_sides_l1217_121746


namespace triangle_PQR_area_l1217_121737

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-2, 2)
def Q : (ℝ × ℝ) := (8, 2)
def R : (ℝ × ℝ) := (4, -4)

-- Define a function to calculate the area of triangle
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Lean statement to prove the area of triangle PQR is 30 square units
theorem triangle_PQR_area : triangle_area P Q R = 30 := by
  sorry

end triangle_PQR_area_l1217_121737


namespace longest_tape_length_l1217_121788

/-!
  Problem: Find the length of the longest tape that can exactly measure the lengths 
  24 m, 36 m, and 54 m in cm.
  
  Solution: Convert the given lengths to the same unit (cm), then find their GCD.
  
  Given: Lengths are 2400 cm, 3600 cm, and 5400 cm.
  To Prove: gcd(2400, 3600, 5400) = 300.
-/

theorem longest_tape_length (a b c : ℕ) : a = 2400 → b = 3600 → c = 5400 → Nat.gcd (Nat.gcd a b) c = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- omitted proof steps
  sorry

end longest_tape_length_l1217_121788


namespace tan_theta_minus_pi_over_4_l1217_121721

theorem tan_theta_minus_pi_over_4 (θ : Real) (k : ℤ)
  (h1 : - (π / 2) + (2 * k * π) < θ)
  (h2 : θ < 2 * k * π)
  (h3 : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 :=
sorry

end tan_theta_minus_pi_over_4_l1217_121721


namespace smallest_integer_l1217_121768

theorem smallest_integer (n : ℕ) (h : n > 0) (h1 : lcm 36 n / gcd 36 n = 24) : n = 96 :=
sorry

end smallest_integer_l1217_121768


namespace parallel_vectors_xy_sum_l1217_121745

theorem parallel_vectors_xy_sum (x y : ℚ) (k : ℚ) 
  (h1 : (2, 4, -5) = (2 * k, 4 * k, -5 * k)) 
  (h2 : (3, x, y) = (2 * k, 4 * k, -5 * k)) 
  (h3 : 3 = 2 * k) : 
  x + y = -3 / 2 :=
by
  sorry

end parallel_vectors_xy_sum_l1217_121745


namespace clayton_total_points_l1217_121761

theorem clayton_total_points 
  (game1 game2 game3 : ℕ)
  (game1_points : game1 = 10)
  (game2_points : game2 = 14)
  (game3_points : game3 = 6)
  (game4 : ℕ)
  (game4_points : game4 = (game1 + game2 + game3) / 3) :
  game1 + game2 + game3 + game4 = 40 :=
sorry

end clayton_total_points_l1217_121761


namespace daily_savings_in_dollars_l1217_121774

-- Define the total savings and the number of days
def total_savings_in_dimes : ℕ := 3
def number_of_days : ℕ := 30

-- Define the conversion factor from dimes to dollars
def dime_to_dollar : ℝ := 0.10

-- Prove that the daily savings in dollars is $0.01
theorem daily_savings_in_dollars : total_savings_in_dimes / number_of_days * dime_to_dollar = 0.01 :=
by sorry

end daily_savings_in_dollars_l1217_121774


namespace purple_cars_count_l1217_121794

theorem purple_cars_count
    (P R G : ℕ)
    (h1 : R = P + 6)
    (h2 : G = 4 * R)
    (h3 : P + R + G = 312) :
    P = 47 :=
by 
  sorry

end purple_cars_count_l1217_121794


namespace percentage_less_A_than_B_l1217_121747

theorem percentage_less_A_than_B :
  ∀ (full_marks A_marks D_marks C_marks B_marks : ℝ),
    full_marks = 500 →
    A_marks = 360 →
    D_marks = 0.80 * full_marks →
    C_marks = (1 - 0.20) * D_marks →
    B_marks = (1 + 0.25) * C_marks →
    ((B_marks - A_marks) / B_marks) * 100 = 10 :=
  by intros full_marks A_marks D_marks C_marks B_marks
     intros h_full h_A h_D h_C h_B
     sorry

end percentage_less_A_than_B_l1217_121747


namespace tables_difference_l1217_121792

theorem tables_difference (N O : ℕ) (h1 : N + O = 40) (h2 : 6 * N + 4 * O = 212) : N - O = 12 :=
sorry

end tables_difference_l1217_121792


namespace number_of_sophomores_l1217_121790

theorem number_of_sophomores (n x : ℕ) (freshmen seniors selected freshmen_selected : ℕ)
  (h_freshmen : freshmen = 450)
  (h_seniors : seniors = 250)
  (h_selected : selected = 60)
  (h_freshmen_selected : freshmen_selected = 27)
  (h_eq : selected / (freshmen + seniors + x) = freshmen_selected / freshmen) :
  x = 300 := by
  sorry

end number_of_sophomores_l1217_121790


namespace M_gt_N_l1217_121773

theorem M_gt_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  M > N := by
  sorry

end M_gt_N_l1217_121773


namespace third_jumper_height_l1217_121778

/-- 
  Ravi can jump 39 inches high.
  Ravi can jump 1.5 times higher than the average height of three other jumpers.
  The three jumpers can jump 23 inches, 27 inches, and some unknown height x.
  Prove that the unknown height x is 28 inches.
-/
theorem third_jumper_height (x : ℝ) (h₁ : 39 = 1.5 * (23 + 27 + x) / 3) : 
  x = 28 :=
sorry

end third_jumper_height_l1217_121778


namespace n_must_be_power_of_3_l1217_121720

theorem n_must_be_power_of_3 (n : ℕ) (h1 : 0 < n) (h2 : Prime (4 ^ n + 2 ^ n + 1)) : ∃ k : ℕ, n = 3 ^ k :=
by
  sorry

end n_must_be_power_of_3_l1217_121720


namespace cinema_cost_comparison_l1217_121752

theorem cinema_cost_comparison (x : ℕ) (hx : x = 1000) :
  let cost_A := if x ≤ 100 then 30 * x else 24 * x + 600
  let cost_B := 27 * x
  cost_A < cost_B :=
by
  sorry

end cinema_cost_comparison_l1217_121752


namespace magnets_per_earring_l1217_121705

theorem magnets_per_earring (M : ℕ) (h : 4 * (3 * M / 2) = 24) : M = 4 :=
by
  sorry

end magnets_per_earring_l1217_121705


namespace irrational_sqrt3_l1217_121763

theorem irrational_sqrt3 : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a * a = 3 * b * b) :=
by
  sorry

end irrational_sqrt3_l1217_121763


namespace wolf_hunger_if_eats_11_kids_l1217_121712

variable (p k : ℝ)  -- Define the satiety values of a piglet and a kid.
variable (H : ℝ)    -- Define the satiety threshold for "enough to remove hunger".

-- Conditions from the problem:
def condition1 : Prop := 3 * p + 7 * k < H  -- The wolf feels hungry after eating 3 piglets and 7 kids.
def condition2 : Prop := 7 * p + k > H      -- The wolf suffers from overeating after eating 7 piglets and 1 kid.

-- Statement to prove:
theorem wolf_hunger_if_eats_11_kids (p k H : ℝ) 
  (h1 : condition1 p k H) (h2 : condition2 p k H) : 11 * k < H :=
by
  sorry

end wolf_hunger_if_eats_11_kids_l1217_121712


namespace billy_finished_before_margaret_l1217_121717

-- Define the conditions
def billy_first_laps_time : ℕ := 2 * 60
def billy_next_three_laps_time : ℕ := 4 * 60
def billy_ninth_lap_time : ℕ := 1 * 60
def billy_tenth_lap_time : ℕ := 150
def margaret_total_time : ℕ := 10 * 60

-- The main statement to prove that Billy finished 30 seconds before Margaret
theorem billy_finished_before_margaret :
  (billy_first_laps_time + billy_next_three_laps_time + billy_ninth_lap_time + billy_tenth_lap_time) + 30 = margaret_total_time :=
by
  sorry

end billy_finished_before_margaret_l1217_121717


namespace infinite_very_good_pairs_l1217_121791

-- Defining what it means for a pair to be "good"
def is_good (m n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ m ↔ p ∣ n)

-- Defining what it means for a pair to be "very good"
def is_very_good (m n : ℕ) : Prop :=
  is_good m n ∧ is_good (m + 1) (n + 1)

-- The theorem to prove: infiniteness of very good pairs
theorem infinite_very_good_pairs : Infinite {p : ℕ × ℕ | is_very_good p.1 p.2} :=
  sorry

end infinite_very_good_pairs_l1217_121791


namespace abc_value_l1217_121715

theorem abc_value (a b c : ℝ) 
  (h0 : (a * (0 : ℝ)^2 + b * (0 : ℝ) + c) = 7) 
  (h1 : (a * (1 : ℝ)^2 + b * (1 : ℝ) + c) = 4) : 
  a + b + 2 * c = 11 :=
by sorry

end abc_value_l1217_121715


namespace intersection_of_sets_l1217_121782

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l1217_121782


namespace find_divisor_l1217_121795

theorem find_divisor (x : ℤ) : 83 = 9 * x + 2 → x = 9 :=
by
  sorry

end find_divisor_l1217_121795


namespace uber_profit_l1217_121707

-- Define conditions
def income : ℕ := 30000
def initial_cost : ℕ := 18000
def trade_in : ℕ := 6000

-- Define depreciation cost
def depreciation_cost : ℕ := initial_cost - trade_in

-- Define the profit
def profit : ℕ := income - depreciation_cost

-- The theorem to be proved
theorem uber_profit : profit = 18000 := by 
  sorry

end uber_profit_l1217_121707


namespace simplify_expression_l1217_121777

theorem simplify_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  ((a^3 - a^2 * b) / (a^2 * b) - (a^2 * b - b^3) / (a * b - b^2) - (a * b) / (a^2 - b^2)) = 
  (-3 * a) / (a^2 - b^2) := 
by
  sorry

end simplify_expression_l1217_121777


namespace max_possible_cables_l1217_121776

theorem max_possible_cables (num_employees : ℕ) (num_brand_X : ℕ) (num_brand_Y : ℕ) 
  (max_connections : ℕ) (num_cables : ℕ) :
  num_employees = 40 →
  num_brand_X = 25 →
  num_brand_Y = 15 →
  max_connections = 3 →
  (∀ x : ℕ, x < max_connections → num_cables ≤ 3 * num_brand_Y) →
  num_cables = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_possible_cables_l1217_121776


namespace count_seating_arrangements_l1217_121709

/-
  Definition of the seating problem at the round table:
  - The committee has six members from each of three species: Martians (M), Venusians (V), and Earthlings (E).
  - The table has 18 seats numbered from 1 to 18.
  - Seat 1 is occupied by a Martian, and seat 18 is occupied by an Earthling.
  - Martians cannot sit immediately to the left of Venusians.
  - Venusians cannot sit immediately to the left of Earthlings.
  - Earthlings cannot sit immediately to the left of Martians.
-/
def num_arrangements_valid_seating : ℕ := -- the number of valid seating arrangements
  sorry

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def N : ℕ := 347

theorem count_seating_arrangements :
  num_arrangements_valid_seating = N * (factorial 6)^3 :=
sorry

end count_seating_arrangements_l1217_121709


namespace change_color_while_preserving_friendship_l1217_121797

-- Definitions
def children := Fin 10000
def colors := Fin 7
def friends (a b : children) : Prop := sorry -- mutual and exactly 11 friends per child
def refuses_to_change (c : children) : Prop := sorry -- only 100 specified children refuse to change color

theorem change_color_while_preserving_friendship :
  ∃ c : children, ¬refuses_to_change c ∧
    ∃ new_color : colors, 
      (∀ friend : children, friends c friend → 
      (∃ current_color current_friend_color : colors, current_color ≠ current_friend_color)) :=
sorry

end change_color_while_preserving_friendship_l1217_121797


namespace correlation_statements_l1217_121725

def heavy_snow_predicts_harvest_year (heavy_snow benefits_wheat : Prop) : Prop := benefits_wheat → heavy_snow
def great_teachers_produce_students (great_teachers outstanding_students : Prop) : Prop := great_teachers → outstanding_students
def smoking_is_harmful (smoking harmful_to_health : Prop) : Prop := smoking → harmful_to_health
def magpies_call_signifies_joy (magpies_call joy_signified : Prop) : Prop := joy_signified → magpies_call

theorem correlation_statements (heavy_snow benefits_wheat great_teachers outstanding_students smoking harmful_to_health magpies_call joy_signified : Prop)
  (H1 : heavy_snow_predicts_harvest_year heavy_snow benefits_wheat)
  (H2 : great_teachers_produce_students great_teachers outstanding_students)
  (H3 : smoking_is_harmful smoking harmful_to_health) :
  ¬ magpies_call_signifies_joy magpies_call joy_signified := sorry

end correlation_statements_l1217_121725


namespace contrapositive_proof_l1217_121708

theorem contrapositive_proof (m : ℕ) (h_pos : 0 < m) :
  (¬ (∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) :=
sorry

end contrapositive_proof_l1217_121708


namespace smallest_x_for_multiple_of_625_l1217_121748

theorem smallest_x_for_multiple_of_625 (x : ℕ) (hx_pos : 0 < x) : (500 * x) % 625 = 0 → x = 5 :=
by
  sorry

end smallest_x_for_multiple_of_625_l1217_121748


namespace find_length_QT_l1217_121703

noncomputable def length_RS : ℝ := 75
noncomputable def length_PQ : ℝ := 36
noncomputable def length_PT : ℝ := 12

theorem find_length_QT :
  ∀ (PQRS : Type)
  (P Q R S T : PQRS)
  (h_RS_perp_PQ : true)
  (h_PQ_perp_RS : true)
  (h_PT_perpendicular_to_PR : true),
  QT = 24 :=
by
  sorry

end find_length_QT_l1217_121703


namespace peter_hunts_3_times_more_than_mark_l1217_121758

theorem peter_hunts_3_times_more_than_mark : 
  ∀ (Sam Rob Mark Peter : ℕ),
  Sam = 6 →
  Rob = Sam / 2 →
  Mark = (Sam + Rob) / 3 →
  Sam + Rob + Mark + Peter = 21 →
  Peter = 3 * Mark :=
by
  intros Sam Rob Mark Peter h1 h2 h3 h4
  sorry

end peter_hunts_3_times_more_than_mark_l1217_121758


namespace jerry_won_games_l1217_121724

theorem jerry_won_games 
  (T : ℕ) (K D J : ℕ) 
  (h1 : T = 32) 
  (h2 : K = D + 5) 
  (h3 : D = J + 3) : 
  J = 7 := 
sorry

end jerry_won_games_l1217_121724


namespace isosceles_triangle_base_angle_l1217_121735

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (base_angle : ℝ) 
  (h1 : vertex_angle = 60) 
  (h2 : 2 * base_angle + vertex_angle = 180) : 
  base_angle = 60 := 
by 
  sorry

end isosceles_triangle_base_angle_l1217_121735


namespace find_a_l1217_121718

theorem find_a (a : ℝ) :
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 :=
sorry

end find_a_l1217_121718


namespace total_num_novels_receiving_prizes_l1217_121750

-- Definitions based on conditions
def total_prize_money : ℕ := 800
def first_place_prize : ℕ := 200
def second_place_prize : ℕ := 150
def third_place_prize : ℕ := 120
def remaining_award_amount : ℕ := 22

-- Total number of novels receiving prizes
theorem total_num_novels_receiving_prizes : 
  (3 + (total_prize_money - (first_place_prize + second_place_prize + third_place_prize)) / remaining_award_amount) = 18 :=
by {
  -- We leave the proof as an exercise (denoted by sorry)
  sorry
}

end total_num_novels_receiving_prizes_l1217_121750


namespace prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l1217_121722

-- Show that if \( p \) is a prime number, then \( p \) divides \( (p-1)! + 1 \).
theorem prime_divides_factorial_plus_one (p : ℕ) (hp : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

-- Show that if \( n \) is not a prime number, then \( n \) does not divide \( (n-1)! + 1 \).
theorem non_prime_not_divides_factorial_plus_one (n : ℕ) (hn : ¬Nat.Prime n) : ¬(n ∣ (Nat.factorial (n - 1) + 1)) :=
sorry

-- Calculate the remainder of the division of \((n-1)!\) by \( n \).
theorem factorial_mod_non_prime_is_zero (n : ℕ) (hn : ¬Nat.Prime n) : (Nat.factorial (n - 1)) % n = 0 :=
sorry

end prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l1217_121722


namespace four_cells_same_color_rectangle_l1217_121789

theorem four_cells_same_color_rectangle (color : Fin 3 → Fin 7 → Bool) :
  ∃ (r₁ r₂ r₃ r₄ : Fin 3) (c₁ c₂ c₃ c₄ : Fin 7), 
    r₁ ≠ r₂ ∧ r₃ ≠ r₄ ∧ c₁ ≠ c₂ ∧ c₃ ≠ c₄ ∧ 
    r₁ = r₃ ∧ r₂ = r₄ ∧ c₁ = c₃ ∧ c₂ = c₄ ∧
    color r₁ c₁ = color r₁ c₂ ∧ color r₂ c₁ = color r₂ c₂ := sorry

end four_cells_same_color_rectangle_l1217_121789


namespace forgot_days_l1217_121760

def July_days : ℕ := 31
def days_took_capsules : ℕ := 27

theorem forgot_days : July_days - days_took_capsules = 4 :=
by
  sorry

end forgot_days_l1217_121760


namespace toothpick_removal_l1217_121704

/-- Given 40 toothpicks used to create 10 squares and 15 triangles, with each square formed by 
4 toothpicks and each triangle formed by 3 toothpicks, prove that removing 10 toothpicks is 
sufficient to ensure no squares or triangles remain. -/
theorem toothpick_removal (n : ℕ) (squares triangles : ℕ) (sq_toothpicks tri_toothpicks : ℕ) 
    (total_toothpicks : ℕ) (remove_toothpicks : ℕ) 
    (h1 : n = 40) 
    (h2 : squares = 10) 
    (h3 : triangles = 15) 
    (h4 : sq_toothpicks = 4) 
    (h5 : tri_toothpicks = 3) 
    (h6 : total_toothpicks = n) 
    (h7 : remove_toothpicks = 10) 
    (h8 : (squares * sq_toothpicks + triangles * tri_toothpicks) = total_toothpicks) :
  remove_toothpicks = 10 :=
by
  sorry

end toothpick_removal_l1217_121704


namespace min_a2_plus_b2_l1217_121754

-- Define circle and line intercept conditions
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 2
def line_eq (a b x y : ℝ) : Prop := a * x + 2 * b * y - 4 = 0
def chord_length (chord_len : ℝ) : Prop := chord_len = 4

-- Define the final minimum value to prove
def min_value (a b : ℝ) : ℝ := a^2 + b^2

-- Proving the specific value considering the conditions
theorem min_a2_plus_b2 (a b : ℝ) (h1 : b = a + 2) (h2 : chord_length 4) : min_value a b = 2 := by
  sorry

end min_a2_plus_b2_l1217_121754


namespace remainder_98765432101_div_240_l1217_121765

theorem remainder_98765432101_div_240 :
  (98765432101 % 240) = 61 :=
by
  -- Proof to be filled in later
  sorry

end remainder_98765432101_div_240_l1217_121765


namespace negation_of_proposition_l1217_121743

theorem negation_of_proposition (x : ℝ) :
  ¬ (∃ x > -1, x^2 + x - 2018 > 0) ↔ ∀ x > -1, x^2 + x - 2018 ≤ 0 := sorry

end negation_of_proposition_l1217_121743


namespace number_of_workers_l1217_121700

theorem number_of_workers (supervisors team_leads_per_supervisor workers_per_team_lead : ℕ) 
    (h_supervisors : supervisors = 13)
    (h_team_leads_per_supervisor : team_leads_per_supervisor = 3)
    (h_workers_per_team_lead : workers_per_team_lead = 10):
    supervisors * team_leads_per_supervisor * workers_per_team_lead = 390 :=
by
  -- to avoid leaving the proof section empty and potentially creating an invalid Lean statement
  sorry

end number_of_workers_l1217_121700


namespace rectangle_ratio_l1217_121749

noncomputable def ratio_of_sides (a b : ℝ) : ℝ := a / b

theorem rectangle_ratio (a b d : ℝ) (h1 : d = Real.sqrt (a^2 + b^2)) (h2 : (a/b)^2 = b/d) : 
  ratio_of_sides a b = (Real.sqrt 5 - 1) / 3 :=
by sorry

end rectangle_ratio_l1217_121749


namespace airplane_distance_difference_l1217_121759

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end airplane_distance_difference_l1217_121759


namespace wheat_flour_packets_correct_l1217_121711

-- Define the initial amount of money Victoria had.
def initial_amount : ℕ := 500

-- Define the cost and quantity of rice packets Victoria bought.
def rice_packet_cost : ℕ := 20
def rice_packets : ℕ := 2

-- Define the cost and quantity of soda Victoria bought.
def soda_cost : ℕ := 150
def soda_quantity : ℕ := 1

-- Define the remaining balance after shopping.
def remaining_balance : ℕ := 235

-- Define the cost of one packet of wheat flour.
def wheat_flour_packet_cost : ℕ := 25

-- Define the total amount spent on rice and soda.
def total_spent_on_rice_and_soda : ℕ :=
  (rice_packets * rice_packet_cost) + (soda_quantity * soda_cost)

-- Define the total amount spent on wheat flour.
def total_spent_on_wheat_flour : ℕ :=
  initial_amount - remaining_balance - total_spent_on_rice_and_soda

-- Define the expected number of wheat flour packets bought.
def wheat_flour_packets_expected : ℕ := 3

-- The statement we want to prove: the number of wheat flour packets bought is 3.
theorem wheat_flour_packets_correct : total_spent_on_wheat_flour / wheat_flour_packet_cost = wheat_flour_packets_expected :=
  sorry

end wheat_flour_packets_correct_l1217_121711


namespace largest_sphere_radius_l1217_121726

noncomputable def torus_inner_radius := 3
noncomputable def torus_outer_radius := 5
noncomputable def torus_center_circle := (4, 0, 1)
noncomputable def torus_radius := 1
noncomputable def torus_table_plane := 0

theorem largest_sphere_radius :
  ∀ (r : ℝ), 
  ∀ (O P : ℝ × ℝ × ℝ), 
  (P = (4, 0, 1)) → 
  (O = (0, 0, r)) → 
  4^2 + (r - 1)^2 = (r + 1)^2 → 
  r = 4 := 
by
  intros
  sorry

end largest_sphere_radius_l1217_121726


namespace simplify_expression_l1217_121729

theorem simplify_expression :
  8 * (18 / 5) * (-40 / 27) = - (128 / 3) := 
by
  sorry

end simplify_expression_l1217_121729


namespace cone_volume_ratio_l1217_121744

noncomputable def ratio_of_volumes (r h : ℝ) : ℝ :=
  let S1 := r^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 12
  let S2 := r^2 * (10 * Real.pi + 3 * Real.sqrt 3) / 12
  S1 / S2

theorem cone_volume_ratio (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  ratio_of_volumes r h = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3) :=
  sorry

end cone_volume_ratio_l1217_121744


namespace find_middle_number_l1217_121766

theorem find_middle_number (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 22) (h4 : x + z = 29) (h5 : y + z = 31) (h6 : x = 10) :
  y = 12 :=
sorry

end find_middle_number_l1217_121766


namespace parallel_lines_a_l1217_121793

theorem parallel_lines_a (a : ℝ) :
  ((∃ k : ℝ, (a + 2) / 6 = k ∧ (a + 3) / (2 * a - 1) = k) ∧ 
   ¬ ((-5 / -5) = ((a + 2) / 6)) ∧ ((a + 3) / (2 * a - 1) = (-5 / -5))) →
  a = -5 / 2 :=
by
  sorry

end parallel_lines_a_l1217_121793


namespace smallest_number_is_33_l1217_121734

theorem smallest_number_is_33 
  (x : ℕ) 
  (h1 : ∀ y z, y = 2 * x → z = 4 * x → (x + y + z) / 3 = 77) : 
  x = 33 :=
by
  sorry

end smallest_number_is_33_l1217_121734


namespace negation_of_proposition_l1217_121739

theorem negation_of_proposition :
  (¬ (∀ a b : ℤ, a = 0 → a * b = 0)) ↔ (∃ a b : ℤ, a = 0 ∧ a * b ≠ 0) :=
by
  sorry

end negation_of_proposition_l1217_121739


namespace sum_of_numbers_l1217_121730

theorem sum_of_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 8)
  (h4 : (a + b + c) / 3 = a + 12) (h5 : (a + b + c) / 3 = c - 20) :
  a + b + c = 48 :=
sorry

end sum_of_numbers_l1217_121730


namespace number_in_sequence_l1217_121713

theorem number_in_sequence : ∃ n : ℕ, n * (n + 2) = 99 :=
by
  sorry

end number_in_sequence_l1217_121713


namespace cos_60_eq_half_l1217_121731

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l1217_121731


namespace line_equation_passing_through_point_and_equal_intercepts_l1217_121719

theorem line_equation_passing_through_point_and_equal_intercepts :
    (∃ k: ℝ, ∀ x y: ℝ, (2, 5) = (x, k * x) ∨ x + y = 7) :=
by
  sorry

end line_equation_passing_through_point_and_equal_intercepts_l1217_121719


namespace integral_3x_plus_sin_x_l1217_121723

theorem integral_3x_plus_sin_x :
  ∫ x in (0 : ℝ)..(π / 2), (3 * x + Real.sin x) = (3 / 8) * π^2 + 1 :=
by
  sorry

end integral_3x_plus_sin_x_l1217_121723


namespace quarters_spent_l1217_121728

theorem quarters_spent (original : ℕ) (remaining : ℕ) (q : ℕ) 
  (h1 : original = 760) 
  (h2 : remaining = 342) 
  (h3 : q = original - remaining) : q = 418 := 
by
  sorry

end quarters_spent_l1217_121728


namespace exp_sum_l1217_121733

theorem exp_sum (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x + 3 * y) = 108 :=
sorry

end exp_sum_l1217_121733


namespace min_value_of_expression_l1217_121769

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  x^2 + 4 * y^2 + 2 * x * y ≥ 3 / 4 :=
sorry

end min_value_of_expression_l1217_121769


namespace no_common_points_lines_l1217_121772

theorem no_common_points_lines (m : ℝ) : 
    ¬∃ x y : ℝ, (x + m^2 * y + 6 = 0) ∧ ((m - 2) * x + 3 * m * y + 2 * m = 0) ↔ m = 0 ∨ m = -1 := 
by 
    sorry

end no_common_points_lines_l1217_121772


namespace no_x_satisfies_inequalities_l1217_121742

theorem no_x_satisfies_inequalities : ¬ ∃ x : ℝ, 4 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 5 :=
sorry

end no_x_satisfies_inequalities_l1217_121742


namespace find_m_l1217_121781

variables (x m : ℝ)

def equation (x m : ℝ) : Prop := 3 * x - 2 * m = 4

theorem find_m (h1 : equation 6 m) : m = 7 :=
by
  sorry

end find_m_l1217_121781


namespace point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l1217_121779

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y - m = 0

def inside_circle (x y m : ℝ) : Prop :=
  (x-1)^2 + (y+2)^2 < 5 + m

theorem point_A_inside_circle (m : ℝ) : -1 < m ∧ m < 4 ↔ inside_circle m (-2) m :=
sorry

def circle_equation_m_4 (x y : ℝ) : Prop :=
  circle_equation x y 4

def dist_square_to_point_H (x y : ℝ) : ℝ :=
  (x - 4)^2 + (y - 2)^2

theorem max_min_dist_square_on_circle (P : ℝ × ℝ) :
  circle_equation_m_4 P.1 P.2 →
  4 ≤ dist_square_to_point_H P.1 P.2 ∧ dist_square_to_point_H P.1 P.2 ≤ 64 :=
sorry

def line_equation (m x y : ℝ) : Prop :=
  y = x + m

theorem chord_through_origin (m : ℝ) :
  ∃ m : ℝ, line_equation m (1 : ℝ) (-2 : ℝ) ∧ 
  (m = -4 ∨ m = 1) :=
sorry

end point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l1217_121779


namespace pizza_eaten_after_six_trips_l1217_121799

theorem pizza_eaten_after_six_trips
  (initial_fraction: ℚ)
  (next_fraction : ℚ -> ℚ)
  (S: ℚ)
  (H0: initial_fraction = 1 / 4)
  (H1: ∀ (n: ℕ), next_fraction n = 1 / 2 ^ (n + 2))
  (H2: S = initial_fraction + (next_fraction 1) + (next_fraction 2) + (next_fraction 3) + (next_fraction 4) + (next_fraction 5)):
  S = 125 / 128 :=
by
  sorry

end pizza_eaten_after_six_trips_l1217_121799


namespace least_number_remainder_l1217_121756

theorem least_number_remainder (n : ℕ) (h : 20 ∣ (n - 5)) : n = 125 := sorry

end least_number_remainder_l1217_121756


namespace range_of_k_l1217_121780

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k+1)*x^2 + (k+3)*x + (2*k-8)) / ((2*k-1)*x^2 + (k+1)*x + (k-4))

theorem range_of_k 
  (k : ℝ) 
  (hk1 : k ≠ -1)
  (hk2 : (k+3)^2 - 4*(k+1)*(2*k-8) ≥ 0)
  (hk3 : (k+1)^2 - 4*(2*k-1)*(k-4) ≤ 0)
  (hk4 : (k+1)/(2*k-1) > 0) :
  k ∈ Set.Iio (-1) ∪ Set.Ioi (1 / 2) ∩ Set.Iic (41 / 7) := 
  sorry

end range_of_k_l1217_121780


namespace sum_of_squares_l1217_121770

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l1217_121770


namespace smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l1217_121785

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem smallest_positive_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem range_of_f : ∀ y : ℝ, y ∈ Set.range f ↔ -1 ≤ y ∧ y ≤ 1 := by
  sorry

theorem intervals_monotonically_increasing : 
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6) → 
  (f (x + Real.pi / 6) - f x) ≥ 0 := by
  sorry

end smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l1217_121785


namespace compute_m_n_sum_l1217_121786

theorem compute_m_n_sum :
  let AB := 10
  let BC := 15
  let height := 30
  let volume_ratio := 9
  let smaller_base_AB := AB / 3
  let smaller_base_BC := BC / 3
  let diagonal_AC := Real.sqrt (AB^2 + BC^2)
  let smaller_diagonal_A'C' := Real.sqrt ((smaller_base_AB)^2 + (smaller_base_BC)^2)
  let y_length := 145 / 9   -- derived from geometric considerations
  let YU := 20 + y_length
  let m := 325
  let n := 9
  YU = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 334 :=
  by
  sorry

end compute_m_n_sum_l1217_121786


namespace daily_expenditure_l1217_121783

theorem daily_expenditure (total_spent : ℕ) (days_in_june : ℕ) (equal_consumption : Prop) :
  total_spent = 372 ∧ days_in_june = 30 ∧ equal_consumption → (372 / 30) = 12.40 := by
  sorry

end daily_expenditure_l1217_121783


namespace days_to_complete_work_l1217_121755

theorem days_to_complete_work {D : ℝ} (h1 : D > 0)
  (h2 : (1 / D) + (2 / D) = 0.3) :
  D = 10 :=
sorry

end days_to_complete_work_l1217_121755


namespace games_given_away_l1217_121701

/-- Gwen had ninety-eight DS games. 
    After she gave some to her friends she had ninety-one left.
    Prove that she gave away 7 DS games. -/
theorem games_given_away (original_games : ℕ) (games_left : ℕ) (games_given : ℕ) 
  (h1 : original_games = 98) 
  (h2 : games_left = 91) 
  (h3 : games_given = original_games - games_left) : 
  games_given = 7 :=
sorry

end games_given_away_l1217_121701


namespace leak_empty_tank_time_l1217_121738

-- Definitions based on given conditions
def rate_A := 1 / 2 -- Rate of Pipe A (1 tank per 2 hours)
def rate_A_plus_L := 2 / 5 -- Combined rate of Pipe A and leak

-- Theorem states the time leak takes to empty full tank is 10 hours
theorem leak_empty_tank_time : 1 / (rate_A - rate_A_plus_L) = 10 :=
by
  -- Proof steps would go here
  sorry

end leak_empty_tank_time_l1217_121738


namespace ways_to_distribute_balls_l1217_121798

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l1217_121798


namespace rectangle_width_l1217_121727

theorem rectangle_width (P l: ℕ) (hP : P = 50) (hl : l = 13) : 
  ∃ w : ℕ, 2 * l + 2 * w = P ∧ w = 12 := 
by
  sorry

end rectangle_width_l1217_121727


namespace largest_of_seven_consecutive_odd_numbers_l1217_121787

theorem largest_of_seven_consecutive_odd_numbers (a b c d e f g : ℤ) 
  (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) 
  (h5: e % 2 = 1) (h6: f % 2 = 1) (h7: g % 2 = 1)
  (h8 : a + b + c + d + e + f + g = 105)
  (h9 : b = a + 2) (h10 : c = a + 4) (h11 : d = a + 6)
  (h12 : e = a + 8) (h13 : f = a + 10) (h14 : g = a + 12) :
  g = 21 :=
by 
  sorry

end largest_of_seven_consecutive_odd_numbers_l1217_121787


namespace ratio_of_potatoes_l1217_121796

def total_potatoes : ℕ := 24
def number_of_people : ℕ := 3
def potatoes_per_person : ℕ := 8
def total_each_person : ℕ := potatoes_per_person * number_of_people

theorem ratio_of_potatoes :
  total_potatoes = total_each_person → (potatoes_per_person : ℚ) / (potatoes_per_person : ℚ) = 1 :=
by
  sorry

end ratio_of_potatoes_l1217_121796


namespace purely_imaginary_l1217_121764

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end purely_imaginary_l1217_121764


namespace bowling_ball_weight_l1217_121762

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = 20 := by
  sorry

end bowling_ball_weight_l1217_121762


namespace sum_of_ages_l1217_121710

-- Defining the ages of Nathan and his twin sisters.
variables (n t : ℕ)

-- Nathan has two twin younger sisters, and the product of their ages equals 72.
def valid_ages (n t : ℕ) : Prop := t < n ∧ n * t * t = 72

-- Prove that the sum of the ages of Nathan and his twin sisters is 14.
theorem sum_of_ages (n t : ℕ) (h : valid_ages n t) : 2 * t + n = 14 :=
sorry

end sum_of_ages_l1217_121710


namespace infinite_divisibility_1986_l1217_121714

theorem infinite_divisibility_1986 :
  ∃ (a : ℕ → ℕ), a 1 = 39 ∧ a 2 = 45 ∧ (∀ n, a (n+2) = a (n+1) ^ 2 - a n) ∧
  ∀ N, ∃ n > N, 1986 ∣ a n :=
sorry

end infinite_divisibility_1986_l1217_121714


namespace sin_double_angle_second_quadrant_l1217_121784

theorem sin_double_angle_second_quadrant (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo (π / 2) π) :
    Real.sin (2 * α) = -24 / 25 := by
  sorry

end sin_double_angle_second_quadrant_l1217_121784


namespace average_age_of_students_l1217_121771

theorem average_age_of_students (A : ℝ) (h1 : ∀ n : ℝ, n = 20 → A + 1 = n) (h2 : ∀ k : ℝ, k = 40 → 19 * A + k = 20 * (A + 1)) : A = 20 :=
by
  sorry

end average_age_of_students_l1217_121771


namespace printer_Y_time_l1217_121753

theorem printer_Y_time (T_y : ℝ) : 
    (12 * (1 / (1 / T_y + 1 / 20)) = 1.8) → T_y = 10 := 
by 
sorry

end printer_Y_time_l1217_121753


namespace cos4_minus_sin4_15_eq_sqrt3_div2_l1217_121775

theorem cos4_minus_sin4_15_eq_sqrt3_div2 :
  (Real.cos 15)^4 - (Real.sin 15)^4 = Real.sqrt 3 / 2 :=
sorry

end cos4_minus_sin4_15_eq_sqrt3_div2_l1217_121775


namespace expression_in_terms_of_p_q_l1217_121741

-- Define the roots and the polynomials conditions
variable (α β γ δ : ℝ)
variable (p q : ℝ)

-- The conditions of the problem
axiom roots_poly1 : α * β = 1 ∧ α + β = -p
axiom roots_poly2 : γ * δ = 1 ∧ γ + δ = -q

theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end expression_in_terms_of_p_q_l1217_121741


namespace line_equation_l1217_121736

theorem line_equation {L : ℝ → ℝ → Prop} (h1 : L (-3) (-2)) 
  (h2 : ∃ a : ℝ, a ≠ 0 ∧ (L a 0 ∧ L 0 a)) :
  (∀ x y, L x y ↔ 2 * x - 3 * y = 0) ∨ (∀ x y, L x y ↔ x + y + 5 = 0) :=
by 
  sorry

end line_equation_l1217_121736


namespace f_positive_for_specific_a_l1217_121740

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x * Real.log x

theorem f_positive_for_specific_a (x : ℝ) (h : x > 0) :
  f x (Real.exp 3 / 4) > 0 := sorry

end f_positive_for_specific_a_l1217_121740


namespace fewest_cookies_l1217_121716

theorem fewest_cookies
  (r a s d1 d2 : ℝ)
  (hr_pos : r > 0)
  (ha_pos : a > 0)
  (hs_pos : s > 0)
  (hd1_pos : d1 > 0)
  (hd2_pos : d2 > 0)
  (h_Alice_cookies : 15 = 15)
  (h_same_dough : true) :
  15 < (15 * (Real.pi * r^2)) / (a^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((3 * Real.sqrt 3 / 2) * s^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((1 / 2) * d1 * d2) :=
by
  sorry

end fewest_cookies_l1217_121716
