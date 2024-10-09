import Mathlib

namespace distance_between_vertices_l214_21451

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end distance_between_vertices_l214_21451


namespace probability_of_observing_change_l214_21487

noncomputable def traffic_light_cycle := 45 + 5 + 45
noncomputable def observable_duration := 5 + 5 + 5
noncomputable def probability_observe_change := observable_duration / (traffic_light_cycle : ℝ)

theorem probability_of_observing_change :
  probability_observe_change = (3 / 19 : ℝ) :=
  by sorry

end probability_of_observing_change_l214_21487


namespace repeating_decimal_product_l214_21424

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l214_21424


namespace find_ab_sum_l214_21402

theorem find_ab_sum 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) 
  : a + b = -14 := by
  sorry

end find_ab_sum_l214_21402


namespace bullet_speed_difference_l214_21427

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l214_21427


namespace specificTriangle_perimeter_l214_21426

-- Assume a type to represent triangle sides
structure IsoscelesTriangle (a b : ℕ) : Prop :=
  (equal_sides : a = b ∨ a + b > max a b)

-- Define the condition where we have specific sides
def specificTriangle : Prop :=
  IsoscelesTriangle 5 2

-- Prove that given the specific sides, the perimeter is 12
theorem specificTriangle_perimeter : specificTriangle → 5 + 5 + 2 = 12 :=
by
  intro h
  cases h
  sorry

end specificTriangle_perimeter_l214_21426


namespace trig_identity_equiv_l214_21486

theorem trig_identity_equiv (α : ℝ) (h : Real.sin (Real.pi - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 :=
by
  sorry

end trig_identity_equiv_l214_21486


namespace set_of_a_where_A_subset_B_l214_21492

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l214_21492


namespace eggs_in_two_boxes_l214_21430

theorem eggs_in_two_boxes (eggs_per_box : ℕ) (number_of_boxes : ℕ) (total_eggs : ℕ) 
  (h1 : eggs_per_box = 3)
  (h2 : number_of_boxes = 2) :
  total_eggs = eggs_per_box * number_of_boxes :=
sorry

end eggs_in_two_boxes_l214_21430


namespace max_yes_answers_100_l214_21414

-- Define the maximum number of "Yes" answers that could be given in a lineup of n people
def maxYesAnswers (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 2)

theorem max_yes_answers_100 : maxYesAnswers 100 = 99 :=
  by sorry

end max_yes_answers_100_l214_21414


namespace milkshake_cost_is_five_l214_21418

def initial_amount : ℝ := 132
def hamburger_cost : ℝ := 4
def num_hamburgers : ℕ := 8
def num_milkshakes : ℕ := 6
def amount_left : ℝ := 70

theorem milkshake_cost_is_five (M : ℝ) (h : initial_amount - (num_hamburgers * hamburger_cost + num_milkshakes * M) = amount_left) : 
  M = 5 :=
by
  sorry

end milkshake_cost_is_five_l214_21418


namespace fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l214_21463

open Complex

def inFourthQuadrant (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) > 0 ∧ (m^2 + 3*m - 28) < 0

def onNegativeHalfXAxis (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) < 0 ∧ (m^2 + 3*m - 28) = 0

def inUpperHalfPlaneIncludingRealAxis (m : ℝ) : Prop :=
  (m^2 + 3*m - 28) ≥ 0

theorem fourth_quadrant_for_m (m : ℝ) :
  (-7 < m ∧ m < 3) ↔ inFourthQuadrant m := 
sorry

theorem negative_half_x_axis_for_m (m : ℝ) :
  (m = 4) ↔ onNegativeHalfXAxis m :=
sorry

theorem upper_half_plane_for_m (m : ℝ) :
  (m ≤ -7 ∨ m ≥ 4) ↔ inUpperHalfPlaneIncludingRealAxis m :=
sorry

end fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l214_21463


namespace simplify_fraction_l214_21413

theorem simplify_fraction :
  (1 / (1 / (1 / 2) ^ 1 + 1 / (1 / 2) ^ 2 + 1 / (1 / 2) ^ 3)) = (1 / 14) :=
by 
  sorry

end simplify_fraction_l214_21413


namespace kangaroo_fiber_intake_l214_21452

-- Suppose kangaroos absorb only 30% of the fiber they eat
def absorption_rate : ℝ := 0.30

-- If a kangaroo absorbed 15 ounces of fiber in one day
def absorbed_fiber : ℝ := 15.0

-- Prove the kangaroo ate 50 ounces of fiber that day
theorem kangaroo_fiber_intake (x : ℝ) (hx : absorption_rate * x = absorbed_fiber) : x = 50 :=
by
  sorry

end kangaroo_fiber_intake_l214_21452


namespace line_length_400_l214_21489

noncomputable def length_of_line (speed_march_kmh speed_run_kmh total_time_min: ℝ) : ℝ :=
  let speed_march_mpm := (speed_march_kmh * 1000) / 60
  let speed_run_mpm := (speed_run_kmh * 1000) / 60
  let len_eq := 1 / (speed_run_mpm - speed_march_mpm) + 1 / (speed_run_mpm + speed_march_mpm)
  (total_time_min * 200 * len_eq) * 400 / len_eq

theorem line_length_400 :
  length_of_line 8 12 7.2 = 400 := by
  sorry

end line_length_400_l214_21489


namespace class_average_weight_l214_21436

theorem class_average_weight (n_A n_B : ℕ) (w_A w_B : ℝ) (h1 : n_A = 50) (h2 : n_B = 40) (h3 : w_A = 50) (h4 : w_B = 70) :
  (n_A * w_A + n_B * w_B) / (n_A + n_B) = 58.89 :=
by
  sorry

end class_average_weight_l214_21436


namespace inequality_proof_l214_21405

noncomputable def sum_expression (a b c : ℝ) : ℝ :=
  (1 / (b * c + a + 1 / a)) + (1 / (c * a + b + 1 / b)) + (1 / (a * b + c + 1 / c))

theorem inequality_proof (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  sum_expression a b c ≤ 27 / 31 :=
by
  sorry

end inequality_proof_l214_21405


namespace horses_lcm_l214_21408

theorem horses_lcm :
  let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
  let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  let time_T := lcm_six
  lcm_six = 420 ∧ (Nat.digits 10 time_T).sum = 6 := by
    let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
    let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
    let time_T := lcm_six
    have h1 : lcm_six = 420 := sorry
    have h2 : (Nat.digits 10 time_T).sum = 6 := sorry
    exact ⟨h1, h2⟩

end horses_lcm_l214_21408


namespace train1_speed_l214_21448

noncomputable def total_distance_in_kilometers : ℝ :=
  (630 + 100 + 200) / 1000

noncomputable def time_in_hours : ℝ :=
  13.998880089592832 / 3600

noncomputable def relative_speed : ℝ :=
  total_distance_in_kilometers / time_in_hours

noncomputable def speed_of_train2 : ℝ :=
  72

noncomputable def speed_of_train1 : ℝ :=
  relative_speed - speed_of_train2

theorem train1_speed : speed_of_train1 = 167.076 := by 
  sorry

end train1_speed_l214_21448


namespace find_f_values_find_f_expression_l214_21401

variable (f : ℕ+ → ℤ)

-- Conditions in Lean
def is_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ {m n : ℕ+}, m < n → f m < f n

axiom h1 : is_increasing f
axiom h2 : f 4 = 5
axiom h3 : ∀ n : ℕ+, ∃ k : ℕ, f n = k
axiom h4 : ∀ m n : ℕ+, f m * f n = f (m * n) + f (m + n - 1)

-- Proof in Lean 4
theorem find_f_values : f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4 :=
by
  sorry

theorem find_f_expression : ∀ n : ℕ+, f n = n + 1 :=
by
  sorry

end find_f_values_find_f_expression_l214_21401


namespace list_price_correct_l214_21421

noncomputable def list_price_satisfied : Prop :=
∃ x : ℝ, 0.25 * (x - 25) + 0.05 * (x - 5) = 0.15 * (x - 15) ∧ x = 28.33

theorem list_price_correct : list_price_satisfied :=
sorry

end list_price_correct_l214_21421


namespace computation_equal_l214_21470

theorem computation_equal (a b c d : ℕ) (inv : ℚ → ℚ) (mul : ℚ → ℕ → ℚ) : 
  a = 3 → b = 1 → c = 6 → d = 2 → 
  inv ((a^b - d + c^2 + b) : ℚ) * 6 = (3 / 19) := by
  intros ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end computation_equal_l214_21470


namespace combinations_sol_eq_l214_21497

theorem combinations_sol_eq (x : ℕ) (h : Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 := sorry

end combinations_sol_eq_l214_21497


namespace not_divisible_by_10100_l214_21454

theorem not_divisible_by_10100 (n : ℕ) : (3^n + 1) % 10100 ≠ 0 := 
by 
  sorry

end not_divisible_by_10100_l214_21454


namespace train_pass_bridge_time_l214_21415

noncomputable def trainLength : ℝ := 360
noncomputable def trainSpeedKMH : ℝ := 45
noncomputable def bridgeLength : ℝ := 160
noncomputable def totalDistance : ℝ := trainLength + bridgeLength
noncomputable def trainSpeedMS : ℝ := trainSpeedKMH * (1000 / 3600)
noncomputable def timeToPassBridge : ℝ := totalDistance / trainSpeedMS

theorem train_pass_bridge_time : timeToPassBridge = 41.6 := sorry

end train_pass_bridge_time_l214_21415


namespace smallest_number_of_integers_l214_21449

theorem smallest_number_of_integers (a b n : ℕ) 
  (h_avg_original : 89 * n = 73 * a + 111 * b) 
  (h_group_sum : a + b = n)
  (h_ratio : 8 * a = 11 * b) : 
  n = 19 :=
sorry

end smallest_number_of_integers_l214_21449


namespace clocks_sync_again_in_lcm_days_l214_21423

-- Defining the given conditions based on the problem statement.

-- Arthur's clock gains 15 minutes per day, taking 48 days to gain 12 hours (720 minutes).
def arthur_days : ℕ := 48

-- Oleg's clock gains 12 minutes per day, taking 60 days to gain 12 hours (720 minutes).
def oleg_days : ℕ := 60

-- The problem asks to prove that the situation repeats after 240 days, which is the LCM of 48 and 60.
theorem clocks_sync_again_in_lcm_days : Nat.lcm arthur_days oleg_days = 240 := 
by 
  sorry

end clocks_sync_again_in_lcm_days_l214_21423


namespace total_workers_is_22_l214_21429

-- Define constants and variables based on conditions
def avg_salary_all : ℝ := 850
def avg_salary_technicians : ℝ := 1000
def avg_salary_rest : ℝ := 780
def num_technicians : ℝ := 7

-- Define the necessary proof statement
theorem total_workers_is_22
  (W : ℝ)
  (h1 : W * avg_salary_all = num_technicians * avg_salary_technicians + (W - num_technicians) * avg_salary_rest) :
  W = 22 :=
by
  sorry

end total_workers_is_22_l214_21429


namespace digits_difference_l214_21481

/-- Given a two-digit number represented as 10X + Y and the number obtained by interchanging its digits as 10Y + X,
    if the difference between the original number and the interchanged number is 81, 
    then the difference between the tens digit X and the units digit Y is 9. -/
theorem digits_difference (X Y : ℕ) (h : (10 * X + Y) - (10 * Y + X) = 81) : X - Y = 9 :=
by
  sorry

end digits_difference_l214_21481


namespace geometric_series_arithmetic_sequence_l214_21458

noncomputable def geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_series_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_seq_ratio a q)
  (h_pos : ∀ n, a n > 0)
  (h_arith : a 1 = (a 0 + 2 * a 1) / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 :=
sorry

end geometric_series_arithmetic_sequence_l214_21458


namespace angle_problem_l214_21479

theorem angle_problem (θ : ℝ) (h1 : 90 - θ = 0.4 * (180 - θ)) (h2 : 180 - θ = 2 * θ) : θ = 30 :=
by
  sorry

end angle_problem_l214_21479


namespace james_spent_6_dollars_l214_21428

-- Define the cost of items
def cost_milk : ℕ := 3
def cost_bananas : ℕ := 2

-- Define the sales tax rate as a decimal
def sales_tax_rate : ℚ := 0.20

-- Define the total cost before tax
def total_cost_before_tax : ℕ := cost_milk + cost_bananas

-- Define the sales tax amount
def sales_tax_amount : ℚ := sales_tax_rate * total_cost_before_tax

-- Define the total amount spent
def total_amount_spent : ℚ := total_cost_before_tax + sales_tax_amount

-- The proof statement
theorem james_spent_6_dollars : total_amount_spent = 6 := by
  sorry

end james_spent_6_dollars_l214_21428


namespace men_absent_l214_21406

/-- 
A group of men decided to do a work in 20 days, but some of them became absent. 
The rest of the group did the work in 40 days. The original number of men was 20. 
Prove that 10 men became absent. 
--/
theorem men_absent 
    (original_men : ℕ) (absent_men : ℕ) (planned_days : ℕ) (actual_days : ℕ)
    (h1 : original_men = 20) (h2 : planned_days = 20) (h3 : actual_days = 40)
    (h_work : original_men * planned_days = (original_men - absent_men) * actual_days) : 
    absent_men = 10 :=
    by 
    rw [h1, h2, h3] at h_work
    -- Proceed to manually solve the equation, but here we add sorry
    sorry

end men_absent_l214_21406


namespace nine_cubed_expansion_l214_21404

theorem nine_cubed_expansion : 9^3 + 3 * 9^2 + 3 * 9 + 1 = 1000 := 
by 
  sorry

end nine_cubed_expansion_l214_21404


namespace stuart_initially_had_20_l214_21412

variable (B T S : ℕ) -- Initial number of marbles for Betty, Tom, and Susan
variable (S_after : ℕ) -- Number of marbles Stuart has after receiving from Betty

-- Given conditions
axiom betty_initially : B = 150
axiom tom_initially : T = 30
axiom susan_initially : S = 20

axiom betty_to_tom : (0.20 : ℚ) * B = 30
axiom betty_to_susan : (0.10 : ℚ) * B = 15
axiom betty_to_stuart : (0.40 : ℚ) * B = 60
axiom stuart_after_receiving : S_after = 80

-- Theorem to prove Stuart initially had 20 marbles
theorem stuart_initially_had_20 : ∃ S_initial : ℕ, S_after - 60 = S_initial ∧ S_initial = 20 :=
by {
  sorry
}

end stuart_initially_had_20_l214_21412


namespace problem1_problem2_l214_21432

variable {a b : ℝ}

-- Proof problem 1
-- Goal: (1)(2a^(2/3)b^(1/2))(-6a^(1/2)b^(1/3)) / (-3a^(1/6)b^(5/6)) = -12a
theorem problem1 (h1 : 0 < a) (h2 : 0 < b) : 
  (1 : ℝ) * (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = -12 * a := 
sorry

-- Proof problem 2
-- Goal: 2(log(sqrt(2)))^2 + log(sqrt(2)) * log(5) + sqrt((log(sqrt(2)))^2 - log(2) + 1) = 1 + (1 / 2) * log(5)
theorem problem2 : 
  2 * (Real.log (Real.sqrt 2))^2 + (Real.log (Real.sqrt 2)) * (Real.log 5) + 
  Real.sqrt ((Real.log (Real.sqrt 2))^2 - Real.log 2 + 1) = 
  1 + 0.5 * (Real.log 5) := 
sorry

end problem1_problem2_l214_21432


namespace dan_time_second_hour_tshirts_l214_21453

-- Definition of conditions
def t_shirts_in_first_hour (rate1 : ℕ) (time : ℕ) : ℕ := time / rate1
def total_t_shirts (hour1_ts hour2_ts : ℕ) : ℕ := hour1_ts + hour2_ts
def time_per_t_shirt_in_second_hour (time : ℕ) (hour2_ts : ℕ) : ℕ := time / hour2_ts

-- Main theorem statement (without proof)
theorem dan_time_second_hour_tshirts
  (rate1 : ℕ) (hour1_time : ℕ) (total_ts : ℕ) (hour_time : ℕ)
  (hour1_ts := t_shirts_in_first_hour rate1 hour1_time)
  (hour2_ts := total_ts - hour1_ts) :
  rate1 = 12 → 
  hour1_time = 60 → 
  total_ts = 15 → 
  hour_time = 60 →
  time_per_t_shirt_in_second_hour hour_time hour2_ts = 6 :=
by
  intros rate1_eq hour1_time_eq total_ts_eq hour_time_eq
  sorry

end dan_time_second_hour_tshirts_l214_21453


namespace area_to_be_painted_l214_21434

variable (h_wall : ℕ) (l_wall : ℕ)
variable (h_window : ℕ) (l_window : ℕ)
variable (h_door : ℕ) (l_door : ℕ)

theorem area_to_be_painted :
  ∀ (h_wall : ℕ) (l_wall : ℕ) (h_window : ℕ) (l_window : ℕ) (h_door : ℕ) (l_door : ℕ),
  h_wall = 10 → l_wall = 15 →
  h_window = 3 → l_window = 5 →
  h_door = 2 → l_door = 3 →
  (h_wall * l_wall) - ((h_window * l_window) + (h_door * l_door)) = 129 :=
by
  intros
  sorry

end area_to_be_painted_l214_21434


namespace first_player_wins_l214_21483

-- Define the game state and requirements
inductive Player
| first : Player
| second : Player

-- Game state consists of a number of stones and whose turn it is
structure GameState where
  stones : Nat
  player : Player

-- Define a simple transition for the game
def take_stones (s : GameState) (n : Nat) : GameState :=
  { s with stones := s.stones - n, player := Player.second }

-- Determine if a player can take n stones
def can_take (s : GameState) (n : Nat) : Prop :=
  n >= 1 ∧ n <= 4 ∧ n <= s.stones

-- Define victory condition
def wins (s : GameState) : Prop :=
  s.stones = 0 ∧ s.player = Player.second

-- Prove that if the first player starts with 18 stones and picks 3 stones initially,
-- they can ensure victory
theorem first_player_wins :
  ∀ (s : GameState),
    s.stones = 18 ∧ s.player = Player.first →
    can_take s 3 →
    wins (take_stones s 3)
:= by
  sorry

end first_player_wins_l214_21483


namespace mary_marbles_l214_21441

theorem mary_marbles (total_marbles joan_marbles mary_marbles : ℕ) 
  (h1 : total_marbles = 12) 
  (h2 : joan_marbles = 3) 
  (h3 : total_marbles = joan_marbles + mary_marbles) : 
  mary_marbles = 9 := 
by
  rw [h1, h2, add_comm] at h3
  linarith

end mary_marbles_l214_21441


namespace initial_money_jennifer_l214_21450

theorem initial_money_jennifer (M : ℝ) (h1 : (1/5) * M + (1/6) * M + (1/2) * M + 12 = M) : M = 90 :=
sorry

end initial_money_jennifer_l214_21450


namespace line_through_points_l214_21478

variable (A1 B1 A2 B2 : ℝ)

def line1 : Prop := -7 * A1 + 9 * B1 = 1
def line2 : Prop := -7 * A2 + 9 * B2 = 1

theorem line_through_points (h1 : line1 A1 B1) (h2 : line1 A2 B2) :
  ∃ (k : ℝ), (∀ (x y : ℝ), y - B1 = k * (x - A1)) ∧ (-7 * (x : ℝ) + 9 * y = 1) := 
by sorry

end line_through_points_l214_21478


namespace number_of_blue_socks_l214_21491

theorem number_of_blue_socks (x : ℕ) (h : ((6 + x ^ 2 - x) / ((6 + x) * (5 + x)) = 1/5)) : x = 4 := 
sorry

end number_of_blue_socks_l214_21491


namespace minimum_value_expression_l214_21465

theorem minimum_value_expression (a b : ℝ) (h : a * b > 0) : 
  ∃ m : ℝ, (∀ x y : ℝ, x * y > 0 → (4 * y / x + (x - 2 * y) / y) ≥ m) ∧ m = 2 :=
by
  sorry

end minimum_value_expression_l214_21465


namespace log_condition_necessary_not_sufficient_l214_21493

noncomputable def base_of_natural_logarithm := Real.exp 1

variable (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < b) (h4 : b ≠ 1)

theorem log_condition_necessary_not_sufficient (h : 0 < a ∧ a < b ∧ b < 1) :
  (Real.log 2 / Real.log a > Real.log base_of_natural_logarithm / Real.log b) :=
sorry

end log_condition_necessary_not_sufficient_l214_21493


namespace determine_f_l214_21476

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 := by
  sorry

end determine_f_l214_21476


namespace loss_per_metre_proof_l214_21482

-- Define the given conditions
def cost_price_per_metre : ℕ := 66
def quantity_sold : ℕ := 200
def total_selling_price : ℕ := 12000

-- Define total cost price based on cost price per metre and quantity sold
def total_cost_price : ℕ := cost_price_per_metre * quantity_sold

-- Define total loss based on total cost price and total selling price
def total_loss : ℕ := total_cost_price - total_selling_price

-- Define loss per metre
def loss_per_metre : ℕ := total_loss / quantity_sold

-- The theorem we need to prove:
theorem loss_per_metre_proof : loss_per_metre = 6 :=
  by
    sorry

end loss_per_metre_proof_l214_21482


namespace number_of_cars_lifted_l214_21420

def total_cars_lifted : ℕ := 6

theorem number_of_cars_lifted : total_cars_lifted = 6 := by
  sorry

end number_of_cars_lifted_l214_21420


namespace number_of_windows_davids_house_l214_21437

theorem number_of_windows_davids_house
  (windows_per_minute : ℕ → ℕ)
  (h1 : ∀ t, windows_per_minute t = (4 * t) / 10)
  (h2 : windows_per_minute 160 = w)
  : w = 64 :=
by
  sorry

end number_of_windows_davids_house_l214_21437


namespace quadratic_roots_l214_21484

theorem quadratic_roots (x : ℝ) (h : x^2 - 1 = 3) : x = 2 ∨ x = -2 :=
by
  sorry

end quadratic_roots_l214_21484


namespace checkerboard_pattern_exists_l214_21490

-- Definitions for the given conditions
def is_black_white_board (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → (board (i, j) = true ∨ board (i, j) = false)

def boundary_cells_black (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i, (i < n → (board (i, 0) = true ∧ board (i, n-1) = true ∧ 
                  board (0, i) = true ∧ board (n-1, i) = true))

def no_monochromatic_square (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n-1 ∧ j < n-1 → ¬(board (i, j) = board (i+1, j) ∧ 
                               board (i, j) = board (i, j+1) ∧ 
                               board (i, j) = board (i+1, j+1))

def exists_checkerboard_2x2 (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∃ i j, i < n-1 ∧ j < n-1 ∧ 
         (board (i, j) ≠ board (i+1, j) ∧ board (i, j) ≠ board (i, j+1) ∧ 
          board (i+1, j) ≠ board (i+1, j+1) ∧ board (i, j+1) ≠ board (i+1, j+1))

-- The theorem statement
theorem checkerboard_pattern_exists (board : ℕ × ℕ → Prop) (n : ℕ) 
  (coloring : is_black_white_board board n)
  (boundary_black : boundary_cells_black board n)
  (no_mono_2x2 : no_monochromatic_square board n) : 
  exists_checkerboard_2x2 board n :=
by
  sorry

end checkerboard_pattern_exists_l214_21490


namespace value_of_x_squared_plus_reciprocal_squared_l214_21480

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : x^4 + (1 / x^4) = 23) :
  x^2 + (1 / x^2) = 5 := by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l214_21480


namespace zero_of_f_inequality_l214_21460

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - Real.log (x^3 + 1)

variable (a b c x : ℝ)
variable (h : 0 < a ∧ a < b ∧ b < c)
variable (hx : f x = 0)
variable (h₀ : f a * f b * f c < 0)

theorem zero_of_f_inequality :
  ¬ (x > c) :=
by 
  sorry

end zero_of_f_inequality_l214_21460


namespace complex_powers_l214_21411

theorem complex_powers (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^(23 : ℕ) + i^(58 : ℕ) = -1 - i :=
by sorry

end complex_powers_l214_21411


namespace arithmetic_mean_twice_y_l214_21417

theorem arithmetic_mean_twice_y (y x : ℝ) (h1 : (8 + y + 24 + 6 + x) / 5 = 12) (h2 : x = 2 * y) :
  y = 22 / 3 ∧ x = 44 / 3 :=
by
  sorry

end arithmetic_mean_twice_y_l214_21417


namespace correct_statements_l214_21499

variable (P Q : Prop)

-- Define statements
def is_neg_false_if_orig_true := (P → ¬P) = False
def is_converse_not_nec_true_if_orig_true := (P → Q) → ¬(Q → P)
def is_neg_true_if_converse_true := (Q → P) → (¬P → ¬Q)
def is_neg_true_if_contrapositive_true := (¬Q → ¬P) → (¬P → False)

-- Main proposition
theorem correct_statements : 
  is_converse_not_nec_true_if_orig_true P Q ∧ 
  is_neg_true_if_converse_true P Q :=
by
  sorry

end correct_statements_l214_21499


namespace sugar_at_home_l214_21467

-- Definitions based on conditions
def bags_of_sugar := 2
def cups_per_bag := 6
def cups_for_batter_per_12_cupcakes := 1
def cups_for_frosting_per_12_cupcakes := 2
def dozens_of_cupcakes := 5

-- Calculation of total sugar needed and bought, in terms of definitions
def total_cupcakes := dozens_of_cupcakes * 12
def total_sugar_needed_for_batter := (total_cupcakes / 12) * cups_for_batter_per_12_cupcakes
def total_sugar_needed_for_frosting := dozens_of_cupcakes * cups_for_frosting_per_12_cupcakes
def total_sugar_needed := total_sugar_needed_for_batter + total_sugar_needed_for_frosting
def total_sugar_bought := bags_of_sugar * cups_per_bag

-- The statement to be proven in Lean
theorem sugar_at_home : total_sugar_needed - total_sugar_bought = 3 := by
  sorry

end sugar_at_home_l214_21467


namespace water_consumption_l214_21407

theorem water_consumption (x y : ℝ)
  (h1 : 120 + 20 * x = 3200000 * y)
  (h2 : 120 + 15 * x = 3000000 * y) :
  x = 200 ∧ y = 50 :=
by
  sorry

end water_consumption_l214_21407


namespace comprehensive_score_correct_l214_21477

-- Conditions
def theoreticalWeight : ℝ := 0.20
def designWeight : ℝ := 0.50
def presentationWeight : ℝ := 0.30

def theoreticalScore : ℕ := 95
def designScore : ℕ := 88
def presentationScore : ℕ := 90

-- Calculate comprehensive score
def comprehensiveScore : ℝ :=
  theoreticalScore * theoreticalWeight +
  designScore * designWeight +
  presentationScore * presentationWeight

-- Lean statement to prove the comprehensive score using the conditions
theorem comprehensive_score_correct :
  comprehensiveScore = 90 := 
  sorry

end comprehensive_score_correct_l214_21477


namespace range_of_m_l214_21462

theorem range_of_m (x m : ℝ) (h1 : (2 * x + m) / (x - 1) = 1) (h2 : x ≥ 0) : m ≤ -1 ∧ m ≠ -2 :=
sorry

end range_of_m_l214_21462


namespace choose_5_with_exactly_one_twin_l214_21439

theorem choose_5_with_exactly_one_twin :
  let total_players := 12
  let twins := 2
  let players_to_choose := 5
  let remaining_players_after_one_twin := total_players - twins + 1 -- 11 players to choose from
  (2 * Nat.choose remaining_players_after_one_twin (players_to_choose - 1)) = 420 := 
by
  sorry

end choose_5_with_exactly_one_twin_l214_21439


namespace find_natural_n_l214_21473

theorem find_natural_n (n x y k : ℕ) (h_rel_prime : Nat.gcd x y = 1) (h_k_gt_one : k > 1) (h_eq : 3^n = x^k + y^k) :
  n = 2 := by
  sorry

end find_natural_n_l214_21473


namespace minimum_erasures_correct_l214_21495

open Nat List

-- define a function that checks if a number represented as a list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- the given problem statement
def given_number := [1, 2, 3, 2, 3, 3, 1, 4]

-- function to find the minimum erasures to make a list a palindrome
noncomputable def min_erasures_to_palindrome (l : List ℕ) : ℕ :=
  sorry -- function implementation skipped

-- the main theorem statement
theorem minimum_erasures_correct : min_erasures_to_palindrome given_number = 3 :=
  sorry

end minimum_erasures_correct_l214_21495


namespace avg_salary_feb_mar_apr_may_l214_21498

def avg_salary_4_months : ℝ := 8000
def salary_jan : ℝ := 3700
def salary_may : ℝ := 6500
def total_salary_4_months := 4 * avg_salary_4_months
def total_salary_feb_mar_apr := total_salary_4_months - salary_jan
def total_salary_feb_mar_apr_may := total_salary_feb_mar_apr + salary_may

theorem avg_salary_feb_mar_apr_may : total_salary_feb_mar_apr_may / 4 = 8700 := by
  sorry

end avg_salary_feb_mar_apr_may_l214_21498


namespace vertical_asymptote_x_value_l214_21440

theorem vertical_asymptote_x_value (x : ℝ) : 
  4 * x - 6 = 0 ↔ x = 3 / 2 :=
by
  sorry

end vertical_asymptote_x_value_l214_21440


namespace option_B_coplanar_l214_21456

-- Define the three vectors in Option B.
def a : ℝ × ℝ × ℝ := (1, 2, -3)
def b : ℝ × ℝ × ℝ := (-2, -4, 6)
def c : ℝ × ℝ × ℝ := (1, 0, 5)

-- Define the coplanarity condition for vectors a, b, and c.
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = k • a

-- Prove that the vectors in Option B are coplanar.
theorem option_B_coplanar : coplanar a b c :=
sorry

end option_B_coplanar_l214_21456


namespace compute_expression_l214_21433

theorem compute_expression (x : ℕ) (h : x = 3) : (x^8 + 8 * x^4 + 16) / (x^4 - 4) = 93 :=
by
  rw [h]
  sorry

end compute_expression_l214_21433


namespace part1_part2_l214_21488

theorem part1 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : a^2 + b^2 = 22 :=
sorry

theorem part2 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : (a - 2) * (b + 2) = 7 :=
sorry

end part1_part2_l214_21488


namespace men_joined_l214_21471

-- Definitions for initial conditions
def initial_men : ℕ := 10
def initial_days : ℕ := 50
def extended_days : ℕ := 25

-- Theorem stating the number of men who joined the camp
theorem men_joined (x : ℕ) 
    (initial_food : initial_men * initial_days = (initial_men + x) * extended_days) : 
    x = 10 := 
sorry

end men_joined_l214_21471


namespace number_of_charms_l214_21443

-- Let x be the number of charms used to make each necklace
variable (x : ℕ)

-- Each charm costs $15
variable (cost_per_charm : ℕ)
axiom cost_per_charm_is_15 : cost_per_charm = 15

-- Tim sells each necklace for $200
variable (selling_price : ℕ)
axiom selling_price_is_200 : selling_price = 200

-- Tim makes a profit of $1500 if he sells 30 necklaces
variable (total_profit : ℕ)
axiom total_profit_is_1500 : total_profit = 1500

theorem number_of_charms (h : 30 * (selling_price - cost_per_charm * x) = total_profit) : x = 10 :=
sorry

end number_of_charms_l214_21443


namespace problem_statement_l214_21422

variable {x y : ℝ}

theorem problem_statement (h1 : x * y = -3) (h2 : x + y = -4) : x^2 + 3 * x * y + y^2 = 13 := sorry

end problem_statement_l214_21422


namespace solve_for_f_2012_l214_21410

noncomputable def f : ℝ → ℝ := sorry -- as the exact function definition isn't provided

variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = -f x)
variable (functional_eqn : ∀ x, f (x + 2) = f x + f 2)
variable (f_one : f 1 = 2)

theorem solve_for_f_2012 : f 2012 = 4024 :=
sorry

end solve_for_f_2012_l214_21410


namespace total_roses_planted_three_days_l214_21459

-- Definitions based on conditions
def susan_roses_two_days_ago : ℕ := 10
def maria_roses_two_days_ago : ℕ := 2 * susan_roses_two_days_ago
def john_roses_two_days_ago : ℕ := susan_roses_two_days_ago + 10
def roses_two_days_ago : ℕ := susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago

def roses_yesterday : ℕ := roses_two_days_ago + 20
def susan_roses_yesterday : ℕ := susan_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def maria_roses_yesterday : ℕ := maria_roses_two_days_ago * roses_yesterday / roses_two_days_ago
def john_roses_yesterday : ℕ := john_roses_two_days_ago * roses_yesterday / roses_two_days_ago

def roses_today : ℕ := 2 * roses_two_days_ago
def susan_roses_today : ℕ := susan_roses_two_days_ago
def maria_roses_today : ℕ := maria_roses_two_days_ago + (maria_roses_two_days_ago * 25 / 100)
def john_roses_today : ℕ := john_roses_two_days_ago - (john_roses_two_days_ago * 10 / 100)

def total_roses_planted : ℕ := 
  (susan_roses_two_days_ago + maria_roses_two_days_ago + john_roses_two_days_ago) +
  (susan_roses_yesterday + maria_roses_yesterday + john_roses_yesterday) +
  (susan_roses_today + maria_roses_today + john_roses_today)

-- The statement that needs to be proved
theorem total_roses_planted_three_days : total_roses_planted = 173 := by 
  sorry

end total_roses_planted_three_days_l214_21459


namespace ellipse_to_parabola_standard_eq_l214_21469

theorem ellipse_to_parabola_standard_eq :
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) → (y^2 = 12 * x) :=
by
  sorry

end ellipse_to_parabola_standard_eq_l214_21469


namespace product_of_0_25_and_0_75_is_0_1875_l214_21419

noncomputable def product_of_decimals : ℝ := 0.25 * 0.75

theorem product_of_0_25_and_0_75_is_0_1875 :
  product_of_decimals = 0.1875 :=
by
  sorry

end product_of_0_25_and_0_75_is_0_1875_l214_21419


namespace find_years_simple_interest_l214_21425

variable (R T : ℝ)
variable (P : ℝ := 6000)
variable (additional_interest : ℝ := 360)
variable (rate_diff : ℝ := 2)
variable (H : P * ((R + rate_diff) / 100) * T = P * (R / 100) * T + additional_interest)

theorem find_years_simple_interest (h : P = 6000) (h₁ : P * ((R + 2) / 100) * T = P * (R / 100) * T + 360) : 
T = 3 :=
sorry

end find_years_simple_interest_l214_21425


namespace find_percentage_l214_21409

theorem find_percentage (P : ℝ) (h1 : (P / 100) * 200 = 30 + 0.60 * 50) : P = 30 :=
by
  sorry

end find_percentage_l214_21409


namespace solution_to_largest_four_digit_fulfilling_conditions_l214_21474

def largest_four_digit_fulfilling_conditions : Prop :=
  ∃ (N : ℕ), N < 10000 ∧ N ≡ 2 [MOD 11] ∧ N ≡ 4 [MOD 7] ∧ N = 9979

theorem solution_to_largest_four_digit_fulfilling_conditions : largest_four_digit_fulfilling_conditions :=
  sorry

end solution_to_largest_four_digit_fulfilling_conditions_l214_21474


namespace find_offset_length_l214_21431

theorem find_offset_length 
  (diagonal_offset_7 : ℝ) 
  (area_of_quadrilateral : ℝ) 
  (diagonal_length : ℝ) 
  (result : ℝ) : 
  (diagonal_length = 10) 
  ∧ (diagonal_offset_7 = 7) 
  ∧ (area_of_quadrilateral = 50) 
  → (∃ x, x = result) :=
by
  sorry

end find_offset_length_l214_21431


namespace elaine_earnings_increase_l214_21457

variable (E P : ℝ)

theorem elaine_earnings_increase :
  (0.25 * (E * (1 + P / 100)) = 1.4375 * 0.20 * E) → P = 15 :=
by
  intro h
  -- Start an intermediate transformation here
  sorry

end elaine_earnings_increase_l214_21457


namespace giselle_paint_l214_21446

theorem giselle_paint (x : ℚ) (h1 : 5/7 = x/21) : x = 15 :=
by
  sorry

end giselle_paint_l214_21446


namespace arithmetic_sequence_problem_l214_21438

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ) 
  (a1 : a 1 = 3) 
  (d : ℕ := 2) 
  (h : ∀ n, a n = a 1 + (n - 1) * d) 
  (h_25 : a n = 25) : 
  n = 12 := 
by
  sorry

end arithmetic_sequence_problem_l214_21438


namespace parallelogram_side_length_l214_21403

theorem parallelogram_side_length (x y : ℚ) (h1 : 3 * x + 2 = 12) (h2 : 5 * y - 3 = 9) : x + y = 86 / 15 :=
by 
  sorry

end parallelogram_side_length_l214_21403


namespace graph_inequality_solution_l214_21466

noncomputable def solution_set : Set (Real × Real) := {
  p | let x := p.1
       let y := p.2
       (y^2 - (Real.arcsin (Real.sin x))^2) *
       (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
       (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0
}

theorem graph_inequality_solution
  (x y : ℝ) :
  (y^2 - (Real.arcsin (Real.sin x))^2) *
  (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
  (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0 ↔
  (x, y) ∈ solution_set :=
by
  sorry

end graph_inequality_solution_l214_21466


namespace find_fraction_divide_equal_l214_21472

theorem find_fraction_divide_equal (x : ℚ) : 
  (3 * x = (1 / (5 / 2))) → (x = 2 / 15) :=
by
  intro h
  sorry

end find_fraction_divide_equal_l214_21472


namespace union_of_A_and_B_l214_21445

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} :=
by
  sorry

end union_of_A_and_B_l214_21445


namespace school_club_profit_l214_21442

def price_per_bar_buy : ℚ := 5 / 6
def price_per_bar_sell : ℚ := 2 / 3
def total_bars : ℕ := 1200
def total_cost : ℚ := total_bars * price_per_bar_buy
def total_revenue : ℚ := total_bars * price_per_bar_sell
def profit : ℚ := total_revenue - total_cost

theorem school_club_profit : profit = -200 := by
  sorry

end school_club_profit_l214_21442


namespace gumballs_initial_count_l214_21464

theorem gumballs_initial_count (x : ℝ) (h : (0.75 ^ 3) * x = 27) : x = 64 :=
by
  sorry

end gumballs_initial_count_l214_21464


namespace inequality_proof_l214_21435

variable (x y z : ℝ)

theorem inequality_proof
  (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 :=
by
  sorry

end inequality_proof_l214_21435


namespace sqrt_inequality_l214_21485

open Real

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z) 
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  sqrt (x + y + z) ≥ sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) :=
sorry

end sqrt_inequality_l214_21485


namespace simplify_expression_l214_21455

theorem simplify_expression (x : ℝ) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) :=
sorry

end simplify_expression_l214_21455


namespace incorrect_statement_about_zero_l214_21468

theorem incorrect_statement_about_zero :
  ¬ (0 > 0) :=
by
  sorry

end incorrect_statement_about_zero_l214_21468


namespace hall_length_l214_21496

variable (breadth length : ℝ)

def condition1 : Prop := length = breadth + 5
def condition2 : Prop := length * breadth = 750

theorem hall_length : condition1 breadth length ∧ condition2 breadth length → length = 30 :=
by
  intros
  sorry

end hall_length_l214_21496


namespace unit_digit_8_pow_1533_l214_21447

theorem unit_digit_8_pow_1533 : (8^1533 % 10) = 8 := by
  sorry

end unit_digit_8_pow_1533_l214_21447


namespace solve_inequality_l214_21444

theorem solve_inequality (a x : ℝ) : 
  (a < 0 → (x ≤ 3 / a ∨ x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 0 → (x ≥ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (0 < a ∧ a < 3 → (1 ≤ x ∧ x ≤ 3 / a) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a = 3 → (x = 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) ∧
  (a > 3 → (3 / a ≤ x ∧ x ≤ 1) ↔ ax^2 - (a + 3) * x + 3 ≤ 0) :=
  sorry

end solve_inequality_l214_21444


namespace find_xyz_l214_21416

theorem find_xyz : ∃ (x y z : ℕ), x + y + z = 12 ∧ 7 * x + 5 * y + 8 * z = 79 ∧ x = 5 ∧ y = 4 ∧ z = 3 :=
by
  sorry

end find_xyz_l214_21416


namespace power_of_54_l214_21494

theorem power_of_54 (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
(h_eq : 54^a = a^b) : ∃ k : ℕ, a = 54^k := by
  sorry

end power_of_54_l214_21494


namespace smallest_base10_integer_l214_21461

theorem smallest_base10_integer :
  ∃ (n A B : ℕ), 
    (A < 5) ∧ (B < 7) ∧ 
    (n = 6 * A) ∧ 
    (n = 8 * B) ∧ 
    n = 24 := 
sorry

end smallest_base10_integer_l214_21461


namespace A_beats_B_by_160_meters_l214_21400

-- Definitions used in conditions
def distance_A := 400 -- meters
def time_A := 60 -- seconds
def distance_B := 400 -- meters
def time_B := 100 -- seconds
def speed_B := distance_B / time_B -- B's speed in meters/second
def time_for_B_in_A_time := time_A -- B's time for the duration A took to finish the race
def distance_B_in_A_time := speed_B * time_for_B_in_A_time -- Distance B covers in A's time

-- Statement to prove
theorem A_beats_B_by_160_meters : distance_A - distance_B_in_A_time = 160 :=
by
  -- This is a placeholder for an eventual proof
  sorry

end A_beats_B_by_160_meters_l214_21400


namespace optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l214_21475

theorem optionA_incorrect (a x : ℝ) : 3 * a * x^2 - 6 * a * x ≠ 3 * (a * x^2 - 2 * a * x) :=
by sorry

theorem optionB_incorrect (a x : ℝ) : (x + a) * (x - a) ≠ x^2 - a^2 :=
by sorry

theorem optionC_incorrect (a b : ℝ) : a^2 + 2 * a * b - 4 * b^2 ≠ (a + 2 * b)^2 :=
by sorry

theorem optionD_correct (a x : ℝ) : -a * x^2 + 2 * a * x - a = -a * (x - 1)^2 :=
by sorry

end optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l214_21475
