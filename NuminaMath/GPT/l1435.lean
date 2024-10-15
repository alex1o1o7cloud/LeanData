import Mathlib

namespace NUMINAMATH_GPT_union_of_sets_l1435_143599

def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 2}

theorem union_of_sets : A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l1435_143599


namespace NUMINAMATH_GPT_inequality_condition_l1435_143576

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) ∨ (False) := 
sorry

end NUMINAMATH_GPT_inequality_condition_l1435_143576


namespace NUMINAMATH_GPT_regular_admission_ticket_price_l1435_143574

theorem regular_admission_ticket_price
  (n : ℕ) (t : ℕ) (p : ℕ)
  (n_r n_s r : ℕ)
  (H1 : n_r = 3 * n_s)
  (H2 : n_s + n_r = n)
  (H3 : n_r * r + n_s * p = t)
  (H4 : n = 3240)
  (H5 : t = 22680)
  (H6 : p = 4) : 
  r = 8 :=
by sorry

end NUMINAMATH_GPT_regular_admission_ticket_price_l1435_143574


namespace NUMINAMATH_GPT_sum_of_remainders_and_parity_l1435_143515

theorem sum_of_remainders_and_parity 
  (n : ℤ) 
  (h₀ : n % 20 = 13) : 
  (n % 4 + n % 5 = 4) ∧ (n % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_and_parity_l1435_143515


namespace NUMINAMATH_GPT_no_real_x_for_sqrt_l1435_143526

theorem no_real_x_for_sqrt :
  ¬ ∃ x : ℝ, - (x^2 + 2 * x + 5) ≥ 0 :=
sorry

end NUMINAMATH_GPT_no_real_x_for_sqrt_l1435_143526


namespace NUMINAMATH_GPT_man_speed_in_still_water_l1435_143522

noncomputable def speedInStillWater 
  (upstreamSpeedWithCurrentAndWind : ℝ)
  (downstreamSpeedWithCurrentAndWind : ℝ)
  (waterCurrentSpeed : ℝ)
  (windSpeedUpstream : ℝ) : ℝ :=
  (upstreamSpeedWithCurrentAndWind + waterCurrentSpeed + windSpeedUpstream + downstreamSpeedWithCurrentAndWind - waterCurrentSpeed + windSpeedUpstream) / 2
  
theorem man_speed_in_still_water :
  speedInStillWater 20 60 5 2.5 = 42.5 :=
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l1435_143522


namespace NUMINAMATH_GPT_total_eggs_emily_collected_l1435_143563

theorem total_eggs_emily_collected :
  let number_of_baskets := 303
  let eggs_per_basket := 28
  number_of_baskets * eggs_per_basket = 8484 :=
by
  let number_of_baskets := 303
  let eggs_per_basket := 28
  sorry -- Proof to be provided

end NUMINAMATH_GPT_total_eggs_emily_collected_l1435_143563


namespace NUMINAMATH_GPT_graph_is_pair_of_straight_lines_l1435_143564

theorem graph_is_pair_of_straight_lines : ∀ (x y : ℝ), 9 * x^2 - y^2 - 6 * x = 0 → ∃ a b c : ℝ, (y = 3 * x - 2 ∨ y = 2 - 3 * x) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_graph_is_pair_of_straight_lines_l1435_143564


namespace NUMINAMATH_GPT_table_tennis_probability_l1435_143538

-- Define the given conditions
def prob_A_wins_set : ℚ := 2 / 3
def prob_B_wins_set : ℚ := 1 / 3
def best_of_five_sets := 5
def needed_wins_for_A := 3
def needed_losses_for_A := 2

-- Define the problem to prove
theorem table_tennis_probability :
  ((prob_A_wins_set ^ 2) * prob_B_wins_set * prob_A_wins_set) = 8 / 27 :=
by
  sorry

end NUMINAMATH_GPT_table_tennis_probability_l1435_143538


namespace NUMINAMATH_GPT_handshake_max_l1435_143598

theorem handshake_max (N : ℕ) (hN : N > 4) (pN pNm1 : ℕ) 
    (hpN : pN ≠ pNm1) (h1 : ∃ p1, pN ≠ p1) (h2 : ∃ p2, pNm1 ≠ p2) :
    ∀ (i : ℕ), i ≤ N - 2 → i ≤ N - 2 :=
sorry

end NUMINAMATH_GPT_handshake_max_l1435_143598


namespace NUMINAMATH_GPT_marble_game_solution_l1435_143597

theorem marble_game_solution (B R : ℕ) (h1 : B + R = 21) (h2 : (B * (B - 1)) / (21 * 20) = 1 / 2) : B^2 + R^2 = 261 :=
by
  sorry

end NUMINAMATH_GPT_marble_game_solution_l1435_143597


namespace NUMINAMATH_GPT_room_width_l1435_143584

theorem room_width (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ)
  (h_length : length = 5.5)
  (h_total_cost : total_cost = 15400)
  (h_rate_per_sqm : rate_per_sqm = 700)
  (h_area : total_cost = rate_per_sqm * (length * width)) :
  width = 4 := 
sorry

end NUMINAMATH_GPT_room_width_l1435_143584


namespace NUMINAMATH_GPT_no_nonzero_ints_l1435_143594

theorem no_nonzero_ints (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A ∣ (A + B) ∨ B ∣ (A - B)) → false :=
sorry

end NUMINAMATH_GPT_no_nonzero_ints_l1435_143594


namespace NUMINAMATH_GPT_minimum_restoration_time_l1435_143547

structure Handicraft :=
  (shaping: ℕ)
  (painting: ℕ)

def handicraft_A : Handicraft := ⟨9, 15⟩
def handicraft_B : Handicraft := ⟨16, 8⟩
def handicraft_C : Handicraft := ⟨10, 14⟩

def total_restoration_time (order: List Handicraft) : ℕ :=
  let rec aux (remaining: List Handicraft) (A_time: ℕ) (B_time: ℕ) (acc: ℕ) : ℕ :=
    match remaining with
    | [] => acc
    | h :: t =>
      let A_next := A_time + h.shaping
      let B_next := max A_next B_time + h.painting
      aux t A_next B_next B_next
  aux order 0 0 0

theorem minimum_restoration_time :
  total_restoration_time [handicraft_A, handicraft_C, handicraft_B] = 46 :=
by
  simp [total_restoration_time, handicraft_A, handicraft_B, handicraft_C]
  sorry

end NUMINAMATH_GPT_minimum_restoration_time_l1435_143547


namespace NUMINAMATH_GPT_croissant_to_orange_ratio_l1435_143581

-- Define the conditions as given in the problem
variables (c o : ℝ)
variable (emily_expenditure : ℝ)
variable (lucas_expenditure : ℝ)

-- Given conditions of expenditures
axiom emily_expenditure_is : emily_expenditure = 5 * c + 4 * o
axiom lucas_expenditure_is : lucas_expenditure = 3 * emily_expenditure
axiom lucas_expenditure_as_purchased : lucas_expenditure = 4 * c + 10 * o

-- Prove the ratio of the cost of a croissant to an orange
theorem croissant_to_orange_ratio : (c / o) = 2 / 11 :=
by sorry

end NUMINAMATH_GPT_croissant_to_orange_ratio_l1435_143581


namespace NUMINAMATH_GPT_composite_divides_factorial_l1435_143550

-- Define the factorial of a number
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement of the problem
theorem composite_divides_factorial (m : ℕ) (hm : m ≠ 4) (hcomposite : ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = m) :
  m ∣ factorial (m - 1) :=
by
  sorry

end NUMINAMATH_GPT_composite_divides_factorial_l1435_143550


namespace NUMINAMATH_GPT_circle_through_point_and_tangent_to_lines_l1435_143579

theorem circle_through_point_and_tangent_to_lines :
  ∃ h k,
     ((h, k) = (4 / 5, 3 / 5) ∨ (h, k) = (4, -1)) ∧ 
     ((x - h)^2 + (y - k)^2 = 5) :=
by
  let P := (3, 1)
  let l1 := fun x y => x + 2 * y + 3 
  let l2 := fun x y => x + 2 * y - 7 
  sorry

end NUMINAMATH_GPT_circle_through_point_and_tangent_to_lines_l1435_143579


namespace NUMINAMATH_GPT_binom_26_6_l1435_143589

theorem binom_26_6 (h₁ : Nat.choose 25 5 = 53130) (h₂ : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 :=
by
  sorry

end NUMINAMATH_GPT_binom_26_6_l1435_143589


namespace NUMINAMATH_GPT_average_speed_of_car_l1435_143544

/-- The car's average speed given it travels 65 km in the first hour and 45 km in the second hour. -/
theorem average_speed_of_car (d1 d2 : ℕ) (t : ℕ) (h1 : d1 = 65) (h2 : d2 = 45) (h3 : t = 2) :
  (d1 + d2) / t = 55 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l1435_143544


namespace NUMINAMATH_GPT_biff_break_even_time_l1435_143551

noncomputable def total_cost_excluding_wifi : ℝ :=
  11 + 3 + 16 + 8 + 10 + 35 + 0.1 * 35

noncomputable def total_cost_including_wifi_connection : ℝ :=
  total_cost_excluding_wifi + 5

noncomputable def effective_hourly_earning : ℝ := 12 - 1

noncomputable def hours_to_break_even : ℝ :=
  total_cost_including_wifi_connection / effective_hourly_earning

theorem biff_break_even_time : hours_to_break_even ≤ 9 := by
  sorry

end NUMINAMATH_GPT_biff_break_even_time_l1435_143551


namespace NUMINAMATH_GPT_train_length_is_correct_l1435_143587

noncomputable def length_of_train (time_in_seconds : ℝ) (relative_speed : ℝ) : ℝ :=
  relative_speed * time_in_seconds

noncomputable def relative_speed_in_mps (speed_of_train_kmph : ℝ) (speed_of_man_kmph : ℝ) : ℝ :=
  (speed_of_train_kmph + speed_of_man_kmph) * (1000 / 3600)

theorem train_length_is_correct :
  let speed_of_train_kmph := 65.99424046076315
  let speed_of_man_kmph := 6
  let time_in_seconds := 6
  length_of_train time_in_seconds (relative_speed_in_mps speed_of_train_kmph speed_of_man_kmph) = 119.9904 := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1435_143587


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_l1435_143578

theorem sum_of_cubes_of_roots:
  (∀ r s t : ℝ, (r + s + t = 8) ∧ (r * s + s * t + t * r = 9) ∧ (r * s * t = 2) → r^3 + s^3 + t^3 = 344) :=
by
  intros r s t h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_l1435_143578


namespace NUMINAMATH_GPT_ratio_of_teenagers_to_toddlers_l1435_143511

theorem ratio_of_teenagers_to_toddlers
  (total_children : ℕ)
  (number_of_toddlers : ℕ)
  (number_of_newborns : ℕ)
  (h1 : total_children = 40)
  (h2 : number_of_toddlers = 6)
  (h3 : number_of_newborns = 4)
  : (total_children - number_of_toddlers - number_of_newborns) / number_of_toddlers = 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_teenagers_to_toddlers_l1435_143511


namespace NUMINAMATH_GPT_problem_l1435_143549

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := sorry
def v : Fin 2 → ℝ := ![7, -3]
def result : Fin 2 → ℝ := ![-14, 6]
def expected : Fin 2 → ℝ := ![112, -48]

theorem problem :
    B.vecMul v = result →
    B.vecMul (B.vecMul (B.vecMul (B.vecMul v))) = expected := 
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_l1435_143549


namespace NUMINAMATH_GPT_symmetric_point_with_respect_to_y_eq_x_l1435_143556

variables (P : ℝ × ℝ) (line : ℝ → ℝ)

theorem symmetric_point_with_respect_to_y_eq_x (P : ℝ × ℝ) (hP : P = (1, 3)) (hline : ∀ x, line x = x) :
  (∃ Q : ℝ × ℝ, Q = (3, 1) ∧ Q = (P.snd, P.fst)) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_with_respect_to_y_eq_x_l1435_143556


namespace NUMINAMATH_GPT_steve_family_time_l1435_143512

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end NUMINAMATH_GPT_steve_family_time_l1435_143512


namespace NUMINAMATH_GPT_evaluate_expression_l1435_143502

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1435_143502


namespace NUMINAMATH_GPT_inequality_solution_l1435_143540

theorem inequality_solution (x : ℝ) : 9 - x^2 < 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1435_143540


namespace NUMINAMATH_GPT_problem_solution_l1435_143596

theorem problem_solution (x : ℝ) (h : x^2 - 8*x - 3 = 0) : (x - 1) * (x - 3) * (x - 5) * (x - 7) = 180 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1435_143596


namespace NUMINAMATH_GPT_roshini_spent_on_sweets_l1435_143582

theorem roshini_spent_on_sweets
  (initial_amount : Real)
  (amount_given_per_friend : Real)
  (num_friends : Nat)
  (total_amount_given : Real)
  (amount_spent_on_sweets : Real) :
  initial_amount = 10.50 →
  amount_given_per_friend = 3.40 →
  num_friends = 2 →
  total_amount_given = amount_given_per_friend * num_friends →
  amount_spent_on_sweets = initial_amount - total_amount_given →
  amount_spent_on_sweets = 3.70 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_roshini_spent_on_sweets_l1435_143582


namespace NUMINAMATH_GPT_sum_of_sequences_l1435_143593

-- Define the sequences and their type
def seq1 : List ℕ := [2, 12, 22, 32, 42]
def seq2 : List ℕ := [10, 20, 30, 40, 50]

-- The property we wish to prove
theorem sum_of_sequences : seq1.sum + seq2.sum = 260 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_sequences_l1435_143593


namespace NUMINAMATH_GPT_sufficient_condition_perpendicular_l1435_143520

-- Definitions of perpendicularity and lines/planes intersections
variables {Plane : Type} {Line : Type}

variable (α β γ : Plane)
variable (m n l : Line)

-- Axioms representing the given conditions
axiom perp_planes (p₁ p₂ : Plane) : Prop -- p₁ is perpendicular to p₂
axiom perp_line_plane (line : Line) (plane : Plane) : Prop -- line is perpendicular to plane

-- Given conditions for the problem.
axiom n_perp_α : perp_line_plane n α
axiom n_perp_β : perp_line_plane n β
axiom m_perp_α : perp_line_plane m α

-- The proposition to be proved.
theorem sufficient_condition_perpendicular (h₁ : perp_line_plane n α)
                                           (h₂ : perp_line_plane n β)
                                           (h₃ : perp_line_plane m α) :
  perp_line_plane m β := sorry

end NUMINAMATH_GPT_sufficient_condition_perpendicular_l1435_143520


namespace NUMINAMATH_GPT_product_remainder_mod_7_l1435_143571

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_product_remainder_mod_7_l1435_143571


namespace NUMINAMATH_GPT_ammonia_formation_l1435_143562

theorem ammonia_formation (Li3N H2O LiOH NH3 : ℕ) (h₁ : Li3N = 1) (h₂ : H2O = 54) (h₃ : Li3N + 3 * H2O = 3 * LiOH + NH3) :
  NH3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_ammonia_formation_l1435_143562


namespace NUMINAMATH_GPT_problem_statement_l1435_143561

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else List.prod (List.range (n+1))

theorem problem_statement : ∃ r : ℕ, r < 13 ∧ (factorial 10) % 13 = r :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1435_143561


namespace NUMINAMATH_GPT_sum_of_digits_in_product_is_fourteen_l1435_143566

def first_number : ℕ := -- Define the 101-digit number 141,414,141,...,414,141
  141 * 10^98 + 141 * 10^95 + 141 * 10^92 -- continue this pattern...

def second_number : ℕ := -- Define the 101-digit number 707,070,707,...,070,707
  707 * 10^98 + 707 * 10^95 + 707 * 10^92 -- continue this pattern...

def units_digit (n : ℕ) : ℕ := n % 10
def ten_thousands_digit (n : ℕ) : ℕ := (n / 10000) % 10

theorem sum_of_digits_in_product_is_fourteen :
  units_digit (first_number * second_number) + ten_thousands_digit (first_number * second_number) = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_in_product_is_fourteen_l1435_143566


namespace NUMINAMATH_GPT_wraps_add_more_l1435_143537

/-- Let John's raw squat be 600 pounds. Let sleeves add 30 pounds to his lift. Let wraps add 25% 
to his squat. We aim to prove that wraps add 120 pounds more to John's squat than sleeves. -/
theorem wraps_add_more (raw_squat : ℝ) (sleeves_bonus : ℝ) (wraps_percentage : ℝ) : 
  raw_squat = 600 → sleeves_bonus = 30 → wraps_percentage = 0.25 → 
  (raw_squat * wraps_percentage) - sleeves_bonus = 120 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_wraps_add_more_l1435_143537


namespace NUMINAMATH_GPT_circle_equation_l1435_143541

theorem circle_equation (x y : ℝ) :
  (∃ a < 0, (x - a)^2 + y^2 = 4 ∧ (0 - a)^2 + 0^2 = 4) ↔ (x + 2)^2 + y^2 = 4 := 
sorry

end NUMINAMATH_GPT_circle_equation_l1435_143541


namespace NUMINAMATH_GPT_minimum_value_expr_eq_neg6680_25_l1435_143573

noncomputable def expr (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200

theorem minimum_value_expr_eq_neg6680_25 : ∃ x : ℝ, (∀ y : ℝ, expr y ≥ expr x) ∧ expr x = -6680.25 :=
sorry

end NUMINAMATH_GPT_minimum_value_expr_eq_neg6680_25_l1435_143573


namespace NUMINAMATH_GPT_time_to_fill_pool_l1435_143535

-- Define the conditions given in the problem
def pool_volume_gallons : ℕ := 30000
def num_hoses : ℕ := 5
def hose_flow_rate_gpm : ℕ := 3

-- Define the total flow rate per minute
def total_flow_rate_gpm : ℕ := num_hoses * hose_flow_rate_gpm

-- Define the total flow rate per hour
def total_flow_rate_gph : ℕ := total_flow_rate_gpm * 60

-- Prove that the time to fill the pool is equal to 34 hours
theorem time_to_fill_pool : pool_volume_gallons / total_flow_rate_gph = 34 :=
by {
  -- Insert detailed proof steps here.
  sorry
}

end NUMINAMATH_GPT_time_to_fill_pool_l1435_143535


namespace NUMINAMATH_GPT_pens_difference_proof_l1435_143503

variables (A B M N X Y : ℕ)

-- Initial number of pens for Alex and Jane
def Alex_initial (A : ℕ) := A
def Jane_initial (B : ℕ) := B

-- Weekly multiplication factors for Alex and Jane
def Alex_weekly_growth (X : ℕ) := X
def Jane_weekly_growth (Y : ℕ) := Y

-- Number of pens after 4 weeks
def Alex_after_4_weeks (A X : ℕ) := A * X^4
def Jane_after_4_weeks (B Y : ℕ) := B * Y^4

-- Proving the difference in the number of pens
theorem pens_difference_proof (hM : M = A * X^4) (hN : N = B * Y^4) :
  M - N = (A * X^4) - (B * Y^4) :=
by sorry

end NUMINAMATH_GPT_pens_difference_proof_l1435_143503


namespace NUMINAMATH_GPT_simon_sand_dollars_l1435_143586

theorem simon_sand_dollars (S G P : ℕ) (h1 : G = 3 * S) (h2 : P = 5 * G) (h3 : S + G + P = 190) : S = 10 := by
  sorry

end NUMINAMATH_GPT_simon_sand_dollars_l1435_143586


namespace NUMINAMATH_GPT_find_M_plus_N_l1435_143529

theorem find_M_plus_N (M N : ℕ)
  (h1 : 4 * 63 = 7 * M)
  (h2 : 4 * N = 7 * 84) :
  M + N = 183 :=
by sorry

end NUMINAMATH_GPT_find_M_plus_N_l1435_143529


namespace NUMINAMATH_GPT_b_investment_calculation_l1435_143548

noncomputable def total_profit : ℝ := 9600
noncomputable def A_investment : ℝ := 2000
noncomputable def A_management_fee : ℝ := 0.10 * total_profit
noncomputable def remaining_profit : ℝ := total_profit - A_management_fee
noncomputable def A_total_received : ℝ := 4416
noncomputable def B_investment : ℝ := 1000

theorem b_investment_calculation (B: ℝ) 
  (h_total_profit: total_profit = 9600)
  (h_A_investment: A_investment = 2000)
  (h_A_management_fee: A_management_fee = 0.10 * total_profit)
  (h_remaining_profit: remaining_profit = total_profit - A_management_fee)
  (h_A_total_received: A_total_received = 4416)
  (h_A_total_formula : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit) :
  B = 1000 :=
by
  have h1 : total_profit = 9600 := h_total_profit
  have h2 : A_investment = 2000 := h_A_investment
  have h3 : A_management_fee = 0.10 * total_profit := h_A_management_fee
  have h4 : remaining_profit = total_profit - A_management_fee := h_remaining_profit
  have h5 : A_total_received = 4416 := h_A_total_received
  have h6 : A_total_received = A_management_fee + (A_investment / (A_investment + B)) * remaining_profit := h_A_total_formula
  
  sorry

end NUMINAMATH_GPT_b_investment_calculation_l1435_143548


namespace NUMINAMATH_GPT_exist_equilateral_triangle_on_parallel_lines_l1435_143590

-- Define the concept of lines and points in a relation to them
def Line := ℝ → ℝ -- For simplicity, let's assume lines are functions

-- Define the points A1, A2, A3
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the concept of parallel lines
def parallel (D1 D2 : Line) : Prop :=
  ∀ x y, D1 x - D2 x = D1 y - D2 y

axiom D1 : Line
axiom D2 : Line
axiom D3 : Line

-- Ensure the lines are parallel
axiom parallel_D1_D2 : parallel D1 D2
axiom parallel_D2_D3 : parallel D2 D3

-- Main statement to prove
theorem exist_equilateral_triangle_on_parallel_lines :
  ∃ (A1 A2 A3 : Point), 
    (A1.y = D1 A1.x) ∧ 
    (A2.y = D2 A2.x) ∧ 
    (A3.y = D3 A3.x) ∧ 
    ((A1.x - A2.x)^2 + (A1.y - A2.y)^2 = (A2.x - A3.x)^2 + (A2.y - A3.y)^2) ∧ 
    ((A2.x - A3.x)^2 + (A2.y - A3.y)^2 = (A3.x - A1.x)^2 + (A3.y - A1.y)^2) := sorry

end NUMINAMATH_GPT_exist_equilateral_triangle_on_parallel_lines_l1435_143590


namespace NUMINAMATH_GPT_dogs_legs_l1435_143555

theorem dogs_legs (num_dogs : ℕ) (legs_per_dog : ℕ) (h1 : num_dogs = 109) (h2 : legs_per_dog = 4) : num_dogs * legs_per_dog = 436 :=
by {
  -- The proof is omitted as it's indicated that it should contain "sorry"
  sorry
}

end NUMINAMATH_GPT_dogs_legs_l1435_143555


namespace NUMINAMATH_GPT_original_profit_percentage_l1435_143532

noncomputable def originalCost : ℝ := 80
noncomputable def P := 30
noncomputable def profitPercentage : ℝ := ((100 - originalCost) / originalCost) * 100

theorem original_profit_percentage:
  ∀ (S C : ℝ),
  C = originalCost →
  ( ∀ (newCost : ℝ),
    newCost = 0.8 * C →
    ∀ (newSell : ℝ),
    newSell = S - 16.8 →
    newSell = 1.3 * newCost → P = 30 ) →
  profitPercentage = 25 := sorry

end NUMINAMATH_GPT_original_profit_percentage_l1435_143532


namespace NUMINAMATH_GPT_series_converges_to_one_l1435_143531

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n > 0 then (3 * (n : ℝ)^2 - 2 * (n : ℝ) + 1) / ((n : ℝ)^4 - (n : ℝ)^3 + (n : ℝ)^2 - (n : ℝ) + 1) else 0

theorem series_converges_to_one : series_sum = 1 := 
  sorry

end NUMINAMATH_GPT_series_converges_to_one_l1435_143531


namespace NUMINAMATH_GPT_hens_not_laying_eggs_l1435_143558

def chickens_on_farm := 440
def number_of_roosters := 39
def total_eggs := 1158
def eggs_per_hen := 3

theorem hens_not_laying_eggs :
  (chickens_on_farm - number_of_roosters) - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end NUMINAMATH_GPT_hens_not_laying_eggs_l1435_143558


namespace NUMINAMATH_GPT_geometric_sequence_min_value_l1435_143568

theorem geometric_sequence_min_value (r : ℝ) (a1 a2 a3 : ℝ) 
  (h1 : a1 = 1) 
  (h2 : a2 = a1 * r) 
  (h3 : a3 = a2 * r) :
  4 * a2 + 5 * a3 ≥ -(4 / 5) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_min_value_l1435_143568


namespace NUMINAMATH_GPT_no_solution_to_system_l1435_143505

theorem no_solution_to_system :
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 12 ∧ 9 * x - 12 * y = 15) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_to_system_l1435_143505


namespace NUMINAMATH_GPT_math_problem_l1435_143553

theorem math_problem
  (p q r : ℝ)
  (h₁ : p + q + r = 5)
  (h₂ : (1 / (p + q)) + (1 / (q + r)) + (1 / (p + r)) = 9) :
  (r / (p + q)) + (p / (q + r)) + (q / (p + r)) = 42 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1435_143553


namespace NUMINAMATH_GPT_younger_brother_silver_fraction_l1435_143565

def frac_silver (x y : ℕ) : ℚ := (100 - x / 7 ) / y

theorem younger_brother_silver_fraction {x y : ℕ} 
    (cond1 : x / 5 + y / 7 = 100) 
    (cond2 : x / 7 + (100 - x / 7) = 100) : 
    frac_silver x y = 5 / 14 := 
sorry

end NUMINAMATH_GPT_younger_brother_silver_fraction_l1435_143565


namespace NUMINAMATH_GPT_rectangle_area_is_200000_l1435_143524

structure Point :=
  (x : ℝ)
  (y : ℝ)

def isRectangle (P Q R S : Point) : Prop :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y) = 
  (R.x - S.x) * (R.x - S.x) + (R.y - S.y) * (R.y - S.y) ∧
  (P.x - S.x) * (P.x - S.x) + (P.y - S.y) * (P.y - S.y) = 
  (Q.x - R.x) * (Q.x - R.x) + (Q.y - R.y) * (Q.y - R.y) ∧
  (P.x - Q.x) * (P.x - S.x) + (P.y - Q.y) * (P.y - S.y) = 0

theorem rectangle_area_is_200000:
  ∀ (P Q R S : Point),
  P = ⟨-15, 30⟩ →
  Q = ⟨985, 230⟩ →
  R.x = 985 → 
  S.x = -13 →
  R.y = S.y → 
  isRectangle P Q R S →
  ( ( (Q.x - P.x)^2 + (Q.y - P.y)^2 ).sqrt *
    ( (S.x - P.x)^2 + (S.y - P.y)^2 ).sqrt ) = 200000 :=
by
  intros P Q R S hP hQ hxR hxS hyR hRect
  sorry

end NUMINAMATH_GPT_rectangle_area_is_200000_l1435_143524


namespace NUMINAMATH_GPT_sin_double_pi_minus_theta_eq_l1435_143530

variable {θ : ℝ}
variable {k : ℤ}
variable (h1 : 3 * (Real.cos θ) ^ 2 = Real.tan θ + 3)
variable (h2 : θ ≠ k * Real.pi)

theorem sin_double_pi_minus_theta_eq :
  Real.sin (2 * (Real.pi - θ)) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_sin_double_pi_minus_theta_eq_l1435_143530


namespace NUMINAMATH_GPT_impossible_tiling_conditions_l1435_143504

theorem impossible_tiling_conditions (m n : ℕ) :
  ¬ (∃ (a b : ℕ), (a - 1) * 4 + (b + 1) * 4 = m * n ∧ a * 4 % 4 = 2 ∧ b * 4 % 4 = 0) :=
sorry

end NUMINAMATH_GPT_impossible_tiling_conditions_l1435_143504


namespace NUMINAMATH_GPT_polynomial_irreducible_if_not_divisible_by_5_l1435_143506

theorem polynomial_irreducible_if_not_divisible_by_5 (k : ℤ) (h1 : ¬ ∃ m : ℤ, k = 5 * m) :
    ¬ ∃ (f g : Polynomial ℤ), (f.degree < 5) ∧ (f * g = x^5 - x + Polynomial.C k) :=
  sorry

end NUMINAMATH_GPT_polynomial_irreducible_if_not_divisible_by_5_l1435_143506


namespace NUMINAMATH_GPT_edge_length_approx_17_1_l1435_143570

-- Define the base dimensions of the rectangular vessel
def length_base : ℝ := 20
def width_base : ℝ := 15

-- Define the rise in water level
def rise_water_level : ℝ := 16.376666666666665

-- Calculate the area of the base
def area_base : ℝ := length_base * width_base

-- Calculate the volume of the cube (which is equal to the volume of water displaced)
def volume_cube : ℝ := area_base * rise_water_level

-- Calculate the edge length of the cube
def edge_length_cube : ℝ := volume_cube^(1/3)

-- Statement: The edge length of the cube is approximately 17.1 cm
theorem edge_length_approx_17_1 : abs (edge_length_cube - 17.1) < 0.1 :=
by sorry

end NUMINAMATH_GPT_edge_length_approx_17_1_l1435_143570


namespace NUMINAMATH_GPT_distinct_divisors_sum_factorial_l1435_143523

theorem distinct_divisors_sum_factorial (n : ℕ) (h : n ≥ 3) :
  ∃ (d : Fin n → ℕ), (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n!) ∧ (n! = (Finset.univ.sum d)) :=
sorry

end NUMINAMATH_GPT_distinct_divisors_sum_factorial_l1435_143523


namespace NUMINAMATH_GPT_uncle_gave_13_l1435_143543

-- Define all the given constants based on the conditions.
def J := 7    -- cost of the jump rope
def B := 12   -- cost of the board game
def P := 4    -- cost of the playground ball
def S := 6    -- savings from Dalton's allowance
def N := 4    -- additional amount needed

-- Derived quantities
def total_cost := J + B + P

-- Statement: to prove Dalton's uncle gave him $13.
theorem uncle_gave_13 : (total_cost - N) - S = 13 := by
  sorry

end NUMINAMATH_GPT_uncle_gave_13_l1435_143543


namespace NUMINAMATH_GPT_average_visitors_on_Sundays_l1435_143542

theorem average_visitors_on_Sundays (S : ℕ) (h1 : 30 = 5 + 25) (h2 : 25 * 240 + 5 * S = 30 * 285) :
  S = 510 := sorry

end NUMINAMATH_GPT_average_visitors_on_Sundays_l1435_143542


namespace NUMINAMATH_GPT_pumps_time_to_empty_pool_l1435_143580

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end NUMINAMATH_GPT_pumps_time_to_empty_pool_l1435_143580


namespace NUMINAMATH_GPT_unable_to_determine_questions_answered_l1435_143521

variable (total_questions : ℕ) (total_time : ℕ) (used_time : ℕ) (remaining_time : ℕ)

theorem unable_to_determine_questions_answered (total_questions_eq : total_questions = 80)
  (total_time_eq : total_time = 60)
  (used_time_eq : used_time = 12)
  (remaining_time_eq : remaining_time = 0) :
  ∀ (answered_rate : ℕ → ℕ), ¬ ∃ questions_answered, answered_rate used_time = questions_answered :=
by sorry

end NUMINAMATH_GPT_unable_to_determine_questions_answered_l1435_143521


namespace NUMINAMATH_GPT_total_blue_marbles_correct_l1435_143539

def total_blue_marbles (j t e : ℕ) : ℕ :=
  j + t + e

theorem total_blue_marbles_correct :
  total_blue_marbles 44 24 36 = 104 :=
by
  sorry

end NUMINAMATH_GPT_total_blue_marbles_correct_l1435_143539


namespace NUMINAMATH_GPT_rachel_total_homework_pages_l1435_143572

-- Define the conditions
def math_homework_pages : Nat := 10
def additional_reading_pages : Nat := 3

-- Define the proof goal
def total_homework_pages (math_pages reading_extra : Nat) : Nat :=
  math_pages + (math_pages + reading_extra)

-- The final statement with the expected result
theorem rachel_total_homework_pages : total_homework_pages math_homework_pages additional_reading_pages = 23 :=
by
  sorry

end NUMINAMATH_GPT_rachel_total_homework_pages_l1435_143572


namespace NUMINAMATH_GPT_chromium_alloy_l1435_143508

theorem chromium_alloy (x : ℝ) (h1 : 0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) : x = 15 := 
by 
  -- statement only, no proof required.
  sorry

end NUMINAMATH_GPT_chromium_alloy_l1435_143508


namespace NUMINAMATH_GPT_gcd_24_36_54_l1435_143528

-- Define the numbers and the gcd function
def num1 : ℕ := 24
def num2 : ℕ := 36
def num3 : ℕ := 54

-- The Lean statement to prove that the gcd of num1, num2, and num3 is 6
theorem gcd_24_36_54 : Nat.gcd (Nat.gcd num1 num2) num3 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_24_36_54_l1435_143528


namespace NUMINAMATH_GPT_ratio_sum_l1435_143507

theorem ratio_sum {x y : ℚ} (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_sum_l1435_143507


namespace NUMINAMATH_GPT_people_speak_neither_l1435_143567

-- Define the total number of people
def total_people : ℕ := 25

-- Define the number of people who can speak Latin
def speak_latin : ℕ := 13

-- Define the number of people who can speak French
def speak_french : ℕ := 15

-- Define the number of people who can speak both Latin and French
def speak_both : ℕ := 9

-- Prove that the number of people who don't speak either Latin or French is 6
theorem people_speak_neither : (total_people - (speak_latin + speak_french - speak_both)) = 6 := by
  sorry

end NUMINAMATH_GPT_people_speak_neither_l1435_143567


namespace NUMINAMATH_GPT_monotonic_power_function_l1435_143500

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2 * a - 2) * x^a

theorem monotonic_power_function (a : ℝ) (h1 : ∀ x : ℝ, ( ∀ x1 x2 : ℝ, x1 < x2 → power_function a x1 < power_function a x2 ) )
  (h2 : a^2 - 2 * a - 2 = 1) (h3 : a > 0) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_power_function_l1435_143500


namespace NUMINAMATH_GPT_last_two_digits_2005_power_1989_l1435_143575

theorem last_two_digits_2005_power_1989 : (2005 ^ 1989) % 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_2005_power_1989_l1435_143575


namespace NUMINAMATH_GPT_berries_ratio_l1435_143577

theorem berries_ratio (total_berries : ℕ) (stacy_berries : ℕ) (ratio_stacy_steve : ℕ)
  (h_total : total_berries = 1100) (h_stacy : stacy_berries = 800)
  (h_ratio : stacy_berries = 4 * ratio_stacy_steve) :
  ratio_stacy_steve / (total_berries - stacy_berries - ratio_stacy_steve) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_berries_ratio_l1435_143577


namespace NUMINAMATH_GPT_smallest_positive_angle_l1435_143525

def coterminal_angle (θ : ℤ) : ℤ := θ % 360

theorem smallest_positive_angle (θ : ℤ) (hθ : θ % 360 ≠ 0) : 
  0 < coterminal_angle θ ∧ coterminal_angle θ = 158 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l1435_143525


namespace NUMINAMATH_GPT_logarithm_identity_l1435_143534

theorem logarithm_identity :
  1 / (Real.log 3 / Real.log 8 + 1) + 
  1 / (Real.log 2 / Real.log 12 + 1) + 
  1 / (Real.log 4 / Real.log 9 + 1) = 3 := 
by
  sorry

end NUMINAMATH_GPT_logarithm_identity_l1435_143534


namespace NUMINAMATH_GPT_probability_of_heads_on_999th_toss_l1435_143533

theorem probability_of_heads_on_999th_toss (fair_coin : Bool → ℝ) :
  (∀ (i : ℕ), fair_coin true = 1 / 2 ∧ fair_coin false = 1 / 2) →
  fair_coin true = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_heads_on_999th_toss_l1435_143533


namespace NUMINAMATH_GPT_smallest_composite_proof_l1435_143592

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end NUMINAMATH_GPT_smallest_composite_proof_l1435_143592


namespace NUMINAMATH_GPT_car_speed_constant_l1435_143583

theorem car_speed_constant (v : ℝ) : 
  (1 / (v / 3600) - 1 / (80 / 3600) = 2) → v = 3600 / 47 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_constant_l1435_143583


namespace NUMINAMATH_GPT_sum_mod_eleven_l1435_143552

variable (x y z : ℕ)

theorem sum_mod_eleven (h1 : (x * y * z) % 11 = 3)
                       (h2 : (7 * z) % 11 = 4)
                       (h3 : (9 * y) % 11 = (5 + y) % 11) :
                       (x + y + z) % 11 = 5 :=
sorry

end NUMINAMATH_GPT_sum_mod_eleven_l1435_143552


namespace NUMINAMATH_GPT_johns_total_amount_l1435_143585

def amount_from_grandpa : ℕ := 30
def multiplier : ℕ := 3
def amount_from_grandma : ℕ := amount_from_grandpa * multiplier
def total_amount : ℕ := amount_from_grandpa + amount_from_grandma

theorem johns_total_amount :
  total_amount = 120 :=
by
  sorry

end NUMINAMATH_GPT_johns_total_amount_l1435_143585


namespace NUMINAMATH_GPT_triangle_lengths_ce_l1435_143588

theorem triangle_lengths_ce (AE BE CE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) (h1 : angle_AEB = 30)
  (h2 : angle_BEC = 45) (h3 : angle_CED = 45) (h4 : AE = 30) (h5 : BE = AE / 2) (h6 : CE = BE) : CE = 15 :=
by sorry

end NUMINAMATH_GPT_triangle_lengths_ce_l1435_143588


namespace NUMINAMATH_GPT_complex_expression_equality_l1435_143591

-- Define the basic complex number properties and operations.
def i : ℂ := Complex.I -- Define the imaginary unit

theorem complex_expression_equality (a b : ℤ) :
  (3 - 4 * i) * ((-4 + 2 * i) ^ 2) = -28 - 96 * i :=
by
  -- Syntactical proof placeholders
  sorry

end NUMINAMATH_GPT_complex_expression_equality_l1435_143591


namespace NUMINAMATH_GPT_find_i_when_x_is_0_point3_l1435_143557

noncomputable def find_i (x : ℝ) (i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

theorem find_i_when_x_is_0_point3 : find_i 0.3 2.9993 :=
by
  sorry

end NUMINAMATH_GPT_find_i_when_x_is_0_point3_l1435_143557


namespace NUMINAMATH_GPT_probability_of_equal_numbers_when_throwing_two_fair_dice_l1435_143559

theorem probability_of_equal_numbers_when_throwing_two_fair_dice :
  let total_outcomes := 36
  let favorable_outcomes := 6
  favorable_outcomes / total_outcomes = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_equal_numbers_when_throwing_two_fair_dice_l1435_143559


namespace NUMINAMATH_GPT_number_of_possible_lists_l1435_143501

/-- 
Define the basic conditions: 
- 18 balls, numbered 1 through 18
- Selection process is repeated 4 times 
- Each selection is independent
- After each selection, the ball is replaced 
- We need to prove the total number of possible lists of four numbers 
--/
def number_of_balls : ℕ := 18
def selections : ℕ := 4

theorem number_of_possible_lists : (number_of_balls ^ selections) = 104976 := by
  sorry

end NUMINAMATH_GPT_number_of_possible_lists_l1435_143501


namespace NUMINAMATH_GPT_value_range_sin_neg_l1435_143513

theorem value_range_sin_neg (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) : 
  Set.Icc (-1) (Real.sqrt 2 / 2) ( - (Real.sin x) ) :=
sorry

end NUMINAMATH_GPT_value_range_sin_neg_l1435_143513


namespace NUMINAMATH_GPT_value_of_xyz_l1435_143509

open Real

theorem value_of_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := 
  sorry

end NUMINAMATH_GPT_value_of_xyz_l1435_143509


namespace NUMINAMATH_GPT_major_axis_of_ellipse_l1435_143518

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define the length of the major axis
def major_axis_length : ℝ := 8

-- The theorem to prove
theorem major_axis_of_ellipse : 
  (∀ x y : ℝ, ellipse_eq x y) → major_axis_length = 8 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_of_ellipse_l1435_143518


namespace NUMINAMATH_GPT_second_tree_ring_groups_l1435_143517

-- Definition of the problem conditions
def group_rings (fat thin : Nat) : Nat := fat + thin

-- Conditions
def FirstTreeRingGroups : Nat := 70
def RingsPerGroup : Nat := group_rings 2 4
def FirstTreeRings : Nat := FirstTreeRingGroups * RingsPerGroup
def AgeDifference : Nat := 180

-- Calculate the total number of rings in the second tree
def SecondTreeRings : Nat := FirstTreeRings - AgeDifference

-- Prove the number of ring groups in the second tree
theorem second_tree_ring_groups : SecondTreeRings / RingsPerGroup = 40 :=
by
  sorry

end NUMINAMATH_GPT_second_tree_ring_groups_l1435_143517


namespace NUMINAMATH_GPT_min_fence_dimensions_l1435_143519

theorem min_fence_dimensions (A : ℝ) (hA : A ≥ 800) (x : ℝ) (hx : 2 * x * x = A) : x = 20 ∧ 2 * x = 40 := by
  sorry

end NUMINAMATH_GPT_min_fence_dimensions_l1435_143519


namespace NUMINAMATH_GPT_andy_coats_l1435_143516

theorem andy_coats 
  (initial_minks : ℕ)
  (offspring_4_minks count_4_offspring : ℕ)
  (offspring_6_minks count_6_offspring : ℕ)
  (offspring_8_minks count_8_offspring : ℕ)
  (freed_percentage coat_requirement total_minks offspring_minks freed_minks remaining_minks coats : ℕ) :
  initial_minks = 30 ∧
  offspring_4_minks = 10 ∧ count_4_offspring = 4 ∧
  offspring_6_minks = 15 ∧ count_6_offspring = 6 ∧
  offspring_8_minks = 5 ∧ count_8_offspring = 8 ∧
  freed_percentage = 60 ∧ coat_requirement = 15 ∧
  total_minks = initial_minks + offspring_minks ∧
  offspring_minks = offspring_4_minks * count_4_offspring + offspring_6_minks * count_6_offspring + offspring_8_minks * count_8_offspring ∧
  freed_minks = total_minks * freed_percentage / 100 ∧
  remaining_minks = total_minks - freed_minks ∧
  coats = remaining_minks / coat_requirement →
  coats = 5 :=
sorry

end NUMINAMATH_GPT_andy_coats_l1435_143516


namespace NUMINAMATH_GPT_eq1_solution_eq2_solution_l1435_143554


-- Theorem for the first equation (4(x + 1)^2 - 25 = 0)
theorem eq1_solution (x : ℝ) : (4 * (x + 1)^2 - 25 = 0) ↔ (x = 3 / 2 ∨ x = -7 / 2) :=
by
  sorry

-- Theorem for the second equation ((x + 10)^3 = -125)
theorem eq2_solution (x : ℝ) : ((x + 10)^3 = -125) ↔ (x = -15) :=
by
  sorry

end NUMINAMATH_GPT_eq1_solution_eq2_solution_l1435_143554


namespace NUMINAMATH_GPT_first_person_days_l1435_143514

-- Define the condition that Tanya is 25% more efficient than the first person and that Tanya takes 12 days to do the work.
def tanya_more_efficient (x : ℕ) : Prop :=
  -- Efficiency relationship: tanya (12 days) = 3 days less than the first person
  12 = x - (x / 4)

-- Define the theorem that the first person takes 15 days to do the work
theorem first_person_days : ∃ x : ℕ, tanya_more_efficient x ∧ x = 15 := 
by
  sorry -- proof is not required

end NUMINAMATH_GPT_first_person_days_l1435_143514


namespace NUMINAMATH_GPT_tile_calc_proof_l1435_143560

noncomputable def total_tiles (length width : ℕ) : ℕ :=
  let border_tiles_length := (2 * (length - 4)) * 2
  let border_tiles_width := (2 * (width - 4)) * 2
  let total_border_tiles := (border_tiles_length + border_tiles_width) * 2 - 8
  let inner_length := (length - 4)
  let inner_width := (width - 4)
  let inner_area := inner_length * inner_width
  let inner_tiles := inner_area / 4
  total_border_tiles + inner_tiles

theorem tile_calc_proof :
  total_tiles 15 20 = 144 :=
by
  sorry

end NUMINAMATH_GPT_tile_calc_proof_l1435_143560


namespace NUMINAMATH_GPT_perfect_square_base9_last_digit_l1435_143546

-- We define the problem conditions
variable {b d f : ℕ} -- all variables are natural numbers
-- Condition 1: Base 9 representation of a perfect square
variable (n : ℕ) -- n is the perfect square number
variable (sqrt_n : ℕ) -- sqrt_n is the square root of n (so, n = sqrt_n^2)
variable (h1 : n = b * 9^3 + d * 9^2 + 4 * 9 + f)
variable (h2 : b ≠ 0)
-- The question becomes that the possible values of f are 0, 1, or 4
theorem perfect_square_base9_last_digit (h3 : n = sqrt_n^2) (hb : b ≠ 0) : 
  (f = 0) ∨ (f = 1) ∨ (f = 4) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_base9_last_digit_l1435_143546


namespace NUMINAMATH_GPT_complex_square_l1435_143536

-- Define z and the condition on i
def z := 5 + (6 * Complex.I)
axiom i_squared : Complex.I ^ 2 = -1

-- State the theorem to prove z^2 = -11 + 60i
theorem complex_square : z ^ 2 = -11 + (60 * Complex.I) := by {
  sorry
}

end NUMINAMATH_GPT_complex_square_l1435_143536


namespace NUMINAMATH_GPT_similar_triangles_area_ratio_l1435_143510

theorem similar_triangles_area_ratio (ratio_angles : ℕ) (area_larger : ℕ) (h_ratio : ratio_angles = 3) (h_area_larger : area_larger = 400) :
  ∃ area_smaller : ℕ, area_smaller = 36 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_area_ratio_l1435_143510


namespace NUMINAMATH_GPT_minimum_deposits_needed_l1435_143569

noncomputable def annual_salary_expense : ℝ := 100000
noncomputable def annual_fixed_expense : ℝ := 170000
noncomputable def interest_rate_paid : ℝ := 0.0225
noncomputable def interest_rate_earned : ℝ := 0.0405

theorem minimum_deposits_needed :
  ∃ (x : ℝ), 
    (interest_rate_earned * x = annual_salary_expense + annual_fixed_expense + interest_rate_paid * x) →
    x = 1500 :=
by
  sorry

end NUMINAMATH_GPT_minimum_deposits_needed_l1435_143569


namespace NUMINAMATH_GPT_exists_infinite_B_with_property_l1435_143527

-- Definition of the sequence A
def seqA (n : ℕ) : ℤ := 5 * n - 2

-- Definition of the sequence B with its general form
def seqB (k : ℕ) (d : ℤ) : ℤ := k * d + 7 - d

-- The proof problem statement
theorem exists_infinite_B_with_property :
  ∃ (B : ℕ → ℤ) (d : ℤ), B 1 = 7 ∧ 
  (∀ k, k > 1 → B k = B (k - 1) + d) ∧
  (∀ n : ℕ, ∃ (k : ℕ), seqB k d = seqA n) :=
sorry

end NUMINAMATH_GPT_exists_infinite_B_with_property_l1435_143527


namespace NUMINAMATH_GPT_min_value_expression_l1435_143595

variable {a b c : ℝ}

theorem min_value_expression (h1 : a < b) (h2 : a > 0) (h3 : b^2 - 4 * a * c ≤ 0) : 
  ∃ m : ℝ, m = 3 ∧ (∀ x : ℝ, ((a + b + c) / (b - a)) ≥ m) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l1435_143595


namespace NUMINAMATH_GPT_brother_combined_age_l1435_143545

-- Define the ages of the brothers as integers
variable (x y : ℕ)

-- Define the condition given in the problem
def combined_age_six_years_ago : Prop := (x - 6) + (y - 6) = 100

-- State the theorem to prove the current combined age
theorem brother_combined_age (h : combined_age_six_years_ago x y): x + y = 112 :=
  sorry

end NUMINAMATH_GPT_brother_combined_age_l1435_143545
