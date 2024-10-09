import Mathlib

namespace average_speed_remaining_l916_91663

theorem average_speed_remaining (D : ℝ) : 
    (0.4 * D / 40 + 0.6 * D / S) = D / 50 → S = 60 :=
by 
  sorry

end average_speed_remaining_l916_91663


namespace find_integers_with_conditions_l916_91683

theorem find_integers_with_conditions :
  ∃ a b c d : ℕ, (1 ≤ a) ∧ (1 ≤ b) ∧ (1 ≤ c) ∧ (1 ≤ d) ∧ a * b * c * d = 2002 ∧ a + b + c + d < 40 := sorry

end find_integers_with_conditions_l916_91683


namespace age_impossibility_l916_91653

/-
Problem statement:
Ann is 5 years older than Kristine.
Their current ages sum up to 24.
Prove that it's impossible for both their ages to be whole numbers.
-/

theorem age_impossibility 
  (K A : ℕ) -- Kristine's and Ann's ages are natural numbers
  (h1 : A = K + 5) -- Ann is 5 years older than Kristine
  (h2 : K + A = 24) -- their combined age is 24
  : false := sorry

end age_impossibility_l916_91653


namespace equilateral_triangle_side_length_l916_91689

theorem equilateral_triangle_side_length 
    (D A B C : ℝ × ℝ)
    (h_distances : dist D A = 2 ∧ dist D B = 3 ∧ dist D C = 5)
    (h_equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
    dist A B = Real.sqrt 19 :=
by
    sorry -- Proof to be filled

end equilateral_triangle_side_length_l916_91689


namespace length_AB_l916_91647

theorem length_AB 
  (P : ℝ × ℝ) 
  (hP : 3 * P.1 + 4 * P.2 + 8 = 0)
  (C : ℝ × ℝ := (1, 1))
  (A B : ℝ × ℝ)
  (hA : (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (3 * A.1 + 4 * A.2 + 8 ≠ 0))
  (hB : (B.1 - 1)^2 + (B.2 - 1)^2 = 1 ∧ (3 * B.1 + 4 * B.2 + 8 ≠ 0)) :
  dist A B = 4 * Real.sqrt 2 / 3 := sorry

end length_AB_l916_91647


namespace painting_prices_l916_91691

theorem painting_prices (P : ℝ) (h₀ : 55000 = 3.5 * P - 500) : 
  P = 15857.14 :=
by
  -- P represents the average price of the previous three paintings.
  -- Given the condition: 55000 = 3.5 * P - 500
  -- We need to prove: P = 15857.14
  sorry

end painting_prices_l916_91691


namespace arithmetic_geom_seq_S5_l916_91637

theorem arithmetic_geom_seq_S5 (a_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * (1/2))
  (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (1/2))
  (h_geom_seq : (a_n 2) * (a_n 14) = (a_n 6) ^ 2) :
  S_n 5 = 25 / 2 :=
by
  sorry

end arithmetic_geom_seq_S5_l916_91637


namespace cranberry_parts_l916_91656

theorem cranberry_parts (L C : ℕ) :
  L = 3 →
  L + C = 72 →
  C = L + 18 →
  C = 21 :=
by
  intros hL hSum hDiff
  sorry

end cranberry_parts_l916_91656


namespace least_positive_integer_condition_l916_91643

theorem least_positive_integer_condition :
  ∃ n > 1, (∀ k ∈ [3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) → n = 25201 := by
  sorry

end least_positive_integer_condition_l916_91643


namespace find_f_2017_l916_91682

noncomputable def f : ℝ → ℝ :=
sorry

axiom cond1 : ∀ x : ℝ, f (1 + x) + f (1 - x) = 0
axiom cond2 : ∀ x : ℝ, f (-x) = f x
axiom cond3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = 2^x - 1

theorem find_f_2017 : f 2017 = 1 :=
by
  sorry

end find_f_2017_l916_91682


namespace max_diagonal_intersections_l916_91604

theorem max_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
    ∃ k, k = n * (n - 1) * (n - 2) * (n - 3) / 24 :=
by
    sorry

end max_diagonal_intersections_l916_91604


namespace customer_wants_score_of_eggs_l916_91675

def Score := 20
def Dozen := 12

def options (n : Nat) : Prop :=
  n = Score ∨ n = 2 * Score ∨ n = 2 * Dozen ∨ n = 3 * Score

theorem customer_wants_score_of_eggs : 
  ∃ n, options n ∧ n = Score := 
by
  exists Score
  constructor
  apply Or.inl
  rfl
  rfl

end customer_wants_score_of_eggs_l916_91675


namespace meetings_percentage_l916_91666

def total_minutes_in_day (hours: ℕ): ℕ := hours * 60
def first_meeting_duration: ℕ := 60
def second_meeting_duration (first_meeting_duration: ℕ): ℕ := 3 * first_meeting_duration
def total_meeting_duration (first_meeting_duration: ℕ) (second_meeting_duration: ℕ): ℕ := first_meeting_duration + second_meeting_duration
def percentage_of_workday_spent_in_meetings (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ): ℚ := (total_meeting_duration / total_minutes_in_day) * 100

theorem meetings_percentage (hours: ℕ) (first_meeting_duration: ℕ) (second_meeting_duration: ℕ) (total_meeting_duration: ℕ) (total_minutes_in_day: ℕ) 
(h1: total_minutes_in_day = 600) 
(h2: first_meeting_duration = 60) 
(h3: second_meeting_duration = 180) 
(h4: total_meeting_duration = 240):
percentage_of_workday_spent_in_meetings total_meeting_duration total_minutes_in_day = 40 := by
  sorry

end meetings_percentage_l916_91666


namespace acid_percentage_in_original_mixture_l916_91644

theorem acid_percentage_in_original_mixture 
  {a w : ℕ} 
  (h1 : a / (a + w + 1) = 1 / 5) 
  (h2 : (a + 1) / (a + w + 2) = 1 / 3) : 
  a / (a + w) = 1 / 4 :=
sorry

end acid_percentage_in_original_mixture_l916_91644


namespace range_of_c_l916_91690

theorem range_of_c (a b c : ℝ) (h1 : 6 < a) (h2 : a < 10) (h3 : a / 2 ≤ b) (h4 : b ≤ 2 * a) (h5 : c = a + b) : 
  9 < c ∧ c < 30 :=
sorry

end range_of_c_l916_91690


namespace race_results_l916_91635

-- Competitor times in seconds
def time_A : ℕ := 40
def time_B : ℕ := 50
def time_C : ℕ := 55

-- Time difference calculations
def time_diff_AB := time_B - time_A
def time_diff_AC := time_C - time_A
def time_diff_BC := time_C - time_B

theorem race_results :
  time_diff_AB = 10 ∧ time_diff_AC = 15 ∧ time_diff_BC = 5 :=
by
  -- Placeholder for proof
  sorry

end race_results_l916_91635


namespace Jessica_cut_40_roses_l916_91605

-- Define the problem's conditions as variables
variables (initialVaseRoses : ℕ) (finalVaseRoses : ℕ) (rosesGivenToSarah : ℕ)

-- Define the number of roses Jessica cut from her garden
def rosesCutFromGarden (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) : ℕ :=
  (finalVaseRoses - initialVaseRoses) + rosesGivenToSarah

-- Problem statement: Prove Jessica cut 40 roses from her garden
theorem Jessica_cut_40_roses (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) :
  initialVaseRoses = 7 →
  finalVaseRoses = 37 →
  rosesGivenToSarah = 10 →
  rosesCutFromGarden initialVaseRoses finalVaseRoses rosesGivenToSarah = 40 :=
by
  intros h1 h2 h3
  sorry

end Jessica_cut_40_roses_l916_91605


namespace fourth_rectangle_area_l916_91699

-- Define the conditions and prove the area of the fourth rectangle
theorem fourth_rectangle_area (x y z w : ℝ)
  (h_xy : x * y = 24)
  (h_xw : x * w = 35)
  (h_zw : z * w = 42)
  (h_sum : x + z = 21) :
  y * w = 33.777 := 
sorry

end fourth_rectangle_area_l916_91699


namespace arithmetic_sequence_x_values_l916_91606

theorem arithmetic_sequence_x_values {x : ℝ} (h_nonzero : x ≠ 0) (h_arith_seq : ∃ (k : ℤ), x - k = 1/2 ∧ x + 1 - (k + 1) = (k + 1) - 1/2) (h_lt_four : x < 4) :
  x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5 :=
by
  sorry

end arithmetic_sequence_x_values_l916_91606


namespace find_a_l916_91696

variable (a b : ℝ × ℝ) 

axiom b_eq : b = (2, -1)
axiom length_eq_one : ‖a + b‖ = 1
axiom parallel_x_axis : (a + b).snd = 0

theorem find_a : a = (-1, 1) ∨ a = (-3, 1) := by
  sorry

end find_a_l916_91696


namespace equivalent_statement_l916_91652

theorem equivalent_statement (x y z w : ℝ)
  (h : (2 * x + y) / (y + z) = (z + w) / (w + 2 * x)) :
  (x = z / 2 ∨ 2 * x + y + z + w = 0) :=
sorry

end equivalent_statement_l916_91652


namespace find_sum_l916_91659

theorem find_sum (a b : ℝ) (ha : a^3 - 3 * a^2 + 5 * a - 17 = 0) (hb : b^3 - 3 * b^2 + 5 * b + 11 = 0) :
  a + b = 2 :=
sorry

end find_sum_l916_91659


namespace B_profit_l916_91632

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit_l916_91632


namespace boxes_per_day_l916_91621

theorem boxes_per_day (apples_per_box fewer_apples_per_day total_apples_two_weeks : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : fewer_apples_per_day = 500)
  (h3 : total_apples_two_weeks = 24500) :
  (∃ x : ℕ, (7 * apples_per_box * x + 7 * (apples_per_box * x - fewer_apples_per_day) = total_apples_two_weeks) ∧ x = 50) := 
sorry

end boxes_per_day_l916_91621


namespace correctPropositions_l916_91695

-- Define the conditions and statement as Lean structures.
structure Geometry :=
  (Line : Type)
  (Plane : Type)
  (parallel : Plane → Plane → Prop)
  (parallelLine : Line → Plane → Prop)
  (perpendicular : Plane → Plane → Prop)
  (perpendicularLine : Line → Plane → Prop)
  (subsetLine : Line → Plane → Prop)

-- Main theorem to be proved in Lean 4
theorem correctPropositions (G : Geometry) :
  (∀ (α β : G.Plane) (a : G.Line), (G.parallel α β) → (G.subsetLine a α) → (G.parallelLine a β)) ∧ 
  (∀ (α β : G.Plane) (a : G.Line), (G.perpendicularLine a α) → (G.perpendicularLine a β) → (G.parallel α β)) :=
sorry -- The proof is omitted, as per instructions

end correctPropositions_l916_91695


namespace money_left_in_wallet_l916_91602

def olivia_initial_money : ℕ := 54
def olivia_spent_money : ℕ := 25

theorem money_left_in_wallet : olivia_initial_money - olivia_spent_money = 29 :=
by
  sorry

end money_left_in_wallet_l916_91602


namespace solve_sine_equation_l916_91648

theorem solve_sine_equation (x : ℝ) (k : ℤ) (h : |Real.sin x| ≠ 1) :
  (8.477 * ((∑' n, Real.sin x ^ n) / (∑' n, ((-1 : ℝ) * Real.sin x) ^ n)) = 4 / (1 + Real.tan x ^ 2)) 
  ↔ (x = (-1)^k * (Real.pi / 6) + k * Real.pi) :=
by
  sorry

end solve_sine_equation_l916_91648


namespace number_of_moles_H2SO4_formed_l916_91655

-- Define the moles of reactants
def initial_moles_SO2 : ℕ := 1
def initial_moles_H2O2 : ℕ := 1

-- Given the balanced chemical reaction
-- SO2 + H2O2 → H2SO4
def balanced_reaction := (1, 1) -- Representing the reactant coefficients for SO2 and H2O2

-- Define the number of moles of product formed
def moles_H2SO4 (moles_SO2 moles_H2O2 : ℕ) : ℕ :=
moles_SO2 -- Since according to balanced equation, 1 mole of each reactant produces 1 mole of product

theorem number_of_moles_H2SO4_formed :
  moles_H2SO4 initial_moles_SO2 initial_moles_H2O2 = 1 := by
  sorry

end number_of_moles_H2SO4_formed_l916_91655


namespace range_of_m_l916_91688

def quadratic_function (m : ℝ) : Prop :=
  ∀ x : ℝ, (x^2 + m * x + 1 = 0 → false)

def ellipse_condition (m : ℝ) : Prop :=
  0 < m

theorem range_of_m (m : ℝ) :
  (quadratic_function m ∨ ellipse_condition m) ∧ ¬ (quadratic_function m ∧ ellipse_condition m) →
  m ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioi 2 :=
by
  sorry

end range_of_m_l916_91688


namespace problem_statement_l916_91645

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l916_91645


namespace smallest_solution_x_squared_abs_x_eq_3x_plus_4_l916_91674

theorem smallest_solution_x_squared_abs_x_eq_3x_plus_4 :
  ∃ x : ℝ, x^2 * |x| = 3 * x + 4 ∧ ∀ y : ℝ, (y^2 * |y| = 3 * y + 4 → y ≥ x) := 
sorry

end smallest_solution_x_squared_abs_x_eq_3x_plus_4_l916_91674


namespace find_number_l916_91673

theorem find_number (x n : ℝ) (h1 : 0.12 / x * n = 12) (h2 : x = 0.1) : n = 10 := by
  sorry

end find_number_l916_91673


namespace grayson_fraction_l916_91662

variable (A G O : ℕ) -- The number of boxes collected by Abigail, Grayson, and Olivia, respectively
variable (C_per_box : ℕ) -- The number of cookies per box
variable (TotalCookies : ℕ) -- The total number of cookies collected by Abigail, Grayson, and Olivia

-- Given conditions
def abigail_boxes : ℕ := 2
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48
def total_cookies : ℕ := 276

-- Prove the fraction of the box that Grayson collected
theorem grayson_fraction :
  G * C_per_box = TotalCookies - (abigail_boxes + olivia_boxes) * cookies_per_box → 
  G / C_per_box = 3 / 4 := 
by
  sorry

-- Assume the variables from conditions
variable (G : ℕ := 36 / 48)
variable (TotalCookies := 276)
variable (C_per_box := 48)
variable (A := 2)
variable (O := 3)


end grayson_fraction_l916_91662


namespace train_crossing_time_l916_91687

theorem train_crossing_time:
  ∀ (length_train : ℝ) (speed_man_kmph : ℝ) (speed_train_kmph : ℝ),
    length_train = 125 →
    speed_man_kmph = 5 →
    speed_train_kmph = 69.994 →
    (125 / ((69.994 + 5) * (1000 / 3600))) = 6.002 :=
by
  intros length_train speed_man_kmph speed_train_kmph h1 h2 h3
  sorry

end train_crossing_time_l916_91687


namespace february_first_is_friday_l916_91694

-- Definition of conditions
def february_has_n_mondays (n : ℕ) : Prop := n = 3
def february_has_n_fridays (n : ℕ) : Prop := n = 5

-- The statement to prove
theorem february_first_is_friday (n_mondays n_fridays : ℕ) (h_mondays : february_has_n_mondays n_mondays) (h_fridays : february_has_n_fridays n_fridays) : 
  (1 : ℕ) % 7 = 5 :=
by
  sorry

end february_first_is_friday_l916_91694


namespace product_of_x_y_l916_91672

theorem product_of_x_y (x y : ℝ) :
  (54 = 5 * y^2 + 20) →
  (8 * x^2 + 2 = 38) →
  x * y = Real.sqrt (30.6) :=
by
  intros h1 h2
  -- these would be the proof steps
  sorry

end product_of_x_y_l916_91672


namespace exists_unique_adjacent_sums_in_circle_l916_91638

theorem exists_unique_adjacent_sums_in_circle :
  ∃ (f : Fin 10 → Fin 11),
    (∀ (i j : Fin 10), i ≠ j → (f i + f (i + 1)) % 11 ≠ (f j + f (j + 1)) % 11) :=
sorry

end exists_unique_adjacent_sums_in_circle_l916_91638


namespace abs_ab_eq_2128_l916_91608

theorem abs_ab_eq_2128 (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ r s : ℤ, r ≠ s ∧ ∃ r' : ℤ, r' = r ∧ 
          (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a)) :
  |a * b| = 2128 :=
sorry

end abs_ab_eq_2128_l916_91608


namespace weekly_earnings_correct_l916_91677

-- Definitions based on the conditions
def hours_weekdays : Nat := 5 * 5
def hours_weekends : Nat := 3 * 2
def hourly_rate_weekday : Nat := 3
def hourly_rate_weekend : Nat := 3 * 2
def earnings_weekdays : Nat := hours_weekdays * hourly_rate_weekday
def earnings_weekends : Nat := hours_weekends * hourly_rate_weekend

-- The total weekly earnings Mitch gets
def weekly_earnings : Nat := earnings_weekdays + earnings_weekends

-- The theorem we need to prove:
theorem weekly_earnings_correct : weekly_earnings = 111 :=
by
  sorry

end weekly_earnings_correct_l916_91677


namespace g_at_5_l916_91692

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_at_5 : g 5 = 74 := 
by {
  sorry
}

end g_at_5_l916_91692


namespace inequality_solution_set_l916_91641

theorem inequality_solution_set (x : ℝ) : (x^2 ≥ 4) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l916_91641


namespace smallest_m_for_no_real_solution_l916_91617

theorem smallest_m_for_no_real_solution : 
  (∀ x : ℝ, ∀ m : ℝ, (m * x^2 - 3 * x + 1 = 0) → false) ↔ (m ≥ 3) :=
by
  sorry

end smallest_m_for_no_real_solution_l916_91617


namespace total_cost_l916_91668

def copper_pipe_length := 10
def plastic_pipe_length := 15
def copper_pipe_cost_per_meter := 5
def plastic_pipe_cost_per_meter := 3

theorem total_cost (h₁ : copper_pipe_length = 10)
                   (h₂ : plastic_pipe_length = 15)
                   (h₃ : copper_pipe_cost_per_meter = 5)
                   (h₄ : plastic_pipe_cost_per_meter = 3) :
  10 * 5 + 15 * 3 = 95 :=
by sorry

end total_cost_l916_91668


namespace meal_cost_is_correct_l916_91657

def samosa_quantity : ℕ := 3
def samosa_price : ℝ := 2
def pakora_quantity : ℕ := 4
def pakora_price : ℝ := 3
def mango_lassi_quantity : ℕ := 1
def mango_lassi_price : ℝ := 2
def biryani_quantity : ℕ := 2
def biryani_price : ℝ := 5.5
def naan_quantity : ℕ := 1
def naan_price : ℝ := 1.5

def tip_rate : ℝ := 0.18
def sales_tax_rate : ℝ := 0.07

noncomputable def total_meal_cost : ℝ :=
  let subtotal := (samosa_quantity * samosa_price) + (pakora_quantity * pakora_price) +
                  (mango_lassi_quantity * mango_lassi_price) + (biryani_quantity * biryani_price) +
                  (naan_quantity * naan_price)
  let sales_tax := subtotal * sales_tax_rate
  let total_before_tip := subtotal + sales_tax
  let tip := total_before_tip * tip_rate
  total_before_tip + tip

theorem meal_cost_is_correct : total_meal_cost = 41.04 := by
  sorry

end meal_cost_is_correct_l916_91657


namespace truck_travel_l916_91619

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l916_91619


namespace min_value_x_squared_plus_6x_l916_91631

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l916_91631


namespace quadratic_roots_in_range_l916_91651

theorem quadratic_roots_in_range (a : ℝ) (α β : ℝ)
  (h_eq : ∀ x : ℝ, x^2 + (a^2 + 1) * x + a - 2 = 0)
  (h_root1 : α > 1)
  (h_root2 : β < -1)
  (h_viete_sum : α + β = -(a^2 + 1))
  (h_viete_prod : α * β = a - 2) :
  0 < a ∧ a < 2 :=
  sorry

end quadratic_roots_in_range_l916_91651


namespace problem_solution_l916_91640

theorem problem_solution (y : Fin 8 → ℝ)
  (h1 : y 0 + 4 * y 1 + 9 * y 2 + 16 * y 3 + 25 * y 4 + 36 * y 5 + 49 * y 6 + 64 * y 7 = 2)
  (h2 : 4 * y 0 + 9 * y 1 + 16 * y 2 + 25 * y 3 + 36 * y 4 + 49 * y 5 + 64 * y 6 + 81 * y 7 = 15)
  (h3 : 9 * y 0 + 16 * y 1 + 25 * y 2 + 36 * y 3 + 49 * y 4 + 64 * y 5 + 81 * y 6 + 100 * y 7 = 156)
  (h4 : 16 * y 0 + 25 * y 1 + 36 * y 2 + 49 * y 3 + 64 * y 4 + 81 * y 5 + 100 * y 6 + 121 * y 7 = 1305) :
  25 * y 0 + 36 * y 1 + 49 * y 2 + 64 * y 3 + 81 * y 4 + 100 * y 5 + 121 * y 6 + 144 * y 7 = 4360 :=
sorry

end problem_solution_l916_91640


namespace weight_of_5_diamonds_l916_91613

-- Define the weight of one diamond and one jade
variables (D J : ℝ)

-- Conditions:
-- 1. Total weight of 4 diamonds and 2 jades
def condition1 : Prop := 4 * D + 2 * J = 140
-- 2. A jade is 10 g heavier than a diamond
def condition2 : Prop := J = D + 10

-- Total weight of 5 diamonds
def total_weight_of_5_diamonds : ℝ := 5 * D

-- Theorem: Prove that the total weight of 5 diamonds is 100 g
theorem weight_of_5_diamonds (h1 : condition1 D J) (h2 : condition2 D J) : total_weight_of_5_diamonds D = 100 :=
by {
  sorry
}

end weight_of_5_diamonds_l916_91613


namespace power_rule_for_fractions_calculate_fraction_l916_91610

theorem power_rule_for_fractions (a b : ℚ) (n : ℕ) : (a / b)^n = (a^n) / (b^n) := 
by sorry

theorem calculate_fraction (a b n : ℕ) (h : a = 3 ∧ b = 5 ∧ n = 3) : (a / b)^n = 27 / 125 :=
by
  obtain ⟨ha, hb, hn⟩ := h
  simp [ha, hb, hn, power_rule_for_fractions (3 : ℚ) (5 : ℚ) 3]

end power_rule_for_fractions_calculate_fraction_l916_91610


namespace chicken_nuggets_cost_l916_91623

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l916_91623


namespace range_a_real_numbers_l916_91639

theorem range_a_real_numbers (x a : ℝ) : 
  (∀ x : ℝ, (x - a) * (1 - (x + a)) < 1) → (a ∈ Set.univ) :=
by
  sorry

end range_a_real_numbers_l916_91639


namespace solution_set_inequality_l916_91612

variable (f : ℝ → ℝ)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

def condition_x_f_prime (f f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x^2 * f' x > 2 * x * f (-x)

-- Main theorem to prove the solution set of inequality
theorem solution_set_inequality (f' : ℝ → ℝ) :
  is_odd_function f →
  derivative f f' →
  condition_x_f_prime f f' →
  ∀ x : ℝ, x^2 * f x < (3 * x - 1)^2 * f (1 - 3 * x) → x < (1 / 4) := 
  by
    intros h_odd h_deriv h_cond x h_ineq
    sorry

end solution_set_inequality_l916_91612


namespace wednesday_more_than_tuesday_l916_91670

noncomputable def monday_minutes : ℕ := 450

noncomputable def tuesday_minutes : ℕ := monday_minutes / 2

noncomputable def wednesday_minutes : ℕ := 300

theorem wednesday_more_than_tuesday : wednesday_minutes - tuesday_minutes = 75 :=
by
  sorry

end wednesday_more_than_tuesday_l916_91670


namespace initial_percentage_reduction_l916_91686

theorem initial_percentage_reduction (x : ℝ) :
  (1 - x / 100) * 1.17649 = 1 → x = 15 :=
by
  sorry

end initial_percentage_reduction_l916_91686


namespace solution_set_of_inequality_system_l916_91625

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_of_inequality_system_l916_91625


namespace construct_inaccessible_angle_bisector_l916_91600

-- Definitions for problem context
structure Point :=
  (x y : ℝ)

structure Line :=
  (p1 p2 : Point)

structure Angle := 
  (vertex : Point)
  (ray1 ray2 : Line)

-- Predicate to determine if a line bisects an angle
def IsAngleBisector (L : Line) (A : Angle) : Prop := sorry

-- The inaccessible vertex angle we are considering
-- Let's assume the vertex is defined but we cannot access it physically in constructions
noncomputable def inaccessible_angle : Angle := sorry

-- Statement to prove: Construct a line that bisects the inaccessible angle
theorem construct_inaccessible_angle_bisector :
  ∃ L : Line, IsAngleBisector L inaccessible_angle :=
sorry

end construct_inaccessible_angle_bisector_l916_91600


namespace max_value_n_for_positive_an_l916_91611

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a d : ℤ) (n : ℤ) := a + (n - 1) * d

-- Define the sum of first n terms of an arithmetic sequence
noncomputable def sum_arith_seq (a d n : ℤ) := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
axiom S15_pos (a d : ℤ) : sum_arith_seq a d 15 > 0
axiom S16_neg (a d : ℤ) : sum_arith_seq a d 16 < 0

-- Proof problem
theorem max_value_n_for_positive_an (a d : ℤ) :
  ∃ n : ℤ, n = 8 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 8) → arithmetic_seq a d m > 0 :=
sorry

end max_value_n_for_positive_an_l916_91611


namespace correlation_coefficient_is_one_l916_91628

noncomputable def correlation_coefficient (x_vals y_vals : List ℝ) : ℝ := sorry

theorem correlation_coefficient_is_one 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h3 : ∀ i, y i = 3 * x i + 1) : 
  correlation_coefficient (List.ofFn x) (List.ofFn y) = 1 := 
sorry

end correlation_coefficient_is_one_l916_91628


namespace earphone_cost_correct_l916_91665

-- Given conditions
def mean_expenditure : ℕ := 500

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thu : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300

def pen_cost : ℕ := 30
def notebook_cost : ℕ := 50

-- Goal: cost of the earphone
def total_expenditure_week : ℕ := 7 * mean_expenditure
def expenditure_6days : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thu + expenditure_sat + expenditure_sun
def expenditure_fri : ℕ := total_expenditure_week - expenditure_6days
def expenditure_fri_items : ℕ := pen_cost + notebook_cost
def earphone_cost : ℕ := expenditure_fri - expenditure_fri_items

theorem earphone_cost_correct :
  earphone_cost = 620 :=
by
  sorry

end earphone_cost_correct_l916_91665


namespace gcd_exponent_min_speed_for_meeting_game_probability_difference_l916_91669

-- Problem p4
theorem gcd_exponent (a b : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : gcd a b = 3) : gcd (2^a - 1) (2^b - 1) = 7 := by
  sorry

-- Problem p5
theorem min_speed_for_meeting (v_S s : ℚ) (h : v_S = 1/2) : ∀ (s : ℚ), (s - v_S) ≥ 1 → s = 3/2 := by
  sorry

-- Problem p6
theorem game_probability_difference (N : ℕ) (p : ℚ) (h1 : N = 1) (h2 : p = 5/16) : N + p = 21/16 := by
  sorry

end gcd_exponent_min_speed_for_meeting_game_probability_difference_l916_91669


namespace distance_between_houses_l916_91618

theorem distance_between_houses
  (alice_speed : ℕ) (bob_speed : ℕ) (alice_distance : ℕ) 
  (alice_walk_time : ℕ) (bob_walk_time : ℕ)
  (alice_start : ℕ) (bob_start : ℕ)
  (bob_start_after_alice : bob_start = alice_start + 1)
  (alice_speed_eq : alice_speed = 5)
  (bob_speed_eq : bob_speed = 4)
  (alice_distance_eq : alice_distance = 25)
  (alice_walk_time_eq : alice_walk_time = alice_distance / alice_speed)
  (bob_walk_time_eq : bob_walk_time = alice_walk_time - 1)
  (bob_distance_eq : bob_walk_time = bob_walk_time * bob_speed)
  (total_distance : ℕ)
  (total_distance_eq : total_distance = alice_distance + bob_distance) :
  total_distance = 41 :=
by sorry

end distance_between_houses_l916_91618


namespace retailer_selling_price_l916_91684

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end retailer_selling_price_l916_91684


namespace part1_case1_part1_case2_part1_case3_part2_l916_91614

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end part1_case1_part1_case2_part1_case3_part2_l916_91614


namespace value_of_k_l916_91654

theorem value_of_k (k : ℝ) :
  ∃ (k : ℝ), k ≠ 1 ∧ (k-1) * (0 : ℝ)^2 + 6 * (0 : ℝ) + k^2 - 1 = 0 ∧ k = -1 :=
by
  sorry

end value_of_k_l916_91654


namespace hypotenuse_is_18_8_l916_91620

def right_triangle_hypotenuse_perimeter_area (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2 * a * b = 24) ∧ (a^2 + b^2 = c^2)

theorem hypotenuse_is_18_8 : ∃ (a b c : ℝ), right_triangle_hypotenuse_perimeter_area a b c ∧ c = 18.8 :=
by
  sorry

end hypotenuse_is_18_8_l916_91620


namespace student_ages_inconsistent_l916_91607

theorem student_ages_inconsistent :
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  total_age_all_students < total_age_group1 + total_age_group2 + total_age_group3 :=
by {
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  have h₁ : total_age_all_students = 24 * 18 := rfl
  have h₂ : total_age_group1 = 6 * 16 := rfl
  have h₃ : total_age_group2 = 10 * 20 := rfl
  have h₄ : total_age_group3 = 7 * 22 := rfl
  have h₅ : 432 = 24 * 18 := by norm_num
  have h₆ : 96 = 6 * 16 := by norm_num
  have h₇ : 200 = 10 * 20 := by norm_num
  have h₈ : 154 = 7 * 22 := by norm_num
  have h₉ : 432 < 96 + 200 + 154 := by norm_num
  exact h₉
}

end student_ages_inconsistent_l916_91607


namespace reporters_not_covering_politics_l916_91603

def total_reporters : ℝ := 8000
def politics_local : ℝ := 0.12 + 0.08 + 0.08 + 0.07 + 0.06 + 0.05 + 0.04 + 0.03 + 0.02 + 0.01
def politics_non_local : ℝ := 0.15
def politics_total : ℝ := politics_local + politics_non_local

theorem reporters_not_covering_politics :
  1 - politics_total = 0.29 :=
by
  -- Required definition and intermediate proof steps.
  sorry

end reporters_not_covering_politics_l916_91603


namespace triangle_is_isosceles_l916_91658

theorem triangle_is_isosceles (A B C a b c : ℝ) (h1 : c = 2 * a * Real.cos B) : 
  A = B → a = b := 
sorry

end triangle_is_isosceles_l916_91658


namespace plus_minus_pairs_l916_91649

theorem plus_minus_pairs (a b p q : ℕ) (h_plus_pairs : p = a) (h_minus_pairs : q = b) : 
  a - b = p - q := 
by 
  sorry

end plus_minus_pairs_l916_91649


namespace lizzy_wealth_after_loan_l916_91646

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l916_91646


namespace lcm_gcd_product_l916_91626

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l916_91626


namespace intersection_empty_l916_91627

def setA : Set ℝ := { x | x^2 - 2 * x > 0 }
def setB : Set ℝ := { x | |x + 1| < 0 }

theorem intersection_empty : setA ∩ setB = ∅ :=
by
  sorry

end intersection_empty_l916_91627


namespace range_of_a_l916_91630

open Real

theorem range_of_a (a : ℝ) :
  (∀ x > 0, ae^x + x + x * log x ≥ x^2) → a ≥ 1 / exp 2 :=
sorry

end range_of_a_l916_91630


namespace main_theorem_l916_91664

variable {a b c : ℝ}

noncomputable def inequality_1 (a b c : ℝ) : Prop :=
  a + b + c ≤ (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b)

noncomputable def inequality_2 (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≤ a^2 / (b * c) + b^2 / (c * a) + c^2 / (a * b)

theorem main_theorem (h : a < 0 ∧ b < 0 ∧ c < 0) :
  inequality_1 a b c ∧ inequality_2 a b c := by sorry

end main_theorem_l916_91664


namespace max_n_for_factorable_quadratic_l916_91697

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l916_91697


namespace distance_AC_l916_91671

theorem distance_AC (A B C : ℤ) (h₁ : abs (B - A) = 5) (h₂ : abs (C - B) = 3) : abs (C - A) = 2 ∨ abs (C - A) = 8 :=
sorry

end distance_AC_l916_91671


namespace find_a_n_l916_91615

variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1_eq : a 1 = 1
axiom rec_relation : ∀ n, a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1)) ^ 2

theorem find_a_n : ∀ n, a n = 1 / n := by
  sorry

end find_a_n_l916_91615


namespace find_x_l916_91667

def cube_volume (s : ℝ) := s^3
def cube_surface_area (s : ℝ) := 6 * s^2

theorem find_x (x : ℝ) (s : ℝ) 
  (hv : cube_volume s = 7 * x)
  (hs : cube_surface_area s = x) : 
  x = 42 := 
by
  sorry

end find_x_l916_91667


namespace part1_part2_l916_91634

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 2 - Real.sqrt 3

theorem part1 : a * b = 1 := 
by 
  unfold a b
  sorry

theorem part2 : a^2 + b^2 - a * b = 13 :=
by 
  unfold a b
  sorry

end part1_part2_l916_91634


namespace find_n_l916_91636

theorem find_n (n : ℕ) (h : Nat.lcm n (n - 30) = n + 1320) : n = 165 := 
sorry

end find_n_l916_91636


namespace unique_positive_real_b_l916_91698

noncomputable def is_am_gm_satisfied (r s t : ℝ) (a : ℝ) : Prop :=
  r + s + t = 2 * a ∧ r * s * t = 2 * a ∧ (r+s+t)/3 = ((r * s * t) ^ (1/3))

noncomputable def poly_roots_real (r s t : ℝ) : Prop :=
  ∀ x : ℝ, (x = r ∨ x = s ∨ x = t)

theorem unique_positive_real_b :
  ∃ b a : ℝ, 0 < a ∧ 0 < b ∧
  (∃ r s t : ℝ, (r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ poly_roots_real r s t) ∧
   is_am_gm_satisfied r s t a ∧
   (x^3 - 2*a*x^2 + b*x - 2*a = (x - r) * (x - s) * (x - t)) ∧
   b = 9) := sorry

end unique_positive_real_b_l916_91698


namespace quadratic_inequality_solution_l916_91622

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x - 1) * x ≥ 2) ↔ (x ≤ -1 ∨ x ≥ 2) := 
sorry

end quadratic_inequality_solution_l916_91622


namespace part1_solution_set_part2_range_of_a_l916_91660

def f (x a : ℝ) : ℝ := abs (x - a^2) + abs (x - 2 * a + 1)

theorem part1_solution_set (x : ℝ) : 
  (f x 2 ≥ 4) ↔ (x ≤ 3 / 2 ∨ x ≥ 11 / 2) :=
sorry

theorem part2_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 4) → (a ≤ -1 ∨ a ≥ 3) :=
sorry

end part1_solution_set_part2_range_of_a_l916_91660


namespace domain_implies_range_a_range_implies_range_a_l916_91678

theorem domain_implies_range_a {a : ℝ} :
  (∀ x : ℝ, ax^2 + 2 * a * x + 1 > 0) → 0 ≤ a ∧ a < 1 :=
sorry

theorem range_implies_range_a {a : ℝ} :
  (∀ y : ℝ, ∃ x : ℝ, ax^2 + 2 * a * x + 1 = y) → 1 ≤ a :=
sorry

end domain_implies_range_a_range_implies_range_a_l916_91678


namespace prove_inequalities_l916_91679

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logb 5 2
noncomputable def z := Real.exp (-1 / 2)

theorem prove_inequalities : y < z ∧ z < x := by
  unfold x y z
  sorry

end prove_inequalities_l916_91679


namespace quadratic_properties_l916_91661

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties 
  (a b c : ℝ) (ha : a ≠ 0) (h_passes_through : quadratic_function a b c 0 = 1) (h_unique_zero : quadratic_function a b c (-1) = 0) :
  quadratic_function a b c = quadratic_function 1 2 1 ∧ 
  (∀ k, ∃ g,
    (k ≤ -2 → g = k + 3) ∧ 
    (-2 < k ∧ k ≤ 6 → g = -((k^2 - 4*k) / 4)) ∧ 
    (6 < k → g = 9 - 2*k)) :=
sorry

end quadratic_properties_l916_91661


namespace derivative_at_pi_over_3_l916_91685

noncomputable def f (x : Real) : Real := 2 * Real.sin x + Real.sqrt 3 * Real.cos x

theorem derivative_at_pi_over_3 : (deriv f) (π / 3) = -1 / 2 := 
by
  sorry

end derivative_at_pi_over_3_l916_91685


namespace find_product_l916_91642

theorem find_product (a b c d : ℚ) 
  (h₁ : 2 * a + 4 * b + 6 * c + 8 * d = 48)
  (h₂ : 4 * (d + c) = b)
  (h₃ : 4 * b + 2 * c = a)
  (h₄ : c + 1 = d) :
  a * b * c * d = -319603200 / 10503489 := sorry

end find_product_l916_91642


namespace g_increasing_g_multiplicative_g_special_case_g_18_value_l916_91629

def g (n : ℕ) : ℕ :=
sorry

theorem g_increasing : ∀ n : ℕ, n > 0 → g (n + 1) > g n :=
sorry

theorem g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n :=
sorry

theorem g_special_case : ∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → m ^ n = n ^ m → g m = n ∨ g n = m :=
sorry

theorem g_18_value : g 18 = 324 :=
sorry

end g_increasing_g_multiplicative_g_special_case_g_18_value_l916_91629


namespace oil_leakage_calculation_l916_91676

def total_oil_leaked : ℕ := 11687
def oil_leaked_while_worked : ℕ := 5165
def oil_leaked_before_work : ℕ := 6522

theorem oil_leakage_calculation :
  oil_leaked_before_work = total_oil_leaked - oil_leaked_while_work :=
sorry

end oil_leakage_calculation_l916_91676


namespace evaluate_f_x_l916_91650

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 2 * x^2 + 4 * x

theorem evaluate_f_x : f 3 - f (-3) = 672 :=
by
  -- Proof omitted
  sorry

end evaluate_f_x_l916_91650


namespace trip_duration_is_6_hours_l916_91633

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours_l916_91633


namespace fraction_ordering_l916_91624

theorem fraction_ordering :
  (4 / 13) < (12 / 37) ∧ (12 / 37) < (15 / 31) ∧ (4 / 13) < (15 / 31) :=
by sorry

end fraction_ordering_l916_91624


namespace factorize_cubic_expression_l916_91680

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l916_91680


namespace find_b_l916_91601

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 :=
by
  sorry

end find_b_l916_91601


namespace base_r_5555_square_palindrome_l916_91609

theorem base_r_5555_square_palindrome (r : ℕ) (a b c d : ℕ) 
  (h1 : r % 2 = 0) 
  (h2 : r >= 18) 
  (h3 : d - c = 2)
  (h4 : ∀ x, (x = 5 * r^3 + 5 * r^2 + 5 * r + 5) → 
    (x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)) : 
  r = 24 := 
sorry

end base_r_5555_square_palindrome_l916_91609


namespace remainder_of_m_div_5_l916_91681

theorem remainder_of_m_div_5 (m n : ℕ) (hpos : 0 < m) (hdef : m = 15 * n - 1) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l916_91681


namespace sum_of_squares_l916_91616

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 28) : x^2 + y^2 = 200 :=
by
  sorry

end sum_of_squares_l916_91616


namespace smallest_root_of_polynomial_l916_91693

theorem smallest_root_of_polynomial :
  ∃ x : ℝ, (24 * x^3 - 106 * x^2 + 116 * x - 70 = 0) ∧ x = 0.67 :=
by
  sorry

end smallest_root_of_polynomial_l916_91693
