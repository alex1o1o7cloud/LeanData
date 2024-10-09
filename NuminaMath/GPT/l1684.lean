import Mathlib

namespace red_paint_intensity_l1684_168481

theorem red_paint_intensity (x : ℝ) (h1 : 0.5 * 10 + 0.5 * x = 15) : x = 20 :=
sorry

end red_paint_intensity_l1684_168481


namespace pentagon_area_l1684_168412

-- Define the lengths of the sides of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 25

-- Define the sides of the rectangle and triangle
def rectangle_length := side4
def rectangle_width := side2
def triangle_base := side1
def triangle_height := rectangle_width

-- Define areas of rectangle and right triangle
def area_rectangle := rectangle_length * rectangle_width
def area_triangle := (triangle_base * triangle_height) / 2

-- Define the total area of the pentagon
def total_area_pentagon := area_rectangle + area_triangle

theorem pentagon_area : total_area_pentagon = 925 := by
  sorry

end pentagon_area_l1684_168412


namespace max_value_of_expression_l1684_168445

theorem max_value_of_expression (x y : ℝ) 
  (h : (x - 4)^2 / 4 + y^2 / 9 = 1) : 
  (x^2 / 4 + y^2 / 9 ≤ 9) ∧ ∃ x y, (x - 4)^2 / 4 + y^2 / 9 = 1 ∧ x^2 / 4 + y^2 / 9 = 9 :=
by
  sorry

end max_value_of_expression_l1684_168445


namespace min_value_of_u_l1684_168424

theorem min_value_of_u : ∀ (x y : ℝ), x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → x * y = -1 → 
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ 12 / 5) :=
by
  intros x y hx hy hxy u hu
  sorry

end min_value_of_u_l1684_168424


namespace pen_more_expensive_than_two_notebooks_l1684_168401

variable (T R C : ℝ)

-- Conditions
axiom cond1 : T + R + C = 120
axiom cond2 : 5 * T + 2 * R + 3 * C = 350

-- Theorem statement
theorem pen_more_expensive_than_two_notebooks :
  R > 2 * T :=
by
  -- omit the actual proof, but check statement correctness
  sorry

end pen_more_expensive_than_two_notebooks_l1684_168401


namespace problem_statement_l1684_168405

theorem problem_statement :
  (¬ (∀ x : ℝ, 2 * x < 3 * x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := 
sorry

end problem_statement_l1684_168405


namespace train_passes_jogger_in_46_seconds_l1684_168449

-- Definitions directly from conditions
def jogger_speed_kmh : ℕ := 10
def train_speed_kmh : ℕ := 46
def initial_distance_m : ℕ := 340
def train_length_m : ℕ := 120

-- Additional computed definitions based on conditions
def relative_speed_ms : ℕ := (train_speed_kmh - jogger_speed_kmh) * 1000 / 3600
def total_distance_m : ℕ := initial_distance_m + train_length_m

-- Prove that the time it takes for the train to pass the jogger is 46 seconds
theorem train_passes_jogger_in_46_seconds : total_distance_m / relative_speed_ms = 46 := by
  sorry

end train_passes_jogger_in_46_seconds_l1684_168449


namespace shares_correct_l1684_168493

open Real

-- Problem setup
def original_problem (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 1020 ∧
  a = (3 / 4) * b ∧
  b = (2 / 3) * c ∧
  c = (1 / 4) * d ∧
  d = (5 / 6) * e

-- Goal
theorem shares_correct : ∃ (a b c d e : ℝ),
  original_problem a b c d e ∧
  abs (a - 58.17) < 0.01 ∧
  abs (b - 77.56) < 0.01 ∧
  abs (c - 116.34) < 0.01 ∧
  abs (d - 349.02) < 0.01 ∧
  abs (e - 419.42) < 0.01 := by
  sorry

end shares_correct_l1684_168493


namespace min_max_value_sum_l1684_168478

variable (a b c d e : ℝ)

theorem min_max_value_sum :
  a + b + c + d + e = 10 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 30 →
  let expr := 5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)
  let m := 42
  let M := 52
  m + M = 94 := sorry

end min_max_value_sum_l1684_168478


namespace only_real_solution_x_eq_6_l1684_168443

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end only_real_solution_x_eq_6_l1684_168443


namespace angle_T_in_pentagon_l1684_168450

theorem angle_T_in_pentagon (P Q R S T : ℝ) 
  (h1 : P = R) (h2 : P = T) (h3 : Q + S = 180) 
  (h4 : P + Q + R + S + T = 540) : T = 120 :=
by
  sorry

end angle_T_in_pentagon_l1684_168450


namespace face_card_then_number_card_prob_l1684_168408

-- Definitions from conditions
def num_cards := 52
def num_face_cards := 12
def num_number_cards := 40
def total_ways_to_pick_two_cards := 52 * 51

-- Theorem statement
theorem face_card_then_number_card_prob : 
  (num_face_cards * num_number_cards) / total_ways_to_pick_two_cards = (40 : ℚ) / 221 :=
by
  sorry

end face_card_then_number_card_prob_l1684_168408


namespace average_weight_l1684_168437

variable (A B C : ℕ)

theorem average_weight (h1 : A + B = 140) (h2 : B + C = 100) (h3 : B = 60) :
  (A + B + C) / 3 = 60 := 
sorry

end average_weight_l1684_168437


namespace chess_group_players_l1684_168406

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
by
  sorry

end chess_group_players_l1684_168406


namespace apple_distribution_l1684_168422

theorem apple_distribution : 
  (∀ (a b c d : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → (a + b + c + d = 30) → 
  ∃ k : ℕ, k = (Nat.choose 29 3) ∧ k = 3276) :=
by
  intros a b c d h_pos h_sum
  use Nat.choose 29 3
  have h_eq : Nat.choose 29 3 = 3276 := by sorry
  exact ⟨rfl, h_eq⟩

end apple_distribution_l1684_168422


namespace q_value_l1684_168457

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end q_value_l1684_168457


namespace problem_statement_l1684_168456

variable (P : ℕ → Prop)

theorem problem_statement
    (h1 : P 2)
    (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 2)) :
    ∀ n : ℕ, n > 0 → 2 ∣ n → P n :=
by
  sorry

end problem_statement_l1684_168456


namespace john_tran_probability_2_9_l1684_168482

def johnArrivalProbability (train_start train_end john_min john_max: ℕ) : ℚ := 
  let overlap_area := ((train_end - train_start - 15) * 15) / 2 
  let total_area := (john_max - john_min) * (train_end - train_start)
  overlap_area / total_area

theorem john_tran_probability_2_9 :
  johnArrivalProbability 30 90 0 90 = 2 / 9 := by
  sorry

end john_tran_probability_2_9_l1684_168482


namespace square_side_length_l1684_168476

theorem square_side_length (x S : ℕ) (h1 : S > 0) (h2 : x = 4) (h3 : 4 * S = 6 * x) : S = 6 := by
  subst h2
  sorry

end square_side_length_l1684_168476


namespace range_of_a_l1684_168490

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) → ((x + a) * (x + 1) > 0)) ∧ 
  (∃ x : ℝ, ¬(x ∈ (Set.Iio (-1) ∪ Set.Ioi 3)) ∧ ((x + a) * (x + 1) > 0)) → 
  a ∈ Set.Iio (-3) := 
  sorry

end range_of_a_l1684_168490


namespace equation_of_line_m_l1684_168438

-- Given conditions
def point (α : Type*) := α × α

def l_eq (p : point ℝ) : Prop := p.1 + 3 * p.2 = 7 -- Equation of line l
def m_intercept : point ℝ := (1, 2) -- Intersection point of l and m
def q : point ℝ := (2, 5) -- Point Q
def q'' : point ℝ := (5, 0) -- Point Q''

-- Proving the equation of line m
theorem equation_of_line_m (m_eq : point ℝ → Prop) :
  (∀ P : point ℝ, m_eq P ↔ P.2 = 2 * P.1 - 2) ↔
  (∃ P : point ℝ, m_eq P ∧ P = (5, 0)) :=
sorry

end equation_of_line_m_l1684_168438


namespace find_y_l1684_168487

noncomputable def x : ℝ := 3.3333333333333335

theorem find_y (y x: ℝ) (h1: x = 3.3333333333333335) (h2: x * 10 / y = x^2) :
  y = 3 :=
by
  sorry

end find_y_l1684_168487


namespace units_digit_of_j_squared_plus_3_power_j_l1684_168462

def j : ℕ := 2023^3 + 3^2023 + 2023

theorem units_digit_of_j_squared_plus_3_power_j (j : ℕ) (h : j = 2023^3 + 3^2023 + 2023) : 
  ((j^2 + 3^j) % 10) = 6 := 
  sorry

end units_digit_of_j_squared_plus_3_power_j_l1684_168462


namespace weaving_sum_first_seven_days_l1684_168433

noncomputable def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

theorem weaving_sum_first_seven_days
  (a_1 d : ℕ) :
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) = 9 →
  (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 6) = 15 →
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) +
  (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 5) +
  (arithmetic_sequence a_1 d 6) + (arithmetic_sequence a_1 d 7) = 35 := by
  sorry

end weaving_sum_first_seven_days_l1684_168433


namespace positive_difference_l1684_168491

theorem positive_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 3 * x = 27) : y - x = 9 :=
sorry

end positive_difference_l1684_168491


namespace a_beats_b_time_difference_l1684_168414

theorem a_beats_b_time_difference
  (d : ℝ) (d_A : ℝ) (d_B : ℝ)
  (t_A : ℝ)
  (h1 : d = 1000)
  (h2 : d_A = d)
  (h3 : d_B = d - 60)
  (h4 : t_A = 235) :
  (t_A - (d_B * t_A / d_A)) = 14.1 :=
by sorry

end a_beats_b_time_difference_l1684_168414


namespace cost_of_dozen_pens_l1684_168447

-- Define the costs and conditions as given in the problem.
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x

-- The given conditions transformed into Lean definitions.
def condition1 (x : ℝ) : Prop := 3 * cost_of_pen x + 5 * cost_of_pencil x = 100
def condition2 (x : ℝ) : Prop := cost_of_pen x / cost_of_pencil x = 5

-- Prove that the cost of one dozen pens is Rs. 300.
theorem cost_of_dozen_pens : ∃ x : ℝ, condition1 x ∧ condition2 x ∧ 12 * cost_of_pen x = 300 := by
  sorry

end cost_of_dozen_pens_l1684_168447


namespace equation_graph_is_ellipse_l1684_168409

theorem equation_graph_is_ellipse :
  ∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * (x^2 - 72 * y^2) + a * x + d = a * c * (y - 6)^2 :=
sorry

end equation_graph_is_ellipse_l1684_168409


namespace pieces_per_serving_l1684_168461

-- Definitions based on conditions
def jaredPopcorn : Nat := 90
def friendPopcorn : Nat := 60
def numberOfFriends : Nat := 3
def totalServings : Nat := 9

-- Statement to verify
theorem pieces_per_serving : 
  ((jaredPopcorn + numberOfFriends * friendPopcorn) / totalServings) = 30 :=
by
  sorry

end pieces_per_serving_l1684_168461


namespace ratio_of_areas_l1684_168480

theorem ratio_of_areas (h a b R : ℝ) (h_triangle : a^2 + b^2 = h^2) (h_circumradius : R = h / 2) :
  (π * R^2) / (1/2 * a * b) = π * h / (4 * R) :=
by sorry

end ratio_of_areas_l1684_168480


namespace distance_to_destination_l1684_168475

theorem distance_to_destination :
  ∀ (D : ℝ) (T : ℝ),
    (15:ℝ) = T →
    (30:ℝ) = T / 2 →
    T - (T / 2) = 3 →
    D = 15 * T → D = 90 :=
by
  intros D T Theon_speed Yara_speed time_difference distance_calc
  sorry

end distance_to_destination_l1684_168475


namespace common_difference_is_4_l1684_168415

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {a_4 a_5 S_6 : ℝ}
variable {d : ℝ}

-- Definitions of conditions given in the problem
def a4_cond : a_4 = a_n 4 := sorry
def a5_cond : a_5 = a_n 5 := sorry
def sum_six : S_6 = (6/2) * (2 * a_n 1 + 5 * d) := sorry
def term_sum : a_4 + a_5 = 24 := sorry

-- Proof statement
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l1684_168415


namespace hilt_miles_traveled_l1684_168463

theorem hilt_miles_traveled (initial_miles lunch_additional_miles : Real) (h_initial : initial_miles = 212.3) (h_lunch : lunch_additional_miles = 372.0) :
  initial_miles + lunch_additional_miles = 584.3 :=
by
  sorry

end hilt_miles_traveled_l1684_168463


namespace inequality_solution_l1684_168420

theorem inequality_solution (x : ℝ) : (3 * x^2 - 4 * x - 4 < 0) ↔ (-2/3 < x ∧ x < 2) :=
sorry

end inequality_solution_l1684_168420


namespace min_sum_of_factors_l1684_168426

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l1684_168426


namespace no_positive_integer_solutions_l1684_168425

def f (x : ℕ) : ℕ := x*x + x

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), a > 0 → b > 0 → 4 * (f a) ≠ (f b) :=
by
  intro a b a_pos b_pos
  sorry

end no_positive_integer_solutions_l1684_168425


namespace Ramu_spent_on_repairs_l1684_168423

theorem Ramu_spent_on_repairs (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : selling_price = 61900) 
  (h3 : profit_percent = 12.545454545454545) 
  (h4 : selling_price = purchase_price + R + (profit_percent / 100) * (purchase_price + R)) : 
  R = 13000 :=
by
  sorry

end Ramu_spent_on_repairs_l1684_168423


namespace samuel_initial_speed_l1684_168441

/-
Samuel is driving to San Francisco’s Comic-Con in his car and he needs to travel 600 miles to the hotel where he made a reservation. 
He drives at a certain speed for 3 hours straight, then he speeds up to 80 miles/hour for 4 hours. 
Now, he is 130 miles away from the hotel. What was his initial speed?
-/

theorem samuel_initial_speed : 
  ∃ v : ℝ, (3 * v + 320 = 470) ↔ (v = 50) :=
by
  use 50
  /- detailed proof goes here -/
  sorry

end samuel_initial_speed_l1684_168441


namespace sufficient_not_necessary_l1684_168442

theorem sufficient_not_necessary (x : ℝ) : abs x < 2 → (x^2 - x - 6 < 0) ∧ (¬(x^2 - x - 6 < 0) → abs x ≥ 2) :=
by
  sorry

end sufficient_not_necessary_l1684_168442


namespace tax_paid_at_fifth_checkpoint_l1684_168439

variable {x : ℚ}

theorem tax_paid_at_fifth_checkpoint (x : ℚ) (h : (x / 2) + (x / 2 * 1 / 3) + (x / 3 * 1 / 4) + (x / 4 * 1 / 5) + (x / 5 * 1 / 6) = 1) :
  (x / 5 * 1 / 6) = 1 / 25 :=
sorry

end tax_paid_at_fifth_checkpoint_l1684_168439


namespace min_value_fraction_l1684_168440

theorem min_value_fraction (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + 2 * y = 3) : 
  (∃ t, t = (1 / (x - y) + 9 / (x + 5 * y)) ∧ t = 8 / 3) :=
by 
  sorry

end min_value_fraction_l1684_168440


namespace option_C_correct_l1684_168413

theorem option_C_correct (x : ℝ) : x^3 * x^2 = x^5 := sorry

end option_C_correct_l1684_168413


namespace range_of_m_l1684_168465

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → (7 / 4) ≤ (x^2 - 3 * x + 4) ∧ (x^2 - 3 * x + 4) ≤ 4) ↔ (3 / 2 ≤ m ∧ m ≤ 3) := 
sorry

end range_of_m_l1684_168465


namespace add_base_3_l1684_168498

def base3_addition : Prop :=
  2 + (1 * 3^2 + 2 * 3^1 + 0 * 3^0) + 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) + 
  (1 * 3^3 + 2 * 3^1 + 0 * 3^0) = 
  (1 * 3^3) + (1 * 3^2) + (0 * 3^1) + (2 * 3^0)

theorem add_base_3 : base3_addition :=
by 
  -- We will skip the proof as per instructions
  sorry

end add_base_3_l1684_168498


namespace intersection_nonempty_condition_l1684_168471

theorem intersection_nonempty_condition (m n : ℝ) :
  (∃ x : ℝ, (m - 1 < x ∧ x < m + 1) ∧ (3 - n < x ∧ x < 4 - n)) ↔ (2 < m + n ∧ m + n < 5) := 
by
  sorry

end intersection_nonempty_condition_l1684_168471


namespace calculation_correct_l1684_168458

theorem calculation_correct : (18 / (3 + 9 - 6)) * 4 = 12 :=
by
  sorry

end calculation_correct_l1684_168458


namespace combine_square_roots_l1684_168407

def can_be_combined (x y: ℝ) : Prop :=
  ∃ k: ℝ, y = k * x

theorem combine_square_roots :
  let sqrt12 := 2 * Real.sqrt 3
  let sqrt1_3 := Real.sqrt 1 / Real.sqrt 3
  let sqrt18 := 3 * Real.sqrt 2
  let sqrt27 := 6 * Real.sqrt 3
  can_be_combined (Real.sqrt 3) sqrt12 ∧
  can_be_combined (Real.sqrt 3) sqrt1_3 ∧
  ¬ can_be_combined (Real.sqrt 3) sqrt18 ∧
  can_be_combined (Real.sqrt 3) sqrt27 :=
by
  sorry

end combine_square_roots_l1684_168407


namespace lattice_point_count_l1684_168497

theorem lattice_point_count :
  (∃ (S : Finset (ℤ × ℤ)), S.card = 16 ∧ ∀ (p : ℤ × ℤ), p ∈ S → (|p.1| - 1) ^ 2 + (|p.2| - 1) ^ 2 < 2) :=
sorry

end lattice_point_count_l1684_168497


namespace iggy_pace_l1684_168489

theorem iggy_pace 
  (monday_miles : ℕ) (tuesday_miles : ℕ) (wednesday_miles : ℕ)
  (thursday_miles : ℕ) (friday_miles : ℕ) (total_hours : ℕ) 
  (h1 : monday_miles = 3) (h2 : tuesday_miles = 4) 
  (h3 : wednesday_miles = 6) (h4 : thursday_miles = 8) 
  (h5 : friday_miles = 3) (h6 : total_hours = 4) :
  (total_hours * 60) / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) = 10 :=
sorry

end iggy_pace_l1684_168489


namespace hyperbola_standard_equation_correct_l1684_168477

-- Define the initial values given in conditions
def a : ℝ := 12
def b : ℝ := 5
def c : ℝ := 4

-- Define the hyperbola equation form based on conditions and focal properties
noncomputable def hyperbola_standard_equation : Prop :=
  let a2 := (8 / 5)
  let b2 := (72 / 5)
  (∀ x y : ℝ, y^2 / a2 - x^2 / b2 = 1)

-- State the final problem as a theorem
theorem hyperbola_standard_equation_correct :
  ∀ x y : ℝ, y^2 / (8 / 5) - x^2 / (72 / 5) = 1 :=
by
  sorry

end hyperbola_standard_equation_correct_l1684_168477


namespace negation_of_cube_of_every_odd_number_is_odd_l1684_168434

theorem negation_of_cube_of_every_odd_number_is_odd:
  ¬ (∀ n : ℤ, (n % 2 = 1 → (n^3 % 2 = 1))) ↔ ∃ n : ℤ, (n % 2 = 1 ∧ ¬ (n^3 % 2 = 1)) := 
by
  sorry

end negation_of_cube_of_every_odd_number_is_odd_l1684_168434


namespace janet_total_pills_l1684_168488

-- Define number of days per week
def days_per_week : ℕ := 7

-- Define pills per day for each week
def pills_first_2_weeks :=
  let multivitamins := 2 * days_per_week * 2
  let calcium := 3 * days_per_week * 2
  let magnesium := 5 * days_per_week * 2
  multivitamins + calcium + magnesium

def pills_third_week :=
  let multivitamins := 2 * days_per_week
  let calcium := 1 * days_per_week
  let magnesium := 0 * days_per_week
  multivitamins + calcium + magnesium

def pills_fourth_week :=
  let multivitamins := 3 * days_per_week
  let calcium := 2 * days_per_week
  let magnesium := 3 * days_per_week
  multivitamins + calcium + magnesium

def total_pills := pills_first_2_weeks + pills_third_week + pills_fourth_week

theorem janet_total_pills : total_pills = 245 := by
  -- Lean will generate a proof goal here with the left-hand side of the equation
  -- equal to an evaluated term, and we say that this equals 245 based on the problem's solution.
  sorry

end janet_total_pills_l1684_168488


namespace cost_prices_max_profit_l1684_168448

theorem cost_prices (a b : ℝ) (x : ℝ) (y : ℝ)
    (h1 : a - b = 500)
    (h2 : 40000 / a = 30000 / b)
    (h3 : 0 ≤ x ∧ x ≤ 20)
    (h4 : 2000 * x + 1500 * (20 - x) ≤ 36000) :
    a = 2000 ∧ b = 1500 := sorry

theorem max_profit (x : ℝ) (y : ℝ)
    (h1 : 0 ≤ x ∧ x ≤ 12) :
    y = 200 * x + 6000 ∧ y ≤ 8400 := sorry

end cost_prices_max_profit_l1684_168448


namespace find_m_values_l1684_168483

theorem find_m_values (α : Real) (m : Real) (h1 : α ∈ Set.Ioo π (3 * π / 2)) 
  (h2 : Real.sin α = (3 * m - 2) / (m + 3)) 
  (h3 : Real.cos α = (m - 5) / (m + 3)) : m = (10 / 9) ∨ m = 2 := by 
  sorry

end find_m_values_l1684_168483


namespace greatest_common_length_cords_l1684_168473

theorem greatest_common_length_cords (l1 l2 l3 l4 : ℝ) (h1 : l1 = Real.sqrt 20) (h2 : l2 = Real.pi) (h3 : l3 = Real.exp 1) (h4 : l4 = Real.sqrt 98) : 
  ∃ d : ℝ, d = 1 ∧ (∀ k1 k2 k3 k4 : ℝ, k1 * d = l1 → k2 * d = l2 → k3 * d = l3 → k4 * d = l4 → ∀i : ℝ, i = d) :=
by
  sorry

end greatest_common_length_cords_l1684_168473


namespace prove_inequalities_l1684_168455

variable {a b c R r_a r_b r_c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_circumradius (a b c R : ℝ) : Prop :=
  ∃ S : ℝ, S = a * b * c / (4 * R)

def has_exradii (a b c r_a r_b r_c : ℝ) : Prop :=
  ∃ S : ℝ, 
    r_a = 2 * S / (b + c - a) ∧
    r_b = 2 * S / (a + c - b) ∧
    r_c = 2 * S / (a + b - c)

theorem prove_inequalities
  (h_triangle : is_triangle a b c)
  (h_circumradius : has_circumradius a b c R)
  (h_exradii : has_exradii a b c r_a r_b r_c)
  (h_two_R_le_r_a : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := 
sorry

end prove_inequalities_l1684_168455


namespace trebled_resultant_is_correct_l1684_168464

-- Definitions based on the conditions provided in step a)
def initial_number : ℕ := 5
def doubled_result : ℕ := initial_number * 2
def added_15_result : ℕ := doubled_result + 15
def trebled_resultant : ℕ := added_15_result * 3

-- We need to prove that the trebled resultant is equal to 75
theorem trebled_resultant_is_correct : trebled_resultant = 75 :=
by
  sorry

end trebled_resultant_is_correct_l1684_168464


namespace perimeter_of_square_C_l1684_168486

theorem perimeter_of_square_C (s_A s_B s_C : ℝ)
  (h1 : 4 * s_A = 16)
  (h2 : 4 * s_B = 32)
  (h3 : s_C = s_B - s_A) :
  4 * s_C = 16 :=
by
  sorry

end perimeter_of_square_C_l1684_168486


namespace reduced_admission_price_is_less_l1684_168452

-- Defining the conditions
def regular_admission_cost : ℕ := 8
def total_people : ℕ := 2 + 3 + 1
def total_cost_before_6pm : ℕ := 30
def cost_per_person_before_6pm : ℕ := total_cost_before_6pm / total_people

-- Stating the theorem
theorem reduced_admission_price_is_less :
  (regular_admission_cost - cost_per_person_before_6pm) = 3 :=
by
  sorry -- Proof to be filled

end reduced_admission_price_is_less_l1684_168452


namespace quadratic_roots_evaluation_l1684_168416

theorem quadratic_roots_evaluation (x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) :
  (1 + x1) + x2 * (1 - x1) = 4 :=
by
  sorry

end quadratic_roots_evaluation_l1684_168416


namespace intersection_point_l1684_168429

def line_eq (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (y + 1) / 0 ∧ (y + 1) / 0 = (z - 1) / -1

def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 2 * y - 4 * z - 8 = 0

theorem intersection_point : 
  ∃ (x y z : ℝ), line_eq x y z ∧ plane_eq x y z ∧ x = -6 ∧ y = -1 ∧ z = 8 :=
by 
  sorry

end intersection_point_l1684_168429


namespace find_XY_in_triangle_l1684_168459

-- Definitions
def Triangle := Type
def angle_measures (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def side_lengths (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def is_30_60_90_triangle (T : Triangle) : Prop := (angle_measures T = (30, 60, 90))

-- Given conditions and statement we want to prove
def triangle_XYZ : Triangle := sorry
def XY : ℕ := 6

-- Proof statement
theorem find_XY_in_triangle :
  is_30_60_90_triangle triangle_XYZ ∧ (side_lengths triangle_XYZ).1 = XY →
  XY = 6 :=
by
  intro h
  sorry

end find_XY_in_triangle_l1684_168459


namespace f_sub_f_inv_eq_2022_l1684_168400

def f (n : ℕ) : ℕ := 2 * n
def f_inv (n : ℕ) : ℕ := n

theorem f_sub_f_inv_eq_2022 : f 2022 - f_inv 2022 = 2022 := by
  -- Proof goes here
  sorry

end f_sub_f_inv_eq_2022_l1684_168400


namespace red_sequence_57_eq_103_l1684_168485

-- Definitions based on conditions described in the problem
def red_sequence : Nat → Nat
| 0 => 1  -- First number is 1
| 1 => 2  -- Next even number
| 2 => 4  -- Next even number
-- Continue defining based on patterns from problem
| (n+3) => -- Each element recursively following the pattern
 sorry  -- Detailed pattern definition is skipped

-- Main theorem: the 57th number in the red subsequence is 103
theorem red_sequence_57_eq_103 : red_sequence 56 = 103 :=
 sorry

end red_sequence_57_eq_103_l1684_168485


namespace log_expression_evaluation_l1684_168472

open Real

theorem log_expression_evaluation : log 5 * log 20 + (log 2) ^ 2 = 1 := 
sorry

end log_expression_evaluation_l1684_168472


namespace mayoral_election_votes_l1684_168430

theorem mayoral_election_votes (Y Z : ℕ) 
  (h1 : 22500 = Y + Y / 2) 
  (h2 : 15000 = Z - Z / 5 * 2)
  : Z = 25000 := 
  sorry

end mayoral_election_votes_l1684_168430


namespace fraction_of_speedsters_l1684_168446

/-- Let S denote the total number of Speedsters and T denote the total inventory. 
    Given the following conditions:
    1. 54 Speedster convertibles constitute 3/5 of all Speedsters (S).
    2. There are 30 vehicles that are not Speedsters.

    Prove that the fraction of the current inventory that is Speedsters is 3/4.
-/
theorem fraction_of_speedsters (S T : ℕ)
  (h1 : 3 / 5 * S = 54)
  (h2 : T = S + 30) :
  (S : ℚ) / T = 3 / 4 :=
by
  sorry

end fraction_of_speedsters_l1684_168446


namespace system1_solution_system2_solution_system3_solution_l1684_168494

theorem system1_solution (x y : ℝ) : 
  (x = 3/2) → (y = 1/2) → (x + 3 * y = 3) ∧ (x - y = 1) :=
by intros; sorry

theorem system2_solution (x y : ℝ) : 
  (x = 0) → (y = 2/5) → ((x + 3 * y) / 2 = 3 / 5) ∧ (5 * (x - 2 * y) = -4) :=
by intros; sorry

theorem system3_solution (x y z : ℝ) : 
  (x = 1) → (y = 2) → (z = 3) → 
  (3 * x + 4 * y + z = 14) ∧ (x + 5 * y + 2 * z = 17) ∧ (2 * x + 2 * y - z = 3) :=
by intros; sorry

end system1_solution_system2_solution_system3_solution_l1684_168494


namespace prime_p_p_plus_15_l1684_168496

theorem prime_p_p_plus_15 (p : ℕ) (hp : Nat.Prime p) (hp15 : Nat.Prime (p + 15)) : p = 2 :=
sorry

end prime_p_p_plus_15_l1684_168496


namespace trajectory_equation_l1684_168495

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

theorem trajectory_equation (P : ℝ × ℝ) (h : |distance P (1, 0) - P.1| = 1) :
  (P.1 ≥ 0 → P.2 ^ 2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
by
  sorry

end trajectory_equation_l1684_168495


namespace jacob_younger_than_michael_l1684_168474

-- Definitions based on the conditions.
def jacob_current_age : ℕ := 9
def michael_current_age : ℕ := 2 * (jacob_current_age + 3) - 3

-- Theorem to prove that Jacob is 12 years younger than Michael.
theorem jacob_younger_than_michael : michael_current_age - jacob_current_age = 12 :=
by
  -- Placeholder for proof
  sorry

end jacob_younger_than_michael_l1684_168474


namespace find_d_minus_r_l1684_168468

theorem find_d_minus_r :
  ∃ (d r : ℕ), d > 1 ∧ 1083 % d = r ∧ 1455 % d = r ∧ 2345 % d = r ∧ d - r = 1 := by
  sorry

end find_d_minus_r_l1684_168468


namespace apples_in_bowl_l1684_168436

variable {A : ℕ}

theorem apples_in_bowl
  (initial_oranges : ℕ)
  (removed_oranges : ℕ)
  (final_oranges : ℕ)
  (total_fruit : ℕ)
  (fraction_apples : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  final_oranges = initial_oranges - removed_oranges →
  fraction_apples = (70 : ℚ) / (100 : ℚ) →
  final_oranges = total_fruit * (30 : ℚ) / (100 : ℚ) →
  A = total_fruit * fraction_apples →
  A = 14 :=
by
  sorry

end apples_in_bowl_l1684_168436


namespace total_cost_of_items_l1684_168460

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end total_cost_of_items_l1684_168460


namespace no_valid_m_n_l1684_168427

theorem no_valid_m_n (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ¬ (m * n ∣ 3^m + 1 ∧ m * n ∣ 3^n + 1) :=
by
  sorry

end no_valid_m_n_l1684_168427


namespace area_of_region_l1684_168453

-- Definitions drawn from conditions
def circle_radius := 36
def num_small_circles := 8

-- Main statement to be proven
theorem area_of_region :
  ∃ K : ℝ, 
    K = π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ∧
    ⌊ K ⌋ = ⌊ π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ⌋ :=
  sorry

end area_of_region_l1684_168453


namespace intersecting_chords_l1684_168432

noncomputable def length_of_other_chord (x : ℝ) : ℝ :=
  3 * x + 8 * x

theorem intersecting_chords
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18) (r1 r2 : ℝ) (h3 : r1/r2 = 3/8) :
  length_of_other_chord 3 = 33 := by
  sorry

end intersecting_chords_l1684_168432


namespace perfect_square_difference_l1684_168466

theorem perfect_square_difference (m n : ℕ) (h : 2001 * m^2 + m = 2002 * n^2 + n) : ∃ k : ℕ, k^2 = m - n :=
sorry

end perfect_square_difference_l1684_168466


namespace sam_pennies_l1684_168499

def pennies_from_washing_clothes (total_money_cents : ℤ) (quarters : ℤ) : ℤ :=
  total_money_cents - (quarters * 25)

theorem sam_pennies :
  pennies_from_washing_clothes 184 7 = 9 :=
by
  sorry

end sam_pennies_l1684_168499


namespace mr_green_yield_l1684_168479

noncomputable def steps_to_feet (steps : ℕ) : ℝ :=
  steps * 2.5

noncomputable def total_yield (steps_x : ℕ) (steps_y : ℕ) (yield_potato_per_sqft : ℝ) (yield_carrot_per_sqft : ℝ) : ℝ :=
  let width := steps_to_feet steps_x
  let height := steps_to_feet steps_y
  let area := width * height
  (area * yield_potato_per_sqft) + (area * yield_carrot_per_sqft)

theorem mr_green_yield :
  total_yield 20 25 0.5 0.25 = 2343.75 :=
by
  sorry

end mr_green_yield_l1684_168479


namespace legacy_earnings_per_hour_l1684_168435

-- Define the conditions
def totalFloors : ℕ := 4
def roomsPerFloor : ℕ := 10
def hoursPerRoom : ℕ := 6
def totalEarnings : ℝ := 3600

-- The statement to prove
theorem legacy_earnings_per_hour :
  (totalFloors * roomsPerFloor * hoursPerRoom) = 240 → 
  (totalEarnings / (totalFloors * roomsPerFloor * hoursPerRoom)) = 15 := by
  intros h
  sorry

end legacy_earnings_per_hour_l1684_168435


namespace ab_is_zero_l1684_168421

-- Define that a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b - 2

-- The main theorem to prove
theorem ab_is_zero (a b : ℝ) (h_odd : is_odd (f a b)) : a * b = 0 := 
sorry

end ab_is_zero_l1684_168421


namespace sin_alpha_in_second_quadrant_l1684_168469

theorem sin_alpha_in_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -1 / 2)  -- tan α = -1/2
  : Real.sin α = Real.sqrt 5 / 5 :=
sorry

end sin_alpha_in_second_quadrant_l1684_168469


namespace max_chord_length_line_eq_orthogonal_vectors_line_eq_l1684_168428

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 2 ∧ y = 1
def line_eq (slope intercept x y : ℝ) : Prop := y = slope * x + intercept

-- Problem 1: Prove the equation of line l that maximizes the length of chord AB
theorem max_chord_length_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq 1 (-1) x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq 1 (-1) x y) :=
by sorry

-- Problem 2: Prove the equation of line l given orthogonality condition of vectors
theorem orthogonal_vectors_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq (-1) 3 x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq (-1) 3 x y) :=
by sorry

end max_chord_length_line_eq_orthogonal_vectors_line_eq_l1684_168428


namespace area_of_side_face_l1684_168484

theorem area_of_side_face (L W H : ℝ) 
  (h1 : W * H = (1/2) * (L * W))
  (h2 : L * W = 1.5 * (H * L))
  (h3 : L * W * H = 648) :
  H * L = 72 := 
by
  sorry

end area_of_side_face_l1684_168484


namespace problem_solution_l1684_168404

noncomputable def problem (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α^2 + p * α - 1 = 0) ∧
  (β^2 + p * β - 1 = 0) ∧
  (γ^2 + q * γ + 1 = 0) ∧
  (δ^2 + q * δ + 1 = 0) →
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2

theorem problem_solution (p q α β γ δ : ℝ) : 
  problem p q α β γ δ := 
by sorry

end problem_solution_l1684_168404


namespace arithmetic_mean_squares_l1684_168403

theorem arithmetic_mean_squares (n : ℕ) (h : 0 < n) :
  let S_n2 := (n * (n + 1) * (2 * n + 1)) / 6 
  let A_n2 := S_n2 / n
  A_n2 = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end arithmetic_mean_squares_l1684_168403


namespace power_mod_eight_l1684_168467

theorem power_mod_eight (n : ℕ) : (3^101 + 5) % 8 = 0 :=
by
  sorry

end power_mod_eight_l1684_168467


namespace total_employees_l1684_168444

def part_time_employees : ℕ := 2047
def full_time_employees : ℕ := 63109
def contractors : ℕ := 1500
def interns : ℕ := 333
def consultants : ℕ := 918

theorem total_employees : 
  part_time_employees + full_time_employees + contractors + interns + consultants = 66907 := 
by
  -- proof goes here
  sorry

end total_employees_l1684_168444


namespace smallest_N_constant_l1684_168454

-- Define the property to be proven
theorem smallest_N_constant (a b c : ℝ) 
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) (h₄ : k = 0):
  (a^2 + b^2 + k) / c^2 > 1 / 2 :=
by
  sorry

end smallest_N_constant_l1684_168454


namespace total_cost_4kg_mangos_3kg_rice_5kg_flour_l1684_168411

def cost_per_kg_mangos (M : ℝ) (R : ℝ) := (10 * M = 24 * R)
def cost_per_kg_flour_equals_rice (F : ℝ) (R : ℝ) := (6 * F = 2 * R)
def cost_of_flour (F : ℝ) := (F = 24)

theorem total_cost_4kg_mangos_3kg_rice_5kg_flour 
  (M R F : ℝ) 
  (h1 : cost_per_kg_mangos M R) 
  (h2 : cost_per_kg_flour_equals_rice F R) 
  (h3 : cost_of_flour F) : 
  4 * M + 3 * R + 5 * F = 1027.2 :=
by {
  sorry
}

end total_cost_4kg_mangos_3kg_rice_5kg_flour_l1684_168411


namespace determine_phi_l1684_168470

theorem determine_phi
  (A ω : ℝ) (φ : ℝ) (x : ℝ)
  (hA : 0 < A)
  (hω : 0 < ω)
  (hφ : abs φ < Real.pi / 2)
  (h_symm : ∃ f : ℝ → ℝ, f (-Real.pi / 4) = A ∨ f (-Real.pi / 4) = -A)
  (h_zero : ∃ x₀ : ℝ, A * Real.sin (ω * x₀ + φ) = 0 ∧ abs (x₀ + Real.pi / 4) = Real.pi / 2) :
  φ = -Real.pi / 4 :=
sorry

end determine_phi_l1684_168470


namespace converse_proposition_converse_proposition_true_l1684_168492

theorem converse_proposition (x : ℝ) (h : x > 0) : x^2 - 1 > 0 :=
by sorry

theorem converse_proposition_true (x : ℝ) (h : x^2 - 1 > 0) : x > 0 :=
by sorry

end converse_proposition_converse_proposition_true_l1684_168492


namespace polygon_sides_l1684_168417

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end polygon_sides_l1684_168417


namespace average_income_proof_l1684_168431

theorem average_income_proof:
  ∀ (A B C : ℝ),
    (A + B) / 2 = 5050 →
    (B + C) / 2 = 6250 →
    A = 4000 →
    (A + C) / 2 = 5200 := by
  sorry

end average_income_proof_l1684_168431


namespace ball_of_yarn_costs_6_l1684_168402

-- Define the conditions as variables and hypotheses
variable (num_sweaters : ℕ := 28)
variable (balls_per_sweater : ℕ := 4)
variable (price_per_sweater : ℕ := 35)
variable (gain_from_sales : ℕ := 308)

-- Define derived values
def total_revenue : ℕ := num_sweaters * price_per_sweater
def total_cost_of_yarn : ℕ := total_revenue - gain_from_sales
def total_balls_of_yarn : ℕ := num_sweaters * balls_per_sweater
def cost_per_ball_of_yarn : ℕ := total_cost_of_yarn / total_balls_of_yarn

-- The theorem to be proven
theorem ball_of_yarn_costs_6 :
  cost_per_ball_of_yarn = 6 :=
by sorry

end ball_of_yarn_costs_6_l1684_168402


namespace trapezoid_area_l1684_168410

theorem trapezoid_area (a b d1 d2 : ℝ) (ha : 0 < a) (hb : 0 < b) (hd1 : 0 < d1) (hd2 : 0 < d2)
  (hbase : a = 11) (hbase2 : b = 4) (hdiagonal1 : d1 = 9) (hdiagonal2 : d2 = 12) :
  (∃ area : ℝ, area = 54) :=
by
  sorry

end trapezoid_area_l1684_168410


namespace quadratic_value_l1684_168419

theorem quadratic_value (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : 4 * a + 2 * b + c = 3) :
  a + 2 * b + 3 * c = 7 :=
by
  sorry

end quadratic_value_l1684_168419


namespace min_value_pt_qu_rv_sw_l1684_168451

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (h1 : p * q * r * s = 8) (h2 : t * u * v * w = 27) :
  (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 ≥ 96 :=
by
  sorry

end min_value_pt_qu_rv_sw_l1684_168451


namespace B_work_time_alone_l1684_168418

theorem B_work_time_alone
  (A_rate : ℝ := 1 / 8)
  (together_rate : ℝ := 3 / 16) :
  ∃ (B_days : ℝ), B_days = 16 :=
by
  sorry

end B_work_time_alone_l1684_168418
