import Mathlib

namespace project_completion_time_l274_27463

theorem project_completion_time
  (A_time B_time : ℕ) 
  (hA : A_time = 20)
  (hB : B_time = 20)
  (A_quit_days : ℕ) 
  (hA_quit : A_quit_days = 10) :
  ∃ x : ℕ, (x - A_quit_days) * (1 / A_time : ℚ) + (x * (1 / B_time : ℚ)) = 1 ∧ x = 15 := by
  sorry

end project_completion_time_l274_27463


namespace installation_cost_l274_27449

-- Definitions
variables (LP : ℝ) (P : ℝ := 16500) (D : ℝ := 0.2) (T : ℝ := 125) (SP : ℝ := 23100) (I : ℝ)

-- Conditions
def purchase_price := P = (1 - D) * LP
def selling_price := SP = 1.1 * LP
def total_cost := P + T + I = SP

-- Proof Statement
theorem installation_cost : I = 6350 :=
  by
    -- sorry is used to skip the proof
    sorry

end installation_cost_l274_27449


namespace fraction_evaluation_l274_27443

theorem fraction_evaluation :
  (1/5 - 1/7) / (3/8 + 2/9) = 144/1505 := 
  by 
    sorry

end fraction_evaluation_l274_27443


namespace incorrect_inequality_l274_27410

theorem incorrect_inequality (x y : ℝ) (h : x > y) : ¬ (-3 * x > -3 * y) :=
by
  sorry

end incorrect_inequality_l274_27410


namespace nandan_gain_l274_27448

theorem nandan_gain (x t : ℝ) (nandan_gain krishan_gain total_gain : ℝ)
  (h1 : krishan_gain = 12 * x * t)
  (h2 : nandan_gain = x * t)
  (h3 : total_gain = nandan_gain + krishan_gain)
  (h4 : total_gain = 78000) :
  nandan_gain = 6000 :=
by
  -- Proof goes here
  sorry

end nandan_gain_l274_27448


namespace problem1_part1_problem1_part2_l274_27495

theorem problem1_part1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) := 
sorry

theorem problem1_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 := 
sorry

end problem1_part1_problem1_part2_l274_27495


namespace parametric_inclination_l274_27425

noncomputable def angle_of_inclination (x y : ℝ) : ℝ := 50

theorem parametric_inclination (t : ℝ) (x y : ℝ) :
  x = t * Real.sin 40 → y = -1 + t * Real.cos 40 → angle_of_inclination x y = 50 :=
by
  intros hx hy
  -- This is where the proof would go, but we skip it.
  sorry

end parametric_inclination_l274_27425


namespace students_in_section_A_l274_27431

theorem students_in_section_A (x : ℕ) (h1 : (40 : ℝ) * x + 44 * 35 = 37.25 * (x + 44)) : x = 36 :=
by
  sorry

end students_in_section_A_l274_27431


namespace winning_candidate_percentage_is_57_l274_27414

def candidate_votes : List ℕ := [1136, 7636, 11628]

def total_votes : ℕ := candidate_votes.sum

def winning_votes : ℕ := candidate_votes.maximum?.getD 0

def winning_percentage (votes : ℕ) (total : ℕ) : ℚ :=
  (votes * 100) / total

theorem winning_candidate_percentage_is_57 :
  winning_percentage winning_votes total_votes = 57 := by
  sorry

end winning_candidate_percentage_is_57_l274_27414


namespace intersection_eq_interval_l274_27496

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {x | x < 5}

theorem intersection_eq_interval : M ∩ N = {x | 1 < x ∧ x < 5} :=
sorry

end intersection_eq_interval_l274_27496


namespace fraction_of_tomatoes_eaten_l274_27457

theorem fraction_of_tomatoes_eaten (original : ℕ) (remaining : ℕ) (birds_ate : ℕ) (h1 : original = 21) (h2 : remaining = 14) (h3 : birds_ate = original - remaining) :
  (birds_ate : ℚ) / original = 1 / 3 :=
by
  sorry

end fraction_of_tomatoes_eaten_l274_27457


namespace solve_equation_l274_27432

theorem solve_equation (x : ℝ) : x * (2 * x - 1) = 4 * x - 2 ↔ x = 2 ∨ x = 1 / 2 := 
by {
  sorry -- placeholder for the proof
}

end solve_equation_l274_27432


namespace chocolate_cost_l274_27451

def cost_of_chocolates (candies_per_box : ℕ) (cost_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies / candies_per_box) * cost_per_box

theorem chocolate_cost : cost_of_chocolates 30 8 450 = 120 :=
by
  -- The proof is not needed per the instructions
  sorry

end chocolate_cost_l274_27451


namespace sum_of_asymptotes_l274_27468

theorem sum_of_asymptotes :
  let c := -3/2
  let d := -1
  c + d = -5/2 :=
by
  -- Definitions corresponding to the problem conditions
  let c := -3/2
  let d := -1
  -- Statement of the theorem
  show c + d = -5/2
  sorry

end sum_of_asymptotes_l274_27468


namespace total_flowers_l274_27497

theorem total_flowers (R T L : ℕ) 
  (hR : R = 58)
  (hT : R = T + 15)
  (hL : R = L - 25) :
  R + T + L = 184 :=
by 
  sorry

end total_flowers_l274_27497


namespace age_of_B_l274_27445

-- Define the ages based on the conditions
def A (x : ℕ) : ℕ := 2 * x + 2
def B (x : ℕ) : ℕ := 2 * x
def C (x : ℕ) : ℕ := x

-- The main statement to be proved
theorem age_of_B (x : ℕ) (h : A x + B x + C x = 72) : B 14 = 28 :=
by
  -- we need the proof here but we will put sorry for now
  sorry

end age_of_B_l274_27445


namespace reflection_coordinates_l274_27405

-- Define the original coordinates of point M
def original_point : (ℝ × ℝ) := (3, -4)

-- Define the function to reflect a point across the x-axis
def reflect_across_x_axis (p: ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1, -p.2)

-- State the theorem to prove the coordinates after reflection
theorem reflection_coordinates :
  reflect_across_x_axis original_point = (3, 4) :=
by
  sorry

end reflection_coordinates_l274_27405


namespace problem1_problem2_l274_27430

variable (x y : ℝ)

theorem problem1 :
  x^4 * x^3 * x - (x^4)^2 + (-2 * x)^3 * x^5 = -8 * x^8 :=
by sorry

theorem problem2 :
  (x - y)^4 * (y - x)^3 / (y - x)^2 = (x - y)^5 :=
by sorry

end problem1_problem2_l274_27430


namespace necessary_but_not_sufficient_l274_27423

-- Define the sets A, B, and C
def A : Set ℝ := { x | x - 1 > 0 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

-- The set A ∪ B in terms of Lean
def A_union_B : Set ℝ := A ∪ B

-- State the necessary and sufficient conditions
theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x ∈ A_union_B → x ∈ C) ∧ ¬ (∀ x : ℝ, x ∈ C → x ∈ A_union_B) :=
sorry

end necessary_but_not_sufficient_l274_27423


namespace john_overall_profit_l274_27400

-- Definitions based on conditions
def cost_grinder : ℕ := 15000
def cost_mobile : ℕ := 8000
def loss_percentage_grinder : ℚ := 4 / 100
def profit_percentage_mobile : ℚ := 15 / 100

-- Calculations based on the conditions
def loss_amount_grinder := cost_grinder * loss_percentage_grinder
def selling_price_grinder := cost_grinder - loss_amount_grinder
def profit_amount_mobile := cost_mobile * profit_percentage_mobile
def selling_price_mobile := cost_mobile + profit_amount_mobile
def total_cost_price := cost_grinder + cost_mobile
def total_selling_price := selling_price_grinder + selling_price_mobile

-- Overall profit calculation
def overall_profit := total_selling_price - total_cost_price

-- Proof statement to prove the overall profit
theorem john_overall_profit : overall_profit = 600 := by
  sorry

end john_overall_profit_l274_27400


namespace find_BD_in_triangle_l274_27426

theorem find_BD_in_triangle (A B C D : Type)
  (distance_AC : Float) (distance_BC : Float)
  (distance_AD : Float) (distance_CD : Float)
  (hAC : distance_AC = 10)
  (hBC : distance_BC = 10)
  (hAD : distance_AD = 12)
  (hCD : distance_CD = 5) :
  ∃ (BD : Float), BD = 6.85435 :=
by 
  sorry

end find_BD_in_triangle_l274_27426


namespace largest_fraction_consecutive_primes_l274_27470

theorem largest_fraction_consecutive_primes (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs : Nat.Prime s)
  (h0 : 0 < p) (h1 : p < q) (h2 : q < r) (h3 : r < s)
  (hconsec : p + 2 = q ∧ q + 2 = r ∧ r + 2 = s) :
  (r + s) / (p + q) > max ((p + q) / (r + s)) (max ((p + s) / (q + r)) (max ((q + r) / (p + s)) ((q + s) / (p + r)))) :=
sorry

end largest_fraction_consecutive_primes_l274_27470


namespace sunzi_system_l274_27483

variable (x y : ℝ)

theorem sunzi_system :
  (y - x = 4.5) ∧ (x - (1/2) * y = 1) :=
by
  sorry

end sunzi_system_l274_27483


namespace clock_angle_7_35_l274_27469

noncomputable def hour_angle (hours : ℤ) (minutes : ℤ) : ℝ :=
  (hours * 30 + (minutes * 30) / 60 : ℝ)

noncomputable def minute_angle (minutes : ℤ) : ℝ :=
  (minutes * 360 / 60 : ℝ)

noncomputable def angle_between (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

theorem clock_angle_7_35 : angle_between (hour_angle 7 35) (minute_angle 35) = 17.5 :=
by
  sorry

end clock_angle_7_35_l274_27469


namespace solution_points_satisfy_equation_l274_27459

theorem solution_points_satisfy_equation (x y : ℝ) :
  x^2 * (y + y^2) = y^3 + x^4 → (y = x ∨ y = -x ∨ y = x^2) := sorry

end solution_points_satisfy_equation_l274_27459


namespace coefficient_of_y_squared_l274_27499

/-- Given the equation ay^2 - 8y + 55 = 59 and y = 2, prove that the coefficient a is 5. -/
theorem coefficient_of_y_squared (a y : ℝ) (h_y : y = 2) (h_eq : a * y^2 - 8 * y + 55 = 59) : a = 5 := by
  sorry

end coefficient_of_y_squared_l274_27499


namespace scientific_notation_l274_27487

def given_number : ℝ := 632000

theorem scientific_notation : given_number = 6.32 * 10^5 :=
by sorry

end scientific_notation_l274_27487


namespace intersection_A_B_l274_27485

def set_A (x : ℝ) : Prop := 2 * x + 1 > 0
def set_B (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_A_B : 
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l274_27485


namespace midpoint_of_points_l274_27479

theorem midpoint_of_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 10) (h3 : x2 = 8) (h4 : y2 = 4) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 7) := 
by
  rw [h1, h2, h3, h4]
  norm_num

end midpoint_of_points_l274_27479


namespace precision_of_21_658_billion_is_hundred_million_l274_27435

theorem precision_of_21_658_billion_is_hundred_million :
  (21.658 : ℝ) * 10^9 % (10^8) = 0 :=
by
  sorry

end precision_of_21_658_billion_is_hundred_million_l274_27435


namespace brokerage_percentage_l274_27447

theorem brokerage_percentage (cash_realized amount_before : ℝ) (h1 : cash_realized = 105.25) (h2 : amount_before = 105) :
  |((amount_before - cash_realized) / amount_before) * 100| = 0.2381 := by
sorry

end brokerage_percentage_l274_27447


namespace find_natural_number_n_l274_27424

theorem find_natural_number_n (n x y : ℕ) (h1 : n + 195 = x^3) (h2 : n - 274 = y^3) : 
  n = 2002 :=
by
  sorry

end find_natural_number_n_l274_27424


namespace find_a_l274_27472

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : (min (log_a a 2) (log_a a 4)) * (max (log_a a 2) (log_a a 4)) = 2) : 
  a = (1 / 2) ∨ a = 2 :=
sorry

end find_a_l274_27472


namespace total_area_l274_27401

variable (A : ℝ)

-- Defining the conditions
def first_carpet : Prop := 0.55 * A = 36
def second_carpet : Prop := 0.25 * A = A * 0.25
def third_carpet : Prop := 0.15 * A = 18 + 6
def remaining_floor : Prop := 0.05 * A + 0.55 * A + 0.25 * A + 0.15 * A = A

-- Main theorem to prove the total area
theorem total_area : first_carpet A → second_carpet A → third_carpet A → remaining_floor A → A = 65.45 :=
by
  sorry

end total_area_l274_27401


namespace compute_result_l274_27438

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l274_27438


namespace find_exponent_l274_27437

theorem find_exponent (y : ℕ) (b : ℕ) (h_b : b = 2)
  (h : 1 / 8 * 2 ^ 40 = b ^ y) : y = 37 :=
by
  sorry

end find_exponent_l274_27437


namespace hexagonal_prism_surface_area_l274_27482

theorem hexagonal_prism_surface_area (h : ℝ) (a : ℝ) (H_h : h = 6) (H_a : a = 4) : 
  let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  let lateral_area := 6 * a * h
  let total_area := lateral_area + base_area
  total_area = 48 * (3 + Real.sqrt 3) :=
by
  -- let base_area := 2 * (3 * a^2 * (Real.sqrt 3) / 2)
  -- let lateral_area := 6 * a * h
  -- let total_area := lateral_area + base_area
  -- total_area = 48 * (3 + Real.sqrt 3)
  sorry

end hexagonal_prism_surface_area_l274_27482


namespace Jill_has_5_peaches_l274_27406

-- Define the variables and their relationships
variables (S Jl Jk : ℕ)

-- Declare the conditions as assumptions
axiom Steven_has_14_peaches : S = 14
axiom Jake_has_6_fewer_peaches_than_Steven : Jk = S - 6
axiom Jake_has_3_more_peaches_than_Jill : Jk = Jl + 3

-- Define the theorem to prove Jill has 5 peaches
theorem Jill_has_5_peaches (S Jk Jl : ℕ) 
  (h1 : S = 14) 
  (h2 : Jk = S - 6)
  (h3 : Jk = Jl + 3) : 
  Jl = 5 := 
by
  sorry

end Jill_has_5_peaches_l274_27406


namespace simplify_abs_eq_l274_27473

variable {x : ℚ}

theorem simplify_abs_eq (hx : |1 - x| = 1 + |x|) : |x - 1| = 1 - x :=
by
  sorry

end simplify_abs_eq_l274_27473


namespace inequality_holds_l274_27454

theorem inequality_holds (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) (h_mul : a * b * c * d = 1) :
  (1 / a) + (1 / b) + (1 / c) + (1 / d) + (12 / (a + b + c + d)) ≥ 7 :=
by
  sorry

end inequality_holds_l274_27454


namespace union_A_B_inter_A_B_comp_int_B_l274_27491

open Set

variable (x : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x < 4}
def B := {x : ℝ | 3 ≤ x}

theorem union_A_B : A ∪ B = (Ici 2) :=
by
  sorry

theorem inter_A_B : A ∩ B = Ico 3 4 :=
by
  sorry

theorem comp_int_B : (univ \ A) ∩ B = Ici 4 :=
by
  sorry

end union_A_B_inter_A_B_comp_int_B_l274_27491


namespace find_m_l274_27402

-- We define the universal set U, the set A with an unknown m, and the complement of A in U.
def U : Set ℕ := {1, 2, 3}
def A (m : ℕ) : Set ℕ := {1, m}
def complement_U_A (m : ℕ) : Set ℕ := U \ A m

-- The main theorem where we need to prove m = 3 given the conditions.
theorem find_m (m : ℕ) (hU : U = {1, 2, 3})
  (hA : ∀ m, A m = {1, m})
  (h_complement : complement_U_A m = {2}) : m = 3 := sorry

end find_m_l274_27402


namespace reduced_price_per_kg_l274_27412

theorem reduced_price_per_kg (P R : ℝ) (Q : ℝ)
  (h1 : R = 0.80 * P)
  (h2 : Q * P = 1500)
  (h3 : (Q + 10) * R = 1500) : R = 30 :=
by
  sorry

end reduced_price_per_kg_l274_27412


namespace study_days_l274_27453

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end study_days_l274_27453


namespace david_on_sixth_platform_l274_27460

theorem david_on_sixth_platform 
  (h₁ : walter_initial_fall = 4)
  (h₂ : walter_additional_fall = 3 * walter_initial_fall)
  (h₃ : total_fall = walter_initial_fall + walter_additional_fall)
  (h₄ : total_platforms = 8)
  (h₅ : total_height = total_fall)
  (h₆ : platform_height = total_height / total_platforms)
  (h₇ : david_fall_distance = walter_initial_fall)
  : (total_height - david_fall_distance) / platform_height = 6 := 
  by sorry

end david_on_sixth_platform_l274_27460


namespace system_of_equations_solution_l274_27411

variable {x y : ℝ}

theorem system_of_equations_solution
  (h1 : x^2 + x * y * Real.sqrt (x * y) + y^2 = 25)
  (h2 : x^2 - x * y * Real.sqrt (x * y) + y^2 = 9) :
  (x, y) = (1, 4) ∨ (x, y) = (4, 1) ∨ (x, y) = (-1, -4) ∨ (x, y) = (-4, -1) :=
by
  sorry

end system_of_equations_solution_l274_27411


namespace simplify_expr1_simplify_expr2_simplify_expr3_l274_27465

theorem simplify_expr1 : -2.48 + 4.33 + (-7.52) + (-4.33) = -10 := by
  sorry

theorem simplify_expr2 : (7/13) * (-9) + (7/13) * (-18) + (7/13) = -14 := by
  sorry

theorem simplify_expr3 : -((20 + 1/19) * 38) = -762 := by
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l274_27465


namespace sum_of_integers_l274_27489

theorem sum_of_integers (m n p q : ℤ) 
(h1 : m ≠ n) (h2 : m ≠ p) 
(h3 : m ≠ q) (h4 : n ≠ p) 
(h5 : n ≠ q) (h6 : p ≠ q) 
(h7 : (5 - m) * (5 - n) * (5 - p) * (5 - q) = 9) : 
m + n + p + q = 20 :=
by
  sorry

end sum_of_integers_l274_27489


namespace line_y_intercept_l274_27471

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 6) (h4 : y2 = 9) :
  ∃ b : ℝ, b = -9 := 
by
  sorry

end line_y_intercept_l274_27471


namespace max_min_diff_c_l274_27433

theorem max_min_diff_c (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a^2 + b^2 + c^2 = 18) : 
  ∃ c_max c_min, 
  (∀ c', (a + b + c' = 3 ∧ a^2 + b^2 + c'^2 = 18) → c_min ≤ c' ∧ c' ≤ c_max) 
  ∧ (c_max - c_min = 6) :=
  sorry

end max_min_diff_c_l274_27433


namespace compare_costs_l274_27492

def cost_X (copies: ℕ) : ℝ :=
  if copies >= 40 then
    (copies * 1.25) * 0.95
  else
    copies * 1.25

def cost_Y (copies: ℕ) : ℝ :=
  if copies >= 100 then
    copies * 2.00
  else if copies >= 60 then
    copies * 2.25
  else
    copies * 2.75

def cost_Z (copies: ℕ) : ℝ :=
  if copies >= 50 then
    (copies * 3.00) * 0.90
  else
    copies * 3.00

def cost_W (copies: ℕ) : ℝ :=
  let bulk_groups := copies / 25
  let remainder := copies % 25
  (bulk_groups * 40) + (remainder * 2.00)

theorem compare_costs : 
  cost_X 60 < cost_Y 60 ∧ 
  cost_X 60 < cost_Z 60 ∧ 
  cost_X 60 < cost_W 60 ∧
  cost_Y 60 - cost_X 60 = 63.75 ∧
  cost_Z 60 - cost_X 60 = 90.75 ∧
  cost_W 60 - cost_X 60 = 28.75 :=
  sorry

end compare_costs_l274_27492


namespace sandwich_percentage_not_vegetables_l274_27446

noncomputable def percentage_not_vegetables (total_weight : ℝ) (vegetable_weight : ℝ) : ℝ :=
  (total_weight - vegetable_weight) / total_weight * 100

theorem sandwich_percentage_not_vegetables :
  percentage_not_vegetables 180 50 = 72.22 :=
by
  sorry

end sandwich_percentage_not_vegetables_l274_27446


namespace point_A_outside_circle_l274_27419

noncomputable def circle_radius := 6
noncomputable def distance_OA := 8

theorem point_A_outside_circle : distance_OA > circle_radius :=
by
  -- Solution will go here
  sorry

end point_A_outside_circle_l274_27419


namespace number_of_dogs_l274_27420

def legs_in_pool : ℕ := 24
def human_legs : ℕ := 4
def legs_per_dog : ℕ := 4

theorem number_of_dogs : (legs_in_pool - human_legs) / legs_per_dog = 5 :=
by
  sorry

end number_of_dogs_l274_27420


namespace percentage_cleared_land_l274_27486

theorem percentage_cleared_land (T C : ℝ) (hT : T = 6999.999999999999) (hC : 0.20 * C + 0.70 * C + 630 = C) :
  (C / T) * 100 = 90 :=
by {
  sorry
}

end percentage_cleared_land_l274_27486


namespace correlation_index_l274_27417

-- Define the conditions given in the problem
def height_explains_weight_variation : Prop :=
  ∃ R : ℝ, R^2 = 0.64

-- State the main conjecture (actual proof omitted for simplicity)
theorem correlation_index (R : ℝ) (h : height_explains_weight_variation) : R^2 = 0.64 := by
  sorry

end correlation_index_l274_27417


namespace arithmetic_sequence_common_difference_l274_27421

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_variance : (1/5) * ((a 1 - (a 3)) ^ 2 + (a 2 - (a 3)) ^ 2 + (a 3 - (a 3)) ^ 2 + (a 4 - (a 3)) ^ 2 + (a 5 - (a 3)) ^ 2) = 8) :
  d = 2 ∨ d = -2 := 
sorry

end arithmetic_sequence_common_difference_l274_27421


namespace total_pencils_l274_27403

theorem total_pencils (pencils_per_box : ℕ) (friends : ℕ) (total_pencils : ℕ) : 
  pencils_per_box = 7 ∧ friends = 5 → total_pencils = pencils_per_box + friends * pencils_per_box → total_pencils = 42 :=
by
  intros h1 h2
  sorry

end total_pencils_l274_27403


namespace total_number_of_animals_is_304_l274_27498

theorem total_number_of_animals_is_304
    (dogs frogs : ℕ) 
    (h1 : frogs = 160) 
    (h2 : frogs = 2 * dogs) 
    (cats : ℕ) 
    (h3 : cats = dogs - (dogs / 5)) :
  cats + dogs + frogs = 304 :=
by
  sorry

end total_number_of_animals_is_304_l274_27498


namespace sum_of_fourth_powers_l274_27408

theorem sum_of_fourth_powers (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1 / 2 :=
by sorry

end sum_of_fourth_powers_l274_27408


namespace seq_eq_a1_b1_l274_27466

theorem seq_eq_a1_b1 {a b : ℕ → ℝ} 
  (h1 : ∀ n, a (n + 1) = 2 * b n - a n)
  (h2 : ∀ n, b (n + 1) = 2 * a n - b n)
  (h3 : ∀ n, a n > 0) :
  a 1 = b 1 := 
sorry

end seq_eq_a1_b1_l274_27466


namespace AJHSMETL_19892_reappears_on_line_40_l274_27409
-- Import the entire Mathlib library

-- Define the conditions
def cycleLengthLetters : ℕ := 8
def cycleLengthDigits : ℕ := 5
def lcm_cycles : ℕ := Nat.lcm cycleLengthLetters cycleLengthDigits

-- Problem statement with proof to be filled in later
theorem AJHSMETL_19892_reappears_on_line_40 :
  lcm_cycles = 40 := 
by
  sorry

end AJHSMETL_19892_reappears_on_line_40_l274_27409


namespace oliver_earnings_l274_27436

-- Define the conditions
def cost_per_kilo : ℝ := 2
def kilos_two_days_ago : ℝ := 5
def kilos_yesterday : ℝ := kilos_two_days_ago + 5
def kilos_today : ℝ := 2 * kilos_yesterday

-- Calculate the total kilos washed over the three days
def total_kilos : ℝ := kilos_two_days_ago + kilos_yesterday + kilos_today

-- Calculate the earnings over the three days
def earnings : ℝ := total_kilos * cost_per_kilo

-- The theorem we want to prove
theorem oliver_earnings : earnings = 70 := by
  sorry

end oliver_earnings_l274_27436


namespace problem_l274_27481

-- Define the concept of reciprocal
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the conditions in the problem
def condition1 : Prop := reciprocal 1.5 = 2/3
def condition2 : Prop := reciprocal 1 = 1

-- Theorem stating our goals
theorem problem : condition1 ∧ condition2 :=
by {
  sorry
}

end problem_l274_27481


namespace line_tangent_to_curve_iff_a_zero_l274_27407

noncomputable def f (x : ℝ) := Real.sin (2 * x)
noncomputable def l (x a : ℝ) := 2 * x + a

theorem line_tangent_to_curve_iff_a_zero (a : ℝ) :
  (∃ x₀ : ℝ, deriv f x₀ = 2 ∧ f x₀ = l x₀ a) → a = 0 :=
sorry

end line_tangent_to_curve_iff_a_zero_l274_27407


namespace height_at_age_10_is_around_146_l274_27462

noncomputable def predicted_height (x : ℝ) : ℝ :=
  7.2 * x + 74

theorem height_at_age_10_is_around_146 :
  abs (predicted_height 10 - 146) < ε :=
by
  let ε := 10
  sorry

end height_at_age_10_is_around_146_l274_27462


namespace stream_speed_l274_27490

theorem stream_speed :
  ∀ (v : ℝ),
  (12 - v) / (12 + v) = 1 / 2 →
  v = 4 :=
by
  sorry

end stream_speed_l274_27490


namespace length_vector_eq_three_l274_27480

theorem length_vector_eq_three (A B : ℝ) (hA : A = -1) (hB : B = 2) : |B - A| = 3 :=
by
  sorry

end length_vector_eq_three_l274_27480


namespace greatest_integer_with_gcd_6_l274_27464

theorem greatest_integer_with_gcd_6 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 18 = 6) : n = 138 :=
sorry

end greatest_integer_with_gcd_6_l274_27464


namespace base8_to_base10_362_eq_242_l274_27494

theorem base8_to_base10_362_eq_242 : 
  let digits := [3, 6, 2]
  let base := 8
  let base10_value := (digits[2] * base^0) + (digits[1] * base^1) + (digits[0] * base^2) 
  base10_value = 242 :=
by
  sorry

end base8_to_base10_362_eq_242_l274_27494


namespace simplify_cbrt_8000_eq_21_l274_27416

theorem simplify_cbrt_8000_eq_21 :
  ∃ (a b : ℕ), a * (b^(1/3)) = 20 * (1^(1/3)) ∧ b = 1 ∧ a + b = 21 :=
by
  sorry

end simplify_cbrt_8000_eq_21_l274_27416


namespace other_x_intercept_l274_27493

noncomputable def quadratic_function_vertex :=
  ∃ (a b c : ℝ), ∀ (x : ℝ), (a ≠ 0) →
  (5, -3) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) ∧
  (x = 1) ∧ (a * x^2 + b * x + c = 0) →
  ∃ (x2 : ℝ), x2 = 9

theorem other_x_intercept :
  quadratic_function_vertex :=
sorry

end other_x_intercept_l274_27493


namespace simplify_expression_correct_l274_27474

def simplify_expression (x : ℝ) : Prop :=
  2 * x - 3 * (2 - x) + 4 * (2 + 3 * x) - 5 * (1 - 2 * x) = 27 * x - 3

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
by
  sorry

end simplify_expression_correct_l274_27474


namespace smallest_solution_floor_eq_l274_27441

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l274_27441


namespace sum_ab_equals_five_l274_27476

-- Definitions for conditions
variables {a b : ℝ}

-- Assumption that establishes the solution set for the quadratic inequality
axiom quadratic_solution_set : ∀ x : ℝ, -2 < x ∧ x < 3 ↔ x^2 + b * x - a < 0

-- Statement to be proved
theorem sum_ab_equals_five : a + b = 5 :=
sorry

end sum_ab_equals_five_l274_27476


namespace geometric_sequence_third_term_l274_27428

-- Define the problem statement in Lean 4
theorem geometric_sequence_third_term :
  ∃ r : ℝ, (a = 1024) ∧ (a_5 = 128) ∧ (a_5 = a * r^4) ∧ 
  (a_3 = a * r^2) ∧ (a_3 = 256) :=
sorry

end geometric_sequence_third_term_l274_27428


namespace non_receivers_after_2020_candies_l274_27429

noncomputable def count_non_receivers (k n : ℕ) : ℕ := 
sorry

theorem non_receivers_after_2020_candies :
  count_non_receivers 73 2020 = 36 :=
sorry

end non_receivers_after_2020_candies_l274_27429


namespace total_fish_purchased_l274_27477

/-- Definition of the conditions based on Roden's visits to the pet shop. -/
def first_visit_goldfish := 15
def first_visit_bluefish := 7
def second_visit_goldfish := 10
def second_visit_bluefish := 12
def second_visit_greenfish := 5
def third_visit_goldfish := 3
def third_visit_bluefish := 7
def third_visit_greenfish := 9

/-- Proof statement in Lean 4. -/
theorem total_fish_purchased :
  first_visit_goldfish + first_visit_bluefish +
  second_visit_goldfish + second_visit_bluefish + second_visit_greenfish +
  third_visit_goldfish + third_visit_bluefish + third_visit_greenfish = 68 :=
by
  sorry

end total_fish_purchased_l274_27477


namespace solve_abs_inequality_l274_27442

theorem solve_abs_inequality (x : ℝ) : (|x + 3| + |x - 4| < 8) ↔ (4 ≤ x ∧ x < 4.5) := sorry

end solve_abs_inequality_l274_27442


namespace Total_Cookies_is_135_l274_27440

-- Define the number of cookies in each pack
def PackA_Cookies : ℕ := 15
def PackB_Cookies : ℕ := 30
def PackC_Cookies : ℕ := 45

-- Define the number of packs bought by Paul and Paula
def Paul_PackA_Count : ℕ := 1
def Paul_PackB_Count : ℕ := 2
def Paula_PackA_Count : ℕ := 1
def Paula_PackC_Count : ℕ := 1

-- Calculate total cookies for Paul
def Paul_Cookies : ℕ := (Paul_PackA_Count * PackA_Cookies) + (Paul_PackB_Count * PackB_Cookies)

-- Calculate total cookies for Paula
def Paula_Cookies : ℕ := (Paula_PackA_Count * PackA_Cookies) + (Paula_PackC_Count * PackC_Cookies)

-- Calculate total cookies for Paul and Paula together
def Total_Cookies : ℕ := Paul_Cookies + Paula_Cookies

theorem Total_Cookies_is_135 : Total_Cookies = 135 := by
  sorry

end Total_Cookies_is_135_l274_27440


namespace largest_8_11_double_l274_27452

def is_8_11_double (M : ℕ) : Prop :=
  let digits_8 := (Nat.digits 8 M)
  let M_11 := Nat.ofDigits 11 digits_8
  M_11 = 2 * M

theorem largest_8_11_double : ∃ (M : ℕ), is_8_11_double M ∧ ∀ (N : ℕ), is_8_11_double N → N ≤ M :=
sorry

end largest_8_11_double_l274_27452


namespace combined_value_of_cookies_sold_l274_27415

theorem combined_value_of_cookies_sold:
  ∀ (total_boxes : ℝ) (plain_boxes : ℝ) (price_plain : ℝ) (price_choco : ℝ),
    total_boxes = 1585 →
    plain_boxes = 793.125 →
    price_plain = 0.75 →
    price_choco = 1.25 →
    (plain_boxes * price_plain + (total_boxes - plain_boxes) * price_choco) = 1584.6875 :=
by
  intros total_boxes plain_boxes price_plain price_choco
  intro h1 h2 h3 h4
  sorry

end combined_value_of_cookies_sold_l274_27415


namespace fertilizer_percentage_l274_27484

theorem fertilizer_percentage (total_volume : ℝ) (vol_74 : ℝ) (vol_53 : ℝ) (perc_74 : ℝ) (perc_53 : ℝ) (final_perc : ℝ) :
  total_volume = 42 ∧ vol_74 = 20 ∧ vol_53 = total_volume - vol_74 ∧ perc_74 = 0.74 ∧ perc_53 = 0.53 
  → final_perc = ((vol_74 * perc_74 + vol_53 * perc_53) / total_volume) * 100
  → final_perc = 63.0 :=
by
  intros
  sorry

end fertilizer_percentage_l274_27484


namespace smallest_number_divisible_by_618_3648_60_l274_27418

theorem smallest_number_divisible_by_618_3648_60 :
  ∃ n : ℕ, (∀ m, (m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0 → m = 1038239) :=
by
  use 1038239
  sorry

end smallest_number_divisible_by_618_3648_60_l274_27418


namespace division_addition_problem_l274_27422

-- Define the terms used in the problem
def ten : ℕ := 10
def one_fifth : ℚ := 1 / 5
def six : ℕ := 6

-- Define the math problem
theorem division_addition_problem :
  (ten / one_fifth : ℚ) + six = 56 :=
by sorry

end division_addition_problem_l274_27422


namespace polynomial_value_at_minus_two_l274_27450

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_minus_two :
  f (-2) = -1 :=
by sorry

end polynomial_value_at_minus_two_l274_27450


namespace closed_polygon_inequality_l274_27467

noncomputable def length_eq (A B C D : ℝ × ℝ × ℝ) (l : ℝ) : Prop :=
  dist A B = l ∧ dist B C = l ∧ dist C D = l ∧ dist D A = l

theorem closed_polygon_inequality 
  (A B C D P : ℝ × ℝ × ℝ) (l : ℝ)
  (hABCD : length_eq A B C D l) :
  dist P A < dist P B + dist P C + dist P D :=
sorry

end closed_polygon_inequality_l274_27467


namespace sqrt_square_eq_17_l274_27404

theorem sqrt_square_eq_17 :
  (Real.sqrt 17) ^ 2 = 17 :=
sorry

end sqrt_square_eq_17_l274_27404


namespace find_x_81_9_729_l274_27427

theorem find_x_81_9_729
  (x : ℝ)
  (h : (81 : ℝ)^(x-2) / (9 : ℝ)^(x-2) = (729 : ℝ)^(2*x-1)) :
  x = 1/5 :=
sorry

end find_x_81_9_729_l274_27427


namespace value_of_a3_a6_a9_l274_27461

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end value_of_a3_a6_a9_l274_27461


namespace intersection_eq_l274_27439

variable {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

def M : Set ℝ := { x | -1/2 < x ∧ x < 1/2 }
def N : Set ℝ := { x | 0 ≤ x ∧ x * x ≤ x }

theorem intersection_eq :
  M ∩ N = { x | 0 ≤ x ∧ x < 1/2 } := by
  sorry

end intersection_eq_l274_27439


namespace train_crossing_time_l274_27478

noncomputable def train_speed_kmph : ℕ := 72
noncomputable def platform_length_m : ℕ := 300
noncomputable def crossing_time_platform_s : ℕ := 33
noncomputable def train_speed_mps : ℕ := (train_speed_kmph * 5) / 18

theorem train_crossing_time (L : ℕ) (hL : L + platform_length_m = train_speed_mps * crossing_time_platform_s) :
  L / train_speed_mps = 18 :=
  by
    have : train_speed_mps = 20 := by
      sorry
    have : L = 360 := by
      sorry
    sorry

end train_crossing_time_l274_27478


namespace pirates_treasure_l274_27444

variable (m : ℕ)
variable (h1 : m / 3 + 1 + m / 4 + 5 + m / 5 + 20 = m)

theorem pirates_treasure :
  m = 120 := 
by {
  sorry
}

end pirates_treasure_l274_27444


namespace cost_difference_proof_l274_27456

noncomputable def sailboat_daily_rent : ℕ := 60
noncomputable def ski_boat_hourly_rent : ℕ := 80
noncomputable def sailboat_hourly_fuel_cost : ℕ := 10
noncomputable def ski_boat_hourly_fuel_cost : ℕ := 20
noncomputable def discount : ℕ := 10

noncomputable def rent_time : ℕ := 3
noncomputable def rent_days : ℕ := 2

noncomputable def ken_sailboat_rent_cost :=
  sailboat_daily_rent * rent_days - sailboat_daily_rent * discount / 100

noncomputable def ken_sailboat_fuel_cost :=
  sailboat_hourly_fuel_cost * rent_time * rent_days

noncomputable def ken_total_cost :=
  ken_sailboat_rent_cost + ken_sailboat_fuel_cost

noncomputable def aldrich_ski_boat_rent_cost :=
  ski_boat_hourly_rent * rent_time * rent_days - (ski_boat_hourly_rent * rent_time * discount / 100)

noncomputable def aldrich_ski_boat_fuel_cost :=
  ski_boat_hourly_fuel_cost * rent_time * rent_days

noncomputable def aldrich_total_cost :=
  aldrich_ski_boat_rent_cost + aldrich_ski_boat_fuel_cost

noncomputable def cost_difference :=
  aldrich_total_cost - ken_total_cost

theorem cost_difference_proof : cost_difference = 402 := by
  sorry

end cost_difference_proof_l274_27456


namespace days_vacuuming_l274_27413

theorem days_vacuuming (V : ℕ) (h1 : ∀ V, 130 = 30 * V + 40) : V = 3 :=
by
    have eq1 : 130 = 30 * V + 40 := h1 V
    sorry

end days_vacuuming_l274_27413


namespace paintable_wall_area_l274_27488

/-- Given 4 bedrooms each with length 15 feet, width 11 feet, and height 9 feet,
and doorways and windows occupying 80 square feet in each bedroom,
prove that the total paintable wall area is 1552 square feet. -/
theorem paintable_wall_area
  (bedrooms : ℕ) (length width height doorway_window_area : ℕ) :
  bedrooms = 4 →
  length = 15 →
  width = 11 →
  height = 9 →
  doorway_window_area = 80 →
  4 * (2 * (length * height) + 2 * (width * height) - doorway_window_area) = 1552 :=
by
  intros bedrooms_eq length_eq width_eq height_eq doorway_window_area_eq
  -- Definition of the problem conditions
  have bedrooms_def : bedrooms = 4 := bedrooms_eq
  have length_def : length = 15 := length_eq
  have width_def : width = 11 := width_eq
  have height_def : height = 9 := height_eq
  have doorway_window_area_def : doorway_window_area = 80 := doorway_window_area_eq
  -- Assertion of the correct answer
  sorry

end paintable_wall_area_l274_27488


namespace boys_seen_l274_27458

theorem boys_seen (total_eyes : ℕ) (eyes_per_boy : ℕ) (h1 : total_eyes = 46) (h2 : eyes_per_boy = 2) : total_eyes / eyes_per_boy = 23 := 
by 
  sorry

end boys_seen_l274_27458


namespace angle_P_in_quadrilateral_l274_27434

theorem angle_P_in_quadrilateral : 
  ∀ (P Q R S : ℝ), (P = 3 * Q) → (P = 4 * R) → (P = 6 * S) → (P + Q + R + S = 360) → P = 206 := 
by
  intros P Q R S hP1 hP2 hP3 hSum
  sorry

end angle_P_in_quadrilateral_l274_27434


namespace expression_positive_l274_27475

variable {a b c : ℝ}

theorem expression_positive (h₀ : 0 < a ∧ a < 2) (h₁ : -2 < b ∧ b < 0) : 0 < b + a^2 :=
by
  sorry

end expression_positive_l274_27475


namespace sum_of_coefficients_eq_zero_l274_27455

theorem sum_of_coefficients_eq_zero 
  (A B C D E F : ℝ) :
  (∀ x, (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) 
  = A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by sorry

end sum_of_coefficients_eq_zero_l274_27455
