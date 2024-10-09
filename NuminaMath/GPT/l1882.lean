import Mathlib

namespace probability_C_l1882_188215

-- Definitions of probabilities
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Main proof statement
theorem probability_C :
  ∀ P_C : ℚ, P_A + P_B + P_C + P_D = 1 → P_C = 1 / 4 :=
by
  intro P_C h
  sorry

end probability_C_l1882_188215


namespace sum_of_angles_is_55_l1882_188239

noncomputable def arc_BR : ℝ := 60
noncomputable def arc_RS : ℝ := 50
noncomputable def arc_AC : ℝ := 0
noncomputable def arc_BS := arc_BR + arc_RS
noncomputable def angle_P := (arc_BS - arc_AC) / 2
noncomputable def angle_R := arc_AC / 2
noncomputable def sum_of_angles := angle_P + angle_R

theorem sum_of_angles_is_55 :
  sum_of_angles = 55 :=
by
  sorry

end sum_of_angles_is_55_l1882_188239


namespace time_in_2700_minutes_is_3_am_l1882_188203

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24
def current_hour : ℕ := 6
def minutes_later : ℕ := 2700

-- Calculate the final hour after adding the given minutes
def final_hour (current_hour minutes_later minutes_in_hour hours_in_day: ℕ) : ℕ :=
  (current_hour + (minutes_later / minutes_in_hour) % hours_in_day) % hours_in_day

theorem time_in_2700_minutes_is_3_am :
  final_hour current_hour minutes_later minutes_in_hour hours_in_day = 3 :=
by
  sorry

end time_in_2700_minutes_is_3_am_l1882_188203


namespace solve_for_x_l1882_188201

def custom_mul (a b : ℤ) : ℤ := a * b + a + b

theorem solve_for_x (x : ℤ) :
  custom_mul 3 (3 * x - 1) = 27 → x = 7 / 3 := by
sorry

end solve_for_x_l1882_188201


namespace clear_board_possible_l1882_188225

def operation (board : Array (Array Nat)) (op_type : String) (index : Fin 8) : Array (Array Nat) :=
  match op_type with
  | "column" => board.map (λ row => row.modify index fun x => x - 1)
  | "row" => board.modify index fun row => row.map (λ x => 2 * x)
  | _ => board

def isZeroBoard (board : Array (Array Nat)) : Prop :=
  board.all (λ row => row.all (λ x => x = 0))

theorem clear_board_possible (initial_board : Array (Array Nat)) : 
  ∃ (ops : List (String × Fin 8)), 
    isZeroBoard (ops.foldl (λ b ⟨t, i⟩ => operation b t i) initial_board) :=
sorry

end clear_board_possible_l1882_188225


namespace greatest_multiple_of_30_less_than_1000_l1882_188243

theorem greatest_multiple_of_30_less_than_1000 : ∃ (n : ℕ), n < 1000 ∧ n % 30 = 0 ∧ ∀ m : ℕ, m < 1000 ∧ m % 30 = 0 → m ≤ n := 
by 
  use 990
  sorry

end greatest_multiple_of_30_less_than_1000_l1882_188243


namespace middle_number_is_correct_l1882_188200

theorem middle_number_is_correct (numbers : List ℝ) (h_length : numbers.length = 11)
  (h_avg11 : numbers.sum / 11 = 9.9)
  (first_6 : List ℝ) (h_first6_length : first_6.length = 6)
  (h_avg6_1 : first_6.sum / 6 = 10.5)
  (last_6 : List ℝ) (h_last6_length : last_6.length = 6)
  (h_avg6_2 : last_6.sum / 6 = 11.4) :
  (∃ m : ℝ, m ∈ first_6 ∧ m ∈ last_6 ∧ m = 22.5) :=
by
  sorry

end middle_number_is_correct_l1882_188200


namespace find_k_l1882_188244

-- Definitions for the given conditions
def slope_of_first_line : ℝ := 2
def alpha : ℝ := slope_of_first_line
def slope_of_second_line : ℝ := 2 * alpha

-- The proof goal
theorem find_k (k : ℝ) : slope_of_second_line = k ↔ k = 4 := by
  sorry

end find_k_l1882_188244


namespace white_tshirt_cost_l1882_188211

-- Define the problem conditions
def total_tshirts : ℕ := 200
def total_minutes : ℕ := 25
def black_tshirt_cost : ℕ := 30
def revenue_per_minute : ℕ := 220

-- Prove the cost of white t-shirts given the conditions
theorem white_tshirt_cost : 
  (total_tshirts / 2) * revenue_per_minute * total_minutes 
  - (total_tshirts / 2) * black_tshirt_cost = 2500
  → 2500 / (total_tshirts / 2) = 25 :=
by
  sorry

end white_tshirt_cost_l1882_188211


namespace count_diff_squares_not_representable_1_to_1000_l1882_188267

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l1882_188267


namespace sum_of_purchases_l1882_188285

variable (J : ℕ) (K : ℕ)

theorem sum_of_purchases :
  J = 230 →
  2 * J = K + 90 →
  J + K = 600 :=
by
  intros hJ hEq
  rw [hJ] at hEq
  sorry

end sum_of_purchases_l1882_188285


namespace train_platform_time_l1882_188209

theorem train_platform_time :
  ∀ (L_train L_platform T_tree S D T_platform : ℝ),
    L_train = 1200 ∧ 
    T_tree = 120 ∧ 
    L_platform = 1100 ∧ 
    S = L_train / T_tree ∧ 
    D = L_train + L_platform ∧ 
    T_platform = D / S →
    T_platform = 230 :=
by
  intros
  sorry

end train_platform_time_l1882_188209


namespace problem_solution_l1882_188271

noncomputable def find_a3_and_sum (a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  (∀ x : ℝ, x^5 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4 + a5 * (x + 2)^5) →
  (a3 = 40 ∧ a0 + a1 + a2 + a4 + a5 = -41)

theorem problem_solution {a0 a1 a2 a3 a4 a5 : ℝ} :
  find_a3_and_sum a0 a1 a2 a3 a4 a5 :=
by
  intros h
  sorry

end problem_solution_l1882_188271


namespace exactly_one_germinates_l1882_188270

theorem exactly_one_germinates (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) : 
  (pA * (1 - pB) + (1 - pA) * pB) = 0.26 :=
by
  sorry

end exactly_one_germinates_l1882_188270


namespace find_other_number_l1882_188284

theorem find_other_number
  (n m lcm gcf : ℕ)
  (h_n : n = 40)
  (h_lcm : lcm = 56)
  (h_gcf : gcf = 10)
  (h_lcm_gcf : lcm * gcf = n * m) : m = 14 :=
by
  sorry

end find_other_number_l1882_188284


namespace quadratic_complete_square_l1882_188238

theorem quadratic_complete_square:
  ∃ (a b c : ℝ), (∀ (x : ℝ), 3 * x^2 + 9 * x - 81 = a * (x + b) * (x + b) + c) ∧ a + b + c = -83.25 :=
by {
  sorry
}

end quadratic_complete_square_l1882_188238


namespace min_a_3b_l1882_188275

theorem min_a_3b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / (a + 3) + 1 / (b + 3) = 1 / 4)) : 
  a + 3*b ≥ 12 + 16*Real.sqrt 3 :=
by sorry

end min_a_3b_l1882_188275


namespace min_value_of_function_l1882_188228

theorem min_value_of_function (x : ℝ) (h: x > 1) :
  ∃ t > 0, x = t + 1 ∧ (t + 3 / t + 3) = 3 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l1882_188228


namespace jennifer_money_left_l1882_188259

theorem jennifer_money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ)
    (initial_eq : initial_amount = 90) 
    (sandwich_eq : sandwich_fraction = 1/5) 
    (museum_eq : museum_fraction = 1/6) 
    (book_eq : book_fraction = 1/2) :
    initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction) = 12 := 
by 
  sorry

end jennifer_money_left_l1882_188259


namespace x_intercept_of_line_l2_l1882_188207

theorem x_intercept_of_line_l2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (∀ x y, 2 * x - y + 3 = 0 → l1 x = y) →
  (∀ x y, 2 * x - y - 6 = 0 → l2 x = y) →
  l1 0 = 6 →
  l2 0 = -6 →
  l2 3 = 0 :=
by
  sorry

end x_intercept_of_line_l2_l1882_188207


namespace right_triangle_side_sums_l1882_188220

theorem right_triangle_side_sums (a b c : ℕ) (h1 : a + b = c + 6) (h2 : a^2 + b^2 = c^2) :
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 9 ∧ b = 12 ∧ c = 15) :=
sorry

end right_triangle_side_sums_l1882_188220


namespace parabola_directrix_l1882_188237

theorem parabola_directrix : 
  ∃ d : ℝ, (∀ (y : ℝ), x = - (1 / 4) * y^2 -> 
  ( - (1 / 4) * y^2 = d)) ∧ d = 1 :=
by
  sorry

end parabola_directrix_l1882_188237


namespace range_of_m_l1882_188296

variable (x y m : ℝ)

theorem range_of_m (h1 : Real.sin x = m * (Real.sin y)^3)
                   (h2 : Real.cos x = m * (Real.cos y)^3) :
                   1 ≤ m ∧ m ≤ Real.sqrt 2 :=
by
  sorry

end range_of_m_l1882_188296


namespace tabitha_color_start_l1882_188256

def add_color_each_year (n : ℕ) : ℕ := n + 1

theorem tabitha_color_start 
  (age_start age_now future_colors years_future current_colors : ℕ)
  (h1 : age_start = 15)
  (h2 : age_now = 18)
  (h3 : years_future = 3)
  (h4 : age_now + years_future = 21)
  (h5 : future_colors = 8)
  (h6 : future_colors - years_future = current_colors + 3)
  (h7 : current_colors = 5)
  : age_start + (current_colors - (age_now - age_start)) = 3 := 
by
  sorry

end tabitha_color_start_l1882_188256


namespace polygon_sides_l1882_188208

theorem polygon_sides (n : ℕ) (h : n - 1 = 2022) : n = 2023 :=
by
  sorry

end polygon_sides_l1882_188208


namespace negation_of_proposition_l1882_188223

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end negation_of_proposition_l1882_188223


namespace vieta_formula_l1882_188240

-- Define what it means to be a root of a polynomial
noncomputable def is_root (p : ℝ) (a b c d : ℝ) : Prop :=
  a * p^3 + b * p^2 + c * p + d = 0

-- Setting up the variables and conditions for the polynomial
variables (p q r : ℝ)
variable (a b c d : ℝ)
variable (ha : a = 5)
variable (hb : b = -10)
variable (hc : c = 17)
variable (hd : d = -7)
variable (hp : is_root p a b c d)
variable (hq : is_root q a b c d)
variable (hr : is_root r a b c d)

-- Lean statement to prove the desired equality using Vieta's formulas
theorem vieta_formula : 
  pq + qr + rp = c / a :=
by
  -- Translate the problem into Lean structure
  sorry

end vieta_formula_l1882_188240


namespace find_a_l1882_188274

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.sqrt x

theorem find_a (a : ℝ) (h_intersect : ∃ x₀, f a x₀ = g x₀) (h_tangent : ∃ x₀, (f a x₀) = g x₀ ∧ (1/x₀ * a = 1/ (2 * Real.sqrt x₀))):
  a = Real.exp 1 / 2 :=
by
  sorry

end find_a_l1882_188274


namespace time_taken_y_alone_l1882_188283

-- Define the work done in terms of rates
def work_done (Rx Ry Rz : ℝ) (W : ℝ) :=
  Rx = W / 8 ∧ (Ry + Rz) = W / 6 ∧ (Rx + Rz) = W / 4

-- Prove that the time taken by y alone is 24 hours
theorem time_taken_y_alone (Rx Ry Rz W : ℝ) (h : work_done Rx Ry Rz W) :
  (1 / Ry) = 24 :=
by
  sorry

end time_taken_y_alone_l1882_188283


namespace solution_inequality_l1882_188241

theorem solution_inequality (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0)
    (h : -q / p > -q' / p') : q / p < q' / p' :=
by
  sorry

end solution_inequality_l1882_188241


namespace bottles_produced_l1882_188255

def machine_rate (total_machines : ℕ) (total_bottles_per_minute : ℕ) : ℕ :=
  total_bottles_per_minute / total_machines

def total_bottles (total_machines : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  total_machines * bottles_per_minute * minutes

theorem bottles_produced (machines1 machines2 minutes : ℕ) (bottles1 : ℕ) :
  machine_rate machines1 bottles1 = bottles1 / machines1 →
  total_bottles machines2 (bottles1 / machines1) minutes = 2160 :=
by
  intros machine_rate_eq
  sorry

end bottles_produced_l1882_188255


namespace students_in_grades_2_and_3_l1882_188222

theorem students_in_grades_2_and_3 (boys_2nd : ℕ) (girls_2nd : ℕ) (third_grade_factor : ℕ) 
  (h_boys_2nd : boys_2nd = 20) (h_girls_2nd : girls_2nd = 11) (h_third_grade_factor : third_grade_factor = 2) :
  (boys_2nd + girls_2nd) + ((boys_2nd + girls_2nd) * third_grade_factor) = 93 := by
  sorry

end students_in_grades_2_and_3_l1882_188222


namespace sub_number_l1882_188281

theorem sub_number : 600 - 333 = 267 := by
  sorry

end sub_number_l1882_188281


namespace calc_value_of_ab_bc_ca_l1882_188202

theorem calc_value_of_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 35) (h2 : ab + bc + ca = 320) (h3 : abc = 600) : 
  (a + b) * (b + c) * (c + a) = 10600 := 
by sorry

end calc_value_of_ab_bc_ca_l1882_188202


namespace volunteer_count_change_l1882_188279

theorem volunteer_count_change :
  let x := 1
  let fall_increase := 1.09
  let winter_increase := 1.15
  let spring_decrease := 0.81
  let summer_increase := 1.12
  let summer_end_decrease := 0.95
  let final_ratio := x * fall_increase * winter_increase * spring_decrease * summer_increase * summer_end_decrease
  (final_ratio - x) / x * 100 = 19.13 :=
by
  sorry

end volunteer_count_change_l1882_188279


namespace set_subset_find_m_l1882_188252

open Set

def A (m : ℝ) : Set ℝ := {1, 3, 2 * m + 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem set_subset_find_m (m : ℝ) : (B m ⊆ A m) → (m = 1 ∨ m = 3) :=
by 
  intro h
  sorry

end set_subset_find_m_l1882_188252


namespace area_transformed_function_l1882_188265

noncomputable def area_g : ℝ := 15

noncomputable def area_4g_shifted : ℝ :=
  4 * area_g

theorem area_transformed_function :
  area_4g_shifted = 60 := by
  sorry

end area_transformed_function_l1882_188265


namespace stadium_length_in_feet_l1882_188288

-- Assume the length of the stadium is 80 yards
def stadium_length_yards := 80

-- Assume the conversion factor is 3 feet per yard
def conversion_factor := 3

-- The length in feet is the product of the length in yards and the conversion factor
def length_in_feet := stadium_length_yards * conversion_factor

-- We want to prove that this length in feet is 240 feet
theorem stadium_length_in_feet : length_in_feet = 240 := by
  -- Definitions and conditions are directly restated here; the proof is sketched as 'sorry'
  sorry

end stadium_length_in_feet_l1882_188288


namespace quadratic_rational_root_contradiction_l1882_188289

def int_coefficients (a b c : ℤ) : Prop := true  -- Placeholder for the condition that coefficients are integers

def is_rational_root (a b c p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ p.gcd q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0  -- p/q is a rational root in simplest form

def ear_even (b c : ℤ) : Prop :=
  b % 2 = 0 ∨ c % 2 = 0

def assume_odd (a b c : ℤ) : Prop :=
  a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem quadratic_rational_root_contradiction (a b c p q : ℤ)
  (h1 : int_coefficients a b c)
  (h2 : a ≠ 0)
  (h3 : is_rational_root a b c p q)
  (h4 : ear_even b c) :
  assume_odd a b c :=
sorry

end quadratic_rational_root_contradiction_l1882_188289


namespace boys_meeting_problem_l1882_188280

theorem boys_meeting_problem (d : ℝ) (t : ℝ)
  (speed1 speed2 : ℝ)
  (h1 : speed1 = 6) 
  (h2 : speed2 = 8) 
  (h3 : t > 0)
  (h4 : ∀ n : ℤ, n * (speed1 + speed2) * t ≠ d) : 
  0 = 0 :=
by 
  sorry

end boys_meeting_problem_l1882_188280


namespace one_intersection_point_two_intersection_points_l1882_188214

variables (k : ℝ)

-- Condition definitions
def parabola_eq (y x : ℝ) : Prop := y^2 = -4 * x
def line_eq (x y k : ℝ) : Prop := y + 1 = k * (x - 2)
def discriminant_non_negative (a b c : ℝ) : Prop := b^2 - 4 * a * c ≥ 0

-- Mathematically equivalent proof problem 1
theorem one_intersection_point (k : ℝ) : 
  (k = 1/2 ∨ k = -1 ∨ k = 0) → 
  ∃ x y : ℝ, parabola_eq y x ∧ line_eq x y k := sorry

-- Mathematically equivalent proof problem 2
theorem two_intersection_points (k : ℝ) : 
  (-1 < k ∧ k < 1/2 ∧ k ≠ 0) → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  (x₁ ≠ x₂ ∧ y₁ ≠ y₂) ∧ parabola_eq y₁ x₁ ∧ parabola_eq y₂ x₂ ∧ 
  line_eq x₁ y₁ k ∧ line_eq x₂ y₂ k := sorry

end one_intersection_point_two_intersection_points_l1882_188214


namespace jack_initial_checked_plates_l1882_188290

-- Define Jack's initial and resultant plate counts
variable (C : Nat)
variable (initial_flower_plates : Nat := 4)
variable (broken_flower_plates : Nat := 1)
variable (polka_dotted_plates := 2 * C)
variable (total_plates : Nat := 27)

-- Statement of the problem
theorem jack_initial_checked_plates (h_eq : 3 + C + 2 * C = total_plates) : C = 8 :=
by
  sorry

end jack_initial_checked_plates_l1882_188290


namespace daniel_age_l1882_188231

def isAgeSet (s : Set ℕ) : Prop :=
  s = {4, 6, 8, 10, 12, 14}

def sumTo18 (s : Set ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = 18 ∧ a ≠ b

def youngerThan11 (s : Set ℕ) : Prop :=
  ∀ (a : ℕ), a ∈ s → a < 11

def staysHome (DanielAge : ℕ) (s : Set ℕ) : Prop :=
  6 ∈ s ∧ DanielAge ∈ s

theorem daniel_age :
  ∀ (ages : Set ℕ) (DanielAge : ℕ),
    isAgeSet ages →
    (∃ s, sumTo18 s ∧ s ⊆ ages) →
    (∃ s, youngerThan11 s ∧ s ⊆ ages ∧ 6 ∉ s) →
    staysHome DanielAge ages →
    DanielAge = 12 :=
by
  intros ages DanielAge isAgeSetAges sumTo18Ages youngerThan11Ages staysHomeDaniel
  sorry

end daniel_age_l1882_188231


namespace abs_inequality_equiv_l1882_188276

theorem abs_inequality_equiv (x : ℝ) : 1 ≤ |x - 2| ∧ |x - 2| ≤ 7 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_inequality_equiv_l1882_188276


namespace negation_of_p_is_neg_p_l1882_188277

-- Define the proposition p
def p : Prop := ∀ n : ℕ, 3^n ≥ n^2 + 1

-- Define the negation of p
def neg_p : Prop := ∃ n_0 : ℕ, 3^n_0 < n_0^2 + 1

-- The proof statement
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_is_neg_p_l1882_188277


namespace find_principal_l1882_188227

-- Definitions based on conditions
def simple_interest (P R T : ℚ) : ℚ := (P * R * T) / 100

-- Given conditions
def SI : ℚ := 6016.75
def R : ℚ := 8
def T : ℚ := 5

-- Stating the proof problem
theorem find_principal : 
  ∃ P : ℚ, simple_interest P R T = SI ∧ P = 15041.875 :=
by {
  sorry
}

end find_principal_l1882_188227


namespace opposite_neg_fraction_l1882_188213

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l1882_188213


namespace number_of_hens_l1882_188298

theorem number_of_hens (H C : ℕ) 
  (h1 : H + C = 60) 
  (h2 : 2 * H + 4 * C = 200) : H = 20 :=
sorry

end number_of_hens_l1882_188298


namespace inequality_cube_l1882_188272

theorem inequality_cube (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_l1882_188272


namespace computation_l1882_188248

theorem computation :
  4.165 * 4.8 + 4.165 * 6.7 - 4.165 / (2 / 3) = 41.65 :=
by
  sorry

end computation_l1882_188248


namespace distance_rowed_downstream_l1882_188219

def speed_of_boat_still_water : ℝ := 70 -- km/h
def distance_upstream : ℝ := 240 -- km
def time_upstream : ℝ := 6 -- hours
def time_downstream : ℝ := 5 -- hours

theorem distance_rowed_downstream :
  let V_b := speed_of_boat_still_water
  let V_upstream := distance_upstream / time_upstream
  let V_s := V_b - V_upstream
  let V_downstream := V_b + V_s
  V_downstream * time_downstream = 500 :=
by
  sorry

end distance_rowed_downstream_l1882_188219


namespace total_rainfall_over_3_days_l1882_188292

def rainfall_sunday : ℕ := 4
def rainfall_monday : ℕ := rainfall_sunday + 3
def rainfall_tuesday : ℕ := 2 * rainfall_monday

theorem total_rainfall_over_3_days : rainfall_sunday + rainfall_monday + rainfall_tuesday = 25 := by
  sorry

end total_rainfall_over_3_days_l1882_188292


namespace evaluate_expression_l1882_188216

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := 
by 
  sorry

end evaluate_expression_l1882_188216


namespace complete_square_result_l1882_188230

theorem complete_square_result (x : ℝ) :
  (∃ r s : ℝ, (16 * x ^ 2 + 32 * x - 1280 = 0) → ((x + r) ^ 2 = s) ∧ s = 81) :=
by
  sorry

end complete_square_result_l1882_188230


namespace inverse_variation_l1882_188210

theorem inverse_variation (a b k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) : (∃ a, b = 4 → a = 1 / 8) :=
by
  sorry

end inverse_variation_l1882_188210


namespace distance_point_line_l1882_188247

theorem distance_point_line (m : ℝ) : 
  abs (m + 1) = 2 ↔ (m = 1 ∨ m = -3) := by
  sorry

end distance_point_line_l1882_188247


namespace bridge_max_weight_l1882_188287

variables (M K Mi B : ℝ)

-- Given conditions
def kelly_weight : K = 34 := sorry
def kelly_megan_relation : K = 0.85 * M := sorry
def mike_megan_relation : Mi = M + 5 := sorry
def total_excess : K + M + Mi = B + 19 := sorry

-- Proof goal: The maximum weight the bridge can hold is 100 kg.
theorem bridge_max_weight : B = 100 :=
by
  sorry

end bridge_max_weight_l1882_188287


namespace profit_8000_l1882_188205

noncomputable def profit (selling_price increase : ℝ) : ℝ :=
  (selling_price - 40 + increase) * (500 - 10 * increase)

theorem profit_8000 (increase : ℝ) :
  profit 50 increase = 8000 →
  ((increase = 10 ∧ (50 + increase = 60) ∧ (500 - 10 * increase = 400)) ∨ 
   (increase = 30 ∧ (50 + increase = 80) ∧ (500 - 10 * increase = 200))) :=
by
  sorry

end profit_8000_l1882_188205


namespace range_of_a_real_root_l1882_188251

theorem range_of_a_real_root :
  (∀ x : ℝ, x^2 - a * x + 4 = 0 → ∃ x : ℝ, (x^2 - a * x + 4 = 0 ∧ (a ≥ 4 ∨ a ≤ -4))) ∨
  (∀ x : ℝ, x^2 + (a-2) * x + 4 = 0 → ∃ x : ℝ, (x^2 + (a-2) * x + 4 = 0 ∧ (a ≥ 6 ∨ a ≤ -2))) ∨
  (∀ x : ℝ, x^2 + 2 * a * x + a^2 + 1 = 0 → False) →
  (a ≥ 4 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_real_root_l1882_188251


namespace min_value_of_expression_l1882_188221

theorem min_value_of_expression
  (x y : ℝ) 
  (h : x + y = 1) : 
  ∃ (m : ℝ), m = 2 * x^2 + 3 * y^2 ∧ m = 6 / 5 := 
sorry

end min_value_of_expression_l1882_188221


namespace seeds_per_can_l1882_188273

theorem seeds_per_can (total_seeds : Float) (cans : Float) (h1 : total_seeds = 54.0) (h2 : cans = 9.0) : total_seeds / cans = 6.0 :=
by
  sorry

end seeds_per_can_l1882_188273


namespace cat_birds_total_l1882_188234

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l1882_188234


namespace cody_money_l1882_188268

theorem cody_money (a b c d : ℕ) (h₁ : a = 45) (h₂ : b = 9) (h₃ : c = 19) (h₄ : d = a + b - c) : d = 35 :=
by
  rw [h₁, h₂, h₃] at h₄
  simp at h₄
  exact h₄

end cody_money_l1882_188268


namespace simplify_expression_l1882_188250

-- Define the constants and variables with required conditions
variables {x y z p q r : ℝ}

-- Assume the required distinctness conditions
axiom h1 : x ≠ p 
axiom h2 : y ≠ q 
axiom h3 : z ≠ r 

-- State the theorem to be proven
theorem simplify_expression (h : p ≠ q ∧ q ≠ r ∧ r ≠ p) : 
  (2 * (x - p) / (3 * (r - z))) * (2 * (y - q) / (3 * (p - x))) * (2 * (z - r) / (3 * (q - y))) = -8 / 27 :=
  sorry

end simplify_expression_l1882_188250


namespace length_of_greater_segment_l1882_188266

-- Definitions based on conditions
variable (shorter longer : ℝ)
variable (h1 : longer = shorter + 2)
variable (h2 : (longer^2) - (shorter^2) = 32)

-- Proof goal
theorem length_of_greater_segment : longer = 9 :=
by
  sorry

end length_of_greater_segment_l1882_188266


namespace pet_store_animals_l1882_188235

theorem pet_store_animals (cats dogs birds : ℕ) 
    (ratio_cats_dogs_birds : 2 * birds = 4 * cats ∧ 3 * cats = 2 * dogs) 
    (num_cats : cats = 20) : dogs = 30 ∧ birds = 40 :=
by 
  -- This is where the proof would go, but we can skip it for this problem statement.
  sorry

end pet_store_animals_l1882_188235


namespace minimum_fence_length_l1882_188212

theorem minimum_fence_length {x y : ℝ} (hxy : x * y = 100) : 2 * (x + y) ≥ 40 :=
by
  sorry

end minimum_fence_length_l1882_188212


namespace cookies_fit_in_box_l1882_188295

variable (box_capacity_pounds : ℕ)
variable (cookie_weight_ounces : ℕ)
variable (ounces_per_pound : ℕ)

theorem cookies_fit_in_box (h1 : box_capacity_pounds = 40)
                           (h2 : cookie_weight_ounces = 2)
                           (h3 : ounces_per_pound = 16) :
                           box_capacity_pounds * (ounces_per_pound / cookie_weight_ounces) = 320 := by
  sorry

end cookies_fit_in_box_l1882_188295


namespace tommy_house_price_l1882_188226

variable (P : ℝ)

theorem tommy_house_price 
  (h1 : 1.25 * P = 125000) : 
  P = 100000 :=
by
  sorry

end tommy_house_price_l1882_188226


namespace least_number_divisible_by_13_l1882_188264

theorem least_number_divisible_by_13 (n : ℕ) :
  (∀ m : ℕ, 2 ≤ m ∧ m ≤ 7 → n % m = 2) ∧ (n % 13 = 0) → n = 1262 :=
by sorry

end least_number_divisible_by_13_l1882_188264


namespace tan_alpha_beta_l1882_188224

noncomputable def tan_alpha := -1 / 3
noncomputable def cos_beta := (Real.sqrt 5) / 5
noncomputable def beta := (1:ℝ) -- Dummy representation for being in first quadrant

theorem tan_alpha_beta (h1 : tan_alpha = -1 / 3) 
                       (h2 : cos_beta = (Real.sqrt 5) / 5) 
                       (h3 : 0 < beta ∧ beta < Real.pi / 2) : 
  Real.tan (α + β) = 1 := 
sorry

end tan_alpha_beta_l1882_188224


namespace centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l1882_188236

-- Defining the variables involved
variables {a v r ω T : ℝ}

-- Main theorem statements representing the problem
theorem centripetal_accel_v_r (v r : ℝ) (h₁ : 0 < r) : a = v^2 / r :=
sorry

theorem centripetal_accel_omega_r (ω r : ℝ) (h₁ : 0 < r) : a = r * ω^2 :=
sorry

theorem centripetal_accel_T_r (T r : ℝ) (h₁ : 0 < r) (h₂ : 0 < T) : a = 4 * π^2 * r / T^2 :=
sorry

end centripetal_accel_v_r_centripetal_accel_omega_r_centripetal_accel_T_r_l1882_188236


namespace multiple_of_75_with_36_divisors_l1882_188249

theorem multiple_of_75_with_36_divisors (n : ℕ) (h1 : n % 75 = 0) (h2 : ∃ (a b c : ℕ), a ≥ 1 ∧ b ≥ 2 ∧ n = 3^a * 5^b * (2^c) ∧ (a+1)*(b+1)*(c+1) = 36) : n / 75 = 24 := 
sorry

end multiple_of_75_with_36_divisors_l1882_188249


namespace chocolate_bar_cost_l1882_188217

def total_bars := 11
def bars_left := 7
def bars_sold := total_bars - bars_left
def total_money := 16
def cost := total_money / bars_sold

theorem chocolate_bar_cost : cost = 4 :=
by
  sorry

end chocolate_bar_cost_l1882_188217


namespace solve_for_y_l1882_188229

theorem solve_for_y (y : ℝ) : (5:ℝ)^(2*y + 3) = (625:ℝ)^y → y = 3/2 :=
by
  intro h
  sorry

end solve_for_y_l1882_188229


namespace edward_initial_lives_l1882_188218

def initialLives (lives_lost lives_left : Nat) : Nat :=
  lives_lost + lives_left

theorem edward_initial_lives (lost left : Nat) (H_lost : lost = 8) (H_left : left = 7) :
  initialLives lost left = 15 :=
by
  sorry

end edward_initial_lives_l1882_188218


namespace fraction_meaningful_iff_l1882_188206

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := sorry

end fraction_meaningful_iff_l1882_188206


namespace percentage_of_students_liking_chess_l1882_188297

theorem percentage_of_students_liking_chess (total_students : ℕ) (basketball_percentage : ℝ) (soccer_percentage : ℝ) 
(identified_chess_or_basketball : ℕ) (students_liking_basketball : ℕ) : 
total_students = 250 ∧ basketball_percentage = 0.40 ∧ soccer_percentage = 0.28 ∧ identified_chess_or_basketball = 125 ∧ 
students_liking_basketball = 100 → ∃ C : ℝ, C = 0.10 :=
by
  sorry

end percentage_of_students_liking_chess_l1882_188297


namespace pirates_total_coins_l1882_188263

theorem pirates_total_coins :
  ∀ (x : ℕ), (∃ (paul_coins pete_coins : ℕ), 
  paul_coins = x ∧ pete_coins = 5 * x ∧ pete_coins = (x * (x + 1)) / 2) → x + 5 * x = 54 := by
  sorry

end pirates_total_coins_l1882_188263


namespace negation_proof_equivalence_l1882_188282

theorem negation_proof_equivalence : 
  ¬(∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
sorry

end negation_proof_equivalence_l1882_188282


namespace minimum_blue_chips_l1882_188299

theorem minimum_blue_chips (w r b : ℕ) : 
  (b ≥ w / 3) ∧ (b ≤ r / 4) ∧ (w + b ≥ 75) → b ≥ 19 :=
by sorry

end minimum_blue_chips_l1882_188299


namespace gcd_power_sub_one_l1882_188257

theorem gcd_power_sub_one (a b : ℕ) (h1 : b = a + 30) : 
  Nat.gcd (2^a - 1) (2^b - 1) = 2^30 - 1 := 
by 
  sorry

end gcd_power_sub_one_l1882_188257


namespace find_a_l1882_188278

theorem find_a (m c a b : ℝ) (h_m : m < 0) (h_radius : (m^2 + 3) = 4) 
  (h_c : c = 1 ∨ c = -3) (h_focus : c > 0) (h_ellipse : b^2 = 3) 
  (h_focus_eq : c^2 = a^2 - b^2) : a = 2 :=
by
  sorry

end find_a_l1882_188278


namespace intersection_distance_to_pole_l1882_188245

theorem intersection_distance_to_pole (rho theta : ℝ) (h1 : rho > 0) (h2 : rho = 2 * theta + 1) (h3 : rho * theta = 1) : rho = 2 :=
by
  -- We replace "sorry" with actual proof steps, if necessary.
  sorry

end intersection_distance_to_pole_l1882_188245


namespace tangent_line_equation_l1882_188204

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_equation :
  let x := (1 : ℝ)
  let y := f x
  ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ (2*x - y - 1 = 0) := by
  sorry

end tangent_line_equation_l1882_188204


namespace valid_for_expression_c_l1882_188258

def expression_a_defined (x : ℝ) : Prop := x ≠ 2
def expression_b_defined (x : ℝ) : Prop := x ≠ 3
def expression_c_defined (x : ℝ) : Prop := x ≥ 2
def expression_d_defined (x : ℝ) : Prop := x ≥ 3

theorem valid_for_expression_c :
  (expression_a_defined 2 = false ∧ expression_a_defined 3 = true) ∧
  (expression_b_defined 2 = true ∧ expression_b_defined 3 = false) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) ∧
  (expression_d_defined 2 = false ∧ expression_d_defined 3 = true) ∧
  (expression_c_defined 2 = true ∧ expression_c_defined 3 = true) := by
  sorry

end valid_for_expression_c_l1882_188258


namespace soccer_balls_per_class_l1882_188260

-- Definitions for all conditions in the problem
def elementary_classes_per_school : ℕ := 4
def middle_school_classes_per_school : ℕ := 5
def number_of_schools : ℕ := 2
def total_soccer_balls_donated : ℕ := 90

-- The total number of classes in one school
def classes_per_school : ℕ := elementary_classes_per_school + middle_school_classes_per_school

-- The total number of classes in both schools
def total_classes : ℕ := classes_per_school * number_of_schools

-- Prove that the number of soccer balls donated per class is 5
theorem soccer_balls_per_class : total_soccer_balls_donated / total_classes = 5 :=
  by sorry

end soccer_balls_per_class_l1882_188260


namespace Alma_test_score_l1882_188269

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l1882_188269


namespace remainder_when_n_plus_2947_divided_by_7_l1882_188232

theorem remainder_when_n_plus_2947_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (n + 2947) % 7 = 3 :=
by
  sorry

end remainder_when_n_plus_2947_divided_by_7_l1882_188232


namespace simplify_fractions_l1882_188262

theorem simplify_fractions :
  (20 / 19) * (15 / 28) * (76 / 45) = 95 / 84 :=
by
  sorry

end simplify_fractions_l1882_188262


namespace Sam_weight_l1882_188286

theorem Sam_weight :
  ∃ (sam_weight : ℕ), (∀ (tyler_weight : ℕ), (∀ (peter_weight : ℕ), peter_weight = 65 → tyler_weight = 2 * peter_weight → tyler_weight = sam_weight + 25 → sam_weight = 105)) :=
by {
    sorry
}

end Sam_weight_l1882_188286


namespace length_of_stone_slab_l1882_188242

theorem length_of_stone_slab 
  (num_slabs : ℕ) 
  (total_area : ℝ) 
  (h_num_slabs : num_slabs = 30) 
  (h_total_area : total_area = 50.7): 
  ∃ l : ℝ, l = 1.3 ∧ l * l * num_slabs = total_area := 
by 
  sorry

end length_of_stone_slab_l1882_188242


namespace proof_m_cd_value_l1882_188261

theorem proof_m_cd_value (a b c d m : ℝ) 
  (H1 : a + b = 0) (H2 : c * d = 1) (H3 : |m| = 3) : 
  m + c * d - (a + b) / (m ^ 2) = 4 ∨ m + c * d - (a + b) / (m ^ 2) = -2 :=
by
  sorry

end proof_m_cd_value_l1882_188261


namespace fraction_simplification_l1882_188293

theorem fraction_simplification :
  (3 / 7 + 5 / 8 + 2 / 9) / (5 / 12 + 1 / 4) = 643 / 336 :=
by
  sorry

end fraction_simplification_l1882_188293


namespace find_initial_terms_l1882_188294

theorem find_initial_terms (a : ℕ → ℕ) (h : ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n))
  (a6 : a 6 = 2288) : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
by
  sorry

end find_initial_terms_l1882_188294


namespace imaginary_unit_sum_l1882_188246

-- Define that i is the imaginary unit, which satisfies \(i^2 = -1\)
def is_imaginary_unit (i : ℂ) := i^2 = -1

-- The theorem to be proven: i + i^2 + i^3 + i^4 = 0 given that i is the imaginary unit
theorem imaginary_unit_sum (i : ℂ) (h : is_imaginary_unit i) : 
  i + i^2 + i^3 + i^4 = 0 := 
sorry

end imaginary_unit_sum_l1882_188246


namespace smaller_integer_of_two_digits_l1882_188291

theorem smaller_integer_of_two_digits (a b : ℕ) (ha : 10 ≤ a ∧ a ≤ 99) (hb: 10 ≤ b ∧ b ≤ 99) (h_diff : a ≠ b)
  (h_eq : (a + b) / 2 = a + b / 100) : a = 49 ∨ b = 49 := 
by
  sorry

end smaller_integer_of_two_digits_l1882_188291


namespace f_500_l1882_188253

-- Define a function f on positive integers
def f (n : ℕ) : ℕ := sorry

-- Assume the given conditions
axiom f_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : f (x * y) = f x + f y
axiom f_10 : f 10 = 14
axiom f_40 : f 40 = 20

-- Prove the required result
theorem f_500 : f 500 = 39 := by
  sorry

end f_500_l1882_188253


namespace values_of_m_l1882_188254

theorem values_of_m (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + (2 - m) * x + 12 = 0)) ↔ (m = -10 ∨ m = 14) := 
by
  sorry

end values_of_m_l1882_188254


namespace student_ticket_price_l1882_188233

theorem student_ticket_price
  (S : ℕ)
  (num_tickets : ℕ := 2000)
  (num_student_tickets : ℕ := 520)
  (price_non_student : ℕ := 11)
  (total_revenue : ℕ := 20960)
  (h : 520 * S + (2000 - 520) * 11 = 20960) :
  S = 9 :=
sorry

end student_ticket_price_l1882_188233
