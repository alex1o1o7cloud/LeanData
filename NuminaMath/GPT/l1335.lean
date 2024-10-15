import Mathlib

namespace NUMINAMATH_GPT_Mikey_leaves_l1335_133569

theorem Mikey_leaves (initial_leaves : ℕ) (leaves_blew_away : ℕ) 
  (h1 : initial_leaves = 356) 
  (h2 : leaves_blew_away = 244) : 
  initial_leaves - leaves_blew_away = 112 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_Mikey_leaves_l1335_133569


namespace NUMINAMATH_GPT_sum_of_coordinates_l1335_133511

noncomputable def endpoint_x (x : ℤ) := (-3 + x) / 2 = 2
noncomputable def endpoint_y (y : ℤ) := (-15 + y) / 2 = -5

theorem sum_of_coordinates : ∃ x y : ℤ, endpoint_x x ∧ endpoint_y y ∧ x + y = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_l1335_133511


namespace NUMINAMATH_GPT_solution_set_Inequality_l1335_133503

theorem solution_set_Inequality : {x : ℝ | abs (1 + x + x^2 / 2) < 1} = {x : ℝ | -2 < x ∧ x < 0} :=
sorry

end NUMINAMATH_GPT_solution_set_Inequality_l1335_133503


namespace NUMINAMATH_GPT_jack_paycheck_l1335_133524

theorem jack_paycheck (P : ℝ) (h1 : 0.15 * 150 + 0.25 * (P - 150) + 30 + 70 / 100 * (P - (0.15 * 150 + 0.25 * (P - 150) + 30)) * 30 / 100 = 50) : P = 242.22 :=
sorry

end NUMINAMATH_GPT_jack_paycheck_l1335_133524


namespace NUMINAMATH_GPT_gcd_problem_l1335_133516

def gcd3 (x y z : ℕ) : ℕ := Int.gcd x (Int.gcd y z)

theorem gcd_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : gcd3 (a^2 - 1) (b^2 - 1) (c^2 - 1) = 1) :
  gcd3 (a * b + c) (b * c + a) (c * a + b) = gcd3 a b c :=
by
  sorry

end NUMINAMATH_GPT_gcd_problem_l1335_133516


namespace NUMINAMATH_GPT_rationalize_denominator_theorem_l1335_133581

noncomputable def rationalize_denominator : Prop :=
  let num := 5
  let den := 2 + Real.sqrt 5
  let conj := 2 - Real.sqrt 5
  let expr := (num * conj) / (den * conj)
  expr = -10 + 5 * Real.sqrt 5

theorem rationalize_denominator_theorem : rationalize_denominator :=
  sorry

end NUMINAMATH_GPT_rationalize_denominator_theorem_l1335_133581


namespace NUMINAMATH_GPT_suff_but_not_nec_l1335_133537

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end NUMINAMATH_GPT_suff_but_not_nec_l1335_133537


namespace NUMINAMATH_GPT_regular_polygon_perimeter_l1335_133555

theorem regular_polygon_perimeter (s : ℝ) (exterior_angle : ℝ) 
  (h1 : s = 7) (h2 : exterior_angle = 45) : 
  8 * s = 56 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_perimeter_l1335_133555


namespace NUMINAMATH_GPT_students_class_division_l1335_133592

theorem students_class_division (n : ℕ) (h1 : n % 15 = 0) (h2 : n % 24 = 0) : n = 120 :=
sorry

end NUMINAMATH_GPT_students_class_division_l1335_133592


namespace NUMINAMATH_GPT_min_value_of_a_l1335_133598

theorem min_value_of_a (a : ℝ) : 
  (∀ x > 1, x + a / (x - 1) ≥ 5) → a ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l1335_133598


namespace NUMINAMATH_GPT_ron_pay_cuts_l1335_133512

-- Define percentages as decimals
def cut_1 : ℝ := 0.05
def cut_2 : ℝ := 0.10
def cut_3 : ℝ := 0.15
def overall_cut : ℝ := 0.27325

-- Define the total number of pay cuts
def total_pay_cuts : ℕ := 3

noncomputable def verify_pay_cuts (cut_1 cut_2 cut_3 overall_cut : ℝ) (total_pay_cuts : ℕ) : Prop :=
  (((1 - cut_1) * (1 - cut_2) * (1 - cut_3) = (1 - overall_cut)) ∧ (total_pay_cuts = 3))

theorem ron_pay_cuts 
  (cut_1 : ℝ := 0.05)
  (cut_2 : ℝ := 0.10)
  (cut_3 : ℝ := 0.15)
  (overall_cut : ℝ := 0.27325)
  (total_pay_cuts : ℕ := 3) 
  : verify_pay_cuts cut_1 cut_2 cut_3 overall_cut total_pay_cuts :=
by sorry

end NUMINAMATH_GPT_ron_pay_cuts_l1335_133512


namespace NUMINAMATH_GPT_largest_N_exists_l1335_133586

noncomputable def parabola_properties (a T : ℤ) :=
    (∀ (x y : ℤ), y = a * x * (x - 2 * T) → (x = 0 ∨ x = 2 * T) → y = 0) ∧ 
    (∀ (v : ℤ × ℤ), v = (2 * T + 1, 28) → 28 = a * (2 * T + 1))

theorem largest_N_exists : 
    ∃ (a T : ℤ), T ≠ 0 ∧ (∀ (P : ℤ × ℤ), P = (0, 0) ∨ P = (2 * T, 0) ∨ P = (2 * T + 1, 28)) 
    ∧ (s = T - a * T^2) ∧ s = 60 :=
sorry

end NUMINAMATH_GPT_largest_N_exists_l1335_133586


namespace NUMINAMATH_GPT_unique_solution_zmod_11_l1335_133582

theorem unique_solution_zmod_11 : 
  ∀ (n : ℕ), 
  (2 ≤ n → 
  (∀ x : ZMod n, (x^2 - 3 * x + 5 = 0) → (∃! x : ZMod n, x^2 - (3 : ZMod n) * x + (5 : ZMod n) = 0)) → 
  n = 11) := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_zmod_11_l1335_133582


namespace NUMINAMATH_GPT_card_dealing_probability_l1335_133501

-- Define the events and their probabilities
def prob_first_card_ace : ℚ := 4 / 52
def prob_second_card_ten_given_ace : ℚ := 4 / 51
def prob_third_card_jack_given_ace_and_ten : ℚ := 2 / 25

-- Define the overall probability
def overall_probability : ℚ :=
  prob_first_card_ace * 
  prob_second_card_ten_given_ace *
  prob_third_card_jack_given_ace_and_ten

-- State the problem
theorem card_dealing_probability :
  overall_probability = 8 / 16575 := by
  sorry

end NUMINAMATH_GPT_card_dealing_probability_l1335_133501


namespace NUMINAMATH_GPT_TimSpentThisMuch_l1335_133528

/-- Tim's lunch cost -/
def lunchCost : ℝ := 50.50

/-- Tip percentage -/
def tipPercent : ℝ := 0.20

/-- Calculate the tip amount -/
def tipAmount := tipPercent * lunchCost

/-- Calculate the total amount spent -/
def totalAmountSpent := lunchCost + tipAmount

/-- Prove that the total amount spent is as expected -/
theorem TimSpentThisMuch : totalAmountSpent = 60.60 :=
  sorry

end NUMINAMATH_GPT_TimSpentThisMuch_l1335_133528


namespace NUMINAMATH_GPT_line_equation_135_deg_l1335_133559

theorem line_equation_135_deg (A : ℝ × ℝ) (theta : ℝ) (l : ℝ → ℝ → Prop) :
  A = (1, -2) →
  theta = 135 →
  (∀ x y, l x y ↔ y = -(x - 1) - 2) →
  ∀ x y, l x y ↔ x + y + 1 = 0 :=
by
  intros hA hTheta hl_form
  sorry

end NUMINAMATH_GPT_line_equation_135_deg_l1335_133559


namespace NUMINAMATH_GPT_calculate_f_one_l1335_133535

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem calculate_f_one : f 1 = 2 := by
  sorry

end NUMINAMATH_GPT_calculate_f_one_l1335_133535


namespace NUMINAMATH_GPT_range_of_reciprocals_l1335_133572

theorem range_of_reciprocals (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) :
  ∃ c ∈ Set.Ici (9 : ℝ), (c = (1/a + 4/b)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_reciprocals_l1335_133572


namespace NUMINAMATH_GPT_muffin_banana_ratio_l1335_133522

variable {R : Type} [LinearOrderedField R]

-- Define the costs of muffins and bananas
variables {m b : R}

-- Susie's cost
def susie_cost (m b : R) := 4 * m + 5 * b

-- Calvin's cost for three times Susie's items
def calvin_cost_tripled (m b : R) := 12 * m + 15 * b

-- Calvin's actual cost
def calvin_cost_actual (m b : R) := 2 * m + 12 * b

theorem muffin_banana_ratio (m b : R) (h : calvin_cost_tripled m b = calvin_cost_actual m b) : m = (3 / 10) * b :=
by sorry

end NUMINAMATH_GPT_muffin_banana_ratio_l1335_133522


namespace NUMINAMATH_GPT_fraction_ordering_l1335_133558

theorem fraction_ordering :
  (8 / 25 : ℚ) < 6 / 17 ∧ 6 / 17 < 10 / 27 ∧ 8 / 25 < 10 / 27 :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l1335_133558


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1335_133593

theorem solve_quadratic_equation :
  ∃ x : ℝ, 2 * x^2 = 4 * x - 1 ∧ (x = (2 + Real.sqrt 2) / 2 ∨ x = (2 - Real.sqrt 2) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1335_133593


namespace NUMINAMATH_GPT_leonardo_initial_money_l1335_133542

theorem leonardo_initial_money (chocolate_cost : ℝ) (borrowed_amount : ℝ) (needed_amount : ℝ)
  (h_chocolate_cost : chocolate_cost = 5)
  (h_borrowed_amount : borrowed_amount = 0.59)
  (h_needed_amount : needed_amount = 0.41) :
  chocolate_cost + borrowed_amount + needed_amount - (chocolate_cost - borrowed_amount) = 4.41 :=
by
  rw [h_chocolate_cost, h_borrowed_amount, h_needed_amount]
  norm_num
  -- Continue with the proof, eventually obtaining the value 4.41
  sorry

end NUMINAMATH_GPT_leonardo_initial_money_l1335_133542


namespace NUMINAMATH_GPT_find_a2_l1335_133576

variable (x : ℝ)
variable (a₀ a₁ a₂ a₃ : ℝ)
axiom condition : ∀ x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

theorem find_a2 : a₂ = 6 :=
by
  -- The proof that involves verifying the Taylor series expansion will come here
  sorry

end NUMINAMATH_GPT_find_a2_l1335_133576


namespace NUMINAMATH_GPT_female_democrats_count_l1335_133543

variable (F M : ℕ)
def total_participants : Prop := F + M = 720
def female_democrats (D_F : ℕ) : Prop := D_F = 1 / 2 * F
def male_democrats (D_M : ℕ) : Prop := D_M = 1 / 4 * M
def total_democrats (D_F D_M : ℕ) : Prop := D_F + D_M = 1 / 3 * 720

theorem female_democrats_count
  (F M D_F D_M : ℕ)
  (h1 : total_participants F M)
  (h2 : female_democrats F D_F)
  (h3 : male_democrats M D_M)
  (h4 : total_democrats D_F D_M) :
  D_F = 120 :=
sorry

end NUMINAMATH_GPT_female_democrats_count_l1335_133543


namespace NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l1335_133505

theorem value_of_x_squared_minus_y_squared (x y : ℝ) 
  (h₁ : x + y = 20) 
  (h₂ : x - y = 6) :
  x^2 - y^2 = 120 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_squared_minus_y_squared_l1335_133505


namespace NUMINAMATH_GPT_customers_not_wanting_change_l1335_133596

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end NUMINAMATH_GPT_customers_not_wanting_change_l1335_133596


namespace NUMINAMATH_GPT_roses_ordered_l1335_133579

theorem roses_ordered (tulips carnations roses : ℕ) (cost_per_flower total_expenses : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : cost_per_flower = 2)
  (h4 : total_expenses = 1890)
  (h5 : total_expenses = (tulips + carnations + roses) * cost_per_flower) :
  roses = 320 :=
by 
  -- Using the mathematical equivalence and conditions provided
  sorry

end NUMINAMATH_GPT_roses_ordered_l1335_133579


namespace NUMINAMATH_GPT_probability_interval_contains_q_l1335_133574

theorem probability_interval_contains_q (P_C P_D : ℝ) (q : ℝ)
    (hC : P_C = 5 / 7) (hD : P_D = 3 / 4) :
    (5 / 28 ≤ q ∧ q ≤ 5 / 7) ↔ (max (P_C + P_D - 1) 0 ≤ q ∧ q ≤ min P_C P_D) :=
by
  sorry

end NUMINAMATH_GPT_probability_interval_contains_q_l1335_133574


namespace NUMINAMATH_GPT_pioneers_club_attendance_l1335_133556

theorem pioneers_club_attendance :
  ∃ (A B : (Fin 11)), A ≠ B ∧
  (∃ (clubs_A clubs_B : Finset (Fin 5)), clubs_A = clubs_B) :=
by
  sorry

end NUMINAMATH_GPT_pioneers_club_attendance_l1335_133556


namespace NUMINAMATH_GPT_solution_for_g0_l1335_133521

variable (g : ℝ → ℝ)

def functional_eq_condition := ∀ x y : ℝ, g (x + y) = g x + g y - 1

theorem solution_for_g0 (h : functional_eq_condition g) : g 0 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_for_g0_l1335_133521


namespace NUMINAMATH_GPT_average_length_of_two_strings_l1335_133550

theorem average_length_of_two_strings (a b : ℝ) (h1 : a = 3.2) (h2 : b = 4.8) :
  (a + b) / 2 = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_average_length_of_two_strings_l1335_133550


namespace NUMINAMATH_GPT_find_B_l1335_133552

noncomputable def g (A B C D x : ℝ) : ℝ :=
  A * x^3 + B * x^2 + C * x + D

theorem find_B (A C D : ℝ) (h1 : ∀ x, g A (-2) C D x = A * (x + 2) * (x - 1) * (x - 2)) 
  (h2 : g A (-2) C D 0 = -8) : 
  (-2 : ℝ) = -2 := 
by
  simp [g] at h2
  sorry

end NUMINAMATH_GPT_find_B_l1335_133552


namespace NUMINAMATH_GPT_find_n_value_l1335_133599

theorem find_n_value (n : ℕ) (h : ∃ k : ℤ, n^2 + 5 * n + 13 = k^2) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_value_l1335_133599


namespace NUMINAMATH_GPT_honey_barrel_problem_l1335_133570

theorem honey_barrel_problem
  (x y : ℝ)
  (h1 : x + y = 56)
  (h2 : x / 2 + y = 34) :
  x = 44 ∧ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_honey_barrel_problem_l1335_133570


namespace NUMINAMATH_GPT_magic_square_l1335_133525

-- Define a 3x3 grid with positions a, b, c and unknowns x, y, z, t, u, v
variables (a b c x y z t u v : ℝ)

-- State the theorem: there exists values for x, y, z, t, u, v
-- such that the sums in each row, column, and both diagonals are the same
theorem magic_square (h1: x = (b + 3*c - 2*a) / 2)
  (h2: y = a + b - c)
  (h3: z = (b + c) / 2)
  (h4: t = 2*c - a)
  (h5: u = b + c - a)
  (h6: v = (2*a + b - c) / 2) :
  x + a + b = y + z + t ∧
  y + z + t = u ∧
  z + t + u = b + z + c ∧
  t + u + v = a + u + c ∧
  x + t + v = u + y + c ∧
  by sorry :=
sorry

end NUMINAMATH_GPT_magic_square_l1335_133525


namespace NUMINAMATH_GPT_solve_ode_l1335_133548

noncomputable def x (t : ℝ) : ℝ :=
  -((1 : ℝ) / 18) * Real.exp (-t) +
  (25 / 54) * Real.exp (5 * t) -
  (11 / 27) * Real.exp (-4 * t)

theorem solve_ode :
  ∀ t : ℝ, 
    (deriv^[2] x t) - (deriv x t) - 20 * x t = Real.exp (-t) ∧
    x 0 = 0 ∧
    (deriv x 0) = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_ode_l1335_133548


namespace NUMINAMATH_GPT_enemies_left_undefeated_l1335_133507

theorem enemies_left_undefeated (points_per_enemy : ℕ) (total_enemies : ℕ) (total_points_earned : ℕ) 
  (h1: points_per_enemy = 9) (h2: total_enemies = 11) (h3: total_points_earned = 72):
  total_enemies - (total_points_earned / points_per_enemy) = 3 :=
by
  sorry

end NUMINAMATH_GPT_enemies_left_undefeated_l1335_133507


namespace NUMINAMATH_GPT_each_friend_pays_l1335_133583

def hamburgers_cost : ℝ := 5 * 3
def fries_cost : ℝ := 4 * 1.20
def soda_cost : ℝ := 5 * 0.50
def spaghetti_cost : ℝ := 1 * 2.70
def total_cost : ℝ := hamburgers_cost + fries_cost + soda_cost + spaghetti_cost
def num_friends : ℝ := 5

theorem each_friend_pays :
  total_cost / num_friends = 5 :=
by
  sorry

end NUMINAMATH_GPT_each_friend_pays_l1335_133583


namespace NUMINAMATH_GPT_chimpanzee_count_l1335_133580

def total_chimpanzees (moving_chimps : ℕ) (staying_chimps : ℕ) : ℕ :=
  moving_chimps + staying_chimps

theorem chimpanzee_count : total_chimpanzees 18 27 = 45 :=
by
  sorry

end NUMINAMATH_GPT_chimpanzee_count_l1335_133580


namespace NUMINAMATH_GPT_find_m_over_n_l1335_133502

noncomputable
def ellipse_intersection_midpoint (m n : ℝ) (P : ℝ × ℝ) : Prop :=
  let M := (P.1, 1 - P.1)
  let N := (1 - P.2, P.2)
  let midpoint_MN := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = midpoint_MN

noncomputable
def ellipse_condition (m n : ℝ) (x y : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

noncomputable
def line_condition (x y : ℝ) : Prop :=
  x + y = 1

noncomputable
def slope_OP_condition (P : ℝ × ℝ) : Prop :=
  P.2 / P.1 = (Real.sqrt 2 / 2)

theorem find_m_over_n
  (m n : ℝ)
  (P : ℝ × ℝ)
  (h1 : ellipse_condition m n P.1 P.2)
  (h2 : line_condition P.1 P.2)
  (h3 : slope_OP_condition P)
  (h4 : ellipse_intersection_midpoint m n P) :
  (m / n = 1) :=
sorry

end NUMINAMATH_GPT_find_m_over_n_l1335_133502


namespace NUMINAMATH_GPT_gold_coins_l1335_133526

theorem gold_coins (n c : Nat) : 
  n = 9 * (c - 2) → n = 6 * c + 3 → n = 45 :=
by 
  intros h1 h2 
  sorry

end NUMINAMATH_GPT_gold_coins_l1335_133526


namespace NUMINAMATH_GPT_probability_of_correct_match_l1335_133557

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_possible_arrangements : ℕ :=
  factorial 4

def correct_arrangements : ℕ :=
  1

def probability_correct_match : ℚ :=
  correct_arrangements / total_possible_arrangements

theorem probability_of_correct_match : probability_correct_match = 1 / 24 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_probability_of_correct_match_l1335_133557


namespace NUMINAMATH_GPT_number_of_people_in_team_l1335_133578

variable (x : ℕ) -- Number of people in the team

-- Conditions as definitions
def average_age_all (x : ℕ) : ℝ := 25
def leader_age : ℝ := 45
def average_age_without_leader (x : ℕ) : ℝ := 23

-- Proof problem statement
theorem number_of_people_in_team (h1 : (x : ℝ) * average_age_all x = x * (average_age_without_leader x - 1) + leader_age) : x = 11 := by
  sorry

end NUMINAMATH_GPT_number_of_people_in_team_l1335_133578


namespace NUMINAMATH_GPT_unique_solution_implies_d_999_l1335_133564

variable (a b c d x y : ℤ)

theorem unique_solution_implies_d_999
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : 3 * x + y = 3005)
  (h5 : y = |x-a| + |x-b| + |x-c| + |x-d|)
  (h6 : ∃! x, 3 * x + |x-a| + |x-b| + |x-c| + |x-d| = 3005) :
  d = 999 :=
sorry

end NUMINAMATH_GPT_unique_solution_implies_d_999_l1335_133564


namespace NUMINAMATH_GPT_bus_capacity_l1335_133513

theorem bus_capacity :
  ∀ (left_seats right_seats people_per_seat back_seat : ℕ),
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat = 11 →
  (left_seats * people_per_seat) + 
  (right_seats * people_per_seat) + 
  back_seat = 92 := by
  intros left_seats right_seats people_per_seat back_seat 
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_bus_capacity_l1335_133513


namespace NUMINAMATH_GPT_range_f_x1_x2_l1335_133560

noncomputable def f (c x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1

theorem range_f_x1_x2 (c x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : x1 < x2) 
  (h4 : 36 - 24 * c > 0) (h5 : ∀ x, f c x = 2 * x ^ 3 - 3 * x ^ 2 + c * x + 1) :
  1 < f c x1 / x2 ∧ f c x1 / x2 < 5 / 2 :=
sorry

end NUMINAMATH_GPT_range_f_x1_x2_l1335_133560


namespace NUMINAMATH_GPT_solve_for_x_l1335_133533

def f (x : ℝ) : ℝ := 3 * x - 5

theorem solve_for_x : ∃ x, 2 * f x - 16 = f (x - 6) ∧ x = 1 := by
  exists 1
  sorry

end NUMINAMATH_GPT_solve_for_x_l1335_133533


namespace NUMINAMATH_GPT_missing_number_is_eight_l1335_133595

theorem missing_number_is_eight (x : ℤ) : (4 + 3) + (x - 3 - 1) = 11 → x = 8 := by
  intro h
  sorry

end NUMINAMATH_GPT_missing_number_is_eight_l1335_133595


namespace NUMINAMATH_GPT_empty_solution_set_of_inequalities_l1335_133551

theorem empty_solution_set_of_inequalities (a : ℝ) : 
  (∀ x : ℝ, ¬ ((2 * x < 5 - 3 * x) ∧ ((x - 1) / 2 > a))) ↔ (0 ≤ a) := 
by
  sorry

end NUMINAMATH_GPT_empty_solution_set_of_inequalities_l1335_133551


namespace NUMINAMATH_GPT_marked_price_each_article_l1335_133534

noncomputable def pair_price : ℝ := 50
noncomputable def discount : ℝ := 0.60
noncomputable def marked_price_pair : ℝ := 50 / 0.40
noncomputable def marked_price_each : ℝ := marked_price_pair / 2

theorem marked_price_each_article : 
  marked_price_each = 62.50 := by
  sorry

end NUMINAMATH_GPT_marked_price_each_article_l1335_133534


namespace NUMINAMATH_GPT_complex_number_calculation_l1335_133597

theorem complex_number_calculation (z : ℂ) (hz : z = 1 - I) : (z^2 / (z - 1)) = 2 := by
  sorry

end NUMINAMATH_GPT_complex_number_calculation_l1335_133597


namespace NUMINAMATH_GPT_part1_part2_part3_l1335_133545

open Real

-- Definition of "$k$-derived point"
def k_derived_point (P : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := (P.1 + k * P.2, k * P.1 + P.2)

-- Problem statements to prove
theorem part1 :
  k_derived_point (-2, 3) 2 = (4, -1) :=
sorry

theorem part2 (P : ℝ × ℝ) (h : k_derived_point P 3 = (9, 11)) :
  P = (3, 2) :=
sorry

theorem part3 (b k : ℝ) (h1 : b > 0) (h2 : |k * b| ≥ 5 * b) :
  k ≥ 5 ∨ k ≤ -5 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1335_133545


namespace NUMINAMATH_GPT_g_f_neg3_l1335_133562

def f (x : ℤ) : ℤ := x^3 - 1
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 1

theorem g_f_neg3 : g (f (-3)) = 2285 :=
by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_g_f_neg3_l1335_133562


namespace NUMINAMATH_GPT_bananas_bought_l1335_133546

def cost_per_banana : ℝ := 5.00
def total_cost : ℝ := 20.00

theorem bananas_bought : total_cost / cost_per_banana = 4 :=
by {
   sorry
}

end NUMINAMATH_GPT_bananas_bought_l1335_133546


namespace NUMINAMATH_GPT_complex_coordinate_l1335_133565

theorem complex_coordinate (i : ℂ) (h : i * i = -1) : i * (1 - i) = 1 + i :=
by sorry

end NUMINAMATH_GPT_complex_coordinate_l1335_133565


namespace NUMINAMATH_GPT_final_result_always_4_l1335_133517

-- The function that performs the operations described in the problem
def transform (x : Nat) : Nat :=
  let step1 := 2 * x
  let step2 := step1 + 3
  let step3 := step2 * 5
  let step4 := step3 + 7
  let last_digit := step4 % 10
  let step6 := last_digit + 18
  step6 / 5

-- The theorem statement claiming that for any single-digit number x, the result of transform x is always 4
theorem final_result_always_4 (x : Nat) (h : x < 10) : transform x = 4 := by
  sorry

end NUMINAMATH_GPT_final_result_always_4_l1335_133517


namespace NUMINAMATH_GPT_find_N_l1335_133531

-- Definitions based on conditions from the problem
def remainder := 6
def dividend := 86
def divisor (Q : ℕ) := 5 * Q
def number_added_to_thrice_remainder (N : ℕ) := 3 * remainder + N
def quotient (Q : ℕ) := Q

-- The condition that relates dividend, divisor, quotient, and remainder
noncomputable def division_equation (Q : ℕ) := dividend = divisor Q * Q + remainder

-- Now, prove the condition
theorem find_N : ∃ N Q : ℕ, division_equation Q ∧ divisor Q = number_added_to_thrice_remainder N ∧ N = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1335_133531


namespace NUMINAMATH_GPT_ferris_wheel_seats_l1335_133553

theorem ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) (h1 : total_people = 16) (h2 : people_per_seat = 4) : (total_people / people_per_seat) = 4 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_seats_l1335_133553


namespace NUMINAMATH_GPT_blueberries_count_l1335_133590

theorem blueberries_count (total_berries raspberries blackberries blueberries : ℕ)
  (h1 : total_berries = 42)
  (h2 : raspberries = total_berries / 2)
  (h3 : blackberries = total_berries / 3)
  (h4 : blueberries = total_berries - raspberries - blackberries) :
  blueberries = 7 :=
sorry

end NUMINAMATH_GPT_blueberries_count_l1335_133590


namespace NUMINAMATH_GPT_A_share_is_9000_l1335_133549

noncomputable def A_share_in_gain (x : ℝ) : ℝ :=
  let total_gain := 27000
  let A_investment_time := 12 * x
  let B_investment_time := 6 * 2 * x
  let C_investment_time := 4 * 3 * x
  let total_investment_time := A_investment_time + B_investment_time + C_investment_time
  total_gain * A_investment_time / total_investment_time

theorem A_share_is_9000 (x : ℝ) : A_share_in_gain x = 27000 / 3 :=
by
  sorry

end NUMINAMATH_GPT_A_share_is_9000_l1335_133549


namespace NUMINAMATH_GPT_similar_triangle_angles_l1335_133585

theorem similar_triangle_angles (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : α + β/2 + γ/2 = Real.pi):
  ∃ (k : ℝ), α = k ∧ β = 2 * k ∧ γ = 4 * k ∧ k = Real.pi / 7 := 
sorry

end NUMINAMATH_GPT_similar_triangle_angles_l1335_133585


namespace NUMINAMATH_GPT_total_original_cost_of_books_l1335_133536

noncomputable def original_cost_price_in_eur (selling_prices : List ℝ) (profit_margin : ℝ) (exchange_rate : ℝ) : ℝ :=
  let original_cost_prices := selling_prices.map (λ price => price / (1 + profit_margin))
  let total_original_cost_usd := original_cost_prices.sum
  total_original_cost_usd * exchange_rate

theorem total_original_cost_of_books : original_cost_price_in_eur [240, 260, 280, 300, 320] 0.20 0.85 = 991.67 :=
  sorry

end NUMINAMATH_GPT_total_original_cost_of_books_l1335_133536


namespace NUMINAMATH_GPT_free_throws_count_l1335_133577

-- Given conditions:
variables (a b x : ℕ) -- α is an abbreviation for natural numbers

-- Condition: number of points from all shots
axiom points_condition : 2 * a + 3 * b + x = 79
-- Condition: three-point shots are twice the points of two-point shots
axiom three_point_condition : 3 * b = 4 * a
-- Condition: number of free throws is one more than the number of two-point shots
axiom free_throw_condition : x = a + 1

-- Prove that the number of free throws is 12
theorem free_throws_count : x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_free_throws_count_l1335_133577


namespace NUMINAMATH_GPT_complex_quadrant_l1335_133568

theorem complex_quadrant (x y: ℝ) (h : x = 1 ∧ y = 2) : x > 0 ∧ y > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1335_133568


namespace NUMINAMATH_GPT_max_volume_of_cuboid_l1335_133589

theorem max_volume_of_cuboid (x y z : ℝ) (h1 : 4 * (x + y + z) = 60) : 
  x * y * z ≤ 125 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_cuboid_l1335_133589


namespace NUMINAMATH_GPT_ben_and_sara_tie_fraction_l1335_133530

theorem ben_and_sara_tie_fraction (ben_wins sara_wins : ℚ) (h1 : ben_wins = 2 / 5) (h2 : sara_wins = 1 / 4) : 
  1 - (ben_wins + sara_wins) = 7 / 20 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_ben_and_sara_tie_fraction_l1335_133530


namespace NUMINAMATH_GPT_polygon_area_is_nine_l1335_133544

-- Definitions of vertices and coordinates.
def vertexA := (0, 0)
def vertexD := (3, 0)
def vertexP := (3, 3)
def vertexM := (0, 3)

-- Area of the polygon formed by the vertices A, D, P, M.
def polygonArea (A D P M : ℕ × ℕ) : ℕ :=
  (D.1 - A.1) * (P.2 - A.2)

-- Statement of the theorem.
theorem polygon_area_is_nine : polygonArea vertexA vertexD vertexP vertexM = 9 := by
  sorry

end NUMINAMATH_GPT_polygon_area_is_nine_l1335_133544


namespace NUMINAMATH_GPT_molecular_weight_N2O3_l1335_133566

variable (atomic_weight_N : ℝ) (atomic_weight_O : ℝ)
variable (n_N_atoms : ℝ) (n_O_atoms : ℝ)
variable (expected_molecular_weight : ℝ)

theorem molecular_weight_N2O3 :
  atomic_weight_N = 14.01 →
  atomic_weight_O = 16.00 →
  n_N_atoms = 2 →
  n_O_atoms = 3 →
  expected_molecular_weight = 76.02 →
  (n_N_atoms * atomic_weight_N + n_O_atoms * atomic_weight_O = expected_molecular_weight) :=
by
  intros
  sorry

end NUMINAMATH_GPT_molecular_weight_N2O3_l1335_133566


namespace NUMINAMATH_GPT_problem_l1335_133540

theorem problem (a b : ℝ) (h1 : abs a = 4) (h2 : b^2 = 9) (h3 : a / b > 0) : a - b = 1 ∨ a - b = -1 := 
sorry

end NUMINAMATH_GPT_problem_l1335_133540


namespace NUMINAMATH_GPT_remainder_of_sum_l1335_133523

theorem remainder_of_sum (f y z : ℤ) 
  (hf : f % 5 = 3)
  (hy : y % 5 = 4)
  (hz : z % 7 = 6) : 
  (f + y + z) % 35 = 13 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l1335_133523


namespace NUMINAMATH_GPT_spaceship_initial_people_count_l1335_133539

/-- For every 100 additional people that board a spaceship, its speed is halved.
     The speed of the spaceship with a certain number of people on board is 500 km per hour.
     The speed of the spaceship when there are 400 people on board is 125 km/hr.
     Prove that the number of people on board when the spaceship was moving at 500 km/hr is 200. -/
theorem spaceship_initial_people_count (speed : ℕ → ℕ) (n : ℕ) :
  (∀ k, speed (k + 100) = speed k / 2) →
  speed n = 500 →
  speed 400 = 125 →
  n = 200 :=
by
  intro half_speed speed_500 speed_400
  sorry

end NUMINAMATH_GPT_spaceship_initial_people_count_l1335_133539


namespace NUMINAMATH_GPT_marble_boxes_l1335_133591

theorem marble_boxes (m : ℕ) : 
  (720 % m = 0) ∧ (m > 1) ∧ (720 / m > 1) ↔ m = 28 := 
sorry

end NUMINAMATH_GPT_marble_boxes_l1335_133591


namespace NUMINAMATH_GPT_division_problem_l1335_133518

theorem division_problem : 160 / (10 + 11 * 2) = 5 := 
  by 
    sorry

end NUMINAMATH_GPT_division_problem_l1335_133518


namespace NUMINAMATH_GPT_f_strictly_increasing_intervals_l1335_133587

noncomputable def f (x : Real) : Real :=
  x * Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real :=
  x * Real.cos x

theorem f_strictly_increasing_intervals :
  ∀ (x : Real), (-π < x ∧ x < -π / 2 ∨ 0 < x ∧ x < π / 2) → f' x > 0 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_f_strictly_increasing_intervals_l1335_133587


namespace NUMINAMATH_GPT_Roberta_spent_on_shoes_l1335_133519

-- Define the conditions as per the problem statement
variables (S B L : ℝ) (h1 : B = S - 17) (h2 : L = B / 4) (h3 : 158 - (S + B + L) = 78)

-- State the theorem to be proved
theorem Roberta_spent_on_shoes : S = 45 :=
by
  -- use variables and conditions
  have := h1
  have := h2
  have := h3
  sorry -- Proof steps can be filled later

end NUMINAMATH_GPT_Roberta_spent_on_shoes_l1335_133519


namespace NUMINAMATH_GPT_solve_quadratic_l1335_133563

theorem solve_quadratic (x : ℝ) : x^2 - 6*x + 5 = 0 ↔ x = 1 ∨ x = 5 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1335_133563


namespace NUMINAMATH_GPT_probability_complement_l1335_133571

theorem probability_complement (P_A : ℝ) (h : P_A = 0.992) : 1 - P_A = 0.008 := by
  sorry

end NUMINAMATH_GPT_probability_complement_l1335_133571


namespace NUMINAMATH_GPT_intersection_M_N_l1335_133527

def M := { x : ℝ | x < 2011 }
def N := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1335_133527


namespace NUMINAMATH_GPT_max_value_of_expression_l1335_133541

theorem max_value_of_expression (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + b = 1) : 
  2 * Real.sqrt (a * b) - 4 * a ^ 2 - b ^ 2 ≤ (Real.sqrt 2 - 1) / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1335_133541


namespace NUMINAMATH_GPT_at_least_1991_red_points_l1335_133514

theorem at_least_1991_red_points (P : Fin 997 → ℝ × ℝ) :
  ∃ (R : Finset (ℝ × ℝ)), 1991 ≤ R.card ∧ (∀ (i j : Fin 997), i ≠ j → ((P i + P j) / 2) ∈ R) :=
sorry

end NUMINAMATH_GPT_at_least_1991_red_points_l1335_133514


namespace NUMINAMATH_GPT_number_division_l1335_133532

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end NUMINAMATH_GPT_number_division_l1335_133532


namespace NUMINAMATH_GPT_solve_x_squared_eq_sixteen_l1335_133520

theorem solve_x_squared_eq_sixteen : ∃ (x1 x2 : ℝ), (x1 = -4 ∧ x2 = 4) ∧ ∀ x : ℝ, x^2 = 16 → (x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_solve_x_squared_eq_sixteen_l1335_133520


namespace NUMINAMATH_GPT_partI_solution_set_partII_range_of_a_l1335_133547

namespace MathProof

-- Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 3)

-- Part (Ⅰ) Proof Problem
theorem partI_solution_set (x : ℝ) : 
  f x (-1) ≤ 1 ↔ -5/2 ≤ x :=
sorry

-- Part (Ⅱ) Proof Problem
theorem partII_range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 4) ↔ -7 ≤ a ∧ a ≤ 7 :=
sorry

end MathProof

end NUMINAMATH_GPT_partI_solution_set_partII_range_of_a_l1335_133547


namespace NUMINAMATH_GPT_C_increases_as_n_increases_l1335_133510

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases_as_n_increases
  (e R r : ℝ) (e_pos : 0 < e) (R_pos : 0 < R) (r_pos : 0 < r) :
  ∀ n : ℝ, 0 < n → ∃ M : ℝ, ∀ N : ℝ, N > n → C e R r N > M :=
by
  sorry

end NUMINAMATH_GPT_C_increases_as_n_increases_l1335_133510


namespace NUMINAMATH_GPT_area_of_triangle_PF1F2_l1335_133554

noncomputable def ellipse := {P : ℝ × ℝ // (4 * P.1^2) / 49 + (P.2^2) / 6 = 1}

noncomputable def area_triangle (P F1 F2 : ℝ × ℝ) :=
  1 / 2 * abs ((F1.1 - P.1) * (F2.2 - P.2) - (F1.2 - P.2) * (F2.1 - P.1))

theorem area_of_triangle_PF1F2 :
  ∀ (F1 F2 : ℝ × ℝ) (P : ellipse), 
    (dist P.1 F1 = 4) →
    (dist P.1 F2 = 3) →
    (dist F1 F2 = 5) →
    area_triangle P.1 F1 F2 = 6 :=
by sorry

end NUMINAMATH_GPT_area_of_triangle_PF1F2_l1335_133554


namespace NUMINAMATH_GPT_family_b_initial_members_l1335_133588

variable (x : ℕ)

theorem family_b_initial_members (h : 6 + (x - 1) + 9 + 12 + 5 + 9 = 48) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_family_b_initial_members_l1335_133588


namespace NUMINAMATH_GPT_correct_input_statement_l1335_133508

-- Definitions based on the conditions
def input_format_A : Prop := sorry
def input_format_B : Prop := sorry
def input_format_C : Prop := sorry
def output_format_D : Prop := sorry

-- The main statement we need to prove
theorem correct_input_statement : input_format_A ∧ ¬ input_format_B ∧ ¬ input_format_C ∧ ¬ output_format_D := 
by sorry

end NUMINAMATH_GPT_correct_input_statement_l1335_133508


namespace NUMINAMATH_GPT_complement_of_P_subset_Q_l1335_133594

-- Definitions based on conditions
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}

-- Theorem statement to prove the correct option C
theorem complement_of_P_subset_Q : {x | ¬ (x < 1)} ⊆ {x | x > -1} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_of_P_subset_Q_l1335_133594


namespace NUMINAMATH_GPT_cost_price_per_meter_l1335_133573

-- Definitions
def selling_price : ℝ := 9890
def meters_sold : ℕ := 92
def profit_per_meter : ℝ := 24

-- Theorem
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 83.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_l1335_133573


namespace NUMINAMATH_GPT_gcd_390_455_546_l1335_133529

theorem gcd_390_455_546 :
  Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
sorry

end NUMINAMATH_GPT_gcd_390_455_546_l1335_133529


namespace NUMINAMATH_GPT_production_days_l1335_133500

variable (n : ℕ) (average_past : ℝ := 50) (production_today : ℝ := 115) (new_average : ℝ := 55)

theorem production_days (h1 : average_past * n + production_today = new_average * (n + 1)) : 
    n = 12 := 
by 
  sorry

end NUMINAMATH_GPT_production_days_l1335_133500


namespace NUMINAMATH_GPT_construction_company_order_l1335_133575

def concrete_weight : ℝ := 0.17
def bricks_weight : ℝ := 0.17
def stone_weight : ℝ := 0.5
def total_weight : ℝ := 0.84

theorem construction_company_order :
  concrete_weight + bricks_weight + stone_weight = total_weight :=
by
  -- The proof would go here but is omitted per instructions.
  sorry

end NUMINAMATH_GPT_construction_company_order_l1335_133575


namespace NUMINAMATH_GPT_at_least_one_admitted_prob_l1335_133515

theorem at_least_one_admitted_prob (pA pB : ℝ) (hA : pA = 0.6) (hB : pB = 0.7) (independent : ∀ (P Q : Prop), P ∧ Q → P ∧ Q):
  (1 - ((1 - pA) * (1 - pB))) = 0.88 :=
by
  rw [hA, hB]
  -- more steps would follow in a complete proof
  sorry

end NUMINAMATH_GPT_at_least_one_admitted_prob_l1335_133515


namespace NUMINAMATH_GPT_only_valid_M_l1335_133504

def digit_sum (n : ℕ) : ℕ :=
  -- definition of digit_sum as a function summing up digits of n
  sorry 

def is_valid_M (M : ℕ) := 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ M → digit_sum (M * k) = digit_sum M

theorem only_valid_M (M : ℕ) :
  is_valid_M M ↔ ∃ n : ℕ, ∀ m : ℕ, M = 10^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_only_valid_M_l1335_133504


namespace NUMINAMATH_GPT_Mateen_garden_area_l1335_133506

theorem Mateen_garden_area :
  ∀ (L W : ℝ), (50 * L = 2000) ∧ (20 * (2 * L + 2 * W) = 2000) → (L * W = 400) :=
by
  intros L W h
  -- We have two conditions based on the problem:
  -- 1. Mateen must walk the length 50 times to cover 2000 meters.
  -- 2. Mateen must walk the perimeter 20 times to cover 2000 meters.
  have h1 : 50 * L = 2000 := h.1
  have h2 : 20 * (2 * L + 2 * W) = 2000 := h.2
  -- We can use these conditions to derive the area of the garden
  sorry

end NUMINAMATH_GPT_Mateen_garden_area_l1335_133506


namespace NUMINAMATH_GPT_one_percent_as_decimal_l1335_133561

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := by
  sorry

end NUMINAMATH_GPT_one_percent_as_decimal_l1335_133561


namespace NUMINAMATH_GPT_option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1335_133538

theorem option_A_incorrect : ¬(Real.sqrt 2 + Real.sqrt 6 = Real.sqrt 8) :=
by sorry

theorem option_B_incorrect : ¬(6 * Real.sqrt 3 - 2 * Real.sqrt 3 = 4) :=
by sorry

theorem option_C_incorrect : ¬(4 * Real.sqrt 2 * 2 * Real.sqrt 3 = 6 * Real.sqrt 6) :=
by sorry

theorem option_D_correct : (1 / (2 - Real.sqrt 3) = 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_GPT_option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l1335_133538


namespace NUMINAMATH_GPT_total_carrots_l1335_133509

-- Definitions from conditions in a)
def JoanCarrots : ℕ := 29
def JessicaCarrots : ℕ := 11

-- Theorem that encapsulates the problem
theorem total_carrots : JoanCarrots + JessicaCarrots = 40 := by
  sorry

end NUMINAMATH_GPT_total_carrots_l1335_133509


namespace NUMINAMATH_GPT_company_p_employees_december_l1335_133584

theorem company_p_employees_december :
  let january_employees := 434.7826086956522
  let percent_more := 0.15
  let december_employees := january_employees + (percent_more * january_employees)
  december_employees = 500 :=
by
  sorry

end NUMINAMATH_GPT_company_p_employees_december_l1335_133584


namespace NUMINAMATH_GPT_room_perimeter_l1335_133567

theorem room_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 12) : 2 * (l + b) = 16 :=
by sorry

end NUMINAMATH_GPT_room_perimeter_l1335_133567
