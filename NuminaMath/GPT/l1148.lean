import Mathlib

namespace intersection_A_B_l1148_114819

def A := {x : ℝ | 2 < x ∧ x < 4}
def B := {x : ℝ | (x-1) * (x-3) < 0}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end intersection_A_B_l1148_114819


namespace total_cost_of_office_supplies_l1148_114828

-- Define the conditions
def cost_of_pencil : ℝ := 0.5
def cost_of_folder : ℝ := 0.9
def count_of_pencils : ℕ := 24
def count_of_folders : ℕ := 20

-- Define the theorem to prove
theorem total_cost_of_office_supplies
  (cop : ℝ := cost_of_pencil)
  (cof : ℝ := cost_of_folder)
  (ncp : ℕ := count_of_pencils)
  (ncg : ℕ := count_of_folders) :
  cop * ncp + cof * ncg = 30 :=
sorry

end total_cost_of_office_supplies_l1148_114828


namespace necessary_but_not_sufficient_condition_l1148_114862

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient_condition (a : ℝ) : (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) := 
  by 
    sorry

end necessary_but_not_sufficient_condition_l1148_114862


namespace smallest_possible_value_of_N_l1148_114869

noncomputable def smallest_N (N : ℕ) : Prop :=
  ∃ l m n : ℕ, l * m * n = N ∧ (l - 1) * (m - 1) * (n - 1) = 378

theorem smallest_possible_value_of_N : smallest_N 560 :=
  by {
    sorry
  }

end smallest_possible_value_of_N_l1148_114869


namespace simplify_cosine_tangent_product_of_cosines_l1148_114821

-- Problem 1
theorem simplify_cosine_tangent :
  Real.cos 40 * (1 + Real.sqrt 3 * Real.tan 10) = 1 :=
sorry

-- Problem 2
theorem product_of_cosines :
  (Real.cos (2 * Real.pi / 7)) * (Real.cos (4 * Real.pi / 7)) * (Real.cos (6 * Real.pi / 7)) = 1 / 8 :=
sorry

end simplify_cosine_tangent_product_of_cosines_l1148_114821


namespace find_side_and_area_l1148_114861

-- Conditions
variables {A B C a b c : ℝ} (S : ℝ)
axiom angle_sum : A + B + C = Real.pi
axiom side_a : a = 4
axiom side_b : b = 5
axiom angle_relation : C = 2 * A

-- Proven equalities
theorem find_side_and_area :
  c = 6 ∧ S = 5 * 6 * (Real.sqrt 7) / 4 / 2 := by
  sorry

end find_side_and_area_l1148_114861


namespace polar_to_rectangular_coordinates_l1148_114858

theorem polar_to_rectangular_coordinates 
  (r θ : ℝ) 
  (hr : r = 7) 
  (hθ : θ = 7 * Real.pi / 4) : 
  (r * Real.cos θ, r * Real.sin θ) = (7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2) := 
by
  sorry

end polar_to_rectangular_coordinates_l1148_114858


namespace sophia_finished_more_pages_l1148_114800

noncomputable def length_of_book : ℝ := 89.99999999999999

noncomputable def total_pages : ℕ := 90  -- Considering the practical purpose

noncomputable def finished_pages : ℕ := total_pages * 2 / 3

noncomputable def remaining_pages : ℕ := total_pages - finished_pages

theorem sophia_finished_more_pages :
  finished_pages - remaining_pages = 30 := 
  by
    -- Use sorry here as placeholder for the proof
    sorry

end sophia_finished_more_pages_l1148_114800


namespace solve_for_a_l1148_114873

theorem solve_for_a (a x : ℝ) (h : (1 / 2) * x + a = -1) (hx : x = 2) : a = -2 :=
by
  sorry

end solve_for_a_l1148_114873


namespace union_of_sets_l1148_114809

def A (x : ℤ) : Set ℤ := {x^2, 2*x - 1, -4}
def B (x : ℤ) : Set ℤ := {x - 5, 1 - x, 9}

theorem union_of_sets (x : ℤ) (hx : x = -3) (h_inter : A x ∩ B x = {9}) :
  A x ∪ B x = {-8, -4, 4, -7, 9} :=
by
  sorry

end union_of_sets_l1148_114809


namespace train_speed_l1148_114805

theorem train_speed (length_train length_platform : ℝ) (time : ℝ) 
  (h_length_train : length_train = 170.0416) 
  (h_length_platform : length_platform = 350) 
  (h_time : time = 26) : 
  (length_train + length_platform) / time * 3.6 = 72 :=
by 
  sorry

end train_speed_l1148_114805


namespace average_age_of_9_students_l1148_114844

theorem average_age_of_9_students (avg_age_17_students : ℕ)
                                   (num_students : ℕ)
                                   (avg_age_5_students : ℕ)
                                   (num_5_students : ℕ)
                                   (age_17th_student : ℕ) :
    avg_age_17_students = 17 →
    num_students = 17 →
    avg_age_5_students = 14 →
    num_5_students = 5 →
    age_17th_student = 75 →
    (144 / 9) = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_age_of_9_students_l1148_114844


namespace parabola_equation_l1148_114826

theorem parabola_equation
  (axis_of_symmetry : ∀ x y : ℝ, x = 1)
  (focus : ∀ x y : ℝ, x = -1 ∧ y = 0) :
  ∀ y x : ℝ, y^2 = -4*x := 
sorry

end parabola_equation_l1148_114826


namespace points_on_circle_l1148_114801

theorem points_on_circle (t : ℝ) : ∃ x y : ℝ, x = Real.cos t ∧ y = Real.sin t ∧ x^2 + y^2 = 1 :=
by
  sorry

end points_on_circle_l1148_114801


namespace chomp_game_configurations_l1148_114898

/-- Number of valid configurations such that 0 ≤ a_1 ≤ a_2 ≤ ... ≤ a_5 ≤ 7 is 330 -/
theorem chomp_game_configurations :
  let valid_configs := {a : Fin 6 → Fin 8 // (∀ i j, i ≤ j → a i ≤ a j)}
  Fintype.card valid_configs = 330 :=
sorry

end chomp_game_configurations_l1148_114898


namespace problem_statement_l1148_114880

noncomputable def universal_set : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0 }

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2 }

def B : Set ℤ := {2, 3, 5}

def complement_U_A : Set ℤ := {x : ℤ | x ∈ universal_set ∧ ¬(x ∈ A)}

theorem problem_statement : 
  (complement_U_A ∩ B) = {3, 5} :=
by 
  sorry

end problem_statement_l1148_114880


namespace kim_earrings_l1148_114830

-- Define the number of pairs of earrings on the first day E as a variable
variable (E : ℕ)

-- Define the total number of gumballs Kim receives based on the earrings she brings each day
def total_gumballs_received (E : ℕ) : ℕ :=
  9 * E + 9 * 2 * E + 9 * (2 * E - 1)

-- Define the total number of gumballs Kim eats in 42 days
def total_gumballs_eaten : ℕ :=
  3 * 42

-- Define the statement to be proved
theorem kim_earrings : 
  total_gumballs_received E = total_gumballs_eaten + 9 → E = 3 :=
by sorry

end kim_earrings_l1148_114830


namespace probability_same_color_correct_l1148_114813

-- Defining the contents of Bag A and Bag B
def bagA : List (String × ℕ) := [("white", 1), ("red", 2), ("black", 3)]
def bagB : List (String × ℕ) := [("white", 2), ("red", 3), ("black", 1)]

-- The probability calculation
noncomputable def probability_same_color (bagA bagB : List (String × ℕ)) : ℚ :=
  let p_white := (1 / 6 : ℚ) * (1 / 3 : ℚ)
  let p_red := (1 / 3 : ℚ) * (1 / 2 : ℚ)
  let p_black := (1 / 2 : ℚ) * (1 / 6 : ℚ)
  p_white + p_red + p_black

-- Proof problem statement
theorem probability_same_color_correct :
  probability_same_color bagA bagB = 11 / 36 := 
by 
  sorry

end probability_same_color_correct_l1148_114813


namespace probability_neither_event_l1148_114840

-- Definitions of given probabilities
def P_soccer_match : ℚ := 5 / 8
def P_science_test : ℚ := 1 / 4

-- Calculations of the complements
def P_no_soccer_match : ℚ := 1 - P_soccer_match
def P_no_science_test : ℚ := 1 - P_science_test

-- Independence of events implies the probability of neither event is the product of their complements
theorem probability_neither_event :
  (P_no_soccer_match * P_no_science_test) = 9 / 32 :=
by
  sorry

end probability_neither_event_l1148_114840


namespace find_f_neg4_l1148_114892

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg4 (a b : ℝ) (h : f a b 4 = 0) : f a b (-4) = 2 := by
  -- sorry to skip the proof
  sorry

end find_f_neg4_l1148_114892


namespace gcd_of_ratio_and_lcm_l1148_114885

theorem gcd_of_ratio_and_lcm (A B : ℕ) (k : ℕ) (hA : A = 5 * k) (hB : B = 6 * k) (hlcm : Nat.lcm A B = 180) : Nat.gcd A B = 6 :=
by
  sorry

end gcd_of_ratio_and_lcm_l1148_114885


namespace log_8_4000_l1148_114816

theorem log_8_4000 : ∃ (n : ℤ), 8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 4000 ∧ 4000 < 4096 ∧ n = 4 :=
by
  sorry

end log_8_4000_l1148_114816


namespace minimum_value_fraction_l1148_114845

theorem minimum_value_fraction (m n : ℝ) (h0 : 0 ≤ m) (h1 : 0 ≤ n) (h2 : m + n = 1) :
  ∃ min_val, min_val = (1 / 4) ∧ (∀ m n, 0 ≤ m → 0 ≤ n → m + n = 1 → (m^2) / (m + 2) + (n^2) / (n + 1) ≥ min_val) :=
sorry

end minimum_value_fraction_l1148_114845


namespace option_D_is_greater_than_reciprocal_l1148_114802

theorem option_D_is_greater_than_reciprocal:
  ∀ (x : ℚ), (x = 2) → x > 1/x := by
  intro x
  intro hx
  rw [hx]
  norm_num

end option_D_is_greater_than_reciprocal_l1148_114802


namespace badminton_players_l1148_114884

theorem badminton_players (B T N Both Total: ℕ) 
  (h1: Total = 35)
  (h2: T = 18)
  (h3: N = 5)
  (h4: Both = 3)
  : B = 15 :=
by
  -- The proof block is intentionally left out.
  sorry

end badminton_players_l1148_114884


namespace custom_operation_example_l1148_114836

def custom_operation (x y : Int) : Int :=
  x * y - 3 * x

theorem custom_operation_example : (custom_operation 7 4) - (custom_operation 4 7) = -9 := by
  sorry

end custom_operation_example_l1148_114836


namespace cows_count_l1148_114851

theorem cows_count (initial_cows last_year_deaths last_year_sales this_year_increase purchases gifts : ℕ)
  (h1 : initial_cows = 39)
  (h2 : last_year_deaths = 25)
  (h3 : last_year_sales = 6)
  (h4 : this_year_increase = 24)
  (h5 : purchases = 43)
  (h6 : gifts = 8) : 
  initial_cows - last_year_deaths - last_year_sales + this_year_increase + purchases + gifts = 83 := by
  sorry

end cows_count_l1148_114851


namespace problem1_problem2_problem3_problem4_l1148_114881

-- Problem 1: Prove X = 93 given X - 12 = 81
theorem problem1 (X : ℝ) (h : X - 12 = 81) : X = 93 :=
by
  sorry

-- Problem 2: Prove X = 5.4 given 5.1 + X = 10.5
theorem problem2 (X : ℝ) (h : 5.1 + X = 10.5) : X = 5.4 :=
by
  sorry

-- Problem 3: Prove X = 0.7 given 6X = 4.2
theorem problem3 (X : ℝ) (h : 6 * X = 4.2) : X = 0.7 :=
by
  sorry

-- Problem 4: Prove X = 5 given X ÷ 0.4 = 12.5
theorem problem4 (X : ℝ) (h : X / 0.4 = 12.5) : X = 5 :=
by
  sorry

end problem1_problem2_problem3_problem4_l1148_114881


namespace largest_divisor_of_product_of_three_consecutive_odd_integers_l1148_114878

theorem largest_divisor_of_product_of_three_consecutive_odd_integers :
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 3 ∧ ∀ m : ℕ, m ∣ ((2*n-1)*(2*n+1)*(2*n+3)) → m ≤ d :=
by
  sorry

end largest_divisor_of_product_of_three_consecutive_odd_integers_l1148_114878


namespace museum_college_students_income_l1148_114820

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l1148_114820


namespace even_and_increasing_on_0_inf_l1148_114872

noncomputable def fA (x : ℝ) : ℝ := x^(2/3)
noncomputable def fB (x : ℝ) : ℝ := (1/2)^x
noncomputable def fC (x : ℝ) : ℝ := Real.log x
noncomputable def fD (x : ℝ) : ℝ := -x^2 + 1

theorem even_and_increasing_on_0_inf (f : ℝ → ℝ) : 
  (∀ x, f x = f (-x)) ∧ (∀ a b, (0 < a ∧ a < b) → f a < f b) ↔ f = fA :=
sorry

end even_and_increasing_on_0_inf_l1148_114872


namespace sufficient_condition_implies_range_l1148_114841

def setA : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def setB (a : ℝ) : Set ℝ := {x | x^2 - a * x ≤ x - a}

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x, x ∉ setA → x ∉ setB a) → (1 ≤ a ∧ a < 3) :=
by
  sorry

end sufficient_condition_implies_range_l1148_114841


namespace skateboard_total_distance_l1148_114875

theorem skateboard_total_distance :
  let a_1 := 8
  let d := 6
  let n := 40
  let distance (m : ℕ) := a_1 + (m - 1) * d
  let S_n := n / 2 * (distance 1 + distance n)
  S_n = 5000 := by
  sorry

end skateboard_total_distance_l1148_114875


namespace angle_APB_l1148_114889

-- Define the problem conditions
variables (XY : Π X Y : ℝ, XY = X + Y) -- Line XY is a straight line
          (semicircle_XAZ : Π X A Z : ℝ, semicircle_XAZ = X + Z - A) -- Semicircle XAZ
          (semicircle_ZBY : Π Z B Y : ℝ, semicircle_ZBY = Z + Y - B) -- Semicircle ZBY
          (PA_tangent_XAZ_at_A : Π P A X Z : ℝ, PA_tangent_XAZ_at_A = P + A + X - Z) -- PA tangent to XAZ at A
          (PB_tangent_ZBY_at_B : Π P B Z Y : ℝ, PB_tangent_ZBY_at_B = P + B + Z - Y) -- PB tangent to ZBY at B
          (arc_XA : ℝ := 45) -- Arc XA is 45 degrees
          (arc_BY : ℝ := 60) -- Arc BY is 60 degrees

-- Main theorem to prove
theorem angle_APB : ∀ P A B: ℝ, 
  540 - 90 - 135 - 120 - 90 = 105 := by 
  -- Proof goes here
  sorry

end angle_APB_l1148_114889


namespace dropping_more_than_eating_l1148_114896

theorem dropping_more_than_eating (n : ℕ) : n = 20 → (n * (n + 1)) / 2 > 10 * n := by
  intros h
  rw [h]
  sorry

end dropping_more_than_eating_l1148_114896


namespace smallest_pos_d_l1148_114879

theorem smallest_pos_d (d : ℕ) (h : d > 0) (hd : ∃ k : ℕ, 3150 * d = k * k) : d = 14 := 
by 
  sorry

end smallest_pos_d_l1148_114879


namespace distance_to_destination_l1148_114812

theorem distance_to_destination 
  (speed : ℝ) (time : ℝ) 
  (h_speed : speed = 100) 
  (h_time : time = 5) : 
  speed * time = 500 :=
by
  rw [h_speed, h_time]
  -- This simplifies to 100 * 5 = 500
  norm_num

end distance_to_destination_l1148_114812


namespace fraction_burritos_given_away_l1148_114807

noncomputable def total_burritos_bought : Nat := 3 * 20
noncomputable def burritos_eaten : Nat := 3 * 10
noncomputable def burritos_left : Nat := 10
noncomputable def burritos_before_eating : Nat := burritos_eaten + burritos_left
noncomputable def burritos_given_away : Nat := total_burritos_bought - burritos_before_eating

theorem fraction_burritos_given_away : (burritos_given_away : ℚ) / total_burritos_bought = 1 / 3 := by
  sorry

end fraction_burritos_given_away_l1148_114807


namespace sin_cos_15_eq_quarter_l1148_114893

theorem sin_cos_15_eq_quarter :
  (Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 / 4) :=
by 
  sorry

end sin_cos_15_eq_quarter_l1148_114893


namespace overtime_hourly_rate_l1148_114863

theorem overtime_hourly_rate
  (hourly_rate_first_40_hours: ℝ)
  (hours_first_40: ℝ)
  (gross_pay: ℝ)
  (overtime_hours: ℝ)
  (total_pay_first_40: ℝ := hours_first_40 * hourly_rate_first_40_hours)
  (pay_overtime: ℝ := gross_pay - total_pay_first_40)
  (hourly_rate_overtime: ℝ := pay_overtime / overtime_hours)
  (h1: hourly_rate_first_40_hours = 11.25)
  (h2: hours_first_40 = 40)
  (h3: gross_pay = 622)
  (h4: overtime_hours = 10.75) :
  hourly_rate_overtime = 16 := 
by
  sorry

end overtime_hourly_rate_l1148_114863


namespace tangent_of_curve_at_point_l1148_114806

def curve (x : ℝ) : ℝ := x^3 - 4 * x

def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

theorem tangent_of_curve_at_point : 
  (∃ (x y : ℝ), x = 1 ∧ y = -3 ∧ tangent_line x y) :=
sorry

end tangent_of_curve_at_point_l1148_114806


namespace yerema_can_pay_exactly_l1148_114883

theorem yerema_can_pay_exactly (t k b m : ℤ) 
    (h_foma : 3 * t + 4 * k + 5 * b = 11 * m) : 
    ∃ n : ℤ, 9 * t + k + 4 * b = 11 * n := 
by 
    sorry

end yerema_can_pay_exactly_l1148_114883


namespace negation_equiv_l1148_114835

open Nat

theorem negation_equiv (P : Prop) :
  (¬ (∃ n : ℕ, (n! * n!) > (2^n))) ↔ (∀ n : ℕ, (n! * n!) ≤ (2^n)) :=
by
  sorry

end negation_equiv_l1148_114835


namespace fishing_rod_price_l1148_114860

theorem fishing_rod_price (initial_price : ℝ) 
  (price_increase_percentage : ℝ) 
  (price_decrease_percentage : ℝ) 
  (new_price : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 50) 
  (h2 : price_increase_percentage = 0.20) 
  (h3 : price_decrease_percentage = 0.15) 
  (h4 : new_price = initial_price * (1 + price_increase_percentage)) 
  (h5 : final_price = new_price * (1 - price_decrease_percentage)) 
  : final_price = 51 :=
sorry

end fishing_rod_price_l1148_114860


namespace barbara_current_savings_l1148_114837

def wristwatch_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def initial_saving_duration : ℕ := 10
def further_saving_duration : ℕ := 16

theorem barbara_current_savings : 
  -- Given:
  -- wristwatch_cost: $100
  -- weekly_allowance: $5
  -- further_saving_duration: 16 weeks
  -- Prove:
  -- Barbara currently has $20
  wristwatch_cost - weekly_allowance * further_saving_duration = 20 :=
by
  sorry

end barbara_current_savings_l1148_114837


namespace complementary_not_supplementary_l1148_114832

theorem complementary_not_supplementary (α β : ℝ) (h₁ : α + β = 90) (h₂ : α + β ≠ 180) : (α + β = 180) = false :=
by 
  sorry

end complementary_not_supplementary_l1148_114832


namespace find_essay_pages_l1148_114865

/-
Conditions:
1. It costs $0.10 to print one page.
2. Jenny wants to print 7 copies of her essay.
3. Jenny wants to buy 7 pens that each cost $1.50.
4. Jenny pays the store with 2 twenty dollar bills and gets $12 in change.
-/

def cost_per_page : Float := 0.10
def number_of_copies : Nat := 7
def cost_per_pen : Float := 1.50
def number_of_pens : Nat := 7
def total_money_given : Float := 40.00  -- 2 twenty dollar bills
def change_received : Float := 12.00

theorem find_essay_pages :
  let total_spent := total_money_given - change_received
  let total_cost_of_pens := Float.ofNat number_of_pens * cost_per_pen
  let total_amount_spent_on_printing := total_spent - total_cost_of_pens
  let number_of_pages := total_amount_spent_on_printing / cost_per_page
  number_of_pages = 175 := by
  sorry

end find_essay_pages_l1148_114865


namespace action_figure_ratio_l1148_114894

variable (initial : ℕ) (sold : ℕ) (remaining : ℕ) (left : ℕ)
variable (h1 : initial = 24)
variable (h2 : sold = initial / 4)
variable (h3 : remaining = initial - sold)
variable (h4 : remaining - left = left)

theorem action_figure_ratio
  (h1 : initial = 24)
  (h2 : sold = initial / 4)
  (h3 : remaining = initial - sold)
  (h4 : remaining - left = left) :
  (remaining - left) * 3 = left :=
by
  sorry

end action_figure_ratio_l1148_114894


namespace triangle_side_range_l1148_114877

theorem triangle_side_range (a : ℝ) :
  1 < a ∧ a < 4 ↔ 3 + (2 * a - 1) > 4 ∧ 3 + 4 > 2 * a - 1 ∧ 4 + (2 * a - 1) > 3 :=
by
  sorry

end triangle_side_range_l1148_114877


namespace max_ab_upper_bound_l1148_114842

noncomputable def circle_center_coords : ℝ × ℝ :=
  let center_x := -1
  let center_y := 2
  (center_x, center_y)

noncomputable def max_ab_value (a b : ℝ) : ℝ :=
  if a = 1 - 2 * b then a * b else 0

theorem max_ab_upper_bound :
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 + 1 = 0}
  let line_cond : ℝ × ℝ := (-1, 2)
  (circle_center_coords = line_cond) →
  (∀ a b : ℝ, max_ab_value a b ≤ 1 / 8) :=
by
  intro circle line_cond h
  -- Proof is omitted as per instruction
  sorry

end max_ab_upper_bound_l1148_114842


namespace find_positive_real_x_l1148_114849

noncomputable def positive_solution :=
  ∃ (x : ℝ), (1/3) * (4 * x^2 - 2) = (x^2 - 75 * x - 15) * (x^2 + 50 * x + 10) ∧ x > 0

theorem find_positive_real_x :
  positive_solution ↔ ∃ (x : ℝ), x = (75 + Real.sqrt 5693) / 2 :=
by sorry

end find_positive_real_x_l1148_114849


namespace mark_current_trees_l1148_114870

theorem mark_current_trees (x : ℕ) (h : x + 12 = 25) : x = 13 :=
by {
  -- proof omitted
  sorry
}

end mark_current_trees_l1148_114870


namespace fraction_of_profit_b_received_l1148_114888

theorem fraction_of_profit_b_received (capital months_a_share months_b_share : ℝ) 
  (hA_contrib : capital * (1/4) * months_a_share = capital * (15/4))
  (hB_contrib : capital * (3/4) * months_b_share = capital * (30/4)) :
  (30/45) = (2/3) :=
by sorry

end fraction_of_profit_b_received_l1148_114888


namespace problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l1148_114864

-- Proof statement for problem 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem problem1_question (x y : ℕ) (h : ¬(is_odd x ∧ is_odd y)) : is_odd (x + y) := sorry

theorem problem1_contrapositive (x y : ℕ) (h : is_odd x ∧ is_odd y) : ¬ is_odd (x + y) := sorry

theorem problem1_negation : ∃ (x y : ℕ), ¬(is_odd x ∧ is_odd y) ∧ ¬ is_odd (x + y) := sorry

-- Proof statement for problem 2

structure Square : Type := (is_rhombus : Prop)

def all_squares_are_rhombuses : Prop := ∀ (sq : Square), sq.is_rhombus

theorem problem2_question : all_squares_are_rhombuses = true := sorry

theorem problem2_contrapositive : ¬ all_squares_are_rhombuses = false := sorry

theorem problem2_negation : ¬(∃ (sq : Square), ¬ sq.is_rhombus) = false := sorry

end problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l1148_114864


namespace carlos_class_number_l1148_114891

theorem carlos_class_number (b : ℕ) :
  (100 < b ∧ b < 200) ∧
  (b + 2) % 4 = 0 ∧
  (b + 3) % 5 = 0 ∧
  (b + 4) % 6 = 0 →
  b = 122 ∨ b = 182 :=
by
  -- The proof implementation goes here
  sorry

end carlos_class_number_l1148_114891


namespace syrup_cost_per_week_l1148_114831

theorem syrup_cost_per_week (gallons_per_week : ℕ) (gallons_per_box : ℕ) (cost_per_box : ℕ) 
  (h1 : gallons_per_week = 180) 
  (h2 : gallons_per_box = 30) 
  (h3 : cost_per_box = 40) : 
  (gallons_per_week / gallons_per_box) * cost_per_box = 240 := 
by
  sorry

end syrup_cost_per_week_l1148_114831


namespace scientific_notation_l1148_114824

theorem scientific_notation (h : 0.000000007 = 7 * 10^(-9)) : 0.000000007 = 7 * 10^(-9) :=
by
  sorry

end scientific_notation_l1148_114824


namespace inequality_proof_l1148_114853

noncomputable def a : ℝ := (1 / 2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
noncomputable def b : ℝ := (2 * Real.tan (13 * Real.pi / 180)) / (1 - (Real.tan (13 * Real.pi / 180))^2)
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)

theorem inequality_proof : a < c ∧ c < b := by
  sorry

end inequality_proof_l1148_114853


namespace find_values_of_m_and_n_l1148_114897

theorem find_values_of_m_and_n (m n : ℝ) (h : m / (1 + I) = 1 - n * I) : 
  m = 2 ∧ n = 1 :=
sorry

end find_values_of_m_and_n_l1148_114897


namespace speed_of_stream_l1148_114857

-- Conditions
variables (b s : ℝ)

-- Downstream and upstream conditions
def downstream_speed := 150 = (b + s) * 5
def upstream_speed := 75 = (b - s) * 7

-- Goal statement
theorem speed_of_stream (h1 : downstream_speed b s) (h2 : upstream_speed b s) : s = 135/14 :=
by sorry

end speed_of_stream_l1148_114857


namespace ratio_of_x_to_y_l1148_114810

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 := 
by
  sorry

end ratio_of_x_to_y_l1148_114810


namespace sid_fraction_left_l1148_114822

noncomputable def fraction_left (original total_spent remaining additional : ℝ) : ℝ :=
  (remaining - additional) / original

theorem sid_fraction_left 
  (original : ℝ := 48) 
  (spent_computer : ℝ := 12) 
  (spent_snacks : ℝ := 8) 
  (remaining : ℝ := 28) 
  (additional : ℝ := 4) :
  fraction_left original (spent_computer + spent_snacks) remaining additional = 1 / 2 :=
by
  sorry

end sid_fraction_left_l1148_114822


namespace trigonometric_identity_l1148_114829

theorem trigonometric_identity :
  Real.sin (17 * Real.pi / 180) * Real.sin (223 * Real.pi / 180) + 
  Real.sin (253 * Real.pi / 180) * Real.sin (313 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l1148_114829


namespace fraction_of_cream_in_cup1_l1148_114804

/-
Problem statement:
Sarah places five ounces of coffee into an eight-ounce cup (Cup 1) and five ounces of cream into a second cup (Cup 2).
After pouring half the coffee from Cup 1 to Cup 2, one ounce of cream is added to Cup 2.
After stirring Cup 2 thoroughly, Sarah then pours half the liquid in Cup 2 back into Cup 1.
Prove that the fraction of the liquid in Cup 1 that is now cream is 4/9.
-/

theorem fraction_of_cream_in_cup1
  (initial_coffee_cup1 : ℝ)
  (initial_cream_cup2 : ℝ)
  (half_initial_coffee : ℝ)
  (added_cream : ℝ)
  (total_mixture : ℝ)
  (half_mixture : ℝ)
  (coffee_fraction : ℝ)
  (cream_fraction : ℝ)
  (coffee_transferred_back : ℝ)
  (cream_transferred_back : ℝ)
  (total_coffee_in_cup1 : ℝ)
  (total_cream_in_cup1 : ℝ)
  (total_liquid_in_cup1 : ℝ)
  :
  initial_coffee_cup1 = 5 →
  initial_cream_cup2 = 5 →
  half_initial_coffee = initial_coffee_cup1 / 2 →
  added_cream = 1 →
  total_mixture = initial_cream_cup2 + half_initial_coffee + added_cream →
  half_mixture = total_mixture / 2 →
  coffee_fraction = half_initial_coffee / total_mixture →
  cream_fraction = (total_mixture - half_initial_coffee) / total_mixture →
  coffee_transferred_back = half_mixture * coffee_fraction →
  cream_transferred_back = half_mixture * cream_fraction →
  total_coffee_in_cup1 = initial_coffee_cup1 - half_initial_coffee + coffee_transferred_back →
  total_cream_in_cup1 = cream_transferred_back →
  total_liquid_in_cup1 = total_coffee_in_cup1 + total_cream_in_cup1 →
  total_cream_in_cup1 / total_liquid_in_cup1 = 4 / 9 :=
by {
  sorry
}

end fraction_of_cream_in_cup1_l1148_114804


namespace james_muffins_l1148_114811

theorem james_muffins (arthur_muffins : ℕ) (times : ℕ) (james_muffins : ℕ) 
  (h1 : arthur_muffins = 115) 
  (h2 : times = 12) 
  (h3 : james_muffins = arthur_muffins * times) : 
  james_muffins = 1380 := 
by 
  sorry

end james_muffins_l1148_114811


namespace recommendation_plans_count_l1148_114882

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count_l1148_114882


namespace stratified_sampling_l1148_114868

noncomputable def employees := 500
noncomputable def under_35 := 125
noncomputable def between_35_and_49 := 280
noncomputable def over_50 := 95
noncomputable def sample_size := 100

theorem stratified_sampling : 
  under_35 * sample_size / employees = 25 := by
  sorry

end stratified_sampling_l1148_114868


namespace angle_B_is_30_degrees_l1148_114815

variable (a b : ℝ)
variable (A B : ℝ)

axiom a_value : a = 2 * Real.sqrt 3
axiom b_value : b = Real.sqrt 6
axiom A_value : A = Real.pi / 4

theorem angle_B_is_30_degrees (h1 : a = 2 * Real.sqrt 3) (h2 : b = Real.sqrt 6) (h3 : A = Real.pi / 4) : B = Real.pi / 6 :=
  sorry

end angle_B_is_30_degrees_l1148_114815


namespace min_value_a_plus_b_l1148_114866

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) : a + b ≥ 2 + Real.sqrt 3 :=
sorry

end min_value_a_plus_b_l1148_114866


namespace find_A_l1148_114818

theorem find_A (A : ℕ) (h : 10 * A + 2 - 23 = 549) : A = 5 :=
by sorry

end find_A_l1148_114818


namespace p_necessary_not_sufficient_q_l1148_114887

-- Define the conditions p and q
def p (a : ℝ) : Prop := a < 1
def q (a : ℝ) : Prop := 0 < a ∧ a < 1

-- State the necessary but not sufficient condition theorem
theorem p_necessary_not_sufficient_q (a : ℝ) : p a → q a → p a ∧ ¬∀ (a : ℝ), p a → q a :=
by
  sorry

end p_necessary_not_sufficient_q_l1148_114887


namespace problem_1_max_value_problem_2_good_sets_count_l1148_114855

noncomputable def goodSetMaxValue : ℤ :=
  2012

noncomputable def goodSetCount : ℤ :=
  1006

theorem problem_1_max_value {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetMaxValue = 2012 :=
sorry

theorem problem_2_good_sets_count {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetCount = 1006 :=
sorry

end problem_1_max_value_problem_2_good_sets_count_l1148_114855


namespace correct_statement_exam_l1148_114847

theorem correct_statement_exam 
  (students_participated : ℕ)
  (students_sampled : ℕ)
  (statement1 : Bool)
  (statement2 : Bool)
  (statement3 : Bool)
  (statement4 : Bool)
  (cond1 : students_participated = 70000)
  (cond2 : students_sampled = 1000)
  (cond3 : statement1 = False)
  (cond4 : statement2 = False)
  (cond5 : statement3 = False)
  (cond6 : statement4 = True) :
  statement4 = True := 
sorry

end correct_statement_exam_l1148_114847


namespace quadratic_equation_in_x_l1148_114848

theorem quadratic_equation_in_x (k x : ℝ) : 
  (k^2 + 1) * x^2 - (k * x - 8) - 1 = 0 := 
sorry

end quadratic_equation_in_x_l1148_114848


namespace correct_expression_l1148_114899

theorem correct_expression (a b c m x y : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b ≠ 0) (h5 : x ≠ y) : 
  ¬ ( (a + m) / (b + m) = a / b ) ∧
  ¬ ( (a + b) / (a + b) = 0 ) ∧ 
  ¬ ( (a * b - 1) / (a * c - 1) = (b - 1) / (c - 1) ) ∧ 
  ( (x - y) / (x^2 - y^2) = 1 / (x + y) ) :=
by
  sorry

end correct_expression_l1148_114899


namespace parallel_lines_m_value_l1148_114808

noncomputable def m_value_parallel (m : ℝ) : Prop :=
  (m-1) / 2 = 1 / -3

theorem parallel_lines_m_value :
  ∀ (m : ℝ), (m_value_parallel m) → m = 1 / 3 :=
by
  intro m
  intro h
  sorry

end parallel_lines_m_value_l1148_114808


namespace findNumberOfIntegers_l1148_114871

def arithmeticSeq (a d n : ℕ) : ℕ :=
  a + d * n

def isInSeq (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 33 ∧ n = arithmeticSeq 1 3 k

def validInterval (n : ℕ) : Bool :=
  (n + 1) / 3 % 2 = 1

theorem findNumberOfIntegers :
  ∃ count : ℕ, count = 66 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ ¬isInSeq n → validInterval n = true) :=
sorry

end findNumberOfIntegers_l1148_114871


namespace ratio_proof_l1148_114859

variable (x y z : ℝ)
variable (h1 : y / z = 1 / 2)
variable (h2 : z / x = 2 / 3)
variable (h3 : x / y = 3 / 1)

theorem ratio_proof : (x / (y * z)) / (y / (z * x)) = 4 / 1 := 
  sorry

end ratio_proof_l1148_114859


namespace chord_midpoint_line_l1148_114867

open Real 

theorem chord_midpoint_line (x y : ℝ) (P : ℝ × ℝ) 
  (hP : P = (1, 1)) (hcircle : ∀ (x y : ℝ), x^2 + y^2 = 10) :
  x + y - 2 = 0 :=
by
  sorry

end chord_midpoint_line_l1148_114867


namespace adil_older_than_bav_by_732_days_l1148_114876

-- Definitions based on the problem conditions
def adilBirthDate : String := "December 31, 2015"
def bavBirthDate : String := "January 1, 2018"

-- Main theorem statement 
theorem adil_older_than_bav_by_732_days :
    let daysIn2016 := 366
    let daysIn2017 := 365
    let transition := 1
    let totalDays := daysIn2016 + daysIn2017 + transition
    totalDays = 732 :=
by
    sorry

end adil_older_than_bav_by_732_days_l1148_114876


namespace range_of_k_l1148_114852

theorem range_of_k 
  (x1 x2 y1 y2 k : ℝ)
  (h1 : y1 = 2 * x1 - k * x1 + 1)
  (h2 : y2 = 2 * x2 - k * x2 + 1)
  (h3 : x1 ≠ x2)
  (h4 : (x1 - x2) * (y1 - y2) < 0) : k > 2 := 
sorry

end range_of_k_l1148_114852


namespace biased_die_probability_l1148_114838

theorem biased_die_probability (P2 : ℝ) (h1 : P2 ≠ 1 / 6) (h2 : 3 * P2 * (1 - P2) ^ 2 = 1 / 4) : 
  P2 = 0.211 :=
sorry

end biased_die_probability_l1148_114838


namespace min_value_of_expression_l1148_114854

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h_eq : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3)

theorem min_value_of_expression :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end min_value_of_expression_l1148_114854


namespace ratio_of_erasers_l1148_114839

theorem ratio_of_erasers (a n : ℕ) (ha : a = 4) (hn : n = a + 12) :
  n / a = 4 :=
by
  sorry

end ratio_of_erasers_l1148_114839


namespace max_gcd_expression_l1148_114825

theorem max_gcd_expression (n : ℕ) (h1 : n > 0) (h2 : n % 3 = 1) : 
  Nat.gcd (15 * n + 5) (9 * n + 4) = 5 :=
by
  sorry

end max_gcd_expression_l1148_114825


namespace gcd_eight_digit_repeating_four_digit_l1148_114846

theorem gcd_eight_digit_repeating_four_digit :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → (∀ m : ℕ, (1000 ≤ m ∧ m < 10000) →
  Nat.gcd (10001 * n) (10001 * m) = 10001) :=
by
  intros n hn m hm
  sorry

end gcd_eight_digit_repeating_four_digit_l1148_114846


namespace problem_statement_l1148_114886

theorem problem_statement (x : ℝ) (h : x = Real.sqrt 3 + 1) : x^2 - 2*x + 1 = 3 :=
sorry

end problem_statement_l1148_114886


namespace quotient_remainder_increase_l1148_114874

theorem quotient_remainder_increase (a b q r q' r' : ℕ) (hb : b ≠ 0) 
    (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) (h4 : 3 * a = 3 * b * q' + r') 
    (h5 : 0 ≤ r') (h6 : r' < 3 * b) :
    q' = q ∧ r' = 3 * r := by
  sorry

end quotient_remainder_increase_l1148_114874


namespace smallest_number_diminished_by_8_divisible_by_9_6_12_18_l1148_114823

theorem smallest_number_diminished_by_8_divisible_by_9_6_12_18 :
  ∃ x : ℕ, (x - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 ∧ ∀ y : ℕ, (y - 8) % Nat.lcm (Nat.lcm 9 6) (Nat.lcm 12 18) = 0 → x ≤ y → x = 44 :=
by
  sorry

end smallest_number_diminished_by_8_divisible_by_9_6_12_18_l1148_114823


namespace sandy_puppies_l1148_114843

theorem sandy_puppies :
  ∀ (initial_puppies puppies_given_away remaining_puppies : ℕ),
  initial_puppies = 8 →
  puppies_given_away = 4 →
  remaining_puppies = initial_puppies - puppies_given_away →
  remaining_puppies = 4 :=
by
  intros initial_puppies puppies_given_away remaining_puppies
  intro h_initial
  intro h_given_away
  intro h_remaining
  rw [h_initial, h_given_away] at h_remaining
  exact h_remaining

end sandy_puppies_l1148_114843


namespace pyramid_area_ratio_l1148_114850

theorem pyramid_area_ratio (S S1 S2 : ℝ) (h1 : S1 = (99 / 100)^2 * S) (h2 : S2 = (1 / 100)^2 * S) :
  S1 / S2 = 9801 := by
  sorry

end pyramid_area_ratio_l1148_114850


namespace average_percentage_decrease_l1148_114856

theorem average_percentage_decrease (p1 p2 : ℝ) (n : ℕ) (h₀ : p1 = 2000) (h₁ : p2 = 1280) (h₂ : n = 2) :
  ((p1 - p2) / p1 * 100) / n = 18 := 
by
  sorry

end average_percentage_decrease_l1148_114856


namespace sequence_sum_l1148_114890

theorem sequence_sum (A B C D E F G H : ℕ) (hC : C = 7) 
    (h_sum : A + B + C = 36 ∧ B + C + D = 36 ∧ C + D + E = 36 ∧ D + E + F = 36 ∧ E + F + G = 36 ∧ F + G + H = 36) :
    A + H = 29 :=
sorry

end sequence_sum_l1148_114890


namespace syllogism_correct_l1148_114833

theorem syllogism_correct 
  (natnum : ℕ → Prop) 
  (intnum : ℤ → Prop) 
  (is_natnum  : natnum 4) 
  (natnum_to_intnum : ∀ n, natnum n → intnum n) : intnum 4 :=
by
  sorry

end syllogism_correct_l1148_114833


namespace lateral_surface_area_ratio_l1148_114895

theorem lateral_surface_area_ratio (r h : ℝ) :
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  cylinder_area / cone_area = 2 :=
by
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  sorry

end lateral_surface_area_ratio_l1148_114895


namespace sarah_toy_cars_l1148_114817

theorem sarah_toy_cars (initial_money toy_car_cost scarf_cost beanie_cost remaining_money: ℕ) 
  (h_initial: initial_money = 53) 
  (h_toy_car_cost: toy_car_cost = 11) 
  (h_scarf_cost: scarf_cost = 10) 
  (h_beanie_cost: beanie_cost = 14) 
  (h_remaining: remaining_money = 7) : 
  (initial_money - remaining_money - scarf_cost - beanie_cost) / toy_car_cost = 2 := 
by 
  sorry

end sarah_toy_cars_l1148_114817


namespace certain_number_l1148_114803

theorem certain_number (x y : ℕ) (h₁ : x = 14) (h₂ : 2^x - 2^(x - 2) = 3 * 2^y) : y = 12 :=
  by
  sorry

end certain_number_l1148_114803


namespace part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l1148_114814

-- Definitions for the sequences
def first_row (n : ℕ) : ℤ := (-3) ^ n
def second_row (n : ℕ) : ℤ := (-3) ^ n - 3
def third_row (n : ℕ) : ℤ := -((-3) ^ n) - 1

-- Statement for part 1
theorem part1_fifth_numbers:
  first_row 5 = -243 ∧ second_row 5 = -246 ∧ third_row 5 = 242 := sorry

-- Statement for part 2
theorem part2_three_adjacent_sum :
  ∃ n : ℕ, first_row (n-1) + first_row n + first_row (n+1) = -1701 ∧
           first_row (n-1) = -243 ∧ first_row n = 729 ∧ first_row (n+1) = -2187 := sorry

-- Statement for part 3
def sum_nth (n : ℕ) : ℤ := first_row n + second_row n + third_row n
theorem part3_difference_largest_smallest (n : ℕ) (m : ℤ) (hn : sum_nth n = m) :
  (∃ diff, (n % 2 = 1 → diff = -2 * m - 6) ∧ (n % 2 = 0 → diff = 2 * m + 9)) := sorry

end part1_fifth_numbers_part2_three_adjacent_sum_part3_difference_largest_smallest_l1148_114814


namespace find_son_age_l1148_114834

variable {S F : ℕ}

theorem find_son_age (h1 : F = S + 35) (h2 : F + 2 = 2 * (S + 2)) : S = 33 :=
sorry

end find_son_age_l1148_114834


namespace find_fraction_eq_l1148_114827

theorem find_fraction_eq 
  {x : ℚ} 
  (h : x / (2 / 3) = (3 / 5) / (6 / 7)) : 
  x = 7 / 15 :=
by
  sorry

end find_fraction_eq_l1148_114827
