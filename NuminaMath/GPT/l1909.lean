import Mathlib

namespace card_dealing_probability_l1909_190904

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

end card_dealing_probability_l1909_190904


namespace car_parking_arrangements_l1909_190907

theorem car_parking_arrangements : 
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  (red_car_positions * arrange_black_cars) = 14400 := 
by
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  sorry

end car_parking_arrangements_l1909_190907


namespace pos_int_fraction_iff_l1909_190917

theorem pos_int_fraction_iff (p : ℕ) (hp : p > 0) : (∃ k : ℕ, 4 * p + 11 = k * (2 * p - 7)) ↔ (p = 4 ∨ p = 5) := 
sorry

end pos_int_fraction_iff_l1909_190917


namespace angle_sum_l1909_190967

theorem angle_sum {A B D F G : Type} 
  (angle_A : ℝ) 
  (angle_AFG : ℝ) 
  (angle_AGF : ℝ) 
  (angle_BFD : ℝ)
  (H1 : angle_A = 30)
  (H2 : angle_AFG = angle_AGF)
  (H3 : angle_BFD = 105)
  (H4 : angle_AFG + angle_BFD = 180) 
  : angle_B + angle_D = 75 := 
by 
  sorry

end angle_sum_l1909_190967


namespace certain_number_is_48_l1909_190910

theorem certain_number_is_48 (x : ℕ) (h : x = 4) : 36 + 3 * x = 48 := by
  sorry

end certain_number_is_48_l1909_190910


namespace math_problem_l1909_190937

open Function

noncomputable def rotate_90_ccw (p : ℝ × ℝ) (c : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (h, k) := c
  (h - (y - k), k + (x - h))

noncomputable def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (y, x)

theorem math_problem (a b : ℝ) :
  reflect_over_y_eq_x (rotate_90_ccw (a, b) (2, 3)) = (4, -5) → b - a = -5 :=
by
  intros h
  sorry

end math_problem_l1909_190937


namespace enemies_left_undefeated_l1909_190911

theorem enemies_left_undefeated (points_per_enemy : ℕ) (total_enemies : ℕ) (total_points_earned : ℕ) 
  (h1: points_per_enemy = 9) (h2: total_enemies = 11) (h3: total_points_earned = 72):
  total_enemies - (total_points_earned / points_per_enemy) = 3 :=
by
  sorry

end enemies_left_undefeated_l1909_190911


namespace marco_paints_8_15_in_32_minutes_l1909_190905

-- Define the rates at which Marco and Carla paint
def marco_rate : ℚ := 1 / 60
def combined_rate : ℚ := 1 / 40

-- Define the function to calculate the fraction of the room painted by Marco alone in a given time
def fraction_painted_by_marco (time: ℚ) : ℚ := time * marco_rate

-- State the theorem to prove
theorem marco_paints_8_15_in_32_minutes :
  (marco_rate + (combined_rate - marco_rate) = combined_rate) →
  fraction_painted_by_marco 32 = 8 / 15 := by
  sorry

end marco_paints_8_15_in_32_minutes_l1909_190905


namespace probability_both_selected_l1909_190990

/- 
Problem statement: Given that the probability of selection of Ram is 5/7 and that of Ravi is 1/5,
prove that the probability that both Ram and Ravi are selected is 1/7.
-/

theorem probability_both_selected (pRam : ℚ) (pRavi : ℚ) (hRam : pRam = 5 / 7) (hRavi : pRavi = 1 / 5) :
  (pRam * pRavi) = 1 / 7 :=
by
  sorry

end probability_both_selected_l1909_190990


namespace normal_level_shortage_l1909_190939

variable (T : ℝ) (normal_capacity : ℝ) (end_of_month_reservoir : ℝ)
variable (h1 : end_of_month_reservoir = 6)
variable (h2 : end_of_month_reservoir = 2 * normal_capacity)
variable (h3 : end_of_month_reservoir = 0.60 * T)

theorem normal_level_shortage :
  normal_capacity = 7 :=
by
  sorry

end normal_level_shortage_l1909_190939


namespace calculation_result_l1909_190950

theorem calculation_result : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by
  sorry

end calculation_result_l1909_190950


namespace larry_correct_evaluation_l1909_190923

theorem larry_correct_evaluation (a b c d e : ℝ) 
(Ha : a = 5) (Hb : b = 3) (Hc : c = 6) (Hd : d = 4) :
a - b + c + d - e = a - (b - (c + (d - e))) → e = 0 :=
by
  -- Not providing the actual proof
  sorry

end larry_correct_evaluation_l1909_190923


namespace complex_combination_l1909_190952

open Complex

def a : ℂ := 2 - I
def b : ℂ := -1 + I

theorem complex_combination : 2 * a + 3 * b = 1 + I :=
by
  -- Proof goes here
  sorry

end complex_combination_l1909_190952


namespace m_range_satisfies_inequality_l1909_190962

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x + sin x

theorem m_range_satisfies_inequality :
  ∀ (m : ℝ), f (2 * m ^ 2 - m + π - 1) ≥ -2 * π ↔ -1 / 2 ≤ m ∧ m ≤ 1 := 
by
  sorry

end m_range_satisfies_inequality_l1909_190962


namespace probability_same_number_l1909_190973

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l1909_190973


namespace one_third_12x_plus_5_l1909_190966

-- Define x as a real number
variable (x : ℝ)

-- Define the hypothesis
def h := 12 * x + 5

-- State the theorem
theorem one_third_12x_plus_5 : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 :=
  by 
    sorry -- Proof is omitted

end one_third_12x_plus_5_l1909_190966


namespace speed_difference_is_36_l1909_190948

open Real

noncomputable def alex_speed : ℝ := 8 / (40 / 60)
noncomputable def jordan_speed : ℝ := 12 / (15 / 60)
noncomputable def speed_difference : ℝ := jordan_speed - alex_speed

theorem speed_difference_is_36 : speed_difference = 36 := by
  have hs1 : alex_speed = 8 / (40 / 60) := rfl
  have hs2 : jordan_speed = 12 / (15 / 60) := rfl
  have hd : speed_difference = jordan_speed - alex_speed := rfl
  rw [hs1, hs2] at hd
  simp [alex_speed, jordan_speed, speed_difference] at hd
  sorry

end speed_difference_is_36_l1909_190948


namespace milk_butterfat_problem_l1909_190925

variable (x : ℝ)

def butterfat_10_percent (x : ℝ) := 0.10 * x
def butterfat_35_percent_in_8_gallons : ℝ := 0.35 * 8
def total_milk (x : ℝ) := x + 8
def total_butterfat (x : ℝ) := 0.20 * (x + 8)

theorem milk_butterfat_problem 
    (h : butterfat_10_percent x + butterfat_35_percent_in_8_gallons = total_butterfat x) : x = 12 :=
by
  sorry

end milk_butterfat_problem_l1909_190925


namespace estimate_diff_and_prod_l1909_190955

variable {x y : ℝ}
variable (hx : x > y) (hy : y > 0)

theorem estimate_diff_and_prod :
  (1.1*x) - (y - 2) = (x - y) + 0.1 * x + 2 ∧ (1.1 * x) * (y - 2) = 1.1 * (x * y) - 2.2 * x :=
by 
  sorry -- Proof details go here

end estimate_diff_and_prod_l1909_190955


namespace geom_seq_42_l1909_190945

variable {α : Type*} [Field α] [CharZero α]

noncomputable def a_n (n : ℕ) (a1 q : α) : α := a1 * q ^ n

theorem geom_seq_42 (a1 q : α) (h1 : a1 = 3) (h2 : a1 * (1 + q^2 + q^4) = 21) :
  a1 * (q^2 + q^4 + q^6) = 42 := 
by
  sorry

end geom_seq_42_l1909_190945


namespace piecewise_function_identity_l1909_190979

theorem piecewise_function_identity (x : ℝ) : 
  (3 * x + abs (5 * x - 10)) = if x < 2 then -2 * x + 10 else 8 * x - 10 := by
  sorry

end piecewise_function_identity_l1909_190979


namespace sector_area_l1909_190957

theorem sector_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : (1/2) * l * r = 3 :=
by
  rw [h_r, h_l]
  norm_num

end sector_area_l1909_190957


namespace seashells_given_l1909_190974

theorem seashells_given (original_seashells : ℕ) (current_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : original_seashells = 35) 
  (h2 : current_seashells = 17) 
  (h3 : given_seashells = original_seashells - current_seashells) : 
  given_seashells = 18 := 
by 
  sorry

end seashells_given_l1909_190974


namespace part1_part2_l1909_190946

namespace Problem

open Real

def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2*x - 8 > 0)

theorem part1 (h : p 1 x ∧ q x) : 2 < x ∧ x < 3:= 
sorry

theorem part2 (hpq : ∀ x, ¬ p a x → ¬ q x) : 
   1 < a ∧ a ≤ 2 := 
sorry

end Problem

end part1_part2_l1909_190946


namespace minimum_value_l1909_190901

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem minimum_value (a m n : ℝ)
    (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1)
    (h_a_on_graph : ∀ x, log_a a (x + 3) - 1 = 0 → x = -2)
    (h_on_line : 2 * m + n = 2)
    (h_mn_pos : m * n > 0) :
    (1 / m) + (2 / n) = 4 :=
by
  sorry

end minimum_value_l1909_190901


namespace min_abs_sum_l1909_190900

theorem min_abs_sum (x : ℝ) : (∃ x : ℝ, ∀ y : ℝ, (|y - 2| + |y - 47| ≥ |x - 2| + |x - 47|)) → (|x - 2| + |x - 47| = 45) :=
by
  sorry

end min_abs_sum_l1909_190900


namespace total_groups_l1909_190986

-- Define the problem conditions
def boys : ℕ := 9
def girls : ℕ := 12

-- Calculate the required combinations
def C (n k: ℕ) : ℕ := n.choose k
def groups_with_two_boys_one_girl : ℕ := C boys 2 * C girls 1
def groups_with_two_girls_one_boy : ℕ := C girls 2 * C boys 1

-- Statement of the theorem to prove
theorem total_groups : groups_with_two_boys_one_girl + groups_with_two_girls_one_boy = 1026 := 
by sorry

end total_groups_l1909_190986


namespace problem_statement_l1909_190924

noncomputable def equation_of_altitude (A B C: (ℝ × ℝ)): (ℝ × ℝ × ℝ) :=
by
  sorry

theorem problem_statement :
  let A := (-1, 4)
  let B := (-2, -1)
  let C := (2, 3)
  equation_of_altitude A B C = (1, 1, -3) ∧
  |1 / 2 * (4 - (-1)) * 4| = 8 :=
by
  sorry

end problem_statement_l1909_190924


namespace scientific_notation_correct_l1909_190942

-- Define the given number
def given_number : ℕ := 138000

-- Define the scientific notation expression
def scientific_notation : ℝ := 1.38 * 10^5

-- The proof goal: Prove that 138,000 expressed in scientific notation is 1.38 * 10^5
theorem scientific_notation_correct : (given_number : ℝ) = scientific_notation := by
  -- Sorry is used to skip the proof
  sorry

end scientific_notation_correct_l1909_190942


namespace box_combination_is_correct_l1909_190995

variables (C A S T t u : ℕ)

theorem box_combination_is_correct
    (h1 : 3 * S % t = C)
    (h2 : 2 * A + C = T)
    (h3 : 2 * C + A + u = T) :
  (1000 * C + 100 * A + 10 * S + T = 7252) :=
sorry

end box_combination_is_correct_l1909_190995


namespace exists_duplicate_in_grid_of_differences_bounded_l1909_190976

theorem exists_duplicate_in_grid_of_differences_bounded :
  ∀ (f : ℕ × ℕ → ℤ), 
  (∀ i j, i < 10 → j < 10 → (i + 1 < 10 → (abs (f (i, j) - f (i + 1, j)) ≤ 5)) 
                             ∧ (j + 1 < 10 → (abs (f (i, j) - f (i, j + 1)) ≤ 5))) → 
  ∃ x y : ℕ × ℕ, x ≠ y ∧ f x = f y :=
by
  intros
  sorry -- Proof goes here

end exists_duplicate_in_grid_of_differences_bounded_l1909_190976


namespace parabola_directrix_l1909_190987

theorem parabola_directrix (x : ℝ) :
  (y = (x^2 - 8 * x + 12) / 16) →
  (∃ y, y = -17/4) :=
by
  intro h
  sorry

end parabola_directrix_l1909_190987


namespace inequality_smallest_integer_solution_l1909_190932

theorem inequality_smallest_integer_solution (x : ℤ) :
    (9 * x + 8) / 6 - x / 3 ≥ -1 → x ≥ -2 := sorry

end inequality_smallest_integer_solution_l1909_190932


namespace plane_determination_l1909_190996

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end plane_determination_l1909_190996


namespace kishore_savings_l1909_190930

noncomputable def rent := 5000
noncomputable def milk := 1500
noncomputable def groceries := 4500
noncomputable def education := 2500
noncomputable def petrol := 2000
noncomputable def miscellaneous := 700
noncomputable def total_expenses := rent + milk + groceries + education + petrol + miscellaneous
noncomputable def salary : ℝ := total_expenses / 0.9 -- given that savings is 10% of salary

theorem kishore_savings : (salary * 0.1) = 1800 :=
by
  sorry

end kishore_savings_l1909_190930


namespace range_of_a_l1909_190981

theorem range_of_a (a : ℝ) (h₀ : a > 0) : (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 :=
sorry

end range_of_a_l1909_190981


namespace expression_eval_l1909_190968

theorem expression_eval : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 :=
by sorry

end expression_eval_l1909_190968


namespace root_quadratic_l1909_190963

theorem root_quadratic (m : ℝ) (h : m^2 - 2*m - 1 = 0) : m^2 + 1/m^2 = 6 :=
sorry

end root_quadratic_l1909_190963


namespace solve_system_solve_equation_l1909_190989

-- 1. System of Equations
theorem solve_system :
  ∀ (x y : ℝ), (x + 2 * y = 9) ∧ (3 * x - 2 * y = 3) → (x = 3) ∧ (y = 3) :=
by sorry

-- 2. Single Equation
theorem solve_equation :
  ∀ (x : ℝ), (2 - x) / (x - 3) + 3 = 2 / (3 - x) → x = 5 / 2 :=
by sorry

end solve_system_solve_equation_l1909_190989


namespace Eric_white_marbles_l1909_190999

theorem Eric_white_marbles (total_marbles blue_marbles green_marbles : ℕ) (h1 : total_marbles = 20) (h2 : blue_marbles = 6) (h3 : green_marbles = 2) : 
  total_marbles - (blue_marbles + green_marbles) = 12 := by
  sorry

end Eric_white_marbles_l1909_190999


namespace b_minus_a_eq_two_l1909_190954

theorem b_minus_a_eq_two (a b : ℤ) (h1 : b = 7) (h2 : a * b = 2 * (a + b) + 11) : b - a = 2 :=
by
  sorry

end b_minus_a_eq_two_l1909_190954


namespace necessary_condition_of_and_is_or_l1909_190928

variable (p q : Prop)

theorem necessary_condition_of_and_is_or (hpq : p ∧ q) : p ∨ q :=
by {
    sorry
}

end necessary_condition_of_and_is_or_l1909_190928


namespace evaluate_expression_l1909_190969

theorem evaluate_expression : (3 : ℚ) / (1 - (2 : ℚ) / 5) = 5 := sorry

end evaluate_expression_l1909_190969


namespace value_of_b_prod_l1909_190920

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 2 ^ (n - 1)

-- The goal is to prove that b_{a_1} * b_{a_3} * b_{a_5} = 4096
theorem value_of_b_prod : b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end value_of_b_prod_l1909_190920


namespace same_terminal_side_angles_l1909_190943

theorem same_terminal_side_angles (k : ℤ) :
  ∃ (k1 k2 : ℤ), k1 * 360 - 1560 = -120 ∧ k2 * 360 - 1560 = 240 :=
by
  -- Conditions and property definitions can be added here if needed
  sorry

end same_terminal_side_angles_l1909_190943


namespace proof_valid_set_exists_l1909_190971

noncomputable def valid_set_exists : Prop :=
∃ (s : Finset ℕ), s.card = 10 ∧ 
(∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → a ≠ b) ∧ 
(∃ (t1 : Finset ℕ), t1 ⊆ s ∧ t1.card = 3 ∧ ∀ n ∈ t1, 5 ∣ n) ∧
(∃ (t2 : Finset ℕ), t2 ⊆ s ∧ t2.card = 4 ∧ ∀ n ∈ t2, 4 ∣ n) ∧
s.sum id < 75

theorem proof_valid_set_exists : valid_set_exists :=
sorry

end proof_valid_set_exists_l1909_190971


namespace cost_of_greenhouses_possible_renovation_plans_l1909_190918

noncomputable def cost_renovation (x y : ℕ) : Prop :=
  (2 * x = y + 6) ∧ (x + 2 * y = 48)

theorem cost_of_greenhouses : ∃ x y, cost_renovation x y ∧ x = 12 ∧ y = 18 :=
by {
  sorry
}

noncomputable def renovation_plan (m : ℕ) : Prop :=
  (5 * m + 3 * (8 - m) ≤ 35) ∧ (12 * m + 18 * (8 - m) ≤ 128)

theorem possible_renovation_plans : ∃ m, renovation_plan m ∧ (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  sorry
}

end cost_of_greenhouses_possible_renovation_plans_l1909_190918


namespace train_speed_first_part_l1909_190982

theorem train_speed_first_part (x v : ℝ) (h1 : 0 < x) (h2 : 0 < v) 
  (h_avg_speed : (3 * x) / (x / v + 2 * x / 20) = 22.5) : v = 30 :=
sorry

end train_speed_first_part_l1909_190982


namespace determine_b_l1909_190953

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iio 2 ∪ Set.Ioi 6 → -x^2 + b * x - 7 < 0) ∧ 
  (∀ x : ℝ, ¬(x ∈ Set.Iio 2 ∪ Set.Ioi 6) → ¬(-x^2 + b * x - 7 < 0)) → 
  b = 8 :=
sorry

end determine_b_l1909_190953


namespace parabola_whose_directrix_is_tangent_to_circle_l1909_190959

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

noncomputable def is_tangent (line_eq : ℝ → ℝ → Prop) (circle_eq : ℝ → ℝ → Prop) : Prop := 
  ∃ p : ℝ × ℝ, (line_eq p.1 p.2) ∧ (circle_eq p.1 p.2) ∧ 
  (∀ q : ℝ × ℝ, (circle_eq q.1 q.2) → (line_eq q.1 q.2) → q = p)

-- Definitions of parabolas
noncomputable def parabola_A_directrix (x y : ℝ) : Prop := y = 2

noncomputable def parabola_B_directrix (x y : ℝ) : Prop := x = 2

noncomputable def parabola_C_directrix (x y : ℝ) : Prop := x = -4

noncomputable def parabola_D_directrix (x y : ℝ) : Prop := y = -1

-- The final statement to prove
theorem parabola_whose_directrix_is_tangent_to_circle :
  is_tangent parabola_D_directrix circle_eq ∧ ¬ is_tangent parabola_A_directrix circle_eq ∧ 
  ¬ is_tangent parabola_B_directrix circle_eq ∧ ¬ is_tangent parabola_C_directrix circle_eq :=
sorry

end parabola_whose_directrix_is_tangent_to_circle_l1909_190959


namespace bushels_needed_l1909_190926

theorem bushels_needed (cows sheep chickens : ℕ) (cows_eat sheep_eat chickens_eat : ℕ) :
  cows = 4 → cows_eat = 2 →
  sheep = 3 → sheep_eat = 2 →
  chickens = 7 → chickens_eat = 3 →
  4 * 2 + 3 * 2 + 7 * 3 = 35 := 
by
  intros hc hec hs hes hch hech
  sorry

end bushels_needed_l1909_190926


namespace find_diagonal_length_l1909_190941

theorem find_diagonal_length (d : ℝ) (offset1 offset2 : ℝ) (area : ℝ)
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 300) :
  (1/2) * d * (offset1 + offset2) = area → d = 40 :=
by
  -- placeholder for proof
  sorry

end find_diagonal_length_l1909_190941


namespace calculate_expression_l1909_190977

theorem calculate_expression : |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by
  sorry

end calculate_expression_l1909_190977


namespace weekly_income_l1909_190988

-- Defining the daily catches
def blue_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 10
  | "Tuesday"   => 8
  | "Wednesday" => 12
  | "Thursday"  => 6
  | "Friday"    => 14
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

def red_crabs_per_bucket (day : String) : ℕ :=
  match day with
  | "Monday"    => 14
  | "Tuesday"   => 16
  | "Wednesday" => 10
  | "Thursday"  => 18
  | "Friday"    => 12
  | "Saturday"  => 10
  | "Sunday"    => 8
  | _           => 0

-- Prices per crab
def price_per_blue_crab : ℕ := 6
def price_per_red_crab : ℕ := 4
def buckets : ℕ := 8

-- Daily income calculation
def daily_income (day : String) : ℕ :=
  let blue_income := (blue_crabs_per_bucket day) * buckets * price_per_blue_crab
  let red_income := (red_crabs_per_bucket day) * buckets * price_per_red_crab
  blue_income + red_income

-- Proving the weekly income is $6080
theorem weekly_income : 
  (daily_income "Monday" +
  daily_income "Tuesday" +
  daily_income "Wednesday" +
  daily_income "Thursday" +
  daily_income "Friday" +
  daily_income "Saturday" +
  daily_income "Sunday") = 6080 :=
by sorry

end weekly_income_l1909_190988


namespace solve_equation_l1909_190998

theorem solve_equation (x : ℝ) :
  3 * x + 6 = abs (-20 + x^2) →
  x = (3 + Real.sqrt 113) / 2 ∨ x = (3 - Real.sqrt 113) / 2 :=
by
  sorry

end solve_equation_l1909_190998


namespace eval_expression_at_values_l1909_190936

theorem eval_expression_at_values : 
  ∀ x y : ℕ, x = 3 ∧ y = 4 → 
  5 * (x^(y+1)) + 6 * (y^(x+1)) + 2 * x * y = 2775 :=
by
  intros x y hxy
  cases hxy
  sorry

end eval_expression_at_values_l1909_190936


namespace problem1_problem2_l1909_190915

-- Proof Problem 1:

theorem problem1 : (5 / 3) ^ 2004 * (3 / 5) ^ 2003 = 5 / 3 := by
  sorry

-- Proof Problem 2:

theorem problem2 (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end problem1_problem2_l1909_190915


namespace mass_percentage_of_N_in_NH4Br_l1909_190947

theorem mass_percentage_of_N_in_NH4Br :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Br := 79.90
  let molar_mass_NH4Br := (1 * molar_mass_N) + (4 * molar_mass_H) + (1 * molar_mass_Br)
  let mass_percentage_N := (molar_mass_N / molar_mass_NH4Br) * 100
  mass_percentage_N = 14.30 :=
by
  sorry

end mass_percentage_of_N_in_NH4Br_l1909_190947


namespace total_miles_ran_l1909_190935

theorem total_miles_ran (miles_monday miles_wednesday miles_friday : ℕ)
  (h1 : miles_monday = 3)
  (h2 : miles_wednesday = 2)
  (h3 : miles_friday = 7) :
  miles_monday + miles_wednesday + miles_friday = 12 := 
by
  sorry

end total_miles_ran_l1909_190935


namespace part_i_part_ii_l1909_190909

variable {b c : ℤ}

theorem part_i (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (p q : ℤ), p > 0 ∧ q > 0 ∧ p ≠ q ∧ 2 * b ^ 2 = p ^ 2 + q ^ 2 :=
sorry

theorem part_ii (hb : b ≠ 0) (hc : c ≠ 0) 
    (cond1 : ∃ m n : ℤ, m ≠ n ∧ (m + n = -b) ∧ (m * n = c))
    (cond2 : ∃ r s : ℤ, r ≠ s ∧ (r + s = -b) ∧ (r * s = -c)) :
    ∃ (r s : ℤ), r > 0 ∧ s > 0 ∧ r ≠ s ∧ b ^ 2 = r ^ 2 + s ^ 2 :=
sorry

end part_i_part_ii_l1909_190909


namespace average_children_l1909_190919

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l1909_190919


namespace find_number_l1909_190927

theorem find_number (x : ℝ) (h : 7 * x + 21.28 = 50.68) : x = 4.2 :=
sorry

end find_number_l1909_190927


namespace expression_value_l1909_190922

theorem expression_value : 7^4 + 4 * 7^3 + 6 * 7^2 + 4 * 7 + 1 = 4096 := 
by 
  -- proof goes here 
  sorry

end expression_value_l1909_190922


namespace linear_equation_solution_l1909_190975

theorem linear_equation_solution (a b : ℤ) (x y : ℤ) (h1 : x = 2) (h2 : y = -1) (h3 : a * x + b * y = -1) : 
  1 + 2 * a - b = 0 :=
by
  sorry

end linear_equation_solution_l1909_190975


namespace grandpa_tomatoes_before_vacation_l1909_190965

theorem grandpa_tomatoes_before_vacation 
  (tomatoes_after_vacation : ℕ) 
  (growth_factor : ℕ) 
  (actual_number : ℕ) 
  (h1 : growth_factor = 100) 
  (h2 : tomatoes_after_vacation = 3564) 
  (h3 : actual_number = tomatoes_after_vacation / growth_factor) : 
  actual_number = 36 := 
by
  -- Here would be the step-by-step proof, but we use sorry to skip it
  sorry

end grandpa_tomatoes_before_vacation_l1909_190965


namespace courtyard_length_l1909_190916

/-- Given the following conditions:
  1. The width of the courtyard is 16.5 meters.
  2. 66 paving stones are required.
  3. Each paving stone measures 2.5 meters by 2 meters.
  Prove that the length of the rectangular courtyard is 20 meters. -/
theorem courtyard_length :
  ∃ L : ℝ, L = 20 ∧ 
           (∃ W : ℝ, W = 16.5) ∧ 
           (∃ n : ℕ, n = 66) ∧ 
           (∃ A : ℝ, A = 2.5 * 2) ∧
           n * A = L * W :=
by
  sorry

end courtyard_length_l1909_190916


namespace product_of_five_consecutive_integers_not_perfect_square_l1909_190961

theorem product_of_five_consecutive_integers_not_perfect_square (n : ℕ) : 
  ¬ ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = k * k :=
by {
  sorry
}

end product_of_five_consecutive_integers_not_perfect_square_l1909_190961


namespace order_b_gt_c_gt_a_l1909_190908

noncomputable def a : ℝ := Real.log 2.6
def b : ℝ := 0.5 * 1.8^2
noncomputable def c : ℝ := 1.1^5

theorem order_b_gt_c_gt_a : b > c ∧ c > a := by
  sorry

end order_b_gt_c_gt_a_l1909_190908


namespace find_number_l1909_190972

theorem find_number (N : ℕ) (h : N / 7 = 12 ∧ N % 7 = 5) : N = 89 := 
by
  sorry

end find_number_l1909_190972


namespace intended_profit_l1909_190994

variables (C P : ℝ)

theorem intended_profit (L S : ℝ) (h1 : L = C * (1 + P)) (h2 : S = 0.90 * L) (h3 : S = 1.17 * C) :
  P = 0.3 + 1 / 3 :=
by
  sorry

end intended_profit_l1909_190994


namespace common_factor_of_polynomial_l1909_190958

variables (x y m n : ℝ)

theorem common_factor_of_polynomial :
  ∃ (k : ℝ), (2 * (m - n)) = k ∧ (4 * x * (m - n) + 2 * y * (m - n)^2) = k * (2 * x * (m - n)) :=
sorry

end common_factor_of_polynomial_l1909_190958


namespace Malou_score_third_quiz_l1909_190938

-- Defining the conditions as Lean definitions
def score1 : ℕ := 91
def score2 : ℕ := 92
def average : ℕ := 91
def num_quizzes : ℕ := 3

-- Proving that score3 equals 90
theorem Malou_score_third_quiz :
  ∃ score3 : ℕ, (score1 + score2 + score3) / num_quizzes = average ∧ score3 = 90 :=
by
  use (90 : ℕ)
  sorry

end Malou_score_third_quiz_l1909_190938


namespace math_proof_problems_l1909_190960

open Real

noncomputable def problem1 (α : ℝ) : Prop :=
  (sin (π - α) - 2 * sin (π / 2 + α) = 0) → (sin α * cos α + sin α ^ 2 = 6 / 5)

noncomputable def problem2 (α β : ℝ) : Prop :=
  (tan (α + β) = -1) → (tan α = 2) → (tan β = 3)

-- Example of how to state these problems as a theorem
theorem math_proof_problems (α β : ℝ) : problem1 α ∧ problem2 α β := by
  sorry

end math_proof_problems_l1909_190960


namespace sum_of_all_possible_values_of_abs_b_l1909_190992

theorem sum_of_all_possible_values_of_abs_b {a b : ℝ}
  {r s : ℝ} (hr : r^3 + a * r + b = 0) (hs : s^3 + a * s + b = 0)
  (hr4 : (r + 4)^3 + a * (r + 4) + b + 240 = 0) (hs3 : (s - 3)^3 + a * (s - 3) + b + 240 = 0) :
  |b| = 20 ∨ |b| = 42 →
  20 + 42 = 62 :=
by
  sorry

end sum_of_all_possible_values_of_abs_b_l1909_190992


namespace bill_spots_l1909_190944

theorem bill_spots (b p : ℕ) (h1 : b + p = 59) (h2 : b = 2 * p - 1) : b = 39 := by
  sorry

end bill_spots_l1909_190944


namespace expand_expression_l1909_190985

theorem expand_expression (x : ℝ) : 24 * (3 * x + 4 - 2) = 72 * x + 48 :=
by 
  sorry

end expand_expression_l1909_190985


namespace Cathy_and_Chris_worked_months_l1909_190951

theorem Cathy_and_Chris_worked_months (Cathy_hours : ℕ) (weekly_hours : ℕ) (weeks_in_month : ℕ) (extra_weekly_hours : ℕ) (weeks_for_Chris_sick : ℕ) : 
  Cathy_hours = 180 →
  weekly_hours = 20 →
  weeks_in_month = 4 →
  extra_weekly_hours = weekly_hours →
  weeks_for_Chris_sick = 1 →
  (Cathy_hours - extra_weekly_hours * weeks_for_Chris_sick) / weekly_hours / weeks_in_month = (2 : ℕ) :=
by
  intros hCathy_hours hweekly_hours hweeks_in_month hextra_weekly_hours hweeks_for_Chris_sick
  rw [hCathy_hours, hweekly_hours, hweeks_in_month, hextra_weekly_hours, hweeks_for_Chris_sick]
  norm_num
  sorry

end Cathy_and_Chris_worked_months_l1909_190951


namespace bert_total_stamp_cost_l1909_190997

theorem bert_total_stamp_cost :
    let numA := 150
    let numB := 90
    let numC := 60
    let priceA := 2
    let priceB := 3
    let priceC := 5
    let costA := numA * priceA
    let costB := numB * priceB
    let costC := numC * priceC
    let total_cost := costA + costB + costC
    total_cost = 870 := 
by
    sorry

end bert_total_stamp_cost_l1909_190997


namespace percentage_proof_l1909_190993

theorem percentage_proof (n : ℝ) (h : 0.3 * 0.4 * n = 24) : 0.4 * 0.3 * n = 24 :=
sorry

end percentage_proof_l1909_190993


namespace simplify_sqrt_sum_l1909_190970

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sqrt_sum_l1909_190970


namespace mrs_sheridan_gave_away_14_cats_l1909_190984

def num_initial_cats : ℝ := 17.0
def num_left_cats : ℝ := 3.0
def num_given_away (x : ℝ) : Prop := num_initial_cats - x = num_left_cats

theorem mrs_sheridan_gave_away_14_cats : num_given_away 14.0 :=
by
  sorry

end mrs_sheridan_gave_away_14_cats_l1909_190984


namespace members_count_l1909_190914

theorem members_count (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end members_count_l1909_190914


namespace find_m_over_n_l1909_190902

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

end find_m_over_n_l1909_190902


namespace equal_rental_costs_l1909_190978

variable {x : ℝ}

def SunshineCarRentalsCost (x : ℝ) : ℝ := 17.99 + 0.18 * x
def CityRentalsCost (x : ℝ) : ℝ := 18.95 + 0.16 * x

theorem equal_rental_costs (x : ℝ) : SunshineCarRentalsCost x = CityRentalsCost x ↔ x = 48 :=
by
  sorry

end equal_rental_costs_l1909_190978


namespace dwarfs_truthful_count_l1909_190912

theorem dwarfs_truthful_count (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) :
    x = 4 ∧ y = 6 :=
by
  sorry

end dwarfs_truthful_count_l1909_190912


namespace chocolate_difference_l1909_190964

theorem chocolate_difference :
  let nick_chocolates := 10
  let alix_chocolates := 3 * nick_chocolates - 5
  alix_chocolates - nick_chocolates = 15 :=
by
  sorry

end chocolate_difference_l1909_190964


namespace ordered_triples_count_l1909_190956

noncomputable def count_valid_triples (n : ℕ) :=
  ∃ x y z : ℕ, ∃ k : ℕ, x * y * z = k ∧ k = 5 ∧ lcm x y = 48 ∧ lcm x z = 450 ∧ lcm y z = 600

theorem ordered_triples_count : count_valid_triples 5 := by
  sorry

end ordered_triples_count_l1909_190956


namespace greatest_possible_median_l1909_190991

theorem greatest_possible_median : 
  ∀ (k m r s t : ℕ),
    k < m → m < r → r < s → s < t →
    (k + m + r + s + t = 90) →
    (t = 40) →
    (r = 23) :=
by
  intros k m r s t h1 h2 h3 h4 h_sum h_t
  sorry

end greatest_possible_median_l1909_190991


namespace most_stable_performance_l1909_190931

-- Given variances for the four people
def S_A_var : ℝ := 0.56
def S_B_var : ℝ := 0.60
def S_C_var : ℝ := 0.50
def S_D_var : ℝ := 0.45

-- We need to prove that the variance for D is the smallest
theorem most_stable_performance :
  S_D_var < S_C_var ∧ S_D_var < S_A_var ∧ S_D_var < S_B_var :=
by
  sorry

end most_stable_performance_l1909_190931


namespace linda_original_savings_l1909_190949

theorem linda_original_savings (S : ℝ) (f : ℝ) (a : ℝ) (t : ℝ) 
  (h1 : f = 7 / 13 * S) (h2 : a = 3 / 13 * S) 
  (h3 : t = S - f - a) (h4 : t = 180) (h5 : a = 360) : 
  S = 1560 :=
by 
  sorry

end linda_original_savings_l1909_190949


namespace production_days_l1909_190903

variable (n : ℕ) (average_past : ℝ := 50) (production_today : ℝ := 115) (new_average : ℝ := 55)

theorem production_days (h1 : average_past * n + production_today = new_average * (n + 1)) : 
    n = 12 := 
by 
  sorry

end production_days_l1909_190903


namespace other_number_of_given_conditions_l1909_190980

theorem other_number_of_given_conditions 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 4620) 
  (h_gcd : Nat.gcd a b = 21) 
  (h_a : a = 210) : 
  b = 462 := 
sorry

end other_number_of_given_conditions_l1909_190980


namespace theta_digit_l1909_190913

theorem theta_digit (Θ : ℕ) (h : Θ ≠ 0) (h1 : 252 / Θ = 10 * 4 + Θ + Θ) : Θ = 5 :=
  sorry

end theta_digit_l1909_190913


namespace solve_equation_solutions_count_l1909_190906

open Real

theorem solve_equation_solutions_count :
  (∃ (x_plural : List ℝ), (∀ x ∈ x_plural, 2 * sqrt 2 * (sin (π * x / 4)) ^ 3 = cos (π / 4 * (1 - x)) ∧ 0 ≤ x ∧ x ≤ 2020) ∧ x_plural.length = 505) :=
sorry

end solve_equation_solutions_count_l1909_190906


namespace snow_globes_in_box_l1909_190934

theorem snow_globes_in_box (S : ℕ) 
  (h1 : ∀ (box_decorations : ℕ), box_decorations = 4 + 1 + S)
  (h2 : ∀ (num_boxes : ℕ), num_boxes = 12)
  (h3 : ∀ (total_decorations : ℕ), total_decorations = 120) :
  S = 5 :=
by
  sorry

end snow_globes_in_box_l1909_190934


namespace vertex_of_parabola_l1909_190983

theorem vertex_of_parabola : ∀ x y : ℝ, y = 2 * (x - 1) ^ 2 + 2 → (1, 2) = (1, 2) :=
by
  sorry

end vertex_of_parabola_l1909_190983


namespace required_run_rate_l1909_190921

/-
In the first 10 overs of a cricket game, the run rate was 3.5. 
What should be the run rate in the remaining 40 overs to reach the target of 320 runs?
-/

def run_rate_in_10_overs : ℝ := 3.5
def overs_played : ℕ := 10
def target_runs : ℕ := 320 
def remaining_overs : ℕ := 40

theorem required_run_rate : 
  (target_runs - (run_rate_in_10_overs * overs_played)) / remaining_overs = 7.125 := by 
sorry

end required_run_rate_l1909_190921


namespace distance_between_A_and_B_l1909_190933

def average_speed : ℝ := 50  -- Speed in miles per hour

def travel_time : ℝ := 15.8  -- Time in hours

noncomputable def total_distance : ℝ := average_speed * travel_time  -- Distance in miles

theorem distance_between_A_and_B :
  total_distance = 790 :=
by
  sorry

end distance_between_A_and_B_l1909_190933


namespace find_n_arithmetic_sequence_l1909_190929

-- Given conditions
def a₁ : ℕ := 20
def aₙ : ℕ := 54
def Sₙ : ℕ := 999

-- Arithmetic sequence sum formula and proof statement of n = 27
theorem find_n_arithmetic_sequence
  (a₁ : ℕ)
  (aₙ : ℕ)
  (Sₙ : ℕ)
  (h₁ : a₁ = 20)
  (h₂ : aₙ = 54)
  (h₃ : Sₙ = 999) : ∃ n : ℕ, n = 27 := 
by
  sorry

end find_n_arithmetic_sequence_l1909_190929


namespace find_value_of_a2004_b2004_l1909_190940

-- Given Definitions and Conditions
def a : ℝ := sorry
def b : ℝ := sorry
def A : Set ℝ := {a, a^2, a * b}
def B : Set ℝ := {1, a, b}

-- The theorem statement
theorem find_value_of_a2004_b2004 (h : A = B) : a ^ 2004 + b ^ 2004 = 1 :=
sorry

end find_value_of_a2004_b2004_l1909_190940
