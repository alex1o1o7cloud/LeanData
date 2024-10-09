import Mathlib

namespace citric_acid_molecular_weight_l557_55710

noncomputable def molecularWeightOfCitricAcid : ℝ :=
  let weight_C := 12.01
  let weight_H := 1.008
  let weight_O := 16.00
  let num_C := 6
  let num_H := 8
  let num_O := 7
  (num_C * weight_C) + (num_H * weight_H) + (num_O * weight_O)

theorem citric_acid_molecular_weight :
  molecularWeightOfCitricAcid = 192.124 :=
by
  -- the step-by-step proof will go here
  sorry

end citric_acid_molecular_weight_l557_55710


namespace simplify_sqrt_power_l557_55791

theorem simplify_sqrt_power : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end simplify_sqrt_power_l557_55791


namespace largest_4_digit_congruent_to_7_mod_19_l557_55789

theorem largest_4_digit_congruent_to_7_mod_19 : 
  ∃ x, (x % 19 = 7) ∧ 1000 ≤ x ∧ x < 10000 ∧ x = 9982 :=
by
  sorry

end largest_4_digit_congruent_to_7_mod_19_l557_55789


namespace triangle_side_length_difference_l557_55736

theorem triangle_side_length_difference :
  (∃ x : ℤ, 3 ≤ x ∧ x ≤ 17 ∧ ∀ a b c : ℤ, x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) →
  (17 - 3 = 14) :=
by
  intros
  sorry

end triangle_side_length_difference_l557_55736


namespace valid_five_letter_words_l557_55771

def num_valid_words : Nat :=
  let total_words := 3^5
  let invalid_3_consec := 5 * 2^3 * 1^2
  let invalid_4_consec := 2 * 2^4 * 1
  let invalid_5_consec := 2^5
  total_words - (invalid_3_consec + invalid_4_consec + invalid_5_consec)

theorem valid_five_letter_words : num_valid_words = 139 := by
  sorry

end valid_five_letter_words_l557_55771


namespace factor_expression_l557_55799

theorem factor_expression (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) :=
sorry

end factor_expression_l557_55799


namespace shares_total_amount_l557_55785

theorem shares_total_amount (Nina_portion : ℕ) (m n o : ℕ) (m_ratio n_ratio o_ratio : ℕ)
  (h_ratio : m_ratio = 2 ∧ n_ratio = 3 ∧ o_ratio = 9)
  (h_Nina : Nina_portion = 60)
  (hk := Nina_portion / n_ratio)
  (h_shares : m = m_ratio * hk ∧ n = n_ratio * hk ∧ o = o_ratio * hk) :
  m + n + o = 280 :=
by 
  sorry

end shares_total_amount_l557_55785


namespace intersection_M_N_l557_55774

def M : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l557_55774


namespace quadratic_inequality_solution_l557_55764

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : 0 > a) 
(h2 : ∀ x : ℝ, (1 < x ∧ x < 2) ↔ (0 < ax^2 + bx + c)) : 
(∀ x : ℝ, (x < 1/2 ∨ 1 < x) ↔ (0 < 2*a*x^2 - 3*a*x + a)) :=
sorry

end quadratic_inequality_solution_l557_55764


namespace probability_of_drawing_red_ball_l557_55770

theorem probability_of_drawing_red_ball :
  let red_balls := 7
  let black_balls := 3
  let total_balls := red_balls + black_balls
  let probability_red := (red_balls : ℚ) / total_balls
  probability_red = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l557_55770


namespace solve_x_l557_55707

-- Define the function f with the given properties
axiom f : ℝ → ℝ → ℝ
axiom f_assoc : ∀ (a b c : ℝ), f a (f b c) = f (f a b) c
axiom f_inv : ∀ (a : ℝ), f a a = 1

-- Define x and the equation to be solved
theorem solve_x : ∃ (x : ℝ), f x 36 = 216 :=
  sorry

end solve_x_l557_55707


namespace calculate_expression_l557_55708

variable (a : ℝ)

theorem calculate_expression (h : a ≠ 0) : (6 * a^2) / (a / 2) = 12 * a := by
  sorry

end calculate_expression_l557_55708


namespace valentines_left_l557_55790

def initial_valentines : ℕ := 60
def valentines_given_away : ℕ := 16
def valentines_received : ℕ := 5

theorem valentines_left : (initial_valentines - valentines_given_away + valentines_received) = 49 :=
by sorry

end valentines_left_l557_55790


namespace verify_value_of_2a10_minus_a12_l557_55752

-- Define the arithmetic sequence and the sum condition
variable {a : ℕ → ℝ}  -- arithmetic sequence
variable {a1 : ℝ}     -- the first term of the sequence
variable {d : ℝ}      -- the common difference of the sequence

-- Assume that the sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + n * d

-- Assume the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The goal is to prove that 2 * a 10 - a 12 = 24
theorem verify_value_of_2a10_minus_a12 (h_arith : arithmetic_sequence a a1 d) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 :=
  sorry

end verify_value_of_2a10_minus_a12_l557_55752


namespace kopecks_payment_l557_55788

theorem kopecks_payment (n : ℕ) (h : n ≥ 8) : ∃ (a b : ℕ), n = 3 * a + 5 * b :=
sorry

end kopecks_payment_l557_55788


namespace clotheslines_per_house_l557_55748

/-- There are a total of 11 children and 20 adults.
Each child has 4 items of clothing on the clotheslines.
Each adult has 3 items of clothing on the clotheslines.
Each clothesline can hold 2 items of clothing.
All of the clotheslines are full.
There are 26 houses on the street.
Show that the number of clotheslines per house is 2. -/
theorem clotheslines_per_house :
  (11 * 4 + 20 * 3) / 2 / 26 = 2 :=
by
  sorry

end clotheslines_per_house_l557_55748


namespace train_passing_time_l557_55721

noncomputable def train_length : ℝ := 180
noncomputable def train_speed_km_hr : ℝ := 36
noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * (1000 / 3600)

theorem train_passing_time : train_length / train_speed_m_s = 18 := by
  sorry

end train_passing_time_l557_55721


namespace solve_f_neg_a_l557_55781

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem solve_f_neg_a (h : f a = 8) : f (-a) = -6 := by
  sorry

end solve_f_neg_a_l557_55781


namespace division_and_subtraction_l557_55766

theorem division_and_subtraction :
  (12 : ℚ) / (1 / 6) - (1 / 3) = 215 / 3 :=
by
  sorry

end division_and_subtraction_l557_55766


namespace miss_adamson_num_classes_l557_55726

theorem miss_adamson_num_classes
  (students_per_class : ℕ)
  (sheets_per_student : ℕ)
  (total_sheets : ℕ)
  (h1 : students_per_class = 20)
  (h2 : sheets_per_student = 5)
  (h3 : total_sheets = 400) :
  let sheets_per_class := sheets_per_student * students_per_class
  let num_classes := total_sheets / sheets_per_class
  num_classes = 4 :=
by
  sorry

end miss_adamson_num_classes_l557_55726


namespace min_correct_answers_l557_55712

theorem min_correct_answers (x : ℕ) : 
  (∃ x, 0 ≤ x ∧ x ≤ 20 ∧ 5 * x - (20 - x) ≥ 88) :=
sorry

end min_correct_answers_l557_55712


namespace parabolas_vertex_condition_l557_55798

theorem parabolas_vertex_condition (p q x₁ x₂ y₁ y₂ : ℝ) (h1: y₂ = p * (x₂ - x₁)^2 + y₁) (h2: y₁ = q * (x₁ - x₂)^2 + y₂) (h3: x₁ ≠ x₂) : p + q = 0 :=
sorry

end parabolas_vertex_condition_l557_55798


namespace find_c_d_l557_55725

theorem find_c_d (C D : ℤ) (h1 : 3 * C - 4 * D = 18) (h2 : C = 2 * D - 5) :
  C = 28 ∧ D = 33 / 2 := by
sorry

end find_c_d_l557_55725


namespace squared_difference_of_roots_l557_55757

theorem squared_difference_of_roots:
  ∀ (Φ φ : ℝ), (∀ x : ℝ, x^2 = 2*x + 1 ↔ (x = Φ ∨ x = φ)) ∧ Φ ≠ φ → (Φ - φ)^2 = 8 :=
by
  intros Φ φ h
  sorry

end squared_difference_of_roots_l557_55757


namespace sufficient_but_not_necessary_condition_l557_55702

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  ((x + 1) * (x - 3) < 0 → x > -1) ∧ ¬ (x > -1 → (x + 1) * (x - 3) < 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l557_55702


namespace sequence_term_is_100th_term_l557_55778

theorem sequence_term_is_100th_term (a : ℕ → ℝ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, a (n + 1) = 2 * a n / (a n + 2)) :
  (∃ n : ℕ, a n = 2 / 101) ∧ ((∃ n : ℕ, a n = 2 / 101) → n = 100) :=
by
  sorry

end sequence_term_is_100th_term_l557_55778


namespace tile_equations_correct_l557_55787

theorem tile_equations_correct (x y : ℕ) (h1 : 24 * x + 12 * y = 2220) (h2 : y = 2 * x - 15) : 
    (24 * x + 12 * y = 2220) ∧ (y = 2 * x - 15) :=
by
  exact ⟨h1, h2⟩

end tile_equations_correct_l557_55787


namespace cost_of_rusted_side_l557_55703

-- Define the conditions
def perimeter (s : ℕ) (l : ℕ) : ℕ :=
  2 * s + 2 * l

def long_side (s : ℕ) : ℕ :=
  3 * s

def cost_per_foot : ℕ :=
  5

-- Given these conditions, we prove the cost of replacing one short side.
theorem cost_of_rusted_side (s l : ℕ) (h1 : perimeter s l = 640) (h2 : l = long_side s) : 
  5 * s = 400 :=
by 
  sorry

end cost_of_rusted_side_l557_55703


namespace find_some_number_l557_55772

theorem find_some_number (a : ℕ) (h₁ : a = 105) (h₂ : a^3 = some_number * 25 * 45 * 49) : some_number = 3 :=
by
  -- definitions and axioms are assumed true from the conditions
  sorry

end find_some_number_l557_55772


namespace cylindrical_log_distance_l557_55783

def cylinder_radius := 3
def R₁ := 104
def R₂ := 64
def R₃ := 84
def straight_segment := 100

theorem cylindrical_log_distance :
  let adjusted_radius₁ := R₁ - cylinder_radius
  let adjusted_radius₂ := R₂ + cylinder_radius
  let adjusted_radius₃ := R₃ - cylinder_radius
  let arc_distance₁ := π * adjusted_radius₁
  let arc_distance₂ := π * adjusted_radius₂
  let arc_distance₃ := π * adjusted_radius₃
  let total_distance := arc_distance₁ + arc_distance₂ + arc_distance₃ + straight_segment
  total_distance = 249 * π + 100 :=
sorry

end cylindrical_log_distance_l557_55783


namespace find_x_l557_55739

def angle_sum_condition (x : ℝ) := 6 * x + 3 * x + x + x + 4 * x = 360

theorem find_x (x : ℝ) (h : angle_sum_condition x) : x = 24 := 
by {
  sorry
}

end find_x_l557_55739


namespace three_dice_probability_even_l557_55751

/-- A die is represented by numbers from 1 to 6. -/
def die := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

/-- Define an event where three dice are thrown, and we check if their sum is even. -/
def three_dice_sum_even (d1 d2 d3 : die) : Prop :=
  (d1.val + d2.val + d3.val) % 2 = 0

/-- Define the probability that a single die shows an odd number. -/
def prob_odd := 1 / 2

/-- Define the probability that a single die shows an even number. -/
def prob_even := 1 / 2

/-- Define the total probability for the sum of three dice to be even. -/
def prob_sum_even : ℚ :=
  prob_even ^ 3 + (3 * prob_odd ^ 2 * prob_even)

theorem three_dice_probability_even :
  prob_sum_even = 1 / 2 :=
by
  sorry

end three_dice_probability_even_l557_55751


namespace convert_to_rectangular_form_l557_55742

noncomputable def rectangular_form (z : ℂ) : ℂ :=
  let e := Complex.exp (13 * Real.pi * Complex.I / 6)
  3 * e

theorem convert_to_rectangular_form :
  rectangular_form (3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 * (Complex.cos (Real.pi / 6)) + 3 * Complex.I * (Complex.sin (Real.pi / 6))) :=
by
  sorry

end convert_to_rectangular_form_l557_55742


namespace intersection_complement_eq_l557_55700

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 })
variable (B : Set ℝ := { x | x > -1 })

theorem intersection_complement_eq :
  A ∩ (U \ B) = { x | -2 ≤ x ∧ x ≤ -1 } :=
by {
  sorry
}

end intersection_complement_eq_l557_55700


namespace range_of_m_l557_55794

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x-1)^2
  else if x > 0 then -(x+1)^2
  else 0

theorem range_of_m (m : ℝ) (h : f (m^2 + 2*m) + f m > 0) : -3 < m ∧ m < 0 := 
by {
  sorry
}

end range_of_m_l557_55794


namespace james_can_lift_546_pounds_l557_55719

def initial_lift_20m : ℝ := 300
def increase_10m : ℝ := 0.30
def strap_increase : ℝ := 0.20
def additional_weight_20m : ℝ := 50
def final_lift_10m_with_straps : ℝ := 546

theorem james_can_lift_546_pounds :
  let initial_lift_10m := initial_lift_20m * (1 + increase_10m)
  let updated_lift_20m := initial_lift_20m + additional_weight_20m
  let ratio := initial_lift_10m / initial_lift_20m
  let updated_lift_10m := updated_lift_20m * ratio
  let lift_with_straps := updated_lift_10m * (1 + strap_increase)
  lift_with_straps = final_lift_10m_with_straps :=
by
  sorry

end james_can_lift_546_pounds_l557_55719


namespace probability_green_or_yellow_l557_55722

def total_marbles (green yellow red blue : Nat) : Nat :=
  green + yellow + red + blue

def marble_probability (green yellow red blue : Nat) : Rat :=
  (green + yellow) / (total_marbles green yellow red blue)

theorem probability_green_or_yellow :
  let green := 4
  let yellow := 3
  let red := 4
  let blue := 2
  marble_probability green yellow red blue = 7 / 13 := by
  sorry

end probability_green_or_yellow_l557_55722


namespace average_remaining_five_l557_55705

theorem average_remaining_five (S S4 S5 : ℕ) 
  (h1 : S = 18 * 9) 
  (h2 : S4 = 8 * 4) 
  (h3 : S5 = S - S4) 
  (h4 : S5 / 5 = 26) : 
  average_of_remaining_5 = 26 :=
by 
  sorry


end average_remaining_five_l557_55705


namespace max_omega_for_increasing_l557_55737

noncomputable def sin_function (ω : ℕ) (x : ℝ) := Real.sin (ω * x + Real.pi / 6)

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem max_omega_for_increasing : ∀ (ω : ℕ), (0 < ω) →
  is_monotonically_increasing_on (sin_function ω) (Real.pi / 6) (Real.pi / 4) ↔ ω ≤ 9 :=
sorry

end max_omega_for_increasing_l557_55737


namespace inequality_solution_l557_55750

theorem inequality_solution (x : ℝ) (h : 3 * x + 4 ≠ 0) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -(4 / 3) ∨ x > -(5 / 3)) := 
by
  sorry

end inequality_solution_l557_55750


namespace f_neg_two_l557_55753

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variables (f_odd : is_odd_function f)
variables (f_two : f 2 = 2)

theorem f_neg_two : f (-2) = -2 :=
by
  -- Given that f is an odd function and f(2) = 2
  sorry

end f_neg_two_l557_55753


namespace find_d_value_l557_55731

open Nat

variable {PA BC PB : ℕ}
noncomputable def d (PA BC PB : ℕ) := PB

theorem find_d_value (h₁ : PA = 6) (h₂ : BC = 9) (h₃ : PB = d PA BC PB) : d PA BC PB = 3 := by
  sorry

end find_d_value_l557_55731


namespace constant_term_q_l557_55735

theorem constant_term_q (p q r : Polynomial ℝ) 
  (hp_const : p.coeff 0 = 6) 
  (hr_const : (p * q).coeff 0 = -18) : q.coeff 0 = -3 :=
sorry

end constant_term_q_l557_55735


namespace sum_reciprocals_squares_l557_55716

theorem sum_reciprocals_squares {a b : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a * b = 11) :
  (1 / (a: ℚ)^2) + (1 / (b: ℚ)^2) = 122 / 121 := 
sorry

end sum_reciprocals_squares_l557_55716


namespace hua_luogeng_optimal_selection_method_l557_55779

theorem hua_luogeng_optimal_selection_method :
  (method_used_in_optimal_selection_method = "Golden ratio") :=
sorry

end hua_luogeng_optimal_selection_method_l557_55779


namespace min_employees_needed_l557_55786

theorem min_employees_needed
  (W A S : Finset ℕ)
  (hW : W.card = 120)
  (hA : A.card = 150)
  (hS : S.card = 100)
  (hWA : (W ∩ A).card = 50)
  (hAS : (A ∩ S).card = 30)
  (hWS : (W ∩ S).card = 20)
  (hWAS : (W ∩ A ∩ S).card = 10) :
  (W ∪ A ∪ S).card = 280 :=
by
  sorry

end min_employees_needed_l557_55786


namespace Nellie_legos_l557_55732

def initial_legos : ℕ := 380
def lost_legos : ℕ := 57
def given_legos : ℕ := 24

def remaining_legos : ℕ := initial_legos - lost_legos - given_legos

theorem Nellie_legos : remaining_legos = 299 := by
  sorry

end Nellie_legos_l557_55732


namespace parallel_lines_a_eq_3_div_2_l557_55784

theorem parallel_lines_a_eq_3_div_2 (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → a = 3 / 2 :=
by sorry

end parallel_lines_a_eq_3_div_2_l557_55784


namespace profit_per_meter_l557_55775

theorem profit_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (total_cost_price : ℕ := cost_price_per_meter * total_meters)
  (total_profit : ℕ := selling_price - total_cost_price)
  (profit_per_meter : ℕ := total_profit / total_meters) :
  total_meters = 75 ∧ selling_price = 4950 ∧ cost_price_per_meter = 51 → profit_per_meter = 15 :=
by
  intros h
  sorry

end profit_per_meter_l557_55775


namespace square_area_from_wire_bent_as_circle_l557_55765

theorem square_area_from_wire_bent_as_circle 
  (radius : ℝ) 
  (h_radius : radius = 56)
  (π_ineq : π > 3.1415) : 
  ∃ (A : ℝ), A = 784 * π^2 := 
by 
  sorry

end square_area_from_wire_bent_as_circle_l557_55765


namespace problem_part1_problem_part2_area_height_l557_55756

theorem problem_part1 (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) : 
  x * y ^ 2 - x ^ 2 * y = -32 * Real.sqrt 2 := 
  sorry

theorem problem_part2_area_height (x y : ℝ) (h : abs (x - 4 - 2 * Real.sqrt 2) + Real.sqrt (y - 4 + 2 * Real.sqrt 2) = 0) :
  let side_length := Real.sqrt 12
  let area := (1 / 2) * x * y
  let height := area / side_length
  area = 4 ∧ height = (2 * Real.sqrt 3) / 3 := 
  sorry

end problem_part1_problem_part2_area_height_l557_55756


namespace complement_intersection_l557_55782

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  ((U \ A) ∩ B) = {0} :=
by
  sorry

end complement_intersection_l557_55782


namespace high_speed_train_equation_l557_55728

theorem high_speed_train_equation (x : ℝ) (h1 : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 :=
by
  sorry

end high_speed_train_equation_l557_55728


namespace sam_initial_money_l557_55780

theorem sam_initial_money (num_books cost_per_book money_left initial_money : ℤ) 
  (h1 : num_books = 9) 
  (h2 : cost_per_book = 7) 
  (h3 : money_left = 16)
  (h4 : initial_money = num_books * cost_per_book + money_left) :
  initial_money = 79 := 
by
  -- Proof is not required, hence we use sorry to complete the statement.
  sorry

end sam_initial_money_l557_55780


namespace seq_bounded_l557_55755

def digit_product (n : ℕ) : ℕ :=
  n.digits 10 |>.prod

def a_seq (a : ℕ → ℕ) (m : ℕ) : Prop :=
  a 0 = m ∧ (∀ n, a (n + 1) = a n + digit_product (a n))

theorem seq_bounded (m : ℕ) : ∃ B, ∀ n, a_seq a m → a n < B :=
by sorry

end seq_bounded_l557_55755


namespace mr_smith_payment_l557_55776

theorem mr_smith_payment {balance : ℝ} {percentage : ℝ} 
  (h_bal : balance = 150) (h_percent : percentage = 0.02) :
  (balance + balance * percentage) = 153 :=
by
  sorry

end mr_smith_payment_l557_55776


namespace mans_speed_upstream_l557_55795

-- Define the conditions
def V_downstream : ℝ := 15  -- Speed with the current (downstream)
def V_current : ℝ := 2.5    -- Speed of the current

-- Calculate the man's speed against the current (upstream)
theorem mans_speed_upstream : V_downstream - 2 * V_current = 10 :=
by
  sorry

end mans_speed_upstream_l557_55795


namespace point_translation_l557_55773

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation_l557_55773


namespace ratio_traditionalists_progressives_l557_55727

-- Define the given conditions
variables (T P C : ℝ)
variables (h1 : C = P + 4 * T)
variables (h2 : 4 * T = 0.75 * C)

-- State the theorem
theorem ratio_traditionalists_progressives (h1 : C = P + 4 * T) (h2 : 4 * T = 0.75 * C) : T / P = 3 / 4 :=
by {
  sorry
}

end ratio_traditionalists_progressives_l557_55727


namespace john_savings_percentage_l557_55723

theorem john_savings_percentage :
  ∀ (savings discounted_price total_price original_price : ℝ),
  savings = 4.5 →
  total_price = 49.5 →
  total_price = discounted_price * 1.10 →
  original_price = discounted_price + savings →
  (savings / original_price) * 100 = 9 := by
  intros
  sorry

end john_savings_percentage_l557_55723


namespace solve_quadratic1_solve_quadratic2_l557_55792

theorem solve_quadratic1 (x : ℝ) :
  x^2 - 4 * x - 7 = 0 →
  (x = 2 - Real.sqrt 11) ∨ (x = 2 + Real.sqrt 11) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  (x - 3)^2 + 2 * (x - 3) = 0 →
  (x = 3) ∨ (x = 1) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l557_55792


namespace no_pos_reals_floor_prime_l557_55717

open Real
open Nat

theorem no_pos_reals_floor_prime : 
  ∀ (a b : ℝ), (0 < a) → (0 < b) → ∃ n : ℕ, ¬ Prime (⌊a * n + b⌋) :=
by
  intro a b a_pos b_pos
  sorry

end no_pos_reals_floor_prime_l557_55717


namespace incorrect_contrapositive_l557_55768

theorem incorrect_contrapositive (x : ℝ) : (x ≠ 1 → ¬ (x^2 - 1 = 0)) ↔ ¬ (x^2 - 1 = 0 → x^2 = 1) := by
  sorry

end incorrect_contrapositive_l557_55768


namespace hanna_gives_roses_l557_55759

-- Conditions as Lean definitions
def initial_budget : ℕ := 300
def price_jenna : ℕ := 2
def price_imma : ℕ := 3
def price_ravi : ℕ := 4
def price_leila : ℕ := 5

def roses_for_jenna (budget : ℕ) : ℕ :=
  budget / price_jenna * 1 / 3

def roses_for_imma (budget : ℕ) : ℕ :=
  budget / price_imma * 1 / 4

def roses_for_ravi (budget : ℕ) : ℕ :=
  budget / price_ravi * 1 / 6

def roses_for_leila (budget : ℕ) : ℕ :=
  budget / price_leila

-- Calculations based on conditions
def roses_jenna : ℕ := Nat.floor (50 * 1/3)
def roses_imma : ℕ := Nat.floor ((100 / price_imma) * 1 / 4)
def roses_ravi : ℕ := Nat.floor ((50 / price_ravi) * 1 / 6)
def roses_leila : ℕ := 50 / price_leila

-- Final statement to be proven
theorem hanna_gives_roses :
  roses_jenna + roses_imma + roses_ravi + roses_leila = 36 := by
  sorry

end hanna_gives_roses_l557_55759


namespace linear_substitution_correct_l557_55797

theorem linear_substitution_correct (x y : ℝ) 
  (h1 : y = x - 1) 
  (h2 : x + 2 * y = 7) : 
  x + 2 * x - 2 = 7 := 
by
  sorry

end linear_substitution_correct_l557_55797


namespace total_pizza_order_cost_l557_55763

def pizza_cost_per_pizza := 10
def topping_cost_per_topping := 1
def tip_amount := 5
def number_of_pizzas := 3
def number_of_toppings := 4

theorem total_pizza_order_cost : 
  (pizza_cost_per_pizza * number_of_pizzas + topping_cost_per_topping * number_of_toppings + tip_amount) = 39 := by
  sorry

end total_pizza_order_cost_l557_55763


namespace find_min_value_x_l557_55760

theorem find_min_value_x (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 10) : 
  ∃ (x_min : ℝ), (∀ (x' : ℝ), (∀ y' z', x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10 → x' ≥ x_min)) ∧ x_min = 2 / 3 :=
sorry

end find_min_value_x_l557_55760


namespace tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l557_55743

noncomputable def f (a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

noncomputable def f_prime (a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_eq_a1 (b : ℝ) (h : f_prime 1 b (-1) = 0) : 
  ∃ m q, m = 1 ∧ q = 1 ∧ ∀ y, y = f 1 b 0 + m * y := sorry

theorem max_value_f_a_gt_1_div_5 (a b : ℝ) 
  (h_gt : a > 1/5) 
  (h_fp_eq : f_prime a b (-1) = 0)
  (h_max : ∀ x, -1 ≤ x ∧ x ≤ 1 → f a b x ≤ 4 * Real.exp 1) : 
  a = (24 * Real.exp 2 - 9) / 15 ∧ b = (12 * Real.exp 2 - 2) / 5 := sorry

end tangent_line_eq_a1_max_value_f_a_gt_1_div_5_l557_55743


namespace cheenu_speed_difference_l557_55762

theorem cheenu_speed_difference :
  let cycling_time := 120 -- minutes
  let cycling_distance := 24 -- miles
  let jogging_time := 180 -- minutes
  let jogging_distance := 18 -- miles
  let cycling_speed := cycling_time / cycling_distance -- minutes per mile
  let jogging_speed := jogging_time / jogging_distance -- minutes per mile
  let speed_difference := jogging_speed - cycling_speed -- minutes per mile
  speed_difference = 5 := by sorry

end cheenu_speed_difference_l557_55762


namespace triangle_inequality_inequality_l557_55741

theorem triangle_inequality_inequality {a b c p q r : ℝ}
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b)
  (h4 : p + q + r = 0) :
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 :=
sorry

end triangle_inequality_inequality_l557_55741


namespace largest_n_unique_k_l557_55701

theorem largest_n_unique_k :
  ∃ (n : ℕ), ( ∃! (k : ℕ), (5 : ℚ) / 11 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 6 / 11 )
    ∧ n = 359 :=
sorry

end largest_n_unique_k_l557_55701


namespace monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l557_55714

noncomputable def f (x : ℝ) := Real.exp x - (1 / 2) * x^2 - x - 1
noncomputable def f' (x : ℝ) := Real.exp x - x - 1
noncomputable def f'' (x : ℝ) := Real.exp x - 1
noncomputable def g (x : ℝ) := -f (-x)

-- Proof of (I)
theorem monotonic_intervals_and_extreme_values_of_f' :
  f' 0 = 0 ∧ (∀ x < 0, f'' x < 0 ∧ f' x > f' 0) ∧ (∀ x > 0, f'' x > 0 ∧ f' x > f' 0) := 
sorry

-- Proof of (II)
theorem f_g_inequality (x : ℝ) (hx : x > 0) : f x > g x :=
sorry

-- Proof of (III)
theorem sum_of_x1_x2 (x1 x2 : ℝ) (h : f x1 + f x2 = 0) (hne : x1 ≠ x2) : x1 + x2 < 0 := 
sorry

end monotonic_intervals_and_extreme_values_of_f_f_g_inequality_sum_of_x1_x2_l557_55714


namespace area_of_polygon_l557_55761

-- Define the conditions
variables (n : ℕ) (s : ℝ)
-- Given that polygon has 32 sides.
def sides := 32
-- Each side is congruent, and the total perimeter is 64 units.
def perimeter := 64
-- Side length of each side
def side_length := perimeter / sides

-- Area of the polygon we need to prove
def target_area := 96

theorem area_of_polygon : side_length * side_length * sides = target_area := 
by {
  -- Here proof would come in reality, we'll skip it by sorry for now.
  sorry
}

end area_of_polygon_l557_55761


namespace inequality_proof_l557_55754

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b + c) / (2 * a) + (c + a) / (2 * b) + (a + b) / (2 * c) ≥ (2 * a) / (b + c) + (2 * b) / (c + a) + (2 * c) / (a + b) :=
by
  sorry

end inequality_proof_l557_55754


namespace cistern_emptying_time_l557_55777

theorem cistern_emptying_time (R L : ℝ) (hR : R = 1 / 6) (hL : L = 1 / 6 - 1 / 8) :
    1 / L = 24 := by
  -- The proof is omitted
  sorry

end cistern_emptying_time_l557_55777


namespace mia_receives_chocolate_l557_55715

-- Given conditions
def total_chocolate : ℚ := 72 / 7
def piles : ℕ := 6
def piles_to_Mia : ℕ := 2

-- Weight of one pile
def weight_of_one_pile (total_chocolate : ℚ) (piles : ℕ) := total_chocolate / piles

-- Total weight Mia receives
def mia_chocolate (weight_of_one_pile : ℚ) (piles_to_Mia : ℕ) := piles_to_Mia * weight_of_one_pile

theorem mia_receives_chocolate : mia_chocolate (weight_of_one_pile total_chocolate piles) piles_to_Mia = 24 / 7 :=
by
  sorry

end mia_receives_chocolate_l557_55715


namespace compute_ab_l557_55740

theorem compute_ab (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 867.75 := 
by
  sorry

end compute_ab_l557_55740


namespace contrapositive_true_l557_55749

theorem contrapositive_true (q p : Prop) (h : q → p) : ¬p → ¬q :=
by sorry

end contrapositive_true_l557_55749


namespace rect_area_correct_l557_55730

-- Defining the function to calculate the area of a rectangle given the coordinates of its vertices
noncomputable def rect_area (x1 y1 x2 y2 x3 y3 x4 y4 : ℤ) : ℤ :=
  let length := abs (x2 - x1)
  let width := abs (y1 - y3)
  length * width

-- The vertices of the rectangle
def x1 : ℤ := -8
def y1 : ℤ := 1
def x2 : ℤ := 1
def y2 : ℤ := 1
def x3 : ℤ := 1
def y3 : ℤ := -7
def x4 : ℤ := -8
def y4 : ℤ := -7

-- Proving that the area of the rectangle is 72 square units
theorem rect_area_correct : rect_area x1 y1 x2 y2 x3 y3 x4 y4 = 72 := by
  sorry

end rect_area_correct_l557_55730


namespace cost_price_of_article_l557_55744

noncomputable def cost_price (M : ℝ) : ℝ := 98.68 / 1.25

theorem cost_price_of_article (M : ℝ)
    (h1 : 0.95 * M = 98.68)
    (h2 : 98.68 = 1.25 * cost_price M) :
    cost_price M = 78.944 :=
by sorry

end cost_price_of_article_l557_55744


namespace athletes_meet_second_time_at_l557_55747

-- Define the conditions given in the problem
def distance_AB : ℕ := 110

def man_uphill_speed : ℕ := 3
def man_downhill_speed : ℕ := 5

def woman_uphill_speed : ℕ := 2
def woman_downhill_speed : ℕ := 3

-- Define the times for the athletes' round trips
def man_round_trip_time : ℚ := (distance_AB / man_uphill_speed) + (distance_AB / man_downhill_speed)
def woman_round_trip_time : ℚ := (distance_AB / woman_uphill_speed) + (distance_AB / woman_downhill_speed)

-- Lean statement for the proof
theorem athletes_meet_second_time_at :
  ∀ (t : ℚ), t = lcm (man_round_trip_time) (woman_round_trip_time) →
  ∃ d : ℚ, d = 330 / 7 := 
by sorry

end athletes_meet_second_time_at_l557_55747


namespace inequality_proof_l557_55711

noncomputable def a : ℝ := 1 + Real.tan (-0.2)
noncomputable def b : ℝ := Real.log (0.8 * Real.exp 1)
noncomputable def c : ℝ := 1 / Real.exp 0.2

theorem inequality_proof : c > a ∧ a > b := by
  sorry

end inequality_proof_l557_55711


namespace collinear_probability_correct_l557_55713

def number_of_dots := 25

def number_of_four_dot_combinations := Nat.choose number_of_dots 4

-- Calculate the different possibilities for collinear sets:
def horizontal_sets := 5 * 5
def vertical_sets := 5 * 5
def diagonal_sets := 2 + 2

def total_collinear_sets := horizontal_sets + vertical_sets + diagonal_sets

noncomputable def probability_collinear : ℚ :=
  total_collinear_sets / number_of_four_dot_combinations

theorem collinear_probability_correct :
  probability_collinear = 6 / 1415 :=
sorry

end collinear_probability_correct_l557_55713


namespace negation_prob1_negation_prob2_negation_prob3_l557_55720

-- Definitions and Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def defines_const_func (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f x1 = f x2

-- Problem 1
theorem negation_prob1 : 
  (∃ n : ℕ, ∀ p : ℕ, is_prime p → p ≤ n) ↔ 
  ¬(∀ n : ℕ, ∃ p : ℕ, is_prime p ∧ n ≤ p) :=
sorry

-- Problem 2
theorem negation_prob2 : 
  (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) ↔ 
  ¬(∀ n : ℤ, ∃! p : ℤ, n + p = 0) :=
sorry

-- Problem 3
theorem negation_prob3 : 
  (∀ y : ℝ, ¬defines_const_func (λ x => x * y) y) ↔ 
  ¬(∃ y : ℝ, defines_const_func (λ x => x * y) y) :=
sorry

end negation_prob1_negation_prob2_negation_prob3_l557_55720


namespace rachel_reading_homework_l557_55724

theorem rachel_reading_homework (math_hw : ℕ) (additional_reading_hw : ℕ) (total_reading_hw : ℕ) 
  (h1 : math_hw = 8) (h2 : additional_reading_hw = 6) (h3 : total_reading_hw = math_hw + additional_reading_hw) :
  total_reading_hw = 14 :=
sorry

end rachel_reading_homework_l557_55724


namespace sanghyeon_questions_l557_55758

variable (S : ℕ)

theorem sanghyeon_questions (h1 : S + (S + 5) = 43) : S = 19 :=
by
    sorry

end sanghyeon_questions_l557_55758


namespace number_of_basketball_cards_l557_55733

theorem number_of_basketball_cards 
  (B : ℕ) -- Number of basketball cards in each box
  (H1 : 4 * B = 40) -- Given condition from equation 4B = 40
  
  (H2 : 4 * B + 40 - 58 = 22) -- Given condition from the total number of cards

: B = 10 := 
by 
  sorry

end number_of_basketball_cards_l557_55733


namespace find_ab_l557_55767

theorem find_ab (a b : ℝ) 
  (period_cond : (π / b) = (2 * π / 5)) 
  (point_cond : a * Real.tan (5 * (π / 10) / 2) = 1) :
  a * b = 5 / 2 :=
sorry

end find_ab_l557_55767


namespace taxi_division_number_of_ways_to_divide_six_people_l557_55769

theorem taxi_division (people : Finset ℕ) (h : people.card = 6) (taxi1 taxi2 : Finset ℕ) 
  (h1 : taxi1.card ≤ 4) (h2 : taxi2.card ≤ 4) (h_union : people = taxi1 ∪ taxi2) (h_disjoint : Disjoint taxi1 taxi2) :
  (taxi1.card = 3 ∧ taxi2.card = 3) ∨ 
  (taxi1.card = 4 ∧ taxi2.card = 2) :=
sorry

theorem number_of_ways_to_divide_six_people : 
  ∃ n : ℕ, n = 50 :=
sorry

end taxi_division_number_of_ways_to_divide_six_people_l557_55769


namespace P_plus_Q_eq_14_l557_55738

variable (P Q : Nat)

-- Conditions:
axiom single_digit_P : P < 10
axiom single_digit_Q : Q < 10
axiom three_P_ends_7 : 3 * P % 10 = 7
axiom two_Q_ends_0 : 2 * Q % 10 = 0

theorem P_plus_Q_eq_14 : P + Q = 14 :=
by
  sorry

end P_plus_Q_eq_14_l557_55738


namespace cubic_three_real_roots_l557_55709

theorem cubic_three_real_roots (a : ℝ) :
  (∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃) ∧
   x₁ ^ 3 - 3 * x₁ - a = 0 ∧
   x₂ ^ 3 - 3 * x₂ - a = 0 ∧
   x₃ ^ 3 - 3 * x₃ - a = 0) ↔ -2 < a ∧ a < 2 :=
by
  sorry

end cubic_three_real_roots_l557_55709


namespace ideal_sleep_hours_l557_55746

open Nat

theorem ideal_sleep_hours 
  (weeknight_sleep : Nat)
  (weekend_sleep : Nat)
  (sleep_deficit : Nat)
  (num_weeknights : Nat := 5)
  (num_weekend_nights : Nat := 2)
  (total_nights : Nat := 7) :
  weeknight_sleep = 5 →
  weekend_sleep = 6 →
  sleep_deficit = 19 →
  ((num_weeknights * weeknight_sleep + num_weekend_nights * weekend_sleep) + sleep_deficit) / total_nights = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end ideal_sleep_hours_l557_55746


namespace interest_rate_part2_l557_55793

noncomputable def total_investment : ℝ := 3400
noncomputable def part1 : ℝ := 1300
noncomputable def part2 : ℝ := total_investment - part1
noncomputable def rate1 : ℝ := 0.03
noncomputable def total_interest : ℝ := 144
noncomputable def interest1 : ℝ := part1 * rate1
noncomputable def interest2 : ℝ := total_interest - interest1
noncomputable def rate2 : ℝ := interest2 / part2

theorem interest_rate_part2 : rate2 = 0.05 := sorry

end interest_rate_part2_l557_55793


namespace max_sum_of_arithmetic_sequence_l557_55704

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (S_seq : ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)) 
  (S16_pos : S 16 > 0) (S17_neg : S 17 < 0) : 
  ∃ m, ∀ n, S n ≤ S m ∧ m = 8 := 
sorry

end max_sum_of_arithmetic_sequence_l557_55704


namespace oz_words_lost_l557_55718

theorem oz_words_lost (letters : Fin 64) (forbidden_letter : Fin 64) (h_forbidden : forbidden_letter.val = 6) : 
  let one_letter_words := 64 
  let two_letter_words := 64 * 64
  let one_letter_lost := if letters = forbidden_letter then 1 else 0
  let two_letter_lost := (if letters = forbidden_letter then 64 else 0) + (if letters = forbidden_letter then 64 else 0) 
  1 + two_letter_lost = 129 :=
by
  sorry

end oz_words_lost_l557_55718


namespace eval_ceil_sqrt_sum_l557_55734

theorem eval_ceil_sqrt_sum :
  (⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉) = 27 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 33 ∧ Real.sqrt 33 < 6 := by sorry
  have h3 : 18 < Real.sqrt 333 ∧ Real.sqrt 333 < 19 := by sorry
  sorry

end eval_ceil_sqrt_sum_l557_55734


namespace moneyEarnedDuringHarvest_l557_55745

-- Define the weekly earnings, duration of harvest, and weekly rent.
def weeklyEarnings : ℕ := 403
def durationOfHarvest : ℕ := 233
def weeklyRent : ℕ := 49

-- Define total earnings and total rent.
def totalEarnings : ℕ := weeklyEarnings * durationOfHarvest
def totalRent : ℕ := weeklyRent * durationOfHarvest

-- Calculate the money earned after rent.
def moneyEarnedAfterRent : ℕ := totalEarnings - totalRent

-- The theorem to prove.
theorem moneyEarnedDuringHarvest : moneyEarnedAfterRent = 82482 :=
  by
  sorry

end moneyEarnedDuringHarvest_l557_55745


namespace number_system_base_l557_55706

theorem number_system_base (a : ℕ) (h : 2 * a^2 + 5 * a + 3 = 136) : a = 7 := 
sorry

end number_system_base_l557_55706


namespace initial_number_proof_l557_55729

-- Definitions for the given problem
def to_add : ℝ := 342.00000000007276
def multiple_of_412 (n : ℤ) : ℝ := 412 * n

-- The initial number
def initial_number : ℝ := 412 - to_add

-- The proof problem statement
theorem initial_number_proof (n : ℤ) (h : multiple_of_412 n = initial_number + to_add) : 
  ∃ x : ℝ, initial_number = x := 
sorry

end initial_number_proof_l557_55729


namespace sum_of_solutions_l557_55796

theorem sum_of_solutions (S : Finset ℝ) (h : ∀ x ∈ S, |x^2 - 10 * x + 29| = 3) : S.sum id = 0 :=
sorry

end sum_of_solutions_l557_55796
