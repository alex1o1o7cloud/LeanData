import Mathlib

namespace students_no_A_l1574_157410

theorem students_no_A (T AH AM AHAM : ℕ) (h1 : T = 35) (h2 : AH = 10) (h3 : AM = 15) (h4 : AHAM = 5) :
  T - (AH + AM - AHAM) = 15 :=
by
  sorry

end students_no_A_l1574_157410


namespace largest_angle_in_ratio_triangle_l1574_157419

theorem largest_angle_in_ratio_triangle (a b c : ℕ) (h_ratios : 2 * c = 3 * b ∧ 3 * b = 4 * a)
  (h_sum : a + b + c = 180) : max a (max b c) = 80 :=
by
  sorry

end largest_angle_in_ratio_triangle_l1574_157419


namespace total_cost_is_correct_l1574_157465

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l1574_157465


namespace count_integers_with_sum_of_digits_18_l1574_157441

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l1574_157441


namespace oranges_needed_l1574_157456

theorem oranges_needed 
  (total_fruit_needed : ℕ := 12) 
  (apples : ℕ := 3) 
  (bananas : ℕ := 4) : 
  total_fruit_needed - (apples + bananas) = 5 :=
by 
  sorry

end oranges_needed_l1574_157456


namespace induction_base_case_l1574_157459

theorem induction_base_case : (-1 : ℤ) + 3 - 5 + (-1)^2 * 1 = (-1 : ℤ) := sorry

end induction_base_case_l1574_157459


namespace side_length_of_square_l1574_157473

noncomputable def area_of_circle : ℝ := 3848.4510006474966
noncomputable def pi : ℝ := Real.pi

theorem side_length_of_square :
  ∃ s : ℝ, (∃ r : ℝ, area_of_circle = pi * r * r ∧ 2 * r = s) ∧ s = 70 := 
by
  sorry

end side_length_of_square_l1574_157473


namespace average_distance_scientific_notation_l1574_157429

theorem average_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ a * 10 ^ n = 384000000 ∧ a = 3.84 ∧ n = 8 :=
sorry

end average_distance_scientific_notation_l1574_157429


namespace pascal_triangle_45th_number_l1574_157423

theorem pascal_triangle_45th_number (n k : ℕ) (h1 : n = 47) (h2 : k = 44) : 
  Nat.choose (n - 1) k = 1035 :=
by
  sorry

end pascal_triangle_45th_number_l1574_157423


namespace solution_range_l1574_157484

-- Given conditions from the table
variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h₁ : f a b c 1.1 = -0.59
axiom h₂ : f a b c 1.2 = 0.84
axiom h₃ : f a b c 1.3 = 2.29
axiom h₄ : f a b c 1.4 = 3.76

theorem solution_range (a b c : ℝ) : 
  ∃ x : ℝ, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
sorry

end solution_range_l1574_157484


namespace find_m_l1574_157438

theorem find_m (m : ℝ) (h1 : |m - 3| = 4) (h2 : m - 7 ≠ 0) : m = -1 :=
sorry

end find_m_l1574_157438


namespace diameter_of_circumscribed_circle_l1574_157409

noncomputable def right_triangle_circumcircle_diameter (a b : ℕ) : ℕ :=
  let hypotenuse := (a * a + b * b).sqrt
  if hypotenuse = max a b then hypotenuse else 2 * max a b

theorem diameter_of_circumscribed_circle
  (a b : ℕ)
  (h : a = 16 ∨ b = 16)
  (h1 : a = 12 ∨ b = 12) :
  right_triangle_circumcircle_diameter a b = 16 ∨ right_triangle_circumcircle_diameter a b = 20 :=
by
  -- The proof goes here.
  sorry

end diameter_of_circumscribed_circle_l1574_157409


namespace inequality_proof_l1574_157448

theorem inequality_proof
  (p q a b c d e : Real)
  (hpq : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (hq : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e)
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p)) ^ 2 :=
sorry

end inequality_proof_l1574_157448


namespace solve_for_x_and_y_l1574_157485

theorem solve_for_x_and_y : 
  (∃ x y : ℝ, 0.65 * 900 = 0.40 * x ∧ 0.35 * 1200 = 0.25 * y) → 
  ∃ x y : ℝ, x + y = 3142.5 :=
by
  sorry

end solve_for_x_and_y_l1574_157485


namespace num_diamonds_in_G6_l1574_157452

noncomputable def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

noncomputable def total_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ k => triangular_number (k + 1)))

theorem num_diamonds_in_G6 :
  total_diamonds 6 = 141 := by
  -- This will be proven
  sorry

end num_diamonds_in_G6_l1574_157452


namespace square_side_factor_l1574_157444

theorem square_side_factor (k : ℝ) (h : k^2 = 1) : k = 1 :=
sorry

end square_side_factor_l1574_157444


namespace sufficient_but_not_necessary_l1574_157405

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end sufficient_but_not_necessary_l1574_157405


namespace probability_of_popped_white_is_12_over_17_l1574_157414

noncomputable def probability_white_given_popped (white_kernels yellow_kernels : ℚ) (pop_white pop_yellow : ℚ) : ℚ :=
  let p_white_popped := white_kernels * pop_white
  let p_yellow_popped := yellow_kernels * pop_yellow
  let p_popped := p_white_popped + p_yellow_popped
  p_white_popped / p_popped

theorem probability_of_popped_white_is_12_over_17 :
  probability_white_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end probability_of_popped_white_is_12_over_17_l1574_157414


namespace simplify_expression_l1574_157461

variable (x : ℝ)

theorem simplify_expression : 
  (3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3 - 3 * x^3) 
  = (-x^3 - x^2 + 23 * x - 3) :=
by
  sorry

end simplify_expression_l1574_157461


namespace total_apples_picked_l1574_157458

def benny_apples : Nat := 2
def dan_apples : Nat := 9

theorem total_apples_picked : benny_apples + dan_apples = 11 := 
by
  sorry

end total_apples_picked_l1574_157458


namespace find_x_minus_y_l1574_157487

theorem find_x_minus_y (x y : ℝ) (hx : |x| = 4) (hy : |y| = 2) (hxy : x * y < 0) : x - y = 6 ∨ x - y = -6 :=
by sorry

end find_x_minus_y_l1574_157487


namespace least_positive_integer_mod_cond_l1574_157476

theorem least_positive_integer_mod_cond (N : ℕ) :
  (N % 6 = 5) ∧ 
  (N % 7 = 6) ∧ 
  (N % 8 = 7) ∧ 
  (N % 9 = 8) ∧ 
  (N % 10 = 9) ∧ 
  (N % 11 = 10) →
  N = 27719 :=
by
  sorry

end least_positive_integer_mod_cond_l1574_157476


namespace square_perimeter_ratio_l1574_157471

theorem square_perimeter_ratio (a b : ℝ) (h : a^2 / b^2 = 16 / 25) :
  (4 * a) / (4 * b) = 4 / 5 :=
by
  sorry

end square_perimeter_ratio_l1574_157471


namespace measure_of_AED_l1574_157466

-- Importing the necessary modules for handling angles and geometry
variables {A B C D E : Type}
noncomputable def angle (p q r : Type) : ℝ := sorry -- Definition to represent angles in general

-- Given conditions
variables
  (hD_on_AC : D ∈ line_segment A C)
  (hE_on_BC : E ∈ line_segment B C)
  (h_angle_ABD : angle A B D = 30)
  (h_angle_BAE : angle B A E = 60)
  (h_angle_CAE : angle C A E = 20)
  (h_angle_CBD : angle C B D = 30)

-- The goal to prove
theorem measure_of_AED :
  angle A E D = 20 :=
by
  -- Proof details will go here
  sorry

end measure_of_AED_l1574_157466


namespace walter_equal_share_l1574_157435

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end walter_equal_share_l1574_157435


namespace polynomial_root_exists_l1574_157472

theorem polynomial_root_exists
  (P : ℝ → ℝ)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
  (h_eq : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)) :
  ∃ r : ℝ, P r = 0 :=
sorry

end polynomial_root_exists_l1574_157472


namespace solution_set_of_inequality_l1574_157462

theorem solution_set_of_inequality (x : ℝ) :
  (abs x * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) := 
by
  sorry

end solution_set_of_inequality_l1574_157462


namespace chessboard_queen_placements_l1574_157497

theorem chessboard_queen_placements :
  ∃ (n : ℕ), n = 864 ∧
  (∀ (qpos : Finset (Fin 8 × Fin 8)), 
    qpos.card = 3 ∧
    (∀ (q1 q2 q3 : Fin 8 × Fin 8), 
      q1 ∈ qpos ∧ q2 ∈ qpos ∧ q3 ∈ qpos ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 → 
      (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) ∧ 
      (q1.1 = q3.1 ∨ q1.2 = q3.2 ∨ abs (q1.1 - q3.1) = abs (q1.2 - q3.2)) ∧ 
      (q2.1 = q3.1 ∨ q2.2 = q3.2 ∨ abs (q2.1 - q3.1) = abs (q2.2 - q3.2)))) ↔ n = 864
:=
by
  sorry

end chessboard_queen_placements_l1574_157497


namespace distance_P_to_y_axis_l1574_157422

-- Definition: Given point P in Cartesian coordinates
def P : ℝ × ℝ := (-3, -4)

-- Definition: Function to calculate distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ := abs p.1

-- Theorem: The distance from point P to the y-axis is 3
theorem distance_P_to_y_axis : distance_to_y_axis P = 3 :=
by
  sorry

end distance_P_to_y_axis_l1574_157422


namespace journey_total_distance_l1574_157425

theorem journey_total_distance (D : ℝ) 
  (train_fraction : ℝ := 3/5) 
  (bus_fraction : ℝ := 7/20) 
  (walk_distance : ℝ := 6.5) 
  (total_fraction : ℝ := 1) : 
  (1 - (train_fraction + bus_fraction)) * D = walk_distance → D = 130 := 
by
  sorry

end journey_total_distance_l1574_157425


namespace train_length_correct_l1574_157447

open Real

-- Define the conditions
def bridge_length : ℝ := 150
def time_to_cross_bridge : ℝ := 7.5
def time_to_cross_lamp_post : ℝ := 2.5

-- Define the length of the train
def train_length : ℝ := 75

theorem train_length_correct :
  ∃ L : ℝ, (L / time_to_cross_lamp_post = (L + bridge_length) / time_to_cross_bridge) ∧ L = train_length :=
by
  sorry

end train_length_correct_l1574_157447


namespace parallel_lines_slope_l1574_157442

-- Define the equations of the lines in Lean
def line1 (x : ℝ) : ℝ := 7 * x + 3
def line2 (c : ℝ) (x : ℝ) : ℝ := (3 * c) * x + 5

-- State the theorem: if the lines are parallel, then c = 7/3
theorem parallel_lines_slope (c : ℝ) :
  (∀ x : ℝ, (7 * x + 3 = (3 * c) * x + 5)) → c = (7/3) :=
by
  sorry

end parallel_lines_slope_l1574_157442


namespace percentage_decrease_l1574_157440

-- Define the initial conditions
def total_cans : ℕ := 600
def initial_people : ℕ := 40
def new_total_cans : ℕ := 420

-- Use the conditions to define the resulting quantities
def cans_per_person : ℕ := total_cans / initial_people
def new_people : ℕ := new_total_cans / cans_per_person

-- Prove the percentage decrease in the number of people
theorem percentage_decrease :
  let original_people := initial_people
  let new_people := new_people
  let decrease := original_people - new_people
  let percentage_decrease := (decrease * 100) / original_people
  percentage_decrease = 30 :=
by
  sorry

end percentage_decrease_l1574_157440


namespace diophantine_eq_solutions_l1574_157433

theorem diophantine_eq_solutions (p q r k : ℕ) (hp : p > 1) (hq : q > 1) (hr : r > 1) 
  (hp_prime : Prime p) (hq_prime : Prime q) (hr_prime : Prime r) (hk : k > 0) :
  p^2 + q^2 + 49*r^2 = 9*k^2 - 101 ↔ 
  (p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8) :=
by sorry

end diophantine_eq_solutions_l1574_157433


namespace no_solution_inequality_l1574_157407

theorem no_solution_inequality (a : ℝ) : (∀ x : ℝ, ¬(|x - 3| + |x - a| < 1)) ↔ (a ≤ 2 ∨ a ≥ 4) := 
sorry

end no_solution_inequality_l1574_157407


namespace side_length_of_S2_l1574_157428

theorem side_length_of_S2 :
  ∀ (r s : ℕ), 
    (2 * r + s = 2000) → 
    (2 * r + 5 * s = 3030) → 
    s = 258 :=
by
  intros r s h1 h2
  sorry

end side_length_of_S2_l1574_157428


namespace sqrt_fraction_sum_as_common_fraction_l1574_157470

theorem sqrt_fraction_sum_as_common_fraction (a b c d : ℚ) (ha : a = 25) (hb : b = 36) (hc : c = 16) (hd : d = 9) :
  Real.sqrt ((a / b) + (c / d)) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_fraction_sum_as_common_fraction_l1574_157470


namespace find_value_of_expression_l1574_157446

theorem find_value_of_expression (m : ℝ) (h_m : m^2 - 3 * m + 1 = 0) : 2 * m^2 - 6 * m - 2024 = -2026 := by
  sorry

end find_value_of_expression_l1574_157446


namespace javier_time_outlining_l1574_157479

variable (O : ℕ)
variable (W : ℕ := O + 28)
variable (P : ℕ := (O + 28) / 2)
variable (total_time : ℕ := O + W + P)

theorem javier_time_outlining
  (h1 : total_time = 117)
  (h2 : W = O + 28)
  (h3 : P = (O + 28) / 2)
  : O = 30 := by 
  sorry

end javier_time_outlining_l1574_157479


namespace deposit_amount_l1574_157491

theorem deposit_amount (P : ℝ) (h₀ : 0.1 * P + 720 = P) : 0.1 * P = 80 :=
by
  sorry

end deposit_amount_l1574_157491


namespace isosceles_triangle_circumradius_l1574_157402

theorem isosceles_triangle_circumradius (b : ℝ) (s : ℝ) (R : ℝ) (hb : b = 6) (hs : s = 5) :
  R = 25 / 8 :=
by 
  sorry

end isosceles_triangle_circumradius_l1574_157402


namespace waiter_slices_l1574_157499

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end waiter_slices_l1574_157499


namespace intersecting_lines_sum_constant_l1574_157493

theorem intersecting_lines_sum_constant
  (c d : ℝ)
  (h1 : 3 = (1 / 3) * 3 + c)
  (h2 : 3 = (1 / 3) * 3 + d) :
  c + d = 4 :=
by
  sorry

end intersecting_lines_sum_constant_l1574_157493


namespace investment_ratio_l1574_157415

-- Definitions of all the conditions
variables (A B C profit b_share: ℝ)

-- Conditions based on the provided problem
def condition1 (n : ℝ) : Prop := A = n * B
def condition2 : Prop := B = (2 / 3) * C
def condition3 : Prop := profit = 4400
def condition4 : Prop := b_share = 800

-- The theorem we want to prove
theorem investment_ratio (n : ℝ) :
  (condition1 A B n) ∧ (condition2 B C) ∧ (condition3 profit) ∧ (condition4 b_share) → A / B = 3 :=
by
  sorry

end investment_ratio_l1574_157415


namespace martha_total_points_l1574_157431

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end martha_total_points_l1574_157431


namespace b_investment_less_c_l1574_157477

theorem b_investment_less_c (A B C : ℕ) (y : ℕ) (total_investment : ℕ) (profit : ℕ) (A_share : ℕ)
    (h1 : A + B + C = total_investment)
    (h2 : A = B + 6000)
    (h3 : C = B + y)
    (h4 : profit = 8640)
    (h5 : A_share = 3168) :
    y = 3000 :=
by
  sorry

end b_investment_less_c_l1574_157477


namespace printer_time_equation_l1574_157424

theorem printer_time_equation (x : ℝ) (rate1 rate2 : ℝ) (flyers1 flyers2 : ℝ)
  (h1 : rate1 = 100) (h2 : flyers1 = 1000) (h3 : flyers2 = 1000) 
  (h4 : flyers1 / rate1 = 10) (h5 : flyers1 / (rate1 + rate2) = 4) : 
  1 / 10 + 1 / x = 1 / 4 :=
by 
  sorry

end printer_time_equation_l1574_157424


namespace root_exists_l1574_157451

variable {R : Type} [LinearOrderedField R]
variables (a b c : R)

def f (x : R) : R := a * x^2 + b * x + c

theorem root_exists (h : f a b c ((a - b - c) / (2 * a)) = 0) : f a b c (-1) = 0 ∨ f a b c 1 = 0 := by
  sorry

end root_exists_l1574_157451


namespace number_of_true_propositions_is_one_l1574_157449

-- Define propositions
def prop1 (a b c : ℝ) : Prop := a > b ∧ c ≠ 0 → a * c > b * c
def prop2 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop3 (a b c : ℝ) : Prop := a * c^2 > b * c^2 → a > b
def prop4 (a b : ℝ) : Prop := a > b → (1 / a) < (1 / b)
def prop5 (a b c d : ℝ) : Prop := a > b ∧ b > 0 ∧ c > d → a * c > b * d

-- The main theorem stating the number of true propositions
theorem number_of_true_propositions_is_one (a b c d : ℝ) :
  (prop3 a b c) ∧ (¬ prop1 a b c) ∧ (¬ prop2 a b c) ∧ (¬ prop4 a b) ∧ (¬ prop5 a b c d) :=
by
  sorry

end number_of_true_propositions_is_one_l1574_157449


namespace find_sachins_age_l1574_157483

variable (S R : ℕ)

theorem find_sachins_age (h1 : R = S + 8) (h2 : S * 9 = R * 7) : S = 28 := by
  sorry

end find_sachins_age_l1574_157483


namespace range_of_f_find_a_l1574_157498

-- Define the function f
def f (a x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- Define the proposition for part (1)
theorem range_of_f (a : ℝ) (h : a > 1) : Set.range (f a) = Set.Iio 1 := sorry

-- Define the proposition for part (2)
theorem find_a (a : ℝ) (h : a > 1) (min_value : ∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → f a x ≥ -7) : a = 2 :=
sorry

end range_of_f_find_a_l1574_157498


namespace triangle_area_l1574_157480

noncomputable def a := 5
noncomputable def b := 4
noncomputable def s := (13 : ℝ) / 2 -- semi-perimeter
noncomputable def area := Real.sqrt (s * (s - a) * (s - b) * (s - b))

theorem triangle_area :
  a + 2 * b = 13 →
  (a > 0) → (b > 0) →
  (a < 2 * b) →
  (a + b > b) → 
  (a + b > b) →
  area = Real.sqrt 61.09375 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- We assume validity of these conditions and skip the proof for brevity.
  sorry

end triangle_area_l1574_157480


namespace odd_square_minus_one_multiple_of_eight_l1574_157460

theorem odd_square_minus_one_multiple_of_eight (a : ℤ) 
  (h₁ : a > 0) 
  (h₂ : a % 2 = 1) : 
  ∃ k : ℤ, a^2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_multiple_of_eight_l1574_157460


namespace problem_l1574_157417

open Set

-- Definitions for set A and set B
def setA : Set ℝ := { x | x^2 + 2 * x - 3 < 0 }
def setB : Set ℤ := { k : ℤ | true }
def evenIntegers : Set ℝ := { x : ℝ | ∃ k : ℤ, x = 2 * k }

-- The intersection of set A and even integers over ℝ
def A_cap_B : Set ℝ := setA ∩ evenIntegers

-- The Proposition that A_cap_B equals {-2, 0}
theorem problem : A_cap_B = ({-2, 0} : Set ℝ) :=
by 
  sorry

end problem_l1574_157417


namespace find_f_neg2_l1574_157418

noncomputable def f : ℝ → ℝ := sorry

axiom f_add (a b : ℝ) : f (a + b) = f a * f b
axiom f_pos (x : ℝ) : f x > 0
axiom f_one : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 :=
by {
  sorry
}

end find_f_neg2_l1574_157418


namespace johns_average_speed_l1574_157486

theorem johns_average_speed :
  let distance1 := 20
  let speed1 := 10
  let distance2 := 30
  let speed2 := 20
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 14.29 :=
by
  sorry

end johns_average_speed_l1574_157486


namespace train_speed_l1574_157420

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h_train_length : train_length = 400) 
  (h_bridge_length : bridge_length = 300) 
  (h_crossing_time : crossing_time = 45) : 
  (train_length + bridge_length) / crossing_time = 700 / 45 := 
  by
    rw [h_train_length, h_bridge_length, h_crossing_time]
    sorry

end train_speed_l1574_157420


namespace simplify_fraction_l1574_157482

theorem simplify_fraction : (2 / (1 - (2 / 3))) = 6 :=
by
  sorry

end simplify_fraction_l1574_157482


namespace M_subset_N_l1574_157488

-- Define M and N using the given conditions
def M : Set ℝ := {α | ∃ (k : ℤ), α = k * 90} ∪ {α | ∃ (k : ℤ), α = k * 180 + 45}
def N : Set ℝ := {α | ∃ (k : ℤ), α = k * 45}

-- Prove that M is a subset of N
theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l1574_157488


namespace jessica_and_sibling_age_l1574_157400

theorem jessica_and_sibling_age
  (J M S : ℕ)
  (h1 : J = M / 2)
  (h2 : M + 10 = 70)
  (h3 : S = J + ((70 - M) / 2)) :
  J = 40 ∧ S = 45 :=
by
  sorry

end jessica_and_sibling_age_l1574_157400


namespace marbles_count_l1574_157463

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l1574_157463


namespace noemi_lost_on_roulette_l1574_157490

theorem noemi_lost_on_roulette (initial_purse := 1700) (final_purse := 800) (loss_on_blackjack := 500) :
  (initial_purse - final_purse) - loss_on_blackjack = 400 := by
  sorry

end noemi_lost_on_roulette_l1574_157490


namespace letters_with_line_not_dot_l1574_157427

-- Defining the conditions
def num_letters_with_dot_and_line : ℕ := 9
def num_letters_with_dot_only : ℕ := 7
def total_letters : ℕ := 40

-- Proving the number of letters with a straight line but not a dot
theorem letters_with_line_not_dot :
  (num_letters_with_dot_and_line + num_letters_with_dot_only + x = total_letters) → x = 24 :=
by
  intros h
  sorry

end letters_with_line_not_dot_l1574_157427


namespace ratio_albert_betty_l1574_157453

theorem ratio_albert_betty (A M B : ℕ) (h1 : A = 2 * M) (h2 : M = A - 10) (h3 : B = 5) :
  A / B = 4 :=
by
  -- the proof goes here
  sorry

end ratio_albert_betty_l1574_157453


namespace sample_size_correct_l1574_157481

-- Define the conditions as lean variables
def total_employees := 120
def male_employees := 90
def female_sample := 9

-- Define the proof problem statement
theorem sample_size_correct : ∃ n : ℕ, (total_employees - male_employees) / total_employees = female_sample / n ∧ n = 36 := by 
  sorry

end sample_size_correct_l1574_157481


namespace geometric_series_has_value_a_l1574_157445

theorem geometric_series_has_value_a (a : ℝ) (S : ℕ → ℝ)
  (h : ∀ n, S (n + 1) = a * (1 / 4) ^ n + 6) :
  a = -3 / 2 :=
sorry

end geometric_series_has_value_a_l1574_157445


namespace ratio_M_N_l1574_157467

-- Definitions of M, Q and N based on the given conditions
variables (M Q P N : ℝ)
variable (h1 : M = 0.40 * Q)
variable (h2 : Q = 0.30 * P)
variable (h3 : N = 0.50 * P)

theorem ratio_M_N : M / N = 6 / 25 :=
by
  -- Proof steps would go here
  sorry

end ratio_M_N_l1574_157467


namespace math_problem_l1574_157492

theorem math_problem (x t : ℝ) (h1 : 6 * x + t = 4 * x - 9) (h2 : t = 7) : x + 4 = -4 := by
  sorry

end math_problem_l1574_157492


namespace inequality_solution_set_l1574_157475

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_A : f 0 = -2)
  (h_B : f 3 = 2) :
  {x : ℝ | |f (x+1)| ≥ 2} = {x | x ≤ -1} ∪ {x | x ≥ 2} :=
sorry

end inequality_solution_set_l1574_157475


namespace total_number_of_rats_l1574_157412

theorem total_number_of_rats (Kenia Hunter Elodie Teagan : ℕ) 
  (h1 : Elodie = 30)
  (h2 : Elodie = Hunter + 10)
  (h3 : Kenia = 3 * (Hunter + Elodie))
  (h4 : Teagan = 2 * Elodie)
  (h5 : Teagan = Kenia - 5) : 
  Kenia + Hunter + Elodie + Teagan = 260 :=
by 
  sorry

end total_number_of_rats_l1574_157412


namespace animal_costs_l1574_157401

theorem animal_costs (S K L : ℕ) (h1 : K = 4 * S) (h2 : L = 4 * K) (h3 : S + 2 * K + L = 200) :
  S = 8 ∧ K = 32 ∧ L = 128 :=
by
  sorry

end animal_costs_l1574_157401


namespace pow_three_not_sum_of_two_squares_l1574_157404

theorem pow_three_not_sum_of_two_squares (k : ℕ) (hk : 0 < k) : 
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 3^k :=
by
  sorry

end pow_three_not_sum_of_two_squares_l1574_157404


namespace hotel_charge_comparison_l1574_157437

def charge_R (R G : ℝ) (P : ℝ) : Prop :=
  P = 0.8 * R ∧ P = 0.9 * G

def discounted_charge_R (R2 : ℝ) (R : ℝ) : Prop :=
  R2 = 0.85 * R

theorem hotel_charge_comparison (R G P R2 : ℝ)
  (h1 : charge_R R G P)
  (h2 : discounted_charge_R R2 R)
  (h3 : R = 1.125 * G) :
  R2 = 0.95625 * G := by
  sorry

end hotel_charge_comparison_l1574_157437


namespace value_of_M_l1574_157478

theorem value_of_M (x y z M : ℚ) : 
  (x + y + z = 48) ∧ (x - 5 = M) ∧ (y + 9 = M) ∧ (z / 5 = M) → M = 52 / 7 :=
by
  sorry

end value_of_M_l1574_157478


namespace complex_sum_to_zero_l1574_157474

noncomputable def z : ℂ := sorry

theorem complex_sum_to_zero 
  (h₁ : z ^ 3 = 1) 
  (h₂ : z ≠ 1) : 
  z ^ 103 + z ^ 104 + z ^ 105 + z ^ 106 + z ^ 107 + z ^ 108 = 0 :=
sorry

end complex_sum_to_zero_l1574_157474


namespace num_intersection_points_l1574_157469

-- Define the equations of the lines as conditions
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- The theorem to prove the number of intersection points
theorem num_intersection_points :
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨ (line2 p.1 p.2 ∧ line3 p.1 p.2) :=
sorry

end num_intersection_points_l1574_157469


namespace sum_of_ages_l1574_157430

def Tyler_age : ℕ := 5

def Clay_age (T C : ℕ) : Prop :=
  T = 3 * C + 1

theorem sum_of_ages (C : ℕ) (h : Clay_age Tyler_age C) :
  Tyler_age + C = 6 :=
sorry

end sum_of_ages_l1574_157430


namespace stepa_multiplied_numbers_l1574_157457

theorem stepa_multiplied_numbers (x : ℤ) (hx : (81 * x) % 16 = 0) :
  ∃ (a b : ℕ), a * b = 54 ∧ a < 10 ∧ b < 10 :=
by {
  sorry
}

end stepa_multiplied_numbers_l1574_157457


namespace find_cos_alpha_l1574_157403

theorem find_cos_alpha (α : ℝ) (h0 : 0 ≤ α ∧ α ≤ π / 2) (h1 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end find_cos_alpha_l1574_157403


namespace parabola_distance_x_coord_l1574_157408

theorem parabola_distance_x_coord
  (M : ℝ × ℝ) 
  (hM : M.2^2 = 4 * M.1)
  (hMF : (M.1 - 1)^2 + M.2^2 = 4^2)
  : M.1 = 3 :=
sorry

end parabola_distance_x_coord_l1574_157408


namespace sum_of_cubes_l1574_157495

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end sum_of_cubes_l1574_157495


namespace nathan_blankets_l1574_157413

theorem nathan_blankets (b : ℕ) (hb : 21 = (b / 2) * 3) : b = 14 :=
by sorry

end nathan_blankets_l1574_157413


namespace heracles_age_l1574_157411

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l1574_157411


namespace find_a_l1574_157426

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + x

-- Define the derivative of the function f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x + 1

-- The main theorem: if the tangent at x = 1 is parallel to the line y = 2x, then a = 1
theorem find_a (a : ℝ) : f' 1 a = 2 → a = 1 :=
by
  intro h
  -- The proof is skipped
  sorry

end find_a_l1574_157426


namespace triangle_ABC_properties_l1574_157494

noncomputable def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
γ - β = β - α

theorem triangle_ABC_properties
  (A B C a c : ℝ)
  (b : ℝ := Real.sqrt 3)
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) :
  is_arithmetic_sequence A B C ∧
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 := by sorry

end triangle_ABC_properties_l1574_157494


namespace man_work_m_alone_in_15_days_l1574_157489

theorem man_work_m_alone_in_15_days (M : ℕ) (h1 : 1/M + 1/10 = 1/6) : M = 15 := sorry

end man_work_m_alone_in_15_days_l1574_157489


namespace total_wages_l1574_157455

-- Definitions and conditions
def A_one_day_work : ℚ := 1 / 10
def B_one_day_work : ℚ := 1 / 15
def A_share_wages : ℚ := 2040

-- Stating the problem in Lean
theorem total_wages (X : ℚ) : (3 / 5) * X = A_share_wages → X = 3400 := 
  by 
  sorry

end total_wages_l1574_157455


namespace total_cost_is_eight_times_l1574_157464

theorem total_cost_is_eight_times (x : ℝ) 
  (h1 : ∀ t, x + t = 2 * x)
  (h2 : ∀ b, x + b = 5 * x)
  (h3 : ∀ s, x + s = 3 * x) :
  ∃ t b s, x + t + b + s = 8 * x :=
by
  sorry

end total_cost_is_eight_times_l1574_157464


namespace min_ratio_number_l1574_157439

theorem min_ratio_number (H T U : ℕ) (h1 : H - T = 8 ∨ T - H = 8) (hH : 1 ≤ H ∧ H ≤ 9) (hT : 0 ≤ T ∧ T ≤ 9) (hU : 0 ≤ U ∧ U ≤ 9) :
  100 * H + 10 * T + U = 190 :=
by sorry

end min_ratio_number_l1574_157439


namespace range_of_a_l1574_157496

-- Definitions from conditions 
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- The Lean statement for the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → x ≤ a) → a ≥ 1 :=
by sorry

end range_of_a_l1574_157496


namespace common_remainder_zero_l1574_157454

theorem common_remainder_zero (n r : ℕ) (h1: n > 1) 
(h2 : n % 25 = r) (h3 : n % 7 = r) (h4 : n = 175) : r = 0 :=
by
  sorry

end common_remainder_zero_l1574_157454


namespace option_d_not_necessarily_true_l1574_157421

theorem option_d_not_necessarily_true (a b c : ℝ) (h: a > b) : ¬(a * c^2 > b * c^2) ↔ c = 0 :=
by sorry

end option_d_not_necessarily_true_l1574_157421


namespace original_weight_of_beef_l1574_157450

variable (W : ℝ)

def first_stage_weight := 0.80 * W
def second_stage_weight := 0.70 * (first_stage_weight W)
def third_stage_weight := 0.75 * (second_stage_weight W)

theorem original_weight_of_beef :
  third_stage_weight W = 392 → W = 933.33 :=
by
  intro h
  sorry

end original_weight_of_beef_l1574_157450


namespace trigonometric_identity_tan_two_l1574_157416

theorem trigonometric_identity_tan_two (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 11 / 5 :=
by
  sorry

end trigonometric_identity_tan_two_l1574_157416


namespace total_flour_needed_l1574_157436

noncomputable def katie_flour : ℝ := 3

noncomputable def sheila_flour : ℝ := katie_flour + 2

noncomputable def john_flour : ℝ := 1.5 * sheila_flour

theorem total_flour_needed :
  katie_flour + sheila_flour + john_flour = 15.5 :=
by
  sorry

end total_flour_needed_l1574_157436


namespace donna_pizza_slices_l1574_157443

theorem donna_pizza_slices :
  ∀ (total_slices : ℕ) (half_eaten_for_lunch : ℕ) (one_third_eaten_for_dinner : ℕ),
  total_slices = 12 →
  half_eaten_for_lunch = total_slices / 2 →
  one_third_eaten_for_dinner = half_eaten_for_lunch / 3 →
  (half_eaten_for_lunch - one_third_eaten_for_dinner) = 4 :=
by
  intros total_slices half_eaten_for_lunch one_third_eaten_for_dinner
  intros h1 h2 h3
  sorry

end donna_pizza_slices_l1574_157443


namespace sum_positive_132_l1574_157434

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem sum_positive_132 {a: ℕ → ℝ}
  (h1: a 66 < 0)
  (h2: a 67 > 0)
  (h3: a 67 > |a 66|):
  ∃ n, ∀ k < n, S k > 0 :=
by
  have h4 : (a 67 - a 66) > 0 := sorry
  have h5 : a 67 + a 66 > 0 := sorry
  have h6 : 66 * (a 67 + a 66) > 0 := sorry
  have h7 : S 132 = 66 * (a 67 + a 66) := sorry
  existsi 132
  intro k hk
  sorry

end sum_positive_132_l1574_157434


namespace percentage_decrease_equivalent_l1574_157468

theorem percentage_decrease_equivalent :
  ∀ (P D : ℝ), 
    (D = 10) →
    ((1.25 * P) - (D / 100) * (1.25 * P) = 1.125 * P) :=
by
  intros P D h
  rw [h]
  sorry

end percentage_decrease_equivalent_l1574_157468


namespace age_difference_l1574_157432

variable (S M : ℕ)

theorem age_difference (hS : S = 28) (hM : M + 2 = 2 * (S + 2)) : M - S = 30 :=
by
  sorry

end age_difference_l1574_157432


namespace isosceles_triangle_problem_l1574_157406

theorem isosceles_triangle_problem 
  (a h b : ℝ) 
  (area_relation : (1/2) * a * h = (1/3) * a ^ 2) 
  (leg_relation : b = a - 1)
  (height_relation : h = (2/3) * a) 
  (pythagorean_theorem : h ^ 2 + (a / 2) ^ 2 = b ^ 2) : 
  a = 6 ∧ b = 5 ∧ h = 4 :=
sorry

end isosceles_triangle_problem_l1574_157406
