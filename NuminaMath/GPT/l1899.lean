import Mathlib

namespace savings_together_vs_separate_l1899_189955

def price_per_window : ℕ := 100

def free_windows_per_5_purchased : ℕ := 2

def daves_windows_needed : ℕ := 10

def dougs_windows_needed : ℕ := 11

def total_windows_needed : ℕ := daves_windows_needed + dougs_windows_needed

-- Cost calculation for Dave's windows with the offer
def daves_cost_with_offer : ℕ := 8 * price_per_window

-- Cost calculation for Doug's windows with the offer
def dougs_cost_with_offer : ℕ := 9 * price_per_window

-- Total cost calculation if purchased separately with the offer
def total_cost_separately_with_offer : ℕ := daves_cost_with_offer + dougs_cost_with_offer

-- Total cost calculation if purchased together with the offer
def total_cost_together_with_offer : ℕ := 17 * price_per_window

-- Calculate additional savings if Dave and Doug purchase together rather than separately
def additional_savings_together_vs_separate := 
  total_cost_separately_with_offer - total_cost_together_with_offer = 0

theorem savings_together_vs_separate : additional_savings_together_vs_separate := by
  sorry

end savings_together_vs_separate_l1899_189955


namespace steve_speed_back_l1899_189919

theorem steve_speed_back :
  ∀ (d v_total : ℕ), d = 10 → v_total = 6 →
  (2 * (15 / 6)) = 5 :=
by
  intros d v_total d_eq v_total_eq
  sorry

end steve_speed_back_l1899_189919


namespace complement_A_inter_B_range_of_a_l1899_189944

open Set

-- Define sets A and B based on the conditions
def A : Set ℝ := {x | -4 ≤ x - 6 ∧ x - 6 ≤ 0}
def B : Set ℝ := {x | 2 * x - 6 ≥ 3 - x}

-- Define set C based on the conditions
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Problem 1: Prove the complement of (A ∩ B) in ℝ is the set of x where (x < 3 or x > 6)
theorem complement_A_inter_B :
  compl (A ∩ B) = {x | x < 3} ∪ {x | x > 6} :=
sorry

-- Problem 2: Prove that A ∩ C = A implies a ∈ [6, ∞)
theorem range_of_a {a : ℝ} (hC : A ∩ C a = A) :
  6 ≤ a :=
sorry

end complement_A_inter_B_range_of_a_l1899_189944


namespace solve_equation_l1899_189929

theorem solve_equation :
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} :=
by
  sorry

end solve_equation_l1899_189929


namespace walk_to_lake_park_restaurant_is_zero_l1899_189957

noncomputable def time_to_hidden_lake : ℕ := 15
noncomputable def time_to_return_from_hidden_lake : ℕ := 7
noncomputable def total_walk_time_dante : ℕ := 22

theorem walk_to_lake_park_restaurant_is_zero :
  ∃ (x : ℕ), (2 * x + time_to_hidden_lake + time_to_return_from_hidden_lake = total_walk_time_dante) → x = 0 :=
by
  use 0
  intros
  sorry

end walk_to_lake_park_restaurant_is_zero_l1899_189957


namespace minimum_height_l1899_189921

theorem minimum_height (y : ℝ) (h : ℝ) (S : ℝ) (hS : S = 10 * y^2) (hS_min : S ≥ 150) (h_height : h = 2 * y) : h = 2 * Real.sqrt 15 :=
  sorry

end minimum_height_l1899_189921


namespace focus_of_given_parabola_l1899_189902

-- Define the given condition as a parameter
def parabola_eq (x y : ℝ) : Prop :=
  y = - (1/2) * x^2

-- Define the property for the focus of the parabola
def is_focus_of_parabola (focus : ℝ × ℝ) : Prop :=
  focus = (0, -1/2)

-- The theorem stating that the given parabola equation has the specific focus
theorem focus_of_given_parabola : 
  (∀ x y : ℝ, parabola_eq x y) → is_focus_of_parabola (0, -1/2) :=
by
  intro h
  unfold parabola_eq at h
  unfold is_focus_of_parabola
  sorry

end focus_of_given_parabola_l1899_189902


namespace geometric_sequence_sum_63_l1899_189937

theorem geometric_sequence_sum_63
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_init : a 1 = 1)
  (h_recurrence : ∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 63 :=
by
  sorry

end geometric_sequence_sum_63_l1899_189937


namespace part1_part2_l1899_189989

variable {x : ℝ}

/-- Prove that the range of the function f(x) = (sqrt(1+x) + sqrt(1-x) + 2) * (sqrt(1-x^2) + 1) for 0 ≤ x ≤ 1 is (0, 8]. -/
theorem part1 (hx : 0 ≤ x ∧ x ≤ 1) :
  0 < ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ∧ 
  ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ≤ 8 :=
sorry

/-- Prove that for 0 ≤ x ≤ 1, there exists a positive number β such that sqrt(1+x) + sqrt(1-x) ≤ 2 - x^2 / β, with the minimal β = 4. -/
theorem part2 (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ β : ℝ, β > 0 ∧ β = 4 ∧ (Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) :=
sorry

end part1_part2_l1899_189989


namespace ratio_of_green_to_blue_l1899_189974

-- Definitions of the areas and the circles
noncomputable def red_area : ℝ := Real.pi * (1 : ℝ) ^ 2
noncomputable def middle_area : ℝ := Real.pi * (2 : ℝ) ^ 2
noncomputable def large_area: ℝ := Real.pi * (3 : ℝ) ^ 2

noncomputable def blue_area : ℝ := middle_area - red_area
noncomputable def green_area : ℝ := large_area - middle_area

-- The proof that the ratio of the green area to the blue area is 5/3
theorem ratio_of_green_to_blue : green_area / blue_area = 5 / 3 := by
  sorry

end ratio_of_green_to_blue_l1899_189974


namespace hypotenuse_right_triangle_l1899_189980

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end hypotenuse_right_triangle_l1899_189980


namespace f_1_geq_25_l1899_189951

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State that f is increasing on the interval [-2, +∞)
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y : ℝ, -2 ≤ x → x ≤ y → f x m ≤ f y m

-- Prove that given the function is increasing on [-2, +∞),
-- then f(1) is at least 25.
theorem f_1_geq_25 (m : ℝ) (h : is_increasing_on_interval m) : f 1 m ≥ 25 :=
  sorry

end f_1_geq_25_l1899_189951


namespace cube_surface_area_l1899_189938

theorem cube_surface_area (V : ℝ) (hV : V = 125) : ∃ A : ℝ, A = 25 :=
by
  sorry

end cube_surface_area_l1899_189938


namespace binomial_variance_is_one_l1899_189965

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem binomial_variance_is_one :
  binomial_variance 4 (1 / 2) = 1 := by
  sorry

end binomial_variance_is_one_l1899_189965


namespace farmer_loss_representative_value_l1899_189913

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end farmer_loss_representative_value_l1899_189913


namespace solve_inequalities_solve_fruit_purchase_l1899_189964

-- Part 1: Inequalities
theorem solve_inequalities {x : ℝ} : 
  (2 * x < 16) ∧ (3 * x > 2 * x + 3) → (3 < x ∧ x < 8) := by
  sorry

-- Part 2: Fruit Purchase
theorem solve_fruit_purchase {x y : ℝ} : 
  (x + y = 7) ∧ (5 * x + 8 * y = 41) → (x = 5 ∧ y = 2) := by
  sorry

end solve_inequalities_solve_fruit_purchase_l1899_189964


namespace price_of_adult_ticket_eq_32_l1899_189990

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end price_of_adult_ticket_eq_32_l1899_189990


namespace inequality_log_range_of_a_l1899_189936

open Real

theorem inequality_log (x : ℝ) (h₀ : 0 < x) : 
  1 - 1 / x ≤ log x ∧ log x ≤ x - 1 := sorry

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 < x ∧ x ≤ 1 → a * (1 - x^2) + x^2 * log x ≥ 0) : 
  a ≥ 1/2 := sorry

end inequality_log_range_of_a_l1899_189936


namespace initial_contribution_l1899_189941

theorem initial_contribution (j k l : ℝ)
  (h1 : j + k + l = 1200)
  (h2 : j - 200 + 3 * (k + l) = 1800) :
  j = 800 :=
sorry

end initial_contribution_l1899_189941


namespace geometric_sequence_solution_l1899_189987

variables (a : ℕ → ℝ) (q : ℝ)
-- Given conditions
def condition1 : Prop := abs (a 1) = 1
def condition2 : Prop := a 5 = -8 * a 2
def condition3 : Prop := a 5 > a 2
-- Proof statement
theorem geometric_sequence_solution :
  condition1 a → condition2 a → condition3 a → ∀ n, a n = (-2)^(n - 1) :=
sorry

end geometric_sequence_solution_l1899_189987


namespace rick_books_division_l1899_189917

theorem rick_books_division (books_per_group initial_books final_groups : ℕ) 
  (h_initial : initial_books = 400) 
  (h_books_per_group : books_per_group = 25) 
  (h_final_groups : final_groups = 16) : 
  ∃ divisions : ℕ, (divisions = 4) ∧ 
    ∃ f : ℕ → ℕ, 
    (f 0 = initial_books) ∧ 
    (f divisions = books_per_group * final_groups) ∧ 
    (∀ n, 1 ≤ n → n ≤ divisions → f n = f (n - 1) / 2) := 
by 
  sorry

end rick_books_division_l1899_189917


namespace initial_tax_rate_l1899_189991

theorem initial_tax_rate 
  (income : ℝ)
  (differential_savings : ℝ)
  (final_tax_rate : ℝ)
  (initial_tax_rate : ℝ) 
  (h1 : income = 42400) 
  (h2 : differential_savings = 4240) 
  (h3 : final_tax_rate = 32)
  (h4 : differential_savings = (initial_tax_rate / 100) * income - (final_tax_rate / 100) * income) :
  initial_tax_rate = 42 :=
sorry

end initial_tax_rate_l1899_189991


namespace baker_cakes_remaining_l1899_189982

theorem baker_cakes_remaining (initial_cakes: ℕ) (fraction_sold: ℚ) (sold_cakes: ℕ) (cakes_remaining: ℕ) :
  initial_cakes = 149 ∧ fraction_sold = 2/5 ∧ sold_cakes = 59 ∧ cakes_remaining = initial_cakes - sold_cakes → cakes_remaining = 90 :=
by
  sorry

end baker_cakes_remaining_l1899_189982


namespace greatest_discarded_oranges_l1899_189997

theorem greatest_discarded_oranges (n : ℕ) : n % 7 ≤ 6 := 
by 
  sorry

end greatest_discarded_oranges_l1899_189997


namespace angle_bisector_inequality_l1899_189945

theorem angle_bisector_inequality {a b c fa fb fc : ℝ} 
  (h_triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angle_bisectors : fa > 0 ∧ fb > 0 ∧ fc > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (1 / fa + 1 / fb + 1 / fc > 1 / a + 1 / b + 1 / c) :=
by
  sorry

end angle_bisector_inequality_l1899_189945


namespace find_boys_and_girls_l1899_189911

noncomputable def number_of_boys_and_girls (a b c d : Nat) : (Nat × Nat) := sorry

theorem find_boys_and_girls : 
  ∃ m d : Nat,
  (∀ (a b c : Nat), 
    ((a = 15 ∨ b = 18 ∨ c = 13) ∧ 
    (a.mod 4 = 3 ∨ b.mod 4 = 2 ∨ c.mod 4 = 1)) 
    → number_of_boys_and_girls a b c d = (16, 14)) :=
sorry

end find_boys_and_girls_l1899_189911


namespace beverage_price_l1899_189966

theorem beverage_price (P : ℝ) :
  (3 * 2.25 + 4 * P + 4 * 1.00) / 6 = 2.79 → P = 1.50 :=
by
  intro h -- Introduce the hypothesis.
  sorry  -- Proof is omitted.

end beverage_price_l1899_189966


namespace solution_set_of_x_abs_x_lt_x_l1899_189975

theorem solution_set_of_x_abs_x_lt_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} :=
by
  sorry

end solution_set_of_x_abs_x_lt_x_l1899_189975


namespace fraction_of_students_received_B_l1899_189959

theorem fraction_of_students_received_B {total_students : ℝ}
  (fraction_A : ℝ)
  (fraction_A_or_B : ℝ)
  (h_fraction_A : fraction_A = 0.7)
  (h_fraction_A_or_B : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 :=
by
  rw [h_fraction_A, h_fraction_A_or_B]
  sorry

end fraction_of_students_received_B_l1899_189959


namespace amelia_wins_l1899_189915

noncomputable def amelia_wins_probability : ℚ := 21609 / 64328

theorem amelia_wins (h_am_heads : ℚ) (h_bl_heads : ℚ) (game_starts : Prop) (game_alternates : Prop) (win_condition : Prop) :
  h_am_heads = 3/7 ∧ h_bl_heads = 1/3 ∧ game_starts ∧ game_alternates ∧ win_condition →
  amelia_wins_probability = 21609 / 64328 :=
sorry

end amelia_wins_l1899_189915


namespace trigonometric_identity_l1899_189956

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + (Real.pi / 3)) = 3 / 5) :
  Real.cos ((Real.pi / 6) - α) = 3 / 5 :=
by
  sorry

end trigonometric_identity_l1899_189956


namespace three_digit_number_count_l1899_189939

theorem three_digit_number_count :
  ∃ n : ℕ, n = 15 ∧
  (∀ a b c : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) →
    (100 * a + 10 * b + c = 37 * (a + b + c) → ∃ k : ℕ, k = n)) :=
sorry

end three_digit_number_count_l1899_189939


namespace find_number_l1899_189904

theorem find_number (X : ℝ) (h : 30 = 0.50 * X + 10) : X = 40 :=
by
  sorry

end find_number_l1899_189904


namespace contrapositive_l1899_189916

theorem contrapositive (a b : ℝ) :
  (a > b → a^2 > b^2) → (a^2 ≤ b^2 → a ≤ b) :=
by
  intro h
  sorry

end contrapositive_l1899_189916


namespace quadratic_inequality_solution_l1899_189930

theorem quadratic_inequality_solution (y : ℝ) : 
  (y^2 - 9 * y + 14 ≤ 0) ↔ (2 ≤ y ∧ y ≤ 7) :=
sorry

end quadratic_inequality_solution_l1899_189930


namespace find_number_added_l1899_189925

theorem find_number_added (x n : ℕ) (h : (x + x + 2 + x + 4 + x + n + x + 22) / 5 = x + 7) : n = 7 :=
by
  sorry

end find_number_added_l1899_189925


namespace x_squared_plus_y_squared_l1899_189983

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := 
by
  sorry

end x_squared_plus_y_squared_l1899_189983


namespace area_difference_l1899_189918

-- Setting up the relevant conditions and entities
def side_red := 8
def length_yellow := 10
def width_yellow := 5

-- Definition of areas
def area_red := side_red * side_red
def area_yellow := length_yellow * width_yellow

-- The theorem we need to prove
theorem area_difference :
  area_red - area_yellow = 14 :=
by
  -- We skip the proof here due to the instruction
  sorry

end area_difference_l1899_189918


namespace incorrect_expression_among_options_l1899_189999

theorem incorrect_expression_among_options :
  ¬(0.75 ^ (-0.3) < 0.75 ^ (0.1)) :=
by
  sorry

end incorrect_expression_among_options_l1899_189999


namespace find_F2_l1899_189960

-- Set up the conditions as definitions
def m : ℝ := 1 -- in kg
def R1 : ℝ := 0.5 -- in meters
def R2 : ℝ := 1 -- in meters
def F1 : ℝ := 1 -- in Newtons

-- Rotational inertia I formula
def I (R : ℝ) : ℝ := m * R^2

-- Equality of angular accelerations
def alpha_eq (F1 F2 R1 R2 : ℝ) : Prop :=
  (F1 * R1) / (I R1) = (F2 * R2) / (I R2)

-- The proof goal
theorem find_F2 (F2 : ℝ) : 
  alpha_eq F1 F2 R1 R2 → F2 = 2 :=
by
  sorry

end find_F2_l1899_189960


namespace original_apples_l1899_189988

-- Define the conditions
def sells_50_percent (initial remaining : ℕ) : Prop :=
  (initial / 2) = remaining

-- Define the goal
theorem original_apples (remaining : ℕ) (initial : ℕ) (h : sells_50_percent initial remaining) : initial = 10000 :=
by
  sorry

end original_apples_l1899_189988


namespace simplify_expression_l1899_189994

theorem simplify_expression (a b : ℝ) :
  ((3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b) = (-a^2 + 2 * b^2) :=
by
  sorry

end simplify_expression_l1899_189994


namespace max_rectangle_area_l1899_189971

theorem max_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) (h1 : l + w = 20) (hlw : l = 10 ∨ w = 10) : 
(l = 10 ∧ w = 10 ∧ l * w = 100) :=
by sorry

end max_rectangle_area_l1899_189971


namespace coffee_on_Thursday_coffee_on_Friday_average_coffee_l1899_189954

noncomputable def coffee_consumption (k h : ℝ) : ℝ := k / h

theorem coffee_on_Thursday : coffee_consumption 24 4 = 6 :=
by sorry

theorem coffee_on_Friday : coffee_consumption 24 10 = 2.4 :=
by sorry

theorem average_coffee : 
  (coffee_consumption 24 8 + coffee_consumption 24 4 + coffee_consumption 24 10) / 3 = 3.8 :=
by sorry

end coffee_on_Thursday_coffee_on_Friday_average_coffee_l1899_189954


namespace arithmetic_sequence_problem_l1899_189962

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) 
  (a1 : a 1 = 1 / 3) 
  (a2_a5 : a 2 + a 5 = 4) 
  (an : ∃ n, a n = 33) :
  ∃ n, a n = 33 ∧ n = 50 := 
by 
  sorry

end arithmetic_sequence_problem_l1899_189962


namespace decreasing_function_iff_m_eq_2_l1899_189912

theorem decreasing_function_iff_m_eq_2 
    (m : ℝ) : 
    (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x^(-5*m - 3) < (m^2 - m - 1) * (x + 1)^(-5*m - 3)) ↔ m = 2 := 
sorry

end decreasing_function_iff_m_eq_2_l1899_189912


namespace find_t_given_V_S_l1899_189949

variables (g V V0 S S0 a t : ℝ)

theorem find_t_given_V_S :
  (V = g * (t - a) + V0) →
  (S = (1 / 2) * g * (t - a) ^ 2 + V0 * (t - a) + S0) →
  t = a + (V - V0) / g :=
by
  intros h1 h2
  sorry

end find_t_given_V_S_l1899_189949


namespace arithmetic_sequence_15th_term_l1899_189905

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 3) (h2 : a2 = 14) (h3 : a3 = 25) (h4 : d = a2 - a1) (h5 : a2 - a1 = a3 - a2) (h6 : n = 15) :
  a1 + (n - 1) * d = 157 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_15th_term_l1899_189905


namespace increase_by_one_unit_l1899_189977

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 + 3 * x

-- State the theorem
theorem increase_by_one_unit (x : ℝ) : regression_eq (x + 1) - regression_eq x = 3 := by
  sorry

end increase_by_one_unit_l1899_189977


namespace sum_n_k_l1899_189935

theorem sum_n_k (n k : ℕ) (h1 : 3 = n - 2 * k) (h2 : 15 = 5 * n - 8 * k) : n + k = 3 :=
by
  -- Use the conditions to conclude the proof.
  sorry

end sum_n_k_l1899_189935


namespace complex_quadrant_l1899_189970

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end complex_quadrant_l1899_189970


namespace S_10_value_l1899_189903

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := n * (a 1 + a n) / 2

theorem S_10_value (a : ℕ → ℕ) (h1 : a 2 = 3) (h2 : a 9 = 17) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) : 
  S 10 a = 100 := 
by
  sorry

end S_10_value_l1899_189903


namespace james_out_of_pocket_l1899_189961

-- Definitions based on conditions
def old_car_value : ℝ := 20000
def old_car_sold_for : ℝ := 0.80 * old_car_value
def new_car_sticker_price : ℝ := 30000
def new_car_bought_for : ℝ := 0.90 * new_car_sticker_price

-- Question and proof statement
def amount_out_of_pocket : ℝ := new_car_bought_for - old_car_sold_for

theorem james_out_of_pocket : amount_out_of_pocket = 11000 := by
  sorry

end james_out_of_pocket_l1899_189961


namespace negation_of_there_exists_l1899_189984

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end negation_of_there_exists_l1899_189984


namespace fraction_of_students_saying_dislike_actually_like_l1899_189927

variables (total_students liking_disliking_students saying_disliking_like_students : ℚ)
          (fraction_like_dislike say_dislike : ℚ)
          (cond1 : 0.7 = liking_disliking_students / total_students) 
          (cond2 : 0.3 = (total_students - liking_disliking_students) / total_students)
          (cond3 : 0.3 * liking_disliking_students = saying_disliking_like_students)
          (cond4 : 0.8 * (total_students - liking_disliking_students) 
                    = say_dislike)

theorem fraction_of_students_saying_dislike_actually_like
    (total_students_eq: total_students = 100) : 
    fraction_like_dislike = 46.67 :=
by
  sorry

end fraction_of_students_saying_dislike_actually_like_l1899_189927


namespace tan_alpha_solution_l1899_189909

variable (α : ℝ)
variable (h₀ : 0 < α ∧ α < π)
variable (h₁ : Real.sin α + Real.cos α = 7 / 13)

theorem tan_alpha_solution : Real.tan α = -12 / 5 := 
by
  sorry

end tan_alpha_solution_l1899_189909


namespace seq_20_eq_5_over_7_l1899_189953

theorem seq_20_eq_5_over_7 :
  ∃ (a : ℕ → ℚ), 
    a 1 = 6 / 7 ∧ 
    (∀ n, (0 ≤ a n ∧ a n < 1) → 
      (a (n + 1) = if a n < 1 / 2 then 2 * a n else 2 * a n - 1)) ∧ 
    a 20 = 5 / 7 := 
sorry

end seq_20_eq_5_over_7_l1899_189953


namespace total_journey_distance_l1899_189910

theorem total_journey_distance
  (T : ℝ) (D : ℝ)
  (h1 : T = 20)
  (h2 : (D / 2) / 21 + (D / 2) / 24 = 20) :
  D = 448 :=
by
  sorry

end total_journey_distance_l1899_189910


namespace daughter_current_age_l1899_189923

-- Define the conditions
def mother_current_age := 42
def years_later := 9
def mother_age_in_9_years := mother_current_age + years_later
def daughter_age_in_9_years (D : ℕ) := D + years_later

-- Define the statement we need to prove
theorem daughter_current_age : ∃ D : ℕ, mother_age_in_9_years = 3 * daughter_age_in_9_years D ∧ D = 8 :=
by {
  sorry
}

end daughter_current_age_l1899_189923


namespace value_of_5_l1899_189981

def q' (q : ℤ) : ℤ := 3 * q - 3

theorem value_of_5'_prime : q' (q' 5) = 33 :=
by
  sorry

end value_of_5_l1899_189981


namespace quadrilateral_is_parallelogram_l1899_189943

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 2 * a * b + 2 * c * d) : a = b ∧ c = d :=
by
  sorry

end quadrilateral_is_parallelogram_l1899_189943


namespace train_speed_l1899_189914

theorem train_speed (len_train len_bridge time : ℝ) (h_len_train : len_train = 120)
  (h_len_bridge : len_bridge = 150) (h_time : time = 26.997840172786177) :
  let total_distance := len_train + len_bridge
  let speed_m_s := total_distance / time
  let speed_km_h := speed_m_s * 3.6
  speed_km_h = 36 :=
by
  -- Proof goes here
  sorry

end train_speed_l1899_189914


namespace find_original_list_size_l1899_189992

theorem find_original_list_size
  (n m : ℤ)
  (h1 : (m + 3) * (n + 1) = m * n + 20)
  (h2 : (m + 1) * (n + 2) = m * n + 22):
  n = 7 :=
sorry

end find_original_list_size_l1899_189992


namespace strawberries_left_correct_l1899_189958

-- Define the initial and given away amounts in kilograms and grams
def initial_strawberries_kg : Int := 3
def initial_strawberries_g : Int := 300
def given_strawberries_kg : Int := 1
def given_strawberries_g : Int := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : Int) : Int := kg * 1000

-- Calculate the total strawberries initially and given away in grams
def total_initial_strawberries_g : Int :=
  (kg_to_g initial_strawberries_kg) + initial_strawberries_g

def total_given_strawberries_g : Int :=
  (kg_to_g given_strawberries_kg) + given_strawberries_g

-- The amount of strawberries left after giving some away
def strawberries_left : Int :=
  total_initial_strawberries_g - total_given_strawberries_g

-- The statement to prove
theorem strawberries_left_correct :
  strawberries_left = 1400 :=
by
  sorry

end strawberries_left_correct_l1899_189958


namespace area_of_triangle_ABC_is_24_l1899_189940

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the area calculation
def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  0.5 * |(v.1 * w.2 - v.2 * w.1)|

theorem area_of_triangle_ABC_is_24 :
  triangleArea A B C = 24 := by
  sorry

end area_of_triangle_ABC_is_24_l1899_189940


namespace parallel_lines_iff_a_eq_1_l1899_189969

theorem parallel_lines_iff_a_eq_1 (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ x + 2*y + 4 = 0) ↔ (a = 1) := 
sorry

end parallel_lines_iff_a_eq_1_l1899_189969


namespace find_A_l1899_189967

theorem find_A (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1/a + 1/b = 2) : A = Real.sqrt 15 :=
by
  /- Proof omitted -/
  sorry

end find_A_l1899_189967


namespace a4_plus_a5_eq_27_l1899_189950

-- Define the geometric sequence conditions
variables (a : ℕ → ℝ) (q : ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a_2 : a 2 = 1 - a 1
axiom a_4 : a 4 = 9 - a 3

-- Define the geometric sequence property
axiom geom_seq : ∀ n, a (n + 1) = a n * q

theorem a4_plus_a5_eq_27 : a 4 + a 5 = 27 := sorry

end a4_plus_a5_eq_27_l1899_189950


namespace increase_in_y_coordinate_l1899_189924

theorem increase_in_y_coordinate (m n : ℝ) (h₁ : m = (n / 5) - 2 / 5) : 
  (5 * (m + 3) + 2) - (5 * m + 2) = 15 :=
by
  sorry

end increase_in_y_coordinate_l1899_189924


namespace license_count_l1899_189993

def num_licenses : ℕ :=
  let num_letters := 3
  let num_digits := 10
  let num_digit_slots := 6
  num_letters * num_digits ^ num_digit_slots

theorem license_count :
  num_licenses = 3000000 := by
  sorry

end license_count_l1899_189993


namespace remainder_x5_3x3_2x2_x_2_div_x_minus_2_l1899_189942

def polynomial (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + x + 2

theorem remainder_x5_3x3_2x2_x_2_div_x_minus_2 :
  polynomial 2 = 68 := 
by 
  sorry

end remainder_x5_3x3_2x2_x_2_div_x_minus_2_l1899_189942


namespace set_equality_x_plus_y_l1899_189900

theorem set_equality_x_plus_y (x y : ℝ) (A B : Set ℝ) (hA : A = {0, |x|, y}) (hB : B = {x, x * y, Real.sqrt (x - y)}) (h : A = B) : x + y = -2 :=
by
  sorry

end set_equality_x_plus_y_l1899_189900


namespace gain_percent_is_correct_l1899_189908

noncomputable def gain_percent (CP SP : ℝ) : ℝ :=
  let gain := SP - CP
  (gain / CP) * 100

theorem gain_percent_is_correct :
  gain_percent 930 1210 = 30.11 :=
by
  sorry

end gain_percent_is_correct_l1899_189908


namespace cube_root_of_sum_of_powers_l1899_189907

theorem cube_root_of_sum_of_powers :
  ∃ (x : ℝ), x = 16 * (4 ^ (1 / 3)) ∧ x = (4^6 + 4^6 + 4^6 + 4^6) ^ (1 / 3) :=
by
  sorry

end cube_root_of_sum_of_powers_l1899_189907


namespace polynomial_evaluation_qin_jiushao_l1899_189979

theorem polynomial_evaluation_qin_jiushao :
  let x := 3
  let V0 := 7
  let V1 := V0 * x + 6
  let V2 := V1 * x + 5
  let V3 := V2 * x + 4
  let V4 := V3 * x + 3
  V4 = 789 :=
by
  -- placeholder for proof
  sorry

end polynomial_evaluation_qin_jiushao_l1899_189979


namespace parametric_curve_C_line_tangent_to_curve_C_l1899_189931

open Real

-- Definitions of the curve C and line l
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 4 * ρ * cos θ + 1 = 0

def line_l (t α x y : ℝ) : Prop := x = 4 + t * sin α ∧ y = t * cos α ∧ 0 ≤ α ∧ α < π

-- Parametric equation of curve C
theorem parametric_curve_C :
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * π →
  ∃ x y : ℝ, (x = 2 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ ∧
              curve_C (sqrt (x^2 + y^2)) θ) :=
sorry

-- Tangency condition for line l and curve C
theorem line_tangent_to_curve_C :
  ∀ α : ℝ, 0 ≤ α ∧ α < π →
  (∃ t : ℝ, ∃ x y : ℝ, (line_l t α x y ∧ (x - 2)^2 + y^2 = 3 ∧
                        ((abs (2 * cos α - 4 * cos α) / sqrt (cos α ^ 2 + sin α ^ 2)) = sqrt 3)) →
                       (α = π / 6 ∧ x = 7 / 2 ∧ y = - sqrt 3 / 2)) :=
sorry

end parametric_curve_C_line_tangent_to_curve_C_l1899_189931


namespace quadratic_complete_square_r_plus_s_l1899_189934

theorem quadratic_complete_square_r_plus_s :
  ∃ r s : ℚ, (∀ x : ℚ, 7 * x^2 - 21 * x - 56 = 0 → (x + r)^2 = s) ∧ r + s = 35 / 4 := sorry

end quadratic_complete_square_r_plus_s_l1899_189934


namespace new_volume_increased_dimensions_l1899_189995

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end new_volume_increased_dimensions_l1899_189995


namespace log_lt_x_l1899_189946

theorem log_lt_x (x : ℝ) (hx : 0 < x) : Real.log (1 + x) < x := 
sorry

end log_lt_x_l1899_189946


namespace base_b_eq_five_l1899_189920

theorem base_b_eq_five (b : ℕ) (h1 : 1225 = b^3 + 2 * b^2 + 2 * b + 5) (h2 : 35 = 3 * b + 5) :
    (3 * b + 5)^2 = b^3 + 2 * b^2 + 2 * b + 5 ↔ b = 5 :=
by
  sorry

end base_b_eq_five_l1899_189920


namespace RelativelyPrimeProbability_l1899_189901

def relatively_prime_probability_42 : Rat :=
  let n := 42
  let total := n
  let rel_prime_count := total - (21 + 14 + 6 - 7 - 3 - 2 + 1)
  let probability := (rel_prime_count : Rat) / total
  probability

theorem RelativelyPrimeProbability : relatively_prime_probability_42 = 2 / 7 :=
sorry

end RelativelyPrimeProbability_l1899_189901


namespace Julie_initial_savings_l1899_189932

theorem Julie_initial_savings (P r : ℝ) 
  (h1 : 100 = P * r * 2) 
  (h2 : 105 = P * (1 + r) ^ 2 - P) : 
  2 * P = 1000 :=
by
  sorry

end Julie_initial_savings_l1899_189932


namespace mode_of_gold_medals_is_8_l1899_189986

def countries : List String := ["Norway", "Germany", "China", "USA", "Sweden", "Netherlands", "Austria"]

def gold_medals : List Nat := [16, 12, 9, 8, 8, 8, 7]

def mode (lst : List Nat) : Nat :=
  lst.foldr
    (fun (x : Nat) acc =>
      if lst.count x > lst.count acc then x else acc)
    lst.head!

theorem mode_of_gold_medals_is_8 :
  mode gold_medals = 8 :=
by sorry

end mode_of_gold_medals_is_8_l1899_189986


namespace find_x_l1899_189926

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x+1, -x)

def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x (x : ℝ) (h : perpendicular vector_a (vector_b x)) : x = 1 :=
by sorry

end find_x_l1899_189926


namespace simplify_expression_l1899_189906

theorem simplify_expression (x y : ℝ) :
  (3 * x^2 * y)^3 + (4 * x * y) * y^4 = 27 * x^6 * y^3 + 4 * x * y^5 :=
by 
  sorry

end simplify_expression_l1899_189906


namespace valid_pairs_l1899_189928

theorem valid_pairs
  (x y : ℕ)
  (h_pos_x : x > 0)
  (h_pos_y : y > 0)
  (h_div : ∃ k : ℕ, k > 0 ∧ k * (2 * x + 7 * y) = 7 * x + 2 * y) :
  ∃ a : ℕ, a > 0 ∧ (x = a ∧ y = a ∨ x = 4 * a ∧ y = a ∨ x = 19 * a ∧ y = a) :=
by
  sorry

end valid_pairs_l1899_189928


namespace current_speed_correct_l1899_189978

noncomputable def speed_of_current : ℝ :=
  let rowing_speed_still_water := 10 -- speed of rowing in still water in kmph
  let distance_meters := 60 -- distance covered in meters
  let time_seconds := 17.998560115190788 -- time taken in seconds
  let distance_km := distance_meters / 1000 -- converting distance to kilometers
  let time_hours := time_seconds / 3600 -- converting time to hours
  let downstream_speed := distance_km / time_hours -- calculating downstream speed
  downstream_speed - rowing_speed_still_water -- calculating and returning the speed of the current

theorem current_speed_correct : speed_of_current = 2.00048 := by
  -- The proof is not provided in this statement as per the requirements.
  sorry

end current_speed_correct_l1899_189978


namespace number_of_pages_in_bible_l1899_189998

-- Definitions based on conditions
def hours_per_day := 2
def pages_per_hour := 50
def weeks := 4
def days_per_week := 7

-- Hypotheses transformed into mathematical facts
def total_days := weeks * days_per_week
def total_hours := total_days * hours_per_day
def total_pages := total_hours * pages_per_hour

-- Theorem to prove the Bible length based on conditions
theorem number_of_pages_in_bible : total_pages = 2800 := 
by
  sorry

end number_of_pages_in_bible_l1899_189998


namespace problem_solution_l1899_189985

def count_valid_n : ℕ :=
  let count_mult_3 := (3000 / 3)
  let count_mult_6 := (3000 / 6)
  count_mult_3 - count_mult_6

theorem problem_solution : count_valid_n = 500 := 
sorry

end problem_solution_l1899_189985


namespace percentage_increase_in_llama_cost_l1899_189968

def cost_of_goat : ℕ := 400
def number_of_goats : ℕ := 3
def total_cost : ℕ := 4800

def llamas_cost (x : ℕ) : Prop :=
  let total_cost_goats := number_of_goats * cost_of_goat
  let total_cost_llamas := total_cost - total_cost_goats
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := total_cost_llamas / number_of_llamas
  let increase := cost_per_llama - cost_of_goat
  ((increase / cost_of_goat) * 100) = x

theorem percentage_increase_in_llama_cost :
  llamas_cost 50 :=
sorry

end percentage_increase_in_llama_cost_l1899_189968


namespace cube_side_length_ratio_l1899_189947

-- Define the conditions and question
variable (s₁ s₂ : ℝ)
variable (weight₁ weight₂ : ℝ)
variable (V₁ V₂ : ℝ)
variable (same_metal : Prop)

-- Conditions
def condition1 (weight₁ : ℝ) : Prop := weight₁ = 4
def condition2 (weight₂ : ℝ) : Prop := weight₂ = 32
def condition3 (V₁ V₂ : ℝ) (s₁ s₂ : ℝ) : Prop := (V₁ = s₁^3) ∧ (V₂ = s₂^3)
def condition4 (same_metal : Prop) : Prop := same_metal

-- Volume definition based on weights and proportion
noncomputable def volume_definition (weight₁ weight₂ V₁ V₂ : ℝ) : Prop :=
(weight₂ / weight₁) = (V₂ / V₁)

-- Define the proof target
theorem cube_side_length_ratio
    (h1 : condition1 weight₁)
    (h2 : condition2 weight₂)
    (h3 : condition3 V₁ V₂ s₁ s₂)
    (h4 : condition4 same_metal)
    (h5 : volume_definition weight₁ weight₂ V₁ V₂) : 
    (s₂ / s₁) = 2 :=
by
  sorry

end cube_side_length_ratio_l1899_189947


namespace evaluate_polynomial_at_two_l1899_189972

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem evaluate_polynomial_at_two : f 2 = 41 := by
  sorry

end evaluate_polynomial_at_two_l1899_189972


namespace solution_set_of_inequality_l1899_189963

theorem solution_set_of_inequality :
  {x : ℝ | x^2 * (x - 4) ≥ 0} = {x : ℝ | x = 0 ∨ x ≥ 4} :=
by
  sorry

end solution_set_of_inequality_l1899_189963


namespace total_cost_correct_l1899_189948

def cost_of_cat_toy := 10.22
def cost_of_cage := 11.73
def cost_of_cat_food := 7.50
def cost_of_leash := 5.15
def cost_of_cat_treats := 3.98

theorem total_cost_correct : 
  cost_of_cat_toy + cost_of_cage + cost_of_cat_food + cost_of_leash + cost_of_cat_treats = 38.58 := 
by
  sorry

end total_cost_correct_l1899_189948


namespace distance_between_foci_l1899_189976

theorem distance_between_foci (a b : ℝ) (h₁ : a^2 = 18) (h₂ : b^2 = 2) :
  2 * (Real.sqrt (a^2 + b^2)) = 4 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_l1899_189976


namespace tom_initial_books_l1899_189952

theorem tom_initial_books (B : ℕ) (h1 : B - 4 + 38 = 39) : B = 5 :=
by
  sorry

end tom_initial_books_l1899_189952


namespace find_m_l1899_189922

-- Define the conditions with variables a, b, and m.
variable (a b m : ℝ)
variable (ha : 2^a = m)
variable (hb : 5^b = m)
variable (hc : 1/a + 1/b = 2)

-- Define the statement to be proven.
theorem find_m : m = Real.sqrt 10 :=
by
  sorry


end find_m_l1899_189922


namespace second_largest_div_second_smallest_l1899_189933

theorem second_largest_div_second_smallest : 
  let a := 10
  let b := 11
  let c := 12
  ∃ second_smallest second_largest, 
    second_smallest = b ∧ second_largest = b ∧ second_largest / second_smallest = 1 := 
by
  let a := 10
  let b := 11
  let c := 12
  use b
  use b
  exact ⟨rfl, rfl, rfl⟩

end second_largest_div_second_smallest_l1899_189933


namespace arithmetic_seq_a4_l1899_189996

theorem arithmetic_seq_a4 (a : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) : 
  a 4 = 8 :=
by
  sorry

end arithmetic_seq_a4_l1899_189996


namespace sqrt_diff_approx_l1899_189973

noncomputable def x : ℝ := Real.sqrt 50 - Real.sqrt 48

theorem sqrt_diff_approx : abs (x - 0.14) < 0.01 :=
by
  sorry

end sqrt_diff_approx_l1899_189973
