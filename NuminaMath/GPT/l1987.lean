import Mathlib

namespace tyler_meal_choices_l1987_198766

-- Define the total number of different meals Tyler can choose given the conditions.
theorem tyler_meal_choices : 
    (3 * (Nat.choose 5 3) * 4 * 4 = 480) := 
by
    -- Using the built-in combination function and the fact that meat, dessert, and drink choices are directly multiplied.
    sorry

end tyler_meal_choices_l1987_198766


namespace draw_probability_l1987_198724

theorem draw_probability (P_A_win : ℝ) (P_A_not_lose : ℝ) (h1 : P_A_win = 0.3) (h2 : P_A_not_lose = 0.8) : 
  ∃ P_draw : ℝ, P_draw = 0.5 := 
by
  sorry

end draw_probability_l1987_198724


namespace sample_size_l1987_198772

theorem sample_size (k n : ℕ) (h_ratio : 4 * k + k + 5 * k = n) 
  (h_middle_aged : 10 * (4 + 1 + 5) = n) : n = 100 := 
by
  sorry

end sample_size_l1987_198772


namespace edward_initial_money_l1987_198704

theorem edward_initial_money (initial_cost_books : ℝ) (discount_percent : ℝ) (num_pens : ℕ) 
  (cost_per_pen : ℝ) (money_left : ℝ) : 
  initial_cost_books = 40 → discount_percent = 0.25 → num_pens = 3 → cost_per_pen = 2 → money_left = 6 → 
  (initial_cost_books * (1 - discount_percent) + num_pens * cost_per_pen + money_left) = 42 :=
by
  sorry

end edward_initial_money_l1987_198704


namespace min_xy_sum_is_7_l1987_198713

noncomputable def min_xy_sum (x y : ℝ) : ℝ := 
x + y

theorem min_xy_sum_is_7 (x y : ℝ) (h1 : x > 1) (h2 : y > 2) (h3 : (x - 1) * (y - 2) = 4) : 
  min_xy_sum x y = 7 := by 
  sorry

end min_xy_sum_is_7_l1987_198713


namespace sam_letters_on_wednesday_l1987_198717

/-- Sam's average letters per day. -/
def average_letters_per_day : ℕ := 5

/-- Number of days Sam wrote letters. -/
def number_of_days : ℕ := 2

/-- Letters Sam wrote on Tuesday. -/
def letters_on_tuesday : ℕ := 7

/-- Total letters Sam wrote in two days. -/
def total_letters : ℕ := average_letters_per_day * number_of_days

/-- Letters Sam wrote on Wednesday. -/
def letters_on_wednesday : ℕ := total_letters - letters_on_tuesday

theorem sam_letters_on_wednesday : letters_on_wednesday = 3 :=
by
  -- placeholder proof
  sorry

end sam_letters_on_wednesday_l1987_198717


namespace sin_beta_equals_sqrt3_div_2_l1987_198757

noncomputable def angles_acute (α β : ℝ) : Prop :=
0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2

theorem sin_beta_equals_sqrt3_div_2 
  (α β : ℝ) 
  (h_acute: angles_acute α β) 
  (h_sin_alpha: Real.sin α = (4/7) * Real.sqrt 3) 
  (h_cos_alpha_plus_beta: Real.cos (α + β) = -(11/14)) 
  : Real.sin β = (Real.sqrt 3) / 2 :=
sorry

end sin_beta_equals_sqrt3_div_2_l1987_198757


namespace floor_x_plus_x_eq_13_div_3_l1987_198723

-- Statement representing the mathematical problem
theorem floor_x_plus_x_eq_13_div_3 (x : ℚ) (h : ⌊x⌋ + x = 13/3) : x = 7/3 := 
sorry

end floor_x_plus_x_eq_13_div_3_l1987_198723


namespace polynomial_divisibility_l1987_198700

theorem polynomial_divisibility (n : ℕ) : (¬ n % 3 = 0) → (x ^ (2 * n) + x ^ n + 1) % (x ^ 2 + x + 1) = 0 :=
by
  sorry

end polynomial_divisibility_l1987_198700


namespace average_water_drunk_l1987_198705

theorem average_water_drunk (d1 d2 d3 : ℕ) (h1 : d1 = 215) (h2 : d2 = d1 + 76) (h3 : d3 = d2 - 53) :
  (d1 + d2 + d3) / 3 = 248 :=
by
  -- placeholder for actual proof
  sorry

end average_water_drunk_l1987_198705


namespace distance_from_point_to_line_l1987_198714

-- Definition of the conditions
def point := (3, 0)
def line_y := 1

-- Problem statement: Prove that the distance between the point (3,0) and the line y=1 is 1.
theorem distance_from_point_to_line (point : ℝ × ℝ) (line_y : ℝ) : abs (point.snd - line_y) = 1 :=
by
  -- insert proof here
  sorry

end distance_from_point_to_line_l1987_198714


namespace parallel_vectors_xy_l1987_198722

theorem parallel_vectors_xy {x y : ℝ} (h : ∃ k : ℝ, (1, y, -3) = (k * x, k * (-2), k * 5)) : x * y = -2 :=
by sorry

end parallel_vectors_xy_l1987_198722


namespace hilary_ears_per_stalk_l1987_198775

-- Define the given conditions
def num_stalks : ℕ := 108
def kernels_per_ear_half1 : ℕ := 500
def kernels_per_ear_half2 : ℕ := 600
def total_kernels_to_shuck : ℕ := 237600

-- Define the number of ears of corn per stalk as the variable to prove
def ears_of_corn_per_stalk : ℕ := 4

-- The proof problem statement
theorem hilary_ears_per_stalk :
  (54 * ears_of_corn_per_stalk * kernels_per_ear_half1) + (54 * ears_of_corn_per_stalk * kernels_per_ear_half2) = total_kernels_to_shuck :=
by
  sorry

end hilary_ears_per_stalk_l1987_198775


namespace pairs_of_integers_solution_l1987_198737

-- Define the main theorem
theorem pairs_of_integers_solution :
  ∃ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ∧ 
               ((x = 63 ∧ y = 58) ∨
               (x = -63 ∧ y = -58) ∨
               (x = 459 ∧ y = 58) ∨
               (x = -459 ∧ y = -58)) :=
by
  sorry

end pairs_of_integers_solution_l1987_198737


namespace distribution_ways_5_to_3_l1987_198706

noncomputable def num_ways (n m : ℕ) : ℕ :=
  m ^ n

theorem distribution_ways_5_to_3 : num_ways 5 3 = 243 := by
  sorry

end distribution_ways_5_to_3_l1987_198706


namespace jack_total_cost_l1987_198712

def cost_of_tires (n : ℕ) (price_per_tire : ℕ) : ℕ := n * price_per_tire
def cost_of_window (price_per_window : ℕ) : ℕ := price_per_window

theorem jack_total_cost :
  cost_of_tires 3 250 + cost_of_window 700 = 1450 :=
by
  sorry

end jack_total_cost_l1987_198712


namespace arithmetic_sequence_a8_l1987_198770

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 90) :
  a 8 = 6 :=
by
  sorry

end arithmetic_sequence_a8_l1987_198770


namespace correct_answer_A_correct_answer_C_correct_answer_D_l1987_198725

variable (f g : ℝ → ℝ)

namespace ProofProblem

-- Assume the given conditions
axiom f_eq : ∀ x, f x = 6 - deriv g x
axiom f_compl : ∀ x, f (1 - x) = 6 + deriv g (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- Proving the correct answers
theorem correct_answer_A : g 0 = 2 :=
sorry

theorem correct_answer_C : ∀ x, g (x + 4) = g x :=
sorry

theorem correct_answer_D : f 1 * g 1 + f 3 * g 3 = 24 :=
sorry

end ProofProblem

end correct_answer_A_correct_answer_C_correct_answer_D_l1987_198725


namespace total_loads_l1987_198787

def shirts_per_load := 3
def sweaters_per_load := 2
def socks_per_load := 4

def white_shirts := 9
def colored_shirts := 12
def white_sweaters := 18
def colored_sweaters := 20
def white_socks := 16
def colored_socks := 24

def white_shirt_loads : ℕ := white_shirts / shirts_per_load
def white_sweater_loads : ℕ := white_sweaters / sweaters_per_load
def white_sock_loads : ℕ := white_socks / socks_per_load

def colored_shirt_loads : ℕ := colored_shirts / shirts_per_load
def colored_sweater_loads : ℕ := colored_sweaters / sweaters_per_load
def colored_sock_loads : ℕ := colored_socks / socks_per_load

def max_white_loads := max (max white_shirt_loads white_sweater_loads) white_sock_loads
def max_colored_loads := max (max colored_shirt_loads colored_sweater_loads) colored_sock_loads

theorem total_loads : max_white_loads + max_colored_loads = 19 := by
  sorry

end total_loads_l1987_198787


namespace roots_identity_l1987_198703

theorem roots_identity (p q r : ℝ) (h₁ : p + q + r = 15) (h₂ : p * q + q * r + r * p = 25) (h₃ : p * q * r = 10) :
  (1 + p) * (1 + q) * (1 + r) = 51 :=
by sorry

end roots_identity_l1987_198703


namespace probability_sum_15_l1987_198729

/-- If three standard 6-faced dice are rolled, the probability that the sum of the face-up integers is 15 is 5/72. -/
theorem probability_sum_15 : (1 / 6 : ℚ) ^ 3 * 3 + (1 / 6 : ℚ) ^ 3 * 6 = 5 / 72 := by 
  sorry

end probability_sum_15_l1987_198729


namespace solution_set_of_inequality_l1987_198781

theorem solution_set_of_inequality (x : ℝ) : (x^2 - |x| > 0) ↔ (x < -1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l1987_198781


namespace circle_equation_a_value_l1987_198715

theorem circle_equation_a_value (a : ℝ) : (∀ x y : ℝ, x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
sorry

end circle_equation_a_value_l1987_198715


namespace find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l1987_198727

-- Define the coordinate functions for point P
def coord_x (m : ℚ) : ℚ := 3 * m + 6
def coord_y (m : ℚ) : ℚ := m - 3

-- Definitions for each condition
def condition1 (m : ℚ) : Prop := coord_x m = coord_y m
def condition2 (m : ℚ) : Prop := coord_y m = coord_x m + 5
def condition3 (m : ℚ) : Prop := coord_x m = 3

-- Proof statements for the coordinates based on each condition
theorem find_coordinates_condition1 : 
  ∃ m, condition1 m ∧ coord_x m = -7.5 ∧ coord_y m = -7.5 :=
by sorry

theorem find_coordinates_condition2 : 
  ∃ m, condition2 m ∧ coord_x m = -15 ∧ coord_y m = -10 :=
by sorry

theorem find_coordinates_condition3 : 
  ∃ m, condition3 m ∧ coord_x m = 3 ∧ coord_y m = -4 :=
by sorry

end find_coordinates_condition1_find_coordinates_condition2_find_coordinates_condition3_l1987_198727


namespace find_n_l1987_198754

noncomputable def e : ℝ := Real.exp 1

-- lean cannot compute non-trivial transcendental solutions, this would need numerical methods
theorem find_n (n : ℝ) (x : ℝ) (y : ℝ) (h1 : x = 3) (h2 : y = 27) :
  Real.log n ^ (n / (2 * Real.sqrt (Real.pi + x))) = y :=
by
  rw [h1, h2]
  sorry

end find_n_l1987_198754


namespace triangle_area_l1987_198762

variables {A B C a b c : ℝ}

/-- In triangle ABC, the sides opposite to angles A, B, and C are denoted as a, b, and c, respectively.
It is given that b * sin C + c * sin B = 4 * a * sin B * sin C and b^2 + c^2 - a^2 = 8.
Prove that the area of triangle ABC is 4 * sqrt 3 / 3. -/
theorem triangle_area (h1 : b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h2 : b^2 + c^2 - a^2 = 8) :
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l1987_198762


namespace worksheets_already_graded_eq_5_l1987_198732

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def remaining_problems : ℕ := 16

def total_problems := total_worksheets * problems_per_worksheet
def graded_problems := total_problems - remaining_problems
def graded_worksheets := graded_problems / problems_per_worksheet

theorem worksheets_already_graded_eq_5 :
  graded_worksheets = 5 :=
by 
  sorry

end worksheets_already_graded_eq_5_l1987_198732


namespace lcm_24_36_42_l1987_198711

-- Definitions of the numbers involved
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 42

-- Statement for the lowest common multiple
theorem lcm_24_36_42 : Nat.lcm (Nat.lcm a b) c = 504 :=
by
  -- The proof will be filled in here
  sorry

end lcm_24_36_42_l1987_198711


namespace smallest_rat_num_l1987_198776

theorem smallest_rat_num (a b c d : ℚ) (ha : a = -6 / 7) (hb : b = 2) (hc : c = 0) (hd : d = -1) :
  min (min a (min b c)) d = -1 :=
sorry

end smallest_rat_num_l1987_198776


namespace kelly_games_left_l1987_198793

-- Definitions based on conditions
def original_games := 80
def additional_games := 31
def games_to_give_away := 105

-- Total games after finding more games
def total_games := original_games + additional_games

-- Number of games left after giving away
def games_left := total_games - games_to_give_away

-- Theorem statement
theorem kelly_games_left : games_left = 6 :=
by
  -- The proof will be here
  sorry

end kelly_games_left_l1987_198793


namespace sum_of_fractions_l1987_198728

theorem sum_of_fractions : 
  (2 / 5 : ℚ) + (4 / 50 : ℚ) + (3 / 500 : ℚ) + (8 / 5000 : ℚ) = 4876 / 10000 :=
by
  -- The proof can be completed by converting fractions and summing them accurately.
  sorry

end sum_of_fractions_l1987_198728


namespace range_of_e_l1987_198743

theorem range_of_e (a b c d e : ℝ) (h₁ : a + b + c + d + e = 8) (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l1987_198743


namespace find_coefficient_of_x_l1987_198747

theorem find_coefficient_of_x :
  ∃ a : ℚ, ∀ (x y : ℚ),
  (x + y = 19) ∧ (x + 3 * y = 1) ∧ (2 * x + y = 5) →
  (a * x + y = 19) ∧ (a = 7) :=
by
  sorry

end find_coefficient_of_x_l1987_198747


namespace find_f7_l1987_198750

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f (x)

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = 2 * x^2

theorem find_f7 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : periodic_function f 4)
  (h3 : specific_values f) :
  f 7 = -2 :=
by
  sorry

end find_f7_l1987_198750


namespace proof_problem_l1987_198748

noncomputable def aₙ (a₁ d : ℝ) (n : ℕ) := a₁ + (n - 1) * d
noncomputable def Sₙ (a₁ d : ℝ) (n : ℕ) := n * a₁ + (n * (n - 1) / 2) * d

def given_conditions (Sₙ : ℕ → ℝ) : Prop :=
  Sₙ 10 = 0 ∧ Sₙ 15 = 25

theorem proof_problem (a₁ d : ℝ) (Sₙ : ℕ → ℝ)
  (h₁ : Sₙ 10 = 0) (h₂ : Sₙ 15 = 25) :
  (aₙ a₁ d 5 = -1/3) ∧
  (∀ n, Sₙ n = (1 / 3) * (n ^ 2 - 10 * n) → n = 5) ∧
  (∀ n, n * Sₙ n = (n ^ 3 / 3) - (10 * n ^ 2 / 3) → min (n * Sₙ n) = -49) ∧
  (¬ ∃ n, (Sₙ n / n) > 0) :=
sorry

end proof_problem_l1987_198748


namespace Diane_age_l1987_198755

variable (C D E : ℝ)

def Carla_age_is_four_times_Diane_age : Prop := C = 4 * D
def Emma_is_eight_years_older_than_Diane : Prop := E = D + 8
def Carla_and_Emma_are_twins : Prop := C = E

theorem Diane_age : Carla_age_is_four_times_Diane_age C D → 
                    Emma_is_eight_years_older_than_Diane D E → 
                    Carla_and_Emma_are_twins C E → 
                    D = 8 / 3 :=
by
  intros hC hE hTwins
  have h1 : C = 4 * D := hC
  have h2 : E = D + 8 := hE
  have h3 : C = E := hTwins
  sorry

end Diane_age_l1987_198755


namespace tangent_line_properties_l1987_198790

noncomputable def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem tangent_line_properties (a b : ℝ) :
  (∀ x : ℝ, curve 0 a b = b) →
  (∀ x : ℝ, x - (curve x a b - b) + 1 = 0 → (∀ x : ℝ, 2*0 + a = 1)) →
  a + b = 2 :=
by
  intros h_curve h_tangent
  have h_b : b = 1 := by sorry
  have h_a : a = 1 := by sorry
  rw [h_a, h_b]
  norm_num

end tangent_line_properties_l1987_198790


namespace inequalities_always_true_l1987_198701

theorem inequalities_always_true (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) 
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ (x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b) := by
  sorry

end inequalities_always_true_l1987_198701


namespace cos_double_angle_of_tangent_is_2_l1987_198716

theorem cos_double_angle_of_tangent_is_2
  (θ : ℝ)
  (h_tan : Real.tan θ = 2) :
  Real.cos (2 * θ) = -3 / 5 := 
by
  sorry

end cos_double_angle_of_tangent_is_2_l1987_198716


namespace molecular_weight_of_compound_l1987_198730

theorem molecular_weight_of_compound :
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  molecular_weight = 156.22615 :=
by
  -- conditions
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  -- prove statement
  have h1 : average_atomic_weight_c = 12.05015 := by sorry
  have h2 : molecular_weight = 156.22615 := by sorry
  exact h2

end molecular_weight_of_compound_l1987_198730


namespace numerical_form_463001_l1987_198765

theorem numerical_form_463001 : 463001 = 463001 := by
  rfl

end numerical_form_463001_l1987_198765


namespace real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l1987_198764

-- Define z as a complex number with components dependent on m
def z (m : ℝ) : ℂ := ⟨m^2 - m, m - 1⟩

-- Statement 1: z is real iff m = 1
theorem real_z_iff (m : ℝ) : (∃ r : ℝ, z m = ⟨r, 0⟩) ↔ m = 1 := 
    sorry

-- Statement 2: z is purely imaginary iff m = 0
theorem imaginary_z_iff (m : ℝ) : (∃ i : ℝ, z m = ⟨0, i⟩ ∧ i ≠ 0) ↔ m = 0 := 
    sorry

-- Statement 3: z is in the first quadrant iff m > 1
theorem first_quadrant_z_iff (m : ℝ) : (z m).re > 0 ∧ (z m).im > 0 ↔ m > 1 := 
    sorry

end real_z_iff_imaginary_z_iff_first_quadrant_z_iff_l1987_198764


namespace quadratic_roots_p_eq_l1987_198744

theorem quadratic_roots_p_eq (b c p q r s : ℝ)
  (h1 : r + s = -b)
  (h2 : r * s = c)
  (h3 : r^2 + s^2 = -p)
  (h4 : r^2 * s^2 = q):
  p = 2 * c - b^2 :=
by sorry

end quadratic_roots_p_eq_l1987_198744


namespace calculate_expression_l1987_198784

theorem calculate_expression : 
  (π - 3.14) ^ 0 - 8 ^ (2 / 3) + (1 / 5) ^ 2 * (Real.logb 2 32) + 5 ^ (Real.logb 5 3) = 1 / 5 :=
by
  sorry

end calculate_expression_l1987_198784


namespace cos_double_angle_l1987_198740

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2 * n) = 12) : Real.cos (2 * θ) = 5 / 6 := 
sorry

end cos_double_angle_l1987_198740


namespace range_of_m_l1987_198796

theorem range_of_m (m : ℝ) : (∀ x, 0 ≤ x ∧ x ≤ m → -6 ≤ x^2 - 4 * x - 2 ∧ x^2 - 4 * x - 2 ≤ -2) → 2 ≤ m ∧ m ≤ 4 :=
by
  sorry

end range_of_m_l1987_198796


namespace max_sum_a_b_l1987_198786

theorem max_sum_a_b (a b : ℝ) (ha : 4 * a + 3 * b ≤ 10) (hb : 3 * a + 6 * b ≤ 12) : a + b ≤ 22 / 7 :=
sorry

end max_sum_a_b_l1987_198786


namespace average_male_students_score_l1987_198783

def average_male_score (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ) : ℕ :=
  let total_sum := (male_count + female_count) * total_avg
  let female_sum := female_count * female_avg
  let male_sum := total_sum - female_sum
  male_sum / male_count

theorem average_male_students_score
  (total_avg : ℕ) (female_avg : ℕ) (male_count : ℕ) (female_count : ℕ)
  (h1 : total_avg = 90) (h2 : female_avg = 92) (h3 : male_count = 8) (h4 : female_count = 20) :
  average_male_score total_avg female_avg male_count female_count = 85 :=
by {
  sorry
}

end average_male_students_score_l1987_198783


namespace molecular_weight_of_NH4Br_l1987_198726

def atomic_weight (element : String) : Real :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | _ => 0.0

def molecular_weight (composition : List (String × Nat)) : Real :=
  composition.foldl (λ acc (elem, count) => acc + count * atomic_weight elem) 0

theorem molecular_weight_of_NH4Br :
  molecular_weight [("N", 1), ("H", 4), ("Br", 1)] = 97.95 :=
by
  sorry

end molecular_weight_of_NH4Br_l1987_198726


namespace even_composite_fraction_l1987_198769

theorem even_composite_fraction : 
  ((4 * 6 * 8 * 10 * 12) : ℚ) / (14 * 16 * 18 * 20 * 22) = 1 / 42 :=
by 
  sorry

end even_composite_fraction_l1987_198769


namespace plywood_problem_exists_squares_l1987_198733

theorem plywood_problem_exists_squares :
  ∃ (a b : ℕ), a^2 + b^2 = 625 ∧ a ≠ 20 ∧ b ≠ 20 ∧ a ≠ 15 ∧ b ≠ 15 := by
  sorry

end plywood_problem_exists_squares_l1987_198733


namespace total_ladybugs_l1987_198739

theorem total_ladybugs (leaves : Nat) (ladybugs_per_leaf : Nat) (total_ladybugs : Nat) : 
  leaves = 84 → 
  ladybugs_per_leaf = 139 → 
  total_ladybugs = leaves * ladybugs_per_leaf → 
  total_ladybugs = 11676 := by
  intros h1 h2 h3
  rw [h1, h2] at h3
  assumption

end total_ladybugs_l1987_198739


namespace bilion_wins_1000000_dollars_l1987_198741

theorem bilion_wins_1000000_dollars :
  ∃ (p : ℕ), (p = 1000000) ∧ (p % 3 = 1) → p = 1000000 :=
by
  sorry

end bilion_wins_1000000_dollars_l1987_198741


namespace problem1_problem2_problem3_problem4_l1987_198761

-- Problem 1: Prove (1 * -6) + -13 = -19
theorem problem1 : (1 * -6) + -13 = -19 := by 
  sorry

-- Problem 2: Prove (3/5) + (-3/4) = -3/20
theorem problem2 : (3/5 : ℚ) + (-3/4) = -3/20 := by 
  sorry

-- Problem 3: Prove 4.7 + (-0.8) + 5.3 + (-8.2) = 1
theorem problem3 : (4.7 + (-0.8) + 5.3 + (-8.2) : ℝ) = 1 := by 
  sorry

-- Problem 4: Prove (-1/6) + (1/3) + (-1/12) = 1/12
theorem problem4 : (-1/6 : ℚ) + (1/3) + (-1/12) = 1/12 := by 
  sorry

end problem1_problem2_problem3_problem4_l1987_198761


namespace hiker_distance_l1987_198742

noncomputable def distance_from_start (north south east west : ℕ) : ℝ :=
  let north_south := north - south
  let east_west := east - west
  Real.sqrt (north_south ^ 2 + east_west ^ 2)

theorem hiker_distance :
  distance_from_start 24 8 15 9 = 2 * Real.sqrt 73 := by
  sorry

end hiker_distance_l1987_198742


namespace distance_of_third_point_on_trip_l1987_198785

theorem distance_of_third_point_on_trip (D : ℝ) (h1 : D + 2 * D + (1/2) * D + 7 * D = 560) :
  (1/2) * D = 27 :=
by
  sorry

end distance_of_third_point_on_trip_l1987_198785


namespace yang_tricks_modulo_l1987_198753

noncomputable def number_of_tricks_result : Nat :=
  let N := 20000
  let modulo := 100000
  N % modulo

theorem yang_tricks_modulo :
  number_of_tricks_result = 20000 :=
by
  sorry

end yang_tricks_modulo_l1987_198753


namespace euler_school_voting_problem_l1987_198788

theorem euler_school_voting_problem :
  let U := 198
  let A := 149
  let B := 119
  let AcBc := 29
  U - AcBc = 169 → 
  A + B - (U - AcBc) = 99 :=
by
  intros h₁
  sorry

end euler_school_voting_problem_l1987_198788


namespace island_inhabitants_even_l1987_198752

theorem island_inhabitants_even 
  (total : ℕ) 
  (knights liars : ℕ)
  (H : total = knights + liars)
  (H1 : ∃ (knk : Prop), (knk → (knights % 2 = 0)) ∧ (¬knk → (knights % 2 = 1)))
  (H2 : ∃ (lkr : Prop), (lkr → (liars % 2 = 1)) ∧ (¬lkr → (liars % 2 = 0)))
  : (total % 2 = 0) := sorry

end island_inhabitants_even_l1987_198752


namespace fraction_of_work_left_l1987_198789

theorem fraction_of_work_left (a_days b_days : ℕ) (together_days : ℕ) 
    (h_a : a_days = 15) (h_b : b_days = 20) (h_together : together_days = 4) : 
    (1 - together_days * ((1/a_days : ℚ) + (1/b_days))) = 8/15 := by
  sorry

end fraction_of_work_left_l1987_198789


namespace num_valid_pairs_l1987_198708

theorem num_valid_pairs (a b : ℕ) (hb : b > a) (h_unpainted_area : ab = 3 * (a - 4) * (b - 4)) :
  (∃ (a b : ℕ), b > a ∧ ab = 3 * (a-4) * (b-4) ∧ (a-6) * (b-6) = 12 ∧ ((a, b) = (7, 18) ∨ (a, b) = (8, 12))) ∧
  (2 = 2) :=
by sorry

end num_valid_pairs_l1987_198708


namespace total_snakes_l1987_198731

def People (n : ℕ) : Prop := n = 59
def OnlyDogs (n : ℕ) : Prop := n = 15
def OnlyCats (n : ℕ) : Prop := n = 10
def OnlyCatsAndDogs (n : ℕ) : Prop := n = 5
def CatsDogsSnakes (n : ℕ) : Prop := n = 3

theorem total_snakes (n_people n_dogs n_cats n_catsdogs n_catdogsnsnakes : ℕ)
  (h_people : People n_people) 
  (h_onlyDogs : OnlyDogs n_dogs)
  (h_onlyCats : OnlyCats n_cats)
  (h_onlyCatsAndDogs : OnlyCatsAndDogs n_catsdogs)
  (h_catsDogsSnakes : CatsDogsSnakes n_catdogsnsnakes) :
  n_catdogsnsnakes >= 3 :=
by
  -- Proof goes here
  sorry

end total_snakes_l1987_198731


namespace initial_amount_l1987_198782

theorem initial_amount (X : ℚ) (F : ℚ) :
  (∀ (X F : ℚ), F = X * (3/4)^3 → F = 37 → X = 37 * 64 / 27) :=
by
  sorry

end initial_amount_l1987_198782


namespace weight_of_five_bowling_balls_l1987_198791

theorem weight_of_five_bowling_balls (b c : ℕ) (hb : 9 * b = 4 * c) (hc : c = 36) : 5 * b = 80 := by
  sorry

end weight_of_five_bowling_balls_l1987_198791


namespace total_jokes_sum_l1987_198763

theorem total_jokes_sum :
  let jessy_week1 := 11
  let alan_week1 := 7
  let tom_week1 := 5
  let emily_week1 := 3
  let jessy_week4 := 11 * 3 ^ 3
  let alan_week4 := 7 * 2 ^ 3
  let tom_week4 := 5 * 4 ^ 3
  let emily_week4 := 3 * 4 ^ 3
  let jessy_total := 11 + 11 * 3 + 11 * 3 ^ 2 + jessy_week4
  let alan_total := 7 + 7 * 2 + 7 * 2 ^ 2 + alan_week4
  let tom_total := 5 + 5 * 4 + 5 * 4 ^ 2 + tom_week4
  let emily_total := 3 + 3 * 4 + 3 * 4 ^ 2 + emily_week4
  jessy_total + alan_total + tom_total + emily_total = 1225 :=
by 
  sorry

end total_jokes_sum_l1987_198763


namespace no_two_digit_prime_with_digit_sum_9_l1987_198795

-- Define the concept of a two-digit number
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Define the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Define the concept of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem statement
theorem no_two_digit_prime_with_digit_sum_9 :
  ∀ n : ℕ, is_two_digit n ∧ digit_sum n = 9 → ¬is_prime n :=
by {
  -- proof omitted
  sorry
}  

end no_two_digit_prime_with_digit_sum_9_l1987_198795


namespace algebraic_expression_value_l1987_198719

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m * n = 3) 
  (h2 : n = m + 1) : 
  (m - n) ^ 2 * ((1 / n) - (1 / m)) = -1 / 3 :=
by sorry

end algebraic_expression_value_l1987_198719


namespace determine_m_value_l1987_198710

-- Define the condition that the roots of the quadratic are given
def quadratic_equation_has_given_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, (8 * x^2 + 4 * x + m = 0) → (x = (-2 + (Complex.I * Real.sqrt 88)) / 8) ∨ (x = (-2 - (Complex.I * Real.sqrt 88)) / 8)

-- The main statement to be proven
theorem determine_m_value (m : ℝ) (h : quadratic_equation_has_given_roots m) : m = 13 / 4 :=
sorry

end determine_m_value_l1987_198710


namespace sum_of_numbers_l1987_198702

theorem sum_of_numbers (a b c : ℕ) (h_order: a ≤ b ∧ b ≤ c) (h_median: b = 10) 
    (h_mean_least: (a + b + c) / 3 = a + 15) (h_mean_greatest: (a + b + c) / 3 = c - 20) :
    a + b + c = 45 :=
  by
  sorry

end sum_of_numbers_l1987_198702


namespace minimum_Q_l1987_198751

def is_special (m : ℕ) : Prop :=
  let d1 := m / 10 
  let d2 := m % 10
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d2 ≠ 0

def F (m : ℕ) : ℤ :=
  let d1 := m / 10
  let d2 := m % 10
  (d1 * 100 + d2 * 10 + d1) - (d2 * 100 + d1 * 10 + d2) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s) / s

variables (a b x y : ℕ)
variables (h1 : 1 ≤ b ∧ b < a ∧ a ≤ 7)
variables (h2 : 1 ≤ x ∧ x ≤ 8)
variables (h3 : 1 ≤ y ∧ y ≤ 8)
variables (hs_is_special : is_special (10 * a + b))
variables (ht_is_special : is_special (10 * x + y))
variables (s := 10 * a + b)
variables (t := 10 * x + y)
variables (h4 : (F s % 5) = 1)
variables (h5 : F t - F s + 18 * x = 36)

theorem minimum_Q : Q s t = -42 / 73 := sorry

end minimum_Q_l1987_198751


namespace volume_ratio_l1987_198720

-- Define the edge lengths
def edge_length_cube1 : ℝ := 4 -- in inches
def edge_length_cube2 : ℝ := 2 * 12 -- 2 feet converted to inches

-- Define the volumes
def volume_cube (a : ℝ) : ℝ := a ^ 3

-- Statement asserting the ratio of the volumes is 1/216
theorem volume_ratio : volume_cube edge_length_cube1 / volume_cube edge_length_cube2 = 1 / 216 :=
by
  -- This is the placeholder to skip the proof
  sorry

end volume_ratio_l1987_198720


namespace one_thirds_in_fraction_l1987_198768

theorem one_thirds_in_fraction : (9 / 5) / (1 / 3) = 27 / 5 := by
  sorry

end one_thirds_in_fraction_l1987_198768


namespace inequality_conditions_l1987_198797

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2 * (A * B + B * C + C * A)) :=
by
  sorry

end inequality_conditions_l1987_198797


namespace similar_triangles_height_l1987_198798

theorem similar_triangles_height (h₁ h₂ : ℝ) (a₁ a₂ : ℝ) 
  (ratio_area : a₁ / a₂ = 1 / 9) (height_small : h₁ = 4) :
  h₂ = 12 :=
sorry

end similar_triangles_height_l1987_198798


namespace kenny_pieces_used_l1987_198758

-- Definitions based on conditions
def mushrooms_cut := 22
def pieces_per_mushroom := 4
def karla_pieces := 42
def remaining_pieces := 8
def total_pieces := mushrooms_cut * pieces_per_mushroom

-- Theorem to be proved
theorem kenny_pieces_used :
  total_pieces - (karla_pieces + remaining_pieces) = 38 := 
by 
  sorry

end kenny_pieces_used_l1987_198758


namespace father_son_fish_problem_l1987_198746

variables {F S x : ℕ}

theorem father_son_fish_problem (h1 : F - x = S + x) (h2 : F + x = 2 * (S - x)) : 
  (F - S) / S = 2 / 5 :=
by sorry

end father_son_fish_problem_l1987_198746


namespace hard_candy_food_colouring_l1987_198780

noncomputable def food_colouring_per_hard_candy (lollipop_use : ℕ) (gummy_use : ℕ)
    (lollipops_per_day : ℕ) (gummies_per_day : ℕ) (hard_candies_per_day : ℕ)
    (total_food_colouring : ℕ) : ℕ := 
by
  -- Let ml_lollipops be the total amount needed for lollipops
  let ml_lollipops := lollipop_use * lollipops_per_day
  -- Let ml_gummy be the total amount needed for gummy candies
  let ml_gummy := gummy_use * gummies_per_day
  -- Let ml_non_hard be the amount for lollipops and gummy candies combined
  let ml_non_hard := ml_lollipops + ml_gummy
  -- Let ml_hard be the amount used for hard candies alone
  let ml_hard := total_food_colouring - ml_non_hard
  -- Compute the food colouring used per hard candy
  exact ml_hard / hard_candies_per_day

theorem hard_candy_food_colouring :
  food_colouring_per_hard_candy 8 3 150 50 20 1950 = 30 :=
by
  unfold food_colouring_per_hard_candy
  sorry

end hard_candy_food_colouring_l1987_198780


namespace median_of_circumscribed_trapezoid_l1987_198760

theorem median_of_circumscribed_trapezoid (a b c d : ℝ) (h1 : a + b + c + d = 12) (h2 : a + b = c + d) : (a + b) / 2 = 3 :=
by
  sorry

end median_of_circumscribed_trapezoid_l1987_198760


namespace solve_numRedBalls_l1987_198745

-- Condition (1): There are a total of 10 balls in the bag
def totalBalls : ℕ := 10

-- Condition (2): The probability of drawing a black ball is 2/5
-- This means the number of black balls is 4
def numBlackBalls : ℕ := 4

-- Condition (3): The probability of drawing at least 1 white ball when drawing 2 balls is 7/9
def probAtLeastOneWhiteBall : ℚ := 7 / 9

-- The number of red balls in the bag is calculated based on the given conditions
def numRedBalls (totalBalls numBlackBalls : ℕ) (probAtLeastOneWhiteBall : ℚ) : ℕ := 
  let totalWhiteAndRedBalls := totalBalls - numBlackBalls
  let probTwoNonWhiteBalls := 1 - probAtLeastOneWhiteBall
  let comb (n k : ℕ) := Nat.choose n k
  let equation := comb totalWhiteAndRedBalls 2 * comb (totalBalls - 2) 0 / comb totalBalls 2
  if equation = probTwoNonWhiteBalls then totalWhiteAndRedBalls else 0

theorem solve_numRedBalls : numRedBalls totalBalls numBlackBalls probAtLeastOneWhiteBall = 1 := by
  sorry

end solve_numRedBalls_l1987_198745


namespace Xiaoyong_age_solution_l1987_198756

theorem Xiaoyong_age_solution :
  ∃ (x y : ℕ), 1 ≤ y ∧ y < x ∧ x < 20 ∧ 2 * x + 5 * y = 97 ∧ x = 16 ∧ y = 13 :=
by
  -- You should provide a suitable proof here
  sorry

end Xiaoyong_age_solution_l1987_198756


namespace positive_integer_solutions_l1987_198767

theorem positive_integer_solutions (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) :
  1 + 2^x + 2^(2*x+1) = y^n ↔ 
  (x = 4 ∧ y = 23 ∧ n = 2) ∨ (∃ t : ℕ, 0 < t ∧ x = t ∧ y = 1 + 2^t + 2^(2*t+1) ∧ n = 1) :=
sorry

end positive_integer_solutions_l1987_198767


namespace bruce_paid_amount_l1987_198792

noncomputable def total_amount_paid :=
  let grapes_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let strawberries_cost := 4 * 90
  let total_cost := grapes_cost + mangoes_cost + oranges_cost + strawberries_cost
  let discount := 0.10 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.05 * discounted_total
  let final_amount := discounted_total + tax
  final_amount

theorem bruce_paid_amount :
  total_amount_paid = 1526.18 :=
by
  sorry

end bruce_paid_amount_l1987_198792


namespace max_sum_marks_l1987_198721

theorem max_sum_marks (a b c : ℕ) (h1 : a + b + c = 2019) (h2 : a ≤ c + 2) : 
  2 * a + b ≤ 2021 :=
by {
  -- We'll skip the proof but formulate the statement following conditions strictly.
  sorry
}

end max_sum_marks_l1987_198721


namespace min_value_m_n_l1987_198779

theorem min_value_m_n (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 4)
    (m n : ℝ) (h_m : m = b + 1 / a) (h_n : n = a + 1 / b) : m + n ≥ 5 :=
by
  sorry

end min_value_m_n_l1987_198779


namespace simplify_fraction_l1987_198778

theorem simplify_fraction : (75 : ℚ) / (100 : ℚ) = (3 : ℚ) / (4 : ℚ) :=
by
  sorry

end simplify_fraction_l1987_198778


namespace problem_l1987_198777

variable (f : ℝ → ℝ)

-- Given condition
axiom h : ∀ x : ℝ, f (1 / x) = 1 / (x + 1)

-- Prove that f(2) = 2/3
theorem problem : f 2 = 2 / 3 :=
sorry

end problem_l1987_198777


namespace calories_per_slice_l1987_198718

theorem calories_per_slice
  (total_calories : ℕ)
  (portion_eaten : ℕ)
  (percentage_eaten : ℝ)
  (slices_in_cheesecake : ℕ)
  (calories_in_slice : ℕ) :
  total_calories = 2800 →
  percentage_eaten = 0.25 →
  portion_eaten = 2 →
  portion_eaten = percentage_eaten * slices_in_cheesecake →
  calories_in_slice = total_calories / slices_in_cheesecake →
  calories_in_slice = 350 :=
by
  intros
  sorry

end calories_per_slice_l1987_198718


namespace correct_average_l1987_198709

theorem correct_average (avg: ℕ) (n: ℕ) (incorrect: ℕ) (correct: ℕ) 
  (h_avg : avg = 16) (h_n : n = 10) (h_incorrect : incorrect = 25) (h_correct : correct = 35) :
  (avg * n + (correct - incorrect)) / n = 17 := 
by
  sorry

end correct_average_l1987_198709


namespace find_A_max_min_l1987_198773

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l1987_198773


namespace max_product_l1987_198734

-- Define the conditions
def sum_condition (x : ℤ) : Prop :=
  (x + (2024 - x) = 2024)

def product (x : ℤ) : ℤ :=
  (x * (2024 - x))

-- The statement to be proved
theorem max_product : ∃ x : ℤ, sum_condition x ∧ product x = 1024144 :=
by
  sorry

end max_product_l1987_198734


namespace hockey_league_num_games_l1987_198749

theorem hockey_league_num_games :
  ∃ (num_teams : ℕ) (num_times : ℕ), 
    num_teams = 16 ∧ num_times = 10 ∧ 
    (num_teams * (num_teams - 1) / 2) * num_times = 2400 := by
  sorry

end hockey_league_num_games_l1987_198749


namespace inv_seq_not_arith_seq_l1987_198774

theorem inv_seq_not_arith_seq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_arith : ∃ d : ℝ, d ≠ 0 ∧ b = a + d ∧ c = a + 2 * d) :
  ¬ ∃ d' : ℝ, ∀ i j k : ℝ, i = 1 / a → j = 1 / b → k = 1 / c → j - i = d' ∧ k - j = d' :=
sorry

end inv_seq_not_arith_seq_l1987_198774


namespace tonya_hamburgers_to_beat_winner_l1987_198794

-- Given conditions
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Calculate the number of hamburgers eaten last year
def hamburgers_eaten_last_year : ℕ := ounces_eaten_last_year / ounces_per_hamburger

-- Prove the number of hamburgers Tonya needs to eat to beat last year's winner
theorem tonya_hamburgers_to_beat_winner : 
  hamburgers_eaten_last_year + 1 = 22 :=
by
  -- It remains to be proven
  sorry

end tonya_hamburgers_to_beat_winner_l1987_198794


namespace points_on_same_side_of_line_l1987_198759

theorem points_on_same_side_of_line (m : ℝ) :
  (2 * 0 + 0 + m > 0 ∧ 2 * -1 + 1 + m > 0) ∨ 
  (2 * 0 + 0 + m < 0 ∧ 2 * -1 + 1 + m < 0) ↔ 
  (m < 0 ∨ m > 1) :=
by
  sorry

end points_on_same_side_of_line_l1987_198759


namespace totalNutsInCar_l1987_198771

-- Definitions based on the conditions
def busySquirrelNutsPerDay : Nat := 30
def busySquirrelDays : Nat := 35
def numberOfBusySquirrels : Nat := 2

def lazySquirrelNutsPerDay : Nat := 20
def lazySquirrelDays : Nat := 40
def numberOfLazySquirrels : Nat := 3

def sleepySquirrelNutsPerDay : Nat := 10
def sleepySquirrelDays : Nat := 45
def numberOfSleepySquirrels : Nat := 1

-- Calculate the total number of nuts stored by each type of squirrels
def totalNutsStoredByBusySquirrels : Nat := numberOfBusySquirrels * (busySquirrelNutsPerDay * busySquirrelDays)
def totalNutsStoredByLazySquirrels : Nat := numberOfLazySquirrels * (lazySquirrelNutsPerDay * lazySquirrelDays)
def totalNutsStoredBySleepySquirrel : Nat := numberOfSleepySquirrels * (sleepySquirrelNutsPerDay * sleepySquirrelDays)

-- The final theorem to prove
theorem totalNutsInCar : totalNutsStoredByBusySquirrels + totalNutsStoredByLazySquirrels + totalNutsStoredBySleepySquirrel = 4950 := by
  sorry

end totalNutsInCar_l1987_198771


namespace arithmetic_sequence_sum_10_l1987_198799

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_10 (a_1 a_3 a_7 a_9 : ℤ)
    (h1 : ∃ a_1, a_3 = a_1 - 4)
    (h2 : a_7 = a_1 - 12)
    (h3 : a_9 = a_1 - 16)
    (h4 : a_7 * a_7 = a_3 * a_9)
    : sum_of_first_n_terms a_1 (-2) 10 = 110 :=
by 
  sorry

end arithmetic_sequence_sum_10_l1987_198799


namespace question_statement_l1987_198735

def line := Type
def plane := Type

-- Definitions for line lying in plane and planes being parallel 
def isIn (a : line) (α : plane) : Prop := sorry
def isParallel (α β : plane) : Prop := sorry
def isParallelLinePlane (a : line) (β : plane) : Prop := sorry

-- Conditions 
variables (a b : line) (α β : plane) 
variable (distinct_lines : a ≠ b)
variable (distinct_planes : α ≠ β)

-- Main statement to prove
theorem question_statement (h_parallel_planes : isParallel α β) (h_line_in_plane : isIn a α) : isParallelLinePlane a β := 
sorry

end question_statement_l1987_198735


namespace Andrew_spent_1395_dollars_l1987_198736

-- Define the conditions
def cookies_per_day := 3
def cost_per_cookie := 15
def days_in_may := 31

-- Define the calculation
def total_spent := cookies_per_day * cost_per_cookie * days_in_may

-- State the theorem
theorem Andrew_spent_1395_dollars :
  total_spent = 1395 := 
by
  sorry

end Andrew_spent_1395_dollars_l1987_198736


namespace range_of_a_l1987_198738

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 5 * a) ↔ (4 ≤ a ∨ a ≤ 1) :=
by
  sorry

end range_of_a_l1987_198738


namespace range_of_a_plus_3b_l1987_198707

theorem range_of_a_plus_3b :
  ∀ (a b : ℝ),
    -1 ≤ a + b ∧ a + b ≤ 1 ∧ 1 ≤ a - 2 * b ∧ a - 2 * b ≤ 3 →
    -11 / 3 ≤ a + 3 * b ∧ a + 3 * b ≤ 7 / 3 :=
by
  sorry

end range_of_a_plus_3b_l1987_198707
