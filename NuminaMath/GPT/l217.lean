import Mathlib

namespace trigonometric_relationship_l217_21748

theorem trigonometric_relationship :
  let a := [10, 9, 8, 7, 6, 4, 3, 2, 1]
  let sum_of_a := a.sum
  let x := Real.sin sum_of_a
  let y := Real.cos sum_of_a
  let z := Real.tan sum_of_a
  sum_of_a = 50 →
  z < x ∧ x < y :=
by
  sorry

end trigonometric_relationship_l217_21748


namespace initial_professors_l217_21721

theorem initial_professors (p : ℕ) (h₁ : p ≥ 5) (h₂ : 6480 % p = 0) (h₃ : 11200 % (p + 3) = 0) : p = 5 :=
by
  sorry

end initial_professors_l217_21721


namespace total_children_on_playground_l217_21775

theorem total_children_on_playground
  (boys : ℕ) (girls : ℕ)
  (h_boys : boys = 44) (h_girls : girls = 53) :
  boys + girls = 97 :=
by 
  -- Proof omitted
  sorry

end total_children_on_playground_l217_21775


namespace negation_of_symmetry_about_y_eq_x_l217_21735

theorem negation_of_symmetry_about_y_eq_x :
  ¬ (∀ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x) ↔ ∃ f : ℝ → ℝ, ∃ x : ℝ, f (f x) ≠ x :=
by sorry

end negation_of_symmetry_about_y_eq_x_l217_21735


namespace cost_price_of_computer_table_l217_21707

theorem cost_price_of_computer_table (CP SP : ℝ) 
  (h1 : SP = CP * 1.15) 
  (h2 : SP = 5750) 
  : CP = 5000 := 
by 
  sorry

end cost_price_of_computer_table_l217_21707


namespace num_pos_int_x_l217_21772

theorem num_pos_int_x (x : ℕ) : 
  (30 < x^2 + 5 * x + 10) ∧ (x^2 + 5 * x + 10 < 60) ↔ x = 3 ∨ x = 4 ∨ x = 5 := 
sorry

end num_pos_int_x_l217_21772


namespace log9_6_eq_mn_over_2_l217_21776

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log9_6_eq_mn_over_2
  (m n : ℝ)
  (h1 : log_base 7 4 = m)
  (h2 : log_base 4 6 = n) : 
  log_base 9 6 = (m * n) / 2 := by
  sorry

end log9_6_eq_mn_over_2_l217_21776


namespace fixed_point_for_any_k_l217_21788

-- Define the function f representing our quadratic equation
def f (k : ℝ) (x : ℝ) : ℝ :=
  8 * x^2 + 3 * k * x - 5 * k
  
-- The statement representing our proof problem
theorem fixed_point_for_any_k :
  ∀ (a b : ℝ), (∀ (k : ℝ), f k a = b) → (a, b) = (5, 200) :=
by
  sorry

end fixed_point_for_any_k_l217_21788


namespace mathematician_daily_questions_l217_21716

/-- Given 518 questions for the first project and 476 for the second project,
if all questions are to be completed in 7 days, prove that the number
of questions completed each day is 142. -/
theorem mathematician_daily_questions (q1 q2 days questions_per_day : ℕ) 
  (h1 : q1 = 518) (h2 : q2 = 476) (h3 : days = 7) 
  (h4 : q1 + q2 = 994) (h5 : questions_per_day = 994 / 7) :
  questions_per_day = 142 :=
sorry

end mathematician_daily_questions_l217_21716


namespace Calvin_insect_count_l217_21768

theorem Calvin_insect_count:
  ∀ (roaches scorpions crickets caterpillars : ℕ), 
    roaches = 12 →
    scorpions = 3 →
    crickets = roaches / 2 →
    caterpillars = scorpions * 2 →
    roaches + scorpions + crickets + caterpillars = 27 := 
by
  intros roaches scorpions crickets caterpillars h_roaches h_scorpions h_crickets h_caterpillars
  rw [h_roaches, h_scorpions, h_crickets, h_caterpillars]
  norm_num
  sorry

end Calvin_insect_count_l217_21768


namespace derivative_of_f_l217_21779

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + (1 / 3) * ((3 * x - 1) / (3 * x^2 - 2 * x + 1))

theorem derivative_of_f :
  ∀ x : ℝ, deriv f x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
by intros; sorry

end derivative_of_f_l217_21779


namespace product_of_three_greater_than_product_of_two_or_four_l217_21794

theorem product_of_three_greater_than_product_of_two_or_four
  (nums : Fin 10 → ℝ)
  (h_positive : ∀ i, 0 < nums i)
  (h_distinct : Function.Injective nums) :
  ∃ (a b c : Fin 10),
    (∃ (d e : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (c ≠ d) ∧ (c ≠ e) ∧ nums a * nums b * nums c > nums d * nums e) ∨
    (∃ (d e f g : Fin 10), (a ≠ b) ∧ (a ≠ c) ∧ (b ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧ (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧ nums a * nums b * nums c > nums d * nums e * nums f * nums g) :=
sorry

end product_of_three_greater_than_product_of_two_or_four_l217_21794


namespace roots_of_quadratic_l217_21761

theorem roots_of_quadratic (x : ℝ) : (5 * x^2 = 4 * x) → (x = 0 ∨ x = 4 / 5) :=
by
  sorry

end roots_of_quadratic_l217_21761


namespace regions_formula_l217_21717

-- Define the number of regions R(n) created by n lines
def regions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

-- Theorem statement: for n lines, no two parallel, no three concurrent, the regions are defined by the formula
theorem regions_formula (n : ℕ) : regions n = 1 + (n * (n + 1)) / 2 := 
by sorry

end regions_formula_l217_21717


namespace range_of_x_l217_21767

variable (x y : ℝ)

theorem range_of_x (h1 : 2 * x - y = 4) (h2 : -2 < y ∧ y ≤ 3) :
  1 < x ∧ x ≤ 7 / 2 :=
  sorry

end range_of_x_l217_21767


namespace alloy_price_per_kg_l217_21724

theorem alloy_price_per_kg (cost_A cost_B ratio_A_B total_cost total_weight price_per_kg : ℤ)
  (hA : cost_A = 68) 
  (hB : cost_B = 96) 
  (hRatio : ratio_A_B = 3) 
  (hTotalCost : total_cost = 3 * cost_A + cost_B) 
  (hTotalWeight : total_weight = 3 + 1)
  (hPricePerKg : price_per_kg = total_cost / total_weight) : 
  price_per_kg = 75 := 
by
  sorry

end alloy_price_per_kg_l217_21724


namespace entrance_exit_plans_l217_21745

-- Definitions as per the conditions in the problem
def south_gates : Nat := 4
def north_gates : Nat := 3
def west_gates : Nat := 2

-- Conditions translated into Lean definitions
def ways_to_enter := south_gates + north_gates
def ways_to_exit := west_gates + north_gates

-- The theorem to be proved: the number of entrance and exit plans
theorem entrance_exit_plans : ways_to_enter * ways_to_exit = 35 := by
  sorry

end entrance_exit_plans_l217_21745


namespace circle_equation_l217_21726

theorem circle_equation (a : ℝ) (h : a = 1) :
  (∀ (C : ℝ × ℝ), C = (a, a) →
  (∀ (r : ℝ), r = dist C (1, 0) →
  r = 1 → ((x - a) ^ 2 + (y - a) ^ 2 = r ^ 2))) :=
by
  sorry

end circle_equation_l217_21726


namespace derivative_of_f_domain_of_f_range_of_f_l217_21719

open Real

noncomputable def f (x : ℝ) := 1 / (x + sqrt (1 + 2 * x^2))

theorem derivative_of_f (x : ℝ) : 
  deriv f x = - ((sqrt (1 + 2 * x^2) + 2 * x) / (sqrt (1 + 2 * x^2) * (x + sqrt (1 + 2 * x^2))^2)) :=
by
  sorry

theorem domain_of_f : ∀ x : ℝ, f x ≠ 0 :=
by
  sorry

theorem range_of_f : 
  ∀ y : ℝ, 0 < y ∧ y ≤ sqrt 2 → ∃ x : ℝ, f x = y :=
by
  sorry

end derivative_of_f_domain_of_f_range_of_f_l217_21719


namespace roots_range_l217_21734

theorem roots_range (b : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + b = 0 → 0 < x) ↔ 0 < b ∧ b ≤ 1 :=
sorry

end roots_range_l217_21734


namespace find_dividing_line_l217_21790

/--
A line passing through point P(1,1) divides the circular region \{(x, y) \mid x^2 + y^2 \leq 4\} into two parts,
making the difference in area between these two parts the largest. Prove that the equation of this line is x + y - 2 = 0.
-/
theorem find_dividing_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∃ (A B C : ℝ), A * 1 + B * 1 + C = 0 ∧
                 (∀ x y, x^2 + y^2 ≤ 4 → A * x + B * y + C = 0 → (x + y - 2) = 0) :=
sorry

end find_dividing_line_l217_21790


namespace continuity_at_x0_l217_21709

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4
def x0 := 3

theorem continuity_at_x0 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_at_x0_l217_21709


namespace one_div_abs_z_eq_sqrt_two_l217_21729

open Complex

theorem one_div_abs_z_eq_sqrt_two (z : ℂ) (h : z = i / (1 - i)) : 1 / Complex.abs z = Real.sqrt 2 :=
by
  sorry

end one_div_abs_z_eq_sqrt_two_l217_21729


namespace contradiction_method_l217_21752

theorem contradiction_method (x y : ℝ) (h : x + y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end contradiction_method_l217_21752


namespace total_cost_3m3_topsoil_l217_21753

def topsoil_cost (V C : ℕ) : ℕ :=
  V * C

theorem total_cost_3m3_topsoil : topsoil_cost 3 12 = 36 :=
by
  unfold topsoil_cost
  exact rfl

end total_cost_3m3_topsoil_l217_21753


namespace triangle_problem_l217_21712

noncomputable def triangle_sum : Real := sorry

theorem triangle_problem
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (hA : A = π / 6) -- A = 30 degrees
  (h_a : a = Real.sqrt 3) -- a = √3
  (h_law_of_sines : ∀ (x : ℝ), x = 2 * triangle_sum * Real.sin x) -- Law of Sines
  (h_sin_30 : Real.sin (π / 6) = 1 / 2) -- sin 30 degrees = 1/2
  : (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) 
  = 2 * Real.sqrt 3 := sorry

end triangle_problem_l217_21712


namespace problem_statement_l217_21758

/-- Let x, y, z be nonzero real numbers such that x + y + z = 0.
    Prove that ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 → x + y + z = 0 → (x^3 + y^3 + z^3) / (x * y * z) = 3. -/
theorem problem_statement (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) (h₁ : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by 
  sorry

end problem_statement_l217_21758


namespace polygon_diagonals_l217_21746

-- Lean statement of the problem

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 2018) : n = 2021 :=
  by sorry

end polygon_diagonals_l217_21746


namespace common_ratio_of_geom_seq_l217_21766

-- Define the conditions: geometric sequence and the given equation
def is_geom_seq (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem common_ratio_of_geom_seq
  (a : ℕ → ℝ)
  (h_geom : is_geom_seq a)
  (h_eq : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, is_geom_seq a ∧ q = 3 := 
sorry

end common_ratio_of_geom_seq_l217_21766


namespace problem_1_problem_2_l217_21755

-- Define propositions
def prop_p (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 / (4 - m) + y^2 / m = 1)

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

def prop_s (m : ℝ) : Prop :=
  ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

-- Problems
theorem problem_1 (m : ℝ) (h : prop_s m) : m < 0 ∨ m ≥ 1 := 
  sorry

theorem problem_2 {m : ℝ} (h1 : prop_p m ∨ prop_q m) (h2 : ¬ prop_q m) : 1 ≤ m ∧ m < 2 :=
  sorry

end problem_1_problem_2_l217_21755


namespace rem_fraction_l217_21771

theorem rem_fraction : 
  let rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋;
  rem (5/7) (-3/4) = -1/28 := 
by
  sorry

end rem_fraction_l217_21771


namespace percentage_increase_l217_21780

def old_price : ℝ := 300
def new_price : ℝ := 330

theorem percentage_increase : ((new_price - old_price) / old_price) * 100 = 10 := by
  sorry

end percentage_increase_l217_21780


namespace operation_1_2010_l217_21750

def operation (m n : ℕ) : ℕ := sorry

axiom operation_initial : operation 1 1 = 2
axiom operation_step (m n : ℕ) : operation m (n + 1) = operation m n + 3

theorem operation_1_2010 : operation 1 2010 = 6029 := sorry

end operation_1_2010_l217_21750


namespace new_paint_intensity_l217_21749

variable (V : ℝ)  -- V is the volume of the original 50% intensity red paint.
variable (I₁ I₂ : ℝ)  -- I₁ is the intensity of the original paint, I₂ is the intensity of the replaced paint.
variable (f : ℝ)  -- f is the fraction of the original paint being replaced.

-- Assume given conditions
axiom intensity_original : I₁ = 0.5
axiom intensity_new : I₂ = 0.25
axiom fraction_replaced : f = 0.8

-- Prove that the new intensity is 30%
theorem new_paint_intensity :
  (f * I₂ + (1 - f) * I₁) = 0.3 := 
by 
  -- This is the main theorem we want to prove
  sorry

end new_paint_intensity_l217_21749


namespace total_onions_l217_21789

theorem total_onions (sara sally fred amy matthew : Nat) 
  (hs : sara = 40) (hl : sally = 55) 
  (hf : fred = 90) (ha : amy = 25) 
  (hm : matthew = 75) :
  sara + sally + fred + amy + matthew = 285 := 
by
  sorry

end total_onions_l217_21789


namespace tan_alpha_cos2alpha_plus_2sin2alpha_l217_21774

theorem tan_alpha_cos2alpha_plus_2sin2alpha (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end tan_alpha_cos2alpha_plus_2sin2alpha_l217_21774


namespace smallest_solution_floor_eq_l217_21731

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l217_21731


namespace sum_of_coordinates_reflection_l217_21762

theorem sum_of_coordinates_reflection (y : ℝ) :
  let A := (3, y)
  let B := (3, -y)
  A.1 + A.2 + B.1 + B.2 = 6 :=
by
  let A := (3, y)
  let B := (3, -y)
  sorry

end sum_of_coordinates_reflection_l217_21762


namespace proportion_solve_x_l217_21713

theorem proportion_solve_x :
  (0.75 / x = 5 / 7) → x = 1.05 :=
by
  sorry

end proportion_solve_x_l217_21713


namespace geometric_sequence_product_l217_21747

-- Define the geometric sequence sum and the initial conditions
variables {S : ℕ → ℚ} {a : ℕ → ℚ}
variables (q : ℚ) (h1 : a 1 = -1/2)
variables (h2 : S 6 / S 3 = 7 / 8)

-- The main proof problem statement
theorem geometric_sequence_product (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 2 * a 4 = 1 / 64 :=
sorry

end geometric_sequence_product_l217_21747


namespace solve_equation_l217_21732

theorem solve_equation (x : ℚ) (h1 : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 := by
  sorry

end solve_equation_l217_21732


namespace A_wins_when_n_is_9_l217_21751

-- Definition of the game conditions and the strategy
def game (n : ℕ) (A_first : Bool) :=
  ∃ strategy : ℕ → ℕ,
    ∀ taken balls_left : ℕ,
      balls_left - taken > 0 →
      taken ≥ 1 → taken ≤ 3 →
      if A_first then
        (balls_left - taken = 0 → strategy (balls_left - taken) = 1) ∧
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)
      else
        (balls_left - taken = 0 → strategy (balls_left - taken) = 0) ∨
        (∀ t : ℕ, t >= 1 ∧ t ≤ 3 → strategy t = balls_left - taken - t)

-- Prove that for n = 9 A has a winning strategy
theorem A_wins_when_n_is_9 : game 9 true :=
sorry

end A_wins_when_n_is_9_l217_21751


namespace mean_of_set_median_is_128_l217_21781

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l217_21781


namespace remainder_of_polynomial_l217_21783

   def polynomial_division_remainder (x : ℝ) : ℝ := x^4 - 4*x^2 + 7

   theorem remainder_of_polynomial : polynomial_division_remainder 1 = 4 :=
   by
     -- This placeholder indicates that the proof is omitted.
     sorry
   
end remainder_of_polynomial_l217_21783


namespace friends_with_john_l217_21784

def total_slices (pizzas slices_per_pizza : Nat) : Nat := pizzas * slices_per_pizza

def total_people (total_slices slices_per_person : Nat) : Nat := total_slices / slices_per_person

def number_of_friends (total_people john : Nat) : Nat := total_people - john

theorem friends_with_john (pizzas slices_per_pizza slices_per_person john friends : Nat) (h_pizzas : pizzas = 3) 
                          (h_slices_per_pizza : slices_per_pizza = 8) (h_slices_per_person : slices_per_person = 4)
                          (h_john : john = 1) (h_friends : friends = 5) :
  number_of_friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) john = friends := by
  sorry

end friends_with_john_l217_21784


namespace smallest_positive_multiple_of_23_mod_89_is_805_l217_21778

theorem smallest_positive_multiple_of_23_mod_89_is_805 : 
  ∃ a : ℕ, 23 * a ≡ 4 [MOD 89] ∧ 23 * a = 805 := 
by
  sorry

end smallest_positive_multiple_of_23_mod_89_is_805_l217_21778


namespace correct_calculation_l217_21759

theorem correct_calculation (x y a b : ℝ) :
  (3*x + 3*y ≠ 6*x*y) ∧
  (x + x ≠ x^2) ∧
  (-9*y^2 + 16*y^2 ≠ 7) ∧
  (9*a^2*b - 9*a^2*b = 0) :=
by
  sorry

end correct_calculation_l217_21759


namespace cheburashkas_erased_l217_21730

theorem cheburashkas_erased (n : ℕ) (rows : ℕ) (krakozyabras : ℕ) 
  (h_spacing : ∀ r, r ≤ rows → krakozyabras = 2 * (n - 1))
  (h_rows : rows = 2)
  (h_krakozyabras : krakozyabras = 29) :
  n = 16 → rows = 2 → krakozyabras = 29 → n = 16 - 5 :=
by
  sorry

end cheburashkas_erased_l217_21730


namespace exists_polynomials_Q_R_l217_21727

noncomputable def polynomial_with_integer_coeff (P : Polynomial ℤ) : Prop :=
  true

theorem exists_polynomials_Q_R (P : Polynomial ℤ) (hP : polynomial_with_integer_coeff P) :
  ∃ (Q R : Polynomial ℤ), 
    (∃ g : Polynomial ℤ, P * Q = Polynomial.comp g (Polynomial.X ^ 2)) ∧ 
    (∃ h : Polynomial ℤ, P * R = Polynomial.comp h (Polynomial.X ^ 3)) :=
by
  sorry

end exists_polynomials_Q_R_l217_21727


namespace no_real_roots_quadratic_l217_21718

theorem no_real_roots_quadratic (k : ℝ) : 
  ∀ (x : ℝ), k * x^2 - 2 * x + 1 / 2 ≠ 0 → k > 2 :=
by 
  intro x h
  have h1 : (-2)^2 - 4 * k * (1/2) < 0 := sorry
  have h2 : 4 - 2 * k < 0 := sorry
  have h3 : 2 < k := sorry
  exact h3

end no_real_roots_quadratic_l217_21718


namespace opposite_of_neg_three_l217_21763

theorem opposite_of_neg_three : -(-3) = 3 := 
by
  sorry

end opposite_of_neg_three_l217_21763


namespace sum_of_circle_areas_constant_l217_21701

theorem sum_of_circle_areas_constant (r OP : ℝ) (h1 : 0 < r) (h2 : 0 ≤ OP ∧ OP < r) 
  (a' b' c' : ℝ) (h3 : a'^2 + b'^2 + c'^2 = OP^2) :
  ∃ (a b c : ℝ), (a^2 + b^2 + c^2 = 3 * r^2 - OP^2) :=
by
  sorry

end sum_of_circle_areas_constant_l217_21701


namespace find_range_a_l217_21765

noncomputable def f (a x : ℝ) : ℝ := abs (2 * x * a + abs (x - 1))

theorem find_range_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 5) ↔ a ≥ 6 :=
by
  sorry

end find_range_a_l217_21765


namespace coplanar_lines_l217_21770

def vector3 := ℝ × ℝ × ℝ

def vec1 : vector3 := (2, -1, 3)
def vec2 (k : ℝ) : vector3 := (3 * k, 1, 2)
def pointVec : vector3 := (-3, 2, -3)

def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem coplanar_lines (k : ℝ) : det3x3 2 (-1) 3 (3 * k) 1 2 (-3) 2 (-3) = 0 → k = -29 / 9 :=
  sorry

end coplanar_lines_l217_21770


namespace percentage_of_blue_flowers_l217_21777

theorem percentage_of_blue_flowers 
  (total_flowers : Nat)
  (red_flowers : Nat)
  (white_flowers : Nat)
  (total_flowers_eq : total_flowers = 10)
  (red_flowers_eq : red_flowers = 4)
  (white_flowers_eq : white_flowers = 2)
  :
  ( (total_flowers - (red_flowers + white_flowers)) * 100 ) / total_flowers = 40 :=
by
  sorry

end percentage_of_blue_flowers_l217_21777


namespace volume_of_intersecting_octahedra_l217_21757

def absolute (x : ℝ) : ℝ := abs x

noncomputable def volume_of_region : ℝ :=
  let region1 (x y z : ℝ) := absolute x + absolute y + absolute z ≤ 2
  let region2 (x y z : ℝ) := absolute x + absolute y + absolute (z - 2) ≤ 2
  -- The region is the intersection of these two inequalities
  -- However, we calculate its volume directly
  (2 / 3 : ℝ)

theorem volume_of_intersecting_octahedra :
  (volume_of_region : ℝ) = (2 / 3 : ℝ) :=
sorry

end volume_of_intersecting_octahedra_l217_21757


namespace percentage_increase_in_rent_l217_21740

theorem percentage_increase_in_rent
  (avg_rent_per_person_before : ℝ)
  (num_friends : ℕ)
  (friend_original_rent : ℝ)
  (avg_rent_per_person_after : ℝ)
  (total_rent_before : ℝ := num_friends * avg_rent_per_person_before)
  (total_rent_after : ℝ := num_friends * avg_rent_per_person_after)
  (rent_increase : ℝ := total_rent_after - total_rent_before)
  (percentage_increase : ℝ := (rent_increase / friend_original_rent) * 100)
  (h1 : avg_rent_per_person_before = 800)
  (h2 : num_friends = 4)
  (h3 : friend_original_rent = 1400)
  (h4 : avg_rent_per_person_after = 870) :
  percentage_increase = 20 :=
by
  sorry

end percentage_increase_in_rent_l217_21740


namespace gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l217_21743

def a : ℕ := 2^1025 - 1
def b : ℕ := 2^1056 - 1
def answer : ℕ := 2147483647

theorem gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1 :
  Int.gcd a b = answer := by
  sorry

end gcd_2_pow_1025_sub_1_and_2_pow_1056_sub_1_l217_21743


namespace min_value_of_f_l217_21764

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (2 * x / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 1 + 2 * Real.sqrt 2) := 
sorry

end min_value_of_f_l217_21764


namespace ratio_B_to_C_l217_21706

-- Definitions for conditions
def total_amount : ℕ := 1440
def B_amt : ℕ := 270
def A_amt := (1 / 3) * B_amt
def C_amt := total_amount - A_amt - B_amt

-- Theorem statement
theorem ratio_B_to_C : (B_amt : ℚ) / C_amt = 1 / 4 :=
  by
    sorry

end ratio_B_to_C_l217_21706


namespace quadratic_equation_solution_l217_21744

theorem quadratic_equation_solution :
  ∃ x1 x2 : ℝ, (x1 = (-1 + Real.sqrt 13) / 2 ∧ x2 = (-1 - Real.sqrt 13) / 2 
  ∧ (∀ x : ℝ, x^2 + x - 3 = 0 → x = x1 ∨ x = x2)) :=
sorry

end quadratic_equation_solution_l217_21744


namespace two_digit_number_ratio_l217_21786

def two_digit_number (a b : ℕ) : ℕ := 10 * a + b
def swapped_two_digit_number (a b : ℕ) : ℕ := 10 * b + a

theorem two_digit_number_ratio (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 1 ≤ b ∧ b ≤ 9) (h_ratio : 6 * two_digit_number a b = 5 * swapped_two_digit_number a b) : 
  two_digit_number a b = 45 :=
by
  sorry

end two_digit_number_ratio_l217_21786


namespace minimum_value_l217_21714

open Classical

variable {a b c : ℝ}

theorem minimum_value (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a + b + c = 4) :
  36 ≤ (9 / a) + (16 / b) + (25 / c) :=
sorry

end minimum_value_l217_21714


namespace problem_l217_21725

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def quadratic_roots (a₃ a₁₀ : ℝ) : Prop :=
a₃^2 - 3 * a₃ - 5 = 0 ∧ a₁₀^2 - 3 * a₁₀ - 5 = 0

theorem problem (a : ℕ → ℝ) (h1 : is_arithmetic_seq a)
  (h2 : quadratic_roots (a 3) (a 10)) :
  a 5 + a 8 = 3 :=
sorry

end problem_l217_21725


namespace area_of_triangle_is_11_25_l217_21738

noncomputable def area_of_triangle : ℝ :=
  let A := (1 / 2, 2)
  let B := (8, 2)
  let C := (2, 5)
  let base := (B.1 - A.1 : ℝ)
  let height := (C.2 - A.2 : ℝ)
  0.5 * base * height

theorem area_of_triangle_is_11_25 :
  area_of_triangle = 11.25 := sorry

end area_of_triangle_is_11_25_l217_21738


namespace calculate_value_of_A_plus_C_l217_21785

theorem calculate_value_of_A_plus_C (A B C : ℕ) (hA : A = 238) (hAB : A = B + 143) (hBC : C = B + 304) : A + C = 637 :=
by
  sorry

end calculate_value_of_A_plus_C_l217_21785


namespace sequence_inequality_l217_21791

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (m n : ℕ)
  (h1 : a 1 = 21/16)
  (h2 : ∀ n ≥ 2, 2 * a n - 3 * a (n - 1) = 3 / 2^(n + 1))
  (h3 : m ≥ 2)
  (h4 : n ≤ m) :
  (a n + 3 / 2^(n + 3))^(1 / m) * (m - (2 / 3)^(n * (m - 1) / m)) < (m^2 - 1) / (m - n + 1) :=
sorry

end sequence_inequality_l217_21791


namespace last_four_digits_5_pow_2011_l217_21742

theorem last_four_digits_5_pow_2011 : 
  (5^2011 % 10000) = 8125 :=
by
  -- Definitions based on conditions in the problem
  have h5 : 5^5 % 10000 = 3125 := sorry
  have h6 : 5^6 % 10000 = 5625 := sorry
  have h7 : 5^7 % 10000 = 8125 := sorry
  
  -- Prove using periodicity and modular arithmetic
  sorry

end last_four_digits_5_pow_2011_l217_21742


namespace scrabble_middle_letter_value_l217_21756

theorem scrabble_middle_letter_value 
  (triple_word_score : ℕ) (single_letter_value : ℕ) (middle_letter_value : ℕ)
  (h1 : triple_word_score = 30)
  (h2 : single_letter_value = 1)
  : 3 * (2 * single_letter_value + middle_letter_value) = triple_word_score → middle_letter_value = 8 :=
by
  sorry

end scrabble_middle_letter_value_l217_21756


namespace find_xnp_l217_21708

theorem find_xnp (x n p : ℕ) (h1 : 0 < x) (h2 : 0 < n) (h3 : Nat.Prime p) 
                  (h4 : 2 * x^3 + x^2 + 10 * x + 5 = 2 * p^n) : x + n + p = 6 :=
by
  sorry

end find_xnp_l217_21708


namespace student_exchanges_l217_21720

theorem student_exchanges (x : ℕ) : x * (x - 1) = 72 :=
sorry

end student_exchanges_l217_21720


namespace find_value_of_x_l217_21703

theorem find_value_of_x (x : ℝ) : (45 * x = 0.4 * 900) -> x = 8 :=
by
  intro h
  sorry

end find_value_of_x_l217_21703


namespace part_a_part_b_l217_21710

def happy (n : ℕ) : Prop :=
  ∃ (a b : ℤ), a^2 + b^2 = n

theorem part_a (t : ℕ) (ht : happy t) : happy (2 * t) := 
sorry

theorem part_b (t : ℕ) (ht : happy t) : ¬ happy (3 * t) := 
sorry

end part_a_part_b_l217_21710


namespace parallel_and_perpendicular_implies_perpendicular_l217_21711

variables (l : Line) (α β : Plane)

axiom line_parallel_plane (l : Line) (π : Plane) : Prop
axiom line_perpendicular_plane (l : Line) (π : Plane) : Prop
axiom planes_are_perpendicular (π₁ π₂ : Plane) : Prop

theorem parallel_and_perpendicular_implies_perpendicular
  (h1 : line_parallel_plane l α)
  (h2 : line_perpendicular_plane l β) 
  : planes_are_perpendicular α β :=
sorry

end parallel_and_perpendicular_implies_perpendicular_l217_21711


namespace f_2009_l217_21705

noncomputable def f : ℝ → ℝ := sorry -- This will be defined by the conditions.

axiom even_f (x : ℝ) : f x = f (-x)
axiom periodic_f (x : ℝ) : f (x + 6) = f x + f 3
axiom f_one : f 1 = 2

theorem f_2009 : f 2009 = 2 :=
by {
  -- The proof would go here, summarizing the logical steps derived in the previous sections.
  sorry
}

end f_2009_l217_21705


namespace probability_of_woman_lawyer_is_54_percent_l217_21792

variable (total_members : ℕ) (women_percentage lawyers_percentage : ℕ)
variable (H_total_members_pos : total_members > 0) 
variable (H_women_percentage : women_percentage = 90)
variable (H_lawyers_percentage : lawyers_percentage = 60)

def probability_woman_lawyer : ℕ :=
  (women_percentage * lawyers_percentage * total_members) / (100 * 100)

theorem probability_of_woman_lawyer_is_54_percent (H_total_members_pos : total_members > 0)
  (H_women_percentage : women_percentage = 90)
  (H_lawyers_percentage : lawyers_percentage = 60) :
  probability_woman_lawyer total_members women_percentage lawyers_percentage = 54 :=
by
  sorry

end probability_of_woman_lawyer_is_54_percent_l217_21792


namespace water_needed_l217_21733

theorem water_needed (nutrient_concentrate : ℝ) (distilled_water : ℝ) (total_volume : ℝ) 
    (h1 : nutrient_concentrate = 0.08) (h2 : distilled_water = 0.04) (h3 : total_volume = 1) :
    total_volume * (distilled_water / (nutrient_concentrate + distilled_water)) = 0.333 :=
by
  sorry

end water_needed_l217_21733


namespace solve_for_x_l217_21795

theorem solve_for_x : ∀ (x : ℝ), 
  (x + 2 * x + 3 * x + 4 * x = 5) → (x = 1 / 2) :=
by 
  intros x H
  sorry

end solve_for_x_l217_21795


namespace washing_machines_removed_per_box_l217_21741

theorem washing_machines_removed_per_box 
  (crates : ℕ) (boxes_per_crate : ℕ) (washing_machines_per_box : ℕ) 
  (total_removed : ℕ) (total_crates : ℕ) (total_boxes_per_crate : ℕ) 
  (total_washing_machines_per_box : ℕ) 
  (h1 : crates = total_crates) (h2 : boxes_per_crate = total_boxes_per_crate) 
  (h3 : washing_machines_per_box = total_washing_machines_per_box) 
  (h4 : total_removed = 60) (h5 : total_crates = 10) 
  (h6 : total_boxes_per_crate = 6) 
  (h7 : total_washing_machines_per_box = 4):
  total_removed / (total_crates * total_boxes_per_crate) = 1 :=
by
  sorry

end washing_machines_removed_per_box_l217_21741


namespace nat_divides_2_pow_n_minus_1_l217_21739

theorem nat_divides_2_pow_n_minus_1 (n : ℕ) (hn : 0 < n) : n ∣ 2^n - 1 ↔ n = 1 :=
  sorry

end nat_divides_2_pow_n_minus_1_l217_21739


namespace snickers_bars_needed_l217_21728

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l217_21728


namespace ribbon_used_l217_21760

def total_ribbon : ℕ := 84
def leftover_ribbon : ℕ := 38
def used_ribbon : ℕ := 46

theorem ribbon_used : total_ribbon - leftover_ribbon = used_ribbon := sorry

end ribbon_used_l217_21760


namespace find_urn_yellow_balls_l217_21737

theorem find_urn_yellow_balls :
  ∃ (M : ℝ), 
    (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
    M = 111 := 
sorry

end find_urn_yellow_balls_l217_21737


namespace chocolates_per_small_box_l217_21700

/-- A large box contains 19 small boxes and each small box contains a certain number of chocolate bars.
There are 475 chocolate bars in the large box. --/
def number_of_chocolate_bars_per_small_box : Prop :=
  ∃ x : ℕ, 475 = 19 * x ∧ x = 25

theorem chocolates_per_small_box : number_of_chocolate_bars_per_small_box :=
by
  sorry -- proof is skipped

end chocolates_per_small_box_l217_21700


namespace regular_ticket_cost_l217_21782

theorem regular_ticket_cost
    (adults : ℕ) (children : ℕ) (cash_given : ℕ) (change_received : ℕ) (adult_cost : ℕ) (child_cost : ℕ) :
    adults = 2 →
    children = 3 →
    cash_given = 40 →
    change_received = 1 →
    child_cost = adult_cost - 2 →
    2 * adult_cost + 3 * child_cost = cash_given - change_received →
    adult_cost = 9 :=
by
  intros h_adults h_children h_cash_given h_change_received h_child_cost h_sum
  sorry

end regular_ticket_cost_l217_21782


namespace initial_amount_of_water_l217_21715

theorem initial_amount_of_water 
  (W : ℚ) 
  (h1 : W - (7/15) * W - (5/8) * (W - (7/15) * W) - (2/3) * (W - (7/15) * W - (5/8) * (W - (7/15) * W)) = 2.6) 
  : W = 39 := 
sorry

end initial_amount_of_water_l217_21715


namespace value_of_x_l217_21702

theorem value_of_x (x : ℝ) (a : ℝ) (h1 : x ^ 2 * 8 ^ 3 / 256 = a) (h2 : a = 450) : x = 15 ∨ x = -15 := by
  sorry

end value_of_x_l217_21702


namespace bobby_initial_candy_count_l217_21798

theorem bobby_initial_candy_count (C : ℕ) (h : C + 4 + 14 = 51) : C = 33 :=
by
  sorry

end bobby_initial_candy_count_l217_21798


namespace calories_in_300g_l217_21773

/-
Define the conditions of the problem.
-/

def lemon_juice_grams := 150
def sugar_grams := 200
def lime_juice_grams := 50
def water_grams := 500

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 390
def lime_juice_calories_per_100g := 20
def water_calories := 0

/-
Define the total weight of the beverage.
-/
def total_weight := lemon_juice_grams + sugar_grams + lime_juice_grams + water_grams

/-
Define the total calories of the beverage.
-/
def total_calories := 
  (lemon_juice_calories_per_100g * lemon_juice_grams / 100) + 
  (sugar_calories_per_100g * sugar_grams / 100) + 
  (lime_juice_calories_per_100g * lime_juice_grams / 100) + 
  water_calories

/-
Prove the number of calories in 300 grams of the beverage.
-/
theorem calories_in_300g : (total_calories / total_weight) * 300 = 278 := by
  sorry

end calories_in_300g_l217_21773


namespace find_x_l217_21769

def op (a b : ℤ) : ℤ := -2 * a + b

theorem find_x (x : ℤ) (h : op x (-5) = 3) : x = -4 :=
by
  sorry

end find_x_l217_21769


namespace number_of_hexagons_l217_21704

-- Definitions based on conditions
def num_pentagons : ℕ := 12

-- Based on the problem statement, the goal is to prove that the number of hexagons is 20
theorem number_of_hexagons (h : num_pentagons = 12) : ∃ (num_hexagons : ℕ), num_hexagons = 20 :=
by {
  -- proof would be here
  sorry
}

end number_of_hexagons_l217_21704


namespace inequality_a2_b2_c2_geq_abc_l217_21796

theorem inequality_a2_b2_c2_geq_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_cond: a + b + c ≥ a * b * c) :
  a^2 + b^2 + c^2 ≥ a * b * c := 
sorry

end inequality_a2_b2_c2_geq_abc_l217_21796


namespace dogs_in_kennel_l217_21787

variable (C D : ℕ)

-- definition of the ratio condition 
def ratio_condition : Prop :=
  C * 4 = 3 * D

-- definition of the difference condition
def difference_condition : Prop :=
  C = D - 8

theorem dogs_in_kennel (h1 : ratio_condition C D) (h2 : difference_condition C D) : D = 32 :=
by 
  -- proof steps go here
  sorry

end dogs_in_kennel_l217_21787


namespace bus_patrons_correct_l217_21754

-- Definitions corresponding to conditions
def number_of_golf_carts : ℕ := 13
def patrons_per_cart : ℕ := 3
def car_patrons : ℕ := 12

-- Multiply to get total patrons transported by golf carts
def total_patrons := number_of_golf_carts * patrons_per_cart

-- Calculate bus patrons
def bus_patrons := total_patrons - car_patrons

-- The statement to prove
theorem bus_patrons_correct : bus_patrons = 27 :=
by
  sorry

end bus_patrons_correct_l217_21754


namespace total_writing_instruments_l217_21793

theorem total_writing_instruments 
 (bags : ℕ) (compartments_per_bag : ℕ) (empty_compartments : ℕ) (one_compartment : ℕ) (remaining_compartments : ℕ) 
 (writing_instruments_per_compartment : ℕ) (writing_instruments_in_one : ℕ) : 
 bags = 16 → 
 compartments_per_bag = 6 → 
 empty_compartments = 5 → 
 one_compartment = 1 → 
 remaining_compartments = 90 →
 writing_instruments_per_compartment = 8 → 
 writing_instruments_in_one = 6 → 
 (remaining_compartments * writing_instruments_per_compartment + one_compartment * writing_instruments_in_one) = 726 := 
  by
   sorry

end total_writing_instruments_l217_21793


namespace find_a_l217_21797

theorem find_a (a : ℤ) (A : Set ℤ) (B : Set ℤ) :
  A = {-2, 3 * a - 1, a^2 - 3} ∧
  B = {a - 2, a - 1, a + 1} ∧
  A ∩ B = {-2} → a = -3 :=
by
  intro H
  sorry

end find_a_l217_21797


namespace estimate_undetected_typos_l217_21799

variables (a b c : ℕ)
-- a, b, c ≥ 0 are non-negative integers representing discovered errors by proofreader A, B, and common errors respectively.

theorem estimate_undetected_typos (h : c ≤ a ∧ c ≤ b) :
  ∃ n : ℕ, n = a * b / c - a - b + c :=
sorry

end estimate_undetected_typos_l217_21799


namespace Elberta_has_21_dollars_l217_21723

theorem Elberta_has_21_dollars
  (Granny_Smith : ℕ)
  (Anjou : ℕ)
  (Elberta : ℕ)
  (h1 : Granny_Smith = 72)
  (h2 : Anjou = Granny_Smith / 4)
  (h3 : Elberta = Anjou + 3) :
  Elberta = 21 := 
  by {
    sorry
  }

end Elberta_has_21_dollars_l217_21723


namespace triangle_internal_angle_A_l217_21736

theorem triangle_internal_angle_A {B C A : ℝ} (hB : Real.tan B = -2) (hC : Real.tan C = 1 / 3) (h_sum: A = π - B - C) : A = π / 4 :=
by
  sorry

end triangle_internal_angle_A_l217_21736


namespace problem_statement_l217_21722

noncomputable def k_value (k : ℝ) : Prop :=
  (∀ (x y : ℝ), x + y = k → x^2 + y^2 = 4) ∧ (∀ (A B : ℝ × ℝ), (∃ (x y : ℝ), A = (x, y) ∧ x^2 + y^2 = 4) ∧ (∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4) ∧ 
  (∃ (xa ya xb yb : ℝ), A = (xa, ya) ∧ B = (xb, yb) ∧ |(xa - xb, ya - yb)| = |(xa, ya)| + |(xb, yb)|)) → k = 2

theorem problem_statement (k : ℝ) (h : k > 0) : k_value k :=
  sorry

end problem_statement_l217_21722
