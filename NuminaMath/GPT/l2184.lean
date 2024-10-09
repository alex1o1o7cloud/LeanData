import Mathlib

namespace rachel_left_24_brownies_at_home_l2184_218401

-- Defining the conditions
def total_brownies : ℕ := 40
def brownies_brought_to_school : ℕ := 16

-- Formulation of the theorem
theorem rachel_left_24_brownies_at_home : (total_brownies - brownies_brought_to_school = 24) :=
by
  sorry

end rachel_left_24_brownies_at_home_l2184_218401


namespace quadratic_real_roots_condition_sufficient_l2184_218445

theorem quadratic_real_roots_condition_sufficient (m : ℝ) : (m < 1 / 4) → ∃ x : ℝ, x^2 + x + m = 0 :=
by
  sorry

end quadratic_real_roots_condition_sufficient_l2184_218445


namespace interval_of_increase_of_f_l2184_218463

noncomputable def f (x : ℝ) := Real.logb (0.5) (x - x^2)

theorem interval_of_increase_of_f :
  ∀ x : ℝ, x ∈ Set.Ioo (1/2) 1 → ∃ ε > 0, ∀ y : ℝ, y ∈ Set.Ioo (x - ε) (x + ε) → f y > f x :=
  by
    sorry

end interval_of_increase_of_f_l2184_218463


namespace value_of_x_l2184_218468

theorem value_of_x 
  (x : ℚ) 
  (h₁ : 6 * x^2 + 19 * x - 7 = 0) 
  (h₂ : 18 * x^2 + 47 * x - 21 = 0) : 
  x = 1 / 3 := 
  sorry

end value_of_x_l2184_218468


namespace problem_solution_l2184_218414

def problem_conditions : Prop :=
  (∃ (students_total excellent_students: ℕ) 
     (classA_excellent classB_not_excellent: ℕ),
     students_total = 110 ∧
     excellent_students = 30 ∧
     classA_excellent = 10 ∧
     classB_not_excellent = 30)

theorem problem_solution
  (students_total excellent_students: ℕ)
  (classA_excellent classB_not_excellent: ℕ)
  (h : problem_conditions) :
  ∃ classA_not_excellent classB_excellent: ℕ,
    classA_not_excellent = 50 ∧
    classB_excellent = 20 ∧
    ((∃ χ_squared: ℝ, χ_squared = 7.5 ∧ χ_squared > 6.635) → true) ∧
    (∃ selectA selectB: ℕ, selectA = 5 ∧ selectB = 3) :=
by {
  sorry
}

end problem_solution_l2184_218414


namespace inequality_solution_l2184_218416

theorem inequality_solution {x : ℝ} (h : |x + 3| - |x - 1| > 0) : x > -1 :=
sorry

end inequality_solution_l2184_218416


namespace tangent_line_to_ex_l2184_218471

theorem tangent_line_to_ex (b : ℝ) : (∃ x0 : ℝ, (∀ x : ℝ, (e^x - e^x0 - (x - x0) * e^x0 = 0) ↔ y = x + b)) → b = 1 :=
by
  sorry

end tangent_line_to_ex_l2184_218471


namespace deepak_present_age_l2184_218488

variable (R D : ℕ)

theorem deepak_present_age 
  (h1 : R + 22 = 26) 
  (h2 : R / D = 4 / 3) : 
  D = 3 := 
sorry

end deepak_present_age_l2184_218488


namespace numberOfTermsArithmeticSequence_l2184_218484

theorem numberOfTermsArithmeticSequence (a1 d l : ℕ) (h1 : a1 = 3) (h2 : d = 4) (h3 : l = 2012) :
  ∃ n : ℕ, 3 + (n - 1) * 4 ≤ 2012 ∧ (n : ℕ) = 502 :=
by {
  sorry
}

end numberOfTermsArithmeticSequence_l2184_218484


namespace function_expression_and_min_value_l2184_218469

def f (x b : ℝ) := x^2 - 2*x + b

theorem function_expression_and_min_value 
    (a b : ℝ)
    (condition1 : f (2 ^ a) b = b)
    (condition2 : f a b = 4) :
    f a b = 5 
    ∧ 
    ∃ c : ℝ, f (2^c) 5 = 4 ∧ c = 0 :=
by
  sorry

end function_expression_and_min_value_l2184_218469


namespace remainder_when_3m_divided_by_5_l2184_218467

theorem remainder_when_3m_divided_by_5 (m : ℤ) (hm : m % 5 = 2) : (3 * m) % 5 = 1 := 
sorry

end remainder_when_3m_divided_by_5_l2184_218467


namespace eval_expression_l2184_218465

theorem eval_expression : 
  ( ( (476 * 100 + 424 * 100) * 2^3 - 4 * (476 * 100 * 424 * 100) ) * (376 - 150) ) / 250 = -7297340160 :=
by
  sorry

end eval_expression_l2184_218465


namespace total_heartbeats_correct_l2184_218494

-- Define the given conditions
def heartbeats_per_minute : ℕ := 160
def pace_per_mile : ℕ := 6
def race_distance : ℕ := 30

-- Define the total heartbeats during the race
def total_heartbeats_during_race : ℕ :=
  pace_per_mile * race_distance * heartbeats_per_minute

-- Theorem stating the mathematically equivalent proof problem
theorem total_heartbeats_correct :
  total_heartbeats_during_race = 28800 :=
by
  -- Proof is omitted
  sorry

end total_heartbeats_correct_l2184_218494


namespace probability_of_four_digit_number_divisible_by_3_l2184_218453

def digits : List ℕ := [0, 1, 2, 3, 4, 5]

def count_valid_four_digit_numbers : Int :=
  let all_digits := digits
  let total_four_digit_numbers := 180
  let valid_four_digit_numbers := 96
  total_four_digit_numbers

def probability_divisible_by_3 : ℚ :=
  (96 : ℚ) / (180 : ℚ)

theorem probability_of_four_digit_number_divisible_by_3 :
  probability_divisible_by_3 = 8 / 15 :=
by
  sorry

end probability_of_four_digit_number_divisible_by_3_l2184_218453


namespace contractor_absent_days_l2184_218425

theorem contractor_absent_days :
  ∃ (x y : ℝ), x + y = 30 ∧ 25 * x - 7.5 * y = 490 ∧ y = 8 :=
by {
  sorry
}

end contractor_absent_days_l2184_218425


namespace determine_f_when_alpha_l2184_218486

noncomputable def solves_functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
∀ (x y : ℝ), 0 < x → 0 < y → f (f x + y) = α * x + 1 / (f (1 / y))

theorem determine_f_when_alpha (α : ℝ) (f : ℝ → ℝ) :
  (α = 1 → ∀ x, 0 < x → f x = x) ∧ (α ≠ 1 → ∀ f, ¬ solves_functional_equation f α) := by
  sorry

end determine_f_when_alpha_l2184_218486


namespace inequality_inequality_l2184_218428

theorem inequality_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_inequality_l2184_218428


namespace sum_of_possible_values_l2184_218495

theorem sum_of_possible_values (x : ℝ) :
  (x + 3) * (x - 4) = 20 →
  ∃ a b, (a ≠ b) ∧ 
         ((x = a) ∨ (x = b)) ∧ 
         (x^2 - x - 32 = 0) ∧ 
         (a + b = 1) :=
by
  sorry

end sum_of_possible_values_l2184_218495


namespace find_a_plus_b_l2184_218475

noncomputable def lines_intersect (a b : ℝ) : Prop := 
  (∃ x y : ℝ, (x = 1/3 * y + a) ∧ (y = 1/3 * x + b) ∧ (x = 3) ∧ (y = 6))

theorem find_a_plus_b (a b : ℝ) (h : lines_intersect a b) : a + b = 6 :=
sorry

end find_a_plus_b_l2184_218475


namespace ex1_ex2_l2184_218406

-- Definition of the "multiplication-subtraction" operation.
def mult_sub (a b : ℚ) : ℚ :=
  if a = 0 then abs b else if b = 0 then abs a else if abs a = abs b then 0 else
  if (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) then abs a - abs b else -(abs a - abs b)

theorem ex1 : mult_sub (mult_sub (3) (-2)) (mult_sub (-9) 0) = -8 :=
  sorry

theorem ex2 : ∃ (a b c : ℚ), (mult_sub (mult_sub a b) c) ≠ (mult_sub a (mult_sub b c)) :=
  ⟨3, -2, 4, by simp [mult_sub]; sorry⟩

end ex1_ex2_l2184_218406


namespace discount_per_issue_l2184_218431

theorem discount_per_issue
  (normal_subscription_cost : ℝ) (months : ℕ) (issues_per_month : ℕ) 
  (promotional_discount : ℝ) :
  normal_subscription_cost = 34 →
  months = 18 →
  issues_per_month = 2 →
  promotional_discount = 9 →
  (normal_subscription_cost - promotional_discount) / (months * issues_per_month) = 0.25 :=
by
  intros h1 h2 h3 h4
  sorry

end discount_per_issue_l2184_218431


namespace quadratic_is_complete_the_square_l2184_218412

theorem quadratic_is_complete_the_square :
  ∃ a b c : ℝ, 15 * (x : ℝ)^2 + 150 * x + 2250 = a * (x + b)^2 + c 
  ∧ a + b + c = 1895 :=
sorry

end quadratic_is_complete_the_square_l2184_218412


namespace history_students_count_l2184_218456

theorem history_students_count
  (total_students : ℕ)
  (sample_students : ℕ)
  (physics_students_sampled : ℕ)
  (history_students_sampled : ℕ)
  (x : ℕ)
  (H1 : total_students = 1500)
  (H2 : sample_students = 120)
  (H3 : physics_students_sampled = 80)
  (H4 : history_students_sampled = sample_students - physics_students_sampled)
  (H5 : x = 1500 * history_students_sampled / sample_students) :
  x = 500 :=
by
  sorry

end history_students_count_l2184_218456


namespace a_is_perfect_square_l2184_218464

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l2184_218464


namespace intersection_complement_eq_l2184_218426

open Set

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  (U = {1, 2, 3, 4, 5, 6}) →
  (A = {1, 3}) →
  (B = {3, 4, 5}) →
  A ∩ (U \ B) = {1} :=
by
  intros hU hA hB
  subst hU
  subst hA
  subst hB
  sorry

end intersection_complement_eq_l2184_218426


namespace find_a_b_l2184_218481

theorem find_a_b (a b : ℝ) (h₁ : a^2 = 64 * b) (h₂ : a^2 = 4 * b) : a = 0 ∧ b = 0 :=
by
  sorry

end find_a_b_l2184_218481


namespace fraction_of_roots_l2184_218446

theorem fraction_of_roots (a b : ℝ) (h : a * b = -209) (h_sum : a + b = -8) : 
  (a * b) / (a + b) = 209 / 8 := 
by 
  sorry

end fraction_of_roots_l2184_218446


namespace equal_points_per_person_l2184_218492

theorem equal_points_per_person :
  let blue_eggs := 12
  let blue_points := 2
  let pink_eggs := 5
  let pink_points := 3
  let golden_eggs := 3
  let golden_points := 5
  let total_people := 4
  (blue_eggs * blue_points + pink_eggs * pink_points + golden_eggs * golden_points) / total_people = 13 :=
by
  -- place the steps based on the conditions and calculations
  sorry

end equal_points_per_person_l2184_218492


namespace identify_different_correlation_l2184_218450

-- Define the concept of correlation
inductive Correlation
| positive
| negative

-- Define the conditions for each option
def option_A : Correlation := Correlation.positive
def option_B : Correlation := Correlation.positive
def option_C : Correlation := Correlation.negative
def option_D : Correlation := Correlation.positive

-- The statement to prove
theorem identify_different_correlation :
  (option_A = Correlation.positive) ∧ 
  (option_B = Correlation.positive) ∧ 
  (option_D = Correlation.positive) ∧ 
  (option_C = Correlation.negative) := 
sorry

end identify_different_correlation_l2184_218450


namespace bacteria_growth_relation_l2184_218443

variable (w1: ℝ := 10.0) (w2: ℝ := 16.0) (w3: ℝ := 25.6)

theorem bacteria_growth_relation :
  (w2 / w1) = (w3 / w2) :=
by
  sorry

end bacteria_growth_relation_l2184_218443


namespace quadratic_form_decomposition_l2184_218441

theorem quadratic_form_decomposition (a b c : ℝ) (h : ∀ x : ℝ, 8 * x^2 + 64 * x + 512 = a * (x + b) ^ 2 + c) :
  a + b + c = 396 := 
sorry

end quadratic_form_decomposition_l2184_218441


namespace katie_earnings_l2184_218499

-- Define the constants for the problem
def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

-- Define the total earnings calculation
def total_necklaces : Nat := bead_necklaces + gemstone_necklaces
def total_earnings : Nat := total_necklaces * cost_per_necklace

-- Statement of the proof problem
theorem katie_earnings : total_earnings = 21 := by
  sorry

end katie_earnings_l2184_218499


namespace factors_of_2550_have_more_than_3_factors_l2184_218493

theorem factors_of_2550_have_more_than_3_factors :
  ∃ n: ℕ, n = 5 ∧
    ∃ d: ℕ, d = 2550 ∧
    (∀ x < n, ∃ y: ℕ, y ∣ d ∧ (∃ z, z ∣ y ∧ z > 3)) :=
sorry

end factors_of_2550_have_more_than_3_factors_l2184_218493


namespace calculate_product_l2184_218466

theorem calculate_product :
  7 * (9 + 2/5) = 65 + 4/5 := 
by
  sorry

end calculate_product_l2184_218466


namespace johns_final_amount_l2184_218403

def initial_amount : ℝ := 45.7
def deposit_amount : ℝ := 18.6
def withdrawal_amount : ℝ := 20.5

theorem johns_final_amount : initial_amount + deposit_amount - withdrawal_amount = 43.8 :=
by
  sorry

end johns_final_amount_l2184_218403


namespace find_value_of_N_l2184_218429

theorem find_value_of_N :
  (2 * ((3.6 * 0.48 * 2.5) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) :=
by {
  sorry
}

end find_value_of_N_l2184_218429


namespace rectangular_garden_width_l2184_218480

theorem rectangular_garden_width
  (w : ℝ)
  (h₁ : ∃ l, l = 3 * w)
  (h₂ : ∃ A, A = l * w ∧ A = 507) : 
  w = 13 :=
by
  sorry

end rectangular_garden_width_l2184_218480


namespace simplify_expr_l2184_218459

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) :
  (3/4) * (8/(x^2) + 12*x - 5) = 6/(x^2) + 9*x - 15/4 := by
  sorry

end simplify_expr_l2184_218459


namespace incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l2184_218440

structure Tetrahedron (α : Type*) [MetricSpace α] :=
(A B C D : α)

def Incenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry
def Circumcenter {α : Type*} [MetricSpace α] (T : Tetrahedron α) : α := sorry

def equidistant_from_faces {α : Type*} [MetricSpace α] (T : Tetrahedron α) (I : α) : Prop := sorry
def equidistant_from_vertices {α : Type*} [MetricSpace α] (T : Tetrahedron α) (O : α) : Prop := sorry
def skew_edges_equal {α : Type*} [MetricSpace α] (T : Tetrahedron α) : Prop := sorry

theorem incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal
  {α : Type*} [MetricSpace α] (T : Tetrahedron α) :
  (∃ I, ∃ O, (Incenter T = I) ∧ (Circumcenter T = O) ∧ 
            (equidistant_from_faces T I) ∧ (equidistant_from_vertices T O)) ↔ (skew_edges_equal T) := 
sorry

end incenter_circumcenter_coincide_if_and_only_if_skew_edges_equal_l2184_218440


namespace geometric_sequence_general_formula_l2184_218490

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 q : ℝ, ∀ n : ℕ, a n = a1 * q ^ (n - 1)

variables (a : ℕ → ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := a 1 + a 3 = 10
def condition2 : Prop := a 4 + a 6 = 5 / 4

-- The final statement to prove
theorem geometric_sequence_general_formula (h : geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) :
  ∀ n : ℕ, a n = 2 ^ (4 - n) :=
sorry

end geometric_sequence_general_formula_l2184_218490


namespace arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l2184_218491

-- Definitions and theorems for the given conditions

-- (1) General formula for the arithmetic sequence
theorem arithmetic_sequence_formula (a S : Nat → Int) (n : Nat) (h1 : a 2 = -1)
  (h2 : S 9 = 5 * S 5) : 
  ∀ n, a n = -8 * n + 15 := 
sorry

-- (2) Minimum value of t - s
theorem min_value_t_minus_s (b : Nat → Rat) (T : Nat → Rat) 
  (h3 : ∀ n, b n = 1 / ((-8 * (n + 1) + 15) * (-8 * (n + 2) + 15))) 
  (h4 : ∀ n, s ≤ T n ∧ T n ≤ t) : 
  t - s = 1 / 72 := 
sorry

-- (3) Maximum value of k
theorem max_value_k (S a : Nat → Int) (k : Rat)
  (h5 : ∀ n, n ≥ 3 → S n / a n ≤ n^2 / (n + k)) :
  k = 80 / 9 := 
sorry

end arithmetic_sequence_formula_min_value_t_minus_s_max_value_k_l2184_218491


namespace find_quadruples_l2184_218474

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

theorem find_quadruples :
  ∀ x y z n : ℕ, is_solution x y z n ↔ 
  (x, y, z, n) = (1, 1, 1, 2) ∨
  (x, y, z, n) = (0, 0, 1, 1) ∨
  (x, y, z, n) = (0, 1, 0, 1) ∨
  (x, y, z, n) = (1, 0, 0, 1) ∨
  (x, y, z, n) = (0, 0, 0, 0) :=
by
  sorry

end find_quadruples_l2184_218474


namespace train_speed_l2184_218423

def train_length : ℝ := 800
def crossing_time : ℝ := 12
def expected_speed : ℝ := 66.67 

theorem train_speed (h_len : train_length = 800) (h_time : crossing_time = 12) : 
  train_length / crossing_time = expected_speed := 
by {
  sorry
}

end train_speed_l2184_218423


namespace angle_of_inclination_of_line_l2184_218437

theorem angle_of_inclination_of_line (x y : ℝ) (h : x - y - 1 = 0) : 
  ∃ α : ℝ, α = π / 4 := 
sorry

end angle_of_inclination_of_line_l2184_218437


namespace curves_intersect_condition_l2184_218409

noncomputable def curves_intersect_exactly_three_points (a : ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x^2 + y^2 = a^2) ∧ (y = x^2 + a) ∧ 
    (y = a → x = 0) ∧ 
    ((2 * a + 1 < 0) → y = -(2 * a + 1) - 1)

theorem curves_intersect_condition (a : ℝ) : 
  curves_intersect_exactly_three_points a ↔ a < -1/2 :=
sorry

end curves_intersect_condition_l2184_218409


namespace stock_yield_percentage_l2184_218433

theorem stock_yield_percentage
  (annual_dividend : ℝ)
  (market_price : ℝ)
  (face_value : ℝ)
  (yield_percentage : ℝ)
  (H1 : annual_dividend = 0.14 * face_value)
  (H2 : market_price = 175)
  (H3 : face_value = 100)
  (H4 : yield_percentage = (annual_dividend / market_price) * 100) :
  yield_percentage = 8 := sorry

end stock_yield_percentage_l2184_218433


namespace inequality_proof_l2184_218457

theorem inequality_proof {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 :=
by
  -- Proof goes here
  sorry

end inequality_proof_l2184_218457


namespace exists_function_passing_through_point_l2184_218451

-- Define the function that satisfies f(2) = 0
theorem exists_function_passing_through_point : ∃ f : ℝ → ℝ, f 2 = 0 := 
sorry

end exists_function_passing_through_point_l2184_218451


namespace total_slices_is_78_l2184_218487

-- Definitions based on conditions
def ratio_buzz_waiter (x : ℕ) : Prop := (5 * x) + (8 * x) = 78
def waiter_condition (x : ℕ) : Prop := (8 * x) - 20 = 28

-- Prove that the total number of slices is 78 given conditions
theorem total_slices_is_78 (x : ℕ) (h1 : ratio_buzz_waiter x) (h2 : waiter_condition x) : (5 * x) + (8 * x) = 78 :=
by
  sorry

end total_slices_is_78_l2184_218487


namespace solve_fractional_equation_l2184_218482

theorem solve_fractional_equation (x : ℝ) (h : (3 * x + 6) / (x ^ 2 + 5 * x - 6) = (3 - x) / (x - 1)) (hx : x ≠ 1) : x = -4 := 
sorry

end solve_fractional_equation_l2184_218482


namespace farmer_farm_size_l2184_218485

theorem farmer_farm_size 
  (sunflowers flax : ℕ)
  (h1 : flax = 80)
  (h2 : sunflowers = flax + 80) :
  (sunflowers + flax = 240) :=
by
  sorry

end farmer_farm_size_l2184_218485


namespace window_area_l2184_218496

def meter_to_feet : ℝ := 3.28084
def length_in_meters : ℝ := 2
def width_in_feet : ℝ := 15

def length_in_feet := length_in_meters * meter_to_feet
def area_in_square_feet := length_in_feet * width_in_feet

theorem window_area : area_in_square_feet = 98.4252 := 
by
  sorry

end window_area_l2184_218496


namespace factory_selection_and_probability_l2184_218436

/-- Total number of factories in districts A, B, and C --/
def factories_A := 18
def factories_B := 27
def factories_C := 18

/-- Total number of factories and sample size --/
def total_factories := factories_A + factories_B + factories_C
def sample_size := 7

/-- Number of factories selected from districts A, B, and C --/
def selected_from_A := factories_A * sample_size / total_factories
def selected_from_B := factories_B * sample_size / total_factories
def selected_from_C := factories_C * sample_size / total_factories

/-- Number of ways to choose 2 factories out of the 7 --/
noncomputable def comb_7_2 := Nat.choose 7 2

/-- Number of favorable outcomes where at least one factory comes from district A --/
noncomputable def favorable_outcomes := 11

/-- Probability that at least one of the 2 factories comes from district A --/
noncomputable def probability := favorable_outcomes / comb_7_2

theorem factory_selection_and_probability :
  selected_from_A = 2 ∧ selected_from_B = 3 ∧ selected_from_C = 2 ∧ probability = 11 / 21 := by
  sorry

end factory_selection_and_probability_l2184_218436


namespace triangle_base_and_area_l2184_218478

theorem triangle_base_and_area
  (height : ℝ)
  (h_height : height = 12)
  (height_base_ratio : ℝ)
  (h_ratio : height_base_ratio = 2 / 3) :
  ∃ (base : ℝ) (area : ℝ),
  base = height / height_base_ratio ∧
  area = base * height / 2 ∧
  base = 18 ∧
  area = 108 :=
by
  sorry

end triangle_base_and_area_l2184_218478


namespace greatest_b_l2184_218476

theorem greatest_b (b : ℝ) : (-b^2 + 9 * b - 14 ≥ 0) → b ≤ 7 := sorry

end greatest_b_l2184_218476


namespace original_number_l2184_218454

theorem original_number (N : ℕ) (h : ∃ k : ℕ, N + 1 = 9 * k) : N = 8 :=
sorry

end original_number_l2184_218454


namespace fraction_to_decimal_l2184_218407

theorem fraction_to_decimal : (17 : ℝ) / 50 = 0.34 := 
by 
  sorry

end fraction_to_decimal_l2184_218407


namespace diana_wins_l2184_218439

noncomputable def probability_diana_wins : ℚ :=
  45 / 100

theorem diana_wins (d : ℕ) (a : ℕ) (hd : 1 ≤ d ∧ d ≤ 10) (ha : 1 ≤ a ∧ a ≤ 10) :
  probability_diana_wins = 9 / 20 :=
by
  sorry

end diana_wins_l2184_218439


namespace shyne_total_plants_l2184_218479

/-- Shyne's seed packets -/
def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10

/-- Seed packets purchased by Shyne -/
def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6

/-- Total number of plants grown by Shyne -/
def total_plants : ℕ := 116

theorem shyne_total_plants :
  eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets = total_plants :=
by
  sorry

end shyne_total_plants_l2184_218479


namespace max_students_on_field_trip_l2184_218408

theorem max_students_on_field_trip 
  (bus_cost : ℕ := 100)
  (bus_capacity : ℕ := 25)
  (student_admission_cost_high : ℕ := 10)
  (student_admission_cost_low : ℕ := 8)
  (discount_threshold : ℕ := 20)
  (teacher_cost : ℕ := 0)
  (budget : ℕ := 350) :
  max_students ≤ bus_capacity ↔ bus_cost + 
  (if max_students ≥ discount_threshold then max_students * student_admission_cost_low
  else max_students * student_admission_cost_high) 
   ≤ budget := 
sorry

end max_students_on_field_trip_l2184_218408


namespace unique_integer_solution_range_l2184_218442

theorem unique_integer_solution_range (a : ℝ) : 
  (∀ x : ℝ, (2 * x + 3 > 5) ∧ (x - a ≤ 0) → (x = 2)) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end unique_integer_solution_range_l2184_218442


namespace initial_cell_count_l2184_218411

-- Defining the constants and parameters given in the problem
def doubling_time : ℕ := 20 -- minutes
def culture_time : ℕ := 240 -- minutes (4 hours converted to minutes)
def final_bacterial_cells : ℕ := 4096

-- Definition to find the number of doublings
def num_doublings (culture_time doubling_time : ℕ) : ℕ :=
  culture_time / doubling_time

-- Definition for exponential growth formula
def exponential_growth (initial_cells : ℕ) (doublings : ℕ) : ℕ :=
  initial_cells * (2 ^ doublings)

-- The main theorem to be proven
theorem initial_cell_count :
  exponential_growth 1 (num_doublings culture_time doubling_time) = final_bacterial_cells :=
  sorry

end initial_cell_count_l2184_218411


namespace solution_per_beaker_l2184_218413

theorem solution_per_beaker (solution_per_tube : ℕ) (num_tubes : ℕ) (num_beakers : ℕ)
    (h1 : solution_per_tube = 7) (h2 : num_tubes = 6) (h3 : num_beakers = 3) :
    (solution_per_tube * num_tubes) / num_beakers = 14 :=
by
  sorry

end solution_per_beaker_l2184_218413


namespace average_speed_for_trip_l2184_218449

theorem average_speed_for_trip :
  ∀ (walk_dist bike_dist drive_dist tot_dist walk_speed bike_speed drive_speed : ℝ)
  (h1 : walk_dist = 5) (h2 : bike_dist = 35) (h3 : drive_dist = 80)
  (h4 : tot_dist = 120) (h5 : walk_speed = 5) (h6 : bike_speed = 15)
  (h7 : drive_speed = 120),
  (tot_dist / (walk_dist / walk_speed + bike_dist / bike_speed + drive_dist / drive_speed)) = 30 :=
by
  intros
  sorry

end average_speed_for_trip_l2184_218449


namespace triple_solution_unique_l2184_218402

theorem triple_solution_unique (a b c n : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : 0 < n) :
  (a^2 + b^2 = n * Nat.lcm a b + n^2) ∧
  (b^2 + c^2 = n * Nat.lcm b c + n^2) ∧
  (c^2 + a^2 = n * Nat.lcm c a + n^2) →
  (a = n ∧ b = n ∧ c = n) :=
by
  sorry

end triple_solution_unique_l2184_218402


namespace initial_birds_179_l2184_218405

theorem initial_birds_179 (B : ℕ) (h1 : B + 38 = 217) : B = 179 :=
sorry

end initial_birds_179_l2184_218405


namespace initial_men_work_count_l2184_218432

-- Define conditions given in the problem
def work_rate (M : ℕ) := 1 / (40 * M)
def initial_men_can_complete_work_in_40_days (M : ℕ) : Prop := M * work_rate M * 40 = 1
def work_done_by_initial_men_in_16_days (M : ℕ) := (M * 16) * work_rate M
def remaining_work_done_by_remaining_men_in_40_days (M : ℕ) := ((M - 14) * 40) * work_rate M

-- Define the main theorem to prove
theorem initial_men_work_count (M : ℕ) :
  initial_men_can_complete_work_in_40_days M →
  work_done_by_initial_men_in_16_days M = 2 / 5 →
  3 / 5 = (remaining_work_done_by_remaining_men_in_40_days M) →
  M = 15 :=
by
  intros h_initial h_16_days h_remaining
  have rate := h_initial
  sorry

end initial_men_work_count_l2184_218432


namespace chromosome_stability_due_to_meiosis_and_fertilization_l2184_218470

/-- Definition of reducing chromosome number during meiosis -/
def meiosis_reduces_chromosome_number (n : ℕ) : ℕ := n / 2

/-- Definition of restoring chromosome number during fertilization -/
def fertilization_restores_chromosome_number (n : ℕ) : ℕ := n * 2

/-- Axiom: Sexual reproduction involves meiosis and fertilization to maintain chromosome stability -/
axiom chromosome_stability (n m : ℕ) (h1 : meiosis_reduces_chromosome_number n = m) 
  (h2 : fertilization_restores_chromosome_number m = n) : n = n

/-- Theorem statement in Lean 4: The chromosome number stability in sexually reproducing organisms is maintained due to meiosis and fertilization -/
theorem chromosome_stability_due_to_meiosis_and_fertilization 
  (n : ℕ) (h_meiosis: meiosis_reduces_chromosome_number n = n / 2) 
  (h_fertilization: fertilization_restores_chromosome_number (n / 2) = n) : 
  n = n := 
by
  apply chromosome_stability
  exact h_meiosis
  exact h_fertilization

end chromosome_stability_due_to_meiosis_and_fertilization_l2184_218470


namespace average_seeds_per_apple_l2184_218461

-- Define the problem conditions and the proof statement

theorem average_seeds_per_apple
  (A : ℕ)
  (total_seeds_requirement : ℕ := 60)
  (pear_seeds_avg : ℕ := 2)
  (grape_seeds_avg : ℕ := 3)
  (num_apples : ℕ := 4)
  (num_pears : ℕ := 3)
  (num_grapes : ℕ := 9)
  (shortfall : ℕ := 3)
  (collected_seeds : ℕ := num_apples * A + num_pears * pear_seeds_avg + num_grapes * grape_seeds_avg)
  (required_seeds : ℕ := total_seeds_requirement - shortfall) :
  collected_seeds = required_seeds → A = 6 := 
by
  sorry

end average_seeds_per_apple_l2184_218461


namespace find_c_l2184_218452

theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 5 * x + 8 * y + c = 0 ∧ x + y = 26) : c = -80 :=
sorry

end find_c_l2184_218452


namespace diagonal_of_rectangular_prism_l2184_218438

theorem diagonal_of_rectangular_prism (x y z : ℝ) (d : ℝ)
  (h_surface_area : 2 * x * y + 2 * x * z + 2 * y * z = 22)
  (h_edge_length : x + y + z = 6) :
  d = Real.sqrt 14 :=
by
  sorry

end diagonal_of_rectangular_prism_l2184_218438


namespace platform_length_605_l2184_218489

noncomputable def length_of_platform (speed_kmh : ℕ) (accel : ℚ) (t_platform : ℚ) (t_man : ℚ) (dist_man_from_platform : ℚ) : ℚ :=
  let speed_ms := (speed_kmh : ℚ) * 1000 / 3600
  let distance_man := speed_ms * t_man + 0.5 * accel * t_man^2
  let train_length := distance_man - dist_man_from_platform
  let distance_platform := speed_ms * t_platform + 0.5 * accel * t_platform^2
  distance_platform - train_length

theorem platform_length_605 :
  length_of_platform 54 0.5 40 20 5 = 605 := by
  sorry

end platform_length_605_l2184_218489


namespace min_value_expression_l2184_218460

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 4) :
  ∃ c : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y * z = 4 → 
  (2 * (x / y) + 3 * (y / z) + 4 * (z / x)) ≥ c) ∧ c = 6 :=
by
  sorry

end min_value_expression_l2184_218460


namespace sum_of_four_smallest_divisors_eq_11_l2184_218444

noncomputable def common_divisors_sum : ℤ :=
  let common_divisors := [1, 2, 3, 5, 6, 10, 15, 30]
  let smallest_four := common_divisors.take 4
  smallest_four.sum

theorem sum_of_four_smallest_divisors_eq_11 :
  common_divisors_sum = 11 := by
  sorry

end sum_of_four_smallest_divisors_eq_11_l2184_218444


namespace solve_for_z_l2184_218497

variable (z : ℂ) (i : ℂ)

theorem solve_for_z
  (h1 : 3 - 2*i*z = 7 + 4*i*z)
  (h2 : i^2 = -1) :
  z = 2*i / 3 :=
by
  sorry

end solve_for_z_l2184_218497


namespace average_water_per_day_l2184_218455

-- Define the given conditions as variables/constants
def day1 := 318
def day2 := 312
def day3_morning := 180
def day3_afternoon := 162

-- Define the total water added on day 3
def day3 := day3_morning + day3_afternoon

-- Define the total water added over three days
def total_water := day1 + day2 + day3

-- Define the number of days
def days := 3

-- The proof statement: the average water added per day is 324 liters
theorem average_water_per_day : total_water / days = 324 :=
by
  -- Placeholder for the proof
  sorry

end average_water_per_day_l2184_218455


namespace sin_cos_fourth_power_l2184_218422

theorem sin_cos_fourth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : Real.sin θ ^ 4 + Real.cos θ ^ 4 = 63 / 64 :=
by
  sorry

end sin_cos_fourth_power_l2184_218422


namespace sum_of_two_digit_factors_is_162_l2184_218418

-- Define the number
def num := 6545

-- Define the condition: num can be written as a product of two two-digit numbers
def are_two_digit_numbers (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = num

-- The theorem to prove
theorem sum_of_two_digit_factors_is_162 : ∃ a b : ℕ, are_two_digit_numbers a b ∧ a + b = 162 :=
sorry

end sum_of_two_digit_factors_is_162_l2184_218418


namespace medicine_liquid_poured_l2184_218417

theorem medicine_liquid_poured (x : ℝ) (h : 63 * (1 - x / 63) * (1 - x / 63) = 28) : x = 18 :=
by
  sorry

end medicine_liquid_poured_l2184_218417


namespace smallest_y_l2184_218498

theorem smallest_y (y : ℕ) :
  (y > 0 ∧ 800 ∣ (540 * y)) ↔ (y = 40) :=
by
  sorry

end smallest_y_l2184_218498


namespace earnings_in_total_l2184_218404

-- Defining the conditions
def hourly_wage : ℝ := 12.50
def hours_per_week : ℝ := 40
def earnings_per_widget : ℝ := 0.16
def widgets_per_week : ℝ := 1250

-- Theorem statement
theorem earnings_in_total : 
  (hours_per_week * hourly_wage) + (widgets_per_week * earnings_per_widget) = 700 := 
by
  sorry

end earnings_in_total_l2184_218404


namespace dance_boys_count_l2184_218419

theorem dance_boys_count (d b : ℕ) (h1 : b = 2 * d) (h2 : b = d - 1 + 8) : b = 14 :=
by
  -- The proof is omitted, denoted by 'sorry'
  sorry

end dance_boys_count_l2184_218419


namespace find_third_angle_of_triangle_l2184_218434

theorem find_third_angle_of_triangle (a b c : ℝ) (h₁ : a = 40) (h₂ : b = 3 * c) (h₃ : a + b + c = 180) : c = 35 := 
by sorry

end find_third_angle_of_triangle_l2184_218434


namespace polynomial_evaluation_x_eq_4_l2184_218400

theorem polynomial_evaluation_x_eq_4 : 
  (4 ^ 4 + 4 ^ 3 + 4 ^ 2 + 4 + 1 = 341) := 
by 
  sorry

end polynomial_evaluation_x_eq_4_l2184_218400


namespace brother_reading_time_l2184_218448

variable (my_time_in_hours : ℕ)
variable (speed_ratio : ℕ)

theorem brother_reading_time
  (h1 : my_time_in_hours = 3)
  (h2 : speed_ratio = 4) :
  my_time_in_hours * 60 / speed_ratio = 45 := 
by
  sorry

end brother_reading_time_l2184_218448


namespace percentage_x_equals_y_l2184_218477

theorem percentage_x_equals_y (x y z : ℝ) (p : ℝ)
    (h1 : 0.45 * z = 0.39 * y)
    (h2 : z = 0.65 * x)
    (h3 : y = (p / 100) * x) : 
    p = 75 := 
sorry

end percentage_x_equals_y_l2184_218477


namespace rectangular_x_value_l2184_218472

theorem rectangular_x_value (x : ℝ)
  (h1 : ∀ (length : ℝ), length = 4 * x)
  (h2 : ∀ (width : ℝ), width = x + 10)
  (h3 : ∀ (length width : ℝ), length * width = 2 * (2 * length + 2 * width))
  : x = (Real.sqrt 41 - 1) / 2 :=
by
  sorry

end rectangular_x_value_l2184_218472


namespace proof_a_eq_b_pow_n_l2184_218435

theorem proof_a_eq_b_pow_n 
  (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := 
by 
  sorry

end proof_a_eq_b_pow_n_l2184_218435


namespace max_value_m_l2184_218447

theorem max_value_m (m : ℝ) : 
  (¬ ∃ x : ℝ, x ≥ 3 ∧ 2 * x - 1 < m) → m ≤ 5 :=
by
  sorry

end max_value_m_l2184_218447


namespace cube_face_sharing_l2184_218415

theorem cube_face_sharing (n : ℕ) :
  (∃ W B : ℕ, (W + B = n^3) ∧ (3 * W = 3 * B) ∧ W = B ∧ W = n^3 / 2) ↔ n % 2 = 0 :=
by
  sorry

end cube_face_sharing_l2184_218415


namespace find_x_l2184_218421

noncomputable def is_solution (x : ℝ) : Prop :=
   (⌊x * ⌊x⌋⌋ = 29)

theorem find_x (x : ℝ) (h : is_solution x) : 5.8 ≤ x ∧ x < 6 :=
sorry

end find_x_l2184_218421


namespace tabitha_item_cost_l2184_218430

theorem tabitha_item_cost :
  ∀ (start_money gave_mom invest fraction_remain spend item_count remain_money item_cost : ℝ),
    start_money = 25 →
    gave_mom = 8 →
    invest = (start_money - gave_mom) / 2 →
    fraction_remain = start_money - gave_mom - invest →
    spend = fraction_remain - remain_money →
    item_count = 5 →
    remain_money = 6 →
    item_cost = spend / item_count →
    item_cost = 0.5 :=
by
  intros
  sorry

end tabitha_item_cost_l2184_218430


namespace smallest_positive_integer_l2184_218473

theorem smallest_positive_integer (n : ℕ) (h : 629 * n ≡ 1181 * n [MOD 35]) : n = 35 :=
sorry

end smallest_positive_integer_l2184_218473


namespace original_people_l2184_218424

-- Declare the original number of people in the room
variable (x : ℕ)

-- Conditions
-- One third of the people in the room left
def remaining_after_one_third_left (x : ℕ) : ℕ := (2 * x) / 3

-- One quarter of the remaining people started to dance
def dancers (remaining : ℕ) : ℕ := remaining / 4

-- Number of people not dancing
def non_dancers (remaining : ℕ) (dancers : ℕ) : ℕ := remaining - dancers

-- Given that there are 18 people not dancing
variable (remaining : ℕ) (dancers : ℕ)
axiom non_dancers_number : non_dancers remaining dancers = 18

-- Theorem to prove
theorem original_people (h_rem: remaining = remaining_after_one_third_left x) 
(h_dancers: dancers = remaining / 4) : x = 36 := by
  sorry

end original_people_l2184_218424


namespace tapA_turned_off_time_l2184_218458

noncomputable def tapA_rate := 1 / 45
noncomputable def tapB_rate := 1 / 40
noncomputable def tapB_fill_time := 23

theorem tapA_turned_off_time :
  ∃ t : ℕ, t * (tapA_rate + tapB_rate) + tapB_fill_time * tapB_rate = 1 ∧ t = 9 :=
by
  sorry

end tapA_turned_off_time_l2184_218458


namespace range_m_n_l2184_218410

noncomputable def f (m n x: ℝ) : ℝ := m * Real.exp x + x^2 + n * x

theorem range_m_n (m n: ℝ) :
  (∃ x, f m n x = 0) ∧ (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_m_n_l2184_218410


namespace correct_equation_l2184_218483

theorem correct_equation (x : ℕ) : 8 * x - 3 = 7 * x + 4 :=
by sorry

end correct_equation_l2184_218483


namespace minimum_value_fraction_l2184_218427

theorem minimum_value_fraction (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_parallel : m / (4 - n) = 1 / 2) : 
  (1 / m + 8 / n) ≥ 9 / 2 :=
by
  sorry

end minimum_value_fraction_l2184_218427


namespace star_evaluation_l2184_218420

def star (a b : ℕ) : ℕ := 3 + b^(a + 1)

theorem star_evaluation : star (star 2 3) 2 = 3 + 2^31 :=
by {
  sorry
}

end star_evaluation_l2184_218420


namespace vector_addition_l2184_218462

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 5)

-- State the theorem that we want to prove
theorem vector_addition : a + 3 • b = (-1, 18) :=
  sorry

end vector_addition_l2184_218462
