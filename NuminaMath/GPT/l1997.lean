import Mathlib

namespace total_number_of_animals_l1997_199753

-- Define the problem conditions
def number_of_cats : ℕ := 645
def number_of_dogs : ℕ := 567

-- State the theorem to be proved
theorem total_number_of_animals : number_of_cats + number_of_dogs = 1212 := by
  sorry

end total_number_of_animals_l1997_199753


namespace find_m_l1997_199735

-- Define the vectors and the real number m
variables {Vec : Type*} [AddCommGroup Vec] [Module ℝ Vec]
variables (e1 e2 : Vec) (m : ℝ)

-- Define the collinearity condition and non-collinearity of the basis vectors.
def non_collinear (v1 v2 : Vec) : Prop := ¬(∃ (a : ℝ), v2 = a • v1)

def collinear (v1 v2 : Vec) : Prop := ∃ (a : ℝ), v2 = a • v1

-- Given conditions
axiom e1_e2_non_collinear : non_collinear e1 e2
axiom AB_eq : ∀ (m : ℝ), Vec
axiom CB_eq : Vec

theorem find_m (h : collinear (e1 + m • e2) (e1 - e2)) : m = -1 :=
sorry

end find_m_l1997_199735


namespace max_term_in_sequence_l1997_199780

theorem max_term_in_sequence (a : ℕ → ℝ)
  (h : ∀ n, a n = (n+1) * (7/8)^n) :
  (∀ n, a n ≤ a 6 ∨ a n ≤ a 7) ∧ (a 6 = max (a 6) (a 7)) ∧ (a 7 = max (a 6) (a 7)) :=
sorry

end max_term_in_sequence_l1997_199780


namespace sam_initial_balloons_l1997_199718

theorem sam_initial_balloons:
  ∀ (S : ℕ), (S - 10 + 16 = 52) → S = 46 :=
by
  sorry

end sam_initial_balloons_l1997_199718


namespace minimum_paper_toys_is_eight_l1997_199748

noncomputable def minimum_paper_toys (s_boats: ℕ) (s_planes: ℕ) : ℕ :=
  s_boats * 8 + s_planes * 6

theorem minimum_paper_toys_is_eight :
  ∀ (s_boats s_planes : ℕ), s_boats >= 1 → minimum_paper_toys s_boats s_planes = 8 → s_planes = 0 :=
by
  intros s_boats s_planes h_boats h_eq
  have h1: s_boats * 8 + s_planes * 6 = 8 := h_eq
  sorry

end minimum_paper_toys_is_eight_l1997_199748


namespace arithmetic_sequence_problem_l1997_199794

variable {a b : ℕ → ℕ}
variable (S T : ℕ → ℕ)

-- Conditions
def condition (n : ℕ) : Prop :=
  S n / T n = (2 * n + 1) / (3 * n + 2)

-- Conjecture to prove
theorem arithmetic_sequence_problem (h : ∀ n, condition S T n) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
by
  sorry

end arithmetic_sequence_problem_l1997_199794


namespace find_p_l1997_199770

-- Definitions
variables {n : ℕ} {p : ℝ}
def X : Type := ℕ -- Assume X is ℕ-valued

-- Conditions
axiom binomial_expectation : n * p = 6
axiom binomial_variance : n * p * (1 - p) = 3

-- Question to prove
theorem find_p : p = 1 / 2 :=
by
  sorry

end find_p_l1997_199770


namespace find_interest_rate_l1997_199757

noncomputable def annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem find_interest_rate :
  annual_interest_rate 5000 5100.50 4 0.5 0.04 :=
by
  sorry

end find_interest_rate_l1997_199757


namespace determine_a_l1997_199715

noncomputable def f (x a : ℝ) := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem determine_a (a : ℝ) 
  (h₁ : ∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f x a ≤ f 0 a)
  (h₂ : f 0 a = -3) :
  a = 2 + Real.sqrt 6 := 
sorry

end determine_a_l1997_199715


namespace larger_of_two_numbers_l1997_199737

theorem larger_of_two_numbers (H : Nat := 15) (f1 : Nat := 11) (f2 : Nat := 15) :
  let lcm := H * f1 * f2;
  ∃ (A B : Nat), A = H * f1 ∧ B = H * f2 ∧ A ≤ B := by
  sorry

end larger_of_two_numbers_l1997_199737


namespace roy_age_product_l1997_199799

theorem roy_age_product (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R = K + (R - J) / 2)
  (h3 : R + 2 = 3 * (J + 2)) :
  (R + 2) * (K + 2) = 96 :=
by
  sorry

end roy_age_product_l1997_199799


namespace shooting_competition_l1997_199782

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end shooting_competition_l1997_199782


namespace number_of_correct_statements_l1997_199710

theorem number_of_correct_statements (stmt1: Prop) (stmt2: Prop) (stmt3: Prop) :
  stmt1 ∧ stmt2 ∧ stmt3 → (∀ n, n = 3) :=
by
  sorry

end number_of_correct_statements_l1997_199710


namespace total_wash_time_l1997_199707

theorem total_wash_time (clothes_time : ℕ) (towels_time : ℕ) (sheets_time : ℕ) (total_time : ℕ) 
  (h1 : clothes_time = 30) 
  (h2 : towels_time = 2 * clothes_time) 
  (h3 : sheets_time = towels_time - 15) 
  (h4 : total_time = clothes_time + towels_time + sheets_time) : 
  total_time = 135 := 
by 
  sorry

end total_wash_time_l1997_199707


namespace percentage_of_copper_in_second_alloy_l1997_199775

theorem percentage_of_copper_in_second_alloy
  (w₁ w₂ w_total : ℝ)
  (p₁ p_total : ℝ)
  (h₁ : w₁ = 66)
  (h₂ : p₁ = 0.10)
  (h₃ : w_total = 121)
  (h₄ : p_total = 0.15) :
  (w_total - w₁) * 0.21 = w_total * p_total - w₁ * p₁ := 
  sorry

end percentage_of_copper_in_second_alloy_l1997_199775


namespace wall_height_to_breadth_ratio_l1997_199749

theorem wall_height_to_breadth_ratio :
  ∀ (b : ℝ) (h : ℝ) (l : ℝ),
  b = 0.4 → h = n * b → l = 8 * h → l * b * h = 12.8 →
  n = 5 :=
by
  intros b h l hb hh hl hv
  sorry

end wall_height_to_breadth_ratio_l1997_199749


namespace function_properties_l1997_199797

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x > 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  f 1 = 3 →
  Set.range f = Set.Icc (-6) 6 :=
sorry

end function_properties_l1997_199797


namespace Jose_Raju_Work_Together_l1997_199778

-- Definitions for the conditions
def JoseWorkRate : ℚ := 1 / 10
def RajuWorkRate : ℚ := 1 / 40
def CombinedWorkRate : ℚ := JoseWorkRate + RajuWorkRate

-- Theorem statement
theorem Jose_Raju_Work_Together :
  1 / CombinedWorkRate = 8 := by
    sorry

end Jose_Raju_Work_Together_l1997_199778


namespace plywood_long_side_length_l1997_199764

theorem plywood_long_side_length (L : ℕ) (h1 : 2 * (L + 5) = 22) : L = 6 :=
by
  sorry

end plywood_long_side_length_l1997_199764


namespace find_fraction_l1997_199795

theorem find_fraction
  (x : ℝ)
  (h : (x)^35 * (1/4)^18 = 1 / (2 * 10^35)) : x = 1/5 :=
by 
  sorry

end find_fraction_l1997_199795


namespace expenditure_on_house_rent_l1997_199716

theorem expenditure_on_house_rent (I : ℝ) (H1 : 0.30 * I = 300) : 0.20 * (I - 0.30 * I) = 140 :=
by
  -- Skip the proof, the statement is sufficient at this stage.
  sorry

end expenditure_on_house_rent_l1997_199716


namespace original_price_is_975_l1997_199760

variable (x : ℝ)
variable (discounted_price : ℝ := 780)
variable (discount : ℝ := 0.20)

-- The condition that Smith bought the shirt for Rs. 780 after a 20% discount
def original_price_calculation (x : ℝ) (discounted_price : ℝ) (discount : ℝ) : Prop :=
  (1 - discount) * x = discounted_price

theorem original_price_is_975 : ∃ x : ℝ, original_price_calculation x 780 0.20 ∧ x = 975 := 
by
  -- Proof will be provided here
  sorry

end original_price_is_975_l1997_199760


namespace pairs_satisfy_condition_l1997_199788

theorem pairs_satisfy_condition (a b : ℝ) :
  (∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a_int b_int : ℤ, a = a_int ∧ b = b_int)) :=
by
  sorry

end pairs_satisfy_condition_l1997_199788


namespace min_sum_a_b2_l1997_199765

theorem min_sum_a_b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) : a + b ≥ 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_a_b2_l1997_199765


namespace triangle_perimeter_range_expression_l1997_199705

-- Part 1: Prove the perimeter of △ABC
theorem triangle_perimeter (a b c : ℝ) (cosB : ℝ) (area : ℝ)
  (h1 : b^2 = a * c) (h2 : cosB = 3 / 5) (h3 : area = 2) :
  a + b + c = Real.sqrt 5 + Real.sqrt 21 :=
sorry

-- Part 2: Prove the range for the given expression
theorem range_expression (a b c : ℝ) (q : ℝ)
  (h1 : b = a * q) (h2 : c = a * q^2) :
  (Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2 :=
sorry

end triangle_perimeter_range_expression_l1997_199705


namespace geometric_sequence_sufficient_condition_l1997_199736

theorem geometric_sequence_sufficient_condition 
  (a_1 : ℝ) (q : ℝ) (h_a1 : a_1 < 0) (h_q : 0 < q ∧ q < 1) :
  ∀ n : ℕ, n > 0 -> a_1 * q^(n-1) < a_1 * q^n :=
sorry

end geometric_sequence_sufficient_condition_l1997_199736


namespace sqrt_two_minus_one_pow_zero_l1997_199728

theorem sqrt_two_minus_one_pow_zero : (Real.sqrt 2 - 1)^0 = 1 := by
  sorry

end sqrt_two_minus_one_pow_zero_l1997_199728


namespace remainder_3_pow_2023_mod_5_l1997_199796

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l1997_199796


namespace problem1_problem2_l1997_199756

def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1
def f (a b : ℝ) : ℝ := 3 * A a b + 6 * B a b

theorem problem1 (a b : ℝ) : f a b = 15 * a * b - 6 * a - 9 :=
by 
  sorry

theorem problem2 (b : ℝ) : (∀ a : ℝ, f a b = -9) → b = 2 / 5 :=
by 
  sorry

end problem1_problem2_l1997_199756


namespace chemical_transport_problem_l1997_199729

theorem chemical_transport_problem :
  (∀ (w r : ℕ), r = w + 420 →
  (900 / r) = (600 / (10 * w)) →
  w = 30 ∧ r = 450) ∧ 
  (∀ (x : ℕ), x + 450 * 3 * 2 + 60 * x ≥ 3600 → x = 15) := by
  sorry

end chemical_transport_problem_l1997_199729


namespace touching_line_eq_l1997_199700

theorem touching_line_eq (f : ℝ → ℝ) (f_def : ∀ x, f x = 3 * x^4 - 4 * x^3) : 
  ∃ l : ℝ → ℝ, (∀ x, l x = - (8 / 9) * x - (4 / 27)) ∧ 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (f x₁ = l x₁ ∧ f x₂ = l x₂) :=
by sorry

end touching_line_eq_l1997_199700


namespace roads_going_outside_city_l1997_199741

theorem roads_going_outside_city (n : ℕ)
  (h : ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 3 ∧
    (n + x1) % 2 = 0 ∧
    (n + x2) % 2 = 0 ∧
    (n + x3) % 2 = 0) :
  ∃ (x1 x2 x3 : ℕ), (x1 = 1) ∧ (x2 = 1) ∧ (x3 = 1) :=
by 
  sorry

end roads_going_outside_city_l1997_199741


namespace max_value_of_expression_l1997_199789

variables (a x1 x2 : ℝ)

theorem max_value_of_expression :
  (x1 < 0) → (0 < x2) → (∀ x, x^2 - a * x + a - 2 > 0 ↔ (x < x1) ∨ (x > x2)) →
  (x1 * x2 = a - 2) → 
  x1 + x2 + 2 / x1 + 2 / x2 ≤ 0 :=
by
  intros h1 h2 h3 h4
  -- Proof goes here
  sorry

end max_value_of_expression_l1997_199789


namespace samantha_mean_correct_l1997_199709

-- Given data: Samantha's assignment scores
def samantha_scores : List ℕ := [84, 89, 92, 88, 95, 91, 93]

-- Definition of the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

-- Prove that the arithmetic mean of Samantha's scores is 90.29
theorem samantha_mean_correct :
  arithmetic_mean samantha_scores = 90.29 := 
by
  -- The proof steps would be filled in here
  sorry

end samantha_mean_correct_l1997_199709


namespace kho_kho_only_l1997_199755

theorem kho_kho_only (kabaddi_total : ℕ) (both_games : ℕ) (total_players : ℕ) (kabaddi_only : ℕ) (kho_kho_only : ℕ) 
  (h1 : kabaddi_total = 10)
  (h2 : both_games = 5)
  (h3 : total_players = 50)
  (h4 : kabaddi_only = 10 - both_games)
  (h5 : kabaddi_only + kho_kho_only + both_games = total_players) :
  kho_kho_only = 40 :=
by
  -- Proof is not required
  sorry

end kho_kho_only_l1997_199755


namespace rachel_picked_apples_l1997_199768

-- Defining the conditions
def original_apples : ℕ := 11
def grown_apples : ℕ := 2
def apples_left : ℕ := 6

-- Defining the equation
def equation (x : ℕ) : Prop :=
  original_apples - x + grown_apples = apples_left

-- Stating the theorem
theorem rachel_picked_apples : ∃ x : ℕ, equation x ∧ x = 7 :=
by 
  -- proof skipped 
  sorry

end rachel_picked_apples_l1997_199768


namespace intersection_locus_is_vertical_line_l1997_199740

/-- 
Given \( 0 < a < b \), lines \( l \) and \( m \) are drawn through the points \( A(a, 0) \) and \( B(b, 0) \), 
respectively, such that these lines intersect the parabola \( y^2 = x \) at four distinct points 
and these four points are concyclic. 

We want to prove that the locus of the intersection point \( P \) of lines \( l \) and \( m \) 
is the vertical line \( x = \frac{a + b}{2} \).
-/
theorem intersection_locus_is_vertical_line (a b : ℝ) (h : 0 < a ∧ a < b) :
  (∃ P : ℝ × ℝ, P.fst = (a + b) / 2) := 
sorry

end intersection_locus_is_vertical_line_l1997_199740


namespace arithmetic_progression_sum_l1997_199761

-- Define the sum of the first 15 terms of the arithmetic progression
theorem arithmetic_progression_sum (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 16) :
  (15 / 2) * (2 * a + 14 * d) = 120 := by
  sorry

end arithmetic_progression_sum_l1997_199761


namespace sequence_solution_l1997_199784

-- Defining the sequence and the condition
def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 1

-- Defining the sequence formula we need to prove
def sequence_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1)

theorem sequence_solution (a S : ℕ → ℝ) (h : sequence_condition a S) :
  sequence_formula a :=
by 
  sorry

end sequence_solution_l1997_199784


namespace total_rainfall_2004_l1997_199721

theorem total_rainfall_2004 (average_rainfall_2003 : ℝ) (increase_percentage : ℝ) (months : ℝ) :
  average_rainfall_2003 = 36 →
  increase_percentage = 0.10 →
  months = 12 →
  (average_rainfall_2003 * (1 + increase_percentage) * months) = 475.2 :=
by
  -- The proof is left as an exercise
  sorry

end total_rainfall_2004_l1997_199721


namespace find_p_plus_q_l1997_199704

/--
In \(\triangle{XYZ}\), \(XY = 12\), \(\angle{X} = 45^\circ\), and \(\angle{Y} = 60^\circ\).
Let \(G, E,\) and \(L\) be points on the line \(YZ\) such that \(XG \perp YZ\), 
\(\angle{XYE} = \angle{EYX}\), and \(YL = LY\). Point \(O\) is the midpoint of 
the segment \(GL\), and point \(Q\) is on ray \(XE\) such that \(QO \perp YZ\).
Prove that \(XQ^2 = \dfrac{81}{2}\) and thus \(p + q = 83\), where \(p\) and \(q\) 
are relatively prime positive integers.
-/
theorem find_p_plus_q :
  ∃ (p q : ℕ), gcd p q = 1 ∧ XQ^2 = 81 / 2 ∧ p + q = 83 :=
sorry

end find_p_plus_q_l1997_199704


namespace minimum_value_l1997_199763

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ min_value_expr x y :=
  sorry

end minimum_value_l1997_199763


namespace natural_numbers_divisible_by_6_l1997_199733

theorem natural_numbers_divisible_by_6 :
  {n : ℕ | 2 ≤ n ∧ n ≤ 88 ∧ 6 ∣ n} = {n | n = 6 * k ∧ 1 ≤ k ∧ k ≤ 14} :=
by
  sorry

end natural_numbers_divisible_by_6_l1997_199733


namespace face_value_amount_of_bill_l1997_199723

def true_discount : ℚ := 45
def bankers_discount : ℚ := 54

theorem face_value_amount_of_bill : 
  ∃ (FV : ℚ), bankers_discount = true_discount + (true_discount * bankers_discount / FV) ∧ FV = 270 :=
by
  sorry

end face_value_amount_of_bill_l1997_199723


namespace Simon_has_72_legos_l1997_199725

theorem Simon_has_72_legos 
  (Kent_legos : ℕ)
  (h1 : Kent_legos = 40) 
  (Bruce_legos : ℕ) 
  (h2 : Bruce_legos = Kent_legos + 20) 
  (Simon_legos : ℕ) 
  (h3 : Simon_legos = Bruce_legos + (Bruce_legos/5)) :
  Simon_legos = 72 := 
  by
    -- Begin proof (not required for the problem)
    -- Proof steps would follow here
    sorry

end Simon_has_72_legos_l1997_199725


namespace fiona_pairs_l1997_199730

theorem fiona_pairs :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 15 → 45 ≤ (n * (n - 1) / 2) ∧ (n * (n - 1) / 2) ≤ 105 :=
by
  intro n
  intro h
  have h₁ : n ≥ 10 := h.left
  have h₂ : n ≤ 15 := h.right
  sorry

end fiona_pairs_l1997_199730


namespace largest_whole_number_value_l1997_199742

theorem largest_whole_number_value (n : ℕ) : 
  (1 : ℚ) / 5 + (n : ℚ) / 8 < 9 / 5 → n ≤ 12 := 
sorry

end largest_whole_number_value_l1997_199742


namespace mushroom_mistake_l1997_199745

theorem mushroom_mistake (p k v : ℝ) (hk : k = p + v - 10) (hp : p = k + v - 7) : 
  ∃ p k : ℝ, ∀ v : ℝ, (p = k + v - 7) ∧ (k = p + v - 10) → false :=
by
  sorry

end mushroom_mistake_l1997_199745


namespace value_range_abs_function_l1997_199758

theorem value_range_abs_function : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 9 → 1 ≤ (abs (x - 3) + 1) ∧ (abs (x - 3) + 1) ≤ 7 :=
by
  intro x hx
  sorry

end value_range_abs_function_l1997_199758


namespace systematic_sampling_eighth_group_number_l1997_199798

theorem systematic_sampling_eighth_group_number (total_students groups students_per_group draw_lots_first : ℕ) 
  (h_total : total_students = 480)
  (h_groups : groups = 30)
  (h_students_per_group : students_per_group = 16)
  (h_draw_lots_first : draw_lots_first = 5) : 
  (8 - 1) * students_per_group + draw_lots_first = 117 :=
by
  sorry

end systematic_sampling_eighth_group_number_l1997_199798


namespace rope_in_two_months_period_l1997_199777

theorem rope_in_two_months_period :
  let week1 := 6
  let week2 := 3 * week1
  let week3 := week2 - 4
  let week4 := - (week2 / 2)
  let week5 := week1 + 2
  let week6 := - (2 / 2)
  let week7 := 3 * (2 / 2)
  let week8 := - 10
  let total_length := (week1 + week2 + week3 + week4 + week5 + week6 + week7 + week8)
  total_length * 12 = 348
:= sorry

end rope_in_two_months_period_l1997_199777


namespace find_d_l1997_199714

theorem find_d (d : ℚ) (h_floor : ∃ x : ℤ, x^2 + 5 * x - 36 = 0 ∧ x = ⌊d⌋)
  (h_frac: ∃ y : ℚ, 3 * y^2 - 11 * y + 2 = 0 ∧ y = d - ⌊d⌋):
  d = 13 / 3 :=
by
  sorry

end find_d_l1997_199714


namespace number_of_video_cassettes_in_first_set_l1997_199720

/-- Let A be the cost of an audio cassette, and V the cost of a video cassette.
  We are given that V = 300, and we have the following conditions:
  1. 7 * A + n * V = 1110,
  2. 5 * A + 4 * V = 1350.
  Prove that n = 3, the number of video cassettes in the first set -/
theorem number_of_video_cassettes_in_first_set 
    (A V n : ℕ) 
    (hV : V = 300)
    (h1 : 7 * A + n * V = 1110)
    (h2 : 5 * A + 4 * V = 1350) : 
    n = 3 := 
sorry

end number_of_video_cassettes_in_first_set_l1997_199720


namespace sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l1997_199754

open Real

theorem sqrt_1_plus_inv_squares_4_5 :
  sqrt (1 + 1/4^2 + 1/5^2) = 1 + 1/20 :=
by
  sorry

theorem sqrt_1_plus_inv_squares_general (n : ℕ) (h : 0 < n) :
  sqrt (1 + 1/n^2 + 1/(n+1)^2) = 1 + 1/(n * (n + 1)) :=
by
  sorry

theorem sqrt_101_100_plus_1_121 :
  sqrt (101/100 + 1/121) = 1 + 1/110 :=
by
  sorry

end sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l1997_199754


namespace range_of_a_l1997_199719

theorem range_of_a (a : ℝ) :
  (∃ x_0 ∈ Set.Icc (-1 : ℝ) 1, |4^x_0 - a * 2^x_0 + 1| ≤ 2^(x_0 + 1)) →
  0 ≤ a ∧ a ≤ (9/2) :=
by
  sorry

end range_of_a_l1997_199719


namespace part1_part2_l1997_199776

def setA (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def setB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem part1 (m : ℝ) (h : m = 1) : 
  {x | x ∈ setA m} ∩ {x | x ∈ setB} = {x | 3 ≤ x ∧ x < 4} :=
by {
  sorry
}

theorem part2 (m : ℝ): 
  ({x | x ∈ setA m} ∪ {x | x ∈ setB} = {x | x ∈ setB}) ↔ (m ≥ 3 ∨ m ≤ -3) :=
by {
  sorry
}

end part1_part2_l1997_199776


namespace solve_cubic_equation_l1997_199793

variable (t : ℝ)

theorem solve_cubic_equation (x : ℝ) :
  x^3 - 2 * t * x^2 + t^3 = 0 ↔ 
  x = t ∨ x = t * (1 + Real.sqrt 5) / 2 ∨ x = t * (1 - Real.sqrt 5) / 2 :=
sorry

end solve_cubic_equation_l1997_199793


namespace sector_area_l1997_199774

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = π / 3) (hr : r = 6) : 
  1/2 * r^2 * α = 6 * π :=
by {
  sorry
}

end sector_area_l1997_199774


namespace number_is_4_l1997_199731

theorem number_is_4 (x : ℕ) (h : x + 5 = 9) : x = 4 := 
by {
  sorry
}

end number_is_4_l1997_199731


namespace radius_of_base_of_cone_l1997_199766

theorem radius_of_base_of_cone (S : ℝ) (hS : S = 9 * Real.pi)
  (H : ∃ (l r : ℝ), (Real.pi * l = 2 * Real.pi * r) ∧ S = Real.pi * r^2 + Real.pi * r * l) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_l1997_199766


namespace remainder_when_divide_by_66_l1997_199708

-- Define the conditions as predicates
def condition_1 (n : ℕ) : Prop := ∃ l : ℕ, n % 22 = 7
def condition_2 (n : ℕ) : Prop := ∃ m : ℕ, n % 33 = 18

-- Define the main theorem
theorem remainder_when_divide_by_66 (n : ℕ) (h1 : condition_1 n) (h2 : condition_2 n) : n % 66 = 51 :=
  sorry

end remainder_when_divide_by_66_l1997_199708


namespace arithmetic_sequence_a6_l1997_199773

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h₀ : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
  (h₁ : ∃ x y : ℝ, x = a 4 ∧ y = a 8 ∧ (x^2 - 4 * x - 1 = 0) ∧ (y^2 - 4 * y - 1 = 0) ∧ (x + y = 4)) :
  a 6 = 2 := 
sorry

end arithmetic_sequence_a6_l1997_199773


namespace determine_distance_l1997_199702

noncomputable def distance_formula (d a b c : ℝ) : Prop :=
  (d / a = (d - 30) / b) ∧
  (d / b = (d - 15) / c) ∧
  (d / a = (d - 40) / c)

theorem determine_distance (d a b c : ℝ) (h : distance_formula d a b c) : d = 90 :=
by {
  sorry
}

end determine_distance_l1997_199702


namespace average_birds_monday_l1997_199779

variable (M : ℕ)

def avg_birds_monday (M : ℕ) : Prop :=
  let total_sites := 5 + 5 + 10
  let total_birds := 5 * M + 5 * 5 + 10 * 8
  (total_birds = total_sites * 7)

theorem average_birds_monday (M : ℕ) (h : avg_birds_monday M) : M = 7 := by
  sorry

end average_birds_monday_l1997_199779


namespace log_abs_is_even_l1997_199790

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

noncomputable def f (x : ℝ) : ℝ := log (abs x)

theorem log_abs_is_even : is_even_function f :=
by
  sorry

end log_abs_is_even_l1997_199790


namespace candy_left_l1997_199744

theorem candy_left (d : ℕ) (s : ℕ) (ate : ℕ) (h_d : d = 32) (h_s : s = 42) (h_ate : ate = 35) : d + s - ate = 39 :=
by
  -- d, s, and ate are given as natural numbers
  -- h_d, h_s, and h_ate are the provided conditions
  -- The goal is to prove d + s - ate = 39
  sorry

end candy_left_l1997_199744


namespace cape_may_multiple_l1997_199732

theorem cape_may_multiple :
  ∃ x : ℕ, 26 = x * 7 + 5 ∧ x = 3 :=
by
  sorry

end cape_may_multiple_l1997_199732


namespace dasha_ate_one_bowl_l1997_199739

-- Define the quantities for Masha, Dasha, Glasha, and Natasha
variables (M D G N : ℕ)

-- Given conditions
def conditions : Prop :=
  (M + D + G + N = 16) ∧
  (G + N = 9) ∧
  (M > D) ∧
  (M > G) ∧
  (M > N)

-- The problem statement rewritten in Lean: Prove that given the conditions, Dasha ate 1 bowl.
theorem dasha_ate_one_bowl (h : conditions M D G N) : D = 1 :=
sorry

end dasha_ate_one_bowl_l1997_199739


namespace lesser_number_of_sum_and_difference_l1997_199726

theorem lesser_number_of_sum_and_difference (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 10) : b = 25 :=
sorry

end lesser_number_of_sum_and_difference_l1997_199726


namespace width_of_river_l1997_199783

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end width_of_river_l1997_199783


namespace original_contribution_amount_l1997_199786

theorem original_contribution_amount (F : ℕ) (N : ℕ) (C : ℕ) (A : ℕ) 
  (hF : F = 14) (hN : N = 19) (hC : C = 4) : A = 90 :=
by 
  sorry

end original_contribution_amount_l1997_199786


namespace bus_problem_l1997_199767

theorem bus_problem : ∀ before_stop after_stop : ℕ, before_stop = 41 → after_stop = 18 → before_stop - after_stop = 23 :=
by
  intros before_stop after_stop h_before h_after
  sorry

end bus_problem_l1997_199767


namespace largest_number_l1997_199762

theorem largest_number (a b c : ℕ) (h1 : c = a + 6) (h2 : b = (a + c) / 2) (h3 : a * b * c = 46332) : 
  c = 39 := 
sorry

end largest_number_l1997_199762


namespace speed_of_stream_l1997_199771

theorem speed_of_stream
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time_eq_upstream_time : downstream_distance / (boat_speed + v) = upstream_distance / (boat_speed - v)) :
  v = 8 :=
by
  let v := 8
  sorry

end speed_of_stream_l1997_199771


namespace arithmetic_mean_value_of_x_l1997_199711

theorem arithmetic_mean_value_of_x (x : ℝ) (h : (x + 10 + 20 + 3 * x + 16 + 3 * x + 6) / 5 = 30) : x = 14 := 
by 
  sorry

end arithmetic_mean_value_of_x_l1997_199711


namespace second_order_arithmetic_sequence_20th_term_l1997_199706

theorem second_order_arithmetic_sequence_20th_term :
  (∀ a : ℕ → ℕ,
    a 1 = 1 ∧
    a 2 = 4 ∧
    a 3 = 9 ∧
    a 4 = 16 ∧
    (∀ n, 2 ≤ n → a n - a (n - 1) = 2 * n - 1) →
    a 20 = 400) :=
by 
  sorry

end second_order_arithmetic_sequence_20th_term_l1997_199706


namespace multiple_of_interest_rate_l1997_199743

theorem multiple_of_interest_rate (P r : ℝ) (m : ℝ) 
  (h1 : P * r^2 = 40) 
  (h2 : P * m^2 * r^2 = 360) : 
  m = 3 :=
by
  sorry

end multiple_of_interest_rate_l1997_199743


namespace proof_problem_l1997_199750

theorem proof_problem (x y : ℚ) : 
  (x ^ 2 - 9 * y ^ 2 = 0) ∧ 
  (x + y = 1) ↔ 
  ((x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2)) :=
by
  sorry

end proof_problem_l1997_199750


namespace find_f2_l1997_199727

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l1997_199727


namespace problem_statement_l1997_199781

theorem problem_statement
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
                 (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + 2*b3*x + c3)) :
  b1 * c1 + b2 * c2 + 2 * b3 * c3 = 0 := 
sorry

end problem_statement_l1997_199781


namespace monkey_climb_time_l1997_199751

theorem monkey_climb_time : 
  ∀ (height hop slip : ℕ), 
    height = 22 ∧ hop = 3 ∧ slip = 2 → 
    ∃ (time : ℕ), time = 20 := 
by
  intros height hop slip h
  rcases h with ⟨h_height, ⟨h_hop, h_slip⟩⟩
  sorry

end monkey_climb_time_l1997_199751


namespace equations_solution_l1997_199701

-- Definition of the conditions
def equation1 := ∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)
def equation2 := ∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)

-- The main statement combining both problems
theorem equations_solution :
  (∀ x : ℝ, x^2 - 2 * x - 3 = 0 -> (x = 3 ∨ x = -1)) ∧
  (∀ x : ℝ, x * (x - 2) + x - 2 = 0 -> (x = -1 ∨ x = 2)) := by
  sorry

end equations_solution_l1997_199701


namespace solve_for_x_l1997_199722

theorem solve_for_x (x : ℝ) : (5 + x) / (8 + x) = (2 + x) / (3 + x) → x = -1 / 2 :=
by
  sorry

end solve_for_x_l1997_199722


namespace evaluate_g_at_neg_four_l1997_199713

def g (x : Int) : Int := 5 * x + 2

theorem evaluate_g_at_neg_four : g (-4) = -18 := by
  sorry

end evaluate_g_at_neg_four_l1997_199713


namespace cos_double_angle_l1997_199712

variable (θ : Real)

theorem cos_double_angle (h : ∑' n, (Real.cos θ)^(2 * n) = 7) : Real.cos (2 * θ) = 5 / 7 := 
  by sorry

end cos_double_angle_l1997_199712


namespace remaining_laps_l1997_199759

theorem remaining_laps (total_laps_friday : ℕ)
                       (total_laps_saturday : ℕ)
                       (laps_sunday_morning : ℕ)
                       (total_required_laps : ℕ)
                       (total_laps_weekend : ℕ)
                       (remaining_laps : ℕ) :
  total_laps_friday = 63 →
  total_laps_saturday = 62 →
  laps_sunday_morning = 15 →
  total_required_laps = 198 →
  total_laps_weekend = total_laps_friday + total_laps_saturday + laps_sunday_morning →
  remaining_laps = total_required_laps - total_laps_weekend →
  remaining_laps = 58 := by
  intros
  sorry

end remaining_laps_l1997_199759


namespace lcm_gcd_product_l1997_199752

theorem lcm_gcd_product (n m : ℕ) (h1 : n = 9) (h2 : m = 10) : 
  Nat.lcm n m * Nat.gcd n m = 90 := by
  sorry

end lcm_gcd_product_l1997_199752


namespace diane_bakes_gingerbreads_l1997_199769

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l1997_199769


namespace expected_reflection_value_l1997_199703

noncomputable def expected_reflections : ℝ :=
  (2 / Real.pi) *
  (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4))

theorem expected_reflection_value :
  expected_reflections = (2 / Real.pi) *
    (3 * Real.arccos (1 / 4) - Real.arcsin (3 / 4) + Real.arccos (3 / 4)) :=
by
  sorry

end expected_reflection_value_l1997_199703


namespace age_of_youngest_child_l1997_199787

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 :=
by
  sorry

end age_of_youngest_child_l1997_199787


namespace bird_families_flew_away_to_Asia_l1997_199772

-- Defining the given conditions
def Total_bird_families_flew_away_for_winter : ℕ := 118
def Bird_families_flew_away_to_Africa : ℕ := 38

-- Proving the main statement
theorem bird_families_flew_away_to_Asia : 
  (Total_bird_families_flew_away_for_winter - Bird_families_flew_away_to_Africa) = 80 :=
by
  sorry

end bird_families_flew_away_to_Asia_l1997_199772


namespace alex_lost_fish_l1997_199792

theorem alex_lost_fish (jacob_initial : ℕ) (alex_catch_ratio : ℕ) (jacob_additional : ℕ) (alex_initial : ℕ) (alex_final : ℕ) : 
  (jacob_initial = 8) → 
  (alex_catch_ratio = 7) → 
  (jacob_additional = 26) →
  (alex_initial = alex_catch_ratio * jacob_initial) →
  (alex_final = (jacob_initial + jacob_additional) - 1) → 
  alex_initial - alex_final = 23 :=
by
  intros
  sorry

end alex_lost_fish_l1997_199792


namespace union_of_sets_l1997_199734

def setA : Set ℝ := { x : ℝ | (x - 2) / (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -2 * x^2 + 7 * x + 4 > 0 }
def unionAB : Set ℝ := { x : ℝ | -1 < x ∧ x < 4 }

theorem union_of_sets :
  ∀ x : ℝ, x ∈ setA ∨ x ∈ setB ↔ x ∈ unionAB :=
by sorry

end union_of_sets_l1997_199734


namespace max_value_of_a2b3c4_l1997_199724

open Real

theorem max_value_of_a2b3c4
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a + b + c = 3) :
  a^2 * b^3 * c^4 ≤ 19683 / 472392 :=
sorry

end max_value_of_a2b3c4_l1997_199724


namespace compute_ratio_d_e_l1997_199785

open Polynomial

noncomputable def quartic_polynomial (a b c d e : ℚ) : Polynomial ℚ := 
  C a * X^4 + C b * X^3 + C c * X^2 + C d * X + C e

def roots_of_quartic (a b c d e: ℚ) : Prop :=
  (quartic_polynomial a b c d e).roots = {1, 2, 3, 5}

theorem compute_ratio_d_e (a b c d e : ℚ) 
    (h : roots_of_quartic a b c d e) :
    d / e = -61 / 30 :=
  sorry

end compute_ratio_d_e_l1997_199785


namespace number_of_possible_teams_l1997_199746

-- Definitions for the conditions
def num_goalkeepers := 3
def num_defenders := 5
def num_midfielders := 5
def num_strikers := 5

-- The number of ways to choose x from y
def choose (y x : ℕ) : ℕ := Nat.factorial y / (Nat.factorial x * Nat.factorial (y - x))

-- Main proof problem statement
theorem number_of_possible_teams :
  (choose num_goalkeepers 1) *
  (choose num_strikers 2) *
  (choose num_midfielders 4) *
  (choose (num_defenders + (num_midfielders - 4)) 4) = 2250 := by
  sorry

end number_of_possible_teams_l1997_199746


namespace max_min_difference_l1997_199738

def y (x : ℝ) : ℝ := x * abs (3 - x) - (x - 3) * abs x

theorem max_min_difference : (0 : ℝ) ≤ x → (x < 3 → y x ≤ y (3 / 4)) ∧ (x < 0 → y x = 0) ∧ (x ≥ 3 → y x = 0) → 
  (y (3 / 4) - (min (y 0) (min (y (-1)) (y 3)))) = 1.125 :=
by
  sorry

end max_min_difference_l1997_199738


namespace Megan_seashells_needed_l1997_199717

-- Let x be the number of additional seashells needed
def seashells_needed (total_seashells desired_seashells : Nat) : Nat :=
  desired_seashells - total_seashells

-- Given conditions
def current_seashells : Nat := 19
def desired_seashells : Nat := 25

-- The equivalent proof problem
theorem Megan_seashells_needed : seashells_needed current_seashells desired_seashells = 6 := by
  sorry

end Megan_seashells_needed_l1997_199717


namespace vec_subtraction_l1997_199747

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (0, 1)

theorem vec_subtraction : a - 2 • b = (-1, 0) := by
  sorry

end vec_subtraction_l1997_199747


namespace value_of_x_plus_y_l1997_199791

theorem value_of_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x / 3 = y^2) (h2 : x / 9 = 9 * y) : 
  x + y = 2214 :=
sorry

end value_of_x_plus_y_l1997_199791
