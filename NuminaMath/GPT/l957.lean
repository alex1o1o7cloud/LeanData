import Mathlib

namespace arithmetic_common_difference_l957_95758

variable {α : Type*} [LinearOrderedField α]

-- Definition of arithmetic sequence
def arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  a + (n - 1) * d

-- Definition of sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_common_difference (a10 : α) (s10 : α) (d : α) (a1 : α) :
  arithmetic_seq a1 d 10 = a10 →
  sum_arithmetic_seq a1 d 10 = s10 →
  d = 2 / 3 :=
by
  sorry

end arithmetic_common_difference_l957_95758


namespace pqrs_sum_l957_95798

/--
Given two pairs of real numbers (x, y) satisfying the equations:
1. x + y = 6
2. 2xy = 6

Prove that the solutions for x in the form x = (p ± q * sqrt(r)) / s give p + q + r + s = 11.
-/
theorem pqrs_sum : ∃ (p q r s : ℕ), (∀ (x y : ℝ), x + y = 6 ∧ 2*x*y = 6 → 
  (x = (p + q * Real.sqrt r) / s) ∨ (x = (p - q * Real.sqrt r) / s)) ∧ 
  p + q + r + s = 11 := 
sorry

end pqrs_sum_l957_95798


namespace simplify_polynomial_l957_95731

theorem simplify_polynomial (x y : ℝ) :
  (15 * x^4 * y^2 - 12 * x^2 * y^3 - 3 * x^2) / (-3 * x^2) = -5 * x^2 * y^2 + 4 * y^3 + 1 :=
by
  sorry

end simplify_polynomial_l957_95731


namespace problem_statement_l957_95709

noncomputable def count_propositions_and_true_statements 
  (statements : List String)
  (is_proposition : String → Bool)
  (is_true_proposition : String → Bool) 
  : Nat × Nat :=
  let props := statements.filter is_proposition
  let true_props := props.filter is_true_proposition
  (props.length, true_props.length)

theorem problem_statement : 
  (count_propositions_and_true_statements 
     ["Isn't an equilateral triangle an isosceles triangle?",
      "Are two lines perpendicular to the same line necessarily parallel?",
      "A number is either positive or negative",
      "What a beautiful coastal city Zhuhai is!",
      "If x + y is a rational number, then x and y are also rational numbers",
      "Construct △ABC ∼ △A₁B₁C₁"]
     (fun s => 
        s = "A number is either positive or negative" ∨ 
        s = "If x + y is a rational number, then x and y are also rational numbers")
     (fun s => false))
  = (2, 0) :=
by
  sorry

end problem_statement_l957_95709


namespace inscribed_circle_implies_rhombus_l957_95738

theorem inscribed_circle_implies_rhombus (AB : ℝ) (AD : ℝ)
  (h_parallelogram : AB = CD ∧ AD = BC) 
  (h_inscribed : AB + CD = AD + BC) : 
  AB = AD := by
  sorry

end inscribed_circle_implies_rhombus_l957_95738


namespace exam_failure_l957_95786

structure ExamData where
  max_marks : ℕ
  passing_percentage : ℚ
  secured_marks : ℕ

def passing_marks (data : ExamData) : ℚ :=
  data.passing_percentage * data.max_marks

theorem exam_failure (data : ExamData)
  (h1 : data.max_marks = 150)
  (h2 : data.passing_percentage = 40 / 100)
  (h3 : data.secured_marks = 40) :
  (passing_marks data - data.secured_marks : ℚ) = 20 := by
    sorry

end exam_failure_l957_95786


namespace find_k_minus_r_l957_95726

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end find_k_minus_r_l957_95726


namespace range_of_f_l957_95736

noncomputable def g (x : ℝ) := 15 - 2 * Real.cos (2 * x) - 4 * Real.sin x

noncomputable def f (x : ℝ) := Real.sqrt (g x ^ 2 - 245)

theorem range_of_f : (Set.range f) = Set.Icc 0 14 := sorry

end range_of_f_l957_95736


namespace k_value_l957_95742

theorem k_value (k : ℝ) (h : 10 * k * (-1)^3 - (-1) - 9 = 0) : k = -4 / 5 :=
by
  sorry

end k_value_l957_95742


namespace total_cows_l957_95754

theorem total_cows (cows : ℕ) (h1 : cows / 3 + cows / 5 + cows / 6 + 12 = cows) : cows = 40 :=
sorry

end total_cows_l957_95754


namespace inequality_solution_l957_95767

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := 3/4 ≤ x ∧ x ≤ 2

-- Theorem statement to prove the equivalence
theorem inequality_solution :
  ∀ x : ℝ, inequality x ↔ solution_set x := by
  sorry

end inequality_solution_l957_95767


namespace product_of_two_numbers_l957_95740

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 16) (h2 : x^2 + y^2 = 200) : x * y = 28 :=
sorry

end product_of_two_numbers_l957_95740


namespace lines_intersect_at_same_point_l957_95793

theorem lines_intersect_at_same_point : 
  (∃ (x y : ℝ), y = 2 * x - 1 ∧ y = -3 * x + 4 ∧ y = 4 * x + m) → m = -3 :=
by
  sorry

end lines_intersect_at_same_point_l957_95793


namespace prime_factorization_2006_expr_l957_95796

theorem prime_factorization_2006_expr :
  let a := 2006
  let b := 669
  let c := 1593
  (a^2 * (b + c) - b^2 * (c + a) + c^2 * (a - b)) =
  2 * 3 * 7 * 13 * 29 * 59 * 61 * 191 :=
by
  let a := 2006
  let b := 669
  let c := 1593
  have h1 : 2262 = b + c := by norm_num
  have h2 : 3599 = c + a := by norm_num
  have h3 : 1337 = a - b := by norm_num
  sorry

end prime_factorization_2006_expr_l957_95796


namespace multiple_choice_questions_count_l957_95703

variable (M F : ℕ)

-- Conditions
def totalQuestions := M + F = 60
def totalStudyTime := 15 * M + 25 * F = 1200

-- Statement to prove
theorem multiple_choice_questions_count (h1 : totalQuestions M F) (h2 : totalStudyTime M F) : M = 30 := by
  sorry

end multiple_choice_questions_count_l957_95703


namespace selection_methods_l957_95717

theorem selection_methods (students lectures : ℕ) (h_stu : students = 4) (h_lect : lectures = 3) : 
  (lectures ^ students) = 81 := 
by
  rw [h_stu, h_lect]
  rfl

end selection_methods_l957_95717


namespace isosceles_triangle_sum_x_l957_95772

noncomputable def sum_possible_values_of_x : ℝ :=
  let x1 : ℝ := 20
  let x2 : ℝ := 50
  let x3 : ℝ := 80
  x1 + x2 + x3

theorem isosceles_triangle_sum_x (x : ℝ) (h1 : x = 20 ∨ x = 50 ∨ x = 80) : sum_possible_values_of_x = 150 :=
  by
    sorry

end isosceles_triangle_sum_x_l957_95772


namespace marcus_saves_34_22_l957_95745

def max_spend : ℝ := 200
def shoe_price : ℝ := 120
def shoe_discount : ℝ := 0.30
def sock_price : ℝ := 25
def sock_discount : ℝ := 0.20
def shirt_price : ℝ := 55
def shirt_discount : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def calc_discounted_price (price discount : ℝ) : ℝ := price * (1 - discount)

def total_cost_before_tax : ℝ :=
  calc_discounted_price shoe_price shoe_discount +
  calc_discounted_price sock_price sock_discount +
  calc_discounted_price shirt_price shirt_discount

def sales_tax : ℝ := total_cost_before_tax * sales_tax_rate

def final_cost : ℝ := total_cost_before_tax + sales_tax

def money_saved : ℝ := max_spend - final_cost

theorem marcus_saves_34_22 :
  money_saved = 34.22 :=
by sorry

end marcus_saves_34_22_l957_95745


namespace saroj_age_proof_l957_95751

def saroj_present_age (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : ℕ :=
  sorry    -- calculation logic would be here but is not needed per instruction

noncomputable def question_conditions (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : Prop :=
  vimal_age_6_years_ago / 6 = saroj_age_6_years_ago / 5 ∧
  (vimal_age_6_years_ago + 10) / 11 = (saroj_age_6_years_ago + 10) / 10 ∧
  saroj_present_age vimal_age_6_years_ago saroj_age_6_years_ago = 16

theorem saroj_age_proof (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) :
  question_conditions vimal_age_6_years_ago saroj_age_6_years_ago :=
  sorry

end saroj_age_proof_l957_95751


namespace triangle_is_isosceles_right_l957_95761

theorem triangle_is_isosceles_right (A B C a b c : ℝ) 
  (h : a / (Real.cos A) = b / (Real.cos B) ∧ b / (Real.cos B) = c / (Real.sin C)) :
  A = π/4 ∧ B = π/4 ∧ C = π/2 := 
sorry

end triangle_is_isosceles_right_l957_95761


namespace maximum_of_fraction_l957_95787

theorem maximum_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end maximum_of_fraction_l957_95787


namespace sum_of_areas_is_858_l957_95739

def length1 : ℕ := 1
def length2 : ℕ := 9
def length3 : ℕ := 25
def length4 : ℕ := 49
def length5 : ℕ := 81
def length6 : ℕ := 121

def base_width : ℕ := 3

def area (width : ℕ) (length : ℕ) : ℕ :=
  width * length

def total_area_of_rectangles : ℕ :=
  area base_width length1 +
  area base_width length2 +
  area base_width length3 +
  area base_width length4 +
  area base_width length5 +
  area base_width length6

theorem sum_of_areas_is_858 : total_area_of_rectangles = 858 := by
  sorry

end sum_of_areas_is_858_l957_95739


namespace simplify_frac_48_72_l957_95762

theorem simplify_frac_48_72 : (48 / 72 : ℚ) = 2 / 3 :=
by
  -- In Lean, we prove the equality of the simplified fractions.
  sorry

end simplify_frac_48_72_l957_95762


namespace correct_statements_l957_95765

-- A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test.
def statement1 := false -- This statement is incorrect because this is systematic sampling, not stratified sampling.

-- In the frequency distribution histogram, the sum of the areas of all small rectangles is 1.
def statement2 := true -- This is correct.

-- In the regression line equation \(\hat{y} = 0.2x + 12\), when the variable \(x\) increases by one unit, the variable \(y\) definitely increases by 0.2 units.
def statement3 := false -- This is incorrect because y increases on average by 0.2 units, not definitely.

-- For two categorical variables \(X\) and \(Y\), calculating the statistic \(K^2\) and its observed value \(k\), the larger the observed value \(k\), the more confident we are that “X and Y are related”.
def statement4 := true -- This is correct.

-- We need to prove that the correct statements are only statement2 and statement4.
theorem correct_statements : (statement1 = false ∧ statement2 = true ∧ statement3 = false ∧ statement4 = true) → (statement2 ∧ statement4) :=
by sorry

end correct_statements_l957_95765


namespace figure_side_length_l957_95766

theorem figure_side_length (number_of_sides : ℕ) (perimeter : ℝ) (length_of_one_side : ℝ) 
  (h1 : number_of_sides = 8) (h2 : perimeter = 23.6) : length_of_one_side = 2.95 :=
by
  sorry

end figure_side_length_l957_95766


namespace Dan_picked_9_plums_l957_95725

-- Define the constants based on the problem
def M : ℕ := 4 -- Melanie's plums
def S : ℕ := 3 -- Sally's plums
def T : ℕ := 16 -- Total plums picked

-- The number of plums Dan picked
def D : ℕ := T - (M + S)

-- The theorem we want to prove
theorem Dan_picked_9_plums : D = 9 := by
  sorry

end Dan_picked_9_plums_l957_95725


namespace no_solution_exists_l957_95779

theorem no_solution_exists (x y : ℝ) :
  ¬(4 * x^2 + 4 * x * y + 19 * y^2 ≤ 2 ∧ x - y ≤ -1) :=
sorry

end no_solution_exists_l957_95779


namespace pipe_B_fill_time_l957_95708

theorem pipe_B_fill_time (T_B : ℝ) : 
  (1/3 + 1/T_B - 1/4 = 1/3) → T_B = 4 :=
sorry

end pipe_B_fill_time_l957_95708


namespace pencils_per_box_l957_95749

theorem pencils_per_box (boxes : ℕ) (total_pencils : ℕ) (h1 : boxes = 3) (h2 : total_pencils = 27) : (total_pencils / boxes) = 9 := 
by
  sorry

end pencils_per_box_l957_95749


namespace total_heads_l957_95768

theorem total_heads (h : ℕ) (c : ℕ) (total_feet : ℕ) 
  (h_count : h = 30)
  (hen_feet : h * 2 + c * 4 = total_feet)
  (total_feet_val : total_feet = 140) 
  : h + c = 50 :=
by
  sorry

end total_heads_l957_95768


namespace proof_C_l957_95716

variable {a b c : Type} [LinearOrder a] [LinearOrder b] [LinearOrder c]
variable {y : Type}

-- Definitions for parallel and perpendicular relationships
def parallel (x1 x2 : Type) : Prop := sorry
def perp (x1 x2 : Type) : Prop := sorry

theorem proof_C (a b c : Type) [LinearOrder a] [LinearOrder b] [LinearOrder c] (y : Type):
  (parallel a b ∧ parallel b c → parallel a c) ∧
  (perp a y ∧ perp b y → parallel a b) :=
by
  sorry

end proof_C_l957_95716


namespace find_origin_coordinates_l957_95747

variable (x y : ℝ)

def original_eq (x y : ℝ) := x^2 - y^2 - 2*x - 2*y - 1 = 0

def transformed_eq (x' y' : ℝ) := x'^2 - y'^2 = 1

theorem find_origin_coordinates (x y : ℝ) :
  original_eq (x - 1) (y + 1) ↔ transformed_eq x y :=
by
  sorry

end find_origin_coordinates_l957_95747


namespace ian_leftover_money_l957_95704

def ianPayments (initial: ℝ) (colin: ℝ) (helen: ℝ) (benedict: ℝ) (emmaInitial: ℝ) (interest: ℝ) (avaAmount: ℝ) (conversionRate: ℝ) : ℝ :=
  let emmaTotal := emmaInitial + (interest * emmaInitial)
  let avaTotal := (avaAmount * 0.75) * conversionRate
  initial - (colin + helen + benedict + emmaTotal + avaTotal)

theorem ian_leftover_money :
  let initial := 100
  let colin := 20
  let twice_colin := 2 * colin
  let half_helen := twice_colin / 2
  let emmaInitial := 15
  let interest := 0.10
  let avaAmount := 8
  let conversionRate := 1.20
  ianPayments initial colin twice_colin half_helen emmaInitial interest avaAmount conversionRate = -3.70
:= by
  sorry

end ian_leftover_money_l957_95704


namespace average_runs_next_10_matches_l957_95724

theorem average_runs_next_10_matches (avg_first_10 : ℕ) (avg_all_20 : ℕ) (n_matches : ℕ) (avg_next_10 : ℕ) :
  avg_first_10 = 40 ∧ avg_all_20 = 35 ∧ n_matches = 10 → avg_next_10 = 30 :=
by
  intros h
  sorry

end average_runs_next_10_matches_l957_95724


namespace min_value_of_expression_l957_95723

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 5) : 
  (1/x + 4/y + 9/z) >= 36/5 :=
sorry

end min_value_of_expression_l957_95723


namespace kate_collected_money_l957_95790

-- Define the conditions
def wand_cost : ℕ := 60
def num_wands_bought : ℕ := 3
def extra_charge : ℕ := 5
def num_wands_sold : ℕ := 2

-- Define the selling price per wand
def selling_price_per_wand : ℕ := wand_cost + extra_charge

-- Define the total amount collected from the sale
def total_collected : ℕ := num_wands_sold * selling_price_per_wand

-- Prove that the total collected is $130
theorem kate_collected_money :
  total_collected = 130 :=
sorry

end kate_collected_money_l957_95790


namespace octopus_dressing_orders_l957_95702

/-- A robotic octopus has four legs, and each leg needs to wear a glove before it can wear a boot.
    Additionally, it has two tentacles that require one bracelet each before putting anything on the legs.
    The total number of valid dressing orders is 1,286,400. -/
theorem octopus_dressing_orders : 
  ∃ (n : ℕ), n = 1286400 :=
by
  sorry

end octopus_dressing_orders_l957_95702


namespace midpoint_of_line_segment_on_hyperbola_l957_95781

def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 9 = 1

theorem midpoint_of_line_segment_on_hyperbola :
  ∃ (A B : ℝ × ℝ), hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧ ((A.1 + B.1)/2, (A.2 + B.2)/2) = (-1,-4) :=
by
  sorry

end midpoint_of_line_segment_on_hyperbola_l957_95781


namespace total_scissors_l957_95752

def initial_scissors : ℕ := 54
def added_scissors : ℕ := 22

theorem total_scissors : initial_scissors + added_scissors = 76 :=
by
  sorry

end total_scissors_l957_95752


namespace expand_expression_l957_95764

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end expand_expression_l957_95764


namespace infinite_nested_radicals_solution_l957_95750

theorem infinite_nested_radicals_solution :
  ∃ x : ℝ, 
    (∃ y z : ℝ, (y = (x * y)^(1/3) ∧ z = (x + z)^(1/3)) ∧ y = z) ∧ 
    0 < x ∧ x = (3 + Real.sqrt 5) / 2 := 
sorry

end infinite_nested_radicals_solution_l957_95750


namespace find_geometric_sequence_element_l957_95757

theorem find_geometric_sequence_element (a b c d e : ℕ) (r : ℚ)
  (h1 : 2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 100)
  (h2 : Nat.gcd a e = 1)
  (h3 : r > 1 ∧ b = a * r ∧ c = a * r^2 ∧ d = a * r^3 ∧ e = a * r^4)
  : c = 36 :=
  sorry

end find_geometric_sequence_element_l957_95757


namespace condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l957_95705

-- Definitions corresponding to each condition
def numMethods_participates_in_one_event (students events : ℕ) : ℕ :=
  events ^ students

def numMethods_event_limit_one_person (students events : ℕ) : ℕ :=
  students * (students - 1) * (students - 2)

def numMethods_person_limit_in_events (students events : ℕ) : ℕ :=
  students ^ events

-- Theorems to be proved
theorem condition1_num_registration_methods : 
  numMethods_participates_in_one_event 6 3 = 729 :=
by
  sorry

theorem condition2_num_registration_methods : 
  numMethods_event_limit_one_person 6 3 = 120 :=
by
  sorry

theorem condition3_num_registration_methods : 
  numMethods_person_limit_in_events 6 3 = 216 :=
by
  sorry

end condition1_num_registration_methods_condition2_num_registration_methods_condition3_num_registration_methods_l957_95705


namespace root_interval_sum_l957_95713

theorem root_interval_sum (a b : Int) (h1 : b - a = 1) (h2 : ∃ x, a < x ∧ x < b ∧ (x^3 - x + 1) = 0) : a + b = -3 := 
sorry

end root_interval_sum_l957_95713


namespace value_calculation_l957_95748

theorem value_calculation :
  6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 :=
by
  sorry

end value_calculation_l957_95748


namespace simplify_expr_l957_95711

theorem simplify_expr (x : ℝ) :
  2 * x^2 * (4 * x^3 - 3 * x + 5) - 4 * (x^3 - x^2 + 3 * x - 8) =
    8 * x^5 - 10 * x^3 + 14 * x^2 - 12 * x + 32 :=
by
  sorry

end simplify_expr_l957_95711


namespace chi_square_relationship_l957_95769

noncomputable def chi_square_statistic {X Y : Type*} (data : X → Y → ℝ) : ℝ := 
  sorry -- Actual definition is omitted for simplicity.

theorem chi_square_relationship (X Y : Type*) (data : X → Y → ℝ) :
  ( ∀ Χ2 : ℝ, Χ2 = chi_square_statistic data →
  (Χ2 = 0 → ∃ (credible : Prop), ¬credible)) → 
  (Χ2 > 0 → ∃ (credible : Prop), credible) :=
sorry

end chi_square_relationship_l957_95769


namespace minimum_value_MP_MF_l957_95784

noncomputable def min_value (M P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := |dist M P + dist M F|

theorem minimum_value_MP_MF :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) →
  ∀ (F : ℝ × ℝ), (F = (1, 0)) →
  ∀ (P : ℝ × ℝ), (P = (3, 1)) →
  min_value M P F = 4 :=
by
  intros M h_para F h_focus P h_fixed
  rw [min_value]
  sorry

end minimum_value_MP_MF_l957_95784


namespace geometric_series_sum_l957_95737

theorem geometric_series_sum :
  let a := -1
  let r := -3
  let n := 8
  let S := (a * (r ^ n - 1)) / (r - 1)
  S = 1640 :=
by 
  sorry 

end geometric_series_sum_l957_95737


namespace ratio_of_a_to_b_in_arithmetic_sequence_l957_95707

theorem ratio_of_a_to_b_in_arithmetic_sequence (a x b : ℝ) (h : a = 0 ∧ b = 2 * x) : (a / b) = 0 :=
  by sorry

end ratio_of_a_to_b_in_arithmetic_sequence_l957_95707


namespace valid_lineup_count_l957_95771

noncomputable def num_valid_lineups : ℕ :=
  let total_lineups := Nat.choose 18 8
  let unwanted_lineups := Nat.choose 14 4
  total_lineups - unwanted_lineups

theorem valid_lineup_count : num_valid_lineups = 42757 := by
  sorry

end valid_lineup_count_l957_95771


namespace scientific_notation_l957_95730

theorem scientific_notation (n : ℝ) (h : n = 1300000) : n = 1.3 * 10^6 :=
by {
  sorry
}

end scientific_notation_l957_95730


namespace compare_probabilities_l957_95756

noncomputable def box_bad_coin_prob_method_one : ℝ := 1 - (0.99 ^ 10)
noncomputable def box_bad_coin_prob_method_two : ℝ := 1 - ((49 / 50) ^ 5)

theorem compare_probabilities : box_bad_coin_prob_method_one < box_bad_coin_prob_method_two := by
  sorry

end compare_probabilities_l957_95756


namespace diameter_of_circle_l957_95777

theorem diameter_of_circle {a b c d e f D : ℕ} 
  (h1 : a = 15) (h2 : b = 20) (h3 : c = 25) (h4 : d = 33) (h5 : e = 56) (h6 : f = 65)
  (h_right_triangle1 : a^2 + b^2 = c^2)
  (h_right_triangle2 : d^2 + e^2 = f^2)
  (h_inscribed_triangles : true) -- This represents that both triangles are inscribed in the circle.
: D = 65 :=
sorry

end diameter_of_circle_l957_95777


namespace doll_cost_is_one_l957_95743

variable (initial_amount : ℕ) (end_amount : ℕ) (number_of_dolls : ℕ)

-- Conditions
def given_conditions : Prop :=
  initial_amount = 100 ∧
  end_amount = 97 ∧
  number_of_dolls = 3

-- Question: Proving the cost of each doll
def cost_per_doll (initial_amount end_amount number_of_dolls : ℕ) : ℕ :=
  (initial_amount - end_amount) / number_of_dolls

theorem doll_cost_is_one (h : given_conditions initial_amount end_amount number_of_dolls) :
  cost_per_doll initial_amount end_amount number_of_dolls = 1 :=
by
  sorry

end doll_cost_is_one_l957_95743


namespace bird_mammal_difference_africa_asia_l957_95763

noncomputable def bird_families_to_africa := 42
noncomputable def bird_families_to_asia := 31
noncomputable def bird_families_to_south_america := 7

noncomputable def mammal_families_to_africa := 24
noncomputable def mammal_families_to_asia := 18
noncomputable def mammal_families_to_south_america := 15

noncomputable def reptile_families_to_africa := 15
noncomputable def reptile_families_to_asia := 9
noncomputable def reptile_families_to_south_america := 5

-- Calculate the total number of families migrating to Africa, Asia, and South America
noncomputable def total_families_to_africa := bird_families_to_africa + mammal_families_to_africa + reptile_families_to_africa
noncomputable def total_families_to_asia := bird_families_to_asia + mammal_families_to_asia + reptile_families_to_asia
noncomputable def total_families_to_south_america := bird_families_to_south_america + mammal_families_to_south_america + reptile_families_to_south_america

-- Calculate the combined total of bird and mammal families going to Africa
noncomputable def bird_and_mammal_families_to_africa := bird_families_to_africa + mammal_families_to_africa

-- Difference between bird and mammal families to Africa and total animal families to Asia
noncomputable def difference := bird_and_mammal_families_to_africa - total_families_to_asia

theorem bird_mammal_difference_africa_asia : difference = 8 := 
by
  sorry

end bird_mammal_difference_africa_asia_l957_95763


namespace remainder_of_101_pow_37_mod_100_l957_95795

theorem remainder_of_101_pow_37_mod_100 : (101 ^ 37) % 100 = 1 := by
  sorry

end remainder_of_101_pow_37_mod_100_l957_95795


namespace find_range_of_m_l957_95783

variable (x m : ℝ)

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + (4 * m - 3) > 0

def proposition_q (m : ℝ) : Prop := (∀ m > 2, m + 1 / (m - 2) ≥ 4) ∧ (∃ m, m + 1 / (m - 2) = 4)

def range_m : Set ℝ := {m | 1 < m ∧ m ≤ 2} ∪ {m | m ≥ 3}

theorem find_range_of_m
  (h_p : proposition_p m ∨ ¬proposition_p m)
  (h_q : proposition_q m ∨ ¬proposition_q m)
  (h_exclusive : (proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m))
  : m ∈ range_m := sorry

end find_range_of_m_l957_95783


namespace color_opposite_orange_is_indigo_l957_95797

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo_l957_95797


namespace symmetric_about_y_axis_l957_95760

-- Condition: f is an odd function defined on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given that f is odd and F is defined as specified
theorem symmetric_about_y_axis (f : ℝ → ℝ)
  (hf : odd_function f) :
  ∀ x : ℝ, |f x| + f (|x|) = |f (-x)| + f (|x|) := 
by
  sorry

end symmetric_about_y_axis_l957_95760


namespace find_y_l957_95710

theorem find_y (x y : ℤ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 :=
sorry

end find_y_l957_95710


namespace jellybean_probability_l957_95792

theorem jellybean_probability :
  let total_ways := Nat.choose 15 4
  let red_ways := Nat.choose 5 2
  let blue_ways := Nat.choose 3 2
  let favorable_ways := red_ways * blue_ways
  let probability := favorable_ways / total_ways
  probability = (2 : ℚ) / 91 := by
  sorry

end jellybean_probability_l957_95792


namespace find_common_difference_l957_95778

def common_difference (S_odd S_even n : ℕ) (d : ℤ) : Prop :=
  S_even - S_odd = n / 2 * d

theorem find_common_difference :
  ∃ d : ℤ, common_difference 132 112 20 d ∧ d = -2 :=
  sorry

end find_common_difference_l957_95778


namespace pushups_fri_is_39_l957_95732

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l957_95732


namespace visited_iceland_l957_95799

variable (total : ℕ) (visitedNorway : ℕ) (visitedBoth : ℕ) (visitedNeither : ℕ)

theorem visited_iceland (h_total : total = 50)
                        (h_visited_norway : visitedNorway = 23)
                        (h_visited_both : visitedBoth = 21)
                        (h_visited_neither : visitedNeither = 23) :
                        (total - (visitedNorway - visitedBoth + visitedNeither) = 25) :=
  sorry

end visited_iceland_l957_95799


namespace elena_total_pens_l957_95727

theorem elena_total_pens (price_x price_y total_cost : ℝ) (num_x : ℕ) (hx1 : price_x = 4.0) (hx2 : price_y = 2.2) 
  (hx3 : total_cost = 42.0) (hx4 : num_x = 6) : 
  ∃ num_total : ℕ, num_total = 14 :=
by
  sorry

end elena_total_pens_l957_95727


namespace TJs_average_time_l957_95728

theorem TJs_average_time 
  (total_distance : ℝ) 
  (distance_half : ℝ)
  (time_first_half : ℝ) 
  (time_second_half : ℝ) 
  (H1 : total_distance = 10) 
  (H2 : distance_half = total_distance / 2) 
  (H3 : time_first_half = 20) 
  (H4 : time_second_half = 30) :
  (time_first_half + time_second_half) / total_distance = 5 :=
by
  sorry

end TJs_average_time_l957_95728


namespace neznaika_incorrect_l957_95721

-- Define the average consumption conditions
def average_consumption_december (total_consumption total_days_cons_december : ℕ) : Prop :=
  total_consumption = 10 * total_days_cons_december

def average_consumption_january (total_consumption total_days_cons_january : ℕ) : Prop :=
  total_consumption = 5 * total_days_cons_january

-- Define the claim to be disproven
def neznaika_claim (days_december_at_least_10 days_january_at_least_10 : ℕ) : Prop :=
  days_december_at_least_10 > days_january_at_least_10

-- Proof statement that the claim is incorrect
theorem neznaika_incorrect (total_days_cons_december total_days_cons_january total_consumption_dec total_consumption_jan : ℕ)
    (days_december_at_least_10 days_january_at_least_10 : ℕ)
    (h1 : average_consumption_december total_consumption_dec total_days_cons_december)
    (h2 : average_consumption_january total_consumption_jan total_days_cons_january)
    (h3 : total_days_cons_december = 31)
    (h4 : total_days_cons_january = 31)
    (h5 : days_december_at_least_10 ≤ total_days_cons_december)
    (h6 : days_january_at_least_10 ≤ total_days_cons_january)
    (h7 : days_december_at_least_10 = 1)
    (h8 : days_january_at_least_10 = 15) : 
    ¬ neznaika_claim days_december_at_least_10 days_january_at_least_10 :=
by
  sorry

end neznaika_incorrect_l957_95721


namespace total_money_made_l957_95735

-- Define the given conditions.
def total_rooms : ℕ := 260
def single_rooms : ℕ := 64
def single_room_cost : ℕ := 35
def double_room_cost : ℕ := 60

-- Define the number of double rooms.
def double_rooms : ℕ := total_rooms - single_rooms

-- Define the total money made from single and double rooms.
def money_from_single_rooms : ℕ := single_rooms * single_room_cost
def money_from_double_rooms : ℕ := double_rooms * double_room_cost

-- State the theorem we want to prove.
theorem total_money_made : 
  (money_from_single_rooms + money_from_double_rooms) = 14000 :=
  by
    sorry -- Proof is omitted.

end total_money_made_l957_95735


namespace det_calculation_l957_95719

-- Given conditions
variables (p q r s : ℤ)
variable (h1 : p * s - q * r = -3)

-- Define the matrix and determinant
def matrix_determinant (a b c d : ℤ) := a * d - b * c

-- Problem statement
theorem det_calculation : matrix_determinant (p + 2 * r) (q + 2 * s) r s = -3 :=
by
  -- Proof goes here
  sorry

end det_calculation_l957_95719


namespace total_wrappers_collected_l957_95722

theorem total_wrappers_collected :
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  Andy_wrappers + Max_wrappers + Zoe_wrappers = 74 :=
by
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  show Andy_wrappers + Max_wrappers + Zoe_wrappers = 74
  sorry

end total_wrappers_collected_l957_95722


namespace typing_speed_ratio_l957_95791

theorem typing_speed_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.25 * M = 14) : M / T = 2 :=
by
  sorry

end typing_speed_ratio_l957_95791


namespace x_intersection_difference_l957_95789

-- Define the conditions
def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5
def parabola2 (x : ℝ) : ℝ := -2 * x^2 - 4 * x + 6

theorem x_intersection_difference :
  let x₁ := (1 + Real.sqrt 6) / 5
  let x₂ := (1 - Real.sqrt 6) / 5
  (parabola1 x₁ = parabola2 x₁) → (parabola1 x₂ = parabola2 x₂) →
  (x₁ - x₂) = (2 * Real.sqrt 6) / 5 := 
by
  sorry

end x_intersection_difference_l957_95789


namespace m_plus_n_eq_five_l957_95770

theorem m_plus_n_eq_five (m n : ℝ) (h1 : m - 2 = 0) (h2 : 1 + n - 2 * m = 0) : m + n = 5 := 
  by 
  sorry

end m_plus_n_eq_five_l957_95770


namespace rotated_number_divisibility_l957_95774

theorem rotated_number_divisibility 
  (a1 a2 a3 a4 a5 a6 : ℕ) 
  (h : 7 ∣ (10^5 * a1 + 10^4 * a2 + 10^3 * a3 + 10^2 * a4 + 10 * a5 + a6)) :
  7 ∣ (10^5 * a6 + 10^4 * a1 + 10^3 * a2 + 10^2 * a3 + 10 * a4 + a5) := 
sorry

end rotated_number_divisibility_l957_95774


namespace angle_between_clock_hands_at_7_30_l957_95706

theorem angle_between_clock_hands_at_7_30:
  let clock_face := 360
  let degree_per_hour := clock_face / 12
  let hour_hand_7_oclock := 7 * degree_per_hour
  let hour_hand_7_30 := hour_hand_7_oclock + degree_per_hour / 2
  let minute_hand_30_minutes := 6 * degree_per_hour 
  let angle := hour_hand_7_30 - minute_hand_30_minutes
  angle = 45 := by sorry

end angle_between_clock_hands_at_7_30_l957_95706


namespace shuttle_speeds_l957_95755

def speed_at_altitude (speed_per_sec : ℕ) : ℕ :=
  speed_per_sec * 3600

theorem shuttle_speeds (speed_300 speed_800 avg_speed : ℕ) :
  speed_at_altitude 7 = 25200 ∧ 
  speed_at_altitude 6 = 21600 ∧ 
  avg_speed = (25200 + 21600) / 2 ∧ 
  avg_speed = 23400 := 
by
  sorry

end shuttle_speeds_l957_95755


namespace solve_for_a_l957_95785

theorem solve_for_a (x a : ℤ) (h1 : x = 3) (h2 : x + 2 * a = -1) : a = -2 :=
by
  sorry

end solve_for_a_l957_95785


namespace linear_function_difference_l957_95741

-- Define the problem in Lean.
theorem linear_function_difference (g : ℕ → ℝ) (h : ∀ x y : ℕ, g x = 3 * x + g 0) (h_condition : g 4 - g 1 = 9) : g 10 - g 1 = 27 := 
by
  sorry -- Proof is omitted.

end linear_function_difference_l957_95741


namespace store_profit_l957_95718

theorem store_profit 
  (cost_per_item : ℕ)
  (selling_price_decrease : ℕ → ℕ)
  (profit : ℤ)
  (x : ℕ) :
  cost_per_item = 40 →
  (∀ x, selling_price_decrease x = 150 - 5 * (x - 50)) →
  profit = 1500 →
  (((x = 50 ∧ selling_price_decrease 50 = 150) ∨ (x = 70 ∧ selling_price_decrease 70 = 50)) ↔ (x = 50 ∨ x = 70) ∧ profit = 1500) :=
by
  sorry

end store_profit_l957_95718


namespace failed_by_35_l957_95782

variables (M S P : ℝ)
variables (hM : M = 153.84615384615384)
variables (hS : S = 45)
variables (hP : P = 0.52 * M)

theorem failed_by_35 (hM : M = 153.84615384615384) (hS : S = 45) (hP : P = 0.52 * M) : P - S = 35 :=
by
  sorry

end failed_by_35_l957_95782


namespace inequality_proof_l957_95701

variable (a b c : ℝ)

noncomputable def specific_condition (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (1 / a + 1 / b + 1 / c = 1)

theorem inequality_proof (h : specific_condition a b c) :
  (a^a * b * c + b^b * c * a + c^c * a * b) ≥ 27 * (b * c + c * a + a * b) := 
by {
  sorry
}

end inequality_proof_l957_95701


namespace h_value_l957_95729

theorem h_value (h : ℝ) : (∃ x : ℝ, x^3 + h * x + 5 = 0 ∧ x = 3) → h = -32 / 3 := by
  sorry

end h_value_l957_95729


namespace find_n_from_binomial_condition_l957_95773

theorem find_n_from_binomial_condition (n : ℕ) (h : Nat.choose n 3 = 7 * Nat.choose n 1) : n = 43 :=
by
  -- The proof steps would be filled in here
  sorry

end find_n_from_binomial_condition_l957_95773


namespace difference_between_percentages_l957_95734

noncomputable def number : ℝ := 140

noncomputable def percentage_65 (x : ℝ) : ℝ := 0.65 * x

noncomputable def fraction_4_5 (x : ℝ) : ℝ := 0.8 * x

theorem difference_between_percentages 
  (x : ℝ) 
  (hx : x = number) 
  : (fraction_4_5 x) - (percentage_65 x) = 21 := 
by 
  sorry

end difference_between_percentages_l957_95734


namespace fraction_to_decimal_l957_95794

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l957_95794


namespace g_value_at_50_l957_95776

noncomputable def g (x : ℝ) : ℝ := (1 - x) / 2

theorem g_value_at_50 :
  (∀ x y : ℝ, 0 < x → 0 < y → 
  (x * g y - y * g x = g (x / y) + x - y)) →
  g 50 = -24.5 :=
by
  intro h
  have h_g : ∀ x : ℝ, 0 < x → g x = (1 - x) / 2 := 
    fun x x_pos => sorry -- g(x) derivation proof goes here
  exact sorry -- Final answer proof goes here

end g_value_at_50_l957_95776


namespace joan_gemstone_samples_l957_95744

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end joan_gemstone_samples_l957_95744


namespace max_basketballs_l957_95712

theorem max_basketballs (x : ℕ) (h1 : 80 * x + 50 * (40 - x) ≤ 2800) : x ≤ 26 := sorry

end max_basketballs_l957_95712


namespace census_suitable_survey_l957_95700

theorem census_suitable_survey (A B C D : Prop) : 
  D := 
sorry

end census_suitable_survey_l957_95700


namespace math_problem_l957_95715

theorem math_problem
  (N O : ℝ)
  (h₁ : 96 / 100 = |(O - 5 * N) / (5 * N)|)
  (h₂ : 5 * N ≠ 0) :
  O = 0.2 * N :=
by
  sorry

end math_problem_l957_95715


namespace n_squared_plus_one_divides_n_plus_one_l957_95733

theorem n_squared_plus_one_divides_n_plus_one (n : ℕ) (h : n^2 + 1 ∣ n + 1) : n = 1 :=
by
  sorry

end n_squared_plus_one_divides_n_plus_one_l957_95733


namespace ammonium_chloride_potassium_hydroxide_ammonia_l957_95775

theorem ammonium_chloride_potassium_hydroxide_ammonia
  (moles_KOH : ℕ) (moles_NH3 : ℕ) (moles_NH4Cl : ℕ) 
  (reaction : moles_KOH = 3 ∧ moles_NH3 = moles_KOH ∧ moles_NH4Cl >= moles_KOH) : 
  moles_NH3 = 3 :=
by
  sorry

end ammonium_chloride_potassium_hydroxide_ammonia_l957_95775


namespace range_of_m_l957_95753

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x - m * x^2

def has_two_extreme_points (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ (∀ x, x = x₁ ∨ x = x₂ ∨ f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)

theorem range_of_m :
  ∀ m : ℝ, has_two_extreme_points (m) ↔ 0 < m ∧ m < 1 / 2 := 
by
  sorry

end range_of_m_l957_95753


namespace no_such_function_exists_l957_95788

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ (∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n) :=
sorry

end no_such_function_exists_l957_95788


namespace initial_distance_between_Seonghyeon_and_Jisoo_l957_95720

theorem initial_distance_between_Seonghyeon_and_Jisoo 
  (D : ℝ)
  (h1 : 2000 = (D - 200) + 1000) : 
  D = 1200 :=
by
  sorry

end initial_distance_between_Seonghyeon_and_Jisoo_l957_95720


namespace find_C_marks_l957_95746

theorem find_C_marks :
  let english := 90
  let math := 92
  let physics := 85
  let biology := 85
  let avg_marks := 87.8
  let total_marks := avg_marks * 5
  let other_marks := english + math + physics + biology
  ∃ C : ℝ, total_marks - other_marks = C ∧ C = 87 :=
by
  sorry

end find_C_marks_l957_95746


namespace sum_of_lengths_of_legs_of_larger_triangle_l957_95780

theorem sum_of_lengths_of_legs_of_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypo_small : ℝ)
  (h_area_small : area_small = 18) (h_area_large : area_large = 288) (h_hypo_small : hypo_small = 10) :
  ∃ (sum_legs_large : ℝ), sum_legs_large = 52 :=
by
  sorry

end sum_of_lengths_of_legs_of_larger_triangle_l957_95780


namespace relationship_between_a_b_c_l957_95714

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (Real.cos (34 * Real.pi / 180) - Real.sin (34 * Real.pi / 180))
noncomputable def b : ℝ := Real.cos (50 * Real.pi / 180) * Real.cos (128 * Real.pi / 180) + Real.cos (40 * Real.pi / 180) * Real.cos (38 * Real.pi / 180)
noncomputable def c : ℝ := (1 / 2) * (Real.cos (80 * Real.pi / 180) - 2 * (Real.cos (50 * Real.pi / 180))^2 + 1)

theorem relationship_between_a_b_c : b > a ∧ a > c :=
  sorry

end relationship_between_a_b_c_l957_95714


namespace find_fraction_l957_95759

theorem find_fraction (x y : ℝ) (hx : 0 < x) (hy : x < y) (h : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt 15 / 3 :=
sorry

end find_fraction_l957_95759
