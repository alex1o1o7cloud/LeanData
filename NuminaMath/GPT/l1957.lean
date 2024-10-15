import Mathlib

namespace NUMINAMATH_GPT_sum_of_values_of_N_l1957_195704

-- Given conditions
variables (N R : ℝ)
-- Condition that needs to be checked
def condition (N R : ℝ) : Prop := N + 3 / N = R ∧ N ≠ 0

-- The statement to prove
theorem sum_of_values_of_N (N R : ℝ) (h: condition N R) : N + (3 / N) = R :=
sorry

end NUMINAMATH_GPT_sum_of_values_of_N_l1957_195704


namespace NUMINAMATH_GPT_max_value_of_m_l1957_195702

variable (m : ℝ)

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0

theorem max_value_of_m (h1 : 0 < m) (h2 : satisfies_inequality m) : m ≤ Real.exp 2 := sorry

end NUMINAMATH_GPT_max_value_of_m_l1957_195702


namespace NUMINAMATH_GPT_prism_faces_vertices_l1957_195772

theorem prism_faces_vertices {L E F V : ℕ} (hE : E = 21) (hEdges : E = 3 * L) 
    (hF : F = L + 2) (hV : V = L) : F = 9 ∧ V = 7 :=
by
  sorry

end NUMINAMATH_GPT_prism_faces_vertices_l1957_195772


namespace NUMINAMATH_GPT_min_days_equal_duties_l1957_195760

/--
Uncle Chernomor appoints 9 or 10 of the 33 warriors to duty each evening. 
Prove that the minimum number of days such that each warrior has been on duty the same number of times is 7.
-/
theorem min_days_equal_duties (k l m : ℕ) (k_nonneg : 0 ≤ k) (l_nonneg : 0 ≤ l)
  (h : 9 * k + 10 * l = 33 * m) (h_min : k + l = 7) : m = 2 :=
by 
  -- The necessary proof will go here
  sorry

end NUMINAMATH_GPT_min_days_equal_duties_l1957_195760


namespace NUMINAMATH_GPT_original_cost_price_l1957_195719

theorem original_cost_price (C : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) 
    (h3 : S_new = 1.25 * C - 14.70) (h4 : S_new = 1.04 * C) : C = 70 := 
by {
  sorry
}

end NUMINAMATH_GPT_original_cost_price_l1957_195719


namespace NUMINAMATH_GPT_rent_expense_calculation_l1957_195783

variable (S : ℝ)
variable (saved_amount : ℝ := 2160)
variable (milk_expense : ℝ := 1500)
variable (groceries_expense : ℝ := 4500)
variable (education_expense : ℝ := 2500)
variable (petrol_expense : ℝ := 2000)
variable (misc_expense : ℝ := 3940)
variable (salary_percent_saved : ℝ := 0.10)

theorem rent_expense_calculation 
  (h1 : salary_percent_saved * S = saved_amount) :
  S = 21600 → 
  0.90 * S - (milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense) = 5000 :=
by
  sorry

end NUMINAMATH_GPT_rent_expense_calculation_l1957_195783


namespace NUMINAMATH_GPT_correct_standardized_statement_l1957_195737

-- Define and state the conditions as Lean 4 definitions and propositions
structure GeometricStatement :=
  (description : String)
  (is_standardized : Prop)

def optionA : GeometricStatement := {
  description := "Line a and b intersect at point m",
  is_standardized := False -- due to use of lowercase 'm'
}

def optionB : GeometricStatement := {
  description := "Extend line AB",
  is_standardized := False -- since a line cannot be further extended
}

def optionC : GeometricStatement := {
  description := "Extend ray AO (where O is the endpoint) in the opposite direction",
  is_standardized := False -- incorrect definition of ray extension
}

def optionD : GeometricStatement := {
  description := "Extend line segment AB to C such that BC=AB",
  is_standardized := True -- correct by geometric principles
}

-- The theorem stating that option D is the correct and standardized statement
theorem correct_standardized_statement : optionD.is_standardized = True ∧
                                         optionA.is_standardized = False ∧
                                         optionB.is_standardized = False ∧
                                         optionC.is_standardized = False :=
  by sorry

end NUMINAMATH_GPT_correct_standardized_statement_l1957_195737


namespace NUMINAMATH_GPT_compute_nested_operations_l1957_195723

def operation (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem compute_nested_operations :
  operation 5 (operation 6 (operation 7 (operation 8 9))) = 3588 / 587 :=
  sorry

end NUMINAMATH_GPT_compute_nested_operations_l1957_195723


namespace NUMINAMATH_GPT_remainder_of_product_l1957_195716

theorem remainder_of_product (a b c : ℕ) (h₁ : a % 7 = 3) (h₂ : b % 7 = 4) (h₃ : c % 7 = 5) :
  (a * b * c) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_product_l1957_195716


namespace NUMINAMATH_GPT_sum_product_distinct_zero_l1957_195701

open BigOperators

theorem sum_product_distinct_zero {n : ℕ} (h : n ≥ 3) (a : Fin n → ℝ) (ha : Function.Injective a) :
  (∑ i, (a i) * ∏ j in Finset.univ \ {i}, (1 / (a i - a j))) = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_product_distinct_zero_l1957_195701


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1957_195711

theorem system1_solution (x y : ℤ) 
  (h1 : x = 2 * y - 1) 
  (h2 : 3 * x + 4 * y = 17) : 
  x = 3 ∧ y = 2 :=
by 
  sorry

theorem system2_solution (x y : ℤ) 
  (h1 : 2 * x - y = 0) 
  (h2 : 3 * x - 2 * y = 5) : 
  x = -5 ∧ y = -10 := 
by 
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1957_195711


namespace NUMINAMATH_GPT_find_x_plus_y_l1957_195722

theorem find_x_plus_y (x y : ℤ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) : x + y = -1 ∨ x + y = -5 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l1957_195722


namespace NUMINAMATH_GPT_peyton_manning_total_yards_l1957_195717

theorem peyton_manning_total_yards :
  let distance_per_throw_50F := 20
  let distance_per_throw_80F := 2 * distance_per_throw_50F
  let throws_saturday := 20
  let throws_sunday := 30
  let total_yards_saturday := distance_per_throw_50F * throws_saturday
  let total_yards_sunday := distance_per_throw_80F * throws_sunday
  total_yards_saturday + total_yards_sunday = 1600 := 
by
  sorry

end NUMINAMATH_GPT_peyton_manning_total_yards_l1957_195717


namespace NUMINAMATH_GPT_simplest_common_denominator_l1957_195727

variable (m n a : ℕ)

theorem simplest_common_denominator (h₁ : m > 0) (h₂ : n > 0) (h₃ : a > 0) :
  ∃ l : ℕ, l = 2 * a^2 := 
sorry

end NUMINAMATH_GPT_simplest_common_denominator_l1957_195727


namespace NUMINAMATH_GPT_gemstone_necklaces_count_l1957_195731

-- Conditions
def num_bead_necklaces : ℕ := 3
def price_per_necklace : ℕ := 7
def total_earnings : ℕ := 70

-- Proof Problem
theorem gemstone_necklaces_count : (total_earnings - num_bead_necklaces * price_per_necklace) / price_per_necklace = 7 := by
  sorry

end NUMINAMATH_GPT_gemstone_necklaces_count_l1957_195731


namespace NUMINAMATH_GPT_hall_ratio_l1957_195707

variable (w l : ℝ)

theorem hall_ratio
  (h1 : w * l = 200)
  (h2 : l - w = 10) :
  w / l = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_hall_ratio_l1957_195707


namespace NUMINAMATH_GPT_diagonals_in_decagon_l1957_195753

theorem diagonals_in_decagon :
  let n := 10
  let d := n * (n - 3) / 2
  d = 35 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_in_decagon_l1957_195753


namespace NUMINAMATH_GPT_celine_buys_two_laptops_l1957_195793

variable (number_of_laptops : ℕ)
variable (laptop_cost : ℕ := 600)
variable (smartphone_cost : ℕ := 400)
variable (number_of_smartphones : ℕ := 4)
variable (total_money_spent : ℕ := 3000)
variable (change_back : ℕ := 200)

def total_spent : ℕ := total_money_spent - change_back

def cost_of_laptops (n : ℕ) : ℕ := n * laptop_cost

def cost_of_smartphones (n : ℕ) : ℕ := n * smartphone_cost

theorem celine_buys_two_laptops :
  cost_of_laptops number_of_laptops + cost_of_smartphones number_of_smartphones = total_spent →
  number_of_laptops = 2 := by
  sorry

end NUMINAMATH_GPT_celine_buys_two_laptops_l1957_195793


namespace NUMINAMATH_GPT_smallest_root_of_equation_l1957_195755

theorem smallest_root_of_equation :
  let a := (x : ℝ) - 4 / 5
  let b := (x : ℝ) - 2 / 5
  let c := (x : ℝ) - 1 / 2
  (a^2 + a * b + c^2 = 0) → (x = 4 / 5 ∨ x = 14 / 15) ∧ (min (4 / 5) (14 / 15) = 14 / 15) :=
by
  sorry

end NUMINAMATH_GPT_smallest_root_of_equation_l1957_195755


namespace NUMINAMATH_GPT_g_values_l1957_195749

variable (g : ℝ → ℝ)

-- Condition: ∀ x y z ∈ ℝ, g(x^2 + y * g(z)) = x * g(x) + 2 * z * g(y)
axiom g_axiom : ∀ x y z : ℝ, g (x^2 + y * g z) = x * g x + 2 * z * g y

-- Proposition: The possible values of g(4) are 0 and 8.
theorem g_values : g 4 = 0 ∨ g 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_g_values_l1957_195749


namespace NUMINAMATH_GPT_f_strictly_decreasing_l1957_195759

-- Define the function g(x) = x^2 - 2x - 3
def g (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the function f(x) = log_{1/2}(g(x))
noncomputable def f (x : ℝ) : ℝ := Real.log (g x) / Real.log (1 / 2)

-- The problem statement to prove: f(x) is strictly decreasing on the interval (3, ∞)
theorem f_strictly_decreasing : ∀ x y : ℝ, 3 < x → x < y → f y < f x := by
  sorry

end NUMINAMATH_GPT_f_strictly_decreasing_l1957_195759


namespace NUMINAMATH_GPT_handrail_length_nearest_tenth_l1957_195786

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end NUMINAMATH_GPT_handrail_length_nearest_tenth_l1957_195786


namespace NUMINAMATH_GPT_simplify_expression_l1957_195785

theorem simplify_expression (x y : ℝ) : 
    3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := 
by 
    sorry

end NUMINAMATH_GPT_simplify_expression_l1957_195785


namespace NUMINAMATH_GPT_determine_a1_a2_a3_l1957_195738

theorem determine_a1_a2_a3 (a a1 a2 a3 : ℝ)
  (h : ∀ x : ℝ, x^3 = a + a1 * (x - 2) + a2 * (x - 2)^2 + a3 * (x - 2)^3) :
  a1 + a2 + a3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_determine_a1_a2_a3_l1957_195738


namespace NUMINAMATH_GPT_tangent_line_eq_l1957_195795

noncomputable def equation_of_tangent_line (x y : ℝ) : Prop := 
  ∃ k : ℝ, (y = k * (x - 2) + 2) ∧ 2 * x + y - 6 = 0

theorem tangent_line_eq :
  ∀ (x y : ℝ), 
    (y = 2 / (x - 1)) ∧ (∃ (a b : ℝ), (a, b) = (1, 4)) ->
    equation_of_tangent_line x y :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1957_195795


namespace NUMINAMATH_GPT_daily_average_books_l1957_195726

theorem daily_average_books (x : ℝ) (h1 : 4 * x + 1.4 * x = 216) : x = 40 :=
by 
  sorry

end NUMINAMATH_GPT_daily_average_books_l1957_195726


namespace NUMINAMATH_GPT_students_arrangement_count_l1957_195770

theorem students_arrangement_count : 
  let total_permutations := Nat.factorial 5
  let a_first_permutations := Nat.factorial 4
  let b_last_permutations := Nat.factorial 4
  let both_permutations := Nat.factorial 3
  total_permutations - a_first_permutations - b_last_permutations + both_permutations = 78 :=
by
  sorry

end NUMINAMATH_GPT_students_arrangement_count_l1957_195770


namespace NUMINAMATH_GPT_line_b_y_intercept_l1957_195781

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end NUMINAMATH_GPT_line_b_y_intercept_l1957_195781


namespace NUMINAMATH_GPT_swim_distance_l1957_195773

theorem swim_distance 
  (v c d : ℝ)
  (h₁ : c = 2)
  (h₂ : (d / (v + c) = 5))
  (h₃ : (25 / (v - c) = 5)) :
  d = 45 :=
by
  sorry

end NUMINAMATH_GPT_swim_distance_l1957_195773


namespace NUMINAMATH_GPT_bill_toilet_paper_duration_l1957_195741

variables (rolls : ℕ) (squares_per_roll : ℕ) (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ)

def total_squares (rolls squares_per_roll : ℕ) : ℕ := rolls * squares_per_roll

def squares_per_day (bathroom_visits_per_day squares_per_visit : ℕ) : ℕ := bathroom_visits_per_day * squares_per_visit

def days_supply_last (total_squares squares_per_day : ℕ) : ℕ := total_squares / squares_per_day

theorem bill_toilet_paper_duration
  (h1 : rolls = 1000)
  (h2 : squares_per_roll = 300)
  (h3 : bathroom_visits_per_day = 3)
  (h4 : squares_per_visit = 5)
  :
  days_supply_last (total_squares rolls squares_per_roll) (squares_per_day bathroom_visits_per_day squares_per_visit) = 20000 := sorry

end NUMINAMATH_GPT_bill_toilet_paper_duration_l1957_195741


namespace NUMINAMATH_GPT_game_rounds_l1957_195709

noncomputable def play_game (A B C D : ℕ) : ℕ := sorry

theorem game_rounds : play_game 16 15 14 13 = 49 :=
by
  sorry

end NUMINAMATH_GPT_game_rounds_l1957_195709


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1957_195784

theorem quadratic_has_two_distinct_real_roots : 
  ∃ α β : ℝ, (α ≠ β) ∧ (2 * α^2 - 3 * α + 1 = 0) ∧ (2 * β^2 - 3 * β + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1957_195784


namespace NUMINAMATH_GPT_Ruth_school_hours_l1957_195782

theorem Ruth_school_hours (d : ℝ) :
  0.25 * 5 * d = 10 → d = 8 :=
by
  sorry

end NUMINAMATH_GPT_Ruth_school_hours_l1957_195782


namespace NUMINAMATH_GPT_part_I_part_II_l1957_195740

variable {a b c : ℝ}
variable (habc : a ∈ Set.Ioi 0)
variable (hbbc : b ∈ Set.Ioi 0)
variable (hcbc : c ∈ Set.Ioi 0)
variable (h_sum : a + b + c = 1)

theorem part_I : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 :=
by
  sorry

theorem part_II : (a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1957_195740


namespace NUMINAMATH_GPT_probability_of_same_color_correct_l1957_195761

/-- Define events and their probabilities based on the given conditions --/
def probability_of_two_black_stones : ℚ := 1 / 7
def probability_of_two_white_stones : ℚ := 12 / 35

/-- Define the probability of drawing two stones of the same color --/
def probability_of_two_same_color_stones : ℚ :=
  probability_of_two_black_stones + probability_of_two_white_stones

theorem probability_of_same_color_correct :
  probability_of_two_same_color_stones = 17 / 35 :=
by
  -- We only set up the theorem, the proof is not considered here
  sorry

end NUMINAMATH_GPT_probability_of_same_color_correct_l1957_195761


namespace NUMINAMATH_GPT_find_m_n_l1957_195788

theorem find_m_n (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) (h_div : (m^3 + n^3) ∣ (m^2 + 20 * m * n + n^2)) :
  (m, n) ∈ [(1, 2), (2, 1), (2, 3), (3, 2), (1, 5), (5, 1)] :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l1957_195788


namespace NUMINAMATH_GPT_sequence_formula_l1957_195706

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n ≥ 2, a n = 2 * a (n - 1) + 1) :
  ∀ n : ℕ, a n = 2 ^ n - 1 :=
sorry

end NUMINAMATH_GPT_sequence_formula_l1957_195706


namespace NUMINAMATH_GPT_heartsuit_ratio_l1957_195703

def heartsuit (n m : ℕ) : ℕ := n^4 * m^3

theorem heartsuit_ratio :
  (heartsuit 2 4) / (heartsuit 4 2) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_heartsuit_ratio_l1957_195703


namespace NUMINAMATH_GPT_rectangle_area_l1957_195732

theorem rectangle_area (r length width : ℝ) (h_ratio : length = 3 * width) (h_incircle : width = 2 * r) (h_r : r = 7) : length * width = 588 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1957_195732


namespace NUMINAMATH_GPT_largest_fraction_among_given_l1957_195715

theorem largest_fraction_among_given (f1 f2 f3 f4 f5 : ℚ)
  (h1 : f1 = 2/5) 
  (h2 : f2 = 4/9) 
  (h3 : f3 = 7/15) 
  (h4 : f4 = 11/18) 
  (h5 : f5 = 16/35) 
  : f1 < f4 ∧ f2 < f4 ∧ f3 < f4 ∧ f5 < f4 :=
by
  sorry

end NUMINAMATH_GPT_largest_fraction_among_given_l1957_195715


namespace NUMINAMATH_GPT_bankers_discount_problem_l1957_195718

theorem bankers_discount_problem
  (BD : ℚ) (TD : ℚ) (SD : ℚ)
  (h1 : BD = 36)
  (h2 : TD = 30)
  (h3 : BD = TD + TD^2 / SD) :
  SD = 150 := 
sorry

end NUMINAMATH_GPT_bankers_discount_problem_l1957_195718


namespace NUMINAMATH_GPT_audio_cassettes_in_first_set_l1957_195771

theorem audio_cassettes_in_first_set (A V : ℝ) (num_audio_cassettes : ℝ) : 
  (V = 300) → (A * num_audio_cassettes + 3 * V = 1110) → (5 * A + 4 * V = 1350) → (A = 30) → (num_audio_cassettes = 7) := 
by
  intros hV hCond1 hCond2 hA
  sorry

end NUMINAMATH_GPT_audio_cassettes_in_first_set_l1957_195771


namespace NUMINAMATH_GPT_function_domain_l1957_195754

theorem function_domain (x : ℝ) :
  (x - 3 > 0) ∧ (5 - x ≥ 0) ↔ (3 < x ∧ x ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_function_domain_l1957_195754


namespace NUMINAMATH_GPT_evaluate_expression_l1957_195763

theorem evaluate_expression (a b c : ℤ) 
  (h1 : c = b - 12) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  ((a + 3) / (a + 2) * (b + 1) / (b - 3) * (c + 10) / (c + 7) = 10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1957_195763


namespace NUMINAMATH_GPT_customers_who_bought_four_paintings_each_l1957_195746

/-- Tracy's art fair conditions:
- 20 people came to look at the art
- Four customers bought two paintings each
- Twelve customers bought one painting each
- Tracy sold a total of 36 paintings

We need to prove the number of customers who bought four paintings each. -/
theorem customers_who_bought_four_paintings_each:
  let total_customers := 20
  let customers_bought_two_paintings := 4
  let customers_bought_one_painting := 12
  let total_paintings_sold := 36
  let paintings_per_customer_buying_two := 2
  let paintings_per_customer_buying_one := 1
  let paintings_per_customer_buying_four := 4
  (customers_bought_two_paintings * paintings_per_customer_buying_two +
   customers_bought_one_painting * paintings_per_customer_buying_one +
   x * paintings_per_customer_buying_four = total_paintings_sold) →
  (customers_bought_two_paintings + customers_bought_one_painting + x = total_customers) →
  x = 4 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_customers_who_bought_four_paintings_each_l1957_195746


namespace NUMINAMATH_GPT_sufficient_condition_l1957_195797

-- Definitions:
-- 1. Arithmetic sequence with first term a_1 and common difference d
-- 2. Define the sum of the first n terms of the arithmetic sequence

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + n * d

def sum_first_n_terms (a_1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a_1 + (n - 1) * d) / 2

-- Conditions given in the problem:
-- Let a_6 = a_1 + 5d
-- Let a_7 = a_1 + 6d
-- Condition p: a_6 + a_7 > 0

def p (a_1 d : ℤ) : Prop := a_1 + 5 * d + a_1 + 6 * d > 0

-- Sum of first 9 terms S_9 and first 3 terms S_3
-- Condition q: S_9 >= S_3

def q (a_1 d : ℤ) : Prop := sum_first_n_terms a_1 d 9 ≥ sum_first_n_terms a_1 d 3

-- The statement to prove:
theorem sufficient_condition (a_1 d : ℤ) : (p a_1 d) -> (q a_1 d) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_l1957_195797


namespace NUMINAMATH_GPT_grazing_months_of_B_l1957_195739

variable (A_cows A_months C_cows C_months D_cows D_months A_rent total_rent : ℕ)
variable (B_cows x : ℕ)

theorem grazing_months_of_B
  (hA_cows : A_cows = 24)
  (hA_months : A_months = 3)
  (hC_cows : C_cows = 35)
  (hC_months : C_months = 4)
  (hD_cows : D_cows = 21)
  (hD_months : D_months = 3)
  (hA_rent : A_rent = 1440)
  (htotal_rent : total_rent = 6500)
  (hB_cows : B_cows = 10) :
  x = 5 := 
sorry

end NUMINAMATH_GPT_grazing_months_of_B_l1957_195739


namespace NUMINAMATH_GPT_remainder_equivalence_l1957_195735

theorem remainder_equivalence (x y q r : ℕ) (hxy : x = q * y + r) (hy_pos : 0 < y) (h_r : 0 ≤ r ∧ r < y) : 
  ((x - 3 * q * y) % y) = r := 
by 
  sorry

end NUMINAMATH_GPT_remainder_equivalence_l1957_195735


namespace NUMINAMATH_GPT_cost_price_computer_table_l1957_195787

variable (C : ℝ) -- Cost price of the computer table
variable (S : ℝ) -- Selling price of the computer table

-- Conditions based on the problem
axiom h1 : S = 1.10 * C
axiom h2 : S = 8800

-- The theorem to be proven
theorem cost_price_computer_table : C = 8000 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1957_195787


namespace NUMINAMATH_GPT_find_angle_l1957_195769

theorem find_angle {x : ℝ} (h1 : ∀ i, 1 ≤ i ∧ i ≤ 9 → ∃ x, x > 0) (h2 : 9 * x = 900) : x = 100 :=
  sorry

end NUMINAMATH_GPT_find_angle_l1957_195769


namespace NUMINAMATH_GPT_anton_stationary_escalator_steps_l1957_195792

theorem anton_stationary_escalator_steps
  (N : ℕ)
  (H1 : N = 30)
  (H2 : 5 * N = 150) :
  (stationary_steps : ℕ) = 50 :=
by
  sorry

end NUMINAMATH_GPT_anton_stationary_escalator_steps_l1957_195792


namespace NUMINAMATH_GPT_cubic_expression_value_l1957_195734

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3 * x - 1 = 0) : x^3 + 5 * x^2 + 5 * x + 18 = 20 :=
by
  sorry

end NUMINAMATH_GPT_cubic_expression_value_l1957_195734


namespace NUMINAMATH_GPT_cos_value_l1957_195789

variable (α : ℝ)

theorem cos_value (h : Real.sin (Real.pi / 6 + α) = 1 / 3) : Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_l1957_195789


namespace NUMINAMATH_GPT_dot_product_necessity_l1957_195748

variables (a b : ℝ → ℝ → ℝ)

def dot_product (a b : ℝ → ℝ → ℝ) (x y : ℝ) : ℝ :=
  a x y * b x y

def angle_is_acute (a b : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  0 < a x y

theorem dot_product_necessity (a b : ℝ → ℝ → ℝ) (x y : ℝ) :
  dot_product a b x y > 0 ↔ angle_is_acute a b x y :=
sorry

end NUMINAMATH_GPT_dot_product_necessity_l1957_195748


namespace NUMINAMATH_GPT_shareCoins_l1957_195700

theorem shareCoins (a b c d e d : ℝ)
  (h1 : b = a - d)
  (h2 : ((a-2*d) + b = a + (a+d) + (a+2*d)))
  (h3 : (a-2*d) + b + a + (a+d) + (a+2*d) = 5) :
  b = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_shareCoins_l1957_195700


namespace NUMINAMATH_GPT_debby_pancakes_l1957_195750

def total_pancakes (B A P : ℕ) : ℕ := B + A + P

theorem debby_pancakes : 
  total_pancakes 20 24 23 = 67 := by 
  sorry

end NUMINAMATH_GPT_debby_pancakes_l1957_195750


namespace NUMINAMATH_GPT_five_letter_word_with_at_least_one_consonant_l1957_195733

def letter_set : Set Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Set Char := {'B', 'C', 'D', 'F'}
def vowels : Set Char := {'A', 'E'}

-- Calculate the total number of 5-letter words using the letter set
def total_words : ℕ := 6^5

-- Calculate the number of 5-letter words using only vowels
def vowel_only_words : ℕ := 2^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant : ℕ := total_words - vowel_only_words

theorem five_letter_word_with_at_least_one_consonant :
  words_with_consonant = 7744 :=
by
  sorry

end NUMINAMATH_GPT_five_letter_word_with_at_least_one_consonant_l1957_195733


namespace NUMINAMATH_GPT_joes_bid_l1957_195794

/--
Nelly tells her daughter she outbid her rival Joe by paying $2000 more than thrice his bid.
Nelly got the painting for $482,000. Prove that Joe's bid was $160,000.
-/
theorem joes_bid (J : ℝ) (h1 : 482000 = 3 * J + 2000) : J = 160000 :=
by
  sorry

end NUMINAMATH_GPT_joes_bid_l1957_195794


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1957_195752

noncomputable def f (x a : ℝ) : ℝ := (x + 1) / x + Real.sin x - a^2

theorem sufficient_but_not_necessary_condition (a : ℝ) (h : a = 1) : 
  (∀ x, f x a + f (-x) a = 0) ↔ (a = 1) ∨ (a = -1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1957_195752


namespace NUMINAMATH_GPT_planes_parallel_l1957_195758

variables {a b c : Type} {α β γ : Type}
variables (h_lines : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)

-- Definitions for parallel and perpendicular relationships
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Conditions based on the propositions
variables (h1 : parallel α γ)
variables (h2 : parallel β γ)

-- Theorem to prove
theorem planes_parallel (h1: parallel α γ) (h2 : parallel β γ) : parallel α β := 
sorry

end NUMINAMATH_GPT_planes_parallel_l1957_195758


namespace NUMINAMATH_GPT_identical_cubes_probability_l1957_195778

/-- Statement of the problem -/
theorem identical_cubes_probability :
  let total_ways := 3^8 * 3^8  -- Total ways to paint two cubes
  let identical_ways := 3 + 72 + 252 + 504  -- Ways for identical appearance after rotation
  (identical_ways : ℝ) / total_ways = 1 / 51814 :=
by
  sorry

end NUMINAMATH_GPT_identical_cubes_probability_l1957_195778


namespace NUMINAMATH_GPT_track_circumference_l1957_195777

theorem track_circumference (A_speed B_speed : ℝ) (y : ℝ) (c : ℝ)
  (A_initial B_initial : ℝ := 0)
  (B_meeting_distance_A_first_meeting : ℝ := 150)
  (A_meeting_distance_B_second_meeting : ℝ := y - 150)
  (A_second_distance : ℝ := 2 * y - 90)
  (B_second_distance : ℝ := y + 90) 
  (first_meeting_eq : B_meeting_distance_A_first_meeting = 150)
  (second_meeting_eq : A_second_distance + 90 = 2 * y)
  (uniform_speed : A_speed / B_speed = (y + 90)/(2 * y - 90)) :
  c = 2 * y → c = 720 :=
by
  sorry

end NUMINAMATH_GPT_track_circumference_l1957_195777


namespace NUMINAMATH_GPT_resulting_solution_percentage_l1957_195705

theorem resulting_solution_percentage (w_original: ℝ) (w_replaced: ℝ) (c_original: ℝ) (c_new: ℝ) :
  c_original = 0.9 → w_replaced = 0.7142857142857143 → c_new = 0.2 →
  (0.2571428571428571 + 0.14285714285714285) / (0.2857142857142857 + 0.7142857142857143) * 100 = 40 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_resulting_solution_percentage_l1957_195705


namespace NUMINAMATH_GPT_angle_slope_condition_l1957_195729

theorem angle_slope_condition (α k : Real) (h₀ : k = Real.tan α) (h₁ : 0 ≤ α ∧ α < Real.pi) : 
  (α < Real.pi / 3) → (k < Real.sqrt 3) ∧ ¬((k < Real.sqrt 3) → (α < Real.pi / 3)) := 
sorry

end NUMINAMATH_GPT_angle_slope_condition_l1957_195729


namespace NUMINAMATH_GPT_john_fan_usage_per_day_l1957_195798

theorem john_fan_usage_per_day
  (power : ℕ := 75) -- fan's power in watts
  (energy_per_month_kwh : ℕ := 18) -- energy consumption per month in kWh
  (days_in_month : ℕ := 30) -- number of days in a month
  : (energy_per_month_kwh * 1000) / power / days_in_month = 8 := 
by
  sorry

end NUMINAMATH_GPT_john_fan_usage_per_day_l1957_195798


namespace NUMINAMATH_GPT_invisible_trees_in_square_l1957_195774

theorem invisible_trees_in_square (n : ℕ) : 
  ∃ (N M : ℕ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
  Nat.gcd (N + i) (M + j) ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_invisible_trees_in_square_l1957_195774


namespace NUMINAMATH_GPT_percentage_of_products_by_m1_l1957_195791

theorem percentage_of_products_by_m1
  (x : ℝ)
  (h1 : 30 / 100 > 0)
  (h2 : 3 / 100 > 0)
  (h3 : 1 / 100 > 0)
  (h4 : 7 / 100 > 0)
  (h_total_defective : 
    0.036 = 
      (0.03 * x / 100) + 
      (0.01 * 30 / 100) + 
      (0.07 * (100 - x - 30) / 100)) :
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_products_by_m1_l1957_195791


namespace NUMINAMATH_GPT_competition_problem_l1957_195728

theorem competition_problem (n : ℕ) (s : ℕ) (correct_first_12 : s = (12 * 13) / 2)
    (gain_708_if_last_12_correct : s + 708 = (n - 11) * (n + 12) / 2):
    n = 71 :=
by
  sorry

end NUMINAMATH_GPT_competition_problem_l1957_195728


namespace NUMINAMATH_GPT_jake_peaches_is_seven_l1957_195743

-- Definitions based on conditions
def steven_peaches : ℕ := 13
def jake_peaches (steven : ℕ) : ℕ := steven - 6

-- The theorem we want to prove
theorem jake_peaches_is_seven : jake_peaches steven_peaches = 7 := sorry

end NUMINAMATH_GPT_jake_peaches_is_seven_l1957_195743


namespace NUMINAMATH_GPT_sum_of_series_l1957_195708

theorem sum_of_series (h1 : 2 + 4 + 6 + 8 + 10 = 30) (h2 : 1 + 3 + 5 + 7 + 9 = 25) : 
  ((2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9)) + ((1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10)) = 61 / 30 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1957_195708


namespace NUMINAMATH_GPT_infinite_solutions_iff_m_eq_2_l1957_195745

theorem infinite_solutions_iff_m_eq_2 (m x y : ℝ) :
  (m*x + 4*y = m + 2 ∧ x + m*y = m) ↔ (m = 2) ∧ (m > 1) :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_iff_m_eq_2_l1957_195745


namespace NUMINAMATH_GPT_andrew_calculation_l1957_195765

theorem andrew_calculation (x y : ℝ) (hx : x ≠ 0) :
  0.4 * 0.5 * x = 0.2 * 0.3 * y → y = (10 / 3) * x :=
by
  sorry

end NUMINAMATH_GPT_andrew_calculation_l1957_195765


namespace NUMINAMATH_GPT_percentage_apples_sold_l1957_195757

noncomputable def original_apples : ℝ := 750
noncomputable def remaining_apples : ℝ := 300

theorem percentage_apples_sold (A P : ℝ) (h1 : A = 750) (h2 : A * (1 - P / 100) = 300) : 
  P = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_apples_sold_l1957_195757


namespace NUMINAMATH_GPT_sandwich_bread_consumption_l1957_195768

theorem sandwich_bread_consumption :
  ∀ (num_bread_per_sandwich : ℕ),
  (2 * num_bread_per_sandwich) + num_bread_per_sandwich = 6 →
  num_bread_per_sandwich = 2 := by
    intros num_bread_per_sandwich h
    sorry

end NUMINAMATH_GPT_sandwich_bread_consumption_l1957_195768


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1957_195775

theorem quadratic_inequality_solution_set :
  (∃ x : ℝ, 2 * x + 3 - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1957_195775


namespace NUMINAMATH_GPT_squirrel_burrow_has_44_walnuts_l1957_195779

def boy_squirrel_initial := 30
def boy_squirrel_gathered := 20
def boy_squirrel_dropped := 4
def boy_squirrel_hid := 8
-- "Forgets where he hid 3 of them" does not affect the main burrow

def girl_squirrel_brought := 15
def girl_squirrel_ate := 5
def girl_squirrel_gave := 4
def girl_squirrel_lost_playing := 3
def girl_squirrel_knocked := 2

def third_squirrel_gathered := 10
def third_squirrel_dropped := 1
def third_squirrel_hid := 3
def third_squirrel_returned := 6 -- Given directly instead of as a formula step; 9-3=6
def third_squirrel_gave := 1 -- Given directly as a friend

def final_walnuts := boy_squirrel_initial + boy_squirrel_gathered
                    - boy_squirrel_dropped - boy_squirrel_hid
                    + girl_squirrel_brought - girl_squirrel_ate
                    - girl_squirrel_gave - girl_squirrel_lost_playing
                    - girl_squirrel_knocked + third_squirrel_returned

theorem squirrel_burrow_has_44_walnuts :
  final_walnuts = 44 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_burrow_has_44_walnuts_l1957_195779


namespace NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_l1957_195756

theorem inequality_one_solution (x : ℝ) :
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3 / 2) :=
sorry

theorem inequality_two_solution (x : ℝ) :
  (x + 1) / (x - 2) ≤ 2 ↔ (x < 2 ∨ x ≥ 5) :=
sorry

end NUMINAMATH_GPT_inequality_one_solution_inequality_two_solution_l1957_195756


namespace NUMINAMATH_GPT_triangle_non_existence_no_solution_max_value_expression_l1957_195713

-- Define sides and angles
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Corresponding opposite sides

-- Define the triangle conditions
def triangle_sides_angles (a b c A B C : ℝ) : Prop := 
  (a^2 = (1 - Real.cos A) / (1 - Real.cos B)) ∧ 
  (b = 1) ∧ 
  -- Additional properties ensuring we have a valid triangle can be added here
  (A ≠ B) -- Non-isosceles condition (equivalent to angles being different).

-- Prove non-existence under given conditions
theorem triangle_non_existence_no_solution (h : triangle_sides_angles a b c A B C) : false := 
sorry 

-- Define the maximization problem
theorem max_value_expression (h : a^2 = (1 - Real.cos A) / (1 - Real.cos B)) : 
(∃ b c, (b = 1) → ∀ a, a > 0 → (c > 0) ∧ ((1/c) * (1/b - 1/a)) ≤ (3 - 2 * Real.sqrt 2)) := 
sorry

end NUMINAMATH_GPT_triangle_non_existence_no_solution_max_value_expression_l1957_195713


namespace NUMINAMATH_GPT_solve_for_a_l1957_195747

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem solve_for_a (a : ℝ) : E a 3 2 = E a 5 3 ↔ a = -1/16 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1957_195747


namespace NUMINAMATH_GPT_negation_of_proposition_l1957_195725

theorem negation_of_proposition (a b : ℝ) :
  ¬(a > b → 2 * a > 2 * b) ↔ (a ≤ b → 2 * a ≤ 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1957_195725


namespace NUMINAMATH_GPT_smallest_integer_for_perfect_square_l1957_195776

theorem smallest_integer_for_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^4 * 7^3 * 8^3 * 9^2
  ∃ z : ℕ, z = 70 ∧ (∃ k : ℕ, y * z = k^2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_for_perfect_square_l1957_195776


namespace NUMINAMATH_GPT_benny_days_worked_l1957_195724

/-- Benny works 3 hours a day and in total he worked for 18 hours. 
We need to prove that he worked for 6 days. -/
theorem benny_days_worked (hours_per_day : ℕ) (total_hours : ℕ)
  (h1 : hours_per_day = 3)
  (h2 : total_hours = 18) :
  total_hours / hours_per_day = 6 := 
by sorry

end NUMINAMATH_GPT_benny_days_worked_l1957_195724


namespace NUMINAMATH_GPT_negation_of_implication_l1957_195744

theorem negation_of_implication (x : ℝ) :
  (¬ (x = 0 ∨ x = 1) → x^2 - x ≠ 0) ↔ (x ≠ 0 ∧ x ≠ 1 → x^2 - x ≠ 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_implication_l1957_195744


namespace NUMINAMATH_GPT_compare_f_values_l1957_195742

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * Real.cos x

theorem compare_f_values :
  f 0 < f (-1 / 3) ∧ f (-1 / 3) < f (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_compare_f_values_l1957_195742


namespace NUMINAMATH_GPT_fx_le_1_l1957_195767

-- Statement
theorem fx_le_1 (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := 
sorry

end NUMINAMATH_GPT_fx_le_1_l1957_195767


namespace NUMINAMATH_GPT_ratio_of_pieces_l1957_195751

-- Definitions from the conditions
def total_length : ℝ := 28
def shorter_piece_length : ℝ := 8.000028571387755

-- Derived definition
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- Statement to prove the ratio
theorem ratio_of_pieces : 
  (shorter_piece_length / longer_piece_length) = 0.400000571428571 :=
by
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_ratio_of_pieces_l1957_195751


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1957_195796

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3 * x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1957_195796


namespace NUMINAMATH_GPT_gcd_of_polynomials_l1957_195762

theorem gcd_of_polynomials (b : ℤ) (h : b % 2 = 1 ∧ 8531 ∣ b) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l1957_195762


namespace NUMINAMATH_GPT_max_value_of_symmetric_function_l1957_195780

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end NUMINAMATH_GPT_max_value_of_symmetric_function_l1957_195780


namespace NUMINAMATH_GPT_initial_population_l1957_195766

theorem initial_population (P : ℝ) (h : 0.78435 * P = 4500) : P = 5738 := 
by 
  sorry

end NUMINAMATH_GPT_initial_population_l1957_195766


namespace NUMINAMATH_GPT_minimum_magnitude_l1957_195736

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem minimum_magnitude (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3 * Complex.I) = 15) :
  smallest_magnitude_z z = (768 / 265 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_minimum_magnitude_l1957_195736


namespace NUMINAMATH_GPT_factor_expression_correct_l1957_195720

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_expression_correct (a b c : ℝ) :
  factor_expression a b c = (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_correct_l1957_195720


namespace NUMINAMATH_GPT_find_a_value_l1957_195710

def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 2) * y + 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := a * x - y + 2

-- Define what it means for two lines to be not parallel
def not_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (line1 a x y ≠ 0 ∧ line2 a x y ≠ 0)

theorem find_a_value (a : ℝ) (h : not_parallel a) : a = 0 ∨ a = -3 :=
  sorry

end NUMINAMATH_GPT_find_a_value_l1957_195710


namespace NUMINAMATH_GPT_set_of_positive_reals_l1957_195799

theorem set_of_positive_reals (S : Set ℝ) (h1 : ∀ x, x ∈ S → 0 < x)
  (h2 : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S)
  (h3 : ∀ (a b : ℝ), 0 < a → a ≤ b → ∃ c d, a ≤ c ∧ c ≤ d ∧ d ≤ b ∧ ∀ x, c ≤ x ∧ x ≤ d → x ∈ S) :
  S = {x : ℝ | 0 < x} :=
sorry

end NUMINAMATH_GPT_set_of_positive_reals_l1957_195799


namespace NUMINAMATH_GPT_vector_parallel_eq_l1957_195730

theorem vector_parallel_eq (m : ℝ) : 
  let a : ℝ × ℝ := (m, 4)
  let b : ℝ × ℝ := (3, -2)
  a.1 * b.2 = a.2 * b.1 -> m = -6 := 
by 
  sorry

end NUMINAMATH_GPT_vector_parallel_eq_l1957_195730


namespace NUMINAMATH_GPT_sum_lent_out_l1957_195764

theorem sum_lent_out (P R : ℝ) (h1 : 780 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 7) / 100) : P = 684 := 
  sorry

end NUMINAMATH_GPT_sum_lent_out_l1957_195764


namespace NUMINAMATH_GPT_enchanted_creatures_gala_handshakes_l1957_195714

theorem enchanted_creatures_gala_handshakes :
  let goblins := 30
  let trolls := 20
  let goblin_handshakes := goblins * (goblins - 1) / 2
  let troll_to_goblin_handshakes := trolls * goblins
  goblin_handshakes + troll_to_goblin_handshakes = 1035 := 
by
  sorry

end NUMINAMATH_GPT_enchanted_creatures_gala_handshakes_l1957_195714


namespace NUMINAMATH_GPT_none_of_these_l1957_195712

variables (a b c d e f : Prop)

-- Given conditions
axiom condition1 : a > b → c > d
axiom condition2 : c < d → e > f

-- Invalid conclusions
theorem none_of_these :
  ¬(a < b → e > f) ∧
  ¬(e > f → a < b) ∧
  ¬(e < f → a > b) ∧
  ¬(a > b → e < f) := sorry

end NUMINAMATH_GPT_none_of_these_l1957_195712


namespace NUMINAMATH_GPT_width_minimizes_fencing_l1957_195721

-- Define the conditions for the problem
def garden_area_cond (w : ℝ) : Prop :=
  w * (w + 10) ≥ 150

-- Define the main statement to prove
theorem width_minimizes_fencing (w : ℝ) (h : w ≥ 0) : garden_area_cond w → w = 10 :=
  by
  sorry

end NUMINAMATH_GPT_width_minimizes_fencing_l1957_195721


namespace NUMINAMATH_GPT_domain_of_function_l1957_195790

theorem domain_of_function :
  {x : ℝ | x < -1 ∨ 4 ≤ x} = {x : ℝ | (x^2 - 7*x + 12) / (x^2 - 2*x - 3) ≥ 0} \ {3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1957_195790
