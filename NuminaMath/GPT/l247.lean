import Mathlib

namespace NUMINAMATH_GPT_repetitions_today_l247_24777

theorem repetitions_today (yesterday_reps : ℕ) (deficit : ℤ) (today_reps : ℕ) : 
  yesterday_reps = 86 ∧ deficit = -13 → 
  today_reps = yesterday_reps + deficit →
  today_reps = 73 :=
by
  intros
  sorry

end NUMINAMATH_GPT_repetitions_today_l247_24777


namespace NUMINAMATH_GPT_exact_fraction_difference_l247_24712

theorem exact_fraction_difference :
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  x - y = (2:ℚ) / 275 :=
by
  -- Definitions from conditions: x = 0.\overline{72} and y = 0.72
  let x := (8:ℚ) / 11
  let y := (18:ℚ) / 25 
  -- Goal is to prove the exact fraction difference
  show x - y = (2:ℚ) / 275
  sorry

end NUMINAMATH_GPT_exact_fraction_difference_l247_24712


namespace NUMINAMATH_GPT_complex_power_of_sum_l247_24715

theorem complex_power_of_sum (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_power_of_sum_l247_24715


namespace NUMINAMATH_GPT_calculate_value_l247_24734

theorem calculate_value : (2 / 3 : ℝ)^0 + Real.log 2 + Real.log 5 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_value_l247_24734


namespace NUMINAMATH_GPT_square_equiv_l247_24701

theorem square_equiv (x : ℝ) : 
  (7 - (x^3 - 49)^(1/3))^2 = 
  49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := 
by 
  sorry

end NUMINAMATH_GPT_square_equiv_l247_24701


namespace NUMINAMATH_GPT_sum_of_box_weights_l247_24735

theorem sum_of_box_weights (heavy_box_weight : ℚ) (difference : ℚ) 
  (h1 : heavy_box_weight = 14 / 15) (h2 : difference = 1 / 10) :
  heavy_box_weight + (heavy_box_weight - difference) = 53 / 30 := 
  by
  sorry

end NUMINAMATH_GPT_sum_of_box_weights_l247_24735


namespace NUMINAMATH_GPT_A_2013_eq_neg_1007_l247_24754

def A (n : ℕ) : ℤ :=
  (-1)^n * ((n + 1) / 2)

theorem A_2013_eq_neg_1007 : A 2013 = -1007 :=
by
  sorry

end NUMINAMATH_GPT_A_2013_eq_neg_1007_l247_24754


namespace NUMINAMATH_GPT_inequality_solution_l247_24766

theorem inequality_solution (x : ℝ) :
  (x + 2) / (x^2 + 3 * x + 10) ≥ 0 ↔ x ≥ -2 := sorry

end NUMINAMATH_GPT_inequality_solution_l247_24766


namespace NUMINAMATH_GPT_corey_gave_more_books_l247_24746

def books_given_by_mike : ℕ := 10
def total_books_received_by_lily : ℕ := 35
def books_given_by_corey : ℕ := total_books_received_by_lily - books_given_by_mike
def difference_in_books (a b : ℕ) : ℕ := a - b

theorem corey_gave_more_books :
  difference_in_books books_given_by_corey books_given_by_mike = 15 := by
sorry

end NUMINAMATH_GPT_corey_gave_more_books_l247_24746


namespace NUMINAMATH_GPT_polynomial_equivalence_l247_24748

variable (x : ℝ) -- Define variable x

-- Define the expressions.
def expr1 := (3 * x^2 + 5 * x + 8) * (x + 2)
def expr2 := (x + 2) * (x^2 + 5 * x - 72)
def expr3 := (4 * x - 15) * (x + 2) * (x + 6)

-- Define the expression to be proved.
def original_expr := expr1 - expr2 + expr3
def simplified_expr := 6 * x^3 + 21 * x^2 + 18 * x

-- The theorem to prove the equivalence of the original and simplified expressions.
theorem polynomial_equivalence : original_expr = simplified_expr :=
by sorry -- proof to be filled in

end NUMINAMATH_GPT_polynomial_equivalence_l247_24748


namespace NUMINAMATH_GPT_math_equivalence_problem_l247_24710

theorem math_equivalence_problem :
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 :=
by
  sorry

end NUMINAMATH_GPT_math_equivalence_problem_l247_24710


namespace NUMINAMATH_GPT_percentage_of_literate_females_is_32_5_l247_24797

noncomputable def percentage_literate_females (inhabitants : ℕ) (percent_male : ℝ) (percent_literate_males : ℝ) (percent_literate_total : ℝ) : ℝ :=
  let males := (percent_male / 100) * inhabitants
  let females := inhabitants - males
  let literate_males := (percent_literate_males / 100) * males
  let literate_total := (percent_literate_total / 100) * inhabitants
  let literate_females := literate_total - literate_males
  (literate_females / females) * 100

theorem percentage_of_literate_females_is_32_5 :
  percentage_literate_females 1000 60 20 25 = 32.5 := 
by 
  unfold percentage_literate_females
  sorry

end NUMINAMATH_GPT_percentage_of_literate_females_is_32_5_l247_24797


namespace NUMINAMATH_GPT_greatest_possible_value_of_median_l247_24711

-- Given conditions as definitions
variables (k m r s t : ℕ)

-- condition 1: The average (arithmetic mean) of the 5 integers is 10
def avg_is_10 : Prop := k + m + r + s + t = 50

-- condition 2: The integers are in a strictly increasing order
def increasing_order : Prop := k < m ∧ m < r ∧ r < s ∧ s < t

-- condition 3: t is 20
def t_is_20 : Prop := t = 20

-- The main statement to prove
theorem greatest_possible_value_of_median : 
  avg_is_10 k m r s t → 
  increasing_order k m r s t → 
  t_is_20 t → 
  r = 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_median_l247_24711


namespace NUMINAMATH_GPT_arc_length_l247_24765

-- Define the radius and central angle
def radius : ℝ := 10
def central_angle : ℝ := 240

-- Theorem to prove the arc length is (40 * π) / 3
theorem arc_length (r : ℝ) (n : ℝ) (h_r : r = radius) (h_n : n = central_angle) : 
  (n * π * r) / 180 = (40 * π) / 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_arc_length_l247_24765


namespace NUMINAMATH_GPT_ben_marble_count_l247_24770

theorem ben_marble_count :
  ∃ k : ℕ, 5 * 2^k > 200 ∧ ∀ m < k, 5 * 2^m ≤ 200 :=
sorry

end NUMINAMATH_GPT_ben_marble_count_l247_24770


namespace NUMINAMATH_GPT_sector_properties_l247_24795

noncomputable def central_angle (l R : ℝ) : ℝ := l / R

noncomputable def area_of_sector (l R : ℝ) : ℝ := (1 / 2) * l * R

theorem sector_properties (R l : ℝ) (hR : R = 8) (hl : l = 12) :
  central_angle l R = 3 / 2 ∧ area_of_sector l R = 48 :=
by
  sorry

end NUMINAMATH_GPT_sector_properties_l247_24795


namespace NUMINAMATH_GPT_sum_of_ages_is_59_l247_24791

variable (juliet maggie ralph nicky lucy lily alex : ℕ)

def juliet_age := 10
def maggie_age := juliet_age - 3
def ralph_age := juliet_age + 2
def nicky_age := ralph_age / 2
def lucy_age := ralph_age + 1
def lily_age := ralph_age + 1
def alex_age := lucy_age - 5

theorem sum_of_ages_is_59 :
  maggie_age + ralph_age + nicky_age + lucy_age + lily_age + alex_age = 59 :=
by
  let maggie := 7
  let ralph := 12
  let nicky := 6
  let lucy := 13
  let lily := 13
  let alex := 8
  show maggie + ralph + nicky + lucy + lily + alex = 59
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_59_l247_24791


namespace NUMINAMATH_GPT_exists_pair_sum_ends_with_last_digit_l247_24723

theorem exists_pair_sum_ends_with_last_digit (a : ℕ → ℕ) (h_distinct: ∀ i j, (i ≠ j) → a i ≠ a j) (h_range: ∀ i, a i < 10) : ∀ (n : ℕ), n < 10 → ∃ i j, (i ≠ j) ∧ (a i + a j) % 10 = n % 10 :=
by sorry

end NUMINAMATH_GPT_exists_pair_sum_ends_with_last_digit_l247_24723


namespace NUMINAMATH_GPT_trader_sold_meters_l247_24793

-- Defining the context and conditions
def cost_price_per_meter : ℝ := 100
def profit_per_meter : ℝ := 5
def total_selling_price : ℝ := 8925

-- Calculating the selling price per meter
def selling_price_per_meter : ℝ := cost_price_per_meter + profit_per_meter

-- The problem statement: proving the number of meters sold is 85
theorem trader_sold_meters : (total_selling_price / selling_price_per_meter) = 85 :=
by
  sorry

end NUMINAMATH_GPT_trader_sold_meters_l247_24793


namespace NUMINAMATH_GPT_required_percentage_to_pass_l247_24736

-- Definitions based on conditions
def obtained_marks : ℕ := 175
def failed_by : ℕ := 56
def max_marks : ℕ := 700
def pass_marks : ℕ := obtained_marks + failed_by

-- Theorem stating the required percentage to pass
theorem required_percentage_to_pass : 
  (pass_marks : ℚ) / max_marks * 100 = 33 := 
by 
  sorry

end NUMINAMATH_GPT_required_percentage_to_pass_l247_24736


namespace NUMINAMATH_GPT_tank_capacity_l247_24729

theorem tank_capacity (C : ℝ) (h1 : 1/4 * C + 180 = 3/4 * C) : C = 360 :=
sorry

end NUMINAMATH_GPT_tank_capacity_l247_24729


namespace NUMINAMATH_GPT_least_positive_24x_16y_l247_24760

theorem least_positive_24x_16y (x y : ℤ) : ∃ a : ℕ, a > 0 ∧ a = 24 * x + 16 * y ∧ ∀ b : ℕ, b = 24 * x + 16 * y → b > 0 → b ≥ a :=
sorry

end NUMINAMATH_GPT_least_positive_24x_16y_l247_24760


namespace NUMINAMATH_GPT_simplify_polynomials_l247_24742

theorem simplify_polynomials :
  (4 * q ^ 4 + 2 * p ^ 3 - 7 * p + 8) + (3 * q ^ 4 - 2 * p ^ 3 + 3 * p ^ 2 - 5 * p + 6) =
  7 * q ^ 4 + 3 * p ^ 2 - 12 * p + 14 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomials_l247_24742


namespace NUMINAMATH_GPT_true_propositions_count_l247_24789

theorem true_propositions_count (a : ℝ) :
  ((a > -3 → a > -6) ∧ (a > -6 → ¬(a ≤ -3)) ∧ (a ≤ -3 → ¬(a > -6)) ∧ (a ≤ -6 → a ≤ -3)) → 
  2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_true_propositions_count_l247_24789


namespace NUMINAMATH_GPT_time_to_sell_all_cars_l247_24779

/-- Conditions: -/
def total_cars : ℕ := 500
def number_of_sales_professionals : ℕ := 10
def cars_per_salesperson_per_month : ℕ := 10

/-- Proof Statement: -/
theorem time_to_sell_all_cars 
  (total_cars : ℕ) 
  (number_of_sales_professionals : ℕ) 
  (cars_per_salesperson_per_month : ℕ) : 
  ((number_of_sales_professionals * cars_per_salesperson_per_month) > 0) →
  (total_cars / (number_of_sales_professionals * cars_per_salesperson_per_month)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_time_to_sell_all_cars_l247_24779


namespace NUMINAMATH_GPT_sequence_geometric_l247_24709

theorem sequence_geometric (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 = 1)
  (h_geom : ∀ k : ℕ, a (k + 1) - a k = (1 / 3) ^ k) :
  a n = (3 / 2) * (1 - (1 / 3) ^ n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_geometric_l247_24709


namespace NUMINAMATH_GPT_equilibrium_problems_l247_24733

-- Definition of equilibrium constant and catalyst relations

def q1 := False -- Any concentration of substances in equilibrium constant
def q2 := False -- Catalysts changing equilibrium constant
def q3 := False -- No shift if equilibrium constant doesn't change
def q4 := False -- ΔH > 0 if K decreases with increasing temperature
def q5 := True  -- Stoichiometric differences affecting equilibrium constants
def q6 := True  -- Equilibrium shift not necessarily changing equilibrium constant
def q7 := True  -- Extent of reaction indicated by both equilibrium constant and conversion rate

-- The theorem includes our problem statements

theorem equilibrium_problems :
  q1 = False ∧ q2 = False ∧ q3 = False ∧
  q4 = False ∧ q5 = True ∧ q6 = True ∧ q7 = True := by
  sorry

end NUMINAMATH_GPT_equilibrium_problems_l247_24733


namespace NUMINAMATH_GPT_problem_inequality_sol1_problem_inequality_sol2_l247_24703

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - (2 * a + 2)

theorem problem_inequality_sol1 (a x : ℝ) :
  (a > -3 / 2 ∧ (x > 2 * a + 2 ∨ x < -1)) ∨
  (a = -3 / 2 ∧ x ≠ -1) ∨
  (a < -3 / 2 ∧ (x > -1 ∨ x < 2 * a + 2)) ↔
  f x a > x :=
sorry

theorem problem_inequality_sol2 (a : ℝ) :
  (∀ x : ℝ, x > -1 → f x a + 3 ≥ 0) ↔
  a ≤ Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_GPT_problem_inequality_sol1_problem_inequality_sol2_l247_24703


namespace NUMINAMATH_GPT_liza_phone_bill_eq_70_l247_24730

theorem liza_phone_bill_eq_70 (initial_balance rent payment paycheck electricity internet final_balance phone_bill : ℝ)
  (h1 : initial_balance = 800)
  (h2 : rent = 450)
  (h3 : paycheck = 1500)
  (h4 : electricity = 117)
  (h5 : internet = 100)
  (h6 : final_balance = 1563)
  (h_balance_before_phone_bill : initial_balance - rent + paycheck - (electricity + internet) = 1633)
  (h_final_balance_def : 1633 - phone_bill = final_balance) :
  phone_bill = 70 := sorry

end NUMINAMATH_GPT_liza_phone_bill_eq_70_l247_24730


namespace NUMINAMATH_GPT_sum_of_extreme_a_l247_24792

theorem sum_of_extreme_a (a : ℝ) (h : ∀ x, x^2 - a*x - 20*a^2 < 0) (h_diff : |5*a - (-4*a)| ≤ 9) : 
  -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → a_min + a_max = 0 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_extreme_a_l247_24792


namespace NUMINAMATH_GPT_negation_equiv_l247_24725

-- Define the initial proposition
def initial_proposition (x : ℝ) : Prop :=
  x^2 - x + 1 > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0

-- The statement asserting the negation equivalence
theorem negation_equiv :
  (¬ ∀ x : ℝ, initial_proposition x) ↔ negated_proposition :=
by sorry

end NUMINAMATH_GPT_negation_equiv_l247_24725


namespace NUMINAMATH_GPT_ratio_of_walkway_to_fountain_l247_24724

theorem ratio_of_walkway_to_fountain (n s d : ℝ) (h₀ : n = 10) (h₁ : n^2 * s^2 = 0.40 * (n*s + 2*n*d)^2) : 
  d / s = 1 / 3.44 := 
sorry

end NUMINAMATH_GPT_ratio_of_walkway_to_fountain_l247_24724


namespace NUMINAMATH_GPT_units_digit_of_result_is_7_l247_24781

theorem units_digit_of_result_is_7 (a b c : ℕ) (h : a = c + 3) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  (original - reversed) % 10 = 7 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_result_is_7_l247_24781


namespace NUMINAMATH_GPT_find_annual_interest_rate_l247_24785

noncomputable def compound_interest (P A : ℝ) (r : ℝ) (n t : ℕ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 6000)
  (hA : A = 6615)
  (ht : t = 2)
  (hn : n = 1)
  (hr : compound_interest P A r n t) :
  r = 0.05 :=
sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l247_24785


namespace NUMINAMATH_GPT_equivalent_problem_l247_24752

variable {x y : Real}

theorem equivalent_problem 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 15) :
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_problem_l247_24752


namespace NUMINAMATH_GPT_greatest_y_value_l247_24764

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) : y ≤ -1 :=
by
  sorry

end NUMINAMATH_GPT_greatest_y_value_l247_24764


namespace NUMINAMATH_GPT_initial_workers_l247_24717

/--
In a factory, some workers were employed, and then 25% more workers have just been hired.
There are now 1065 employees in the factory. Prove that the number of workers initially employed is 852.
-/
theorem initial_workers (x : ℝ) (h1 : x + 0.25 * x = 1065) : x = 852 :=
sorry

end NUMINAMATH_GPT_initial_workers_l247_24717


namespace NUMINAMATH_GPT_relationship_between_products_l247_24716

variable {a₁ a₂ b₁ b₂ : ℝ}

theorem relationship_between_products (h₁ : a₁ < a₂) (h₂ : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := 
sorry

end NUMINAMATH_GPT_relationship_between_products_l247_24716


namespace NUMINAMATH_GPT_min_packs_needed_l247_24700

theorem min_packs_needed (P8 P15 P30 : ℕ) (h: P8 * 8 + P15 * 15 + P30 * 30 = 120) : P8 + P15 + P30 = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_packs_needed_l247_24700


namespace NUMINAMATH_GPT_count_integers_within_range_l247_24763

theorem count_integers_within_range : 
  ∃ (count : ℕ), count = 57 ∧ ∀ n : ℤ, -5.5 * Real.pi ≤ n ∧ n ≤ 12.5 * Real.pi → n ≥ -17 ∧ n ≤ 39 :=
by
  sorry

end NUMINAMATH_GPT_count_integers_within_range_l247_24763


namespace NUMINAMATH_GPT_range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l247_24706

-- Problem I Statement
theorem range_of_m_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 8 * x - 20 ≤ 0) → (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (-Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
by sorry

-- Problem II Statement
theorem range_of_m_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, ¬(x^2 - 8 * x - 20 ≤ 0) → ¬(1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) →
  (m ≤ -3 ∨ m ≥ 3) :=
by sorry

end NUMINAMATH_GPT_range_of_m_necessary_condition_range_of_m_not_sufficient_condition_l247_24706


namespace NUMINAMATH_GPT_inequality_solution_set_inequality_range_of_a_l247_24753

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : a = -8) :
  (|x - 3| + |x + 2| ≤ |a + 1|) ↔ (-3 ≤ x ∧ x ≤ 4) :=
by sorry

theorem inequality_range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x + 2| ≤ |a + 1|) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_inequality_range_of_a_l247_24753


namespace NUMINAMATH_GPT_money_left_after_purchase_l247_24719

def initial_toonies : Nat := 4
def value_per_toonie : Nat := 2
def total_coins : Nat := 10
def value_per_loonie : Nat := 1
def frappuccino_cost : Nat := 3

def toonies_value : Nat := initial_toonies * value_per_toonie
def loonies : Nat := total_coins - initial_toonies
def loonies_value : Nat := loonies * value_per_loonie
def initial_total : Nat := toonies_value + loonies_value
def remaining_money : Nat := initial_total - frappuccino_cost

theorem money_left_after_purchase : remaining_money = 11 := by
  sorry

end NUMINAMATH_GPT_money_left_after_purchase_l247_24719


namespace NUMINAMATH_GPT_num_positive_divisors_36_l247_24705

theorem num_positive_divisors_36 :
  let n := 36
  let d := (2 + 1) * (2 + 1)
  d = 9 :=
by
  sorry

end NUMINAMATH_GPT_num_positive_divisors_36_l247_24705


namespace NUMINAMATH_GPT_vector_c_expression_l247_24751

-- Define the vectors a, b, c
def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (-1, 1)
def vector_c : ℤ × ℤ := (1, 5)

-- Define the addition of vectors in ℤ × ℤ
def vec_add (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of vectors in ℤ × ℤ
def scalar_mul (k : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (k * v.1, k * v.2)

-- Given the conditions
def condition1 := vector_a = (1, 2)
def condition2 := vec_add vector_a vector_b = (0, 3)

-- The goal is to prove that vector_c = 2 * vector_a + vector_b
theorem vector_c_expression : vec_add (scalar_mul 2 vector_a) vector_b = vector_c := by
  sorry

end NUMINAMATH_GPT_vector_c_expression_l247_24751


namespace NUMINAMATH_GPT_david_more_pushups_l247_24744

theorem david_more_pushups (d z : ℕ) (h1 : d = 51) (h2 : d + z = 53) : d - z = 49 := by
  sorry

end NUMINAMATH_GPT_david_more_pushups_l247_24744


namespace NUMINAMATH_GPT_inequality_solution_l247_24720

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l247_24720


namespace NUMINAMATH_GPT_toy_selling_price_l247_24737

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_toy_selling_price_l247_24737


namespace NUMINAMATH_GPT_irrigation_tank_final_amount_l247_24787

theorem irrigation_tank_final_amount : 
  let initial_amount := 300.0
  let evaporation := 1.0
  let addition := 0.3
  let days := 45
  let daily_change := addition - evaporation
  let total_change := daily_change * days
  initial_amount + total_change = 268.5 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_irrigation_tank_final_amount_l247_24787


namespace NUMINAMATH_GPT_inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l247_24727

variables {x y g : ℝ}
variables (hx : 0 < x) (hy : 0 < y)
variable (hg : g = Real.sqrt (x * y))

theorem inf_geometric_mean_gt_3 :
  g ≥ 3 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≥ 2 / Real.sqrt (1 + g)) :=
by
  sorry

theorem inf_geometric_mean_le_2 :
  g ≤ 2 → (1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) ≤ 2 / Real.sqrt (1 + g)) :=
by
  sorry

end NUMINAMATH_GPT_inf_geometric_mean_gt_3_inf_geometric_mean_le_2_l247_24727


namespace NUMINAMATH_GPT_find_x_when_y_3_l247_24750

variable (y x k : ℝ)

axiom h₁ : x = k / (y ^ 2)
axiom h₂ : y = 9 → x = 0.1111111111111111
axiom y_eq_3 : y = 3

theorem find_x_when_y_3 : y = 3 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_y_3_l247_24750


namespace NUMINAMATH_GPT_C_investment_l247_24798

def A_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 36 = (1 / 6 : ℝ) * C * (1 / 6 : ℝ) * T

def B_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 9 = (1 / 3 : ℝ) * C * (1 / 3 : ℝ) * T

def C_investment_eq (x : ℝ) : Prop :=
  ∀ (C T : ℝ), x * C * T = (x : ℝ) * C * T

theorem C_investment (x : ℝ) :
  (∀ (C T : ℝ), A_investment_eq) ∧
  (∀ (C T : ℝ), B_investment_eq) ∧
  (∀ (C T : ℝ), C_investment_eq x) ∧
  (∀ (C T : ℝ), 100 / 2300 = (C * T / 36) / ((C * T / 36) + (C * T / 9) + (x * C * T))) →
  x = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_C_investment_l247_24798


namespace NUMINAMATH_GPT_problem_statement_l247_24784

-- Definitions of the events as described in the problem conditions.
def event1 (a b : ℝ) : Prop := a * b < 0 → a + b < 0
def event2 (a b : ℝ) : Prop := a * b < 0 → a - b > 0
def event3 (a b : ℝ) : Prop := a * b < 0 → a * b > 0
def event4 (a b : ℝ) : Prop := a * b < 0 → a / b < 0

-- The problem statement combining the conditions and the conclusion.
theorem problem_statement (a b : ℝ) (h1 : a * b < 0):
  (event4 a b) ∧ ¬(event3 a b) ∧ (event1 a b ∨ ¬(event1 a b)) ∧ (event2 a b ∨ ¬(event2 a b)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l247_24784


namespace NUMINAMATH_GPT_fraction_product_l247_24743

theorem fraction_product :
  (5 / 8) * (7 / 9) * (11 / 13) * (3 / 5) * (17 / 19) * (8 / 15) = 14280 / 1107000 :=
by sorry

end NUMINAMATH_GPT_fraction_product_l247_24743


namespace NUMINAMATH_GPT_janice_total_earnings_l247_24757

-- Defining the working conditions as constants
def days_per_week : ℕ := 5  -- Janice works 5 days a week
def earning_per_day : ℕ := 30  -- Janice earns $30 per day
def overtime_earning_per_shift : ℕ := 15  -- Janice earns $15 per overtime shift
def overtime_shifts : ℕ := 3  -- Janice works three overtime shifts

-- Defining Janice's total earnings for the week
def total_earnings : ℕ := (days_per_week * earning_per_day) + (overtime_shifts * overtime_earning_per_shift)

-- Statement to prove that Janice's total earnings are $195
theorem janice_total_earnings : total_earnings = 195 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_janice_total_earnings_l247_24757


namespace NUMINAMATH_GPT_percentage_of_students_receiving_certificates_l247_24702

theorem percentage_of_students_receiving_certificates
  (boys girls : ℕ)
  (pct_boys pct_girls : ℕ)
  (h_boys : boys = 30)
  (h_girls : girls = 20)
  (h_pct_boys : pct_boys = 30)
  (h_pct_girls : pct_girls = 40)
  :
  (pct_boys * boys + pct_girls * girls) / (100 * (boys + girls)) * 100 = 34 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_students_receiving_certificates_l247_24702


namespace NUMINAMATH_GPT_shorter_side_ratio_l247_24721

variable {x y : ℝ}
variables (h1 : x < y)
variables (h2 : x + y - Real.sqrt (x^2 + y^2) = 1/2 * y)

theorem shorter_side_ratio (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = 1 / 2 * y) : x / y = 3 / 4 := 
sorry

end NUMINAMATH_GPT_shorter_side_ratio_l247_24721


namespace NUMINAMATH_GPT_circle_diameter_l247_24774

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ (d : ℝ), d = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l247_24774


namespace NUMINAMATH_GPT_maximize_a_minus_b_plus_c_l247_24758

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem maximize_a_minus_b_plus_c
  {a b c : ℝ}
  (h : ∀ x : ℝ, f a b c x ≥ -1) :
  a - b + c ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximize_a_minus_b_plus_c_l247_24758


namespace NUMINAMATH_GPT_circumference_irrational_l247_24775

theorem circumference_irrational (d : ℚ) : ¬ ∃ (r : ℚ), r = π * d :=
sorry

end NUMINAMATH_GPT_circumference_irrational_l247_24775


namespace NUMINAMATH_GPT_tens_digit_2015_pow_2016_minus_2017_l247_24726

theorem tens_digit_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 = 8 := 
sorry

end NUMINAMATH_GPT_tens_digit_2015_pow_2016_minus_2017_l247_24726


namespace NUMINAMATH_GPT_value_of_a_l247_24794

theorem value_of_a {a x : ℝ} (h1 : x > 0) (h2 : 2 * x + 1 > a * x) : a ≤ 2 :=
sorry

end NUMINAMATH_GPT_value_of_a_l247_24794


namespace NUMINAMATH_GPT_steve_num_nickels_l247_24731

-- Definitions for the conditions
def num_nickels (N : ℕ) : Prop :=
  ∃ D Q : ℕ, D = N + 4 ∧ Q = D + 3 ∧ 5 * N + 10 * D + 25 * Q + 5 = 380

-- Statement of the problem
theorem steve_num_nickels : num_nickels 4 :=
sorry

end NUMINAMATH_GPT_steve_num_nickels_l247_24731


namespace NUMINAMATH_GPT_simplify_expression_l247_24704

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3))
  = 3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l247_24704


namespace NUMINAMATH_GPT_find_positives_xyz_l247_24767

theorem find_positives_xyz (x y z : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0)
    (heq : (1 : ℚ)/x + (1 : ℚ)/y + (1 : ℚ)/z = 4 / 5) :
    (x = 2 ∧ y = 4 ∧ z = 20) ∨ (x = 2 ∧ y = 5 ∧ z = 10) :=
by
  sorry

-- This theorem states that there are only two sets of positive integers (x, y, z)
-- that satisfy the equation (1/x) + (1/y) + (1/z) = 4/5, specifically:
-- (2, 4, 20) and (2, 5, 10).

end NUMINAMATH_GPT_find_positives_xyz_l247_24767


namespace NUMINAMATH_GPT_proposition_holds_n_2019_l247_24745

theorem proposition_holds_n_2019 (P: ℕ → Prop) 
  (H1: ∀ k : ℕ, k > 0 → ¬ P (k + 1) → ¬ P k) 
  (H2: P 2018) : 
  P 2019 :=
by 
  sorry

end NUMINAMATH_GPT_proposition_holds_n_2019_l247_24745


namespace NUMINAMATH_GPT_no_solution_if_n_eq_neg_one_l247_24778

theorem no_solution_if_n_eq_neg_one (n x y z : ℝ) :
  (n * x + y + z = 2) ∧ (x + n * y + z = 2) ∧ (x + y + n * z = 2) ↔ n = -1 → false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_if_n_eq_neg_one_l247_24778


namespace NUMINAMATH_GPT_rectangular_prism_sides_multiples_of_5_l247_24782

noncomputable def rectangular_prism_sides_multiples_product_condition 
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) (prod_eq_450 : l * w = 450) : Prop :=
  l ∣ 450 ∧ w ∣ 450

theorem rectangular_prism_sides_multiples_of_5
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) :
  rectangular_prism_sides_multiples_product_condition l w hl hw (by sorry) :=
sorry

end NUMINAMATH_GPT_rectangular_prism_sides_multiples_of_5_l247_24782


namespace NUMINAMATH_GPT_problem_statement_l247_24713

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : (x - 3)^4 + 81 / (x - 3)^4 = 63 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l247_24713


namespace NUMINAMATH_GPT_pairs_of_values_l247_24772

theorem pairs_of_values (x y : ℂ) :
  (y = (x + 2)^3 ∧ x * y + 2 * y = 2) →
  (∃ (r1 r2 i1 i2 : ℂ), (r1.im = 0 ∧ r2.im = 0) ∧ (i1.im ≠ 0 ∧ i2.im ≠ 0) ∧ 
    ((r1, (r1 + 2)^3) = (x, y) ∨ (r2, (r2 + 2)^3) = (x, y) ∨
     (i1, (i1 + 2)^3) = (x, y) ∨ (i2, (i2 + 2)^3) = (x, y))) :=
sorry

end NUMINAMATH_GPT_pairs_of_values_l247_24772


namespace NUMINAMATH_GPT_fraction_is_five_over_nine_l247_24768

theorem fraction_is_five_over_nine (f k t : ℝ) (h1 : t = f * (k - 32)) (h2 : t = 50) (h3 : k = 122) : f = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_five_over_nine_l247_24768


namespace NUMINAMATH_GPT_find_g_zero_l247_24708

noncomputable def g (x : ℝ) : ℝ := sorry  -- fourth-degree polynomial

-- Conditions
axiom cond1 : |g 1| = 16
axiom cond2 : |g 3| = 16
axiom cond3 : |g 4| = 16
axiom cond4 : |g 5| = 16
axiom cond5 : |g 6| = 16
axiom cond6 : |g 7| = 16

-- statement to prove
theorem find_g_zero : |g 0| = 54 := 
by sorry

end NUMINAMATH_GPT_find_g_zero_l247_24708


namespace NUMINAMATH_GPT_find_k_and_general_term_l247_24776

noncomputable def sum_of_first_n_terms (n k : ℝ) : ℝ :=
  -n^2 + (10 + k) * n + (k - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  -2 * n + 12

theorem find_k_and_general_term :
  (∀ n k : ℝ, sum_of_first_n_terms n k = sum_of_first_n_terms n (1 : ℝ)) ∧
  (∀ n : ℕ, ∃ an : ℝ, an = general_term n) :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_general_term_l247_24776


namespace NUMINAMATH_GPT_original_triangle_area_l247_24747

theorem original_triangle_area (A_orig A_new : ℝ) (h1 : A_new = 256) (h2 : A_new = 16 * A_orig) : A_orig = 16 :=
by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l247_24747


namespace NUMINAMATH_GPT_license_plate_count_l247_24738

-- Define the conditions as constants
def even_digit_count : Nat := 5
def consonant_count : Nat := 20
def vowel_count : Nat := 6

-- Define the problem as a theorem to prove
theorem license_plate_count : even_digit_count * consonant_count * vowel_count * consonant_count = 12000 := 
by
  -- The proof is not required, so we leave it as sorry
  sorry

end NUMINAMATH_GPT_license_plate_count_l247_24738


namespace NUMINAMATH_GPT_peak_infection_day_l247_24796

-- Given conditions
def initial_cases : Nat := 20
def increase_rate : Nat := 50
def decrease_rate : Nat := 30
def total_infections : Nat := 8670
def total_days : Nat := 30

-- Peak Day and infections on that day
def peak_day : Nat := 12

-- Theorem stating what we want to prove
theorem peak_infection_day :
  ∃ n : Nat, n = initial_cases + increase_rate * (peak_day - 1) - decrease_rate * (30 - peak_day) :=
sorry

end NUMINAMATH_GPT_peak_infection_day_l247_24796


namespace NUMINAMATH_GPT_queenie_total_earnings_l247_24722

-- Define the conditions
def daily_wage : ℕ := 150
def overtime_wage_per_hour : ℕ := 5
def days_worked : ℕ := 5
def overtime_hours : ℕ := 4

-- Define the main problem
theorem queenie_total_earnings : 
  (daily_wage * days_worked + overtime_wage_per_hour * overtime_hours) = 770 :=
by
  sorry

end NUMINAMATH_GPT_queenie_total_earnings_l247_24722


namespace NUMINAMATH_GPT_power_boat_travel_time_l247_24755

theorem power_boat_travel_time {r p t : ℝ} (h1 : r > 0) (h2 : p > 0) 
  (h3 : (p + r) * t + (p - r) * (9 - t) = 9 * r) : t = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_power_boat_travel_time_l247_24755


namespace NUMINAMATH_GPT_can_cross_all_rivers_and_extra_material_l247_24769

-- Definitions for river widths, bridge length, and additional material.
def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def bridge_length : ℕ := 295
def additional_material : ℕ := 1020

-- Calculations for material needed for each river.
def material_needed_for_river1 : ℕ := river1_width - bridge_length
def material_needed_for_river2 : ℕ := river2_width - bridge_length
def material_needed_for_river3 : ℕ := river3_width - bridge_length

-- Total material needed to cross all three rivers.
def total_material_needed : ℕ := material_needed_for_river1 + material_needed_for_river2 + material_needed_for_river3

-- The main theorem statement to prove.
theorem can_cross_all_rivers_and_extra_material :
  total_material_needed <= additional_material ∧ (additional_material - total_material_needed = 421) := 
by 
  sorry

end NUMINAMATH_GPT_can_cross_all_rivers_and_extra_material_l247_24769


namespace NUMINAMATH_GPT_sum_lengths_AMC_l247_24749

theorem sum_lengths_AMC : 
  let length_A := 2 * (Real.sqrt 2) + 2
  let length_M := 3 + 3 + 2 * (Real.sqrt 2)
  let length_C := 3 + 3 + 2
  length_A + length_M + length_C = 13 + 4 * (Real.sqrt 2)
  := by
  sorry

end NUMINAMATH_GPT_sum_lengths_AMC_l247_24749


namespace NUMINAMATH_GPT_speed_of_car_in_second_hour_l247_24707

noncomputable def speed_in_first_hour : ℝ := 90
noncomputable def average_speed : ℝ := 82.5
noncomputable def total_time : ℝ := 2

theorem speed_of_car_in_second_hour : 
  ∃ (speed_in_second_hour : ℝ), 
  (speed_in_first_hour + speed_in_second_hour) / total_time = average_speed ∧ 
  speed_in_first_hour = 90 ∧ 
  average_speed = 82.5 → 
  speed_in_second_hour = 75 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_car_in_second_hour_l247_24707


namespace NUMINAMATH_GPT_parallel_lines_slope_l247_24740

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, ax + 3 * y + 1 = 0 → 2 * x + (a + 1) * y + 1 = 0) →
  a = -3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l247_24740


namespace NUMINAMATH_GPT_numDifferentSignals_l247_24786

-- Number of indicator lights in a row
def numLights : Nat := 6

-- Number of lights that light up each time
def lightsLit : Nat := 3

-- Number of colors each light can show
def numColors : Nat := 3

-- Function to calculate number of different signals
noncomputable def calculateSignals (n m k : Nat) : Nat :=
  -- Number of possible arrangements of "adjacent, adjacent, separate" and "separate, adjacent, adjacent"
  let arrangements := 4 + 4
  -- Number of color combinations for the lit lights
  let colors := k * k * k
  arrangements * colors

-- Theorem stating the total number of different signals is 324
theorem numDifferentSignals : calculateSignals numLights lightsLit numColors = 324 := 
by
  sorry

end NUMINAMATH_GPT_numDifferentSignals_l247_24786


namespace NUMINAMATH_GPT_fixed_monthly_charge_l247_24759

-- Given conditions
variable (F C_J : ℕ)
axiom january_bill : F + C_J = 46
axiom february_bill : F + 2 * C_J = 76

-- Proof problem
theorem fixed_monthly_charge : F = 16 :=
by
  sorry

end NUMINAMATH_GPT_fixed_monthly_charge_l247_24759


namespace NUMINAMATH_GPT_shift_line_down_4_units_l247_24741

theorem shift_line_down_4_units :
  ∀ (x : ℝ), y = - (3 / 4) * x → (y - 4 = - (3 / 4) * x - 4) := by
  sorry

end NUMINAMATH_GPT_shift_line_down_4_units_l247_24741


namespace NUMINAMATH_GPT_right_triangle_distance_l247_24788

theorem right_triangle_distance (x h d : ℝ) :
  x + Real.sqrt ((x + 2 * h) ^ 2 + d ^ 2) = 2 * h + d → 
  x = (h * d) / (2 * h + d) :=
by
  intros h_eq_d
  sorry

end NUMINAMATH_GPT_right_triangle_distance_l247_24788


namespace NUMINAMATH_GPT_max_value_of_linear_combination_l247_24780

theorem max_value_of_linear_combination (x y : ℝ) (h : x^2 - 3 * x + 4 * y = 7) : 
  3 * x + 4 * y ≤ 16 :=
sorry

end NUMINAMATH_GPT_max_value_of_linear_combination_l247_24780


namespace NUMINAMATH_GPT_find_x_plus_y_l247_24771

theorem find_x_plus_y (x y : ℚ) (h1 : 3 * x - 4 * y = 18) (h2 : x + 3 * y = -1) :
  x + y = 29 / 13 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l247_24771


namespace NUMINAMATH_GPT_number_of_people_for_cheaper_second_caterer_l247_24790

theorem number_of_people_for_cheaper_second_caterer : 
  ∃ (x : ℕ), (150 + 20 * x > 250 + 15 * x + 50) ∧ 
  ∀ (y : ℕ), (y < x → ¬ (150 + 20 * y > 250 + 15 * y + 50)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_for_cheaper_second_caterer_l247_24790


namespace NUMINAMATH_GPT_find_all_possible_f_l247_24732

-- Noncomputability is needed here since we cannot construct a function 
-- like f deterministically via computation due to the nature of the problem.
noncomputable def functional_equation_solution (f : ℕ → ℕ) := 
  (∀ a b : ℕ, f a + f b ∣ 2 * (a + b - 1)) → 
  (∀ x : ℕ, f x = 1) ∨ (∀ x : ℕ, f x = 2 * x - 1)

-- Statement of the mathematically equivalent proof problem.
theorem find_all_possible_f (f : ℕ → ℕ) : functional_equation_solution f := 
sorry

end NUMINAMATH_GPT_find_all_possible_f_l247_24732


namespace NUMINAMATH_GPT_harmonic_mean_closest_to_one_l247_24762

-- Define the given conditions a = 1/4 and b = 2048
def a : ℚ := 1 / 4
def b : ℚ := 2048

-- Define the harmonic mean of two numbers
def harmonic_mean (x y : ℚ) : ℚ := 2 * x * y / (x + y)

-- State the theorem proving the harmonic mean is closest to 1
theorem harmonic_mean_closest_to_one : abs (harmonic_mean a b - 1) < 1 :=
sorry

end NUMINAMATH_GPT_harmonic_mean_closest_to_one_l247_24762


namespace NUMINAMATH_GPT_complex_number_solution_l247_24783

theorem complex_number_solution {i z : ℂ} (h : (2 : ℂ) / (1 + i) = z + i) : z = 1 + 2 * i :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l247_24783


namespace NUMINAMATH_GPT_range_of_m_l247_24714

theorem range_of_m (m : ℝ) : (-6 < m ∧ m < 2) ↔ ∃ x : ℝ, |x - m| + |x + 2| < 4 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l247_24714


namespace NUMINAMATH_GPT_smallest_x_fraction_floor_l247_24739

theorem smallest_x_fraction_floor (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 :=
sorry

end NUMINAMATH_GPT_smallest_x_fraction_floor_l247_24739


namespace NUMINAMATH_GPT_valid_numbers_l247_24773

def is_valid_100_digit_number (N N' : ℕ) (k m n : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 100 ∧ 0 ≤ m ∧ m < 10^k ∧ 
  N = m + 10^k * a + 10^(k + 2) * n ∧ 
  N' = m + 10^k * n ∧
  N = 87 * N'

theorem valid_numbers : ∀ (N : ℕ), (∃ N' k m n a, is_valid_100_digit_number N N' k m n a) →
  N = 435 * 10^97 ∨ 
  N = 1305 * 10^96 ∨ 
  N = 2175 * 10^96 ∨ 
  N = 3045 * 10^96 :=
by
  sorry

end NUMINAMATH_GPT_valid_numbers_l247_24773


namespace NUMINAMATH_GPT_find_integer_solutions_l247_24756

theorem find_integer_solutions (x y : ℤ) :
  8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y ↔
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
by 
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l247_24756


namespace NUMINAMATH_GPT_Nick_sister_age_l247_24799

theorem Nick_sister_age
  (Nick_age : ℕ := 13)
  (Bro_in_5_years : ℕ := 21)
  (H : ∃ S : ℕ, (Nick_age + S) / 2 + 5 = Bro_in_5_years) :
  ∃ S : ℕ, S = 19 :=
by
  sorry

end NUMINAMATH_GPT_Nick_sister_age_l247_24799


namespace NUMINAMATH_GPT_find_x_such_that_ceil_mul_x_eq_168_l247_24718

theorem find_x_such_that_ceil_mul_x_eq_168 (x : ℝ) (h_pos : x > 0)
  (h_eq : ⌈x⌉ * x = 168) (h_ceil: ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) :
  x = 168 / 13 :=
by
  sorry

end NUMINAMATH_GPT_find_x_such_that_ceil_mul_x_eq_168_l247_24718


namespace NUMINAMATH_GPT_probability_of_rolling_5_is_1_over_9_l247_24761

def num_sides_dice : ℕ := 6

def favorable_combinations : List (ℕ × ℕ) :=
[(1, 4), (2, 3), (3, 2), (4, 1)]

def total_combinations : ℕ :=
num_sides_dice * num_sides_dice

def favorable_count : ℕ := favorable_combinations.length

def probability_rolling_5 : ℚ :=
favorable_count / total_combinations

theorem probability_of_rolling_5_is_1_over_9 :
  probability_rolling_5 = 1 / 9 :=
sorry

end NUMINAMATH_GPT_probability_of_rolling_5_is_1_over_9_l247_24761


namespace NUMINAMATH_GPT_engineering_student_max_marks_l247_24728

/-- 
If an engineering student has to secure 36% marks to pass, and he gets 130 marks but fails by 14 marks, 
then the maximum number of marks is 400.
-/
theorem engineering_student_max_marks (M : ℝ) (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (pass_marks : ℝ) :
  passing_percentage = 0.36 →
  marks_obtained = 130 →
  marks_failed_by = 14 →
  pass_marks = marks_obtained + marks_failed_by →
  pass_marks = passing_percentage * M →
  M = 400 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_engineering_student_max_marks_l247_24728
