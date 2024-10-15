import Mathlib

namespace NUMINAMATH_CALUDE_negation_and_contrary_l1973_197357

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def original_statement : Prop :=
  ∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_contrary :
  (¬original_statement ↔ ∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬divisible_by_five n) ∧
  (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → ¬divisible_by_five n) :=
sorry

end NUMINAMATH_CALUDE_negation_and_contrary_l1973_197357


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1973_197338

theorem geometric_sequence_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_geometric : b^2 = a * c) : 
  a^2 + b^2 + c^2 > (a - b + c)^2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1973_197338


namespace NUMINAMATH_CALUDE_sid_computer_accessories_cost_l1973_197374

/-- Calculates the amount spent on computer accessories given the initial amount,
    snack cost, and remaining amount after purchases. -/
def computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  initial_amount - snack_cost - remaining_amount

/-- Proves that Sid spent $12 on computer accessories given the problem conditions. -/
theorem sid_computer_accessories_cost :
  let initial_amount : ℕ := 48
  let snack_cost : ℕ := 8
  let remaining_amount : ℕ := (initial_amount / 2) + 4
  computer_accessories_cost initial_amount snack_cost remaining_amount = 12 := by
  sorry

#eval computer_accessories_cost 48 8 28

end NUMINAMATH_CALUDE_sid_computer_accessories_cost_l1973_197374


namespace NUMINAMATH_CALUDE_password_decryption_probability_l1973_197342

theorem password_decryption_probability :
  let p₁ : ℚ := 1/5
  let p₂ : ℚ := 2/5
  let p₃ : ℚ := 1/2
  let prob_at_least_one_success : ℚ := 1 - (1 - p₁) * (1 - p₂) * (1 - p₃)
  prob_at_least_one_success = 19/25 :=
by sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l1973_197342


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l1973_197300

def teaching_problem (v a d e n : ℕ) : Prop :=
  v + a + d + e + n = 225 ∧
  v = a + 9 ∧
  v = d - 15 ∧
  e = a - 3 ∧
  e = n + 7

theorem dennis_teaching_years :
  ∀ v a d e n : ℕ, teaching_problem v a d e n → d = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l1973_197300


namespace NUMINAMATH_CALUDE_consumption_increase_l1973_197348

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.76 * original_tax
  let revenue_decrease := 0.1488
  let new_revenue := (1 - revenue_decrease) * (original_tax * original_consumption)
  ∃ (consumption_increase : ℝ), 
    new_tax * (original_consumption * (1 + consumption_increase)) = new_revenue ∧ 
    consumption_increase = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l1973_197348


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l1973_197352

theorem no_natural_numbers_satisfying_condition : 
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ (k : ℕ), b^2 + 4*a = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l1973_197352


namespace NUMINAMATH_CALUDE_fruit_pie_theorem_l1973_197369

/-- Represents the quantities of fruits used in pie making -/
structure FruitQuantities where
  apples : ℕ
  peaches : ℕ
  pears : ℕ
  plums : ℕ

/-- The ratio of fruits used per apple in pie making -/
structure FruitRatio where
  peaches_per_apple : ℕ
  pears_per_apple : ℕ
  plums_per_apple : ℕ

/-- Calculate the quantities of fruits used given the number of apples and the ratio -/
def calculate_used_fruits (apples_used : ℕ) (ratio : FruitRatio) : FruitQuantities :=
  { apples := apples_used,
    peaches := apples_used * ratio.peaches_per_apple,
    pears := apples_used * ratio.pears_per_apple,
    plums := apples_used * ratio.plums_per_apple }

theorem fruit_pie_theorem (initial_apples initial_peaches initial_pears initial_plums : ℕ)
                          (ratio : FruitRatio)
                          (apples_left : ℕ) :
  initial_apples = 40 →
  initial_peaches = 54 →
  initial_pears = 60 →
  initial_plums = 48 →
  ratio.peaches_per_apple = 2 →
  ratio.pears_per_apple = 3 →
  ratio.plums_per_apple = 4 →
  apples_left = 39 →
  calculate_used_fruits (initial_apples - apples_left) ratio =
    { apples := 1, peaches := 2, pears := 3, plums := 4 } :=
by sorry


end NUMINAMATH_CALUDE_fruit_pie_theorem_l1973_197369


namespace NUMINAMATH_CALUDE_sqrt_12_equals_2_sqrt_3_l1973_197378

theorem sqrt_12_equals_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_equals_2_sqrt_3_l1973_197378


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1973_197381

theorem sufficient_condition_for_inequality (x : ℝ) : 
  (∀ x, x > 1 → 1 - 1/x > 0) ∧ 
  (∃ x, 1 - 1/x > 0 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1973_197381


namespace NUMINAMATH_CALUDE_zeros_of_composition_l1973_197325

/-- Given functions f and g, prove that the zeros of their composition h are ±√2 -/
theorem zeros_of_composition (f g h : ℝ → ℝ) :
  (∀ x, f x = 2 * x - 4) →
  (∀ x, g x = x^2) →
  (∀ x, h x = f (g x)) →
  {x : ℝ | h x = 0} = {-Real.sqrt 2, Real.sqrt 2} := by
  sorry

end NUMINAMATH_CALUDE_zeros_of_composition_l1973_197325


namespace NUMINAMATH_CALUDE_max_value_linear_program_l1973_197344

theorem max_value_linear_program (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + 2*y ≤ 4) 
  (h3 : x - 2*y ≤ 2) : 
  ∃ (z : ℝ), z = x + 3*y ∧ z ≤ 16/3 ∧ 
  (∀ (x' y' : ℝ), x' - y' ≥ 0 → x' + 2*y' ≤ 4 → x' - 2*y' ≤ 2 → x' + 3*y' ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l1973_197344


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1973_197391

theorem inequality_equivalence (x : ℝ) :
  -1 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1973_197391


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1973_197388

-- Define the universe type
inductive Universe : Type
  | a | b | c | d | e

-- Define the sets
def I : Set Universe := {Universe.a, Universe.b, Universe.c, Universe.d, Universe.e}
def M : Set Universe := {Universe.a, Universe.b, Universe.c}
def N : Set Universe := {Universe.b, Universe.d, Universe.e}

-- State the theorem
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {Universe.d, Universe.e} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1973_197388


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1973_197306

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The equation of a line y = mx + c -/
structure Line where
  m : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def distance_to_line (p : Point) (l : Line) : ℝ := sorry

def eccentricity (e : Ellipse) : ℝ := sorry

theorem ellipse_theorem (e : Ellipse) 
  (h_distance : distance_to_line ⟨0, 0⟩ 
    {m := -e.b / e.a, c := e.a * e.b / (e.a^2 + e.b^2)} = 2 * Real.sqrt 5 / 5)
  (h_eccentricity : eccentricity e = Real.sqrt 3 / 2) :
  ∃ (l : Line), 
    (e.a = 2 ∧ e.b = 1) ∧ 
    (l.c = 5/3) ∧ 
    (l.m = 3 * Real.sqrt 14 / 14 ∨ l.m = -3 * Real.sqrt 14 / 14) ∧
    (∃ (m n : Point), 
      m.x^2 / 4 + m.y^2 = 1 ∧ 
      n.x^2 / 4 + n.y^2 = 1 ∧ 
      m.y = l.m * m.x + l.c ∧ 
      n.y = l.m * n.x + l.c ∧ 
      (m.x - 0)^2 + (m.y - 5/3)^2 = 4 * ((n.x - 0)^2 + (n.y - 5/3)^2)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1973_197306


namespace NUMINAMATH_CALUDE_system_solution_l1973_197385

theorem system_solution :
  let x₁ : ℝ := 5
  let x₂ : ℝ := -5
  let x₃ : ℝ := 0
  let x₄ : ℝ := 2
  let x₅ : ℝ := -1
  let x₆ : ℝ := 1
  (x₁ + x₃ + 2*x₄ + 3*x₅ - 4*x₆ = 20) ∧
  (2*x₁ + x₂ - 3*x₃ + x₅ + 2*x₆ = -13) ∧
  (5*x₁ - x₂ + x₃ + 2*x₄ + 6*x₅ = 20) ∧
  (2*x₁ - 2*x₂ + 3*x₃ + 2*x₅ + 2*x₆ = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1973_197385


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1973_197324

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
  Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2 →
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1973_197324


namespace NUMINAMATH_CALUDE_triangle_angle_sequence_range_l1973_197363

theorem triangle_angle_sequence_range (A B C a b c k : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- angles are positive
  A + B + C = π ∧  -- sum of angles in a triangle
  B - A = C - B ∧  -- arithmetic sequence
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- sides are positive
  a^2 + c^2 = k * b^2 →  -- given equation
  1 < k ∧ k ≤ 2 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_sequence_range_l1973_197363


namespace NUMINAMATH_CALUDE_contractor_absence_l1973_197315

/-- Proves that given the specified contract conditions, the contractor was absent for 10 days -/
theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ)
  (h_total_days : total_days = 30)
  (h_daily_pay : daily_pay = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_received : total_received = 425) :
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    days_worked * daily_pay - days_absent * daily_fine = total_received ∧
    days_absent = 10 :=
by sorry

end NUMINAMATH_CALUDE_contractor_absence_l1973_197315


namespace NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l1973_197370

def f (x : ℝ) : ℝ := -x^2 + 3

theorem quadratic_function_satisfies_conditions :
  (∀ x y : ℝ, x < y → f x > f y) ∧ 
  f 0 = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_satisfies_conditions_l1973_197370


namespace NUMINAMATH_CALUDE_not_divisible_five_power_minus_one_by_four_power_minus_one_l1973_197329

theorem not_divisible_five_power_minus_one_by_four_power_minus_one (n : ℕ) :
  ¬(∃ k : ℕ, 5^n - 1 = k * (4^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_five_power_minus_one_by_four_power_minus_one_l1973_197329


namespace NUMINAMATH_CALUDE_sin_30_plus_cos_60_quadratic_equation_solutions_l1973_197364

-- Problem 1
theorem sin_30_plus_cos_60 : Real.sin (π / 6) + Real.cos (π / 3) = 1 := by sorry

-- Problem 2
theorem quadratic_equation_solutions (x : ℝ) : 
  x^2 - 4*x = 12 ↔ x = 6 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_sin_30_plus_cos_60_quadratic_equation_solutions_l1973_197364


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l1973_197383

/-- Given a quadratic equation 16x^2 + 32x - 1280 = 0, prove that when rewritten
    in the form (x + r)^2 = s by completing the square, the value of s is 81. -/
theorem quadratic_complete_square (x : ℝ) :
  (16 * x^2 + 32 * x - 1280 = 0) →
  ∃ (r s : ℝ), ((x + r)^2 = s ∧ s = 81) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l1973_197383


namespace NUMINAMATH_CALUDE_train_speed_l1973_197317

/-- Proves that the speed of a train is 72 km/hr given its length and time to pass a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 100) (h2 : time = 5) :
  (length / time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1973_197317


namespace NUMINAMATH_CALUDE_valid_sequences_count_l1973_197333

/-- The number of distinct coin flip sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The number of distinct coin flip sequences of length n starting with two heads -/
def sequences_starting_with_two_heads (n : ℕ) : ℕ := 2^(n-2)

/-- The number of valid coin flip sequences of length 10, excluding those starting with two heads -/
def valid_sequences : ℕ := total_sequences 10 - sequences_starting_with_two_heads 10

theorem valid_sequences_count : valid_sequences = 768 := by sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l1973_197333


namespace NUMINAMATH_CALUDE_erasers_left_in_box_l1973_197303

/-- The number of erasers left in the box after Doris, Mark, and Ellie take some out. -/
def erasers_left (initial : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ) : ℕ :=
  initial - doris_takes - mark_takes - ellie_takes

/-- Theorem stating that 105 erasers are left in the box -/
theorem erasers_left_in_box :
  erasers_left 250 75 40 30 = 105 := by
  sorry

end NUMINAMATH_CALUDE_erasers_left_in_box_l1973_197303


namespace NUMINAMATH_CALUDE_last_day_vases_proof_l1973_197334

/-- The number of vases Jane can arrange in a day -/
def vases_per_day : ℕ := 16

/-- The total number of vases to be arranged -/
def total_vases : ℕ := 248

/-- The number of vases Jane will arrange on the last day -/
def last_day_vases : ℕ := total_vases - (vases_per_day * (total_vases / vases_per_day))

theorem last_day_vases_proof :
  last_day_vases = 8 :=
by sorry

end NUMINAMATH_CALUDE_last_day_vases_proof_l1973_197334


namespace NUMINAMATH_CALUDE_first_bouquet_carnations_l1973_197305

/-- The number of carnations in the first bouquet -/
def carnations_in_first_bouquet (total_bouquets : ℕ) 
  (carnations_in_second : ℕ) (carnations_in_third : ℕ) (average : ℕ) : ℕ :=
  total_bouquets * average - carnations_in_second - carnations_in_third

/-- Theorem stating the number of carnations in the first bouquet -/
theorem first_bouquet_carnations :
  carnations_in_first_bouquet 3 14 13 12 = 9 := by
  sorry

#eval carnations_in_first_bouquet 3 14 13 12

end NUMINAMATH_CALUDE_first_bouquet_carnations_l1973_197305


namespace NUMINAMATH_CALUDE_angle_range_in_third_quadrant_l1973_197398

theorem angle_range_in_third_quadrant (θ : Real) (k : Int) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.sin (θ/4) < Real.cos (θ/4)) →  -- sin(θ/4) < cos(θ/4)
  (∃ k : Int, 
    ((2*k*π + 5*π/4 < θ/4 ∧ θ/4 < 2*k*π + 11*π/8) ∨ 
     (2*k*π + 7*π/4 < θ/4 ∧ θ/4 < 2*k*π + 15*π/8))) := by
  sorry

end NUMINAMATH_CALUDE_angle_range_in_third_quadrant_l1973_197398


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l1973_197318

theorem tan_alpha_and_expression (α : Real) 
  (h : Real.tan (π / 4 + α) = 1 / 2) : 
  Real.tan α = -1 / 3 ∧ 
  (Real.sin (2 * α + 2 * π) - Real.sin (π / 2 - α) ^ 2) / 
  (1 - Real.cos (π - 2 * α) + Real.sin α ^ 2) = -15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l1973_197318


namespace NUMINAMATH_CALUDE_min_expected_weight_l1973_197368

theorem min_expected_weight (x y e : ℝ) :
  y = 0.85 * x - 88 + e →
  |e| ≤ 4 →
  x = 160 →
  ∃ y_min : ℝ, y_min = 44 ∧ ∀ y' : ℝ, (∃ e' : ℝ, y' = 0.85 * x - 88 + e' ∧ |e'| ≤ 4) → y' ≥ y_min :=
by sorry

end NUMINAMATH_CALUDE_min_expected_weight_l1973_197368


namespace NUMINAMATH_CALUDE_basketball_score_l1973_197350

theorem basketball_score (total_shots : ℕ) (three_point_shots : ℕ) : 
  total_shots = 11 → three_point_shots = 4 → 
  3 * three_point_shots + 2 * (total_shots - three_point_shots) = 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_l1973_197350


namespace NUMINAMATH_CALUDE_library_book_difference_prove_book_difference_l1973_197321

theorem library_book_difference (initial_books : ℕ) (bought_two_years_ago : ℕ) 
  (donated_this_year : ℕ) (current_total : ℕ) : ℕ :=
  let books_before_last_year := initial_books + bought_two_years_ago
  let books_bought_last_year := current_total - books_before_last_year + donated_this_year
  books_bought_last_year - bought_two_years_ago

theorem prove_book_difference :
  library_book_difference 500 300 200 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_library_book_difference_prove_book_difference_l1973_197321


namespace NUMINAMATH_CALUDE_distribution_of_distinct_objects_l1973_197328

theorem distribution_of_distinct_objects (n : ℕ) (m : ℕ) :
  n = 6 → m = 12 → n^m = 2985984 := by
  sorry

end NUMINAMATH_CALUDE_distribution_of_distinct_objects_l1973_197328


namespace NUMINAMATH_CALUDE_fifth_root_of_eight_to_fifteen_l1973_197343

theorem fifth_root_of_eight_to_fifteen (x : ℝ) : x = (8 ^ (1 / 5 : ℝ)) → x^15 = 512 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_eight_to_fifteen_l1973_197343


namespace NUMINAMATH_CALUDE_monica_has_27_peaches_l1973_197302

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  monica : ℕ

/-- The conditions given in the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 16 ∧
  p.jake = p.steven - 7 ∧
  p.jake = p.jill + 9 ∧
  p.monica = 3 * p.jake

/-- Theorem: Given the conditions, Monica has 27 peaches -/
theorem monica_has_27_peaches (p : Peaches) (h : peach_conditions p) : p.monica = 27 := by
  sorry

end NUMINAMATH_CALUDE_monica_has_27_peaches_l1973_197302


namespace NUMINAMATH_CALUDE_chocolate_sales_l1973_197353

theorem chocolate_sales (cost_price selling_price : ℝ) (N : ℕ) : 
  (121 * cost_price = N * selling_price) →
  (selling_price = cost_price * (1 + 57.142857142857146 / 100)) →
  N = 77 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_sales_l1973_197353


namespace NUMINAMATH_CALUDE_abc_order_l1973_197335

theorem abc_order : 
  let a : ℝ := 1/2
  let b : ℝ := Real.log (3/2)
  let c : ℝ := (π/2) * Real.sin (1/2)
  b < a ∧ a < c :=
by sorry

end NUMINAMATH_CALUDE_abc_order_l1973_197335


namespace NUMINAMATH_CALUDE_park_population_l1973_197394

theorem park_population (lions leopards elephants zebras : ℕ) : 
  lions = 200 →
  lions = 2 * leopards →
  elephants = (lions + leopards) / 2 →
  zebras = elephants + leopards →
  lions + leopards + elephants + zebras = 700 := by
sorry

end NUMINAMATH_CALUDE_park_population_l1973_197394


namespace NUMINAMATH_CALUDE_trader_gain_percentage_l1973_197384

/-- The gain percentage of a trader selling pens -/
def gain_percentage (num_sold : ℕ) (num_gain : ℕ) : ℚ :=
  (num_gain : ℚ) / (num_sold : ℚ) * 100

/-- Theorem: The trader's gain percentage is 33.33% -/
theorem trader_gain_percentage : 
  ∃ (ε : ℚ), abs (gain_percentage 90 30 - 100/3) < ε ∧ ε < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_trader_gain_percentage_l1973_197384


namespace NUMINAMATH_CALUDE_number_problem_l1973_197308

theorem number_problem : ∃! x : ℝ, x + (2/3) * x + 1 = 10 ∧ x = 27/5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1973_197308


namespace NUMINAMATH_CALUDE_math_club_female_members_l1973_197382

theorem math_club_female_members :
  ∀ (female_members male_members : ℕ),
    female_members > 0 →
    male_members = 2 * female_members →
    female_members + male_members = 18 →
    female_members = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_club_female_members_l1973_197382


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1973_197390

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof :
  rectangle_area 1600 10 = 160 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1973_197390


namespace NUMINAMATH_CALUDE_negative_a_range_l1973_197377

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x < 1/2 ∨ x > 3}

-- Theorem statement
theorem negative_a_range (a : ℝ) (h_neg : a < 0) :
  (complementA ∩ B a = B a) ↔ -1/4 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_a_range_l1973_197377


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1973_197361

theorem system_of_inequalities_solution (x : ℝ) :
  (x^2 > x + 2 ∧ 4*x^2 ≤ 4*x + 15) ↔ 
  (x ∈ Set.Icc (-3/2) (-1) ∪ Set.Ioc 2 (5/2)) :=
sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1973_197361


namespace NUMINAMATH_CALUDE_largest_fraction_l1973_197309

theorem largest_fraction :
  let a := 5 / 13
  let b := 7 / 16
  let c := 23 / 46
  let d := 51 / 101
  let e := 203 / 405
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l1973_197309


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1973_197326

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 25 ∧ |x₂ - 3| = 25 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 50 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l1973_197326


namespace NUMINAMATH_CALUDE_solutions_equation1_solution_equation2_l1973_197365

-- Define the equations
def equation1 (x : ℝ) : Prop := (x - 2)^2 = 36
def equation2 (x : ℝ) : Prop := (2*x - 1)^3 = -125

-- Statement for the first equation
theorem solutions_equation1 : 
  (∃ x : ℝ, equation1 x) ∧ 
  (∀ x : ℝ, equation1 x ↔ (x = 8 ∨ x = -4)) :=
sorry

-- Statement for the second equation
theorem solution_equation2 : 
  (∃ x : ℝ, equation2 x) ∧
  (∀ x : ℝ, equation2 x ↔ x = -2) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solution_equation2_l1973_197365


namespace NUMINAMATH_CALUDE_registered_students_calculation_l1973_197347

/-- The number of students registered for a science course. -/
def registered_students (students_yesterday : ℕ) (students_absent_today : ℕ) : ℕ :=
  let students_today := (2 * students_yesterday) - (2 * students_yesterday / 10)
  students_today + students_absent_today

/-- Theorem stating the number of registered students given the problem conditions. -/
theorem registered_students_calculation :
  registered_students 70 30 = 156 := by
  sorry

#eval registered_students 70 30

end NUMINAMATH_CALUDE_registered_students_calculation_l1973_197347


namespace NUMINAMATH_CALUDE_logarithm_equality_implies_golden_ratio_l1973_197320

theorem logarithm_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 9 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_implies_golden_ratio_l1973_197320


namespace NUMINAMATH_CALUDE_bananas_in_basket_e_l1973_197313

/-- Given 5 baskets of fruits with an average of 25 fruits per basket, 
    where basket A contains 15 apples, B has 30 mangoes, C has 20 peaches, 
    D has 25 pears, and E has an unknown number of bananas, 
    prove that basket E contains 35 bananas. -/
theorem bananas_in_basket_e :
  let num_baskets : ℕ := 5
  let avg_fruits_per_basket : ℕ := 25
  let fruits_a : ℕ := 15
  let fruits_b : ℕ := 30
  let fruits_c : ℕ := 20
  let fruits_d : ℕ := 25
  let total_fruits : ℕ := num_baskets * avg_fruits_per_basket
  let fruits_abcd : ℕ := fruits_a + fruits_b + fruits_c + fruits_d
  let fruits_e : ℕ := total_fruits - fruits_abcd
  fruits_e = 35 := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_basket_e_l1973_197313


namespace NUMINAMATH_CALUDE_indeterminate_disjunction_l1973_197367

theorem indeterminate_disjunction (p q : Prop) 
  (h1 : ¬p) 
  (h2 : ¬(p ∧ q)) : 
  ¬∀ (r : Prop), r ↔ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_indeterminate_disjunction_l1973_197367


namespace NUMINAMATH_CALUDE_output_is_72_l1973_197379

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 38 then step1 * 2 else step1 - 10

theorem output_is_72 : function_machine 12 = 72 := by
  sorry

end NUMINAMATH_CALUDE_output_is_72_l1973_197379


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2903_l1973_197362

theorem smallest_prime_factor_of_2903 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2903 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2903 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2903_l1973_197362


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l1973_197323

theorem smallest_five_digit_mod_9 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≡ 4 [MOD 9] → 10003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_l1973_197323


namespace NUMINAMATH_CALUDE_compute_expression_l1973_197386

theorem compute_expression : 
  20 * (200 / 3 + 36 / 9 + 16 / 25 + 3) = 13212.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1973_197386


namespace NUMINAMATH_CALUDE_eighty_percent_of_forty_l1973_197331

theorem eighty_percent_of_forty (x : ℚ) : x * 20 + 16 = 32 → x = 4/5 := by sorry

end NUMINAMATH_CALUDE_eighty_percent_of_forty_l1973_197331


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l1973_197304

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 5/8
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 15/32 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l1973_197304


namespace NUMINAMATH_CALUDE_cornmeal_mixture_proof_l1973_197366

/-- Proves that mixing 40 pounds of cornmeal with soybean meal results in a 280 lb mixture
    that is 13% protein, given that soybean meal is 14% protein and cornmeal is 7% protein. -/
theorem cornmeal_mixture_proof (total_weight : ℝ) (soybean_protein : ℝ) (cornmeal_protein : ℝ)
    (desired_protein : ℝ) (cornmeal_weight : ℝ) :
  total_weight = 280 →
  soybean_protein = 0.14 →
  cornmeal_protein = 0.07 →
  desired_protein = 0.13 →
  cornmeal_weight = 40 →
  let soybean_weight := total_weight - cornmeal_weight
  (soybean_protein * soybean_weight + cornmeal_protein * cornmeal_weight) / total_weight = desired_protein :=
by sorry

end NUMINAMATH_CALUDE_cornmeal_mixture_proof_l1973_197366


namespace NUMINAMATH_CALUDE_student_calculation_error_l1973_197332

theorem student_calculation_error (N : ℝ) : (5/4)*N - (4/5)*N = 36 → N = 80 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_error_l1973_197332


namespace NUMINAMATH_CALUDE_second_integer_value_l1973_197395

/-- Given three consecutive odd integers where the sum of the first and third is 144,
    prove that the second integer is 72. -/
theorem second_integer_value (a b c : ℤ) : 
  (∃ n : ℤ, a = n - 2 ∧ b = n ∧ c = n + 2) →  -- consecutive odd integers
  (a + c = 144) →                            -- sum of first and third is 144
  b = 72 :=                                  -- second integer is 72
by sorry

end NUMINAMATH_CALUDE_second_integer_value_l1973_197395


namespace NUMINAMATH_CALUDE_commodity_price_equality_l1973_197359

/-- The year when commodity X costs 40 cents more than commodity Y -/
def target_year : ℕ := 2007

/-- The base year for price comparison -/
def base_year : ℕ := 2001

/-- The initial price of commodity X in dollars -/
def initial_price_X : ℚ := 4.20

/-- The initial price of commodity Y in dollars -/
def initial_price_Y : ℚ := 4.40

/-- The yearly price increase of commodity X in dollars -/
def price_increase_X : ℚ := 0.30

/-- The yearly price increase of commodity Y in dollars -/
def price_increase_Y : ℚ := 0.20

/-- The price difference between X and Y in the target year, in dollars -/
def price_difference : ℚ := 0.40

theorem commodity_price_equality :
  initial_price_X + price_increase_X * (target_year - base_year : ℚ) =
  initial_price_Y + price_increase_Y * (target_year - base_year : ℚ) + price_difference :=
by sorry

end NUMINAMATH_CALUDE_commodity_price_equality_l1973_197359


namespace NUMINAMATH_CALUDE_karl_drove_420_miles_l1973_197314

/-- Represents Karl's car and trip details --/
structure KarlsTrip where
  miles_per_gallon : ℝ
  tank_capacity : ℝ
  initial_distance : ℝ
  gas_bought : ℝ
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the trip details --/
def total_distance (trip : KarlsTrip) : ℝ :=
  trip.initial_distance

/-- Theorem stating that Karl drove 420 miles --/
theorem karl_drove_420_miles :
  let trip := KarlsTrip.mk 30 16 420 10 (3/4)
  total_distance trip = 420 := by sorry

end NUMINAMATH_CALUDE_karl_drove_420_miles_l1973_197314


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1973_197393

/-- Given a set {6, 13, 18, 4, x} where 10 is the arithmetic mean, prove that x = 9 -/
theorem arithmetic_mean_problem (x : ℝ) : 
  (6 + 13 + 18 + 4 + x) / 5 = 10 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1973_197393


namespace NUMINAMATH_CALUDE_car_speed_difference_l1973_197301

/-- Prove that given two cars P and R traveling 300 miles, where car R's speed is 34.05124837953327 mph
    and car P takes 2 hours less than car R, the difference in their average speeds is 10 mph. -/
theorem car_speed_difference (distance : ℝ) (speed_R : ℝ) (time_difference : ℝ) :
  distance = 300 →
  speed_R = 34.05124837953327 →
  time_difference = 2 →
  let time_R := distance / speed_R
  let time_P := time_R - time_difference
  let speed_P := distance / time_P
  speed_P - speed_R = 10 := by sorry

end NUMINAMATH_CALUDE_car_speed_difference_l1973_197301


namespace NUMINAMATH_CALUDE_find_x_l1973_197336

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 1/x}

theorem find_x : ∃ x : ℝ, B x ⊆ A ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1973_197336


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1973_197397

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.arctan ((2 * b / a) / (1 - b^2 / a^2)) = π / 4) →
  a / b = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1973_197397


namespace NUMINAMATH_CALUDE_first_file_size_is_80_l1973_197349

/-- Calculates the size of the first file given the internet speed, total download time, and sizes of two other files. -/
def first_file_size (speed : ℝ) (time : ℝ) (file2 : ℝ) (file3 : ℝ) : ℝ :=
  speed * time * 60 - file2 - file3

/-- Proves that given the specified conditions, the size of the first file is 80 megabits. -/
theorem first_file_size_is_80 :
  first_file_size 2 2 90 70 = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_file_size_is_80_l1973_197349


namespace NUMINAMATH_CALUDE_harry_apples_l1973_197327

/-- The number of apples each person has -/
structure Apples where
  martha : ℕ
  tim : ℕ
  harry : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.martha = 68 ∧
  a.tim = a.martha - 30 ∧
  a.harry = a.tim / 2 ∧
  a.jane = (a.tim + a.martha) / 4

/-- The theorem stating that Harry has 19 apples -/
theorem harry_apples (a : Apples) (h : apple_conditions a) : a.harry = 19 := by
  sorry

end NUMINAMATH_CALUDE_harry_apples_l1973_197327


namespace NUMINAMATH_CALUDE_largest_angle_bound_l1973_197316

/-- Triangle DEF with sides e, f, and d -/
structure Triangle where
  e : ℝ
  f : ℝ
  d : ℝ

/-- The angle opposite to side d in degrees -/
def angle_opposite_d (t : Triangle) : ℝ := sorry

theorem largest_angle_bound (t : Triangle) (y : ℝ) :
  t.e = 2 →
  t.f = 2 →
  t.d > 2 * Real.sqrt 2 →
  (∀ z, z > y → angle_opposite_d t > z) →
  y = 120 := by sorry

end NUMINAMATH_CALUDE_largest_angle_bound_l1973_197316


namespace NUMINAMATH_CALUDE_train_length_l1973_197375

/-- Proves that a train traveling at 45 km/hr crossing a 255 m bridge in 30 seconds has a length of 120 m -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_speed * 1000 / 3600) * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1973_197375


namespace NUMINAMATH_CALUDE_student_number_problem_l1973_197396

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1973_197396


namespace NUMINAMATH_CALUDE_millet_majority_on_tuesday_l1973_197354

/-- Represents the proportion of millet seeds remaining after birds eat -/
def milletRemaining : ℝ := 0.7

/-- Calculates the amount of millet seeds in the feeder after n days -/
def milletAmount (n : ℕ) : ℝ := 1 - milletRemaining ^ n

/-- The day when more than half the seeds are millet -/
def milletMajorityDay : ℕ := 2

theorem millet_majority_on_tuesday :
  milletAmount milletMajorityDay > 0.5 ∧
  ∀ k : ℕ, k < milletMajorityDay → milletAmount k ≤ 0.5 :=
sorry

end NUMINAMATH_CALUDE_millet_majority_on_tuesday_l1973_197354


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1973_197322

/-- The distance from a point P(-2, 3) to the y-axis is 2. -/
theorem distance_to_y_axis :
  let P : ℝ × ℝ := (-2, 3)
  abs P.1 = 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1973_197322


namespace NUMINAMATH_CALUDE_total_fish_fillets_l1973_197380

theorem total_fish_fillets (team1 team2 team3 : ℕ) 
  (h1 : team1 = 189) 
  (h2 : team2 = 131) 
  (h3 : team3 = 180) : 
  team1 + team2 + team3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_fillets_l1973_197380


namespace NUMINAMATH_CALUDE_alyssa_total_games_l1973_197392

/-- The total number of soccer games Alyssa will attend over three years -/
def total_games (this_year last_year next_year : ℕ) : ℕ :=
  this_year + last_year + next_year

/-- Proof that Alyssa will attend 39 soccer games in total -/
theorem alyssa_total_games :
  total_games 11 13 15 = 39 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_total_games_l1973_197392


namespace NUMINAMATH_CALUDE_tiffany_towels_l1973_197337

theorem tiffany_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) :
  packs * towels_per_pack = 27 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_towels_l1973_197337


namespace NUMINAMATH_CALUDE_project_hours_difference_l1973_197311

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 153 →
  kate_hours + 2 * kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1973_197311


namespace NUMINAMATH_CALUDE_software_contract_probability_l1973_197376

theorem software_contract_probability
  (p_hardware : ℝ)
  (p_at_least_one : ℝ)
  (p_both : ℝ)
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 11/30) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_software_contract_probability_l1973_197376


namespace NUMINAMATH_CALUDE_slide_problem_l1973_197340

/-- The number of additional boys who went down the slide -/
def additional_boys (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem slide_problem (initial : ℕ) (total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) : 
  additional_boys initial total = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_slide_problem_l1973_197340


namespace NUMINAMATH_CALUDE_reflect_distance_C_l1973_197355

/-- The length of the segment from a point to its reflection over the x-axis --/
def reflect_distance (p : ℝ × ℝ) : ℝ :=
  2 * |p.2|

theorem reflect_distance_C : reflect_distance (-3, 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_reflect_distance_C_l1973_197355


namespace NUMINAMATH_CALUDE_class_size_is_37_l1973_197371

/-- Represents the number of students in a class with specific age distribution. -/
def number_of_students (common_age : ℕ) (total_age_sum : ℕ) : ℕ :=
  (total_age_sum + 3) / common_age

/-- Theorem stating the number of students in the class is 37. -/
theorem class_size_is_37 :
  ∃ (common_age : ℕ),
    common_age > 0 ∧
    number_of_students common_age 330 = 37 ∧
    330 = 7 * (common_age - 1) + 2 * (common_age + 2) + (37 - 9) * common_age :=
sorry

end NUMINAMATH_CALUDE_class_size_is_37_l1973_197371


namespace NUMINAMATH_CALUDE_yellow_crayon_count_l1973_197346

theorem yellow_crayon_count (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  red = 14 → 
  blue = red + 5 → 
  yellow = 2 * blue - 6 → 
  yellow = 32 := by
sorry

end NUMINAMATH_CALUDE_yellow_crayon_count_l1973_197346


namespace NUMINAMATH_CALUDE_students_taking_only_history_l1973_197373

theorem students_taking_only_history (total : ℕ) (history : ℕ) (statistics : ℕ) (physics : ℕ) (chemistry : ℕ)
  (hist_stat : ℕ) (hist_phys : ℕ) (hist_chem : ℕ) (stat_phys : ℕ) (stat_chem : ℕ) (phys_chem : ℕ) (all_four : ℕ)
  (h_total : total = 500)
  (h_history : history = 150)
  (h_statistics : statistics = 130)
  (h_physics : physics = 120)
  (h_chemistry : chemistry = 100)
  (h_hist_stat : hist_stat = 60)
  (h_hist_phys : hist_phys = 50)
  (h_hist_chem : hist_chem = 40)
  (h_stat_phys : stat_phys = 35)
  (h_stat_chem : stat_chem = 30)
  (h_phys_chem : phys_chem = 25)
  (h_all_four : all_four = 20) :
  history - hist_stat - hist_phys - hist_chem + all_four = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_only_history_l1973_197373


namespace NUMINAMATH_CALUDE_german_students_count_l1973_197319

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → french = 41 → both = 9 → neither = 24 → 
  ∃ german : ℕ, german = 22 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_german_students_count_l1973_197319


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1973_197399

theorem inequality_solution_set (x : ℝ) : 
  (x / (x + 1) + (x + 3) / (2 * x) ≥ 2) ↔ (0 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1973_197399


namespace NUMINAMATH_CALUDE_hidden_primes_average_l1973_197307

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem hidden_primes_average (visible1 visible2 visible3 hidden1 hidden2 hidden3 : ℕ) :
  visible1 = 42 →
  visible2 = 59 →
  visible3 = 36 →
  is_prime hidden1 →
  is_prime hidden2 →
  is_prime hidden3 →
  visible1 + hidden1 = visible2 + hidden2 →
  visible2 + hidden2 = visible3 + hidden3 →
  visible1 ≠ visible2 ∧ visible2 ≠ visible3 ∧ visible1 ≠ visible3 →
  hidden1 ≠ hidden2 ∧ hidden2 ≠ hidden3 ∧ hidden1 ≠ hidden3 →
  (hidden1 + hidden2 + hidden3) / 3 = 56 / 3 := by
sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l1973_197307


namespace NUMINAMATH_CALUDE_jake_weight_ratio_l1973_197358

/-- Jake's weight problem -/
theorem jake_weight_ratio :
  let jake_present_weight : ℚ := 196
  let total_weight : ℚ := 290
  let weight_loss : ℚ := 8
  let jake_new_weight := jake_present_weight - weight_loss
  let sister_weight := total_weight - jake_present_weight
  jake_new_weight / sister_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_ratio_l1973_197358


namespace NUMINAMATH_CALUDE_unique_fraction_property_l1973_197312

theorem unique_fraction_property : ∃! (a b : ℕ), 
  b ≠ 0 ∧ 
  (a : ℚ) / b = (a + 4 : ℚ) / (b + 10) ∧ 
  (a : ℚ) / b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_property_l1973_197312


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1973_197345

theorem largest_prime_factor_of_1729 : ∃ (p : Nat), p.Prime ∧ p ∣ 1729 ∧ ∀ (q : Nat), q.Prime → q ∣ 1729 → q ≤ p := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l1973_197345


namespace NUMINAMATH_CALUDE_game_draw_fraction_l1973_197372

theorem game_draw_fraction (ben_wins tom_wins : ℚ) 
  (h1 : ben_wins = 4/9) 
  (h2 : tom_wins = 1/3) : 
  1 - (ben_wins + tom_wins) = 2/9 := by
sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l1973_197372


namespace NUMINAMATH_CALUDE_harry_potion_kits_l1973_197360

/-- The number of spellbooks Harry needs to buy -/
def num_spellbooks : ℕ := 5

/-- The cost of one spellbook in gold -/
def cost_spellbook : ℕ := 5

/-- The cost of one potion kit in silver -/
def cost_potion_kit : ℕ := 20

/-- The cost of one owl in gold -/
def cost_owl : ℕ := 28

/-- The number of silver in one gold -/
def silver_per_gold : ℕ := 9

/-- The total amount Harry will pay in silver -/
def total_cost : ℕ := 537

/-- The number of potion kits Harry needs to buy -/
def num_potion_kits : ℕ := (total_cost - (num_spellbooks * cost_spellbook * silver_per_gold + cost_owl * silver_per_gold)) / cost_potion_kit

theorem harry_potion_kits : num_potion_kits = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_potion_kits_l1973_197360


namespace NUMINAMATH_CALUDE_largest_divisible_n_l1973_197389

theorem largest_divisible_n : ∃ (n : ℕ), n = 910 ∧ 
  (∀ m : ℕ, m > n → ¬(m - 10 ∣ m^3 - 100)) ∧ 
  (n - 10 ∣ n^3 - 100) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l1973_197389


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1973_197387

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Moves the last digit to the front -/
def ThreeDigitNumber.rotateDigits (n : ThreeDigitNumber) : ThreeDigitNumber :=
  ⟨n.ones, n.hundreds, n.tens, by sorry⟩

theorem unique_three_digit_number :
  ∃! (n : ThreeDigitNumber),
    n.ones = 1 ∧
    (n.toNat - n.rotateDigits.toNat : Int) = (10 * (3 ^ 2) : Int) ∧
    n.toNat = 211 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1973_197387


namespace NUMINAMATH_CALUDE_max_triangle_area_l1973_197310

/-- The maximum area of a triangle ABC with side length constraints -/
theorem max_triangle_area (AB BC CA : ℝ) 
  (hAB : 0 ≤ AB ∧ AB ≤ 1)
  (hBC : 1 ≤ BC ∧ BC ≤ 2)
  (hCA : 2 ≤ CA ∧ CA ≤ 3) :
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (a : ℝ), (∃ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ 1 ∧
    1 ≤ y ∧ y ≤ 2 ∧
    2 ≤ z ∧ z ≤ 3 ∧
    a = (x + y + z) / 2 * ((x + y + z) / 2 - x) * ((x + y + z) / 2 - y) * ((x + y + z) / 2 - z)) →
  a ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l1973_197310


namespace NUMINAMATH_CALUDE_line_segment_proportions_l1973_197351

/-- Given line segments a and b, prove the fourth proportional and mean proportional -/
theorem line_segment_proportions (a b : ℝ) (ha : a = 5) (hb : b = 3) :
  let fourth_prop := b * (a - b) / a
  let mean_prop := Real.sqrt ((a + b) * (a - b))
  fourth_prop = 1.2 ∧ mean_prop = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportions_l1973_197351


namespace NUMINAMATH_CALUDE_right_triangle_vector_problem_l1973_197356

/-- Given a right-angled triangle ABC where AB is the hypotenuse,
    vector CA = (3, -9), and vector CB = (-3, x), prove that x = -1. -/
theorem right_triangle_vector_problem (A B C : ℝ × ℝ) (x : ℝ) :
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 
    + (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 →
  (C.1 - A.1, C.2 - A.2) = (3, -9) →
  (C.1 - B.1, C.2 - B.2) = (-3, x) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_vector_problem_l1973_197356


namespace NUMINAMATH_CALUDE_division_properties_7529_l1973_197341

theorem division_properties_7529 : 
  (7529 % 9 = 5) ∧ ¬(11 ∣ 7529) := by
  sorry

end NUMINAMATH_CALUDE_division_properties_7529_l1973_197341


namespace NUMINAMATH_CALUDE_infinite_solutions_iff_a_eq_neg_twelve_l1973_197339

theorem infinite_solutions_iff_a_eq_neg_twelve (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_iff_a_eq_neg_twelve_l1973_197339


namespace NUMINAMATH_CALUDE_hotel_towels_l1973_197330

/-- A hotel with a fixed number of rooms, people per room, and towels per person. -/
structure Hotel where
  rooms : ℕ
  peoplePerRoom : ℕ
  towelsPerPerson : ℕ

/-- Calculate the total number of towels handed out in a full hotel. -/
def totalTowels (h : Hotel) : ℕ :=
  h.rooms * h.peoplePerRoom * h.towelsPerPerson

/-- Theorem stating that a specific hotel configuration hands out 60 towels. -/
theorem hotel_towels :
  ∃ (h : Hotel), h.rooms = 10 ∧ h.peoplePerRoom = 3 ∧ h.towelsPerPerson = 2 ∧ totalTowels h = 60 :=
by
  sorry


end NUMINAMATH_CALUDE_hotel_towels_l1973_197330
