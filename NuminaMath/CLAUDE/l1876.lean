import Mathlib

namespace potato_yield_difference_l1876_187661

/-- Represents the yield difference between varietal and non-varietal potatoes -/
def yield_difference (
  non_varietal_area : ℝ
  ) (varietal_area : ℝ
  ) (yield_difference : ℝ
  ) : Prop :=
  let total_area := non_varietal_area + varietal_area
  let x := non_varietal_area
  let y := varietal_area
  ∃ (non_varietal_yield varietal_yield : ℝ),
    (non_varietal_yield * x + varietal_yield * y) / total_area = 
    non_varietal_yield + yield_difference ∧
    varietal_yield - non_varietal_yield = yield_difference

/-- Theorem stating the yield difference between varietal and non-varietal potatoes -/
theorem potato_yield_difference :
  yield_difference 14 4 90 := by
  sorry

end potato_yield_difference_l1876_187661


namespace correct_answers_for_given_score_l1876_187626

/-- Represents a test with a scoring system and a student's performance. -/
structure Test where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers. -/
def calculate_score (test : Test) : ℤ :=
  (test.correct_answers : ℤ) - 2 * ((test.total_questions - test.correct_answers) : ℤ)

theorem correct_answers_for_given_score (test : Test) :
  test.total_questions = 100 ∧
  test.score = 64 ∧
  calculate_score test = test.score →
  test.correct_answers = 88 := by
  sorry

end correct_answers_for_given_score_l1876_187626


namespace division_problem_l1876_187650

theorem division_problem : (120 : ℚ) / ((6 : ℚ) / 2) = 40 := by sorry

end division_problem_l1876_187650


namespace cos_270_degrees_l1876_187640

theorem cos_270_degrees (h : ∀ θ, Real.cos (360 - θ) = Real.cos θ) : 
  Real.cos 270 = 0 := by
sorry

end cos_270_degrees_l1876_187640


namespace C_power_50_l1876_187675

def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 2; -8, -5]

theorem C_power_50 : C^50 = !![(-199 : ℤ), -100; 400, 199] := by sorry

end C_power_50_l1876_187675


namespace distinct_arrangements_apples_l1876_187644

def word_length : ℕ := 6
def repeated_letter_count : ℕ := 2
def single_letter_count : ℕ := 1
def number_of_single_letters : ℕ := 4

theorem distinct_arrangements_apples :
  (word_length.factorial) / (repeated_letter_count.factorial * (single_letter_count.factorial ^ number_of_single_letters)) = 360 := by
  sorry

end distinct_arrangements_apples_l1876_187644


namespace deck_width_is_four_feet_l1876_187641

/-- Given a rectangular pool and a surrounding deck, this theorem proves
    that the deck width is 4 feet under specific conditions. -/
theorem deck_width_is_four_feet 
  (pool_length : ℝ) 
  (pool_width : ℝ) 
  (total_area : ℝ) 
  (h1 : pool_length = 10)
  (h2 : pool_width = 12)
  (h3 : total_area = 360)
  (w : ℝ) -- deck width
  (h4 : (pool_length + 2 * w) * (pool_width + 2 * w) = total_area) :
  w = 4 := by
  sorry

#check deck_width_is_four_feet

end deck_width_is_four_feet_l1876_187641


namespace total_assignment_plans_l1876_187686

def number_of_male_doctors : ℕ := 6
def number_of_female_doctors : ℕ := 4
def number_of_selected_male_doctors : ℕ := 3
def number_of_selected_female_doctors : ℕ := 2
def number_of_regions : ℕ := 5

def assignment_plans : ℕ := 12960

theorem total_assignment_plans :
  (number_of_male_doctors = 6) →
  (number_of_female_doctors = 4) →
  (number_of_selected_male_doctors = 3) →
  (number_of_selected_female_doctors = 2) →
  (number_of_regions = 5) →
  assignment_plans = 12960 :=
by sorry

end total_assignment_plans_l1876_187686


namespace age_difference_l1876_187642

theorem age_difference : ∀ (a b : ℕ),
  (10 ≤ 10 * a + b) ∧ (10 * a + b < 100) ∧  -- Jack's age is two-digit
  (10 ≤ 10 * b + a) ∧ (10 * b + a < 100) ∧  -- Bill's age is two-digit
  (10 * a + b + 10 = 3 * (10 * b + a + 10))  -- In 10 years, Jack will be 3 times Bill's age
  → (10 * a + b) - (10 * b + a) = 54 := by
sorry

end age_difference_l1876_187642


namespace xy_value_l1876_187678

theorem xy_value (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = -9 := by
  sorry

end xy_value_l1876_187678


namespace least_three_digit_multiple_of_11_l1876_187621

theorem least_three_digit_multiple_of_11 : ∃ (n : ℕ), n = 110 ∧ 
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 11 ∣ m → n ≤ m) ∧ 
  n ≥ 100 ∧ n < 1000 ∧ 11 ∣ n :=
sorry

end least_three_digit_multiple_of_11_l1876_187621


namespace parabola_equation_l1876_187636

-- Define a parabola
structure Parabola where
  equation : ℝ → ℝ → Prop

-- Define the properties of the parabola
def has_vertex_at_origin (p : Parabola) : Prop :=
  p.equation 0 0

def focus_on_coordinate_axis (p : Parabola) : Prop :=
  ∃ (k : ℝ), (p.equation k 0 ∨ p.equation 0 k) ∧ k ≠ 0

def passes_through_point (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Theorem statement
theorem parabola_equation :
  ∀ (p : Parabola),
    has_vertex_at_origin p →
    focus_on_coordinate_axis p →
    passes_through_point p (-2) 4 →
    (∀ (x y : ℝ), p.equation x y ↔ x^2 = y) ∨
    (∀ (x y : ℝ), p.equation x y ↔ y^2 = -8*x) :=
sorry

end parabola_equation_l1876_187636


namespace equidistant_function_c_squared_l1876_187670

/-- A complex function that is equidistant from its input and the origin -/
def EquidistantFunction (f : ℂ → ℂ) : Prop :=
  ∀ z : ℂ, Complex.abs (f z - z) = Complex.abs (f z)

theorem equidistant_function_c_squared
  (a c : ℝ)
  (f : ℂ → ℂ)
  (h1 : f = fun z ↦ (a + c * Complex.I) * z)
  (h2 : EquidistantFunction f)
  (h3 : Complex.abs (a + c * Complex.I) = 5) :
  c^2 = 24.75 := by
  sorry

end equidistant_function_c_squared_l1876_187670


namespace quadratic_properties_l1876_187638

/-- A quadratic function with the property that y > 0 for -2 < x < 3 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ∀ x : ℝ, -2 < x → x < 3 → 0 < a * x^2 + b * x + c

/-- The properties of the quadratic function that we want to prove -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.b = -f.a ∧
  ∃ x₁ x₂ : ℝ, x₁ = -1/3 ∧ x₂ = 1/2 ∧
    f.c * x₁^2 - f.b * x₁ + f.a = 0 ∧
    f.c * x₂^2 - f.b * x₂ + f.a = 0 :=
by sorry

end quadratic_properties_l1876_187638


namespace fishing_contest_result_l1876_187617

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The number of salmons Hazel's father caught -/
def father_catch : ℕ := 27

/-- The total number of salmons caught by Hazel and her father -/
def total_catch : ℕ := hazel_catch + father_catch

theorem fishing_contest_result : total_catch = 51 := by
  sorry

end fishing_contest_result_l1876_187617


namespace polynomial_change_l1876_187695

/-- Given a polynomial f(x) = 2x^2 - 5 and a positive real number b,
    the change in the polynomial's value when x changes by ±b is 4bx ± 2b^2 -/
theorem polynomial_change (x b : ℝ) (h : b > 0) :
  let f : ℝ → ℝ := λ t ↦ 2 * t^2 - 5
  (f (x + b) - f x) = 4 * b * x + 2 * b^2 ∧
  (f (x - b) - f x) = -4 * b * x + 2 * b^2 := by
  sorry

end polynomial_change_l1876_187695


namespace mosquito_blood_consumption_proof_l1876_187674

/-- The number of drops of blood per liter -/
def drops_per_liter : ℕ := 5000

/-- The number of liters of blood loss that leads to death -/
def lethal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause death by feeding -/
def lethal_mosquito_count : ℕ := 750

/-- The number of drops of blood a single mosquito sucks in one feeding -/
def mosquito_blood_consumption : ℕ := 20

theorem mosquito_blood_consumption_proof :
  mosquito_blood_consumption = (drops_per_liter * lethal_blood_loss) / lethal_mosquito_count :=
by sorry

end mosquito_blood_consumption_proof_l1876_187674


namespace gcd_factorial_problem_l1876_187692

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2 * 2^3) = 5760 := by
  sorry

end gcd_factorial_problem_l1876_187692


namespace circle_line_intersection_l1876_187630

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define a line passing through point P
def line_through_P (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the chord length
def chord_length (k : ℝ) : ℝ := sorry

-- Define the arc ratio
def arc_ratio (k : ℝ) : ℝ := sorry

theorem circle_line_intersection :
  ∀ (k : ℝ),
  (chord_length k = 2 → (k = 0 ∨ k = 3/4)) ∧
  (arc_ratio k = 3/1 → (k = 1/3 ∨ k = -3)) :=
sorry

end circle_line_intersection_l1876_187630


namespace prob_two_defective_out_of_three_l1876_187676

/-- The probability of selecting exactly 2 defective items out of 3 randomly chosen items
    from a set of 100 products containing 10 defective items. -/
theorem prob_two_defective_out_of_three (total_products : ℕ) (defective_items : ℕ) 
    (selected_items : ℕ) (h1 : total_products = 100) (h2 : defective_items = 10) 
    (h3 : selected_items = 3) :
  (Nat.choose defective_items 2 * Nat.choose (total_products - defective_items) 1) / 
  Nat.choose total_products selected_items = 27 / 1078 := by
  sorry

end prob_two_defective_out_of_three_l1876_187676


namespace edwards_initial_spending_l1876_187647

/-- Given Edward's initial balance, additional spending, and final balance,
    prove the amount he spent initially. -/
theorem edwards_initial_spending
  (initial_balance : ℕ)
  (additional_spending : ℕ)
  (final_balance : ℕ)
  (h1 : initial_balance = 34)
  (h2 : additional_spending = 8)
  (h3 : final_balance = 17)
  : initial_balance - additional_spending - final_balance = 9 := by
  sorry

#check edwards_initial_spending

end edwards_initial_spending_l1876_187647


namespace propositions_p_and_q_l1876_187684

theorem propositions_p_and_q : 
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b) ∧ 
  (∀ x : ℝ, Real.sin x + Real.cos x < 3/2) := by
  sorry

end propositions_p_and_q_l1876_187684


namespace arithmetic_calculation_l1876_187654

theorem arithmetic_calculation : 8 / 2 - 3 + 2 * (4 - 3)^2 = 3 := by
  sorry

end arithmetic_calculation_l1876_187654


namespace shaded_area_percentage_l1876_187643

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side_length : α

/-- Calculates the area of a square -/
def Square.area {α : Type*} [LinearOrderedField α] (s : Square α) : α :=
  s.side_length * s.side_length

/-- Represents the shaded regions in the square -/
structure ShadedRegions (α : Type*) [LinearOrderedField α] where
  small_square_side : α
  medium_square_side : α
  large_square_side : α

/-- Calculates the total shaded area -/
def ShadedRegions.total_area {α : Type*} [LinearOrderedField α] (sr : ShadedRegions α) : α :=
  sr.small_square_side * sr.small_square_side +
  (sr.medium_square_side * sr.medium_square_side - sr.small_square_side * sr.small_square_side) +
  (sr.large_square_side * sr.large_square_side - sr.medium_square_side * sr.medium_square_side)

/-- Theorem: The percentage of shaded area in square ABCD is (36/49) * 100 -/
theorem shaded_area_percentage
  (square : Square ℝ)
  (shaded : ShadedRegions ℝ)
  (h1 : square.side_length = 7)
  (h2 : shaded.small_square_side = 2)
  (h3 : shaded.medium_square_side = 4)
  (h4 : shaded.large_square_side = 6) :
  (shaded.total_area / square.area) * 100 = (36 / 49) * 100 :=
sorry

end shaded_area_percentage_l1876_187643


namespace sixteen_black_squares_with_odd_numbers_l1876_187627

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat
  number : Nat
  isBlack : Bool

/-- Represents a chessboard -/
def Chessboard := List Square

/-- Creates a standard 8x8 chessboard with alternating black and white squares,
    numbered from 1 to 64 left to right and top to bottom, with 1 on a black square -/
def createStandardChessboard : Chessboard := sorry

/-- Counts the number of black squares containing odd numbers on the chessboard -/
def countBlackSquaresWithOddNumbers (board : Chessboard) : Nat := sorry

/-- Theorem stating that there are exactly 16 black squares containing odd numbers
    on a standard 8x8 chessboard -/
theorem sixteen_black_squares_with_odd_numbers :
  ∀ (board : Chessboard),
    board = createStandardChessboard →
    countBlackSquaresWithOddNumbers board = 16 := by
  sorry

end sixteen_black_squares_with_odd_numbers_l1876_187627


namespace subtract_from_21_to_get_8_l1876_187697

theorem subtract_from_21_to_get_8 : ∃ x : ℝ, 21 - x = 8 ∧ x = 13 := by
  sorry

end subtract_from_21_to_get_8_l1876_187697


namespace rhombus_diagonal_length_l1876_187699

/-- Proves that in a rhombus with an area of 88 cm² and one diagonal of 11 cm, the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_area : area = 88) 
  (h_d1 : d1 = 11) 
  (h_rhombus_area : area = (d1 * d2) / 2) : d2 = 16 := by
  sorry

end rhombus_diagonal_length_l1876_187699


namespace milk_glass_density_ratio_l1876_187645

/-- Prove that the density of milk is 0.2 times the density of glass -/
theorem milk_glass_density_ratio 
  (m_CT : ℝ) -- mass of empty glass jar
  (m_M : ℝ)  -- mass of milk
  (V_CT : ℝ) -- volume of glass
  (V_M : ℝ)  -- volume of milk
  (h1 : m_CT + m_M = 3 * m_CT) -- mass of full jar is 3 times mass of empty jar
  (h2 : V_M = 10 * V_CT) -- volume of milk is 10 times volume of glass
  : m_M / V_M = 0.2 * (m_CT / V_CT) := by
  sorry

#check milk_glass_density_ratio

end milk_glass_density_ratio_l1876_187645


namespace prob_two_unmarked_correct_l1876_187671

/-- The probability of selecting two unmarked items from a set of 10 items where 3 are marked -/
def prob_two_unmarked (total : Nat) (marked : Nat) (select : Nat) : Rat :=
  if total = 10 ∧ marked = 3 ∧ select = 2 then
    7 / 15
  else
    0

theorem prob_two_unmarked_correct :
  prob_two_unmarked 10 3 2 = 7 / 15 := by
  sorry

end prob_two_unmarked_correct_l1876_187671


namespace system_solution_l1876_187687

theorem system_solution :
  ∀ x y z t : ℝ,
  (x * y - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18) →
  ((x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∧ t = 0)) :=
by sorry

end system_solution_l1876_187687


namespace calculate_savings_person_savings_l1876_187681

/-- Calculates a person's savings given their income sources and expenses --/
theorem calculate_savings (total_income : ℝ) 
  (source_a_percent source_b_percent source_c_percent : ℝ)
  (expense_a_percent expense_b_percent expense_c_percent : ℝ) : ℝ :=
  let source_a := source_a_percent * total_income
  let source_b := source_b_percent * total_income
  let source_c := source_c_percent * total_income
  let expense_a := expense_a_percent * source_a
  let expense_b := expense_b_percent * source_b
  let expense_c := expense_c_percent * source_c
  let total_expenses := expense_a + expense_b + expense_c
  total_income - total_expenses

/-- Proves that the person's savings is Rs. 19,005 given the specified conditions --/
theorem person_savings : 
  calculate_savings 21000 0.5 0.3 0.2 0.1 0.05 0.15 = 19005 := by
  sorry

end calculate_savings_person_savings_l1876_187681


namespace license_plate_count_l1876_187603

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters in a license plate --/
def letters_count : ℕ := 4

/-- The number of digits in a license plate --/
def digits_count : ℕ := 3

/-- The number of available digits (0-9) --/
def available_digits : ℕ := 10

/-- Calculates the number of license plate combinations --/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (Nat.choose (alphabet_size - 1) 2) *
  (Nat.choose letters_count 2) *
  2 *
  available_digits *
  (available_digits - 1) *
  (available_digits - 2)

theorem license_plate_count :
  license_plate_combinations = 67392000 := by
  sorry

end license_plate_count_l1876_187603


namespace camel_cannot_move_to_adjacent_l1876_187601

def Board := Fin 10 × Fin 10

def adjacent (a b : Board) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

def camel_move (a b : Board) : Prop :=
  (a.1 = b.1 + 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.1 = b.1 - 1 ∧ (a.2 = b.2 + 3 ∨ a.2 = b.2 - 3)) ∨
  (a.2 = b.2 + 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3)) ∨
  (a.2 = b.2 - 1 ∧ (a.1 = b.1 + 3 ∨ a.1 = b.1 - 3))

theorem camel_cannot_move_to_adjacent :
  ∀ (start finish : Board), adjacent start finish → ¬ camel_move start finish :=
by sorry

end camel_cannot_move_to_adjacent_l1876_187601


namespace product_mod_seven_l1876_187653

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_mod_seven_l1876_187653


namespace min_value_abc_l1876_187633

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (a + 3 * b) * (b + 3 * c) * (a * c + 2) ≥ 288 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 8 ∧
    (a' + 3 * b') * (b' + 3 * c') * (a' * c' + 2) = 288 :=
by sorry

end min_value_abc_l1876_187633


namespace motel_billing_solution_l1876_187649

/-- Represents the motel's billing system -/
structure MotelBilling where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Fixed rate for subsequent nights

/-- Calculates the total cost for a stay -/
def totalCost (billing : MotelBilling) (nights : ℕ) : ℝ :=
  billing.flatFee + billing.nightlyRate * (nights - 1 : ℝ) -
    if nights > 4 then 25 else 0

/-- The motel billing system satisfies the given conditions -/
theorem motel_billing_solution :
  ∃ (billing : MotelBilling),
    totalCost billing 4 = 215 ∧
    totalCost billing 7 = 360 ∧
    billing.flatFee = 45 := by
  sorry


end motel_billing_solution_l1876_187649


namespace identity_proof_l1876_187607

theorem identity_proof (a b x y θ φ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h1 : (a - b) * Real.sin (θ / 2) * Real.cos (φ / 2) + 
        (a + b) * Real.cos (θ / 2) * Real.sin (φ / 2) = 0)
  (h2 : x / a * Real.cos θ + y / b * Real.sin θ = 1)
  (h3 : x / a * Real.cos φ + y / b * Real.sin φ = 1) :
  x^2 / a^2 + (b^2 - a^2) / b^4 * y^2 = 1 := by
sorry

end identity_proof_l1876_187607


namespace locus_equidistant_point_line_l1876_187635

/-- The locus of points equidistant from a point and a line is a parabola -/
theorem locus_equidistant_point_line (x y : ℝ) : 
  let F : ℝ × ℝ := (0, -3)
  let line_eq : ℝ → ℝ → Prop := λ x y => y + 5 = 0
  let distance_to_point : ℝ × ℝ → ℝ := λ p => Real.sqrt ((p.1 - F.1)^2 + (p.2 - F.2)^2)
  let distance_to_line : ℝ × ℝ → ℝ := λ p => |p.2 + 5|
  distance_to_point (x, y) = distance_to_line (x, y) ↔ y = (1/4) * x^2 - 4 := by
sorry

end locus_equidistant_point_line_l1876_187635


namespace complex_equation_solution_l1876_187600

theorem complex_equation_solution (z : ℂ) : 2 * z * Complex.I = 1 + 3 * Complex.I → z = (3 / 2 : ℂ) - (1 / 2 : ℂ) * Complex.I := by
  sorry

end complex_equation_solution_l1876_187600


namespace mark_and_carolyn_money_l1876_187614

theorem mark_and_carolyn_money : 
  let mark_money : ℚ := 3/4
  let carolyn_money : ℚ := 3/10
  mark_money + carolyn_money = 21/20 := by sorry

end mark_and_carolyn_money_l1876_187614


namespace sum_of_integers_l1876_187677

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 56) :
  x.val + y.val = 12 * Real.sqrt 2 := by
  sorry

end sum_of_integers_l1876_187677


namespace triangle_existence_condition_l1876_187651

/-- A triangle with given altitudes and median -/
structure Triangle where
  ma : ℝ  -- altitude to side a
  mb : ℝ  -- altitude to side b
  sc : ℝ  -- median to side c
  ma_pos : 0 < ma
  mb_pos : 0 < mb
  sc_pos : 0 < sc

/-- The existence condition for a triangle with given altitudes and median -/
def triangle_exists (t : Triangle) : Prop :=
  t.ma < 2 * t.sc ∧ t.mb < 2 * t.sc

/-- Theorem stating the necessary and sufficient condition for triangle existence -/
theorem triangle_existence_condition (t : Triangle) :
  ∃ (triangle : Triangle), triangle.ma = t.ma ∧ triangle.mb = t.mb ∧ triangle.sc = t.sc ↔ triangle_exists t :=
sorry

end triangle_existence_condition_l1876_187651


namespace complement_A_intersect_B_l1876_187665

-- Define the universal set I
def I : Set (ℝ × ℝ) := Set.univ

-- Define set A
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 - 3 = p.1 - 2 ∧ p.1 ≠ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + 1}

-- State the theorem
theorem complement_A_intersect_B : (I \ A) ∩ B = {(2, 3)} := by sorry

end complement_A_intersect_B_l1876_187665


namespace function_properties_l1876_187608

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b (-2)) ∧
    (f a b (-2) = 4) ∧
    (a = 3 ∧ b = 0) ∧
    (∀ x ∈ Set.Icc (-3) 1, f a b x ≤ 4) ∧
    (∃ x ∈ Set.Icc (-3) 1, f a b x = 0) :=
by
  sorry


end function_properties_l1876_187608


namespace thabo_books_l1876_187602

/-- The number of books Thabo owns -/
def total_books : ℕ := 220

/-- The number of hardcover nonfiction books Thabo owns -/
def hardcover_nonfiction : ℕ := sorry

/-- The number of paperback nonfiction books Thabo owns -/
def paperback_nonfiction : ℕ := sorry

/-- The number of paperback fiction books Thabo owns -/
def paperback_fiction : ℕ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem thabo_books :
  (paperback_nonfiction = hardcover_nonfiction + 20) ∧
  (paperback_fiction = 2 * paperback_nonfiction) ∧
  (hardcover_nonfiction + paperback_nonfiction + paperback_fiction = total_books) →
  hardcover_nonfiction = 40 := by
  sorry

end thabo_books_l1876_187602


namespace cube_edge_ratio_l1876_187660

theorem cube_edge_ratio (a b : ℝ) (h : a ^ 3 / b ^ 3 = 8 / 1) : a / b = 2 / 1 := by
  sorry

end cube_edge_ratio_l1876_187660


namespace inequality_condition_l1876_187612

theorem inequality_condition (a b : ℝ) : 
  (a < b ∧ b < 0 → 1/a > 1/b) ∧ 
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(a < b ∧ b < 0) := by
  sorry

end inequality_condition_l1876_187612


namespace f_10_equals_222_l1876_187688

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_10_equals_222 (y : ℝ) (h : f 2 y = 30) : f 10 y = 222 := by
  sorry

end f_10_equals_222_l1876_187688


namespace family_income_problem_l1876_187622

/-- Proves that in a family of 4 members with an average income of 10000,
    if three members earn 8000, 6000, and 11000 respectively,
    then the income of the fourth member is 15000. -/
theorem family_income_problem (family_size : ℕ) (average_income : ℕ) 
  (member1_income : ℕ) (member2_income : ℕ) (member3_income : ℕ) :
  family_size = 4 →
  average_income = 10000 →
  member1_income = 8000 →
  member2_income = 6000 →
  member3_income = 11000 →
  average_income * family_size - (member1_income + member2_income + member3_income) = 15000 :=
by sorry

end family_income_problem_l1876_187622


namespace jelly_cost_theorem_l1876_187637

/-- The cost of jelly for all sandwiches is $1.68 --/
theorem jelly_cost_theorem (N B J : ℕ) : 
  N > 1 → 
  B > 0 → 
  J > 0 → 
  N * (3 * B + 7 * J) = 336 → 
  (N * J * 7 : ℚ) / 100 = 1.68 := by
  sorry

end jelly_cost_theorem_l1876_187637


namespace smallest_number_with_all_factors_l1876_187605

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_number_with_all_factors :
  ∃ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors alice_number m → k ≤ m :=
by sorry

end smallest_number_with_all_factors_l1876_187605


namespace complex_angle_proof_l1876_187672

theorem complex_angle_proof (z : ℂ) : z = -1 - Real.sqrt 3 * I → ∃ r θ : ℝ, z = r * Complex.exp (θ * I) ∧ θ = (4 * Real.pi) / 3 := by
  sorry

end complex_angle_proof_l1876_187672


namespace mod_eight_power_difference_l1876_187648

theorem mod_eight_power_difference : (47^2023 - 22^2023) % 8 = 1 := by
  sorry

end mod_eight_power_difference_l1876_187648


namespace sqrt_9x_lt_3x_squared_iff_x_gt_1_l1876_187613

theorem sqrt_9x_lt_3x_squared_iff_x_gt_1 :
  ∀ x : ℝ, x > 0 → (Real.sqrt (9 * x) < 3 * x^2 ↔ x > 1) := by
sorry

end sqrt_9x_lt_3x_squared_iff_x_gt_1_l1876_187613


namespace four_parts_of_400_l1876_187623

theorem four_parts_of_400 (a b c d : ℝ) 
  (sum_eq_400 : a + b + c + d = 400)
  (parts_equal : a + 1 = b - 2 ∧ b - 2 = 3 * c ∧ 3 * c = d / 4)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 := by
sorry

end four_parts_of_400_l1876_187623


namespace class_transfer_equation_l1876_187620

theorem class_transfer_equation (x : ℕ) : 
  (∀ (total : ℕ), total = 98 → 
    (∀ (transfer : ℕ), transfer = 3 →
      (total - x) + transfer = x - transfer)) ↔ 
  (98 - x) + 3 = x - 3 :=
sorry

end class_transfer_equation_l1876_187620


namespace tree_branches_after_eight_weeks_l1876_187668

def branch_growth (g : ℕ → ℕ) : Prop :=
  g 2 = 1 ∧
  g 3 = 2 ∧
  (∀ n ≥ 3, g (n + 1) = g n + g (n - 1)) ∧
  g 5 = 5

theorem tree_branches_after_eight_weeks (g : ℕ → ℕ) 
  (h : branch_growth g) : g 8 = 21 := by
  sorry

end tree_branches_after_eight_weeks_l1876_187668


namespace distinguishable_arrangements_l1876_187696

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 420 := by
  sorry

end distinguishable_arrangements_l1876_187696


namespace sum_of_squares_l1876_187646

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end sum_of_squares_l1876_187646


namespace team_combinations_eq_18018_l1876_187604

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 18

/-- The number of quadruplets in the team --/
def num_quadruplets : ℕ := 4

/-- The size of the team to be formed --/
def team_size : ℕ := 8

/-- The number of quadruplets that must be in the team --/
def required_quadruplets : ℕ := 2

/-- The number of ways to choose 8 players from a team of 18 players, 
    including exactly 2 out of 4 quadruplets --/
def team_combinations : ℕ :=
  choose num_quadruplets required_quadruplets * 
  choose (total_players - num_quadruplets) (team_size - required_quadruplets)

theorem team_combinations_eq_18018 : team_combinations = 18018 := by
  sorry

end team_combinations_eq_18018_l1876_187604


namespace smallest_multiple_forty_satisfies_forty_is_smallest_l1876_187657

theorem smallest_multiple (y : ℕ) : y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

theorem forty_satisfies : 800 ∣ (540 * 40) :=
sorry

theorem forty_is_smallest : ∀ y : ℕ, y > 0 ∧ 800 ∣ (540 * y) → y ≥ 40 :=
sorry

end smallest_multiple_forty_satisfies_forty_is_smallest_l1876_187657


namespace river_width_proof_l1876_187655

theorem river_width_proof (total_distance : ℝ) (prob_find : ℝ) (x : ℝ) : 
  total_distance = 500 →
  prob_find = 4/5 →
  x / total_distance = 1 - prob_find →
  x = 100 := by
sorry

end river_width_proof_l1876_187655


namespace floor_plus_s_eq_15_4_l1876_187680

theorem floor_plus_s_eq_15_4 (s : ℝ) : 
  (⌊s⌋ : ℝ) + s = 15.4 → s = 7.4 := by
sorry

end floor_plus_s_eq_15_4_l1876_187680


namespace line_through_circle_center_perpendicular_to_given_line_l1876_187659

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + 2*x + y^2 = 0

-- Define the given line equation
def given_line (x y : ℝ) : Prop := x + y = 0

-- Define the equation of the line we want to prove
def target_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_given_line :
  ∃ (cx cy : ℝ),
    (∀ x y, circle_equation x y ↔ (x - cx)^2 + (y - cy)^2 = (-cx)^2 + (-cy)^2) ∧
    target_line cx cy ∧
    (∀ x y, target_line x y → given_line x y → (x - cx) * (x - cx) + (y - cy) * (y - cy) = 0) :=
sorry

end line_through_circle_center_perpendicular_to_given_line_l1876_187659


namespace digit_reversal_l1876_187679

theorem digit_reversal (n : ℕ) : 
  let B := n^2 + 1
  (n^2 * (n^2 + 2)^2 = 1 * B^3 + 0 * B^2 + (B - 2) * B + (B - 1)) ∧
  (n^4 * (n^2 + 2)^2 = 1 * B^3 + (B - 2) * B^2 + 0 * B + (B - 1)) := by
sorry

end digit_reversal_l1876_187679


namespace pat_calculation_l1876_187609

theorem pat_calculation (x : ℝ) : (x / 7 + 10 = 20) → (x * 7 - 10 = 480) := by
  sorry

end pat_calculation_l1876_187609


namespace smallest_resolvable_debt_proof_smallest_resolvable_debt_achievable_l1876_187631

/-- The smallest positive debt that can be resolved using chairs and tables -/
def smallest_resolvable_debt : ℕ := 60

/-- The value of a chair in dollars -/
def chair_value : ℕ := 240

/-- The value of a table in dollars -/
def table_value : ℕ := 180

theorem smallest_resolvable_debt_proof :
  ∀ (d : ℕ), d > 0 →
  (∃ (c t : ℤ), d = chair_value * c + table_value * t) →
  d ≥ smallest_resolvable_debt := by
sorry

theorem smallest_resolvable_debt_achievable :
  ∃ (c t : ℤ), smallest_resolvable_debt = chair_value * c + table_value * t := by
sorry

end smallest_resolvable_debt_proof_smallest_resolvable_debt_achievable_l1876_187631


namespace y_values_from_x_equation_l1876_187673

theorem y_values_from_x_equation (x : ℝ) :
  x^2 + 5 * (x / (x - 3))^2 = 50 →
  ∃ y : ℝ, y = (x - 3)^2 * (x + 4) / (2*x - 5) ∧
    (∃ k : ℝ, (k = 5 + Real.sqrt 55 ∨ k = 5 - Real.sqrt 55 ∨
               k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6) ∧
              y = (k - 3)^2 * (k + 4) / (2*k - 5)) :=
by sorry

end y_values_from_x_equation_l1876_187673


namespace tax_deduction_for_jacob_l1876_187628

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a 2% tax rate, the local tax deduction is 50 cents. -/
theorem tax_deduction_for_jacob :
  localTaxDeduction 25 2 = 50 := by
  sorry

end tax_deduction_for_jacob_l1876_187628


namespace marie_erasers_l1876_187666

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by sorry

end marie_erasers_l1876_187666


namespace toothpicks_12th_stage_l1876_187685

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- Theorem: The 12th stage of the pattern contains 36 toothpicks -/
theorem toothpicks_12th_stage : toothpicks 12 = 36 := by
  sorry

end toothpicks_12th_stage_l1876_187685


namespace collinear_vectors_m_value_l1876_187629

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 2)
  collinear a b → m = (1 : ℝ) / 2 := by
  sorry

end collinear_vectors_m_value_l1876_187629


namespace f_has_zero_in_interval_l1876_187619

-- Define the function f(x) = x³ + 3x - 3
def f (x : ℝ) : ℝ := x^3 + 3*x - 3

-- Theorem statement
theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
sorry

end f_has_zero_in_interval_l1876_187619


namespace circle_condition_l1876_187606

/-- The equation of a circle in terms of parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0

/-- Theorem stating the condition for m to represent a circle -/
theorem circle_condition (m : ℝ) : 
  (∃ x y : ℝ, circle_equation x y m) ↔ m < 1 := by
  sorry

end circle_condition_l1876_187606


namespace total_cost_calculation_l1876_187689

def normal_pretzel_price : ℝ := 4
def discounted_pretzel_price : ℝ := 3.5
def normal_chip_price : ℝ := 7
def discounted_chip_price : ℝ := 6
def pretzel_discount_threshold : ℕ := 3
def chip_discount_threshold : ℕ := 2

def pretzel_packs_bought : ℕ := 3
def chip_packs_bought : ℕ := 4

def calculate_pretzel_cost (packs : ℕ) : ℝ :=
  if packs ≥ pretzel_discount_threshold then
    packs * discounted_pretzel_price
  else
    packs * normal_pretzel_price

def calculate_chip_cost (packs : ℕ) : ℝ :=
  if packs ≥ chip_discount_threshold then
    packs * discounted_chip_price
  else
    packs * normal_chip_price

theorem total_cost_calculation :
  calculate_pretzel_cost pretzel_packs_bought + calculate_chip_cost chip_packs_bought = 34.5 := by
  sorry

end total_cost_calculation_l1876_187689


namespace half_abs_diff_squares_plus_five_l1876_187683

theorem half_abs_diff_squares_plus_five : 
  (|20^2 - 12^2| / 2 : ℝ) + 5 = 133 := by sorry

end half_abs_diff_squares_plus_five_l1876_187683


namespace green_ball_probability_l1876_187632

/-- Represents a container of balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containerX : Container := ⟨3, 7⟩
def containerY : Container := ⟨7, 3⟩
def containerZ : Container := ⟨7, 3⟩

/-- The probability of selecting each container -/
def containerProb : ℚ := 1 / 3

/-- The probability of selecting a green ball -/
def greenBallProb : ℚ :=
  containerProb * greenProbability containerX +
  containerProb * greenProbability containerY +
  containerProb * greenProbability containerZ

theorem green_ball_probability :
  greenBallProb = 13 / 30 := by
  sorry

end green_ball_probability_l1876_187632


namespace statements_are_equivalent_l1876_187656

-- Define propositions
variable (R : Prop) -- R represents "It rains"
variable (G : Prop) -- G represents "I go outside"

-- Define the original statement
def original_statement : Prop := ¬R → ¬G

-- Define the equivalent statement
def equivalent_statement : Prop := G → R

-- Theorem stating the logical equivalence
theorem statements_are_equivalent : original_statement R G ↔ equivalent_statement R G := by
  sorry

end statements_are_equivalent_l1876_187656


namespace q_squared_minus_one_div_24_l1876_187625

/-- The largest prime number with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_24 : 24 ∣ (q^2 - 1) := by sorry

end q_squared_minus_one_div_24_l1876_187625


namespace system_solution_l1876_187698

theorem system_solution : ∃ (x y : ℝ), x = 4 ∧ y = 1 ∧ x - y = 3 ∧ 2*(x - y) = 6*y := by
  sorry

end system_solution_l1876_187698


namespace statement_to_equation_l1876_187618

theorem statement_to_equation (a : ℝ) : 
  (3 * a + 5 = 4 * a) ↔ 
  (∃ x : ℝ, x = 3 * a + 5 ∧ x = 4 * a) :=
by sorry

end statement_to_equation_l1876_187618


namespace projection_theorem_l1876_187658

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_theorem :
  let p := projection
  p (1, -2) = (3/2, -3/2) →
  p (-4, 1) = (-5/2, 5/2) := by sorry

end projection_theorem_l1876_187658


namespace man_speed_l1876_187662

/-- The speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed : Real) (time_to_pass : Real) :
  train_length = 110 ∧ 
  train_speed = 40 ∧ 
  time_to_pass = 9 →
  ∃ (man_speed : Real),
    man_speed > 0 ∧ 
    man_speed < train_speed ∧
    abs (man_speed - train_speed) * time_to_pass / 3600 = train_length / 1000 ∧
    abs (man_speed - 3.992) < 0.001 :=
sorry

end man_speed_l1876_187662


namespace max_value_expression_l1876_187690

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) * 
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 9.5 := by
  sorry

end max_value_expression_l1876_187690


namespace womens_average_age_l1876_187694

theorem womens_average_age (n : ℕ) (initial_avg : ℝ) :
  n = 8 ∧
  initial_avg > 0 ∧
  (n * initial_avg + 60) / n = initial_avg + 2 →
  60 / 2 = 30 := by
sorry

end womens_average_age_l1876_187694


namespace max_value_of_y_l1876_187663

-- Define the function y
def y (x a : ℝ) : ℝ := |x - a| + |x + 19| + |x - a - 96|

-- State the theorem
theorem max_value_of_y (a : ℝ) (h1 : 19 < a) (h2 : a < 96) :
  ∃ (max_y : ℝ), max_y = 211 ∧ ∀ x, a ≤ x → x ≤ 96 → y x a ≤ max_y :=
sorry

end max_value_of_y_l1876_187663


namespace lcm_36_105_l1876_187615

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l1876_187615


namespace inequality_implies_m_range_l1876_187667

theorem inequality_implies_m_range (m : ℝ) : 
  (∀ x > 0, (m * Real.exp x) / x ≥ 6 - 4 * x) → m ≥ 2 * Real.exp (-1/2) := by
  sorry

end inequality_implies_m_range_l1876_187667


namespace greatest_divisor_four_consecutive_integers_l1876_187639

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 24 ∧ ¬(m ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  ∀ k : ℕ, k ≤ 24 → (k ∣ (n * (n + 1) * (n + 2) * (n + 3))) :=
by sorry

end greatest_divisor_four_consecutive_integers_l1876_187639


namespace min_value_expression_l1876_187682

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) ≥ 12 ∧
  ((6 * a) / (b + 2 * c) + (6 * b) / (c + 2 * a) + (2 * c) / (a + 2 * b) + (6 * c) / (2 * a + b) = 12 ↔ a = b ∧ b = c) :=
by sorry

end min_value_expression_l1876_187682


namespace sum_exponents_15_factorial_l1876_187616

/-- The largest perfect square that divides n! -/
def largestPerfectSquareDivisor (n : ℕ) : ℕ := sorry

/-- The sum of the exponents of the prime factors of the square root of a number -/
def sumExponentsOfSquareRoot (n : ℕ) : ℕ := sorry

/-- Theorem stating that the sum of the exponents of the prime factors of the square root
    of the largest perfect square that divides 15! is equal to 10 -/
theorem sum_exponents_15_factorial :
  sumExponentsOfSquareRoot (largestPerfectSquareDivisor 15) = 10 := by sorry

end sum_exponents_15_factorial_l1876_187616


namespace expand_product_l1876_187693

theorem expand_product (x : ℝ) : 4 * (x + 3) * (2 * x + 7) = 8 * x^2 + 52 * x + 84 := by
  sorry

end expand_product_l1876_187693


namespace greatest_integer_satisfying_conditions_l1876_187634

theorem greatest_integer_satisfying_conditions :
  ∃ (n : ℕ), n < 150 ∧
  (∃ (a : ℕ), n = 9 * a - 2) ∧
  (∃ (b : ℕ), n = 11 * b - 4) ∧
  (∃ (c : ℕ), n = 5 * c + 1) ∧
  (∀ (m : ℕ), m < 150 →
    (∃ (a' : ℕ), m = 9 * a' - 2) →
    (∃ (b' : ℕ), m = 11 * b' - 4) →
    (∃ (c' : ℕ), m = 5 * c' + 1) →
    m ≤ n) ∧
  n = 142 :=
by sorry

end greatest_integer_satisfying_conditions_l1876_187634


namespace transformed_graph_equivalence_l1876_187691

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Define the horizontal shift transformation
noncomputable def shift (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - h)

-- Define the horizontal compression transformation
noncomputable def compress (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (k * x)

-- Theorem statement
theorem transformed_graph_equivalence :
  ∀ x : ℝ, g x = (compress (1/2) (shift (-1/2) f)) x := by sorry

end transformed_graph_equivalence_l1876_187691


namespace mineral_water_recycling_l1876_187652

/-- Calculates the total number of bottles that can be drunk given an initial number of bottles -/
def total_bottles_drunk (initial_bottles : ℕ) : ℕ :=
  sorry

/-- Calculates the initial number of bottles needed to drink a given total number of bottles -/
def initial_bottles_needed (total_drunk : ℕ) : ℕ :=
  sorry

theorem mineral_water_recycling :
  (total_bottles_drunk 1999 = 2665) ∧
  (initial_bottles_needed 3126 = 2345) :=
by sorry

end mineral_water_recycling_l1876_187652


namespace number_properties_l1876_187611

theorem number_properties (n : ℕ) (h : n > 0) :
  (∃ (factors : Set ℕ), Finite factors ∧ ∀ k ∈ factors, n % k = 0) ∧
  (∃ (multiples : Set ℕ), ¬Finite multiples ∧ ∀ m ∈ multiples, m % n = 0) ∧
  (∀ k : ℕ, k ∣ n → k ≥ 1) ∧
  (∀ k : ℕ, k ∣ n → k ≤ n) ∧
  (∀ m : ℕ, n ∣ m → m ≥ n) := by
sorry

end number_properties_l1876_187611


namespace outfit_combinations_l1876_187610

def num_shirts : ℕ := 5
def num_pants : ℕ := 6
def num_hats : ℕ := 2

theorem outfit_combinations : num_shirts * num_pants * num_hats = 60 := by
  sorry

end outfit_combinations_l1876_187610


namespace unique_fixed_point_of_rotation_invariant_function_l1876_187624

/-- A function is rotation-invariant if rotating its graph by π/2 around the origin
    results in the same graph. -/
def RotationInvariant (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f (-y) = x

theorem unique_fixed_point_of_rotation_invariant_function
  (f : ℝ → ℝ) (h : RotationInvariant f) :
  ∃! x, f x = x :=
sorry

end unique_fixed_point_of_rotation_invariant_function_l1876_187624


namespace min_dihedral_angle_cube_l1876_187664

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A unit cube with vertices ABCD-A₁B₁C₁D₁ -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- A point P on edge AB of the cube -/
def P (cube : UnitCube) (t : ℝ) : Point3D :=
  { x := cube.A.x + t * (cube.B.x - cube.A.x),
    y := cube.A.y + t * (cube.B.y - cube.A.y),
    z := cube.A.z + t * (cube.B.z - cube.A.z) }

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 : Plane3D) (plane2 : Plane3D) : ℝ := sorry

/-- The plane PDB₁ -/
def planePDB₁ (cube : UnitCube) (p : Point3D) : Plane3D := sorry

/-- The plane ADD₁A₁ -/
def planeADD₁A₁ (cube : UnitCube) : Plane3D := sorry

theorem min_dihedral_angle_cube (cube : UnitCube) :
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
    ∀ (s : ℝ), s ∈ Set.Icc 0 1 →
      dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) ≤
      dihedralAngle (planePDB₁ cube (P cube s)) (planeADD₁A₁ cube) ∧
    dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) = Real.arctan (Real.sqrt 2 / 2) := by
  sorry


end min_dihedral_angle_cube_l1876_187664


namespace similar_triangles_dimensions_l1876_187669

theorem similar_triangles_dimensions (h₁ base₁ h₂ base₂ : ℝ) : 
  h₁ > 0 → base₁ > 0 → h₂ > 0 → base₂ > 0 →
  (h₁ * base₁) / (h₂ * base₂) = 1 / 9 →
  h₁ = 5 → base₁ = 6 →
  h₂ = 15 ∧ base₂ = 18 := by
  sorry

end similar_triangles_dimensions_l1876_187669
