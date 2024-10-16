import Mathlib

namespace NUMINAMATH_CALUDE_figure_circumference_value_l192_19251

/-- The circumference of a figure formed by one large semicircular arc and 8 identical small semicircular arcs -/
def figure_circumference (d : ℝ) (π : ℝ) : ℝ :=
  π * d

/-- Theorem stating that the circumference of the described figure is 75.36 -/
theorem figure_circumference_value :
  let d : ℝ := 24
  let π : ℝ := 3.14
  figure_circumference d π = 75.36 := by sorry

end NUMINAMATH_CALUDE_figure_circumference_value_l192_19251


namespace NUMINAMATH_CALUDE_ham_slices_per_sandwich_l192_19235

theorem ham_slices_per_sandwich :
  ∀ (initial_slices : ℕ) (additional_slices : ℕ) (total_sandwiches : ℕ),
    initial_slices = 31 →
    additional_slices = 119 →
    total_sandwiches = 50 →
    (initial_slices + additional_slices) / total_sandwiches = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ham_slices_per_sandwich_l192_19235


namespace NUMINAMATH_CALUDE_max_a_for_inequality_l192_19263

theorem max_a_for_inequality : 
  (∃ (a_max : ℝ), 
    (∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) ∧
  (∀ (a_max : ℝ), 
    ((∀ (a : ℝ), (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a * Real.cos x ≥ 0) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), 1 - (2/3) * Real.cos (2*x) + a_max * Real.cos x ≥ 0)) → 
    a_max = 1/3) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_inequality_l192_19263


namespace NUMINAMATH_CALUDE_total_selling_price_l192_19290

/-- Calculate the total selling price of three items given their cost prices and profit/loss percentages -/
theorem total_selling_price
  (cost_A cost_B cost_C : ℝ)
  (loss_A gain_B loss_C : ℝ)
  (h_cost_A : cost_A = 1400)
  (h_cost_B : cost_B = 2500)
  (h_cost_C : cost_C = 3200)
  (h_loss_A : loss_A = 0.15)
  (h_gain_B : gain_B = 0.10)
  (h_loss_C : loss_C = 0.05) :
  cost_A * (1 - loss_A) + cost_B * (1 + gain_B) + cost_C * (1 - loss_C) = 6980 :=
by sorry

end NUMINAMATH_CALUDE_total_selling_price_l192_19290


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l192_19206

theorem sufficient_not_necessary_condition (x : ℝ) (h : x > 0) :
  (x + 1 / x ≥ 2) ∧ (∃ a : ℝ, a > 1 ∧ ∀ y : ℝ, y > 0 → y + a / y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l192_19206


namespace NUMINAMATH_CALUDE_min_value_of_function_l192_19249

theorem min_value_of_function (x : ℝ) (h : x ≥ 2) :
  x + 5 / (x + 1) ≥ 11 / 3 ∧
  (x + 5 / (x + 1) = 11 / 3 ↔ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l192_19249


namespace NUMINAMATH_CALUDE_inequality_solution_l192_19278

theorem inequality_solution (α x : ℝ) : α * x^2 - 2 ≥ 2 * x - α * x ↔
  (α = 0 ∧ x ≤ -1) ∨
  (α > 0 ∧ (x ≥ 2 / α ∨ x ≤ -1)) ∨
  (-2 < α ∧ α < 0 ∧ 2 / α ≤ x ∧ x ≤ -1) ∨
  (α = -2 ∧ x = -1) ∨
  (α < -2 ∧ -1 ≤ x ∧ x ≤ 2 / α) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l192_19278


namespace NUMINAMATH_CALUDE_secretary_work_ratio_l192_19285

/-- Represents the work hours of three secretaries on a project. -/
structure SecretaryWork where
  total : ℝ
  longest : ℝ
  second : ℝ
  third : ℝ

/-- Theorem stating the ratio of work hours for three secretaries. -/
theorem secretary_work_ratio (work : SecretaryWork) 
  (h_total : work.total = 120)
  (h_longest : work.longest = 75)
  (h_sum : work.second + work.third = work.total - work.longest) :
  ∃ (b c : ℝ), work.second = b ∧ work.third = c ∧ b + c = 45 := by
  sorry

#check secretary_work_ratio

end NUMINAMATH_CALUDE_secretary_work_ratio_l192_19285


namespace NUMINAMATH_CALUDE_interesting_factor_exists_l192_19209

/-- A natural number is interesting if it can be represented both as the sum of two consecutive integers and as the sum of three consecutive integers. -/
def is_interesting (n : ℕ) : Prop :=
  ∃ k m : ℤ, n = (k + (k + 1)) ∧ n = (m - 1 + m + (m + 1))

/-- The theorem states that if the product of five different natural numbers is interesting,
    then at least one of these natural numbers is interesting. -/
theorem interesting_factor_exists (a b c d e : ℕ) (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
    (h_interesting : is_interesting (a * b * c * d * e)) :
    is_interesting a ∨ is_interesting b ∨ is_interesting c ∨ is_interesting d ∨ is_interesting e :=
  sorry

end NUMINAMATH_CALUDE_interesting_factor_exists_l192_19209


namespace NUMINAMATH_CALUDE_liam_activity_balance_l192_19265

/-- Utility function for Liam's activities -/
def utility (reading : ℝ) (basketball : ℝ) : ℝ := reading * basketball

/-- Wednesday's utility calculation -/
def wednesday_utility (t : ℝ) : ℝ := utility (10 - t) t

/-- Thursday's utility calculation -/
def thursday_utility (t : ℝ) : ℝ := utility (t + 4) (3 - t)

/-- The theorem stating that t = 3 is the only valid solution -/
theorem liam_activity_balance :
  ∃! t : ℝ, t > 0 ∧ t < 10 ∧ wednesday_utility t = thursday_utility t ∧ t = 3 := by sorry

end NUMINAMATH_CALUDE_liam_activity_balance_l192_19265


namespace NUMINAMATH_CALUDE_deer_leap_distance_proof_l192_19277

/-- The distance the tiger needs to catch the deer -/
def catch_distance : ℝ := 800

/-- The number of tiger leaps behind the deer initially -/
def initial_leaps_behind : ℕ := 50

/-- The number of leaps the tiger takes per minute -/
def tiger_leaps_per_minute : ℕ := 5

/-- The number of leaps the deer takes per minute -/
def deer_leaps_per_minute : ℕ := 4

/-- The distance the tiger covers per leap in meters -/
def tiger_leap_distance : ℝ := 8

/-- The distance the deer covers per leap in meters -/
def deer_leap_distance : ℝ := 5

theorem deer_leap_distance_proof :
  deer_leap_distance = 5 :=
sorry

end NUMINAMATH_CALUDE_deer_leap_distance_proof_l192_19277


namespace NUMINAMATH_CALUDE_debt_payment_additional_amount_l192_19234

theorem debt_payment_additional_amount 
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (remaining_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (average_payment : ℚ)
  (h1 : total_installments = 52)
  (h2 : first_payment_count = 12)
  (h3 : remaining_payment_count = total_installments - first_payment_count)
  (h4 : first_payment_amount = 410)
  (h5 : average_payment = 460) :
  let additional_amount := (total_installments * average_payment - 
    first_payment_count * first_payment_amount) / remaining_payment_count - 
    first_payment_amount
  additional_amount = 65 := by sorry

end NUMINAMATH_CALUDE_debt_payment_additional_amount_l192_19234


namespace NUMINAMATH_CALUDE_percentage_problem_l192_19223

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 660 = 12 / 100 * 1500 - 15 → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l192_19223


namespace NUMINAMATH_CALUDE_soap_cost_theorem_l192_19245

/-- The cost of soap for a year, given the duration of one bar, cost per bar, and months in a year. -/
def soap_cost_per_year (months_per_bar : ℚ) (cost_per_bar : ℚ) (months_in_year : ℕ) : ℚ :=
  (months_in_year / months_per_bar) * cost_per_bar

/-- Theorem: The cost of soap for a year is $48.00 -/
theorem soap_cost_theorem : soap_cost_per_year 2 8 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_soap_cost_theorem_l192_19245


namespace NUMINAMATH_CALUDE_optimal_price_and_quantity_l192_19231

/-- Represents the pricing and sales model of a shopping mall --/
structure ShoppingMall where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_elasticity : ℝ
  target_profit : ℝ

/-- Calculates the monthly sales volume based on the new price --/
def sales_volume (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  mall.initial_sales - mall.price_elasticity * (new_price - mall.initial_price)

/-- Calculates the monthly profit based on the new price --/
def monthly_profit (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  (new_price - mall.cost_price) * sales_volume mall new_price

/-- Theorem stating that the new price and purchase quantity achieve the target profit --/
theorem optimal_price_and_quantity (mall : ShoppingMall) 
  (h_cost : mall.cost_price = 20)
  (h_initial_price : mall.initial_price = 30)
  (h_initial_sales : mall.initial_sales = 800)
  (h_elasticity : mall.price_elasticity = 20)
  (h_target : mall.target_profit = 12000) :
  ∃ (new_price : ℝ), 
    new_price = 40 ∧ 
    sales_volume mall new_price = 600 ∧ 
    monthly_profit mall new_price = mall.target_profit := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_quantity_l192_19231


namespace NUMINAMATH_CALUDE_mikeys_leaves_mikeys_leaves_specific_l192_19222

/-- The number of leaves Mikey has after receiving more leaves -/
def total_leaves (initial : ℝ) (new : ℝ) : ℝ :=
  initial + new

/-- Theorem stating that Mikey's total leaves is the sum of initial and new leaves -/
theorem mikeys_leaves (initial : ℝ) (new : ℝ) :
  total_leaves initial new = initial + new := by
  sorry

/-- Specific instance of Mikey's leaves problem -/
theorem mikeys_leaves_specific :
  total_leaves 356.0 112.0 = 468.0 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_mikeys_leaves_specific_l192_19222


namespace NUMINAMATH_CALUDE_john_slurpees_l192_19207

def slurpee_problem (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : ℕ :=
  (money_given - change) / slurpee_cost

theorem john_slurpees :
  slurpee_problem 20 2 8 = 6 :=
by sorry

end NUMINAMATH_CALUDE_john_slurpees_l192_19207


namespace NUMINAMATH_CALUDE_students_taking_statistics_l192_19273

theorem students_taking_statistics 
  (total : ℕ) 
  (history : ℕ) 
  (history_or_statistics : ℕ) 
  (history_not_statistics : ℕ) 
  (h_total : total = 90)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 29) :
  ∃ (statistics : ℕ), statistics = 30 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_statistics_l192_19273


namespace NUMINAMATH_CALUDE_age_difference_l192_19215

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 28 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l192_19215


namespace NUMINAMATH_CALUDE_square_root_49_squared_l192_19255

theorem square_root_49_squared : Real.sqrt 49 ^ 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_root_49_squared_l192_19255


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l192_19286

theorem log_equality_implies_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.log a / Real.log 9 = Real.log b / Real.log 12) ∧ 
  (Real.log a / Real.log 9 = Real.log (3*a + b) / Real.log 16) →
  b / a = (1 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l192_19286


namespace NUMINAMATH_CALUDE_square_area_is_eight_l192_19219

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 36

/-- The square inscribes the circle -/
def inscribed_square (s : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), ∀ (x y : ℝ),
    circle_equation x y →
    (x - x₀)^2 ≤ (s/2)^2 ∧ (y - y₀)^2 ≤ (s/2)^2

/-- One side of the square is parallel to the x-axis -/
def parallel_to_x_axis (s : ℝ) : Prop :=
  ∃ (y : ℝ), ∀ (x : ℝ),
    x ≥ 0 ∧ x ≤ s → circle_equation x y

theorem square_area_is_eight :
  ∃ (s : ℝ), inscribed_square s ∧ parallel_to_x_axis s ∧ s^2 = 8 := by sorry

end NUMINAMATH_CALUDE_square_area_is_eight_l192_19219


namespace NUMINAMATH_CALUDE_only_b_q_rotationally_symmetric_l192_19228

/-- Represents an English letter -/
inductive Letter
| B
| D
| P
| Q

/-- Defines rotational symmetry between two letters -/
def rotationallySymmetric (l1 l2 : Letter) : Prop :=
  match l1, l2 with
  | Letter.B, Letter.Q => True
  | Letter.Q, Letter.B => True
  | _, _ => False

/-- Theorem stating that only B and Q are rotationally symmetric -/
theorem only_b_q_rotationally_symmetric :
  ∀ (l1 l2 : Letter),
    rotationallySymmetric l1 l2 ↔ (l1 = Letter.B ∧ l2 = Letter.Q) ∨ (l1 = Letter.Q ∧ l2 = Letter.B) :=
by sorry

#check only_b_q_rotationally_symmetric

end NUMINAMATH_CALUDE_only_b_q_rotationally_symmetric_l192_19228


namespace NUMINAMATH_CALUDE_adjacent_probability_l192_19252

/-- The number of students in the arrangement -/
def total_students : ℕ := 9

/-- The number of rows in the seating arrangement -/
def rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def columns : ℕ := 3

/-- The number of ways to arrange n students -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of adjacent pairs in a row or column -/
def adjacent_pairs_per_line : ℕ := 2

/-- The number of ways to arrange two specific students in an adjacent pair -/
def ways_to_arrange_pair : ℕ := 2

/-- The probability of two specific students being adjacent in a 3x3 grid -/
theorem adjacent_probability :
  (((rows * adjacent_pairs_per_line + columns * adjacent_pairs_per_line) * ways_to_arrange_pair * 
    (arrangements (total_students - 2))) : ℚ) / 
  (arrangements total_students) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_l192_19252


namespace NUMINAMATH_CALUDE_transform_458_to_14_l192_19264

def double (n : ℕ) : ℕ := 2 * n

def eraseLast (n : ℕ) : ℕ := n / 10

inductive Operation
| Double
| EraseLast

def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.Double => double n
  | Operation.EraseLast => eraseLast n

def applyOperations (ops : List Operation) (start : ℕ) : ℕ :=
  ops.foldl (fun n op => applyOperation op n) start

theorem transform_458_to_14 :
  ∃ (ops : List Operation), applyOperations ops 458 = 14 :=
sorry

end NUMINAMATH_CALUDE_transform_458_to_14_l192_19264


namespace NUMINAMATH_CALUDE_roommate_payment_is_757_l192_19271

/-- Calculates the total payment for one roommate given the costs for rent, utilities, and groceries -/
def roommateTotalPayment (rent utilities groceries : ℕ) : ℚ :=
  (rent + utilities + groceries : ℚ) / 2

/-- Proves that one roommate's total payment is $757 given the specified costs -/
theorem roommate_payment_is_757 :
  roommateTotalPayment 1100 114 300 = 757 := by
  sorry

end NUMINAMATH_CALUDE_roommate_payment_is_757_l192_19271


namespace NUMINAMATH_CALUDE_triangle_folding_theorem_l192_19237

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Represents a folding method for a triangle -/
structure FoldingMethod where
  apply : Triangle → ℕ

/-- Represents the result of applying a folding method to a triangle -/
structure FoldedTriangle where
  original : Triangle
  method : FoldingMethod
  layers : ℕ

/-- A folded triangle has uniform thickness if all points have the same number of layers -/
def hasUniformThickness (ft : FoldedTriangle) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ ft.original.a :: ft.original.b :: ft.original.c :: [] → 
    ft.method.apply ft.original = ft.layers

theorem triangle_folding_theorem :
  ∀ t : Triangle, ∃ fm : FoldingMethod, 
    let ft := FoldedTriangle.mk t fm 2020
    hasUniformThickness ft ∧ ft.layers = 2020 := by
  sorry

end NUMINAMATH_CALUDE_triangle_folding_theorem_l192_19237


namespace NUMINAMATH_CALUDE_loan_shark_fees_l192_19288

/-- Calculates the total fees for a loan with a doubling weekly rate -/
def totalFees (loanAmount : ℝ) (initialRate : ℝ) (weeks : ℕ) : ℝ :=
  let weeklyFees := fun w => loanAmount * initialRate * (2 ^ w)
  (Finset.range weeks).sum weeklyFees

/-- Theorem stating that the total fees for a $100 loan at 5% initial rate for 2 weeks is $15 -/
theorem loan_shark_fees : totalFees 100 0.05 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_loan_shark_fees_l192_19288


namespace NUMINAMATH_CALUDE_total_students_l192_19233

theorem total_students (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : 
  boys_ratio = 8 → girls_ratio = 5 → num_girls = 160 → 
  (boys_ratio + girls_ratio) * (num_girls / girls_ratio) = 416 := by
sorry

end NUMINAMATH_CALUDE_total_students_l192_19233


namespace NUMINAMATH_CALUDE_hash_three_times_100_l192_19205

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + N

-- Theorem statement
theorem hash_three_times_100 : hash (hash (hash 100)) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_hash_three_times_100_l192_19205


namespace NUMINAMATH_CALUDE_equation_solution_l192_19203

theorem equation_solution (x y z k : ℝ) :
  (9 / (x - y) = k / (x + z)) ∧ (k / (x + z) = 16 / (z + y)) → k = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l192_19203


namespace NUMINAMATH_CALUDE_marikas_mothers_age_l192_19212

/-- Given:
  - Marika was 10 years old in 2006
  - On Marika's 10th birthday, her mother's age was five times Marika's age

  Prove that the year when Marika's mother's age will be twice Marika's age is 2036
-/
theorem marikas_mothers_age (marika_birth_year : ℕ) (mothers_birth_year : ℕ) : 
  marika_birth_year = 1996 →
  mothers_birth_year = 1956 →
  ∃ (future_year : ℕ), future_year = 2036 ∧ 
    (future_year - mothers_birth_year) = 2 * (future_year - marika_birth_year) := by
  sorry

end NUMINAMATH_CALUDE_marikas_mothers_age_l192_19212


namespace NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l192_19293

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_two_neg_one_l192_19293


namespace NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l192_19241

/-- Triangle ABC in 3D space -/
structure Triangle3D where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (t : Triangle3D) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of the given triangle is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let t : Triangle3D := {
    A := (1, 2, 3),
    B := (5, 3, 1),
    C := (3, 4, 5)
  }
  orthocenter t = (5/2, 3, 7/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_specific_triangle_l192_19241


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l192_19208

def A : Set Int := {-1, 0, 2}
def B : Set Int := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l192_19208


namespace NUMINAMATH_CALUDE_one_black_one_white_probability_l192_19210

/-- The probability of picking one black ball and one white ball from a jar -/
theorem one_black_one_white_probability (black_balls white_balls : ℕ) : 
  black_balls = 5 → white_balls = 2 → 
  (black_balls * white_balls : ℚ) / ((black_balls + white_balls) * (black_balls + white_balls - 1) / 2) = 10/21 := by
sorry

end NUMINAMATH_CALUDE_one_black_one_white_probability_l192_19210


namespace NUMINAMATH_CALUDE_total_books_calculation_l192_19220

/-- The number of book shelves -/
def num_shelves : ℕ := 350

/-- The number of books per shelf -/
def books_per_shelf : ℕ := 25

/-- The total number of books on all shelves -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem total_books_calculation : total_books = 8750 := by
  sorry

end NUMINAMATH_CALUDE_total_books_calculation_l192_19220


namespace NUMINAMATH_CALUDE_age_problem_l192_19211

/-- Given the ages of five people a, b, c, d, and e satisfying certain conditions,
    prove that b is 16 years old. -/
theorem age_problem (a b c d e : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  e = d - 3 →
  a + b + c + d + e = 52 →
  b = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l192_19211


namespace NUMINAMATH_CALUDE_matthew_friends_count_l192_19244

/-- Given that Matthew had 32 crackers initially and each person ate 8 crackers,
    prove that the number of friends Matthew gave crackers and cakes to is 4. -/
theorem matthew_friends_count (initial_crackers : ℕ) (crackers_per_person : ℕ) :
  initial_crackers = 32 →
  crackers_per_person = 8 →
  initial_crackers / crackers_per_person = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_matthew_friends_count_l192_19244


namespace NUMINAMATH_CALUDE_cubic_roots_product_l192_19281

theorem cubic_roots_product (a b c : ℝ) : 
  (x^3 - 15*x^2 + 25*x - 12 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (2 + a) * (2 + b) * (2 + c) = 130 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_product_l192_19281


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l192_19296

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ
  is_not_square : width ≠ height

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  qrsc : Rectangle
  ap : ℕ
  qr : ℕ
  bp : ℕ
  br : ℕ
  sc : ℕ

/-- The main theorem statement --/
theorem shaded_area_theorem (config : Configuration) :
  config.abcd.width * config.abcd.height = 35 →
  config.ap < config.qr →
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 24) ∨
  (config.abcd.width * config.abcd.height - 
   (config.bp * config.br + config.ap * config.sc) = 26) := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l192_19296


namespace NUMINAMATH_CALUDE_hyperbola_equation_l192_19224

/-- Given an ellipse and a hyperbola with the same foci, if one asymptote of the hyperbola
    is y = √2 x, then the equation of the hyperbola is 2y^2 - 4x^2 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), (4 * x^2 + y^2 = 1) ∧ 
   (∃ (m : ℝ), 0 < m ∧ m < 3/4 ∧ 
     y^2 / m - x^2 / ((3/4) - m) = 1) ∧
   (∃ (k : ℝ), y = k * x ∧ k^2 = 2)) →
  2 * y^2 - 4 * x^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l192_19224


namespace NUMINAMATH_CALUDE_man_rowing_speed_l192_19280

/-- Calculates the speed of a man rowing in still water given his downstream speed and current speed -/
theorem man_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 6 →
  distance = 100 →
  time = 14.998800095992323 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 18 := by
sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l192_19280


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l192_19218

theorem circle_intersection_radius_range :
  ∀ r : ℝ,
  r > 0 →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) →
  1 < r ∧ r < 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l192_19218


namespace NUMINAMATH_CALUDE_wedge_volume_l192_19240

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (angle : ℝ) : 
  d = 20 →
  angle = 60 →
  (1 / 6) * d^3 * π = 667 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l192_19240


namespace NUMINAMATH_CALUDE_shaded_area_in_square_l192_19292

/-- The area of a symmetric shaded region in a square -/
theorem shaded_area_in_square (square_side : ℝ) (point_A_x : ℝ) (point_B_x : ℝ) : 
  square_side = 10 →
  point_A_x = 7.5 →
  point_B_x = 7.5 →
  let shaded_area := 2 * (1/2 * (square_side/4) * (square_side/2))
  shaded_area = 28.125 := by sorry

end NUMINAMATH_CALUDE_shaded_area_in_square_l192_19292


namespace NUMINAMATH_CALUDE_science_club_teams_l192_19246

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of girls in the science club -/
def num_girls : ℕ := 4

/-- The number of boys in the science club -/
def num_boys : ℕ := 7

/-- The number of girls required for each team -/
def girls_per_team : ℕ := 3

/-- The number of boys required for each team -/
def boys_per_team : ℕ := 2

theorem science_club_teams :
  (choose num_girls girls_per_team) * (choose num_boys boys_per_team) = 84 :=
by sorry

end NUMINAMATH_CALUDE_science_club_teams_l192_19246


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l192_19298

theorem jelly_bean_probability (p_red p_orange p_blue p_yellow : ℝ) :
  p_red = 0.25 →
  p_orange = 0.4 →
  p_blue = 0.15 →
  p_red + p_orange + p_blue + p_yellow = 1 →
  p_yellow = 0.2 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l192_19298


namespace NUMINAMATH_CALUDE_distribute_four_balls_l192_19239

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1)

/-- The number of ways to distribute 4 distinguishable balls into 2 indistinguishable boxes is 8 -/
theorem distribute_four_balls : distribute_balls 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_balls_l192_19239


namespace NUMINAMATH_CALUDE_inequality_solution_set_empty_implies_k_range_l192_19250

theorem inequality_solution_set_empty_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * |x - 1| + 6 * k ≥ 0) → 
  k ≥ (1 + Real.sqrt 7) / 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_empty_implies_k_range_l192_19250


namespace NUMINAMATH_CALUDE_distance_squared_is_53_l192_19295

/-- A notched circle with specific measurements -/
structure NotchedCircle where
  radius : ℝ
  AB : ℝ
  BC : ℝ
  right_angle : Bool

/-- The square of the distance from point B to the center of the circle -/
def distance_squared (nc : NotchedCircle) : ℝ :=
  sorry

/-- Theorem stating the square of the distance from B to the center is 53 -/
theorem distance_squared_is_53 (nc : NotchedCircle) 
  (h1 : nc.radius = Real.sqrt 72)
  (h2 : nc.AB = 8)
  (h3 : nc.BC = 3)
  (h4 : nc.right_angle = true) :
  distance_squared nc = 53 :=
sorry

end NUMINAMATH_CALUDE_distance_squared_is_53_l192_19295


namespace NUMINAMATH_CALUDE_total_capacity_l192_19276

/-- The capacity of a circus tent with five seating sections -/
def circus_tent_capacity (regular_section_capacity : ℕ) (special_section_capacity : ℕ) : ℕ :=
  4 * regular_section_capacity + special_section_capacity

/-- Theorem: The circus tent can accommodate 1298 people -/
theorem total_capacity : circus_tent_capacity 246 314 = 1298 := by
  sorry

end NUMINAMATH_CALUDE_total_capacity_l192_19276


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l192_19282

theorem minimum_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + 1/c^2 ≥ Real.sqrt 3 + 2 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ c₀ ≠ 0 ∧
    a₀^2 + b₀^2 + c₀^2 + 1/a₀^2 + b₀/a₀ + 1/c₀^2 = Real.sqrt 3 + 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l192_19282


namespace NUMINAMATH_CALUDE_probability_select_from_both_sets_l192_19230

/-- The probability of selecting one card from each of two sets when drawing two cards at random without replacement, given a total of 13 cards where one set has 6 cards and the other has 7 cards. -/
theorem probability_select_from_both_sets : 
  ∀ (total : ℕ) (set1 : ℕ) (set2 : ℕ),
  total = 13 → set1 = 6 → set2 = 7 →
  (set1 / total * set2 / (total - 1) + set2 / total * set1 / (total - 1) : ℚ) = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_probability_select_from_both_sets_l192_19230


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l192_19221

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (distance_between_trees : ℝ) : ℝ :=
  (num_trees - 1) * distance_between_trees

/-- Theorem: The length of a yard with 26 trees planted at equal distances,
    with one tree at each end and 10 meters between consecutive trees, is 250 meters. -/
theorem yard_length_26_trees :
  yard_length 26 10 = 250 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l192_19221


namespace NUMINAMATH_CALUDE_joyce_apples_l192_19213

def initial_apples : ℕ := 75
def apples_given : ℕ := 52
def apples_left : ℕ := 23

theorem joyce_apples : initial_apples = apples_given + apples_left := by
  sorry

end NUMINAMATH_CALUDE_joyce_apples_l192_19213


namespace NUMINAMATH_CALUDE_find_M_l192_19261

theorem find_M (x y z M : ℝ) : 
  x + y + z = 120 ∧ 
  x - 10 = M ∧ 
  y + 10 = M ∧ 
  z / 10 = M 
  → M = 10 := by
sorry

end NUMINAMATH_CALUDE_find_M_l192_19261


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l192_19253

theorem ceiling_floor_difference : ⌈(15 : ℚ) / 7 * (-27 : ℚ) / 3⌉ - ⌊(15 : ℚ) / 7 * ⌈(-27 : ℚ) / 3⌉⌋ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l192_19253


namespace NUMINAMATH_CALUDE_no_x_squared_term_l192_19289

theorem no_x_squared_term (a : ℚ) : 
  (∀ x, (-9 * x^3 + (-6*a - 4) * x^2 - 3*x) = (-9 * x^3 - 3*x)) ↔ a = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l192_19289


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l192_19275

theorem inequality_system_solution_set (x : ℝ) : 
  (Set.Icc (-2 : ℝ) 0).inter (Set.Ioo (-3 : ℝ) 1) = 
  {x | |x^2 + 5*x| < 6 ∧ |x + 1| ≤ 1} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l192_19275


namespace NUMINAMATH_CALUDE_octagon_ratio_l192_19257

/-- Represents an octagon with specific properties -/
structure Octagon where
  total_area : ℝ
  unit_squares : ℕ
  pq_divides_equally : Prop
  below_pq_square : ℝ
  below_pq_triangle_base : ℝ
  xq_plus_qy : ℝ

/-- The theorem to be proved -/
theorem octagon_ratio (o : Octagon) 
  (h1 : o.total_area = 12)
  (h2 : o.unit_squares = 12)
  (h3 : o.pq_divides_equally)
  (h4 : o.below_pq_square = 1)
  (h5 : o.below_pq_triangle_base = 6)
  (h6 : o.xq_plus_qy = 6) :
  ∃ (xq qy : ℝ), xq / qy = 2 ∧ xq + qy = o.xq_plus_qy :=
sorry

end NUMINAMATH_CALUDE_octagon_ratio_l192_19257


namespace NUMINAMATH_CALUDE_march_1900_rainfall_average_l192_19274

/-- The average rainfall per minute given total rainfall and number of days -/
def average_rainfall_per_minute (total_rainfall : ℚ) (days : ℕ) : ℚ :=
  total_rainfall / (days * 24 * 60)

/-- Theorem stating that 620 inches of rainfall over 15 days results in an average of 31/1080 inches per minute -/
theorem march_1900_rainfall_average : 
  average_rainfall_per_minute 620 15 = 31 / 1080 := by
  sorry

end NUMINAMATH_CALUDE_march_1900_rainfall_average_l192_19274


namespace NUMINAMATH_CALUDE_inequality_proof_l192_19201

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c < 1 / a + 1 / b + 1 / c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l192_19201


namespace NUMINAMATH_CALUDE_painting_price_decrease_l192_19238

theorem painting_price_decrease (original_price : ℝ) (h_positive : original_price > 0) :
  let first_year_price := original_price * 1.25
  let final_price := original_price * 1.0625
  let second_year_decrease := (first_year_price - final_price) / first_year_price
  second_year_decrease = 0.15 := by sorry

end NUMINAMATH_CALUDE_painting_price_decrease_l192_19238


namespace NUMINAMATH_CALUDE_omega_on_real_axis_l192_19283

theorem omega_on_real_axis (z : ℂ) (h1 : z.re ≠ 0) (h2 : Complex.abs z = 1) :
  let ω := z + z⁻¹
  ω.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_omega_on_real_axis_l192_19283


namespace NUMINAMATH_CALUDE_regular_pentagon_diagonal_l192_19202

/-- For a regular pentagon with side length a, its diagonal d satisfies d = (√5 + 1)/2 * a -/
theorem regular_pentagon_diagonal (a : ℝ) (h : a > 0) :
  ∃ d : ℝ, d > 0 ∧ d = (Real.sqrt 5 + 1) / 2 * a := by
  sorry

end NUMINAMATH_CALUDE_regular_pentagon_diagonal_l192_19202


namespace NUMINAMATH_CALUDE_cylinder_height_equals_sphere_radius_l192_19269

theorem cylinder_height_equals_sphere_radius 
  (r_sphere : ℝ) 
  (d_cylinder : ℝ) 
  (h_cylinder : ℝ) :
  r_sphere = 3 →
  d_cylinder = 6 →
  2 * π * (d_cylinder / 2) * h_cylinder = 4 * π * r_sphere^2 →
  h_cylinder = 6 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_equals_sphere_radius_l192_19269


namespace NUMINAMATH_CALUDE_dwarf_heights_l192_19232

/-- The heights of Mr. Ticháček's dwarfs -/
theorem dwarf_heights :
  ∀ (F J M : ℕ),
  (J + F = M) →
  (M + F = J + 34) →
  (M + J = F + 72) →
  (F = 17 ∧ J = 36 ∧ M = 53) :=
by
  sorry

end NUMINAMATH_CALUDE_dwarf_heights_l192_19232


namespace NUMINAMATH_CALUDE_teacher_li_flags_l192_19242

theorem teacher_li_flags : ∃ (x : ℕ), x > 0 ∧ 4 * x + 20 = 44 ∧ 4 * x + 20 > 8 * (x - 1) ∧ 4 * x + 20 < 8 * x :=
by sorry

end NUMINAMATH_CALUDE_teacher_li_flags_l192_19242


namespace NUMINAMATH_CALUDE_prime_sum_inequality_l192_19272

theorem prime_sum_inequality (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r ≥ 1 →
  ({p, q, r} : Set ℕ) = {2, 3, 5} :=
sorry

end NUMINAMATH_CALUDE_prime_sum_inequality_l192_19272


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l192_19294

-- Define the function y
def y (a b x : ℝ) : ℝ := a * x^2 + x - b

-- Part 1
theorem part_one (a : ℝ) :
  (∃! x, y a 1 x = 0) → (a = -1/4 ∨ a = 0) :=
sorry

-- Part 2
theorem part_two (a b x : ℝ) :
  y a b x < (a-1) * x^2 + (b+2) * x - 2*b ↔
    (b < 1 ∧ b < x ∧ x < 1) ∨
    (b > 1 ∧ 1 < x ∧ x < b) :=
sorry

-- Part 3
theorem part_three (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, y a b x > 0 ∧ -2-t < x ∧ x < -2+t) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, y a' b' x > 0 ∧ -2-t < x ∧ x < -2+t) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l192_19294


namespace NUMINAMATH_CALUDE_sum_of_first_five_primes_mod_sixth_prime_l192_19214

def first_five_primes : List Nat := [2, 3, 5, 7, 11]
def sixth_prime : Nat := 13

theorem sum_of_first_five_primes_mod_sixth_prime :
  (first_five_primes.sum % sixth_prime) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_five_primes_mod_sixth_prime_l192_19214


namespace NUMINAMATH_CALUDE_games_purchased_l192_19217

theorem games_purchased (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 104 → spent_amount = 41 → game_cost = 9 →
  (initial_amount - spent_amount) / game_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_games_purchased_l192_19217


namespace NUMINAMATH_CALUDE_fraction_equality_l192_19297

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a^2 + b - a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 4 3) / (hash_op 4 3) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l192_19297


namespace NUMINAMATH_CALUDE_opposite_absolute_values_l192_19287

theorem opposite_absolute_values (a b : ℝ) :
  (|a - 1| + |b - 2| = 0) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_absolute_values_l192_19287


namespace NUMINAMATH_CALUDE_only_drug_effectiveness_suitable_l192_19279

/-- Represents the suitability of an option for a sampling survey. -/
inductive Suitability
  | Suitable
  | NotSuitable

/-- Represents the different survey options. -/
inductive SurveyOption
  | DrugEffectiveness
  | ClassVision
  | EmployeeExamination
  | SatelliteInspection

/-- Determines the suitability of a survey option for sampling. -/
def suitabilityForSampling (option : SurveyOption) : Suitability :=
  match option with
  | SurveyOption.DrugEffectiveness => Suitability.Suitable
  | _ => Suitability.NotSuitable

/-- Theorem stating that only the drug effectiveness option is suitable for sampling. -/
theorem only_drug_effectiveness_suitable :
  ∀ (option : SurveyOption),
    suitabilityForSampling option = Suitability.Suitable ↔
    option = SurveyOption.DrugEffectiveness :=
by
  sorry

#check only_drug_effectiveness_suitable

end NUMINAMATH_CALUDE_only_drug_effectiveness_suitable_l192_19279


namespace NUMINAMATH_CALUDE_clock_problem_l192_19226

/-- Represents a clock with special striking properties -/
structure StrikingClock where
  /-- Time for each stroke and interval between strokes (in seconds) -/
  stroke_time : ℝ
  /-- Calculates the total time lapse for striking a given hour -/
  time_lapse : ℕ → ℝ
  /-- The time lapse is (2n - 1) * stroke_time, where n is the hour -/
  time_lapse_eq : ∀ (hour : ℕ), time_lapse hour = (2 * hour - 1) * stroke_time

/-- The theorem representing our clock problem -/
theorem clock_problem (clock : StrikingClock) 
    (h1 : clock.time_lapse 7 = 26) 
    (h2 : ∃ (hour : ℕ), clock.time_lapse hour = 22) : 
  ∃ (hour : ℕ), hour = 6 ∧ clock.time_lapse hour = 22 :=
sorry

end NUMINAMATH_CALUDE_clock_problem_l192_19226


namespace NUMINAMATH_CALUDE_solve_abc_l192_19258

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_abc (a b c : ℝ) : 
  A a ≠ B b c ∧ 
  A a ∪ B b c = {-3, 4} ∧
  A a ∩ B b c = {-3} →
  a = -1 ∧ b = 6 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_solve_abc_l192_19258


namespace NUMINAMATH_CALUDE_books_for_vacation_l192_19262

/-- The number of books that can be read given reading speed, book parameters, and reading time -/
def books_to_read (reading_speed : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (reading_time : ℕ) : ℕ :=
  (reading_speed * reading_time * 60) / (words_per_page * pages_per_book)

/-- Theorem stating that given the specific conditions, the number of books to read is 6 -/
theorem books_for_vacation : books_to_read 40 100 80 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_books_for_vacation_l192_19262


namespace NUMINAMATH_CALUDE_minimum_beans_purchase_l192_19243

theorem minimum_beans_purchase (r b : ℝ) : 
  (r ≥ 2 * b + 8 ∧ r ≤ 3 * b) → b ≥ 8 := by sorry

end NUMINAMATH_CALUDE_minimum_beans_purchase_l192_19243


namespace NUMINAMATH_CALUDE_solution_in_fourth_quadrant_l192_19248

-- Define the equation system
def equation_system (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ y = -x + 1

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem solution_in_fourth_quadrant :
  ∃ x y : ℝ, equation_system x y ∧ fourth_quadrant x y :=
sorry

end NUMINAMATH_CALUDE_solution_in_fourth_quadrant_l192_19248


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l192_19227

/-- Given a line equation x/a + y/b = 1 where a > 0 and b > 0, 
    and the line passes through the point (2, 3),
    prove that the minimum value of 2a + b is 7 + 4√3 -/
theorem min_value_of_2a_plus_b (a b : ℝ) : 
  a > 0 → b > 0 → 2 / a + 3 / b = 1 → 
  ∀ x y, x / a + y / b = 1 → 
  (2 * a + b) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_b_l192_19227


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l192_19204

/-- Given a cylinder with volume 150π cm³, prove that:
    1. A cone with the same base radius and height as the cylinder has a volume of 50π cm³
    2. A sphere with the same radius as the cylinder has a volume of 200π cm³ -/
theorem cylinder_cone_sphere_volumes (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  π * r^2 * h = 150 * π →
  (1/3 : ℝ) * π * r^2 * h = 50 * π ∧ 
  (4/3 : ℝ) * π * r^3 = 200 * π := by
  sorry

#check cylinder_cone_sphere_volumes

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volumes_l192_19204


namespace NUMINAMATH_CALUDE_inequality_proof_l192_19268

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_given : (3 : ℝ) / (a * b * c) ≥ a + b + c) :
  1 / a + 1 / b + 1 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l192_19268


namespace NUMINAMATH_CALUDE_square_expression_equals_289_l192_19266

theorem square_expression_equals_289 (x : ℝ) (h : x = 5) : 
  (2 * x + 5 + 2)^2 = 289 := by sorry

end NUMINAMATH_CALUDE_square_expression_equals_289_l192_19266


namespace NUMINAMATH_CALUDE_equal_chord_lengths_l192_19284

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

/-- The first line -/
def line1 (k x y : ℝ) : Prop := k * x + y - 2 = 0

/-- The second line -/
def line2 (k x y : ℝ) : Prop := y = k * x + 2

/-- Length of the chord intercepted by a line on the ellipse -/
noncomputable def chord_length (line : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem equal_chord_lengths (k : ℝ) :
  chord_length (line1 k) = chord_length (line2 k) :=
sorry

end NUMINAMATH_CALUDE_equal_chord_lengths_l192_19284


namespace NUMINAMATH_CALUDE_outfit_combinations_l192_19216

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (ties : ℕ) (belts : ℕ) 
  (h_shirts : shirts = 8)
  (h_pants : pants = 5)
  (h_ties : ties = 4)
  (h_belts : belts = 2) :
  shirts * pants * (ties + 1) * (belts + 1) = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l192_19216


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l192_19259

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) 
  (h1 : total_students = 485)
  (h2 : boys = 208)
  (h3 : boys < total_students - boys) :
  total_students - boys - boys = 69 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l192_19259


namespace NUMINAMATH_CALUDE_hawks_score_l192_19229

/-- 
Given:
- The total points scored by both teams is 50
- The Eagles won by a margin of 18 points

Prove that the Hawks scored 16 points
-/
theorem hawks_score (total_points eagles_points hawks_points : ℕ) : 
  total_points = 50 →
  eagles_points = hawks_points + 18 →
  eagles_points + hawks_points = total_points →
  hawks_points = 16 := by
sorry

end NUMINAMATH_CALUDE_hawks_score_l192_19229


namespace NUMINAMATH_CALUDE_graham_crackers_leftover_l192_19299

/-- Represents the number of boxes of Graham crackers Lionel bought -/
def graham_crackers : ℕ := 14

/-- Represents the number of packets of Oreos Lionel bought -/
def oreos : ℕ := 15

/-- Represents the number of boxes of Graham crackers needed for one cheesecake -/
def graham_crackers_per_cake : ℕ := 2

/-- Represents the number of packets of Oreos needed for one cheesecake -/
def oreos_per_cake : ℕ := 3

/-- Calculates the number of boxes of Graham crackers left over after making
    the maximum number of Oreo cheesecakes -/
def graham_crackers_left : ℕ :=
  graham_crackers - graham_crackers_per_cake * (min (graham_crackers / graham_crackers_per_cake) (oreos / oreos_per_cake))

theorem graham_crackers_leftover :
  graham_crackers_left = 4 := by sorry

end NUMINAMATH_CALUDE_graham_crackers_leftover_l192_19299


namespace NUMINAMATH_CALUDE_smallest_number_l192_19247

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = -3 ∧ b = -2 ∧ c = 0 ∧ d = 1/3 → 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by sorry

end NUMINAMATH_CALUDE_smallest_number_l192_19247


namespace NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l192_19256

/-- Represents a tennis tournament with the given conditions -/
structure TennisTournament where
  num_teams : Nat
  players_per_team : Nat
  abstaining_player : Nat
  abstained_team : Nat

/-- Calculates the number of handshakes in the tournament -/
def count_handshakes (t : TennisTournament) : Nat :=
  sorry

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  ∀ (t : TennisTournament),
    t.num_teams = 4 ∧
    t.players_per_team = 2 ∧
    t.abstaining_player ≥ 1 ∧
    t.abstaining_player ≤ 8 ∧
    t.abstained_team ≥ 1 ∧
    t.abstained_team ≤ 4 ∧
    t.abstained_team ≠ ((t.abstaining_player - 1) / 2 + 1) →
    count_handshakes t = 22 :=
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l192_19256


namespace NUMINAMATH_CALUDE_factorize_2x_squared_minus_18_l192_19225

theorem factorize_2x_squared_minus_18 (x : ℝ) :
  2 * x^2 - 18 = 2 * (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_factorize_2x_squared_minus_18_l192_19225


namespace NUMINAMATH_CALUDE_area_equality_l192_19291

-- Define the points
variable (A B C D M N K L : Plane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Plane) : Prop := sorry

-- Define that M is on AB and N is on CD
def on_segment (P Q R : Plane) : Prop := sorry

-- Define the ratio condition
def ratio_condition (A M B C N D : Plane) : Prop := sorry

-- Define the intersection points
def intersect_at (P Q R S T : Plane) : Prop := sorry

-- Define the area of a polygon
def area (points : List Plane) : ℝ := sorry

-- Theorem statement
theorem area_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment A M B)
  (h3 : on_segment C N D)
  (h4 : ratio_condition A M B C N D)
  (h5 : intersect_at A N D M K)
  (h6 : intersect_at B N C M L) :
  area [K, M, L, N] = area [A, D, K] + area [B, C, L] := by sorry

end NUMINAMATH_CALUDE_area_equality_l192_19291


namespace NUMINAMATH_CALUDE_valid_k_characterization_l192_19260

/-- A function f: ℤ → ℤ is nonlinear if there exist x, y ∈ ℤ such that 
    f(x + y) ≠ f(x) + f(y) or f(ax) ≠ af(x) for some a ∈ ℤ -/
def Nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ x y : ℤ, f (x + y) ≠ f x + f y ∨ ∃ a : ℤ, f (a * x) ≠ a * f x

/-- The set of non-negative integer values of k for which there exists a nonlinear function
    f: ℤ → ℤ satisfying the given equation for all integers a, b, c with a + b + c = 0 -/
def ValidK : Set ℕ :=
  {k : ℕ | ∃ f : ℤ → ℤ, Nonlinear f ∧
    ∀ a b c : ℤ, a + b + c = 0 →
      f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k}

theorem valid_k_characterization : ValidK = {0, 1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_valid_k_characterization_l192_19260


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l192_19267

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    (n - 2) * 180 = n * interior_angle →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l192_19267


namespace NUMINAMATH_CALUDE_xixi_cards_problem_l192_19254

theorem xixi_cards_problem (x y : ℕ) :
  (x + 3 = 3 * (y - 3) ∧ y + 4 = 4 * (x - 4)) ∨
  (x + 3 = 3 * (y - 3) ∧ x + 5 = 5 * (y - 5)) ∨
  (y + 4 = 4 * (x - 4) ∧ x + 5 = 5 * (y - 5)) →
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_xixi_cards_problem_l192_19254


namespace NUMINAMATH_CALUDE_integer_decimal_parts_of_2_plus_sqrt_6_l192_19236

theorem integer_decimal_parts_of_2_plus_sqrt_6 :
  let x := Int.floor (2 + Real.sqrt 6)
  let y := (2 + Real.sqrt 6) - x
  (x = 4 ∧ y = Real.sqrt 6 - 2 ∧ Real.sqrt (x - 1) = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_integer_decimal_parts_of_2_plus_sqrt_6_l192_19236


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l192_19200

-- Define the function f
def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

-- State the theorem
theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l192_19200


namespace NUMINAMATH_CALUDE_cross_product_example_l192_19270

/-- The cross product of two 3D vectors -/
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2.1 * v.2.2 - u.2.2 * v.2.1,
   u.2.2 * v.1 - u.1 * v.2.2,
   u.1 * v.2.1 - u.2.1 * v.1)

theorem cross_product_example : 
  cross_product (3, -4, 7) (2, 5, -3) = (-23, 23, 23) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_example_l192_19270
