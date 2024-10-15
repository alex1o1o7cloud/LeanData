import Mathlib

namespace NUMINAMATH_CALUDE_tim_has_33_books_l3764_376458

/-- The number of books Tim has, given the initial conditions -/
def tims_books (benny_initial : ℕ) (sandy_received : ℕ) (total : ℕ) : ℕ :=
  total - (benny_initial - sandy_received)

/-- Theorem stating that Tim has 33 books under the given conditions -/
theorem tim_has_33_books :
  tims_books 24 10 47 = 33 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_33_books_l3764_376458


namespace NUMINAMATH_CALUDE_price_increase_consumption_reduction_l3764_376481

/-- Theorem: If the price of a commodity increases by 25%, a person must reduce their consumption by 20% to maintain the same expenditure. -/
theorem price_increase_consumption_reduction (P C : ℝ) (h : P > 0) (h' : C > 0) :
  let new_price := P * 1.25
  let new_consumption := C * 0.8
  new_price * new_consumption = P * C := by
  sorry

end NUMINAMATH_CALUDE_price_increase_consumption_reduction_l3764_376481


namespace NUMINAMATH_CALUDE_pasture_rent_is_870_l3764_376488

/-- Represents the rental information for a person --/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total rent for a pasture given rental information and a known payment --/
def calculate_total_rent (a b c : RentalInfo) (b_payment : ℕ) : ℕ :=
  let total_horse_months := a.horses * a.months + b.horses * b.months + c.horses * c.months
  let cost_per_horse_month := b_payment / (b.horses * b.months)
  cost_per_horse_month * total_horse_months

/-- Theorem stating that the total rent for the pasture is 870 --/
theorem pasture_rent_is_870 (a b c : RentalInfo) (h1 : a.horses = 12) (h2 : a.months = 8)
    (h3 : b.horses = 16) (h4 : b.months = 9) (h5 : c.horses = 18) (h6 : c.months = 6)
    (h7 : calculate_total_rent a b c 360 = 870) : 
  calculate_total_rent a b c 360 = 870 := by
  sorry

end NUMINAMATH_CALUDE_pasture_rent_is_870_l3764_376488


namespace NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3764_376443

-- Define P and Q as propositions depending on a real number x
def P (x : ℝ) : Prop := (2*x - 3)^2 < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- State the theorem
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
sorry

end NUMINAMATH_CALUDE_P_sufficient_not_necessary_for_Q_l3764_376443


namespace NUMINAMATH_CALUDE_fifth_element_row_20_l3764_376417

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The element at position k in row n of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ :=
  binomial n (k - 1)

/-- The fifth element in row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_element 20 5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_fifth_element_row_20_l3764_376417


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l3764_376402

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the original point P
def P : Point := (3, -5)

-- Define the symmetry operation with respect to x-axis
def symmetry_x_axis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  symmetry_x_axis P = (3, 5) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l3764_376402


namespace NUMINAMATH_CALUDE_alphametic_puzzle_impossibility_l3764_376454

theorem alphametic_puzzle_impossibility : ¬ ∃ (f : Char → Nat),
  (∀ x y : Char, x ≠ y → f x ≠ f y) ∧
  (∀ x : Char, x ∈ ['K', 'O', 'T', 'U', 'C', 'E', 'N', 'W', 'Y'] → f x ∈ Set.range (Fin.val : Fin 9 → Nat)) ∧
  (f 'K' * f 'O' * f 'T' = f 'U' * f 'C' * f 'E' * f 'N' * f 'W' * f 'Y') :=
by sorry


end NUMINAMATH_CALUDE_alphametic_puzzle_impossibility_l3764_376454


namespace NUMINAMATH_CALUDE_area_implies_m_value_existence_implies_a_range_l3764_376408

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Theorem for part (1)
theorem area_implies_m_value (m : ℝ) (h1 : m > 3) 
  (h2 : (1/2) * ((m - 1)/2 - (-(m + 1)/2) + 3) * (m - 3) = 7/2) : 
  m = 4 := by sorry

-- Theorem for part (2)
theorem existence_implies_a_range (a : ℝ) 
  (h : ∃ x ∈ Set.Icc 0 2, f x ≥ |a - 3|) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_area_implies_m_value_existence_implies_a_range_l3764_376408


namespace NUMINAMATH_CALUDE_negation_of_exists_negation_of_quadratic_equation_l3764_376407

theorem negation_of_exists (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - 3*x + 2 = 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_negation_of_quadratic_equation_l3764_376407


namespace NUMINAMATH_CALUDE_tan_difference_equals_one_eighth_l3764_376478

theorem tan_difference_equals_one_eighth 
  (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/3) 
  (h2 : Real.tan (π/6 - β) = 1/2) : 
  Real.tan (α - π/6) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_tan_difference_equals_one_eighth_l3764_376478


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l3764_376487

/-- Given a geometric sequence {a_n} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a1 : a 1 = -2) 
  (h_a5 : a 5 = -4) : 
  a 3 = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l3764_376487


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l3764_376418

theorem product_and_reciprocal_relation_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b = 16 ∧ 1 / a = 3 / b → a + b = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_sum_l3764_376418


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3764_376424

/-- Given a quarter circle sector with radius 5, the radius of the inscribed circle
    tangent to both radii and the arc is 5√2 - 5. -/
theorem inscribed_circle_radius (r : ℝ) : 
  r > 0 ∧ 
  r * (1 + Real.sqrt 2) = 5 → 
  r = 5 * Real.sqrt 2 - 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l3764_376424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3764_376404

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, n < m → a n < a m) →  -- increasing sequence
  (a 4)^2 - 10 * (a 4) + 24 = 0 →   -- a_4 is a root
  (a 6)^2 - 10 * (a 6) + 24 = 0 →   -- a_6 is a root
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 20 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3764_376404


namespace NUMINAMATH_CALUDE_power_product_sum_equality_l3764_376437

theorem power_product_sum_equality : (3^5 * 6^3) + 3^3 = 52515 := by
  sorry

end NUMINAMATH_CALUDE_power_product_sum_equality_l3764_376437


namespace NUMINAMATH_CALUDE_car_selling_problem_l3764_376409

/-- Represents the selling price and profit information for two types of cars -/
structure CarInfo where
  price_a : ℕ  -- Selling price of type A car in yuan
  price_b : ℕ  -- Selling price of type B car in yuan
  profit_a : ℕ  -- Profit from selling one type A car in yuan
  profit_b : ℕ  -- Profit from selling one type B car in yuan

/-- Represents a purchasing plan -/
structure PurchasePlan where
  count_a : ℕ  -- Number of type A cars purchased
  count_b : ℕ  -- Number of type B cars purchased

/-- Theorem stating the properties of the car selling problem -/
theorem car_selling_problem (info : CarInfo) 
  (h1 : 2 * info.price_a + 3 * info.price_b = 800000)
  (h2 : 3 * info.price_a + 2 * info.price_b = 950000)
  (h3 : info.profit_a = 8000)
  (h4 : info.profit_b = 5000) :
  info.price_a = 250000 ∧ 
  info.price_b = 100000 ∧ 
  (∃ (plans : Finset PurchasePlan), 
    (∀ plan ∈ plans, plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000) ∧
    plans.card = 3 ∧
    (∀ plan ∈ plans, ∀ other_plan : PurchasePlan, 
      other_plan.count_a * info.price_a + other_plan.count_b * info.price_b = 2000000 →
      other_plan ∈ plans)) ∧
  (∃ (max_profit : ℕ), 
    max_profit = 91000 ∧
    ∀ plan : PurchasePlan, 
      plan.count_a * info.price_a + plan.count_b * info.price_b = 2000000 →
      plan.count_a * info.profit_a + plan.count_b * info.profit_b ≤ max_profit) := by
  sorry


end NUMINAMATH_CALUDE_car_selling_problem_l3764_376409


namespace NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3764_376432

theorem set_inclusion_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {1, a^2 - a}
  A ⊇ B → a = -1 ∨ a = 0 := by
sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_a_value_l3764_376432


namespace NUMINAMATH_CALUDE_ages_solution_l3764_376429

/-- Represents the ages of Henry, Jill, and Alex -/
structure Ages where
  henry : ℕ
  jill : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.henry + ages.jill + ages.alex = 90 ∧
  ages.henry - 5 = 2 * (ages.jill - 5) ∧
  ages.henry + ages.jill - 10 = ages.alex

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.henry = 32 ∧ ages.jill = 18 ∧ ages.alex = 40 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l3764_376429


namespace NUMINAMATH_CALUDE_equation_roots_iff_q_condition_l3764_376410

/-- The equation x^4 + qx^3 + 2x^2 + qx + 4 = 0 has at least two distinct negative real roots
    if and only if q ≤ 3/√2 -/
theorem equation_roots_iff_q_condition (q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
    x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 4 = 0 ∧
    x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 4 = 0) ↔
  q ≤ 3 / Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_iff_q_condition_l3764_376410


namespace NUMINAMATH_CALUDE_science_fiction_books_l3764_376425

theorem science_fiction_books (pages_per_book : ℕ) (total_pages : ℕ) (h1 : pages_per_book = 478) (h2 : total_pages = 3824) :
  total_pages / pages_per_book = 8 := by
  sorry

end NUMINAMATH_CALUDE_science_fiction_books_l3764_376425


namespace NUMINAMATH_CALUDE_steves_take_home_pay_l3764_376414

/-- Calculates the take-home pay given salary and deduction rates -/
def takeHomePay (salary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  salary - (salary * taxRate) - (salary * healthcareRate) - unionDues

/-- Theorem: Steve's take-home pay is $27,200 -/
theorem steves_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

#eval takeHomePay 40000 0.20 0.10 800

end NUMINAMATH_CALUDE_steves_take_home_pay_l3764_376414


namespace NUMINAMATH_CALUDE_A_3_1_equals_13_l3764_376438

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_1_equals_13 : A 3 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_A_3_1_equals_13_l3764_376438


namespace NUMINAMATH_CALUDE_monotonicity_condition_solution_set_l3764_376434

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 2*a

-- Theorem for monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (f a)) ↔ (a ≥ -1/2 ∨ a ≤ -5/2) := by sorry

-- Theorem for solution set of f(x) < 0
theorem solution_set (a : ℝ) :
  {x : ℝ | f a x < 0} = 
    if a = -1/2 then ∅ 
    else if a < -1/2 then Set.Ioo 1 (-2*a)
    else Set.Ioo (-2*a) 1 := by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_solution_set_l3764_376434


namespace NUMINAMATH_CALUDE_optimal_price_l3764_376466

def sales_volume (x : ℝ) : ℝ := -10 * x + 800

theorem optimal_price (production_cost : ℝ) (max_price : ℝ) (target_profit : ℝ) :
  production_cost = 20 →
  max_price = 45 →
  target_profit = 8000 →
  sales_volume 30 = 500 →
  sales_volume 40 = 400 →
  ∃ (price : ℝ), price ≤ max_price ∧
                 (price - production_cost) * sales_volume price = target_profit ∧
                 price = 40 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_l3764_376466


namespace NUMINAMATH_CALUDE_expected_string_length_l3764_376491

/-- Represents the states of Clayton's progress -/
inductive State
  | S0  -- No letters written
  | S1  -- M written
  | S2  -- M and A written
  | S3  -- M, A, and T written
  | S4  -- M, A, T, and H written (final state)

/-- The hexagon with vertices M, M, A, T, H, S -/
def Hexagon : Type := Unit

/-- Clayton's starting position (M adjacent to M and A) -/
def start_position : Hexagon := Unit.unit

/-- Probability of moving to an adjacent vertex -/
def move_probability : ℚ := 1/2

/-- Expected number of steps to reach the final state from a given state -/
noncomputable def expected_steps : State → ℚ
  | State.S0 => 5
  | State.S1 => 4
  | State.S2 => 3
  | State.S3 => 2
  | State.S4 => 0

/-- The main theorem: Expected length of Clayton's string is 6 -/
theorem expected_string_length :
  expected_steps State.S0 + 1 = 6 := by sorry

end NUMINAMATH_CALUDE_expected_string_length_l3764_376491


namespace NUMINAMATH_CALUDE_sibling_ages_l3764_376449

theorem sibling_ages (sister_age brother_age : ℕ) : 
  (brother_age - 2 = 2 * (sister_age - 2)) →
  (brother_age - 8 = 5 * (sister_age - 8)) →
  (sister_age = 10 ∧ brother_age = 18) :=
by sorry

end NUMINAMATH_CALUDE_sibling_ages_l3764_376449


namespace NUMINAMATH_CALUDE_NaNO3_formed_l3764_376470

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

def balancedReaction : Reaction :=
  { reactants := [
      { name := "NH4NO3", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NaNO3", moles := 1 },
      { name := "NH3", moles := 1 },
      { name := "H2O", moles := 1 }
    ]
  }

def initialNH4NO3 : Compound :=
  { name := "NH4NO3", moles := 3 }

def initialNaOH : Compound :=
  { name := "NaOH", moles := 3 }

/-- Calculates the moles of a product formed in a reaction -/
def molesFormed (reaction : Reaction) (initialReactants : List Compound) (product : String) : ℝ :=
  sorry

theorem NaNO3_formed :
  molesFormed balancedReaction [initialNH4NO3, initialNaOH] "NaNO3" = 3 := by
  sorry

end NUMINAMATH_CALUDE_NaNO3_formed_l3764_376470


namespace NUMINAMATH_CALUDE_percentage_of_number_l3764_376447

theorem percentage_of_number (n : ℝ) : n * 0.001 = 0.24 → n = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_l3764_376447


namespace NUMINAMATH_CALUDE_range_of_x_for_positive_f_l3764_376482

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a-4)*x + 4-2*a

-- State the theorem
theorem range_of_x_for_positive_f :
  ∀ a ∈ Set.Icc (-1 : ℝ) 1,
    (∀ x, f a x > 0) ↔ (∀ x, x < 1 ∨ x > 3) := by sorry

end NUMINAMATH_CALUDE_range_of_x_for_positive_f_l3764_376482


namespace NUMINAMATH_CALUDE_odd_product_over_sum_equals_fifteen_fourths_l3764_376472

theorem odd_product_over_sum_equals_fifteen_fourths : 
  (1 * 3 * 5 * 7) / (1 + 2 + 3 + 4 + 5 + 6 + 7) = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_product_over_sum_equals_fifteen_fourths_l3764_376472


namespace NUMINAMATH_CALUDE_monic_polynomial_divisibility_l3764_376400

open Polynomial

theorem monic_polynomial_divisibility (n k : ℕ) (h_pos_n : n > 0) (h_pos_k : k > 0) :
  ∀ (f : Polynomial ℤ),
    Monic f →
    (Polynomial.degree f = n) →
    (∀ (a : ℤ), f.eval a ≠ 0 → (f.eval a ∣ f.eval (2 * a ^ k))) →
    f = X ^ n :=
by sorry

end NUMINAMATH_CALUDE_monic_polynomial_divisibility_l3764_376400


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3764_376422

theorem solution_set_inequality (x : ℝ) :
  (((2 * x - 1) / (x + 2)) > 1) ↔ (x < -2 ∨ x > 3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3764_376422


namespace NUMINAMATH_CALUDE_cow_count_l3764_376440

/-- Given a group of cows and hens, prove that the number of cows is 4 when the total number of legs
    is 8 more than twice the number of heads. -/
theorem cow_count (cows hens : ℕ) : 
  (4 * cows + 2 * hens = 2 * (cows + hens) + 8) → cows = 4 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l3764_376440


namespace NUMINAMATH_CALUDE_blake_change_l3764_376498

def oranges_cost : ℝ := 40
def apples_cost : ℝ := 50
def mangoes_cost : ℝ := 60
def strawberries_cost : ℝ := 30
def bananas_cost : ℝ := 20
def strawberries_discount : ℝ := 0.10
def bananas_discount : ℝ := 0.05
def blake_money : ℝ := 300

theorem blake_change :
  let discounted_strawberries := strawberries_cost * (1 - strawberries_discount)
  let discounted_bananas := bananas_cost * (1 - bananas_discount)
  let total_cost := oranges_cost + apples_cost + mangoes_cost + discounted_strawberries + discounted_bananas
  blake_money - total_cost = 104 := by sorry

end NUMINAMATH_CALUDE_blake_change_l3764_376498


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3764_376483

-- Define the solution set for the inequality
def solution_set (m : ℝ) : Set ℝ :=
  if m < -4 then { x | -1 < x ∧ x < 1 / (m + 3) }
  else if m = -4 then ∅
  else if m > -4 ∧ m < -3 then { x | 1 / (m + 3) < x ∧ x < -1 }
  else if m = -3 then { x | x > -1 }
  else { x | x < -1 ∨ x > 1 / (m + 3) }

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  { x : ℝ | (m + 3) * x - 1 > 0 } = solution_set m := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3764_376483


namespace NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_for_three_roots_l3764_376439

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (A B C : ℝ), A ≠ 0 ∧ 
  (∀ x y : ℝ, y = f x → (A * x + B * y + C = 0) ↔ x = 2) ∧
  A = 12 ∧ B = -1 ∧ C = -17 :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_three_roots :
  ∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f x₁ + m = 0 ∧ f x₂ + m = 0 ∧ f x₃ + m = 0) ↔ 
  -3 < m ∧ m < -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_range_of_m_for_three_roots_l3764_376439


namespace NUMINAMATH_CALUDE_total_weight_is_350_l3764_376413

/-- Represents the weight of a single box in kilograms -/
def box_weight : ℕ := 25

/-- Represents the number of columns with 3 boxes -/
def columns_with_3 : ℕ := 1

/-- Represents the number of columns with 2 boxes -/
def columns_with_2 : ℕ := 4

/-- Represents the number of columns with 1 box -/
def columns_with_1 : ℕ := 3

/-- Calculates the total number of boxes in the stack -/
def total_boxes : ℕ := columns_with_3 * 3 + columns_with_2 * 2 + columns_with_1 * 1

/-- Calculates the total weight of all boxes in kilograms -/
def total_weight : ℕ := total_boxes * box_weight

/-- Theorem stating that the total weight of all boxes is 350 kg -/
theorem total_weight_is_350 : total_weight = 350 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_350_l3764_376413


namespace NUMINAMATH_CALUDE_lines_properties_l3764_376484

/-- Two lines in 2D space -/
structure TwoLines where
  m : ℝ
  l1 : ℝ → ℝ → Prop := λ x y ↦ x + m * y - 1 = 0
  l2 : ℝ → ℝ → Prop := λ x y ↦ m * x + y - 1 = 0

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : TwoLines) : ℝ :=
  sorry

/-- Predicate for perpendicular lines -/
def are_perpendicular (lines : TwoLines) : Prop :=
  sorry

theorem lines_properties (lines : TwoLines) :
  (lines.l1 = lines.l2 → distance_between_parallel_lines lines = Real.sqrt 2) ∧
  (are_perpendicular lines → lines.m = 0) ∧
  lines.l2 0 1 := by
  sorry

end NUMINAMATH_CALUDE_lines_properties_l3764_376484


namespace NUMINAMATH_CALUDE_gilbert_cricket_ratio_l3764_376468

/-- The number of crickets Gilbert eats per week at 90°F -/
def crickets_90 : ℕ := 4

/-- The total number of weeks -/
def total_weeks : ℕ := 15

/-- The fraction of time the temperature is 90°F -/
def temp_90_fraction : ℚ := 4/5

/-- The total number of crickets eaten over the entire period -/
def total_crickets : ℕ := 72

/-- The number of crickets Gilbert eats per week at 100°F -/
def crickets_100 : ℕ := 8

theorem gilbert_cricket_ratio :
  crickets_100 / crickets_90 = 2 :=
sorry

end NUMINAMATH_CALUDE_gilbert_cricket_ratio_l3764_376468


namespace NUMINAMATH_CALUDE_milk_butter_revenue_l3764_376445

/-- Calculates the total revenue from selling milk and butter --/
def total_revenue (num_cows : ℕ) (milk_per_cow : ℕ) (milk_price : ℚ) (butter_sticks_per_gallon : ℕ) (butter_price : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let milk_revenue := total_milk * milk_price
  milk_revenue

theorem milk_butter_revenue :
  let num_cows : ℕ := 12
  let milk_per_cow : ℕ := 4
  let milk_price : ℚ := 3
  let butter_sticks_per_gallon : ℕ := 2
  let butter_price : ℚ := 3/2
  total_revenue num_cows milk_per_cow milk_price butter_sticks_per_gallon butter_price = 144 := by
  sorry

end NUMINAMATH_CALUDE_milk_butter_revenue_l3764_376445


namespace NUMINAMATH_CALUDE_arrangements_count_l3764_376494

def num_tour_groups : ℕ := 4
def num_scenic_spots : ℕ := 4

/-- The number of arrangements for tour groups choosing scenic spots -/
def num_arrangements : ℕ :=
  (num_tour_groups.choose 2) * (num_scenic_spots * (num_scenic_spots - 1) * (num_scenic_spots - 2))

theorem arrangements_count :
  num_arrangements = 144 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l3764_376494


namespace NUMINAMATH_CALUDE_height_comparison_l3764_376455

theorem height_comparison (ashis_height babji_height : ℝ) 
  (h : babji_height = ashis_height * (1 - 0.2)) :
  (ashis_height - babji_height) / babji_height = 0.25 := by
sorry

end NUMINAMATH_CALUDE_height_comparison_l3764_376455


namespace NUMINAMATH_CALUDE_min_sum_floor_l3764_376403

theorem min_sum_floor (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (⌊(2*x + y) / z⌋ : ℤ) + ⌊(y + 2*z) / x⌋ + ⌊(2*z + x) / y⌋ = 9 ∧
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (⌊(2*a + b) / c⌋ : ℤ) + ⌊(b + 2*c) / a⌋ + ⌊(2*c + a) / b⌋ ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_floor_l3764_376403


namespace NUMINAMATH_CALUDE_range_of_f_l3764_376495

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x - 2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.sqrt 3 / 2),
  ∃ y ∈ Set.Icc (-3) (-2),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-3) (-2) :=
by sorry

-- Define the trigonometric identity
axiom cos_triple_angle (θ : ℝ) :
  Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ)

end NUMINAMATH_CALUDE_range_of_f_l3764_376495


namespace NUMINAMATH_CALUDE_greatest_integer_pi_minus_five_l3764_376496

theorem greatest_integer_pi_minus_five :
  ⌊Real.pi - 5⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_pi_minus_five_l3764_376496


namespace NUMINAMATH_CALUDE_complex_number_properties_l3764_376493

theorem complex_number_properties (z : ℂ) (h : z = -1/2 + Complex.I * (Real.sqrt 3 / 2)) : 
  z^3 = 1 ∧ z^2 + z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3764_376493


namespace NUMINAMATH_CALUDE_textbook_profit_example_l3764_376415

/-- The profit of a textbook sale given its cost and selling prices -/
def textbook_profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook sold by a bookstore is $11,
    given that the cost price is $44 and the selling price is $55. -/
theorem textbook_profit_example : textbook_profit 44 55 = 11 := by
  sorry

end NUMINAMATH_CALUDE_textbook_profit_example_l3764_376415


namespace NUMINAMATH_CALUDE_apple_juice_percentage_is_40_percent_l3764_376428

/-- Represents the juice yield from fruits -/
structure JuiceYield where
  apples : Nat
  appleJuice : Nat
  bananas : Nat
  bananaJuice : Nat

/-- Calculates the percentage of apple juice in a blend -/
def appleJuicePercentage (yield : JuiceYield) : Rat :=
  let appleJuicePerFruit := yield.appleJuice / yield.apples
  let bananaJuicePerFruit := yield.bananaJuice / yield.bananas
  let totalJuice := appleJuicePerFruit + bananaJuicePerFruit
  appleJuicePerFruit / totalJuice

/-- Theorem: The percentage of apple juice in the blend is 40% -/
theorem apple_juice_percentage_is_40_percent (yield : JuiceYield) 
    (h1 : yield.apples = 5)
    (h2 : yield.appleJuice = 10)
    (h3 : yield.bananas = 4)
    (h4 : yield.bananaJuice = 12) : 
  appleJuicePercentage yield = 2/5 := by
  sorry

#eval (2 : Rat) / 5

end NUMINAMATH_CALUDE_apple_juice_percentage_is_40_percent_l3764_376428


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3764_376423

theorem quadratic_factorization (a b : ℤ) : 
  (∀ x, 25 * x^2 - 115 * x - 150 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -55 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3764_376423


namespace NUMINAMATH_CALUDE_complex_cube_root_l3764_376476

theorem complex_cube_root (a b : ℕ+) :
  (a + b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  a + b * Complex.I = 2 + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l3764_376476


namespace NUMINAMATH_CALUDE_total_trip_time_l3764_376412

/-- Given that Tim drove for 5 hours and was stuck in traffic for twice as long as he was driving,
    prove that the total trip time is 15 hours. -/
theorem total_trip_time (driving_time : ℕ) (traffic_time : ℕ) : 
  driving_time = 5 →
  traffic_time = 2 * driving_time →
  driving_time + traffic_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_trip_time_l3764_376412


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3764_376459

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the eccentricity of the hyperbola is √(17)/3 -/
theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (F : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : b = 1)
  (h4 : F = (c, 0))
  (h5 : B = (0, 1))
  (h6 : A.1^2 / a^2 - A.2^2 / b^2 = 1)  -- A is on the hyperbola
  (h7 : A.1^2 = 4 * A.2)                -- A is on the parabola
  (h8 : (A.1 - B.1, A.2 - B.2) = 3 * (F.1 - A.1, F.2 - A.2))  -- BA = 3AF
  : c / a = Real.sqrt 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3764_376459


namespace NUMINAMATH_CALUDE_negation_equivalence_l3764_376406

theorem negation_equivalence :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3764_376406


namespace NUMINAMATH_CALUDE_zero_not_in_N_star_l3764_376411

-- Define the set of natural numbers
def N : Set ℕ := {n : ℕ | n > 0}

-- Define the set of positive integers (N*)
def N_star : Set ℕ := N

-- Define the set of rational numbers
def Q : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Theorem statement
theorem zero_not_in_N_star : 0 ∉ N_star := by
  sorry

end NUMINAMATH_CALUDE_zero_not_in_N_star_l3764_376411


namespace NUMINAMATH_CALUDE_twenty_seven_thousand_six_hundred_scientific_notation_l3764_376467

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 27600 = ScientificNotation.mk 2.76 4 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_twenty_seven_thousand_six_hundred_scientific_notation_l3764_376467


namespace NUMINAMATH_CALUDE_tulip_arrangement_l3764_376416

/-- The number of red tulips needed for the smile -/
def smile_tulips : ℕ := 18

/-- The number of yellow tulips for the background is 9 times the number of red tulips in the smile -/
def background_tulips : ℕ := 9 * smile_tulips

/-- The total number of tulips needed -/
def total_tulips : ℕ := 196

/-- The number of red tulips needed for each eye -/
def eye_tulips : ℕ := 8

theorem tulip_arrangement : 
  2 * eye_tulips + smile_tulips + background_tulips = total_tulips :=
sorry

end NUMINAMATH_CALUDE_tulip_arrangement_l3764_376416


namespace NUMINAMATH_CALUDE_gumball_machine_problem_l3764_376430

theorem gumball_machine_problem (red blue green : ℕ) : 
  blue = red / 2 →
  green = 4 * blue →
  red + blue + green = 56 →
  red = 16 := by
  sorry

end NUMINAMATH_CALUDE_gumball_machine_problem_l3764_376430


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3764_376441

/-- Given that the solution set of ax^2 + bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≤ -3 or x ≥ -2} -/
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, ax^2 + b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) →
  (∀ x, x^2 - b*x - a ≥ 0 ↔ x ≤ -3 ∨ x ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3764_376441


namespace NUMINAMATH_CALUDE_cable_cost_per_person_l3764_376480

/-- Represents the cable program tiers and discount rates --/
structure CableProgram where
  tier1_channels : ℕ := 100
  tier1_cost : ℚ := 100
  tier2_channels : ℕ := 150
  tier2_cost : ℚ := 75
  tier3_channels : ℕ := 200
  tier4_channels : ℕ := 250
  discount_200 : ℚ := 0.1
  discount_300 : ℚ := 0.15
  discount_500 : ℚ := 0.2

/-- Calculates the cost for a given number of channels --/
def calculateCost (program : CableProgram) (channels : ℕ) : ℚ :=
  sorry

/-- Applies the appropriate discount based on the number of channels --/
def applyDiscount (program : CableProgram) (cost : ℚ) (channels : ℕ) : ℚ :=
  sorry

/-- Theorem: The cost per person for 375 channels split among 4 people is $57.11 --/
theorem cable_cost_per_person (program : CableProgram) :
  let total_cost := calculateCost program 375
  let discounted_cost := applyDiscount program total_cost 375
  let cost_per_person := discounted_cost / 4
  cost_per_person = 57.11 := by
  sorry

end NUMINAMATH_CALUDE_cable_cost_per_person_l3764_376480


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3764_376435

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 5 = Real.sqrt (5^a * 5^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3764_376435


namespace NUMINAMATH_CALUDE_remainder_of_2367905_div_5_l3764_376475

theorem remainder_of_2367905_div_5 : 2367905 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2367905_div_5_l3764_376475


namespace NUMINAMATH_CALUDE_log_inequality_l3764_376462

theorem log_inequality (x : ℝ) (h : x > 0) : Real.log (1 + x) < x / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3764_376462


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3764_376433

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2/a + 3/b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3764_376433


namespace NUMINAMATH_CALUDE_jokes_theorem_l3764_376490

def calculate_jokes (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial + 8 * initial + 16 * initial

def total_jokes : ℕ :=
  calculate_jokes 11 + calculate_jokes 7 + calculate_jokes 5 + calculate_jokes 3

theorem jokes_theorem : total_jokes = 806 := by
  sorry

end NUMINAMATH_CALUDE_jokes_theorem_l3764_376490


namespace NUMINAMATH_CALUDE_S_five_three_l3764_376460

-- Define the operation ∘
def S (a b : ℕ) : ℕ := 4 * a + 3 * b

-- Theorem statement
theorem S_five_three : S 5 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_S_five_three_l3764_376460


namespace NUMINAMATH_CALUDE_polar_equation_is_line_and_circle_l3764_376477

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2 * Real.sin (2 * θ)

-- Define what it means for a curve to be a line in polar coordinates
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ θ₀ : ℝ, ∀ ρ θ : ℝ, f ρ θ → θ = θ₀

-- Define what it means for a curve to be a circle in polar coordinates
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ → (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

-- Theorem statement
theorem polar_equation_is_line_and_circle :
  is_line polar_equation ∧ is_circle polar_equation := by sorry

end NUMINAMATH_CALUDE_polar_equation_is_line_and_circle_l3764_376477


namespace NUMINAMATH_CALUDE_susie_piggy_bank_total_l3764_376419

/-- Calculates the total amount in Susie's piggy bank after two years -/
def piggy_bank_total (initial_amount : ℝ) (first_year_addition : ℝ) (second_year_addition : ℝ) (interest_rate : ℝ) : ℝ :=
  let first_year_total := (initial_amount + initial_amount * first_year_addition) * (1 + interest_rate)
  let second_year_total := (first_year_total + first_year_total * second_year_addition) * (1 + interest_rate)
  second_year_total

/-- Theorem stating that Susie's piggy bank total after two years is $343.98 -/
theorem susie_piggy_bank_total :
  piggy_bank_total 200 0.2 0.3 0.05 = 343.98 := by
  sorry

end NUMINAMATH_CALUDE_susie_piggy_bank_total_l3764_376419


namespace NUMINAMATH_CALUDE_expression_evaluation_l3764_376492

theorem expression_evaluation : 
  (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3764_376492


namespace NUMINAMATH_CALUDE_systematic_sampling_correct_l3764_376427

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  start : ℕ

/-- Generates the sequence of selected student numbers -/
def generate_sequence (s : SystematicSampling) : List ℕ :=
  List.range s.sample_size |>.map (fun i => s.start + i * (s.total_students / s.sample_size))

/-- Theorem: The systematic sampling of 6 students from 60 results in the correct sequence -/
theorem systematic_sampling_correct : 
  let s : SystematicSampling := ⟨60, 6, 6⟩
  generate_sequence s = [6, 16, 26, 36, 46, 56] := by
  sorry

#eval generate_sequence ⟨60, 6, 6⟩

end NUMINAMATH_CALUDE_systematic_sampling_correct_l3764_376427


namespace NUMINAMATH_CALUDE_divisible_by_three_or_six_percentage_l3764_376465

theorem divisible_by_three_or_six_percentage (n : Nat) : 
  n = 200 → 
  (((Finset.filter (fun x => x % 3 = 0 ∨ x % 6 = 0) (Finset.range (n + 1))).card : ℚ) / n) * 100 = 33 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_three_or_six_percentage_l3764_376465


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l3764_376450

theorem no_solutions_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (2 / x + 3 / y = 1 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l3764_376450


namespace NUMINAMATH_CALUDE_max_cables_cut_l3764_376401

/-- Represents a computer network -/
structure ComputerNetwork where
  numComputers : ℕ
  numCables : ℕ
  numClusters : ℕ

/-- The initial state of the computer network -/
def initialNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345
  , numClusters := 1 }

/-- The final state of the computer network after cutting cables -/
def finalNetwork : ComputerNetwork :=
  { numComputers := 200
  , numCables := 345 - 153
  , numClusters := 8 }

/-- Theorem stating the maximum number of cables that can be cut -/
theorem max_cables_cut (initial : ComputerNetwork) (final : ComputerNetwork) :
  initial.numComputers = 200 →
  initial.numCables = 345 →
  initial.numClusters = 1 →
  final.numComputers = initial.numComputers →
  final.numClusters = 8 →
  final.numCables = initial.numCables - 153 →
  ∀ n : ℕ, n > 153 → 
    ¬∃ (network : ComputerNetwork), 
      network.numComputers = initial.numComputers ∧
      network.numClusters = final.numClusters ∧
      network.numCables = initial.numCables - n :=
by sorry


end NUMINAMATH_CALUDE_max_cables_cut_l3764_376401


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l3764_376420

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line l
structure Line where
  k : ℝ
  b : ℝ

-- Define the problem conditions
def circle_conditions (C : Circle) : Prop :=
  let (x, y) := C.center
  x > 0 ∧ y > 0 ∧  -- Center is in the first quadrant
  3 * x = y ∧      -- Center lies on the line 3x - y = 0
  C.radius = y ∧   -- Circle is tangent to x-axis
  (2 * Real.sqrt 7) ^ 2 = 4 * (C.radius ^ 2 - x ^ 2)  -- Chord length condition

def line_intersects_circle (l : Line) (C : Circle) : Prop :=
  ∃ (x y : ℝ), l.k * x - y - 2 * l.k + 5 = 0 ∧
                (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2

-- Theorem statement
theorem circle_and_line_properties :
  ∀ (C : Circle) (l : Line),
    circle_conditions C →
    line_intersects_circle l C →
    (∀ (x y : ℝ), (x - 1) ^ 2 + (y - 3) ^ 2 = 9 ↔ (x - C.center.1) ^ 2 + (y - C.center.2) ^ 2 = C.radius ^ 2) ∧
    (∃ (k : ℝ), l.k = k ∧ l.b = 5 - 2 * k) ∧
    (∃ (l_shortest : Line), 
      l_shortest.k = -1/2 ∧ 
      l_shortest.b = 6 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (d d' : ℝ), 
          d = (abs (l_shortest.k * C.center.1 - C.center.2 + l_shortest.b)) / Real.sqrt (l_shortest.k ^ 2 + 1) ∧
          d' = (abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1) ∧
          d < d') ∧
    (∃ (shortest_chord : ℝ), shortest_chord = 4 ∧
      ∀ (l' : Line), l'.k ≠ -1/2 → 
        ∃ (chord : ℝ), 
          chord = 2 * Real.sqrt (C.radius ^ 2 - ((abs (l'.k * C.center.1 - C.center.2 + l'.b)) / Real.sqrt (l'.k ^ 2 + 1)) ^ 2) ∧
          chord > shortest_chord) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l3764_376420


namespace NUMINAMATH_CALUDE_yearly_increase_fraction_l3764_376461

theorem yearly_increase_fraction (initial_value final_value : ℝ) (f : ℝ) 
    (h1 : initial_value = 51200)
    (h2 : final_value = 64800)
    (h3 : initial_value * (1 + f)^2 = final_value) :
  f = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_yearly_increase_fraction_l3764_376461


namespace NUMINAMATH_CALUDE_min_colors_needed_l3764_376436

/-- Represents a color assignment for hats and ribbons --/
structure ColorAssignment (n : ℕ) where
  hatColors : Fin n → Fin n
  ribbonColors : Fin n → Fin n → Fin n

/-- A valid color assignment satisfies the problem constraints --/
def isValidColorAssignment (n : ℕ) (ca : ColorAssignment n) : Prop :=
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors i) ∧
  (∀ i j : Fin n, i ≠ j → ca.ribbonColors i j ≠ ca.hatColors j) ∧
  (∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → ca.ribbonColors i j ≠ ca.ribbonColors i k)

/-- The main theorem: n colors are sufficient and necessary --/
theorem min_colors_needed (n : ℕ) (h : n ≥ 2) :
  (∃ ca : ColorAssignment n, isValidColorAssignment n ca) ∧
  (∀ m : ℕ, m < n → ¬∃ ca : ColorAssignment m, isValidColorAssignment m ca) :=
sorry

end NUMINAMATH_CALUDE_min_colors_needed_l3764_376436


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l3764_376421

def remaining_pastries (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

theorem baker_remaining_pastries :
  remaining_pastries 56 29 = 27 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l3764_376421


namespace NUMINAMATH_CALUDE_second_bag_popped_kernels_l3764_376497

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_popped_kernels 
  (bag1 : PopcornBag)
  (bag2 : PopcornBag)
  (bag3 : PopcornBag)
  (h1 : bag1.total = 75)
  (h2 : bag1.popped = 60)
  (h3 : bag2.total = 50)
  (h4 : bag3.total = 100)
  (h5 : bag3.popped = 82)
  (h6 : (popPercentage bag1 + popPercentage bag2 + popPercentage bag3) / 3 = 82) :
  bag2.popped = 42 := by
  sorry

#eval PopcornBag.popped { total := 50, popped := 42 }

end NUMINAMATH_CALUDE_second_bag_popped_kernels_l3764_376497


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3764_376405

theorem nested_fraction_evaluation : 
  1 / (1 + 1 / (2 + 1 / (4^2))) = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3764_376405


namespace NUMINAMATH_CALUDE_product_evaluation_l3764_376456

theorem product_evaluation (n : ℤ) (h : n = 3) : 
  (n - 4) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = -5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3764_376456


namespace NUMINAMATH_CALUDE_f_composition_l3764_376448

def f (x : ℝ) := 2 * x + 1

theorem f_composition (x : ℝ) : f (2 * x - 1) = 4 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_l3764_376448


namespace NUMINAMATH_CALUDE_optimal_production_value_l3764_376457

/-- Represents the production plan for products A and B -/
structure ProductionPlan where
  a : ℝ  -- Amount of product A in kg
  b : ℝ  -- Amount of product B in kg

/-- Calculates the total value of a production plan -/
def totalValue (plan : ProductionPlan) : ℝ :=
  600 * plan.a + 400 * plan.b

/-- Checks if a production plan is feasible given the raw material constraints -/
def isFeasible (plan : ProductionPlan) : Prop :=
  4 * plan.a + 2 * plan.b ≤ 100 ∧  -- Raw material A constraint
  2 * plan.a + 3 * plan.b ≤ 120    -- Raw material B constraint

/-- The optimal production plan -/
def optimalPlan : ProductionPlan :=
  { a := 7.5, b := 35 }

theorem optimal_production_value :
  (∀ plan : ProductionPlan, isFeasible plan → totalValue plan ≤ totalValue optimalPlan) ∧
  isFeasible optimalPlan ∧
  totalValue optimalPlan = 18500 := by
  sorry

end NUMINAMATH_CALUDE_optimal_production_value_l3764_376457


namespace NUMINAMATH_CALUDE_total_material_bought_l3764_376486

/-- The total amount of material bought by a construction company -/
theorem total_material_bought (gravel sand : ℝ) (h1 : gravel = 5.91) (h2 : sand = 8.11) :
  gravel + sand = 14.02 := by
  sorry

end NUMINAMATH_CALUDE_total_material_bought_l3764_376486


namespace NUMINAMATH_CALUDE_fencing_calculation_l3764_376431

/-- Represents a rectangular field with given dimensions and fencing requirements -/
structure RectangularField where
  length : ℝ
  width : ℝ
  uncoveredSide : ℝ
  area : ℝ

/-- Calculates the required fencing for a rectangular field -/
def requiredFencing (field : RectangularField) : ℝ :=
  2 * field.width + field.length

/-- Theorem stating the required fencing for the given field specifications -/
theorem fencing_calculation (field : RectangularField) 
  (h1 : field.length = 20)
  (h2 : field.area = 390)
  (h3 : field.area = field.length * field.width)
  (h4 : field.uncoveredSide = field.length) :
  requiredFencing field = 59 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l3764_376431


namespace NUMINAMATH_CALUDE_male_alligators_mating_season_l3764_376479

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adultFemales : ℕ
  juvenileFemales : ℕ

/-- Calculates the total number of alligators -/
def totalAlligators (pop : AlligatorPopulation) : ℕ :=
  pop.males + pop.adultFemales + pop.juvenileFemales

/-- Represents the population ratio of males:adult females:juvenile females -/
structure PopulationRatio where
  maleRatio : ℕ
  adultFemaleRatio : ℕ
  juvenileFemaleRatio : ℕ

/-- Theorem: Given the conditions, the number of male alligators during mating season is 10 -/
theorem male_alligators_mating_season
  (ratio : PopulationRatio)
  (nonMatingAdultFemales : ℕ)
  (resourceLimit : ℕ)
  (turtleRatio : ℕ)
  (h1 : ratio.maleRatio = 2 ∧ ratio.adultFemaleRatio = 3 ∧ ratio.juvenileFemaleRatio = 5)
  (h2 : nonMatingAdultFemales = 15)
  (h3 : resourceLimit = 200)
  (h4 : turtleRatio = 3)
  : ∃ (pop : AlligatorPopulation),
    pop.males = 10 ∧
    pop.adultFemales = 2 * nonMatingAdultFemales ∧
    totalAlligators pop ≤ resourceLimit ∧
    turtleRatio * (totalAlligators pop) ≤ 3 * resourceLimit :=
by sorry


end NUMINAMATH_CALUDE_male_alligators_mating_season_l3764_376479


namespace NUMINAMATH_CALUDE_dawns_lemonade_price_l3764_376452

/-- The price of Dawn's lemonade in cents -/
def dawns_price : ℕ := sorry

/-- The number of glasses Bea sold -/
def bea_glasses : ℕ := 10

/-- The number of glasses Dawn sold -/
def dawn_glasses : ℕ := 8

/-- The price of Bea's lemonade in cents -/
def bea_price : ℕ := 25

/-- The difference in earnings between Bea and Dawn in cents -/
def earnings_difference : ℕ := 26

theorem dawns_lemonade_price :
  dawns_price = 28 ∧
  bea_glasses * bea_price = dawn_glasses * dawns_price + earnings_difference :=
sorry

end NUMINAMATH_CALUDE_dawns_lemonade_price_l3764_376452


namespace NUMINAMATH_CALUDE_fly_distance_from_floor_l3764_376485

theorem fly_distance_from_floor (x y z h : ℝ) :
  x = 2 →
  y = 5 →
  h - z = 7 →
  x^2 + y^2 + z^2 = 11^2 →
  h = Real.sqrt 92 + 7 := by
sorry

end NUMINAMATH_CALUDE_fly_distance_from_floor_l3764_376485


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3764_376464

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x^2 + y^2 = 2) 
  (h2 : |x| ≠ |y|) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), a^2 + b^2 = 2 → |a| ≠ |b| → 
    (1 / (a + b)^2 + 1 / (a - b)^2) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3764_376464


namespace NUMINAMATH_CALUDE_g_of_5_equals_22_l3764_376426

/-- Given that g(x) = 4x + 2 for all x, prove that g(5) = 22 -/
theorem g_of_5_equals_22 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x + 2) : g 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_equals_22_l3764_376426


namespace NUMINAMATH_CALUDE_valid_words_length_10_l3764_376451

/-- Represents the number of valid words of length n -/
def validWords : ℕ → ℕ
  | 0 => 1  -- Base case: empty word
  | 1 => 2  -- Base case: "a" and "b"
  | (n+2) => validWords (n+1) + validWords n

/-- The problem statement -/
theorem valid_words_length_10 : validWords 10 = 144 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_length_10_l3764_376451


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3764_376442

theorem quadratic_one_solution (m : ℚ) :
  (∃! x, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3764_376442


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3764_376499

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 17 + 23 + 7 + y) / 5 = 15 → y = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3764_376499


namespace NUMINAMATH_CALUDE_correct_arrangements_l3764_376469

/-- The number of people in the row -/
def n : ℕ := 8

/-- The number of special people (A, B, C, D, E) -/
def k : ℕ := 5

/-- Function to calculate the number of arrangements -/
def count_arrangements (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements -/
theorem correct_arrangements :
  count_arrangements n k = 11520 := by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l3764_376469


namespace NUMINAMATH_CALUDE_ab_less_than_a_plus_b_l3764_376489

theorem ab_less_than_a_plus_b (a b : ℝ) (ha : a < 1) (hb : b > 1) : a * b < a + b := by
  sorry

end NUMINAMATH_CALUDE_ab_less_than_a_plus_b_l3764_376489


namespace NUMINAMATH_CALUDE_lemonade_percentage_in_second_solution_l3764_376446

/-- Represents a solution mixture --/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents the mixture of two solutions --/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (total_carbonated_water : ℝ)

/-- The theorem to be proved --/
theorem lemonade_percentage_in_second_solution 
  (mix : Mixture) 
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution1.carbonated_water = 0.8)
  (h3 : mix.solution2.lemonade + mix.solution2.carbonated_water = 1)
  (h4 : mix.proportion1 = 0.4)
  (h5 : mix.proportion2 = 0.6)
  (h6 : mix.total_carbonated_water = 0.65) :
  mix.solution2.lemonade = 0.9945 :=
sorry

end NUMINAMATH_CALUDE_lemonade_percentage_in_second_solution_l3764_376446


namespace NUMINAMATH_CALUDE_set_product_theorem_l3764_376473

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | 0 ≤ y}

-- Define the operation ×
def setProduct (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Theorem statement
theorem set_product_theorem :
  setProduct A B = {x | -1 ≤ x ∧ x < 0 ∨ 1 < x} :=
by sorry

end NUMINAMATH_CALUDE_set_product_theorem_l3764_376473


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3764_376444

theorem quadratic_root_relation (b c : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁^2 = -x₂) →
  (b^3 - 3*b*c - c^2 - c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3764_376444


namespace NUMINAMATH_CALUDE_fraction_equality_l3764_376471

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3764_376471


namespace NUMINAMATH_CALUDE_complex_modulus_3_plus_2i_l3764_376474

theorem complex_modulus_3_plus_2i : 
  Complex.abs (3 + 2 * Complex.I) = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_3_plus_2i_l3764_376474


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l3764_376453

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights chosen for the quest -/
def chosen_knights : ℕ := 5

/-- The probability that at least two of the chosen knights are sitting next to each other -/
def P : ℚ := 141505 / 142506

/-- Theorem stating the probability of adjacent chosen knights -/
theorem adjacent_knights_probability :
  (1 : ℚ) - (Nat.choose (total_knights - chosen_knights - (chosen_knights - 1)) (chosen_knights - 1) : ℚ) / 
  (Nat.choose total_knights chosen_knights : ℚ) = P := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l3764_376453


namespace NUMINAMATH_CALUDE_necessary_condition_implies_m_range_l3764_376463

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
def B (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ [-1, 1] ∧ y = 1/3 * x + m}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem necessary_condition_implies_m_range :
  ∀ m : ℝ, (∀ x : ℝ, q m x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q m x) →
  m > 1/3 ∧ m < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_implies_m_range_l3764_376463
