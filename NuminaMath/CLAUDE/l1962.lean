import Mathlib

namespace fermats_little_theorem_distinct_colorings_l1962_196237

theorem fermats_little_theorem (p : ℕ) (n : ℤ) (hp : Nat.Prime p) :
  (↑n ^ p - n : ℤ) % ↑p = 0 := by
  sorry

theorem distinct_colorings (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  ∃ k : ℕ, (n ^ p - n : ℕ) / p + n = k := by
  sorry

end fermats_little_theorem_distinct_colorings_l1962_196237


namespace other_communities_count_l1962_196220

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 40 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑⌊(1 - (muslim_percent + hindu_percent + sikh_percent)) * total_boys⌋ = 187 :=
by sorry

end other_communities_count_l1962_196220


namespace farmer_crops_after_pest_destruction_l1962_196262

-- Define the constants
def corn_cobs_per_row : ℕ := 9
def potatoes_per_row : ℕ := 30
def corn_rows : ℕ := 10
def potato_rows : ℕ := 5
def pest_destruction_ratio : ℚ := 1/2

-- Define the theorem
theorem farmer_crops_after_pest_destruction :
  (corn_rows * corn_cobs_per_row + potato_rows * potatoes_per_row) * pest_destruction_ratio = 120 := by
  sorry

end farmer_crops_after_pest_destruction_l1962_196262


namespace polynomial_equivalence_l1962_196291

theorem polynomial_equivalence (x : ℝ) (y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 4*x^2 + x + 1 = x^2 * (y^2 + y - 6) :=
by sorry

end polynomial_equivalence_l1962_196291


namespace triangle_abc_properties_l1962_196299

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1/2 →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧ 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    1/2 * a' * b' * Real.sin C ≤ 3 * Real.sqrt 3) := by
  sorry

end triangle_abc_properties_l1962_196299


namespace cubic_roots_sum_l1962_196228

theorem cubic_roots_sum (m : ℤ) (a b c : ℤ) :
  (∀ x : ℤ, x^3 - 2015*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 100 :=
by sorry

end cubic_roots_sum_l1962_196228


namespace no_other_products_of_three_primes_l1962_196263

/-- The reverse of a natural number -/
def reverse (n : ℕ) : ℕ := sorry

/-- Predicate for a number being the product of exactly three distinct primes -/
def isProductOfThreeDistinctPrimes (n : ℕ) : Prop := sorry

theorem no_other_products_of_three_primes : 
  let original := 2017
  let reversed := 7102
  -- 7102 is the reverse of 2017
  reverse original = reversed →
  -- 7102 is the product of three distinct primes p, q, and r
  ∃ (p q r : ℕ), isProductOfThreeDistinctPrimes reversed ∧ 
                 reversed = p * q * r ∧ 
                 p ≠ q ∧ p ≠ r ∧ q ≠ r →
  -- There are no other positive integers that are products of three distinct primes 
  -- summing to the same value as p + q + r
  ¬∃ (n : ℕ), n ≠ reversed ∧ 
              isProductOfThreeDistinctPrimes n ∧
              (∃ (p1 p2 p3 : ℕ), n = p1 * p2 * p3 ∧ 
                                 p1 + p2 + p3 = p + q + r ∧
                                 p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
by sorry

end no_other_products_of_three_primes_l1962_196263


namespace ratio_of_sum_and_difference_l1962_196265

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end ratio_of_sum_and_difference_l1962_196265


namespace max_t_value_l1962_196287

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 16 →
  k < m → m < r → r < s → s < t →
  r ≤ 17 →
  t ≤ 42 :=
by sorry

end max_t_value_l1962_196287


namespace root_sum_theorem_l1962_196281

def polynomial (x : ℂ) : ℂ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem root_sum_theorem (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : polynomial z₁ = 0)
  (h₂ : polynomial z₂ = 0)
  (h₃ : polynomial z₃ = 0)
  (h₄ : polynomial z₄ = 0)
  (h₅ : polynomial z₅ = 0) :
  (z₁ / (z₁^2 + 1) + z₂ / (z₂^2 + 1) + z₃ / (z₃^2 + 1) + z₄ / (z₄^2 + 1) + z₅ / (z₅^2 + 1)) = 4/17 := by
  sorry

end root_sum_theorem_l1962_196281


namespace solution_set_intersection_condition_l1962_196245

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - |x + 1|

-- Define the quadratic function
def g (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Theorem for part (1)
theorem solution_set (x : ℝ) : f 5 x > 2 ↔ -3/2 < x ∧ x < 3/2 :=
sorry

-- Theorem for part (2)
theorem intersection_condition (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m y = g x) ↔ m ≥ 4 :=
sorry

end solution_set_intersection_condition_l1962_196245


namespace compound_molecular_weight_l1962_196209

/-- Atomic weight of Calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- Number of Calcium atoms in the compound -/
def Ca_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 2

/-- Number of Hydrogen atoms in the compound -/
def H_count : ℕ := 2

/-- Calculates the molecular weight of the compound -/
def molecular_weight : ℝ :=
  Ca_count * Ca_weight + O_count * O_weight + H_count * H_weight

/-- Theorem stating that the molecular weight of the compound is approximately 74.094 g/mol -/
theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 74.094| < ε :=
by sorry

end compound_molecular_weight_l1962_196209


namespace tensor_inequality_implies_a_range_l1962_196231

-- Define the operation ⊗
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- Theorem statement
theorem tensor_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → tensor (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 :=
by sorry

end tensor_inequality_implies_a_range_l1962_196231


namespace b_95_mod_49_l1962_196272

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- The remainder of b_95 when divided by 49 is 36 -/
theorem b_95_mod_49 : b 95 % 49 = 36 := by
  sorry

end b_95_mod_49_l1962_196272


namespace line_segments_not_in_proportion_l1962_196271

theorem line_segments_not_in_proportion :
  let a : ℝ := 4
  let b : ℝ := 5
  let c : ℝ := 6
  let d : ℝ := 10
  (a / b) ≠ (c / d) :=
by sorry

end line_segments_not_in_proportion_l1962_196271


namespace smallest_number_above_threshold_l1962_196219

theorem smallest_number_above_threshold : 
  let numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let threshold : ℚ := 1.1
  let above_threshold := numbers.filter (λ x => x > threshold)
  above_threshold.minimum? = some 1.2 :=
by sorry

end smallest_number_above_threshold_l1962_196219


namespace vector_dot_product_l1962_196259

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (1, -3) ∧ a - b = (3, 7) → a • b = -12 := by sorry

end vector_dot_product_l1962_196259


namespace combined_weight_l1962_196238

/-- The combined weight of John, Mary, and Jamison is 540 lbs -/
theorem combined_weight (mary_weight jamison_weight john_weight : ℝ) :
  mary_weight = 160 →
  jamison_weight = mary_weight + 20 →
  john_weight = mary_weight * (5/4) →
  mary_weight + jamison_weight + john_weight = 540 := by
sorry

end combined_weight_l1962_196238


namespace stationery_profit_theorem_l1962_196290

/-- Profit function for a stationery item --/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 600 * x - 8000

/-- Daily sales volume function --/
def sales_volume (x : ℝ) : ℝ := -10 * x + 400

/-- Purchase price of the stationery item --/
def purchase_price : ℝ := 20

/-- Theorem stating the properties of the profit function and its maximum --/
theorem stationery_profit_theorem :
  (∀ x, profit_function x = (x - purchase_price) * sales_volume x) ∧
  (∃ x_max, ∀ x, profit_function x ≤ profit_function x_max ∧ x_max = 30) ∧
  (∃ x_constrained, 
    sales_volume x_constrained ≥ 120 ∧
    (∀ x, sales_volume x ≥ 120 → profit_function x ≤ profit_function x_constrained) ∧
    x_constrained = 28 ∧
    profit_function x_constrained = 960) :=
by sorry

end stationery_profit_theorem_l1962_196290


namespace triangle_conditions_l1962_196222

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

-- Condition A
def condition_A (t : Triangle) : Prop :=
  t.a = 1/3 ∧ t.b = 1/4 ∧ t.c = 1/5

-- Condition B (using angle ratios)
def condition_B (A B C : ℝ) : Prop :=
  A / B = 1/3 ∧ A / C = 1/2 ∧ B / C = 3/2

-- Condition C
def condition_C (t : Triangle) : Prop :=
  (t.b + t.c) * (t.b - t.c) = t.a^2

theorem triangle_conditions :
  (∃ t1 t2 : Triangle, condition_A t1 ∧ is_right_triangle t1 ∧
                       condition_A t2 ∧ ¬is_right_triangle t2) ∧
  (∀ A B C : ℝ, condition_B A B C → A + B + C = 180 → B = 90) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) :=
sorry

end triangle_conditions_l1962_196222


namespace triangle_consecutive_numbers_l1962_196280

/-- Represents the state of the triangle cells -/
def TriangleState := List Int

/-- Represents an operation on two adjacent cells -/
inductive Operation
| Add : Nat → Nat → Operation
| Subtract : Nat → Nat → Operation

/-- Checks if two cells are adjacent in the triangle -/
def are_adjacent (i j : Nat) : Bool := sorry

/-- Applies an operation to the triangle state -/
def apply_operation (state : TriangleState) (op : Operation) : TriangleState := sorry

/-- Checks if a list contains consecutive integers from n to n+8 -/
def is_consecutive_from_n (l : List Int) (n : Int) : Prop := sorry

/-- The main theorem -/
theorem triangle_consecutive_numbers :
  ∀ (initial_state : TriangleState),
  (initial_state.length = 9 ∧ initial_state.all (· = 0)) →
  ∃ (n : Int) (final_state : TriangleState),
  (∃ (ops : List Operation), 
    (∀ op ∈ ops, ∃ i j, are_adjacent i j ∧ (op = Operation.Add i j ∨ op = Operation.Subtract i j)) ∧
    (final_state = ops.foldl apply_operation initial_state)) ∧
  is_consecutive_from_n final_state n →
  n = 2 := by
  sorry

end triangle_consecutive_numbers_l1962_196280


namespace valid_prices_count_l1962_196260

def valid_digits : List Nat := [1, 1, 4, 5, 6, 6]

def is_valid_start (n : Nat) : Bool :=
  n ≥ 4

def count_valid_prices (digits : List Nat) : Nat :=
  digits.filter is_valid_start
    |>.map (λ d => (digits.erase d).permutations.length)
    |>.sum

theorem valid_prices_count :
  count_valid_prices valid_digits = 90 := by
  sorry

end valid_prices_count_l1962_196260


namespace functional_equation_solution_l1962_196283

/-- A function f: ℝ₊ → ℝ₊ satisfying the given functional equation. -/
def FunctionalEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x > 0 → f y > 0 → f (f x + y) = α * x + 1 / f (1 / y)

/-- The main theorem stating the solution to the functional equation. -/
theorem functional_equation_solution :
  ∀ α : ℝ, α ≠ 0 →
    (∃ f : ℝ → ℝ, FunctionalEquation f α) ↔ (α = 1 ∧ ∃ f : ℝ → ℝ, FunctionalEquation f 1 ∧ ∀ x, x > 0 → f x = x) :=
by sorry

end functional_equation_solution_l1962_196283


namespace investment_in_bank_a_l1962_196207

def total_investment : ℝ := 1500
def bank_a_rate : ℝ := 0.04
def bank_b_rate : ℝ := 0.06
def years : ℕ := 3
def final_amount : ℝ := 1740.54

theorem investment_in_bank_a (x : ℝ) :
  x * (1 + bank_a_rate) ^ years + (total_investment - x) * (1 + bank_b_rate) ^ years = final_amount →
  x = 695 := by
sorry

end investment_in_bank_a_l1962_196207


namespace arithmetic_computation_l1962_196257

theorem arithmetic_computation : -9 * 3 - (-7 * -4) + (-11 * -6) = 11 := by
  sorry

end arithmetic_computation_l1962_196257


namespace special_function_properties_l1962_196297

/-- A function satisfying the given properties -/
structure SpecialFunction where
  f : ℝ → ℝ
  property : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y
  zero_map : f 0 = 0
  pi_half_map : f (Real.pi / 2) = 1

/-- The function is odd -/
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- The function is periodic with period 2π -/
def is_periodic_2pi (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

/-- Main theorem: The special function is odd and periodic with period 2π -/
theorem special_function_properties (sf : SpecialFunction) :
    is_odd sf.f ∧ is_periodic_2pi sf.f := by
  sorry

end special_function_properties_l1962_196297


namespace eva_is_speed_skater_l1962_196202

-- Define the people and sports
inductive Person : Type
| Ben : Person
| Filip : Person
| Eva : Person
| Andrea : Person

inductive Sport : Type
| SpeedSkating : Sport
| Skiing : Sport
| Hockey : Sport
| Snowboarding : Sport

-- Define the positions at the table
inductive Position : Type
| Top : Position
| Right : Position
| Bottom : Position
| Left : Position

-- Define the seating arrangement
def SeatingArrangement : Type := Person → Position

-- Define the sport assignment
def SportAssignment : Type := Person → Sport

-- Define the conditions
def Conditions (seating : SeatingArrangement) (sports : SportAssignment) : Prop :=
  ∃ (skier hockey_player : Person),
    -- The skier sat at Andrea's left hand
    seating Person.Andrea = Position.Top ∧ seating skier = Position.Left
    -- The speed skater sat opposite Ben
    ∧ seating Person.Ben = Position.Left
    ∧ sports Person.Ben ≠ Sport.SpeedSkating
    -- Eva and Filip sat next to each other
    ∧ (seating Person.Eva = Position.Right ∧ seating Person.Filip = Position.Bottom
    ∨ seating Person.Eva = Position.Bottom ∧ seating Person.Filip = Position.Right)
    -- A woman sat at the hockey player's left hand
    ∧ ((seating hockey_player = Position.Right ∧ seating Person.Andrea = Position.Top)
    ∨ (seating hockey_player = Position.Bottom ∧ seating Person.Eva = Position.Right))

-- The theorem to prove
theorem eva_is_speed_skater (seating : SeatingArrangement) (sports : SportAssignment) :
  Conditions seating sports → sports Person.Eva = Sport.SpeedSkating :=
sorry

end eva_is_speed_skater_l1962_196202


namespace arithmetic_sequence_problem_l1962_196269

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) 
    (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 9 - a 10 = 24 := by
  sorry

end arithmetic_sequence_problem_l1962_196269


namespace christmas_decorations_l1962_196200

theorem christmas_decorations (boxes : ℕ) (used : ℕ) (given_away : ℕ) : 
  boxes = 4 → used = 35 → given_away = 25 → (used + given_away) / boxes = 15 := by
  sorry

end christmas_decorations_l1962_196200


namespace harriet_miles_run_l1962_196223

theorem harriet_miles_run (total_miles : ℝ) (katarina_miles : ℝ) (adriana_miles : ℝ) 
  (h1 : total_miles = 285)
  (h2 : katarina_miles = 51)
  (h3 : adriana_miles = 74)
  (h4 : ∃ (x : ℝ), x * 3 + katarina_miles + adriana_miles = total_miles) :
  ∃ (harriet_miles : ℝ), harriet_miles = 53.33 ∧ 
    harriet_miles * 3 + katarina_miles + adriana_miles = total_miles :=
by
  sorry

end harriet_miles_run_l1962_196223


namespace symmetry_axis_l1962_196206

-- Define a function g with the given symmetry property
def g : ℝ → ℝ := sorry

-- State the symmetry property of g
axiom g_symmetry : ∀ x : ℝ, g x = g (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) g := by sorry

end symmetry_axis_l1962_196206


namespace expression_evaluation_l1962_196247

theorem expression_evaluation : (2^8 + 4^5) * (1^3 - (-1)^3)^2 = 5120 := by
  sorry

end expression_evaluation_l1962_196247


namespace arithmetic_sequence_a7_l1962_196236

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a)
    (h_a1 : a 1 = 2)
    (h_sum : a 3 + a 5 = 8) :
  a 7 = 6 := by
  sorry

end arithmetic_sequence_a7_l1962_196236


namespace calculate_expression_l1962_196250

theorem calculate_expression : (-2022)^0 - 2 * Real.tan (π/4) + |-2| + Real.sqrt 9 = 4 := by
  sorry

end calculate_expression_l1962_196250


namespace divisible_by_35_l1962_196216

theorem divisible_by_35 (n : ℕ) : ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35 * k := by
  sorry

end divisible_by_35_l1962_196216


namespace mean_temperature_l1962_196255

def temperatures : List ℤ := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -2 := by
  sorry

end mean_temperature_l1962_196255


namespace percent_of_x_l1962_196212

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 10 + x / 25) / x * 100 = 14 := by
  sorry

end percent_of_x_l1962_196212


namespace tangent_slope_at_point_one_l1962_196252

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = -3 ∧ f' 1 = -5 := by sorry

end tangent_slope_at_point_one_l1962_196252


namespace frog_climb_time_l1962_196264

/-- Represents the frog's climbing scenario -/
structure FrogClimb where
  wellDepth : ℝ
  climbUp : ℝ
  slipDown : ℝ
  slipTime : ℝ
  timeAt3mBelow : ℝ

/-- Calculates the time taken for the frog to reach the top of the well -/
def timeToReachTop (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to reach the top -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.wellDepth = 12)
  (h2 : f.climbUp = 3)
  (h3 : f.slipDown = 1)
  (h4 : f.slipTime = f.climbUp / 3)
  (h5 : f.timeAt3mBelow = 17) :
  timeToReachTop f = 22 :=
sorry

end frog_climb_time_l1962_196264


namespace base_7_addition_l1962_196292

/-- Addition in base 7 --/
def add_base_7 (a b : Nat) : Nat :=
  sorry

/-- Conversion from base 10 to base 7 --/
def to_base_7 (n : Nat) : Nat :=
  sorry

/-- Conversion from base 7 to base 10 --/
def from_base_7 (n : Nat) : Nat :=
  sorry

theorem base_7_addition :
  add_base_7 (from_base_7 25) (from_base_7 54) = from_base_7 112 :=
by sorry

end base_7_addition_l1962_196292


namespace f_increasing_on_interval_l1962_196282

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 5

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x < y ∧ y ≤ 1 → f x < f y :=
sorry

end f_increasing_on_interval_l1962_196282


namespace consecutive_product_problem_l1962_196270

theorem consecutive_product_problem :
  let n : ℕ := 77
  let product := n * (n + 1) * (n + 2)
  (product ≥ 100000 ∧ product < 1000000) ∧  -- six-digit number
  (product / 10000 = 47) ∧                  -- left-hand digits are '47'
  (product % 100 = 74)                      -- right-hand digits are '74'
  :=
by sorry

end consecutive_product_problem_l1962_196270


namespace sector_area_ratio_l1962_196243

/-- Given a circular sector AOB with central angle α (in radians),
    and a line drawn through point B and the midpoint C of radius OA,
    the ratio of the area of triangle BCO to the area of figure ABC
    is sin(α) / (2α - sin(α)). -/
theorem sector_area_ratio (α : Real) :
  let R : Real := 1  -- Assume unit radius for simplicity
  let S : Real := (1/2) * R^2 * α  -- Area of sector AOB
  let S_BCO : Real := (1/4) * R^2 * Real.sin α  -- Area of triangle BCO
  let S_ABC : Real := S - S_BCO  -- Area of figure ABC
  S_BCO / S_ABC = Real.sin α / (2 * α - Real.sin α) := by
sorry

end sector_area_ratio_l1962_196243


namespace paint_needed_for_columns_l1962_196233

-- Define constants
def num_columns : ℕ := 20
def column_height : ℝ := 20
def column_diameter : ℝ := 10
def paint_coverage : ℝ := 350

-- Theorem statement
theorem paint_needed_for_columns :
  ∃ (gallons : ℕ),
    gallons * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) ∧
    ∀ (g : ℕ), g * paint_coverage ≥ num_columns * (2 * Real.pi * (column_diameter / 2) * column_height) → g ≥ gallons :=
by sorry

end paint_needed_for_columns_l1962_196233


namespace train_crossing_time_l1962_196276

/-- Calculates the time taken for a train to cross a signal post -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 350 → 
  train_speed_kmph = 72 → 
  (train_length / (train_speed_kmph * 1000 / 3600)) = 17.5 := by
  sorry

end train_crossing_time_l1962_196276


namespace games_attended_l1962_196295

def total_games : ℕ := 39
def missed_games : ℕ := 25

theorem games_attended : total_games - missed_games = 14 := by
  sorry

end games_attended_l1962_196295


namespace product_sequence_sum_l1962_196241

theorem product_sequence_sum (a b : ℕ) : 
  (a : ℚ) / 3 = 16 → b = a - 1 → a + b = 95 := by sorry

end product_sequence_sum_l1962_196241


namespace f_increasing_on_negative_l1962_196267

def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem f_increasing_on_negative (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  ∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f m x < f m y :=
sorry

end f_increasing_on_negative_l1962_196267


namespace scientific_notation_450_million_l1962_196203

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by sorry

end scientific_notation_450_million_l1962_196203


namespace coefficient_of_negative_six_xy_l1962_196226

/-- The coefficient of a monomial is the numeric factor that multiplies the variable parts. -/
def coefficient (m : ℤ) (x : String) (y : String) : ℤ := m

theorem coefficient_of_negative_six_xy :
  coefficient (-6) "x" "y" = -6 := by sorry

end coefficient_of_negative_six_xy_l1962_196226


namespace exists_good_pair_for_all_constructed_pair_is_good_l1962_196286

/-- A pair of natural numbers (m, n) is good if mn and (m+1)(n+1) are perfect squares -/
def is_good_pair (m n : ℕ) : Prop :=
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2

/-- For every natural number m, there exists a good pair (m, n) with n > m -/
theorem exists_good_pair_for_all (m : ℕ) : ∃ n : ℕ, n > m ∧ is_good_pair m n := by
  sorry

/-- The constructed pair (m, m(4m + 3)²) is good for any natural number m -/
theorem constructed_pair_is_good (m : ℕ) : is_good_pair m (m * (4 * m + 3) ^ 2) := by
  sorry

end exists_good_pair_for_all_constructed_pair_is_good_l1962_196286


namespace tom_payment_l1962_196277

/-- The total amount Tom paid to the shopkeeper -/
def total_amount (apple_quantity apple_rate mango_quantity mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1145 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 65 = 1145 := by
  sorry

end tom_payment_l1962_196277


namespace binomial_sum_equals_higher_binomial_l1962_196211

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem
theorem binomial_sum_equals_higher_binomial :
  binomial 6 3 + binomial 6 2 = binomial 7 3 := by
  sorry

end binomial_sum_equals_higher_binomial_l1962_196211


namespace sqrt_65_bound_l1962_196244

theorem sqrt_65_bound (n : ℕ+) (h : (n : ℝ) < Real.sqrt 65 ∧ Real.sqrt 65 < (n : ℝ) + 1) : n = 8 := by
  sorry

end sqrt_65_bound_l1962_196244


namespace lower_variance_more_stable_l1962_196246

/-- Represents a set of data -/
structure DataSet where
  variance : ℝ

/-- Defines the stability relation between two data sets -/
def more_stable (a b : DataSet) : Prop := a.variance < b.variance

/-- Theorem stating that a data set with lower variance is more stable -/
theorem lower_variance_more_stable (A B : DataSet) 
  (hA : A.variance = 0.01) (hB : B.variance = 0.1) : 
  more_stable A B := by
  sorry

#check lower_variance_more_stable

end lower_variance_more_stable_l1962_196246


namespace barge_length_is_125_steps_l1962_196234

/-- Represents the scenario of Jake walking along a barge on a river -/
structure BargeProblem where
  -- Length of Jake's step upstream
  step_length : ℝ
  -- Length the barge moves while Jake takes one step
  barge_speed : ℝ
  -- Length of the barge
  barge_length : ℝ
  -- Jake walks faster than the barge
  jake_faster : barge_speed < step_length
  -- 300 steps downstream from back to front
  downstream_eq : 300 * (1.5 * step_length) = barge_length + 300 * barge_speed
  -- 60 steps upstream from front to back
  upstream_eq : 60 * step_length = barge_length - 60 * barge_speed

/-- The length of the barge is 125 times Jake's upstream step length -/
theorem barge_length_is_125_steps (p : BargeProblem) : p.barge_length = 125 * p.step_length := by
  sorry


end barge_length_is_125_steps_l1962_196234


namespace instrument_players_l1962_196284

theorem instrument_players (total_people : ℕ) 
  (at_least_one_ratio : ℚ) (exactly_one_prob : ℝ) 
  (h1 : total_people = 800)
  (h2 : at_least_one_ratio = 1/5)
  (h3 : exactly_one_prob = 0.12) : 
  ℕ := by
  sorry

#check instrument_players

end instrument_players_l1962_196284


namespace average_of_remaining_numbers_l1962_196215

theorem average_of_remaining_numbers 
  (n : ℕ) 
  (total_avg : ℚ) 
  (subset_sum : ℚ) 
  (h1 : n = 5) 
  (h2 : total_avg = 20) 
  (h3 : subset_sum = 48) : 
  ((n : ℚ) * total_avg - subset_sum) / ((n : ℚ) - 3) = 26 := by
sorry

end average_of_remaining_numbers_l1962_196215


namespace sebastians_age_l1962_196298

theorem sebastians_age (sebastian_age sister_age father_age : ℕ) : 
  (sebastian_age - 5) + (sister_age - 5) = 3 * (father_age - 5) / 4 →
  sebastian_age = sister_age + 10 →
  father_age = 85 →
  sebastian_age = 40 := by
sorry

end sebastians_age_l1962_196298


namespace second_draw_probability_l1962_196240

/-- Represents the probability of drawing a red sweet in the second draw -/
def probability_second_red (x y : ℕ) : ℚ :=
  y / (x + y)

/-- Theorem stating that the probability of drawing a red sweet in the second draw
    is equal to the initial ratio of red sweets to total sweets -/
theorem second_draw_probability (x y : ℕ) (hxy : x + y > 0) :
  probability_second_red x y = y / (x + y) := by
  sorry

end second_draw_probability_l1962_196240


namespace boat_speed_in_still_water_l1962_196249

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : ∃ (x : ℝ),
  (x + 3) * (24 / 60) = 7.2 ∧ x = 15 := by
  sorry

end boat_speed_in_still_water_l1962_196249


namespace factorization_equality_l1962_196293

theorem factorization_equality (a m n : ℝ) :
  -3 * a * m^2 + 12 * a * n^2 = -3 * a * (m + 2*n) * (m - 2*n) := by
  sorry

end factorization_equality_l1962_196293


namespace cos_225_degrees_l1962_196254

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l1962_196254


namespace commercial_reduction_percentage_l1962_196232

theorem commercial_reduction_percentage 
  (original_length : ℝ) 
  (shortened_length : ℝ) 
  (h1 : original_length = 30) 
  (h2 : shortened_length = 21) : 
  (original_length - shortened_length) / original_length * 100 = 30 := by
sorry

end commercial_reduction_percentage_l1962_196232


namespace subtract_preserves_inequality_l1962_196227

theorem subtract_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end subtract_preserves_inequality_l1962_196227


namespace plate_arrangement_theorem_l1962_196210

def blue_plates : ℕ := 6
def red_plates : ℕ := 3
def green_plates : ℕ := 2
def yellow_plates : ℕ := 2

def total_plates : ℕ := blue_plates + red_plates + green_plates + yellow_plates

def circular_arrangements (n : ℕ) (k : List ℕ) : ℕ :=
  Nat.factorial (n - 1) / (k.map Nat.factorial).prod

theorem plate_arrangement_theorem :
  let total_arrangements := circular_arrangements total_plates [blue_plates, red_plates, green_plates, yellow_plates]
  let green_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, 1, yellow_plates]
  let yellow_adjacent := circular_arrangements (total_plates - 1) [blue_plates, red_plates, green_plates, 1]
  let both_adjacent := circular_arrangements (total_plates - 2) [blue_plates, red_plates, 1, 1]
  total_arrangements - green_adjacent - yellow_adjacent + both_adjacent = 50400 := by
  sorry

end plate_arrangement_theorem_l1962_196210


namespace cubic_roots_sum_l1962_196214

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 25*a - 10 = 0 →
  b^3 - 15*b^2 + 25*b - 10 = 0 →
  c^3 - 15*c^2 + 25*c - 10 = 0 →
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 175/11 := by
sorry

end cubic_roots_sum_l1962_196214


namespace constant_function_shifted_l1962_196258

-- Define g as a function from real numbers to real numbers
def g : ℝ → ℝ := fun _ ↦ -3

-- Theorem statement
theorem constant_function_shifted (x : ℝ) : g (x - 5) = -3 := by
  sorry

end constant_function_shifted_l1962_196258


namespace correct_purchase_ways_l1962_196248

def num_cookie_types : ℕ := 7
def num_cupcake_types : ℕ := 4
def total_items : ℕ := 4

def purchase_ways : ℕ := sorry

theorem correct_purchase_ways : purchase_ways = 4054 := by sorry

end correct_purchase_ways_l1962_196248


namespace natasha_hill_climbing_l1962_196217

/-- Natasha's hill climbing problem -/
theorem natasha_hill_climbing
  (time_up : ℝ)
  (time_down : ℝ)
  (avg_speed_total : ℝ)
  (h_time_up : time_up = 4)
  (h_time_down : time_down = 2)
  (h_avg_speed_total : avg_speed_total = 1.5) :
  let distance := avg_speed_total * (time_up + time_down) / 2
  let avg_speed_up := distance / time_up
  avg_speed_up = 1.125 := by
sorry

end natasha_hill_climbing_l1962_196217


namespace preceding_binary_l1962_196296

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (binary : List Bool) : ℕ :=
  binary.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

theorem preceding_binary (N : ℕ) (h : binaryToNat [true, true, false, false, false] = N) :
  natToBinary (N - 1) = [true, false, true, true, true] := by
  sorry

end preceding_binary_l1962_196296


namespace seven_division_theorem_l1962_196266

/-- Given a natural number n, returns the sum of its digits. -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Given a natural number n, returns the number of digits in n. -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number consists only of the digit 7. -/
def all_sevens (n : ℕ) : Prop := sorry

theorem seven_division_theorem (N : ℕ) :
  digit_sum N = 2021 →
  ∃ q : ℕ, N = 7 * q ∧ all_sevens q →
  num_digits q = 503 := by sorry

end seven_division_theorem_l1962_196266


namespace decimal_representation_theorem_l1962_196201

theorem decimal_representation_theorem (n m : ℕ) (h1 : n > m) (h2 : m ≥ 1) 
  (h3 : ∃ k : ℕ, ∃ p : ℕ, 0 < p ∧ p < n ∧ 
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 ≥ 143 ∧
    (((10^k : ℚ) * (m : ℚ) / (n : ℚ)) - ((10^k : ℚ) * (m : ℚ) / (n : ℚ)).floor) * 1000 < 144) :
  n > 125 := by
  sorry

end decimal_representation_theorem_l1962_196201


namespace min_value_expression_l1962_196213

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = (1 / 2 : ℝ) ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 →
    (1 / a) - (4 * b / (b + 1)) ≥ min :=
by sorry

end min_value_expression_l1962_196213


namespace unique_line_configuration_l1962_196273

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The specific configuration described in the problem -/
def problem_config : LineConfiguration :=
  { n := 8,  -- This is what we want to prove
    total_intersections := 16,
    triple_intersections := 6 }

/-- Theorem stating that the problem configuration is the only valid one -/
theorem unique_line_configuration :
  ∀ (config : LineConfiguration),
    (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) →  -- every pair of lines intersects
    (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n → 
      i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
      ¬∃ (p : ℕ), p < config.total_intersections) →  -- no four lines pass through a single point
    config.total_intersections = 16 →
    config.triple_intersections = 6 →
    config = problem_config :=
by sorry

end unique_line_configuration_l1962_196273


namespace shopkeeper_weight_problem_l1962_196229

theorem shopkeeper_weight_problem (actual_weight : ℝ) (profit_percentage : ℝ) :
  actual_weight = 800 →
  profit_percentage = 25 →
  ∃ standard_weight : ℝ,
    standard_weight = 1000 ∧
    (standard_weight - actual_weight) / actual_weight * 100 = profit_percentage :=
by sorry

end shopkeeper_weight_problem_l1962_196229


namespace geometric_sequence_sum_l1962_196224

/-- Given a geometric sequence {a_n} where a_3 + a_7 = 5, 
    prove that a_2a_4 + 2a_4a_6 + a_6a_8 = 25 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum : a 3 + a 7 = 5) :
    a 2 * a 4 + 2 * a 4 * a 6 + a 6 * a 8 = 25 := by
  sorry

end geometric_sequence_sum_l1962_196224


namespace final_amount_calculation_l1962_196205

-- Define the variables
def initial_amount : ℕ := 45
def amount_spent : ℕ := 20
def additional_amount : ℕ := 46

-- Define the theorem
theorem final_amount_calculation :
  initial_amount - amount_spent + additional_amount = 71 := by
  sorry

end final_amount_calculation_l1962_196205


namespace monotone_decreasing_cubic_l1962_196278

/-- A function f is monotonically decreasing on an open interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem monotone_decreasing_cubic (a : ℝ) :
  MonotonicallyDecreasing (fun x => x^3 - a*x^2 + 1) 0 2 → a ≥ 3 := by
  sorry

end monotone_decreasing_cubic_l1962_196278


namespace expression_always_positive_l1962_196253

theorem expression_always_positive (x : ℝ) : (x - 3) * (x - 5) + 2 > 0 := by
  sorry

end expression_always_positive_l1962_196253


namespace quadratic_root_implies_k_l1962_196289

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x - k = 0 ∧ x = 1) → k = -2 := by
  sorry

end quadratic_root_implies_k_l1962_196289


namespace function_composition_equality_l1962_196261

theorem function_composition_equality (b : ℚ) : 
  let p : ℚ → ℚ := λ x => 3 * x - 5
  let q : ℚ → ℚ := λ x => 4 * x - b
  p (q 3) = 9 → b = 22 / 3 := by
sorry

end function_composition_equality_l1962_196261


namespace field_length_calculation_l1962_196256

theorem field_length_calculation (width : ℝ) (pond_side : ℝ) : 
  width > 0 →
  pond_side = 4 →
  2 * width * width = 8 * (pond_side * pond_side) →
  2 * width = 16 := by
sorry

end field_length_calculation_l1962_196256


namespace a_5_equals_5_l1962_196294

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  h1 : a 3 + a 11 = 18  -- Condition 1
  h2 : (a 1 + a 2 + a 3) = -3  -- Condition 2 (S₃ = -3)

/-- The theorem stating that a₅ = 5 for the given arithmetic sequence -/
theorem a_5_equals_5 (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end a_5_equals_5_l1962_196294


namespace probability_three_black_balls_l1962_196208

-- Define the number of white and black balls
def white_balls : ℕ := 4
def black_balls : ℕ := 8

-- Define the total number of balls
def total_balls : ℕ := white_balls + black_balls

-- Define the number of balls drawn
def drawn_balls : ℕ := 3

-- Define the probability function
def probability_all_black : ℚ :=
  (Nat.choose black_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ)

-- State the theorem
theorem probability_three_black_balls :
  probability_all_black = 14 / 55 := by
  sorry

end probability_three_black_balls_l1962_196208


namespace sqrt_42_minus_1_range_l1962_196221

theorem sqrt_42_minus_1_range : 5 < Real.sqrt 42 - 1 ∧ Real.sqrt 42 - 1 < 6 := by
  have h1 : 36 < 42 := by sorry
  have h2 : 42 < 49 := by sorry
  have h3 : Real.sqrt 36 = 6 := by sorry
  have h4 : Real.sqrt 49 = 7 := by sorry
  sorry

end sqrt_42_minus_1_range_l1962_196221


namespace first_pay_cut_percentage_l1962_196251

theorem first_pay_cut_percentage 
  (overall_decrease : Real) 
  (second_cut : Real) 
  (third_cut : Real) 
  (h1 : overall_decrease = 27.325)
  (h2 : second_cut = 10)
  (h3 : third_cut = 15) : 
  ∃ (first_cut : Real), 
    first_cut = 5 ∧ 
    (1 - overall_decrease / 100) = 
    (1 - first_cut / 100) * (1 - second_cut / 100) * (1 - third_cut / 100) := by
  sorry


end first_pay_cut_percentage_l1962_196251


namespace complex_roots_cubic_l1962_196239

theorem complex_roots_cubic (a b c : ℂ) 
  (h1 : a + b + c = 1)
  (h2 : a * b + a * c + b * c = 0)
  (h3 : a * b * c = -1) :
  (∀ x : ℂ, x^3 - x^2 + 1 = 0 ↔ x = a ∨ x = b ∨ x = c) :=
by sorry

end complex_roots_cubic_l1962_196239


namespace inequality_implies_sum_l1962_196235

/-- Given that (x-a)(x-b)/(x-c) ≤ 0 if and only if x < -6 or |x-30| ≤ 2, and a < b,
    prove that a + 2b + 3c = 74 -/
theorem inequality_implies_sum (a b c : ℝ) :
  (∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  a < b →
  a + 2*b + 3*c = 74 := by
sorry

end inequality_implies_sum_l1962_196235


namespace equation_proof_l1962_196230

theorem equation_proof : 3889 + 12.808 - 47.806 = 3854.002 := by
  sorry

end equation_proof_l1962_196230


namespace complement_of_A_in_S_l1962_196218

def S : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}

theorem complement_of_A_in_S :
  (S \ A) = {x : ℝ | x < -1 ∨ x > 3} := by sorry

end complement_of_A_in_S_l1962_196218


namespace equation_solution_l1962_196242

theorem equation_solution : ∃! x : ℝ, (x + 1)^63 + (x + 1)^62*(x - 1) + (x + 1)^61*(x - 1)^2 + (x - 1)^63 = 0 ∧ x = 0 := by
  sorry

end equation_solution_l1962_196242


namespace eliza_walking_distance_l1962_196204

/-- Proves that Eliza walked 4.5 kilometers given the conditions of the problem -/
theorem eliza_walking_distance :
  ∀ (total_time : ℝ) (rollerblade_speed : ℝ) (walk_speed : ℝ) (distance : ℝ),
    total_time = 1.5 →  -- 90 minutes converted to hours
    rollerblade_speed = 12 →
    walk_speed = 4 →
    (distance / rollerblade_speed) + (distance / walk_speed) = total_time →
    distance = 4.5 := by
  sorry

#check eliza_walking_distance

end eliza_walking_distance_l1962_196204


namespace ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l1962_196279

-- Define the ellipse equations
def ellipse1 (x y : ℝ) : Prop := x^2 / 144 + y^2 / 169 = 1
def ellipse2 (x y m : ℝ) : Prop := x^2 / m^2 + y^2 / (m^2 + 1) = 1
def ellipse3 (x y : ℝ) : Prop := x^2 / 16 + y^2 / 7 = 1
def ellipse4 (x y m : ℝ) : Prop := x^2 / (m - 5) + y^2 / (m + 4) = 1

-- Define focal points
def focal_points (a b : ℝ) : Set (ℝ × ℝ) := {(-a, 0), (a, 0)} ∪ {(0, -b), (0, b)}

-- Theorem statements
theorem ellipse1_focal_points :
  ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 5 ∧ 
  ∀ (x y : ℝ), ellipse1 x y → (x, y) ∈ f := sorry

theorem ellipse2_focal_points :
  ∀ (m : ℝ), ∃ (f : Set (ℝ × ℝ)), f = focal_points 0 1 ∧ 
  ∀ (x y : ℝ), ellipse2 x y m → (x, y) ∈ f := sorry

theorem ellipses_different_focal_points :
  ∀ (m : ℝ), m > 0 →
  ¬∃ (f : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), ellipse3 x y → (x, y) ∈ f) ∧
    (∀ (x y : ℝ), ellipse4 x y m → (x, y) ∈ f) := sorry

end ellipse1_focal_points_ellipse2_focal_points_ellipses_different_focal_points_l1962_196279


namespace barbara_candies_l1962_196285

theorem barbara_candies (original_boxes : Nat) (original_candies_per_box : Nat)
                         (new_boxes : Nat) (new_candies_per_box : Nat) :
  original_boxes = 9 →
  original_candies_per_box = 25 →
  new_boxes = 18 →
  new_candies_per_box = 35 →
  original_boxes * original_candies_per_box + new_boxes * new_candies_per_box = 855 :=
by sorry

end barbara_candies_l1962_196285


namespace unique_three_digit_number_l1962_196288

/-- A three-digit number is represented by its digits a, b, and c. -/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- The product of the digits of a three-digit number. -/
def digit_product (a b c : ℕ) : ℕ := a * b * c

/-- Predicate for a valid three-digit number. -/
def is_valid_three_digit (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- The main theorem: 175 is the only three-digit number that is 5 times the product of its digits. -/
theorem unique_three_digit_number :
  ∀ a b c : ℕ,
    is_valid_three_digit a b c →
    (three_digit_number a b c = 5 * digit_product a b c) →
    (a = 1 ∧ b = 7 ∧ c = 5) :=
by sorry

end unique_three_digit_number_l1962_196288


namespace speakers_cost_calculation_l1962_196274

/-- The amount spent on speakers, given the total amount spent on car parts and the amount spent on new tires. -/
def amount_spent_on_speakers (total_spent : ℚ) (tires_cost : ℚ) : ℚ :=
  total_spent - tires_cost

/-- Theorem stating that the amount spent on speakers is $118.54, given the total spent and the cost of tires. -/
theorem speakers_cost_calculation (total_spent tires_cost : ℚ) 
  (h1 : total_spent = 224.87)
  (h2 : tires_cost = 106.33) : 
  amount_spent_on_speakers total_spent tires_cost = 118.54 := by
  sorry

#eval amount_spent_on_speakers 224.87 106.33

end speakers_cost_calculation_l1962_196274


namespace e_sequence_property_l1962_196275

/-- Definition of an E-sequence -/
def is_e_sequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → |a (k + 1) - a k| = 1

/-- The sequence is increasing -/
def is_increasing (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ k, k < n - 1 → a k < a (k + 1)

theorem e_sequence_property (a : ℕ → ℤ) :
  is_e_sequence a 2000 →
  a 1 = 13 →
  (is_increasing a 2000 ↔ a 2000 = 2012) :=
by sorry

end e_sequence_property_l1962_196275


namespace smallest_n_for_candy_l1962_196268

theorem smallest_n_for_candy (n : ℕ) : (∀ k : ℕ, k > 0 ∧ k < n → ¬(10 ∣ 25*k ∧ 18 ∣ 25*k ∧ 20 ∣ 25*k)) ∧ 
                                       (10 ∣ 25*n ∧ 18 ∣ 25*n ∧ 20 ∣ 25*n) → 
                                       n = 16 :=
sorry

end smallest_n_for_candy_l1962_196268


namespace probability_allison_wins_l1962_196225

structure Cube where
  faces : List ℕ
  valid : faces.length = 6

def allison_cube : Cube := ⟨List.replicate 6 5, rfl⟩
def brian_cube : Cube := ⟨[1, 2, 3, 4, 5, 6], rfl⟩
def noah_cube : Cube := ⟨[2, 2, 2, 6, 6, 6], rfl⟩

def prob_roll_less_than (n : ℕ) (c : Cube) : ℚ :=
  (c.faces.filter (· < n)).length / c.faces.length

theorem probability_allison_wins : 
  prob_roll_less_than 5 brian_cube * prob_roll_less_than 5 noah_cube = 1/3 := by
  sorry

end probability_allison_wins_l1962_196225
