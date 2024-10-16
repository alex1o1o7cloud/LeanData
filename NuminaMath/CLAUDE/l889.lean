import Mathlib

namespace NUMINAMATH_CALUDE_unique_line_through_5_2_l889_88944

/-- A line in the xy-plane is represented by its x and y intercepts -/
structure Line where
  x_intercept : ℕ
  y_intercept : ℕ

/-- Check if a number is prime -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Check if a number is a power of 2 -/
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

/-- Check if a line passes through the point (5,2) -/
def passes_through_5_2 (l : Line) : Prop :=
  5 / l.x_intercept + 2 / l.y_intercept = 1

/-- The main theorem to be proved -/
theorem unique_line_through_5_2 : 
  ∃! l : Line, 
    is_prime l.x_intercept ∧ 
    is_power_of_two l.y_intercept ∧ 
    passes_through_5_2 l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_5_2_l889_88944


namespace NUMINAMATH_CALUDE_fraction_problem_l889_88948

theorem fraction_problem (x : ℚ) : 
  x / (4 * x + 5) = 3 / 7 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l889_88948


namespace NUMINAMATH_CALUDE_tangerines_left_l889_88981

theorem tangerines_left (total : ℕ) (given : ℕ) (h1 : total = 27) (h2 : given = 18) :
  total - given = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l889_88981


namespace NUMINAMATH_CALUDE_vertical_multiplication_puzzle_l889_88915

theorem vertical_multiplication_puzzle :
  ∀ a b : ℕ,
    10 < a ∧ a < 20 →
    10 < b ∧ b < 20 →
    100 ≤ a * b ∧ a * b < 1000 →
    (a * b) / 100 = 2 →
    a * b % 10 = 7 →
    (a = 13 ∧ b = 19) ∨ (a = 19 ∧ b = 13) :=
by sorry

end NUMINAMATH_CALUDE_vertical_multiplication_puzzle_l889_88915


namespace NUMINAMATH_CALUDE_wage_problem_solution_l889_88973

/-- Represents daily wages -/
structure DailyWage where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a sum of money -/
def SumOfMoney : Type := ℝ

/-- Given conditions of the problem -/
structure WageProblem where
  S : SumOfMoney
  B : DailyWage
  C : DailyWage
  S_pays_C_24_days : S = 24 * C.amount
  S_pays_both_8_days : S = 8 * (B.amount + C.amount)

/-- The theorem to prove -/
theorem wage_problem_solution (p : WageProblem) : 
  p.S = 12 * p.B.amount := by sorry

end NUMINAMATH_CALUDE_wage_problem_solution_l889_88973


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l889_88994

theorem isosceles_triangle_condition 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : a = 2 * b * Real.cos C)
  (h6 : a > 0 ∧ b > 0 ∧ c > 0)
  : B = C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l889_88994


namespace NUMINAMATH_CALUDE_tax_base_amount_l889_88962

/-- Proves that given a tax rate of 82% and a tax amount of $82, the base amount is $100. -/
theorem tax_base_amount (tax_rate : ℝ) (tax_amount : ℝ) (base_amount : ℝ) : 
  tax_rate = 82 ∧ tax_amount = 82 → base_amount = 100 := by
  sorry

end NUMINAMATH_CALUDE_tax_base_amount_l889_88962


namespace NUMINAMATH_CALUDE_unique_solution_system_l889_88992

theorem unique_solution_system (x : ℝ) : 
  (3 * x^2 + 8 * x - 3 = 0 ∧ 3 * x^4 + 2 * x^3 - 10 * x^2 + 30 * x - 9 = 0) ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l889_88992


namespace NUMINAMATH_CALUDE_sqrt_two_subset_P_l889_88957

def P : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem sqrt_two_subset_P : {Real.sqrt 2} ⊆ P := by sorry

end NUMINAMATH_CALUDE_sqrt_two_subset_P_l889_88957


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l889_88914

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 - 2*x + b > 0 ↔ -3 < x ∧ x < 1) →
  (a = -1 ∧ b = 3 ∧ 
   ∀ x, 3*x^2 - x - 2 ≤ 0 ↔ -2/3 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l889_88914


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l889_88924

/-- The equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is equivalent to "If the equation x^2 + x - m = 0 does not have real roots, then m ≤ 0" -/
theorem contrapositive_equivalence : 
  (¬(has_real_roots m) → m ≤ 0) ↔ (m > 0 → has_real_roots m) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l889_88924


namespace NUMINAMATH_CALUDE_football_tickets_problem_l889_88958

/-- Given a ticket price and budget, calculates the maximum number of tickets that can be purchased. -/
def max_tickets (price : ℕ) (budget : ℕ) : ℕ :=
  (budget / price : ℕ)

/-- Proves that given a ticket price of 15 and a budget of 120, the maximum number of tickets that can be purchased is 8. -/
theorem football_tickets_problem :
  max_tickets 15 120 = 8 := by
  sorry

end NUMINAMATH_CALUDE_football_tickets_problem_l889_88958


namespace NUMINAMATH_CALUDE_cubic_sum_l889_88977

theorem cubic_sum (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c →
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_l889_88977


namespace NUMINAMATH_CALUDE_limit_rational_function_l889_88906

/-- The limit of (x^3 - 3x - 2) / (x - 2) as x approaches 2 is 9 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → 
    |(x^3 - 3*x - 2) / (x - 2) - 9| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l889_88906


namespace NUMINAMATH_CALUDE_solve_a_given_set_membership_l889_88989

theorem solve_a_given_set_membership (a : ℝ) : 
  -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_a_given_set_membership_l889_88989


namespace NUMINAMATH_CALUDE_fibonacci_sequence_ones_l889_88963

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define x_n sequence
def x : ℕ → ℕ → ℕ → ℚ
| 0, k, m => fib k / fib m
| (n + 1), k, m => 
  let prev := x n k m
  if prev = 1 then 1 else (2 * prev - 1) / (1 - prev)

-- Theorem statement
theorem fibonacci_sequence_ones (k m : ℕ) (h : m > k) :
  (∃ n, x n k m = 1) ↔ (∃ i : ℕ, k = 2 * i ∧ m = 2 * i + 1) :=
sorry

end NUMINAMATH_CALUDE_fibonacci_sequence_ones_l889_88963


namespace NUMINAMATH_CALUDE_least_possible_b_value_l889_88976

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least possible value of b satisfying the given conditions -/
def least_b : ℕ := 42

theorem least_possible_b_value (a b : ℕ+) 
  (ha : num_factors a = 4)
  (hb : num_factors b = a.val)
  (hd : a.val + 1 ∣ b) :
  b ≥ least_b ∧ ∃ (b' : ℕ+), b' = least_b ∧ 
    num_factors b' = a.val ∧ 
    a.val + 1 ∣ b' := by sorry

end NUMINAMATH_CALUDE_least_possible_b_value_l889_88976


namespace NUMINAMATH_CALUDE_age_difference_proof_l889_88942

theorem age_difference_proof (jack_age bill_age : ℕ) : 
  jack_age = 3 * bill_age →
  (jack_age + 3) = 2 * (bill_age + 3) →
  jack_age - bill_age = 6 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l889_88942


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_equation_l889_88921

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle is tangent to the y-axis --/
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The equation of a circle --/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_tangent_to_y_axis_equation 
  (c : Circle) 
  (h1 : c.center = (1, 2)) 
  (h2 : is_tangent_to_y_axis c) : 
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_equation_l889_88921


namespace NUMINAMATH_CALUDE_unique_representation_l889_88961

theorem unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_representation_l889_88961


namespace NUMINAMATH_CALUDE_parallel_line_equation_l889_88912

/-- A line passing through point (2,1) and parallel to y = -3x + 2 has equation y = -3x + 7 -/
theorem parallel_line_equation :
  let point : ℝ × ℝ := (2, 1)
  let parallel_line : ℝ → ℝ := λ x => -3 * x + 2
  let line : ℝ → ℝ := λ x => -3 * x + 7
  (∀ x : ℝ, line x - parallel_line x = line 0 - parallel_line 0) ∧
  line point.1 = point.2 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l889_88912


namespace NUMINAMATH_CALUDE_pocket_money_problem_l889_88953

theorem pocket_money_problem (older_initial : ℕ) (younger_initial : ℕ) (difference : ℕ) (amount_given : ℕ) : 
  older_initial = 2800 →
  younger_initial = 1500 →
  older_initial - amount_given = younger_initial + amount_given + difference →
  difference = 360 →
  amount_given = 470 := by
  sorry

end NUMINAMATH_CALUDE_pocket_money_problem_l889_88953


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_five_l889_88933

theorem sqrt_plus_square_zero_implies_diff_five (x y : ℝ) 
  (h : Real.sqrt (x - 3) + (y + 2)^2 = 0) : x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_five_l889_88933


namespace NUMINAMATH_CALUDE_average_speed_problem_l889_88929

/-- The average speed for an hour drive, given that driving twice as fast for 4 hours covers 528 miles. -/
theorem average_speed_problem (v : ℝ) : v > 0 → 2 * v * 4 = 528 → v = 66 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_problem_l889_88929


namespace NUMINAMATH_CALUDE_joint_order_savings_l889_88968

/-- Represents the cost and discount structure for photocopies -/
structure PhotocopyOrder where
  cost_per_copy : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the total cost of an order with potential discount -/
def total_cost (order : PhotocopyOrder) (num_copies : ℕ) : ℚ :=
  let base_cost := order.cost_per_copy * num_copies
  if num_copies > order.discount_threshold then
    base_cost * (1 - order.discount_rate)
  else
    base_cost

/-- Theorem: Steve and David each save $0.40 by submitting a joint order -/
theorem joint_order_savings (steve_copies david_copies : ℕ) :
  let order := PhotocopyOrder.mk 0.02 0.25 100
  let individual_cost := total_cost order steve_copies
  let joint_copies := steve_copies + david_copies
  let joint_cost := total_cost order joint_copies
  steve_copies = 80 ∧ david_copies = 80 →
  individual_cost - (joint_cost / 2) = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_joint_order_savings_l889_88968


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l889_88947

theorem theater_ticket_sales (total_tickets : ℕ) (advanced_price door_price : ℚ) (total_revenue : ℚ) 
  (h1 : total_tickets = 800)
  (h2 : advanced_price = 14.5)
  (h3 : door_price = 22)
  (h4 : total_revenue = 16640) :
  ∃ (door_tickets : ℕ), 
    door_tickets = 672 ∧ 
    (total_tickets - door_tickets) * advanced_price + door_tickets * door_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l889_88947


namespace NUMINAMATH_CALUDE_bagel_savings_theorem_l889_88982

/-- The cost of a single bagel in dollars -/
def single_bagel_cost : ℚ := 2.25

/-- The cost of a dozen bagels in dollars -/
def dozen_bagels_cost : ℚ := 24

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The savings per bagel in cents when buying a dozen -/
def savings_per_bagel : ℚ :=
  ((single_bagel_cost * dozen - dozen_bagels_cost) / dozen) * 100

theorem bagel_savings_theorem :
  savings_per_bagel = 25 := by sorry

end NUMINAMATH_CALUDE_bagel_savings_theorem_l889_88982


namespace NUMINAMATH_CALUDE_absolute_sum_inequality_l889_88955

theorem absolute_sum_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_inequality_l889_88955


namespace NUMINAMATH_CALUDE_marathon_remainder_l889_88965

/-- Represents the length of a marathon in miles and yards -/
structure Marathon where
  miles : ℕ
  yards : ℕ

/-- Represents a distance in miles and yards -/
structure Distance where
  miles : ℕ
  yards : ℕ

def marathon_length : Marathon :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def number_of_marathons : ℕ := 5

theorem marathon_remainder (m : ℕ) (y : ℕ) 
  (h : Distance.mk m y = 
    { miles := number_of_marathons * marathon_length.miles + (number_of_marathons * marathon_length.yards) / yards_per_mile,
      yards := (number_of_marathons * marathon_length.yards) % yards_per_mile }) 
  (h_range : y < yards_per_mile) : 
  y = 165 := by
  sorry

end NUMINAMATH_CALUDE_marathon_remainder_l889_88965


namespace NUMINAMATH_CALUDE_swimming_problem_l889_88990

structure Triangle :=
  (A B C : ℝ × ℝ)

def isEquilateral (t : Triangle) : Prop :=
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = d t.B t.C ∧ d t.B t.C = d t.C t.A ∧ d t.C t.A = d t.A t.B

def isWestOf (p q : ℝ × ℝ) : Prop :=
  p.1 < q.1 ∧ p.2 = q.2

def swimmingPath (A B : ℝ × ℝ) (x y : ℕ) : Prop :=
  ∃ P : ℝ × ℝ, 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = x^2 ∧
    P.1 = B.1 + y ∧
    P.2 = B.2

theorem swimming_problem (t : Triangle) (x y : ℕ) :
  isEquilateral t →
  isWestOf t.B t.C →
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = 86 →
  swimmingPath t.A t.B x y →
  x > 0 →
  y > 0 →
  y = 6 :=
by sorry

end NUMINAMATH_CALUDE_swimming_problem_l889_88990


namespace NUMINAMATH_CALUDE_share_multiple_l889_88975

theorem share_multiple (total a b c k : ℕ) : 
  total = 585 →
  c = 260 →
  4 * a = k * b →
  4 * a = 3 * c →
  a + b + c = total →
  k = 6 :=
by sorry

end NUMINAMATH_CALUDE_share_multiple_l889_88975


namespace NUMINAMATH_CALUDE_disc_interaction_conservation_l889_88999

/-- Represents a disc with radius and angular velocity -/
structure Disc where
  radius : ℝ
  angularVelocity : ℝ

/-- Theorem: Conservation of angular momentum for two interacting discs -/
theorem disc_interaction_conservation
  (d1 d2 : Disc)
  (h_positive_radius : d1.radius > 0 ∧ d2.radius > 0)
  (h_same_material : True)  -- Placeholder for identical material property
  (h_same_thickness : True) -- Placeholder for identical thickness property
  (h_halt : True) -- Placeholder for the condition that both discs come to a halt
  : d1.angularVelocity * d1.radius ^ 3 = d2.angularVelocity * d2.radius ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_disc_interaction_conservation_l889_88999


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l889_88904

theorem triangle_isosceles_or_right_angled (α β γ : Real) :
  α + β + γ = Real.pi →
  0 < α ∧ 0 < β ∧ 0 < γ →
  Real.tan β * Real.sin γ * Real.sin γ = Real.tan γ * Real.sin β * Real.sin β →
  (β = γ) ∨ (β + γ = Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l889_88904


namespace NUMINAMATH_CALUDE_melissa_banana_count_l889_88987

/-- Calculates the final number of bananas Melissa has -/
def melissasFinalBananas (initialBananas buyMultiplier sharedBananas : ℕ) : ℕ :=
  let remainingBananas := initialBananas - sharedBananas
  let boughtBananas := buyMultiplier * remainingBananas
  remainingBananas + boughtBananas

theorem melissa_banana_count :
  melissasFinalBananas 88 3 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_melissa_banana_count_l889_88987


namespace NUMINAMATH_CALUDE_max_difference_consecutive_means_l889_88936

theorem max_difference_consecutive_means (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a < 1000 ∧ b < 1000 →
  ∃ (k : ℕ), (a + b) / 2 = 2 * k + 1 ∧ Real.sqrt (a * b) = 2 * k - 1 →
  a - b ≤ 62 := by
sorry

end NUMINAMATH_CALUDE_max_difference_consecutive_means_l889_88936


namespace NUMINAMATH_CALUDE_square_of_recurring_third_l889_88917

/-- The repeating decimal 0.333... --/
def recurring_third : ℚ := 1/3

/-- Theorem: The square of 0.333... is equal to 1/9 --/
theorem square_of_recurring_third : recurring_third ^ 2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_recurring_third_l889_88917


namespace NUMINAMATH_CALUDE_two_a_minus_a_equals_a_l889_88967

theorem two_a_minus_a_equals_a (a : ℝ) : 2 * a - a = a := by
  sorry

end NUMINAMATH_CALUDE_two_a_minus_a_equals_a_l889_88967


namespace NUMINAMATH_CALUDE_fraction_problem_l889_88900

theorem fraction_problem (x y : ℚ) : 
  (x + 2) / (y + 1) = 1 → 
  (x + 4) / (y + 2) = 1/2 → 
  x / y = 5/4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l889_88900


namespace NUMINAMATH_CALUDE_students_passing_both_subjects_l889_88946

theorem students_passing_both_subjects (total_english : ℕ) (total_math : ℕ) (diff_only_english : ℕ) :
  total_english = 30 →
  total_math = 20 →
  diff_only_english = 10 →
  ∃ (both : ℕ),
    both = 10 ∧
    total_english = both + (both + diff_only_english) ∧
    total_math = both + both :=
by sorry

end NUMINAMATH_CALUDE_students_passing_both_subjects_l889_88946


namespace NUMINAMATH_CALUDE_tangent_line_equation_l889_88991

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ m * x - y + b = 0) ∧
  (∀ (x : ℝ), (x - P.1) * (f x - P.2) ≤ m * (x - P.1)^2) ∧
  m * P.1 - P.2 + b = 0 ∧
  m = 3 ∧ b = -3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l889_88991


namespace NUMINAMATH_CALUDE_temperature_range_l889_88980

/-- Given the highest and lowest temperatures on a day, 
    prove that any temperature on that day lies within this range -/
theorem temperature_range (t : ℝ) (highest lowest : ℝ) 
  (h_highest : highest = 5)
  (h_lowest : lowest = -2)
  (h_t_le_highest : t ≤ highest)
  (h_t_ge_lowest : t ≥ lowest) : 
  -2 ≤ t ∧ t ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_range_l889_88980


namespace NUMINAMATH_CALUDE_base6_addition_proof_l889_88993

/-- Converts a base 6 number represented as a list of digits to a natural number. -/
def fromBase6 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Checks if a number is a single digit in base 6. -/
def isSingleDigitBase6 (n : Nat) : Prop := n < 6

theorem base6_addition_proof (C D : Nat) 
  (hC : isSingleDigitBase6 C) 
  (hD : isSingleDigitBase6 D) : 
  fromBase6 [1, 1, C] + fromBase6 [5, 2, D] + fromBase6 [C, 2, 4] = fromBase6 [4, 4, 3] → 
  (if C ≥ D then C - D else D - C) = 3 := by
  sorry

end NUMINAMATH_CALUDE_base6_addition_proof_l889_88993


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l889_88940

theorem quadratic_inequality_solution_set (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l889_88940


namespace NUMINAMATH_CALUDE_expression_equality_l889_88934

theorem expression_equality : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l889_88934


namespace NUMINAMATH_CALUDE_function_symmetry_l889_88964

/-- Given a function f: ℝ → ℝ, if the graph of f(x-1) is symmetric to the curve y = e^x 
    with respect to the y-axis, then f(x) = e^(-x-1) -/
theorem function_symmetry (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = Real.exp (-x)) → 
  (∀ x : ℝ, f x = Real.exp (-x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_l889_88964


namespace NUMINAMATH_CALUDE_parabola_point_coordinates_l889_88951

/-- Theorem: For a point P(x, y) on the parabola y² = 4x, if its distance from the focus is 4, then x = 3 and y = ±2√3 -/
theorem parabola_point_coordinates (x y : ℝ) :
  y^2 = 4*x →                           -- P is on the parabola y² = 4x
  (x - 1)^2 + y^2 = 16 →                -- Distance from P to focus (1, 0) is 4
  (x = 3 ∧ y = 2*Real.sqrt 3 ∨ y = -2*Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_parabola_point_coordinates_l889_88951


namespace NUMINAMATH_CALUDE_crayon_selection_problem_l889_88960

theorem crayon_selection_problem :
  let n : ℕ := 20  -- Total number of crayons
  let k : ℕ := 6   -- Number of crayons to select
  Nat.choose n k = 38760 := by
  sorry

end NUMINAMATH_CALUDE_crayon_selection_problem_l889_88960


namespace NUMINAMATH_CALUDE_exists_k_for_prime_divisor_inequality_l889_88919

/-- The largest prime divisor of a positive integer greater than 1 -/
def largest_prime_divisor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any odd prime q, there exists a positive integer k such that
    the largest prime divisor of (q^(2^k) - 1) is less than q, and
    q is less than the largest prime divisor of (q^(2^k) + 1) -/
theorem exists_k_for_prime_divisor_inequality (q : ℕ) (hq : q.Prime) (hq_odd : q % 2 = 1) :
  ∃ k : ℕ, k > 0 ∧
    largest_prime_divisor (q^(2^k) - 1) < q ∧
    q < largest_prime_divisor (q^(2^k) + 1) :=
  sorry

end NUMINAMATH_CALUDE_exists_k_for_prime_divisor_inequality_l889_88919


namespace NUMINAMATH_CALUDE_volume_of_rotated_specific_cone_l889_88913

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle in 3D space -/
structure Triangle3D where
  O : Point3D
  A : Point3D
  B : Point3D

/-- Represents a cone in 3D space -/
structure Cone3D where
  base_center : Point3D
  apex : Point3D
  base_radius : ℝ

/-- Function to create a cone by rotating a triangle around the x-axis -/
def createConeFromTriangle (t : Triangle3D) : Cone3D :=
  { base_center := ⟨t.A.x, 0, 0⟩,
    apex := t.O,
    base_radius := t.B.y - t.A.y }

/-- Function to calculate the volume of a solid obtained by rotating a cone around the y-axis -/
noncomputable def volumeOfRotatedCone (c : Cone3D) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_of_rotated_specific_cone :
  let t : Triangle3D := { O := ⟨0, 0, 0⟩, A := ⟨1, 0, 0⟩, B := ⟨1, 1, 0⟩ }
  let c : Cone3D := createConeFromTriangle t
  volumeOfRotatedCone c = (8 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_rotated_specific_cone_l889_88913


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l889_88997

/-- The line x = k intersects the parabola x = -3y^2 - 4y + 7 at exactly one point if and only if k = 25/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 4 * y + 7) ↔ k = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l889_88997


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l889_88941

def pear_juice_yield : ℚ := 10 / 2
def orange_juice_yield : ℚ := 6 / 3
def pears_used : ℕ := 4
def oranges_used : ℕ := 6

theorem orange_juice_percentage :
  (oranges_used * orange_juice_yield) / 
  (pears_used * pear_juice_yield + oranges_used * orange_juice_yield) = 375 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l889_88941


namespace NUMINAMATH_CALUDE_students_only_english_l889_88920

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total = (german - both) + both + (total - german)) :
  total - german = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_only_english_l889_88920


namespace NUMINAMATH_CALUDE_regression_analysis_considerations_l889_88945

/-- Represents the key considerations in regression analysis predictions -/
inductive RegressionConsideration
  | ApplicabilityToSamplePopulation
  | Temporality
  | InfluenceOfSampleRange
  | PredictionPrecision

/-- Represents a regression analysis model -/
structure RegressionModel where
  considerations : List RegressionConsideration

/-- Theorem stating the key considerations in regression analysis predictions -/
theorem regression_analysis_considerations (model : RegressionModel) :
  model.considerations = [
    RegressionConsideration.ApplicabilityToSamplePopulation,
    RegressionConsideration.Temporality,
    RegressionConsideration.InfluenceOfSampleRange,
    RegressionConsideration.PredictionPrecision
  ] := by sorry


end NUMINAMATH_CALUDE_regression_analysis_considerations_l889_88945


namespace NUMINAMATH_CALUDE_password_decryption_probability_l889_88969

theorem password_decryption_probability :
  let p_a : ℝ := 1/5  -- Probability of A's success
  let p_b : ℝ := 1/3  -- Probability of B's success
  let p_c : ℝ := 1/4  -- Probability of C's success
  let p_success : ℝ := 1 - (1 - p_a) * (1 - p_b) * (1 - p_c)  -- Probability of successful decryption
  p_success = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l889_88969


namespace NUMINAMATH_CALUDE_octagon_area_l889_88966

/-- Given two concentric squares with side length 2 and a line segment AB of length 3/4 between the squares,
    the area of the resulting octagon ABCDEFGH is 6. -/
theorem octagon_area (square_side : ℝ) (AB_length : ℝ) (h1 : square_side = 2) (h2 : AB_length = 3/4) :
  let triangle_area := (1/2) * square_side * AB_length
  let octagon_area := 8 * triangle_area
  octagon_area = 6 := by sorry

end NUMINAMATH_CALUDE_octagon_area_l889_88966


namespace NUMINAMATH_CALUDE_absent_laborers_l889_88995

theorem absent_laborers (W : ℝ) : 
  let L := 17.5
  let original_days := 6
  let actual_days := 10
  let absent := L * (1 - (original_days : ℝ) / (actual_days : ℝ))
  absent = 14 := by sorry

end NUMINAMATH_CALUDE_absent_laborers_l889_88995


namespace NUMINAMATH_CALUDE_namjoon_has_greater_sum_l889_88938

def jimin_numbers : List Nat := [1, 7]
def namjoon_numbers : List Nat := [6, 3]

theorem namjoon_has_greater_sum :
  List.sum namjoon_numbers > List.sum jimin_numbers := by
  sorry

end NUMINAMATH_CALUDE_namjoon_has_greater_sum_l889_88938


namespace NUMINAMATH_CALUDE_x_equals_eight_l889_88972

theorem x_equals_eight (y : ℝ) (some_number : ℝ) 
  (h1 : 2 * x - y = some_number) 
  (h2 : y = 2) 
  (h3 : some_number = 14) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_eight_l889_88972


namespace NUMINAMATH_CALUDE_pet_weights_l889_88954

/-- Given the weights of pets owned by Evan, Ivan, and Kara, prove their total weight -/
theorem pet_weights (evan_dog : ℕ) (ivan_dog : ℕ) (kara_cat : ℕ)
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog)) :
  evan_dog + ivan_dog + kara_cat = 432 := by
  sorry

end NUMINAMATH_CALUDE_pet_weights_l889_88954


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l889_88907

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    ∃ B C : ℝ × ℝ, 
      B.1 = c ∧ C.1 = c ∧ 
      B.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧ 
      C.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧
      (B.2 - C.2)^2 = 2 * (c + a)^2) →
  c / a = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l889_88907


namespace NUMINAMATH_CALUDE_expression_simplification_l889_88926

theorem expression_simplification (y : ℝ) :
  3 * y - 5 * y^2 + 10 - (8 - 3 * y + 5 * y^2 - y^3) = y^3 - 10 * y^2 + 6 * y + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l889_88926


namespace NUMINAMATH_CALUDE_darcie_father_age_l889_88996

def darcie_age : ℕ := 4

theorem darcie_father_age (mother_age father_age : ℕ) 
  (h1 : darcie_age = mother_age / 6)
  (h2 : mother_age * 5 = father_age * 4) : 
  father_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_darcie_father_age_l889_88996


namespace NUMINAMATH_CALUDE_chiefs_gold_l889_88931

/-- A graph representing druids and their willingness to shake hands. -/
structure DruidGraph where
  /-- The set of vertices (druids) in the graph. -/
  V : Type
  /-- The edge relation, representing willingness to shake hands. -/
  E : V → V → Prop
  /-- The graph has no cycles of length 4 or more. -/
  no_long_cycles : ∀ (a b c d : V), E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a DruidGraph. -/
def num_vertices (G : DruidGraph) : ℕ := sorry

/-- The number of edges in a DruidGraph. -/
def num_edges (G : DruidGraph) : ℕ := sorry

/-- 
The chief's gold theorem: In a DruidGraph, the chief can keep at least 3 gold coins.
This is equivalent to showing that 3n - 2e ≥ 3, where n is the number of vertices and e is the number of edges.
-/
theorem chiefs_gold (G : DruidGraph) : 
  3 * (num_vertices G) - 2 * (num_edges G) ≥ 3 := by sorry

end NUMINAMATH_CALUDE_chiefs_gold_l889_88931


namespace NUMINAMATH_CALUDE_candy_cost_l889_88928

theorem candy_cost (tickets_game1 tickets_game2 candies : ℕ) 
  (h1 : tickets_game1 = 3)
  (h2 : tickets_game2 = 5)
  (h3 : candies = 2) :
  (tickets_game1 + tickets_game2) / candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_l889_88928


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l889_88902

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length : ℝ) (breadth : ℝ) (cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Theorem: The total cost of fencing the given rectangular plot is 5300 currency units -/
theorem fencing_cost_theorem :
  let length : ℝ := 63
  let breadth : ℝ := 37
  let cost_per_meter : ℝ := 26.50
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 63 37 26.50

end NUMINAMATH_CALUDE_fencing_cost_theorem_l889_88902


namespace NUMINAMATH_CALUDE_planting_cost_l889_88943

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_planting_cost_l889_88943


namespace NUMINAMATH_CALUDE_parabola_intercept_sum_l889_88984

/-- Represents a parabola of the form x = 3y^2 - 9y + 5 --/
def Parabola (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- The x-intercept of the parabola --/
def x_intercept (a : ℝ) : Prop := Parabola a 0

/-- The y-intercepts of the parabola --/
def y_intercepts (b c : ℝ) : Prop := Parabola 0 b ∧ Parabola 0 c ∧ b ≠ c

theorem parabola_intercept_sum (a b c : ℝ) :
  x_intercept a → y_intercepts b c → a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercept_sum_l889_88984


namespace NUMINAMATH_CALUDE_quadratic_max_value_l889_88911

/-- The quadratic function y = -(x-m)^2 + m^2 + 1 has a maximum value of 4 when -2 ≤ x ≤ 1. -/
theorem quadratic_max_value (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → -(x-m)^2 + m^2 + 1 ≤ 4) ∧ 
  (∃ x, -2 ≤ x ∧ x ≤ 1 ∧ -(x-m)^2 + m^2 + 1 = 4) →
  m = 2 ∨ m = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l889_88911


namespace NUMINAMATH_CALUDE_percentage_difference_l889_88985

theorem percentage_difference : (0.7 * 40) - (4 / 5 * 25) = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l889_88985


namespace NUMINAMATH_CALUDE_fraction_simplification_l889_88970

theorem fraction_simplification (x : ℝ) (h : x = 3) : 
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l889_88970


namespace NUMINAMATH_CALUDE_find_n_l889_88937

theorem find_n : ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) ∧ (n = -7/2) := by
  sorry

end NUMINAMATH_CALUDE_find_n_l889_88937


namespace NUMINAMATH_CALUDE_remaining_square_exists_l889_88949

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square of cells -/
structure Square :=
  (size : ℕ)

theorem remaining_square_exists (g : Grid) (s : Square) (num_removed : ℕ) :
  g.rows = 29 →
  g.cols = 29 →
  s.size = 2 →
  num_removed = 99 →
  ∃ (remaining : ℕ), remaining ≥ 1 ∧ remaining = (g.rows / s.size) * (g.cols / s.size) - num_removed :=
by sorry

#check remaining_square_exists

end NUMINAMATH_CALUDE_remaining_square_exists_l889_88949


namespace NUMINAMATH_CALUDE_quadratic_equation_two_real_roots_l889_88998

theorem quadratic_equation_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k + 1) * x^2 - 2 * x + 1 = 0 ∧ (k + 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_two_real_roots_l889_88998


namespace NUMINAMATH_CALUDE_minimum_at_one_positive_when_minimum_less_than_one_l889_88988

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + 1) * (a * x - 1) / x

-- Theorem 1: If the minimum point of f(x) is at x_0 = 1, then a = 1
theorem minimum_at_one (a : ℝ) :
  (∀ x > 0, f a x ≥ f a 1) → a = 1 := by sorry

-- Theorem 2: If 0 < x_0 < 1, where x_0 is the minimum point of f(x), then f(x) > 0 for all x > 0
theorem positive_when_minimum_less_than_one (a : ℝ) (x_0 : ℝ) :
  (0 < x_0 ∧ x_0 < 1) →
  (∀ x > 0, f a x ≥ f a x_0) →
  (∀ x > 0, f a x > 0) := by sorry

end NUMINAMATH_CALUDE_minimum_at_one_positive_when_minimum_less_than_one_l889_88988


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l889_88910

def solution_set : Set (ℕ × ℕ) :=
  {(5, 20), (6, 12), (8, 8), (12, 6), (20, 5)}

theorem diophantine_equation_solution :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
    (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 4 ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l889_88910


namespace NUMINAMATH_CALUDE_sum_lent_problem_l889_88959

/-- Proves that given a sum P lent at 5% per annum simple interest for 8 years,
    if the interest is $360 less than P, then P equals $600. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.05 * 8 = P - 360) → P = 600 := by
  sorry

end NUMINAMATH_CALUDE_sum_lent_problem_l889_88959


namespace NUMINAMATH_CALUDE_alpha_and_function_range_l889_88939

open Real

theorem alpha_and_function_range 
  (α : ℝ) 
  (h1 : 2 * sin α * tan α = 3) 
  (h2 : 0 < α) 
  (h3 : α < π) : 
  α = π / 3 ∧ 
  ∀ x ∈ Set.Icc 0 (π / 4), 
    -1 ≤ 4 * sin x * sin (x - α) ∧ 
    4 * sin x * sin (x - α) ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_alpha_and_function_range_l889_88939


namespace NUMINAMATH_CALUDE_base8_157_equals_base10_111_l889_88916

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- Theorem: The base-8 number 157 is equal to the base-10 number 111 --/
theorem base8_157_equals_base10_111 : base8_to_base10 157 = 111 := by
  sorry

end NUMINAMATH_CALUDE_base8_157_equals_base10_111_l889_88916


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l889_88918

/-- Given the conditions of frisbee sales, proves the minimum number of $4 frisbees sold -/
theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (price_low price_high : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60)
  (h_price_low : price_low = 3)
  (h_price_high : price_high = 4)
  (h_receipts : total_receipts = 200) :
  ∃ (x y : ℕ), 
    x + y = total_frisbees ∧ 
    price_low * x + price_high * y = total_receipts ∧
    y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l889_88918


namespace NUMINAMATH_CALUDE_factorization_problems_l889_88978

theorem factorization_problems :
  (∀ x : ℝ, 4*x^2 - 16 = 4*(x+2)*(x-2)) ∧
  (∀ x y : ℝ, 2*x^3 - 12*x^2*y + 18*x*y^2 = 2*x*(x-3*y)^2) := by
sorry

end NUMINAMATH_CALUDE_factorization_problems_l889_88978


namespace NUMINAMATH_CALUDE_parabola_point_y_coordinate_l889_88927

/-- The y-coordinate of a point on a parabola at a given distance from the focus -/
theorem parabola_point_y_coordinate (x y : ℝ) :
  y = -4 * x^2 →  -- Point M is on the parabola y = -4x²
  (x^2 + (y - 1/4)^2) = 1 →  -- Distance from M to focus (0, 1/4) is 1
  y = -15/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_y_coordinate_l889_88927


namespace NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l889_88905

theorem fraction_product_cube (a b c d : ℚ) : 
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem specific_fraction_product : 
  (5 / 8 : ℚ) ^ 3 * (4 / 9 : ℚ) ^ 3 = 125 / 5832 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cube_specific_fraction_product_l889_88905


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l889_88950

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 2*y

-- Define the line passing through A(0,-2) and B(t,0)
def line (t x y : ℝ) : Prop := y = (2/t)*x - 2

-- Define the condition for no intersection
def no_intersection (t : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y → ¬(line t x y)

-- Theorem statement
theorem parabola_line_intersection (t : ℝ) :
  no_intersection t ↔ t < -1 ∨ t > 1 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l889_88950


namespace NUMINAMATH_CALUDE_temperature_is_dependent_variable_l889_88974

/-- Represents a variable in the solar water heating process -/
inductive Variable
  | Temperature
  | Duration
  | Intensity
  | Heater

/-- Represents the relationship between variables in the solar water heating process -/
structure SolarWaterHeating where
  temp : Variable
  duration : Variable
  changes_with : Variable → Variable → Prop

/-- Definition of a dependent variable -/
def is_dependent_variable (v : Variable) (swh : SolarWaterHeating) : Prop :=
  ∃ (other : Variable), swh.changes_with v other

/-- Theorem stating that the temperature is the dependent variable in the solar water heating process -/
theorem temperature_is_dependent_variable (swh : SolarWaterHeating) 
  (h1 : swh.temp = Variable.Temperature)
  (h2 : swh.duration = Variable.Duration)
  (h3 : swh.changes_with swh.temp swh.duration) :
  is_dependent_variable swh.temp swh :=
by sorry


end NUMINAMATH_CALUDE_temperature_is_dependent_variable_l889_88974


namespace NUMINAMATH_CALUDE_percent_composition_l889_88956

-- Define the % operations
def percent_right (x : ℤ) : ℤ := 8 - x
def percent_left (x : ℤ) : ℤ := x - 8

-- Theorem statement
theorem percent_composition : percent_left (percent_right 10) = -10 := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l889_88956


namespace NUMINAMATH_CALUDE_carls_playground_area_l889_88901

/-- Represents a rectangular playground with fence posts. -/
structure Playground where
  total_posts : ℕ
  post_spacing : ℝ
  short_side_posts : ℕ
  long_side_posts : ℕ

/-- Calculates the area of the playground given its specifications. -/
def calculate_area (p : Playground) : ℝ :=
  ((p.short_side_posts - 1) * p.post_spacing) * ((p.long_side_posts - 1) * p.post_spacing)

/-- Theorem stating the area of Carl's playground is 324 square yards. -/
theorem carls_playground_area :
  ∃ (p : Playground),
    p.total_posts = 24 ∧
    p.post_spacing = 3 ∧
    p.long_side_posts = 2 * p.short_side_posts ∧
    calculate_area p = 324 := by
  sorry

end NUMINAMATH_CALUDE_carls_playground_area_l889_88901


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_l889_88952

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_first_two : a 1 + a 2 = 1
  sum_third_fourth : a 3 + a 4 = 9

/-- The sum of the fourth and fifth terms is either 27 or -27 -/
theorem fourth_fifth_sum (seq : GeometricSequence) : 
  seq.a 4 + seq.a 5 = 27 ∨ seq.a 4 + seq.a 5 = -27 := by
  sorry


end NUMINAMATH_CALUDE_fourth_fifth_sum_l889_88952


namespace NUMINAMATH_CALUDE_expression_calculation_l889_88908

theorem expression_calculation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end NUMINAMATH_CALUDE_expression_calculation_l889_88908


namespace NUMINAMATH_CALUDE_empty_can_weight_l889_88923

/-- Given a can that weighs 34 kg when full of milk and 17.5 kg when half-full, 
    prove that the empty can weighs 1 kg. -/
theorem empty_can_weight (full_weight half_weight : ℝ) 
  (h_full : full_weight = 34)
  (h_half : half_weight = 17.5) : 
  ∃ (empty_weight milk_weight : ℝ),
    empty_weight + milk_weight = full_weight ∧
    empty_weight + milk_weight / 2 = half_weight ∧
    empty_weight = 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_can_weight_l889_88923


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l889_88979

theorem quadratic_equation_solution (a b : ℝ) : 
  ∃ x : ℝ, (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l889_88979


namespace NUMINAMATH_CALUDE_wheels_per_row_calculation_l889_88925

/-- Calculates the number of wheels per row given the total number of wheels,
    number of trains, carriages per train, and rows of wheels per carriage. -/
def wheels_per_row (total_wheels : ℕ) (num_trains : ℕ) (carriages_per_train : ℕ) (rows_per_carriage : ℕ) : ℕ :=
  total_wheels / (num_trains * carriages_per_train * rows_per_carriage)

/-- Theorem stating that given 4 trains, 4 carriages per train, 3 rows of wheels per carriage,
    and 240 wheels in total, the number of wheels in each row is 5. -/
theorem wheels_per_row_calculation :
  wheels_per_row 240 4 4 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_wheels_per_row_calculation_l889_88925


namespace NUMINAMATH_CALUDE_brians_breath_holding_factor_l889_88922

/-- Given Brian's breath-holding practice over three weeks, prove the factor of increase after the first week. -/
theorem brians_breath_holding_factor
  (initial_time : ℝ)
  (final_time : ℝ)
  (h_initial : initial_time = 10)
  (h_final : final_time = 60)
  (F : ℝ)
  (h_week2 : F * initial_time * 2 = F * initial_time * 2)
  (h_week3 : F * initial_time * 2 * 1.5 = final_time) :
  F = 2 := by
  sorry

end NUMINAMATH_CALUDE_brians_breath_holding_factor_l889_88922


namespace NUMINAMATH_CALUDE_straight_line_distance_l889_88983

/-- The straight-line distance between two points, where one point is 20 yards south
    and 50 yards east of the other, is 10√29 yards. -/
theorem straight_line_distance (south east : ℝ) (h1 : south = 20) (h2 : east = 50) :
  Real.sqrt (south^2 + east^2) = 10 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_straight_line_distance_l889_88983


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l889_88971

theorem roots_sum_reciprocals (p q : ℝ) (x₁ x₂ : ℝ) (hx₁ : x₁^2 + p*x₁ + q = 0) (hx₂ : x₂^2 + p*x₂ + q = 0) (hq : q ≠ 0) :
  x₁/x₂ + x₂/x₁ = (p^2 - 2*q) / q :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l889_88971


namespace NUMINAMATH_CALUDE_transmission_time_is_8_67_minutes_l889_88930

/-- Represents the number of chunks in a regular block -/
def regular_block_chunks : ℕ := 800

/-- Represents the number of chunks in a large block -/
def large_block_chunks : ℕ := 1600

/-- Represents the number of regular blocks -/
def num_regular_blocks : ℕ := 70

/-- Represents the number of large blocks -/
def num_large_blocks : ℕ := 30

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Calculates the total number of chunks to be transmitted -/
def total_chunks : ℕ := 
  num_regular_blocks * regular_block_chunks + num_large_blocks * large_block_chunks

/-- Calculates the transmission time in seconds -/
def transmission_time_seconds : ℕ := total_chunks / transmission_rate

/-- Theorem stating that the transmission time is 8.67 minutes -/
theorem transmission_time_is_8_67_minutes : 
  (transmission_time_seconds : ℚ) / 60 = 8.67 := by sorry

end NUMINAMATH_CALUDE_transmission_time_is_8_67_minutes_l889_88930


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l889_88932

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  -- Line equation
  (∀ x y, a * x - b * y - 2 = 0 → True) →
  -- Curve equation
  (∀ x, f x = x^3 + x) →
  -- Point P
  f 1 = 2 →
  -- Perpendicular tangents at P
  (a / b) * (f' 1) = -1 →
  -- Conclusion
  a / b = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l889_88932


namespace NUMINAMATH_CALUDE_two_books_selection_ways_l889_88986

/-- The number of ways to select two books of different subjects from three shelves -/
def select_two_books (chinese_books : ℕ) (math_books : ℕ) (english_books : ℕ) : ℕ :=
  chinese_books * math_books + chinese_books * english_books + math_books * english_books

/-- Theorem stating that selecting two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_selection_ways :
  select_two_books 10 9 8 = 242 := by
  sorry

end NUMINAMATH_CALUDE_two_books_selection_ways_l889_88986


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l889_88903

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l889_88903


namespace NUMINAMATH_CALUDE_smallest_circle_area_l889_88909

theorem smallest_circle_area (p1 p2 : ℝ × ℝ) (h : p1 = (-3, -2) ∧ p2 = (2, 4)) :
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let r := d / 2
  let A := π * r^2
  A = (61 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_circle_area_l889_88909


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l889_88935

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a = (-1, 2) ∧ b = (m, 1)) :
  a.1 * b.1 + a.2 * b.2 = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l889_88935
