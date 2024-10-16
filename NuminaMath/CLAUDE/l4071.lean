import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_nested_logs_l4071_407143

-- Define the logarithm functions
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem sum_of_nested_logs (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧
  log 3 (log 4 (log 2 y)) = 0 ∧
  log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nested_logs_l4071_407143


namespace NUMINAMATH_CALUDE_summer_program_sophomores_l4071_407132

theorem summer_program_sophomores :
  ∀ (total_students : ℕ) 
    (non_soph_jun : ℕ)
    (soph_debate_ratio : ℚ)
    (jun_debate_ratio : ℚ),
  total_students = 40 →
  non_soph_jun = 5 →
  soph_debate_ratio = 1/5 →
  jun_debate_ratio = 1/4 →
  ∃ (sophomores juniors : ℚ),
    sophomores + juniors = total_students - non_soph_jun ∧
    sophomores * soph_debate_ratio = juniors * jun_debate_ratio ∧
    sophomores = 175/9 :=
by sorry

end NUMINAMATH_CALUDE_summer_program_sophomores_l4071_407132


namespace NUMINAMATH_CALUDE_cubed_49_plus_1_l4071_407117

theorem cubed_49_plus_1 : 49^3 + 3*(49^2) + 3*49 + 1 = 125000 := by
  sorry

end NUMINAMATH_CALUDE_cubed_49_plus_1_l4071_407117


namespace NUMINAMATH_CALUDE_injective_properties_l4071_407198

variable {A B : Type}
variable (f : A → B)

theorem injective_properties (h : Function.Injective f) :
  (∀ (x₁ x₂ : A), x₁ ≠ x₂ → f x₁ ≠ f x₂) ∧
  (∀ (b : B), ∃! (a : A), f a = b) :=
by sorry

end NUMINAMATH_CALUDE_injective_properties_l4071_407198


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l4071_407128

def n : ℕ := 5040000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem fifth_largest_divisor_of_n :
  kth_largest_divisor 5 n = 315000000 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_of_n_l4071_407128


namespace NUMINAMATH_CALUDE_fraction_sum_equals_three_halves_l4071_407197

theorem fraction_sum_equals_three_halves (a b : ℕ+) :
  ∃ x y : ℕ+, (x : ℚ) / ((y : ℚ) + (a : ℚ)) + (y : ℚ) / ((x : ℚ) + (b : ℚ)) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_three_halves_l4071_407197


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l4071_407186

theorem quadratic_inequality_condition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l4071_407186


namespace NUMINAMATH_CALUDE_sum_of_unknowns_l4071_407171

theorem sum_of_unknowns (x₁ x₂ x₃ : ℝ) 
  (h : (1 + 2 + 3 + 4 + x₁ + x₂ + x₃) / 7 = 8) : 
  x₁ + x₂ + x₃ = 46 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unknowns_l4071_407171


namespace NUMINAMATH_CALUDE_sum_a4_a5_a6_l4071_407119

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_a2_a3 : a 2 + a 3 = 13
  a1_eq_2 : a 1 = 2

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_a4_a5_a6 (seq : ArithmeticSequence) : seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_a4_a5_a6_l4071_407119


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l4071_407112

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x - 1

theorem root_sum_reciprocal (a b c : ℝ) (m n : ℕ) :
  f a = 0 → f b = 0 → f c = 0 →
  (1 / (a^3 + b^3) + 1 / (b^3 + c^3) + 1 / (c^3 + a^3) : ℝ) = m / n →
  m > 0 → n > 0 →
  Nat.gcd m n = 1 →
  100 * m + n = 3989 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l4071_407112


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4071_407105

-- Define the arithmetic sequence a_n
def a (n : ℕ+) : ℚ := 2 * n + 1

-- Define S_n as the sum of the first n terms of a_n
def S (n : ℕ+) : ℚ := n^2 + 2*n

-- Define b_n
def b (n : ℕ+) : ℚ := 1 / (a n^2 - 1)

-- Define T_n as the sum of the first n terms of b_n
def T (n : ℕ+) : ℚ := n / (4 * (n + 1))

-- State the theorem
theorem arithmetic_sequence_properties :
  (a 2 = 5) ∧
  (a 4 + a 6 = 22) ∧
  (∀ n : ℕ+, a n = 2*n + 1) ∧
  (∀ n : ℕ+, S n = n^2 + 2*n) ∧
  (∀ n : ℕ+, T n = n / (4 * (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l4071_407105


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4071_407157

def U : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {-2, 2}

theorem complement_of_A_in_U :
  U \ A = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4071_407157


namespace NUMINAMATH_CALUDE_reading_difference_l4071_407183

/-- Calculates the total pages read given a list of (rate, days) pairs -/
def totalPagesRead (readingPlan : List (Nat × Nat)) : Nat :=
  readingPlan.map (fun (rate, days) => rate * days) |>.sum

theorem reading_difference : 
  let gregPages := totalPagesRead [(18, 7), (22, 14)]
  let bradPages := totalPagesRead [(26, 5), (20, 12)]
  let emilyPages := totalPagesRead [(15, 3), (24, 7), (18, 7)]
  gregPages + bradPages - emilyPages = 465 := by
  sorry

#eval totalPagesRead [(18, 7), (22, 14)] -- Greg's pages
#eval totalPagesRead [(26, 5), (20, 12)] -- Brad's pages
#eval totalPagesRead [(15, 3), (24, 7), (18, 7)] -- Emily's pages

end NUMINAMATH_CALUDE_reading_difference_l4071_407183


namespace NUMINAMATH_CALUDE_quadratic_roots_l4071_407110

/-- Represents a quadratic equation of the form 2x^2 + (m+2)x + m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  2 * x^2 + (m + 2) * x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m + 2)^2 - 4 * 2 * m

theorem quadratic_roots (m : ℝ) :
  (∀ x, ∃ y z, quadratic_equation m x → x = y ∨ x = z) ∧
  (discriminant 2 = 0) ∧
  (quadratic_equation 2 (-1) ∧ ∀ x, quadratic_equation 2 x → x = -1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l4071_407110


namespace NUMINAMATH_CALUDE_circus_performers_standing_time_l4071_407185

/-- The combined time that Pulsar, Polly, and Petra stand on their back legs -/
theorem circus_performers_standing_time :
  let pulsar_time : ℕ := 10
  let polly_time : ℕ := 3 * pulsar_time
  let petra_time : ℕ := polly_time / 6
  pulsar_time + polly_time + petra_time = 45 := by
sorry

end NUMINAMATH_CALUDE_circus_performers_standing_time_l4071_407185


namespace NUMINAMATH_CALUDE_cake_shop_work_duration_l4071_407103

/-- Calculates the number of months worked given the total hours worked by Cathy -/
def months_worked (total_hours : ℕ) : ℚ :=
  let hours_per_week : ℕ := 20
  let weeks_per_month : ℕ := 4
  let extra_hours : ℕ := 20
  let regular_hours : ℕ := total_hours - extra_hours
  let regular_weeks : ℚ := regular_hours / hours_per_week
  regular_weeks / weeks_per_month

theorem cake_shop_work_duration :
  months_worked 180 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_shop_work_duration_l4071_407103


namespace NUMINAMATH_CALUDE_mad_hatter_waiting_time_l4071_407114

/-- The rate at which the Mad Hatter's clock runs compared to normal time -/
def mad_hatter_rate : ℚ := 5/4

/-- The rate at which the March Hare's clock runs compared to normal time -/
def march_hare_rate : ℚ := 5/6

/-- The agreed meeting time in hours after noon -/
def meeting_time : ℚ := 5

theorem mad_hatter_waiting_time :
  let mad_hatter_arrival := meeting_time / mad_hatter_rate
  let march_hare_arrival := meeting_time / march_hare_rate
  march_hare_arrival - mad_hatter_arrival = 2 := by sorry

end NUMINAMATH_CALUDE_mad_hatter_waiting_time_l4071_407114


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_is_constant_l4071_407129

def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * q^n

def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 1

theorem arithmetic_and_geometric_is_constant (a : ℕ → ℝ) :
  is_arithmetic_progression a → is_geometric_progression a → is_constant_sequence a :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_is_constant_l4071_407129


namespace NUMINAMATH_CALUDE_inverse_g_at_neg_138_l4071_407187

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg_138 :
  g⁻¹ (-138) = -3 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_neg_138_l4071_407187


namespace NUMINAMATH_CALUDE_factorization_equality_l4071_407150

theorem factorization_equality (x y : ℝ) : 4 * x^2 * y - 12 * x * y = 4 * x * y * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4071_407150


namespace NUMINAMATH_CALUDE_chris_least_money_l4071_407113

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Chris : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- State the conditions
axiom different_amounts : ∀ p q : Person, p ≠ q → money p ≠ money q
axiom chris_less_than_bob : money Person.Chris < money Person.Bob
axiom dana_less_than_bob : money Person.Dana < money Person.Bob
axiom alice_more_than_chris : money Person.Chris < money Person.Alice
axiom eve_more_than_chris : money Person.Chris < money Person.Eve
axiom dana_equal_eve : money Person.Dana = money Person.Eve
axiom dana_less_than_alice : money Person.Dana < money Person.Alice
axiom bob_more_than_eve : money Person.Eve < money Person.Bob

-- State the theorem
theorem chris_least_money :
  ∀ p : Person, p ≠ Person.Chris → money Person.Chris ≤ money p :=
sorry

end NUMINAMATH_CALUDE_chris_least_money_l4071_407113


namespace NUMINAMATH_CALUDE_central_projection_items_correct_l4071_407190

-- Define the set of all items
inductive Item : Type
  | Searchlight
  | CarLight
  | Sun
  | Moon
  | DeskLamp

-- Define a predicate for items that form central projections
def FormsCentralProjection (item : Item) : Prop :=
  match item with
  | Item.Searchlight => True
  | Item.CarLight => True
  | Item.Sun => False
  | Item.Moon => False
  | Item.DeskLamp => True

-- Define the set of items that form central projections
def CentralProjectionItems : Set Item :=
  {item : Item | FormsCentralProjection item}

-- Theorem statement
theorem central_projection_items_correct :
  CentralProjectionItems = {Item.Searchlight, Item.CarLight, Item.DeskLamp} := by
  sorry


end NUMINAMATH_CALUDE_central_projection_items_correct_l4071_407190


namespace NUMINAMATH_CALUDE_parabola_properties_l4071_407180

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y = x^2 + 2

-- State the theorem
theorem parabola_properties :
  (∀ x y : ℝ, parabola x y → (x = 0 → y = 2)) ∧ 
  (∀ x y : ℝ, parabola x y → ∀ h : ℝ, h > 0 → parabola x (y + h)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l4071_407180


namespace NUMINAMATH_CALUDE_dance_studio_dancers_l4071_407125

/-- The number of performances -/
def num_performances : ℕ := 40

/-- The number of dancers in each performance -/
def dancers_per_performance : ℕ := 10

/-- The maximum number of times any pair of dancers can perform together -/
def max_pair_performances : ℕ := 1

/-- The minimum number of dancers required -/
def min_dancers : ℕ := 60

theorem dance_studio_dancers :
  ∀ (n : ℕ), n ≥ min_dancers →
  (n.choose 2) ≥ num_performances * (dancers_per_performance.choose 2) :=
by sorry

end NUMINAMATH_CALUDE_dance_studio_dancers_l4071_407125


namespace NUMINAMATH_CALUDE_two_lines_iff_m_eq_one_l4071_407134

/-- The equation x^2 - my^2 + 2x + 2y = 0 represents two lines if and only if m = 1 -/
theorem two_lines_iff_m_eq_one (m : ℝ) :
  (∃ (a b c d : ℝ), ∀ (x y : ℝ),
    (x^2 - m*y^2 + 2*x + 2*y = 0) ↔ ((a*x + b*y = 1) ∨ (c*x + d*y = 1))) ↔
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_two_lines_iff_m_eq_one_l4071_407134


namespace NUMINAMATH_CALUDE_hundredth_odd_followed_by_hundredth_even_l4071_407159

/-- The nth odd positive integer -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

/-- The nth even positive integer -/
def nth_even (n : ℕ) : ℕ := 2 * n

theorem hundredth_odd_followed_by_hundredth_even :
  nth_odd 100 = 199 ∧ nth_even 100 = nth_odd 100 + 1 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_followed_by_hundredth_even_l4071_407159


namespace NUMINAMATH_CALUDE_periodic_odd_function_value_l4071_407182

def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)

theorem periodic_odd_function_value (f : ℝ → ℝ) 
  (h : periodic_odd_function f) : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_value_l4071_407182


namespace NUMINAMATH_CALUDE_abs_three_minus_pi_l4071_407144

theorem abs_three_minus_pi : |3 - Real.pi| = Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_three_minus_pi_l4071_407144


namespace NUMINAMATH_CALUDE_triangle_inequality_l4071_407148

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * (a + b + c) * (a * b + b * c + c * a) ≤ (a + b + c) * (a^2 + b^2 + c^2) + 9 * a * b * c :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4071_407148


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l4071_407108

theorem bowling_ball_weight (b k : ℝ) 
  (h1 : 5 * b = 3 * k) 
  (h2 : 4 * k = 120) : 
  b = 18 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l4071_407108


namespace NUMINAMATH_CALUDE_glass_bowl_selling_price_l4071_407107

theorem glass_bowl_selling_price
  (total_bowls : ℕ)
  (cost_per_bowl : ℚ)
  (sold_bowls : ℕ)
  (gain_percentage : ℚ)
  (h1 : total_bowls = 110)
  (h2 : cost_per_bowl = 10)
  (h3 : sold_bowls = 100)
  (h4 : gain_percentage = 27.27272727272727 / 100) :
  let total_cost : ℚ := total_bowls * cost_per_bowl
  let total_revenue : ℚ := total_cost * (1 + gain_percentage)
  let selling_price : ℚ := total_revenue / sold_bowls
  selling_price = 14 := by sorry

end NUMINAMATH_CALUDE_glass_bowl_selling_price_l4071_407107


namespace NUMINAMATH_CALUDE_tenth_root_of_unity_l4071_407164

theorem tenth_root_of_unity : 
  ∃ n : ℕ, 0 ≤ n ∧ n < 10 ∧ 
  (Complex.tan (π / 6) + Complex.I) / (Complex.tan (π / 6) - Complex.I) = 
  Complex.exp (2 * π * I * (n : ℂ) / 10) := by sorry

end NUMINAMATH_CALUDE_tenth_root_of_unity_l4071_407164


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l4071_407137

/-- Given Andrew's purchase of grapes and mangoes, prove the rate per kg for mangoes -/
theorem mango_rate_calculation (grapes_kg : ℕ) (grapes_rate : ℕ) (mangoes_kg : ℕ) (total_paid : ℕ) :
  grapes_kg = 11 →
  grapes_rate = 98 →
  mangoes_kg = 7 →
  total_paid = 1428 →
  (total_paid - grapes_kg * grapes_rate) / mangoes_kg = 50 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l4071_407137


namespace NUMINAMATH_CALUDE_angle_theta_value_l4071_407121

theorem angle_theta_value (θ : Real) (A B : Set Real) : 
  A = {1, Real.cos θ} →
  B = {0, 1/2, 1} →
  A ⊆ B →
  0 < θ →
  θ < π/2 →
  θ = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_theta_value_l4071_407121


namespace NUMINAMATH_CALUDE_no_solution_exists_l4071_407154

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 3 * x

/-- A function that is even (symmetric about y-axis) -/
def IsEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x = f (-x)

/-- There is no function satisfying both the functional equation and evenness -/
theorem no_solution_exists : ¬ ∃ f : ℝ → ℝ, SatisfiesFunctionalEq f ∧ IsEvenFunction f := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4071_407154


namespace NUMINAMATH_CALUDE_max_carlson_jars_l4071_407140

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars :=
  (carlson_weight : ℕ)  -- Total weight of Carlson's jars
  (baby_weight : ℕ)     -- Total weight of Baby's jars
  (lightest_jar : ℕ)    -- Weight of Carlson's lightest jar

/-- The conditions of the problem -/
def valid_jam_state (j : JamJars) : Prop :=
  j.carlson_weight = 13 * j.baby_weight ∧
  j.carlson_weight - j.lightest_jar = 8 * (j.baby_weight + j.lightest_jar)

/-- The maximum number of jars Carlson could have initially -/
def max_jars (j : JamJars) : ℕ := j.carlson_weight / j.lightest_jar

/-- The theorem to prove -/
theorem max_carlson_jars :
  ∀ j : JamJars, valid_jam_state j → max_jars j ≤ 23 :=
by sorry

end NUMINAMATH_CALUDE_max_carlson_jars_l4071_407140


namespace NUMINAMATH_CALUDE_b_finishing_time_l4071_407189

/-- The number of days it takes B to finish the remaining work after A leaves -/
def days_for_B_to_finish (a_days b_days collab_days : ℚ) : ℚ :=
  let total_work := 1
  let a_rate := 1 / a_days
  let b_rate := 1 / b_days
  let combined_rate := a_rate + b_rate
  let work_done_together := combined_rate * collab_days
  let remaining_work := total_work - work_done_together
  remaining_work / b_rate

/-- Theorem stating that B will take 76/5 days to finish the remaining work -/
theorem b_finishing_time :
  days_for_B_to_finish 5 16 2 = 76 / 5 := by
  sorry

end NUMINAMATH_CALUDE_b_finishing_time_l4071_407189


namespace NUMINAMATH_CALUDE_shoe_cost_theorem_l4071_407151

/-- Calculates the final price after applying discount and tax -/
def calculate_price (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

/-- Calculates the total cost of four pairs of shoes -/
def total_cost (price1 price2 price3 : ℝ) : ℝ :=
  let pair1 := calculate_price price1 0.10 0.05
  let pair2 := calculate_price (price1 * 1.5) 0.15 0.07
  let pair3_4 := price3 * (1 + 0.12)
  pair1 + pair2 + pair3_4

theorem shoe_cost_theorem :
  total_cost 22 (22 * 1.5) 40 = 95.60 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_theorem_l4071_407151


namespace NUMINAMATH_CALUDE_system_solutions_l4071_407133

def is_solution (x y z : ℝ) : Prop :=
  x + y + z = 3 ∧
  x + 2*y - z = 2 ∧
  x + y*z + z*x = 3

theorem system_solutions :
  (∃ (x y z : ℝ), is_solution x y z) ∧
  (∀ (x y z : ℝ), is_solution x y z →
    ((x = 6 + Real.sqrt 29 ∧
      y = (-7 - 2 * Real.sqrt 29) / 3 ∧
      z = (-2 - Real.sqrt 29) / 3) ∨
     (x = 6 - Real.sqrt 29 ∧
      y = (-7 + 2 * Real.sqrt 29) / 3 ∧
      z = (-2 + Real.sqrt 29) / 3))) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l4071_407133


namespace NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l4071_407115

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a function to get the common external tangent intersection point of two circles
def commonExternalTangentIntersection (c1 c2 : Circle) : Point :=
  sorry

-- Theorem statement
theorem external_tangent_intersections_collinear (c1 c2 c3 : Circle) :
  let A := commonExternalTangentIntersection c1 c2
  let B := commonExternalTangentIntersection c2 c3
  let C := commonExternalTangentIntersection c3 c1
  ∃ (m b : ℝ), (A.1 = m * A.2 + b) ∧ (B.1 = m * B.2 + b) ∧ (C.1 = m * C.2 + b) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l4071_407115


namespace NUMINAMATH_CALUDE_career_preference_circle_graph_l4071_407146

theorem career_preference_circle_graph (total_students : ℝ) (h_positive : total_students > 0) :
  let male_ratio : ℝ := 2
  let female_ratio : ℝ := 3
  let male_preference_ratio : ℝ := 1/4
  let female_preference_ratio : ℝ := 3/4
  let total_ratio : ℝ := male_ratio + female_ratio
  let male_students : ℝ := (male_ratio / total_ratio) * total_students
  let female_students : ℝ := (female_ratio / total_ratio) * total_students
  let preference_students : ℝ := male_preference_ratio * male_students + female_preference_ratio * female_students
  let preference_ratio : ℝ := preference_students / total_students
  let circle_degrees : ℝ := 360
  
  preference_ratio * circle_degrees = 198 := by sorry

end NUMINAMATH_CALUDE_career_preference_circle_graph_l4071_407146


namespace NUMINAMATH_CALUDE_new_car_sticker_price_l4071_407168

/-- Calculates the sticker price of a new car based on given conditions --/
theorem new_car_sticker_price 
  (old_car_value : ℝ)
  (old_car_sale_percentage : ℝ)
  (new_car_purchase_percentage : ℝ)
  (out_of_pocket : ℝ)
  (h1 : old_car_value = 20000)
  (h2 : old_car_sale_percentage = 0.8)
  (h3 : new_car_purchase_percentage = 0.9)
  (h4 : out_of_pocket = 11000)
  : ∃ (sticker_price : ℝ), 
    sticker_price * new_car_purchase_percentage - old_car_value * old_car_sale_percentage = out_of_pocket ∧ 
    sticker_price = 30000 := by
  sorry

end NUMINAMATH_CALUDE_new_car_sticker_price_l4071_407168


namespace NUMINAMATH_CALUDE_x_value_l4071_407175

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4071_407175


namespace NUMINAMATH_CALUDE_sons_age_l4071_407152

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l4071_407152


namespace NUMINAMATH_CALUDE_dice_sum_probability_l4071_407193

theorem dice_sum_probability : 
  let die := Finset.range 6
  let outcomes := die.product die
  let favorable_outcomes := outcomes.filter (fun (x, y) => x + y + 2 ≥ 10)
  (favorable_outcomes.card : ℚ) / outcomes.card = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l4071_407193


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l4071_407184

def is_composite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p < 20 → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529 ∧ has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l4071_407184


namespace NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l4071_407191

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (|x + y| / 2) + |x - y| = 1

-- Define a rhombus
def is_rhombus (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  S = {(x, y) | |x| / a + |y| / b = 1}

-- Define the set of points satisfying the curve equation
def curve_set : Set (ℝ × ℝ) :=
  {(x, y) | curve_equation x y}

-- Theorem statement
theorem curve_is_rhombus_not_square :
  is_rhombus curve_set ∧ ¬(∃ (a : ℝ), curve_set = {(x, y) | |x| / a + |y| / a = 1}) :=
sorry

end NUMINAMATH_CALUDE_curve_is_rhombus_not_square_l4071_407191


namespace NUMINAMATH_CALUDE_exist_three_distinct_digits_forming_squares_l4071_407192

/-- A function that constructs a three-digit number from three digits -/
def threeDigitNumber (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

/-- Theorem stating the existence of three distinct digits forming squares -/
theorem exist_three_distinct_digits_forming_squares :
  ∃ (A B C : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∃ (x y z : Nat),
      threeDigitNumber A B C = x^2 ∧
      threeDigitNumber C B A = y^2 ∧
      threeDigitNumber C A B = z^2 :=
by
  sorry

#eval threeDigitNumber 9 6 1
#eval threeDigitNumber 1 6 9
#eval threeDigitNumber 1 9 6

end NUMINAMATH_CALUDE_exist_three_distinct_digits_forming_squares_l4071_407192


namespace NUMINAMATH_CALUDE_distance_ratio_in_pyramid_l4071_407142

/-- A regular square pyramid with vertex P and base ABCD -/
structure RegularSquarePyramid where
  base_side_length : ℝ
  height : ℝ

/-- A point inside the base of the pyramid -/
structure PointInBase where
  x : ℝ
  y : ℝ

/-- Sum of distances from a point to all faces of the pyramid -/
def sum_distances_to_faces (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- Sum of distances from a point to all edges of the base -/
def sum_distances_to_base_edges (p : RegularSquarePyramid) (e : PointInBase) : ℝ := sorry

/-- The main theorem stating the ratio of distances -/
theorem distance_ratio_in_pyramid (p : RegularSquarePyramid) (e : PointInBase) 
  (h_centroid : e ≠ PointInBase.mk (p.base_side_length / 2) (p.base_side_length / 2)) :
  sum_distances_to_faces p e / sum_distances_to_base_edges p e = 
    8 * Real.sqrt (p.height^2 + p.base_side_length^2 / 2) / p.base_side_length := by
  sorry

end NUMINAMATH_CALUDE_distance_ratio_in_pyramid_l4071_407142


namespace NUMINAMATH_CALUDE_polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l4071_407118

/-- A real coefficient polynomial f(x) = x^4 + (a-1)x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + (a-1)*x^2 + 1

/-- Theorem: If f(x) is non-negative for all real x, then a ≥ -1 -/
theorem polynomial_nonnegative_implies_a_range (a : ℝ) 
  (h : ∀ x : ℝ, f a x ≥ 0) : a ≥ -1 := by
  sorry

/-- Theorem: If a ≥ -1, then f(x) is non-negative for all real x -/
theorem a_range_implies_polynomial_nonnegative (a : ℝ) 
  (h : a ≥ -1) : ∀ x : ℝ, f a x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_nonnegative_implies_a_range_a_range_implies_polynomial_nonnegative_l4071_407118


namespace NUMINAMATH_CALUDE_pond_length_l4071_407173

/-- Given a rectangular field and a square pond, prove the length of the pond -/
theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_length : ℝ) : 
  field_length = 16 →
  field_length = 2 * field_width →
  pond_length ^ 2 = (field_length * field_width) / 2 →
  pond_length = 8 := by
sorry

end NUMINAMATH_CALUDE_pond_length_l4071_407173


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l4071_407101

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Part 1: Solution set for f(x) > 1 when a = -2
theorem solution_set_part1 :
  {x : ℝ | f (-2) x > 1} = {x : ℝ | x < -3 ∨ x > 1} := by sorry

-- Part 2: Range of a when f(x) > 0 for all x ∈ [1, +∞)
theorem range_of_a_part2 :
  (∀ x : ℝ, x ≥ 1 → f a x > 0) ↔ a > -3 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l4071_407101


namespace NUMINAMATH_CALUDE_projection_magnitude_l4071_407120

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (k, 3)

theorem projection_magnitude (k : ℝ) 
  (h : (a.1 + (b k).1, a.2 + (b k).2) • a = 0) : 
  |(a.1 * (b k).1 + a.2 * (b k).2) / Real.sqrt ((b k).1^2 + (b k).2^2)| = 1 :=
sorry

end NUMINAMATH_CALUDE_projection_magnitude_l4071_407120


namespace NUMINAMATH_CALUDE_lcm_of_72_108_126_156_l4071_407170

theorem lcm_of_72_108_126_156 : Nat.lcm 72 (Nat.lcm 108 (Nat.lcm 126 156)) = 19656 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_72_108_126_156_l4071_407170


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l4071_407145

theorem negative_fraction_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l4071_407145


namespace NUMINAMATH_CALUDE_polygon_area_is_two_l4071_407160

/-- Represents a point in 2D space -/
structure Point where
  x : ℤ
  y : ℤ

/-- Calculates the area of a polygon given its vertices -/
noncomputable def polygonArea (vertices : List Point) : ℚ :=
  sorry

/-- The list of vertices of the polygon -/
def polygonVertices : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 1⟩, ⟨2, 0⟩, ⟨3, 0⟩, ⟨3, 1⟩,
  ⟨3, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨1, 2⟩, ⟨0, 2⟩, ⟨0, 1⟩
]

/-- The theorem stating that the area of the polygon is 2 square units -/
theorem polygon_area_is_two :
  polygonArea polygonVertices = 2 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_is_two_l4071_407160


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4071_407130

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2 * a^2 - 5 * a + a^2 + 4 * a - 3 * a^2 = -a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : 2 * (a^2 + 3 * b^3) - (1/3) * (9 * a^2 - 12 * b^3) = -a^2 + 10 * b^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l4071_407130


namespace NUMINAMATH_CALUDE_initial_puppies_count_l4071_407158

/-- The number of puppies Sandy's dog had initially -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has now -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 :=
by
  sorry

#check initial_puppies_count

end NUMINAMATH_CALUDE_initial_puppies_count_l4071_407158


namespace NUMINAMATH_CALUDE_edmund_normal_chores_l4071_407162

/-- The number of chores Edmund normally has to do in a week -/
def normal_chores : ℕ := sorry

/-- The number of chores Edmund does per day -/
def chores_per_day : ℕ := 4

/-- The number of days Edmund works -/
def work_days : ℕ := 14

/-- The total amount Edmund earns -/
def total_earnings : ℕ := 64

/-- The payment per extra chore -/
def payment_per_chore : ℕ := 2

theorem edmund_normal_chores :
  normal_chores = 12 :=
by sorry

end NUMINAMATH_CALUDE_edmund_normal_chores_l4071_407162


namespace NUMINAMATH_CALUDE_button_probability_l4071_407111

def initial_red_c : ℕ := 6
def initial_green_c : ℕ := 12
def initial_total_c : ℕ := initial_red_c + initial_green_c

def remaining_fraction : ℚ := 3/4

theorem button_probability : 
  ∃ (removed_red removed_green : ℕ),
    removed_red = removed_green ∧
    initial_total_c - (removed_red + removed_green) = (remaining_fraction * initial_total_c).num ∧
    (initial_green_c - removed_green : ℚ) / (initial_total_c - (removed_red + removed_green) : ℚ) *
    (removed_green : ℚ) / ((removed_red + removed_green) : ℚ) = 5/14 :=
by sorry

end NUMINAMATH_CALUDE_button_probability_l4071_407111


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l4071_407169

theorem regular_polygon_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → (exterior_angle = 30 * Real.pi / 180) → (n * exterior_angle = 2 * Real.pi) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l4071_407169


namespace NUMINAMATH_CALUDE_simplify_expression_l4071_407147

theorem simplify_expression (a b : ℝ) : 105*a - 38*a + 27*b - 12*b = 67*a + 15*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4071_407147


namespace NUMINAMATH_CALUDE_calculation_proof_l4071_407194

theorem calculation_proof : (42 / (12 - 10 + 3)) ^ 2 * 7 = 493.92 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4071_407194


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_proof_l4071_407100

def initial_pens : ℕ := 7
def mikes_pens : ℕ := 22
def final_pens : ℕ := 39

def pens_given_to_sharon : ℕ := 19

theorem pens_given_to_sharon_proof :
  ((initial_pens + mikes_pens) * 2) - final_pens = pens_given_to_sharon := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_proof_l4071_407100


namespace NUMINAMATH_CALUDE_queen_then_spade_probability_l4071_407179

-- Define the total number of cards in a standard deck
def totalCards : ℕ := 52

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the number of spades in a standard deck
def numSpades : ℕ := 13

-- Define the probability of drawing a Queen as the first card and a spade as the second card
def probQueenThenSpade : ℚ := numQueens / totalCards * numSpades / (totalCards - 1)

-- Theorem statement
theorem queen_then_spade_probability :
  probQueenThenSpade = 4 / 17 := by
  sorry

end NUMINAMATH_CALUDE_queen_then_spade_probability_l4071_407179


namespace NUMINAMATH_CALUDE_continuity_at_6_l4071_407178

def f (x : ℝ) := 5 * x^2 - 1

theorem continuity_at_6 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 6| < δ → |f x - f 6| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_6_l4071_407178


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_is_two_l4071_407106

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_water_speed : ℝ) (upstream_downstream_ratio : ℝ) : ℝ :=
  let stream_speed := 2
  let downstream_speed := still_water_speed + stream_speed
  let upstream_speed := still_water_speed - stream_speed
  have h1 : still_water_speed = 6 := by sorry
  have h2 : upstream_downstream_ratio = 2 := by sorry
  have h3 : (1 / upstream_speed) = upstream_downstream_ratio * (1 / downstream_speed) := by sorry
  stream_speed

/-- The main theorem stating that the stream speed is 2 km/h -/
theorem stream_speed_is_two : stream_speed 6 2 = 2 := by sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_is_two_l4071_407106


namespace NUMINAMATH_CALUDE_car_value_reduction_l4071_407109

-- Define the original price of the car
def original_price : ℝ := 4000

-- Define the reduction rate
def reduction_rate : ℝ := 0.30

-- Define the current value of the car
def current_value : ℝ := original_price * (1 - reduction_rate)

-- Theorem to prove
theorem car_value_reduction : current_value = 2800 := by
  sorry

end NUMINAMATH_CALUDE_car_value_reduction_l4071_407109


namespace NUMINAMATH_CALUDE_rice_profit_calculation_l4071_407166

/-- Calculates the profit from selling a sack of rice -/
theorem rice_profit_calculation (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) : 
  weight = 50 ∧ cost = 50 ∧ price_per_kg = 1.2 → price_per_kg * weight - cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_profit_calculation_l4071_407166


namespace NUMINAMATH_CALUDE_min_distance_complex_circle_l4071_407172

theorem min_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_complex_circle_l4071_407172


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l4071_407153

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l4071_407153


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l4071_407135

/-- If the side of a square is measured with a 2% excess error, 
    then the percentage of error in the calculated area of the square is 4.04%. -/
theorem square_area_error_percentage (s : ℝ) (s' : ℝ) (A : ℝ) (A' : ℝ) :
  s' = s * (1 + 0.02) →
  A = s^2 →
  A' = s'^2 →
  (A' - A) / A * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_percentage_l4071_407135


namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l4071_407124

/-- Proves that Yolanda's walking rate is 3 miles per hour given the problem conditions -/
theorem yolandas_walking_rate 
  (total_distance : ℝ) 
  (bobs_delay : ℝ) 
  (bobs_rate : ℝ) 
  (bobs_distance : ℝ) 
  (h1 : total_distance = 24)
  (h2 : bobs_delay = 1)
  (h3 : bobs_rate = 4)
  (h4 : bobs_distance = 12) : 
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + bobs_delay) = 3 := by
  sorry

#check yolandas_walking_rate

end NUMINAMATH_CALUDE_yolandas_walking_rate_l4071_407124


namespace NUMINAMATH_CALUDE_largest_share_proof_l4071_407161

def profit_distribution (ratio : List Nat) (total_profit : Nat) : List Nat :=
  let total_parts := ratio.sum
  let part_value := total_profit / total_parts
  ratio.map (· * part_value)

theorem largest_share_proof (ratio : List Nat) (total_profit : Nat) :
  ratio = [3, 3, 4, 5, 6] → total_profit = 42000 →
  (profit_distribution ratio total_profit).maximum? = some 12000 := by
  sorry

end NUMINAMATH_CALUDE_largest_share_proof_l4071_407161


namespace NUMINAMATH_CALUDE_vocabulary_test_score_l4071_407122

theorem vocabulary_test_score (total_words : ℕ) (target_score : ℚ) 
  (h1 : total_words = 600) 
  (h2 : target_score = 90 / 100) : 
  ∃ (words_to_learn : ℕ), 
    (words_to_learn : ℚ) / total_words = target_score ∧ 
    words_to_learn = 540 := by
  sorry

end NUMINAMATH_CALUDE_vocabulary_test_score_l4071_407122


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l4071_407139

/-- The foci coordinates of the hyperbola x^2/4 - y^2 = 1 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  ∃ (c : ℝ), c^2 = 5 ∧ 
    (∀ (x y : ℝ), (x, y) ∈ hyperbola → 
      ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l4071_407139


namespace NUMINAMATH_CALUDE_building_entrances_l4071_407123

/-- Represents a building with multiple entrances -/
structure Building where
  floors : ℕ
  apartments_per_floor : ℕ
  total_apartments : ℕ

/-- Calculates the number of entrances in a building -/
def number_of_entrances (b : Building) : ℕ :=
  b.total_apartments / (b.floors * b.apartments_per_floor)

/-- Theorem stating the number of entrances in the specific building -/
theorem building_entrances :
  let b : Building := {
    floors := 9,
    apartments_per_floor := 4,
    total_apartments := 180
  }
  number_of_entrances b = 5 := by
  sorry

end NUMINAMATH_CALUDE_building_entrances_l4071_407123


namespace NUMINAMATH_CALUDE_waiter_tables_problem_l4071_407127

theorem waiter_tables_problem (initial_tables : ℝ) : 
  (initial_tables - 12.0) * 8.0 = 256 → initial_tables = 44.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_problem_l4071_407127


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l4071_407188

-- Define the circle's center
structure CircleCenter where
  x : ℝ
  y : ℝ

-- Define the conditions for the circle's center
def satisfiesConditions (c : CircleCenter) : Prop :=
  c.x - 2 * c.y = 0 ∧
  3 * c.x - 4 * c.y = 10

-- Theorem statement
theorem circle_center_coordinates :
  ∃ (c : CircleCenter), satisfiesConditions c ∧ c.x = 10 ∧ c.y = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l4071_407188


namespace NUMINAMATH_CALUDE_pencil_notebook_cost_l4071_407163

/-- The cost of pencils and notebooks given specific conditions --/
theorem pencil_notebook_cost :
  ∀ (p n : ℝ),
  -- Condition 1: 9 pencils and 10 notebooks cost $5.35
  9 * p + 10 * n = 5.35 →
  -- Condition 2: 6 pencils and 4 notebooks cost $2.50
  6 * p + 4 * n = 2.50 →
  -- The cost of 24 pencils (with 10% discount on packs of 4) and 15 notebooks is $9.24
  24 * (0.9 * p) + 15 * n = 9.24 :=
by
  sorry


end NUMINAMATH_CALUDE_pencil_notebook_cost_l4071_407163


namespace NUMINAMATH_CALUDE_waiter_shift_earnings_l4071_407199

/-- Calculates the waiter's earnings during a shift --/
def waiter_earnings (total_customers : ℕ) 
                    (three_dollar_tippers : ℕ) 
                    (four_fifty_tippers : ℕ) 
                    (non_tippers : ℕ) 
                    (tip_pool_contribution : ℚ) 
                    (meal_cost : ℚ) : ℚ :=
  (3 * three_dollar_tippers + 4.5 * four_fifty_tippers) - tip_pool_contribution - meal_cost

theorem waiter_shift_earnings :
  waiter_earnings 15 6 4 5 10 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_waiter_shift_earnings_l4071_407199


namespace NUMINAMATH_CALUDE_emily_elephant_four_hops_l4071_407138

/-- The distance covered in a single hop, given the remaining distance to the target -/
def hop_distance (remaining : ℚ) : ℚ := (1 / 4) * remaining

/-- The remaining distance to the target after a hop -/
def remaining_after_hop (remaining : ℚ) : ℚ := remaining - hop_distance remaining

/-- The total distance covered after n hops -/
def total_distance (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) (remaining : ℚ) (acc : ℚ) : ℚ :=
    if k = 0 then acc
    else aux (k - 1) (remaining_after_hop remaining) (acc + hop_distance remaining)
  aux n 1 0

theorem emily_elephant_four_hops :
  total_distance 4 = 175 / 256 := by sorry

end NUMINAMATH_CALUDE_emily_elephant_four_hops_l4071_407138


namespace NUMINAMATH_CALUDE_damien_picked_fraction_l4071_407195

/-- Proves that Damien picked 3/5 of the fruits from the trees --/
theorem damien_picked_fraction (apples plums : ℕ) (picked_fraction : ℚ) : 
  apples = 3 * plums →  -- The number of apples is three times the number of plums
  apples = 180 →  -- The initial number of apples is 180
  (1 - picked_fraction) * (apples + plums) = 96 →  -- After picking, 96 fruits remain
  picked_fraction = 3 / 5 := by
sorry


end NUMINAMATH_CALUDE_damien_picked_fraction_l4071_407195


namespace NUMINAMATH_CALUDE_lukes_final_balance_l4071_407174

/-- Calculates Luke's final balance after six months of financial activities --/
def lukesFinalBalance (initialAmount : ℝ) (februarySpendingRate : ℝ) 
  (marchSpending marchIncome : ℝ) (monthlyPiggyBankRate : ℝ) : ℝ :=
  let afterFebruary := initialAmount * (1 - februarySpendingRate)
  let afterMarch := afterFebruary - marchSpending + marchIncome
  let afterApril := afterMarch * (1 - monthlyPiggyBankRate)
  let afterMay := afterApril * (1 - monthlyPiggyBankRate)
  let afterJune := afterMay * (1 - monthlyPiggyBankRate)
  afterJune

/-- Theorem stating Luke's final balance after six months --/
theorem lukes_final_balance :
  lukesFinalBalance 48 0.3 11 21 0.1 = 31.79 := by
  sorry

end NUMINAMATH_CALUDE_lukes_final_balance_l4071_407174


namespace NUMINAMATH_CALUDE_mikes_score_l4071_407177

def passing_threshold : ℝ := 0.30
def max_score : ℕ := 770
def shortfall : ℕ := 19

theorem mikes_score : 
  ⌊(passing_threshold * max_score : ℝ)⌋ - shortfall = 212 := by
  sorry

end NUMINAMATH_CALUDE_mikes_score_l4071_407177


namespace NUMINAMATH_CALUDE_self_reciprocal_set_l4071_407104

def self_reciprocal (x : ℝ) : Prop := x ≠ 0 ∧ x = 1 / x

theorem self_reciprocal_set :
  ∃ (S : Set ℝ), (∀ x, x ∈ S ↔ self_reciprocal x) ∧ S = {1, -1} :=
sorry

end NUMINAMATH_CALUDE_self_reciprocal_set_l4071_407104


namespace NUMINAMATH_CALUDE_P_in_first_quadrant_l4071_407167

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point P(3,2) -/
def P : CartesianPoint :=
  { x := 3, y := 2 }

/-- Theorem: Point P(3,2) lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry


end NUMINAMATH_CALUDE_P_in_first_quadrant_l4071_407167


namespace NUMINAMATH_CALUDE_smallest_divisible_number_proof_l4071_407126

/-- The smallest 5-digit number divisible by 15, 32, 45, and a multiple of 9 and 6 -/
def smallest_divisible_number : ℕ := 11520

theorem smallest_divisible_number_proof :
  smallest_divisible_number ≥ 10000 ∧
  smallest_divisible_number < 100000 ∧
  smallest_divisible_number % 15 = 0 ∧
  smallest_divisible_number % 32 = 0 ∧
  smallest_divisible_number % 45 = 0 ∧
  smallest_divisible_number % 9 = 0 ∧
  smallest_divisible_number % 6 = 0 ∧
  ∀ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧
    n % 15 = 0 ∧ n % 32 = 0 ∧ n % 45 = 0 ∧ n % 9 = 0 ∧ n % 6 = 0 →
    n ≥ smallest_divisible_number :=
by sorry

#eval smallest_divisible_number

end NUMINAMATH_CALUDE_smallest_divisible_number_proof_l4071_407126


namespace NUMINAMATH_CALUDE_circle_through_two_points_tangent_to_line_l4071_407141

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the condition for a point to be on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the condition for a circle to pass through a point
def circlePassesThroughPoint (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

-- Define the condition for a circle to be tangent to a line
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  ∃ p : Point, pointOnLine p l ∧ circlePassesThroughPoint c p ∧
  ∀ q : Point, pointOnLine q l → (c.center.x - q.x)^2 + (c.center.y - q.y)^2 ≥ c.radius^2

-- Theorem statement
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) : 
  ∃ c : Circle, circlePassesThroughPoint c A ∧ 
                circlePassesThroughPoint c B ∧ 
                circleTangentToLine c l :=
sorry

end NUMINAMATH_CALUDE_circle_through_two_points_tangent_to_line_l4071_407141


namespace NUMINAMATH_CALUDE_vector_sum_equality_implies_same_direction_l4071_407196

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def same_direction (a b : n) : Prop := ∃ (k : ℝ), k > 0 ∧ a = k • b

theorem vector_sum_equality_implies_same_direction (a b : n) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖a‖ + ‖b‖) :
  same_direction a b := by sorry

end NUMINAMATH_CALUDE_vector_sum_equality_implies_same_direction_l4071_407196


namespace NUMINAMATH_CALUDE_distance_between_foci_l4071_407131

def ellipse (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y + 9)^2) = 22

def focus1 : ℝ × ℝ := (4, -5)
def focus2 : ℝ × ℝ := (-6, 9)

theorem distance_between_foci :
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 74 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l4071_407131


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l4071_407102

/-- The perimeter of a hexagon with side length 4 inches is 24 inches. -/
theorem hexagon_perimeter (side_length : ℝ) : side_length = 4 → 6 * side_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l4071_407102


namespace NUMINAMATH_CALUDE_simplify_and_sum_fraction_l4071_407136

theorem simplify_and_sum_fraction : ∃ (a b : ℕ), 
  (75 : ℚ) / 100 = (a : ℚ) / b ∧ 
  (∀ (c d : ℕ), (75 : ℚ) / 100 = (c : ℚ) / d → a ≤ c ∧ b ≤ d) ∧
  a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_sum_fraction_l4071_407136


namespace NUMINAMATH_CALUDE_largest_c_for_g_range_two_l4071_407155

/-- The quadratic function g(x) = x^2 - 6x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- Theorem: The largest value of c for which 2 is in the range of g(x) is 11 -/
theorem largest_c_for_g_range_two :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 2) ↔ c ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_g_range_two_l4071_407155


namespace NUMINAMATH_CALUDE_number_of_students_l4071_407165

theorem number_of_students (n : ℕ) : 
  n % 5 = 0 ∧ 
  ∃ t : ℕ, (n + 1) * t = 527 ∧
  (n + 1) ∣ 527 ∧
  (n + 1) % 5 = 1 →
  n = 30 := by
sorry

end NUMINAMATH_CALUDE_number_of_students_l4071_407165


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l4071_407181

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l4071_407181


namespace NUMINAMATH_CALUDE_chris_birthday_money_l4071_407116

theorem chris_birthday_money (x : ℕ) : 
  x + 25 + 20 + 75 = 279 → x = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_birthday_money_l4071_407116


namespace NUMINAMATH_CALUDE_car_race_bet_l4071_407149

theorem car_race_bet (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (winning_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  karen_delay = 4 / 60 →
  winning_margin = 4 →
  ∃ w : ℝ, w = 8 / 3 ∧ 
    karen_speed * (w / tom_speed - karen_delay) = w + winning_margin :=
by sorry

end NUMINAMATH_CALUDE_car_race_bet_l4071_407149


namespace NUMINAMATH_CALUDE_constant_term_in_system_of_equations_l4071_407156

theorem constant_term_in_system_of_equations :
  ∀ (x y k : ℝ),
  (7 * x + y = 19) →
  (x + 3 * y = k) →
  (2 * x + y = 5) →
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_constant_term_in_system_of_equations_l4071_407156


namespace NUMINAMATH_CALUDE_circle_radius_is_six_l4071_407176

theorem circle_radius_is_six (r : ℝ) : r > 0 → 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_six_l4071_407176
