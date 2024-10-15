import Mathlib

namespace NUMINAMATH_CALUDE_cosine_sum_inequality_l1546_154654

theorem cosine_sum_inequality (x y z : ℝ) (h : x + y + z = 0) :
  |Real.cos x| + |Real.cos y| + |Real.cos z| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_inequality_l1546_154654


namespace NUMINAMATH_CALUDE_at_least_one_parabola_has_two_roots_l1546_154657

-- Define the parabolas
def parabola1 (a b c x : ℝ) : ℝ := a * x^2 + 2 * b * x + c
def parabola2 (a b c x : ℝ) : ℝ := b * x^2 + 2 * c * x + a
def parabola3 (a b c x : ℝ) : ℝ := c * x^2 + 2 * a * x + b

-- Define a function to check if a parabola has two distinct roots
def has_two_distinct_roots (f : ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0

-- State the theorem
theorem at_least_one_parabola_has_two_roots (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  has_two_distinct_roots (parabola1 a b c) ∨ 
  has_two_distinct_roots (parabola2 a b c) ∨ 
  has_two_distinct_roots (parabola3 a b c) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_parabola_has_two_roots_l1546_154657


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l1546_154608

theorem opera_ticket_price_increase (initial_price new_price : ℝ) 
  (h1 : initial_price = 85)
  (h2 : new_price = 102) :
  (new_price - initial_price) / initial_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l1546_154608


namespace NUMINAMATH_CALUDE_mono_decreasing_inequality_l1546_154660

/-- A function f is monotonically decreasing on ℝ -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a monotonically decreasing function f on ℝ,
    if f(2m) > f(1+m), then m < 1 -/
theorem mono_decreasing_inequality (f : ℝ → ℝ) (m : ℝ)
    (h_mono : MonotonicallyDecreasing f) (h_ineq : f (2 * m) > f (1 + m)) :
    m < 1 :=
  sorry

end NUMINAMATH_CALUDE_mono_decreasing_inequality_l1546_154660


namespace NUMINAMATH_CALUDE_biker_passes_l1546_154672

/-- Represents a biker's total travels along the road -/
structure BikerTravel where
  travels : ℕ

/-- Represents the scenario of two bikers on a road -/
structure BikerScenario where
  biker1 : BikerTravel
  biker2 : BikerTravel

/-- Calculates the number of passes between two bikers -/
def calculatePasses (scenario : BikerScenario) : ℕ :=
  sorry

theorem biker_passes (scenario : BikerScenario) :
  scenario.biker1.travels = 11 →
  scenario.biker2.travels = 7 →
  calculatePasses scenario = 8 :=
sorry

end NUMINAMATH_CALUDE_biker_passes_l1546_154672


namespace NUMINAMATH_CALUDE_optimal_advertising_strategy_l1546_154696

/-- Sales revenue function -/
def R (x₁ x₂ : ℝ) : ℝ := -2 * x₁^2 - x₂^2 + 13 * x₁ + 11 * x₂ - 28

/-- Profit function -/
def profit (x₁ x₂ : ℝ) : ℝ := R x₁ x₂ - x₁ - x₂

theorem optimal_advertising_strategy :
  (∃ (x₁ x₂ : ℝ), x₁ + x₂ = 5 ∧ 
    ∀ (y₁ y₂ : ℝ), y₁ + y₂ = 5 → profit x₁ x₂ ≥ profit y₁ y₂) ∧
  profit 2 3 = 9 ∧
  (∀ (y₁ y₂ : ℝ), profit 3 5 ≥ profit y₁ y₂) ∧
  profit 3 5 = 15 := by sorry

end NUMINAMATH_CALUDE_optimal_advertising_strategy_l1546_154696


namespace NUMINAMATH_CALUDE_problem_solution_l1546_154633

theorem problem_solution : ∃! x : ℝ, (0.8 * x) = ((4 / 5) * 25 + 28) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1546_154633


namespace NUMINAMATH_CALUDE_santa_mandarins_l1546_154669

/-- Represents the exchange game with Santa Claus --/
structure ExchangeGame where
  /-- Number of first type exchanges (5 mandarins for 3 firecrackers and 1 candy) --/
  first_exchanges : ℕ
  /-- Number of second type exchanges (2 firecrackers for 3 mandarins and 1 candy) --/
  second_exchanges : ℕ
  /-- Total number of candies received --/
  total_candies : ℕ
  /-- Constraint: Total exchanges equal total candies --/
  exchanges_eq_candies : first_exchanges + second_exchanges = total_candies
  /-- Constraint: Firecrackers balance out --/
  firecrackers_balance : 3 * first_exchanges = 2 * second_exchanges

/-- The main theorem to prove --/
theorem santa_mandarins (game : ExchangeGame) (h : game.total_candies = 50) :
  5 * game.first_exchanges - 3 * game.second_exchanges = 10 := by
  sorry

end NUMINAMATH_CALUDE_santa_mandarins_l1546_154669


namespace NUMINAMATH_CALUDE_decomposition_fifth_power_fourth_l1546_154673

-- Define the function that gives the starting odd number for m^n
def startOdd (m n : ℕ) : ℕ := 
  2 * (m - 1) * (n - 1) + 1

-- Define the function that gives the k-th odd number in the sequence
def kthOdd (start k : ℕ) : ℕ := 
  start + 2 * (k - 1)

-- Theorem statement
theorem decomposition_fifth_power_fourth (m : ℕ) (h : m = 5) : 
  kthOdd (startOdd m 4) 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_decomposition_fifth_power_fourth_l1546_154673


namespace NUMINAMATH_CALUDE_product_probabilities_l1546_154628

/-- The probability of a product having a defect -/
def p₁ : ℝ := 0.1

/-- The probability of the controller detecting an existing defect -/
def p₂ : ℝ := 0.8

/-- The probability of the controller mistakenly rejecting a non-defective product -/
def p₃ : ℝ := 0.3

/-- The probability of a product being mistakenly rejected -/
def P_A₁ : ℝ := (1 - p₁) * p₃

/-- The probability of a product being passed into finished goods with a defect -/
def P_A₂ : ℝ := p₁ * (1 - p₂)

/-- The probability of a product being rejected -/
def P_A₃ : ℝ := p₁ * p₂ + (1 - p₁) * p₃

theorem product_probabilities :
  P_A₁ = 0.27 ∧ P_A₂ = 0.02 ∧ P_A₃ = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_product_probabilities_l1546_154628


namespace NUMINAMATH_CALUDE_fraction_problem_l1546_154607

theorem fraction_problem : 
  ∃ (x y : ℚ), x / y > 0 ∧ y ≠ 0 ∧ ((377 / 13) / 29) * (x / y) / 2 = 1 / 8 ∧ x / y = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l1546_154607


namespace NUMINAMATH_CALUDE_pauls_journey_time_l1546_154620

theorem pauls_journey_time (paul_time : ℝ) : 
  (paul_time + 7 * (paul_time + 2) = 46) → paul_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_pauls_journey_time_l1546_154620


namespace NUMINAMATH_CALUDE_product_mod_23_l1546_154667

theorem product_mod_23 : (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_23_l1546_154667


namespace NUMINAMATH_CALUDE_kite_coefficient_sum_l1546_154677

/-- Represents a parabola in the form y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- Represents a kite formed by the intersection of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola
  area : ℝ

/-- The sum of coefficients a and b for two parabolas forming a kite with area 12 -/
def coefficient_sum (k : Kite) : ℝ := k.p1.a + (-k.p2.a)

/-- Theorem stating that the sum of coefficients a and b is 1.5 for the given conditions -/
theorem kite_coefficient_sum :
  ∀ (k : Kite),
    k.p1.c = -2 ∧ 
    k.p2.c = 4 ∧ 
    k.area = 12 →
    coefficient_sum k = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_kite_coefficient_sum_l1546_154677


namespace NUMINAMATH_CALUDE_tangent_and_roots_l1546_154602

noncomputable section

def F (x : ℝ) := x * Real.log x

def tangent_line (x y : ℝ) := 2 * x - y - Real.exp 1 = 0

def has_two_roots (t : ℝ) :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    Real.exp (-2) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧
    F x₁ = t ∧ F x₂ = t

theorem tangent_and_roots :
  (∀ x y, F x = y → x = Real.exp 1 → tangent_line x y) ∧
  (∀ t, has_two_roots t ↔ -Real.exp (-1) < t ∧ t ≤ -2 * Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_roots_l1546_154602


namespace NUMINAMATH_CALUDE_max_product_sum_300_l1546_154663

theorem max_product_sum_300 :
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 ∧ ∃ a b : ℤ, a + b = 300 ∧ a * b = 22500 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l1546_154663


namespace NUMINAMATH_CALUDE_no_equal_sums_for_given_sequences_l1546_154671

theorem no_equal_sums_for_given_sequences : ¬ ∃ (n : ℕ), n > 0 ∧
  (let a₁ := 9
   let d₁ := 6
   let t₁ := n * (2 * a₁ + (n - 1) * d₁) / 2
   let a₂ := 11
   let d₂ := 3
   let t₂ := n * (2 * a₂ + (n - 1) * d₂) / 2
   t₁ = t₂) :=
sorry

end NUMINAMATH_CALUDE_no_equal_sums_for_given_sequences_l1546_154671


namespace NUMINAMATH_CALUDE_article_cost_l1546_154611

/-- Proves that if selling an article for 350 gains 5% more than selling it for 340, then the cost is 140 -/
theorem article_cost (sell_price_high : ℝ) (sell_price_low : ℝ) (cost : ℝ) :
  sell_price_high = 350 ∧
  sell_price_low = 340 ∧
  (sell_price_high - cost) = (sell_price_low - cost) * 1.05 →
  cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1546_154611


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1546_154640

/-- The constant term in the expansion of (√x + 3/x)^12 -/
def constantTerm : ℕ := 40095

/-- The binomial coefficient (12 choose 8) -/
def binomialCoeff : ℕ := 495

theorem constant_term_expansion :
  constantTerm = binomialCoeff * 3^4 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1546_154640


namespace NUMINAMATH_CALUDE_basketball_score_l1546_154682

theorem basketball_score (three_pointers two_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (two_pointers = 2 * free_throws) →
  (3 * three_pointers + 2 * two_pointers + free_throws = 73) →
  free_throws = 8 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_l1546_154682


namespace NUMINAMATH_CALUDE_problem_statement_l1546_154694

theorem problem_statement (a b c : ℝ) 
  (h1 : a * b * c ≠ 0) 
  (h2 : a + b + c = 2) 
  (h3 : a^2 + b^2 + c^2 = 2) : 
  (1 - a)^2 / (b * c) + (1 - b)^2 / (c * a) + (1 - c)^2 / (a * b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1546_154694


namespace NUMINAMATH_CALUDE_cosine_sine_sum_equals_sqrt_two_over_two_l1546_154695

theorem cosine_sine_sum_equals_sqrt_two_over_two : 
  Real.cos (70 * π / 180) * Real.cos (335 * π / 180) + 
  Real.sin (110 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sine_sum_equals_sqrt_two_over_two_l1546_154695


namespace NUMINAMATH_CALUDE_parabola_translation_l1546_154697

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically by a given amount -/
def translateVertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + dy }

/-- Translates a parabola horizontally by a given amount -/
def translateHorizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * dx, c := p.c + p.a * dx^2 - p.b * dx }

theorem parabola_translation (p : Parabola) :
  p.a = 1 ∧ p.b = -2 ∧ p.c = 4 →
  let p' := translateHorizontal (translateVertical p 3) 1
  p'.a = 1 ∧ p'.b = -4 ∧ p'.c = 10 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1546_154697


namespace NUMINAMATH_CALUDE_price_increase_problem_l1546_154647

theorem price_increase_problem (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ) (chocolate_new : ℝ)
  (candy_increase : ℝ) (soda_increase : ℝ) (chips_increase : ℝ) (chocolate_increase : ℝ)
  (h_candy : candy_new = 10) (h_soda : soda_new = 6) (h_chips : chips_new = 4) (h_chocolate : chocolate_new = 2)
  (h_candy_inc : candy_increase = 0.25) (h_soda_inc : soda_increase = 0.5)
  (h_chips_inc : chips_increase = 0.4) (h_chocolate_inc : chocolate_increase = 0.75) :
  (candy_new / (1 + candy_increase)) + (soda_new / (1 + soda_increase)) +
  (chips_new / (1 + chips_increase)) + (chocolate_new / (1 + chocolate_increase)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_problem_l1546_154647


namespace NUMINAMATH_CALUDE_employee_payment_l1546_154636

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 260 :=
by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l1546_154636


namespace NUMINAMATH_CALUDE_cos_BAE_value_l1546_154648

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point E on BC
def E (triangle : Triangle) : ℝ × ℝ := sorry

-- Define the lengths of the sides
def AB (triangle : Triangle) : ℝ := 4
def AC (triangle : Triangle) : ℝ := 8
def BC (triangle : Triangle) : ℝ := 10

-- Define that AE bisects angle BAC
def AE_bisects_BAC (triangle : Triangle) : Prop := sorry

-- Define the cosine of angle BAE
def cos_BAE (triangle : Triangle) : ℝ := sorry

-- Theorem statement
theorem cos_BAE_value (triangle : Triangle) 
  (h1 : AB triangle = 4) 
  (h2 : AC triangle = 8) 
  (h3 : BC triangle = 10) 
  (h4 : AE_bisects_BAC triangle) : 
  cos_BAE triangle = Real.sqrt (11/32) := by
  sorry

end NUMINAMATH_CALUDE_cos_BAE_value_l1546_154648


namespace NUMINAMATH_CALUDE_symmetric_point_on_number_line_l1546_154623

/-- Given points A, B, and C on a number line, where A represents √7, B represents 1,
    and C is symmetric to A with respect to B, prove that C represents 2 - √7. -/
theorem symmetric_point_on_number_line (A B C : ℝ) : 
  A = Real.sqrt 7 → B = 1 → (A + C) / 2 = B → C = 2 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_on_number_line_l1546_154623


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1546_154643

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 3) * x^(m^2 - 7) + m*x - 2 = a*x^2 + b*x + c) ∧ 
  (m + 3 ≠ 0) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1546_154643


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1546_154688

/-- The line kx+y+2=0 intersects the circle (x-1)^2+(y+2)^2=16 for all real k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + 2 = 0) ∧ ((x - 1)^2 + (y + 2)^2 = 16) := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1546_154688


namespace NUMINAMATH_CALUDE_lcm_of_20_and_36_l1546_154649

theorem lcm_of_20_and_36 : Nat.lcm 20 36 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_and_36_l1546_154649


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1546_154689

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : 4 * a - 4 * b + c > 0) 
  (h2 : a + 2 * b + c < 0) : 
  b^2 > a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1546_154689


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1546_154634

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x : ℕ), x ≤ d - 1 ∧ (n - x) % d = 0 ∧ ∀ (y : ℕ), y < x → (n - y) % d ≠ 0 :=
sorry

theorem problem_solution :
  ∃ (x : ℕ), x = 8 ∧ (42398 - x) % 15 = 0 ∧ ∀ (y : ℕ), y < x → (42398 - y) % 15 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1546_154634


namespace NUMINAMATH_CALUDE_correct_operation_l1546_154625

theorem correct_operation (a : ℝ) : -a + 5*a = 4*a := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1546_154625


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1546_154666

theorem quadratic_function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f (f 0) = 0 ∧ f (f 1) = 0 ∧ f 0 ≠ f 1) → f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1546_154666


namespace NUMINAMATH_CALUDE_hcf_problem_l1546_154637

/-- Given two positive integers with HCF H and LCM (H * 13 * 14),
    where the larger number is 350, prove that H = 70 -/
theorem hcf_problem (a b : ℕ) (H : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 350)
  (h4 : H = Nat.gcd a b) (h5 : Nat.lcm a b = H * 13 * 14) : H = 70 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l1546_154637


namespace NUMINAMATH_CALUDE_quadratic_function_a_value_l1546_154642

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-value of a quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

/-- The x-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexX (f : QuadraticFunction) : ℚ := -f.b / (2 * f.a)

/-- The y-coordinate of the vertex of a quadratic function -/
def QuadraticFunction.vertexY (f : QuadraticFunction) : ℚ :=
  f.eval (f.vertexX)

theorem quadratic_function_a_value (f : QuadraticFunction) :
  f.vertexX = 2 ∧ f.vertexY = 5 ∧ f.eval 1 = 2 ∧ f.eval 3 = 2 → f.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_a_value_l1546_154642


namespace NUMINAMATH_CALUDE_absolute_value_subtraction_l1546_154621

theorem absolute_value_subtraction : 2 - |(-3)| = -1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_subtraction_l1546_154621


namespace NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l1546_154676

/-- Proves that the ratio of Sachin's age to Rahul's age is 7:9 given their age difference --/
theorem age_ratio_sachin_rahul :
  ∀ (sachin_age rahul_age : ℚ),
    sachin_age = 31.5 →
    rahul_age = sachin_age + 9 →
    ∃ (a b : ℕ), a = 7 ∧ b = 9 ∧ sachin_age / rahul_age = a / b := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_sachin_rahul_l1546_154676


namespace NUMINAMATH_CALUDE_equation_solutions_count_l1546_154630

theorem equation_solutions_count : 
  (Finset.filter (fun p : ℕ × ℕ => 4 * p.1 + 7 * p.2 = 588 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 589) (Finset.range 589))).card = 21 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l1546_154630


namespace NUMINAMATH_CALUDE_abs_value_complex_l1546_154639

/-- The absolute value of ((1+i)³)/2 is equal to √2 -/
theorem abs_value_complex : Complex.abs ((1 + Complex.I) ^ 3 / 2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_complex_l1546_154639


namespace NUMINAMATH_CALUDE_parabola_line_single_intersection_l1546_154670

/-- The value of a that makes the parabola y = ax^2 + 3x + 1 intersect
    the line y = -2x - 3 at only one point is 25/16 -/
theorem parabola_line_single_intersection :
  ∃! a : ℚ, ∀ x : ℚ,
    (a * x^2 + 3 * x + 1 = -2 * x - 3) →
    (∀ y : ℚ, y ≠ x → a * y^2 + 3 * y + 1 ≠ -2 * y - 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_single_intersection_l1546_154670


namespace NUMINAMATH_CALUDE_two_candles_burn_time_l1546_154612

/-- Burning time of candle 1 -/
def burn_time_1 : ℕ := 30

/-- Burning time of candle 2 -/
def burn_time_2 : ℕ := 40

/-- Burning time of candle 3 -/
def burn_time_3 : ℕ := 50

/-- Time all three candles burn simultaneously -/
def time_all_three : ℕ := 10

/-- Time only one candle burns -/
def time_one_candle : ℕ := 20

/-- Theorem stating that exactly two candles burn simultaneously for 35 minutes -/
theorem two_candles_burn_time :
  (burn_time_1 + burn_time_2 + burn_time_3) - (3 * time_all_three + time_one_candle) = 70 :=
by sorry

end NUMINAMATH_CALUDE_two_candles_burn_time_l1546_154612


namespace NUMINAMATH_CALUDE_min_value_theorem_l1546_154631

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1546_154631


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1546_154614

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) :
  b₁ = 2 → 
  b₂ = b₁ * s →
  b₃ = b₂ * s →
  (∃ (x : ℝ), 3 * b₂ + 4 * b₃ ≥ x) →
  (∀ (x : ℝ), (3 * b₂ + 4 * b₃ ≥ x) → x ≤ -9/8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1546_154614


namespace NUMINAMATH_CALUDE_log_equality_l1546_154698

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equality : log10 2 + 2 * log10 5 = 1 + log10 5 := by sorry

end NUMINAMATH_CALUDE_log_equality_l1546_154698


namespace NUMINAMATH_CALUDE_tenth_prime_l1546_154674

def is_prime (n : ℕ) : Prop := sorry

def nth_prime (n : ℕ) : ℕ := sorry

theorem tenth_prime :
  (nth_prime 5 = 11) → (nth_prime 10 = 29) := by sorry

end NUMINAMATH_CALUDE_tenth_prime_l1546_154674


namespace NUMINAMATH_CALUDE_crayon_count_l1546_154693

theorem crayon_count (initial_crayons added_crayons : ℕ) 
  (h1 : initial_crayons = 9)
  (h2 : added_crayons = 3) : 
  initial_crayons + added_crayons = 12 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l1546_154693


namespace NUMINAMATH_CALUDE_triangle_abc_obtuse_l1546_154624

theorem triangle_abc_obtuse (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * A → a = 1 → b = 4/3 → 0 < A → A < π → B > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_obtuse_l1546_154624


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l1546_154678

theorem seven_eighths_of_48 : (7 : ℚ) / 8 * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l1546_154678


namespace NUMINAMATH_CALUDE_f_fixed_points_l1546_154605

def f (x : ℝ) : ℝ := x^2 - 5*x

theorem f_fixed_points : 
  {x : ℝ | f (f x) = f x} = {0, -2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_f_fixed_points_l1546_154605


namespace NUMINAMATH_CALUDE_product_equals_64_l1546_154638

theorem product_equals_64 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 * (1 / 4096) * 8192 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_64_l1546_154638


namespace NUMINAMATH_CALUDE_train_crossing_time_l1546_154629

/-- Proves that a train of given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 200 →
  train_speed_kmh = 180 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 4 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l1546_154629


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1546_154687

theorem sufficient_not_necessary (a : ℝ) :
  (a = 1/8 → ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/8 ∧ ∀ x : ℝ, x > 0 → 2*x + a/x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1546_154687


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1546_154665

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) →
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) →
  a + b ≤ c + d ∧ a + b ≥ 6.5 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1546_154665


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1546_154615

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 4 * π * r^2) : 
  R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1546_154615


namespace NUMINAMATH_CALUDE_function_upper_bound_condition_l1546_154627

theorem function_upper_bound_condition (a : ℝ) (h_a : a > 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → a * x - x^2 ≤ 1) ↔ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_condition_l1546_154627


namespace NUMINAMATH_CALUDE_roof_area_l1546_154616

/-- Calculates the area of a rectangular roof given the conditions --/
theorem roof_area (width : ℝ) (length : ℝ) : 
  length = 4 * width → 
  length - width = 42 → 
  width * length = 784 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l1546_154616


namespace NUMINAMATH_CALUDE_field_trip_students_l1546_154661

/-- Proves that the number of students on a field trip is equal to the product of seats per bus and number of buses -/
theorem field_trip_students (seats_per_bus : ℕ) (num_buses : ℕ) (h1 : seats_per_bus = 9) (h2 : num_buses = 5) :
  seats_per_bus * num_buses = 45 := by
  sorry

#check field_trip_students

end NUMINAMATH_CALUDE_field_trip_students_l1546_154661


namespace NUMINAMATH_CALUDE_emily_age_l1546_154609

theorem emily_age :
  ∀ (e g : ℕ),
  g = 15 * e →
  g - e = 70 →
  e = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_emily_age_l1546_154609


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1546_154684

def f (a : ℝ) (x : ℝ) := a * x^2 + 2 * a * x + 1

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1546_154684


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1546_154656

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ) * (2 + a * Complex.I) * (a - 2 * Complex.I) = -4 * Complex.I → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1546_154656


namespace NUMINAMATH_CALUDE_road_trip_duration_l1546_154659

/-- Road trip duration calculation -/
theorem road_trip_duration (jenna_distance : ℝ) (friend_distance : ℝ) 
  (jenna_speed : ℝ) (friend_speed : ℝ) (num_breaks : ℕ) (break_duration : ℝ) :
  jenna_distance = 200 →
  friend_distance = 100 →
  jenna_speed = 50 →
  friend_speed = 20 →
  num_breaks = 2 →
  break_duration = 0.5 →
  (jenna_distance / jenna_speed) + (friend_distance / friend_speed) + 
    (num_breaks : ℝ) * break_duration = 10 := by
  sorry

#check road_trip_duration

end NUMINAMATH_CALUDE_road_trip_duration_l1546_154659


namespace NUMINAMATH_CALUDE_shirt_cost_equation_l1546_154683

theorem shirt_cost_equation (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive (number of shirts can't be negative or zero)
  (h2 : 1.5 * x > 0) -- Ensure 1.5x is positive
  (h3 : 7800 = (1.5 * x) * ((6400 / x) - 30)) -- Total cost of type A shirts
  (h4 : 6400 = x * (6400 / x)) -- Total cost of type B shirts
  : 7800 / (1.5 * x) + 30 = 6400 / x := by
  sorry

#check shirt_cost_equation

end NUMINAMATH_CALUDE_shirt_cost_equation_l1546_154683


namespace NUMINAMATH_CALUDE_solve_for_y_l1546_154685

theorem solve_for_y (x y : ℝ) (h1 : x^2 = 2*y - 6) (h2 : x = 3) : y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1546_154685


namespace NUMINAMATH_CALUDE_base_seven_divisibility_l1546_154652

theorem base_seven_divisibility (d : Nat) : 
  d ≤ 6 → (2 * 7^3 + d * 7^2 + d * 7 + 7) % 13 = 0 ↔ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_divisibility_l1546_154652


namespace NUMINAMATH_CALUDE_quadratic_product_is_square_l1546_154626

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Derivative of a quadratic polynomial -/
def QuadraticPolynomial.deriv (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

theorem quadratic_product_is_square (f g : QuadraticPolynomial) 
  (h : ∀ x : ℝ, f.deriv x * g.deriv x ≥ |f.eval x| + |g.eval x|) :
  ∃ h : QuadraticPolynomial, ∀ x : ℝ, f.eval x * g.eval x = (h.eval x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_is_square_l1546_154626


namespace NUMINAMATH_CALUDE_charlie_votes_l1546_154662

/-- Represents a candidate in the student council election -/
inductive Candidate
| Alex
| Brenda
| Charlie
| Dana

/-- Represents the vote count for each candidate -/
def VoteCount := Candidate → ℕ

/-- The total number of votes cast in the election -/
def TotalVotes (votes : VoteCount) : ℕ :=
  votes Candidate.Alex + votes Candidate.Brenda + votes Candidate.Charlie + votes Candidate.Dana

theorem charlie_votes (votes : VoteCount) : 
  votes Candidate.Brenda = 40 ∧ 
  4 * votes Candidate.Brenda = TotalVotes votes ∧
  votes Candidate.Charlie = votes Candidate.Dana + 10 →
  votes Candidate.Charlie = 45 := by
  sorry

end NUMINAMATH_CALUDE_charlie_votes_l1546_154662


namespace NUMINAMATH_CALUDE_equation_solutions_l1546_154658

theorem equation_solutions (x : ℝ) : 2 * x * (x + 1) = x + 1 ↔ x = -1 ∨ x = 1/2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1546_154658


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l1546_154680

/-- The number of eggs in a full container -/
def full_container : ℕ := 15

/-- The number of eggs in an underfilled container -/
def underfilled_container : ℕ := 14

/-- The number of underfilled containers -/
def num_underfilled : ℕ := 3

/-- The minimum number of eggs initially bought -/
def min_initial_eggs : ℕ := 151

theorem smallest_number_of_eggs (n : ℕ) (h : n > min_initial_eggs) : 
  (∃ (c : ℕ), n = c * full_container - num_underfilled * (full_container - underfilled_container)) →
  162 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l1546_154680


namespace NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l1546_154641

theorem unique_square_divisible_by_six_in_range : ∃! x : ℕ, 
  (∃ n : ℕ, x = n^2) ∧ 
  (x % 6 = 0) ∧ 
  (50 ≤ x) ∧ 
  (x ≤ 150) ∧ 
  x = 144 := by
sorry

end NUMINAMATH_CALUDE_unique_square_divisible_by_six_in_range_l1546_154641


namespace NUMINAMATH_CALUDE_stockholm_malmo_distance_l1546_154619

/-- The scale factor of the map, representing kilometers per centimeter. -/
def scale : ℝ := 12

/-- The distance between Stockholm and Malmö on the map, in centimeters. -/
def map_distance : ℝ := 120

/-- The actual distance between Stockholm and Malmö, in kilometers. -/
def actual_distance : ℝ := map_distance * scale

/-- Theorem stating that the actual distance between Stockholm and Malmö is 1440 km. -/
theorem stockholm_malmo_distance : actual_distance = 1440 := by
  sorry

end NUMINAMATH_CALUDE_stockholm_malmo_distance_l1546_154619


namespace NUMINAMATH_CALUDE_mountain_climb_speed_l1546_154600

-- Define the parameters
def total_time : ℝ := 20
def total_distance : ℝ := 80

-- Define the variables
variable (v : ℝ)  -- Speed on the first day
variable (t : ℝ)  -- Time spent on the first day

-- Define the theorem
theorem mountain_climb_speed :
  -- Conditions
  (t + (t - 2) + (t + 1) = total_time) →
  (v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = total_distance) →
  -- Conclusion
  (v + 0.5 = 4.575) :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_mountain_climb_speed_l1546_154600


namespace NUMINAMATH_CALUDE_factor_calculation_l1546_154691

theorem factor_calculation (f : ℝ) : f * (2 * 20 + 5) = 135 → f = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_calculation_l1546_154691


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1546_154681

theorem polynomial_remainder (x : ℝ) : 
  (x^4 + x^3 + 1) % (x - 2) = 25 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1546_154681


namespace NUMINAMATH_CALUDE_leopard_arrangement_l1546_154668

theorem leopard_arrangement (n : ℕ) (h : n = 9) : 
  (2 : ℕ) * Nat.factorial 2 * Nat.factorial (n - 3) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_leopard_arrangement_l1546_154668


namespace NUMINAMATH_CALUDE_photo_album_completion_l1546_154651

theorem photo_album_completion 
  (total_slots : ℕ) 
  (cristina_photos : ℕ) 
  (john_photos : ℕ) 
  (sarah_photos : ℕ) 
  (h1 : total_slots = 40) 
  (h2 : cristina_photos = 7) 
  (h3 : john_photos = 10) 
  (h4 : sarah_photos = 9) : 
  total_slots - (cristina_photos + john_photos + sarah_photos) = 14 := by
  sorry

end NUMINAMATH_CALUDE_photo_album_completion_l1546_154651


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1546_154606

theorem geometric_series_sum (a b : ℝ) (h : b ≠ 1) (h2 : b ≠ 0) :
  (∑' n, a / b^n) = 2 →
  (∑' n, a / (2*a + b)^n) = 2/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1546_154606


namespace NUMINAMATH_CALUDE_square_side_length_average_l1546_154679

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l1546_154679


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1546_154690

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = 1}
def N (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

-- Define the set of possible values for a
def A : Set ℝ := {-1, 1, 0}

-- State the theorem
theorem possible_values_of_a (a : ℝ) : N a ⊆ M → a ∈ A := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1546_154690


namespace NUMINAMATH_CALUDE_modular_inverse_of_2_mod_199_l1546_154644

theorem modular_inverse_of_2_mod_199 : ∃ x : ℤ, 2 * x ≡ 1 [ZMOD 199] ∧ 0 ≤ x ∧ x < 199 :=
  by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_2_mod_199_l1546_154644


namespace NUMINAMATH_CALUDE_base8_to_base10_77_l1546_154664

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (a b : Nat) : Nat :=
  a * 8 + b

/-- The given number in base 8 -/
def number_base8 : Nat × Nat := (7, 7)

theorem base8_to_base10_77 :
  base8_to_base10 number_base8.1 number_base8.2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_77_l1546_154664


namespace NUMINAMATH_CALUDE_total_birds_on_fence_l1546_154699

def initial_birds : ℕ := 4
def additional_birds : ℕ := 6

theorem total_birds_on_fence : 
  initial_birds + additional_birds = 10 := by sorry

end NUMINAMATH_CALUDE_total_birds_on_fence_l1546_154699


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1546_154650

theorem sum_remainder_mod_seven :
  (123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1546_154650


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1546_154655

theorem sum_of_squares_lower_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l1546_154655


namespace NUMINAMATH_CALUDE_monkey_climb_theorem_l1546_154653

/-- The time taken for a monkey to climb a tree, given the tree height, hop distance, slip distance, and final hop distance. -/
def monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) (final_hop : ℕ) : ℕ :=
  let net_progress := hop_distance - slip_distance
  let distance_before_final_hop := tree_height - final_hop
  distance_before_final_hop / net_progress + 1

/-- Theorem stating that a monkey climbing a 20 ft tree, hopping 3 ft and slipping 2 ft each hour, with a final 3 ft hop, takes 18 hours to reach the top. -/
theorem monkey_climb_theorem :
  monkey_climb_time 20 3 2 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_theorem_l1546_154653


namespace NUMINAMATH_CALUDE_new_ellipse_and_hyperbola_l1546_154645

/-- New distance between two points -/
def new_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

/-- New ellipse -/
def on_new_ellipse (x y c d a : ℝ) : Prop :=
  new_distance x y c d + new_distance x y (-c) (-d) = 2 * a

/-- New hyperbola -/
def on_new_hyperbola (x y c d a : ℝ) : Prop :=
  |new_distance x y c d - new_distance x y (-c) (-d)| = 2 * a

/-- Main theorem for new ellipse and hyperbola -/
theorem new_ellipse_and_hyperbola (x y c d a : ℝ) :
  (on_new_ellipse x y c d a ↔ 
    |x - c| + |y - d| + |x + c| + |y + d| = 2 * a) ∧
  (on_new_hyperbola x y c d a ↔ 
    |(|x - c| + |y - d|) - (|x + c| + |y + d|)| = 2 * a) :=
by sorry

end NUMINAMATH_CALUDE_new_ellipse_and_hyperbola_l1546_154645


namespace NUMINAMATH_CALUDE_square_area_ratio_l1546_154610

theorem square_area_ratio (p1 p2 : ℕ) (h1 : p1 = 32) (h2 : p2 = 20) : 
  (p1 / 4) ^ 2 / (p2 / 4) ^ 2 = 64 / 25 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1546_154610


namespace NUMINAMATH_CALUDE_unique_solution_l1546_154646

/-- The functional equation satisfied by g --/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x + y) * g (x - y) = (g x - g y)^2 - 6 * x^2 * g y

/-- There is only one function satisfying the functional equation --/
theorem unique_solution :
  ∃! g : ℝ → ℝ, FunctionalEquation g ∧ ∀ x : ℝ, g x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1546_154646


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l1546_154603

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 12 * x - 4 * y + 20

-- Define the square
structure Square where
  side_length : ℝ
  parallel_to_x_axis : Prop

-- Define the inscribed circle
structure InscribedCircle where
  equation : (ℝ → ℝ → Prop)
  square : Square

-- Theorem statement
theorem inscribed_circle_square_area 
  (circle : InscribedCircle) 
  (h : circle.equation = circle_equation) :
  circle.square.side_length^2 = 80 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l1546_154603


namespace NUMINAMATH_CALUDE_coefficient_of_x3y5_l1546_154622

/-- The coefficient of x^3y^5 in the expansion of (2/3x - y/3)^8 -/
def coefficient : ℚ := -448/6561

/-- The binomial expansion of (a + b)^n -/
def binomial_expansion (a b : ℚ) (n : ℕ) (k : ℕ) : ℚ := 
  (n.choose k) * (a^(n-k)) * (b^k)

theorem coefficient_of_x3y5 :
  coefficient = binomial_expansion (2/3) (-1/3) 8 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x3y5_l1546_154622


namespace NUMINAMATH_CALUDE_expression_evaluation_l1546_154632

theorem expression_evaluation : 
  (4+8-16+32+64-128+256)/(8+16-32+64+128-256+512) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1546_154632


namespace NUMINAMATH_CALUDE_total_flights_climbed_l1546_154604

/-- Represents a landmark with flights of stairs going up and down -/
structure Landmark where
  name : String
  flightsUp : ℕ
  flightsDown : ℕ

/-- Calculates the total flights for a landmark -/
def totalFlights (l : Landmark) : ℕ := l.flightsUp + l.flightsDown

/-- The landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104 }
]

/-- Theorem: The total number of flights Rachel climbed is 2343 -/
theorem total_flights_climbed : (landmarks.map totalFlights).sum = 2343 := by
  sorry

end NUMINAMATH_CALUDE_total_flights_climbed_l1546_154604


namespace NUMINAMATH_CALUDE_wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l1546_154686

/-- Calculates the additional Wednesday discount for jeans given the original price,
    summer discount percentage, and final price after all discounts. -/
theorem wednesday_discount_percentage 
  (original_price : ℝ) 
  (summer_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_summer_discount := original_price * (1 - summer_discount_percent / 100)
  let additional_discount := price_after_summer_discount - final_price
  let wednesday_discount_percent := (additional_discount / price_after_summer_discount) * 100
  wednesday_discount_percent

/-- The additional Wednesday discount for jeans is approximately 40.82% -/
theorem wednesday_jeans_discount_approx_40_82_percent : 
  ∃ ε > 0, abs (wednesday_discount_percentage 49 50 14.5 - 40.82) < ε :=
sorry

end NUMINAMATH_CALUDE_wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l1546_154686


namespace NUMINAMATH_CALUDE_dilation_and_shift_result_l1546_154635

/-- Represents a complex number -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- Applies a dilation to a complex number -/
def dilate (center : ComplexNumber) (scale : ℝ) (z : ComplexNumber) : ComplexNumber :=
  { re := center.re + scale * (z.re - center.re),
    im := center.im + scale * (z.im - center.im) }

/-- Shifts a complex number by another complex number -/
def shift (z : ComplexNumber) (s : ComplexNumber) : ComplexNumber :=
  { re := z.re - s.re,
    im := z.im - s.im }

/-- The main theorem to be proved -/
theorem dilation_and_shift_result :
  let initial := ComplexNumber.mk 1 (-2)
  let center := ComplexNumber.mk 1 2
  let scale := 2
  let shiftAmount := ComplexNumber.mk 3 4
  let dilated := dilate center scale initial
  let final := shift dilated shiftAmount
  final = ComplexNumber.mk (-2) (-10) := by
  sorry

end NUMINAMATH_CALUDE_dilation_and_shift_result_l1546_154635


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1546_154617

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1546_154617


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l1546_154601

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l1546_154601


namespace NUMINAMATH_CALUDE_no_function_exists_l1546_154613

theorem no_function_exists : ¬∃ (a : ℕ → ℕ), (a 0 = 0) ∧ (∀ n : ℕ, a n = n - a (a n)) := by
  sorry

end NUMINAMATH_CALUDE_no_function_exists_l1546_154613


namespace NUMINAMATH_CALUDE_two_minus_repeating_decimal_l1546_154692

/-- The value of the repeating decimal 1.888... -/
def repeating_decimal : ℚ := 17 / 9

/-- Theorem stating that 2 minus the repeating decimal 1.888... equals 1/9 -/
theorem two_minus_repeating_decimal :
  2 - repeating_decimal = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_two_minus_repeating_decimal_l1546_154692


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1546_154618

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k = 0 ∧ x = 3) → 
  (∃ y : ℝ, y^2 - 2*y - k = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1546_154618


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l1546_154675

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let r' := r * (1 - x)
  let A' := π * r'^2
  A' = 0.25 * A →
  x = 0.5
  := by sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l1546_154675
