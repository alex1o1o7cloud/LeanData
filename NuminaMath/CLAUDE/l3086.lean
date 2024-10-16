import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_condition_l3086_308616

/-- The diamond-shaped region defined by |x| + |y - 1| = 1 -/
def DiamondRegion (x y : ℝ) : Prop :=
  (abs x) + (abs (y - 1)) = 1

/-- The line defined by y = a * x + 2012 -/
def Line (a x y : ℝ) : Prop :=
  y = a * x + 2012

/-- The system of equations has a unique solution -/
def UniqueSystemSolution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), DiamondRegion x y ∧ Line a x y

/-- The theorem stating the condition for a unique solution -/
theorem unique_solution_condition (a : ℝ) :
  UniqueSystemSolution a ↔ (a = 2011 ∨ a = -2011) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3086_308616


namespace NUMINAMATH_CALUDE_profit_loss_recording_l3086_308628

/-- Represents the financial record of a store. -/
inductive FinancialRecord
  | profit (amount : ℤ)
  | loss (amount : ℤ)

/-- Records a financial transaction. -/
def recordTransaction (transaction : FinancialRecord) : ℤ :=
  match transaction with
  | FinancialRecord.profit amount => amount
  | FinancialRecord.loss amount => -amount

/-- The theorem stating how profits and losses should be recorded. -/
theorem profit_loss_recording (profitAmount lossAmount : ℤ) 
  (h : profitAmount = 20 ∧ lossAmount = 10) : 
  recordTransaction (FinancialRecord.profit profitAmount) = 20 ∧
  recordTransaction (FinancialRecord.loss lossAmount) = -10 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_recording_l3086_308628


namespace NUMINAMATH_CALUDE_baker_cake_difference_l3086_308676

/-- Given the initial number of cakes, the number of cakes sold, and the number of cakes bought,
    prove that the difference between cakes sold and cakes bought is 47. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
    (h1 : initial = 170)
    (h2 : sold = 78)
    (h3 : bought = 31) :
  sold - bought = 47 := by
  sorry

end NUMINAMATH_CALUDE_baker_cake_difference_l3086_308676


namespace NUMINAMATH_CALUDE_safari_count_difference_l3086_308643

/-- The number of animals Josie counted on safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- The conditions of Josie's safari count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.antelopes = 80 ∧
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas < count.antelopes + count.rabbits ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating the difference between hyenas and the sum of antelopes and rabbits --/
theorem safari_count_difference (count : SafariCount) 
  (h : safari_conditions count) : 
  count.antelopes + count.rabbits - count.hyenas = 42 := by
  sorry

end NUMINAMATH_CALUDE_safari_count_difference_l3086_308643


namespace NUMINAMATH_CALUDE_tan_double_angle_l3086_308671

theorem tan_double_angle (x : ℝ) (h : Real.tan (Real.pi - x) = 3 / 4) : 
  Real.tan (2 * x) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l3086_308671


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3086_308695

theorem sum_of_fractions_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l3086_308695


namespace NUMINAMATH_CALUDE_simplify_expression_l3086_308642

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b + b/a - 2/(a*b) = 1 - 1/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3086_308642


namespace NUMINAMATH_CALUDE_article_cost_l3086_308655

/-- The cost of an article given specific selling prices and gain percentages -/
theorem article_cost : ∃ (cost : ℝ), 
  (895 - cost) = 1.075 * (785 - cost) ∧ 
  cost > 0 ∧ 
  cost < 785 := by
sorry

end NUMINAMATH_CALUDE_article_cost_l3086_308655


namespace NUMINAMATH_CALUDE_janet_change_l3086_308638

def muffin_price : ℚ := 75 / 100
def num_muffins : ℕ := 12
def amount_paid : ℚ := 20

theorem janet_change :
  amount_paid - (num_muffins : ℚ) * muffin_price = 11 := by
  sorry

end NUMINAMATH_CALUDE_janet_change_l3086_308638


namespace NUMINAMATH_CALUDE_max_sum_perpendicular_distances_l3086_308604

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a ≥ b
  h_bc : b ≥ c
  h_pos : 0 < c

-- Define the inradius of a triangle
def inradius (t : Triangle) : ℝ := sorry

-- Define the sum of perpendicular distances from a point to the sides of the triangle
def sum_perpendicular_distances (t : Triangle) (P : ℝ × ℝ) : ℝ := sorry

theorem max_sum_perpendicular_distances (t : Triangle) :
  ∀ P, sum_perpendicular_distances t P ≤ 2 * (inradius t) * (t.a + t.b + t.c) :=
sorry

end NUMINAMATH_CALUDE_max_sum_perpendicular_distances_l3086_308604


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3086_308636

/-- An arithmetic sequence with a specific condition -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition : a 3 + a 6 + 3 * a 7 = 20

/-- The theorem stating that for any arithmetic sequence satisfying the given condition, 2a₇ - a₈ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 2 * seq.a 7 - seq.a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3086_308636


namespace NUMINAMATH_CALUDE_g_ln_inverse_2017_l3086_308664

noncomputable section

variable (a : ℝ)
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

-- Define the properties of f
axiom f_property (m n : ℝ) : f (m + n) = f m + f n - 1

-- Define g in terms of f and a
axiom g_def (x : ℝ) : g x = f x + (a^x / (a^x + 1))

-- Given conditions
axiom a_positive : a > 0
axiom a_not_one : a ≠ 1
axiom g_ln_2017 : g (Real.log 2017) = 2018

-- Theorem to prove
theorem g_ln_inverse_2017 : g (Real.log (1 / 2017)) = -2015 :=
sorry

end

end NUMINAMATH_CALUDE_g_ln_inverse_2017_l3086_308664


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3086_308637

theorem rectangular_field_perimeter
  (area : ℝ) (width : ℝ) (h_area : area = 300) (h_width : width = 15) :
  2 * (area / width + width) = 70 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3086_308637


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l3086_308640

/-- Given a cone formed from a 270-degree sector of a circle with radius 18,
    prove that the volume of the cone divided by π is equal to 60.75√141.75 -/
theorem cone_volume_divided_by_pi (r : ℝ) (h : ℝ) :
  r = 13.5 →
  h^2 = 141.75 →
  (1/3 * π * r^2 * h) / π = 60.75 * Real.sqrt 141.75 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l3086_308640


namespace NUMINAMATH_CALUDE_unique_integer_representation_l3086_308625

theorem unique_integer_representation (A m n p : ℕ) : 
  A > 0 ∧ 
  m ≥ n ∧ n ≥ p ∧ p ≥ 1 ∧
  A = (m - 1/n) * (n - 1/p) * (p - 1/m) →
  A = 21 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_representation_l3086_308625


namespace NUMINAMATH_CALUDE_rectangle_in_circle_l3086_308693

/-- A rectangle with sides 7 cm and 24 cm is inscribed in a circle. -/
theorem rectangle_in_circle (a b r : ℝ) (h1 : a = 7) (h2 : b = 24) 
  (h3 : a^2 + b^2 = (2*r)^2) : 
  (2 * π * r = 25 * π) ∧ (a * b = 168) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_in_circle_l3086_308693


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3086_308600

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k : ℝ) = k * 2 + (-1) + 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3086_308600


namespace NUMINAMATH_CALUDE_second_sum_calculation_l3086_308692

/-- Given a total sum of 2665 Rs divided into two parts, where the interest on the first part
    for 5 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is equal to 1332.5 Rs. -/
theorem second_sum_calculation (total : ℝ) (first_part : ℝ) (second_part : ℝ) :
  total = 2665 →
  first_part + second_part = total →
  (first_part * 3 * 5) / 100 = (second_part * 5 * 3) / 100 →
  second_part = 1332.5 := by
  sorry

end NUMINAMATH_CALUDE_second_sum_calculation_l3086_308692


namespace NUMINAMATH_CALUDE_tangent_slope_implies_tan_l3086_308663

/-- Given a function f and a point x₀ where the slope of the tangent line is 1,
    prove that tan(x₀) = -√3 -/
theorem tangent_slope_implies_tan (f : ℝ → ℝ) (x₀ : ℝ) : 
  (∀ x, f x = (1/2) * x - (1/4) * Real.sin x - (Real.sqrt 3 / 4) * Real.cos x) →
  (HasDerivAt f 1 x₀) →
  Real.tan x₀ = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_tan_l3086_308663


namespace NUMINAMATH_CALUDE_leila_order_cost_l3086_308618

/-- Calculates the total cost of Leila's cake order --/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
                (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Proves that the total cost of Leila's order is $168 --/
theorem leila_order_cost : 
  total_cost 3 12 6 22 = 168 := by
  sorry

#eval total_cost 3 12 6 22

end NUMINAMATH_CALUDE_leila_order_cost_l3086_308618


namespace NUMINAMATH_CALUDE_total_earnings_l3086_308659

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem total_earnings :
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l3086_308659


namespace NUMINAMATH_CALUDE_rectangle_point_s_l3086_308610

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a rectangle formed by four points -/
def isRectangle (p q r s : Point2D) : Prop :=
  (p.x = q.x ∧ r.x = s.x ∧ p.y = s.y ∧ q.y = r.y) ∨
  (p.x = s.x ∧ q.x = r.x ∧ p.y = q.y ∧ r.y = s.y)

/-- The theorem stating that given P, Q, and R, the point S forms a rectangle -/
theorem rectangle_point_s (p q r : Point2D)
  (h_p : p = ⟨3, -2⟩)
  (h_q : q = ⟨3, 1⟩)
  (h_r : r = ⟨7, 1⟩) :
  ∃ s : Point2D, s = ⟨7, -2⟩ ∧ isRectangle p q r s :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_s_l3086_308610


namespace NUMINAMATH_CALUDE_odd_function_property_l3086_308698

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_even : IsEven (fun x ↦ f (x + 1)))
  (h_def : ∀ x ∈ Set.Icc 0 1, f x = x * (3 - 2 * x)) :
  f (31/2) = -1 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_property_l3086_308698


namespace NUMINAMATH_CALUDE_sum_of_products_l3086_308678

theorem sum_of_products (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 14)
  (eq4 : b + c + d = 9) :
  a * b + c * d = 338 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3086_308678


namespace NUMINAMATH_CALUDE_subtraction_of_mixed_numbers_l3086_308660

theorem subtraction_of_mixed_numbers : (2 + 5/6) - (1 + 1/3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_mixed_numbers_l3086_308660


namespace NUMINAMATH_CALUDE_system_solution_l3086_308661

noncomputable def solve_system (x : Fin 12 → ℚ) : Prop :=
  x 0 + 12 * x 1 = 15 ∧
  x 0 - 12 * x 1 + 11 * x 2 = 2 ∧
  x 0 - 11 * x 2 + 10 * x 3 = 2 ∧
  x 0 - 10 * x 3 + 9 * x 4 = 2 ∧
  x 0 - 9 * x 4 + 8 * x 5 = 2 ∧
  x 0 - 8 * x 5 + 7 * x 6 = 2 ∧
  x 0 - 7 * x 6 + 6 * x 7 = 2 ∧
  x 0 - 6 * x 7 + 5 * x 8 = 2 ∧
  x 0 - 5 * x 8 + 4 * x 9 = 2 ∧
  x 0 - 4 * x 9 + 3 * x 10 = 2 ∧
  x 0 - 3 * x 10 + 2 * x 11 = 2 ∧
  x 0 - 2 * x 11 = 2

theorem system_solution :
  ∃! x : Fin 12 → ℚ, solve_system x ∧
    x 0 = 37/12 ∧ x 1 = 143/144 ∧ x 2 = 65/66 ∧ x 3 = 39/40 ∧
    x 4 = 26/27 ∧ x 5 = 91/96 ∧ x 6 = 13/14 ∧ x 7 = 65/72 ∧
    x 8 = 13/15 ∧ x 9 = 13/16 ∧ x 10 = 13/18 ∧ x 11 = 13/24 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3086_308661


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3086_308639

/-- The line passing through (-1, 1) and perpendicular to x + y = 0 has equation x - y + 2 = 0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + 1 = 0 ∧ y - 1 = 0) →  -- Point (-1, 1)
  (∀ x y, x + y = 0 → True) →  -- Given line x + y = 0
  x - y + 2 = 0 := by  -- Resulting perpendicular line
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3086_308639


namespace NUMINAMATH_CALUDE_divisor_problem_l3086_308615

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 15698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 176 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l3086_308615


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3086_308617

theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  (∃ θ : ℝ, θ = Real.pi/4 ∧ 
    (∀ t : ℝ, ∃ x y : ℝ, x = t ∧ y = (b/a)*t ∨ x = t ∧ y = -(b/a)*t) ∧
    θ = Real.arctan ((2*(b/a))/(1 - (b/a)^2))) →
  a/b = Real.sqrt 2 + 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3086_308617


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3086_308622

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 15^4 →
  m % 10 = 9 →
  n % 10 = 5 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3086_308622


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_one_over_x_l3086_308614

open Real MeasureTheory Interval

theorem integral_x_squared_plus_one_over_x :
  ∫ x in (1 : ℝ)..2, (x^2 + 1) / x = 3/2 + Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_one_over_x_l3086_308614


namespace NUMINAMATH_CALUDE_simplify_fraction_l3086_308690

theorem simplify_fraction : 25 * (9 / 14) * (2 / 27) = 25 / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3086_308690


namespace NUMINAMATH_CALUDE_f_of_A_eq_l3086_308633

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 2, 3]

/-- The polynomial function f --/
def f (x : Matrix (Fin 2) (Fin 2) ℤ) : Matrix (Fin 2) (Fin 2) ℤ := x^2 - 5 • x

/-- Theorem stating that f(A) equals the given result --/
theorem f_of_A_eq : f A = !![(-6), 1; (-2), (-8)] := by sorry

end NUMINAMATH_CALUDE_f_of_A_eq_l3086_308633


namespace NUMINAMATH_CALUDE_expression_evaluation_l3086_308669

theorem expression_evaluation (a b : ℝ) : 
  (a^2 + a - 6 = 0) → 
  (b^2 + b - 6 = 0) → 
  a ≠ b →
  (a / (a^2 - b^2) - 1 / (a + b)) / (1 / (a^2 - a*b)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3086_308669


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3086_308606

theorem binomial_coefficient_equality (p k : ℕ) (hp : Prime p) :
  ∃ n : ℕ, (n.choose p) = ((n + k).choose p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3086_308606


namespace NUMINAMATH_CALUDE_probability_of_at_least_three_successes_l3086_308688

def probability_of_success : ℚ := 4/5

def number_of_trials : ℕ := 4

def at_least_successes : ℕ := 3

theorem probability_of_at_least_three_successes :
  (Finset.sum (Finset.range (number_of_trials - at_least_successes + 1))
    (fun k => Nat.choose number_of_trials (number_of_trials - k) *
      probability_of_success ^ (number_of_trials - k) *
      (1 - probability_of_success) ^ k)) = 512/625 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_at_least_three_successes_l3086_308688


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3086_308602

theorem inscribed_square_area (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) 
  (h4 : a * b = x^2) (h5 : a = 34) (h6 : b = 66) : x^2 = 2244 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3086_308602


namespace NUMINAMATH_CALUDE_intersection_of_complement_and_Q_l3086_308666

-- Define the sets P and Q
def P : Set ℝ := {x | x - 1 ≤ 0}
def Q : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the complement of P in ℝ
def C_R_P : Set ℝ := {x | ¬(x ∈ P)}

-- State the theorem
theorem intersection_of_complement_and_Q : 
  (C_R_P ∩ Q) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_complement_and_Q_l3086_308666


namespace NUMINAMATH_CALUDE_smallest_prime_perfect_square_plus_20_l3086_308667

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square plus 20
def isPerfectSquarePlus20 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + 20

-- Theorem statement
theorem smallest_prime_perfect_square_plus_20 :
  isPrime 29 ∧ isPerfectSquarePlus20 29 ∧
  ∀ m : ℕ, m < 29 → ¬(isPrime m ∧ isPerfectSquarePlus20 m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perfect_square_plus_20_l3086_308667


namespace NUMINAMATH_CALUDE_gcd_8a_plus_3_5a_plus_2_is_1_l3086_308685

theorem gcd_8a_plus_3_5a_plus_2_is_1 (a : ℕ) : Nat.gcd (8 * a + 3) (5 * a + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8a_plus_3_5a_plus_2_is_1_l3086_308685


namespace NUMINAMATH_CALUDE_relationship_abc_l3086_308624

theorem relationship_abc :
  let a := (3/5 : ℝ) ^ (2/5 : ℝ)
  let b := (2/5 : ℝ) ^ (3/5 : ℝ)
  let c := (2/5 : ℝ) ^ (2/5 : ℝ)
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3086_308624


namespace NUMINAMATH_CALUDE_unique_integer_l3086_308673

theorem unique_integer (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_unique_integer_l3086_308673


namespace NUMINAMATH_CALUDE_students_not_in_biology_or_chemistry_l3086_308680

theorem students_not_in_biology_or_chemistry
  (total : ℕ)
  (biology_percent : ℚ)
  (chemistry_percent : ℚ)
  (both_percent : ℚ)
  (h_total : total = 880)
  (h_biology : biology_percent = 40 / 100)
  (h_chemistry : chemistry_percent = 30 / 100)
  (h_both : both_percent = 10 / 100) :
  total - (total * biology_percent + total * chemistry_percent - total * both_percent).floor = 352 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_or_chemistry_l3086_308680


namespace NUMINAMATH_CALUDE_vector_difference_sum_l3086_308687

theorem vector_difference_sum : 
  let v1 : Fin 2 → ℝ := ![5, -8]
  let v2 : Fin 2 → ℝ := ![2, 6]
  let v3 : Fin 2 → ℝ := ![-1, 4]
  let scalar : ℝ := 5
  v1 - scalar • v2 + v3 = ![-6, -34] := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_sum_l3086_308687


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3086_308683

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3086_308683


namespace NUMINAMATH_CALUDE_unique_pairs_count_l3086_308646

theorem unique_pairs_count (num_teenagers num_adults : ℕ) : 
  num_teenagers = 12 → num_adults = 8 → 
  (num_teenagers.choose 2) + (num_adults.choose 2) + (num_teenagers * num_adults) = 190 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_count_l3086_308646


namespace NUMINAMATH_CALUDE_mixture_cost_july_l3086_308656

/-- The cost of a mixture of milk powder and coffee in July -/
def mixture_cost (june_cost : ℝ) : ℝ :=
  let july_coffee_cost := june_cost * 4
  let july_milk_cost := june_cost * 0.2
  (1.5 * july_coffee_cost) + (1.5 * july_milk_cost)

/-- Theorem: The cost of a 3 lbs mixture of equal parts milk powder and coffee in July is $6.30 -/
theorem mixture_cost_july : ∃ (june_cost : ℝ), 
  (june_cost * 0.2 = 0.20) ∧ (mixture_cost june_cost = 6.30) := by
  sorry

end NUMINAMATH_CALUDE_mixture_cost_july_l3086_308656


namespace NUMINAMATH_CALUDE_probability_two_copresidents_selected_l3086_308679

def choose (n k : ℕ) : ℕ := Nat.choose n k

def prob_copresident_selected (n : ℕ) : ℚ :=
  (choose (n - 2) 2 : ℚ) / (choose n 4 : ℚ)

def total_probability : ℚ :=
  (1 : ℚ) / 3 * (prob_copresident_selected 6 + prob_copresident_selected 8 + prob_copresident_selected 9)

theorem probability_two_copresidents_selected : total_probability = 82 / 315 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_copresidents_selected_l3086_308679


namespace NUMINAMATH_CALUDE_half_vector_AB_l3086_308601

/-- Given two points A and B in a 2D plane, prove that half of the vector from A to B is (2, 1) -/
theorem half_vector_AB (A B : ℝ × ℝ) (h1 : A = (-1, 0)) (h2 : B = (3, 2)) :
  (1 / 2 : ℝ) • (B.1 - A.1, B.2 - A.2) = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_half_vector_AB_l3086_308601


namespace NUMINAMATH_CALUDE_largest_n_base_7_double_l3086_308686

def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

def from_base_7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 7 + d) 0

theorem largest_n_base_7_double : ∀ n : ℕ, n > 156 → 2 * n ≠ from_base_7 (to_base_7 n) :=
sorry

end NUMINAMATH_CALUDE_largest_n_base_7_double_l3086_308686


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3086_308652

theorem sum_of_roots_eq_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3086_308652


namespace NUMINAMATH_CALUDE_min_adults_in_park_l3086_308689

theorem min_adults_in_park (total_people : ℕ) (total_amount : ℚ) 
  (adult_price youth_price child_price : ℚ) :
  total_people = 100 →
  total_amount = 100 →
  adult_price = 3 →
  youth_price = 2 →
  child_price = (3 : ℚ) / 10 →
  ∃ (adults youths children : ℕ),
    adults + youths + children = total_people ∧
    adult_price * adults + youth_price * youths + child_price * children = total_amount ∧
    adults = 2 ∧
    ∀ (a y c : ℕ),
      a + y + c = total_people →
      adult_price * a + youth_price * y + child_price * c = total_amount →
      a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_adults_in_park_l3086_308689


namespace NUMINAMATH_CALUDE_max_expenditure_l3086_308654

def linear_regression (x : ℝ) (b a e : ℝ) : ℝ := b * x + a + e

theorem max_expenditure (x : ℝ) (e : ℝ) 
  (h1 : x = 10) 
  (h2 : |e| ≤ 0.5) : 
  linear_regression x 0.8 2 e ≤ 10.5 := by
  sorry

end NUMINAMATH_CALUDE_max_expenditure_l3086_308654


namespace NUMINAMATH_CALUDE_quadratic_roots_not_uniformly_increased_l3086_308697

theorem quadratic_roots_not_uniformly_increased (b c : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0) :
  ¬∃ y1 y2 : ℝ, y1 ≠ y2 ∧ 
    y1^2 + (b+1)*y1 + (c+1) = 0 ∧ 
    y2^2 + (b+1)*y2 + (c+1) = 0 ∧
    ∃ x1 x2 : ℝ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 ∧ y1 = x1 + 1 ∧ y2 = x2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_not_uniformly_increased_l3086_308697


namespace NUMINAMATH_CALUDE_weight_of_three_moles_l3086_308605

/-- The atomic weight of Carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Carbon atoms in C6H8O6 -/
def carbon_count : ℕ := 6

/-- The number of Hydrogen atoms in C6H8O6 -/
def hydrogen_count : ℕ := 8

/-- The number of Oxygen atoms in C6H8O6 -/
def oxygen_count : ℕ := 6

/-- The number of moles of C6H8O6 -/
def mole_count : ℝ := 3

/-- The molecular weight of C6H8O6 in g/mol -/
def molecular_weight : ℝ := 
  carbon_count * carbon_weight + 
  hydrogen_count * hydrogen_weight + 
  oxygen_count * oxygen_weight

/-- The total weight of 3 moles of C6H8O6 in grams -/
theorem weight_of_three_moles : 
  mole_count * molecular_weight = 528.42 := by sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_l3086_308605


namespace NUMINAMATH_CALUDE_square_area_thirteen_l3086_308694

/-- The area of a square with vertices at (1, 1), (-2, 3), (-1, 8), and (2, 4) is 13 square units. -/
theorem square_area_thirteen : 
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-2, 3)
  let R : ℝ × ℝ := (-1, 8)
  let S : ℝ × ℝ := (2, 4)
  let square_area := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
  square_area = 13 := by sorry

end NUMINAMATH_CALUDE_square_area_thirteen_l3086_308694


namespace NUMINAMATH_CALUDE_functional_equation_properties_l3086_308603

/-- A function satisfying the given properties -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x, f x > 0) ∧ 
  (∀ a b, f a * f b = f (a + b))

/-- Main theorem stating the properties of f -/
theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l3086_308603


namespace NUMINAMATH_CALUDE_three_percent_difference_l3086_308630

theorem three_percent_difference (x y : ℝ) 
  (hx : 3 = 0.15 * x) (hy : 3 = 0.05 * y) : y - x = 40 := by
  sorry

end NUMINAMATH_CALUDE_three_percent_difference_l3086_308630


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3086_308607

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ (56 * y + 8) % 26 = 6 % 26 ∧
  ∀ (z : ℕ), z > 0 ∧ (56 * z + 8) % 26 = 6 % 26 → y ≤ z :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3086_308607


namespace NUMINAMATH_CALUDE_count_ordered_pairs_eq_18_l3086_308619

/-- Given that 1372 = 2^2 * 7^2 * 11, this function returns the number of ordered pairs of positive integers (x, y) satisfying x * y = 1372. -/
def count_ordered_pairs : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 2), (7, 2), (11, 1)]
  (prime_factorization.map (λ (p, e) => e + 1)).prod

/-- The number of ordered pairs of positive integers (x, y) satisfying x * y = 1372 is 18. -/
theorem count_ordered_pairs_eq_18 : count_ordered_pairs = 18 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_eq_18_l3086_308619


namespace NUMINAMATH_CALUDE_largest_multiple_less_than_negative_fifty_l3086_308629

theorem largest_multiple_less_than_negative_fifty :
  ∀ n : ℤ, n * 12 < -50 → n * 12 ≤ -48 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_less_than_negative_fifty_l3086_308629


namespace NUMINAMATH_CALUDE_E_is_true_l3086_308682

-- Define the statements as propositions
variable (A B C D E : Prop)

-- Define the condition that only one statement is true
def only_one_true (A B C D E : Prop) : Prop :=
  (A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ B ∧ ¬C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ C ∧ ¬D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ D ∧ ¬E) ∨
  (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ E)

-- Define the content of each statement
def statement_definitions (A B C D E : Prop) : Prop :=
  (A ↔ B) ∧
  (B ↔ ¬E) ∧
  (C ↔ (A ∧ B ∧ C ∧ D ∧ E)) ∧
  (D ↔ (¬A ∧ ¬B ∧ ¬C ∧ ¬D ∧ ¬E)) ∧
  (E ↔ ¬A)

-- Theorem stating that E is the only true statement
theorem E_is_true (A B C D E : Prop) 
  (h1 : only_one_true A B C D E) 
  (h2 : statement_definitions A B C D E) : 
  E ∧ ¬A ∧ ¬B ∧ ¬C ∧ ¬D :=
sorry

end NUMINAMATH_CALUDE_E_is_true_l3086_308682


namespace NUMINAMATH_CALUDE_simplify_fraction_l3086_308674

theorem simplify_fraction : (140 : ℚ) / 9800 * 35 = 1 / 70 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3086_308674


namespace NUMINAMATH_CALUDE_last_digit_of_base4_conversion_l3086_308647

def base5_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

def decimal_to_base4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

def base5_number : List Nat := [4, 3, 2, 1]

theorem last_digit_of_base4_conversion :
  (decimal_to_base4 (base5_to_decimal base5_number)).getLast? = some 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_base4_conversion_l3086_308647


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3086_308691

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 1) : 
  a + b + c + d - 1 ≥ 16*a*b*c*d ∧ 
  (a + b + c + d - 1 = 16*a*b*c*d ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2 ∧ d = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3086_308691


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_divisors_l3086_308670

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def last_three_digits (n : ℕ) : ℕ := n % 1000

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_five_digit_multiple_with_16_divisors :
  ∃ (n : ℕ), is_five_digit n ∧ 2014 ∣ n ∧ count_divisors (last_three_digits n) = 16 ∧
  ∀ (m : ℕ), is_five_digit m ∧ 2014 ∣ m ∧ count_divisors (last_three_digits m) = 16 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_with_16_divisors_l3086_308670


namespace NUMINAMATH_CALUDE_equation_solution_l3086_308657

theorem equation_solution : ∃! x : ℝ, 2 * x + 4 = |(-17 + 3)| :=
  by
    -- The unique solution is x = 5
    use 5
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_equation_solution_l3086_308657


namespace NUMINAMATH_CALUDE_or_and_not_implication_l3086_308634

theorem or_and_not_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implication_l3086_308634


namespace NUMINAMATH_CALUDE_dividend_rate_for_given_stock_l3086_308612

/-- Represents a stock with its characteristics -/
structure Stock where
  nominal_percentage : ℝ  -- The nominal percentage of the stock
  quote : ℝ             -- The quoted price of the stock
  yield : ℝ              -- The yield of the stock as a percentage

/-- Calculates the dividend rate of a stock -/
def dividend_rate (s : Stock) : ℝ :=
  s.nominal_percentage

/-- Theorem stating that for a 25% stock quoted at 125 with a 20% yield, the dividend rate is 25 -/
theorem dividend_rate_for_given_stock :
  let s : Stock := { nominal_percentage := 25, quote := 125, yield := 20 }
  dividend_rate s = 25 := by
  sorry


end NUMINAMATH_CALUDE_dividend_rate_for_given_stock_l3086_308612


namespace NUMINAMATH_CALUDE_initial_position_proof_l3086_308620

def moves : List Int := [-5, 4, 2, -3, 1]
def final_position : Int := 6

theorem initial_position_proof :
  (moves.foldl (· + ·) final_position) = 7 := by sorry

end NUMINAMATH_CALUDE_initial_position_proof_l3086_308620


namespace NUMINAMATH_CALUDE_simplify_expression_l3086_308648

theorem simplify_expression (a b : ℝ) :
  3 * (a - b)^2 - 6 * (a - b)^2 + 2 * (a - b)^2 = -(a - b)^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l3086_308648


namespace NUMINAMATH_CALUDE_b_investment_is_1000_l3086_308621

/-- Represents the business partnership between a and b -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  total_profit : ℕ
  management_fee_percent : ℚ
  a_total_received : ℕ

/-- Calculates b's investment given the partnership details -/
def calculate_b_investment (p : Partnership) : ℕ :=
  sorry

/-- Theorem stating that b's investment is 1000 given the problem conditions -/
theorem b_investment_is_1000 (p : Partnership) 
  (h1 : p.a_investment = 2000)
  (h2 : p.total_profit = 9600)
  (h3 : p.management_fee_percent = 1/10)
  (h4 : p.a_total_received = 4416) :
  calculate_b_investment p = 1000 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_is_1000_l3086_308621


namespace NUMINAMATH_CALUDE_bird_count_problem_l3086_308632

/-- The number of grey birds initially in the cage -/
def initial_grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := initial_grey_birds + 6

/-- The number of grey birds remaining in the cage after ten minutes -/
def remaining_grey_birds : ℕ := initial_grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := 66

theorem bird_count_problem :
  white_birds + remaining_grey_birds = total_remaining_birds :=
sorry

end NUMINAMATH_CALUDE_bird_count_problem_l3086_308632


namespace NUMINAMATH_CALUDE_savings_calculation_l3086_308684

def total_expenses : ℚ := 30150
def savings_rate : ℚ := 1/5

theorem savings_calculation (salary : ℚ) (h1 : salary * savings_rate + total_expenses = salary) :
  salary * savings_rate = 7537.5 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3086_308684


namespace NUMINAMATH_CALUDE_average_weight_increase_l3086_308699

theorem average_weight_increase (initial_count : ℕ) (old_weight new_weight : ℝ) :
  initial_count = 10 →
  old_weight = 65 →
  new_weight = 90 →
  (new_weight - old_weight) / initial_count = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3086_308699


namespace NUMINAMATH_CALUDE_winner_equal_victories_defeats_iff_odd_l3086_308644

/-- A tournament of n nations where each nation plays against every other nation exactly once. -/
structure Tournament (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- The number of victories for a team in a tournament. -/
def victories (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- The number of defeats for a team in a tournament. -/
def defeats (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- A team is a winner if it has the maximum number of victories. -/
def is_winner (t : Tournament n) (team : Fin n) : Prop :=
  ∀ other : Fin n, victories t team ≥ victories t other

theorem winner_equal_victories_defeats_iff_odd (n : ℕ) :
  (∃ (t : Tournament n) (w : Fin n), is_winner t w ∧ victories t w = defeats t w) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_winner_equal_victories_defeats_iff_odd_l3086_308644


namespace NUMINAMATH_CALUDE_miriam_monday_pushups_l3086_308681

/-- Represents the number of push-ups Miriam did on each day of the week --/
structure PushUps where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of push-ups done before Thursday --/
def totalBeforeThursday (p : PushUps) : ℕ :=
  p.monday + p.tuesday + p.wednesday

/-- Represents Miriam's push-up routine for the week --/
def miriamPushUps (monday : ℕ) : PushUps :=
  { monday := monday
  , tuesday := 7
  , wednesday := 2 * 7
  , thursday := (totalBeforeThursday { monday := monday, tuesday := 7, wednesday := 2 * 7, thursday := 0, friday := 0 }) / 2
  , friday := 39
  }

/-- Theorem stating that Miriam did 5 push-ups on Monday --/
theorem miriam_monday_pushups :
  ∃ (p : PushUps), p = miriamPushUps 5 ∧
    p.monday + p.tuesday + p.wednesday + p.thursday = p.friday :=
  sorry

end NUMINAMATH_CALUDE_miriam_monday_pushups_l3086_308681


namespace NUMINAMATH_CALUDE_complex_magnitude_l3086_308613

theorem complex_magnitude (z : ℂ) (h : z * (1 + 2*I) = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3086_308613


namespace NUMINAMATH_CALUDE_davantes_girl_friends_l3086_308623

def days_in_week : ℕ := 7

def davantes_friends (days : ℕ) : ℕ := 2 * days

def boy_friends : ℕ := 11

theorem davantes_girl_friends :
  davantes_friends days_in_week - boy_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_davantes_girl_friends_l3086_308623


namespace NUMINAMATH_CALUDE_market_purchase_cost_l3086_308611

/-- The total cost of buying tomatoes and cabbage -/
def total_cost (a b : ℝ) : ℝ :=
  30 * a + 50 * b

/-- Theorem: The total cost of buying 30 kg of tomatoes at 'a' yuan per kg
    and 50 kg of cabbage at 'b' yuan per kg is 30a + 50b yuan -/
theorem market_purchase_cost (a b : ℝ) :
  total_cost a b = 30 * a + 50 * b := by
  sorry

end NUMINAMATH_CALUDE_market_purchase_cost_l3086_308611


namespace NUMINAMATH_CALUDE_cookery_club_committee_probability_l3086_308627

def total_members : ℕ := 24
def num_boys : ℕ := 14
def num_girls : ℕ := 10
def committee_size : ℕ := 5

theorem cookery_club_committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let committees_with_fewer_than_two_girls := 
    Nat.choose num_boys committee_size + 
    (num_girls * Nat.choose num_boys (committee_size - 1))
  let committees_with_at_least_two_girls := 
    total_committees - committees_with_fewer_than_two_girls
  (committees_with_at_least_two_girls : ℚ) / total_committees = 2541 / 3542 := by
  sorry

end NUMINAMATH_CALUDE_cookery_club_committee_probability_l3086_308627


namespace NUMINAMATH_CALUDE_largest_number_l3086_308677

theorem largest_number (x y z : ℝ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y + z = 82) (h4 : z - y = 10) (h5 : y - x = 4) :
  z = 106 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3086_308677


namespace NUMINAMATH_CALUDE_alternating_sum_2023_l3086_308626

/-- Calculates the sum of the alternating series from 1 to n -/
def alternatingSum (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

/-- The sum of the series 1-2+3-4+5-6+...-2022+2023 equals 1012 -/
theorem alternating_sum_2023 :
  alternatingSum 2023 = 1012 := by
  sorry

#eval alternatingSum 2023

end NUMINAMATH_CALUDE_alternating_sum_2023_l3086_308626


namespace NUMINAMATH_CALUDE_interval_intersection_l3086_308668

theorem interval_intersection (x : ℝ) :
  (2 < 3*x ∧ 3*x < 3 ∧ 2 < 4*x ∧ 4*x < 3) ↔ (2/3 < x ∧ x < 3/4) :=
by sorry

end NUMINAMATH_CALUDE_interval_intersection_l3086_308668


namespace NUMINAMATH_CALUDE_equation_roots_l3086_308696

theorem equation_roots : ∃ (x₁ x₂ x₃ : ℝ), 
  (x₁ = -x₂) ∧ 
  (x₁ = Real.sqrt 2 ∨ x₁ = -Real.sqrt 2) ∧
  (x₂ = Real.sqrt 2 ∨ x₂ = -Real.sqrt 2) ∧
  (x₃ = 1/2) ∧
  (2 * x₁^5 - x₁^4 - 2 * x₁^3 + x₁^2 - 4 * x₁ + 2 = 0) ∧
  (2 * x₂^5 - x₂^4 - 2 * x₂^3 + x₂^2 - 4 * x₂ + 2 = 0) ∧
  (2 * x₃^5 - x₃^4 - 2 * x₃^3 + x₃^2 - 4 * x₃ + 2 = 0) := by
  sorry

#check equation_roots

end NUMINAMATH_CALUDE_equation_roots_l3086_308696


namespace NUMINAMATH_CALUDE_mean_of_remaining_two_l3086_308609

def numbers : List ℕ := [1870, 1996, 2022, 2028, 2112, 2124]

theorem mean_of_remaining_two (four_numbers : List ℕ) 
  (h1 : four_numbers.length = 4)
  (h2 : four_numbers.all (· ∈ numbers))
  (h3 : (four_numbers.sum : ℚ) / 4 = 2011) :
  let remaining_two := numbers.filter (λ x => x ∉ four_numbers)
  (remaining_two.sum : ℚ) / 2 = 2054 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_two_l3086_308609


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l3086_308608

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a * x + 5
  else if x ≥ -2 ∧ x ≤ 2 then x - 7
  else 3 * x - b

-- State the theorem
theorem continuous_piecewise_function_sum (a b : ℝ) :
  Continuous (f a b) → a + b = -2 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l3086_308608


namespace NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l3086_308645

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 3} := by sorry

-- Theorem for (∁ᵤA) ∩ B
theorem intersection_complement_A_B : (Aᶜ : Set ℝ) ∩ B = {x | 3 ≤ x ∧ x < 5} := by sorry

-- Theorem for A ∩ (∁ᵤB)
theorem intersection_A_complement_B : A ∩ (Bᶜ : Set ℝ) = {x | -1 < x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_intersection_complement_A_B_intersection_A_complement_B_l3086_308645


namespace NUMINAMATH_CALUDE_handmade_ornaments_excess_l3086_308662

/-- Proves that the number of handmade ornaments exceeds 1/6 of the total ornaments by 20 -/
theorem handmade_ornaments_excess (total : ℕ) (handmade : ℕ) (antique : ℕ) : 
  total = 60 →
  3 * antique = total →
  2 * antique = handmade →
  antique = 20 →
  handmade - (total / 6) = 20 := by
  sorry

end NUMINAMATH_CALUDE_handmade_ornaments_excess_l3086_308662


namespace NUMINAMATH_CALUDE_quadrilateral_side_difference_l3086_308635

theorem quadrilateral_side_difference (a b c d : ℝ) (h1 : a + b + c + d = 120) 
  (h2 : a + c = 50) (h3 : a^2 + c^2 = 40^2) : |b - d| = 2 * Real.sqrt 775 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_difference_l3086_308635


namespace NUMINAMATH_CALUDE_dividend_calculation_l3086_308641

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 17) 
  (h2 : quotient = 9) 
  (h3 : remainder = 6) : 
  divisor * quotient + remainder = 159 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3086_308641


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3086_308658

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x ≠ 2 ∨ y ≠ 4
def q (x y : ℝ) : Prop := x + y ≠ 6

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l3086_308658


namespace NUMINAMATH_CALUDE_quadratic_coefficient_values_l3086_308631

/-- Given an algebraic expression x^2 + px + q, prove that p = 0 and q = -6
    when the expression equals -5 for x = -1 and 3 for x = 3. -/
theorem quadratic_coefficient_values (p q : ℝ) : 
  ((-1)^2 + p*(-1) + q = -5) ∧ (3^2 + p*3 + q = 3) → p = 0 ∧ q = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_values_l3086_308631


namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3086_308649

/-- The equation of the latus rectum of the parabola y = -1/4 * x^2 is y = 1 -/
theorem latus_rectum_of_parabola :
  ∀ (x y : ℝ), y = -(1/4) * x^2 → (∃ (x₀ : ℝ), y = 1 ∧ x₀^2 = -4*y) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l3086_308649


namespace NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3086_308651

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def fair_12_sided_die : Finset ℕ := Finset.range 12

/-- The probability of each outcome for a fair 12-sided die -/
def prob_each_outcome : ℚ := 1 / 12

/-- The expected value of rolling a fair 12-sided die -/
def expected_value : ℚ := (fair_12_sided_die.sum (λ x => (x + 1) * prob_each_outcome))

/-- Theorem: The expected value of rolling a fair 12-sided die is 6.5 -/
theorem expected_value_fair_12_sided_die : expected_value = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_fair_12_sided_die_l3086_308651


namespace NUMINAMATH_CALUDE_chrome_users_l3086_308675

theorem chrome_users (total : ℕ) (angle : ℕ) (chrome_users : ℕ) : 
  total = 530 → angle = 216 → chrome_users = 318 →
  (chrome_users : ℚ) / total * 360 = angle := by
  sorry

end NUMINAMATH_CALUDE_chrome_users_l3086_308675


namespace NUMINAMATH_CALUDE_choose_five_items_eq_48_l3086_308653

/-- The number of ways to choose 5 items from 3 distinct types, 
    where no two consecutive items can be of the same type -/
def choose_five_items : ℕ :=
  let first_choice := 3  -- 3 choices for the first item
  let subsequent_choices := 2  -- 2 choices for each subsequent item
  first_choice * subsequent_choices^4

theorem choose_five_items_eq_48 : choose_five_items = 48 := by
  sorry

end NUMINAMATH_CALUDE_choose_five_items_eq_48_l3086_308653


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3086_308650

/-- Represents a satellite with modular units and sensors -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
  (h1 : s.units = 24)
  (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) :
  upgraded_fraction s = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l3086_308650


namespace NUMINAMATH_CALUDE_bird_cages_count_l3086_308672

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℝ := 6.0

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℝ := 2.0

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 48

/-- Theorem stating that the number of bird cages is correct given the conditions -/
theorem bird_cages_count :
  (parrots_per_cage + parakeets_per_cage) * num_cages = total_birds := by
  sorry

end NUMINAMATH_CALUDE_bird_cages_count_l3086_308672


namespace NUMINAMATH_CALUDE_consecutive_price_increase_l3086_308665

theorem consecutive_price_increase (P : ℝ) (h : P > 0) :
  P * (1 + 0.1) * (1 + 0.1) = P * (1 + 0.21) := by
  sorry

#check consecutive_price_increase

end NUMINAMATH_CALUDE_consecutive_price_increase_l3086_308665
