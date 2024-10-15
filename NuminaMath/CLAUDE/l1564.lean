import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1564_156482

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
theorem ellipse_major_axis_length : ∀ (f₁ f₂ : ℝ × ℝ),
  f₁ = (10, 25) →
  f₂ = (50, 65) →
  ∃ (x : ℝ), -- point where ellipse is tangent to y-axis
  (∀ (p : ℝ × ℝ), p.1 = 0 → dist p f₁ + dist p f₂ ≥ dist (0, x) f₁ + dist (0, x) f₂) →
  dist (0, x) f₁ + dist (0, x) f₂ = 10 * Real.sqrt 117 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1564_156482


namespace NUMINAMATH_CALUDE_sixth_quiz_score_l1564_156441

def existing_scores : List ℕ := [86, 91, 83, 88, 97]
def target_mean : ℕ := 90
def num_quizzes : ℕ := 6

theorem sixth_quiz_score (x : ℕ) :
  (existing_scores.sum + x) / num_quizzes = target_mean ↔ x = 95 := by
  sorry

end NUMINAMATH_CALUDE_sixth_quiz_score_l1564_156441


namespace NUMINAMATH_CALUDE_always_true_inequality_l1564_156409

theorem always_true_inequality (x : ℝ) : x + 2 < x + 3 := by
  sorry

end NUMINAMATH_CALUDE_always_true_inequality_l1564_156409


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l1564_156428

theorem contrapositive_theorem (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b - 1) ↔
  (∀ a b, 2^a ≤ 2^b - 1 → a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l1564_156428


namespace NUMINAMATH_CALUDE_missing_fraction_proof_l1564_156413

theorem missing_fraction_proof :
  let given_fractions : List ℚ := [1/3, 1/2, -5/6, 1/4, -9/20, -5/6]
  let missing_fraction : ℚ := 56/30
  let total_sum : ℚ := 5/6
  (given_fractions.sum + missing_fraction = total_sum) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_proof_l1564_156413


namespace NUMINAMATH_CALUDE_abs_gt_iff_square_gt_l1564_156430

theorem abs_gt_iff_square_gt (x y : ℝ) : |x| > |y| ↔ x^2 > y^2 := by
  sorry

end NUMINAMATH_CALUDE_abs_gt_iff_square_gt_l1564_156430


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l1564_156418

/-- Proves that the actual distance traveled is 10 km given the conditions of the problem -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (extra_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 15)
  (h3 : extra_distance = 20)
  (h4 : ∀ t, fast_speed * t = slow_speed * t + extra_distance) : 
  ∃ d, d = 10 ∧ slow_speed * (d / slow_speed) = d ∧ fast_speed * (d / slow_speed) = d + extra_distance :=
by
  sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l1564_156418


namespace NUMINAMATH_CALUDE_x4_plus_y4_l1564_156431

theorem x4_plus_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_y4_l1564_156431


namespace NUMINAMATH_CALUDE_nuts_left_over_project_nuts_left_over_l1564_156465

theorem nuts_left_over (bolt_boxes : ℕ) (bolts_per_box : ℕ) (nut_boxes : ℕ) (nuts_per_box : ℕ) 
  (bolts_left : ℕ) (total_used : ℕ) : ℕ :=
  let total_bolts := bolt_boxes * bolts_per_box
  let total_nuts := nut_boxes * nuts_per_box
  let bolts_used := total_bolts - bolts_left
  let nuts_used := total_used - bolts_used
  let nuts_left := total_nuts - nuts_used
  nuts_left

theorem project_nuts_left_over : 
  nuts_left_over 7 11 3 15 3 113 = 6 := by
  sorry

end NUMINAMATH_CALUDE_nuts_left_over_project_nuts_left_over_l1564_156465


namespace NUMINAMATH_CALUDE_function_properties_l1564_156483

/-- Given function f with parameter a > 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1/a) * Real.log x + 1/x - x

theorem function_properties (a : ℝ) (h : a > 1) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/a), (deriv (f a)) x < 0) ∧
  (∀ x ∈ Set.Ioo (1/a) 1, (deriv (f a)) x > 0) ∧
  (a ≥ 3 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (deriv (f a)) x₁ = (deriv (f a)) x₂ ∧ x₁ + x₂ > 6/5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1564_156483


namespace NUMINAMATH_CALUDE_minimum_interval_for_f_l1564_156419

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem minimum_interval_for_f (t s : ℝ) (h : f t = g s) :
  ∃ (a : ℝ), (a > (1/2) ∧ a < Real.log 2) ∧
  (∀ (t' s' : ℝ), f t' = g s' → s' - t' ≥ s - t → f t = a) :=
sorry

end NUMINAMATH_CALUDE_minimum_interval_for_f_l1564_156419


namespace NUMINAMATH_CALUDE_john_non_rent_expenses_l1564_156411

/-- Represents John's computer business finances --/
structure ComputerBusiness where
  parts_cost : ℝ
  selling_price_multiplier : ℝ
  computers_per_month : ℕ
  monthly_rent : ℝ
  monthly_profit : ℝ

/-- Calculates the non-rent extra expenses for John's computer business --/
def non_rent_extra_expenses (business : ComputerBusiness) : ℝ :=
  let selling_price := business.parts_cost * business.selling_price_multiplier
  let total_revenue := selling_price * business.computers_per_month
  let total_cost_components := business.parts_cost * business.computers_per_month
  let total_expenses := total_revenue - business.monthly_profit
  total_expenses - business.monthly_rent - total_cost_components

/-- Theorem stating that John's non-rent extra expenses are $3000 per month --/
theorem john_non_rent_expenses :
  let john_business : ComputerBusiness := {
    parts_cost := 800,
    selling_price_multiplier := 1.4,
    computers_per_month := 60,
    monthly_rent := 5000,
    monthly_profit := 11200
  }
  non_rent_extra_expenses john_business = 3000 := by
  sorry

end NUMINAMATH_CALUDE_john_non_rent_expenses_l1564_156411


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l1564_156494

theorem min_value_floor_sum (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∃ (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0),
    ⌊(x + y + z) / w⌋ + ⌊(y + z + w) / x⌋ + ⌊(z + w + x) / y⌋ + ⌊(w + x + y) / z⌋ = 9 ∧
    ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
      ⌊(a + b + c) / d⌋ + ⌊(b + c + d) / a⌋ + ⌊(c + d + a) / b⌋ + ⌊(d + a + b) / c⌋ ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l1564_156494


namespace NUMINAMATH_CALUDE_units_digit_sum_l1564_156402

/-- The base of the number system -/
def base : ℕ := 8

/-- The first number in base 8 -/
def num1 : ℕ := 63

/-- The second number in base 8 -/
def num2 : ℕ := 74

/-- The units digit of the first number -/
def units_digit1 : ℕ := 3

/-- The units digit of the second number -/
def units_digit2 : ℕ := 4

/-- Theorem: The units digit of the sum of num1 and num2 in base 8 is 7 -/
theorem units_digit_sum : (num1 + num2) % base = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_l1564_156402


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l1564_156401

-- Define the function f(x) = (x-1)^2 - 2
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x ≤ y → f x ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l1564_156401


namespace NUMINAMATH_CALUDE_sum_interior_angles_increases_l1564_156421

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem: The sum of interior angles increases as the number of sides increases from 3 to n -/
theorem sum_interior_angles_increases (n : ℕ) (h : n > 3) :
  sum_interior_angles n > sum_interior_angles 3 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_increases_l1564_156421


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1564_156407

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1) ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1564_156407


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1564_156406

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isSumOfFiveDifferentPrimes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n = p₁ + p₂ + p₃ + p₄ + p₅

theorem smallest_prime_sum_of_five_primes :
  isPrime 43 ∧
  isSumOfFiveDifferentPrimes 43 ∧
  ∀ n : ℕ, n < 43 → ¬(isPrime n ∧ isSumOfFiveDifferentPrimes n) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1564_156406


namespace NUMINAMATH_CALUDE_beths_sister_age_l1564_156422

theorem beths_sister_age (beth_age : ℕ) (future_years : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → 
  future_years = 8 → 
  beth_age + future_years = 2 * (sister_age + future_years) → 
  sister_age = 5 := by
sorry

end NUMINAMATH_CALUDE_beths_sister_age_l1564_156422


namespace NUMINAMATH_CALUDE_total_corn_harvest_l1564_156459

-- Define the cornfield properties
def johnson_field : ℝ := 1
def johnson_yield : ℝ := 80
def johnson_period : ℝ := 2

def smith_field : ℝ := 2
def smith_yield_factor : ℝ := 2

def brown_field : ℝ := 1.5
def brown_yield : ℝ := 50
def brown_period : ℝ := 3

def taylor_field : ℝ := 0.5
def taylor_yield : ℝ := 30
def taylor_period : ℝ := 1

def total_months : ℝ := 6

-- Define the theorem
theorem total_corn_harvest :
  let johnson_total := (total_months / johnson_period) * johnson_yield
  let smith_total := (total_months / johnson_period) * (smith_field * smith_yield_factor * johnson_yield)
  let brown_total := (total_months / brown_period) * (brown_field * brown_yield)
  let taylor_total := (total_months / taylor_period) * taylor_yield
  johnson_total + smith_total + brown_total + taylor_total = 1530 := by
  sorry

end NUMINAMATH_CALUDE_total_corn_harvest_l1564_156459


namespace NUMINAMATH_CALUDE_equality_of_fractions_l1564_156400

theorem equality_of_fractions (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  a^7 / (1 - a)^2 - a^7 / (1 + a)^2 = b^7 / (1 - b)^2 - b^7 / (1 + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l1564_156400


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1564_156456

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = (1 : ℝ) / 2 ∧ x₂ = 1 ∧ 
  (2 * x₁^2 - 3 * x₁ + 1 = 0) ∧ (2 * x₂^2 - 3 * x₂ + 1 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1564_156456


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l1564_156489

theorem largest_power_of_two_dividing_difference (n : ℕ) : 
  n = 18^5 - 14^5 → ∃ k : ℕ, 2^k = 64 ∧ 2^k ∣ n ∧ ∀ m : ℕ, 2^m ∣ n → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_difference_l1564_156489


namespace NUMINAMATH_CALUDE_multiplication_of_powers_l1564_156492

theorem multiplication_of_powers (a : ℝ) : 4 * (a^2) * (a^3) = 4 * (a^5) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_powers_l1564_156492


namespace NUMINAMATH_CALUDE_bill_work_hours_l1564_156446

/-- Calculates the total pay for a given number of hours worked, 
    with a base rate for the first 40 hours and a double rate thereafter. -/
def calculatePay (baseRate : ℕ) (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    baseRate * hours
  else
    baseRate * 40 + baseRate * 2 * (hours - 40)

/-- Proves that working 50 hours results in a total pay of $1200, 
    given the specified pay rates. -/
theorem bill_work_hours (baseRate : ℕ) (totalPay : ℕ) :
  baseRate = 20 → totalPay = 1200 → ∃ hours, calculatePay baseRate hours = totalPay ∧ hours = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_bill_work_hours_l1564_156446


namespace NUMINAMATH_CALUDE_min_p_minus_q_equals_zero_l1564_156450

theorem min_p_minus_q_equals_zero
  (x y p q : ℤ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (eq1 : (3 : ℚ) / (x * p) = 8)
  (eq2 : (5 : ℚ) / (y * q) = 18)
  (hmin : ∀ x' y' p' q' : ℤ,
    x' ≠ 0 → y' ≠ 0 → p' ≠ 0 → q' ≠ 0 →
    (3 : ℚ) / (x' * p') = 8 →
    (5 : ℚ) / (y' * q') = 18 →
    (x' ≤ x ∧ y' ≤ y)) :
  p - q = 0 := by
sorry

end NUMINAMATH_CALUDE_min_p_minus_q_equals_zero_l1564_156450


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l1564_156437

def A (x y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; x, y]

theorem matrix_is_own_inverse (x y : ℝ) :
  A x y * A x y = 1 ↔ x = 15/2 ∧ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l1564_156437


namespace NUMINAMATH_CALUDE_fibonacci_mod_13_not_4_l1564_156449

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_mod_13_not_4 (n : ℕ) : fibonacci n % 13 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_mod_13_not_4_l1564_156449


namespace NUMINAMATH_CALUDE_expand_expression_l1564_156438

theorem expand_expression (x : ℝ) : (x - 2) * (x + 2) * (x^2 + x + 6) = x^4 + x^3 + 2*x^2 - 4*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1564_156438


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l1564_156498

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l1564_156498


namespace NUMINAMATH_CALUDE_set_relationships_l1564_156444

-- Define the sets M, N, and P
def M : Set ℚ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℚ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p/2 + 1/6}

-- State the theorem
theorem set_relationships : M ⊆ N ∧ N = P := by sorry

end NUMINAMATH_CALUDE_set_relationships_l1564_156444


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1564_156495

theorem trigonometric_identity (α φ : ℝ) : 
  Real.cos α ^ 2 + Real.cos φ ^ 2 + Real.cos (α + φ) ^ 2 - 
  2 * Real.cos α * Real.cos φ * Real.cos (α + φ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1564_156495


namespace NUMINAMATH_CALUDE_digits_of_2_12_times_5_8_l1564_156457

theorem digits_of_2_12_times_5_8 : 
  (Nat.log 10 (2^12 * 5^8) + 1 : ℕ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_2_12_times_5_8_l1564_156457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1564_156462

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a₁ d : ℝ) (h1 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 7 = 22)
  (h2 : arithmetic_sequence a₁ d 4 + arithmetic_sequence a₁ d 10 = 40) :
  d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1564_156462


namespace NUMINAMATH_CALUDE_hyperbola_focus_equation_l1564_156469

/-- Given a hyperbola of the form x²/m - y² = 1 with one focus at (-2√2, 0),
    prove that m = 7 -/
theorem hyperbola_focus_equation (m : ℝ) : 
  (∃ (x y : ℝ), x^2 / m - y^2 = 1) →  -- hyperbola equation
  ((-2 * Real.sqrt 2, 0) : ℝ × ℝ) ∈ {(x, y) | x^2 / m - y^2 = 1} →  -- focus condition
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_equation_l1564_156469


namespace NUMINAMATH_CALUDE_dress_price_inconsistency_l1564_156427

theorem dress_price_inconsistency :
  ¬∃ (D : ℝ), D > 0 ∧ 7 * D + 4 * 5 + 8 * 15 + 6 * 20 = 250 := by
  sorry

end NUMINAMATH_CALUDE_dress_price_inconsistency_l1564_156427


namespace NUMINAMATH_CALUDE_no_equal_xyz_l1564_156432

theorem no_equal_xyz : ¬∃ t : ℝ, (1 - 3*t = 2*t - 3) ∧ (1 - 3*t = 4*t^2 - 5*t + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_xyz_l1564_156432


namespace NUMINAMATH_CALUDE_isosceles_triangle_proof_l1564_156488

/-- Represents the sides of an isosceles triangle --/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_proof :
  let rope_length : ℝ := 20
  let triangle1 : IsoscelesTriangle := { base := 8, leg := 6 }
  let triangle2 : IsoscelesTriangle := { base := 4, leg := 8 }
  
  -- Part 1
  (triangle1.base + 2 * triangle1.leg = rope_length) ∧
  (triangle1.base - triangle1.leg = 2) ∧
  (is_valid_triangle triangle1.base triangle1.leg triangle1.leg) ∧
  
  -- Part 2
  (triangle2.base + 2 * triangle2.leg = rope_length) ∧
  (is_valid_triangle triangle2.base triangle2.leg triangle2.leg) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_proof_l1564_156488


namespace NUMINAMATH_CALUDE_remainder_problem_l1564_156470

theorem remainder_problem (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1564_156470


namespace NUMINAMATH_CALUDE_junk_mail_calculation_l1564_156460

/-- Calculates the total number of junk mail pieces per block -/
def total_junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) : ℕ :=
  houses_per_block * mail_per_house

/-- Theorem stating that the total junk mail per block is 640 -/
theorem junk_mail_calculation :
  total_junk_mail_per_block 20 32 = 640 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_calculation_l1564_156460


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l1564_156453

theorem binomial_expansion_sum_abs_coeff :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_abs_coeff_l1564_156453


namespace NUMINAMATH_CALUDE_red_balls_in_box_l1564_156497

theorem red_balls_in_box (total_balls : ℕ) (prob_red : ℚ) (num_red : ℕ) : 
  total_balls = 6 → 
  prob_red = 1/3 → 
  (num_red : ℚ) / total_balls = prob_red → 
  num_red = 2 := by
sorry

end NUMINAMATH_CALUDE_red_balls_in_box_l1564_156497


namespace NUMINAMATH_CALUDE_modulus_of_complex_product_l1564_156463

theorem modulus_of_complex_product : ∃ (z : ℂ), z = (Complex.I - 2) * (2 * Complex.I + 1) ∧ Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_product_l1564_156463


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1564_156451

theorem inequality_system_solution (x : ℝ) : 
  (3 * x + 1) / 2 > x ∧ 4 * (x - 2) ≤ x - 5 → -1 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1564_156451


namespace NUMINAMATH_CALUDE_max_value_theorem_l1564_156485

/-- The function f(x) = x^3 + x -/
def f (x : ℝ) : ℝ := x^3 + x

/-- The theorem stating the maximum value of a√(1 + b^2) -/
theorem max_value_theorem (a b : ℝ) (h : f (a^2) + f (2 * b^2 - 3) = 0) :
  ∃ (M : ℝ), M = (5 * Real.sqrt 2) / 4 ∧ a * Real.sqrt (1 + b^2) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1564_156485


namespace NUMINAMATH_CALUDE_collinear_points_unique_k_l1564_156440

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that k = -33 is the unique value for which
    the points (1,4), (3,-2), and (6, k/3) are collinear -/
theorem collinear_points_unique_k :
  ∃! k : ℝ, collinear (1, 4) (3, -2) (6, k/3) ∧ k = -33 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_unique_k_l1564_156440


namespace NUMINAMATH_CALUDE_multiplication_factor_exists_l1564_156425

theorem multiplication_factor_exists (x : ℝ) (hx : x = 2.6666666666666665) :
  ∃ y : ℝ, Real.sqrt ((x * y) / 3) = x ∧ abs (y - 8) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_factor_exists_l1564_156425


namespace NUMINAMATH_CALUDE_expression_value_l1564_156416

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x + 5 - 4 * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1564_156416


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_length_l1564_156435

-- Define the rhombus properties
def diagonal1 : ℝ := 7.4
def area : ℝ := 21.46

-- Theorem to prove
theorem rhombus_other_diagonal_length :
  let diagonal2 := (2 * area) / diagonal1
  ∃ ε > 0, abs (diagonal2 - 5.8) < ε :=
by sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_length_l1564_156435


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1564_156414

/-- Given that the total marks in physics, chemistry, and mathematics is 180 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 90. -/
theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: marks in physics, C: marks in chemistry, M: marks in mathematics
  (h : P + C + M = P + 180) -- Given condition
  : (C + M) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l1564_156414


namespace NUMINAMATH_CALUDE_anna_phone_chargers_l1564_156429

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_phone_chargers_l1564_156429


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l1564_156467

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence 3 (4*y) 5 = 768 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l1564_156467


namespace NUMINAMATH_CALUDE_min_sum_of_constrained_integers_l1564_156496

theorem min_sum_of_constrained_integers (x y : ℕ) 
  (h1 : x - y < 1)
  (h2 : 2 * x - y > 2)
  (h3 : x < 5) :
  ∃ (a b : ℕ), a + b = 6 ∧ 
    (∀ (x' y' : ℕ), x' - y' < 1 → 2 * x' - y' > 2 → x' < 5 → x' + y' ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_constrained_integers_l1564_156496


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1564_156423

theorem absolute_value_equality (x : ℝ) :
  |x - 3| = |x - 5| → x = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1564_156423


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1564_156447

/-- Given vectors a and b, if the angle between them is π/6, then the second component of b is √3. -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b.1 = 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.cos (π / 6) →
  b.2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1564_156447


namespace NUMINAMATH_CALUDE_shirt_cost_l1564_156424

theorem shirt_cost (total_money : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (money_left : ℕ) :
  total_money = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  money_left = 74 →
  ∃ shirt_cost : ℕ, shirt_cost * num_shirts + pants_cost = total_money - money_left ∧ shirt_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_l1564_156424


namespace NUMINAMATH_CALUDE_impossible_tiling_l1564_156445

/-- Represents an L-tromino -/
structure LTromino :=
  (cells : Fin 3 → (Fin 5 × Fin 7))

/-- Represents a tiling of a 5x7 rectangle with L-trominos -/
structure Tiling :=
  (trominos : List LTromino)
  (coverage : Fin 5 → Fin 7 → ℕ)

/-- Theorem stating the impossibility of tiling a 5x7 rectangle with L-trominos 
    such that each cell is covered by the same number of trominos -/
theorem impossible_tiling : 
  ∀ (t : Tiling), ¬(∀ (i : Fin 5) (j : Fin 7), ∃ (k : ℕ), t.coverage i j = k) :=
sorry

end NUMINAMATH_CALUDE_impossible_tiling_l1564_156445


namespace NUMINAMATH_CALUDE_chris_age_l1564_156491

/-- The ages of four friends satisfying certain conditions -/
def FriendsAges (a b c d : ℝ) : Prop :=
  -- The average age is 12
  (a + b + c + d) / 4 = 12 ∧
  -- Five years ago, Chris was twice as old as Amy
  c - 5 = 2 * (a - 5) ∧
  -- In 2 years, Ben's age will be three-quarters of Amy's age
  b + 2 = 3/4 * (a + 2) ∧
  -- Diana is 15 years old
  d = 15

/-- Chris's age is 16 given the conditions -/
theorem chris_age (a b c d : ℝ) (h : FriendsAges a b c d) : c = 16 := by
  sorry

end NUMINAMATH_CALUDE_chris_age_l1564_156491


namespace NUMINAMATH_CALUDE_math_expressions_evaluation_l1564_156490

theorem math_expressions_evaluation :
  (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt (x * y) = Real.sqrt x * Real.sqrt y) →
  (∀ (x : ℝ), x ≥ 0 → (Real.sqrt x) ^ 2 = x) →
  (∀ (x y : ℝ), y ≠ 0 → Real.sqrt (x / y) = Real.sqrt x / Real.sqrt y) →
  (Real.sqrt 5 * Real.sqrt 15 - Real.sqrt 12 = 3 * Real.sqrt 3) ∧
  ((Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2) = 1) ∧
  ((Real.sqrt 20 + 5) / Real.sqrt 5 = 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_math_expressions_evaluation_l1564_156490


namespace NUMINAMATH_CALUDE_remainder_of_2543_base12_div_7_l1564_156480

/-- Converts a base-12 number to decimal --/
def base12ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (12 ^ i)) 0

/-- The base-12 representation of 2543₁₂ --/
def number : List Nat := [3, 4, 5, 2]

theorem remainder_of_2543_base12_div_7 :
  (base12ToDecimal number) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2543_base12_div_7_l1564_156480


namespace NUMINAMATH_CALUDE_extra_bananas_l1564_156408

theorem extra_bananas (total_children : ℕ) (original_bananas_per_child : ℕ) (absent_children : ℕ) : 
  total_children = 740 →
  original_bananas_per_child = 2 →
  absent_children = 370 →
  (total_children * original_bananas_per_child) / (total_children - absent_children) - original_bananas_per_child = 2 := by
sorry

end NUMINAMATH_CALUDE_extra_bananas_l1564_156408


namespace NUMINAMATH_CALUDE_kelly_snacks_total_weight_l1564_156464

theorem kelly_snacks_total_weight 
  (peanuts_weight : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : peanuts_weight = 0.1)
  (h2 : raisins_weight = 0.4) : 
  peanuts_weight + raisins_weight = 0.5 := by
sorry

end NUMINAMATH_CALUDE_kelly_snacks_total_weight_l1564_156464


namespace NUMINAMATH_CALUDE_subset_relation_l1564_156415

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem subset_relation :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) := by
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1564_156415


namespace NUMINAMATH_CALUDE_worker_wage_problem_l1564_156461

/-- Represents the daily wages and work days of three workers -/
structure WorkerData where
  a_wage : ℚ
  b_wage : ℚ
  c_wage : ℚ
  a_days : ℕ
  b_days : ℕ
  c_days : ℕ

/-- The theorem statement for the worker wage problem -/
theorem worker_wage_problem (data : WorkerData) 
  (h_ratio : data.a_wage / 3 = data.b_wage / 4 ∧ data.b_wage / 4 = data.c_wage / 5)
  (h_days : data.a_days = 6 ∧ data.b_days = 9 ∧ data.c_days = 4)
  (h_total : data.a_wage * data.a_days + data.b_wage * data.b_days + data.c_wage * data.c_days = 1850) :
  data.c_wage = 125 := by
  sorry

end NUMINAMATH_CALUDE_worker_wage_problem_l1564_156461


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_l1564_156487

/-- Given a hyperbola 4x^2 - y^2 = 1, the value of t for which one of its asymptotes
    is perpendicular to the line tx + y + 1 = 0 is ±1/2 -/
theorem hyperbola_asymptote_perpendicular (x y t : ℝ) : 
  (4 * x^2 - y^2 = 1) → 
  (∃ (m : ℝ), (y = m * x ∨ y = -m * x) ∧ 
              (m * (-1/t) = -1 ∨ (-m) * (-1/t) = -1)) → 
  (t = 1/2 ∨ t = -1/2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_perpendicular_l1564_156487


namespace NUMINAMATH_CALUDE_sphere_volume_hexagonal_prism_l1564_156452

/-- The volume of a sphere circumscribing a hexagonal prism -/
theorem sphere_volume_hexagonal_prism (h : ℝ) (p : ℝ) : 
  h = Real.sqrt 3 →
  p = 3 →
  (4 / 3 * Real.pi : ℝ) = (4 / 3 * Real.pi * (((h^2 + (p / 6)^2) / 4)^(3/2))) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_hexagonal_prism_l1564_156452


namespace NUMINAMATH_CALUDE_initial_money_l1564_156403

theorem initial_money (x : ℝ) : x + 13 + 3 = 18 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_l1564_156403


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1564_156417

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l1564_156417


namespace NUMINAMATH_CALUDE_bird_difference_l1564_156468

/-- Proves the difference between white birds and original grey birds -/
theorem bird_difference (initial_grey : ℕ) (total_remaining : ℕ) 
  (h1 : initial_grey = 40)
  (h2 : total_remaining = 66) :
  total_remaining - initial_grey / 2 - initial_grey = 6 := by
  sorry

end NUMINAMATH_CALUDE_bird_difference_l1564_156468


namespace NUMINAMATH_CALUDE_points_per_enemy_is_10_l1564_156472

/-- The number of points for killing one enemy in Tom's game -/
def points_per_enemy : ℕ := sorry

/-- The number of enemies Tom killed -/
def enemies_killed : ℕ := 150

/-- Tom's total score -/
def total_score : ℕ := 2250

/-- The bonus multiplier for killing at least 100 enemies -/
def bonus_multiplier : ℚ := 1.5

theorem points_per_enemy_is_10 :
  points_per_enemy = 10 ∧
  enemies_killed ≥ 100 ∧
  (points_per_enemy * enemies_killed : ℚ) * bonus_multiplier = total_score := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_is_10_l1564_156472


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1564_156477

theorem reciprocal_sum_theorem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 3 * x * y + 2) :
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1564_156477


namespace NUMINAMATH_CALUDE_g_of_2_eq_0_l1564_156476

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem g_of_2_eq_0 : g 2 = 0 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_0_l1564_156476


namespace NUMINAMATH_CALUDE_tangent_circle_height_l1564_156410

/-- A circle tangent to y = x^3 at two points -/
structure TangentCircle where
  a : ℝ  -- x-coordinate of the tangent point
  b : ℝ  -- y-coordinate of the circle's center
  r : ℝ  -- radius of the circle

/-- The circle is tangent to y = x^3 at (a, a^3) and (-a, a^3) -/
def is_tangent (c : TangentCircle) : Prop :=
  c.a^2 + (c.a^3 - c.b)^2 = c.r^2 ∧
  c.a^6 + (1 - 2*c.b)*c.a^3 + c.b^2 - c.r^2 = 0

/-- The center of the circle is higher than the tangent points by 1/2 -/
theorem tangent_circle_height (c : TangentCircle) (h : is_tangent c) : 
  c.b - c.a^3 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_height_l1564_156410


namespace NUMINAMATH_CALUDE_solve_equation_l1564_156466

theorem solve_equation : ∃ (A : ℕ), A < 10 ∧ A * 100 + 72 - 23 = 549 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1564_156466


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_zero_l1564_156458

theorem sum_of_even_coefficients_zero (a : Fin 7 → ℝ) :
  (∀ x : ℝ, (x - 1)^6 = a 0 * x^6 + a 1 * x^5 + a 2 * x^4 + a 3 * x^3 + a 4 * x^2 + a 5 * x + a 6) →
  a 0 + a 2 + a 4 + a 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_zero_l1564_156458


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l1564_156473

theorem binomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l1564_156473


namespace NUMINAMATH_CALUDE_right_triangle_area_l1564_156442

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 64) (hb : b^2 = 36) (hc : c^2 = 121) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1564_156442


namespace NUMINAMATH_CALUDE_work_completion_time_l1564_156455

/-- Given workers A, B, and C, where:
    - A can complete the work in 6 days
    - C can complete the work in 7.5 days
    - A, B, and C together complete the work in 2 days
    Prove that B can complete the work alone in 5 days -/
theorem work_completion_time (A B C : ℝ) 
  (hA : A = 1 / 6)  -- A's work rate per day
  (hC : C = 1 / 7.5)  -- C's work rate per day
  (hABC : A + B + C = 1 / 2)  -- Combined work rate of A, B, and C
  : B = 1 / 5 := by  -- B's work rate per day
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1564_156455


namespace NUMINAMATH_CALUDE_sock_combinations_proof_l1564_156434

/-- The number of ways to choose 4 socks out of 7, with at least one red sock -/
def sockCombinations : ℕ := 20

/-- The total number of socks -/
def totalSocks : ℕ := 7

/-- The number of socks to be chosen -/
def chosenSocks : ℕ := 4

/-- The number of non-red socks -/
def nonRedSocks : ℕ := 6

theorem sock_combinations_proof :
  sockCombinations = Nat.choose totalSocks chosenSocks - Nat.choose nonRedSocks chosenSocks :=
by sorry

end NUMINAMATH_CALUDE_sock_combinations_proof_l1564_156434


namespace NUMINAMATH_CALUDE_divisible_by_27_l1564_156486

theorem divisible_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  ∃ k : ℤ, x + y + z = 27 * k := by
sorry

end NUMINAMATH_CALUDE_divisible_by_27_l1564_156486


namespace NUMINAMATH_CALUDE_biology_books_count_l1564_156433

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different chemistry books -/
def chem_books : ℕ := 8

/-- The total number of ways to choose 2 books of each type -/
def total_ways : ℕ := 1260

/-- The number of different biology books -/
def bio_books : ℕ := 10

theorem biology_books_count :
  choose_two bio_books * choose_two chem_books = total_ways :=
sorry

#check biology_books_count

end NUMINAMATH_CALUDE_biology_books_count_l1564_156433


namespace NUMINAMATH_CALUDE_event_attendees_l1564_156475

theorem event_attendees (num_children : ℕ) (num_adults : ℕ) : 
  num_children = 28 → 
  num_children = 2 * num_adults → 
  num_children + num_adults = 42 := by
  sorry

end NUMINAMATH_CALUDE_event_attendees_l1564_156475


namespace NUMINAMATH_CALUDE_intersection_count_l1564_156426

/-- Represents a line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The five lines given in the problem --/
def line1 : Line := { a := 3, b := -2, c := 9 }
def line2 : Line := { a := 6, b := 4, c := -12 }
def line3 : Line := { a := 1, b := 0, c := 3 }
def line4 : Line := { a := 0, b := 1, c := 1 }
def line5 : Line := { a := 2, b := 1, c := -1 }

/-- Determines if two lines intersect --/
def intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- Counts the number of unique intersection points --/
def countIntersections (lines : List Line) : ℕ :=
  sorry

theorem intersection_count :
  countIntersections [line1, line2, line3, line4, line5] = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l1564_156426


namespace NUMINAMATH_CALUDE_inequality_proof_l1564_156436

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1564_156436


namespace NUMINAMATH_CALUDE_system_inequalities_solution_l1564_156478

theorem system_inequalities_solution (x : ℝ) :
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_inequalities_solution_l1564_156478


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1564_156439

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 7) * (Real.sqrt 9 / Real.sqrt 11) * (Real.sqrt 13 / Real.sqrt 15) = 
  (3 * Real.sqrt 3003) / 231 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1564_156439


namespace NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l1564_156499

/-- The function f(x) = (x-1)(x-a)^2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)^2

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := (x - a) * (3*x - a - 2)

theorem function_minimum_and_tangent_line 
  (h₁ : ∃ δ > 0, ∀ x ∈ Set.Ioo (-δ) δ, f (-2) 0 ≤ f (-2) x) :
  (a = -2) ∧ 
  (∃ xp : ℝ, xp ≠ 1 ∧ f' (-2) xp = f' (-2) 1 ∧ 
    (9 : ℝ) * xp - f (-2) xp + 23 = 0) := by
  sorry


end NUMINAMATH_CALUDE_function_minimum_and_tangent_line_l1564_156499


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l1564_156474

/-- Given a rope of length 18 cm forming an isosceles triangle with one side of 5 cm,
    the length of the other two sides can be either 5 cm or 6.5 cm. -/
theorem isosceles_triangle_sides (rope_length : ℝ) (given_side : ℝ) : 
  rope_length = 18 → given_side = 5 → 
  ∃ (other_side : ℝ), (other_side = 5 ∨ other_side = 6.5) ∧ 
  ((2 * other_side + given_side = rope_length) ∨ 
   (2 * given_side + other_side = rope_length)) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l1564_156474


namespace NUMINAMATH_CALUDE_factorial_315_trailing_zeros_l1564_156484

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- The factorial of 315 ends with 77 zeros -/
theorem factorial_315_trailing_zeros :
  trailingZeros 315 = 77 := by
  sorry

end NUMINAMATH_CALUDE_factorial_315_trailing_zeros_l1564_156484


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l1564_156448

theorem triangle_trig_identity (A B C : ℝ) 
  (h1 : A + B + C = Real.pi)
  (h2 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = 1) :
  (Real.cos (2*A) + Real.cos (2*B) + Real.cos (2*C)) / (Real.cos A + Real.cos B + Real.cos C) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l1564_156448


namespace NUMINAMATH_CALUDE_first_eligible_retirement_year_l1564_156404

/-- Rule of 70 retirement eligibility -/
def rule_of_70 (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed ≥ 70

/-- Year of hire -/
def hire_year : ℕ := 1990

/-- Age at hire -/
def age_at_hire : ℕ := 32

/-- First year of retirement eligibility -/
def retirement_year : ℕ := 2009

/-- Theorem: The employee is first eligible to retire in 2009 -/
theorem first_eligible_retirement_year :
  rule_of_70 (age_at_hire + (retirement_year - hire_year)) (retirement_year - hire_year) ∧
  ∀ (year : ℕ), year < retirement_year → 
    ¬rule_of_70 (age_at_hire + (year - hire_year)) (year - hire_year) :=
by sorry

end NUMINAMATH_CALUDE_first_eligible_retirement_year_l1564_156404


namespace NUMINAMATH_CALUDE_customers_left_l1564_156405

/-- A problem about customers leaving a waiter's section. -/
theorem customers_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : 
  initial_customers = 22 → remaining_tables = 2 → people_per_table = 4 →
  initial_customers - (remaining_tables * people_per_table) = 14 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l1564_156405


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l1564_156471

theorem max_value_of_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z ≤ 17) ∧ 
  (∃ (a b c : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ 
                  (10 ≤ b ∧ b ≤ 99) ∧ 
                  (10 ≤ c ∧ c ≤ 99) ∧ 
                  ((a + b + c) / 3 = 60) ∧ 
                  ((a + b) / c = 17)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l1564_156471


namespace NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1564_156443

theorem triangle_pentagon_side_ratio : ∀ (t p : ℝ),
  (3 * t = 24) →  -- Perimeter of equilateral triangle
  (5 * p = 24) →  -- Perimeter of regular pentagon
  (t / p = 5 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_pentagon_side_ratio_l1564_156443


namespace NUMINAMATH_CALUDE_rope_section_length_l1564_156479

theorem rope_section_length 
  (total_length : ℝ) 
  (art_fraction : ℝ) 
  (friend_fraction : ℝ) 
  (num_sections : ℕ) :
  total_length = 50 →
  art_fraction = 1/5 →
  friend_fraction = 1/2 →
  num_sections = 10 →
  let remaining_after_art := total_length * (1 - art_fraction)
  let remaining_after_friend := remaining_after_art * (1 - friend_fraction)
  remaining_after_friend / num_sections = 2 := by
sorry

end NUMINAMATH_CALUDE_rope_section_length_l1564_156479


namespace NUMINAMATH_CALUDE_smallest_angle_is_90_l1564_156412

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic sequence
  d : ℝ
  -- Assertion that angles are in arithmetic sequence
  angle_sequence : List ℝ := [a, a + d, a + 2*d, a + 3*d]
  -- Assertion that the sum of any two consecutive angles is 180°
  consecutive_sum : a + (a + d) = 180 ∧ (a + d) + (a + 2*d) = 180 ∧ (a + 2*d) + (a + 3*d) = 180
  -- Assertion that the second largest angle is 150°
  second_largest : a + 2*d = 150

/-- 
Theorem: In a trapezoid where the angles form an arithmetic sequence 
and the second largest angle is 150°, the smallest angle measures 90°.
-/
theorem smallest_angle_is_90 (t : ArithmeticTrapezoid) : t.a = 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_is_90_l1564_156412


namespace NUMINAMATH_CALUDE_rocket_coaster_capacity_l1564_156420

/-- Represents a roller coaster with two types of cars -/
structure RollerCoaster where
  total_cars : ℕ
  four_passenger_cars : ℕ
  six_passenger_cars : ℕ

/-- Calculates the total capacity of a roller coaster -/
def total_capacity (rc : RollerCoaster) : ℕ :=
  rc.four_passenger_cars * 4 + rc.six_passenger_cars * 6

/-- The Rocket Coaster specification -/
def rocket_coaster : RollerCoaster := {
  total_cars := 15,
  four_passenger_cars := 9,
  six_passenger_cars := 15 - 9
}

theorem rocket_coaster_capacity :
  total_capacity rocket_coaster = 72 := by
  sorry

end NUMINAMATH_CALUDE_rocket_coaster_capacity_l1564_156420


namespace NUMINAMATH_CALUDE_karen_start_time_l1564_156454

/-- Proves that Karen starts 4 minutes late in the car race with Tom -/
theorem karen_start_time (karen_speed tom_speed tom_distance karen_win_margin : ℝ) 
  (h1 : karen_speed = 60) 
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) : 
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry


end NUMINAMATH_CALUDE_karen_start_time_l1564_156454


namespace NUMINAMATH_CALUDE_candidate_percentage_l1564_156481

theorem candidate_percentage (passing_marks total_marks : ℕ) 
  (first_candidate_marks second_candidate_marks : ℕ) : 
  passing_marks = 160 →
  first_candidate_marks = passing_marks - 40 →
  second_candidate_marks = passing_marks + 20 →
  second_candidate_marks = total_marks * 30 / 100 →
  first_candidate_marks * 100 / total_marks = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_percentage_l1564_156481


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l1564_156493

def vector_problem (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  let sum := (a.1 + b.1, a.2 + b.2)
  dot_product = 10 ∧ 
  magnitude sum = 5 * Real.sqrt 2 ∧ 
  a = (2, 1) →
  magnitude b = 5

theorem vector_magnitude_proof : 
  ∀ (a b : ℝ × ℝ), vector_problem a b :=
sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l1564_156493
