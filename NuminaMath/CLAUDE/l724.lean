import Mathlib

namespace NUMINAMATH_CALUDE_sequence_problem_l724_72494

-- Define the arithmetic sequence
def is_arithmetic_sequence (x y z w : ℝ) : Prop :=
  y - x = z - y ∧ z - y = w - z

-- Define the geometric sequence
def is_geometric_sequence (x y z w v : ℝ) : Prop :=
  ∃ r : ℝ, y = x * r ∧ z = y * r ∧ w = z * r ∧ v = w * r

theorem sequence_problem (a b c d e : ℝ) :
  is_arithmetic_sequence (-1) a b (-4) →
  is_geometric_sequence (-1) c d e (-4) →
  c = -1 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l724_72494


namespace NUMINAMATH_CALUDE_arithmetic_sequence_contains_2017_l724_72406

/-- An arithmetic sequence containing 25, 41, and 65 also contains 2017 -/
theorem arithmetic_sequence_contains_2017 (a₁ d : ℤ) (k n m : ℕ) 
  (h_pos : d > 0)
  (h_25 : 25 = a₁ + k * d)
  (h_41 : 41 = a₁ + n * d)
  (h_65 : 65 = a₁ + m * d) :
  ∃ l : ℕ, 2017 = a₁ + l * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_contains_2017_l724_72406


namespace NUMINAMATH_CALUDE_hyperbola_equation_l724_72460

/-- A hyperbola with specific properties -/
structure Hyperbola where
  conjugate_axis_length : ℝ
  eccentricity : ℝ
  focal_length : ℝ
  point_m : ℝ × ℝ
  point_p : ℝ × ℝ
  point_q : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / 25 - x^2 / 75 = 1

/-- Theorem stating the standard equation of the specific hyperbola -/
theorem hyperbola_equation (h : Hyperbola)
  (h_conjugate : h.conjugate_axis_length = 12)
  (h_eccentricity : h.eccentricity = 5/4)
  (h_focal : h.focal_length = 26)
  (h_point_m : h.point_m = (0, 12))
  (h_point_p : h.point_p = (-3, 2 * Real.sqrt 7))
  (h_point_q : h.point_q = (-6 * Real.sqrt 2, -7)) :
  ∀ x y, standard_equation h x y ↔ 
    (x = h.point_m.1 ∧ y = h.point_m.2) ∨
    (x = h.point_p.1 ∧ y = h.point_p.2) ∨
    (x = h.point_q.1 ∧ y = h.point_q.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l724_72460


namespace NUMINAMATH_CALUDE_two_men_absent_l724_72478

/-- Represents the work completion scenario -/
structure WorkCompletion where
  total_men : ℕ
  planned_days : ℕ
  actual_days : ℕ

/-- Calculates the number of absent men given the work completion scenario -/
def calculate_absent_men (w : WorkCompletion) : ℕ :=
  w.total_men - (w.total_men * w.planned_days) / w.actual_days

/-- Theorem stating that 2 men became absent in the given scenario -/
theorem two_men_absent (w : WorkCompletion) 
  (h1 : w.total_men = 22)
  (h2 : w.planned_days = 20)
  (h3 : w.actual_days = 22) : 
  calculate_absent_men w = 2 := by
  sorry

#eval calculate_absent_men ⟨22, 20, 22⟩

end NUMINAMATH_CALUDE_two_men_absent_l724_72478


namespace NUMINAMATH_CALUDE_smallest_four_digit_pascal_l724_72465

/-- Pascal's triangle is represented as a function from row and column to natural number -/
def pascal : ℕ → ℕ → ℕ
  | 0, _ => 1
  | n + 1, 0 => 1
  | n + 1, k + 1 => pascal n k + pascal n (k + 1)

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_pascal :
  ∃ (r c : ℕ), isFourDigit (pascal r c) ∧
    ∀ (r' c' : ℕ), isFourDigit (pascal r' c') → pascal r c ≤ pascal r' c' :=
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_pascal_l724_72465


namespace NUMINAMATH_CALUDE_value_of_a_l724_72477

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a*x^3

theorem value_of_a : 
  ∀ a : ℝ, (deriv (f a)) 1 = 5 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l724_72477


namespace NUMINAMATH_CALUDE_trig_values_150_degrees_l724_72493

/-- Given a point P on the unit circle corresponding to an angle of 150°, 
    prove that tan(150°) = -√3 and sin(150°) = √3/2 -/
theorem trig_values_150_degrees : 
  ∀ (P : ℝ × ℝ), 
  (P.1^2 + P.2^2 = 1) →  -- P is on the unit circle
  (P.1 = -1/2 ∧ P.2 = Real.sqrt 3 / 2) →  -- P corresponds to 150°
  (Real.tan (150 * π / 180) = -Real.sqrt 3 ∧ 
   Real.sin (150 * π / 180) = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_values_150_degrees_l724_72493


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l724_72479

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ 
  (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l724_72479


namespace NUMINAMATH_CALUDE_cost_price_percentage_l724_72438

theorem cost_price_percentage (cost_price selling_price : ℝ) (profit_percent : ℝ) :
  profit_percent = 150 →
  selling_price = cost_price + (profit_percent / 100) * cost_price →
  (cost_price / selling_price) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l724_72438


namespace NUMINAMATH_CALUDE_expression_evaluation_l724_72404

/-- Proves that the given expression evaluates to -8 when a = 2 and b = -1 -/
theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  3 * (2 * a^2 * b - 3 * a * b^2 - 1) - 2 * (3 * a^2 * b - 4 * a * b^2 + 1) - 1 = -8 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l724_72404


namespace NUMINAMATH_CALUDE_smallest_k_for_two_trailing_zeros_l724_72489

theorem smallest_k_for_two_trailing_zeros : ∃ k : ℕ+, k = 13 ∧ 
  (∀ m : ℕ+, m < k → ¬(100 ∣ Nat.choose (2 * m) m)) ∧ 
  (100 ∣ Nat.choose (2 * k) k) := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_two_trailing_zeros_l724_72489


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l724_72480

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  (x + 3) / (x^2 - 4) / (2 - (x + 1) / (x + 2)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l724_72480


namespace NUMINAMATH_CALUDE_mrs_hilt_money_l724_72439

/-- Mrs. Hilt's pencil purchase problem -/
theorem mrs_hilt_money (pencil_cost remaining_money : ℕ) 
  (h1 : pencil_cost = 11)
  (h2 : remaining_money = 4) : 
  pencil_cost + remaining_money = 15 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_money_l724_72439


namespace NUMINAMATH_CALUDE_cost_ratio_when_b_tripled_x_halved_l724_72414

/-- The cost ratio when b is tripled and x is halved in the formula C = at(bx)^6 -/
theorem cost_ratio_when_b_tripled_x_halved (a t b x : ℝ) :
  let original_cost := a * t * (b * x)^6
  let new_cost := a * t * (3 * b * (x / 2))^6
  (new_cost / original_cost) * 100 = 1139.0625 := by
sorry

end NUMINAMATH_CALUDE_cost_ratio_when_b_tripled_x_halved_l724_72414


namespace NUMINAMATH_CALUDE_proportion_solution_l724_72492

theorem proportion_solution (x : ℝ) : (0.60 / x = 6 / 2) → x = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l724_72492


namespace NUMINAMATH_CALUDE_min_value_f_l724_72418

theorem min_value_f (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l724_72418


namespace NUMINAMATH_CALUDE_extreme_value_point_l724_72499

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := x^3 - 3*x + 2

-- Theorem stating that -2 is an extreme value point of f
theorem extreme_value_point : 
  ∃ (δ : ℝ), δ > 0 ∧ 
  (∀ x₁ ∈ Set.Ioo (-2 - δ) (-2), f' x₁ < 0) ∧
  (∀ x₂ ∈ Set.Ioo (-2) (-2 + δ), f' x₂ > 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_point_l724_72499


namespace NUMINAMATH_CALUDE_pascal_triangle_first_25_rows_sum_l724_72433

/-- The number of elements in the nth row of Pascal's Triangle -/
def pascal_row_elements (n : ℕ) : ℕ := n + 1

/-- The sum of elements in the first n rows of Pascal's Triangle -/
def pascal_triangle_sum (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

theorem pascal_triangle_first_25_rows_sum :
  pascal_triangle_sum 24 = 325 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_first_25_rows_sum_l724_72433


namespace NUMINAMATH_CALUDE_complex_power_of_four_l724_72445

theorem complex_power_of_four :
  (3 * Complex.cos (30 * Real.pi / 180) + 3 * Complex.I * Complex.sin (30 * Real.pi / 180)) ^ 4 =
  -40.5 + 40.5 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_of_four_l724_72445


namespace NUMINAMATH_CALUDE_no_right_triangle_with_sqrt_2016_side_l724_72481

theorem no_right_triangle_with_sqrt_2016_side : ¬ ∃ (a b : ℕ) (c : ℝ), 
  c = Real.sqrt 2016 ∧ (a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ c * c + b * b = a * a) :=
sorry

end NUMINAMATH_CALUDE_no_right_triangle_with_sqrt_2016_side_l724_72481


namespace NUMINAMATH_CALUDE_smallest_shift_l724_72469

-- Define a periodic function g with period 30
def g (x : ℝ) : ℝ := sorry

-- State the periodicity of g
axiom g_periodic (x : ℝ) : g (x + 30) = g x

-- Define the property we want to prove
def property (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 3) = g (x / 3)

-- State the theorem
theorem smallest_shift :
  (∃ b > 0, property b) ∧ 
  (∀ b > 0, property b → b ≥ 90) ∧
  property 90 := by sorry

end NUMINAMATH_CALUDE_smallest_shift_l724_72469


namespace NUMINAMATH_CALUDE_unique_periodic_modulus_l724_72422

/-- The binomial coefficient C(n,k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence x_n = C(2n, n) -/
def x_seq (n : ℕ) : ℕ := binomial (2 * n) n

/-- A sequence is eventually periodic modulo m if there exist positive integers N and T
    such that for all n ≥ N, x_(n+T) ≡ x_n (mod m) -/
def eventually_periodic_mod (x : ℕ → ℕ) (m : ℕ) : Prop :=
  ∃ (N T : ℕ), T > 0 ∧ ∀ n ≥ N, x (n + T) % m = x n % m

/-- The main theorem: 2 is the only positive integer h > 1 such that 
    the sequence x_n = C(2n, n) is eventually periodic modulo h -/
theorem unique_periodic_modulus :
  ∀ h : ℕ, h > 1 → (eventually_periodic_mod x_seq h ↔ h = 2) := by sorry

end NUMINAMATH_CALUDE_unique_periodic_modulus_l724_72422


namespace NUMINAMATH_CALUDE_glow_interval_l724_72488

/-- The time interval between glows of a light, given the total time period and number of glows. -/
theorem glow_interval (total_time : ℕ) (num_glows : ℝ) 
  (h1 : total_time = 4969)
  (h2 : num_glows = 382.2307692307692) :
  ∃ (interval : ℝ), abs (interval - 13) < 0.0000001 ∧ interval = total_time / num_glows :=
sorry

end NUMINAMATH_CALUDE_glow_interval_l724_72488


namespace NUMINAMATH_CALUDE_dog_walking_homework_diff_l724_72474

/-- Represents the time in minutes for various activities -/
structure ActivityTimes where
  total : ℕ
  homework : ℕ
  cleaning : ℕ
  trash : ℕ
  remaining : ℕ

/-- Calculates the time spent walking the dog -/
def walkingTime (t : ActivityTimes) : ℕ :=
  t.total - t.remaining - (t.homework + t.cleaning + t.trash)

/-- Theorem stating the difference between dog walking and homework time -/
theorem dog_walking_homework_diff (t : ActivityTimes) : 
  t.total = 120 ∧ 
  t.homework = 30 ∧ 
  t.cleaning = t.homework / 2 ∧ 
  t.trash = t.homework / 6 ∧ 
  t.remaining = 35 → 
  walkingTime t - t.homework = 5 := by
  sorry


end NUMINAMATH_CALUDE_dog_walking_homework_diff_l724_72474


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l724_72457

/-- A quadratic polynomial of the form x^2 - (p+q)x + pq -/
def QuadraticPolynomial (p q : ℝ) : ℝ → ℝ := fun x ↦ x^2 - (p+q)*x + p*q

/-- The composite function p(p(x)) -/
def CompositePolynomial (p q : ℝ) : ℝ → ℝ :=
  fun x ↦ let px := QuadraticPolynomial p q x
          (QuadraticPolynomial p q) px

/-- Predicate that checks if a polynomial has exactly four distinct real roots -/
def HasFourDistinctRealRoots (f : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem to be proved -/
theorem quadratic_polynomial_property :
  ∃ (p q : ℝ),
    HasFourDistinctRealRoots (CompositePolynomial p q) ∧
    (∀ (p' q' : ℝ),
      HasFourDistinctRealRoots (CompositePolynomial p' q') →
      (let f := QuadraticPolynomial p q
       let f' := QuadraticPolynomial p' q'
       ∀ (a b c d : ℝ),
         f a = a → f b = b → f c = c → f d = d →
         ∀ (a' b' c' d' : ℝ),
           f' a' = a' → f' b' = b' → f' c' = c' → f' d' = d' →
           a * b * c * d ≥ a' * b' * c' * d')) →
    QuadraticPolynomial p q 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l724_72457


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l724_72403

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 - 2*x^5 + x^4 - x^2 - 2*x + 1 = 
  ((x^2 - 1) * (x - 2) * (x + 2)) * q + (2*x^3 - 9*x^2 + 3*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l724_72403


namespace NUMINAMATH_CALUDE_infinite_sum_n_over_n4_plus_1_l724_72437

/-- The infinite sum of n / (n^4 + 1) from n = 1 to infinity equals 1. -/
theorem infinite_sum_n_over_n4_plus_1 : 
  ∑' n : ℕ+, (n : ℝ) / ((n : ℝ)^4 + 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_infinite_sum_n_over_n4_plus_1_l724_72437


namespace NUMINAMATH_CALUDE_original_to_half_ratio_l724_72472

theorem original_to_half_ratio (x : ℝ) (h : x / 2 = 9) : x / (x / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_original_to_half_ratio_l724_72472


namespace NUMINAMATH_CALUDE_maria_oatmeal_cookies_l724_72495

/-- The number of oatmeal cookies Maria had -/
def num_oatmeal_cookies (cookies_per_bag : ℕ) (num_chocolate_chip : ℕ) (num_baggies : ℕ) : ℕ :=
  num_baggies * cookies_per_bag - num_chocolate_chip

/-- Theorem stating that Maria had 2 oatmeal cookies -/
theorem maria_oatmeal_cookies :
  num_oatmeal_cookies 5 33 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_maria_oatmeal_cookies_l724_72495


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l724_72441

theorem unique_digit_divisibility : ∃! (B : ℕ), B < 10 ∧ 45 % B = 0 ∧ (451 * 10 + B * 1 + 7) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l724_72441


namespace NUMINAMATH_CALUDE_min_sum_proof_l724_72405

/-- The minimum sum of m and n satisfying the conditions -/
def min_sum : ℕ := 106

/-- The value of m in the minimal solution -/
def m_min : ℕ := 3

/-- The value of n in the minimal solution -/
def n_min : ℕ := 103

/-- Checks if two numbers are congruent modulo 1000 -/
def congruent_mod_1000 (a b : ℕ) : Prop :=
  a % 1000 = b % 1000

theorem min_sum_proof :
  ∀ m n : ℕ,
    n > m →
    m ≥ 1 →
    congruent_mod_1000 (1978^n) (1978^m) →
    m + n ≥ min_sum ∧
    (m + n = min_sum → m = m_min ∧ n = n_min) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_proof_l724_72405


namespace NUMINAMATH_CALUDE_books_read_total_l724_72424

def total_books (megan kelcie john greg alice : ℝ) : ℝ :=
  megan + kelcie + john + greg + alice

theorem books_read_total :
  ∀ (megan kelcie john greg alice : ℝ),
    megan = 45 →
    kelcie = megan / 3 →
    john = kelcie + 7 →
    greg = 2 * john + 11 →
    alice = 2.5 * greg - 10 →
    total_books megan kelcie john greg alice = 264.5 :=
by
  sorry

end NUMINAMATH_CALUDE_books_read_total_l724_72424


namespace NUMINAMATH_CALUDE_packing_peanuts_calculation_l724_72476

/-- The amount of packing peanuts (in grams) needed for each large order -/
def large_order_peanuts : ℕ := sorry

/-- The total amount of packing peanuts (in grams) used -/
def total_peanuts : ℕ := 800

/-- The number of large orders -/
def num_large_orders : ℕ := 3

/-- The number of small orders -/
def num_small_orders : ℕ := 4

/-- The amount of packing peanuts (in grams) needed for each small order -/
def small_order_peanuts : ℕ := 50

theorem packing_peanuts_calculation :
  large_order_peanuts * num_large_orders + small_order_peanuts * num_small_orders = total_peanuts ∧
  large_order_peanuts = 200 := by sorry

end NUMINAMATH_CALUDE_packing_peanuts_calculation_l724_72476


namespace NUMINAMATH_CALUDE_nora_muffin_sales_l724_72427

/-- The number of cases of muffins Nora needs to sell to raise $120 -/
def cases_needed (packs_per_case : ℕ) (muffins_per_pack : ℕ) (price_per_muffin : ℕ) (target_amount : ℕ) : ℕ :=
  target_amount / (packs_per_case * muffins_per_pack * price_per_muffin)

/-- Proof that Nora needs to sell 5 cases of muffins to raise $120 -/
theorem nora_muffin_sales :
  cases_needed 3 4 2 120 = 5 := by
  sorry

end NUMINAMATH_CALUDE_nora_muffin_sales_l724_72427


namespace NUMINAMATH_CALUDE_difference_of_squares_l724_72415

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l724_72415


namespace NUMINAMATH_CALUDE_gardener_hours_per_day_l724_72482

/-- Calculates the number of hours a gardener works each day given the project details --/
theorem gardener_hours_per_day
  (total_cost : ℕ)
  (num_rose_bushes : ℕ)
  (cost_per_rose_bush : ℕ)
  (gardener_hourly_rate : ℕ)
  (num_work_days : ℕ)
  (soil_volume : ℕ)
  (soil_cost_per_unit : ℕ)
  (h_total_cost : total_cost = 4100)
  (h_num_rose_bushes : num_rose_bushes = 20)
  (h_cost_per_rose_bush : cost_per_rose_bush = 150)
  (h_gardener_hourly_rate : gardener_hourly_rate = 30)
  (h_num_work_days : num_work_days = 4)
  (h_soil_volume : soil_volume = 100)
  (h_soil_cost_per_unit : soil_cost_per_unit = 5) :
  (total_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * soil_cost_per_unit)) / gardener_hourly_rate / num_work_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_gardener_hours_per_day_l724_72482


namespace NUMINAMATH_CALUDE_probability_of_same_group_l724_72455

def card_count : ℕ := 20
def people_count : ℕ := 4
def first_drawn : ℕ := 5
def second_drawn : ℕ := 14

def same_group_probability : ℚ := 7 / 51

theorem probability_of_same_group :
  let remaining_cards := card_count - people_count + 2
  let favorable_outcomes := (card_count - second_drawn) * (card_count - second_drawn - 1) +
                            (first_drawn - 1) * (first_drawn - 2)
  let total_outcomes := remaining_cards * (remaining_cards - 1)
  (favorable_outcomes : ℚ) / total_outcomes = same_group_probability :=
sorry

end NUMINAMATH_CALUDE_probability_of_same_group_l724_72455


namespace NUMINAMATH_CALUDE_largest_integer_solution_l724_72497

theorem largest_integer_solution : 
  ∃ (x : ℕ), (1/4 : ℚ) + (x/5 : ℚ) < 2 ∧ 
  ∀ (y : ℕ), y > x → (1/4 : ℚ) + (y/5 : ℚ) ≥ 2 :=
by
  use 23
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_l724_72497


namespace NUMINAMATH_CALUDE_volunteer_arrangement_l724_72486

theorem volunteer_arrangement (n : ℕ) (k : ℕ) : n = 7 ∧ k = 3 → 
  (Nat.choose n k) * (Nat.choose (n - k) k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_l724_72486


namespace NUMINAMATH_CALUDE_ryan_load_is_correct_l724_72434

/-- The number of packages Sarah's trucks can carry in one load -/
def sarah_load : ℕ := 18

/-- The total number of packages shipped by both services -/
def total_packages : ℕ := 198

/-- Predicate to check if a number is a valid load size for Ryan's trucks -/
def is_valid_ryan_load (n : ℕ) : Prop :=
  n > sarah_load ∧ total_packages % n = 0

/-- The number of packages Ryan's trucks can carry in one load -/
def ryan_load : ℕ := 22

theorem ryan_load_is_correct : 
  is_valid_ryan_load ryan_load ∧ 
  ∀ (n : ℕ), is_valid_ryan_load n → n ≥ ryan_load :=
sorry

end NUMINAMATH_CALUDE_ryan_load_is_correct_l724_72434


namespace NUMINAMATH_CALUDE_bakery_children_count_l724_72420

theorem bakery_children_count (initial_count : ℕ) (girls_entered : ℕ) (boys_left : ℕ) 
  (h1 : initial_count = 85) (h2 : girls_entered = 24) (h3 : boys_left = 31) :
  initial_count + girls_entered - boys_left = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_bakery_children_count_l724_72420


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l724_72491

/-- A quadratic function of the form f(x) = x^2 + ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the unique quadratic function satisfying the given condition -/
theorem unique_quadratic_function (a b : ℝ) :
  (∀ x, (f a b (f a b x - x)) / (f a b x) = x^2 + 2023*x + 1777) →
  a = 2025 ∧ b = 249 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l724_72491


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l724_72471

theorem consecutive_integers_product (a : ℕ) : 
  (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) = 15120) → (a + 4 = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l724_72471


namespace NUMINAMATH_CALUDE_polynomial_expansion_l724_72453

theorem polynomial_expansion :
  ∀ z : ℂ, (3 * z^3 + 2 * z^2 - 4 * z + 1) * (2 * z^4 - 3 * z^2 + z - 5) =
  6 * z^7 + 4 * z^6 - 4 * z^5 - 9 * z^3 + 7 * z^2 + z - 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l724_72453


namespace NUMINAMATH_CALUDE_count_valid_m_l724_72430

theorem count_valid_m : ∃! (S : Finset ℤ), 
  (∀ m ∈ S, (∀ x : ℝ, (3 - 3*x < x - 5 ∧ x - m > -1) ↔ x > 2) ∧ 
             (∃ x : ℕ+, (2*x - m) / 3 = 1)) ∧
  S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_m_l724_72430


namespace NUMINAMATH_CALUDE_andrey_gifts_l724_72454

theorem andrey_gifts :
  ∃ (n : ℕ) (a : ℕ),
    n > 2 ∧
    n * (n - 2) = a * (n - 1) + 16 ∧
    n = 18 :=
by sorry

end NUMINAMATH_CALUDE_andrey_gifts_l724_72454


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l724_72409

-- Problem 1
theorem problem_1 : (-1)^3 + Real.sqrt 4 - (2 - Real.sqrt 2)^0 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 3) * (a - 3) - a * (a - 2) = 2 * a - 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l724_72409


namespace NUMINAMATH_CALUDE_tomatoes_left_l724_72485

/-- Given 21 initial tomatoes and birds eating one-third of them, prove that 14 tomatoes are left -/
theorem tomatoes_left (initial : ℕ) (eaten_fraction : ℚ) (h1 : initial = 21) (h2 : eaten_fraction = 1/3) :
  initial - (initial * eaten_fraction).floor = 14 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_left_l724_72485


namespace NUMINAMATH_CALUDE_range_of_a_l724_72407

/-- Definition of the circle D -/
def circle_D (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

/-- Definition of point B -/
def point_B : ℝ × ℝ := (-1, 0)

/-- Definition of point C -/
def point_C (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Theorem stating the range of a -/
theorem range_of_a (A B C : ℝ × ℝ) (a : ℝ) :
  (∃ x y, A = (x, y) ∧ circle_D x y) →  -- A lies on circle D
  B = point_B →                         -- B is at (-1, 0)
  C = point_C a →                       -- C is at (a, 0)
  (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0 →  -- Right angle at A
  14/5 ≤ a ∧ a ≤ 16/3 :=                -- Range of a
by sorry

end NUMINAMATH_CALUDE_range_of_a_l724_72407


namespace NUMINAMATH_CALUDE_negation_p_necessary_not_sufficient_l724_72425

theorem negation_p_necessary_not_sufficient (p q : Prop) :
  (¬(¬p → ¬(p ∨ q))) ∧ (∃ (p q : Prop), ¬p ∧ (p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_negation_p_necessary_not_sufficient_l724_72425


namespace NUMINAMATH_CALUDE_equation_solution_l724_72450

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 3) * x = 14 ∧ x = 147 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l724_72450


namespace NUMINAMATH_CALUDE_paint_used_approximation_l724_72487

/-- The amount of paint Joe starts with in gallons -/
def initial_paint : ℝ := 720

/-- The fraction of paint used in the first week -/
def first_week_fraction : ℚ := 2/7

/-- The fraction of remaining paint used in the second week -/
def second_week_fraction : ℚ := 3/8

/-- The fraction of remaining paint used in the third week -/
def third_week_fraction : ℚ := 5/11

/-- The fraction of remaining paint used in the fourth week -/
def fourth_week_fraction : ℚ := 4/13

/-- The total amount of paint used after four weeks -/
def total_paint_used : ℝ :=
  let first_week := initial_paint * (first_week_fraction : ℝ)
  let second_week := (initial_paint - first_week) * (second_week_fraction : ℝ)
  let third_week := (initial_paint - first_week - second_week) * (third_week_fraction : ℝ)
  let fourth_week := (initial_paint - first_week - second_week - third_week) * (fourth_week_fraction : ℝ)
  first_week + second_week + third_week + fourth_week

/-- Theorem stating that the total paint used is approximately 598.620 gallons -/
theorem paint_used_approximation : 
  598.619 < total_paint_used ∧ total_paint_used < 598.621 :=
sorry

end NUMINAMATH_CALUDE_paint_used_approximation_l724_72487


namespace NUMINAMATH_CALUDE_reservoir_capacity_proof_l724_72417

theorem reservoir_capacity_proof (current_amount : ℝ) (normal_level : ℝ) (total_capacity : ℝ) 
  (h1 : current_amount = 14)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.7 * total_capacity) :
  total_capacity - normal_level = 13 := by
sorry

end NUMINAMATH_CALUDE_reservoir_capacity_proof_l724_72417


namespace NUMINAMATH_CALUDE_bowl_capacity_ratio_l724_72446

theorem bowl_capacity_ratio :
  ∀ (capacity_1 capacity_2 : ℕ),
    capacity_1 < capacity_2 →
    capacity_2 = 600 →
    capacity_1 + capacity_2 = 1050 →
    (capacity_1 : ℚ) / capacity_2 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_bowl_capacity_ratio_l724_72446


namespace NUMINAMATH_CALUDE_reservoir_capacity_difference_l724_72468

/-- Represents the properties of a reservoir --/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ

/-- Theorem about the difference between total capacity and normal level after evaporation --/
theorem reservoir_capacity_difference (r : Reservoir) 
  (h1 : r.current_level = 14)
  (h2 : r.current_level = 2 * r.normal_level)
  (h3 : r.current_level = 0.7 * r.total_capacity)
  (h4 : r.evaporation_rate = 0.1) :
  r.total_capacity - (r.normal_level * (1 - r.evaporation_rate)) = 13.7 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_capacity_difference_l724_72468


namespace NUMINAMATH_CALUDE_G_of_two_eq_six_l724_72470

noncomputable def G (x : ℝ) : ℝ :=
  1.2 * Real.sqrt (abs (x + 1.5)) + (7 / Real.pi) * Real.arctan (1.1 * Real.sqrt (abs (x + 1.5)))

theorem G_of_two_eq_six : G 2 = 6 := by sorry

end NUMINAMATH_CALUDE_G_of_two_eq_six_l724_72470


namespace NUMINAMATH_CALUDE_intersection_sum_l724_72448

theorem intersection_sum (a b m : ℝ) : 
  ((-m + a = 8) ∧ (m + b = 8)) → a + b = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l724_72448


namespace NUMINAMATH_CALUDE_expression_simplification_l724_72461

theorem expression_simplification (x : ℝ) (h : x^2 - 2*x - 2 = 0) :
  ((x - 1) / x - (x - 2) / (x + 1)) / ((2*x^2 - x) / (x^2 + 2*x + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l724_72461


namespace NUMINAMATH_CALUDE_spinner_probability_l724_72466

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l724_72466


namespace NUMINAMATH_CALUDE_diameter_length_l724_72475

/-- Represents a circle with diameter AB and perpendicular chord CD -/
structure Circle where
  AB : ℕ
  CD : ℕ
  is_two_digit : 10 ≤ AB ∧ AB < 100
  is_reversed : CD = (AB % 10) * 10 + (AB / 10)

/-- The distance OH is rational -/
def rational_OH (c : Circle) : Prop :=
  ∃ (q : ℚ), q > 0 ∧ q^2 * 4 = 99 * (c.AB / 10 - c.AB % 10) * (c.AB / 10 + c.AB % 10)

theorem diameter_length (c : Circle) (h : rational_OH c) : c.AB = 65 :=
sorry

end NUMINAMATH_CALUDE_diameter_length_l724_72475


namespace NUMINAMATH_CALUDE_complex_equation_sum_l724_72459

theorem complex_equation_sum (x y : ℝ) : 
  Complex.mk (2 * x - 1) 1 = Complex.mk y (y - 3) → x + y = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l724_72459


namespace NUMINAMATH_CALUDE_no_perfect_squares_l724_72408

theorem no_perfect_squares (n : ℕ+) : 
  ¬(∃ (a b c : ℕ), (2 * n^2 + 1 = a^2) ∧ (3 * n^2 + 1 = b^2) ∧ (6 * n^2 + 1 = c^2)) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_squares_l724_72408


namespace NUMINAMATH_CALUDE_hexagram_shell_placement_l724_72462

def hexagram_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem hexagram_shell_placement :
  hexagram_arrangements 12 = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_hexagram_shell_placement_l724_72462


namespace NUMINAMATH_CALUDE_chord_with_midpoint_A_no_chord_with_midpoint_B_l724_72451

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define a chord of the hyperbola
def is_chord (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  hyperbola x₁ y₁ ∧ hyperbola x₂ y₂

-- Define the midpoint of a chord
def is_midpoint (x y x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2

-- Theorem 1: Chord with midpoint A(2,1)
theorem chord_with_midpoint_A :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 2 1 x₁ y₁ x₂ y₂ ∧
    ∀ (x y : ℝ), y = 6*x - 11 ↔ ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

-- Theorem 2: No chord with midpoint B(1,1)
theorem no_chord_with_midpoint_B :
  ¬∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_chord x₁ y₁ x₂ y₂ ∧
    is_midpoint 1 1 x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_chord_with_midpoint_A_no_chord_with_midpoint_B_l724_72451


namespace NUMINAMATH_CALUDE_chicken_nugget_ratio_l724_72483

theorem chicken_nugget_ratio : 
  ∀ (keely kendall : ℕ),
  keely + kendall + 20 = 100 →
  (keely + kendall) / 20 = 4 := by
sorry

end NUMINAMATH_CALUDE_chicken_nugget_ratio_l724_72483


namespace NUMINAMATH_CALUDE_charity_fundraising_contribution_l724_72449

theorem charity_fundraising_contribution 
  (total_goal : ℝ) 
  (collected : ℝ) 
  (num_people : ℕ) 
  (h1 : total_goal = 2400)
  (h2 : collected = 300)
  (h3 : num_people = 8) :
  (total_goal - collected) / num_people = 262.5 := by
sorry

end NUMINAMATH_CALUDE_charity_fundraising_contribution_l724_72449


namespace NUMINAMATH_CALUDE_no_three_five_powers_l724_72401

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_three_five_powers (n : ℕ) :
  ∀ α β : ℕ, v n ≠ 3^α * 5^β :=
by sorry

end NUMINAMATH_CALUDE_no_three_five_powers_l724_72401


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l724_72484

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ), 
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -2) →
    b = 1 ∧ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l724_72484


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l724_72421

theorem isosceles_right_triangle_area (leg : ℝ) (h_leg : leg = 3) :
  let triangle_area := (1 / 2) * leg * leg
  triangle_area = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l724_72421


namespace NUMINAMATH_CALUDE_special_function_value_l724_72447

/-- A function satisfying f(x + y) = f(x) + f(y) + 2xy for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y

theorem special_function_value :
  ∀ f : ℝ → ℝ, special_function f → f 1 = 2 → f (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l724_72447


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l724_72443

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) / (1 - 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l724_72443


namespace NUMINAMATH_CALUDE_brads_red_balloons_l724_72432

/-- Given that Brad has a total of 17 balloons and 9 of them are green,
    prove that he has 8 red balloons. -/
theorem brads_red_balloons (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end NUMINAMATH_CALUDE_brads_red_balloons_l724_72432


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l724_72473

theorem negation_of_existence (f : ℝ → Prop) :
  (¬ ∃ x : ℝ, f x) ↔ ∀ x : ℝ, ¬ f x := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 3 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l724_72473


namespace NUMINAMATH_CALUDE_ellipse_property_l724_72452

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the foci of the ellipse
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define the angle between PF1 and PF2
def angle_F1PF2 (P : ℝ × ℝ) : ℝ := 120

-- Theorem statement
theorem ellipse_property (P : ℝ × ℝ) 
  (h1 : is_on_ellipse P.1 P.2) 
  (h2 : angle_F1PF2 P = 120) : 
  Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) * 
  Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_property_l724_72452


namespace NUMINAMATH_CALUDE_inequality_solution_set_l724_72464

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (2 * x - 1) / (x - 1) ≥ 1 ↔ x ≤ 0 ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l724_72464


namespace NUMINAMATH_CALUDE_san_antonio_bound_passes_ten_buses_l724_72463

/-- Represents the schedule and trip details of buses between Austin and San Antonio -/
structure BusSchedule where
  austin_to_sa_interval : ℕ -- Interval in minutes for Austin to San Antonio buses
  sa_to_austin_interval : ℕ -- Interval in minutes for San Antonio to Austin buses
  sa_to_austin_offset : ℕ   -- Offset in minutes for San Antonio to Austin buses
  trip_duration : ℕ         -- Trip duration in minutes

/-- Calculates the number of buses passed on the highway -/
def buses_passed (schedule : BusSchedule) : ℕ :=
  sorry -- Proof to be implemented

/-- Main theorem: A San Antonio-bound bus passes 10 Austin-bound buses on the highway -/
theorem san_antonio_bound_passes_ten_buses :
  let schedule : BusSchedule := {
    austin_to_sa_interval := 30,
    sa_to_austin_interval := 45,
    sa_to_austin_offset := 15,
    trip_duration := 240  -- 4 hours in minutes
  }
  buses_passed schedule = 10 := by sorry

end NUMINAMATH_CALUDE_san_antonio_bound_passes_ten_buses_l724_72463


namespace NUMINAMATH_CALUDE_point_in_plane_region_l724_72456

def in_plane_region (x y : ℝ) : Prop := 2*x + y - 6 < 0

theorem point_in_plane_region :
  in_plane_region 0 1 ∧
  ¬(in_plane_region 5 0) ∧
  ¬(in_plane_region 0 7) ∧
  ¬(in_plane_region 2 3) :=
by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l724_72456


namespace NUMINAMATH_CALUDE_max_sum_geometric_sequence_l724_72412

/-- Given integers a, b, and c forming a strictly increasing geometric sequence with abc = 216,
    the maximum value of a + b + c is 43. -/
theorem max_sum_geometric_sequence (a b c : ℤ) : 
  a < b ∧ b < c ∧                 -- strictly increasing
  (∃ r : ℤ, r > 1 ∧ b = a * r ∧ c = b * r) ∧  -- geometric sequence
  a * b * c = 216 →               -- product condition
  (∀ x y z : ℤ, 
    x < y ∧ y < z ∧
    (∃ r : ℤ, r > 1 ∧ y = x * r ∧ z = y * r) ∧
    x * y * z = 216 →
    x + y + z ≤ a + b + c) ∧
  a + b + c = 43 := by
sorry

end NUMINAMATH_CALUDE_max_sum_geometric_sequence_l724_72412


namespace NUMINAMATH_CALUDE_marbles_in_larger_container_l724_72442

/-- Given that a container with a volume of 24 cm³ can hold 75 marbles,
    prove that a container with a volume of 72 cm³ can hold 225 marbles,
    assuming the ratio of marbles to volume is constant. -/
theorem marbles_in_larger_container (v₁ v₂ : ℝ) (m₁ m₂ : ℕ) 
    (h₁ : v₁ = 24) (h₂ : m₁ = 75) (h₃ : v₂ = 72) :
    (m₁ : ℝ) / v₁ = m₂ / v₂ → m₂ = 225 := by
  sorry

end NUMINAMATH_CALUDE_marbles_in_larger_container_l724_72442


namespace NUMINAMATH_CALUDE_remaining_work_days_l724_72436

/-- Given two workers x and y, where x can finish a job in 36 days and y in 24 days,
    prove that x needs 18 days to finish the remaining work after y worked for 12 days. -/
theorem remaining_work_days (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 36) (hy : y_days = 24) (hw : y_worked_days = 12) : 
  (x_days : ℚ) / 2 = 18 := by
  sorry

#check remaining_work_days

end NUMINAMATH_CALUDE_remaining_work_days_l724_72436


namespace NUMINAMATH_CALUDE_inequalities_equivalence_l724_72416

theorem inequalities_equivalence (x : ℝ) :
  (2 * (x + 1) - 1 < 3 * x + 2 ↔ x > -1) ∧
  ((x + 3) / 2 - 1 ≥ (2 * x - 3) / 3 ↔ x ≤ 9) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_equivalence_l724_72416


namespace NUMINAMATH_CALUDE_sphere_radius_in_cube_l724_72440

/-- The radius of spheres packed in a cube -/
theorem sphere_radius_in_cube (n : ℕ) (side_length : ℝ) (radius : ℝ) : 
  n = 8 →  -- There are 8 spheres
  side_length = 2 →  -- The cube has side length 2
  radius > 0 →  -- The radius is positive
  (2 * radius = side_length / 2 + radius) →  -- Condition for spheres to be tangent
  radius = 1 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_cube_l724_72440


namespace NUMINAMATH_CALUDE_system_solution_l724_72490

theorem system_solution : ∃ (x y z : ℝ), 
  (x + y = 1) ∧ (x + z = 0) ∧ (y + z = -1) ∧ 
  (x = 1) ∧ (y = 0) ∧ (z = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l724_72490


namespace NUMINAMATH_CALUDE_A_equiv_B_l724_72458

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define set A
def A : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, (sumOfDigits n + 1 = 5 * k ∨ sumOfDigits n - 1 = 5 * k))}

-- Define set B
def B : Set ℕ := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧
  (∃ k : ℤ, sumOfDigits n = 5 * k ∨ sumOfDigits n - 2 = 5 * k)}

-- Theorem statement
theorem A_equiv_B : Equiv A B := by sorry

end NUMINAMATH_CALUDE_A_equiv_B_l724_72458


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l724_72498

theorem pure_imaginary_product (b : ℝ) : 
  let z₁ : ℂ := 1 + Complex.I
  let z₂ : ℂ := 1 - b * Complex.I
  (z₁ * z₂).re = 0 ∧ (z₁ * z₂).im ≠ 0 → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l724_72498


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l724_72428

/-- The number of base pizza flavors -/
def base_flavors : ℕ := 4

/-- The number of topping combinations (including no additional toppings) -/
def topping_combinations : ℕ := 4

/-- Calculates the total number of pizza varieties -/
def total_varieties : ℕ := base_flavors * topping_combinations

theorem pizza_varieties_count :
  total_varieties = 16 := by
  sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l724_72428


namespace NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l724_72444

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_t6_plus_t4_l724_72444


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l724_72435

/-- Given a positive integer n, this function checks if a postage value can be formed
    using stamps of denominations 7, n, and n+1 cents. -/
def can_form_postage (n : ℕ+) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 7 * a + n * b + (n + 1) * c

/-- This theorem states that 18 is the unique positive integer n such that,
    given stamps of denominations 7, n, and n+1 cents, 106 cents is the
    greatest postage that cannot be formed. -/
theorem unique_stamp_denomination :
  ∃! (n : ℕ+),
    (¬ can_form_postage n 106) ∧
    (∀ m : ℕ, m > 106 → can_form_postage n m) ∧
    (∀ k : ℕ, k < 106 → ¬ can_form_postage n k → can_form_postage n (k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l724_72435


namespace NUMINAMATH_CALUDE_marty_voters_l724_72419

theorem marty_voters (total : ℕ) (biff_percent : ℚ) (undecided_percent : ℚ) 
  (h1 : total = 200)
  (h2 : biff_percent = 45 / 100)
  (h3 : undecided_percent = 8 / 100) :
  ⌊(1 - biff_percent - undecided_percent) * total⌋ = 94 := by
  sorry

end NUMINAMATH_CALUDE_marty_voters_l724_72419


namespace NUMINAMATH_CALUDE_total_weight_loss_is_correct_l724_72496

/-- The total weight loss of Seth, Jerome, Veronica, and Maya -/
def totalWeightLoss (sethLoss : ℝ) : ℝ :=
  let jeromeLoss := 3 * sethLoss
  let veronicaLoss := sethLoss + 1.56
  let sethVeronicaCombined := sethLoss + veronicaLoss
  let mayaLoss := sethVeronicaCombined * 0.75
  sethLoss + jeromeLoss + veronicaLoss + mayaLoss

/-- Theorem stating that the total weight loss is 116.675 pounds -/
theorem total_weight_loss_is_correct :
  totalWeightLoss 17.53 = 116.675 := by
  sorry

#eval totalWeightLoss 17.53

end NUMINAMATH_CALUDE_total_weight_loss_is_correct_l724_72496


namespace NUMINAMATH_CALUDE_small_triangles_to_cover_large_l724_72426

/-- The number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem small_triangles_to_cover_large (large_side small_side : ℝ) : 
  large_side = 12 → small_side = 2 → 
  (large_side^2 / small_side^2 : ℝ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_small_triangles_to_cover_large_l724_72426


namespace NUMINAMATH_CALUDE_grid_sum_property_l724_72431

def Grid := Matrix (Fin 2) (Fin 3) ℕ

def is_valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 9 ∧
  (∀ i₁ i₂ j, i₁ ≠ i₂ → g i₁ j ≠ g i₂ j) ∧
  (∀ i j₁ j₂, j₁ ≠ j₂ → g i j₁ ≠ g i j₂)

theorem grid_sum_property (g : Grid) (h : is_valid_grid g) :
  (g 0 0 + g 0 1 + g 0 2 = 23) →
  (g 0 0 + g 1 0 = 14) →
  (g 0 1 + g 1 1 = 16) →
  (g 0 2 + g 1 2 = 17) →
  g 1 0 + 2 * g 1 1 + 3 * g 1 2 = 49 := by
sorry

end NUMINAMATH_CALUDE_grid_sum_property_l724_72431


namespace NUMINAMATH_CALUDE_circle_distance_theorem_l724_72400

theorem circle_distance_theorem (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → ∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 = 1 ∧ 
    x2^2 + y2^2 = 1 ∧ 
    (x1 - a)^2 + (y1 - 1)^2 = 4 ∧ 
    (x2 - a)^2 + (y2 - 1)^2 = 4 ∧ 
    (x1, y1) ≠ (x2, y2)) → 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_distance_theorem_l724_72400


namespace NUMINAMATH_CALUDE_square_root_property_l724_72410

theorem square_root_property (p k : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) 
  (hk : k > 0) (h_sqrt : ∃ n : ℕ, n > 0 ∧ n^2 = k^2 - p*k) : 
  k = (p + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_property_l724_72410


namespace NUMINAMATH_CALUDE_inverse_42_mod_53_l724_72423

theorem inverse_42_mod_53 (h : (11⁻¹ : ZMod 53) = 31) : (42⁻¹ : ZMod 53) = 22 := by
  sorry

end NUMINAMATH_CALUDE_inverse_42_mod_53_l724_72423


namespace NUMINAMATH_CALUDE_largest_package_size_and_cost_l724_72411

def lucas_notebooks : ℕ := 36
def maria_notebooks : ℕ := 60
def package_cost : ℕ := 3

theorem largest_package_size_and_cost :
  let max_package_size := Nat.gcd lucas_notebooks maria_notebooks
  (max_package_size = 12) ∧ (package_cost = 3) := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_and_cost_l724_72411


namespace NUMINAMATH_CALUDE_kameron_has_100_kangaroos_l724_72402

/-- The number of kangaroos Bert currently has -/
def bert_initial : ℕ := 20

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days : ℕ := 40

/-- The number of kangaroos Bert buys per day -/
def bert_rate : ℕ := 2

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := bert_initial + days * bert_rate

theorem kameron_has_100_kangaroos : kameron_kangaroos = 100 := by
  sorry

end NUMINAMATH_CALUDE_kameron_has_100_kangaroos_l724_72402


namespace NUMINAMATH_CALUDE_rectangle_side_relationship_l724_72467

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.x + r.y)

/-- Theorem: For a rectangle with perimeter 50 cm, y = 25 - x -/
theorem rectangle_side_relationship (r : Rectangle) 
  (h : perimeter r = 50) : r.y = 25 - r.x := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_relationship_l724_72467


namespace NUMINAMATH_CALUDE_basketball_not_football_l724_72413

theorem basketball_not_football (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) 
  (h1 : total = 30)
  (h2 : basketball = 15)
  (h3 : football = 8)
  (h4 : neither = 8) :
  ∃ (x : ℕ), x = basketball - (basketball + football - total + neither) ∧ x = 14 :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_football_l724_72413


namespace NUMINAMATH_CALUDE_least_positive_period_is_36_l724_72429

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_positive_period_is_36 (f : ℝ → ℝ) (h : has_property f) :
  (∃ p : ℝ, p > 0 ∧ is_period f p) →
  (∀ q : ℝ, q > 0 → is_period f q → q ≥ 36) ∧ is_period f 36 :=
sorry

end NUMINAMATH_CALUDE_least_positive_period_is_36_l724_72429
