import Mathlib

namespace NUMINAMATH_CALUDE_simplify_trig_expression_l575_57517

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l575_57517


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l575_57537

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {2,3,5,6}
def B : Set Nat := {1,3,4,6,7}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2,5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l575_57537


namespace NUMINAMATH_CALUDE_value_of_k_l575_57512

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-collinear vectors e₁ and e₂
variable (e₁ e₂ : V)
variable (h_non_collinear : ∀ (a b : ℝ), a • e₁ + b • e₂ = 0 → a = 0 ∧ b = 0)

-- Define points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- Define the given vector relationships
variable (h_AB : B - A = 2 • e₁ + k • e₂)
variable (h_CB : B - C = e₁ + 3 • e₂)
variable (h_CD : D - C = 2 • e₁ - e₂)

-- Define collinearity of points A, B, and D
variable (h_collinear : ∃ (t : ℝ), B - A = t • (D - B))

-- Theorem statement
theorem value_of_k : k = -8 := by sorry

end NUMINAMATH_CALUDE_value_of_k_l575_57512


namespace NUMINAMATH_CALUDE_min_value_of_expression_l575_57568

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) / (2 * b - 3) = -a / (2 * b)) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 : ℝ) / (2 * y - 3) = -x / (2 * y) → 2 * a + 3 * b ≤ 2 * x + 3 * y) ∧
  (2 * a + 3 * b = 25 / 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l575_57568


namespace NUMINAMATH_CALUDE_oranges_per_day_l575_57574

/-- Proves that the number of sacks harvested per day is 4, given 56 sacks over 14 days -/
theorem oranges_per_day (total_sacks : ℕ) (total_days : ℕ) 
  (h1 : total_sacks = 56) (h2 : total_days = 14) : 
  total_sacks / total_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_day_l575_57574


namespace NUMINAMATH_CALUDE_garden_furniture_cost_l575_57536

/-- The combined cost of a garden table and a bench -/
def combined_cost (bench_cost : ℝ) (table_cost : ℝ) : ℝ :=
  bench_cost + table_cost

theorem garden_furniture_cost :
  ∀ (bench_cost : ℝ) (table_cost : ℝ),
  bench_cost = 250.0 →
  table_cost = 2 * bench_cost →
  combined_cost bench_cost table_cost = 750.0 := by
sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_l575_57536


namespace NUMINAMATH_CALUDE_undamaged_tins_count_l575_57518

theorem undamaged_tins_count (cases : ℕ) (tins_per_case : ℕ) (damage_percent : ℚ) : 
  cases = 15 → 
  tins_per_case = 24 → 
  damage_percent = 5 / 100 →
  cases * tins_per_case * (1 - damage_percent) = 342 := by
sorry

end NUMINAMATH_CALUDE_undamaged_tins_count_l575_57518


namespace NUMINAMATH_CALUDE_equation_solutions_l575_57560

theorem equation_solutions :
  (∀ x : ℝ, 16 * x^2 = 49 ↔ x = 7/4 ∨ x = -7/4) ∧
  (∀ x : ℝ, (x - 2)^2 = 64 ↔ x = 10 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l575_57560


namespace NUMINAMATH_CALUDE_son_work_time_l575_57561

/-- Given a man can do a piece of work in 5 days, and together with his son they can do it in 3 days,
    prove that the son can do the work alone in 7.5 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) 
    (h1 : man_time = 5)
    (h2 : combined_time = 3) :
    son_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l575_57561


namespace NUMINAMATH_CALUDE_max_balloon_surface_area_l575_57597

/-- The maximum surface area of a spherical balloon inscribed in a cube --/
theorem max_balloon_surface_area (a : ℝ) (h : a > 0) :
  ∃ (A : ℝ), A = 2 * Real.pi * a^2 ∧ 
  ∀ (r : ℝ), r > 0 → r ≤ a * Real.sqrt 2 / 2 → 
  4 * Real.pi * r^2 ≤ A := by
  sorry

end NUMINAMATH_CALUDE_max_balloon_surface_area_l575_57597


namespace NUMINAMATH_CALUDE_nine_digit_multiply_six_property_l575_57520

/-- A function that checks if a natural number contains each digit from 1 to 9 exactly once --/
def containsAllDigitsOnce (n : ℕ) : Prop :=
  ∀ d : Fin 9, ∃! p : ℕ, n / 10^p % 10 = d.val + 1

/-- A function that represents the multiplication of a 9-digit number by 6 --/
def multiplyBySix (n : ℕ) : ℕ := n * 6

/-- Theorem stating the existence of 9-digit numbers with the required property --/
theorem nine_digit_multiply_six_property :
  ∃ n : ℕ, 
    100000000 ≤ n ∧ n < 1000000000 ∧
    containsAllDigitsOnce n ∧
    containsAllDigitsOnce (multiplyBySix n) :=
sorry

end NUMINAMATH_CALUDE_nine_digit_multiply_six_property_l575_57520


namespace NUMINAMATH_CALUDE_circle_area_ratio_l575_57544

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * Real.pi * r₁) = (45 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l575_57544


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l575_57548

def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l575_57548


namespace NUMINAMATH_CALUDE_horatio_sonnets_l575_57511

/-- Proves that Horatio wrote 12 sonnets in total -/
theorem horatio_sonnets (lines_per_sonnet : ℕ) (read_sonnets : ℕ) (unread_lines : ℕ) : 
  lines_per_sonnet = 14 → read_sonnets = 7 → unread_lines = 70 →
  read_sonnets + (unread_lines / lines_per_sonnet) = 12 := by
  sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l575_57511


namespace NUMINAMATH_CALUDE_min_sum_squares_l575_57576

theorem min_sum_squares (x y z : ℝ) (h : 2 * x + 3 * y + 3 * z = 1) :
  (∀ a b c : ℝ, 2 * a + 3 * b + 3 * c = 1 → x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) →
  x^2 + y^2 + z^2 = 1 / 22 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l575_57576


namespace NUMINAMATH_CALUDE_root_difference_equals_2000_l575_57595

theorem root_difference_equals_2000 : ∃ (a b : ℝ), 
  ((1998 * a)^2 - 1997 * 1999 * a - 1 = 0 ∧ 
   ∀ x, (1998 * x)^2 - 1997 * 1999 * x - 1 = 0 → x ≤ a) ∧
  (b^2 + 1998 * b - 1999 = 0 ∧ 
   ∀ y, y^2 + 1998 * y - 1999 = 0 → b ≤ y) ∧
  a - b = 2000 := by
sorry

end NUMINAMATH_CALUDE_root_difference_equals_2000_l575_57595


namespace NUMINAMATH_CALUDE_sum_of_powers_inequality_l575_57533

theorem sum_of_powers_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 ≥ 6 ∧
  (a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 = 6 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_inequality_l575_57533


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l575_57503

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c * Real.cos A = 4 →
  a * c * Real.sin B = 8 * Real.sin A →
  A = π / 3 ∧ 0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l575_57503


namespace NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l575_57540

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∃ (k : ℝ), k = 1 ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ∧
  (∀ k' : ℝ, k' > k → ∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x < k' * x) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_f_geq_kx_l575_57540


namespace NUMINAMATH_CALUDE_circle_plus_two_four_l575_57588

-- Define the operation ⊕
def circle_plus (a b : ℝ) : ℝ := 5 * a + 2 * b

-- Theorem statement
theorem circle_plus_two_four : circle_plus 2 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_two_four_l575_57588


namespace NUMINAMATH_CALUDE_min_product_value_l575_57500

def is_monic_nonneg_int_coeff (p : ℕ → ℕ) : Prop :=
  p 0 = 1 ∧ ∀ n, p n ≥ 0

def satisfies_inequality (p q : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, x ≥ 2 → (1 : ℚ) / (5 * x) ≥ 1 / (q x) - 1 / (p x) ∧ 1 / (q x) - 1 / (p x) ≥ 1 / (3 * x^2)

theorem min_product_value (p q : ℕ → ℕ) :
  is_monic_nonneg_int_coeff p →
  is_monic_nonneg_int_coeff q →
  satisfies_inequality p q →
  (∀ p' q' : ℕ → ℕ, is_monic_nonneg_int_coeff p' → is_monic_nonneg_int_coeff q' → 
    satisfies_inequality p' q' → p' 1 * q' 1 ≥ p 1 * q 1) →
  p 1 * q 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_product_value_l575_57500


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l575_57566

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ) * (((a - Complex.I) / (1 - Complex.I)).im) = ((a - Complex.I) / (1 - Complex.I)) → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l575_57566


namespace NUMINAMATH_CALUDE_group_frequency_number_l575_57509

-- Define the sample capacity
def sample_capacity : ℕ := 100

-- Define the frequency of the group
def group_frequency : ℚ := 3/10

-- Define the frequency number calculation
def frequency_number (capacity : ℕ) (frequency : ℚ) : ℚ := capacity * frequency

-- Theorem statement
theorem group_frequency_number :
  frequency_number sample_capacity group_frequency = 30 := by sorry

end NUMINAMATH_CALUDE_group_frequency_number_l575_57509


namespace NUMINAMATH_CALUDE_problem_solution_l575_57584

def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

def S : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 ∈ X ∧ t.2.1 ∈ X ∧ t.2.2 ∈ X ∧
    ((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∨
     (t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∨
     (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.1 < t.2.2 ∧ t.2.2 < t.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1))}

theorem problem_solution (x y z w : ℕ) 
  (h1 : (x, y, z) ∈ S) (h2 : (z, w, x) ∈ S) :
  (y, z, w) ∈ S ∧ (x, y, w) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l575_57584


namespace NUMINAMATH_CALUDE_expression_simplification_l575_57507

theorem expression_simplification 
  (a b c d x y : ℝ) 
  (h : c * x ≠ d * y) : 
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) - 
   d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / 
  (c * x - d * y) = 
  b^2 * x^2 + a^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l575_57507


namespace NUMINAMATH_CALUDE_pen_difference_l575_57501

/-- A collection of pens and pencils -/
structure PenCollection where
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ
  pencils : ℕ

/-- Properties of the pen collection -/
def valid_collection (c : PenCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.blue_pens + c.black_pens + c.red_pens = 48 ∧
  c.red_pens < c.pencils

theorem pen_difference (c : PenCollection) 
  (h : valid_collection c) : c.pencils - c.red_pens = 2 := by
  sorry

end NUMINAMATH_CALUDE_pen_difference_l575_57501


namespace NUMINAMATH_CALUDE_factorial_calculation_l575_57590

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_calculation :
  (5 * factorial 6 + 30 * factorial 5) / factorial 7 = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l575_57590


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l575_57527

theorem triangle_angle_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * B) (h3 : B = 15) : A = 120 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l575_57527


namespace NUMINAMATH_CALUDE_unique_value_of_2n_plus_m_l575_57587

theorem unique_value_of_2n_plus_m (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_of_2n_plus_m_l575_57587


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l575_57522

theorem sqrt_equation_solution (x : ℝ) (h : x > 1) :
  (Real.sqrt (5 * x) / Real.sqrt (3 * (x - 1)) = 2) → x = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l575_57522


namespace NUMINAMATH_CALUDE_alberts_brother_age_difference_l575_57529

/-- Proves that Albert's brother is 2 years younger than Albert given the problem conditions -/
theorem alberts_brother_age_difference : ℕ → Prop :=
  fun albert_age : ℕ =>
    ∀ (father_age mother_age brother_age : ℕ),
      father_age = albert_age + 48 →
      mother_age = brother_age + 46 →
      father_age = mother_age + 4 →
      brother_age < albert_age →
      albert_age - brother_age = 2

/-- Proof of the theorem -/
lemma prove_alberts_brother_age_difference :
  ∀ albert_age : ℕ, alberts_brother_age_difference albert_age :=
by
  sorry

#check prove_alberts_brother_age_difference

end NUMINAMATH_CALUDE_alberts_brother_age_difference_l575_57529


namespace NUMINAMATH_CALUDE_divisibility_condition_l575_57577

theorem divisibility_condition (a b : ℕ+) : 
  (∃ k : ℕ, (a^2 * b + a + b : ℕ) = k * (a * b^2 + b + 7)) ↔ 
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l575_57577


namespace NUMINAMATH_CALUDE_jack_morning_emails_l575_57556

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 10

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 7

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 17

/-- Theorem stating that Jack received 10 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = afternoon_emails + 3 := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l575_57556


namespace NUMINAMATH_CALUDE_prime_condition_theorem_l575_57589

def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧
  ∀ q : ℕ, Nat.Prime q → q < p →
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q →
      ∀ a : ℕ, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_theorem :
  {p : ℕ | satisfies_condition p} = {2, 3, 5, 7, 13} :=
sorry

end NUMINAMATH_CALUDE_prime_condition_theorem_l575_57589


namespace NUMINAMATH_CALUDE_sin_150_cos_30_l575_57510

theorem sin_150_cos_30 : Real.sin (150 * π / 180) * Real.cos (30 * π / 180) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_cos_30_l575_57510


namespace NUMINAMATH_CALUDE_find_A_l575_57571

theorem find_A : ∀ A : ℕ, (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l575_57571


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_l575_57523

theorem scientific_notation_exponent (n : ℤ) : 12368000 = 1.2368 * (10 : ℝ) ^ n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_l575_57523


namespace NUMINAMATH_CALUDE_removed_triangles_area_l575_57505

/-- Given a square with side length x, from which isosceles right triangles
    are removed from each corner to form a rectangle with diagonal 15,
    prove that the total area of the four removed triangles is 112.5. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  (x - r)^2 + (x - s)^2 = 15^2 →
  r + s = x →
  (4 : ℝ) * (1/2 * r * s) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l575_57505


namespace NUMINAMATH_CALUDE_accuracy_of_rounded_number_l575_57555

def is_accurate_to_hundreds_place (n : ℕ) : Prop :=
  n % 1000 ≠ 0 ∧ n % 100 = 0

theorem accuracy_of_rounded_number :
  ∀ (n : ℕ), 
    (31500 ≤ n ∧ n < 32500) →
    is_accurate_to_hundreds_place n :=
by
  sorry

end NUMINAMATH_CALUDE_accuracy_of_rounded_number_l575_57555


namespace NUMINAMATH_CALUDE_simplify_expressions_l575_57575

theorem simplify_expressions :
  (∃ x, x = Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1/5) ∧ x = (6 * Real.sqrt 5) / 5) ∧
  (∃ y, y = (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1/2) * Real.sqrt 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l575_57575


namespace NUMINAMATH_CALUDE_complex_equation_solution_l575_57528

theorem complex_equation_solution (Z : ℂ) :
  (1 + 2*Complex.I)^3 * Z = 1 + 2*Complex.I →
  Z = -3/25 + 24/125*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l575_57528


namespace NUMINAMATH_CALUDE_ratio_problem_l575_57598

theorem ratio_problem (x : ℝ) : x / 10 = 17.5 / 1 → x = 175 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l575_57598


namespace NUMINAMATH_CALUDE_triangle_count_after_12_iterations_l575_57524

/-- The number of triangles after n iterations of the division process -/
def num_triangles (n : ℕ) : ℕ := 3^n

/-- The side length of triangles after n iterations -/
def side_length (n : ℕ) : ℚ := 1 / 2^n

theorem triangle_count_after_12_iterations :
  num_triangles 12 = 531441 ∧ side_length 12 = 1 / 2^12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_after_12_iterations_l575_57524


namespace NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l575_57578

theorem reciprocals_not_arithmetic_sequence (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (b - a ≠ 0) →
  (c - b = b - a) →
  ¬(1/b - 1/a = 1/c - 1/b) := by
sorry

end NUMINAMATH_CALUDE_reciprocals_not_arithmetic_sequence_l575_57578


namespace NUMINAMATH_CALUDE_diagonals_from_vertex_is_six_l575_57591

/-- A polygon with internal angles of 140 degrees -/
structure Polygon140 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The internal angle of the polygon is 140 degrees -/
  internal_angle : sides * 140 = (sides - 2) * 180

/-- The number of diagonals from a single vertex in a Polygon140 -/
def diagonals_from_vertex (p : Polygon140) : ℕ :=
  p.sides - 3

/-- Theorem: The number of diagonals from a vertex in a Polygon140 is 6 -/
theorem diagonals_from_vertex_is_six (p : Polygon140) :
  diagonals_from_vertex p = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_from_vertex_is_six_l575_57591


namespace NUMINAMATH_CALUDE_sector_area_l575_57592

/-- The area of a circular sector with radius 6 cm and central angle 30° is 3π cm². -/
theorem sector_area : 
  let r : ℝ := 6
  let α : ℝ := 30 * π / 180  -- Convert degrees to radians
  (1/2) * r^2 * α = 3 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l575_57592


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l575_57573

/-- Given an arithmetic sequence {a_n}, if a_2^2 + 2a_2a_8 + a_6a_10 = 16, then a_4a_6 = 4 -/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16 →
  a 4 * a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l575_57573


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l575_57554

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 12) 
  (h3 : c^2 = a^2 + b^2) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l575_57554


namespace NUMINAMATH_CALUDE_mika_bought_26_stickers_l575_57550

/-- Represents the number of stickers Mika has at different stages -/
structure StickerCount where
  initial : Nat
  birthday : Nat
  given_away : Nat
  used : Nat
  remaining : Nat

/-- Calculates the number of stickers Mika bought from the store -/
def stickers_bought (s : StickerCount) : Nat :=
  s.remaining + s.given_away + s.used - s.initial - s.birthday

/-- Theorem stating that Mika bought 26 stickers from the store -/
theorem mika_bought_26_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.birthday = 20)
  (h3 : s.given_away = 6)
  (h4 : s.used = 58)
  (h5 : s.remaining = 2) :
  stickers_bought s = 26 := by
  sorry

#eval stickers_bought { initial := 20, birthday := 20, given_away := 6, used := 58, remaining := 2 }

end NUMINAMATH_CALUDE_mika_bought_26_stickers_l575_57550


namespace NUMINAMATH_CALUDE_ratio_of_bases_l575_57541

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  /-- Length of the larger base AB -/
  AB : ℝ
  /-- Length of the smaller base CD -/
  CD : ℝ
  /-- Area of triangle with base CD -/
  area_CD : ℝ
  /-- Area of triangle adjacent to CD (clockwise) -/
  area_adj_CD : ℝ
  /-- Area of triangle with base AB -/
  area_AB : ℝ
  /-- Area of triangle adjacent to AB (counter-clockwise) -/
  area_adj_AB : ℝ
  /-- AB is longer than CD -/
  h_AB_gt_CD : AB > CD
  /-- The trapezoid is isosceles -/
  h_isosceles : True  -- We don't need to specify this condition explicitly for the proof
  /-- The bases are parallel -/
  h_parallel : True   -- We don't need to specify this condition explicitly for the proof
  /-- Areas of triangles -/
  h_areas : area_CD = 5 ∧ area_adj_CD = 7 ∧ area_AB = 9 ∧ area_adj_AB = 3

/-- The ratio of bases in the isosceles trapezoid with given triangle areas -/
theorem ratio_of_bases (t : IsoscelesTrapezoidWithPoint) : t.AB / t.CD = 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_bases_l575_57541


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l575_57586

theorem simplify_and_evaluate : 
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  5 * (3 * a^2 * b - a * b^2) - (a * b^2 + 3 * a^2 * b) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l575_57586


namespace NUMINAMATH_CALUDE_expression_simplification_l575_57532

variable (a b : ℝ)

theorem expression_simplification :
  (2*a^2 - 3*a^3 + 5*a + 2*a^3 - a^2 = a^2 - a^3 + 5*a) ∧
  (2/3*(2*a - b) + 2*(b - 2*a) - 3*(2*a - b) - 4/3*(b - 2*a) = -6*a + 3*b) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l575_57532


namespace NUMINAMATH_CALUDE_ripe_oranges_calculation_l575_57531

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := 52

/-- The number of days of harvest -/
def harvest_days : ℕ := 26

/-- The total number of sacks of oranges after the harvest period -/
def total_oranges : ℕ := 2080

/-- The number of sacks of ripe oranges harvested per day -/
def ripe_oranges_per_day : ℕ := 28

theorem ripe_oranges_calculation :
  ripe_oranges_per_day * harvest_days + unripe_oranges_per_day * harvest_days = total_oranges :=
by sorry

end NUMINAMATH_CALUDE_ripe_oranges_calculation_l575_57531


namespace NUMINAMATH_CALUDE_max_leap_years_in_200_years_l575_57599

/-- 
In a calendrical system where leap years occur every three years without exception,
the maximum number of leap years in a 200-year period is 66.
-/
theorem max_leap_years_in_200_years : 
  ∀ (leap_year_count : ℕ → ℕ),
  (∀ n : ℕ, leap_year_count (3 * n) = n) →
  leap_year_count 200 = 66 := by
sorry

end NUMINAMATH_CALUDE_max_leap_years_in_200_years_l575_57599


namespace NUMINAMATH_CALUDE_angle_halving_l575_57567

-- Define what it means for an angle to be in the fourth quadrant
def in_fourth_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < θ ∧ θ < 2 * k * Real.pi

-- Define what it means for an angle to be in the first or third quadrant
def in_first_or_third_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < θ ∧ θ < k * Real.pi + Real.pi / 2

theorem angle_halving (θ : Real) :
  in_fourth_quadrant θ → in_first_or_third_quadrant (-θ/2) :=
by sorry

end NUMINAMATH_CALUDE_angle_halving_l575_57567


namespace NUMINAMATH_CALUDE_percentage_problem_l575_57526

theorem percentage_problem (p : ℝ) : (p / 100) * 40 = 140 → p = 350 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l575_57526


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l575_57535

theorem necessary_not_sufficient (a b : ℝ) : 
  ((a > b) → (a > b - 1)) ∧ ¬((a > b - 1) → (a > b)) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l575_57535


namespace NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l575_57525

theorem integer_solutions_cubic_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ 
    (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_cubic_equation_l575_57525


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l575_57515

theorem system_of_inequalities_solution (x : ℝ) : 
  (5 / (x + 3) ≥ 1 ∧ x^2 + x - 2 ≥ 0) ↔ ((-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l575_57515


namespace NUMINAMATH_CALUDE_trigonometric_problem_l575_57521

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (1 / (Real.cos x ^ 2 - Real.sin x ^ 2) = 25/7) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l575_57521


namespace NUMINAMATH_CALUDE_johns_allowance_l575_57549

theorem johns_allowance (A : ℝ) : A = 2.40 ↔ 
  ∃ (arcade_spent toy_store_spent candy_store_spent : ℝ),
    arcade_spent = (3/5) * A ∧
    toy_store_spent = (1/3) * (A - arcade_spent) ∧
    candy_store_spent = A - arcade_spent - toy_store_spent ∧
    candy_store_spent = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l575_57549


namespace NUMINAMATH_CALUDE_triangle_count_is_twenty_l575_57516

/-- Represents a point on the 3x3 grid -/
structure GridPoint where
  x : Fin 3
  y : Fin 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The set of all possible triangles on the 3x3 grid -/
def allGridTriangles : Set GridTriangle := sorry

/-- Counts the number of triangles in the 3x3 grid -/
def countTriangles : ℕ := sorry

/-- Theorem stating that the number of triangles in the 3x3 grid is 20 -/
theorem triangle_count_is_twenty : countTriangles = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_count_is_twenty_l575_57516


namespace NUMINAMATH_CALUDE_no_quadratic_trinomials_with_integer_roots_l575_57539

theorem no_quadratic_trinomials_with_integer_roots : 
  ¬ ∃ (a b c x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, a * x^2 + b * x + c = a * (x - x₁) * (x - x₂)) ∧ 
    (∀ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = (a + 1) * (x - x₃) * (x - x₄)) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomials_with_integer_roots_l575_57539


namespace NUMINAMATH_CALUDE_sum_of_squares_equivalence_l575_57557

theorem sum_of_squares_equivalence (n : ℕ) :
  (∃ (a b : ℤ), (n : ℤ) = a^2 + b^2) ↔ (∃ (c d : ℤ), (2 * n : ℤ) = c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equivalence_l575_57557


namespace NUMINAMATH_CALUDE_perpendicular_lines_l575_57583

theorem perpendicular_lines (a : ℝ) : 
  (∃ (x y : ℝ), x + a * y - a = 0 ∧ a * x - (2 * a - 3) * y - 1 = 0) →
  ((-1 : ℝ) / a) * (a / (2 * a - 3)) = -1 →
  a = 0 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l575_57583


namespace NUMINAMATH_CALUDE_fundraising_amount_scientific_notation_l575_57519

/-- Represents the amount in yuan --/
def amount : ℝ := 2.175e9

/-- Represents the number of significant figures to preserve --/
def significant_figures : ℕ := 3

/-- Converts a number to scientific notation with a specified number of significant figures --/
noncomputable def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem fundraising_amount_scientific_notation :
  to_scientific_notation amount significant_figures = (2.18, 9) := by sorry

end NUMINAMATH_CALUDE_fundraising_amount_scientific_notation_l575_57519


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l575_57552

/-- Given vectors a and b in ℝ², if a + k * b is perpendicular to a - b, then k = 11/20 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1)) 
  (h2 : b = (-1, 4)) 
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - b.1, a.2 - b.2) = 0) : 
  k = 11/20 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l575_57552


namespace NUMINAMATH_CALUDE_sixth_sample_number_l575_57580

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (between 000 and 799) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 799

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The main theorem --/
theorem sixth_sample_number
  (table : RandomNumberTable)
  (startRow : Nat)
  (startCol : Nat) :
  findNthValidNumber (table.join.drop (startRow * table.head!.length + startCol)) 6 = some 245 :=
sorry

end NUMINAMATH_CALUDE_sixth_sample_number_l575_57580


namespace NUMINAMATH_CALUDE_rectangle_division_integer_dimension_l575_57502

/-- A rectangle with dimensions a and b can be divided into unit-width strips -/
structure RectangleDivision (a b : ℝ) : Prop where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (can_divide : ∃ (strips : Set (ℝ × ℝ)), 
    (∀ s ∈ strips, (s.1 = 1 ∨ s.2 = 1)) ∧ 
    (∀ x y, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b → 
      ∃ s ∈ strips, (0 ≤ x - s.1 ∧ x < s.1) ∧ (0 ≤ y - s.2 ∧ y < s.2)))

/-- If a rectangle can be divided into unit-width strips, then one of its dimensions is an integer -/
theorem rectangle_division_integer_dimension (a b : ℝ) 
  (h : RectangleDivision a b) : 
  ∃ n : ℕ, (a = n) ∨ (b = n) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_integer_dimension_l575_57502


namespace NUMINAMATH_CALUDE_circus_ticket_price_l575_57572

theorem circus_ticket_price :
  ∀ (adult_price kid_price : ℝ),
    kid_price = (1/2) * adult_price →
    6 * kid_price + 2 * adult_price = 50 →
    kid_price = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l575_57572


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l575_57551

theorem sin_cos_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l575_57551


namespace NUMINAMATH_CALUDE_second_smallest_divisor_l575_57508

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem second_smallest_divisor (n : ℕ) : 
  (is_divisible (n + 3) 12 ∧ 
   is_divisible (n + 3) 35 ∧ 
   is_divisible (n + 3) 40) →
  (∀ m : ℕ, m < n → ¬(is_divisible (m + 3) 12 ∧ 
                      is_divisible (m + 3) 35 ∧ 
                      is_divisible (m + 3) 40)) →
  (∃ d : ℕ, d ≠ 1 ∧ is_divisible (n + 3) d ∧ 
   d ≠ 12 ∧ d ≠ 35 ∧ d ≠ 40 ∧
   (∀ k : ℕ, 1 < k → k < d → ¬is_divisible (n + 3) k)) →
  is_divisible (n + 3) 3 :=
by sorry

end NUMINAMATH_CALUDE_second_smallest_divisor_l575_57508


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l575_57538

theorem sqrt_x_minus_3_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l575_57538


namespace NUMINAMATH_CALUDE_product_difference_sum_problem_l575_57542

theorem product_difference_sum_problem : 
  ∃ (a b : ℕ+), (a * b = 18) ∧ (max a b - min a b = 3) → (a + b = 9) :=
by sorry

end NUMINAMATH_CALUDE_product_difference_sum_problem_l575_57542


namespace NUMINAMATH_CALUDE_restaurant_bill_theorem_l575_57530

def bill_with_discount (bill : Float) (discount_rate : Float) : Float :=
  bill * (1 - discount_rate / 100)

def total_bill (bob_bill kate_bill john_bill sarah_bill : Float)
               (bob_discount kate_discount john_discount sarah_discount : Float) : Float :=
  bill_with_discount bob_bill bob_discount +
  bill_with_discount kate_bill kate_discount +
  bill_with_discount john_bill john_discount +
  bill_with_discount sarah_bill sarah_discount

theorem restaurant_bill_theorem :
  total_bill 35.50 29.75 43.20 27.35 5.75 2.35 3.95 9.45 = 128.76945 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_theorem_l575_57530


namespace NUMINAMATH_CALUDE_triangle_inequality_l575_57553

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a + b) * (b + c) * (c + a) ≥ 8 * (a + b - c) * (b + c - a) * (c + a - b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l575_57553


namespace NUMINAMATH_CALUDE_engine_capacity_l575_57581

/-- The engine capacity (in cc) for which 85 litres of diesel is required to travel 600 km -/
def C : ℝ := 595

/-- The volume of diesel (in litres) required for the reference engine -/
def V₁ : ℝ := 170

/-- The capacity (in cc) of the reference engine -/
def C₁ : ℝ := 1200

/-- The volume of diesel (in litres) required for the engine capacity C -/
def V₂ : ℝ := 85

/-- The ratio of volume to capacity is constant -/
axiom volume_capacity_ratio : V₁ / C₁ = V₂ / C

theorem engine_capacity : C = 595 := by sorry

end NUMINAMATH_CALUDE_engine_capacity_l575_57581


namespace NUMINAMATH_CALUDE_polygon_sides_l575_57585

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 900 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l575_57585


namespace NUMINAMATH_CALUDE_school_distance_is_150km_l575_57563

/-- The distance from Xiaoming's home to school in kilometers. -/
def school_distance : ℝ := 150

/-- Xiaoming's walking speed in km/h. -/
def walking_speed : ℝ := 5

/-- The car speed in km/h. -/
def car_speed : ℝ := 15

/-- The time difference between going to school and returning home in hours. -/
def time_difference : ℝ := 2

/-- Theorem stating the distance from Xiaoming's home to school is 150 km. -/
theorem school_distance_is_150km :
  let d := school_distance
  let v_walk := walking_speed
  let v_car := car_speed
  let t_diff := time_difference
  (d / (2 * v_walk) + d / (2 * v_car) = d / (3 * v_car) + 2 * d / (3 * v_walk) + t_diff) →
  d = 150 := by
  sorry


end NUMINAMATH_CALUDE_school_distance_is_150km_l575_57563


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l575_57570

/-- The value of 'a' for which the focus of the parabola y = ax^2 (a > 0) 
    coincides with one of the foci of the hyperbola y^2 - x^2 = 2 -/
theorem parabola_hyperbola_focus_coincidence (a : ℝ) : 
  a > 0 → 
  (∃ (x y : ℝ), y = a * x^2 ∧ y^2 - x^2 = 2 ∧ 
    ((x = 0 ∧ y = 1 / (4 * a)) ∨ (x = 0 ∧ y = 2) ∨ (x = 0 ∧ y = -2))) → 
  a = 1/8 := by
sorry


end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincidence_l575_57570


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l575_57545

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l575_57545


namespace NUMINAMATH_CALUDE_paint_calculation_l575_57543

theorem paint_calculation (initial_paint : ℚ) : 
  (initial_paint / 9 + (initial_paint - initial_paint / 9) / 5 = 104) →
  initial_paint = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l575_57543


namespace NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l575_57569

/-- The length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
theorem train_length_calculation (length_A : ℝ) (speed_A speed_B : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_A_ms := speed_A * 1000 / 3600
  let speed_B_ms := speed_B * 1000 / 3600
  let relative_speed := speed_A_ms + speed_B_ms
  let total_distance := relative_speed * crossing_time
  total_distance - length_A

/-- The length of Train B is approximately 299.95 meters. -/
theorem train_B_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 200 120 80 9 - 299.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_train_B_length_l575_57569


namespace NUMINAMATH_CALUDE_rice_weight_scientific_notation_l575_57582

theorem rice_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.000035 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.5 ∧ n = -5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_scientific_notation_l575_57582


namespace NUMINAMATH_CALUDE_sine_cosine_extreme_value_l575_57504

open Real

theorem sine_cosine_extreme_value (a b : ℝ) (h : a < b) :
  ∃ f g : ℝ → ℝ,
    (∀ x ∈ Set.Icc a b, f x = sin x ∧ g x = cos x) ∧
    g a * g b < 0 ∧
    ¬(∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, g x ≤ g y ∨ g x ≥ g y) :=
by sorry

end NUMINAMATH_CALUDE_sine_cosine_extreme_value_l575_57504


namespace NUMINAMATH_CALUDE_quadratic_minimum_l575_57564

theorem quadratic_minimum (p q : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x, y x = x^2 + p*x + q) ∧ 
   (∃ x₀, ∀ x, y x₀ ≤ y x) ∧ 
   (∃ x₁, y x₁ = 0)) →
  q = p^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l575_57564


namespace NUMINAMATH_CALUDE_unique_n_with_no_constant_term_l575_57579

/-- The expansion of (1+x+x²)(x+1/x³)ⁿ has no constant term -/
def has_no_constant_term (n : ℕ) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → (1 + x + x^2) * (x + 1/x^3)^n ≠ 1

theorem unique_n_with_no_constant_term :
  ∃! (n : ℕ), 2 ≤ n ∧ n ≤ 8 ∧ has_no_constant_term n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_no_constant_term_l575_57579


namespace NUMINAMATH_CALUDE_probability_of_perfect_square_sum_l575_57596

/-- The number of faces on a standard die -/
def standardDieFaces : ℕ := 6

/-- The set of possible sums when rolling two dice -/
def possibleSums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of perfect squares within the possible sums -/
def perfectSquareSums : Set ℕ := {4, 9}

/-- The number of ways to get a sum of 4 -/
def waysToGetFour : ℕ := 3

/-- The number of ways to get a sum of 9 -/
def waysToGetNine : ℕ := 4

/-- The total number of favorable outcomes -/
def favorableOutcomes : ℕ := waysToGetFour + waysToGetNine

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := standardDieFaces * standardDieFaces

/-- Theorem: The probability of rolling two standard 6-sided dice and getting a sum that is a perfect square is 7/36 -/
theorem probability_of_perfect_square_sum :
  (favorableOutcomes : ℚ) / totalOutcomes = 7 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_perfect_square_sum_l575_57596


namespace NUMINAMATH_CALUDE_expression_simplification_l575_57562

theorem expression_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 + b^2 ≠ 0) :
  (1 / (a - b) - (2 * a * b) / (a^3 - a^2 * b + a * b^2 - b^3)) /
  ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + b / (a^2 + b^2)) =
  (a - b) / (a + b) :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_l575_57562


namespace NUMINAMATH_CALUDE_function_composition_equality_l575_57546

theorem function_composition_equality (m n p q c : ℝ) :
  let f := fun (x : ℝ) => m * x + n + c
  let g := fun (x : ℝ) => p * x + q + c
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l575_57546


namespace NUMINAMATH_CALUDE_salary_change_l575_57594

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.15)
  let final_salary := increased_salary * (1 - 0.15)
  (final_salary - initial_salary) / initial_salary = -0.0225 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l575_57594


namespace NUMINAMATH_CALUDE_sum_min_max_x_l575_57513

theorem sum_min_max_x (x y z : ℝ) (sum_eq : x + y + z = 5) (sum_sq_eq : x^2 + y^2 + z^2 = 8) :
  ∃ (m M : ℝ), (∀ x' y' z' : ℝ, x' + y' + z' = 5 → x'^2 + y'^2 + z'^2 = 8 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 4 :=
sorry

end NUMINAMATH_CALUDE_sum_min_max_x_l575_57513


namespace NUMINAMATH_CALUDE_calculate_expression_l575_57514

theorem calculate_expression : (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l575_57514


namespace NUMINAMATH_CALUDE_commutator_power_zero_l575_57558

open Matrix

theorem commutator_power_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (h_n : n ≥ 2) 
  (h_x : ∃ x : ℝ, x ≠ 0 ∧ x ≠ 1/2 ∧ x ≠ 1 ∧ x • (A * B) + (1 - x) • (B * A) = 1) :
  (A * B - B * A) ^ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_commutator_power_zero_l575_57558


namespace NUMINAMATH_CALUDE_sun_op_example_l575_57547

-- Define the ☼ operation
def sunOp (a b : ℚ) : ℚ := a^3 - 2*a*b + 4

-- Theorem statement
theorem sun_op_example : sunOp 4 (-9) = 140 := by sorry

end NUMINAMATH_CALUDE_sun_op_example_l575_57547


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l575_57534

/-- For a parabola defined by y = 2x^2, the distance from its focus to its directrix is 1/2 -/
theorem parabola_focus_directrix_distance (x y : ℝ) :
  y = 2 * x^2 → ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ 
     focus_y = 1/4 ∧
     directrix_y = -1/4 ∧
     focus_y - directrix_y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l575_57534


namespace NUMINAMATH_CALUDE_sufficient_condition_for_monotonic_decrease_l575_57593

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the property of being monotonic decreasing on an interval
def monotonic_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y ≤ f x

-- Theorem statement
theorem sufficient_condition_for_monotonic_decrease :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f' x) →
    (monotonic_decreasing_on (fun x ↦ f (x + 1)) 0 1) ∧
    ¬(∀ g : ℝ → ℝ, (∀ x, deriv g x = f' x) → 
      monotonic_decreasing_on (fun x ↦ g (x + 1)) 0 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_monotonic_decrease_l575_57593


namespace NUMINAMATH_CALUDE_least_valid_tree_count_l575_57559

def is_valid_tree_count (n : ℕ) : Prop :=
  n ≥ 100 ∧ n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0

theorem least_valid_tree_count :
  ∃ (n : ℕ), is_valid_tree_count n ∧ ∀ m < n, ¬is_valid_tree_count m :=
by sorry

end NUMINAMATH_CALUDE_least_valid_tree_count_l575_57559


namespace NUMINAMATH_CALUDE_shooting_test_probability_l575_57565

/-- The number of shots in the test -/
def num_shots : ℕ := 3

/-- The minimum number of successful shots required to pass -/
def min_success : ℕ := 2

/-- The probability of making a single shot -/
def shot_probability : ℝ := 0.6

/-- The probability of passing the test -/
def pass_probability : ℝ := 0.648

/-- Theorem stating that the calculated probability of passing the test is correct -/
theorem shooting_test_probability : 
  (Finset.sum (Finset.range (num_shots - min_success + 1))
    (λ k => Nat.choose num_shots (num_shots - k) * 
      shot_probability ^ (num_shots - k) * 
      (1 - shot_probability) ^ k)) = pass_probability := by
  sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l575_57565


namespace NUMINAMATH_CALUDE_overall_length_is_13_l575_57506

/-- The length of each ruler in centimeters -/
def ruler_length : ℝ := 10

/-- The mark on the first ruler that aligns with the second ruler -/
def align_mark1 : ℝ := 3

/-- The mark on the second ruler that aligns with the first ruler -/
def align_mark2 : ℝ := 4

/-- The overall length when the rulers are aligned as described -/
def L : ℝ := ruler_length + (ruler_length - align_mark2) - (align_mark2 - align_mark1)

theorem overall_length_is_13 : L = 13 := by
  sorry

end NUMINAMATH_CALUDE_overall_length_is_13_l575_57506
