import Mathlib

namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l142_14225

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with length 8 cm, breadth 6 cm, and surface area 432 cm² has a height of 12 cm -/
theorem cuboid_height_calculation (h : ℝ) : 
  cuboidSurfaceArea 8 6 h = 432 → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l142_14225


namespace NUMINAMATH_CALUDE_equation_solution_l142_14251

theorem equation_solution : ∃ X : ℝ,
  (15.2 * 0.25 - 48.51 / 14.7) / X =
  ((13/44 - 2/11 - 5/66 / (5/2)) * (6/5)) / (3.2 + 0.8 * (5.5 - 3.25)) ∧
  X = 137.5 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l142_14251


namespace NUMINAMATH_CALUDE_standard_deviation_of_data_set_l142_14226

def data_set : List ℝ := [11, 13, 15, 17, 19]

theorem standard_deviation_of_data_set :
  let n : ℕ := data_set.length
  let mean : ℝ := data_set.sum / n
  let variance : ℝ := (data_set.map (λ x => (x - mean)^2)).sum / n
  let std_dev : ℝ := Real.sqrt variance
  (mean = 15) → (std_dev = 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_standard_deviation_of_data_set_l142_14226


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l142_14211

theorem largest_prime_divisor_to_test (n : ℕ) (h : 1000 ≤ n ∧ n ≤ 1100) :
  (∀ p : ℕ, p.Prime → p ≤ 31 → ¬(p ∣ n)) → n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l142_14211


namespace NUMINAMATH_CALUDE_next_five_even_sum_l142_14235

/-- Given a sum of 5 consecutive even positive integers with one divisible by 13,
    the sum of the next 5 even consecutive integers is 50 more than the original sum. -/
theorem next_five_even_sum (a : ℕ) (x : ℕ) : 
  (∃ k : ℕ, x = 26 * k) →
  a = (x - 4) + (x - 2) + x + (x + 2) + (x + 4) →
  (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = a + 50 := by
sorry

end NUMINAMATH_CALUDE_next_five_even_sum_l142_14235


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l142_14255

theorem polynomial_divisibility (p q : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x - 6)) ↔ 
  (p = 0 ∧ q = -9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l142_14255


namespace NUMINAMATH_CALUDE_sugar_water_ratio_l142_14210

theorem sugar_water_ratio (total_cups sugar_cups : ℕ) : 
  total_cups = 84 → sugar_cups = 28 → 
  ∃ (a b : ℕ), a = 1 ∧ b = 2 ∧ sugar_cups * b = (total_cups - sugar_cups) * a :=
by sorry

end NUMINAMATH_CALUDE_sugar_water_ratio_l142_14210


namespace NUMINAMATH_CALUDE_sum_medial_areas_is_one_third_l142_14217

/-- Definition of a medial triangle -/
def medialTriangle (T : Set ℝ × Set ℝ) : Set ℝ × Set ℝ := sorry

/-- Area of a triangle -/
def area (T : Set ℝ × Set ℝ) : ℝ := sorry

/-- Sequence of medial triangles -/
def medialSequence (T : Set ℝ × Set ℝ) : ℕ → Set ℝ × Set ℝ
  | 0 => T
  | n + 1 => medialTriangle (medialSequence T n)

/-- Sum of areas of medial triangles -/
def sumMedialAreas (T : Set ℝ × Set ℝ) : ℝ := sorry

theorem sum_medial_areas_is_one_third (T : Set ℝ × Set ℝ) 
  (h : area T = 1) : sumMedialAreas T = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_medial_areas_is_one_third_l142_14217


namespace NUMINAMATH_CALUDE_polynomial_factors_l142_14279

/-- The polynomial 3x^4 - hx^2 + kx - 7 has x+1 and x-3 as factors if and only if h = 124/3 and k = 136/3 -/
theorem polynomial_factors (h k : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 3) ∣ (3 * x^4 - h * x^2 + k * x - 7)) ↔ 
  (h = 124/3 ∧ k = 136/3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factors_l142_14279


namespace NUMINAMATH_CALUDE_problem_solution_l142_14249

theorem problem_solution : 
  (3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2) ∧ 
  (Real.sqrt 6 * Real.sqrt 3 + Real.sqrt 2 - 6 * Real.sqrt (1/2) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l142_14249


namespace NUMINAMATH_CALUDE_commute_days_eq_seventeen_l142_14296

/-- Represents the commute options for a woman over a period of working days. -/
structure CommuteData where
  total_days : ℕ
  morning_bus : ℕ
  afternoon_car : ℕ
  total_car : ℕ

/-- Theorem stating that given the specific commute data, the total number of working days is 17. -/
theorem commute_days_eq_seventeen (data : CommuteData) 
  (h1 : data.morning_bus = 10)
  (h2 : data.afternoon_car = 13)
  (h3 : data.total_car = 11)
  (h4 : data.morning_bus + (data.total_car - data.afternoon_car) = data.total_days)
  : data.total_days = 17 := by
  sorry

#check commute_days_eq_seventeen

end NUMINAMATH_CALUDE_commute_days_eq_seventeen_l142_14296


namespace NUMINAMATH_CALUDE_parabola_vertex_l142_14293

/-- Given a quadratic function f(x) = -x^2 + px + q where f(x) ≤ 0 has roots at x = -2 and x = 8,
    the vertex of the parabola defined by f(x) is at (3, -7). -/
theorem parabola_vertex (p q : ℝ) (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = -x^2 + p*x + q)
    (h_roots : f (-2) = 0 ∧ f 8 = 0)
    (h_solution : ∀ x, x ∈ Set.Icc (-2) 8 ↔ f x ≤ 0) :
    ∃ (vertex : ℝ × ℝ), vertex = (3, -7) ∧ 
    ∀ x, f x ≤ f (vertex.1) ∧ 
    (x ≠ vertex.1 → f x < f (vertex.1)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l142_14293


namespace NUMINAMATH_CALUDE_unique_circle_digits_l142_14265

theorem unique_circle_digits : ∃! (a b c d e : ℕ),
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (a + b = (c + d + e) / 7) ∧
  (a + c = (b + d + e) / 5) :=
by sorry

end NUMINAMATH_CALUDE_unique_circle_digits_l142_14265


namespace NUMINAMATH_CALUDE_integer_product_equivalence_l142_14237

theorem integer_product_equivalence (a : ℝ) :
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = m) ↔
  (∃ k : ℤ, a = k / 6) :=
by sorry

end NUMINAMATH_CALUDE_integer_product_equivalence_l142_14237


namespace NUMINAMATH_CALUDE_eight_b_value_l142_14262

theorem eight_b_value (a b : ℚ) 
  (eq1 : 4 * a + 3 * b = 5)
  (eq2 : a = b - 3) :
  8 * b = 136 / 7 := by
sorry

end NUMINAMATH_CALUDE_eight_b_value_l142_14262


namespace NUMINAMATH_CALUDE_prob_select_green_is_101_180_l142_14245

def container_I : ℕ × ℕ := (12, 6)
def container_II : ℕ × ℕ := (4, 6)
def container_III : ℕ × ℕ := (3, 9)

def total_balls (c : ℕ × ℕ) : ℕ := c.1 + c.2

def prob_green (c : ℕ × ℕ) : ℚ :=
  c.2 / (total_balls c)

def prob_select_green : ℚ :=
  (1/3) * (prob_green container_I) +
  (1/3) * (prob_green container_II) +
  (1/3) * (prob_green container_III)

theorem prob_select_green_is_101_180 :
  prob_select_green = 101/180 := by sorry

end NUMINAMATH_CALUDE_prob_select_green_is_101_180_l142_14245


namespace NUMINAMATH_CALUDE_bert_tax_percentage_l142_14232

/-- Represents the tax percentage as a real number between 0 and 1 -/
def tax_percentage : ℝ := sorry

/-- The amount by which Bert increases the price when selling -/
def price_increase : ℝ := 10

/-- The selling price of the barrel -/
def selling_price : ℝ := 90

/-- Bert's profit on the sale -/
def profit : ℝ := 1

theorem bert_tax_percentage :
  tax_percentage = 0.1 ∧
  selling_price = (selling_price - price_increase) + price_increase ∧
  profit = selling_price - (selling_price - price_increase) - (tax_percentage * selling_price) :=
by sorry

end NUMINAMATH_CALUDE_bert_tax_percentage_l142_14232


namespace NUMINAMATH_CALUDE_percent_greater_than_average_l142_14299

theorem percent_greater_than_average (M N : ℝ) (h : M > N) :
  (M - (M + N) / 2) / ((M + N) / 2) * 100 = 200 * (M - N) / (M + N) := by
  sorry

end NUMINAMATH_CALUDE_percent_greater_than_average_l142_14299


namespace NUMINAMATH_CALUDE_roots_properties_l142_14221

def i : ℂ := Complex.I

def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = -4 + 8*i

def roots (z₁ z₂ : ℂ) : Prop :=
  quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ z₁ ≠ z₂

theorem roots_properties :
  ∃ z₁ z₂ : ℂ, roots z₁ z₂ ∧
  (z₁.re * z₂.re = -7) ∧
  (z₁.im + z₂.im = 0) := by sorry

end NUMINAMATH_CALUDE_roots_properties_l142_14221


namespace NUMINAMATH_CALUDE_eunji_score_l142_14250

theorem eunji_score (minyoung_score yuna_score eunji_score : ℕ) : 
  minyoung_score = 55 →
  yuna_score = 57 →
  eunji_score > minyoung_score →
  eunji_score < yuna_score →
  eunji_score = 56 := by
sorry

end NUMINAMATH_CALUDE_eunji_score_l142_14250


namespace NUMINAMATH_CALUDE_tank_capacity_l142_14227

theorem tank_capacity (x : ℝ) 
  (h1 : x / 4 + 180 = 2 * x / 3) : x = 432 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l142_14227


namespace NUMINAMATH_CALUDE_triangle_inequality_power_l142_14244

theorem triangle_inequality_power (a b c S : ℝ) (n : ℝ) : 
  a > 0 → b > 0 → c > 0 → -- triangle side lengths are positive
  a + b > c → b + c > a → c + a > b → -- triangle inequality
  2 * S = a + b + c → -- perimeter definition
  n ≥ 1 → -- condition on n
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ (2/3)^(n-2) * S^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_power_l142_14244


namespace NUMINAMATH_CALUDE_rotten_oranges_percentage_l142_14284

theorem rotten_oranges_percentage 
  (total_oranges : ℕ) 
  (total_bananas : ℕ) 
  (rotten_bananas_percentage : ℚ) 
  (good_fruits_percentage : ℚ) :
  total_oranges = 600 →
  total_bananas = 400 →
  rotten_bananas_percentage = 8 / 100 →
  good_fruits_percentage = 878 / 1000 →
  (total_oranges - (good_fruits_percentage * (total_oranges + total_bananas : ℚ) - rotten_bananas_percentage * total_bananas)) / total_oranges = 15 / 100 :=
by sorry

end NUMINAMATH_CALUDE_rotten_oranges_percentage_l142_14284


namespace NUMINAMATH_CALUDE_triangle_qca_area_l142_14258

/-- Given points Q, A, C in a coordinate plane and that triangle QCA is right-angled at C,
    prove that the area of triangle QCA is (36 - 3p) / 2 -/
theorem triangle_qca_area (p : ℝ) : 
  let Q : ℝ × ℝ := (0, 12)
  let A : ℝ × ℝ := (3, 12)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := (1 / 2) * (A.1 - Q.1) * (Q.2 - C.2)
  triangle_area = (36 - 3*p) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_qca_area_l142_14258


namespace NUMINAMATH_CALUDE_hcf_problem_l142_14267

/-- Given two positive integers with specific LCM and maximum value properties, prove their HCF is 4 -/
theorem hcf_problem (a b : ℕ+) : 
  (∃ (lcm : ℕ+), Nat.lcm a b = lcm ∧ ∃ (hcf : ℕ+), lcm = hcf * 10 * 20) →
  (max a b = 840) →
  Nat.gcd a b = 4 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l142_14267


namespace NUMINAMATH_CALUDE_figure_area_solution_l142_14281

theorem figure_area_solution (x : ℝ) : 
  (3 * x)^2 + (6 * x)^2 + (1/2 * 3 * x * 6 * x) = 1950 → 
  x = (5 * Real.sqrt 13) / 3 := by
sorry

end NUMINAMATH_CALUDE_figure_area_solution_l142_14281


namespace NUMINAMATH_CALUDE_partition_sum_property_l142_14230

theorem partition_sum_property (n : ℕ) (A B C : Finset ℕ) :
  n > 0 →
  A ∪ B ∪ C = Finset.range (3 * n) →
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ →
  Finset.card A = n ∧ Finset.card B = n ∧ Finset.card C = n →
  ∃ (x y z : ℕ), x ∈ A ∧ y ∈ B ∧ z ∈ C ∧
    (x + y = z ∨ x + z = y ∨ y + z = x) :=
by sorry

end NUMINAMATH_CALUDE_partition_sum_property_l142_14230


namespace NUMINAMATH_CALUDE_symmetric_circle_l142_14295

/-- Given a circle with equation (x+2)^2 + y^2 = 5 and a line of symmetry y = x,
    the symmetric circle has the equation x^2 + (y+2)^2 = 5 -/
theorem symmetric_circle (x y : ℝ) :
  (∃ (x₀ y₀ : ℝ), (x₀ + 2)^2 + y₀^2 = 5 ∧ y₀ = x₀) →
  (∃ (x₁ y₁ : ℝ), x₁^2 + (y₁ + 2)^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_l142_14295


namespace NUMINAMATH_CALUDE_binary_of_28_l142_14248

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_of_28 :
  decimal_to_binary 28 = [1, 1, 1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_binary_of_28_l142_14248


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l142_14277

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_2_dividing_32_factorial_l142_14277


namespace NUMINAMATH_CALUDE_harmonic_set_odd_cardinality_min_harmonic_set_cardinality_l142_14215

/-- A set of positive integers is a "harmonic set" if removing any element
    results in the remaining elements being divisible into two disjoint sets
    with equal sum of elements. -/
def is_harmonic_set (A : Finset ℕ) : Prop :=
  A.card ≥ 3 ∧ ∀ a ∈ A, ∃ B C : Finset ℕ,
    B ⊆ A \ {a} ∧ C ⊆ A \ {a} ∧ B ∩ C = ∅ ∧ B ∪ C = A \ {a} ∧
    (B.sum id = C.sum id)

theorem harmonic_set_odd_cardinality (A : Finset ℕ) (h : is_harmonic_set A) :
  Odd A.card :=
sorry

theorem min_harmonic_set_cardinality :
  ∃ A : Finset ℕ, is_harmonic_set A ∧ A.card = 7 ∧
    ∀ B : Finset ℕ, is_harmonic_set B → B.card ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_harmonic_set_odd_cardinality_min_harmonic_set_cardinality_l142_14215


namespace NUMINAMATH_CALUDE_simplify_expression_l142_14292

theorem simplify_expression (b c : ℝ) :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l142_14292


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l142_14236

/-- Given a quadratic equation x^2 - 2(k-1)x + 2k^2 - 12k + 17 = 0 with roots x₁ and x₂,
    this theorem proves properties about the maximum and minimum values of x₁² + x₂²
    and the roots at these values. -/
theorem quadratic_roots_properties :
  ∀ k x₁ x₂ : ℝ,
  (x₁^2 - 2*(k-1)*x₁ + 2*k^2 - 12*k + 17 = 0) →
  (x₂^2 - 2*(k-1)*x₂ + 2*k^2 - 12*k + 17 = 0) →
  (∃ kmax : ℝ, (x₁^2 + x₂^2 ≤ 98) ∧ (k = kmax → x₁^2 + x₂^2 = 98) ∧ (k = kmax → x₁ = 7 ∧ x₂ = 7)) ∧
  (∃ kmin : ℝ, (x₁^2 + x₂^2 ≥ 2) ∧ (k = kmin → x₁^2 + x₂^2 = 2) ∧ (k = kmin → x₁ = 1 ∧ x₂ = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l142_14236


namespace NUMINAMATH_CALUDE_kim_morning_routine_time_l142_14241

/-- Represents the time in minutes for Kim's morning routine -/
def morning_routine_time (coffee_time : ℕ) (status_update_time : ℕ) (payroll_update_time : ℕ) (num_employees : ℕ) : ℕ :=
  coffee_time + (status_update_time + payroll_update_time) * num_employees

/-- Theorem stating that Kim's morning routine takes 50 minutes -/
theorem kim_morning_routine_time :
  morning_routine_time 5 2 3 9 = 50 := by
  sorry

#eval morning_routine_time 5 2 3 9

end NUMINAMATH_CALUDE_kim_morning_routine_time_l142_14241


namespace NUMINAMATH_CALUDE_average_marks_l142_14214

theorem average_marks (total_subjects : Nat) (subjects_avg_5 : Nat) (subject_6_mark : Nat) :
  total_subjects = 6 →
  subjects_avg_5 = 74 →
  subject_6_mark = 92 →
  ((subjects_avg_5 * 5 + subject_6_mark) : ℚ) / total_subjects = 77 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l142_14214


namespace NUMINAMATH_CALUDE_isosceles_triangle_equation_l142_14223

def isIsosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def isRoot (x n : ℝ) : Prop := x^2 - 8*x + n = 0

theorem isosceles_triangle_equation (n : ℝ) : 
  (∃ (a b : ℝ), isIsosceles 3 a b ∧ isRoot a n ∧ isRoot b n) → (n = 15 ∨ n = 16) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_equation_l142_14223


namespace NUMINAMATH_CALUDE_pipe_C_empty_time_l142_14216

/-- Represents the time (in minutes) it takes for pipe C to empty the cistern. -/
def empty_time (fill_time_A fill_time_B fill_time_all : ℚ) : ℚ :=
  let rate_A := 1 / fill_time_A
  let rate_B := 1 / fill_time_B
  let rate_all := 1 / fill_time_all
  let rate_C := rate_A + rate_B - rate_all
  1 / rate_C

/-- Theorem stating that given the fill times for pipes A and B, and the fill time when all pipes are open,
    the time it takes for pipe C to empty the cistern is 72 minutes. -/
theorem pipe_C_empty_time :
  empty_time 45 60 40 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pipe_C_empty_time_l142_14216


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_no_even_digits_l142_14287

-- Define a function to check if a natural number has no even digits
def has_no_even_digits (k : ℕ) : Prop := sorry

-- State the theorem
theorem exists_m_for_all_n_no_even_digits :
  ∃ m : ℕ+, ∀ n : ℕ+, has_no_even_digits ((5 : ℕ) ^ n.val * m.val) := by sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_no_even_digits_l142_14287


namespace NUMINAMATH_CALUDE_baker_initial_cakes_l142_14275

theorem baker_initial_cakes 
  (bought : ℕ) 
  (sold : ℕ) 
  (difference : ℕ) 
  (h1 : bought = 139)
  (h2 : sold = 145)
  (h3 : sold = bought + difference)
  (h4 : difference = 6) : 
  sold - bought = 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_initial_cakes_l142_14275


namespace NUMINAMATH_CALUDE_sqrt_expression_value_l142_14218

theorem sqrt_expression_value (x y : ℝ) 
  (h : Real.sqrt (x + 5) + (2 * x - y)^2 = 0) : 
  Real.sqrt (x^2 - 2*x*y + y^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_value_l142_14218


namespace NUMINAMATH_CALUDE_container_volume_ratio_l142_14204

theorem container_volume_ratio : 
  ∀ (A B C : ℝ),
  A > 0 → B > 0 → C > 0 →
  (8/9 : ℝ) * A = (7/9 : ℝ) * B →
  (7/9 : ℝ) * B + (1/2 : ℝ) * C = C →
  A / C = 63 / 112 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l142_14204


namespace NUMINAMATH_CALUDE_nickys_pace_l142_14282

/-- Prove Nicky's pace given the race conditions -/
theorem nickys_pace (race_length : ℝ) (head_start : ℝ) (cristina_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_length = 500)
  (h2 : head_start = 12)
  (h3 : cristina_pace = 5)
  (h4 : catch_up_time = 30) :
  cristina_pace = catch_up_time * race_length / (catch_up_time * cristina_pace) :=
by sorry

end NUMINAMATH_CALUDE_nickys_pace_l142_14282


namespace NUMINAMATH_CALUDE_medicine_tablets_l142_14201

theorem medicine_tablets (num_b : ℕ) (num_a : ℕ) (min_extract : ℕ) : 
  num_b = 14 → 
  min_extract = 16 → 
  min_extract = num_b + 2 →
  num_a = 2 :=
by sorry

end NUMINAMATH_CALUDE_medicine_tablets_l142_14201


namespace NUMINAMATH_CALUDE_trees_on_specific_road_l142_14288

/-- Calculates the number of trees that can be planted along a road -/
def treesAlongRoad (roadLength : ℕ) (treeSpacing : ℕ) : ℕ :=
  let intervalsPerSide := roadLength / treeSpacing
  let treesPerSide := intervalsPerSide - 1
  2 * treesPerSide

/-- The theorem stating the number of trees along the specific road -/
theorem trees_on_specific_road :
  treesAlongRoad 100 5 = 38 := by
  sorry

#eval treesAlongRoad 100 5

end NUMINAMATH_CALUDE_trees_on_specific_road_l142_14288


namespace NUMINAMATH_CALUDE_candy_bar_cost_l142_14261

theorem candy_bar_cost 
  (num_soft_drinks : ℕ) 
  (cost_per_soft_drink : ℕ) 
  (num_candy_bars : ℕ) 
  (total_spent : ℕ) 
  (h1 : num_soft_drinks = 2)
  (h2 : cost_per_soft_drink = 4)
  (h3 : num_candy_bars = 5)
  (h4 : total_spent = 28) :
  (total_spent - num_soft_drinks * cost_per_soft_drink) / num_candy_bars = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l142_14261


namespace NUMINAMATH_CALUDE_determinant_of_roots_l142_14259

theorem determinant_of_roots (p q : ℝ) (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 4*a^2 + p*a + q = 0 →
  b^3 - 4*b^2 + p*b + q = 0 →
  c^3 - 4*c^2 + p*c + q = 0 →
  Matrix.det !![a, b, c; b, c, a; c, a, b] = -64 + 12*p - 2*q := by
  sorry

end NUMINAMATH_CALUDE_determinant_of_roots_l142_14259


namespace NUMINAMATH_CALUDE_figure_50_squares_l142_14239

/-- The number of nonoverlapping unit squares in figure n -/
def g (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom pattern_holds : g 0 = 2 ∧ g 1 = 8 ∧ g 2 = 18 ∧ g 3 = 32

theorem figure_50_squares : g 50 = 5202 := by sorry

end NUMINAMATH_CALUDE_figure_50_squares_l142_14239


namespace NUMINAMATH_CALUDE_max_dominoes_8x8_10removed_l142_14280

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Nat)

/-- Calculates the maximum number of dominoes that can be placed on a chessboard -/
def max_dominoes (board : Chessboard) : Nat :=
  let remaining := board.size * board.size - board.removed
  let worst_case_color := min (board.size * board.size / 2) (remaining - (board.size * board.size / 2 - board.removed))
  worst_case_color

/-- Theorem stating the maximum number of dominoes on an 8x8 chessboard with 10 squares removed -/
theorem max_dominoes_8x8_10removed :
  max_dominoes { size := 8, removed := 10 } = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_dominoes_8x8_10removed_l142_14280


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l142_14256

/-- Given a geometric sequence with common ratio q > 0, prove that if S_2 = 3a_2 + 2 and S_4 = 3a_4 + 2, then a_1 = -1 -/
theorem geometric_sequence_problem (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_q_pos : q > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (h_sum : ∀ n, S n = a 1 * (1 - q^n) / (1 - q))
  (h_S2 : S 2 = 3 * a 2 + 2)
  (h_S4 : S 4 = 3 * a 4 + 2) :
  a 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l142_14256


namespace NUMINAMATH_CALUDE_three_digit_integer_property_l142_14247

def three_digit_integer (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def digit_sum (a b c : ℕ) : ℕ := a + b + c

def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem three_digit_integer_property (a b c : ℕ) 
  (h1 : a < 10 ∧ b < 10 ∧ c < 10) 
  (h2 : three_digit_integer a b c - 7 * digit_sum a b c = 100) :
  ∃ y : ℕ, reversed_number a b c = y * digit_sum a b c ∧ y = 43 := by
sorry

end NUMINAMATH_CALUDE_three_digit_integer_property_l142_14247


namespace NUMINAMATH_CALUDE_frequency_41_to_45_l142_14238

/-- Represents a school with teachers divided into age groups -/
structure School where
  total_teachers : ℕ
  age_groups : ℕ
  teachers_41_to_45 : ℕ

/-- Calculates the frequency of teachers in a specific age group -/
def frequency (s : School) : ℚ :=
  s.teachers_41_to_45 / s.total_teachers

/-- Theorem stating that the frequency of teachers aged 41-45 is 0.14 -/
theorem frequency_41_to_45 (s : School) 
  (h1 : s.total_teachers = 100) 
  (h2 : s.teachers_41_to_45 = 14) : 
  frequency s = 14/100 := by sorry

end NUMINAMATH_CALUDE_frequency_41_to_45_l142_14238


namespace NUMINAMATH_CALUDE_binomial_expansion_example_l142_14203

theorem binomial_expansion_example : 12^4 + 4*(12^3) + 6*(12^2) + 4*12 + 1 = 28561 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_example_l142_14203


namespace NUMINAMATH_CALUDE_intersection_distance_through_focus_m_value_for_perpendicular_intersections_l142_14205

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line x y m ∧ p ≠ (0, 0)}

-- Theorem 1: Distance between intersection points when line passes through focus
theorem intersection_distance_through_focus :
  ∀ m : ℝ, line 2 0 m →
  ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 2 :=
sorry

-- Theorem 2: Value of m when intersection points form right angle with origin
theorem m_value_for_perpendicular_intersections :
  ∃ m : ℝ, ∀ A B : ℝ × ℝ,
  A ∈ intersection_points m → B ∈ intersection_points m → A ≠ B →
  A.1 * B.1 + A.2 * B.2 = 0 → m = -8 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_through_focus_m_value_for_perpendicular_intersections_l142_14205


namespace NUMINAMATH_CALUDE_one_man_work_time_l142_14252

-- Define the work as a unit
def total_work : ℝ := 1

-- Define the time taken by the group
def group_time : ℝ := 6

-- Define the number of men and women in the group
def num_men : ℝ := 10
def num_women : ℝ := 15

-- Define the time taken by one woman
def woman_time : ℝ := 225

-- Define the time taken by one man (to be proved)
def man_time : ℝ := 100

-- Theorem statement
theorem one_man_work_time :
  (num_men / man_time + num_women / woman_time) * group_time = total_work →
  1 / man_time = 1 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_one_man_work_time_l142_14252


namespace NUMINAMATH_CALUDE_digit_theta_value_l142_14257

theorem digit_theta_value : ∃! (Θ : ℕ), 
  Θ > 0 ∧ Θ < 10 ∧ (252 : ℚ) / Θ = 30 + 2 * Θ := by
  sorry

end NUMINAMATH_CALUDE_digit_theta_value_l142_14257


namespace NUMINAMATH_CALUDE_triangle_inequality_l142_14269

/-- Triangle Inequality Theorem: For any triangle, the sum of the lengths of any two sides
    is greater than the length of the remaining side. -/
theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l142_14269


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l142_14229

theorem longest_side_of_triangle (y : ℝ) : 
  8 + (y + 5) + (3 * y + 2) = 47 → 
  max 8 (max (y + 5) (3 * y + 2)) = 26 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l142_14229


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l142_14222

theorem quadratic_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*m - 3)*x + m - 150 = 0 ∧
               y^2 + (2*m - 3)*y + m - 150 = 0 ∧
               x > 2 ∧ y < 2) ↔
  m > 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l142_14222


namespace NUMINAMATH_CALUDE_three_solutions_implies_a_gt_one_l142_14220

/-- The equation has three different real solutions -/
def has_three_solutions (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (1 / (x + 2) = a * |x|) ∧
    (1 / (y + 2) = a * |y|) ∧
    (1 / (z + 2) = a * |z|)

/-- If the equation has three different real solutions, then a > 1 -/
theorem three_solutions_implies_a_gt_one :
  ∀ a : ℝ, has_three_solutions a → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_three_solutions_implies_a_gt_one_l142_14220


namespace NUMINAMATH_CALUDE_special_numbers_count_l142_14213

/-- A function that checks if a number's digits are consecutive integers -/
def has_consecutive_digits (n : ℕ) : Prop := sorry

/-- A function that returns the number of integers satisfying the given conditions -/
def count_special_numbers : ℕ := sorry

/-- Theorem stating that there are exactly 66 numbers satisfying the given conditions -/
theorem special_numbers_count :
  (∃ (S : Finset ℕ), 
    S.card = 66 ∧ 
    (∀ n ∈ S, 
      1000 ≤ n ∧ n < 10000 ∧
      has_consecutive_digits n ∧
      n % 3 = 0) ∧
    (∀ n : ℕ, 
      1000 ≤ n ∧ n < 10000 ∧
      has_consecutive_digits n ∧
      n % 3 = 0 → n ∈ S)) :=
by sorry

#check special_numbers_count

end NUMINAMATH_CALUDE_special_numbers_count_l142_14213


namespace NUMINAMATH_CALUDE_right_handed_players_count_l142_14253

/-- Represents a football team with various player categories -/
structure FootballTeam where
  total_players : ℕ
  thrower_percentage : ℚ
  kicker_percentage : ℚ
  left_handed_remaining_percentage : ℚ
  left_handed_kicker_percentage : ℚ
  exclusive_thrower_percentage : ℚ

/-- Calculates the number of right-handed players and exclusive throwers -/
def calculate_right_handed_players (team : FootballTeam) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of right-handed players and exclusive throwers -/
theorem right_handed_players_count (team : FootballTeam) 
  (h1 : team.total_players = 180)
  (h2 : team.thrower_percentage = 3/10)
  (h3 : team.kicker_percentage = 9/40)
  (h4 : team.left_handed_remaining_percentage = 3/7)
  (h5 : team.left_handed_kicker_percentage = 1/4)
  (h6 : team.exclusive_thrower_percentage = 3/5) :
  calculate_right_handed_players team = (134, 32) := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l142_14253


namespace NUMINAMATH_CALUDE_smallest_odd_n_for_product_l142_14272

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem smallest_odd_n_for_product : 
  ∃ (n : ℕ), is_odd n ∧ 
    (∀ m : ℕ, is_odd m → m < n → (2 : ℝ)^((m+1)^2/7) ≤ 1000) ∧ 
    (2 : ℝ)^((n+1)^2/7) > 1000 ∧ 
    n = 9 := by sorry

end NUMINAMATH_CALUDE_smallest_odd_n_for_product_l142_14272


namespace NUMINAMATH_CALUDE_distance_between_lakes_l142_14209

/-- The distance between two lakes given bird migration data -/
theorem distance_between_lakes (num_birds : ℕ) (disney_to_london : ℝ) (total_distance : ℝ) :
  num_birds = 20 →
  disney_to_london = 60 →
  total_distance = 2200 →
  ∃ jim_to_disney : ℝ, jim_to_disney = 50 ∧
    num_birds * (jim_to_disney + disney_to_london) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_distance_between_lakes_l142_14209


namespace NUMINAMATH_CALUDE_trig_identity_proof_l142_14208

theorem trig_identity_proof :
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / Real.cos (17 * π / 180) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l142_14208


namespace NUMINAMATH_CALUDE_uncle_james_height_difference_l142_14283

theorem uncle_james_height_difference :
  ∀ (james_initial_height uncle_height james_growth : ℝ),
    james_initial_height = (2/3) * uncle_height →
    uncle_height = 72 →
    james_growth = 10 →
    uncle_height - (james_initial_height + james_growth) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_james_height_difference_l142_14283


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l142_14297

theorem min_value_fraction (x : ℝ) : (x^2 + 8) / Real.sqrt (x^2 + 4) ≥ 4 := by
  sorry

theorem equality_condition : ∃ x : ℝ, (x^2 + 8) / Real.sqrt (x^2 + 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l142_14297


namespace NUMINAMATH_CALUDE_sphere_expansion_l142_14243

theorem sphere_expansion (r₁ r₂ : ℝ) (h : r₁ > 0) :
  (4 / 3 * Real.pi * r₂^3) = 8 * (4 / 3 * Real.pi * r₁^3) →
  (4 * Real.pi * r₂^2) = 4 * (4 * Real.pi * r₁^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_expansion_l142_14243


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l142_14212

/-- The product of the coordinates of the midpoint of a segment with endpoints (8, -4) and (-2, 10) is 9. -/
theorem midpoint_coordinate_product : 
  let x1 : ℝ := 8
  let y1 : ℝ := -4
  let x2 : ℝ := -2
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x * midpoint_y = 9 := by sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l142_14212


namespace NUMINAMATH_CALUDE_largest_angle_cosine_l142_14231

theorem largest_angle_cosine (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) :
  let cos_largest := min (min ((a^2 + b^2 - c^2) / (2*a*b)) ((b^2 + c^2 - a^2) / (2*b*c))) ((c^2 + a^2 - b^2) / (2*c*a))
  cos_largest = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_cosine_l142_14231


namespace NUMINAMATH_CALUDE_existence_of_c_l142_14264

theorem existence_of_c (p r a b : ℤ) : 
  Prime p → 
  p ∣ (r^7 - 1) → 
  p ∣ (r + 1 - a^2) → 
  p ∣ (r^2 + 1 - b^2) → 
  ∃ c : ℤ, p ∣ (r^3 + 1 - c^2) := by
sorry

end NUMINAMATH_CALUDE_existence_of_c_l142_14264


namespace NUMINAMATH_CALUDE_triangle_area_is_eight_l142_14202

/-- A line in 2D space represented by its equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- The triangle formed by the intersection of three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Calculate the area of a triangle given its three lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { l1 := { m := 0, b := 7 },
    l2 := { m := 2, b := 3 },
    l3 := { m := -2, b := 3 } }

theorem triangle_area_is_eight :
  triangleArea problemTriangle = 8 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_eight_l142_14202


namespace NUMINAMATH_CALUDE_eccentricity_range_l142_14294

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
structure PointOnEllipse (C : Ellipse) where
  P : ℝ × ℝ
  on_ellipse : sorry -- Condition for P being on the ellipse C

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (C : Ellipse) : ℝ := sorry

theorem eccentricity_range (C : Ellipse) (P : PointOnEllipse C) 
  (h : distance P.P C.F₁ = 3/2 * distance C.F₁ C.F₂) : 
  1/4 ≤ eccentricity C ∧ eccentricity C ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_eccentricity_range_l142_14294


namespace NUMINAMATH_CALUDE_olivias_phone_pictures_l142_14233

theorem olivias_phone_pictures :
  ∀ (phone_pics camera_pics total_pics albums pics_per_album : ℕ),
    camera_pics = 35 →
    albums = 8 →
    pics_per_album = 5 →
    total_pics = albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    phone_pics = 5 := by
  sorry

end NUMINAMATH_CALUDE_olivias_phone_pictures_l142_14233


namespace NUMINAMATH_CALUDE_transformation_symmetry_l142_14219

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Theorem statement
theorem transformation_symmetry (p : Point2D) :
  symmetricToXAxis p (transform p) := by
  sorry


end NUMINAMATH_CALUDE_transformation_symmetry_l142_14219


namespace NUMINAMATH_CALUDE_existence_equivalence_l142_14254

theorem existence_equivalence : 
  (∃ (x : ℝ), x^2 + 1 < 0) ↔ (∃ (x : ℝ), x^2 + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_equivalence_l142_14254


namespace NUMINAMATH_CALUDE_equilateral_parallelogram_diagonal_l142_14240

/-- A parallelogram composed of four equilateral triangles -/
structure EquilateralParallelogram where
  -- Define the vertices of the parallelogram
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure the parallelogram is made up of four equilateral triangles
  is_equilateral : 
    (dist A B = 2) ∧ 
    (dist B C = 2) ∧ 
    (dist C D = 2) ∧ 
    (dist D A = 2) ∧
    (dist A C = dist B D)
  -- Ensure each equilateral triangle has side length 1
  triangle_side_length : dist A B / 2 = 1

/-- The length of the diagonal in an equilateral parallelogram is √7 -/
theorem equilateral_parallelogram_diagonal 
  (p : EquilateralParallelogram) : dist p.A p.C = Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_equilateral_parallelogram_diagonal_l142_14240


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l142_14246

/-- Represents the number of students in each grade and the total sample size -/
structure SchoolSample where
  total_students : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  sample_size : ℕ

/-- Calculates the number of students to be sampled from each grade -/
def stratifiedSample (school : SchoolSample) : ℕ × ℕ × ℕ :=
  let sample_fraction := school.sample_size / school.total_students
  let first := school.first_grade * sample_fraction
  let second := school.second_grade * sample_fraction
  let third := school.third_grade * sample_fraction
  (first, second, third)

/-- Theorem stating the correct stratified sample for the given school -/
theorem correct_stratified_sample 
  (school : SchoolSample) 
  (h1 : school.total_students = 900)
  (h2 : school.first_grade = 300)
  (h3 : school.second_grade = 200)
  (h4 : school.third_grade = 400)
  (h5 : school.sample_size = 45) :
  stratifiedSample school = (15, 10, 20) := by
  sorry

#eval stratifiedSample { 
  total_students := 900, 
  first_grade := 300, 
  second_grade := 200, 
  third_grade := 400, 
  sample_size := 45 
}

end NUMINAMATH_CALUDE_correct_stratified_sample_l142_14246


namespace NUMINAMATH_CALUDE_amy_candy_difference_l142_14273

/-- Amy's candy distribution problem -/
theorem amy_candy_difference (initial : ℕ) (given_away : ℕ) (left : ℕ) : 
  given_away = 6 → left = 5 → given_away - left = 1 := by
  sorry

end NUMINAMATH_CALUDE_amy_candy_difference_l142_14273


namespace NUMINAMATH_CALUDE_class_average_marks_l142_14263

/-- Calculates the average marks for a class given specific score distributions -/
theorem class_average_marks (total_students : ℕ) 
  (high_score_students : ℕ) (high_score : ℕ) 
  (mid_score_students : ℕ) (mid_score_diff : ℕ)
  (h1 : total_students = 50)
  (h2 : high_score_students = 10)
  (h3 : high_score = 90)
  (h4 : mid_score_students = 15)
  (h5 : mid_score_diff = 10) :
  let low_score_students := total_students - (high_score_students + mid_score_students)
  let low_score := 60
  let total_marks := high_score_students * high_score + 
                     mid_score_students * (high_score - mid_score_diff) + 
                     low_score_students * low_score
  total_marks / total_students = 72 := by
sorry

end NUMINAMATH_CALUDE_class_average_marks_l142_14263


namespace NUMINAMATH_CALUDE_time_addition_theorem_l142_14285

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The initial time (8:00:00 a.m.) -/
def initialTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 7500

/-- The expected final time (10:05:00 a.m.) -/
def expectedFinalTime : Time :=
  { hours := 10, minutes := 5, seconds := 0 }

theorem time_addition_theorem :
  addSeconds initialTime secondsToAdd = expectedFinalTime := by
  sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l142_14285


namespace NUMINAMATH_CALUDE_distribution_scheme_count_l142_14290

/-- The number of ways to choose 2 items from 4 items -/
def choose_4_2 : ℕ := 6

/-- The number of ways to arrange 3 items in 3 positions -/
def arrange_3_3 : ℕ := 6

/-- The number of ways to distribute 4 students into 3 laboratories -/
def distribute_students : ℕ := choose_4_2 * arrange_3_3

theorem distribution_scheme_count :
  distribute_students = 36 :=
by sorry

end NUMINAMATH_CALUDE_distribution_scheme_count_l142_14290


namespace NUMINAMATH_CALUDE_bruce_purchase_l142_14298

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_qty : ℕ) (grape_rate : ℕ) (mango_qty : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_qty * grape_rate + mango_qty * mango_rate

/-- Theorem stating that Bruce paid 1110 for his purchase -/
theorem bruce_purchase : total_amount 8 70 10 55 = 1110 := by
  sorry

end NUMINAMATH_CALUDE_bruce_purchase_l142_14298


namespace NUMINAMATH_CALUDE_eighth_term_ratio_l142_14276

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem eighth_term_ratio
  (a₁ d b₁ e : ℚ)
  (h : ∀ n : ℕ, arithmetic_sum a₁ d n / arithmetic_sum b₁ e n = (5 * n + 6 : ℚ) / (3 * n + 30 : ℚ)) :
  (arithmetic_sequence a₁ d 8) / (arithmetic_sequence b₁ e 8) = 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_eighth_term_ratio_l142_14276


namespace NUMINAMATH_CALUDE_linear_function_proof_l142_14271

/-- A linear function y = kx + 3 passing through (1, -2) with negative slope -/
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

theorem linear_function_proof (k : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → linear_function k x₁ > linear_function k x₂) ∧
  linear_function k 1 = -2 →
  k = -5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l142_14271


namespace NUMINAMATH_CALUDE_binomial_15_12_l142_14206

theorem binomial_15_12 : Nat.choose 15 12 = 455 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_12_l142_14206


namespace NUMINAMATH_CALUDE_area_of_remaining_rectangle_l142_14200

theorem area_of_remaining_rectangle (s : ℝ) (h1 : s = 3) : s^2 - (1 * 3 + 1^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_remaining_rectangle_l142_14200


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l142_14291

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔
  (∀ x : ℝ, (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l142_14291


namespace NUMINAMATH_CALUDE_alpha_plus_beta_l142_14270

theorem alpha_plus_beta (α β : ℝ) :
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 90*x + 1980) / (x^2 + 70*x - 3570)) →
  α + β = 123 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_l142_14270


namespace NUMINAMATH_CALUDE_books_read_in_may_l142_14278

theorem books_read_in_may (may june july total : ℕ) 
  (h1 : june = 6)
  (h2 : july = 10)
  (h3 : total = 18)
  (h4 : may + june + july = total) :
  may = 2 := by
sorry

end NUMINAMATH_CALUDE_books_read_in_may_l142_14278


namespace NUMINAMATH_CALUDE_product_of_invertible_labels_l142_14274

-- Define the function types
inductive FunctionType
| Quadratic
| ScatterPlot
| Sine
| Reciprocal

-- Define the structure for a function
structure Function where
  label : Nat
  type : FunctionType
  invertible : Bool

-- Define the problem setup
def problemSetup : List Function := [
  { label := 2, type := FunctionType.Quadratic, invertible := false },
  { label := 3, type := FunctionType.ScatterPlot, invertible := true },
  { label := 4, type := FunctionType.Sine, invertible := true },
  { label := 5, type := FunctionType.Reciprocal, invertible := true }
]

-- Theorem statement
theorem product_of_invertible_labels :
  (problemSetup.filter (λ f => f.invertible)).foldl (λ acc f => acc * f.label) 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_of_invertible_labels_l142_14274


namespace NUMINAMATH_CALUDE_original_number_l142_14224

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 117 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l142_14224


namespace NUMINAMATH_CALUDE_system_solution_l142_14289

theorem system_solution :
  ∃ (x y : ℝ), 
    (x + 2*y = (7 - x) + (3 - 2*y)) ∧ 
    (x - 3*y = (x + 2) - (y - 2)) ∧ 
    x = 9 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l142_14289


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l142_14286

theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_ratio : ℝ)
  (fair_hair_ratio : ℝ)
  (h1 : women_fair_hair_ratio = 0.1)
  (h2 : fair_hair_ratio = 0.25) :
  women_fair_hair_ratio / fair_hair_ratio = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l142_14286


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l142_14234

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n = 85 ∧ 
  (∀ m : ℕ, m ≤ 99 ∧ m ≥ 10 ∧ 17 ∣ m → m ≤ n) ∧ 
  17 ∣ n ∧ n ≤ 99 ∧ n ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l142_14234


namespace NUMINAMATH_CALUDE_cube_surface_area_7cm_l142_14207

def cube_surface_area (edge_length : ℝ) : ℝ := 6 * edge_length * edge_length

theorem cube_surface_area_7cm :
  cube_surface_area 7 = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_7cm_l142_14207


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l142_14268

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l142_14268


namespace NUMINAMATH_CALUDE_prob_even_sum_is_8_15_l142_14266

def wheel1 : Finset ℕ := {1, 2, 3, 4, 5}
def wheel2 : Finset ℕ := {1, 2, 3}

def isEven (n : ℕ) : Bool := n % 2 = 0

def probEvenSum : ℚ :=
  (Finset.filter (fun (pair : ℕ × ℕ) => isEven (pair.1 + pair.2)) (wheel1.product wheel2)).card /
  (wheel1.card * wheel2.card : ℚ)

theorem prob_even_sum_is_8_15 : probEvenSum = 8 / 15 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_8_15_l142_14266


namespace NUMINAMATH_CALUDE_green_fish_count_l142_14260

theorem green_fish_count (total : ℕ) (blue : ℕ) (orange : ℕ) (green : ℕ) : 
  total = 80 →
  blue = total / 2 →
  orange = blue - 15 →
  total = blue + orange + green →
  green = 15 := by
  sorry

end NUMINAMATH_CALUDE_green_fish_count_l142_14260


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l142_14242

theorem tan_sum_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 :=
by
  have h : Real.tan (45 * π / 180) = (Real.tan (10 * π / 180) + Real.tan (35 * π / 180)) /
    (1 - Real.tan (10 * π / 180) * Real.tan (35 * π / 180)) := by sorry
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l142_14242


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l142_14228

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l142_14228
