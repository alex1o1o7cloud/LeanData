import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_l311_31104

theorem sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l311_31104


namespace NUMINAMATH_CALUDE_isosceles_base_length_l311_31187

/-- Represents a triangle with sides a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

/-- An equilateral triangle is a triangle with all sides equal. -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- An isosceles triangle is a triangle with at least two sides equal. -/
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- The perimeter of a triangle is the sum of its sides. -/
def Perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

theorem isosceles_base_length
  (equi : Triangle)
  (iso : Triangle)
  (h_equi_equilateral : IsEquilateral equi)
  (h_iso_isosceles : IsIsosceles iso)
  (h_equi_perimeter : Perimeter equi = 60)
  (h_iso_perimeter : Perimeter iso = 70)
  (h_shared_side : equi.a = iso.a) :
  iso.c = 30 :=
sorry

end NUMINAMATH_CALUDE_isosceles_base_length_l311_31187


namespace NUMINAMATH_CALUDE_min_sum_squares_l311_31176

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l311_31176


namespace NUMINAMATH_CALUDE_total_pay_calculation_l311_31136

/-- Calculates the total pay for a worker given regular and overtime hours --/
def total_pay (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) : ℝ :=
  let overtime_rate := 2 * regular_rate
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that under the given conditions, the total pay is $192 --/
theorem total_pay_calculation :
  let regular_rate := 3
  let regular_hours := 40
  let overtime_hours := 12
  total_pay regular_rate regular_hours overtime_hours = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_pay_calculation_l311_31136


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l311_31178

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧  -- f is an odd function
  (∀ x y : ℝ, 0 < x → x < y → f x < f y)  -- f is monotonically increasing on (0, +∞)
  := by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l311_31178


namespace NUMINAMATH_CALUDE_lune_area_l311_31119

/-- The area of a lune formed by two semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h : r₁ = 2 * r₂) : 
  let lune_area := π * r₂^2 / 2 + r₁ * r₂ - π * r₁^2 / 4
  lune_area = 1 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_l311_31119


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l311_31113

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l311_31113


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l311_31153

theorem sum_of_squares_zero_implies_sum (a b c d : ℝ) :
  (a - 2)^2 + (b - 5)^2 + (c - 6)^2 + (d - 3)^2 = 0 →
  a + b + c + d = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_implies_sum_l311_31153


namespace NUMINAMATH_CALUDE_land_properties_l311_31144

/-- Represents a piece of land with specific measurements -/
structure Land where
  triangle_area : ℝ
  ac_length : ℝ
  cd_length : ℝ
  de_length : ℝ

/-- Calculates the total area of the land -/
def total_area (land : Land) : ℝ := sorry

/-- Calculates the length of CF to divide the land equally -/
def equal_division_length (land : Land) : ℝ := sorry

theorem land_properties (land : Land) 
  (h1 : land.triangle_area = 120)
  (h2 : land.ac_length = 20)
  (h3 : land.cd_length = 10)
  (h4 : land.de_length = 10) :
  total_area land = 270 ∧ equal_division_length land = 1.5 := by sorry

end NUMINAMATH_CALUDE_land_properties_l311_31144


namespace NUMINAMATH_CALUDE_square_difference_equals_736_l311_31149

theorem square_difference_equals_736 : (23 + 16)^2 - (23^2 + 16^2) = 736 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_736_l311_31149


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l311_31121

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 120 → books_sold = 39 → books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 := by
sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l311_31121


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l311_31171

theorem quadratic_equation_roots (k : ℝ) : 
  let eq := fun x : ℝ => x^2 + (2*k - 1)*x + k^2 - 1
  ∃ x₁ x₂ : ℝ, 
    (eq x₁ = 0 ∧ eq x₂ = 0) ∧ 
    (x₁ ≠ x₂) ∧
    (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l311_31171


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l311_31191

theorem largest_integer_less_than_100_with_remainder_5_mod_8 :
  ∃ (n : ℕ), n < 100 ∧ n % 8 = 5 ∧ ∀ (m : ℕ), m < 100 → m % 8 = 5 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_with_remainder_5_mod_8_l311_31191


namespace NUMINAMATH_CALUDE_billys_age_l311_31155

/-- Given that the sum of Billy's and Joe's ages is 60 and Billy is three times as old as Joe,
    prove that Billy is 45 years old. -/
theorem billys_age (billy joe : ℕ) 
    (sum_condition : billy + joe = 60)
    (age_ratio : billy = 3 * joe) : 
  billy = 45 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l311_31155


namespace NUMINAMATH_CALUDE_shirt_price_is_16_30_l311_31193

/-- Calculates the final price of a shirt given the cost price, profit percentage, discount percentage, tax rate, and packaging fee. -/
def final_shirt_price (cost_price : ℝ) (profit_percentage : ℝ) (discount_percentage : ℝ) (tax_rate : ℝ) (packaging_fee : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + packaging_fee

/-- Theorem stating that the final price of the shirt is $16.30 given the specific conditions. -/
theorem shirt_price_is_16_30 :
  final_shirt_price 20 0.30 0.50 0.10 2 = 16.30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_is_16_30_l311_31193


namespace NUMINAMATH_CALUDE_eleven_divides_four_digit_palindromes_l311_31148

/-- A four-digit palindrome is a number of the form abba where a and b are digits. -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem eleven_divides_four_digit_palindromes :
  ∀ n : ℕ, FourDigitPalindrome n → 11 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_eleven_divides_four_digit_palindromes_l311_31148


namespace NUMINAMATH_CALUDE_ladder_slide_l311_31143

theorem ladder_slide (ladder_length : ℝ) (initial_base : ℝ) (slip_distance : ℝ) :
  ladder_length = 25 →
  initial_base = 7 →
  slip_distance = 4 →
  ∃ (slide_distance : ℝ),
    slide_distance = 8 ∧
    (ladder_length ^ 2 = (initial_base + slide_distance) ^ 2 + (Real.sqrt (ladder_length ^ 2 - initial_base ^ 2) - slip_distance) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_ladder_slide_l311_31143


namespace NUMINAMATH_CALUDE_inequality_range_l311_31132

theorem inequality_range (a : ℝ) : 
  (∀ (x θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≥ 7/2 ∨ a ≤ Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l311_31132


namespace NUMINAMATH_CALUDE_second_movie_length_second_movie_is_one_and_half_hours_l311_31150

/-- Calculates the length of the second movie given Henri's schedule --/
theorem second_movie_length 
  (total_time : ℝ) 
  (first_movie : ℝ) 
  (reading_rate : ℝ) 
  (words_read : ℝ) : ℝ :=
  let reading_time : ℝ := words_read / (reading_rate * 60)
  let second_movie : ℝ := total_time - first_movie - reading_time
  second_movie

/-- Proves that the length of the second movie is 1.5 hours --/
theorem second_movie_is_one_and_half_hours :
  second_movie_length 8 3.5 10 1800 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_second_movie_length_second_movie_is_one_and_half_hours_l311_31150


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_three_l311_31130

theorem unique_solution_implies_a_equals_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 9 = 0) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_three_l311_31130


namespace NUMINAMATH_CALUDE_square_nonnegative_l311_31163

theorem square_nonnegative (a : ℝ) : a^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_nonnegative_l311_31163


namespace NUMINAMATH_CALUDE_intersection_singleton_l311_31158

/-- The set A defined by the equation y = ax + 1 -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * p.1 + 1}

/-- The set B defined by the equation y = |x| -/
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = |p.1|}

/-- Theorem stating the condition for A ∩ B to be a singleton set -/
theorem intersection_singleton (a : ℝ) : (A a ∩ B).Finite ∧ (A a ∩ B).Nonempty ↔ a ≥ 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_l311_31158


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l311_31124

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem subset_implies_a_equals_one (a : ℝ) (h1 : B a ⊆ A) (h2 : a > 0) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l311_31124


namespace NUMINAMATH_CALUDE_mrs_white_carrot_yield_l311_31195

/-- Calculates the expected carrot yield from a rectangular garden --/
def expected_carrot_yield (length_steps : ℕ) (width_steps : ℕ) (step_size : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_size * (width_steps : ℚ) * step_size * yield_per_sqft

/-- Proves that the expected carrot yield for Mrs. White's garden is 1875 pounds --/
theorem mrs_white_carrot_yield : 
  expected_carrot_yield 18 25 (5/2) (2/3) = 1875 := by
  sorry

end NUMINAMATH_CALUDE_mrs_white_carrot_yield_l311_31195


namespace NUMINAMATH_CALUDE_point_not_in_transformed_plane_l311_31102

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies the equation of a plane -/
def pointSatisfiesPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := -1, y := 1, z := -2 }
  let a : Plane3D := { a := 4, b := -1, c := 3, d := -6 }
  let k : ℝ := -5/3
  let transformedPlane := transformPlane a k
  ¬ pointSatisfiesPlane A transformedPlane :=
by
  sorry


end NUMINAMATH_CALUDE_point_not_in_transformed_plane_l311_31102


namespace NUMINAMATH_CALUDE_perimeter_difference_rectangle_and_squares_l311_31169

/-- The perimeter of a rectangle with given length and width -/
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * (length + width)

/-- The perimeter of a single unit square -/
def unit_square_perimeter : ℕ := 4

/-- The perimeter of n non-overlapping unit squares arranged in a straight line -/
def n_unit_squares_perimeter (n : ℕ) : ℕ := n * unit_square_perimeter

theorem perimeter_difference_rectangle_and_squares : 
  (rectangle_perimeter 6 1) - (n_unit_squares_perimeter 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_rectangle_and_squares_l311_31169


namespace NUMINAMATH_CALUDE_chandler_skateboard_savings_l311_31105

/-- Calculates the minimum number of full weeks required to save for a skateboard -/
def min_weeks_to_save (skateboard_cost : ℕ) (gift_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  ((skateboard_cost - gift_money + weekly_earnings - 1) / weekly_earnings : ℕ)

theorem chandler_skateboard_savings :
  min_weeks_to_save 550 130 18 = 24 := by
  sorry

end NUMINAMATH_CALUDE_chandler_skateboard_savings_l311_31105


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l311_31166

theorem lcm_from_product_and_hcf (a b : ℕ+) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l311_31166


namespace NUMINAMATH_CALUDE_sandy_work_hours_l311_31162

/-- Sandy's work problem -/
theorem sandy_work_hours (hourly_rate : ℚ) (friday_hours : ℚ) (saturday_hours : ℚ) (total_earnings : ℚ) :
  hourly_rate = 15 →
  friday_hours = 10 →
  saturday_hours = 6 →
  total_earnings = 450 →
  (total_earnings - (friday_hours + saturday_hours) * hourly_rate) / hourly_rate = 14 :=
by sorry

end NUMINAMATH_CALUDE_sandy_work_hours_l311_31162


namespace NUMINAMATH_CALUDE_only_one_milk_chocolate_affordable_l311_31114

-- Define the prices of chocolates
def dark_chocolate_price : ℚ := 5
def milk_chocolate_price : ℚ := 9/2
def white_chocolate_price : ℚ := 6

-- Define the sales tax rate
def sales_tax_rate : ℚ := 7/100

-- Define Leonardo's budget
def leonardo_budget : ℚ := 459/100

-- Function to calculate price with tax
def price_with_tax (price : ℚ) : ℚ := price * (1 + sales_tax_rate)

-- Theorem statement
theorem only_one_milk_chocolate_affordable :
  (price_with_tax dark_chocolate_price > leonardo_budget) ∧
  (price_with_tax white_chocolate_price > leonardo_budget) ∧
  (price_with_tax milk_chocolate_price ≤ leonardo_budget) ∧
  (2 * price_with_tax milk_chocolate_price > leonardo_budget) :=
by sorry

end NUMINAMATH_CALUDE_only_one_milk_chocolate_affordable_l311_31114


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l311_31151

theorem gcd_lcm_sum : Nat.gcd 15 45 + Nat.lcm 15 30 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l311_31151


namespace NUMINAMATH_CALUDE_horner_method_proof_l311_31170

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℤ) (x : ℤ) : ℤ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^7 + 2x^5 + 4x^3 + x -/
def f_coeffs : List ℤ := [3, 0, 2, 0, 4, 0, 1, 0]

theorem horner_method_proof :
  horner f_coeffs 3 = 7158 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l311_31170


namespace NUMINAMATH_CALUDE_equation_equality_l311_31183

theorem equation_equality (x : ℝ) : 
  4 * x^4 + x^3 - 2*x + 5 + (-4 * x^4 + x^3 - 7 * x^2 + 2*x - 1) = 2 * x^3 - 7 * x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l311_31183


namespace NUMINAMATH_CALUDE_inequality_proof_l311_31142

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l311_31142


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l311_31194

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 13*x + 30 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 5*x - 50 = (x + b)*(x - c)) →
  a + b + c = 18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l311_31194


namespace NUMINAMATH_CALUDE_discount_comparison_l311_31175

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.10]
def option2_discounts : List ℝ := [0.30, 0.10, 0.10]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem discount_comparison :
  apply_successive_discounts initial_amount option1_discounts -
  apply_successive_discounts initial_amount option2_discounts = 135 :=
by sorry

end NUMINAMATH_CALUDE_discount_comparison_l311_31175


namespace NUMINAMATH_CALUDE_problem_statement_l311_31122

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
  (¬(∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x - Real.log x ≤ 0)) ∧
  (∀ p q : Prop, (p ∨ q → p ∧ q) → False) ∧
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l311_31122


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l311_31108

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → (x < -1 ∨ x > 1)) ∧
  ¬(∀ x : ℝ, (x < -1 ∨ x > 1) → x < -1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l311_31108


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l311_31115

theorem quadratic_root_relation (b c : ℚ) : 
  (∃ r s : ℚ, 5 * r^2 - 8 * r + 2 = 0 ∧ 5 * s^2 - 8 * s + 2 = 0 ∧
   (r - 3)^2 + b * (r - 3) + c = 0 ∧ (s - 3)^2 + b * (s - 3) + c = 0) →
  c = 23/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l311_31115


namespace NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l311_31186

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRandomVariable) (a b : ℝ) : ℝ := sorry

theorem normal_distribution_probability_theorem (ξ : NormalRandomVariable) 
  (h1 : prob_less ξ (-3) = 0.2)
  (h2 : prob_greater ξ 1 = 0.2) :
  prob_between ξ (-1) 1 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_theorem_l311_31186


namespace NUMINAMATH_CALUDE_percentage_relation_l311_31189

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A = 0.06 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l311_31189


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt_5_minus_3_l311_31188

theorem quadratic_root_sqrt_5_minus_3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
  (a = 1 ∧ b = 6 ∧ c = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt_5_minus_3_l311_31188


namespace NUMINAMATH_CALUDE_prop1_false_prop2_true_prop3_true_prop4_true_l311_31198

-- Define the basic geometric objects
variable (Line Plane : Type)

-- Define the geometric relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)
variable (line_of_intersection : Plane → Plane → Line)

-- Proposition 1 (false)
theorem prop1_false :
  ¬(∀ (p1 p2 p3 : Plane) (l1 l2 : Line),
    line_in_plane l1 p1 → line_in_plane l2 p1 →
    line_parallel_to_plane l1 p2 → line_parallel_to_plane l2 p2 →
    parallel p1 p2) := sorry

-- Proposition 2 (true)
theorem prop2_true :
  ∀ (p1 p2 : Plane) (l : Line),
    line_perpendicular_to_plane l p1 →
    line_in_plane l p2 →
    perpendicular p1 p2 := sorry

-- Proposition 3 (true)
theorem prop3_true :
  ∀ (l1 l2 l3 : Line),
    lines_perpendicular l1 l3 →
    lines_perpendicular l2 l3 →
    lines_perpendicular l1 l2 := sorry

-- Proposition 4 (true)
theorem prop4_true :
  ∀ (p1 p2 : Plane) (l : Line),
    perpendicular p1 p2 →
    line_in_plane l p1 →
    ¬(lines_perpendicular l (line_of_intersection p1 p2)) →
    ¬(line_perpendicular_to_plane l p2) := sorry

end NUMINAMATH_CALUDE_prop1_false_prop2_true_prop3_true_prop4_true_l311_31198


namespace NUMINAMATH_CALUDE_peters_erasers_l311_31192

/-- Peter's erasers problem -/
theorem peters_erasers (initial_erasers additional_erasers : ℕ) :
  initial_erasers = 8 →
  additional_erasers = 3 →
  initial_erasers + additional_erasers = 11 := by
  sorry

end NUMINAMATH_CALUDE_peters_erasers_l311_31192


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l311_31127

/-- The minimum distance from any point on the curve y = e^x to the line y = x - 1 is √2 -/
theorem min_distance_exp_curve_to_line : 
  ∀ (x₀ y₀ : ℝ), y₀ = Real.exp x₀ → 
  (∃ (d : ℝ), d = |y₀ - (x₀ - 1)| / Real.sqrt 2 ∧ 
   ∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) → 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  (∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l311_31127


namespace NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l311_31184

theorem quadratic_equation_and_inequality :
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x - m ≠ 0) ∧
  (∀ x : ℝ, x^2 + x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_and_inequality_l311_31184


namespace NUMINAMATH_CALUDE_roberts_chocolates_l311_31100

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : 
  nickel_chocolates = 3 → difference = 9 → nickel_chocolates + difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_roberts_chocolates_l311_31100


namespace NUMINAMATH_CALUDE_cubic_symmetry_about_origin_l311_31172

def f (x : ℝ) : ℝ := x^3

theorem cubic_symmetry_about_origin :
  ∀ x : ℝ, f (-x) = -f x :=
by sorry

end NUMINAMATH_CALUDE_cubic_symmetry_about_origin_l311_31172


namespace NUMINAMATH_CALUDE_brothers_snowballs_l311_31157

theorem brothers_snowballs (janet_snowballs : ℕ) (janet_percentage : ℚ) : 
  janet_snowballs = 50 → janet_percentage = 1/4 → 
  (1 - janet_percentage) * (janet_snowballs / janet_percentage) = 150 := by
  sorry

end NUMINAMATH_CALUDE_brothers_snowballs_l311_31157


namespace NUMINAMATH_CALUDE_envelope_touches_all_C_a_l311_31106

/-- The curve C_a is defined by the equation (y - a^2)^2 = x^2(a^2 - x^2) for a > 0 -/
def C_a (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ (y - a^2)^2 = x^2 * (a^2 - x^2)

/-- The envelope curve -/
def envelope_curve (x y : ℝ) : Prop :=
  y = (3 * x^2) / 4

/-- Theorem stating that the envelope curve touches all C_a curves -/
theorem envelope_touches_all_C_a :
  ∀ (a x y : ℝ), C_a a x y → ∃ (x₀ y₀ : ℝ), 
    envelope_curve x₀ y₀ ∧ 
    C_a a x₀ y₀ ∧
    (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x' y' : ℝ), 
        ((x' - x₀)^2 + (y' - y₀)^2 < δ^2) →
        (envelope_curve x' y' → ¬C_a a x' y') ∧
        (C_a a x' y' → ¬envelope_curve x' y')) :=
by sorry

end NUMINAMATH_CALUDE_envelope_touches_all_C_a_l311_31106


namespace NUMINAMATH_CALUDE_difference_of_squares_l311_31128

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_l311_31128


namespace NUMINAMATH_CALUDE_geometry_propositions_l311_31182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (¬ ∀ (m n : Line) (α β : Plane), 
    parallel_line_plane m α → 
    parallel_line_plane n β → 
    parallel_plane_plane α β → 
    parallel_line_line m n) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    perpendicular_line_plane m α → 
    perpendicular_line_plane n β → 
    perpendicular_plane_plane α β → 
    perpendicular_line_line m n) ∧ 
  (¬ ∀ (m n : Line) (α : Plane), 
    parallel_line_plane m α → 
    parallel_line_line m n → 
    parallel_line_plane n α) ∧ 
  (∀ (m n : Line) (α β : Plane), 
    parallel_plane_plane α β → 
    perpendicular_line_plane m α → 
    parallel_line_plane n β → 
    perpendicular_line_line m n) := by
  sorry

end NUMINAMATH_CALUDE_geometry_propositions_l311_31182


namespace NUMINAMATH_CALUDE_plant_supplier_earnings_l311_31168

theorem plant_supplier_earnings :
  let orchid_count : ℕ := 20
  let orchid_price : ℕ := 50
  let money_plant_count : ℕ := 15
  let money_plant_price : ℕ := 25
  let worker_count : ℕ := 2
  let worker_pay : ℕ := 40
  let pot_cost : ℕ := 150
  let total_earnings := orchid_count * orchid_price + money_plant_count * money_plant_price
  let total_expenses := worker_count * worker_pay + pot_cost
  total_earnings - total_expenses = 1145 :=
by
  sorry

#check plant_supplier_earnings

end NUMINAMATH_CALUDE_plant_supplier_earnings_l311_31168


namespace NUMINAMATH_CALUDE_distance_equals_radius_l311_31125

/-- A circle resting on the x-axis and tangent to the line x=3 -/
structure TangentCircle where
  /-- The x-coordinate of the circle's center -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle rests on the x-axis and is tangent to x=3 -/
  tangent_condition : r = |3 - h|

/-- The distance from the center to the point of tangency equals the radius -/
theorem distance_equals_radius (c : TangentCircle) :
  |3 - c.h| = c.r := by sorry

end NUMINAMATH_CALUDE_distance_equals_radius_l311_31125


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l311_31140

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem complex_fraction_simplification :
  (1 + i) / (1 - i) = i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l311_31140


namespace NUMINAMATH_CALUDE_sequence_bounded_above_l311_31145

/-- The sequence {aₙ} defined by the given recurrence relation is bounded above. -/
theorem sequence_bounded_above (α : ℝ) (h_α : α > 1) :
  ∃ (M : ℝ), ∀ (a : ℕ → ℝ), 
    (a 1 ∈ Set.Ioo 0 1) → 
    (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + (a n / n)^α) → 
    (∀ n : ℕ, n ≥ 1 → a n ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_sequence_bounded_above_l311_31145


namespace NUMINAMATH_CALUDE_evaluate_expression_l311_31141

theorem evaluate_expression : -(18 / 3 * 8 - 80 + 4^2 * 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l311_31141


namespace NUMINAMATH_CALUDE_max_regions_correct_l311_31134

/-- The maximum number of regions into which n circles can divide the plane -/
def max_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_regions gives the maximum number of regions -/
theorem max_regions_correct (n : ℕ) :
  max_regions n = n^2 - n + 2 :=
by sorry

end NUMINAMATH_CALUDE_max_regions_correct_l311_31134


namespace NUMINAMATH_CALUDE_range_of_m_l311_31129

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 * x^2 + 2*m*x - 4 < 2*x^2 + 4*x) → 
  -2 < m ∧ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l311_31129


namespace NUMINAMATH_CALUDE_sphere_only_identical_views_l311_31174

-- Define the set of common 3D solids
inductive Solid
  | Sphere
  | Cylinder
  | TriangularPrism
  | Cone

-- Define a function to check if all views are identical
def allViewsIdentical (s : Solid) : Prop :=
  match s with
  | Solid.Sphere => True
  | _ => False

-- Theorem statement
theorem sphere_only_identical_views :
  ∀ s : Solid, allViewsIdentical s ↔ s = Solid.Sphere :=
sorry

end NUMINAMATH_CALUDE_sphere_only_identical_views_l311_31174


namespace NUMINAMATH_CALUDE_seashells_given_theorem_l311_31161

/-- The number of seashells Sam gave to Joan -/
def seashells_given_to_joan (initial_seashells current_seashells : ℕ) : ℕ :=
  initial_seashells - current_seashells

/-- Theorem stating that the number of seashells Sam gave to Joan
    is the difference between his initial and current number of seashells -/
theorem seashells_given_theorem (initial_seashells current_seashells : ℕ) 
  (h : initial_seashells ≥ current_seashells) :
  seashells_given_to_joan initial_seashells current_seashells = 
  initial_seashells - current_seashells :=
by
  sorry

#eval seashells_given_to_joan 35 17  -- Should output 18

end NUMINAMATH_CALUDE_seashells_given_theorem_l311_31161


namespace NUMINAMATH_CALUDE_triangular_array_sum_l311_31165

/-- Represents the sum of numbers in the nth row of the triangular array -/
def f (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 1.5 * f (n - 1)

/-- The triangular array property -/
theorem triangular_array_sum : f 10 = 38.443359375 := by
  sorry

end NUMINAMATH_CALUDE_triangular_array_sum_l311_31165


namespace NUMINAMATH_CALUDE_alex_age_l311_31123

theorem alex_age (inez_age : ℕ) (zack_age : ℕ) (jose_age : ℕ) (alex_age : ℕ)
  (h1 : inez_age = 18)
  (h2 : zack_age = inez_age + 5)
  (h3 : jose_age = zack_age - 6)
  (h4 : alex_age = jose_age - 2) :
  alex_age = 15 := by
  sorry

end NUMINAMATH_CALUDE_alex_age_l311_31123


namespace NUMINAMATH_CALUDE_total_seashells_l311_31135

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def seashells_day3 (x y : ℕ) : ℕ := 2 * (x + y)

theorem total_seashells : 
  seashells_day1 + seashells_day2 + seashells_day3 seashells_day1 seashells_day2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l311_31135


namespace NUMINAMATH_CALUDE_g_of_5_l311_31103

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_of_5 : g 5 = 74 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l311_31103


namespace NUMINAMATH_CALUDE_min_value_3x_plus_4y_l311_31199

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y = 5*x*y) :
  3*x + 4*y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 5*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_plus_4y_l311_31199


namespace NUMINAMATH_CALUDE_product_of_17_terms_geometric_sequence_l311_31117

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the product of the first n terms of a sequence
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * a (i + 1)) 1

-- Theorem statement
theorem product_of_17_terms_geometric_sequence 
  (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_a9 : a 9 = -2) :
  product_of_first_n_terms a 17 = -2^17 := by
  sorry

end NUMINAMATH_CALUDE_product_of_17_terms_geometric_sequence_l311_31117


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l311_31160

theorem restaurant_bill_calculation :
  let num_bankers : ℕ := 4
  let num_clients : ℕ := 5
  let total_people : ℕ := num_bankers + num_clients
  let cost_per_person : ℚ := 70
  let gratuity_rate : ℚ := 0.20
  let pre_gratuity_total : ℚ := total_people * cost_per_person
  let gratuity_amount : ℚ := pre_gratuity_total * gratuity_rate
  let total_bill : ℚ := pre_gratuity_total + gratuity_amount
  total_bill = 756 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l311_31160


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l311_31173

/-- The probability of drawing two white balls from a box with white and black balls -/
theorem two_white_balls_probability
  (total_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 7)
  (h3 : black_balls = 8) :
  (white_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l311_31173


namespace NUMINAMATH_CALUDE_max_value_of_f_l311_31159

noncomputable def f (x a : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + (5/8) * a + 1

theorem max_value_of_f (a : ℝ) :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ Real.pi / 2 → f y a ≤ f x a) →
  (a < 0 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (5/8) * a + 2) ∧
  (0 ≤ a ∧ a ≤ 2 → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = a^2 / 4 + (5/8) * a + 2) ∧
  (2 < a → ∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x a = (13/8) * a + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l311_31159


namespace NUMINAMATH_CALUDE_combination_sum_equals_462_l311_31120

theorem combination_sum_equals_462 : 
  (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_equals_462_l311_31120


namespace NUMINAMATH_CALUDE_remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l311_31126

/-- The area of a square with side length 6, minus specific triangular cutouts, equals 27 -/
theorem remaining_area_after_cutouts (square_side : ℝ) (cutout_area : ℝ) : 
  square_side = 6 → 
  cutout_area = 9 → 
  square_side^2 - cutout_area = 27 := by
  sorry

/-- The area of triangular cutouts in a 6x6 square equals 9 -/
theorem cutout_area_is_nine (dark_gray_rect_area light_gray_rect_area : ℝ) :
  dark_gray_rect_area = 3 →
  light_gray_rect_area = 6 →
  dark_gray_rect_area + light_gray_rect_area = 9 := by
  sorry

/-- The area of a rectangle formed by dark gray triangles is 3 -/
theorem dark_gray_rectangle_area (length width : ℝ) :
  length = 1 →
  width = 3 →
  length * width = 3 := by
  sorry

/-- The area of a rectangle formed by light gray triangles is 6 -/
theorem light_gray_rectangle_area (length width : ℝ) :
  length = 2 →
  width = 3 →
  length * width = 6 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l311_31126


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l311_31137

theorem same_terminal_side_angles (π : ℝ) : 
  {β : ℝ | ∃ k : ℤ, β = π / 3 + 2 * k * π ∧ -2 * π ≤ β ∧ β < 4 * π} = 
  {-5 * π / 3, π / 3, 7 * π / 3} := by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l311_31137


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l311_31146

/-- The hyperbola struct represents a hyperbola with semi-major axis a and semi-minor axis b. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The Point struct represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The Focus struct represents a focus point of the hyperbola. -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Represents the condition that P is on the right branch of the hyperbola. -/
def is_on_right_branch (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1 ∧ p.x > 0

/-- Represents the condition that the distance from O to PF₁ equals the real semi-axis. -/
def distance_condition (h : Hyperbola) (p : Point) (f₁ : Focus) : Prop :=
  ∃ (d : ℝ), d = h.a ∧ d = abs (f₁.y * p.x - f₁.x * p.y) / Real.sqrt ((p.x - f₁.x)^2 + (p.y - f₁.y)^2)

/-- The main theorem stating the eccentricity of the hyperbola under given conditions. -/
theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) (f₁ f₂ : Focus) :
  is_on_right_branch h p →
  (p.x - f₂.x)^2 + (p.y - f₂.y)^2 = (f₁.x - f₂.x)^2 + (f₁.y - f₂.y)^2 →
  distance_condition h p f₁ →
  let e := Real.sqrt (1 + h.b^2 / h.a^2)
  e = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l311_31146


namespace NUMINAMATH_CALUDE_calculation_proof_l311_31185

theorem calculation_proof :
  (9.5 * 101 = 959.5) ∧
  (12.5 * 8.8 = 110) ∧
  (38.4 * 187 - 15.4 * 384 + 3.3 * 16 = 1320) ∧
  (5.29 * 73 + 52.9 * 2.7 = 529) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l311_31185


namespace NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l311_31179

theorem special_divisor_property_implies_prime (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d : ℕ, d > 0 → d ∣ n → (d + 1) ∣ (n + 1)) :
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_special_divisor_property_implies_prime_l311_31179


namespace NUMINAMATH_CALUDE_plant_purchase_cost_l311_31112

/-- Calculates the actual amount spent on plants given the original cost and discount. -/
def actualCost (originalCost discount : ℚ) : ℚ :=
  originalCost - discount

/-- Theorem stating that given the specific original cost and discount, the actual amount spent is $68.00. -/
theorem plant_purchase_cost :
  let originalCost : ℚ := 467
  let discount : ℚ := 399
  actualCost originalCost discount = 68 := by
sorry

end NUMINAMATH_CALUDE_plant_purchase_cost_l311_31112


namespace NUMINAMATH_CALUDE_integral_exp_plus_x_l311_31156

theorem integral_exp_plus_x : ∫ x in (0 : ℝ)..(1 : ℝ), (Real.exp x + x) = Real.exp 1 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_exp_plus_x_l311_31156


namespace NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l311_31139

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices or edge length here

/-- An octahedron formed by the midpoints of a tetrahedron's edges -/
def midpoint_octahedron (t : RegularTetrahedron) : Set (Fin 4 → ℝ) :=
  sorry

/-- The volume of a regular tetrahedron -/
def volume_tetrahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- The volume of the octahedron formed by midpoints -/
def volume_midpoint_octahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- Theorem: The ratio of the volume of the midpoint octahedron to the volume of the regular tetrahedron is 3/16 -/
theorem midpoint_octahedron_volume_ratio (t : RegularTetrahedron) :
  volume_midpoint_octahedron t / volume_tetrahedron t = 3 / 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_octahedron_volume_ratio_l311_31139


namespace NUMINAMATH_CALUDE_number_line_problem_l311_31190

theorem number_line_problem (a b c : ℚ) : 
  a = (-4)^2 - 8 →
  b = -c →
  |c - a| = 3 →
  ((b = -5 ∧ c = 5) ∨ (b = -11 ∧ c = 11)) ∧
  (-a^2 + b - c = -74 ∨ -a^2 + b - c = -86) :=
by sorry

end NUMINAMATH_CALUDE_number_line_problem_l311_31190


namespace NUMINAMATH_CALUDE_cody_candy_count_l311_31177

/-- The number of boxes of chocolate candy Cody bought -/
def chocolate_boxes : ℕ := 7

/-- The number of boxes of caramel candy Cody bought -/
def caramel_boxes : ℕ := 3

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 8

/-- The total number of candy pieces Cody has -/
def total_candy : ℕ := (chocolate_boxes + caramel_boxes) * pieces_per_box

theorem cody_candy_count : total_candy = 80 := by
  sorry

end NUMINAMATH_CALUDE_cody_candy_count_l311_31177


namespace NUMINAMATH_CALUDE_polygon_sides_count_l311_31167

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The theorem stating that a polygon satisfying the given condition has 9 sides -/
theorem polygon_sides_count : 
  ∃ (n : ℕ), n > 0 ∧ 
  (num_diagonals (2*n) - 2*n) - (num_diagonals n - n) = 99 ∧ 
  n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l311_31167


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l311_31109

theorem fence_cost_per_foot
  (area : ℝ)
  (total_cost : ℝ)
  (h_area : area = 289)
  (h_total_cost : total_cost = 3944) :
  total_cost / (4 * Real.sqrt area) = 58 :=
by sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l311_31109


namespace NUMINAMATH_CALUDE_trees_planted_correct_l311_31138

/-- The number of maple trees planted in a park --/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts --/
theorem trees_planted_correct (initial final : ℕ) (h : final ≥ initial) :
  trees_planted initial final = final - initial :=
by sorry

end NUMINAMATH_CALUDE_trees_planted_correct_l311_31138


namespace NUMINAMATH_CALUDE_total_nails_and_claws_is_524_l311_31116

/-- The total number of nails and claws Cassie needs to cut -/
def total_nails_and_claws : ℕ :=
  -- Dogs
  4 * 4 * 4 +
  -- Parrots
  (7 * 2 * 3 + 1 * 2 * 4 + 1 * 2 * 2) +
  -- Cats
  (1 * 2 * 5 + 1 * 2 * 4 + 1) +
  -- Rabbits
  (5 * 4 * 9 + 3 * 9 + 2) +
  -- Lizards
  (4 * 4 * 5 + 1 * 4 * 4) +
  -- Tortoises
  (2 * 4 * 4 + 3 * 4 + 5 + 3 * 4 + 3)

/-- Theorem stating that the total number of nails and claws is 524 -/
theorem total_nails_and_claws_is_524 : total_nails_and_claws = 524 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_and_claws_is_524_l311_31116


namespace NUMINAMATH_CALUDE_certain_number_problem_l311_31197

theorem certain_number_problem (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l311_31197


namespace NUMINAMATH_CALUDE_inequality_proof_l311_31110

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 4) : 
  |a*c + b*d| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l311_31110


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l311_31131

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- Variance of a linear transformation of a random variable -/
def variance_linear_transform (X : BinomialRV) (a b : ℝ) : ℝ := a^2 * variance X

/-- Main theorem: Variance of 4ξ + 3 for ξ ~ B(100, 0.2) -/
theorem variance_of_transformed_binomial :
  ∃ (ξ : BinomialRV), ξ.n = 100 ∧ ξ.p = 0.2 ∧ variance_linear_transform ξ 4 3 = 256 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l311_31131


namespace NUMINAMATH_CALUDE_frog_count_frog_count_correct_l311_31118

theorem frog_count (num_crocodiles : ℕ) (total_eyes : ℕ) (frog_eyes : ℕ) (crocodile_eyes : ℕ) : ℕ :=
  let num_frogs := (total_eyes - num_crocodiles * crocodile_eyes) / frog_eyes
  num_frogs

theorem frog_count_correct :
  frog_count 6 52 2 2 = 20 := by sorry

end NUMINAMATH_CALUDE_frog_count_frog_count_correct_l311_31118


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l311_31196

theorem fifteen_percent_of_600_is_90 : ∃ x : ℝ, (15 / 100) * x = 90 ∧ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l311_31196


namespace NUMINAMATH_CALUDE_rare_coin_collection_l311_31152

theorem rare_coin_collection (initial_gold : ℕ) (initial_silver : ℕ) : 
  initial_gold = initial_silver / 3 →
  (initial_gold + 15 : ℕ) = initial_silver / 2 →
  initial_gold + initial_silver + 15 = 135 :=
by sorry

end NUMINAMATH_CALUDE_rare_coin_collection_l311_31152


namespace NUMINAMATH_CALUDE_sarah_vacation_reading_l311_31154

/-- Given Sarah's reading speed, book characteristics, and available reading time, prove she can read 6 books. -/
theorem sarah_vacation_reading 
  (reading_speed : ℕ) 
  (words_per_page : ℕ) 
  (pages_per_book : ℕ) 
  (reading_hours : ℕ) 
  (h1 : reading_speed = 40)
  (h2 : words_per_page = 100)
  (h3 : pages_per_book = 80)
  (h4 : reading_hours = 20) : 
  (reading_hours * 60) / ((words_per_page * pages_per_book) / reading_speed) = 6 := by
  sorry

#check sarah_vacation_reading

end NUMINAMATH_CALUDE_sarah_vacation_reading_l311_31154


namespace NUMINAMATH_CALUDE_area_of_PQRSUV_l311_31147

-- Define the polygon and its components
structure Polygon where
  PQ : ℝ
  QR : ℝ
  UV : ℝ
  SU : ℝ
  TU : ℝ
  RS : ℝ

-- Define the conditions
def polygon_conditions (p : Polygon) : Prop :=
  p.PQ = 8 ∧
  p.QR = 10 ∧
  p.UV = 6 ∧
  p.SU = 3 ∧
  p.PQ = p.TU + p.UV ∧
  p.QR = p.RS + p.SU

-- Define the area calculation
def area_PQRSUV (p : Polygon) : ℝ :=
  p.PQ * p.QR - p.SU * p.UV

-- Theorem statement
theorem area_of_PQRSUV (p : Polygon) (h : polygon_conditions p) :
  area_PQRSUV p = 62 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PQRSUV_l311_31147


namespace NUMINAMATH_CALUDE_min_value_expression_l311_31164

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  y / x + 16 * x / (2 * x + y) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l311_31164


namespace NUMINAMATH_CALUDE_special_number_theorem_l311_31181

def is_nine_digit_number (n : ℕ) : Prop :=
  100000000 ≤ n ∧ n ≤ 999999999

def has_special_form (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    a < 1000 ∧ b < 1000 ∧
    n = a * 1000000 + b * 1000 + a ∧
    b = 2 * a

def is_product_of_five_primes_squared (n : ℕ) : Prop :=
  ∃ (p q r s t : ℕ),
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ Nat.Prime t ∧
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
    r ≠ s ∧ r ≠ t ∧
    s ≠ t ∧
    n = (p * q * r * s * t) ^ 2

theorem special_number_theorem (n : ℕ) :
  is_nine_digit_number n ∧
  has_special_form n ∧
  satisfies_condition n ∧
  is_product_of_five_primes_squared n →
  n = 100200100 ∨ n = 225450225 := by
sorry

end NUMINAMATH_CALUDE_special_number_theorem_l311_31181


namespace NUMINAMATH_CALUDE_percentage_calculation_l311_31180

theorem percentage_calculation (x : ℝ) :
  (30 / 100) * ((60 / 100) * ((70 / 100) * x)) = (126 / 1000) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l311_31180


namespace NUMINAMATH_CALUDE_system_solution_l311_31101

theorem system_solution : ∃! (x y : ℝ), x + y = 5 ∧ x - y = 3 ∧ x = 4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l311_31101


namespace NUMINAMATH_CALUDE_other_stamp_price_l311_31111

-- Define the total number of stamps
def total_stamps : ℕ := 75

-- Define the total amount received in cents
def total_amount : ℕ := 480

-- Define the price of the known stamp type
def known_stamp_price : ℕ := 8

-- Define the number of stamps sold of one kind
def stamps_of_one_kind : ℕ := 40

-- Define the function to calculate the price of the unknown stamp type
def unknown_stamp_price (x : ℕ) : Prop :=
  (stamps_of_one_kind * known_stamp_price + (total_stamps - stamps_of_one_kind) * x = total_amount) ∧
  (x > 0) ∧ (x < known_stamp_price)

-- Theorem stating that the price of the unknown stamp type is 5 cents
theorem other_stamp_price : unknown_stamp_price 5 := by
  sorry

end NUMINAMATH_CALUDE_other_stamp_price_l311_31111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l311_31133

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The main theorem -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 4 + a 7 = 39 →
  a 2 + a 5 + a 8 = 33 →
  a 3 + a 6 + a 9 = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l311_31133


namespace NUMINAMATH_CALUDE_balloon_sum_equals_total_l311_31107

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_sum_equals_total :
  fred_balloons + sam_balloons + mary_balloons = total_balloons :=
by sorry

end NUMINAMATH_CALUDE_balloon_sum_equals_total_l311_31107
