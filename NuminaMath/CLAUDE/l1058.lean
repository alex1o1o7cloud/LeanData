import Mathlib

namespace NUMINAMATH_CALUDE_strokes_over_par_tom_strokes_over_par_l1058_105872

theorem strokes_over_par (rounds : ℕ) (avg_strokes : ℕ) (par_value : ℕ) : ℕ :=
  let total_strokes := rounds * avg_strokes
  let total_par := rounds * par_value
  total_strokes - total_par

theorem tom_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_strokes_over_par_tom_strokes_over_par_l1058_105872


namespace NUMINAMATH_CALUDE_multiply_658217_by_99999_l1058_105875

theorem multiply_658217_by_99999 : 658217 * 99999 = 65821034183 := by
  sorry

end NUMINAMATH_CALUDE_multiply_658217_by_99999_l1058_105875


namespace NUMINAMATH_CALUDE_swimming_pool_area_l1058_105816

/-- Represents a rectangular swimming pool with given properties -/
structure SwimmingPool where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_condition : length = 2 * width + 40
  perimeter_condition : perimeter = 2 * (length + width)
  perimeter_value : perimeter = 800

/-- Calculates the area of a rectangular swimming pool -/
def pool_area (pool : SwimmingPool) : ℝ :=
  pool.width * pool.length

/-- Theorem stating that a swimming pool with the given properties has an area of 33600 square feet -/
theorem swimming_pool_area (pool : SwimmingPool) : pool_area pool = 33600 := by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_area_l1058_105816


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l1058_105888

theorem sum_of_squares_divisible_by_three (a b : ℤ) : 
  (3 ∣ a^2 + b^2) → (3 ∣ a) ∧ (3 ∣ b) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_three_l1058_105888


namespace NUMINAMATH_CALUDE_prime_square_mod_six_l1058_105865

theorem prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  p^2 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_six_l1058_105865


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l1058_105849

theorem gcd_lcm_product (a b c : ℕ+) :
  let D := Nat.gcd a (Nat.gcd b c)
  let m := Nat.lcm a (Nat.lcm b c)
  D * m = a * b * c := by
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l1058_105849


namespace NUMINAMATH_CALUDE_coupon_savings_difference_l1058_105814

/-- Represents the savings from Coupon A (15% off the listed price) -/
def savingsA (price : ℝ) : ℝ := 0.15 * price

/-- Represents the savings from Coupon B ($30 off the listed price) -/
def savingsB : ℝ := 30

/-- Represents the savings from Coupon C (25% off the amount exceeding $100) -/
def savingsC (price : ℝ) : ℝ := 0.25 * (price - 100)

/-- The theorem stating the difference between max and min prices where Coupon A is optimal -/
theorem coupon_savings_difference : 
  ∃ (x y : ℝ), 
    x > 100 ∧ y > 100 ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≥ x) ∧
    (∀ p, p > 100 → savingsA p ≥ savingsB → savingsA p ≥ savingsC p → p ≤ y) ∧
    y - x = 50 := by
  sorry

end NUMINAMATH_CALUDE_coupon_savings_difference_l1058_105814


namespace NUMINAMATH_CALUDE_max_value_and_monotonicity_l1058_105890

noncomputable def f (x : ℝ) : ℝ := (3 * Real.log (x + 2) - Real.log (x - 2)) / 2

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f x

theorem max_value_and_monotonicity (h : ∀ x, f x ≥ f 4) :
  (∀ x ∈ Set.Icc 3 7, f x ≤ f 7) ∧
  (∀ a ≥ 1, Monotone (F a) ∧ ∀ a < 1, ¬Monotone (F a)) := by sorry

end NUMINAMATH_CALUDE_max_value_and_monotonicity_l1058_105890


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1058_105842

theorem unique_integer_solution :
  ∃! z : ℤ, (5 * z ≤ 2 * z - 8) ∧ (-3 * z ≥ 18) ∧ (7 * z ≤ -3 * z - 21) := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1058_105842


namespace NUMINAMATH_CALUDE_max_phi_symmetric_sine_l1058_105815

/-- Given a function f(x) = 2sin(4x + φ) where φ < 0, if the graph of f(x) is symmetric
    about the line x = π/24, then the maximum value of φ is -2π/3. -/
theorem max_phi_symmetric_sine (φ : ℝ) (hφ : φ < 0) :
  (∀ x : ℝ, 2 * Real.sin (4 * x + φ) = 2 * Real.sin (4 * (π / 12 - x) + φ)) →
  (∃ (φ_max : ℝ), φ_max = -2 * π / 3 ∧ φ ≤ φ_max ∧ ∀ ψ, ψ < 0 → ψ ≤ φ_max) :=
by sorry

end NUMINAMATH_CALUDE_max_phi_symmetric_sine_l1058_105815


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1058_105802

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) (R : ℝ) :
  R > 0 →
  a > 0 →
  b > 0 →
  c > 0 →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B →
  a = 2 * R * Real.sin A →
  b = 2 * R * Real.sin B →
  c = 2 * R * Real.sin C →
  C = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1058_105802


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1058_105804

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1058_105804


namespace NUMINAMATH_CALUDE_last_number_systematic_sampling_l1058_105810

/-- Systematic sampling function -/
def systematicSampling (totalEmployees : ℕ) (sampleSize : ℕ) (firstNumber : ℕ) : ℕ :=
  let interval := totalEmployees / sampleSize
  firstNumber + (sampleSize - 1) * interval

/-- Theorem: Last number in systematic sampling -/
theorem last_number_systematic_sampling :
  systematicSampling 1000 50 15 = 995 := by
  sorry

#eval systematicSampling 1000 50 15

end NUMINAMATH_CALUDE_last_number_systematic_sampling_l1058_105810


namespace NUMINAMATH_CALUDE_number_added_problem_l1058_105813

theorem number_added_problem (x : ℝ) : 
  3 * (2 * 5 + x) = 57 → x = 9 := by
sorry

end NUMINAMATH_CALUDE_number_added_problem_l1058_105813


namespace NUMINAMATH_CALUDE_smaller_circle_area_l1058_105847

-- Define the radius of the smaller circle
def r : ℝ := sorry

-- Define the radius of the larger circle
def R : ℝ := 3 * r

-- Define the length of the common tangent
def tangent_length : ℝ := 5

-- Theorem statement
theorem smaller_circle_area : 
  r^2 + tangent_length^2 = (R - r)^2 → π * r^2 = 25 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_circle_area_l1058_105847


namespace NUMINAMATH_CALUDE_salary_problem_l1058_105889

theorem salary_problem (A B : ℝ) 
  (h1 : A + B = 2000)
  (h2 : 0.05 * A = 0.15 * B) :
  A = 1500 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1058_105889


namespace NUMINAMATH_CALUDE_negative_two_inequality_l1058_105826

theorem negative_two_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end NUMINAMATH_CALUDE_negative_two_inequality_l1058_105826


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1058_105861

theorem sum_of_x_and_y (x y : ℝ) 
  (eq1 : x^2 + x*y + y = 14) 
  (eq2 : y^2 + x*y + x = 28) : 
  x + y = -7 ∨ x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1058_105861


namespace NUMINAMATH_CALUDE_jar_problem_l1058_105833

theorem jar_problem (total_jars small_jars : ℕ) 
  (small_capacity large_capacity : ℕ) (total_capacity : ℕ) :
  total_jars = 100 →
  small_jars = 62 →
  small_capacity = 3 →
  large_capacity = 5 →
  total_capacity = 376 →
  ∃ large_jars : ℕ, 
    small_jars + large_jars = total_jars ∧
    small_jars * small_capacity + large_jars * large_capacity = total_capacity ∧
    large_jars = 38 :=
by sorry

end NUMINAMATH_CALUDE_jar_problem_l1058_105833


namespace NUMINAMATH_CALUDE_parallelepiped_net_theorem_l1058_105877

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped :=
  (length : ℕ)
  (width : ℕ)
  (height : ℕ)

/-- Represents a net of a parallelepiped -/
structure Net :=
  (squares : ℕ)

/-- Function to unfold a parallelepiped into a net -/
def unfold (p : Parallelepiped) : Net :=
  { squares := 2 * (p.length * p.width + p.length * p.height + p.width * p.height) }

/-- Function to remove one square from a net -/
def remove_square (n : Net) : Net :=
  { squares := n.squares - 1 }

/-- Theorem stating that a 2 × 1 × 1 parallelepiped unfolds into a net with 10 squares,
    and removing one square results in a valid net with 9 squares -/
theorem parallelepiped_net_theorem :
  let p : Parallelepiped := ⟨2, 1, 1⟩
  let full_net : Net := unfold p
  let cut_net : Net := remove_square full_net
  full_net.squares = 10 ∧ cut_net.squares = 9 := by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_net_theorem_l1058_105877


namespace NUMINAMATH_CALUDE_equation_solution_l1058_105874

theorem equation_solution :
  ∃ x : ℚ, x - 1/2 = 7/8 - 2/3 ∧ x = 17/24 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1058_105874


namespace NUMINAMATH_CALUDE_inequality_proof_l1058_105820

theorem inequality_proof (n : ℕ) (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_max : a = max a (max b (max c (max x (max y z)))))
  (h_sum : a + b + c = x + y + z)
  (h_prod : a * b * c = x * y * z) :
  a^n + b^n + c^n ≥ x^n + y^n + z^n := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1058_105820


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l1058_105828

theorem min_value_quadratic_sum (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) :
  a^2 + 4 * b^2 + 9 * c^2 ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l1058_105828


namespace NUMINAMATH_CALUDE_max_d_value_l1058_105886

def a (n : ℕ+) : ℕ := 100 + n^2 + 3*n

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), d m = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1058_105886


namespace NUMINAMATH_CALUDE_imo_inequality_l1058_105879

theorem imo_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_imo_inequality_l1058_105879


namespace NUMINAMATH_CALUDE_log_inequalities_l1058_105892

/-- Proves the inequalities for logarithms with different bases -/
theorem log_inequalities :
  (Real.log 4 / Real.log 8 > Real.log 4 / Real.log 9) ∧
  (Real.log 4 / Real.log 9 > Real.log 4 / Real.log 10) ∧
  (Real.log 4 / Real.log 0.3 < 0.3^2) ∧
  (0.3^2 < 2^0.4) := by
  sorry

end NUMINAMATH_CALUDE_log_inequalities_l1058_105892


namespace NUMINAMATH_CALUDE_marble_jar_problem_l1058_105864

theorem marble_jar_problem (g y : ℕ) : 
  (g - 1 : ℚ) / (g + y - 1 : ℚ) = 1 / 8 →
  (g : ℚ) / (g + y - 3 : ℚ) = 1 / 6 →
  g + y = 9 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l1058_105864


namespace NUMINAMATH_CALUDE_dove_hatching_fraction_l1058_105819

theorem dove_hatching_fraction (initial_doves : ℕ) (eggs_per_dove : ℕ) (total_doves_after : ℕ) :
  initial_doves = 20 →
  eggs_per_dove = 3 →
  total_doves_after = 65 →
  (total_doves_after - initial_doves : ℚ) / (initial_doves * eggs_per_dove) = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_dove_hatching_fraction_l1058_105819


namespace NUMINAMATH_CALUDE_emily_gave_cards_l1058_105812

/-- The number of cards Martha starts with -/
def initial_cards : ℕ := 3

/-- The number of cards Martha ends up with -/
def final_cards : ℕ := 79

/-- The number of cards Emily gave to Martha -/
def cards_from_emily : ℕ := final_cards - initial_cards

theorem emily_gave_cards : cards_from_emily = 76 := by
  sorry

end NUMINAMATH_CALUDE_emily_gave_cards_l1058_105812


namespace NUMINAMATH_CALUDE_x_minus_y_squared_l1058_105800

theorem x_minus_y_squared (x y : ℝ) (hx : x^2 = 4) (hy : y^2 = 9) :
  (x - y)^2 = 25 ∨ (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_squared_l1058_105800


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1058_105803

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_A_and_B : A ∪ B = {x | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1058_105803


namespace NUMINAMATH_CALUDE_megan_homework_pages_l1058_105898

def remaining_pages (total_problems completed_problems problems_per_page : ℕ) : ℕ :=
  ((total_problems - completed_problems) + problems_per_page - 1) / problems_per_page

theorem megan_homework_pages : remaining_pages 40 26 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_pages_l1058_105898


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1058_105822

theorem sqrt_equation_solution :
  ∀ a b : ℕ+,
    a < b →
    Real.sqrt (1 + Real.sqrt (45 + 16 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
    a = 1 ∧ b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1058_105822


namespace NUMINAMATH_CALUDE_divisibility_implication_l1058_105876

theorem divisibility_implication (a b : ℕ) : 
  a < 1000 → (∃ k : ℕ, a^21 = k * b^10) → (∃ m : ℕ, a^2 = m * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1058_105876


namespace NUMINAMATH_CALUDE_exam_mode_l1058_105884

def scores : List Nat := [7, 10, 9, 8, 7, 9, 9, 8]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem exam_mode :
  mode scores = some 9 := by
  sorry

end NUMINAMATH_CALUDE_exam_mode_l1058_105884


namespace NUMINAMATH_CALUDE_triangle_side_inequality_l1058_105808

/-- Triangle inequality for side lengths a, b, c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main inequality to be proved -/
def main_inequality (a b c : ℝ) : ℝ :=
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a)

theorem triangle_side_inequality (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (tri : triangle_inequality a b c) : 
  main_inequality a b c ≥ 0 ∧ 
  (main_inequality a b c = 0 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_inequality_l1058_105808


namespace NUMINAMATH_CALUDE_mountain_climbing_speed_ratio_l1058_105895

/-- Proves that the ratio of ascending to descending speeds is 3:4 given the conditions of the problem -/
theorem mountain_climbing_speed_ratio 
  (s : ℝ) -- Total distance of the mountain path
  (x : ℝ) -- Jia's ascending speed
  (y : ℝ) -- Yi's descending speed
  (h1 : s > 0) -- The distance is positive
  (h2 : x > 0) -- Ascending speed is positive
  (h3 : y > 0) -- Descending speed is positive
  (h4 : s / x - s / (x + y) = 16) -- Time difference for Jia after meeting
  (h5 : s / y - s / (x + y) = 9) -- Time difference for Yi after meeting
  : x / y = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_mountain_climbing_speed_ratio_l1058_105895


namespace NUMINAMATH_CALUDE_max_height_of_smaller_box_l1058_105836

/-- The maximum height of a smaller box that can fit in a larger box --/
theorem max_height_of_smaller_box 
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (max_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_width = 0.5 →
  max_boxes = 1000 →
  ∃ (h : ℝ), h ≤ 0.4 ∧ 
    (max_boxes : ℝ) * small_length * small_width * h ≤ 
    large_length * large_width * large_height :=
by sorry

end NUMINAMATH_CALUDE_max_height_of_smaller_box_l1058_105836


namespace NUMINAMATH_CALUDE_andrew_donut_problem_l1058_105897

/-- The number of donuts Andrew ate on Monday -/
def monday_donuts : ℕ := 14

/-- The number of donuts Andrew ate on Tuesday -/
def tuesday_donuts : ℕ := monday_donuts / 2

/-- The total number of donuts Andrew ate in three days -/
def total_donuts : ℕ := 49

/-- The multiplier for the number of donuts Andrew ate on Wednesday compared to Monday -/
def wednesday_multiplier : ℚ := 2

theorem andrew_donut_problem :
  monday_donuts + tuesday_donuts + (wednesday_multiplier * monday_donuts) = total_donuts :=
sorry

end NUMINAMATH_CALUDE_andrew_donut_problem_l1058_105897


namespace NUMINAMATH_CALUDE_gcd_not_eight_l1058_105801

theorem gcd_not_eight (x y : ℕ+) (h : y = x^2 + 8) : Nat.gcd x.val y.val ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_not_eight_l1058_105801


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l1058_105844

/-- Given a complex number z = 3 - i, prove that the imaginary part of z² is -6 -/
theorem imaginary_part_of_z_squared (z : ℂ) (h : z = 3 - I) : 
  (z^2).im = -6 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_squared_l1058_105844


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1058_105866

/-- Given a geometric sequence {a_n} with common ratio q = 1/2 and sum of first n terms S_n,
    prove that S_4 / a_4 = 15 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence condition
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Sum formula
  q = (1 : ℝ) / 2 →  -- Common ratio
  S 4 / a 4 = 15 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1058_105866


namespace NUMINAMATH_CALUDE_rectangular_solid_pythagorean_l1058_105817

/-- A rectangular solid with given dimensions and body diagonal -/
structure RectangularSolid where
  p : ℝ  -- length
  q : ℝ  -- width
  r : ℝ  -- height
  d : ℝ  -- body diagonal length

/-- The Pythagorean theorem for rectangular solids -/
theorem rectangular_solid_pythagorean (solid : RectangularSolid) :
  solid.p^2 + solid.q^2 + solid.r^2 = solid.d^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_solid_pythagorean_l1058_105817


namespace NUMINAMATH_CALUDE_no_partition_with_translation_l1058_105893

theorem no_partition_with_translation (A B : Set ℝ) (a : ℝ) : 
  A ⊆ Set.Icc 0 1 → 
  B ⊆ Set.Icc 0 1 → 
  A ∩ B = ∅ → 
  B = {x | ∃ y ∈ A, x = y + a} → 
  False :=
sorry

end NUMINAMATH_CALUDE_no_partition_with_translation_l1058_105893


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1058_105841

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_no_small_factors :
  (∀ m : ℕ, m < 4091 →
    is_prime m ∨
    is_perfect_square m ∨
    ¬(has_no_prime_factor_less_than m 60)) ∧
  ¬(is_prime 4091) ∧
  ¬(is_perfect_square 4091) ∧
  has_no_prime_factor_less_than 4091 60 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1058_105841


namespace NUMINAMATH_CALUDE_students_who_left_zoo_l1058_105838

/-- Proves the number of students who left the zoo given the initial conditions and remaining individuals --/
theorem students_who_left_zoo 
  (initial_students : Nat) 
  (initial_chaperones : Nat) 
  (initial_teachers : Nat) 
  (remaining_individuals : Nat) 
  (chaperones_who_left : Nat)
  (h1 : initial_students = 20)
  (h2 : initial_chaperones = 5)
  (h3 : initial_teachers = 2)
  (h4 : remaining_individuals = 15)
  (h5 : chaperones_who_left = 2) :
  initial_students - (remaining_individuals - chaperones_who_left - initial_teachers) = 9 := by
  sorry


end NUMINAMATH_CALUDE_students_who_left_zoo_l1058_105838


namespace NUMINAMATH_CALUDE_congruence_characterization_l1058_105853

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Characterization of integers satisfying the given congruence -/
theorem congruence_characterization (n : ℕ) (h : n > 2) :
  (phi n / 2) % 6 = 1 ↔ 
    n = 3 ∨ n = 4 ∨ n = 6 ∨ 
    (∃ (p k : ℕ), p.Prime ∧ p % 12 = 11 ∧ (n = p^(2*k) ∨ n = 2 * p^(2*k))) :=
  sorry

end NUMINAMATH_CALUDE_congruence_characterization_l1058_105853


namespace NUMINAMATH_CALUDE_johnny_weekly_earnings_l1058_105855

/-- Represents Johnny's dog walking business --/
structure DogWalker where
  dogs_per_walk : ℕ
  pay_30min : ℕ
  pay_60min : ℕ
  hours_per_day : ℕ
  long_walks_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates Johnny's weekly earnings --/
def weekly_earnings (dw : DogWalker) : ℕ :=
  sorry

/-- Johnny's specific situation --/
def johnny : DogWalker := {
  dogs_per_walk := 3,
  pay_30min := 15,
  pay_60min := 20,
  hours_per_day := 4,
  long_walks_per_day := 6,
  work_days_per_week := 5
}

/-- Theorem stating Johnny's weekly earnings --/
theorem johnny_weekly_earnings : weekly_earnings johnny = 1500 :=
  sorry

end NUMINAMATH_CALUDE_johnny_weekly_earnings_l1058_105855


namespace NUMINAMATH_CALUDE_math_competition_problem_l1058_105839

theorem math_competition_problem (a b c : ℕ) 
  (h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h2 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h3 : (1/a + 1/b + 1/c - 1/a*1/b - 1/a*1/c - 1/b*1/c + 1/a*1/b*1/c : ℚ) = 7/15) :
  ((1 - 1/a) * (1 - 1/b) * (1 - 1/c) : ℚ) = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l1058_105839


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1058_105850

/-- The inradius of a right triangle with side lengths 12, 35, and 37 is 5 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 12 ∧ b = 35 ∧ c = 37 →  -- Side lengths
  a^2 + b^2 = c^2 →           -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1058_105850


namespace NUMINAMATH_CALUDE_math_competition_team_selection_l1058_105806

theorem math_competition_team_selection (n : ℕ) (k : ℕ) (total : ℕ) (exclude : ℕ) :
  n = 10 →
  k = 3 →
  total = Nat.choose (n - 1) k →
  exclude = Nat.choose (n - 3) k →
  total - exclude = 49 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_team_selection_l1058_105806


namespace NUMINAMATH_CALUDE_bagel_store_spending_l1058_105857

theorem bagel_store_spending (B D : ℝ) : 
  D = (9/10) * B →
  B = D + 15 →
  B + D = 285 :=
by sorry

end NUMINAMATH_CALUDE_bagel_store_spending_l1058_105857


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l1058_105832

theorem minutes_to_seconds (minutes : Real) (seconds_per_minute : Nat) :
  minutes * seconds_per_minute = 468 → minutes = 7.8 ∧ seconds_per_minute = 60 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l1058_105832


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1058_105870

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1058_105870


namespace NUMINAMATH_CALUDE_creatures_conference_handshakes_l1058_105869

def num_goblins : ℕ := 25
def num_elves : ℕ := 18
def num_fairies : ℕ := 20

def handshakes_among (n : ℕ) : ℕ := n * (n - 1) / 2

def handshakes_between (n : ℕ) (m : ℕ) : ℕ := n * m

def total_handshakes : ℕ :=
  handshakes_among num_goblins +
  handshakes_among num_elves +
  handshakes_between num_goblins num_fairies +
  handshakes_between num_elves num_fairies

theorem creatures_conference_handshakes :
  total_handshakes = 1313 := by sorry

end NUMINAMATH_CALUDE_creatures_conference_handshakes_l1058_105869


namespace NUMINAMATH_CALUDE_maria_towels_result_l1058_105823

/-- The number of towels Maria ended up with after shopping and giving some to her mother. -/
def towels_maria_kept (green_towels white_towels towels_given : ℕ) : ℕ :=
  green_towels + white_towels - towels_given

/-- Theorem stating that Maria ended up with 22 towels. -/
theorem maria_towels_result :
  towels_maria_kept 35 21 34 = 22 := by
  sorry

end NUMINAMATH_CALUDE_maria_towels_result_l1058_105823


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1058_105862

theorem polynomial_equation_solution (x : ℝ) : 
  let p : ℝ → ℝ := λ x => (1 + Real.sqrt 109) / 2
  p (x^2) - p (x^2 - 3) = (p x)^2 + 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1058_105862


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l1058_105856

def flower_pots_cost (n : ℕ) (price_difference : ℚ) (largest_pot_price : ℚ) : ℚ :=
  let smallest_pot_price := largest_pot_price - (n - 1 : ℚ) * price_difference
  (n : ℚ) * smallest_pot_price + ((n - 1) * n / 2 : ℚ) * price_difference

theorem flower_pots_total_cost :
  flower_pots_cost 6 (3/10) (85/40) = 33/4 :=
sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l1058_105856


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1058_105883

theorem triangle_angle_measure (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_area : (1 / (4 * Real.sqrt 3)) * (b^2 + c^2 - a^2) = 
            (1 / 2) * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1058_105883


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1058_105896

theorem cos_seven_pi_sixths : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l1058_105896


namespace NUMINAMATH_CALUDE_product_nine_sum_undetermined_l1058_105858

theorem product_nine_sum_undetermined : 
  ∃ (a b c d : ℤ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a * b * c * d = 9 ∧
    ¬∃! (s : ℤ), s = a + b + c + d :=
by sorry

end NUMINAMATH_CALUDE_product_nine_sum_undetermined_l1058_105858


namespace NUMINAMATH_CALUDE_integer_solution_exists_l1058_105863

theorem integer_solution_exists (a b : ℤ) : ∃ (x y z t : ℤ), 
  (x + y + 2*z + 2*t = a) ∧ (2*x - 2*y + z - t = b) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l1058_105863


namespace NUMINAMATH_CALUDE_min_diff_integers_avg_l1058_105851

theorem min_diff_integers_avg (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧  -- Five different positive integers
  (a + b + c + d + e) / 5 = 5 ∧    -- Average is 5
  ∀ x y z w v : ℕ,                 -- For any other set of 5 different positive integers
    x < y ∧ y < z ∧ z < w ∧ w < v ∧
    (x + y + z + w + v) / 5 = 5 →
    (e - a) ≤ (v - x) →            -- with minimum difference
  (b + c + d) / 3 = 5 :=           -- Average of middle three is 5
by sorry

end NUMINAMATH_CALUDE_min_diff_integers_avg_l1058_105851


namespace NUMINAMATH_CALUDE_workday_meeting_percentage_l1058_105837

/-- Represents the duration of a workday in hours -/
def workday_hours : ℝ := 10

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℝ := 60

/-- Calculates the total workday time in minutes -/
def workday_minutes : ℝ := workday_hours * 60

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℝ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℝ := first_meeting_minutes + second_meeting_minutes

/-- Theorem: The percentage of the workday spent in meetings is 30% -/
theorem workday_meeting_percentage :
  (total_meeting_minutes / workday_minutes) * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_workday_meeting_percentage_l1058_105837


namespace NUMINAMATH_CALUDE_chord_intersection_probability_is_one_twelfth_l1058_105887

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2020

/-- The probability that two randomly chosen chords intersect -/
def chord_intersection_probability : ℚ := 1 / 12

/-- Theorem stating that the probability of two randomly chosen chords intersecting is 1/12 -/
theorem chord_intersection_probability_is_one_twelfth :
  chord_intersection_probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_is_one_twelfth_l1058_105887


namespace NUMINAMATH_CALUDE_corey_candy_count_l1058_105891

theorem corey_candy_count (total : ℕ) (difference : ℕ) (corey : ℕ) : 
  total = 66 → difference = 8 → corey + (corey + difference) = total → corey = 29 := by
  sorry

end NUMINAMATH_CALUDE_corey_candy_count_l1058_105891


namespace NUMINAMATH_CALUDE_northern_village_population_l1058_105867

/-- The number of people in the western village -/
def western_village : ℕ := 7488

/-- The number of people in the southern village -/
def southern_village : ℕ := 6912

/-- The total number of people conscripted from all three villages -/
def total_conscripted : ℕ := 300

/-- The number of people conscripted from the northern village -/
def northern_conscripted : ℕ := 108

/-- The number of people in the northern village -/
def northern_village : ℕ := 4206

theorem northern_village_population :
  (northern_conscripted : ℚ) / total_conscripted =
  northern_village / (northern_village + western_village + southern_village) :=
sorry

end NUMINAMATH_CALUDE_northern_village_population_l1058_105867


namespace NUMINAMATH_CALUDE_smallest_positive_root_floor_l1058_105835

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

def is_smallest_positive_root (s : ℝ) : Prop :=
  s > 0 ∧ g s = 0 ∧ ∀ x, 0 < x ∧ x < s → g x ≠ 0

theorem smallest_positive_root_floor :
  ∃ s, is_smallest_positive_root s ∧ ⌊s⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_smallest_positive_root_floor_l1058_105835


namespace NUMINAMATH_CALUDE_birds_in_second_tree_l1058_105868

/-- Represents the number of birds in each tree -/
structure TreeBirds where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The initial state of birds in the trees -/
def initial_state : TreeBirds := sorry

/-- The state after birds have flown away -/
def final_state : TreeBirds := sorry

theorem birds_in_second_tree :
  /- Total number of birds initially -/
  initial_state.first + initial_state.second + initial_state.third = 60 →
  /- Birds that flew away from each tree -/
  initial_state.first - final_state.first = 6 →
  initial_state.second - final_state.second = 8 →
  initial_state.third - final_state.third = 4 →
  /- Equal number of birds in each tree after flying away -/
  final_state.first = final_state.second →
  final_state.second = final_state.third →
  /- The number of birds originally in the second tree was 22 -/
  initial_state.second = 22 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_second_tree_l1058_105868


namespace NUMINAMATH_CALUDE_symmetric_curve_equation_l1058_105854

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line of symmetry
def symmetry_line : ℝ := 2

-- Define the symmetric point
def symmetric_point (x y : ℝ) : ℝ × ℝ := (4 - x, y)

-- Theorem statement
theorem symmetric_curve_equation :
  ∀ x y : ℝ, original_curve (4 - x) y → y^2 = 16 - 4*x :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_equation_l1058_105854


namespace NUMINAMATH_CALUDE_rose_discount_percentage_l1058_105859

theorem rose_discount_percentage (dozen_count : ℕ) (cost_per_rose : ℕ) (final_amount : ℕ) : 
  dozen_count = 5 → 
  cost_per_rose = 6 → 
  final_amount = 288 → 
  (1 - (final_amount : ℚ) / (dozen_count * 12 * cost_per_rose)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rose_discount_percentage_l1058_105859


namespace NUMINAMATH_CALUDE_triangle_inequality_l1058_105824

/-- Given a triangle with sides a, b, c and angle γ opposite side c,
    prove that c ≥ (a + b) * sin(γ/2) --/
theorem triangle_inequality (a b c γ : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angle : 0 < γ ∧ γ < π)
  (h_opposite : γ = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) :
  c ≥ (a + b) * Real.sin (γ / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1058_105824


namespace NUMINAMATH_CALUDE_waiter_tables_l1058_105830

theorem waiter_tables (customers_per_table : ℕ) (total_customers : ℕ) (h1 : customers_per_table = 8) (h2 : total_customers = 48) :
  total_customers / customers_per_table = 6 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1058_105830


namespace NUMINAMATH_CALUDE_simple_interest_solution_l1058_105825

/-- Simple interest calculation -/
def simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) : Prop :=
  let principal := interest / (rate * time / 100)
  principal = 8935

/-- Theorem stating the solution to the simple interest problem -/
theorem simple_interest_solution :
  simple_interest_problem 4020.75 9 5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_solution_l1058_105825


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_system_l1058_105852

theorem unique_solution_quadratic_system :
  ∃! x : ℚ, (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) ∧ (x = 1/5) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_system_l1058_105852


namespace NUMINAMATH_CALUDE_solution_form_l1058_105882

open Real

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 1 → y > 1 → f x - f y = (y - x) * f (x * y)

/-- The theorem stating that any function satisfying the equation must be of the form k/x -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    ∃ k : ℝ, ∀ x, x > 1 → f x = k / x := by
  sorry

end NUMINAMATH_CALUDE_solution_form_l1058_105882


namespace NUMINAMATH_CALUDE_existence_of_periodic_even_function_l1058_105821

theorem existence_of_periodic_even_function :
  ∃ f : ℝ → ℝ,
    (f 0 ≠ 0) ∧
    (∀ x : ℝ, f x = f (-x)) ∧
    (∀ x : ℝ, f (x + π) = f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_periodic_even_function_l1058_105821


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1058_105831

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1058_105831


namespace NUMINAMATH_CALUDE_log_identity_l1058_105878

theorem log_identity (a : ℝ) (h : a = Real.log 3 / Real.log 4) : 
  2^a + 2^(-a) = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1058_105878


namespace NUMINAMATH_CALUDE_count_quadruples_l1058_105846

theorem count_quadruples : 
  let S := {q : Fin 10 × Fin 10 × Fin 10 × Fin 10 | true}
  Fintype.card S = 10000 := by sorry

end NUMINAMATH_CALUDE_count_quadruples_l1058_105846


namespace NUMINAMATH_CALUDE_tiangong_survey_method_l1058_105827

/-- Represents the types of survey methods --/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the requirements for the survey --/
structure SurveyRequirements where
  high_precision : Bool
  no_errors_allowed : Bool

/-- Determines the appropriate survey method based on the given requirements --/
def appropriate_survey_method (requirements : SurveyRequirements) : SurveyMethod :=
  if requirements.high_precision && requirements.no_errors_allowed then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sampling

theorem tiangong_survey_method :
  let requirements : SurveyRequirements := ⟨true, true⟩
  appropriate_survey_method requirements = SurveyMethod.Comprehensive :=
by sorry

end NUMINAMATH_CALUDE_tiangong_survey_method_l1058_105827


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l1058_105848

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The law of sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- Theorem: In triangle ABC, if angle A = 120°, side a = 2, and side b = (2√3)/3, then angle B = π/6 -/
theorem angle_B_is_pi_over_six (t : Triangle) 
  (h1 : t.A = 2 * π / 3)  -- 120° in radians
  (h2 : t.a = 2)
  (h3 : t.b = 2 * Real.sqrt 3 / 3) :
  t.B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_is_pi_over_six_l1058_105848


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1058_105881

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 8 ∧ x₁^2 + y^2 = 169) ∧
  (∃ y : ℝ, y = 8 ∧ x₂^2 + y^2 = 169) ∧
  (∀ x : ℝ, (∃ y : ℝ, y = 8 ∧ x^2 + y^2 = 169) → (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l1058_105881


namespace NUMINAMATH_CALUDE_max_diagonals_regular_1000_gon_l1058_105860

/-- The number of sides in the regular polygon -/
def n : ℕ := 1000

/-- The number of different diagonal lengths in a regular n-gon -/
def num_diagonal_lengths (n : ℕ) : ℕ := n / 2

/-- The total number of diagonals in a regular n-gon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals of each length in a regular n-gon -/
def diagonals_per_length (n : ℕ) : ℕ := n

/-- The maximum number of diagonals that can be selected such that among any three of the chosen diagonals, at least two have the same length -/
def max_selected_diagonals (n : ℕ) : ℕ := 2 * diagonals_per_length n

theorem max_diagonals_regular_1000_gon :
  max_selected_diagonals n = 2000 :=
sorry

end NUMINAMATH_CALUDE_max_diagonals_regular_1000_gon_l1058_105860


namespace NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_11_l1058_105894

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

def is_smallest_ending_in_9_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ ends_in_9 n ∧ n % 11 = 0 ∧
  ∀ m : ℕ, m > 0 → ends_in_9 m → m % 11 = 0 → m ≥ n

theorem smallest_ending_in_9_divisible_by_11 :
  is_smallest_ending_in_9_divisible_by_11 319 := by
sorry

end NUMINAMATH_CALUDE_smallest_ending_in_9_divisible_by_11_l1058_105894


namespace NUMINAMATH_CALUDE_laces_for_shoes_l1058_105807

theorem laces_for_shoes (num_pairs : ℕ) (laces_per_pair : ℕ) (h1 : num_pairs = 26) (h2 : laces_per_pair = 2) :
  num_pairs * laces_per_pair = 52 := by
  sorry

end NUMINAMATH_CALUDE_laces_for_shoes_l1058_105807


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1058_105834

theorem coefficient_x_squared_in_expansion : 
  let n : ℕ := 6
  let k : ℕ := 2
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * b^k * a^(n-k) = 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1058_105834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1058_105809

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- a₂₀ is the 20th term
- a₃₀ is the 30th term
This theorem states that if a₁ = 3 and a₂₀ = 41, then a₃₀ = 61.
-/
theorem arithmetic_sequence_30th_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + m) - a n = m * (a 2 - a 1))
  (h_first : a 1 = 3)
  (h_twentieth : a 20 = 41) : 
  a 30 = 61 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l1058_105809


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1058_105811

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1058_105811


namespace NUMINAMATH_CALUDE_corner_to_triangle_ratio_is_one_l1058_105840

/-- Represents a square board with four equally spaced lines passing through its center -/
structure Board :=
  (side_length : ℝ)
  (is_square : side_length > 0)

/-- Represents the area of a triangular section in the board -/
def triangular_area (b : Board) : ℝ := sorry

/-- Represents the area of a corner region in the board -/
def corner_area (b : Board) : ℝ := sorry

/-- Theorem stating that the ratio of corner area to triangular area is 1 for a board with side length 2 -/
theorem corner_to_triangle_ratio_is_one :
  ∀ (b : Board), b.side_length = 2 → corner_area b / triangular_area b = 1 := by sorry

end NUMINAMATH_CALUDE_corner_to_triangle_ratio_is_one_l1058_105840


namespace NUMINAMATH_CALUDE_beatrice_book_cost_l1058_105880

/-- Calculates the total cost of books given the pricing rules and number of books purchased. -/
def book_cost (regular_price : ℕ) (discount : ℕ) (regular_quantity : ℕ) (total_quantity : ℕ) : ℕ :=
  let regular_cost := regular_price * regular_quantity
  let discounted_quantity := total_quantity - regular_quantity
  let discounted_price := regular_price - discount
  let discounted_cost := discounted_quantity * discounted_price
  regular_cost + discounted_cost

/-- Proves that given the specific pricing rules and Beatrice's purchase, the total cost is $370. -/
theorem beatrice_book_cost :
  let regular_price := 20
  let discount := 2
  let regular_quantity := 5
  let total_quantity := 20
  book_cost regular_price discount regular_quantity total_quantity = 370 := by
  sorry

#eval book_cost 20 2 5 20  -- This should output 370

end NUMINAMATH_CALUDE_beatrice_book_cost_l1058_105880


namespace NUMINAMATH_CALUDE_highest_average_speed_l1058_105843

def time_periods : Fin 5 → String
| 0 => "8-9 am"
| 1 => "9-10 am"
| 2 => "10-11 am"
| 3 => "2-3 pm"
| 4 => "3-4 pm"

def distances : Fin 5 → ℝ
| 0 => 50
| 1 => 70
| 2 => 60
| 3 => 80
| 4 => 40

def average_speed (i : Fin 5) : ℝ := distances i

def highest_speed_period : Fin 5 := 3

theorem highest_average_speed :
  ∀ (i : Fin 5), average_speed highest_speed_period ≥ average_speed i :=
by sorry

end NUMINAMATH_CALUDE_highest_average_speed_l1058_105843


namespace NUMINAMATH_CALUDE_surface_area_union_cones_l1058_105829

/-- The surface area of the union of two right cones with specific dimensions -/
theorem surface_area_union_cones (r h : ℝ) (hr : r = 4) (hh : h = 3) :
  let L := Real.sqrt (r^2 + h^2)
  let surface_area_one_cone := π * r^2 + π * r * L
  let lateral_area_half_cone := π * (r/2) * (Real.sqrt ((r/2)^2 + (h/2)^2))
  2 * (surface_area_one_cone - lateral_area_half_cone) = 62 * π :=
by sorry

end NUMINAMATH_CALUDE_surface_area_union_cones_l1058_105829


namespace NUMINAMATH_CALUDE_vowel_initials_probability_l1058_105873

/-- Represents the set of possible initials --/
def Initials : Type := Char

/-- The set of all possible initials --/
def all_initials : Finset Initials := sorry

/-- The set of vowel initials --/
def vowel_initials : Finset Initials := sorry

/-- The number of students in the class --/
def class_size : ℕ := 30

/-- No two students have the same initials --/
axiom unique_initials : class_size ≤ Finset.card all_initials

/-- The probability of picking a student with vowel initials --/
def vowel_probability : ℚ := (Finset.card vowel_initials : ℚ) / (Finset.card all_initials : ℚ)

/-- Main theorem: The probability of picking a student with vowel initials is 5/26 --/
theorem vowel_initials_probability : vowel_probability = 5 / 26 := by sorry

end NUMINAMATH_CALUDE_vowel_initials_probability_l1058_105873


namespace NUMINAMATH_CALUDE_fair_coin_same_side_probability_l1058_105805

theorem fair_coin_same_side_probability :
  let n : ℕ := 10
  let p : ℝ := 1 / 2
  (p ^ n : ℝ) = 1 / 1024 := by sorry

end NUMINAMATH_CALUDE_fair_coin_same_side_probability_l1058_105805


namespace NUMINAMATH_CALUDE_shirt_cost_l1058_105871

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 76) →
  shirt_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l1058_105871


namespace NUMINAMATH_CALUDE_equivalent_workout_l1058_105885

/-- Represents the weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ → ℕ
| 0 => 15
| 1 => 20
| _ => 0

/-- Calculates the total weight lifted given the dumbbell type and number of repetitions -/
def total_weight (dumbbell_type : ℕ) (repetitions : ℕ) : ℕ :=
  2 * dumbbell_weight dumbbell_type * repetitions

/-- Proves that lifting two 15-pound weights 16 times is equivalent to lifting two 20-pound weights 12 times -/
theorem equivalent_workout : total_weight 0 16 = total_weight 1 12 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_workout_l1058_105885


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1058_105818

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion term
def expansionTerm (a : ℝ) (r : ℕ) : ℝ := 
  (-1)^r * a^(8 - r) * binomial 8 r

-- State the theorem
theorem expansion_coefficient (a : ℝ) : 
  (expansionTerm a 4 = 70) → (a = 1 ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1058_105818


namespace NUMINAMATH_CALUDE_simplify_expression_l1058_105899

theorem simplify_expression (w : ℝ) : 2*w + 3 - 4*w - 5 + 6*w + 7 - 8*w - 9 = -4*w - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1058_105899


namespace NUMINAMATH_CALUDE_sum_of_angles_x_and_y_l1058_105845

-- Define a circle divided into 16 equal arcs
def circle_arcs : ℕ := 16

-- Define the span of angle x
def x_span : ℕ := 3

-- Define the span of angle y
def y_span : ℕ := 5

-- Theorem statement
theorem sum_of_angles_x_and_y (x y : Real) :
  (x = (360 / circle_arcs * x_span) / 2) →
  (y = (360 / circle_arcs * y_span) / 2) →
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_x_and_y_l1058_105845
