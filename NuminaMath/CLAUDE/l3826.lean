import Mathlib

namespace complex_equidistant_modulus_l3826_382672

theorem complex_equidistant_modulus (z : ℂ) : 
  Complex.abs z = Complex.abs (z - 1) ∧ 
  Complex.abs z = Complex.abs (z - Complex.I) → 
  Complex.abs z = Real.sqrt 2 / 2 := by
sorry

end complex_equidistant_modulus_l3826_382672


namespace quadratic_equation_roots_approximate_roots_l3826_382639

/-- The quadratic equation √3x² + √17x - √6 = 0 has two real roots -/
theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  x₁ ≠ x₂ :=
by sorry

/-- The roots of the equation √3x² + √17x - √6 = 0 are approximately 0.492 and -2.873 -/
theorem approximate_roots : ∃ (x₁ x₂ : ℝ),
  Real.sqrt 3 * x₁^2 + Real.sqrt 17 * x₁ - Real.sqrt 6 = 0 ∧
  Real.sqrt 3 * x₂^2 + Real.sqrt 17 * x₂ - Real.sqrt 6 = 0 ∧
  abs (x₁ - 0.492) < 0.0005 ∧
  abs (x₂ + 2.873) < 0.0005 :=
by sorry

end quadratic_equation_roots_approximate_roots_l3826_382639


namespace second_discount_is_fifteen_percent_l3826_382613

/-- Calculates the final price of a car after three successive discounts -/
def finalPrice (initialPrice : ℝ) (discount1 discount2 discount3 : ℝ) : ℝ :=
  let price1 := initialPrice * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  price2 * (1 - discount3)

/-- Theorem stating that given the initial price and three discounts, 
    the second discount is 15% when the final price is $7,752 -/
theorem second_discount_is_fifteen_percent 
  (initialPrice : ℝ) 
  (discount1 : ℝ) 
  (discount3 : ℝ) :
  initialPrice = 12000 →
  discount1 = 0.20 →
  discount3 = 0.05 →
  finalPrice initialPrice discount1 0.15 discount3 = 7752 :=
by
  sorry

#eval finalPrice 12000 0.20 0.15 0.05

end second_discount_is_fifteen_percent_l3826_382613


namespace polynomial_ratio_l3826_382653

-- Define the polynomial function
def p (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  a₀ + a₁ * (2 - x) + a₂ * (2 - x)^2 + a₃ * (2 - x)^3 + a₄ * (2 - x)^4 + a₅ * (2 - x)^5

-- State the theorem
theorem polynomial_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = p x a₀ a₁ a₂ a₃ a₄ a₅) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61/60 := by
  sorry

end polynomial_ratio_l3826_382653


namespace lower_circle_radius_is_153_l3826_382696

/-- Configuration of circles and square between parallel lines -/
structure GeometricConfiguration where
  -- Distance between parallel lines
  line_distance : ℝ
  -- Side length of the square
  square_side : ℝ
  -- Radius of the upper circle
  upper_radius : ℝ
  -- The configuration satisfies the given conditions
  h1 : line_distance = 400
  h2 : square_side = 279
  h3 : upper_radius = 65

/-- Calculate the radius of the lower circle -/
def lower_circle_radius (config : GeometricConfiguration) : ℝ :=
  -- Placeholder for the actual calculation
  153

/-- Theorem stating that the radius of the lower circle is 153 units -/
theorem lower_circle_radius_is_153 (config : GeometricConfiguration) :
  lower_circle_radius config = 153 := by
  sorry

#check lower_circle_radius_is_153

end lower_circle_radius_is_153_l3826_382696


namespace arithmetic_sequence_eighth_term_l3826_382614

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arithmetic : ArithmeticSequence a)
  (h_fourth : a 4 = 23)
  (h_sixth : a 6 = 47) :
  a 8 = 71 := by
  sorry

end arithmetic_sequence_eighth_term_l3826_382614


namespace lens_price_proof_l3826_382676

theorem lens_price_proof (price_no_discount : ℝ) (discount_rate : ℝ) (cheaper_lens_price : ℝ) :
  price_no_discount = 300 ∧
  discount_rate = 0.2 ∧
  cheaper_lens_price = 220 ∧
  price_no_discount * (1 - discount_rate) = cheaper_lens_price + 20 :=
by sorry

end lens_price_proof_l3826_382676


namespace opposite_of_negative_one_fourth_l3826_382622

theorem opposite_of_negative_one_fourth :
  -((-1 : ℚ) / 4) = 1 / 4 := by sorry

end opposite_of_negative_one_fourth_l3826_382622


namespace chord_arithmetic_sequence_l3826_382616

theorem chord_arithmetic_sequence (n : ℕ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5*x}
  let point := (5/2, 3/2)
  let shortest_chord := 4
  let longest_chord := 5
  ∀ d : ℝ, 1/6 < d ∧ d ≤ 1/3 →
    (n > 0 ∧ 
     shortest_chord + (n - 1) * d = longest_chord ∧
     point ∈ circle) →
    n ∈ ({4, 5, 6} : Set ℕ) :=
by sorry

end chord_arithmetic_sequence_l3826_382616


namespace ribbon_leftover_l3826_382610

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) :
  total_ribbon = 18 ∧ num_gifts = 6 ∧ ribbon_per_gift = 2 →
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
sorry

end ribbon_leftover_l3826_382610


namespace simplify_sqrt_expression_l3826_382667

theorem simplify_sqrt_expression (y : ℝ) (h : y ≠ 0) : 
  Real.sqrt (4 + ((y^3 - 2) / (3 * y))^2) = (Real.sqrt (y^6 - 4*y^3 + 36*y^2 + 4)) / (3 * y) :=
by sorry

end simplify_sqrt_expression_l3826_382667


namespace smallest_sum_with_factors_and_perfect_square_l3826_382682

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_sum_with_factors_and_perfect_square :
  ∃ (a b : ℕ+),
    num_factors a = 15 ∧
    num_factors b = 20 ∧
    is_perfect_square (a.val + b.val) ∧
    ∀ (c d : ℕ+),
      num_factors c = 15 →
      num_factors d = 20 →
      is_perfect_square (c.val + d.val) →
      a.val + b.val ≤ c.val + d.val ∧
      a.val + b.val = 576 :=
sorry

end smallest_sum_with_factors_and_perfect_square_l3826_382682


namespace least_number_of_cookies_l3826_382660

theorem least_number_of_cookies (a : ℕ) : 
  a > 0 ∧ 
  a % 4 = 3 ∧ 
  a % 5 = 2 ∧ 
  a % 7 = 4 ∧ 
  (∀ b : ℕ, b > 0 ∧ b % 4 = 3 ∧ b % 5 = 2 ∧ b % 7 = 4 → a ≤ b) → 
  a = 67 := by
sorry

end least_number_of_cookies_l3826_382660


namespace housing_boom_construction_l3826_382625

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses in Lawrence County after the housing boom -/
def houses_after : ℕ := 2000

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := houses_after - houses_before

theorem housing_boom_construction : houses_built = 574 := by
  sorry

end housing_boom_construction_l3826_382625


namespace ratio_problem_l3826_382666

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 1.875)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a / b = 1.40625 := by
  sorry

end ratio_problem_l3826_382666


namespace chocolate_distribution_l3826_382611

theorem chocolate_distribution (x y : ℕ) : 
  (y = x + 1) →  -- If each person is given 1 chocolate, then 1 chocolate is left
  (y = 2 * (x - 1)) →  -- If each person is given 2 chocolates, then 1 person will be left
  (x + y = 7) :=  -- The sum of persons and chocolates is 7
by
  sorry

#check chocolate_distribution

end chocolate_distribution_l3826_382611


namespace complex_power_six_l3826_382665

theorem complex_power_six : (1 + 2 * Complex.I) ^ 6 = 117 + 44 * Complex.I := by
  sorry

end complex_power_six_l3826_382665


namespace no_common_divisor_l3826_382640

theorem no_common_divisor (a b n : ℕ) 
  (ha : a > 1) 
  (hb : b > 1)
  (hn : n > 0)
  (div_a : a ∣ (2^n - 1))
  (div_b : b ∣ (2^n + 1)) :
  ¬∃ k : ℕ, (a ∣ (2^k + 1)) ∧ (b ∣ (2^k - 1)) :=
sorry

end no_common_divisor_l3826_382640


namespace complex_magnitude_sum_reciprocals_l3826_382694

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end complex_magnitude_sum_reciprocals_l3826_382694


namespace largest_three_digit_geometric_l3826_382655

def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ b = a * r ∧ c = b * r

def digits_are_distinct (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

theorem largest_three_digit_geometric : ∀ n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  digits_are_distinct n ∧
  is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10) ∧
  n / 100 ≤ 8 →
  n ≤ 842 :=
sorry

end largest_three_digit_geometric_l3826_382655


namespace min_value_expression_min_value_achievable_l3826_382629

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) ≥ 4 :=
by sorry

theorem min_value_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ((3*a*b - 6*b + a*(1 - a))^2 + (9*b^2 + 2*a + 3*b*(1 - a))^2) / (a^2 + 9*b^2) = 4 :=
by sorry

end min_value_expression_min_value_achievable_l3826_382629


namespace range_of_m_l3826_382624

theorem range_of_m (x m : ℝ) : 
  (∀ x, (|x - 1| < 2 → -1 < x ∧ x < m + 1) ∧ 
   ∃ x, (-1 < x ∧ x < m + 1 ∧ ¬(|x - 1| < 2))) →
  m > 2 :=
sorry

end range_of_m_l3826_382624


namespace function_inequality_l3826_382601

open Set

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f)
  (h_sym : ∀ x, f x = f (2 - x))
  (h_mono : ∀ x ∈ Iio 1, (x - 1) * deriv f x < 0) :
  f 3 < f 0 ∧ f 0 < f (1/2) := by
sorry

end function_inequality_l3826_382601


namespace stratified_sampling_girls_count_l3826_382678

theorem stratified_sampling_girls_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girl_boy_diff : ℕ) : 
  total_students = 1750 →
  sample_size = 250 →
  girl_boy_diff = 20 →
  ∃ (girls_in_sample : ℕ) (boys_in_sample : ℕ),
    girls_in_sample + boys_in_sample = sample_size ∧
    boys_in_sample = girls_in_sample + girl_boy_diff ∧
    (girls_in_sample : ℚ) / (sample_size : ℚ) = 
      ((total_students - (boys_in_sample * total_students / sample_size)) : ℚ) / (total_students : ℚ) ∧
    total_students - (boys_in_sample * total_students / sample_size) = 805 :=
by sorry

end stratified_sampling_girls_count_l3826_382678


namespace expression_evaluation_l3826_382697

theorem expression_evaluation (a b : ℤ) (ha : a = -4) (hb : b = 3) :
  -2 * a - b^3 + 2 * a * b + b^2 = -34 := by sorry

end expression_evaluation_l3826_382697


namespace root_of_multiplicity_l3826_382668

theorem root_of_multiplicity (k : ℝ) : 
  (∃ x : ℝ, (x - 1) / (x - 3) = k / (x - 3) ∧ 
   ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, |y - x| < δ → 
   |((y - 1) / (y - 3) - k / (y - 3))| < ε * |y - x|) ↔ 
  k = 2 := by
sorry

end root_of_multiplicity_l3826_382668


namespace reflection_squared_is_identity_l3826_382621

open Matrix

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : Fin 2 → ℝ) (h : v ≠ 0) : Matrix (Fin 2) (Fin 2) ℝ := sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity 
  (v : Fin 2 → ℝ) (h : v ≠ 0) :
  (reflection_matrix v h) ^ 2 = 1 := by sorry

end reflection_squared_is_identity_l3826_382621


namespace ellipse_fixed_point_theorem_l3826_382662

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (1, 0)

-- Define the upper vertex
noncomputable def upper_vertex : ℝ × ℝ := (0, 1)

-- Define the slope condition
def slope_condition (M N : ℝ × ℝ) : Prop :=
  let Q := upper_vertex
  let k_QM := (M.2 - Q.2) / (M.1 - Q.1)
  let k_QN := (N.2 - Q.2) / (N.1 - Q.1)
  k_QM + k_QN = 1

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, -1)

-- Theorem statement
theorem ellipse_fixed_point_theorem (M N : ℝ × ℝ) :
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M ≠ upper_vertex →
  N ≠ upper_vertex →
  slope_condition M N →
  ∃ (k t : ℝ), M.2 = k * M.1 + t ∧ N.2 = k * N.1 + t ∧ fixed_point.2 = k * fixed_point.1 + t :=
sorry

end ellipse_fixed_point_theorem_l3826_382662


namespace percentage_problem_l3826_382689

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.1 * 500 - 5 → x = 180 := by
  sorry

end percentage_problem_l3826_382689


namespace smallest_cube_ending_632_l3826_382618

theorem smallest_cube_ending_632 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 632 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 632 [ZMOD 1000] → n ≤ m ∧ n = 192 := by
  sorry

end smallest_cube_ending_632_l3826_382618


namespace champion_is_c_l3826_382626

-- Define the athletes
inductive Athlete : Type
| a : Athlete
| b : Athlete
| c : Athlete

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the correctness of a statement
inductive Correctness : Type
| Correct : Correctness
| HalfCorrect : Correctness
| Incorrect : Correctness

-- Define the champion
def champion : Athlete := Athlete.c

-- Define the statements made by each student
def statement (s : Student) : Athlete × Athlete :=
  match s with
  | Student.A => (Athlete.b, Athlete.c)
  | Student.B => (Athlete.b, Athlete.a)
  | Student.C => (Athlete.c, Athlete.b)

-- Define the correctness of each student's statement
def studentCorrectness (s : Student) : Correctness :=
  match s with
  | Student.A => Correctness.Correct
  | Student.B => Correctness.HalfCorrect
  | Student.C => Correctness.Incorrect

-- Theorem to prove
theorem champion_is_c :
  (∀ s : Student, (statement s).1 ≠ champion → (statement s).2 = champion ↔ studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Correct) ∧
  (∃! s : Student, studentCorrectness s = Correctness.HalfCorrect) ∧
  (∃! s : Student, studentCorrectness s = Correctness.Incorrect) →
  champion = Athlete.c := by
  sorry

end champion_is_c_l3826_382626


namespace semicircle_perimeter_approx_l3826_382677

/-- The perimeter of a semicircle with radius 12 is approximately 61.7 units. -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 12
  let π_approx : ℝ := 3.14159
  let semicircle_perimeter := π_approx * r + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 61.7) < ε :=
by
  sorry

end semicircle_perimeter_approx_l3826_382677


namespace haris_contribution_haris_contribution_is_9720_l3826_382619

/-- Calculates Hari's contribution to the capital given the investment conditions --/
theorem haris_contribution (praveen_investment : ℕ) (praveen_months : ℕ) (hari_months : ℕ) 
  (profit_ratio_praveen : ℕ) (profit_ratio_hari : ℕ) : ℕ :=
  let total_months := praveen_months
  let hari_contribution := (praveen_investment * praveen_months * profit_ratio_hari) / 
                           (hari_months * profit_ratio_praveen)
  hari_contribution

/-- Proves that Hari's contribution is 9720 given the specific conditions --/
theorem haris_contribution_is_9720 : 
  haris_contribution 3780 12 7 2 3 = 9720 := by
  sorry

end haris_contribution_haris_contribution_is_9720_l3826_382619


namespace total_sightings_l3826_382664

def animal_sightings (january february march : ℕ) : Prop :=
  february = 3 * january ∧ march = february / 2

theorem total_sightings (january : ℕ) (h : animal_sightings january (3 * january) ((3 * january) / 2)) :
  january + (3 * january) + ((3 * january) / 2) = 143 :=
by
  sorry

#check total_sightings 26

end total_sightings_l3826_382664


namespace prime_triplets_equation_l3826_382656

theorem prime_triplets_equation (p q r : ℕ) : 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  (p : ℚ) / q = 8 / (r - 1 : ℚ) + 1 ↔ 
  ((p = 3 ∧ q = 2 ∧ r = 17) ∨ 
   (p = 7 ∧ q = 3 ∧ r = 7) ∨ 
   (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end prime_triplets_equation_l3826_382656


namespace right_triangle_perimeter_l3826_382632

-- Define a right-angled triangle with one side of length 11 and the other two sides being natural numbers
def RightTriangle (a b c : ℕ) : Prop :=
  a = 11 ∧ a^2 + b^2 = c^2

-- Define the perimeter of the triangle
def Perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem statement
theorem right_triangle_perimeter :
  ∃ (a b c : ℕ), RightTriangle a b c ∧ Perimeter a b c = 132 :=
sorry

end right_triangle_perimeter_l3826_382632


namespace fort_blocks_count_l3826_382687

/-- Represents the dimensions of a rectangular fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and specifications --/
def calculateFortBlocks (d : FortDimensions) : ℕ :=
  let totalVolume := d.length * d.width * d.height
  let internalLength := d.length - 2
  let internalWidth := d.width - 2
  let internalHeight := d.height - 1
  let internalVolume := internalLength * internalWidth * internalHeight
  let partitionVolume := 1 * internalWidth * internalHeight
  totalVolume - internalVolume + partitionVolume

/-- Theorem stating that a fort with the given dimensions requires 458 blocks --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨14, 12, 6⟩
  calculateFortBlocks fortDims = 458 := by
  sorry

#eval calculateFortBlocks ⟨14, 12, 6⟩

end fort_blocks_count_l3826_382687


namespace real_part_expression1_real_part_expression2_real_part_expression3_l3826_382661

open Complex

-- Define the function f that returns the real part of a complex number
def f (z : ℂ) : ℝ := z.re

-- Theorem 1
theorem real_part_expression1 : f ((1 + 2*I)^2 + 3*(1 - I)) / (2 + I) = 1/5 := by sorry

-- Theorem 2
theorem real_part_expression2 : f (1 + (1 - I) / (1 + I)^2 + (1 + I) / (1 - I)^2) = -1 := by sorry

-- Theorem 3
theorem real_part_expression3 : f (1 + (1 - Complex.I * Real.sqrt 3) / (Real.sqrt 3 + I)^2) = 3/4 := by sorry

end real_part_expression1_real_part_expression2_real_part_expression3_l3826_382661


namespace initial_boys_count_l3826_382620

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := sorry

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := 35

/-- Theorem stating that the initial number of boys is 22 -/
theorem initial_boys_count : initial_boys = 22 := by
  have h : initial_boys + additional_boys = total_boys := sorry
  sorry

end initial_boys_count_l3826_382620


namespace min_value_problem_l3826_382680

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end min_value_problem_l3826_382680


namespace grass_field_length_l3826_382604

/-- Represents a rectangular grass field with a surrounding path. -/
structure GrassField where
  length : ℝ
  width : ℝ
  pathWidth : ℝ

/-- Calculates the area of the path surrounding the grass field. -/
def pathArea (field : GrassField) : ℝ :=
  (field.length + 2 * field.pathWidth) * (field.width + 2 * field.pathWidth) - field.length * field.width

/-- Theorem stating the length of the grass field given specific conditions. -/
theorem grass_field_length : 
  ∀ (field : GrassField),
  field.width = 55 →
  field.pathWidth = 2.5 →
  pathArea field = 1250 →
  field.length = 190 := by
sorry

end grass_field_length_l3826_382604


namespace factorial_sum_equals_720_l3826_382602

def factorial : ℕ → ℕ
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_sum_equals_720 : 
  5 * factorial 5 + 4 * factorial 4 + factorial 4 = 720 := by
sorry

end factorial_sum_equals_720_l3826_382602


namespace complex_multiplication_l3826_382630

theorem complex_multiplication (i : ℂ) (h : i * i = -1) : 
  (-1 + i) * (2 - i) = -1 + 3*i := by sorry

end complex_multiplication_l3826_382630


namespace absolute_difference_100th_terms_l3826_382612

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem absolute_difference_100th_terms :
  let C := arithmetic_sequence 35 7
  let D := arithmetic_sequence 35 (-7)
  |C 100 - D 100| = 1386 := by
sorry

end absolute_difference_100th_terms_l3826_382612


namespace arithmetic_calculations_l3826_382635

theorem arithmetic_calculations :
  (-1^2 + |(-3)| + 5 / (-5) = 1) ∧
  (2 * (-3)^2 + 24 * (1/4 - 3/8 - 1/12) = 4) := by
  sorry

end arithmetic_calculations_l3826_382635


namespace complex_multiplication_l3826_382698

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l3826_382698


namespace angle_bisector_median_inequality_l3826_382615

variable (a b c : ℝ)
variable (s : ℝ)
variable (f₁ f₂ s₃ : ℝ)

/-- Given a triangle with sides a, b, c, semiperimeter s, 
    angle bisectors f₁ and f₂, and median s₃, 
    prove that f₁ + f₂ + s₃ ≤ √3 * s -/
theorem angle_bisector_median_inequality 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : s = (a + b + c) / 2)
  (h_f₁ : f₁^2 = (b * c * ((b + c)^2 - a^2)) / (b + c)^2)
  (h_f₂ : f₂^2 = (a * b * ((a + b)^2 - c^2)) / (a + b)^2)
  (h_s₃ : (2 * s₃)^2 = 2 * a^2 + 2 * c^2 - b^2) :
  f₁ + f₂ + s₃ ≤ Real.sqrt 3 * s :=
sorry

end angle_bisector_median_inequality_l3826_382615


namespace annual_population_increase_rate_l3826_382633

theorem annual_population_increase_rate (initial_population final_population : ℕ) 
  (h : initial_population = 14000 ∧ final_population = 16940) : 
  ∃ r : ℝ, initial_population * (1 + r)^2 = final_population := by
  sorry

end annual_population_increase_rate_l3826_382633


namespace exists_number_with_property_l3826_382699

-- Define small numbers
def isSmall (n : ℕ) : Prop := n ≤ 150

-- Define the property we're looking for
def hasProperty (N : ℕ) : Prop :=
  ∃ (a b : ℕ), isSmall a ∧ isSmall b ∧ b = a + 1 ∧
  ¬(N % a = 0) ∧ ¬(N % b = 0) ∧
  ∀ k, isSmall k → k ≠ a → k ≠ b → N % k = 0

-- Theorem statement
theorem exists_number_with_property :
  ∃ N : ℕ, hasProperty N :=
sorry

end exists_number_with_property_l3826_382699


namespace min_rice_purchase_l3826_382608

theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + 2 * r) 
  (h2 : o ≤ 3 * r) : 
  r ≥ 4 := by
sorry

end min_rice_purchase_l3826_382608


namespace problem_statement_l3826_382683

theorem problem_statement (a b c : ℕ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a ≥ b ∧ b ≥ c) (h3 : Nat.Prime ((a - c) / 2))
  (h4 : a^2 + b^2 + c^2 - 2*(a*b + b*c + c*a) = b) :
  Nat.Prime b ∨ ∃ k : ℕ, b = k^2 := by
sorry

end problem_statement_l3826_382683


namespace curve_properties_l3826_382669

-- Define the curve equation
def curve_equation (x y t : ℝ) : Prop :=
  x^2 / (4 - t) + y^2 / (t - 1) = 1

-- Define what it means for the curve to be a hyperbola
def is_hyperbola (t : ℝ) : Prop :=
  t < 1 ∨ t > 4

-- Define what it means for the curve to be an ellipse with foci on the x-axis
def is_ellipse_x_foci (t : ℝ) : Prop :=
  1 < t ∧ t < 5/2

-- State the theorem
theorem curve_properties :
  ∀ t : ℝ,
    (∀ x y : ℝ, curve_equation x y t → is_hyperbola t) ∧
    (∀ x y : ℝ, curve_equation x y t → is_ellipse_x_foci t) :=
by sorry

end curve_properties_l3826_382669


namespace tan_strictly_increasing_interval_l3826_382609

theorem tan_strictly_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x ↦ Real.tan (2 * x - π / 3))
    (Set.Ioo (k * π / 2 - π / 12) (k * π / 2 + 5 * π / 12)) := by
  sorry

end tan_strictly_increasing_interval_l3826_382609


namespace original_savings_calculation_l3826_382658

theorem original_savings_calculation (savings : ℚ) : 
  (3 / 4 : ℚ) * savings + (1 / 4 : ℚ) * savings = savings ∧ 
  (1 / 4 : ℚ) * savings = 240 → 
  savings = 960 := by
  sorry

end original_savings_calculation_l3826_382658


namespace largest_N_is_120_l3826_382627

/-- A type representing a 6 × N table with entries from 1 to 6 -/
def Table (N : ℕ) := Fin 6 → Fin N → Fin 6

/-- Predicate to check if a column is a permutation of 1 to 6 -/
def IsPermutation (t : Table N) (col : Fin N) : Prop :=
  ∀ i : Fin 6, ∃ j : Fin 6, t j col = i

/-- Predicate to check if any two columns have a common entry in some row -/
def HasCommonEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ r : Fin 6, t r i = t r j

/-- Predicate to check if any two columns have a different entry in some row -/
def HasDifferentEntry (t : Table N) : Prop :=
  ∀ i j : Fin N, i ≠ j → ∃ s : Fin 6, t s i ≠ t s j

/-- The main theorem stating the largest possible N -/
theorem largest_N_is_120 :
  (∃ N : ℕ, N > 0 ∧ ∃ t : Table N,
    (∀ col, IsPermutation t col) ∧
    HasCommonEntry t ∧
    HasDifferentEntry t) ∧
  (∀ M : ℕ, M > 120 →
    ¬∃ t : Table M,
      (∀ col, IsPermutation t col) ∧
      HasCommonEntry t ∧
      HasDifferentEntry t) :=
sorry

end largest_N_is_120_l3826_382627


namespace optimal_transport_plan_l3826_382695

/-- Represents the transportation problem for fruits A, B, and C -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ

/-- Calculates the profit for a given transportation plan -/
def calculate_profit (ft : FruitTransport) (trucks_A trucks_B trucks_C : ℕ) : ℕ :=
  trucks_A * ft.tons_per_truck_A * ft.profit_per_ton_A +
  trucks_B * ft.tons_per_truck_B * ft.profit_per_ton_B +
  trucks_C * ft.tons_per_truck_C * ft.profit_per_ton_C

/-- The main theorem stating the optimal transportation plan and maximum profit -/
theorem optimal_transport_plan (ft : FruitTransport)
  (h1 : ft.total_trucks = 20)
  (h2 : ft.total_tons = 100)
  (h3 : ft.tons_per_truck_A = 6)
  (h4 : ft.tons_per_truck_B = 5)
  (h5 : ft.tons_per_truck_C = 4)
  (h6 : ft.profit_per_ton_A = 500)
  (h7 : ft.profit_per_ton_B = 600)
  (h8 : ft.profit_per_ton_C = 400) :
  ∃ (trucks_A trucks_B trucks_C : ℕ),
    trucks_A + trucks_B + trucks_C = ft.total_trucks ∧
    trucks_A * ft.tons_per_truck_A + trucks_B * ft.tons_per_truck_B + trucks_C * ft.tons_per_truck_C = ft.total_tons ∧
    trucks_A ≥ 2 ∧ trucks_B ≥ 2 ∧ trucks_C ≥ 2 ∧
    trucks_A = 2 ∧ trucks_B = 16 ∧ trucks_C = 2 ∧
    calculate_profit ft trucks_A trucks_B trucks_C = 57200 ∧
    ∀ (a b c : ℕ), a + b + c = ft.total_trucks →
      a * ft.tons_per_truck_A + b * ft.tons_per_truck_B + c * ft.tons_per_truck_C = ft.total_tons →
      a ≥ 2 → b ≥ 2 → c ≥ 2 →
      calculate_profit ft a b c ≤ calculate_profit ft trucks_A trucks_B trucks_C :=
by sorry

end optimal_transport_plan_l3826_382695


namespace matrix_power_2023_l3826_382649

def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A^2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l3826_382649


namespace race_time_calculation_l3826_382693

/-- Given a race where runner A beats runner B by both distance and time, 
    this theorem proves the time taken by runner A to complete the race. -/
theorem race_time_calculation (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 1000 ∧ 
  distance_diff = 48 ∧ 
  time_diff = 12 →
  ∃ (time_A : ℝ), time_A = 250 ∧ 
    race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff) :=
by sorry

end race_time_calculation_l3826_382693


namespace triangle_abc_properties_l3826_382646

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 →
  A = π/3 ∧ a = 2 := by
sorry


end triangle_abc_properties_l3826_382646


namespace arithmetic_to_geometric_sequence_l3826_382617

theorem arithmetic_to_geometric_sequence (a₁ a₂ a₃ a₄ d : ℝ) : 
  d ≠ 0 →
  a₂ = a₁ + d →
  a₃ = a₁ + 2*d →
  a₄ = a₁ + 3*d →
  ((a₂^2 = a₁ * a₃) ∨ (a₂^2 = a₁ * a₄) ∨ (a₃^2 = a₁ * a₄) ∨ (a₃^2 = a₂ * a₄)) →
  (a₁ / d = -4 ∨ a₁ / d = 1) :=
by sorry

end arithmetic_to_geometric_sequence_l3826_382617


namespace negative_slope_decreasing_l3826_382692

/-- A linear function with negative slope -/
structure NegativeSlopeLinearFunction where
  k : ℝ
  b : ℝ
  h : k < 0

/-- The function corresponding to a NegativeSlopeLinearFunction -/
def NegativeSlopeLinearFunction.toFun (f : NegativeSlopeLinearFunction) : ℝ → ℝ := 
  fun x ↦ f.k * x + f.b

theorem negative_slope_decreasing (f : NegativeSlopeLinearFunction) 
    (x₁ x₂ : ℝ) (h : x₁ < x₂) : 
    f.toFun x₁ > f.toFun x₂ := by
  sorry

end negative_slope_decreasing_l3826_382692


namespace count_pairs_eq_nine_l3826_382607

/-- The number of distinct ordered pairs of positive integers (x, y) such that 1/x + 1/y = 1/6 -/
def count_pairs : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card

theorem count_pairs_eq_nine : count_pairs = 9 := by
  sorry

end count_pairs_eq_nine_l3826_382607


namespace derivative_f_at_4_l3826_382628

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem derivative_f_at_4 : 
  deriv f 4 = -1/16 := by sorry

end derivative_f_at_4_l3826_382628


namespace initial_tank_capacity_initial_tank_capacity_solution_l3826_382600

theorem initial_tank_capacity 
  (initial_tanks : ℕ) 
  (additional_tanks : ℕ) 
  (fish_per_additional_tank : ℕ) 
  (total_fish : ℕ) : ℕ :=
  let fish_in_additional_tanks := additional_tanks * fish_per_additional_tank
  let remaining_fish := total_fish - fish_in_additional_tanks
  remaining_fish / initial_tanks

theorem initial_tank_capacity_solution 
  (h1 : initial_tanks = 3)
  (h2 : additional_tanks = 3)
  (h3 : fish_per_additional_tank = 10)
  (h4 : total_fish = 75) :
  initial_tank_capacity initial_tanks additional_tanks fish_per_additional_tank total_fish = 15 := by
  sorry

end initial_tank_capacity_initial_tank_capacity_solution_l3826_382600


namespace line_slope_l3826_382690

/-- Given a line with equation y - 3 = 4(x + 1), its slope is 4 -/
theorem line_slope (x y : ℝ) : y - 3 = 4 * (x + 1) → (y - 3) / (x - (-1)) = 4 := by
  sorry

end line_slope_l3826_382690


namespace arrow_symmetry_axis_l3826_382606

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a geometric figure on a grid --/
structure GeometricFigure where
  points : Set GridPoint

/-- Represents a line on a grid --/
inductive GridLine
  | Vertical : Int → GridLine
  | Horizontal : Int → GridLine
  | DiagonalTopLeftToBottomRight : GridLine
  | DiagonalBottomLeftToTopRight : GridLine

/-- Predicate to check if a figure is arrow-shaped --/
def isArrowShaped (figure : GeometricFigure) : Prop := sorry

/-- Predicate to check if a line is an axis of symmetry for a figure --/
def isAxisOfSymmetry (line : GridLine) (figure : GeometricFigure) : Prop := sorry

/-- Theorem: An arrow-shaped figure with only one axis of symmetry has a vertical line through the center as its axis of symmetry --/
theorem arrow_symmetry_axis (figure : GeometricFigure) (h1 : isArrowShaped figure) 
    (h2 : ∃! (line : GridLine), isAxisOfSymmetry line figure) : 
    ∃ (x : Int), isAxisOfSymmetry (GridLine.Vertical x) figure := by
  sorry

end arrow_symmetry_axis_l3826_382606


namespace sum_of_fractions_l3826_382605

theorem sum_of_fractions : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by
  sorry

end sum_of_fractions_l3826_382605


namespace square_diameter_double_area_l3826_382634

theorem square_diameter_double_area (d₁ : ℝ) (d₂ : ℝ) : 
  d₁ = 4 * Real.sqrt 2 →
  (d₂ / 2)^2 = 2 * (d₁ / 2)^2 →
  d₂ = 8 :=
by sorry

end square_diameter_double_area_l3826_382634


namespace friends_assignment_l3826_382643

-- Define the types for names, surnames, and grades
inductive Name : Type
  | Petya | Kolya | Alyosha | Misha | Dima | Borya | Vasya

inductive Surname : Type
  | Ivanov | Petrov | Krylov | Orlov

inductive Grade : Type
  | First | Second | Third | Fourth

-- Define a function to represent the assignment of names, surnames, and grades
def Assignment := Name → Surname × Grade

-- Define the conditions
def not_first_grader (a : Assignment) (n : Name) : Prop :=
  (a n).2 ≠ Grade.First

def different_streets (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 ≠ (a n2).1

def one_year_older (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

def neighbors (a : Assignment) (n1 n2 : Name) : Prop :=
  (a n1).1 = (a n2).1

def met_year_ago_first_grade (a : Assignment) (n : Name) : Prop :=
  (a n).2 = Grade.Second

def gave_last_year_textbook (a : Assignment) (n1 n2 : Name) : Prop :=
  match (a n1).2, (a n2).2 with
  | Grade.Second, Grade.First => True
  | Grade.Third, Grade.Second => True
  | Grade.Fourth, Grade.Third => True
  | _, _ => False

-- Define the theorem
theorem friends_assignment (a : Assignment) :
  not_first_grader a Name.Borya
  ∧ different_streets a Name.Vasya Name.Dima
  ∧ one_year_older a Name.Misha Name.Dima
  ∧ neighbors a Name.Borya Name.Vasya
  ∧ met_year_ago_first_grade a Name.Misha
  ∧ gave_last_year_textbook a Name.Vasya Name.Borya
  → a Name.Dima = (Surname.Ivanov, Grade.First)
  ∧ a Name.Misha = (Surname.Krylov, Grade.Second)
  ∧ a Name.Borya = (Surname.Petrov, Grade.Third)
  ∧ a Name.Vasya = (Surname.Orlov, Grade.Fourth) :=
by
  sorry

end friends_assignment_l3826_382643


namespace strawberry_rows_l3826_382663

/-- Given that each row of strawberry plants produces 268 kg of fruit
    and the total harvest is 1876 kg, prove that there are 7 rows of strawberry plants. -/
theorem strawberry_rows (yield_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : yield_per_row = 268)
  (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := by
  sorry

end strawberry_rows_l3826_382663


namespace angle_x_value_l3826_382671

theorem angle_x_value (equilateral_angle : ℝ) (isosceles_vertex : ℝ) (straight_line_sum : ℝ) :
  equilateral_angle = 60 →
  isosceles_vertex = 30 →
  straight_line_sum = 180 →
  ∃ x y : ℝ,
    y + y + isosceles_vertex = straight_line_sum ∧
    x + y + equilateral_angle = straight_line_sum ∧
    x = 45 :=
by sorry

end angle_x_value_l3826_382671


namespace john_share_l3826_382654

/-- Given a total amount and a ratio, calculates the share for a specific part -/
def calculateShare (totalAmount : ℚ) (ratio : List ℚ) (part : ℚ) : ℚ :=
  let totalParts := ratio.sum
  let valuePerPart := totalAmount / totalParts
  valuePerPart * part

/-- Proves that given a total amount of 4200 and a ratio of 2:4:6:8, 
    the amount received by the person with 2 parts is 420 -/
theorem john_share : 
  let totalAmount : ℚ := 4200
  let ratio : List ℚ := [2, 4, 6, 8]
  calculateShare totalAmount ratio 2 = 420 := by sorry

end john_share_l3826_382654


namespace missing_term_in_geometric_sequence_l3826_382651

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem missing_term_in_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_first : a 1 = 2)
  (h_second : a 2 = 6)
  (h_third : a 3 = 18)
  (h_fourth : a 4 = 54)
  (h_sixth : a 6 = 486) :
  a 5 = 162 := by
sorry


end missing_term_in_geometric_sequence_l3826_382651


namespace inequality_solution_abs_inequality_l3826_382673

def f (x : ℝ) := |x - 2|

theorem inequality_solution :
  ∀ x : ℝ, (f x + f (x + 1) ≥ 5) ↔ (x ≥ 4 ∨ x ≤ -1) :=
sorry

theorem abs_inequality :
  ∀ a b : ℝ, |a| > 1 → |a*b - 2| > |a| * |b/a - 2| → |b| > 2 :=
sorry

end inequality_solution_abs_inequality_l3826_382673


namespace probability_under20_is_one_tenth_l3826_382636

/-- Represents a group of people with age distribution --/
structure AgeGroup where
  total : ℕ
  over30 : ℕ
  under20 : ℕ
  h1 : total = over30 + under20
  h2 : over30 < total

/-- Calculates the probability of selecting a person under 20 from the group --/
def probabilityUnder20 (group : AgeGroup) : ℚ :=
  group.under20 / group.total

/-- The main theorem to prove --/
theorem probability_under20_is_one_tenth
  (group : AgeGroup)
  (h3 : group.total = 100)
  (h4 : group.over30 = 90) :
  probabilityUnder20 group = 1/10 := by
  sorry


end probability_under20_is_one_tenth_l3826_382636


namespace max_triangles_theorem_max_squares_theorem_l3826_382685

/-- The maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
def max_triangles_in_triangle : ℕ := 6

/-- The maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
def max_squares_in_square : ℕ := 8

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel triangles that can be fitted into a triangle -/
theorem max_triangles_theorem : max_triangles_in_triangle = 6 := by sorry

/-- Theorem stating the maximum number of identical, non-overlapping, and parallel squares that can be fitted into a square -/
theorem max_squares_theorem : max_squares_in_square = 8 := by sorry

end max_triangles_theorem_max_squares_theorem_l3826_382685


namespace prime_square_mod_twelve_l3826_382657

theorem prime_square_mod_twelve (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end prime_square_mod_twelve_l3826_382657


namespace min_side_length_triangle_l3826_382644

theorem min_side_length_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b = 2 →
  C = 2 * π / 3 →
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) →
  c ≥ Real.sqrt 3 ∧ (c = Real.sqrt 3 ↔ a = b) := by
sorry

end min_side_length_triangle_l3826_382644


namespace max_ratio_on_circle_l3826_382688

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Definition of the circle x^2 + y^2 = 25 -/
def on_circle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 25

/-- Definition of irrational distance between two points -/
def irrational_distance (p q : IntPoint) : Prop :=
  ∃ (d : ℝ), d^2 = (p.x - q.x)^2 + (p.y - q.y)^2 ∧ Irrational d

/-- Theorem statement -/
theorem max_ratio_on_circle (P Q R S : IntPoint) :
  on_circle P → on_circle Q → on_circle R → on_circle S →
  irrational_distance P Q → irrational_distance R S →
  ∃ (ratio : ℝ), (∀ (d_PQ d_RS : ℝ),
    d_PQ^2 = (P.x - Q.x)^2 + (P.y - Q.y)^2 →
    d_RS^2 = (R.x - S.x)^2 + (R.y - S.y)^2 →
    d_PQ / d_RS ≤ ratio) ∧
  ratio = 5 * Real.sqrt 2 :=
sorry

end max_ratio_on_circle_l3826_382688


namespace grace_earnings_l3826_382686

/-- Calculates the number of weeks needed to earn a target amount given a weekly rate and biweekly payment schedule. -/
def weeksToEarn (weeklyRate : ℕ) (targetAmount : ℕ) : ℕ :=
  let biweeklyEarnings := weeklyRate * 2
  let numPayments := targetAmount / biweeklyEarnings
  numPayments * 2

/-- Proves that it takes 6 weeks to earn 1800 dollars with a weekly rate of 300 dollars and biweekly payments. -/
theorem grace_earnings : weeksToEarn 300 1800 = 6 := by
  sorry

end grace_earnings_l3826_382686


namespace fraction_equals_seven_l3826_382645

theorem fraction_equals_seven : (2^2016 + 3 * 2^2014) / (2^2016 - 3 * 2^2014) = 7 := by
  sorry

end fraction_equals_seven_l3826_382645


namespace number_with_special_divisor_property_l3826_382603

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def divisor_difference_property (n : ℕ) : Prop :=
  ∀ d₁ d₂ : ℕ, proper_divisor d₁ n → proper_divisor d₂ n → d₁ ≠ d₂ → (d₁ - d₂) ∣ n

theorem number_with_special_divisor_property (n : ℕ) :
  n ≥ 2 →
  (∃ (d₁ d₂ d₃ d₄ : ℕ), d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
    ∀ d : ℕ, d ∣ n → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) →
  divisor_difference_property n →
  n = 4 ∨ is_prime n :=
sorry

end number_with_special_divisor_property_l3826_382603


namespace sin_15_cos_15_eq_quarter_l3826_382659

theorem sin_15_cos_15_eq_quarter : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/4 := by
  sorry

end sin_15_cos_15_eq_quarter_l3826_382659


namespace expression_evaluation_l3826_382670

theorem expression_evaluation :
  let x : ℝ := 3
  let y : ℝ := Real.sqrt 3
  (x - 2*y)^2 - (x + 2*y)*(x - 2*y) + 4*x*y = 24 := by
sorry

end expression_evaluation_l3826_382670


namespace watermelon_price_per_pound_l3826_382648

/-- The price per pound of watermelons sold by Farmer Kent -/
def price_per_pound (watermelon_weight : ℕ) (num_watermelons : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (watermelon_weight * num_watermelons)

/-- Theorem stating that the price per pound of Farmer Kent's watermelons is $2 -/
theorem watermelon_price_per_pound :
  price_per_pound 23 18 828 = 2 := by
  sorry

end watermelon_price_per_pound_l3826_382648


namespace hilt_snow_amount_l3826_382684

/-- The amount of snow at Brecknock Elementary School in inches -/
def school_snow : ℕ := 17

/-- The additional amount of snow at Mrs. Hilt's house compared to the school in inches -/
def additional_snow : ℕ := 12

/-- The total amount of snow at Mrs. Hilt's house in inches -/
def hilt_snow : ℕ := school_snow + additional_snow

/-- Theorem stating that the amount of snow at Mrs. Hilt's house is 29 inches -/
theorem hilt_snow_amount : hilt_snow = 29 := by sorry

end hilt_snow_amount_l3826_382684


namespace point_in_fourth_quadrant_l3826_382638

def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : 
  let x : ℝ := 1
  let y : ℝ := -5
  fourth_quadrant x y :=
by
  sorry

end point_in_fourth_quadrant_l3826_382638


namespace coat_price_reduction_l3826_382691

/-- Given a coat with an original price and a reduction amount, calculate the percent reduction. -/
theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 200) :
  (reduction_amount / original_price) * 100 = 40 := by
  sorry

end coat_price_reduction_l3826_382691


namespace number_of_students_l3826_382637

def candy_bar_cost : ℚ := 2
def chips_cost : ℚ := 1/2

def student_purchase_cost : ℚ := candy_bar_cost + 2 * chips_cost

def total_cost : ℚ := 15

theorem number_of_students : 
  (total_cost / student_purchase_cost : ℚ) = 5 := by sorry

end number_of_students_l3826_382637


namespace square_root_of_two_squared_equals_two_l3826_382674

theorem square_root_of_two_squared_equals_two :
  (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end square_root_of_two_squared_equals_two_l3826_382674


namespace regular_nonagon_diagonal_sum_l3826_382631

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
structure RegularNonagon where
  side_length : ℝ
  shortest_diagonal : ℝ
  longest_diagonal : ℝ

/-- 
In a regular nonagon, the longest diagonal is equal to the sum of 
the side length and the shortest diagonal.
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.longest_diagonal = n.side_length + n.shortest_diagonal := by
  sorry

end regular_nonagon_diagonal_sum_l3826_382631


namespace inequality_proof_l3826_382623

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  Real.sqrt x + Real.sqrt y + Real.sqrt z ≥ x * y + y * z + z * x := by
  sorry

end inequality_proof_l3826_382623


namespace initial_girls_on_team_l3826_382650

theorem initial_girls_on_team (initial_boys : ℕ) (girls_joined : ℕ) (boys_quit : ℕ) (final_total : ℕ) :
  initial_boys = 15 →
  girls_joined = 7 →
  boys_quit = 4 →
  final_total = 36 →
  ∃ initial_girls : ℕ, initial_girls + initial_boys = final_total - girls_joined + boys_quit :=
by
  sorry

end initial_girls_on_team_l3826_382650


namespace problem_solution_l3826_382675

theorem problem_solution (a b : ℝ) (h : |a - 1| + |b + 2| = 0) : 
  (a + b)^2013 + |b| = 1 := by
sorry

end problem_solution_l3826_382675


namespace order_of_numbers_l3826_382642

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by sorry

end order_of_numbers_l3826_382642


namespace tv_selection_problem_l3826_382641

theorem tv_selection_problem (type_a : ℕ) (type_b : ℕ) (total_selection : ℕ) :
  type_a = 4 →
  type_b = 5 →
  total_selection = 3 →
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 :=
by sorry

end tv_selection_problem_l3826_382641


namespace tree_height_after_three_years_l3826_382652

/-- Tree growth function -/
def tree_height (initial_height : ℝ) (growth_factor : ℝ) (years : ℕ) : ℝ :=
  initial_height * growth_factor ^ years

theorem tree_height_after_three_years
  (initial_height : ℝ)
  (growth_factor : ℝ)
  (h1 : initial_height = 1)
  (h2 : growth_factor = 3)
  (h3 : tree_height initial_height growth_factor 5 = 243) :
  tree_height initial_height growth_factor 3 = 27 := by
sorry

end tree_height_after_three_years_l3826_382652


namespace closest_to_500_div_025_l3826_382647

def options : List ℝ := [1000, 1500, 2000, 2500, 3000]

theorem closest_to_500_div_025 :
  ∃ (x : ℝ), x ∈ options ∧ 
  ∀ (y : ℝ), y ∈ options → |x - 500/0.25| ≤ |y - 500/0.25| ∧
  x = 2000 :=
by sorry

end closest_to_500_div_025_l3826_382647


namespace difference_of_percentages_l3826_382681

theorem difference_of_percentages : (0.7 * 40) - ((4 / 5) * 25) = 8 := by
  sorry

end difference_of_percentages_l3826_382681


namespace fruit_arrangement_count_l3826_382679

/-- The number of ways to arrange fruits with constraints -/
def fruitArrangements (apples oranges bananas : ℕ) : ℕ :=
  (Nat.factorial (apples + oranges + bananas)) / 
  (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) * 
  (Nat.choose (apples + bananas) apples)

/-- Theorem stating the number of fruit arrangements -/
theorem fruit_arrangement_count :
  fruitArrangements 4 2 2 = 18900 := by
  sorry

end fruit_arrangement_count_l3826_382679
