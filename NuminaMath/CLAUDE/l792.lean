import Mathlib

namespace NUMINAMATH_CALUDE_product_of_four_primes_l792_79257

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the properties of the product of four specific primes -/
theorem product_of_four_primes (A B : ℕ) 
  (hA : isPrime A) 
  (hB : isPrime B) 
  (hAminusB : isPrime (A - B)) 
  (hAplusB : isPrime (A + B)) : 
  ∃ (p : ℕ), p = A * B * (A - B) * (A + B) ∧ Even p ∧ p % 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_product_of_four_primes_l792_79257


namespace NUMINAMATH_CALUDE_real_part_of_z_l792_79284

theorem real_part_of_z (z : ℂ) (h1 : Complex.abs z = 1) (h2 : Complex.abs (z - 1.45) = 1.05) :
  z.re = 20 / 29 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l792_79284


namespace NUMINAMATH_CALUDE_min_value_inequality_l792_79281

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a^2 - 1) * (1 / b^2 - 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l792_79281


namespace NUMINAMATH_CALUDE_angle_subtraction_l792_79216

-- Define a structure for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨20, 18⟩

-- Define the operation to subtract an Angle from 90 degrees
def subtractFrom90 (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- State the theorem
theorem angle_subtraction :
  subtractFrom90 angle1 = ⟨69, 42⟩ := by
  sorry


end NUMINAMATH_CALUDE_angle_subtraction_l792_79216


namespace NUMINAMATH_CALUDE_sum_10_terms_formula_l792_79209

/-- An arithmetic progression with sum of 4th and 12th terms equal to 20 -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 20

/-- The sum of the first 10 terms of the arithmetic progression -/
def sum_10_terms (ap : ArithmeticProgression) : ℝ :=
  5 * (2*ap.a + 9*ap.d)

/-- Theorem: The sum of the first 10 terms equals 100 - 25d -/
theorem sum_10_terms_formula (ap : ArithmeticProgression) :
  sum_10_terms ap = 100 - 25*ap.d := by
  sorry

end NUMINAMATH_CALUDE_sum_10_terms_formula_l792_79209


namespace NUMINAMATH_CALUDE_G_in_third_quadrant_implies_x_negative_l792_79206

/-- A point in the rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The specific point G(x, x-5) -/
def G (x : ℝ) : Point :=
  { x := x, y := x - 5 }

theorem G_in_third_quadrant_implies_x_negative (x : ℝ) :
  isInThirdQuadrant (G x) → x < 0 := by
  sorry

end NUMINAMATH_CALUDE_G_in_third_quadrant_implies_x_negative_l792_79206


namespace NUMINAMATH_CALUDE_hyperbola_solutions_l792_79291

-- Define the hyperbola equation
def hyperbola (x y : ℤ) : Prop := x^2 - y^2 = 2500^2

-- Define a function to count the number of integer solutions
def count_solutions : ℕ := sorry

-- Theorem stating that the number of solutions is 70
theorem hyperbola_solutions : count_solutions = 70 := by sorry

end NUMINAMATH_CALUDE_hyperbola_solutions_l792_79291


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l792_79274

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l792_79274


namespace NUMINAMATH_CALUDE_symmetry_properties_l792_79269

noncomputable def f (a b c p q x : ℝ) : ℝ := (a * 4^x + b * 2^x + c) / (p * 2^x + q)

theorem symmetry_properties (a b c p q : ℝ) :
  (p = 0 ∧ q ≠ 0 ∧ a^2 + b^2 ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∧
    ¬(∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p ≠ 0 ∧ q = 0 ∧ a * c ≠ 0 →
    (∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x)) ∨
    (∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t)) ∧
  (p * q ≠ 0 ∧ a = 0 ∧ b^2 + c^2 ≠ 0 →
    ∃t, ∀x, f a b c p q (t + x) + f a b c p q (t - x) = 2 * f a b c p q t) ∧
  (p * q ≠ 0 ∧ a ≠ 0 →
    ¬(∃t, ∀x, f a b c p q (t + x) = f a b c p q (t - x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_properties_l792_79269


namespace NUMINAMATH_CALUDE_alices_favorite_number_l792_79248

def is_favorite_number (n : ℕ) : Prop :=
  50 ≤ n ∧ n ≤ 100 ∧ 
  n % 11 = 0 ∧
  n % 2 ≠ 0 ∧
  (n / 10 + n % 10) % 5 = 0

theorem alices_favorite_number :
  ∃! n : ℕ, is_favorite_number n ∧ n = 55 := by
sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l792_79248


namespace NUMINAMATH_CALUDE_cube_volume_fourth_power_l792_79201

/-- The volume of a cube with surface area 864 square units, expressed as the fourth power of its side length -/
theorem cube_volume_fourth_power (s : ℝ) (h : 6 * s^2 = 864) : s^4 = 20736 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_fourth_power_l792_79201


namespace NUMINAMATH_CALUDE_product_difference_equality_l792_79298

theorem product_difference_equality : 2012.25 * 2013.75 - 2010.25 * 2015.75 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equality_l792_79298


namespace NUMINAMATH_CALUDE_comic_book_problem_l792_79215

theorem comic_book_problem (initial_books : ℕ) : 
  (initial_books / 2 + 6 = 17) → initial_books = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_comic_book_problem_l792_79215


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_proof_l792_79266

/-- The smallest four-digit number divisible by 2, 3, 8, and 9 -/
def smallest_four_digit_divisible : ℕ := 1008

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_proof :
  is_four_digit smallest_four_digit_divisible ∧
  smallest_four_digit_divisible % 2 = 0 ∧
  smallest_four_digit_divisible % 3 = 0 ∧
  smallest_four_digit_divisible % 8 = 0 ∧
  smallest_four_digit_divisible % 9 = 0 ∧
  ∀ n : ℕ, is_four_digit n →
    n % 2 = 0 → n % 3 = 0 → n % 8 = 0 → n % 9 = 0 →
    n ≥ smallest_four_digit_divisible :=
by sorry

#eval smallest_four_digit_divisible

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_proof_l792_79266


namespace NUMINAMATH_CALUDE_blue_face_prob_is_five_eighths_l792_79247

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  total_is_sum : total_faces = blue_faces + red_faces
  total_is_eight : total_faces = 8

/-- The probability of rolling a blue face on a colored octahedron -/
def blue_face_probability (o : ColoredOctahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Our specific octahedron with 5 blue faces and 3 red faces -/
def our_octahedron : ColoredOctahedron where
  blue_faces := 5
  red_faces := 3
  total_faces := 8
  total_is_sum := by rfl
  total_is_eight := by rfl

/-- Theorem: The probability of rolling a blue face on our octahedron is 5/8 -/
theorem blue_face_prob_is_five_eighths :
  blue_face_probability our_octahedron = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_prob_is_five_eighths_l792_79247


namespace NUMINAMATH_CALUDE_expression_simplification_l792_79275

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l792_79275


namespace NUMINAMATH_CALUDE_power_multiplication_l792_79243

theorem power_multiplication (a : ℕ) : 10^a * 10^(a-1) = 10^(2*a-1) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l792_79243


namespace NUMINAMATH_CALUDE_complex_equation_solution_l792_79251

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l792_79251


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l792_79252

/-- Given a mixture of wine and water, calculate the new percentage of water after adding more water. -/
theorem water_mixture_percentage
  (total_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_water : ℝ)
  (h1 : total_volume = 120)
  (h2 : initial_water_percentage = 20)
  (h3 : added_water = 8) :
  let initial_water := total_volume * (initial_water_percentage / 100)
  let new_water := initial_water + added_water
  let new_total := total_volume + added_water
  new_water / new_total * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l792_79252


namespace NUMINAMATH_CALUDE_electricity_cost_for_1200_watts_l792_79271

/-- Calculates the total cost of electricity usage based on tiered pricing, late fees, and discounts --/
def calculate_electricity_cost (usage : ℕ) : ℚ :=
  let tier1_limit : ℕ := 300
  let tier2_limit : ℕ := 800
  let tier1_rate : ℚ := 4
  let tier2_rate : ℚ := 3.5
  let tier3_rate : ℚ := 3
  let late_fee_tier1 : ℚ := 150
  let late_fee_tier2 : ℚ := 200
  let late_fee_tier3 : ℚ := 250
  let discount_lower : ℕ := 900
  let discount_upper : ℕ := 1100

  let tier1_cost := min usage tier1_limit * tier1_rate
  let tier2_cost := max 0 (min usage tier2_limit - tier1_limit) * tier2_rate
  let tier3_cost := max 0 (usage - tier2_limit) * tier3_rate

  let total_electricity_cost := tier1_cost + tier2_cost + tier3_cost

  let late_fee := 
    if usage ≤ 600 then late_fee_tier1
    else if usage ≤ 1000 then late_fee_tier2
    else late_fee_tier3

  let total_cost := total_electricity_cost + late_fee

  -- No discount applied as usage is not in the 3rd highest quartile

  total_cost

theorem electricity_cost_for_1200_watts :
  calculate_electricity_cost 1200 = 4400 := by
  sorry

end NUMINAMATH_CALUDE_electricity_cost_for_1200_watts_l792_79271


namespace NUMINAMATH_CALUDE_basketball_probability_l792_79253

theorem basketball_probability (p_at_least_one p_hs3 p_pro3 : ℝ) :
  p_at_least_one = 0.9333333333333333 →
  p_hs3 = 1/2 →
  p_pro3 = 1/3 →
  1 - (1 - p_hs3) * (1 - p_pro3) * (1 - 0.8) = p_at_least_one :=
by sorry

end NUMINAMATH_CALUDE_basketball_probability_l792_79253


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l792_79241

theorem necessary_not_sufficient (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, e^a + 2*a = e^b + 3*b → a > b) ∧
  (∃ a b, a > b ∧ e^a + 2*a ≠ e^b + 3*b) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l792_79241


namespace NUMINAMATH_CALUDE_smallest_value_in_different_bases_l792_79285

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

theorem smallest_value_in_different_bases :
  let base_9 := to_decimal [8, 5] 9
  let base_6 := to_decimal [2, 1, 0] 6
  let base_4 := to_decimal [1, 0, 0, 0] 4
  let base_2 := to_decimal [1, 1, 1, 1, 1, 1] 2
  base_2 = min base_9 (min base_6 (min base_4 base_2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_in_different_bases_l792_79285


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l792_79218

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x + 1

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -3 → f x < f y) ∧
  (∀ x y, -3 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ 10) ∧
  (∀ x, f x ≥ -2/3) ∧
  (∃ x, f x = 10) ∧
  (∃ x, f x = -2/3) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l792_79218


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l792_79219

theorem arithmetic_geometric_sequence_relation :
  ∃ (a g : ℕ → ℕ) (n : ℕ),
    (∀ k, a (k + 1) = a k + (a 2 - a 1)) ∧  -- arithmetic sequence
    (∀ k, g (k + 1) = g k * (g 2 / g 1)) ∧  -- geometric sequence
    n = 14 ∧
    a 1 = g 1 ∧ a 2 = g 2 ∧ a 5 = g 3 ∧ a n = g 4 ∧
    g 1 + g 2 + g 3 + g 4 = 80 ∧
    a 1 = 2 ∧ a 2 = 6 ∧ a 5 = 18 ∧ a n = 54 ∧
    g 1 = 2 ∧ g 2 = 6 ∧ g 3 = 18 ∧ g 4 = 54 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_relation_l792_79219


namespace NUMINAMATH_CALUDE_m_plus_p_equals_11_l792_79262

/-- Sum of odd numbers from 1 to n -/
def sumOddNumbers (n : ℕ) : ℕ :=
  (n + 1) * n / 2

/-- Decomposition of p^3 for positive integers p ≥ 2 -/
def decompositionP3 (p : ℕ) : ℕ :=
  2 * p * p - 1

theorem m_plus_p_equals_11 (m p : ℕ) 
  (h1 : m ^ 2 = sumOddNumbers 6)
  (h2 : decompositionP3 p = 21) : 
  m + p = 11 := by
  sorry

#eval sumOddNumbers 6  -- Should output 36
#eval decompositionP3 5  -- Should output 21

end NUMINAMATH_CALUDE_m_plus_p_equals_11_l792_79262


namespace NUMINAMATH_CALUDE_chord_intersection_probability_2010_l792_79255

/-- Given a circle with n distinct points evenly placed around it,
    this function returns the probability that when four distinct points
    are randomly chosen, the chord formed by two of these points
    intersects the chord formed by the other two points. -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  1 / 3

/-- Theorem stating that for a circle with 2010 distinct points,
    the probability of chord intersection is 1/3 -/
theorem chord_intersection_probability_2010 :
  chord_intersection_probability 2010 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_2010_l792_79255


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l792_79282

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∀ (c : ℝ), c > 0 → a^2 + b^2 = c^2 → (1/2) * a * b ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l792_79282


namespace NUMINAMATH_CALUDE_cat_grooming_time_l792_79234

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (
  claws_per_foot : ℕ
  ) (
  feet : ℕ
  ) (
  clip_time : ℕ
  ) (
  ear_clean_time : ℕ
  ) (
  shampoo_time : ℕ
  ) : ℕ :=
  let total_claws := claws_per_foot * feet
  let total_clip_time := total_claws * clip_time
  let total_ear_clean_time := 2 * ear_clean_time
  let total_shampoo_time := shampoo_time * 60
  total_clip_time + total_ear_clean_time + total_shampoo_time

theorem cat_grooming_time :
  total_grooming_time 4 4 10 90 5 = 640 := by
  sorry

end NUMINAMATH_CALUDE_cat_grooming_time_l792_79234


namespace NUMINAMATH_CALUDE_collinear_vectors_imply_fixed_point_l792_79228

/-- Two 2D vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 = t * w.1 ∧ v.2 = t * w.2

/-- A point (x, y) is on a line y = mx + c if y = mx + c -/
def on_line (m c : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + c

theorem collinear_vectors_imply_fixed_point (k b : ℝ) :
  collinear (k + 2, 1) (-b, 1) →
  on_line k b (1, -2) :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_imply_fixed_point_l792_79228


namespace NUMINAMATH_CALUDE_rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l792_79220

theorem rational_product_sum (a b : ℚ) : 
  a * b = 6 → a + b ≠ 0 := by sorry

theorem integer_sum_product_max (a b : ℤ) : 
  a + b = -5 → a * b ≤ 6 := by sorry

theorem rational_product_negative_sum_comparison (a b : ℚ) : 
  a * b < 0 → 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y < 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y = 0) ∧ 
  (∃ (x y : ℚ), x * y < 0 ∧ x + y > 0) := by sorry

end NUMINAMATH_CALUDE_rational_product_sum_integer_sum_product_max_rational_product_negative_sum_comparison_l792_79220


namespace NUMINAMATH_CALUDE_negative_six_times_negative_one_l792_79208

theorem negative_six_times_negative_one : (-6 : ℤ) * (-1 : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_negative_six_times_negative_one_l792_79208


namespace NUMINAMATH_CALUDE_probability_of_letter_selection_l792_79226

theorem probability_of_letter_selection (total_letters : ℕ) (unique_letters : ℕ) 
  (h1 : total_letters = 26) (h2 : unique_letters = 8) :
  (unique_letters : ℚ) / total_letters = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_selection_l792_79226


namespace NUMINAMATH_CALUDE_min_value_theorem_l792_79254

theorem min_value_theorem (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq : p + q + r + s + t + u = 11) :
  (3/p + 12/q + 27/r + 48/s + 75/t + 108/u) ≥ 819/11 ∧ 
  ∃ (p' q' r' s' t' u' : ℝ), 
    p' > 0 ∧ q' > 0 ∧ r' > 0 ∧ s' > 0 ∧ t' > 0 ∧ u' > 0 ∧
    p' + q' + r' + s' + t' + u' = 11 ∧
    (3/p' + 12/q' + 27/r' + 48/s' + 75/t' + 108/u') = 819/11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l792_79254


namespace NUMINAMATH_CALUDE_seven_eighths_of_48_l792_79223

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end NUMINAMATH_CALUDE_seven_eighths_of_48_l792_79223


namespace NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l792_79278

theorem alpha_beta_difference_bounds (α β : ℝ) (h : -1 < α ∧ α < β ∧ β < 1) :
  -2 < α - β ∧ α - β < 0 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_difference_bounds_l792_79278


namespace NUMINAMATH_CALUDE_stating_non_parallel_necessary_not_sufficient_l792_79273

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Represents a system of two linear equations -/
structure LinearSystem where
  line1 : Line
  line2 : Line

/-- Checks if a linear system has a unique solution -/
def has_unique_solution (sys : LinearSystem) : Prop :=
  sys.line1.a * sys.line2.b ≠ sys.line1.b * sys.line2.a

/-- 
Theorem stating that non-parallel lines are a necessary but insufficient condition
for a system of two linear equations to have a unique solution
-/
theorem non_parallel_necessary_not_sufficient (sys : LinearSystem) :
  has_unique_solution sys → ¬(are_parallel sys.line1 sys.line2) ∧
  ¬(¬(are_parallel sys.line1 sys.line2) → has_unique_solution sys) :=
by sorry

end NUMINAMATH_CALUDE_stating_non_parallel_necessary_not_sufficient_l792_79273


namespace NUMINAMATH_CALUDE_allStarSeatingArrangements_l792_79227

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

-- Define the number of All-Stars for each team
def cubs : ℕ := 4
def redSox : ℕ := 3
def yankees : ℕ := 2

-- Define the total number of All-Stars
def totalAllStars : ℕ := cubs + redSox + yankees

-- Define the number of team blocks (excluding the fixed block)
def remainingTeamBlocks : ℕ := 2

theorem allStarSeatingArrangements :
  factorial remainingTeamBlocks * factorial cubs * factorial redSox * factorial yankees = 576 := by
  sorry

end NUMINAMATH_CALUDE_allStarSeatingArrangements_l792_79227


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l792_79232

/-- The line 2x + 4y + m = 0 is tangent to the parabola y^2 = 16x if and only if m = 32 -/
theorem line_tangent_to_parabola (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 4 * y + m = 0 → y^2 = 16 * x) ∧
  (∃! p : ℝ × ℝ, 2 * p.1 + 4 * p.2 + m = 0 ∧ p.2^2 = 16 * p.1) ↔
  m = 32 := by sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l792_79232


namespace NUMINAMATH_CALUDE_ducks_drinking_order_l792_79277

theorem ducks_drinking_order (total_ducks : ℕ) (ducks_before_a : ℕ) (ducks_after_a : ℕ) :
  total_ducks = 20 →
  ducks_before_a = 11 →
  ducks_after_a = total_ducks - (ducks_before_a + 1) →
  ducks_after_a = 8 :=
by sorry

end NUMINAMATH_CALUDE_ducks_drinking_order_l792_79277


namespace NUMINAMATH_CALUDE_largest_solution_and_fraction_l792_79296

theorem largest_solution_and_fraction (a b c d : ℤ) : 
  (∃ x : ℚ, (5 * x) / 6 + 1 = 3 / x ∧ 
             x = (a + b * Real.sqrt c) / d ∧ 
             ∀ y : ℚ, (5 * y) / 6 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -55 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_fraction_l792_79296


namespace NUMINAMATH_CALUDE_f_2_eq_125_l792_79268

/-- Horner's method for evaluating polynomials -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^5 + 3x^4 + 2x^3 - 4x + 5 -/
def f (x : ℝ) : ℝ :=
  horner [5, -4, 0, 2, 3, 2] x

theorem f_2_eq_125 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_125_l792_79268


namespace NUMINAMATH_CALUDE_ice_cream_sales_l792_79267

theorem ice_cream_sales (sales : List ℝ) (mean : ℝ) : 
  sales.length = 6 →
  sales = [100, 92, 109, 96, 103, 105] →
  mean = 100.1 →
  (sales.sum + (7 * mean - sales.sum)) / 7 = mean →
  7 * mean - sales.sum = 95.7 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l792_79267


namespace NUMINAMATH_CALUDE_side_c_value_l792_79283

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  -- Add conditions for a valid triangle if necessary
  true

-- State the theorem
theorem side_c_value 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_b : b = 3)
  (h_a : a = Real.sqrt 3)
  (h_A : A = 30 * π / 180) -- Convert 30° to radians
  : c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_side_c_value_l792_79283


namespace NUMINAMATH_CALUDE_max_students_distribution_l792_79229

theorem max_students_distribution (num_pens num_pencils : ℕ) :
  let max_students := Nat.gcd num_pens num_pencils
  ∃ (pens_per_student pencils_per_student : ℕ),
    num_pens = max_students * pens_per_student ∧
    num_pencils = max_students * pencils_per_student ∧
    ∀ (n : ℕ),
      (∃ (p q : ℕ), num_pens = n * p ∧ num_pencils = n * q) →
      n ≤ max_students :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l792_79229


namespace NUMINAMATH_CALUDE_rod_friction_coefficient_l792_79202

noncomputable def coefficient_of_friction (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) : ℝ :=
  (1 - initial_normal_force_ratio * Real.cos tilt_angle) / (initial_normal_force_ratio * Real.sin tilt_angle)

theorem rod_friction_coefficient (initial_normal_force_ratio : ℝ) (tilt_angle : ℝ) 
  (h1 : initial_normal_force_ratio = 11)
  (h2 : tilt_angle = 80 * π / 180) :
  ∃ ε > 0, |coefficient_of_friction initial_normal_force_ratio tilt_angle - 0.17| < ε :=
sorry

end NUMINAMATH_CALUDE_rod_friction_coefficient_l792_79202


namespace NUMINAMATH_CALUDE_fraction_addition_l792_79260

theorem fraction_addition : (1 : ℚ) / 3 + (-1 / 2) = -1 / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_addition_l792_79260


namespace NUMINAMATH_CALUDE_carol_position_after_2304_moves_l792_79264

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a direction in the hexagonal grid -/
inductive Direction
  | North
  | NorthEast
  | SouthEast
  | South
  | SouthWest
  | NorthWest

/-- Represents Carol's movement pattern -/
def carolPattern (cycle : ℕ) : List (Direction × ℕ) :=
  [(Direction.North, cycle + 1),
   (Direction.NorthEast, cycle + 1),
   (Direction.SouthEast, cycle + 2),
   (Direction.South, cycle + 2),
   (Direction.SouthWest, cycle + 3),
   (Direction.NorthWest, cycle + 3)]

/-- Calculates the total steps in a given number of cycles -/
def totalStepsInCycles (k : ℕ) : ℕ :=
  k * (k + 1) + 2 * ((k + 1) * (k + 2))

/-- Theorem: Carol's position after 2304 moves -/
theorem carol_position_after_2304_moves :
  ∃ (finalPos : Point),
    (finalPos.x = 5 * Real.sqrt 3 / 2) ∧
    (finalPos.y = 23.5) ∧
    (∃ (k : ℕ),
      totalStepsInCycles k ≤ 2304 ∧
      totalStepsInCycles (k + 1) > 2304 ∧
      finalPos = -- position after completing k cycles and remaining steps
        let remainingSteps := 2304 - totalStepsInCycles k
        let partialCycle := carolPattern (k + 1)
        -- logic to apply remaining steps using partialCycle
        sorry) := by
  sorry

end NUMINAMATH_CALUDE_carol_position_after_2304_moves_l792_79264


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l792_79239

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l792_79239


namespace NUMINAMATH_CALUDE_equation_solution_l792_79207

theorem equation_solution : ∃! y : ℚ, y ≠ 2 ∧ (7 * y) / (y - 2) - 4 / (y - 2) = 3 / (y - 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l792_79207


namespace NUMINAMATH_CALUDE_molecular_weight_N2O5H3P_l792_79221

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_P : ℝ := 30.97

-- Define the molecular formula
def N_count : ℕ := 2
def O_count : ℕ := 5
def H_count : ℕ := 3
def P_count : ℕ := 1

-- Theorem statement
theorem molecular_weight_N2O5H3P :
  N_count * atomic_weight_N +
  O_count * atomic_weight_O +
  H_count * atomic_weight_H +
  P_count * atomic_weight_P = 142.02 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_N2O5H3P_l792_79221


namespace NUMINAMATH_CALUDE_books_per_shelf_l792_79233

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : total_books = 72)
  (h2 : mystery_shelves = 3)
  (h3 : picture_shelves = 5)
  (h4 : ∃ x : ℕ, total_books = x * (mystery_shelves + picture_shelves)) :
  ∃ x : ℕ, x = 9 ∧ total_books = x * (mystery_shelves + picture_shelves) :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l792_79233


namespace NUMINAMATH_CALUDE_least_coins_coins_exist_l792_79214

theorem least_coins (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 → n ≥ 24 := by
  sorry

theorem coins_exist : ∃ n : ℕ, n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2 ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_least_coins_coins_exist_l792_79214


namespace NUMINAMATH_CALUDE_rectangular_prism_problem_l792_79205

theorem rectangular_prism_problem :
  ∀ x y : ℕ,
  x < 4 →
  y < 15 →
  15 * 5 * 4 - y * 5 * x = 120 →
  (x = 3 ∧ y = 12) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_problem_l792_79205


namespace NUMINAMATH_CALUDE_baking_powder_yesterday_l792_79231

def baking_powder_today : ℝ := 0.3
def difference_yesterday : ℝ := 0.1

theorem baking_powder_yesterday : baking_powder_today + difference_yesterday = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_baking_powder_yesterday_l792_79231


namespace NUMINAMATH_CALUDE_difference_of_squares_l792_79292

theorem difference_of_squares : 435^2 - 365^2 = 56000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l792_79292


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l792_79246

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  (∀ n : ℕ, a n = 2^(4-n)) ∧ (a 1 + a 2 + a 3 + a 4 = 15) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l792_79246


namespace NUMINAMATH_CALUDE_total_farm_tax_collected_l792_79244

/-- Theorem: Total farm tax collected from a village
Given:
- Farm tax is levied on 75% of the cultivated land
- Mr. William paid $480 as farm tax
- Mr. William's land represents 16.666666666666668% of the total taxable land in the village

Prove: The total amount collected through farm tax from the village is $2880 -/
theorem total_farm_tax_collected (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 16.666666666666668) :
  william_tax / (william_land_percentage / 100) = 2880 := by
  sorry

#check total_farm_tax_collected

end NUMINAMATH_CALUDE_total_farm_tax_collected_l792_79244


namespace NUMINAMATH_CALUDE_specific_management_structure_count_l792_79280

/-- The number of ways to form a management structure --/
def management_structure_count (total_employees : ℕ) (ceo_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  total_employees * 
  (Nat.choose (total_employees - 1) vp_count) * 
  (Nat.choose (total_employees - 1 - vp_count) managers_per_vp) * 
  (Nat.choose (total_employees - 1 - vp_count - managers_per_vp) managers_per_vp)

/-- Theorem stating the number of ways to form the specific management structure --/
theorem specific_management_structure_count :
  management_structure_count 13 1 2 3 = 349800 := by
  sorry

end NUMINAMATH_CALUDE_specific_management_structure_count_l792_79280


namespace NUMINAMATH_CALUDE_cost_per_pound_of_beef_l792_79250

/-- Given a grocery bill with chicken, beef, and oil, prove the cost per pound of beef. -/
theorem cost_per_pound_of_beef
  (total_bill : ℝ)
  (chicken_weight : ℝ)
  (beef_weight : ℝ)
  (oil_volume : ℝ)
  (oil_cost : ℝ)
  (chicken_cost : ℝ)
  (h1 : total_bill = 16)
  (h2 : chicken_weight = 2)
  (h3 : beef_weight = 3)
  (h4 : oil_volume = 1)
  (h5 : oil_cost = 1)
  (h6 : chicken_cost = 3) :
  (total_bill - chicken_cost - oil_cost) / beef_weight = 4 := by
sorry

end NUMINAMATH_CALUDE_cost_per_pound_of_beef_l792_79250


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l792_79276

theorem min_value_of_expression (x : ℝ) : 
  (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by sorry

theorem lower_bound_achievable : 
  ∃ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 = 3625 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l792_79276


namespace NUMINAMATH_CALUDE_triangle_area_approx_l792_79204

-- Define the triangle DEF and point Q
structure Triangle :=
  (D E F Q : ℝ × ℝ)

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  let d_to_q := Real.sqrt ((t.D.1 - t.Q.1)^2 + (t.D.2 - t.Q.2)^2)
  let e_to_q := Real.sqrt ((t.E.1 - t.Q.1)^2 + (t.E.2 - t.Q.2)^2)
  let f_to_q := Real.sqrt ((t.F.1 - t.Q.1)^2 + (t.F.2 - t.Q.2)^2)
  d_to_q = 5 ∧ e_to_q = 13 ∧ f_to_q = 12

def is_equilateral (t : Triangle) : Prop :=
  let d_to_e := Real.sqrt ((t.D.1 - t.E.1)^2 + (t.D.2 - t.E.2)^2)
  let e_to_f := Real.sqrt ((t.E.1 - t.F.1)^2 + (t.E.2 - t.F.2)^2)
  let f_to_d := Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)
  d_to_e = e_to_f ∧ e_to_f = f_to_d

-- Define the theorem
theorem triangle_area_approx (t : Triangle) 
  (h1 : is_valid_triangle t) (h2 : is_equilateral t) : 
  ∃ (area : ℝ), abs (area - 132) < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l792_79204


namespace NUMINAMATH_CALUDE_correct_reasoning_definitions_l792_79294

-- Define the types of reasoning
inductive ReasoningType
  | Inductive
  | Deductive
  | Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
  | PartToWhole
  | GeneralToSpecific
  | SpecificToSpecific

-- Define the relationship between reasoning types and directions
def reasoningDirection (t : ReasoningType) : ReasoningDirection :=
  match t with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating the correct definitions of reasoning types
theorem correct_reasoning_definitions :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
by sorry

end NUMINAMATH_CALUDE_correct_reasoning_definitions_l792_79294


namespace NUMINAMATH_CALUDE_softball_players_count_l792_79289

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 16)
  (h2 : hockey = 12)
  (h3 : football = 18)
  (h4 : total = 59) :
  total - (cricket + hockey + football) = 13 := by
  sorry

end NUMINAMATH_CALUDE_softball_players_count_l792_79289


namespace NUMINAMATH_CALUDE_sine_ratio_equals_two_l792_79236

/-- Triangle ABC with vertices A(-1, 0), C(1, 0), and B on the ellipse x²/4 + y²/3 = 1 -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (-1, 0)
  h_C : C = (1, 0)
  h_B : (B.1^2 / 4) + (B.2^2 / 3) = 1

/-- The sine of an angle in the triangle -/
noncomputable def sin_angle (t : Triangle) (v : Fin 3) : ℝ :=
  sorry

/-- Theorem stating that (sin A + sin C) / sin B = 2 for the given triangle -/
theorem sine_ratio_equals_two (t : Triangle) :
  (sin_angle t 0 + sin_angle t 2) / sin_angle t 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_sine_ratio_equals_two_l792_79236


namespace NUMINAMATH_CALUDE_part_one_part_two_l792_79290

/-- The set A -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

/-- The set B -/
def B (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2*m - 3}

/-- Part 1: p is sufficient but not necessary for q -/
theorem part_one (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) ∧ (∃ x, x ∈ B m ∧ x ∉ A) → m ≥ 4 := by
  sorry

/-- Part 2: A ∪ B = A -/
theorem part_two (m : ℝ) : A ∪ B m = A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l792_79290


namespace NUMINAMATH_CALUDE_happyTails_cats_count_l792_79213

/-- Represents the number of cats that can perform a specific combination of tricks -/
structure CatTricks where
  jump : Nat
  fetch : Nat
  spin : Nat
  jumpFetch : Nat
  fetchSpin : Nat
  jumpSpin : Nat
  allThree : Nat
  none : Nat

/-- Calculates the total number of cats at HappyTails Training Center -/
def totalCats (ct : CatTricks) : Nat :=
  ct.jump + ct.fetch + ct.spin - ct.jumpFetch - ct.fetchSpin - ct.jumpSpin + ct.allThree + ct.none

/-- Theorem stating that the total number of cats at HappyTails Training Center is 70 -/
theorem happyTails_cats_count (ct : CatTricks)
  (h1 : ct.jump = 40)
  (h2 : ct.fetch = 25)
  (h3 : ct.spin = 30)
  (h4 : ct.jumpFetch = 15)
  (h5 : ct.fetchSpin = 10)
  (h6 : ct.jumpSpin = 12)
  (h7 : ct.allThree = 5)
  (h8 : ct.none = 7) :
  totalCats ct = 70 := by
  sorry

end NUMINAMATH_CALUDE_happyTails_cats_count_l792_79213


namespace NUMINAMATH_CALUDE_solution_system_equations_l792_79212

theorem solution_system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (eq₁ : x₁ + x₂ = x₃^2)
  (eq₂ : x₂ + x₃ = x₄^2)
  (eq₃ : x₃ + x₄ = x₅^2)
  (eq₄ : x₄ + x₅ = x₁^2)
  (eq₅ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l792_79212


namespace NUMINAMATH_CALUDE_min_value_of_a_l792_79270

theorem min_value_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0 ∧ 
                 9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) →
  a ≥ 0 ∧ ∀ ε > 0, ∃ a' : ℝ, a' < ε ∧ 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 9 * x₁ - (4 + a') * 3 * x₁ + 4 = 0 ∧ 
                  9 * x₂ - (4 + a') * 3 * x₂ + 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l792_79270


namespace NUMINAMATH_CALUDE_ratio_subtraction_l792_79272

theorem ratio_subtraction (x : ℕ) : 
  x = 3 ∧ 
  (6 - x : ℚ) / (7 - x) < 16 / 21 ∧ 
  ∀ y : ℕ, y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21 → 
  6 = 6 :=
sorry

end NUMINAMATH_CALUDE_ratio_subtraction_l792_79272


namespace NUMINAMATH_CALUDE_complex_number_properties_l792_79203

open Complex

theorem complex_number_properties (z₁ z₂ z : ℂ) (b : ℝ) : 
  z₁ = 1 - I ∧ 
  z₂ = 4 + 6*I ∧ 
  z = 1 + b*I ∧ 
  (z + z₁).im = 0 →
  (abs z₁ + z₂ = Complex.mk (Real.sqrt 2 + 4) 6) ∧ 
  abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_properties_l792_79203


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l792_79293

/-- Represents an isosceles triangle with base a and leg b -/
structure IsoscelesTriangle where
  a : ℝ  -- base
  b : ℝ  -- leg
  ma : ℝ  -- height corresponding to base
  mb : ℝ  -- height corresponding to leg
  h_isosceles : b > 0 ∧ ma > 0 ∧ mb > 0 ∧ a * ma = b * mb

/-- Given the sums and differences of sides and heights, 
    prove the existence of an isosceles triangle -/
theorem isosceles_triangle_exists 
  (sum_sides : ℝ) 
  (sum_heights : ℝ) 
  (diff_sides : ℝ) 
  (diff_heights : ℝ) 
  (h_positive : sum_sides > 0 ∧ sum_heights > 0 ∧ diff_sides > 0 ∧ diff_heights > 0) :
  ∃ t : IsoscelesTriangle, 
    t.a + t.b = sum_sides ∧ 
    t.ma + t.mb = sum_heights ∧
    t.b - t.a = diff_sides ∧
    t.ma - t.mb = diff_heights :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l792_79293


namespace NUMINAMATH_CALUDE_prop_one_prop_three_prop_five_l792_79299

-- Proposition 1
theorem prop_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 1) / (b + 1) > a / b) : a < b := by
  sorry

-- Proposition 3
theorem prop_three : ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0 := by
  sorry

-- Proposition 5
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

theorem prop_five : binary_to_decimal [true, false, true, true, true] = 23 := by
  sorry

end NUMINAMATH_CALUDE_prop_one_prop_three_prop_five_l792_79299


namespace NUMINAMATH_CALUDE_card_play_combinations_l792_79238

/-- Represents the number of ways to play 5 cards (2 twos and 3 aces) -/
def ways_to_play_cards : ℕ :=
  Nat.factorial 5 + 
  Nat.factorial 2 + 
  Nat.factorial 4 + 
  (Nat.choose 3 2 * Nat.factorial 3) + 
  Nat.factorial 3 + 
  (Nat.choose 3 2 * Nat.factorial 4)

/-- Theorem stating that the number of ways to play the cards is 242 -/
theorem card_play_combinations : ways_to_play_cards = 242 := by
  sorry

end NUMINAMATH_CALUDE_card_play_combinations_l792_79238


namespace NUMINAMATH_CALUDE_max_sundays_in_45_days_l792_79279

/-- The number of days we're considering at the start of the year -/
def days_considered : ℕ := 45

/-- The day of the week, represented as a number from 0 to 6 -/
inductive DayOfWeek : Type
  | Sunday : DayOfWeek
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek

/-- Function to count the number of Sundays in the first n days of a year starting on a given day -/
def count_sundays (start_day : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of Sundays in the first 45 days of a year starting on Sunday is 7 -/
theorem max_sundays_in_45_days : 
  count_sundays DayOfWeek.Sunday days_considered = 7 :=
sorry

end NUMINAMATH_CALUDE_max_sundays_in_45_days_l792_79279


namespace NUMINAMATH_CALUDE_triangle_property_l792_79200

open Real

theorem triangle_property (A B C a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  b * cos C + sqrt 3 * b * sin C = a + c →
  B = π / 3 ∧
  (b = sqrt 3 → -sqrt 3 < 2 * a - c ∧ 2 * a - c < 2 * sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l792_79200


namespace NUMINAMATH_CALUDE_parabola_ellipse_intersection_l792_79230

theorem parabola_ellipse_intersection (p : ℝ) (m n k : ℝ) : 
  p > 0 → m > n → n > 0 → 
  ∃ (x₀ y₀ : ℝ), 
    y₀^2 = 2*p*x₀ ∧ 
    (x₀ + p/2)^2 + y₀^2 = 3^2 ∧ 
    x₀^2 + y₀^2 = 9 →
  ∃ (c : ℝ), 
    c = 2 ∧ 
    m^2 - n^2 = c^2 ∧
    2/m = 1/2 →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/m^2 + y₁^2/n^2 = 1 ∧
    x₂^2/m^2 + y₂^2/n^2 = 1 ∧
    y₁ = k*x₁ - 4 ∧
    y₂ = k*x₂ - 4 ∧
    x₁ ≠ x₂ ∧
    x₁*x₂ + y₁*y₂ > 0 →
  (-2*Real.sqrt 3/3 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < 2*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_ellipse_intersection_l792_79230


namespace NUMINAMATH_CALUDE_min_sum_squares_l792_79297

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l792_79297


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l792_79210

theorem arithmetic_calculation : -16 - (-12) - 24 + 18 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l792_79210


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l792_79235

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 3 * x^2 + 8 * x - 16 = 0 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l792_79235


namespace NUMINAMATH_CALUDE_rhombus_area_l792_79245

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2 : ℝ) * d1 * d2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l792_79245


namespace NUMINAMATH_CALUDE_reciprocal_of_difference_l792_79249

theorem reciprocal_of_difference : (((1 : ℚ) / 3 - (1 : ℚ) / 4)⁻¹ : ℚ) = 12 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_difference_l792_79249


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l792_79237

-- Define the variables and constants
def june_total_A : ℝ := 12000
def june_total_B : ℝ := 9000
def july_total_quantity : ℝ := 5000
def july_min_total : ℝ := 23400
def cost_A : ℝ := 2.7
def cost_B : ℝ := 3.5

-- Define the theorem
theorem fruit_sales_problem :
  ∃ (june_price_A : ℝ) (july_quantity_A : ℝ) (july_profit : ℝ),
    -- Conditions
    (june_total_A / june_price_A - june_total_B / (1.5 * june_price_A) = 1000) ∧
    (0.7 * june_price_A * july_quantity_A + 0.6 * 1.5 * june_price_A * (july_total_quantity - july_quantity_A) ≥ july_min_total) ∧
    -- Conclusions
    (june_price_A = 6) ∧
    (july_quantity_A = 3000) ∧
    (july_profit = (0.7 * june_price_A - cost_A) * july_quantity_A + 
                   (0.6 * 1.5 * june_price_A - cost_B) * (july_total_quantity - july_quantity_A)) ∧
    (july_profit = 8300) :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l792_79237


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l792_79222

-- Define the set of valid 'a' values
def ValidA : Set ℝ := { x | (0 < x ∧ x < 1) ∨ (1 < x) }

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x + 2

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h : a ∈ ValidA) : f a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l792_79222


namespace NUMINAMATH_CALUDE_algebra_class_size_l792_79224

/-- Given an Algebra 1 class where there are 11 girls and 5 fewer girls than boys,
    prove that the total number of students in the class is 27. -/
theorem algebra_class_size :
  ∀ (num_girls num_boys : ℕ),
    num_girls = 11 →
    num_girls + 5 = num_boys →
    num_girls + num_boys = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_algebra_class_size_l792_79224


namespace NUMINAMATH_CALUDE_problem_I_problem_II_l792_79211

theorem problem_I (α : Real) (h : α = π / 6) : 
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) / 
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

theorem problem_II (α : Real) (h : Real.tan α / (Real.tan α - 6) = -1) : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = -7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_I_problem_II_l792_79211


namespace NUMINAMATH_CALUDE_both_selected_probability_l792_79261

theorem both_selected_probability 
  (p_ram : ℝ) (p_ravi : ℝ) 
  (h_ram : p_ram = 1 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_independent : True) -- Assuming independence
  : p_ram * p_ravi = 1 / 35 := by
  sorry

end NUMINAMATH_CALUDE_both_selected_probability_l792_79261


namespace NUMINAMATH_CALUDE_product_ratio_l792_79217

def range_start : Int := -2020
def range_end : Int := 2019

theorem product_ratio :
  let smallest_product := range_start * (range_start + 1) * (range_start + 2)
  let largest_product := (range_end - 2) * (range_end - 1) * range_end
  (smallest_product : ℚ) / largest_product = -2020 / 2017 := by
sorry

end NUMINAMATH_CALUDE_product_ratio_l792_79217


namespace NUMINAMATH_CALUDE_add_and_convert_to_base7_37_45_l792_79240

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 10 and returns the result in base 7 -/
def addAndConvertToBase7 (a b : ℕ) : ℕ :=
  toBase7 (a + b)

theorem add_and_convert_to_base7_37_45 :
  addAndConvertToBase7 37 45 = 145 := by sorry

end NUMINAMATH_CALUDE_add_and_convert_to_base7_37_45_l792_79240


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l792_79256

theorem rational_solutions_quadratic (m : ℕ+) : 
  (∃ x : ℚ, m * x^2 + 25 * x + m = 0) ↔ (m = 10 ∨ m = 12) :=
by sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l792_79256


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l792_79225

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 1 + a 2 = 4/9 →  -- First condition
  a 3 + a 4 + a 5 + a 6 = 40 →  -- Second condition
  (a 7 + a 8 + a 9) / 9 = 117 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l792_79225


namespace NUMINAMATH_CALUDE_dans_work_time_l792_79258

theorem dans_work_time (D : ℝ) : 
  (∀ (annie_rate : ℝ), annie_rate = 1 / 10 →
   ∀ (dan_rate : ℝ), dan_rate = 1 / D →
   6 * dan_rate + 6 * annie_rate = 1) →
  D = 15 := by
sorry

end NUMINAMATH_CALUDE_dans_work_time_l792_79258


namespace NUMINAMATH_CALUDE_perfect_squares_problem_l792_79259

theorem perfect_squares_problem :
  ¬∃ (x : ℝ), x^2 = 5^2025 ∧
  ∃ (a : ℝ), a^2 = 3^2024 ∧
  ∃ (b : ℝ), b^2 = 7^2026 ∧
  ∃ (c : ℝ), c^2 = 8^2027 ∧
  ∃ (d : ℝ), d^2 = 9^2028 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_problem_l792_79259


namespace NUMINAMATH_CALUDE_percentage_ratio_l792_79288

theorem percentage_ratio (x : ℝ) (a b : ℝ) (ha : a = 0.08 * x) (hb : b = 0.16 * x) :
  a / b = 0.5 := by sorry

end NUMINAMATH_CALUDE_percentage_ratio_l792_79288


namespace NUMINAMATH_CALUDE_mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l792_79287

-- Define the multiplications
def mult_A (x y : ℝ) := (3*x + 7*y) * (3*x - 7*y)
def mult_B (m n : ℝ) := (5*m - n) * (n - 5*m)
def mult_C (x : ℝ) := (-0.2*x - 0.3) * (-0.2*x + 0.3)
def mult_D (m n : ℝ) := (-3*n - m*n) * (3*n - m*n)

-- Define the square of binomial form
def square_of_binomial (a b : ℝ) := (a + b)^2

-- Define the difference of squares form
def diff_of_squares (a b : ℝ) := a^2 - b^2

theorem mult_B_not_binomial_square_or_diff_squares :
  ∀ m n : ℝ, ¬∃ a b : ℝ, mult_B m n = square_of_binomial a b ∨ mult_B m n = diff_of_squares a b :=
sorry

theorem other_mults_are_diff_squares :
  (∀ x y : ℝ, ∃ a b : ℝ, mult_A x y = diff_of_squares a b) ∧
  (∀ x : ℝ, ∃ a b : ℝ, mult_C x = diff_of_squares a b) ∧
  (∀ m n : ℝ, ∃ a b : ℝ, mult_D m n = diff_of_squares a b) :=
sorry

end NUMINAMATH_CALUDE_mult_B_not_binomial_square_or_diff_squares_other_mults_are_diff_squares_l792_79287


namespace NUMINAMATH_CALUDE_correct_initial_amounts_l792_79286

/-- Represents the initial amounts of money John and Richard had --/
structure InitialMoney where
  john : ℚ
  richard : ℚ

/-- Represents the final amounts of money John and Richard had after transactions --/
structure FinalMoney where
  john : ℚ
  richard : ℚ

/-- Calculates the final money based on initial money and described transactions --/
def calculateFinalMoney (initial : InitialMoney) : FinalMoney :=
  { john := initial.john - (initial.richard + initial.john),
    richard := 2 * initial.richard + 2 * initial.john }

/-- Theorem stating the correct initial amounts given the final amounts --/
theorem correct_initial_amounts :
  ∃ (initial : InitialMoney),
    let final := calculateFinalMoney initial
    final.john = 7/2 ∧ final.richard = 3 ∧
    initial.john = 5/2 ∧ initial.richard = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_correct_initial_amounts_l792_79286


namespace NUMINAMATH_CALUDE_parabolas_no_intersection_l792_79263

/-- The parabolas y = 3x^2 - 6x + 6 and y = -2x^2 + x + 3 do not intersect in the real plane. -/
theorem parabolas_no_intersection : 
  ∀ x y : ℝ, (y = 3*x^2 - 6*x + 6) → (y = -2*x^2 + x + 3) → False :=
by
  sorry

end NUMINAMATH_CALUDE_parabolas_no_intersection_l792_79263


namespace NUMINAMATH_CALUDE_min_vertical_distance_l792_79295

/-- The absolute value function -/
def abs_func (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def quad_func (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between the two functions -/
def vert_distance (x : ℝ) : ℝ := |abs_func x - quad_func x|

theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 3 ∧ ∀ (x : ℝ), vert_distance x ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l792_79295


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l792_79265

-- Define the repeating decimal 6.8181...
def repeating_decimal : ℚ := 6 + 81 / 99

-- Theorem stating that the repeating decimal equals 75/11
theorem repeating_decimal_equals_fraction : repeating_decimal = 75 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l792_79265


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l792_79242

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 8 →
  f 3 = 27 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l792_79242
