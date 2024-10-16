import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2310_231055

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a4_eq_1 : a 4 = 1
  S15_eq_75 : S 15 = 75

/-- The theorem statement -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = n - 3) ∧
  (∃ c : ℝ, c ≠ 0 ∧
    (∀ n m, (seq.S (n + 1) / ((n + 1) + c) - seq.S n / (n + c)) =
            (seq.S (m + 1) / ((m + 1) + c) - seq.S m / (m + c))) →
    c = -5) :=
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2310_231055


namespace NUMINAMATH_CALUDE_cucumber_packing_l2310_231029

theorem cucumber_packing (total_cucumbers : ℕ) (basket_capacity : ℕ) 
  (h1 : total_cucumbers = 216)
  (h2 : basket_capacity = 23) :
  ∃ (filled_baskets : ℕ) (remaining_cucumbers : ℕ),
    filled_baskets * basket_capacity + remaining_cucumbers = total_cucumbers ∧
    filled_baskets = 9 ∧
    remaining_cucumbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_packing_l2310_231029


namespace NUMINAMATH_CALUDE_rectangle_area_change_l2310_231048

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.4 * L
  let new_width := W / 2
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area / original_area) = 0.7 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l2310_231048


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2310_231024

/-- Given a 2x2 matrix B with specific properties, prove that the sum of squares of its elements is 2 -/
theorem sum_of_squares_equals_two (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = B⁻¹) →
  (x^2 + y^2 = 1) →
  (z^2 + w^2 = 1) →
  (y + z = 1/2) →
  (x^2 + y^2 + z^2 + w^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_two_l2310_231024


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l2310_231070

-- Define the function f
def f (b x : ℝ) : ℝ := x^2 + b*x + 2

-- Theorem statement
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -2) ↔ b ∈ Set.Ioo (-4 : ℝ) 4 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l2310_231070


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l2310_231089

/-- The perpendicular bisector of a line segment passing through two points. -/
structure PerpendicularBisector where
  -- The equation of the line: x + y = b
  b : ℝ
  -- The two points defining the line segment
  p1 : ℝ × ℝ := (2, 4)
  p2 : ℝ × ℝ := (6, 8)
  -- The condition that the line is a perpendicular bisector
  is_perp_bisector : b = p1.1 + p1.2 + p2.1 + p2.2

/-- The value of b for the perpendicular bisector of the line segment from (2,4) to (6,8) is 10. -/
theorem perpendicular_bisector_value : 
  ∀ (pb : PerpendicularBisector), pb.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l2310_231089


namespace NUMINAMATH_CALUDE_ab_product_theorem_l2310_231040

theorem ab_product_theorem (a b : ℝ) 
  (h1 : (27 : ℝ) ^ a = 3 ^ (10 * (b + 2)))
  (h2 : (125 : ℝ) ^ b = 5 ^ (a - 3)) : 
  a * b = 330 := by sorry

end NUMINAMATH_CALUDE_ab_product_theorem_l2310_231040


namespace NUMINAMATH_CALUDE_unique_prime_solution_l2310_231073

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p * q * r = 5 * (p + q + r) → 
    (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨ (p = 5 ∧ q = 2 ∧ r = 7) ∨ 
    (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

#check unique_prime_solution

end NUMINAMATH_CALUDE_unique_prime_solution_l2310_231073


namespace NUMINAMATH_CALUDE_square_equation_solution_l2310_231015

theorem square_equation_solution : ∃! x : ℝ, 97 + x * (19 + 91 / x) = 321 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2310_231015


namespace NUMINAMATH_CALUDE_software_contract_probability_l2310_231056

theorem software_contract_probability
  (p_hardware : ℝ)
  (p_at_least_one : ℝ)
  (p_both : ℝ)
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 5/6)
  (h3 : p_both = 11/30) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_software_contract_probability_l2310_231056


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2310_231095

/-- The number of ways to distribute n indistinguishable items into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops from 3 basic flavors -/
theorem ice_cream_flavors : distribute 5 3 = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2310_231095


namespace NUMINAMATH_CALUDE_tan_1450_degrees_solution_l2310_231059

theorem tan_1450_degrees_solution (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1450 * π / 180) →
  n = 10 ∨ n = -170 := by
sorry

end NUMINAMATH_CALUDE_tan_1450_degrees_solution_l2310_231059


namespace NUMINAMATH_CALUDE_equation_solutions_l2310_231053

theorem equation_solutions : 
  (∃ S₁ : Set ℝ, S₁ = {x : ℝ | x * (x - 2) + x - 2 = 0} ∧ S₁ = {2, -1}) ∧
  (∃ S₂ : Set ℝ, S₂ = {x : ℝ | 2 * x^2 + 5 * x + 3 = 0} ∧ S₂ = {-1, -3/2}) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l2310_231053


namespace NUMINAMATH_CALUDE_square_area_proof_l2310_231006

theorem square_area_proof (x : ℝ) : 
  (5 * x - 20 : ℝ) = (25 - 4 * x : ℝ) → 
  (5 * x - 20 : ℝ) ^ 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l2310_231006


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2310_231000

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (2*k - 1)*x₁ + k^2 - 1 = 0 ∧ 
    x₂^2 + (2*k - 1)*x₂ + k^2 - 1 = 0 ∧ 
    x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2310_231000


namespace NUMINAMATH_CALUDE_unique_natural_number_a_l2310_231074

theorem unique_natural_number_a : ∃! (a : ℕ), 
  (1000 ≤ 4 * a^2) ∧ (4 * a^2 < 10000) ∧ 
  (1000 ≤ (4/3) * a^3) ∧ ((4/3) * a^3 < 10000) ∧
  (∃ (n : ℕ), (4/3) * a^3 = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_natural_number_a_l2310_231074


namespace NUMINAMATH_CALUDE_zhang_qiujian_suanjing_problem_l2310_231003

theorem zhang_qiujian_suanjing_problem (a : ℕ → ℚ) :
  (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
  a 1 + a 2 + a 3 = 4 →                         -- sum of first 3 terms
  a 8 + a 9 + a 10 = 3 →                        -- sum of last 3 terms
  a 5 + a 6 = 7/3 :=                            -- sum of 5th and 6th terms
by sorry

end NUMINAMATH_CALUDE_zhang_qiujian_suanjing_problem_l2310_231003


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2310_231014

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → 
  (m = 2 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2310_231014


namespace NUMINAMATH_CALUDE_birds_landed_on_fence_l2310_231004

/-- Given an initial number of birds and a final total number of birds on a fence,
    calculate the number of birds that landed on the fence. -/
def birds_landed (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that 8 birds landed on the fence given the initial and final counts. -/
theorem birds_landed_on_fence : birds_landed 12 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_birds_landed_on_fence_l2310_231004


namespace NUMINAMATH_CALUDE_three_x_intercepts_l2310_231098

/-- The function representing the curve x = y^3 - 4y^2 + 3y + 2 -/
def f (y : ℝ) : ℝ := y^3 - 4*y^2 + 3*y + 2

/-- Theorem stating that the equation f(y) = 0 has exactly 3 real solutions -/
theorem three_x_intercepts : ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ y : ℝ, y ∈ s ↔ f y = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_x_intercepts_l2310_231098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2310_231097

/-- An arithmetic sequence. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the first, third, and fifth terms is 9. -/
def SumOdd (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 = 9

/-- The sum of the second, fourth, and sixth terms is 15. -/
def SumEven (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 = 15

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : SumOdd a) 
  (h3 : SumEven a) : 
  a 3 + a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2310_231097


namespace NUMINAMATH_CALUDE_parallelepiped_surface_area_l2310_231023

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  a : ℝ  -- First diagonal of the base
  b : ℝ  -- Second diagonal of the base
  sphere_inscribed : Bool  -- Indicator that a sphere is inscribed

-- Define the total surface area function
def total_surface_area (p : RectangularParallelepiped) : ℝ :=
  3 * p.a * p.b

-- Theorem statement
theorem parallelepiped_surface_area 
  (p : RectangularParallelepiped) 
  (h : p.sphere_inscribed = true) :
  total_surface_area p = 3 * p.a * p.b :=
by
  sorry


end NUMINAMATH_CALUDE_parallelepiped_surface_area_l2310_231023


namespace NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2310_231019

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem smallest_positive_period_of_f :
  ∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → S < T → ¬ is_periodic f S :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_period_of_f_l2310_231019


namespace NUMINAMATH_CALUDE_student_number_problem_l2310_231068

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2310_231068


namespace NUMINAMATH_CALUDE_cylinder_max_volume_l2310_231077

/-- Given a cylinder with an axial cross-section circumference of 90 cm,
    prove that its maximum volume is 3375π cm³. -/
theorem cylinder_max_volume (d m : ℝ) (h : d + m = 45) :
  ∃ (V : ℝ), V ≤ 3375 * Real.pi ∧ ∃ (r : ℝ), V = π * r^2 * m ∧ d = 2 * r :=
sorry

end NUMINAMATH_CALUDE_cylinder_max_volume_l2310_231077


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2310_231085

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b > 0 → a > 0 ∧ b > 0) ↔ (a + b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2310_231085


namespace NUMINAMATH_CALUDE_third_number_is_one_l2310_231079

/-- Define a sequence where each segment starts with 1 and counts up by one more number than the previous segment -/
def special_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| n + 2 => 
  let segment := n / 2 + 1
  let position := n % (segment + 1)
  if position = 0 then 1 else position + 1

/-- The third number in the special sequence is 1 -/
theorem third_number_is_one : special_sequence 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_number_is_one_l2310_231079


namespace NUMINAMATH_CALUDE_negation_and_contrary_l2310_231094

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def original_statement : Prop :=
  ∀ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_contrary :
  (¬original_statement ↔ ∃ n : ℤ, (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬divisible_by_five n) ∧
  (∀ n : ℤ, (last_digit n ≠ 0 ∧ last_digit n ≠ 5) → ¬divisible_by_five n) :=
sorry

end NUMINAMATH_CALUDE_negation_and_contrary_l2310_231094


namespace NUMINAMATH_CALUDE_no_natural_n_power_of_two_l2310_231034

theorem no_natural_n_power_of_two : ∀ n : ℕ, ¬∃ k : ℕ, 6 * n^2 + 5 * n = 2^k := by sorry

end NUMINAMATH_CALUDE_no_natural_n_power_of_two_l2310_231034


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2310_231025

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) ≥ 4 :=
by sorry

theorem equality_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (a * b) + 1 / (a * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2310_231025


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2310_231041

-- Problem 1
theorem problem_one : Real.sqrt 12 - Real.sqrt 3 + 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt 18 / Real.sqrt 6 * Real.sqrt 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2310_231041


namespace NUMINAMATH_CALUDE_rectangle_area_l2310_231009

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 160) : ∃ (length width : ℝ),
  length = 4 * width ∧
  2 * (length + width) = perimeter ∧
  length * width = 1024 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2310_231009


namespace NUMINAMATH_CALUDE_age_difference_l2310_231072

/-- Given a father and daughter whose ages sum to 54, and the daughter is 16 years old,
    prove that the difference between their ages is 22 years. -/
theorem age_difference (father_age daughter_age : ℕ) : 
  father_age + daughter_age = 54 →
  daughter_age = 16 →
  father_age - daughter_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2310_231072


namespace NUMINAMATH_CALUDE_cricket_average_l2310_231064

theorem cricket_average (initial_average : ℚ) : 
  (10 * initial_average + 65 = 11 * (initial_average + 3)) → initial_average = 32 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l2310_231064


namespace NUMINAMATH_CALUDE_complex_set_sum_l2310_231033

/-- A set of complex numbers with closure under multiplication property -/
def ClosedMultSet (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S := {a, b, c, d}
  ClosedMultSet S →
  a^2 = 1 →
  b^2 = 1 →
  c^2 = b →
  b + c + d = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_set_sum_l2310_231033


namespace NUMINAMATH_CALUDE_power_zero_minus_pi_l2310_231044

theorem power_zero_minus_pi (x : ℝ) : (x - Real.pi) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_minus_pi_l2310_231044


namespace NUMINAMATH_CALUDE_inequality_proof_l2310_231084

theorem inequality_proof (a b c : ℕ+) (h : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2310_231084


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l2310_231007

/-- Calculates the number of free donuts earned in a summer reading challenge -/
def free_donuts (books_per_week : ℕ) (weeks : ℕ) (books_per_coupon : ℕ) : ℕ :=
  (books_per_week * weeks) / books_per_coupon

/-- Proves that Quinn is eligible for 4 free donuts -/
theorem quinn_free_donuts : free_donuts 2 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l2310_231007


namespace NUMINAMATH_CALUDE_jordans_mangoes_l2310_231091

theorem jordans_mangoes (total : ℕ) (ripe : ℕ) (unripe : ℕ) (kept : ℕ) (jars : ℕ) (mangoes_per_jar : ℕ) : 
  ripe = total / 3 →
  unripe = 2 * total / 3 →
  kept = 16 →
  jars = 5 →
  mangoes_per_jar = 4 →
  unripe = kept + jars * mangoes_per_jar →
  total = ripe + unripe →
  total = 54 :=
by sorry

end NUMINAMATH_CALUDE_jordans_mangoes_l2310_231091


namespace NUMINAMATH_CALUDE_gcd_1443_999_l2310_231013

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l2310_231013


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l2310_231012

/-- Proves that given 30 pencils and 5 more pencils than pens, the ratio of pens to pencils is 5:6 -/
theorem pen_pencil_ratio :
  ∀ (num_pens num_pencils : ℕ),
    num_pencils = 30 →
    num_pencils = num_pens + 5 →
    (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l2310_231012


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_inter_B_range_of_a_l2310_231016

-- Define the sets A, B, and C
def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 10} := by sorry

-- Theorem for the intersection of complement of A and B
theorem complement_A_inter_B : (Set.univ \ A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 10} := by sorry

-- Theorem for the range of a when A ∩ C is nonempty
theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) : a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_inter_B_range_of_a_l2310_231016


namespace NUMINAMATH_CALUDE_school_pupils_count_school_pupils_count_proof_l2310_231092

theorem school_pupils_count : ℕ → ℕ → ℕ → Prop :=
  fun num_girls girls_boys_diff total_pupils =>
    (num_girls = 868) →
    (girls_boys_diff = 281) →
    (total_pupils = num_girls + (num_girls - girls_boys_diff)) →
    total_pupils = 1455

-- The proof is omitted
theorem school_pupils_count_proof : school_pupils_count 868 281 1455 := by
  sorry

end NUMINAMATH_CALUDE_school_pupils_count_school_pupils_count_proof_l2310_231092


namespace NUMINAMATH_CALUDE_cow_count_l2310_231080

/-- The number of days over which the husk consumption is measured -/
def days : ℕ := 50

/-- The number of bags of husk consumed by the group of cows -/
def group_consumption : ℕ := 50

/-- The number of bags of husk consumed by one cow -/
def single_cow_consumption : ℕ := 1

/-- The number of cows in the farm -/
def num_cows : ℕ := group_consumption

theorem cow_count : num_cows = 50 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l2310_231080


namespace NUMINAMATH_CALUDE_square_sum_equality_l2310_231051

theorem square_sum_equality (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l2310_231051


namespace NUMINAMATH_CALUDE_exists_polygon_with_bisecting_point_l2310_231065

/-- A polygon represented by a list of points in 2D space -/
def Polygon : Type := List (ℝ × ℝ)

/-- A point in 2D space -/
def Point : Type := ℝ × ℝ

/-- Check if a point is on the boundary of a polygon -/
def isOnBoundary (p : Point) (poly : Polygon) : Prop := sorry

/-- The area of a polygon -/
def area (poly : Polygon) : ℝ := sorry

/-- A line represented by two points -/
def Line : Type := Point × Point

/-- The area of a polygon on one side of a line -/
def areaOneSide (poly : Polygon) (l : Line) : ℝ := sorry

/-- Theorem: There exists a polygon and a point on its boundary such that 
    any line passing through this point bisects the area of the polygon -/
theorem exists_polygon_with_bisecting_point : 
  ∃ (poly : Polygon) (p : Point), 
    isOnBoundary p poly ∧ 
    ∀ (l : Line), 
      (l.1 = p ∨ l.2 = p) → 
      areaOneSide poly l = area poly / 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_polygon_with_bisecting_point_l2310_231065


namespace NUMINAMATH_CALUDE_defendant_statement_implies_innocence_l2310_231045

-- Define the types of people on the island
inductive Person
| Knight
| Liar

-- Define the crime and accusation
def Crime : Type := Unit
def Accusation : Type := Unit

-- Define the statement made by the defendant
def DefendantStatement (criminal : Person) : Prop :=
  criminal = Person.Liar

-- Define the concept of telling the truth
def TellsTruth (p : Person) (statement : Prop) : Prop :=
  match p with
  | Person.Knight => statement
  | Person.Liar => ¬statement

-- Theorem: The defendant's statement implies innocence regardless of their type
theorem defendant_statement_implies_innocence 
  (defendant : Person) 
  (crime : Crime) 
  (accusation : Accusation) :
  TellsTruth defendant (DefendantStatement (Person.Liar)) → 
  defendant ≠ Person.Liar :=
sorry

end NUMINAMATH_CALUDE_defendant_statement_implies_innocence_l2310_231045


namespace NUMINAMATH_CALUDE_lioness_age_l2310_231022

theorem lioness_age (hyena_age lioness_age : ℕ) : 
  lioness_age = 2 * hyena_age →
  (hyena_age / 2 + 5) + (lioness_age / 2 + 5) = 19 →
  lioness_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_lioness_age_l2310_231022


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2310_231042

theorem solve_exponential_equation :
  ∃! x : ℝ, (8 : ℝ)^(x - 1) / (2 : ℝ)^(x - 1) = (64 : ℝ)^(2 * x) ∧ x = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2310_231042


namespace NUMINAMATH_CALUDE_triangle_area_doubles_l2310_231005

theorem triangle_area_doubles (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let area := (1 / 2) * a * b * Real.sin θ
  let new_area := (1 / 2) * (2 * a) * b * Real.sin θ
  new_area = 2 * area := by sorry

end NUMINAMATH_CALUDE_triangle_area_doubles_l2310_231005


namespace NUMINAMATH_CALUDE_game_draw_fraction_l2310_231086

theorem game_draw_fraction (ben_wins tom_wins : ℚ) 
  (h1 : ben_wins = 4/9) 
  (h2 : tom_wins = 1/3) : 
  1 - (ben_wins + tom_wins) = 2/9 := by
sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l2310_231086


namespace NUMINAMATH_CALUDE_equation_proof_l2310_231037

/-- Given a > 0 and -∛(√a) ≤ b < ∛(a³ - √a), prove that A = 1 when
    2.334 A = √(a³-b³+√a) · (√(a³/² + √(b³+√a)) · √(a³/² - √(b³+√a))) / √((a³+b³)² - a(4a²b³+1)) -/
theorem equation_proof (a b A : ℝ) 
  (ha : a > 0) 
  (hb : -Real.rpow a (1/6) ≤ b ∧ b < Real.rpow (a^3 - Real.sqrt a) (1/3)) 
  (heq : 2.334 * A = Real.sqrt (a^3 - b^3 + Real.sqrt a) * 
    (Real.sqrt (Real.sqrt (a^3) + Real.sqrt (b^3 + Real.sqrt a)) * 
     Real.sqrt (Real.sqrt (a^3) - Real.sqrt (b^3 + Real.sqrt a))) / 
    Real.sqrt ((a^3 + b^3)^2 - a * (4 * a^2 * b^3 + 1))) : 
  A = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2310_231037


namespace NUMINAMATH_CALUDE_largest_constant_inequality_two_is_largest_constant_l2310_231069

theorem largest_constant_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
sorry

theorem two_is_largest_constant :
  ∀ ε > 0, ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_two_is_largest_constant_l2310_231069


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2310_231030

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 2) / (x - 1) = 0 ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2310_231030


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l2310_231067

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_iff_equal_slopes {m₁ m₂ : ℝ} : 
  (m₁ = m₂) ↔ (∀ (x y : ℝ), m₁ * x + y = 0 ↔ m₂ * x + y = 0)

/-- The slope of a line ax + by = c is -a/b -/
axiom slope_of_line {a b c : ℝ} (hb : b ≠ 0) : 
  ∀ (x y : ℝ), a * x + b * y = c → -a/b * x + y = c/b

theorem parallel_lines_condition (m : ℝ) :
  (∀ (x y : ℝ), (m - 1) * x + y = 4 * m - 1 ↔ 2 * x - 3 * y = 5) ↔ m = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l2310_231067


namespace NUMINAMATH_CALUDE_average_difference_number_of_elements_averaged_l2310_231078

/-- Given two real numbers with an average of 45, and two real numbers with an average of 90,
    prove that the difference between the third and first number is 90. -/
theorem average_difference (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

/-- The number of elements being averaged in both situations is 2. -/
theorem number_of_elements_averaged (n m : ℕ) 
  (h1 : ∃ (a b : ℝ), (a + b) / n = 45)
  (h2 : ∃ (b c : ℝ), (b + c) / m = 90) :
  n = 2 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_number_of_elements_averaged_l2310_231078


namespace NUMINAMATH_CALUDE_peters_pizza_consumption_l2310_231088

theorem peters_pizza_consumption :
  ∀ (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ),
    total_slices = 16 →
    whole_slices = 2 →
    shared_slice = 1/3 →
    (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_peters_pizza_consumption_l2310_231088


namespace NUMINAMATH_CALUDE_equation_solutions_l2310_231061

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + x = 19.5 ∧ x = 3.9) ∧
  (∃ x : ℝ, 26.4 - 3 * x = 14.4 ∧ x = 4) ∧
  (∃ x : ℝ, 2 * x - 0.5 * 2 = 0.8 ∧ x = 0.9) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2310_231061


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l2310_231047

theorem fractional_equation_positive_root (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = a / (2 - x) - 2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l2310_231047


namespace NUMINAMATH_CALUDE_quadratic_expression_evaluation_l2310_231026

theorem quadratic_expression_evaluation :
  let x : ℝ := 2
  (x^2 - 3*x + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_evaluation_l2310_231026


namespace NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l2310_231052

theorem root_minus_one_implies_k_eq_neg_two (k : ℝ) :
  ((-1 : ℝ)^2 - k*(-1) + 1 = 0) → k = -2 :=
by sorry

end NUMINAMATH_CALUDE_root_minus_one_implies_k_eq_neg_two_l2310_231052


namespace NUMINAMATH_CALUDE_red_peach_count_l2310_231021

/-- Represents the count of peaches of different colors in a basket -/
structure PeachBasket where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Given a basket of peaches with 8 green peaches and 1 more green peach than red peaches,
    prove that there are 7 red peaches -/
theorem red_peach_count (basket : PeachBasket) 
    (green_count : basket.green = 8)
    (green_red_diff : basket.green = basket.red + 1) : 
  basket.red = 7 := by
  sorry

end NUMINAMATH_CALUDE_red_peach_count_l2310_231021


namespace NUMINAMATH_CALUDE_largest_not_expressible_l2310_231018

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_expressible (n : ℕ) : Prop :=
  ∃ k c, k > 0 ∧ is_composite c ∧ n = 37 * k + c

theorem largest_not_expressible :
  (∀ n > 66, is_expressible n) ∧ ¬is_expressible 66 :=
sorry

end NUMINAMATH_CALUDE_largest_not_expressible_l2310_231018


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l2310_231099

/-- Given a hyperbola with equation x²/36 - y²/b² = 1 where b > 0,
    eccentricity e = 5/3, and a point P on the hyperbola such that |PF₁| = 15,
    prove that |PF₂| = 27 -/
theorem hyperbola_focal_distance (b : ℝ) (P : ℝ × ℝ) :
  b > 0 →
  (P.1^2 / 36 - P.2^2 / b^2 = 1) →
  (Real.sqrt (36 + b^2) / 6 = 5 / 3) →
  (Real.sqrt ((P.1 + Real.sqrt (36 + b^2))^2 + P.2^2) = 15) →
  Real.sqrt ((P.1 - Real.sqrt (36 + b^2))^2 + P.2^2) = 27 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l2310_231099


namespace NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l2310_231075

/-- The standard equation of a circle with diameter endpoints (0,2) and (4,4) -/
theorem circle_equation_with_given_diameter :
  let p₁ : ℝ × ℝ := (0, 2)
  let p₂ : ℝ × ℝ := (4, 4)
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (t : ℝ), p = (1 - t) • p₁ + t • p₂ ∧ 0 ≤ t ∧ t ≤ 1}
  ∀ (x y : ℝ), (x, y) ∈ M ↔ (x - 2)^2 + (y - 3)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_given_diameter_l2310_231075


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_zero_l2310_231076

theorem sum_of_reciprocals_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) : 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_zero_l2310_231076


namespace NUMINAMATH_CALUDE_right_triangle_vector_problem_l2310_231093

/-- Given a right-angled triangle ABC where AB is the hypotenuse,
    vector CA = (3, -9), and vector CB = (-3, x), prove that x = -1. -/
theorem right_triangle_vector_problem (A B C : ℝ × ℝ) (x : ℝ) :
  (B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 = (C.1 - A.1) ^ 2 + (C.2 - A.2) ^ 2 
    + (C.1 - B.1) ^ 2 + (C.2 - B.2) ^ 2 →
  (C.1 - A.1, C.2 - A.2) = (3, -9) →
  (C.1 - B.1, C.2 - B.2) = (-3, x) →
  x = -1 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_vector_problem_l2310_231093


namespace NUMINAMATH_CALUDE_sum_of_squares_l2310_231017

theorem sum_of_squares (a b c : ℝ) : 
  a + b + c = 3 → 
  (a - 1)^3 + (b - 1)^3 + (c - 1)^3 = 0 → 
  a = 2 → 
  a^2 + b^2 + c^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2310_231017


namespace NUMINAMATH_CALUDE_fourth_side_length_l2310_231039

/-- A quadrilateral inscribed in a circle with radius 150√2, where three sides have lengths 150, 150, and 150√3 -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The length of the first side -/
  side1 : ℝ
  /-- The length of the second side -/
  side2 : ℝ
  /-- The length of the third side -/
  side3 : ℝ
  /-- The length of the fourth side -/
  side4 : ℝ
  /-- The radius is 150√2 -/
  radius_eq : radius = 150 * Real.sqrt 2
  /-- The first side has length 150 -/
  side1_eq : side1 = 150
  /-- The second side has length 150 -/
  side2_eq : side2 = 150
  /-- The third side has length 150√3 -/
  side3_eq : side3 = 150 * Real.sqrt 3

/-- The theorem stating that the fourth side has length 150√7 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : q.side4 = 150 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l2310_231039


namespace NUMINAMATH_CALUDE_greatest_b_value_l2310_231020

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2310_231020


namespace NUMINAMATH_CALUDE_point_not_in_region_l2310_231001

def region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region :
  ¬(region 2 0) ∧ 
  (region 0 0) ∧ 
  (region 1 1) ∧ 
  (region 0 2) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l2310_231001


namespace NUMINAMATH_CALUDE_negative_a_range_l2310_231057

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - 7 * x + 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a < 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x < 1/2 ∨ x > 3}

-- Theorem statement
theorem negative_a_range (a : ℝ) (h_neg : a < 0) :
  (complementA ∩ B a = B a) ↔ -1/4 ≤ a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_a_range_l2310_231057


namespace NUMINAMATH_CALUDE_largest_number_in_set_l2310_231043

def S (a : ℝ) : Set ℝ := {-3*a, 4*a, 24/a, a^2, 2*a+6, 1}

theorem largest_number_in_set (a : ℝ) (h : a = 3) :
  (∀ x ∈ S a, x ≤ 4*a) ∧ (∀ x ∈ S a, x ≤ 2*a+6) ∧ (4*a ∈ S a) ∧ (2*a+6 ∈ S a) :=
sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l2310_231043


namespace NUMINAMATH_CALUDE_arithmetic_fraction_difference_l2310_231096

theorem arithmetic_fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_difference_l2310_231096


namespace NUMINAMATH_CALUDE_max_value_theorem_l2310_231054

theorem max_value_theorem (x y : ℝ) (h : x^2 + 4*y^2 = 3) :
  (1/2 : ℝ)*x + y ≤ Real.sqrt 6 / 2 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*y₀^2 = 3 ∧ (1/2 : ℝ)*x₀ + y₀ = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2310_231054


namespace NUMINAMATH_CALUDE_no_nonzero_rational_solution_l2310_231038

theorem no_nonzero_rational_solution :
  ∀ (x y z : ℚ), x^3 + 3*y^3 + 9*z^3 = 9*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nonzero_rational_solution_l2310_231038


namespace NUMINAMATH_CALUDE_unique_prime_pair_l2310_231082

def f (x : ℕ) : ℕ := x^2 + x + 1

theorem unique_prime_pair : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ f p = f q + 242 ∧ p = 61 ∧ q = 59 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l2310_231082


namespace NUMINAMATH_CALUDE_fraction_cubed_l2310_231028

theorem fraction_cubed : (2 / 3 : ℚ) ^ 3 = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cubed_l2310_231028


namespace NUMINAMATH_CALUDE_unique_egyptian_fraction_representation_l2310_231036

theorem unique_egyptian_fraction_representation (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  ∃! (x y : ℕ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (2 : ℚ) / p = 1 / x + 1 / y := by
  sorry

end NUMINAMATH_CALUDE_unique_egyptian_fraction_representation_l2310_231036


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l2310_231008

theorem divisible_by_eleven (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∃ k : ℤ, (n^2 + 4^n + 7^n : ℤ) = k * n) : 
  ∃ m : ℤ, (n^2 + 4^n + 7^n : ℤ) / n = 11 * m := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l2310_231008


namespace NUMINAMATH_CALUDE_jake_weight_ratio_l2310_231083

/-- Jake's weight problem -/
theorem jake_weight_ratio :
  let jake_present_weight : ℚ := 196
  let total_weight : ℚ := 290
  let weight_loss : ℚ := 8
  let jake_new_weight := jake_present_weight - weight_loss
  let sister_weight := total_weight - jake_present_weight
  jake_new_weight / sister_weight = 2 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_ratio_l2310_231083


namespace NUMINAMATH_CALUDE_four_digit_square_palindromes_l2310_231062

theorem four_digit_square_palindromes :
  (∃! (s : Finset Nat), 
    ∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
  (∃ s : Finset Nat, 
    (∀ n, n ∈ s ↔ 
      32 ≤ n ∧ n ≤ 99 ∧ 
      1000 ≤ n^2 ∧ n^2 ≤ 9999 ∧ 
      (∃ a b : Nat, n^2 = a * 1000 + b * 100 + b * 10 + a)) ∧ 
    s.card = 2) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_square_palindromes_l2310_231062


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2310_231049

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2310_231049


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2310_231087

/-- Given an arithmetic sequence {aₙ} where (a₂ + a₅ = 4) and (a₆ + a₉ = 20),
    prove that (a₄ + a₇) = 12 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 5 = 4 →
  a 6 + a 9 = 20 →
  a 4 + a 7 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2310_231087


namespace NUMINAMATH_CALUDE_triangle_angle_values_l2310_231060

theorem triangle_angle_values (A B C : Real) (AB AC : Real) :
  AB = 2 →
  AC = Real.sqrt 2 →
  B = 30 * Real.pi / 180 →
  A = 105 * Real.pi / 180 ∨ A = 15 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_values_l2310_231060


namespace NUMINAMATH_CALUDE_isosceles_triangle_cosine_l2310_231066

/-- Theorem: In an isosceles triangle with two sides of length 3 and the third side of length √15 - √3,
    the cosine of the angle opposite the third side is equal to √5/3. -/
theorem isosceles_triangle_cosine (a b c : ℝ) (h1 : a = 3) (h2 : b = 3) (h3 : c = Real.sqrt 15 - Real.sqrt 3) :
  let cosC := (a^2 + b^2 - c^2) / (2 * a * b)
  cosC = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_cosine_l2310_231066


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2310_231002

theorem complex_modulus_problem (z : ℂ) (h : (z - 2*Complex.I)*(1 - Complex.I) = -2) :
  Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2310_231002


namespace NUMINAMATH_CALUDE_vector_cross_product_solution_l2310_231046

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem vector_cross_product_solution 
  (a b c d : V) 
  (h : a + b + c + d = 0) :
  ∃! (k m : ℝ), ∀ (a b c d : V),
    a + b + c + d = 0 →
    k • (cross_product b a) + m • (cross_product d c) + 
    cross_product b c + cross_product c a + cross_product d b = 0 ∧
    k = 2 ∧ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_cross_product_solution_l2310_231046


namespace NUMINAMATH_CALUDE_power_digits_theorem_l2310_231010

/-- The number of digits to the right of the decimal place in a given number -/
def decimalDigits (x : ℝ) : ℕ :=
  sorry

/-- The result of raising a number to a power -/
def powerResult (base : ℝ) (exponent : ℕ) : ℝ :=
  base ^ exponent

theorem power_digits_theorem :
  let base := 10^4 * 3.456789
  decimalDigits (powerResult base 11) = 22 := by
  sorry

end NUMINAMATH_CALUDE_power_digits_theorem_l2310_231010


namespace NUMINAMATH_CALUDE_centroid_on_line_segment_l2310_231063

/-- Given a triangle ABC with points M on AB and N on AC, if BM/MA + CN/NA = 1,
    then the centroid of triangle ABC is collinear with M and N. -/
theorem centroid_on_line_segment (A B C M N : EuclideanSpace ℝ (Fin 2)) :
  (∃ s t : ℝ, 0 < s ∧ s < 1 ∧ 0 < t ∧ t < 1 ∧
   M = (1 - s) • A + s • B ∧
   N = (1 - t) • A + t • C ∧
   s / (1 - s) + t / (1 - t) = 1) →
  ∃ u : ℝ, (1/3 : ℝ) • (A + B + C) = (1 - u) • M + u • N :=
by sorry

end NUMINAMATH_CALUDE_centroid_on_line_segment_l2310_231063


namespace NUMINAMATH_CALUDE_tangent_line_problem_l2310_231081

theorem tangent_line_problem (a : ℝ) : 
  (∃ (k : ℝ), 
    (∃ (x₀ : ℝ), 
      (x₀^3 = k * (x₀ - 1)) ∧ 
      (a * x₀^2 + 15/4 * x₀ - 9 = k * (x₀ - 1)) ∧
      (3 * x₀^2 = k))) →
  (a = -25/64 ∨ a = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l2310_231081


namespace NUMINAMATH_CALUDE_compute_expression_l2310_231058

theorem compute_expression : 
  20 * (200 / 3 + 36 / 9 + 16 / 25 + 3) = 13212.8 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2310_231058


namespace NUMINAMATH_CALUDE_paint_remaining_is_one_fourth_l2310_231050

/-- The fraction of paint remaining after three days of painting -/
def paint_remaining : ℚ :=
  let day1_remaining := 1 - (1/4 : ℚ)
  let day2_remaining := day1_remaining - (1/2 * day1_remaining)
  day2_remaining - (1/3 * day2_remaining)

/-- Theorem stating that the remaining paint after three days is 1/4 of the original amount -/
theorem paint_remaining_is_one_fourth :
  paint_remaining = (1/4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_paint_remaining_is_one_fourth_l2310_231050


namespace NUMINAMATH_CALUDE_multiply_mistake_l2310_231031

theorem multiply_mistake (x : ℝ) : 97 * x - 89 * x = 4926 → x = 615.75 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mistake_l2310_231031


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_conditions_l2310_231090

theorem smallest_positive_integer_satisfying_conditions (a : ℕ) :
  (∀ b : ℕ, b > 0 ∧ b % 3 = 1 ∧ 5 ∣ b → a ≤ b) ∧
  a > 0 ∧ a % 3 = 1 ∧ 5 ∣ a →
  a = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_conditions_l2310_231090


namespace NUMINAMATH_CALUDE_original_price_is_360_l2310_231035

/-- The original price of a product satisfies two conditions:
    1. When sold at 75% of the original price, there's a loss of $12 per item.
    2. When sold at 90% of the original price, there's a profit of $42 per item. -/
theorem original_price_is_360 (price : ℝ) 
    (h1 : 0.75 * price + 12 = 0.9 * price - 42) : 
    price = 360 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_360_l2310_231035


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2310_231011

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, (7 - 4 * x > 25) → x ≥ -5 ∧ (7 - 4 * (-5) > 25) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2310_231011


namespace NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2310_231071

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with vertex at origin and focus at (p/2, 0) -/
structure Parabola where
  p : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a parabola -/
def onParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point lies on a line -/
def onLine (point : Point) (line : Line) : Prop :=
  line.a * point.x + line.b * point.y + line.c = 0

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (p : Point) (q : Point) : Prop :=
  m.x = (p.x + q.x) / 2 ∧ m.y = (p.y + q.y) / 2

theorem parabola_intersection_midpoint 
  (parabola : Parabola)
  (C : Point)
  : 
  parabola.p = 2 ∧ C.x = 2 ∧ C.y = 1 →
  ∃ (l : Line) (M N : Point),
    l.a = 2 ∧ l.b = -1 ∧ l.c = -3 ∧
    onLine C l ∧
    onLine M l ∧ onLine N l ∧
    onParabola M parabola ∧ onParabola N parabola ∧
    isMidpoint C M N := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_midpoint_l2310_231071


namespace NUMINAMATH_CALUDE_odd_solutions_count_l2310_231032

theorem odd_solutions_count (x : ℕ) : 
  (∃ (s : Finset ℕ), 
    (∀ y ∈ s, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29) ∧ 
    (∀ y, 20 ≤ y ∧ y ≤ 150 ∧ Odd y ∧ (y + 17) % 29 = 65 % 29 → y ∈ s) ∧
    Finset.card s = 3) := by
  sorry

end NUMINAMATH_CALUDE_odd_solutions_count_l2310_231032


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_five_l2310_231027

theorem sqrt_equality_implies_one_and_five (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a < b) :
  Real.sqrt (1 + Real.sqrt (45 + 20 * Real.sqrt 5)) = Real.sqrt a + Real.sqrt b →
  a = 1 ∧ b = 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_one_and_five_l2310_231027
