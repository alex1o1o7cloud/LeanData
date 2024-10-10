import Mathlib

namespace finite_solutions_factorial_cube_plus_eight_l1426_142699

theorem finite_solutions_factorial_cube_plus_eight :
  {p : ℕ × ℕ | (p.1.factorial = p.2^3 + 8)}.Finite := by
  sorry

end finite_solutions_factorial_cube_plus_eight_l1426_142699


namespace peter_hunts_triple_mark_l1426_142686

/-- The number of animals hunted by each person in a day --/
structure HuntingData where
  sam : ℕ
  rob : ℕ
  mark : ℕ
  peter : ℕ

/-- The conditions of the hunting problem --/
def huntingProblem (h : HuntingData) : Prop :=
  h.sam = 6 ∧
  h.rob = h.sam / 2 ∧
  h.mark = (h.sam + h.rob) / 3 ∧
  h.sam + h.rob + h.mark + h.peter = 21

/-- The theorem stating that Peter hunts 3 times more animals than Mark --/
theorem peter_hunts_triple_mark (h : HuntingData) 
  (hcond : huntingProblem h) : h.peter = 3 * h.mark := by
  sorry

end peter_hunts_triple_mark_l1426_142686


namespace function_satisfying_cross_ratio_is_linear_l1426_142684

/-- A function satisfying the given cross-ratio condition is linear -/
theorem function_satisfying_cross_ratio_is_linear (f : ℝ → ℝ) :
  (∀ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    (a - b) / (b - c) + (a - d) / (d - c) = 0 →
    f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f d ∧ f a ≠ f c ∧ f a ≠ f d ∧ f b ≠ f d →
    (f a - f b) / (f b - f c) + (f a - f d) / (f d - f c) = 0) →
  ∃ (k m : ℝ), ∀ x, f x = k * x + m :=
by sorry

end function_satisfying_cross_ratio_is_linear_l1426_142684


namespace geometric_series_equality_l1426_142613

/-- Defines the sum of the first n terms of the geometric series A_n -/
def A (n : ℕ) : ℚ := 704 * (1 - (1/2)^n) / (1 - 1/2)

/-- Defines the sum of the first n terms of the geometric series B_n -/
def B (n : ℕ) : ℚ := 1984 * (1 - (1/(-2))^n) / (1 + 1/2)

/-- Proves that the smallest positive integer n for which A_n = B_n is 5 -/
theorem geometric_series_equality :
  ∀ n : ℕ, n ≥ 1 → (A n = B n ↔ n = 5) :=
sorry

end geometric_series_equality_l1426_142613


namespace fraction_sum_equality_l1426_142650

theorem fraction_sum_equality : 
  (2 + 4 + 6 + 8 + 10) / (1 + 3 + 5 + 7 + 9) + 
  (1 + 3 + 5 + 7 + 9) / (2 + 4 + 6 + 8 + 10) = 61 / 30 := by
  sorry

end fraction_sum_equality_l1426_142650


namespace expression_value_l1426_142695

theorem expression_value : ∀ a b : ℝ, 
  (a - 2)^2 + |b + 3| = 0 → 
  3*a^2*b - (2*a*b^2 - 2*(a*b - 3/2*a^2*b) + a*b) + 3*a*b^2 = 12 :=
by sorry

end expression_value_l1426_142695


namespace cab_driver_average_income_l1426_142663

theorem cab_driver_average_income (incomes : List ℝ) 
  (h1 : incomes = [300, 150, 750, 200, 600]) : 
  (incomes.sum / incomes.length) = 400 := by
  sorry

end cab_driver_average_income_l1426_142663


namespace equation_transformation_l1426_142641

theorem equation_transformation (x y : ℝ) (h : y = x - 1/x) :
  x^6 + x^5 - 5*x^4 + 2*x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 3) = 0 :=
by sorry

end equation_transformation_l1426_142641


namespace cube_minus_reciprocal_cube_l1426_142615

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end cube_minus_reciprocal_cube_l1426_142615


namespace haley_albums_l1426_142635

theorem haley_albums (total_pics : ℕ) (first_album_pics : ℕ) (pics_per_album : ℕ) 
  (h1 : total_pics = 65)
  (h2 : first_album_pics = 17)
  (h3 : pics_per_album = 8) :
  (total_pics - first_album_pics) / pics_per_album = 6 := by
  sorry

end haley_albums_l1426_142635


namespace eldest_age_is_32_l1426_142661

/-- Represents the ages of three people A, B, and C -/
structure Ages where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The present ages are in the ratio 5:7:8 -/
def present_ratio (ages : Ages) : Prop :=
  7 * ages.a = 5 * ages.b ∧ 8 * ages.a = 5 * ages.c

/-- The sum of ages 7 years ago was 59 -/
def past_sum (ages : Ages) : Prop :=
  (ages.a - 7) + (ages.b - 7) + (ages.c - 7) = 59

/-- Theorem stating that given the conditions, the eldest person's age is 32 -/
theorem eldest_age_is_32 (ages : Ages) 
  (h1 : present_ratio ages) 
  (h2 : past_sum ages) : 
  ages.c = 32 := by
  sorry


end eldest_age_is_32_l1426_142661


namespace trapezoid_diagonal_relation_l1426_142687

-- Define a structure for a trapezoid
structure Trapezoid where
  a : ℝ  -- larger base
  c : ℝ  -- smaller base
  e : ℝ  -- diagonal
  f : ℝ  -- diagonal
  d : ℝ  -- side
  b : ℝ  -- side
  h_ac : a > c  -- condition that a > c

-- Theorem statement
theorem trapezoid_diagonal_relation (T : Trapezoid) :
  (T.e^2 + T.f^2) / (T.a^2 - T.b^2) = (T.a + T.c) / (T.a - T.c) := by
  sorry

end trapezoid_diagonal_relation_l1426_142687


namespace sum_of_coefficients_for_specific_polynomial_l1426_142690

/-- A polynomial with real coefficients -/
def RealPolynomial (p q r s : ℝ) : ℂ → ℂ :=
  fun x => x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem: Sum of coefficients for a specific polynomial -/
theorem sum_of_coefficients_for_specific_polynomial
  (p q r s : ℝ) :
  (RealPolynomial p q r s (3*I) = 0) →
  (RealPolynomial p q r s (3+I) = 0) →
  p + q + r + s = 49 := by
  sorry

end sum_of_coefficients_for_specific_polynomial_l1426_142690


namespace new_scheme_fixed_salary_is_1000_l1426_142623

/-- Represents the salesman's compensation scheme -/
structure CompensationScheme where
  fixedSalary : ℕ
  commissionRate : ℚ
  commissionThreshold : ℕ

/-- Calculates the total compensation for a given sales amount and compensation scheme -/
def calculateCompensation (sales : ℕ) (scheme : CompensationScheme) : ℚ :=
  scheme.fixedSalary + scheme.commissionRate * max (sales - scheme.commissionThreshold) 0

/-- Theorem stating that the fixed salary in the new scheme is 1000 -/
theorem new_scheme_fixed_salary_is_1000 (totalSales : ℕ) (oldScheme newScheme : CompensationScheme) :
  totalSales = 12000 →
  oldScheme.fixedSalary = 0 →
  oldScheme.commissionRate = 1/20 →
  oldScheme.commissionThreshold = 0 →
  newScheme.commissionRate = 1/40 →
  newScheme.commissionThreshold = 4000 →
  calculateCompensation totalSales newScheme = calculateCompensation totalSales oldScheme + 600 →
  newScheme.fixedSalary = 1000 := by
  sorry

#eval calculateCompensation 12000 { fixedSalary := 1000, commissionRate := 1/40, commissionThreshold := 4000 }
#eval calculateCompensation 12000 { fixedSalary := 0, commissionRate := 1/20, commissionThreshold := 0 }

end new_scheme_fixed_salary_is_1000_l1426_142623


namespace max_min_difference_l1426_142679

def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem max_min_difference (M N : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ M) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = M) ∧
  (∀ x ∈ Set.Icc (-3) 2, N ≤ f x) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = N) →
  M - N = 20 :=
by sorry

end max_min_difference_l1426_142679


namespace identical_solutions_quadratic_linear_l1426_142620

theorem identical_solutions_quadratic_linear (k : ℝ) :
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k ∧ 
   ∀ x' y' : ℝ, y' = x'^2 ∧ y' = 4*x' + k → x' = x ∧ y' = y) ↔ k = -4 :=
by sorry

end identical_solutions_quadratic_linear_l1426_142620


namespace gcd_98_75_l1426_142675

theorem gcd_98_75 : Nat.gcd 98 75 = 1 := by
  sorry

end gcd_98_75_l1426_142675


namespace base_10_to_base_7_l1426_142625

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    875 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 0 := by
  sorry

end base_10_to_base_7_l1426_142625


namespace expression_simplification_l1426_142619

theorem expression_simplification (x : ℝ) (h : x = (1/2)⁻¹) :
  (x^2 - 2*x + 1) / (x^2 - 1) * (1 + 1/x) = 1/2 := by
  sorry

end expression_simplification_l1426_142619


namespace range_of_a_l1426_142652

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end range_of_a_l1426_142652


namespace parallel_line_k_value_l1426_142644

/-- Given a line passing through (3, -5) and (k, 21) that is parallel to 4x - 5y = 20, prove k = 35.5 -/
theorem parallel_line_k_value (k : ℝ) :
  (∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (x = 3 ∧ y = -5) ∨ (x = k ∧ y = 21)) ∧
                 (∀ x y : ℝ, y = (4/5) * x - 4 ↔ 4*x - 5*y = 20)) →
  k = 35.5 := by
  sorry

end parallel_line_k_value_l1426_142644


namespace inscribed_circle_radius_l1426_142662

theorem inscribed_circle_radius (s : ℝ) (r : ℝ) (h : s > 0) :
  3 * s = π * r^2 ∧ r = (s * Real.sqrt 3) / 6 →
  r = 6 * Real.sqrt 3 / π :=
by sorry

end inscribed_circle_radius_l1426_142662


namespace shortest_distance_on_specific_cone_l1426_142643

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of a cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ  -- Angle from a reference line on the surface

/-- Calculates the shortest distance between two points on the surface of a cone -/
def shortestDistanceOnCone (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem shortest_distance_on_specific_cone :
  let c : Cone := { baseRadius := 500, height := 400 }
  let p1 : ConePoint := { distanceFromVertex := 150, angle := 0 }
  let p2 : ConePoint := { distanceFromVertex := 400 * Real.sqrt 2, angle := π }
  shortestDistanceOnCone c p1 p2 = 25 * Real.sqrt 741 := by
  sorry

end shortest_distance_on_specific_cone_l1426_142643


namespace lcm_of_ratio_and_hcf_l1426_142660

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  a.val * 13 = b.val * 7 →
  Nat.gcd a.val b.val = 23 →
  Nat.lcm a.val b.val = 2093 := by
sorry

end lcm_of_ratio_and_hcf_l1426_142660


namespace distinct_z_values_exist_l1426_142668

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values_exist :
  ∃ (x y : ℕ), is_four_digit x ∧ is_four_digit y ∧ 
  y = reverse_digits x ∧ z x ≠ z y :=
sorry

end distinct_z_values_exist_l1426_142668


namespace equilateral_is_isosceles_l1426_142651

-- Define a triangle type
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

-- Define what it means for a triangle to be equilateral
def IsEquilateral (t : Triangle) : Prop :=
  t.side1 = t.side2 ∧ t.side2 = t.side3

-- Define what it means for a triangle to be isosceles
def IsIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side2 = t.side3 ∨ t.side3 = t.side1

-- Theorem: Every equilateral triangle is isosceles
theorem equilateral_is_isosceles (t : Triangle) :
  IsEquilateral t → IsIsosceles t := by
  sorry


end equilateral_is_isosceles_l1426_142651


namespace power_product_l1426_142603

theorem power_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_product_l1426_142603


namespace triangle_equation_no_real_roots_l1426_142617

theorem triangle_equation_no_real_roots 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∀ x : ℝ, a^2 * x^2 - (c^2 - a^2 - b^2) * x + b^2 ≠ 0 := by
sorry

end triangle_equation_no_real_roots_l1426_142617


namespace smallest_b_value_l1426_142633

theorem smallest_b_value (a b c : ℕ+) (h : (31 : ℚ) / 72 = a / 8 + b / 9 - c) : 
  ∀ b' : ℕ+, b' < b → ¬∃ (a' c' : ℕ+), (31 : ℚ) / 72 = a' / 8 + b' / 9 - c' :=
by sorry

end smallest_b_value_l1426_142633


namespace restaurant_hotdogs_l1426_142698

theorem restaurant_hotdogs (hotdogs : ℕ) (pizzas : ℕ) : 
  pizzas = hotdogs + 40 →
  30 * (hotdogs + pizzas) = 4800 →
  hotdogs = 60 := by
sorry

end restaurant_hotdogs_l1426_142698


namespace pond_length_l1426_142680

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 15)
  (h_depth : depth = 5)
  (h_volume : volume = 1500)
  : volume / (width * depth) = 20 := by
  sorry

end pond_length_l1426_142680


namespace yeonseo_skirt_count_l1426_142666

/-- Given that Yeonseo has more than two types of skirts and pants each,
    there are 4 types of pants, and 7 ways to choose pants or skirts,
    prove that the number of types of skirts is 3. -/
theorem yeonseo_skirt_count :
  ∀ (S P : ℕ),
  S > 2 →
  P > 2 →
  P = 4 →
  S + P = 7 →
  S = 3 := by
sorry

end yeonseo_skirt_count_l1426_142666


namespace arithmetic_sequence_length_l1426_142637

/-- An arithmetic sequence starting with 1, having a common difference of -2, and ending with -89, has 46 terms. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ → ℤ), 
    (a 0 = 1) →  -- First term is 1
    (∀ n, a (n + 1) - a n = -2) →  -- Common difference is -2
    (∃ N, a N = -89 ∧ ∀ k, k > N → a k < -89) →  -- Sequence ends at -89
    (∃ N, N = 46 ∧ a (N - 1) = -89) :=
by sorry

end arithmetic_sequence_length_l1426_142637


namespace sequence_is_arithmetic_l1426_142669

/-- Definition of the sequence sum -/
def S (n : ℕ) : ℕ := n^2

/-- Definition of the sequence terms -/
def a (n : ℕ) : ℕ := S (n + 1) - S n

/-- Proposition: The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∃ (d : ℕ), ∀ (n : ℕ), a (n + 1) = a n + d := by
  sorry

end sequence_is_arithmetic_l1426_142669


namespace log_stack_sum_l1426_142608

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 5 logs
Prove that the total number of logs in the stack is 110.
-/
theorem log_stack_sum : 
  ∀ (a l n : ℕ), 
    a = 15 → 
    l = 5 → 
    n = a - l + 1 → 
    (n : ℚ) / 2 * (a + l) = 110 := by
  sorry

end log_stack_sum_l1426_142608


namespace B_equals_D_l1426_142653

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D (real numbers not less than 1)
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem statement
theorem B_equals_D : B = D := by sorry

end B_equals_D_l1426_142653


namespace elliptical_cone_theorem_l1426_142682

/-- Given a cone with a 30° aperture and an elliptical base, 
    prove that the square of the minor axis of the ellipse 
    is equal to the product of the shortest and longest slant heights of the cone. -/
theorem elliptical_cone_theorem (b : ℝ) (AC BC : ℝ) : 
  b > 0 → AC > 0 → BC > 0 → (2 * b)^2 = AC * BC := by
  sorry

end elliptical_cone_theorem_l1426_142682


namespace tenth_replacement_in_january_l1426_142647

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Calculates the month after a given number of months have passed -/
def monthAfter (start : Month) (months : ℕ) : Month := sorry

/-- The number of months between battery replacements -/
def replacementInterval : ℕ := 4

/-- The ordinal number of the replacement we're interested in -/
def targetReplacement : ℕ := 10

/-- Theorem stating that the 10th replacement will occur in January -/
theorem tenth_replacement_in_january :
  monthAfter Month.January ((targetReplacement - 1) * replacementInterval) = Month.January := by
  sorry

end tenth_replacement_in_january_l1426_142647


namespace product_35_42_base7_l1426_142654

/-- Converts a base-7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-7 number --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Main theorem --/
theorem product_35_42_base7 :
  let a := toDecimal 35
  let b := toDecimal 42
  let product := a * b
  let base7Product := toBase7 product
  let digitSum := sumOfDigitsBase7 base7Product
  digitSum = 5 ∧ digitSum % 5 = 5 := by sorry

end product_35_42_base7_l1426_142654


namespace jerrys_age_l1426_142639

theorem jerrys_age (mickey_age jerry_age : ℝ) : 
  mickey_age = 2.5 * jerry_age - 5 →
  mickey_age = 20 →
  jerry_age = 10 := by
sorry

end jerrys_age_l1426_142639


namespace prime_power_divides_l1426_142694

theorem prime_power_divides (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → p ∣ a^n → p^n ∣ a^n := by sorry

end prime_power_divides_l1426_142694


namespace dimes_in_jar_l1426_142618

/-- The number of dimes in a jar with equal numbers of dimes, quarters, and half-dollars totaling $20.40 -/
def num_dimes : ℕ := 24

/-- The total value of coins in cents -/
def total_value : ℕ := 2040

theorem dimes_in_jar : 
  10 * num_dimes + 25 * num_dimes + 50 * num_dimes = total_value := by
  sorry

end dimes_in_jar_l1426_142618


namespace peanut_difference_l1426_142626

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end peanut_difference_l1426_142626


namespace sum_of_x_and_y_l1426_142689

theorem sum_of_x_and_y (x y : ℝ) (h : x - 1 = 1 - y) : x + y = 2 := by
  sorry

#check sum_of_x_and_y

end sum_of_x_and_y_l1426_142689


namespace angle_range_l1426_142676

def a (x : ℝ) : Fin 2 → ℝ := ![2, x]
def b : Fin 2 → ℝ := ![1, 3]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem angle_range (x : ℝ) :
  is_acute_angle (a x) b → x ∈ {y : ℝ | y > -2/3 ∧ y ≠ -2/3} := by
  sorry

end angle_range_l1426_142676


namespace sum_of_hundred_consecutive_integers_l1426_142655

theorem sum_of_hundred_consecutive_integers : ∃ k : ℕ, 
  50 * (2 * k + 99) = 1627384950 := by
  sorry

end sum_of_hundred_consecutive_integers_l1426_142655


namespace georges_earnings_l1426_142672

-- Define the daily wages and hours worked
def monday_wage : ℝ := 5
def monday_hours : ℝ := 7
def tuesday_wage : ℝ := 6
def tuesday_hours : ℝ := 2
def wednesday_wage : ℝ := 4
def wednesday_hours : ℝ := 5
def saturday_wage : ℝ := 7
def saturday_hours : ℝ := 3

-- Define the tax rate and uniform fee
def tax_rate : ℝ := 0.1
def uniform_fee : ℝ := 15

-- Calculate total earnings before deductions
def total_earnings : ℝ := 
  monday_wage * monday_hours + 
  tuesday_wage * tuesday_hours + 
  wednesday_wage * wednesday_hours + 
  saturday_wage * saturday_hours

-- Calculate earnings after tax deduction
def earnings_after_tax : ℝ := total_earnings * (1 - tax_rate)

-- Calculate final earnings after uniform fee deduction
def final_earnings : ℝ := earnings_after_tax - uniform_fee

-- Theorem statement
theorem georges_earnings : final_earnings = 64.2 := by
  sorry

end georges_earnings_l1426_142672


namespace cos_315_degrees_l1426_142609

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l1426_142609


namespace first_month_sale_l1426_142610

def sales_month_2_to_6 : List ℕ := [6927, 6855, 7230, 6562, 4891]
def average_sale : ℕ := 6500
def number_of_months : ℕ := 6

theorem first_month_sale :
  (average_sale * number_of_months - sales_month_2_to_6.sum) = 6535 := by
  sorry

end first_month_sale_l1426_142610


namespace inequality_reversal_l1426_142645

theorem inequality_reversal (a b : ℝ) (h : a > b) : ¬(a / (-2) > b / (-2)) := by
  sorry

end inequality_reversal_l1426_142645


namespace union_of_A_and_B_l1426_142683

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end union_of_A_and_B_l1426_142683


namespace other_juice_cost_is_five_l1426_142658

/-- Represents the cost and quantity information for a juice bar order --/
structure JuiceOrder where
  totalSpent : ℕ
  pineappleCost : ℕ
  pineappleSpent : ℕ
  totalPeople : ℕ

/-- Calculates the cost per glass of the other type of juice --/
def otherJuiceCost (order : JuiceOrder) : ℕ :=
  let pineappleGlasses := order.pineappleSpent / order.pineappleCost
  let otherGlasses := order.totalPeople - pineappleGlasses
  let otherSpent := order.totalSpent - order.pineappleSpent
  otherSpent / otherGlasses

/-- Theorem stating that the cost of the other type of juice is $5 per glass --/
theorem other_juice_cost_is_five (order : JuiceOrder) 
  (h1 : order.totalSpent = 94)
  (h2 : order.pineappleCost = 6)
  (h3 : order.pineappleSpent = 54)
  (h4 : order.totalPeople = 17) :
  otherJuiceCost order = 5 := by
  sorry

end other_juice_cost_is_five_l1426_142658


namespace town_population_theorem_l1426_142624

theorem town_population_theorem (total_population : ℕ) 
  (females_with_glasses : ℕ) (female_glasses_percentage : ℚ) :
  total_population = 5000 →
  females_with_glasses = 900 →
  female_glasses_percentage = 30/100 →
  (females_with_glasses : ℚ) / female_glasses_percentage = 3000 →
  total_population - 3000 = 2000 := by
sorry

end town_population_theorem_l1426_142624


namespace geometric_sequence_sum_r_value_l1426_142685

-- Define the sum of the first n terms of the geometric sequence
def S (n : ℕ) (r : ℚ) : ℚ := 3^(n-1) - r

-- Define the geometric sequence
def a (n : ℕ) (r : ℚ) : ℚ := S n r - S (n-1) r

-- Theorem statement
theorem geometric_sequence_sum_r_value :
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 2 → a n r = 2 * 3^(n-2) ∧ a 1 r = 1 - r → r = 1/3 := by
  sorry

end geometric_sequence_sum_r_value_l1426_142685


namespace geometric_sequence_11th_term_l1426_142632

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The 11th term of a geometric sequence is 648, given that its 5th term is 8 and its 8th term is 72. -/
theorem geometric_sequence_11th_term (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h5 : a 5 = 8) (h8 : a 8 = 72) : a 11 = 648 := by
  sorry

end geometric_sequence_11th_term_l1426_142632


namespace rectangle_new_perimeter_l1426_142648

/-- Given a rectangle with width 10 meters and original area 150 square meters,
    if its length is increased so that the new area is 1 (1/3) times the original area,
    then the new perimeter is 60 meters. -/
theorem rectangle_new_perimeter (width : ℝ) (original_area : ℝ) (new_area : ℝ) :
  width = 10 →
  original_area = 150 →
  new_area = original_area * (4/3) →
  2 * (width + new_area / width) = 60 :=
by
  sorry


end rectangle_new_perimeter_l1426_142648


namespace no_valid_operation_l1426_142649

def equation (op : ℝ → ℝ → ℝ) : Prop :=
  op 8 2 * 3 + 7 - (5 - 3) = 16

theorem no_valid_operation : 
  ¬ (equation (·/·) ∨ equation (·*·) ∨ equation (·+·) ∨ equation (·-·)) :=
by sorry

end no_valid_operation_l1426_142649


namespace ceiling_fraction_evaluation_l1426_142634

theorem ceiling_fraction_evaluation :
  (⌈(19 : ℚ) / 11 - ⌈(35 : ℚ) / 22⌉⌉) / (⌈(35 : ℚ) / 11 + ⌈(11 * 22 : ℚ) / 35⌉⌉) = 1 / 10 := by
  sorry

end ceiling_fraction_evaluation_l1426_142634


namespace intersection_A_B_union_A_complement_B_range_of_a_l1426_142630

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1
theorem intersection_A_B : A ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem union_A_complement_B : A ∪ (U \ B) = {x | x < 4} := by sorry

-- Theorem 3
theorem range_of_a (h : A ⊆ C a) : a ≥ 4 := by sorry

end intersection_A_B_union_A_complement_B_range_of_a_l1426_142630


namespace inverse_function_point_and_sum_l1426_142674

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverses of each other
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Given condition: (2,3) is on the graph of y = f(x)/2
axiom point_on_f : f 2 = 6

-- Theorem to prove
theorem inverse_function_point_and_sum :
  (f_inv 6 = 2) ∧ (6, 1) ∈ {p : ℝ × ℝ | p.2 = f_inv p.1 / 2} ∧ (6 + 1 = 7) :=
sorry

end inverse_function_point_and_sum_l1426_142674


namespace solution_equation_l1426_142622

theorem solution_equation (p q : ℝ) (h1 : p ≠ q) (h2 : p ≠ 0) (h3 : q ≠ 0) :
  ∃ x : ℝ, (x + p)^2 - (x + q)^2 = 4*(p-q)^2 ∧ x = 2*p - 2*q :=
by sorry

end solution_equation_l1426_142622


namespace x_value_l1426_142638

variables (x y z k l m : ℝ)

theorem x_value (h1 : x * y = k * (x + y))
                (h2 : x * z = l * (x + z))
                (h3 : y * z = m * (y + z))
                (hk : k ≠ 0) (hl : l ≠ 0) (hm : m ≠ 0)
                (hkl : k * l + k * m - l * m ≠ 0) :
  x = (2 * k * l * m) / (k * l + k * m - l * m) :=
by sorry

end x_value_l1426_142638


namespace group_size_calculation_l1426_142614

theorem group_size_calculation (average_increase : ℝ) (original_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3.5 →
  original_weight = 75 →
  new_weight = 99.5 →
  (new_weight - original_weight) / average_increase = 7 := by
sorry

end group_size_calculation_l1426_142614


namespace find_m_l1426_142605

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

-- State the theorem
theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, f (x - 1) = |x| - |x - 2|) ∧ 
  f (f m) = f 2002 - 7/2 → 
  m = -3/8 := by sorry

end find_m_l1426_142605


namespace bricks_decrease_by_one_l1426_142616

/-- Represents a brick wall with a given number of rows, total bricks, and bricks in the bottom row. -/
structure BrickWall where
  rows : ℕ
  totalBricks : ℕ
  bottomRowBricks : ℕ

/-- Calculates the number of bricks in a given row of the wall. -/
def bricksInRow (wall : BrickWall) (row : ℕ) : ℕ :=
  wall.bottomRowBricks - (row - 1)

/-- Theorem stating that for a specific brick wall, the number of bricks decreases by 1 in each row going up. -/
theorem bricks_decrease_by_one (wall : BrickWall)
    (h1 : wall.rows = 5)
    (h2 : wall.totalBricks = 200)
    (h3 : wall.bottomRowBricks = 38) :
    ∀ row : ℕ, row > 1 → row ≤ wall.rows →
      bricksInRow wall row = bricksInRow wall (row - 1) - 1 := by
  sorry

end bricks_decrease_by_one_l1426_142616


namespace positive_reals_inequality_l1426_142678

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x - y) : x^2 + 4*y^2 < 1 := by
  sorry

end positive_reals_inequality_l1426_142678


namespace state_a_selection_percentage_l1426_142665

theorem state_a_selection_percentage :
  ∀ (total_candidates : ℕ) (state_b_percentage : ℚ) (additional_selected : ℕ),
    total_candidates = 8000 →
    state_b_percentage = 7 / 100 →
    additional_selected = 80 →
    ∃ (state_a_percentage : ℚ),
      state_a_percentage * total_candidates + additional_selected = state_b_percentage * total_candidates ∧
      state_a_percentage = 6 / 100 := by
  sorry

end state_a_selection_percentage_l1426_142665


namespace simplify_expression_l1426_142628

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 + 1/2) :
  (a - Real.sqrt 3) * (a + Real.sqrt 3) - a * (a - 6) = 6 * Real.sqrt 3 := by
  sorry

end simplify_expression_l1426_142628


namespace fathers_age_l1426_142627

theorem fathers_age (man_age father_age : ℝ) : 
  man_age = (2 / 5) * father_age ∧ 
  man_age + 5 = (1 / 2) * (father_age + 5) → 
  father_age = 25 := by
sorry

end fathers_age_l1426_142627


namespace line_intersects_circle_l1426_142664

/-- Given a point (a,b) outside a circle x^2 + y^2 = r^2 (r ≠ 0),
    prove that the line ax + by = r^2 intersects the circle. -/
theorem line_intersects_circle 
  (a b r : ℝ) 
  (r_nonzero : r ≠ 0)
  (point_outside : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 :=
sorry

end line_intersects_circle_l1426_142664


namespace max_value_of_expression_l1426_142631

theorem max_value_of_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2-a)^2 * (2-b)^2 * (2-c)^2)) ≤ 16 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ a' ≤ 2 ∧ 
                    0 ≤ b' ∧ b' ≤ 2 ∧ 
                    0 ≤ c' ∧ c' ≤ 2 ∧ 
                    Real.sqrt (a'^2 * b'^2 * c'^2) + Real.sqrt ((2-a')^2 * (2-b')^2 * (2-c')^2) = 16 :=
by sorry

end max_value_of_expression_l1426_142631


namespace worker_b_completion_time_worker_b_time_is_9_l1426_142657

/-- Given workers a, b, and c who can complete a task together or individually,
    this theorem proves the time taken by worker b to complete the task alone. -/
theorem worker_b_completion_time
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  rate_b = 1 / 9 := by
  sorry

/-- The time taken by worker b to complete the task alone -/
def time_b : ℝ := 9

/-- Proves that the time taken by worker b is indeed 9 days -/
theorem worker_b_time_is_9
  (total_rate : ℝ)
  (rate_a : ℝ)
  (rate_b : ℝ)
  (rate_c : ℝ)
  (h1 : total_rate = rate_a + rate_b + rate_c)
  (h2 : total_rate = 1 / 4)
  (h3 : rate_a = 1 / 12)
  (h4 : rate_c = 1 / 18) :
  time_b = 1 / rate_b := by
  sorry

end worker_b_completion_time_worker_b_time_is_9_l1426_142657


namespace compound_proposition_falsehood_l1426_142602

theorem compound_proposition_falsehood (p q : Prop) 
  (hp : p) (hq : q) : 
  (p ∨ q) ∧ (p ∧ q) ∧ ¬(¬p ∧ q) ∧ (¬p ∨ q) := by
  sorry

end compound_proposition_falsehood_l1426_142602


namespace fishing_ratio_proof_l1426_142667

/-- The ratio of Brian's fishing trips to Chris's fishing trips -/
def fishing_ratio : ℚ := 26/15

theorem fishing_ratio_proof (brian_catch : ℕ) (total_catch : ℕ) (chris_trips : ℕ) :
  brian_catch = 400 →
  total_catch = 13600 →
  chris_trips = 10 →
  (∃ (brian_trips : ℚ),
    brian_trips * brian_catch + chris_trips * (brian_catch * 5/3) = total_catch ∧
    brian_trips / chris_trips = fishing_ratio) :=
by sorry

end fishing_ratio_proof_l1426_142667


namespace total_amount_correct_l1426_142671

/-- Represents the total amount lent out in rupees -/
def total_amount : ℝ := 11501.6

/-- Represents the amount lent at 8% p.a. in rupees -/
def amount_at_8_percent : ℝ := 15008

/-- Represents the total interest received after one year in rupees -/
def total_interest : ℝ := 850

/-- Theorem stating that the total amount lent out is correct given the conditions -/
theorem total_amount_correct :
  ∃ (amount_at_10_percent : ℝ),
    amount_at_8_percent + amount_at_10_percent = total_amount ∧
    0.08 * amount_at_8_percent + 0.1 * amount_at_10_percent = total_interest :=
by sorry

end total_amount_correct_l1426_142671


namespace number_line_relations_l1426_142640

/-- Definition of "A is k related to B" --/
def is_k_related (A B C : ℝ) (k : ℝ) : Prop :=
  |C - A| = k * |C - B| ∧ k > 1

/-- Problem statement --/
theorem number_line_relations (x t k : ℝ) : 
  let A := -3
  let B := 6
  let P := x
  let Q := 6 - 2*t
  (
    /- Part 1 -/
    (is_k_related A B P 2 → x = 3) ∧ 
    
    /- Part 2 -/
    (|x + 2| + |x - 1| = 3 ∧ is_k_related A B P k → 1/8 ≤ k ∧ k ≤ 4/5) ∧
    
    /- Part 3 -/
    (is_k_related (-3 + t) A Q 3 → t = 3/2)
  ) := by sorry

end number_line_relations_l1426_142640


namespace johann_delivery_correct_l1426_142642

/-- The number of pieces Johann needs to deliver -/
def johann_delivery (total friend1 friend2 friend3 friend4 : ℕ) : ℕ :=
  total - (friend1 + friend2 + friend3 + friend4)

/-- Theorem stating that Johann's delivery is correct -/
theorem johann_delivery_correct (total friend1 friend2 friend3 friend4 : ℕ) 
  (h_total : total = 250)
  (h_friend1 : friend1 = 35)
  (h_friend2 : friend2 = 42)
  (h_friend3 : friend3 = 38)
  (h_friend4 : friend4 = 45) :
  johann_delivery total friend1 friend2 friend3 friend4 = 90 := by
  sorry

#eval johann_delivery 250 35 42 38 45

end johann_delivery_correct_l1426_142642


namespace akeno_spent_more_l1426_142604

def akeno_expenditure : ℕ := 2985
def lev_expenditure : ℕ := akeno_expenditure / 3
def ambrocio_expenditure : ℕ := lev_expenditure - 177

theorem akeno_spent_more :
  akeno_expenditure - (lev_expenditure + ambrocio_expenditure) = 1172 := by
  sorry

end akeno_spent_more_l1426_142604


namespace x_value_l1426_142646

theorem x_value (x : ℝ) (h_pos : x > 0) (h_percent : x * (x / 100) = 9) (h_multiple : ∃ k : ℤ, x = 3 * k) : x = 30 := by
  sorry

end x_value_l1426_142646


namespace component_qualification_l1426_142621

def lower_limit : ℝ := 20 - 0.05
def upper_limit : ℝ := 20 + 0.02

def is_qualified (diameter : ℝ) : Prop :=
  lower_limit ≤ diameter ∧ diameter ≤ upper_limit

theorem component_qualification :
  is_qualified 19.96 ∧
  ¬is_qualified 19.50 ∧
  ¬is_qualified 20.2 ∧
  ¬is_qualified 20.05 := by
  sorry

end component_qualification_l1426_142621


namespace toms_age_l1426_142677

theorem toms_age (j t : ℕ) 
  (h1 : j - 6 = 3 * (t - 6))  -- John was thrice as old as Tom 6 years ago
  (h2 : j + 4 = 2 * (t + 4))  -- John will be 2 times as old as Tom in 4 years
  : t = 16 := by  -- Tom's current age is 16
  sorry

end toms_age_l1426_142677


namespace toy_boxes_theorem_l1426_142697

/-- Represents the number of toy cars in each box -/
structure ToyBoxes :=
  (box1 : ℕ)
  (box2 : ℕ)
  (box3 : ℕ)
  (box4 : ℕ)
  (box5 : ℕ)

/-- The initial state of the toy boxes -/
def initial_state : ToyBoxes :=
  { box1 := 21
  , box2 := 31
  , box3 := 19
  , box4 := 45
  , box5 := 27 }

/-- The final state of the toy boxes after moving 12 cars from box1 to box4 -/
def final_state : ToyBoxes :=
  { box1 := 9
  , box2 := 31
  , box3 := 19
  , box4 := 57
  , box5 := 27 }

/-- The number of cars moved from box1 to box4 -/
def cars_moved : ℕ := 12

theorem toy_boxes_theorem (initial : ToyBoxes) (final : ToyBoxes) (moved : ℕ) :
  initial = initial_state →
  moved = cars_moved →
  final.box1 = initial.box1 - moved ∧
  final.box2 = initial.box2 ∧
  final.box3 = initial.box3 ∧
  final.box4 = initial.box4 + moved ∧
  final.box5 = initial.box5 →
  final = final_state :=
by sorry

end toy_boxes_theorem_l1426_142697


namespace soda_pizza_ratio_is_one_to_two_l1426_142659

/-- Represents the cost of items and the number of people -/
structure PurchaseInfo where
  num_people : ℕ
  pizza_cost : ℚ
  total_spent : ℚ

/-- Calculates the ratio of soda cost to pizza cost -/
def soda_to_pizza_ratio (info : PurchaseInfo) : ℚ × ℚ :=
  let pizza_total := info.pizza_cost * info.num_people
  let soda_total := info.total_spent - pizza_total
  let soda_cost := soda_total / info.num_people
  (soda_cost, info.pizza_cost)

/-- Theorem stating the ratio of soda cost to pizza cost is 1:2 -/
theorem soda_pizza_ratio_is_one_to_two (info : PurchaseInfo) 
  (h1 : info.num_people = 6)
  (h2 : info.pizza_cost = 1)
  (h3 : info.total_spent = 9) :
  soda_to_pizza_ratio info = (1/2, 1) := by
  sorry

end soda_pizza_ratio_is_one_to_two_l1426_142659


namespace cubic_polynomial_integer_root_l1426_142601

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a point -/
def CubicPolynomial.eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- Property of having infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinitely_many_equal_products (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P.eval x = y * P.eval y ∧ (abs x > n ∨ abs y > n)

/-- Main theorem: If a cubic polynomial with integer coefficients has infinitely many pairs of 
    distinct integers (x, y) such that xP(x) = yP(y), then it has an integer root -/
theorem cubic_polynomial_integer_root (P : CubicPolynomial) 
    (h : has_infinitely_many_equal_products P) : 
    ∃ k : ℤ, P.eval k = 0 := by
  sorry

end cubic_polynomial_integer_root_l1426_142601


namespace spaceship_journey_theorem_l1426_142600

/-- A spaceship's journey to another planet -/
def spaceship_journey (total_journey_time : ℕ) (initial_travel_time : ℕ) (first_break : ℕ) (second_travel_time : ℕ) (second_break : ℕ) (travel_segment : ℕ) (break_duration : ℕ) : Prop :=
  let total_hours : ℕ := total_journey_time * 24
  let initial_breaks : ℕ := first_break + second_break
  let initial_total_time : ℕ := initial_travel_time + second_travel_time + initial_breaks
  let remaining_time : ℕ := total_hours - initial_total_time
  let full_segments : ℕ := remaining_time / (travel_segment + break_duration)
  let total_breaks : ℕ := initial_breaks + full_segments * break_duration
  total_breaks = 8

theorem spaceship_journey_theorem :
  spaceship_journey 3 10 3 10 1 11 1 := by sorry

end spaceship_journey_theorem_l1426_142600


namespace max_books_borrowed_l1426_142636

theorem max_books_borrowed (total_students : Nat) (no_books : Nat) (one_book : Nat) (two_books : Nat) 
  (avg_books : Nat) (h1 : total_students = 32) (h2 : no_books = 2) (h3 : one_book = 12) (h4 : two_books = 10) 
  (h5 : avg_books = 2) : 
  ∃ (max_books : Nat), max_books = 11 ∧ 
  (∀ (student_books : Nat), student_books ≤ max_books) ∧
  (∃ (rest_books : Nat), 
    rest_books * (total_students - no_books - one_book - two_books) + 
    no_books * 0 + one_book * 1 + two_books * 2 + max_books = 
    total_students * avg_books) :=
by sorry

end max_books_borrowed_l1426_142636


namespace complex_fraction_simplification_l1426_142688

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof goes here
  sorry

end complex_fraction_simplification_l1426_142688


namespace cell_division_3_hours_l1426_142670

/-- The number of cells after a given number of 30-minute intervals -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- The number of 30-minute intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_division_3_hours : 
  num_cells intervals_in_3_hours = 128 := by
  sorry

end cell_division_3_hours_l1426_142670


namespace specific_glued_cubes_surface_area_l1426_142611

/-- Represents a 3D shape formed by gluing two cubes --/
structure GluedCubes where
  large_edge_length : ℝ
  small_edge_length : ℝ

/-- Calculates the surface area of the GluedCubes shape --/
def surface_area (shape : GluedCubes) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the specific GluedCubes shape is 136 --/
theorem specific_glued_cubes_surface_area :
  ∃ (shape : GluedCubes),
    shape.large_edge_length = 4 ∧
    shape.small_edge_length = 1 ∧
    surface_area shape = 136 :=
  sorry

end specific_glued_cubes_surface_area_l1426_142611


namespace complex_magnitude_proof_l1426_142673

theorem complex_magnitude_proof (i a : ℂ) : 
  i ^ 2 = -1 →
  a.im = 0 →
  (∃ k : ℝ, (2 - i) / (a + i) = k * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 := by
  sorry

end complex_magnitude_proof_l1426_142673


namespace smallest_divisor_of_repeated_three_digit_number_l1426_142691

theorem smallest_divisor_of_repeated_three_digit_number : ∀ a b c : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
  let abc := 100 * a + 10 * b + c
  let abcabcabc := 1000000 * abc + 1000 * abc + abc
  (101 ∣ abcabcabc) ∧ ∀ d : ℕ, 1 < d → d < 101 → ¬(d ∣ abcabcabc) :=
by sorry

#check smallest_divisor_of_repeated_three_digit_number

end smallest_divisor_of_repeated_three_digit_number_l1426_142691


namespace units_digit_37_power_37_l1426_142681

theorem units_digit_37_power_37 : 37^37 % 10 = 7 := by sorry

end units_digit_37_power_37_l1426_142681


namespace imaginary_part_of_z_l1426_142656

theorem imaginary_part_of_z (z : ℂ) : z = Complex.I * (1 + Complex.I) → Complex.im z = 1 := by
  sorry

end imaginary_part_of_z_l1426_142656


namespace pentagon_condition_l1426_142606

/-- Represents the lengths of five segments cut from a wire -/
structure WireSegments where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  sum_eq_two : a + b + c + d + e = 2
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e

/-- Checks if the given segments can form a pentagon -/
def can_form_pentagon (segments : WireSegments) : Prop :=
  segments.a + segments.b + segments.c + segments.d > segments.e ∧
  segments.a + segments.b + segments.c + segments.e > segments.d ∧
  segments.a + segments.b + segments.d + segments.e > segments.c ∧
  segments.a + segments.c + segments.d + segments.e > segments.b ∧
  segments.b + segments.c + segments.d + segments.e > segments.a

/-- Theorem stating the necessary and sufficient condition for forming a pentagon -/
theorem pentagon_condition (segments : WireSegments) :
  can_form_pentagon segments ↔ segments.a < 1 ∧ segments.b < 1 ∧ segments.c < 1 ∧ segments.d < 1 ∧ segments.e < 1 :=
sorry

end pentagon_condition_l1426_142606


namespace quadratic_real_roots_condition_l1426_142693

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, x^2 - 4*x + (k - 1) = 0) ↔ k ≤ 5 := by
  sorry

end quadratic_real_roots_condition_l1426_142693


namespace taxi_fare_calculation_l1426_142612

/-- Represents the taxi fare structure and proves the cost for a 100-mile ride -/
theorem taxi_fare_calculation (base_fare : ℝ) (rate : ℝ) 
  (h1 : base_fare = 10)
  (h2 : base_fare + 80 * rate = 150) :
  base_fare + 100 * rate = 185 := by
  sorry

#check taxi_fare_calculation

end taxi_fare_calculation_l1426_142612


namespace inscribed_circle_equation_l1426_142696

-- Define the line
def line (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, 3)

-- Define origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the inscribed circle equation
def is_inscribed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Theorem statement
theorem inscribed_circle_equation :
  ∀ x y : ℝ,
  is_inscribed_circle x y ↔
  (∃ r : ℝ, r > 0 ∧
    (x - r)^2 + (y - r)^2 = r^2 ∧
    (x - 4)^2 + y^2 = r^2 ∧
    x^2 + (y - 3)^2 = r^2) :=
by sorry

end inscribed_circle_equation_l1426_142696


namespace max_value_of_expression_max_value_achievable_l1426_142692

theorem max_value_of_expression (x : ℝ) :
  (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) ≤ 1/17 :=
by sorry

theorem max_value_achievable :
  ∃ x : ℝ, (x^4) / (x^8 + 4*x^6 + x^4 + 4*x^2 + 16) = 1/17 :=
by sorry

end max_value_of_expression_max_value_achievable_l1426_142692


namespace book_arrangement_theorem_l1426_142629

def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  let math_group := 1
  let english_groups := 2
  let total_groups := math_group + english_groups
  let math_arrangements := Nat.factorial math_books
  let english_group_size := english_books / english_groups
  let english_group_arrangements := Nat.factorial english_group_size
  Nat.factorial total_groups * math_arrangements * english_group_arrangements * english_group_arrangements

theorem book_arrangement_theorem :
  arrange_books 4 6 = 5184 := by
  sorry

end book_arrangement_theorem_l1426_142629


namespace arithmetic_sequence_sum_l1426_142607

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a (-2))
  (h_sum : (Finset.range 33).sum (fun i => a (3 * i + 1)) = 50) :
  (Finset.range 33).sum (fun i => a (3 * i + 3)) = -82 :=
sorry

end arithmetic_sequence_sum_l1426_142607
