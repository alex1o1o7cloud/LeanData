import Mathlib

namespace NUMINAMATH_CALUDE_two_thousand_sixteenth_smallest_n_l3872_387295

/-- The number of ways Yang can reach (n,0) under the given movement rules -/
def a (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number satisfies the condition an ≡ 1 (mod 5) -/
def satisfies_condition (n : ℕ) : Prop :=
  a n % 5 = 1

/-- The function that returns the kth smallest positive integer satisfying the condition -/
def kth_smallest (k : ℕ) : ℕ := sorry

theorem two_thousand_sixteenth_smallest_n :
  kth_smallest 2016 = 475756 :=
sorry

end NUMINAMATH_CALUDE_two_thousand_sixteenth_smallest_n_l3872_387295


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3872_387225

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∃ q : ℝ, ∀ n, a (n + 1) = a 1 * q^n :=
sorry

theorem main_theorem (a : ℕ → ℝ) (h1 : geometric_sequence a) 
  (h2 : ∀ n, a n > 0)
  (h3 : 2 * (1/2 * a 3) = 3 * a 1 + 2 * a 2) :
  (a 10 + a 12 + a 15 + a 19 + a 20 + a 23) / 
  (a 8 + a 10 + a 13 + a 17 + a 18 + a 21) = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_main_theorem_l3872_387225


namespace NUMINAMATH_CALUDE_fraction_product_l3872_387227

theorem fraction_product : (2 : ℚ) / 9 * 5 / 11 = 10 / 99 := by sorry

end NUMINAMATH_CALUDE_fraction_product_l3872_387227


namespace NUMINAMATH_CALUDE_unique_a_for_three_element_set_l3872_387269

theorem unique_a_for_three_element_set : ∃! (a : ℝ), 
  let A : Set ℝ := {a^2, 2-a, 4}
  (Fintype.card A = 3) ∧ (a = 6) := by sorry

end NUMINAMATH_CALUDE_unique_a_for_three_element_set_l3872_387269


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_l3872_387284

/-- The equation has exactly one solution if and only if a = 1 -/
theorem unique_solution_iff_a_eq_one (a : ℝ) :
  (∃! x : ℝ, 5^(x^2 - 6*a*x + 9*a^2) = a*x^2 - 6*a^2*x + 9*a^3 + a^2 - 6*a + 6) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_one_l3872_387284


namespace NUMINAMATH_CALUDE_restaurant_outdoor_area_l3872_387207

/-- The area of a rectangular section with width 4 feet and length 6 feet is 24 square feet. -/
theorem restaurant_outdoor_area : 
  ∀ (width length area : ℝ), 
    width = 4 → 
    length = 6 → 
    area = width * length → 
    area = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_outdoor_area_l3872_387207


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3872_387210

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3872_387210


namespace NUMINAMATH_CALUDE_interval_covering_theorem_l3872_387232

/-- Definition of the interval I_k -/
def I (a : ℝ → ℝ) (k : ℕ) : Set ℝ := {x | a k ≤ x ∧ x ≤ a k + 1}

/-- The main theorem stating the minimum and maximum values of N -/
theorem interval_covering_theorem (N : ℕ) (a : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 100, ∃ k ∈ Finset.range N, x ∈ I a k) →
  (∀ k ∈ Finset.range N, ∃ x ∈ Set.Icc 0 100, ∀ i ∈ Finset.range N, i ≠ k → x ∉ I a i) →
  100 ≤ N ∧ N ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_interval_covering_theorem_l3872_387232


namespace NUMINAMATH_CALUDE_no_valid_combination_l3872_387281

def nickel : ℕ := 5
def dime : ℕ := 10
def half_dollar : ℕ := 50

def is_valid_combination (coins : List ℕ) : Prop :=
  coins.all (λ c => c = nickel ∨ c = dime ∨ c = half_dollar) ∧
  coins.length = 6 ∧
  coins.sum = 90

theorem no_valid_combination : ¬ ∃ (coins : List ℕ), is_valid_combination coins := by
  sorry

end NUMINAMATH_CALUDE_no_valid_combination_l3872_387281


namespace NUMINAMATH_CALUDE_unique_intersection_l3872_387233

-- Define the line equation
def line (x b : ℝ) : ℝ := 2 * x + b

-- Define the parabola equation
def parabola (x b : ℝ) : ℝ := x^2 + b * x + 1

-- Define the y-intercept of the parabola
def y_intercept (b : ℝ) : ℝ := parabola 0 b

-- Theorem statement
theorem unique_intersection :
  ∃! b : ℝ, line 0 b = y_intercept b := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l3872_387233


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l3872_387238

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ 5 ≤ x}

-- Theorems to prove
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

theorem complement_B_union_P : (U \ B) ∪ P = {x | x ≤ 0 ∨ 3 < x} := by sorry

theorem intersection_AB_complement_P : (A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l3872_387238


namespace NUMINAMATH_CALUDE_basketball_team_enrollment_l3872_387279

theorem basketball_team_enrollment (total : ℕ) (math : ℕ) (both : ℕ) (physics : ℕ) : 
  total = 15 → math = 9 → both = 4 → physics = total - (math - both) → physics = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_enrollment_l3872_387279


namespace NUMINAMATH_CALUDE_y_is_75_percent_of_x_l3872_387234

/-- Given that 45% of z equals 96% of y and z equals 160% of x, prove that y equals 75% of x -/
theorem y_is_75_percent_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : z = 1.60 * x) : 
  y = 0.75 * x := by
sorry

end NUMINAMATH_CALUDE_y_is_75_percent_of_x_l3872_387234


namespace NUMINAMATH_CALUDE_rotate_5_plus_2i_l3872_387291

/-- Rotates a complex number by 90 degrees counter-clockwise around the origin -/
def rotate90 (z : ℂ) : ℂ := z * Complex.I

/-- The result of rotating 5 + 2i by 90 degrees counter-clockwise around the origin -/
theorem rotate_5_plus_2i : rotate90 (5 + 2*Complex.I) = -2 + 5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotate_5_plus_2i_l3872_387291


namespace NUMINAMATH_CALUDE_angle_properties_l3872_387285

-- Define the angle α
variable (α : Real)

-- Define the condition that the terminal side of α lies on y = -√3x
def terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), y = -Real.sqrt 3 * x ∧ x = Real.cos α ∧ y = Real.sin α

-- Define the set S of angles with the same terminal side as α
def S : Set Real :=
  {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}

-- State the theorem
theorem angle_properties (h : terminal_side α) :
  (Real.tan α = -Real.sqrt 3) ∧
  (S = {β | ∃ (k : ℤ), β = k * Real.pi + 2 * Real.pi / 3}) ∧
  ((Real.sqrt 3 * Real.sin (α - Real.pi) + 5 * Real.cos (2 * Real.pi - α)) /
   (-Real.sqrt 3 * Real.cos (3 * Real.pi / 2 + α) + Real.cos (Real.pi + α)) = 4) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l3872_387285


namespace NUMINAMATH_CALUDE_f_max_value_f_min_value_l3872_387223

/-- The function f(x) = 2x³ - 6x² - 18x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The maximum value of f(x) is 17 -/
theorem f_max_value : ∃ (x : ℝ), f x = 17 ∧ ∀ (y : ℝ), f y ≤ 17 := by sorry

/-- The minimum value of f(x) is -47 -/
theorem f_min_value : ∃ (x : ℝ), f x = -47 ∧ ∀ (y : ℝ), f y ≥ -47 := by sorry

end NUMINAMATH_CALUDE_f_max_value_f_min_value_l3872_387223


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3872_387215

theorem rectangle_perimeter (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 192) :
  2 * (b + l) = 64 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3872_387215


namespace NUMINAMATH_CALUDE_brian_tennis_balls_l3872_387289

/-- Given the number of tennis balls for Lily, Frodo, and Brian, prove that Brian has 22 tennis balls. -/
theorem brian_tennis_balls (lily frodo brian : ℕ) 
  (h1 : lily = 3)
  (h2 : frodo = lily + 8)
  (h3 : brian = 2 * frodo) :
  brian = 22 := by
  sorry

end NUMINAMATH_CALUDE_brian_tennis_balls_l3872_387289


namespace NUMINAMATH_CALUDE_ab_equation_sum_l3872_387203

theorem ab_equation_sum (A B : ℕ) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  (10 * A + B) * 6 = 100 * B + 10 * B + B → 
  A + B = 11 :=
by sorry

end NUMINAMATH_CALUDE_ab_equation_sum_l3872_387203


namespace NUMINAMATH_CALUDE_betty_needs_five_more_l3872_387280

-- Define the cost of the wallet
def wallet_cost : ℕ := 100

-- Define Betty's initial savings
def betty_initial_savings : ℕ := wallet_cost / 2

-- Define the amount Betty's parents give her
def parents_contribution : ℕ := 15

-- Define the amount Betty's grandparents give her
def grandparents_contribution : ℕ := 2 * parents_contribution

-- Define Betty's total savings after contributions
def betty_total_savings : ℕ := betty_initial_savings + parents_contribution + grandparents_contribution

-- Theorem: Betty needs $5 more to buy the wallet
theorem betty_needs_five_more : wallet_cost - betty_total_savings = 5 := by
  sorry

end NUMINAMATH_CALUDE_betty_needs_five_more_l3872_387280


namespace NUMINAMATH_CALUDE_det_A_zero_l3872_387299

theorem det_A_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A = A * B - B * A + A^2 * B - 2 * A * B * A + B * A^2 + A^2 * B * A - A * B * A^2) : 
  Matrix.det A = 0 := by
sorry

end NUMINAMATH_CALUDE_det_A_zero_l3872_387299


namespace NUMINAMATH_CALUDE_unique_solution_l3872_387220

/-- Returns the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digit_count (n / 10)

/-- Represents k as overline(1n) -/
def k (n : ℕ) : ℕ := 10^(digit_count n) + n

/-- The main theorem stating that (11, 7) is the only solution -/
theorem unique_solution :
  ∀ m n : ℕ, m^2 = n * k n + 2 → (m = 11 ∧ n = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3872_387220


namespace NUMINAMATH_CALUDE_swordfish_pufferfish_ratio_l3872_387275

/-- The ratio of swordfish to pufferfish in an aquarium -/
theorem swordfish_pufferfish_ratio 
  (total_fish : ℕ) 
  (pufferfish : ℕ) 
  (n : ℕ) 
  (h1 : total_fish = 90)
  (h2 : pufferfish = 15)
  (h3 : total_fish = n * pufferfish + pufferfish) :
  (n * pufferfish) / pufferfish = 5 := by
sorry

end NUMINAMATH_CALUDE_swordfish_pufferfish_ratio_l3872_387275


namespace NUMINAMATH_CALUDE_steven_shirt_count_l3872_387257

def brian_shirts : ℕ := 3
def andrew_shirts : ℕ := 6 * brian_shirts
def steven_shirts : ℕ := 4 * andrew_shirts

theorem steven_shirt_count : steven_shirts = 72 := by
  sorry

end NUMINAMATH_CALUDE_steven_shirt_count_l3872_387257


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l3872_387282

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * 0.95 = 1.102 → x = 16 := by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l3872_387282


namespace NUMINAMATH_CALUDE_inverse_co_complementary_angles_equal_l3872_387293

/-- For any two angles α and β, if their co-complementary angles are equal, then α and β are equal. -/
theorem inverse_co_complementary_angles_equal (α β : Real) :
  (90 - α = 90 - β) → α = β := by
  sorry

end NUMINAMATH_CALUDE_inverse_co_complementary_angles_equal_l3872_387293


namespace NUMINAMATH_CALUDE_sixth_number_tenth_row_l3872_387245

/-- Represents a triangular number array with specific properties -/
structure TriangularArray where
  -- The first number of each row forms an arithmetic sequence
  first_term : ℚ
  common_difference : ℚ
  -- The numbers in each row form a geometric sequence
  common_ratio : ℚ

/-- Get the nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Get the nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The main theorem -/
theorem sixth_number_tenth_row (arr : TriangularArray) 
  (h1 : arr.first_term = 1/4)
  (h2 : arr.common_difference = 1/4)
  (h3 : arr.common_ratio = 1/2) :
  let first_number_tenth_row := arithmeticSequenceTerm arr.first_term arr.common_difference 10
  geometricSequenceTerm first_number_tenth_row arr.common_ratio 6 = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_tenth_row_l3872_387245


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l3872_387206

theorem purely_imaginary_fraction (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a - 1 : ℂ) * I →
  z.re = 0 →
  z.im ≠ 0 →
  (a + I^2024) / (1 - I) = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l3872_387206


namespace NUMINAMATH_CALUDE_nine_to_fourth_equals_three_to_eighth_l3872_387237

theorem nine_to_fourth_equals_three_to_eighth : (9 : ℕ) ^ 4 = 3 ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_nine_to_fourth_equals_three_to_eighth_l3872_387237


namespace NUMINAMATH_CALUDE_weekly_allowance_calculation_l3872_387262

/-- Represents the daily calorie allowance for a person in their 60's. -/
def daily_allowance : ℕ := 2000

/-- Represents the number of days in a week. -/
def days_in_week : ℕ := 7

/-- Calculates the weekly calorie allowance based on the daily allowance. -/
def weekly_allowance : ℕ := daily_allowance * days_in_week

/-- Proves that the weekly calorie allowance for a person in their 60's
    with an average daily allowance of 2000 calories is equal to 10500 calories. -/
theorem weekly_allowance_calculation :
  weekly_allowance = 10500 := by
  sorry

end NUMINAMATH_CALUDE_weekly_allowance_calculation_l3872_387262


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3872_387286

/-- Proves that the repair cost is $300 given the initial purchase price,
    selling price, and gain percentage. -/
theorem repair_cost_calculation (purchase_price selling_price : ℝ) (gain_percentage : ℝ) :
  purchase_price = 900 →
  selling_price = 1500 →
  gain_percentage = 25 →
  (selling_price / (1 + gain_percentage / 100)) - purchase_price = 300 := by
sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l3872_387286


namespace NUMINAMATH_CALUDE_sum_property_implies_isosceles_l3872_387294

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : a + b + c = π

-- Define a quadrilateral
structure Quadrilateral where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ
  angle_sum : w + x + y + z = 2 * π

-- Define the property that for any two angles of the triangle, 
-- there is an angle in the quadrilateral equal to their sum
def has_sum_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∃ (i j : Fin 3) (k : Fin 4), 
    i ≠ j ∧ 
    match i, j with
    | 0, 1 | 1, 0 => q.w = t.a + t.b ∨ q.x = t.a + t.b ∨ q.y = t.a + t.b ∨ q.z = t.a + t.b
    | 0, 2 | 2, 0 => q.w = t.a + t.c ∨ q.x = t.a + t.c ∨ q.y = t.a + t.c ∨ q.z = t.a + t.c
    | 1, 2 | 2, 1 => q.w = t.b + t.c ∨ q.x = t.b + t.c ∨ q.y = t.b + t.c ∨ q.z = t.b + t.c
    | _, _ => False

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- The theorem to be proved
theorem sum_property_implies_isosceles (t : Triangle) (q : Quadrilateral) :
  has_sum_property t q → is_isosceles t :=
by sorry

end NUMINAMATH_CALUDE_sum_property_implies_isosceles_l3872_387294


namespace NUMINAMATH_CALUDE_square_area_with_circles_l3872_387283

theorem square_area_with_circles (r : ℝ) (h : r = 3) : 
  let d := 2 * r
  let s := 2 * d
  s^2 = 144 :=
by sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l3872_387283


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_students_l3872_387249

/-- Represents a school with classes and students -/
structure School where
  num_classes : ℕ
  students_per_class : ℕ
  selected_students : ℕ

/-- The sample size of a school's "Student Congress" -/
def sample_size (school : School) : ℕ :=
  school.selected_students

theorem sample_size_equals_selected_students (school : School) 
  (h1 : school.num_classes = 40)
  (h2 : school.students_per_class = 50)
  (h3 : school.selected_students = 150) :
  sample_size school = 150 := by
  sorry

#check sample_size_equals_selected_students

end NUMINAMATH_CALUDE_sample_size_equals_selected_students_l3872_387249


namespace NUMINAMATH_CALUDE_eighth_group_selection_l3872_387242

/-- Represents the systematic sampling method for a population --/
def systematicSampling (populationSize : Nat) (groupCount : Nat) (t : Nat) : Nat → Nat :=
  fun k => (t + k - 1) % 10 + (k - 1) * 10

/-- Theorem stating the correct number selected from the 8th group --/
theorem eighth_group_selection
  (populationSize : Nat)
  (groupCount : Nat)
  (t : Nat)
  (h1 : populationSize = 100)
  (h2 : groupCount = 10)
  (h3 : t = 7) :
  systematicSampling populationSize groupCount t 8 = 75 := by
  sorry

#check eighth_group_selection

end NUMINAMATH_CALUDE_eighth_group_selection_l3872_387242


namespace NUMINAMATH_CALUDE_unique_solution_l3872_387228

/-- The infinite series representation of the equation -/
def infiniteSeries (x : ℝ) : ℝ := 2 - x + x^2 - x^3 + x^4 - x^5

/-- The condition for series convergence -/
def seriesConverges (x : ℝ) : Prop := abs x < 1

theorem unique_solution : 
  ∃! x : ℝ, (x = infiniteSeries x) ∧ seriesConverges x ∧ x = -1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3872_387228


namespace NUMINAMATH_CALUDE_circle_C_theorem_l3872_387235

-- Define the circle C
def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 5}

-- Define the lines
def line_l1 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l2 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 - Real.sqrt 3 = 0
def line_l3 (m a : ℝ) (x y : ℝ) : Prop := m * x - y + Real.sqrt a + 1 = 0

-- Define the theorem
theorem circle_C_theorem (center : ℝ × ℝ) (M N : ℝ × ℝ) :
  line_l1 center.1 center.2 →
  M ∈ circle_C center →
  N ∈ circle_C center →
  line_l2 M.1 M.2 →
  line_l2 N.1 N.2 →
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 17 →
  (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m 5 p.1 p.2) →
  (((center.1 = 0 ∧ center.2 = 1) ∨
    (center.1 = 3 + Real.sqrt 3 ∧ center.2 = 4 + Real.sqrt 3)) ∧
   (∀ (a : ℝ), (∀ (m : ℝ), ∃ (p : ℝ × ℝ), p ∈ circle_C center ∧ line_l3 m a p.1 p.2) → 0 ≤ a ∧ a ≤ 5)) :=
by sorry


end NUMINAMATH_CALUDE_circle_C_theorem_l3872_387235


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3872_387292

theorem min_value_of_expression (a b : ℤ) (h : a > b) :
  (((a^2 + b^2) / (a^2 - b^2)) + ((a^2 - b^2) / (a^2 + b^2)) : ℚ) ≥ 2 ∧
  ∃ (a' b' : ℤ), a' > b' ∧ (((a'^2 + b'^2) / (a'^2 - b'^2)) + ((a'^2 - b'^2) / (a'^2 + b'^2)) : ℚ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3872_387292


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l3872_387253

def euler_family_ages : List ℕ := [5, 8, 8, 8, 12, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 53 / 6 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l3872_387253


namespace NUMINAMATH_CALUDE_product_maximum_l3872_387296

theorem product_maximum (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x * y ≥ a * b ∧
  x * y = s^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_product_maximum_l3872_387296


namespace NUMINAMATH_CALUDE_angle_AO2B_greater_than_90_degrees_l3872_387221

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the center of circle O₂
def O2_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem angle_AO2B_greater_than_90_degrees :
  let angle_AO2B := sorry
  angle_AO2B > 90 := by sorry

end NUMINAMATH_CALUDE_angle_AO2B_greater_than_90_degrees_l3872_387221


namespace NUMINAMATH_CALUDE_b_completes_in_24_days_l3872_387212

/-- Worker represents a person who can complete a task -/
structure Worker where
  rate : ℚ  -- work rate in units of work per day

/-- Represents a work scenario with three workers -/
structure WorkScenario where
  a : Worker
  b : Worker
  c : Worker
  combined_time_ab : ℚ  -- time for a and b to complete work together
  time_a : ℚ           -- time for a to complete work alone
  time_c : ℚ           -- time for c to complete work alone

/-- Calculate the time for worker b to complete the work alone -/
def time_for_b_alone (w : WorkScenario) : ℚ :=
  1 / (1 / w.combined_time_ab - 1 / w.time_a)

/-- Theorem stating that given the conditions, b takes 24 days to complete the work alone -/
theorem b_completes_in_24_days (w : WorkScenario) 
  (h1 : w.combined_time_ab = 8)
  (h2 : w.time_a = 12)
  (h3 : w.time_c = 18) :
  time_for_b_alone w = 24 := by
  sorry

#eval time_for_b_alone { a := ⟨1/12⟩, b := ⟨1/24⟩, c := ⟨1/18⟩, combined_time_ab := 8, time_a := 12, time_c := 18 }

end NUMINAMATH_CALUDE_b_completes_in_24_days_l3872_387212


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3872_387263

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →  -- q is positive
  (∀ n, a (n + 1) = a n * q) →  -- {a_n} is a geometric sequence with common ratio q
  (a 3 * a 9 = 2 * (a 5)^2) →  -- given condition
  q = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3872_387263


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3872_387247

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the property that f must satisfy
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((floor x : ℝ) * y) = f x * (floor (f y) : ℝ)

-- Theorem statement
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_equation f →
    (∀ x : ℝ, f x = 0) ∨ 
    (∃ C : ℝ, 1 ≤ C ∧ C < 2 ∧ ∀ x : ℝ, f x = C) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3872_387247


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l3872_387246

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = ![![-2, 3], ![1, -5]]) : 
  (A^2)⁻¹ = ![![7, -21], ![-7, 28]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l3872_387246


namespace NUMINAMATH_CALUDE_stating_bacteria_fill_time_l3872_387259

/-- 
Represents the time (in minutes) it takes to fill a bottle with bacteria,
given the initial number of bacteria and their division rate.
-/
def fill_time (initial_bacteria : ℕ) (a : ℕ) : ℕ :=
  if initial_bacteria = 1 then a
  else a - 1

/-- 
Theorem stating that if one bacterium fills a bottle in 'a' minutes,
then two bacteria will fill the same bottle in 'a - 1' minutes,
given that each bacterium divides into two every minute.
-/
theorem bacteria_fill_time (a : ℕ) (h : a > 0) :
  fill_time 2 a = a - 1 :=
sorry

end NUMINAMATH_CALUDE_stating_bacteria_fill_time_l3872_387259


namespace NUMINAMATH_CALUDE_nominal_rate_for_given_ear_l3872_387239

/-- Given an effective annual rate and compounding frequency, 
    calculate the nominal rate of interest per annum. -/
def nominal_rate (ear : ℝ) (n : ℕ) : ℝ :=
  n * ((1 + ear) ^ (1 / n) - 1)

/-- Theorem stating that for an effective annual rate of 12.36% 
    with half-yearly compounding, the nominal rate is approximately 11.66% -/
theorem nominal_rate_for_given_ear :
  let ear := 0.1236
  let n := 2
  abs (nominal_rate ear n - 0.1166) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_nominal_rate_for_given_ear_l3872_387239


namespace NUMINAMATH_CALUDE_division_problem_l3872_387255

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1575) (h3 : a = b * q + 15) : q = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3872_387255


namespace NUMINAMATH_CALUDE_factors_of_M_l3872_387222

/-- The number of natural-number factors of M, where M = 2^3 · 3^5 · 5^3 · 7^1 · 11^2 -/
def number_of_factors (M : ℕ) : ℕ :=
  if M = 2^3 * 3^5 * 5^3 * 7^1 * 11^2 then 576 else 0

/-- Theorem stating that the number of natural-number factors of M is 576 -/
theorem factors_of_M :
  number_of_factors (2^3 * 3^5 * 5^3 * 7^1 * 11^2) = 576 :=
by sorry

end NUMINAMATH_CALUDE_factors_of_M_l3872_387222


namespace NUMINAMATH_CALUDE_men_work_hours_l3872_387297

theorem men_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (women_hours : ℕ) (H : ℚ) :
  men = 15 →
  women = 21 →
  men_days = 21 →
  women_days = 60 →
  women_hours = 3 →
  (3 : ℚ) * men * men_days * H = 2 * women * women_days * women_hours →
  H = 8 := by
sorry

end NUMINAMATH_CALUDE_men_work_hours_l3872_387297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3872_387229

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₂ = 5 and a₅ = 33,
    prove that a₃ + a₄ = 38. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 33) :
  a 3 + a 4 = 38 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3872_387229


namespace NUMINAMATH_CALUDE_william_max_riding_time_l3872_387287

/-- Represents the maximum number of hours William can ride his horse per day -/
def max_riding_time : ℝ := 6

/-- The total number of days William rode -/
def total_days : ℕ := 6

/-- The number of days William rode for the maximum time -/
def max_time_days : ℕ := 2

/-- The number of days William rode for 1.5 hours -/
def short_ride_days : ℕ := 2

/-- The number of days William rode for half the maximum time -/
def half_time_days : ℕ := 2

/-- The duration of a short ride in hours -/
def short_ride_duration : ℝ := 1.5

/-- The total riding time over all days in hours -/
def total_riding_time : ℝ := 21

theorem william_max_riding_time :
  max_riding_time * max_time_days +
  short_ride_duration * short_ride_days +
  (max_riding_time / 2) * half_time_days = total_riding_time ∧
  max_time_days + short_ride_days + half_time_days = total_days :=
by sorry

end NUMINAMATH_CALUDE_william_max_riding_time_l3872_387287


namespace NUMINAMATH_CALUDE_modular_equation_solution_l3872_387216

theorem modular_equation_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 151 ∧ (150 * n + 3) % 151 = 45 % 151 ∧ n = 109 := by
  sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l3872_387216


namespace NUMINAMATH_CALUDE_f_at_4_l3872_387240

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem f_at_4 : f 4 = 371 := by
  sorry

end NUMINAMATH_CALUDE_f_at_4_l3872_387240


namespace NUMINAMATH_CALUDE_regression_relationships_l3872_387243

/-- Represents the possibility that x is not related to y -/
def notRelatedPossibility : ℝ → ℝ := sorry

/-- Represents the fitting effect of the regression line -/
def fittingEffect : ℝ → ℝ := sorry

/-- Represents the degree of fit -/
def degreeOfFit : ℝ → ℝ := sorry

theorem regression_relationships :
  (∀ k₁ k₂ : ℝ, k₁ < k₂ → notRelatedPossibility k₁ > notRelatedPossibility k₂) ∧
  (∀ s₁ s₂ : ℝ, s₁ < s₂ → fittingEffect s₁ > fittingEffect s₂) ∧
  (∀ r₁ r₂ : ℝ, r₁ < r₂ → degreeOfFit r₁ < degreeOfFit r₂) :=
by sorry

end NUMINAMATH_CALUDE_regression_relationships_l3872_387243


namespace NUMINAMATH_CALUDE_vector_operation_l3872_387217

theorem vector_operation (a b c : ℝ × ℝ × ℝ) :
  a = (2, 0, 1) →
  b = (-3, 1, -1) →
  c = (1, 1, 0) →
  a + 2 • b - 3 • c = (-7, -1, -1) := by
sorry

end NUMINAMATH_CALUDE_vector_operation_l3872_387217


namespace NUMINAMATH_CALUDE_last_four_digits_theorem_l3872_387226

theorem last_four_digits_theorem :
  ∃ (N : ℕ+),
    (∃ (a b c d : ℕ),
      a ≠ 0 ∧
      a ≠ 6 ∧ b ≠ 6 ∧ c ≠ 6 ∧
      N % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      (N * N) % 10000 = a * 1000 + b * 100 + c * 10 + d ∧
      a * 100 + b * 10 + c = 106) :=
by sorry

end NUMINAMATH_CALUDE_last_four_digits_theorem_l3872_387226


namespace NUMINAMATH_CALUDE_triangle_construction_from_polygon_centers_l3872_387252

/-- Centers of regular n-sided polygons externally inscribed on triangle sides -/
structure PolygonCenters (n : ℕ) where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Triangle vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation by angle α around point P -/
def rotate (P : ℝ × ℝ) (α : ℝ) (Q : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if three points form a regular triangle -/
def isRegularTriangle (P Q R : ℝ × ℝ) : Prop := sorry

/-- Theorem about triangle construction from polygon centers -/
theorem triangle_construction_from_polygon_centers (n : ℕ) (centers : PolygonCenters n) :
  (n ≥ 4 → ∃! t : Triangle, 
    rotate centers.Y (2 * π / n) t.A = t.C ∧
    rotate centers.X (2 * π / n) t.C = t.B ∧
    rotate centers.Z (2 * π / n) t.B = t.A) ∧
  (n = 3 → isRegularTriangle centers.X centers.Y centers.Z → 
    ∃ t : Set Triangle, Infinite t ∧ 
    ∀ tri ∈ t, rotate centers.Y (2 * π / 3) tri.A = tri.C ∧
               rotate centers.X (2 * π / 3) tri.C = tri.B ∧
               rotate centers.Z (2 * π / 3) tri.B = tri.A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_construction_from_polygon_centers_l3872_387252


namespace NUMINAMATH_CALUDE_unique_number_l3872_387213

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def contains_digit_5 (n : ℕ) : Prop := ∃ a b, n = 10*a + 5 + b ∧ 0 ≤ b ∧ b < 10

def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3*k

theorem unique_number : 
  ∃! n : ℕ, 
    144 < n ∧ 
    n < 169 ∧ 
    is_odd n ∧ 
    contains_digit_5 n ∧ 
    divisible_by_3 n ∧ 
    n = 165 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_l3872_387213


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3872_387264

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x - 1)

theorem f_derivative_at_one : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3872_387264


namespace NUMINAMATH_CALUDE_square_side_length_l3872_387219

/-- Right triangle PQR with legs PQ and PR, and a square inside --/
structure RightTriangleWithSquare where
  /-- Length of leg PQ --/
  pq : ℝ
  /-- Length of leg PR --/
  pr : ℝ
  /-- Side length of the square --/
  s : ℝ
  /-- PQ is 9 cm --/
  pq_length : pq = 9
  /-- PR is 12 cm --/
  pr_length : pr = 12
  /-- The square has one side on hypotenuse QR and one vertex on each leg --/
  square_position : s > 0 ∧ s < pq ∧ s < pr

/-- The side length of the square is 15/2 cm --/
theorem square_side_length (t : RightTriangleWithSquare) : t.s = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3872_387219


namespace NUMINAMATH_CALUDE_fourth_power_complex_equality_l3872_387272

theorem fourth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_complex_equality_l3872_387272


namespace NUMINAMATH_CALUDE_board_game_impossibility_l3872_387241

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The operation of replacing two numbers with their difference -/
def replace_with_diff (s : ℤ) (a b : ℤ) : ℤ := s - 2 * min a b

/-- Theorem: It's impossible to reduce the sum of numbers from 1 to 1989 to zero
    by repeatedly replacing any two numbers with their difference -/
theorem board_game_impossibility :
  ∀ (ops : ℕ),
  ∃ (result : ℤ),
  result ≠ 0 ∧
  (∃ (numbers : List ℤ),
    numbers.sum = result ∧
    numbers.length + ops = 1989 ∧
    (∀ (x : ℤ), x ∈ numbers → x ≥ 0)) :=
by sorry


end NUMINAMATH_CALUDE_board_game_impossibility_l3872_387241


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3872_387202

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l3872_387202


namespace NUMINAMATH_CALUDE_planting_cost_l3872_387208

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end NUMINAMATH_CALUDE_planting_cost_l3872_387208


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l3872_387298

theorem opposite_of_negative_2023 : -(-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l3872_387298


namespace NUMINAMATH_CALUDE_saras_quarters_l3872_387211

/-- Sara's quarters problem -/
theorem saras_quarters (initial_quarters borrowed_quarters : ℕ) 
  (h1 : initial_quarters = 4937)
  (h2 : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 :=
by sorry

end NUMINAMATH_CALUDE_saras_quarters_l3872_387211


namespace NUMINAMATH_CALUDE_license_plate_palindrome_theorem_l3872_387250

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate -/
def letter_length : ℕ := 4

/-- The length of the digit sequence in the license plate -/
def digit_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome -/
def license_plate_palindrome_probability : ℚ := 655 / 57122

/-- 
Theorem: The probability of a license plate containing at least one palindrome 
(either in the four-letter or four-digit arrangement) is 655/57122.
-/
theorem license_plate_palindrome_theorem : 
  license_plate_palindrome_probability = 655 / 57122 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_theorem_l3872_387250


namespace NUMINAMATH_CALUDE_function_periodicity_l3872_387214

/-- A function f: ℝ → ℝ satisfying the given property is periodic with period 2a -/
theorem function_periodicity (f : ℝ → ℝ) (a : ℝ) (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l3872_387214


namespace NUMINAMATH_CALUDE_workers_read_both_books_l3872_387248

/-- The number of workers who have read both Saramago's and Kureishi's latest books -/
def workers_read_both (total : ℕ) (saramago : ℕ) (kureishi : ℕ) (neither : ℕ) : ℕ :=
  saramago + kureishi - (total - neither)

theorem workers_read_both_books :
  let total := 42
  let saramago := total / 2
  let kureishi := total / 6
  let neither := saramago - kureishi - 1
  workers_read_both total saramago kureishi neither = 6 := by
  sorry

#eval workers_read_both 42 21 7 20

end NUMINAMATH_CALUDE_workers_read_both_books_l3872_387248


namespace NUMINAMATH_CALUDE_factored_equation_difference_l3872_387278

theorem factored_equation_difference (p q : ℝ) : 
  (∃ (x : ℝ), x^2 - 6*x + q = 0 ∧ (x - p)^2 = 7) → p - q = 1 := by
  sorry

end NUMINAMATH_CALUDE_factored_equation_difference_l3872_387278


namespace NUMINAMATH_CALUDE_patricks_age_l3872_387267

/-- Given that Patrick is half the age of his elder brother Robert, and Robert will turn 30 after 2 years, prove that Patrick's current age is 14 years. -/
theorem patricks_age (robert_age_in_two_years : ℕ) (robert_current_age : ℕ) (patrick_age : ℕ) : 
  robert_age_in_two_years = 30 → 
  robert_current_age = robert_age_in_two_years - 2 →
  patrick_age = robert_current_age / 2 →
  patrick_age = 14 := by
sorry

end NUMINAMATH_CALUDE_patricks_age_l3872_387267


namespace NUMINAMATH_CALUDE_problem_solution_l3872_387204

theorem problem_solution (x : ℚ) : x - 2/5 = 7/15 - 1/3 - 1/6 → x = 11/30 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3872_387204


namespace NUMINAMATH_CALUDE_eunji_confetti_l3872_387231

theorem eunji_confetti (red : ℕ) (green : ℕ) (given : ℕ) : 
  red = 1 → green = 9 → given = 4 → red + green - given = 6 := by sorry

end NUMINAMATH_CALUDE_eunji_confetti_l3872_387231


namespace NUMINAMATH_CALUDE_min_value_and_relationship_l3872_387200

theorem min_value_and_relationship (a b : ℝ) : 
  (4 + (a + b)^2 ≥ 4) ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_relationship_l3872_387200


namespace NUMINAMATH_CALUDE_monomials_like_terms_iff_l3872_387251

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ v, m1 v = m2 v

/-- The first monomial 4ab^n -/
def monomial1 (n : ℕ) : ℕ → ℕ
| 0 => 1  -- exponent of a
| 1 => n  -- exponent of b
| _ => 0  -- other variables

/-- The second monomial -2a^mb^4 -/
def monomial2 (m : ℕ) : ℕ → ℕ
| 0 => m  -- exponent of a
| 1 => 4  -- exponent of b
| _ => 0  -- other variables

/-- Theorem: The monomials 4ab^n and -2a^mb^4 are like terms if and only if m = 1 and n = 4 -/
theorem monomials_like_terms_iff (m n : ℕ) :
  like_terms (monomial1 n) (monomial2 m) ↔ m = 1 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomials_like_terms_iff_l3872_387251


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3872_387274

/-- The line equation passing through a fixed point -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 2) * x - y + 3 * m + 2 = 0

/-- Theorem stating that the line always passes through the point (-3, 8) -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-3) 8 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3872_387274


namespace NUMINAMATH_CALUDE_time_to_fill_tank_with_hole_l3872_387270

/-- Time to fill tank with hole present -/
theorem time_to_fill_tank_with_hole 
  (pipe_fill_time : ℝ) 
  (hole_empty_time : ℝ) 
  (h1 : pipe_fill_time = 15) 
  (h2 : hole_empty_time = 60.000000000000014) : 
  (1 : ℝ) / ((1 / pipe_fill_time) - (1 / hole_empty_time)) = 20.000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_time_to_fill_tank_with_hole_l3872_387270


namespace NUMINAMATH_CALUDE_average_words_per_puzzle_l3872_387254

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Represents the number of weeks a pencil lasts -/
def weeks_per_pencil : ℕ := 2

/-- Represents the total number of words to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- Represents Bert's daily crossword puzzle habit -/
def puzzles_per_day : ℕ := 1

/-- Theorem stating the average number of words in each crossword puzzle -/
theorem average_words_per_puzzle :
  (words_per_pencil / (weeks_per_pencil * days_per_week)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_words_per_puzzle_l3872_387254


namespace NUMINAMATH_CALUDE_area_of_fifth_rectangle_l3872_387261

/-- Given a rectangle divided into five smaller rectangles, prove the area of the fifth rectangle --/
theorem area_of_fifth_rectangle
  (x y n k m : ℝ)
  (a b c d : ℝ)
  (h1 : a = k * (y - n))
  (h2 : b = (m - k) * (y - n))
  (h3 : c = m * (y - n))
  (h4 : d = (x - m) * n)
  (h5 : 0 < x ∧ 0 < y ∧ 0 < n ∧ 0 < k ∧ 0 < m)
  (h6 : n < y ∧ k < m ∧ m < x) :
  x * y - a - b - c - d = x * y - x * n :=
sorry

end NUMINAMATH_CALUDE_area_of_fifth_rectangle_l3872_387261


namespace NUMINAMATH_CALUDE_triplets_equal_sum_l3872_387266

/-- The number of ordered triplets (m, n, p) of nonnegative integers satisfying m + 3n + 5p ≤ 600 -/
def countTriplets : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 ≤ 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).card

/-- The sum of (i+1) for all nonnegative integer solutions of i + 3j + 5k = 600 -/
def sumSolutions : ℕ :=
  (Finset.filter (fun t : ℕ × ℕ × ℕ => t.1 + 3 * t.2.1 + 5 * t.2.2 = 600) (Finset.product (Finset.range 601) (Finset.product (Finset.range 201) (Finset.range 121)))).sum (fun t => t.1 + 1)

theorem triplets_equal_sum : countTriplets = sumSolutions := by
  sorry

end NUMINAMATH_CALUDE_triplets_equal_sum_l3872_387266


namespace NUMINAMATH_CALUDE_mary_fruits_left_l3872_387201

/-- The number of fruits Mary has left after buying, using for salad, eating, and giving away -/
def fruits_left (apples oranges blueberries grapes kiwis : ℕ) 
  (apples_salad oranges_salad blueberries_salad : ℕ)
  (apples_eaten oranges_eaten kiwis_eaten : ℕ)
  (apples_given oranges_given blueberries_given grapes_given kiwis_given : ℕ) : ℕ :=
  (apples - apples_salad - apples_eaten - apples_given) +
  (oranges - oranges_salad - oranges_eaten - oranges_given) +
  (blueberries - blueberries_salad - blueberries_given) +
  (grapes - grapes_given) +
  (kiwis - kiwis_eaten - kiwis_given)

/-- Theorem stating that Mary has 61 fruits left -/
theorem mary_fruits_left : 
  fruits_left 26 35 18 12 22 6 10 8 2 3 1 5 7 4 3 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruits_left_l3872_387201


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3872_387290

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : geometric_sequence a)
  (h_prod : a 1 * a 3 = 16)
  (h_sum : a 3 + a 4 = 24) :
  a 5 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3872_387290


namespace NUMINAMATH_CALUDE_quadratic_rational_roots_l3872_387273

theorem quadratic_rational_roots (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a = 1 ∧ b = 2 ∧ c = -3 →
  ∃ (x y : ℚ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rational_roots_l3872_387273


namespace NUMINAMATH_CALUDE_min_triangle_area_l3872_387288

-- Define the triangle and square
structure Triangle :=
  (X Y Z : ℝ × ℝ)

structure Square :=
  (side : ℝ)
  (area : ℝ)

-- Define the properties
def is_acute_angled (t : Triangle) : Prop := sorry

def square_inscribed (t : Triangle) (s : Square) : Prop := sorry

-- Theorem statement
theorem min_triangle_area 
  (t : Triangle) 
  (s : Square) 
  (h_acute : is_acute_angled t) 
  (h_inscribed : square_inscribed t s) 
  (h_area : s.area = 2017) : 
  ∃ (min_area : ℝ), min_area = 2017/2 ∧ 
  ∀ (actual_area : ℝ), actual_area ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_l3872_387288


namespace NUMINAMATH_CALUDE_order_of_3_is_2_l3872_387218

def f (x : ℕ) : ℕ := x^2 % 13

def iterate_f (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem order_of_3_is_2 : 
  (∃ m : ℕ, m > 0 ∧ iterate_f m 3 = 3) ∧ 
  (∀ k : ℕ, k > 0 ∧ k < 2 → iterate_f k 3 ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_order_of_3_is_2_l3872_387218


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l3872_387271

/-- Represents the volume multiplication factor of a cylinder when its height is tripled and radius is increased by 300% -/
def cylinder_volume_factor : ℝ := 48

/-- Theorem stating that when a cylinder's height is tripled and its radius is increased by 300%, its volume is multiplied by a factor of 48 -/
theorem cylinder_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let new_r := 4 * r
  let new_h := 3 * h
  (π * new_r^2 * new_h) / (π * r^2 * h) = cylinder_volume_factor :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l3872_387271


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3872_387268

/-- Proves that in a class with 35 students, where there are seven more girls than boys, 
    the ratio of girls to boys is 3:2 -/
theorem girls_to_boys_ratio (total : ℕ) (girls boys : ℕ) : 
  total = 35 →
  girls = boys + 7 →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3872_387268


namespace NUMINAMATH_CALUDE_insertion_methods_eq_336_l3872_387230

/- Given 5 books originally and 3 books to insert -/
def original_books : ℕ := 5
def books_to_insert : ℕ := 3

/- The number of gaps increases after each insertion -/
def gaps (n : ℕ) : ℕ := n + 1

/- The total number of insertion methods -/
def insertion_methods : ℕ :=
  (gaps original_books) * (gaps (original_books + 1)) * (gaps (original_books + 2))

/- Theorem stating that the number of insertion methods is 336 -/
theorem insertion_methods_eq_336 : insertion_methods = 336 := by
  sorry

end NUMINAMATH_CALUDE_insertion_methods_eq_336_l3872_387230


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l3872_387236

theorem right_triangle_leg_length (c a b : ℝ) : 
  c = 10 →  -- hypotenuse length
  a = 6 →   -- length of one leg
  c^2 = a^2 + b^2 →  -- Pythagorean theorem (right-angled triangle condition)
  b = 8 := by  -- length of the other leg
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l3872_387236


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3872_387276

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (1 - Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3872_387276


namespace NUMINAMATH_CALUDE_principal_amount_correct_l3872_387258

/-- The principal amount borrowed -/
def P : ℝ := 22539.53

/-- The total interest paid after 3 years -/
def total_interest : ℝ := 9692

/-- The interest rate for the first year -/
def r1 : ℝ := 0.12

/-- The interest rate for the second year -/
def r2 : ℝ := 0.14

/-- The interest rate for the third year -/
def r3 : ℝ := 0.17

/-- Theorem stating that the given principal amount results in the specified total interest -/
theorem principal_amount_correct : 
  P * r1 + P * r2 + P * r3 = total_interest := by sorry

end NUMINAMATH_CALUDE_principal_amount_correct_l3872_387258


namespace NUMINAMATH_CALUDE_exists_points_with_longer_inner_vector_sum_l3872_387205

/-- A regular polygon with 1976 sides -/
structure RegularPolygon1976 where
  vertices : Fin 1976 → ℝ × ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is inside the regular 1976-gon -/
def isInside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Checks if a point is outside the regular 1976-gon -/
def isOutside (p : Point) (poly : RegularPolygon1976) : Prop :=
  sorry

/-- Sum of vectors from a point to all vertices of the 1976-gon -/
def vectorSum (p : Point) (poly : RegularPolygon1976) : ℝ × ℝ :=
  sorry

/-- Length of a 2D vector -/
def vectorLength (v : ℝ × ℝ) : ℝ :=
  sorry

/-- Theorem stating the existence of points A and B satisfying the conditions -/
theorem exists_points_with_longer_inner_vector_sum (poly : RegularPolygon1976) :
  ∃ (A B : Point),
    isInside A poly ∧
    isOutside B poly ∧
    vectorLength (vectorSum A poly) > vectorLength (vectorSum B poly) :=
  sorry

end NUMINAMATH_CALUDE_exists_points_with_longer_inner_vector_sum_l3872_387205


namespace NUMINAMATH_CALUDE_range_of_3a_minus_2b_l3872_387277

theorem range_of_3a_minus_2b (a b : ℝ) 
  (h1 : -3 ≤ a + b ∧ a + b ≤ 2) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 4) : 
  -4 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 11 := by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_2b_l3872_387277


namespace NUMINAMATH_CALUDE_commodity_tax_consumption_l3872_387260

theorem commodity_tax_consumption (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.8 * T
  let new_revenue := 0.92 * T * C
  ∃ new_consumption, 
    new_tax * new_consumption = new_revenue ∧ 
    new_consumption = 1.15 * C := by
sorry

end NUMINAMATH_CALUDE_commodity_tax_consumption_l3872_387260


namespace NUMINAMATH_CALUDE_passes_through_point_l3872_387209

/-- A linear function that passes through the point (0, 3) -/
def linearFunction (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- Theorem: The linear function passes through the point (0, 3) for any slope m -/
theorem passes_through_point (m : ℝ) : linearFunction m 0 = 3 := by
  sorry

end NUMINAMATH_CALUDE_passes_through_point_l3872_387209


namespace NUMINAMATH_CALUDE_calculate_wins_l3872_387265

/-- Given a team's home game statistics, calculate the number of wins -/
theorem calculate_wins (total_games losses : ℕ) (h1 : total_games = 56) (h2 : losses = 12) : 
  total_games - losses - (losses / 2) = 38 := by
  sorry

#check calculate_wins

end NUMINAMATH_CALUDE_calculate_wins_l3872_387265


namespace NUMINAMATH_CALUDE_cube_midpoint_planes_l3872_387256

-- Define a cube type
structure Cube where
  -- Add necessary properties of a cube

-- Define a plane type
structure Plane where
  -- Add necessary properties of a plane

-- Define a function to check if a plane contains a midpoint of a cube's edge
def containsMidpoint (p : Plane) (c : Cube) : Prop :=
  sorry

-- Define a function to count the number of midpoints a plane contains
def countMidpoints (p : Plane) (c : Cube) : ℕ :=
  sorry

-- Define a function to check if a plane contains at least 3 midpoints
def containsAtLeastThreeMidpoints (p : Plane) (c : Cube) : Prop :=
  countMidpoints p c ≥ 3

-- Define a function to count the number of planes containing at least 3 midpoints
def countPlanesWithAtLeastThreeMidpoints (c : Cube) : ℕ :=
  sorry

-- Theorem statement
theorem cube_midpoint_planes (c : Cube) :
  countPlanesWithAtLeastThreeMidpoints c = 81 :=
sorry

end NUMINAMATH_CALUDE_cube_midpoint_planes_l3872_387256


namespace NUMINAMATH_CALUDE_recreation_spending_percentage_l3872_387224

/-- Calculates the percentage of this week's recreation spending compared to last week's. -/
theorem recreation_spending_percentage 
  (last_week_wage : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.40) 
  (h2 : this_week_wage_reduction = 0.05) 
  (h3 : this_week_recreation_percent = 0.50) : 
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wage) / 
  (last_week_recreation_percent * last_week_wage) * 100 = 118.75 :=
by sorry

end NUMINAMATH_CALUDE_recreation_spending_percentage_l3872_387224


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3872_387244

/-- A hyperbola is defined by its equation in the form ax^2 + by^2 = c,
    where a, b, and c are real numbers and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_opposite_signs : a * b < 0

/-- The point (x, y) in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  h.a * p.x^2 + h.b * p.y^2 = h.c

/-- Two hyperbolas have the same asymptotes if their equations are proportional -/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ h1.a = k * h2.a ∧ h1.b = k * h2.b

theorem hyperbola_equation (h1 : Hyperbola) (h2 : Hyperbola) (p : Point) :
  same_asymptotes h1 { a := 1, b := -1/4, c := 1, h_opposite_signs := sorry } →
  point_on_hyperbola h2 { x := 2, y := 0 } →
  h2 = { a := 1/4, b := -1/16, c := 1, h_opposite_signs := sorry } :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3872_387244
