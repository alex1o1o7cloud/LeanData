import Mathlib

namespace NUMINAMATH_CALUDE_ten_times_average_sum_positions_elida_length_adrianna_length_l348_34810

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : ℕ :=
  (c.toNat - 'A'.toNat + 1)

/-- The name Elida -/
def elida : String := "ELIDA"

/-- The name Adrianna -/
def adrianna : String := "ADRIANNA"

/-- Sum of alphabetical positions of letters in a name -/
def sumAlphabeticalPositions (name : String) : ℕ :=
  name.toList.map alphabeticalPosition |>.sum

/-- Theorem stating that 10 times the average of the sum of alphabetical positions
    in both names is 465 -/
theorem ten_times_average_sum_positions : 
  (10 : ℚ) * ((sumAlphabeticalPositions elida + sumAlphabeticalPositions adrianna) / 2) = 465 := by
  sorry

/-- Elida has 5 letters -/
theorem elida_length : elida.length = 5 := by sorry

/-- Adrianna has 2 less than twice the number of letters Elida has -/
theorem adrianna_length : adrianna.length = 2 * elida.length - 2 := by sorry

end NUMINAMATH_CALUDE_ten_times_average_sum_positions_elida_length_adrianna_length_l348_34810


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l348_34827

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The theorem stating the sum of the 6th, 7th, and 8th terms -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l348_34827


namespace NUMINAMATH_CALUDE_acute_angle_solution_l348_34817

theorem acute_angle_solution (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  Real.cos α * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) = 1 →
  α = 40 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_solution_l348_34817


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l348_34873

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m * x₂ + 1 = 0) →
  m > 2 ∧
  ∃ m₀ : ℝ, m₀ > 2 ∧ ¬(∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - Real.sqrt m₀ * x₁ + 1 = 0 ∧ x₂^2 - Real.sqrt m₀ * x₂ + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l348_34873


namespace NUMINAMATH_CALUDE_trinomial_perfect_fourth_power_l348_34879

/-- A trinomial is a perfect fourth power for all integers if and only if its quadratic and linear coefficients are zero. -/
theorem trinomial_perfect_fourth_power (a b c : ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^4) → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_perfect_fourth_power_l348_34879


namespace NUMINAMATH_CALUDE_sum_is_composite_l348_34815

theorem sum_is_composite (m n : ℕ) (h : 88 * m = 81 * n) : 
  ∃ (k : ℕ), k > 1 ∧ k < m + n ∧ k ∣ (m + n) :=
by sorry

end NUMINAMATH_CALUDE_sum_is_composite_l348_34815


namespace NUMINAMATH_CALUDE_inequality_relationship_l348_34822

theorem inequality_relationship (a b : ℝ) (h : a < 1 / b) :
  (a > 0 ∧ b > 0 → 1 / a > b) ∧
  (a < 0 ∧ b < 0 → 1 / a > b) ∧
  (a < 0 ∧ b > 0 → 1 / a < b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l348_34822


namespace NUMINAMATH_CALUDE_triangle_area_l348_34828

/-- Given a triangle ABC with the following properties:
  1. The side opposite to angle B has length 1
  2. Angle B measures π/6 radians
  3. 1/tan(A) + 1/tan(C) = 2
Prove that the area of the triangle is 1/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 1 → 
  B = π / 6 → 
  1 / Real.tan A + 1 / Real.tan C = 2 → 
  (1 / 2) * a * b * Real.sin C = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l348_34828


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l348_34850

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_linear_function (a : ℝ) : 
  let f := λ x : ℝ => a * x - 3 + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l348_34850


namespace NUMINAMATH_CALUDE_prism_volume_l348_34826

/-- Given a right rectangular prism with dimensions a, b, and c,
    if the areas of three faces are 30, 45, and 54 square centimeters,
    then the volume of the prism is 270 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : a * c = 45) 
  (h3 : b * c = 54) : 
  a * b * c = 270 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l348_34826


namespace NUMINAMATH_CALUDE_factor_implies_c_value_l348_34848

/-- The polynomial P(x) -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + c*x + 15

/-- Theorem: If x - 3 is a factor of P(x), then c = -23 -/
theorem factor_implies_c_value (c : ℝ) : 
  (∀ x, P c x = 0 ↔ x = 3) → c = -23 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_c_value_l348_34848


namespace NUMINAMATH_CALUDE_divisor_sum_property_l348_34824

def divisors (n : ℕ) : List ℕ := sorry

def D (n : ℕ) : ℕ := sorry

theorem divisor_sum_property (n : ℕ) (h : n > 1) :
  let d := divisors n
  D n < n^2 ∧ (D n ∣ n^2 ↔ Nat.Prime n) := by sorry

end NUMINAMATH_CALUDE_divisor_sum_property_l348_34824


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l348_34883

theorem tan_half_sum_of_angles (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 5/13) : 
  Real.tan ((a + b)/2) = 25/39 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l348_34883


namespace NUMINAMATH_CALUDE_f_difference_l348_34852

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + 3*x^4 - 4*x^3 + x^2 + 2*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = -204 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l348_34852


namespace NUMINAMATH_CALUDE_range_of_m_l348_34807

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2 / (2*m) - y^2 / (m-1) = 1 → x^2 / (a^2) + y^2 / (b^2) = 1 ∧ a > b

def q (m : ℝ) : Prop := ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ ∀ (x y : ℝ), y^2 / 5 - x^2 / m = 1 → x^2 / (5*e^2) - y^2 / 5 = 1

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, (¬(p m) ∧ ¬(q m)) ∧ (p m ∨ q m) → 1/3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l348_34807


namespace NUMINAMATH_CALUDE_no_real_roots_l348_34859

theorem no_real_roots : ∀ x : ℝ, x^2 + x + 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l348_34859


namespace NUMINAMATH_CALUDE_min_value_theorem_l348_34833

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let f := fun x : ℝ => a * x^3 + b * x + 2^x
  (∀ x ∈ Set.Icc 0 1, f x ≤ 4) ∧ (∃ x ∈ Set.Icc 0 1, f x = 4) →
  (∀ x ∈ Set.Icc (-1) 0, f x ≥ -3/2) ∧ (∃ x ∈ Set.Icc (-1) 0, f x = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l348_34833


namespace NUMINAMATH_CALUDE_fish_weight_is_eight_l348_34832

/-- Represents the weight of a fish with three parts: tail, head, and body. -/
structure FishWeight where
  tail : ℝ
  head : ℝ
  body : ℝ

/-- The conditions given in the problem -/
def fish_conditions (f : FishWeight) : Prop :=
  f.tail = 1 ∧
  f.head = f.tail + f.body / 2 ∧
  f.body = f.head + f.tail

/-- The theorem stating that a fish satisfying the given conditions weighs 8 kg -/
theorem fish_weight_is_eight (f : FishWeight) 
  (h : fish_conditions f) : f.tail + f.head + f.body = 8 := by
  sorry


end NUMINAMATH_CALUDE_fish_weight_is_eight_l348_34832


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocals_l348_34867

theorem roots_sum_of_squares_reciprocals (α : ℝ) :
  let f (x : ℝ) := x^2 + x * Real.sin α + 1
  let g (x : ℝ) := x^2 + x * Real.cos α - 1
  ∀ a b c d : ℝ,
    f a = 0 → f b = 0 → g c = 0 → g d = 0 →
    1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_reciprocals_l348_34867


namespace NUMINAMATH_CALUDE_intersection_A_B_l348_34880

def A : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l348_34880


namespace NUMINAMATH_CALUDE_minimum_employment_age_is_25_l348_34804

/-- The minimum age required to be employed at the company -/
def minimum_employment_age : ℕ := 25

/-- Jane's current age -/
def jane_current_age : ℕ := 28

/-- Years until Dara reaches minimum employment age -/
def years_until_dara_reaches_minimum_age : ℕ := 14

/-- Years until Dara is half Jane's age -/
def years_until_dara_half_jane_age : ℕ := 6

theorem minimum_employment_age_is_25 :
  minimum_employment_age = 25 :=
by sorry

end NUMINAMATH_CALUDE_minimum_employment_age_is_25_l348_34804


namespace NUMINAMATH_CALUDE_refrigerator_savings_l348_34845

/-- Calculates the savings when paying cash for a refrigerator instead of installments --/
theorem refrigerator_savings (cash_price deposit installment_amount : ℕ) (num_installments : ℕ) :
  cash_price = 8000 →
  deposit = 3000 →
  installment_amount = 300 →
  num_installments = 30 →
  deposit + num_installments * installment_amount - cash_price = 4000 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l348_34845


namespace NUMINAMATH_CALUDE_camryn_practice_schedule_l348_34849

/-- Represents the number of days between Camryn's trumpet practices -/
def trumpet_interval : ℕ := 11

/-- Represents the number of days until Camryn practices both instruments again -/
def next_joint_practice : ℕ := 33

/-- Represents the number of days between Camryn's flute practices -/
def flute_interval : ℕ := 3

theorem camryn_practice_schedule :
  (trumpet_interval > 1) ∧
  (flute_interval > 1) ∧
  (flute_interval < trumpet_interval) ∧
  (next_joint_practice % trumpet_interval = 0) ∧
  (next_joint_practice % flute_interval = 0) :=
by sorry

end NUMINAMATH_CALUDE_camryn_practice_schedule_l348_34849


namespace NUMINAMATH_CALUDE_extra_marks_for_second_candidate_l348_34895

/-- The total number of marks in the exam -/
def T : ℝ := 300

/-- The passing marks -/
def P : ℝ := 120

/-- The percentage of marks obtained by the first candidate -/
def first_candidate_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the second candidate -/
def second_candidate_percentage : ℝ := 0.45

/-- The number of marks by which the first candidate fails -/
def failing_margin : ℝ := 30

theorem extra_marks_for_second_candidate : 
  second_candidate_percentage * T - P = 15 := by sorry

end NUMINAMATH_CALUDE_extra_marks_for_second_candidate_l348_34895


namespace NUMINAMATH_CALUDE_min_value_of_expression_l348_34887

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ y : ℝ, y = 1/a + 4/b → y ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l348_34887


namespace NUMINAMATH_CALUDE_sum_squares_theorem_l348_34885

theorem sum_squares_theorem (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_theorem_l348_34885


namespace NUMINAMATH_CALUDE_least_positive_h_divisible_by_1999_l348_34844

def sequence_a : ℕ → ℤ
  | 0 => 0
  | 1 => 3
  | (n + 2) => 8 * sequence_a (n + 1) + 9 * sequence_a n + 16

def is_divisible_by_1999 (x : ℤ) : Prop := ∃ k : ℤ, x = 1999 * k

theorem least_positive_h_divisible_by_1999 :
  ∀ n : ℕ, is_divisible_by_1999 (sequence_a (n + 18) - sequence_a n) ∧
  ∀ h : ℕ, h > 0 ∧ h < 18 → ∃ m : ℕ, ¬is_divisible_by_1999 (sequence_a (m + h) - sequence_a m) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_h_divisible_by_1999_l348_34844


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l348_34860

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, x = y^3

def is_perfect_square (x : ℚ) : Prop := ∃ y : ℚ, x = y^2

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 1 ∧
    is_perfect_cube (2002 * n) ∧
    is_perfect_square (n / 2002 : ℚ) ∧
    (∀ m : ℕ, m ≥ 1 →
      is_perfect_cube (2002 * m) →
      is_perfect_square (m / 2002 : ℚ) →
      n ≤ m) ∧
    n = 2002^5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l348_34860


namespace NUMINAMATH_CALUDE_dinner_payment_difference_l348_34868

/-- The problem of calculating the difference in payment between John and Jane --/
theorem dinner_payment_difference :
  let original_price : ℝ := 36.000000000000036
  let discount_rate : ℝ := 0.10
  let tip_rate : ℝ := 0.15
  let discounted_price := original_price * (1 - discount_rate)
  let john_tip := original_price * tip_rate
  let jane_tip := discounted_price * tip_rate
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.5400000000000023 :=
by sorry

end NUMINAMATH_CALUDE_dinner_payment_difference_l348_34868


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l348_34809

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (-2 + a * Complex.I) / (1 + Complex.I)
  (z.re = 0) ↔ (a = 2) := by
sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l348_34809


namespace NUMINAMATH_CALUDE_distribute_four_students_three_groups_l348_34834

/-- The number of ways to distribute n distinct students into k distinct groups,
    where each student is in exactly one group and each group has at least one member. -/
def distribute_students (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 4 distinct students into 3 distinct groups,
    where each student is in exactly one group and each group has at least one member, is 36. -/
theorem distribute_four_students_three_groups :
  distribute_students 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_four_students_three_groups_l348_34834


namespace NUMINAMATH_CALUDE_sum_remainder_l348_34823

theorem sum_remainder (x y z : ℕ) 
  (hx : x % 53 = 31) 
  (hy : y % 53 = 45) 
  (hz : z % 53 = 6) : 
  (x + y + z) % 53 = 29 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_l348_34823


namespace NUMINAMATH_CALUDE_triangle_properties_l348_34899

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * Real.sin t.A = t.b + t.c ∧
  t.a = 6 ∧
  1/2 * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3

-- State the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = π/3 ∧ t.a + t.b + t.c = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l348_34899


namespace NUMINAMATH_CALUDE_one_slice_left_l348_34855

/-- Represents the number of bread slices used each day of the week -/
structure WeeklyBreadUsage where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat

/-- Calculates the number of bread slices left after a week -/
def slicesLeft (initialSlices : Nat) (usage : WeeklyBreadUsage) : Nat :=
  initialSlices - (usage.monday + usage.tuesday + usage.wednesday + 
                   usage.thursday + usage.friday + usage.saturday + usage.sunday)

/-- Theorem stating that 1 slice of bread is left after the week -/
theorem one_slice_left (initialSlices : Nat) (usage : WeeklyBreadUsage) :
  initialSlices = 22 ∧
  usage.monday = 2 ∧
  usage.tuesday = 3 ∧
  usage.wednesday = 4 ∧
  usage.thursday = 1 ∧
  usage.friday = 3 ∧
  usage.saturday = 5 ∧
  usage.sunday = 3 →
  slicesLeft initialSlices usage = 1 := by
  sorry

#check one_slice_left

end NUMINAMATH_CALUDE_one_slice_left_l348_34855


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l348_34801

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ y : ℝ, (3 - 4 * Complex.I) * (a + b * Complex.I) = y * Complex.I) : 
  a / b = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l348_34801


namespace NUMINAMATH_CALUDE_survey_results_l348_34800

/-- Represents the preferences of students in a class --/
structure ClassPreferences where
  total_students : Nat
  men_like_math : Nat
  men_like_lit : Nat
  men_dislike_both : Nat
  women_dislike_both : Nat
  total_men : Nat
  like_both : Nat
  like_only_math : Nat

/-- Theorem stating the results of the survey --/
theorem survey_results (prefs : ClassPreferences)
  (h1 : prefs.total_students = 35)
  (h2 : prefs.men_like_math = 7)
  (h3 : prefs.men_like_lit = 6)
  (h4 : prefs.men_dislike_both = 5)
  (h5 : prefs.women_dislike_both = 8)
  (h6 : prefs.total_men = 16)
  (h7 : prefs.like_both = 5)
  (h8 : prefs.like_only_math = 11) :
  (∃ (men_like_both women_like_only_lit : Nat),
    men_like_both = 2 ∧
    women_like_only_lit = 6) := by
  sorry


end NUMINAMATH_CALUDE_survey_results_l348_34800


namespace NUMINAMATH_CALUDE_red_pencils_count_l348_34812

/-- Given a box of pencils with blue, red, and green colors, prove that the number of red pencils is 6 --/
theorem red_pencils_count (B R G : ℕ) : 
  B + R + G = 20 →  -- Total number of pencils
  B = 6 * G →       -- Blue pencils are 6 times green pencils
  R < B →           -- Fewer red pencils than blue ones
  R = 6 :=
by sorry

end NUMINAMATH_CALUDE_red_pencils_count_l348_34812


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l348_34864

theorem alcohol_mixture_percentage (initial_volume : ℝ) (initial_percentage : ℝ) (added_pure_alcohol : ℝ) : 
  initial_volume = 6 →
  initial_percentage = 35 / 100 →
  added_pure_alcohol = 1.8 →
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_pure_alcohol
  let final_volume := initial_volume + added_pure_alcohol
  final_alcohol / final_volume = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l348_34864


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_right_triangle_max_ratio_equality_l348_34836

theorem right_triangle_max_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  (a^2 + b^2 + a*b) / c^2 ≤ 1.5 := by
sorry

theorem right_triangle_max_ratio_equality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ (a^2 + b^2 + a*b) / c^2 = 1.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_right_triangle_max_ratio_equality_l348_34836


namespace NUMINAMATH_CALUDE_locus_of_right_angle_vertex_l348_34847

-- Define the right triangle
def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the perpendicular lines (x-axis and y-axis)
def OnXAxis (P : ℝ × ℝ) : Prop := P.2 = 0
def OnYAxis (P : ℝ × ℝ) : Prop := P.1 = 0

-- Define the locus of point C
def LocusC (C : ℝ × ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), RightTriangle A B C ∧ OnXAxis A ∧ OnYAxis B

-- State the theorem
theorem locus_of_right_angle_vertex :
  ∃ (S₁ S₂ : Set (ℝ × ℝ)), 
    (∀ C, LocusC C ↔ C ∈ S₁ ∨ C ∈ S₂) ∧ 
    (∃ (a b c d : ℝ × ℝ), S₁ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • a + t • b}) ∧
    (∃ (e f g h : ℝ × ℝ), S₂ = {x | ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ x = (1-t) • e + t • f}) :=
sorry

end NUMINAMATH_CALUDE_locus_of_right_angle_vertex_l348_34847


namespace NUMINAMATH_CALUDE_sum_of_cubes_l348_34862

theorem sum_of_cubes (x y z : ℕ+) :
  (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 504 →
  (x : ℕ) + y + z = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l348_34862


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l348_34846

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l348_34846


namespace NUMINAMATH_CALUDE_minimum_gloves_needed_l348_34863

theorem minimum_gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) : 
  participants = 82 → gloves_per_participant = 2 → participants * gloves_per_participant = 164 := by
sorry

end NUMINAMATH_CALUDE_minimum_gloves_needed_l348_34863


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l348_34820

/-- The total number of dogwood trees after planting -/
def total_trees (initial : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial + planted_today + planted_tomorrow

/-- Theorem stating that the total number of dogwood trees after planting is 100 -/
theorem dogwood_tree_count :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l348_34820


namespace NUMINAMATH_CALUDE_ratio_consequent_l348_34854

theorem ratio_consequent (antecedent : ℚ) (consequent : ℚ) : 
  antecedent = 30 → (4 : ℚ) / 6 = antecedent / consequent → consequent = 45 := by
  sorry

end NUMINAMATH_CALUDE_ratio_consequent_l348_34854


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l348_34891

theorem robotics_club_enrollment (total : ℕ) (engineering : ℕ) (computer_science : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : engineering = 45)
  (h3 : computer_science = 35)
  (h4 : both = 25) :
  total - (engineering + computer_science - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l348_34891


namespace NUMINAMATH_CALUDE_largest_valid_number_l348_34894

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧
  (n.digits 10).sum % 6 = 0

theorem largest_valid_number :
  is_valid_number 936 ∧ ∀ m, is_valid_number m → m ≤ 936 :=
sorry

end NUMINAMATH_CALUDE_largest_valid_number_l348_34894


namespace NUMINAMATH_CALUDE_min_nSn_l348_34877

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℤ  -- The sequence
  S : ℕ+ → ℤ  -- Sum of first n terms
  h4 : S 4 = -2
  h5 : S 5 = 0
  h6 : S 6 = 3

/-- The product of n and S_n -/
def nSn (seq : ArithmeticSequence) (n : ℕ+) : ℤ :=
  n * seq.S n

theorem min_nSn (seq : ArithmeticSequence) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), nSn seq m ≤ nSn seq n ∧ nSn seq m = -9 :=
sorry

end NUMINAMATH_CALUDE_min_nSn_l348_34877


namespace NUMINAMATH_CALUDE_eleven_day_rental_cost_l348_34851

/-- Calculates the cost of a car rental given the number of days, daily rate, and weekly rate. -/
def rentalCost (days : ℕ) (dailyRate : ℕ) (weeklyRate : ℕ) : ℕ :=
  if days ≥ 7 then
    weeklyRate + (days - 7) * dailyRate
  else
    days * dailyRate

/-- Proves that the rental cost for 11 days is $310 given the specified rates. -/
theorem eleven_day_rental_cost :
  rentalCost 11 30 190 = 310 := by
  sorry

end NUMINAMATH_CALUDE_eleven_day_rental_cost_l348_34851


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_l348_34813

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem intersection_perpendicular_line :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧
    (∀ (x y : ℝ), result_line x y → 
      ((x - x₀) * 2 + (y - y₀) * 1 = 0)) ∧
    result_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_l348_34813


namespace NUMINAMATH_CALUDE_depreciation_is_one_fourth_l348_34872

/-- Represents the depreciation of a scooter over one year -/
def scooter_depreciation (initial_value : ℚ) (value_after_one_year : ℚ) : ℚ :=
  (initial_value - value_after_one_year) / initial_value

/-- Theorem stating that for the given initial value and value after one year,
    the depreciation fraction is 1/4 -/
theorem depreciation_is_one_fourth :
  scooter_depreciation 40000 30000 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_depreciation_is_one_fourth_l348_34872


namespace NUMINAMATH_CALUDE_hoopit_students_count_l348_34814

/-- Represents the number of toes on each hand for Hoopits -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes on each hand for Neglarts -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands for Neglarts -/
def neglart_hands : ℕ := 5

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that the number of Hoopit students on the bus is 7 -/
theorem hoopit_students_count : 
  ∃ (h : ℕ), h * (hoopit_toes_per_hand * hoopit_hands) + 
             neglart_students * (neglart_toes_per_hand * neglart_hands) = total_toes ∧ 
             h = 7 := by
  sorry

end NUMINAMATH_CALUDE_hoopit_students_count_l348_34814


namespace NUMINAMATH_CALUDE_negation_of_universal_quantification_l348_34819

theorem negation_of_universal_quantification :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantification_l348_34819


namespace NUMINAMATH_CALUDE_valley_of_five_lakes_streams_l348_34839

structure Lake :=
  (name : String)

structure Valley :=
  (lakes : Finset Lake)
  (streams : Finset (Lake × Lake))
  (start : Lake)

def Valley.valid (v : Valley) : Prop :=
  v.lakes.card = 5 ∧
  ∃ S B : Lake,
    S ∈ v.lakes ∧
    B ∈ v.lakes ∧
    S ≠ B ∧
    (∀ fish : ℕ → Lake,
      fish 0 = v.start →
      (∀ i < 4, (fish i, fish (i + 1)) ∈ v.streams) →
      (fish 4 = S ∧ fish 4 = v.start) ∨ fish 4 = B)

theorem valley_of_five_lakes_streams (v : Valley) :
  v.valid → v.streams.card = 3 :=
sorry

end NUMINAMATH_CALUDE_valley_of_five_lakes_streams_l348_34839


namespace NUMINAMATH_CALUDE_upstream_speed_is_27_l348_34830

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  downstream : ℝ
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the upstream speed given downstream and still water speeds -/
def calculateUpstreamSpeed (downstream stillWater : ℝ) : ℝ :=
  2 * stillWater - downstream

/-- Theorem stating that given the conditions, the upstream speed is 27 kmph -/
theorem upstream_speed_is_27 (speed : RowingSpeed) 
    (h1 : speed.downstream = 35)
    (h2 : speed.stillWater = 31) :
    speed.upstream = 27 :=
  by sorry

end NUMINAMATH_CALUDE_upstream_speed_is_27_l348_34830


namespace NUMINAMATH_CALUDE_sphere_surface_area_l348_34884

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l348_34884


namespace NUMINAMATH_CALUDE_simons_school_students_l348_34811

def total_students : ℕ := 2500

theorem simons_school_students (linas_students : ℕ) 
  (h1 : linas_students * 5 = total_students) : 
  linas_students * 4 = 2000 := by
  sorry

#check simons_school_students

end NUMINAMATH_CALUDE_simons_school_students_l348_34811


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l348_34842

theorem power_of_three_mod_ten (k : ℕ) : 3^(4*k + 3) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l348_34842


namespace NUMINAMATH_CALUDE_odd_function_property_l348_34876

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is increasing on [a,b] if x₁ ≤ x₂ implies f x₁ ≤ f x₂ for all x₁, x₂ in [a,b] -/
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ ≤ x₂ ∧ x₂ ≤ b → f x₁ ≤ f x₂

theorem odd_function_property (f : ℝ → ℝ) :
  IsOdd f →
  IncreasingOn f 3 7 →
  (∀ x ∈ Set.Icc 3 6, f x ≤ 8) →
  (∀ x ∈ Set.Icc 3 6, 1 ≤ f x) →
  f (-3) + 2 * f 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l348_34876


namespace NUMINAMATH_CALUDE_solve_problem_l348_34898

-- Define the type for gender
inductive Gender
| Boy
| Girl

-- Define the type for a child
structure Child :=
  (name : String)
  (gender : Gender)
  (statement : Gender)

-- Define the problem setup
def problem_setup (sasha zhenya : Child) : Prop :=
  (sasha.name = "Sasha" ∧ zhenya.name = "Zhenya") ∧
  (sasha.gender ≠ zhenya.gender) ∧
  (sasha.statement = Gender.Boy) ∧
  (zhenya.statement = Gender.Girl) ∧
  (sasha.statement ≠ sasha.gender ∨ zhenya.statement ≠ zhenya.gender)

-- Theorem to prove
theorem solve_problem (sasha zhenya : Child) :
  problem_setup sasha zhenya →
  sasha.gender = Gender.Girl ∧ zhenya.gender = Gender.Boy :=
by
  sorry

end NUMINAMATH_CALUDE_solve_problem_l348_34898


namespace NUMINAMATH_CALUDE_vector_properties_l348_34886

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def opposite_direction (a b : V) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ b = k • a

theorem vector_properties (a : V) (h : a ≠ 0) :
  opposite_direction a (-3 • a) ∧ a - 3 • a = -2 • a := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l348_34886


namespace NUMINAMATH_CALUDE_candy_distribution_l348_34875

theorem candy_distribution (x : ℝ) (x_pos : x > 0) : 
  let al_share := (4/9 : ℝ) * x
  let bert_share := (1/3 : ℝ) * (x - al_share)
  let carl_share := (2/9 : ℝ) * (x - al_share - bert_share)
  al_share + bert_share + carl_share = x :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l348_34875


namespace NUMINAMATH_CALUDE_diana_wins_probability_l348_34892

/-- The number of sides on Apollo's die -/
def apollo_sides : ℕ := 8

/-- The number of sides on Diana's die -/
def diana_sides : ℕ := 5

/-- The probability that Diana's roll is larger than Apollo's roll -/
def probability_diana_wins : ℚ := 1/4

/-- Theorem stating that the probability of Diana winning is 1/4 -/
theorem diana_wins_probability : 
  probability_diana_wins = 1/4 := by sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l348_34892


namespace NUMINAMATH_CALUDE_chinese_sexagenary_cycle_properties_l348_34888

/-- Represents the Chinese sexagenary cycle -/
structure SexagenaryCycle where
  heavenly_stems : Fin 10
  earthly_branches : Fin 12

/-- Calculates the next year with the same combination in the sexagenary cycle -/
def next_same_year (year : Int) : Int :=
  year + 60

/-- Calculates the previous year with the same combination in the sexagenary cycle -/
def prev_same_year (year : Int) : Int :=
  year - 60

/-- Calculates a year with a specific offset in the cycle -/
def year_with_offset (base_year : Int) (offset : Int) : Int :=
  base_year + offset

theorem chinese_sexagenary_cycle_properties :
  let ren_wu_2002 : SexagenaryCycle := ⟨9, 7⟩ -- Ren (9th stem), Wu (7th branch)
  -- 1. Next Ren Wu year
  (next_same_year 2002 = 2062) ∧
  -- 2. Jiawu War year (Jia Wu)
  (year_with_offset 2002 (-108) = 1894) ∧
  -- 3. Wuxu Reform year (Wu Xu)
  (year_with_offset 2002 (-104) = 1898) ∧
  -- 4. Geng Shen years in the 20th century
  (year_with_offset 2002 (-82) = 1920) ∧
  (year_with_offset 2002 (-22) = 1980) := by
  sorry

end NUMINAMATH_CALUDE_chinese_sexagenary_cycle_properties_l348_34888


namespace NUMINAMATH_CALUDE_unique_solution_abc_squared_l348_34889

theorem unique_solution_abc_squared (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_squared_l348_34889


namespace NUMINAMATH_CALUDE_xy_max_and_x_plus_y_min_l348_34858

theorem xy_max_and_x_plus_y_min (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x + 2 * y = 6) :
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x * y ≥ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x + y ≤ a + b) ∧
  (x * y = 2 ∨ x + y = 4 * Real.sqrt 2 - 3) :=
sorry

end NUMINAMATH_CALUDE_xy_max_and_x_plus_y_min_l348_34858


namespace NUMINAMATH_CALUDE_board_numbers_prime_or_one_l348_34816

theorem board_numbers_prime_or_one (a : ℕ) (h_a_odd : Odd a) (h_a_gt_100 : a > 100)
  (h_prime : ∀ n : ℕ, n ≤ Real.sqrt (a / 5) → Nat.Prime ((a - n^2) / 4)) :
  ∀ n : ℕ, Nat.Prime ((a - n^2) / 4) ∨ (a - n^2) / 4 = 1 :=
by sorry

end NUMINAMATH_CALUDE_board_numbers_prime_or_one_l348_34816


namespace NUMINAMATH_CALUDE_min_benches_in_hall_l348_34843

/-- The minimum number of benches required in a school hall -/
def min_benches (male_students : ℕ) (female_ratio : ℕ) (students_per_bench : ℕ) : ℕ :=
  ((male_students * (female_ratio + 1) + students_per_bench - 1) / students_per_bench : ℕ)

/-- Theorem: Given the conditions, the minimum number of benches required is 29 -/
theorem min_benches_in_hall :
  min_benches 29 4 5 = 29 := by
  sorry

end NUMINAMATH_CALUDE_min_benches_in_hall_l348_34843


namespace NUMINAMATH_CALUDE_least_three_digit_seven_heavy_l348_34897

def is_seven_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_seven_heavy : 
  (∀ n : ℕ, is_three_digit n → is_seven_heavy n → 103 ≤ n) ∧ 
  is_three_digit 103 ∧ 
  is_seven_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_seven_heavy_l348_34897


namespace NUMINAMATH_CALUDE_max_profit_at_optimal_price_l348_34825

/-- Profit function for a store selling items -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_increase_rate : ℝ) (price_reduction : ℝ) : ℝ :=
  (initial_price - price_reduction - cost_price) * (initial_sales + sales_increase_rate * price_reduction)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit_at_optimal_price 
  (cost_price : ℝ) 
  (initial_price : ℝ) 
  (initial_sales : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : cost_price = 2)
  (h2 : initial_price = 13)
  (h3 : initial_sales = 500)
  (h4 : sales_increase_rate = 100) :
  ∃ (optimal_price_reduction : ℝ),
    optimal_price_reduction = 3 ∧
    profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction = 6400 ∧
    ∀ (price_reduction : ℝ), 
      profit_function cost_price initial_price initial_sales sales_increase_rate price_reduction ≤ 
      profit_function cost_price initial_price initial_sales sales_increase_rate optimal_price_reduction :=
by
  sorry

#check max_profit_at_optimal_price

end NUMINAMATH_CALUDE_max_profit_at_optimal_price_l348_34825


namespace NUMINAMATH_CALUDE_subtraction_rearrangement_l348_34829

theorem subtraction_rearrangement (a b c : ℤ) : a - b - c = a - (b + c) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_rearrangement_l348_34829


namespace NUMINAMATH_CALUDE_no_perfect_squares_with_conditions_l348_34878

theorem no_perfect_squares_with_conditions : 
  ¬∃ (n : ℕ), 
    n^2 < 20000 ∧ 
    4 ∣ n^2 ∧ 
    ∃ (k : ℕ), n^2 = (k + 1)^2 - k^2 :=
by sorry

end NUMINAMATH_CALUDE_no_perfect_squares_with_conditions_l348_34878


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l348_34865

theorem arithmetic_sequence_sum (n : ℕ) (a : ℕ → ℝ) : 
  (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
  ((n + 1) * a (n + 1) = 4) →  -- sum of odd-numbered terms
  (n * a (n + 1) = 3) →  -- sum of even-numbered terms
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l348_34865


namespace NUMINAMATH_CALUDE_bacteria_fill_count_l348_34874

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of ways to fill a table with bacteria -/
def bacteriaFillWays (m n : ℕ) : ℕ :=
  2^(n-1) * (fib (2*n+1))^(m-1)

/-- Theorem: The number of ways to fill an m×n table with non-overlapping bacteria -/
theorem bacteria_fill_count (m n : ℕ) :
  bacteriaFillWays m n = 2^(n-1) * (fib (2*n+1))^(m-1) :=
by
  sorry

/-- Property: Bacteria have horizontal bodies of natural length -/
axiom bacteria_body_natural_length : True

/-- Property: Bacteria have nonnegative number of vertical feet -/
axiom bacteria_feet_nonnegative : True

/-- Property: Bacteria feet have nonnegative natural length -/
axiom bacteria_feet_natural_length : True

/-- Property: Bacteria do not overlap in the table -/
axiom bacteria_no_overlap : True

end NUMINAMATH_CALUDE_bacteria_fill_count_l348_34874


namespace NUMINAMATH_CALUDE_emily_total_songs_l348_34870

/-- Calculates the total number of songs Emily has after buying more. -/
def total_songs (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem: Emily's total number of songs is 13 given the initial and bought amounts. -/
theorem emily_total_songs :
  total_songs 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_emily_total_songs_l348_34870


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_sqrt_differences_l348_34803

theorem max_value_of_sum_of_sqrt_differences (x y z : ℝ) 
  (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) ≤ Real.sqrt 2 + 1 ∧
  ∃ x y z, x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ z ∈ Set.Icc 0 1 ∧
    Real.sqrt (|x - y|) + Real.sqrt (|y - z|) + Real.sqrt (|z - x|) = Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_sqrt_differences_l348_34803


namespace NUMINAMATH_CALUDE_line_contains_both_points_l348_34866

/-- The line equation is 2 - kx = -4y -/
def line_equation (k x y : ℝ) : Prop := 2 - k * x = -4 * y

/-- The first point (2, -1) -/
def point1 : ℝ × ℝ := (2, -1)

/-- The second point (3, -1.5) -/
def point2 : ℝ × ℝ := (3, -1.5)

/-- The line contains both points when k = -1 -/
theorem line_contains_both_points :
  ∃! k : ℝ, line_equation k point1.1 point1.2 ∧ line_equation k point2.1 point2.2 ∧ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_both_points_l348_34866


namespace NUMINAMATH_CALUDE_total_rectangles_count_l348_34841

/-- Represents the number of rectangles a single cell can form -/
structure CellRectangles where
  count : Nat

/-- Represents a group of cells with the same rectangle-forming property -/
structure CellGroup where
  cells : Nat
  rectangles : CellRectangles

/-- Calculates the total number of rectangles for a cell group -/
def totalRectangles (group : CellGroup) : Nat :=
  group.cells * group.rectangles.count

/-- The main theorem stating the total number of rectangles -/
theorem total_rectangles_count 
  (total_cells : Nat)
  (group1 : CellGroup)
  (group2 : CellGroup)
  (h1 : total_cells = group1.cells + group2.cells)
  (h2 : total_cells = 40)
  (h3 : group1.cells = 36)
  (h4 : group1.rectangles.count = 4)
  (h5 : group2.cells = 4)
  (h6 : group2.rectangles.count = 8) :
  totalRectangles group1 + totalRectangles group2 = 176 := by
  sorry

#check total_rectangles_count

end NUMINAMATH_CALUDE_total_rectangles_count_l348_34841


namespace NUMINAMATH_CALUDE_stuffed_animals_difference_l348_34805

theorem stuffed_animals_difference (thor jake quincy : ℕ) : 
  quincy = 100 * (thor + jake) →
  jake = 2 * thor + 15 →
  quincy = 4000 →
  quincy - jake = 3969 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_difference_l348_34805


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l348_34890

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- Sequence of integers indexed by natural numbers
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_a2 : a 2 = 9)  -- Given: a_2 = 9
  (h_a5 : a 5 = 33)  -- Given: a_5 = 33
  : a 2 - a 1 = 8 :=  -- Conclusion: The common difference is 8
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l348_34890


namespace NUMINAMATH_CALUDE_complex_equation_solution_l348_34861

theorem complex_equation_solution (z : ℂ) :
  (3 + Complex.I) * z = 4 - 2 * Complex.I →
  z = 1 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l348_34861


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l348_34856

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (5 * x - 1) + Real.sqrt (x - 1) = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l348_34856


namespace NUMINAMATH_CALUDE_test_question_points_l348_34808

theorem test_question_points :
  ∀ (total_points total_questions two_point_questions : ℕ) 
    (other_question_points : ℚ),
  total_points = 100 →
  total_questions = 40 →
  two_point_questions = 30 →
  total_points = 2 * two_point_questions + (total_questions - two_point_questions) * other_question_points →
  other_question_points = 4 := by
sorry

end NUMINAMATH_CALUDE_test_question_points_l348_34808


namespace NUMINAMATH_CALUDE_simple_interest_rate_l348_34882

/-- Given a principal amount that grows to 7/6 of itself in 7 years under simple interest, 
    the annual interest rate is 1/42. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  ∃ R : ℝ, R > 0 ∧ P * (1 + 7 * R) = 7/6 * P ∧ R = 1/42 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l348_34882


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l348_34896

/-- Represents the worth of apples in terms of pears -/
def apple_worth (apples : ℚ) (pears : ℚ) : Prop :=
  apples = pears

theorem apple_pear_equivalence :
  apple_worth (3/4 * 12) 9 →
  apple_worth (2/3 * 6) 4 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l348_34896


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l348_34881

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- Given two points M(a,3) and N(-4,b) symmetric with respect to the origin, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin a 3 (-4) b) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l348_34881


namespace NUMINAMATH_CALUDE_quadratic_factorization_l348_34838

theorem quadratic_factorization (C D : ℤ) :
  (∀ y, 15 * y^2 - 74 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l348_34838


namespace NUMINAMATH_CALUDE_average_wage_is_21_l348_34857

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_workers : ℕ := male_workers + female_workers + child_workers

def total_wages : ℕ := male_workers * male_wage + female_workers * female_wage + child_workers * child_wage

theorem average_wage_is_21 : total_wages / total_workers = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_wage_is_21_l348_34857


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_81_factorial_l348_34869

/-- The largest power of 3 that divides n! -/
def largest_power_of_three_dividing_factorial (n : ℕ) : ℕ :=
  (n / 3) + (n / 9) + (n / 27) + (n / 81)

/-- The ones digit of 3^n -/
def ones_digit_of_power_of_three (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 0  -- This case should never occur

theorem ones_digit_of_largest_power_of_three_dividing_81_factorial :
  ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81) = 1 := by
  sorry

#eval ones_digit_of_power_of_three (largest_power_of_three_dividing_factorial 81)

end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_three_dividing_81_factorial_l348_34869


namespace NUMINAMATH_CALUDE_circle_and_tangent_properties_l348_34840

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 9

-- Define the center of the circle
def center (x y : ℝ) : Prop :=
  y = 2 * x ∧ x > 0 ∧ y > 0

-- Define the tangent lines
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 4 * x + 3 * y = 25

theorem circle_and_tangent_properties :
  ∃ (cx cy : ℝ),
    -- The center lies on y = 2x in the first quadrant
    center cx cy ∧
    -- The circle passes through (1, -1)
    circle_C 1 (-1) ∧
    -- (4, 3) is outside the circle
    ¬ circle_C 4 3 ∧
    -- The tangent lines touch the circle at exactly one point each
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_1 tx) ∧
    (∃ (tx ty : ℝ), circle_C tx ty ∧ tangent_line_2 tx ty) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_properties_l348_34840


namespace NUMINAMATH_CALUDE_expand_expression_l348_34806

theorem expand_expression (x : ℝ) : (15 * x^2 + 5) * 3 * x^3 = 45 * x^5 + 15 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l348_34806


namespace NUMINAMATH_CALUDE_cupcakes_theorem_l348_34871

def cupcakes_problem (initial_cupcakes : ℕ) 
                     (delmont_class : ℕ) 
                     (donnelly_class : ℕ) 
                     (teachers : ℕ) 
                     (staff : ℕ) : Prop :=
  let given_away := delmont_class + donnelly_class + teachers + staff
  initial_cupcakes - given_away = 2

theorem cupcakes_theorem : 
  cupcakes_problem 40 18 16 2 2 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_theorem_l348_34871


namespace NUMINAMATH_CALUDE_yw_equals_five_l348_34837

/-- Triangle with sides a, b, c --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle --/
structure PerimeterPoint where
  distanceFromY : ℝ

/-- Definition of the meeting point of two ants crawling from X in opposite directions --/
def meetingPoint (t : Triangle) : PerimeterPoint :=
  { distanceFromY := 5 }

/-- Theorem stating that YW = 5 for the given triangle and ant movement --/
theorem yw_equals_five (t : Triangle) 
    (h1 : t.a = 7) 
    (h2 : t.b = 8) 
    (h3 : t.c = 9) : 
  (meetingPoint t).distanceFromY = 5 := by
  sorry

end NUMINAMATH_CALUDE_yw_equals_five_l348_34837


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l348_34802

/-- Represents a co-ed softball team -/
structure SoftballTeam where
  men : ℕ
  women : ℕ

/-- Properties of the softball team -/
def validTeam (team : SoftballTeam) : Prop :=
  team.women = team.men + 6 ∧ team.men + team.women = 16

theorem men_to_women_ratio (team : SoftballTeam) (h : validTeam team) :
  (team.men : ℚ) / team.women = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l348_34802


namespace NUMINAMATH_CALUDE_complex_product_l348_34893

/-- Given complex numbers Q, E, and D, prove their product is 116i -/
theorem complex_product (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2*I ∧ D = 7 - 3*I → Q * E * D = 116*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l348_34893


namespace NUMINAMATH_CALUDE_cos_alpha_value_l348_34831

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.cos α = (3 * Real.sqrt 3 + 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l348_34831


namespace NUMINAMATH_CALUDE_paint_per_statue_l348_34821

theorem paint_per_statue (total_paint : ℚ) (num_statues : ℕ) 
  (h1 : total_paint = 7 / 16)
  (h2 : num_statues = 7) :
  total_paint / num_statues = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_per_statue_l348_34821


namespace NUMINAMATH_CALUDE_pages_copied_l348_34835

/-- The cost in cents to copy one page -/
def cost_per_page : ℕ := 4

/-- The amount available in dollars -/
def available_dollars : ℕ := 25

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The number of pages that can be copied for $25 is 625 -/
theorem pages_copied (cost_per_page : ℕ) (available_dollars : ℕ) (cents_per_dollar : ℕ) :
  (available_dollars * cents_per_dollar) / cost_per_page = 625 :=
sorry

end NUMINAMATH_CALUDE_pages_copied_l348_34835


namespace NUMINAMATH_CALUDE_positive_root_range_log_function_range_l348_34853

-- Part 1
theorem positive_root_range (a : ℝ) :
  (∃ x > 0, 4^x + 2^x = a^2 + a) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-2) :=
sorry

-- Part 2
theorem log_function_range (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, Real.log (x^2 + a*x + 1) = y) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-2) :=
sorry

end NUMINAMATH_CALUDE_positive_root_range_log_function_range_l348_34853


namespace NUMINAMATH_CALUDE_quadratic_inequality_l348_34818

theorem quadratic_inequality (x : ℝ) : x^2 - 7*x + 6 < 0 ↔ 1 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l348_34818
