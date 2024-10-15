import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_area_l1036_103641

theorem right_triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : c = 13) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1036_103641


namespace NUMINAMATH_CALUDE_critical_point_and_zeros_l1036_103611

noncomputable section

def f (a : ℝ) (x : ℝ) := Real.exp x * Real.sin x - a * Real.log (x + 1)

def f_derivative (a : ℝ) (x : ℝ) := Real.exp x * (Real.sin x + Real.cos x) - a / (x + 1)

theorem critical_point_and_zeros (a : ℝ) :
  (f_derivative a 0 = 0 → a = 1) ∧
  ((∃ x₁ ∈ Set.Ioo (-1 : ℝ) 0, f a x₁ = 0) ∧
   (∃ x₂ ∈ Set.Ioo (Real.pi / 4) Real.pi, f a x₂ = 0) →
   0 < a ∧ a < 1) :=
by sorry

-- Given condition
axiom given_inequality : Real.sqrt 2 / 2 * Real.exp (Real.pi / 4) > 1

end NUMINAMATH_CALUDE_critical_point_and_zeros_l1036_103611


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l1036_103607

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y)
  (h1 : x + 3 / x = y + 3 / y) (h2 : x + y = 4) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l1036_103607


namespace NUMINAMATH_CALUDE_sqrt_300_approximation_l1036_103659

theorem sqrt_300_approximation (ε δ : ℝ) (ε_pos : ε > 0) (δ_pos : δ > 0) 
  (h : |Real.sqrt 3 - 1.732| < δ) : 
  |Real.sqrt 300 - 17.32| < ε := by
  sorry

end NUMINAMATH_CALUDE_sqrt_300_approximation_l1036_103659


namespace NUMINAMATH_CALUDE_ellipse_equation_l1036_103655

/-- Proves that an ellipse with given conditions has the equation x^2 + 4y^2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x - y + 1 = 0
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    C.1 = 0 ∧
    line C.1 C.2 ∧
    (3 * (B.1 - A.1), 3 * (B.2 - A.2)) = (2 * (C.1 - B.1), 2 * (C.2 - B.2)) →
  e^2 * a^2 = a^2 - b^2 →
  ∀ (x y : ℝ), x^2 + 4*y^2 = 1 ↔ ellipse x y := by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1036_103655


namespace NUMINAMATH_CALUDE_average_coins_per_day_l1036_103665

def coins_collected (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day < 7 then 10 * day
  else 10 * 7 + 20

def total_coins : ℕ := (List.range 7).map (λ i => coins_collected (i + 1)) |>.sum

theorem average_coins_per_day :
  (total_coins : ℚ) / 7 = 300 / 7 := by sorry

end NUMINAMATH_CALUDE_average_coins_per_day_l1036_103665


namespace NUMINAMATH_CALUDE_tom_seashells_count_l1036_103627

/-- The number of seashells Tom and Fred found together -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := total_seashells - fred_seashells

theorem tom_seashells_count : tom_seashells = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_count_l1036_103627


namespace NUMINAMATH_CALUDE_x_range_given_inequality_l1036_103625

theorem x_range_given_inequality (a : ℝ) (h_a : a ∈ Set.Icc (-1) 1) :
  (∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0) →
  {x : ℝ | x < 1 ∨ x > 3}.Nonempty :=
by sorry

end NUMINAMATH_CALUDE_x_range_given_inequality_l1036_103625


namespace NUMINAMATH_CALUDE_farm_tax_total_l1036_103645

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  /-- Total amount collected from the village -/
  total : ℝ
  /-- Amount paid by Mr. William -/
  william_paid : ℝ
  /-- Percentage of total taxable land owned by Mr. William -/
  william_percentage : ℝ
  /-- Assertion that Mr. William's percentage is 50% -/
  h_percentage : william_percentage = 50
  /-- Assertion that Mr. William paid $480 -/
  h_william_paid : william_paid = 480
  /-- The total tax is twice what Mr. William paid -/
  h_total : total = 2 * william_paid

/-- Theorem stating that the total farm tax collected is $960 -/
theorem farm_tax_total (ft : FarmTax) : ft.total = 960 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_total_l1036_103645


namespace NUMINAMATH_CALUDE_bianca_extra_flowers_l1036_103653

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Proof that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end NUMINAMATH_CALUDE_bianca_extra_flowers_l1036_103653


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1036_103690

theorem least_subtraction_for_divisibility (n m : ℕ) : 
  ∃ k, k ≤ m ∧ (n - k) % m = 0 ∧ ∀ j, j < k → (n - j) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ k, k ≤ 87 ∧ (13604 - k) % 87 = 0 ∧ ∀ j, j < k → (13604 - j) % 87 ≠ 0 ∧ k = 32 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l1036_103690


namespace NUMINAMATH_CALUDE_complex_number_location_l1036_103642

theorem complex_number_location (z : ℂ) (h : z + z * Complex.I = 3 + 2 * Complex.I) : 
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1036_103642


namespace NUMINAMATH_CALUDE_equation_equivalence_l1036_103688

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a ≤ 10) 
  (hb : 0 < b ∧ b ≤ 10) 
  (hc : 0 < c ∧ c ≤ 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a^2 + 100 * a + 11 * b * c ↔ b + 11 * c = 10 * a :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1036_103688


namespace NUMINAMATH_CALUDE_soccer_league_teams_l1036_103631

theorem soccer_league_teams (total_games : ℕ) (h_games : total_games = 45) : 
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_soccer_league_teams_l1036_103631


namespace NUMINAMATH_CALUDE_find_B_l1036_103698

theorem find_B (A B : ℕ) (h1 : A = 21) (h2 : Nat.gcd A B = 7) (h3 : Nat.lcm A B = 105) :
  B = 35 := by
sorry

end NUMINAMATH_CALUDE_find_B_l1036_103698


namespace NUMINAMATH_CALUDE_cade_initial_marbles_l1036_103600

/-- The number of marbles Cade gave away -/
def marbles_given : ℕ := 8

/-- The number of marbles Cade has left -/
def marbles_left : ℕ := 79

/-- The initial number of marbles Cade had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem cade_initial_marbles : initial_marbles = 87 := by
  sorry

end NUMINAMATH_CALUDE_cade_initial_marbles_l1036_103600


namespace NUMINAMATH_CALUDE_garage_wheels_eq_22_l1036_103668

/-- The number of wheels in Timmy's parents' garage -/
def garage_wheels : ℕ :=
  let num_cars : ℕ := 2
  let num_lawnmowers : ℕ := 1
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 1
  let num_unicycles : ℕ := 1
  let wheels_per_car : ℕ := 4
  let wheels_per_lawnmower : ℕ := 4
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_cars * wheels_per_car +
  num_lawnmowers * wheels_per_lawnmower +
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem garage_wheels_eq_22 : garage_wheels = 22 := by
  sorry

end NUMINAMATH_CALUDE_garage_wheels_eq_22_l1036_103668


namespace NUMINAMATH_CALUDE_middle_income_sample_size_l1036_103680

/-- Calculates the number of middle-income households to be sampled in a stratified sampling method. -/
theorem middle_income_sample_size 
  (total_households : ℕ) 
  (middle_income_households : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_households = 600) 
  (h2 : middle_income_households = 360) 
  (h3 : sample_size = 80) :
  (middle_income_households : ℚ) / (total_households : ℚ) * (sample_size : ℚ) = 48 := by
  sorry

end NUMINAMATH_CALUDE_middle_income_sample_size_l1036_103680


namespace NUMINAMATH_CALUDE_base7_divisibility_l1036_103604

/-- Converts a base-7 number of the form 3dd6_7 to base 10 -/
def base7ToBase10 (d : ℕ) : ℕ := 3 * 7^3 + d * 7^2 + d * 7 + 6

/-- Checks if a number is a valid base-7 digit -/
def isValidBase7Digit (d : ℕ) : Prop := d ≤ 6

theorem base7_divisibility :
  ∀ d : ℕ, isValidBase7Digit d → (base7ToBase10 d % 13 = 0 ↔ d = 4) :=
by sorry

end NUMINAMATH_CALUDE_base7_divisibility_l1036_103604


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1036_103626

/-- Represents a repeating decimal with a single-digit repetend -/
def SingleDigitRepeatDecimal (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with a two-digit repetend -/
def TwoDigitRepeatDecimal (n : ℕ) : ℚ := n / 99

/-- Represents a repeating decimal with a three-digit repetend -/
def ThreeDigitRepeatDecimal (n : ℕ) : ℚ := n / 999

/-- The sum of 0.1̅, 0.02̅, and 0.003̅ is equal to 164/1221 -/
theorem sum_of_repeating_decimals :
  SingleDigitRepeatDecimal 1 + TwoDigitRepeatDecimal 2 + ThreeDigitRepeatDecimal 3 = 164 / 1221 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1036_103626


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1036_103633

def A : Set ℝ := {x | |x| < 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1036_103633


namespace NUMINAMATH_CALUDE_v_closed_under_multiplication_l1036_103694

def v : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^3}

theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
by sorry

end NUMINAMATH_CALUDE_v_closed_under_multiplication_l1036_103694


namespace NUMINAMATH_CALUDE_nested_radical_fifteen_l1036_103679

theorem nested_radical_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_fifteen_l1036_103679


namespace NUMINAMATH_CALUDE_sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1036_103682

/-- The largest odd number less than 100 -/
def largest_odd_two_digit : ℕ :=
  99

/-- The smallest even number greater than or equal to 100 -/
def smallest_even_three_digit : ℕ :=
  100

/-- Theorem stating the sum of the largest odd two-digit number
    and the smallest even three-digit number -/
theorem sum_special_numbers :
  largest_odd_two_digit + smallest_even_three_digit = 199 := by
  sorry

/-- Proof that largest_odd_two_digit is indeed the largest odd number less than 100 -/
theorem largest_odd_two_digit_correct :
  largest_odd_two_digit < 100 ∧
  largest_odd_two_digit % 2 = 1 ∧
  ∀ n : ℕ, n < 100 → n % 2 = 1 → n ≤ largest_odd_two_digit := by
  sorry

/-- Proof that smallest_even_three_digit is indeed the smallest even number ≥ 100 -/
theorem smallest_even_three_digit_correct :
  smallest_even_three_digit ≥ 100 ∧
  smallest_even_three_digit % 2 = 0 ∧
  ∀ n : ℕ, n ≥ 100 → n % 2 = 0 → n ≥ smallest_even_three_digit := by
  sorry

end NUMINAMATH_CALUDE_sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1036_103682


namespace NUMINAMATH_CALUDE_trip_duration_proof_l1036_103648

/-- Calculates the total time spent on a trip visiting three countries. -/
def total_trip_time (first_country_stay : ℕ) : ℕ :=
  first_country_stay + 2 * first_country_stay * 2

/-- Proves that the total trip time is 10 weeks given the specified conditions. -/
theorem trip_duration_proof :
  let first_country_stay := 2
  total_trip_time first_country_stay = 10 := by
  sorry

#eval total_trip_time 2

end NUMINAMATH_CALUDE_trip_duration_proof_l1036_103648


namespace NUMINAMATH_CALUDE_breaking_process_result_l1036_103686

/-- Represents a triangle with its three angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Determines if a triangle is acute-angled -/
def Triangle.isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

/-- Represents the operation of breaking a triangle -/
def breakTriangle (t : Triangle) : List Triangle :=
  sorry  -- Implementation details omitted

/-- Counts the total number of triangles after breaking process -/
def countTriangles (initial : Triangle) : ℕ :=
  sorry  -- Implementation details omitted

/-- The theorem to be proved -/
theorem breaking_process_result (t : Triangle) 
  (h1 : t.angle1 = 3)
  (h2 : t.angle2 = 88)
  (h3 : t.angle3 = 89) :
  countTriangles t = 11 :=
sorry

end NUMINAMATH_CALUDE_breaking_process_result_l1036_103686


namespace NUMINAMATH_CALUDE_log_equation_sum_l1036_103669

theorem log_equation_sum : ∃ (X Y Z : ℕ+),
  (∀ d : ℕ+, d ∣ X ∧ d ∣ Y ∧ d ∣ Z → d = 1) ∧
  (X : ℝ) * Real.log 3 / Real.log 180 + (Y : ℝ) * Real.log 5 / Real.log 180 = Z ∧
  X + Y + Z = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l1036_103669


namespace NUMINAMATH_CALUDE_a_8_value_l1036_103622

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m * n) = a m * a n

theorem a_8_value (a : ℕ+ → ℝ) (h_prop : sequence_property a) (h_a2 : a 2 = 3) :
  a 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_a_8_value_l1036_103622


namespace NUMINAMATH_CALUDE_equal_revenue_for_all_sellers_l1036_103652

/-- Represents an apple seller with their apple count -/
structure AppleSeller :=
  (apples : ℕ)

/-- Calculates the revenue for an apple seller given the pricing scheme -/
def revenue (seller : AppleSeller) : ℕ :=
  let batches := seller.apples / 7
  let leftovers := seller.apples % 7
  batches + 3 * leftovers

/-- The list of apple sellers with their respective apple counts -/
def sellers : List AppleSeller :=
  [⟨20⟩, ⟨40⟩, ⟨60⟩, ⟨80⟩, ⟨100⟩, ⟨120⟩, ⟨140⟩]

theorem equal_revenue_for_all_sellers :
  ∀ s ∈ sellers, revenue s = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_revenue_for_all_sellers_l1036_103652


namespace NUMINAMATH_CALUDE_queen_middle_school_teachers_l1036_103664

structure School where
  students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

def number_of_teachers (school : School) : ℕ :=
  (school.students * school.classes_per_student) / (school.students_per_class * school.classes_per_teacher)

theorem queen_middle_school_teachers :
  let queen_middle : School := {
    students := 1500,
    classes_per_student := 5,
    classes_per_teacher := 5,
    students_per_class := 25
  }
  number_of_teachers queen_middle = 60 := by
  sorry

end NUMINAMATH_CALUDE_queen_middle_school_teachers_l1036_103664


namespace NUMINAMATH_CALUDE_second_shift_widget_fraction_l1036_103601

/-- The fraction of total widgets produced by the second shift in a factory --/
theorem second_shift_widget_fraction :
  -- Define the relative productivity of second shift compared to first shift
  ∀ (second_shift_productivity : ℚ)
  -- Define the relative number of employees in first shift compared to second shift
  (first_shift_employees : ℚ),
  -- Condition: Second shift productivity is 2/3 of first shift
  second_shift_productivity = 2 / 3 →
  -- Condition: First shift has 3/4 as many employees as second shift
  first_shift_employees = 3 / 4 →
  -- Conclusion: The fraction of total widgets produced by second shift is 8/17
  (second_shift_productivity * (1 / first_shift_employees)) /
  (1 + second_shift_productivity * (1 / first_shift_employees)) = 8 / 17 := by
sorry

end NUMINAMATH_CALUDE_second_shift_widget_fraction_l1036_103601


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1036_103603

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_of_inequality :
  {x : ℝ | f x ≥ x^2 - 8*x + 15} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1036_103603


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_999973_l1036_103650

theorem sum_of_prime_factors_999973 :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    999973 = p * q * r ∧
    p + q + r = 171 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_999973_l1036_103650


namespace NUMINAMATH_CALUDE_triangle_angle_bisector_length_l1036_103619

noncomputable def angleBisectorLength (PQ PR : ℝ) (cosP : ℝ) : ℝ :=
  let QR := Real.sqrt (PQ^2 + PR^2 - 2 * PQ * PR * cosP)
  let cosHalfP := Real.sqrt ((1 + cosP) / 2)
  let QT := (5 * Real.sqrt 73) / 13
  Real.sqrt (PQ^2 + QT^2 - 2 * PQ * QT * cosHalfP)

theorem triangle_angle_bisector_length :
  ∀ (ε : ℝ), ε > 0 → 
  |angleBisectorLength 5 8 (1/5) - 5.05| < ε :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_bisector_length_l1036_103619


namespace NUMINAMATH_CALUDE_square_carpet_side_length_l1036_103614

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ (side : ℝ), side * side = area ∧ 3 < side ∧ side < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_carpet_side_length_l1036_103614


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1036_103658

theorem earth_inhabitable_fraction :
  let water_free_fraction : ℚ := 1/4
  let inhabitable_land_fraction : ℚ := 1/3
  let inhabitable_fraction : ℚ := water_free_fraction * inhabitable_land_fraction
  inhabitable_fraction = 1/12 := by
sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l1036_103658


namespace NUMINAMATH_CALUDE_football_practice_hours_l1036_103613

/-- Calculates the daily practice hours for a football team -/
def dailyPracticeHours (totalHours weekDays missedDays : ℕ) : ℚ :=
  totalHours / (weekDays - missedDays)

/-- Proves that the football team practices 5 hours daily -/
theorem football_practice_hours :
  let totalHours : ℕ := 30
  let weekDays : ℕ := 7
  let missedDays : ℕ := 1
  dailyPracticeHours totalHours weekDays missedDays = 5 := by
sorry

end NUMINAMATH_CALUDE_football_practice_hours_l1036_103613


namespace NUMINAMATH_CALUDE_triangle_theorem_l1036_103651

noncomputable section

variables {a b c : ℝ} {A B C : Real}

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a) 
  (h2 : 3 * t.c * Real.sin t.B = 4 * t.a * Real.sin t.C) : 
  Real.cos t.B = -1/4 ∧ Real.sin (2 * t.B + π/6) = -(3 * Real.sqrt 5 + 7)/16 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l1036_103651


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_proof_l1036_103666

/-- The greatest two-digit whole number whose digits have a product of 12 -/
def greatest_two_digit_product_12 : ℕ := 62

/-- Predicate to check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Function to get the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Function to get the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem greatest_two_digit_product_12_proof :
  (is_two_digit greatest_two_digit_product_12) ∧
  (tens_digit greatest_two_digit_product_12 * ones_digit greatest_two_digit_product_12 = 12) ∧
  (∀ m : ℕ, is_two_digit m → 
    tens_digit m * ones_digit m = 12 → 
    m ≤ greatest_two_digit_product_12) :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_proof_l1036_103666


namespace NUMINAMATH_CALUDE_doghouse_area_doghouse_area_value_l1036_103699

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : Real) (rope_length : Real) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3) : 
  Real := by
  sorry

#check doghouse_area

theorem doghouse_area_value : 
  doghouse_area 2 3 rfl rfl = (22 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_doghouse_area_doghouse_area_value_l1036_103699


namespace NUMINAMATH_CALUDE_dog_roaming_area_l1036_103634

/-- The area available for a dog to roam when tied to the corner of an L-shaped garden wall. -/
theorem dog_roaming_area (wall_length : ℝ) (rope_length : ℝ) : wall_length = 16 ∧ rope_length = 8 → 
  (2 * (1/4 * Real.pi * rope_length^2)) = 32 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l1036_103634


namespace NUMINAMATH_CALUDE_simplify_expression_l1036_103605

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x - 2) = 2*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1036_103605


namespace NUMINAMATH_CALUDE_grade12_selection_l1036_103643

/-- Represents the number of students selected from each grade -/
structure GradeSelection where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the ratio of students in grades 10, 11, and 12 -/
structure GradeRatio where
  k : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Theorem: Given the conditions, prove that 360 students were selected from grade 12 -/
theorem grade12_selection
  (total_sample : ℕ)
  (ratio : GradeRatio)
  (selection : GradeSelection)
  (h1 : total_sample = 1200)
  (h2 : ratio = { k := 2, grade11 := 5, grade12 := 3 })
  (h3 : selection.grade10 = 240)
  (h4 : selection.grade10 + selection.grade11 + selection.grade12 = total_sample)
  (h5 : selection.grade10 * (ratio.k + ratio.grade11 + ratio.grade12) = 
        total_sample * ratio.k) :
  selection.grade12 = 360 := by
  sorry


end NUMINAMATH_CALUDE_grade12_selection_l1036_103643


namespace NUMINAMATH_CALUDE_darry_full_ladder_steps_l1036_103620

/-- The number of times Darry climbs his full ladder -/
def full_ladder_climbs : ℕ := 10

/-- The number of steps in Darry's smaller ladder -/
def small_ladder_steps : ℕ := 6

/-- The number of times Darry climbs his smaller ladder -/
def small_ladder_climbs : ℕ := 7

/-- The total number of steps Darry climbed -/
def total_steps : ℕ := 152

/-- The number of steps in Darry's full ladder -/
def full_ladder_steps : ℕ := 11

theorem darry_full_ladder_steps :
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs = total_steps :=
by sorry

end NUMINAMATH_CALUDE_darry_full_ladder_steps_l1036_103620


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l1036_103693

-- Define the grid dimensions
def rows : Nat := 4
def columns : Nat := 10

-- Define the total number of moves
def total_moves : Nat := rows + columns - 2

-- Define the number of unrestricted paths
def unrestricted_paths : Nat := Nat.choose total_moves (rows - 1)

-- Define the number of paths through the first forbidden segment
def forbidden_paths1 : Nat := 360

-- Define the number of paths through the second forbidden segment
def forbidden_paths2 : Nat := 420

-- Theorem statement
theorem number_of_valid_paths :
  unrestricted_paths - forbidden_paths1 - forbidden_paths2 = 221 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l1036_103693


namespace NUMINAMATH_CALUDE_initial_distance_is_54km_l1036_103644

/-- Represents the cycling scenario described in the problem -/
structure CyclingScenario where
  v : ℝ  -- Initial speed in km/h
  t : ℝ  -- Time shown on cycle computer in hours
  d : ℝ  -- Initial distance from home in km

/-- The conditions of the cycling scenario -/
def scenario_conditions (s : CyclingScenario) : Prop :=
  s.d = s.v * s.t ∧  -- Initial condition
  s.d = (2/3 * s.v) + (s.v - 1) * s.t ∧  -- After first speed change
  s.d = (2/3 * s.v) + (3/4 * (s.v - 1)) + (s.v - 2) * s.t  -- After second speed change

/-- The theorem stating that the initial distance is 54 km -/
theorem initial_distance_is_54km (s : CyclingScenario) 
  (h : scenario_conditions s) : s.d = 54 := by
  sorry

#check initial_distance_is_54km

end NUMINAMATH_CALUDE_initial_distance_is_54km_l1036_103644


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1036_103629

/-- Given an arithmetic sequence {a_n} with first term a₁ = -1 and common difference d = 2,
    prove that if a_{n-1} = 15, then n = 10. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 2) →  -- Common difference is 2
  a 1 = -1 →                    -- First term is -1
  a (n - 1) = 15 →              -- a_{n-1} = 15
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1036_103629


namespace NUMINAMATH_CALUDE_solve_maple_tree_price_l1036_103639

/-- Represents the problem of calculating the price per maple tree --/
def maple_tree_price_problem (initial_cash : ℕ) (cypress_trees : ℕ) (pine_trees : ℕ) (maple_trees : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (cabin_price : ℕ) (remaining_cash : ℕ) : Prop :=
  let total_after_sale := cabin_price + remaining_cash
  let total_from_trees := total_after_sale - initial_cash
  let cypress_revenue := cypress_trees * cypress_price
  let pine_revenue := pine_trees * pine_price
  let maple_revenue := total_from_trees - cypress_revenue - pine_revenue
  maple_revenue / maple_trees = 300

/-- The main theorem stating the solution to the problem --/
theorem solve_maple_tree_price :
  maple_tree_price_problem 150 20 600 24 100 200 129000 350 := by
  sorry

#check solve_maple_tree_price

end NUMINAMATH_CALUDE_solve_maple_tree_price_l1036_103639


namespace NUMINAMATH_CALUDE_two_integers_sum_l1036_103670

theorem two_integers_sum (a b : ℕ+) : 
  a * b + a + b = 103 →
  Nat.gcd a b = 1 →
  a < 20 →
  b < 20 →
  a + b = 19 := by
sorry

end NUMINAMATH_CALUDE_two_integers_sum_l1036_103670


namespace NUMINAMATH_CALUDE_circular_cross_section_shapes_l1036_103697

-- Define the geometric shapes
inductive GeometricShape
  | Cube
  | Sphere
  | Cylinder
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cylinder => true
  | _ => false

-- Theorem stating that only sphere and cylinder can have circular cross-sections
theorem circular_cross_section_shapes :
  ∀ (shape : GeometricShape),
    canHaveCircularCrossSection shape ↔ (shape = GeometricShape.Sphere ∨ shape = GeometricShape.Cylinder) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_shapes_l1036_103697


namespace NUMINAMATH_CALUDE_hat_cost_l1036_103630

/-- Given a sale of clothes where shirts cost $5 each, jeans cost $10 per pair,
    and the total cost for 3 shirts, 2 pairs of jeans, and 4 hats is $51,
    prove that each hat costs $4. -/
theorem hat_cost (shirt_cost jeans_cost total_cost : ℕ) (hat_cost : ℕ) :
  shirt_cost = 5 →
  jeans_cost = 10 →
  total_cost = 51 →
  3 * shirt_cost + 2 * jeans_cost + 4 * hat_cost = total_cost →
  hat_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_hat_cost_l1036_103630


namespace NUMINAMATH_CALUDE_residue_of_9_pow_2010_mod_17_l1036_103647

theorem residue_of_9_pow_2010_mod_17 : 9^2010 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_of_9_pow_2010_mod_17_l1036_103647


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1036_103649

/-- Calculates the total sample size for a stratified sampling method given workshop productions and a known sample from one workshop. -/
theorem stratified_sample_size 
  (production_A production_B production_C : ℕ) 
  (sample_C : ℕ) : 
  production_A = 120 → 
  production_B = 80 → 
  production_C = 60 → 
  sample_C = 3 → 
  (production_A + production_B + production_C) * sample_C / production_C = 13 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1036_103649


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_l1036_103636

theorem average_monthly_growth_rate 
  (initial_production : ℕ) 
  (final_production : ℕ) 
  (months : ℕ) 
  (growth_rate : ℝ) :
  initial_production = 100 →
  final_production = 144 →
  months = 2 →
  initial_production * (1 + growth_rate) ^ months = final_production →
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_l1036_103636


namespace NUMINAMATH_CALUDE_nine_digit_multiply_six_property_l1036_103671

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

end NUMINAMATH_CALUDE_nine_digit_multiply_six_property_l1036_103671


namespace NUMINAMATH_CALUDE_f_properties_l1036_103663

noncomputable section

variable (a : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a ≠ 1)

def f (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties :
  (∀ x, f a x = -f a (-x)) ∧ 
  (StrictMono (f a)) ∧
  (∀ m, 1 < m → m < Real.sqrt 2 → f a (1 - m) + f a (1 - m^2) < 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1036_103663


namespace NUMINAMATH_CALUDE_decoration_time_is_five_hours_l1036_103621

/-- Represents the number of eggs Mia can decorate per hour -/
def mia_eggs_per_hour : ℕ := 24

/-- Represents the number of eggs Billy can decorate per hour -/
def billy_eggs_per_hour : ℕ := 10

/-- Represents the total number of eggs that need to be decorated -/
def total_eggs : ℕ := 170

/-- Calculates the time taken to decorate all eggs when Mia and Billy work together -/
def decoration_time : ℚ :=
  total_eggs / (mia_eggs_per_hour + billy_eggs_per_hour : ℚ)

/-- Theorem stating that the decoration time is 5 hours -/
theorem decoration_time_is_five_hours :
  decoration_time = 5 := by sorry

end NUMINAMATH_CALUDE_decoration_time_is_five_hours_l1036_103621


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l1036_103609

theorem cubic_sum_minus_product (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (sum_product_eq : x * y + x * z + y * z = 30) :
  x^3 + y^3 + z^3 - 3*x*y*z = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l1036_103609


namespace NUMINAMATH_CALUDE_paula_paint_usage_l1036_103610

/-- Represents the paint capacity and usage scenario --/
structure PaintScenario where
  initial_capacity : ℕ  -- Initial room painting capacity
  lost_cans : ℕ         -- Number of paint cans lost
  remaining_capacity : ℕ -- Remaining room painting capacity

/-- Calculates the number of cans used given a paint scenario --/
def cans_used (scenario : PaintScenario) : ℕ :=
  scenario.remaining_capacity / ((scenario.initial_capacity - scenario.remaining_capacity) / scenario.lost_cans)

/-- Theorem stating that for the given scenario, 17 cans were used --/
theorem paula_paint_usage : 
  let scenario : PaintScenario := { 
    initial_capacity := 42, 
    lost_cans := 4, 
    remaining_capacity := 34 
  }
  cans_used scenario = 17 := by sorry

end NUMINAMATH_CALUDE_paula_paint_usage_l1036_103610


namespace NUMINAMATH_CALUDE_paiges_drawers_l1036_103654

theorem paiges_drawers (clothing_per_drawer : ℕ) (total_clothing : ℕ) (num_drawers : ℕ) :
  clothing_per_drawer = 2 →
  total_clothing = 8 →
  num_drawers * clothing_per_drawer = total_clothing →
  num_drawers = 4 := by
sorry

end NUMINAMATH_CALUDE_paiges_drawers_l1036_103654


namespace NUMINAMATH_CALUDE_wire_cutting_l1036_103678

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1036_103678


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l1036_103618

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = x + (x + 1) + (x + 2) + (x + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l1036_103618


namespace NUMINAMATH_CALUDE_sum_div_four_l1036_103615

theorem sum_div_four : (4 + 44 + 444) / 4 = 123 := by
  sorry

end NUMINAMATH_CALUDE_sum_div_four_l1036_103615


namespace NUMINAMATH_CALUDE_music_program_band_members_l1036_103660

theorem music_program_band_members :
  ∀ (total_students : ℕ) 
    (band_percentage : ℚ) 
    (chorus_percentage : ℚ) 
    (band_members : ℕ) 
    (chorus_members : ℕ),
  total_students = 36 →
  band_percentage = 1/5 →
  chorus_percentage = 1/4 →
  band_members + chorus_members = total_students →
  (band_percentage * band_members : ℚ) = (chorus_percentage * chorus_members : ℚ) →
  band_members = 16 := by
sorry

end NUMINAMATH_CALUDE_music_program_band_members_l1036_103660


namespace NUMINAMATH_CALUDE_log_cube_of_nine_l1036_103676

-- Define a tolerance for approximation
def tolerance : ℝ := 0.000000000000002

-- Define the approximate equality
def approx_equal (a b : ℝ) : Prop := abs (a - b) < tolerance

theorem log_cube_of_nine (x y : ℝ) :
  approx_equal x 9 → (Real.log x^3 / Real.log 9 = y) → y = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_cube_of_nine_l1036_103676


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1036_103612

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1036_103612


namespace NUMINAMATH_CALUDE_missing_number_in_proportion_l1036_103638

theorem missing_number_in_proportion : 
  ∃ x : ℚ, (2 : ℚ) / x = (4 : ℚ) / 3 / (10 : ℚ) / 3 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_in_proportion_l1036_103638


namespace NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1036_103695

theorem cylinder_in_sphere_volume (r h R : ℝ) (hr : r = 4) (hR : R = 7) 
  (hh : h^2 = 180) : 
  (4/3 * π * R^3 - π * r^2 * h) = (728/3) * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_in_sphere_volume_l1036_103695


namespace NUMINAMATH_CALUDE_solution_l1036_103685

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The amount of gold coins Foma should give to Ierema to equalize their wealth -/
def coins_to_equalize (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

/-- Theorem stating the solution to the problem -/
theorem solution (w : MerchantWealth) 
  (h : problem_conditions w) : 
  coins_to_equalize w = 55 := by
  sorry

end NUMINAMATH_CALUDE_solution_l1036_103685


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1036_103628

theorem quadratic_factorization (a b : ℕ) (h1 : a > b) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 77 = (x - a)*(x - b)) : 
  3*b - a = 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1036_103628


namespace NUMINAMATH_CALUDE_function_value_at_four_l1036_103684

/-- Given a function g: ℝ → ℝ satisfying g(x) + 2g(1 - x) = 6x^2 - 2x for all x,
    prove that g(4) = 32/3 -/
theorem function_value_at_four
  (g : ℝ → ℝ)
  (h : ∀ x, g x + 2 * g (1 - x) = 6 * x^2 - 2 * x) :
  g 4 = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_four_l1036_103684


namespace NUMINAMATH_CALUDE_vector_coordinates_l1036_103674

/-- A vector in a 2D Cartesian coordinate system -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The standard basis vectors -/
def i : Vector2D := ⟨1, 0⟩
def j : Vector2D := ⟨0, 1⟩

/-- Vector addition -/
def add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Scalar multiplication -/
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  ⟨r * v.x, r * v.y⟩

/-- The main theorem -/
theorem vector_coordinates (x y : ℝ) :
  let a := add (smul x i) (smul y j)
  a = ⟨x, y⟩ := by sorry

end NUMINAMATH_CALUDE_vector_coordinates_l1036_103674


namespace NUMINAMATH_CALUDE_equation_solution_l1036_103691

theorem equation_solution (x : ℝ) : (10 - x)^2 = 4 * x^2 ↔ x = 10/3 ∨ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1036_103691


namespace NUMINAMATH_CALUDE_local_minima_dense_of_continuous_nowhere_monotone_l1036_103640

open Set
open Topology
open Function

/-- A function is nowhere monotone if it is not monotone on any subinterval -/
def NowhereMonotone (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y z, a ≤ x ∧ x < y ∧ y < z ∧ z ≤ b →
    (f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z)

/-- The set of local minima of a function -/
def LocalMinima (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ ε > 0, ∀ y, |y - x| < ε → f y ≥ f x}

theorem local_minima_dense_of_continuous_nowhere_monotone
  (f : ℝ → ℝ)
  (hf_cont : ContinuousOn f (Icc 0 1))
  (hf_nm : NowhereMonotone f 0 1) :
  Dense (LocalMinima f ∩ Icc 0 1) :=
sorry

end NUMINAMATH_CALUDE_local_minima_dense_of_continuous_nowhere_monotone_l1036_103640


namespace NUMINAMATH_CALUDE_x_fifth_minus_seven_x_equals_222_l1036_103675

theorem x_fifth_minus_seven_x_equals_222 (x : ℝ) (h : x = 3) : x^5 - 7*x = 222 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_seven_x_equals_222_l1036_103675


namespace NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_l1036_103692

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b → 
  ∃ (triangle : Set (ℝ × ℝ)), true := by sorry

theorem possible_third_side : ∃ (triangle : Set (ℝ × ℝ)), 
  (∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 9 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ b + c > a ∧ c + a > b) := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_possible_third_side_l1036_103692


namespace NUMINAMATH_CALUDE_binomial_sum_36_implies_n_8_l1036_103677

theorem binomial_sum_36_implies_n_8 (n : ℕ+) :
  (Nat.choose n 1 + Nat.choose n 2 = 36) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_36_implies_n_8_l1036_103677


namespace NUMINAMATH_CALUDE_prob_both_odd_bounds_l1036_103606

def range_start : ℕ := 1
def range_end : ℕ := 1000

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd : ℕ := (range_end - range_start + 1) / 2

def prob_first_odd : ℚ := count_odd / range_end

def prob_second_odd : ℚ := (count_odd - 1) / (range_end - 1)

def prob_both_odd : ℚ := prob_first_odd * prob_second_odd

theorem prob_both_odd_bounds : 1/6 < prob_both_odd ∧ prob_both_odd < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_odd_bounds_l1036_103606


namespace NUMINAMATH_CALUDE_negative_sum_l1036_103689

theorem negative_sum (a b c : ℝ) 
  (ha : 1 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -2 < c ∧ c < -1) : 
  c + b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l1036_103689


namespace NUMINAMATH_CALUDE_corn_height_after_ten_weeks_l1036_103687

/-- Represents the growth of corn plants over 10 weeks -/
def corn_growth : List ℝ := [
  2,       -- Week 1
  4,       -- Week 2
  16,      -- Week 3
  22,      -- Week 4
  8,       -- Week 5
  16,      -- Week 6
  12.33,   -- Week 7
  7.33,    -- Week 8
  24,      -- Week 9
  36       -- Week 10
]

/-- The total height of the corn plants after 10 weeks -/
def total_height : ℝ := corn_growth.sum

/-- Theorem stating that the total height of the corn plants after 10 weeks is 147.66 inches -/
theorem corn_height_after_ten_weeks : total_height = 147.66 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_ten_weeks_l1036_103687


namespace NUMINAMATH_CALUDE_arithmetic_matrix_middle_value_l1036_103667

/-- Represents a 5x5 matrix where each row and column forms an arithmetic sequence -/
def ArithmeticMatrix := Matrix (Fin 5) (Fin 5) ℝ

/-- Checks if a given row or column of the matrix forms an arithmetic sequence -/
def isArithmeticSequence (seq : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 5, i.val < 4 → seq (i + 1) = seq i + d

/-- The property that all rows and columns of the matrix form arithmetic sequences -/
def allArithmeticSequences (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 5, isArithmeticSequence (λ j => M i j)) ∧
  (∀ j : Fin 5, isArithmeticSequence (λ i => M i j))

theorem arithmetic_matrix_middle_value
  (M : ArithmeticMatrix)
  (all_arithmetic : allArithmeticSequences M)
  (first_row_start : M 0 0 = 3)
  (first_row_end : M 0 4 = 15)
  (last_row_start : M 4 0 = 25)
  (last_row_end : M 4 4 = 65) :
  M 2 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_matrix_middle_value_l1036_103667


namespace NUMINAMATH_CALUDE_last_two_digits_product_l1036_103617

theorem last_two_digits_product (n : ℤ) : 
  (∃ k : ℤ, n = 8 * k) → -- n is divisible by 8
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n % 100 = 10 * a + b ∧ a + b = 15) → -- last two digits sum to 15
  (n % 10) * ((n / 10) % 10) = 54 := by
sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l1036_103617


namespace NUMINAMATH_CALUDE_dinner_cost_l1036_103616

theorem dinner_cost (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (service_rate : ℝ)
  (h_total : total_bill = 34.5)
  (h_tax : tax_rate = 0.095)
  (h_tip : tip_rate = 0.18)
  (h_service : service_rate = 0.05) :
  ∃ (base_cost : ℝ), 
    base_cost * (1 + tax_rate + tip_rate + service_rate) = total_bill ∧ 
    base_cost = 26 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_l1036_103616


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1036_103673

theorem sqrt_equation_solution (x : ℝ) (h : x > 1) :
  (Real.sqrt (5 * x) / Real.sqrt (3 * (x - 1)) = 2) → x = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1036_103673


namespace NUMINAMATH_CALUDE_f_shape_perimeter_l1036_103683

/-- The perimeter of a shape formed by two rectangles arranged in an F shape -/
def f_perimeter (h1 w1 h2 w2 overlap_h overlap_w : ℝ) : ℝ :=
  2 * (h1 + w1) + 2 * (h2 + w2) - 2 * overlap_w

/-- Theorem: The perimeter of the F shape is 18 inches -/
theorem f_shape_perimeter :
  f_perimeter 5 3 1 5 1 3 = 18 := by
  sorry

#eval f_perimeter 5 3 1 5 1 3

end NUMINAMATH_CALUDE_f_shape_perimeter_l1036_103683


namespace NUMINAMATH_CALUDE_cat_toy_cost_l1036_103602

def initial_amount : ℚ := 1173 / 100
def amount_left : ℚ := 151 / 100

theorem cat_toy_cost : initial_amount - amount_left = 1022 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cat_toy_cost_l1036_103602


namespace NUMINAMATH_CALUDE_undamaged_tins_count_l1036_103661

theorem undamaged_tins_count (cases : ℕ) (tins_per_case : ℕ) (damage_percent : ℚ) : 
  cases = 15 → 
  tins_per_case = 24 → 
  damage_percent = 5 / 100 →
  cases * tins_per_case * (1 - damage_percent) = 342 := by
sorry

end NUMINAMATH_CALUDE_undamaged_tins_count_l1036_103661


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1036_103608

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1036_103608


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l1036_103637

theorem basketball_score_ratio : 
  ∀ (marks_two_pointers marks_three_pointers marks_free_throws : ℕ)
    (total_points : ℕ),
  marks_two_pointers = 25 →
  marks_three_pointers = 8 →
  marks_free_throws = 10 →
  total_points = 201 →
  ∃ (ratio : ℚ),
    ratio = 1/2 ∧
    (2 * marks_two_pointers * 2 + ratio * (marks_three_pointers * 3 + marks_free_throws)) +
    (marks_two_pointers * 2 + marks_three_pointers * 3 + marks_free_throws) = total_points :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l1036_103637


namespace NUMINAMATH_CALUDE_flour_calculation_l1036_103646

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := sorry

/-- The total number of cups of flour required by the recipe -/
def total_flour_required : ℕ := 10

/-- The number of cups of flour Mary still needs to add -/
def flour_to_be_added : ℕ := 4

/-- Theorem: The number of cups of flour Mary has already put in is equal to
    the difference between the total cups of flour required and the cups of flour
    she still needs to add -/
theorem flour_calculation :
  flour_already_added = total_flour_required - flour_to_be_added :=
sorry

end NUMINAMATH_CALUDE_flour_calculation_l1036_103646


namespace NUMINAMATH_CALUDE_symmetric_function_properties_l1036_103696

/-- A function satisfying certain symmetry properties -/
structure SymmetricFunction where
  f : ℝ → ℝ
  sym_2 : ∀ x, f (2 - x) = f (2 + x)
  sym_7 : ∀ x, f (7 - x) = f (7 + x)
  zero_at_origin : f 0 = 0

/-- The number of zeros of a function in an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

/-- A function is periodic with period p -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- Main theorem about SymmetricFunction -/
theorem symmetric_function_properties (sf : SymmetricFunction) :
  num_zeros sf.f (-30) 30 ≥ 13 ∧ is_periodic sf.f 10 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_properties_l1036_103696


namespace NUMINAMATH_CALUDE_lindas_savings_l1036_103656

theorem lindas_savings (savings : ℕ) : 
  (3 : ℚ) / 4 * savings + 250 = savings → savings = 1000 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l1036_103656


namespace NUMINAMATH_CALUDE_sum_of_powers_inequality_l1036_103681

theorem sum_of_powers_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 ≥ 6 ∧
  (a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 = 6 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_inequality_l1036_103681


namespace NUMINAMATH_CALUDE_highway_extension_proof_l1036_103632

def highway_extension (current_length final_length first_day_miles : ℕ) : Prop :=
  let second_day_miles := 3 * first_day_miles
  let total_built := first_day_miles + second_day_miles
  let total_extension := final_length - current_length
  let remaining_miles := total_extension - total_built
  remaining_miles = 250

theorem highway_extension_proof :
  highway_extension 200 650 50 :=
sorry

end NUMINAMATH_CALUDE_highway_extension_proof_l1036_103632


namespace NUMINAMATH_CALUDE_sin_alpha_value_l1036_103657

theorem sin_alpha_value (α : Real) :
  let P : Real × Real := (-2 * Real.sin (60 * π / 180), 2 * Real.cos (30 * π / 180))
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.sin α = Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l1036_103657


namespace NUMINAMATH_CALUDE_george_oranges_l1036_103623

def orange_problem (betty sandra emily frank george : ℕ) : Prop :=
  betty = 12 ∧
  sandra = 3 * betty ∧
  emily = 7 * sandra ∧
  frank = 5 * emily ∧
  george = (5/2 : ℚ) * frank

theorem george_oranges :
  ∀ betty sandra emily frank george : ℕ,
  orange_problem betty sandra emily frank george →
  george = 3150 :=
by
  sorry

end NUMINAMATH_CALUDE_george_oranges_l1036_103623


namespace NUMINAMATH_CALUDE_factorization_proof_l1036_103662

theorem factorization_proof (x : ℝ) : 
  3 * x^2 * (x - 2) + 4 * x * (x - 2) + 2 * (x - 2) = (x - 2) * (x + 2) * (3 * x + 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l1036_103662


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1036_103672

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (1 / (Real.cos x ^ 2 - Real.sin x ^ 2) = 25/7) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1036_103672


namespace NUMINAMATH_CALUDE_problem_solution_l1036_103635

def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

theorem problem_solution :
  (∀ m : ℝ, S m → (m < 0 ∨ m ≥ 1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1036_103635


namespace NUMINAMATH_CALUDE_total_vehicles_l1036_103624

/-- Proves that the total number of vehicles on a lot is 400, given the specified conditions -/
theorem total_vehicles (total dodge hyundai kia : ℕ) : 
  dodge = total / 2 →
  hyundai = dodge / 2 →
  kia = 100 →
  total = dodge + hyundai + kia →
  total = 400 := by
sorry

end NUMINAMATH_CALUDE_total_vehicles_l1036_103624
