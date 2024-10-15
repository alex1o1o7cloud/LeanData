import Mathlib

namespace NUMINAMATH_CALUDE_overtime_compensation_l729_72947

def total_employees : ℕ := 350
def men_pay_rate : ℚ := 10
def women_pay_rate : ℚ := 815/100

theorem overtime_compensation 
  (total_men : ℕ) 
  (men_accepted : ℕ) 
  (h1 : total_men ≤ total_employees) 
  (h2 : men_accepted ≤ total_men) 
  (h3 : ∀ (m : ℕ), m ≤ total_men → 
    men_pay_rate * m + women_pay_rate * (total_employees - m) = 
    men_pay_rate * men_accepted + women_pay_rate * (total_employees - total_men)) :
  women_pay_rate * (total_employees - total_men) = 122250/100 := by
  sorry

end NUMINAMATH_CALUDE_overtime_compensation_l729_72947


namespace NUMINAMATH_CALUDE_probability_three_digit_ending_4_divisible_by_3_l729_72940

/-- A three-digit positive integer ending in 4 -/
def ThreeDigitEndingIn4 : Type := { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = 4 }

/-- The count of three-digit positive integers ending in 4 -/
def totalCount : ℕ := 90

/-- The count of three-digit positive integers ending in 4 that are divisible by 3 -/
def divisibleBy3Count : ℕ := 33

/-- The probability that a three-digit positive integer ending in 4 is divisible by 3 -/
def probabilityDivisibleBy3 : ℚ := divisibleBy3Count / totalCount

theorem probability_three_digit_ending_4_divisible_by_3 :
  probabilityDivisibleBy3 = 11 / 30 := by sorry

end NUMINAMATH_CALUDE_probability_three_digit_ending_4_divisible_by_3_l729_72940


namespace NUMINAMATH_CALUDE_bella_stamp_difference_l729_72958

/-- Calculates the difference between truck stamps and rose stamps -/
def stamp_difference (snowflake : ℕ) (truck_surplus : ℕ) (total : ℕ) : ℕ :=
  let truck := snowflake + truck_surplus
  let rose := total - (snowflake + truck)
  truck - rose

/-- Proves that the difference between truck stamps and rose stamps is 13 -/
theorem bella_stamp_difference :
  stamp_difference 11 9 38 = 13 := by
  sorry

end NUMINAMATH_CALUDE_bella_stamp_difference_l729_72958


namespace NUMINAMATH_CALUDE_edward_work_hours_l729_72987

theorem edward_work_hours (hourly_rate : ℝ) (max_regular_hours : ℕ) (total_earnings : ℝ) :
  hourly_rate = 7 →
  max_regular_hours = 40 →
  total_earnings = 210 →
  ∃ (hours_worked : ℕ), hours_worked = 30 ∧ (hours_worked : ℝ) * hourly_rate = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_edward_work_hours_l729_72987


namespace NUMINAMATH_CALUDE_transformation_is_rotation_and_scaling_l729_72967

def rotation_90 : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def scaling_2 : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def transformation : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_is_rotation_and_scaling :
  transformation = scaling_2 * rotation_90 :=
sorry

end NUMINAMATH_CALUDE_transformation_is_rotation_and_scaling_l729_72967


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l729_72965

/-- Given a polynomial equation, prove the sum of specific coefficients --/
theorem polynomial_coefficient_sum :
  ∀ (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, a + a₁ * (x + 2) + a₂ * (x + 2)^2 + a₃ * (x + 2)^3 + a₄ * (x + 2)^4 + 
             a₅ * (x + 2)^5 + a₆ * (x + 2)^6 + a₇ * (x + 2)^7 + a₈ * (x + 2)^8 + 
             a₉ * (x + 2)^9 + a₁₀ * (x + 2)^10 + a₁₁ * (x + 2)^11 + a₁₂ * (x + 2)^12 = 
             (x^2 - 2*x - 2)^6) →
  2*a₂ + 6*a₃ + 12*a₄ + 20*a₅ + 30*a₆ + 42*a₇ + 56*a₈ + 72*a₉ + 90*a₁₀ + 110*a₁₁ + 132*a₁₂ = 492 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l729_72965


namespace NUMINAMATH_CALUDE_range_of_a_sum_of_a_and_b_l729_72927

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|
def g (a x : ℝ) : ℝ := a - |x - 2|

-- Theorem 1: If f(x) < g(x) has solutions, then a > 4
theorem range_of_a (a : ℝ) : 
  (∃ x, f x < g a x) → a > 4 := by sorry

-- Theorem 2: If the solution set of f(x) < g(x) is (b, 7/2), then a + b = 6
theorem sum_of_a_and_b (a b : ℝ) : 
  (∀ x, f x < g a x ↔ b < x ∧ x < 7/2) → a + b = 6 := by sorry

end NUMINAMATH_CALUDE_range_of_a_sum_of_a_and_b_l729_72927


namespace NUMINAMATH_CALUDE_intersection_M_N_l729_72908

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l729_72908


namespace NUMINAMATH_CALUDE_absolute_value_fraction_l729_72948

theorem absolute_value_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 10*a*b) :
  |((a - b) / (a + b))| = Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_fraction_l729_72948


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l729_72993

/-- Represents a cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let runsInFirstPart := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.targetRuns - runsInFirstPart
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPartOvers = 10)
  (h3 : game.firstPartRunRate = 6.2)
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 5.5 := by
  sorry

#eval requiredRunRate { totalOvers := 50, firstPartOvers := 10, firstPartRunRate := 6.2, targetRuns := 282 }

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l729_72993


namespace NUMINAMATH_CALUDE_distance_between_trees_l729_72977

/-- Given a yard with trees planted at equal distances, calculate the distance between consecutive trees. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : 
  yard_length = 360 ∧ num_trees = 31 → 
  (yard_length / (num_trees - 1 : ℝ)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l729_72977


namespace NUMINAMATH_CALUDE_megans_books_l729_72903

theorem megans_books (books_per_shelf : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ)
  (h1 : books_per_shelf = 7)
  (h2 : mystery_shelves = 8)
  (h3 : picture_shelves = 2) :
  books_per_shelf * (mystery_shelves + picture_shelves) = 70 :=
by sorry

end NUMINAMATH_CALUDE_megans_books_l729_72903


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_l729_72917

theorem mardi_gras_necklaces 
  (boudreaux_necklaces : ℕ)
  (rhonda_necklaces : ℕ)
  (latch_necklaces : ℕ)
  (h1 : boudreaux_necklaces = 12)
  (h2 : rhonda_necklaces = boudreaux_necklaces / 2)
  (h3 : latch_necklaces = 3 * rhonda_necklaces - 4)
  : latch_necklaces = 14 := by
  sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_l729_72917


namespace NUMINAMATH_CALUDE_tan_negative_1125_degrees_l729_72932

theorem tan_negative_1125_degrees : Real.tan ((-1125 : ℝ) * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_1125_degrees_l729_72932


namespace NUMINAMATH_CALUDE_chocolate_eggs_duration_l729_72935

/-- The number of chocolate eggs Maddy has -/
def N : ℕ := 40

/-- The number of eggs Maddy eats per weekday -/
def eggs_per_day : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weeks the chocolate eggs will last -/
def weeks_lasted : ℕ := N / (eggs_per_day * weekdays)

theorem chocolate_eggs_duration : weeks_lasted = 4 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_eggs_duration_l729_72935


namespace NUMINAMATH_CALUDE_condition_relationship_l729_72979

theorem condition_relationship (x : ℝ) : 
  (∀ x, abs x < 1 → x > -1) ∧ 
  (∃ x, x > -1 ∧ ¬(abs x < 1)) :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l729_72979


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l729_72989

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving the given rectangular floor is Rs. 28,875 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 1400 = 28875 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l729_72989


namespace NUMINAMATH_CALUDE_daeyoung_pencils_l729_72975

/-- Given the conditions of Daeyoung's purchase, prove that he bought 3 pencils. -/
theorem daeyoung_pencils :
  ∀ (E P : ℕ),
  E + P = 8 →
  300 * E + 500 * P = 3000 →
  E ≥ 1 →
  P ≥ 1 →
  P = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_daeyoung_pencils_l729_72975


namespace NUMINAMATH_CALUDE_tower_count_mod_1000_l729_72955

/-- A function that calculates the number of towers for n cubes -/
def tower_count (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 32
  | m + 4 => 4 * tower_count (m + 3)

/-- The theorem stating that the number of towers for 10 cubes is congruent to 288 mod 1000 -/
theorem tower_count_mod_1000 :
  tower_count 10 ≡ 288 [MOD 1000] :=
sorry

end NUMINAMATH_CALUDE_tower_count_mod_1000_l729_72955


namespace NUMINAMATH_CALUDE_empty_set_implies_m_zero_l729_72991

theorem empty_set_implies_m_zero (m : ℝ) : (∀ x : ℝ, m * x ≠ 1) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_set_implies_m_zero_l729_72991


namespace NUMINAMATH_CALUDE_cos_is_periodic_l729_72934

-- Define the concept of a periodic function
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define the concept of a trigonometric function
def IsTrigonometric (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, f = fun x ↦ a * Real.cos (b * x) + c * Real.sin (b * x)

-- State the theorem
theorem cos_is_periodic :
  (∀ f : ℝ → ℝ, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (fun x ↦ Real.cos x) →
  IsPeriodic (fun x ↦ Real.cos x) :=
by
  sorry

end NUMINAMATH_CALUDE_cos_is_periodic_l729_72934


namespace NUMINAMATH_CALUDE_circle_radius_l729_72959

/-- A circle with center (0, k) where k < -6 is tangent to y = x, y = -x, and y = -6.
    Its radius is 6√2. -/
theorem circle_radius (k : ℝ) (h : k < -6) :
  let center := (0 : ℝ × ℝ)
  let radius := Real.sqrt 2 * 6
  (∀ p : ℝ × ℝ, (p.1 = p.2 ∨ p.1 = -p.2 ∨ p.2 = -6) →
    ‖p - center‖ = radius) →
  radius = Real.sqrt 2 * 6 :=
by sorry


end NUMINAMATH_CALUDE_circle_radius_l729_72959


namespace NUMINAMATH_CALUDE_lcm_of_denominators_l729_72983

theorem lcm_of_denominators (a b c d e : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 6) (hd : d = 8) (he : e = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_denominators_l729_72983


namespace NUMINAMATH_CALUDE_simple_interest_principal_l729_72950

/-- Simple interest calculation -/
theorem simple_interest_principal (interest rate time principal : ℝ) :
  interest = principal * rate * time →
  rate = 0.09 →
  time = 1 →
  interest = 900 →
  principal = 10000 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l729_72950


namespace NUMINAMATH_CALUDE_inequality_condition_l729_72960

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 4 → (1 + x) * Real.log x + x ≤ x * a) ↔ 
  a ≥ (5 * Real.log 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l729_72960


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l729_72915

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_minimum_value 
  (a b c : ℝ) 
  (ha : a ≠ 0) 
  (hmin : ∀ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x ≠ (4 * a * c - b^2) / (4 * a)) :
  ∃ x ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
    quadratic_function a b c x = (4 * a^2 + 4 * a * c - b^2) / (4 * a) ∧
    ∀ y ∈ Set.Icc (-b / (2 * a)) ((2 * a - b) / (2 * a)), 
      quadratic_function a b c y ≥ (4 * a^2 + 4 * a * c - b^2) / (4 * a) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l729_72915


namespace NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l729_72986

-- Define the points
variable (A B C P Q M : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect ∠ABC
axiom bisect_ABC : angle A B P = angle P B C ∧ angle A B Q = angle Q B C

-- BM trisects ∠PBQ
axiom trisect_PBQ : angle P B M = angle M B Q ∧ 3 * angle M B Q = angle P B Q

-- Theorem to prove
theorem ratio_MBQ_ABQ : angle M B Q / angle A B Q = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ratio_MBQ_ABQ_l729_72986


namespace NUMINAMATH_CALUDE_function_properties_l729_72928

open Set

def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
  (h_even : IsEven f)
  (h_periodic : IsPeriodic f 2)
  (h_interval : ∀ x ∈ Icc 1 2, f x = x^2 + 2*x - 1) :
  ∀ x ∈ Icc 0 1, f x = x^2 - 6*x + 7 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l729_72928


namespace NUMINAMATH_CALUDE_quadratic_root_value_l729_72994

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 5 * x^2 + 20 * x + k = 0 ↔ x = (-20 + Real.sqrt 60) / 10 ∨ x = (-20 - Real.sqrt 60) / 10) 
  → k = 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l729_72994


namespace NUMINAMATH_CALUDE_carly_grape_lollipops_l729_72964

/-- The number of grape lollipops in Carly's collection --/
def grape_lollipops (total : ℕ) (cherry : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total - cherry) / non_cherry_flavors

/-- Theorem stating the number of grape lollipops in Carly's collection --/
theorem carly_grape_lollipops : 
  grape_lollipops 42 (42 / 2) 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_carly_grape_lollipops_l729_72964


namespace NUMINAMATH_CALUDE_initial_girls_count_l729_72938

theorem initial_girls_count (b g : ℚ) : 
  (3 * (g - 20) = b) →
  (6 * (b - 60) = g - 20) →
  g = 700 / 17 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l729_72938


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l729_72939

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 5 * width →
  width * length = 720 →
  length - width = 48 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l729_72939


namespace NUMINAMATH_CALUDE_high_school_total_students_l729_72943

/-- Proves that the total number of students in a high school is 1800 given specific sampling conditions --/
theorem high_school_total_students
  (first_grade_students : ℕ)
  (total_sample_size : ℕ)
  (second_grade_sample : ℕ)
  (third_grade_sample : ℕ)
  (h1 : first_grade_students = 600)
  (h2 : total_sample_size = 45)
  (h3 : second_grade_sample = 20)
  (h4 : third_grade_sample = 10)
  (h5 : ∃ (total_students : ℕ), 
    (total_sample_size : ℚ) / total_students = 
    ((total_sample_size - second_grade_sample - third_grade_sample) : ℚ) / first_grade_students) :
  ∃ (total_students : ℕ), total_students = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_total_students_l729_72943


namespace NUMINAMATH_CALUDE_expression_as_square_of_binomial_l729_72984

/-- Represents the expression (-4b-3a)(-3a+4b) -/
def expression (a b : ℝ) : ℝ := (-4*b - 3*a) * (-3*a + 4*b)

/-- Represents the square of binomial form (x - y)(x + y) = x^2 - y^2 -/
def squareOfBinomialForm (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that the given expression can be rewritten in a form 
    related to the square of a binomial -/
theorem expression_as_square_of_binomial (a b : ℝ) : 
  ∃ (x y : ℝ), expression a b = squareOfBinomialForm x y := by
  sorry

end NUMINAMATH_CALUDE_expression_as_square_of_binomial_l729_72984


namespace NUMINAMATH_CALUDE_fifth_power_sum_l729_72900

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 22)
  (h4 : a*x^4 + b*y^4 = 60) :
  a*x^5 + b*y^5 = 97089/203 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l729_72900


namespace NUMINAMATH_CALUDE_simplify_fraction_l729_72963

theorem simplify_fraction : (66 : ℚ) / 4356 = 1 / 66 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l729_72963


namespace NUMINAMATH_CALUDE_fifteen_ways_to_assign_teachers_l729_72971

/-- The number of ways to assign teachers to classes -/
def assign_teachers (n_teachers : ℕ) (n_classes : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (Nat.choose n_classes classes_per_teacher * 
   Nat.choose (n_classes - classes_per_teacher) classes_per_teacher * 
   Nat.choose (n_classes - 2 * classes_per_teacher) classes_per_teacher) / 
  Nat.factorial n_teachers

/-- Theorem stating that there are 15 ways to assign 3 teachers to 6 classes -/
theorem fifteen_ways_to_assign_teachers : 
  assign_teachers 3 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_ways_to_assign_teachers_l729_72971


namespace NUMINAMATH_CALUDE_odd_square_mod_eight_l729_72904

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_square_mod_eight_l729_72904


namespace NUMINAMATH_CALUDE_final_student_count_l729_72909

theorem final_student_count (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ)
  (h1 : initial_students = 33)
  (h2 : students_left = 18)
  (h3 : new_students = 14) :
  initial_students - students_left + new_students = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l729_72909


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l729_72990

theorem tan_theta_minus_pi_over_four (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - π/4) = -1/2 := by sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_over_four_l729_72990


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l729_72945

/-- Represents the sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The sum of interior numbers in the sixth row of Pascal's Triangle -/
def sixth_row_sum : ℕ := 30

/-- Theorem: If the sum of interior numbers in the sixth row of Pascal's Triangle is 30,
    then the sum of interior numbers in the eighth row is 126 -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = sixth_row_sum → interior_sum 8 = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l729_72945


namespace NUMINAMATH_CALUDE_prove_b_value_l729_72952

theorem prove_b_value (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 35 * b) : b = 63 := by
  sorry

end NUMINAMATH_CALUDE_prove_b_value_l729_72952


namespace NUMINAMATH_CALUDE_total_mascots_is_16x_l729_72970

/-- Represents the number of mascots Jina has -/
structure Mascots where
  x : ℕ  -- number of teddies
  y : ℕ  -- number of bunnies
  z : ℕ  -- number of koalas

/-- Calculates the total number of mascots after Jina's mom gives her more teddies -/
def totalMascots (m : Mascots) : ℕ :=
  let x_new := m.x + 2 * m.y
  x_new + m.y + m.z

/-- Theorem stating the total number of mascots is 16 times the original number of teddies -/
theorem total_mascots_is_16x (m : Mascots)
    (h1 : m.y = 3 * m.x)  -- Jina has 3 times more bunnies than teddies
    (h2 : m.z = 2 * m.y)  -- Jina has twice the number of koalas as she has bunnies
    : totalMascots m = 16 * m.x := by
  sorry

#check total_mascots_is_16x

end NUMINAMATH_CALUDE_total_mascots_is_16x_l729_72970


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l729_72953

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define an isosceles triangle
def Isosceles (t : Triangle A B C) : Prop :=
  ‖A - B‖ = ‖B - C‖

-- Define the angle measure function
def AngleMeasure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_triangle_angle_measure 
  (A B C D : ℝ × ℝ) 
  (t : Triangle A B C) 
  (h_isosceles : Isosceles t) 
  (h_angle_C : AngleMeasure B C A = 50) :
  AngleMeasure C B D = 115 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l729_72953


namespace NUMINAMATH_CALUDE_common_remainder_l729_72949

theorem common_remainder : ∃ r : ℕ, 
  r < 9 ∧ r < 11 ∧ r < 17 ∧
  (3374 % 9 = r) ∧ (3374 % 11 = r) ∧ (3374 % 17 = r) ∧
  r = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_remainder_l729_72949


namespace NUMINAMATH_CALUDE_fence_length_of_area_200_l729_72907

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  short_side : ℝ
  area : ℝ
  area_eq : area = 2 * short_side * short_side

/-- The total fence length of the special rectangle -/
def fence_length (r : SpecialRectangle) : ℝ :=
  2 * r.short_side + 2 * r.short_side

/-- Theorem: The fence length of a special rectangle with area 200 is 40 -/
theorem fence_length_of_area_200 :
  ∃ r : SpecialRectangle, r.area = 200 ∧ fence_length r = 40 := by
  sorry


end NUMINAMATH_CALUDE_fence_length_of_area_200_l729_72907


namespace NUMINAMATH_CALUDE_lagrange_mvt_example_l729_72998

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

-- State the theorem
theorem lagrange_mvt_example :
  ∃ c ∈ Set.Ioo (-1 : ℝ) 3,
    (f 3 - f (-1)) / (3 - (-1)) = 2*c + 6 :=
by
  sorry


end NUMINAMATH_CALUDE_lagrange_mvt_example_l729_72998


namespace NUMINAMATH_CALUDE_systems_solutions_l729_72996

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ 3 * x + 4 * y = 2

def system2 (x y : ℝ) : Prop :=
  3 * x - y = 8 ∧ (y - 1) / 3 = (x + 5) / 5

-- State the theorem
theorem systems_solutions :
  (∃ (x y : ℝ), system1 x y ∧ x = 2 ∧ y = -1) ∧
  (∃ (x y : ℝ), system2 x y ∧ x = 5 ∧ y = 7) := by
  sorry

end NUMINAMATH_CALUDE_systems_solutions_l729_72996


namespace NUMINAMATH_CALUDE_even_painted_faces_5x5x1_l729_72966

/-- Represents a 3D rectangular block -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a given block -/
def count_even_painted_faces (b : Block) : ℕ :=
  sorry

/-- The theorem stating that a 5x5x1 block has 12 cubes with an even number of painted faces -/
theorem even_painted_faces_5x5x1 :
  let b : Block := { length := 5, width := 5, height := 1 }
  count_even_painted_faces b = 12 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_5x5x1_l729_72966


namespace NUMINAMATH_CALUDE_sophie_germain_identity_l729_72936

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4*b^4 = (a^2 - 2*a*b + 2*b^2) * (a^2 + 2*a*b + 2*b^2) := by
  sorry

end NUMINAMATH_CALUDE_sophie_germain_identity_l729_72936


namespace NUMINAMATH_CALUDE_equation_solution_l729_72911

theorem equation_solution :
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ∧ y = -5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l729_72911


namespace NUMINAMATH_CALUDE_hexagon_triangle_area_l729_72913

/-- The area of an equilateral triangle formed by connecting the second, third, and fifth vertices
    of a regular hexagon with side length 12 cm is 36√3 cm^2. -/
theorem hexagon_triangle_area :
  let hexagon_side : ℝ := 12
  let triangle_side : ℝ := hexagon_side
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side ^ 2
  triangle_area = 36 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_area_l729_72913


namespace NUMINAMATH_CALUDE_inequality_proof_l729_72999

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : c > 1) : 
  a * b^c > b * a^c := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l729_72999


namespace NUMINAMATH_CALUDE_choose_marbles_eq_990_l729_72956

/-- The number of ways to choose 5 marbles out of 15, where exactly 2 are chosen from a set of 4 special marbles -/
def choose_marbles : ℕ :=
  let total_marbles : ℕ := 15
  let special_marbles : ℕ := 4
  let choose_total : ℕ := 5
  let choose_special : ℕ := 2
  let normal_marbles : ℕ := total_marbles - special_marbles
  let choose_normal : ℕ := choose_total - choose_special
  (Nat.choose special_marbles choose_special) * (Nat.choose normal_marbles choose_normal)

theorem choose_marbles_eq_990 : choose_marbles = 990 := by
  sorry

end NUMINAMATH_CALUDE_choose_marbles_eq_990_l729_72956


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l729_72929

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₃ + a₁₀ + a₁₁ = 36, a₃ + a₁₀ = 18 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 3 + a 10 + a 11 = 36 →
  a 3 + a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l729_72929


namespace NUMINAMATH_CALUDE_reflected_quadrilateral_area_l729_72919

/-- Represents a convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  is_convex : Bool

/-- Represents a point inside a convex quadrilateral -/
structure PointInQuadrilateral where
  quad : ConvexQuadrilateral
  is_inside : Bool

/-- Represents the quadrilateral formed by reflecting a point with respect to the midpoints of a quadrilateral's sides -/
def ReflectedQuadrilateral (p : PointInQuadrilateral) : ConvexQuadrilateral :=
  sorry

/-- The theorem stating that the area of the reflected quadrilateral is twice the area of the original quadrilateral -/
theorem reflected_quadrilateral_area 
  (q : ConvexQuadrilateral) 
  (p : PointInQuadrilateral) 
  (h1 : p.quad = q) 
  (h2 : p.is_inside = true) 
  (h3 : q.is_convex = true) :
  (ReflectedQuadrilateral p).area = 2 * q.area :=
sorry

end NUMINAMATH_CALUDE_reflected_quadrilateral_area_l729_72919


namespace NUMINAMATH_CALUDE_intersection_distance_l729_72933

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : 
  ∃ (p₁ p₂ : ℝ × ℝ), 
    (p₁.1^2 + p₁.2 = 12 ∧ p₁.1 + p₁.2 = 12) ∧ 
    (p₂.1^2 + p₂.2 = 12 ∧ p₂.1 + p₂.2 = 12) ∧ 
    p₁ ≠ p₂ ∧
    Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l729_72933


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l729_72922

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 3200

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 80

/-- Represents the number of fish in the second catch -/
def second_catch : ℕ := 80

/-- Calculates the expected number of tagged fish in the second catch -/
def expected_tagged_in_second_catch : ℚ :=
  (tagged_fish : ℚ) * (second_catch : ℚ) / (total_fish : ℚ)

theorem tagged_fish_in_second_catch :
  ⌊expected_tagged_in_second_catch⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l729_72922


namespace NUMINAMATH_CALUDE_johnny_total_planks_l729_72961

/-- Represents the number of planks needed for a table surface -/
def surface_planks (table_type : String) : ℕ :=
  match table_type with
  | "small" => 3
  | "medium" => 5
  | "large" => 7
  | _ => 0

/-- Represents the number of planks needed for table legs -/
def leg_planks : ℕ := 4

/-- Calculates the total planks needed for a given number of tables of a specific type -/
def planks_for_table_type (table_type : String) (num_tables : ℕ) : ℕ :=
  num_tables * (surface_planks table_type + leg_planks)

/-- Theorem: The total number of planks needed for Johnny's tables is 50 -/
theorem johnny_total_planks : 
  planks_for_table_type "small" 3 + 
  planks_for_table_type "medium" 2 + 
  planks_for_table_type "large" 1 = 50 := by
  sorry


end NUMINAMATH_CALUDE_johnny_total_planks_l729_72961


namespace NUMINAMATH_CALUDE_street_trees_l729_72905

theorem street_trees (road_length : ℕ) (tree_interval : ℕ) (h1 : road_length = 2575) (h2 : tree_interval = 25) : 
  (road_length / tree_interval) + 1 = 104 := by
  sorry

end NUMINAMATH_CALUDE_street_trees_l729_72905


namespace NUMINAMATH_CALUDE_roselyn_remaining_books_l729_72997

/-- The number of books Roselyn has after giving books to Mara and Rebecca -/
def books_remaining (initial_books rebecca_books : ℕ) : ℕ :=
  initial_books - (rebecca_books + 3 * rebecca_books)

/-- Theorem stating that Roselyn has 60 books after giving books to Mara and Rebecca -/
theorem roselyn_remaining_books :
  books_remaining 220 40 = 60 := by
  sorry

end NUMINAMATH_CALUDE_roselyn_remaining_books_l729_72997


namespace NUMINAMATH_CALUDE_seating_arrangement_solution_l729_72972

/-- A seating arrangement with rows of 7 or 9 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_7 : ℕ

/-- The total number of people seated -/
def total_seated (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 7 * s.rows_of_7

/-- The seating arrangement is valid if it seats exactly 112 people -/
def is_valid (s : SeatingArrangement) : Prop :=
  total_seated s = 112

theorem seating_arrangement_solution :
  ∃ (s : SeatingArrangement), is_valid s ∧ s.rows_of_9 = 7 :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_solution_l729_72972


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l729_72906

/-- Given a pipe of length 68 feet cut into two pieces, where one piece is 12 feet shorter than the other, 
    the length of the shorter piece is 28 feet. -/
theorem pipe_cut_theorem : 
  ∀ (shorter_piece longer_piece : ℝ),
  shorter_piece + longer_piece = 68 →
  longer_piece = shorter_piece + 12 →
  shorter_piece = 28 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l729_72906


namespace NUMINAMATH_CALUDE_systems_solutions_l729_72957

theorem systems_solutions :
  (∃ x y : ℝ, x = 2*y - 1 ∧ 3*x + 4*y = 17 ∧ x = 3 ∧ y = 2) ∧
  (∃ x y : ℝ, 2*x - y = 0 ∧ 3*x - 2*y = 5 ∧ x = -5 ∧ y = -10) :=
by sorry

end NUMINAMATH_CALUDE_systems_solutions_l729_72957


namespace NUMINAMATH_CALUDE_unique_k_solution_l729_72944

theorem unique_k_solution (k : ℤ) : 
  (∀ (a b c : ℝ), (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_unique_k_solution_l729_72944


namespace NUMINAMATH_CALUDE_max_value_inequality_l729_72923

theorem max_value_inequality (m n : ℝ) (hm : m ≠ -3) :
  (∀ x : ℝ, x - 3 * Real.log x + 1 ≥ m * Real.log x + n) →
  (∃ k : ℝ, k = (n - 3) / (m + 3) ∧
    k ≤ -Real.log 2 ∧
    ∀ l : ℝ, l = (n - 3) / (m + 3) → l ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l729_72923


namespace NUMINAMATH_CALUDE_family_probability_l729_72988

theorem family_probability : 
  let p_boy : ℝ := 1/2
  let p_girl : ℝ := 1/2
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  let p_at_least_one_of_each : ℝ := 1 - p_all_boys - p_all_girls
  p_at_least_one_of_each = 7/8 := by
sorry

end NUMINAMATH_CALUDE_family_probability_l729_72988


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l729_72924

theorem fifth_term_of_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l729_72924


namespace NUMINAMATH_CALUDE_special_number_between_18_and_57_l729_72910

theorem special_number_between_18_and_57 :
  ∃! n : ℕ, 18 ≤ n ∧ n ≤ 57 ∧ 
  7 ∣ n ∧ 
  (∀ p : ℕ, Prime p → p ≠ 7 → ¬(p ∣ n)) ∧
  n = 49 ∧
  Real.sqrt n = 7 := by
sorry

end NUMINAMATH_CALUDE_special_number_between_18_and_57_l729_72910


namespace NUMINAMATH_CALUDE_constant_term_expansion_l729_72931

theorem constant_term_expansion (n : ℕ+) : 
  (∃ r : ℕ, r = 6 ∧ 3*n - 4*r = 0) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l729_72931


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l729_72951

theorem no_function_satisfies_condition : 
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x + f y) = f x + Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l729_72951


namespace NUMINAMATH_CALUDE_quadratic_factorization_l729_72982

/-- Given a quadratic expression 2y^2 - 5y - 12 that can be factored as (2y + a)(y + b) 
    where a and b are integers, prove that a - b = 7 -/
theorem quadratic_factorization (a b : ℤ) : 
  (∀ y, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b)) → a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l729_72982


namespace NUMINAMATH_CALUDE_geometric_mean_max_value_l729_72978

theorem geometric_mean_max_value (a b : ℝ) (h : a^2 = (1 + 2*b) * (1 - 2*b)) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, x = (8*a*b)/(|a| + 2*|b|) → x ≤ M :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_max_value_l729_72978


namespace NUMINAMATH_CALUDE_lisa_coffee_consumption_l729_72962

/-- The number of cups of coffee Lisa drank -/
def cups_of_coffee : ℕ := sorry

/-- The amount of caffeine in milligrams per cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- Lisa's daily caffeine limit in milligrams -/
def daily_limit : ℕ := 200

/-- The amount of caffeine Lisa consumed over her daily limit in milligrams -/
def excess_caffeine : ℕ := 40

/-- Theorem stating that Lisa drank 3 cups of coffee -/
theorem lisa_coffee_consumption : cups_of_coffee = 3 := by sorry

end NUMINAMATH_CALUDE_lisa_coffee_consumption_l729_72962


namespace NUMINAMATH_CALUDE_mashed_potatoes_bacon_difference_l729_72995

/-- The number of students who suggested adding bacon -/
def bacon_students : ℕ := 269

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes_students : ℕ := 330

/-- The number of students who suggested adding tomatoes -/
def tomatoes_students : ℕ := 76

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested bacon -/
theorem mashed_potatoes_bacon_difference :
  mashed_potatoes_students - bacon_students = 61 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_bacon_difference_l729_72995


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l729_72941

/-- Represents a number in a given base --/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Converts a base 2 number to base 10 --/
def binaryToDecimal (bn : BaseNumber) : Nat :=
  bn.digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a base 10 number to base 4 --/
def decimalToQuaternary (n : Nat) : BaseNumber :=
  let rec toDigits (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toDigits (m / 4) ((m % 4) :: acc)
  { digits := toDigits n [], base := 4 }

/-- The main theorem --/
theorem binary_to_quaternary_conversion :
  let binary := BaseNumber.mk [1,0,1,1,0,0,1,0,1] 2
  let quaternary := BaseNumber.mk [2,3,0,1,1] 4
  decimalToQuaternary (binaryToDecimal binary) = quaternary := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l729_72941


namespace NUMINAMATH_CALUDE_prob_A_given_B_value_l729_72985

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 4

/-- The number of tourist spots -/
def num_spots : ℕ := 4

/-- Event A: All 4 people visit different spots -/
def event_A : Prop := True

/-- Event B: Xiao Zhao visits a spot alone -/
def event_B : Prop := True

/-- The number of ways for 3 people to visit 3 spots -/
def ways_3_people_3_spots : ℕ := 3 * 3 * 3

/-- The number of ways for Xiao Zhao to visit a spot alone -/
def ways_xiao_zhao_alone : ℕ := num_spots * ways_3_people_3_spots

/-- The number of ways 4 people can visit different spots -/
def ways_all_different : ℕ := 4 * 3 * 2 * 1

/-- The probability of event A given event B -/
def prob_A_given_B : ℚ := ways_all_different / ways_xiao_zhao_alone

theorem prob_A_given_B_value : prob_A_given_B = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_given_B_value_l729_72985


namespace NUMINAMATH_CALUDE_prob_not_all_same_five_8sided_dice_l729_72918

/-- The number of sides on each die -/
def n : ℕ := 8

/-- The number of dice rolled -/
def k : ℕ := 5

/-- The probability that not all k n-sided dice show the same number -/
def prob_not_all_same (n k : ℕ) : ℚ :=
  1 - (n : ℚ) / (n ^ k : ℚ)

/-- Theorem: The probability of not all five 8-sided dice showing the same number is 4095/4096 -/
theorem prob_not_all_same_five_8sided_dice :
  prob_not_all_same n k = 4095 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_five_8sided_dice_l729_72918


namespace NUMINAMATH_CALUDE_first_day_exceeding_threshold_l729_72902

-- Define the growth function for the bacteria colony
def bacteriaCount (n : ℕ) : ℕ := 4 * 3^n

-- Define the threshold
def threshold : ℕ := 200

-- Theorem statement
theorem first_day_exceeding_threshold :
  (∃ n : ℕ, bacteriaCount n > threshold) ∧
  (∀ k : ℕ, k < 4 → bacteriaCount k ≤ threshold) ∧
  (bacteriaCount 4 > threshold) := by
  sorry

end NUMINAMATH_CALUDE_first_day_exceeding_threshold_l729_72902


namespace NUMINAMATH_CALUDE_range_of_a_l729_72946

theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
   (x^2 + 2*x - 3 ≤ 0 → x ≤ a) ∧ 
   ∃ x, x^2 + 2*x - 3 > 0 ∧ x > a) →
  a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l729_72946


namespace NUMINAMATH_CALUDE_volleyball_tournament_triples_l729_72901

/-- Represents a round-robin volleyball tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)

/-- Represents the number of triples where each team wins once against the others -/
def count_special_triples (t : Tournament) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem volleyball_tournament_triples (t : Tournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.wins_per_team = 7) :
  count_special_triples t = 140 :=
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_triples_l729_72901


namespace NUMINAMATH_CALUDE_no_alpha_exists_for_inequality_l729_72912

theorem no_alpha_exists_for_inequality :
  ∀ α : ℝ, α > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
  sorry

end NUMINAMATH_CALUDE_no_alpha_exists_for_inequality_l729_72912


namespace NUMINAMATH_CALUDE_unfenced_side_length_is_ten_l729_72974

/-- Represents a rectangular yard with fencing on three sides -/
structure FencedYard where
  length : ℝ
  width : ℝ
  area : ℝ
  fenceLength : ℝ

/-- The unfenced side length of a rectangular yard -/
def unfencedSideLength (yard : FencedYard) : ℝ := yard.length

/-- Theorem stating the conditions and the result to be proved -/
theorem unfenced_side_length_is_ten
  (yard : FencedYard)
  (area_constraint : yard.area = 200)
  (fence_constraint : yard.fenceLength = 50)
  (rectangle_constraint : yard.area = yard.length * yard.width)
  (fence_sides_constraint : yard.fenceLength = 2 * yard.width + yard.length) :
  unfencedSideLength yard = 10 := by sorry

end NUMINAMATH_CALUDE_unfenced_side_length_is_ten_l729_72974


namespace NUMINAMATH_CALUDE_second_class_size_l729_72973

theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℚ) 
  (second_class_avg : ℚ) (total_avg : ℚ) :
  first_class_size = 30 →
  first_class_avg = 40 →
  second_class_avg = 80 →
  total_avg = 65 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size) = total_avg ∧
    second_class_size = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_class_size_l729_72973


namespace NUMINAMATH_CALUDE_fish_length_theorem_l729_72968

theorem fish_length_theorem (x : ℚ) :
  (1 / 3 : ℚ) * x + (1 / 4 : ℚ) * x + 3 = x → x = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_theorem_l729_72968


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l729_72930

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)

-- Define the length of a side
def side_length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter of a parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  side_length p.A p.B + side_length p.B p.C +
  side_length p.C p.D + side_length p.D p.A

-- Theorem statement
theorem parallelogram_perimeter (ABCD : Parallelogram)
  (h1 : side_length ABCD.A ABCD.B = 14)
  (h2 : side_length ABCD.B ABCD.C = 16) :
  perimeter ABCD = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l729_72930


namespace NUMINAMATH_CALUDE_num_orderings_eq_1554_l729_72976

/-- The number of designs --/
def n : ℕ := 12

/-- The set of all design labels --/
def designs : Finset ℕ := Finset.range n

/-- The set of completed designs --/
def completed : Finset ℕ := {10, 11}

/-- The set of designs that could still be in the pile --/
def remaining : Finset ℕ := (designs \ completed).filter (· ≤ 9)

/-- The number of possible orderings for completing the remaining designs --/
def num_orderings : ℕ :=
  Finset.sum (Finset.powerset remaining) (fun S => S.card + 2)

theorem num_orderings_eq_1554 : num_orderings = 1554 := by
  sorry

end NUMINAMATH_CALUDE_num_orderings_eq_1554_l729_72976


namespace NUMINAMATH_CALUDE_area_of_region_l729_72925

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 4*y - 6*x + 9

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (π * radius^2 = 17 * π) :=
sorry

end NUMINAMATH_CALUDE_area_of_region_l729_72925


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l729_72914

theorem mistaken_multiplication (correct_multiplier : ℕ) (actual_number : ℕ) (difference : ℕ) :
  correct_multiplier = 43 →
  actual_number = 135 →
  actual_number * correct_multiplier - actual_number * (correct_multiplier - (difference / actual_number)) = difference →
  correct_multiplier - (difference / actual_number) = 34 :=
by sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l729_72914


namespace NUMINAMATH_CALUDE_flower_problem_l729_72937

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (tulips : ℕ) (carnations : ℕ) : 
  total = 40 →
  roses_fraction = 2 / 5 →
  tulips = 10 →
  carnations = total - (roses_fraction * total).num - tulips →
  carnations = 14 := by
sorry

end NUMINAMATH_CALUDE_flower_problem_l729_72937


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l729_72980

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem necessary_not_sufficient_condition (a b : ℝ) :
  let z : ℂ := ⟨a, b⟩
  (is_pure_imaginary z → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary z) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l729_72980


namespace NUMINAMATH_CALUDE_charles_share_l729_72920

/-- The number of sheep in the inheritance problem -/
structure SheepInheritance where
  john : ℕ
  alfred : ℕ
  charles : ℕ
  alfred_more_than_john : alfred = (120 * john) / 100
  alfred_more_than_charles : alfred = (125 * charles) / 100
  john_share : john = 3600

/-- Theorem stating that Charles receives 3456 sheep -/
theorem charles_share (s : SheepInheritance) : s.charles = 3456 := by
  sorry

end NUMINAMATH_CALUDE_charles_share_l729_72920


namespace NUMINAMATH_CALUDE_max_area_cyclic_quadrilateral_l729_72942

/-- The maximum area of a cyclic quadrilateral with side lengths 1, 4, 7, and 8 is 18 -/
theorem max_area_cyclic_quadrilateral :
  let a : ℝ := 1
  let b : ℝ := 4
  let c : ℝ := 7
  let d : ℝ := 8
  let s : ℝ := (a + b + c + d) / 2
  let area : ℝ := Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))
  area = 18 := by sorry

end NUMINAMATH_CALUDE_max_area_cyclic_quadrilateral_l729_72942


namespace NUMINAMATH_CALUDE_radio_dealer_profit_l729_72969

theorem radio_dealer_profit (n d : ℕ) (h_d_pos : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 10) - d = 100) → n ≥ 13 :=
by
  sorry

end NUMINAMATH_CALUDE_radio_dealer_profit_l729_72969


namespace NUMINAMATH_CALUDE_sine_inequality_solution_set_l729_72954

theorem sine_inequality_solution_set 
  (a : ℝ) 
  (h1 : -1 < a) 
  (h2 : a < 0) 
  (θ : ℝ) 
  (h3 : θ = Real.arcsin a) : 
  {x : ℝ | ∃ (n : ℤ), (2*n - 1)*π - θ < x ∧ x < 2*n*π + θ} = 
  {x : ℝ | Real.sin x < a} := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_solution_set_l729_72954


namespace NUMINAMATH_CALUDE_parabola_equation_l729_72916

/-- Represents a parabola with focus (5,5) and directrix 4x + 9y = 36 -/
structure Parabola where
  focus : ℝ × ℝ := (5, 5)
  directrix : ℝ → ℝ → ℝ := fun x y => 4*x + 9*y - 36

/-- Represents the equation of a conic in general form -/
structure ConicEquation where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

def ConicEquation.isValid (eq : ConicEquation) : Prop :=
  eq.a > 0 ∧ Int.gcd eq.a.natAbs (Int.gcd eq.b.natAbs (Int.gcd eq.c.natAbs (Int.gcd eq.d.natAbs (Int.gcd eq.e.natAbs eq.f.natAbs)))) = 1

/-- The equation of the parabola matches the given conic equation -/
def equationMatches (p : Parabola) (eq : ConicEquation) : Prop :=
  ∀ x y : ℝ, eq.a * x^2 + eq.b * x * y + eq.c * y^2 + eq.d * x + eq.e * y + eq.f = 0 ↔
    (x - p.focus.1)^2 + (y - p.focus.2)^2 = ((4*x + 9*y - 36) / Real.sqrt 97)^2

theorem parabola_equation (p : Parabola) :
  ∃ eq : ConicEquation, eq.isValid ∧ equationMatches p eq ∧
    eq.a = 81 ∧ eq.b = -60 ∧ eq.c = 273 ∧ eq.d = -2162 ∧ eq.e = -5913 ∧ eq.f = 19407 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l729_72916


namespace NUMINAMATH_CALUDE_number_divided_by_three_l729_72926

theorem number_divided_by_three : ∃ x : ℤ, x / 3 = x - 24 ∧ x = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l729_72926


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l729_72921

/-- The smallest possible percentage of a population playing both football and basketball,
    given that 85% play football and 75% play basketball. -/
theorem smallest_overlap_percentage (total population_football population_basketball : ℝ) :
  population_football = 0.85 * total →
  population_basketball = 0.75 * total →
  total > 0 →
  ∃ (overlap : ℝ), 
    overlap ≥ 0.60 * total ∧
    overlap ≤ population_football ∧
    overlap ≤ population_basketball ∧
    ∀ (x : ℝ), 
      x ≥ 0 ∧ 
      x ≤ population_football ∧ 
      x ≤ population_basketball ∧ 
      population_football + population_basketball - x ≤ total → 
      x ≥ overlap :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l729_72921


namespace NUMINAMATH_CALUDE_shopkeeper_cheating_profit_l729_72992

/-- The percentage by which the shopkeeper increases the weight when buying from the supplier -/
def supplier_increase_percent : ℝ := 20

/-- The profit percentage the shopkeeper aims to achieve -/
def target_profit_percent : ℝ := 32

/-- The percentage by which the shopkeeper increases the weight when selling to the customer -/
def customer_increase_percent : ℝ := 26.67

theorem shopkeeper_cheating_profit (initial_weight : ℝ) (h : initial_weight > 0) :
  let actual_weight := initial_weight * (1 + supplier_increase_percent / 100)
  let selling_weight := actual_weight * (1 + customer_increase_percent / 100)
  (selling_weight - actual_weight) / initial_weight * 100 = target_profit_percent := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_cheating_profit_l729_72992


namespace NUMINAMATH_CALUDE_average_candies_sikyung_l729_72981

def sikyung_group : Finset ℕ := {16, 22, 30, 26, 18, 20}

theorem average_candies_sikyung : 
  (sikyung_group.sum id) / sikyung_group.card = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_candies_sikyung_l729_72981
