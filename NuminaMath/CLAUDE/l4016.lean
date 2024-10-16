import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4016_401639

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 5 = 0 ∧ 
               2 * s^2 - 4 * s - 5 = 0 ∧
               ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4016_401639


namespace NUMINAMATH_CALUDE_largest_fraction_l4016_401658

theorem largest_fraction : 
  let a := 4 / (2 - 1/4)
  let b := 4 / (2 + 1/4)
  let c := 4 / (2 - 1/3)
  let d := 4 / (2 + 1/3)
  let e := 4 / (2 - 1/2)
  (e ≥ a) ∧ (e ≥ b) ∧ (e ≥ c) ∧ (e ≥ d) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l4016_401658


namespace NUMINAMATH_CALUDE_sin_225_degrees_l4016_401618

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l4016_401618


namespace NUMINAMATH_CALUDE_domain_equivalence_l4016_401678

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2^x)
def domain_f_exp : Set ℝ := Set.Ioo 1 2

-- Define the domain of f(√(x^2 - 1))
def domain_f_sqrt : Set ℝ := Set.union (Set.Ioo (-Real.sqrt 17) (-Real.sqrt 5)) (Set.Ioo (Real.sqrt 5) (Real.sqrt 17))

-- Theorem statement
theorem domain_equivalence (h : ∀ x ∈ domain_f_exp, f (2^x) = f (2^x)) :
  ∀ x ∈ domain_f_sqrt, f (Real.sqrt (x^2 - 1)) = f (Real.sqrt (x^2 - 1)) :=
sorry

end NUMINAMATH_CALUDE_domain_equivalence_l4016_401678


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4016_401622

theorem inequality_solution_set (a : ℝ) (h : a > 0) :
  let S := {x : ℝ | x + 1/x > a}
  (a > 1 → S = {x | 0 < x ∧ x < 1/a} ∪ {x | x > a}) ∧
  (a = 1 → S = {x | x > 0 ∧ x ≠ 1}) ∧
  (0 < a ∧ a < 1 → S = {x | 0 < x ∧ x < a} ∪ {x | x > 1/a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4016_401622


namespace NUMINAMATH_CALUDE_yadav_clothes_transport_expense_l4016_401696

/-- Represents Mr. Yadav's monthly finances -/
structure YadavFinances where
  monthlySalary : ℝ
  consumablePercentage : ℝ
  clothesTransportPercentage : ℝ
  yearlySavings : ℝ

/-- Calculates the monthly amount spent on clothes and transport -/
def clothesTransportExpense (y : YadavFinances) : ℝ :=
  y.monthlySalary * (1 - y.consumablePercentage) * y.clothesTransportPercentage

theorem yadav_clothes_transport_expense :
  ∀ (y : YadavFinances),
    y.consumablePercentage = 0.6 →
    y.clothesTransportPercentage = 0.5 →
    y.yearlySavings = 46800 →
    y.monthlySalary * (1 - y.consumablePercentage) * (1 - y.clothesTransportPercentage) = y.yearlySavings / 12 →
    clothesTransportExpense y = 3900 := by
  sorry

#check yadav_clothes_transport_expense

end NUMINAMATH_CALUDE_yadav_clothes_transport_expense_l4016_401696


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l4016_401647

theorem complex_modulus_equality (ω : ℂ) (h : ω = 5 + 4*I) :
  Complex.abs (ω^2 + 4*ω + 41) = 2 * Real.sqrt 2009 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l4016_401647


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l4016_401603

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l4016_401603


namespace NUMINAMATH_CALUDE_sum_of_roots_l4016_401691

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 15*x = 12) 
  (hy : y^3 - 6*y^2 + 15*y = 16) : 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4016_401691


namespace NUMINAMATH_CALUDE_jerry_earnings_duration_l4016_401608

def jerry_earnings : ℕ := 14 + 31 + 20

def jerry_weekly_expenses : ℕ := 5 + 10 + 8

theorem jerry_earnings_duration : 
  ⌊(jerry_earnings : ℚ) / jerry_weekly_expenses⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_jerry_earnings_duration_l4016_401608


namespace NUMINAMATH_CALUDE_student_marks_l4016_401617

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (marks_obtained : ℕ) : 
  total_marks = 800 →
  passing_percentage = 33 / 100 →
  failed_by = 89 →
  marks_obtained = total_marks * passing_percentage - failed_by →
  marks_obtained = 175 := by
sorry

#eval (800 : ℕ) * (33 : ℚ) / 100 - 89  -- Expected output: 175

end NUMINAMATH_CALUDE_student_marks_l4016_401617


namespace NUMINAMATH_CALUDE_crayons_per_child_l4016_401601

theorem crayons_per_child (total_children : ℕ) (total_crayons : ℕ) 
  (h1 : total_children = 7) 
  (h2 : total_crayons = 56) : 
  total_crayons / total_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_child_l4016_401601


namespace NUMINAMATH_CALUDE_roots_and_element_imply_value_l4016_401632

theorem roots_and_element_imply_value (a : ℝ) :
  let A := {x : ℝ | (x - a) * (x - a + 1) = 0}
  2 ∈ A → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_element_imply_value_l4016_401632


namespace NUMINAMATH_CALUDE_courtyard_length_l4016_401657

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) :
  width = 16 →
  brick_length = 0.2 →
  brick_width = 0.1 →
  num_bricks = 14400 →
  width * (num_bricks * brick_length * brick_width / width) = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l4016_401657


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4016_401652

/-- An arithmetic sequence of integers -/
def ArithSeq (a₁ d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => ArithSeq a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def ArithSeqSum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_first_term (a₁ : ℤ) :
  (∃ d : ℤ, d > 0 ∧
    let S := ArithSeqSum a₁ d 9
    (ArithSeq a₁ d 4) * (ArithSeq a₁ d 17) > S - 4 ∧
    (ArithSeq a₁ d 12) * (ArithSeq a₁ d 9) < S + 60) →
  a₁ ∈ ({-10, -9, -8, -7, -5, -4, -3, -2} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l4016_401652


namespace NUMINAMATH_CALUDE_lisa_total_miles_l4016_401662

/-- The total miles flown by Lisa -/
def total_miles_flown (distance_per_trip : Float) (num_trips : Float) : Float :=
  distance_per_trip * num_trips

/-- Theorem stating that Lisa's total miles flown is 8192.0 -/
theorem lisa_total_miles :
  total_miles_flown 256.0 32.0 = 8192.0 := by
  sorry

end NUMINAMATH_CALUDE_lisa_total_miles_l4016_401662


namespace NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l4016_401686

-- Define the equations
def equation1 (x : ℚ) : Prop := 5 * x - 2 * (x - 1) = 3
def equation2 (x : ℚ) : Prop := (x + 3) / 2 - 1 = (2 * x - 1) / 3

-- Theorem statements
theorem solve_equation1 : ∃ x : ℚ, equation1 x ∧ x = 1/3 := by sorry

theorem solve_equation2 : ∃ x : ℚ, equation2 x ∧ x = 3 := by sorry

end NUMINAMATH_CALUDE_solve_equation1_solve_equation2_l4016_401686


namespace NUMINAMATH_CALUDE_two_integers_sum_l4016_401620

theorem two_integers_sum (x y : ℤ) (h1 : x - y = 1) (h2 : x = -4) (h3 : y = -5) : x + y = -9 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_sum_l4016_401620


namespace NUMINAMATH_CALUDE_smallest_multiple_360_l4016_401680

theorem smallest_multiple_360 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 6 ∣ n ∧ 5 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n → n ≥ 360 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_360_l4016_401680


namespace NUMINAMATH_CALUDE_square_binomial_simplification_l4016_401624

theorem square_binomial_simplification (x : ℝ) (h : 3 * x^2 - 12 ≥ 0) :
  (7 - Real.sqrt (3 * x^2 - 12))^2 = 3 * x^2 + 37 - 14 * Real.sqrt (3 * x^2 - 12) := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_simplification_l4016_401624


namespace NUMINAMATH_CALUDE_calculate_mixed_number_l4016_401609

theorem calculate_mixed_number : 7 * (9 + 2/5) - 3 = 62 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_mixed_number_l4016_401609


namespace NUMINAMATH_CALUDE_hyperbola_properties_l4016_401659

-- Define the hyperbola
def Hyperbola (a b h k : ℝ) (x y : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

-- Define the asymptotes
def Asymptote1 (x y : ℝ) : Prop := y = 3 * x + 6
def Asymptote2 (x y : ℝ) : Prop := y = -3 * x - 2

theorem hyperbola_properties :
  ∃ (a b h k : ℝ),
    (∀ x y, Asymptote1 x y ∨ Asymptote2 x y → Hyperbola a b h k x y) ∧
    Hyperbola a b h k 1 9 ∧
    a + h = (21 * Real.sqrt 6 - 8) / 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l4016_401659


namespace NUMINAMATH_CALUDE_min_value_3x_4y_l4016_401675

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = x * y) :
  3 * x + 4 * y ≥ 25 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = x * y ∧ 3 * x + 4 * y = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_4y_l4016_401675


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4016_401654

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a) → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l4016_401654


namespace NUMINAMATH_CALUDE_professors_age_l4016_401682

def guesses : List Nat := [34, 37, 39, 41, 43, 46, 48, 51, 53, 56]

def is_prime (n : Nat) : Prop := Nat.Prime n

theorem professors_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    is_prime age ∧
    (guesses.filter (· < age)).length ≥ guesses.length / 2 ∧
    (guesses.filter (fun x => x = age - 1 ∨ x = age + 1)).length = 2 ∧
    age = 47 :=
  sorry

end NUMINAMATH_CALUDE_professors_age_l4016_401682


namespace NUMINAMATH_CALUDE_challenge_points_40_l4016_401607

/-- Calculates the number of activities required for a given number of challenge points. -/
def activities_required (n : ℕ) : ℕ :=
  let segment := (n - 1) / 10
  (n.min 10) * 1 + 
  ((n - 10).max 0).min 10 * 2 + 
  ((n - 20).max 0).min 10 * 3 + 
  ((n - 30).max 0).min 10 * 4 +
  ((n - 40).max 0) * (segment + 1)

/-- Proves that 40 challenge points require 100 activities. -/
theorem challenge_points_40 : activities_required 40 = 100 := by
  sorry

#eval activities_required 40

end NUMINAMATH_CALUDE_challenge_points_40_l4016_401607


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4016_401651

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2*x - 3*y + 1 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (1, 1)

-- Define the perpendicular line l
def l (x y : ℝ) : Prop := x - y = 0

theorem intersection_and_perpendicular_line :
  -- P is on both l₁ and l₂
  (l₁ P.1 P.2 ∧ l₂ P.1 P.2) ∧ 
  -- l is perpendicular to l₂ and passes through P
  (∀ x y : ℝ, l x y → (x - P.1) * 1 + (y - P.2) * 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l4016_401651


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4016_401667

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4016_401667


namespace NUMINAMATH_CALUDE_power_of_power_ten_l4016_401625

theorem power_of_power_ten : (10^2)^5 = 10^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_ten_l4016_401625


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_XY_length_l4016_401666

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop := sorry

def longerSide (t : Triangle) (s1 s2 : ℝ × ℝ) : Prop := sorry

def triangleArea (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle_XY_length 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longerSide t t.X t.Y) 
  (h3 : triangleArea t = 36) : 
  ‖t.X - t.Y‖ = 12 := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_XY_length_l4016_401666


namespace NUMINAMATH_CALUDE_ab_value_l4016_401615

theorem ab_value (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 * b^2 + a^2 * b^3 = 20) : a * b = 2 ∨ a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l4016_401615


namespace NUMINAMATH_CALUDE_factorization_equality_l4016_401683

theorem factorization_equality (a b : ℝ) : a * b^2 - 2*a*b + a = a * (b - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4016_401683


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_solutions_l4016_401634

theorem infinitely_many_primes_with_solutions : 
  ¬ (∃ (S : Finset Nat), ∀ (p : Nat), 
    (Nat.Prime p ∧ (∃ (x y : ℤ), x^2 + x + 1 = p * y)) → p ∈ S) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_solutions_l4016_401634


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l4016_401689

theorem largest_mu_inequality :
  ∃ (μ : ℝ), μ = 3/4 ∧ 
  (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
    a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*(b*c + d*a) + c*d) ∧
  (∀ (μ' : ℝ), μ' > μ →
    ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
      a^2 + b^2 + c^2 + d^2 < a*b + μ'*(b*c + d*a) + c*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l4016_401689


namespace NUMINAMATH_CALUDE_b_work_days_l4016_401626

/-- The number of days it takes A to finish the work alone -/
def a_days : ℝ := 10

/-- The total wages when A and B work together -/
def total_wages : ℝ := 3200

/-- A's share of the wages when working together with B -/
def a_wages : ℝ := 1920

/-- The number of days it takes B to finish the work alone -/
def b_days : ℝ := 15

/-- Theorem stating that given the conditions, B can finish the work alone in 15 days -/
theorem b_work_days (a_days : ℝ) (total_wages : ℝ) (a_wages : ℝ) (b_days : ℝ) :
  a_days = 10 ∧ 
  total_wages = 3200 ∧ 
  a_wages = 1920 ∧
  (1 / a_days) / ((1 / a_days) + (1 / b_days)) = a_wages / total_wages →
  b_days = 15 :=
by sorry

end NUMINAMATH_CALUDE_b_work_days_l4016_401626


namespace NUMINAMATH_CALUDE_stratified_sampling_girls_count_l4016_401619

theorem stratified_sampling_girls_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 6)
  (h4 : sample_size = (sample_size / 2 - girls_boys_diff / 2) * 2 + girls_boys_diff) :
  (sample_size / 2 - girls_boys_diff / 2) * (total_students / sample_size) = 970 := by
  sorry

#check stratified_sampling_girls_count

end NUMINAMATH_CALUDE_stratified_sampling_girls_count_l4016_401619


namespace NUMINAMATH_CALUDE_sum_mod_nine_l4016_401636

theorem sum_mod_nine : (1234 + 1235 + 1236 + 1237 + 1238) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l4016_401636


namespace NUMINAMATH_CALUDE_simplify_trig_expression_find_sin_beta_plus_pi_over_4_l4016_401628

-- Part 1
theorem simplify_trig_expression :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by sorry

-- Part 2
theorem find_sin_beta_plus_pi_over_4 (α β : Real) 
  (h1 : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 3/5)
  (h2 : π < β ∧ β < 3*π/2) :  -- β is in the third quadrant
  Real.sin (β + π/4) = -7*Real.sqrt 2/10 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_find_sin_beta_plus_pi_over_4_l4016_401628


namespace NUMINAMATH_CALUDE_complex_square_root_l4016_401633

theorem complex_square_root : 
  ∃ (z₁ z₂ : ℂ), z₁^2 = -45 + 28*I ∧ z₂^2 = -45 + 28*I ∧ 
  z₁ = 2 + 7*I ∧ z₂ = -2 - 7*I ∧
  ∀ (z : ℂ), z^2 = -45 + 28*I → z = z₁ ∨ z = z₂ := by
sorry

end NUMINAMATH_CALUDE_complex_square_root_l4016_401633


namespace NUMINAMATH_CALUDE_square_difference_equality_l4016_401670

theorem square_difference_equality : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l4016_401670


namespace NUMINAMATH_CALUDE_virginia_eggs_l4016_401640

theorem virginia_eggs (initial_eggs : ℕ) (taken_eggs : ℕ) (final_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → final_eggs = initial_eggs - taken_eggs → final_eggs = 93 := by
  sorry

end NUMINAMATH_CALUDE_virginia_eggs_l4016_401640


namespace NUMINAMATH_CALUDE_valid_outfit_count_l4016_401600

/-- Represents the number of shirts, pants, and hats -/
def total_items : ℕ := 8

/-- Represents the number of colors for each item -/
def total_colors : ℕ := 8

/-- Represents the number of colors with matching sets -/
def matching_colors : ℕ := 6

/-- Calculates the total number of outfit combinations -/
def total_combinations : ℕ := total_items * total_items * total_items

/-- Calculates the number of restricted combinations for one pair of matching items -/
def restricted_per_pair : ℕ := matching_colors * total_items

/-- Calculates the total number of restricted combinations -/
def total_restricted : ℕ := 3 * restricted_per_pair

/-- Represents the number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - total_restricted

theorem valid_outfit_count : valid_outfits = 368 := by
  sorry

end NUMINAMATH_CALUDE_valid_outfit_count_l4016_401600


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l4016_401604

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 7 = 5 ∧ ∀ m : ℕ, m < 100 ∧ m % 7 = 5 → m ≤ n :=
  by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l4016_401604


namespace NUMINAMATH_CALUDE_origin_on_circle_l4016_401611

theorem origin_on_circle (center_x center_y radius : ℝ) 
  (h1 : center_x = 5)
  (h2 : center_y = 12)
  (h3 : radius = 13) :
  (center_x^2 + center_y^2).sqrt = radius :=
sorry

end NUMINAMATH_CALUDE_origin_on_circle_l4016_401611


namespace NUMINAMATH_CALUDE_cistern_filling_time_l4016_401614

theorem cistern_filling_time (x : ℝ) 
  (h1 : x > 0)
  (h2 : 1/x + 1/12 - 1/15 = 7/60) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l4016_401614


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4016_401665

/-- The common fraction form of the repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 4 / 11

/-- The reciprocal of the common fraction form of 0.363636... -/
def reciprocal : ℚ := 11 / 4

theorem reciprocal_of_repeating_decimal : (repeating_decimal)⁻¹ = reciprocal := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_decimal_l4016_401665


namespace NUMINAMATH_CALUDE_range_of_m_existence_of_a_b_l4016_401621

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Part 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 :=
sorry

-- Part 2: Existence of integers a and b
theorem existence_of_a_b :
  ∃ (a b : ℤ), a < b ∧
  (∀ x : ℝ, a ≤ f (↑a + ↑b - 1) x ∧ f (↑a + ↑b - 1) x ≤ b ↔ a ≤ x ∧ x ≤ b) ∧
  a = 0 ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_existence_of_a_b_l4016_401621


namespace NUMINAMATH_CALUDE_johns_trip_duration_l4016_401698

/-- Represents the duration of stay in weeks for each country visited -/
structure TripDuration where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total duration of a trip given the stay durations -/
def totalDuration (t : TripDuration) : ℕ :=
  t.first + t.second + t.third

/-- Theorem: The total duration of John's trip is 10 weeks -/
theorem johns_trip_duration :
  ∃ (t : TripDuration),
    t.first = 2 ∧
    t.second = 2 * t.first ∧
    t.third = 2 * t.first ∧
    totalDuration t = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_trip_duration_l4016_401698


namespace NUMINAMATH_CALUDE_chantel_bracelets_l4016_401635

/-- Represents the number of bracelets Chantel makes per day in the last four days -/
def x : ℕ := sorry

/-- The total number of bracelets Chantel has at the end -/
def total_bracelets : ℕ := 13

/-- The number of bracelets Chantel makes in the first 5 days -/
def first_phase_bracelets : ℕ := 2 * 5

/-- The number of bracelets Chantel gives away after the first phase -/
def first_giveaway : ℕ := 3

/-- The number of bracelets Chantel gives away after the second phase -/
def second_giveaway : ℕ := 6

/-- The number of days in the second phase -/
def second_phase_days : ℕ := 4

theorem chantel_bracelets : 
  first_phase_bracelets - first_giveaway + x * second_phase_days - second_giveaway = total_bracelets ∧ 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l4016_401635


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l4016_401692

/-- The focal length of the hyperbola x²/3 - y² = 1 is 4 -/
theorem hyperbola_focal_length : ∃ (f : ℝ), f = 4 ∧ 
  f = 2 * Real.sqrt ((3 : ℝ) + 1) ∧
  ∀ (x y : ℝ), x^2 / 3 - y^2 = 1 → 
    ∃ (a b c : ℝ), a^2 = 3 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ f = 2 * c :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l4016_401692


namespace NUMINAMATH_CALUDE_remainder_seven_count_l4016_401694

theorem remainder_seven_count : 
  (Finset.filter (fun d => d > 7 ∧ 59 % d = 7) (Finset.range 60)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_count_l4016_401694


namespace NUMINAMATH_CALUDE_triangle_problem_l4016_401655

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * cos B + b * cos A = 2 * c * cos C →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧
  (∃ (area : ℝ), area ≤ 3 * Real.sqrt 3 ∧
    ∀ (area' : ℝ), area' = 1/2 * a * b * sin C → area' ≤ area) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l4016_401655


namespace NUMINAMATH_CALUDE_circle_intersection_r_range_l4016_401697

/-- Two circles have a common point if and only if the distance between their centers
    is between the difference and sum of their radii. -/
axiom circles_intersect (c₁ c₂ : ℝ × ℝ) (r₁ r₂ : ℝ) :
  ∃ (p : ℝ × ℝ), (p.1 - c₁.1)^2 + (p.2 - c₁.2)^2 = r₁^2 ∧
                 (p.1 - c₂.1)^2 + (p.2 - c₂.2)^2 = r₂^2 ↔
  |r₁ - r₂| ≤ Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) ∧
  Real.sqrt ((c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2) ≤ r₁ + r₂

/-- The theorem stating the range of r for two intersecting circles -/
theorem circle_intersection_r_range :
  ∀ r : ℝ, r > 0 →
  (∃ (p : ℝ × ℝ), p.1^2 + p.2^2 = r^2 ∧ (p.1 - 3)^2 + (p.2 + 4)^2 = 49) →
  2 ≤ r ∧ r ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_r_range_l4016_401697


namespace NUMINAMATH_CALUDE_basketball_league_female_fraction_l4016_401623

theorem basketball_league_female_fraction :
  -- Define variables
  let last_year_males : ℕ := 30
  let last_year_females : ℕ := 15  -- Derived from the solution
  let male_increase_rate : ℚ := 11/10
  let female_increase_rate : ℚ := 5/4
  let total_increase_rate : ℚ := 23/20

  -- Define this year's participants
  let this_year_males : ℚ := last_year_males * male_increase_rate
  let this_year_females : ℚ := last_year_females * female_increase_rate
  let this_year_total : ℚ := (last_year_males + last_year_females) * total_increase_rate

  -- The fraction of female participants this year
  this_year_females / this_year_total = 75 / 207 := by
sorry

end NUMINAMATH_CALUDE_basketball_league_female_fraction_l4016_401623


namespace NUMINAMATH_CALUDE_division_remainder_l4016_401641

theorem division_remainder : ∃ q : ℕ, 1234567 = 257 * q + 123 ∧ 123 < 257 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l4016_401641


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l4016_401656

theorem quadratic_roots_sum_of_squares : ∃ (x₁ x₂ : ℝ),
  (x₁^2 - 9*x₁ + 9 = 0) ∧
  (x₂^2 - 9*x₂ + 9 = 0) ∧
  (x₁^2 + x₂^2 = 63) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l4016_401656


namespace NUMINAMATH_CALUDE_star_not_associative_l4016_401627

-- Define the set T as non-zero real numbers
def T := {x : ℝ | x ≠ 0}

-- Define the binary operation ★
def star (x y : ℝ) : ℝ := 3 * x * y + x + y

-- Theorem stating that ★ is not associative over T
theorem star_not_associative :
  ∃ (x y z : T), star (star x y) z ≠ star x (star y z) := by
  sorry

end NUMINAMATH_CALUDE_star_not_associative_l4016_401627


namespace NUMINAMATH_CALUDE_track_completion_time_is_80_l4016_401687

/-- Represents a runner on the circular track -/
structure Runner :=
  (id : Nat)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℕ)

/-- The circular track -/
def Track : Type := Unit

/-- Time for one runner to complete the track -/
def trackCompletionTime (track : Track) : ℕ := sorry

/-- Theorem stating the time to complete the track is 80 minutes -/
theorem track_completion_time_is_80 (track : Track) 
  (r1 r2 r3 : Runner)
  (m1 : Meeting)
  (m2 : Meeting)
  (m3 : Meeting)
  (h1 : m1.runner1 = r1 ∧ m1.runner2 = r2)
  (h2 : m2.runner1 = r2 ∧ m2.runner2 = r3)
  (h3 : m3.runner1 = r3 ∧ m3.runner2 = r1)
  (h4 : m2.time - m1.time = 15)
  (h5 : m3.time - m2.time = 25) :
  trackCompletionTime track = 80 := by sorry

end NUMINAMATH_CALUDE_track_completion_time_is_80_l4016_401687


namespace NUMINAMATH_CALUDE_parallel_line_slope_l4016_401631

/-- The slope of any line parallel to the line containing points (3, -2) and (1, 5) is -7/2 -/
theorem parallel_line_slope : ∀ (m : ℚ), 
  (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b → 
    (∃ (k : ℚ), y - (-2) = m * (x - 3) ∧ y - 5 = m * (x - 1))) → 
  m = -7/2 := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l4016_401631


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l4016_401663

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 17

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 34

theorem bowling_ball_weight_proof :
  (10 * bowling_ball_weight = 5 * canoe_weight) ∧
  (3 * canoe_weight = 102) →
  bowling_ball_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l4016_401663


namespace NUMINAMATH_CALUDE_total_fruits_three_days_l4016_401649

/-- Represents the number of fruits eaten by a dog on a given day -/
def fruits_eaten (initial : ℕ) (day : ℕ) : ℕ :=
  initial * 2^(day - 1)

/-- Represents the total fruits eaten by all dogs over a period of days -/
def total_fruits (bonnies_day1 : ℕ) (days : ℕ) : ℕ :=
  let blueberries_day1 := (3 * bonnies_day1) / 4
  let apples_day1 := 3 * blueberries_day1
  let cherries_day1 := 5 * apples_day1
  (Finset.sum (Finset.range days) (λ d => fruits_eaten bonnies_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten blueberries_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten apples_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten cherries_day1 (d + 1)))

theorem total_fruits_three_days :
  total_fruits 60 3 = 6405 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_three_days_l4016_401649


namespace NUMINAMATH_CALUDE_remainder_of_3_to_20_mod_7_l4016_401672

theorem remainder_of_3_to_20_mod_7 : 3^20 % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_of_3_to_20_mod_7_l4016_401672


namespace NUMINAMATH_CALUDE_souvenir_theorem_l4016_401674

/-- Represents the souvenirs sold at the Beijing Winter Olympics store -/
structure Souvenir where
  costA : ℕ  -- Cost price of souvenir A
  costB : ℕ  -- Cost price of souvenir B
  totalA : ℕ -- Total cost for souvenir A
  totalB : ℕ -- Total cost for souvenir B

/-- Represents the sales data for the souvenirs -/
structure SalesData where
  initPriceA : ℕ  -- Initial selling price of A
  initPriceB : ℕ  -- Initial selling price of B
  initSoldA : ℕ   -- Initial units of A sold per day
  initSoldB : ℕ   -- Initial units of B sold per day
  priceChangeA : ℤ -- Price change for A
  priceChangeB : ℤ -- Price change for B
  soldChangeA : ℕ  -- Change in units sold for A per 1 yuan price change
  soldChangeB : ℕ  -- Change in units sold for B per 1 yuan price change
  totalSold : ℕ   -- Total souvenirs sold on a certain day

/-- Theorem stating the cost prices and maximum profit -/
theorem souvenir_theorem (s : Souvenir) (d : SalesData) 
  (h1 : s.costB = s.costA + 9)
  (h2 : s.totalA = 10400)
  (h3 : s.totalB = 14000)
  (h4 : d.initPriceA = 46)
  (h5 : d.initPriceB = 45)
  (h6 : d.initSoldA = 40)
  (h7 : d.initSoldB = 80)
  (h8 : d.soldChangeA = 4)
  (h9 : d.soldChangeB = 2)
  (h10 : d.totalSold = 140) :
  s.costA = 26 ∧ s.costB = 35 ∧ 
  ∃ (profit : ℕ), profit = 2000 ∧ 
  ∀ (p : ℕ), p ≤ profit := by
    sorry

end NUMINAMATH_CALUDE_souvenir_theorem_l4016_401674


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l4016_401661

theorem unique_solution_for_equation (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  a = 12 ∧ b = 10 ∧ c = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l4016_401661


namespace NUMINAMATH_CALUDE_triangle_property_l4016_401699

theorem triangle_property (A B C : ℝ) (a b c : ℝ) :
  let triangle_exists := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  let vectors_dot_product := a * c * Real.cos B + 2 * b * c * Real.cos A = b * a * Real.cos C
  let side_angle_relation := 2 * a * Real.cos C = 2 * b - c
  triangle_exists →
  vectors_dot_product →
  side_angle_relation →
  Real.sin A / Real.sin C = Real.sqrt 2 ∧
  Real.cos B = (3 * Real.sqrt 2 - Real.sqrt 10) / 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l4016_401699


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l4016_401679

/-- The number of players on the team -/
def total_players : ℕ := 15

/-- The size of the starting lineup -/
def lineup_size : ℕ := 6

/-- The number of players guaranteed to be in the starting lineup -/
def guaranteed_players : ℕ := 3

/-- The number of remaining players to choose from -/
def remaining_players : ℕ := total_players - guaranteed_players

/-- The number of additional players needed to complete the lineup -/
def players_to_choose : ℕ := lineup_size - guaranteed_players

theorem starting_lineup_combinations :
  Nat.choose remaining_players players_to_choose = 220 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l4016_401679


namespace NUMINAMATH_CALUDE_proportion_solution_l4016_401690

theorem proportion_solution (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l4016_401690


namespace NUMINAMATH_CALUDE_set_operations_l4016_401605

def U : Set ℝ := {x | x ≤ 5}
def A : Set ℝ := {x | -2 < x ∧ x < 3}
def B : Set ℝ := {x | -4 < x ∧ x ≤ 2}

theorem set_operations :
  (A ∩ B = {x | -2 < x ∧ x ≤ 2}) ∧
  (A ∩ (U \ B) = {x | 2 < x ∧ x < 3}) ∧
  ((U \ A) ∪ B = {x | x ≤ 2 ∨ (3 ≤ x ∧ x ≤ 5)}) ∧
  ((U \ A) ∪ (U \ B) = {x | x ≤ -2 ∨ (2 < x ∧ x ≤ 5)}) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l4016_401605


namespace NUMINAMATH_CALUDE_solve_for_k_l4016_401645

theorem solve_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 8)) →
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l4016_401645


namespace NUMINAMATH_CALUDE_mark_collection_l4016_401602

/-- The amount Mark collects for the homeless -/
theorem mark_collection (households_per_day : ℕ) (days : ℕ) (giving_ratio : ℚ) (donation_amount : ℕ) : 
  households_per_day = 20 →
  days = 5 →
  giving_ratio = 1/2 →
  donation_amount = 40 →
  (households_per_day * days : ℚ) * giving_ratio * donation_amount = 2000 := by
  sorry

#check mark_collection

end NUMINAMATH_CALUDE_mark_collection_l4016_401602


namespace NUMINAMATH_CALUDE_both_colors_percentage_l4016_401669

structure FlagDistribution where
  totalFlags : ℕ
  numChildren : ℕ
  bluePercent : ℚ
  redPercent : ℚ
  hEven : Even totalFlags
  hTwoEach : numChildren = totalFlags / 2
  hAllUsed : numChildren * 2 = totalFlags
  hBluePercent : bluePercent = 60 / 100
  hRedPercent : redPercent = 65 / 100

theorem both_colors_percentage (fd : FlagDistribution) :
  (fd.bluePercent + fd.redPercent - 1 : ℚ) = 25 / 100 := by
  sorry

end NUMINAMATH_CALUDE_both_colors_percentage_l4016_401669


namespace NUMINAMATH_CALUDE_monday_temperature_l4016_401644

theorem monday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 40)
  (some_day_42 : ∃ i, temp i = 42)
  (friday_10 : temp 4 = 10) :
  temp 0 = 42 := by
sorry

end NUMINAMATH_CALUDE_monday_temperature_l4016_401644


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l4016_401673

-- Define the power function
def f (m : ℕ+) (x : ℝ) : ℝ := x^(m.val - 2)

-- Theorem statement
theorem power_function_not_through_origin (m : ℕ+) :
  (∀ x ≠ 0, f m x ≠ 0) → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l4016_401673


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l4016_401606

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : 0 ≤ r ∧ r < 1) -- Condition for convergence of geometric series
  (h2 : a / (1 - r) = 20) -- Sum of terms is 20
  (h3 : a^2 / (1 - r^2) = 80) -- Sum of squares of terms is 80
  : a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l4016_401606


namespace NUMINAMATH_CALUDE_group_messages_in_week_l4016_401684

/-- Calculates the total number of messages sent in a week by remaining members of a group -/
theorem group_messages_in_week 
  (initial_members : ℕ) 
  (removed_members : ℕ) 
  (messages_per_day : ℕ) 
  (days_in_week : ℕ) 
  (h1 : initial_members = 150) 
  (h2 : removed_members = 20) 
  (h3 : messages_per_day = 50) 
  (h4 : days_in_week = 7) :
  (initial_members - removed_members) * messages_per_day * days_in_week = 45500 :=
by sorry

end NUMINAMATH_CALUDE_group_messages_in_week_l4016_401684


namespace NUMINAMATH_CALUDE_kennel_dogs_l4016_401660

theorem kennel_dogs (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 3 / 4 →
  cats = dogs - 8 →
  dogs = 32 := by
sorry

end NUMINAMATH_CALUDE_kennel_dogs_l4016_401660


namespace NUMINAMATH_CALUDE_aisha_mp3_song_count_l4016_401630

/-- The number of songs on Aisha's mp3 player after a series of additions and removals -/
def final_song_count (initial : ℕ) (first_addition : ℕ) (removed : ℕ) : ℕ :=
  let after_first_addition := initial + first_addition
  let doubled := after_first_addition * 2
  let before_removal := after_first_addition + doubled
  before_removal - removed

/-- Theorem stating that given the initial conditions, the final number of songs is 2950 -/
theorem aisha_mp3_song_count :
  final_song_count 500 500 50 = 2950 := by
  sorry

end NUMINAMATH_CALUDE_aisha_mp3_song_count_l4016_401630


namespace NUMINAMATH_CALUDE_new_mean_after_combining_l4016_401648

theorem new_mean_after_combining (n1 n2 : ℕ) (mean1 mean2 additional : ℚ) :
  let sum1 : ℚ := n1 * mean1
  let sum2 : ℚ := n2 * mean2
  let total_sum : ℚ := sum1 + sum2 + additional
  let total_count : ℕ := n1 + n2 + 1
  (total_sum / total_count : ℚ) = (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) :=
by
  sorry

-- Example usage with the given problem values
example : 
  let n1 : ℕ := 7
  let n2 : ℕ := 9
  let mean1 : ℚ := 15
  let mean2 : ℚ := 28
  let additional : ℚ := 100
  (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) = 457 / 17 :=
by
  sorry

end NUMINAMATH_CALUDE_new_mean_after_combining_l4016_401648


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l4016_401676

theorem polynomial_functional_equation (p : ℝ → ℝ) :
  (∀ x : ℝ, p (x^3) - p (x^3 - 2) = (p x)^2 + 18) →
  (∃ a : ℝ, a^2 = 30 ∧ (∀ x : ℝ, p x = 6 * x^3 + a)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l4016_401676


namespace NUMINAMATH_CALUDE_trajectory_and_no_line_exist_l4016_401664

-- Define the points and vectors
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Define the conditions
def condition1 (B P : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  AB.1 * BP.1 + AB.2 * BP.2 = 0

def condition2 (B C P : ℝ × ℝ) : Prop :=
  (C.1 - B.1, C.2 - B.2) = (P.1 - C.1, P.2 - C.2)

def on_y_axis (B : ℝ × ℝ) : Prop := B.1 = 0
def on_x_axis (C : ℝ × ℝ) : Prop := C.2 = 0

-- Define the trajectory
def trajectory (P : ℝ × ℝ) : Prop := P.2^2 = -4 * P.1

-- Define the line passing through A
def line_through_A (k : ℝ) (x y : ℝ) : Prop := y = k * x - 8 * k

-- Define the dot product condition
def dot_product_condition (M N : ℝ × ℝ) : Prop :=
  let QM := (M.1 - Q.1, M.2 - Q.2)
  let QN := (N.1 - Q.1, N.2 - Q.2)
  QM.1 * QN.1 + QM.2 * QN.2 = 97

-- The main theorem
theorem trajectory_and_no_line_exist :
  ∀ B C P, on_y_axis B → on_x_axis C →
  condition1 B P → condition2 B C P →
  (trajectory P ∧
   ¬∃ k M N, line_through_A k M.1 M.2 ∧ line_through_A k N.1 N.2 ∧
              trajectory M ∧ trajectory N ∧ dot_product_condition M N) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_no_line_exist_l4016_401664


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4016_401637

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 6

-- Define the solution set condition
def solution_set (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

-- Theorem statement
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, f a x > 4 ↔ x ∈ solution_set a b) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, ∀ x, a * x^2 - (a * c + b) * x + b * c < 0 ↔ 1 < x ∧ x < 2 * c) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4016_401637


namespace NUMINAMATH_CALUDE_total_skips_five_throws_l4016_401613

-- Define the skip function
def S (n : ℕ) : ℕ := n^2 + n

-- Define the sum of skips from 1 to n
def sum_skips (n : ℕ) : ℕ :=
  (List.range n).map S |> List.sum

-- Theorem statement
theorem total_skips_five_throws :
  sum_skips 5 = 70 := by sorry

end NUMINAMATH_CALUDE_total_skips_five_throws_l4016_401613


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_max_q_minus_r_l4016_401646

theorem greatest_q_minus_r (q r : ℕ+) (h : 1025 = 23 * q + r) : 
  ∀ (q' r' : ℕ+), 1025 = 23 * q' + r' → q - r ≥ q' - r' :=
by
  sorry

theorem max_q_minus_r : ∃ (q r : ℕ+), 1025 = 23 * q + r ∧ q - r = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_max_q_minus_r_l4016_401646


namespace NUMINAMATH_CALUDE_gcd_lcm_calculation_l4016_401616

theorem gcd_lcm_calculation (a b : ℕ) (ha : a = 84) (hb : b = 3780) :
  (Nat.gcd a b + Nat.lcm a b) * (Nat.lcm a b * Nat.gcd a b) - 
  (Nat.lcm a b * Nat.gcd a b) = 1227194880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_calculation_l4016_401616


namespace NUMINAMATH_CALUDE_max_value_of_f_l4016_401695

-- Define the function f(x)
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -2) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≥ -2) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = 25) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l4016_401695


namespace NUMINAMATH_CALUDE_inequality_solution_l4016_401677

theorem inequality_solution (x : ℕ+) : 4 * x - 3 < 2 * x + 1 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4016_401677


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l4016_401610

-- Define the circle
def circle_radius : ℝ := 8

-- Define the diameter
def diameter : ℝ := 2 * circle_radius

-- Define the height of the triangle (equal to the radius)
def triangle_height : ℝ := circle_radius

-- Theorem statement
theorem largest_inscribed_triangle_area :
  let triangle_area := (1 / 2) * diameter * triangle_height
  triangle_area = 64 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_area_l4016_401610


namespace NUMINAMATH_CALUDE_extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l4016_401612

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- Part I
theorem extreme_values_when_a_is_4 :
  let f := f 4
  (∃ x, ∀ y, f y ≤ f x) ∧ (∃ x, ∀ y, f y ≥ f x) ∧
  (∀ x, f x ≤ 1) ∧ (∀ x, f x ≥ -1) :=
sorry

-- Part II
theorem a_range_when_f_geq_4_on_interval :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x ≥ 4) → a ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_extreme_values_when_a_is_4_a_range_when_f_geq_4_on_interval_l4016_401612


namespace NUMINAMATH_CALUDE_scientific_notation_159600_l4016_401653

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_159600 :
  toScientificNotation 159600 = ScientificNotation.mk 1.596 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_159600_l4016_401653


namespace NUMINAMATH_CALUDE_equation_solution_l4016_401693

/-- Proves that the solution to the equation (x-1)/2 - 1 = (2x+1)/3 is x = -11 -/
theorem equation_solution : ∃! x : ℚ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 ∧ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4016_401693


namespace NUMINAMATH_CALUDE_total_earnings_l4016_401668

def markese_earnings : ℕ := 16
def difference : ℕ := 5

theorem total_earnings : 
  markese_earnings + (markese_earnings + difference) = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l4016_401668


namespace NUMINAMATH_CALUDE_sum_of_composition_equals_negative_ten_l4016_401642

def p (x : ℝ) : ℝ := abs x - 2

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composition_equals_negative_ten :
  (evaluation_points.map (λ x => q (p x))).sum = -10 := by sorry

end NUMINAMATH_CALUDE_sum_of_composition_equals_negative_ten_l4016_401642


namespace NUMINAMATH_CALUDE_expression_equivalence_l4016_401681

theorem expression_equivalence (a b c : ℝ) : a - (2*b - 3*c) = a + (-2*b + 3*c) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l4016_401681


namespace NUMINAMATH_CALUDE_line_passes_through_point_three_common_tangents_implies_a_8_l4016_401643

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - 1 + m = 0

-- Define the second circle
def circle_2 (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Line l always passes through the fixed point (-1, 1)
theorem line_passes_through_point :
  ∀ m : ℝ, line_l m (-1) 1 :=
sorry

-- Theorem 2: If circle C and circle_2 have exactly three common tangents, then a = 8
theorem three_common_tangents_implies_a_8 :
  (∃! (t1 t2 t3 : ℝ × ℝ), 
    (∀ x y, circle_C x y → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                           (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                           (x - t3.1)^2 + (y - t3.2)^2 = 0) ∧
    (∀ x y, circle_2 x y a → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                              (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                              (x - t3.1)^2 + (y - t3.2)^2 = 0)) →
  a = 8 :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_point_three_common_tangents_implies_a_8_l4016_401643


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4016_401638

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 → a 1 + a 3 = 5 → a 2 + a 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4016_401638


namespace NUMINAMATH_CALUDE_solution_product_l4016_401629

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 10) = p^2 - 15 * p + 56 →
  (q - 6) * (2 * q + 10) = q^2 - 15 * q + 56 →
  p ≠ q →
  (p + 4) * (q + 4) = -40 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l4016_401629


namespace NUMINAMATH_CALUDE_product_of_second_and_third_smallest_l4016_401688

theorem product_of_second_and_third_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  (max a (max b c)) * (max (min a b) (min (max a b) c)) = 132 := by
  sorry

end NUMINAMATH_CALUDE_product_of_second_and_third_smallest_l4016_401688


namespace NUMINAMATH_CALUDE_mean_equality_problem_l4016_401650

theorem mean_equality_problem (y : ℝ) : 
  (5 + 8 + 17) / 3 = (12 + y) / 2 → y = 8 := by sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l4016_401650


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l4016_401685

theorem simplify_polynomial_expression (x : ℝ) :
  (3 * x - 4) * (x + 9) + (x + 6) * (3 * x + 2) = 6 * x^2 + 43 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l4016_401685


namespace NUMINAMATH_CALUDE_no_separable_representation_l4016_401671

theorem no_separable_representation : ¬∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), 1 + x^2016 * y^2016 = f x * g y := by
  sorry

end NUMINAMATH_CALUDE_no_separable_representation_l4016_401671
