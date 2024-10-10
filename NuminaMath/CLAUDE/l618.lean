import Mathlib

namespace salary_increase_l618_61881

/-- Represents the regression line for a worker's monthly salary based on labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem stating that an increase of 1 unit in labor productivity results in a 90 yuan increase in salary -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end salary_increase_l618_61881


namespace quadratic_decrease_interval_l618_61864

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decrease_interval (b c : ℝ) :
  f b c 1 = 0 → f b c 3 = 0 → 
  ∀ x y : ℝ, x < y → y < 2 → f b c x > f b c y := by sorry

end quadratic_decrease_interval_l618_61864


namespace determinant_equality_l618_61833

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 7 → Matrix.det !![p - 3*r, q - 3*s; r, s] = 7 := by
  sorry

end determinant_equality_l618_61833


namespace complex_modulus_from_square_l618_61890

theorem complex_modulus_from_square (z : ℂ) (h : z^2 = -48 + 64*I) : 
  Complex.abs z = 4 * Real.sqrt 5 := by
sorry

end complex_modulus_from_square_l618_61890


namespace painting_wings_count_l618_61870

/-- Represents a museum with paintings and artifacts -/
structure Museum where
  total_wings : Nat
  artifacts_per_wing : Nat
  large_painting_wings : Nat
  small_painting_wings : Nat
  paintings_per_small_wing : Nat

/-- The number of wings dedicated to paintings in the museum -/
def painting_wings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings

/-- The number of wings dedicated to artifacts in the museum -/
def artifact_wings (m : Museum) : Nat :=
  m.total_wings - painting_wings m

/-- The total number of paintings in the museum -/
def total_paintings (m : Museum) : Nat :=
  m.large_painting_wings + m.small_painting_wings * m.paintings_per_small_wing

/-- The total number of artifacts in the museum -/
def total_artifacts (m : Museum) : Nat :=
  m.artifacts_per_wing * artifact_wings m

theorem painting_wings_count (m : Museum)
  (h1 : m.total_wings = 8)
  (h2 : total_artifacts m = 4 * total_paintings m)
  (h3 : m.large_painting_wings = 1)
  (h4 : m.small_painting_wings = 2)
  (h5 : m.paintings_per_small_wing = 12)
  (h6 : m.artifacts_per_wing = 20) :
  painting_wings m = 3 := by
  sorry

end painting_wings_count_l618_61870


namespace inequality_solution_set_l618_61809

-- Define the inequality function
def f (x : ℝ) : Prop := (3 * x + 5) / (x - 1) > x

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < -1 ∨ (1 < x ∧ x < 5)

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, x ≠ 1 → (f x ↔ solution_set x) :=
by sorry

end inequality_solution_set_l618_61809


namespace product_from_gcd_lcm_l618_61879

theorem product_from_gcd_lcm (a b : ℕ+) : 
  Nat.gcd a b = 8 → Nat.lcm a b = 72 → a * b = 576 := by
  sorry

end product_from_gcd_lcm_l618_61879


namespace equality_multiplication_l618_61875

theorem equality_multiplication (a b c : ℝ) : a = b → a * c = b * c := by
  sorry

end equality_multiplication_l618_61875


namespace percentage_class_a_is_40_percent_l618_61828

/-- Represents a school with three classes -/
structure School where
  total_students : ℕ
  class_a : ℕ
  class_b : ℕ
  class_c : ℕ

/-- Calculates the percentage of students in class A -/
def percentage_class_a (s : School) : ℚ :=
  (s.class_a : ℚ) / (s.total_students : ℚ) * 100

/-- Theorem stating the percentage of students in class A -/
theorem percentage_class_a_is_40_percent (s : School) 
  (h1 : s.total_students = 80)
  (h2 : s.class_b = s.class_a - 21)
  (h3 : s.class_c = 37)
  (h4 : s.total_students = s.class_a + s.class_b + s.class_c) :
  percentage_class_a s = 40 := by
  sorry

#eval percentage_class_a {
  total_students := 80,
  class_a := 32,
  class_b := 11,
  class_c := 37
}

end percentage_class_a_is_40_percent_l618_61828


namespace cosine_equality_l618_61807

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cosine_equality_l618_61807


namespace lending_amount_calculation_l618_61830

theorem lending_amount_calculation (P : ℝ) 
  (h1 : (P * 0.115 * 3) - (P * 0.10 * 3) = 157.5) : P = 3500 := by
  sorry

end lending_amount_calculation_l618_61830


namespace tangent_line_at_point_l618_61811

/-- The circle with equation x²+(y-1)²=2 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 2}

/-- The point (1,2) on the circle -/
def Point : ℝ × ℝ := (1, 2)

/-- The proposed tangent line with equation x+y-3=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 3 = 0}

theorem tangent_line_at_point :
  Point ∈ Circle ∧
  Point ∈ TangentLine ∧
  ∀ p ∈ Circle, p ≠ Point → p ∉ TangentLine :=
by sorry

end tangent_line_at_point_l618_61811


namespace total_ants_count_l618_61822

/-- The total number of ants employed for all tasks in the construction site. -/
def total_ants : ℕ :=
  let red_carrying := 413
  let black_carrying := 487
  let yellow_carrying := 360
  let red_digging := 356
  let black_digging := 518
  let green_digging := 250
  let red_assembling := 298
  let black_assembling := 392
  let blue_assembling := 200
  let black_food := black_carrying / 4
  red_carrying + black_carrying + yellow_carrying +
  red_digging + black_digging + green_digging +
  red_assembling + black_assembling + blue_assembling -
  black_food

/-- Theorem stating that the total number of ants employed for all tasks is 3153. -/
theorem total_ants_count : total_ants = 3153 := by
  sorry

end total_ants_count_l618_61822


namespace power_equation_solution_l618_61806

def solution_set : Set ℝ := {-3, 1, 2}

theorem power_equation_solution (x : ℝ) : 
  (2*x - 3)^(x + 3) = 1 ↔ x ∈ solution_set :=
sorry

end power_equation_solution_l618_61806


namespace sequence_length_l618_61832

/-- Proves that an arithmetic sequence starting at 2.5, ending at 67.5, with a common difference of 5, has 14 terms. -/
theorem sequence_length : 
  ∀ (a : ℚ) (d : ℚ) (last : ℚ) (n : ℕ),
  a = 2.5 ∧ d = 5 ∧ last = 67.5 →
  last = a + (n - 1) * d →
  n = 14 := by
sorry

end sequence_length_l618_61832


namespace paint_cans_used_l618_61895

theorem paint_cans_used (initial_capacity : ℕ) (lost_cans : ℕ) (remaining_capacity : ℕ) : 
  initial_capacity = 40 → 
  lost_cans = 4 → 
  remaining_capacity = 30 → 
  ∃ (cans_per_room : ℚ), 
    cans_per_room > 0 ∧
    initial_capacity = (initial_capacity - remaining_capacity) / lost_cans * lost_cans + remaining_capacity ∧
    (initial_capacity : ℚ) / cans_per_room - lost_cans = remaining_capacity / cans_per_room ∧
    remaining_capacity / cans_per_room = 12 := by
  sorry

end paint_cans_used_l618_61895


namespace solution_set_part1_range_of_a_l618_61842

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part 1
theorem solution_set_part1 :
  ∀ x : ℝ, f (-4) x ≥ 6 ↔ x ≤ 0 ∨ x ≥ 6 :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 3|) → a ∈ Set.Icc (-1) 0 :=
sorry

end solution_set_part1_range_of_a_l618_61842


namespace self_reciprocal_set_l618_61801

def self_reciprocal (x : ℝ) : Prop := x ≠ 0 ∧ x = 1 / x

theorem self_reciprocal_set :
  ∃ (S : Set ℝ), (∀ x, x ∈ S ↔ self_reciprocal x) ∧ S = {1, -1} :=
sorry

end self_reciprocal_set_l618_61801


namespace range_of_a_l618_61812

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) 
  (h1 : ∀ x, p x → q x)
  (h2 : ∃ x, q x ∧ ¬(p x))
  (hp : ∀ x, p x ↔ x^2 - 2*x - 3 < 0)
  (hq : ∀ x, q x ↔ x > a) :
  a ≤ -1 := by
sorry

end range_of_a_l618_61812


namespace rubber_bands_distribution_l618_61880

/-- The number of rubber bands Aira had -/
def aira_bands : ℕ := sorry

/-- The number of rubber bands Samantha had -/
def samantha_bands : ℕ := sorry

/-- The number of rubber bands Joe had -/
def joe_bands : ℕ := sorry

/-- The total number of rubber bands -/
def total_bands : ℕ := sorry

theorem rubber_bands_distribution :
  -- Condition 1 and 2: Equal division resulting in 6 bands each
  total_bands = 3 * 6 ∧
  -- Condition 3: Samantha had 5 more bands than Aira
  samantha_bands = aira_bands + 5 ∧
  -- Condition 4: Aira had 1 fewer band than Joe
  aira_bands + 1 = joe_bands ∧
  -- Total bands is the sum of all individual bands
  total_bands = aira_bands + samantha_bands + joe_bands →
  -- Conclusion: Aira had 4 rubber bands
  aira_bands = 4 := by
sorry

end rubber_bands_distribution_l618_61880


namespace total_friends_l618_61843

theorem total_friends (initial_friends additional_friends : ℕ) 
  (h1 : initial_friends = 4) 
  (h2 : additional_friends = 3) : 
  initial_friends + additional_friends = 7 := by
    sorry

end total_friends_l618_61843


namespace cake_shop_work_duration_l618_61800

/-- Calculates the number of months worked given the total hours worked by Cathy -/
def months_worked (total_hours : ℕ) : ℚ :=
  let hours_per_week : ℕ := 20
  let weeks_per_month : ℕ := 4
  let extra_hours : ℕ := 20
  let regular_hours : ℕ := total_hours - extra_hours
  let regular_weeks : ℚ := regular_hours / hours_per_week
  regular_weeks / weeks_per_month

theorem cake_shop_work_duration :
  months_worked 180 = 2 := by
  sorry

end cake_shop_work_duration_l618_61800


namespace inverse_proportion_ratio_l618_61853

theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (h_nonzero : x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ y₁ ≠ 0 ∧ y₂ ≠ 0) 
  (h_inverse : ∃ k : ℝ, x₁ * y₁ = k ∧ x₂ * y₂ = k) (h_ratio : x₁ / x₂ = 3 / 4) : 
  y₁ / y₂ = 4 / 3 := by
  sorry

end inverse_proportion_ratio_l618_61853


namespace sequence_sum_properties_l618_61802

/-- Defines the sequence where a_1 = 1 and between the k-th 1 and the (k+1)-th 1, there are 2^(k-1) terms of 2 -/
def a : ℕ → ℕ :=
  sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℕ :=
  sorry

theorem sequence_sum_properties :
  (S 1998 = 3985) ∧ (∀ n : ℕ, S n ≠ 2001) := by
  sorry

end sequence_sum_properties_l618_61802


namespace a_zero_necessary_not_sufficient_l618_61850

def is_pure_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = (0 : ℝ) + b * Complex.I

theorem a_zero_necessary_not_sufficient :
  (∀ a b : ℝ, is_pure_imaginary (Complex.ofReal a + Complex.I * b) → a = 0) ∧
  ¬(∀ a b : ℝ, a = 0 → is_pure_imaginary (Complex.ofReal a + Complex.I * b)) :=
by sorry

end a_zero_necessary_not_sufficient_l618_61850


namespace triangle_point_C_l618_61820

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

def isMedian (t : Triangle) (M : Point) : Prop :=
  M.x = (t.A.x + t.B.x) / 2 ∧ M.y = (t.A.y + t.B.y) / 2

def isAngleBisector (t : Triangle) (L : Point) : Prop :=
  -- We can't define this precisely without more geometric functions,
  -- so we'll leave it as an axiom for now
  True

theorem triangle_point_C (t : Triangle) (M L : Point) :
  t.A = Point.mk 2 8 →
  M = Point.mk 4 11 →
  L = Point.mk 6 6 →
  isMedian t M →
  isAngleBisector t L →
  t.C = Point.mk 14 2 := by
  sorry


end triangle_point_C_l618_61820


namespace business_profit_l618_61813

theorem business_profit (total_profit : ℝ) : 
  (0.25 * total_profit) + 2 * (0.25 * (0.75 * total_profit)) = 50000 →
  total_profit = 80000 := by
sorry

end business_profit_l618_61813


namespace extreme_value_implies_parameters_l618_61845

/-- The function f with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- Theorem stating that if f has an extreme value of 10 at x=1, then (a,b) = (-4,11) -/
theorem extreme_value_implies_parameters
  (a b : ℝ)
  (extreme_value : f a b 1 = 10)
  (is_extreme : ∀ x, f a b x ≤ f a b 1) :
  a = -4 ∧ b = 11 := by
sorry

end extreme_value_implies_parameters_l618_61845


namespace complex_number_properties_l618_61876

def z : ℂ := (1 - Complex.I)^2 + 1 + 3 * Complex.I

theorem complex_number_properties :
  (z = 3 + 3 * Complex.I) ∧
  (Complex.abs z = 3 * Real.sqrt 2) ∧
  (∃ (a b : ℝ), z^2 + a * z + b = 1 - Complex.I ∧ a = -6 ∧ b = 10) := by
  sorry

end complex_number_properties_l618_61876


namespace revenue_maximized_at_five_l618_61869

def revenue (x : ℝ) : ℝ := (400 - 20*x) * (50 + 5*x)

theorem revenue_maximized_at_five :
  ∃ (max : ℝ), revenue 5 = max ∧ ∀ (x : ℝ), revenue x ≤ max :=
by sorry

end revenue_maximized_at_five_l618_61869


namespace newspaper_pieces_not_all_found_l618_61847

theorem newspaper_pieces_not_all_found :
  ¬∃ (k p v : ℕ), 1988 = k + 4 * p + 8 * v ∧ k > 0 := by
  sorry

end newspaper_pieces_not_all_found_l618_61847


namespace tangent_line_property_l618_61878

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def hasTangentAt (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

def tangentLineEquation (f : ℝ → ℝ) : Prop :=
  ∃ (y : ℝ), 2 + 2 * y + 1 = 0 ∧ f 2 = y

-- State the theorem
theorem tangent_line_property (f : ℝ → ℝ) 
  (h1 : hasTangentAt f) 
  (h2 : tangentLineEquation f) : 
  f 2 - 2 * (deriv f 2) = -1/2 := by
sorry

end tangent_line_property_l618_61878


namespace five_line_configurations_l618_61846

/-- Represents a configuration of five lines in a plane -/
structure LineConfiguration where
  /-- The number of intersection points -/
  intersections : ℕ
  /-- The number of sets of parallel lines -/
  parallel_sets : ℕ

/-- The total count is the sum of intersection points and parallel sets -/
def total_count (config : LineConfiguration) : ℕ :=
  config.intersections + config.parallel_sets

/-- Possible configurations of five lines in a plane -/
def possible_configurations : List LineConfiguration :=
  [
    ⟨0, 1⟩,  -- All 5 lines parallel
    ⟨4, 1⟩,  -- 4 parallel lines and 1 intersecting
    ⟨6, 2⟩,  -- Two sets of parallel lines (2 and 3)
    ⟨7, 1⟩,  -- 3 parallel lines and 2 intersecting
    ⟨8, 2⟩,  -- Two pairs of parallel lines
    ⟨9, 1⟩,  -- 1 pair of parallel lines
    ⟨10, 0⟩  -- No parallel lines
  ]

theorem five_line_configurations :
  (possible_configurations.map total_count).toFinset = {1, 5, 8, 10} := by sorry

end five_line_configurations_l618_61846


namespace M_mod_100_l618_61840

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def trailingZeros (n : ℕ) : ℕ :=
  let rec count_fives (m : ℕ) (acc : ℕ) : ℕ :=
    if m < 5 then acc
    else count_fives (m / 5) (acc + m / 5)
  count_fives n 0

def M : ℕ := trailingZeros (factorial 50)

theorem M_mod_100 : M % 100 = 12 := by sorry

end M_mod_100_l618_61840


namespace product_no_linear_quadratic_terms_l618_61896

theorem product_no_linear_quadratic_terms 
  (p q : ℚ) 
  (h : ∀ x : ℚ, (x + 3*p) * (x^2 - x + 1/3*q) = x^3 + p*q) : 
  p = 1/3 ∧ q = 3 ∧ p^2020 * q^2021 = 3 := by
  sorry

end product_no_linear_quadratic_terms_l618_61896


namespace circle_increase_l618_61819

theorem circle_increase (r : ℝ) (hr : r > 0) :
  let new_radius := 2.5 * r
  let area_increase_percent := ((π * new_radius^2 - π * r^2) / (π * r^2)) * 100
  let circumference_increase_percent := ((2 * π * new_radius - 2 * π * r) / (2 * π * r)) * 100
  area_increase_percent = 525 ∧ circumference_increase_percent = 150 := by
sorry


end circle_increase_l618_61819


namespace complex_product_quadrant_l618_61891

theorem complex_product_quadrant : 
  let z : ℂ := (1 + 3*I) * (3 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end complex_product_quadrant_l618_61891


namespace other_asymptote_equation_l618_61803

/-- A hyperbola with given asymptote and foci x-coordinate -/
structure Hyperbola where
  asymptote : ℝ → ℝ
  foci_x : ℝ
  asymptote_eq : asymptote = fun x ↦ 2 * x + 3
  foci_x_eq : foci_x = 7

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 31

/-- Theorem stating that the other asymptote has the correct equation -/
theorem other_asymptote_equation (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 31 := by
  sorry


end other_asymptote_equation_l618_61803


namespace triangle_area_maximum_l618_61858

/-- The area of a triangle with two fixed sides is maximized when the angle between these sides is 90°. -/
theorem triangle_area_maximum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧
    ∀ φ : ℝ, φ ∈ Set.Icc 0 π →
      (1 / 2) * a * b * Real.sin θ ≥ (1 / 2) * a * b * Real.sin φ :=
  sorry

end triangle_area_maximum_l618_61858


namespace remainder_444_power_444_mod_13_l618_61838

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l618_61838


namespace cube_volume_surface_area_l618_61839

theorem cube_volume_surface_area (x : ℝ) :
  (∃ s : ℝ, s > 0 ∧ s^3 = 27*x ∧ 6*s^2 = 3*x) → x = 5832 := by
  sorry

end cube_volume_surface_area_l618_61839


namespace min_segments_to_return_l618_61834

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    and the measure of angle ABC is 80 degrees, prove that the minimum number of segments
    needed to return to the starting point is 18. -/
theorem min_segments_to_return (m_angle_ABC : ℝ) (n : ℕ) : 
  m_angle_ABC = 80 → 
  (∀ m : ℕ, 100 * n = 360 * m) → 
  n ≥ 18 ∧ 
  (∀ k < n, ¬(∀ m : ℕ, 100 * k = 360 * m)) := by
  sorry

end min_segments_to_return_l618_61834


namespace sequence_problem_l618_61851

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a r : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => geometricSequence a r n * r

theorem sequence_problem :
  (arithmeticSequence 12 4 3 = 24) ∧
  (arithmeticSequence 12 4 4 = 28) ∧
  (geometricSequence 2 2 3 = 16) ∧
  (geometricSequence 2 2 4 = 32) := by
  sorry

end sequence_problem_l618_61851


namespace polynomial_expansion_l618_61814

theorem polynomial_expansion (z : ℝ) :
  (3 * z^3 + 2 * z^2 - 4 * z + 1) * (4 * z^4 - 3 * z^2 + 2) =
  12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2 := by
  sorry

end polynomial_expansion_l618_61814


namespace repeating_decimal_equals_fraction_l618_61877

/-- The value of the repeating decimal 0.565656... -/
def repeating_decimal : ℚ := 0.56565656

/-- The fraction 56/99 -/
def fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l618_61877


namespace tens_digit_of_2015_pow_2016_minus_2017_l618_61885

theorem tens_digit_of_2015_pow_2016_minus_2017 :
  (2015^2016 - 2017) % 100 / 10 = 0 :=
by sorry

end tens_digit_of_2015_pow_2016_minus_2017_l618_61885


namespace gummy_worms_problem_l618_61805

theorem gummy_worms_problem (initial_amount : ℕ) : 
  (((initial_amount / 2) / 2) / 2) / 2 = 4 → initial_amount = 64 :=
by
  sorry

end gummy_worms_problem_l618_61805


namespace elementary_symmetric_polynomials_l618_61859

variable (x y z : ℝ)

/-- Elementary symmetric polynomial of degree 1 -/
def σ₁ (x y z : ℝ) : ℝ := x + y + z

/-- Elementary symmetric polynomial of degree 2 -/
def σ₂ (x y z : ℝ) : ℝ := x*y + y*z + z*x

/-- Elementary symmetric polynomial of degree 3 -/
def σ₃ (x y z : ℝ) : ℝ := x*y*z

theorem elementary_symmetric_polynomials (x y z : ℝ) :
  ((x + y) * (y + z) * (x + z) = σ₂ x y z * σ₁ x y z - σ₃ x y z) ∧
  (x^3 + y^3 + z^3 - 3*x*y*z = σ₁ x y z * (σ₁ x y z^2 - 3 * σ₂ x y z)) ∧
  (x^3 + y^3 = σ₁ x y 0^3 - 3 * σ₁ x y 0 * σ₂ x y 0) ∧
  ((x^2 + y^2) * (y^2 + z^2) * (x^2 + z^2) = 
    σ₁ x y z^2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₂ x y z * σ₃ x y z - 
    2 * σ₂ x y z^3 - 2 * σ₁ x y z^3 * σ₃ x y z - σ₃ x y z^2) ∧
  (x^4 + y^4 + z^4 = 
    σ₁ x y z^4 - 4 * σ₁ x y z^2 * σ₂ x y z + 2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₃ x y z) :=
by sorry

end elementary_symmetric_polynomials_l618_61859


namespace oil_temperature_increase_rate_l618_61865

def oil_temperature (t : ℕ) : ℝ :=
  if t = 0 then 10
  else if t = 10 then 30
  else if t = 20 then 50
  else if t = 30 then 70
  else if t = 40 then 90
  else 0  -- undefined for other values

theorem oil_temperature_increase_rate :
  ∀ t : ℕ, t < 40 →
    oil_temperature (t + 10) - oil_temperature t = 20 :=
sorry

end oil_temperature_increase_rate_l618_61865


namespace absent_student_percentage_l618_61808

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : total_students = boys + girls)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h5 : boys_absent_fraction = 1 / 7)
  (h6 : girls_absent_fraction = 1 / 5) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 1 / 6 := by
  sorry

end absent_student_percentage_l618_61808


namespace drama_club_revenue_l618_61884

theorem drama_club_revenue : 
  let total_tickets : ℕ := 1500
  let adult_price : ℕ := 12
  let student_price : ℕ := 6
  let student_tickets : ℕ := 300
  let adult_tickets : ℕ := total_tickets - student_tickets
  let total_revenue : ℕ := adult_tickets * adult_price + student_tickets * student_price
  total_revenue = 16200 := by
sorry

end drama_club_revenue_l618_61884


namespace next_number_is_1461_l618_61899

/-- Represents the sequence generator function -/
def sequenceGenerator (n : ℕ) : ℕ := 
  100 + 15 + (n * (n + 1))

/-- Proves that the next number after 1445 in the sequence is 1461 -/
theorem next_number_is_1461 : 
  ∃ k, sequenceGenerator k = 1445 ∧ sequenceGenerator (k + 1) = 1461 :=
sorry

end next_number_is_1461_l618_61899


namespace ordering_of_constants_l618_61860

theorem ordering_of_constants : 
  Real.log 17 < 3 ∧ 3 < Real.exp (Real.sqrt 2) := by sorry

end ordering_of_constants_l618_61860


namespace sons_age_l618_61816

theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  (father_age - 8) = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end sons_age_l618_61816


namespace average_age_combined_rooms_l618_61825

theorem average_age_combined_rooms (room_a_count room_b_count room_c_count : ℕ)
                                   (room_a_avg room_b_avg room_c_avg : ℝ)
                                   (h1 : room_a_count = 8)
                                   (h2 : room_b_count = 5)
                                   (h3 : room_c_count = 7)
                                   (h4 : room_a_avg = 35)
                                   (h5 : room_b_avg = 30)
                                   (h6 : room_c_avg = 50) :
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  total_age / total_count = 39 := by
sorry

end average_age_combined_rooms_l618_61825


namespace pool_length_l618_61841

theorem pool_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → 
  area = 30 → 
  area = length * width → 
  length = 10 := by
sorry

end pool_length_l618_61841


namespace train_speed_l618_61835

/-- The speed of a train given its length and time to cross a point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 800) (h2 : time = 10) :
  length / time = 80 := by
  sorry

end train_speed_l618_61835


namespace disjoint_sets_property_l618_61897

theorem disjoint_sets_property (A B : Set ℕ) (h1 : A ∩ B = ∅) (h2 : A ∪ B = Set.univ) :
  ∀ n : ℕ, ∃ a b : ℕ, a ≠ b ∧ a > n ∧ b > n ∧
    (({a, b, a + b} : Set ℕ) ⊆ A ∨ ({a, b, a + b} : Set ℕ) ⊆ B) :=
by sorry

end disjoint_sets_property_l618_61897


namespace distinct_sums_largest_value_l618_61862

theorem distinct_sums_largest_value (A B C D : ℕ) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A + C ≠ B + C ∧ A + C ≠ B + D ∧ A + C ≠ D + A ∧
   B + C ≠ B + D ∧ B + C ≠ D + A ∧
   B + D ≠ D + A) →
  ({A, B, C, D, A + C, B + C, B + D, D + A} : Finset ℕ) = {1, 2, 3, 4, 5, 6, 7, 8} →
  A > B ∧ A > C ∧ A > D →
  A = 12 := by
sorry

end distinct_sums_largest_value_l618_61862


namespace y_percentage_more_than_z_l618_61854

/-- Proves that given the conditions, y gets 20% more than z -/
theorem y_percentage_more_than_z (total : ℝ) (z_share : ℝ) (x_more_than_y : ℝ) :
  total = 1480 →
  z_share = 400 →
  x_more_than_y = 0.25 →
  (((total - z_share) / (2 + x_more_than_y) - z_share) / z_share) * 100 = 20 := by
  sorry

end y_percentage_more_than_z_l618_61854


namespace bank_robbery_culprits_l618_61892

theorem bank_robbery_culprits (Alexey Boris Veniamin Grigory : Prop) :
  (¬Grigory → Boris ∧ ¬Alexey) →
  (Veniamin → ¬Alexey ∧ ¬Boris) →
  (Grigory → Boris) →
  (Boris → Alexey ∨ Veniamin) →
  (Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin) :=
by sorry

end bank_robbery_culprits_l618_61892


namespace calculate_expression_solve_system_of_equations_l618_61836

-- Problem 1
theorem calculate_expression : (-3)^2 - 3^0 + (-2) = 6 := by sorry

-- Problem 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x - y = 3 ∧ x + y = 6 ∧ x = 3 ∧ y = 3 := by sorry

end calculate_expression_solve_system_of_equations_l618_61836


namespace division_addition_equality_l618_61810

theorem division_addition_equality : 0.2 / 0.005 + 0.1 = 40.1 := by
  sorry

end division_addition_equality_l618_61810


namespace circle_area_to_circumference_l618_61874

theorem circle_area_to_circumference (A : ℝ) (h : A = 196 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * Real.pi * r = 28 * Real.pi :=
sorry

end circle_area_to_circumference_l618_61874


namespace round_trip_speed_calculation_l618_61804

/-- Proves that given specific conditions for a round trip, the return speed must be 45 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_there : ℝ) (avg_speed : ℝ) :
  distance = 180 →
  speed_there = 90 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_there + distance / (2 * avg_speed - speed_there)) = avg_speed →
  2 * avg_speed - speed_there = 45 := by
  sorry

end round_trip_speed_calculation_l618_61804


namespace triangle_side_length_l618_61883

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (area : ℝ) :
  b = 3 →
  c = 4 →
  area = 3 * Real.sqrt 3 →
  area = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  (a = Real.sqrt 13 ∨ a = Real.sqrt 37) :=
by sorry

end triangle_side_length_l618_61883


namespace max_value_of_c_max_value_of_c_achieved_l618_61863

theorem max_value_of_c (x : ℝ) (c : ℝ) (h1 : x > 1) (h2 : c = 2 - x + 2 * Real.sqrt (x - 1)) :
  c ≤ 2 :=
by sorry

theorem max_value_of_c_achieved (x : ℝ) :
  ∃ c, x > 1 ∧ c = 2 - x + 2 * Real.sqrt (x - 1) ∧ c = 2 :=
by sorry

end max_value_of_c_max_value_of_c_achieved_l618_61863


namespace solution_set_quadratic_inequality_l618_61826

/-- The solution set of the inequality -x^2 - x + 6 > 0 is the open interval (-3, 2) -/
theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3) 2 := by
  sorry

end solution_set_quadratic_inequality_l618_61826


namespace coefficient_of_n_l618_61857

theorem coefficient_of_n (n : ℤ) : 
  (∃ (values : Finset ℤ), 
    (∀ m ∈ values, 1 < 4 * m + 7 ∧ 4 * m + 7 < 40) ∧ 
    Finset.card values = 10) → 
  (∃ k : ℤ, ∀ m : ℤ, 4 * m + 7 = k * m + 7 → k = 4) :=
sorry

end coefficient_of_n_l618_61857


namespace weaving_increase_proof_l618_61887

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16/29

/-- The amount woven on the first day -/
def first_day_output : ℚ := 5

/-- The number of days -/
def num_days : ℕ := 30

/-- The total amount woven in 30 days -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  first_day_output * num_days + (num_days * (num_days - 1) / 2) * daily_increase = total_output :=
sorry

end weaving_increase_proof_l618_61887


namespace curve_E_equation_line_l_equation_l618_61882

/-- The curve E is defined by the constant sum of distances to two fixed points -/
def CurveE (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

/-- The line l passes through (0, -2) and intersects curve E at points C and D -/
def LineL (l : ℝ → ℝ) (C D : ℝ × ℝ) : Prop :=
  l 0 = -2 ∧ CurveE C ∧ CurveE D ∧ C.2 = l C.1 ∧ D.2 = l D.1

/-- The dot product of OC and OD is zero -/
def OrthogonalIntersection (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem curve_E_equation (P : ℝ × ℝ) (h : CurveE P) :
  P.1^2 / 4 + P.2^2 = 1 :=
sorry

theorem line_l_equation (l : ℝ → ℝ) (C D : ℝ × ℝ)
  (hl : LineL l C D) (horth : OrthogonalIntersection C D) :
  (∀ x, l x = 2*x - 2) ∨ (∀ x, l x = -2*x - 2) :=
sorry

end curve_E_equation_line_l_equation_l618_61882


namespace necessary_condition_inequality_l618_61829

theorem necessary_condition_inequality (a b c : ℝ) (hc : c ≠ 0) :
  (∀ a b c, c ≠ 0 → (a * c^2 > b * c^2 → a > b)) :=
by sorry

end necessary_condition_inequality_l618_61829


namespace inverse_proportion_k_negative_l618_61861

theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (h1 : k ≠ 0)
  (h2 : y₁ = k / (-2))
  (h3 : y₂ = k / 5)
  (h4 : y₁ > y₂) :
  k < 0 := by
sorry

end inverse_proportion_k_negative_l618_61861


namespace exponent_division_l618_61873

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end exponent_division_l618_61873


namespace tim_income_percentage_l618_61866

/-- Proves that Tim's income is 60% less than Juan's income given the conditions --/
theorem tim_income_percentage (tim mart juan : ℝ) 
  (h1 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  (h2 : mart = 0.64 * juan)  -- Mart's income is 64% of Juan's
  : tim = 0.4 * juan :=  -- Tim's income is 40% of Juan's (equivalent to 60% less)
by
  sorry

#check tim_income_percentage

end tim_income_percentage_l618_61866


namespace sandys_shopping_money_l618_61852

theorem sandys_shopping_money (watch_price : ℝ) (money_left : ℝ) (spent_percentage : ℝ) : 
  watch_price = 50 →
  money_left = 210 →
  spent_percentage = 0.3 →
  ∃ (total_money : ℝ), 
    total_money = watch_price + (money_left / (1 - spent_percentage)) ∧
    total_money = 350 :=
by sorry

end sandys_shopping_money_l618_61852


namespace grape_juice_percentage_l618_61815

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice -/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_pure_juice : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_concentration = 0.1)
  (h3 : added_pure_juice = 20)
  : (initial_volume * initial_concentration + added_pure_juice) / (initial_volume + added_pure_juice) = 0.4 := by
  sorry

#check grape_juice_percentage

end grape_juice_percentage_l618_61815


namespace scaling_factor_of_similar_cubes_l618_61867

theorem scaling_factor_of_similar_cubes (v1 v2 : ℝ) (h1 : v1 = 343) (h2 : v2 = 2744) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end scaling_factor_of_similar_cubes_l618_61867


namespace m_range_l618_61888

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem m_range (m : ℝ) : 
  (m > 0) → 
  (∀ x, ¬(p x) → ¬(q x m)) → 
  (∃ x, ¬(p x) ∧ (q x m)) → 
  m ≥ 9 :=
sorry

end m_range_l618_61888


namespace binary_1010101_equals_85_l618_61871

def binaryToDecimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1010101_equals_85 :
  binaryToDecimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end binary_1010101_equals_85_l618_61871


namespace height_prediction_at_10_l618_61849

/-- Represents a linear regression model for height vs age -/
structure HeightModel where
  slope : ℝ
  intercept : ℝ

/-- Calculates the predicted height for a given age using the model -/
def predictHeight (model : HeightModel) (age : ℝ) : ℝ :=
  model.slope * age + model.intercept

/-- Defines what it means for a prediction to be "around" a value -/
def isAround (predicted : ℝ) (target : ℝ) (tolerance : ℝ) : Prop :=
  abs (predicted - target) ≤ tolerance

theorem height_prediction_at_10 (model : HeightModel) 
  (h1 : model.slope = 7.19) 
  (h2 : model.intercept = 73.93) : 
  ∃ (tolerance : ℝ), tolerance > 0 ∧ isAround (predictHeight model 10) 145.83 tolerance :=
sorry

end height_prediction_at_10_l618_61849


namespace legal_fee_participants_l618_61824

/-- The number of participants paying legal fees -/
def num_participants : ℕ := 8

/-- The total legal costs in francs -/
def total_cost : ℕ := 800

/-- The number of participants who cannot pay -/
def non_paying_participants : ℕ := 3

/-- The additional amount each paying participant contributes in francs -/
def additional_payment : ℕ := 60

/-- Theorem stating that the number of participants satisfies the given conditions -/
theorem legal_fee_participants :
  (total_cost : ℚ) / num_participants + additional_payment = 
  total_cost / (num_participants - non_paying_participants) :=
by sorry

end legal_fee_participants_l618_61824


namespace scientific_notation_of_rural_population_l618_61831

theorem scientific_notation_of_rural_population :
  ∃ (x : ℝ), x = 42.39 * 10^6 ∧ x = 4.239 * 10^7 := by
  sorry

end scientific_notation_of_rural_population_l618_61831


namespace a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l618_61868

theorem a_less_than_one_necessary_not_sufficient_for_ln_a_negative :
  (∀ a : ℝ, (Real.log a < 0) → (a < 1)) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0)) :=
sorry

end a_less_than_one_necessary_not_sufficient_for_ln_a_negative_l618_61868


namespace impossible_sum_240_l618_61886

theorem impossible_sum_240 : ¬ ∃ (a b c d e f g h i : ℕ), 
  (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (10 ≤ c ∧ c ≤ 99) ∧
  (10 ≤ d ∧ d ≤ 99) ∧ (10 ≤ e ∧ e ≤ 99) ∧ (10 ≤ f ∧ f ≤ 99) ∧
  (10 ≤ g ∧ g ≤ 99) ∧ (10 ≤ h ∧ h ≤ 99) ∧ (10 ≤ i ∧ i ≤ 99) ∧
  (a % 10 = 9 ∨ a / 10 = 9) ∧ (b % 10 = 9 ∨ b / 10 = 9) ∧
  (c % 10 = 9 ∨ c / 10 = 9) ∧ (d % 10 = 9 ∨ d / 10 = 9) ∧
  (e % 10 = 9 ∨ e / 10 = 9) ∧ (f % 10 = 9 ∨ f / 10 = 9) ∧
  (g % 10 = 9 ∨ g / 10 = 9) ∧ (h % 10 = 9 ∨ h / 10 = 9) ∧
  (i % 10 = 9 ∨ i / 10 = 9) ∧
  a + b + c + d + e + f + g + h + i = 240 :=
by sorry

end impossible_sum_240_l618_61886


namespace lisa_to_total_ratio_l618_61856

def total_earnings : ℝ := 60

def lisa_earnings (l : ℝ) : Prop := 
  ∃ (j t : ℝ), l + j + t = total_earnings ∧ t = l / 2 ∧ l = t + 15

theorem lisa_to_total_ratio : 
  ∀ l : ℝ, lisa_earnings l → l / total_earnings = 1 / 2 := by sorry

end lisa_to_total_ratio_l618_61856


namespace geometric_sequence_range_l618_61898

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 3) →
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧ a₄ = a₁ * q^3) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 9) :=
by sorry

end geometric_sequence_range_l618_61898


namespace vector_expression_simplification_l618_61855

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_simplification (a b : V) :
  (2/3 : ℝ) • ((4 • a - 3 • b) + (1/3 : ℝ) • b - (1/4 : ℝ) • (6 • a - 7 • b)) =
  (5/3 : ℝ) • a - (11/18 : ℝ) • b := by sorry

end vector_expression_simplification_l618_61855


namespace min_basketballs_is_two_l618_61827

/-- Represents the number of items sold for each type of sporting good. -/
structure ItemsSold where
  frisbees : ℕ
  baseballs : ℕ
  basketballs : ℕ

/-- Checks if the given ItemsSold satisfies all conditions of the problem. -/
def satisfiesConditions (items : ItemsSold) : Prop :=
  items.frisbees + items.baseballs + items.basketballs = 180 ∧
  3 * items.frisbees + 5 * items.baseballs + 10 * items.basketballs = 800 ∧
  items.frisbees > items.baseballs ∧
  items.baseballs > items.basketballs

/-- The minimum number of basketballs that could have been sold. -/
def minBasketballs : ℕ := 2

/-- Theorem stating that the minimum number of basketballs sold is 2. -/
theorem min_basketballs_is_two :
  ∀ items : ItemsSold,
    satisfiesConditions items →
    items.basketballs ≥ minBasketballs :=
by
  sorry

#check min_basketballs_is_two

end min_basketballs_is_two_l618_61827


namespace solve_parking_lot_l618_61848

def parking_lot (num_bikes : ℕ) (total_wheels : ℕ) (wheels_per_car : ℕ) (wheels_per_bike : ℕ) : Prop :=
  ∃ (num_cars : ℕ), 
    num_cars * wheels_per_car + num_bikes * wheels_per_bike = total_wheels

theorem solve_parking_lot : 
  parking_lot 5 66 4 2 → ∃ (num_cars : ℕ), num_cars = 14 := by
  sorry

end solve_parking_lot_l618_61848


namespace probability_of_red_ball_l618_61817

theorem probability_of_red_ball (basketA_white basketA_red basketB_yellow basketB_red basketB_black : ℕ)
  (probA probB : ℝ) : 
  basketA_white = 10 →
  basketA_red = 5 →
  basketB_yellow = 4 →
  basketB_red = 6 →
  basketB_black = 5 →
  probA = 0.6 →
  probB = 0.4 →
  (basketA_red / (basketA_white + basketA_red : ℝ)) * probA +
  (basketB_red / (basketB_yellow + basketB_red + basketB_black : ℝ)) * probB = 0.36 :=
by sorry

end probability_of_red_ball_l618_61817


namespace absolute_value_calculation_l618_61821

theorem absolute_value_calculation : |-6| - (-4) + (-7) = 3 := by
  sorry

end absolute_value_calculation_l618_61821


namespace number_of_black_balls_l618_61894

/-- Given a bag with red, white, and black balls, prove the number of black balls -/
theorem number_of_black_balls
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (prob_red : ℚ)
  (prob_white : ℚ)
  (h1 : red_balls = 21)
  (h2 : prob_red = 21 / total_balls)
  (h3 : prob_white = white_balls / total_balls)
  (h4 : prob_red = 42 / 100)
  (h5 : prob_white = 28 / 100)
  (h6 : total_balls = red_balls + white_balls + black_balls) :
  black_balls = 15 := by
  sorry

end number_of_black_balls_l618_61894


namespace locus_of_centers_l618_61844

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₃ with equation (x-1)² + y² = 25 -/
def C₃ : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 25}

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_to_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₃ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_to_C₃ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 1)^2 + center.2^2 = (5 - radius)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and
    internally tangent to C₃ satisfies the equation 5a² + 9b² + 80a - 400 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_to_C₁ (a, b) r ∧ internally_tangent_to_C₃ (a, b) r) →
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0 := by
  sorry

end locus_of_centers_l618_61844


namespace range_of_g_range_of_a_l618_61823

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Theorem for the range of g(x)
theorem range_of_g : Set.range g = Set.Icc (-1 : ℝ) 1 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) := by sorry

end range_of_g_range_of_a_l618_61823


namespace trailing_zeros_for_specific_fraction_l618_61893

/-- The number of trailing zeros in the decimal representation of a rational number -/
def trailingZeros (n d : ℕ) : ℕ :=
  sorry

/-- The main theorem: number of trailing zeros for 1 / (2^3 * 5^7) -/
theorem trailing_zeros_for_specific_fraction :
  trailingZeros 1 (2^3 * 5^7) = 5 := by
  sorry

end trailing_zeros_for_specific_fraction_l618_61893


namespace counterexample_absolute_value_inequality_l618_61837

theorem counterexample_absolute_value_inequality : 
  ∃ (a b : ℝ), (abs a > abs b) ∧ (a ≤ b) := by
  sorry

end counterexample_absolute_value_inequality_l618_61837


namespace fraction_equality_l618_61889

theorem fraction_equality (a b : ℚ) (h : b / a = 5 / 13) : 
  (a - b) / (a + b) = 4 / 9 := by sorry

end fraction_equality_l618_61889


namespace greatest_three_digit_multiple_of_13_l618_61818

theorem greatest_three_digit_multiple_of_13 : 
  ∃ n : ℕ, n = 988 ∧ 
  n % 13 = 0 ∧
  n ≥ 100 ∧ n < 1000 ∧
  ∀ m : ℕ, m % 13 = 0 → m ≥ 100 → m < 1000 → m ≤ n :=
sorry

end greatest_three_digit_multiple_of_13_l618_61818


namespace ceiling_floor_product_range_l618_61872

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 110 → -11 < y ∧ y < -10 := by
  sorry

end ceiling_floor_product_range_l618_61872
