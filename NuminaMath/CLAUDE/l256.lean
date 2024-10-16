import Mathlib

namespace NUMINAMATH_CALUDE_correct_number_of_plants_l256_25691

/-- The number of large salads Anna needs -/
def salads_needed : ℕ := 12

/-- The fraction of lettuce that will survive (not lost to insects and rabbits) -/
def survival_rate : ℚ := 1/2

/-- The number of large salads each lettuce plant provides -/
def salads_per_plant : ℕ := 3

/-- The number of lettuce plants Anna should grow -/
def plants_to_grow : ℕ := 8

/-- Theorem stating that the number of plants Anna should grow is correct -/
theorem correct_number_of_plants : 
  plants_to_grow * salads_per_plant * survival_rate ≥ salads_needed := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_plants_l256_25691


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l256_25638

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x ≥ 0 → 2*x + 1/(2*x + 1) ≥ 1) ∧
  (∃ x, 2*x + 1/(2*x + 1) ≥ 1 ∧ x < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l256_25638


namespace NUMINAMATH_CALUDE_gasoline_consumption_reduction_l256_25687

theorem gasoline_consumption_reduction 
  (original_price original_quantity : ℝ) 
  (price_increase : ℝ) 
  (spending_increase : ℝ) 
  (h1 : price_increase = 0.25) 
  (h2 : spending_increase = 0.10) :
  let new_price := original_price * (1 + price_increase)
  let new_total_cost := original_price * original_quantity * (1 + spending_increase)
  let new_quantity := new_total_cost / new_price
  (original_quantity - new_quantity) / original_quantity = 0.12 := by
sorry

end NUMINAMATH_CALUDE_gasoline_consumption_reduction_l256_25687


namespace NUMINAMATH_CALUDE_work_completion_time_l256_25634

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / a + 1 / b = 3 / 10) : 1 / b = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l256_25634


namespace NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l256_25673

theorem tan_value_from_sin_cos_equation (α : Real) 
  (h : Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) : 
  Real.tan α = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_cos_equation_l256_25673


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l256_25675

def given_numbers : List Nat := [0, 2, 3, 4, 6]

def is_valid_three_digit (n : Nat) : Bool :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 ∈ given_numbers) ∧ ((n / 10) % 10 ∈ given_numbers) ∧ (n % 10 ∈ given_numbers)

def count_valid_three_digit : Nat :=
  (List.range 1000).filter is_valid_three_digit |>.length

def is_divisible_by_three (n : Nat) : Bool :=
  n % 3 = 0

def count_valid_three_digit_divisible_by_three : Nat :=
  (List.range 1000).filter (λ n => is_valid_three_digit n ∧ is_divisible_by_three n) |>.length

theorem three_digit_numbers_count :
  count_valid_three_digit = 48 ∧
  count_valid_three_digit_divisible_by_three = 20 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_numbers_count_l256_25675


namespace NUMINAMATH_CALUDE_min_sum_at_6_l256_25631

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.S 10 = -2 ∧ seq.S 20 = 16

/-- The main theorem -/
theorem min_sum_at_6 (seq : ArithmeticSequence) 
  (h : problem_conditions seq) :
  ∀ n : ℕ, n ≠ 0 → seq.S 6 ≤ seq.S n :=
sorry

end NUMINAMATH_CALUDE_min_sum_at_6_l256_25631


namespace NUMINAMATH_CALUDE_arithmetic_proof_l256_25671

theorem arithmetic_proof : -3 + 15 - (-8) = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l256_25671


namespace NUMINAMATH_CALUDE_find_k_value_l256_25667

/-- Given two functions f and g, prove that k = -15.8 when f(5) - g(5) = 15 -/
theorem find_k_value (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 5 * x^2 - 3 * x + 6) → 
  (∀ x, g x = 2 * x^2 - k * x + 2) → 
  f 5 - g 5 = 15 → 
  k = -15.8 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l256_25667


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_25_20_l256_25690

theorem half_abs_diff_squares_25_20 : (1/2 : ℝ) * |25^2 - 20^2| = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_25_20_l256_25690


namespace NUMINAMATH_CALUDE_district_a_schools_l256_25625

/-- Represents the three types of schools in Veenapaniville -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts in Veenapaniville -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools in Veenapaniville -/
def totalSchools : Nat := 50

/-- The number of public schools in Veenapaniville -/
def publicSchools : Nat := 25

/-- The number of parochial schools in Veenapaniville -/
def parochialSchools : Nat := 16

/-- The number of private independent schools in Veenapaniville -/
def privateIndependentSchools : Nat := 9

/-- The number of high schools in District B -/
def districtBSchools : Nat := 17

/-- The number of private independent schools in District B -/
def districtBPrivateIndependentSchools : Nat := 2

/-- Function to calculate the number of schools in District C -/
def districtCSchools : Nat := 3 * (min publicSchools (min parochialSchools privateIndependentSchools))

/-- Theorem stating that the number of high schools in District A is 6 -/
theorem district_a_schools :
  totalSchools - (districtBSchools + districtCSchools) = 6 := by
  sorry


end NUMINAMATH_CALUDE_district_a_schools_l256_25625


namespace NUMINAMATH_CALUDE_second_class_size_l256_25604

/-- Given two classes of students, prove that the second class has 50 students. -/
theorem second_class_size (first_class_size : ℕ) (first_class_avg : ℝ) 
  (second_class_avg : ℝ) (total_avg : ℝ) :
  first_class_size = 30 →
  first_class_avg = 30 →
  second_class_avg = 60 →
  total_avg = 48.75 →
  ∃ (second_class_size : ℕ),
    (first_class_size * first_class_avg + second_class_size * second_class_avg) / 
    (first_class_size + second_class_size : ℝ) = total_avg ∧
    second_class_size = 50 :=
by sorry

end NUMINAMATH_CALUDE_second_class_size_l256_25604


namespace NUMINAMATH_CALUDE_blueberry_picking_total_l256_25696

theorem blueberry_picking_total (annie kathryn ben sam : ℕ) : 
  annie = 16 ∧ 
  kathryn = 2 * annie + 2 ∧ 
  ben = kathryn / 2 - 3 ∧ 
  sam = 2 * (ben + kathryn) / 3 → 
  annie + kathryn + ben + sam = 96 := by
sorry

end NUMINAMATH_CALUDE_blueberry_picking_total_l256_25696


namespace NUMINAMATH_CALUDE_product_expansion_l256_25649

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l256_25649


namespace NUMINAMATH_CALUDE_power_of_two_problem_l256_25611

theorem power_of_two_problem (k : ℕ) (N : ℕ) :
  2^k = N → 2^(2*k + 2) = 64 → N = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l256_25611


namespace NUMINAMATH_CALUDE_save_sign_white_area_l256_25685

/-- Represents the area covered by a letter on the sign -/
structure LetterArea where
  s : ℕ
  a : ℕ
  v : ℕ
  e : ℕ

/-- The sign with the word "SAVE" painted on it -/
structure Sign where
  width : ℕ
  height : ℕ
  letterAreas : LetterArea

/-- Calculate the white area of the sign -/
def whiteArea (sign : Sign) : ℕ :=
  sign.width * sign.height - (sign.letterAreas.s + sign.letterAreas.a + sign.letterAreas.v + sign.letterAreas.e)

/-- Theorem stating the white area of the sign is 86 square units -/
theorem save_sign_white_area :
  ∀ (sign : Sign),
    sign.width = 20 ∧
    sign.height = 7 ∧
    sign.letterAreas.s = 14 ∧
    sign.letterAreas.a = 16 ∧
    sign.letterAreas.v = 12 ∧
    sign.letterAreas.e = 12 →
    whiteArea sign = 86 := by
  sorry

end NUMINAMATH_CALUDE_save_sign_white_area_l256_25685


namespace NUMINAMATH_CALUDE_room_occupancy_l256_25656

theorem room_occupancy (total_chairs : ℕ) (total_people : ℕ) : 
  (3 * total_chairs / 4 = total_chairs - 6) →  -- Three-fourths of chairs are occupied
  (2 * total_people / 3 = 3 * total_chairs / 4) →  -- Two-thirds of people are seated
  total_people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l256_25656


namespace NUMINAMATH_CALUDE_polynomial_simplification_l256_25620

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 7) = -4 * s + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l256_25620


namespace NUMINAMATH_CALUDE_base_conversion_problem_l256_25613

theorem base_conversion_problem (n d : ℕ) (hn : n > 0) (hd : d ≤ 9) :
  3 * n^2 + 2 * n + d = 263 ∧ 3 * n^2 + 2 * n + 4 = 253 + 6 * d → n + d = 11 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l256_25613


namespace NUMINAMATH_CALUDE_car_final_velocity_l256_25640

/-- Calculates the final velocity of a car parallel to the ground after accelerating on an inclined slope. -/
theorem car_final_velocity (u : Real) (a : Real) (t : Real) (θ : Real) :
  u = 10 ∧ a = 2 ∧ t = 3 ∧ θ = 15 * π / 180 →
  ∃ v : Real, abs (v - (u + a * t) * Real.cos θ) < 0.0001 ∧ abs (v - 15.4544) < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_car_final_velocity_l256_25640


namespace NUMINAMATH_CALUDE_tom_weeds_earnings_l256_25641

/-- Tom's lawn mowing business -/
def tom_lawn_business (weeds_earnings : ℕ) : Prop :=
  let lawns_mowed : ℕ := 3
  let charge_per_lawn : ℕ := 12
  let gas_cost : ℕ := 17
  let total_profit : ℕ := 29
  let mowing_profit : ℕ := lawns_mowed * charge_per_lawn - gas_cost
  weeds_earnings = total_profit - mowing_profit

theorem tom_weeds_earnings : 
  ∃ (weeds_earnings : ℕ), tom_lawn_business weeds_earnings ∧ weeds_earnings = 10 :=
sorry

end NUMINAMATH_CALUDE_tom_weeds_earnings_l256_25641


namespace NUMINAMATH_CALUDE_crackers_distribution_l256_25652

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 36 →
  num_friends = 6 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l256_25652


namespace NUMINAMATH_CALUDE_interesting_pairs_characterization_l256_25643

/-- An ordered pair (a, b) of positive integers is interesting if for any positive integer n,
    there exists a positive integer k such that a^k + b is divisible by 2^n. -/
def IsInteresting (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, (a.val ^ k.val + b.val) % (2^n.val) = 0

/-- Characterization of interesting pairs -/
theorem interesting_pairs_characterization (a b : ℕ+) :
  IsInteresting a b ↔ 
  (∃ (k l q : ℕ+), k ≥ 2 ∧ l.val % 2 = 1 ∧ q.val % 2 = 1 ∧
    ((a = 2^k.val * l + 1 ∧ b = 2^k.val * q - 1) ∨
     (a = 2^k.val * l - 1 ∧ b = 2^k.val * q + 1))) :=
sorry

end NUMINAMATH_CALUDE_interesting_pairs_characterization_l256_25643


namespace NUMINAMATH_CALUDE_complex_magnitude_l256_25693

-- Define complex numbers w and z
variable (w z : ℂ)

-- Define the given conditions
theorem complex_magnitude (h1 : w * z = 20 - 15 * I) (h2 : Complex.abs w = 5) :
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l256_25693


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l256_25629

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ x y : ℕ, n = 10 * x + y ∧ 
    10 ≤ x + y ∧ x + y < 100 ∧
    x = y / 4 ∧
    n = 28) :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l256_25629


namespace NUMINAMATH_CALUDE_jason_car_count_l256_25686

theorem jason_car_count (purple : ℕ) (red : ℕ) (green : ℕ) : 
  purple = 47 →
  red = purple + 6 →
  green = 4 * red →
  purple + red + green = 312 := by
sorry

end NUMINAMATH_CALUDE_jason_car_count_l256_25686


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l256_25644

theorem scientific_notation_equality : 284000000 = 2.84 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l256_25644


namespace NUMINAMATH_CALUDE_class_transfer_problem_l256_25623

/-- Proof of the class transfer problem -/
theorem class_transfer_problem :
  -- Define the total number of students
  ∀ (total : ℕ),
  -- Define the number of students transferred from A to B
  ∀ (transfer_a_to_b : ℕ),
  -- Define the number of students transferred from B to C
  ∀ (transfer_b_to_c : ℕ),
  -- Condition: total students is 92
  total = 92 →
  -- Condition: 5 students transferred from A to B
  transfer_a_to_b = 5 →
  -- Condition: 32 students transferred from B to C
  transfer_b_to_c = 32 →
  -- Condition: After transfers, students in A = 3 * students in B
  ∃ (final_a final_b : ℕ),
    final_a = 3 * final_b ∧
    final_a + final_b = total - transfer_b_to_c →
  -- Conclusion: Originally 45 students in A and 47 in B
  ∃ (original_a original_b : ℕ),
    original_a = 45 ∧
    original_b = 47 ∧
    original_a + original_b = total :=
by sorry

end NUMINAMATH_CALUDE_class_transfer_problem_l256_25623


namespace NUMINAMATH_CALUDE_brand_preference_l256_25603

theorem brand_preference (total_respondents : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) 
  (h1 : total_respondents = 80)
  (h2 : ratio_x = 3)
  (h3 : ratio_y = 1) :
  (total_respondents * ratio_x) / (ratio_x + ratio_y) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_l256_25603


namespace NUMINAMATH_CALUDE_discount_percentage_for_eight_percent_profit_l256_25699

/-- Proves that the discount percentage resulting in an 8% profit is 10% --/
theorem discount_percentage_for_eight_percent_profit 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percentage : ℝ) :
  cost_price = 10000 →
  selling_price = 12000 →
  profit_percentage = 0.08 →
  (selling_price - (cost_price * (1 + profit_percentage))) / selling_price * 100 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_for_eight_percent_profit_l256_25699


namespace NUMINAMATH_CALUDE_system_solution_l256_25617

theorem system_solution (a b c x y z T : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  x = Real.sqrt (y^2 - a^2) + Real.sqrt (z^2 - a^2) →
  y = Real.sqrt (z^2 - b^2) + Real.sqrt (x^2 - b^2) →
  z = Real.sqrt (x^2 - c^2) + Real.sqrt (y^2 - c^2) →
  1 / T^2 = 2 / (a^2 * b^2) + 2 / (b^2 * c^2) + 2 / (c^2 * a^2) - 1 / a^4 - 1 / b^4 - 1 / c^4 →
  1 / T^2 > 0 →
  x = 2 * T / a ∧ y = 2 * T / b ∧ z = 2 * T / c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l256_25617


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l256_25695

/-- The value of d for which the line y = 3x + d is tangent to the parabola y² = 12x -/
theorem tangent_line_to_parabola : ∃! d : ℝ,
  ∀ x y : ℝ, (y = 3 * x + d ∧ y^2 = 12 * x) →
  (∃! x₀ y₀ : ℝ, y₀ = 3 * x₀ + d ∧ y₀^2 = 12 * x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l256_25695


namespace NUMINAMATH_CALUDE_not_p_false_range_p_necessary_not_sufficient_range_l256_25670

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2*x + a^2 + 3*a - 3

-- Define proposition p
def p (a : ℝ) : Prop := ∃ x, f x a < 0

-- Define proposition r
def r (a x : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 1 + a

-- Theorem for part (1)
theorem not_p_false_range (a : ℝ) : 
  ¬(¬(p a)) → a ∈ Set.Ioo (-4 : ℝ) 1 :=
sorry

-- Theorem for part (2)
theorem p_necessary_not_sufficient_range (a : ℝ) :
  (∀ x, r a x → p a) ∧ (∃ x, p a ∧ ¬r a x) → a ∈ Set.Ici 5 :=
sorry

end NUMINAMATH_CALUDE_not_p_false_range_p_necessary_not_sufficient_range_l256_25670


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l256_25683

/-- Given two positive integers with LCM 750 and HCF 25, prove their product is 18750 -/
theorem product_from_lcm_hcf (a b : ℕ+) : 
  Nat.lcm a b = 750 → Nat.gcd a b = 25 → a * b = 18750 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l256_25683


namespace NUMINAMATH_CALUDE_train_pass_man_time_l256_25642

def train_speed : Real := 36 -- km/hr
def platform_length : Real := 180 -- meters
def time_pass_platform : Real := 30 -- seconds

theorem train_pass_man_time : 
  ∃ (train_length : Real),
    (train_speed * 1000 / 3600 * time_pass_platform = train_length + platform_length) ∧
    (train_length / (train_speed * 1000 / 3600) = 12) :=
by sorry

end NUMINAMATH_CALUDE_train_pass_man_time_l256_25642


namespace NUMINAMATH_CALUDE_probability_theorem_l256_25618

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd (a * b) ∧ (is_prime a ∨ is_prime b)

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 42

theorem probability_theorem : 
  (valid_pairs : ℚ) / total_pairs = 21 / 95 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l256_25618


namespace NUMINAMATH_CALUDE_kants_clock_problem_l256_25621

/-- Kant's Clock Problem -/
theorem kants_clock_problem (T_F T_2 T_S : ℝ) :
  ∃ T : ℝ, T = T_F + (T_2 - T_S) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_kants_clock_problem_l256_25621


namespace NUMINAMATH_CALUDE_counterexample_square_inequality_l256_25661

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_square_inequality_l256_25661


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l256_25647

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ),
    (C₁ A.1 A.2 ∧ C₂ A.1 A.2) ∧
    (C₁ B.1 B.2 ∧ C₂ B.1 B.2) ∧
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l256_25647


namespace NUMINAMATH_CALUDE_difference_of_squares_form_l256_25632

theorem difference_of_squares_form (x y : ℝ) : 
  ∃ (a b : ℝ), (-x + y) * (x + y) = a^2 - b^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_form_l256_25632


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3001m_24567n_l256_25615

theorem smallest_positive_integer_3001m_24567n : 
  ∃ (m n : ℤ), 3001 * m + 24567 * n = (Nat.gcd 3001 24567 : ℤ) ∧
  ∀ (k : ℤ), (∃ (a b : ℤ), k = 3001 * a + 24567 * b) → k = 0 ∨ abs k ≥ (Nat.gcd 3001 24567 : ℤ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3001m_24567n_l256_25615


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l256_25635

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is c/a, where c² = a² + b² --/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  c / a = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l256_25635


namespace NUMINAMATH_CALUDE_chess_group_size_l256_25676

-- Define the number of players in the chess group
def num_players : ℕ := 30

-- Define the total number of games played
def total_games : ℕ := 435

-- Theorem stating that the number of players is correct given the conditions
theorem chess_group_size :
  (num_players.choose 2 = total_games) ∧ (num_players > 0) := by
  sorry

end NUMINAMATH_CALUDE_chess_group_size_l256_25676


namespace NUMINAMATH_CALUDE_total_gas_cost_l256_25619

-- Define the given parameters
def miles_per_gallon : ℝ := 50
def miles_per_day : ℝ := 75
def price_per_gallon : ℝ := 3
def number_of_days : ℝ := 10

-- Define the theorem
theorem total_gas_cost : 
  (number_of_days * miles_per_day / miles_per_gallon) * price_per_gallon = 45 :=
by
  sorry


end NUMINAMATH_CALUDE_total_gas_cost_l256_25619


namespace NUMINAMATH_CALUDE_eugene_pencils_left_l256_25607

/-- The number of pencils Eugene has left after giving some away -/
def pencils_left (initial : Real) (given_away : Real) : Real :=
  initial - given_away

/-- Theorem: Eugene has 199.0 pencils left after giving away 35.0 pencils from his initial 234.0 pencils -/
theorem eugene_pencils_left : pencils_left 234.0 35.0 = 199.0 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_left_l256_25607


namespace NUMINAMATH_CALUDE_floor_sqrt_20_squared_l256_25653

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_20_squared_l256_25653


namespace NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_geq_eight_l256_25663

theorem product_of_reciprocal_minus_one_geq_eight (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_geq_eight_l256_25663


namespace NUMINAMATH_CALUDE_bruno_score_l256_25645

/-- Given that Richard's score is 62 and Bruno's score is 14 points lower than Richard's,
    prove that Bruno's score is 48. -/
theorem bruno_score (richard_score : ℕ) (bruno_diff : ℕ) : 
  richard_score = 62 → 
  bruno_diff = 14 → 
  richard_score - bruno_diff = 48 := by
  sorry

end NUMINAMATH_CALUDE_bruno_score_l256_25645


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l256_25665

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 11) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 1296/11 := by
  sorry

theorem min_value_sum_reciprocals_achievable (ε : ℝ) (hε : ε > 0) :
  ∃ p q r s t u : ℝ, 
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0 ∧ u > 0 ∧
    p + q + r + s + t + u = 11 ∧
    1/p + 9/q + 25/r + 49/s + 81/t + 121/u < 1296/11 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_sum_reciprocals_achievable_l256_25665


namespace NUMINAMATH_CALUDE_files_left_theorem_l256_25657

/-- Calculates the number of files left after deletion -/
def files_left (initial_files : ℕ) (deleted_files : ℕ) : ℕ :=
  initial_files - deleted_files

/-- Theorem: The number of files left is the difference between initial files and deleted files -/
theorem files_left_theorem (initial_files deleted_files : ℕ) 
  (h : deleted_files ≤ initial_files) : 
  files_left initial_files deleted_files = initial_files - deleted_files :=
by
  sorry

#eval files_left 21 14  -- Should output 7

end NUMINAMATH_CALUDE_files_left_theorem_l256_25657


namespace NUMINAMATH_CALUDE_initial_money_calculation_l256_25697

/-- Proves that if a person spends half of their initial money, then half of the remaining money, 
    and is left with 1250 won, their initial amount was 5000 won. -/
theorem initial_money_calculation (initial_money : ℝ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l256_25697


namespace NUMINAMATH_CALUDE_find_d_l256_25659

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) : 
  (∀ x, f c (g c x) = -15 * x + d) → d = -18 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l256_25659


namespace NUMINAMATH_CALUDE_point_translation_l256_25680

/-- Given a point B with coordinates (5, -1) that is translated upwards by 2 units
    to obtain point A with coordinates (a+b, a-b), prove that a = 3 and b = 2. -/
theorem point_translation (a b : ℝ) : 
  (5 : ℝ) = a + b ∧ (1 : ℝ) = a - b → a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_l256_25680


namespace NUMINAMATH_CALUDE_income_increase_percentage_l256_25674

/-- Proves that given the ratio of expenditure to savings is 3:2, if savings increase by 6% and expenditure increases by 21%, then the income increases by 15% -/
theorem income_increase_percentage 
  (I : ℝ) -- Initial income
  (E : ℝ) -- Initial expenditure
  (S : ℝ) -- Initial savings
  (h1 : E / S = 3 / 2) -- Ratio of expenditure to savings is 3:2
  (h2 : I = E + S) -- Income is the sum of expenditure and savings
  (h3 : S * 1.06 + E * 1.21 = I * (1 + 15/100)) -- New savings + new expenditure = new income
  : ∃ (x : ℝ), x = 15 ∧ I * (1 + x/100) = S * 1.06 + E * 1.21 :=
by sorry

end NUMINAMATH_CALUDE_income_increase_percentage_l256_25674


namespace NUMINAMATH_CALUDE_fraction_irreducible_l256_25633

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l256_25633


namespace NUMINAMATH_CALUDE_last_digit_fibonacci_mod12_l256_25626

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_mod12 (n : ℕ) : ℕ := fibonacci n % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod12 k = d

theorem last_digit_fibonacci_mod12 :
  ∀ d : ℕ, d < 12 →
    (digit_appears d 21 → digit_appears 11 22) ∧
    (¬ digit_appears 11 21) ∧
    digit_appears 11 22 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_fibonacci_mod12_l256_25626


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l256_25639

/-- Calculates the area of a square sheet of wrapping paper needed to wrap a rectangular box -/
theorem wrapping_paper_area (box_length box_width box_height extra_fold : ℝ) :
  box_length = 10 ∧ box_width = 10 ∧ box_height = 5 ∧ extra_fold = 2 →
  (box_width / 2 + box_height + extra_fold) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l256_25639


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l256_25681

/-- The parabola equation -/
def parabola (x d : ℝ) : ℝ := x^2 - 6*x + d

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 3

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (d : ℝ) : ℝ := parabola vertex_x d

theorem vertex_on_x_axis (d : ℝ) : vertex_y d = 0 ↔ d = 9 := by sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l256_25681


namespace NUMINAMATH_CALUDE_tommy_wheel_count_l256_25689

/-- The number of wheels on each truck -/
def truck_wheels : ℕ := 4

/-- The number of wheels on each car -/
def car_wheels : ℕ := 4

/-- The number of trucks Tommy saw -/
def trucks_seen : ℕ := 12

/-- The number of cars Tommy saw -/
def cars_seen : ℕ := 13

/-- The total number of wheels Tommy saw -/
def total_wheels : ℕ := (trucks_seen * truck_wheels) + (cars_seen * car_wheels)

theorem tommy_wheel_count : total_wheels = 100 := by
  sorry

end NUMINAMATH_CALUDE_tommy_wheel_count_l256_25689


namespace NUMINAMATH_CALUDE_saree_ultimate_cost_l256_25609

/-- Calculates the ultimate cost of a saree after discounts and commission -/
def ultimate_cost (initial_price : ℝ) (discount1 discount2 discount3 commission : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_discount3 := price_after_discount2 * (1 - discount3)
  let final_price := price_after_discount3 * (1 - commission)
  final_price

/-- Theorem stating the ultimate cost of the saree -/
theorem saree_ultimate_cost :
  ultimate_cost 340 0.2 0.15 0.1 0.05 = 197.676 := by
  sorry

end NUMINAMATH_CALUDE_saree_ultimate_cost_l256_25609


namespace NUMINAMATH_CALUDE_juanita_drum_contest_loss_l256_25650

/-- Calculates the net profit (or loss) for a drum contest participant -/
def drumContestProfit (entryFee : ℚ) (threshold : ℕ) (earningsPerDrum : ℚ) (drumsHit : ℕ) : ℚ :=
  let earningsDrums := max (drumsHit - threshold) 0
  earningsDrums * earningsPerDrum - entryFee

/-- Proves that Juanita's net loss in the drum contest is $7.50 -/
theorem juanita_drum_contest_loss :
  drumContestProfit 10 200 (25 / 1000) 300 = -15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_juanita_drum_contest_loss_l256_25650


namespace NUMINAMATH_CALUDE_part1_part2_l256_25651

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - 3*a) / (a - 2*x) ≥ 0 ∧ a > 0

def q (x : ℝ) : Prop := 2*x^2 - 7*x + 6 < 0

-- Part 1
theorem part1 (x : ℝ) : 
  p x 1 ∧ q x → 3/2 < x ∧ x < 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x a) → 
  2/3 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l256_25651


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l256_25672

def f (x : ℝ) : ℝ := 3 * x + 1

theorem f_monotone_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l256_25672


namespace NUMINAMATH_CALUDE_granger_spam_cans_l256_25668

/-- Represents the grocery items and their prices --/
structure GroceryItems where
  spam_price : ℕ
  peanut_butter_price : ℕ
  bread_price : ℕ

/-- Represents the quantities of items bought --/
structure Quantities where
  spam_cans : ℕ
  peanut_butter_jars : ℕ
  bread_loaves : ℕ

/-- Calculates the total cost of the groceries --/
def total_cost (items : GroceryItems) (quantities : Quantities) : ℕ :=
  items.spam_price * quantities.spam_cans +
  items.peanut_butter_price * quantities.peanut_butter_jars +
  items.bread_price * quantities.bread_loaves

/-- Theorem stating that Granger bought 4 cans of Spam --/
theorem granger_spam_cans :
  ∀ (items : GroceryItems) (quantities : Quantities),
    items.spam_price = 3 →
    items.peanut_butter_price = 5 →
    items.bread_price = 2 →
    quantities.peanut_butter_jars = 3 →
    quantities.bread_loaves = 4 →
    total_cost items quantities = 59 →
    quantities.spam_cans = 4 :=
by sorry

end NUMINAMATH_CALUDE_granger_spam_cans_l256_25668


namespace NUMINAMATH_CALUDE_problem_solution_l256_25610

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l256_25610


namespace NUMINAMATH_CALUDE_odd_prime_equation_l256_25654

theorem odd_prime_equation (p a b : ℕ) : 
  Prime p → 
  Odd p → 
  a > 0 → 
  b > 0 → 
  (p + 1)^a - p^b = 1 → 
  a = 1 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_equation_l256_25654


namespace NUMINAMATH_CALUDE_direction_vector_form_l256_25636

/-- Given a line passing through two points, prove that its direction vector has a specific form -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-3, 2) → p2 = (2, -3) → 
  (∃ k : ℝ, k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) → 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_form_l256_25636


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element5_value_l256_25622

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The fifth element (k = 4) in Row 20 of Pascal's triangle -/
def pascal_triangle_row20_element5 : ℕ := binomial 20 4

theorem pascal_triangle_row20_element5_value :
  pascal_triangle_row20_element5 = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element5_value_l256_25622


namespace NUMINAMATH_CALUDE_other_factor_proof_l256_25688

theorem other_factor_proof (a : ℕ) (h1 : a = 363) : 
  ∃ n : ℕ, n * 33 * a * 43 * 62 * 1311 = 33 * 363 * 38428986 :=
by
  sorry

end NUMINAMATH_CALUDE_other_factor_proof_l256_25688


namespace NUMINAMATH_CALUDE_math_team_selection_l256_25669

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem math_team_selection :
  let girls := 5
  let boys := 5
  let girls_on_team := 3
  let boys_on_team := 2
  (choose girls girls_on_team) * (choose boys boys_on_team) = 100 := by
  sorry

end NUMINAMATH_CALUDE_math_team_selection_l256_25669


namespace NUMINAMATH_CALUDE_total_rainfall_l256_25612

theorem total_rainfall (monday tuesday wednesday : ℚ) 
  (h1 : monday = 0.16666666666666666)
  (h2 : tuesday = 0.4166666666666667)
  (h3 : wednesday = 0.08333333333333333) :
  monday + tuesday + wednesday = 0.6666666666666667 := by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l256_25612


namespace NUMINAMATH_CALUDE_termite_ridden_collapsing_fraction_value_l256_25646

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of homes on Gotham Street that are termite-ridden but not collapsing -/
def termite_ridden_not_collapsing_fraction : ℚ := 1/10

/-- The fraction of termite-ridden homes that are collapsing -/
def termite_ridden_collapsing_fraction : ℚ := 
  (termite_ridden_fraction - termite_ridden_not_collapsing_fraction) / termite_ridden_fraction

theorem termite_ridden_collapsing_fraction_value : 
  termite_ridden_collapsing_fraction = 7/30 := by
  sorry

end NUMINAMATH_CALUDE_termite_ridden_collapsing_fraction_value_l256_25646


namespace NUMINAMATH_CALUDE_symmetry_shift_l256_25601

noncomputable def smallest_shift_for_symmetry : ℝ := 7 * Real.pi / 6

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem symmetry_shift :
  let f (x m : ℝ) := Real.cos (x + m) - Real.sqrt 3 * Real.sin (x + m)
  ∀ m : ℝ, m > 0 → (
    (is_symmetric_about_y_axis (f · m)) ↔ 
    m ≥ smallest_shift_for_symmetry
  ) :=
sorry

end NUMINAMATH_CALUDE_symmetry_shift_l256_25601


namespace NUMINAMATH_CALUDE_complex_expression_calculation_l256_25614

theorem complex_expression_calculation (a b : ℂ) :
  a = 3 + 2*I ∧ b = 1 - 3*I → 4*a + 5*b + a*b = 26 - 14*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_calculation_l256_25614


namespace NUMINAMATH_CALUDE_min_removed_length_345_square_l256_25658

/-- Represents a right-angled triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right_angled : a^2 + b^2 = c^2

/-- Represents a square formed by four right-angled triangles -/
structure TriangleSquare where
  triangle : RightTriangle
  side_length : ℕ
  is_valid : side_length = triangle.a + triangle.b

/-- The minimum length of line segments to be removed to make the figure drawable in one stroke -/
def min_removed_length (square : TriangleSquare) : ℕ := sorry

/-- Theorem stating that the minimum length of removed line segments is 7 for a square formed by four 3-4-5 triangles -/
theorem min_removed_length_345_square :
  ∀ (square : TriangleSquare),
    square.triangle = { a := 3, b := 4, c := 5, is_right_angled := by norm_num }
    → min_removed_length square = 7 := by sorry

end NUMINAMATH_CALUDE_min_removed_length_345_square_l256_25658


namespace NUMINAMATH_CALUDE_parabola_tangent_property_l256_25637

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Given a parabola Γ: y² = 2px (p > 0) with focus F, and a point Q outside Γ (not on the x-axis),
    let tangents QA and QB intersect Γ at A and B respectively, and the y-axis at C and D.
    If M is the circumcenter of triangle QAB, then FM is tangent to the circumcircle of triangle FCD. -/
theorem parabola_tangent_property (Γ : Parabola) (F Q A B C D M : Point) :
  Q.x ≠ 0 →  -- Q is not on y-axis
  Q.y ≠ 0 →  -- Q is not on x-axis
  (∃ t : ℝ, A = Point.mk (t^2 / (2 * Γ.p)) t) →  -- A is on the parabola
  (∃ t : ℝ, B = Point.mk (t^2 / (2 * Γ.p)) t) →  -- B is on the parabola
  F = Point.mk (Γ.p / 2) 0 →  -- F is the focus
  C = Point.mk 0 C.y →  -- C is on y-axis
  D = Point.mk 0 D.y →  -- D is on y-axis
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * A.x + l.b * A.y + l.c = 0) →  -- QA is a line
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0) →  -- QB is a line
  (∀ P : Point, (P.x - M.x)^2 + (P.y - M.y)^2 = (A.x - M.x)^2 + (A.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (B.x - M.x)^2 + (B.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (Q.x - M.x)^2 + (Q.y - M.y)^2) →  -- M is circumcenter of QAB
  (∃ T : Point, ∃ r : ℝ,
    (T.x - F.x)^2 + (T.y - F.y)^2 = (C.x - F.x)^2 + (C.y - F.y)^2 ∧
    (T.x - F.x)^2 + (T.y - F.y)^2 = (D.x - F.x)^2 + (D.y - F.y)^2 ∧
    (M.x - F.x) * (T.x - F.x) + (M.y - F.y) * (T.y - F.y) = r^2) →  -- FM is tangent to circumcircle of FCD
  True :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_property_l256_25637


namespace NUMINAMATH_CALUDE_unique_prime_root_equation_l256_25694

theorem unique_prime_root_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    (p + q : ℝ) ^ (1 / n : ℝ) = p - q →
    p = 5 ∧ q = 3 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_root_equation_l256_25694


namespace NUMINAMATH_CALUDE_fraction_multiplication_l256_25684

theorem fraction_multiplication (x : ℝ) : 
  (1 : ℝ) / 3 * 2 / 7 * 9 / 13 * x / 17 = 18 * x / 4911 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l256_25684


namespace NUMINAMATH_CALUDE_largest_package_size_l256_25664

theorem largest_package_size (ming catherine alex : ℕ) 
  (h_ming : ming = 36) 
  (h_catherine : catherine = 60) 
  (h_alex : alex = 48) : 
  Nat.gcd ming (Nat.gcd catherine alex) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l256_25664


namespace NUMINAMATH_CALUDE_dinitrogen_pentoxide_weight_l256_25624

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen pentoxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen pentoxide -/
def O_count : ℕ := 5

/-- The molecular weight of Dinitrogen pentoxide in g/mol -/
def molecular_weight_N2O5 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

theorem dinitrogen_pentoxide_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end NUMINAMATH_CALUDE_dinitrogen_pentoxide_weight_l256_25624


namespace NUMINAMATH_CALUDE_davids_math_marks_l256_25678

def english_marks : ℝ := 90
def physics_marks : ℝ := 85
def chemistry_marks : ℝ := 87
def biology_marks : ℝ := 85
def average_marks : ℝ := 87.8
def total_subjects : ℕ := 5

theorem davids_math_marks :
  ∃ (math_marks : ℝ),
    (english_marks + physics_marks + chemistry_marks + biology_marks + math_marks) / total_subjects = average_marks ∧
    math_marks = 92 := by
  sorry

end NUMINAMATH_CALUDE_davids_math_marks_l256_25678


namespace NUMINAMATH_CALUDE_simplification_proof_equation_solution_proof_l256_25616

-- Problem 1: Simplification
theorem simplification_proof (a : ℝ) (ha : a ≠ 0 ∧ a ≠ 1) :
  (a - 1/a) / ((a^2 - 2*a + 1) / a) = (a + 1) / (a - 1) := by sorry

-- Problem 2: Equation Solving
theorem equation_solution_proof :
  ∀ x : ℝ, x = -1 ↔ 2*x/(x-2) = 1 - 1/(2-x) := by sorry

end NUMINAMATH_CALUDE_simplification_proof_equation_solution_proof_l256_25616


namespace NUMINAMATH_CALUDE_a_8_equals_15_l256_25682

-- Define the sequence S_n
def S (n : ℕ) : ℕ := n^2

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else S n - S (n-1)

-- Theorem statement
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_a_8_equals_15_l256_25682


namespace NUMINAMATH_CALUDE_female_listeners_count_l256_25608

/-- Represents the survey results from radio station KMAT -/
structure SurveyResults where
  total_listeners : Nat
  total_non_listeners : Nat
  male_listeners : Nat
  male_non_listeners : Nat
  female_non_listeners : Nat
  undeclared_listeners : Nat
  undeclared_non_listeners : Nat

/-- Calculates the number of female listeners based on the survey results -/
def female_listeners (results : SurveyResults) : Nat :=
  results.total_listeners - results.male_listeners - results.undeclared_listeners

/-- Theorem stating that the number of female listeners is 65 -/
theorem female_listeners_count (results : SurveyResults)
  (h1 : results.total_listeners = 160)
  (h2 : results.total_non_listeners = 235)
  (h3 : results.male_listeners = 75)
  (h4 : results.male_non_listeners = 85)
  (h5 : results.female_non_listeners = 135)
  (h6 : results.undeclared_listeners = 20)
  (h7 : results.undeclared_non_listeners = 15) :
  female_listeners results = 65 := by
  sorry

#check female_listeners_count

end NUMINAMATH_CALUDE_female_listeners_count_l256_25608


namespace NUMINAMATH_CALUDE_parallel_line_equation_l256_25605

/-- Given a line L passing through the point (1, 0) and parallel to the line x - 2y - 2 = 0,
    prove that the equation of L is x - 2y - 1 = 0 -/
theorem parallel_line_equation : 
  ∀ (L : Set (ℝ × ℝ)),
  (∀ p : ℝ × ℝ, p ∈ L ↔ p.1 - 2 * p.2 - 1 = 0) →
  (1, 0) ∈ L →
  (∀ p q : ℝ × ℝ, p ∈ L → q ∈ L → p ≠ q → (p.1 - q.1) / (p.2 - q.2) = 1 / 2) →
  true :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l256_25605


namespace NUMINAMATH_CALUDE_ant_probability_after_six_moves_l256_25606

/-- Represents the vertices of a cube -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Represents the probability distribution over the vertices -/
def ProbabilityDistribution := Vertex → ℚ

/-- The initial probability distribution -/
def initial_distribution : ProbabilityDistribution :=
  fun v => match v with
  | Vertex.A => 1
  | _ => 0

/-- The transition function for one move -/
def transition (p : ProbabilityDistribution) : ProbabilityDistribution :=
  fun v => match v with
  | Vertex.A => (1/3) * (p Vertex.B + p Vertex.D + p Vertex.E)
  | Vertex.B => (1/3) * (p Vertex.A + p Vertex.C + p Vertex.F)
  | Vertex.C => (1/3) * (p Vertex.B + p Vertex.D + p Vertex.G)
  | Vertex.D => (1/3) * (p Vertex.A + p Vertex.C + p Vertex.H)
  | Vertex.E => (1/3) * (p Vertex.A + p Vertex.F + p Vertex.H)
  | Vertex.F => (1/3) * (p Vertex.B + p Vertex.E + p Vertex.G)
  | Vertex.G => (1/3) * (p Vertex.C + p Vertex.F + p Vertex.H)
  | Vertex.H => (1/3) * (p Vertex.D + p Vertex.E + p Vertex.G)

/-- Apply the transition function n times -/
def iterate_transition (n : ℕ) (p : ProbabilityDistribution) : ProbabilityDistribution :=
  match n with
  | 0 => p
  | n + 1 => transition (iterate_transition n p)

theorem ant_probability_after_six_moves :
  (iterate_transition 6 initial_distribution) Vertex.A = 61 / 243 := by
  sorry


end NUMINAMATH_CALUDE_ant_probability_after_six_moves_l256_25606


namespace NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l256_25677

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_div_sqrt_2_equals_3_l256_25677


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l256_25630

noncomputable section

variables (x : ℝ) (p : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a - 1/x - Real.log x

def g (x : ℝ) (p : ℝ) : ℝ := Real.log x - 2*(x-p)/(x+p) - Real.log p

theorem unique_solution_implies_a_eq_one :
  (∃! x, f 1 x = 0) → 1 = 1 := by sorry

theorem g_monotone_increasing (hp : p > 0) :
  Monotone (g · p) := by sorry

theorem sum_of_zeros_lt_two :
  ∃ x₁ x₂, x₁ < x₂ ∧ f 1 x₁ = 0 ∧ f 1 x₂ = 0 → x₁ + x₂ < 2 := by sorry

end

end NUMINAMATH_CALUDE_unique_solution_implies_a_eq_one_g_monotone_increasing_sum_of_zeros_lt_two_l256_25630


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l256_25692

theorem digit_sum_puzzle (a b : ℕ) : 
  a ∈ (Set.Icc 1 9) → 
  b ∈ (Set.Icc 1 9) → 
  82 * 10 * a + 7 + 6 * b = 190 → 
  a + 2 * b = 7 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l256_25692


namespace NUMINAMATH_CALUDE_unique_solution_l256_25628

-- Define the possible colors
inductive Color
| Red
| Blue

-- Define a structure for a child's outfit
structure Outfit :=
  (tshirt : Color)
  (shorts : Color)

-- Define the children
structure Children :=
  (Alyna : Outfit)
  (Bohdan : Outfit)
  (Vika : Outfit)
  (Grysha : Outfit)

-- Define the conditions
def satisfies_conditions (c : Children) : Prop :=
  (c.Alyna.tshirt = Color.Red) ∧
  (c.Bohdan.tshirt = Color.Red) ∧
  (c.Alyna.shorts ≠ c.Bohdan.shorts) ∧
  (c.Vika.tshirt ≠ c.Grysha.tshirt) ∧
  (c.Vika.shorts = Color.Blue) ∧
  (c.Grysha.shorts = Color.Blue) ∧
  (c.Alyna.tshirt ≠ c.Vika.tshirt) ∧
  (c.Alyna.shorts ≠ c.Vika.shorts)

-- Define the correct answer
def correct_answer : Children :=
  { Alyna := { tshirt := Color.Red, shorts := Color.Red },
    Bohdan := { tshirt := Color.Red, shorts := Color.Blue },
    Vika := { tshirt := Color.Blue, shorts := Color.Blue },
    Grysha := { tshirt := Color.Red, shorts := Color.Blue } }

-- The theorem to prove
theorem unique_solution :
  ∀ c : Children, satisfies_conditions c → c = correct_answer :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l256_25628


namespace NUMINAMATH_CALUDE_problem_solution_l256_25627

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 6) : 
  (x^5 + 3*y^3) / 9 = 99 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l256_25627


namespace NUMINAMATH_CALUDE_sum_of_defined_values_l256_25602

theorem sum_of_defined_values : 
  let x : ℝ := -2 + 3
  let y : ℝ := |(-5)|
  let z : ℝ := 4 * (-1/4)
  x + y + z = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_defined_values_l256_25602


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l256_25679

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) →
  a = 274 ∨ a = 40 :=
by sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l256_25679


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l256_25655

/-- Given a shopkeeper who sells 15 articles at the cost price of 20 articles, 
    prove that the profit percentage is 1/3. -/
theorem shopkeeper_profit_percentage 
  (cost_price : ℝ) (cost_price_positive : cost_price > 0) : 
  let selling_price := 20 * cost_price
  let total_cost := 15 * cost_price
  let profit := selling_price - total_cost
  let profit_percentage := profit / total_cost
  profit_percentage = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l256_25655


namespace NUMINAMATH_CALUDE_mr_grey_polo_shirts_l256_25648

/-- Represents the purchase of gifts by Mr. Grey -/
structure GiftPurchase where
  polo_shirt_price : ℕ
  necklace_price : ℕ
  computer_game_price : ℕ
  necklace_count : ℕ
  rebate : ℕ
  total_cost : ℕ

/-- Calculates the number of polo shirts bought given the gift purchase details -/
def calculate_polo_shirts (purchase : GiftPurchase) : ℕ :=
  (purchase.total_cost + purchase.rebate - purchase.necklace_price * purchase.necklace_count - purchase.computer_game_price) / purchase.polo_shirt_price

/-- Theorem stating that Mr. Grey bought 3 polo shirts -/
theorem mr_grey_polo_shirts :
  let purchase : GiftPurchase := {
    polo_shirt_price := 26,
    necklace_price := 83,
    computer_game_price := 90,
    necklace_count := 2,
    rebate := 12,
    total_cost := 322
  }
  calculate_polo_shirts purchase = 3 := by
  sorry

end NUMINAMATH_CALUDE_mr_grey_polo_shirts_l256_25648


namespace NUMINAMATH_CALUDE_fraction_simplification_l256_25662

theorem fraction_simplification (c : ℝ) : (5 + 6 * c) / 9 + 3 = (32 + 6 * c) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l256_25662


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l256_25698

/-- A linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The four quadrants in a Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: y = 2x - 3 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  ¬ passesThrough { m := 2, b := -3 } Quadrant.II :=
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l256_25698


namespace NUMINAMATH_CALUDE_delta_properties_l256_25666

def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

theorem delta_properties :
  (delta (-4) 4 = 0) ∧
  (delta (1/3) (1/4) = delta 3 4) ∧
  ∃ (m n : ℚ), delta (-m) n ≠ delta m (-n) := by
  sorry

end NUMINAMATH_CALUDE_delta_properties_l256_25666


namespace NUMINAMATH_CALUDE_square_in_S_l256_25660

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2

def S : Set ℕ :=
  {n | is_sum_of_two_squares (n - 1) ∧ 
       is_sum_of_two_squares n ∧ 
       is_sum_of_two_squares (n + 1)}

theorem square_in_S (n : ℕ) (hn : n ∈ S) : n^2 ∈ S := by
  sorry

end NUMINAMATH_CALUDE_square_in_S_l256_25660


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l256_25600

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |2 - x| = 2 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l256_25600
