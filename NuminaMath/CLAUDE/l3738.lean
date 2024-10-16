import Mathlib

namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3738_373855

/-- Given a circle with equation x^2 - 8x + y^2 + 16y = -100, 
    prove that the sum of the x-coordinate of the center, 
    the y-coordinate of the center, and the radius is -4 + 2√5 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -4 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3738_373855


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_and_sum_l3738_373829

theorem max_value_cube_root_sum_and_sum (x y : ℝ) :
  (x^(1/3) + y^(1/3) = 2) →
  (x + y = 20) →
  max x y = 10 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_and_sum_l3738_373829


namespace NUMINAMATH_CALUDE_hyperbola_rational_parameterization_l3738_373858

theorem hyperbola_rational_parameterization
  (x p q : ℚ) 
  (h : p^2 - x*q^2 = 1) :
  ∃ (a b : ℤ), 
    p = (a^2 + x*b^2) / (a^2 - x*b^2) ∧
    q = 2*a*b / (a^2 - x*b^2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_rational_parameterization_l3738_373858


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3738_373805

theorem cubic_root_sum (a b c : ℝ) : 
  (45 * a^3 - 70 * a^2 + 28 * a - 2 = 0) →
  (45 * b^3 - 70 * b^2 + 28 * b - 2 = 0) →
  (45 * c^3 - 70 * c^2 + 28 * c - 2 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  -1 < a → a < 1 →
  -1 < b → b < 1 →
  -1 < c → c < 1 →
  1/(1-a) + 1/(1-b) + 1/(1-c) = 13/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3738_373805


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l3738_373852

theorem min_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  (1 / x + 1 / y + 1 / z) ≥ 10 / 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 3 ∧ y₀ = 2 * x₀ ∧
  (1 / x₀ + 1 / y₀ + 1 / z₀) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l3738_373852


namespace NUMINAMATH_CALUDE_roots_of_x_squared_equals_16_l3738_373887

theorem roots_of_x_squared_equals_16 :
  let f : ℝ → ℝ := λ x ↦ x^2 - 16
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 4 ∧ x₂ = -4 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_x_squared_equals_16_l3738_373887


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l3738_373890

theorem fraction_zero_implies_x_zero (x : ℝ) : 
  (x^2 - x) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_zero_l3738_373890


namespace NUMINAMATH_CALUDE_triangle_area_change_l3738_373897

theorem triangle_area_change (b h : ℝ) (h1 : b > 0) (h2 : h > 0) : 
  (3 * b) * (2 * h) / 2 ≠ 4 * (b * h / 2) := by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l3738_373897


namespace NUMINAMATH_CALUDE_cubic_poly_b_value_l3738_373861

/-- Represents a cubic polynomial of the form x^3 - ax^2 + bx - b --/
def cubic_poly (a b : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + b*x - b

/-- Predicate to check if all roots of the polynomial are real and positive --/
def all_roots_real_positive (a b : ℝ) : Prop :=
  ∀ x : ℝ, cubic_poly a b x = 0 → x > 0

/-- The main theorem stating the value of b --/
theorem cubic_poly_b_value :
  ∃ (a : ℝ), a > 0 ∧
  (∀ a' : ℝ, a' > 0 → all_roots_real_positive a' (a'^2/3) → a ≤ a') ∧
  all_roots_real_positive a (a^2/3) ∧
  a^2/3 = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_poly_b_value_l3738_373861


namespace NUMINAMATH_CALUDE_restaurant_time_is_ten_l3738_373846

-- Define the times as natural numbers (in minutes)
def time_to_hidden_lake : ℕ := 15
def time_from_hidden_lake : ℕ := 7
def total_journey_time : ℕ := 32

-- Define the time to Lake Park restaurant as a function
def time_to_restaurant : ℕ := total_journey_time - (time_to_hidden_lake + time_from_hidden_lake)

-- Theorem statement
theorem restaurant_time_is_ten : time_to_restaurant = 10 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_time_is_ten_l3738_373846


namespace NUMINAMATH_CALUDE_gcf_lcm_product_4_12_l3738_373883

theorem gcf_lcm_product_4_12 : 
  (Nat.gcd 4 12) * (Nat.lcm 4 12) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_product_4_12_l3738_373883


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3738_373867

/-- The polynomial function we're analyzing -/
def g (x : ℝ) : ℝ := x^10 + 9*x^9 + 20*x^8 + 2000*x^7 - 1500*x^6

/-- Theorem stating that g(x) = 0 has exactly one positive real solution -/
theorem unique_positive_solution : ∃! x : ℝ, x > 0 ∧ g x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3738_373867


namespace NUMINAMATH_CALUDE_legs_heads_difference_l3738_373813

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (group : AnimalGroup) : ℕ :=
  2 * group.ducks + 4 * group.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (group : AnimalGroup) : ℕ :=
  group.ducks + group.cows

/-- The main theorem about the difference between legs and twice the heads -/
theorem legs_heads_difference (group : AnimalGroup) 
    (h : group.cows = 18) : 
    totalLegs group - 2 * totalHeads group = 36 := by
  sorry


end NUMINAMATH_CALUDE_legs_heads_difference_l3738_373813


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3738_373809

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15) % (4 * x - 8) = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3738_373809


namespace NUMINAMATH_CALUDE_percentage_relationships_l3738_373821

/-- Given the relationships between a, b, c, d, and e, prove the relative percentages. -/
theorem percentage_relationships (a b c d e : ℝ) 
  (hc_a : c = 0.25 * a)
  (hc_b : c = 0.5 * b)
  (hd_a : d = 0.4 * a)
  (hd_b : d = 0.2 * b)
  (he_d : e = 0.35 * d)
  (he_c : e = 0.15 * c) :
  b = 0.5 * a ∧ c = 0.625 * d ∧ d = (1 / 0.35) * e := by
  sorry


end NUMINAMATH_CALUDE_percentage_relationships_l3738_373821


namespace NUMINAMATH_CALUDE_common_root_condition_l3738_373820

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_common_root_condition_l3738_373820


namespace NUMINAMATH_CALUDE_all_faces_dirty_l3738_373826

/-- Represents the state of a wise man's face -/
inductive FaceState
| Clean
| Dirty

/-- Represents a wise man -/
structure WiseMan :=
  (id : Nat)
  (faceState : FaceState)

/-- Represents the knowledge of a wise man about the others' faces -/
def Knowledge := WiseMan → FaceState

/-- Represents whether a wise man is laughing -/
def isLaughing (w : WiseMan) (k : Knowledge) : Prop :=
  ∃ (other : WiseMan), k other = FaceState.Dirty

/-- The main theorem -/
theorem all_faces_dirty 
  (men : Finset WiseMan) 
  (h_three_men : men.card = 3) 
  (k : WiseMan → Knowledge) 
  (h_correct_knowledge : ∀ (w₁ w₂ : WiseMan), w₁ ≠ w₂ → k w₁ w₂ = w₂.faceState) 
  (h_all_laughing : ∀ (w : WiseMan), w ∈ men → isLaughing w (k w)) :
  ∀ (w : WiseMan), w ∈ men → w.faceState = FaceState.Dirty :=
sorry

end NUMINAMATH_CALUDE_all_faces_dirty_l3738_373826


namespace NUMINAMATH_CALUDE_avery_donation_ratio_l3738_373843

/-- Proves that the ratio of pants to shirts is 2:1 given the conditions of Avery's donation --/
theorem avery_donation_ratio :
  ∀ (pants : ℕ) (shorts : ℕ),
  let shirts := 4
  shorts = pants / 2 →
  shirts + pants + shorts = 16 →
  pants / shirts = 2 := by
sorry

end NUMINAMATH_CALUDE_avery_donation_ratio_l3738_373843


namespace NUMINAMATH_CALUDE_direct_proportion_m_value_l3738_373847

/-- A linear function y = mx + b is a direct proportion if and only if b = 0 -/
def is_direct_proportion (m b : ℝ) : Prop := b = 0

/-- Given that y = mx + (m - 2) is a direct proportion function, prove that m = 2 -/
theorem direct_proportion_m_value (m : ℝ) 
  (h : is_direct_proportion m (m - 2)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_m_value_l3738_373847


namespace NUMINAMATH_CALUDE_base_eight_1263_equals_691_l3738_373873

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_1263_equals_691 :
  base_eight_to_ten [3, 6, 2, 1] = 691 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1263_equals_691_l3738_373873


namespace NUMINAMATH_CALUDE_ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l3738_373800

-- Define a four-digit number
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Define the sum of digits
def SumOfDigits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

-- Theorem statement
theorem ExistsFourDigitNumberDivisibleBy11WithDigitSum10 :
  ∃ n : ℕ, FourDigitNumber n ∧ SumOfDigits n = 10 ∧ n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ExistsFourDigitNumberDivisibleBy11WithDigitSum10_l3738_373800


namespace NUMINAMATH_CALUDE_root_sum_bound_implies_m_range_l3738_373885

theorem root_sum_bound_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + m + 2 = 0 ∧
               x₂^2 - 2*x₂ + m + 2 = 0 ∧
               x₁ ≠ x₂ ∧
               |x₁| + |x₂| ≤ 3) →
  -13/4 ≤ m ∧ m ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_bound_implies_m_range_l3738_373885


namespace NUMINAMATH_CALUDE_wages_calculation_l3738_373881

/-- The wages calculation problem -/
theorem wages_calculation 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (initial_wages : ℚ) 
  (new_workers : ℕ) 
  (new_days : ℕ) 
  (h1 : initial_workers = 15) 
  (h2 : initial_days = 6) 
  (h3 : initial_wages = 9450) 
  (h4 : new_workers = 19) 
  (h5 : new_days = 5) : 
  (initial_wages / (initial_workers * initial_days : ℚ)) * (new_workers * new_days) = 9975 :=
by sorry

end NUMINAMATH_CALUDE_wages_calculation_l3738_373881


namespace NUMINAMATH_CALUDE_chairs_arrangement_l3738_373837

/-- Given a total number of chairs and chairs per row, calculates the number of rows -/
def calculate_rows (total_chairs : ℕ) (chairs_per_row : ℕ) : ℕ :=
  total_chairs / chairs_per_row

/-- Theorem: For 432 chairs arranged in rows of 16, there are 27 rows -/
theorem chairs_arrangement :
  calculate_rows 432 16 = 27 := by
  sorry

end NUMINAMATH_CALUDE_chairs_arrangement_l3738_373837


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3738_373839

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 6 + a 8 = 16)
  (h_a4 : a 4 = 1) :
  a 10 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a10_l3738_373839


namespace NUMINAMATH_CALUDE_hostel_rate_is_15_l3738_373823

/-- Represents the lodging problem for Jimmy's vacation --/
def lodging_problem (hostel_rate : ℚ) : Prop :=
  let hostel_nights : ℕ := 3
  let cabin_nights : ℕ := 2
  let cabin_rate : ℚ := 45
  let cabin_people : ℕ := 3
  let total_cost : ℚ := 75
  (hostel_nights : ℚ) * hostel_rate + 
    (cabin_nights : ℚ) * (cabin_rate / cabin_people) = total_cost

/-- Theorem stating that the hostel rate is $15 per night --/
theorem hostel_rate_is_15 : 
  lodging_problem 15 := by sorry

end NUMINAMATH_CALUDE_hostel_rate_is_15_l3738_373823


namespace NUMINAMATH_CALUDE_percent_equality_l3738_373893

theorem percent_equality (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end NUMINAMATH_CALUDE_percent_equality_l3738_373893


namespace NUMINAMATH_CALUDE_log_expression_equals_one_l3738_373825

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_expression_equals_one :
  (log10 2)^2 + log10 20 * log10 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equals_one_l3738_373825


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3738_373804

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : -4*a - b + 1 = 0) :
  (1/a + 4/b) ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3738_373804


namespace NUMINAMATH_CALUDE_fraction_equality_l3738_373875

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3/7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3738_373875


namespace NUMINAMATH_CALUDE_mean_of_fractions_l3738_373822

theorem mean_of_fractions (a b c : ℚ) (ha : a = 1/2) (hb : b = 1/4) (hc : c = 1/8) :
  (a + b + c) / 3 = 7/24 := by
sorry

end NUMINAMATH_CALUDE_mean_of_fractions_l3738_373822


namespace NUMINAMATH_CALUDE_investment_principal_l3738_373838

/-- 
Given two investments with the same principal and interest rate:
1. Peter's investment yields $815 after 3 years
2. David's investment yields $850 after 4 years
3. Both use simple interest

This theorem proves that the principal invested is $710
-/
theorem investment_principal (P r : ℚ) : 
  (P + P * r * 3 = 815) →
  (P + P * r * 4 = 850) →
  P = 710 := by
sorry

end NUMINAMATH_CALUDE_investment_principal_l3738_373838


namespace NUMINAMATH_CALUDE_gcd_of_45139_34481_4003_l3738_373824

theorem gcd_of_45139_34481_4003 : Nat.gcd 45139 (Nat.gcd 34481 4003) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45139_34481_4003_l3738_373824


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l3738_373878

theorem power_tower_mod_500 : 
  5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l3738_373878


namespace NUMINAMATH_CALUDE_range_of_a_l3738_373844

def A : Set ℝ := {x | 2 < x ∧ x < 8}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a - 2}

theorem range_of_a (a : ℝ) : B a ⊆ A → a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3738_373844


namespace NUMINAMATH_CALUDE_find_m_l3738_373869

theorem find_m : ∃ m : ℤ, (|m| = 2 ∧ m - 2 ≠ 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3738_373869


namespace NUMINAMATH_CALUDE_min_value_expression_l3738_373836

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hsum : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3738_373836


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3738_373812

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that (m + n)^2023 = -1 --/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (m = 3 ∧ n = -4) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3738_373812


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3738_373802

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 5
  let b : ℚ := 2
  (Nat.choose n (n / 2)) * (a ^ (n / 2)) * (b ^ (n / 2)) = 700000 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3738_373802


namespace NUMINAMATH_CALUDE_length_AC_is_12_l3738_373803

/-- Two circles in a plane with given properties -/
structure TwoCircles where
  A : ℝ × ℝ  -- Center of larger circle
  B : ℝ × ℝ  -- Center of smaller circle
  C : ℝ × ℝ  -- Point on line segment AB
  rA : ℝ     -- Radius of larger circle
  rB : ℝ     -- Radius of smaller circle

/-- The theorem to be proved -/
theorem length_AC_is_12 (circles : TwoCircles)
  (h1 : circles.rA = 12)
  (h2 : circles.rB = 7)
  (h3 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ circles.C = (1 - t) • circles.A + t • circles.B)
  (h4 : ‖circles.C - circles.B‖ = circles.rB) :
  ‖circles.A - circles.C‖ = 12 :=
sorry

end NUMINAMATH_CALUDE_length_AC_is_12_l3738_373803


namespace NUMINAMATH_CALUDE_smallest_square_area_l3738_373819

/-- A square in the plane --/
structure RotatedSquare where
  center : ℤ × ℤ
  sideLength : ℝ
  rotation : ℝ

/-- Count the number of lattice points on the boundary of a rotated square --/
def countBoundaryLatticePoints (s : RotatedSquare) : ℕ :=
  sorry

/-- The area of a square --/
def squareArea (s : RotatedSquare) : ℝ :=
  s.sideLength ^ 2

/-- The theorem stating the area of the smallest square meeting the conditions --/
theorem smallest_square_area : 
  ∃ (s : RotatedSquare), 
    (∀ (s' : RotatedSquare), 
      countBoundaryLatticePoints s' = 5 → squareArea s ≤ squareArea s') ∧ 
    countBoundaryLatticePoints s = 5 ∧ 
    squareArea s = 32 :=
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l3738_373819


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3738_373848

theorem rectangle_area_increase (initial_length initial_width : ℝ) 
  (h_positive_length : initial_length > 0)
  (h_positive_width : initial_width > 0) :
  let increase_factor := 1.44
  let side_increase_factor := Real.sqrt increase_factor
  side_increase_factor = 1.2 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3738_373848


namespace NUMINAMATH_CALUDE_group_size_l3738_373882

theorem group_size (B S B_intersect_S : ℕ) 
  (hB : B = 50)
  (hS : S = 70)
  (hIntersect : B_intersect_S = 20) :
  B + S - B_intersect_S = 100 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l3738_373882


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3738_373801

theorem purely_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (m^2 - m) + m * Complex.I
  (∃ (y : ℝ), z = y * Complex.I) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3738_373801


namespace NUMINAMATH_CALUDE_friend_age_problem_l3738_373818

theorem friend_age_problem (A B C : ℕ) 
  (h1 : A - B = 2)
  (h2 : A - C = 5)
  (h3 : A + B + C = 110) :
  A = 39 := by
sorry

end NUMINAMATH_CALUDE_friend_age_problem_l3738_373818


namespace NUMINAMATH_CALUDE_fifty_second_card_is_ace_l3738_373877

-- Define the card ranks
inductive Rank
| King | Queen | Jack | Ten | Nine | Eight | Seven | Six | Five | Four | Three | Two | Ace

-- Define the reversed order of cards
def reversedOrder : List Rank := [
  Rank.King, Rank.Queen, Rank.Jack, Rank.Ten, Rank.Nine, Rank.Eight, Rank.Seven,
  Rank.Six, Rank.Five, Rank.Four, Rank.Three, Rank.Two, Rank.Ace
]

-- Define the number of cards in a cycle
def cardsPerCycle : Nat := 13

-- Define the position we're interested in
def targetPosition : Nat := 52

-- Theorem: The 52nd card in the reversed deck is an Ace
theorem fifty_second_card_is_ace :
  (targetPosition - 1) % cardsPerCycle = cardsPerCycle - 1 →
  reversedOrder[(targetPosition - 1) % cardsPerCycle] = Rank.Ace :=
by
  sorry

#check fifty_second_card_is_ace

end NUMINAMATH_CALUDE_fifty_second_card_is_ace_l3738_373877


namespace NUMINAMATH_CALUDE_sum_remainder_l3738_373898

theorem sum_remainder (x y z : ℕ+) 
  (hx : x ≡ 30 [ZMOD 59])
  (hy : y ≡ 27 [ZMOD 59])
  (hz : z ≡ 4 [ZMOD 59]) :
  (x + y + z : ℤ) ≡ 2 [ZMOD 59] := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l3738_373898


namespace NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3738_373899

theorem circle_y_axis_intersection_sum (h k r : ℝ) : 
  h = 5 → k = -3 → r = 13 →
  let y₁ := k + (r^2 - h^2).sqrt
  let y₂ := k - (r^2 - h^2).sqrt
  y₁ + y₂ = -6 :=
by sorry

end NUMINAMATH_CALUDE_circle_y_axis_intersection_sum_l3738_373899


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l3738_373845

theorem sum_of_real_solutions (b : ℝ) (h : b > 2) :
  ∃ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + y)) = y ∧
  y = (Real.sqrt (4 * b - 3) - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l3738_373845


namespace NUMINAMATH_CALUDE_prob_at_least_two_dice_less_than_10_l3738_373815

/-- The probability of a single 20-sided die showing a number less than 10 -/
def p_less_than_10 : ℚ := 9 / 20

/-- The probability of a single 20-sided die showing a number 10 or above -/
def p_10_or_above : ℚ := 11 / 20

/-- The number of dice rolled -/
def n : ℕ := 5

/-- The probability of exactly k dice showing a number less than 10 -/
def prob_k (k : ℕ) : ℚ :=
  (n.choose k) * (p_less_than_10 ^ k) * (p_10_or_above ^ (n - k))

/-- The probability of at least two dice showing a number less than 10 -/
def prob_at_least_two : ℚ :=
  prob_k 2 + prob_k 3 + prob_k 4 + prob_k 5

theorem prob_at_least_two_dice_less_than_10 :
  prob_at_least_two = 157439 / 20000 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_dice_less_than_10_l3738_373815


namespace NUMINAMATH_CALUDE_roof_ratio_l3738_373828

theorem roof_ratio (length width : ℝ) 
  (area_eq : length * width = 676)
  (diff_eq : length - width = 39) :
  length / width = 4 :=
by sorry

end NUMINAMATH_CALUDE_roof_ratio_l3738_373828


namespace NUMINAMATH_CALUDE_almond_weight_in_mixture_l3738_373895

/-- Given a mixture of nuts where the ratio of almonds to walnuts is 5:1 by weight,
    and the total weight is 140 pounds, the weight of almonds is 116.67 pounds. -/
theorem almond_weight_in_mixture (almond_parts : ℕ) (walnut_parts : ℕ) (total_weight : ℝ) :
  almond_parts = 5 →
  walnut_parts = 1 →
  total_weight = 140 →
  (almond_parts * total_weight) / (almond_parts + walnut_parts) = 116.67 := by
  sorry

end NUMINAMATH_CALUDE_almond_weight_in_mixture_l3738_373895


namespace NUMINAMATH_CALUDE_new_people_total_weight_l3738_373862

/-- Proves that the total weight of five new people joining a group is 270kg -/
theorem new_people_total_weight (initial_count : ℕ) (first_replacement_count : ℕ) (second_replacement_count : ℕ)
  (initial_average_increase : ℝ) (second_average_decrease : ℝ) 
  (first_outgoing_weights : Fin 3 → ℝ) (second_outgoing_total : ℝ) :
  initial_count = 20 ∧ 
  first_replacement_count = 3 ∧
  second_replacement_count = 2 ∧
  initial_average_increase = 2.5 ∧
  second_average_decrease = 1.8 ∧
  first_outgoing_weights 0 = 36 ∧
  first_outgoing_weights 1 = 48 ∧
  first_outgoing_weights 2 = 62 ∧
  second_outgoing_total = 110 →
  (initial_count : ℝ) * initial_average_increase + (first_outgoing_weights 0 + first_outgoing_weights 1 + first_outgoing_weights 2) +
  (second_outgoing_total - (initial_count : ℝ) * second_average_decrease) = 270 := by
  sorry

end NUMINAMATH_CALUDE_new_people_total_weight_l3738_373862


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l3738_373886

theorem quadratic_form_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) ∧ (a + b + c = 228) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l3738_373886


namespace NUMINAMATH_CALUDE_min_disks_for_jamal_files_l3738_373810

/-- Represents the minimum number of disks needed to store files --/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (file_size_a : ℚ) (count_a : ℕ) 
  (file_size_b : ℚ) (count_b : ℕ) 
  (file_size_c : ℚ) : ℕ :=
  sorry

theorem min_disks_for_jamal_files : 
  min_disks 35 2 0.95 5 0.85 15 0.5 = 14 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_jamal_files_l3738_373810


namespace NUMINAMATH_CALUDE_descending_order_inequality_l3738_373835

theorem descending_order_inequality (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  x * y > x * y^2 ∧ x * y^2 > x := by
  sorry

end NUMINAMATH_CALUDE_descending_order_inequality_l3738_373835


namespace NUMINAMATH_CALUDE_first_oil_price_first_oil_price_is_40_l3738_373811

/-- Given two varieties of oil mixed together, calculate the price of the first variety. -/
theorem first_oil_price 
  (second_oil_volume : ℝ) 
  (second_oil_price : ℝ) 
  (mixture_price : ℝ) 
  (first_oil_volume : ℝ) : ℝ :=
  let total_volume := first_oil_volume + second_oil_volume
  let second_oil_total_cost := second_oil_volume * second_oil_price
  let mixture_total_cost := total_volume * mixture_price
  let first_oil_total_cost := mixture_total_cost - second_oil_total_cost
  first_oil_total_cost / first_oil_volume

/-- The price of the first variety of oil is 40, given the specified conditions. -/
theorem first_oil_price_is_40 : 
  first_oil_price 240 60 52 160 = 40 := by
  sorry

end NUMINAMATH_CALUDE_first_oil_price_first_oil_price_is_40_l3738_373811


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3738_373874

def f (a b x : ℚ) : ℚ := a * x^3 - 7 * x^2 + b * x - 6

theorem polynomial_remainder (a b : ℚ) :
  (f a b 2 = -8) ∧ (f a b (-1) = -18) → a = 2/3 ∧ b = 13/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3738_373874


namespace NUMINAMATH_CALUDE_adams_farm_animals_l3738_373888

theorem adams_farm_animals (cows sheep pigs : ℕ) : 
  sheep = 2 * cows →
  pigs = 3 * sheep →
  cows + sheep + pigs = 108 →
  cows = 12 := by
sorry

end NUMINAMATH_CALUDE_adams_farm_animals_l3738_373888


namespace NUMINAMATH_CALUDE_xyz_product_l3738_373840

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 156)
  (h2 : y * (z + x) = 175)
  (h3 : z * (x + y) = 195) :
  x * y * z = 800 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3738_373840


namespace NUMINAMATH_CALUDE_cosine_ratio_equals_one_l3738_373832

theorem cosine_ratio_equals_one :
  (Real.cos (66 * π / 180) * Real.cos (6 * π / 180) + Real.cos (84 * π / 180) * Real.cos (24 * π / 180)) /
  (Real.cos (65 * π / 180) * Real.cos (5 * π / 180) + Real.cos (85 * π / 180) * Real.cos (25 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cosine_ratio_equals_one_l3738_373832


namespace NUMINAMATH_CALUDE_seashells_count_l3738_373866

theorem seashells_count (mary_shells jessica_shells : ℕ) 
  (h1 : mary_shells = 18) 
  (h2 : jessica_shells = 41) : 
  mary_shells + jessica_shells = 59 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l3738_373866


namespace NUMINAMATH_CALUDE_root_sum_theorem_l3738_373857

-- Define the polynomial
def P (k x : ℝ) : ℝ := k * (x^2 - x) + x + 7

-- Define the condition for k1 and k2
def K_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, P k a = 0 ∧ P k b = 0 ∧ a/b + b/a = 3/7

-- State the theorem
theorem root_sum_theorem (k1 k2 : ℝ) :
  K_condition k1 ∧ K_condition k2 →
  k1/k2 + k2/k1 = 322 :=
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l3738_373857


namespace NUMINAMATH_CALUDE_public_transportation_users_l3738_373806

/-- Calculates the number of employees using public transportation -/
theorem public_transportation_users
  (total_employees : ℕ)
  (drive_percentage : ℚ)
  (public_transport_fraction : ℚ)
  (h1 : total_employees = 100)
  (h2 : drive_percentage = 60 / 100)
  (h3 : public_transport_fraction = 1 / 2) :
  ⌊(total_employees : ℚ) * (1 - drive_percentage) * public_transport_fraction⌋ = 20 := by
  sorry

end NUMINAMATH_CALUDE_public_transportation_users_l3738_373806


namespace NUMINAMATH_CALUDE_forty_percent_value_l3738_373884

theorem forty_percent_value (x : ℝ) : (0.6 * x = 240) → (0.4 * x = 160) := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_value_l3738_373884


namespace NUMINAMATH_CALUDE_subtraction_result_l3738_373816

theorem subtraction_result : 3.57 - 2.15 = 1.42 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3738_373816


namespace NUMINAMATH_CALUDE_new_train_distance_calculation_l3738_373871

/-- The distance traveled by the new train given the distance traveled by the old train and the percentage increase -/
def new_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem: Given that a new train travels 30% farther than an old train in the same time,
    and the old train travels 300 miles, the new train travels 390 miles. -/
theorem new_train_distance_calculation :
  new_train_distance 300 0.3 = 390 := by
  sorry

#eval new_train_distance 300 0.3

end NUMINAMATH_CALUDE_new_train_distance_calculation_l3738_373871


namespace NUMINAMATH_CALUDE_total_molecular_weight_l3738_373808

-- Define atomic weights
def carbon_weight : ℝ := 12.01
def hydrogen_weight : ℝ := 1.008
def oxygen_weight : ℝ := 16.00

-- Define molecular formulas
def ascorbic_acid_carbon : ℕ := 6
def ascorbic_acid_hydrogen : ℕ := 8
def ascorbic_acid_oxygen : ℕ := 6

def citric_acid_carbon : ℕ := 6
def citric_acid_hydrogen : ℕ := 8
def citric_acid_oxygen : ℕ := 7

-- Define number of moles
def ascorbic_acid_moles : ℕ := 7
def citric_acid_moles : ℕ := 5

-- Calculate molecular weights
def ascorbic_acid_weight : ℝ :=
  (ascorbic_acid_carbon * carbon_weight) +
  (ascorbic_acid_hydrogen * hydrogen_weight) +
  (ascorbic_acid_oxygen * oxygen_weight)

def citric_acid_weight : ℝ :=
  (citric_acid_carbon * carbon_weight) +
  (citric_acid_hydrogen * hydrogen_weight) +
  (citric_acid_oxygen * oxygen_weight)

-- Theorem statement
theorem total_molecular_weight :
  (ascorbic_acid_moles * ascorbic_acid_weight) +
  (citric_acid_moles * citric_acid_weight) = 2193.488 :=
by sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l3738_373808


namespace NUMINAMATH_CALUDE_carls_cupcakes_l3738_373860

/-- Carl's cupcake selling problem -/
theorem carls_cupcakes (days : ℕ) (cupcakes_per_day : ℕ) (cupcakes_for_bonnie : ℕ) : 
  days = 2 → cupcakes_per_day = 60 → cupcakes_for_bonnie = 24 →
  days * cupcakes_per_day + cupcakes_for_bonnie = 144 := by
  sorry

end NUMINAMATH_CALUDE_carls_cupcakes_l3738_373860


namespace NUMINAMATH_CALUDE_lilias_peaches_l3738_373876

/-- Represents the problem of calculating how many peaches Lilia sold to her friends. -/
theorem lilias_peaches (total_peaches : ℕ) (friends_price : ℚ) (relatives_peaches : ℕ) (relatives_price : ℚ) (kept_peaches : ℕ) (total_earned : ℚ) (total_sold : ℕ) :
  total_peaches = 15 →
  friends_price = 2 →
  relatives_peaches = 4 →
  relatives_price = 5/4 →
  kept_peaches = 1 →
  total_earned = 25 →
  total_sold = 14 →
  ∃ (friends_peaches : ℕ), 
    friends_peaches + relatives_peaches + kept_peaches = total_peaches ∧
    friends_peaches * friends_price + relatives_peaches * relatives_price = total_earned ∧
    friends_peaches = 10 :=
by sorry

end NUMINAMATH_CALUDE_lilias_peaches_l3738_373876


namespace NUMINAMATH_CALUDE_equation_solution_l3738_373879

theorem equation_solution : 
  {x : ℝ | (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2} = {-1/2, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3738_373879


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3738_373833

theorem complex_magnitude_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3738_373833


namespace NUMINAMATH_CALUDE_veg_eaters_count_l3738_373864

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ

/-- Theorem stating that the number of people who eat veg in the family is 20 -/
theorem veg_eaters_count (family : FamilyEatingHabits)
  (h1 : family.only_veg = 11)
  (h2 : family.only_non_veg = 6)
  (h3 : family.both_veg_and_non_veg = 9) :
  family.only_veg + family.both_veg_and_non_veg = 20 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l3738_373864


namespace NUMINAMATH_CALUDE_twelve_mile_ride_cost_l3738_373863

/-- Calculates the cost of a taxi ride given the specified conditions -/
def taxiRideCost (baseFare mileRate discountThreshold discountRate miles : ℚ) : ℚ :=
  let totalBeforeDiscount := baseFare + mileRate * miles
  if miles > discountThreshold then
    totalBeforeDiscount * (1 - discountRate)
  else
    totalBeforeDiscount

theorem twelve_mile_ride_cost :
  taxiRideCost 2 (30/100) 10 (10/100) 12 = 504/100 := by
  sorry

#eval taxiRideCost 2 (30/100) 10 (10/100) 12

end NUMINAMATH_CALUDE_twelve_mile_ride_cost_l3738_373863


namespace NUMINAMATH_CALUDE_melanie_dimes_l3738_373868

theorem melanie_dimes (x : ℕ) : x + 8 + 4 = 19 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l3738_373868


namespace NUMINAMATH_CALUDE_function_properties_l3738_373830

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem function_properties (φ : ℝ) (h : φ > 0) :
  (∀ x, f x φ = f (x + π) φ) ∧ 
  (∃ φ', ∀ x, f x φ' = f (-x) φ') ∧
  (∀ x ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), ∀ y ∈ Set.Icc (π - φ/2) (3*π/2 - φ/2), 
    x < y → f x φ > f y φ) ∧
  (∀ x, f x φ = Real.cos (2 * (x - φ/2))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3738_373830


namespace NUMINAMATH_CALUDE_expression_evaluation_l3738_373872

theorem expression_evaluation : (3 * 10^9) / (6 * 10^5) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3738_373872


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l3738_373870

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² = a² + ac + c², then the measure of angle B is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 = a^2 + a*c + c^2) :
  let angle_B := Real.arccos (-1/2)
  angle_B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l3738_373870


namespace NUMINAMATH_CALUDE_min_cost_is_80_yuan_l3738_373827

/-- Represents the swimming trip problem -/
structure SwimmingTripProblem where
  card_cost : ℕ            -- Cost of each swim card in yuan
  students : ℕ             -- Number of students
  swims_per_student : ℕ    -- Number of swims each student needs
  bus_cost : ℕ             -- Cost of bus rental per trip in yuan

/-- Calculates the minimum cost per student for the swimming trip -/
def min_cost_per_student (problem : SwimmingTripProblem) : ℚ :=
  let total_swims := problem.students * problem.swims_per_student
  let cards := 8  -- Optimal number of cards to buy
  let trips := total_swims / cards
  let total_cost := problem.card_cost * cards + problem.bus_cost * trips
  (total_cost : ℚ) / problem.students

/-- Theorem stating that the minimum cost per student is 80 yuan -/
theorem min_cost_is_80_yuan (problem : SwimmingTripProblem) 
    (h1 : problem.card_cost = 240)
    (h2 : problem.students = 48)
    (h3 : problem.swims_per_student = 8)
    (h4 : problem.bus_cost = 40) : 
  min_cost_per_student problem = 80 := by
  sorry

#eval min_cost_per_student { card_cost := 240, students := 48, swims_per_student := 8, bus_cost := 40 }

end NUMINAMATH_CALUDE_min_cost_is_80_yuan_l3738_373827


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3738_373807

-- Define variables
variable (x y : ℝ)

-- Theorem for the first expression
theorem simplify_expression_1 : (2*x + 1) - (3 - x) = 3*x - 2 := by sorry

-- Theorem for the second expression
theorem simplify_expression_2 : x^2*y - (2*x*y^2 - 5*x^2*y) + 3*x*y^2 - y^3 = 6*x^2*y + x*y^2 - y^3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3738_373807


namespace NUMINAMATH_CALUDE_triangle_line_equations_l3738_373880

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  CM : ℝ → ℝ → Prop
  BH : ℝ → ℝ → Prop

/-- The given triangle satisfies the problem conditions -/
def given_triangle : Triangle where
  A := (5, 1)
  CM := fun x y ↦ 2 * x - y - 5 = 0
  BH := fun x y ↦ x - 2 * y - 5 = 0

/-- Line equation represented as ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem stating the equations of line BC and its symmetric line -/
theorem triangle_line_equations (t : Triangle) 
  (h : t = given_triangle) : 
  ∃ (BC symmetric_BC : LineEquation),
    (BC.a = 6 ∧ BC.b = -5 ∧ BC.c = -9) ∧
    (symmetric_BC.a = 38 ∧ symmetric_BC.b = -9 ∧ symmetric_BC.c = -125) := by
  sorry

end NUMINAMATH_CALUDE_triangle_line_equations_l3738_373880


namespace NUMINAMATH_CALUDE_absolute_value_of_w_l3738_373889

theorem absolute_value_of_w (w : ℂ) (h : w^2 = -48 + 14*I) : Complex.abs w = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_w_l3738_373889


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3738_373854

theorem necessary_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 2 * (1 - m) * x + 3 > 0) →
  (m > 0 ∧ ∃ m' > 0, ¬(∀ x : ℝ, (m' - 1) * x^2 + 2 * (1 - m') * x + 3 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3738_373854


namespace NUMINAMATH_CALUDE_age_difference_l3738_373892

/-- Given three people A, B, and C, where C is 13 years younger than A,
    prove that the sum of ages of A and B is 13 years more than the sum of ages of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 13) :
  A + B - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3738_373892


namespace NUMINAMATH_CALUDE_sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l3738_373853

/-- Given two triangles ABC and ADE with specified areas and coordinates for points B, C, D, and E,
    prove that the sum of all possible x-coordinates of point A is 400. -/
theorem sum_of_possible_x_coordinates_of_A : ℝ → Prop :=
  fun sum_x =>
    ∀ (A B C D E : ℝ × ℝ)
      (area_ABC area_ADE : ℝ),
    B = (0, 0) →
    C = (200, 0) →
    D = (600, 400) →
    E = (610, 410) →
    area_ABC = 3000 →
    area_ADE = 6000 →
    (∃ (x₁ x₂ : ℝ), 
      (A.1 = x₁ ∨ A.1 = x₂) ∧ 
      sum_x = x₁ + x₂) →
    sum_x = 400

/-- Proof of the theorem -/
theorem sum_of_possible_x_coordinates_of_A_is_400 :
  sum_of_possible_x_coordinates_of_A 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l3738_373853


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3738_373856

theorem x_intercepts_count (x : ℝ) : 
  ∃! x, (x - 5) * (x^2 + x + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3738_373856


namespace NUMINAMATH_CALUDE_bobbys_shoe_cost_bobbys_shoe_cost_is_968_l3738_373859

/-- Calculates the total cost of Bobby's handmade shoes -/
theorem bobbys_shoe_cost (mold_cost : ℝ) (labor_rate : ℝ) (work_hours : ℝ) 
  (labor_discount : ℝ) (materials_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_labor_cost := labor_rate * work_hours * labor_discount
  let total_before_tax := mold_cost + discounted_labor_cost + materials_cost
  let tax := total_before_tax * tax_rate
  let total_with_tax := total_before_tax + tax
  
  total_with_tax

theorem bobbys_shoe_cost_is_968 :
  bobbys_shoe_cost 250 75 8 0.8 150 0.1 = 968 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoe_cost_bobbys_shoe_cost_is_968_l3738_373859


namespace NUMINAMATH_CALUDE_expected_sides_theorem_expected_sides_rectangle_limit_l3738_373851

/-- The expected number of sides of a randomly selected polygon after cuts -/
def expected_sides (n k : ℕ) : ℚ :=
  (n + 4 * k) / (k + 1)

/-- Theorem: The expected number of sides of a randomly selected polygon
    after k cuts, starting with an n-sided polygon, is (n + 4k) / (k + 1) -/
theorem expected_sides_theorem (n k : ℕ) :
  expected_sides n k = (n + 4 * k) / (k + 1) := by
  sorry

/-- Corollary: For a rectangle (n = 4) and large k, the expectation approaches 4 -/
theorem expected_sides_rectangle_limit :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides 4 k - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_sides_theorem_expected_sides_rectangle_limit_l3738_373851


namespace NUMINAMATH_CALUDE_cubic_one_real_root_l3738_373891

/-- A cubic equation with coefficients a and b -/
def cubic_equation (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- Condition for the cubic equation to have only one real root -/
def has_one_real_root (a b : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a b x = 0

theorem cubic_one_real_root :
  (has_one_real_root (-3) (-3)) ∧
  (∀ b > 2, has_one_real_root (-3) b) ∧
  (has_one_real_root 0 2) :=
sorry

end NUMINAMATH_CALUDE_cubic_one_real_root_l3738_373891


namespace NUMINAMATH_CALUDE_right_triangle_probability_l3738_373814

/-- A 3x3 grid of nine unit squares -/
structure Grid :=
  (vertices : Fin 16 → ℝ × ℝ)

/-- Three vertices selected from the grid -/
structure SelectedVertices :=
  (v1 v2 v3 : Fin 16)

/-- Predicate to check if three vertices form a right triangle -/
def is_right_triangle (g : Grid) (sv : SelectedVertices) : Prop :=
  sorry

/-- The total number of ways to select three vertices from 16 -/
def total_selections : ℕ := Nat.choose 16 3

/-- The number of right triangles that can be formed -/
def right_triangle_count (g : Grid) : ℕ :=
  sorry

/-- The main theorem stating the probability -/
theorem right_triangle_probability (g : Grid) :
  (right_triangle_count g : ℚ) / total_selections = 5 / 14 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_probability_l3738_373814


namespace NUMINAMATH_CALUDE_cubic_system_solution_l3738_373896

theorem cubic_system_solution :
  ∃ (x y z : ℝ), 
    x + y + z = 1 ∧
    x^2 + y^2 + z^2 = 1 ∧
    x^3 + y^3 + z^3 = 89/125 ∧
    ((x = 2/5 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = 2/5 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 - Real.sqrt 33)/10) ∨
     (x = (3 + Real.sqrt 33)/10 ∧ y = (3 - Real.sqrt 33)/10 ∧ z = 2/5) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = 2/5 ∧ z = (3 + Real.sqrt 33)/10) ∨
     (x = (3 - Real.sqrt 33)/10 ∧ y = (3 + Real.sqrt 33)/10 ∧ z = 2/5)) :=
by
  sorry


end NUMINAMATH_CALUDE_cubic_system_solution_l3738_373896


namespace NUMINAMATH_CALUDE_price_increase_proof_l3738_373841

theorem price_increase_proof (x : ℝ) : 
  (1 + x)^2 = 1.44 → x = 0.2 := by sorry

end NUMINAMATH_CALUDE_price_increase_proof_l3738_373841


namespace NUMINAMATH_CALUDE_complex_number_conditions_l3738_373865

theorem complex_number_conditions (z : ℂ) : 
  (∃ (a : ℝ), a > 0 ∧ (z - 3*I) / (z + I) = -a) ∧ 
  (∃ (b : ℝ), b ≠ 0 ∧ (z - 3) / (z + 1) = b*I) → 
  z = -1 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_conditions_l3738_373865


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3738_373849

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ (2 * a * b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3738_373849


namespace NUMINAMATH_CALUDE_angle_B_is_60_degrees_side_c_and_area_l3738_373842

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a < t.b ∧ t.b < t.c ∧
  Real.sqrt 3 * t.a = 2 * t.b * Real.sin t.A

-- Theorem 1: Prove that angle B is 60 degrees
theorem angle_B_is_60_degrees (t : Triangle) (h : validTriangle t) :
  t.B = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove side c length and area when a = 2 and b = √7
theorem side_c_and_area (t : Triangle) (h : validTriangle t)
  (ha : t.a = 2) (hb : t.b = Real.sqrt 7) :
  t.c = 3 ∧ (1/2 * t.a * t.c * Real.sin t.B) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_60_degrees_side_c_and_area_l3738_373842


namespace NUMINAMATH_CALUDE_population_average_age_l3738_373894

theorem population_average_age 
  (ratio_women_men : ℚ) 
  (avg_age_women : ℚ) 
  (avg_age_men : ℚ) 
  (h_ratio : ratio_women_men = 7 / 5) 
  (h_women_age : avg_age_women = 38) 
  (h_men_age : avg_age_men = 36) : 
  (ratio_women_men * avg_age_women + avg_age_men) / (ratio_women_men + 1) = 37 + 1/6 := by
sorry

end NUMINAMATH_CALUDE_population_average_age_l3738_373894


namespace NUMINAMATH_CALUDE_product_of_1101_base2_and_102_base3_l3738_373817

def base2_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 2^i) 0

def base3_to_dec (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, b) acc => acc + b * 3^i) 0

theorem product_of_1101_base2_and_102_base3 :
  let n1 := base2_to_dec [1, 0, 1, 1]
  let n2 := base3_to_dec [2, 0, 1]
  n1 * n2 = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1101_base2_and_102_base3_l3738_373817


namespace NUMINAMATH_CALUDE_sustainable_tree_planting_l3738_373834

theorem sustainable_tree_planting (trees_first_half trees_second_half trees_to_plant : ℕ) :
  trees_first_half = 200 →
  trees_second_half = 300 →
  trees_to_plant = 1500 →
  (trees_to_plant : ℚ) / (trees_first_half + trees_second_half : ℚ) = 3 := by
sorry

end NUMINAMATH_CALUDE_sustainable_tree_planting_l3738_373834


namespace NUMINAMATH_CALUDE_deli_sandwich_count_l3738_373850

-- Define the types of sandwich components
structure SandwichComponents where
  breads : Nat
  meats : Nat
  cheeses : Nat

-- Define the forbidden combinations
structure ForbiddenCombinations where
  ham_cheddar : Nat
  white_chicken : Nat
  turkey_swiss : Nat

-- Define the function to calculate the number of possible sandwiches
def calculate_sandwiches (components : SandwichComponents) (forbidden : ForbiddenCombinations) : Nat :=
  components.breads * components.meats * components.cheeses - 
  (forbidden.ham_cheddar + forbidden.white_chicken + forbidden.turkey_swiss)

-- Theorem statement
theorem deli_sandwich_count :
  let components := SandwichComponents.mk 5 7 6
  let forbidden := ForbiddenCombinations.mk 5 6 5
  calculate_sandwiches components forbidden = 194 := by
  sorry


end NUMINAMATH_CALUDE_deli_sandwich_count_l3738_373850


namespace NUMINAMATH_CALUDE_min_unique_score_above_90_l3738_373831

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 5 * c - 2 * w

/-- Represents the total number of questions in the exam -/
def total_questions : ℕ := 35

/-- Theorem stating that 91 is the minimum score above 90 with a unique solution -/
theorem min_unique_score_above_90 :
  ∀ s : ℕ, s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  s ≥ 91 :=
sorry

end NUMINAMATH_CALUDE_min_unique_score_above_90_l3738_373831
