import Mathlib

namespace NUMINAMATH_CALUDE_ordered_pairs_count_l2734_273497

theorem ordered_pairs_count : ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
  p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 32) (Finset.product (Finset.range 33) (Finset.range 33))).card ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_count_l2734_273497


namespace NUMINAMATH_CALUDE_distance_in_scientific_notation_l2734_273420

/-- Given a distance of 14,000,000 meters between two mountain peaks,
    prove that its representation in scientific notation is 1.4 × 10^7 -/
theorem distance_in_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    14000000 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧ 
    a = 1.4 ∧ 
    n = 7 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_scientific_notation_l2734_273420


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_equals_one_l2734_273464

theorem purely_imaginary_iff_a_equals_one (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_equals_one_l2734_273464


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l2734_273449

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (faces : Fin 4)
  (vertices : Fin 4)
  (edges : Fin 6)

/-- A stripe configuration on a tetrahedron -/
def StripeConfiguration := Tetrahedron → Fin 3

/-- Predicate for a continuous stripe encircling the tetrahedron -/
def IsContinuousStripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def TotalConfigurations : ℕ := 3^4

/-- The number of stripe configurations that result in a continuous stripe -/
def ContinuousStripeConfigurations : ℕ := 2^4

/-- The probability of a continuous stripe encircling the tetrahedron -/
def ProbabilityContinuousStripe : ℚ :=
  ContinuousStripeConfigurations / TotalConfigurations

theorem continuous_stripe_probability :
  ProbabilityContinuousStripe = 16 / 81 :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l2734_273449


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2734_273414

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3.5 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := 3
  }
  tetrahedronVolume t = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2734_273414


namespace NUMINAMATH_CALUDE_dave_tray_capacity_l2734_273465

theorem dave_tray_capacity (trays_table1 trays_table2 num_trips : ℕ) 
  (h1 : trays_table1 = 17)
  (h2 : trays_table2 = 55)
  (h3 : num_trips = 8) :
  (trays_table1 + trays_table2) / num_trips = 9 := by
  sorry

#check dave_tray_capacity

end NUMINAMATH_CALUDE_dave_tray_capacity_l2734_273465


namespace NUMINAMATH_CALUDE_equation_solvable_by_factoring_l2734_273472

/-- The equation to be solved -/
def equation (x : ℝ) : Prop := (5*x - 1)^2 = 3*(5*x - 1)

/-- Factoring method can be applied if the equation can be written as a product of factors equal to zero -/
def factoring_method_applicable (f : ℝ → Prop) : Prop :=
  ∃ g h : ℝ → ℝ, ∀ x, f x ↔ g x * h x = 0

/-- The theorem stating that the given equation can be solved using the factoring method -/
theorem equation_solvable_by_factoring : factoring_method_applicable equation := by
  sorry

end NUMINAMATH_CALUDE_equation_solvable_by_factoring_l2734_273472


namespace NUMINAMATH_CALUDE_conical_flask_height_l2734_273495

/-- The height of a conical flask given water depths in two positions -/
theorem conical_flask_height (h : ℝ) : 
  (h > 0) →  -- The height is positive
  (h^3 - (h-1)^3 = 8) →  -- Volume equation derived from the two water depths
  h = 1/2 + Real.sqrt 93 / 6 := by
sorry

end NUMINAMATH_CALUDE_conical_flask_height_l2734_273495


namespace NUMINAMATH_CALUDE_triangle_side_length_l2734_273460

theorem triangle_side_length (B C BDC : Real) (BD CD : Real) :
  B = 30 * π / 180 →
  C = 45 * π / 180 →
  BDC = 150 * π / 180 →
  BD = 5 →
  CD = 5 →
  ∃ AB : Real, AB = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2734_273460


namespace NUMINAMATH_CALUDE_circle_radius_is_5_l2734_273405

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 - 21 = 0

/-- The radius of the circle defined by circle_equation -/
def circle_radius : ℝ := 5

/-- Theorem: The radius of the circle defined by circle_equation is 5 -/
theorem circle_radius_is_5 : 
  ∀ x y : ℝ, circle_equation x y → 
  ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_5_l2734_273405


namespace NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l2734_273477

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 7

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Theorem statement
theorem soccer_team_lineup_combinations :
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup)) = 1980 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_combinations_l2734_273477


namespace NUMINAMATH_CALUDE_expression_value_l2734_273463

theorem expression_value : 3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2734_273463


namespace NUMINAMATH_CALUDE_simplify_expression_l2734_273481

theorem simplify_expression : (0.3 * 0.2) / (0.4 * 0.5) - (0.1 * 0.6) = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2734_273481


namespace NUMINAMATH_CALUDE_distinct_arrangements_eq_360_l2734_273462

/-- Represents the number of members in the committee -/
def total_members : ℕ := 10

/-- Represents the number of women (rocking chairs) -/
def women : ℕ := 7

/-- Represents the number of men (stools) -/
def men : ℕ := 2

/-- Represents the number of children (benches) -/
def children : ℕ := 1

/-- Calculates the number of distinct arrangements of seats -/
def distinct_arrangements : ℕ := total_members * (men * (men + 1) / 2)

/-- Theorem stating that the number of distinct arrangements is 360 -/
theorem distinct_arrangements_eq_360 : distinct_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_eq_360_l2734_273462


namespace NUMINAMATH_CALUDE_problem_solution_l2734_273443

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - m < 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

-- Main theorem
theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - 2 < 0) ∧
  (a < 1) ∧
  (A a ⊆ B) ∧
  (A a ≠ B) →
  (B = Set.Ioi 2) ∧
  (2/3 ≤ a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2734_273443


namespace NUMINAMATH_CALUDE_min_disks_required_l2734_273469

/-- Represents the number of files of each size --/
structure FileDistribution :=
  (large : Nat)  -- 0.9 MB files
  (medium : Nat) -- 0.8 MB files
  (small : Nat)  -- 0.5 MB files

/-- Represents the problem setup --/
def diskProblem : FileDistribution :=
  { large := 5
    medium := 15
    small := 20 }

/-- Disk capacity in MB --/
def diskCapacity : Rat := 2

/-- File sizes in MB --/
def largeFileSize : Rat := 9/10
def mediumFileSize : Rat := 4/5
def smallFileSize : Rat := 1/2

/-- The theorem stating the minimum number of disks required --/
theorem min_disks_required (fd : FileDistribution) 
  (h1 : fd.large + fd.medium + fd.small = 40)
  (h2 : fd = diskProblem) :
  ∃ (n : Nat), n = 18 ∧ 
  (∀ (m : Nat), m < n → 
    m * diskCapacity < 
    fd.large * largeFileSize + fd.medium * mediumFileSize + fd.small * smallFileSize) :=
  sorry

end NUMINAMATH_CALUDE_min_disks_required_l2734_273469


namespace NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_min_value_achieved_l2734_273474

theorem min_value_sqrt_reciprocal (x : ℝ) (h : x > 0) : 
  3 * Real.sqrt (2 * x) + 4 / x ≥ 8 := by
  sorry

theorem min_value_achieved (x : ℝ) (h : x > 0) : 
  ∃ y > 0, 3 * Real.sqrt (2 * y) + 4 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_reciprocal_min_value_achieved_l2734_273474


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l2734_273461

theorem pure_imaginary_implies_x_equals_one (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x + 1) * I = (0 : ℂ) + y * I) ∧ y ≠ 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_x_equals_one_l2734_273461


namespace NUMINAMATH_CALUDE_system_solution_l2734_273476

theorem system_solution : 
  ∀ (a b c d : ℝ), 
    a + c = -7 ∧ 
    a * c + b + d = 18 ∧ 
    a * d + b * c = -22 ∧ 
    b * d = 12 → 
    ((a = -5 ∧ b = 6 ∧ c = -2 ∧ d = 2) ∨ 
     (a = -2 ∧ b = 2 ∧ c = -5 ∧ d = 6)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2734_273476


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l2734_273411

/-- Calculates the local tax deduction in cents per hour given an hourly wage in dollars and a tax rate percentage. -/
def local_tax_deduction (hourly_wage : ℚ) (tax_rate_percent : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate_percent / 100)

/-- Theorem: Given Alicia's hourly wage of $25 and a local tax rate of 2%, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem alicia_tax_deduction :
  local_tax_deduction 25 2 = 50 := by
  sorry

#eval local_tax_deduction 25 2

end NUMINAMATH_CALUDE_alicia_tax_deduction_l2734_273411


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l2734_273440

/-- The midpoint coordinates of the line segment cut by the parabola y^2 = 4x from the line y = x - 1 are (3, 2). -/
theorem midpoint_coordinates (x y : ℝ) : 
  y^2 = 4*x ∧ y = x - 1 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 1)^2 = 4*x1 ∧ 
    (x2^2 - 1)^2 = 4*x2 ∧
    ((x1 + x2) / 2 = 3 ∧ ((x1 - 1) + (x2 - 1)) / 2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l2734_273440


namespace NUMINAMATH_CALUDE_fourth_root_cubed_l2734_273412

theorem fourth_root_cubed (x : ℝ) : (x^(1/4))^3 = 729 → x = 6561 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_cubed_l2734_273412


namespace NUMINAMATH_CALUDE_ice_cream_pudding_cost_difference_l2734_273468

-- Define the quantities and prices
def ice_cream_quantity : ℕ := 15
def pudding_quantity : ℕ := 5
def ice_cream_price : ℕ := 5
def pudding_price : ℕ := 2

-- Define the theorem
theorem ice_cream_pudding_cost_difference :
  (ice_cream_quantity * ice_cream_price) - (pudding_quantity * pudding_price) = 65 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_pudding_cost_difference_l2734_273468


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2734_273437

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2734_273437


namespace NUMINAMATH_CALUDE_car_trip_mpg_l2734_273419

/-- Calculates the average miles per gallon for a trip given odometer readings and gas fill-ups --/
def average_mpg (initial_reading : ℕ) (final_reading : ℕ) (gas_used : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / gas_used

theorem car_trip_mpg : 
  let initial_reading : ℕ := 48500
  let second_reading : ℕ := 48800
  let final_reading : ℕ := 49350
  let first_fillup : ℕ := 8
  let second_fillup : ℕ := 10
  let third_fillup : ℕ := 15
  let total_gas_used : ℕ := second_fillup + third_fillup
  average_mpg initial_reading final_reading total_gas_used = 34 := by
  sorry

#eval average_mpg 48500 49350 25

end NUMINAMATH_CALUDE_car_trip_mpg_l2734_273419


namespace NUMINAMATH_CALUDE_alex_total_cost_l2734_273446

/-- Calculates the total cost of a cell phone plan given the usage and rates. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℚ) : ℚ :=
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  base_cost + (text_cost * texts_sent) + (extra_minute_cost * extra_minutes)

/-- Proves that Alex's total cost is $109.00 given the specified plan and usage. -/
theorem alex_total_cost : 
  calculate_total_cost 25 25 0.08 0.15 150 33 = 109 := by
  sorry

end NUMINAMATH_CALUDE_alex_total_cost_l2734_273446


namespace NUMINAMATH_CALUDE_value_of_expression_l2734_273433

theorem value_of_expression (x y : ℚ) (hx : x = 2/3) (hy : y = 3/2) :
  (1/3) * x^5 * y^6 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2734_273433


namespace NUMINAMATH_CALUDE_investment_percentage_l2734_273490

/-- Proves that given the investment conditions, the percentage of the other investment is 7% -/
theorem investment_percentage (total_investment : ℝ) (investment_at_8_percent : ℝ) (total_interest : ℝ)
  (h1 : total_investment = 22000)
  (h2 : investment_at_8_percent = 17000)
  (h3 : total_interest = 1710) :
  (total_interest - investment_at_8_percent * 0.08) / (total_investment - investment_at_8_percent) = 0.07 := by
  sorry


end NUMINAMATH_CALUDE_investment_percentage_l2734_273490


namespace NUMINAMATH_CALUDE_fourth_root_equality_l2734_273480

theorem fourth_root_equality (M : ℝ) (h : M > 1) :
  (M^2 * (M * M^(1/4))^(1/3))^(1/4) = M^(29/48) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equality_l2734_273480


namespace NUMINAMATH_CALUDE_conference_attendees_l2734_273425

theorem conference_attendees (men : ℕ) : 
  (men : ℝ) * 0.1 + 300 * 0.6 + 500 * 0.7 = (men + 300 + 500 : ℝ) * (1 - 0.5538461538461539) →
  men = 500 := by
sorry

end NUMINAMATH_CALUDE_conference_attendees_l2734_273425


namespace NUMINAMATH_CALUDE_cos_4theta_from_exp_l2734_273418

theorem cos_4theta_from_exp (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (1 - Complex.I * Real.sqrt 8) / 3 → 
  Real.cos (4 * θ) = 17 / 81 := by
sorry

end NUMINAMATH_CALUDE_cos_4theta_from_exp_l2734_273418


namespace NUMINAMATH_CALUDE_cos_difference_l2734_273467

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_l2734_273467


namespace NUMINAMATH_CALUDE_abracadabra_anagrams_l2734_273421

/-- The number of anagrams of a word with given letter frequencies -/
def anagrams (total : ℕ) (frequencies : List ℕ) : ℕ :=
  Nat.factorial total / (frequencies.map Nat.factorial).prod

/-- The frequencies of letters in "ABRACADABRA" -/
def abracadabra_frequencies : List ℕ := [5, 2, 2, 1, 1]

/-- The total number of letters in "ABRACADABRA" -/
def abracadabra_total : ℕ := 11

theorem abracadabra_anagrams :
  anagrams abracadabra_total abracadabra_frequencies = 83160 := by
  sorry

end NUMINAMATH_CALUDE_abracadabra_anagrams_l2734_273421


namespace NUMINAMATH_CALUDE_range_of_m_l2734_273484

/-- Given sets A and B, where B is a subset of A, prove that m ≤ 0 -/
theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
  let B : Set ℝ := {x | m + 1 < x ∧ x < 2*m + 3}
  B ⊆ A → m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2734_273484


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2734_273482

theorem cubic_equation_solution (p q : ℝ) : 
  (3 * p^2 - 5 * p - 21 = 0) → 
  (3 * q^2 - 5 * q - 21 = 0) → 
  p ≠ q →
  (9 * p^3 - 9 * q^3) / (p - q) = 88 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2734_273482


namespace NUMINAMATH_CALUDE_expression_simplification_l2734_273489

theorem expression_simplification (p : ℤ) :
  ((7 * p + 2) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2734_273489


namespace NUMINAMATH_CALUDE_smallest_valid_subtrahend_l2734_273457

def is_valid_subtrahend (n : ℕ) : Prop :=
  ∃ m : ℕ, m = 134 - n ∧ 
            m % 3 = 0 ∧ 
            m % 5 = 0 ∧ 
            m % 2 = 0

theorem smallest_valid_subtrahend :
  is_valid_subtrahend 14 ∧ 
  ∀ k : ℕ, k < 14 → ¬ is_valid_subtrahend k :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_subtrahend_l2734_273457


namespace NUMINAMATH_CALUDE_finite_squared_nilpotent_matrices_l2734_273434

/-- Given a 3x3 matrix A with real entries such that A^4 = 0, 
    the set of all possible A^2 matrices is finite. -/
theorem finite_squared_nilpotent_matrices 
  (A : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : A ^ 4 = 0) : 
  Set.Finite {B : Matrix (Fin 3) (Fin 3) ℝ | ∃ (A : Matrix (Fin 3) (Fin 3) ℝ), A ^ 4 = 0 ∧ B = A ^ 2} :=
sorry

end NUMINAMATH_CALUDE_finite_squared_nilpotent_matrices_l2734_273434


namespace NUMINAMATH_CALUDE_range_of_a_l2734_273459

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 > 2*x - 1) : a ≠ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2734_273459


namespace NUMINAMATH_CALUDE_geometric_progressions_terms_l2734_273406

theorem geometric_progressions_terms (a₁ b₁ q₁ q₂ : ℚ) (sum : ℚ) :
  a₁ = 20 →
  q₁ = 3/4 →
  b₁ = 4 →
  q₂ = 2/3 →
  sum = 158.75 →
  (∃ n : ℕ, sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) →
  (∃ n : ℕ, n = 7 ∧
    sum = (a₁ * b₁) * (1 - (q₁ * q₂)^n) / (1 - q₁ * q₂)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progressions_terms_l2734_273406


namespace NUMINAMATH_CALUDE_largest_base_not_18_l2734_273499

/-- Represents a number in a given base as a list of digits -/
def Digits := List Nat

/-- Calculates the sum of digits -/
def sum_of_digits (digits : Digits) : Nat :=
  digits.sum

/-- Converts a number to its representation in a given base -/
def to_base (n : Nat) (base : Nat) : Digits :=
  sorry

theorem largest_base_not_18 :
  ∃ (max_base : Nat),
    (sum_of_digits (to_base (12^3) 10) = 18) ∧
    (12^3 = 1728) ∧
    (∀ b > 10, to_base (12^3) b = to_base 1728 b) ∧
    (to_base (12^3) 9 = [1, 4, 6, 7]) ∧
    (to_base (12^3) 8 = [1, 3, 7, 6]) ∧
    (∀ b > max_base, sum_of_digits (to_base (12^3) b) = 18) ∧
    (sum_of_digits (to_base (12^3) max_base) ≠ 18) ∧
    max_base = 8 :=
  sorry

end NUMINAMATH_CALUDE_largest_base_not_18_l2734_273499


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l2734_273492

/-- A line tangent to both y = x² and y = -1/x -/
structure TangentLine where
  -- The slope of the tangent line
  m : ℝ
  -- The y-intercept of the tangent line
  b : ℝ
  -- The x-coordinate of the point of tangency on y = x²
  x₁ : ℝ
  -- The x-coordinate of the point of tangency on y = -1/x
  x₂ : ℝ
  -- Condition: The line is tangent to y = x² at (x₁, x₁²)
  h₁ : m * x₁ + b = x₁^2
  -- Condition: The slope at the point of tangency on y = x² is correct
  h₂ : m = 2 * x₁
  -- Condition: The line is tangent to y = -1/x at (x₂, -1/x₂)
  h₃ : m * x₂ + b = -1 / x₂
  -- Condition: The slope at the point of tangency on y = -1/x is correct
  h₄ : m = 1 / x₂^2

/-- The area of the triangle formed by a tangent line and the coordinate axes is 2 -/
theorem tangent_line_triangle_area (l : TangentLine) : 
  (1 / 2) * (1 / l.m) * (-l.b) = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l2734_273492


namespace NUMINAMATH_CALUDE_fancy_sandwich_cost_l2734_273453

/-- The cost of a fancy ham and cheese sandwich given Teresa's shopping list and total spent --/
theorem fancy_sandwich_cost (num_sandwiches : ℕ) (salami_cost brie_cost olive_price_per_pound feta_price_per_pound bread_cost total_spent : ℚ) 
  (olive_weight feta_weight : ℚ) : 
  num_sandwiches = 2 ∧ 
  salami_cost = 4 ∧ 
  brie_cost = 3 * salami_cost ∧ 
  olive_price_per_pound = 10 ∧ 
  olive_weight = 1/4 ∧ 
  feta_price_per_pound = 8 ∧ 
  feta_weight = 1/2 ∧ 
  bread_cost = 2 ∧ 
  total_spent = 40 → 
  (total_spent - (salami_cost + brie_cost + olive_price_per_pound * olive_weight + 
    feta_price_per_pound * feta_weight + bread_cost)) / num_sandwiches = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_fancy_sandwich_cost_l2734_273453


namespace NUMINAMATH_CALUDE_paths_in_10x5_grid_avoiding_point_l2734_273486

/-- The number of paths in a grid that avoid a specific point -/
def grid_paths_avoiding_point (m n a b c d : ℕ) : ℕ :=
  Nat.choose (m + n) n - Nat.choose (a + b) b * Nat.choose ((m - a) + (n - b)) (n - b)

/-- Theorem stating the number of paths in a 10x5 grid from (0,0) to (10,5) avoiding (5,3) -/
theorem paths_in_10x5_grid_avoiding_point : 
  grid_paths_avoiding_point 10 5 5 3 5 2 = 1827 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_10x5_grid_avoiding_point_l2734_273486


namespace NUMINAMATH_CALUDE_curve_to_line_equation_l2734_273471

/-- The curve parameterized by (x,y) = (3t + 6, 5t - 7) can be expressed as y = (5/3)x - 17 --/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 → y = (5/3) * x - 17 := by
  sorry

end NUMINAMATH_CALUDE_curve_to_line_equation_l2734_273471


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2734_273483

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x^2 - 3*x - 4 = 0 → x ≠ -1 →
  (2 - (x - 1) / (x + 1)) / ((x^2 + 6*x + 9) / (x^2 - 1)) = (x - 1) / (x + 3) ∧
  (x - 1) / (x + 3) = 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2734_273483


namespace NUMINAMATH_CALUDE_bells_toll_together_once_l2734_273422

def bell_intervals : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23]

def lcm_list (L : List ℕ) : ℕ :=
  L.foldl Nat.lcm 1

theorem bells_toll_together_once (intervals : List ℕ) (duration : ℕ) : 
  intervals = bell_intervals → duration = 60 * 60 → 
  (duration / (lcm_list intervals) + 1 : ℕ) = 1 :=
by sorry

end NUMINAMATH_CALUDE_bells_toll_together_once_l2734_273422


namespace NUMINAMATH_CALUDE_ben_gross_income_l2734_273441

-- Define Ben's financial situation
def ben_finances (gross_income : ℝ) : Prop :=
  ∃ (after_tax_income : ℝ),
    -- 20% of after-tax income is spent on car
    0.2 * after_tax_income = 400 ∧
    -- 1/3 of gross income is paid in taxes
    gross_income - (1/3) * gross_income = after_tax_income

-- Theorem statement
theorem ben_gross_income :
  ∃ (gross_income : ℝ), ben_finances gross_income ∧ gross_income = 3000 :=
sorry

end NUMINAMATH_CALUDE_ben_gross_income_l2734_273441


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l2734_273404

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 68 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l2734_273404


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_21_l2734_273450

-- Define the functions f and F
def f (a : ℝ) : ℝ := a - 2
def F (a b : ℝ) : ℝ := a * b + b^2

-- State the theorem
theorem F_of_4_f_of_5_equals_21 : F 4 (f 5) = 21 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_21_l2734_273450


namespace NUMINAMATH_CALUDE_inequality_solution_l2734_273498

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 0}
  else if 0 < a ∧ a < 1 then {x | (1 - Real.sqrt (1 - a^2)) / a < x ∧ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a ≥ 1 then ∅
  else if -1 < a ∧ a < 0 then {x | x > (1 - Real.sqrt (1 - a^2)) / a ∨ x < (1 + Real.sqrt (1 - a^2)) / a}
  else if a = -1 then {x | x ≠ 1 / a}
  else Set.univ

theorem inequality_solution (a : ℝ) :
  {x : ℝ | a * x^2 - 2 * x + a < 0} = solution_set a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2734_273498


namespace NUMINAMATH_CALUDE_inequality_theorem_l2734_273473

theorem inequality_theorem (a b c d : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ((a < c ∧ c < b) ∨ (a < d ∧ d < b) ∨ (c < a ∧ a < d) ∨ (c < b ∧ b < d)) →
  Real.sqrt ((a + b) * (c + d)) > Real.sqrt (a * b) + Real.sqrt (c * d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2734_273473


namespace NUMINAMATH_CALUDE_perfect_square_addition_subtraction_l2734_273423

theorem perfect_square_addition_subtraction : ∃! n : ℤ, 
  (∃ u : ℤ, n + 5 = u^2) ∧ (∃ v : ℤ, n - 11 = v^2) :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_addition_subtraction_l2734_273423


namespace NUMINAMATH_CALUDE_angle_theorem_l2734_273416

theorem angle_theorem (α β : Real) (P : Real × Real) :
  P = (3, 4) → -- Point P is (3,4)
  (∃ r : Real, r > 0 ∧ P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α) → -- P is on terminal side of α
  Real.cos β = 5/13 → -- cos β = 5/13
  β ∈ Set.Icc 0 (Real.pi / 2) → -- β ∈ [0, π/2]
  Real.sin α = 4/5 ∧ Real.cos (α - β) = 63/65 := by
  sorry

end NUMINAMATH_CALUDE_angle_theorem_l2734_273416


namespace NUMINAMATH_CALUDE_distance_to_incenter_in_isosceles_right_triangle_l2734_273400

/-- An isosceles right triangle with hypotenuse length 6 -/
structure IsoscelesRightTriangle where
  -- A is the right angle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  isIsosceles : AB = BC
  isRight : AC = 6

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to the incenter -/
def distanceToIncenter (t : IsoscelesRightTriangle) : ℝ := sorry

theorem distance_to_incenter_in_isosceles_right_triangle (t : IsoscelesRightTriangle) :
  distanceToIncenter t = 6 * Real.sqrt 2 - 6 := by sorry

end NUMINAMATH_CALUDE_distance_to_incenter_in_isosceles_right_triangle_l2734_273400


namespace NUMINAMATH_CALUDE_combined_height_l2734_273432

theorem combined_height (kirill_height brother_height : ℕ) : 
  kirill_height = 49 →
  brother_height = kirill_height + 14 →
  kirill_height + brother_height = 112 := by
sorry

end NUMINAMATH_CALUDE_combined_height_l2734_273432


namespace NUMINAMATH_CALUDE_zoe_pictures_l2734_273407

theorem zoe_pictures (total_pictures : ℕ) (dolphin_show_pictures : ℕ) 
  (h1 : total_pictures = 44)
  (h2 : dolphin_show_pictures = 16) :
  total_pictures - dolphin_show_pictures = 28 := by
  sorry

end NUMINAMATH_CALUDE_zoe_pictures_l2734_273407


namespace NUMINAMATH_CALUDE_cubic_equation_proof_l2734_273417

theorem cubic_equation_proof :
  let f : ℝ → ℝ := fun x ↦ x^3 - 5*x - 2
  ∃ (x₁ x₂ x₃ : ℝ),
    (∀ x, f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    (x₁ * x₂ * x₃ = x₁ + x₂ + x₃ + 2) ∧
    (x₁^2 + x₂^2 + x₃^2 = 10) ∧
    (x₁^3 + x₂^3 + x₃^3 = 6) :=
by sorry

#check cubic_equation_proof

end NUMINAMATH_CALUDE_cubic_equation_proof_l2734_273417


namespace NUMINAMATH_CALUDE_food_rent_ratio_l2734_273466

/-- Esperanza's monthly financial situation -/
structure EsperanzaFinances where
  rent : ℝ
  food : ℝ
  mortgage : ℝ
  savings : ℝ
  taxes : ℝ
  salary : ℝ

/-- Conditions for Esperanza's finances -/
def validFinances (e : EsperanzaFinances) : Prop :=
  e.rent = 600 ∧
  e.mortgage = 3 * e.food ∧
  e.savings = 2000 ∧
  e.taxes = 2/5 * e.savings ∧
  e.salary = 4840 ∧
  e.salary = e.rent + e.food + e.mortgage + e.savings + e.taxes

/-- The theorem to prove -/
theorem food_rent_ratio (e : EsperanzaFinances) 
  (h : validFinances e) : e.food / e.rent = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_food_rent_ratio_l2734_273466


namespace NUMINAMATH_CALUDE_neg_p_and_q_implies_not_p_and_q_l2734_273448

theorem neg_p_and_q_implies_not_p_and_q (p q : Prop) :
  (¬p ∧ q) → (¬p ∧ q) :=
by
  sorry

end NUMINAMATH_CALUDE_neg_p_and_q_implies_not_p_and_q_l2734_273448


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_and_midpoint_distance_l2734_273431

theorem circle_radius_from_chords_and_midpoint_distance 
  (chord1 : ℝ) (chord2 : ℝ) (midpoint_distance : ℝ) (radius : ℝ) : 
  chord1 = 10 → 
  chord2 = 12 → 
  midpoint_distance = 4 → 
  (8 * (2 * radius - 8) = 6 * 6) → 
  radius = 6.25 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_and_midpoint_distance_l2734_273431


namespace NUMINAMATH_CALUDE_lewis_harvest_earnings_l2734_273429

/-- Calculates the total earnings during harvest season after paying rent -/
def harvest_earnings (weekly_earnings : ℕ) (weekly_rent : ℕ) (harvest_weeks : ℕ) : ℕ :=
  (weekly_earnings - weekly_rent) * harvest_weeks

/-- Theorem: Lewis's earnings during harvest season -/
theorem lewis_harvest_earnings :
  harvest_earnings 403 49 233 = 82782 := by
  sorry

end NUMINAMATH_CALUDE_lewis_harvest_earnings_l2734_273429


namespace NUMINAMATH_CALUDE_symmetry_x_axis_l2734_273426

/-- Given two points P and Q in the Cartesian coordinate system,
    prove that if P is symmetric to Q with respect to the x-axis,
    then the sum of their x-coordinates minus 3 and the negation of Q's y-coordinate minus 1
    is equal to 3. -/
theorem symmetry_x_axis (a b : ℝ) :
  let P : ℝ × ℝ := (a - 3, 1)
  let Q : ℝ × ℝ := (2, b + 1)
  (P.1 = Q.1) →  -- x-coordinates are equal
  (P.2 = -Q.2) → -- y-coordinates are opposite
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_x_axis_l2734_273426


namespace NUMINAMATH_CALUDE_program_list_orders_l2734_273494

/-- Represents the number of items in the program list -/
def n : ℕ := 6

/-- Represents the number of items that must be adjacent -/
def adjacent_items : ℕ := 2

/-- Represents the number of slots available for inserting the item that can't be first -/
def available_slots : ℕ := n - 1

/-- Calculates the number of different orders for the program list -/
def program_orders : ℕ :=
  (Nat.factorial (n - adjacent_items + 1)) *
  (Nat.choose available_slots 1) *
  (Nat.factorial adjacent_items)

theorem program_list_orders :
  program_orders = 192 := by sorry

end NUMINAMATH_CALUDE_program_list_orders_l2734_273494


namespace NUMINAMATH_CALUDE_pirate_costume_cost_l2734_273491

theorem pirate_costume_cost (num_friends : ℕ) (total_spent : ℕ) : 
  num_friends = 8 → total_spent = 40 → total_spent / num_friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_pirate_costume_cost_l2734_273491


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l2734_273413

/-- Represents the worth of animals in taels of gold -/
structure AnimalWorth where
  cow : ℝ
  sheep : ℝ

/-- Represents the total worth of a group of animals -/
def groupWorth (w : AnimalWorth) (cows sheep : ℕ) : ℝ :=
  cows * w.cow + sheep * w.sheep

/-- The problem statement from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (w : AnimalWorth) : 
  (groupWorth w 5 2 = 10 ∧ groupWorth w 2 5 = 8) ↔ 
  (5 * w.cow + 2 * w.sheep = 10 ∧ 2 * w.cow + 5 * w.sheep = 8) := by
sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l2734_273413


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2734_273408

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) → 
  (10 * x₂^2 + 15 * x₂ - 20 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2734_273408


namespace NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l2734_273493

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_6_range_of_a_l2734_273493


namespace NUMINAMATH_CALUDE_union_equality_implies_range_l2734_273445

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem union_equality_implies_range (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_range_l2734_273445


namespace NUMINAMATH_CALUDE_larger_number_problem_l2734_273424

theorem larger_number_problem (x y : ℤ) 
  (h1 : y = x + 10) 
  (h2 : x + y = 34) : 
  y = 22 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2734_273424


namespace NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l2734_273430

/-- Two unit vectors in a plane have equal squared magnitudes. -/
theorem unit_vectors_equal_squared_magnitude
  (e₁ e₂ : ℝ × ℝ)
  (h₁ : ‖e₁‖ = 1)
  (h₂ : ‖e₂‖ = 1) :
  ‖e₁‖^2 = ‖e₂‖^2 := by
  sorry

end NUMINAMATH_CALUDE_unit_vectors_equal_squared_magnitude_l2734_273430


namespace NUMINAMATH_CALUDE_jackson_chairs_l2734_273435

/-- The number of chairs Jackson needs to buy for his restaurant -/
def total_chairs (tables_with_4_seats tables_with_6_seats : ℕ) : ℕ :=
  tables_with_4_seats * 4 + tables_with_6_seats * 6

/-- Proof that Jackson needs to buy 96 chairs -/
theorem jackson_chairs : total_chairs 6 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_jackson_chairs_l2734_273435


namespace NUMINAMATH_CALUDE_f_lower_bound_f_one_less_than_two_l2734_273444

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1 + a| + |x - a|

-- Part 1
theorem f_lower_bound (x a : ℝ) (h : a ≥ 2) : f x a ≥ 3 := by
  sorry

-- Part 2
theorem f_one_less_than_two (a : ℝ) : 
  (f 1 a < 2) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_one_less_than_two_l2734_273444


namespace NUMINAMATH_CALUDE_smallest_cube_ending_392_l2734_273496

theorem smallest_cube_ending_392 :
  ∃ (n : ℕ), n > 0 ∧ n^3 ≡ 392 [ZMOD 1000] ∧ ∀ (m : ℕ), m > 0 ∧ m^3 ≡ 392 [ZMOD 1000] → n ≤ m :=
by
  use 48
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_392_l2734_273496


namespace NUMINAMATH_CALUDE_total_filets_meeting_limit_l2734_273478

/- Define fish species -/
inductive Species
| Bluefish
| Yellowtail
| RedSnapper

/- Define a structure for a fish -/
structure Fish where
  species : Species
  length : Nat

/- Define the minimum size limits -/
def minSizeLimit (s : Species) : Nat :=
  match s with
  | Species.Bluefish => 7
  | Species.Yellowtail => 6
  | Species.RedSnapper => 8

/- Define a function to check if a fish meets the size limit -/
def meetsLimit (f : Fish) : Bool :=
  f.length ≥ minSizeLimit f.species

/- Define the list of all fish caught -/
def allFish : List Fish := [
  {species := Species.Bluefish, length := 5},
  {species := Species.Bluefish, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.Yellowtail, length := 9},
  {species := Species.RedSnapper, length := 11},
  {species := Species.Bluefish, length := 6},
  {species := Species.Yellowtail, length := 6},
  {species := Species.Yellowtail, length := 10},
  {species := Species.RedSnapper, length := 4},
  {species := Species.Bluefish, length := 8},
  {species := Species.RedSnapper, length := 3},
  {species := Species.Yellowtail, length := 7},
  {species := Species.Yellowtail, length := 12},
  {species := Species.Bluefish, length := 12},
  {species := Species.Bluefish, length := 12}
]

/- Define the number of filets per fish -/
def filetsPerFish : Nat := 2

/- Theorem: The total number of filets from fish meeting size limits is 22 -/
theorem total_filets_meeting_limit : 
  (allFish.filter meetsLimit).length * filetsPerFish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_filets_meeting_limit_l2734_273478


namespace NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l2734_273403

/-- Given an integer N represented as 777 in base b, 
    18 is the smallest b for which N is a fourth power -/
theorem smallest_base_for_fourth_power (N : ℤ) (b : ℤ) : 
  N = 7 * b^2 + 7 * b + 7 → -- N's representation in base b is 777
  (∃ (a : ℤ), N = a^4) →    -- N is a fourth power
  b ≥ 18 :=                 -- 18 is the smallest such b
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_fourth_power_l2734_273403


namespace NUMINAMATH_CALUDE_realtor_earnings_problem_l2734_273436

/-- A realtor's earnings and house sales problem -/
theorem realtor_earnings_problem 
  (base_salary : ℕ) 
  (commission_rate : ℚ) 
  (num_houses : ℕ) 
  (total_earnings : ℕ) 
  (house_a_cost : ℕ) :
  base_salary = 3000 →
  commission_rate = 2 / 100 →
  num_houses = 3 →
  total_earnings = 8000 →
  house_a_cost = 60000 →
  ∃ (house_b_cost house_c_cost : ℕ),
    house_b_cost = 3 * house_a_cost ∧
    ∃ (subtracted_amount : ℕ),
      house_c_cost = 2 * house_a_cost - subtracted_amount ∧
      house_a_cost + house_b_cost + house_c_cost = 
        (total_earnings - base_salary) / commission_rate ∧
      subtracted_amount = 110000 :=
by sorry

end NUMINAMATH_CALUDE_realtor_earnings_problem_l2734_273436


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l2734_273479

theorem three_digit_divisible_by_nine :
  ∀ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
    n % 10 = 2 ∧          -- Units digit is 2
    n / 100 = 4 ∧         -- Hundreds digit is 4
    n % 9 = 0             -- Divisible by 9
    → n = 432 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l2734_273479


namespace NUMINAMATH_CALUDE_geometric_relations_l2734_273470

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelLL : Line → Line → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularLL : Line → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define the theorem
theorem geometric_relations 
  (a b c : Line) (α β γ : Plane) :
  -- Proposition 2
  (skew a b ∧ 
   contains α a ∧ 
   contains β b ∧ 
   parallelLP a β ∧ 
   parallelLP b α → 
   parallel α β) ∧
  -- Proposition 3
  (intersect α β a ∧ 
   intersect β γ b ∧ 
   intersect γ α c ∧ 
   parallelLL a b → 
   parallelLP c β) ∧
  -- Proposition 4
  (skew a b ∧ 
   parallelLP a α ∧ 
   parallelLP b α ∧ 
   perpendicularLL c a ∧ 
   perpendicularLL c b → 
   perpendicularLP c α) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l2734_273470


namespace NUMINAMATH_CALUDE_product_325_4_base_7_l2734_273427

/-- Converts a number from base 7 to base 10 -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 7 -/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Multiplies two numbers in base 7 -/
def mult_base_7 (a b : List Nat) : List Nat :=
  to_base_7 (to_base_10 a * to_base_10 b)

theorem product_325_4_base_7 :
  mult_base_7 [5, 2, 3] [4] = [6, 3, 6, 1] := by sorry

end NUMINAMATH_CALUDE_product_325_4_base_7_l2734_273427


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2734_273451

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P(x₀, y₀) on its right branch such that the difference between
    its distances to the left and right foci is 8, and the product of its
    distances to the two asymptotes is 16/5, prove that the eccentricity of
    the hyperbola is √5/2. -/
theorem hyperbola_eccentricity (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0)
    (heq : x₀^2 / a^2 - y₀^2 / b^2 = 1)
    (hright : x₀ > 0)
    (hfoci : 2 * a = 8)
    (hasymptotes : (b * x₀ - a * y₀) * (b * x₀ + a * y₀) / (a^2 + b^2) = 16/5) :
    let c := Real.sqrt (a^2 + b^2)
    c / a = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2734_273451


namespace NUMINAMATH_CALUDE_sharmila_average_earnings_l2734_273439

/-- Represents Sharmila's work schedule and earnings --/
structure WorkSchedule where
  job1_long_days : Nat -- Number of 10-hour days in job 1
  job1_short_days : Nat -- Number of 8-hour days in job 1
  job1_hourly_rate : ℚ -- Hourly rate for job 1
  job1_long_day_bonus : ℚ -- Bonus for 10-hour days in job 1
  job2_hours : Nat -- Hours worked in job 2
  job2_hourly_rate : ℚ -- Hourly rate for job 2
  job2_bonus : ℚ -- Bonus for job 2

/-- Calculates the average hourly earnings --/
def average_hourly_earnings (schedule : WorkSchedule) : ℚ :=
  let job1_hours := schedule.job1_long_days * 10 + schedule.job1_short_days * 8
  let job1_earnings := job1_hours * schedule.job1_hourly_rate + schedule.job1_long_days * schedule.job1_long_day_bonus
  let job2_earnings := schedule.job2_hours * schedule.job2_hourly_rate + schedule.job2_bonus
  let total_earnings := job1_earnings + job2_earnings
  let total_hours := job1_hours + schedule.job2_hours
  total_earnings / total_hours

/-- Sharmila's work schedule --/
def sharmila_schedule : WorkSchedule := {
  job1_long_days := 3
  job1_short_days := 2
  job1_hourly_rate := 15
  job1_long_day_bonus := 20
  job2_hours := 5
  job2_hourly_rate := 12
  job2_bonus := 10
}

/-- Theorem stating Sharmila's average hourly earnings --/
theorem sharmila_average_earnings :
  average_hourly_earnings sharmila_schedule = 16.08 := by
  sorry


end NUMINAMATH_CALUDE_sharmila_average_earnings_l2734_273439


namespace NUMINAMATH_CALUDE_biology_exam_failure_count_l2734_273475

theorem biology_exam_failure_count : 
  ∀ (total_students : ℕ) 
    (perfect_score_fraction : ℚ)
    (passing_score_fraction : ℚ),
  total_students = 80 →
  perfect_score_fraction = 2/5 →
  passing_score_fraction = 1/2 →
  (total_students : ℚ) * perfect_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * passing_score_fraction +
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 
  (total_students : ℚ) →
  (total_students : ℚ) * (1 - perfect_score_fraction) * (1 - passing_score_fraction) = 24 :=
by sorry

end NUMINAMATH_CALUDE_biology_exam_failure_count_l2734_273475


namespace NUMINAMATH_CALUDE_max_remainder_2018_l2734_273456

theorem max_remainder_2018 (d : ℕ) (h1 : 1 ≤ d) (h2 : d ≤ 1000) : 
  ∃ (q r : ℕ), 2018 = q * d + r ∧ r < d ∧ r ≤ 672 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_2018_l2734_273456


namespace NUMINAMATH_CALUDE_square_sum_proof_l2734_273488

theorem square_sum_proof (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_proof_l2734_273488


namespace NUMINAMATH_CALUDE_unique_sums_count_l2734_273454

def bag_C : Finset ℕ := {1, 2, 3, 4}
def bag_D : Finset ℕ := {3, 5, 7}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun (p : ℕ × ℕ) => p.1 + p.2)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l2734_273454


namespace NUMINAMATH_CALUDE_sequence_next_terms_l2734_273415

def sequence1 : List ℕ := [7, 11, 19, 35]
def sequence2 : List ℕ := [1, 4, 9, 16, 25]

def next_in_sequence1 (seq : List ℕ) : ℕ :=
  let diffs := List.zipWith (·-·) (seq.tail) seq
  let last_diff := diffs.getLast!
  seq.getLast! + (2 * last_diff)

def next_in_sequence2 (seq : List ℕ) : ℕ :=
  (seq.length + 1) ^ 2

theorem sequence_next_terms :
  next_in_sequence1 sequence1 = 67 ∧ next_in_sequence2 sequence2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sequence_next_terms_l2734_273415


namespace NUMINAMATH_CALUDE_orange_boxes_weight_l2734_273485

/-- Calculates the total weight of oranges in three boxes given their capacities, fill ratios, and orange weights. -/
theorem orange_boxes_weight (capacity1 capacity2 capacity3 : ℕ)
                            (fill1 fill2 fill3 : ℚ)
                            (weight1 weight2 weight3 : ℚ) :
  capacity1 = 80 →
  capacity2 = 50 →
  capacity3 = 60 →
  fill1 = 3/4 →
  fill2 = 3/5 →
  fill3 = 2/3 →
  weight1 = 1/4 →
  weight2 = 3/10 →
  weight3 = 2/5 →
  (capacity1 * fill1 * weight1 + capacity2 * fill2 * weight2 + capacity3 * fill3 * weight3 : ℚ) = 40 := by
  sorry

#eval (80 * (3/4) * (1/4) + 50 * (3/5) * (3/10) + 60 * (2/3) * (2/5) : ℚ)

end NUMINAMATH_CALUDE_orange_boxes_weight_l2734_273485


namespace NUMINAMATH_CALUDE_two_from_four_one_from_pair_l2734_273452

/-- The number of ways to select 2 students from a group of 4, where exactly one is chosen from a specific pair --/
theorem two_from_four_one_from_pair : ℕ := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_one_from_pair_l2734_273452


namespace NUMINAMATH_CALUDE_triangle_sine_identity_l2734_273402

theorem triangle_sine_identity (A B C : ℝ) (h₁ : 0 < A) (h₂ : 0 < B) (h₃ : 0 < C) 
  (h₄ : A + B + C = π) (h₅ : 4 * Real.sin A * Real.sin B * Real.cos C = Real.sin A ^ 2 + Real.sin B ^ 2) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 = 2 * Real.sin C ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_identity_l2734_273402


namespace NUMINAMATH_CALUDE_alices_number_l2734_273447

theorem alices_number (x : ℝ) : 3 * (3 * x - 6) = 141 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l2734_273447


namespace NUMINAMATH_CALUDE_dot_product_is_two_l2734_273487

/-- A rhombus with side length 2 and angle BAC of 60° -/
structure Rhombus :=
  (A B C D : ℝ × ℝ)
  (is_rhombus : sorry)
  (side_length : sorry)
  (angle_BAC : sorry)

/-- The dot product of vectors BC and AC in the given rhombus -/
def dot_product_BC_AC (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The dot product of vectors BC and AC in the given rhombus is 2 -/
theorem dot_product_is_two (r : Rhombus) : dot_product_BC_AC r = 2 :=
  sorry

end NUMINAMATH_CALUDE_dot_product_is_two_l2734_273487


namespace NUMINAMATH_CALUDE_four_primes_sum_l2734_273438

theorem four_primes_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (p₁ * p₂ * p₃ * p₄ ∣ 16^4 + 16^2 + 1) ∧
  p₁ + p₂ + p₃ + p₄ = 264 := by
  sorry

end NUMINAMATH_CALUDE_four_primes_sum_l2734_273438


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2734_273401

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℝ) : Prop :=
  ∃ m : ℝ, n = m^2

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬(isPerfectSquare x) ∧ ∀ y z : ℝ, (y > 0 ∧ z > 0 ∧ x = y * z) → ¬(isPerfectSquare y)

-- State the theorem
theorem simplest_quadratic_radical :
  ¬(isSimplestQuadraticRadical 0.5) ∧
  ¬(isSimplestQuadraticRadical 8) ∧
  ¬(isSimplestQuadraticRadical 27) ∧
  ∀ a : ℝ, isSimplestQuadraticRadical (a^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2734_273401


namespace NUMINAMATH_CALUDE_line_passes_through_point_l2734_273455

/-- Given a line with equation 2 + 3kx = -7y that passes through the point (-1/3, 4),
    prove that k = 30. -/
theorem line_passes_through_point (k : ℝ) : 
  (2 + 3 * k * (-1/3) = -7 * 4) → k = 30 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l2734_273455


namespace NUMINAMATH_CALUDE_train_passing_time_l2734_273428

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 20 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * (1000 / 3600))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2734_273428


namespace NUMINAMATH_CALUDE_palindrome_square_base_l2734_273458

theorem palindrome_square_base (r : ℕ) (x : ℕ) (p : ℕ) (n : ℕ) :
  r > 3 →
  x = p * r^3 + p * r^2 + 2 * p * r + 2 * p →
  (∃ (a b c : ℕ), x^2 = a * r^6 + b * r^5 + c * r^4 + c * r^3 + c * r^2 + b * r + a) →
  (∃ m : ℕ, r = 3 * m^2 ∧ m > 1) :=
sorry

end NUMINAMATH_CALUDE_palindrome_square_base_l2734_273458


namespace NUMINAMATH_CALUDE_hilary_kernels_to_shuck_l2734_273410

/-- Calculates the total number of kernels Hilary has to shuck --/
def total_kernels (ears_per_stalk : ℕ) (num_stalks : ℕ) (kernels_first_half : ℕ) (additional_kernels_second_half : ℕ) : ℕ :=
  let total_ears := ears_per_stalk * num_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels_second_half
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

/-- Theorem stating that Hilary has 237,600 kernels to shuck --/
theorem hilary_kernels_to_shuck :
  total_kernels 4 108 500 100 = 237600 := by
  sorry

end NUMINAMATH_CALUDE_hilary_kernels_to_shuck_l2734_273410


namespace NUMINAMATH_CALUDE_dist_P_F₂_eq_two_l2734_273442

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 2 = 1

-- Define the foci
variable (F₁ F₂ : ℝ × ℝ)

-- Define a point on the ellipse
variable (P : ℝ × ℝ)

-- Axiom: P is on the ellipse
axiom P_on_ellipse : is_on_ellipse P.1 P.2

-- Axiom: Distance from P to F₁ is 4
axiom dist_P_F₁ : Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) = 4

-- Theorem to prove
theorem dist_P_F₂_eq_two : Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dist_P_F₂_eq_two_l2734_273442


namespace NUMINAMATH_CALUDE_complex_ln_def_l2734_273409

-- Define the complex logarithm function
def complex_ln (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ k : ℤ, w = Complex.log (Complex.abs z) + Complex.I * (Complex.arg z + 2 * k * Real.pi)}

-- State the theorem
theorem complex_ln_def (z : ℂ) :
  ∀ w ∈ complex_ln z, Complex.exp w = z :=
by sorry

end NUMINAMATH_CALUDE_complex_ln_def_l2734_273409
