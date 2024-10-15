import Mathlib

namespace NUMINAMATH_CALUDE_number_relationship_l3666_366629

theorem number_relationship (x : ℝ) : 
  (5 * x = 2 * x + 10) → (5 * x - 2 * x = 10) := by sorry

end NUMINAMATH_CALUDE_number_relationship_l3666_366629


namespace NUMINAMATH_CALUDE_expression_simplification_l3666_366619

theorem expression_simplification (x y : ℤ) : 
  (x = 1) → (y = -2) → 
  2 * x^2 - (3 * (-5/3 * x^2 + 2/3 * x * y) - (x * y - 3 * x^2)) + 2 * x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3666_366619


namespace NUMINAMATH_CALUDE_sarah_and_bob_walking_l3666_366625

/-- Sarah's walking rate in miles per minute -/
def sarah_rate : ℚ := 1 / 18

/-- Time Sarah walks in minutes -/
def sarah_time : ℚ := 15

/-- Distance Sarah walks in miles -/
def sarah_distance : ℚ := sarah_rate * sarah_time

/-- Bob's walking rate in miles per minute -/
def bob_rate : ℚ := 2 * sarah_rate

/-- Time Bob takes to walk Sarah's distance in minutes -/
def bob_time : ℚ := sarah_distance / bob_rate

theorem sarah_and_bob_walking :
  sarah_distance = 5 / 6 ∧ bob_time = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sarah_and_bob_walking_l3666_366625


namespace NUMINAMATH_CALUDE_max_integer_with_divisor_difference_twenty_four_satisfies_condition_l3666_366651

theorem max_integer_with_divisor_difference (n : ℕ) : 
  (∀ k : ℕ, k > 0 → k ≤ n / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₂ - d₁ = k) →
  n ≤ 24 :=
by sorry

theorem twenty_four_satisfies_condition : 
  ∀ k : ℕ, k > 0 → k ≤ 24 / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ 24 ∧ d₂ ∣ 24 ∧ d₂ - d₁ = k :=
by sorry

end NUMINAMATH_CALUDE_max_integer_with_divisor_difference_twenty_four_satisfies_condition_l3666_366651


namespace NUMINAMATH_CALUDE_midpoint_locus_l3666_366617

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (A B P : ℝ × ℝ) (m n x y : ℝ) :
  A = (4, -2) →
  B = (m, n) →
  m^2 + n^2 = 4 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l3666_366617


namespace NUMINAMATH_CALUDE_acute_angles_inequality_l3666_366659

theorem acute_angles_inequality (α β : Real) (h_α : 0 < α ∧ α < π / 2) (h_β : 0 < β ∧ β < π / 2) :
  Real.cos α * Real.sin (2 * α) * Real.sin (2 * β) ≤ 4 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_acute_angles_inequality_l3666_366659


namespace NUMINAMATH_CALUDE_fractions_product_one_l3666_366657

theorem fractions_product_one :
  ∃ (a b c : ℕ), 
    2 ≤ a ∧ a ≤ 2016 ∧
    2 ≤ b ∧ b ≤ 2016 ∧
    2 ≤ c ∧ c ≤ 2016 ∧
    (a : ℚ) / (2018 - a) * (b : ℚ) / (2018 - b) * (c : ℚ) / (2018 - c) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fractions_product_one_l3666_366657


namespace NUMINAMATH_CALUDE_shooting_to_total_ratio_total_time_breakdown_running_weightlifting_relation_l3666_366669

/-- Represents Kyle's basketball practice schedule --/
structure BasketballPractice where
  total_time : ℕ           -- Total practice time in minutes
  weightlifting_time : ℕ   -- Time spent weightlifting in minutes
  running_time : ℕ         -- Time spent running in minutes
  shooting_time : ℕ        -- Time spent shooting in minutes

/-- Kyle's basketball practice satisfies the given conditions --/
def kyle_practice : BasketballPractice :=
  { total_time := 120,        -- 2 hours = 120 minutes
    weightlifting_time := 20, -- Given in the problem
    running_time := 40,       -- Twice the weightlifting time
    shooting_time := 60 }     -- Remaining time

/-- The ratio of shooting time to total practice time is 1:2 --/
theorem shooting_to_total_ratio :
  kyle_practice.shooting_time * 2 = kyle_practice.total_time :=
by sorry

/-- All practice activities sum up to the total time --/
theorem total_time_breakdown :
  kyle_practice.weightlifting_time + kyle_practice.running_time + kyle_practice.shooting_time = kyle_practice.total_time :=
by sorry

/-- Running time is twice the weightlifting time --/
theorem running_weightlifting_relation :
  kyle_practice.running_time = 2 * kyle_practice.weightlifting_time :=
by sorry

end NUMINAMATH_CALUDE_shooting_to_total_ratio_total_time_breakdown_running_weightlifting_relation_l3666_366669


namespace NUMINAMATH_CALUDE_james_chore_time_l3666_366662

/-- The time James spends vacuuming, in hours -/
def vacuum_time : ℝ := 3

/-- The factor by which the time spent on other chores exceeds vacuuming time -/
def other_chores_factor : ℝ := 3

/-- The total time James spends on his chores, in hours -/
def total_chore_time : ℝ := vacuum_time + other_chores_factor * vacuum_time

theorem james_chore_time : total_chore_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_james_chore_time_l3666_366662


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3666_366667

theorem quadratic_equation_coefficients :
  ∀ (x : ℝ), 3 * x^2 + 1 = 5 * x ↔ 3 * x^2 + (-5) * x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3666_366667


namespace NUMINAMATH_CALUDE_ellipse_point_distance_l3666_366640

theorem ellipse_point_distance (P : ℝ × ℝ) :
  (P.1^2 / 6 + P.2^2 / 2 = 1) →
  (Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) +
   Real.sqrt (P.1^2 + (P.2 + 1)^2) + Real.sqrt (P.1^2 + (P.2 - 1)^2) = 4 * Real.sqrt 6) →
  (abs P.2 = Real.sqrt (6 / 13)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_l3666_366640


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l3666_366690

/-- The y-intercept of the line 5x - 3y = 15 is (0, -5) -/
theorem y_intercept_of_line (x y : ℝ) :
  5 * x - 3 * y = 15 → y = -5 ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l3666_366690


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3666_366647

theorem smallest_lcm_with_gcd_5 (k ℓ : ℕ) :
  k ≥ 1000 ∧ k < 10000 ∧ ℓ ≥ 1000 ∧ ℓ < 10000 ∧ Nat.gcd k ℓ = 5 →
  Nat.lcm k ℓ ≥ 201000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l3666_366647


namespace NUMINAMATH_CALUDE_train_distance_problem_l3666_366692

/-- Proves that the distance between two stations is 540 km given the conditions of the train problem -/
theorem train_distance_problem (v1 v2 : ℝ) (d : ℝ) :
  v1 = 20 →  -- Speed of train 1 in km/hr
  v2 = 25 →  -- Speed of train 2 in km/hr
  v2 > v1 →  -- Train 2 is faster than train 1
  d = (v2 - v1) * (v1 * v2)⁻¹ * 60 →  -- Difference in distance traveled
  v1 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) + 
  v2 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) = 540 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3666_366692


namespace NUMINAMATH_CALUDE_power_fraction_equality_l3666_366634

theorem power_fraction_equality : (2^2015 + 2^2013 + 2^2011) / (2^2015 - 2^2013 + 2^2011) = 21/13 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l3666_366634


namespace NUMINAMATH_CALUDE_amy_chore_money_l3666_366697

/-- Calculates the money earned from chores given initial amount, birthday money, and final amount --/
def money_from_chores (initial_amount birthday_money final_amount : ℕ) : ℕ :=
  final_amount - initial_amount - birthday_money

/-- Theorem stating that Amy's money from chores is 13 dollars --/
theorem amy_chore_money :
  money_from_chores 2 3 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_amy_chore_money_l3666_366697


namespace NUMINAMATH_CALUDE_tenth_term_is_399_l3666_366616

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1)

theorem tenth_term_is_399 : a 10 = 399 := by sorry

end NUMINAMATH_CALUDE_tenth_term_is_399_l3666_366616


namespace NUMINAMATH_CALUDE_round_balloons_count_l3666_366628

/-- The number of balloons in each bag of round balloons -/
def round_balloons_per_bag : ℕ := sorry

/-- The number of bags of round balloons -/
def round_balloon_bags : ℕ := 5

/-- The number of bags of long balloons -/
def long_balloon_bags : ℕ := 4

/-- The number of long balloons in each bag -/
def long_balloons_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst_balloons : ℕ := 5

/-- The total number of balloons left -/
def total_balloons_left : ℕ := 215

theorem round_balloons_count : round_balloons_per_bag = 20 := by
  sorry

end NUMINAMATH_CALUDE_round_balloons_count_l3666_366628


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3666_366688

theorem geometric_series_sum (a : ℕ+) (n : ℕ+) (h : (a : ℝ) / (1 - 1 / (n : ℝ)) = 3) :
  (a : ℝ) + (a : ℝ) / (n : ℝ) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3666_366688


namespace NUMINAMATH_CALUDE_two_abs_plus_x_nonnegative_l3666_366682

theorem two_abs_plus_x_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_two_abs_plus_x_nonnegative_l3666_366682


namespace NUMINAMATH_CALUDE_sandy_carrots_l3666_366613

/-- Given that Sandy and Sam grew carrots together, with Sam growing 3 carrots
and a total of 9 carrots grown, prove that Sandy grew 6 carrots. -/
theorem sandy_carrots (total : ℕ) (sam : ℕ) (sandy : ℕ) 
  (h1 : total = 9)
  (h2 : sam = 3)
  (h3 : total = sam + sandy) :
  sandy = 6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l3666_366613


namespace NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_equals_zero_l3666_366695

/-- The minimum value of max_{0 ≤ x ≤ 2} |x^2 - 2xy| for y in ℝ is 0 -/
theorem min_max_abs_x_squared_minus_2xy_equals_zero :
  ∃ y : ℝ, ∀ y' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → |x^2 - 2*x*y| ≤ |x^2 - 2*x*y'|) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - 2*x*y| = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_x_squared_minus_2xy_equals_zero_l3666_366695


namespace NUMINAMATH_CALUDE_response_change_difference_l3666_366632

theorem response_change_difference (initial_yes initial_no final_yes final_no : ℚ) :
  initial_yes = 40/100 →
  initial_no = 60/100 →
  final_yes = 80/100 →
  final_no = 20/100 →
  initial_yes + initial_no = 1 →
  final_yes + final_no = 1 →
  ∃ (min_change max_change : ℚ),
    (∀ (change : ℚ), change ≥ min_change ∧ change ≤ max_change) ∧
    max_change - min_change = 20/100 :=
by sorry

end NUMINAMATH_CALUDE_response_change_difference_l3666_366632


namespace NUMINAMATH_CALUDE_min_sum_of_product_2310_l3666_366646

theorem min_sum_of_product_2310 (a b c : ℕ+) (h : a * b * c = 2310) :
  (∀ x y z : ℕ+, x * y * z = 2310 → a + b + c ≤ x + y + z) ∧ a + b + c = 52 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2310_l3666_366646


namespace NUMINAMATH_CALUDE_unique_solution_l3666_366621

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The main theorem stating the unique solution to the functional equation -/
theorem unique_solution :
  ∃! (α : ℝ) (f : ℝ → ℝ), SatisfiesEquation f α ∧ α = -1 ∧ ∀ x, f x = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3666_366621


namespace NUMINAMATH_CALUDE_triangular_seating_theorem_l3666_366658

/-- Represents a triangular seating arrangement in a cinema -/
structure TriangularSeating where
  /-- The number of the best seat (at the center of the height from the top vertex) -/
  best_seat : ℕ
  /-- The total number of seats in the arrangement -/
  total_seats : ℕ

/-- 
Theorem: In a triangular seating arrangement where the best seat 
(at the center of the height from the top vertex) is numbered 265, 
the total number of seats is 1035.
-/
theorem triangular_seating_theorem (ts : TriangularSeating) 
  (h : ts.best_seat = 265) : ts.total_seats = 1035 := by
  sorry

#check triangular_seating_theorem

end NUMINAMATH_CALUDE_triangular_seating_theorem_l3666_366658


namespace NUMINAMATH_CALUDE_age_difference_theorem_l3666_366615

/-- Represents a two-digit age --/
structure TwoDigitAge where
  tens : Nat
  ones : Nat
  h1 : tens ≤ 9
  h2 : ones ≤ 9

def TwoDigitAge.toNat (age : TwoDigitAge) : Nat :=
  10 * age.tens + age.ones

theorem age_difference_theorem (anna ella : TwoDigitAge) 
  (h : anna.tens = ella.ones ∧ anna.ones = ella.tens) 
  (future_relation : (anna.toNat + 10) = 3 * (ella.toNat + 10)) :
  anna.toNat - ella.toNat = 54 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_theorem_l3666_366615


namespace NUMINAMATH_CALUDE_area_between_circles_l3666_366633

/-- The area between two externally tangent circles and their circumscribing circle -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - π * r₁^2 - π * r₂^2 = 40 * π := by sorry

end NUMINAMATH_CALUDE_area_between_circles_l3666_366633


namespace NUMINAMATH_CALUDE_power_of_negative_product_l3666_366668

theorem power_of_negative_product (a b : ℝ) : (-a^3 * b^5)^2 = a^6 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l3666_366668


namespace NUMINAMATH_CALUDE_max_value_of_s_l3666_366624

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10) 
  (sum_products_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) : 
  s ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l3666_366624


namespace NUMINAMATH_CALUDE_events_independent_l3666_366672

-- Define the sample space
def Ω : Type := (Bool × Bool)

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- Define events A, B, and C
def A : Set Ω := {ω | ω.1 = true}
def B : Set Ω := {ω | ω.2 = true}
def C : Set Ω := {ω | ω.1 = ω.2}

-- State the theorem
theorem events_independent :
  (P (A ∩ B) = P A * P B) ∧
  (P (B ∩ C) = P B * P C) ∧
  (P (A ∩ C) = P A * P C) := by sorry

end NUMINAMATH_CALUDE_events_independent_l3666_366672


namespace NUMINAMATH_CALUDE_g_5_equals_104_l3666_366698

def g (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 47 * x^2 - 44 * x + 24

theorem g_5_equals_104 : g 5 = 104 := by sorry

end NUMINAMATH_CALUDE_g_5_equals_104_l3666_366698


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l3666_366620

theorem polynomial_product_expansion (x : ℝ) :
  (1 + x^2 + 2*x - x^4) * (3 - x^3 + 2*x^2 - 5*x) =
  x^7 - 2*x^6 + 4*x^5 - 3*x^4 - 2*x^3 - 4*x^2 + x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l3666_366620


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l3666_366691

/-- Given a quadratic equation y = ax^2 + bx + c, 
    this theorem proves that 2a - b + c = 2 
    when a = 2, b = 3, and c = 1 -/
theorem quadratic_equation_value (a b c : ℝ) : 
  a = 2 ∧ b = 3 ∧ c = 1 → 2*a - b + c = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l3666_366691


namespace NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_11_proof_l3666_366618

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def smallest_odd_digit_multiple_of_11 : ℕ := 11341

theorem smallest_odd_digit_multiple_of_11_proof :
  (smallest_odd_digit_multiple_of_11 > 10000) ∧
  (has_only_odd_digits smallest_odd_digit_multiple_of_11) ∧
  (smallest_odd_digit_multiple_of_11 % 11 = 0) ∧
  (∀ n : ℕ, n > 10000 → has_only_odd_digits n → n % 11 = 0 → n ≥ smallest_odd_digit_multiple_of_11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_11_proof_l3666_366618


namespace NUMINAMATH_CALUDE_unique_solution_implies_any_real_l3666_366604

theorem unique_solution_implies_any_real (a : ℝ) : 
  (∃! x : ℝ, x^2 - 2*a*x + a^2 = 0) → ∀ b : ℝ, ∃ a : ℝ, a = b :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_any_real_l3666_366604


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l3666_366611

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {1, 2}

theorem complement_union_equals_set : 
  (U \ (A ∪ B)) = {-2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l3666_366611


namespace NUMINAMATH_CALUDE_cuboid_area_volume_l3666_366639

/-- Cuboid properties -/
def Cuboid (a b c : ℝ) : Prop :=
  c * Real.sqrt (a^2 + b^2) = 60 ∧
  a * Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 153 ∧
  b * Real.sqrt (a^2 + c^2) = 12 * Real.sqrt 10

/-- Theorem: Surface area and volume of the cuboid -/
theorem cuboid_area_volume (a b c : ℝ) (h : Cuboid a b c) :
  2 * (a * b + b * c + a * c) = 192 ∧ a * b * c = 144 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_volume_l3666_366639


namespace NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3666_366663

def is_composite (n : ℕ) : Prop := ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n : ℕ, n > 217 → is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 217 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3666_366663


namespace NUMINAMATH_CALUDE_max_product_l3666_366664

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_arrangement (a b c d e : ℕ) : Prop :=
  {a, b, c, d, e} = digits.toFinset

def four_digit_num (a b c d : ℕ) : ℕ := 
  1000 * a + 100 * b + 10 * c + d

def product (a b c d e : ℕ) : ℕ :=
  (four_digit_num a b c d) * e

theorem max_product :
  ∀ a b c d e,
    is_valid_arrangement a b c d e →
    product a b c d e ≤ product 8 5 3 1 9 :=
sorry

end NUMINAMATH_CALUDE_max_product_l3666_366664


namespace NUMINAMATH_CALUDE_correct_calculation_of_one_fifth_sum_of_acute_angles_l3666_366652

-- Define acute angle
def is_acute_angle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

-- Theorem statement
theorem correct_calculation_of_one_fifth_sum_of_acute_angles 
  (α β : ℝ) 
  (h_α : is_acute_angle α) 
  (h_β : is_acute_angle β) : 
  18 < (1/5) * (α + β) ∧ 
  (1/5) * (α + β) < 54 ∧ 
  (42 ∈ {17, 42, 56, 73} ∩ Set.Icc 18 54) ∧ 
  ({17, 42, 56, 73} ∩ Set.Icc 18 54 = {42}) :=
sorry

end NUMINAMATH_CALUDE_correct_calculation_of_one_fifth_sum_of_acute_angles_l3666_366652


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l3666_366602

theorem intersection_implies_m_range (m : ℝ) :
  (∃ x : ℝ, m * (4 : ℝ)^x - 3 * (2 : ℝ)^(x + 1) - 2 = 0) →
  m ≥ -9/2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l3666_366602


namespace NUMINAMATH_CALUDE_range_of_x1_l3666_366643

/-- A function f: ℝ → ℝ is increasing -/
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem -/
theorem range_of_x1 (f : ℝ → ℝ) (h_inc : Increasing f)
  (h_ineq : ∀ x₁ x₂ : ℝ, x₁ + x₂ = 1 → f x₁ + f 0 > f x₂ + f 1) :
  ∀ x₁ : ℝ, (∃ x₂ : ℝ, x₁ + x₂ = 1 ∧ f x₁ + f 0 > f x₂ + f 1) → x₁ > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x1_l3666_366643


namespace NUMINAMATH_CALUDE_triangle_frame_angles_l3666_366666

/-- A frame consisting of congruent triangles surrounding a square --/
structure TriangleFrame where
  /-- The number of triangles in the frame --/
  num_triangles : ℕ
  /-- The angles of each triangle in the frame --/
  triangle_angles : Fin 3 → ℝ
  /-- The sum of angles in each triangle is 180° --/
  angle_sum : (triangle_angles 0) + (triangle_angles 1) + (triangle_angles 2) = 180
  /-- The triangles form a complete circle at each corner of the square --/
  corner_sum : 4 * (triangle_angles 0) + 90 = 360
  /-- The triangles along each side of the square form a straight line --/
  side_sum : (triangle_angles 1) + (triangle_angles 2) + 90 = 180

/-- The theorem stating the angles of the triangles in the frame --/
theorem triangle_frame_angles (frame : TriangleFrame) 
  (h : frame.num_triangles = 21) : 
  frame.triangle_angles 0 = 67.5 ∧ 
  frame.triangle_angles 1 = 22.5 ∧ 
  frame.triangle_angles 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_frame_angles_l3666_366666


namespace NUMINAMATH_CALUDE_thabo_total_books_l3666_366655

/-- The number of books Thabo owns of each type and in total. -/
structure ThabosBooks where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ
  total : ℕ

/-- The conditions of Thabo's book collection. -/
def thabo_book_conditions (books : ThabosBooks) : Prop :=
  books.hardcover_nonfiction = 30 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction ∧
  books.total = books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction

/-- Theorem stating that given the conditions, Thabo owns 180 books in total. -/
theorem thabo_total_books :
  ∀ books : ThabosBooks, thabo_book_conditions books → books.total = 180 := by
  sorry


end NUMINAMATH_CALUDE_thabo_total_books_l3666_366655


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l3666_366689

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), (a^2 + b + c : ℕ) = x^2 ∧ (b^2 + c + a : ℕ) = y^2 ∧ (c^2 + a + b : ℕ) = z^2) :=
sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l3666_366689


namespace NUMINAMATH_CALUDE_upstream_distance_calculation_l3666_366635

/-- Represents the problem of calculating the upstream distance rowed by a man --/
theorem upstream_distance_calculation
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (upstream_time : ℝ)
  (current_velocity : ℝ)
  (h1 : downstream_distance = 32)
  (h2 : downstream_time = 6)
  (h3 : upstream_time = 6)
  (h4 : current_velocity = 1.5)
  (h5 : downstream_time > 0)
  (h6 : upstream_time > 0)
  (h7 : current_velocity ≥ 0) :
  let still_water_speed := downstream_distance / downstream_time - current_velocity
  let upstream_distance := (still_water_speed - current_velocity) * upstream_time
  upstream_distance = 14 := by sorry


end NUMINAMATH_CALUDE_upstream_distance_calculation_l3666_366635


namespace NUMINAMATH_CALUDE_dogsled_race_speed_l3666_366678

/-- Proves that given a 300-mile course, if one team (A) finishes 3 hours faster than another team (T)
    and has an average speed 5 mph greater, then the slower team's (T) average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ) :
  course_length = 300 →
  time_difference = 3 →
  speed_difference = 5 →
  ∃ (speed_T : ℝ) (time_T : ℝ) (time_A : ℝ),
    course_length = speed_T * time_T ∧
    course_length = (speed_T + speed_difference) * (time_T - time_difference) ∧
    speed_T = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_l3666_366678


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3666_366637

theorem quadratic_function_proof (f g : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is quadratic
  f 0 = 12 →                                      -- f(0) = 12
  (∀ x, g x = 2^x * f x) →                        -- g(x) = 2^x * f(x)
  (∀ x, g (x + 1) - g x ≥ 2^(x + 1) * x^2) →      -- g(x+1) - g(x) ≥ 2^(x+1) * x^2
  (∀ x, f x = 2 * x^2 - 8 * x + 12) ∧             -- f(x) = 2x^2 - 8x + 12
  (∀ x, g x = (2 * x^2 - 8 * x + 12) * 2^x) :=    -- g(x) = (2x^2 - 8x + 12) * 2^x
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3666_366637


namespace NUMINAMATH_CALUDE_no_solution_for_lcm_gcd_equation_l3666_366661

theorem no_solution_for_lcm_gcd_equation : 
  ¬ ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.lcm n 60 = Nat.gcd n 60 + 200) ∧ 
    (Nat.Prime n) ∧ 
    (60 % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_lcm_gcd_equation_l3666_366661


namespace NUMINAMATH_CALUDE_vector_magnitude_minimization_l3666_366636

/-- Given two unit vectors e₁ and e₂ with an angle of 60° between them,
    prove that |2e₁ + te₂| is minimized when t = -1 -/
theorem vector_magnitude_minimization (e₁ e₂ : ℝ × ℝ) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  e₁ • e₂ = 1/2 →
  ∃ (t : ℝ), ∀ (s : ℝ), ‖2 • e₁ + t • e₂‖ ≤ ‖2 • e₁ + s • e₂‖ ∧ t = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_minimization_l3666_366636


namespace NUMINAMATH_CALUDE_digit_equality_proof_l3666_366654

theorem digit_equality_proof (n : ℕ) (a : ℕ) (k : ℕ) :
  (n ≥ 4) →
  (a ≤ 9) →
  (k ≥ 1) →
  (∃ n k, n * (n + 1) / 2 = (10^k - 1) * a / 9) ↔ (a = 5 ∨ a = 6) :=
by sorry

end NUMINAMATH_CALUDE_digit_equality_proof_l3666_366654


namespace NUMINAMATH_CALUDE_negation_of_implication_l3666_366681

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3666_366681


namespace NUMINAMATH_CALUDE_randy_lunch_cost_l3666_366660

theorem randy_lunch_cost (initial_amount : ℝ) (remaining_amount : ℝ) (lunch_cost : ℝ) :
  initial_amount = 30 →
  remaining_amount = 15 →
  remaining_amount = initial_amount - lunch_cost - (1/4) * (initial_amount - lunch_cost) →
  lunch_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_randy_lunch_cost_l3666_366660


namespace NUMINAMATH_CALUDE_seedling_packaging_l3666_366685

/-- The number of seedlings to be placed in packets -/
def total_seedlings : ℕ := 420

/-- The number of seeds required in each packet -/
def seeds_per_packet : ℕ := 7

/-- The number of packets needed to place all seedlings -/
def packets_needed : ℕ := total_seedlings / seeds_per_packet

theorem seedling_packaging : packets_needed = 60 := by
  sorry

end NUMINAMATH_CALUDE_seedling_packaging_l3666_366685


namespace NUMINAMATH_CALUDE_complement_of_M_l3666_366601

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem complement_of_M : U \ M = {3, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l3666_366601


namespace NUMINAMATH_CALUDE_complement_of_A_l3666_366645

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 0}

theorem complement_of_A : Set.compl A = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3666_366645


namespace NUMINAMATH_CALUDE_problem_statement_l3666_366675

theorem problem_statement (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a * b = 10) :
  (Real.log a + Real.log b > 0) ∧
  (Real.log a - Real.log b > 0) ∧
  (Real.log a * Real.log b < 1/4) ∧
  (¬ ∀ x y : ℝ, x > y ∧ y > 0 ∧ x * y = 10 → Real.log x / Real.log y > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3666_366675


namespace NUMINAMATH_CALUDE_power_seven_strictly_increasing_l3666_366603

theorem power_seven_strictly_increasing (m n : ℝ) (h : m < n) : m^7 < n^7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_strictly_increasing_l3666_366603


namespace NUMINAMATH_CALUDE_turnip_potato_ratio_l3666_366642

/-- Given a ratio of potatoes to turnips and a new amount of potatoes, 
    calculate the amount of turnips that maintains the same ratio -/
def calculate_turnips (potato_ratio : ℚ) (turnip_ratio : ℚ) (new_potato : ℚ) : ℚ :=
  (new_potato * turnip_ratio) / potato_ratio

/-- Prove that given the initial ratio of 5 cups of potatoes to 2 cups of turnips,
    the amount of turnips that can be mixed with 20 cups of potatoes while 
    maintaining the same ratio is 8 cups -/
theorem turnip_potato_ratio : 
  let initial_potato : ℚ := 5
  let initial_turnip : ℚ := 2
  let new_potato : ℚ := 20
  calculate_turnips initial_potato initial_turnip new_potato = 8 := by
  sorry

end NUMINAMATH_CALUDE_turnip_potato_ratio_l3666_366642


namespace NUMINAMATH_CALUDE_conner_average_speed_l3666_366626

/-- The average speed of Conner's dune buggy given different terrain conditions -/
theorem conner_average_speed 
  (flat_speed : ℝ) 
  (downhill_speed_increase : ℝ) 
  (uphill_speed_decrease : ℝ) 
  (h1 : flat_speed = 60) 
  (h2 : downhill_speed_increase = 12) 
  (h3 : uphill_speed_decrease = 18) :
  (flat_speed + (flat_speed + downhill_speed_increase) + (flat_speed - uphill_speed_decrease)) / 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_conner_average_speed_l3666_366626


namespace NUMINAMATH_CALUDE_line_point_k_value_l3666_366648

/-- A line contains the points (5,10), (-3,k), and (-11,5). This theorem proves that k = 7.5. -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 
    (10 = m * 5 + b) ∧ 
    (k = m * (-3) + b) ∧ 
    (5 = m * (-11) + b)) → 
  k = 7.5 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l3666_366648


namespace NUMINAMATH_CALUDE_book_gain_percent_l3666_366649

theorem book_gain_percent (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let cost_price := 0.64 * marked_price
  let selling_price := 0.88 * marked_price
  let profit := selling_price - cost_price
  let gain_percent := (profit / cost_price) * 100
  gain_percent = 37.5 := by sorry

end NUMINAMATH_CALUDE_book_gain_percent_l3666_366649


namespace NUMINAMATH_CALUDE_carols_allowance_l3666_366627

/-- Carol's allowance problem -/
theorem carols_allowance
  (fixed_allowance : ℚ)
  (extra_chore_pay : ℚ)
  (weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : extra_chore_pay = 1.5)
  (h2 : weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  fixed_allowance = 20 := by
  sorry

end NUMINAMATH_CALUDE_carols_allowance_l3666_366627


namespace NUMINAMATH_CALUDE_sqrt_65_bound_l3666_366677

theorem sqrt_65_bound (n : ℕ) (h1 : 0 < n) (h2 : n < Real.sqrt 65) (h3 : Real.sqrt 65 < n + 1) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bound_l3666_366677


namespace NUMINAMATH_CALUDE_jason_earnings_l3666_366653

/-- Represents the earnings of a person given their initial and final amounts -/
def earnings (initial final : ℕ) : ℕ := final - initial

theorem jason_earnings :
  let fred_initial : ℕ := 49
  let jason_initial : ℕ := 3
  let fred_final : ℕ := 112
  let jason_final : ℕ := 63
  earnings jason_initial jason_final = 60 := by
sorry

end NUMINAMATH_CALUDE_jason_earnings_l3666_366653


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3666_366656

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1 / y = 2) 
  (h2 : y + 1 / z = 2) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3666_366656


namespace NUMINAMATH_CALUDE_marys_thursday_payment_l3666_366607

theorem marys_thursday_payment 
  (credit_limit : ℕ) 
  (tuesday_payment : ℕ) 
  (remaining_balance : ℕ) 
  (h1 : credit_limit = 100)
  (h2 : tuesday_payment = 15)
  (h3 : remaining_balance = 62) :
  credit_limit - tuesday_payment - remaining_balance = 23 := by
sorry

end NUMINAMATH_CALUDE_marys_thursday_payment_l3666_366607


namespace NUMINAMATH_CALUDE_train_turn_radians_l3666_366686

/-- Given a circular railway arc with radius 2 km and a train moving at 30 km/h,
    the number of radians the train turns through in 10 seconds is 1/24. -/
theorem train_turn_radians (r : ℝ) (v : ℝ) (t : ℝ) :
  r = 2 →  -- radius in km
  v = 30 → -- speed in km/h
  t = 10 / 3600 → -- time in hours (10 seconds converted to hours)
  (v * t) / r = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_train_turn_radians_l3666_366686


namespace NUMINAMATH_CALUDE_selling_price_achieves_target_profit_selling_price_minimizes_inventory_l3666_366638

/-- Represents the selling price of a helmet -/
def selling_price : ℝ := 50

/-- Represents the cost price of a helmet -/
def cost_price : ℝ := 30

/-- Represents the initial selling price -/
def initial_price : ℝ := 40

/-- Represents the initial monthly sales volume -/
def initial_sales : ℝ := 600

/-- Represents the rate of decrease in sales volume per dollar increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Represents the target monthly profit -/
def target_profit : ℝ := 10000

/-- Calculates the monthly sales volume based on the selling price -/
def monthly_sales (price : ℝ) : ℝ := initial_sales - sales_decrease_rate * (price - initial_price)

/-- Calculates the monthly profit based on the selling price -/
def monthly_profit (price : ℝ) : ℝ := (price - cost_price) * monthly_sales price

/-- Theorem stating that the selling price achieves the target monthly profit -/
theorem selling_price_achieves_target_profit : 
  monthly_profit selling_price = target_profit :=
sorry

/-- Theorem stating that the selling price minimizes inventory -/
theorem selling_price_minimizes_inventory :
  ∀ (price : ℝ), monthly_profit price = target_profit → price ≥ selling_price :=
sorry

end NUMINAMATH_CALUDE_selling_price_achieves_target_profit_selling_price_minimizes_inventory_l3666_366638


namespace NUMINAMATH_CALUDE_circles_intersect_l3666_366609

-- Define Circle C1
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 1

-- Define Circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

-- Theorem stating that C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l3666_366609


namespace NUMINAMATH_CALUDE_second_calculator_price_l3666_366641

def total_calculators : ℕ := 85
def total_sales : ℚ := 3875
def first_calculator_count : ℕ := 35
def first_calculator_price : ℚ := 67

theorem second_calculator_price :
  let second_calculator_count := total_calculators - first_calculator_count
  let first_calculator_total := first_calculator_count * first_calculator_price
  let second_calculator_total := total_sales - first_calculator_total
  second_calculator_total / second_calculator_count = 30.6 := by
sorry

end NUMINAMATH_CALUDE_second_calculator_price_l3666_366641


namespace NUMINAMATH_CALUDE_least_m_for_x_sequence_l3666_366665

def x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 7 * x n + 12) / (x n + 8)

theorem least_m_for_x_sequence :
  ∃ m : ℕ, (∀ k < m, x k > 6 + 1 / 2^22) ∧ x m ≤ 6 + 1 / 2^22 ∧ m = 204 :=
by sorry

end NUMINAMATH_CALUDE_least_m_for_x_sequence_l3666_366665


namespace NUMINAMATH_CALUDE_folding_square_pt_length_l3666_366644

/-- Square with special folding property -/
structure FoldingSquare where
  -- Side length of the square
  side_length : ℝ
  -- Length of PT (and SU by symmetry)
  pt_length : ℝ
  -- Condition that when folded, PR and SR coincide on diagonal RQ
  folding_condition : pt_length ≤ side_length ∧ 
    2 * (side_length - pt_length) / Real.sqrt 2 = side_length * Real.sqrt 2

/-- Theorem about the specific square in the problem -/
theorem folding_square_pt_length :
  ∀ (sq : FoldingSquare), 
  sq.side_length = 2 → 
  sq.pt_length = Real.sqrt 8 - 2 := by
  sorry

end NUMINAMATH_CALUDE_folding_square_pt_length_l3666_366644


namespace NUMINAMATH_CALUDE_expression_bounds_l3666_366670

theorem expression_bounds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  let k : ℝ := 2
  let expr := Real.sqrt (k * a^2 + (2 - b)^2) + Real.sqrt (k * b^2 + (2 - c)^2) + Real.sqrt (k * c^2 + (2 - a)^2)
  6 * Real.sqrt 2 ≤ expr ∧ expr ≤ 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l3666_366670


namespace NUMINAMATH_CALUDE_roots_of_equation_l3666_366676

theorem roots_of_equation (x : ℝ) : 
  (2 * x^2 - x = 0) ↔ (x = 0 ∨ x = 1/2) := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3666_366676


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3666_366679

/-- Represents a partitioned triangle with four regions -/
structure PartitionedTriangle where
  /-- Area of the first triangle -/
  area1 : ℝ
  /-- Area of the second triangle -/
  area2 : ℝ
  /-- Area of the third triangle -/
  area3 : ℝ
  /-- Area of the fourth triangle -/
  area4 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- Theorem stating that given the areas of the four triangles, 
    the area of the quadrilateral is 18 -/
theorem quadrilateral_area 
  (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) 
  (h2 : t.area2 = 9) 
  (h3 : t.area3 = 24/5) 
  (h4 : t.area4 = 9) : 
  t.areaQuad = 18 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_l3666_366679


namespace NUMINAMATH_CALUDE_percentage_relationship_z_less_than_y_l3666_366614

theorem percentage_relationship (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  z = 0.54 * y :=
by sorry

-- The final result can be derived from this theorem
theorem z_less_than_y (w e y z : ℝ) 
  (hw : w = 0.6 * e)
  (he : e = 0.6 * y)
  (hz : z = w * 1.5000000000000002) :
  (y - z) / y = 0.46 :=
by sorry

end NUMINAMATH_CALUDE_percentage_relationship_z_less_than_y_l3666_366614


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3666_366612

/-- The polynomial f(x) = x^3 - 3x^2 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3666_366612


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3666_366610

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 → -- α is acute
  0 < β ∧ β < π/2 → -- β is acute
  |Real.sin α - 1/2| + Real.sqrt ((Real.tan β - 1)^2) = 0 →
  α + β = π/2.4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3666_366610


namespace NUMINAMATH_CALUDE_work_schedule_lcm_l3666_366671

theorem work_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_work_schedule_lcm_l3666_366671


namespace NUMINAMATH_CALUDE_original_equals_scientific_l3666_366673

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to represent in scientific notation -/
def original_number : ℕ := 650000

/-- The scientific notation representation of the original number -/
def scientific_repr : ScientificNotation := {
  coefficient := 6.5,
  exponent := 5,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_repr.coefficient * (10 : ℝ) ^ scientific_repr.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l3666_366673


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3666_366699

theorem unique_solution_exists : ∃! x : ℝ, 0.6667 * x - 10 = 0.25 * x := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3666_366699


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l3666_366623

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 9) 
  (h2 : material2 = 1 / 3) 
  (h3 : leftover = 8 / 24) : 
  material1 + material2 - leftover = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l3666_366623


namespace NUMINAMATH_CALUDE_imaginary_sum_l3666_366606

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum : i^55 + i^555 + i^5 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_l3666_366606


namespace NUMINAMATH_CALUDE_repeating_decimal_difference_l3666_366650

theorem repeating_decimal_difference : 
  (8 : ℚ) / 11 - 72 / 100 = 2 / 275 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_difference_l3666_366650


namespace NUMINAMATH_CALUDE_lcm_36_98_l3666_366630

theorem lcm_36_98 : Nat.lcm 36 98 = 1764 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_98_l3666_366630


namespace NUMINAMATH_CALUDE_seed_mixture_ryegrass_percentage_l3666_366674

/-- Given two seed mixtures X and Y, prove that Y contains 25% ryegrass -/
theorem seed_mixture_ryegrass_percentage :
  -- Definitions based on the problem conditions
  let x_ryegrass : ℝ := 0.40  -- 40% ryegrass in X
  let x_bluegrass : ℝ := 0.60  -- 60% bluegrass in X
  let y_fescue : ℝ := 0.75  -- 75% fescue in Y
  let final_ryegrass : ℝ := 0.32  -- 32% ryegrass in final mixture
  let x_proportion : ℝ := 0.4667  -- 46.67% of final mixture is X
  
  -- The percentage of ryegrass in Y
  ∃ y_ryegrass : ℝ,
    -- Conditions
    x_ryegrass + x_bluegrass = 1 ∧  -- X components sum to 100%
    y_ryegrass + y_fescue = 1 ∧  -- Y components sum to 100%
    x_proportion * x_ryegrass + (1 - x_proportion) * y_ryegrass = final_ryegrass →
    -- Conclusion
    y_ryegrass = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_ryegrass_percentage_l3666_366674


namespace NUMINAMATH_CALUDE_linear_function_segment_l3666_366608

-- Define the linear function
def f (x : ℝ) := -2 * x + 3

-- Define the domain
def domain : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem linear_function_segment :
  ∃ (A B : ℝ × ℝ), 
    (∀ x ∈ domain, ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      (x, f x) = (1 - t) • A + t • B) ∧
    (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
      ((1 - t) • A.1 + t • B.1) ∈ domain) :=
by sorry


end NUMINAMATH_CALUDE_linear_function_segment_l3666_366608


namespace NUMINAMATH_CALUDE_difference_squared_l3666_366693

theorem difference_squared (x y a b : ℝ) 
  (h1 : x * y = b) 
  (h2 : x / y + y / x = a) : 
  (x - y)^2 = a * b - 2 * b := by
sorry

end NUMINAMATH_CALUDE_difference_squared_l3666_366693


namespace NUMINAMATH_CALUDE_angle_value_proof_l3666_366622

theorem angle_value_proof (α : Real) (P : Real × Real) : 
  0 < α → α < Real.pi / 2 →
  P.1 = Real.sin (-50 * Real.pi / 180) →
  P.2 = Real.cos (130 * Real.pi / 180) →
  P ∈ {p : Real × Real | p.1 = Real.sin (5 * α) ∧ p.2 = Real.cos (5 * α)} →
  α = 44 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_value_proof_l3666_366622


namespace NUMINAMATH_CALUDE_ten_strikes_l3666_366600

/-- Represents the time it takes for a clock to strike a given number of times -/
def strike_time (strikes : ℕ) : ℝ :=
  sorry

/-- The clock takes 42 seconds to strike 7 times -/
axiom seven_strikes : strike_time 7 = 42

/-- Theorem: It takes 60 seconds for the clock to strike 10 times -/
theorem ten_strikes : strike_time 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_ten_strikes_l3666_366600


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l3666_366631

theorem smallest_integer_in_set (m : ℤ) : 
  (m + 3 < 3*m - 5) → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l3666_366631


namespace NUMINAMATH_CALUDE_log_equation_solution_l3666_366696

-- Define the equation
def log_equation (x : ℝ) : Prop :=
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 15

-- State the theorem
theorem log_equation_solution :
  ∀ x : ℝ, x > 0 → log_equation x → x = 2^(90/11) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3666_366696


namespace NUMINAMATH_CALUDE_roots_expression_value_l3666_366684

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → 
  (x₁ + x₂) / (1 + x₁ * x₂) = 3/2 := by sorry

end NUMINAMATH_CALUDE_roots_expression_value_l3666_366684


namespace NUMINAMATH_CALUDE_weight_of_oranges_l3666_366694

/-- Proves that the weight of oranges is 1 kilogram, given the total weight of fruits
    and the weights of apples, grapes, and strawberries. -/
theorem weight_of_oranges (total_weight apple_weight grape_weight strawberry_weight : ℕ)
  (h_total : total_weight = 10)
  (h_apple : apple_weight = 3)
  (h_grape : grape_weight = 3)
  (h_strawberry : strawberry_weight = 3) :
  total_weight - (apple_weight + grape_weight + strawberry_weight) = 1 := by
  sorry

#check weight_of_oranges

end NUMINAMATH_CALUDE_weight_of_oranges_l3666_366694


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l3666_366683

/-- A figure consisting of six identical squares arranged in a "T" shape -/
structure TShapeFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure is 150 cm² -/
  total_area_eq : 6 * square_side^2 = 150

/-- The perimeter of the T-shaped figure -/
def perimeter (fig : TShapeFigure) : ℝ :=
  9 * fig.square_side

/-- Theorem stating that the perimeter of the T-shaped figure is 45 cm -/
theorem t_shape_perimeter (fig : TShapeFigure) : perimeter fig = 45 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l3666_366683


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3666_366687

theorem similar_triangle_shortest_side
  (a b c : ℝ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h3 : a = 15)           -- Length of one leg of the first triangle
  (h4 : c = 39)           -- Length of hypotenuse of the first triangle
  (k : ℝ)
  (h5 : k > 0)
  (h6 : k * c = 78)       -- Length of hypotenuse of the second triangle
  : k * a = 30            -- Length of the shortest side of the second triangle
:= by sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3666_366687


namespace NUMINAMATH_CALUDE_g_expression_l3666_366605

theorem g_expression (x : ℝ) (g : ℝ → ℝ) :
  (2 * x^5 + 4 * x^3 - 3 * x + g x = 7 * x^3 + 5 * x - 2) →
  (g x = -2 * x^5 + 3 * x^3 + 8 * x - 2) := by
  sorry

end NUMINAMATH_CALUDE_g_expression_l3666_366605


namespace NUMINAMATH_CALUDE_student_count_l3666_366680

/-- The number of storybooks distributed to the class -/
def total_books : ℕ := 60

/-- The number of students in the class -/
def num_students : ℕ := 20

theorem student_count :
  (num_students < total_books) ∧ 
  (total_books - num_students) % 2 = 0 ∧
  (total_books - num_students) / 2 = num_students :=
by sorry

end NUMINAMATH_CALUDE_student_count_l3666_366680
