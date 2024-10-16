import Mathlib

namespace NUMINAMATH_CALUDE_negative_expressions_l266_26651

/-- Represents a number with an approximate value -/
structure ApproxNumber where
  value : ℝ

/-- Given approximate values for P, Q, R, S, and T -/
def P : ApproxNumber := ⟨-4.2⟩
def Q : ApproxNumber := ⟨-2.3⟩
def R : ApproxNumber := ⟨0⟩
def S : ApproxNumber := ⟨1.1⟩
def T : ApproxNumber := ⟨2.7⟩

/-- Helper function to extract the value from ApproxNumber -/
def getValue (x : ApproxNumber) : ℝ := x.value

/-- Theorem stating which expressions are negative -/
theorem negative_expressions :
  (getValue P - getValue Q < 0) ∧
  (getValue P + getValue T < 0) ∧
  (getValue P * getValue Q ≥ 0) ∧
  ((getValue S / getValue Q) * getValue P ≥ 0) ∧
  (getValue R / (getValue P * getValue Q) ≥ 0) ∧
  ((getValue S + getValue T) / getValue R ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negative_expressions_l266_26651


namespace NUMINAMATH_CALUDE_intersection_range_l266_26660

theorem intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ 
    x₁^2 - y₁^2 = 6 ∧ 
    x₂^2 - y₂^2 = 6 ∧ 
    y₁ = k * x₁ + 2 ∧ 
    y₂ = k * x₂ + 2 ∧ 
    x₁ > 0 ∧ 
    x₂ > 0) → 
  -Real.sqrt 15 / 3 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l266_26660


namespace NUMINAMATH_CALUDE_largest_A_at_125_l266_26640

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence A_k -/
def A (k : ℕ) : ℝ := binomial 500 k * (0.3 ^ k)

theorem largest_A_at_125 : 
  ∀ k ∈ Finset.range 501, A 125 ≥ A k :=
sorry

end NUMINAMATH_CALUDE_largest_A_at_125_l266_26640


namespace NUMINAMATH_CALUDE_fraction_subtraction_l266_26699

theorem fraction_subtraction : 
  (((2 + 4 + 6 + 8) : ℚ) / (1 + 3 + 5 + 7)) - ((1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l266_26699


namespace NUMINAMATH_CALUDE_count_parallelepipeds_l266_26612

/-- The number of parallelepipeds formed in a rectangular parallelepiped -/
def num_parallelepipeds (m n k : ℕ) : ℚ :=
  (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8

/-- Theorem: The number of parallelepipeds formed in a rectangular parallelepiped
    with dimensions m × n × k, divided into unit cubes, is equal to
    (m * n * k * (m+1) * (n+1) * (k+1)) / 8 -/
theorem count_parallelepipeds (m n k : ℕ) :
  num_parallelepipeds m n k = (m * n * k * (m + 1) * (n + 1) * (k + 1) : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_count_parallelepipeds_l266_26612


namespace NUMINAMATH_CALUDE_line_intersection_canonical_form_l266_26649

/-- Given two planes in 3D space, this theorem proves that their line of intersection
    can be represented by specific canonical equations. -/
theorem line_intersection_canonical_form :
  ∀ (x y z : ℝ),
  (x + y - 2*z - 2 = 0 ∧ x - y + z + 2 = 0) →
  ∃ (t : ℝ), x = -t ∧ y = -3*t + 2 ∧ z = -2*t := by sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_form_l266_26649


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l266_26682

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 12) / (Nat.factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l266_26682


namespace NUMINAMATH_CALUDE_complex_fraction_real_l266_26685

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l266_26685


namespace NUMINAMATH_CALUDE_race_probability_l266_26627

theorem race_probability (total_cars : ℕ) (prob_X : ℚ) (prob_Z : ℚ) (prob_XYZ : ℚ) :
  total_cars = 8 →
  prob_X = 1/2 →
  prob_Z = 1/3 →
  prob_XYZ = 13/12 →
  ∃ (prob_Y : ℚ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_Y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l266_26627


namespace NUMINAMATH_CALUDE_exists_negative_monomial_degree_5_l266_26665

/-- A monomial in x and y -/
structure Monomial where
  coeff : ℤ
  x_power : ℕ
  y_power : ℕ

/-- The degree of a monomial -/
def Monomial.degree (m : Monomial) : ℕ := m.x_power + m.y_power

/-- A monomial is negative if its coefficient is negative -/
def Monomial.isNegative (m : Monomial) : Prop := m.coeff < 0

theorem exists_negative_monomial_degree_5 :
  ∃ m : Monomial, m.isNegative ∧ m.degree = 5 :=
sorry

end NUMINAMATH_CALUDE_exists_negative_monomial_degree_5_l266_26665


namespace NUMINAMATH_CALUDE_tan_alpha_and_tan_alpha_minus_pi_fourth_l266_26691

theorem tan_alpha_and_tan_alpha_minus_pi_fourth (α : Real) 
  (h : Real.tan (α / 2) = 1 / 2) : 
  Real.tan α = 4 / 3 ∧ Real.tan (α - π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_tan_alpha_minus_pi_fourth_l266_26691


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l266_26652

-- System 1
theorem system_one_solution (x y : ℝ) : 
  2 * x - y = 1 ∧ 7 * x - 3 * y = 4 → x = 1 ∧ y = 1 := by sorry

-- System 2
theorem system_two_solution (x y : ℝ) : 
  x / 2 + y / 3 = 6 ∧ x - y = -3 → x = 6 ∧ y = 9 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l266_26652


namespace NUMINAMATH_CALUDE_linear_function_value_l266_26667

/-- A function that is linear in both arguments -/
def LinearInBoth (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ y₁ y₂ a b : ℝ, 
    f (a*x₁ + b*x₂) y₁ = a*(f x₁ y₁) + b*(f x₂ y₁) ∧
    f x₁ (a*y₁ + b*y₂) = a*(f x₁ y₁) + b*(f x₁ y₂)

/-- The main theorem -/
theorem linear_function_value (f : ℝ → ℝ → ℝ) 
  (h_linear : LinearInBoth f)
  (h_3_3 : f 3 3 = 1/(3*3))
  (h_3_4 : f 3 4 = 1/(3*4))
  (h_4_3 : f 4 3 = 1/(4*3))
  (h_4_4 : f 4 4 = 1/(4*4)) :
  f 5 5 = 1/36 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l266_26667


namespace NUMINAMATH_CALUDE_shortest_median_le_longest_angle_bisector_l266_26613

/-- Represents a triangle with side lengths a ≤ b ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≤ b
  hbc : b ≤ c

/-- The length of the shortest median in a triangle -/
def shortestMedian (t : Triangle) : ℝ := sorry

/-- The length of the longest angle bisector in a triangle -/
def longestAngleBisector (t : Triangle) : ℝ := sorry

/-- Theorem: The shortest median is always shorter than or equal to the longest angle bisector -/
theorem shortest_median_le_longest_angle_bisector (t : Triangle) :
  shortestMedian t ≤ longestAngleBisector t := by sorry

end NUMINAMATH_CALUDE_shortest_median_le_longest_angle_bisector_l266_26613


namespace NUMINAMATH_CALUDE_remaining_money_after_gifts_l266_26672

def initial_budget : ℚ := 999
def shoes_cost : ℚ := 165
def yoga_mat_cost : ℚ := 85
def sports_watch_cost : ℚ := 215
def hand_weights_cost : ℚ := 60

theorem remaining_money_after_gifts :
  initial_budget - (shoes_cost + yoga_mat_cost + sports_watch_cost + hand_weights_cost) = 474 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_gifts_l266_26672


namespace NUMINAMATH_CALUDE_f_increasing_interval_f_greater_than_linear_l266_26617

noncomputable section

def f (x : ℝ) := Real.log x - (1/2) * (x - 1)^2

theorem f_increasing_interval (x : ℝ) (hx : x > 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = Real.sqrt 2 ∧
  ∀ y z, a < y ∧ y < z ∧ z < b → f y < f z :=
sorry

theorem f_greater_than_linear (k : ℝ) :
  (∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_f_greater_than_linear_l266_26617


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l266_26619

/-- A geometric sequence with a_2 = 2 and a_5 = 1/4 has a common ratio of 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1/4) :
  a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l266_26619


namespace NUMINAMATH_CALUDE_fraction_subtraction_l266_26639

theorem fraction_subtraction : 
  (3 + 7 + 11) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 7 + 11) = 33 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l266_26639


namespace NUMINAMATH_CALUDE_total_mail_delivered_l266_26615

/-- Represents the types of mail --/
inductive MailType
  | JunkMail
  | Magazine
  | Newspaper
  | Bill
  | Postcard

/-- Represents the mail distribution for a single house --/
structure HouseMailDistribution where
  junkMail : Nat
  magazines : Nat
  newspapers : Nat
  bills : Nat
  postcards : Nat

/-- Calculates the total pieces of mail for a single house --/
def totalMailForHouse (dist : HouseMailDistribution) : Nat :=
  dist.junkMail + dist.magazines + dist.newspapers + dist.bills + dist.postcards

/-- The mail distribution for the first house --/
def house1 : HouseMailDistribution :=
  { junkMail := 6, magazines := 5, newspapers := 3, bills := 4, postcards := 2 }

/-- The mail distribution for the second house --/
def house2 : HouseMailDistribution :=
  { junkMail := 4, magazines := 7, newspapers := 2, bills := 5, postcards := 3 }

/-- The mail distribution for the third house --/
def house3 : HouseMailDistribution :=
  { junkMail := 8, magazines := 3, newspapers := 4, bills := 6, postcards := 1 }

/-- Theorem stating that the total pieces of mail delivered to all three houses is 63 --/
theorem total_mail_delivered :
  totalMailForHouse house1 + totalMailForHouse house2 + totalMailForHouse house3 = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_mail_delivered_l266_26615


namespace NUMINAMATH_CALUDE_replaced_person_weight_l266_26616

/-- The weight of the replaced person in a group of 6 people -/
def weight_of_replaced_person (initial_count : ℕ) (new_person_weight : ℝ) (average_increase : ℝ) : ℝ :=
  new_person_weight - initial_count * average_increase

/-- Theorem stating the weight of the replaced person -/
theorem replaced_person_weight :
  weight_of_replaced_person 6 68 3.5 = 47 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l266_26616


namespace NUMINAMATH_CALUDE_a_2_equals_3_l266_26664

-- Define the sequence a_n
def a : ℕ → ℕ
  | n => 3^(n-1)

-- State the theorem
theorem a_2_equals_3 : a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2_equals_3_l266_26664


namespace NUMINAMATH_CALUDE_smallest_a_for_polynomial_l266_26675

theorem smallest_a_for_polynomial (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  r₁ * r₂ * r₃ = 1806 →
  r₁ + r₂ + r₃ = a →
  ∀ a' : ℤ, (∃ b' r₁' r₂' r₃' : ℕ+, 
    r₁' * r₂' * r₃' = 1806 ∧ 
    r₁' + r₂' + r₃' = a') → 
  a ≤ a' →
  a = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_for_polynomial_l266_26675


namespace NUMINAMATH_CALUDE_minimum_value_implies_c_l266_26629

def f (c : ℝ) (x : ℝ) : ℝ := x^4 - 8*x^2 + c

theorem minimum_value_implies_c (c : ℝ) :
  (∃ x₀ ∈ Set.Icc (-1) 3, f c x₀ = -14 ∧ ∀ x ∈ Set.Icc (-1) 3, f c x ≥ -14) →
  c = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_c_l266_26629


namespace NUMINAMATH_CALUDE_hamburger_cost_l266_26608

/-- Calculates the cost of a hamburger given the school lunch scenario --/
theorem hamburger_cost (third_grade_classes fourth_grade_classes fifth_grade_classes : ℕ)
  (third_grade_students fourth_grade_students fifth_grade_students : ℕ)
  (carrot_cost cookie_cost : ℚ)
  (total_lunch_cost : ℚ)
  (h1 : third_grade_classes = 5)
  (h2 : fourth_grade_classes = 4)
  (h3 : fifth_grade_classes = 4)
  (h4 : third_grade_students = 30)
  (h5 : fourth_grade_students = 28)
  (h6 : fifth_grade_students = 27)
  (h7 : carrot_cost = 0.5)
  (h8 : cookie_cost = 0.2)
  (h9 : total_lunch_cost = 1036) :
  let total_students := third_grade_classes * third_grade_students +
                        fourth_grade_classes * fourth_grade_students +
                        fifth_grade_classes * fifth_grade_students
  let total_side_cost := total_students * (carrot_cost + cookie_cost)
  let total_hamburger_cost := total_lunch_cost - total_side_cost
  total_hamburger_cost / total_students = 2.1 := by
sorry

end NUMINAMATH_CALUDE_hamburger_cost_l266_26608


namespace NUMINAMATH_CALUDE_dihedral_angle_measure_l266_26603

/-- Represents a dihedral angle formed by two plane mirrors -/
structure DihedralAngle where
  angle : ℝ

/-- Represents a light ray in the context of the problem -/
structure LightRay where
  perpendicular_to_edge : Bool
  parallel_to_first_mirror : Bool

/-- Represents the reflection pattern of the light ray -/
inductive ReflectionPattern
  | Alternating : ReflectionPattern

/-- Represents the result of the light ray's path -/
inductive PathResult
  | ReturnsAlongSamePath : PathResult

theorem dihedral_angle_measure 
  (d : DihedralAngle) 
  (l : LightRay) 
  (r : ReflectionPattern) 
  (p : PathResult) :
  l.perpendicular_to_edge = true →
  l.parallel_to_first_mirror = true →
  r = ReflectionPattern.Alternating →
  p = PathResult.ReturnsAlongSamePath →
  d.angle = 30 := by
  sorry

end NUMINAMATH_CALUDE_dihedral_angle_measure_l266_26603


namespace NUMINAMATH_CALUDE_red_subsequence_2019th_element_l266_26678

/-- Represents the number of elements in the nth group of the red-colored subsequence -/
def group_size (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the last element of the nth group in the red-colored subsequence -/
def last_element_of_group (n : ℕ) : ℕ := n * (2 * n - 1)

/-- Represents the sum of elements in the first n groups of the red-colored subsequence -/
def sum_of_elements (n : ℕ) : ℕ := (1 + group_size n) * n / 2

/-- The group number containing the 2019th element -/
def target_group : ℕ := 45

/-- The position of the 2019th element within its group -/
def position_in_group : ℕ := 83

/-- The theorem stating that the 2019th number in the red-colored subsequence is 3993 -/
theorem red_subsequence_2019th_element : 
  last_element_of_group (target_group - 1) + 1 + (position_in_group - 1) * 2 = 3993 :=
sorry

end NUMINAMATH_CALUDE_red_subsequence_2019th_element_l266_26678


namespace NUMINAMATH_CALUDE_min_value_theorem_l266_26657

theorem min_value_theorem (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  (1 / (2 * x)) + (x / (y + 1)) ≥ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l266_26657


namespace NUMINAMATH_CALUDE_trig_expression_equality_l266_26600

theorem trig_expression_equality : 
  (Real.sin (20 * π / 180) * Real.sqrt (1 + Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l266_26600


namespace NUMINAMATH_CALUDE_town_population_proof_l266_26656

theorem town_population_proof (new_people : ℕ) (moved_out : ℕ) (years : ℕ) (final_population : ℕ) :
  new_people = 100 →
  moved_out = 400 →
  years = 4 →
  final_population = 60 →
  (∃ original_population : ℕ,
    original_population = 1260 ∧
    final_population = ((original_population + new_people - moved_out) / 2^years)) :=
by sorry

end NUMINAMATH_CALUDE_town_population_proof_l266_26656


namespace NUMINAMATH_CALUDE_chips_probability_and_count_l266_26653

def total_bags : ℕ := 9
def bbq_bags : ℕ := 5

def prob_three_bbq : ℚ := 10 / 84

theorem chips_probability_and_count :
  (total_bags = 9) →
  (bbq_bags = 5) →
  (prob_three_bbq = 10 / 84) →
  (Nat.choose bbq_bags 3 * Nat.choose (total_bags - bbq_bags) 0) / Nat.choose total_bags 3 = prob_three_bbq ∧
  total_bags - bbq_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_chips_probability_and_count_l266_26653


namespace NUMINAMATH_CALUDE_casey_stay_is_three_months_l266_26610

/-- Calculates the number of months Casey stays at the motel --/
def casey_stay_duration (weekly_rate : ℕ) (monthly_rate : ℕ) (weeks_per_month : ℕ) (total_savings : ℕ) : ℕ :=
  let monthly_cost_weekly := weekly_rate * weeks_per_month
  let savings_per_month := monthly_cost_weekly - monthly_rate
  total_savings / savings_per_month

/-- Proves that Casey stays for 3 months given the specified rates and savings --/
theorem casey_stay_is_three_months :
  casey_stay_duration 280 1000 4 360 = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_stay_is_three_months_l266_26610


namespace NUMINAMATH_CALUDE_max_value_sin_cos_product_l266_26668

theorem max_value_sin_cos_product (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 4.5 ∧
  ∃ x y z : ℝ, (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
              (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) = 4.5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_product_l266_26668


namespace NUMINAMATH_CALUDE_money_division_l266_26662

theorem money_division (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 3200 →
  r - q = 4000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l266_26662


namespace NUMINAMATH_CALUDE_ellipse_chord_properties_l266_26674

/-- Given an ellipse with equation x²/2 + y² = 1, this theorem proves various properties
    related to chords and their midpoints. -/
theorem ellipse_chord_properties :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/2 + y^2 = 1}
  let P := (1/2, 1/2)
  let A := (2, 1)
  ∀ (x y : ℝ), (x, y) ∈ ellipse →
    (∃ (m b : ℝ), 2*x + 4*y - 3 = 0 ∧ 
      ∀ (x' y' : ℝ), (x', y') ∈ ellipse → 
        (y' - P.2 = m*(x' - P.1) + b ↔ y - P.2 = m*(x - P.1) + b)) ∧
    (∃ (x₀ y₀ : ℝ), x₀ + 4*y₀ = 0 ∧ -Real.sqrt 2 < x₀ ∧ x₀ < Real.sqrt 2 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₂ - y₁)/(x₂ - x₁) = 2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 - 2*x₀ + 2*y₀^2 - 2*y₀ = 0 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁ - A.2)/(x₁ - A.1) = (y₂ - A.2)/(x₂ - A.1) ∧
        x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) ∧
    (∃ (x₀ y₀ : ℝ), x₀^2 + 2*y₀^2 = 1 ∧
      ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ ellipse ∧ (x₂, y₂) ∈ ellipse ∧
        (y₁/x₁) * (y₂/x₂) = -1/2 ∧ x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_properties_l266_26674


namespace NUMINAMATH_CALUDE_ae_bc_ratio_l266_26618

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the triangles
def is_equilateral (X Y Z : ℝ × ℝ) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Define the midpoint
def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem ae_bc_ratio (A B C D E : ℝ × ℝ) :
  is_equilateral A B C →
  is_equilateral B C D →
  is_equilateral C D E →
  is_midpoint E C D →
  dist A E / dist B C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ae_bc_ratio_l266_26618


namespace NUMINAMATH_CALUDE_cube_difference_problem_l266_26663

theorem cube_difference_problem (a b c : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
  (sum_of_squares : a^2 + b^2 + c^2 = 160)
  (largest_sum : a = b + c)
  (difference : b - c = 4) :
  |b^3 - c^3| = 320 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_problem_l266_26663


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_l266_26634

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : x + y = -10) 
  (h2 : x = 25 / y) : 
  x^2 + y^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_l266_26634


namespace NUMINAMATH_CALUDE_salary_distribution_difference_l266_26690

/-- Proves that given a salary distribution among A, B, C, D in the proportion of 2 : 3 : 4 : 6,
    where B's share is $1050, the difference between D's and C's share is $700. -/
theorem salary_distribution_difference (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  2 * total = 15 * b →
  b = 1050 →
  6 * a = 2 * total →
  6 * b = 3 * total →
  6 * c = 4 * total →
  6 * d = 6 * total →
  d - c = 700 := by
  sorry

end NUMINAMATH_CALUDE_salary_distribution_difference_l266_26690


namespace NUMINAMATH_CALUDE_birthday_crayons_proof_l266_26681

/-- The number of crayons Paul got for his birthday. -/
def birthday_crayons : ℕ := 253

/-- The number of crayons Paul lost or gave away. -/
def lost_crayons : ℕ := 70

/-- The number of crayons Paul had left by the end of the school year. -/
def remaining_crayons : ℕ := 183

/-- Theorem stating that the number of crayons Paul got for his birthday
    is equal to the sum of lost crayons and remaining crayons. -/
theorem birthday_crayons_proof :
  birthday_crayons = lost_crayons + remaining_crayons :=
by sorry

end NUMINAMATH_CALUDE_birthday_crayons_proof_l266_26681


namespace NUMINAMATH_CALUDE_intersection_A_B_l266_26645

-- Define set A
def A : Set ℝ := {x | ∃ y, (x^2)/4 + (3*y^2)/4 = 1}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = x^2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l266_26645


namespace NUMINAMATH_CALUDE_loan_interest_period_l266_26644

/-- The problem of determining the number of years for B's gain --/
theorem loan_interest_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) : 
  principal = 1500 →
  rate_A = 0.10 →
  rate_C = 0.115 →
  gain = 67.5 →
  (rate_C - rate_A) * principal * 3 = gain :=
by sorry

end NUMINAMATH_CALUDE_loan_interest_period_l266_26644


namespace NUMINAMATH_CALUDE_sum_of_fractions_integer_l266_26630

theorem sum_of_fractions_integer (n : ℕ+) :
  (1/2 + 1/3 + 1/5 + 1/n.val : ℚ).isInt → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_integer_l266_26630


namespace NUMINAMATH_CALUDE_mode_of_sports_shoes_l266_26655

/-- Represents the sales data for a particular shoe size -/
structure SalesData :=
  (size : Float)
  (sales : Nat)

/-- Finds the mode of a list of SalesData -/
def findMode (data : List SalesData) : Float :=
  sorry

/-- The sales data for the sports shoes -/
def salesData : List SalesData := [
  ⟨24, 1⟩,
  ⟨24.5, 3⟩,
  ⟨25, 10⟩,
  ⟨25.5, 4⟩,
  ⟨26, 2⟩
]

theorem mode_of_sports_shoes :
  findMode salesData = 25 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_sports_shoes_l266_26655


namespace NUMINAMATH_CALUDE_star_four_three_l266_26697

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2

-- Theorem statement
theorem star_four_three : star 4 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_star_four_three_l266_26697


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l266_26698

theorem trigonometric_simplification (α : ℝ) :
  Real.sin ((5 / 2) * Real.pi + 4 * α) - 
  Real.sin ((5 / 2) * Real.pi + 2 * α) ^ 6 + 
  Real.cos ((7 / 2) * Real.pi - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (8 * α) * Real.sin (4 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l266_26698


namespace NUMINAMATH_CALUDE_optimal_viewing_distance_l266_26679

/-- The optimal distance from which to view a painting -/
theorem optimal_viewing_distance (a b : ℝ) (ha : a > 0) (hb : b > a) :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → 
    (b - a) / (y + a * b / y) ≤ (b - a) / (x + a * b / x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_optimal_viewing_distance_l266_26679


namespace NUMINAMATH_CALUDE_probability_of_one_hit_l266_26646

/-- Represents a single shot result -/
inductive Shot
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShots :=
  (first second third : Shot)

/-- Counts the number of hits in a ThreeShots -/
def count_hits (shots : ThreeShots) : Nat :=
  match shots with
  | ⟨Shot.Hit, Shot.Hit, Shot.Hit⟩ => 3
  | ⟨Shot.Hit, Shot.Hit, Shot.Miss⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Hit⟩ => 2
  | ⟨Shot.Miss, Shot.Hit, Shot.Hit⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Hit, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Hit⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Miss⟩ => 0

/-- Converts a digit to a Shot -/
def digit_to_shot (d : Nat) : Shot :=
  if d ∈ [1, 2, 3, 4] then Shot.Hit else Shot.Miss

/-- Converts a three-digit number to ThreeShots -/
def number_to_three_shots (n : Nat) : ThreeShots :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digit_to_shot d1, digit_to_shot d2, digit_to_shot d3⟩

theorem probability_of_one_hit (data : List Nat) : 
  data.length = 20 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length = 9 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length / data.length = 9 / 20 :=
sorry

end NUMINAMATH_CALUDE_probability_of_one_hit_l266_26646


namespace NUMINAMATH_CALUDE_power_of_27_l266_26607

theorem power_of_27 : (27 : ℝ) ^ (5/3) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_27_l266_26607


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l266_26654

/-- The longest segment in a cylinder with radius 5 and height 10 is 10√2 -/
theorem longest_segment_in_cylinder : ∀ (r h : ℝ),
  r = 5 → h = 10 → 
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l266_26654


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l266_26635

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 1000) → (∀ m : ℕ, m * (m + 1) < 1000 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 63 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l266_26635


namespace NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l266_26626

/-- Given a line segment with midpoint (-1, 3) and one endpoint (2, -4),
    prove that the other endpoint is (-4, 10). -/
theorem other_endpoint_of_line_segment
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (-1, 3))
  (h_endpoint1 : endpoint1 = (2, -4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-4, 10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end NUMINAMATH_CALUDE_other_endpoint_of_line_segment_l266_26626


namespace NUMINAMATH_CALUDE_game_eventually_ends_l266_26648

/-- The game state after k rounds of questioning -/
structure GameState where
  a : ℕ+  -- Player A's number
  b : ℕ+  -- Player B's number
  x : ℕ+  -- Smaller number on the board
  y : ℕ+  -- Larger number on the board
  k : ℕ   -- Number of rounds of questioning

/-- The game's rules and conditions -/
def validGame (g : GameState) : Prop :=
  g.x < g.y ∧ (g.a + g.b = g.x ∨ g.a + g.b = g.y)

/-- The condition for the game to end (one player knows the other's number) -/
def gameEnds (g : GameState) : Prop :=
  (g.k % 2 = 0 → g.y - g.x * (g.k + 1) < g.b) ∧
  (g.k % 2 = 1 → g.y - g.x * (g.k + 1) < g.a)

/-- The main theorem: the game will eventually end -/
theorem game_eventually_ends :
  ∀ g : GameState, validGame g → ∃ n : ℕ, gameEnds {a := g.a, b := g.b, x := g.x, y := g.y, k := n} :=
sorry

end NUMINAMATH_CALUDE_game_eventually_ends_l266_26648


namespace NUMINAMATH_CALUDE_equation_roots_min_modulus_l266_26666

noncomputable def find_a_b : ℝ × ℝ := sorry

theorem equation_roots (a b : ℝ) :
  find_a_b = (3, 3) :=
sorry

theorem min_modulus (a b : ℝ) (z : ℂ) :
  find_a_b = (a, b) →
  Complex.abs (z - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  ∀ w : ℂ, Complex.abs (w - (a + b * Complex.I)) = Complex.abs (2 / (1 + Complex.I)) →
  Complex.abs z ≤ Complex.abs w →
  Complex.abs z = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_equation_roots_min_modulus_l266_26666


namespace NUMINAMATH_CALUDE_range_of_f_l266_26623

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l266_26623


namespace NUMINAMATH_CALUDE_l_shape_area_is_22_l266_26687

def big_rectangle_length : ℕ := 8
def big_rectangle_width : ℕ := 5
def small_rectangle_length : ℕ := big_rectangle_length - 2
def small_rectangle_width : ℕ := big_rectangle_width - 2

def l_shape_area : ℕ := big_rectangle_length * big_rectangle_width - small_rectangle_length * small_rectangle_width

theorem l_shape_area_is_22 : l_shape_area = 22 := by
  sorry

end NUMINAMATH_CALUDE_l_shape_area_is_22_l266_26687


namespace NUMINAMATH_CALUDE_journey_time_calculation_l266_26661

theorem journey_time_calculation (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
    (h1 : total_distance = 560)
    (h2 : speed1 = 21)
    (h3 : speed2 = 24) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_calculation_l266_26661


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l266_26684

theorem modulus_of_complex_quotient :
  Complex.abs (Complex.I / (1 + 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l266_26684


namespace NUMINAMATH_CALUDE_bob_distance_when_meeting_l266_26628

/-- Prove that Bob walked 35 miles when he met Yolanda, given the following conditions:
  - The total distance between X and Y is 65 miles
  - Yolanda starts walking from X to Y
  - Bob starts walking from Y to X one hour after Yolanda
  - Yolanda's walking rate is 5 miles per hour
  - Bob's walking rate is 7 miles per hour
-/
theorem bob_distance_when_meeting (total_distance : ℝ) (yolanda_rate : ℝ) (bob_rate : ℝ)
  (h1 : total_distance = 65)
  (h2 : yolanda_rate = 5)
  (h3 : bob_rate = 7) :
  let time_to_meet := (total_distance - yolanda_rate) / (yolanda_rate + bob_rate)
  bob_rate * time_to_meet = 35 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_when_meeting_l266_26628


namespace NUMINAMATH_CALUDE_max_fruits_bought_l266_26601

/-- Represents the cost of each fruit in RM -/
structure FruitCost where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Represents the number of each fruit bought -/
structure FruitCount where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Calculates the total cost of fruits bought -/
def totalCost (cost : FruitCost) (count : FruitCount) : ℕ :=
  cost.apple * count.apple + cost.mango * count.mango + cost.papaya * count.papaya

/-- Calculates the total number of fruits bought -/
def totalFruits (count : FruitCount) : ℕ :=
  count.apple + count.mango + count.papaya

/-- Theorem stating the maximum number of fruits that can be bought under given conditions -/
theorem max_fruits_bought (cost : FruitCost) (count : FruitCount) 
    (h_apple_cost : cost.apple = 3)
    (h_mango_cost : cost.mango = 4)
    (h_papaya_cost : cost.papaya = 5)
    (h_at_least_one : count.apple ≥ 1 ∧ count.mango ≥ 1 ∧ count.papaya ≥ 1)
    (h_total_cost : totalCost cost count = 50) :
    totalFruits count ≤ 15 ∧ ∃ (max_count : FruitCount), totalFruits max_count = 15 ∧ totalCost cost max_count = 50 :=
  sorry


end NUMINAMATH_CALUDE_max_fruits_bought_l266_26601


namespace NUMINAMATH_CALUDE_average_temperature_last_four_days_l266_26647

/-- Given the temperatures for a week, prove the average temperature for the last four days. -/
theorem average_temperature_last_four_days 
  (temp_mon : ℝ)
  (temp_tue : ℝ)
  (temp_wed : ℝ)
  (temp_thu : ℝ)
  (temp_fri : ℝ)
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : temp_mon = 42)
  (h3 : temp_fri = 34) :
  (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 46 := by
sorry

end NUMINAMATH_CALUDE_average_temperature_last_four_days_l266_26647


namespace NUMINAMATH_CALUDE_power_23_mod_25_l266_26611

theorem power_23_mod_25 : 23^2057 % 25 = 16 := by sorry

end NUMINAMATH_CALUDE_power_23_mod_25_l266_26611


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l266_26689

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  (P = (2, -5 * π / 3) →
   symmetric_polar = (2, -2 * π / 3) ∧
   symmetric_cartesian = (-1, -Real.sqrt 3)) := by
  sorry

#check symmetric_point_coordinates

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l266_26689


namespace NUMINAMATH_CALUDE_ratio_change_proof_l266_26636

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original solution ratio -/
def original_ratio : SolutionRatio := ⟨2, 40, 100⟩

/-- The altered solution ratio -/
def altered_ratio : SolutionRatio := ⟨6, 60, 300⟩

/-- The factor by which the ratio of detergent to water changes -/
def ratio_change_factor : ℚ := 2

theorem ratio_change_proof : 
  (original_ratio.detergent / original_ratio.water) / 
  (altered_ratio.detergent / altered_ratio.water) = ratio_change_factor := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_proof_l266_26636


namespace NUMINAMATH_CALUDE_min_framing_feet_l266_26642

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photo -/
theorem min_framing_feet (original_width original_height border_width : ℕ) : 
  original_width = 5 →
  original_height = 7 →
  border_width = 3 →
  (((2 * original_width + 2 * border_width) + 
    (2 * original_height + 2 * border_width)) : ℕ) / 12 + 
    (if ((2 * original_width + 2 * border_width) + 
         (2 * original_height + 2 * border_width)) % 12 = 0 then 0 else 1) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_framing_feet_l266_26642


namespace NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l266_26620

theorem tomatoes_eaten_by_birds 
  (total_grown : ℕ) 
  (remaining : ℕ) 
  (h1 : total_grown = 127) 
  (h2 : remaining = 54) 
  (h3 : remaining * 2 = total_grown - (total_grown - remaining * 2)) : 
  total_grown - remaining * 2 = 19 := by
sorry

end NUMINAMATH_CALUDE_tomatoes_eaten_by_birds_l266_26620


namespace NUMINAMATH_CALUDE_bakers_purchase_problem_l266_26625

/-- A baker's purchase problem -/
theorem bakers_purchase_problem 
  (total_cost : ℕ)
  (flour_cost : ℕ)
  (egg_cost : ℕ)
  (egg_quantity : ℕ)
  (milk_cost : ℕ)
  (milk_quantity : ℕ)
  (soda_cost : ℕ)
  (soda_quantity : ℕ)
  (h1 : total_cost = 80)
  (h2 : flour_cost = 3)
  (h3 : egg_cost = 10)
  (h4 : egg_quantity = 3)
  (h5 : milk_cost = 5)
  (h6 : milk_quantity = 7)
  (h7 : soda_cost = 3)
  (h8 : soda_quantity = 2) :
  ∃ (flour_quantity : ℕ), 
    flour_quantity * flour_cost + 
    egg_quantity * egg_cost + 
    milk_quantity * milk_cost + 
    soda_quantity * soda_cost = total_cost ∧ 
    flour_quantity = 3 :=
by sorry

end NUMINAMATH_CALUDE_bakers_purchase_problem_l266_26625


namespace NUMINAMATH_CALUDE_set_equalities_l266_26694

-- Definition for even numbers
def IsEven (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k

-- Set 1
def Set1 : Set ℤ := {x | -1 ≤ x ∧ x < 5 ∧ IsEven x}

-- Set 2
def Set2 : Set ℤ := {0, 1, 2, 3, 4, 5}

-- Set 3
def Set3 : Set ℝ := {x | |x| = 1}

theorem set_equalities :
  (Set1 = {0, 2, 4}) ∧
  (Set2 = {x : ℤ | 0 ≤ x ∧ x ≤ 5}) ∧
  (Set3 = {x : ℝ | |x| = 1}) := by
  sorry


end NUMINAMATH_CALUDE_set_equalities_l266_26694


namespace NUMINAMATH_CALUDE_problem_statement_l266_26609

def n : ℕ := 2^2015 - 1

def s_q (q k : ℕ) : ℕ := sorry

def f_n (x : ℕ) : ℕ := sorry

def N : ℕ := sorry

theorem problem_statement : 
  N ≡ 382 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_problem_statement_l266_26609


namespace NUMINAMATH_CALUDE_gcf_lcm_problem_l266_26614

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := sorry

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := sorry

-- Theorem statement
theorem gcf_lcm_problem : GCF (LCM 9 15) (LCM 10 21) = 15 := by sorry

end NUMINAMATH_CALUDE_gcf_lcm_problem_l266_26614


namespace NUMINAMATH_CALUDE_vlad_sister_height_difference_l266_26650

/-- The height difference between two people given their heights in centimeters -/
def height_difference (height1 : ℝ) (height2 : ℝ) : ℝ :=
  height1 - height2

/-- Vlad's height in centimeters -/
def vlad_height : ℝ := 190.5

/-- Vlad's sister's height in centimeters -/
def sister_height : ℝ := 86.36

/-- Theorem: The height difference between Vlad and his sister is 104.14 centimeters -/
theorem vlad_sister_height_difference :
  height_difference vlad_height sister_height = 104.14 := by
  sorry


end NUMINAMATH_CALUDE_vlad_sister_height_difference_l266_26650


namespace NUMINAMATH_CALUDE_sum_of_xy_l266_26686

theorem sum_of_xy (x y : ℕ+) (h : x + y + x * y = 54) : x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xy_l266_26686


namespace NUMINAMATH_CALUDE_total_car_parts_cost_l266_26641

/-- The amount Mike spent on speakers -/
def speakers_cost : ℚ := 118.54

/-- The amount Mike spent on tires -/
def tires_cost : ℚ := 106.33

/-- The total amount Mike spent on car parts -/
def total_cost : ℚ := speakers_cost + tires_cost

/-- Theorem stating that the total cost of car parts is $224.87 -/
theorem total_car_parts_cost : total_cost = 224.87 := by sorry

end NUMINAMATH_CALUDE_total_car_parts_cost_l266_26641


namespace NUMINAMATH_CALUDE_cactus_jump_difference_l266_26632

theorem cactus_jump_difference (num_cacti : ℕ) (total_distance : ℝ) 
  (derek_hops_per_gap : ℕ) (rory_jumps_per_gap : ℕ) 
  (h1 : num_cacti = 31) 
  (h2 : total_distance = 3720) 
  (h3 : derek_hops_per_gap = 30) 
  (h4 : rory_jumps_per_gap = 10) : 
  ∃ (diff : ℝ), abs (diff - 8.27) < 0.01 ∧ 
  diff = (total_distance / ((num_cacti - 1) * rory_jumps_per_gap)) - 
         (total_distance / ((num_cacti - 1) * derek_hops_per_gap)) :=
by sorry

end NUMINAMATH_CALUDE_cactus_jump_difference_l266_26632


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l266_26638

/-- Geometric sequence with first term a and common ratio r -/
def geometric_sequence (a r : ℝ) : ℕ → ℝ := fun n => a * r^(n - 1)

/-- Sequence b_n as defined in the problem -/
def b_sequence (a : ℕ → ℝ) : ℕ → ℝ := 
  fun n => (Finset.range n).sum (fun k => (n - k) * a (k + 1))

/-- Sum of first n terms of a sequence -/
def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := (Finset.range n).sum (fun k => a (k + 1))

theorem geometric_sequence_problem (m : ℝ) (h_m : m ≠ 0) :
  ∃ (a : ℕ → ℝ), 
    (∃ r, a = geometric_sequence m r) ∧ 
    b_sequence a 1 = m ∧
    b_sequence a 2 = 3/2 * m ∧
    (∀ n : ℕ, n > 0 → 1 ≤ sequence_sum a n ∧ sequence_sum a n ≤ 3) →
    (∀ n, a n = m * (-1/2)^(n-1)) ∧
    (2 ≤ m ∧ m ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l266_26638


namespace NUMINAMATH_CALUDE_daria_concert_friends_l266_26692

def ticket_cost : ℕ := 90
def current_money : ℕ := 189
def additional_money_needed : ℕ := 171

def total_cost : ℕ := current_money + additional_money_needed

def total_tickets : ℕ := total_cost / ticket_cost

def number_of_friends : ℕ := total_tickets - 1

theorem daria_concert_friends : number_of_friends = 3 := by
  sorry

end NUMINAMATH_CALUDE_daria_concert_friends_l266_26692


namespace NUMINAMATH_CALUDE_equal_distance_travel_l266_26602

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 4) (h3 : v3 = 5) (h4 : t = 47/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 3 :=
by sorry

end NUMINAMATH_CALUDE_equal_distance_travel_l266_26602


namespace NUMINAMATH_CALUDE_infinite_solutions_implies_c_value_l266_26676

/-- If infinitely many values of y satisfy the equation 3(5 + 2cy) = 15y + 15 + y^2, then c = 2.5 -/
theorem infinite_solutions_implies_c_value (c : ℝ) : 
  (∀ y : ℝ, 3 * (5 + 2 * c * y) = 15 * y + 15 + y^2) → c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_implies_c_value_l266_26676


namespace NUMINAMATH_CALUDE_jacket_final_price_l266_26677

/-- The final price of a jacket after multiple discounts -/
theorem jacket_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) 
  (h1 : original_price = 25)
  (h2 : discount1 = 0.40)
  (h3 : discount2 = 0.25)
  (h4 : discount3 = 0.10) : 
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 10.125 := by
  sorry

end NUMINAMATH_CALUDE_jacket_final_price_l266_26677


namespace NUMINAMATH_CALUDE_car_speed_problem_l266_26622

theorem car_speed_problem (V : ℝ) (x : ℝ) : 
  let V1 := V * (1 - x / 100)
  let V2 := V1 * (1 + 0.5 * x / 100)
  V2 = V * (1 - 0.6 * x / 100) →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l266_26622


namespace NUMINAMATH_CALUDE_midpoint_locus_l266_26670

/-- Given a circle C with center (3,6) and radius 2√5, and a fixed point Q(-3,-6),
    the locus of the midpoint M of any point P on C and Q is described by the equation x^2 + y^2 = 5. -/
theorem midpoint_locus (P : ℝ × ℝ) (M : ℝ × ℝ) :
  (P.1 - 3)^2 + (P.2 - 6)^2 = 20 →
  M.1 = (P.1 + (-3)) / 2 →
  M.2 = (P.2 + (-6)) / 2 →
  M.1^2 + M.2^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l266_26670


namespace NUMINAMATH_CALUDE_equation_solution_l266_26669

theorem equation_solution :
  let S : Set ℂ := {x | (x - 4)^4 + (x - 6)^4 = 16}
  S = {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l266_26669


namespace NUMINAMATH_CALUDE_inequality_theorem_l266_26659

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  1 / (1 + a)^2 + 1 / (1 + b)^2 + 1 / (1 + c)^2 + 1 / (1 + d)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l266_26659


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l266_26696

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Represents the ratio of different pizza sizes ordered -/
structure PizzaRatio where
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) : ℕ :=
  let ratioSum := ratio.small + ratio.medium + ratio.large + ratio.extraLarge
  let pizzasPerRatio := totalPizzas / ratioSum
  (slices.small * ratio.small * pizzasPerRatio) +
  (slices.medium * ratio.medium * pizzasPerRatio) +
  (slices.large * ratio.large * pizzasPerRatio) +
  (slices.extraLarge * ratio.extraLarge * pizzasPerRatio)

theorem pizza_slices_theorem (slices : PizzaSlices) (ratio : PizzaRatio) (totalPizzas : ℕ) :
  slices.small = 6 →
  slices.medium = 8 →
  slices.large = 12 →
  slices.extraLarge = 16 →
  ratio.small = 3 →
  ratio.medium = 2 →
  ratio.large = 4 →
  ratio.extraLarge = 1 →
  totalPizzas = 20 →
  totalSlices slices ratio totalPizzas = 196 := by
  sorry

#eval totalSlices ⟨6, 8, 12, 16⟩ ⟨3, 2, 4, 1⟩ 20

end NUMINAMATH_CALUDE_pizza_slices_theorem_l266_26696


namespace NUMINAMATH_CALUDE_kelly_cheese_packages_l266_26693

/-- The number of packages of string cheese needed for school lunches --/
def string_cheese_packages (days_per_week : ℕ) (oldest_daily : ℕ) (youngest_daily : ℕ) 
  (cheeses_per_package : ℕ) (num_weeks : ℕ) : ℕ :=
  let total_cheeses := (oldest_daily + youngest_daily) * days_per_week * num_weeks
  (total_cheeses + cheeses_per_package - 1) / cheeses_per_package

/-- Theorem: Kelly needs 2 packages of string cheese for 4 weeks of school lunches --/
theorem kelly_cheese_packages : 
  string_cheese_packages 5 2 1 30 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_kelly_cheese_packages_l266_26693


namespace NUMINAMATH_CALUDE_two_inequalities_l266_26683

theorem two_inequalities :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_two_inequalities_l266_26683


namespace NUMINAMATH_CALUDE_bicycle_problem_l266_26671

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) : 
  let speed_B := 
    distance * (speed_ratio - 1) / (distance * speed_ratio * time_difference - distance * time_difference)
  speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_problem_l266_26671


namespace NUMINAMATH_CALUDE_transistors_in_2010_l266_26637

/-- The number of transistors in a typical CPU doubles every three years -/
def doubling_period : ℕ := 3

/-- The number of transistors in a typical CPU in 1992 -/
def initial_transistors : ℕ := 2000000

/-- The year from which we start counting -/
def initial_year : ℕ := 1992

/-- The year for which we want to calculate the number of transistors -/
def target_year : ℕ := 2010

/-- Calculates the number of transistors in a given year -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / doubling_period)

theorem transistors_in_2010 :
  transistors_in_year target_year = 128000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l266_26637


namespace NUMINAMATH_CALUDE_rachels_age_l266_26631

theorem rachels_age (rachel leah sam alex : ℝ) 
  (h1 : rachel = leah + 4)
  (h2 : rachel + leah = 2 * sam)
  (h3 : alex = 2 * rachel)
  (h4 : rachel + leah + sam + alex = 92) :
  rachel = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_rachels_age_l266_26631


namespace NUMINAMATH_CALUDE_problem_solution_l266_26688

def problem (a b m : ℝ × ℝ) : Prop :=
  let midpoint := λ (x y : ℝ × ℝ) => ((x.1 + y.1) / 2, (x.2 + y.2) / 2)
  m = midpoint (2 • a) (2 • b) ∧
  m = (4, 6) ∧
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 32

theorem problem_solution (a b m : ℝ × ℝ) :
  problem a b m := by sorry

end NUMINAMATH_CALUDE_problem_solution_l266_26688


namespace NUMINAMATH_CALUDE_max_savings_for_60_pieces_l266_26633

/-- Represents the pricing structure for raisin bread -/
structure BreadPricing where
  single_price : ℚ
  seven_price : ℚ
  dozen_price : ℚ

/-- Calculates the maximum amount that can be saved when buying bread -/
def max_savings (pricing : BreadPricing) (budget : ℚ) (min_pieces : ℕ) : ℚ :=
  sorry

/-- The theorem stating the maximum savings possible -/
theorem max_savings_for_60_pieces (pricing : BreadPricing) (budget : ℚ) :
  pricing.single_price = 0.30 →
  pricing.seven_price = 1 →
  pricing.dozen_price = 1.80 →
  budget = 10 →
  max_savings pricing budget 60 = 1.20 :=
by sorry

end NUMINAMATH_CALUDE_max_savings_for_60_pieces_l266_26633


namespace NUMINAMATH_CALUDE_four_number_sequence_l266_26624

theorem four_number_sequence (a b c d : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → -- Geometric sequence condition
  a + b + c = 19 →
  (∃ q : ℝ, c = b + q ∧ d = c + q) → -- Arithmetic sequence condition
  b + c + d = 12 →
  ((a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_four_number_sequence_l266_26624


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l266_26606

/-- A function that checks if a number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≤ 9

/-- A function that checks if four numbers are distinct -/
def areDistinct (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_sum_of_digits (A B C D : ℕ) :
  isDigit A → isDigit B → isDigit C → isDigit D →
  areDistinct A B C D →
  A + B + C + D = 17 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 ∧ ∃ (A' B' C' D' : ℕ), 
    isDigit A' ∧ isDigit B' ∧ isDigit C' ∧ isDigit D' ∧
    areDistinct A' B' C' D' ∧
    A' + B' + C' + D' = 17 ∧
    (A' + B') % (C' + D') = 0 ∧
    A' + B' = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l266_26606


namespace NUMINAMATH_CALUDE_charlie_age_when_jenny_twice_bobby_l266_26605

theorem charlie_age_when_jenny_twice_bobby (jenny charlie bobby : ℕ) : 
  jenny = charlie + 5 → 
  charlie = bobby + 3 → 
  ∃ x : ℕ, jenny + x = 2 * (bobby + x) ∧ charlie + x = 11 :=
by sorry

end NUMINAMATH_CALUDE_charlie_age_when_jenny_twice_bobby_l266_26605


namespace NUMINAMATH_CALUDE_population_change_l266_26658

theorem population_change (P : ℝ) : 
  P * 1.12 * 0.88 = 14784 → P = 15000 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l266_26658


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l266_26695

/-- Given two rectangles of equal area, where one rectangle measures 15 inches by 20 inches
    and the other rectangle is 6 inches long, prove that the width of the second rectangle
    is 50 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
  (h1 : carol_length = 15)
  (h2 : carol_width = 20)
  (h3 : jordan_length = 6)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 50 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l266_26695


namespace NUMINAMATH_CALUDE_factorial_equation_l266_26604

/-- Definition of factorial for positive integers -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem stating the equality of 7! * 11! and 15 * 12! -/
theorem factorial_equation : factorial 7 * factorial 11 = 15 * factorial 12 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l266_26604


namespace NUMINAMATH_CALUDE_anais_toy_difference_l266_26680

/-- Given a total number of toys and the number of toys Kamari has,
    calculates how many more toys Anais has than Kamari. -/
def toyDifference (total : ℕ) (kamariToys : ℕ) : ℕ :=
  total - kamariToys - kamariToys

/-- Proves that given the specific conditions, Anais has 30 more toys than Kamari. -/
theorem anais_toy_difference :
  let total := 160
  let kamariToys := 65
  toyDifference total kamariToys = 30 := by
  sorry

#eval toyDifference 160 65  -- Should output 30

end NUMINAMATH_CALUDE_anais_toy_difference_l266_26680


namespace NUMINAMATH_CALUDE_f_range_f_range_complete_l266_26621

noncomputable def f (x : ℝ) : ℝ :=
  |Real.sin x| / Real.sin x + Real.cos x / |Real.cos x| + |Real.tan x| / Real.tan x

theorem f_range :
  ∀ x : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 →
    f x = -1 ∨ f x = 3 :=
by sorry

theorem f_range_complete :
  ∃ x y : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 ∧
             Real.sin y ≠ 0 ∧ Real.cos y ≠ 0 ∧
             f x = -1 ∧ f y = 3 :=
by sorry

end NUMINAMATH_CALUDE_f_range_f_range_complete_l266_26621


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l266_26643

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l266_26643


namespace NUMINAMATH_CALUDE_speed_conversion_l266_26673

/-- Conversion factor from km/h to m/s -/
def km_h_to_m_s : ℝ := 0.27777777777778

/-- Given speed in km/h -/
def speed_km_h : ℝ := 0.8666666666666666

/-- Calculated speed in m/s -/
def speed_m_s : ℝ := 0.24074074074074

theorem speed_conversion : speed_km_h * km_h_to_m_s = speed_m_s := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l266_26673
