import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l693_69366

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

-- Define the condition for f having two zeros
def has_two_zeros (m : ℝ) : Prop := ∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0

-- Define the condition q
def condition_q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Main theorem
theorem range_of_m :
  ∀ m : ℝ, has_two_zeros m ∧ ¬(condition_q m) →
  m < -2 ∨ m ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l693_69366


namespace NUMINAMATH_CALUDE_salary_calculation_l693_69304

/-- Represents the man's original monthly salary in Rupees -/
def original_salary : ℝ := sorry

/-- The man's original savings rate as a decimal -/
def savings_rate : ℝ := 0.20

/-- The man's original rent expense rate as a decimal -/
def rent_rate : ℝ := 0.40

/-- The man's original utilities expense rate as a decimal -/
def utilities_rate : ℝ := 0.30

/-- The man's original groceries expense rate as a decimal -/
def groceries_rate : ℝ := 0.20

/-- The increase rate for rent as a decimal -/
def rent_increase : ℝ := 0.15

/-- The increase rate for utilities as a decimal -/
def utilities_increase : ℝ := 0.20

/-- The increase rate for groceries as a decimal -/
def groceries_increase : ℝ := 0.10

/-- The reduced savings amount in Rupees -/
def reduced_savings : ℝ := 180

theorem salary_calculation :
  original_salary * savings_rate - reduced_savings =
  original_salary * (rent_rate * (1 + rent_increase) +
                     utilities_rate * (1 + utilities_increase) +
                     groceries_rate * (1 + groceries_increase)) -
  original_salary * (rent_rate + utilities_rate + groceries_rate) ∧
  original_salary = 3000 :=
sorry

end NUMINAMATH_CALUDE_salary_calculation_l693_69304


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l693_69377

/-- Represents the sales prediction for a car dealership -/
structure SalesPrediction where
  sportsCarsRatio : ℕ
  sedansRatio : ℕ
  predictedSportsCars : ℕ

/-- Calculates the expected sedan sales and total vehicles needed -/
def calculateSales (pred : SalesPrediction) : ℕ × ℕ :=
  let expectedSedans := pred.predictedSportsCars * pred.sedansRatio / pred.sportsCarsRatio
  let totalVehicles := pred.predictedSportsCars + expectedSedans
  (expectedSedans, totalVehicles)

/-- Theorem stating the expected sales for the given scenario -/
theorem dealership_sales_prediction :
  let pred : SalesPrediction := {
    sportsCarsRatio := 3,
    sedansRatio := 5,
    predictedSportsCars := 36
  }
  calculateSales pred = (60, 96) := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l693_69377


namespace NUMINAMATH_CALUDE_paint_combinations_l693_69306

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 4

/-- The total number of combinations of color and painting method -/
def total_combinations : ℕ := num_colors * num_methods

theorem paint_combinations : total_combinations = 20 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l693_69306


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l693_69398

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) 
  (h1 : total_used = 1 / 2)
  (h2 : num_presents = 5) :
  total_used / num_presents = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l693_69398


namespace NUMINAMATH_CALUDE_article_cost_changes_l693_69372

theorem article_cost_changes (original_cost : ℝ) : 
  original_cost = 75 → 
  (original_cost * 1.2) * 0.8 = 72 := by
sorry

end NUMINAMATH_CALUDE_article_cost_changes_l693_69372


namespace NUMINAMATH_CALUDE_complex_equation_sum_l693_69347

theorem complex_equation_sum (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  Complex.mk a b = Complex.mk 1 1 * Complex.mk 2 (-1) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l693_69347


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_800_l693_69312

theorem modular_inverse_of_7_mod_800 :
  let a : ℕ := 7
  let m : ℕ := 800
  let inv : ℕ := 343
  (Nat.gcd a m = 1) →
  (inv < m) →
  (a * inv) % m = 1 →
  ∃ x : ℕ, x < m ∧ (a * x) % m = 1 ∧ x = inv :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_800_l693_69312


namespace NUMINAMATH_CALUDE_max_value_x_5_minus_4x_l693_69359

theorem max_value_x_5_minus_4x (x : ℝ) (h1 : 0 < x) (h2 : x < 5/4) :
  x * (5 - 4*x) ≤ 25/16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_5_minus_4x_l693_69359


namespace NUMINAMATH_CALUDE_equation_solutions_l693_69385

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^4 + (3 - x₁)^4 = 82) ∧ 
    (x₂^4 + (3 - x₂)^4 = 82) ∧ 
    (x₁ = 1.5 + Real.sqrt 1.375) ∧ 
    (x₂ = 1.5 - Real.sqrt 1.375) ∧
    (∀ x : ℝ, x^4 + (3 - x)^4 = 82 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l693_69385


namespace NUMINAMATH_CALUDE_long_jump_challenge_l693_69332

/-- Represents a student in the long jump challenge -/
structure Student where
  success_prob : ℚ
  deriving Repr

/-- Calculates the probability of a student achieving excellence -/
def excellence_prob (s : Student) : ℚ :=
  s.success_prob + (1 - s.success_prob) * s.success_prob

/-- Calculates the probability of a student achieving a good rating -/
def good_prob (s : Student) : ℚ :=
  1 - excellence_prob s

/-- The probability that exactly two out of three students achieve a good rating -/
def prob_two_good (s1 s2 s3 : Student) : ℚ :=
  excellence_prob s1 * good_prob s2 * good_prob s3 +
  good_prob s1 * excellence_prob s2 * good_prob s3 +
  good_prob s1 * good_prob s2 * excellence_prob s3

theorem long_jump_challenge (s1 s2 s3 : Student)
  (h1 : s1.success_prob = 3/4)
  (h2 : s2.success_prob = 1/2)
  (h3 : s3.success_prob = 1/3) :
  prob_two_good s1 s2 s3 = 77/576 := by
  sorry

#eval prob_two_good ⟨3/4⟩ ⟨1/2⟩ ⟨1/3⟩

end NUMINAMATH_CALUDE_long_jump_challenge_l693_69332


namespace NUMINAMATH_CALUDE_expression_simplification_l693_69346

variable (a b x : ℝ)

theorem expression_simplification (h : x ≥ a) :
  (Real.sqrt (b^2 + a^2 + x^2) - (x^3 - a^3) / Real.sqrt (b^2 + a^2 + x^2)) / (b^2 + a^2 + x^2) = 
  (b^2 + a^2 + a^3) / (b^2 + a^2 + x^2)^(3/2) := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l693_69346


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l693_69389

theorem bingo_prize_distribution (total_prize : ℕ) (first_winner_fraction : ℚ) 
  (remaining_winners : ℕ) (remaining_fraction : ℚ) :
  total_prize = 2400 →
  first_winner_fraction = 1 / 3 →
  remaining_winners = 10 →
  remaining_fraction = 1 / 10 →
  (total_prize - (first_winner_fraction * total_prize).num) / remaining_winners = 160 := by
  sorry

end NUMINAMATH_CALUDE_bingo_prize_distribution_l693_69389


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l693_69379

/-- Represents a quadratic function y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The symmetric axis of the quadratic function is x = 1 -/
def symmetric_axis (f : QuadraticFunction) : ℝ := 1

/-- The quadratic function passes through the point (-1, y₁) -/
def passes_through_minus_one (f : QuadraticFunction) (y₁ : ℝ) : Prop :=
  f.a * (-1)^2 + f.b * (-1) + f.c = y₁

/-- The quadratic function passes through the point (2, y₂) -/
def passes_through_two (f : QuadraticFunction) (y₂ : ℝ) : Prop :=
  f.a * 2^2 + f.b * 2 + f.c = y₂

/-- Theorem stating that y₁ > y₂ for the given conditions -/
theorem y1_greater_than_y2 (f : QuadraticFunction) (y₁ y₂ : ℝ)
  (h1 : passes_through_minus_one f y₁)
  (h2 : passes_through_two f y₂) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l693_69379


namespace NUMINAMATH_CALUDE_largest_nineteen_times_digit_sum_l693_69319

/-- The sum of digits of a positive integer -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: 399 is the largest positive integer equal to 19 times the sum of its digits -/
theorem largest_nineteen_times_digit_sum :
  ∀ n : ℕ, n > 0 → n = 19 * sum_of_digits n → n ≤ 399 := by
  sorry

end NUMINAMATH_CALUDE_largest_nineteen_times_digit_sum_l693_69319


namespace NUMINAMATH_CALUDE_joanne_coins_l693_69321

def coins_problem (first_hour : ℕ) (next_two_hours : ℕ) (fourth_hour : ℕ) (total_after : ℕ) : Prop :=
  let total_collected := first_hour + 2 * next_two_hours + fourth_hour
  total_collected - total_after = 15

theorem joanne_coins : coins_problem 15 35 50 120 := by
  sorry

end NUMINAMATH_CALUDE_joanne_coins_l693_69321


namespace NUMINAMATH_CALUDE_five_letter_words_with_at_least_two_vowels_l693_69364

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def count_words (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n

def count_words_with_exactly_k_vowels (k : Nat) : Nat :=
  Nat.choose word_length k * vowels.card ^ k * (alphabet.card - vowels.card) ^ (word_length - k)

theorem five_letter_words_with_at_least_two_vowels : 
  count_words alphabet word_length - 
  (count_words_with_exactly_k_vowels 0 + count_words_with_exactly_k_vowels 1) = 4192 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_with_at_least_two_vowels_l693_69364


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l693_69360

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 6) :
  (1 + 2 / (x + 1)) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l693_69360


namespace NUMINAMATH_CALUDE_cuboid_diagonal_squared_l693_69343

/-- The square of the diagonal of a cuboid equals the sum of the squares of its length, width, and height. -/
theorem cuboid_diagonal_squared (l w h d : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  d^2 = l^2 + w^2 + h^2 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_diagonal_squared_l693_69343


namespace NUMINAMATH_CALUDE_negation_of_existence_real_root_l693_69397

theorem negation_of_existence_real_root : 
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_real_root_l693_69397


namespace NUMINAMATH_CALUDE_equal_angles_l693_69382

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the right focus of the ellipse
def right_focus (F : ℝ × ℝ) : Prop := F.1 > 0 ∧ F.1^2 / 2 + F.2^2 = 1

-- Define a line passing through a point
def line_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop := l p.1 = p.2

-- Define the intersection points of the line and the ellipse
def intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line_through l A ∧ line_through l B ∧ A ≠ B

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_angles (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  right_focus F →
  line_through l F →
  intersection_points A B l →
  angle (0, 0) (2, 0) A = angle (0, 0) (2, 0) B :=
sorry

end NUMINAMATH_CALUDE_equal_angles_l693_69382


namespace NUMINAMATH_CALUDE_planes_parallel_to_same_plane_not_axiom_l693_69390

-- Define the type for geometric propositions
inductive GeometricProposition
  | PlanesParallelToSamePlaneAreParallel
  | ThreePointsDetermineUniquePlane
  | LineInPlaneImpliesAllPointsInPlane
  | TwoPlanesWithCommonPointHaveCommonLine

-- Define the set of axioms
def geometryAxioms : Set GeometricProposition :=
  { GeometricProposition.ThreePointsDetermineUniquePlane,
    GeometricProposition.LineInPlaneImpliesAllPointsInPlane,
    GeometricProposition.TwoPlanesWithCommonPointHaveCommonLine }

-- Theorem statement
theorem planes_parallel_to_same_plane_not_axiom :
  GeometricProposition.PlanesParallelToSamePlaneAreParallel ∉ geometryAxioms :=
by sorry

end NUMINAMATH_CALUDE_planes_parallel_to_same_plane_not_axiom_l693_69390


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l693_69327

theorem brand_preference_ratio (total : ℕ) (brand_x : ℕ) (h1 : total = 400) (h2 : brand_x = 360) :
  (brand_x : ℚ) / (total - brand_x : ℚ) = 9 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l693_69327


namespace NUMINAMATH_CALUDE_inequality_proof_l693_69356

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  Real.sqrt (x + (y - z)^2 / 12) + Real.sqrt (y + (x - z)^2 / 12) + Real.sqrt (z + (x - y)^2 / 12) ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l693_69356


namespace NUMINAMATH_CALUDE_area_of_bcd_l693_69307

-- Define the right triangular prism
structure RightTriangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  x : ℝ
  y : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_abc : x = (1/2) * a * b
  h_adc : y = (1/2) * b * c

-- Theorem statement
theorem area_of_bcd (prism : RightTriangularPrism) : 
  (1/2) * prism.b * prism.c = prism.y := by
  sorry

end NUMINAMATH_CALUDE_area_of_bcd_l693_69307


namespace NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l693_69365

/-- Given a cylinder with volume 72π cm³, prove that a sphere with the same radius has volume 96π cm³ -/
theorem sphere_volume_from_cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 → π * r^2 * h = 72 * π → (4/3) * π * r^3 = 96 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_cylinder_volume_l693_69365


namespace NUMINAMATH_CALUDE_si_o_bond_is_polar_covalent_l693_69374

-- Define the electronegativity values
def electronegativity_Si : ℝ := 1.90
def electronegativity_O : ℝ := 3.44

-- Define the range for polar covalent bonds
def polar_covalent_lower_bound : ℝ := 0.5
def polar_covalent_upper_bound : ℝ := 1.7

-- Define a function to check if a bond is polar covalent
def is_polar_covalent (electronegativity_diff : ℝ) : Prop :=
  polar_covalent_lower_bound ≤ electronegativity_diff ∧
  electronegativity_diff ≤ polar_covalent_upper_bound

-- Theorem: The silicon-oxygen bonds in SiO2 are polar covalent
theorem si_o_bond_is_polar_covalent :
  is_polar_covalent (electronegativity_O - electronegativity_Si) :=
by
  sorry


end NUMINAMATH_CALUDE_si_o_bond_is_polar_covalent_l693_69374


namespace NUMINAMATH_CALUDE_factor_polynomial_l693_69381

theorem factor_polynomial (x : ℝ) : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l693_69381


namespace NUMINAMATH_CALUDE_additional_racks_needed_additional_racks_needed_is_one_l693_69308

-- Define the given constants
def flour_per_bag : ℕ := 12
def bags_of_flour : ℕ := 5
def cups_per_pound : ℕ := 3
def pounds_per_rack : ℕ := 5
def owned_racks : ℕ := 3

-- Define the theorem
theorem additional_racks_needed : ℕ :=
  let total_flour : ℕ := flour_per_bag * bags_of_flour
  let total_pounds : ℕ := total_flour / cups_per_pound
  let capacity : ℕ := owned_racks * pounds_per_rack
  let remaining : ℕ := total_pounds - capacity
  (remaining + pounds_per_rack - 1) / pounds_per_rack

-- Proof
theorem additional_racks_needed_is_one : additional_racks_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_additional_racks_needed_additional_racks_needed_is_one_l693_69308


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l693_69330

theorem arithmetic_calculation : 4 * (8 - 3)^2 / 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l693_69330


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l693_69302

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid arrangements for a centipede to put on its socks and shoes -/
theorem centipede_sock_shoe_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_legs) := by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_arrangements_l693_69302


namespace NUMINAMATH_CALUDE_element_in_set_M_l693_69358

def U : Finset Nat := {1, 2, 3, 4, 5}

theorem element_in_set_M (M : Finset Nat) 
  (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_M_l693_69358


namespace NUMINAMATH_CALUDE_dad_has_three_eyes_l693_69314

/-- A monster family with a specified number of eyes for each member -/
structure MonsterFamily where
  mom_eyes : ℕ
  dad_eyes : ℕ
  num_kids : ℕ
  kid_eyes : ℕ
  total_eyes : ℕ

/-- Theorem stating that given the conditions of the monster family, the dad must have 3 eyes -/
theorem dad_has_three_eyes (family : MonsterFamily)
  (h1 : family.mom_eyes = 1)
  (h2 : family.num_kids = 3)
  (h3 : family.kid_eyes = 4)
  (h4 : family.total_eyes = 16) :
  family.dad_eyes = 3 := by
  sorry


end NUMINAMATH_CALUDE_dad_has_three_eyes_l693_69314


namespace NUMINAMATH_CALUDE_balloon_count_total_l693_69305

/-- Calculate the total number of balloons for each color given Sara's and Sandy's balloons -/
theorem balloon_count_total 
  (R1 G1 B1 Y1 O1 R2 G2 B2 Y2 O2 : ℕ) 
  (h1 : R1 = 31) (h2 : G1 = 15) (h3 : B1 = 12) (h4 : Y1 = 18) (h5 : O1 = 10)
  (h6 : R2 = 24) (h7 : G2 = 7)  (h8 : B2 = 14) (h9 : Y2 = 20) (h10 : O2 = 16) :
  R1 + R2 = 55 ∧ 
  G1 + G2 = 22 ∧ 
  B1 + B2 = 26 ∧ 
  Y1 + Y2 = 38 ∧ 
  O1 + O2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_total_l693_69305


namespace NUMINAMATH_CALUDE_smallest_a_is_2_pow_16_l693_69316

/-- The number of factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The smallest natural number satisfying the given condition -/
def smallest_a : ℕ := sorry

/-- The theorem statement -/
theorem smallest_a_is_2_pow_16 :
  (∀ a : ℕ, num_factors (a^2) = num_factors a + 16 → a ≥ smallest_a) ∧
  num_factors (smallest_a^2) = num_factors smallest_a + 16 ∧
  smallest_a = 2^16 := by sorry

end NUMINAMATH_CALUDE_smallest_a_is_2_pow_16_l693_69316


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_proof_l693_69396

def students_not_eating_lunch (total_students cafeteria_students : ℕ) : ℕ :=
  total_students - (cafeteria_students + 3 * cafeteria_students)

theorem students_not_eating_lunch_proof :
  students_not_eating_lunch 60 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_proof_l693_69396


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l693_69335

theorem max_value_trig_expression (x : ℝ) : 2 * Real.cos x + 3 * Real.sin x + 1 ≤ Real.sqrt 13 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l693_69335


namespace NUMINAMATH_CALUDE_expression_simplification_l693_69323

theorem expression_simplification :
  ∀ q : ℚ, ((7*q+3)-3*q*2)*4+(5-2/4)*(8*q-12) = 40*q - 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l693_69323


namespace NUMINAMATH_CALUDE_xy_value_l693_69383

theorem xy_value (x y : ℝ) 
  (eq1 : (4 : ℝ)^x / (2 : ℝ)^(x + y) = 8)
  (eq2 : (9 : ℝ)^(x + y) / (3 : ℝ)^(5 * y) = 243) :
  x * y = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l693_69383


namespace NUMINAMATH_CALUDE_simplify_expression_l693_69367

theorem simplify_expression (m n : ℝ) : m - n - (m + n) = -2 * n := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l693_69367


namespace NUMINAMATH_CALUDE_inverse_function_property_l693_69345

-- Define the function f
def f : ℕ → ℕ
| 1 => 3
| 2 => 13
| 3 => 8
| 5 => 1
| 8 => 0
| 13 => 5
| _ => 0  -- Default case for other inputs

-- Define the inverse function f_inv
def f_inv : ℕ → ℕ
| 0 => 8
| 1 => 5
| 3 => 1
| 5 => 13
| 8 => 3
| 13 => 2
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_function_property :
  f_inv ((f_inv 5 + f_inv 13) / f_inv 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_property_l693_69345


namespace NUMINAMATH_CALUDE_circle_M_equations_l693_69303

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of circle M lies in part (I)
def centerLine (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the equation of circle M for part (I)
def circleM1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 25

-- Define the equation of circle M for part (II)
def circleM2 (x y : ℝ) : Prop := (x + 7/2)^2 + (y - 1/2)^2 = 25/2

theorem circle_M_equations :
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM1 x0 y0) →
    (∃ xc yc : ℝ, centerLine xc yc ∧ circleM1 x y)) ∧
  (∀ x y : ℝ, (∃ x0 y0 : ℝ, circle1 x0 y0 ∧ circle2 x0 y0 ∧ circleM2 x0 y0) →
    circleM2 x y) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_equations_l693_69303


namespace NUMINAMATH_CALUDE_first_two_digits_sum_l693_69370

/-- The number of integer lattice points (x, y) satisfying 4x^2 + 9y^2 ≤ 1000000000 -/
def N : ℕ := sorry

/-- The first digit of N -/
def a : ℕ := sorry

/-- The second digit of N -/
def b : ℕ := sorry

/-- Theorem stating that 10a + b equals 52 -/
theorem first_two_digits_sum : 10 * a + b = 52 := by sorry

end NUMINAMATH_CALUDE_first_two_digits_sum_l693_69370


namespace NUMINAMATH_CALUDE_alyssa_marbles_cost_l693_69395

/-- The amount Alyssa spent on marbles -/
def marbles_cost (football_cost total_cost : ℚ) : ℚ :=
  total_cost - football_cost

/-- Proof that Alyssa spent $6.59 on marbles -/
theorem alyssa_marbles_cost :
  let football_cost : ℚ := 571/100
  let total_cost : ℚ := 1230/100
  marbles_cost football_cost total_cost = 659/100 := by
sorry

end NUMINAMATH_CALUDE_alyssa_marbles_cost_l693_69395


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l693_69388

/-- Given that Bryan has 56 books in each bookshelf and 504 books in total,
    prove that he has 9 bookshelves. -/
theorem bryans_bookshelves (books_per_shelf : ℕ) (total_books : ℕ) 
    (h1 : books_per_shelf = 56) (h2 : total_books = 504) :
    total_books / books_per_shelf = 9 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l693_69388


namespace NUMINAMATH_CALUDE_least_value_expression_l693_69371

theorem least_value_expression (x : ℝ) (h : x < -2) :
  (2 * x ≤ x) ∧ (2 * x ≤ x + 2) ∧ (2 * x ≤ (1/2) * x) ∧ (2 * x ≤ x - 2) := by
  sorry

end NUMINAMATH_CALUDE_least_value_expression_l693_69371


namespace NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l693_69340

/-- Given a line segment with one endpoint (6, 2) and midpoint (5, 7),
    the sum of the coordinates of the other endpoint is 16. -/
theorem endpoint_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 2) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 16

/-- Proof of the theorem -/
theorem endpoint_sum_proof : ∃ (endpoint2 : ℝ × ℝ), endpoint_sum (6, 2) (5, 7) endpoint2 :=
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_endpoint_sum_proof_l693_69340


namespace NUMINAMATH_CALUDE_product_equals_143_l693_69368

/-- Convert a binary number (represented as a list of bits) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Convert a ternary number (represented as a list of digits) to its decimal equivalent -/
def ternary_to_decimal (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The binary representation of 1101₂ -/
def binary_num : List Bool := [true, false, true, true]

/-- The ternary representation of 102₃ -/
def ternary_num : List ℕ := [2, 0, 1]

theorem product_equals_143 : 
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_143_l693_69368


namespace NUMINAMATH_CALUDE_min_four_digit_number_l693_69325

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the first two digits (AB) of a four-digit number -/
def first_two_digits (n : FourDigitNumber) : ℕ :=
  n.value / 100

/-- Returns the last two digits (CD) of a four-digit number -/
def last_two_digits (n : FourDigitNumber) : ℕ :=
  n.value % 100

/-- The property that ABCD + AB × CD is a multiple of 1111 -/
def satisfies_condition (n : FourDigitNumber) : Prop :=
  ∃ k : ℕ, n.value + (first_two_digits n) * (last_two_digits n) = 1111 * k

theorem min_four_digit_number :
  ∀ n : FourDigitNumber, satisfies_condition n → n.value ≥ 1729 :=
by sorry

end NUMINAMATH_CALUDE_min_four_digit_number_l693_69325


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l693_69352

/-- A binomial distribution with n trials and probability p -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type := Unit

/-- The probability of getting at least k successes in a binomial distribution -/
def P_at_least (dist : binomial_distribution n p) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem 
  (p : ℝ) 
  (ξ : binomial_distribution 2 p) 
  (η : binomial_distribution 4 p) 
  (h : P_at_least ξ 1 = 5/9) :
  P_at_least η 2 = 11/27 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l693_69352


namespace NUMINAMATH_CALUDE_target_digit_is_seven_l693_69333

/-- The decimal representation of 13/481 -/
def decimal_rep : ℚ := 13 / 481

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 3

/-- The position of the digit we're looking for -/
def target_position : ℕ := 222

/-- The function that returns the nth digit after the decimal point -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem target_digit_is_seven : nth_digit target_position = 7 := by
  sorry

end NUMINAMATH_CALUDE_target_digit_is_seven_l693_69333


namespace NUMINAMATH_CALUDE_distinct_subscription_selections_l693_69351

def number_of_providers : ℕ := 25
def number_of_siblings : ℕ := 4

theorem distinct_subscription_selections :
  (number_of_providers - 0) *
  (number_of_providers - 1) *
  (number_of_providers - 2) *
  (number_of_providers - 3) = 303600 := by
  sorry

end NUMINAMATH_CALUDE_distinct_subscription_selections_l693_69351


namespace NUMINAMATH_CALUDE_circular_segment_probability_l693_69391

/-- The ratio of circumference to diameter in ancient Chinese mathematics -/
def ancient_pi : ℚ := 3

/-- The area of a circular segment given chord length and height difference -/
def segment_area (a c : ℚ) : ℚ := (1/2) * a * (a + c)

/-- The probability of a point mass landing in a circular segment -/
theorem circular_segment_probability (c a : ℚ) (h1 : c = 6) (h2 : a = 1) :
  let r := (c^2 / 4 + a^2) / (2 * a)
  let circle_area := ancient_pi * r^2
  segment_area a c / circle_area = 7 / 150 := by
  sorry

end NUMINAMATH_CALUDE_circular_segment_probability_l693_69391


namespace NUMINAMATH_CALUDE_sport_to_standard_ratio_l693_69331

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation ratio -/
def standard : DrinkRatio :=
  { flavoring := 1,
    corn_syrup := 12,
    water := 30 }

/-- The sport formulation ratio -/
def sport : DrinkRatio :=
  { flavoring := (15 : ℚ) / 60,
    corn_syrup := 1,
    water := 15 }

/-- The ratio of flavoring to corn syrup for a given formulation -/
def flavoring_to_corn_syrup_ratio (d : DrinkRatio) : ℚ :=
  d.flavoring / d.corn_syrup

theorem sport_to_standard_ratio :
  flavoring_to_corn_syrup_ratio sport / flavoring_to_corn_syrup_ratio standard = 3 := by
  sorry

end NUMINAMATH_CALUDE_sport_to_standard_ratio_l693_69331


namespace NUMINAMATH_CALUDE_A_power_2023_l693_69337

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, -1, 0;
     1,  0, 0;
     0,  0, 1]

theorem A_power_2023 :
  A ^ 2023 = !![0,  1, 0;
                -1,  0, 0;
                 0,  0, 1] := by sorry

end NUMINAMATH_CALUDE_A_power_2023_l693_69337


namespace NUMINAMATH_CALUDE_folding_theorem_l693_69353

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a line segment -/
structure Segment where
  length : ℝ

/-- Represents the folding problem -/
def FoldingProblem (rect : Rectangle) : Prop :=
  ∃ (CC' EF : Segment),
    rect.width = 240 ∧
    rect.height = 288 ∧
    CC'.length = 312 ∧
    EF.length = 260

/-- The main theorem -/
theorem folding_theorem (rect : Rectangle) :
  FoldingProblem rect :=
sorry

end NUMINAMATH_CALUDE_folding_theorem_l693_69353


namespace NUMINAMATH_CALUDE_larger_number_l693_69311

theorem larger_number (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l693_69311


namespace NUMINAMATH_CALUDE_kates_discount_is_eight_percent_l693_69373

-- Define the bills and total paid
def bobs_bill : ℚ := 30
def kates_bill : ℚ := 25
def total_paid : ℚ := 53

-- Define the discount percentage
def discount_percentage : ℚ := (bobs_bill + kates_bill - total_paid) / kates_bill * 100

-- Theorem statement
theorem kates_discount_is_eight_percent :
  discount_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_kates_discount_is_eight_percent_l693_69373


namespace NUMINAMATH_CALUDE_blue_to_red_ratio_l693_69320

def cube_side_length : ℕ := 13

def red_face_area : ℕ := 6 * cube_side_length^2

def total_face_area : ℕ := 6 * cube_side_length^3

def blue_face_area : ℕ := total_face_area - red_face_area

theorem blue_to_red_ratio :
  blue_face_area / red_face_area = 12 := by
  sorry

end NUMINAMATH_CALUDE_blue_to_red_ratio_l693_69320


namespace NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l693_69350

theorem power_three_plus_four_mod_five : 3^75 + 4 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_three_plus_four_mod_five_l693_69350


namespace NUMINAMATH_CALUDE_bank_charge_increase_l693_69361

/-- The percentage increase in the ratio of price to transactions from the old
    charging system to the new charging system -/
theorem bank_charge_increase (old_price : ℝ) (old_transactions : ℕ)
    (new_price : ℝ) (new_transactions : ℕ) :
    old_price = 1 →
    old_transactions = 5 →
    new_price = 0.75 →
    new_transactions = 3 →
    (((new_price / new_transactions) - (old_price / old_transactions)) /
     (old_price / old_transactions)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bank_charge_increase_l693_69361


namespace NUMINAMATH_CALUDE_multiply_fractions_result_l693_69309

theorem multiply_fractions_result : (77 / 4) * (5 / 2) = 48 + 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_result_l693_69309


namespace NUMINAMATH_CALUDE_angle_between_sides_l693_69363

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] :=
  (a b c d : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)

-- Define the theorem
theorem angle_between_sides (q : CyclicQuadrilateral ℝ) :
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) =
  Real.arccos ((q.a^2 + q.b^2 - q.d^2 - q.c^2) / (2 * (q.a * q.b + q.d * q.c))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_sides_l693_69363


namespace NUMINAMATH_CALUDE_equation_solution_l693_69300

theorem equation_solution (a b c d : ℝ) :
  a^2 + b^2 + c^2 + 1 = d + Real.sqrt (a + b + c - d) →
  d = 5/4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l693_69300


namespace NUMINAMATH_CALUDE_workshop_workers_l693_69357

theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) (h1 : average_salary = 6750) 
  (h2 : technician_salary = 12000) (h3 : other_salary = 6000) 
  (h4 : num_technicians = 7) : 
  ∃ (total_workers : ℕ), total_workers = 56 ∧ 
  average_salary * total_workers = 
    num_technicians * technician_salary + 
    (total_workers - num_technicians) * other_salary :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l693_69357


namespace NUMINAMATH_CALUDE_polynomial_equality_l693_69324

theorem polynomial_equality (P : ℝ → ℝ) :
  (∀ a b c : ℝ, P (a + b - 2 * c) + P (b + c - 2 * a) + P (c + a - 2 * b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ a b : ℝ, ∀ x : ℝ, P x = a * x^2 + b * x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l693_69324


namespace NUMINAMATH_CALUDE_total_card_value_is_244_l693_69338

def jenny_initial_cards : ℕ := 6
def jenny_rare_percentage : ℚ := 1/2
def orlando_extra_cards : ℕ := 2
def orlando_rare_percentage : ℚ := 2/5
def richard_card_multiplier : ℕ := 3
def richard_rare_percentage : ℚ := 1/4
def jenny_additional_cards : ℕ := 4
def holographic_card_value : ℕ := 15
def first_edition_card_value : ℕ := 8
def rare_card_value : ℕ := 10
def non_rare_card_value : ℕ := 3

def total_card_value : ℕ := sorry

theorem total_card_value_is_244 : total_card_value = 244 := by sorry

end NUMINAMATH_CALUDE_total_card_value_is_244_l693_69338


namespace NUMINAMATH_CALUDE_otimes_example_l693_69393

/-- Custom operation ⊗ defined as a ⊗ b = a² - ab -/
def otimes (a b : ℤ) : ℤ := a^2 - a * b

/-- Theorem stating that 4 ⊗ [2 ⊗ (-5)] = -40 -/
theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end NUMINAMATH_CALUDE_otimes_example_l693_69393


namespace NUMINAMATH_CALUDE_polynomial_factorization_l693_69301

/-- A polynomial in x and y with a parameter n -/
def polynomial (n : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + 2*x + n*y - n

/-- Predicate to check if a polynomial can be factored into two linear factors with integer coefficients -/
def has_linear_factors (p : ℤ → ℤ → ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ x y, p x y = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (n : ℤ) :
  has_linear_factors (polynomial n) ↔ n = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l693_69301


namespace NUMINAMATH_CALUDE_sum_x_y_equals_negative_two_l693_69349

theorem sum_x_y_equals_negative_two (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_negative_two_l693_69349


namespace NUMINAMATH_CALUDE_c_rent_share_is_72_l693_69310

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of a renter in ox-months -/
def share (r : Renter) : ℕ := r.oxen * r.months

/-- Represents the pasture rental scenario -/
structure PastureRental where
  a : Renter
  b : Renter
  c : Renter
  totalRent : ℕ

/-- Calculates the total share of all renters -/
def totalShare (pr : PastureRental) : ℕ :=
  share pr.a + share pr.b + share pr.c

/-- Calculates the rent share for a specific renter -/
def rentShare (pr : PastureRental) (r : Renter) : ℚ :=
  (share r : ℚ) / (totalShare pr : ℚ) * pr.totalRent

theorem c_rent_share_is_72 (pr : PastureRental) : 
  pr.a = { oxen := 10, months := 7 } →
  pr.b = { oxen := 12, months := 5 } →
  pr.c = { oxen := 15, months := 3 } →
  pr.totalRent = 280 →
  rentShare pr pr.c = 72 := by
  sorry

#check c_rent_share_is_72

end NUMINAMATH_CALUDE_c_rent_share_is_72_l693_69310


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l693_69339

/-- Represents a right circular cone -/
structure Cone :=
  (diameter : ℝ)
  (altitude : ℝ)

/-- Represents a right circular cylinder -/
structure Cylinder :=
  (radius : ℝ)

/-- 
Theorem: The radius of a cylinder inscribed in a cone
Given:
  - The cylinder's diameter is equal to its height
  - The cone has a diameter of 12 and an altitude of 15
  - The axes of the cylinder and cone coincide
Prove: The radius of the cylinder is 10/3
-/
theorem inscribed_cylinder_radius (cone : Cone) (cyl : Cylinder) :
  cone.diameter = 12 →
  cone.altitude = 15 →
  cyl.radius * 2 = cyl.radius * 2 →  -- cylinder's diameter equals its height
  cyl.radius = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l693_69339


namespace NUMINAMATH_CALUDE_rectangle_area_l693_69348

theorem rectangle_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * (Real.sqrt (diagonal^2 - side^2)) → area = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l693_69348


namespace NUMINAMATH_CALUDE_prob_all_same_color_l693_69318

/-- The probability of picking all same-colored candies from a jar -/
theorem prob_all_same_color (red blue : ℕ) (h_red : red = 15) (h_blue : blue = 5) :
  let total := red + blue
  let prob_terry_red := (red * (red - 1)) / (total * (total - 1))
  let prob_mary_red_given_terry_red := (red - 2) / (total - 2)
  let prob_all_red := prob_terry_red * prob_mary_red_given_terry_red
  let prob_terry_blue := (blue * (blue - 1)) / (total * (total - 1))
  let prob_mary_blue_given_terry_blue := (blue - 2) / (total - 2)
  let prob_all_blue := prob_terry_blue * prob_mary_blue_given_terry_blue
  prob_all_red + prob_all_blue = 31 / 76 := by
sorry

end NUMINAMATH_CALUDE_prob_all_same_color_l693_69318


namespace NUMINAMATH_CALUDE_cone_volume_ratio_cone_C_D_volume_ratio_l693_69380

/-- The ratio of the volumes of two cones with swapped radius and height is 1/2 -/
theorem cone_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

/-- The ratio of the volumes of cones C and D is 1/2 -/
theorem cone_C_D_volume_ratio : 
  let r : ℝ := 16.4
  let h : ℝ := 32.8
  (1 / 3 * Real.pi * r^2 * h) / (1 / 3 * Real.pi * h^2 * r) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_cone_C_D_volume_ratio_l693_69380


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l693_69341

/-- The area of a rectangle with perimeter 20 meters and one side length x meters --/
def rectangle_area (x : ℝ) : ℝ := x * (10 - x)

/-- Theorem: The area of a rectangle with perimeter 20 meters and one side length x meters is x(10 - x) square meters --/
theorem rectangle_area_theorem (x : ℝ) (h : x > 0 ∧ x < 10) : 
  rectangle_area x = x * (10 - x) ∧ 
  2 * (x + (10 - x)) = 20 := by
  sorry

#check rectangle_area_theorem

end NUMINAMATH_CALUDE_rectangle_area_theorem_l693_69341


namespace NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l693_69399

-- Define sequence a_n
def a : ℕ → ℕ
| n => if n % 2 = 1 then 4 * ((n + 1) / 2) - 2 else 4 * (n / 2) - 1

-- Define sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then 8 * ((n - 1) / 2) + 3 else 8 * (n / 2) - 2

-- Define the theorem
theorem odd_square_sum_of_consecutive_b (k : ℕ) (h : k ≥ 1) :
  ∃ n : ℕ, (2 * k + 1)^2 = b n + b (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_sum_of_consecutive_b_l693_69399


namespace NUMINAMATH_CALUDE_parabola_c_value_l693_69378

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) 
    (point_condition : p.x_coord 0 = 1)
    (vertex_condition : p.x_coord 2 = 3 ∧ (∀ y, p.x_coord y ≤ p.x_coord 2)) :
  p.c = 1 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l693_69378


namespace NUMINAMATH_CALUDE_elena_book_purchase_l693_69326

theorem elena_book_purchase (total_money : ℝ) (total_books : ℕ) (book_price : ℝ) 
  (h1 : book_price > 0) 
  (h2 : total_books > 0)
  (h3 : total_money / 3 = (total_books / 2 : ℝ) * book_price) : 
  total_money - total_books * book_price = total_money / 3 := by
sorry

end NUMINAMATH_CALUDE_elena_book_purchase_l693_69326


namespace NUMINAMATH_CALUDE_inequality_proof_l693_69392

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a / b + b / c + c / a)^2 ≥ 3 * (a / c + c / b + b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l693_69392


namespace NUMINAMATH_CALUDE_sale_price_calculation_l693_69369

/-- Calculates the sale price including tax given the cost price, profit rate, and tax rate -/
def salePriceWithTax (costPrice : ℝ) (profitRate : ℝ) (taxRate : ℝ) : ℝ :=
  costPrice * (1 + profitRate) * (1 + taxRate)

/-- The sale price including tax is 677.60 given the specified conditions -/
theorem sale_price_calculation :
  let costPrice : ℝ := 535.65
  let profitRate : ℝ := 0.15
  let taxRate : ℝ := 0.10
  ∃ ε > 0, |salePriceWithTax costPrice profitRate taxRate - 677.60| < ε :=
by
  sorry

#eval salePriceWithTax 535.65 0.15 0.10

end NUMINAMATH_CALUDE_sale_price_calculation_l693_69369


namespace NUMINAMATH_CALUDE_angle_quadrant_from_point_l693_69384

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def angle_in_fourth_quadrant (α : ℝ) : Prop := 
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem angle_quadrant_from_point (α : ℝ) :
  point_in_third_quadrant (Real.sin α) (Real.tan α) →
  angle_in_fourth_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_angle_quadrant_from_point_l693_69384


namespace NUMINAMATH_CALUDE_parabola_directrix_a_value_l693_69354

/-- A parabola with equation y² = ax and directrix x = 1 has a = -4 -/
theorem parabola_directrix_a_value :
  ∀ (a : ℝ),
  (∀ (x y : ℝ), y^2 = a*x → (∃ (p : ℝ), x = -p ∧ x = 1)) →
  a = -4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_a_value_l693_69354


namespace NUMINAMATH_CALUDE_triangle_theorem_l693_69387

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (abc : Triangle) (h1 : abc.b / abc.a = Real.sin abc.B / Real.sin (2 * abc.A))
  (h2 : abc.b = 2 * Real.sqrt 3) (h3 : 1/2 * abc.b * abc.c * Real.sin abc.A = 3 * Real.sqrt 3 / 2) :
  abc.A = π/3 ∧ abc.a = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l693_69387


namespace NUMINAMATH_CALUDE_parabola_properties_l693_69375

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about the slope of line AB and the equation of the parabola -/
theorem parabola_properties (para : Parabola) (F H A B : Point) :
  -- Conditions
  (A.y^2 = 2 * para.p * A.x) →  -- A is on the parabola
  (B.y^2 = 2 * para.p * B.x) →  -- B is on the parabola
  (H.x = -para.p/2 ∧ H.y = 0) →  -- H is on the x-axis at (-p/2, 0)
  (F.x = para.p/2 ∧ F.y = 0) →  -- F is the focus at (p/2, 0)
  ((B.x - F.x)^2 + (B.y - F.y)^2 = 4 * ((A.x - F.x)^2 + (A.y - F.y)^2)) →  -- |BF| = 2|AF|
  -- Conclusions
  let slope := (B.y - A.y) / (B.x - A.x)
  (slope = 2*Real.sqrt 2/3 ∨ slope = -2*Real.sqrt 2/3) ∧
  (((B.x - A.x) * (B.y + A.y) / 2 = Real.sqrt 2) → para.p = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l693_69375


namespace NUMINAMATH_CALUDE_f_intersects_y_axis_at_zero_one_l693_69329

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Theorem statement
theorem f_intersects_y_axis_at_zero_one : f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_y_axis_at_zero_one_l693_69329


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l693_69342

theorem sqrt_difference_equals_seven : Real.sqrt (36 + 64) - Real.sqrt (25 - 16) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_l693_69342


namespace NUMINAMATH_CALUDE_factorial_sum_equation_l693_69344

theorem factorial_sum_equation (x y : ℕ) (z : ℤ) 
  (h_odd : ∃ k : ℤ, z = 2 * k + 1)
  (h_eq : x.factorial + y.factorial = 48 * z + 2017) :
  ((x = 1 ∧ y = 6 ∧ z = -27) ∨
   (x = 6 ∧ y = 1 ∧ z = -27) ∨
   (x = 1 ∧ y = 7 ∧ z = 63) ∨
   (x = 7 ∧ y = 1 ∧ z = 63)) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_equation_l693_69344


namespace NUMINAMATH_CALUDE_enough_money_for_jump_ropes_l693_69355

/-- The cost of a single jump rope in yuan -/
def jump_rope_cost : ℕ := 8

/-- The number of jump ropes to be purchased -/
def num_jump_ropes : ℕ := 31

/-- The amount of money available in yuan -/
def available_money : ℕ := 250

/-- Theorem stating that the class has enough money to buy the jump ropes -/
theorem enough_money_for_jump_ropes :
  jump_rope_cost * num_jump_ropes ≤ available_money := by
  sorry

end NUMINAMATH_CALUDE_enough_money_for_jump_ropes_l693_69355


namespace NUMINAMATH_CALUDE_steelyard_scale_construction_l693_69317

/-- Represents a steelyard (balance) --/
structure Steelyard where
  l : ℝ  -- length of the steelyard
  Q : ℝ  -- weight of the steelyard
  a : ℝ  -- distance where 1 kg balances the steelyard

/-- Theorem for the steelyard scale construction --/
theorem steelyard_scale_construction (S : Steelyard) (p x : ℝ) 
  (h1 : S.l > 0)
  (h2 : S.Q > 0)
  (h3 : S.a > 0)
  (h4 : S.a < S.l)
  (h5 : x > 0)
  (h6 : x < S.l) :
  p * x / S.a = (S.l - x) / (S.l - S.a) :=
sorry

end NUMINAMATH_CALUDE_steelyard_scale_construction_l693_69317


namespace NUMINAMATH_CALUDE_min_unique_integers_l693_69362

theorem min_unique_integers (L : List ℕ) (h : L = [1, 2, 3, 4, 5, 6, 7, 8, 9]) : 
  ∃ (f : ℕ → ℕ), 
    (∀ n, f n = n + 2 ∨ f n = n + 5) ∧ 
    (Finset.card (Finset.image f L.toFinset) = 6) ∧
    (∀ g : ℕ → ℕ, (∀ n, g n = n + 2 ∨ g n = n + 5) → 
      Finset.card (Finset.image g L.toFinset) ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_min_unique_integers_l693_69362


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l693_69376

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.38 →
  p_white = 0.34 →
  p_red + p_white + p_black = 1 →
  p_black = 0.28 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l693_69376


namespace NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_fraction_l693_69328

theorem sum_real_imag_parts_of_complex_fraction : 
  let z : ℂ := (1 + 3*I) / (1 - I)
  (z.re + z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_real_imag_parts_of_complex_fraction_l693_69328


namespace NUMINAMATH_CALUDE_states_fraction_l693_69334

/-- Given 30 total states and 15 states joining during a specific decade,
    prove that the fraction of states joining in that decade is 1/2. -/
theorem states_fraction (total_states : ℕ) (decade_states : ℕ) 
    (h1 : total_states = 30) 
    (h2 : decade_states = 15) : 
    (decade_states : ℚ) / total_states = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_l693_69334


namespace NUMINAMATH_CALUDE_point_conditions_imply_m_value_l693_69315

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the point P based on the parameter m -/
def P (m : ℝ) : Point :=
  { x := 3 - m, y := 2 * m + 6 }

/-- Condition: P is in the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Condition: P is equidistant from the coordinate axes -/
def equidistant_from_axes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Theorem: If P(3-m, 2m+6) is in the fourth quadrant and equidistant from the axes, then m = -9 -/
theorem point_conditions_imply_m_value :
  ∀ m : ℝ, in_fourth_quadrant (P m) ∧ equidistant_from_axes (P m) → m = -9 :=
by sorry

end NUMINAMATH_CALUDE_point_conditions_imply_m_value_l693_69315


namespace NUMINAMATH_CALUDE_sum_of_variables_l693_69313

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : b + c = 12 - 3*a)
  (eq2 : a + c = -14 - 3*b)
  (eq3 : a + b = 7 - 3*c) :
  2*a + 2*b + 2*c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l693_69313


namespace NUMINAMATH_CALUDE_ani_winning_strategy_l693_69336

/-- Represents the state of the game with three buckets -/
structure GameState :=
  (bucket1 bucket2 bucket3 : ℕ)

/-- Defines a valid game state where each bucket has at least one marble -/
def ValidGameState (state : GameState) : Prop :=
  state.bucket1 > 0 ∧ state.bucket2 > 0 ∧ state.bucket3 > 0

/-- Defines the total number of marbles in the game -/
def TotalMarbles (state : GameState) : ℕ :=
  state.bucket1 + state.bucket2 + state.bucket3

/-- Defines a valid move in the game -/
def ValidMove (marbles : ℕ) : Prop :=
  marbles = 1 ∨ marbles = 2 ∨ marbles = 3

/-- Defines whether a game state is a winning position for the current player -/
def IsWinningPosition (state : GameState) : Prop :=
  sorry

/-- Theorem: Ani has a winning strategy if and only if n is even and n ≥ 6 -/
theorem ani_winning_strategy (n : ℕ) :
  (∃ (initialState : GameState),
    ValidGameState initialState ∧
    TotalMarbles initialState = n ∧
    IsWinningPosition initialState) ↔
  (Even n ∧ n ≥ 6) :=
sorry

end NUMINAMATH_CALUDE_ani_winning_strategy_l693_69336


namespace NUMINAMATH_CALUDE_problem_solution_l693_69386

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1)
  (h2 : x + 1 / z = 8)
  (h3 : y + 1 / x = 29) :
  z + 1 / y = 13 / 77 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l693_69386


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_l693_69322

theorem x_squared_plus_inverse (x : ℝ) (h : 47 = x^6 + 1 / x^6) : x^2 + 1 / x^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_l693_69322


namespace NUMINAMATH_CALUDE_inequality_always_holds_l693_69394

theorem inequality_always_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) < 1 / (a^2 * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l693_69394
