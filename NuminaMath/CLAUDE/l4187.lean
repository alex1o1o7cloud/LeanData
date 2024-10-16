import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_remainder_l4187_418714

theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 20 = 80) (h2 : Q 100 = 20) :
  ∃ R : ℝ → ℝ, ∀ x, Q x = (x - 20) * (x - 100) * R x + (-3/4 * x + 95) := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l4187_418714


namespace NUMINAMATH_CALUDE_exists_selling_price_with_50_percent_profit_l4187_418782

/-- Represents the pricing model for a printer -/
structure PricingModel where
  baseSellPrice : ℝ
  baseProfit : ℝ
  taxRate1 : ℝ
  taxRate2 : ℝ
  taxThreshold1 : ℝ
  taxThreshold2 : ℝ
  discountRate : ℝ
  discountIncrement : ℝ

/-- Calculates the selling price that yields the target profit percentage -/
def findSellingPrice (model : PricingModel) (targetProfit : ℝ) : ℝ :=
  sorry

/-- Theorem: There exists a selling price that yields a 50% profit on the cost of the printer -/
theorem exists_selling_price_with_50_percent_profit (model : PricingModel) :
  ∃ (sellPrice : ℝ), findSellingPrice model 0.5 = sellPrice :=
by
  sorry

end NUMINAMATH_CALUDE_exists_selling_price_with_50_percent_profit_l4187_418782


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l4187_418723

theorem sum_of_roots_equals_seven : 
  ∀ (x y : ℝ), x^2 - 7*x + 12 = 0 ∧ y^2 - 7*y + 12 = 0 ∧ x ≠ y → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_seven_l4187_418723


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_is_false_l4187_418741

theorem range_of_a_when_proposition_is_false :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_is_false_l4187_418741


namespace NUMINAMATH_CALUDE_boxes_in_smallest_cube_l4187_418799

def box_width : ℕ := 8
def box_length : ℕ := 12
def box_height : ℕ := 30

def smallest_cube_side : ℕ := lcm (lcm box_width box_length) box_height

def box_volume : ℕ := box_width * box_length * box_height
def cube_volume : ℕ := smallest_cube_side ^ 3

theorem boxes_in_smallest_cube :
  cube_volume / box_volume = 600 := by sorry

end NUMINAMATH_CALUDE_boxes_in_smallest_cube_l4187_418799


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l4187_418756

theorem x_squared_plus_reciprocal (x : ℝ) (h : 59 = x^4 + 1/x^4) : x^2 + 1/x^2 = Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l4187_418756


namespace NUMINAMATH_CALUDE_flour_weight_qualified_l4187_418790

def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

theorem flour_weight_qualified :
  is_qualified 24.80 := by sorry

end NUMINAMATH_CALUDE_flour_weight_qualified_l4187_418790


namespace NUMINAMATH_CALUDE_solution_exists_l4187_418720

theorem solution_exists (x : ℝ) (h1 : x > 0) (h2 : x * 3^x = 3^18) :
  ∃ k : ℕ, k = 15 ∧ k < x ∧ x < k + 1 := by
sorry

end NUMINAMATH_CALUDE_solution_exists_l4187_418720


namespace NUMINAMATH_CALUDE_cauchy_schwarz_2d_l4187_418773

theorem cauchy_schwarz_2d (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_2d_l4187_418773


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4187_418775

theorem imaginary_part_of_complex_fraction : Complex.im ((1 - Complex.I) / (1 + Complex.I) + 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l4187_418775


namespace NUMINAMATH_CALUDE_sin_two_theta_plus_pi_sixth_l4187_418719

theorem sin_two_theta_plus_pi_sixth (θ : Real) 
  (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ) : 
  Real.sin (2 * θ + π / 6) = 97 / 98 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_theta_plus_pi_sixth_l4187_418719


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4187_418769

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property :
  ∀ a : ℕ → ℝ,
  (is_geometric_sequence a → ∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧
  (∃ a : ℕ → ℝ, (∀ n : ℕ, a n ^ 2 = a (n - 1) * a (n + 1)) ∧ ¬is_geometric_sequence a) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4187_418769


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4187_418732

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 - (k + 1) * x - 6 = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - (k + 1) * y - 6 = 0 ∧ y = -3 ∧ k = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4187_418732


namespace NUMINAMATH_CALUDE_farm_corn_cobs_l4187_418725

theorem farm_corn_cobs (field1_rows field1_cobs_per_row : ℕ)
                       (field2_rows field2_cobs_per_row : ℕ)
                       (field3_rows field3_cobs_per_row : ℕ)
                       (field4_rows field4_cobs_per_row : ℕ)
                       (h1 : field1_rows = 13 ∧ field1_cobs_per_row = 8)
                       (h2 : field2_rows = 16 ∧ field2_cobs_per_row = 12)
                       (h3 : field3_rows = 9 ∧ field3_cobs_per_row = 10)
                       (h4 : field4_rows = 20 ∧ field4_cobs_per_row = 6) :
  field1_rows * field1_cobs_per_row +
  field2_rows * field2_cobs_per_row +
  field3_rows * field3_cobs_per_row +
  field4_rows * field4_cobs_per_row = 506 := by
  sorry

end NUMINAMATH_CALUDE_farm_corn_cobs_l4187_418725


namespace NUMINAMATH_CALUDE_definite_integral_problem_l4187_418726

open Real MeasureTheory Interval Set

theorem definite_integral_problem :
  ∫ x in Icc 0 π, (2 * x^2 + 4 * x + 7) * cos (2 * x) = π := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l4187_418726


namespace NUMINAMATH_CALUDE_paul_collected_24_l4187_418729

/-- Represents the number of seashells collected by each person -/
structure Seashells where
  henry : ℕ
  paul : ℕ
  leo : ℕ

/-- The initial state of seashell collection -/
def initial_collection : Seashells → Prop
  | s => s.henry = 11 ∧ s.henry + s.paul + s.leo = 59

/-- The state after Leo gave away a quarter of his seashells -/
def after_leo_gives : Seashells → Prop
  | s => s.henry + s.paul + (s.leo - s.leo / 4) = 53

/-- Theorem stating that Paul collected 24 seashells -/
theorem paul_collected_24 (s : Seashells) :
  initial_collection s → after_leo_gives s → s.paul = 24 := by
  sorry

#check paul_collected_24

end NUMINAMATH_CALUDE_paul_collected_24_l4187_418729


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l4187_418724

theorem inverse_proportion_percentage_change (x y a b : ℝ) (k : ℝ) : 
  x > 0 → y > 0 → 
  (x * y = k) → 
  ((1 + a / 100) * x) * ((1 - b / 100) * y) = k → 
  b = |100 * a / (100 + a)| := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l4187_418724


namespace NUMINAMATH_CALUDE_function_zeros_inequality_l4187_418707

open Real

theorem function_zeros_inequality (a b c : ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1 → b > 0 →
  let f := fun x => a * exp x - b * x - c
  f x₁ = 0 → f x₂ = 0 → x₁ > x₂ →
  exp x₁ / a + exp x₂ / (1 - a) > 4 * b / a :=
by sorry

end NUMINAMATH_CALUDE_function_zeros_inequality_l4187_418707


namespace NUMINAMATH_CALUDE_cube_space_division_l4187_418772

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a cube and its face planes divide space into -/
def space_division (c : Cube) : Nat :=
  sorry

/-- Theorem stating that a cube and its face planes divide space into 27 parts -/
theorem cube_space_division (c : Cube) : space_division c = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_space_division_l4187_418772


namespace NUMINAMATH_CALUDE_f_plus_3_abs_l4187_418736

noncomputable def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x ≤ 0 then -2 - x
  else if 0 < x ∧ x ≤ 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if 2 < x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- undefined for other x values

theorem f_plus_3_abs (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 3) : 
  |f x + 3| = f x + 3 :=
by sorry

end NUMINAMATH_CALUDE_f_plus_3_abs_l4187_418736


namespace NUMINAMATH_CALUDE_green_blue_difference_l4187_418777

/-- Represents the number of beads of each color in Sue's necklace -/
structure BeadCount where
  purple : Nat
  blue : Nat
  green : Nat

/-- The conditions of Sue's necklace -/
def sueNecklace : BeadCount where
  purple := 7
  blue := 2 * 7
  green := 46 - (7 + 2 * 7)

theorem green_blue_difference :
  sueNecklace.green - sueNecklace.blue = 11 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l4187_418777


namespace NUMINAMATH_CALUDE_garland_arrangement_l4187_418709

theorem garland_arrangement (blue : Nat) (red : Nat) (white : Nat) :
  blue = 8 →
  red = 7 →
  white = 12 →
  (Nat.choose (blue + red) blue) * (Nat.choose (blue + red + 1) white) = 11711700 :=
by sorry

end NUMINAMATH_CALUDE_garland_arrangement_l4187_418709


namespace NUMINAMATH_CALUDE_ratio_equality_l4187_418727

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc_sq : a^2 + b^2 + c^2 = 1)
  (sum_xyz_sq : x^2 + y^2 + z^2 = 4)
  (sum_prod : a*x + b*y + c*z = 2) :
  (a + b + c) / (x + y + z) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l4187_418727


namespace NUMINAMATH_CALUDE_triangle_area_sine_relation_l4187_418776

theorem triangle_area_sine_relation (a b c : ℝ) (A : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (a^2 - b^2 - c^2 + 2*b*c = (1/2) * b * c * Real.sin A) →
  Real.sin A = 8/17 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_sine_relation_l4187_418776


namespace NUMINAMATH_CALUDE_superadditive_continuous_function_is_linear_l4187_418731

/-- A function satisfying the given conditions -/
def SuperadditiveContinuousFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ f 0 = 0 ∧ ∀ x y : ℝ, f (x + y) ≥ f x + f y

/-- The main theorem -/
theorem superadditive_continuous_function_is_linear
    (f : ℝ → ℝ) (hf : SuperadditiveContinuousFunction f) :
    ∃ a : ℝ, ∀ x : ℝ, f x = a * x := by
  sorry

end NUMINAMATH_CALUDE_superadditive_continuous_function_is_linear_l4187_418731


namespace NUMINAMATH_CALUDE_first_number_in_set_l4187_418746

theorem first_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [x, 40, 60]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_set_l4187_418746


namespace NUMINAMATH_CALUDE_min_sum_positive_reals_min_sum_positive_reals_tight_l4187_418796

theorem min_sum_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ (1 / 2 : ℝ) :=
by sorry

theorem min_sum_positive_reals_tight (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (3 * b) + b / (6 * c) + c / (9 * a) < (1 / 2 : ℝ) + ε :=
by sorry

end NUMINAMATH_CALUDE_min_sum_positive_reals_min_sum_positive_reals_tight_l4187_418796


namespace NUMINAMATH_CALUDE_membership_ratio_is_three_to_one_l4187_418716

/-- Represents the monthly costs and sign-up fees for two gym memberships --/
structure GymMemberships where
  cheap_monthly : ℚ
  cheap_signup : ℚ
  expensive_signup_months : ℚ
  total_first_year : ℚ

/-- Calculates the ratio of expensive gym's monthly cost to cheap gym's monthly cost --/
def membership_ratio (g : GymMemberships) : ℚ :=
  let cheap_yearly := g.cheap_signup + 12 * g.cheap_monthly
  let expensive_yearly := g.total_first_year - cheap_yearly
  let expensive_monthly := expensive_yearly / (g.expensive_signup_months + 12)
  expensive_monthly / g.cheap_monthly

/-- Theorem stating that the membership ratio is 3:1 for the given conditions --/
theorem membership_ratio_is_three_to_one (g : GymMemberships)
    (h1 : g.cheap_monthly = 10)
    (h2 : g.cheap_signup = 50)
    (h3 : g.expensive_signup_months = 4)
    (h4 : g.total_first_year = 650) :
    membership_ratio g = 3 := by
  sorry

end NUMINAMATH_CALUDE_membership_ratio_is_three_to_one_l4187_418716


namespace NUMINAMATH_CALUDE_factorization_identities_l4187_418728

theorem factorization_identities (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a^4 - b^4 = (a - b) * (a + b) * (a^2 + b^2)) ∧
  (a + b - 2 * Real.sqrt (a * b) = (Real.sqrt a - Real.sqrt b)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l4187_418728


namespace NUMINAMATH_CALUDE_same_side_of_line_l4187_418780

/-- 
Given a line 2x - y + 1 = 0 and two points (1, 2) and (1, 0),
prove that these points are on the same side of the line.
-/
theorem same_side_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 1 ∧ y₁ = 2 ∧ x₂ = 1 ∧ y₂ = 0 →
  (2 * x₁ - y₁ + 1 > 0) ∧ (2 * x₂ - y₂ + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_same_side_of_line_l4187_418780


namespace NUMINAMATH_CALUDE_dispatch_methods_count_l4187_418763

def num_male_servants : ℕ := 5
def num_female_servants : ℕ := 4
def num_total_servants : ℕ := num_male_servants + num_female_servants
def num_selected : ℕ := 3
def num_areas : ℕ := 3

theorem dispatch_methods_count :
  (Nat.choose num_total_servants num_selected - 
   Nat.choose num_male_servants num_selected - 
   Nat.choose num_female_servants num_selected) * 
  (Nat.factorial num_selected) = 420 := by
  sorry

end NUMINAMATH_CALUDE_dispatch_methods_count_l4187_418763


namespace NUMINAMATH_CALUDE_inequality_range_l4187_418739

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x + 4 / x ≥ a) ↔ a ∈ Set.Iic 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l4187_418739


namespace NUMINAMATH_CALUDE_determinant_of_trigonometric_matrix_l4187_418793

theorem determinant_of_trigonometric_matrix (α β γ : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.cos α * Real.cos β, Real.cos α * Real.sin β, -Real.sin (α + γ)],
    ![-Real.sin β, Real.cos β * Real.cos γ, Real.sin β * Real.sin γ],
    ![Real.sin (α + γ) * Real.cos β, Real.sin (α + γ) * Real.sin β, Real.cos (α + γ)]
  ]
  Matrix.det M = 1 := by
sorry

end NUMINAMATH_CALUDE_determinant_of_trigonometric_matrix_l4187_418793


namespace NUMINAMATH_CALUDE_num_factors_of_given_number_l4187_418768

/-- The number of distinct, natural-number factors of 4³ * 5⁴ * 6² -/
def num_factors : ℕ := 135

/-- The given number -/
def given_number : ℕ := 4^3 * 5^4 * 6^2

theorem num_factors_of_given_number :
  (Finset.filter (· ∣ given_number) (Finset.range (given_number + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_num_factors_of_given_number_l4187_418768


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l4187_418787

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = 2 - 3*I ∨ x = 2 + 3*I) ∧
    (a = -4 ∧ b = 13) := by
  sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l4187_418787


namespace NUMINAMATH_CALUDE_students_in_one_subject_is_32_l4187_418795

/-- Represents the number of students in each class and their intersections -/
structure ClassEnrollment where
  calligraphy : ℕ
  art : ℕ
  instrumental : ℕ
  calligraphy_art : ℕ
  calligraphy_instrumental : ℕ
  art_instrumental : ℕ
  all_three : ℕ

/-- Calculates the number of students enrolled in only one subject -/
def studentsInOneSubject (e : ClassEnrollment) : ℕ :=
  e.calligraphy + e.art + e.instrumental - 2 * (e.calligraphy_art + e.calligraphy_instrumental + e.art_instrumental) + 3 * e.all_three

/-- The main theorem stating that given the enrollment conditions, 32 students are in only one subject -/
theorem students_in_one_subject_is_32 (e : ClassEnrollment)
  (h1 : e.calligraphy = 29)
  (h2 : e.art = 28)
  (h3 : e.instrumental = 27)
  (h4 : e.calligraphy_art = 13)
  (h5 : e.calligraphy_instrumental = 12)
  (h6 : e.art_instrumental = 11)
  (h7 : e.all_three = 5) :
  studentsInOneSubject e = 32 := by
  sorry


end NUMINAMATH_CALUDE_students_in_one_subject_is_32_l4187_418795


namespace NUMINAMATH_CALUDE_fraction_comparison_l4187_418785

theorem fraction_comparison : (200200201 : ℚ) / 200200203 > (300300301 : ℚ) / 300300304 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l4187_418785


namespace NUMINAMATH_CALUDE_expected_successful_trials_value_l4187_418759

/-- A trial is successful if at least one of two dice shows a 4 or a 5 -/
def is_successful_trial (dice1 dice2 : Nat) : Bool :=
  dice1 = 4 ∨ dice1 = 5 ∨ dice2 = 4 ∨ dice2 = 5

/-- The probability of a successful trial -/
def prob_success : ℚ := 5 / 9

/-- The number of trials -/
def num_trials : ℕ := 10

/-- The expected number of successful trials -/
def expected_successful_trials : ℚ := num_trials * prob_success

theorem expected_successful_trials_value :
  expected_successful_trials = 50 / 9 := by sorry

end NUMINAMATH_CALUDE_expected_successful_trials_value_l4187_418759


namespace NUMINAMATH_CALUDE_josiah_saved_24_days_l4187_418765

/-- The number of days Josiah saved -/
def josiah_days : ℕ := sorry

/-- Josiah's daily savings in dollars -/
def josiah_daily_savings : ℚ := 1/4

/-- Leah's daily savings in dollars -/
def leah_daily_savings : ℚ := 1/2

/-- Number of days Leah saved -/
def leah_days : ℕ := 20

/-- Number of days Megan saved -/
def megan_days : ℕ := 12

/-- Total amount saved by all three children in dollars -/
def total_savings : ℚ := 28

theorem josiah_saved_24_days :
  josiah_days = 24 ∧
  josiah_daily_savings * josiah_days + 
  leah_daily_savings * leah_days + 
  (2 * leah_daily_savings) * megan_days = total_savings := by sorry

end NUMINAMATH_CALUDE_josiah_saved_24_days_l4187_418765


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l4187_418761

theorem polar_to_rectangular_conversion :
  ∀ (x y ρ θ : ℝ),
  ρ = Real.sin θ + Real.cos θ →
  x = ρ * Real.cos θ →
  y = ρ * Real.sin θ →
  ρ^2 = x^2 + y^2 →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l4187_418761


namespace NUMINAMATH_CALUDE_percentage_relationship_l4187_418752

theorem percentage_relationship (A B T : ℝ) 
  (h1 : B = 0.14 * T) 
  (h2 : A = 0.5 * B) : 
  A = 0.07 * T := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l4187_418752


namespace NUMINAMATH_CALUDE_factorization1_factorization2_factorization3_l4187_418708

-- Given formulas
axiom formula1 (x a b : ℝ) : (x + a) * (x + b) = x^2 + (a + b) * x + a * b
axiom formula2 (x y : ℝ) : (x + y)^2 + 2 * (x + y) + 1 = (x + y + 1)^2

-- Theorems to prove
theorem factorization1 (x : ℝ) : x^2 + 4 * x + 3 = (x + 3) * (x + 1) := by sorry

theorem factorization2 (x y : ℝ) : (x - y)^2 - 10 * (x - y) + 25 = (x - y - 5)^2 := by sorry

theorem factorization3 (m : ℝ) : (m^2 - 2 * m) * (m^2 - 2 * m + 4) + 3 = (m^2 - 2 * m + 3) * (m - 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization1_factorization2_factorization3_l4187_418708


namespace NUMINAMATH_CALUDE_a_range_l4187_418757

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 ≤ 0

-- Define the theorem
theorem a_range (a : ℝ) : 
  (a > 0) →
  (∀ x, ¬(p x) → ¬(q x a)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_a_range_l4187_418757


namespace NUMINAMATH_CALUDE_interest_rate_proof_l4187_418743

/-- Proves that given specific conditions, the interest rate is 18% --/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (interest_difference : ℝ) : 
  principal = 4000 →
  time = 2 →
  interest_difference = 480 →
  (principal * time * (18 / 100)) = (principal * time * (12 / 100) + interest_difference) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l4187_418743


namespace NUMINAMATH_CALUDE_congruence_solutions_count_l4187_418758

theorem congruence_solutions_count :
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, x > 0 ∧ x < 150 ∧ (x + 15) % 45 = 75 % 45) ∧ 
    (∀ x, x > 0 ∧ x < 150 ∧ (x + 15) % 45 = 75 % 45 → x ∈ s) ∧
    s.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_congruence_solutions_count_l4187_418758


namespace NUMINAMATH_CALUDE_pudding_cost_pudding_cost_is_two_l4187_418711

/-- The cost of each cup of pudding, given the conditions of Jane's purchase -/
theorem pudding_cost (num_ice_cream : ℕ) (num_pudding : ℕ) (ice_cream_price : ℕ) (extra_spent : ℕ) : ℕ :=
  let total_ice_cream := num_ice_cream * ice_cream_price
  let pudding_cost := (total_ice_cream - extra_spent) / num_pudding
  pudding_cost

/-- Proof that each cup of pudding costs $2 -/
theorem pudding_cost_is_two :
  pudding_cost 15 5 5 65 = 2 := by
  sorry

end NUMINAMATH_CALUDE_pudding_cost_pudding_cost_is_two_l4187_418711


namespace NUMINAMATH_CALUDE_square_root_equality_l4187_418705

theorem square_root_equality (x a : ℝ) (hx : x > 0) : 
  Real.sqrt x = 2 * a - 3 ∧ Real.sqrt x = 5 - a → a = 8/3 ∧ x = 49/9 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l4187_418705


namespace NUMINAMATH_CALUDE_math_team_combinations_l4187_418702

theorem math_team_combinations : ℕ := by
  -- Define the total number of girls and boys in the math club
  let total_girls : ℕ := 5
  let total_boys : ℕ := 5
  
  -- Define the number of girls and boys needed for the team
  let team_girls : ℕ := 3
  let team_boys : ℕ := 3
  
  -- Define the total team size
  let team_size : ℕ := team_girls + team_boys
  
  -- Calculate the number of ways to choose the team
  let result := (total_girls.choose team_girls) * (total_boys.choose team_boys)
  
  -- Prove that the result is equal to 100
  have h : result = 100 := by sorry
  
  -- Return the result
  exact result

end NUMINAMATH_CALUDE_math_team_combinations_l4187_418702


namespace NUMINAMATH_CALUDE_average_decrease_l4187_418710

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 → 
  old_avg = 14 → 
  new_obs = 7 → 
  old_avg - (n * old_avg + new_obs) / (n + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_average_decrease_l4187_418710


namespace NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l4187_418783

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l4187_418783


namespace NUMINAMATH_CALUDE_min_tenth_game_score_l4187_418753

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  first_five : ℝ  -- Total score of first 5 games
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ
  ninth : ℝ
  tenth : ℝ

/-- Theorem stating the minimum score required for the 10th game -/
theorem min_tenth_game_score (scores : BasketballScores) 
  (h1 : scores.sixth = 23)
  (h2 : scores.seventh = 14)
  (h3 : scores.eighth = 11)
  (h4 : scores.ninth = 20)
  (h5 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth) / 9 > 
        scores.first_five / 5)
  (h6 : (scores.first_five + scores.sixth + scores.seventh + scores.eighth + scores.ninth + scores.tenth) / 10 > 18) :
  scores.tenth ≥ 29 := by
  sorry

end NUMINAMATH_CALUDE_min_tenth_game_score_l4187_418753


namespace NUMINAMATH_CALUDE_candy_sampling_problem_l4187_418766

theorem candy_sampling_problem (caught_percentage : ℝ) (total_sampling_percentage : ℝ) 
  (h1 : caught_percentage = 22)
  (h2 : total_sampling_percentage = 25) :
  total_sampling_percentage - caught_percentage = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_sampling_problem_l4187_418766


namespace NUMINAMATH_CALUDE_triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l4187_418748

def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

theorem triangular_number_gcd_bound (n : ℕ+) : 
  Nat.gcd (6 * triangular_number n) (n + 1) ≤ 3 :=
sorry

theorem triangular_number_gcd_achieves_three : 
  ∃ n : ℕ+, Nat.gcd (6 * triangular_number n) (n + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_triangular_number_gcd_bound_triangular_number_gcd_achieves_three_l4187_418748


namespace NUMINAMATH_CALUDE_women_to_men_ratio_l4187_418738

/-- Given an event with guests, prove the ratio of women to men --/
theorem women_to_men_ratio 
  (total_guests : ℕ) 
  (num_men : ℕ) 
  (num_children_after : ℕ) 
  (h1 : total_guests = 80) 
  (h2 : num_men = 40) 
  (h3 : num_children_after = 30) :
  (total_guests - num_men - (num_children_after - 10)) / num_men = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_women_to_men_ratio_l4187_418738


namespace NUMINAMATH_CALUDE_center_sum_l4187_418712

theorem center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 + 4*X - 6*Y = 3 ↔ (X - x)^2 + (Y - y)^2 = 16) → 
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_center_sum_l4187_418712


namespace NUMINAMATH_CALUDE_total_flowers_and_sticks_l4187_418786

/-- The number of pots -/
def num_pots : ℕ := 466

/-- The number of flowers in each pot -/
def flowers_per_pot : ℕ := 53

/-- The number of sticks in each pot -/
def sticks_per_pot : ℕ := 181

/-- The total number of flowers and sticks in all pots -/
def total_items : ℕ := num_pots * flowers_per_pot + num_pots * sticks_per_pot

theorem total_flowers_and_sticks : total_items = 109044 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_and_sticks_l4187_418786


namespace NUMINAMATH_CALUDE_solution_days_is_forty_l4187_418740

/-- The number of days required to solve all problems given the conditions -/
def solution_days (a b c : ℕ) : ℕ :=
  let total_problems := 5 * (11 * a + 7 * b + 9 * c)
  40

/-- The theorem stating that the solution_days function returns 40 given the problem conditions -/
theorem solution_days_is_forty (a b c : ℕ) :
  (5 * (11 * a + 7 * b + 9 * c) = 16 * (4 * a + 2 * b + 3 * c)) →
  solution_days a b c = 40 := by
  sorry

#check solution_days_is_forty

end NUMINAMATH_CALUDE_solution_days_is_forty_l4187_418740


namespace NUMINAMATH_CALUDE_mirasol_initial_balance_l4187_418747

/-- Mirasol's initial account balance -/
def initial_balance : ℕ := sorry

/-- Amount spent on coffee beans -/
def coffee_cost : ℕ := 10

/-- Amount spent on tumbler -/
def tumbler_cost : ℕ := 30

/-- Amount left in account -/
def remaining_balance : ℕ := 10

/-- Theorem: Mirasol's initial account balance was $50 -/
theorem mirasol_initial_balance :
  initial_balance = coffee_cost + tumbler_cost + remaining_balance :=
by sorry

end NUMINAMATH_CALUDE_mirasol_initial_balance_l4187_418747


namespace NUMINAMATH_CALUDE_special_tetrahedron_equal_angle_l4187_418784

/-- A tetrahedron with specific dihedral angle properties -/
structure SpecialTetrahedron where
  /-- The tetrahedron has three dihedral angles of 90° that do not belong to the same vertex -/
  three_right_angles : ℕ
  /-- All other dihedral angles are equal -/
  equal_other_angles : ℝ
  /-- The number of 90° angles is exactly 3 -/
  right_angle_count : three_right_angles = 3

/-- The theorem stating the value of the equal dihedral angles in the special tetrahedron -/
theorem special_tetrahedron_equal_angle (t : SpecialTetrahedron) :
  t.equal_other_angles = Real.arccos ((Real.sqrt 5 - 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_special_tetrahedron_equal_angle_l4187_418784


namespace NUMINAMATH_CALUDE_tangent_circle_slope_l4187_418706

/-- Circle represented by its equation in the form x² + y² + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a line y = mx contains the center of a circle tangent to two given circles -/
def has_tangent_circle (w₁ w₂ : Circle) (m : ℝ) : Prop :=
  ∃ (x y r : ℝ),
    y = m * x ∧
    (x - 4)^2 + (y - 10)^2 = (r + 3)^2 ∧
    (x + 4)^2 + (y - 10)^2 = (11 - r)^2

/-- The main theorem -/
theorem tangent_circle_slope (w₁ w₂ : Circle) :
  w₁.a = 8 ∧ w₁.b = -20 ∧ w₁.c = -75 ∧
  w₂.a = -8 ∧ w₂.b = -20 ∧ w₂.c = 125 →
  ∃ (m : ℝ),
    m > 0 ∧
    has_tangent_circle w₁ w₂ m ∧
    (∀ m' : ℝ, 0 < m' ∧ m' < m → ¬ has_tangent_circle w₁ w₂ m') ∧
    m^2 = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_slope_l4187_418706


namespace NUMINAMATH_CALUDE_hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l4187_418770

-- Define the sample space
def Ω : Type := Unit

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Define the event of missing the target both times
def miss_both_times : Set Ω := sorry

-- Theorem: hit_at_least_once and miss_both_times are mutually exclusive
theorem hit_at_least_once_miss_both_times_mutually_exclusive :
  hit_at_least_once ∩ miss_both_times = ∅ :=
sorry

-- Theorem: hit_at_least_once and miss_both_times are complementary
theorem hit_at_least_once_miss_both_times_complementary :
  hit_at_least_once ∪ miss_both_times = Set.univ :=
sorry

end NUMINAMATH_CALUDE_hit_at_least_once_miss_both_times_mutually_exclusive_hit_at_least_once_miss_both_times_complementary_l4187_418770


namespace NUMINAMATH_CALUDE_no_real_solution_l4187_418718

theorem no_real_solution :
  ¬∃ (x y : ℝ), 4 * x^2 + 9 * y^2 - 16 * x - 36 * y + 64 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l4187_418718


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l4187_418734

/-- Proves that the minimum bailing rate is 13 gallons per minute -/
theorem minimum_bailing_rate
  (distance_to_shore : ℝ)
  (water_intake_rate : ℝ)
  (boat_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance_to_shore = 3)
  (h2 : water_intake_rate = 15)
  (h3 : boat_capacity = 60)
  (h4 : rowing_speed = 6)
  : ∃ (bailing_rate : ℝ), 
    bailing_rate = 13 ∧ 
    bailing_rate * (distance_to_shore / rowing_speed * 60) ≥ 
    water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity ∧
    ∀ (r : ℝ), r < bailing_rate → 
      r * (distance_to_shore / rowing_speed * 60) < 
      water_intake_rate * (distance_to_shore / rowing_speed * 60) - boat_capacity :=
by sorry


end NUMINAMATH_CALUDE_minimum_bailing_rate_l4187_418734


namespace NUMINAMATH_CALUDE_square_root_meaningful_l4187_418798

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_l4187_418798


namespace NUMINAMATH_CALUDE_solve_equation_l4187_418781

theorem solve_equation : 
  ∃ x : ℝ, (4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470) ∧ (x = 13.26) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4187_418781


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4187_418778

/-- A geometric sequence is a sequence where each term after the first is found by multiplying 
    the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sixth_term 
  (a : ℕ → ℚ) 
  (h_geo : is_geometric_sequence a) 
  (h_sum1 : a 1 + a 3 = 5/2) 
  (h_sum2 : a 2 + a 4 = 5/4) : 
  a 6 = 1/16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l4187_418778


namespace NUMINAMATH_CALUDE_selling_price_is_correct_l4187_418717

/-- Calculates the selling price per copy of a program given the production cost,
    advertisement revenue, number of copies to be sold, and desired profit. -/
def calculate_selling_price (production_cost : ℚ) (ad_revenue : ℚ) (copies : ℕ) (desired_profit : ℚ) : ℚ :=
  (desired_profit + (production_cost * copies) - ad_revenue) / copies

theorem selling_price_is_correct : 
  let production_cost : ℚ := 70/100
  let ad_revenue : ℚ := 15000
  let copies : ℕ := 35000
  let desired_profit : ℚ := 8000
  calculate_selling_price production_cost ad_revenue copies desired_profit = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_correct_l4187_418717


namespace NUMINAMATH_CALUDE_no_solution_exists_l4187_418792

theorem no_solution_exists (a c : ℝ) : ¬∃ x : ℝ, 
  ((a + x) / 2 = 110) ∧ 
  ((x + c) / 2 = 170) ∧ 
  (a - c = 120) := by
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4187_418792


namespace NUMINAMATH_CALUDE_eleven_subtractions_to_zero_l4187_418797

def digit_sum (n : ℕ) : ℕ := sorry

def subtract_digit_sum (n : ℕ) : ℕ := n - digit_sum n

def repeat_subtract_digit_sum (n : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => repeat_subtract_digit_sum (subtract_digit_sum n) k

theorem eleven_subtractions_to_zero (n : ℕ) (h : 100 ≤ n ∧ n ≤ 109) :
  repeat_subtract_digit_sum n 11 = 0 := by sorry

end NUMINAMATH_CALUDE_eleven_subtractions_to_zero_l4187_418797


namespace NUMINAMATH_CALUDE_equation_A_is_linear_l4187_418715

/-- An equation is linear in two variables if it can be written in the form ax + by + c = 0,
    where a, b, and c are constants, and a and b are not both zero. -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y + c = 0

/-- The equation (2y-1)/5 = 2 - (3x-2)/4 -/
def equation_A (x y : ℝ) : Prop :=
  (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

theorem equation_A_is_linear :
  is_linear_equation_in_two_variables equation_A :=
sorry

end NUMINAMATH_CALUDE_equation_A_is_linear_l4187_418715


namespace NUMINAMATH_CALUDE_binary_representation_of_2015_l4187_418754

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_representation_of_2015 :
  toBinary 2015 = [true, true, true, true, true, false, true, true, true, true, true] :=
by sorry

#eval fromBinary [true, true, true, true, true, false, true, true, true, true, true]

end NUMINAMATH_CALUDE_binary_representation_of_2015_l4187_418754


namespace NUMINAMATH_CALUDE_sample_capacity_l4187_418735

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ) 
  (h1 : frequency = 30)
  (h2 : frequency_rate = 1/4)
  (h3 : n = frequency / frequency_rate) :
  n = 120 := by
sorry

end NUMINAMATH_CALUDE_sample_capacity_l4187_418735


namespace NUMINAMATH_CALUDE_product_digit_sum_base7_l4187_418701

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base-7 number --/
def sumDigitsBase7 (n : ℕ) : ℕ := sorry

/-- Multiplies two base-7 numbers --/
def multiplyBase7 (a b : ℕ) : ℕ := 
  toBase7 (toBase10 a * toBase10 b)

theorem product_digit_sum_base7 : 
  sumDigitsBase7 (multiplyBase7 35 42) = 21 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_base7_l4187_418701


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l4187_418771

theorem arithmetic_expression_equality : 5 + 2 * (8 - 3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l4187_418771


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l4187_418789

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π/4 - α) = 3/5) : 
  Real.sin (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l4187_418789


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4187_418745

/-- Given a hyperbola with equation x²/4 - y²/9 = -1, its asymptotes are y = ±(3/2)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 4 - y^2 / 9 = -1 →
  ∃ (k : ℝ), k = 3/2 ∧ (y = k*x ∨ y = -k*x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4187_418745


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l4187_418755

theorem r_value_when_n_is_3 (n m s r : ℕ) :
  m = 3 ∧ s = 2^n - m ∧ r = 3^s + s ∧ n = 3 → r = 248 := by
  sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l4187_418755


namespace NUMINAMATH_CALUDE_valid_numbers_count_l4187_418749

/-- Converts a base-10 number to base-12 --/
def toBase12 (n : ℕ) : ℕ := sorry

/-- Checks if a base-12 number uses only digits 0-9 --/
def usesOnlyDigits0to9 (n : ℕ) : Bool := sorry

/-- Counts numbers up to n (base-10) that use only digits 0-9 in base-12 --/
def countValidNumbers (n : ℕ) : ℕ := sorry

theorem valid_numbers_count :
  countValidNumbers 1200 = 90 := by sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l4187_418749


namespace NUMINAMATH_CALUDE_total_limes_picked_l4187_418750

theorem total_limes_picked (fred_limes alyssa_limes nancy_limes david_limes eileen_limes : ℕ)
  (h1 : fred_limes = 36)
  (h2 : alyssa_limes = 32)
  (h3 : nancy_limes = 35)
  (h4 : david_limes = 42)
  (h5 : eileen_limes = 50) :
  fred_limes + alyssa_limes + nancy_limes + david_limes + eileen_limes = 195 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_picked_l4187_418750


namespace NUMINAMATH_CALUDE_dracula_is_alive_l4187_418751

-- Define the propositions
variable (T : Prop) -- "The Transylvanian is human"
variable (D : Prop) -- "Count Dracula is alive"

-- Define the Transylvanian's statements
variable (statement1 : T)
variable (statement2 : T → D)

-- Define the Transylvanian's ability to reason logically
variable (logical_reasoning : T)

-- Theorem to prove
theorem dracula_is_alive : D := by
  sorry

end NUMINAMATH_CALUDE_dracula_is_alive_l4187_418751


namespace NUMINAMATH_CALUDE_workout_calculation_l4187_418733

-- Define the exercise parameters
def bicep_curls_weight : ℕ := 20
def bicep_curls_dumbbells : ℕ := 2
def bicep_curls_reps : ℕ := 10
def bicep_curls_sets : ℕ := 3

def shoulder_press_weight1 : ℕ := 30
def shoulder_press_weight2 : ℕ := 40
def shoulder_press_reps : ℕ := 8
def shoulder_press_sets : ℕ := 2

def lunges_weight : ℕ := 30
def lunges_dumbbells : ℕ := 2
def lunges_reps : ℕ := 12
def lunges_sets : ℕ := 4

def bench_press_weight : ℕ := 40
def bench_press_dumbbells : ℕ := 2
def bench_press_reps : ℕ := 6
def bench_press_sets : ℕ := 3

-- Define the theorem
theorem workout_calculation :
  -- Total weight calculation
  (bicep_curls_weight * bicep_curls_dumbbells * bicep_curls_reps * bicep_curls_sets) +
  ((shoulder_press_weight1 + shoulder_press_weight2) * shoulder_press_reps * shoulder_press_sets) +
  (lunges_weight * lunges_dumbbells * lunges_reps * lunges_sets) +
  (bench_press_weight * bench_press_dumbbells * bench_press_reps * bench_press_sets) = 6640 ∧
  -- Average weight per rep for each exercise
  bicep_curls_weight * bicep_curls_dumbbells = 40 ∧
  shoulder_press_weight1 + shoulder_press_weight2 = 70 ∧
  lunges_weight * lunges_dumbbells = 60 ∧
  bench_press_weight * bench_press_dumbbells = 80 := by
  sorry

end NUMINAMATH_CALUDE_workout_calculation_l4187_418733


namespace NUMINAMATH_CALUDE_min_value_xy_l4187_418704

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (Real.log x) * (Real.log y) = 1 / 16) : 
  x * y ≥ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l4187_418704


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4187_418742

/-- The perimeter of a semicircle with radius 9 is approximately 46.26 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 9
  let π_approx : ℝ := 3.14
  let semicircle_perimeter := r * π_approx + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 46.26) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4187_418742


namespace NUMINAMATH_CALUDE_a_values_l4187_418788

def A (a : ℝ) : Set ℝ := {0, 1, a^2 - 2*a}

theorem a_values (a : ℝ) (h : a ∈ A a) : a = 1 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l4187_418788


namespace NUMINAMATH_CALUDE_part_one_part_two_l4187_418730

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (I)
theorem part_one (a x : ℝ) (h1 : a > 0) (h2 : a = 1) (h3 : p a x ∧ q x) : 
  2 < x ∧ x < 3 := by sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x)) 
  (h3 : ∃ x, ¬(p a x) ∧ q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4187_418730


namespace NUMINAMATH_CALUDE_stock_percentage_l4187_418764

/-- Given the income, price per unit, and total investment of a stock,
    calculate the percentage of the stock. -/
theorem stock_percentage
  (income : ℝ)
  (price_per_unit : ℝ)
  (total_investment : ℝ)
  (h1 : income = 900)
  (h2 : price_per_unit = 102)
  (h3 : total_investment = 4590)
  : (income / total_investment) * 100 = (900 : ℝ) / 4590 * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_l4187_418764


namespace NUMINAMATH_CALUDE_determinant_problem_l4187_418737

theorem determinant_problem (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = -12 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l4187_418737


namespace NUMINAMATH_CALUDE_parabola_and_max_area_line_l4187_418774

-- Define the parabola
def Parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point on the parabola
def PointOnParabola (p : ℝ) (x₀ : ℝ) : Prop := Parabola p x₀ 4

-- Define the distance from a point to the focus
def DistanceToFocus (p : ℝ) (x₀ : ℝ) : Prop := x₀ + p/2 = 4

-- Define the angle bisector condition
def AngleBisectorCondition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ ≤ 0 ∧ y₂ ≤ 0 ∧ (4 - y₁)/(2 - x₁) = -(4 - y₂)/(2 - x₂)

-- Main theorem
theorem parabola_and_max_area_line
  (p : ℝ) (x₀ : ℝ)
  (h₁ : PointOnParabola p x₀)
  (h₂ : DistanceToFocus p x₀) :
  (∀ x y, Parabola p x y ↔ y^2 = 8*x) ∧
  (∃ x₁ y₁ x₂ y₂,
    AngleBisectorCondition x₁ y₁ x₂ y₂ ∧
    Parabola p x₁ y₁ ∧ Parabola p x₂ y₂ ∧
    (∀ a b, (Parabola p a b ∧ b ≤ 0) →
      (x₁ - 2)*(y₂ - 4) - (x₂ - 2)*(y₁ - 4) ≤ (x₁ - 2)*(b - 4) - (a - 2)*(y₁ - 4)) ∧
    x₁ + y₁ = 0 ∧ x₂ + y₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_max_area_line_l4187_418774


namespace NUMINAMATH_CALUDE_f_four_times_one_l4187_418779

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_four_times_one : f (f (f (f 1))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_four_times_one_l4187_418779


namespace NUMINAMATH_CALUDE_max_box_volume_l4187_418791

/-- The length of the cardboard in centimeters -/
def cardboard_length : ℝ := 30

/-- The width of the cardboard in centimeters -/
def cardboard_width : ℝ := 14

/-- The volume of the box as a function of the side length of the cut squares -/
def box_volume (x : ℝ) : ℝ := (cardboard_length - 2*x) * (cardboard_width - 2*x) * x

/-- The maximum volume of the box -/
def max_volume : ℝ := 576

theorem max_box_volume :
  ∃ x : ℝ, 0 < x ∧ x < cardboard_width / 2 ∧
  (∀ y : ℝ, 0 < y ∧ y < cardboard_width / 2 → box_volume y ≤ box_volume x) ∧
  box_volume x = max_volume :=
sorry

end NUMINAMATH_CALUDE_max_box_volume_l4187_418791


namespace NUMINAMATH_CALUDE_bags_sold_on_tuesday_l4187_418762

theorem bags_sold_on_tuesday (total_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ) 
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : wednesday_sales = 100)
  (h4 : thursday_sales = 110)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℝ) * 0.25 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) :
  tuesday_sales = 70 := by
  sorry

end NUMINAMATH_CALUDE_bags_sold_on_tuesday_l4187_418762


namespace NUMINAMATH_CALUDE_base_conversion_403_6_to_8_l4187_418721

/-- Converts a number from base 6 to base 10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 8 --/
def decimal_to_base8 (n : ℕ) : ℕ :=
  if n < 8 then n
  else (decimal_to_base8 (n / 8)) * 10 + (n % 8)

theorem base_conversion_403_6_to_8 :
  decimal_to_base8 (base6_to_decimal 403) = 223 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_403_6_to_8_l4187_418721


namespace NUMINAMATH_CALUDE_ab_value_for_given_equation_l4187_418794

theorem ab_value_for_given_equation (a b : ℕ+) 
  (h : (2 * a + b) * (2 * b + a) = 4752) : 
  a * b = 520 := by
sorry

end NUMINAMATH_CALUDE_ab_value_for_given_equation_l4187_418794


namespace NUMINAMATH_CALUDE_difference_is_2_5q_minus_15_l4187_418703

/-- The difference in dimes between two people's quarter amounts -/
def difference_in_dimes (q : ℝ) : ℝ :=
  let samantha_quarters : ℝ := 3 * q + 2
  let bob_quarters : ℝ := 2 * q + 8
  let quarter_to_dime : ℝ := 2.5
  quarter_to_dime * (samantha_quarters - bob_quarters)

/-- Theorem stating the difference in dimes -/
theorem difference_is_2_5q_minus_15 (q : ℝ) :
  difference_in_dimes q = 2.5 * q - 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_2_5q_minus_15_l4187_418703


namespace NUMINAMATH_CALUDE_cloak_purchase_change_l4187_418744

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakPurchase where
  silver_paid : ℕ
  gold_change : ℕ

/-- Proves that buying an invisibility cloak for 14 gold coins results in a change of 10 silver coins --/
theorem cloak_purchase_change 
  (purchase1 : CloakPurchase)
  (purchase2 : CloakPurchase)
  (h1 : purchase1.silver_paid = 20 ∧ purchase1.gold_change = 4)
  (h2 : purchase2.silver_paid = 15 ∧ purchase2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14)
  : ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end NUMINAMATH_CALUDE_cloak_purchase_change_l4187_418744


namespace NUMINAMATH_CALUDE_first_part_interest_rate_l4187_418713

/-- Proves that given the specified conditions, the interest rate of the first part is 3% -/
theorem first_part_interest_rate 
  (total_investment : ℝ) 
  (first_part : ℝ) 
  (second_part_rate : ℝ) 
  (total_interest : ℝ) : 
  total_investment = 4000 →
  first_part = 2800 →
  second_part_rate = 0.05 →
  total_interest = 144 →
  (first_part * (3 / 100) + (total_investment - first_part) * second_part_rate = total_interest) :=
by
  sorry

#check first_part_interest_rate

end NUMINAMATH_CALUDE_first_part_interest_rate_l4187_418713


namespace NUMINAMATH_CALUDE_a0_value_l4187_418760

theorem a0_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₀ = 32 := by
sorry

end NUMINAMATH_CALUDE_a0_value_l4187_418760


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4187_418722

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 → 15 * n < 500 → 15 * n ≤ 495 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l4187_418722


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4187_418767

/-- Given an arithmetic sequence {a_n} where a₂ + 1 is the arithmetic mean of a₁ and a₄,
    the common difference of the sequence is 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_mean : a 1 + a 4 = 2 * (a 2 + 1))  -- a₂ + 1 is the arithmetic mean of a₁ and a₄
  : a 2 - a 1 = 2 :=  -- The common difference is 2
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l4187_418767


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4187_418700

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = -2 → x^2 = 4) ∧
  ¬(∀ x : ℝ, x^2 = 4 → x = -2) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4187_418700
