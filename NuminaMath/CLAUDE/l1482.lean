import Mathlib

namespace NUMINAMATH_CALUDE_a_lt_c_lt_b_l1482_148224

noncomputable section

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Conditions
axiom derivative_f : ∀ x, HasDerivAt f (f' x) x

axiom symmetry_f' : ∀ x, f' (x - 1) = f' (1 - x)

axiom symmetry_f : ∀ x, f x = f (2 - x)

axiom monotone_f : MonotoneOn f (Set.Icc (-7) (-6))

-- Define a, b, and c
def a : ℝ := f (Real.log (6 * Real.exp 1 / 5))
def b : ℝ := f (Real.exp 0.2 - 1)
def c : ℝ := f (2 / 9)

-- Theorem to prove
theorem a_lt_c_lt_b : a < c ∧ c < b :=
  sorry

end NUMINAMATH_CALUDE_a_lt_c_lt_b_l1482_148224


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l1482_148272

theorem max_reciprocal_sum (t q u₁ u₂ : ℝ) : 
  (u₁ * u₂ = q) →
  (u₁ + u₂ = t) →
  (u₁ + u₂ = u₁^2 + u₂^2) →
  (u₁ + u₂ = u₁^4 + u₂^4) →
  (∃ (x : ℝ), x^2 - t*x + q = 0) →
  (∀ (v₁ v₂ : ℝ), v₁ * v₂ = q ∧ v₁ + v₂ = t ∧ v₁ + v₂ = v₁^2 + v₂^2 ∧ v₁ + v₂ = v₁^4 + v₂^4 →
    1/u₁^2009 + 1/u₂^2009 ≥ 1/v₁^2009 + 1/v₂^2009) →
  1/u₁^2009 + 1/u₂^2009 = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l1482_148272


namespace NUMINAMATH_CALUDE_angle_DAE_measure_l1482_148268

-- Define the points
variable (A B C D E F : Point)

-- Define the shapes
def is_equilateral_triangle (A B C : Point) : Prop := sorry

def is_regular_pentagon (B C D E F : Point) : Prop := sorry

-- Define the shared side
def share_side (A B C D E F : Point) : Prop := sorry

-- Define the angle measurement
def angle_measure (A D E : Point) : ℝ := sorry

-- Theorem statement
theorem angle_DAE_measure 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_regular_pentagon B C D E F) 
  (h3 : share_side A B C D E F) : 
  angle_measure A D E = 108 := by sorry

end NUMINAMATH_CALUDE_angle_DAE_measure_l1482_148268


namespace NUMINAMATH_CALUDE_circle_intersects_lines_iff_radius_in_range_l1482_148229

/-- A circle in a 2D Cartesian coordinate system. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a point is on a circle. -/
def on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Predicate to check if a point is at distance 1 from x-axis. -/
def dist_1_from_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 1 ∨ p.2 = -1

/-- The main theorem statement. -/
theorem circle_intersects_lines_iff_radius_in_range (r : ℝ) :
  (∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
    on_circle { center := (3, -5), radius := r } p1 ∧
    on_circle { center := (3, -5), radius := r } p2 ∧
    dist_1_from_x_axis p1 ∧
    dist_1_from_x_axis p2) ↔
  (4 < r ∧ r < 6) :=
sorry

end NUMINAMATH_CALUDE_circle_intersects_lines_iff_radius_in_range_l1482_148229


namespace NUMINAMATH_CALUDE_white_paper_bunches_l1482_148239

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

theorem white_paper_bunches :
  white_bunches * sheets_per_bunch = 
    total_sheets_removed - 
    (colored_bundles * sheets_per_bundle + scrap_heaps * sheets_per_heap) :=
by sorry

end NUMINAMATH_CALUDE_white_paper_bunches_l1482_148239


namespace NUMINAMATH_CALUDE_bird_count_problem_l1482_148249

/-- The number of grey birds initially in the cage -/
def initial_grey_birds : ℕ := 40

/-- The number of white birds next to the cage -/
def white_birds : ℕ := initial_grey_birds + 6

/-- The number of grey birds remaining in the cage after ten minutes -/
def remaining_grey_birds : ℕ := initial_grey_birds / 2

/-- The total number of birds remaining after ten minutes -/
def total_remaining_birds : ℕ := 66

theorem bird_count_problem :
  white_birds + remaining_grey_birds = total_remaining_birds :=
sorry

end NUMINAMATH_CALUDE_bird_count_problem_l1482_148249


namespace NUMINAMATH_CALUDE_min_value_theorem_l1482_148298

/-- Given a function f(x) = (1/3)ax³ + (1/2)bx² - x with a > 0 and b > 0,
    if f has a local minimum at x = 1, then the minimum value of (1/a) + (4/b) is 9 -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f : ℝ → ℝ := λ x ↦ (1/3) * a * x^3 + (1/2) * b * x^2 - x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  (∀ p q : ℝ, p > 0 → q > 0 → p + q = 1 → (1/p) + (4/q) ≥ 9) ∧
  (∃ p q : ℝ, p > 0 ∧ q > 0 ∧ p + q = 1 ∧ (1/p) + (4/q) = 9) :=
by sorry


end NUMINAMATH_CALUDE_min_value_theorem_l1482_148298


namespace NUMINAMATH_CALUDE_fourth_grade_students_l1482_148288

theorem fourth_grade_students (initial_students : ℕ) (left_students : ℕ) (increase_percentage : ℚ) : 
  initial_students = 10 →
  left_students = 4 →
  increase_percentage = 70/100 →
  (initial_students - left_students + (initial_students - left_students) * increase_percentage).floor = 10 :=
by sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l1482_148288


namespace NUMINAMATH_CALUDE_division_problem_l1482_148208

theorem division_problem (L S Q : ℕ) (h1 : L - S = 1365) (h2 : L = 1631) (h3 : L = S * Q + 35) : Q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1482_148208


namespace NUMINAMATH_CALUDE_prob_red_after_transfer_l1482_148205

-- Define the initial contents of bags A and B
def bag_A : Finset (Fin 3) := {0, 1, 2}
def bag_B : Finset (Fin 3) := {0, 1, 2}

-- Define the number of balls of each color in bag A
def red_A : ℕ := 3
def white_A : ℕ := 2
def black_A : ℕ := 5

-- Define the number of balls of each color in bag B
def red_B : ℕ := 3
def white_B : ℕ := 3
def black_B : ℕ := 4

-- Define the total number of balls in each bag
def total_A : ℕ := red_A + white_A + black_A
def total_B : ℕ := red_B + white_B + black_B

-- Define the probability of drawing a red ball from bag B after transfer
def prob_red_B : ℚ := 3 / 10

-- State the theorem
theorem prob_red_after_transfer : 
  (red_A * (red_B + 1) + white_A * red_B + black_A * red_B) / 
  (total_A * (total_B + 1)) = prob_red_B := by sorry

end NUMINAMATH_CALUDE_prob_red_after_transfer_l1482_148205


namespace NUMINAMATH_CALUDE_charley_pencils_loss_l1482_148278

theorem charley_pencils_loss (initial_pencils : ℕ) (lost_moving : ℕ) (current_pencils : ℕ)
  (h1 : initial_pencils = 30)
  (h2 : lost_moving = 6)
  (h3 : current_pencils = 16) :
  (initial_pencils - lost_moving - current_pencils : ℚ) / (initial_pencils - lost_moving : ℚ) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_charley_pencils_loss_l1482_148278


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1482_148213

-- Define the mixed number addition function
def mixed_number_add (a b c d : ℚ) : ℚ := a + b + c + d

-- Theorem 1
theorem problem_1 : 
  mixed_number_add (-2020 - 2/3) (2019 + 3/4) (-2018 - 5/6) (2017 + 1/2) = -2 - 1/4 := by sorry

-- Theorem 2
theorem problem_2 : 
  mixed_number_add (-1 - 1/2) (-2000 - 5/6) (4000 + 3/4) (-1999 - 2/3) = -5/4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1482_148213


namespace NUMINAMATH_CALUDE_system_solution_unique_l1482_148212

theorem system_solution_unique (x y : ℝ) : 
  (x + 3 * y = 2 ∧ 4 * x - y = 8) ↔ (x = 2 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1482_148212


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1482_148244

theorem complex_number_modulus : 
  let z : ℂ := (4 - 2*I) / (1 + I)
  ‖z‖ = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1482_148244


namespace NUMINAMATH_CALUDE_mara_marbles_l1482_148254

theorem mara_marbles (mara_bags : ℕ) (markus_bags : ℕ) (markus_marbles_per_bag : ℕ) :
  mara_bags = 12 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  ∃ (mara_marbles_per_bag : ℕ),
    mara_bags * mara_marbles_per_bag + 2 = markus_bags * markus_marbles_per_bag ∧
    mara_marbles_per_bag = 2 :=
by sorry

end NUMINAMATH_CALUDE_mara_marbles_l1482_148254


namespace NUMINAMATH_CALUDE_gdp_growth_time_l1482_148223

theorem gdp_growth_time (initial_gdp : ℝ) (growth_rate : ℝ) (target_gdp : ℝ) :
  initial_gdp = 8000 →
  growth_rate = 0.1 →
  target_gdp = 16000 →
  (∀ n : ℕ, n < 5 → initial_gdp * (1 + growth_rate) ^ n ≤ target_gdp) ∧
  initial_gdp * (1 + growth_rate) ^ 5 > target_gdp :=
by sorry

end NUMINAMATH_CALUDE_gdp_growth_time_l1482_148223


namespace NUMINAMATH_CALUDE_complex_fraction_power_2000_l1482_148232

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_power_2000 : ((1 - i) / (1 + i)) ^ 2000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_2000_l1482_148232


namespace NUMINAMATH_CALUDE_smallest_reducible_fraction_l1482_148246

theorem smallest_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (n - 17 : ℤ) ≠ 0 ∧
  (7 * n + 2 : ℤ) ≠ 0 ∧
  (∃ (k : ℤ), k > 1 ∧ k ∣ (n - 17) ∧ k ∣ (7 * n + 2)) ∧
  (∀ (m : ℕ), m > 0 ∧ m < n →
    (m - 17 : ℤ) = 0 ∨
    (7 * m + 2 : ℤ) = 0 ∨
    (∀ (k : ℤ), k > 1 → ¬(k ∣ (m - 17) ∧ k ∣ (7 * m + 2)))) ∧
  n = 28 :=
by sorry

end NUMINAMATH_CALUDE_smallest_reducible_fraction_l1482_148246


namespace NUMINAMATH_CALUDE_sandy_fish_count_l1482_148280

theorem sandy_fish_count (initial_fish : ℕ) (bought_fish : ℕ) : 
  initial_fish = 26 → bought_fish = 6 → initial_fish + bought_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l1482_148280


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l1482_148206

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l1482_148206


namespace NUMINAMATH_CALUDE_projection_magnitude_l1482_148215

theorem projection_magnitude (a b : ℝ × ℝ) :
  (a.1 * b.1 + a.2 * b.2 = -2) →
  (b = (1, Real.sqrt 3)) →
  let c := ((a.1 * b.1 + a.2 * b.2) / (b.1 ^ 2 + b.2 ^ 2)) • b
  (b.1 - c.1) ^ 2 + (b.2 - c.2) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_projection_magnitude_l1482_148215


namespace NUMINAMATH_CALUDE_school_journey_time_l1482_148281

/-- Calculates the remaining time to reach the classroom given the total time available,
    time to reach the school gate, and time to reach the school building from the gate. -/
def remaining_time (total_time gate_time building_time : ℕ) : ℕ :=
  total_time - (gate_time + building_time)

/-- Proves that given 30 minutes total time, 15 minutes to reach the gate,
    and 6 minutes to reach the building, there are 9 minutes left to reach the room. -/
theorem school_journey_time : remaining_time 30 15 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_school_journey_time_l1482_148281


namespace NUMINAMATH_CALUDE_student_count_prove_student_count_l1482_148277

theorem student_count (weight_difference : ℝ) (average_decrease : ℝ) : ℝ :=
  weight_difference / average_decrease

theorem prove_student_count :
  let weight_difference : ℝ := 120 - 60
  let average_decrease : ℝ := 6
  student_count weight_difference average_decrease = 10 := by
    sorry

end NUMINAMATH_CALUDE_student_count_prove_student_count_l1482_148277


namespace NUMINAMATH_CALUDE_jake_flower_charge_l1482_148261

/-- The amount Jake should charge for planting flowers -/
def flower_charge (mowing_rate : ℚ) (desired_rate : ℚ) (mowing_time : ℚ) (planting_time : ℚ) : ℚ :=
  planting_time * desired_rate + (desired_rate - mowing_rate) * mowing_time

/-- Theorem: Jake should charge $45 for planting flowers -/
theorem jake_flower_charge :
  flower_charge 15 20 1 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jake_flower_charge_l1482_148261


namespace NUMINAMATH_CALUDE_circular_film_radius_l1482_148245

/-- The radius of a circular film formed by pouring a liquid from a rectangular tank onto water -/
theorem circular_film_radius (tank_length tank_width tank_height film_thickness : ℝ)
  (tank_length_pos : tank_length > 0)
  (tank_width_pos : tank_width > 0)
  (tank_height_pos : tank_height > 0)
  (film_thickness_pos : film_thickness > 0)
  (h_tank_length : tank_length = 8)
  (h_tank_width : tank_width = 4)
  (h_tank_height : tank_height = 10)
  (h_film_thickness : film_thickness = 0.2) :
  let tank_volume := tank_length * tank_width * tank_height
  let film_radius := Real.sqrt (tank_volume / (π * film_thickness))
  film_radius = Real.sqrt (1600 / π) :=
by sorry

end NUMINAMATH_CALUDE_circular_film_radius_l1482_148245


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1482_148299

/-- The x-coordinate of the intersection point of two lines -/
def a : ℝ := 5.5

/-- The y-coordinate of the intersection point of two lines -/
def b : ℝ := 2.5

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := y = -x + 8

/-- The second line equation -/
def line2 (x y : ℝ) : Prop := 173 * y = -289 * x + 2021

theorem intersection_point_sum :
  line1 a b ∧ line2 a b → a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1482_148299


namespace NUMINAMATH_CALUDE_max_value_cube_roots_l1482_148222

theorem max_value_cube_roots (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) : 
  (a * b * c) ^ (1/3 : ℝ) + ((2 - a) * (2 - b) * (2 - c)) ^ (1/3 : ℝ) ≤ 2 ∧ 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 
    (x * x * x) ^ (1/3 : ℝ) + ((2 - x) * (2 - x) * (2 - x)) ^ (1/3 : ℝ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_roots_l1482_148222


namespace NUMINAMATH_CALUDE_transformed_quadratic_has_root_l1482_148201

/-- Given a quadratic polynomial with two roots, adding one root to the linear coefficient
    and subtracting its square from the constant term results in a polynomial with at least one root -/
theorem transformed_quadratic_has_root (a b r : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0 ∧ x ≠ y) →
  (∃ z : ℝ, z^2 + (a + r)*z + (b - r^2) = 0) ∧ 
  (r^2 + a*r + b = 0) :=
sorry

end NUMINAMATH_CALUDE_transformed_quadratic_has_root_l1482_148201


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1482_148217

theorem complex_modulus_problem (z : ℂ) (x : ℝ) 
  (h1 : z * Complex.I = 2 * Complex.I + x)
  (h2 : z.im = 2) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1482_148217


namespace NUMINAMATH_CALUDE_base4_equals_base2_l1482_148274

-- Define a function to convert a number from base 4 to decimal
def base4ToDecimal (n : ℕ) : ℕ := sorry

-- Define a function to convert a number from base 2 to decimal
def base2ToDecimal (n : ℕ) : ℕ := sorry

theorem base4_equals_base2 : base4ToDecimal 1010 = base2ToDecimal 1000100 := by sorry

end NUMINAMATH_CALUDE_base4_equals_base2_l1482_148274


namespace NUMINAMATH_CALUDE_scientific_notation_equality_l1482_148236

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 384000000

/-- The coefficient in scientific notation -/
def coefficient : ℝ := 3.84

/-- The exponent in scientific notation -/
def exponent : ℕ := 8

/-- Theorem stating that the original number is equal to its scientific notation form -/
theorem scientific_notation_equality :
  original_number = coefficient * (10 : ℝ) ^ exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equality_l1482_148236


namespace NUMINAMATH_CALUDE_sqrt_cubed_equals_64_l1482_148204

theorem sqrt_cubed_equals_64 (x : ℝ) : (Real.sqrt x)^3 = 64 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cubed_equals_64_l1482_148204


namespace NUMINAMATH_CALUDE_fraction_equality_l1482_148294

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 15) :
  m / q = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1482_148294


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1482_148255

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) : 
  (∀ x : ℤ, (2 - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1482_148255


namespace NUMINAMATH_CALUDE_repunit_existence_l1482_148291

theorem repunit_existence (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_11 : p > 11) :
  ∃ k : Nat, ∃ n : Nat, p * k = (10^n - 1) / 9 := by
  sorry

end NUMINAMATH_CALUDE_repunit_existence_l1482_148291


namespace NUMINAMATH_CALUDE_ab_value_l1482_148221

theorem ab_value (a b : ℝ) (h : |3*a - 1| + b^2 = 0) : a^b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1482_148221


namespace NUMINAMATH_CALUDE_regression_independence_correct_statement_l1482_148216

/-- Definition of regression analysis -/
def regression_analysis : Type := Unit

/-- Definition of independence test -/
def independence_test : Type := Unit

/-- Property: Regression analysis studies correlation between two variables -/
axiom regression_studies_correlation : regression_analysis → Prop

/-- Property: Independence test analyzes relationship between two variables -/
axiom independence_analyzes_relationship : independence_test → Prop

/-- Property: Independence test cannot determine relationships with 100% certainty -/
axiom independence_not_certain : independence_test → Prop

/-- The correct statement about regression analysis and independence test -/
def correct_statement : Prop :=
  ∃ (ra : regression_analysis) (it : independence_test),
    regression_studies_correlation ra ∧
    independence_analyzes_relationship it

theorem regression_independence_correct_statement :
  correct_statement :=
sorry

end NUMINAMATH_CALUDE_regression_independence_correct_statement_l1482_148216


namespace NUMINAMATH_CALUDE_tangent_double_angle_subtraction_l1482_148207

theorem tangent_double_angle_subtraction (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/5) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tangent_double_angle_subtraction_l1482_148207


namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l1482_148253

/-- Represents a sampling method -/
inductive SamplingMethod
| Stratified
| Simple
| Cluster
| Systematic

/-- Represents the student population -/
structure Population where
  total : Nat
  male : Nat
  female : Nat

/-- Represents the sample -/
structure Sample where
  total : Nat
  male : Nat
  female : Nat

def is_stratified (pop : Population) (sam : Sample) : Prop :=
  (pop.male : Real) / pop.total = (sam.male : Real) / sam.total ∧
  (pop.female : Real) / pop.total = (sam.female : Real) / sam.total

theorem stratified_sampling_proof 
  (pop : Population) 
  (sam : Sample) 
  (h1 : pop.total = 1000) 
  (h2 : pop.male = 400) 
  (h3 : pop.female = 600) 
  (h4 : sam.total = 100) 
  (h5 : sam.male = 40) 
  (h6 : sam.female = 60) 
  (h7 : is_stratified pop sam) : 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l1482_148253


namespace NUMINAMATH_CALUDE_tonya_final_stamps_l1482_148243

/-- The number of matches equivalent to one stamp -/
def matches_per_stamp : ℕ := 12

/-- The number of matches in each matchbook -/
def matches_per_matchbook : ℕ := 24

/-- The number of stamps Tonya initially has -/
def tonya_initial_stamps : ℕ := 13

/-- The number of matchbooks Jimmy has -/
def jimmy_matchbooks : ℕ := 5

/-- Calculate the number of stamps Tonya has left after trading with Jimmy -/
def tonya_stamps_left : ℕ := 
  tonya_initial_stamps - (jimmy_matchbooks * matches_per_matchbook) / matches_per_stamp

theorem tonya_final_stamps : tonya_stamps_left = 3 := by
  sorry

end NUMINAMATH_CALUDE_tonya_final_stamps_l1482_148243


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1482_148214

theorem solution_set_of_inequality (x : ℝ) : 
  (2 * x - x^2 > 0) ↔ (x > 0 ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1482_148214


namespace NUMINAMATH_CALUDE_pascals_identity_l1482_148230

theorem pascals_identity (n k : ℕ) : 
  Nat.choose n k + Nat.choose n (k + 1) = Nat.choose (n + 1) (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_pascals_identity_l1482_148230


namespace NUMINAMATH_CALUDE_mixed_doubles_selection_count_l1482_148247

theorem mixed_doubles_selection_count :
  let male_count : ℕ := 5
  let female_count : ℕ := 4
  male_count * female_count = 20 :=
by sorry

end NUMINAMATH_CALUDE_mixed_doubles_selection_count_l1482_148247


namespace NUMINAMATH_CALUDE_program_output_25_l1482_148226

theorem program_output_25 (x : ℝ) : 
  ((x < 0 ∧ (x + 1)^2 = 25) ∨ (x ≥ 0 ∧ (x - 1)^2 = 25)) ↔ (x = 6 ∨ x = -6) := by
  sorry

end NUMINAMATH_CALUDE_program_output_25_l1482_148226


namespace NUMINAMATH_CALUDE_teacher_distribution_count_l1482_148256

/-- The number of ways to distribute 4 teachers among 3 middle schools -/
def distribute_teachers : ℕ :=
  Nat.choose 4 2 * Nat.factorial 3

/-- Theorem: The number of ways to distribute 4 teachers among 3 middle schools,
    with each school having at least one teacher, is equal to 36 -/
theorem teacher_distribution_count : distribute_teachers = 36 := by
  sorry

end NUMINAMATH_CALUDE_teacher_distribution_count_l1482_148256


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_320_l1482_148218

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_320 :
  closest_perfect_square 320 = 324 :=
sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_320_l1482_148218


namespace NUMINAMATH_CALUDE_value_of_3a_plus_6b_l1482_148295

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2*b - 1 = 0) : 3*a + 6*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3a_plus_6b_l1482_148295


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1482_148297

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l1482_148297


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1482_148225

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1482_148225


namespace NUMINAMATH_CALUDE_divisibility_properties_l1482_148290

theorem divisibility_properties (a : ℤ) : 
  (2 ∣ (a^2 - a)) ∧ (3 ∣ (a^3 - a)) := by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l1482_148290


namespace NUMINAMATH_CALUDE_sum_first_44_is_116_l1482_148210

/-- Represents the sequence where the nth 1 is followed by n 3s -/
def specialSequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| k + 1 => if k < (n * (n + 1)) / 2 then
             if k = (n * (n - 1)) / 2 then 1 else 3
           else specialSequence (n + 1) k

/-- The sum of the first 44 terms of the special sequence -/
def sumFirst44 : ℕ := (List.range 44).map (specialSequence 1) |>.sum

/-- Theorem stating that the sum of the first 44 terms is 116 -/
theorem sum_first_44_is_116 : sumFirst44 = 116 := by sorry

end NUMINAMATH_CALUDE_sum_first_44_is_116_l1482_148210


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1482_148265

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1482_148265


namespace NUMINAMATH_CALUDE_hotel_rooms_l1482_148270

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℚ) (total_revenue : ℚ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l1482_148270


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1482_148219

/-- The speed of a man rowing in still water, given his downstream speed and current speed. -/
theorem mans_rowing_speed (downstream_distance : ℝ) (downstream_time : ℝ) (current_speed : ℝ) : 
  downstream_distance / downstream_time * 3600 / 1000 - current_speed = 6 :=
by
  sorry

#check mans_rowing_speed 110 44 3

end NUMINAMATH_CALUDE_mans_rowing_speed_l1482_148219


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1482_148286

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1482_148286


namespace NUMINAMATH_CALUDE_expand_polynomial_product_l1482_148259

theorem expand_polynomial_product : ∀ x : ℝ,
  (3 * x^2 - 2 * x + 4) * (-4 * x^2 + 3 * x - 6) =
  -12 * x^4 + 17 * x^3 - 40 * x^2 + 24 * x - 24 :=
by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_product_l1482_148259


namespace NUMINAMATH_CALUDE_final_combined_price_theorem_l1482_148269

/-- Calculates the final price of an item after applying discount and tax --/
def finalPrice (initialPrice : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  initialPrice * (1 - discount) * (1 + tax)

/-- Calculates the price of an accessory after applying tax --/
def accessoryPrice (price : ℝ) (tax : ℝ) : ℝ :=
  price * (1 + tax)

/-- Theorem stating the final combined price of iPhone and accessories --/
theorem final_combined_price_theorem 
  (iPhoneInitialPrice : ℝ) 
  (iPhoneDiscount1 iPhoneDiscount2 : ℝ)
  (iPhoneTax1 iPhoneTax2 : ℝ)
  (screenProtectorPrice casePrice : ℝ)
  (accessoriesTax : ℝ)
  (h1 : iPhoneInitialPrice = 1000)
  (h2 : iPhoneDiscount1 = 0.1)
  (h3 : iPhoneDiscount2 = 0.2)
  (h4 : iPhoneTax1 = 0.08)
  (h5 : iPhoneTax2 = 0.06)
  (h6 : screenProtectorPrice = 30)
  (h7 : casePrice = 50)
  (h8 : accessoriesTax = 0.05) :
  let iPhoneFinalPrice := finalPrice (finalPrice iPhoneInitialPrice iPhoneDiscount1 iPhoneTax1) iPhoneDiscount2 iPhoneTax2
  let totalAccessoriesPrice := accessoryPrice screenProtectorPrice accessoriesTax + accessoryPrice casePrice accessoriesTax
  iPhoneFinalPrice + totalAccessoriesPrice = 908.256 := by
    sorry


end NUMINAMATH_CALUDE_final_combined_price_theorem_l1482_148269


namespace NUMINAMATH_CALUDE_factorizations_of_2079_l1482_148251

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def valid_factorization (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧ a * b = 2079

def distinct_factorizations (f1 f2 : ℕ × ℕ) : Prop :=
  f1.1 ≠ f2.1 ∧ f1.1 ≠ f2.2

theorem factorizations_of_2079 :
  ∃ (f1 f2 : ℕ × ℕ),
    valid_factorization f1.1 f1.2 ∧
    valid_factorization f2.1 f2.2 ∧
    distinct_factorizations f1 f2 ∧
    ∀ (f : ℕ × ℕ), valid_factorization f.1 f.2 →
      (f = f1 ∨ f = f2 ∨ f = (f1.2, f1.1) ∨ f = (f2.2, f2.1)) :=
sorry

end NUMINAMATH_CALUDE_factorizations_of_2079_l1482_148251


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l1482_148227

theorem distance_to_origin_of_complex_fraction : 
  let z : ℂ := (2 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_fraction_l1482_148227


namespace NUMINAMATH_CALUDE_scientific_notation_19672_l1482_148263

theorem scientific_notation_19672 :
  19672 = 1.9672 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_19672_l1482_148263


namespace NUMINAMATH_CALUDE_unique_solution_is_two_l1482_148237

theorem unique_solution_is_two : 
  ∃! (x : ℝ), x > 0 ∧ x^(2^2) = 2^(x^2) ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_two_l1482_148237


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l1482_148262

def question_values : List ℕ := [100, 300, 600, 1000, 1500, 2500, 4000, 6500]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase :
  let pairs := consecutive_pairs question_values
  let increases := List.map (fun (p : ℕ × ℕ) => percent_increase p.1 p.2) pairs
  List.argmin id increases = some 3 := by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l1482_148262


namespace NUMINAMATH_CALUDE_sector_central_angle_l1482_148257

theorem sector_central_angle (r : ℝ) (θ : ℝ) (h : r > 0) :
  2 * r + r * θ = π * r / 2 → θ = π - 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1482_148257


namespace NUMINAMATH_CALUDE_xiao_ma_calculation_l1482_148200

theorem xiao_ma_calculation (x : ℤ) : 41 - x = 12 → 41 + x = 70 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ma_calculation_l1482_148200


namespace NUMINAMATH_CALUDE_total_cans_in_both_closets_l1482_148231

/-- Represents the capacity of a closet for storing cans -/
structure ClosetCapacity where
  cansPerRow : Nat
  rowsPerShelf : Nat
  shelves : Nat

/-- Calculates the total number of cans that can be stored in a closet -/
def totalCansInCloset (capacity : ClosetCapacity) : Nat :=
  capacity.cansPerRow * capacity.rowsPerShelf * capacity.shelves

/-- The capacity of the first closet -/
def firstCloset : ClosetCapacity :=
  { cansPerRow := 12, rowsPerShelf := 4, shelves := 10 }

/-- The capacity of the second closet -/
def secondCloset : ClosetCapacity :=
  { cansPerRow := 15, rowsPerShelf := 5, shelves := 8 }

/-- Theorem stating the total number of cans Jack can store in both closets -/
theorem total_cans_in_both_closets :
  totalCansInCloset firstCloset + totalCansInCloset secondCloset = 1080 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_in_both_closets_l1482_148231


namespace NUMINAMATH_CALUDE_correct_verb_forms_l1482_148267

-- Define the structure of a sentence
structure Sentence where
  subject : String
  verb1 : String
  verb2 : String

-- Define a predicate for plural subjects
def is_plural (s : String) : Prop := s.endsWith "s"

-- Define a predicate for partial references
def is_partial_reference (s : String) : Prop := s = "some"

-- Define a predicate for plural verb forms
def is_plural_verb (v : String) : Prop := v = "are" ∨ v = "seem"

-- Theorem statement
theorem correct_verb_forms (s : Sentence) 
  (h1 : is_plural s.subject) 
  (h2 : is_partial_reference "some") : 
  is_plural_verb s.verb1 ∧ is_plural_verb s.verb2 := by
  sorry

-- Example usage
def example_sentence : Sentence := {
  subject := "Such phenomena"
  verb1 := "are"
  verb2 := "seem"
}

#check correct_verb_forms example_sentence

end NUMINAMATH_CALUDE_correct_verb_forms_l1482_148267


namespace NUMINAMATH_CALUDE_nearest_multiple_of_21_to_2304_l1482_148292

theorem nearest_multiple_of_21_to_2304 :
  ∀ n : ℤ, n ≠ 2304 → 21 ∣ n → |n - 2304| ≥ |2310 - 2304| :=
by sorry

end NUMINAMATH_CALUDE_nearest_multiple_of_21_to_2304_l1482_148292


namespace NUMINAMATH_CALUDE_no_complex_root_for_integer_polynomial_l1482_148238

/-- A polynomial of degree 4 with leading coefficient 1 and integer coefficients -/
def IntegerPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℤ, ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d

/-- The property that a polynomial has two integer roots -/
def HasTwoIntegerRoots (P : ℝ → ℝ) : Prop :=
  ∃ p q : ℤ, P p = 0 ∧ P q = 0

/-- Complex number of the form (a + b*i)/2 where a and b are integers and b is non-zero -/
def ComplexRoot (z : ℂ) : Prop :=
  ∃ a b : ℤ, z = (a + b*Complex.I)/2 ∧ b ≠ 0

theorem no_complex_root_for_integer_polynomial (P : ℝ → ℝ) :
  IntegerPolynomial P → HasTwoIntegerRoots P →
  ¬∃ z : ℂ, ComplexRoot z ∧ (P z.re = 0 ∧ P z.im = 0) :=
sorry

end NUMINAMATH_CALUDE_no_complex_root_for_integer_polynomial_l1482_148238


namespace NUMINAMATH_CALUDE_min_abs_z_minus_one_l1482_148285

/-- For any complex number Z satisfying |Z-1| = |Z+1|, the minimum value of |Z-1| is 1. -/
theorem min_abs_z_minus_one (Z : ℂ) (h : Complex.abs (Z - 1) = Complex.abs (Z + 1)) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (W : ℂ), Complex.abs (W - 1) = Complex.abs (W + 1) → Complex.abs (W - 1) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_minus_one_l1482_148285


namespace NUMINAMATH_CALUDE_alex_not_reading_probability_l1482_148235

theorem alex_not_reading_probability (p : ℚ) (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_alex_not_reading_probability_l1482_148235


namespace NUMINAMATH_CALUDE_parabola_vertex_l1482_148271

/-- The parabola defined by y = 2(x+9)^2 - 3 has vertex at (-9, -3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x + 9)^2 - 3 → (∃ a b : ℝ, (a, b) = (-9, -3) ∧ ∀ x, y ≥ 2 * (x + 9)^2 - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1482_148271


namespace NUMINAMATH_CALUDE_train_speed_in_kmh_l1482_148233

-- Define the length of the train in meters
def train_length : ℝ := 280

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 20

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem to prove
theorem train_speed_in_kmh :
  (train_length / crossing_time) * ms_to_kmh = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_in_kmh_l1482_148233


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1482_148273

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  0 < b ∧ b < a ∧ a + b = 7 * (a - b) → a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l1482_148273


namespace NUMINAMATH_CALUDE_f_of_A_eq_l1482_148250

/-- The matrix A --/
def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 2, 3]

/-- The polynomial function f --/
def f (x : Matrix (Fin 2) (Fin 2) ℤ) : Matrix (Fin 2) (Fin 2) ℤ := x^2 - 5 • x

/-- Theorem stating that f(A) equals the given result --/
theorem f_of_A_eq : f A = !![(-6), 1; (-2), (-8)] := by sorry

end NUMINAMATH_CALUDE_f_of_A_eq_l1482_148250


namespace NUMINAMATH_CALUDE_chip_price_reduction_l1482_148282

theorem chip_price_reduction (a b : ℝ) : 
  (∃ (price_after_first_reduction : ℝ), 
    price_after_first_reduction = a * (1 - 0.1) ∧
    b = price_after_first_reduction * (1 - 0.2)) →
  b = a * (1 - 0.1) * (1 - 0.2) := by
sorry

end NUMINAMATH_CALUDE_chip_price_reduction_l1482_148282


namespace NUMINAMATH_CALUDE_inverse_prop_is_false_l1482_148209

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the interval [a,b]
variable (a b : ℝ)

-- State that f is continuous on [a,b]
variable (hf : ContinuousOn f (Set.Icc a b))

-- Define the original proposition
def original_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, f a * f b < 0 → ∃ c ∈ Set.Ioo a b, f c = 0

-- Define the inverse proposition
def inverse_prop : Prop :=
  ∀ x ∈ Set.Ioo a b, (∃ c ∈ Set.Ioo a b, f c = 0) → f a * f b < 0

-- State the theorem
theorem inverse_prop_is_false
  (h : original_prop f a b) : ¬(inverse_prop f a b) := by
  sorry


end NUMINAMATH_CALUDE_inverse_prop_is_false_l1482_148209


namespace NUMINAMATH_CALUDE_probability_x_plus_y_even_l1482_148241

def X := Finset.range 5
def Y := Finset.range 4

theorem probability_x_plus_y_even :
  let total_outcomes := X.card * Y.card
  let favorable_outcomes := (X.filter (λ x => x % 2 = 0)).card * (Y.filter (λ y => y % 2 = 0)).card +
                            (X.filter (λ x => x % 2 = 1)).card * (Y.filter (λ y => y % 2 = 1)).card
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_even_l1482_148241


namespace NUMINAMATH_CALUDE_value_of_x_minus_4y_l1482_148258

theorem value_of_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) :
  x - 4 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_minus_4y_l1482_148258


namespace NUMINAMATH_CALUDE_max_area_is_12_l1482_148279

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 5 ∧
  dist q.B q.C = 5 ∧
  dist q.C q.D = 5 ∧
  dist q.D q.A = 3

-- Define the deformation that maximizes ∠ABC
def max_angle_deformation (q : Quadrilateral) : Quadrilateral :=
  sorry

-- Define the area calculation function
def area (q : Quadrilateral) : ℝ :=
  sorry

-- Theorem statement
theorem max_area_is_12 (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  area (max_angle_deformation q) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_area_is_12_l1482_148279


namespace NUMINAMATH_CALUDE_root_sum_equation_l1482_148276

theorem root_sum_equation (n m : ℝ) (hn : n ≠ 0) 
  (hroot : n^2 + m*n + 3*n = 0) : m + n = -3 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equation_l1482_148276


namespace NUMINAMATH_CALUDE_quadratic_coefficient_values_l1482_148248

/-- Given an algebraic expression x^2 + px + q, prove that p = 0 and q = -6
    when the expression equals -5 for x = -1 and 3 for x = 3. -/
theorem quadratic_coefficient_values (p q : ℝ) : 
  ((-1)^2 + p*(-1) + q = -5) ∧ (3^2 + p*3 + q = 3) → p = 0 ∧ q = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_values_l1482_148248


namespace NUMINAMATH_CALUDE_divisor_between_l1482_148284

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : 0 < a) (hb : 0 < b)
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (hneq : a ≠ b)
  (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b := by
sorry

end NUMINAMATH_CALUDE_divisor_between_l1482_148284


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1482_148287

theorem p_sufficient_not_necessary_for_q :
  (∃ x : ℝ, x = 2 ∧ x^2 ≠ 4) ∨
  (∃ x : ℝ, x^2 = 4 ∧ x ≠ 2) ∨
  (∀ x : ℝ, x = 2 → x^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1482_148287


namespace NUMINAMATH_CALUDE_line_points_property_l1482_148211

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l1482_148211


namespace NUMINAMATH_CALUDE_second_pipe_fill_time_l1482_148234

theorem second_pipe_fill_time (t1 t2 t3 t_all : ℝ) (h1 : t1 = 10) (h2 : t3 = 40) (h3 : t_all = 6.31578947368421) 
  (h4 : 1 / t1 + 1 / t2 - 1 / t3 = 1 / t_all) : t2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_pipe_fill_time_l1482_148234


namespace NUMINAMATH_CALUDE_max_class_size_is_17_l1482_148242

/-- Represents a school with students and buses for an excursion. -/
structure School where
  total_students : ℕ
  num_buses : ℕ
  seats_per_bus : ℕ

/-- Checks if it's possible to seat all students with the given maximum class size. -/
def can_seat_all (s : School) (max_class_size : ℕ) : Prop :=
  ∀ (class_sizes : List ℕ),
    (class_sizes.sum = s.total_students) →
    (∀ size ∈ class_sizes, size ≤ max_class_size) →
    ∃ (allocation : List (List ℕ)),
      (allocation.length ≤ s.num_buses) ∧
      (∀ bus ∈ allocation, bus.sum ≤ s.seats_per_bus) ∧
      (allocation.join.sum = s.total_students)

/-- The main theorem stating that 17 is the maximum class size for the given school configuration. -/
theorem max_class_size_is_17 (s : School)
    (h1 : s.total_students = 920)
    (h2 : s.num_buses = 16)
    (h3 : s.seats_per_bus = 71) :
    (can_seat_all s 17 ∧ ¬can_seat_all s 18) :=
  sorry


end NUMINAMATH_CALUDE_max_class_size_is_17_l1482_148242


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l1482_148202

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the x-axis --/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to prove --/
theorem symmetric_point_about_x_axis :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨2, -1⟩
  symmetricAboutXAxis A B := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l1482_148202


namespace NUMINAMATH_CALUDE_total_over_budget_l1482_148240

def project_budget (project : Char) : ℕ :=
  match project with
  | 'A' => 150000
  | 'B' => 120000
  | 'C' => 80000
  | _ => 0

def allocation_count (project : Char) : ℕ :=
  match project with
  | 'A' => 10
  | 'B' => 6
  | 'C' => 18
  | _ => 0

def allocation_period (project : Char) : ℕ :=
  match project with
  | 'A' => 2
  | 'B' => 3
  | 'C' => 1
  | _ => 0

def actual_spent (project : Char) : ℕ :=
  match project with
  | 'A' => 98450
  | 'B' => 72230
  | 'C' => 43065
  | _ => 0

def months_passed : ℕ := 9

def expected_expenditure (project : Char) : ℚ :=
  (project_budget project : ℚ) / (allocation_count project : ℚ) *
  ((months_passed : ℚ) / (allocation_period project : ℚ)).floor

def project_difference (project : Char) : ℚ :=
  (actual_spent project : ℚ) - expected_expenditure project

theorem total_over_budget :
  (project_difference 'A' + project_difference 'B' + project_difference 'C') = 38745 := by
  sorry

end NUMINAMATH_CALUDE_total_over_budget_l1482_148240


namespace NUMINAMATH_CALUDE_socorro_training_days_l1482_148228

/-- Calculates the number of days required to complete a training program. -/
def trainingDays (totalHours : ℕ) (dailyMinutes : ℕ) : ℕ :=
  (totalHours * 60) / dailyMinutes

/-- Proves that given 5 hours of total training time and 30 minutes of daily training,
    it takes 10 days to complete the training. -/
theorem socorro_training_days :
  trainingDays 5 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_socorro_training_days_l1482_148228


namespace NUMINAMATH_CALUDE_intersection_implies_a_nonpositive_l1482_148296

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem intersection_implies_a_nonpositive (a : ℝ) :
  (A ∩ B a).Nonempty → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_nonpositive_l1482_148296


namespace NUMINAMATH_CALUDE_bedroom_curtain_length_l1482_148275

theorem bedroom_curtain_length :
  let total_fabric_area : ℝ := 16 * 12
  let living_room_curtain_area : ℝ := 4 * 6
  let bedroom_curtain_width : ℝ := 2
  let remaining_fabric_area : ℝ := 160
  let bedroom_curtain_area : ℝ := total_fabric_area - living_room_curtain_area - remaining_fabric_area
  bedroom_curtain_area / bedroom_curtain_width = 4 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_curtain_length_l1482_148275


namespace NUMINAMATH_CALUDE_right_triangle_identification_l1482_148220

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  ¬(is_right_triangle (Real.sqrt 2) (Real.sqrt 3) 2) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 9 16 25) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l1482_148220


namespace NUMINAMATH_CALUDE_max_expensive_price_theorem_l1482_148266

/-- Represents a set of products with their prices -/
structure ProductSet where
  prices : Finset ℝ
  count : Nat
  avg_price : ℝ
  min_price : ℝ
  low_price_count : Nat
  low_price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (ps : ProductSet) : ℝ :=
  ps.count * ps.avg_price
    - (ps.low_price_count * ps.min_price
      + (ps.count - ps.low_price_count - 1) * ps.low_price_threshold)

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_theorem (ps : ProductSet)
  (h_count : ps.count = 25)
  (h_avg_price : ps.avg_price = 1200)
  (h_min_price : ps.min_price = 400)
  (h_low_price_count : ps.low_price_count = 12)
  (h_low_price_threshold : ps.low_price_threshold = 1000)
  (h_prices_above_min : ∀ p ∈ ps.prices, p ≥ ps.min_price)
  (h_low_price_count_correct : (ps.prices.filter (· < ps.low_price_threshold)).card = ps.low_price_count) :
  max_expensive_price ps = 13200 := by
  sorry

end NUMINAMATH_CALUDE_max_expensive_price_theorem_l1482_148266


namespace NUMINAMATH_CALUDE_conclusion_one_conclusion_three_l1482_148203

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem for the first correct conclusion
theorem conclusion_one : custom_op 2 (-2) = 6 := by sorry

-- Theorem for the third correct conclusion
theorem conclusion_three (a b : ℝ) (h : a + b = 0) :
  custom_op a a + custom_op b b = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_conclusion_one_conclusion_three_l1482_148203


namespace NUMINAMATH_CALUDE_walk_distance_proof_l1482_148264

/-- The distance Rajesh and Hiro walked together -/
def distance_together : ℝ := 7

/-- Hiro's walking distance -/
def hiro_distance : ℝ := distance_together

/-- Rajesh's walking distance -/
def rajesh_distance : ℝ := 18

theorem walk_distance_proof :
  (4 * hiro_distance - 10 = rajesh_distance) →
  distance_together = 7 := by
  sorry

end NUMINAMATH_CALUDE_walk_distance_proof_l1482_148264


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l1482_148260

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∀ x, f x ≥ n) ∧ (m + n = 0) := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l1482_148260


namespace NUMINAMATH_CALUDE_fixed_fee_calculation_l1482_148283

/-- Represents a cable service bill -/
structure CableBill where
  fixed_fee : ℝ
  hourly_rate : ℝ
  usage_hours : ℝ

/-- Calculates the total bill amount -/
def bill_amount (b : CableBill) : ℝ :=
  b.fixed_fee + b.hourly_rate * b.usage_hours

theorem fixed_fee_calculation 
  (feb : CableBill) (mar : CableBill) 
  (h_feb_amount : bill_amount feb = 20.72)
  (h_mar_amount : bill_amount mar = 35.28)
  (h_same_fee : feb.fixed_fee = mar.fixed_fee)
  (h_same_rate : feb.hourly_rate = mar.hourly_rate)
  (h_triple_usage : mar.usage_hours = 3 * feb.usage_hours) :
  feb.fixed_fee = 13.44 := by
sorry

end NUMINAMATH_CALUDE_fixed_fee_calculation_l1482_148283


namespace NUMINAMATH_CALUDE_correct_percentage_l1482_148293

theorem correct_percentage (x : ℕ) : 
  let total := 6 * x
  let missed := 2 * x
  let correct := total - missed
  (correct : ℚ) / total * 100 = 200 / 3 := by sorry

end NUMINAMATH_CALUDE_correct_percentage_l1482_148293


namespace NUMINAMATH_CALUDE_only_unit_circle_has_nontrivial_solution_l1482_148289

theorem only_unit_circle_has_nontrivial_solution :
  ∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ Real.sqrt (a^2 + b^2) = 1 ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = a - b → a = 0 ∧ b = 0) ∧
  (∀ (a b : ℝ), Real.sqrt (a^2 + b^2) = 3 * (a + b) → a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_unit_circle_has_nontrivial_solution_l1482_148289


namespace NUMINAMATH_CALUDE_triangle_height_l1482_148252

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 8 → area = 16 → area = (base * height) / 2 → height = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l1482_148252
