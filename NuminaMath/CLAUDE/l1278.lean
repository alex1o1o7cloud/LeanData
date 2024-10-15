import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l1278_127827

theorem fraction_sum_simplification :
  3 / 462 + 17 / 42 = 95 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l1278_127827


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l1278_127869

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  foci_x_eq : foci1.1 = foci2.1
  passes_through : (point.1 - h)^2 / a^2 + (point.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) 
  (h_foci1 : e.foci1 = (-4, 1)) 
  (h_foci2 : e.foci2 = (-4, 5)) 
  (h_point : e.point = (1, 3)) : 
  e.a + e.k = 8 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_eq_eight_l1278_127869


namespace NUMINAMATH_CALUDE_rational_function_property_l1278_127891

theorem rational_function_property (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_rational_function_property_l1278_127891


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1278_127888

theorem quadratic_root_relation (p : ℚ) : 
  (∃ x y : ℚ, x = 3 * y ∧ 
   x^2 - (3*p - 2)*x + p^2 - 1 = 0 ∧ 
   y^2 - (3*p - 2)*y + p^2 - 1 = 0) ↔ 
  (p = 2 ∨ p = 14/11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1278_127888


namespace NUMINAMATH_CALUDE_optimal_box_height_l1278_127890

/-- Represents the dimensions of an open-top rectangular box with a square base -/
structure BoxDimensions where
  side : ℝ
  height : ℝ

/-- The volume of the box -/
def volume (b : BoxDimensions) : ℝ := b.side^2 * b.height

/-- The surface area of the box -/
def surfaceArea (b : BoxDimensions) : ℝ := b.side^2 + 4 * b.side * b.height

/-- The constraint that the volume must be 4 -/
def volumeConstraint (b : BoxDimensions) : Prop := volume b = 4

theorem optimal_box_height :
  ∃ (b : BoxDimensions), volumeConstraint b ∧
    (∀ (b' : BoxDimensions), volumeConstraint b' → surfaceArea b ≤ surfaceArea b') ∧
    b.height = 1 := by
  sorry

end NUMINAMATH_CALUDE_optimal_box_height_l1278_127890


namespace NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l1278_127851

theorem product_seven_reciprocal_squares_sum (a b : ℕ) (h : a * b = 7) :
  (1 : ℚ) / (a ^ 2) + (1 : ℚ) / (b ^ 2) = 50 / 49 := by
  sorry

end NUMINAMATH_CALUDE_product_seven_reciprocal_squares_sum_l1278_127851


namespace NUMINAMATH_CALUDE_inverse_mod_53_l1278_127820

theorem inverse_mod_53 (h : (21⁻¹ : ZMod 53) = 17) : (32⁻¹ : ZMod 53) = 36 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l1278_127820


namespace NUMINAMATH_CALUDE_pool_filling_proof_l1278_127841

/-- The rate at which the first hose sprays water in gallons per hour -/
def first_hose_rate : ℝ := 50

/-- The rate at which the second hose sprays water in gallons per hour -/
def second_hose_rate : ℝ := 70

/-- The capacity of the pool in gallons -/
def pool_capacity : ℝ := 390

/-- The time the first hose was used alone in hours -/
def first_hose_time : ℝ := 3

/-- The time both hoses were used together in hours -/
def both_hoses_time : ℝ := 2

theorem pool_filling_proof : 
  first_hose_rate * first_hose_time + 
  (first_hose_rate + second_hose_rate) * both_hoses_time = 
  pool_capacity := by sorry

end NUMINAMATH_CALUDE_pool_filling_proof_l1278_127841


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l1278_127878

def n : ℕ := 2024

theorem floor_expression_equals_eight :
  ⌊(2025^3 : ℚ) / (2023 * 2024) - (2023^3 : ℚ) / (2024 * 2025)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l1278_127878


namespace NUMINAMATH_CALUDE_choir_arrangement_theorem_l1278_127863

theorem choir_arrangement_theorem :
  ∃ (m : ℕ), 
    (∃ (k : ℕ), m = k^2 + 6) ∧ 
    (∃ (n : ℕ), m = n * (n + 6)) ∧
    (∀ (x : ℕ), 
      ((∃ (y : ℕ), x = y^2 + 6) ∧ 
       (∃ (z : ℕ), x = z * (z + 6))) → 
      x ≤ m) ∧
    m = 294 := by
  sorry

end NUMINAMATH_CALUDE_choir_arrangement_theorem_l1278_127863


namespace NUMINAMATH_CALUDE_original_plus_increase_equals_current_l1278_127816

/-- The number of bacteria originally in the petri dish -/
def original_bacteria : ℕ := 600

/-- The current number of bacteria in the petri dish -/
def current_bacteria : ℕ := 8917

/-- The increase in the number of bacteria -/
def bacteria_increase : ℕ := 8317

/-- Theorem stating that the original number of bacteria plus the increase
    equals the current number of bacteria -/
theorem original_plus_increase_equals_current :
  original_bacteria + bacteria_increase = current_bacteria := by
  sorry

end NUMINAMATH_CALUDE_original_plus_increase_equals_current_l1278_127816


namespace NUMINAMATH_CALUDE_kendra_change_l1278_127821

/-- Calculates the change received after a purchase -/
def calculate_change (toy_price hat_price : ℕ) (num_toys num_hats : ℕ) (paid : ℕ) : ℕ :=
  paid - (toy_price * num_toys + hat_price * num_hats)

/-- Proves that Kendra received $30 in change -/
theorem kendra_change : calculate_change 20 10 2 3 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_kendra_change_l1278_127821


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1278_127871

-- Define the universal set U
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define set A
def A : Set Char := {'a', 'b', 'c', 'd'}

-- Define set B
def B : Set Char := {'d', 'e'}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1278_127871


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l1278_127844

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each fourth-grade classroom -/
def students_per_classroom : ℕ := 24

/-- The number of guinea pigs in each fourth-grade classroom -/
def guinea_pigs_per_classroom : ℕ := 2

/-- The difference between the total number of students and guinea pigs in all classrooms -/
theorem student_guinea_pig_difference :
  num_classrooms * students_per_classroom - num_classrooms * guinea_pigs_per_classroom = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l1278_127844


namespace NUMINAMATH_CALUDE_no_prime_pairs_for_square_diff_l1278_127875

theorem no_prime_pairs_for_square_diff (a b : ℕ) : 
  a ≤ 100 → b ≤ 100 → Prime a → Prime b → a^2 - b^2 ≠ 25 :=
by sorry

end NUMINAMATH_CALUDE_no_prime_pairs_for_square_diff_l1278_127875


namespace NUMINAMATH_CALUDE_cos_600_degrees_l1278_127883

theorem cos_600_degrees : Real.cos (600 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_600_degrees_l1278_127883


namespace NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l1278_127899

/-- The equation of a potential ellipse with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop :=
  x^2 / (k - 1) + y^2 / (5 - k) = 1

/-- Condition for k -/
def k_condition (k : ℝ) : Prop :=
  1 < k ∧ k < 5

/-- Definition of an ellipse (simplified for this problem) -/
def is_ellipse (k : ℝ) : Prop :=
  k_condition k ∧ k ≠ 3

theorem ellipse_condition_necessary_not_sufficient :
  (∀ k, is_ellipse k → k_condition k) ∧
  ¬(∀ k, k_condition k → is_ellipse k) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_necessary_not_sufficient_l1278_127899


namespace NUMINAMATH_CALUDE_triangle_max_area_l1278_127804

/-- Given a triangle ABC where the sum of two sides is 4 and one angle is 30°, 
    the maximum area of the triangle is 1 -/
theorem triangle_max_area (a b : ℝ) (C : ℝ) (h1 : a + b = 4) (h2 : C = 30 * π / 180) :
  ∀ S : ℝ, S = 1/2 * a * b * Real.sin C → S ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1278_127804


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1278_127822

theorem sunzi_wood_measurement_problem (x y : ℝ) : 
  (x - y = 4.5 ∧ (x / 2) + 1 = y) ↔ (x - y = 4.5 ∧ y - x / 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_problem_l1278_127822


namespace NUMINAMATH_CALUDE_min_value_of_f_l1278_127864

/-- The quadratic function f(x) = x^2 + 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 36

/-- The minimum value of f(x) is 0 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1278_127864


namespace NUMINAMATH_CALUDE_deepak_age_l1278_127892

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 22 = 26 →
  deepak_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l1278_127892


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1278_127861

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The positive difference between two terms of an arithmetic sequence -/
def positiveDifference (a₁ : ℤ) (d : ℤ) (m n : ℕ) : ℕ :=
  (arithmeticSequenceTerm a₁ d m - arithmeticSequenceTerm a₁ d n).natAbs

theorem arithmetic_sequence_difference :
  positiveDifference (-8) 8 1020 1000 = 160 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1278_127861


namespace NUMINAMATH_CALUDE_ruel_stamps_count_l1278_127881

theorem ruel_stamps_count : ∀ (books_of_10 books_of_15 : ℕ),
  books_of_10 = 4 →
  books_of_15 = 6 →
  books_of_10 * 10 + books_of_15 * 15 = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_ruel_stamps_count_l1278_127881


namespace NUMINAMATH_CALUDE_least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l1278_127833

theorem least_divisible_by_1920 (a : ℕ) : a^6 % 1920 = 0 → a ≥ 8 :=
sorry

theorem eight_divisible_by_1920 : 8^6 % 1920 = 0 :=
sorry

theorem eight_is_least_divisible_by_1920 : ∃ (a : ℕ), a^6 % 1920 = 0 ∧ ∀ (b : ℕ), b < a → b^6 % 1920 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l1278_127833


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1278_127897

def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

theorem cistern_fill_time :
  let rate1 : ℚ := 1 / 10
  let rate2 : ℚ := 1 / 12
  let rate3 : ℚ := -1 / 25
  fill_time rate1 rate2 rate3 = 300 / 43 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1278_127897


namespace NUMINAMATH_CALUDE_connect_four_games_l1278_127818

/-- 
Given:
- The ratio of games Kaleb won to games he lost is 3:2
- Kaleb won 18 games

Prove: The total number of games played is 30
-/
theorem connect_four_games (games_won : ℕ) (games_lost : ℕ) : 
  (games_won : ℚ) / games_lost = 3 / 2 → 
  games_won = 18 → 
  games_won + games_lost = 30 := by
sorry

end NUMINAMATH_CALUDE_connect_four_games_l1278_127818


namespace NUMINAMATH_CALUDE_min_distance_sum_l1278_127811

open Complex

theorem min_distance_sum (z₁ z₂ : ℂ) (h₁ : z₁ = -Real.sqrt 3 - I) (h₂ : z₂ = 3 + Real.sqrt 3 * I) :
  (∃ (θ : ℝ), ∀ (z : ℂ), z = (2 + Real.cos θ) + I * Real.sin θ →
    ∀ (w : ℂ), abs (w - z₁) + abs (w - z₂) ≥ abs (z - z₁) + abs (z - z₂)) ∧
  (∃ (z : ℂ), abs (z - z₁) + abs (z - z₂) = 2 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_sum_l1278_127811


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1278_127868

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, ax^2 - 2*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1278_127868


namespace NUMINAMATH_CALUDE_betty_rice_purchase_l1278_127857

theorem betty_rice_purchase (o r : ℝ) : 
  (o ≥ 8 + r / 3 ∧ o ≤ 3 * r) → r ≥ 3 := by sorry

end NUMINAMATH_CALUDE_betty_rice_purchase_l1278_127857


namespace NUMINAMATH_CALUDE_age_difference_l1278_127852

/-- Given two people p and q, prove that p was half of q's age 6 years ago,
    given their current age ratio and sum. -/
theorem age_difference (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →  -- Current age ratio
  p + q = 21 →           -- Sum of current ages
  ∃ (y : ℕ), p - y = (q - y) / 2 ∧ y = 6 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l1278_127852


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l1278_127847

theorem least_number_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1056 + y) % 35 = 0 ∧ (1056 + y) % 51 = 0)) ∧
  ((1056 + x) % 35 = 0 ∧ (1056 + x) % 51 = 0) →
  x = 729 := by
sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l1278_127847


namespace NUMINAMATH_CALUDE_min_value_theorem_l1278_127848

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (4 * x) / (x + 3 * y) + (3 * y) / x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1278_127848


namespace NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1278_127850

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + (Real.pi^2/4)*(x^2 - 3*x + 9)

theorem g_range :
  ∀ y ∈ Set.range g, π^2/4 ≤ y ∧ y ≤ 37*π^2/4 :=
by sorry

theorem g_range_achieves_bounds :
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
    g x₁ = π^2/4 ∧ g x₂ = 37*π^2/4 :=
by sorry

end NUMINAMATH_CALUDE_g_range_g_range_achieves_bounds_l1278_127850


namespace NUMINAMATH_CALUDE_unique_solution_iff_a_eq_plus_minus_two_l1278_127817

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := x^2 + y^2 + z^2 + 4*y = 0
def equation2 (a x y z : ℝ) : Prop := x + a*y + a*z - a = 0

-- Define what it means for the system to have a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y z : ℝ), equation1 x y z ∧ equation2 a x y z

-- State the theorem
theorem unique_solution_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, has_unique_solution a ↔ (a = 2 ∨ a = -2) := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_a_eq_plus_minus_two_l1278_127817


namespace NUMINAMATH_CALUDE_function_derivative_at_two_l1278_127870

/-- Given a function f(x) = a*ln(x) + b/x where f(1) = -2 and f'(1) = 0, prove that f'(2) = -1/2 -/
theorem function_derivative_at_two 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x > 0, f x = a * Real.log x + b / x)
  (h2 : f 1 = -2)
  (h3 : deriv f 1 = 0) :
  deriv f 2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_function_derivative_at_two_l1278_127870


namespace NUMINAMATH_CALUDE_caitlin_bracelets_l1278_127823

/-- The number of bracelets Caitlin can make given the conditions -/
def num_bracelets : ℕ :=
  let total_beads : ℕ := 528
  let large_beads_per_bracelet : ℕ := 12
  let small_beads_per_bracelet : ℕ := 2 * large_beads_per_bracelet
  let large_beads : ℕ := total_beads / 2
  let small_beads : ℕ := total_beads / 2
  min (large_beads / large_beads_per_bracelet) (small_beads / small_beads_per_bracelet)

/-- Theorem stating that Caitlin can make 22 bracelets -/
theorem caitlin_bracelets : num_bracelets = 22 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_bracelets_l1278_127823


namespace NUMINAMATH_CALUDE_monotonicity_criterion_other_statements_incorrect_l1278_127886

/-- A function f is monotonically decreasing on ℝ if for all x₁ < x₂, f(x₁) > f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

theorem monotonicity_criterion (f : ℝ → ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ ≤ f x₂) → ¬(MonotonicallyDecreasing f) :=
by sorry

theorem other_statements_incorrect (f : ℝ → ℝ) :
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂ → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₂ > 0, (∀ x₁, f x₁ < f (x₁ + x₂)) → (∀ y₁ y₂, y₁ < y₂ → f y₁ < f y₂)) ∧
  ¬(∀ x₁ x₂, x₁ < x₂ → f x₁ ≥ f x₂ → MonotonicallyDecreasing f) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_criterion_other_statements_incorrect_l1278_127886


namespace NUMINAMATH_CALUDE_set_operation_result_l1278_127800

def A : Set Nat := {1, 2, 6}
def B : Set Nat := {2, 4}
def C : Set Nat := {1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l1278_127800


namespace NUMINAMATH_CALUDE_track_circumference_l1278_127893

/-- Represents the circular track and the movement of A and B -/
structure TrackSystem where
  /-- Half of the track's circumference in yards -/
  half_circumference : ℝ
  /-- Speed of A in yards per unit time -/
  speed_a : ℝ
  /-- Speed of B in yards per unit time -/
  speed_b : ℝ

/-- The theorem stating the conditions and the result to be proven -/
theorem track_circumference (ts : TrackSystem) 
  (h1 : ts.speed_a > 0 ∧ ts.speed_b > 0)  -- A and B travel at uniform (positive) speeds
  (h2 : ts.speed_a + ts.speed_b = ts.half_circumference / 75)  -- They meet after B travels 150 yards
  (h3 : 2 * ts.half_circumference - 90 = (ts.half_circumference + 90) * (ts.speed_a / ts.speed_b)) 
      -- Second meeting condition
  : ts.half_circumference = 360 :=
sorry

end NUMINAMATH_CALUDE_track_circumference_l1278_127893


namespace NUMINAMATH_CALUDE_line_decreasing_direct_proportion_range_l1278_127837

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = k * x₁ + b
  eq₂ : y₂ = k * x₂ + b

/-- A direct proportion function passing through two points -/
structure DirectProportion where
  m : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  eq₁ : y₁ = (1 - 2*m) * x₁
  eq₂ : y₂ = (1 - 2*m) * x₂

theorem line_decreasing (l : Line) (h₁ : l.k < 0) (h₂ : l.x₁ < l.x₂) : l.y₁ > l.y₂ := by
  sorry

theorem direct_proportion_range (d : DirectProportion) (h₁ : d.x₁ < d.x₂) (h₂ : d.y₁ > d.y₂) : d.m > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_decreasing_direct_proportion_range_l1278_127837


namespace NUMINAMATH_CALUDE_rob_doubles_l1278_127882

/-- Rob has some baseball cards, and Jess has 5 times as many doubles as Rob. 
    Jess has 40 doubles baseball cards. -/
theorem rob_doubles (rob_cards : ℕ) (rob_doubles : ℕ) (jess_doubles : ℕ) 
    (h1 : rob_cards ≥ rob_doubles)
    (h2 : jess_doubles = 5 * rob_doubles)
    (h3 : jess_doubles = 40) : 
  rob_doubles = 8 := by
  sorry

end NUMINAMATH_CALUDE_rob_doubles_l1278_127882


namespace NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l1278_127801

/-- Given two circles with radius 4 units centered at (0, 20) and (7, 13),
    and a line passing through (4, 0) that divides the total area of both circles equally,
    prove that the absolute value of the slope of this line is 33/15 -/
theorem equal_area_dividing_line_slope (r : ℝ) (c₁ c₂ : ℝ × ℝ) (p : ℝ × ℝ) (m : ℝ) :
  r = 4 →
  c₁ = (0, 20) →
  c₂ = (7, 13) →
  p = (4, 0) →
  (∀ x y, y = m * (x - p.1) + p.2) →
  (∀ x y, (x - c₁.1)^2 + (y - c₁.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  (∀ x y, (x - c₂.1)^2 + (y - c₂.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  abs m = 33 / 15 := by
sorry


end NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l1278_127801


namespace NUMINAMATH_CALUDE_turquoise_more_blue_count_l1278_127849

/-- Represents the results of a survey about the color turquoise -/
structure TurquoiseSurvey where
  total : ℕ
  more_green : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of people who believe turquoise is "more blue" -/
def more_blue (survey : TurquoiseSurvey) : ℕ :=
  survey.total - (survey.more_green - survey.both) - survey.neither - survey.both

/-- Theorem stating that in the given survey, 65 people believe turquoise is "more blue" -/
theorem turquoise_more_blue_count :
  ∃ (survey : TurquoiseSurvey),
    survey.total = 150 ∧
    survey.more_green = 95 ∧
    survey.both = 35 ∧
    survey.neither = 25 ∧
    more_blue survey = 65 := by
  sorry

#eval more_blue ⟨150, 95, 35, 25⟩

end NUMINAMATH_CALUDE_turquoise_more_blue_count_l1278_127849


namespace NUMINAMATH_CALUDE_smallest_positive_angle_neg_1050_l1278_127855

/-- The smallest positive angle (in degrees) with the same terminal side as a given angle -/
def smallestPositiveEquivalentAngle (angle : Int) : Int :=
  (angle % 360 + 360) % 360

/-- Theorem: The smallest positive angle with the same terminal side as -1050° is 30° -/
theorem smallest_positive_angle_neg_1050 :
  smallestPositiveEquivalentAngle (-1050) = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_neg_1050_l1278_127855


namespace NUMINAMATH_CALUDE_exponential_decreasing_implies_cubic_increasing_l1278_127873

theorem exponential_decreasing_implies_cubic_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x y : ℝ, x < y → a^x > a^y) :
  (∀ x y : ℝ, x < y → (2 - a) * x^3 < (2 - a) * y^3) ∧
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧
    (∀ x y : ℝ, x < y → (2 - b) * x^3 < (2 - b) * y^3) ∧
    ¬(∀ x y : ℝ, x < y → b^x > b^y)) :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_implies_cubic_increasing_l1278_127873


namespace NUMINAMATH_CALUDE_product_of_coefficients_l1278_127856

theorem product_of_coefficients (x y z w A B : ℝ) 
  (eq1 : 4 * x * z + y * w = 3)
  (eq2 : x * w + y * z = 6)
  (eq3 : (A * x + y) * (B * z + w) = 15) :
  A * B = 4 := by sorry

end NUMINAMATH_CALUDE_product_of_coefficients_l1278_127856


namespace NUMINAMATH_CALUDE_six_lines_intersection_possibilities_l1278_127805

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  lines : Finset Line
  intersections : Finset IntersectionPoint
  no_triple_intersections : ∀ p : IntersectionPoint, p ∈ intersections → 
    (∃! l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ p ∈ intersections)

theorem six_lines_intersection_possibilities 
  (config : LineConfiguration) 
  (h_six_lines : config.lines.card = 6) :
  (∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 12) ∧
  ¬(∃ config' : LineConfiguration, config'.lines = config.lines ∧ config'.intersections.card = 16) :=
sorry

end NUMINAMATH_CALUDE_six_lines_intersection_possibilities_l1278_127805


namespace NUMINAMATH_CALUDE_smallest_additional_divisor_l1278_127839

def divisors : Set Nat := {30, 48, 74, 100}

theorem smallest_additional_divisor :
  ∃ (n : Nat), n > 0 ∧ 
  (∀ m ∈ divisors, (44402 + 2) % m = 0) ∧
  (44402 + 2) % n = 0 ∧
  n ∉ divisors ∧
  (∀ k : Nat, 0 < k ∧ k < n → (44402 + 2) % k ≠ 0 ∨ k ∈ divisors) ∧
  n = 37 := by
  sorry

end NUMINAMATH_CALUDE_smallest_additional_divisor_l1278_127839


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1278_127842

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 200)
  (x_eq : 8 * x = n)
  (y_eq : y = n + 12)
  (z_eq : z = n - 12)
  (x_smallest : x < y ∧ x < z) : 
  x * y * z = 502147200 / 4913 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1278_127842


namespace NUMINAMATH_CALUDE_least_x_value_l1278_127835

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q ≠ 2 ∧ x = 12 * p * q) : x ≥ 72 := by
  sorry

end NUMINAMATH_CALUDE_least_x_value_l1278_127835


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_15_l1278_127828

theorem binomial_coefficient_16_15 : Nat.choose 16 15 = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_15_l1278_127828


namespace NUMINAMATH_CALUDE_min_luxury_owners_l1278_127838

structure Village where
  population : ℕ
  refrigerator_owners : Finset Nat
  television_owners : Finset Nat
  computer_owners : Finset Nat
  air_conditioner_owners : Finset Nat
  washing_machine_owners : Finset Nat
  microwave_owners : Finset Nat
  internet_owners : Finset Nat
  top_earners : Finset Nat

def Owlna (v : Village) : Prop :=
  v.refrigerator_owners.card = (67 * v.population) / 100 ∧
  v.television_owners.card = (74 * v.population) / 100 ∧
  v.computer_owners.card = (77 * v.population) / 100 ∧
  v.air_conditioner_owners.card = (83 * v.population) / 100 ∧
  v.washing_machine_owners.card = (55 * v.population) / 100 ∧
  v.microwave_owners.card = (48 * v.population) / 100 ∧
  v.internet_owners.card = (42 * v.population) / 100 ∧
  (v.television_owners ∩ v.computer_owners).card = (35 * v.population) / 100 ∧
  (v.washing_machine_owners ∩ v.microwave_owners).card = (30 * v.population) / 100 ∧
  (v.air_conditioner_owners ∩ v.refrigerator_owners).card = (27 * v.population) / 100 ∧
  v.top_earners.card = (10 * v.population) / 100 ∧
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners) ⊆ v.top_earners

theorem min_luxury_owners (v : Village) (h : Owlna v) :
  (v.refrigerator_owners ∩ v.television_owners ∩ v.computer_owners ∩
   v.air_conditioner_owners ∩ v.washing_machine_owners ∩ v.microwave_owners ∩
   v.internet_owners ∩ v.top_earners).card = (10 * v.population) / 100 :=
by sorry

end NUMINAMATH_CALUDE_min_luxury_owners_l1278_127838


namespace NUMINAMATH_CALUDE_diplomat_languages_l1278_127825

theorem diplomat_languages (total : ℕ) (french : ℕ) (not_russian : ℕ) (both_percent : ℚ) 
  (h_total : total = 180)
  (h_french : french = 14)
  (h_not_russian : not_russian = 32)
  (h_both_percent : both_percent = 1/10) : 
  (total - (french + (total - not_russian) - (both_percent * total))) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diplomat_languages_l1278_127825


namespace NUMINAMATH_CALUDE_four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l1278_127814

-- Define the types for points and lines in space
variable (Point Line : Type)

-- Define the properties
variable (coplanar : Point → Point → Point → Point → Prop)
variable (collinear : Point → Point → Point → Prop)
variable (have_common_point : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Theorem 1
theorem four_points_not_coplanar_iff_any_three_not_collinear 
  (p1 p2 p3 p4 : Point) : 
  ¬(coplanar p1 p2 p3 p4) ↔ 
  (¬(collinear p1 p2 p3) ∧ ¬(collinear p1 p2 p4) ∧ 
   ¬(collinear p1 p3 p4) ∧ ¬(collinear p2 p3 p4)) :=
sorry

-- Theorem 2
theorem lines_no_common_point_iff_skew (l1 l2 : Line) :
  ¬(have_common_point l1 l2) ↔ skew l1 l2 :=
sorry

end NUMINAMATH_CALUDE_four_points_not_coplanar_iff_any_three_not_collinear_lines_no_common_point_iff_skew_l1278_127814


namespace NUMINAMATH_CALUDE_book_cost_l1278_127802

theorem book_cost : ∃ (x : ℝ), x = 1 + (1/2) * x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_book_cost_l1278_127802


namespace NUMINAMATH_CALUDE_min_chord_length_implies_m_l1278_127853

/-- Given a circle C: x^2 + y^2 = 4 and a line l: y = kx + m, 
    prove that if the minimum chord length cut by l on C is 2, then m = ±√3 -/
theorem min_chord_length_implies_m (k : ℝ) :
  (∀ x y, x^2 + y^2 = 4 → ∃ m, y = k*x + m) →
  (∃ m, ∀ x y, x^2 + y^2 = 4 ∧ y = k*x + m → 
    ∀ x1 y1 x2 y2, x1^2 + y1^2 = 4 ∧ y1 = k*x1 + m ∧ 
                   x2^2 + y2^2 = 4 ∧ y2 = k*x2 + m →
    (x1 - x2)^2 + (y1 - y2)^2 ≥ 4) →
  ∃ m, m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_chord_length_implies_m_l1278_127853


namespace NUMINAMATH_CALUDE_sum_three_consecutive_divisible_by_three_l1278_127872

theorem sum_three_consecutive_divisible_by_three (n : ℕ) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_divisible_by_three_l1278_127872


namespace NUMINAMATH_CALUDE_right_triangle_sets_l1278_127840

theorem right_triangle_sets : 
  (1^2 + Real.sqrt 2^2 = Real.sqrt 3^2) ∧ 
  (3^2 + 4^2 = 5^2) ∧ 
  (9^2 + 12^2 = 15^2) ∧ 
  (4^2 + 5^2 ≠ 6^2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l1278_127840


namespace NUMINAMATH_CALUDE_square_difference_equality_l1278_127895

theorem square_difference_equality : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1278_127895


namespace NUMINAMATH_CALUDE_calculation_proof_l1278_127830

theorem calculation_proof : (2013 : ℚ) / (25 * 52 - 46 * 15) * 10 = 33 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1278_127830


namespace NUMINAMATH_CALUDE_binary_to_decimal_110101_l1278_127880

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_110101_l1278_127880


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_l1278_127829

/-- Given points A and B, prove the equation of the circle with AB as diameter -/
theorem circle_equation_with_diameter (A B : ℝ × ℝ) (h : A = (-4, 0) ∧ B = (0, 2)) :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 5 ↔ 
  (x - A.1)^2 + (y - A.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 ∧
  (x - B.1)^2 + (y - B.2)^2 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_l1278_127829


namespace NUMINAMATH_CALUDE_first_watermelon_weight_l1278_127865

theorem first_watermelon_weight (total_weight second_weight : ℝ) 
  (h1 : total_weight = 14.02)
  (h2 : second_weight = 4.11) :
  total_weight - second_weight = 9.91 := by
  sorry

end NUMINAMATH_CALUDE_first_watermelon_weight_l1278_127865


namespace NUMINAMATH_CALUDE_peters_remaining_money_l1278_127843

/-- Represents Peter's shopping trips and calculates his remaining money -/
def petersShopping (initialAmount : ℚ) : ℚ :=
  let firstTripTax := 0.05
  let secondTripDiscount := 0.1

  let firstTripItems := [
    (6, 2),    -- potatoes
    (9, 3),    -- tomatoes
    (5, 4),    -- cucumbers
    (3, 5),    -- bananas
    (2, 3.5),  -- apples
    (7, 4.25), -- oranges
    (4, 6),    -- grapes
    (8, 5.5)   -- strawberries
  ]

  let secondTripItems := [
    (2, 1.5),  -- potatoes
    (5, 2.75)  -- tomatoes
  ]

  let firstTripCost := (firstTripItems.map (λ (k, p) => k * p)).sum * (1 + firstTripTax)
  let secondTripCost := (secondTripItems.map (λ (k, p) => k * p)).sum * (1 - secondTripDiscount)

  initialAmount - firstTripCost - secondTripCost

/-- Theorem stating that Peter's remaining money is $297.24 -/
theorem peters_remaining_money :
  petersShopping 500 = 297.24 := by
  sorry


end NUMINAMATH_CALUDE_peters_remaining_money_l1278_127843


namespace NUMINAMATH_CALUDE_expression_evaluation_l1278_127877

theorem expression_evaluation : 2 + (0 * 2^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1278_127877


namespace NUMINAMATH_CALUDE_niko_sock_profit_l1278_127836

theorem niko_sock_profit : ∀ (total_pairs : ℕ) (cost_per_pair : ℚ) 
  (profit_percent : ℚ) (profit_amount : ℚ) (high_profit_pairs : ℕ) (low_profit_pairs : ℕ),
  total_pairs = 9 →
  cost_per_pair = 2 →
  profit_percent = 25 / 100 →
  profit_amount = 1 / 5 →
  high_profit_pairs = 4 →
  low_profit_pairs = 5 →
  high_profit_pairs + low_profit_pairs = total_pairs →
  (high_profit_pairs : ℚ) * (cost_per_pair * profit_percent) + 
  (low_profit_pairs : ℚ) * profit_amount = 3 := by
sorry

end NUMINAMATH_CALUDE_niko_sock_profit_l1278_127836


namespace NUMINAMATH_CALUDE_complex_root_sum_square_l1278_127860

theorem complex_root_sum_square (p q : ℝ) : 
  (6 * (p + q * I)^3 + 5 * (p + q * I)^2 - (p + q * I) + 14 = 0) →
  (6 * (p - q * I)^3 + 5 * (p - q * I)^2 - (p - q * I) + 14 = 0) →
  p + q^2 = 21/4 := by
sorry

end NUMINAMATH_CALUDE_complex_root_sum_square_l1278_127860


namespace NUMINAMATH_CALUDE_complex_multiplication_l1278_127874

theorem complex_multiplication (i : ℂ) : i^2 = -1 → i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1278_127874


namespace NUMINAMATH_CALUDE_pizza_segment_length_squared_l1278_127879

theorem pizza_segment_length_squared (diameter : ℝ) (num_pieces : ℕ) (m : ℝ) : 
  diameter = 18 →
  num_pieces = 4 →
  m = 2 * (diameter / 2) * Real.sin (π / (2 * num_pieces)) →
  m^2 = 162 := by sorry

end NUMINAMATH_CALUDE_pizza_segment_length_squared_l1278_127879


namespace NUMINAMATH_CALUDE_tobys_friends_l1278_127884

theorem tobys_friends (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  girls = 27 →
  total = boys + girls →
  boys = 33 := by
sorry

end NUMINAMATH_CALUDE_tobys_friends_l1278_127884


namespace NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l1278_127867

def num_meals : ℕ := 4
def num_fruits : ℕ := 4

def prob_same_fruit_all_day : ℚ := (1 / num_fruits) ^ num_meals * num_fruits

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit_all_day = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_different_fruits_l1278_127867


namespace NUMINAMATH_CALUDE_jim_distance_driven_l1278_127845

/-- The distance Jim has driven so far in his journey -/
def distance_driven (total_journey : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_journey - remaining_distance

/-- Theorem stating that Jim has driven 215 miles -/
theorem jim_distance_driven :
  distance_driven 1200 985 = 215 := by
  sorry

end NUMINAMATH_CALUDE_jim_distance_driven_l1278_127845


namespace NUMINAMATH_CALUDE_surface_area_after_corner_removal_l1278_127810

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the problem setup -/
structure ProblemSetup where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions
  numCorners : ℕ

/-- Theorem stating that the surface area remains unchanged after removing corner cubes -/
theorem surface_area_after_corner_removal (p : ProblemSetup) 
  (h1 : p.originalCube.length = 4)
  (h2 : p.originalCube.width = 4)
  (h3 : p.originalCube.height = 4)
  (h4 : p.cornerCube.length = 2)
  (h5 : p.cornerCube.width = 2)
  (h6 : p.cornerCube.height = 2)
  (h7 : p.numCorners = 8) :
  surfaceArea p.originalCube = 96 := by
  sorry

#eval surfaceArea { length := 4, width := 4, height := 4 }

end NUMINAMATH_CALUDE_surface_area_after_corner_removal_l1278_127810


namespace NUMINAMATH_CALUDE_solution_equation_l1278_127898

theorem solution_equation : ∃ x : ℝ, 0.4 * x + (0.3 * 0.2) = 0.26 ∧ x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_solution_equation_l1278_127898


namespace NUMINAMATH_CALUDE_flu_infection_model_l1278_127813

/-- 
Given two rounds of flu infection where:
- In each round, on average, one person infects x people
- After two rounds, a total of 144 people had the flu

This theorem states that the equation (1+x)^2 = 144 correctly models 
the total number of people infected after two rounds.
-/
theorem flu_infection_model (x : ℝ) : 
  (∃ (infected_first_round infected_second_round : ℕ),
    infected_first_round = x ∧ 
    infected_second_round = x * infected_first_round ∧
    1 + infected_first_round + infected_second_round = 144) ↔ 
  (1 + x)^2 = 144 :=
sorry

end NUMINAMATH_CALUDE_flu_infection_model_l1278_127813


namespace NUMINAMATH_CALUDE_survey_result_l1278_127896

theorem survey_result : ∀ (total : ℕ) (dangerous : ℕ) (fire : ℕ),
  (dangerous : ℚ) / total = 825 / 1000 →
  (fire : ℚ) / dangerous = 524 / 1000 →
  fire = 27 →
  total = 63 := by
sorry

end NUMINAMATH_CALUDE_survey_result_l1278_127896


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1278_127819

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_first : a 0 = 2^(1/2 : ℝ))
  (h_second : a 1 = 2^(1/4 : ℝ))
  (h_third : a 2 = 2^(1/8 : ℝ)) :
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l1278_127819


namespace NUMINAMATH_CALUDE_shareholders_profit_decrease_l1278_127859

/-- Represents the problem of calculating the percentage decrease in shareholders' profit --/
theorem shareholders_profit_decrease (total_machines : ℝ) (operational_machines : ℝ) 
  (annual_output : ℝ) (profit_percentage : ℝ) :
  total_machines = 14 →
  operational_machines = total_machines - 7.14 →
  annual_output = 70000 →
  profit_percentage = 0.125 →
  let new_output := (operational_machines / total_machines) * annual_output
  let original_profit := profit_percentage * annual_output
  let new_profit := profit_percentage * new_output
  let percentage_decrease := ((original_profit - new_profit) / original_profit) * 100
  percentage_decrease = 51 := by
sorry

end NUMINAMATH_CALUDE_shareholders_profit_decrease_l1278_127859


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l1278_127862

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l1278_127862


namespace NUMINAMATH_CALUDE_correct_guess_probability_l1278_127803

/-- The number of possible choices for the last digit -/
def last_digit_choices : ℕ := 4

/-- The number of possible choices for the second-to-last digit -/
def second_last_digit_choices : ℕ := 3

/-- The probability of correctly guessing the two-digit code -/
def guess_probability : ℚ := 1 / (last_digit_choices * second_last_digit_choices)

theorem correct_guess_probability : guess_probability = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_guess_probability_l1278_127803


namespace NUMINAMATH_CALUDE_jerry_action_figures_l1278_127885

theorem jerry_action_figures
  (complete_collection : ℕ)
  (cost_per_figure : ℕ)
  (total_cost_to_complete : ℕ)
  (h1 : complete_collection = 16)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost_to_complete = 72) :
  complete_collection - (total_cost_to_complete / cost_per_figure) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l1278_127885


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l1278_127887

theorem triangle_side_and_area 
  (A B C : Real) -- Angles
  (a b c : Real) -- Sides
  (h1 : b = Real.sqrt 7)
  (h2 : c = 1)
  (h3 : B = 2 * π / 3) -- 120° in radians
  (h4 : 0 < a ∧ 0 < b ∧ 0 < c) -- Triangle inequality
  (h5 : b^2 = a^2 + c^2 - 2*a*c*Real.cos B) -- Cosine rule
  : a = 2 ∧ (1/2) * a * c * Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_and_area_l1278_127887


namespace NUMINAMATH_CALUDE_ratio_problem_l1278_127809

theorem ratio_problem (a b c d : ℝ) (h1 : b / a = 4) (h2 : d / c = 2) : (a + b) / (c + d) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1278_127809


namespace NUMINAMATH_CALUDE_min_sum_position_l1278_127806

/-- An arithmetic sequence {a_n} with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

/-- The theorem stating when S_n reaches its minimum value -/
theorem min_sum_position (seq : ArithmeticSequence) 
  (h1 : seq.a 2 = -2)
  (h2 : seq.S 4 = -4) :
  ∃ n : ℕ, (n = 2 ∨ n = 3) ∧ 
    (∀ m : ℕ, m ≥ 1 → seq.S n ≤ seq.S m) :=
sorry

end NUMINAMATH_CALUDE_min_sum_position_l1278_127806


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l1278_127831

theorem cubic_polynomial_root (x : ℝ) : x = Real.rpow 4 (1/3) + 2 → x^3 - 6*x^2 + 12*x - 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l1278_127831


namespace NUMINAMATH_CALUDE_line_for_equal_diagonals_l1278_127876

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 9

-- Define the line l passing through (-1, 0)
def l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersectionPoints (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂

-- Define vector OS as the sum of OA and OB
def vectorOS (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x = x₁ + x₂ ∧ y = y₁ + y₂

-- Define the condition for equal diagonals in quadrilateral OASB
def equalDiagonals (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), C x₁ y₁ ∧ C x₂ y₂ ∧ l k x₁ y₁ ∧ l k x₂ y₂ ∧
    x₁^2 + y₁^2 = x₂^2 + y₂^2

-- Theorem statement
theorem line_for_equal_diagonals :
  ∃! k, intersectionPoints k ∧ equalDiagonals k ∧ k = 1 :=
sorry

end NUMINAMATH_CALUDE_line_for_equal_diagonals_l1278_127876


namespace NUMINAMATH_CALUDE_apple_cost_is_14_l1278_127807

/-- Represents the cost of groceries in dollars -/
structure GroceryCost where
  total : ℕ
  bananas : ℕ
  bread : ℕ
  milk : ℕ

/-- Calculates the cost of apples given the total cost and costs of other items -/
def appleCost (g : GroceryCost) : ℕ :=
  g.total - (g.bananas + g.bread + g.milk)

/-- Theorem stating that the cost of apples is 14 dollars given the specific grocery costs -/
theorem apple_cost_is_14 (g : GroceryCost) 
    (h1 : g.total = 42)
    (h2 : g.bananas = 12)
    (h3 : g.bread = 9)
    (h4 : g.milk = 7) : 
  appleCost g = 14 := by
  sorry

#eval appleCost { total := 42, bananas := 12, bread := 9, milk := 7 }

end NUMINAMATH_CALUDE_apple_cost_is_14_l1278_127807


namespace NUMINAMATH_CALUDE_x_one_value_l1278_127834

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + (x₃-x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_x_one_value_l1278_127834


namespace NUMINAMATH_CALUDE_sum_of_complex_numbers_l1278_127832

theorem sum_of_complex_numbers : 
  (2 : ℂ) + 5*I + (3 : ℂ) - 7*I + (-1 : ℂ) + 2*I = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_complex_numbers_l1278_127832


namespace NUMINAMATH_CALUDE_vertical_bisecting_line_of_circles_l1278_127854

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 6*y + 2 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y - 4 = 0

-- Define the vertical bisecting line
def bisecting_line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- Theorem statement
theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ), (circle1 x y ∧ circle2 x y) → bisecting_line x y :=
by sorry

end NUMINAMATH_CALUDE_vertical_bisecting_line_of_circles_l1278_127854


namespace NUMINAMATH_CALUDE_intersection_points_10_5_l1278_127808

/-- The number of intersection points formed by line segments connecting points on x and y axes -/
def intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating that 10 points on x-axis and 5 points on y-axis result in 450 intersection points -/
theorem intersection_points_10_5 :
  intersection_points 10 5 = 450 := by sorry

end NUMINAMATH_CALUDE_intersection_points_10_5_l1278_127808


namespace NUMINAMATH_CALUDE_common_difference_is_three_l1278_127889

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem common_difference_is_three
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum1 : a 2 + a 3 = 9)
  (h_sum2 : a 4 + a 5 = 21) :
  ∃ d, d = 3 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l1278_127889


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_m_l1278_127858

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_purely_imaginary_m (m : ℝ) :
  is_purely_imaginary ((m^2 - m) + m * I) → m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_m_l1278_127858


namespace NUMINAMATH_CALUDE_day_crew_load_fraction_l1278_127866

theorem day_crew_load_fraction (D : ℝ) (W_d : ℝ) (W_d_pos : W_d > 0) : 
  let night_boxes_per_worker := (1 / 2) * D
  let night_workers := (4 / 5) * W_d
  let day_total := D * W_d
  let night_total := night_boxes_per_worker * night_workers
  (day_total) / (day_total + night_total) = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_day_crew_load_fraction_l1278_127866


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l1278_127824

/-- Given that z varies inversely as √w, prove that w = 64 when z = 2, 
    given that z = 8 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w : ℝ, w > 0 → z * Real.sqrt w = k) 
    (h1 : 8 * Real.sqrt 4 = z * Real.sqrt w) : 
    z = 2 → w = 64 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l1278_127824


namespace NUMINAMATH_CALUDE_distance_minus_nine_to_nine_l1278_127812

-- Define the distance function for points on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem distance_minus_nine_to_nine : distance (-9) 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_minus_nine_to_nine_l1278_127812


namespace NUMINAMATH_CALUDE_tan_75_degrees_l1278_127894

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l1278_127894


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1278_127846

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (4 * x - 8 ≤ 0) ∧ ((x + 3) / 2 > 3 - x)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  1 < x ∧ x ≤ 2

-- Theorem statement
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1278_127846


namespace NUMINAMATH_CALUDE_tyler_remaining_money_l1278_127815

/-- Calculates the remaining money after Tyler's purchases -/
def remaining_money (initial_amount : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) 
                    (eraser_count : ℕ) (eraser_price : ℕ) : ℕ :=
  initial_amount - (scissors_count * scissors_price + eraser_count * eraser_price)

/-- Theorem stating that Tyler will have $20 remaining after his purchases -/
theorem tyler_remaining_money : 
  remaining_money 100 8 5 10 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tyler_remaining_money_l1278_127815


namespace NUMINAMATH_CALUDE_office_population_l1278_127826

theorem office_population (men women : ℕ) : 
  men = women →
  6 = women / 5 →
  men + women = 60 := by
sorry

end NUMINAMATH_CALUDE_office_population_l1278_127826
