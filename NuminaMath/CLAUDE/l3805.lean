import Mathlib

namespace NUMINAMATH_CALUDE_natalia_clip_sales_l3805_380574

/-- The number of clips Natalia sold in April and May combined -/
def total_clips (april_sales : ℕ) (may_sales : ℕ) : ℕ :=
  april_sales + may_sales

/-- Theorem stating that given the conditions, Natalia sold 72 clips in total -/
theorem natalia_clip_sales : 
  ∀ (april_sales : ℕ) (may_sales : ℕ),
    april_sales = 48 →
    may_sales = april_sales / 2 →
    total_clips april_sales may_sales = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_natalia_clip_sales_l3805_380574


namespace NUMINAMATH_CALUDE_two_colored_cubes_count_l3805_380504

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Represents a hollow cube with outer and inner dimensions -/
structure HollowCube where
  outer : Cube
  inner : Cube

/-- Calculates the number of smaller cubes with paint on exactly two sides -/
def cubesWithTwoColoredSides (hc : HollowCube) (smallCubeSide : ℕ) : ℕ :=
  12 * (hc.outer.side / smallCubeSide - 2)

theorem two_colored_cubes_count 
  (bigCube : Cube)
  (smallCube : Cube)
  (tinyCube : Cube)
  (hc : HollowCube) :
  bigCube.side = 27 →
  smallCube.side = 9 →
  tinyCube.side = 3 →
  hc.outer = bigCube →
  hc.inner = smallCube →
  cubesWithTwoColoredSides hc tinyCube.side = 84 := by
  sorry

#check two_colored_cubes_count

end NUMINAMATH_CALUDE_two_colored_cubes_count_l3805_380504


namespace NUMINAMATH_CALUDE_min_teachers_cover_all_subjects_l3805_380548

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- The maximum number of subjects a teacher can teach -/
def maxSubjectsPerTeacher : Nat := 3

/-- The total number of subjects -/
def totalSubjects : Nat := 3

/-- Given the number of teachers for each subject, calculates the minimum number
    of teachers required to cover all subjects -/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  sorry

theorem min_teachers_cover_all_subjects (counts : TeacherCounts) :
  counts.maths = 7 →
  counts.physics = 6 →
  counts.chemistry = 5 →
  minTeachersRequired counts = 7 :=
sorry

end NUMINAMATH_CALUDE_min_teachers_cover_all_subjects_l3805_380548


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_i_l3805_380515

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_equals_i :
  i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_i_l3805_380515


namespace NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3805_380571

theorem sqrt_three_times_sqrt_twelve : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_sqrt_twelve_l3805_380571


namespace NUMINAMATH_CALUDE_unique_point_for_equal_angles_l3805_380549

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the point P
def P : ℝ × ℝ := (4, 0)

-- Define a chord passing through the focus
def chord (a b : ℝ × ℝ) : Prop :=
  a.1 ≠ b.1 ∨ a.2 ≠ b.2  -- Ensure A and B are distinct points
  ∧ ellipse a.1 a.2      -- A is on the ellipse
  ∧ ellipse b.1 b.2      -- B is on the ellipse
  ∧ (b.2 - a.2) * (a.1 - 1) = (b.1 - a.1) * (a.2 - 0)  -- AB passes through F(1,0)

-- Define the equality of angles APF and BPF
def equal_angles (a b : ℝ × ℝ) : Prop :=
  (a.2 - 0) * (b.1 - 4) = (b.2 - 0) * (a.1 - 4)

theorem unique_point_for_equal_angles :
  ∀ a b : ℝ × ℝ, chord a b → equal_angles a b ∧
  ∀ p : ℝ, p > 0 ∧ p ≠ 4 →
    ∃ c d : ℝ × ℝ, chord c d ∧ ¬(c.2 - 0) * (d.1 - p) = (d.2 - 0) * (c.1 - p) :=
sorry

end NUMINAMATH_CALUDE_unique_point_for_equal_angles_l3805_380549


namespace NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3805_380556

/-- Represents the position of a particle as a pair of integers -/
def Position := ℤ × ℤ

/-- Represents the movement pattern of the particle -/
inductive MovementPattern
| OddSequence
| EvenSequence

/-- Calculates the next position based on the current position, movement pattern, and side length -/
def nextPosition (pos : Position) (pattern : MovementPattern) (side : ℕ) : Position :=
  match pattern with
  | MovementPattern.OddSequence => (pos.1 - side, pos.2 - side)
  | MovementPattern.EvenSequence => (pos.1 + side, pos.2 + side)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- Theorem stating that the particle's position after 2023 minutes is (-43, -43) -/
theorem particle_position_after_2023_minutes :
  particlePosition 2023 = (-43, -43) :=
  sorry

end NUMINAMATH_CALUDE_particle_position_after_2023_minutes_l3805_380556


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l3805_380564

/-- Triangle with side lengths -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle -/
structure InscribedRectangle (t : Triangle) where
  area : ℝ → ℝ

/-- The theorem stating the value of β in the area formula of the inscribed rectangle -/
theorem inscribed_rectangle_area_coefficient (t : Triangle) 
  (r : InscribedRectangle t) : 
  t.a = 12 → t.b = 25 → t.c = 17 → 
  (∃ α β : ℝ, ∀ ω, r.area ω = α * ω - β * ω^2) → 
  (∃ β : ℝ, β = 36 / 125) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l3805_380564


namespace NUMINAMATH_CALUDE_dick_jane_age_problem_l3805_380524

theorem dick_jane_age_problem :
  ∃ (d n : ℕ), 
    d > 27 ∧
    10 ≤ 27 + n ∧ 27 + n ≤ 99 ∧
    10 ≤ d + n ∧ d + n ≤ 99 ∧
    ∃ (a b : ℕ), 
      27 + n = 10 * a + b ∧
      d + n = 10 * b + a ∧
      Nat.Prime (a + b) ∧
      1 ≤ a ∧ a < b ∧ b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_dick_jane_age_problem_l3805_380524


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l3805_380525

theorem units_digit_of_sum_of_powers : (47^4 + 28^4) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_l3805_380525


namespace NUMINAMATH_CALUDE_bathroom_size_is_150_l3805_380557

/-- Represents the size and cost of a modular home --/
structure ModularHome where
  totalSize : Nat
  kitchenSize : Nat
  kitchenCost : Nat
  bathroomCount : Nat
  bathroomCost : Nat
  otherCost : Nat
  totalCost : Nat

/-- Calculates the size of the bathroom module --/
def bathroomSize (home : ModularHome) : Nat :=
  (home.totalSize - home.kitchenSize - 
   (home.totalCost - home.kitchenCost - home.bathroomCount * home.bathroomCost) / home.otherCost) / 
  (2 * home.bathroomCount)

/-- Theorem stating that the bathroom size is 150 square feet --/
theorem bathroom_size_is_150 (home : ModularHome) 
  (h1 : home.totalSize = 2000)
  (h2 : home.kitchenSize = 400)
  (h3 : home.kitchenCost = 20000)
  (h4 : home.bathroomCount = 2)
  (h5 : home.bathroomCost = 12000)
  (h6 : home.otherCost = 100)
  (h7 : home.totalCost = 174000) :
  bathroomSize home = 150 := by
  sorry

#eval bathroomSize {
  totalSize := 2000,
  kitchenSize := 400,
  kitchenCost := 20000,
  bathroomCount := 2,
  bathroomCost := 12000,
  otherCost := 100,
  totalCost := 174000
}

end NUMINAMATH_CALUDE_bathroom_size_is_150_l3805_380557


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3805_380520

def f (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_function_properties :
  ∀ b c d : ℝ,
  f 0 b c d = 2 →
  (∀ y : ℝ, 6*(-1) - y + 7 = 0 ↔ y = f (-1) b c d) →
  (∀ x : ℝ, f x b c d = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ x : ℝ, x < 1 - Real.sqrt 2 ∨ x > 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d > f x b c d) ∧
  (∀ x : ℝ, 1 - Real.sqrt 2 < x ∧ x < 1 + Real.sqrt 2 → 
    ∀ h : ℝ, h > 0 → f (x + h) b c d < f x b c d) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3805_380520


namespace NUMINAMATH_CALUDE_intersection_complement_equals_l3805_380596

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {1, 3, 6}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem to prove
theorem intersection_complement_equals : A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_l3805_380596


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l3805_380517

def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6)

theorem f_derivative_at_zero : 
  deriv f 0 = 720 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l3805_380517


namespace NUMINAMATH_CALUDE_sallys_raise_l3805_380554

/-- Given Sally's earnings last month and the total for two months, calculate her percentage raise. -/
theorem sallys_raise (last_month : ℝ) (total_two_months : ℝ) : 
  last_month = 1000 → total_two_months = 2100 → 
  (total_two_months - last_month) / last_month * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sallys_raise_l3805_380554


namespace NUMINAMATH_CALUDE_number_of_pupils_l3805_380562

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l3805_380562


namespace NUMINAMATH_CALUDE_savings_problem_l3805_380531

theorem savings_problem (a b : ℕ) : 
  a = 5 * b ∧ 
  (b + 60) = 2 * (a - 60) → 
  a = 100 ∧ b = 20 := by
sorry

end NUMINAMATH_CALUDE_savings_problem_l3805_380531


namespace NUMINAMATH_CALUDE_smallest_m_inequality_l3805_380537

theorem smallest_m_inequality (a b c : ℝ) :
  ∃ (M : ℝ), M = 9 / (16 * Real.sqrt 2) ∧
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ (M' : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ M' * (x^2 + y^2 + z^2)^2) → M ≤ M' :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_inequality_l3805_380537


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3805_380534

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3805_380534


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3805_380565

theorem smallest_m_for_integral_solutions : 
  let f (m : ℕ) (x : ℤ) := 12 * x^2 - m * x + 360
  ∃ (m₀ : ℕ), m₀ > 0 ∧ 
    (∃ (x : ℤ), f m₀ x = 0) ∧ 
    (∀ (m : ℕ), 0 < m ∧ m < m₀ → ∀ (x : ℤ), f m x ≠ 0) ∧
    m₀ = 132 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l3805_380565


namespace NUMINAMATH_CALUDE_expression_value_l3805_380501

theorem expression_value (x y : ℝ) (h1 : x = 3 * y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = 3^(-2 * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3805_380501


namespace NUMINAMATH_CALUDE_article_pricing_l3805_380582

/-- The value of x when the cost price of 20 articles equals the selling price of x articles with a 25% profit -/
theorem article_pricing (C : ℝ) (x : ℝ) (h1 : C > 0) :
  20 * C = x * (C * (1 + 0.25)) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_article_pricing_l3805_380582


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3805_380540

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
  Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) ≥ (5 : ℝ) / 2 ∧
  ((a / (b + c)) + (b / (c + a)) + (c / (a + b)) + 
   Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2)) = (5 : ℝ) / 2 ↔ 
   a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3805_380540


namespace NUMINAMATH_CALUDE_min_turn_angles_sum_l3805_380526

/-- Represents a broken line path in a circular arena -/
structure BrokenLinePath where
  /-- Radius of the circular arena in meters -/
  arena_radius : ℝ
  /-- Total length of the path in meters -/
  total_length : ℝ
  /-- List of angles between consecutive segments in radians -/
  turn_angles : List ℝ

/-- Theorem: The sum of turn angles in a broken line path is at least 2998 radians
    given the specified arena radius and path length -/
theorem min_turn_angles_sum (path : BrokenLinePath)
    (h_radius : path.arena_radius = 10)
    (h_length : path.total_length = 30000) :
    (path.turn_angles.sum ≥ 2998) := by
  sorry


end NUMINAMATH_CALUDE_min_turn_angles_sum_l3805_380526


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3805_380570

def set_A : Set ℝ := {x | Real.sqrt (x + 1) < 2}
def set_B : Set ℝ := {x | 1 < x ∧ x < 4}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3805_380570


namespace NUMINAMATH_CALUDE_seed_mixture_weights_l3805_380561

/-- Represents a seed mixture with percentages of different grass types -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ
  clover : ℝ
  sum_to_100 : ryegrass + bluegrass + fescue + clover = 100

/-- The final mixture of seeds -/
def FinalMixture (x y z : ℝ) : SeedMixture :=
  { ryegrass := 35
    bluegrass := 30
    fescue := 35
    clover := 0
    sum_to_100 := by norm_num }

/-- The seed mixtures X, Y, and Z -/
def X : SeedMixture :=
  { ryegrass := 40
    bluegrass := 50
    fescue := 0
    clover := 10
    sum_to_100 := by norm_num }

def Y : SeedMixture :=
  { ryegrass := 25
    bluegrass := 0
    fescue := 70
    clover := 5
    sum_to_100 := by norm_num }

def Z : SeedMixture :=
  { ryegrass := 30
    bluegrass := 20
    fescue := 50
    clover := 0
    sum_to_100 := by norm_num }

/-- The theorem stating the weights of seed mixtures X, Y, and Z in the final mixture -/
theorem seed_mixture_weights (x y z : ℝ) 
  (h_total : x + y + z = 8)
  (h_ratio : x / 3 = y / 2 ∧ x / 3 = z / 3)
  (h_final : FinalMixture x y z = 
    { ryegrass := (X.ryegrass * x + Y.ryegrass * y + Z.ryegrass * z) / 8
      bluegrass := (X.bluegrass * x + Y.bluegrass * y + Z.bluegrass * z) / 8
      fescue := (X.fescue * x + Y.fescue * y + Z.fescue * z) / 8
      clover := (X.clover * x + Y.clover * y + Z.clover * z) / 8
      sum_to_100 := sorry }) :
  x = 3 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_seed_mixture_weights_l3805_380561


namespace NUMINAMATH_CALUDE_minimum_score_for_average_l3805_380535

/-- Given five tests with a maximum score of 100 points each, prove that the minimum score
    needed in one of the remaining two tests to achieve an average score of 81 over all five
    tests is 48, given that the first three test scores are 76, 94, and 87. -/
theorem minimum_score_for_average (test1 test2 test3 test4 test5 : ℕ) : 
  test1 = 76 → test2 = 94 → test3 = 87 →
  test1 ≤ 100 → test2 ≤ 100 → test3 ≤ 100 → test4 ≤ 100 → test5 ≤ 100 →
  (test1 + test2 + test3 + test4 + test5) / 5 = 81 →
  (test4 = 100 ∧ test5 = 48) ∨ (test4 = 48 ∧ test5 = 100) :=
by sorry

end NUMINAMATH_CALUDE_minimum_score_for_average_l3805_380535


namespace NUMINAMATH_CALUDE_pool_filling_rate_prove_pool_filling_rate_l3805_380551

/-- Proves that the rate of filling the pool during the second and third hours is 10 gallons/hour -/
theorem pool_filling_rate : ℝ → Prop :=
  fun (R : ℝ) ↦
    (8 : ℝ) +         -- Water added in 1st hour
    (R * 2) +         -- Water added in 2nd and 3rd hours
    (14 : ℝ) -        -- Water added in 4th hour
    (8 : ℝ) =         -- Water lost in 5th hour
    (34 : ℝ) →        -- Total water after 5 hours
    R = (10 : ℝ)      -- Rate during 2nd and 3rd hours

/-- Proof of the theorem -/
theorem prove_pool_filling_rate : pool_filling_rate (10 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_rate_prove_pool_filling_rate_l3805_380551


namespace NUMINAMATH_CALUDE_triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l3805_380577

/-- A triangle with two altitudes greater than 100 has an area greater than 1 -/
theorem triangle_area_lower_bound (a b c h1 h2 : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) (halt : h1 > 100 ∧ h2 > 100) : 
  (1/2) * a * h1 > 1 := by
  sorry

/-- There does not exist a triangle with two altitudes greater than 100 and area less than 1 -/
theorem no_triangle_large_altitudes_small_area : 
  ¬ ∃ (a b c h1 h2 : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧ 
  h1 > 100 ∧ h2 > 100 ∧ 
  (1/2) * a * h1 < 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_lower_bound_no_triangle_large_altitudes_small_area_l3805_380577


namespace NUMINAMATH_CALUDE_mean_quiz_score_l3805_380505

def quiz_scores : List ℝ := [88, 90, 94, 86, 85, 91]

theorem mean_quiz_score : 
  (quiz_scores.sum / quiz_scores.length : ℝ) = 89 := by sorry

end NUMINAMATH_CALUDE_mean_quiz_score_l3805_380505


namespace NUMINAMATH_CALUDE_constant_point_on_line_l3805_380559

/-- The line equation passing through a constant point regardless of m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x - (2 * m - 1) * y = 3 * m - 4

/-- The theorem stating that (-1, -2) satisfies the line equation for all m -/
theorem constant_point_on_line :
  ∀ m : ℝ, line_equation m (-1) (-2) :=
by sorry

end NUMINAMATH_CALUDE_constant_point_on_line_l3805_380559


namespace NUMINAMATH_CALUDE_inequality_proof_l3805_380593

theorem inequality_proof (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3805_380593


namespace NUMINAMATH_CALUDE_function_value_at_2_l3805_380506

/-- Given a function f(x) = ax^5 - bx + |x| - 1 where f(-2) = 2, prove that f(2) = 0 -/
theorem function_value_at_2 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x + |x| - 1)
    (h2 : f (-2) = 2) : 
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_2_l3805_380506


namespace NUMINAMATH_CALUDE_part_one_part_two_l3805_380578

/-- The function f(x) = mx^2 - mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

/-- Theorem for part 1 of the problem --/
theorem part_one :
  ∀ x : ℝ, f (1/2) x < 0 ↔ -1 < x ∧ x < 2 := by sorry

/-- Theorem for part 2 of the problem --/
theorem part_two (m : ℝ) (x : ℝ) :
  f m x < (m - 1) * x^2 + 2 * x - 2 * m - 1 ↔
    (m < 2 ∧ m < x ∧ x < 2) ∨ (m > 2 ∧ 2 < x ∧ x < m) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3805_380578


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3805_380568

/-- Represents the cost of animals in taels of silver -/
structure AnimalCost where
  cow : ℝ
  sheep : ℝ

/-- The total cost of a group of animals -/
def totalCost (c : AnimalCost) (numCows numSheep : ℕ) : ℝ :=
  c.cow * (numCows : ℝ) + c.sheep * (numSheep : ℝ)

/-- The theorem representing the ancient Chinese mathematical problem -/
theorem ancient_chinese_math_problem (c : AnimalCost) : 
  totalCost c 5 2 = 19 ∧ totalCost c 2 3 = 12 ↔ 
  (5 * c.cow + 2 * c.sheep = 19 ∧ 2 * c.cow + 3 * c.sheep = 12) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l3805_380568


namespace NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3805_380569

/-- An increasing function f with the property f(-x) + f(x) = 0 for all x -/
def IncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) + f x = 0)

theorem range_of_m2_plus_n2 
  (f : ℝ → ℝ) (m n : ℝ) 
  (h_f : IncreasingOddFunction f) 
  (h_ineq : f (m^2 - 6*m + 21) + f (n^2 - 8*n) < 0) :
  9 < m^2 + n^2 ∧ m^2 + n^2 < 49 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m2_plus_n2_l3805_380569


namespace NUMINAMATH_CALUDE_initial_girls_count_l3805_380503

theorem initial_girls_count (initial_boys : ℕ) (new_girls : ℕ) (total_pupils : ℕ) 
  (h1 : initial_boys = 222)
  (h2 : new_girls = 418)
  (h3 : total_pupils = 1346)
  : ∃ initial_girls : ℕ, initial_girls + initial_boys + new_girls = total_pupils ∧ initial_girls = 706 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3805_380503


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l3805_380536

theorem units_digit_of_7_pow_5 : (7^5) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_5_l3805_380536


namespace NUMINAMATH_CALUDE_parallel_transitivity_l3805_380592

-- Define a type for lines in a plane
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 :=
by
  sorry

#check parallel_transitivity

end NUMINAMATH_CALUDE_parallel_transitivity_l3805_380592


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3805_380579

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : 3 * a + 4 * b + 2 * c = 3) :
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) ≥ (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3805_380579


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3805_380513

/-- The distance between the vertices of the hyperbola (x^2 / 121) - (y^2 / 49) = 1 is 22 -/
theorem hyperbola_vertices_distance : 
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ (x^2 / 121) - (y^2 / 49) - 1
  let vertices := {p : ℝ × ℝ | f p = 0 ∧ p.2 = 0}
  let distance := fun (p q : ℝ × ℝ) ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (v1 v2 : ℝ × ℝ), v1 ∈ vertices ∧ v2 ∈ vertices ∧ v1 ≠ v2 ∧ distance v1 v2 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l3805_380513


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3805_380519

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 - Complex.I) : 
  z.im = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3805_380519


namespace NUMINAMATH_CALUDE_train_length_l3805_380558

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 108) (h2 : time = 10) :
  speed * time = 1080 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3805_380558


namespace NUMINAMATH_CALUDE_milk_fraction_is_two_thirds_l3805_380586

/-- Represents the content of a cup --/
structure CupContent where
  milk : ℚ
  honey : ℚ

/-- Performs the transfers between cups as described in the problem --/
def performTransfers (initial1 initial2 : CupContent) : CupContent × CupContent :=
  let afterFirstTransfer1 := CupContent.mk (initial1.milk / 2) 0
  let afterFirstTransfer2 := CupContent.mk (initial1.milk / 2) initial2.honey
  
  let totalSecond := afterFirstTransfer2.milk + afterFirstTransfer2.honey
  let secondToFirst := totalSecond / 2
  let milkRatio := afterFirstTransfer2.milk / totalSecond
  
  let afterSecondTransfer1 := CupContent.mk 
    (afterFirstTransfer1.milk + secondToFirst * milkRatio)
    (secondToFirst * (1 - milkRatio))
  let afterSecondTransfer2 := CupContent.mk 
    (afterFirstTransfer2.milk - secondToFirst * milkRatio)
    (afterFirstTransfer2.honey - secondToFirst * (1 - milkRatio))
  
  let thirdTransferAmount := (afterSecondTransfer1.milk + afterSecondTransfer1.honey) / 3
  let finalFirst := CupContent.mk 
    (afterSecondTransfer1.milk - thirdTransferAmount)
    afterSecondTransfer1.honey
  let finalSecond := CupContent.mk 
    (afterSecondTransfer2.milk + thirdTransferAmount)
    afterSecondTransfer2.honey
  
  (finalFirst, finalSecond)

/-- The main theorem stating that the fraction of milk in the second cup is 2/3 after transfers --/
theorem milk_fraction_is_two_thirds :
  let initial1 := CupContent.mk 8 0
  let initial2 := CupContent.mk 0 6
  let (_, finalSecond) := performTransfers initial1 initial2
  finalSecond.milk / (finalSecond.milk + finalSecond.honey) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_milk_fraction_is_two_thirds_l3805_380586


namespace NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l3805_380591

theorem range_of_a_given_quadratic_inequality (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_given_quadratic_inequality_l3805_380591


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_7_times_3_l3805_380510

theorem binomial_coefficient_20_7_times_3 : 3 * (Nat.choose 20 7) = 16608 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_7_times_3_l3805_380510


namespace NUMINAMATH_CALUDE_floor_breadth_correct_l3805_380552

/-- The length of the rectangular floor in meters -/
def floor_length : ℝ := 16.25

/-- The number of square tiles required to cover the floor -/
def number_of_tiles : ℕ := 3315

/-- The breadth of the rectangular floor in meters -/
def floor_breadth : ℝ := 204

/-- Theorem stating that the given breadth is correct for the rectangular floor -/
theorem floor_breadth_correct : 
  floor_length * floor_breadth = (number_of_tiles : ℝ) := by sorry

end NUMINAMATH_CALUDE_floor_breadth_correct_l3805_380552


namespace NUMINAMATH_CALUDE_bookstore_sales_percentage_l3805_380509

theorem bookstore_sales_percentage (book_sales magazine_sales other_sales : ℝ) :
  book_sales = 45 →
  magazine_sales = 25 →
  book_sales + magazine_sales + other_sales = 100 →
  other_sales = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_bookstore_sales_percentage_l3805_380509


namespace NUMINAMATH_CALUDE_max_consecutive_funny_numbers_l3805_380580

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number is funny if it's divisible by the sum of its digits plus one -/
def isFunny (n : ℕ) : Prop := n % (sumOfDigits n + 1) = 0

/-- The maximum number of consecutive funny numbers is 1 -/
theorem max_consecutive_funny_numbers :
  ∀ n : ℕ, isFunny n → isFunny (n + 1) → False := by sorry

end NUMINAMATH_CALUDE_max_consecutive_funny_numbers_l3805_380580


namespace NUMINAMATH_CALUDE_measure_one_kg_cereal_l3805_380500

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Balanced
  | Unbalanced

/-- Represents a weighing operation -/
def weighing (left right : ℕ) : WeighingResult :=
  if left = right then WeighingResult.Balanced else WeighingResult.Unbalanced

/-- Represents the process of measuring cereal -/
def measureCereal (totalCereal weight : ℕ) (maxWeighings : ℕ) : Prop :=
  ∃ (firstLeft firstRight secondLeft secondRight : ℕ),
    firstLeft + firstRight = totalCereal ∧
    secondLeft + secondRight ≤ firstLeft ∧
    weighing (firstLeft - secondLeft) (firstRight + weight) = WeighingResult.Balanced ∧
    weighing secondLeft weight = WeighingResult.Balanced ∧
    secondRight = 1 ∧
    2 ≤ maxWeighings

/-- Theorem stating that it's possible to measure 1 kg of cereal from 11 kg using a 3 kg weight in two weighings -/
theorem measure_one_kg_cereal :
  measureCereal 11 3 2 := by sorry

end NUMINAMATH_CALUDE_measure_one_kg_cereal_l3805_380500


namespace NUMINAMATH_CALUDE_loraine_wax_usage_l3805_380597

/-- The number of wax sticks used for all animals -/
def total_wax_sticks (large_animal_wax small_animal_wax : ℕ) 
  (small_animal_ratio : ℕ) (small_animal_total_wax : ℕ) : ℕ :=
  small_animal_total_wax + 
  (small_animal_total_wax / small_animal_wax) / small_animal_ratio * large_animal_wax

/-- Proof that Loraine used 20 sticks of wax for all animals -/
theorem loraine_wax_usage : 
  total_wax_sticks 4 2 3 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_loraine_wax_usage_l3805_380597


namespace NUMINAMATH_CALUDE_word_problems_count_l3805_380576

theorem word_problems_count (total_questions : ℕ) 
                             (addition_subtraction_problems : ℕ) 
                             (steve_answered : ℕ) : 
  total_questions = 45 →
  addition_subtraction_problems = 28 →
  steve_answered = 38 →
  total_questions - steve_answered = 7 →
  total_questions - addition_subtraction_problems = 17 := by
  sorry

end NUMINAMATH_CALUDE_word_problems_count_l3805_380576


namespace NUMINAMATH_CALUDE_assignment_methods_count_l3805_380516

/-- The number of companies available for internship --/
def num_companies : ℕ := 4

/-- The number of interns to be assigned --/
def num_interns : ℕ := 5

/-- The number of ways to assign interns to companies --/
def assignment_count : ℕ := num_companies ^ num_interns

/-- Theorem stating that the number of assignment methods is 1024 --/
theorem assignment_methods_count : assignment_count = 1024 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l3805_380516


namespace NUMINAMATH_CALUDE_range_of_a_l3805_380502

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≠ 0) →  -- p is true
  (∃ x : ℝ, x > 0 ∧ 2^x - a ≤ 0) →  -- q is false
  a ∈ Set.Ioo 1 2 :=  -- a is in the open interval (1, 2)
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3805_380502


namespace NUMINAMATH_CALUDE_rectangle_area_bounds_l3805_380538

/-- Represents the reported dimension of a rectangular tile -/
structure ReportedDimension where
  value : ℝ
  min : ℝ := value - 1.0
  max : ℝ := value + 1.0

/-- Represents a rectangular tile with reported dimensions -/
structure ReportedRectangle where
  length : ReportedDimension
  width : ReportedDimension

/-- Calculates the minimum area of a reported rectangle -/
def minArea (rect : ReportedRectangle) : ℝ :=
  rect.length.min * rect.width.min

/-- Calculates the maximum area of a reported rectangle -/
def maxArea (rect : ReportedRectangle) : ℝ :=
  rect.length.max * rect.width.max

theorem rectangle_area_bounds :
  let rect : ReportedRectangle := {
    length := { value := 4 },
    width := { value := 6 }
  }
  minArea rect = 15.0 ∧ maxArea rect = 35.0 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_bounds_l3805_380538


namespace NUMINAMATH_CALUDE_inequality_not_holding_l3805_380527

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-3*x > -3*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_not_holding_l3805_380527


namespace NUMINAMATH_CALUDE_pizza_cost_l3805_380514

/-- Proves that the cost of each pizza is $11 given the conditions of the problem -/
theorem pizza_cost (total_money : ℕ) (initial_bill : ℕ) (final_bill : ℕ) (num_pizzas : ℕ) :
  total_money = 42 →
  initial_bill = 30 →
  final_bill = 39 →
  num_pizzas = 3 →
  ∃ (pizza_cost : ℕ), 
    pizza_cost * num_pizzas = total_money - (final_bill - initial_bill) ∧
    pizza_cost = 11 :=
by
  sorry

#check pizza_cost

end NUMINAMATH_CALUDE_pizza_cost_l3805_380514


namespace NUMINAMATH_CALUDE_complement_union_M_N_l3805_380507

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2}
def N : Set Nat := {3, 4}

theorem complement_union_M_N : (M ∪ N)ᶜ = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l3805_380507


namespace NUMINAMATH_CALUDE_jeans_pricing_markup_l3805_380545

theorem jeans_pricing_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - cost) / cost = 0.82 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_markup_l3805_380545


namespace NUMINAMATH_CALUDE_hospital_staff_count_l3805_380533

theorem hospital_staff_count (doctors nurses : ℕ) : 
  (doctors : ℚ) / nurses = 8 / 11 → 
  nurses = 264 → 
  doctors + nurses = 456 :=
by sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l3805_380533


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_problem_2_l3805_380599

/-- Two quantities are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 4 = 36 →
  x 9 = 16 :=
by sorry

theorem inverse_proportion_problem_2 :
  ∀ a b : ℝ → ℝ,
  InverselyProportional a b →
  a 5 = 50 →
  a 10 = 25 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_inverse_proportion_problem_2_l3805_380599


namespace NUMINAMATH_CALUDE_base_b_is_five_l3805_380541

/-- The base in which 200 (base 10) is represented with exactly 4 digits -/
def base_b : ℕ := 5

/-- 200 in base 10 -/
def number : ℕ := 200

theorem base_b_is_five :
  ∃! b : ℕ, b > 1 ∧ 
  (b ^ 3 ≤ number) ∧ 
  (number < b ^ 4) ∧
  (∀ d : ℕ, d < b → number ≥ d * b ^ 3) :=
sorry

end NUMINAMATH_CALUDE_base_b_is_five_l3805_380541


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l3805_380572

def arithmetic_sum (a₁ n : ℕ) : ℕ := n * (a₁ + (a₁ + n - 1)) / 2

theorem piggy_bank_problem (initial_amount final_amount : ℕ) : 
  final_amount = 1478 →
  arithmetic_sum 1 52 = 1378 →
  initial_amount = final_amount - arithmetic_sum 1 52 →
  initial_amount = 100 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l3805_380572


namespace NUMINAMATH_CALUDE_math_competition_problem_l3805_380539

theorem math_competition_problem :
  ∀ (total students_only_A students_A_and_others students_only_B students_only_C students_B_and_C : ℕ),
    total = 25 →
    total = students_only_A + students_A_and_others + students_only_B + students_only_C + students_B_and_C →
    students_only_B + students_B_and_C = 2 * (students_only_C + students_B_and_C) →
    students_only_A = students_A_and_others + 1 →
    2 * (students_only_B + students_only_C) = students_only_A →
    students_only_B = 6 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_problem_l3805_380539


namespace NUMINAMATH_CALUDE_randy_pictures_l3805_380585

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end NUMINAMATH_CALUDE_randy_pictures_l3805_380585


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3805_380598

def smaller_number : ℝ := 20

def larger_number : ℝ := 6 * smaller_number

theorem ratio_of_numbers : larger_number / smaller_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3805_380598


namespace NUMINAMATH_CALUDE_student_count_bound_l3805_380543

theorem student_count_bound (N M k ℓ : ℕ) (h1 : M = k * N / 100) 
  (h2 : 100 * (M + 1) = ℓ * (N + 3)) (h3 : ℓ < 100) : N ≤ 197 := by
  sorry

end NUMINAMATH_CALUDE_student_count_bound_l3805_380543


namespace NUMINAMATH_CALUDE_set_T_is_hexagon_l3805_380595

/-- The set T of points (x, y) satisfying the given conditions forms a hexagon -/
theorem set_T_is_hexagon (b : ℝ) (hb : b > 0) :
  let T : Set (ℝ × ℝ) :=
    {p | b ≤ p.1 ∧ p.1 ≤ 3*b ∧
         b ≤ p.2 ∧ p.2 ≤ 3*b ∧
         p.1 + p.2 ≥ 2*b ∧
         p.1 + 2*b ≥ 2*p.2 ∧
         p.2 + 2*b ≥ 2*p.1}
  ∃ (vertices : Finset (ℝ × ℝ)), vertices.card = 6 ∧
    ∀ p ∈ T, p ∈ convexHull ℝ (vertices : Set (ℝ × ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_set_T_is_hexagon_l3805_380595


namespace NUMINAMATH_CALUDE_xyz_inequality_l3805_380523

theorem xyz_inequality (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l3805_380523


namespace NUMINAMATH_CALUDE_sandi_spending_ratio_l3805_380553

/-- Proves that the ratio of Sandi's spending to her initial amount is 1:2 --/
theorem sandi_spending_ratio :
  ∀ (sandi_initial sandi_spent gillian_spent : ℚ),
  sandi_initial = 600 →
  gillian_spent = 3 * sandi_spent + 150 →
  gillian_spent = 1050 →
  sandi_spent / sandi_initial = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sandi_spending_ratio_l3805_380553


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3805_380567

theorem quadratic_rewrite (x : ℝ) :
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 24 = (a * x + b : ℝ)^2 + c ∧ a * b = -20 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3805_380567


namespace NUMINAMATH_CALUDE_factorial_ratio_l3805_380529

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 47 = 117600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3805_380529


namespace NUMINAMATH_CALUDE_stock_price_calculation_l3805_380594

/-- Given a closing price and percent increase, calculate the opening price of a stock. -/
theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) :
  closing_price = 29 ∧ 
  percent_increase = 3.571428571428581 ∧
  (closing_price - opening_price) / opening_price * 100 = percent_increase →
  opening_price = 28 := by
sorry


end NUMINAMATH_CALUDE_stock_price_calculation_l3805_380594


namespace NUMINAMATH_CALUDE_car_rental_cost_l3805_380590

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost of renting a car for 5 days at $30 per day and driving 500 miles at $0.25 per mile is $275. -/
theorem car_rental_cost : rental_cost 30 (1/4) 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_cost_l3805_380590


namespace NUMINAMATH_CALUDE_coefficient_equals_20th_term_l3805_380560

theorem coefficient_equals_20th_term : 
  let binomial (n k : ℕ) := Nat.choose n k
  let coefficient := binomial 5 4 + binomial 6 4 + binomial 7 4
  let a (n : ℕ) := 3 * n - 5
  coefficient = a 20 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_20th_term_l3805_380560


namespace NUMINAMATH_CALUDE_cube_surface_area_l3805_380518

theorem cube_surface_area (x : ℝ) (h : x > 0) :
  let volume := x^3
  let side_length := x
  let surface_area := 6 * side_length^2
  volume = x^3 → surface_area = 6 * x^2 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3805_380518


namespace NUMINAMATH_CALUDE_other_number_proof_l3805_380581

theorem other_number_proof (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 33)
  (h3 : a = 231) : 
  b = 660 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l3805_380581


namespace NUMINAMATH_CALUDE_faculty_size_l3805_380555

/-- The number of students studying numeric methods -/
def nm : ℕ := 240

/-- The number of students studying automatic control of airborne vehicles -/
def acav : ℕ := 423

/-- The number of students studying both numeric methods and automatic control -/
def nm_acav : ℕ := 134

/-- The number of students studying advanced robotics -/
def ar : ℕ := 365

/-- The number of students studying both numeric methods and advanced robotics -/
def nm_ar : ℕ := 75

/-- The number of students studying both automatic control and advanced robotics -/
def acav_ar : ℕ := 95

/-- The number of students studying all three subjects -/
def all_three : ℕ := 45

/-- The proportion of second year students to total students -/
def second_year_ratio : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 905

theorem faculty_size :
  (nm + acav + ar - nm_acav - nm_ar - acav_ar + all_three : ℚ) / second_year_ratio = total_students := by
  sorry

end NUMINAMATH_CALUDE_faculty_size_l3805_380555


namespace NUMINAMATH_CALUDE_animals_in_field_l3805_380521

/-- The number of animals running through a field -/
def total_animals (dog : ℕ) (cats : ℕ) (rabbits_per_cat : ℕ) (hares_per_rabbit : ℕ) : ℕ :=
  dog + cats + (cats * rabbits_per_cat) + (cats * rabbits_per_cat * hares_per_rabbit)

/-- Theorem stating the total number of animals in the field -/
theorem animals_in_field : total_animals 1 4 2 3 = 37 := by
  sorry

end NUMINAMATH_CALUDE_animals_in_field_l3805_380521


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_18_mod_25_l3805_380542

theorem largest_five_digit_congruent_18_mod_25 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≤ 99999 ∧ n % 25 = 18 → 
    n ≤ 99993 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_18_mod_25_l3805_380542


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l3805_380508

theorem fraction_equality_solution : ∃! x : ℚ, (4 + x) / (6 + x) = (1 + x) / (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l3805_380508


namespace NUMINAMATH_CALUDE_cos_double_angle_for_point_l3805_380544

/-- Given a point P(-1, 2) on the terminal side of angle α, prove that cos(2α) = -3/5 -/
theorem cos_double_angle_for_point (α : ℝ) : 
  let P : ℝ × ℝ := (-1, 2)
  (P.1 = -1 ∧ P.2 = 2) → -- P has coordinates (-1, 2)
  (P.1 = -1 * Real.sqrt 5 * Real.cos α ∧ P.2 = 2 * Real.sqrt 5 * Real.sin α) → -- P is on the terminal side of angle α
  Real.cos (2 * α) = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_for_point_l3805_380544


namespace NUMINAMATH_CALUDE_length_of_chord_AB_equation_of_line_PQ_l3805_380589

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def line_l (x y : ℝ) : Prop := x - y + 2 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2*x + 2*Real.sqrt 3*y

-- Theorem for the length of chord AB
theorem length_of_chord_AB :
  ∃ (A B : ℝ × ℝ),
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
sorry

-- Theorem for the equation of line PQ
theorem equation_of_line_PQ :
  ∃ (P Q : ℝ × ℝ),
    circle_O P.1 P.2 ∧ circle_O Q.1 Q.2 ∧
    circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧
    ∀ (x y : ℝ), (x - P.1) * (Q.2 - P.2) = (y - P.2) * (Q.1 - P.1) ↔
      x + Real.sqrt 3 * y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_length_of_chord_AB_equation_of_line_PQ_l3805_380589


namespace NUMINAMATH_CALUDE_absent_present_probability_l3805_380563

theorem absent_present_probability (p : ℝ) (h1 : p = 2/30) :
  let q := 1 - p
  2 * (p * q) = 28/225 := by sorry

end NUMINAMATH_CALUDE_absent_present_probability_l3805_380563


namespace NUMINAMATH_CALUDE_page_number_added_twice_l3805_380511

theorem page_number_added_twice (n : ℕ) (h1 : n > 0) : 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630) → 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l3805_380511


namespace NUMINAMATH_CALUDE_min_value_problem_l3805_380512

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3805_380512


namespace NUMINAMATH_CALUDE_employee_payments_l3805_380573

theorem employee_payments (total_payment : ℕ) (base_c : ℕ) (commission_c : ℕ) :
  total_payment = 2000 ∧
  base_c = 400 ∧
  commission_c = 100 →
  ∃ (payment_a payment_b payment_c : ℕ),
    payment_a = (3 * payment_b) / 2 ∧
    payment_c = base_c + commission_c ∧
    payment_a + payment_b + payment_c = total_payment ∧
    payment_a = 900 ∧
    payment_b = 600 ∧
    payment_c = 500 :=
by
  sorry


end NUMINAMATH_CALUDE_employee_payments_l3805_380573


namespace NUMINAMATH_CALUDE_rd_participation_and_optimality_l3805_380546

/-- Represents a firm engaged in R&D -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario in country A -/
structure RDScenario where
  V : ℝ  -- Value of successful solo development
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total expected profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total expected profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- Theorem stating the conditions for participation and social optimality -/
theorem rd_participation_and_optimality (s : RDScenario) 
    (h_α_pos : 0 < s.α) (h_α_lt_one : s.α < 1) :
  bothParticipateCondition s ↔ 
    expectedRevenueBoth s ≥ s.IC ∧
    (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → 
      bothParticipateCondition s ∧ totalProfitOne s > totalProfitBoth s) :=
sorry

end NUMINAMATH_CALUDE_rd_participation_and_optimality_l3805_380546


namespace NUMINAMATH_CALUDE_count_m_with_integer_roots_l3805_380588

def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  x^2 - m*x + m + 2006 = 0

def has_integer_roots (m : ℤ) : Prop :=
  ∃ a b : ℤ, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ quadratic_equation m a ∧ quadratic_equation m b

theorem count_m_with_integer_roots :
  ∃! (S : Finset ℤ), (∀ m : ℤ, m ∈ S ↔ has_integer_roots m) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_m_with_integer_roots_l3805_380588


namespace NUMINAMATH_CALUDE_sum_fraction_bounds_l3805_380550

theorem sum_fraction_bounds (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (b + c + a) + c / (c + d + b) + d / (d + a + c)
  1 < S ∧ S < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_bounds_l3805_380550


namespace NUMINAMATH_CALUDE_least_clock_equivalent_square_l3805_380530

def clock_equivalent (a b : ℕ) : Prop :=
  (a - b) % 24 = 0 ∨ (b - a) % 24 = 0

theorem least_clock_equivalent_square : 
  ∀ n : ℕ, n > 6 → n < 9 → ¬(clock_equivalent n (n^2)) ∧ clock_equivalent 9 (9^2) :=
by sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_square_l3805_380530


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_and_side_range_l3805_380583

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  top : ℝ
  bottom : ℝ
  side1 : ℝ
  side2 : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.top + t.bottom + t.side1 + t.side2

/-- Theorem stating the relationship between perimeter and side length,
    and the valid range for the variable side length -/
theorem trapezoid_perimeter_and_side_range (x : ℝ) :
  let t := Trapezoid.mk 4 7 12 x
  (perimeter t = x + 23) ∧ (9 < x ∧ x < 15) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_and_side_range_l3805_380583


namespace NUMINAMATH_CALUDE_circle_center_correct_l3805_380566

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 + 6*y - 16 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, -3)

/-- Theorem: The center of the circle with equation x^2 + 4x + y^2 + 6y - 16 = 0 is (-2, -3) -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l3805_380566


namespace NUMINAMATH_CALUDE_area_of_closed_figure_l3805_380532

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x

theorem area_of_closed_figure : 
  ∫ x in (1/2)..1, (1/x + 2*x - 3) = 3/4 - Real.log 2 := by sorry

end NUMINAMATH_CALUDE_area_of_closed_figure_l3805_380532


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l3805_380575

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 44 ↔ n * (n + 1) ≤ 2000 := by
  sorry

theorem largest_n_sum_less_than_1000 : ∀ k > 44, k * (k + 1) > 2000 := by
  sorry

theorem max_consecutive_integers_1000 : 
  (∀ m ≤ 44, m * (m + 1) ≤ 2000) ∧
  (∀ k > 44, k * (k + 1) > 2000) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_sum_less_than_1000_max_consecutive_integers_1000_l3805_380575


namespace NUMINAMATH_CALUDE_end_with_same_digits_l3805_380528

/-- A function that returns the last four digits of a number -/
def lastFourDigits (n : ℕ) : ℕ := n % 10000

/-- A function that returns the first three digits of a four-digit number -/
def firstThreeDigits (n : ℕ) : ℕ := n / 10

theorem end_with_same_digits (N : ℕ) (h1 : N > 0) 
  (h2 : lastFourDigits N = lastFourDigits (N^2)) 
  (h3 : lastFourDigits N ≥ 1000) : firstThreeDigits (lastFourDigits N) = 937 := by
  sorry

end NUMINAMATH_CALUDE_end_with_same_digits_l3805_380528


namespace NUMINAMATH_CALUDE_f_two_equals_negative_eight_l3805_380522

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem f_two_equals_negative_eight
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x^3) :
  f 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_f_two_equals_negative_eight_l3805_380522


namespace NUMINAMATH_CALUDE_restaurant_menu_combinations_l3805_380547

theorem restaurant_menu_combinations : 
  (12 * 11) * (12 * 10) = 15840 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_combinations_l3805_380547


namespace NUMINAMATH_CALUDE_village_assistants_selection_l3805_380584

theorem village_assistants_selection (n : ℕ) (k : ℕ) (a b : ℕ) :
  n = 10 → k = 3 → a ≠ b → a ≤ n → b ≤ n →
  (Nat.choose n k - Nat.choose (n - 2) k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l3805_380584


namespace NUMINAMATH_CALUDE_modulus_of_complex_square_l3805_380587

theorem modulus_of_complex_square : ∃ (z : ℂ), z = (3 - Complex.I)^2 ∧ Complex.abs z = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_square_l3805_380587
