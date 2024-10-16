import Mathlib

namespace NUMINAMATH_CALUDE_square_greater_than_abs_l1366_136688

theorem square_greater_than_abs (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_abs_l1366_136688


namespace NUMINAMATH_CALUDE_sergey_age_l1366_136680

/-- Calculates the number of full years given a person's age components --/
def fullYears (years months weeks days hours : ℕ) : ℕ :=
  years + (months / 12) + ((weeks * 7 + days) / 365)

/-- Theorem stating that given the specific age components, the result is 39 full years --/
theorem sergey_age : fullYears 36 36 36 36 36 = 39 := by
  sorry

end NUMINAMATH_CALUDE_sergey_age_l1366_136680


namespace NUMINAMATH_CALUDE_blue_paint_to_change_fuchsia_to_mauve_l1366_136654

/-- Represents the composition of paint mixtures -/
structure PaintMixture where
  red : ℚ
  blue : ℚ

/-- The ratio of red to blue paint in fuchsia -/
def fuchsia_ratio : PaintMixture :=
  { red := 6, blue := 3 }

/-- The ratio of red to blue paint in mauve -/
def mauve_ratio : PaintMixture :=
  { red := 4, blue := 5 }

/-- 
Given:
- F is the amount of fuchsia paint
- B is the amount of blue paint added to F
Prove that B = F/2 is the amount of blue paint needed to change F amount of fuchsia paint to mauve paint.
-/
theorem blue_paint_to_change_fuchsia_to_mauve (F B : ℚ) :
  (F * fuchsia_ratio.red) / (F * fuchsia_ratio.blue + B * mauve_ratio.blue) = 
  mauve_ratio.red / mauve_ratio.blue →
  B = F / 2 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_to_change_fuchsia_to_mauve_l1366_136654


namespace NUMINAMATH_CALUDE_exists_good_pair_for_all_constructed_pair_is_good_l1366_136632

/-- A pair of natural numbers (m, n) is good if mn and (m+1)(n+1) are perfect squares -/
def is_good_pair (m n : ℕ) : Prop :=
  ∃ a b : ℕ, m * n = a ^ 2 ∧ (m + 1) * (n + 1) = b ^ 2

/-- For every natural number m, there exists a good pair (m, n) with n > m -/
theorem exists_good_pair_for_all (m : ℕ) : ∃ n : ℕ, n > m ∧ is_good_pair m n := by
  sorry

/-- The constructed pair (m, m(4m + 3)²) is good for any natural number m -/
theorem constructed_pair_is_good (m : ℕ) : is_good_pair m (m * (4 * m + 3) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_good_pair_for_all_constructed_pair_is_good_l1366_136632


namespace NUMINAMATH_CALUDE_max_ab_empty_solution_set_l1366_136699

theorem max_ab_empty_solution_set (a b : ℝ) : 
  (∀ x > 0, x - a * Real.log x + a - b ≥ 0) → 
  ab ≤ (1/2 : ℝ) * Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_max_ab_empty_solution_set_l1366_136699


namespace NUMINAMATH_CALUDE_cube_root_identity_l1366_136679

theorem cube_root_identity : (2^3 * 5^6 * 7^3 : ℝ)^(1/3 : ℝ) = 350 := by sorry

end NUMINAMATH_CALUDE_cube_root_identity_l1366_136679


namespace NUMINAMATH_CALUDE_sum_of_parts_l1366_136635

theorem sum_of_parts (x y : ℝ) : 
  x + y = 52 → 
  y = 30.333333333333332 → 
  10 * x + 22 * y = 884 := by
sorry

end NUMINAMATH_CALUDE_sum_of_parts_l1366_136635


namespace NUMINAMATH_CALUDE_line_equation_proof_l1366_136655

/-- Given two lines in the form y = mx + b, they are parallel if and only if they have the same slope m -/
def parallel_lines (m1 b1 m2 b2 : ℝ) : Prop := m1 = m2

/-- A point (x, y) lies on a line y = mx + b if and only if y = mx + b -/
def point_on_line (x y m b : ℝ) : Prop := y = m * x + b

theorem line_equation_proof (x y : ℝ) : 
  parallel_lines (3/2) 3 (3/2) (-11/2) ∧ 
  point_on_line 3 (-1) (3/2) (-11/2) ∧
  3 * x - 2 * y - 11 = 0 ↔ y = (3/2) * x - 11/2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1366_136655


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l1366_136659

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the line
def line (x : ℝ) : ℝ := 6*x - 4

-- Theorem statement
theorem line_satisfies_conditions :
  -- Condition 1: The line passes through (2, 8)
  (line 2 = 8) ∧
  -- Condition 2: There exists a k where x = k intersects both curves 4 units apart
  (∃ k : ℝ, |parabola k - line k| = 4) ∧
  -- Condition 3: The y-intercept is not 0
  (line 0 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l1366_136659


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1366_136614

theorem quadratic_equation_roots_ratio (q : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ / r₂ = 3 ∧ 
   r₁^2 + 10*r₁ + q = 0 ∧ r₂^2 + 10*r₂ + q = 0) → 
  q = 18.75 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1366_136614


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l1366_136683

theorem number_of_girls_in_school (total_boys : ℕ) (total_sections : ℕ) 
  (h1 : total_boys = 408)
  (h2 : total_sections = 27)
  (h3 : total_boys % total_sections = 0) -- Boys are divided into equal sections
  : ∃ (total_girls : ℕ), 
    total_girls = 324 ∧ 
    total_girls % total_sections = 0 ∧ -- Girls are divided into equal sections
    (total_boys / total_sections + total_girls / total_sections = total_sections) :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_girls_in_school_l1366_136683


namespace NUMINAMATH_CALUDE_max_value_theorem_l1366_136626

theorem max_value_theorem (x y z : ℝ) (h : x + 3 * y + z = 6) :
  ∃ (max : ℝ), max = 4 ∧ ∀ (a b c : ℝ), a + 3 * b + c = 6 → 2 * a * b + a * c + b * c ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1366_136626


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l1366_136604

def M : Set ℤ := {0, 1, 2, 3, 4}
def N : Set ℤ := {-2, 0, 2}

theorem set_intersection_theorem : M ∩ N = {0, 2} := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l1366_136604


namespace NUMINAMATH_CALUDE_smallest_candy_count_l1366_136612

theorem smallest_candy_count : ∃ (n : ℕ), 
  n = 128 ∧ 
  100 ≤ n ∧ n < 1000 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(9 ∣ (m + 7) ∧ 7 ∣ (m - 9))) ∧
  9 ∣ (n + 7) ∧
  7 ∣ (n - 9) := by
  sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l1366_136612


namespace NUMINAMATH_CALUDE_daily_medicine_dose_l1366_136653

theorem daily_medicine_dose (total_medicine : ℝ) (daily_fraction : ℝ) :
  total_medicine = 426 →
  daily_fraction = 0.06 →
  total_medicine * daily_fraction = 25.56 := by
  sorry

end NUMINAMATH_CALUDE_daily_medicine_dose_l1366_136653


namespace NUMINAMATH_CALUDE_minus_eight_representation_l1366_136649

-- Define a type for temperature
structure Temperature where
  value : ℤ
  unit : String

-- Define a function to represent temperature above or below zero
def aboveZero (t : Temperature) : Bool :=
  t.value > 0

-- Define the given condition
axiom plus_ten_above_zero : aboveZero (Temperature.mk 10 "°C") = true

-- State the theorem to be proved
theorem minus_eight_representation :
  Temperature.mk (-8) "°C" = Temperature.mk (-8) "°C" :=
sorry

end NUMINAMATH_CALUDE_minus_eight_representation_l1366_136649


namespace NUMINAMATH_CALUDE_triangle_sides_from_divided_areas_l1366_136643

/-- Given a triangle with an inscribed circle, if the segments from the vertices to the center
    of the inscribed circle divide the triangle's area into parts of 28, 60, and 80,
    then the sides of the triangle are 14, 30, and 40. -/
theorem triangle_sides_from_divided_areas (a b c : ℝ) (r : ℝ) :
  (1/2 * a * r = 28) →
  (1/2 * b * r = 60) →
  (1/2 * c * r = 80) →
  (a = 14 ∧ b = 30 ∧ c = 40) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_from_divided_areas_l1366_136643


namespace NUMINAMATH_CALUDE_common_tangent_line_sum_of_coefficients_l1366_136631

/-- Parabola P₁ -/
def P₁ (x y : ℝ) : Prop := y = x^2 + 121/100

/-- Parabola P₂ -/
def P₂ (x y : ℝ) : Prop := x = y^2 + 49/4

/-- The common tangent line L -/
def L (x y : ℝ) : Prop := x + 25*y = 12

/-- The theorem stating that L is a common tangent to P₁ and P₂, 
    and 1, 25, 12 are the smallest positive integers satisfying the equation -/
theorem common_tangent_line : 
  (∀ x y : ℝ, P₁ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₁ u v → L u v → (u, v) = (x, y))) ∧ 
  (∀ x y : ℝ, P₂ x y → L x y → (∃ u : ℝ, ∀ v : ℝ, P₂ u v → L u v → (u, v) = (x, y))) ∧
  (∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → a ≥ 1 ∧ b ≥ 25 ∧ c ≥ 12) :=
sorry

/-- The sum of the coefficients -/
def coefficient_sum : ℕ := 38

/-- Theorem stating that the sum of coefficients is 38 -/
theorem sum_of_coefficients : 
  ∀ a b c : ℕ+, (∀ x y : ℝ, a*x + b*y = c ↔ L x y) → (a : ℕ) + (b : ℕ) + (c : ℕ) = coefficient_sum :=
sorry

end NUMINAMATH_CALUDE_common_tangent_line_sum_of_coefficients_l1366_136631


namespace NUMINAMATH_CALUDE_second_walking_speed_l1366_136623

/-- Proves that the second walking speed is 6 km/h given the problem conditions -/
theorem second_walking_speed (distance : ℝ) (speed1 : ℝ) (miss_time : ℝ) (early_time : ℝ) (v : ℝ) : 
  distance = 13.5 ∧ 
  speed1 = 5 ∧ 
  miss_time = 12 / 60 ∧ 
  early_time = 15 / 60 ∧ 
  distance / speed1 - miss_time = distance / v + early_time → 
  v = 6 := by
  sorry

end NUMINAMATH_CALUDE_second_walking_speed_l1366_136623


namespace NUMINAMATH_CALUDE_backyard_area_l1366_136645

theorem backyard_area (length width : ℝ) 
  (h1 : 40 * length = 1000) 
  (h2 : 8 * (2 * (length + width)) = 1000) : 
  length * width = 937.5 := by
sorry

end NUMINAMATH_CALUDE_backyard_area_l1366_136645


namespace NUMINAMATH_CALUDE_tree_walk_properties_l1366_136634

/-- Represents a random walk on a line of trees. -/
structure TreeWalk where
  n : ℕ
  trees : Fin (2 * n + 1) → ℕ
  start : Fin (2 * n + 1)
  prob_left : ℚ
  prob_stay : ℚ
  prob_right : ℚ

/-- The probability of ending at a specific tree after the walk. -/
def end_probability (w : TreeWalk) (i : Fin (2 * w.n + 1)) : ℚ :=
  (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))

/-- The expected distance from the starting point after the walk. -/
def expected_distance (w : TreeWalk) : ℚ :=
  (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n))

/-- Theorem stating the properties of the random walk. -/
theorem tree_walk_properties (w : TreeWalk) 
  (h1 : w.n > 0)
  (h2 : w.start = ⟨w.n + 1, by sorry⟩)
  (h3 : w.prob_left = 1/4)
  (h4 : w.prob_stay = 1/2)
  (h5 : w.prob_right = 1/4) :
  (∀ i : Fin (2 * w.n + 1), end_probability w i = (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))) ∧
  expected_distance w = (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n)) := by
  sorry


end NUMINAMATH_CALUDE_tree_walk_properties_l1366_136634


namespace NUMINAMATH_CALUDE_composition_of_f_and_g_l1366_136665

-- Define the functions f and g
def f (A B : ℝ) (x : ℝ) : ℝ := A * x^2 - B^2
def g (B : ℝ) (x : ℝ) : ℝ := B * x + B^2

-- State the theorem
theorem composition_of_f_and_g (A B : ℝ) (h : B ≠ 0) :
  g B (f A B 1) = B * A - B^3 + B^2 := by
  sorry

end NUMINAMATH_CALUDE_composition_of_f_and_g_l1366_136665


namespace NUMINAMATH_CALUDE_range_of_a_l1366_136640

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1366_136640


namespace NUMINAMATH_CALUDE_weight_difference_proof_l1366_136642

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is -6.5 kg, given the conditions of the original problem. -/
theorem weight_difference_proof (n : ℕ) (x : ℝ) : 
  -- Joe's weight
  let joe_weight : ℝ := 43
  -- Initial average weight
  let initial_avg : ℝ := 30
  -- New average weight after Joe joins
  let new_avg : ℝ := 31
  -- Number of students in original group
  n = (joe_weight - initial_avg) / (new_avg - initial_avg)
  -- Average weight of two departing students
  → x = (new_avg * (n + 1) - initial_avg * (n - 1)) / 2
  -- Difference between average weight of departing students and Joe's weight
  → x - joe_weight = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l1366_136642


namespace NUMINAMATH_CALUDE_x_value_l1366_136641

theorem x_value (x : ℝ) (h1 : x^2 - 2*x = 0) (h2 : x ≠ 0) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1366_136641


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1366_136622

/-- Given a geometric sequence {a_n} where a_1 = 1 and 4a_2, 2a_3, a_4 form an arithmetic sequence,
    prove that the sum a_2 + a_3 + a_4 equals 14. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 →                            -- a_1 = 1
  4 * a 2 - 2 * a 3 = 2 * a 3 - a 4 →  -- arithmetic sequence condition
  a 2 + a 3 + a 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1366_136622


namespace NUMINAMATH_CALUDE_expression_simplification_l1366_136606

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3) :
  (3 / (a - 1) + 1) / ((a^2 + 2*a) / (a^2 - 1)) = (3 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1366_136606


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l1366_136602

theorem square_sum_reciprocal (m : ℝ) (hm : m > 0) (h : m - 1/m = 3) : 
  m^2 + 1/m^2 = 11 := by
sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l1366_136602


namespace NUMINAMATH_CALUDE_divide_ten_items_between_two_people_l1366_136660

theorem divide_ten_items_between_two_people : 
  Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_divide_ten_items_between_two_people_l1366_136660


namespace NUMINAMATH_CALUDE_scientific_notation_1500_l1366_136673

theorem scientific_notation_1500 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500 = a * (10 : ℝ) ^ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_1500_l1366_136673


namespace NUMINAMATH_CALUDE_circles_intersect_l1366_136661

/-- Definition of Circle O₁ -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- Definition of Circle O₂ -/
def circle_O₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- Center of Circle O₁ -/
def center_O₁ : ℝ × ℝ := (0, 1)

/-- Center of Circle O₂ -/
def center_O₂ : ℝ × ℝ := (1, 2)

/-- Radius of Circle O₁ -/
def radius_O₁ : ℝ := 1

/-- Radius of Circle O₂ -/
def radius_O₂ : ℝ := 2

/-- Theorem: Circles O₁ and O₂ are intersecting -/
theorem circles_intersect : 
  (radius_O₁ + radius_O₂ > Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2)) ∧
  (Real.sqrt ((center_O₂.1 - center_O₁.1)^2 + (center_O₂.2 - center_O₁.2)^2) > |radius_O₂ - radius_O₁|) :=
by sorry

end NUMINAMATH_CALUDE_circles_intersect_l1366_136661


namespace NUMINAMATH_CALUDE_third_number_proof_l1366_136647

theorem third_number_proof :
  ∃ x : ℝ, 12.1212 + 17.0005 - x = 20.011399999999995 ∧ x = 9.110300000000005 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l1366_136647


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l1366_136698

theorem positive_real_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a * b + a + b + 1) * (a * b + a * c + b * c + c^2) ≥ 16 * a * b * c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l1366_136698


namespace NUMINAMATH_CALUDE_min_steps_parallel_line_l1366_136675

/-- A line in a plane -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a construction step using either a line or a circle -/
inductive ConstructionStep
  | line : Line → ConstructionStep
  | circle : Circle → ConstructionStep

/-- Checks if a point is on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- The main theorem stating that the minimum number of construction steps to create a parallel line is 3 -/
theorem min_steps_parallel_line 
  (a : Line) (O : Point) (h : ¬ O.onLine a) :
  ∃ (steps : List ConstructionStep) (l : Line),
    steps.length = 3 ∧
    O.onLine l ∧
    l.parallel a ∧
    (∀ (steps' : List ConstructionStep) (l' : Line),
      steps'.length < 3 →
      ¬(O.onLine l' ∧ l'.parallel a)) :=
sorry

end NUMINAMATH_CALUDE_min_steps_parallel_line_l1366_136675


namespace NUMINAMATH_CALUDE_power_of_power_equals_six_l1366_136684

theorem power_of_power_equals_six (m : ℝ) : (m^2)^3 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_equals_six_l1366_136684


namespace NUMINAMATH_CALUDE_cersei_cousin_fraction_l1366_136671

def initial_candies : ℕ := 50
def given_to_siblings : ℕ := 5 + 5
def eaten_by_cersei : ℕ := 12
def left_after_eating : ℕ := 18

theorem cersei_cousin_fraction :
  let remaining_after_siblings := initial_candies - given_to_siblings
  let given_to_cousin := remaining_after_siblings - (left_after_eating + eaten_by_cersei)
  (given_to_cousin : ℚ) / remaining_after_siblings = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_cersei_cousin_fraction_l1366_136671


namespace NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1366_136694

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem least_prime_angle_in_right_triangle :
  ∀ a b : ℕ,
    a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
    a > b →        -- Given condition: a > b
    isPrime a →    -- a is prime
    isPrime b →    -- b is prime
    b ≥ 7 :=       -- The least possible value of b is 7
by
  sorry  -- Proof is omitted as per instructions


end NUMINAMATH_CALUDE_least_prime_angle_in_right_triangle_l1366_136694


namespace NUMINAMATH_CALUDE_beaver_problem_l1366_136603

theorem beaver_problem (initial_beavers final_beavers : ℕ) : 
  final_beavers = initial_beavers + 1 → 
  final_beavers = 3 → 
  initial_beavers = 2 := by
sorry

end NUMINAMATH_CALUDE_beaver_problem_l1366_136603


namespace NUMINAMATH_CALUDE_k_range_for_equation_solution_l1366_136669

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem k_range_for_equation_solution :
  ∀ k : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    otimes 1 (2 * k - 3 - k * x₁) = 1 + Real.sqrt (4 - x₁^2) ∧
    otimes 1 (2 * k - 3 - k * x₂) = 1 + Real.sqrt (4 - x₂^2)) →
  k > 5/12 ∧ k ≤ 3/4 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_equation_solution_l1366_136669


namespace NUMINAMATH_CALUDE_maddie_friday_episodes_l1366_136689

/-- Represents the TV watching schedule for a week -/
structure TVSchedule where
  total_episodes : ℕ
  episode_duration : ℕ
  monday_minutes : ℕ
  thursday_minutes : ℕ
  weekend_minutes : ℕ

/-- Calculates the number of episodes watched on Friday -/
def episodes_on_friday (schedule : TVSchedule) : ℕ :=
  let total_minutes := schedule.total_episodes * schedule.episode_duration
  let other_days_minutes := schedule.monday_minutes + schedule.thursday_minutes + schedule.weekend_minutes
  let friday_minutes := total_minutes - other_days_minutes
  friday_minutes / schedule.episode_duration

/-- Theorem stating that Maddie watched 2 episodes on Friday -/
theorem maddie_friday_episodes :
  let schedule := TVSchedule.mk 8 44 138 21 105
  episodes_on_friday schedule = 2 := by
  sorry

end NUMINAMATH_CALUDE_maddie_friday_episodes_l1366_136689


namespace NUMINAMATH_CALUDE_smallest_m_for_multiple_factorizations_l1366_136638

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def has_multiple_factorizations (n : ℕ) : Prop :=
  ∃ (f1 f2 : List ℕ), 
    f1 ≠ f2 ∧ 
    f1.length = 16 ∧ 
    f2.length = 16 ∧ 
    f1.Nodup ∧ 
    f2.Nodup ∧ 
    f1.prod = n ∧ 
    f2.prod = n

theorem smallest_m_for_multiple_factorizations :
  (∀ m : ℕ, m > 0 ∧ m < 24 → ¬has_multiple_factorizations (factorial 15 * m)) ∧
  has_multiple_factorizations (factorial 15 * 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_multiple_factorizations_l1366_136638


namespace NUMINAMATH_CALUDE_leo_current_weight_l1366_136666

def leo_weight_problem (leo_weight kendra_weight : ℝ) : Prop :=
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 150)

theorem leo_current_weight :
  ∃ (leo_weight kendra_weight : ℝ),
    leo_weight_problem leo_weight kendra_weight ∧
    leo_weight = 86 := by
  sorry

end NUMINAMATH_CALUDE_leo_current_weight_l1366_136666


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1366_136617

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |5 * y - 6| = 0 ∧ y = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1366_136617


namespace NUMINAMATH_CALUDE_exactly_two_single_intersection_lines_l1366_136648

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a point on the parabola
def point_on_parabola : ℝ × ℝ := (2, 4)

-- Define a function to count the number of lines intersecting the parabola at exactly one point
def count_single_intersection_lines : ℕ := sorry

-- Theorem statement
theorem exactly_two_single_intersection_lines :
  parabola point_on_parabola.1 point_on_parabola.2 ∧ count_single_intersection_lines = 2 := by sorry

end NUMINAMATH_CALUDE_exactly_two_single_intersection_lines_l1366_136648


namespace NUMINAMATH_CALUDE_ends_with_1994_l1366_136664

theorem ends_with_1994 : ∃ n : ℕ+, 1994 * 1993^(n : ℕ) ≡ 1994 [MOD 10000] := by
  sorry

end NUMINAMATH_CALUDE_ends_with_1994_l1366_136664


namespace NUMINAMATH_CALUDE_sock_counting_l1366_136652

theorem sock_counting (initial : ℕ) (thrown_away : ℕ) (new_bought : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + new_bought = initial + new_bought - thrown_away :=
by sorry

end NUMINAMATH_CALUDE_sock_counting_l1366_136652


namespace NUMINAMATH_CALUDE_sum_of_roots_l1366_136690

theorem sum_of_roots (k d y₁ y₂ : ℝ) 
  (h₁ : y₁ ≠ y₂) 
  (h₂ : 4 * y₁^2 - k * y₁ = d) 
  (h₃ : 4 * y₂^2 - k * y₂ = d) : 
  y₁ + y₂ = k / 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1366_136690


namespace NUMINAMATH_CALUDE_gcd_5factorial_8factorial_div_3factorial_l1366_136624

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5factorial_8factorial_div_3factorial : 
  Nat.gcd (factorial 5) ((factorial 8) / (factorial 3)) = 120 := by sorry

end NUMINAMATH_CALUDE_gcd_5factorial_8factorial_div_3factorial_l1366_136624


namespace NUMINAMATH_CALUDE_min_groups_needed_l1366_136696

def total_students : ℕ := 24
def max_group_size : ℕ := 10

theorem min_groups_needed : 
  (∃ (group_size : ℕ), 
    group_size > 0 ∧ 
    group_size ≤ max_group_size ∧ 
    total_students % group_size = 0 ∧
    total_students / group_size = 3) ∧
  (∀ (n : ℕ), 
    n > 0 → 
    n < 3 → 
    (∀ (group_size : ℕ), 
      group_size > 0 → 
      group_size ≤ max_group_size → 
      total_students % group_size = 0 → 
      total_students / group_size ≠ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_groups_needed_l1366_136696


namespace NUMINAMATH_CALUDE_min_distance_to_point_l1366_136625

theorem min_distance_to_point (x y : ℝ) : 
  x^2 + y^2 + 4*x - 2*y + 4 = 0 → 
  ∃ (min : ℝ), min = Real.sqrt 10 - 1 ∧ 
  ∀ (x' y' : ℝ), x'^2 + y'^2 + 4*x' - 2*y' + 4 = 0 → 
  Real.sqrt ((x' - 1)^2 + y'^2) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_point_l1366_136625


namespace NUMINAMATH_CALUDE_total_cost_proof_l1366_136611

def sandwich_price : ℚ := 245/100
def soda_price : ℚ := 87/100
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4

theorem total_cost_proof : 
  (num_sandwiches : ℚ) * sandwich_price + (num_sodas : ℚ) * soda_price = 838/100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l1366_136611


namespace NUMINAMATH_CALUDE_remainder_problem_l1366_136620

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1366_136620


namespace NUMINAMATH_CALUDE_bag_weight_out_of_range_l1366_136685

/-- The lower bound of the acceptable weight range -/
def lower_bound : ℝ := 9.5

/-- The upper bound of the acceptable weight range -/
def upper_bound : ℝ := 10.5

/-- The weight of the flour bag in question -/
def bag_weight : ℝ := 10.7

/-- Theorem stating that the bag weight is not within the acceptable range -/
theorem bag_weight_out_of_range : 
  ¬(lower_bound ≤ bag_weight ∧ bag_weight ≤ upper_bound) := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_out_of_range_l1366_136685


namespace NUMINAMATH_CALUDE_apples_to_eat_raw_l1366_136674

theorem apples_to_eat_raw (total : ℕ) (wormy : ℕ) (bruised : ℕ) : 
  total = 85 → 
  wormy = total / 5 →
  bruised = wormy + 9 →
  total - wormy - bruised = 42 := by
sorry

end NUMINAMATH_CALUDE_apples_to_eat_raw_l1366_136674


namespace NUMINAMATH_CALUDE_ball_ratio_problem_l1366_136615

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 5 / 3 →
  white_balls = 15 →
  red_balls = 9 := by
sorry

end NUMINAMATH_CALUDE_ball_ratio_problem_l1366_136615


namespace NUMINAMATH_CALUDE_thor_fraction_is_two_ninths_l1366_136636

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Thor
| Ott

-- Define the function that returns the fraction of money given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/6
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Ott => 1/3
  | Friend.Thor => 0

-- Define the amount of money given by each friend
def amountGiven : ℚ := 2

-- Define the total money of the group
def totalMoney : ℚ := (amountGiven / fractionGiven Friend.Moe) +
                      (amountGiven / fractionGiven Friend.Loki) +
                      (amountGiven / fractionGiven Friend.Nick) +
                      (amountGiven / fractionGiven Friend.Ott)

-- Define Thor's share
def thorShare : ℚ := 4 * amountGiven

-- Theorem to prove
theorem thor_fraction_is_two_ninths :
  thorShare / totalMoney = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_thor_fraction_is_two_ninths_l1366_136636


namespace NUMINAMATH_CALUDE_rectangle_area_after_length_decrease_l1366_136637

theorem rectangle_area_after_length_decrease (square_area : ℝ) 
  (rectangle_length_decrease_percent : ℝ) : 
  square_area = 49 →
  rectangle_length_decrease_percent = 20 →
  let square_side := Real.sqrt square_area
  let initial_rectangle_length := square_side
  let initial_rectangle_width := 2 * square_side
  let new_rectangle_length := initial_rectangle_length * (1 - rectangle_length_decrease_percent / 100)
  let new_rectangle_width := initial_rectangle_width
  new_rectangle_length * new_rectangle_width = 78.4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_after_length_decrease_l1366_136637


namespace NUMINAMATH_CALUDE_unique_solution_for_euler_equation_l1366_136609

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- The statement to prove -/
theorem unique_solution_for_euler_equation :
  ∀ a n : ℕ, a ≠ 0 ∧ n ≠ 0 → (φ (a^n + n) = 2^n) → (a = 2 ∧ n = 1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_euler_equation_l1366_136609


namespace NUMINAMATH_CALUDE_sams_test_score_l1366_136667

theorem sams_test_score (initial_students : ℕ) (initial_average : ℚ) (new_average : ℚ) 
  (h1 : initial_students = 19)
  (h2 : initial_average = 85)
  (h3 : new_average = 86) :
  (initial_students + 1) * new_average - initial_students * initial_average = 105 :=
by sorry

end NUMINAMATH_CALUDE_sams_test_score_l1366_136667


namespace NUMINAMATH_CALUDE_prime_factors_count_l1366_136662

/-- The total number of prime factors in the given expression -/
def total_prime_factors : ℕ :=
  (2 * 17) + (2 * 13) + (3 * 7) + (5 * 3) + (7 * 19)

/-- The theorem stating that the total number of prime factors in the given expression is 229 -/
theorem prime_factors_count :
  total_prime_factors = 229 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_count_l1366_136662


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l1366_136672

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l1366_136672


namespace NUMINAMATH_CALUDE_expression_simplification_l1366_136616

theorem expression_simplification (x : ℝ) (h : x ≠ -1) :
  x / (x + 1) - 3 * x / (2 * (x + 1)) - 1 = (-3 * x - 2) / (2 * (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1366_136616


namespace NUMINAMATH_CALUDE_square_geq_bound_l1366_136618

theorem square_geq_bound (a : ℝ) : (∀ x > 1, x^2 ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_geq_bound_l1366_136618


namespace NUMINAMATH_CALUDE_total_price_is_680_l1366_136686

/-- Calculates the total price for jewelry and paintings after price increases -/
def total_price (jewelry_original : ℕ) (painting_original : ℕ) 
  (jewelry_increase : ℕ) (painting_increase_percent : ℕ) : ℕ :=
  let jewelry_new := jewelry_original + jewelry_increase
  let painting_new := painting_original + painting_original * painting_increase_percent / 100
  2 * jewelry_new + 5 * painting_new

/-- Proves that the total price for 2 pieces of jewelry and 5 paintings is $680 -/
theorem total_price_is_680 : 
  total_price 30 100 10 20 = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_price_is_680_l1366_136686


namespace NUMINAMATH_CALUDE_combined_molecular_weight_l1366_136633

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := 2 * N_weight + 3 * O_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of CO2 in g/mol -/
def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- Combined molecular weight of 4 moles of N2O3, 3.5 moles of H2O, and 2 moles of CO2 in grams -/
theorem combined_molecular_weight :
  4 * N2O3_weight + 3.5 * H2O_weight + 2 * CO2_weight = 455.17 := by
  sorry

end NUMINAMATH_CALUDE_combined_molecular_weight_l1366_136633


namespace NUMINAMATH_CALUDE_function_property_l1366_136605

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1366_136605


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_l1366_136646

/-- The smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
def smallest_n : ℕ := 18

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

theorem smallest_n_is_correct :
  smallest_n = 18 ∧
  (∀ z : ℂ, f z = 0 → z^smallest_n = 1) ∧
  (∀ m : ℕ, m < smallest_n → ∃ z : ℂ, f z = 0 ∧ z^m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_l1366_136646


namespace NUMINAMATH_CALUDE_tangent_circles_diametric_intersection_l1366_136627

-- Define the types for our geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given circles and points
variable (c c1 c2 : Circle)
variable (A B P Q : Point)

-- Define the property of internal tangency
def internallyTangent (c1 c2 : Circle) (P : Point) : Prop :=
  -- The distance between centers is the difference of radii
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius - c2.radius)^2
  -- P lies on both circles
  ∧ (P.x - c1.center.1)^2 + (P.y - c1.center.2)^2 = c1.radius^2
  ∧ (P.x - c2.center.1)^2 + (P.y - c2.center.2)^2 = c2.radius^2

-- Define the property of a point lying on a circle
def pointOnCircle (P : Point) (c : Circle) : Prop :=
  (P.x - c.center.1)^2 + (P.y - c.center.2)^2 = c.radius^2

-- Define the property of points being diametrically opposite on a circle
def diametricallyOpposite (P Q : Point) (c : Circle) : Prop :=
  (P.x - c.center.1) = -(Q.x - c.center.1) ∧ (P.y - c.center.2) = -(Q.y - c.center.2)

-- State the theorem
theorem tangent_circles_diametric_intersection :
  internallyTangent c c1 A
  → internallyTangent c c2 B
  → ∃ (M N : Point),
    pointOnCircle M c
    ∧ pointOnCircle N c
    ∧ diametricallyOpposite M N c
    ∧ (∃ (t : ℝ), M = ⟨A.x + t * (P.x - A.x), A.y + t * (P.y - A.y)⟩)
    ∧ (∃ (s : ℝ), N = ⟨B.x + s * (Q.x - B.x), B.y + s * (Q.y - B.y)⟩) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_diametric_intersection_l1366_136627


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l1366_136651

theorem cricket_team_average_age
  (n : ℕ) -- Total number of players
  (a : ℝ) -- Average age of the whole team
  (h1 : n = 11)
  (h2 : a = 28)
  (h3 : ((n * a) - (a + (a + 3))) / (n - 2) = a - 1) :
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l1366_136651


namespace NUMINAMATH_CALUDE_rohan_farm_earnings_l1366_136663

/-- Represents a coconut farm with given characteristics -/
structure CoconutFarm where
  size : ℕ  -- farm size in square meters
  trees_per_sqm : ℕ  -- number of trees per square meter
  coconuts_per_tree : ℕ  -- number of coconuts per tree
  harvest_frequency : ℕ  -- harvest frequency in months
  price_per_coconut : ℚ  -- price per coconut in dollars
  time_period : ℕ  -- time period in months

/-- Calculates the earnings from a coconut farm over a given time period -/
def calculate_earnings (farm : CoconutFarm) : ℚ :=
  let total_trees := farm.size * farm.trees_per_sqm
  let total_coconuts_per_harvest := total_trees * farm.coconuts_per_tree
  let number_of_harvests := farm.time_period / farm.harvest_frequency
  let total_coconuts := total_coconuts_per_harvest * number_of_harvests
  total_coconuts * farm.price_per_coconut

/-- Theorem stating that the earnings from Rohan's coconut farm after 6 months is $240 -/
theorem rohan_farm_earnings :
  let farm : CoconutFarm := {
    size := 20,
    trees_per_sqm := 2,
    coconuts_per_tree := 6,
    harvest_frequency := 3,
    price_per_coconut := 1/2,
    time_period := 6
  }
  calculate_earnings farm = 240 := by sorry

end NUMINAMATH_CALUDE_rohan_farm_earnings_l1366_136663


namespace NUMINAMATH_CALUDE_sea_horse_count_l1366_136600

theorem sea_horse_count : 
  ∀ (s p : ℕ), 
  (s : ℚ) / p = 5 / 11 → 
  p = s + 85 → 
  s = 70 := by
sorry

end NUMINAMATH_CALUDE_sea_horse_count_l1366_136600


namespace NUMINAMATH_CALUDE_decimal_to_binary_88_l1366_136613

theorem decimal_to_binary_88 : 
  (88 : ℕ).digits 2 = [0, 0, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_88_l1366_136613


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1366_136601

theorem system_solution_ratio (x y c d : ℝ) : 
  x ≠ 0 → y ≠ 0 → d ≠ 0 → 
  (4 * x + 5 * y = c) → (8 * y - 10 * x = d) → 
  c / d = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1366_136601


namespace NUMINAMATH_CALUDE_tangerines_left_l1366_136670

def total_tangerines : ℕ := 27
def eaten_tangerines : ℕ := 18

theorem tangerines_left : total_tangerines - eaten_tangerines = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l1366_136670


namespace NUMINAMATH_CALUDE_same_even_number_probability_l1366_136691

-- Define a standard die
def standardDie : ℕ := 6

-- Define the number of even faces on a standard die
def evenFaces : ℕ := 3

-- Define the number of dice rolled
def numDice : ℕ := 4

-- Theorem statement
theorem same_even_number_probability :
  let p : ℚ := (evenFaces / standardDie) * (1 / standardDie)^(numDice - 1)
  p = 1 / 432 := by
  sorry


end NUMINAMATH_CALUDE_same_even_number_probability_l1366_136691


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1366_136650

-- Define the equation
def equation (x : ℝ) : Prop := (4 * x + 6) * (3 * x - 7) = 0

-- State the theorem
theorem sum_of_solutions : 
  ∃ (s : ℝ), (∀ (x : ℝ), equation x → x = s ∨ x = (5/6 - s)) ∧ s + (5/6 - s) = 5/6 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1366_136650


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l1366_136687

theorem ice_cream_distribution (total_sandwiches : Nat) (num_nieces : Nat) 
  (h1 : total_sandwiches = 143)
  (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l1366_136687


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1366_136607

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => valid_arrangements (n + 1) + valid_arrangements n

def total_outcomes : ℕ := 2^num_people

theorem no_adjacent_standing_probability :
  (valid_arrangements num_people : ℚ) / total_outcomes = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1366_136607


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_3_and_5_l1366_136697

theorem smallest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧  -- three-digit number
  (n % 3 = 0 ∧ n % 5 = 0) ∧  -- multiple of 3 and 5
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (m % 3 = 0 ∧ m % 5 = 0) → m ≥ n) ∧  -- smallest such number
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_3_and_5_l1366_136697


namespace NUMINAMATH_CALUDE_flower_shop_profit_l1366_136678

-- Define the profit function
def profit (n : ℕ) : ℤ :=
  if n < 16 then 10 * n - 80 else 80

-- Define the probability distribution
def prob (x : ℤ) : ℝ :=
  if x = 60 then 0.1
  else if x = 70 then 0.2
  else if x = 80 then 0.7
  else 0

-- Define the expected value
def expected_profit : ℝ :=
  60 * prob 60 + 70 * prob 70 + 80 * prob 80

-- Define the variance
def variance_profit : ℝ :=
  (60 - expected_profit)^2 * prob 60 +
  (70 - expected_profit)^2 * prob 70 +
  (80 - expected_profit)^2 * prob 80

-- Theorem statement
theorem flower_shop_profit :
  expected_profit = 76 ∧ variance_profit = 44 :=
sorry

end NUMINAMATH_CALUDE_flower_shop_profit_l1366_136678


namespace NUMINAMATH_CALUDE_cone_prism_volume_ratio_l1366_136630

/-- The ratio of the volume of a right circular cone inscribed in a right rectangular prism
    to the volume of the prism is π/16. -/
theorem cone_prism_volume_ratio :
  ∀ (cone_volume prism_volume : ℝ) (prism_base_length prism_base_width prism_height : ℝ),
  prism_base_length = 3 →
  prism_base_width = 4 →
  prism_height = 5 →
  prism_volume = prism_base_length * prism_base_width * prism_height →
  cone_volume = (1/3) * π * (prism_base_length/2)^2 * prism_height →
  cone_volume / prism_volume = π/16 := by
sorry

end NUMINAMATH_CALUDE_cone_prism_volume_ratio_l1366_136630


namespace NUMINAMATH_CALUDE_katrina_lunch_sales_l1366_136676

/-- The number of cookies sold during the lunch rush -/
def lunch_rush_sales (initial : ℕ) (morning_dozens : ℕ) (afternoon : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12) - afternoon - remaining

/-- Proof that Katrina sold 57 cookies during the lunch rush -/
theorem katrina_lunch_sales :
  lunch_rush_sales 120 3 16 11 = 57 := by
  sorry

end NUMINAMATH_CALUDE_katrina_lunch_sales_l1366_136676


namespace NUMINAMATH_CALUDE_digit_with_value_difference_l1366_136693

def numeral : List Nat := [6, 5, 7, 9, 3]

def local_value (digit : Nat) (place : Nat) : Nat :=
  digit * (10 ^ place)

def face_value (digit : Nat) : Nat := digit

theorem digit_with_value_difference (diff : Nat) :
  ∃ (index : Fin 5), 
    local_value (numeral[index]) (4 - index) - face_value (numeral[index]) = diff →
    numeral[index] = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_digit_with_value_difference_l1366_136693


namespace NUMINAMATH_CALUDE_same_terminal_side_l1366_136668

theorem same_terminal_side (k : ℤ) : 
  (2 * k * π + π / 5 : ℝ) = 21 * π / 5 → 
  ∃ n : ℤ, (21 * π / 5 : ℝ) = 2 * n * π + π / 5 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1366_136668


namespace NUMINAMATH_CALUDE_min_value_theorem_l1366_136639

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 2) (h4 : m * n > 0) :
  2 / m + 1 / n ≥ 9 / 2 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2 * m₀ + n₀ = 2 ∧ m₀ * n₀ > 0 ∧ 2 / m₀ + 1 / n₀ = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1366_136639


namespace NUMINAMATH_CALUDE_rectangle_area_is_75_l1366_136621

/-- Represents a rectangle with length and breadth -/
structure Rectangle where
  length : ℝ
  breadth : ℝ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.breadth)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.breadth

/-- Theorem: A rectangle with length thrice its breadth and perimeter 40 has an area of 75 -/
theorem rectangle_area_is_75 (r : Rectangle) 
    (h1 : r.length = 3 * r.breadth) 
    (h2 : perimeter r = 40) : 
  area r = 75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_75_l1366_136621


namespace NUMINAMATH_CALUDE_ray_remaining_nickels_l1366_136657

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def initial_amount : ℕ := 285

/-- Amount given to Peter in cents -/
def peter_amount : ℕ := 55

/-- Amount given to Paula in cents -/
def paula_amount : ℕ := 45

/-- Calculates the number of nickels from a given amount of cents -/
def cents_to_nickels (cents : ℕ) : ℕ := cents / nickel_value

theorem ray_remaining_nickels :
  let initial_nickels := cents_to_nickels initial_amount
  let peter_nickels := cents_to_nickels peter_amount
  let randi_nickels := cents_to_nickels (3 * peter_amount)
  let paula_nickels := cents_to_nickels paula_amount
  initial_nickels - (peter_nickels + randi_nickels + paula_nickels) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ray_remaining_nickels_l1366_136657


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l1366_136629

def sequence_a (n : ℕ) : ℕ :=
  2 * n

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_b (n : ℕ) : ℚ :=
  1 / (n * (n + 1))

def sum_T (n : ℕ) : ℚ :=
  n / (n + 1)

theorem sequence_sum_theorem (n : ℕ) :
  sequence_a 2 = 4 ∧
  (∀ k : ℕ, sequence_a (k + 1) = sequence_a k + 2) ∧
  (∀ k : ℕ, sum_S k = k * k) ∧
  (∀ k : ℕ, sequence_b k = 1 / sum_S k) →
  sum_T n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l1366_136629


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1366_136682

theorem modular_congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1366_136682


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1366_136681

theorem triangle_side_calculation (a b c : ℝ) (A B C : ℝ) :
  2 * Real.sin (2 * B + π / 6) = 2 →
  a * c = 3 * Real.sqrt 3 →
  a + c = 4 →
  b ^ 2 = 16 - 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1366_136681


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_l1366_136692

def problem (A B C D : ℕ) : Prop :=
  A ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  B ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  C ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  D ∈ ({2, 3, 4, 5} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A : ℚ) / B - (C : ℚ) / D = 1

theorem sum_of_A_and_C (A B C D : ℕ) (h : problem A B C D) : A + C = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_l1366_136692


namespace NUMINAMATH_CALUDE_pizza_fraction_proof_l1366_136619

theorem pizza_fraction_proof (michael_fraction lamar_fraction treshawn_fraction : ℚ) : 
  michael_fraction = 1/3 →
  lamar_fraction = 1/6 →
  michael_fraction + lamar_fraction + treshawn_fraction = 1 →
  treshawn_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_pizza_fraction_proof_l1366_136619


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1366_136644

-- Define the function f(x) = |x-a| + x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l1366_136644


namespace NUMINAMATH_CALUDE_ice_cream_ratio_l1366_136677

theorem ice_cream_ratio (sunday pints : ℕ) (k : ℕ) : 
  sunday = 4 →
  let monday := k * sunday
  let tuesday := monday / 3
  let wednesday := tuesday / 2
  18 = sunday + monday + tuesday - wednesday →
  monday / sunday = 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_l1366_136677


namespace NUMINAMATH_CALUDE_leaf_collection_time_l1366_136656

/-- Represents the leaf collection problem --/
structure LeafCollection where
  totalLeaves : ℕ
  collectionRate : ℕ
  scatterRate : ℕ
  cycleTime : ℕ

/-- Calculates the time needed to collect all leaves --/
def collectionTime (lc : LeafCollection) : ℚ :=
  let netIncrease := lc.collectionRate - lc.scatterRate
  let cycles := (lc.totalLeaves - lc.scatterRate) / netIncrease
  let totalSeconds := (cycles + 1) * lc.cycleTime
  totalSeconds / 60

/-- Theorem stating that the collection time for the given problem is 21.5 minutes --/
theorem leaf_collection_time :
  let lc : LeafCollection := {
    totalLeaves := 45,
    collectionRate := 4,
    scatterRate := 3,
    cycleTime := 30
  }
  collectionTime lc = 21.5 := by sorry

end NUMINAMATH_CALUDE_leaf_collection_time_l1366_136656


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l1366_136608

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial weaving output on the first day -/
def initial_output : ℚ := 5

/-- Represents the total number of days -/
def total_days : ℕ := 30

/-- Represents the total amount of fabric woven over the period -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  (initial_output + (total_days - 1) * daily_increase / 2) * total_days = total_output := by
  sorry

end NUMINAMATH_CALUDE_weaving_increase_proof_l1366_136608


namespace NUMINAMATH_CALUDE_quadrilateral_impossibility_l1366_136628

theorem quadrilateral_impossibility : ¬ ∃ (a b c d : ℝ),
  (2 * a^2 - 18 * a + 36 = 0 ∨ a^2 - 20 * a + 75 = 0) ∧
  (2 * b^2 - 18 * b + 36 = 0 ∨ b^2 - 20 * b + 75 = 0) ∧
  (2 * c^2 - 18 * c + 36 = 0 ∨ c^2 - 20 * c + 75 = 0) ∧
  (2 * d^2 - 18 * d + 36 = 0 ∨ d^2 - 20 * d + 75 = 0) ∧
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a) ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_impossibility_l1366_136628


namespace NUMINAMATH_CALUDE_firm_employs_50_looms_l1366_136610

/-- Represents the number of looms employed by a textile manufacturing firm. -/
def number_of_looms : ℕ := sorry

/-- The aggregate sales value of the output of the looms in rupees. -/
def aggregate_sales : ℕ := 500000

/-- The monthly manufacturing expenses in rupees. -/
def manufacturing_expenses : ℕ := 150000

/-- The monthly establishment charges in rupees. -/
def establishment_charges : ℕ := 75000

/-- The decrease in profit when one loom breaks down for a month, in rupees. -/
def profit_decrease : ℕ := 7000

/-- Theorem stating that the number of looms employed by the firm is 50. -/
theorem firm_employs_50_looms :
  number_of_looms = 50 ∧
  aggregate_sales / number_of_looms - manufacturing_expenses / number_of_looms = profit_decrease :=
sorry

end NUMINAMATH_CALUDE_firm_employs_50_looms_l1366_136610


namespace NUMINAMATH_CALUDE_increasing_function_condition_l1366_136695

open Real

/-- The function f(x) = (ln x) / x - kx is increasing on (0, +∞) iff k ≤ -1/(2e³) -/
theorem increasing_function_condition (k : ℝ) :
  (∀ x > 0, StrictMono (λ x => (log x) / x - k * x)) ↔ k ≤ -1 / (2 * (exp 3)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l1366_136695


namespace NUMINAMATH_CALUDE_max_basketballs_l1366_136658

-- Define the cost of soccer balls and basketballs
def cost_3_soccer_2_basket : ℕ := 490
def cost_2_soccer_4_basket : ℕ := 660
def total_balls : ℕ := 62
def max_total_cost : ℕ := 6750

-- Define the function to calculate the total cost
def total_cost (soccer_balls : ℕ) (basketballs : ℕ) : ℕ :=
  let soccer_cost := (cost_3_soccer_2_basket * 2 - cost_2_soccer_4_basket * 3) / 2
  let basket_cost := (cost_2_soccer_4_basket * 3 - cost_3_soccer_2_basket * 2) / 2
  soccer_cost * soccer_balls + basket_cost * basketballs

-- Theorem to prove
theorem max_basketballs :
  ∃ (m : ℕ), m = 39 ∧
  (∀ (n : ℕ), n > m → total_cost (total_balls - n) n > max_total_cost) ∧
  total_cost (total_balls - m) m ≤ max_total_cost :=
sorry

end NUMINAMATH_CALUDE_max_basketballs_l1366_136658
