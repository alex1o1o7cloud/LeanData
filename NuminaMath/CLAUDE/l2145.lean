import Mathlib

namespace NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l2145_214562

theorem students_neither_football_nor_cricket 
  (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : football = 325)
  (h3 : cricket = 175)
  (h4 : both = 140) :
  total - (football + cricket - both) = 50 := by
sorry

end NUMINAMATH_CALUDE_students_neither_football_nor_cricket_l2145_214562


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2145_214537

theorem complex_magnitude_equation (x : ℝ) : 
  (x > 0 ∧ Complex.abs (-3 + x * Complex.I) = 5 * Real.sqrt 5) → x = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2145_214537


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2145_214579

theorem quadratic_root_value (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2145_214579


namespace NUMINAMATH_CALUDE_intersection_projection_distance_l2145_214501

/-- Given a line and a circle intersecting at two points, 
    prove that the distance between the projections of these points on the x-axis is 4. -/
theorem intersection_projection_distance (A B C D : ℝ × ℝ) : 
  -- Line equation
  (∀ (x y : ℝ), (x, y) ∈ {(x, y) | x - Real.sqrt 3 * y + 6 = 0} → 
    (A.1 - Real.sqrt 3 * A.2 + 6 = 0 ∧ B.1 - Real.sqrt 3 * B.2 + 6 = 0)) →
  -- Circle equation
  (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12) →
  -- A and B are distinct points
  A ≠ B →
  -- C and D are projections of A and B on x-axis
  (C = (A.1, 0) ∧ D = (B.1, 0)) →
  -- Distance between C and D is 4
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4 :=
by sorry


end NUMINAMATH_CALUDE_intersection_projection_distance_l2145_214501


namespace NUMINAMATH_CALUDE_f_2015_value_l2145_214513

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_01 : ∀ x, x ∈ Set.Icc 0 1 → f x = 3^x - 1) :
  f 2015 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_2015_value_l2145_214513


namespace NUMINAMATH_CALUDE_binomial_1294_2_l2145_214509

theorem binomial_1294_2 : Nat.choose 1294 2 = 836161 := by sorry

end NUMINAMATH_CALUDE_binomial_1294_2_l2145_214509


namespace NUMINAMATH_CALUDE_divisibility_property_l2145_214555

theorem divisibility_property (y : ℕ) (hy : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2145_214555


namespace NUMINAMATH_CALUDE_mary_found_47_shells_l2145_214595

/-- The number of seashells Sam found -/
def sam_shells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_shells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_shells : ℕ := total_shells - sam_shells

theorem mary_found_47_shells : mary_shells = 47 := by
  sorry

end NUMINAMATH_CALUDE_mary_found_47_shells_l2145_214595


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_2850_l2145_214504

theorem consecutive_negative_integers_product_2850 :
  ∃ (n : ℤ), n < 0 ∧ n * (n + 1) = 2850 → (n + (n + 1)) = -107 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_2850_l2145_214504


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2145_214523

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 2 + m = 0) → ((-1)^2 - (-1) + m = 0) := by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2145_214523


namespace NUMINAMATH_CALUDE_log_inequality_equiv_range_l2145_214500

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_equiv_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_range_l2145_214500


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l2145_214582

theorem largest_divisor_of_n (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ (v : ℕ), v = 12 ∧ v ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ v :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l2145_214582


namespace NUMINAMATH_CALUDE_power_of_power_l2145_214550

theorem power_of_power (a : ℝ) : (a^3)^4 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2145_214550


namespace NUMINAMATH_CALUDE_courtyard_width_is_20_l2145_214592

/-- Represents a rectangular paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a paving stone -/
def area_paving_stone (stone : PavingStone) : ℝ :=
  stone.length * stone.width

/-- Calculates the area of a courtyard -/
def area_courtyard (yard : Courtyard) : ℝ :=
  yard.length * yard.width

/-- Theorem: The width of the courtyard is 20 meters -/
theorem courtyard_width_is_20 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 4)
    (h2 : stone.width = 2)
    (h3 : yard.length = 40)
    (h4 : area_courtyard yard = 100 * area_paving_stone stone) :
    yard.width = 20 := by
  sorry

#check courtyard_width_is_20

end NUMINAMATH_CALUDE_courtyard_width_is_20_l2145_214592


namespace NUMINAMATH_CALUDE_unique_provider_choices_l2145_214524

theorem unique_provider_choices (n m : ℕ) (hn : n = 23) (hm : m = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 213840 := by
  sorry

end NUMINAMATH_CALUDE_unique_provider_choices_l2145_214524


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2145_214520

theorem quadratic_root_difference (p : ℝ) : 
  let a := 1
  let b := -(2*p + 1)
  let c := p^2 - 5
  let discriminant := b^2 - 4*a*c
  let root_difference := Real.sqrt discriminant / (2*a)
  root_difference = Real.sqrt (2*p^2 + 4*p + 11) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2145_214520


namespace NUMINAMATH_CALUDE_smallest_AC_l2145_214564

/-- Triangle ABC with point D on AC --/
structure Triangle :=
  (AC : ℕ)
  (CD : ℕ)
  (BD : ℝ)

/-- Conditions for the triangle --/
def ValidTriangle (t : Triangle) : Prop :=
  t.AC = t.AC  -- AB = AC (isosceles)
  ∧ t.CD ≤ t.AC  -- D is on AC
  ∧ t.BD ^ 2 = 85  -- BD² = 85
  ∧ t.AC ^ 2 = (t.AC - t.CD) ^ 2 + 85  -- Pythagorean theorem

/-- The smallest possible AC value is 11 --/
theorem smallest_AC : 
  ∀ t : Triangle, ValidTriangle t → t.AC ≥ 11 ∧ ∃ t' : Triangle, ValidTriangle t' ∧ t'.AC = 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_AC_l2145_214564


namespace NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l2145_214516

/-- Given a circle with an inscribed rectangle of dimensions 9 cm by 12 cm,
    the circumference of the circle is 15π cm. -/
theorem circle_circumference_with_inscribed_rectangle :
  ∀ (C : ℝ → ℝ → Prop) (r : ℝ),
    (∃ (x y : ℝ), C x y ∧ x^2 + y^2 = r^2 ∧ x = 9 ∧ y = 12) →
    2 * π * r = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_circumference_with_inscribed_rectangle_l2145_214516


namespace NUMINAMATH_CALUDE_cube_minus_square_equals_zero_l2145_214507

theorem cube_minus_square_equals_zero : 4^3 - 8^2 = 0 :=
by
  -- Given conditions (not used in the proof, but included for completeness)
  have h1 : 2^3 - 7^2 = 1 := by sorry
  have h2 : 3^3 - 6^2 = 9 := by sorry
  have h3 : 5^3 - 9^2 = 16 := by sorry
  
  -- Proof
  sorry

end NUMINAMATH_CALUDE_cube_minus_square_equals_zero_l2145_214507


namespace NUMINAMATH_CALUDE_function_range_l2145_214599

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem function_range :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y < 3} := by sorry

end NUMINAMATH_CALUDE_function_range_l2145_214599


namespace NUMINAMATH_CALUDE_angle_D_measure_l2145_214559

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define the quadrilateral formed by drawing a line
structure Quadrilateral (t : Triangle) where
  D : ℝ
  line_sum_180 : D + (180 - t.A - t.B) = 180

-- Theorem statement
theorem angle_D_measure (t : Triangle) (q : Quadrilateral t) 
  (h1 : t.A = 85) (h2 : t.B = 34) : q.D = 119 := by
  sorry

end NUMINAMATH_CALUDE_angle_D_measure_l2145_214559


namespace NUMINAMATH_CALUDE_equation_solution_l2145_214542

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 9*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3 * Real.sqrt 3 ∧ 
  x = 34 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2145_214542


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l2145_214576

theorem triangle_angle_problem (α : Real) 
  (h1 : 0 < α ∧ α < π) -- α is an internal angle of a triangle
  (h2 : Real.sin α + Real.cos α = 1/5) :
  (Real.tan α = -4/3) ∧ 
  ((Real.sin (3*π/2 + α) * Real.sin (π/2 - α) * Real.tan (π - α)^3) / 
   (Real.cos (π/2 + α) * Real.cos (3*π/2 - α)) = -4/3) := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l2145_214576


namespace NUMINAMATH_CALUDE_last_guard_hours_l2145_214526

/-- Represents the number of hours in a night shift -/
def total_hours : ℕ := 9

/-- Represents the number of guards -/
def num_guards : ℕ := 4

/-- Represents the hours taken by the first guard -/
def first_guard_hours : ℕ := 3

/-- Represents the hours taken by each middle guard -/
def middle_guard_hours : ℕ := 2

/-- Represents the number of middle guards -/
def num_middle_guards : ℕ := 2

theorem last_guard_hours :
  total_hours - (first_guard_hours + num_middle_guards * middle_guard_hours) = 2 := by
  sorry

end NUMINAMATH_CALUDE_last_guard_hours_l2145_214526


namespace NUMINAMATH_CALUDE_large_power_of_two_appears_early_l2145_214566

/-- Represents the state of cards on the table at any given time -/
structure CardState where
  totalCards : Nat
  oddCards : Nat
  maxPowerOfTwo : Nat

/-- The initial state of cards -/
def initialState : CardState :=
  { totalCards := 100, oddCards := 43, maxPowerOfTwo := 0 }

/-- Function to calculate the next state after one minute -/
def nextState (state : CardState) : CardState :=
  { totalCards := state.totalCards + 1,
    oddCards := if state.oddCards = 43 then 44 else 44,
    maxPowerOfTwo := state.maxPowerOfTwo + 1 }

/-- Function to calculate the state after n minutes -/
def stateAfterMinutes (n : Nat) : CardState :=
  match n with
  | 0 => initialState
  | n + 1 => nextState (stateAfterMinutes n)

theorem large_power_of_two_appears_early (n : Nat) :
  (stateAfterMinutes n).maxPowerOfTwo ≥ 10000 →
  (stateAfterMinutes 1440).maxPowerOfTwo ≥ 10000 :=
by
  sorry

#check large_power_of_two_appears_early

end NUMINAMATH_CALUDE_large_power_of_two_appears_early_l2145_214566


namespace NUMINAMATH_CALUDE_triangle_cosine_relation_l2145_214505

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, if S + a² = (b + c)², then cos A = -15/17 -/
theorem triangle_cosine_relation (a b c S : ℝ) (A : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < S →  -- positive area
  S = (1/2) * b * c * Real.sin A →  -- area formula
  a^2 + b^2 - 2 * a * b * Real.cos A = c^2 →  -- cosine law
  S + a^2 = (b + c)^2 →  -- given condition
  Real.cos A = -15/17 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_relation_l2145_214505


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2145_214560

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line L
def line_L (x y m : ℝ) : Prop := x - Real.sqrt 2 * y - m = 0

-- Define the point P
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for intersection points A and B
def intersection_condition (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_L x₁ y₁ m ∧ line_L x₂ y₂ m ∧
    ((x₁ - m)^2 + y₁^2) * ((x₂ - m)^2 + y₂^2) = 1

-- Theorem statement
theorem intersection_points_theorem :
  ∀ m : ℝ, intersection_condition m ↔ m = 1 + Real.sqrt 7 / 2 ∨ m = 1 - Real.sqrt 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2145_214560


namespace NUMINAMATH_CALUDE_buddy_fraction_l2145_214518

theorem buddy_fraction (s n : ℕ) (hs : s > 0) (hn : n > 0) : 
  (n : ℚ) / 3 = (2 : ℚ) * s / 5 → 
  ((n : ℚ) / 3 + (2 : ℚ) * s / 5) / (n + s : ℚ) = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_buddy_fraction_l2145_214518


namespace NUMINAMATH_CALUDE_smallest_constant_inequality_l2145_214567

theorem smallest_constant_inequality (x y : ℝ) :
  (∀ D : ℝ, (∀ x y : ℝ, x^4 + y^4 + 1 ≥ D * (x^2 + y^2)) → D ≤ Real.sqrt 2) ∧
  (∀ x y : ℝ, x^4 + y^4 + 1 ≥ Real.sqrt 2 * (x^2 + y^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_constant_inequality_l2145_214567


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l2145_214577

/-- The length of the courtyard in centimeters -/
def courtyard_length : ℕ := 378

/-- The width of the courtyard in centimeters -/
def courtyard_width : ℕ := 525

/-- The size of the largest square tile in centimeters -/
def largest_tile_size : ℕ := 21

theorem largest_square_tile_size :
  (courtyard_length % largest_tile_size = 0) ∧
  (courtyard_width % largest_tile_size = 0) ∧
  ∀ (tile_size : ℕ), tile_size > largest_tile_size →
    (courtyard_length % tile_size ≠ 0) ∨ (courtyard_width % tile_size ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l2145_214577


namespace NUMINAMATH_CALUDE_banana_permutations_l2145_214583

/-- The number of distinct permutations of a sequence with repeated elements -/
def distinctPermutations (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations :
  distinctPermutations 6 [3, 2, 1] = 60 := by
  sorry

#eval distinctPermutations 6 [3, 2, 1]

end NUMINAMATH_CALUDE_banana_permutations_l2145_214583


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l2145_214532

/-- The angle between two intersecting lines in degrees -/
def angle_between_lines : ℝ := 10

/-- The maximum number of reflections before hitting perpendicularly -/
def max_reflections : ℕ := 18

/-- Theorem stating the maximum number of reflections -/
theorem max_reflections_theorem : 
  ∀ (n : ℕ), n * angle_between_lines ≤ 180 → n ≤ max_reflections :=
sorry

end NUMINAMATH_CALUDE_max_reflections_theorem_l2145_214532


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l2145_214548

theorem a_gt_one_sufficient_not_necessary_for_a_sq_gt_a :
  (∀ a : ℝ, a > 1 → a^2 > a) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > a) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_sq_gt_a_l2145_214548


namespace NUMINAMATH_CALUDE_power_function_odd_l2145_214530

def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ n : ℤ, ∀ x : ℝ, f x = x ^ n

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem power_function_odd (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 1 = 3) :
  isOddFunction f := by
  sorry

end NUMINAMATH_CALUDE_power_function_odd_l2145_214530


namespace NUMINAMATH_CALUDE_equal_money_after_transfer_l2145_214508

/-- Given that Lucy originally has $20 and Linda has $10, prove that if Lucy gives $5 to Linda,
    they will have the same amount of money. -/
theorem equal_money_after_transfer (lucy_initial : ℕ) (linda_initial : ℕ) (transfer_amount : ℕ) : 
  lucy_initial = 20 →
  linda_initial = 10 →
  transfer_amount = 5 →
  lucy_initial - transfer_amount = linda_initial + transfer_amount :=
by sorry

end NUMINAMATH_CALUDE_equal_money_after_transfer_l2145_214508


namespace NUMINAMATH_CALUDE_subsets_sum_to_negative_eight_l2145_214556

def S : Finset Int := {-6, -4, -2, -1, 1, 2, 3, 4, 6}

theorem subsets_sum_to_negative_eight :
  ∃! (subsets : Finset (Finset Int)),
    (∀ subset ∈ subsets, subset ⊆ S ∧ (subset.sum id = -8)) ∧
    subsets.card = 6 :=
by sorry

end NUMINAMATH_CALUDE_subsets_sum_to_negative_eight_l2145_214556


namespace NUMINAMATH_CALUDE_tank_capacities_l2145_214563

theorem tank_capacities (x y z : ℚ) : 
  x + y + z = 1620 →
  z = x + (1/5) * y →
  z = y + (1/3) * x →
  x = 540 ∧ y = 450 ∧ z = 630 := by
sorry

end NUMINAMATH_CALUDE_tank_capacities_l2145_214563


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l2145_214525

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron where
  /-- Length of one pair of opposite edges -/
  a : ℝ
  /-- Length of the other pair of opposite edges -/
  b : ℝ
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Ensure a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- Ensure r is positive -/
  hr : 0 < r

/-- The radius of the inscribed sphere is less than ab/(2(a+b)) -/
theorem inscribed_sphere_radius_bound (t : Tetrahedron) : t.r < (t.a * t.b) / (2 * (t.a + t.b)) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_bound_l2145_214525


namespace NUMINAMATH_CALUDE_like_terms_sum_l2145_214531

theorem like_terms_sum (a b : ℝ) (x y : ℝ) 
  (h : 3 * a^(7*x) * b^(y+7) = 5 * a^(2-4*y) * b^(2*x)) : x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l2145_214531


namespace NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_7_l2145_214547

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_imply_a_equals_7 :
  let l1 : Line := { a := 2, b := 1, c := -1 }
  let l2 : Line := { a := a - 1, b := 3, c := -2 }
  parallel l1 l2 → a = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_imply_a_equals_7_l2145_214547


namespace NUMINAMATH_CALUDE_dice_probability_l2145_214590

/-- The number of possible outcomes when rolling 7 six-sided dice -/
def total_outcomes : ℕ := 6^7

/-- The number of ways to choose 2 numbers from 6 -/
def choose_two_from_six : ℕ := Nat.choose 6 2

/-- The number of ways to arrange 2 pairs in 7 positions -/
def arrange_two_pairs : ℕ := Nat.choose 7 2 * Nat.choose 5 2

/-- The number of ways to arrange remaining dice for two pairs case -/
def arrange_remaining_two_pairs : ℕ := 4 * 3 * 2

/-- The number of ways to arrange triplet and pair -/
def arrange_triplet_pair : ℕ := Nat.choose 7 3 * Nat.choose 4 2

/-- The number of ways to arrange remaining dice for triplet and pair case -/
def arrange_remaining_triplet_pair : ℕ := 4 * 3

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ :=
  (choose_two_from_six * arrange_two_pairs * arrange_remaining_two_pairs) +
  (2 * choose_two_from_six * arrange_triplet_pair * arrange_remaining_triplet_pair)

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 525 / 972 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2145_214590


namespace NUMINAMATH_CALUDE_product_of_roots_l2145_214554

theorem product_of_roots (x : ℝ) : 
  (6 = 2 * x^2 + 4 * x) → 
  (let a := 2
   let b := 4
   let c := -6
   c / a = -3) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2145_214554


namespace NUMINAMATH_CALUDE_original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l2145_214587

-- Define the properties
def is_rectangle (q : Quadrilateral) : Prop := sorry
def has_opposite_sides_equal (q : Quadrilateral) : Prop := sorry

-- Define the original statement
theorem original_statement : 
  ∀ q : Quadrilateral, is_rectangle q → has_opposite_sides_equal q := sorry

-- Prove that the converse is not always true
theorem converse_not_always_true : 
  ¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q) := sorry

-- Prove that the inverse is not always true
theorem inverse_not_always_true : 
  ¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q) := sorry

-- Combine the results
theorem neither_converse_nor_inverse_always_true : 
  (¬(∀ q : Quadrilateral, has_opposite_sides_equal q → is_rectangle q)) ∧
  (¬(∀ q : Quadrilateral, ¬is_rectangle q → ¬has_opposite_sides_equal q)) := sorry

end NUMINAMATH_CALUDE_original_statement_converse_not_always_true_inverse_not_always_true_neither_converse_nor_inverse_always_true_l2145_214587


namespace NUMINAMATH_CALUDE_function_characterization_l2145_214598

theorem function_characterization (f : ℚ → ℚ) 
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l2145_214598


namespace NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l2145_214565

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_equals_pi_plus_one :
  f (f (f (-2))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_pi_plus_one_l2145_214565


namespace NUMINAMATH_CALUDE_exponent_simplification_l2145_214594

theorem exponent_simplification (x : ℝ) : 4 * x^3 - 3 * x^3 = x^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l2145_214594


namespace NUMINAMATH_CALUDE_two_sin_sixty_degrees_l2145_214551

theorem two_sin_sixty_degrees : 2 * Real.sin (π / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_sin_sixty_degrees_l2145_214551


namespace NUMINAMATH_CALUDE_max_juggling_time_max_juggling_time_value_l2145_214571

/-- Represents the time in seconds before Bobo drops a cow when juggling n cows -/
def drop_time (n : ℕ) : ℕ :=
  match n with
  | 1 => 64
  | 2 => 55
  | 3 => 47
  | 4 => 40
  | 5 => 33
  | 6 => 27
  | 7 => 22
  | 8 => 18
  | 9 => 14
  | 10 => 13
  | 11 => 12
  | 12 => 11
  | 13 => 10
  | 14 => 9
  | 15 => 8
  | 16 => 7
  | 17 => 6
  | 18 => 5
  | 19 => 4
  | 20 => 3
  | 21 => 2
  | 22 => 1
  | _ => 0

/-- Calculates the total juggling time for n cows -/
def total_time (n : ℕ) : ℕ := n * drop_time n

/-- The maximum number of cows Bobo can juggle -/
def max_cows : ℕ := 22

/-- Theorem: The maximum total juggling time is achieved with 5 cows -/
theorem max_juggling_time :
  ∀ n : ℕ, n ≤ max_cows → total_time 5 ≥ total_time n :=
by
  sorry

/-- Corollary: The maximum total juggling time is 165 seconds -/
theorem max_juggling_time_value : total_time 5 = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_max_juggling_time_max_juggling_time_value_l2145_214571


namespace NUMINAMATH_CALUDE_x_greater_than_y_l2145_214575

theorem x_greater_than_y (x y z : ℝ) 
  (eq1 : x + y + z = 28)
  (eq2 : 2 * x - y = 32)
  (pos_x : x > 0)
  (pos_y : y > 0)
  (pos_z : z > 0) :
  x > y := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l2145_214575


namespace NUMINAMATH_CALUDE_all_same_number_probability_l2145_214586

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice thrown -/
def num_dice : ℕ := 5

/-- The total number of possible outcomes when throwing the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (all dice showing the same number) -/
def favorable_outcomes : ℕ := num_faces

/-- The probability of all dice showing the same number -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem all_same_number_probability :
  probability = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_all_same_number_probability_l2145_214586


namespace NUMINAMATH_CALUDE_pizza_coworkers_l2145_214553

theorem pizza_coworkers (num_pizzas : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) :
  num_pizzas = 3 →
  slices_per_pizza = 8 →
  slices_per_person = 2 →
  (num_pizzas * slices_per_pizza) / slices_per_person = 12 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coworkers_l2145_214553


namespace NUMINAMATH_CALUDE_cone_lateral_area_l2145_214549

/-- Given a cone with a central angle of 120° in its unfolded diagram and a base circle radius of 2 cm,
    prove that its lateral area is 12π cm². -/
theorem cone_lateral_area (central_angle : Real) (base_radius : Real) (lateral_area : Real) :
  central_angle = 120 * (π / 180) →
  base_radius = 2 →
  lateral_area = 12 * π →
  lateral_area = (1 / 2) * (2 * π * base_radius) * ((2 * π * base_radius) / (2 * π * (central_angle / (2 * π)))) :=
by sorry


end NUMINAMATH_CALUDE_cone_lateral_area_l2145_214549


namespace NUMINAMATH_CALUDE_electric_power_is_4_l2145_214578

-- Define the constants and variables
variable (k_star : ℝ) (e_tau : ℝ) (a_star : ℝ) (N_H : ℝ) (N_e : ℝ)

-- Define the conditions
axiom k_star_def : k_star = 1/3
axiom e_tau_a_star_def : e_tau * a_star = 0.15
axiom N_H_def : N_H = 80

-- Define the electric power equation
def electric_power (k_star e_tau a_star N_H : ℝ) : ℝ :=
  k_star * e_tau * a_star * N_H

-- State the theorem
theorem electric_power_is_4 :
  electric_power k_star e_tau a_star N_H = 4 :=
sorry

end NUMINAMATH_CALUDE_electric_power_is_4_l2145_214578


namespace NUMINAMATH_CALUDE_fraction_of_lunch_eaten_l2145_214534

def total_calories : ℕ := 40
def recommended_calories : ℕ := 25
def extra_calories : ℕ := 5

def actual_calories : ℕ := recommended_calories + extra_calories

theorem fraction_of_lunch_eaten :
  (actual_calories : ℚ) / total_calories = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_lunch_eaten_l2145_214534


namespace NUMINAMATH_CALUDE_age_of_30th_student_l2145_214515

theorem age_of_30th_student 
  (total_students : Nat)
  (avg_age_all : ℝ)
  (num_group1 : Nat) (avg_age_group1 : ℝ)
  (num_group2 : Nat) (avg_age_group2 : ℝ)
  (num_group3 : Nat) (avg_age_group3 : ℝ)
  (age_single_student : ℝ)
  (h1 : total_students = 30)
  (h2 : avg_age_all = 23.5)
  (h3 : num_group1 = 9)
  (h4 : avg_age_group1 = 21.3)
  (h5 : num_group2 = 12)
  (h6 : avg_age_group2 = 19.7)
  (h7 : num_group3 = 7)
  (h8 : avg_age_group3 = 24.2)
  (h9 : age_single_student = 35)
  (h10 : num_group1 + num_group2 + num_group3 + 1 + 1 = total_students) :
  total_students * avg_age_all - 
  (num_group1 * avg_age_group1 + num_group2 * avg_age_group2 + 
   num_group3 * avg_age_group3 + age_single_student) = 72.5 := by
  sorry

end NUMINAMATH_CALUDE_age_of_30th_student_l2145_214515


namespace NUMINAMATH_CALUDE_alternating_pair_sum_50_eq_2550_l2145_214527

def alternatingPairSum (n : Nat) : Int :=
  let f (k : Nat) : Int :=
    if k % 4 ≤ 1 then (n - k + 1)^2 else -(n - k + 1)^2
  (List.range n).map f |>.sum

theorem alternating_pair_sum_50_eq_2550 :
  alternatingPairSum 50 = 2550 := by
  sorry

end NUMINAMATH_CALUDE_alternating_pair_sum_50_eq_2550_l2145_214527


namespace NUMINAMATH_CALUDE_seating_arrangements_l2145_214511

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of children -/
def total_children : ℕ := num_boys + num_girls

/-- Calculates the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways boys can sit together -/
def boys_together : ℕ := permutations num_boys num_boys * permutations (num_girls + 1) (num_girls + 1)

/-- The number of arrangements where no two girls sit next to each other -/
def girls_not_adjacent : ℕ := permutations num_boys num_boys * permutations (num_boys + 1) num_girls

/-- The number of ways boys can sit together and girls can sit together -/
def boys_and_girls_together : ℕ := permutations num_boys num_boys * permutations num_girls num_girls * permutations 2 2

/-- The number of arrangements where a specific boy doesn't sit at the beginning and a specific girl doesn't sit at the end -/
def specific_positions : ℕ := permutations total_children total_children - 2 * permutations (total_children - 1) (total_children - 1) + permutations (total_children - 2) (total_children - 2)

theorem seating_arrangements :
  boys_together = 576 ∧
  girls_not_adjacent = 1440 ∧
  boys_and_girls_together = 288 ∧
  specific_positions = 3720 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l2145_214511


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l2145_214593

theorem smallest_five_digit_congruent_to_2_mod_17 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 → n ≥ 10013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_2_mod_17_l2145_214593


namespace NUMINAMATH_CALUDE_busy_schedule_starts_26th_l2145_214570

/-- Represents the reading schedule for September --/
structure ReadingSchedule where
  total_pages : ℕ
  total_days : ℕ
  busy_days : ℕ
  special_day : ℕ
  special_day_pages : ℕ
  daily_pages : ℕ

/-- Calculates the start day of the busy schedule --/
def busy_schedule_start (schedule : ReadingSchedule) : ℕ :=
  schedule.total_days - 
  ((schedule.total_pages - schedule.special_day_pages) / schedule.daily_pages) - 
  1

/-- Theorem stating that the busy schedule starts on the 26th --/
theorem busy_schedule_starts_26th (schedule : ReadingSchedule) 
  (h1 : schedule.total_pages = 600)
  (h2 : schedule.total_days = 30)
  (h3 : schedule.busy_days = 4)
  (h4 : schedule.special_day = 23)
  (h5 : schedule.special_day_pages = 100)
  (h6 : schedule.daily_pages = 20) :
  busy_schedule_start schedule = 26 := by
  sorry

#eval busy_schedule_start {
  total_pages := 600,
  total_days := 30,
  busy_days := 4,
  special_day := 23,
  special_day_pages := 100,
  daily_pages := 20
}

end NUMINAMATH_CALUDE_busy_schedule_starts_26th_l2145_214570


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2145_214529

/-- An infinite geometric sequence where any term is equal to the sum of all terms following it has a common ratio of 1/2 -/
theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h : a ≠ 0) :
  (∀ n : ℕ, a * q^n = ∑' k, a * q^(n + k + 1)) → q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2145_214529


namespace NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l2145_214569

theorem smallest_cube_divisible_by_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r → p ≠ 1 → q ≠ 1 → r ≠ 1 →
  (pqr2_cube : ℕ) → pqr2_cube = (p * q * r^2)^3 →
  (∀ m : ℕ, m^3 ∣ p^2 * q^3 * r^5 → m^3 ≥ pqr2_cube) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_divisible_by_primes_l2145_214569


namespace NUMINAMATH_CALUDE_cube_root_of_nested_roots_l2145_214517

theorem cube_root_of_nested_roots (x : ℝ) (h : x ≥ 0) :
  (x * (x * x^(1/3))^(1/2))^(1/3) = x^(5/9) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_nested_roots_l2145_214517


namespace NUMINAMATH_CALUDE_least_integral_b_value_l2145_214572

theorem least_integral_b_value : 
  (∃ b : ℤ, (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) → 
  (∃ b : ℤ, b = 2 ∧ 
   (∀ x y : ℝ, (x^2 + y^2)^2 ≤ b * (x^4 + y^4)) ∧ 
   (∀ b' : ℤ, b' < b → ∃ x y : ℝ, (x^2 + y^2)^2 > b' * (x^4 + y^4))) :=
by sorry

end NUMINAMATH_CALUDE_least_integral_b_value_l2145_214572


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2145_214591

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
  (∀ x, 3 * x = x^2 - 2) →
  (∀ x, a * x^2 + b * x + c = 0) →
  (a = 1 ∧ b = -3 ∧ c = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2145_214591


namespace NUMINAMATH_CALUDE_four_math_six_english_arrangements_l2145_214512

/-- The number of ways to arrange books and a trophy on a shelf -/
def shelfArrangements (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * 2 * (Nat.factorial mathBooks) * (Nat.factorial englishBooks)

/-- Theorem stating the number of arrangements for 4 math books and 6 English books -/
theorem four_math_six_english_arrangements :
  shelfArrangements 4 6 = 69120 := by
  sorry

#eval shelfArrangements 4 6

end NUMINAMATH_CALUDE_four_math_six_english_arrangements_l2145_214512


namespace NUMINAMATH_CALUDE_two_and_one_third_of_x_is_42_l2145_214596

theorem two_and_one_third_of_x_is_42 : ∃ x : ℚ, (7/3) * x = 42 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_and_one_third_of_x_is_42_l2145_214596


namespace NUMINAMATH_CALUDE_inequality_solution_l2145_214535

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2145_214535


namespace NUMINAMATH_CALUDE_power_function_through_point_l2145_214544

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 4 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2145_214544


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2145_214584

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 1

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  Complex.re z > 0 ∧ Complex.im z < 0

-- State the theorem
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, z_condition z ∧ in_fourth_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2145_214584


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2145_214581

theorem complex_equation_solution (z : ℂ) : 
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I → z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2145_214581


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2145_214533

/-- The complex number z defined as (2-i)^2 -/
def z : ℂ := (2 - Complex.I) ^ 2

/-- Theorem stating that z lies in the fourth quadrant of the complex plane -/
theorem z_in_fourth_quadrant : 
  z.re > 0 ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2145_214533


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2145_214588

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Main theorem
theorem arithmetic_sequence_problem
  (a : ℕ → ℝ) (d m : ℝ) (h_d : d ≠ 0)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_seq : arithmetic_sequence a d)
  (h_m : ∃ m : ℕ, a m = 8) :
  ∃ m : ℕ, m = 8 ∧ a m = 8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2145_214588


namespace NUMINAMATH_CALUDE_prasanna_speed_l2145_214503

-- Define the speeds and distance
def laxmi_speed : ℝ := 40
def total_distance : ℝ := 78
def time : ℝ := 1

-- Theorem to prove Prasanna's speed
theorem prasanna_speed : 
  ∃ (prasanna_speed : ℝ), 
    laxmi_speed * time + prasanna_speed * time = total_distance ∧ 
    prasanna_speed = 38 := by
  sorry

end NUMINAMATH_CALUDE_prasanna_speed_l2145_214503


namespace NUMINAMATH_CALUDE_smallest_three_digit_even_in_pascal_l2145_214561

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Check if a number is in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = m

/-- The smallest three-digit even number in Pascal's triangle -/
def smallestThreeDigitEvenInPascal : ℕ := 120

theorem smallest_three_digit_even_in_pascal :
  (inPascalTriangle smallestThreeDigitEvenInPascal) ∧
  (smallestThreeDigitEvenInPascal % 2 = 0) ∧
  (smallestThreeDigitEvenInPascal ≥ 100) ∧
  (smallestThreeDigitEvenInPascal < 1000) ∧
  (∀ m : ℕ, m < smallestThreeDigitEvenInPascal →
    m % 2 = 0 → m ≥ 100 → m < 1000 → ¬(inPascalTriangle m)) := by
  sorry

#check smallest_three_digit_even_in_pascal

end NUMINAMATH_CALUDE_smallest_three_digit_even_in_pascal_l2145_214561


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2145_214552

theorem smallest_number_with_given_remainders : ∃ (b : ℕ), b > 0 ∧
  b % 4 = 2 ∧ b % 3 = 2 ∧ b % 5 = 3 ∧
  ∀ (n : ℕ), n > 0 ∧ n % 4 = 2 ∧ n % 3 = 2 ∧ n % 5 = 3 → b ≤ n :=
by
  use 38
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l2145_214552


namespace NUMINAMATH_CALUDE_no_triangle_solution_l2145_214574

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  A : ℝ

-- Theorem stating that no triangle exists with the given conditions
theorem no_triangle_solution :
  ¬ ∃ (t : Triangle), t.a = 181 ∧ t.b = 209 ∧ t.A = 121 := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_solution_l2145_214574


namespace NUMINAMATH_CALUDE_solve_earnings_l2145_214536

def earnings_problem (first_month_daily : ℝ) : Prop :=
  let second_month_daily := 2 * first_month_daily
  let third_month_daily := second_month_daily
  let first_month_total := 30 * first_month_daily
  let second_month_total := 30 * second_month_daily
  let third_month_total := 15 * third_month_daily
  first_month_total + second_month_total + third_month_total = 1200

theorem solve_earnings : ∃ (x : ℝ), earnings_problem x ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_earnings_l2145_214536


namespace NUMINAMATH_CALUDE_greatest_b_for_no_negative_seven_in_range_l2145_214522

theorem greatest_b_for_no_negative_seven_in_range : 
  ∃ (b : ℤ), b = 10 ∧ 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 20 ≠ -7) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^2 + (b' : ℝ) * x + 20 = -7) :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_for_no_negative_seven_in_range_l2145_214522


namespace NUMINAMATH_CALUDE_blue_marble_difference_l2145_214528

theorem blue_marble_difference (jar1_blue jar1_green jar2_blue jar2_green : ℕ) :
  jar1_blue + jar1_green = jar2_blue + jar2_green →
  jar1_blue = 9 * jar1_green →
  jar2_blue = 8 * jar2_green →
  jar1_green + jar2_green = 95 →
  jar1_blue - jar2_blue = 5 := by
sorry

end NUMINAMATH_CALUDE_blue_marble_difference_l2145_214528


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2145_214510

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 21 / 55 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2145_214510


namespace NUMINAMATH_CALUDE_bakers_total_cost_l2145_214541

/-- Calculates the total cost of baker's ingredients --/
theorem bakers_total_cost : 
  let flour_boxes := 3
  let flour_price := 3
  let egg_trays := 3
  let egg_price := 10
  let milk_liters := 7
  let milk_price := 5
  let soda_boxes := 2
  let soda_price := 3
  
  flour_boxes * flour_price + 
  egg_trays * egg_price + 
  milk_liters * milk_price + 
  soda_boxes * soda_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_bakers_total_cost_l2145_214541


namespace NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l2145_214539

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Represents the operation of replacing a vertex with its symmetric point -/
def symmetricOperation (q : Quadrilateral) : Quadrilateral :=
  sorry

/-- Checks if a quadrilateral is permissible (sides are pairwise different and it remains convex) -/
def isPermissible (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a quadrilateral is inscribed in a circle -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if two quadrilaterals are equal -/
def areEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- Main theorem statement -/
theorem quadrilateral_symmetry_theorem (q : Quadrilateral) 
  (h_permissible : isPermissible q) :
  (∃ (q_inscribed : Quadrilateral), isInscribed q_inscribed ∧ 
    isPermissible q_inscribed ∧ 
    areEqual (symmetricOperation (symmetricOperation (symmetricOperation q_inscribed))) q_inscribed) ∧
  (areEqual (symmetricOperation (symmetricOperation (symmetricOperation 
    (symmetricOperation (symmetricOperation (symmetricOperation q)))))) q) :=
  sorry


end NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l2145_214539


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2145_214597

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = 6 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2145_214597


namespace NUMINAMATH_CALUDE_sqrt_inequality_reciprocal_sum_inequality_l2145_214521

-- Part 1
theorem sqrt_inequality (b : ℝ) (h : b ≥ 2) :
  Real.sqrt (b + 1) - Real.sqrt b < Real.sqrt (b - 1) - Real.sqrt (b - 2) :=
sorry

-- Part 2
theorem reciprocal_sum_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) (h4 : a + b = 2) :
  1 / a + 1 / b > 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_reciprocal_sum_inequality_l2145_214521


namespace NUMINAMATH_CALUDE_min_value_theorem_l2145_214558

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2 * x + y = 2) :
  ∃ (min_val : ℝ), min_val = 9/4 ∧ ∀ (z : ℝ), z = 2/(x + 1) + 1/y → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2145_214558


namespace NUMINAMATH_CALUDE_gotham_street_homes_l2145_214545

theorem gotham_street_homes (total_homes : ℚ) : 
  let termite_ridden := (1 / 3 : ℚ) * total_homes
  let collapsing := (1 / 4 : ℚ) * termite_ridden
  termite_ridden - collapsing = (1 / 4 : ℚ) * total_homes :=
by sorry

end NUMINAMATH_CALUDE_gotham_street_homes_l2145_214545


namespace NUMINAMATH_CALUDE_flower_vase_problem_l2145_214514

/-- Calculates the number of vases needed to hold flowers given the vase capacity and flower counts. -/
def vases_needed (vase_capacity : ℕ) (carnations : ℕ) (roses : ℕ) : ℕ :=
  (carnations + roses + vase_capacity - 1) / vase_capacity

/-- Proves that given 9 flowers per vase, 4 carnations, and 23 roses, 3 vases are needed. -/
theorem flower_vase_problem : vases_needed 9 4 23 = 3 := by
  sorry

#eval vases_needed 9 4 23

end NUMINAMATH_CALUDE_flower_vase_problem_l2145_214514


namespace NUMINAMATH_CALUDE_roots_equation_s_value_l2145_214543

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) :
  c^2 - n*c + 3 = 0 →
  d^2 - n*d + 3 = 0 →
  (c + 1/d)^2 - r*(c + 1/d) + s = 0 →
  (d + 1/c)^2 - r*(d + 1/c) + s = 0 →
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_s_value_l2145_214543


namespace NUMINAMATH_CALUDE_expression_simplification_l2145_214568

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2145_214568


namespace NUMINAMATH_CALUDE_green_peaches_count_l2145_214589

/-- Given a basket of fruits with the following properties:
  * There are p total fruits
  * There are r red peaches
  * The rest are green peaches
  * The sum of red peaches and twice the green peaches is 3 more than the total fruits
  Then the number of green peaches is always 3 -/
theorem green_peaches_count (p r : ℕ) (h1 : p = r + (p - r)) 
    (h2 : r + 2 * (p - r) = p + 3) : p - r = 3 := by
  sorry

#check green_peaches_count

end NUMINAMATH_CALUDE_green_peaches_count_l2145_214589


namespace NUMINAMATH_CALUDE_family_chips_consumption_l2145_214573

/-- Calculates the number of chocolate chips each family member eats given the following conditions:
  - Each batch contains 12 cookies
  - The family has 4 total people
  - Kendra made three batches
  - Each cookie contains 2 chocolate chips
  - All family members get the same number of cookies
-/
def chips_per_person (cookies_per_batch : ℕ) (family_size : ℕ) (batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_size
  cookies_per_person * chips_per_cookie

/-- Proves that given the conditions in the problem, each family member eats 18 chocolate chips -/
theorem family_chips_consumption :
  chips_per_person 12 4 3 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_family_chips_consumption_l2145_214573


namespace NUMINAMATH_CALUDE_part1_part2_l2145_214519

/-- Definition of the function f -/
def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

/-- Theorem for part 1 -/
theorem part1 (a : ℝ) : 
  f a 1 > 0 ↔ (3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3) := by sorry

/-- Theorem for part 2 -/
theorem part2 (a b : ℝ) : 
  (∀ x, f a x > b ↔ -1 < x ∧ x < 3) → 
  ((a = 3 - Real.sqrt 3 ∨ a = 3 + Real.sqrt 3) ∧ b = -3) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2145_214519


namespace NUMINAMATH_CALUDE_largest_initial_number_l2145_214557

theorem largest_initial_number : ∃ (a b c d e : ℕ), 
  189 + a + b + c + d + e = 200 ∧ 
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
  189 % a ≠ 0 ∧ 189 % b ≠ 0 ∧ 189 % c ≠ 0 ∧ 189 % d ≠ 0 ∧ 189 % e ≠ 0 ∧
  ∀ n > 189, ¬(∃ (x y z w v : ℕ), 
    n + x + y + z + w + v = 200 ∧ 
    x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
    n % x ≠ 0 ∧ n % y ≠ 0 ∧ n % z ≠ 0 ∧ n % w ≠ 0 ∧ n % v ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l2145_214557


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l2145_214538

/-- The number of ways to arrange 6 students (3 male and 3 female) with 3 female students adjacent -/
def adjacent_arrangement (n : ℕ) : ℕ := n

/-- The number of ways to arrange 6 students (3 male and 3 female) with 3 female students not adjacent -/
def not_adjacent_arrangement (n : ℕ) : ℕ := n

/-- The number of ways to arrange 6 students (3 male and 3 female) with one specific male student not at the beginning or end -/
def specific_male_arrangement (n : ℕ) : ℕ := n

theorem student_arrangement_theorem :
  adjacent_arrangement 144 = 144 ∧
  not_adjacent_arrangement 144 = 144 ∧
  specific_male_arrangement 480 = 480 :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l2145_214538


namespace NUMINAMATH_CALUDE_proposition_truth_l2145_214540

theorem proposition_truth : (∃ x₀ : ℝ, x₀ - 2 > 0) ∧ ¬(∀ x : ℝ, 2^x > x^2) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l2145_214540


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2145_214502

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2145_214502


namespace NUMINAMATH_CALUDE_discount_savings_l2145_214585

/-- Given a purchase with a 10% discount, calculate the amount saved -/
theorem discount_savings (purchase_price : ℝ) (discount_rate : ℝ) (savings : ℝ) : 
  purchase_price = 100 →
  discount_rate = 0.1 →
  savings = purchase_price * discount_rate →
  savings = 10 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_l2145_214585


namespace NUMINAMATH_CALUDE_total_boxes_sold_l2145_214506

def boxes_sold (friday saturday sunday monday : ℕ) : ℕ :=
  friday + saturday + sunday + monday

theorem total_boxes_sold :
  ∀ (friday saturday sunday monday : ℕ),
    friday = 40 →
    saturday = 2 * friday - 10 →
    sunday = saturday / 2 →
    monday = sunday + (sunday / 4 + 1) →
    boxes_sold friday saturday sunday monday = 189 :=
by sorry

end NUMINAMATH_CALUDE_total_boxes_sold_l2145_214506


namespace NUMINAMATH_CALUDE_total_rewards_distributed_l2145_214580

theorem total_rewards_distributed (students_A students_B students_C : ℕ)
  (rewards_per_student_A rewards_per_student_B rewards_per_student_C : ℕ) :
  students_A = students_B + 4 →
  students_B = students_C + 4 →
  rewards_per_student_A + 3 = rewards_per_student_B →
  rewards_per_student_B + 5 = rewards_per_student_C →
  students_A * rewards_per_student_A = students_B * rewards_per_student_B + 3 →
  students_B * rewards_per_student_B = students_C * rewards_per_student_C + 5 →
  students_A * rewards_per_student_A +
  students_B * rewards_per_student_B +
  students_C * rewards_per_student_C = 673 :=
by sorry

end NUMINAMATH_CALUDE_total_rewards_distributed_l2145_214580


namespace NUMINAMATH_CALUDE_bug_path_theorem_l2145_214546

/-- Represents a rectangular floor with a broken tile -/
structure Floor :=
  (width : ℕ)
  (length : ℕ)
  (broken_tile : ℕ × ℕ)

/-- Calculates the number of tiles a bug visits when walking diagonally across the floor -/
def tiles_visited (f : Floor) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

/-- Theorem: A bug walking diagonally across a 12x25 floor with a broken tile visits 36 tiles -/
theorem bug_path_theorem (f : Floor) 
    (h_width : f.width = 12)
    (h_length : f.length = 25)
    (h_broken : f.broken_tile = (12, 18)) : 
  tiles_visited f = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_path_theorem_l2145_214546
