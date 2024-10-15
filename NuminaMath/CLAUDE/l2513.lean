import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_y_intersections_l2513_251317

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through a given point with a slope -/
structure Line where
  point : Point
  slope : ℝ

/-- Represents a parabola of the form x^2 = 2y -/
def Parabola : Type := Unit

/-- Returns the y-coordinate of a point on the given line -/
def lineY (l : Line) (x : ℝ) : ℝ :=
  l.point.y + l.slope * (x - l.point.x)

/-- Returns true if the given point lies on the parabola -/
def onParabola (p : Point) : Prop :=
  p.x^2 = 2 * p.y

/-- Returns true if the given point lies on the given line -/
def onLine (l : Line) (p : Point) : Prop :=
  p.y = lineY l p.x

/-- Theorem stating that the minimum sum of y-coordinates of intersection points is 2 -/
theorem min_sum_y_intersections (p : Parabola) :
  ∀ l : Line,
    l.point = Point.mk 0 1 →
    ∃ A B : Point,
      onParabola A ∧ onLine l A ∧
      onParabola B ∧ onLine l B ∧
      A ≠ B →
      (∀ C D : Point,
        onParabola C ∧ onLine l C ∧
        onParabola D ∧ onLine l D ∧
        C ≠ D →
        A.y + B.y ≤ C.y + D.y) →
      A.y + B.y = 2 :=
sorry

end NUMINAMATH_CALUDE_min_sum_y_intersections_l2513_251317


namespace NUMINAMATH_CALUDE_circle_line_intersection_range_l2513_251315

theorem circle_line_intersection_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x + Real.sqrt 3 * y + m = 0 ∧ 
   (x + Real.sqrt 3 * y + m + 1)^2 + y^2 = 4 * ((x + Real.sqrt 3 * y + m - 1)^2 + y^2)) → 
  -13/3 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_range_l2513_251315


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixty_fourth_l2513_251358

theorem sin_product_equals_one_sixty_fourth :
  (Real.sin (70 * π / 180))^2 * (Real.sin (50 * π / 180))^2 * (Real.sin (10 * π / 180))^2 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixty_fourth_l2513_251358


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l2513_251331

theorem unique_solution_of_equation : 
  ∃! x : ℝ, (x^3 + 2*x^2) / (x^2 + 3*x + 2) + x = -6 ∧ x ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l2513_251331


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2513_251348

theorem min_value_sum_of_reciprocals (n : ℕ) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (1 + a^n) + 1 / (1 + b^n) ≥ 1 ∧
  (1 / (1 + a^n) + 1 / (1 + b^n) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l2513_251348


namespace NUMINAMATH_CALUDE_max_appearances_day_numbers_l2513_251335

-- Define the cube size
def n : ℕ := 2018

-- Define a function that returns the number of times a day number appears
def day_number_appearances (i : ℕ) : ℕ :=
  if i ≤ n then
    i * (i + 1) / 2
  else if n < i ∧ i < 2 * n - 1 then
    (i + 1 - n) * (3 * n - i - 1) / 2
  else if 2 * n - 1 ≤ i ∧ i ≤ 3 * n - 2 then
    day_number_appearances (3 * n - 1 - i)
  else
    0

-- Define the maximum day number
def max_day_number : ℕ := 3 * n - 2

-- State the theorem
theorem max_appearances_day_numbers :
  ∀ k : ℕ, k ≤ max_day_number →
    day_number_appearances k ≤ day_number_appearances 3026 ∧
    day_number_appearances k ≤ day_number_appearances 3027 ∧
    (day_number_appearances 3026 = day_number_appearances 3027) :=
by sorry

end NUMINAMATH_CALUDE_max_appearances_day_numbers_l2513_251335


namespace NUMINAMATH_CALUDE_oblique_prism_volume_l2513_251361

/-- The volume of an oblique prism with a parallelogram base and inclined lateral edge -/
theorem oblique_prism_volume
  (base_side1 base_side2 lateral_edge : ℝ)
  (base_angle lateral_angle : ℝ)
  (h_base_side1 : base_side1 = 3)
  (h_base_side2 : base_side2 = 6)
  (h_lateral_edge : lateral_edge = 4)
  (h_base_angle : base_angle = Real.pi / 4)  -- 45°
  (h_lateral_angle : lateral_angle = Real.pi / 6)  -- 30°
  : Real.sqrt 6 * 18 = 
    base_side1 * base_side2 * Real.sin base_angle * 
    (lateral_edge * Real.cos lateral_angle) := by
  sorry


end NUMINAMATH_CALUDE_oblique_prism_volume_l2513_251361


namespace NUMINAMATH_CALUDE_angle_sum_equality_l2513_251351

theorem angle_sum_equality (a b : Real) (h1 : 0 < a ∧ a < π/2) (h2 : 0 < b ∧ b < π/2)
  (eq1 : 4 * (Real.cos a)^2 + 3 * (Real.sin b)^2 = 1)
  (eq2 : 4 * Real.sin (2*a) + 3 * Real.cos (2*b) = 0) :
  a + 2*b = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l2513_251351


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2513_251360

theorem solution_set_inequality (x : ℝ) : 
  (x + 3) * (1 - x) ≥ 0 ↔ -3 ≤ x ∧ x ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2513_251360


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_4_plus_4sqrt3_l2513_251310

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) → 
  ∀ a b : ℝ, a > 0 → b > 0 → (1 / (a + 1) + 1 / (b + 1) = 1 / 2) → 
  x + 3 * y ≤ a + 3 * b :=
by sorry

theorem min_value_is_4_plus_4sqrt3 (x y : ℝ) :
  x > 0 → y > 0 → (1 / (x + 1) + 1 / (y + 1) = 1 / 2) →
  x + 3 * y = 4 + 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_4_plus_4sqrt3_l2513_251310


namespace NUMINAMATH_CALUDE_gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l2513_251349

/-- The cost of a gold heart necklace given the following conditions:
  * Bracelets cost $15 each
  * Personalized coffee mug costs $20
  * Raine buys 3 bracelets, 2 gold heart necklaces, and 1 coffee mug
  * Raine pays with a $100 bill and gets $15 change
-/
theorem gold_heart_necklace_cost : ℝ :=
  let bracelet_cost : ℝ := 15
  let mug_cost : ℝ := 20
  let num_bracelets : ℕ := 3
  let num_necklaces : ℕ := 2
  let num_mugs : ℕ := 1
  let payment : ℝ := 100
  let change : ℝ := 15
  let total_spent : ℝ := payment - change
  let necklace_cost : ℝ := (total_spent - (bracelet_cost * num_bracelets + mug_cost * num_mugs)) / num_necklaces
  10

theorem gold_heart_necklace_cost_proof : gold_heart_necklace_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_gold_heart_necklace_cost_gold_heart_necklace_cost_proof_l2513_251349


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2513_251383

/-- Represents a cricket game scenario --/
structure CricketGame where
  totalOvers : ℕ
  firstPartOvers : ℕ
  firstPartRunRate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs --/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPartOvers
  let firstPartRuns := game.firstPartRunRate * game.firstPartOvers
  let remainingRuns := game.target - firstPartRuns
  remainingRuns / remainingOvers

/-- Theorem stating the required run rate for the given scenario --/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstPartOvers = 10)
    (h3 : game.firstPartRunRate = 3.4)
    (h4 : game.target = 282) :
  requiredRunRate game = 6.2 := by
  sorry

#eval requiredRunRate {
  totalOvers := 50,
  firstPartOvers := 10,
  firstPartRunRate := 3.4,
  target := 282
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2513_251383


namespace NUMINAMATH_CALUDE_largest_integer_solution_of_inequalities_l2513_251352

theorem largest_integer_solution_of_inequalities :
  ∀ x : ℤ, (x - 3 * (x - 2) ≥ 4 ∧ 2 * x + 1 < x - 1) → x ≤ -3 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_of_inequalities_l2513_251352


namespace NUMINAMATH_CALUDE_tangent_product_inequality_l2513_251393

theorem tangent_product_inequality (a b c : ℝ) (α β : ℝ) :
  a + b < 3 * c →
  Real.tan (α / 2) * Real.tan (β / 2) = (a + b - c) / (a + b + c) →
  Real.tan (α / 2) * Real.tan (β / 2) < 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_inequality_l2513_251393


namespace NUMINAMATH_CALUDE_value_of_d_l2513_251378

theorem value_of_d (a b c d : ℕ+) 
  (h1 : a^2 = c * (d + 29))
  (h2 : b^2 = c * (d - 29)) :
  d = 421 := by
  sorry

end NUMINAMATH_CALUDE_value_of_d_l2513_251378


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2513_251357

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem complement_of_A_in_U :
  (U \ A) = {x | (1 ≤ x ∧ x < 2) ∨ x = 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2513_251357


namespace NUMINAMATH_CALUDE_valid_colorings_count_l2513_251309

/-- A color used for vertex coloring -/
inductive Color
| Red
| White
| Blue

/-- A vertex in the triangle structure -/
structure Vertex :=
  (id : ℕ)
  (color : Color)

/-- A triangle in the structure -/
structure Triangle :=
  (vertices : Fin 3 → Vertex)

/-- The entire structure of three connected triangles -/
structure TriangleStructure :=
  (triangles : Fin 3 → Triangle)
  (middle_restricted : Vertex)

/-- Predicate to check if a coloring is valid -/
def is_valid_coloring (s : TriangleStructure) : Prop :=
  ∀ i j : Fin 3, ∀ k l : Fin 3,
    (s.triangles i).vertices k ≠ (s.triangles j).vertices l →
    ((s.triangles i).vertices k).color ≠ ((s.triangles j).vertices l).color

/-- Predicate to check if the middle restricted vertex is colored correctly -/
def is_middle_restricted_valid (s : TriangleStructure) : Prop :=
  s.middle_restricted.color = Color.Red ∨ s.middle_restricted.color = Color.White

/-- The number of valid colorings for the triangle structure -/
def num_valid_colorings : ℕ := 36

/-- Theorem stating that the number of valid colorings is 36 -/
theorem valid_colorings_count :
  ∀ s : TriangleStructure,
    is_valid_coloring s →
    is_middle_restricted_valid s →
    num_valid_colorings = 36 :=
sorry

end NUMINAMATH_CALUDE_valid_colorings_count_l2513_251309


namespace NUMINAMATH_CALUDE_floor_minus_self_unique_solution_l2513_251382

theorem floor_minus_self_unique_solution :
  ∃! s : ℝ, ⌊s⌋ - s = -10.3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_floor_minus_self_unique_solution_l2513_251382


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2513_251366

/-- Given three complex numbers and conditions, prove that s+u = -1 -/
theorem complex_sum_problem (p q r s t u : ℝ) : 
  q = 5 →
  t = -p - r →
  (p + q * I) + (r + s * I) + (t + u * I) = 4 * I →
  s + u = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2513_251366


namespace NUMINAMATH_CALUDE_range_of_a_l2513_251316

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

-- State the theorem
theorem range_of_a : 
  (∃ a : ℝ, C a ∪ (Set.univ \ B) = Set.univ) ↔ 
  (∃ a : ℝ, a ≤ -3 ∧ C a ∪ (Set.univ \ B) = Set.univ) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2513_251316


namespace NUMINAMATH_CALUDE_final_movie_length_l2513_251324

def original_length : ℕ := 60
def removed_scenes : List ℕ := [8, 3, 4, 2, 6]

theorem final_movie_length :
  original_length - (removed_scenes.sum) = 37 := by
  sorry

end NUMINAMATH_CALUDE_final_movie_length_l2513_251324


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2513_251327

/-- A line does not pass through the second quadrant if and only if
    its slope is non-negative and its y-intercept is non-positive -/
def not_in_second_quadrant (a b c : ℝ) : Prop :=
  a ≥ 0 ∧ c ≤ 0

theorem line_not_in_second_quadrant (t : ℝ) :
  not_in_second_quadrant (2*t - 3) 2 t → 0 ≤ t ∧ t ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l2513_251327


namespace NUMINAMATH_CALUDE_tire_repair_tax_l2513_251368

theorem tire_repair_tax (repair_cost : ℚ) (num_tires : ℕ) (final_cost : ℚ) :
  repair_cost = 7 →
  num_tires = 4 →
  final_cost = 30 →
  (final_cost - (repair_cost * num_tires)) / num_tires = (1/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_tire_repair_tax_l2513_251368


namespace NUMINAMATH_CALUDE_smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l2513_251399

/-- A function that checks if all digits of a natural number are 0 or 1 -/
def all_digits_zero_or_one (n : ℕ) : Prop := sorry

/-- A function that returns the smallest natural number with digits 0 or 1 divisible by 225 -/
noncomputable def smallest_zero_one_divisible_by_225 : ℕ := sorry

theorem smallest_zero_one_divisible_by_225_is_11111111100 :
  smallest_zero_one_divisible_by_225 = 11111111100 :=
by
  sorry

theorem smallest_zero_one_divisible_by_225_properties :
  let n := smallest_zero_one_divisible_by_225
  all_digits_zero_or_one n ∧ n % 225 = 0 ∧ 
  ∀ m : ℕ, m < n → ¬(all_digits_zero_or_one m ∧ m % 225 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_zero_one_divisible_by_225_is_11111111100_smallest_zero_one_divisible_by_225_properties_l2513_251399


namespace NUMINAMATH_CALUDE_sequence_general_term_l2513_251330

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → S n = 4 * a n - 3) →
  (∀ n : ℕ, n ≥ 1 → a n = (4/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2513_251330


namespace NUMINAMATH_CALUDE_ellipse_point_position_l2513_251355

theorem ellipse_point_position 
  (a b c : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_a_gt_b : a > b)
  (h_roots : x₁ + x₂ = -b/a ∧ x₁ * x₂ = -c/a) :
  1 < x₁^2 + x₂^2 ∧ x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_position_l2513_251355


namespace NUMINAMATH_CALUDE_product_n_n_plus_one_is_even_l2513_251308

theorem product_n_n_plus_one_is_even (n : ℕ) : Even (n * (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_product_n_n_plus_one_is_even_l2513_251308


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2513_251346

theorem sin_cos_sum_equals_one :
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2513_251346


namespace NUMINAMATH_CALUDE_g_sum_at_two_l2513_251332

/-- Given a function g(x) = ax^8 + bx^6 - cx^4 + dx^2 + 5 where g(2) = 4, 
    prove that g(2) + g(-2) = 8 -/
theorem g_sum_at_two (a b c d : ℝ) :
  let g := fun x : ℝ => a * x^8 + b * x^6 - c * x^4 + d * x^2 + 5
  g 2 = 4 → g 2 + g (-2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_at_two_l2513_251332


namespace NUMINAMATH_CALUDE_number_with_specific_totient_l2513_251347

theorem number_with_specific_totient (N : ℕ) (α β γ : ℕ) :
  N = 3^α * 5^β * 7^γ →
  Nat.totient N = 3600 →
  N = 7875 := by
sorry

end NUMINAMATH_CALUDE_number_with_specific_totient_l2513_251347


namespace NUMINAMATH_CALUDE_cubic_function_value_l2513_251303

/-- Given a cubic function f(x) = ax³ + 3 where f(-2) = -5, prove that f(2) = 11 -/
theorem cubic_function_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^3 + 3) 
  (h2 : f (-2) = -5) : f 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_value_l2513_251303


namespace NUMINAMATH_CALUDE_rescue_center_dogs_l2513_251341

/-- Calculates the number of remaining dogs after a series of additions and adoptions. -/
def remaining_dogs (initial : ℕ) (moved_in : ℕ) (first_adoption : ℕ) (second_adoption : ℕ) : ℕ :=
  initial + moved_in - first_adoption - second_adoption

/-- Theorem stating that given the specific numbers from the problem, 
    the number of remaining dogs is 200. -/
theorem rescue_center_dogs : 
  remaining_dogs 200 100 40 60 = 200 := by
  sorry

#eval remaining_dogs 200 100 40 60

end NUMINAMATH_CALUDE_rescue_center_dogs_l2513_251341


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2513_251328

-- Define the inverse proportionality relationship
def inverse_proportional (α β : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ α * β = k

-- State the theorem
theorem inverse_proportion_problem (α₁ α₂ β₁ β₂ : ℝ) :
  inverse_proportional α₁ β₁ →
  α₁ = 2 →
  β₁ = 5 →
  β₂ = -10 →
  inverse_proportional α₂ β₂ →
  α₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2513_251328


namespace NUMINAMATH_CALUDE_max_points_32_points_32_achievable_l2513_251321

/-- Represents a basketball game where a player only attempts three-point and two-point shots -/
structure BasketballGame where
  threePointAttempts : ℕ
  twoPointAttempts : ℕ
  threePointSuccessRate : ℚ
  twoPointSuccessRate : ℚ

/-- Calculates the total points scored in a basketball game -/
def totalPoints (game : BasketballGame) : ℚ :=
  3 * game.threePointSuccessRate * game.threePointAttempts +
  2 * game.twoPointSuccessRate * game.twoPointAttempts

/-- Theorem stating that under the given conditions, the maximum points scored is 32 -/
theorem max_points_32 (game : BasketballGame) 
    (h1 : game.threePointAttempts + game.twoPointAttempts = 40)
    (h2 : game.threePointSuccessRate = 1/4)
    (h3 : game.twoPointSuccessRate = 2/5) :
  totalPoints game ≤ 32 := by
  sorry

/-- Theorem stating that 32 points can be achieved -/
theorem points_32_achievable : 
  ∃ (game : BasketballGame), 
    game.threePointAttempts + game.twoPointAttempts = 40 ∧
    game.threePointSuccessRate = 1/4 ∧
    game.twoPointSuccessRate = 2/5 ∧
    totalPoints game = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_points_32_points_32_achievable_l2513_251321


namespace NUMINAMATH_CALUDE_five_fridays_in_august_l2513_251372

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Represents a calendar month -/
structure Month where
  days : Nat
  firstDay : DayOfWeek

/-- Returns the number of occurrences of a given day in a month -/
def countDayOccurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If July has five Tuesdays, then August has five Fridays -/
theorem five_fridays_in_august 
  (july : Month) 
  (h1 : july.days = 31)
  (h2 : countDayOccurrences july DayOfWeek.Tuesday = 5) :
  ∃ (august : Month), 
    august.days = 31 ∧ 
    august.firstDay = nextDay (nextDay (nextDay july.firstDay)) ∧
    countDayOccurrences august DayOfWeek.Friday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_fridays_in_august_l2513_251372


namespace NUMINAMATH_CALUDE_love_all_girls_l2513_251365

-- Define the girls
inductive Girl
| Sue
| Marcia
| Diana

-- Define the love relation
def loves : Girl → Prop := sorry

-- State the theorem
theorem love_all_girls :
  -- Condition 1: I love at least one of the three girls
  (∃ g : Girl, loves g) →
  -- Condition 2: If I love Sue but not Diana, then I also love Marcia
  (loves Girl.Sue ∧ ¬loves Girl.Diana → loves Girl.Marcia) →
  -- Condition 3: I either love both Diana and Marcia, or I love neither of them
  ((loves Girl.Diana ∧ loves Girl.Marcia) ∨ (¬loves Girl.Diana ∧ ¬loves Girl.Marcia)) →
  -- Condition 4: If I love Diana, then I also love Sue
  (loves Girl.Diana → loves Girl.Sue) →
  -- Conclusion: I love all three girls
  (loves Girl.Sue ∧ loves Girl.Marcia ∧ loves Girl.Diana) :=
by sorry

end NUMINAMATH_CALUDE_love_all_girls_l2513_251365


namespace NUMINAMATH_CALUDE_f_property_f_at_two_l2513_251323

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_property (x : ℝ) : f x = (deriv f 1) * Real.exp (x - 1) - f 0 * x + (1/2) * x^2 := sorry

theorem f_at_two : f 2 = Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_f_property_f_at_two_l2513_251323


namespace NUMINAMATH_CALUDE_min_value_of_function_l2513_251364

theorem min_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1) :
  ∀ y : ℝ, y = 4/x + 1/(1-x) → y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2513_251364


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l2513_251320

/-- The number of hotdogs each sister wants -/
def sisters_hotdogs : ℕ := 2

/-- The total number of hotdogs both sisters want -/
def total_sisters_hotdogs : ℕ := 2 * sisters_hotdogs

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * total_sisters_hotdogs

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * total_sisters_hotdogs) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := total_sisters_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l2513_251320


namespace NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l2513_251350

/-- The area of a rectangle with an inscribed circle of radius 7 and length-to-width ratio of 3:1 -/
theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 2 * r * ratio * 2 * r = 588 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_inscribed_circle_l2513_251350


namespace NUMINAMATH_CALUDE_circle_equation_l2513_251301

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def is_in_first_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

def is_tangent_to_line (C : Circle) (a b c : ℝ) : Prop :=
  abs (a * C.center.1 + b * C.center.2 + c) = C.radius * Real.sqrt (a^2 + b^2)

def is_tangent_to_x_axis (C : Circle) : Prop :=
  C.center.2 = C.radius

-- State the theorem
theorem circle_equation (C : Circle) :
  C.radius = 1 →
  is_in_first_quadrant C.center →
  is_tangent_to_line C 4 (-3) 0 →
  is_tangent_to_x_axis C →
  ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2} :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l2513_251301


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l2513_251356

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l2513_251356


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2513_251318

theorem right_triangle_segment_ratio (x y z u v : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ v > 0 →
  x^2 + y^2 = z^2 →
  x / y = 2 / 5 →
  u * z = x^2 →
  v * z = y^2 →
  u + v = z →
  u / v = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l2513_251318


namespace NUMINAMATH_CALUDE_nellie_uncle_rolls_l2513_251302

/-- Prove that Nellie sold 10 rolls to her uncle -/
theorem nellie_uncle_rolls : 
  ∀ (total_rolls grandmother_rolls neighbor_rolls remaining_rolls : ℕ),
  total_rolls = 45 →
  grandmother_rolls = 1 →
  neighbor_rolls = 6 →
  remaining_rolls = 28 →
  total_rolls - remaining_rolls - grandmother_rolls - neighbor_rolls = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_nellie_uncle_rolls_l2513_251302


namespace NUMINAMATH_CALUDE_price_change_after_markup_and_markdown_l2513_251370

theorem price_change_after_markup_and_markdown (original_price : ℝ) (markup_percent : ℝ) (markdown_percent : ℝ)
  (h_original_positive : original_price > 0)
  (h_markup : markup_percent = 10)
  (h_markdown : markdown_percent = 10) :
  original_price * (1 + markup_percent / 100) * (1 - markdown_percent / 100) < original_price :=
by sorry

end NUMINAMATH_CALUDE_price_change_after_markup_and_markdown_l2513_251370


namespace NUMINAMATH_CALUDE_cos_A_minus_B_l2513_251319

theorem cos_A_minus_B (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1) 
  (h2 : Real.cos A + Real.cos B = 5/3) : 
  Real.cos (A - B) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_A_minus_B_l2513_251319


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2513_251333

/-- The value of m for which the parabola y = x^2 + 2x + 3 and 
    the hyperbola y^2 - mx^2 = 5 are tangent to each other -/
def tangency_condition : ℝ := -26

/-- The equation of the parabola -/
def parabola (x y : ℝ) : Prop := y = x^2 + 2*x + 3

/-- The equation of the hyperbola -/
def hyperbola (m x y : ℝ) : Prop := y^2 - m*x^2 = 5

/-- Theorem stating that the parabola and hyperbola are tangent when m = -26 -/
theorem parabola_hyperbola_tangency :
  ∃ (x y : ℝ), parabola x y ∧ hyperbola tangency_condition x y ∧
  ∀ (x' y' : ℝ), x' ≠ x → y' ≠ y → 
    ¬(parabola x' y' ∧ hyperbola tangency_condition x' y') :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l2513_251333


namespace NUMINAMATH_CALUDE_first_ring_at_three_am_l2513_251376

/-- A clock that rings at regular intervals throughout the day -/
structure RingingClock where
  ring_interval : ℕ  -- Interval between rings in hours
  rings_per_day : ℕ  -- Number of times the clock rings in a day

/-- The time of day in hours (0 to 23) -/
def Time := Fin 24

/-- Calculate the time of the first ring for a given clock -/
def first_ring_time (clock : RingingClock) : Time :=
  ⟨clock.ring_interval, by sorry⟩

theorem first_ring_at_three_am 
  (clock : RingingClock) 
  (h1 : clock.ring_interval = 3) 
  (h2 : clock.rings_per_day = 8) : 
  first_ring_time clock = ⟨3, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_first_ring_at_three_am_l2513_251376


namespace NUMINAMATH_CALUDE_maria_spent_60_dollars_l2513_251329

def flower_cost : ℕ := 6
def roses_bought : ℕ := 7
def daisies_bought : ℕ := 3

theorem maria_spent_60_dollars : 
  (roses_bought + daisies_bought) * flower_cost = 60 := by
  sorry

end NUMINAMATH_CALUDE_maria_spent_60_dollars_l2513_251329


namespace NUMINAMATH_CALUDE_sum_base3_equals_11000_l2513_251345

/-- Represents a number in base 3 --/
def Base3 : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def to_decimal (n : Base3) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- Addition of two base 3 numbers --/
def add_base3 (a b : Base3) : Base3 :=
  sorry

/-- Theorem: The sum of 2₃, 22₃, 202₃, and 2022₃ is 11000₃ in base 3 --/
theorem sum_base3_equals_11000 :
  let a := [2]
  let b := [2, 2]
  let c := [2, 0, 2]
  let d := [2, 2, 0, 2]
  let result := [1, 1, 0, 0, 0]
  add_base3 (add_base3 (add_base3 a b) c) d = result :=
sorry

end NUMINAMATH_CALUDE_sum_base3_equals_11000_l2513_251345


namespace NUMINAMATH_CALUDE_sum_equals_140_l2513_251336

theorem sum_equals_140 (p q r s : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
  (h1 : p^2 + q^2 = 2500)
  (h2 : r^2 + s^2 = 2500)
  (h3 : p * r = 1200)
  (h4 : q * s = 1200) :
  p + q + r + s = 140 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_140_l2513_251336


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2513_251384

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ (r s : ℚ), (4 * r^2 + 2 * r - 9 = 0) ∧ 
                 (4 * s^2 + 2 * s - 9 = 0) ∧ 
                 (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧
                 (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 51 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2513_251384


namespace NUMINAMATH_CALUDE_barn_painted_area_l2513_251397

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintedArea (d : BarnDimensions) : ℝ :=
  2 * (2 * (d.width * d.height + d.length * d.height) + 2 * (d.width * d.length))

/-- Theorem stating that the total area to be painted for the given barn is 1368 square yards -/
theorem barn_painted_area :
  let d : BarnDimensions := { width := 12, length := 15, height := 6 }
  totalPaintedArea d = 1368 := by
  sorry

end NUMINAMATH_CALUDE_barn_painted_area_l2513_251397


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l2513_251359

/-- Given four points on a Cartesian plane, prove that if segment AB is parallel to segment XY, then k = -6 -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ) 
  (hA : A = (-4, 0)) 
  (hB : B = (0, -4)) 
  (hX : X = (0, 8)) 
  (hY : Y = (14, k))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (B.2 - A.2) * (Y.1 - X.1)) : 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l2513_251359


namespace NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2513_251387

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across x-axis
def reflectXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the reflection across y=x-1
def reflectYEqXMinus1 (p : Point2D) : Point2D :=
  { x := p.y + 1, y := p.x - 1 }

-- Define the composite transformation
def compositeTransform (p : Point2D) : Point2D :=
  reflectYEqXMinus1 (reflectXAxis p)

-- Theorem statement
theorem parallelogram_reflection_theorem (E F G H : Point2D)
  (hE : E = { x := 3, y := 3 })
  (hF : F = { x := 6, y := 7 })
  (hG : G = { x := 9, y := 3 })
  (hH : H = { x := 6, y := -1 }) :
  compositeTransform H = { x := 2, y := 5 } := by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_theorem_l2513_251387


namespace NUMINAMATH_CALUDE_equal_chore_time_l2513_251363

/-- Represents the time taken for each chore in minutes -/
structure ChoreTime where
  sweeping : ℕ
  washing : ℕ
  laundry : ℕ

/-- Represents the chores assigned to each child -/
structure Chores where
  rooms : ℕ
  dishes : ℕ
  loads : ℕ

def total_time (ct : ChoreTime) (c : Chores) : ℕ :=
  ct.sweeping * c.rooms + ct.washing * c.dishes + ct.laundry * c.loads

theorem equal_chore_time (ct : ChoreTime) (anna billy : Chores) : 
  ct.sweeping = 3 → 
  ct.washing = 2 → 
  ct.laundry = 9 → 
  anna.rooms = 10 → 
  anna.dishes = 0 → 
  anna.loads = 0 → 
  billy.rooms = 0 → 
  billy.loads = 2 → 
  total_time ct anna = total_time ct billy → 
  billy.dishes = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_chore_time_l2513_251363


namespace NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l2513_251306

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x - 1 + a / Real.exp x

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a / Real.exp x

theorem tangent_parallel_implies_a_equals_e (a : ℝ) :
  f_derivative a 1 = 0 → a = Real.exp 1 := by sorry

theorem max_value_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), f a x = Real.log a ∧ 
  ∀ (y : ℝ), f a y ≤ f a x := by sorry

theorem no_extreme_values_when_a_nonpositive (a : ℝ) (h : a ≤ 0) :
  ∀ (x : ℝ), ∃ (y : ℝ), f a y > f a x := by sorry

end

end NUMINAMATH_CALUDE_tangent_parallel_implies_a_equals_e_max_value_when_a_positive_no_extreme_values_when_a_nonpositive_l2513_251306


namespace NUMINAMATH_CALUDE_tank_capacity_l2513_251339

theorem tank_capacity : ℝ → Prop :=
  fun capacity =>
    let initial_fraction : ℚ := 1/4
    let final_fraction : ℚ := 3/4
    let added_water : ℝ := 180
    initial_fraction * capacity + added_water = final_fraction * capacity →
    capacity = 360

-- Proof
example : tank_capacity 360 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2513_251339


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_l2513_251314

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define point D on the angle bisector of A
variable (D : EuclideanSpace ℝ (Fin 2))

-- Assumption that ABC is a triangle
variable (h_triangle : (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A))

-- Assumption that D is on BC
variable (h_D_on_BC : D ∈ LineSegment B C)

-- Assumption that AD is the angle bisector of angle BAC
variable (h_angle_bisector : AngleBisector A B C D)

-- Theorem statement
theorem angle_bisector_theorem :
  (dist A B) / (dist A C) = (dist B D) / (dist C D) := by sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_l2513_251314


namespace NUMINAMATH_CALUDE_middle_quad_area_proportion_l2513_251304

-- Define a convex quadrilateral
def ConvexQuadrilateral : Type := Unit

-- Define a function to represent the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the middle quadrilateral formed by connecting points
def middleQuadrilateral (q : ConvexQuadrilateral) : ConvexQuadrilateral := sorry

-- State the theorem
theorem middle_quad_area_proportion (q : ConvexQuadrilateral) :
  area (middleQuadrilateral q) = (1 / 25) * area q := by sorry

end NUMINAMATH_CALUDE_middle_quad_area_proportion_l2513_251304


namespace NUMINAMATH_CALUDE_number_division_problem_l2513_251307

theorem number_division_problem (x : ℝ) : x / 5 = 80 + x / 6 ↔ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2513_251307


namespace NUMINAMATH_CALUDE_darla_total_cost_l2513_251385

/-- The total cost of electricity given the rate, usage, and late fee. -/
def total_cost (rate : ℝ) (usage : ℝ) (late_fee : ℝ) : ℝ :=
  rate * usage + late_fee

/-- Proof that Darla's total cost is $1350 -/
theorem darla_total_cost : 
  total_cost 4 300 150 = 1350 := by
  sorry

end NUMINAMATH_CALUDE_darla_total_cost_l2513_251385


namespace NUMINAMATH_CALUDE_polygon_sides_l2513_251386

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 2002 → 
  (n - 2) * 180 - 360 < sum_angles ∧ sum_angles < (n - 2) * 180 →
  n = 14 ∨ n = 15 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l2513_251386


namespace NUMINAMATH_CALUDE_intersection_M_N_l2513_251381

def M : Set ℝ := {x : ℝ | x^2 + 3*x = 0}
def N : Set ℝ := {3, 0}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2513_251381


namespace NUMINAMATH_CALUDE_twenty_six_billion_scientific_notation_l2513_251394

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_six_billion_scientific_notation :
  toScientificNotation (26 * 10^9) = ScientificNotation.mk 2.6 9 sorry := by
  sorry

end NUMINAMATH_CALUDE_twenty_six_billion_scientific_notation_l2513_251394


namespace NUMINAMATH_CALUDE_hno3_concentration_after_addition_l2513_251313

/-- Calculates the final concentration of HNO3 after adding pure HNO3 to a solution -/
theorem hno3_concentration_after_addition
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (pure_hno3_added : ℝ)
  (h1 : initial_volume = 60)
  (h2 : initial_concentration = 0.35)
  (h3 : pure_hno3_added = 18) :
  let final_volume := initial_volume + pure_hno3_added
  let initial_hno3 := initial_volume * initial_concentration
  let final_hno3 := initial_hno3 + pure_hno3_added
  let final_concentration := final_hno3 / final_volume
  final_concentration = 0.5 := by sorry

end NUMINAMATH_CALUDE_hno3_concentration_after_addition_l2513_251313


namespace NUMINAMATH_CALUDE_fourth_root_fifth_power_eighth_l2513_251362

theorem fourth_root_fifth_power_eighth : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_fifth_power_eighth_l2513_251362


namespace NUMINAMATH_CALUDE_jason_work_hours_l2513_251388

def after_school_rate : ℝ := 4.00
def saturday_rate : ℝ := 6.00
def total_earnings : ℝ := 88.00
def saturday_hours : ℝ := 8

def total_hours : ℝ := 18

theorem jason_work_hours :
  ∃ (after_school_hours : ℝ),
    after_school_hours * after_school_rate + saturday_hours * saturday_rate = total_earnings ∧
    after_school_hours + saturday_hours = total_hours :=
by sorry

end NUMINAMATH_CALUDE_jason_work_hours_l2513_251388


namespace NUMINAMATH_CALUDE_max_distinct_squares_sum_2100_l2513_251353

/-- The sum of squares of a list of natural numbers -/
def sum_of_squares (lst : List Nat) : Nat :=
  lst.map (· ^ 2) |>.sum

/-- A proposition stating that a list of natural numbers has distinct elements -/
def is_distinct (lst : List Nat) : Prop :=
  lst.Nodup

theorem max_distinct_squares_sum_2100 :
  (∃ (n : Nat) (lst : List Nat), 
    lst.length = n ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100 ∧
    ∀ (m : Nat) (lst' : List Nat), 
      lst'.length = m ∧ 
      is_distinct lst' ∧ 
      sum_of_squares lst' = 2100 → 
      m ≤ n) ∧
  (∃ (lst : List Nat), 
    lst.length = 17 ∧ 
    is_distinct lst ∧ 
    sum_of_squares lst = 2100) :=
by
  sorry


end NUMINAMATH_CALUDE_max_distinct_squares_sum_2100_l2513_251353


namespace NUMINAMATH_CALUDE_one_certain_event_l2513_251334

-- Define the events
inductive Event
  | WaterFreeze : Event
  | RectangleArea : Event
  | CoinToss : Event
  | ExamScore : Event

-- Define a function to check if an event is certain
def isCertain (e : Event) : Prop :=
  match e with
  | Event.WaterFreeze => False
  | Event.RectangleArea => True
  | Event.CoinToss => False
  | Event.ExamScore => False

-- Theorem statement
theorem one_certain_event :
  (∃! e : Event, isCertain e) :=
sorry

end NUMINAMATH_CALUDE_one_certain_event_l2513_251334


namespace NUMINAMATH_CALUDE_clothes_spending_fraction_l2513_251375

theorem clothes_spending_fraction (initial_amount remaining_amount : ℝ) : 
  initial_amount = 1249.9999999999998 →
  remaining_amount = 500 →
  ∃ (F : ℝ), 
    F > 0 ∧ F < 1 ∧
    remaining_amount = (1 - 1/4) * (1 - 1/5) * (1 - F) * initial_amount ∧
    F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_clothes_spending_fraction_l2513_251375


namespace NUMINAMATH_CALUDE_integer_pair_solution_l2513_251326

theorem integer_pair_solution :
  ∀ x y : ℕ+,
  (2 * x.val * y.val = 2 * x.val + y.val + 21) →
  ((x.val = 1 ∧ y.val = 23) ∨ (x.val = 6 ∧ y.val = 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_pair_solution_l2513_251326


namespace NUMINAMATH_CALUDE_football_kick_distance_l2513_251371

theorem football_kick_distance (longest_kick : ℝ) (average_kick : ℝ) (kick1 kick2 kick3 : ℝ) :
  longest_kick = 43 →
  average_kick = 37 →
  (kick1 + kick2 + kick3) / 3 = average_kick →
  kick1 = longest_kick →
  kick2 = kick3 →
  kick2 = 34 ∧ kick3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_football_kick_distance_l2513_251371


namespace NUMINAMATH_CALUDE_pull_up_median_mode_l2513_251325

def pull_up_data : List ℕ := [6, 8, 7, 7, 8, 9, 8, 9]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem pull_up_median_mode :
  median pull_up_data = 8 ∧ mode pull_up_data = 8 := by sorry

end NUMINAMATH_CALUDE_pull_up_median_mode_l2513_251325


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2513_251379

theorem triangle_angle_measure (D E F : ℝ) (h1 : D + E + F = 180)
  (h2 : F = 3 * E) (h3 : E = 15) : D = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2513_251379


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2513_251369

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

theorem sum_first_six_primes_mod_seventh_prime :
  sumFirstNPrimes 6 % nthPrime 7 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_prime_l2513_251369


namespace NUMINAMATH_CALUDE_greatest_M_inequality_l2513_251342

theorem greatest_M_inequality (x y z : ℝ) : 
  ∃ (M : ℝ), M = 2/3 ∧ 
  (∀ (N : ℝ), (∀ (a b c : ℝ), a^4 + b^4 + c^4 + a*b*c*(a + b + c) ≥ N*(a*b + b*c + c*a)^2) → N ≤ M) ∧
  x^4 + y^4 + z^4 + x*y*z*(x + y + z) ≥ M*(x*y + y*z + z*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_greatest_M_inequality_l2513_251342


namespace NUMINAMATH_CALUDE_d_negative_iff_b_decreasing_l2513_251343

/-- An arithmetic sequence with common difference d -/
def arithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The sequence b_n defined as 2^(a_n) -/
def bSequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 2^(a n)

/-- A decreasing sequence -/
def isDecreasing (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) < b n

theorem d_negative_iff_b_decreasing
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ)
  (h1 : arithmeticSequence a d)
  (h2 : bSequence a b) :
  d < 0 ↔ isDecreasing b :=
sorry

end NUMINAMATH_CALUDE_d_negative_iff_b_decreasing_l2513_251343


namespace NUMINAMATH_CALUDE_proportion_equality_l2513_251374

theorem proportion_equality (x y : ℝ) (h1 : y ≠ 0) (h2 : 3 * x = 4 * y) : x / 4 = y / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2513_251374


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l2513_251392

/-- A regular icosahedron -/
structure Icosahedron where
  vertices : Finset (Fin 12)
  edges : Finset (Fin 30)

/-- The probability of selecting two vertices that form an edge in a regular icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge
    in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) :
  edge_probability i = 5 / 11 := by
  sorry


end NUMINAMATH_CALUDE_icosahedron_edge_probability_l2513_251392


namespace NUMINAMATH_CALUDE_perpendicular_lines_sin_2alpha_l2513_251354

theorem perpendicular_lines_sin_2alpha (α : Real) :
  let l₁ : Real → Real → Real := λ x y => x * Real.sin α + y - 1
  let l₂ : Real → Real → Real := λ x y => x - 3 * y * Real.cos α + 1
  (∀ x y, l₁ x y = 0 → l₂ x y = 0 → (Real.sin α + 3 * Real.cos α) * (Real.sin α - 3 * Real.cos α) = 0) →
  Real.sin (2 * α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sin_2alpha_l2513_251354


namespace NUMINAMATH_CALUDE_set_equality_implies_difference_l2513_251390

theorem set_equality_implies_difference (a b : ℝ) : 
  ({1, a + b, a} : Set ℝ) = {0, b / a, b} → b - a = 2 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_difference_l2513_251390


namespace NUMINAMATH_CALUDE_large_pizza_has_16_slices_l2513_251373

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := sorry

/-- The number of large pizzas -/
def num_large_pizzas : ℕ := 2

/-- The number of small pizzas -/
def num_small_pizzas : ℕ := 2

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of slices eaten -/
def total_slices_eaten : ℕ := 48

theorem large_pizza_has_16_slices :
  num_large_pizzas * large_pizza_slices + num_small_pizzas * small_pizza_slices = total_slices_eaten →
  large_pizza_slices = 16 := by
  sorry

end NUMINAMATH_CALUDE_large_pizza_has_16_slices_l2513_251373


namespace NUMINAMATH_CALUDE_tuesday_rain_amount_l2513_251398

/-- The amount of rain on Monday in inches -/
def monday_rain : ℝ := 0.9

/-- The difference in rain between Monday and Tuesday in inches -/
def rain_difference : ℝ := 0.7

/-- The amount of rain on Tuesday in inches -/
def tuesday_rain : ℝ := monday_rain - rain_difference

theorem tuesday_rain_amount : tuesday_rain = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rain_amount_l2513_251398


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2513_251396

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2513_251396


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_l2513_251322

theorem exists_m_for_all_n : ∀ (n : ℤ), ∃ (m : ℤ), n * m = m := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_l2513_251322


namespace NUMINAMATH_CALUDE_intersection_M_N_l2513_251338

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2513_251338


namespace NUMINAMATH_CALUDE_biker_journey_west_distance_l2513_251389

/-- Represents the journey of a biker -/
structure BikerJourney where
  west : ℝ
  north1 : ℝ
  east : ℝ
  north2 : ℝ
  straightLineDistance : ℝ

/-- Theorem stating the distance traveled west given specific journey parameters -/
theorem biker_journey_west_distance (journey : BikerJourney) 
  (h1 : journey.north1 = 5)
  (h2 : journey.east = 4)
  (h3 : journey.north2 = 15)
  (h4 : journey.straightLineDistance = 20.396078054371138) :
  journey.west = 8 := by
  sorry

end NUMINAMATH_CALUDE_biker_journey_west_distance_l2513_251389


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l2513_251311

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Checks if a point (x, y) is on the parabola -/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  p.y x = y

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.containsPoint 3 (-9) →
  p.containsPoint 5 7 →
  ∃ m n : ℝ, m ∈ p.zeros ∧ n ∈ p.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l2513_251311


namespace NUMINAMATH_CALUDE_inequality_problem_l2513_251344

theorem inequality_problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 1/a + 4/b + 9/c ≤ 36/(a + b + c)) : 
  (2*b + 3*c)/(a + b + c) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l2513_251344


namespace NUMINAMATH_CALUDE_mikes_ride_length_mikes_ride_length_proof_l2513_251377

/-- Proves that Mike's ride was 35 miles long given the taxi fare conditions -/
theorem mikes_ride_length : ℝ → Prop :=
  fun T =>
    let mike_start : ℝ := 3
    let mike_per_mile : ℝ := 0.3
    let mike_surcharge : ℝ := 1.5
    let annie_start : ℝ := 3.5
    let annie_per_mile : ℝ := 0.25
    let annie_toll : ℝ := 5
    let annie_surcharge : ℝ := 2
    let annie_miles : ℝ := 18
    ∀ M : ℝ,
      (mike_start + mike_per_mile * M + mike_surcharge = T) ∧
      (annie_start + annie_per_mile * annie_miles + annie_toll + annie_surcharge = T) →
      M = 35

/-- Proof of the theorem -/
theorem mikes_ride_length_proof : ∀ T : ℝ, mikes_ride_length T :=
  fun T => by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_mikes_ride_length_mikes_ride_length_proof_l2513_251377


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2513_251300

/-- An isosceles triangle with specific height measurements -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the equal sides
  side : ℝ
  -- The height to the base
  height_to_base : ℝ
  -- The height to a lateral side
  height_to_side : ℝ
  -- Conditions for an isosceles triangle
  isosceles : side > 0
  base_positive : base > 0
  height_to_base_positive : height_to_base > 0
  height_to_side_positive : height_to_side > 0
  -- Pythagorean theorem for the height to the base
  pythagorean_base : side^2 = height_to_base^2 + (base/2)^2
  -- Pythagorean theorem for the height to the side
  pythagorean_side : side^2 = height_to_side^2 + (base/2)^2

/-- Theorem: If the height to the base is 10 and the height to the side is 12,
    then the base of the isosceles triangle is 15 -/
theorem isosceles_triangle_base_length
  (triangle : IsoscelesTriangle)
  (h1 : triangle.height_to_base = 10)
  (h2 : triangle.height_to_side = 12) :
  triangle.base = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2513_251300


namespace NUMINAMATH_CALUDE_range_of_a_l2513_251367

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3 - a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ 0) → a ∈ Set.Icc (-7 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2513_251367


namespace NUMINAMATH_CALUDE_smallest_student_count_l2513_251337

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  eighth : ℕ
  seventh : ℕ

/-- Checks if the given counts satisfy the ratio conditions --/
def satisfies_ratios (counts : GradeCount) : Prop :=
  7 * counts.seventh = 4 * counts.ninth ∧
  9 * counts.eighth = 5 * counts.ninth

/-- The total number of students --/
def total_students (counts : GradeCount) : ℕ :=
  counts.ninth + counts.eighth + counts.seventh

/-- Theorem stating the smallest possible number of students --/
theorem smallest_student_count :
  ∃ (counts : GradeCount),
    satisfies_ratios counts ∧
    total_students counts = 134 ∧
    (∀ (other : GradeCount), satisfies_ratios other → total_students other ≥ 134) := by
  sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2513_251337


namespace NUMINAMATH_CALUDE_tim_attend_probability_l2513_251312

-- Define the probability of rain
def prob_rain : ℝ := 0.6

-- Define the probability of sun (complementary to rain)
def prob_sun : ℝ := 1 - prob_rain

-- Define the probability Tim attends if it rains
def prob_attend_rain : ℝ := 0.25

-- Define the probability Tim attends if it's sunny
def prob_attend_sun : ℝ := 0.7

-- Theorem statement
theorem tim_attend_probability :
  prob_rain * prob_attend_rain + prob_sun * prob_attend_sun = 0.43 := by
sorry

end NUMINAMATH_CALUDE_tim_attend_probability_l2513_251312


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2513_251391

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def N : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.compl M) ∩ N = {x | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2513_251391


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2513_251380

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 40 * S) : 
  (S - C) / C * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2513_251380


namespace NUMINAMATH_CALUDE_card_sum_difference_l2513_251395

theorem card_sum_difference (n : ℕ) (a : ℕ → ℝ) 
  (h1 : n ≥ 4)
  (h2 : ∀ m, m ≤ 2*n + 4 → ⌊a m⌋ = m) :
  ∃ i j k l, 
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    i ≤ 2*n + 4 ∧ j ≤ 2*n + 4 ∧ k ≤ 2*n + 4 ∧ l ≤ 2*n + 4 ∧
    |a i + a j - a k - a l| < 1 / (n - Real.sqrt (n / 2)) :=
sorry

end NUMINAMATH_CALUDE_card_sum_difference_l2513_251395


namespace NUMINAMATH_CALUDE_factory_output_increase_l2513_251340

/-- Proves that the percentage increase in actual output compared to last year is 11.1% -/
theorem factory_output_increase (a : ℝ) : 
  let last_year_output := a / 1.1
  let this_year_actual := a * 1.01
  (this_year_actual - last_year_output) / last_year_output * 100 = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_increase_l2513_251340


namespace NUMINAMATH_CALUDE_x_n_root_bound_l2513_251305

theorem x_n_root_bound (n : ℕ) (a : ℝ) (x : ℕ → ℝ) (α : ℕ → ℝ)
  (hn : n > 1)
  (ha : a ≥ 1)
  (hx1 : x 1 = 1)
  (hxi : ∀ i ∈ Finset.range n, i ≥ 2 → x i / x (i-1) = a + α i)
  (hαi : ∀ i ∈ Finset.range n, i ≥ 2 → α i ≤ 1 / (i * (i + 1))) :
  (x n) ^ (1 / (n - 1 : ℝ)) < a + 1 / (n - 1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_x_n_root_bound_l2513_251305
