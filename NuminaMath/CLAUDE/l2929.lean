import Mathlib

namespace NUMINAMATH_CALUDE_distinct_sums_count_l2929_292972

def bag_A : Finset Nat := {1, 3, 5, 7}
def bag_B : Finset Nat := {2, 4, 6, 8}

def possible_sums : Finset Nat :=
  Finset.image (λ (pair : Nat × Nat) => pair.1 + pair.2) (bag_A.product bag_B)

theorem distinct_sums_count : Finset.card possible_sums = 7 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l2929_292972


namespace NUMINAMATH_CALUDE_inscribed_octagon_area_l2929_292921

theorem inscribed_octagon_area (circle_area : ℝ) (octagon_area : ℝ) : 
  circle_area = 64 * Real.pi →
  octagon_area = 8 * (1 / 2 * (circle_area / Real.pi) * Real.sin (Real.pi / 4)) →
  octagon_area = 128 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_octagon_area_l2929_292921


namespace NUMINAMATH_CALUDE_inverse_negation_correct_l2929_292939

/-- Represents a triangle ABC -/
structure Triangle where
  isIsosceles : Bool
  hasEqualAngles : Bool

/-- The original proposition -/
def originalProposition (t : Triangle) : Prop :=
  ¬t.isIsosceles → ¬t.hasEqualAngles

/-- The inverse negation of the original proposition -/
def inverseNegation (t : Triangle) : Prop :=
  t.hasEqualAngles → t.isIsosceles

/-- Theorem stating that the inverse negation is correct -/
theorem inverse_negation_correct :
  ∀ t : Triangle, inverseNegation t ↔ ¬(¬originalProposition t) :=
sorry

end NUMINAMATH_CALUDE_inverse_negation_correct_l2929_292939


namespace NUMINAMATH_CALUDE_prob_not_square_l2929_292926

def total_figures : ℕ := 10
def num_triangles : ℕ := 5
def num_squares : ℕ := 3
def num_circles : ℕ := 2

theorem prob_not_square :
  (num_triangles + num_circles : ℚ) / total_figures = 7 / 10 :=
sorry

end NUMINAMATH_CALUDE_prob_not_square_l2929_292926


namespace NUMINAMATH_CALUDE_leading_zeros_count_l2929_292962

theorem leading_zeros_count (n : ℕ) (h : n = 20^22) :
  (∃ k : ℕ, (1 : ℚ) / n = k / 10^28 ∧ k ≥ 10^27 ∧ k < 10^28) :=
sorry

end NUMINAMATH_CALUDE_leading_zeros_count_l2929_292962


namespace NUMINAMATH_CALUDE_chicken_wings_distribution_l2929_292984

theorem chicken_wings_distribution (friends : ℕ) (pre_cooked : ℕ) (additional : ℕ) :
  friends = 3 →
  pre_cooked = 8 →
  additional = 10 →
  (pre_cooked + additional) / friends = 6 :=
by sorry

end NUMINAMATH_CALUDE_chicken_wings_distribution_l2929_292984


namespace NUMINAMATH_CALUDE_work_completion_time_l2929_292908

theorem work_completion_time (T : ℝ) 
  (h1 : 100 * T = 200 * (T - 35)) 
  (h2 : T > 35) : T = 70 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2929_292908


namespace NUMINAMATH_CALUDE_percentage_relation_l2929_292989

theorem percentage_relation (x a b : ℝ) (ha : a = 0.06 * x) (hb : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l2929_292989


namespace NUMINAMATH_CALUDE_max_k_value_l2929_292922

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-4 + Real.sqrt 29) / 13 := by
  sorry

end NUMINAMATH_CALUDE_max_k_value_l2929_292922


namespace NUMINAMATH_CALUDE_cube_inequality_l2929_292910

theorem cube_inequality (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l2929_292910


namespace NUMINAMATH_CALUDE_system_solution_l2929_292918

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x * (x + y + z) = a^2)
  (eq2 : y * (x + y + z) = b^2)
  (eq3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧
   y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧
   z = c^2 / Real.sqrt (a^2 + b^2 + c^2)) ∨
  (x = -a^2 / Real.sqrt (a^2 + b^2 + c^2) ∧
   y = -b^2 / Real.sqrt (a^2 + b^2 + c^2) ∧
   z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2929_292918


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2929_292949

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (current_speed : ℝ)
  (h1 : speed_with_current = 18)
  (h2 : current_speed = 3.4) :
  speed_with_current - 2 * current_speed = 11.2 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l2929_292949


namespace NUMINAMATH_CALUDE_range_of_a_l2929_292941

def P (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Q → x ∈ P a) → -1 ≤ a ∧ a ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2929_292941


namespace NUMINAMATH_CALUDE_no_real_solutions_l2929_292914

theorem no_real_solutions : ¬∃ (x : ℝ), x > 0 ∧ x^(1/4) = 15 / (8 - 2 * x^(1/4)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2929_292914


namespace NUMINAMATH_CALUDE_acceleration_at_two_l2929_292935

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 units
theorem acceleration_at_two : a 2 = 14 := by
  sorry

-- Lemma: The velocity function is the derivative of the distance function
lemma velocity_is_derivative_of_distance (t : ℝ) : 
  deriv s t = v t := by
  sorry

-- Lemma: The acceleration function is the derivative of the velocity function
lemma acceleration_is_derivative_of_velocity (t : ℝ) : 
  deriv v t = a t := by
  sorry

end NUMINAMATH_CALUDE_acceleration_at_two_l2929_292935


namespace NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2929_292981

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2*n + 1) = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_integers_l2929_292981


namespace NUMINAMATH_CALUDE_divisor_and_expression_l2929_292943

theorem divisor_and_expression (k : ℕ) : 
  (30^k : ℕ) ∣ 929260 → 3^k - k^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisor_and_expression_l2929_292943


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2929_292967

theorem polygon_sides_count (n : ℕ) (k : ℕ) (r : ℚ) : 
  k = n * (n - 3) / 2 →
  k = r * n →
  r = 3 / 2 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2929_292967


namespace NUMINAMATH_CALUDE_seokjin_paper_left_l2929_292976

/-- Given the initial number of sheets, number of notebooks, and pages per notebook,
    calculate the remaining sheets of paper. -/
def remaining_sheets (initial_sheets : ℕ) (num_notebooks : ℕ) (pages_per_notebook : ℕ) : ℕ :=
  initial_sheets - (num_notebooks * pages_per_notebook)

/-- Theorem stating that given 100 initial sheets, 3 notebooks with 30 pages each,
    the remaining sheets is 10. -/
theorem seokjin_paper_left : remaining_sheets 100 3 30 = 10 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_paper_left_l2929_292976


namespace NUMINAMATH_CALUDE_two_identical_solutions_l2929_292906

/-- The value of k for which the equations y = x^2 and y = 4x + k have two identical solutions -/
def k_value : ℝ := -4

/-- First equation: y = x^2 -/
def eq1 (x y : ℝ) : Prop := y = x^2

/-- Second equation: y = 4x + k -/
def eq2 (x y k : ℝ) : Prop := y = 4*x + k

/-- Two identical solutions exist when k = k_value -/
theorem two_identical_solutions (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    eq1 x₁ y₁ ∧ eq2 x₁ y₁ k ∧ 
    eq1 x₂ y₂ ∧ eq2 x₂ y₂ k) ↔ 
  k = k_value :=
sorry

end NUMINAMATH_CALUDE_two_identical_solutions_l2929_292906


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l2929_292928

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  (∀ b : ℕ, b < a → (Nat.gcd b 60 = 1 ∨ Nat.gcd b 75 = 1)) ∧
  Nat.gcd a 60 ≠ 1 ∧ Nat.gcd a 75 ≠ 1 → a = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l2929_292928


namespace NUMINAMATH_CALUDE_min_segments_11x11_grid_l2929_292986

/-- Represents a grid of lines -/
structure Grid :=
  (horizontal_lines : ℕ)
  (vertical_lines : ℕ)

/-- Calculates the number of internal nodes in a grid -/
def internal_nodes (g : Grid) : ℕ :=
  (g.horizontal_lines - 2) * (g.vertical_lines - 2)

/-- Calculates the minimum number of segments to erase -/
def min_segments_to_erase (g : Grid) : ℕ :=
  (internal_nodes g + 1) / 2

/-- The theorem stating the minimum number of segments to erase in an 11x11 grid -/
theorem min_segments_11x11_grid :
  ∃ (g : Grid), g.horizontal_lines = 11 ∧ g.vertical_lines = 11 ∧
  min_segments_to_erase g = 41 :=
sorry

end NUMINAMATH_CALUDE_min_segments_11x11_grid_l2929_292986


namespace NUMINAMATH_CALUDE_first_robber_guarantee_l2929_292904

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def FirstRobberOutcome (game : CoinGame) : ℕ := 
  min game.totalCoins (game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1)))

/-- Theorem stating the guaranteed minimum coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  FirstRobberOutcome game ≥ 146 := by
  sorry

#eval FirstRobberOutcome { totalCoins := 300, maxBags := 11 }

end NUMINAMATH_CALUDE_first_robber_guarantee_l2929_292904


namespace NUMINAMATH_CALUDE_geometric_place_of_tangent_points_l2929_292995

/-- Given a circle with center O(0,0) and radius r in a right-angled coordinate system,
    the geometric place of points S(x,y) whose adjoint lines are tangents to the circle
    is defined by the equation 1/x^2 + 1/y^2 = 1/r^2 -/
theorem geometric_place_of_tangent_points (r : ℝ) (h : r > 0) :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 →
    (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = r^2 ∧ x₁ * x + y₁ * y = r^2) ↔
    1 / x^2 + 1 / y^2 = 1 / r^2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_place_of_tangent_points_l2929_292995


namespace NUMINAMATH_CALUDE_max_value_theorem_l2929_292960

theorem max_value_theorem (x y : ℝ) (h : x + y = 5) :
  ∃ (max : ℝ), max = (1175 : ℝ) / 16 ∧
  ∀ (z : ℝ), x^3 * y + x^2 * y + x * y + x * y^2 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2929_292960


namespace NUMINAMATH_CALUDE_cone_slant_height_l2929_292983

/-- Given a cone with base radius 5 cm and unfolded side area 60π cm², 
    prove that its slant height is 12 cm -/
theorem cone_slant_height (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 5 → A = 60 * Real.pi → A = (Real.pi * r * l) → l = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l2929_292983


namespace NUMINAMATH_CALUDE_community_cleaning_event_l2929_292919

theorem community_cleaning_event (C : ℝ) : 
  (0.3 * C) + (2 * 0.3 * C) + 200 = C → C = 2000 := by
sorry

end NUMINAMATH_CALUDE_community_cleaning_event_l2929_292919


namespace NUMINAMATH_CALUDE_work_completion_time_l2929_292937

/-- The number of days it takes for person A to complete the work -/
def days_A : ℝ := 18

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_2_days : ℝ := 0.19444444444444442

/-- The number of days it takes for person B to complete the work -/
def days_B : ℝ := 24

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * 2 = work_completed_2_days :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2929_292937


namespace NUMINAMATH_CALUDE_composition_three_reflections_is_glide_reflection_l2929_292946

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for lines in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a reflection transformation
def Reflection (l : Line2D) : Point2D → Point2D := sorry

-- Define a translation transformation
def Translation (dx dy : ℝ) : Point2D → Point2D := sorry

-- Define a glide reflection transformation
def GlideReflection (l : Line2D) (t : ℝ) : Point2D → Point2D := sorry

-- Define a predicate to check if three lines pass through the same point
def passThroughSamePoint (l1 l2 l3 : Line2D) : Prop := sorry

-- Define a predicate to check if three lines are parallel to the same line
def parallelToSameLine (l1 l2 l3 : Line2D) : Prop := sorry

-- Theorem statement
theorem composition_three_reflections_is_glide_reflection 
  (l1 l2 l3 : Line2D) 
  (h1 : ¬ passThroughSamePoint l1 l2 l3) 
  (h2 : ¬ parallelToSameLine l1 l2 l3) :
  ∃ (l : Line2D) (t : ℝ), 
    ∀ p : Point2D, 
      (Reflection l3 ∘ Reflection l2 ∘ Reflection l1) p = GlideReflection l t p :=
sorry

end NUMINAMATH_CALUDE_composition_three_reflections_is_glide_reflection_l2929_292946


namespace NUMINAMATH_CALUDE_triangle_abc_area_l2929_292942

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 3 under the given conditions. -/
theorem triangle_abc_area (a b c : ℝ) (A B C : ℝ) : 
  a = Real.sqrt 5 →
  b = 3 →
  Real.sin C = 2 * Real.sin A →
  (1/2) * a * c * Real.sin B = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_area_l2929_292942


namespace NUMINAMATH_CALUDE_special_rectangle_ratio_l2929_292930

/-- A rectangle with the property that the square of the ratio of its short side to its long side
    is equal to the ratio of its long side to its diagonal. -/
structure SpecialRectangle where
  short : ℝ
  long : ℝ
  diagonal : ℝ
  short_positive : 0 < short
  long_positive : 0 < long
  diagonal_positive : 0 < diagonal
  pythagorean : diagonal^2 = short^2 + long^2
  special_property : (short / long)^2 = long / diagonal

/-- The ratio of the short side to the long side in a SpecialRectangle is (√5 - 1) / 3. -/
theorem special_rectangle_ratio (r : SpecialRectangle) : 
  r.short / r.long = (Real.sqrt 5 - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_ratio_l2929_292930


namespace NUMINAMATH_CALUDE_juan_distance_l2929_292953

/-- Given a speed and time, calculate the distance traveled. -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Juan's distance traveled is 80 miles. -/
theorem juan_distance : distance 10 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_juan_distance_l2929_292953


namespace NUMINAMATH_CALUDE_least_whole_number_for_ratio_l2929_292901

theorem least_whole_number_for_ratio (x : ℕ) : x = 3 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
  (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by sorry

end NUMINAMATH_CALUDE_least_whole_number_for_ratio_l2929_292901


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2929_292900

theorem arithmetic_equality : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2929_292900


namespace NUMINAMATH_CALUDE_eve_distance_difference_l2929_292917

theorem eve_distance_difference : 
  let ran_distance : ℝ := 0.7
  let walked_distance : ℝ := 0.6
  ran_distance - walked_distance = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l2929_292917


namespace NUMINAMATH_CALUDE_min_value_theorem_l2929_292931

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : -2 * m - n + 2 = 0) :
  2 / m + 1 / n ≥ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2929_292931


namespace NUMINAMATH_CALUDE_stacys_height_l2929_292957

/-- Stacy's height problem -/
theorem stacys_height (stacy_last_year : ℕ) (stacy_growth_diff : ℕ) (brother_growth : ℕ) :
  stacy_last_year = 50 →
  stacy_growth_diff = 6 →
  brother_growth = 1 →
  stacy_last_year + (stacy_growth_diff + brother_growth) = 57 :=
by sorry

end NUMINAMATH_CALUDE_stacys_height_l2929_292957


namespace NUMINAMATH_CALUDE_colored_pencil_drawings_l2929_292959

theorem colored_pencil_drawings (total : ℕ) (blending_markers : ℕ) (charcoal : ℕ) 
  (h1 : total = 25)
  (h2 : blending_markers = 7)
  (h3 : charcoal = 4) :
  total - (blending_markers + charcoal) = 14 := by
  sorry

end NUMINAMATH_CALUDE_colored_pencil_drawings_l2929_292959


namespace NUMINAMATH_CALUDE_valid_paths_count_l2929_292940

/-- Represents a point in the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents the grid with its dimensions and blocked points -/
structure Grid where
  width : Nat
  height : Nat
  blockedPoints : List GridPoint

/-- Calculates the number of valid paths in the grid -/
def countValidPaths (g : Grid) : Nat :=
  sorry

/-- The specific grid from the problem -/
def problemGrid : Grid :=
  { width := 5
  , height := 3
  , blockedPoints := [⟨2, 1⟩, ⟨3, 1⟩] }

theorem valid_paths_count :
  countValidPaths problemGrid = 39 :=
by sorry

end NUMINAMATH_CALUDE_valid_paths_count_l2929_292940


namespace NUMINAMATH_CALUDE_range_of_a_l2929_292991

theorem range_of_a (a : ℝ) : Real.sqrt (a^2) = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2929_292991


namespace NUMINAMATH_CALUDE_soap_brand_usage_l2929_292999

/-- The number of households using both brand R and brand B soap -/
def households_using_both : ℕ := 15

/-- The total number of households surveyed -/
def total_households : ℕ := 200

/-- The number of households using neither brand R nor brand B -/
def households_using_neither : ℕ := 80

/-- The number of households using only brand R -/
def households_using_only_R : ℕ := 60

/-- For every household using both brands, this many use only brand B -/
def ratio_B_to_both : ℕ := 3

theorem soap_brand_usage :
  households_using_both * (ratio_B_to_both + 1) + 
  households_using_neither + 
  households_using_only_R = 
  total_households := by sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l2929_292999


namespace NUMINAMATH_CALUDE_sqrt_2_times_sqrt_6_l2929_292964

theorem sqrt_2_times_sqrt_6 : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_times_sqrt_6_l2929_292964


namespace NUMINAMATH_CALUDE_product_in_first_quadrant_l2929_292903

def complex_multiply (a b c d : ℝ) : ℂ :=
  Complex.mk (a * c - b * d) (a * d + b * c)

theorem product_in_first_quadrant :
  let z : ℂ := complex_multiply 1 3 3 (-1)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_product_in_first_quadrant_l2929_292903


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2929_292944

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2929_292944


namespace NUMINAMATH_CALUDE_fence_cost_l2929_292977

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 57) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let cost := perimeter * price_per_foot
  cost = 3876 := by sorry

end NUMINAMATH_CALUDE_fence_cost_l2929_292977


namespace NUMINAMATH_CALUDE_single_displacement_equivalent_l2929_292907

-- Define a type for plane figures
structure PlaneFigure where
  -- Add necessary properties for a plane figure

-- Define a function for parallel displacement
def parallelDisplacement (F : PlaneFigure) (v : ℝ × ℝ) : PlaneFigure :=
  sorry

-- Theorem statement
theorem single_displacement_equivalent (F : PlaneFigure) (v1 v2 : ℝ × ℝ) :
  ∃ v : ℝ × ℝ, parallelDisplacement F v = parallelDisplacement (parallelDisplacement F v1) v2 :=
sorry

end NUMINAMATH_CALUDE_single_displacement_equivalent_l2929_292907


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l2929_292932

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

theorem tangent_line_at_one (x : ℝ) :
  let p : ℝ × ℝ := (1, 0)
  let m : ℝ := 2
  (fun x => m * (x - p.1) + p.2) = (fun x => 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l2929_292932


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2929_292993

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2929_292993


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l2929_292912

theorem spinster_cat_ratio : 
  ∀ (x : ℚ), 
    (22 : ℚ) / x = 7 →  -- ratio of spinsters to cats is x:7
    x = 22 + 55 →      -- there are 55 more cats than spinsters
    (2 : ℚ) / 7 = 22 / x -- the ratio of spinsters to cats is 2:7
  := by sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l2929_292912


namespace NUMINAMATH_CALUDE_sarah_borrowed_l2929_292934

/-- Calculates the earnings for a given number of hours based on the described wage structure --/
def earnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 8
  let remainingHours := hours % 8
  fullCycles * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) + 
    (List.range remainingHours).sum.succ

/-- The amount Sarah borrowed is equal to her earnings for 40 hours of work --/
theorem sarah_borrowed (borrowedAmount : ℕ) : borrowedAmount = earnings 40 := by
  sorry

#eval earnings 40  -- Should output 180

end NUMINAMATH_CALUDE_sarah_borrowed_l2929_292934


namespace NUMINAMATH_CALUDE_die_throws_for_most_likely_two_l2929_292963

theorem die_throws_for_most_likely_two (n : ℕ) : 
  let p : ℚ := 1/6  -- probability of rolling a two
  let q : ℚ := 5/6  -- probability of not rolling a two
  let k₀ : ℕ := 32  -- most likely number of times a two is rolled
  (n * p - q ≤ k₀ ∧ k₀ ≤ n * p + p) → (191 ≤ n ∧ n ≤ 197) :=
by sorry

end NUMINAMATH_CALUDE_die_throws_for_most_likely_two_l2929_292963


namespace NUMINAMATH_CALUDE_geometry_relations_l2929_292948

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_line : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : subset_line_plane l β) :
  (parallel_plane α β → perp_line m l) ∧
  (parallel_line m l → perp_plane α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l2929_292948


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2929_292913

def P (x : ℝ) : ℝ := 5*x^3 - 12*x^2 + 6*x - 15

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), P = λ x => q x * (x - 3) + 30 :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2929_292913


namespace NUMINAMATH_CALUDE_division_multiplication_equality_l2929_292951

theorem division_multiplication_equality : (-150) / (-50) * (1/3 : ℚ) = 1 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_equality_l2929_292951


namespace NUMINAMATH_CALUDE_loss_percentage_book1_l2929_292975

-- Define the total cost of both books
def total_cost : ℝ := 450

-- Define the cost of the first book (sold at a loss)
def cost_book1 : ℝ := 262.5

-- Define the gain percentage on the second book
def gain_percentage : ℝ := 0.19

-- Define the function to calculate the selling price of the second book
def selling_price_book2 (cost : ℝ) : ℝ := cost * (1 + gain_percentage)

-- Define the theorem
theorem loss_percentage_book1 : 
  let cost_book2 := total_cost - cost_book1
  let sp := selling_price_book2 cost_book2
  let loss_percentage := (cost_book1 - sp) / cost_book1 * 100
  loss_percentage = 15 := by sorry

end NUMINAMATH_CALUDE_loss_percentage_book1_l2929_292975


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l2929_292970

-- Define the polynomial terms
def term1 (x y : ℤ) : ℤ := 7 * x^2 * y
def term2 (x y : ℤ) : ℤ := 21 * x * y^2

-- Define the common factor
def common_factor (x y : ℤ) : ℤ := 7 * x * y

-- Theorem statement
theorem common_factor_is_gcf :
  ∀ (x y : ℤ), 
    (∃ (a b : ℤ), term1 x y = common_factor x y * a ∧ term2 x y = common_factor x y * b) ∧
    (∀ (z : ℤ), (∃ (c d : ℤ), term1 x y = z * c ∧ term2 x y = z * d) → z ∣ common_factor x y) :=
sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l2929_292970


namespace NUMINAMATH_CALUDE_largest_angle_is_90_l2929_292990

-- Define an isosceles triangle with angles α, β, and γ
structure IsoscelesTriangle where
  α : Real
  β : Real
  γ : Real
  isIsosceles : (α = β) ∨ (α = γ) ∨ (β = γ)
  sumIs180 : α + β + γ = 180
  nonNegative : α ≥ 0 ∧ β ≥ 0 ∧ γ ≥ 0

-- Define the condition that two angles are in the ratio 1:2
def hasRatio1to2 (t : IsoscelesTriangle) : Prop :=
  (t.α = 2 * t.β) ∨ (t.β = 2 * t.α) ∨ (t.α = 2 * t.γ) ∨ (t.γ = 2 * t.α) ∨ (t.β = 2 * t.γ) ∨ (t.γ = 2 * t.β)

-- Theorem statement
theorem largest_angle_is_90 (t : IsoscelesTriangle) (h : hasRatio1to2 t) :
  max t.α (max t.β t.γ) = 90 := by sorry

end NUMINAMATH_CALUDE_largest_angle_is_90_l2929_292990


namespace NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l2929_292974

theorem complex_square_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I) ^ 2 = Complex.I * y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_on_negative_y_axis_l2929_292974


namespace NUMINAMATH_CALUDE_f_10_equals_222_l2929_292969

-- Define the function f
def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_10_equals_222 (y : ℝ) (h : f 2 y = 30) : f 10 y = 222 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_222_l2929_292969


namespace NUMINAMATH_CALUDE_always_odd_l2929_292915

theorem always_odd (k : ℤ) : Odd (2007 + 2 * k^2) := by sorry

end NUMINAMATH_CALUDE_always_odd_l2929_292915


namespace NUMINAMATH_CALUDE_pyramid_apex_distance_l2929_292909

/-- Pyramid structure with a square base -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)
  (sphere_radius : ℝ)

/-- The distance between the center of the base and the apex of the pyramid -/
def apex_distance (p : Pyramid) : ℝ := sorry

theorem pyramid_apex_distance (p : Pyramid) 
  (h1 : p.base_side = 2 * Real.sqrt 2)
  (h2 : p.height = 1)
  (h3 : p.sphere_radius = 2 * Real.sqrt 2) :
  apex_distance p = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_pyramid_apex_distance_l2929_292909


namespace NUMINAMATH_CALUDE_vanessa_video_files_vanessa_video_files_proof_l2929_292982

theorem vanessa_video_files 
  (initial_music_files : ℕ) 
  (deleted_files : ℕ) 
  (remaining_files : ℕ) : ℕ :=
  let initial_total_files := remaining_files + deleted_files
  let initial_video_files := initial_total_files - initial_music_files
  initial_video_files

-- Proof
theorem vanessa_video_files_proof 
  (initial_music_files : ℕ) 
  (deleted_files : ℕ) 
  (remaining_files : ℕ) 
  (h1 : initial_music_files = 13) 
  (h2 : deleted_files = 10) 
  (h3 : remaining_files = 33) : 
  vanessa_video_files initial_music_files deleted_files remaining_files = 30 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_video_files_vanessa_video_files_proof_l2929_292982


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l2929_292987

/-- Proves the factorization of x^2(x-1)-x+1 -/
theorem factorization_1 (x : ℝ) : x^2 * (x - 1) - x + 1 = (x - 1)^2 * (x + 1) := by
  sorry

/-- Proves the factorization of 3p(x+1)^3y^2+6p(x+1)^2y+3p(x+1) -/
theorem factorization_2 (p x y : ℝ) : 
  3 * p * (x + 1)^3 * y^2 + 6 * p * (x + 1)^2 * y + 3 * p * (x + 1) = 
  3 * p * (x + 1) * (x * y + y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l2929_292987


namespace NUMINAMATH_CALUDE_largest_crossed_out_prime_l2929_292925

theorem largest_crossed_out_prime (count_primes : ℕ) 
  (h_count : count_primes = 168) 
  (p q : ℕ) 
  (h_p_prime : Nat.Prime p) 
  (h_q_prime : Nat.Prime q) 
  (h_p_odd : ¬Even p) 
  (h_q_odd : ¬Even q) 
  (h_q_ge_3 : q ≥ 3) 
  (h_pq_le_1000 : p * q ≤ 1000) 
  (h_p_ne_q : p ≠ q) :
  p ≤ 331 ∧ ∃ (m n : ℕ), m ≤ 1000 ∧ n ≤ 1000 ∧ 
    ((m = p * q ∧ n = p * p) ∨ (m = p * q ∧ n = q * q) ∨ (m = p * p ∧ n = q * q)) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_crossed_out_prime_l2929_292925


namespace NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l2929_292994

/-- The shortest distance between a point and a parabola -/
theorem shortest_distance_point_to_parabola :
  let point := (7, 15)
  let parabola := λ x : ℝ => (x, x^2)
  ∃ d : ℝ, d = 2 * Real.sqrt 13 ∧
    ∀ x : ℝ, d ≤ Real.sqrt ((7 - x)^2 + (15 - x^2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_point_to_parabola_l2929_292994


namespace NUMINAMATH_CALUDE_newberg_airport_passengers_l2929_292955

/-- The number of passengers who landed on time in Newberg last year -/
def on_time_passengers : ℕ := 14507

/-- The number of passengers who landed late in Newberg last year -/
def late_passengers : ℕ := 213

/-- The number of passengers who had connecting flights in Newberg last year -/
def connecting_passengers : ℕ := 320

/-- The total number of passengers who landed in Newberg last year -/
def total_passengers : ℕ := on_time_passengers + late_passengers + connecting_passengers

theorem newberg_airport_passengers :
  total_passengers = 15040 :=
sorry

end NUMINAMATH_CALUDE_newberg_airport_passengers_l2929_292955


namespace NUMINAMATH_CALUDE_g_geq_h_implies_a_leq_one_l2929_292905

noncomputable def g (x : ℝ) : ℝ := Real.exp x - Real.exp 1 * x - 1

noncomputable def h (a x : ℝ) : ℝ := a * Real.sin x - Real.exp 1 * x

theorem g_geq_h_implies_a_leq_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g x ≥ h a x) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_g_geq_h_implies_a_leq_one_l2929_292905


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2929_292980

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (1, 2) and b = (1-m, 2m-4), then m = 3/2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![1 - m, 2 * m - 4]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2929_292980


namespace NUMINAMATH_CALUDE_prime_perfect_square_triples_l2929_292936

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def solution_set (p q r : ℕ) : Prop :=
  (p = 2 ∧ q = 5 ∧ r = 2) ∨
  (p = 2 ∧ q = 2 ∧ r = 5) ∨
  (p = 2 ∧ q = 3 ∧ r = 3) ∨
  (p = 3 ∧ q = 3 ∧ r = 2) ∨
  (∃ n : ℕ, p = 2 ∧ q = 2*n + 1 ∧ r = 2*n + 1)

theorem prime_perfect_square_triples (p q r : ℕ) :
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
  (is_perfect_square (p^q + p^r) ↔ solution_set p q r) := by
  sorry

end NUMINAMATH_CALUDE_prime_perfect_square_triples_l2929_292936


namespace NUMINAMATH_CALUDE_solution_product_l2929_292927

theorem solution_product (p q : ℝ) : 
  (p - 3) * (3 * p + 18) = p^2 - 15 * p + 54 →
  (q - 3) * (3 * q + 18) = q^2 - 15 * q + 54 →
  p ≠ q →
  (p + 2) * (q + 2) = -80 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2929_292927


namespace NUMINAMATH_CALUDE_even_multiples_sum_difference_l2929_292979

theorem even_multiples_sum_difference : 
  let n : ℕ := 2025
  let even_sum : ℕ := n * (2 + 2 * n)
  let multiples_of_three_sum : ℕ := n * (3 + 3 * n)
  (even_sum : ℤ) - (multiples_of_three_sum : ℤ) = -2052155 := by
  sorry

end NUMINAMATH_CALUDE_even_multiples_sum_difference_l2929_292979


namespace NUMINAMATH_CALUDE_twice_a_plus_one_nonnegative_l2929_292968

theorem twice_a_plus_one_nonnegative (a : ℝ) : (2 * a + 1 ≥ 0) ↔ (∀ x : ℝ, x = 2 * a + 1 → x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_twice_a_plus_one_nonnegative_l2929_292968


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2929_292950

open Real

theorem trigonometric_problem (α β : Real)
  (h1 : α ∈ Set.Ioo 0 (π/2))
  (h2 : β ∈ Set.Ioo (π/2) π)
  (h3 : cos β = -1/3)
  (h4 : sin (α + β) = (4 - Real.sqrt 2) / 6) :
  tan (2 * β) = (4 * Real.sqrt 2) / 7 ∧ α = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2929_292950


namespace NUMINAMATH_CALUDE_high_jump_probabilities_l2929_292997

/-- Probability of success for athlete A -/
def pA : ℝ := 0.7

/-- Probability of success for athlete B -/
def pB : ℝ := 0.6

/-- Probability that athlete A succeeds on the third attempt for the first time -/
def prob_A_third : ℝ := (1 - pA) * (1 - pA) * pA

/-- Probability that at least one athlete succeeds in their first attempt -/
def prob_at_least_one : ℝ := 1 - (1 - pA) * (1 - pB)

/-- Probability that after two attempts each, A has exactly one more successful attempt than B -/
def prob_A_one_more : ℝ := 
  2 * pA * (1 - pA) * (1 - pB) * (1 - pB) + 
  pA * pA * 2 * pB * (1 - pB)

theorem high_jump_probabilities :
  prob_A_third = 0.063 ∧ 
  prob_at_least_one = 0.88 ∧ 
  prob_A_one_more = 0.3024 := by
  sorry

end NUMINAMATH_CALUDE_high_jump_probabilities_l2929_292997


namespace NUMINAMATH_CALUDE_systematic_sampling_first_stage_l2929_292938

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a stage in the sampling process -/
inductive SamplingStage
  | First
  | Later

/-- Defines the relationship between sampling methods and stages -/
def sampling_relationship (method : SamplingMethod) (stage : SamplingStage) : Prop :=
  match method, stage with
  | SamplingMethod.Systematic, SamplingStage.First => true
  | _, _ => false

/-- Theorem stating that systematic sampling generally uses simple random sampling in the first stage -/
theorem systematic_sampling_first_stage :
  sampling_relationship SamplingMethod.Systematic SamplingStage.First = true :=
by
  sorry

#check systematic_sampling_first_stage

end NUMINAMATH_CALUDE_systematic_sampling_first_stage_l2929_292938


namespace NUMINAMATH_CALUDE_carol_peanuts_count_l2929_292952

/-- The number of peanuts Carol initially collects -/
def initial_peanuts : ℕ := 2

/-- The number of peanuts Carol's father gives her -/
def given_peanuts : ℕ := 5

/-- The total number of peanuts Carol has -/
def total_peanuts : ℕ := initial_peanuts + given_peanuts

theorem carol_peanuts_count : total_peanuts = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_peanuts_count_l2929_292952


namespace NUMINAMATH_CALUDE_quadratic_function_and_tangent_line_l2929_292947

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A point x is a zero of function f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- A line y = kx + m is tangent to the graph of f if there exists exactly one point
    where the line touches the graph of f -/
def IsTangent (f : ℝ → ℝ) (k m : ℝ) : Prop :=
  ∃! x, f x = k * x + m

theorem quadratic_function_and_tangent_line 
  (f : ℝ → ℝ) (b c k m : ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c)
  (h2 : IsEven f)
  (h3 : IsZero f 1)
  (h4 : k > 0)
  (h5 : IsTangent f k m) :
  (∀ x, f x = x^2 - 1) ∧ 
  (∀ k m, k > 0 → IsTangent f k m → m * k ≤ -4) ∧
  (∃ k m, k > 0 ∧ IsTangent f k m ∧ m * k = -4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_tangent_line_l2929_292947


namespace NUMINAMATH_CALUDE_three_digit_sum_l2929_292945

theorem three_digit_sum (A B : ℕ) : 
  A < 10 → 
  B < 10 → 
  100 ≤ 14 * 10 + A → 
  14 * 10 + A < 1000 → 
  100 ≤ 100 * B + 73 → 
  100 * B + 73 < 1000 → 
  14 * 10 + A + 100 * B + 73 = 418 → 
  A = 5 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_l2929_292945


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2929_292988

def n₁ : ℕ := 263
def n₂ : ℕ := 935
def n₃ : ℕ := 1383
def r : ℕ := 7
def d : ℕ := 32

theorem greatest_divisor_with_remainder (m : ℕ) :
  (m > d → ¬(n₁ % m = r ∧ n₂ % m = r ∧ n₃ % m = r)) ∧
  (n₁ % d = r ∧ n₂ % d = r ∧ n₃ % d = r) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainder_l2929_292988


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l2929_292902

theorem parabola_tangent_line (b c : ℝ) : 
  (∃ x y : ℝ, y = -2 * x^2 + b * x + c ∧ 
              y = x - 3 ∧ 
              x = 2 ∧ 
              y = -1) → 
  b + c = -2 :=
sorry


end NUMINAMATH_CALUDE_parabola_tangent_line_l2929_292902


namespace NUMINAMATH_CALUDE_exists_points_with_midpoint_l2929_292920

-- Define the hyperbola equation
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the midpoint of two points
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

-- Theorem statement
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by
  sorry

end NUMINAMATH_CALUDE_exists_points_with_midpoint_l2929_292920


namespace NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l2929_292911

/-- Calculates the downstream distance given swimming conditions -/
theorem downstream_distance (v_m : ℝ) (t : ℝ) (d_upstream : ℝ) : ℝ :=
  let v_s := v_m - d_upstream / t
  let v_downstream := v_m + v_s
  v_downstream * t

/-- Proves the downstream distance for the given problem conditions -/
theorem man_downstream_distance :
  downstream_distance 4 6 18 = 30 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_man_downstream_distance_l2929_292911


namespace NUMINAMATH_CALUDE_reaction_outcome_l2929_292978

/-- Represents a chemical reaction with reactants and products -/
structure ChemicalReaction where
  alMoles : ℚ
  h2so4Moles : ℚ
  al2so43Moles : ℚ
  h2Moles : ℚ

/-- The balanced equation for the reaction -/
def balancedReaction : ChemicalReaction :=
  { alMoles := 2
  , h2so4Moles := 3
  , al2so43Moles := 1
  , h2Moles := 3 }

/-- The given amounts of reactants -/
def givenReactants : ChemicalReaction :=
  { alMoles := 2
  , h2so4Moles := 3
  , al2so43Moles := 0
  , h2Moles := 0 }

/-- Checks if the reaction is balanced -/
def isBalanced (r : ChemicalReaction) : Prop :=
  r.alMoles / balancedReaction.alMoles = r.h2so4Moles / balancedReaction.h2so4Moles

/-- Calculates the limiting factor of the reaction -/
def limitingFactor (r : ChemicalReaction) : ℚ :=
  min (r.alMoles / balancedReaction.alMoles) (r.h2so4Moles / balancedReaction.h2so4Moles)

/-- Calculates the products formed in the reaction -/
def productsFormed (r : ChemicalReaction) : ChemicalReaction :=
  let factor := limitingFactor r
  { alMoles := r.alMoles - factor * balancedReaction.alMoles
  , h2so4Moles := r.h2so4Moles - factor * balancedReaction.h2so4Moles
  , al2so43Moles := factor * balancedReaction.al2so43Moles
  , h2Moles := factor * balancedReaction.h2Moles }

theorem reaction_outcome :
  isBalanced givenReactants ∧
  (productsFormed givenReactants).al2so43Moles = 1 ∧
  (productsFormed givenReactants).h2Moles = 3 ∧
  (productsFormed givenReactants).alMoles = 0 ∧
  (productsFormed givenReactants).h2so4Moles = 0 :=
sorry

end NUMINAMATH_CALUDE_reaction_outcome_l2929_292978


namespace NUMINAMATH_CALUDE_new_students_average_age_l2929_292996

/-- Given a class where:
    - The original number of students is 8
    - The original average age is 40 years
    - 8 new students join
    - The new average age of the entire class is 36 years
    This theorem proves that the average age of the new students is 32 years. -/
theorem new_students_average_age
  (original_count : Nat)
  (original_avg : ℝ)
  (new_count : Nat)
  (new_total_avg : ℝ)
  (h1 : original_count = 8)
  (h2 : original_avg = 40)
  (h3 : new_count = 8)
  (h4 : new_total_avg = 36) :
  (((original_count + new_count) * new_total_avg) - (original_count * original_avg)) / new_count = 32 := by
  sorry


end NUMINAMATH_CALUDE_new_students_average_age_l2929_292996


namespace NUMINAMATH_CALUDE_beta_value_l2929_292933

open Real

theorem beta_value (α β : ℝ) 
  (h1 : sin α = (4/7) * Real.sqrt 3)
  (h2 : cos (α + β) = -11/14)
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) : 
  β = π/3 := by
sorry

end NUMINAMATH_CALUDE_beta_value_l2929_292933


namespace NUMINAMATH_CALUDE_tangent_circles_area_ratio_l2929_292998

/-- Regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- Circle tangent to two lines of the hexagon -/
structure TangentCircle (h : RegularHexagon) :=
  (radius : ℝ)
  (tangent_to_side : Bool)
  (tangent_to_ef : Bool)

/-- The ratio of areas of two tangent circles is 1 -/
theorem tangent_circles_area_ratio 
  (h : RegularHexagon) 
  (c1 c2 : TangentCircle h) 
  (h1 : c1.tangent_to_side = true) 
  (h2 : c2.tangent_to_side = true) 
  (h3 : c1.tangent_to_ef = true) 
  (h4 : c2.tangent_to_ef = true) : 
  (c2.radius^2) / (c1.radius^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_area_ratio_l2929_292998


namespace NUMINAMATH_CALUDE_time_after_2051_hours_l2929_292954

/-- Calculates the time on a 12-hour clock after a given number of hours have passed -/
def timeAfter (startTime : Nat) (hoursPassed : Nat) : Nat :=
  (startTime + hoursPassed) % 12

/-- Proves that 2051 hours after 9 o'clock, it will be 8 o'clock on a 12-hour clock -/
theorem time_after_2051_hours :
  timeAfter 9 2051 = 8 := by
  sorry

#eval timeAfter 9 2051  -- This should output 8

end NUMINAMATH_CALUDE_time_after_2051_hours_l2929_292954


namespace NUMINAMATH_CALUDE_smallest_number_in_set_l2929_292923

def number_set : Set ℤ := {0, -2, 1, 5}

theorem smallest_number_in_set : 
  ∃ x ∈ number_set, ∀ y ∈ number_set, x ≤ y ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_set_l2929_292923


namespace NUMINAMATH_CALUDE_box_cube_volume_l2929_292956

/-- Given a box with dimensions 10 cm x 18 cm x 4 cm, filled completely with 60 identical cubes,
    the volume of each cube is 8 cubic centimeters. -/
theorem box_cube_volume (length width height : ℕ) (num_cubes : ℕ) (cube_volume : ℕ) :
  length = 10 ∧ width = 18 ∧ height = 4 ∧ num_cubes = 60 →
  length * width * height = num_cubes * cube_volume →
  cube_volume = 8 := by
  sorry

#check box_cube_volume

end NUMINAMATH_CALUDE_box_cube_volume_l2929_292956


namespace NUMINAMATH_CALUDE_chris_winning_configurations_l2929_292929

/-- Modified nim-value for a single wall in the brick removal game -/
def modified_nim_value (n : ℕ) : ℕ := sorry

/-- Nim-sum of a list of natural numbers -/
def nim_sum (l : List ℕ) : ℕ := sorry

/-- Represents a game configuration as a list of wall sizes -/
def GameConfig := List ℕ

/-- Determines if Chris (second player) can guarantee a win for a given game configuration -/
def chris_wins (config : GameConfig) : Prop :=
  nim_sum (config.map modified_nim_value) = 0

theorem chris_winning_configurations :
  ∀ config : GameConfig,
    (chris_wins config ↔ 
      (config = [7, 5, 2] ∨ config = [7, 5, 3])) :=
by sorry

end NUMINAMATH_CALUDE_chris_winning_configurations_l2929_292929


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2929_292992

theorem simplify_fraction_product : 
  (360 : ℚ) / 24 * (10 : ℚ) / 240 * (6 : ℚ) / 3 * (9 : ℚ) / 18 = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2929_292992


namespace NUMINAMATH_CALUDE_jeans_wednesday_calls_l2929_292966

/-- Represents the number of calls Jean answered each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Calculates the total number of calls in a week --/
def total_calls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Jean's calls for the week --/
def jeans_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,  -- This is what we want to prove
  thursday := 61,
  friday := 31
}

/-- Theorem stating that Jean answered 27 calls on Wednesday --/
theorem jeans_wednesday_calls :
  jeans_calls.wednesday = 27 ∧
  total_calls jeans_calls = average_calls * working_days :=
sorry

end NUMINAMATH_CALUDE_jeans_wednesday_calls_l2929_292966


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2929_292916

theorem largest_multiple_of_9_under_100 : ∃ (n : ℕ), n = 99 ∧ 
  (∀ m : ℕ, m < 100 ∧ 9 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2929_292916


namespace NUMINAMATH_CALUDE_solve_equation_l2929_292985

theorem solve_equation (r : ℚ) : (r - 45) / 2 = (3 - 2 * r) / 5 → r = 77 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2929_292985


namespace NUMINAMATH_CALUDE_seconds_in_month_l2929_292924

-- Define the number of seconds in one day
def seconds_per_day : ℝ := 8.64 * 10^4

-- Define the number of days in one month
def days_per_month : ℕ := 30

-- Theorem: The number of seconds in one month is 2.592 × 10^5
theorem seconds_in_month : 
  seconds_per_day * days_per_month = 2.592 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_month_l2929_292924


namespace NUMINAMATH_CALUDE_mass_NaHCO3_required_mass_NaHCO3_proof_l2929_292958

/-- The mass of NaHCO3 required to neutralize H2SO4 -/
theorem mass_NaHCO3_required (volume_H2SO4 : Real) (concentration_H2SO4 : Real) 
  (molar_mass_NaHCO3 : Real) (stoichiometric_ratio : Real) : Real :=
  let moles_H2SO4 := volume_H2SO4 * concentration_H2SO4
  let moles_NaHCO3 := moles_H2SO4 * stoichiometric_ratio
  let mass_NaHCO3 := moles_NaHCO3 * molar_mass_NaHCO3
  mass_NaHCO3

/-- Proof that the mass of NaHCO3 required is 0.525 g -/
theorem mass_NaHCO3_proof :
  mass_NaHCO3_required 0.025 0.125 84 2 = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_mass_NaHCO3_required_mass_NaHCO3_proof_l2929_292958


namespace NUMINAMATH_CALUDE_factorization_problem_1_l2929_292965

theorem factorization_problem_1 (x : ℝ) : -27 + 3 * x^2 = -3 * (3 + x) * (3 - x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l2929_292965


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l2929_292971

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) : 
  f 1 > f (-2) ∧ f (-2) > f 3 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l2929_292971


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_undetermined_l2929_292973

/-- Represents a non-normal probability distribution --/
structure NonNormalDistribution where
  mean : ℝ
  std_dev : ℝ
  skewness : ℝ
  kurtosis : ℝ
  is_non_normal : Bool

/-- The value that is exactly 3 standard deviations less than the mean --/
def three_std_dev_below_mean (d : NonNormalDistribution) : ℝ := sorry

/-- Theorem stating that the value 3 standard deviations below the mean cannot be determined
    for a non-normal distribution without additional information --/
theorem three_std_dev_below_mean_undetermined
  (d : NonNormalDistribution)
  (h_mean : d.mean = 15)
  (h_std_dev : d.std_dev = 1.5)
  (h_skewness : d.skewness = 0.5)
  (h_kurtosis : d.kurtosis = 0.6)
  (h_non_normal : d.is_non_normal = true) :
  ¬ ∃ (x : ℝ), three_std_dev_below_mean d = x :=
sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_undetermined_l2929_292973


namespace NUMINAMATH_CALUDE_least_cans_required_l2929_292961

def maaza : ℕ := 80
def pepsi : ℕ := 144
def sprite : ℕ := 368

theorem least_cans_required (maaza pepsi sprite : ℕ) 
  (h1 : maaza = 80) 
  (h2 : pepsi = 144) 
  (h3 : sprite = 368) : 
  ∃ (can_volume : ℕ), 
    can_volume > 0 ∧ 
    maaza % can_volume = 0 ∧ 
    pepsi % can_volume = 0 ∧ 
    sprite % can_volume = 0 ∧ 
    maaza / can_volume + pepsi / can_volume + sprite / can_volume = 37 ∧
    ∀ (other_volume : ℕ), 
      (other_volume > 0 ∧ 
       maaza % other_volume = 0 ∧ 
       pepsi % other_volume = 0 ∧ 
       sprite % other_volume = 0) → 
      other_volume ≤ can_volume :=
sorry

end NUMINAMATH_CALUDE_least_cans_required_l2929_292961
