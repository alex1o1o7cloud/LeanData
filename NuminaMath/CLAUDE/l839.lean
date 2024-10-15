import Mathlib

namespace NUMINAMATH_CALUDE_exists_triangle_from_polygon_with_inscribed_circle_l839_83922

/-- A polygon with an inscribed circle. -/
structure PolygonWithInscribedCircle where
  /-- The number of sides of the polygon. -/
  n : ℕ
  /-- The lengths of the sides of the polygon. -/
  sides : Fin n → ℝ
  /-- The radius of the inscribed circle. -/
  radius : ℝ
  /-- All sides are positive. -/
  sides_positive : ∀ i, sides i > 0
  /-- The inscribed circle is tangent to all sides. -/
  tangent_to_all_sides : ∀ i, ∃ t, 0 < t ∧ t < sides i ∧ t = radius

/-- Theorem: In a polygon with an inscribed circle, there exist three sides that form a triangle. -/
theorem exists_triangle_from_polygon_with_inscribed_circle
  (p : PolygonWithInscribedCircle)
  (h : p.n ≥ 3) :
  ∃ i j k : Fin p.n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    p.sides i + p.sides j > p.sides k ∧
    p.sides j + p.sides k > p.sides i ∧
    p.sides k + p.sides i > p.sides j :=
  sorry

end NUMINAMATH_CALUDE_exists_triangle_from_polygon_with_inscribed_circle_l839_83922


namespace NUMINAMATH_CALUDE_moon_arrangements_count_l839_83913

/-- The number of distinct arrangements of letters in "MOON" -/
def moon_arrangements : ℕ := 12

/-- The total number of letters in "MOON" -/
def total_letters : ℕ := 4

/-- The number of times 'O' appears in "MOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of letters in "MOON" is 12 -/
theorem moon_arrangements_count : 
  moon_arrangements = (total_letters.factorial) / (o_count.factorial) := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_count_l839_83913


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l839_83928

theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius ^ 2 = 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l839_83928


namespace NUMINAMATH_CALUDE_cycling_route_length_l839_83952

theorem cycling_route_length (upper_segments : List ℝ) (left_segments : List ℝ) :
  upper_segments = [4, 7, 2] →
  left_segments = [6, 7] →
  2 * (upper_segments.sum + left_segments.sum) = 52 := by
  sorry

end NUMINAMATH_CALUDE_cycling_route_length_l839_83952


namespace NUMINAMATH_CALUDE_expression_evaluation_l839_83953

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  ((3*m + n) * (m - n) - (2*m - n)^2 + (m - 2*n) * (m + 2*n)) / (n / 2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l839_83953


namespace NUMINAMATH_CALUDE_largest_integral_x_l839_83911

theorem largest_integral_x : ∃ x : ℤ, x = 4 ∧ 
  (∀ y : ℤ, (1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 7/9 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l839_83911


namespace NUMINAMATH_CALUDE_f_properties_l839_83978

def f (x : ℝ) : ℝ := x^2 * (x - 3) * (x + 1)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  f (-1) = 0 ∧ 
  f 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_f_properties_l839_83978


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l839_83983

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U : Aᶜ = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l839_83983


namespace NUMINAMATH_CALUDE_max_projection_area_is_one_l839_83939

/-- Represents a tetrahedron with specific properties -/
structure Tetrahedron where
  /-- Two adjacent faces are isosceles right triangles -/
  adjacent_faces_isosceles_right : Bool
  /-- Hypotenuse of the isosceles right triangles is 2 -/
  hypotenuse : ℝ
  /-- Dihedral angle between the two adjacent faces is 60 degrees -/
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the maximum projection area is 1 -/
theorem max_projection_area_is_one (t : Tetrahedron) 
  (h1 : t.adjacent_faces_isosceles_right = true)
  (h2 : t.hypotenuse = 2)
  (h3 : t.dihedral_angle = π / 3) : 
  max_projection_area t = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_is_one_l839_83939


namespace NUMINAMATH_CALUDE_polynomial_value_l839_83967

theorem polynomial_value (x : ℝ) (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l839_83967


namespace NUMINAMATH_CALUDE_money_distribution_inconsistency_l839_83915

/-- Prove that the given conditions about money distribution are inconsistent -/
theorem money_distribution_inconsistency :
  ¬∃ (a b c : ℤ),
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧  -- Money amounts are non-negative
    a + c = 200 ∧            -- A and C together have 200
    b + c = 350 ∧            -- B and C together have 350
    c = 250                  -- C has 250
    := by sorry

end NUMINAMATH_CALUDE_money_distribution_inconsistency_l839_83915


namespace NUMINAMATH_CALUDE_sin_15_cos_15_l839_83951

theorem sin_15_cos_15 : 
  (∀ θ : ℝ, Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ) →
  Real.sin (30 * π / 180) = 1 / 2 →
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_l839_83951


namespace NUMINAMATH_CALUDE_basketball_tournament_matches_l839_83933

/-- The number of matches in a round-robin tournament with n teams -/
def roundRobinMatches (n : ℕ) : ℕ := n.choose 2

/-- The total number of matches played in the basketball tournament -/
def totalMatches (groups numTeams : ℕ) : ℕ :=
  groups * roundRobinMatches numTeams + roundRobinMatches groups

theorem basketball_tournament_matches :
  totalMatches 5 6 = 85 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_matches_l839_83933


namespace NUMINAMATH_CALUDE_cupcake_problem_l839_83970

theorem cupcake_problem (cupcake_cost : ℚ) (individual_payment : ℚ) :
  cupcake_cost = 3/2 →
  individual_payment = 9 →
  (2 * individual_payment) / cupcake_cost = 12 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_problem_l839_83970


namespace NUMINAMATH_CALUDE_line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l839_83959

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (linePerpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Plane → Prop)

-- Theorem 1
theorem line_perp_plane_parallel_plane_implies_planes_perp
  (α β : Plane) (l : Line)
  (h1 : linePerpendicular l α)
  (h2 : lineParallel l β) :
  perpendicular α β :=
sorry

-- Theorem 2
theorem planes_perp_parallel_implies_perp
  (α β γ : Plane)
  (h1 : perpendicular α β)
  (h2 : parallel α γ) :
  perpendicular γ β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_parallel_plane_implies_planes_perp_planes_perp_parallel_implies_perp_l839_83959


namespace NUMINAMATH_CALUDE_sin_alpha_minus_nine_pi_halves_l839_83930

theorem sin_alpha_minus_nine_pi_halves (α : Real)
  (h1 : π / 2 < α)
  (h2 : α < π)
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.cos α) :
  Real.sin (α - 9 * π / 2) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_minus_nine_pi_halves_l839_83930


namespace NUMINAMATH_CALUDE_night_crew_ratio_l839_83944

theorem night_crew_ratio (D N : ℕ) (B : ℝ) (h1 : D > 0) (h2 : N > 0) (h3 : B > 0) :
  (D * B) / ((D * B) + (N * (B / 2))) = 5 / 7 →
  (N : ℝ) / D = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l839_83944


namespace NUMINAMATH_CALUDE_loss_calculation_l839_83917

/-- Calculates the loss for the investor with larger capital -/
def loss_larger_investor (total_loss : ℚ) : ℚ :=
  (9 / 10) * total_loss

theorem loss_calculation (total_loss : ℚ) (pyarelal_loss : ℚ) 
  (h1 : total_loss = 900) 
  (h2 : pyarelal_loss = loss_larger_investor total_loss) : 
  pyarelal_loss = 810 := by
  sorry

end NUMINAMATH_CALUDE_loss_calculation_l839_83917


namespace NUMINAMATH_CALUDE_square_divisibility_l839_83908

theorem square_divisibility (n d : ℕ+) : 
  (n.val % d.val = 0) → 
  ((n.val^2 + d.val^2) % (d.val^2 * n.val + 1) = 0) → 
  n = d^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l839_83908


namespace NUMINAMATH_CALUDE_complex_number_existence_l839_83995

theorem complex_number_existence : ∃ z : ℂ, 
  (∃ r : ℝ, z + 5 / z = r) ∧ 
  (Complex.re (z + 3) = 2 * Complex.im z) ∧ 
  (z = (1 : ℝ) + (2 : ℝ) * Complex.I ∨ z = (-11 : ℝ) / 5 - (2 : ℝ) / 5 * Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l839_83995


namespace NUMINAMATH_CALUDE_power_of_81_l839_83963

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by sorry

end NUMINAMATH_CALUDE_power_of_81_l839_83963


namespace NUMINAMATH_CALUDE_smallest_valid_l839_83918

/-- A positive integer n is valid if 2n is a perfect square and 3n is a perfect cube. -/
def is_valid (n : ℕ+) : Prop :=
  ∃ k m : ℕ+, 2 * n = k^2 ∧ 3 * n = m^3

/-- 72 is the smallest positive integer that is valid. -/
theorem smallest_valid : (∀ n : ℕ+, n < 72 → ¬ is_valid n) ∧ is_valid 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_l839_83918


namespace NUMINAMATH_CALUDE_internal_angle_pentadecagon_is_156_l839_83921

/-- The measure of one internal angle of a regular pentadecagon -/
def internal_angle_pentadecagon : ℝ :=
  156

/-- The number of sides in a pentadecagon -/
def pentadecagon_sides : ℕ := 15

theorem internal_angle_pentadecagon_is_156 :
  internal_angle_pentadecagon = 156 :=
by
  sorry

#check internal_angle_pentadecagon_is_156

end NUMINAMATH_CALUDE_internal_angle_pentadecagon_is_156_l839_83921


namespace NUMINAMATH_CALUDE_largest_k_for_tree_graph_condition_l839_83987

/-- A tree graph with k vertices -/
structure TreeGraph (k : ℕ) where
  (vertices : Finset (Fin k))
  (edges : Finset (Fin k × Fin k))
  -- Add properties to ensure it's a tree

/-- Path between two vertices in a graph -/
def path (G : TreeGraph k) (u v : Fin k) : Finset (Fin k) := sorry

/-- Length of a path -/
def pathLength (p : Finset (Fin k)) : ℕ := sorry

/-- The condition for the existence of vertices u and v -/
def satisfiesCondition (G : TreeGraph k) (m n : ℕ) : Prop :=
  ∃ u v : Fin k, ∀ w : Fin k, 
    (∃ p : Finset (Fin k), p = path G u w ∧ pathLength p ≤ m) ∨
    (∃ p : Finset (Fin k), p = path G v w ∧ pathLength p ≤ n)

theorem largest_k_for_tree_graph_condition (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (∀ k : ℕ, k ≤ min (2*n + 2*m + 2) (3*n + 2) → 
    ∀ G : TreeGraph k, satisfiesCondition G m n) ∧
  (∀ k : ℕ, k > min (2*n + 2*m + 2) (3*n + 2) → 
    ∃ G : TreeGraph k, ¬satisfiesCondition G m n) :=
sorry

end NUMINAMATH_CALUDE_largest_k_for_tree_graph_condition_l839_83987


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l839_83941

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), 
    (a = n - 2 ∧ b = n ∧ c = n + 2) ∧  -- Three consecutive odd integers
    (Odd a ∧ Odd b ∧ Odd c) ∧           -- All are odd
    (a + c = 152)) →                    -- Sum of first and third is 152
  n = 76 :=                             -- Second integer is 76
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l839_83941


namespace NUMINAMATH_CALUDE_yan_distance_ratio_l839_83985

/-- Represents the scenario of Yan's journey between home and stadium. -/
structure YanJourney where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to his home
  y : ℝ  -- Distance from Yan to the stadium
  h_positive : w > 0 -- Assumption that walking speed is positive
  h_between : x > 0 ∧ y > 0 -- Assumption that Yan is between home and stadium

/-- The theorem stating the ratio of Yan's distances. -/
theorem yan_distance_ratio (j : YanJourney) : 
  j.y / j.w = j.x / j.w + (j.x + j.y) / (7 * j.w) → j.x / j.y = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_yan_distance_ratio_l839_83985


namespace NUMINAMATH_CALUDE_wrapping_cost_calculation_l839_83964

/-- Represents the number of boxes a roll of wrapping paper can wrap -/
structure WrapCapacity where
  shirt : ℕ
  xl : ℕ

/-- Represents the number of boxes to be wrapped -/
structure BoxesToWrap where
  shirt : ℕ
  xl : ℕ

/-- Calculates the total cost of wrapping paper needed -/
def totalCost (capacity : WrapCapacity) (boxes : BoxesToWrap) (price_per_roll : ℚ) : ℚ :=
  let rolls_needed_shirt := (boxes.shirt + capacity.shirt - 1) / capacity.shirt
  let rolls_needed_xl := (boxes.xl + capacity.xl - 1) / capacity.xl
  (rolls_needed_shirt + rolls_needed_xl : ℚ) * price_per_roll

theorem wrapping_cost_calculation 
  (capacity : WrapCapacity) 
  (boxes : BoxesToWrap) 
  (price_per_roll : ℚ) :
  capacity.shirt = 5 →
  capacity.xl = 3 →
  boxes.shirt = 20 →
  boxes.xl = 12 →
  price_per_roll = 4 →
  totalCost capacity boxes price_per_roll = 32 :=
sorry

end NUMINAMATH_CALUDE_wrapping_cost_calculation_l839_83964


namespace NUMINAMATH_CALUDE_sin_150_degrees_l839_83966

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l839_83966


namespace NUMINAMATH_CALUDE_parallelogram_df_and_area_l839_83991

/-- Represents a parallelogram ABCD with altitudes DE and DF -/
structure Parallelogram where
  -- Length of side DC
  dc : ℝ
  -- Length of EB (part of base AB)
  eb : ℝ
  -- Length of altitude DE
  de : ℝ
  -- Assumption that ABCD is a parallelogram
  is_parallelogram : True

/-- Properties of the parallelogram -/
def parallelogram_properties (p : Parallelogram) : Prop :=
  p.dc = 15 ∧ p.eb = 3 ∧ p.de = 5

/-- Theorem about the length of DF and the area of the parallelogram -/
theorem parallelogram_df_and_area (p : Parallelogram) 
  (h : parallelogram_properties p) :
  ∃ (df area : ℝ), df = 5 ∧ area = 75 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_df_and_area_l839_83991


namespace NUMINAMATH_CALUDE_max_inequality_constant_l839_83943

theorem max_inequality_constant : ∃ (M : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
  (x^2 + y^2)^3 ≥ M * (x^3 + y^3) * (x*y - x - y)) ∧ 
  (∀ (M' : ℝ), (∀ (x y : ℝ), x + y ≥ 0 → 
    (x^2 + y^2)^3 ≥ M' * (x^3 + y^3) * (x*y - x - y)) → M' ≤ M) ∧
  M = 32 :=
by sorry

end NUMINAMATH_CALUDE_max_inequality_constant_l839_83943


namespace NUMINAMATH_CALUDE_magnitude_of_z_l839_83958

open Complex

theorem magnitude_of_z : ∃ z : ℂ, z = 1 + 2*I + I^3 ∧ abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l839_83958


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l839_83916

theorem power_of_three_mod_seven : 3^2023 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l839_83916


namespace NUMINAMATH_CALUDE_trajectory_and_point_existence_l839_83904

-- Define the plane and points
variable (x y : ℝ)
def F : ℝ × ℝ := (1, 0)
def S : ℝ × ℝ := (x, y)

-- Define the distance ratio condition
def distance_ratio (S : ℝ × ℝ) : Prop :=
  Real.sqrt ((S.1 - F.1)^2 + S.2^2) / |S.1 - 2| = Real.sqrt 2 / 2

-- Define the trajectory equation
def trajectory_equation (S : ℝ × ℝ) : Prop :=
  S.1^2 / 2 + S.2^2 = 1

-- Define the line l (not perpendicular to x-axis)
variable (k : ℝ)
def line_l (x : ℝ) : ℝ := k * (x - 1)

-- Define points P and Q on the intersection of line_l and trajectory
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define point M
variable (m : ℝ)
def M : ℝ × ℝ := (m, 0)

-- Define the dot product condition
def dot_product_condition (M P Q : ℝ × ℝ) : Prop :=
  let MP := (P.1 - M.1, P.2 - M.2)
  let MQ := (Q.1 - M.1, Q.2 - M.2)
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  (MP.1 + MQ.1) * PQ.1 + (MP.2 + MQ.2) * PQ.2 = 0

-- Main theorem
theorem trajectory_and_point_existence :
  ∀ S, distance_ratio S →
    (trajectory_equation S ∧
     ∃ m, 0 ≤ m ∧ m < 1/2 ∧
       ∀ k ≠ 0, dot_product_condition (M m) P Q) := by sorry

end NUMINAMATH_CALUDE_trajectory_and_point_existence_l839_83904


namespace NUMINAMATH_CALUDE_arithmetic_operations_l839_83940

theorem arithmetic_operations : 
  (12 - (-18) + (-7) + (-15) = 8) ∧ 
  ((-1)^7 * 2 + (-3)^2 / 9 = -1) := by sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l839_83940


namespace NUMINAMATH_CALUDE_tangent_slope_and_sum_inequality_l839_83946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem tangent_slope_and_sum_inequality
  (a : ℝ)
  (h1 : (deriv (f a)) 0 = -1)
  (x₁ x₂ : ℝ)
  (h2 : x₁ < Real.log 2)
  (h3 : x₂ > Real.log 2)
  (h4 : f a x₁ = f a x₂) :
  x₁ + x₂ < 2 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_and_sum_inequality_l839_83946


namespace NUMINAMATH_CALUDE_door_lock_problem_l839_83976

def num_buttons : ℕ := 10
def buttons_to_press : ℕ := 3
def time_per_attempt : ℕ := 2

def total_combinations : ℕ := (num_buttons.choose buttons_to_press)

theorem door_lock_problem :
  (total_combinations * time_per_attempt = 240) ∧
  ((1 + total_combinations) / 2 * time_per_attempt = 121) ∧
  (((60 / time_per_attempt) - 1 : ℚ) / total_combinations = 29 / 120) := by
  sorry

end NUMINAMATH_CALUDE_door_lock_problem_l839_83976


namespace NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l839_83979

theorem square_sum_given_sum_square_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_square_and_product_l839_83979


namespace NUMINAMATH_CALUDE_pages_read_second_day_l839_83999

theorem pages_read_second_day 
  (total_pages : ℕ) 
  (pages_first_day : ℕ) 
  (pages_left : ℕ) 
  (h1 : total_pages = 95) 
  (h2 : pages_first_day = 18) 
  (h3 : pages_left = 19) : 
  total_pages - pages_left - pages_first_day = 58 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_second_day_l839_83999


namespace NUMINAMATH_CALUDE_simplify_expression_l839_83996

theorem simplify_expression (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (9 * x^2 * y^3) / (12 * x * y^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l839_83996


namespace NUMINAMATH_CALUDE_triangle_inequality_l839_83992

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan (B / 2) * Real.tan (C / 2) ≤ ((1 - Real.sin (A / 2)) / Real.cos (A / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l839_83992


namespace NUMINAMATH_CALUDE_correct_answers_count_l839_83934

/-- Represents a test with a specific scoring system. -/
structure Test where
  total_questions : ℕ
  score : ℕ → ℕ → ℤ
  all_answered : ℕ → ℕ → Prop

/-- Theorem stating the number of correct answers given the test conditions. -/
theorem correct_answers_count (test : Test)
    (h_total : test.total_questions = 100)
    (h_score : ∀ c i, test.score c i = c - 2 * i)
    (h_all_answered : ∀ c i, test.all_answered c i ↔ c + i = test.total_questions)
    (h_student_score : ∃ c i, test.all_answered c i ∧ test.score c i = 73) :
    ∃ c i, test.all_answered c i ∧ test.score c i = 73 ∧ c = 91 := by
  sorry

#check correct_answers_count

end NUMINAMATH_CALUDE_correct_answers_count_l839_83934


namespace NUMINAMATH_CALUDE_incircle_tangent_concurrency_l839_83990

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric concepts
variable (is_convex_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_incircle : Circle → Point → Point → Point → Prop)
variable (center_of : Circle → Point)
variable (second_common_external_tangent_touches : Circle → Circle → Point → Point → Prop)
variable (line_through : Point → Point → Set Point)
variable (concurrent : Set Point → Set Point → Set Point → Prop)

-- State the theorem
theorem incircle_tangent_concurrency 
  (A B C D : Point) 
  (ωA ωB : Circle) 
  (I J K L : Point) :
  is_convex_quadrilateral A B C D →
  is_incircle ωA A C D →
  is_incircle ωB B C D →
  I = center_of ωA →
  J = center_of ωB →
  second_common_external_tangent_touches ωA ωB K L →
  concurrent (line_through A K) (line_through B L) (line_through I J) :=
by sorry

end NUMINAMATH_CALUDE_incircle_tangent_concurrency_l839_83990


namespace NUMINAMATH_CALUDE_x_plus_y_value_l839_83955

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 2010)
  (eq2 : x + 2010 * Real.sin y = 2011)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2011 + Real.pi := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l839_83955


namespace NUMINAMATH_CALUDE_brent_candy_count_l839_83949

/-- The number of pieces of candy Brent has left after trick-or-treating and giving some away. -/
def candy_left : ℕ :=
  let kit_kat := 5
  let hershey := 3 * kit_kat
  let nerds := 8
  let lollipops := 11
  let baby_ruth := 10
  let reeses := baby_ruth / 2
  let total := kit_kat + hershey + nerds + lollipops + baby_ruth + reeses
  let given_away := 5
  total - given_away

/-- Theorem stating that Brent has 49 pieces of candy left. -/
theorem brent_candy_count : candy_left = 49 := by
  sorry

end NUMINAMATH_CALUDE_brent_candy_count_l839_83949


namespace NUMINAMATH_CALUDE_only_finance_opposite_meanings_l839_83900

-- Define a type for quantity pairs
inductive QuantityPair
  | Distance (d1 d2 : ℕ)
  | Finance (f1 f2 : ℤ)
  | HeightWeight (h w : ℚ)
  | Scores (s1 s2 : ℕ)

-- Define a function to check if a pair has opposite meanings
def hasOppositeMeanings (pair : QuantityPair) : Prop :=
  match pair with
  | QuantityPair.Finance f1 f2 => f1 * f2 < 0
  | _ => False

-- Theorem statement
theorem only_finance_opposite_meanings 
  (a : QuantityPair) 
  (b : QuantityPair) 
  (c : QuantityPair) 
  (d : QuantityPair) 
  (ha : a = QuantityPair.Distance 500 200)
  (hb : b = QuantityPair.Finance (-3000) 12000)
  (hc : c = QuantityPair.HeightWeight 1.5 (-2.4))
  (hd : d = QuantityPair.Scores 50 70) :
  hasOppositeMeanings b ∧ 
  ¬hasOppositeMeanings a ∧ 
  ¬hasOppositeMeanings c ∧ 
  ¬hasOppositeMeanings d := by
  sorry

end NUMINAMATH_CALUDE_only_finance_opposite_meanings_l839_83900


namespace NUMINAMATH_CALUDE_cynthia_potato_harvest_l839_83960

theorem cynthia_potato_harvest :
  ∀ (P : ℕ),
  (P ≥ 13) →
  (P - 13) % 2 = 0 →
  ((P - 13) / 2 - 13 = 436) →
  P = 911 :=
by
  sorry

end NUMINAMATH_CALUDE_cynthia_potato_harvest_l839_83960


namespace NUMINAMATH_CALUDE_sum_of_roots_l839_83929

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 2026*x = 2023)
  (hy : y^3 + 6*y^2 + 2035*y = -4053) : 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l839_83929


namespace NUMINAMATH_CALUDE_am_length_l839_83910

/-- Given points M, A, and B on a straight line, with AM twice as long as BM and AB = 6,
    the length of AM is either 4 or 12. -/
theorem am_length (M A B : ℝ) : 
  (∃ t : ℝ, M = t * A + (1 - t) * B) →  -- M, A, B are collinear
  abs (A - M) = 2 * abs (B - M) →       -- AM is twice as long as BM
  abs (A - B) = 6 →                     -- AB = 6
  abs (A - M) = 4 ∨ abs (A - M) = 12 := by
sorry


end NUMINAMATH_CALUDE_am_length_l839_83910


namespace NUMINAMATH_CALUDE_max_intersection_points_l839_83907

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of intersection points between a circle and a line --/
def intersection_count (circle : Circle) (line : Line) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of intersection points between a circle and a line is 2 --/
theorem max_intersection_points (circle : Circle) (line : Line) :
  intersection_count circle line ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_intersection_points_l839_83907


namespace NUMINAMATH_CALUDE_water_volume_for_spheres_in_cylinder_l839_83923

/-- The volume of water required to cover two spheres in a cylinder -/
theorem water_volume_for_spheres_in_cylinder (cylinder_diameter cylinder_height : ℝ)
  (small_sphere_radius large_sphere_radius : ℝ) :
  cylinder_diameter = 27 →
  cylinder_height = 30 →
  small_sphere_radius = 6 →
  large_sphere_radius = 9 →
  (π * (cylinder_diameter / 2)^2 * (large_sphere_radius + small_sphere_radius + large_sphere_radius)) -
  (4/3 * π * small_sphere_radius^3 + 4/3 * π * large_sphere_radius^3) = 3114 * π :=
by sorry

end NUMINAMATH_CALUDE_water_volume_for_spheres_in_cylinder_l839_83923


namespace NUMINAMATH_CALUDE_chinese_sturgeon_probability_l839_83956

theorem chinese_sturgeon_probability (p_maturity p_spawn_reproduce : ℝ) 
  (h_maturity : p_maturity = 0.15)
  (h_spawn_reproduce : p_spawn_reproduce = 0.05) :
  p_spawn_reproduce / p_maturity = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_sturgeon_probability_l839_83956


namespace NUMINAMATH_CALUDE_exam_average_l839_83919

theorem exam_average (total_candidates : ℕ) (first_ten_avg : ℚ) (last_eleven_avg : ℚ) (eleventh_candidate_score : ℕ) :
  total_candidates = 22 →
  first_ten_avg = 55 →
  last_eleven_avg = 40 →
  eleventh_candidate_score = 66 →
  (((first_ten_avg * 10) + eleventh_candidate_score + (last_eleven_avg * 11 - eleventh_candidate_score)) / total_candidates : ℚ) = 45 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l839_83919


namespace NUMINAMATH_CALUDE_division_of_monomials_l839_83948

-- Define variables
variable (x y : ℝ)

-- Define the theorem
theorem division_of_monomials (x y : ℝ) :
  x ≠ 0 → y ≠ 0 → (-4 * x^5 * y^3) / (2 * x^3 * y) = -2 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_monomials_l839_83948


namespace NUMINAMATH_CALUDE_tan_a_plus_pi_third_l839_83984

theorem tan_a_plus_pi_third (a : Real) (h : Real.tan a = Real.sqrt 3) : 
  Real.tan (a + π/3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_a_plus_pi_third_l839_83984


namespace NUMINAMATH_CALUDE_cube_sum_squares_l839_83977

theorem cube_sum_squares (a b t : ℝ) (h : a + b = t^2) :
  ∃ x y z : ℝ, 2 * (a^3 + b^3) = x^2 + y^2 + z^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_squares_l839_83977


namespace NUMINAMATH_CALUDE_counterfeit_coin_findable_l839_83927

/-- Represents the type of scale: regular or magical -/
inductive ScaleType
| Regular
| Magical

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a coin -/
structure Coin := (id : Nat)

/-- Represents a weighing action -/
structure Weighing := 
  (left : List Coin)
  (right : List Coin)

/-- Represents the state of the problem -/
structure ProblemState :=
  (coins : List Coin)
  (counterfeitCoin : Coin)
  (scaleType : ScaleType)

/-- Function to perform a weighing -/
def performWeighing (state : ProblemState) (w : Weighing) : WeighingResult :=
  sorry

/-- Function representing a strategy to find the counterfeit coin -/
def findCounterfeitStrategy : ProblemState → List Weighing → Option Coin :=
  sorry

/-- Theorem stating that it's possible to find the counterfeit coin in 3 weighings -/
theorem counterfeit_coin_findable :
  ∀ (coins : List Coin) (counterfeitCoin : Coin) (scaleType : ScaleType),
    coins.length = 12 →
    counterfeitCoin ∈ coins →
    ∃ (strategy : List Weighing),
      strategy.length ≤ 3 ∧
      (findCounterfeitStrategy ⟨coins, counterfeitCoin, scaleType⟩ strategy = some counterfeitCoin) :=
sorry

end NUMINAMATH_CALUDE_counterfeit_coin_findable_l839_83927


namespace NUMINAMATH_CALUDE_last_digit_base4_last_digit_390_base4_l839_83954

/-- Convert a natural number to its base-4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- The last digit of a number in base-4 is the same as the remainder when divided by 4 -/
theorem last_digit_base4 (n : ℕ) : 
  (toBase4 n).getLast? = some (n % 4) :=
sorry

/-- The last digit of 390 in base-4 is 2 -/
theorem last_digit_390_base4 : 
  (toBase4 390).getLast? = some 2 :=
sorry

end NUMINAMATH_CALUDE_last_digit_base4_last_digit_390_base4_l839_83954


namespace NUMINAMATH_CALUDE_club_member_ratio_l839_83994

/-- 
Given a club with current members and additional members,
prove that the ratio of new total members to current members is 5:2.
-/
theorem club_member_ratio (current_members additional_members : ℕ) 
  (h1 : current_members = 10)
  (h2 : additional_members = 15) : 
  (current_members + additional_members) / current_members = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_club_member_ratio_l839_83994


namespace NUMINAMATH_CALUDE_range_of_t_l839_83902

/-- Given a set A containing 1 and a real number t, prove that the range of t is all real numbers except 1. -/
theorem range_of_t (t : ℝ) (A : Set ℝ) (h : A = {1, t}) : 
  {x : ℝ | x ≠ 1} = {x : ℝ | ∃ (s : Set ℝ), s = {1, x} ∧ s = A} :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l839_83902


namespace NUMINAMATH_CALUDE_circle_area_radius_increase_l839_83938

theorem circle_area_radius_increase : 
  ∀ (r : ℝ) (r' : ℝ), r > 0 → r' > 0 →
  (π * r' ^ 2 = 4 * π * r ^ 2) → 
  (r' - r) / r * 100 = 100 := by
sorry

end NUMINAMATH_CALUDE_circle_area_radius_increase_l839_83938


namespace NUMINAMATH_CALUDE_batsman_second_set_matches_l839_83932

/-- Given information about a batsman's performance, prove the number of matches in the second set -/
theorem batsman_second_set_matches 
  (first_set_matches : ℕ) 
  (total_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (total_average : ℝ) 
  (h1 : first_set_matches = 35)
  (h2 : total_matches = 49)
  (h3 : first_set_average = 36)
  (h4 : second_set_average = 15)
  (h5 : total_average = 30) :
  total_matches - first_set_matches = 14 := by
  sorry

#check batsman_second_set_matches

end NUMINAMATH_CALUDE_batsman_second_set_matches_l839_83932


namespace NUMINAMATH_CALUDE_intersection_A_B_l839_83931

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : Set ℝ := {x | |x - 2| < 2}

theorem intersection_A_B : ∀ x : ℝ, x ∈ (A ∩ B) ↔ 0 < x ∧ x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l839_83931


namespace NUMINAMATH_CALUDE_product_of_pairs_l839_83901

/-- Given three pairs of real numbers satisfying specific equations, 
    their product in a certain form equals a specific value -/
theorem product_of_pairs (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (eq1₁ : x₁^3 - 3*x₁*y₁^2 = 2007)
  (eq2₁ : y₁^3 - 3*x₁^2*y₁ = 2006)
  (eq1₂ : x₂^3 - 3*x₂*y₂^2 = 2007)
  (eq2₂ : y₂^3 - 3*x₂^2*y₂ = 2006)
  (eq1₃ : x₃^3 - 3*x₃*y₃^2 = 2007)
  (eq2₃ : y₃^3 - 3*x₃^2*y₃ = 2006) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/1003.5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_pairs_l839_83901


namespace NUMINAMATH_CALUDE_fraction_equals_decimal_l839_83936

theorem fraction_equals_decimal : (1 : ℚ) / 4 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_decimal_l839_83936


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_less_120_l839_83981

theorem greatest_common_multiple_9_15_less_120 : ∃ n : ℕ,
  n = 90 ∧
  9 ∣ n ∧
  15 ∣ n ∧
  n < 120 ∧
  ∀ m : ℕ, (9 ∣ m ∧ 15 ∣ m ∧ m < 120) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_less_120_l839_83981


namespace NUMINAMATH_CALUDE_birthday_250_years_ago_l839_83920

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week that is n days before the given day -/
def daysBefore (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  sorry

/-- Calculates the number of leap years in a 250-year period, excluding certain century years -/
def leapYearsIn250Years : ℕ :=
  sorry

/-- Represents the number of days to go backwards for 250 years -/
def daysBackFor250Years : ℕ :=
  sorry

theorem birthday_250_years_ago (anniversary_day : DayOfWeek) : 
  anniversary_day = DayOfWeek.Tuesday → 
  daysBefore anniversary_day daysBackFor250Years = DayOfWeek.Saturday :=
sorry

end NUMINAMATH_CALUDE_birthday_250_years_ago_l839_83920


namespace NUMINAMATH_CALUDE_amoebas_after_two_weeks_l839_83972

/-- The number of amoebas in the tank on a given day -/
def amoebas (day : ℕ) : ℕ :=
  if day ≤ 7 then
    2^day
  else
    2^7 * 3^(day - 7)

/-- Theorem stating the number of amoebas after 14 days -/
theorem amoebas_after_two_weeks : amoebas 14 = 279936 := by
  sorry

end NUMINAMATH_CALUDE_amoebas_after_two_weeks_l839_83972


namespace NUMINAMATH_CALUDE_complex_equation_solution_l839_83965

theorem complex_equation_solution (z : ℂ) :
  z / (1 - Complex.I) = Complex.I ^ 2016 + Complex.I ^ 2017 → z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l839_83965


namespace NUMINAMATH_CALUDE_complement_of_A_l839_83989

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem complement_of_A : Set.compl A = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l839_83989


namespace NUMINAMATH_CALUDE_tournament_has_cycle_of_length_3_l839_83926

/-- A tournament is a complete directed graph where each edge represents a match outcome. -/
def Tournament (n : ℕ) := Fin n → Fin n → Prop

/-- In a valid tournament, every pair of distinct players has exactly one match outcome. -/
def is_valid_tournament (t : Tournament n) : Prop :=
  ∀ i j : Fin n, i ≠ j → (t i j ∧ ¬t j i) ∨ (t j i ∧ ¬t i j)

/-- A player wins at least one match if there exists another player they defeated. -/
def player_wins_at_least_one (t : Tournament n) (i : Fin n) : Prop :=
  ∃ j : Fin n, t i j

/-- A cycle of length 3 in a tournament. -/
def has_cycle_of_length_3 (t : Tournament n) : Prop :=
  ∃ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ t a b ∧ t b c ∧ t c a

theorem tournament_has_cycle_of_length_3 :
  ∀ (t : Tournament 12),
    is_valid_tournament t →
    (∀ i : Fin 12, player_wins_at_least_one t i) →
    has_cycle_of_length_3 t :=
by sorry


end NUMINAMATH_CALUDE_tournament_has_cycle_of_length_3_l839_83926


namespace NUMINAMATH_CALUDE_ricardo_coin_difference_l839_83998

theorem ricardo_coin_difference :
  ∀ (one_cent five_cent : ℕ),
    one_cent + five_cent = 2020 →
    one_cent ≥ 1 →
    five_cent ≥ 1 →
    (5 * 2019 + 1) - (2019 + 5) = 8072 :=
by
  sorry

end NUMINAMATH_CALUDE_ricardo_coin_difference_l839_83998


namespace NUMINAMATH_CALUDE_angle_not_in_second_quadrant_l839_83945

def is_in_second_quadrant (angle : ℝ) : Prop :=
  let normalized_angle := angle % 360
  90 < normalized_angle ∧ normalized_angle ≤ 180

theorem angle_not_in_second_quadrant :
  is_in_second_quadrant 160 ∧
  is_in_second_quadrant 480 ∧
  is_in_second_quadrant (-960) ∧
  ¬ is_in_second_quadrant 1530 :=
by sorry

end NUMINAMATH_CALUDE_angle_not_in_second_quadrant_l839_83945


namespace NUMINAMATH_CALUDE_johns_quilt_cost_l839_83957

/-- The cost of a rectangular quilt -/
def quilt_cost (length width price_per_sqft : ℝ) : ℝ :=
  length * width * price_per_sqft

/-- Theorem: The cost of John's quilt is $2240 -/
theorem johns_quilt_cost :
  quilt_cost 7 8 40 = 2240 := by
  sorry

end NUMINAMATH_CALUDE_johns_quilt_cost_l839_83957


namespace NUMINAMATH_CALUDE_inner_hexagon_area_l839_83993

/-- Given a hexagon ABCDEF with specific area properties, prove the area of the inner hexagon A₁B₁C₁D₁E₁F₁ -/
theorem inner_hexagon_area 
  (area_ABCDEF : ℝ) 
  (area_triangle : ℝ) 
  (area_shaded : ℝ) 
  (h1 : area_ABCDEF = 2010) 
  (h2 : area_triangle = 335) 
  (h3 : area_shaded = 670) : 
  area_ABCDEF - (6 * area_triangle + area_shaded) / 2 = 670 := by
sorry

end NUMINAMATH_CALUDE_inner_hexagon_area_l839_83993


namespace NUMINAMATH_CALUDE_max_negative_integers_l839_83935

theorem max_negative_integers
  (a b c d e f : ℤ)
  (h : a * b + c * d * e * f < 0) :
  ∃ (neg_count : ℕ),
    neg_count ≤ 4 ∧
    (∃ (na nb nc nd ne nf : ℕ),
      (na + nb + nc + nd + ne + nf = neg_count) ∧
      (a < 0 ↔ na = 1) ∧
      (b < 0 ↔ nb = 1) ∧
      (c < 0 ↔ nc = 1) ∧
      (d < 0 ↔ nd = 1) ∧
      (e < 0 ↔ ne = 1) ∧
      (f < 0 ↔ nf = 1)) ∧
    ∀ (m : ℕ), m > neg_count →
      ¬∃ (ma mb mc md me mf : ℕ),
        (ma + mb + mc + md + me + mf = m) ∧
        (a < 0 ↔ ma = 1) ∧
        (b < 0 ↔ mb = 1) ∧
        (c < 0 ↔ mc = 1) ∧
        (d < 0 ↔ md = 1) ∧
        (e < 0 ↔ me = 1) ∧
        (f < 0 ↔ mf = 1) := by
  sorry

end NUMINAMATH_CALUDE_max_negative_integers_l839_83935


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_count_l839_83909

/- Define the number of schools -/
def num_schools : ℕ := 3

/- Define the number of members per school -/
def members_per_school : ℕ := 6

/- Define the number of representatives from the host school -/
def host_representatives : ℕ := 3

/- Define the number of representatives from each non-host school -/
def non_host_representatives : ℕ := 1

/- Function to calculate the number of ways to arrange the meeting -/
def presidency_meeting_arrangements : ℕ :=
  num_schools * (members_per_school.choose host_representatives) * 
  (members_per_school.choose non_host_representatives) * 
  (members_per_school.choose non_host_representatives)

/- Theorem stating the number of arrangements -/
theorem presidency_meeting_arrangements_count :
  presidency_meeting_arrangements = 2160 := by
  sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_count_l839_83909


namespace NUMINAMATH_CALUDE_first_divisor_problem_l839_83988

theorem first_divisor_problem (n d : ℕ) : 
  n > 1 →
  n % d = 1 →
  n % 7 = 1 →
  (∀ m : ℕ, m > 1 ∧ m % d = 1 ∧ m % 7 = 1 → m ≥ n) →
  n = 175 →
  d = 29 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l839_83988


namespace NUMINAMATH_CALUDE_complex_equation_solution_l839_83914

theorem complex_equation_solution (a : ℝ) : (Complex.mk 2 a) * (Complex.mk a (-2)) = 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l839_83914


namespace NUMINAMATH_CALUDE_triangle_inequality_l839_83969

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  Real.sin t.A * Real.cos t.C + t.A * Real.cos t.B > 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l839_83969


namespace NUMINAMATH_CALUDE_tenth_diagram_shading_l839_83961

/-- Represents a square grid with a specific shading pattern -/
structure ShadedGrid (n : ℕ) where
  size : ℕ
  shaded_squares : ℕ
  h_size : size = n * n
  h_shaded : shaded_squares = (n - 1) * (n / 2) + n

/-- The fraction of shaded squares in the grid -/
def shaded_fraction (grid : ShadedGrid n) : ℚ :=
  grid.shaded_squares / grid.size

theorem tenth_diagram_shading :
  ∃ (grid : ShadedGrid 10), shaded_fraction grid = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_tenth_diagram_shading_l839_83961


namespace NUMINAMATH_CALUDE_pond_diameter_l839_83942

/-- The diameter of a circular pond given specific conditions -/
theorem pond_diameter : ∃ (h k r : ℝ),
  (4 - h)^2 + (11 - k)^2 = r^2 ∧
  (12 - h)^2 + (9 - k)^2 = r^2 ∧
  (2 - h)^2 + (7 - k)^2 = (r - 1)^2 ∧
  2 * r = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_pond_diameter_l839_83942


namespace NUMINAMATH_CALUDE_power_of_five_times_112_l839_83974

theorem power_of_five_times_112 : (112 * 5^4) = 70000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_times_112_l839_83974


namespace NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l839_83924

-- Define the fractions
def f1 : ℚ := 1 / 13
def f2 : ℚ := 1 / 11

-- Define the sum of the fractions
def sum : ℚ := f1 + f2

-- Define a function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem thirtieth_digit_of_sum : nthDigitAfterDecimal sum 30 = 9 := by sorry

end NUMINAMATH_CALUDE_thirtieth_digit_of_sum_l839_83924


namespace NUMINAMATH_CALUDE_certain_number_problem_l839_83947

theorem certain_number_problem (x : ℝ) : 
  3.6 * x * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001 → x = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l839_83947


namespace NUMINAMATH_CALUDE_wraps_percentage_increase_l839_83968

/-- Given John's raw squat weight, the additional weight from sleeves, and the difference between wraps and sleeves, 
    calculate the percentage increase wraps provide to his raw squat. -/
theorem wraps_percentage_increase 
  (raw_squat : ℝ) 
  (sleeves_addition : ℝ) 
  (wraps_vs_sleeves_difference : ℝ) 
  (h1 : raw_squat = 600) 
  (h2 : sleeves_addition = 30) 
  (h3 : wraps_vs_sleeves_difference = 120) : 
  (raw_squat + sleeves_addition + wraps_vs_sleeves_difference - raw_squat) / raw_squat * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_wraps_percentage_increase_l839_83968


namespace NUMINAMATH_CALUDE_birds_joining_fence_l839_83925

/-- Proves that 2 additional birds joined the fence given the initial and final conditions -/
theorem birds_joining_fence :
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let additional_birds : ℕ := 2
  let final_birds : ℕ := initial_birds + additional_birds
  let final_storks : ℕ := initial_storks
  final_birds = final_storks + 1 :=
by sorry

end NUMINAMATH_CALUDE_birds_joining_fence_l839_83925


namespace NUMINAMATH_CALUDE_suji_age_is_16_l839_83937

/-- Represents the ages of Abi, Suji, and Ravi -/
structure Ages where
  x : ℕ
  deriving Repr

def Ages.abi (a : Ages) : ℕ := 5 * a.x
def Ages.suji (a : Ages) : ℕ := 4 * a.x
def Ages.ravi (a : Ages) : ℕ := 3 * a.x

def Ages.future_abi (a : Ages) : ℕ := a.abi + 6
def Ages.future_suji (a : Ages) : ℕ := a.suji + 6
def Ages.future_ravi (a : Ages) : ℕ := a.ravi + 6

/-- The theorem stating that Suji's present age is 16 years -/
theorem suji_age_is_16 (a : Ages) : 
  (a.future_abi / a.future_suji = 13 / 11) ∧ 
  (a.future_suji / a.future_ravi = 11 / 9) → 
  a.suji = 16 := by
  sorry

#eval Ages.suji { x := 4 }

end NUMINAMATH_CALUDE_suji_age_is_16_l839_83937


namespace NUMINAMATH_CALUDE_symmetric_difference_A_B_l839_83982

/-- Set difference -/
def set_difference (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}

/-- Symmetric difference -/
def symmetric_difference (M N : Set ℝ) : Set ℝ :=
  set_difference M N ∪ set_difference N M

/-- Set A -/
def A : Set ℝ := {t | ∃ x, t = x^2 - 3*x}

/-- Set B -/
def B : Set ℝ := {x | ∃ y, y = Real.log (-x)}

theorem symmetric_difference_A_B :
  symmetric_difference A B = {x | x < -9/4 ∨ x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_A_B_l839_83982


namespace NUMINAMATH_CALUDE_sector_max_area_l839_83975

/-- Given a sector with constant perimeter a, prove that the maximum area is a²/16
    and this occurs when the central angle α is 2. -/
theorem sector_max_area (a : ℝ) (h : a > 0) :
  ∃ (S : ℝ) (α : ℝ),
    S = a^2 / 16 ∧
    α = 2 ∧
    ∀ (S' : ℝ) (α' : ℝ),
      (∃ (r : ℝ), 2 * r + r * α' = a ∧ S' = r^2 * α' / 2) →
      S' ≤ S :=
by sorry

end NUMINAMATH_CALUDE_sector_max_area_l839_83975


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_3a2_eq_b2_plus_1_l839_83905

theorem no_integer_solutions_for_3a2_eq_b2_plus_1 :
  ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_3a2_eq_b2_plus_1_l839_83905


namespace NUMINAMATH_CALUDE_laptop_cost_proof_l839_83973

/-- The cost of the laptop satisfies the given conditions -/
theorem laptop_cost_proof (monthly_installment : ℝ) (down_payment_percent : ℝ) 
  (additional_down_payment : ℝ) (months_paid : ℕ) (remaining_balance : ℝ) :
  monthly_installment = 65 →
  down_payment_percent = 0.2 →
  additional_down_payment = 20 →
  months_paid = 4 →
  remaining_balance = 520 →
  ∃ (cost : ℝ), 
    cost - (down_payment_percent * cost + additional_down_payment + monthly_installment * months_paid) = remaining_balance ∧
    cost = 1000 := by
  sorry

end NUMINAMATH_CALUDE_laptop_cost_proof_l839_83973


namespace NUMINAMATH_CALUDE_prob_third_white_specific_urn_l839_83980

/-- An urn with white and black balls -/
structure Urn where
  white : ℕ
  black : ℕ

/-- The probability of drawing a white ball as the third ball -/
def prob_third_white (u : Urn) : ℚ :=
  u.white / (u.white + u.black)

/-- The theorem statement -/
theorem prob_third_white_specific_urn :
  let u : Urn := ⟨6, 5⟩
  prob_third_white u = 6 / 11 := by
  sorry

#eval prob_third_white ⟨6, 5⟩

end NUMINAMATH_CALUDE_prob_third_white_specific_urn_l839_83980


namespace NUMINAMATH_CALUDE_age_ratio_problem_l839_83971

/-- Given Tom's current age t and Sara's current age s, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_problem (t s : ℕ) (h1 : t - 3 = 2 * (s - 3)) (h2 : t - 8 = 3 * (s - 8)) :
  ∃ x : ℕ, x = 7 ∧ (t + x : ℚ) / (s + x) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l839_83971


namespace NUMINAMATH_CALUDE_equation_solutions_l839_83903

theorem equation_solutions :
  (∃ x : ℚ, 2 * x - 3 = 3 * (x + 1) ∧ x = -6) ∧
  (∃ x : ℚ, (1/2) * x - (9 * x - 2) / 6 - 2 = 0 ∧ x = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l839_83903


namespace NUMINAMATH_CALUDE_cube_cutting_l839_83986

theorem cube_cutting (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^3 : ℕ) = 98 + b^3 → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l839_83986


namespace NUMINAMATH_CALUDE_algebra_test_male_students_l839_83950

/-- Proves that given the conditions of the algebra test problem, the number of male students is 8 -/
theorem algebra_test_male_students
  (total_average : ℝ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ)
  (h_total_average : total_average = 90)
  (h_male_average : male_average = 83)
  (h_female_average : female_average = 92)
  (h_female_count : female_count = 28) :
  ∃ (male_count : ℕ),
    (male_count : ℝ) * male_average + (female_count : ℝ) * female_average =
      (male_count + female_count : ℝ) * total_average ∧
    male_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_male_students_l839_83950


namespace NUMINAMATH_CALUDE_tan_G_in_right_triangle_l839_83912

theorem tan_G_in_right_triangle (GH FG : ℝ) (h_right_triangle : GH^2 + FG^2 = 25^2)
  (h_GH : GH = 20) (h_FG : FG = 25) : Real.tan (Real.arcsin (GH / FG)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_G_in_right_triangle_l839_83912


namespace NUMINAMATH_CALUDE_crayons_remaining_l839_83906

theorem crayons_remaining (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) : 
  initial_crayons = 48 → 
  kiley_fraction = 1/4 →
  joe_fraction = 1/2 →
  (initial_crayons - (kiley_fraction * initial_crayons).floor - 
   (joe_fraction * (initial_crayons - (kiley_fraction * initial_crayons).floor)).floor) = 18 :=
by sorry

end NUMINAMATH_CALUDE_crayons_remaining_l839_83906


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l839_83962

theorem fractional_equation_solution :
  ∀ x : ℝ, x ≠ 0 → x ≠ 3 → (2 / (x - 3) = 3 / x) ↔ x = 9 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l839_83962


namespace NUMINAMATH_CALUDE_max_value_K_l839_83997

/-- The maximum value of K for x₁, x₂, x₃, x₄ ∈ [0,1] --/
theorem max_value_K : 
  ∃ (K_max : ℝ), K_max = Real.sqrt 5 / 125 ∧ 
  ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    0 ≤ x₁ ∧ x₁ ≤ 1 ∧ 
    0 ≤ x₂ ∧ x₂ ≤ 1 ∧ 
    0 ≤ x₃ ∧ x₃ ≤ 1 ∧ 
    0 ≤ x₄ ∧ x₄ ≤ 1 → 
    let K := |x₁ - x₂| * |x₁ - x₃| * |x₁ - x₄| * |x₂ - x₃| * |x₂ - x₄| * |x₃ - x₄|
    K ≤ K_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_K_l839_83997
