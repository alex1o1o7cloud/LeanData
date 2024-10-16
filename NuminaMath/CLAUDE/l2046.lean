import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2046_204623

theorem reciprocal_sum_theorem (a b c : ℝ) (h : 1 / a + 1 / b = 1 / c) : c = a * b / (b + a) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l2046_204623


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2046_204625

theorem triangle_angle_measure (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 - c^2 + b^2 = -Real.sqrt 3 * a * b) : 
  Real.cos (Real.pi / 6) = (a^2 + b^2 - c^2) / (2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2046_204625


namespace NUMINAMATH_CALUDE_vectors_theorem_l2046_204675

/-- Two non-collinear vectors in a plane -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  noncollinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vectors AB, CB, and CD -/
def vectors_relation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) (k : ℝ) : Prop :=
  ∃ (A B C D : V),
    B - A = ncv.e₁ - k • ncv.e₂ ∧
    B - C = 2 • ncv.e₁ + ncv.e₂ ∧
    D - C = 3 • ncv.e₁ - ncv.e₂

/-- Collinearity of points A, B, and D -/
def collinear (V : Type*) [AddCommGroup V] [Module ℝ V] (A B D : V) : Prop :=
  ∃ (t : ℝ), D - A = t • (B - A)

/-- The main theorem -/
theorem vectors_theorem (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (ncv : NonCollinearVectors V) :
  ∀ k, vectors_relation V ncv k → 
  (∃ A B D, collinear V A B D) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_vectors_theorem_l2046_204675


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_property_l2046_204630

theorem smallest_integer_with_divisibility_property : ∃ (n : ℕ), 
  (∀ i ∈ Finset.range 28, i.succ ∣ n) ∧ 
  ¬(29 ∣ n) ∧ 
  ¬(30 ∣ n) ∧
  (∀ m : ℕ, m < n → ¬(∀ i ∈ Finset.range 28, i.succ ∣ m) ∨ (29 ∣ m) ∨ (30 ∣ m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_property_l2046_204630


namespace NUMINAMATH_CALUDE_sin_390_degrees_l2046_204667

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_390_degrees_l2046_204667


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l2046_204601

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l2046_204601


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2046_204695

/-- Proves that if an article is sold for 250 Rs. with a 25% profit, its cost price is 200 Rs. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 250)
  (h2 : profit_percentage = 25) :
  selling_price / (1 + profit_percentage / 100) = 200 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l2046_204695


namespace NUMINAMATH_CALUDE_range_of_m_m_value_for_diameter_l2046_204647

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x y, circle_eq x y m) → m < 37/4 :=
sorry

-- Define the condition for PQ being diameter of circle passing through origin
def pq_diameter_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = 0

-- Theorem for the value of m when PQ is diameter of circle passing through origin
theorem m_value_for_diameter (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    circle_eq x₁ y₁ m ∧ circle_eq x₂ y₂ m ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    pq_diameter_through_origin x₁ y₁ x₂ y₂) →
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_for_diameter_l2046_204647


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2046_204645

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = 
    (polygon_interior_angle_sum octagon_sides) / octagon_sides :=
by sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2046_204645


namespace NUMINAMATH_CALUDE_three_zero_points_implies_k_leq_neg_two_l2046_204613

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 2 else Real.log x

theorem three_zero_points_implies_k_leq_neg_two (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    |f k x₁| + k = 0 ∧ |f k x₂| + k = 0 ∧ |f k x₃| + k = 0) →
  k ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_three_zero_points_implies_k_leq_neg_two_l2046_204613


namespace NUMINAMATH_CALUDE_two_lines_in_cube_l2046_204685

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Represents a line in 3D space -/
structure Line where
  point : Point
  direction : ℝ × ℝ × ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  point : Point
  normal : ℝ × ℝ × ℝ

/-- Calculates the angle between a line and a plane -/
def angle_line_plane (l : Line) (p : Plane) : ℝ := sorry

/-- Checks if a point is on an edge of the cube -/
def point_on_edge (c : Cube) (p : Point) : Prop := sorry

/-- Counts the number of lines passing through a point and making a specific angle with two planes -/
def count_lines (c : Cube) (p : Point) (angle : ℝ) (plane1 plane2 : Plane) : ℕ := sorry

/-- The main theorem statement -/
theorem two_lines_in_cube (c : Cube) (p : Point) :
  point_on_edge c p →
  let plane_abcd := Plane.mk sorry sorry
  let plane_abc1d1 := Plane.mk sorry sorry
  count_lines c p (30 * π / 180) plane_abcd plane_abc1d1 = 2 := by sorry

end NUMINAMATH_CALUDE_two_lines_in_cube_l2046_204685


namespace NUMINAMATH_CALUDE_log_relation_l2046_204679

theorem log_relation (c b : ℝ) (hc : c = Real.log 81 / Real.log 4) (hb : b = Real.log 3 / Real.log 2) : 
  c = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2046_204679


namespace NUMINAMATH_CALUDE_bicycle_speed_problem_l2046_204650

/-- Proves that given a distance of 12 km, if person A's speed is 1.2 times person B's speed,
    and A arrives 1/6 hour earlier than B, then B's speed is 12 km/h. -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
    (h1 : distance = 12)
    (h2 : speed_ratio = 1.2)
    (h3 : time_difference = 1/6) : 
  let speed_B := distance / (distance / (speed_ratio * (distance / time_difference)) + time_difference)
  speed_B = 12 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_speed_problem_l2046_204650


namespace NUMINAMATH_CALUDE_range_of_a_l2046_204690

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - a ≥ 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) ↔ 
  (-1 < a ∧ a ≤ 0) ∨ (a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2046_204690


namespace NUMINAMATH_CALUDE_cost_of_900_candies_l2046_204626

/-- The cost of buying a given number of chocolate candies -/
def cost_of_candies (num_candies : ℕ) : ℚ :=
  let candies_per_box : ℕ := 30
  let cost_per_box : ℚ := 7.5
  let discount_threshold : ℕ := 500
  let discount_rate : ℚ := 0.1
  let num_boxes : ℕ := num_candies / candies_per_box
  let discounted_cost_per_box : ℚ := if num_candies > discount_threshold then cost_per_box * (1 - discount_rate) else cost_per_box
  (num_boxes : ℚ) * discounted_cost_per_box

/-- The cost of 900 chocolate candies is $202.50 -/
theorem cost_of_900_candies : cost_of_candies 900 = 202.5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_900_candies_l2046_204626


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2046_204649

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = polygon_interior_angle_sum octagon_sides / octagon_sides := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_measure_l2046_204649


namespace NUMINAMATH_CALUDE_gcd_280_2155_l2046_204604

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := by
  sorry

end NUMINAMATH_CALUDE_gcd_280_2155_l2046_204604


namespace NUMINAMATH_CALUDE_min_employees_needed_l2046_204617

/-- The minimum number of employees needed for pollution monitoring -/
theorem min_employees_needed (water air soil water_air air_soil water_soil all_three : ℕ)
  (h1 : water = 120)
  (h2 : air = 150)
  (h3 : soil = 100)
  (h4 : water_air = 50)
  (h5 : air_soil = 30)
  (h6 : water_soil = 20)
  (h7 : all_three = 10) :
  water + air + soil - water_air - air_soil - water_soil + all_three = 280 := by
  sorry

end NUMINAMATH_CALUDE_min_employees_needed_l2046_204617


namespace NUMINAMATH_CALUDE_choir_composition_theorem_l2046_204680

/-- Represents the choir composition and ratio changes -/
structure ChoirComposition where
  b : ℝ  -- Initial number of blonde girls
  x : ℝ  -- Number of blonde girls added

/-- Theorem about the choir composition changes -/
theorem choir_composition_theorem (choir : ChoirComposition) :
  -- Initial ratio of blonde to black-haired girls is 3:5
  (choir.b) / ((5/3) * choir.b) = 3/5 →
  -- After adding x blonde girls, the ratio becomes 3:2
  (choir.b + choir.x) / ((5/3) * choir.b) = 3/2 →
  -- The final number of black-haired girls is (5/3)b
  (5/3) * choir.b = (5/3) * choir.b ∧
  -- The relationship between x and b is x = (3/2)b
  choir.x = (3/2) * choir.b :=
by sorry

end NUMINAMATH_CALUDE_choir_composition_theorem_l2046_204680


namespace NUMINAMATH_CALUDE_monochromatic_four_clique_exists_l2046_204603

/-- A two-color edge coloring of a complete graph. -/
def TwoColorEdgeColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- The existence of a monochromatic 4-clique in a two-color edge coloring of K_18. -/
theorem monochromatic_four_clique_exists :
  ∀ (coloring : TwoColorEdgeColoring 18),
  ∃ (vertices : Fin 4 → Fin 18),
    (∀ (i j : Fin 4), i ≠ j →
      coloring (vertices i) (vertices j) = coloring (vertices 0) (vertices 1)) :=
by sorry

end NUMINAMATH_CALUDE_monochromatic_four_clique_exists_l2046_204603


namespace NUMINAMATH_CALUDE_complex_equation_proof_l2046_204632

theorem complex_equation_proof (z : ℂ) (h : z = -1/2 + (Real.sqrt 3 / 2) * Complex.I) : 
  z^2 + z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_equation_proof_l2046_204632


namespace NUMINAMATH_CALUDE_nathan_air_hockey_games_l2046_204631

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost of each game in tokens -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := 18

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

theorem nathan_air_hockey_games :
  air_hockey_games = (total_tokens - basketball_games * tokens_per_game) / tokens_per_game :=
by sorry

end NUMINAMATH_CALUDE_nathan_air_hockey_games_l2046_204631


namespace NUMINAMATH_CALUDE_divisor_problem_l2046_204699

theorem divisor_problem (d : ℕ) : d > 0 ∧ 109 = 9 * d + 1 → d = 12 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2046_204699


namespace NUMINAMATH_CALUDE_knights_selection_ways_l2046_204627

/-- Represents the number of knights at the round table -/
def total_knights : ℕ := 12

/-- Represents the number of knights to be chosen -/
def knights_to_choose : ℕ := 5

/-- Represents the number of ways to choose knights in a linear arrangement -/
def linear_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose + 1) knights_to_choose

/-- Represents the number of invalid arrangements (where first and last knights are adjacent) -/
def invalid_arrangements : ℕ := Nat.choose (total_knights - knights_to_choose - 1) (knights_to_choose - 2)

/-- Theorem stating the number of ways to choose knights under the given conditions -/
theorem knights_selection_ways : 
  linear_arrangements - invalid_arrangements = 36 := by sorry

end NUMINAMATH_CALUDE_knights_selection_ways_l2046_204627


namespace NUMINAMATH_CALUDE_ellipse_equation_l2046_204621

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0,
    focal length = 4, and passing through point P(√2, √3),
    prove that the equation of the ellipse is x²/8 + y²/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) →
  (2 / a^2 + 3 / b^2 = 1) →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2046_204621


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2046_204636

theorem min_coefficient_value (a b c d : ℤ) :
  (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40) →
  (∃ (min_box : ℤ), 
    (∃ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 ∧ box ≥ min_box) ∧
    (∀ (box : ℤ), (a * X + b) * (c * X + d) = 40 * X^2 + box * X + 40 → box ≥ min_box) ∧
    min_box = 89) :=
by sorry


end NUMINAMATH_CALUDE_min_coefficient_value_l2046_204636


namespace NUMINAMATH_CALUDE_triangle_external_angle_l2046_204686

theorem triangle_external_angle (a b c : ℝ) (h1 : a = 50) (h2 : b = 60) 
  (h3 : a + b + c = 180) (h4 : c + x = 180) : x = 70 := by
  sorry

end NUMINAMATH_CALUDE_triangle_external_angle_l2046_204686


namespace NUMINAMATH_CALUDE_total_fat_is_3600_l2046_204616

/-- Represents the fat content of different fish types and the number of fish served -/
structure FishData where
  herring_fat : ℕ
  eel_fat : ℕ
  pike_fat_extra : ℕ
  fish_count : ℕ

/-- Calculates the total fat content from all fish served -/
def total_fat (data : FishData) : ℕ :=
  data.fish_count * data.herring_fat +
  data.fish_count * data.eel_fat +
  data.fish_count * (data.eel_fat + data.pike_fat_extra)

/-- Theorem stating that the total fat content is 3600 oz given the specific fish data -/
theorem total_fat_is_3600 (data : FishData)
  (h1 : data.herring_fat = 40)
  (h2 : data.eel_fat = 20)
  (h3 : data.pike_fat_extra = 10)
  (h4 : data.fish_count = 40) :
  total_fat data = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_fat_is_3600_l2046_204616


namespace NUMINAMATH_CALUDE_negation_of_forall_proposition_l2046_204655

theorem negation_of_forall_proposition :
  (¬ ∀ x : ℝ, x > 2 → x^2 + 2 > 6) ↔ (∃ x : ℝ, x > 2 ∧ x^2 + 2 ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_proposition_l2046_204655


namespace NUMINAMATH_CALUDE_hedgehog_strawberries_l2046_204691

/-- The number of strawberries in each basket, given the conditions of the hedgehog problem -/
theorem hedgehog_strawberries (num_hedgehogs : ℕ) (num_baskets : ℕ) 
  (strawberries_per_hedgehog : ℕ) (remaining_fraction : ℚ) :
  num_hedgehogs = 2 →
  num_baskets = 3 →
  strawberries_per_hedgehog = 1050 →
  remaining_fraction = 2/9 →
  ∃ (total_strawberries : ℕ),
    total_strawberries = num_hedgehogs * strawberries_per_hedgehog / (1 - remaining_fraction) ∧
    total_strawberries / num_baskets = 900 :=
by sorry

end NUMINAMATH_CALUDE_hedgehog_strawberries_l2046_204691


namespace NUMINAMATH_CALUDE_consecutive_integers_product_990_l2046_204652

theorem consecutive_integers_product_990 (a b c : ℤ) : 
  b = a + 1 ∧ c = b + 1 ∧ a * b * c = 990 → a + b + c = 30 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_990_l2046_204652


namespace NUMINAMATH_CALUDE_range_of_x_l2046_204620

theorem range_of_x (x : ℝ) 
  (hP : x^2 - 2*x - 3 ≥ 0)
  (hQ : |1 - x/2| ≥ 1) :
  x ≥ 4 ∨ x ≤ -1 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l2046_204620


namespace NUMINAMATH_CALUDE_parallel_vectors_l2046_204635

/-- Given two 2D vectors a and b, find the value of k that makes (k*a + b) parallel to (a - 3*b) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  ∃ k : ℝ, k * a.1 + b.1 = (a.1 - 3 * b.1) * ((k * a.2 + b.2) / (a.2 - 3 * b.2)) ∧ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2046_204635


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l2046_204634

/-- Two variables are inversely proportional if their product is constant -/
def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  inversely_proportional x y →
  x + y = 40 →
  x - y = 8 →
  x = 7 →
  y = 54 + 6/7 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l2046_204634


namespace NUMINAMATH_CALUDE_bales_stored_l2046_204668

theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 22)
  (h2 : final_bales = 89) :
  final_bales - initial_bales = 67 := by
  sorry

end NUMINAMATH_CALUDE_bales_stored_l2046_204668


namespace NUMINAMATH_CALUDE_mean_d_formula_l2046_204698

/-- The set of all positive integers with n 1s, n 2s, n 3s, ..., n ms -/
def S (m n : ℕ) : Set ℕ := sorry

/-- The sum of absolute differences between all pairs of adjacent digits in N -/
def d (N : ℕ) : ℕ := sorry

/-- The mean value of d(N) for N in S(m, n) -/
def mean_d (m n : ℕ) : ℚ := sorry

theorem mean_d_formula {m n : ℕ} (hm : 0 < m ∧ m < 10) (hn : 0 < n) :
  mean_d m n = n * (m^2 - 1) / 3 := by sorry

end NUMINAMATH_CALUDE_mean_d_formula_l2046_204698


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_64_l2046_204694

theorem product_of_fractions_equals_64 :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_64_l2046_204694


namespace NUMINAMATH_CALUDE_vector_difference_l2046_204607

/-- Given two vectors AB and AC in 2D space, prove that BC is their difference --/
theorem vector_difference (AB AC : Fin 2 → ℝ) (h1 : AB = ![2, 3]) (h2 : AC = ![4, 7]) :
  AC - AB = ![2, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_l2046_204607


namespace NUMINAMATH_CALUDE_school_boys_count_l2046_204638

theorem school_boys_count (girls : ℕ) (difference : ℕ) (boys : ℕ) : 
  girls = 635 → difference = 510 → boys = girls + difference → boys = 1145 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l2046_204638


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2046_204689

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point with respect to the origin -/
def symmetricPoint (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The symmetric point of P(3, 2) with respect to the origin is (-3, -2) -/
theorem symmetric_point_theorem :
  let P : Point := { x := 3, y := 2 }
  let P' : Point := symmetricPoint P
  P'.x = -3 ∧ P'.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l2046_204689


namespace NUMINAMATH_CALUDE_hezekiah_age_l2046_204651

/-- Given that Ryanne is 7 years older than Hezekiah and their combined age is 15, 
    prove that Hezekiah is 4 years old. -/
theorem hezekiah_age (hezekiah_age ryanne_age : ℕ) 
  (h1 : ryanne_age = hezekiah_age + 7)
  (h2 : hezekiah_age + ryanne_age = 15) : 
  hezekiah_age = 4 := by
  sorry

end NUMINAMATH_CALUDE_hezekiah_age_l2046_204651


namespace NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2046_204661

/-- Represents a vertex of a cube -/
inductive Vertex
| A | B | C | D | E | F | G | H

/-- Checks if two vertices are connected by an edge -/
def isConnected (v1 v2 : Vertex) : Prop := sorry

/-- Represents an assignment of natural numbers to the vertices of a cube -/
def CubeAssignment := Vertex → Nat

/-- Checks if the assignment satisfies the divisibility condition for connected vertices -/
def satisfiesConnectedDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, isConnected v1 v2 → 
    (assignment v1 ∣ assignment v2) ∨ (assignment v2 ∣ assignment v1)

/-- Checks if the assignment satisfies the non-divisibility condition for non-connected vertices -/
def satisfiesNonConnectedNonDivisibility (assignment : CubeAssignment) : Prop :=
  ∀ v1 v2, ¬isConnected v1 v2 → 
    ¬(assignment v1 ∣ assignment v2) ∧ ¬(assignment v2 ∣ assignment v1)

/-- The main theorem stating that a valid assignment exists -/
theorem valid_cube_assignment_exists : 
  ∃ (assignment : CubeAssignment), 
    satisfiesConnectedDivisibility assignment ∧ 
    satisfiesNonConnectedNonDivisibility assignment := by
  sorry

end NUMINAMATH_CALUDE_valid_cube_assignment_exists_l2046_204661


namespace NUMINAMATH_CALUDE_contrapositive_even_sum_l2046_204610

theorem contrapositive_even_sum (x y : ℤ) :
  (¬(Even (x + y)) → ¬(Even x ∧ Even y)) ↔
  (∀ x y : ℤ, Even x → Even y → Even (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_even_sum_l2046_204610


namespace NUMINAMATH_CALUDE_pineapple_cost_proof_l2046_204628

/-- Given the cost of pineapples and shipping, prove the total cost per pineapple -/
theorem pineapple_cost_proof (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) 
  (h1 : pineapple_cost = 5/4)  -- $1.25 represented as a rational number
  (h2 : num_pineapples = 12)
  (h3 : shipping_cost = 21) :
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples = 3 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_cost_proof_l2046_204628


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2046_204678

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 5/12) (h2 : x - y = 1/36) : x^2 - y^2 = 5/432 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l2046_204678


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2046_204670

/-- Given a quadratic equation with complex coefficients that has real roots, 
    prove that the coefficient 'a' must be equal to -1. -/
theorem quadratic_equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a * (1 + Complex.I)) * x^2 + (1 + a^2 * Complex.I) * x + (a^2 + Complex.I) = 0) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2046_204670


namespace NUMINAMATH_CALUDE_linear_equation_transformation_l2046_204653

theorem linear_equation_transformation (x y : ℝ) :
  (3 * x + 4 * y = 5) ↔ (x = (5 - 4 * y) / 3) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_transformation_l2046_204653


namespace NUMINAMATH_CALUDE_total_spent_correct_l2046_204665

def calculate_total_spent (sandwich_price : Float) (sandwich_discount : Float)
                          (salad_price : Float) (salad_tax : Float)
                          (soda_price : Float) (soda_tax : Float)
                          (tip_percentage : Float) : Float :=
  let discounted_sandwich := sandwich_price * (1 - sandwich_discount)
  let taxed_salad := salad_price * (1 + salad_tax)
  let taxed_soda := soda_price * (1 + soda_tax)
  let subtotal := discounted_sandwich + taxed_salad + taxed_soda
  let total_with_tip := subtotal * (1 + tip_percentage)
  (total_with_tip * 100).round / 100

theorem total_spent_correct :
  calculate_total_spent 10.50 0.15 5.25 0.07 1.75 0.05 0.20 = 19.66 := by
  sorry


end NUMINAMATH_CALUDE_total_spent_correct_l2046_204665


namespace NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l2046_204633

theorem cube_squared_equals_sixth_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_cube_squared_equals_sixth_power_l2046_204633


namespace NUMINAMATH_CALUDE_eight_routes_A_to_B_l2046_204643

/-- The number of different routes from A to B, given that all routes must pass through C -/
def routes_A_to_B (roads_A_to_C roads_C_to_B : ℕ) : ℕ :=
  roads_A_to_C * roads_C_to_B

/-- Theorem stating that there are 8 different routes from A to B -/
theorem eight_routes_A_to_B :
  routes_A_to_B 4 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_routes_A_to_B_l2046_204643


namespace NUMINAMATH_CALUDE_down_payment_correct_l2046_204600

/-- Represents the down payment problem for a car purchase. -/
structure DownPayment where
  total : ℕ
  contributionA : ℕ
  contributionB : ℕ
  contributionC : ℕ
  contributionD : ℕ

/-- Theorem stating that the given contributions satisfy the problem conditions. -/
theorem down_payment_correct (dp : DownPayment) : 
  dp.total = 3500 ∧
  dp.contributionA = 1225 ∧
  dp.contributionB = 875 ∧
  dp.contributionC = 700 ∧
  dp.contributionD = 700 ∧
  dp.contributionA + dp.contributionB + dp.contributionC + dp.contributionD = dp.total ∧
  dp.contributionA = (35 * dp.total) / 100 ∧
  dp.contributionB = (25 * dp.total) / 100 ∧
  dp.contributionC = (20 * dp.total) / 100 ∧
  dp.contributionD = dp.total - (dp.contributionA + dp.contributionB + dp.contributionC) :=
by sorry


end NUMINAMATH_CALUDE_down_payment_correct_l2046_204600


namespace NUMINAMATH_CALUDE_ibrahim_lacking_money_l2046_204660

/-- The amount of money Ibrahim lacks to buy all items -/
def money_lacking (mp3_cost cd_cost headphones_cost case_cost savings father_contribution : ℕ) : ℕ :=
  (mp3_cost + cd_cost + headphones_cost + case_cost) - (savings + father_contribution)

/-- Theorem stating that Ibrahim lacks 165 euros -/
theorem ibrahim_lacking_money : 
  money_lacking 135 25 50 30 55 20 = 165 := by
  sorry

end NUMINAMATH_CALUDE_ibrahim_lacking_money_l2046_204660


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l2046_204666

def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75
def total_money : ℕ := 279

theorem chris_money_before_birthday :
  total_money - (grandmother_gift + aunt_uncle_gift + parents_gift) = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l2046_204666


namespace NUMINAMATH_CALUDE_number_equality_l2046_204629

theorem number_equality (x : ℝ) : (0.4 * x = 0.3 * 50) → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2046_204629


namespace NUMINAMATH_CALUDE_three_chords_when_sixty_degrees_l2046_204641

/-- Represents a configuration of concentric circles with tangent chords -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection on the larger circle -/
  angle : ℝ
  /-- The number of chords needed to form a closed polygon -/
  num_chords : ℕ

/-- Theorem stating that when the angle between chords is 60°, exactly 3 chords are needed -/
theorem three_chords_when_sixty_degrees (config : ConcentricCirclesWithChords) :
  config.angle = 60 → config.num_chords = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_chords_when_sixty_degrees_l2046_204641


namespace NUMINAMATH_CALUDE_infinite_primes_l2046_204663

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l2046_204663


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2046_204606

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (a b c : ℝ × ℝ × ℝ) : Prop := sorry

theorem collinear_points_sum (p q : ℝ) :
  collinear (2, p, q) (p, 3, q) (p, q, 4) → p + q = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2046_204606


namespace NUMINAMATH_CALUDE_jennas_profit_calculation_l2046_204669

/-- Calculates Jenna's total profit after taxes for her wholesale business --/
def jennas_profit (supplier_a_price supplier_b_price resell_price rent tax_rate worker_salary shipping_fee supplier_a_qty supplier_b_qty : ℚ) : ℚ :=
  let total_widgets := supplier_a_qty + supplier_b_qty
  let purchase_cost := supplier_a_price * supplier_a_qty + supplier_b_price * supplier_b_qty
  let shipping_cost := shipping_fee * total_widgets
  let worker_cost := 4 * worker_salary
  let total_expenses := purchase_cost + shipping_cost + rent + worker_cost
  let revenue := resell_price * total_widgets
  let profit_before_tax := revenue - total_expenses
  let tax := tax_rate * profit_before_tax
  profit_before_tax - tax

theorem jennas_profit_calculation :
  jennas_profit 3.5 4 8 10000 0.25 2500 0.25 3000 2000 = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_jennas_profit_calculation_l2046_204669


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l2046_204611

def camp_cedar (num_boys : ℕ) (num_girls : ℕ) (boy_ratio : ℕ) (girl_ratio : ℕ) : ℕ :=
  (num_boys + boy_ratio - 1) / boy_ratio + (num_girls + girl_ratio - 1) / girl_ratio

theorem camp_cedar_counselors :
  let num_boys : ℕ := 80
  let num_girls : ℕ := 6 * num_boys - 40
  let boy_ratio : ℕ := 5
  let girl_ratio : ℕ := 12
  camp_cedar num_boys num_girls boy_ratio girl_ratio = 53 := by
  sorry

#eval camp_cedar 80 (6 * 80 - 40) 5 12

end NUMINAMATH_CALUDE_camp_cedar_counselors_l2046_204611


namespace NUMINAMATH_CALUDE_existence_of_monotonic_tail_l2046_204648

def IsMonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ j ∈ Finset.range (m - 1), a (i + j) < a (i + j + 1)) ∨
  (∀ j ∈ Finset.range (m - 1), a (i + j) > a (i + j + 1))

theorem existence_of_monotonic_tail
  (a : ℕ → ℝ)
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (monotonic_segment : ∀ k, ∃ i m, k ∈ Finset.range m ∧ IsMonotonicSegment a i (k + 1)) :
  ∃ N, (∀ i j, N ≤ i → i < j → a i < a j) ∨ (∀ i j, N ≤ i → i < j → a i > a j) :=
sorry

end NUMINAMATH_CALUDE_existence_of_monotonic_tail_l2046_204648


namespace NUMINAMATH_CALUDE_april_greatest_drop_l2046_204640

/-- Represents the months from January to June --/
inductive Month
| January
| February
| March
| April
| May
| June

/-- Returns the price of the smartphone at the end of the given month --/
def price (m : Month) : Int :=
  match m with
  | Month.January => 350
  | Month.February => 330
  | Month.March => 370
  | Month.April => 340
  | Month.May => 320
  | Month.June => 300

/-- Calculates the price drop from one month to the next --/
def priceDrop (m : Month) : Int :=
  match m with
  | Month.January => price Month.January - price Month.February
  | Month.February => price Month.February - price Month.March
  | Month.March => price Month.March - price Month.April
  | Month.April => price Month.April - price Month.May
  | Month.May => price Month.May - price Month.June
  | Month.June => 0  -- No next month defined

/-- Theorem stating that April had the greatest monthly drop in price --/
theorem april_greatest_drop :
  ∀ m : Month, m ≠ Month.April → priceDrop Month.April ≥ priceDrop m :=
by sorry

end NUMINAMATH_CALUDE_april_greatest_drop_l2046_204640


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l2046_204654

theorem equation_has_real_roots (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l2046_204654


namespace NUMINAMATH_CALUDE_eighth_power_sum_l2046_204619

theorem eighth_power_sum (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) : 
  a^8 + b^8 = 47 := by
  sorry

end NUMINAMATH_CALUDE_eighth_power_sum_l2046_204619


namespace NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l2046_204671

theorem x_plus_y_equals_negative_one (x y : ℝ) : 
  (x - 1)^2 + |y + 2| = 0 → x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_negative_one_l2046_204671


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l2046_204696

-- Define the number of rings and fingers
def total_rings : ℕ := 9
def rings_to_arrange : ℕ := 5
def fingers : ℕ := 5

-- Define the function to calculate the number of arrangements
def ring_arrangements (total : ℕ) (arrange : ℕ) (fingers : ℕ) : ℕ :=
  (Nat.choose total arrange) * (Nat.factorial arrange) * (Nat.choose (arrange + fingers - 1) (fingers - 1))

-- Theorem statement
theorem ring_arrangement_count :
  ring_arrangements total_rings rings_to_arrange fingers = 1900800 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l2046_204696


namespace NUMINAMATH_CALUDE_cube_construction_problem_l2046_204674

theorem cube_construction_problem :
  ∃! (a b c : ℕ+), a^3 + b^3 + c^3 + 648 = (a + b + c)^3 :=
sorry

end NUMINAMATH_CALUDE_cube_construction_problem_l2046_204674


namespace NUMINAMATH_CALUDE_divisibility_condition_implies_prime_relation_l2046_204662

theorem divisibility_condition_implies_prime_relation (m n : ℕ) : 
  m ≥ 2 → n ≥ 2 → 
  (∀ a : ℕ, a ∈ Finset.range n → (a^n - 1) % m = 0) →
  Nat.Prime m ∧ n = m - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_implies_prime_relation_l2046_204662


namespace NUMINAMATH_CALUDE_cookie_process_time_l2046_204681

/-- Represents the cookie-making process with given times for each step -/
structure CookieProcess where
  total_time : ℕ
  baking_time : ℕ
  white_icing_time : ℕ
  chocolate_icing_time : ℕ

/-- Calculates the time to make dough and cool cookies -/
def dough_and_cooling_time (process : CookieProcess) : ℕ :=
  process.total_time - (process.baking_time + process.white_icing_time + process.chocolate_icing_time)

/-- Theorem stating that the time to make dough and cool cookies is 45 minutes -/
theorem cookie_process_time (process : CookieProcess) 
  (h1 : process.total_time = 120)
  (h2 : process.baking_time = 15)
  (h3 : process.white_icing_time = 30)
  (h4 : process.chocolate_icing_time = 30) : 
  dough_and_cooling_time process = 45 := by
  sorry

end NUMINAMATH_CALUDE_cookie_process_time_l2046_204681


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l2046_204658

theorem chinese_remainder_theorem_application (n : ℤ) : 
  n % 158 = 50 → n % 176 = 66 → n % 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_application_l2046_204658


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2046_204657

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "lies within" relation for a line and a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define non-coincidence for lines
variable (non_coincident : Line → Line → Line → Prop)

theorem perpendicular_condition 
  (l m n : Line) (α : Plane)
  (h_non_coincident : non_coincident l m n)
  (h_m_in_α : line_in_plane m α)
  (h_n_in_α : line_in_plane n α) :
  (perp_line_plane l α → perp_line_line l m ∧ perp_line_line l n) ∧
  ¬(perp_line_line l m ∧ perp_line_line l n → perp_line_plane l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2046_204657


namespace NUMINAMATH_CALUDE_product_72_difference_sum_l2046_204682

theorem product_72_difference_sum (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A * B = 72 →
  C * D = 72 →
  A - B = C + D + 2 →
  A = 6 := by
sorry

end NUMINAMATH_CALUDE_product_72_difference_sum_l2046_204682


namespace NUMINAMATH_CALUDE_tens_digit_3_100_is_zero_l2046_204622

/-- The tens digit of 3^100 in decimal notation -/
def tens_digit_3_100 : ℕ :=
  (3^100 / 10) % 10

/-- Theorem stating that the tens digit of 3^100 is 0 -/
theorem tens_digit_3_100_is_zero : tens_digit_3_100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_3_100_is_zero_l2046_204622


namespace NUMINAMATH_CALUDE_baking_time_undetermined_l2046_204612

/-- Represents the cookie-making process with given information -/
structure CookieBaking where
  total_cookies : ℕ
  mixing_time : ℕ
  eaten_cookies : ℕ
  remaining_cookies : ℕ

/-- States that the baking time cannot be determined from the given information -/
theorem baking_time_undetermined (cb : CookieBaking) 
  (h1 : cb.total_cookies = 32)
  (h2 : cb.mixing_time = 24)
  (h3 : cb.eaten_cookies = 9)
  (h4 : cb.remaining_cookies = 23)
  (h5 : cb.total_cookies = cb.eaten_cookies + cb.remaining_cookies) :
  ¬ ∃ (baking_time : ℕ), baking_time = cb.mixing_time ∨ baking_time ≠ cb.mixing_time :=
by sorry


end NUMINAMATH_CALUDE_baking_time_undetermined_l2046_204612


namespace NUMINAMATH_CALUDE_positive_x_solution_l2046_204676

/-- Given a system of equations, prove that the positive solution for x is 3 -/
theorem positive_x_solution (x y z : ℝ) 
  (eq1 : x * y = 6 - 2*x - 3*y)
  (eq2 : y * z = 6 - 4*y - 2*z)
  (eq3 : x * z = 30 - 4*x - 3*z)
  (x_pos : x > 0) :
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_solution_l2046_204676


namespace NUMINAMATH_CALUDE_f_properties_l2046_204683

def f (x : ℝ) : ℝ := |x| + 2

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2046_204683


namespace NUMINAMATH_CALUDE_entrance_charge_is_twelve_l2046_204615

/-- The entrance charge for the strawberry fields -/
def entrance_charge (standard_price : ℕ) (paid_amount : ℕ) (picked_amount : ℕ) : ℕ :=
  standard_price * picked_amount - paid_amount

/-- Proof that the entrance charge is $12 -/
theorem entrance_charge_is_twelve :
  entrance_charge 20 128 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_entrance_charge_is_twelve_l2046_204615


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2046_204664

/-- The slope of a line parallel to 3x + 6y = 15 is -1/2 -/
theorem parallel_line_slope :
  ∀ (m : ℚ), (∃ (b : ℚ), ∀ (x y : ℚ), y = m * x + b ↔ 3 * x + 6 * y = 15) →
  m = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2046_204664


namespace NUMINAMATH_CALUDE_abc_sum_mod_8_l2046_204609

theorem abc_sum_mod_8 (a b c : ℕ) : 
  0 < a ∧ a < 8 ∧ 
  0 < b ∧ b < 8 ∧ 
  0 < c ∧ c < 8 ∧ 
  (a * b * c) % 8 = 1 ∧ 
  (4 * b * c) % 8 = 3 ∧ 
  (5 * b) % 8 = (3 + b) % 8 
  → (a + b + c) % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_mod_8_l2046_204609


namespace NUMINAMATH_CALUDE_catch_up_distance_l2046_204692

/-- Prove that B catches up with A 200 km from the start -/
theorem catch_up_distance (speed_A speed_B : ℝ) (time_diff : ℝ) : 
  speed_A = 10 → 
  speed_B = 20 → 
  time_diff = 10 → 
  speed_B * (time_diff + (speed_B * time_diff - speed_A * time_diff) / (speed_B - speed_A)) = 200 := by
  sorry

#check catch_up_distance

end NUMINAMATH_CALUDE_catch_up_distance_l2046_204692


namespace NUMINAMATH_CALUDE_terrys_spending_ratio_l2046_204618

/-- Terry's spending problem -/
theorem terrys_spending_ratio :
  ∀ (monday tuesday wednesday total : ℚ),
    monday = 6 →
    tuesday = 2 * monday →
    total = monday + tuesday + wednesday →
    total = 54 →
    wednesday = 2 * (monday + tuesday) :=
by sorry

end NUMINAMATH_CALUDE_terrys_spending_ratio_l2046_204618


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2046_204624

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 5 ∧ b = 12 ∧ c = 13

/-- Square inscribed in the right triangle with vertex at right angle -/
def inscribed_square_vertex (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x ≤ t.a ∧ x ≤ t.b ∧ x / t.a = x / t.b

/-- Square inscribed in the right triangle with side on hypotenuse -/
def inscribed_square_hypotenuse (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y ≤ t.c ∧ (t.b / t.a) * y + y + (t.a / t.b) * y = t.c

theorem inscribed_squares_ratio (t1 t2 : RightTriangle) (x y : ℝ)
  (h1 : inscribed_square_vertex t1 x)
  (h2 : inscribed_square_hypotenuse t2 y) :
  x / y = 39 / 51 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2046_204624


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l2046_204608

theorem min_sum_absolute_values : 
  ∃ (x : ℝ), (∀ (y : ℝ), |y - 1| + |y - 2| + |y - 3| ≥ |x - 1| + |x - 2| + |x - 3|) ∧ 
  |x - 1| + |x - 2| + |x - 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l2046_204608


namespace NUMINAMATH_CALUDE_john_james_age_relation_james_brother_age_is_16_l2046_204688

-- Define the ages
def john_age : ℕ := 39
def james_age : ℕ := 12

-- Define the relationship between John and James' ages
theorem john_james_age_relation : john_age - 3 = 2 * (james_age + 6) := by sorry

-- Define James' older brother's age
def james_brother_age : ℕ := james_age + 4

-- Theorem to prove
theorem james_brother_age_is_16 : james_brother_age = 16 := by sorry

end NUMINAMATH_CALUDE_john_james_age_relation_james_brother_age_is_16_l2046_204688


namespace NUMINAMATH_CALUDE_lemonade_glasses_count_l2046_204693

/-- The number of glasses of lemonade that can be served from one pitcher -/
def glasses_per_pitcher : ℕ := 5

/-- The number of pitchers of lemonade prepared -/
def number_of_pitchers : ℕ := 6

/-- The total number of glasses of lemonade that can be served -/
def total_glasses : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_glasses_count : total_glasses = 30 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_glasses_count_l2046_204693


namespace NUMINAMATH_CALUDE_central_cell_value_l2046_204673

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) :
  satisfies_conditions t → t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l2046_204673


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2046_204639

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, x = 1 → f x = (1/2) * x + 2) :
  f 1 + (deriv f) 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2046_204639


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2046_204602

theorem right_triangle_max_ratio (a b c A : ℝ) : 
  a > 0 → b > 0 → c > 0 → A > 0 →
  c^2 = a^2 + b^2 →  -- Pythagorean theorem
  A = (1/2) * a * b →  -- Area formula
  (a + b + A) / c ≤ (5/4) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2046_204602


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l2046_204637

theorem sqrt_expression_equals_sqrt_three : 
  Real.sqrt 48 - 6 * Real.sqrt (1/3) - Real.sqrt 18 / Real.sqrt 6 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_three_l2046_204637


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2046_204697

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 2 = 2 →
  a 6 = 32 →
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2046_204697


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2046_204605

/-- Represents the scores of a basketball team over four quarters -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if the scores form an increasing geometric sequence -/
def is_increasing_geometric (s : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    s.q2 = s.q1 * r ∧
    s.q3 = s.q1 * r^2 ∧
    s.q4 = s.q1 * r^3

/-- Checks if the scores form an increasing arithmetic sequence -/
def is_increasing_arithmetic (s : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧
    s.q2 = s.q1 + d ∧
    s.q3 = s.q1 + 2*d ∧
    s.q4 = s.q1 + 3*d

/-- The main theorem about the basketball game -/
theorem basketball_game_scores 
  (eagles lions : TeamScores)
  (h1 : eagles.q1 = lions.q1)
  (h2 : is_increasing_geometric eagles)
  (h3 : is_increasing_arithmetic lions)
  (h4 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 = 
        lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2)
  (h5 : eagles.q1 + eagles.q2 + eagles.q3 + eagles.q4 ≤ 100)
  (h6 : lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 100) :
  eagles.q1 + eagles.q2 + lions.q1 + lions.q2 = 43 :=
sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l2046_204605


namespace NUMINAMATH_CALUDE_intersection_union_theorem_complement_intersection_theorem_l2046_204684

def A : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 3}
def B (a : ℝ) : Set ℝ := {x | |x| + a < 0}

theorem intersection_union_theorem :
  ∀ a : ℝ, a = -4 →
    (A ∩ B a = {x : ℝ | 1/2 ≤ x ∧ x ≤ 3}) ∧
    (A ∪ B a = {x : ℝ | -4 < x ∧ x < 4}) :=
by sorry

theorem complement_intersection_theorem :
  ∀ a : ℝ, (Aᶜ ∩ B a = B a) ↔ a ≥ -1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_union_theorem_complement_intersection_theorem_l2046_204684


namespace NUMINAMATH_CALUDE_video_votes_l2046_204672

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percent : ℚ) (dislike_percent : ℚ) : 
  score = 120 ∧ 
  like_percent = 58 / 100 ∧ 
  dislike_percent = 30 / 100 ∧ 
  (like_percent - dislike_percent) * total_votes = score →
  total_votes = 429 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l2046_204672


namespace NUMINAMATH_CALUDE_mrs_hilt_reading_l2046_204646

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading : total_chapters = 68 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_reading_l2046_204646


namespace NUMINAMATH_CALUDE_original_class_strength_l2046_204642

theorem original_class_strength (original_average : ℝ) (new_students : ℕ) 
  (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ x : ℕ, x = 12 ∧ 
    (x + new_students : ℝ) * (original_average - average_decrease) = 
    x * original_average + (new_students : ℝ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_original_class_strength_l2046_204642


namespace NUMINAMATH_CALUDE_function_equation_solution_l2046_204677

theorem function_equation_solution (f : ℚ → ℚ) 
  (h0 : f 0 = 0)
  (h1 : ∀ x y : ℚ, f (f x + f y) = x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := by
sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2046_204677


namespace NUMINAMATH_CALUDE_million_place_seven_digits_l2046_204656

/-- A place value in a number system. -/
inductive PlaceValue
  | Units
  | Tens
  | Hundreds
  | Thousands
  | TenThousands
  | HundredThousands
  | Millions

/-- The number of digits in a place value. -/
def PlaceValue.digits : PlaceValue → Nat
  | Units => 1
  | Tens => 2
  | Hundreds => 3
  | Thousands => 4
  | TenThousands => 5
  | HundredThousands => 6
  | Millions => 7

/-- A number with its highest place being the million place has 7 digits. -/
theorem million_place_seven_digits :
  PlaceValue.digits PlaceValue.Millions = 7 := by
  sorry

end NUMINAMATH_CALUDE_million_place_seven_digits_l2046_204656


namespace NUMINAMATH_CALUDE_no_real_solution_l2046_204644

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x + 1/y = 5 ∧ y + 1/x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l2046_204644


namespace NUMINAMATH_CALUDE_log_property_l2046_204659

theorem log_property (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) (h4 : f 9 = 2) : 
  f (a ^ a) = 3 := by
sorry

end NUMINAMATH_CALUDE_log_property_l2046_204659


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2046_204614

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n)
  List.foldl (λ acc d => acc * 10 + d) 0 digits

def has_even_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ Nat.digits 10 n ∧ Even d

theorem sum_with_reverse_has_even_digit (n : ℕ) (h : is_17_digit n) :
  has_even_digit (n + reverse_number n) := by
  sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l2046_204614


namespace NUMINAMATH_CALUDE_equilateral_triangle_formation_l2046_204687

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : ℕ) : Prop := n % 3 = 0

/-- Predicate to check if it's possible to form an equilateral triangle from n sticks -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  divisible_by_three (sum_to_n n) ∧ 
  ∃ (partition : ℕ → ℕ → ℕ), 
    (∀ i j, i < j → j ≤ n → partition i j ≤ sum_to_n n / 3) ∧
    (∀ i, i ≤ n → ∃ j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ j ≤ n ∧ k ≤ n ∧
      partition i j + partition j k + partition k i = sum_to_n n / 3)

theorem equilateral_triangle_formation :
  ¬can_form_equilateral_triangle 100 ∧ can_form_equilateral_triangle 99 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_formation_l2046_204687
