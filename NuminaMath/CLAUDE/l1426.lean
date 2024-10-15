import Mathlib

namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1426_142604

theorem simplify_trig_expression (x : ℝ) (h : 5 * Real.pi / 2 < x ∧ x < 3 * Real.pi) :
  Real.sqrt ((1 - Real.sin (3 * Real.pi / 2 - x)) / 2) = -Real.cos (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1426_142604


namespace NUMINAMATH_CALUDE_road_width_calculation_l1426_142636

/-- Given a rectangular lawn with two roads running through the middle,
    calculate the width of each road based on the cost of traveling. -/
theorem road_width_calculation (lawn_length lawn_width total_cost cost_per_sqm : ℝ)
    (h1 : lawn_length = 80)
    (h2 : lawn_width = 60)
    (h3 : total_cost = 5625)
    (h4 : cost_per_sqm = 3)
    (h5 : total_cost = (lawn_length + lawn_width) * road_width * cost_per_sqm) :
    road_width = total_cost / (cost_per_sqm * (lawn_length + lawn_width)) :=
by sorry

end NUMINAMATH_CALUDE_road_width_calculation_l1426_142636


namespace NUMINAMATH_CALUDE_root_product_sum_l1426_142658

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 4046 * x₁ - 1 = 0) ∧
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 4046 * x₂ - 1 = 0) ∧
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 4046 * x₃ - 1 = 0) →
  x₂ * (x₁ + x₃) = 2 + 1 / 2023 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1426_142658


namespace NUMINAMATH_CALUDE_complex_product_l1426_142656

theorem complex_product (z₁ z₂ : ℂ) :
  Complex.abs z₁ = 1 →
  Complex.abs z₂ = 1 →
  z₁ + z₂ = (-7/5 : ℂ) + (1/5 : ℂ) * Complex.I →
  z₁ * z₂ = (24/25 : ℂ) - (7/25 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_l1426_142656


namespace NUMINAMATH_CALUDE_rain_probability_l1426_142607

/-- The probability of rain on Friday -/
def prob_rain_friday : ℝ := 0.40

/-- The probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.35

/-- The probability of rain on both Friday and Monday -/
def prob_rain_both : ℝ := prob_rain_friday * prob_rain_monday

theorem rain_probability : prob_rain_both = 0.14 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l1426_142607


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_l1426_142689

theorem opposite_numbers_equation (x : ℚ) : 
  x / 5 + (3 - 2 * x) / 2 = 0 → x = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_l1426_142689


namespace NUMINAMATH_CALUDE_gcd_of_factorials_l1426_142666

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem gcd_of_factorials : Nat.gcd (factorial 8) (Nat.gcd (factorial 10) (factorial 11)) = factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_factorials_l1426_142666


namespace NUMINAMATH_CALUDE_tile_draw_probability_l1426_142652

/-- The number of tiles in box A -/
def box_a_size : ℕ := 25

/-- The number of tiles in box B -/
def box_b_size : ℕ := 30

/-- The lowest number on a tile in box A -/
def box_a_min : ℕ := 1

/-- The highest number on a tile in box A -/
def box_a_max : ℕ := 25

/-- The lowest number on a tile in box B -/
def box_b_min : ℕ := 10

/-- The highest number on a tile in box B -/
def box_b_max : ℕ := 39

/-- The threshold for "less than" condition in box A -/
def box_a_threshold : ℕ := 18

/-- The threshold for "greater than" condition in box B -/
def box_b_threshold : ℕ := 30

theorem tile_draw_probability : 
  (((box_a_threshold - box_a_min : ℚ) / box_a_size) * 
   ((box_b_size - (box_b_threshold - box_b_min + 1) / 2 + (box_b_max - box_b_threshold)) / box_b_size)) = 323 / 750 := by
  sorry


end NUMINAMATH_CALUDE_tile_draw_probability_l1426_142652


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1426_142696

theorem fraction_to_decimal : (45 : ℚ) / (5^3) = (360 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1426_142696


namespace NUMINAMATH_CALUDE_smallest_sum_is_26_l1426_142665

def is_valid_pair (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧ (1978^m) % 1000 = (1978^n) % 1000

theorem smallest_sum_is_26 :
  ∃ (m n : ℕ), is_valid_pair m n ∧ m + n = 26 ∧
  ∀ (m' n' : ℕ), is_valid_pair m' n' → m' + n' ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_is_26_l1426_142665


namespace NUMINAMATH_CALUDE_sports_club_size_l1426_142657

/-- The number of members in a sports club -/
def sports_club_members (B T BT N : ℕ) : ℕ :=
  B + T - BT + N

/-- Theorem: The sports club has 30 members -/
theorem sports_club_size :
  ∃ (B T BT N : ℕ),
    B = 17 ∧
    T = 17 ∧
    BT = 6 ∧
    N = 2 ∧
    sports_club_members B T BT N = 30 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_size_l1426_142657


namespace NUMINAMATH_CALUDE_number_problem_l1426_142664

theorem number_problem (x : ℚ) : (54/2 : ℚ) + 3 * x = 75 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1426_142664


namespace NUMINAMATH_CALUDE_weight_calculation_l1426_142660

/-- Given a box containing 16 equal weights with a total weight of 17.88 kg,
    and an empty box weighing 0.6 kg, the weight of 7 such weights is 7.56 kg. -/
theorem weight_calculation (total_weight : ℝ) (box_weight : ℝ) (num_weights : ℕ) (target_weights : ℕ)
    (hw : total_weight = 17.88)
    (hb : box_weight = 0.6)
    (hn : num_weights = 16)
    (ht : target_weights = 7) :
    (total_weight - box_weight) / num_weights * target_weights = 7.56 := by
  sorry

end NUMINAMATH_CALUDE_weight_calculation_l1426_142660


namespace NUMINAMATH_CALUDE_shortest_path_length_l1426_142654

/-- Regular tetrahedron with edge length 2 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 2)

/-- Point on the surface of a regular tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (coordinates : ℝ × ℝ × ℝ)

/-- Midpoint of an edge on a regular tetrahedron -/
def edge_midpoint (t : RegularTetrahedron) : SurfacePoint t :=
  sorry

/-- Distance between two points on the surface of a regular tetrahedron -/
def surface_distance (t : RegularTetrahedron) (p q : SurfacePoint t) : ℝ :=
  sorry

/-- Sequentially next edge midpoint -/
def next_edge_midpoint (t : RegularTetrahedron) (p : SurfacePoint t) : SurfacePoint t :=
  sorry

/-- Theorem: Shortest path between midpoints of sequentially next edges is √6 -/
theorem shortest_path_length (t : RegularTetrahedron) (p : SurfacePoint t) :
  surface_distance t p (next_edge_midpoint t p) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l1426_142654


namespace NUMINAMATH_CALUDE_ten_markers_five_friends_l1426_142685

/-- The number of ways to distribute n identical markers among k friends,
    where each friend must have at least one marker -/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical markers among 5 friends,
    where each friend must have at least one marker, is 126 -/
theorem ten_markers_five_friends :
  distributionWays 10 5 = 126 := by sorry

end NUMINAMATH_CALUDE_ten_markers_five_friends_l1426_142685


namespace NUMINAMATH_CALUDE_always_odd_l1426_142690

theorem always_odd (n : ℤ) : ∃ k : ℤ, (n + 1)^3 - n^3 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1426_142690


namespace NUMINAMATH_CALUDE_single_point_conic_section_l1426_142648

/-- If the graph of 3x^2 + y^2 + 6x - 6y + d = 0 consists of a single point, then d = 12 -/
theorem single_point_conic_section (d : ℝ) :
  (∃! p : ℝ × ℝ, 3 * p.1^2 + p.2^2 + 6 * p.1 - 6 * p.2 + d = 0) →
  d = 12 := by
  sorry

end NUMINAMATH_CALUDE_single_point_conic_section_l1426_142648


namespace NUMINAMATH_CALUDE_sum_of_first_three_coefficients_l1426_142640

theorem sum_of_first_three_coefficients (b : ℝ) : 
  let expansion := (1 + 2/b)^7
  let first_term_coeff := 1
  let second_term_coeff := 7 * 2 / b
  let third_term_coeff := (7 * 6 / 2) * (2 / b)^2
  first_term_coeff + second_term_coeff + third_term_coeff = 211 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_three_coefficients_l1426_142640


namespace NUMINAMATH_CALUDE_problem_solution_l1426_142659

theorem problem_solution : ∃ x : ℚ, (x + x/4 = 80 * 3/4) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1426_142659


namespace NUMINAMATH_CALUDE_smallest_fd_minus_de_is_eight_l1426_142609

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given triangle satisfies the triangle inequality -/
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- The main theorem stating the smallest difference between FD and DE -/
theorem smallest_fd_minus_de_is_eight :
  ∀ t : Triangle,
    t.de + t.ef + t.fd = 3009 →
    t.de < t.ef →
    t.ef ≤ t.fd →
    satisfies_triangle_inequality t →
    (∀ t' : Triangle,
      t'.de + t'.ef + t'.fd = 3009 →
      t'.de < t'.ef →
      t'.ef ≤ t'.fd →
      satisfies_triangle_inequality t' →
      t'.fd - t'.de ≥ t.fd - t.de) →
    t.fd - t.de = 8 := by
  sorry

#check smallest_fd_minus_de_is_eight

end NUMINAMATH_CALUDE_smallest_fd_minus_de_is_eight_l1426_142609


namespace NUMINAMATH_CALUDE_coefficient_of_inverse_x_l1426_142602

theorem coefficient_of_inverse_x (x : ℝ) : 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = c / x + (terms_without_inverse_x : ℝ)) → 
  (∃ c : ℝ, (x / 2 - 2 / x)^5 = -20 / x + (terms_without_inverse_x : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_of_inverse_x_l1426_142602


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l1426_142661

theorem lawn_mowing_problem (mary_rate tom_rate : ℚ) (tom_work_time : ℚ) 
  (h1 : mary_rate = 1 / 4)
  (h2 : tom_rate = 1 / 5)
  (h3 : tom_work_time = 2) :
  1 - tom_rate * tom_work_time = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l1426_142661


namespace NUMINAMATH_CALUDE_intersection_distance_product_l1426_142692

/-- Parabola defined by y² = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Line with equation y = x - 2 -/
def line (x y : ℝ) : Prop := y = x - 2

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (2, 0)

/-- Theorem stating that the product of distances from focus to intersection points is 32 -/
theorem intersection_distance_product : 
  ∃ A B : ℝ × ℝ, 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    (A.1 - focus.1)^2 + (A.2 - focus.2)^2 * 
    (B.1 - focus.1)^2 + (B.2 - focus.2)^2 = 32^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l1426_142692


namespace NUMINAMATH_CALUDE_consecutive_integers_with_square_factors_l1426_142620

theorem consecutive_integers_with_square_factors (n : ℕ) :
  ∃ x : ℤ, ∀ k : ℕ, k ≥ 1 → k ≤ n →
    ∃ m : ℕ, m > 1 ∧ ∃ y : ℤ, x + k = m^2 * y := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_with_square_factors_l1426_142620


namespace NUMINAMATH_CALUDE_percentage_problem_l1426_142694

theorem percentage_problem :
  ∃ x : ℝ, (18 : ℝ) / x = (45 : ℝ) / 100 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1426_142694


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1426_142614

theorem inequality_solution_set (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1426_142614


namespace NUMINAMATH_CALUDE_cylinder_volume_equality_l1426_142619

/-- Proves that for two cylinders with initial radius 5 inches and height 4 inches, 
    if the radius of one and the height of the other are increased by y inches, 
    and their volumes become equal, then y = 5/4 inches. -/
theorem cylinder_volume_equality (y : ℚ) : 
  y ≠ 0 → 
  π * (5 + y)^2 * 4 = π * 5^2 * (4 + y) → 
  y = 5/4 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_equality_l1426_142619


namespace NUMINAMATH_CALUDE_majority_can_play_and_ride_l1426_142643

/-- Represents a person's location and height -/
structure Person where
  location : ℝ × ℝ
  height : ℝ

/-- The population of the country -/
def Population := List Person

/-- Checks if a person is taller than the majority within a given radius -/
def isTallerThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person is shorter than the majority within a given radius -/
def isShorterThanMajority (p : Person) (pop : Population) (radius : ℝ) : Bool :=
  sorry

/-- Checks if a person can play basketball (i.e., can choose a radius to be taller than majority) -/
def canPlayBasketball (p : Person) (pop : Population) : Bool :=
  sorry

/-- Checks if a person is entitled to free transportation (i.e., can choose a radius to be shorter than majority) -/
def hasFreeTrans (p : Person) (pop : Population) : Bool :=
  sorry

/-- Calculates the percentage of people satisfying a given condition -/
def percentageSatisfying (pop : Population) (condition : Person → Population → Bool) : ℝ :=
  sorry

theorem majority_can_play_and_ride (pop : Population) :
  percentageSatisfying pop canPlayBasketball ≥ 90 ∧
  percentageSatisfying pop hasFreeTrans ≥ 90 :=
sorry

end NUMINAMATH_CALUDE_majority_can_play_and_ride_l1426_142643


namespace NUMINAMATH_CALUDE_initial_money_correct_l1426_142628

/-- The amount of money Little John had initially -/
def initial_money : ℚ := 10.50

/-- The amount Little John spent on sweets -/
def sweets_cost : ℚ := 2.25

/-- The amount Little John gave to each friend -/
def money_per_friend : ℚ := 2.20

/-- The number of friends Little John gave money to -/
def number_of_friends : ℕ := 2

/-- The amount of money Little John had left -/
def money_left : ℚ := 3.85

/-- Theorem stating that the initial amount of money is correct given the conditions -/
theorem initial_money_correct : 
  initial_money = sweets_cost + (money_per_friend * number_of_friends) + money_left :=
by sorry

end NUMINAMATH_CALUDE_initial_money_correct_l1426_142628


namespace NUMINAMATH_CALUDE_wintersweet_bouquet_solution_l1426_142684

/-- Represents the number of branches in a bouquet --/
structure BouquetComposition where
  typeA : ℕ
  typeB : ℕ

/-- Represents the total number of branches available --/
structure TotalBranches where
  typeA : ℕ
  typeB : ℕ

/-- Represents the number of bouquets of each type --/
structure BouquetCounts where
  alpha : ℕ
  beta : ℕ

def totalBranches : TotalBranches := { typeA := 142, typeB := 104 }

def alphaBouquet : BouquetComposition := { typeA := 6, typeB := 4 }
def betaBouquet : BouquetComposition := { typeA := 5, typeB := 4 }

/-- The theorem states that given the total branches and bouquet compositions,
    the solution of 12 Alpha bouquets and 14 Beta bouquets is correct --/
theorem wintersweet_bouquet_solution :
  ∃ (solution : BouquetCounts),
    solution.alpha = 12 ∧
    solution.beta = 14 ∧
    solution.alpha * alphaBouquet.typeA + solution.beta * betaBouquet.typeA = totalBranches.typeA ∧
    solution.alpha * alphaBouquet.typeB + solution.beta * betaBouquet.typeB = totalBranches.typeB :=
by sorry

end NUMINAMATH_CALUDE_wintersweet_bouquet_solution_l1426_142684


namespace NUMINAMATH_CALUDE_max_interior_angles_less_than_120_is_5_l1426_142639

/-- A convex polygon with 532 sides -/
structure ConvexPolygon532 where
  sides : ℕ
  convex : Bool
  sidesEq532 : sides = 532

/-- The maximum number of interior angles less than 120° in a ConvexPolygon532 -/
def maxInteriorAnglesLessThan120 (p : ConvexPolygon532) : ℕ :=
  5

/-- Theorem stating that the maximum number of interior angles less than 120° in a ConvexPolygon532 is 5 -/
theorem max_interior_angles_less_than_120_is_5 (p : ConvexPolygon532) :
  maxInteriorAnglesLessThan120 p = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_interior_angles_less_than_120_is_5_l1426_142639


namespace NUMINAMATH_CALUDE_binomial_coefficient_9_5_l1426_142675

theorem binomial_coefficient_9_5 : Nat.choose 9 5 = 126 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_9_5_l1426_142675


namespace NUMINAMATH_CALUDE_inequality_problem_l1426_142671

theorem inequality_problem (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  (b / a < (b + c^2) / (a + c^2)) ∧ (a^2 - 1/a > b^2 - 1/b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1426_142671


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_abs_minus_two_l1426_142621

theorem product_of_solutions_abs_eq_three_abs_minus_two (x : ℝ) :
  (∃ x₁ x₂ : ℝ, (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) →
  x₁ * x₂ = -9) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_eq_three_abs_minus_two_l1426_142621


namespace NUMINAMATH_CALUDE_system_solution_l1426_142612

theorem system_solution (x y k : ℝ) : 
  x + y - 5 * k = 0 → 
  x - y - 9 * k = 0 → 
  2 * x + 3 * y = 6 → 
  4 * k - 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1426_142612


namespace NUMINAMATH_CALUDE_edge_probability_in_cube_l1426_142653

/-- A regular cube -/
structure RegularCube where
  vertices : Nat
  edges_per_vertex : Nat

/-- The probability of selecting two vertices that form an edge in a regular cube -/
def edge_probability (cube : RegularCube) : ℚ :=
  (cube.vertices * cube.edges_per_vertex / 2) / (cube.vertices.choose 2)

/-- Theorem stating the probability of selecting two vertices that form an edge in a regular cube -/
theorem edge_probability_in_cube :
  ∃ (cube : RegularCube), cube.vertices = 8 ∧ cube.edges_per_vertex = 3 ∧ edge_probability cube = 3/7 :=
sorry

end NUMINAMATH_CALUDE_edge_probability_in_cube_l1426_142653


namespace NUMINAMATH_CALUDE_union_P_complement_Q_l1426_142667

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_R_Q : Set ℝ := {x | ¬(x ∈ Q)}

-- State the theorem
theorem union_P_complement_Q : P ∪ C_R_Q = Ioc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_union_P_complement_Q_l1426_142667


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1426_142634

/-- Given vectors a, b, and c in ℝ², where a = (1,2), b = (0,1), and c = (-2,k),
    if (a + 2b) is perpendicular to c, then k = 1/2. -/
theorem perpendicular_vectors_k_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![0, 1]
  let c : Fin 2 → ℝ := ![-2, k]
  (∀ i : Fin 2, (a i + 2 * b i) * c i = 0) →
  k = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l1426_142634


namespace NUMINAMATH_CALUDE_power_multiplication_l1426_142698

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1426_142698


namespace NUMINAMATH_CALUDE_plan_d_cheaper_at_291_l1426_142638

def plan_c_cost (minutes : ℕ) : ℚ := 15 * minutes

def plan_d_cost (minutes : ℕ) : ℚ :=
  if minutes ≤ 100 then
    2500 + 4 * minutes
  else
    2900 + 5 * (minutes - 100)

theorem plan_d_cheaper_at_291 :
  ∀ m : ℕ, m < 291 → plan_c_cost m ≤ plan_d_cost m ∧
  plan_c_cost 291 > plan_d_cost 291 :=
by sorry

end NUMINAMATH_CALUDE_plan_d_cheaper_at_291_l1426_142638


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l1426_142676

/-- For any positive real number a not equal to 1, 
    the function f(x) = a^x + 1 passes through the point (0, 2) -/
theorem exponential_function_fixed_point 
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 1
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l1426_142676


namespace NUMINAMATH_CALUDE_quadricycle_count_l1426_142688

theorem quadricycle_count (total_children : ℕ) (total_wheels : ℕ) 
  (h1 : total_children = 10) 
  (h2 : total_wheels = 30) : ∃ (b t q : ℕ), 
  b + t + q = total_children ∧ 
  2 * b + 3 * t + 4 * q = total_wheels ∧ 
  q = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadricycle_count_l1426_142688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1426_142647

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1426_142647


namespace NUMINAMATH_CALUDE_prove_late_time_l1426_142649

def late_time_problem (charlize_late : ℕ) (classmates_extra : ℕ) (num_classmates : ℕ) : Prop :=
  let classmate_late := charlize_late + classmates_extra
  let total_classmates_late := num_classmates * classmate_late
  let total_late := total_classmates_late + charlize_late
  total_late = 140

theorem prove_late_time : late_time_problem 20 10 4 := by
  sorry

end NUMINAMATH_CALUDE_prove_late_time_l1426_142649


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1426_142673

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1426_142673


namespace NUMINAMATH_CALUDE_product_sum_base_c_l1426_142672

def base_c_to_decimal (n : ℕ) (c : ℕ) : ℕ := c + n

def decimal_to_base_c (n : ℕ) (c : ℕ) : ℕ := n - c

theorem product_sum_base_c (c : ℕ) : 
  (base_c_to_decimal 12 c) * (base_c_to_decimal 14 c) * (base_c_to_decimal 18 c) = 
    5 * c^3 + 3 * c^2 + 2 * c + 0 →
  decimal_to_base_c (base_c_to_decimal 12 c + base_c_to_decimal 14 c + 
                     base_c_to_decimal 18 c + base_c_to_decimal 20 c) c = 40 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_base_c_l1426_142672


namespace NUMINAMATH_CALUDE_range_of_a_l1426_142605

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def condition (a : ℝ) : Prop :=
  (∀ x, ¬(q x) → ¬(p x a)) ∧ 
  (∃ x, ¬(q x) ∧ p x a)

-- State the theorem
theorem range_of_a (a : ℝ) :
  condition a → (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1426_142605


namespace NUMINAMATH_CALUDE_primes_rounding_to_40_l1426_142683

def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

theorem primes_rounding_to_40 :
  ∃! (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p ∧ roundToNearestTen p = 40) ∧ 
    (∀ p, Nat.Prime p → roundToNearestTen p = 40 → p ∈ S) ∧ 
    S.card = 3 :=
by sorry

end NUMINAMATH_CALUDE_primes_rounding_to_40_l1426_142683


namespace NUMINAMATH_CALUDE_unique_tangent_line_l1426_142615

/-- The function whose graph we are considering -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 - 26*x^2

/-- The line we are trying to prove is unique -/
def L (x : ℝ) : ℝ := -60*x - 225

/-- Predicate to check if a point (x, y) is on or above the line L -/
def onOrAboveLine (x y : ℝ) : Prop := y ≥ L x

/-- Predicate to check if a point (x, y) is on the graph of f -/
def onGraph (x y : ℝ) : Prop := y = f x

/-- The main theorem stating the uniqueness of the line L -/
theorem unique_tangent_line :
  (∀ x y, onGraph x y → onOrAboveLine x y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (L x₁) ∧ onGraph x₂ (L x₂)) ∧
  (∀ a b, (∀ x y, onGraph x y → y ≥ a*x + b) ∧
          (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (a*x₁ + b) ∧ onGraph x₂ (a*x₂ + b))
          → a = -60 ∧ b = -225) :=
by sorry

end NUMINAMATH_CALUDE_unique_tangent_line_l1426_142615


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l1426_142622

theorem larger_solution_quadratic (x : ℝ) : 
  x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l1426_142622


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l1426_142641

/-- Calculates the loss percentage given the cost price and selling price -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Theorem stating that the loss percentage for the given cost and selling prices is 17% -/
theorem radio_loss_percentage : 
  loss_percentage 1500 1245 = 17 := by sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l1426_142641


namespace NUMINAMATH_CALUDE_parabola_hyperbola_shared_focus_l1426_142618

/-- The value of p for which the focus of the parabola y^2 = 2px (p > 0) 
    is also a focus of the hyperbola x^2 - y^2 = 8 -/
theorem parabola_hyperbola_shared_focus (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ x^2 - y^2 = 8 ∧ 
    ((x - p)^2 + y^2 = p^2 ∨ (x + p)^2 + y^2 = p^2)) → 
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_shared_focus_l1426_142618


namespace NUMINAMATH_CALUDE_least_sum_exponents_for_896_l1426_142674

theorem least_sum_exponents_for_896 :
  ∃ (a b c : ℕ), 
    (a < b ∧ b < c) ∧ 
    (2^a + 2^b + 2^c = 896) ∧
    (∀ (x y z : ℕ), x < y ∧ y < z ∧ 2^x + 2^y + 2^z = 896 → a + b + c ≤ x + y + z) ∧
    (a + b + c = 24) := by
  sorry

end NUMINAMATH_CALUDE_least_sum_exponents_for_896_l1426_142674


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1426_142699

theorem smallest_solution_abs_equation :
  ∃ x : ℝ, (∀ y : ℝ, |y - 2| = |y - 3| + 1 → x ≤ y) ∧ |x - 2| = |x - 3| + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1426_142699


namespace NUMINAMATH_CALUDE_cloth_sales_calculation_l1426_142691

/-- Calculates the total sales given the commission rate and commission amount -/
def totalSales (commissionRate : ℚ) (commissionAmount : ℚ) : ℚ :=
  commissionAmount / (commissionRate / 100)

/-- Theorem: Given a commission rate of 2.5% and a commission of 18, the total sales is 720 -/
theorem cloth_sales_calculation :
  totalSales (2.5 : ℚ) 18 = 720 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sales_calculation_l1426_142691


namespace NUMINAMATH_CALUDE_prob_red_or_black_prob_not_green_l1426_142642

/-- Represents the colors of balls in the box -/
inductive Color
  | Red
  | Black
  | White
  | Green

/-- The total number of balls in the box -/
def totalBalls : ℕ := 12

/-- The number of balls of each color -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 5
  | Color.Black => 4
  | Color.White => 2
  | Color.Green => 1

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing either a red or black ball is 3/4 -/
theorem prob_red_or_black :
  probability Color.Red + probability Color.Black = 3/4 := by sorry

/-- Theorem: The probability of drawing a ball that is not green is 11/12 -/
theorem prob_not_green :
  1 - probability Color.Green = 11/12 := by sorry

end NUMINAMATH_CALUDE_prob_red_or_black_prob_not_green_l1426_142642


namespace NUMINAMATH_CALUDE_division_sum_equals_two_l1426_142693

theorem division_sum_equals_two : (101 : ℚ) / 101 + (99 : ℚ) / 99 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_equals_two_l1426_142693


namespace NUMINAMATH_CALUDE_monotonic_intervals_max_value_when_a_2_l1426_142697

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 4 * (a + 1) * x^3 + 6 * a * x^2 - 12

-- Theorem for the intervals of monotonic increase
theorem monotonic_intervals (a : ℝ) (h : a > 0) :
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ a → f a x < f a y) ∧
  (∀ x y, 1 ≤ x ∧ x < y → f a x < f a y) ∧
  (a = 1 → ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (a > 1 → (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x < f a y) ∧
           (∀ x y, a ≤ x ∧ x < y → f a x < f a y)) :=
sorry

-- Theorem for the maximum value when a = 2
theorem max_value_when_a_2 :
  ∀ x, f 2 x ≤ f 2 1 ∧ f 2 1 = -9 :=
sorry

end NUMINAMATH_CALUDE_monotonic_intervals_max_value_when_a_2_l1426_142697


namespace NUMINAMATH_CALUDE_parallel_vectors_m_values_l1426_142633

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The theorem statement -/
theorem parallel_vectors_m_values (m : ℝ) :
  parallel (2, m) (m, 2) → m = -2 ∨ m = 2 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_m_values_l1426_142633


namespace NUMINAMATH_CALUDE_tan_sum_three_angles_l1426_142617

theorem tan_sum_three_angles (α β γ : ℝ) : 
  Real.tan (α + β + γ) = (Real.tan α + Real.tan β + Real.tan γ - Real.tan α * Real.tan β * Real.tan γ) / 
                         (1 - Real.tan α * Real.tan β - Real.tan β * Real.tan γ - Real.tan γ * Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_three_angles_l1426_142617


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1426_142613

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (l : ℝ) -- lateral edge length
  (α : ℝ) -- angle between lateral edge and base plane
  (h1 : l > 0) -- lateral edge length is positive
  (h2 : 0 < α ∧ α < π/2) -- angle is between 0 and π/2
  : ∃ (V : ℝ), V = (Real.sqrt 3 * l^3 * Real.cos α^2 * Real.sin α) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l1426_142613


namespace NUMINAMATH_CALUDE_certain_number_exists_l1426_142631

theorem certain_number_exists : ∃ x : ℝ, 0.35 * x - (1/3) * (0.35 * x) = 42 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l1426_142631


namespace NUMINAMATH_CALUDE_trig_identity_l1426_142600

/-- Proves that sin 69° cos 9° - sin 21° cos 81° = √3/2 -/
theorem trig_identity : Real.sin (69 * π / 180) * Real.cos (9 * π / 180) - 
  Real.sin (21 * π / 180) * Real.cos (81 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1426_142600


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1426_142680

theorem arithmetic_sequence_length : 
  ∀ (a d : ℤ) (n : ℕ), 
    a - d * (n - 1) = 39 → 
    a = 147 → 
    d = 3 → 
    n = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1426_142680


namespace NUMINAMATH_CALUDE_power_multiplication_l1426_142632

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1426_142632


namespace NUMINAMATH_CALUDE_students_playing_soccer_l1426_142611

theorem students_playing_soccer 
  (total_students : ℕ) 
  (boy_students : ℕ) 
  (girls_not_playing : ℕ) 
  (soccer_boys_percentage : ℚ) :
  total_students = 420 →
  boy_students = 320 →
  girls_not_playing = 65 →
  soccer_boys_percentage = 86/100 →
  ∃ (students_playing_soccer : ℕ), 
    students_playing_soccer = 250 ∧
    (total_students - boy_students - girls_not_playing : ℚ) = 
      (1 - soccer_boys_percentage) * students_playing_soccer := by
  sorry

end NUMINAMATH_CALUDE_students_playing_soccer_l1426_142611


namespace NUMINAMATH_CALUDE_circle_condition_l1426_142637

/-- 
Given a real number a and the equation ax^2 + ay^2 - 4(a-1)x + 4y = 0,
this theorem states that the equation represents a circle if and only if a ≠ 0.
-/
theorem circle_condition (a : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ 
    ∀ (x y : ℝ), ax^2 + ay^2 - 4*(a-1)*x + 4*y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ 
  a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l1426_142637


namespace NUMINAMATH_CALUDE_probability_of_drawing_k_l1426_142616

/-- The probability of drawing a "K" from a standard deck of 54 playing cards -/
theorem probability_of_drawing_k (total_cards : ℕ) (k_cards : ℕ) : 
  total_cards = 54 → k_cards = 4 → (k_cards : ℚ) / total_cards = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_k_l1426_142616


namespace NUMINAMATH_CALUDE_pentagon_perimeter_is_49_l1426_142630

/-- The perimeter of a pentagon with given side lengths -/
def pentagon_perimeter (x y z : ℝ) : ℝ :=
  3*x + 5*y + 6*z + 4*x + 7*y

/-- Theorem: The perimeter of the specified pentagon is 49 cm -/
theorem pentagon_perimeter_is_49 :
  pentagon_perimeter 1 2 3 = 49 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_is_49_l1426_142630


namespace NUMINAMATH_CALUDE_rhonda_marbles_l1426_142635

theorem rhonda_marbles (total : ℕ) (diff : ℕ) (rhonda : ℕ) : 
  total = 215 → diff = 55 → total = rhonda + (rhonda + diff) → rhonda = 80 := by
  sorry

end NUMINAMATH_CALUDE_rhonda_marbles_l1426_142635


namespace NUMINAMATH_CALUDE_solve_system_for_y_l1426_142629

theorem solve_system_for_y (x y : ℚ) 
  (eq1 : 2 * x - 3 * y = 18) 
  (eq2 : x + 2 * y = 8) : 
  y = -2 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l1426_142629


namespace NUMINAMATH_CALUDE_vector_linear_combination_l1426_142663

/-- Given vectors a, b, and c in ℝ², prove that c is a linear combination of a and b. -/
theorem vector_linear_combination (a b c : ℝ × ℝ) : 
  a = (1, 1) → b = (1, -1) → c = (-1, 2) → c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l1426_142663


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l1426_142624

/-- Given that (0,0), (a,11), and (b,37) form an equilateral triangle, prove that ab = 315 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((a + 11 * Complex.I) * (Complex.exp (Complex.I * Real.pi / 3)) = b + 37 * Complex.I) →
  a * b = 315 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_product_l1426_142624


namespace NUMINAMATH_CALUDE_handshakes_count_l1426_142606

/-- Represents a social event with two groups of people -/
structure SocialEvent where
  total_people : ℕ
  group1_size : ℕ  -- People who know each other
  group2_size : ℕ  -- People who don't know anyone
  h_total : total_people = group1_size + group2_size

/-- Calculates the number of handshakes in a social event -/
def count_handshakes (event : SocialEvent) : ℕ :=
  (event.group2_size * (event.total_people - 1)) / 2

/-- Theorem stating the number of handshakes in the specific social event -/
theorem handshakes_count :
  ∃ (event : SocialEvent),
    event.total_people = 40 ∧
    event.group1_size = 25 ∧
    event.group2_size = 15 ∧
    count_handshakes event = 292 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l1426_142606


namespace NUMINAMATH_CALUDE_nearest_integer_to_sum_l1426_142610

theorem nearest_integer_to_sum (x y : ℝ) 
  (h1 : abs x - y = 5)
  (h2 : abs x * y - x^2 = -12) : 
  round (x + y) = -5 := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_sum_l1426_142610


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1426_142655

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → 
    (¬(24 ∣ m^2) ∨ ¬(900 ∣ m^3) ∨ ¬(1024 ∣ m^4))) ∧
  24 ∣ n^2 ∧ 900 ∣ n^3 ∧ 1024 ∣ n^4 ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1426_142655


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l1426_142625

def small_bottle_capacity : ℚ := 45
def large_bottle_1_capacity : ℚ := 630
def large_bottle_2_capacity : ℚ := 850

theorem minimum_bottles_needed : 
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_1_capacity + large_bottle_2_capacity → m ≥ n ∧
  n = 33 := by
  sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l1426_142625


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1426_142668

/-- 
Given a point P rotated about a center Q:
1. 510 degrees clockwise rotation reaches point R
2. y degrees counterclockwise rotation also reaches point R
3. y < 360

Prove that y = 210
-/
theorem rotation_equivalence (y : ℝ) 
  (h1 : y < 360)
  (h2 : (510 % 360 : ℝ) = (360 - y) % 360) : y = 210 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1426_142668


namespace NUMINAMATH_CALUDE_ternary_to_decimal_l1426_142650

theorem ternary_to_decimal :
  (1 * 3^2 + 2 * 3^1 + 1 * 3^0 : ℕ) = 16 := by sorry

end NUMINAMATH_CALUDE_ternary_to_decimal_l1426_142650


namespace NUMINAMATH_CALUDE_park_length_l1426_142686

/-- A rectangular park with given dimensions and tree density. -/
structure Park where
  width : ℝ
  length : ℝ
  treeCount : ℕ
  treeDensity : ℝ

/-- The park satisfies the given conditions. -/
def validPark (p : Park) : Prop :=
  p.width = 2000 ∧
  p.treeCount = 100000 ∧
  p.treeDensity = 1 / 20

/-- The theorem stating the length of the park given the conditions. -/
theorem park_length (p : Park) (h : validPark p) : p.length = 1000 := by
  sorry

#check park_length

end NUMINAMATH_CALUDE_park_length_l1426_142686


namespace NUMINAMATH_CALUDE_problem_solution_l1426_142662

theorem problem_solution (p q r : ℝ) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_equation : p / (q - r) + q / (r - p) + r / (p - q) = 1) : 
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1426_142662


namespace NUMINAMATH_CALUDE_sqrt_172_01_l1426_142623

theorem sqrt_172_01 (h1 : Real.sqrt 1.7201 = 1.311) (h2 : Real.sqrt 17.201 = 4.147) :
  Real.sqrt 172.01 = 13.11 ∨ Real.sqrt 172.01 = -13.11 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_172_01_l1426_142623


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l1426_142651

theorem consecutive_squares_sum (x : ℕ) : 
  (x - 1)^2 + x^2 + (x + 1)^2 = 8 * ((x - 1) + x + (x + 1)) + 2 →
  ∃ n : ℕ, (n - 1)^2 + n^2 + (n + 1)^2 = 194 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l1426_142651


namespace NUMINAMATH_CALUDE_three_arcs_must_intersect_l1426_142687

/-- Represents a great circle arc on a sphere --/
structure GreatCircleArc where
  length : ℝ
  start_point : Sphere
  end_point : Sphere

/-- Defines a sphere --/
class Sphere where
  center : Point
  radius : ℝ

/-- Checks if two great circle arcs intersect or share an endpoint --/
def arcs_intersect (arc1 arc2 : GreatCircleArc) : Prop :=
  sorry

/-- Theorem: It's impossible to place three 300° great circle arcs on a sphere without intersections --/
theorem three_arcs_must_intersect (s : Sphere) :
  ∀ (arc1 arc2 arc3 : GreatCircleArc),
    arc1.length = 300 ∧ arc2.length = 300 ∧ arc3.length = 300 →
    arcs_intersect arc1 arc2 ∨ arcs_intersect arc2 arc3 ∨ arcs_intersect arc1 arc3 :=
by
  sorry

end NUMINAMATH_CALUDE_three_arcs_must_intersect_l1426_142687


namespace NUMINAMATH_CALUDE_quadratic_roots_l1426_142603

theorem quadratic_roots (d : ℚ) : 
  (∀ x : ℚ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt (2*d))/2 ∨ x = (-7 - Real.sqrt (2*d))/2) → 
  d = 49/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1426_142603


namespace NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l1426_142670

/-- Represents the percentage of green ducks in the larger pond -/
def larger_pond_green_percentage : ℝ := 15

theorem green_ducks_percentage_in_larger_pond :
  let smaller_pond_ducks : ℕ := 20
  let larger_pond_ducks : ℕ := 80
  let smaller_pond_green_percentage : ℝ := 20
  let total_green_percentage : ℝ := 16
  larger_pond_green_percentage = 
    (total_green_percentage * (smaller_pond_ducks + larger_pond_ducks) - 
     smaller_pond_green_percentage * smaller_pond_ducks) / larger_pond_ducks := by
  sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_in_larger_pond_l1426_142670


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l1426_142679

theorem sufficient_conditions_for_x_squared_less_than_one :
  (∀ x : ℝ, (0 < x ∧ x < 1) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 0) → x^2 < 1) ∧
  (∀ x : ℝ, (-1 < x ∧ x < 1) → x^2 < 1) ∧
  (∃ x : ℝ, x < 1 ∧ ¬(x^2 < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_x_squared_less_than_one_l1426_142679


namespace NUMINAMATH_CALUDE_households_with_bike_only_l1426_142695

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90) 
  (h2 : neither = 11) 
  (h3 : both = 20) 
  (h4 : with_car = 44) : 
  total - neither - (with_car - both) - both = 35 := by
sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l1426_142695


namespace NUMINAMATH_CALUDE_taxi_trip_length_l1426_142627

/-- Calculates the trip length in miles given the taxi fare parameters and total charge -/
def trip_length (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let segments := (total_charge - initial_fee) / charge_per_segment
  segments * segment_length

theorem taxi_trip_length :
  let initial_fee : ℚ := 225/100
  let charge_per_segment : ℚ := 35/100
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 54/10
  trip_length initial_fee charge_per_segment segment_length total_charge = 36/10 := by
  sorry

end NUMINAMATH_CALUDE_taxi_trip_length_l1426_142627


namespace NUMINAMATH_CALUDE_sixth_term_is_46_l1426_142669

/-- The sequence of small circles in each figure -/
def circleSequence (n : ℕ) : ℕ := n * (n + 1) + 4

/-- The theorem stating that the 6th term of the sequence is 46 -/
theorem sixth_term_is_46 : circleSequence 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_46_l1426_142669


namespace NUMINAMATH_CALUDE_binary_10011_equals_19_l1426_142677

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 10011₂ -/
def binary_10011 : List Bool := [true, true, false, false, true]

/-- Theorem stating that 10011₂ is equal to 19 in decimal -/
theorem binary_10011_equals_19 : binary_to_decimal binary_10011 = 19 := by
  sorry

end NUMINAMATH_CALUDE_binary_10011_equals_19_l1426_142677


namespace NUMINAMATH_CALUDE_faye_candy_count_l1426_142601

/-- Calculates the remaining candy count after a given number of days -/
def remaining_candy (initial : ℕ) (daily_consumption : ℕ) (daily_addition : ℕ) (days : ℕ) : ℤ :=
  initial + days * daily_addition - days * daily_consumption

/-- Theorem: Faye's remaining candy count after y days -/
theorem faye_candy_count :
  ∀ (x y z : ℕ), remaining_candy 47 x z y = 47 + y * z - y * x :=
by
  sorry

#check faye_candy_count

end NUMINAMATH_CALUDE_faye_candy_count_l1426_142601


namespace NUMINAMATH_CALUDE_bus_motion_time_is_24_minutes_l1426_142645

/-- Represents the bus journey on a highway -/
structure BusJourney where
  distance : ℝ  -- Total distance in km
  num_stops : ℕ -- Number of intermediate stops
  stop_duration : ℝ -- Duration of each stop in minutes
  speed_difference : ℝ -- Difference in km/h between non-stop speed and average speed with stops

/-- Calculates the time the bus is in motion -/
def motion_time (journey : BusJourney) : ℝ :=
  sorry

/-- The main theorem stating that the motion time is 24 minutes for the given conditions -/
theorem bus_motion_time_is_24_minutes (journey : BusJourney) 
  (h1 : journey.distance = 10)
  (h2 : journey.num_stops = 6)
  (h3 : journey.stop_duration = 1)
  (h4 : journey.speed_difference = 5) :
  motion_time journey = 24 :=
sorry

end NUMINAMATH_CALUDE_bus_motion_time_is_24_minutes_l1426_142645


namespace NUMINAMATH_CALUDE_total_marbles_is_72_l1426_142646

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of yellow:blue:green marbles -/
def marbleRatio : MarbleCounts := ⟨2, 3, 4⟩

/-- The actual number of green marbles in the bag -/
def greenMarbleCount : ℕ := 32

/-- Calculate the total number of marbles in the bag -/
def totalMarbles (mc : MarbleCounts) : ℕ :=
  mc.yellow + mc.blue + mc.green

/-- Theorem stating that the total number of marbles is 72 -/
theorem total_marbles_is_72 :
  ∃ (factor : ℕ), 
    factor * marbleRatio.green = greenMarbleCount ∧
    totalMarbles (MarbleCounts.mk 
      (factor * marbleRatio.yellow)
      (factor * marbleRatio.blue)
      greenMarbleCount) = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_72_l1426_142646


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l1426_142681

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l1426_142681


namespace NUMINAMATH_CALUDE_graph_number_example_intersection_condition_l1426_142608

-- Define the "graph number" type
def GraphNumber := ℝ × ℝ × ℝ

-- Define a function to get the graph number of a quadratic function
def getGraphNumber (a b c : ℝ) : GraphNumber :=
  (a, b, c)

-- Define a function to check if a quadratic function intersects x-axis at one point
def intersectsAtOnePoint (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

-- Theorem 1: The graph number of y = (1/3)x^2 - x - 1
theorem graph_number_example : getGraphNumber (1/3) (-1) (-1) = (1/3, -1, -1) := by
  sorry

-- Theorem 2: For [m, m+1, m+1] intersecting x-axis at one point, m = -1 or m = 1/3
theorem intersection_condition (m : ℝ) :
  intersectsAtOnePoint m (m+1) (m+1) → m = -1 ∨ m = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_graph_number_example_intersection_condition_l1426_142608


namespace NUMINAMATH_CALUDE_tan_beta_value_l1426_142682

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = -3/4)
  (h2 : Real.tan (α + β) = 1) : 
  Real.tan β = 7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l1426_142682


namespace NUMINAMATH_CALUDE_toy_store_shelves_l1426_142644

theorem toy_store_shelves (initial_stock : ℕ) (new_shipment : ℕ) (bears_per_shelf : ℕ) : 
  initial_stock = 6 →
  new_shipment = 18 →
  bears_per_shelf = 6 →
  (initial_stock + new_shipment) / bears_per_shelf = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_toy_store_shelves_l1426_142644


namespace NUMINAMATH_CALUDE_book_words_per_page_l1426_142626

theorem book_words_per_page 
  (total_pages : ℕ)
  (words_per_page : ℕ)
  (max_words_per_page : ℕ)
  (total_words_mod : ℕ)
  (h1 : total_pages = 224)
  (h2 : words_per_page ≤ max_words_per_page)
  (h3 : max_words_per_page = 150)
  (h4 : (total_pages * words_per_page) % 253 = total_words_mod)
  (h5 : total_words_mod = 156) :
  words_per_page = 106 := by
sorry

end NUMINAMATH_CALUDE_book_words_per_page_l1426_142626


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l1426_142678

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number -/
def base7Number : List Nat := [6, 4, 2]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l1426_142678
