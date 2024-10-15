import Mathlib

namespace NUMINAMATH_CALUDE_trees_difference_l157_15782

theorem trees_difference (initial_trees : ℕ) (died_trees : ℕ) : 
  initial_trees = 14 → died_trees = 9 → died_trees - (initial_trees - died_trees) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trees_difference_l157_15782


namespace NUMINAMATH_CALUDE_point_B_coordinates_l157_15770

-- Define the point A
def A : ℝ × ℝ := (-3, 2)

-- Define the length of AB
def AB_length : ℝ := 4

-- Define the possible coordinates of point B
def B1 : ℝ × ℝ := (-7, 2)
def B2 : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  (B.2 = A.2) →                        -- AB is parallel to x-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = AB_length^2) →  -- Length of AB is 4
  (B = B1 ∨ B = B2) :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_point_B_coordinates_l157_15770


namespace NUMINAMATH_CALUDE_inequality_proof_l157_15731

theorem inequality_proof :
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + b) ≤ (1 / 4) * (1 / a + 1 / b)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 → 1 / x₁ + 1 / x₂ + 1 / x₃ = 1 →
    (x₁ + x₂ + x₃) / (x₁ * x₃ + x₃ * x₂) + 
    (x₁ + x₂ + x₃) / (x₁ * x₂ + x₃ * x₁) + 
    (x₁ + x₂ + x₃) / (x₂ * x₁ + x₃ * x₂) ≤ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l157_15731


namespace NUMINAMATH_CALUDE_oil_added_to_mixture_l157_15735

/-- Proves that the amount of oil added to mixture A is 2 kilograms -/
theorem oil_added_to_mixture (mixture_a_weight : ℝ) (oil_percentage : ℝ) (material_b_percentage : ℝ)
  (added_mixture_a : ℝ) (final_material_b_percentage : ℝ) :
  mixture_a_weight = 8 →
  oil_percentage = 0.2 →
  material_b_percentage = 0.8 →
  added_mixture_a = 6 →
  final_material_b_percentage = 0.7 →
  ∃ (x : ℝ),
    x = 2 ∧
    (material_b_percentage * mixture_a_weight + material_b_percentage * added_mixture_a) =
      final_material_b_percentage * (mixture_a_weight + x + added_mixture_a) :=
by sorry

end NUMINAMATH_CALUDE_oil_added_to_mixture_l157_15735


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l157_15700

theorem sqrt_sum_fractions : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l157_15700


namespace NUMINAMATH_CALUDE_particle_average_velocity_l157_15733

/-- Given a particle with motion law s = t^2 + 3, its average velocity 
    during the time interval (3, 3+Δx) is equal to 6 + Δx. -/
theorem particle_average_velocity (Δx : ℝ) : 
  let s (t : ℝ) := t^2 + 3
  ((s (3 + Δx) - s 3) / Δx) = 6 + Δx :=
sorry

end NUMINAMATH_CALUDE_particle_average_velocity_l157_15733


namespace NUMINAMATH_CALUDE_gavin_green_shirts_l157_15772

/-- The number of green shirts Gavin has -/
def num_green_shirts (total_shirts blue_shirts : ℕ) : ℕ :=
  total_shirts - blue_shirts

/-- Theorem stating that Gavin has 17 green shirts -/
theorem gavin_green_shirts : 
  num_green_shirts 23 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gavin_green_shirts_l157_15772


namespace NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l157_15777

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ n : ℕ, n > 0 ∧ 
    is_multiple n 45 ∧ 
    is_multiple n 75 ∧ 
    ¬is_multiple n 20 ∧
    ∀ m : ℕ, m > 0 → 
      is_multiple m 45 → 
      is_multiple m 75 → 
      ¬is_multiple m 20 → 
      n ≤ m ∧
  n = 225 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_45_and_75_not_20_l157_15777


namespace NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l157_15748

theorem consecutive_blue_gumballs_probability 
  (pink_prob : ℝ) 
  (h_pink_prob : pink_prob = 1/3) : 
  let blue_prob := 1 - pink_prob
  (blue_prob * blue_prob) = 4/9 := by sorry

end NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l157_15748


namespace NUMINAMATH_CALUDE_square_root_problem_l157_15725

theorem square_root_problem (x : ℝ) : Real.sqrt x - (Real.sqrt 625 / Real.sqrt 25) = 12 → x = 289 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l157_15725


namespace NUMINAMATH_CALUDE_linear_system_solution_l157_15723

theorem linear_system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y + z = 0 →
  3*x + 6*y + 5*z = 0 →
  (k = 90/41 ∧ y*z/x^2 = 41/30) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l157_15723


namespace NUMINAMATH_CALUDE_sqrt_0_1681_l157_15734

theorem sqrt_0_1681 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_0_1681_l157_15734


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l157_15716

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 1, a - 1)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l157_15716


namespace NUMINAMATH_CALUDE_max_of_two_numbers_l157_15707

theorem max_of_two_numbers (a b : ℕ) (ha : a = 2) (hb : b = 3) :
  max a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_of_two_numbers_l157_15707


namespace NUMINAMATH_CALUDE_min_distance_to_A_l157_15726

-- Define the space
variable (X : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] [CompleteSpace X]

-- Define points A, B, and P
variable (A B P : X)

-- Define the conditions
variable (h1 : ‖A - B‖ = 4)
variable (h2 : ‖P - A‖ - ‖P - B‖ = 3)

-- State the theorem
theorem min_distance_to_A :
  ∃ (min_dist : ℝ), min_dist = 7/2 ∧ ∀ P, ‖P - A‖ - ‖P - B‖ = 3 → ‖P - A‖ ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_A_l157_15726


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l157_15789

/-- Given that Danny can reach Steve's house in 35 minutes and it takes Steve twice as long to reach Danny's house,
    prove that Steve will take 17.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 35 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 17.5 := by
sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l157_15789


namespace NUMINAMATH_CALUDE_fair_coin_prob_TTHH_l157_15740

/-- The probability of getting tails on a single flip of a fair coin -/
def prob_tails : ℚ := 1 / 2

/-- The number of times the coin is flipped -/
def num_flips : ℕ := 4

/-- The probability of getting tails on the first two flips and heads on the last two flips -/
def prob_TTHH : ℚ := prob_tails * prob_tails * (1 - prob_tails) * (1 - prob_tails)

theorem fair_coin_prob_TTHH :
  prob_TTHH = 1 / 16 :=
sorry

end NUMINAMATH_CALUDE_fair_coin_prob_TTHH_l157_15740


namespace NUMINAMATH_CALUDE_coplanar_condition_l157_15786

open Vector

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

def isCoplanar (p₁ p₂ p₃ p₄ p₅ : V) : Prop :=
  ∃ (a b c d : ℝ), a • (p₂ - p₁) + b • (p₃ - p₁) + c • (p₄ - p₁) + d • (p₅ - p₁) = 0

theorem coplanar_condition (O E F G H I : V) (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) - 2 • (I - O) = 0) →
  (isCoplanar E F G H I ↔ m = -5) := by
  sorry

end NUMINAMATH_CALUDE_coplanar_condition_l157_15786


namespace NUMINAMATH_CALUDE_divisible_by_six_l157_15762

theorem divisible_by_six (n : ℤ) : ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_six_l157_15762


namespace NUMINAMATH_CALUDE_number_calculation_l157_15732

theorem number_calculation (n : ℝ) : (0.1 * 0.2 * 0.35 * 0.4 * n = 84) → n = 300000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l157_15732


namespace NUMINAMATH_CALUDE_square_root_difference_l157_15774

theorem square_root_difference : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_l157_15774


namespace NUMINAMATH_CALUDE_sequence_properties_l157_15769

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := 2 * n + 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := 2^n

-- Define the sum of the first n terms of a_n + b_n
def S (n : ℕ) : ℝ := n^2 + 2*n + 2^(n+1) - 2

theorem sequence_properties :
  (a 2 = 5) ∧
  (a 1 + a 4 = 12) ∧
  (∀ n, b n > 0) ∧
  (∀ n, b n * b (n+1) = 2^(a n)) ∧
  (∀ n, a n = 2*n + 1) ∧
  (∀ n, b (n+1) / b n = 2) ∧
  (∀ n, S n = n^2 + 2*n + 2^(n+1) - 2) :=
by sorry


end NUMINAMATH_CALUDE_sequence_properties_l157_15769


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l157_15710

theorem arithmetic_evaluation : 12 / 4 - 3 - 6 + 3 * 5 = 9 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l157_15710


namespace NUMINAMATH_CALUDE_unpainted_area_proof_l157_15714

def board_width_1 : ℝ := 4
def board_width_2 : ℝ := 6
def intersection_angle : ℝ := 60

theorem unpainted_area_proof :
  let parallelogram_base := board_width_2 / Real.sin (intersection_angle * Real.pi / 180)
  let parallelogram_height := board_width_1
  parallelogram_base * parallelogram_height = 16 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_proof_l157_15714


namespace NUMINAMATH_CALUDE_five_students_three_companies_l157_15755

/-- The number of ways to assign n students to k companies, where each company must receive at least one student. -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2)

/-- Theorem stating that the number of ways to assign 5 students to 3 companies, where each company must receive at least one student, is 150. -/
theorem five_students_three_companies : assignment_count 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_companies_l157_15755


namespace NUMINAMATH_CALUDE_angle_ABG_measure_l157_15701

/-- A regular octagon is a polygon with 8 sides of equal length and 8 interior angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Angle ABG in a regular octagon ABCDEFGH -/
def angle_ABG (octagon : RegularOctagon) : ℝ := sorry

/-- The measure of angle ABG in a regular octagon is 22.5 degrees -/
theorem angle_ABG_measure (octagon : RegularOctagon) : angle_ABG octagon = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_ABG_measure_l157_15701


namespace NUMINAMATH_CALUDE_semicircle_radius_with_inscribed_circles_l157_15741

/-- The radius of a semicircle that inscribes two externally touching circles -/
theorem semicircle_radius_with_inscribed_circles 
  (r₁ r₂ R : ℝ) 
  (h₁ : r₁ = Real.sqrt 19)
  (h₂ : r₂ = Real.sqrt 76)
  (h_touch : r₁ + r₂ = R - r₁ + R - r₂) 
  (h_inscribed : R^2 = (R - r₁)^2 + r₁^2 ∧ R^2 = (R - r₂)^2 + r₂^2) :
  R = 4 * Real.sqrt 19 := by
sorry

end NUMINAMATH_CALUDE_semicircle_radius_with_inscribed_circles_l157_15741


namespace NUMINAMATH_CALUDE_box_volume_constraint_l157_15758

theorem box_volume_constraint (x : ℕ+) : 
  (∃! x, (2 * x + 6 : ℝ) * ((x : ℝ)^3 - 8) * ((x : ℝ)^2 + 4) < 1200) := by
  sorry

end NUMINAMATH_CALUDE_box_volume_constraint_l157_15758


namespace NUMINAMATH_CALUDE_cyclic_system_solutions_l157_15791

def cyclicSystem (x : Fin 5 → ℝ) (y : ℝ) : Prop :=
  ∀ i : Fin 5, x i + x ((i + 2) % 5) = y * x ((i + 1) % 5)

theorem cyclic_system_solutions :
  ∀ x : Fin 5 → ℝ, ∀ y : ℝ,
    cyclicSystem x y ↔
      ((∀ i : Fin 5, x i = 0) ∨
      (y = 2 ∧ ∃ s : ℝ, ∀ i : Fin 5, x i = s) ∨
      ((y = (-1 + Real.sqrt 5) / 2 ∨ y = (-1 - Real.sqrt 5) / 2) ∧
        ∃ s t : ℝ, x 0 = s ∧ x 1 = t ∧ x 2 = -s + y*t ∧ x 3 = -y*s - t ∧ x 4 = y*s - t)) :=
by
  sorry


end NUMINAMATH_CALUDE_cyclic_system_solutions_l157_15791


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l157_15761

theorem complex_modulus_theorem (t : ℝ) (i : ℂ) (h_i : i^2 = -1) :
  let z : ℂ := (1 - t*i) / (1 + i)
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs (Real.sqrt 3 + t*i) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l157_15761


namespace NUMINAMATH_CALUDE_largest_proportional_part_l157_15797

theorem largest_proportional_part (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a / b = 2 ∧ a / c = 3 →
  max (total * a / (a + b + c)) (max (total * b / (a + b + c)) (total * c / (a + b + c))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_proportional_part_l157_15797


namespace NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l157_15757

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) : 
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_at_most_one_obtuse_l157_15757


namespace NUMINAMATH_CALUDE_percent_decrease_long_distance_call_l157_15767

def original_cost : ℝ := 50
def new_cost : ℝ := 10

theorem percent_decrease_long_distance_call :
  (original_cost - new_cost) / original_cost * 100 = 80 := by sorry

end NUMINAMATH_CALUDE_percent_decrease_long_distance_call_l157_15767


namespace NUMINAMATH_CALUDE_green_pieces_count_l157_15745

/-- The number of green pieces of candy in a jar, given the total number of pieces and the number of red and blue pieces. -/
def green_pieces (total red blue : ℚ) : ℚ :=
  total - red - blue

/-- Theorem: The number of green pieces is 9468 given the specified conditions. -/
theorem green_pieces_count :
  let total : ℚ := 12509.72
  let red : ℚ := 568.29
  let blue : ℚ := 2473.43
  green_pieces total red blue = 9468 := by
  sorry

end NUMINAMATH_CALUDE_green_pieces_count_l157_15745


namespace NUMINAMATH_CALUDE_actress_not_lead_plays_l157_15778

theorem actress_not_lead_plays (total_plays : ℕ) (lead_percentage : ℚ) 
  (h1 : total_plays = 100)
  (h2 : lead_percentage = 80 / 100) :
  total_plays - (total_plays * lead_percentage).floor = 20 := by
sorry

end NUMINAMATH_CALUDE_actress_not_lead_plays_l157_15778


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l157_15784

theorem expression_simplification_and_evaluation (a b : ℝ) 
  (h1 : a = 1) (h2 : b = -1) : 
  (2*a^2*b - 2*a*b^2 - b^3) / b - (a + b)*(a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l157_15784


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l157_15768

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 480)
  (h2 : garden.length = 140) :
  garden.breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l157_15768


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l157_15722

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 13) = 
  (3 * Real.sqrt 10010) / 1001 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l157_15722


namespace NUMINAMATH_CALUDE_simplified_cow_bull_ratio_l157_15751

/-- Represents the number of cattle on the farm -/
def total_cattle : ℕ := 555

/-- Represents the number of bulls on the farm -/
def bulls : ℕ := 405

/-- Calculates the number of cows on the farm -/
def cows : ℕ := total_cattle - bulls

/-- Represents the ratio of cows to bulls as a pair of natural numbers -/
def cow_bull_ratio : ℕ × ℕ := (cows, bulls)

/-- The theorem stating that the simplified ratio of cows to bulls is 10:27 -/
theorem simplified_cow_bull_ratio : 
  ∃ (k : ℕ), k > 0 ∧ cow_bull_ratio.1 = 10 * k ∧ cow_bull_ratio.2 = 27 * k := by
  sorry

end NUMINAMATH_CALUDE_simplified_cow_bull_ratio_l157_15751


namespace NUMINAMATH_CALUDE_angle_side_inequality_l157_15773

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the angle and side length functions
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry
def side_length (t : Triangle) (v : Fin 3) : ℝ := sorry

-- State the theorem
theorem angle_side_inequality (t : Triangle) :
  angle t 0 > angle t 1 → side_length t 0 > side_length t 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_side_inequality_l157_15773


namespace NUMINAMATH_CALUDE_inscribed_circle_area_l157_15728

/-- The area of a circle inscribed in a sector of a circle -/
theorem inscribed_circle_area (R a : ℝ) (h₁ : R > 0) (h₂ : a > 0) :
  let r := R * a / (R + a)
  π * r^2 = π * (R * a / (R + a))^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_l157_15728


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l157_15744

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l157_15744


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l157_15780

theorem opposite_of_negative_five : -(-5) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l157_15780


namespace NUMINAMATH_CALUDE_unique_solution_for_specific_k_and_a_l157_15746

/-- The equation (x + 2) / (kx - ax - 1) = x has exactly one solution when k = 0 and a = 1/2 -/
theorem unique_solution_for_specific_k_and_a :
  ∃! x : ℝ, (x + 2) / (0 * x - (1/2) * x - 1) = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_specific_k_and_a_l157_15746


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l157_15705

theorem subset_implies_a_range (a : ℝ) : 
  let M := {x : ℝ | (x - 1) * (x - 2) < 0}
  let N := {x : ℝ | x < a}
  (M ⊆ N) → a ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l157_15705


namespace NUMINAMATH_CALUDE_fence_poles_count_l157_15729

-- Define the parameters
def total_path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the function to calculate the number of fence poles
def fence_poles : ℕ :=
  let path_to_line := total_path_length - bridge_length
  let poles_one_side := path_to_line / pole_spacing
  2 * poles_one_side

-- Theorem statement
theorem fence_poles_count : fence_poles = 286 := by
  sorry

end NUMINAMATH_CALUDE_fence_poles_count_l157_15729


namespace NUMINAMATH_CALUDE_sequence_arrangements_l157_15715

-- Define a type for our sequence
def Sequence := Fin 5 → Fin 5

-- Define a predicate for valid permutations
def is_valid_permutation (s : Sequence) : Prop :=
  Function.Injective s ∧ Function.Surjective s

-- Define a predicate for non-adjacent odd and even numbers
def non_adjacent_odd_even (s : Sequence) : Prop :=
  ∀ i : Fin 4, (s i).val % 2 ≠ (s (i + 1)).val % 2

-- Define a predicate for decreasing then increasing sequence
def decreasing_then_increasing (s : Sequence) : Prop :=
  ∃ j : Fin 4, (∀ i : Fin 5, i < j → s i > s (i + 1)) ∧
               (∀ i : Fin 5, i ≥ j → s i < s (i + 1))

-- Define a predicate for the specific inequality condition
def specific_inequality (s : Sequence) : Prop :=
  s 0 < s 1 ∧ s 1 > s 2 ∧ s 2 > s 3 ∧ s 3 < s 4

-- State the theorem
theorem sequence_arrangements (s : Sequence) 
  (h : is_valid_permutation s) : 
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ non_adjacent_odd_even s') ∧ l.length = 12) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ decreasing_then_increasing s') ∧ l.length = 14) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ specific_inequality s') ∧ l.length = 11) :=
sorry

end NUMINAMATH_CALUDE_sequence_arrangements_l157_15715


namespace NUMINAMATH_CALUDE_one_student_in_all_activities_l157_15711

/-- Represents the number of students participating in various combinations of activities -/
structure ActivityParticipation where
  total : ℕ
  chess : ℕ
  soccer : ℕ
  music : ℕ
  atLeastTwo : ℕ

/-- The conditions of the problem -/
def clubConditions : ActivityParticipation where
  total := 30
  chess := 15
  soccer := 18
  music := 12
  atLeastTwo := 14

/-- Theorem stating that exactly one student participates in all three activities -/
theorem one_student_in_all_activities (ap : ActivityParticipation) 
  (h1 : ap = clubConditions) : 
  ∃! x : ℕ, x = (ap.chess + ap.soccer + ap.music) - (2 * ap.atLeastTwo) + ap.total - ap.atLeastTwo :=
by sorry

end NUMINAMATH_CALUDE_one_student_in_all_activities_l157_15711


namespace NUMINAMATH_CALUDE_number_of_friends_prove_number_of_friends_l157_15779

theorem number_of_friends (original_bill : ℝ) (discount_percent : ℝ) (individual_payment : ℝ) : ℝ :=
  let discounted_bill := original_bill * (1 - discount_percent / 100)
  discounted_bill / individual_payment

theorem prove_number_of_friends :
  number_of_friends 100 6 18.8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_friends_prove_number_of_friends_l157_15779


namespace NUMINAMATH_CALUDE_no_integer_solution_l157_15727

theorem no_integer_solution :
  ∀ (a b : ℤ), 
    0 ≤ a ∧ 
    0 < b ∧ 
    a < 9 ∧ 
    b < 4 →
    ¬(1 < (a : ℝ) + (b : ℝ) * Real.sqrt 5 ∧ (a : ℝ) + (b : ℝ) * Real.sqrt 5 < 9 + 4 * Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_no_integer_solution_l157_15727


namespace NUMINAMATH_CALUDE_largest_common_difference_and_terms_l157_15718

def is_decreasing_arithmetic_progression (a b c : ℤ) : Prop :=
  ∃ d : ℤ, d < 0 ∧ b = a + d ∧ c = a + 2*d

def has_two_roots (a b c : ℤ) : Prop :=
  b^2 - 4*a*c ≥ 0

theorem largest_common_difference_and_terms 
  (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : is_decreasing_arithmetic_progression a b c)
  (h3 : has_two_roots (2*a) (2*b) c)
  (h4 : has_two_roots (2*a) c (2*b))
  (h5 : has_two_roots (2*b) (2*a) c)
  (h6 : has_two_roots (2*b) c (2*a))
  (h7 : has_two_roots c (2*a) (2*b))
  (h8 : has_two_roots c (2*b) (2*a)) :
  ∃ d : ℤ, d = -5 ∧ a = 4 ∧ b = -1 ∧ c = -6 ∧ 
  ∀ d' : ℤ, (∃ a' b' c' : ℤ, 
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧
    is_decreasing_arithmetic_progression a' b' c' ∧
    has_two_roots (2*a') (2*b') c' ∧
    has_two_roots (2*a') c' (2*b') ∧
    has_two_roots (2*b') (2*a') c' ∧
    has_two_roots (2*b') c' (2*a') ∧
    has_two_roots c' (2*a') (2*b') ∧
    has_two_roots c' (2*b') (2*a') ∧
    d' < 0) → d' ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_common_difference_and_terms_l157_15718


namespace NUMINAMATH_CALUDE_x35x_divisible_by_18_l157_15759

def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 350 + x

theorem x35x_divisible_by_18 : 
  ∃! (x : ℕ), is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 8 :=
sorry

end NUMINAMATH_CALUDE_x35x_divisible_by_18_l157_15759


namespace NUMINAMATH_CALUDE_square_of_1085_l157_15702

theorem square_of_1085 : (1085 : ℕ)^2 = 1177225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1085_l157_15702


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_half_l157_15730

/-- If the terminal side of angle α passes through the point P(-1, √3), then cos(α - π/2) = √3/2 -/
theorem cos_alpha_minus_pi_half (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α * Real.cos 0 - Real.sin α * Real.sin 0 ∧ 
                    y = Real.sin α * Real.cos 0 + Real.cos α * Real.sin 0) →
  Real.cos (α - π/2) = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_half_l157_15730


namespace NUMINAMATH_CALUDE_restaurant_ratio_change_l157_15737

/-- Given a restaurant with an initial ratio of cooks to waiters of 3:8,
    9 cooks, and 12 additional waiters hired, prove that the new ratio
    of cooks to waiters is 1:4. -/
theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
    (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_waiters = (8 * initial_cooks) / 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_restaurant_ratio_change_l157_15737


namespace NUMINAMATH_CALUDE_trigonometric_equality_l157_15775

theorem trigonometric_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
   Real.cos (159 * π / 180) * Real.cos (99 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l157_15775


namespace NUMINAMATH_CALUDE_soda_cost_l157_15712

theorem soda_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : 4 * alice_burgers + 3 * alice_sodas = alice_total)
  (h_bill : 3 * bill_burgers + 2 * bill_sodas = bill_total)
  (h_alice_total : alice_total = 500)
  (h_bill_total : bill_total = 370)
  (h_same_prices : alice_burgers = bill_burgers ∧ alice_sodas = bill_sodas) :
  alice_sodas = 20 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_l157_15712


namespace NUMINAMATH_CALUDE_bike_price_l157_15783

theorem bike_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_bike_price_l157_15783


namespace NUMINAMATH_CALUDE_initial_money_calculation_l157_15781

theorem initial_money_calculation (initial_money : ℚ) : 
  (2 / 5 : ℚ) * initial_money = 400 → initial_money = 1000 :=
by
  sorry

#check initial_money_calculation

end NUMINAMATH_CALUDE_initial_money_calculation_l157_15781


namespace NUMINAMATH_CALUDE_comparison_theorem_l157_15719

theorem comparison_theorem :
  (2 * Real.sqrt 3 < 3 * Real.sqrt 2) ∧
  ((Real.sqrt 10 - 1) / 3 > 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_comparison_theorem_l157_15719


namespace NUMINAMATH_CALUDE_circle_tangent_and_symmetric_points_l157_15747

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 6 = 0

-- Define point M
def point_M : ℝ × ℝ := (-5, 11)

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the dot product of OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

-- Theorem statement
theorem circle_tangent_and_symmetric_points :
  ∃ (P Q : ℝ × ℝ) (m : ℝ),
    (∀ x y, circle_C x y ↔ (x + 1)^2 + (y - 3)^2 = 16) ∧
    (∀ x y, (x = -5 ∨ 3*x + 4*y - 29 = 0) ↔ 
      (circle_C x y ∧ ∃ t, x = point_M.1 + t * (x - point_M.1) ∧ 
                           y = point_M.2 + t * (y - point_M.2) ∧ 
                           t ≠ 0)) ∧
    circle_C P.1 P.2 ∧ 
    circle_C Q.1 Q.2 ∧
    line_eq m P.1 P.2 ∧
    line_eq m Q.1 Q.2 ∧
    dot_product_OP_OQ P Q = -7 ∧
    m = -1 ∧
    (∀ x y, (y = -x ∨ y = -x + 2) ↔ 
      (∃ t, x = P.1 + t * (Q.1 - P.1) ∧ y = P.2 + t * (Q.2 - P.2))) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_and_symmetric_points_l157_15747


namespace NUMINAMATH_CALUDE_right_triangle_area_l157_15788

/-- The area of a right triangle with hypotenuse 5 and shortest side 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ),
  a = 3 →
  c = 5 →
  a ≤ b →
  b ≤ c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l157_15788


namespace NUMINAMATH_CALUDE_syllogism_invalid_l157_15708

-- Define the sets and properties
def Geese : Type := Unit
def Senators : Type := Unit
def eats_cabbage (α : Type) : α → Prop := fun _ => True

-- Define the syllogism
def invalid_syllogism (g : Geese) (s : Senators) : Prop :=
  eats_cabbage Geese g ∧ eats_cabbage Senators s → s = g

-- Theorem stating that the syllogism is invalid
theorem syllogism_invalid :
  ¬∀ (g : Geese) (s : Senators), invalid_syllogism g s :=
sorry

end NUMINAMATH_CALUDE_syllogism_invalid_l157_15708


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l157_15795

theorem imaginary_part_of_complex_fraction :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l157_15795


namespace NUMINAMATH_CALUDE_max_sum_squares_and_products_l157_15724

def S : Finset ℕ := {2, 4, 6, 8}

theorem max_sum_squares_and_products (f g h j : ℕ) 
  (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S) (hj : j ∈ S)
  (hsum : f + g + h + j = 20) :
  (∃ (f' g' h' j' : ℕ), f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧ j' ∈ S ∧ 
    f' + g' + h' + j' = 20 ∧
    f'^2 + g'^2 + h'^2 + j'^2 ≤ 120 ∧
    (f'^2 + g'^2 + h'^2 + j'^2 = 120 → 
      f' * g' + g' * h' + h' * j' + f' * j' = 100)) ∧
  f^2 + g^2 + h^2 + j^2 ≤ 120 ∧
  (f^2 + g^2 + h^2 + j^2 = 120 → 
    f * g + g * h + h * j + f * j = 100) :=
sorry

end NUMINAMATH_CALUDE_max_sum_squares_and_products_l157_15724


namespace NUMINAMATH_CALUDE_intersection_perpendicular_line_max_distance_to_origin_l157_15720

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y - 3 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3*x + y - 1 = 0

-- Define the general form of line l
def line_l (m x y : ℝ) : Prop := m*x + y - 2*(m+1) = 0

-- Part I
theorem intersection_perpendicular_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, line1 x y ∧ line2 x y → a*x + b*y + c = 0) ∧
    (∀ x y : ℝ, (a*x + b*y + c = 0) → (3*a + b = 0)) ∧
    (a = 5 ∧ b = -15 ∧ c = -18) :=
sorry

-- Part II
theorem max_distance_to_origin :
  ∃ (d : ℝ), 
    (∀ m x y : ℝ, line_l m x y → (x^2 + y^2 ≤ d^2)) ∧
    (∃ m x y : ℝ, line_l m x y ∧ x^2 + y^2 = d^2) ∧
    (d = 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_perpendicular_line_max_distance_to_origin_l157_15720


namespace NUMINAMATH_CALUDE_necessary_condition_transitivity_l157_15736

theorem necessary_condition_transitivity (A B C : Prop) :
  (B → C) → (A → B) → (A → C) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_transitivity_l157_15736


namespace NUMINAMATH_CALUDE_pentagon_reconstruction_l157_15756

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A pentagon in 2D space -/
structure Pentagon where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- The reflected points of a pentagon -/
structure ReflectedPoints where
  A1 : Point2D
  B1 : Point2D
  C1 : Point2D
  D1 : Point2D
  E1 : Point2D

/-- Function to reflect a point with respect to another point -/
def reflect (p : Point2D) (center : Point2D) : Point2D :=
  { x := 2 * center.x - p.x
    y := 2 * center.y - p.y }

/-- Theorem stating that a pentagon can be reconstructed from its reflected points -/
theorem pentagon_reconstruction (reflectedPoints : ReflectedPoints) :
  ∃! (original : Pentagon),
    reflectedPoints.A1 = reflect original.A original.B ∧
    reflectedPoints.B1 = reflect original.B original.C ∧
    reflectedPoints.C1 = reflect original.C original.D ∧
    reflectedPoints.D1 = reflect original.D original.E ∧
    reflectedPoints.E1 = reflect original.E original.A :=
  sorry


end NUMINAMATH_CALUDE_pentagon_reconstruction_l157_15756


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l157_15771

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(2*x + 4) = (8 : ℝ)^(3*x + 4) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l157_15771


namespace NUMINAMATH_CALUDE_smallest_sum_pell_equation_l157_15790

theorem smallest_sum_pell_equation :
  ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^2 - 29*y^2 = 1 ∧
  ∀ (x' y' : ℕ), x' ≥ 1 → y' ≥ 1 → x'^2 - 29*y'^2 = 1 → x + y ≤ x' + y' ∧
  x + y = 11621 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_pell_equation_l157_15790


namespace NUMINAMATH_CALUDE_cookie_is_circle_with_radius_sqrt35_l157_15738

-- Define the equation of the cookie's boundary
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 10 = 6*x + 12*y

-- Theorem stating that the cookie's boundary is a circle with radius √35
theorem cookie_is_circle_with_radius_sqrt35 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_boundary x y ↔ (x - h)^2 + (y - k)^2 = 35 :=
sorry

end NUMINAMATH_CALUDE_cookie_is_circle_with_radius_sqrt35_l157_15738


namespace NUMINAMATH_CALUDE_green_ball_probability_l157_15742

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

theorem green_ball_probability :
  let c1 := Container.mk 8 4
  let c2 := Container.mk 3 5
  let c3 := Container.mk 4 6
  totalGreenProbability c1 c2 c3 = 187 / 360 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l157_15742


namespace NUMINAMATH_CALUDE_cindys_calculation_l157_15704

theorem cindys_calculation (x : ℝ) : (x - 12) / 4 = 72 → (x - 5) / 9 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l157_15704


namespace NUMINAMATH_CALUDE_irrationality_of_32_minus_sqrt3_l157_15752

theorem irrationality_of_32_minus_sqrt3 : Irrational (32 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_irrationality_of_32_minus_sqrt3_l157_15752


namespace NUMINAMATH_CALUDE_sum_of_roots_l157_15760

theorem sum_of_roots (x : ℝ) : (x + 2) * (x - 3) = 16 → ∃ y : ℝ, (y + 2) * (y - 3) = 16 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l157_15760


namespace NUMINAMATH_CALUDE_inequality_solution_l157_15739

theorem inequality_solution (x : ℝ) : (2 * x) / 5 ≤ 3 + x ∧ 3 + x < -3 * (1 + x) ↔ x ∈ Set.Ici (-5) ∩ Set.Iio (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l157_15739


namespace NUMINAMATH_CALUDE_quadratic_max_value_l157_15717

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a ≤ 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, x^2 + 2*x + a = 4) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l157_15717


namespace NUMINAMATH_CALUDE_decagon_interior_exterior_angle_sum_l157_15721

theorem decagon_interior_exterior_angle_sum (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 ↔ n = 10 :=
sorry

end NUMINAMATH_CALUDE_decagon_interior_exterior_angle_sum_l157_15721


namespace NUMINAMATH_CALUDE_min_value_problem_l157_15713

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 4*x - 4*y - 1 = 0}
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) →
  (2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
   ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 
     2 / a' + 3 / b' = 5 + 2 * Real.sqrt 6 ∧
     ∃ (p q : ℝ × ℝ), p ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧ 
       q ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧
       p ∈ circle ∧ q ∈ circle ∧
       Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l157_15713


namespace NUMINAMATH_CALUDE_man_speed_with_current_l157_15787

/-- Calculates the man's speed with the current given his speed against the current and the current speed. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Proves that given a man's speed against a current of 9.6 km/hr and a current speed of 3.2 km/hr, 
    the man's speed with the current is 16.0 km/hr. -/
theorem man_speed_with_current :
  speed_with_current 9.6 3.2 = 16.0 := by
  sorry

#eval speed_with_current 9.6 3.2

end NUMINAMATH_CALUDE_man_speed_with_current_l157_15787


namespace NUMINAMATH_CALUDE_factors_of_45_proportion_l157_15749

theorem factors_of_45_proportion :
  ∃ (a b c d : ℕ), 
    (a ∣ 45) ∧ (b ∣ 45) ∧ (c ∣ 45) ∧ (d ∣ 45) ∧
    (b = 3 * a) ∧ (d = 3 * c) ∧
    (b : ℚ) / a = (d : ℚ) / c ∧ (b : ℚ) / a = 3 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_45_proportion_l157_15749


namespace NUMINAMATH_CALUDE_complement_of_A_l157_15794

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

theorem complement_of_A : (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l157_15794


namespace NUMINAMATH_CALUDE_polynomial_factorization_l157_15754

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 3) * (x + 7) * (x + 8) * (x + 12) - 5 * x^2 =
  (2 * x^2 + (60 - Real.sqrt 5) * x + 80) * (2 * x^2 + (60 + Real.sqrt 5) * x + 80) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l157_15754


namespace NUMINAMATH_CALUDE_melissa_shoe_repair_time_l157_15743

/-- The total time Melissa spends repairing shoes -/
theorem melissa_shoe_repair_time (buckle_time heel_time strap_time sole_time : ℕ) 
  (num_pairs : ℕ) : 
  buckle_time = 5 → 
  heel_time = 10 → 
  strap_time = 7 → 
  sole_time = 12 → 
  num_pairs = 8 → 
  (buckle_time + heel_time + strap_time + sole_time) * 2 * num_pairs = 544 :=
by sorry

end NUMINAMATH_CALUDE_melissa_shoe_repair_time_l157_15743


namespace NUMINAMATH_CALUDE_negation_of_proposition_l157_15763

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x > Real.log x)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ ≤ Real.log x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l157_15763


namespace NUMINAMATH_CALUDE_triangle_theorem_l157_15785

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  angle_sum : A + B + C = π
  sine_rule_ab : a / (Real.sin A) = b / (Real.sin B)
  sine_rule_bc : b / (Real.sin B) = c / (Real.sin C)

/-- Main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.A * Real.sin t.B + Real.sin t.B * Real.sin t.C + Real.cos (2 * t.B) = 1) :
  -- Part 1: a, b, c are in arithmetic progression
  t.a + t.c = 2 * t.b ∧ 
  -- Part 2: If C = 2π/3, then a/b = 3/5
  (t.C = 2 * π / 3 → t.a / t.b = 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l157_15785


namespace NUMINAMATH_CALUDE_tape_length_for_circular_base_l157_15799

/-- The length of tape needed for a circular lamp base -/
theorem tape_length_for_circular_base :
  let area : ℝ := 176
  let π_approx : ℝ := 22 / 7
  let extra_tape : ℝ := 3
  let radius : ℝ := Real.sqrt (area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let total_length : ℝ := circumference + extra_tape
  ∃ ε > 0, abs (total_length - 50.058) < ε :=
by sorry

end NUMINAMATH_CALUDE_tape_length_for_circular_base_l157_15799


namespace NUMINAMATH_CALUDE_race_winning_post_distance_l157_15703

theorem race_winning_post_distance
  (speed_ratio : ℝ)
  (head_start : ℝ)
  (h_speed_ratio : speed_ratio = 1.75)
  (h_head_start : head_start = 84)
  : ∃ (distance : ℝ),
    distance = 196 ∧
    distance / speed_ratio = (distance - head_start) / 1 :=
by sorry

end NUMINAMATH_CALUDE_race_winning_post_distance_l157_15703


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l157_15709

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 4 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y - 2 = 0

-- Theorem for parallel lines
theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = 1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y → l₂ m x y → x * x + y * y = 0) → m = -2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l157_15709


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l157_15792

theorem arithmetic_calculations :
  (456 - 9 * 8 = 384) ∧
  (387 + 126 - 212 = 301) ∧
  (533 - (108 + 209) = 216) ∧
  ((746 - 710) / 6 = 6) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l157_15792


namespace NUMINAMATH_CALUDE_unique_solution_cubic_linear_l157_15796

/-- The system of equations y = x^3 and y = 4x + m has exactly one real solution if and only if m = -8 -/
theorem unique_solution_cubic_linear (m : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^3 = 4*p.1 + m ∧ p.2 = p.1^3) ↔ m = -8 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_linear_l157_15796


namespace NUMINAMATH_CALUDE_y_value_l157_15764

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l157_15764


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l157_15798

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_a4 : a 4 = 4) :
  a 2 * a 6 = a 4 * a 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l157_15798


namespace NUMINAMATH_CALUDE_cube_sum_implies_sum_l157_15776

theorem cube_sum_implies_sum (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_implies_sum_l157_15776


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l157_15765

theorem triangle_angle_inequality (a b c α β γ : Real) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_sides : (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l157_15765


namespace NUMINAMATH_CALUDE_wattage_increase_percentage_l157_15753

theorem wattage_increase_percentage (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 110)
  (h2 : new_wattage = 143) : 
  (new_wattage - original_wattage) / original_wattage * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_wattage_increase_percentage_l157_15753


namespace NUMINAMATH_CALUDE_bill_sue_score_ratio_l157_15766

theorem bill_sue_score_ratio :
  ∀ (john_score sue_score : ℕ),
    45 = john_score + 20 →
    45 + john_score + sue_score = 160 →
    (45 : ℚ) / sue_score = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bill_sue_score_ratio_l157_15766


namespace NUMINAMATH_CALUDE_prob_green_ball_is_13_28_l157_15793

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers A, B, C, and D -/
def containers : List Container := [
  ⟨4, 6⟩,  -- Container A
  ⟨8, 6⟩,  -- Container B
  ⟨8, 6⟩,  -- Container C
  ⟨3, 7⟩   -- Container D
]

/-- The number of containers -/
def numContainers : ℕ := containers.length

/-- The probability of selecting a green ball -/
def probGreenBall : ℚ := 
  (1 / numContainers) * (containers.map greenProbability).sum

theorem prob_green_ball_is_13_28 : probGreenBall = 13/28 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_ball_is_13_28_l157_15793


namespace NUMINAMATH_CALUDE_tetrahedron_volume_zero_l157_15706

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k, k < 11 → a (k + 1) = a k + d

-- Define the vertices of the tetrahedron
def tetrahedron_vertices (a : ℕ → ℝ) : Fin 4 → ℝ × ℝ × ℝ
| 0 => (a 1 ^ 2, a 2 ^ 2, a 3 ^ 2)
| 1 => (a 4 ^ 2, a 5 ^ 2, a 6 ^ 2)
| 2 => (a 7 ^ 2, a 8 ^ 2, a 9 ^ 2)
| 3 => (a 10 ^ 2, a 11 ^ 2, a 12 ^ 2)

-- Define the volume of a tetrahedron
def tetrahedron_volume (vertices : Fin 4 → ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_zero (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_progression a d →
  tetrahedron_volume (tetrahedron_vertices a) = 0 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_zero_l157_15706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l157_15750

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (x : ℝ) :
  arithmetic_sequence a →
  a 1 = f (x + 1) →
  a 2 = 0 →
  a 3 = f (x - 1) →
  (∀ n : ℕ, a n = 2*n - 4) ∨ (∀ n : ℕ, a n = 4 - 2*n) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l157_15750
