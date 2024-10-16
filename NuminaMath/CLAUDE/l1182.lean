import Mathlib

namespace NUMINAMATH_CALUDE_fraction_of_books_sold_l1182_118207

/-- Given a collection of books where some were sold and some remained unsold,
    this theorem proves that the fraction of books sold is 2/3 under specific conditions. -/
theorem fraction_of_books_sold (total_books : ℕ) (sold_books : ℕ) : 
  (total_books > 50) →
  (sold_books = total_books - 50) →
  (sold_books * 5 = 500) →
  (sold_books : ℚ) / total_books = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_books_sold_l1182_118207


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l1182_118257

theorem negative_64_to_four_thirds_equals_256 : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_equals_256_l1182_118257


namespace NUMINAMATH_CALUDE_square_room_perimeter_l1182_118241

theorem square_room_perimeter (area : ℝ) (perimeter : ℝ) : 
  area = 500 → perimeter = 40 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_square_room_perimeter_l1182_118241


namespace NUMINAMATH_CALUDE_ellipse_equation_not_standard_l1182_118292

theorem ellipse_equation_not_standard (a c : ℝ) (h1 : a = 6) (h2 : c = 1) :
  let b := Real.sqrt (a^2 - c^2)
  ¬ ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/36 + y^2/35 = 1) ∨
     (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ y^2/36 + x^2/35 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_not_standard_l1182_118292


namespace NUMINAMATH_CALUDE_special_polyhedron_volume_l1182_118231

/-- A convex polyhedron with specific properties -/
structure SpecialPolyhedron where
  -- The polyhedron is convex
  isConvex : Bool
  -- Number of square faces
  numSquareFaces : Nat
  -- Number of hexagonal faces
  numHexagonalFaces : Nat
  -- No two square faces share a vertex
  noSharedSquareVertices : Bool
  -- All edges have unit length
  unitEdgeLength : Bool

/-- The volume of the special polyhedron -/
noncomputable def specialPolyhedronVolume (p : SpecialPolyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the special polyhedron -/
theorem special_polyhedron_volume :
  ∀ (p : SpecialPolyhedron),
    p.isConvex = true ∧
    p.numSquareFaces = 6 ∧
    p.numHexagonalFaces = 8 ∧
    p.noSharedSquareVertices = true ∧
    p.unitEdgeLength = true →
    specialPolyhedronVolume p = 8 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_volume_l1182_118231


namespace NUMINAMATH_CALUDE_trent_travel_distance_l1182_118220

/-- The distance Trent walked from his house to the bus stop -/
def distance_to_bus_stop : ℕ := 4

/-- The distance Trent rode the bus to the library -/
def distance_on_bus : ℕ := 7

/-- The total distance Trent traveled in blocks -/
def total_distance : ℕ := 2 * (distance_to_bus_stop + distance_on_bus)

theorem trent_travel_distance : total_distance = 22 := by
  sorry

end NUMINAMATH_CALUDE_trent_travel_distance_l1182_118220


namespace NUMINAMATH_CALUDE_function_value_at_eight_l1182_118249

theorem function_value_at_eight (f : ℝ → ℝ) (h : ∀ x, f (3 * x + 2) = 9 * x + 8) :
  f 8 = 26 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_eight_l1182_118249


namespace NUMINAMATH_CALUDE_train_crossing_time_l1182_118217

theorem train_crossing_time (length : ℝ) (time_second : ℝ) (crossing_time : ℝ) : 
  length = 120 →
  time_second = 12 →
  crossing_time = 10.909090909090908 →
  (length / (length / time_second + (2 * length) / crossing_time - length / time_second)) = 10 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1182_118217


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1182_118294

/-- A geometric sequence with given second and fifth terms -/
structure GeometricSequence where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- The third term of the geometric sequence -/
def third_term (g : GeometricSequence) : ℝ := 49

/-- The sum of the first four terms of the geometric sequence -/
def sum_first_four (g : GeometricSequence) : ℝ := 183.75

/-- Theorem stating the properties of the geometric sequence -/
theorem geometric_sequence_properties (g : GeometricSequence) :
  third_term g = 49 ∧ sum_first_four g = 183.75 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_properties_l1182_118294


namespace NUMINAMATH_CALUDE_concave_iff_m_nonneg_l1182_118280

/-- A function f is concave on a set A if for any x₁, x₂ ∈ A,
    f((x₁ + x₂)/2) ≤ (1/2)[f(x₁) + f(x₂)] -/
def IsConcave (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The function f(x) = mx² + x -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x

theorem concave_iff_m_nonneg (m : ℝ) :
  IsConcave (f m) ↔ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_concave_iff_m_nonneg_l1182_118280


namespace NUMINAMATH_CALUDE_count_non_degenerate_triangles_l1182_118202

/-- The number of points in the figure -/
def total_points : ℕ := 16

/-- The number of collinear points on the base of the triangle -/
def base_points : ℕ := 5

/-- The number of collinear points on the semicircle -/
def semicircle_points : ℕ := 5

/-- The number of non-collinear points -/
def other_points : ℕ := total_points - base_points - semicircle_points

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of non-degenerate triangles -/
def non_degenerate_triangles : ℕ := 
  choose total_points 3 - 2 * choose base_points 3

theorem count_non_degenerate_triangles : 
  non_degenerate_triangles = 540 := by sorry

end NUMINAMATH_CALUDE_count_non_degenerate_triangles_l1182_118202


namespace NUMINAMATH_CALUDE_parabola_point_comparison_l1182_118240

theorem parabola_point_comparison 
  (m : ℝ) (t x₁ x₂ y₁ y₂ : ℝ) 
  (h_m : m > 0)
  (h_x₁ : t < x₁ ∧ x₁ < t + 1)
  (h_x₂ : t + 2 < x₂ ∧ x₂ < t + 3)
  (h_y₁ : y₁ = m * x₁^2 - 2 * m * x₁ + 1)
  (h_y₂ : y₂ = m * x₂^2 - 2 * m * x₂ + 1)
  (h_t : t ≥ 1) :
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_parabola_point_comparison_l1182_118240


namespace NUMINAMATH_CALUDE_multiple_of_nine_square_greater_than_144_less_than_30_l1182_118232

theorem multiple_of_nine_square_greater_than_144_less_than_30 (x : ℕ) :
  (∃ k : ℕ, x = 9 * k) →
  x^2 > 144 →
  x < 30 →
  x = 18 ∨ x = 27 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_nine_square_greater_than_144_less_than_30_l1182_118232


namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l1182_118271

/-- Represents the financial data of a person -/
structure PersonFinance where
  income : ℕ
  savings : ℕ
  expenditure : ℕ

/-- The problem statement -/
theorem expenditure_ratio_proof 
  (p1 p2 : PersonFinance)
  (h1 : p1.income = 3000)
  (h2 : p1.income * 4 = p2.income * 5)
  (h3 : p1.savings = 1200)
  (h4 : p2.savings = 1200)
  (h5 : p1.expenditure = p1.income - p1.savings)
  (h6 : p2.expenditure = p2.income - p2.savings)
  : p1.expenditure * 2 = p2.expenditure * 3 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_proof_l1182_118271


namespace NUMINAMATH_CALUDE_circplus_commutative_l1182_118273

/-- The ⊕ operation -/
def circplus (a b : ℝ) : ℝ := a^2 + a*b + b^2

/-- Theorem: x ⊕ y = y ⊕ x for all real x and y -/
theorem circplus_commutative : ∀ x y : ℝ, circplus x y = circplus y x := by
  sorry

end NUMINAMATH_CALUDE_circplus_commutative_l1182_118273


namespace NUMINAMATH_CALUDE_tree_spacing_l1182_118230

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last tree
    is 175 feet. -/
theorem tree_spacing (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (n - 1) * d / 4 = 175 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l1182_118230


namespace NUMINAMATH_CALUDE_modular_exponentiation_l1182_118225

theorem modular_exponentiation (m : ℕ) : 
  0 ≤ m ∧ m < 29 ∧ (4 * m) % 29 = 1 → (5^m)^4 % 29 - 3 = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_l1182_118225


namespace NUMINAMATH_CALUDE_adjacent_roll_probability_l1182_118277

/-- The number of people sitting around the circular table -/
def n : ℕ := 5

/-- The number of sides on the die -/
def d : ℕ := 8

/-- The probability of no two adjacent people rolling the same number -/
def prob : ℚ := 637 / 2048

/-- Theorem stating the probability of no two adjacent people rolling the same number -/
theorem adjacent_roll_probability : 
  (n = 5 ∧ d = 8) → prob = 637 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_roll_probability_l1182_118277


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1182_118247

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x + 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y + 1 → x ≤ y ∧
  x = (-7 - Real.sqrt 349) / 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1182_118247


namespace NUMINAMATH_CALUDE_max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l1182_118266

theorem max_moves_in_grid (n : ℕ) : n > 0 → n * (n + 1) ≤ 200 → n ≤ 14 := by
  sorry

theorem fourteen_fits_grid : 14 * (14 + 1) ≤ 200 := by
  sorry

theorem fifteen_exceeds_grid : 15 * (15 + 1) > 200 := by
  sorry

theorem max_moves_is_fourteen : 
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) ≤ 200 ∧ ∀ (m : ℕ), m > n → m * (m + 1) > 200 := by
  sorry

end NUMINAMATH_CALUDE_max_moves_in_grid_fourteen_fits_grid_fifteen_exceeds_grid_max_moves_is_fourteen_l1182_118266


namespace NUMINAMATH_CALUDE_remainder_sum_l1182_118285

theorem remainder_sum (a b : ℤ) (ha : a % 70 = 64) (hb : b % 105 = 99) :
  (a + b) % 35 = 23 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1182_118285


namespace NUMINAMATH_CALUDE_product_mod_seven_l1182_118218

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1182_118218


namespace NUMINAMATH_CALUDE_guppies_count_l1182_118290

/-- The number of guppies Haylee has -/
def haylee_guppies : ℕ := 3 * 12

/-- The number of guppies Jose has -/
def jose_guppies : ℕ := haylee_guppies / 2

/-- The number of guppies Charliz has -/
def charliz_guppies : ℕ := jose_guppies / 3

/-- The number of guppies Nicolai has -/
def nicolai_guppies : ℕ := charliz_guppies * 4

/-- The total number of guppies owned by all four friends -/
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

theorem guppies_count : total_guppies = 84 := by
  sorry

end NUMINAMATH_CALUDE_guppies_count_l1182_118290


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1182_118256

theorem perfect_square_condition (k : ℝ) :
  (∀ x y : ℝ, ∃ z : ℝ, 4 * x^2 - (k - 1) * x * y + 9 * y^2 = z^2) →
  k = 13 ∨ k = -11 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1182_118256


namespace NUMINAMATH_CALUDE_geometric_product_and_quotient_l1182_118261

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem geometric_product_and_quotient
  (a b : ℕ → ℝ)
  (ha : IsGeometric a)
  (hb : IsGeometric b)
  (hb_nonzero : ∀ n, b n ≠ 0) :
  IsGeometric (fun n ↦ a n * b n) ∧
  IsGeometric (fun n ↦ a n / b n) :=
sorry

end NUMINAMATH_CALUDE_geometric_product_and_quotient_l1182_118261


namespace NUMINAMATH_CALUDE_circular_fields_area_comparison_l1182_118269

theorem circular_fields_area_comparison (r₁ r₂ : ℝ) (h : r₂ = (5/2) * r₁) :
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_comparison_l1182_118269


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l1182_118279

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12 ∧ a + b = 49 ∧
  ∀ (c d : ℕ+), c ≠ d → (1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12 → c + d ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l1182_118279


namespace NUMINAMATH_CALUDE_circle_radius_on_right_triangle_l1182_118228

theorem circle_radius_on_right_triangle (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a = 7.5 →  -- shorter leg
  b = 10 →  -- longer leg (diameter of circle)
  6^2 + (c - r)^2 = r^2 →  -- chord condition
  r = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_on_right_triangle_l1182_118228


namespace NUMINAMATH_CALUDE_index_difference_proof_l1182_118216

/-- Calculates the index for a subgroup within a larger group -/
def calculate_index (n k x : ℕ) : ℚ :=
  (n - k : ℚ) / n * (n - x : ℚ) / n

theorem index_difference_proof (n k x_f x_m : ℕ) 
  (h_n : n = 25)
  (h_k : k = 8)
  (h_x_f : x_f = 6)
  (h_x_m : x_m = 10) :
  calculate_index n k x_f - calculate_index n (n - k) x_m = 203 / 625 := by
  sorry

#eval calculate_index 25 8 6 - calculate_index 25 17 10

end NUMINAMATH_CALUDE_index_difference_proof_l1182_118216


namespace NUMINAMATH_CALUDE_complex_inequalities_l1182_118281

theorem complex_inequalities :
  (∀ z w : ℂ, Complex.abs z + Complex.abs w ≤ Complex.abs (z + w) + Complex.abs (z - w)) ∧
  (∀ z₁ z₂ z₃ z₄ : ℂ, 
    Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ ≤
    Complex.abs (z₁ + z₂) + Complex.abs (z₁ + z₃) + Complex.abs (z₁ + z₄) +
    Complex.abs (z₂ + z₃) + Complex.abs (z₂ + z₄) + Complex.abs (z₃ + z₄)) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequalities_l1182_118281


namespace NUMINAMATH_CALUDE_distance_after_walk_l1182_118296

/-- A regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  regular : side_length = 3

/-- Walking distance along the perimeter -/
def walking_distance : ℝ := 10

/-- Calculate the distance between start and end points after walking along the perimeter -/
def distance_from_start (h : RegularHexagon) (d : ℝ) : ℝ := sorry

/-- Theorem: The distance from start to end after walking 10 km on a regular hexagon with 3 km sides is 1 km -/
theorem distance_after_walk (h : RegularHexagon) :
  distance_from_start h walking_distance = 1 := by sorry

end NUMINAMATH_CALUDE_distance_after_walk_l1182_118296


namespace NUMINAMATH_CALUDE_work_completion_time_l1182_118270

/-- Given workers a, b, and c who can complete a work in 16, x, and 12 days respectively,
    and together they complete the work in 3.2 days, prove that x = 6. -/
theorem work_completion_time (x : ℝ) 
  (h1 : 1/16 + 1/x + 1/12 = 1/3.2) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1182_118270


namespace NUMINAMATH_CALUDE_median_in_interval_75_79_l1182_118204

structure ScoreInterval :=
  (lower upper : ℕ)
  (count : ℕ)

def total_students : ℕ := 100

def score_distribution : List ScoreInterval :=
  [⟨85, 89, 18⟩, ⟨80, 84, 15⟩, ⟨75, 79, 20⟩, ⟨70, 74, 25⟩, ⟨65, 69, 12⟩, ⟨60, 64, 10⟩]

def cumulative_count (n : ℕ) : ℕ :=
  (score_distribution.take n).foldl (λ acc interval => acc + interval.count) 0

theorem median_in_interval_75_79 :
  ∃ k, k ∈ [75, 76, 77, 78, 79] ∧
    cumulative_count 2 < total_students / 2 ∧
    total_students / 2 ≤ cumulative_count 3 :=
  sorry

end NUMINAMATH_CALUDE_median_in_interval_75_79_l1182_118204


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1182_118254

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l1182_118254


namespace NUMINAMATH_CALUDE_clare_remaining_money_l1182_118255

/-- Given Clare's initial money and her purchases, calculate the remaining money. -/
def remaining_money (initial_money bread_price milk_price bread_quantity milk_quantity : ℕ) : ℕ :=
  initial_money - (bread_price * bread_quantity + milk_price * milk_quantity)

/-- Theorem: Clare has $35 left after her purchases. -/
theorem clare_remaining_money :
  remaining_money 47 2 2 4 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_clare_remaining_money_l1182_118255


namespace NUMINAMATH_CALUDE_turn_duration_is_one_hour_l1182_118264

/-- Represents the time taken to complete the work individually -/
structure WorkTime where
  a : ℝ
  b : ℝ

/-- Represents the amount of work done per hour -/
structure WorkRate where
  a : ℝ
  b : ℝ

/-- The duration of each turn when working alternately -/
def turn_duration (wt : WorkTime) (wr : WorkRate) : ℝ :=
  sorry

/-- The theorem stating that the turn duration is 1 hour -/
theorem turn_duration_is_one_hour (wt : WorkTime) (wr : WorkRate) :
  wt.a = 4 →
  wt.b = 12 →
  wr.a = 1 / wt.a →
  wr.b = 1 / wt.b →
  (3 * wr.a * turn_duration wt wr + 3 * wr.b * turn_duration wt wr = 1) →
  turn_duration wt wr = 1 :=
sorry

end NUMINAMATH_CALUDE_turn_duration_is_one_hour_l1182_118264


namespace NUMINAMATH_CALUDE_jills_hair_braiding_l1182_118226

/-- Given the conditions of Jill's hair braiding for the dance team, 
    prove that each dancer has 5 braids. -/
theorem jills_hair_braiding 
  (num_dancers : ℕ) 
  (time_per_braid : ℕ) 
  (total_time_minutes : ℕ) 
  (h1 : num_dancers = 8)
  (h2 : time_per_braid = 30)
  (h3 : total_time_minutes = 20) :
  (total_time_minutes * 60) / (time_per_braid * num_dancers) = 5 :=
sorry

end NUMINAMATH_CALUDE_jills_hair_braiding_l1182_118226


namespace NUMINAMATH_CALUDE_pencils_removed_l1182_118287

/-- Given a jar of pencils, prove that the number of pencils removed is correct. -/
theorem pencils_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 87 → remaining = 83 → removed = original - remaining → removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_removed_l1182_118287


namespace NUMINAMATH_CALUDE_problem_1_l1182_118259

theorem problem_1 : -53 + 21 - (-79) - 37 = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_l1182_118259


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1182_118221

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1182_118221


namespace NUMINAMATH_CALUDE_eduardo_classes_l1182_118267

theorem eduardo_classes (x : ℕ) : 
  x + 2 * x = 9 → x = 3 := by sorry

end NUMINAMATH_CALUDE_eduardo_classes_l1182_118267


namespace NUMINAMATH_CALUDE_plane_through_three_points_l1182_118236

/-- A plane passing through three points in 3D space. -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane. -/
def Point3D.liesOn (p : Point3D) (plane : Plane3D) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The three given points. -/
def P₀ : Point3D := ⟨2, -1, 2⟩
def P₁ : Point3D := ⟨4, 3, 0⟩
def P₂ : Point3D := ⟨5, 2, 1⟩

/-- The plane equation we want to prove. -/
def targetPlane : Plane3D := ⟨1, -2, -3, 2⟩

theorem plane_through_three_points :
  P₀.liesOn targetPlane ∧ P₁.liesOn targetPlane ∧ P₂.liesOn targetPlane := by
  sorry


end NUMINAMATH_CALUDE_plane_through_three_points_l1182_118236


namespace NUMINAMATH_CALUDE_a_less_than_neg_one_l1182_118200

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem a_less_than_neg_one (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by sorry

end NUMINAMATH_CALUDE_a_less_than_neg_one_l1182_118200


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1182_118219

theorem x_minus_y_value (x y : ℝ) (h : x^2 + 6*x + 9 + Real.sqrt (y - 3) = 0) : 
  x - y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1182_118219


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1182_118263

/-- Given an arithmetic sequence {a_n} where S₁₀ = 4, prove a₃ + a₈ = 4/5 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- Definition of sum for arithmetic sequence
  (∀ i j k, a i - a j = a j - a k) →    -- Definition of arithmetic sequence
  S 10 = 4 →                            -- Given condition
  a 3 + a 8 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1182_118263


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1182_118278

/-- Given a > 0 and the terminal side of angle α passes through point P(-3a, 4a),
    prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos_alpha (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : ∃ (t : ℝ), t > 0 ∧ -3 * a = t * Real.cos α ∧ 4 * a = t * Real.sin α) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1182_118278


namespace NUMINAMATH_CALUDE_simplify_expression_l1182_118284

theorem simplify_expression (a b : ℝ) : (22*a + 60*b) + (10*a + 29*b) - (9*a + 50*b) = 23*a + 39*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1182_118284


namespace NUMINAMATH_CALUDE_equation_solution_l1182_118286

theorem equation_solution : 
  ∀ x : ℝ, x ≠ -3 → 
  ((7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ↔ 
  (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1182_118286


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l1182_118234

theorem percentage_of_hindu_boys (total_boys : ℕ) 
  (muslim_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 850 →
  muslim_percentage = 44 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 153 →
  (total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l1182_118234


namespace NUMINAMATH_CALUDE_unique_solution_is_one_l1182_118206

/-- A function satisfying f(x)f(y) = f(x-y) for all x and y, and is nonzero at some point -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (∀ x y, f x * f y = f (x - y)) ∧ (∃ x, f x ≠ 0)

/-- The constant function 1 is the unique solution to the functional equation -/
theorem unique_solution_is_one :
  ∀ f : ℝ → ℝ, FunctionalEquation f → (∀ x, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_l1182_118206


namespace NUMINAMATH_CALUDE_xy_power_equality_l1182_118209

theorem xy_power_equality (x y : ℕ) (h : x ≠ y) :
  x ^ y = y ^ x ↔ (x = 2 ∧ y = 4) ∨ (x = 4 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_xy_power_equality_l1182_118209


namespace NUMINAMATH_CALUDE_area_of_B_l1182_118237

-- Define the set A
def A : Set (ℝ × ℝ) := {p | p.1 + p.2 ≤ 1 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

-- Define the transformation function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

-- Define the set B
def B : Set (ℝ × ℝ) := f '' A

-- State the theorem
theorem area_of_B : MeasureTheory.volume B = 1 := by sorry

end NUMINAMATH_CALUDE_area_of_B_l1182_118237


namespace NUMINAMATH_CALUDE_range_of_b_l1182_118246

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem range_of_b (a : ℝ) (h : a = -3) :
  (∀ b : ℝ, (A ∩ B a b).Nonempty) → ∀ b : ℝ, b > -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1182_118246


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1182_118243

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1182_118243


namespace NUMINAMATH_CALUDE_polynomial_product_l1182_118203

-- Define the polynomials
def P (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x
def Q (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3

-- State the theorem
theorem polynomial_product :
  ∀ x : ℝ, P x * Q x = 4 * x^7 - 2 * x^6 - 6 * x^5 + 9 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_l1182_118203


namespace NUMINAMATH_CALUDE_consecutive_binomial_coefficients_l1182_118262

theorem consecutive_binomial_coefficients (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 2 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 3 / 4 →
  n + k = 47 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_coefficients_l1182_118262


namespace NUMINAMATH_CALUDE_product_equals_57_over_168_l1182_118275

def product : ℚ :=
  (2^3 - 1) / (2^3 + 1) *
  (3^3 - 1) / (3^3 + 1) *
  (4^3 - 1) / (4^3 + 1) *
  (5^3 - 1) / (5^3 + 1) *
  (6^3 - 1) / (6^3 + 1) *
  (7^3 - 1) / (7^3 + 1)

theorem product_equals_57_over_168 : product = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_57_over_168_l1182_118275


namespace NUMINAMATH_CALUDE_boat_downstream_speed_l1182_118238

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a boat given its speed in still water and upstream -/
def downstreamSpeed (b : BoatSpeed) : ℝ :=
  2 * b.stillWater - b.upstream

/-- Theorem stating that a boat with 11 km/hr speed in still water and 7 km/hr upstream 
    will have a downstream speed of 15 km/hr -/
theorem boat_downstream_speed :
  let b : BoatSpeed := { stillWater := 11, upstream := 7 }
  downstreamSpeed b = 15 := by sorry

end NUMINAMATH_CALUDE_boat_downstream_speed_l1182_118238


namespace NUMINAMATH_CALUDE_quadratic_functions_intersect_l1182_118265

/-- A quadratic function of the form f(x) = x^2 + px + q where p + q = 2002 -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : p + q = 2002

/-- The theorem stating that all quadratic functions satisfying the condition
    intersect at the point (1, 2003) -/
theorem quadratic_functions_intersect (f : QuadraticFunction) :
  f.p + f.q^2 + f.p + f.q = 2003 := by
  sorry

#check quadratic_functions_intersect

end NUMINAMATH_CALUDE_quadratic_functions_intersect_l1182_118265


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l1182_118251

theorem system_of_equations_solutions :
  -- System 1
  (∃ (x y : ℚ), x = 1 - y ∧ 3 * x + y = 1 ∧ x = 0 ∧ y = 1) ∧
  -- System 2
  (∃ (x y : ℚ), 3 * x + y = 18 ∧ 2 * x - y = -11 ∧ x = 7/5 ∧ y = 69/5) :=
by sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l1182_118251


namespace NUMINAMATH_CALUDE_square_field_diagonal_l1182_118245

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 450 → diagonal = 30 → diagonal^2 = 2 * area :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l1182_118245


namespace NUMINAMATH_CALUDE_gcd_power_two_l1182_118248

theorem gcd_power_two : 
  Nat.gcd (2^2100 - 1) (2^2091 + 31) = Nat.gcd (2^2091 + 31) 511 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_l1182_118248


namespace NUMINAMATH_CALUDE_polynomial_equation_l1182_118233

-- Define polynomials over real numbers
variable (x : ℝ)

-- Define f(x) and h(x) as polynomials
def f (x : ℝ) : ℝ := x^4 + 2*x^3 - x^2 - 4*x + 1
def h (x : ℝ) : ℝ := -x^4 - 2*x^3 + 4*x^2 + 9*x - 5

-- State the theorem
theorem polynomial_equation :
  f x + h x = 3*x^2 + 5*x - 4 := by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l1182_118233


namespace NUMINAMATH_CALUDE_unique_solution_l1182_118252

theorem unique_solution (x y z : ℝ) 
  (hx : x > 5) (hy : y > 5) (hz : z > 5)
  (h : ((x + 3)^2 / (y + z - 3)) + ((y + 5)^2 / (z + x - 5)) + ((z + 7)^2 / (x + y - 7)) = 45) :
  x = 15 ∧ y = 15 ∧ z = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1182_118252


namespace NUMINAMATH_CALUDE_fans_with_all_items_l1182_118297

def stadium_capacity : ℕ := 4500
def tshirt_interval : ℕ := 60
def hat_interval : ℕ := 45
def keychain_interval : ℕ := 75

theorem fans_with_all_items :
  (stadium_capacity / (Nat.lcm tshirt_interval (Nat.lcm hat_interval keychain_interval))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l1182_118297


namespace NUMINAMATH_CALUDE_max_large_chips_l1182_118268

theorem max_large_chips (total : ℕ) (is_prime : ℕ → Prop) : 
  total = 54 →
  ∃ (small large prime : ℕ), 
    is_prime prime ∧
    small + large = total ∧
    small = large + prime ∧
    ∀ (l : ℕ), (∃ (s p : ℕ), is_prime p ∧ s + l = total ∧ s = l + p) → l ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_large_chips_l1182_118268


namespace NUMINAMATH_CALUDE_unique_three_digit_factorial_product_l1182_118272

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digits_of (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (a, b, c)

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_three_digit_factorial_product :
  ∃! n : ℕ, is_three_digit n ∧
    let (a, b, c) := digits_of n
    2 * n = 3 * (factorial a * factorial b * factorial c) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_factorial_product_l1182_118272


namespace NUMINAMATH_CALUDE_tangerines_per_day_l1182_118224

theorem tangerines_per_day 
  (initial : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : initial > remaining) 
  (h2 : days > 0) : 
  (initial - remaining) / days = (initial - remaining) / days :=
by sorry

end NUMINAMATH_CALUDE_tangerines_per_day_l1182_118224


namespace NUMINAMATH_CALUDE_ryan_sandwich_slices_l1182_118227

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of slices of bread needed for one sandwich -/
def slices_per_sandwich : ℕ := 3

/-- The total number of slices needed for all sandwiches -/
def total_slices : ℕ := num_sandwiches * slices_per_sandwich

theorem ryan_sandwich_slices : total_slices = 15 := by
  sorry

end NUMINAMATH_CALUDE_ryan_sandwich_slices_l1182_118227


namespace NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l1182_118282

theorem sum_of_three_smallest_solutions : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x ≥ x₁) ∧
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x = x₁ ∨ x ≥ x₂) ∧
    (∀ x : ℝ, x > 0 → x - ⌊x⌋ = 1 / ⌊x⌋ + 1 / ⌊x⌋^2 → x = x₁ ∨ x = x₂ ∨ x ≥ x₃) ∧
    x₁ - ⌊x₁⌋ = 1 / ⌊x₁⌋ + 1 / ⌊x₁⌋^2 ∧
    x₂ - ⌊x₂⌋ = 1 / ⌊x₂⌋ + 1 / ⌊x₂⌋^2 ∧
    x₃ - ⌊x₃⌋ = 1 / ⌊x₃⌋ + 1 / ⌊x₃⌋^2 ∧
    x₁ + x₂ + x₃ = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l1182_118282


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1182_118293

theorem system_of_equations_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_eq1 : x^2 + x*y + y^2 = 108)
  (h_eq2 : y^2 + y*z + z^2 = 49)
  (h_eq3 : z^2 + x*z + x^2 = 157) :
  x*y + y*z + x*z = 104 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1182_118293


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1182_118242

theorem inequality_solution_set (x : ℝ) :
  -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1182_118242


namespace NUMINAMATH_CALUDE_value_of_r_l1182_118258

theorem value_of_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_r_l1182_118258


namespace NUMINAMATH_CALUDE_binomial_10_3_l1182_118235

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l1182_118235


namespace NUMINAMATH_CALUDE_sphere_radii_difference_l1182_118253

theorem sphere_radii_difference (r₁ r₂ : ℝ) 
  (h₁ : 4 * π * (r₁^2 - r₂^2) = 48 * π) 
  (h₂ : 2 * π * r₁ + 2 * π * r₂ = 12 * π) : 
  |r₁ - r₂| = 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_difference_l1182_118253


namespace NUMINAMATH_CALUDE_hexagonal_tiling_chromatic_number_l1182_118205

/-- A type representing colors -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a hexagonal tile in the plane -/
structure HexTile :=
  (id : ℕ)

/-- A function type that assigns colors to hexagonal tiles -/
def Coloring := HexTile → Color

/-- Predicate to check if two hexagonal tiles are adjacent (share a side) -/
def adjacent : HexTile → HexTile → Prop := sorry

/-- Predicate to check if a coloring is valid (no adjacent tiles have the same color) -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ h1 h2, adjacent h1 h2 → c h1 ≠ c h2

/-- The main theorem: The minimum number of colors needed is 3 -/
theorem hexagonal_tiling_chromatic_number :
  (∃ c : Coloring, valid_coloring c) ∧
  (∀ c : Coloring, valid_coloring c → (Set.range c).ncard ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_hexagonal_tiling_chromatic_number_l1182_118205


namespace NUMINAMATH_CALUDE_barrel_leak_percentage_l1182_118214

theorem barrel_leak_percentage (initial_volume : ℝ) (remaining_volume : ℝ) : 
  initial_volume = 220 →
  remaining_volume = 198 →
  (initial_volume - remaining_volume) / initial_volume * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_barrel_leak_percentage_l1182_118214


namespace NUMINAMATH_CALUDE_circle_intersection_exists_l1182_118239

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (A B : Point)
variable (S : Circle)
variable (α : ℝ)

-- Define the intersection angle between two circles
def intersectionAngle (c1 c2 : Circle) : ℝ := sorry

-- Define a function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Theorem statement
theorem circle_intersection_exists :
  ∃ (C : Circle), isOnCircle A C ∧ isOnCircle B C ∧ intersectionAngle C S = α := by sorry

end NUMINAMATH_CALUDE_circle_intersection_exists_l1182_118239


namespace NUMINAMATH_CALUDE_different_course_selections_l1182_118283

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem different_course_selections (total_courses : ℕ) (courses_per_person : ℕ) : 
  total_courses = 4 → courses_per_person = 2 →
  (choose total_courses courses_per_person * choose total_courses courses_per_person) - 
  (choose total_courses courses_per_person) = 30 := by
  sorry

end NUMINAMATH_CALUDE_different_course_selections_l1182_118283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1182_118213

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_property (b : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence b)
  (h_incr : ∀ n : ℕ, b n < b (n + 1))
  (h_prod : b 4 * b 7 = 24) :
  b 3 * b 8 = 200 / 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1182_118213


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l1182_118222

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b → y = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l1182_118222


namespace NUMINAMATH_CALUDE_theater_audience_l1182_118288

/-- Proves the number of children in the audience given theater conditions -/
theorem theater_audience (total_seats : ℕ) (adult_price child_price : ℚ) (total_income : ℚ) 
  (h_seats : total_seats = 200)
  (h_adult_price : adult_price = 3)
  (h_child_price : child_price = (3/2))
  (h_total_income : total_income = 510) :
  ∃ (adults children : ℕ), 
    adults + children = total_seats ∧ 
    adult_price * adults + child_price * children = total_income ∧
    children = 60 := by
  sorry

end NUMINAMATH_CALUDE_theater_audience_l1182_118288


namespace NUMINAMATH_CALUDE_aunt_marge_candy_count_l1182_118291

/-- The number of candy pieces each child receives -/
structure CandyDistribution where
  kate : ℕ
  bill : ℕ
  robert : ℕ
  mary : ℕ

/-- The conditions of Aunt Marge's candy distribution -/
def is_valid_distribution (d : CandyDistribution) : Prop :=
  d.robert = d.kate + 2 ∧
  d.bill = d.mary - 6 ∧
  d.mary = d.robert + 2 ∧
  d.kate = d.bill + 2 ∧
  d.kate = 4

/-- The theorem stating that Aunt Marge has 24 pieces of candy in total -/
theorem aunt_marge_candy_count (d : CandyDistribution) 
  (h : is_valid_distribution d) : 
  d.kate + d.bill + d.robert + d.mary = 24 := by
  sorry

#check aunt_marge_candy_count

end NUMINAMATH_CALUDE_aunt_marge_candy_count_l1182_118291


namespace NUMINAMATH_CALUDE_max_a_value_l1182_118223

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- The condition for m -/
def MCondition (m a : ℚ) : Prop := 1/2 < m ∧ m < a

/-- The main theorem -/
theorem max_a_value :
  ∃ (a : ℚ), a = 75/149 ∧
  (∀ (m : ℚ), MCondition m a →
    ∀ (x y : ℤ), 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y) ∧
  (∀ (a' : ℚ), a < a' →
    ∃ (m : ℚ), MCondition m a' ∧
    ∃ (x y : ℤ), 0 < x ∧ x ≤ 150 ∧ LatticePoint x y ∧ LineEquation m x y) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1182_118223


namespace NUMINAMATH_CALUDE_unique_intersection_l1182_118260

/-- Parabola C defined by x²=4y -/
def parabola_C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line MH passing through points M(t,0) and H(2t,t²) -/
def line_MH (t x y : ℝ) : Prop := y = t*(x - t)

/-- Point H on parabola C -/
def point_H (t : ℝ) : ℝ × ℝ := (2*t, t^2)

theorem unique_intersection (t : ℝ) (h : t ≠ 0) :
  ∀ x y : ℝ, parabola_C x y ∧ line_MH t x y → (x, y) = point_H t :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_l1182_118260


namespace NUMINAMATH_CALUDE_jogger_distance_l1182_118274

theorem jogger_distance (actual_speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  actual_speed = 12 →
  faster_speed = 16 →
  extra_distance = 10 →
  (∃ time : ℝ, time > 0 ∧ faster_speed * time = actual_speed * time + extra_distance) →
  actual_speed * (extra_distance / (faster_speed - actual_speed)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_jogger_distance_l1182_118274


namespace NUMINAMATH_CALUDE_lemon_problem_l1182_118299

theorem lemon_problem (levi jayden eli ian : ℕ) : 
  levi = 5 →
  jayden > levi →
  jayden * 3 = eli →
  eli * 2 = ian →
  levi + jayden + eli + ian = 115 →
  jayden - levi = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_lemon_problem_l1182_118299


namespace NUMINAMATH_CALUDE_charge_difference_l1182_118211

/-- Represents the pricing scheme of a psychologist -/
structure PricingScheme where
  firstHourCharge : ℝ
  additionalHourCharge : ℝ
  fiveHourTotal : ℝ
  twoHourTotal : ℝ

/-- Theorem stating the difference in charges for a specific pricing scheme -/
theorem charge_difference (p : PricingScheme) 
  (h1 : p.firstHourCharge > p.additionalHourCharge)
  (h2 : p.firstHourCharge + 4 * p.additionalHourCharge = p.fiveHourTotal)
  (h3 : p.firstHourCharge + p.additionalHourCharge = p.twoHourTotal)
  (h4 : p.fiveHourTotal = 350)
  (h5 : p.twoHourTotal = 161) : 
  p.firstHourCharge - p.additionalHourCharge = 35 := by
  sorry

end NUMINAMATH_CALUDE_charge_difference_l1182_118211


namespace NUMINAMATH_CALUDE_carol_initial_blocks_l1182_118244

/-- The number of blocks Carol started with -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Carol lost -/
def lost_blocks : ℕ := 25

/-- The number of blocks Carol ended with -/
def final_blocks : ℕ := 17

/-- Theorem stating that Carol started with 42 blocks -/
theorem carol_initial_blocks : initial_blocks = 42 := by sorry

end NUMINAMATH_CALUDE_carol_initial_blocks_l1182_118244


namespace NUMINAMATH_CALUDE_minyoung_line_size_l1182_118210

/-- Represents a line of people ordered by height -/
structure HeightLine where
  people : ℕ
  tallestToShortest : Fin people → Fin people

/-- A person's position from the tallest in the line -/
def positionFromTallest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.tallestToShortest person + 1

/-- A person's position from the shortest in the line -/
def positionFromShortest (line : HeightLine) (person : Fin line.people) : ℕ :=
  line.people - line.tallestToShortest person

theorem minyoung_line_size
  (line : HeightLine)
  (minyoung : Fin line.people)
  (h1 : positionFromTallest line minyoung = 2)
  (h2 : positionFromShortest line minyoung = 4) :
  line.people = 5 := by
  sorry

end NUMINAMATH_CALUDE_minyoung_line_size_l1182_118210


namespace NUMINAMATH_CALUDE_volume_P4_l1182_118250

/-- Recursive definition of the volume of Pᵢ --/
def volume (i : ℕ) : ℚ :=
  match i with
  | 0 => 1
  | n + 1 => volume n + (4^n * (1 / 27))

/-- Theorem stating the volume of P₄ --/
theorem volume_P4 : volume 4 = 367 / 27 := by
  sorry

#eval volume 4

end NUMINAMATH_CALUDE_volume_P4_l1182_118250


namespace NUMINAMATH_CALUDE_complex_modulus_one_l1182_118229

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l1182_118229


namespace NUMINAMATH_CALUDE_graph_shift_l1182_118298

/-- Given a function f and real numbers a and b, 
    the graph of y = f(x - a) + b is obtained by shifting 
    the graph of y = f(x) a units right and b units up. -/
theorem graph_shift (f : ℝ → ℝ) (a b : ℝ) :
  ∀ x y, y = f (x - a) + b ↔ y - b = f (x - a) :=
by sorry

end NUMINAMATH_CALUDE_graph_shift_l1182_118298


namespace NUMINAMATH_CALUDE_line_ellipse_intersections_l1182_118201

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The ellipse equation (x-1)^2 + 4y^2 = 4 -/
def ellipse_eq (x y : ℝ) : Prop := (x - 1)^2 + 4 * y^2 = 4

/-- The number of intersections between the line and the ellipse -/
def num_intersections : ℕ := 0

/-- Theorem stating that the number of intersections between the line and the ellipse is 0 -/
theorem line_ellipse_intersections :
  ∀ x y : ℝ, line_eq x y ∧ ellipse_eq x y → num_intersections = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersections_l1182_118201


namespace NUMINAMATH_CALUDE_gcd_problem_l1182_118212

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8531) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1182_118212


namespace NUMINAMATH_CALUDE_elephant_ratio_is_three_l1182_118215

/-- The number of elephants at We Preserve For Future park -/
def we_preserve_elephants : ℕ := 70

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := 280

/-- The ratio of elephants at Gestures For Good park to We Preserve For Future park -/
def elephant_ratio : ℚ := (total_elephants - we_preserve_elephants) / we_preserve_elephants

theorem elephant_ratio_is_three : elephant_ratio = 3 := by
  sorry

end NUMINAMATH_CALUDE_elephant_ratio_is_three_l1182_118215


namespace NUMINAMATH_CALUDE_third_grade_students_l1182_118289

theorem third_grade_students (num_buses : ℕ) (seats_per_bus : ℕ) (empty_seats_per_bus : ℕ) :
  num_buses = 18 →
  seats_per_bus = 15 →
  empty_seats_per_bus = 3 →
  num_buses * (seats_per_bus - empty_seats_per_bus) = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_third_grade_students_l1182_118289


namespace NUMINAMATH_CALUDE_red_balls_count_l1182_118276

theorem red_balls_count (x : ℕ) (h : (4 : ℝ) / (x + 4) = (1 : ℝ) / 5) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1182_118276


namespace NUMINAMATH_CALUDE_units_digit_of_5_to_10_l1182_118295

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_5_to_10 : unitsDigit (5^10) = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_5_to_10_l1182_118295


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l1182_118208

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (ha : a = 9) (hb : b = 11) (hθ : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*Real.cos θ ∧ c = Real.sqrt (202 + 99 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l1182_118208
