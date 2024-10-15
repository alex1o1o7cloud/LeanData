import Mathlib

namespace NUMINAMATH_CALUDE_initial_value_proof_l951_95137

theorem initial_value_proof (increase_rate : ℝ) (final_value : ℝ) (years : ℕ) : 
  increase_rate = 1/8 →
  years = 2 →
  final_value = 8100 →
  final_value = 6400 * (1 + increase_rate)^years →
  6400 = 6400 := by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l951_95137


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_l951_95190

theorem binomial_coefficient_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_l951_95190


namespace NUMINAMATH_CALUDE_positive_integer_solution_iff_n_eq_three_l951_95110

theorem positive_integer_solution_iff_n_eq_three (n : ℕ) :
  (∃ (x y z : ℕ+), x^2 + y^2 + z^2 = n * x * y * z) ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_positive_integer_solution_iff_n_eq_three_l951_95110


namespace NUMINAMATH_CALUDE_factorial_division_l951_95131

theorem factorial_division : (Nat.factorial 8) / (Nat.factorial (8 - 2)) = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l951_95131


namespace NUMINAMATH_CALUDE_negation_equivalence_l951_95149

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l951_95149


namespace NUMINAMATH_CALUDE_modulus_of_z_l951_95115

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l951_95115


namespace NUMINAMATH_CALUDE_solution_implies_a_equals_six_l951_95141

theorem solution_implies_a_equals_six :
  ∀ a : ℝ, (2 * 1 + 5 = 1 + a) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_equals_six_l951_95141


namespace NUMINAMATH_CALUDE_smallest_n_value_l951_95187

theorem smallest_n_value (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ (21 * N)) : 
  (∀ m : ℕ, m > 70 ∧ 70 ∣ (21 * m) → m ≥ N) → N = 80 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l951_95187


namespace NUMINAMATH_CALUDE_inequality_proof_l951_95155

theorem inequality_proof (a : ℝ) (ha : a > 0) : 2 * a / (1 + a^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l951_95155


namespace NUMINAMATH_CALUDE_passenger_gate_probability_l951_95164

def num_gates : ℕ := 15
def distance_between_gates : ℕ := 90
def max_walking_distance : ℕ := 360

theorem passenger_gate_probability : 
  let total_possibilities := num_gates * (num_gates - 1)
  let valid_possibilities := (
    2 * (4 + 5 + 6 + 7) +  -- Gates 1,2,3,4 and 12,13,14,15
    4 * 8 +                -- Gates 5,6,10,11
    3 * 8                  -- Gates 7,8,9
  )
  (valid_possibilities : ℚ) / total_possibilities = 10 / 21 :=
sorry

end NUMINAMATH_CALUDE_passenger_gate_probability_l951_95164


namespace NUMINAMATH_CALUDE_angela_figures_theorem_l951_95133

def calculate_remaining_figures (initial : ℕ) : ℕ :=
  let increased := initial + (initial * 15 / 100)
  let after_selling := increased - (increased / 4)
  let after_giving_to_daughter := after_selling - (after_selling / 3)
  let final := after_giving_to_daughter - (after_giving_to_daughter * 20 / 100)
  final

theorem angela_figures_theorem :
  calculate_remaining_figures 24 = 12 := by
  sorry

end NUMINAMATH_CALUDE_angela_figures_theorem_l951_95133


namespace NUMINAMATH_CALUDE_triangle_db_length_l951_95116

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (right_angle_ABC : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0)
  (right_angle_ADB : (B.1 - A.1) * (D.1 - A.1) + (B.2 - A.2) * (D.2 - A.2) = 0)
  (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 19)
  (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4)

-- Theorem statement
theorem triangle_db_length (t : Triangle) : 
  Real.sqrt ((t.B.1 - t.D.1)^2 + (t.B.2 - t.D.2)^2) = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_db_length_l951_95116


namespace NUMINAMATH_CALUDE_banana_sharing_l951_95150

theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  total_bananas = num_friends * bananas_per_friend →
  bananas_per_friend = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_banana_sharing_l951_95150


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_l951_95143

theorem quadratic_roots_distinct (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - (k+3)*x₁ + k = 0) ∧ 
  (x₂^2 - (k+3)*x₂ + k = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_l951_95143


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l951_95166

theorem intersection_point_x_coordinate 
  (f : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.log x) 
  (P Q : ℝ × ℝ) 
  (hP : P = (2, Real.log 2)) 
  (hQ : Q = (500, Real.log 500)) 
  (hPQ : P.1 < Q.1) 
  (R : ℝ × ℝ) 
  (hR : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) 
  (T : ℝ × ℝ) 
  (hT : T.2 = R.2 ∧ f T.1 = T.2 ∧ T.1 ≠ R.1) : 
  T.1 = Real.sqrt 1000 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l951_95166


namespace NUMINAMATH_CALUDE_xy_squared_value_l951_95167

theorem xy_squared_value (x y : ℤ) (h : y^2 + 2*x^2*y^2 = 20*x^2 + 412) : 2*x*y^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_value_l951_95167


namespace NUMINAMATH_CALUDE_square_of_99_9_l951_95127

theorem square_of_99_9 : (99.9 : ℝ)^2 = 10000 - 20 + 0.01 := by
  sorry

end NUMINAMATH_CALUDE_square_of_99_9_l951_95127


namespace NUMINAMATH_CALUDE_common_chord_length_l951_95183

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l951_95183


namespace NUMINAMATH_CALUDE_triangle_properties_l951_95184

-- Define an acute triangle ABC
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  acute : angle_A > 0 ∧ angle_A < Real.pi/2 ∧
          angle_B > 0 ∧ angle_B < Real.pi/2 ∧
          angle_C > 0 ∧ angle_C < Real.pi/2

-- State the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = t.a * t.b)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.angle_C = (3 * Real.sqrt 3) / 2) :
  t.angle_C = Real.pi/3 ∧ t.a + t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l951_95184


namespace NUMINAMATH_CALUDE_square_division_area_l951_95140

theorem square_division_area : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≠ 1 ∧ 
  x^2 = 24 + y^2 ∧
  x^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_division_area_l951_95140


namespace NUMINAMATH_CALUDE_original_amount_is_48_l951_95139

/-- Proves that the original amount is 48 rupees given the described transactions --/
theorem original_amount_is_48 (x : ℚ) : 
  ((2/3 * ((2/3 * x + 10) + 20)) = x) → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_is_48_l951_95139


namespace NUMINAMATH_CALUDE_haley_necklace_count_l951_95125

/-- The number of necklaces Haley, Jason, and Josh have satisfy the given conditions -/
def NecklaceProblem (h j q : ℕ) : Prop :=
  (h = j + 5) ∧ (q = j / 2) ∧ (h = q + 15)

/-- Theorem: If the necklace counts satisfy the given conditions, then Haley has 25 necklaces -/
theorem haley_necklace_count
  (h j q : ℕ) (hcond : NecklaceProblem h j q) : h = 25 := by
  sorry

end NUMINAMATH_CALUDE_haley_necklace_count_l951_95125


namespace NUMINAMATH_CALUDE_modified_cube_properties_l951_95123

/-- Represents a cube with removals as described in the problem -/
structure ModifiedCube where
  side_length : ℕ
  small_cube_size : ℕ
  center_removal_size : ℕ
  unit_removal : Bool

/-- Calculates the remaining volume after removals -/
def remaining_volume (c : ModifiedCube) : ℕ := sorry

/-- Calculates the surface area after removals -/
def surface_area (c : ModifiedCube) : ℕ := sorry

/-- The main theorem stating the properties of the modified cube -/
theorem modified_cube_properties :
  let c : ModifiedCube := {
    side_length := 12,
    small_cube_size := 2,
    center_removal_size := 2,
    unit_removal := true
  }
  remaining_volume c = 1463 ∧ surface_area c = 4598 := by sorry

end NUMINAMATH_CALUDE_modified_cube_properties_l951_95123


namespace NUMINAMATH_CALUDE_count_non_zero_area_triangles_l951_95109

/-- The total number of dots in the grid -/
def total_dots : ℕ := 17

/-- The number of collinear dots in each direction (horizontal and vertical) -/
def collinear_dots : ℕ := 9

/-- The number of ways to choose 3 dots from the total dots -/
def total_combinations : ℕ := Nat.choose total_dots 3

/-- The number of ways to choose 3 collinear dots -/
def collinear_combinations : ℕ := Nat.choose collinear_dots 3

/-- The number of lines with collinear dots (horizontal and vertical) -/
def collinear_lines : ℕ := 2

/-- The number of triangles with non-zero area -/
def non_zero_area_triangles : ℕ := total_combinations - collinear_lines * collinear_combinations

theorem count_non_zero_area_triangles : non_zero_area_triangles = 512 := by
  sorry

end NUMINAMATH_CALUDE_count_non_zero_area_triangles_l951_95109


namespace NUMINAMATH_CALUDE_race_solution_l951_95100

/-- Race between A and B from M to N and back -/
structure Race where
  distance : ℝ  -- Distance between M and N
  time_A : ℝ    -- Time taken by A
  time_B : ℝ    -- Time taken by B

/-- Conditions of the race -/
def race_conditions (r : Race) : Prop :=
  -- A reaches N sooner than B
  r.time_A < r.time_B
  -- A meets B 100 meters before N on the way back
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (2 * r.distance - 100)
  -- A arrives at M 4 minutes earlier than B
  ∧ r.time_B = r.time_A + 4
  -- If A turns around at M, they meet B at 1/5 of the M to N distance
  ∧ ∃ t : ℝ, t < r.time_A ∧ t * (r.distance / r.time_A) = (1/5) * r.distance

/-- The theorem to be proved -/
theorem race_solution :
  ∃ r : Race, race_conditions r ∧ r.distance = 1000 ∧ r.time_A = 18 ∧ r.time_B = 22 := by
  sorry

end NUMINAMATH_CALUDE_race_solution_l951_95100


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l951_95144

theorem tangent_sum_simplification :
  (Real.tan (30 * π / 180) + Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180)) / Real.cos (20 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l951_95144


namespace NUMINAMATH_CALUDE_function_f_properties_l951_95119

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂) ∧
  (∀ x y, f (x + y) = f x * f y)

/-- Theorem stating the properties of the function f -/
theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  f 0 = 1 ∧ ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_f_properties_l951_95119


namespace NUMINAMATH_CALUDE_sequence_bound_l951_95192

theorem sequence_bound (x : ℕ → ℝ) (b : ℝ) : 
  (∀ n : ℕ, x (n + 1) = x n ^ 2 - 4 * x n) →
  (∀ x₁ : ℝ, x₁ ≠ 0 → ∃ k : ℕ, x k ≥ b) →
  b = (3 + Real.sqrt 21) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_bound_l951_95192


namespace NUMINAMATH_CALUDE_grid_lines_formula_l951_95103

/-- The number of straight lines needed to draw an n × n square grid -/
def grid_lines (n : ℕ) : ℕ := 2 * (n + 1)

/-- Theorem stating that the number of straight lines needed to draw an n × n square grid is 2(n + 1) -/
theorem grid_lines_formula (n : ℕ) : grid_lines n = 2 * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_grid_lines_formula_l951_95103


namespace NUMINAMATH_CALUDE_baker_cakes_problem_l951_95173

theorem baker_cakes_problem (sold : ℕ) (left : ℕ) (h1 : sold = 41) (h2 : left = 13) :
  sold + left = 54 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_problem_l951_95173


namespace NUMINAMATH_CALUDE_ice_cream_survey_l951_95154

theorem ice_cream_survey (total : ℕ) (strawberry_percent : ℚ) (vanilla_percent : ℚ) (chocolate_percent : ℚ)
  (h_total : total = 500)
  (h_strawberry : strawberry_percent = 46 / 100)
  (h_vanilla : vanilla_percent = 71 / 100)
  (h_chocolate : chocolate_percent = 85 / 100) :
  ∃ (all_three : ℕ), all_three ≥ 10 ∧
    (strawberry_percent * total + vanilla_percent * total + chocolate_percent * total
      = (total - all_three) * 2 + all_three * 3) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_survey_l951_95154


namespace NUMINAMATH_CALUDE_shaded_area_l951_95145

/-- Given a square and a rhombus with shared side, calculates the area of the region inside the square but outside the rhombus -/
theorem shaded_area (square_area rhombus_area : ℝ) : 
  square_area = 25 →
  rhombus_area = 20 →
  ∃ (shaded_area : ℝ), shaded_area = square_area - (rhombus_area * 0.7) ∧ shaded_area = 11 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_l951_95145


namespace NUMINAMATH_CALUDE_parallel_vectors_angle_l951_95130

theorem parallel_vectors_angle (α : Real) : 
  let a : Fin 2 → Real := ![1 - Real.cos α, Real.sqrt 3]
  let b : Fin 2 → Real := ![Real.sin α, 3]
  (∀ (i j : Fin 2), a i * b j = a j * b i) →  -- parallel condition
  0 < α → α < Real.pi / 2 →                   -- acute angle condition
  α = Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_angle_l951_95130


namespace NUMINAMATH_CALUDE_daily_income_ratio_l951_95179

/-- The ratio of daily income in a business where:
    - Initial income on day 1 is 3
    - Income on day 15 is 36
    - Each day's income is a multiple of the previous day's income
-/
theorem daily_income_ratio : ∃ (r : ℝ), 
  r > 0 ∧ 
  3 * r^14 = 36 ∧ 
  r = 2^(1/7) * 3^(1/14) := by
  sorry

end NUMINAMATH_CALUDE_daily_income_ratio_l951_95179


namespace NUMINAMATH_CALUDE_min_side_length_is_optimal_l951_95177

/-- The minimum side length of a square piece of land with an area of at least 400 square feet -/
def min_side_length : ℝ := 20

/-- The area of the square land is at least 400 square feet -/
axiom area_constraint : min_side_length ^ 2 ≥ 400

/-- The minimum side length is optimal -/
theorem min_side_length_is_optimal :
  ∀ s : ℝ, s ^ 2 ≥ 400 → s ≥ min_side_length :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_is_optimal_l951_95177


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l951_95176

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 5 < 4 ∧ (3 * x + 1) / 2 ≥ 2 * x - 1}
  S = {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l951_95176


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l951_95196

def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4/3 ≤ a ∧ a ≤ 2 := by sorry

theorem intersection_condition (a : ℝ) : A ∩ B a = {x | 3 < x ∧ x < 4} ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l951_95196


namespace NUMINAMATH_CALUDE_park_diameter_l951_95146

/-- Given a circular park with a fountain, garden ring, and walking path, 
    prove that the diameter of the outer boundary is 38 feet. -/
theorem park_diameter (fountain_diameter walking_path_width garden_ring_width : ℝ) 
  (h1 : fountain_diameter = 10)
  (h2 : walking_path_width = 6)
  (h3 : garden_ring_width = 8) :
  2 * (fountain_diameter / 2 + garden_ring_width + walking_path_width) = 38 := by
  sorry


end NUMINAMATH_CALUDE_park_diameter_l951_95146


namespace NUMINAMATH_CALUDE_demokhar_lifespan_l951_95198

theorem demokhar_lifespan :
  ∀ (x : ℚ),
  (1 / 4 : ℚ) * x + (1 / 5 : ℚ) * x + (1 / 3 : ℚ) * x + 13 = x →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_demokhar_lifespan_l951_95198


namespace NUMINAMATH_CALUDE_sandy_average_book_price_l951_95117

theorem sandy_average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) : 
  books1 = 65 → 
  books2 = 55 → 
  price1 = 1480 → 
  price2 = 920 → 
  (price1 + price2) / (books1 + books2 : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_sandy_average_book_price_l951_95117


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l951_95156

theorem slope_angle_of_line (x y : ℝ) :
  x + Real.sqrt 3 * y + 5 = 0 →
  Real.arctan (-Real.sqrt 3 / 3) = 150 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l951_95156


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_f_l951_95104

-- Define the function f
noncomputable def f : ℝ → ℤ
| x => if x > -1 then Int.ceil (1 / (x + 1))
       else if x < -1 then Int.floor (1 / (x + 1))
       else 0  -- This value doesn't matter as f is undefined at x = -1

-- Theorem statement
theorem zero_not_in_range_of_f :
  ∀ x : ℝ, x ≠ -1 → f x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_f_l951_95104


namespace NUMINAMATH_CALUDE_min_distance_is_sqrt_2_l951_95178

/-- Two moving lines that intersect at point A -/
structure IntersectingLines where
  a : ℝ
  b : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => a * x + a + b * y + 3 * b = 0
  l₂ : ℝ → ℝ → Prop := λ x y => b * x - 3 * b - a * y + a = 0

/-- The intersection point of the two lines -/
def intersectionPoint (lines : IntersectingLines) : ℝ × ℝ := sorry

/-- The origin point -/
def origin : ℝ × ℝ := (0, 0)

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The minimum value of the length of segment OA is √2 -/
theorem min_distance_is_sqrt_2 (lines : IntersectingLines) :
  ∃ (min_dist : ℝ), ∀ (a b : ℝ),
    let lines' := { a := a, b := b : IntersectingLines }
    min_dist = Real.sqrt 2 ∧
    distance origin (intersectionPoint lines') ≥ min_dist :=
  sorry

end NUMINAMATH_CALUDE_min_distance_is_sqrt_2_l951_95178


namespace NUMINAMATH_CALUDE_seating_arrangements_mod_1000_l951_95134

/-- Represents a seating arrangement of ambassadors and advisors. -/
structure SeatingArrangement where
  ambassador_seats : Finset (Fin 6)
  advisor_seats : Finset (Fin 12)

/-- The set of all valid seating arrangements. -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of valid seating arrangements. -/
def N : ℕ := Finset.card validArrangements

/-- Theorem stating that the number of valid seating arrangements
    is congruent to 520 modulo 1000. -/
theorem seating_arrangements_mod_1000 :
  N % 1000 = 520 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_mod_1000_l951_95134


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l951_95108

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  A.lcm B = 240 → A.val * 6 = B.val * 5 → A.gcd B = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l951_95108


namespace NUMINAMATH_CALUDE_spherical_coordinate_negation_l951_95182

/-- Given a point with rectangular coordinates (-3, 5, -2) and corresponding
    spherical coordinates (r, θ, φ), prove that the point with spherical
    coordinates (r, -θ, φ) has rectangular coordinates (-3, -5, -2). -/
theorem spherical_coordinate_negation (r θ φ : ℝ) :
  (r * Real.sin φ * Real.cos θ = -3 ∧
   r * Real.sin φ * Real.sin θ = 5 ∧
   r * Real.cos φ = -2) →
  (r * Real.sin φ * Real.cos (-θ) = -3 ∧
   r * Real.sin φ * Real.sin (-θ) = -5 ∧
   r * Real.cos φ = -2) := by
  sorry


end NUMINAMATH_CALUDE_spherical_coordinate_negation_l951_95182


namespace NUMINAMATH_CALUDE_expand_expression_l951_95106

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l951_95106


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l951_95180

theorem largest_n_with_unique_k : 
  ∀ n : ℕ+, n ≤ 112 ↔ 
    (∃! k : ℤ, (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l951_95180


namespace NUMINAMATH_CALUDE_concert_friends_count_l951_95105

theorem concert_friends_count : 
  ∀ (P : ℝ), P > 0 → 
  ∃ (F : ℕ), 
    (F : ℝ) * P = ((F + 1 : ℕ) : ℝ) * P * (1 - 0.25) ∧ 
    F = 3 := by
  sorry

end NUMINAMATH_CALUDE_concert_friends_count_l951_95105


namespace NUMINAMATH_CALUDE_adam_has_23_tattoos_l951_95114

/-- Calculates the number of tattoos Adam has given Jason's tattoo configuration -/
def adam_tattoos (jason_arm_tattoos jason_leg_tattoos jason_arms jason_legs : ℕ) : ℕ :=
  2 * (jason_arm_tattoos * jason_arms + jason_leg_tattoos * jason_legs) + 3

/-- Proves that Adam has 23 tattoos given Jason's tattoo configuration -/
theorem adam_has_23_tattoos :
  adam_tattoos 2 3 2 2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_adam_has_23_tattoos_l951_95114


namespace NUMINAMATH_CALUDE_max_profit_theorem_l951_95163

/-- Represents the profit function for a souvenir shop -/
def profit_function (x : ℝ) : ℝ := -20 * x + 3200

/-- Represents the constraint on the number of type A souvenirs -/
def constraint (x : ℝ) : Prop := x ≥ 10

/-- Theorem stating the maximum profit and the number of type A souvenirs that achieves it -/
theorem max_profit_theorem :
  ∃ (x : ℝ), constraint x ∧
  (∀ (y : ℝ), constraint y → profit_function x ≥ profit_function y) ∧
  x = 10 ∧ profit_function x = 3000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l951_95163


namespace NUMINAMATH_CALUDE_sixth_term_seq1_sixth_term_seq2_l951_95129

-- Define the first sequence
def seq1 (n : ℕ) : ℕ := 3 * n

-- Define the second sequence
def seq2 (n : ℕ) : ℕ := n * n

-- Theorem for the first sequence
theorem sixth_term_seq1 : seq1 5 = 15 := by sorry

-- Theorem for the second sequence
theorem sixth_term_seq2 : seq2 6 = 36 := by sorry

end NUMINAMATH_CALUDE_sixth_term_seq1_sixth_term_seq2_l951_95129


namespace NUMINAMATH_CALUDE_smallest_year_after_2000_with_digit_sum_15_l951_95113

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem smallest_year_after_2000_with_digit_sum_15 :
  (∀ y : ℕ, 2000 < y ∧ y < 2049 → sumOfDigits y ≠ 15) ∧ 
  2000 < 2049 ∧ 
  sumOfDigits 2049 = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_year_after_2000_with_digit_sum_15_l951_95113


namespace NUMINAMATH_CALUDE_larger_tv_diagonal_l951_95101

theorem larger_tv_diagonal (d : ℝ) : d > 0 →
  (d^2 / 2) = (17^2 / 2) + 143.5 →
  d = 24 := by
sorry

end NUMINAMATH_CALUDE_larger_tv_diagonal_l951_95101


namespace NUMINAMATH_CALUDE_distance_sum_inequality_l951_95126

theorem distance_sum_inequality (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 5| + |x - 1| < a) ↔ a > 4 := by sorry

end NUMINAMATH_CALUDE_distance_sum_inequality_l951_95126


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l951_95161

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) ∧
    (∀ (m : ℕ), m > n → 
      ¬(∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
        m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12))) ∧
  (∀ (n : ℕ), (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) →
    n ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l951_95161


namespace NUMINAMATH_CALUDE_find_m_value_l951_95157

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l951_95157


namespace NUMINAMATH_CALUDE_minimum_guests_l951_95197

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ 
  ∀ (n : ℕ), n * max_per_guest ≥ total_food → n ≥ min_guests :=
by sorry

end NUMINAMATH_CALUDE_minimum_guests_l951_95197


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l951_95153

theorem largest_mu_inequality : 
  ∃ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 3*c*d) ∧ 
  (∀ (μ' : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 3*c*d) → μ' ≤ μ) ∧ 
  μ = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l951_95153


namespace NUMINAMATH_CALUDE_inequality_solutions_l951_95174

-- Define the solution sets
def solution_set1 : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}
def solution_set2 : Set ℝ := {x : ℝ | x < -2 ∨ x > 3}

-- State the theorem
theorem inequality_solutions :
  (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0 ↔ x ∈ solution_set1) ∧
  (∀ x : ℝ, x - x^2 + 6 < 0 ↔ x ∈ solution_set2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solutions_l951_95174


namespace NUMINAMATH_CALUDE_prob_diff_games_l951_95111

/-- Probability of getting heads on a single toss of the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails on a single toss of the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game A -/
def p_win_game_a : ℚ := 
  4 * (p_heads^3 * p_tails) + p_heads^4

/-- Probability of winning Game B -/
def p_win_game_b : ℚ := 
  (p_heads^2 + p_tails^2)^2

/-- The difference in probabilities between winning Game A and Game B -/
theorem prob_diff_games : p_win_game_a - p_win_game_b = 89/256 := by
  sorry

end NUMINAMATH_CALUDE_prob_diff_games_l951_95111


namespace NUMINAMATH_CALUDE_tangent_line_equation_l951_95186

/-- A line passing through (2,0) and tangent to y = 1/x has equation x + y - 2 = 0 -/
theorem tangent_line_equation : ∃ (k : ℝ),
  (∀ x y : ℝ, y = k * (x - 2) → y = 1 / x → x * x * k - 2 * x * k - 1 = 0) ∧
  (4 * k * k + 4 * k = 0) ∧
  (∀ x y : ℝ, y = k * (x - 2) ↔ x + y - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l951_95186


namespace NUMINAMATH_CALUDE_total_sum_lent_total_sum_lent_proof_l951_95142

/-- Proves that the total sum lent is 2795 rupees given the problem conditions -/
theorem total_sum_lent : ℕ → Prop := fun total_sum =>
  ∃ (first_part second_part : ℕ),
    -- The sum is divided into two parts
    total_sum = first_part + second_part ∧
    -- Interest on first part for 8 years at 3% per annum equals interest on second part for 3 years at 5% per annum
    (first_part * 3 * 8) = (second_part * 5 * 3) ∧
    -- The second part is Rs. 1720
    second_part = 1720 ∧
    -- The total sum lent is 2795 rupees
    total_sum = 2795

/-- The proof of the theorem -/
theorem total_sum_lent_proof : total_sum_lent 2795 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_lent_total_sum_lent_proof_l951_95142


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l951_95181

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem statement
theorem f_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l951_95181


namespace NUMINAMATH_CALUDE_triangle_proof_l951_95107

theorem triangle_proof 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : A < π / 2) 
  (h2 : Real.sin (A - π / 4) = Real.sqrt 2 / 10) 
  (h3 : (1 / 2) * b * c * Real.sin A = 24) 
  (h4 : b = 10) : 
  Real.sin A = 4 / 5 ∧ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_proof_l951_95107


namespace NUMINAMATH_CALUDE_base_8_sum_4321_l951_95102

def base_8_sum (n : ℕ) : ℕ :=
  (n.digits 8).sum

theorem base_8_sum_4321 : base_8_sum 4321 = 9 := by sorry

end NUMINAMATH_CALUDE_base_8_sum_4321_l951_95102


namespace NUMINAMATH_CALUDE_faster_bike_speed_l951_95195

/-- Proves that given two motorbikes traveling the same distance,
    where one bike is faster and takes 1 hour less than the other bike,
    the speed of the faster bike is 60 kmph. -/
theorem faster_bike_speed
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_diff : ℝ)
  (h1 : distance = 960)
  (h2 : speed_fast = 60)
  (h3 : time_diff = 1)
  (h4 : distance / speed_fast + time_diff = distance / (distance / (distance / speed_fast + time_diff))) :
  speed_fast = 60 := by
  sorry

end NUMINAMATH_CALUDE_faster_bike_speed_l951_95195


namespace NUMINAMATH_CALUDE_crayon_production_l951_95193

theorem crayon_production (num_colors : ℕ) (crayons_per_color : ℕ) (boxes_per_hour : ℕ) (hours : ℕ) :
  num_colors = 4 →
  crayons_per_color = 2 →
  boxes_per_hour = 5 →
  hours = 4 →
  (num_colors * crayons_per_color * boxes_per_hour * hours) = 160 :=
by sorry

end NUMINAMATH_CALUDE_crayon_production_l951_95193


namespace NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l951_95124

theorem infinitely_many_primes_6n_plus_5 :
  ∀ k : ℕ, ∃ p : ℕ, p > k ∧ Prime p ∧ ∃ n : ℕ, p = 6 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_6n_plus_5_l951_95124


namespace NUMINAMATH_CALUDE_car_instantaneous_speed_l951_95158

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - t^2 + 2

-- State the theorem
theorem car_instantaneous_speed : 
  (deriv s) 1 = 4 := by sorry

end NUMINAMATH_CALUDE_car_instantaneous_speed_l951_95158


namespace NUMINAMATH_CALUDE_division_equality_l951_95185

theorem division_equality : (203515 : ℕ) / 2015 = 101 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l951_95185


namespace NUMINAMATH_CALUDE_cube_has_twelve_edges_l951_95175

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where
  -- We don't need to specify any fields for this definition

/-- The number of edges in a cube. -/
def num_edges (c : Cube) : ℕ := 12

/-- Theorem: A cube has 12 edges. -/
theorem cube_has_twelve_edges (c : Cube) : num_edges c = 12 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_twelve_edges_l951_95175


namespace NUMINAMATH_CALUDE_xyz_equation_solutions_l951_95112

theorem xyz_equation_solutions (n : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃! k : ℕ, k = 3 * (n + 1) ∧
  ∃ S : Finset (ℕ × ℕ × ℕ),
    S.card = k ∧
    ∀ (x y z : ℕ), (x, y, z) ∈ S ↔ 
      x > 0 ∧ y > 0 ∧ z > 0 ∧ 
      x * y * z = p ^ (n : ℕ) * (x + y + z) :=
sorry

end NUMINAMATH_CALUDE_xyz_equation_solutions_l951_95112


namespace NUMINAMATH_CALUDE_jerome_bicycle_trip_distance_l951_95188

/-- The total distance of Jerome's bicycle trip -/
def total_distance (daily_distance : ℕ) (num_days : ℕ) (last_day_distance : ℕ) : ℕ :=
  daily_distance * num_days + last_day_distance

/-- Theorem stating that Jerome's bicycle trip is 150 miles long -/
theorem jerome_bicycle_trip_distance :
  total_distance 12 12 6 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jerome_bicycle_trip_distance_l951_95188


namespace NUMINAMATH_CALUDE_floor_of_3_2_l951_95189

theorem floor_of_3_2 : ⌊(3.2 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_2_l951_95189


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_power_10_l951_95121

theorem last_two_digits_of_7_power_10 : 7^10 ≡ 49 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_power_10_l951_95121


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l951_95122

def U : Finset ℕ := {2, 0, 1, 5}
def A : Finset ℕ := {0, 2}

theorem complement_of_A_in_U :
  (U \ A) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l951_95122


namespace NUMINAMATH_CALUDE_octopus_equality_month_l951_95138

theorem octopus_equality_month : 
  (∀ k : ℕ, k < 4 → 3^(k + 1) ≠ 15 * 5^k) ∧ 
  3^(4 + 1) = 15 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_octopus_equality_month_l951_95138


namespace NUMINAMATH_CALUDE_gcd_of_four_numbers_l951_95135

theorem gcd_of_four_numbers : Nat.gcd 546 (Nat.gcd 1288 (Nat.gcd 3042 5535)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_four_numbers_l951_95135


namespace NUMINAMATH_CALUDE_oblique_line_plane_angle_range_l951_95120

-- Define the angle between an oblique line and a plane
def angle_oblique_line_plane (θ : Real) : Prop := 
  θ > 0 ∧ θ < Real.pi / 2

-- Theorem statement
theorem oblique_line_plane_angle_range :
  ∀ θ : Real, angle_oblique_line_plane θ ↔ 0 < θ ∧ θ < Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_oblique_line_plane_angle_range_l951_95120


namespace NUMINAMATH_CALUDE_functional_equation_solution_l951_95165

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + x = x * f y + f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (1/2) = 0 → f (-201) = 403 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l951_95165


namespace NUMINAMATH_CALUDE_negative_eight_to_negative_four_thirds_l951_95172

theorem negative_eight_to_negative_four_thirds :
  Real.rpow (-8 : ℝ) (-4/3 : ℝ) = (1/16 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_negative_eight_to_negative_four_thirds_l951_95172


namespace NUMINAMATH_CALUDE_prime_factorial_divisibility_l951_95128

theorem prime_factorial_divisibility (p k n : ℕ) (hp : Prime p) :
  p^k ∣ n! → (p!)^k ∣ n! := by
  sorry

end NUMINAMATH_CALUDE_prime_factorial_divisibility_l951_95128


namespace NUMINAMATH_CALUDE_main_theorem_l951_95159

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def even_symmetric (f : ℝ → ℝ) : Prop := ∀ x, f x - f (-x) = 0

def symmetric_about_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (2 - x)

def increasing_on_zero_two (f : ℝ → ℝ) : Prop := 
  ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Define the theorems to be proved
def periodic_four (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

def decreasing_on_two_four (f : ℝ → ℝ) : Prop := 
  ∀ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f y < f x

-- Main theorem
theorem main_theorem (heven : even_symmetric f) 
                     (hsym : symmetric_about_two f) 
                     (hinc : increasing_on_zero_two f) : 
  periodic_four f ∧ decreasing_on_two_four f := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l951_95159


namespace NUMINAMATH_CALUDE_polynomial_factorization_l951_95151

theorem polynomial_factorization (x : ℝ) :
  x^15 + x^10 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l951_95151


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l951_95191

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
sorry

theorem problem_solution : (112^2 - 97^2) / 15 = 209 := by
  have h : 112 > 97 := by sorry
  have key := square_difference_division 112 97 h
  sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l951_95191


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l951_95118

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a_1, a_3, and a_4 form a geometric sequence
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l951_95118


namespace NUMINAMATH_CALUDE_book_price_change_book_price_problem_l951_95136

theorem book_price_change (initial_price : ℝ) 
  (decrease_percent : ℝ) (increase_percent : ℝ) : ℝ :=
  let price_after_decrease := initial_price * (1 - decrease_percent)
  let final_price := price_after_decrease * (1 + increase_percent)
  final_price

theorem book_price_problem : 
  book_price_change 400 0.15 0.40 = 476 := by
  sorry

end NUMINAMATH_CALUDE_book_price_change_book_price_problem_l951_95136


namespace NUMINAMATH_CALUDE_absolute_value_of_c_l951_95199

def complex_equation (a b c : ℤ) : Prop :=
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0

theorem absolute_value_of_c (a b c : ℤ) :
  complex_equation a b c →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 109 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_c_l951_95199


namespace NUMINAMATH_CALUDE_commission_calculation_l951_95162

/-- The commission calculation problem -/
theorem commission_calculation
  (commission_rate : ℝ)
  (total_sales : ℝ)
  (h1 : commission_rate = 0.04)
  (h2 : total_sales = 312.5) :
  commission_rate * total_sales = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l951_95162


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l951_95152

theorem danny_bottle_caps (found_new : ℕ) (total_after : ℕ) (difference : ℕ) 
  (h1 : found_new = 50)
  (h2 : total_after = 60)
  (h3 : found_new = difference + 44) : 
  found_new - difference = 6 := by
sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l951_95152


namespace NUMINAMATH_CALUDE_x_value_l951_95171

theorem x_value (w y z x : ℤ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 12)
  (hx : x = y + 7) : 
  x = 134 := by sorry

end NUMINAMATH_CALUDE_x_value_l951_95171


namespace NUMINAMATH_CALUDE_marked_line_points_l951_95132

/-- Represents a line with marked points -/
structure MarkedLine where
  points : ℕ  -- Total number of points
  a_inside : ℕ  -- Number of segments A is inside
  b_inside : ℕ  -- Number of segments B is inside

/-- Theorem stating the number of points on the line -/
theorem marked_line_points (l : MarkedLine) 
  (ha : l.a_inside = 50) 
  (hb : l.b_inside = 56) : 
  l.points = 16 := by
  sorry

#check marked_line_points

end NUMINAMATH_CALUDE_marked_line_points_l951_95132


namespace NUMINAMATH_CALUDE_cosine_roots_l951_95147

theorem cosine_roots (t : ℝ) : 
  (32 * (Real.cos (6 * π / 180))^5 - 40 * (Real.cos (6 * π / 180))^3 + 10 * Real.cos (6 * π / 180) - Real.sqrt 3 = 0) →
  (32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3 = 0 ↔ 
    t = Real.cos (66 * π / 180) ∨ 
    t = Real.cos (78 * π / 180) ∨ 
    t = Real.cos (138 * π / 180) ∨ 
    t = Real.cos (150 * π / 180) ∨ 
    t = Real.cos (6 * π / 180)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_roots_l951_95147


namespace NUMINAMATH_CALUDE_no_positive_sequence_with_recurrence_l951_95168

theorem no_positive_sequence_with_recurrence : 
  ¬ ∃ (a : ℕ → ℝ), 
    (∀ n, a n > 0) ∧ 
    (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sequence_with_recurrence_l951_95168


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l951_95148

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c ≠ 0) :
  ¬ (∀ a b c, (a - b) / c > 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l951_95148


namespace NUMINAMATH_CALUDE_expression_value_l951_95160

def point_on_terminal_side (α : Real) : Prop :=
  ∃ (x y : Real), x = 1 ∧ y = -2 ∧ x = Real.cos α ∧ y = Real.sin α

theorem expression_value (α : Real) (h : point_on_terminal_side α) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l951_95160


namespace NUMINAMATH_CALUDE_infinite_common_elements_l951_95194

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

def subsequence_a (k : ℕ) : ℤ := sequence_a (2 * k + 1)

def subsequence_b (k : ℕ) : ℤ := sequence_b (3 * k + 1)

theorem infinite_common_elements : 
  ∀ k : ℕ, subsequence_a k = subsequence_b k :=
sorry

end NUMINAMATH_CALUDE_infinite_common_elements_l951_95194


namespace NUMINAMATH_CALUDE_polygon_sides_l951_95169

theorem polygon_sides (interior_angle_sum : ℝ) : interior_angle_sum = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = interior_angle_sum := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l951_95169


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l951_95170

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l951_95170
