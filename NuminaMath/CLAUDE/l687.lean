import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l687_68763

/-- The minimum distance between two points on parallel lines -/
theorem min_distance_parallel_lines :
  ∃ (P₁ P₂ : ℝ × ℝ),
    (P₁.1 + 3 * P₁.2 - 9 = 0) ∧
    (P₂.1 + 3 * P₂.2 + 1 = 0) ∧
    ∀ (Q₁ Q₂ : ℝ × ℝ),
      (Q₁.1 + 3 * Q₁.2 - 9 = 0) →
      (Q₂.1 + 3 * Q₂.2 + 1 = 0) →
      Real.sqrt 10 ≤ Real.sqrt ((Q₁.1 - Q₂.1)^2 + (Q₁.2 - Q₂.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_parallel_lines_l687_68763


namespace NUMINAMATH_CALUDE_tile_arrangements_count_l687_68788

def brown_tiles : ℕ := 2
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 2
def yellow_tiles : ℕ := 2
def orange_tiles : ℕ := 1

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles + orange_tiles

theorem tile_arrangements_count :
  (Nat.factorial total_tiles) / (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles * Nat.factorial orange_tiles) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_count_l687_68788


namespace NUMINAMATH_CALUDE_michelle_taxi_cost_l687_68752

/-- Calculates the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxi_cost (initial_fee : ℝ) (distance : ℝ) (per_mile_charge : ℝ) : ℝ :=
  initial_fee + distance * per_mile_charge

/-- Theorem stating that for the given conditions, the total cost of Michelle's taxi ride is $12. -/
theorem michelle_taxi_cost : taxi_cost 2 4 2.5 = 12 := by
  sorry

end NUMINAMATH_CALUDE_michelle_taxi_cost_l687_68752


namespace NUMINAMATH_CALUDE_baseball_team_average_l687_68743

theorem baseball_team_average (total_points : ℕ) (total_players : ℕ) 
  (high_scorers : ℕ) (high_scorer_average : ℕ) (remaining_average : ℕ) : 
  total_points = 270 → 
  total_players = 9 → 
  high_scorers = 5 → 
  high_scorer_average = 50 → 
  remaining_average = 5 → 
  total_points = high_scorers * high_scorer_average + (total_players - high_scorers) * remaining_average :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_team_average_l687_68743


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l687_68736

theorem cosine_equation_solution :
  ∃! x : ℝ, 0 < x ∧ x < π ∧ 2 * Real.cos (x - π/4) = 1 :=
by
  use 7*π/12
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l687_68736


namespace NUMINAMATH_CALUDE_time_after_2011_minutes_l687_68792

/-- Represents a time with day, hour, and minute components -/
structure DateTime where
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Converts total minutes to a DateTime structure -/
def minutesToDateTime (totalMinutes : ℕ) : DateTime :=
  let totalHours := totalMinutes / 60
  let days := totalHours / 24
  let hours := totalHours % 24
  let minutes := totalMinutes % 60
  { day := days + 1, hour := hours, minute := minutes }

/-- The starting date and time -/
def startDateTime : DateTime := { day := 1, hour := 0, minute := 0 }

/-- The number of minutes elapsed -/
def elapsedMinutes : ℕ := 2011

theorem time_after_2011_minutes :
  minutesToDateTime elapsedMinutes = { day := 2, hour := 9, minute := 31 } := by
  sorry

end NUMINAMATH_CALUDE_time_after_2011_minutes_l687_68792


namespace NUMINAMATH_CALUDE_angle_A_measure_l687_68786

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the condition √3b = 2a*sin(B)
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * t.b = 2 * t.a * Real.sin t.B

-- State the theorem
theorem angle_A_measure (t : Triangle) (h : condition t) :
  t.A = π / 3 ∨ t.A = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l687_68786


namespace NUMINAMATH_CALUDE_measuring_cups_l687_68783

theorem measuring_cups (a : Int) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : Int),
    (b ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (c ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (d ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (e ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (f ∈ ({-2, -1, 0, 1, 2} : Set Int)) ∧
    (a = 625*b + 125*c + 25*d + 5*e + f) :=
by sorry

end NUMINAMATH_CALUDE_measuring_cups_l687_68783


namespace NUMINAMATH_CALUDE_third_quadrant_angle_sum_l687_68730

theorem third_quadrant_angle_sum (θ : Real) : 
  (π < θ ∧ θ < 3*π/2) →  -- θ is in the third quadrant
  (Real.tan (θ - π/4) = 1/3) → 
  (Real.sin θ + Real.cos θ = -3/5 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_sum_l687_68730


namespace NUMINAMATH_CALUDE_lowest_fraction_job_l687_68718

/-- Given three people who can individually complete a job in 4, 6, and 8 hours respectively,
    the lowest fraction of the job that can be done in 1 hour by 2 of the people working together is 7/24. -/
theorem lowest_fraction_job (person_a person_b person_c : ℝ) 
    (ha : person_a = 1 / 4) (hb : person_b = 1 / 6) (hc : person_c = 1 / 8) : 
    min (person_a + person_b) (min (person_a + person_c) (person_b + person_c)) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_lowest_fraction_job_l687_68718


namespace NUMINAMATH_CALUDE_birds_in_tree_l687_68757

/-- Given a tree with an initial number of birds and additional birds that fly up to it,
    prove that the total number of birds is the sum of the initial and additional birds. -/
theorem birds_in_tree (initial_birds additional_birds : ℕ) 
  (h1 : initial_birds = 179)
  (h2 : additional_birds = 38) :
  initial_birds + additional_birds = 217 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l687_68757


namespace NUMINAMATH_CALUDE_roe_savings_l687_68750

def savings_problem (x : ℝ) : Prop :=
  let jan_to_jul := 7 * x
  let aug_to_nov := 4 * 15
  let december := 20
  jan_to_jul + aug_to_nov + december = 150

theorem roe_savings : ∃ x : ℝ, savings_problem x ∧ x = 10 :=
  sorry

end NUMINAMATH_CALUDE_roe_savings_l687_68750


namespace NUMINAMATH_CALUDE_main_diagonal_contains_all_numbers_l687_68749

/-- A square matrix of odd size n, where each row and column contains all numbers from 1 to n,
    and the matrix is symmetric with respect to the main diagonal. -/
structure SymmetricLatinSquare (n : ℕ) :=
  (matrix : Fin n → Fin n → Fin n)
  (odd : Odd n)
  (latin_row : ∀ i j : Fin n, ∃ k : Fin n, matrix i k = j)
  (latin_col : ∀ i j : Fin n, ∃ k : Fin n, matrix k i = j)
  (symmetric : ∀ i j : Fin n, matrix i j = matrix j i)

/-- Theorem stating that all numbers from 1 to n appear on the main diagonal of a SymmetricLatinSquare. -/
theorem main_diagonal_contains_all_numbers (n : ℕ) (sls : SymmetricLatinSquare n) :
  ∀ k : Fin n, ∃ i : Fin n, sls.matrix i i = k := by
  sorry

end NUMINAMATH_CALUDE_main_diagonal_contains_all_numbers_l687_68749


namespace NUMINAMATH_CALUDE_value_of_c_l687_68797

theorem value_of_c (x b c : ℝ) (h1 : x - 1/x = 2*b) (h2 : x^3 - 1/x^3 = c) : c = 8*b^3 + 6*b := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l687_68797


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l687_68724

theorem like_terms_exponent_sum (m n : ℤ) : 
  (∃ (k : ℚ), k * x * y^2 = x^(m-2) * y^(n+3)) → m + n = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l687_68724


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l687_68773

/-- A regular triangle with an inscribed square -/
structure RegularTriangleWithInscribedSquare where
  -- Side length of the regular triangle
  triangleSide : ℝ
  -- Distance from a vertex of the triangle to the nearest vertex of the square on the opposite side
  vertexToSquareDistance : ℝ
  -- Assumption that the triangle is regular (equilateral)
  regular : triangleSide > 0
  -- Assumption that the square is inscribed (vertexToSquareDistance < triangleSide)
  inscribed : vertexToSquareDistance < triangleSide

/-- The side length of the inscribed square -/
def squareSideLength (t : RegularTriangleWithInscribedSquare) : ℝ := 
  t.triangleSide - t.vertexToSquareDistance

/-- Theorem stating that for a regular triangle with side length 30 and vertexToSquareDistance 29, 
    the side length of the inscribed square is 30 -/
theorem inscribed_square_side_length 
  (t : RegularTriangleWithInscribedSquare) 
  (h1 : t.triangleSide = 30) 
  (h2 : t.vertexToSquareDistance = 29) : 
  squareSideLength t = 30 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l687_68773


namespace NUMINAMATH_CALUDE_josie_remaining_money_l687_68746

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific initial amount and purchase costs, the remaining amount is correct -/
theorem josie_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_josie_remaining_money_l687_68746


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l687_68722

-- Problem 1
theorem problem_1 : (-2)^2 + (Real.sqrt 2 - 1)^0 - 1 = 4 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (A B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) :
  a > 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l687_68722


namespace NUMINAMATH_CALUDE_students_with_A_grade_l687_68713

theorem students_with_A_grade (total : ℕ) (h1 : total ≤ 50) 
  (h2 : total % 3 = 0) (h3 : total % 13 = 0) : ℕ := by
  have h4 : total = 39 := by sorry
  have h5 : (total / 3 : ℕ) + (5 * total / 13 : ℕ) + 1 + 10 = total := by sorry
  exact 10

#check students_with_A_grade

end NUMINAMATH_CALUDE_students_with_A_grade_l687_68713


namespace NUMINAMATH_CALUDE_pi_half_irrational_l687_68767

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l687_68767


namespace NUMINAMATH_CALUDE_cube_inequality_l687_68740

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x > a^y) : x^3 < y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l687_68740


namespace NUMINAMATH_CALUDE_leap_year_1996_not_others_l687_68791

def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

theorem leap_year_1996_not_others : 
  is_leap_year 1996 ∧ 
  ¬is_leap_year 1998 ∧ 
  ¬is_leap_year 2010 ∧ 
  ¬is_leap_year 2100 :=
by sorry

end NUMINAMATH_CALUDE_leap_year_1996_not_others_l687_68791


namespace NUMINAMATH_CALUDE_functional_equation_solution_l687_68796

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

-- Define the solution conditions
def IsSolution (f : FunctionType) : Prop :=
  (∀ x : ℝ, f x = 0) ∨
  ((∀ x : ℝ, x ≠ 0 → f x = 1) ∧ ∃ c : ℝ, f 0 = c)

-- Theorem statement
theorem functional_equation_solution (f : FunctionType) :
  SatisfiesEquation f → IsSolution f :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l687_68796


namespace NUMINAMATH_CALUDE_polynomial_roots_nature_l687_68734

theorem polynomial_roots_nature :
  ∃ (r₁ r₂ r₃ : ℝ), 
    (r₁ > 0 ∧ r₂ < 0 ∧ r₃ < 0) ∧
    (∀ x : ℝ, x^3 - 7*x^2 + 14*x - 8 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_nature_l687_68734


namespace NUMINAMATH_CALUDE_unique_divisible_by_72_l687_68733

theorem unique_divisible_by_72 : ∃! n : ℕ,
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 1000000000 + 20222023 * 10 + b) ∧
  n % 72 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_72_l687_68733


namespace NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l687_68707

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
by sorry

end NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l687_68707


namespace NUMINAMATH_CALUDE_height_difference_l687_68795

/-- Given three people A, B, and C, where A's height is 30% less than B's,
    and C's height is 20% more than A's, prove that the percentage difference
    between B's height and C's height is 16%. -/
theorem height_difference (h_b : ℝ) (h_b_pos : h_b > 0) : 
  let h_a := 0.7 * h_b
  let h_c := 1.2 * h_a
  ((h_b - h_c) / h_b) * 100 = 16 := by sorry

end NUMINAMATH_CALUDE_height_difference_l687_68795


namespace NUMINAMATH_CALUDE_max_value_of_s_l687_68799

theorem max_value_of_s (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10)
  (product_condition : p * q + p * r + p * s + q * r + q * s + r * s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l687_68799


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l687_68716

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_line : Line) 
  (point : ℝ × ℝ) : 
  ∃ (result_line : Line), 
    result_line.contains point.1 point.2 ∧ 
    result_line.parallel given_line ∧
    result_line.a = 1 ∧ 
    result_line.b = 2 ∧ 
    result_line.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_l687_68716


namespace NUMINAMATH_CALUDE_sin_315_degrees_l687_68706

theorem sin_315_degrees :
  Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l687_68706


namespace NUMINAMATH_CALUDE_complex_division_result_l687_68700

theorem complex_division_result (z : ℂ) (h : z = 1 + I) : z / (1 - I) = I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l687_68700


namespace NUMINAMATH_CALUDE_parking_lot_tires_l687_68780

/-- Represents the number of tires for a vehicle type -/
structure VehicleTires where
  count : Nat
  wheels : Nat
  spares : Nat

/-- Calculates the total number of tires for a vehicle type -/
def totalTires (v : VehicleTires) : Nat :=
  v.count * (v.wheels + v.spares)

/-- Theorem: The total number of tires in the parking lot is 310 -/
theorem parking_lot_tires :
  let cars := VehicleTires.mk 30 4 1
  let motorcycles := VehicleTires.mk 20 2 2
  let trucks := VehicleTires.mk 10 6 1
  let bicycles := VehicleTires.mk 5 2 0
  totalTires cars + totalTires motorcycles + totalTires trucks + totalTires bicycles = 310 :=
by sorry

end NUMINAMATH_CALUDE_parking_lot_tires_l687_68780


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_13_18_l687_68759

theorem smallest_divisible_by_15_13_18 : ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 13 ∣ n ∧ 18 ∣ n ∧ ∀ (m : ℕ), m > 0 → 15 ∣ m → 13 ∣ m → 18 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_13_18_l687_68759


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l687_68772

/-- Given a triangle XYZ with points D on XY and E on YZ satisfying certain ratios,
    prove that DE:EF = 1:4 when DE intersects XZ at F. -/
theorem triangle_ratio_theorem (X Y Z D E F : ℝ × ℝ) : 
  -- Triangle XYZ exists
  (∃ (a b c : ℝ), X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) →
  -- D is on XY with XD:DY = 4:1
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ D = (1 - t) • X + t • Y ∧ t = 1/5) →
  -- E is on YZ with YE:EZ = 4:1
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ E = (1 - s) • Y + s • Z ∧ s = 4/5) →
  -- DE intersects XZ at F
  (∃ r : ℝ, F = (1 - r) • D + r • E ∧ 
            ∃ q : ℝ, F = (1 - q) • X + q • Z) →
  -- Then DE:EF = 1:4
  ‖E - D‖ / ‖F - E‖ = 1/4 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l687_68772


namespace NUMINAMATH_CALUDE_similar_triangles_dimensions_l687_68761

theorem similar_triangles_dimensions (h₁ base₁ h₂ base₂ : ℝ) : 
  h₁ > 0 → base₁ > 0 → h₂ > 0 → base₂ > 0 →
  (h₁ * base₁) / (h₂ * base₂) = 1 / 9 →
  h₁ = 5 → base₁ = 6 →
  h₂ = 15 ∧ base₂ = 18 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_dimensions_l687_68761


namespace NUMINAMATH_CALUDE_product_plus_five_l687_68766

theorem product_plus_five : (-11 * -8) + 5 = 93 := by
  sorry

end NUMINAMATH_CALUDE_product_plus_five_l687_68766


namespace NUMINAMATH_CALUDE_four_integer_average_l687_68771

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧ 
  d = 90 ∧ 
  a ≥ 37 → 
  (a + b + c + d) / 4 ≥ 51 := by
sorry

end NUMINAMATH_CALUDE_four_integer_average_l687_68771


namespace NUMINAMATH_CALUDE_star_polygon_forms_pyramid_net_iff_l687_68729

/-- A structure representing two concentric circles with a star-shaped polygon construction -/
structure ConcentricCirclesWithStarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  (r_positive : r > 0)
  (R_greater : R > r)

/-- The condition for the star-shaped polygon to form the net of a pyramid -/
def forms_pyramid_net (c : ConcentricCirclesWithStarPolygon) : Prop :=
  c.R > 2 * c.r

/-- Theorem stating the necessary and sufficient condition for the star-shaped polygon
    to form the net of a pyramid -/
theorem star_polygon_forms_pyramid_net_iff (c : ConcentricCirclesWithStarPolygon) :
  forms_pyramid_net c ↔ c.R > 2 * c.r :=
sorry

end NUMINAMATH_CALUDE_star_polygon_forms_pyramid_net_iff_l687_68729


namespace NUMINAMATH_CALUDE_complex_square_condition_l687_68703

theorem complex_square_condition (a b : ℝ) : 
  (∃ (x y : ℝ), (x + y * Complex.I) ^ 2 = 2 * Complex.I ∧ (x ≠ 1 ∨ y ≠ 1)) ∧ 
  ((1 : ℝ) + (1 : ℝ) * Complex.I) ^ 2 = 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_condition_l687_68703


namespace NUMINAMATH_CALUDE_water_height_in_cylinder_l687_68764

/-- The height of water in a cylinder when poured from an inverted cone -/
theorem water_height_in_cylinder (cone_radius cone_height cylinder_radius : ℝ) 
  (h_cone_radius : cone_radius = 10)
  (h_cone_height : cone_height = 15)
  (h_cylinder_radius : cylinder_radius = 20) :
  (1 / 3 * π * cone_radius^2 * cone_height) / (π * cylinder_radius^2) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_water_height_in_cylinder_l687_68764


namespace NUMINAMATH_CALUDE_parabola_focus_x_coord_l687_68739

/-- A parabola defined by parametric equations -/
structure ParametricParabola where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The focus of a parabola -/
structure ParabolaFocus where
  x : ℝ
  y : ℝ

/-- Theorem: The x-coordinate of the focus of the parabola given by x = 4t² and y = 4t is 1 -/
theorem parabola_focus_x_coord (p : ParametricParabola) 
  (h1 : p.x = fun t => 4 * t^2)
  (h2 : p.y = fun t => 4 * t) : 
  ∃ f : ParabolaFocus, f.x = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_x_coord_l687_68739


namespace NUMINAMATH_CALUDE_max_value_trigonometric_expression_l687_68705

theorem max_value_trigonometric_expression :
  ∃ (M : ℝ), M = 5 ∧ ∀ (x : ℝ), 3 * Real.cos x + 4 * Real.sin x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_expression_l687_68705


namespace NUMINAMATH_CALUDE_ribbon_tape_remaining_l687_68735

theorem ribbon_tape_remaining (total length_used_ribbon length_used_gift : ℝ) 
  (h1 : total = 1.6)
  (h2 : length_used_ribbon = 0.8)
  (h3 : length_used_gift = 0.3) :
  total - length_used_ribbon - length_used_gift = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_tape_remaining_l687_68735


namespace NUMINAMATH_CALUDE_function_value_at_2017_l687_68789

/-- Given a function f(x) = x^2 - x * f'(0) - 1, prove that f(2017) = 2016 * 2018 -/
theorem function_value_at_2017 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - x * (deriv f 0) - 1) : 
  f 2017 = 2016 * 2018 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2017_l687_68789


namespace NUMINAMATH_CALUDE_flight_cost_X_to_Y_l687_68717

/-- The cost to fly between two cities given the distance and cost parameters. -/
def flight_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem stating that the flight cost from X to Y is $660. -/
theorem flight_cost_X_to_Y :
  flight_cost 4500 0.12 120 = 660 := by
  sorry

end NUMINAMATH_CALUDE_flight_cost_X_to_Y_l687_68717


namespace NUMINAMATH_CALUDE_sum_of_first_100_terms_l687_68793

-- Define the function f
def f (n : ℕ) : ℤ :=
  if n % 2 = 1 then n^2 else -(n^2)

-- Define the sequence a_n
def a (n : ℕ) : ℤ := f n + f (n + 1)

-- State the theorem
theorem sum_of_first_100_terms :
  (Finset.range 100).sum (λ i => a (i + 1)) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_terms_l687_68793


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l687_68711

theorem sqrt_equation_solution (c : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 9*c)) + Real.sqrt (3 + Real.sqrt (3 + c)) = 3 + 3 * Real.sqrt 3 →
  c = 33 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l687_68711


namespace NUMINAMATH_CALUDE_function_with_two_zeros_properties_l687_68755

/-- A function f(x) = ax² - e^x with two positive zeros -/
structure FunctionWithTwoZeros where
  a : ℝ
  x₁ : ℝ
  x₂ : ℝ
  h₁ : 0 < x₁
  h₂ : x₁ < x₂
  h₃ : a * x₁^2 = Real.exp x₁
  h₄ : a * x₂^2 = Real.exp x₂

/-- The range of a and the sum of zeros for a function with two positive zeros -/
theorem function_with_two_zeros_properties (f : FunctionWithTwoZeros) :
  f.a > Real.exp 2 / 4 ∧ f.x₁ + f.x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_properties_l687_68755


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l687_68714

/-- Given a rectangle with dimensions 12 × 10 inches, containing an inner rectangle
    of 6 × 2 inches, and a shaded area of 116 square inches, prove that the
    perimeter of the non-shaded region is 10 inches. -/
theorem non_shaded_perimeter (outer_length outer_width inner_length inner_width shaded_area : ℝ)
  (h_outer_length : outer_length = 12)
  (h_outer_width : outer_width = 10)
  (h_inner_length : inner_length = 6)
  (h_inner_width : inner_width = 2)
  (h_shaded_area : shaded_area = 116)
  (h_right_angles : ∀ angle, angle = 90) :
  let total_area := outer_length * outer_width
  let inner_area := inner_length * inner_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_length := 4
  let non_shaded_width := 1
  2 * (non_shaded_length + non_shaded_width) = 10 := by
    sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l687_68714


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l687_68742

-- Define the sets M and N
def M (l : ℝ) : Set ℝ := { x | -l < x ∧ x < 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 2 }

-- Theorem statement
theorem intersection_of_M_and_N (l : ℝ) (h : l > 0) :
  M l ∩ N = { x | 0 ≤ x ∧ x < 1 } := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l687_68742


namespace NUMINAMATH_CALUDE_mans_age_double_sons_l687_68745

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 32

/-- Represents the current age of the son -/
def son_age : ℕ := 30

/-- Represents the number of years until the man's age is twice his son's age -/
def years_until_double : ℕ := 2

/-- Theorem stating that in 'years_until_double' years, the man's age will be twice his son's age -/
theorem mans_age_double_sons (y : ℕ) :
  y = years_until_double ↔ 
  (son_age + age_difference + y = 2 * (son_age + y)) :=
sorry

end NUMINAMATH_CALUDE_mans_age_double_sons_l687_68745


namespace NUMINAMATH_CALUDE_buqing_college_students_l687_68726

/-- Represents the number of students in each college -/
structure CollegeStudents where
  a₁ : ℕ  -- Buqing College
  a₂ : ℕ  -- Jiazhen College
  a₃ : ℕ  -- Hede College
  a₄ : ℕ  -- Wangdao College

/-- Checks if the given numbers form an arithmetic sequence with the specified common difference -/
def isArithmeticSequence (a b c : ℕ) (d : ℕ) : Prop :=
  b = a + d ∧ c = b + d

/-- Checks if the given numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = a * r ∧ c = b * r

/-- The main theorem to prove -/
theorem buqing_college_students 
  (s : CollegeStudents) 
  (total : s.a₁ + s.a₂ + s.a₃ + s.a₄ = 474) 
  (arith_seq : isArithmeticSequence s.a₁ s.a₂ s.a₃ 12)
  (geom_seq : isGeometricSequence s.a₁ s.a₃ s.a₄) : 
  s.a₁ = 96 := by
  sorry

end NUMINAMATH_CALUDE_buqing_college_students_l687_68726


namespace NUMINAMATH_CALUDE_card_distribution_theorem_l687_68762

-- Define the number of cards and boxes
def num_cards : ℕ := 6
def num_boxes : ℕ := 4

-- Define a function to calculate the number of arrangements
def count_arrangements (cards : ℕ) (boxes : ℕ) : ℕ := sorry

-- Define a function to calculate the number of arrangements with two specific cards not in the same box
def count_arrangements_with_restriction (cards : ℕ) (boxes : ℕ) : ℕ := sorry

-- Theorem statement
theorem card_distribution_theorem :
  count_arrangements_with_restriction num_cards num_boxes = 1320 := by sorry

end NUMINAMATH_CALUDE_card_distribution_theorem_l687_68762


namespace NUMINAMATH_CALUDE_helicopter_rental_theorem_l687_68719

/-- Calculates the total cost of renting a helicopter given the daily rental hours, number of days, and hourly rate. -/
def helicopter_rental_cost (hours_per_day : ℕ) (days : ℕ) (rate_per_hour : ℕ) : ℕ :=
  hours_per_day * days * rate_per_hour

/-- Proves that renting a helicopter for 2 hours a day for 3 days at $75 per hour costs $450 in total. -/
theorem helicopter_rental_theorem : helicopter_rental_cost 2 3 75 = 450 := by
  sorry

end NUMINAMATH_CALUDE_helicopter_rental_theorem_l687_68719


namespace NUMINAMATH_CALUDE_train_travel_rate_l687_68769

/-- Given a train's travel information, prove the rate of additional hours per mile -/
theorem train_travel_rate (initial_distance : ℝ) (initial_time : ℝ) 
  (additional_distance : ℝ) (additional_time : ℝ) 
  (h1 : initial_distance = 360) 
  (h2 : initial_time = 3) 
  (h3 : additional_distance = 240) 
  (h4 : additional_time = 2) :
  (additional_time / additional_distance) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_rate_l687_68769


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_120_and_m_l687_68702

theorem greatest_common_divisor_of_120_and_m (m : ℕ) : 
  (∃ d₁ d₂ d₃ d₄ : ℕ, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
    {d : ℕ | d ∣ 120 ∧ d ∣ m} = {d₁, d₂, d₃, d₄}) →
  Nat.gcd 120 m = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_120_and_m_l687_68702


namespace NUMINAMATH_CALUDE_problem_solution_l687_68709

theorem problem_solution : 
  |(-5)| - 2 * 3^0 + Real.tan (π/4) + Real.sqrt 9 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l687_68709


namespace NUMINAMATH_CALUDE_force_at_200000_l687_68704

/-- Represents the gravitational force at a given distance -/
def gravitational_force (d : ℝ) : ℝ := sorry

/-- The gravitational force follows the inverse square law -/
axiom inverse_square_law (d₁ d₂ : ℝ) :
  gravitational_force d₁ * d₁^2 = gravitational_force d₂ * d₂^2

/-- The gravitational force at 5,000 miles is 500 Newtons -/
axiom force_at_5000 : gravitational_force 5000 = 500

/-- Theorem: The gravitational force at 200,000 miles is 5/16 Newtons -/
theorem force_at_200000 : gravitational_force 200000 = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_force_at_200000_l687_68704


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l687_68728

/-- A square is a quadrilateral with four equal sides and four right angles. -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Two squares are congruent if they have the same side length. -/
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

/-- Theorem: Not all squares are congruent to each other. -/
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

/-- All squares are equiangular. -/
theorem all_squares_equiangular : True := by
  sorry

/-- All squares are rectangles. -/
theorem all_squares_rectangles : True := by
  sorry

/-- All squares are regular polygons. -/
theorem all_squares_regular_polygons : True := by
  sorry

/-- All squares are similar to each other. -/
theorem all_squares_similar : True := by
  sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_all_squares_equiangular_all_squares_rectangles_all_squares_regular_polygons_all_squares_similar_l687_68728


namespace NUMINAMATH_CALUDE_oplus_three_equals_fifteen_implies_a_equals_eleven_l687_68741

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 3*a - 2*b^2

-- Theorem statement
theorem oplus_three_equals_fifteen_implies_a_equals_eleven :
  ∀ a : ℝ, oplus a 3 = 15 → a = 11 := by
sorry

end NUMINAMATH_CALUDE_oplus_three_equals_fifteen_implies_a_equals_eleven_l687_68741


namespace NUMINAMATH_CALUDE_hex_to_binary_bits_l687_68798

/-- The number of bits required to represent a positive integer in binary. -/
def bitsRequired (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- The decimal representation of the hexadecimal number 1A1A1. -/
def hexNumber : ℕ := 106913

theorem hex_to_binary_bits :
  bitsRequired hexNumber = 17 := by
  sorry

end NUMINAMATH_CALUDE_hex_to_binary_bits_l687_68798


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l687_68782

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^2 + 3

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l687_68782


namespace NUMINAMATH_CALUDE_quadratic_sum_equals_113_l687_68794

theorem quadratic_sum_equals_113 (x y : ℝ) 
  (eq1 : 3*x + 2*y = 7) 
  (eq2 : 2*x + 3*y = 8) : 
  13*x^2 + 22*x*y + 13*y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_equals_113_l687_68794


namespace NUMINAMATH_CALUDE_exists_password_with_twenty_permutations_l687_68744

/-- Represents a password as a list of characters -/
def Password := List Char

/-- Counts the number of unique permutations of a password -/
def countUniquePermutations (p : Password) : Nat :=
  sorry

/-- Theorem: There exists a 5-character password with exactly 20 different permutations -/
theorem exists_password_with_twenty_permutations :
  ∃ (p : Password), p.length = 5 ∧ countUniquePermutations p = 20 := by
  sorry

end NUMINAMATH_CALUDE_exists_password_with_twenty_permutations_l687_68744


namespace NUMINAMATH_CALUDE_special_line_unique_l687_68790

/-- A line that satisfies the given conditions -/
structure SpecialLine where
  m : ℝ
  b : ℝ
  b_nonzero : b ≠ 0
  passes_through : m * 2 + b = 7

/-- The condition for the intersection points -/
def intersectionCondition (l : SpecialLine) (k : ℝ) : Prop :=
  |k^2 + 4*k + 3 - (l.m * k + l.b)| = 4

/-- The main theorem -/
theorem special_line_unique (l : SpecialLine) :
  (∃! k, intersectionCondition l k) → l.m = 10 ∧ l.b = -13 := by
  sorry

end NUMINAMATH_CALUDE_special_line_unique_l687_68790


namespace NUMINAMATH_CALUDE_average_minus_tenth_l687_68712

theorem average_minus_tenth (x : ℚ) : x = (1/8 + 1/3) / 2 - 1/10 → x = 31/240 := by
  sorry

end NUMINAMATH_CALUDE_average_minus_tenth_l687_68712


namespace NUMINAMATH_CALUDE_angle_measure_l687_68778

theorem angle_measure : 
  ∀ x : ℝ, 
  (x + (4 * x + 7) = 90) →  -- Condition 2 (complementary angles)
  x = 83 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l687_68778


namespace NUMINAMATH_CALUDE_uba_capital_suvs_l687_68723

/-- Represents the number of SUVs purchased by UBA Capital --/
def num_suvs (total_vehicles : ℕ) : ℕ :=
  let toyota_count := (9 * total_vehicles) / 10
  let honda_count := total_vehicles - toyota_count
  let toyota_suvs := (90 * toyota_count) / 100
  let honda_suvs := (10 * honda_count) / 100
  toyota_suvs + honda_suvs

/-- Theorem stating that the number of SUVs purchased is 8 --/
theorem uba_capital_suvs :
  ∃ (total_vehicles : ℕ), num_suvs total_vehicles = 8 :=
sorry

end NUMINAMATH_CALUDE_uba_capital_suvs_l687_68723


namespace NUMINAMATH_CALUDE_circle_through_three_points_l687_68738

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The standard equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_three_points :
  let A : Point := ⟨-4, 0⟩
  let B : Point := ⟨0, 2⟩
  let O : Point := ⟨0, 0⟩
  let center : Point := ⟨-2, 1⟩
  let radius : ℝ := Real.sqrt 5
  (CircleEquation center radius) ∧
  (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
  (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
  (center.x - O.x)^2 + (center.y - O.y)^2 = radius^2 :=
by
  sorry

#check circle_through_three_points

end NUMINAMATH_CALUDE_circle_through_three_points_l687_68738


namespace NUMINAMATH_CALUDE_power_difference_l687_68784

theorem power_difference (m n : ℕ) (h1 : 3^m = 8) (h2 : 3^n = 2) : 3^(m-n) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l687_68784


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l687_68701

def circle_equation (x y : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 36

theorem max_sum_on_circle :
  ∃ (max : ℤ),
    (∀ x y : ℤ, circle_equation x y → x + y ≤ max) ∧
    (∃ x y : ℤ, circle_equation x y ∧ x + y = max) ∧
    max = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l687_68701


namespace NUMINAMATH_CALUDE_rectangle_division_l687_68765

/-- Given a rectangle with vertices (x, 0), (9, 0), (x, 2), and (9, 2),
    if a line passing through the origin with slope 0.2 divides the rectangle
    into two identical quadrilaterals, then x = 1. -/
theorem rectangle_division (x : ℝ) : 
  (∃ (l : Set (ℝ × ℝ)), 
    -- l is a line passing through the origin
    (0, 0) ∈ l ∧
    -- l has slope 0.2
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 0.2) ∧
    -- l divides the rectangle into two identical quadrilaterals
    (∃ (m : ℝ × ℝ), m ∈ l ∧ m.1 = (x + 9) / 2 ∧ m.2 = 1)) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_division_l687_68765


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l687_68777

theorem min_value_fraction_sum (a b c : ℤ) (h1 : a > b) (h2 : b > c) :
  let x := (a + b + c : ℚ) / (a - b - c : ℚ)
  2 ≤ x + 1 / x ∧ ∃ a b c : ℤ, a > b ∧ b > c ∧ x + 1 / x = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l687_68777


namespace NUMINAMATH_CALUDE_modulus_of_z_l687_68731

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 3 + 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l687_68731


namespace NUMINAMATH_CALUDE_yvette_sundae_cost_l687_68776

-- Define the costs of sundaes and other parameters
def alicia_sundae_cost : ℝ := 7.50
def brant_sundae_cost : ℝ := 10.00
def josh_sundae_cost : ℝ := 8.50
def tip_percentage : ℝ := 0.20
def final_bill : ℝ := 42.00

-- Define the theorem
theorem yvette_sundae_cost : 
  ∃ (yvette_cost : ℝ), 
    yvette_cost = final_bill - (alicia_sundae_cost + brant_sundae_cost + josh_sundae_cost + tip_percentage * final_bill) ∧
    yvette_cost = 7.60 := by
  sorry

end NUMINAMATH_CALUDE_yvette_sundae_cost_l687_68776


namespace NUMINAMATH_CALUDE_hoseok_took_fewest_l687_68774

/-- Represents the number of cards taken by each person -/
structure CardCount where
  jungkook : ℕ
  hoseok : ℕ
  seokjin : ℕ

/-- Defines the conditions of the problem -/
def problemConditions (cc : CardCount) : Prop :=
  cc.jungkook = 10 ∧
  cc.hoseok = 7 ∧
  cc.seokjin = cc.jungkook - 2

/-- Theorem stating that Hoseok took the fewest cards -/
theorem hoseok_took_fewest (cc : CardCount) 
  (h : problemConditions cc) : 
  cc.hoseok < cc.jungkook ∧ cc.hoseok < cc.seokjin :=
by
  sorry

#check hoseok_took_fewest

end NUMINAMATH_CALUDE_hoseok_took_fewest_l687_68774


namespace NUMINAMATH_CALUDE_ninth_term_of_specific_sequence_l687_68775

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem ninth_term_of_specific_sequence :
  ∀ seq : GeometricSequence,
    nth_term seq 5 = 80 →
    nth_term seq 7 = 320 →
    nth_term seq 9 = 1280 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_specific_sequence_l687_68775


namespace NUMINAMATH_CALUDE_combinations_equal_twelve_l687_68720

/-- The number of wall color choices -/
def wall_colors : Nat := 4

/-- The number of flooring type choices -/
def flooring_types : Nat := 3

/-- The total number of combinations of wall color and flooring type -/
def total_combinations : Nat := wall_colors * flooring_types

/-- Theorem: The total number of combinations is 12 -/
theorem combinations_equal_twelve : total_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_twelve_l687_68720


namespace NUMINAMATH_CALUDE_two_inequalities_always_true_l687_68754

theorem two_inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hxa : x < a) 
  (hyb : y < b) 
  (hx_neg : x < 0) 
  (hy_neg : y < 0) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) :
  ∃! n : ℕ, n = (Bool.toNat (x + y < a + b)) + 
               (Bool.toNat (x - y < a - b)) + 
               (Bool.toNat (x * y < a * b)) + 
               (Bool.toNat (x / y < a / b)) ∧ 
               n = 2 := by
sorry

end NUMINAMATH_CALUDE_two_inequalities_always_true_l687_68754


namespace NUMINAMATH_CALUDE_sqrt_five_irrational_l687_68753

theorem sqrt_five_irrational :
  ∀ (x : ℝ), x ^ 2 = 5 → ¬ (∃ (a b : ℤ), b ≠ 0 ∧ x = a / b) :=
by sorry

def zero_rational : ℚ := 0

def three_point_fourteen_rational : ℚ := 314 / 100

def negative_eight_sevenths_rational : ℚ := -8 / 7

#check sqrt_five_irrational
#check zero_rational
#check three_point_fourteen_rational
#check negative_eight_sevenths_rational

end NUMINAMATH_CALUDE_sqrt_five_irrational_l687_68753


namespace NUMINAMATH_CALUDE_xiaochun_current_age_l687_68732

-- Define Xiaochun's current age
def xiaochun_age : ℕ := sorry

-- Define Xiaochun's brother's current age
def brother_age : ℕ := sorry

-- Condition 1: Xiaochun's age is 18 years less than his brother's age
axiom age_difference : xiaochun_age = brother_age - 18

-- Condition 2: In 3 years, Xiaochun's age will be half of his brother's age
axiom future_age_relation : xiaochun_age + 3 = (brother_age + 3) / 2

-- Theorem to prove
theorem xiaochun_current_age : xiaochun_age = 15 := by sorry

end NUMINAMATH_CALUDE_xiaochun_current_age_l687_68732


namespace NUMINAMATH_CALUDE_melanie_gumball_sale_l687_68725

/-- Represents the sale of gumballs -/
structure GumballSale where
  price_per_gumball : ℕ
  total_money : ℕ

/-- Calculates the number of gumballs sold -/
def gumballs_sold (sale : GumballSale) : ℕ :=
  sale.total_money / sale.price_per_gumball

/-- Theorem: Melanie sold 4 gumballs -/
theorem melanie_gumball_sale :
  let sale : GumballSale := { price_per_gumball := 8, total_money := 32 }
  gumballs_sold sale = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_gumball_sale_l687_68725


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l687_68779

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola) : (ℝ → ℝ) × (ℝ → ℝ) := sorry

/-- A perpendicular line from a point to a line -/
def perpendicular_line (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ) : ℝ × ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  let f := right_focus h
  let (asym1, asym2) := asymptotes h
  let perp := perpendicular_line f asym1
  let a := intersection_point perp asym1
  let b := intersection_point perp asym2
  (b.1 - f.1)^2 + (b.2 - f.2)^2 = 4 * ((a.1 - f.1)^2 + (a.2 - f.2)^2) →
  eccentricity h = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l687_68779


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_value_l687_68727

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) → n = 36 ∨ n = -36 :=
by sorry

theorem positive_n_value (n : ℝ) :
  (∃! x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ n > 0 → n = 36 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_value_l687_68727


namespace NUMINAMATH_CALUDE_unique_unbeatable_city_l687_68768

/-- Represents a city with two bulldozers -/
structure City where
  leftBulldozer : ℕ
  rightBulldozer : ℕ

/-- Represents the road with n cities -/
def Road (n : ℕ) := Fin n → City

/-- A city i overtakes city j if its right bulldozer can reach j -/
def overtakes (road : Road n) (i j : Fin n) : Prop :=
  i < j ∧ ∀ k, i < k ∧ k ≤ j → (road i).rightBulldozer > (road k).leftBulldozer

/-- There exists a unique city that cannot be overtaken -/
theorem unique_unbeatable_city (n : ℕ) (road : Road n)
  (h1 : ∀ i j : Fin n, i ≠ j → (road i).leftBulldozer ≠ (road j).leftBulldozer)
  (h2 : ∀ i j : Fin n, i ≠ j → (road i).rightBulldozer ≠ (road j).rightBulldozer)
  (h3 : ∀ i : Fin n, (road i).leftBulldozer ≠ (road i).rightBulldozer) :
  ∃! i : Fin n, ∀ j : Fin n, j ≠ i → ¬(overtakes road j i) :=
sorry

end NUMINAMATH_CALUDE_unique_unbeatable_city_l687_68768


namespace NUMINAMATH_CALUDE_total_soldiers_on_great_wall_l687_68710

-- Define the parameters
def wall_length : ℕ := 7300
def tower_interval : ℕ := 5
def soldiers_per_tower : ℕ := 2

-- Theorem statement
theorem total_soldiers_on_great_wall :
  (wall_length / tower_interval) * soldiers_per_tower = 2920 :=
by sorry

end NUMINAMATH_CALUDE_total_soldiers_on_great_wall_l687_68710


namespace NUMINAMATH_CALUDE_wall_construction_l687_68737

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 1800

/-- Rate of the first bricklayer in bricks per hour -/
def rate1 : ℚ := total_bricks / 12

/-- Rate of the second bricklayer in bricks per hour -/
def rate2 : ℚ := total_bricks / 15

/-- Combined rate reduction when working together -/
def rate_reduction : ℕ := 15

/-- Time taken to complete the wall together -/
def time_taken : ℕ := 6

theorem wall_construction :
  (time_taken : ℚ) * (rate1 + rate2 - rate_reduction) = total_bricks := by sorry

end NUMINAMATH_CALUDE_wall_construction_l687_68737


namespace NUMINAMATH_CALUDE_xy_value_l687_68747

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l687_68747


namespace NUMINAMATH_CALUDE_general_term_formula_first_term_l687_68785

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

/-- The general term of the sequence -/
def a (n : ℕ+) : ℤ := 4 * n.val - 5

/-- Theorem stating that the general term formula is correct -/
theorem general_term_formula (n : ℕ+) : a n = S n - S (n - 1) := by
  sorry

/-- Theorem stating that the formula holds for the first term -/
theorem first_term : a 1 = S 1 := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_first_term_l687_68785


namespace NUMINAMATH_CALUDE_parabola_vertex_l687_68748

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, -2)

/-- Theorem: The vertex of the parabola y = -3(x-1)^2 - 2 is at the point (1, -2) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l687_68748


namespace NUMINAMATH_CALUDE_sum_three_consecutive_terms_l687_68760

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sum_three_consecutive_terms
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 21) :
  a 4 + a 5 + a 6 = 63 :=
sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_terms_l687_68760


namespace NUMINAMATH_CALUDE_cattle_transport_speed_l687_68770

/-- Proves that the speed of a truck transporting cattle is 60 miles per hour given specific conditions -/
theorem cattle_transport_speed (total_cattle : ℕ) (distance : ℕ) (truck_capacity : ℕ) (total_time : ℕ) :
  total_cattle = 400 →
  distance = 60 →
  truck_capacity = 20 →
  total_time = 40 →
  (distance * 2 * (total_cattle / truck_capacity)) / total_time = 60 := by
  sorry

#check cattle_transport_speed

end NUMINAMATH_CALUDE_cattle_transport_speed_l687_68770


namespace NUMINAMATH_CALUDE_chris_candy_distribution_chris_total_candy_l687_68781

theorem chris_candy_distribution (first_group : Nat) (first_amount : Nat) 
  (second_group : Nat) (remaining_amount : Nat) : Nat :=
  let total_first := first_group * first_amount
  let total_second := second_group * (2 * first_amount)
  total_first + total_second + remaining_amount

theorem chris_total_candy : chris_candy_distribution 10 12 7 50 = 338 := by
  sorry

end NUMINAMATH_CALUDE_chris_candy_distribution_chris_total_candy_l687_68781


namespace NUMINAMATH_CALUDE_drop_1m_l687_68715

def water_level_change (change : ℝ) : ℝ := change

axiom rise_positive (x : ℝ) : x > 0 → water_level_change x > 0
axiom rise_4m : water_level_change 4 = 4

theorem drop_1m : water_level_change (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_drop_1m_l687_68715


namespace NUMINAMATH_CALUDE_arithmetic_sequence_intersection_l687_68721

/-- Given two arithmetic sequences {a_n} and {b_n}, prove that they intersect at n = 5 -/
theorem arithmetic_sequence_intersection :
  let a : ℕ → ℤ := λ n => 2 + 3 * (n - 1)
  let b : ℕ → ℤ := λ n => -2 + 4 * (n - 1)
  ∃! n : ℕ, a n = b n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_intersection_l687_68721


namespace NUMINAMATH_CALUDE_max_bus_stop_distance_l687_68756

/-- Represents the problem of finding the maximum distance between bus stops -/
theorem max_bus_stop_distance (peter_speed : ℝ) (bus_speed : ℝ) (sight_distance : ℝ) :
  peter_speed > 0 →
  bus_speed = 3 * peter_speed →
  sight_distance = 0.8 →
  ∃ (max_distance : ℝ),
    max_distance = 0.6 ∧
    ∀ (d : ℝ), 0 < d ∧ d ≤ max_distance →
      (∀ (x : ℝ), 0 ≤ x ∧ x ≤ d →
        (x + sight_distance) / peter_speed ≤ (d - x) / bus_speed ∨
        (2 * x + sight_distance) / peter_speed ≤ d / bus_speed) :=
by sorry

end NUMINAMATH_CALUDE_max_bus_stop_distance_l687_68756


namespace NUMINAMATH_CALUDE_permutation_calculation_l687_68758

/-- Definition of permutation notation -/
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Theorem stating that A₆² A₄² equals 360 -/
theorem permutation_calculation : A 6 2 * A 4 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_permutation_calculation_l687_68758


namespace NUMINAMATH_CALUDE_garden_tiles_count_l687_68787

/-- Represents a square garden covered with square tiles -/
structure SquareGarden where
  side_length : ℕ
  diagonal_tiles : ℕ

/-- The total number of tiles in a square garden -/
def total_tiles (garden : SquareGarden) : ℕ :=
  garden.side_length * garden.side_length

/-- The number of tiles on both diagonals of a square garden -/
def diagonal_tiles_count (garden : SquareGarden) : ℕ :=
  2 * garden.side_length - 1

theorem garden_tiles_count (garden : SquareGarden) 
  (h : diagonal_tiles_count garden = 25) : 
  total_tiles garden = 169 := by
  sorry

end NUMINAMATH_CALUDE_garden_tiles_count_l687_68787


namespace NUMINAMATH_CALUDE_constant_term_expansion_l687_68708

/-- The constant term in the expansion of (2x + 1/x)^6 -/
def constantTerm : ℕ := 160

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The general term of the expansion -/
def generalTerm (r : ℕ) : ℚ :=
  2^(6 - r) * binomial 6 r * (1 : ℚ)

theorem constant_term_expansion :
  constantTerm = generalTerm 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l687_68708


namespace NUMINAMATH_CALUDE_intersection_condition_union_condition_l687_68751

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a-1)*x + (a^2-5) = 0}

-- Part 1: When A ∩ B = {2}, find the value of a
theorem intersection_condition (a : ℝ) : A ∩ B a = {2} → a = -5 ∨ a = 1 := by
  sorry

-- Part 2: When A ∪ B = A, find the range of a
theorem union_condition (a : ℝ) : A ∪ B a = A → a > 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_union_condition_l687_68751
