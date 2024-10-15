import Mathlib

namespace NUMINAMATH_CALUDE_new_average_weight_l674_67416

def original_players : ℕ := 7
def original_average_weight : ℝ := 112
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight := original_players * original_average_weight
  let total_new_weight := total_original_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  (total_new_weight / new_total_players : ℝ) = 106 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l674_67416


namespace NUMINAMATH_CALUDE_angles_do_not_determine_triangle_uniquely_l674_67479

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  sum_angles : α + β + γ = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define a function to check if two triangles have the same angles
def SameAngles (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem statement
theorem angles_do_not_determine_triangle_uniquely :
  ∃ (t1 t2 : Triangle), SameAngles t1 t2 ∧ t1 ≠ t2 := by sorry

end NUMINAMATH_CALUDE_angles_do_not_determine_triangle_uniquely_l674_67479


namespace NUMINAMATH_CALUDE_shaded_area_is_9_sqrt_3_l674_67449

-- Define the square
structure Square where
  side : ℝ
  height : ℝ
  bottomRight : ℝ × ℝ

-- Define the equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  height : ℝ
  bottomLeft : ℝ × ℝ

-- Define the problem setup
def problemSetup (s : Square) (t : EquilateralTriangle) : Prop :=
  s.side = 14 ∧
  t.side = 18 ∧
  s.height = t.height ∧
  s.bottomRight = (14, 0) ∧
  t.bottomLeft = (14, 0)

-- Define the shaded area
def shadedArea (s : Square) (t : EquilateralTriangle) : ℝ := sorry

-- Theorem statement
theorem shaded_area_is_9_sqrt_3 (s : Square) (t : EquilateralTriangle) :
  problemSetup s t → shadedArea s t = 9 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_9_sqrt_3_l674_67449


namespace NUMINAMATH_CALUDE_expression_value_l674_67457

theorem expression_value : 
  let x : ℝ := 2
  2 * x^2 + 3 * x^2 = 20 := by sorry

end NUMINAMATH_CALUDE_expression_value_l674_67457


namespace NUMINAMATH_CALUDE_polynomial_factorization_l674_67493

theorem polynomial_factorization (x : ℝ) :
  (x^4 - 4*x^2 + 1) * (x^4 + 3*x^2 + 1) + 10*x^4 = 
  (x + 1)^2 * (x - 1)^2 * (x^2 + x + 1) * (x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l674_67493


namespace NUMINAMATH_CALUDE_marker_cost_l674_67469

/-- The cost of notebooks and markers -/
theorem marker_cost (n m : ℝ) 
  (eq1 : 3 * n + 4 * m = 5.70)
  (eq2 : 5 * n + 2 * m = 4.90) : 
  m = 0.9857 := by
sorry

end NUMINAMATH_CALUDE_marker_cost_l674_67469


namespace NUMINAMATH_CALUDE_journey_fraction_l674_67404

theorem journey_fraction (total_journey : ℝ) (bus_fraction : ℝ) (foot_distance : ℝ)
  (h1 : total_journey = 130)
  (h2 : bus_fraction = 17 / 20)
  (h3 : foot_distance = 6.5) :
  (total_journey - bus_fraction * total_journey - foot_distance) / total_journey = 1 / 10 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_l674_67404


namespace NUMINAMATH_CALUDE_basketball_only_count_l674_67427

theorem basketball_only_count (total : ℕ) (basketball : ℕ) (table_tennis : ℕ) (neither : ℕ)
  (h_total : total = 30)
  (h_basketball : basketball = 15)
  (h_table_tennis : table_tennis = 10)
  (h_neither : neither = 8)
  (h_sum : total = basketball + table_tennis - (basketball + table_tennis - total + neither) + neither) :
  basketball - (basketball + table_tennis - total + neither) = 12 := by
  sorry

end NUMINAMATH_CALUDE_basketball_only_count_l674_67427


namespace NUMINAMATH_CALUDE_total_strings_needed_johns_total_strings_l674_67438

theorem total_strings_needed (num_basses : ℕ) (strings_per_bass : ℕ) 
  (strings_per_guitar : ℕ) (strings_per_8string_guitar : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string_guitars := num_guitars - 3
  let bass_strings := num_basses * strings_per_bass
  let guitar_strings := num_guitars * strings_per_guitar
  let eight_string_guitar_strings := num_8string_guitars * strings_per_8string_guitar
  bass_strings + guitar_strings + eight_string_guitar_strings

theorem johns_total_strings : 
  total_strings_needed 3 4 6 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_total_strings_needed_johns_total_strings_l674_67438


namespace NUMINAMATH_CALUDE_hannah_strawberry_harvest_l674_67488

/-- Hannah's strawberry harvest problem -/
theorem hannah_strawberry_harvest :
  let daily_harvest : ℕ := 5
  let days_in_april : ℕ := 30
  let given_away : ℕ := 20
  let stolen : ℕ := 30
  let total_harvested : ℕ := daily_harvest * days_in_april
  let remaining_after_giving : ℕ := total_harvested - given_away
  let final_count : ℕ := remaining_after_giving - stolen
  final_count = 100 := by sorry

end NUMINAMATH_CALUDE_hannah_strawberry_harvest_l674_67488


namespace NUMINAMATH_CALUDE_power_equation_solution_l674_67460

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 → n = 21 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l674_67460


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l674_67426

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Define the point that the line must pass through
def point : ℝ × ℝ := (1, 3)

-- Define the equation of the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- State the theorem
theorem perpendicular_line_through_point :
  (perpendicular_line point.1 point.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (y - point.2) = -(x - point.1) * (1 / (2 : ℝ))) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l674_67426


namespace NUMINAMATH_CALUDE_product_upper_bound_l674_67425

theorem product_upper_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) : a * b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_product_upper_bound_l674_67425


namespace NUMINAMATH_CALUDE_inequality_solution_set_l674_67422

theorem inequality_solution_set :
  {x : ℝ | 4 + 2*x > -6} = {x : ℝ | x > -5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l674_67422


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l674_67492

theorem quadratic_solution_sum (x p q : ℝ) : 
  (5 * x^2 + 7 = 4 * x - 12) →
  (∃ (i : ℂ), x = p + q * i ∨ x = p - q * i) →
  p + q^2 = 101 / 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l674_67492


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l674_67474

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) : 
  (a * x^2 + b * x + c = 0) → ¬(Odd a ∧ Odd b ∧ Odd c) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l674_67474


namespace NUMINAMATH_CALUDE_three_digit_number_rearrangement_l674_67497

def digit_sum (n : ℕ) : ℕ := n / 100 + (n / 10) % 10 + n % 10

def rearrangement_sum (abc : ℕ) : ℕ :=
  let a := abc / 100
  let b := (abc / 10) % 10
  let c := abc % 10
  (a * 100 + c * 10 + b) +
  (b * 100 + c * 10 + a) +
  (b * 100 + a * 10 + c) +
  (c * 100 + a * 10 + b) +
  (c * 100 + b * 10 + a)

theorem three_digit_number_rearrangement (abc : ℕ) :
  abc ≥ 100 ∧ abc < 1000 ∧ rearrangement_sum abc = 2670 → abc = 528 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_rearrangement_l674_67497


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l674_67491

theorem complex_fraction_simplification :
  let z : ℂ := (1 + I) / (3 - 4*I)
  z = -(1/25) + (7/25)*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l674_67491


namespace NUMINAMATH_CALUDE_possible_m_values_l674_67413

theorem possible_m_values (m : ℝ) : 
  (2 ∈ ({m - 1, 2 * m, m^2 - 1} : Set ℝ)) → 
  (m ∈ ({3, Real.sqrt 3, -Real.sqrt 3} : Set ℝ)) := by
sorry

end NUMINAMATH_CALUDE_possible_m_values_l674_67413


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l674_67485

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x : ℝ | x^2 - 3*x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l674_67485


namespace NUMINAMATH_CALUDE_salt_price_reduction_l674_67444

/-- Given a 20% reduction in the price of salt allows 10 kgs more to be purchased for Rs. 400,
    prove that the original price per kg of salt was Rs. 10. -/
theorem salt_price_reduction (P : ℝ) 
  (h1 : P > 0) -- The price is positive
  (h2 : ∃ (X : ℝ), 400 / P = X ∧ 400 / (0.8 * P) = X + 10) -- Condition from the problem
  : P = 10 := by
  sorry

end NUMINAMATH_CALUDE_salt_price_reduction_l674_67444


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l674_67417

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 246 → ¬(∃ k : ℤ, n^3 + 150 = k * (n + 12)) ∧
  ∃ k : ℤ, 246^3 + 150 = k * (246 + 12) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l674_67417


namespace NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l674_67499

/-- Proves that for a rectangular hall with width equal to half its length
    and an area of 800 square meters, the difference between its length
    and width is 20 meters. -/
theorem rectangular_hall_dimension_difference
  (length width : ℝ)
  (h1 : width = length / 2)
  (h2 : length * width = 800) :
  length - width = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimension_difference_l674_67499


namespace NUMINAMATH_CALUDE_weights_representation_l674_67406

def weights : List ℤ := [1, 3, 9, 27]

def is_representable (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧
    n = 27*a + 9*b + 3*c + d

theorem weights_representation :
  ∀ n : ℤ, 0 ≤ n → n < 41 → is_representable n :=
by sorry

end NUMINAMATH_CALUDE_weights_representation_l674_67406


namespace NUMINAMATH_CALUDE_regular_star_polygon_n_value_l674_67471

/-- An n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  edgeCount : ℕ
  edgeCount_eq : edgeCount = 2 * n
  angleA : ℝ
  angleB : ℝ
  angle_difference : angleB - angleA = 15

/-- The theorem stating that for a regular star polygon with the given properties, n must be 24 -/
theorem regular_star_polygon_n_value (star : RegularStarPolygon) : star.n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_star_polygon_n_value_l674_67471


namespace NUMINAMATH_CALUDE_cut_cube_theorem_l674_67445

/-- Given a cube cut into equal smaller cubes, this function calculates
    the total number of smaller cubes created. -/
def total_smaller_cubes (n : ℕ) : ℕ := (n + 1)^3

/-- This function calculates the number of smaller cubes painted on exactly 2 faces. -/
def cubes_with_two_painted_faces (n : ℕ) : ℕ := 12 * (n - 1)

/-- Theorem stating that when a cube is cut such that 12 smaller cubes are painted
    on exactly 2 faces, the total number of smaller cubes is 27. -/
theorem cut_cube_theorem :
  ∃ n : ℕ, cubes_with_two_painted_faces n = 12 ∧ total_smaller_cubes n = 27 :=
sorry

end NUMINAMATH_CALUDE_cut_cube_theorem_l674_67445


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l674_67475

theorem rectangle_dimension_change (L W x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let new_area := L * (1 + x / 100) * W * (1 - x / 100)
  let original_area := L * W
  new_area = original_area * (1 + 4 / 100) →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l674_67475


namespace NUMINAMATH_CALUDE_median_mean_difference_l674_67431

/-- The distribution of scores on an algebra quiz -/
structure ScoreDistribution where
  score_70 : ℝ
  score_80 : ℝ
  score_90 : ℝ
  score_100 : ℝ

/-- The properties of the score distribution -/
def valid_distribution (d : ScoreDistribution) : Prop :=
  d.score_70 = 0.1 ∧
  d.score_80 = 0.35 ∧
  d.score_90 = 0.3 ∧
  d.score_100 = 0.25 ∧
  d.score_70 + d.score_80 + d.score_90 + d.score_100 = 1

/-- Calculate the mean score -/
def mean_score (d : ScoreDistribution) : ℝ :=
  70 * d.score_70 + 80 * d.score_80 + 90 * d.score_90 + 100 * d.score_100

/-- The median score -/
def median_score : ℝ := 90

/-- The main theorem: the difference between median and mean is 3 -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h : valid_distribution d) : median_score - mean_score d = 3 := by
  sorry

end NUMINAMATH_CALUDE_median_mean_difference_l674_67431


namespace NUMINAMATH_CALUDE_oil_mixture_volume_constant_oil_problem_solution_l674_67432

/-- Represents the properties of an oil mixture -/
structure OilMixture where
  V_hot : ℝ  -- Volume of hot oil
  V_cold : ℝ  -- Volume of cold oil
  T_hot : ℝ  -- Temperature of hot oil
  T_cold : ℝ  -- Temperature of cold oil
  beta : ℝ  -- Coefficient of thermal expansion

/-- Calculates the final volume of an oil mixture at thermal equilibrium -/
def final_volume (mix : OilMixture) : ℝ :=
  mix.V_hot + mix.V_cold

/-- Theorem stating that the final volume of the oil mixture at thermal equilibrium
    is equal to the sum of the initial volumes -/
theorem oil_mixture_volume_constant (mix : OilMixture) :
  final_volume mix = mix.V_hot + mix.V_cold :=
by sorry

/-- Specific instance of the oil mixture problem -/
def oil_problem : OilMixture :=
  { V_hot := 2
  , V_cold := 1
  , T_hot := 100
  , T_cold := 20
  , beta := 2e-3
  }

/-- The final volume of the specific oil mixture problem is 3 liters -/
theorem oil_problem_solution :
  final_volume oil_problem = 3 :=
by sorry

end NUMINAMATH_CALUDE_oil_mixture_volume_constant_oil_problem_solution_l674_67432


namespace NUMINAMATH_CALUDE_odd_symmetric_latin_square_diagonal_l674_67454

/-- A square matrix of size n × n filled with integers from 1 to n -/
def LatinSquare (n : ℕ) := Matrix (Fin n) (Fin n) (Fin n)

/-- Predicate to check if a LatinSquare has all numbers from 1 to n in each row and column -/
def is_valid_latin_square (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, (∃ k : Fin n, A i k = j) ∧ (∃ k : Fin n, A k j = i)

/-- Predicate to check if a LatinSquare is symmetric -/
def is_symmetric (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, A i j = A j i

/-- Predicate to check if all numbers from 1 to n appear on the main diagonal -/
def all_on_diagonal (A : LatinSquare n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, A i i = k

/-- Theorem stating that for odd n, a valid symmetric Latin square has all numbers on its diagonal -/
theorem odd_symmetric_latin_square_diagonal (n : ℕ) (hn : Odd n) (A : LatinSquare n)
  (hvalid : is_valid_latin_square A) (hsym : is_symmetric A) :
  all_on_diagonal A :=
sorry

end NUMINAMATH_CALUDE_odd_symmetric_latin_square_diagonal_l674_67454


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l674_67480

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27 * x^3) 
  (h3 : a - b = x) : 
  a = (x * (1 + Real.sqrt (115 / 3))) / 2 ∨ 
  a = (x * (1 - Real.sqrt (115 / 3))) / 2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l674_67480


namespace NUMINAMATH_CALUDE_land_of_computation_base_l674_67421

/-- Represents a number in base s --/
def BaseS (coeffs : List Nat) (s : Nat) : Nat :=
  coeffs.enum.foldl (fun acc (i, a) => acc + a * s^i) 0

/-- The problem statement --/
theorem land_of_computation_base (s : Nat) : 
  s > 1 → 
  BaseS [0, 5, 5] s + BaseS [0, 2, 4] s = BaseS [0, 0, 1, 1] s → 
  s = 7 := by
sorry

end NUMINAMATH_CALUDE_land_of_computation_base_l674_67421


namespace NUMINAMATH_CALUDE_four_vertices_unique_distances_five_vertices_not_unique_distances_l674_67495

/-- Represents a regular 13-sided polygon -/
structure RegularPolygon13 where
  vertices : Fin 13 → ℝ × ℝ

/-- Calculates the distance between two vertices in a regular 13-sided polygon -/
def distance (p : RegularPolygon13) (v1 v2 : Fin 13) : ℝ := sorry

/-- Checks if all pairwise distances in a set of vertices are unique -/
def all_distances_unique (p : RegularPolygon13) (vs : Finset (Fin 13)) : Prop := sorry

theorem four_vertices_unique_distances (p : RegularPolygon13) :
  ∃ (vs : Finset (Fin 13)), vs.card = 4 ∧ all_distances_unique p vs := sorry

theorem five_vertices_not_unique_distances (p : RegularPolygon13) :
  ¬∃ (vs : Finset (Fin 13)), vs.card = 5 ∧ all_distances_unique p vs := sorry

end NUMINAMATH_CALUDE_four_vertices_unique_distances_five_vertices_not_unique_distances_l674_67495


namespace NUMINAMATH_CALUDE_doctors_visit_cost_is_250_l674_67419

/-- Calculates the cost of a doctor's visit given the following conditions:
  * Number of vaccines needed
  * Cost per vaccine
  * Insurance coverage percentage
  * Cost of the trip
  * Total amount paid by Tom
-/
def doctors_visit_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (insurance_coverage : ℚ) 
                       (trip_cost : ℚ) (total_paid : ℚ) : ℚ :=
  let total_vaccine_cost := num_vaccines * vaccine_cost
  let medical_bills := total_vaccine_cost + (total_paid - trip_cost) / (1 - insurance_coverage)
  medical_bills - total_vaccine_cost

/-- Proves that the cost of the doctor's visit is $250 given the specified conditions -/
theorem doctors_visit_cost_is_250 : 
  doctors_visit_cost 10 45 0.8 1200 1340 = 250 := by
  sorry

end NUMINAMATH_CALUDE_doctors_visit_cost_is_250_l674_67419


namespace NUMINAMATH_CALUDE_james_older_brother_age_is_16_l674_67428

/-- The age of James' older brother given the conditions in the problem -/
def james_older_brother_age (john_current_age : ℕ) : ℕ :=
  let john_age_3_years_ago := john_current_age - 3
  let james_age_in_6_years := john_age_3_years_ago / 2
  let james_current_age := james_age_in_6_years - 6
  james_current_age + 4

/-- Theorem stating that James' older brother's age is 16 -/
theorem james_older_brother_age_is_16 :
  james_older_brother_age 39 = 16 := by
  sorry

end NUMINAMATH_CALUDE_james_older_brother_age_is_16_l674_67428


namespace NUMINAMATH_CALUDE_no_integer_roots_l674_67468

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (a * 0^2 + b * 0 + c))
  (h1 : Odd (a * 1^2 + b * 1 + c)) :
  ∀ t : ℤ, a * t^2 + b * t + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l674_67468


namespace NUMINAMATH_CALUDE_area_of_region_l674_67458

/-- The region defined by the inequality |4x-14| + |3y-9| ≤ 6 -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |4 * p.1 - 14| + |3 * p.2 - 9| ≤ 6}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The area of the region defined by |4x-14| + |3y-9| ≤ 6 is 6 -/
theorem area_of_region : area Region = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_region_l674_67458


namespace NUMINAMATH_CALUDE_tan_half_sum_of_angles_l674_67440

theorem tan_half_sum_of_angles (x y : Real) 
  (h1 : Real.cos x + Real.cos y = 3/5)
  (h2 : Real.sin x + Real.sin y = 1/5) :
  Real.tan ((x + y) / 2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_tan_half_sum_of_angles_l674_67440


namespace NUMINAMATH_CALUDE_intersection_of_lines_l674_67483

/-- Two lines intersect if and only if their slopes are not equal -/
def lines_intersect (m₁ m₂ : ℝ) : Prop := m₁ ≠ m₂

/-- The slope of a line in the form y = mx + b is m -/
def slope_of_line (m : ℝ) : ℝ := m

theorem intersection_of_lines :
  let line1_slope : ℝ := -1  -- slope of x + y - 1 = 0
  let line2_slope : ℝ := 1   -- slope of y = x - 1
  lines_intersect (slope_of_line line1_slope) (slope_of_line line2_slope) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l674_67483


namespace NUMINAMATH_CALUDE_greatest_common_measure_l674_67409

theorem greatest_common_measure (a b c : ℕ) (ha : a = 18000) (hb : b = 50000) (hc : c = 1520) :
  Nat.gcd a (Nat.gcd b c) = 40 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l674_67409


namespace NUMINAMATH_CALUDE_n_div_16_equals_4_pow_8086_l674_67472

theorem n_div_16_equals_4_pow_8086 (n : ℕ) : n = 16^4044 → n / 16 = 4^8086 := by
  sorry

end NUMINAMATH_CALUDE_n_div_16_equals_4_pow_8086_l674_67472


namespace NUMINAMATH_CALUDE_percentage_difference_l674_67462

theorem percentage_difference (x y : ℝ) (P : ℝ) : 
  x = y * 0.9 →                 -- x is 10% less than y
  y = 125 * (1 + P / 100) →     -- y is P% more than 125
  x = 123.75 →                  -- x is equal to 123.75
  P = 10 :=                     -- P is equal to 10
by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l674_67462


namespace NUMINAMATH_CALUDE_find_k_l674_67441

theorem find_k : ∃ (k : ℤ) (m : ℝ), ∀ (n : ℝ), 
  n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l674_67441


namespace NUMINAMATH_CALUDE_range_of_t_l674_67484

/-- Set A definition -/
def A : Set ℝ := {x : ℝ | (x + 8) / (x - 5) ≤ 0}

/-- Set B definition -/
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 1}

/-- Theorem stating the range of t -/
theorem range_of_t (t : ℝ) : 
  (∃ x, x ∈ B t) → -- B is non-empty
  (A ∩ B t = ∅) → -- A and B have no intersection
  t ≥ 4 := by sorry

end NUMINAMATH_CALUDE_range_of_t_l674_67484


namespace NUMINAMATH_CALUDE_initial_profit_percentage_is_correct_l674_67415

/-- Represents the profit percentage as a real number between 0 and 1 -/
def ProfitPercentage : Type := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The cost price of the book -/
def costPrice : ℝ := 300

/-- The additional amount added to the initial selling price -/
def additionalAmount : ℝ := 18

/-- The profit percentage if the book is sold with the additional amount -/
def newProfitPercentage : ProfitPercentage := ⟨0.18, by sorry⟩

/-- Calculate the selling price given a profit percentage -/
def sellingPrice (p : ProfitPercentage) : ℝ :=
  costPrice * (1 + p.val)

/-- The initial profit percentage -/
def initialProfitPercentage : ProfitPercentage := ⟨0.12, by sorry⟩

/-- Theorem stating that the initial profit percentage is correct -/
theorem initial_profit_percentage_is_correct :
  sellingPrice initialProfitPercentage + additionalAmount =
  sellingPrice newProfitPercentage :=
by sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_is_correct_l674_67415


namespace NUMINAMATH_CALUDE_cos_50_minus_tan_40_equals_sqrt_3_l674_67459

theorem cos_50_minus_tan_40_equals_sqrt_3 :
  4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_50_minus_tan_40_equals_sqrt_3_l674_67459


namespace NUMINAMATH_CALUDE_puzzle_assembly_time_l674_67448

-- Define the number of pieces in the puzzle
def puzzle_pieces : ℕ := 121

-- Define the time it takes to assemble the puzzle with the original method
def original_time : ℕ := 120

-- Define the function for the original assembly method (2 pieces per minute)
def original_assembly (t : ℕ) : ℕ := puzzle_pieces - t

-- Define the function for the new assembly method (3 pieces per minute)
def new_assembly (t : ℕ) : ℕ := puzzle_pieces - 2 * t

-- State the theorem
theorem puzzle_assembly_time :
  ∃ (new_time : ℕ), 
    (original_assembly original_time = 1) ∧ 
    (new_assembly new_time = 1) ∧ 
    (new_time = original_time / 2) := by
  sorry

end NUMINAMATH_CALUDE_puzzle_assembly_time_l674_67448


namespace NUMINAMATH_CALUDE_geraint_on_time_speed_l674_67477

/-- The distance Geraint cycles to work in kilometers. -/
def distance : ℝ := sorry

/-- The time in hours that Geraint's journey should take to arrive on time. -/
def on_time : ℝ := sorry

/-- The speed in km/h at which Geraint arrives on time. -/
def on_time_speed : ℝ := sorry

/-- Theorem stating that Geraint's on-time speed is 20 km/h. -/
theorem geraint_on_time_speed : 
  (distance / 15 = on_time + 1/6) →  -- At 15 km/h, he's 10 minutes (1/6 hour) late
  (distance / 30 = on_time - 1/6) →  -- At 30 km/h, he's 10 minutes (1/6 hour) early
  on_time_speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_geraint_on_time_speed_l674_67477


namespace NUMINAMATH_CALUDE_volleyball_team_math_players_l674_67439

theorem volleyball_team_math_players 
  (total_players : ℕ) 
  (physics_players : ℕ) 
  (both_subjects : ℕ) 
  (h1 : total_players = 15)
  (h2 : physics_players = 10)
  (h3 : both_subjects = 4)
  (h4 : physics_players ≤ total_players)
  (h5 : both_subjects ≤ physics_players)
  (h6 : ∀ p, p ∈ (Finset.range total_players) → 
    (p ∈ (Finset.range physics_players) ∨ 
     p ∈ (Finset.range (total_players - physics_players + both_subjects)))) :
  total_players - physics_players + both_subjects = 9 := by
sorry

end NUMINAMATH_CALUDE_volleyball_team_math_players_l674_67439


namespace NUMINAMATH_CALUDE_hotel_expenditure_l674_67408

/-- The total expenditure of a group of men, where most spend a fixed amount and one spends more than the average -/
def total_expenditure (n : ℕ) (m : ℕ) (fixed_spend : ℚ) (extra_spend : ℚ) : ℚ :=
  let avg := (m * fixed_spend + ((m * fixed_spend + extra_spend) / n)) / n
  n * avg

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem hotel_expenditure :
  round_to_nearest (total_expenditure 9 8 3 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l674_67408


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l674_67433

theorem quadratic_equation_roots (x y : ℝ) : 
  x + y = 10 →
  |x - y| = 4 →
  x * y = 21 →
  x^2 - 10*x + 21 = 0 ∧ y^2 - 10*y + 21 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l674_67433


namespace NUMINAMATH_CALUDE_sports_club_membership_l674_67420

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ)
  (h_total : total = 27)
  (h_badminton : badminton = 17)
  (h_tennis : tennis = 19)
  (h_both : both = 11) :
  total - (badminton + tennis - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_membership_l674_67420


namespace NUMINAMATH_CALUDE_binary_repr_24_l674_67494

/-- The binary representation of a natural number -/
def binary_repr (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Theorem: The binary representation of 24 is [false, false, false, true, true] -/
theorem binary_repr_24 : binary_repr 24 = [false, false, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_repr_24_l674_67494


namespace NUMINAMATH_CALUDE_stock_market_value_l674_67400

/-- Calculates the market value of a stock given its dividend rate, yield, and face value. -/
def market_value (dividend_rate : ℚ) (yield : ℚ) (face_value : ℚ) : ℚ :=
  (dividend_rate * face_value / yield) * 100

/-- Theorem stating that a 13% stock yielding 8% with a face value of $100 has a market value of $162.50 -/
theorem stock_market_value :
  let dividend_rate : ℚ := 13 / 100
  let yield : ℚ := 8 / 100
  let face_value : ℚ := 100
  market_value dividend_rate yield face_value = 162.5 := by
  sorry

#eval market_value (13/100) (8/100) 100

end NUMINAMATH_CALUDE_stock_market_value_l674_67400


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l674_67481

theorem quadratic_roots_range (m : ℝ) : 
  (∀ x, x^2 + (m-2)*x + (5-m) = 0 → x > 2) → 
  m ∈ Set.Ioo (-5) (-4) ∪ {-4} :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l674_67481


namespace NUMINAMATH_CALUDE_murtha_pebble_collection_l674_67402

def pebbles_on_day (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * (n - 1) + 2

def total_pebbles (days : ℕ) : ℕ :=
  (List.range days).map pebbles_on_day |>.sum

theorem murtha_pebble_collection :
  total_pebbles 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_murtha_pebble_collection_l674_67402


namespace NUMINAMATH_CALUDE_odd_periodic_sum_zero_l674_67403

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_sum_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_periodic : is_periodic f 2) :
  f 1 + f 4 + f 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_sum_zero_l674_67403


namespace NUMINAMATH_CALUDE_james_final_amount_proof_l674_67405

/-- The amount of money owned by James after paying off Lucas' debt -/
def james_final_amount : ℝ := 170

/-- The total amount owned by Lucas, James, and Ali -/
def total_amount : ℝ := 300

/-- The amount of Lucas' debt -/
def lucas_debt : ℝ := 25

/-- The difference between James' and Ali's initial amounts -/
def james_ali_difference : ℝ := 40

theorem james_final_amount_proof :
  ∃ (ali james lucas : ℝ),
    ali + james + lucas = total_amount ∧
    james = ali + james_ali_difference ∧
    lucas = -lucas_debt ∧
    james - (lucas_debt / 2) = james_final_amount :=
sorry

end NUMINAMATH_CALUDE_james_final_amount_proof_l674_67405


namespace NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_sin_10_l674_67401

theorem sin_20_cos_10_minus_cos_160_sin_10 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_sin_10_l674_67401


namespace NUMINAMATH_CALUDE_inequality_equivalence_l674_67414

theorem inequality_equivalence (x : ℝ) : x / 3 - 2 < 0 ↔ x < 6 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l674_67414


namespace NUMINAMATH_CALUDE_intersection_point_equivalence_l674_67450

theorem intersection_point_equivalence 
  (m n a b : ℝ) 
  (h1 : m * a + 2 * m * b = 5) 
  (h2 : n * a - 2 * n * b = 7) :
  (5 / (2 * m) - a / 2 = b) ∧ (a / 2 - 7 / (2 * n) = b) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_equivalence_l674_67450


namespace NUMINAMATH_CALUDE_graph_shift_up_by_two_l674_67423

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the transformed function g
def g (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem graph_shift_up_by_two :
  ∀ x : ℝ, g x = f x + 2 := by sorry

end NUMINAMATH_CALUDE_graph_shift_up_by_two_l674_67423


namespace NUMINAMATH_CALUDE_interest_for_one_rupee_l674_67464

/-- Given that for 5000 rs, the interest is 200 paise, prove that the interest for 1 rs is 0.04 paise. -/
theorem interest_for_one_rupee (interest_5000 : ℝ) (h : interest_5000 = 200) :
  interest_5000 / 5000 = 0.04 := by
sorry

end NUMINAMATH_CALUDE_interest_for_one_rupee_l674_67464


namespace NUMINAMATH_CALUDE_triangle_side_length_l674_67461

theorem triangle_side_length (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 →
  Real.sin A = Real.sqrt 3 / 2 →
  B = π / 6 →
  b = a * Real.sin B / Real.sin A →
  b = 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l674_67461


namespace NUMINAMATH_CALUDE_doubling_function_m_range_l674_67446

/-- A function f is a doubling function if there exists an interval [a, b] such that f([a, b]) = [2a, 2b] -/
def DoublingFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a < b ∧ Set.image f (Set.Icc a b) = Set.Icc (2*a) (2*b)

/-- The main theorem stating that if ln(e^x + m) is a doubling function, then m is in the open interval (-1/4, 0) -/
theorem doubling_function_m_range :
  ∀ m : ℝ, DoublingFunction (fun x ↦ Real.log (Real.exp x + m)) → m ∈ Set.Ioo (-1/4) 0 :=
by sorry

end NUMINAMATH_CALUDE_doubling_function_m_range_l674_67446


namespace NUMINAMATH_CALUDE_complex_equation_solution_l674_67466

theorem complex_equation_solution (z : ℂ) (a : ℝ) 
  (h1 : Complex.abs z = 2) 
  (h2 : (z - a)^2 = a) : 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l674_67466


namespace NUMINAMATH_CALUDE_statements_b_and_c_are_correct_l674_67470

theorem statements_b_and_c_are_correct (a b c d : ℝ) :
  (ab > 0 ∧ b*c - a*d > 0 → c/a - d/b > 0) ∧
  (a > b ∧ c > d → a - d > b - c) := by
sorry

end NUMINAMATH_CALUDE_statements_b_and_c_are_correct_l674_67470


namespace NUMINAMATH_CALUDE_soda_difference_l674_67435

theorem soda_difference (diet_soda : ℕ) (regular_soda : ℕ) 
  (h1 : diet_soda = 19) (h2 : regular_soda = 60) : 
  regular_soda - diet_soda = 41 := by
  sorry

end NUMINAMATH_CALUDE_soda_difference_l674_67435


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l674_67434

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  let sphere_area := 4 * π * r^2
  let base_area := π * r^2
  let excluded_base_area := (1/4) * base_area
  let hemisphere_curved_area := (1/2) * sphere_area
  hemisphere_curved_area + base_area - excluded_base_area = 275 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l674_67434


namespace NUMINAMATH_CALUDE_dog_fruits_total_l674_67436

/-- Represents the number of fruits eaten by each dog -/
structure DogFruits where
  apples : ℕ
  blueberries : ℕ
  bonnies : ℕ
  cherries : ℕ

/-- The conditions of the problem and the theorem to prove -/
theorem dog_fruits_total (df : DogFruits) : 
  df.apples = 3 * df.blueberries →
  df.blueberries = (3 * df.bonnies) / 4 →
  df.cherries = 5 * df.apples →
  df.bonnies = 60 →
  df.apples + df.blueberries + df.bonnies + df.cherries = 915 := by
  sorry

#check dog_fruits_total

end NUMINAMATH_CALUDE_dog_fruits_total_l674_67436


namespace NUMINAMATH_CALUDE_money_lent_to_B_l674_67498

/-- Proves that the amount lent to B is 4000, given the problem conditions --/
theorem money_lent_to_B (total : ℕ) (rate_A rate_B : ℚ) (years : ℕ) (interest_diff : ℕ) :
  total = 10000 →
  rate_A = 15 / 100 →
  rate_B = 18 / 100 →
  years = 2 →
  interest_diff = 360 →
  ∃ (amount_A amount_B : ℕ),
    amount_A + amount_B = total ∧
    amount_A * rate_A * years = (amount_B * rate_B * years + interest_diff) ∧
    amount_B = 4000 :=
by sorry

end NUMINAMATH_CALUDE_money_lent_to_B_l674_67498


namespace NUMINAMATH_CALUDE_compare_cubic_and_quadratic_diff_l674_67486

theorem compare_cubic_and_quadratic_diff (a b : ℝ) :
  (a ≥ b → a^3 - b^3 ≥ a*b^2 - a^2*b) ∧
  (a < b → a^3 - b^3 ≤ a*b^2 - a^2*b) :=
by sorry

end NUMINAMATH_CALUDE_compare_cubic_and_quadratic_diff_l674_67486


namespace NUMINAMATH_CALUDE_x1_x2_ratio_lt_ae_l674_67451

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / a - Real.exp x

theorem x1_x2_ratio_lt_ae (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) : 
  x₁ / x₂ < a * Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_x1_x2_ratio_lt_ae_l674_67451


namespace NUMINAMATH_CALUDE_system_solution_l674_67411

theorem system_solution (p q u v : ℝ) : 
  (p * u + q * v = 2 * (p^2 - q^2)) ∧ 
  (v / (p - q) - u / (p + q) = (p^2 + q^2) / (p * q)) →
  ((p * q * (p^2 - q^2) ≠ 0 ∧ q ≠ 1 + Real.sqrt 2 ∧ q ≠ 1 - Real.sqrt 2) →
    (u = (p^2 - q^2) / p ∧ v = (p^2 - q^2) / q)) ∧
  ((u ≠ 0 ∧ v ≠ 0 ∧ u^2 ≠ v^2) →
    (p = u * v^2 / (v^2 - u^2) ∧ q = u^2 * v / (v^2 - u^2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l674_67411


namespace NUMINAMATH_CALUDE_floor_ceil_sum_seven_l674_67447

theorem floor_ceil_sum_seven (x : ℝ) : 
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_seven_l674_67447


namespace NUMINAMATH_CALUDE_simplify_expression_l674_67410

theorem simplify_expression (a b c : ℝ) (h1 : 1 - a * b ≠ 0) (h2 : 1 + c * a ≠ 0) :
  ((a + b) / (1 - a * b) + (c - a) / (1 + c * a)) / (1 - ((a + b) / (1 - a * b) * (c - a) / (1 + c * a))) =
  (b + c) / (1 - b * c) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l674_67410


namespace NUMINAMATH_CALUDE_sharon_wants_254_supplies_l674_67489

/-- Calculates the total number of kitchen supplies Sharon wants to buy -/
def sharons_kitchen_supplies (angela_pots : ℕ) : ℕ :=
  let angela_plates := 6 + 3 * angela_pots
  let angela_cutlery := angela_plates / 2
  let sharon_pots := angela_pots / 2
  let sharon_plates := 3 * angela_plates - 20
  let sharon_cutlery := 2 * angela_cutlery
  sharon_pots + sharon_plates + sharon_cutlery

/-- Theorem stating that Sharon wants to buy 254 kitchen supplies -/
theorem sharon_wants_254_supplies : sharons_kitchen_supplies 20 = 254 := by
  sorry

end NUMINAMATH_CALUDE_sharon_wants_254_supplies_l674_67489


namespace NUMINAMATH_CALUDE_rational_x_y_l674_67455

theorem rational_x_y (x y : ℝ) 
  (h : ∀ (p q : ℕ), Prime p → Prime q → Odd p → Odd q → p ≠ q → 
    ∃ (r : ℚ), (x^p + y^q : ℝ) = (r : ℝ)) : 
  ∃ (a b : ℚ), (x = (a : ℝ) ∧ y = (b : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_rational_x_y_l674_67455


namespace NUMINAMATH_CALUDE_power_of_three_equality_l674_67429

theorem power_of_three_equality (n : ℕ) : 3^n = 27 * 9^2 * (81^3) / 3^4 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equality_l674_67429


namespace NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l674_67476

theorem least_five_digit_divisible_by_12_15_18 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ 
    12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n →
    10080 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_divisible_by_12_15_18_l674_67476


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l674_67424

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 6) = 10 → x = 106 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l674_67424


namespace NUMINAMATH_CALUDE_farthest_poles_distance_l674_67412

/-- The number of utility poles -/
def num_poles : ℕ := 45

/-- The interval between each pole in meters -/
def interval : ℕ := 60

/-- The distance between the first and last pole in kilometers -/
def distance : ℚ := 2.64

theorem farthest_poles_distance :
  (((num_poles - 1) * interval) : ℚ) / 1000 = distance := by sorry

end NUMINAMATH_CALUDE_farthest_poles_distance_l674_67412


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l674_67418

theorem geometric_sequence_first_term (a b c : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ 16 = c * r ∧ 32 = 16 * r) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l674_67418


namespace NUMINAMATH_CALUDE_kamal_math_marks_l674_67478

/-- Represents a student's marks in various subjects -/
structure StudentMarks where
  english : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  mathematics : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Kamal's marks and average, his Mathematics marks must be 60 -/
theorem kamal_math_marks (kamal : StudentMarks) 
  (h1 : kamal.english = 76)
  (h2 : kamal.physics = 72)
  (h3 : kamal.chemistry = 65)
  (h4 : kamal.biology = 82)
  (h5 : kamal.average = 71)
  (h6 : kamal.total_subjects = 5) :
  kamal.mathematics = 60 := by
  sorry

end NUMINAMATH_CALUDE_kamal_math_marks_l674_67478


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l674_67453

/-- Given points A, B, and C on the inverse proportion function y = 3/x,
    prove that y₂ < y₁ < y₃ -/
theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 3 / (-5) →
  y₂ = 3 / (-3) →
  y₃ = 3 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l674_67453


namespace NUMINAMATH_CALUDE_equation_solutions_l674_67487

theorem equation_solutions : 
  ∀ n m : ℕ, m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ 
  ((n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) ∨ (n = 6 ∧ m = 54) ∨ (n = 6 ∧ m = 27)) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l674_67487


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l674_67473

theorem triangle_side_lengths : 
  let S := {(a, b, c) : ℕ × ℕ × ℕ | 
    a ≤ b ∧ b ≤ c ∧ 
    b^2 = a * c ∧ 
    (a = 100 ∨ c = 100)}
  S = {(49,70,100), (64,80,100), (81,90,100), (100,100,100), 
       (100,110,121), (100,120,144), (100,130,169), (100,140,196), 
       (100,150,225), (100,160,256)} := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l674_67473


namespace NUMINAMATH_CALUDE_problem_1_l674_67442

theorem problem_1 : (1/2)⁻¹ - Real.tan (π/4) + |Real.sqrt 2 - 1| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l674_67442


namespace NUMINAMATH_CALUDE_luke_score_l674_67463

/-- A trivia game where a player gains points each round -/
structure TriviaGame where
  points_per_round : ℕ
  num_rounds : ℕ

/-- Calculate the total points scored in a trivia game -/
def total_points (game : TriviaGame) : ℕ :=
  game.points_per_round * game.num_rounds

/-- Luke's trivia game -/
def luke_game : TriviaGame :=
  { points_per_round := 3
    num_rounds := 26 }

/-- Theorem: Luke scored 78 points in the trivia game -/
theorem luke_score : total_points luke_game = 78 := by
  sorry

end NUMINAMATH_CALUDE_luke_score_l674_67463


namespace NUMINAMATH_CALUDE_max_value_4tau_minus_n_l674_67496

/-- τ(n) is the number of positive divisors of n -/
def τ (n : ℕ+) : ℕ := (Nat.divisors n.val).card

/-- The maximum value of 4τ(n) - n over all positive integers n is 12 -/
theorem max_value_4tau_minus_n :
  (∀ n : ℕ+, (4 * τ n : ℤ) - n.val ≤ 12) ∧
  (∃ n : ℕ+, (4 * τ n : ℤ) - n.val = 12) :=
sorry

end NUMINAMATH_CALUDE_max_value_4tau_minus_n_l674_67496


namespace NUMINAMATH_CALUDE_smith_family_mean_age_l674_67452

def smith_family_ages : List ℕ := [5, 5, 5, 12, 13, 16]

theorem smith_family_mean_age :
  (smith_family_ages.sum : ℚ) / smith_family_ages.length = 9.33 := by
  sorry

end NUMINAMATH_CALUDE_smith_family_mean_age_l674_67452


namespace NUMINAMATH_CALUDE_derivative_symmetry_l674_67430

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Theorem statement
theorem derivative_symmetry (a b c : ℝ) :
  f' a b 1 = 2 → f' a b (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetry_l674_67430


namespace NUMINAMATH_CALUDE_solution_of_equation_l674_67467

theorem solution_of_equation (z : ℂ) : 
  (z^6 - 6*z^4 + 9*z^2 = 0) ↔ (z = -Real.sqrt 3 ∨ z = 0 ∨ z = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l674_67467


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l674_67407

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter p of the parabola is equal to 1. -/
theorem parabola_hyperbola_intersection (p a b : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → a > 0 → b > 0 → x₀ ≠ 0 →
  y₀^2 = 2 * p * x₀ →  -- Point A satisfies parabola equation
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- Point A satisfies hyperbola equation
  y₀ = 2 * x₀ →  -- Point A is on the asymptote y = 2x
  (x₀ - 0)^2 + y₀^2 = p^4 →  -- Distance from A to parabola's axis of symmetry is p²
  (a^2 + b^2) / a^2 = 5 →  -- Eccentricity of hyperbola is √5
  p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l674_67407


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l674_67490

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (vertex_angle : ℝ)
  (base_angle : ℝ)
  (is_isosceles : base_angle = base_angle)

-- Define the exterior angle
def exterior_angle (t : IsoscelesTriangle) : ℝ := 100

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (t : IsoscelesTriangle) 
  (h_exterior : exterior_angle t = 100) :
  t.vertex_angle = 20 ∨ t.vertex_angle = 80 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l674_67490


namespace NUMINAMATH_CALUDE_simplify_expression_l674_67456

theorem simplify_expression (x y z : ℝ) : (x - (y - z)) - ((x - y) - z) = 2 * z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l674_67456


namespace NUMINAMATH_CALUDE_otimes_composition_l674_67482

-- Define the ⊗ operation
def otimes (x y : ℝ) : ℝ := x^3 - y

-- Theorem statement
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l674_67482


namespace NUMINAMATH_CALUDE_not_always_equal_distribution_l674_67465

/-- Represents the state of the pies on plates -/
structure PieState where
  numPlates : Nat
  totalPies : Nat
  blackPies : Nat
  whitePies : Nat

/-- Represents a move in the game -/
inductive Move
  | transfer : Nat → Move

/-- Checks if a pie state is valid -/
def isValidState (state : PieState) : Prop :=
  state.numPlates = 20 ∧
  state.totalPies = 40 ∧
  state.blackPies + state.whitePies = state.totalPies

/-- Checks if a pie state has equal distribution -/
def hasEqualDistribution (state : PieState) : Prop :=
  state.blackPies = state.whitePies

/-- Applies a move to a pie state -/
def applyMove (state : PieState) (move : Move) : PieState :=
  match move with
  | Move.transfer n => 
      { state with 
        blackPies := state.blackPies + n,
        whitePies := state.whitePies - n
      }

/-- Theorem: It's not always possible to achieve equal distribution -/
theorem not_always_equal_distribution :
  ∃ (initialState : PieState),
    isValidState initialState ∧
    ∀ (moves : List Move),
      ¬hasEqualDistribution (moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_not_always_equal_distribution_l674_67465


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l674_67437

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 5 → x > 3) ∧ 
  (∃ x : ℝ, x > 3 ∧ ¬(x > 5)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l674_67437


namespace NUMINAMATH_CALUDE_chess_tournament_games_l674_67443

/-- The number of games in a chess tournament -/
def tournament_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays
    four times with each opponent, the total number of games is 1740 -/
theorem chess_tournament_games :
  tournament_games 30 4 = 1740 := by
  sorry

#eval tournament_games 30 4

end NUMINAMATH_CALUDE_chess_tournament_games_l674_67443
