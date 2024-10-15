import Mathlib

namespace NUMINAMATH_CALUDE_parabola_properties_l1462_146252

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 4)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define point R
def R : ℝ × ℝ := (4, 6)

-- Define the property that R is the midpoint of PQ
def R_is_midpoint (P Q : ℝ × ℝ) : Prop :=
  R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2

-- Define point A as the intersection of tangents
def A : ℝ × ℝ := (4, -2)

-- Theorem statement
theorem parabola_properties
  (P Q : ℝ × ℝ)
  (hP : parabola P.1 P.2)
  (hQ : parabola Q.1 Q.2)
  (hl_P : line_l P.1 P.2)
  (hl_Q : line_l Q.1 Q.2)
  (hR : R_is_midpoint P Q) :
  (∃ (AF : ℝ), AF ≠ 4 * Real.sqrt 2 ∧ AF = Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2)) ∧
  (∃ (PQ : ℝ), PQ = 12 ∧ PQ = Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
  (¬ ((P.2 - Q.2) * (A.1 - focus.1) = -(P.1 - Q.1) * (A.2 - focus.2))) ∧
  (∃ (center : ℝ × ℝ), 
    (center.1 - P.1)^2 + (center.2 - P.2)^2 = (center.1 - Q.1)^2 + (center.2 - Q.2)^2 ∧
    (center.1 - A.1)^2 + (center.2 - A.2)^2 = (center.1 - P.1)^2 + (center.2 - P.2)^2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1462_146252


namespace NUMINAMATH_CALUDE_system_solution_existence_l1462_146218

/-- The system of equations has at least one solution if and only if 0.5 ≤ a ≤ 2 -/
theorem system_solution_existence (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (|x - 0.5| + |y| - a) / (Real.sqrt 3 * y - x) = 0) ↔ 
  0.5 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l1462_146218


namespace NUMINAMATH_CALUDE_ramsey_type_theorem_l1462_146202

theorem ramsey_type_theorem (n r : ℕ) (hn : n > 0) (hr : r > 0) :
  ∃ m : ℕ, m > 0 ∧
  (∀ (A : Fin r → Set ℕ),
    (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) →
    (⋃ i : Fin r, A i) = Finset.range m →
    ∃ (i : Fin r) (a b : ℕ), a ∈ A i ∧ b ∈ A i ∧ b < a ∧ a ≤ (n + 1) * b / n) ∧
  (∀ k : ℕ, 0 < k → k < m →
    ∃ (A : Fin r → Set ℕ),
      (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) ∧
      (⋃ i : Fin r, A i) = Finset.range k ∧
      ∀ (i : Fin r) (a b : ℕ), a ∈ A i → b ∈ A i → b < a → a > (n + 1) * b / n) ∧
  m = (n + 1) * r :=
by sorry

end NUMINAMATH_CALUDE_ramsey_type_theorem_l1462_146202


namespace NUMINAMATH_CALUDE_multiply_subtract_difference_l1462_146275

theorem multiply_subtract_difference (x : ℝ) (h : x = 13) : 3 * x - (36 - x) = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_difference_l1462_146275


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1462_146259

theorem sum_of_x_and_y (x y : ℝ) (hxy : x ≠ y)
  (det1 : Matrix.det ![![2, 5, 10], ![4, x, y], ![4, y, x]] = 0)
  (det2 : Matrix.det ![![x, y], ![y, x]] = 16) :
  x + y = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1462_146259


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1462_146291

theorem simplify_complex_fraction :
  1 / ((1 / (Real.sqrt 3 + 1)) + (2 / (Real.sqrt 5 - 1))) = 
  (Real.sqrt 3 + 2 * Real.sqrt 5 - 1) / (2 + 4 * Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1462_146291


namespace NUMINAMATH_CALUDE_faulty_engine_sampling_l1462_146256

/-- Given a set of 33 items where 8 are faulty, this theorem proves:
    1. The probability of identifying all faulty items by sampling 32 items
    2. The expected number of samplings required to identify all faulty items -/
theorem faulty_engine_sampling (n : Nat) (k : Nat) (h1 : n = 33) (h2 : k = 8) :
  let p := Nat.choose (n - 1) (k - 1) / Nat.choose n k
  let e := (n * k) / (n - k + 1)
  (p = 25 / 132) ∧ (e = 272 / 9) := by
  sorry

#check faulty_engine_sampling

end NUMINAMATH_CALUDE_faulty_engine_sampling_l1462_146256


namespace NUMINAMATH_CALUDE_milkshake_cost_l1462_146205

theorem milkshake_cost (initial_amount : ℕ) (hamburger_cost : ℕ) (num_hamburgers : ℕ) (num_milkshakes : ℕ) (remaining_amount : ℕ) :
  initial_amount = 120 →
  hamburger_cost = 4 →
  num_hamburgers = 8 →
  num_milkshakes = 6 →
  remaining_amount = 70 →
  ∃ (milkshake_cost : ℕ), 
    initial_amount - (hamburger_cost * num_hamburgers) - (milkshake_cost * num_milkshakes) = remaining_amount ∧
    milkshake_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_milkshake_cost_l1462_146205


namespace NUMINAMATH_CALUDE_paper_length_proof_l1462_146221

theorem paper_length_proof (cube_volume : ℝ) (paper_width : ℝ) (inches_per_foot : ℝ) :
  cube_volume = 8 →
  paper_width = 72 →
  inches_per_foot = 12 →
  ∃ (paper_length : ℝ),
    paper_length * paper_width = (cube_volume^(1/3) * inches_per_foot)^2 ∧
    paper_length = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paper_length_proof_l1462_146221


namespace NUMINAMATH_CALUDE_function_difference_theorem_l1462_146232

theorem function_difference_theorem (m : ℚ) : 
  let f : ℚ → ℚ := λ x => 4 * x^2 - 3 * x + 5
  let g : ℚ → ℚ := λ x => 2 * x^2 - m * x + 8
  (f 5 - g 5 = 15) → m = -17/5 := by
sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l1462_146232


namespace NUMINAMATH_CALUDE_percentage_increase_l1462_146222

theorem percentage_increase (x : ℝ) (h : x = 78.4) : 
  (x - 70) / 70 * 100 = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l1462_146222


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1462_146264

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = I - 3 → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1462_146264


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1462_146208

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2018*x + n = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1462_146208


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1462_146283

/-- The constant term in the expansion of (3x^2 + 2/x)^8 -/
def constant_term : ℕ :=
  (Nat.choose 8 4) * (3^4) * (2^4)

/-- Theorem stating that the constant term in the expansion of (3x^2 + 2/x)^8 is 90720 -/
theorem constant_term_expansion :
  constant_term = 90720 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1462_146283


namespace NUMINAMATH_CALUDE_distance_between_points_l1462_146278

/-- The distance between points (0, 12) and (5, 6) is √61 -/
theorem distance_between_points : Real.sqrt 61 = Real.sqrt ((5 - 0)^2 + (6 - 12)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1462_146278


namespace NUMINAMATH_CALUDE_sum_of_factors_360_l1462_146293

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive factors of 360 is 1170 -/
theorem sum_of_factors_360 : sum_of_factors 360 = 1170 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_360_l1462_146293


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1462_146277

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1462_146277


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1462_146294

theorem unique_triple_solution : 
  ∃! (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    Nat.Prime p ∧ Nat.Prime q ∧
    (r^2 - 5*q^2) / (p^2 - 1) = 2 ∧
    p = 3 ∧ q = 2 ∧ r = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1462_146294


namespace NUMINAMATH_CALUDE_vector_equation_proof_l1462_146266

/-- Prove that the given values of a and b satisfy the vector equation -/
theorem vector_equation_proof :
  let a : ℚ := -3/14
  let b : ℚ := 107/14
  let v1 : Fin 2 → ℚ := ![3, 4]
  let v2 : Fin 2 → ℚ := ![1, 6]
  let result : Fin 2 → ℚ := ![7, 45]
  (a • v1 + b • v2) = result :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_proof_l1462_146266


namespace NUMINAMATH_CALUDE_max_red_squares_l1462_146206

/-- A configuration of red squares on a 5x5 grid -/
def RedConfiguration := Fin 5 → Fin 5 → Bool

/-- Checks if four points form an axis-parallel rectangle -/
def isAxisParallelRectangle (p1 p2 p3 p4 : Fin 5 × Fin 5) : Bool :=
  sorry

/-- Counts the number of red squares in a configuration -/
def countRedSquares (config : RedConfiguration) : Nat :=
  sorry

/-- Checks if a configuration is valid (no axis-parallel rectangles) -/
def isValidConfiguration (config : RedConfiguration) : Prop :=
  ∀ p1 p2 p3 p4 : Fin 5 × Fin 5,
    config p1.1 p1.2 ∧ config p2.1 p2.2 ∧ config p3.1 p3.2 ∧ config p4.1 p4.2 →
    ¬isAxisParallelRectangle p1 p2 p3 p4

/-- The maximum number of red squares in a valid configuration is 12 -/
theorem max_red_squares :
  (∃ config : RedConfiguration, isValidConfiguration config ∧ countRedSquares config = 12) ∧
  (∀ config : RedConfiguration, isValidConfiguration config → countRedSquares config ≤ 12) :=
sorry

end NUMINAMATH_CALUDE_max_red_squares_l1462_146206


namespace NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1462_146260

/-- Given that a^3 varies inversely with b^4, prove that if a = 5 when b = 2, then a = 5/2 when b = 4 -/
theorem inverse_variation_cube_fourth (a b : ℝ) (h : ∃ k : ℝ, ∀ a b, a^3 * b^4 = k) :
  (5^3 * 2^4 = a^3 * 4^4) → a = 5/2 := by sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_fourth_l1462_146260


namespace NUMINAMATH_CALUDE_second_square_area_l1462_146237

/-- An isosceles right triangle with two inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the first inscribed square -/
  s : ℝ
  /-- Area of the first inscribed square is 484 -/
  h_area : s^2 = 484
  /-- Side length of the second inscribed square -/
  S : ℝ
  /-- The second square shares one side with the hypotenuse and its opposite vertex touches the midpoint of the hypotenuse -/
  h_S : S = (2 * s * Real.sqrt 2) / 3

/-- The area of the second inscribed square is 3872/9 -/
theorem second_square_area (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 3872 / 9 := by
  sorry

end NUMINAMATH_CALUDE_second_square_area_l1462_146237


namespace NUMINAMATH_CALUDE_circle_radius_l1462_146226

/-- The equation of a circle in the form x^2 + y^2 + 2x = 0 has radius 1 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 + 2*x = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l1462_146226


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1462_146215

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →                       -- non-zero common difference
  a 1 = 1 →                     -- a_1 = 1
  (a 3)^2 = a 1 * a 13 →        -- a_1, a_3, a_13 form a geometric sequence
  d = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1462_146215


namespace NUMINAMATH_CALUDE_problem_solution_l1462_146249

def A : Set ℝ := {-2, 3, 4, 6}
def B (a : ℝ) : Set ℝ := {3, a, a^2}

theorem problem_solution (a : ℝ) :
  (B a ⊆ A → a = 2) ∧
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1462_146249


namespace NUMINAMATH_CALUDE_sqrt_36_divided_by_itself_is_one_l1462_146241

theorem sqrt_36_divided_by_itself_is_one : 
  (Real.sqrt 36) / (Real.sqrt 36) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_divided_by_itself_is_one_l1462_146241


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l1462_146296

/-- Given two vectors a and b in R², prove that if they are parallel and have the given components, then m equals either -√2 or √2. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  (∃ (k : ℝ), a = k • b) → (m = -Real.sqrt 2 ∨ m = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l1462_146296


namespace NUMINAMATH_CALUDE_apple_seed_average_l1462_146276

theorem apple_seed_average (total_seeds : ℕ) (pear_avg : ℕ) (grape_avg : ℕ)
  (apple_count : ℕ) (pear_count : ℕ) (grape_count : ℕ) (seeds_needed : ℕ)
  (h1 : total_seeds = 60)
  (h2 : pear_avg = 2)
  (h3 : grape_avg = 3)
  (h4 : apple_count = 4)
  (h5 : pear_count = 3)
  (h6 : grape_count = 9)
  (h7 : seeds_needed = 3) :
  ∃ (apple_avg : ℕ), apple_avg = 6 ∧
    apple_count * apple_avg + pear_count * pear_avg + grape_count * grape_avg
    = total_seeds - seeds_needed :=
by sorry

end NUMINAMATH_CALUDE_apple_seed_average_l1462_146276


namespace NUMINAMATH_CALUDE_defective_firecracker_fraction_l1462_146253

theorem defective_firecracker_fraction :
  ∀ (initial_firecrackers confiscated_firecrackers good_firecrackers_set_off : ℕ),
    initial_firecrackers = 48 →
    confiscated_firecrackers = 12 →
    good_firecrackers_set_off = 15 →
    good_firecrackers_set_off * 2 = initial_firecrackers - confiscated_firecrackers - 
      (initial_firecrackers - confiscated_firecrackers - good_firecrackers_set_off * 2) →
    (initial_firecrackers - confiscated_firecrackers - good_firecrackers_set_off * 2) / 
    (initial_firecrackers - confiscated_firecrackers) = 1 / 6 :=
by sorry

end NUMINAMATH_CALUDE_defective_firecracker_fraction_l1462_146253


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1462_146224

/-- The speed of a boat in still water, given its downstream travel time and distance, and the speed of the stream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 5)
  (h3 : downstream_distance = 125) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1462_146224


namespace NUMINAMATH_CALUDE_quadratic_p_value_l1462_146295

/-- The quadratic function passing through specific points -/
def quadratic_function (p : ℝ) (x : ℝ) : ℝ := p * x^2 + 5 * x + p

theorem quadratic_p_value :
  ∃ (p : ℝ), 
    (quadratic_function p 0 = -2) ∧ 
    (quadratic_function p (1/2) = 0) ∧ 
    (quadratic_function p 2 = 0) ∧
    (p = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_p_value_l1462_146295


namespace NUMINAMATH_CALUDE_variance_describes_dispersion_l1462_146228

-- Define the type for statistical measures
inductive StatMeasure
  | Mean
  | Variance
  | Median
  | Mode

-- Define a property for measures that describe dispersion
def describes_dispersion (m : StatMeasure) : Prop :=
  match m with
  | StatMeasure.Variance => True
  | _ => False

-- Theorem statement
theorem variance_describes_dispersion :
  ∀ m : StatMeasure, describes_dispersion m ↔ m = StatMeasure.Variance :=
by sorry

end NUMINAMATH_CALUDE_variance_describes_dispersion_l1462_146228


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l1462_146279

/-- Given two non-collinear vectors in a plane, prove that m = -2/3 when the given conditions are met. -/
theorem collinear_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ ∃ (k : ℝ), a = k • b →  -- a and b are non-collinear
  ∃ (A B C : ℝ × ℝ),
    B - A = 2 • a + m • b ∧  -- AB = 2a + mb
    C - B = 3 • a - b ∧  -- BC = 3a - b
    ∃ (t : ℝ), C - A = t • (B - A) →  -- A, B, C are collinear
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l1462_146279


namespace NUMINAMATH_CALUDE_find_m_l1462_146280

theorem find_m (U A : Set ℤ) (m : ℤ) : 
  U = {2, 3, m^2 + m - 4} →
  A = {m, 2} →
  U \ A = {3} →
  m = -2 := by sorry

end NUMINAMATH_CALUDE_find_m_l1462_146280


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1462_146201

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1462_146201


namespace NUMINAMATH_CALUDE_cowboy_shortest_path_l1462_146231

/-- The shortest path for a cowboy to travel from his position to a stream and then to his cabin -/
theorem cowboy_shortest_path (cowboy_pos cabin_pos : ℝ × ℝ) (stream_y : ℝ) :
  cowboy_pos = (0, -5) →
  cabin_pos = (6, 4) →
  stream_y = 0 →
  let dist_to_stream := |cowboy_pos.2 - stream_y|
  let dist_stream_to_cabin := Real.sqrt ((cabin_pos.1 - cowboy_pos.1)^2 + (cabin_pos.2 - stream_y)^2)
  dist_to_stream + dist_stream_to_cabin = 5 + 2 * Real.sqrt 58 :=
by sorry

end NUMINAMATH_CALUDE_cowboy_shortest_path_l1462_146231


namespace NUMINAMATH_CALUDE_fred_dimes_remaining_l1462_146270

theorem fred_dimes_remaining (initial_dimes borrowed_dimes : ℕ) :
  initial_dimes = 7 →
  borrowed_dimes = 3 →
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_remaining_l1462_146270


namespace NUMINAMATH_CALUDE_reflection_line_l1462_146242

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 9), then m + b = 38/3 -/
theorem reflection_line (m b : ℝ) : 
  (∃ (x y : ℝ), x = 10 ∧ y = 9 ∧ 
    (x - 2)^2 + (y - 3)^2 = ((x - 2) * m + (y - 3))^2 / (m^2 + 1) ∧
    y - 3 = -m * (x - 2)) →
  m + b = 38/3 :=
by sorry

end NUMINAMATH_CALUDE_reflection_line_l1462_146242


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_power_of_two_l1462_146246

theorem exists_number_divisible_by_power_of_two (n : ℕ) :
  ∃ k : ℕ, (∀ d : ℕ, d < n → (k / 10^d % 10 = 1 ∨ k / 10^d % 10 = 2)) ∧ k % 2^n = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_power_of_two_l1462_146246


namespace NUMINAMATH_CALUDE_f_composition_of_three_l1462_146271

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4*n + 2

theorem f_composition_of_three : f (f (f 3)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l1462_146271


namespace NUMINAMATH_CALUDE_paco_cookies_l1462_146207

/-- Given that Paco had 22 sweet cookies initially and ate 15 sweet cookies,
    prove that he had 7 sweet cookies left. -/
theorem paco_cookies (initial_sweet : ℕ) (eaten_sweet : ℕ) 
  (h1 : initial_sweet = 22) 
  (h2 : eaten_sweet = 15) : 
  initial_sweet - eaten_sweet = 7 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l1462_146207


namespace NUMINAMATH_CALUDE_ancient_pi_approximation_l1462_146281

theorem ancient_pi_approximation (V : ℝ) (r : ℝ) (d : ℝ) :
  V = (4 / 3) * Real.pi * r^3 →
  d = (16 / 9 * V)^(1/3) →
  (6 * 9) / 16 = 3.375 :=
by sorry

end NUMINAMATH_CALUDE_ancient_pi_approximation_l1462_146281


namespace NUMINAMATH_CALUDE_only_zero_satisfies_equations_l1462_146286

theorem only_zero_satisfies_equations (x y a : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : x^3 + y^3 = a) 
  (eq3 : x^5 + y^5 = a) : 
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_only_zero_satisfies_equations_l1462_146286


namespace NUMINAMATH_CALUDE_min_value_expression_l1462_146203

theorem min_value_expression (x y : ℝ) : 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1462_146203


namespace NUMINAMATH_CALUDE_line_through_point_and_circle_center_l1462_146223

/-- The equation of the line passing through (2,1) and the center of the circle (x-1)^2 + (y+2)^2 = 5 is 3x - y - 5 = 0 -/
theorem line_through_point_and_circle_center :
  let point : ℝ × ℝ := (2, 1)
  let circle_center : ℝ × ℝ := (1, -2)
  let line_equation (x y : ℝ) := 3 * x - y - 5 = 0
  ∀ x y : ℝ, line_equation x y ↔ (y - point.2) / (x - point.1) = (circle_center.2 - point.2) / (circle_center.1 - point.1) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_and_circle_center_l1462_146223


namespace NUMINAMATH_CALUDE_larger_number_proof_l1462_146250

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1462_146250


namespace NUMINAMATH_CALUDE_isabel_country_albums_l1462_146261

/-- Represents the number of songs in each album -/
def songs_per_album : ℕ := 8

/-- Represents the number of pop albums bought -/
def pop_albums : ℕ := 5

/-- Represents the total number of songs bought -/
def total_songs : ℕ := 72

/-- Represents the number of country albums bought -/
def country_albums : ℕ := (total_songs - pop_albums * songs_per_album) / songs_per_album

theorem isabel_country_albums : country_albums = 4 := by
  sorry

end NUMINAMATH_CALUDE_isabel_country_albums_l1462_146261


namespace NUMINAMATH_CALUDE_students_either_not_both_is_38_l1462_146238

/-- The number of students taking either geometry or history but not both -/
def students_either_not_both (students_both : ℕ) (students_geometry : ℕ) (students_only_history : ℕ) : ℕ :=
  (students_geometry - students_both) + students_only_history

/-- Theorem stating the number of students taking either geometry or history but not both -/
theorem students_either_not_both_is_38 :
  students_either_not_both 15 35 18 = 38 := by
  sorry

#check students_either_not_both_is_38

end NUMINAMATH_CALUDE_students_either_not_both_is_38_l1462_146238


namespace NUMINAMATH_CALUDE_auction_result_l1462_146255

/-- Calculates the total amount received from selling a TV and a phone at an auction -/
def auction_total (tv_cost phone_cost : ℚ) (tv_increase phone_increase : ℚ) : ℚ :=
  (tv_cost + tv_cost * tv_increase) + (phone_cost + phone_cost * phone_increase)

/-- Theorem stating the total amount received from the auction -/
theorem auction_result : 
  auction_total 500 400 (2/5) (40/100) = 1260 := by sorry

end NUMINAMATH_CALUDE_auction_result_l1462_146255


namespace NUMINAMATH_CALUDE_square_ratio_proof_l1462_146284

theorem square_ratio_proof (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 75 / 128 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 5 ∧ b = 6 ∧ c = 16 →
  a + b + c = 27 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_proof_l1462_146284


namespace NUMINAMATH_CALUDE_kids_staying_home_l1462_146236

def total_kids : ℕ := 1059955
def camp_kids : ℕ := 564237

theorem kids_staying_home : total_kids - camp_kids = 495718 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l1462_146236


namespace NUMINAMATH_CALUDE_overtake_at_eight_hours_l1462_146268

/-- Represents the chase between a pirate ship and a trading vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  pirate_initial_speed : ℝ
  trading_initial_speed : ℝ
  damage_time : ℝ
  pirate_damaged_distance : ℝ
  trading_damaged_distance : ℝ

/-- The time at which the pirate ship overtakes the trading vessel -/
def overtake_time (scenario : ChaseScenario) : ℝ :=
  sorry

/-- The specific chase scenario described in the problem -/
def given_scenario : ChaseScenario :=
  { initial_distance := 15
  , pirate_initial_speed := 14
  , trading_initial_speed := 10
  , damage_time := 3
  , pirate_damaged_distance := 18
  , trading_damaged_distance := 17 }

/-- Theorem stating that the overtake time for the given scenario is 8 hours -/
theorem overtake_at_eight_hours :
  overtake_time given_scenario = 8 :=
sorry

end NUMINAMATH_CALUDE_overtake_at_eight_hours_l1462_146268


namespace NUMINAMATH_CALUDE_noahs_yearly_call_cost_l1462_146245

/-- The total cost of Noah's calls to his Grammy for a year -/
def total_cost (weeks_per_year : ℕ) (minutes_per_call : ℕ) (cost_per_minute : ℚ) : ℚ :=
  (weeks_per_year * minutes_per_call : ℕ) * cost_per_minute

/-- Theorem: Noah's yearly call cost to Grammy is $78 -/
theorem noahs_yearly_call_cost :
  total_cost 52 30 (5/100) = 78 := by
  sorry

end NUMINAMATH_CALUDE_noahs_yearly_call_cost_l1462_146245


namespace NUMINAMATH_CALUDE_missing_number_proof_l1462_146248

theorem missing_number_proof (x : ℝ) : 
  (x + 42 + 78 + 104) / 4 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  x = 74 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1462_146248


namespace NUMINAMATH_CALUDE_tony_sand_and_water_problem_l1462_146204

/-- Represents the problem of Tony filling his sandbox with sand and drinking water --/
theorem tony_sand_and_water_problem 
  (bucket_capacity : ℕ)
  (sandbox_depth sandbox_width sandbox_length : ℕ)
  (sand_weight_per_cubic_foot : ℕ)
  (water_per_session : ℕ)
  (water_bottle_volume : ℕ)
  (water_bottle_cost : ℕ)
  (initial_money : ℕ)
  (change_after_buying : ℕ)
  (h1 : bucket_capacity = 2)
  (h2 : sandbox_depth = 2)
  (h3 : sandbox_width = 4)
  (h4 : sandbox_length = 5)
  (h5 : sand_weight_per_cubic_foot = 3)
  (h6 : water_per_session = 3)
  (h7 : water_bottle_volume = 15)
  (h8 : water_bottle_cost = 2)
  (h9 : initial_money = 10)
  (h10 : change_after_buying = 4) :
  (sandbox_depth * sandbox_width * sandbox_length * sand_weight_per_cubic_foot) / bucket_capacity / 
  ((initial_money - change_after_buying) / water_bottle_cost * water_bottle_volume / water_per_session) = 4 :=
by sorry

end NUMINAMATH_CALUDE_tony_sand_and_water_problem_l1462_146204


namespace NUMINAMATH_CALUDE_function_inequality_l1462_146254

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem function_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f a x₀ ≤ 0) →
  (a ≥ (Real.exp 2 + 1) / (Real.exp 1 - 1) ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1462_146254


namespace NUMINAMATH_CALUDE_third_side_is_seven_l1462_146217

/-- A triangle with two known sides and even perimeter -/
structure EvenPerimeterTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 2
  side2_eq : side2 = 7
  even_perimeter : ∃ n : ℕ, side1 + side2 + side3 = 2 * n

/-- The third side of an EvenPerimeterTriangle is 7 -/
theorem third_side_is_seven (t : EvenPerimeterTriangle) : t.side3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_side_is_seven_l1462_146217


namespace NUMINAMATH_CALUDE_angle_2_measure_l1462_146216

-- Define complementary angles
def complementary (a1 a2 : ℝ) : Prop := a1 + a2 = 90

-- Theorem statement
theorem angle_2_measure (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_angle_2_measure_l1462_146216


namespace NUMINAMATH_CALUDE_employee_payment_proof_l1462_146244

/-- The weekly payment for employee B -/
def payment_B : ℝ := 180

/-- The weekly payment for employee A -/
def payment_A : ℝ := 1.5 * payment_B

/-- The total weekly payment for both employees -/
def total_payment : ℝ := 450

theorem employee_payment_proof :
  payment_A + payment_B = total_payment :=
by sorry

end NUMINAMATH_CALUDE_employee_payment_proof_l1462_146244


namespace NUMINAMATH_CALUDE_purely_imaginary_reciprocal_l1462_146234

theorem purely_imaginary_reciprocal (m : ℝ) :
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → z⁻¹ = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_reciprocal_l1462_146234


namespace NUMINAMATH_CALUDE_colored_disk_overlap_l1462_146262

/-- Represents a disk with colored sectors -/
structure ColoredDisk :=
  (total_sectors : ℕ)
  (colored_sectors : ℕ)
  (h_total : total_sectors > 0)
  (h_colored : colored_sectors ≤ total_sectors)

/-- Counts the number of positions with at most k overlapping colored sectors -/
def count_low_overlap_positions (d1 d2 : ColoredDisk) (k : ℕ) : ℕ :=
  sorry

theorem colored_disk_overlap (d1 d2 : ColoredDisk) 
  (h1 : d1.total_sectors = 1985) (h2 : d2.total_sectors = 1985)
  (h3 : d1.colored_sectors = 200) (h4 : d2.colored_sectors = 200) :
  count_low_overlap_positions d1 d2 20 ≥ 80 := by
  sorry

end NUMINAMATH_CALUDE_colored_disk_overlap_l1462_146262


namespace NUMINAMATH_CALUDE_longest_frog_vs_shortest_grasshopper_l1462_146288

def frog_jumps : List ℕ := [39, 45, 50]
def grasshopper_jumps : List ℕ := [17, 22, 28, 31]

theorem longest_frog_vs_shortest_grasshopper :
  (List.maximum frog_jumps).get! - (List.minimum grasshopper_jumps).get! = 33 := by
  sorry

end NUMINAMATH_CALUDE_longest_frog_vs_shortest_grasshopper_l1462_146288


namespace NUMINAMATH_CALUDE_flower_cost_minimization_l1462_146211

/-- The cost of one lily in dollars -/
def lily_cost : ℝ := 5

/-- The cost of one carnation in dollars -/
def carnation_cost : ℝ := 6

/-- The total number of flowers to be bought -/
def total_flowers : ℕ := 12

/-- The minimum number of carnations to be bought -/
def min_carnations : ℕ := 5

/-- The cost function for buying x lilies -/
def cost_function (x : ℝ) : ℝ := -x + 72

theorem flower_cost_minimization :
  ∃ (x : ℝ),
    x ≤ total_flowers - min_carnations ∧
    x ≥ 0 ∧
    ∀ (y : ℝ),
      y ≤ total_flowers - min_carnations ∧
      y ≥ 0 →
      cost_function x ≤ cost_function y ∧
      cost_function x = 65 :=
sorry

end NUMINAMATH_CALUDE_flower_cost_minimization_l1462_146211


namespace NUMINAMATH_CALUDE_triangle_shape_l1462_146290

theorem triangle_shape (A : Real) (h1 : 0 < A ∧ A < π) 
  (h2 : Real.sin A + Real.cos A = 7/12) : 
  ∃ (B C : Real), 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π ∧ π/2 < A := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1462_146290


namespace NUMINAMATH_CALUDE_bike_retail_price_l1462_146251

/-- The retail price of a bike, given Maria's savings, her mother's offer, and the additional amount needed. -/
theorem bike_retail_price
  (maria_savings : ℕ)
  (mother_offer : ℕ)
  (additional_needed : ℕ)
  (h1 : maria_savings = 120)
  (h2 : mother_offer = 250)
  (h3 : additional_needed = 230) :
  maria_savings + mother_offer + additional_needed = 600 :=
by sorry

end NUMINAMATH_CALUDE_bike_retail_price_l1462_146251


namespace NUMINAMATH_CALUDE_limit_expression_l1462_146214

/-- The limit of (2 - e^(arcsin²(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_expression : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_expression_l1462_146214


namespace NUMINAMATH_CALUDE_abs_w_equals_3_fourth_root_2_l1462_146213

-- Define w as a complex number
variable (w : ℂ)

-- State the theorem
theorem abs_w_equals_3_fourth_root_2 (h : w^2 = -18 + 18*I) : 
  Complex.abs w = 3 * (2 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_abs_w_equals_3_fourth_root_2_l1462_146213


namespace NUMINAMATH_CALUDE_prob_white_same_color_five_balls_l1462_146229

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of drawing two white balls given that the drawn balls are of the same color -/
def prob_white_given_same_color (total white black : ℕ) : ℚ :=
  let white_ways := choose white 2
  let black_ways := choose black 2
  white_ways / (white_ways + black_ways)

theorem prob_white_same_color_five_balls :
  prob_white_given_same_color 5 3 2 = 3/4 := by sorry

end NUMINAMATH_CALUDE_prob_white_same_color_five_balls_l1462_146229


namespace NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1462_146285

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_mod_7_l1462_146285


namespace NUMINAMATH_CALUDE_valid_distribution_exists_l1462_146220

/-- Represents a part of the city -/
structure CityPart where
  id : Nat

/-- Represents a currency exchange point -/
structure ExchangePoint where
  id : Nat

/-- A distribution of exchange points across city parts -/
def Distribution := CityPart → Finset ExchangePoint

/-- The property that each city part contains exactly two exchange points -/
def ValidDistribution (d : Distribution) (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint) : Prop :=
  ∀ cp ∈ cityParts, (d cp).card = 2

/-- The main theorem stating that a valid distribution exists -/
theorem valid_distribution_exists (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint)
    (h1 : cityParts.card = 4) (h2 : exchangePoints.card = 4) :
    ∃ d : Distribution, ValidDistribution d cityParts exchangePoints := by
  sorry

end NUMINAMATH_CALUDE_valid_distribution_exists_l1462_146220


namespace NUMINAMATH_CALUDE_variance_fluctuation_relationship_l1462_146272

/-- Definition of variance for a list of numbers -/
def variance (data : List ℝ) : ℝ := sorry

/-- Definition of fluctuation for a list of numbers -/
def fluctuation (data : List ℝ) : ℝ := sorry

/-- Theorem: If the variance of data set A is greater than the variance of data set B,
    then the fluctuation of A is greater than the fluctuation of B -/
theorem variance_fluctuation_relationship (A B : List ℝ) :
  variance A > variance B → fluctuation A > fluctuation B := by sorry

end NUMINAMATH_CALUDE_variance_fluctuation_relationship_l1462_146272


namespace NUMINAMATH_CALUDE_expression_simplification_l1462_146235

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1462_146235


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l1462_146287

/-- The quartic polynomial q(x) that satisfies given conditions -/
def q (x : ℚ) : ℚ := (1/6) * x^4 - (8/3) * x^3 - (14/3) * x^2 - (8/3) * x - 16/3

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q 1 = -8 ∧ q 2 = -18 ∧ q 3 = -40 ∧ q 4 = -80 ∧ q 5 = -140 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l1462_146287


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1462_146210

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 5*m + 3 = 0) → 
  (n^2 + 5*n + 3 = 0) → 
  m * Real.sqrt (n / m) + n * Real.sqrt (m / n) = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1462_146210


namespace NUMINAMATH_CALUDE_five_sundays_april_implies_five_mondays_may_l1462_146263

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific month -/
structure Month where
  numDays : Nat
  firstDay : DayOfWeek

/-- Given a day, returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of occurrences of a specific day in a month -/
def countDaysInMonth (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If April has five Sundays, then May has five Mondays -/
theorem five_sundays_april_implies_five_mondays_may :
  ∀ (april : Month) (may : Month),
    april.numDays = 30 →
    may.numDays = 31 →
    may.firstDay = nextDay april.firstDay →
    countDaysInMonth april DayOfWeek.Sunday = 5 →
    countDaysInMonth may DayOfWeek.Monday = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_sundays_april_implies_five_mondays_may_l1462_146263


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l1462_146200

theorem polynomial_coefficients (a b c : ℚ) :
  let f : ℚ → ℚ := λ x => c * x^4 + a * x^3 - 3 * x^2 + b * x - 8
  (f 2 = -8) ∧ (f (-3) = -68) → (a = 5 ∧ b = 7 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l1462_146200


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1462_146225

def binary_to_nat (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else to_binary_aux (m / 2) ((m % 2 = 1) :: acc)
  to_binary_aux n []

def binary_mult (a b : List Bool) : List Bool :=
  nat_to_binary (binary_to_nat a * binary_to_nat b)

theorem binary_multiplication_theorem :
  binary_mult [true, false, false, true, true] [true, true, true] = 
  [true, true, true, true, true, false, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1462_146225


namespace NUMINAMATH_CALUDE_amount_difference_l1462_146233

theorem amount_difference (x : ℝ) (h : x = 690) : (0.25 * 1500) - (0.5 * x) = 30 := by
  sorry

end NUMINAMATH_CALUDE_amount_difference_l1462_146233


namespace NUMINAMATH_CALUDE_gcf_294_108_l1462_146240

theorem gcf_294_108 : Nat.gcd 294 108 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_294_108_l1462_146240


namespace NUMINAMATH_CALUDE_libby_quarters_l1462_146267

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The cost of the dress in dollars -/
def dress_cost : ℕ := 35

/-- The number of quarters Libby has left after paying for the dress -/
def quarters_left : ℕ := 20

/-- The initial number of quarters Libby had -/
def initial_quarters : ℕ := dress_cost * quarters_per_dollar + quarters_left

theorem libby_quarters : initial_quarters = 160 := by
  sorry

end NUMINAMATH_CALUDE_libby_quarters_l1462_146267


namespace NUMINAMATH_CALUDE_statue_selling_price_l1462_146292

/-- The selling price of a statue given its original cost and profit percentage -/
def selling_price (original_cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  original_cost * (1 + profit_percentage)

/-- Theorem: The selling price of the statue is $660 -/
theorem statue_selling_price :
  let original_cost : ℝ := 550
  let profit_percentage : ℝ := 0.20
  selling_price original_cost profit_percentage = 660 := by
  sorry

end NUMINAMATH_CALUDE_statue_selling_price_l1462_146292


namespace NUMINAMATH_CALUDE_sum_of_specific_T_l1462_146282

/-- Definition of T_n for n ≥ 2 -/
def T (n : ℕ) : ℤ :=
  if n < 2 then 0 else
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

/-- Theorem stating that T₂₀ + T₃₆ + T₄₅ = -5 -/
theorem sum_of_specific_T : T 20 + T 36 + T 45 = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_l1462_146282


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1462_146209

theorem sqrt_inequality (n : ℝ) (h : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1462_146209


namespace NUMINAMATH_CALUDE_pentagonal_prism_sum_l1462_146243

/-- A pentagonal prism is a three-dimensional geometric shape with specific properties. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : ℕ
  /-- The number of edges in a pentagonal prism -/
  edges : ℕ
  /-- The number of vertices in a pentagonal prism -/
  vertices : ℕ
  /-- A pentagonal prism has 7 faces (2 pentagonal bases + 5 rectangular lateral faces) -/
  faces_count : faces = 7
  /-- A pentagonal prism has 15 edges (5 for each base + 5 connecting edges) -/
  edges_count : edges = 15
  /-- A pentagonal prism has 10 vertices (5 for each base) -/
  vertices_count : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32. -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_prism_sum_l1462_146243


namespace NUMINAMATH_CALUDE_division_remainder_549547_by_7_l1462_146269

theorem division_remainder_549547_by_7 : 
  549547 % 7 = 5 := by sorry

end NUMINAMATH_CALUDE_division_remainder_549547_by_7_l1462_146269


namespace NUMINAMATH_CALUDE_rounded_avg_mb_per_minute_is_one_l1462_146274

/-- Represents the number of days of music in the library -/
def days_of_music : ℕ := 15

/-- Represents the total disk space occupied by the library in megabytes -/
def total_disk_space : ℕ := 20000

/-- Calculates the total number of minutes of music in the library -/
def total_minutes : ℕ := days_of_music * 24 * 60

/-- Calculates the average megabytes per minute of music -/
def avg_mb_per_minute : ℚ := total_disk_space / total_minutes

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- Theorem stating that the rounded average megabytes per minute is 1 -/
theorem rounded_avg_mb_per_minute_is_one :
  round_to_nearest avg_mb_per_minute = 1 := by sorry

end NUMINAMATH_CALUDE_rounded_avg_mb_per_minute_is_one_l1462_146274


namespace NUMINAMATH_CALUDE_basketball_practice_time_l1462_146219

theorem basketball_practice_time (school_day_practice : ℕ) : 
  (5 * school_day_practice + 2 * (2 * school_day_practice) = 135) → 
  school_day_practice = 15 := by
sorry

end NUMINAMATH_CALUDE_basketball_practice_time_l1462_146219


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1462_146257

def starts_with_19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≥ 19 * 10^k ∧ n < 20 * 10^k

def ends_with_89 (n : ℕ) : Prop :=
  n % 100 = 89

def is_valid_number (n : ℕ) : Prop :=
  starts_with_19 (n^2) ∧ ends_with_89 (n^2)

theorem smallest_valid_number :
  is_valid_number 1383 ∧ ∀ m : ℕ, m < 1383 → ¬(is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1462_146257


namespace NUMINAMATH_CALUDE_unripe_oranges_zero_l1462_146298

/-- Represents the daily harvest of oranges -/
structure DailyHarvest where
  ripe : ℕ
  unripe : ℕ

/-- Represents the total harvest over a period of days -/
structure TotalHarvest where
  days : ℕ
  ripe : ℕ

/-- Proves that the number of unripe oranges harvested per day is zero -/
theorem unripe_oranges_zero 
  (daily : DailyHarvest) 
  (total : TotalHarvest) 
  (h1 : daily.ripe = 82)
  (h2 : total.days = 25)
  (h3 : total.ripe = 2050)
  (h4 : daily.ripe * total.days = total.ripe) :
  daily.unripe = 0 := by
  sorry

#check unripe_oranges_zero

end NUMINAMATH_CALUDE_unripe_oranges_zero_l1462_146298


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1462_146297

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_first_term
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_relation : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 2) :
  a 1 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1462_146297


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l1462_146299

/-- The number of apples Sarah gives to teachers -/
def apples_to_teachers (initial : ℕ) (locker : ℕ) (classmates : ℕ) (friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - locker - classmates - friends - eaten - left

theorem sarah_apples_to_teachers :
  apples_to_teachers 50 10 8 5 1 4 = 22 := by
  sorry

#eval apples_to_teachers 50 10 8 5 1 4

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l1462_146299


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1462_146227

theorem simplify_complex_fraction :
  1 / ((3 / (Real.sqrt 5 + 2)) - (4 / (Real.sqrt 7 + 2))) =
  3 * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) /
  ((9 * Real.sqrt 5 - 4 * Real.sqrt 7 - 10) * (9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1462_146227


namespace NUMINAMATH_CALUDE_eventually_monotonic_sequence_l1462_146212

/-- An infinite sequence of real numbers where no two members are equal -/
def UniqueMemberSequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- A monotonic segment of length m starting at index i -/
def MonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ k, k < m - 1 → a (i + k) < a (i + k + 1)) ∨
  (∀ k, k < m - 1 → a (i + k) > a (i + k + 1))

/-- For each natural k, the term aₖ is contained in some monotonic segment of length k + 1 -/
def ContainedInMonotonicSegment (a : ℕ → ℝ) : Prop :=
  ∀ k, ∃ i, k ∈ Finset.range (k + 1) ∧ MonotonicSegment a i (k + 1)

/-- The sequence is eventually monotonic -/
def EventuallyMonotonic (a : ℕ → ℝ) : Prop :=
  ∃ N, (∀ n ≥ N, a n < a (n + 1)) ∨ (∀ n ≥ N, a n > a (n + 1))

theorem eventually_monotonic_sequence
  (a : ℕ → ℝ)
  (h1 : UniqueMemberSequence a)
  (h2 : ContainedInMonotonicSegment a) :
  EventuallyMonotonic a :=
sorry

end NUMINAMATH_CALUDE_eventually_monotonic_sequence_l1462_146212


namespace NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_one_fourth_l1462_146239

theorem division_by_fraction (a b : ℚ) (hb : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_fourth :
  12 / (1 / 4) = 48 := by sorry

end NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_one_fourth_l1462_146239


namespace NUMINAMATH_CALUDE_total_tickets_sold_l1462_146273

/-- Represents the ticket sales for a theater performance --/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- The pricing and sales data for the theater --/
def theaterData : TicketSales → Prop := fun ts =>
  12 * ts.orchestra + 8 * ts.balcony = 3320 ∧
  ts.balcony = ts.orchestra + 240

/-- Theorem stating that the total number of tickets sold is 380 --/
theorem total_tickets_sold (ts : TicketSales) (h : theaterData ts) : 
  ts.orchestra + ts.balcony = 380 := by
  sorry

#check total_tickets_sold

end NUMINAMATH_CALUDE_total_tickets_sold_l1462_146273


namespace NUMINAMATH_CALUDE_greatest_p_value_l1462_146265

theorem greatest_p_value (x : ℝ) (p : ℝ) : 
  (∃ x, 2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 = 
        p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))) →
  p ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_p_value_l1462_146265


namespace NUMINAMATH_CALUDE_largest_divisor_fifth_largest_divisor_l1462_146258

def n : ℕ := 1516000000

-- Define a function to get the kth largest divisor
def kthLargestDivisor (k : ℕ) : ℕ := sorry

-- The largest divisor of n is itself
theorem largest_divisor : kthLargestDivisor 1 = n := sorry

-- The fifth-largest divisor of n is 94,750,000
theorem fifth_largest_divisor : kthLargestDivisor 5 = 94750000 := sorry

end NUMINAMATH_CALUDE_largest_divisor_fifth_largest_divisor_l1462_146258


namespace NUMINAMATH_CALUDE_least_number_of_cans_l1462_146247

def maaza_volume : ℕ := 10
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_number_of_cans (can_volume : ℕ) 
  (h1 : can_volume > 0)
  (h2 : can_volume ∣ maaza_volume)
  (h3 : can_volume ∣ pepsi_volume)
  (h4 : can_volume ∣ sprite_volume)
  (h5 : ∀ x : ℕ, x > can_volume → ¬(x ∣ maaza_volume ∧ x ∣ pepsi_volume ∧ x ∣ sprite_volume)) :
  maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = 261 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l1462_146247


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l1462_146289

/-- Given points P, Q, R, and S on a line segment PQ where PQ = 4PS and PQ = 8QR,
    the probability that a randomly selected point on PQ is between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q →  -- Points are in order on the line
  Q - P = 4 * (S - P) →    -- PQ = 4PS
  Q - P = 8 * (Q - R) →    -- PQ = 8QR
  (S - R) / (Q - P) = 5/8  -- Probability is length of RS divided by length of PQ
  := by sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l1462_146289


namespace NUMINAMATH_CALUDE_sphere_packing_radius_l1462_146230

/-- A sphere in a unit cube -/
structure SpherePacking where
  radius : ℝ
  center_at_vertex : Bool
  touches_three_faces : Bool
  tangent_to_six_neighbors : Bool

/-- The theorem stating the radius of spheres in the specific packing -/
theorem sphere_packing_radius (s : SpherePacking) :
  s.center_at_vertex ∧ 
  s.touches_three_faces ∧ 
  s.tangent_to_six_neighbors →
  s.radius = (Real.sqrt 3 * (Real.sqrt 3 - 1)) / 4 :=
sorry

end NUMINAMATH_CALUDE_sphere_packing_radius_l1462_146230
