import Mathlib

namespace NUMINAMATH_CALUDE_james_beats_record_l2923_292365

/-- The number of points James beat the old record by -/
def points_above_record (touchdowns_per_game : ℕ) (points_per_touchdown : ℕ) 
  (games_in_season : ℕ) (two_point_conversions : ℕ) (old_record : ℕ) : ℕ :=
  let total_points := touchdowns_per_game * points_per_touchdown * games_in_season + 
                      two_point_conversions * 2
  total_points - old_record

/-- Theorem stating that James beat the old record by 72 points -/
theorem james_beats_record : points_above_record 4 6 15 6 300 = 72 := by
  sorry

end NUMINAMATH_CALUDE_james_beats_record_l2923_292365


namespace NUMINAMATH_CALUDE_x_equals_y_l2923_292362

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : x = y :=
by sorry

end NUMINAMATH_CALUDE_x_equals_y_l2923_292362


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2923_292353

/-- Given a complex number i such that i^2 = -1, 
    prove that (2-i)/(1+4i) = -2/17 - (9/17)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2923_292353


namespace NUMINAMATH_CALUDE_sum_of_altitudes_l2923_292395

/-- The sum of altitudes of a triangle formed by the line 10x + 8y = 80 and the coordinate axes --/
theorem sum_of_altitudes (x y : ℝ) : 
  (10 * x + 8 * y = 80) →
  (∃ (a b c : ℝ), 
    (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) ∧
    (10 * a + 8 * b = 80) ∧
    (a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41) ∧
    (c = 40 / Real.sqrt 41)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_l2923_292395


namespace NUMINAMATH_CALUDE_solve_for_q_l2923_292373

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/72)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 140 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l2923_292373


namespace NUMINAMATH_CALUDE_milk_cartons_accepted_l2923_292363

theorem milk_cartons_accepted (total_cartons : ℕ) (num_customers : ℕ) (damaged_per_customer : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : num_customers = 4)
  (h3 : damaged_per_customer = 60) :
  (total_cartons / num_customers - damaged_per_customer) * num_customers = 160 :=
by sorry

end NUMINAMATH_CALUDE_milk_cartons_accepted_l2923_292363


namespace NUMINAMATH_CALUDE_area_of_R_l2923_292331

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | floor (p.1 ^ 2) = floor p.2 ∧ floor (p.2 ^ 2) = floor p.1}

/-- The area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the area of region R -/
theorem area_of_R : area R = 4 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_area_of_R_l2923_292331


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2923_292370

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a2 : a 2 = 4) :
  ∀ n : ℕ, a n = 2^n := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2923_292370


namespace NUMINAMATH_CALUDE_lighthouse_angle_elevation_l2923_292306

/-- Given a lighthouse and two ships, proves that the angle of elevation from one ship is 30 degrees -/
theorem lighthouse_angle_elevation 
  (h : ℝ) -- height of the lighthouse
  (d : ℝ) -- distance between the ships
  (θ₁ : ℝ) -- angle of elevation from the first ship
  (θ₂ : ℝ) -- angle of elevation from the second ship
  (h_height : h = 100) -- lighthouse height is 100 m
  (h_distance : d = 273.2050807568877) -- distance between ships
  (h_angle₂ : θ₂ = 45 * π / 180) -- angle from second ship is 45°
  : θ₁ = 30 * π / 180 := by 
sorry


end NUMINAMATH_CALUDE_lighthouse_angle_elevation_l2923_292306


namespace NUMINAMATH_CALUDE_chord_length_is_four_l2923_292399

/-- Given a line and a circle in 2D space, prove that the length of the chord
    intercepted by the line and the circle is equal to 4. -/
theorem chord_length_is_four (x y : ℝ) : 
  (x + 2 * y - 2 = 0) →  -- Line equation
  ((x - 2)^2 + y^2 = 4) →  -- Circle equation
  ∃ (a b c d : ℝ), 
    (a + 2 * b - 2 = 0) ∧  -- Point (a, b) on the line
    ((a - 2)^2 + b^2 = 4) ∧  -- Point (a, b) on the circle
    (c + 2 * d - 2 = 0) ∧  -- Point (c, d) on the line
    ((c - 2)^2 + d^2 = 4) ∧  -- Point (c, d) on the circle
    (a ≠ c ∨ b ≠ d) ∧  -- (a, b) and (c, d) are distinct points
    ((a - c)^2 + (b - d)^2 = 4^2)  -- Distance between points is 4
  := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l2923_292399


namespace NUMINAMATH_CALUDE_cubic_function_minimum_l2923_292352

/-- The function f(x) = x³ - 3x² + 1 reaches its global minimum at x = 2 -/
theorem cubic_function_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 + 1
  ∀ y : ℝ, f 2 ≤ f y := by sorry

end NUMINAMATH_CALUDE_cubic_function_minimum_l2923_292352


namespace NUMINAMATH_CALUDE_equation_solution_l2923_292392

theorem equation_solution (y : ℝ) : 
  (y^3 - 3*y^2)/(y^2 + 2*y + 1) + 2*y = -1 ↔ y = 1/Real.sqrt 3 ∨ y = -1/Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2923_292392


namespace NUMINAMATH_CALUDE_seven_points_non_isosceles_l2923_292396

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define an isosceles triangle
def IsIsosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p3.x - p1.x)^2 + (p3.y - p1.y)^2 = (p3.x - p2.x)^2 + (p3.y - p2.y)^2

-- Main theorem
theorem seven_points_non_isosceles (points : Fin 7 → Point) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ ¬IsIsosceles (points i) (points j) (points k) := by
  sorry

end NUMINAMATH_CALUDE_seven_points_non_isosceles_l2923_292396


namespace NUMINAMATH_CALUDE_max_a_inequality_max_a_equality_l2923_292339

theorem max_a_inequality (a : ℝ) : 
  (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) → a ≤ Real.exp 1 := by
  sorry

theorem max_a_equality : 
  ∃ a : ℝ, a = Real.exp 1 ∧ (∀ x > 0, Real.log (a * x) + a * x ≤ x + Real.exp x) := by
  sorry

end NUMINAMATH_CALUDE_max_a_inequality_max_a_equality_l2923_292339


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2923_292381

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) / (1 - 2*x) ≥ 0 ↔ x ∈ Set.Ioo (1/2 : ℝ) 1 ∪ {1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2923_292381


namespace NUMINAMATH_CALUDE_doberman_schnauzer_relationship_num_dobermans_proof_l2923_292385

/-- The number of Doberman puppies -/
def num_dobermans : ℝ := 37.5

/-- The number of Schnauzers -/
def num_schnauzers : ℕ := 55

/-- Theorem stating the relationship between Doberman puppies and Schnauzers -/
theorem doberman_schnauzer_relationship : 
  3 * num_dobermans - 5 + (num_dobermans - num_schnauzers) = 90 :=
by sorry

/-- Theorem proving the number of Doberman puppies -/
theorem num_dobermans_proof : num_dobermans = 37.5 :=
by sorry

end NUMINAMATH_CALUDE_doberman_schnauzer_relationship_num_dobermans_proof_l2923_292385


namespace NUMINAMATH_CALUDE_min_sum_squares_l2923_292377

theorem min_sum_squares (a b : ℝ) (ha : a ≠ 0) :
  (∃ x ∈ Set.Icc 3 4, (a + 2) / x = a * x + 2 * b + 1) →
  ∃ m : ℝ, m = 1 / 100 ∧ ∀ a' b' : ℝ, 
    (∃ x ∈ Set.Icc 3 4, (a' + 2) / x = a' * x + 2 * b' + 1) → 
    a' ^ 2 + b' ^ 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2923_292377


namespace NUMINAMATH_CALUDE_sqrt_greater_than_cube_root_l2923_292332

theorem sqrt_greater_than_cube_root (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 + y^2) > (x^3 + y^3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_greater_than_cube_root_l2923_292332


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2923_292349

theorem complex_number_quadrant : ∀ z : ℂ, 
  z = (3 + 4*I) / I → (z.re > 0 ∧ z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2923_292349


namespace NUMINAMATH_CALUDE_point_outside_circle_l2923_292386

/-- Given a circle with center O and radius 3, and a point P such that OP = 5,
    prove that P is outside the circle. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : dist O P = 5) :
  dist O P > r := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2923_292386


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2923_292340

/-- Given that b is an even multiple of 5959, 
    the greatest common divisor of (4b^2 + 73b + 156) and (4b + 15) is 1 -/
theorem gcd_of_quadratic_and_linear (b : ℤ) 
  (h : ∃ k : ℤ, b = 2 * 5959 * k) : 
  Int.gcd (4*b^2 + 73*b + 156) (4*b + 15) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l2923_292340


namespace NUMINAMATH_CALUDE_horner_v2_value_horner_method_correctness_l2923_292364

/-- Horner's method intermediate value -/
def v2 (x : ℝ) : ℝ := 2 * x - 3 * x + 1

/-- The polynomial function -/
def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

theorem horner_v2_value :
  v2 (-1) = 6 :=
by sorry

theorem horner_method_correctness (x : ℝ) :
  f x = (((2*x - 3)*x + 1)*x + 2)*x + 1 :=
by sorry

#check horner_v2_value
#check horner_method_correctness

end NUMINAMATH_CALUDE_horner_v2_value_horner_method_correctness_l2923_292364


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2923_292334

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $262.5 -/
theorem stock_price_calculation :
  final_stock_price 150 1.5 0.3 = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2923_292334


namespace NUMINAMATH_CALUDE_pittsburgh_schools_l2923_292372

theorem pittsburgh_schools (
  pittsburgh_stores : ℕ)
  (pittsburgh_hospitals : ℕ)
  (pittsburgh_police : ℕ)
  (new_city_total : ℕ) :
  pittsburgh_stores = 2000 →
  pittsburgh_hospitals = 500 →
  pittsburgh_police = 20 →
  new_city_total = 2175 →
  ∃ (pittsburgh_schools : ℕ),
    pittsburgh_schools = 200 ∧
    new_city_total = 
      pittsburgh_stores / 2 + 
      pittsburgh_hospitals * 2 + 
      (pittsburgh_schools - 50) + 
      (pittsburgh_police + 5) :=
by sorry

end NUMINAMATH_CALUDE_pittsburgh_schools_l2923_292372


namespace NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l2923_292320

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a polygon with 150 sides is 11025 -/
theorem diagonals_150_sided_polygon : num_diagonals 150 = 11025 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_150_sided_polygon_l2923_292320


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_9_l2923_292304

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

theorem smallest_perfect_square_divisible_by_4_and_9 :
  ∀ n : ℕ, n > 0 → is_perfect_square n → n % 4 = 0 → n % 9 = 0 → n ≥ 36 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_4_and_9_l2923_292304


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2923_292367

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence
  a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2923_292367


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2923_292390

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    if there exists a point P on the ellipse such that the angle F₁PF₂ is 60°,
    then the eccentricity e is in the range [1/2, 1). -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ (e : ℝ), e = Real.sqrt (1 - b^2 / a^2) ∧
    ∃ (F1 F2 : ℝ × ℝ),
      F1 = (-a * e, 0) ∧ F2 = (a * e, 0) ∧
      Real.cos (60 * π / 180) = ((x - F1.1)^2 + (y - F1.2)^2 + (x - F2.1)^2 + (y - F2.2)^2 - 4 * a^2 * e^2) /
        (2 * Real.sqrt ((x - F1.1)^2 + (y - F1.2)^2) * Real.sqrt ((x - F2.1)^2 + (y - F2.2)^2))) →
  1/2 ≤ e ∧ e < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2923_292390


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2923_292325

/-- Given a geometric sequence where a₅ = 16 and a₈ = 4√2, prove that a₁₁ = 2√2 -/
theorem geometric_sequence_11th_term 
  (a : ℕ → ℝ) 
  (is_geometric : ∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n) ^ m)
  (a5 : a 5 = 16)
  (a8 : a 8 = 4 * Real.sqrt 2) :
  a 11 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l2923_292325


namespace NUMINAMATH_CALUDE_total_income_is_53_l2923_292300

def tshirt_price : ℕ := 5
def pants_price : ℕ := 4
def skirt_price : ℕ := 6
def refurbished_tshirt_price : ℕ := tshirt_price / 2

def tshirts_sold : ℕ := 2
def pants_sold : ℕ := 1
def skirts_sold : ℕ := 4
def refurbished_tshirts_sold : ℕ := 6

def total_income : ℕ := 
  tshirts_sold * tshirt_price + 
  pants_sold * pants_price + 
  skirts_sold * skirt_price + 
  refurbished_tshirts_sold * refurbished_tshirt_price

theorem total_income_is_53 : total_income = 53 := by
  sorry

end NUMINAMATH_CALUDE_total_income_is_53_l2923_292300


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2923_292316

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2923_292316


namespace NUMINAMATH_CALUDE_x_value_proof_l2923_292359

def sum_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem x_value_proof (x y : ℕ) : 
  x = sum_integers 10 20 → 
  y = count_even_integers 10 20 → 
  x + y = 171 → 
  x = 165 := by sorry

end NUMINAMATH_CALUDE_x_value_proof_l2923_292359


namespace NUMINAMATH_CALUDE_prime_factor_difference_l2923_292347

theorem prime_factor_difference (a b : ℕ) : 
  Prime a → Prime b → b > a → 
  456456 = 2^3 * a * 7 * 11 * 13 * b → 
  b - a = 16 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_difference_l2923_292347


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2923_292324

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : a 3 = 3/2) 
  (h2 : S 3 = 9/2) : 
  ∃ q : ℚ, (q = 1 ∨ q = -1/2) ∧ 
    (∀ n : ℕ, n ≥ 1 → a n = a 1 * q^(n-1)) ∧
    (∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2923_292324


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l2923_292318

/-- The distance between the foci of the ellipse x^2/36 + y^2/9 = 5 is 6√15 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2/36 + y^2/9 = 5}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, ‖p - f₁‖ + ‖p - f₂‖ = 2 * Real.sqrt (180 : ℝ) ∧
    ‖f₁ - f₂‖ = 6 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l2923_292318


namespace NUMINAMATH_CALUDE_tax_reduction_problem_l2923_292313

theorem tax_reduction_problem (T C : ℝ) (X : ℝ) 
  (h1 : X > 0 ∧ X < 100) -- Ensure X is a valid percentage
  (h2 : T > 0 ∧ C > 0)   -- Ensure initial tax and consumption are positive
  (h3 : T * (1 - X / 100) * C * 1.25 = 0.75 * T * C) -- Revenue equation
  : X = 40 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_problem_l2923_292313


namespace NUMINAMATH_CALUDE_fraction_sum_l2923_292305

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : 
  m + n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2923_292305


namespace NUMINAMATH_CALUDE_cookie_making_time_l2923_292379

/-- Given the total time to make cookies, baking time, and icing hardening times,
    prove that the time for making dough and cooling cookies is 45 minutes. -/
theorem cookie_making_time (total_time baking_time white_icing_time chocolate_icing_time : ℕ)
  (h1 : total_time = 120)
  (h2 : baking_time = 15)
  (h3 : white_icing_time = 30)
  (h4 : chocolate_icing_time = 30) :
  total_time - (baking_time + white_icing_time + chocolate_icing_time) = 45 := by
  sorry

#check cookie_making_time

end NUMINAMATH_CALUDE_cookie_making_time_l2923_292379


namespace NUMINAMATH_CALUDE_jo_age_proof_l2923_292361

theorem jo_age_proof (j d g : ℕ) : 
  (∃ (x y z : ℕ), j = 2 * x ∧ d = 2 * y ∧ g = 2 * z) →  -- ages are even
  j * d * g = 2024 →                                   -- product of ages is 2024
  j ≥ d ∧ j ≥ g →                                      -- Jo's age is the largest
  j = 46 :=                                            -- Jo's age is 46
by sorry

end NUMINAMATH_CALUDE_jo_age_proof_l2923_292361


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2923_292309

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℚ),
    ∀ (x : ℚ), x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 9) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6) ∧
      P = -8/15 ∧ Q = -7/6 ∧ R = 27/10 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2923_292309


namespace NUMINAMATH_CALUDE_complex_square_roots_l2923_292389

theorem complex_square_roots (z : ℂ) : z^2 = -45 - 54*I ↔ z = 3 - 9*I ∨ z = -3 + 9*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_roots_l2923_292389


namespace NUMINAMATH_CALUDE_sally_sunday_sandwiches_l2923_292345

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of pieces of bread used per sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

theorem sally_sunday_sandwiches :
  sunday_sandwiches = total_bread / bread_per_sandwich - saturday_sandwiches :=
sorry

end NUMINAMATH_CALUDE_sally_sunday_sandwiches_l2923_292345


namespace NUMINAMATH_CALUDE_class_attendance_multiple_l2923_292317

/-- Proves that the largest whole number multiple of students present yesterday
    that is less than or equal to 90% of students attending today is 0. -/
theorem class_attendance_multiple (total_registered : ℕ) (present_yesterday : ℕ) (absent_today : ℕ)
  (h1 : total_registered = 156)
  (h2 : present_yesterday = 70)
  (h3 : absent_today = 30)
  (h4 : present_yesterday > absent_today) :
  (∀ n : ℕ, n * present_yesterday ≤ (present_yesterday - absent_today) * 9 / 10 → n = 0) :=
by sorry

end NUMINAMATH_CALUDE_class_attendance_multiple_l2923_292317


namespace NUMINAMATH_CALUDE_servant_payment_is_40_l2923_292368

/-- Calculates the cash payment to a servant who leaves early -/
def servant_cash_payment (total_yearly_salary : ℚ) (turban_price : ℚ) (months_worked : ℚ) : ℚ :=
  (total_yearly_salary * (months_worked / 12)) - turban_price

/-- Proof that the servant receives Rs. 40 in cash -/
theorem servant_payment_is_40 :
  let total_yearly_salary : ℚ := 200
  let turban_price : ℚ := 110
  let months_worked : ℚ := 9
  servant_cash_payment total_yearly_salary turban_price months_worked = 40 := by
sorry

#eval servant_cash_payment 200 110 9

end NUMINAMATH_CALUDE_servant_payment_is_40_l2923_292368


namespace NUMINAMATH_CALUDE_combined_tennis_preference_l2923_292333

theorem combined_tennis_preference (east_total : ℕ) (west_total : ℕ) 
  (east_tennis_percent : ℚ) (west_tennis_percent : ℚ) :
  east_total = 2000 →
  west_total = 2500 →
  east_tennis_percent = 22 / 100 →
  west_tennis_percent = 40 / 100 →
  (east_total * east_tennis_percent + west_total * west_tennis_percent) / 
  (east_total + west_total) = 32 / 100 :=
by sorry

end NUMINAMATH_CALUDE_combined_tennis_preference_l2923_292333


namespace NUMINAMATH_CALUDE_defect_probability_l2923_292302

/-- The probability of a randomly chosen unit being defective from two machines -/
theorem defect_probability
  (machine_a_ratio : ℝ)
  (machine_b_ratio : ℝ)
  (machine_a_defect_rate : ℝ)
  (machine_b_defect_rate : ℝ)
  (h1 : machine_a_ratio = 0.4)
  (h2 : machine_b_ratio = 0.6)
  (h3 : machine_a_ratio + machine_b_ratio = 1)
  (h4 : machine_a_defect_rate = 9 / 1000)
  (h5 : machine_b_defect_rate = 1 / 50) :
  machine_a_ratio * machine_a_defect_rate + machine_b_ratio * machine_b_defect_rate = 0.0156 :=
by sorry


end NUMINAMATH_CALUDE_defect_probability_l2923_292302


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l2923_292338

/-- Given a triangle ABC, this theorem proves the position of point Q
    based on the given ratios of points G and H on the sides of the triangle. -/
theorem intersection_point_coordinates (A B C G H Q : ℝ × ℝ) : 
  (∃ t : ℝ, G = (1 - t) • A + t • B ∧ t = 2/5) →
  (∃ s : ℝ, H = (1 - s) • B + s • C ∧ s = 3/4) →
  (∃ r : ℝ, Q = (1 - r) • A + r • G) →
  (∃ u : ℝ, Q = (1 - u) • C + u • H) →
  Q = (3/8) • A + (1/4) • B + (3/8) • C :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l2923_292338


namespace NUMINAMATH_CALUDE_complex_equation_result_l2923_292398

theorem complex_equation_result (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (a + 4 * i) * i = b + i) : a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2923_292398


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2923_292371

theorem geometric_series_ratio (a : ℝ) (r : ℝ) : 
  (∃ S : ℝ, S = a / (1 - r) ∧ S = 81 * (a * r^4 / (1 - r))) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2923_292371


namespace NUMINAMATH_CALUDE_no_valid_a_l2923_292344

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_a_l2923_292344


namespace NUMINAMATH_CALUDE_circumcircle_radius_is_13_l2923_292321

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Ratio of the shorter base to the longer base -/
  base_ratio : ℚ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- The midline of the trapezoid equals its height -/
  midline_eq_height : True

/-- Calculate the radius of the circumcircle of an isosceles trapezoid -/
def circumcircle_radius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with given properties, its circumcircle radius is 13 -/
theorem circumcircle_radius_is_13 (t : IsoscelesTrapezoid) 
  (h1 : t.base_ratio = 5 / 12)
  (h2 : t.height = 17) : 
  circumcircle_radius t = 13 := by
  sorry

end NUMINAMATH_CALUDE_circumcircle_radius_is_13_l2923_292321


namespace NUMINAMATH_CALUDE_pyramid_face_area_l2923_292310

/-- The total area of triangular faces of a right square-based pyramid -/
theorem pyramid_face_area (base_edge : ℝ) (lateral_edge : ℝ) : 
  base_edge = 8 → lateral_edge = 9 → 
  (4 * (1/2 * base_edge * Real.sqrt ((lateral_edge^2) - (base_edge/2)^2))) = 16 * Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_face_area_l2923_292310


namespace NUMINAMATH_CALUDE_count_numbers_3000_l2923_292378

/-- Returns true if the given number contains the digit '2' in its base-10 representation -/
def contains_two (n : ℕ) : Bool :=
  sorry

/-- Returns the count of numbers less than or equal to n that contain '2' and are divisible by 3 -/
def count_numbers (n : ℕ) : ℕ :=
  sorry

theorem count_numbers_3000 : count_numbers 3000 = 384 :=
  sorry

end NUMINAMATH_CALUDE_count_numbers_3000_l2923_292378


namespace NUMINAMATH_CALUDE_equilateral_triangles_count_l2923_292319

/-- Counts the number of equilateral triangles in an equilateral triangular grid -/
def count_equilateral_triangles (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) * (n + 3) / 24

/-- The side length of the equilateral triangular grid -/
def grid_side_length : ℕ := 4

/-- Theorem: The number of equilateral triangles in a grid of side length 4 is 35 -/
theorem equilateral_triangles_count :
  count_equilateral_triangles grid_side_length = 35 := by
  sorry

#eval count_equilateral_triangles grid_side_length

end NUMINAMATH_CALUDE_equilateral_triangles_count_l2923_292319


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extreme_l2923_292326

open Real

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem f_monotonicity_and_extreme :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, f x ≤ f 1) ∧
  (f 1 = 1 / Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extreme_l2923_292326


namespace NUMINAMATH_CALUDE_perimeter_area_ratio_bound_l2923_292343

/-- A shape in the plane formed by a union of finitely many unit squares -/
structure UnitSquareShape where
  squares : Finset (ℤ × ℤ)

/-- The perimeter of a UnitSquareShape -/
def perimeter (S : UnitSquareShape) : ℝ := sorry

/-- The area of a UnitSquareShape -/
def area (S : UnitSquareShape) : ℝ := S.squares.card

/-- The theorem stating that the ratio of perimeter to area is at most 8 -/
theorem perimeter_area_ratio_bound (S : UnitSquareShape) :
  perimeter S / area S ≤ 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_area_ratio_bound_l2923_292343


namespace NUMINAMATH_CALUDE_reflection_sum_l2923_292322

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (8, 7), then m + b = 11 -/
theorem reflection_sum (m b : ℝ) : 
  (∀ (x y : ℝ), y = m * x + b → 
    (x - 2) * (x - 8) + (y - 3) * (y - 7) = 0 ∧ 
    (x - 5) * (1 + m * m) = m * (y - 5)) → 
  m + b = 11 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_l2923_292322


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2923_292383

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the arithmetic sequence a_n
def a (n : ℕ) (d : ℝ) : ℝ := 2*n + 1

theorem arithmetic_sequence_formula (d : ℝ) :
  (∀ n : ℕ, a (n + 1) d - a n d = d) ∧
  a 1 d = f (d - 1) ∧
  a 3 d = f (d + 1) →
  ∀ n : ℕ, a n d = 2*n + 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2923_292383


namespace NUMINAMATH_CALUDE_unique_solution_for_alpha_minus_one_l2923_292342

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : RealFunction) (α : ℝ) : Prop :=
  ∀ x y : ℝ, f (f (x + y)) = f (x + y) + f x * f y + α * x * y

-- State the theorem
theorem unique_solution_for_alpha_minus_one (α : ℝ) (hα : α ≠ 0) :
  (∃ f : RealFunction, SatisfiesEquation f α) ↔ (α = -1 ∧ ∃ f : RealFunction, SatisfiesEquation f (-1) ∧ ∀ x, f x = x) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_alpha_minus_one_l2923_292342


namespace NUMINAMATH_CALUDE_expression_factorization_l2923_292312

theorem expression_factorization (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2923_292312


namespace NUMINAMATH_CALUDE_john_guests_correct_l2923_292397

/-- The number of guests John wants for his wedding. -/
def john_guests : ℕ := 50

/-- The venue cost for the wedding. -/
def venue_cost : ℕ := 10000

/-- The cost per guest for the wedding. -/
def cost_per_guest : ℕ := 500

/-- The total cost of the wedding if John's wife gets her way. -/
def total_cost : ℕ := 50000

/-- Theorem stating that the number of guests John wants is correct. -/
theorem john_guests_correct :
  venue_cost + cost_per_guest * (john_guests + (60 * john_guests) / 100) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_john_guests_correct_l2923_292397


namespace NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l2923_292337

theorem a_less_than_b_less_than_one
  (x a b : ℝ)
  (hx : x > 0)
  (ha : a > 0)
  (hb : b > 0)
  (h : a^x < b^x ∧ b^x < 1) :
  a < b ∧ b < 1 :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_b_less_than_one_l2923_292337


namespace NUMINAMATH_CALUDE_largest_value_proof_l2923_292355

theorem largest_value_proof (a b c d e : ℚ) :
  a = 0.9387 →
  b = 0.9381 →
  c = 9385 / 10000 →
  d = 0.9379 →
  e = 0.9389 →
  max a (max b (max c (max d e))) = e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_proof_l2923_292355


namespace NUMINAMATH_CALUDE_willy_tv_series_completion_time_l2923_292323

/-- The number of days required to finish a TV series -/
def days_to_finish (seasons : ℕ) (episodes_per_season : ℕ) (episodes_per_day : ℕ) : ℕ :=
  (seasons * episodes_per_season) / episodes_per_day

/-- Theorem: It takes 30 days to finish the TV series under given conditions -/
theorem willy_tv_series_completion_time :
  days_to_finish 3 20 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_willy_tv_series_completion_time_l2923_292323


namespace NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_equation_l2923_292329

theorem new_energy_vehicle_sales_growth_equation 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (growth_period : ℕ) 
  (x : ℝ) 
  (h1 : initial_sales = 298) 
  (h2 : final_sales = 850) 
  (h3 : growth_period = 2) :
  initial_sales * (1 + x)^growth_period = final_sales :=
by sorry

end NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_equation_l2923_292329


namespace NUMINAMATH_CALUDE_g_sum_negative_one_l2923_292393

noncomputable section

variable (f g : ℝ → ℝ)

axiom functional_equation : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y
axiom f_equality : f (-2) = f 1
axiom f_nonzero : f 1 ≠ 0

theorem g_sum_negative_one : g 1 + g (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_negative_one_l2923_292393


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2923_292387

theorem geometric_sequence_problem (a : ℝ) :
  a > 0 ∧
  (∃ (r : ℝ), 210 * r = a ∧ a * r = 63 / 40) →
  a = 18.1875 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2923_292387


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_distinct_roots_l2923_292366

/-- The quadratic equation (a-3)x^2 - 4x - 1 = 0 -/
def quadratic_equation (a : ℝ) (x : ℝ) : Prop :=
  (a - 3) * x^2 - 4 * x - 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ :=
  16 - 4 * (a - 3) * (-1)

theorem quadratic_equation_roots (a : ℝ) :
  (∃ x : ℝ, quadratic_equation a x ∧
    (∀ y : ℝ, quadratic_equation a y → y = x)) →
  a = -1 ∧ (∀ x : ℝ, quadratic_equation a x → x = -1/2) :=
sorry

theorem quadratic_equation_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation a x ∧ quadratic_equation a y) →
  (a > -1 ∧ a ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_distinct_roots_l2923_292366


namespace NUMINAMATH_CALUDE_adult_admission_fee_if_all_receipts_from_adults_l2923_292374

/-- Proves that if all receipts came from adult tickets, the adult admission fee would be the total receipts divided by the number of adults -/
theorem adult_admission_fee_if_all_receipts_from_adults 
  (total_attendees : ℕ) 
  (total_receipts : ℚ) 
  (num_adults : ℕ) 
  (h1 : total_attendees = 578)
  (h2 : total_receipts = 985)
  (h3 : num_adults = 342)
  (h4 : num_adults ≤ total_attendees)
  (h5 : num_adults > 0) :
  let adult_fee := total_receipts / num_adults
  adult_fee * num_adults = total_receipts :=
by sorry

#eval (985 : ℚ) / 342

end NUMINAMATH_CALUDE_adult_admission_fee_if_all_receipts_from_adults_l2923_292374


namespace NUMINAMATH_CALUDE_inequality_solution_l2923_292348

def linear_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

def negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f x < 0

theorem inequality_solution 
  (f : ℝ → ℝ) 
  (h_linear : linear_function f) 
  (h_neg : negative_for_positive f) 
  (n : ℕ) 
  (hn : n > 0) 
  (a : ℝ) 
  (ha : a < 0) :
  ∀ x : ℝ, 
    (1 / n : ℝ) * f (a * x^2) - f x > (1 / n : ℝ) * f (a^2 * x) - f a ↔ 
      (a < -Real.sqrt n ∧ (x > n / a ∨ x < a)) ∨ 
      (a = -Real.sqrt n ∧ x ≠ -Real.sqrt n) ∨ 
      (-Real.sqrt n < a ∧ (x > a ∨ x < n / a)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2923_292348


namespace NUMINAMATH_CALUDE_total_machines_is_five_l2923_292308

/-- Represents the production scenario with new and old machines -/
structure ProductionScenario where
  totalProduction : ℕ
  newMachineProduction : ℕ
  oldMachineProduction : ℕ
  totalMachines : ℕ

/-- Represents the conditions of the production problem -/
def productionProblem : ProductionScenario → Prop
  | s => s.totalProduction = 9000 ∧
         s.oldMachineProduction = s.newMachineProduction / 2 ∧
         s.totalProduction = (s.totalMachines - 1) * s.newMachineProduction + s.oldMachineProduction

/-- Represents the scenario if the old machine is replaced -/
def replacedScenario (s : ProductionScenario) : ProductionScenario :=
  { totalProduction := s.totalProduction
  , newMachineProduction := s.newMachineProduction - 200
  , oldMachineProduction := s.newMachineProduction - 200
  , totalMachines := s.totalMachines }

/-- The main theorem stating that the total number of machines is 5 -/
theorem total_machines_is_five :
  ∃ s : ProductionScenario, productionProblem s ∧
    productionProblem (replacedScenario s) ∧
    s.totalMachines = 5 := by
  sorry


end NUMINAMATH_CALUDE_total_machines_is_five_l2923_292308


namespace NUMINAMATH_CALUDE_zach_scored_42_points_l2923_292303

def ben_points : ℝ := 21.0
def total_points : ℝ := 63

def zach_points : ℝ := total_points - ben_points

theorem zach_scored_42_points :
  zach_points = 42 := by sorry

end NUMINAMATH_CALUDE_zach_scored_42_points_l2923_292303


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_l2923_292307

/-- Represents a square lattice made of toothpicks -/
structure SquareLattice where
  size : Nat
  total_toothpicks : Nat
  boundary_toothpicks : Nat
  internal_grid_toothpicks : Nat
  diagonal_toothpicks : Nat

/-- Theorem: Minimum number of toothpicks to remove to eliminate all squares and triangles -/
theorem min_toothpicks_to_remove (lattice : SquareLattice) 
  (h1 : lattice.size = 3)
  (h2 : lattice.total_toothpicks = 40)
  (h3 : lattice.boundary_toothpicks = 12)
  (h4 : lattice.internal_grid_toothpicks = 4)
  (h5 : lattice.diagonal_toothpicks = 12)
  (h6 : lattice.boundary_toothpicks + lattice.internal_grid_toothpicks + lattice.diagonal_toothpicks = lattice.total_toothpicks) :
  ∃ (n : Nat), n = lattice.boundary_toothpicks + lattice.internal_grid_toothpicks ∧ 
               n = 16 ∧
               (∀ m : Nat, m < n → ∃ (square : Bool) (triangle : Bool), square ∨ triangle) :=
by sorry


end NUMINAMATH_CALUDE_min_toothpicks_to_remove_l2923_292307


namespace NUMINAMATH_CALUDE_positive_correlation_missing_data_point_l2923_292341

-- Define the regression line
def regression_line (x : ℝ) : ℝ := 6.5 * x + 17.5

-- Define the data points
def data_points : List (ℝ × ℝ) := [(2, 30), (4, 40), (5, 60), (6, 50), (8, 70)]

-- Theorem 1: Positive correlation
theorem positive_correlation : 
  ∀ x₁ x₂, x₁ < x₂ → regression_line x₁ < regression_line x₂ :=
by sorry

-- Theorem 2: Missing data point
theorem missing_data_point : 
  ∃ y, (2, y) ∈ data_points ∧ y = 30 :=
by sorry

end NUMINAMATH_CALUDE_positive_correlation_missing_data_point_l2923_292341


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_10_l2923_292376

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → (d = 1 ∨ d = 2)

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_number_with_digit_sum_10 :
  ∀ n : ℕ,
    is_valid_number n ∧ digit_sum n = 10 →
    111111112 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_10_l2923_292376


namespace NUMINAMATH_CALUDE_cafe_cake_division_l2923_292369

theorem cafe_cake_division (total_cake : ℚ) (tom_portion bob_portion jerry_portion : ℚ) :
  total_cake = 8/9 →
  tom_portion = 2 * bob_portion →
  tom_portion = 2 * jerry_portion →
  total_cake = tom_portion + bob_portion + jerry_portion →
  bob_portion = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_cafe_cake_division_l2923_292369


namespace NUMINAMATH_CALUDE_couch_cost_is_750_l2923_292350

/-- The cost of the couch Daria bought -/
def couch_cost : ℕ := sorry

/-- The amount Daria has saved -/
def savings : ℕ := 500

/-- The cost of the table Daria bought -/
def table_cost : ℕ := 100

/-- The cost of the lamp Daria bought -/
def lamp_cost : ℕ := 50

/-- The amount Daria still owes after paying her savings -/
def remaining_debt : ℕ := 400

/-- Theorem stating that the couch cost is $750 -/
theorem couch_cost_is_750 :
  couch_cost = 750 ∧
  couch_cost + table_cost + lamp_cost = savings + remaining_debt :=
sorry

end NUMINAMATH_CALUDE_couch_cost_is_750_l2923_292350


namespace NUMINAMATH_CALUDE_equation_solutions_l2923_292314

theorem equation_solutions :
  (∀ x : ℝ, (x + 4) * (x - 2) = 3 * (x - 2) ↔ x = -1 ∨ x = 2) ∧
  (∀ x : ℝ, x^2 - x - 3 = 0 ↔ x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2923_292314


namespace NUMINAMATH_CALUDE_complex_addition_simplification_l2923_292360

theorem complex_addition_simplification :
  (7 - 4 * Complex.I) + (3 + 9 * Complex.I) = 10 + 5 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_addition_simplification_l2923_292360


namespace NUMINAMATH_CALUDE_football_cost_l2923_292336

def total_cost : ℝ := 20.52
def marbles_cost : ℝ := 9.05
def baseball_cost : ℝ := 6.52

theorem football_cost :
  ∃ (football_cost : ℝ),
    football_cost = total_cost - marbles_cost - baseball_cost ∧
    football_cost = 5.45 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_l2923_292336


namespace NUMINAMATH_CALUDE_circle_equation_equivalence_l2923_292382

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 + y^2 - 2*x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_equivalence_l2923_292382


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l2923_292356

theorem points_on_line_procedure (x : ℕ) : ∃ x, 9 * x - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l2923_292356


namespace NUMINAMATH_CALUDE_trig_product_equals_one_l2923_292384

theorem trig_product_equals_one (θ : Real) (h : 0 < θ ∧ θ < π / 2) :
  (1 - 1 / Real.cos θ) * (1 + 1 / Real.sin (π / 2 - θ)) *
  (1 - 1 / Real.sin θ) * (1 + 1 / Real.cos (π / 2 - θ)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_l2923_292384


namespace NUMINAMATH_CALUDE_simplify_expression_l2923_292351

theorem simplify_expression (a x y : ℝ) : a^2 * x^2 - a^2 * y^2 = a^2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2923_292351


namespace NUMINAMATH_CALUDE_frankie_pet_count_l2923_292394

/-- Represents the number of pets Frankie has of each type -/
structure PetCounts where
  dogs : Nat
  cats : Nat
  parrots : Nat
  snakes : Nat

/-- Calculates the total number of pets -/
def totalPets (p : PetCounts) : Nat :=
  p.dogs + p.cats + p.parrots + p.snakes

/-- Represents the conditions given in the problem -/
structure PetConditions (p : PetCounts) : Prop where
  dog_count : p.dogs = 2
  four_legged : p.dogs + p.cats = 6
  parrot_count : p.parrots = p.cats - 1
  snake_count : p.snakes = p.cats + 6

/-- Theorem stating that given the conditions, Frankie has 19 pets in total -/
theorem frankie_pet_count (p : PetCounts) (h : PetConditions p) : totalPets p = 19 := by
  sorry


end NUMINAMATH_CALUDE_frankie_pet_count_l2923_292394


namespace NUMINAMATH_CALUDE_expand_expression_l2923_292301

theorem expand_expression (x : ℝ) : (17 * x + 18) * (3 * x + 4) = 51 * x^2 + 122 * x + 72 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2923_292301


namespace NUMINAMATH_CALUDE_integer_quotient_problem_l2923_292330

theorem integer_quotient_problem (x y : ℤ) : 
  1996 * x + y / 96 = x + y → y / x = 2016 ∨ x / y = 1 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_integer_quotient_problem_l2923_292330


namespace NUMINAMATH_CALUDE_twenty_bees_honey_production_l2923_292346

/-- The amount of honey (in grams) produced by a given number of bees in 20 days. -/
def honey_production (num_bees : ℕ) : ℝ :=
  num_bees * 1

/-- Theorem stating that 20 honey bees produce 20 grams of honey in 20 days. -/
theorem twenty_bees_honey_production :
  honey_production 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_bees_honey_production_l2923_292346


namespace NUMINAMATH_CALUDE_no_g_sequence_to_nine_l2923_292354

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n^3 + 9 else n / 2

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem no_g_sequence_to_nine :
  ∀ n : ℤ, -100 ≤ n ∧ n ≤ 100 → ¬∃ k : ℕ, iterateG n k = 9 :=
by sorry

end NUMINAMATH_CALUDE_no_g_sequence_to_nine_l2923_292354


namespace NUMINAMATH_CALUDE_ice_cream_volume_l2923_292311

/-- The volume of ice cream on a cone -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ)
  (h_cone_pos : h_cone > 0)
  (r_pos : r > 0)
  (h_cylinder_pos : h_cylinder > 0) :
  let v_cone := (1 / 3) * π * r^2 * h_cone
  let v_cylinder := π * r^2 * h_cylinder
  let v_hemisphere := (2 / 3) * π * r^3
  v_cylinder + v_hemisphere = 14.25 * π :=
by
  sorry

#check ice_cream_volume 10 1.5 2

end NUMINAMATH_CALUDE_ice_cream_volume_l2923_292311


namespace NUMINAMATH_CALUDE_perfect_div_by_three_perfect_div_by_seven_l2923_292375

/-- Definition of a perfect number -/
def isPerfect (n : ℕ) : Prop :=
  n > 0 ∧ n = (Finset.filter (· < n) (Finset.range (n + 1))).sum id

/-- Theorem for perfect numbers divisible by 3 -/
theorem perfect_div_by_three (n : ℕ) (h1 : isPerfect n) (h2 : n > 6) (h3 : 3 ∣ n) : 9 ∣ n := by
  sorry

/-- Theorem for perfect numbers divisible by 7 -/
theorem perfect_div_by_seven (n : ℕ) (h1 : isPerfect n) (h2 : n > 28) (h3 : 7 ∣ n) : 49 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_perfect_div_by_three_perfect_div_by_seven_l2923_292375


namespace NUMINAMATH_CALUDE_operation_difference_l2923_292388

def operation (x y : ℤ) : ℤ := x * y - 2 * x + 3 * y

theorem operation_difference : operation 7 4 - operation 4 7 = -15 := by
  sorry

end NUMINAMATH_CALUDE_operation_difference_l2923_292388


namespace NUMINAMATH_CALUDE_triangle_area_and_angle_C_l2923_292380

noncomputable section

def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_area_and_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle A B C a b c)
  (h_cos_B : Real.cos B = 3/5)
  (h_dot_product : a * c * Real.cos B = -21) :
  let S := (1/2) * a * c * Real.sin B
  ∃ (S : ℝ), S = 14 ∧ 
  (a = 7 → C = Real.pi/4) := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_area_and_angle_C_l2923_292380


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2923_292358

/-- Sam's current age -/
def s : ℕ := by sorry

/-- Anna's current age -/
def a : ℕ := by sorry

/-- The number of years until the ratio of their ages is 3:2 -/
def x : ℕ := by sorry

theorem age_ratio_problem :
  (s - 3 = 4 * (a - 3)) ∧ 
  (s - 5 = 6 * (a - 5)) →
  (x = 22 ∧ (s + x) * 2 = (a + x) * 3) := by sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2923_292358


namespace NUMINAMATH_CALUDE_professor_percentage_l2923_292335

theorem professor_percentage (total : ℝ) (women_percent : ℝ) (tenured_percent : ℝ) (men_tenured_percent : ℝ) :
  women_percent = 70 →
  tenured_percent = 70 →
  men_tenured_percent = 50 →
  let women := total * (women_percent / 100)
  let tenured := total * (tenured_percent / 100)
  let men := total - women
  let men_tenured := men * (men_tenured_percent / 100)
  let women_tenured := tenured - men_tenured
  let women_or_tenured := women + tenured - women_tenured
  (women_or_tenured / total) * 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_professor_percentage_l2923_292335


namespace NUMINAMATH_CALUDE_mod_inverse_sum_17_l2923_292357

theorem mod_inverse_sum_17 :
  ∃ (a b : ℤ), (2 * a) % 17 = 1 ∧ (4 * b) % 17 = 1 ∧ (a + b) % 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_inverse_sum_17_l2923_292357


namespace NUMINAMATH_CALUDE_f_properties_l2923_292328

noncomputable def f (x : ℝ) := Real.sin x * (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem f_properties :
  ∃ (T : ℝ) (M : ℝ) (S : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (∀ t, 0 < t → t < T → ¬ (∀ x, f (x + t) = f x)) ∧
    (∀ x, f x ≤ M) ∧
    (∃ x, f x = M) ∧
    T = π ∧
    M = 3/2 ∧
    (∀ A B C a b c : ℝ,
      0 < A ∧ A < π/2 ∧
      0 < B ∧ B < π/2 ∧
      0 < C ∧ C < π/2 ∧
      A + B + C = π ∧
      f (A/2) = 1 ∧
      a = 2 * Real.sqrt 3 ∧
      a = b * Real.sin C ∧
      b = c * Real.sin A ∧
      c = a * Real.sin B →
      1/2 * b * c * Real.sin A ≤ S) ∧
    S = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2923_292328


namespace NUMINAMATH_CALUDE_M_greater_than_N_l2923_292391

theorem M_greater_than_N : ∀ a : ℝ, 2 * a * (a - 2) > (a + 1) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_M_greater_than_N_l2923_292391


namespace NUMINAMATH_CALUDE_account_balance_first_year_l2923_292315

/-- Proves that given an initial deposit and interest accrued, the account balance
    at the end of the first year is the sum of the initial deposit and interest accrued. -/
theorem account_balance_first_year
  (initial_deposit : ℝ)
  (interest_accrued : ℝ)
  (h1 : initial_deposit = 1000)
  (h2 : interest_accrued = 100) :
  initial_deposit + interest_accrued = 1100 := by
  sorry

end NUMINAMATH_CALUDE_account_balance_first_year_l2923_292315


namespace NUMINAMATH_CALUDE_similar_right_triangle_longest_side_l2923_292327

theorem similar_right_triangle_longest_side
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_sides : a = 8 ∧ b = 15 ∧ c = 17)
  (k : ℝ)
  (h_perimeter : k * (a + b + c) = 160)
  : k * c = 68 :=
by sorry

end NUMINAMATH_CALUDE_similar_right_triangle_longest_side_l2923_292327
