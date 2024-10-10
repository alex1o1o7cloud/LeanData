import Mathlib

namespace tiffany_bags_on_monday_l2684_268489

/-- The number of bags Tiffany had on Monday -/
def bags_on_monday : ℕ := sorry

/-- The number of bags Tiffany found on Tuesday -/
def bags_on_tuesday : ℕ := 4

/-- The total number of bags Tiffany had -/
def total_bags : ℕ := 8

/-- Theorem: Tiffany had 4 bags on Monday -/
theorem tiffany_bags_on_monday : bags_on_monday = 4 := by
  sorry

end tiffany_bags_on_monday_l2684_268489


namespace number_problem_l2684_268422

theorem number_problem : ∃! x : ℚ, (1 / 4 : ℚ) * x > (1 / 5 : ℚ) * (x + 1) + 1 := by
  sorry

end number_problem_l2684_268422


namespace days_2000_to_2005_l2684_268464

/-- The number of days in a given range of years -/
def totalDays (totalYears : ℕ) (leapYears : ℕ) (nonLeapDays : ℕ) (leapDays : ℕ) : ℕ :=
  (totalYears - leapYears) * nonLeapDays + leapYears * leapDays

/-- Theorem stating that the total number of days from 2000 to 2005 (inclusive) is 2192 -/
theorem days_2000_to_2005 : totalDays 6 2 365 366 = 2192 := by
  sorry

end days_2000_to_2005_l2684_268464


namespace rational_fraction_implies_integer_sum_squares_over_sum_l2684_268490

theorem rational_fraction_implies_integer_sum_squares_over_sum (a b c : ℕ+) :
  (∃ (r s : ℤ), (r : ℚ) / s = (a * Real.sqrt 3 + b) / (b * Real.sqrt 3 + c)) →
  ∃ (k : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℚ) / (a + b + c) = k := by
sorry

end rational_fraction_implies_integer_sum_squares_over_sum_l2684_268490


namespace sum_of_fractions_minus_ten_equals_zero_l2684_268483

theorem sum_of_fractions_minus_ten_equals_zero :
  5 / 3 + 10 / 6 + 20 / 12 + 40 / 24 + 80 / 48 + 160 / 96 - 10 = 0 := by
  sorry

end sum_of_fractions_minus_ten_equals_zero_l2684_268483


namespace sphere_volume_l2684_268439

theorem sphere_volume (d : ℝ) (a : ℝ) (h1 : d = 2) (h2 : a = π) :
  let r := Real.sqrt (1^2 + d^2)
  (4 / 3) * π * r^3 = (20 * Real.sqrt 5 * π) / 3 := by
sorry

end sphere_volume_l2684_268439


namespace area_ratio_theorem_l2684_268436

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumradius and inradius
def circumradius (t : Triangle) : ℝ := sorry
def inradius (t : Triangle) : ℝ := sorry

-- Define the points A1, B1, C1
def angle_bisector_points (t : Triangle) : Triangle := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

theorem area_ratio_theorem (t : Triangle) :
  let t1 := angle_bisector_points t
  area t / area t1 = 2 * inradius t / circumradius t := by
  sorry

end area_ratio_theorem_l2684_268436


namespace expected_straight_flying_airplanes_l2684_268499

def flyProbability : ℚ := 3/4
def notStraightProbability : ℚ := 5/6
def totalAirplanes : ℕ := 80

theorem expected_straight_flying_airplanes :
  (totalAirplanes : ℚ) * flyProbability * (1 - notStraightProbability) = 10 := by
  sorry

end expected_straight_flying_airplanes_l2684_268499


namespace square_root_fraction_simplification_l2684_268417

theorem square_root_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (64 + 36) = 5 / 2 := by
  sorry

end square_root_fraction_simplification_l2684_268417


namespace two_points_at_distance_from_line_l2684_268412

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance between a point and a line in 3D space -/
def distance_point_to_line (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Check if a line segment is perpendicular to a line in 3D space -/
def is_perpendicular (p1 p2 : ℝ × ℝ × ℝ) (l : Line3D) : Prop :=
  sorry

theorem two_points_at_distance_from_line 
  (L : Line3D) (d : ℝ) (P : ℝ × ℝ × ℝ) :
  ∃ (Q1 Q2 : ℝ × ℝ × ℝ),
    distance_point_to_line Q1 L = d ∧
    distance_point_to_line Q2 L = d ∧
    is_perpendicular P Q1 L ∧
    is_perpendicular P Q2 L :=
  sorry

end two_points_at_distance_from_line_l2684_268412


namespace tangent_circles_count_l2684_268427

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation between two circles
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

theorem tangent_circles_count (c1 c2 : Circle) : 
  c1.radius = 1 →
  c2.radius = 1 →
  are_tangent c1 c2 →
  ∃ (s : Finset Circle), 
    s.card = 6 ∧ 
    (∀ c ∈ s, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2) ∧
    (∀ c : Circle, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2 → c ∈ s) :=
sorry

end tangent_circles_count_l2684_268427


namespace function_symmetric_about_origin_l2684_268497

/-- The function f(x) = x^3 - x is symmetric about the origin. -/
theorem function_symmetric_about_origin (x : ℝ) : let f := λ x : ℝ => x^3 - x
  f (-x) = -f x := by
  sorry

end function_symmetric_about_origin_l2684_268497


namespace arccos_sin_three_equals_three_minus_pi_half_l2684_268470

theorem arccos_sin_three_equals_three_minus_pi_half :
  Real.arccos (Real.sin 3) = 3 - π / 2 := by sorry

end arccos_sin_three_equals_three_minus_pi_half_l2684_268470


namespace perpendicular_vector_l2684_268425

def vector_AB : Fin 2 → ℝ := ![1, 1]
def vector_AC : Fin 2 → ℝ := ![2, 3]
def vector_BC : Fin 2 → ℝ := ![1, 2]
def vector_D : Fin 2 → ℝ := ![-6, 3]

theorem perpendicular_vector : 
  (vector_AB = ![1, 1]) → 
  (vector_AC = ![2, 3]) → 
  (vector_BC = vector_AC - vector_AB) →
  (vector_D • vector_BC = 0) :=
by sorry

end perpendicular_vector_l2684_268425


namespace quadratic_inequality_solution_sets_l2684_268410

theorem quadratic_inequality_solution_sets (m x : ℝ) :
  let f := fun x => m * x^2 - (m + 1) * x + 1
  (m = 2 → (f x < 0 ↔ 1/2 < x ∧ x < 1)) ∧
  (m > 0 →
    ((0 < m ∧ m < 1) → (f x < 0 ↔ 1 < x ∧ x < 1/m)) ∧
    (m = 1 → ¬∃ x, f x < 0) ∧
    (m > 1 → (f x < 0 ↔ 1/m < x ∧ x < 1))) := by
  sorry

end quadratic_inequality_solution_sets_l2684_268410


namespace distance_calculation_l2684_268449

/-- Given a journey time of 8 hours and an average speed of 23 miles per hour,
    the distance traveled is 184 miles. -/
theorem distance_calculation (journey_time : ℝ) (average_speed : ℝ) 
  (h1 : journey_time = 8)
  (h2 : average_speed = 23) :
  journey_time * average_speed = 184 := by
  sorry

end distance_calculation_l2684_268449


namespace prime_and_power_characterization_l2684_268446

theorem prime_and_power_characterization (n : ℕ) (h : n ≥ 2) :
  (Nat.Prime n ↔ n ∣ (Nat.factorial (n - 1) + 1)) ∧
  (∃ k : ℕ, n^k = Nat.factorial (n - 1) + 1 ↔ n = 2 ∨ n = 3) := by
  sorry

end prime_and_power_characterization_l2684_268446


namespace sin_cube_identity_l2684_268484

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = -1/4 * Real.sin (3 * θ) + 3/4 * Real.sin θ := by
  sorry

end sin_cube_identity_l2684_268484


namespace circle_transformation_l2684_268428

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Translates a point to the right by a given amount -/
def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

/-- The main theorem -/
theorem circle_transformation :
  let T : ℝ × ℝ := (-2, 6)
  let reflected := reflect_x T
  let final := translate_right reflected 5
  final = (3, -6) := by sorry

end circle_transformation_l2684_268428


namespace parabola_intersection_l2684_268411

-- Define the two parabolas
def f (x : ℝ) : ℝ := 3*x^2 + 4*x - 5
def g (x : ℝ) : ℝ := x^2 + 11

-- Theorem stating the intersection points
theorem parabola_intersection :
  (∃ (x y : ℝ), f x = g x ∧ y = f x) ↔
  (∃ (x y : ℝ), (x = -4 ∧ y = 27) ∨ (x = 2 ∧ y = 15)) :=
by sorry

end parabola_intersection_l2684_268411


namespace quadratic_roots_range_l2684_268423

theorem quadratic_roots_range (m : ℝ) : 
  m > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ x < 1 ∧ y < 1 ∧ 
    m * x^2 + (2*m - 1) * x - m + 2 = 0 ∧
    m * y^2 + (2*m - 1) * y - m + 2 = 0) →
  m > (3 + Real.sqrt 7) / 4 :=
by sorry

end quadratic_roots_range_l2684_268423


namespace shaded_area_semicircles_l2684_268475

/-- The area of shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircles (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 12 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 9 * π := by
  sorry

end shaded_area_semicircles_l2684_268475


namespace braking_distance_problems_l2684_268402

/-- Braking distance formula -/
def braking_distance (t v k : ℝ) : ℝ := t * v + k * v^2

/-- Braking coefficient -/
def k : ℝ := 0.1

/-- Initial reaction time before alcohol consumption -/
def t_initial : ℝ := 0.5

theorem braking_distance_problems :
  /- (1) -/
  braking_distance t_initial 10 k = 15 ∧
  /- (2) -/
  ∃ t : ℝ, braking_distance t 15 k = 52.5 ∧ t = 2 ∧
  /- (3) -/
  braking_distance 2 10 k = 30 ∧
  /- (4) -/
  braking_distance 2 10 k - braking_distance t_initial 10 k = 15 ∧
  /- (5) -/
  ∀ t : ℝ, braking_distance t 12 k < 42 → t < 2.3 :=
by sorry

end braking_distance_problems_l2684_268402


namespace det_A_equals_six_l2684_268452

theorem det_A_equals_six (a d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![2*a, 1; -1, d]
  A + B⁻¹ = 0 → Matrix.det A = 6 := by sorry

end det_A_equals_six_l2684_268452


namespace min_value_implications_l2684_268460

/-- Given a > 0, b > 0, and that the function f(x) = |x+a| + |x-b| has a minimum value of 2,
    prove the following inequalities -/
theorem min_value_implications (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) 
    (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
    (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end min_value_implications_l2684_268460


namespace min_sum_dimensions_l2684_268437

theorem min_sum_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → a + b + c ≥ 45 := by
  sorry

end min_sum_dimensions_l2684_268437


namespace cube_diff_divisibility_l2684_268414

theorem cube_diff_divisibility (a b : ℤ) (n : ℕ) 
  (ha : Odd a) (hb : Odd b) : 
  (2^n : ℤ) ∣ (a^3 - b^3) ↔ (2^n : ℤ) ∣ (a - b) := by
  sorry

end cube_diff_divisibility_l2684_268414


namespace reflection_coordinates_l2684_268494

/-- Given points P and M in a 2D plane, this function returns the coordinates of Q, 
    which is the reflection of P about M. -/
def reflection_point (P M : ℝ × ℝ) : ℝ × ℝ :=
  (2 * M.1 - P.1, 2 * M.2 - P.2)

theorem reflection_coordinates :
  let P : ℝ × ℝ := (1, -2)
  let M : ℝ × ℝ := (3, 0)
  reflection_point P M = (5, 2) := by sorry

end reflection_coordinates_l2684_268494


namespace bobby_candy_consumption_l2684_268400

theorem bobby_candy_consumption (morning afternoon evening total : ℕ) : 
  morning = 26 →
  afternoon = 3 * morning →
  evening = afternoon / 2 →
  total = morning + afternoon + evening →
  total = 143 := by
sorry

end bobby_candy_consumption_l2684_268400


namespace cloak_change_theorem_l2684_268454

/-- Represents the price and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the change in silver coins when buying a cloak with gold coins -/
def calculate_silver_change (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1) :
  calculate_silver_change t1 t2 14 = 10 :=
sorry

end cloak_change_theorem_l2684_268454


namespace lemonade_stand_profit_is_35_l2684_268492

/-- Lemonade stand profit calculation -/
def lemonade_stand_profit : ℝ :=
  let small_yield_per_gallon : ℝ := 16
  let medium_yield_per_gallon : ℝ := 10
  let large_yield_per_gallon : ℝ := 6

  let small_cost_per_gallon : ℝ := 2.00
  let medium_cost_per_gallon : ℝ := 3.50
  let large_cost_per_gallon : ℝ := 5.00

  let small_price_per_glass : ℝ := 1.00
  let medium_price_per_glass : ℝ := 1.75
  let large_price_per_glass : ℝ := 2.50

  let gallons_made_each_size : ℝ := 2

  let small_glasses_produced : ℝ := small_yield_per_gallon * gallons_made_each_size
  let medium_glasses_produced : ℝ := medium_yield_per_gallon * gallons_made_each_size
  let large_glasses_produced : ℝ := large_yield_per_gallon * gallons_made_each_size

  let small_glasses_unsold : ℝ := 4
  let medium_glasses_unsold : ℝ := 4
  let large_glasses_unsold : ℝ := 2

  let setup_cost : ℝ := 15.00
  let advertising_cost : ℝ := 10.00

  let small_revenue := (small_glasses_produced - small_glasses_unsold) * small_price_per_glass
  let medium_revenue := (medium_glasses_produced - medium_glasses_unsold) * medium_price_per_glass
  let large_revenue := (large_glasses_produced - large_glasses_unsold) * large_price_per_glass

  let small_cost := gallons_made_each_size * small_cost_per_gallon
  let medium_cost := gallons_made_each_size * medium_cost_per_gallon
  let large_cost := gallons_made_each_size * large_cost_per_gallon

  let total_revenue := small_revenue + medium_revenue + large_revenue
  let total_cost := small_cost + medium_cost + large_cost + setup_cost + advertising_cost

  total_revenue - total_cost

theorem lemonade_stand_profit_is_35 : lemonade_stand_profit = 35 := by
  sorry

end lemonade_stand_profit_is_35_l2684_268492


namespace laptop_arrangement_impossible_l2684_268413

/-- Represents the number of laptops of each type in a row -/
structure LaptopRow :=
  (typeA : ℕ)
  (typeB : ℕ)
  (typeC : ℕ)

/-- The total number of laptops -/
def totalLaptops : ℕ := 44

/-- The number of rows -/
def numRows : ℕ := 5

/-- Checks if a LaptopRow satisfies the ratio condition -/
def satisfiesRatio (row : LaptopRow) : Prop :=
  3 * row.typeA = 2 * row.typeB ∧ 2 * row.typeC = 3 * row.typeB

/-- Checks if a LaptopRow has at least one of each type -/
def hasAllTypes (row : LaptopRow) : Prop :=
  row.typeA > 0 ∧ row.typeB > 0 ∧ row.typeC > 0

/-- Theorem stating the impossibility of the laptop arrangement -/
theorem laptop_arrangement_impossible : 
  ¬ ∃ (row : LaptopRow), 
    (row.typeA + row.typeB + row.typeC) * numRows = totalLaptops ∧
    satisfiesRatio row ∧
    hasAllTypes row :=
by sorry

end laptop_arrangement_impossible_l2684_268413


namespace triangle_theorem_l2684_268488

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem: In a triangle ABC where c sin B = √3 b cos C and a + b = 6,
    the angle C is π/3 and the minimum value of c is 3 -/
theorem triangle_theorem (t : Triangle) 
    (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C) 
    (h2 : t.a + t.b = 6) : 
    t.C = π / 3 ∧ t.c ≥ 3 ∧ ∃ (t' : Triangle), t'.c = 3 := by
  sorry


end triangle_theorem_l2684_268488


namespace prob_n₂_div_2310_eq_l2684_268435

/-- The product of the first 25 primes -/
def n₀ : ℕ := sorry

/-- The Euler totient function -/
def φ : ℕ → ℕ := sorry

/-- The probability of choosing a divisor n of m, proportional to φ(n) -/
def prob_divisor (n m : ℕ) : ℚ := sorry

/-- The probability that a randomly chosen n₂ (which is a random divisor of n₁, 
    which itself is a random divisor of n₀) is divisible by 2310 -/
def prob_n₂_div_2310 : ℚ := sorry

/-- Main theorem: The probability that n₂ ≡ 0 (mod 2310) is 256/5929 -/
theorem prob_n₂_div_2310_eq : prob_n₂_div_2310 = 256 / 5929 := by sorry

end prob_n₂_div_2310_eq_l2684_268435


namespace max_value_on_circle_l2684_268443

theorem max_value_on_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 1 →
  ∃ (max : ℝ), (∀ (x' y' : ℝ), (x' - 1)^2 + y'^2 = 1 → 2*x' + y' ≤ max) ∧ max = Real.sqrt 5 + 2 := by
  sorry

end max_value_on_circle_l2684_268443


namespace divisibility_by_three_divisibility_by_eleven_l2684_268418

-- Part (a)
theorem divisibility_by_three (a : ℤ) (h : ∃ k : ℤ, a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

-- Part (b)
theorem divisibility_by_eleven (a b : ℤ) (h1 : ∃ m : ℤ, 2 + a = 11 * m) (h2 : ∃ n : ℤ, 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end divisibility_by_three_divisibility_by_eleven_l2684_268418


namespace line_does_not_intersect_circle_l2684_268405

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between a point and a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects a circle -/
def lineIntersectsCircle (c : Circle) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), distancePointToLine p l = 0 ∧ (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem line_does_not_intersect_circle
  (c : Circle) (l : Line) (h : distancePointToLine c.center l > c.radius) :
  ¬ lineIntersectsCircle c l :=
sorry

end line_does_not_intersect_circle_l2684_268405


namespace geometric_sequence_sum_l2684_268481

/-- Given a geometric sequence with sum of first m terms Sm, prove S3m = 70 -/
theorem geometric_sequence_sum (m : ℕ) (Sm S2m S3m : ℝ) : 
  Sm = 10 → 
  S2m = 30 → 
  (∃ r : ℝ, r ≠ 0 ∧ S2m - Sm = r * Sm ∧ S3m - S2m = r * (S2m - Sm)) →
  S3m = 70 := by
sorry

end geometric_sequence_sum_l2684_268481


namespace table_formula_proof_l2684_268462

def f (x : ℤ) : ℤ := x^2 - 4*x + 1

theorem table_formula_proof :
  (f 1 = -2) ∧ 
  (f 2 = 0) ∧ 
  (f 3 = 4) ∧ 
  (f 4 = 10) ∧ 
  (f 5 = 18) := by
  sorry

end table_formula_proof_l2684_268462


namespace largest_value_l2684_268431

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  a^2 + b^2 = max (max (max (a^2 + b^2) (2*a*b)) a) (1/2) := by
  sorry

end largest_value_l2684_268431


namespace z_in_third_quadrant_l2684_268403

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition i · z = 1 - 2i
axiom condition : Complex.I * z = 1 - 2 * Complex.I

-- Theorem to prove
theorem z_in_third_quadrant : (z.re < 0) ∧ (z.im < 0) := by
  sorry

end z_in_third_quadrant_l2684_268403


namespace percentage_problem_l2684_268456

theorem percentage_problem (P : ℝ) : 
  (0.5 * 456 = (P / 100) * 120 + 180) → P = 40 := by
  sorry

end percentage_problem_l2684_268456


namespace stevens_collection_group_size_l2684_268421

theorem stevens_collection_group_size :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let num_groups : ℕ := 154
  let total_items : ℕ := skittles + erasers
  (total_items / num_groups : ℕ) = 57 := by
  sorry

end stevens_collection_group_size_l2684_268421


namespace fruit_distribution_l2684_268451

/-- Given 30 pieces of fruit to be distributed equally among 4 friends,
    the smallest number of pieces to remove for equal distribution is 2. -/
theorem fruit_distribution (total_fruit : Nat) (friends : Nat) (pieces_to_remove : Nat) : 
  total_fruit = 30 →
  friends = 4 →
  pieces_to_remove = 2 →
  (total_fruit - pieces_to_remove) % friends = 0 ∧
  ∀ n : Nat, n < pieces_to_remove → (total_fruit - n) % friends ≠ 0 :=
by sorry

end fruit_distribution_l2684_268451


namespace evaluate_expression_l2684_268466

theorem evaluate_expression : 2 + 0 - 2 * 0 = 2 := by sorry

end evaluate_expression_l2684_268466


namespace smallest_n_congruence_l2684_268455

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2015 [MOD 26]) ↔ n = 21 := by
  sorry

end smallest_n_congruence_l2684_268455


namespace right_triangle_area_l2684_268474

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 8 * Real.sqrt 3 →
  angle = 45 * π / 180 →
  let area := (hypotenuse^2 / 4) * Real.sin angle
  area = 48 :=
by
  sorry

end right_triangle_area_l2684_268474


namespace number_difference_l2684_268419

theorem number_difference (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := by
  sorry

end number_difference_l2684_268419


namespace inequality_proof_l2684_268404

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by sorry

end inequality_proof_l2684_268404


namespace max_a_is_eight_l2684_268409

/-- The quadratic polynomial f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a is 8 -/
theorem max_a_is_eight :
  (∃ a : ℝ, condition a) →
  (∀ a : ℝ, condition a → a ≤ 8) ∧
  condition 8 :=
sorry

end max_a_is_eight_l2684_268409


namespace local_minimum_implies_m_equals_two_l2684_268479

/-- The function f(x) = x(x-m)² -/
def f (x m : ℝ) : ℝ := x * (x - m)^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (x m : ℝ) : ℝ := (x - m)^2 + 2*x*(x - m)

theorem local_minimum_implies_m_equals_two :
  ∀ m : ℝ, (∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x m ≥ f 2 m) →
  f_derivative 2 m = 0 →
  m = 2 :=
sorry

end local_minimum_implies_m_equals_two_l2684_268479


namespace inscribed_square_area_l2684_268485

/-- The parabola function y = x^2 - 6x + 8 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A square inscribed in the region bounded by the parabola and the x-axis -/
structure InscribedSquare where
  side : ℝ
  center_x : ℝ
  lower_left : ℝ × ℝ
  upper_right : ℝ × ℝ
  on_x_axis : lower_left.2 = 0 ∧ upper_right.2 = side
  on_parabola : parabola upper_right.1 = upper_right.2

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∀ (s : InscribedSquare), s.side^2 = 24 - 8 * Real.sqrt 5 := by
  sorry

end inscribed_square_area_l2684_268485


namespace range_of_m_l2684_268441

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x ∈ Set.Icc 0 1, x^2 - m*x - 2 = 0

def q (m : ℝ) : Prop := ∀ x ≥ 1, 
  (∀ y ≥ x, (y^2 - 2*m*y + 1/2) / (x^2 - 2*m*x + 1/2) ≥ 1) ∧ 
  (x^2 - 2*m*x + 1/2 > 0)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (¬(p m) ∧ (p m ∨ q m)) → (m > -1 ∧ m < 3/4) :=
sorry

end range_of_m_l2684_268441


namespace negation_of_existence_inequality_l2684_268433

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by
  sorry

end negation_of_existence_inequality_l2684_268433


namespace students_taking_both_french_and_german_l2684_268450

theorem students_taking_both_french_and_german 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 78) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 24) :
  french + german - (total - neither) = 9 :=
by sorry

end students_taking_both_french_and_german_l2684_268450


namespace vector_parallel_sum_l2684_268407

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is 3. -/
theorem vector_parallel_sum (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) :
  a.1 = 1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = 3 := by
  sorry

end vector_parallel_sum_l2684_268407


namespace unique_tuple_existence_l2684_268461

theorem unique_tuple_existence 
  (p q : ℝ) 
  (h_pos_p : 0 < p) 
  (h_pos_q : 0 < q) 
  (h_sum : p + q = 1) 
  (y : Fin 2017 → ℝ) : 
  ∃! x : Fin 2018 → ℝ, 
    (∀ i : Fin 2017, p * max (x i) (x (i + 1)) + q * min (x i) (x (i + 1)) = y i) ∧ 
    x 0 = x 2017 := by
  sorry

end unique_tuple_existence_l2684_268461


namespace ball_probability_theorem_l2684_268447

/-- Probability of drawing a white ball from the nth box -/
def P (n : ℕ) : ℚ :=
  1/2 * (1/3)^n + 1/2

theorem ball_probability_theorem (n : ℕ) :
  n ≥ 2 →
  (P 2 = 5/9) ∧
  (∀ k : ℕ, k ≥ 2 → P k = 1/2 * (1/3)^k + 1/2) :=
by sorry

end ball_probability_theorem_l2684_268447


namespace nearest_integer_to_sum_l2684_268467

def fraction1 : ℚ := 2007 / 2999
def fraction2 : ℚ := 8001 / 5998
def fraction3 : ℚ := 2001 / 3999

def sum : ℚ := fraction1 + fraction2 + fraction3

theorem nearest_integer_to_sum :
  round sum = 3 := by sorry

end nearest_integer_to_sum_l2684_268467


namespace inheritance_division_l2684_268495

theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 527500 →
  num_people = 5 →
  amount_per_person = total_amount / num_people →
  amount_per_person = 105500 := by
  sorry

end inheritance_division_l2684_268495


namespace find_B_l2684_268432

theorem find_B (A B : ℚ) : (1 / 4 : ℚ) * (1 / 8 : ℚ) = 1 / (4 * A) ∧ 1 / (4 * A) = 1 / B → B = 32 := by
  sorry

end find_B_l2684_268432


namespace at_least_one_inequality_holds_l2684_268434

-- Define a triangle in 2D space
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Function to check if a point is inside a triangle
def isInside (t : Triangle) (p : Point) : Prop := sorry

-- Function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Theorem statement
theorem at_least_one_inequality_holds (t : Triangle) (M N : Point) :
  isInside t M →
  isInside t N →
  M ≠ N →
  (distance t.A N > distance t.A M) ∨
  (distance t.B N > distance t.B M) ∨
  (distance t.C N > distance t.C M) :=
sorry

end at_least_one_inequality_holds_l2684_268434


namespace fraction_difference_equals_negative_one_l2684_268429

theorem fraction_difference_equals_negative_one 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x * y) : 
  1 / x - 1 / y = -1 := by
sorry

end fraction_difference_equals_negative_one_l2684_268429


namespace max_value_of_a_plus_2b_for_tangent_line_l2684_268477

/-- Given a line ax + by = 1 (where a > 0, b > 0) tangent to the circle x² + y² = 1,
    the maximum value of a + 2b is √5. -/
theorem max_value_of_a_plus_2b_for_tangent_line :
  ∀ a b : ℝ,
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 = 1) →
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  (∀ c : ℝ, c ≥ a + 2*b → c ≥ Real.sqrt 5) ∧
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    (∀ x y : ℝ, a' * x + b' * y = 1 → x^2 + y^2 = 1) ∧
    (∃ x y : ℝ, a' * x + b' * y = 1 ∧ x^2 + y^2 = 1) ∧
    a' + 2*b' = Real.sqrt 5) :=
by sorry

end max_value_of_a_plus_2b_for_tangent_line_l2684_268477


namespace paris_hair_theorem_l2684_268478

theorem paris_hair_theorem (population : ℕ) (max_hair_count : ℕ) 
  (h1 : population > 2000000) 
  (h2 : max_hair_count = 150000) : 
  ∃ (hair_count : ℕ), hair_count ≤ max_hair_count ∧ 
  (∃ (group : Finset (Fin population)), group.card ≥ 14 ∧ 
  ∀ i ∈ group, hair_count = (i : ℕ)) :=
sorry

end paris_hair_theorem_l2684_268478


namespace evaluate_expression_l2684_268401

theorem evaluate_expression (a x : ℝ) (h : x = a + 9) : x - a + 5 = 14 := by
  sorry

end evaluate_expression_l2684_268401


namespace factor_calculation_l2684_268498

theorem factor_calculation : ∃ f : ℚ, (2 * 7 + 9) * f = 69 ∧ f = 3 := by
  sorry

end factor_calculation_l2684_268498


namespace absolute_value_inequality_l2684_268426

theorem absolute_value_inequality (y : ℝ) : 
  |((7 - y) / 4)| ≤ 3 ↔ -5 ≤ y ∧ y ≤ 19 := by sorry

end absolute_value_inequality_l2684_268426


namespace chris_previous_savings_l2684_268406

/-- Represents the amount of money Chris received as birthday gifts in different currencies -/
structure BirthdayGifts where
  usd : ℝ
  eur : ℝ
  cad : ℝ
  gbp : ℝ

/-- Represents the conversion rates from different currencies to USD -/
structure ConversionRates where
  eur_to_usd : ℝ
  cad_to_usd : ℝ
  gbp_to_usd : ℝ

/-- Calculates Chris's savings before his birthday -/
def calculate_previous_savings (gifts : BirthdayGifts) (rates : ConversionRates) (total_after : ℝ) : ℝ :=
  total_after - (gifts.usd + 
                 gifts.eur * rates.eur_to_usd + 
                 gifts.cad * rates.cad_to_usd + 
                 gifts.gbp * rates.gbp_to_usd)

/-- Theorem stating that Chris's savings before his birthday were 128.80 USD -/
theorem chris_previous_savings 
  (gifts : BirthdayGifts) 
  (rates : ConversionRates) 
  (total_after : ℝ) : 
  gifts.usd = 25 ∧ 
  gifts.eur = 20 ∧ 
  gifts.cad = 75 ∧ 
  gifts.gbp = 30 ∧
  rates.eur_to_usd = 1 / 0.85 ∧
  rates.cad_to_usd = 1 / 1.25 ∧
  rates.gbp_to_usd = 1 / 0.72 ∧
  total_after = 279 →
  calculate_previous_savings gifts rates total_after = 128.80 := by
    sorry

end chris_previous_savings_l2684_268406


namespace age_double_time_l2684_268453

/-- Given two brothers with current ages 15 and 5, this theorem proves that
    it will take 5 years for the older brother's age to be twice the younger brother's age. -/
theorem age_double_time (older_age younger_age : ℕ) (h1 : older_age = 15) (h2 : younger_age = 5) :
  ∃ (years : ℕ), years = 5 ∧ older_age + years = 2 * (younger_age + years) :=
sorry

end age_double_time_l2684_268453


namespace comic_books_liked_by_females_l2684_268473

/-- Given a comic store with the following properties:
  - There are 300 comic books in total
  - Males like 120 comic books
  - 30% of comic books are disliked by both males and females
  Prove that the percentage of comic books liked by females is 30% -/
theorem comic_books_liked_by_females 
  (total_comics : ℕ) 
  (liked_by_males : ℕ) 
  (disliked_percentage : ℚ) :
  total_comics = 300 →
  liked_by_males = 120 →
  disliked_percentage = 30 / 100 →
  (total_comics - (disliked_percentage * total_comics).num - liked_by_males) / total_comics = 30 / 100 := by
sorry

end comic_books_liked_by_females_l2684_268473


namespace trail_mix_fruit_percentage_l2684_268496

-- Define the trail mix compositions
def sue_mix : ℝ := 5
def sue_nuts_percent : ℝ := 0.3
def sue_fruit_percent : ℝ := 0.7

def jane_mix : ℝ := 7
def jane_nuts_percent : ℝ := 0.6

def tom_mix : ℝ := 9
def tom_nuts_percent : ℝ := 0.4
def tom_fruit_percent : ℝ := 0.5

-- Define the combined mixture properties
def combined_nuts_percent : ℝ := 0.45

-- Theorem to prove
theorem trail_mix_fruit_percentage :
  let total_nuts := sue_mix * sue_nuts_percent + jane_mix * jane_nuts_percent + tom_mix * tom_nuts_percent
  let total_weight := total_nuts / combined_nuts_percent
  let total_fruit := sue_mix * sue_fruit_percent + tom_mix * tom_fruit_percent
  let fruit_percentage := total_fruit / total_weight * 100
  abs (fruit_percentage - 38.71) < 0.01 := by
sorry

end trail_mix_fruit_percentage_l2684_268496


namespace chelsea_cupcake_time_l2684_268440

/-- The time it takes to make cupcakes given the number of batches and time per batch -/
def cupcake_time (num_batches : ℕ) (bake_time : ℕ) (ice_time : ℕ) : ℕ :=
  num_batches * (bake_time + ice_time)

/-- Theorem: Chelsea's cupcake-making time -/
theorem chelsea_cupcake_time :
  cupcake_time 4 20 30 = 200 := by
  sorry

end chelsea_cupcake_time_l2684_268440


namespace min_distance_M_to_F₂_l2684_268491

-- Define the rectangle and its properties
def Rectangle (a b : ℝ) := a > b ∧ a > 0 ∧ b > 0

-- Define the points on the sides of the rectangle
def Points (n : ℕ) (a b : ℝ) := n ≥ 5

-- Define the ellipse F₁
def F₁ (x y a b : ℝ) := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola F₂
def F₂ (x y a b : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point M on F₁
def M (b : ℝ) := (0, b)

-- Theorem statement
theorem min_distance_M_to_F₂ (n : ℕ) (a b : ℝ) :
  Rectangle a b →
  Points n a b →
  ∀ (x y : ℝ), F₂ x y a b →
  Real.sqrt ((x - 0)^2 + (y - b)^2) ≥ a * Real.sqrt ((a^2 + 2*b^2) / (a^2 + b^2)) :=
sorry

end min_distance_M_to_F₂_l2684_268491


namespace geometric_sequence_sum_l2684_268482

theorem geometric_sequence_sum (a b c : ℝ) : 
  (1 < a ∧ a < b ∧ b < c ∧ c < 16) →
  (∃ q : ℝ, q ≠ 0 ∧ a = 1 * q ∧ b = a * q ∧ c = b * q ∧ 16 = c * q) →
  (a + c = 10 ∨ a + c = -10) :=
sorry

end geometric_sequence_sum_l2684_268482


namespace book_difference_l2684_268487

/- Define the number of books for each category -/
def total_books : ℕ := 220
def hardcover_nonfiction : ℕ := 40

/- Define the properties of the book categories -/
def book_categories (paperback_fiction paperback_nonfiction : ℕ) : Prop :=
  paperback_fiction + paperback_nonfiction + hardcover_nonfiction = total_books ∧
  paperback_nonfiction > hardcover_nonfiction ∧
  paperback_fiction = 2 * paperback_nonfiction

/- Theorem statement -/
theorem book_difference :
  ∃ (paperback_fiction paperback_nonfiction : ℕ),
    book_categories paperback_fiction paperback_nonfiction ∧
    paperback_nonfiction - hardcover_nonfiction = 20 :=
by sorry

end book_difference_l2684_268487


namespace chinese_money_plant_price_is_25_l2684_268459

/-- The price of each potted Chinese money plant -/
def chinese_money_plant_price : ℕ := sorry

/-- The number of orchids sold -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def chinese_money_plants_sold : ℕ := 15

/-- The payment for each worker -/
def worker_payment : ℕ := 40

/-- The number of workers -/
def number_of_workers : ℕ := 2

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after expenses -/
def amount_left : ℕ := 1145

theorem chinese_money_plant_price_is_25 :
  chinese_money_plant_price = 25 ∧
  orchids_sold * orchid_price + chinese_money_plants_sold * chinese_money_plant_price =
  amount_left + number_of_workers * worker_payment + new_pots_cost :=
sorry

end chinese_money_plant_price_is_25_l2684_268459


namespace students_math_or_history_not_both_l2684_268472

theorem students_math_or_history_not_both 
  (both : ℕ) 
  (math_total : ℕ) 
  (history_only : ℕ) 
  (h1 : both = 15) 
  (h2 : math_total = 30) 
  (h3 : history_only = 12) : 
  (math_total - both) + history_only = 27 := by
  sorry

end students_math_or_history_not_both_l2684_268472


namespace arcsin_neg_half_eq_neg_pi_sixth_l2684_268416

theorem arcsin_neg_half_eq_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end arcsin_neg_half_eq_neg_pi_sixth_l2684_268416


namespace function_equation_implies_identity_l2684_268493

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_equation_implies_identity_l2684_268493


namespace black_socks_bought_l2684_268408

/-- Represents the number of pairs of socks of each color -/
structure SockCount where
  blue : ℕ
  black : ℕ
  white : ℕ

/-- The initial sock count before buying more black socks -/
def initialSocks : SockCount :=
  { blue := 6, black := 18, white := 12 }

/-- The proportion of black socks after buying more -/
def blackProportion : ℚ := 3 / 5

/-- Calculates the total number of sock pairs -/
def totalSocks (s : SockCount) : ℕ :=
  s.blue + s.black + s.white

/-- Theorem stating the number of black sock pairs Dmitry bought -/
theorem black_socks_bought (x : ℕ) : 
  (initialSocks.black + x : ℚ) / (totalSocks initialSocks + x : ℚ) = blackProportion →
  x = 9 := by
  sorry

end black_socks_bought_l2684_268408


namespace food_drive_mark_cans_l2684_268463

/-- Represents the number of cans brought by each person -/
structure Cans where
  mark : ℕ
  jaydon : ℕ
  sophie : ℕ
  rachel : ℕ

/-- Represents the conditions of the food drive -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 4 * c.jaydon ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.mark + c.jaydon + c.sophie = 225 ∧
  4 * c.jaydon = 3 * c.mark ∧
  3 * c.sophie = 2 * c.mark

theorem food_drive_mark_cans :
  ∀ c : Cans, FoodDrive c → c.mark = 100 :=
by
  sorry

end food_drive_mark_cans_l2684_268463


namespace intersecting_chords_theorem_l2684_268415

/-- The number of points marked on the circle -/
def n : ℕ := 20

/-- The number of sets of three intersecting chords with endpoints chosen from n points on a circle -/
def intersecting_chords_count (n : ℕ) : ℕ :=
  Nat.choose n 3 + 
  8 * Nat.choose n 4 + 
  5 * Nat.choose n 5 + 
  Nat.choose n 6

/-- Theorem stating that the number of sets of three intersecting chords 
    with endpoints chosen from 20 points on a circle is 156180 -/
theorem intersecting_chords_theorem : 
  intersecting_chords_count n = 156180 := by sorry

end intersecting_chords_theorem_l2684_268415


namespace log_8_y_value_l2684_268457

theorem log_8_y_value (y : ℝ) (h : Real.log y / Real.log 8 = 3.25) : y = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end log_8_y_value_l2684_268457


namespace tricia_age_l2684_268438

/-- Represents the ages of individuals in the problem -/
structure Ages where
  tricia : ℕ
  amilia : ℕ
  yorick : ℕ
  eugene : ℕ
  khloe : ℕ
  rupert : ℕ
  vincent : ℕ
  selena : ℕ
  cora : ℕ
  brody : ℕ

/-- Defines the relationships between ages as given in the problem -/
def valid_ages (a : Ages) : Prop :=
  a.tricia = a.amilia / 3 ∧
  a.amilia = a.yorick / 4 ∧
  a.yorick = 2 * a.eugene ∧
  a.khloe = a.eugene / 3 ∧
  a.rupert = a.khloe + 10 ∧
  a.rupert = a.vincent - 2 ∧
  a.vincent = 22 ∧
  a.yorick = a.selena + 5 ∧
  a.selena = a.amilia + 3 ∧
  a.cora = (a.vincent + a.amilia) / 2 ∧
  a.brody = a.tricia + a.vincent

/-- Theorem stating that if the ages satisfy the given relationships, then Tricia's age is 5 -/
theorem tricia_age (a : Ages) (h : valid_ages a) : a.tricia = 5 := by
  sorry


end tricia_age_l2684_268438


namespace initial_volume_proof_l2684_268458

/-- Proves that the initial volume of a solution is 40 liters given the conditions of the problem -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 4.5 = 0.13 * (V + 10)) → V = 40 := by
  sorry

end initial_volume_proof_l2684_268458


namespace seeds_sown_l2684_268486

/-- Given a farmer who started with 8.75 buckets of seeds and ended with 6 buckets,
    prove that the number of buckets sown is 2.75. -/
theorem seeds_sown (initial : ℝ) (remaining : ℝ) (h1 : initial = 8.75) (h2 : remaining = 6) :
  initial - remaining = 2.75 := by
  sorry

end seeds_sown_l2684_268486


namespace rectangular_plot_area_l2684_268420

theorem rectangular_plot_area 
  (breadth : ℝ) 
  (length : ℝ) 
  (h1 : breadth = 12)
  (h2 : length = 3 * breadth) : 
  breadth * length = 432 := by
sorry

end rectangular_plot_area_l2684_268420


namespace roots_sum_value_l2684_268471

-- Define the quadratic equation
def quadratic (x : ℝ) : Prop := x^2 - x - 1 = 0

-- Define the roots a and b
variable (a b : ℝ)

-- State the theorem
theorem roots_sum_value (ha : quadratic a) (hb : quadratic b) (hab : a ≠ b) :
  3 * a^2 + 4 * b + 2 / a^2 = 11 := by sorry

end roots_sum_value_l2684_268471


namespace connie_needs_4999_l2684_268476

/-- Calculates the additional amount Connie needs to buy the items --/
def additional_amount_needed (saved : ℚ) (watch_price : ℚ) (strap_original : ℚ) (strap_discount : ℚ) 
  (case_price : ℚ) (protector_price_eur : ℚ) (tax_rate : ℚ) (exchange_rate : ℚ) : ℚ :=
  let strap_price := strap_original * (1 - strap_discount)
  let protector_price_usd := protector_price_eur * exchange_rate
  let subtotal := watch_price + strap_price + case_price + protector_price_usd
  let total_with_tax := subtotal * (1 + tax_rate)
  (total_with_tax - saved).ceil / 100

/-- The theorem stating the additional amount Connie needs --/
theorem connie_needs_4999 : 
  additional_amount_needed 39 55 20 0.25 10 2 0.08 1.2 = 4999 / 100 := by
  sorry

end connie_needs_4999_l2684_268476


namespace alberto_engine_spending_l2684_268444

/-- Represents the spending on car maintenance -/
structure CarSpending where
  oil : ℕ
  tires : ℕ
  detailing : ℕ

/-- Calculates the total spending for a CarSpending instance -/
def total_spending (s : CarSpending) : ℕ := s.oil + s.tires + s.detailing

/-- Represents Samara's spending -/
def samara_spending : CarSpending := { oil := 25, tires := 467, detailing := 79 }

/-- The amount Alberto spent more than Samara -/
def alberto_extra_spending : ℕ := 1886

/-- Theorem: Alberto's spending on the new engine is $2457 -/
theorem alberto_engine_spending :
  total_spending samara_spending + alberto_extra_spending = 2457 := by
  sorry

end alberto_engine_spending_l2684_268444


namespace equation_solution_l2684_268445

theorem equation_solution : 
  ∃! x : ℝ, x + 36 / (x - 3) = -9 ∧ x = -3 := by
  sorry

end equation_solution_l2684_268445


namespace at_least_one_greater_than_one_l2684_268480

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by
  sorry

end at_least_one_greater_than_one_l2684_268480


namespace water_depth_is_12_feet_l2684_268448

/-- The height of Ron in feet -/
def ron_height : ℝ := 14

/-- The difference in height between Ron and Dean in feet -/
def height_difference : ℝ := 8

/-- The height of Dean in feet -/
def dean_height : ℝ := ron_height - height_difference

/-- The depth of the water as a multiple of Dean's height -/
def water_depth_factor : ℝ := 2

/-- The depth of the water in feet -/
def water_depth : ℝ := water_depth_factor * dean_height

theorem water_depth_is_12_feet : water_depth = 12 := by
  sorry

end water_depth_is_12_feet_l2684_268448


namespace aquarium_count_l2684_268430

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) 
  (h3 : animals_per_aquarium > 0) : 
  total_animals / animals_per_aquarium = 20 := by
  sorry

end aquarium_count_l2684_268430


namespace range_of_a_when_p_is_false_l2684_268468

theorem range_of_a_when_p_is_false :
  (¬∃ (x : ℝ), x > 0 ∧ x + 1/x < a) ↔ a ≤ 2 :=
by sorry

end range_of_a_when_p_is_false_l2684_268468


namespace point_outside_ellipse_l2684_268442

theorem point_outside_ellipse (m n : ℝ) 
  (h_intersect : ∃ x y : ℝ, m * x + n * y = 4 ∧ x^2 + y^2 = 4) :
  m^2 / 4 + n^2 / 3 > 1 := by
  sorry

end point_outside_ellipse_l2684_268442


namespace fraction_meaningful_l2684_268469

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (x + 2)) ↔ x ≠ -2 := by
sorry

end fraction_meaningful_l2684_268469


namespace sum_equals_target_l2684_268465

theorem sum_equals_target : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end sum_equals_target_l2684_268465


namespace class_mean_calculation_l2684_268424

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = group1_students + group2_students →
  group1_students = 40 →
  group2_students = 10 →
  group1_mean = 68 / 100 →
  group2_mean = 74 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 692 / 1000 := by
  sorry

end class_mean_calculation_l2684_268424
