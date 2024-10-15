import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l2367_236717

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_line_at_one_two :
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*x + 1
  (∀ x, HasDerivAt f (f' x) x) →
  f 1 = 2 →
  (λ x ↦ 2*x) = λ x ↦ f 1 + f' 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l2367_236717


namespace NUMINAMATH_CALUDE_ages_solution_l2367_236766

/-- Represents the ages of four persons --/
structure Ages where
  a : ℕ  -- oldest
  b : ℕ  -- second oldest
  c : ℕ  -- third oldest
  d : ℕ  -- youngest

/-- The conditions given in the problem --/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.a = ages.d + 16 ∧
  ages.b = ages.d + 8 ∧
  ages.c = ages.d + 4 ∧
  ages.a - 6 = 3 * (ages.d - 6) ∧
  ages.a - 6 = 2 * (ages.b - 6) ∧
  ages.a - 6 = (ages.c - 6) + 4

/-- The theorem to be proved --/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.a = 30 ∧ ages.b = 22 ∧ ages.c = 18 ∧ ages.d = 14 :=
  sorry

end NUMINAMATH_CALUDE_ages_solution_l2367_236766


namespace NUMINAMATH_CALUDE_alligator_count_theorem_l2367_236764

/-- The total number of alligators seen by Samara and her friends -/
def total_alligators (samara_count : ℕ) (friend_count : ℕ) (friend_average : ℕ) : ℕ :=
  samara_count + friend_count * friend_average

/-- Theorem stating the total number of alligators seen -/
theorem alligator_count_theorem :
  total_alligators 20 3 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_alligator_count_theorem_l2367_236764


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2367_236760

def f (x : ℝ) : ℝ := x * (1 + x)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2367_236760


namespace NUMINAMATH_CALUDE_frustum_smaller_cone_altitude_l2367_236711

-- Define the frustum
structure Frustum where
  altitude : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem frustum_smaller_cone_altitude (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi)
  (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (smallerConeAltitude : ℝ), smallerConeAltitude = f.altitude := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_cone_altitude_l2367_236711


namespace NUMINAMATH_CALUDE_male_female_ratio_l2367_236721

theorem male_female_ratio (M F : ℝ) (h1 : M > 0) (h2 : F > 0) : 
  (1/4 * M + 3/4 * F) / (M + F) = 198 / 360 → M / F = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_male_female_ratio_l2367_236721


namespace NUMINAMATH_CALUDE_remaining_lives_total_l2367_236759

def game_scenario (initial_players : ℕ) (first_quitters : ℕ) (second_quitters : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players - first_quitters - second_quitters) * lives_per_player

theorem remaining_lives_total :
  game_scenario 15 5 4 7 = 42 := by
  sorry

end NUMINAMATH_CALUDE_remaining_lives_total_l2367_236759


namespace NUMINAMATH_CALUDE_hospital_staff_count_l2367_236783

theorem hospital_staff_count (total : ℕ) (d_ratio n_ratio : ℕ) (h1 : total = 456) (h2 : d_ratio = 8) (h3 : n_ratio = 11) : 
  ∃ (doctors nurses : ℕ), 
    doctors + nurses = total ∧ 
    doctors * n_ratio = nurses * d_ratio ∧ 
    nurses = 264 := by
  sorry

end NUMINAMATH_CALUDE_hospital_staff_count_l2367_236783


namespace NUMINAMATH_CALUDE_first_bear_price_correct_l2367_236792

/-- The price of the first bear in a sequence of bear prices -/
def first_bear_price : ℚ := 57 / 2

/-- The number of bears purchased -/
def num_bears : ℕ := 101

/-- The discount applied to each bear after the first -/
def discount : ℚ := 1 / 2

/-- The total cost of all bears -/
def total_cost : ℚ := 354

/-- Theorem stating that the first bear price is correct given the conditions -/
theorem first_bear_price_correct :
  (num_bears : ℚ) / 2 * (2 * first_bear_price - (num_bears - 1) * discount) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_first_bear_price_correct_l2367_236792


namespace NUMINAMATH_CALUDE_seven_point_circle_triangle_count_l2367_236736

/-- A circle with points and chords -/
structure CircleWithChords where
  numPoints : ℕ
  noTripleIntersection : Bool

/-- Count of triangles formed by chord intersections -/
def triangleCount (c : CircleWithChords) : ℕ := sorry

/-- Theorem: For 7 points on a circle with no triple intersections, 
    the number of triangles formed by chord intersections is 7 -/
theorem seven_point_circle_triangle_count 
  (c : CircleWithChords) 
  (h1 : c.numPoints = 7) 
  (h2 : c.noTripleIntersection = true) : 
  triangleCount c = 7 := by sorry

end NUMINAMATH_CALUDE_seven_point_circle_triangle_count_l2367_236736


namespace NUMINAMATH_CALUDE_line_relationship_l2367_236723

-- Define the concept of lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  ¬ parallel c b := by
  sorry

end NUMINAMATH_CALUDE_line_relationship_l2367_236723


namespace NUMINAMATH_CALUDE_club_women_count_l2367_236791

/-- Proves the number of women in a club given certain conditions -/
theorem club_women_count (total : ℕ) (attendees : ℕ) (men : ℕ) (women : ℕ) :
  total = 30 →
  attendees = 18 →
  men + women = total →
  men + (women / 3) = attendees →
  women = 18 := by
  sorry

end NUMINAMATH_CALUDE_club_women_count_l2367_236791


namespace NUMINAMATH_CALUDE_lori_earnings_l2367_236720

/-- Calculates the total earnings for a car rental company given the number of cars,
    rental rates, and rental duration. -/
def total_earnings (red_cars white_cars : ℕ) (red_rate white_rate : ℚ) (hours : ℕ) : ℚ :=
  (red_cars * red_rate + white_cars * white_rate) * (hours * 60)

/-- Proves that given the specific conditions of Lori's car rental business,
    the total earnings are $2340. -/
theorem lori_earnings :
  total_earnings 3 2 3 2 3 = 2340 := by
  sorry

#eval total_earnings 3 2 3 2 3

end NUMINAMATH_CALUDE_lori_earnings_l2367_236720


namespace NUMINAMATH_CALUDE_part_one_part_two_l2367_236793

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def B : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def C (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem part_one (b c : ℝ) : 
  (A ∩ B = {x | b*x^2 + 10*x + c ≥ 0}) → (b = -2 ∧ c = -12) := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : 
  (C a ⊆ B ∪ (U \ A)) → (a ≥ -11/6 ∧ a ≤ 9/4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2367_236793


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l2367_236794

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = 2, prove that its asymptotes are y = ± √3 * x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := fun (c : ℝ) ↦ c / a = e
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  ∃ c, eccentricity c ∧ (∀ x y, asymptotes x y ↔ (hyperbola x y ∧ x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l2367_236794


namespace NUMINAMATH_CALUDE_linear_function_proof_l2367_236702

def f (x : ℝ) : ℝ := -x + 1

theorem linear_function_proof :
  (∀ x y t : ℝ, f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y) ∧
  f 0 = 1 ∧
  ∀ x1 x2 : ℝ, x2 > x1 → f x2 < f x1 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l2367_236702


namespace NUMINAMATH_CALUDE_computer_table_price_l2367_236748

/-- The selling price of an item given its cost price and markup percentage -/
def selling_price (cost : ℕ) (markup_percent : ℕ) : ℕ :=
  cost + cost * markup_percent / 100

/-- Theorem stating that for a computer table with cost price 3000 and 20% markup, 
    the selling price is 3600 -/
theorem computer_table_price : selling_price 3000 20 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_price_l2367_236748


namespace NUMINAMATH_CALUDE_cereal_eating_time_l2367_236795

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eat_time (swift_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (swift_rate + slow_rate)

/-- Theorem: Mr. Swift and Mr. Slow will take 45 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  let swift_rate : ℚ := 1 / 15  -- Mr. Swift's eating rate in pounds per minute
  let slow_rate : ℚ := 1 / 45   -- Mr. Slow's eating rate in pounds per minute
  let total_amount : ℚ := 4     -- Total amount of cereal in pounds
  eat_time swift_rate slow_rate total_amount = 45 := by
sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l2367_236795


namespace NUMINAMATH_CALUDE_min_sum_a_c_l2367_236708

theorem min_sum_a_c (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a - c) * (b - d) = -4)
  (h2 : (a + c) / 2 ≥ (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)) :
  ∀ ε > 0, a + c ≥ 4 * Real.sqrt 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_c_l2367_236708


namespace NUMINAMATH_CALUDE_not_perfect_cube_l2367_236751

theorem not_perfect_cube (t : ℤ) : ¬ ∃ (k : ℤ), 7 * t + 3 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l2367_236751


namespace NUMINAMATH_CALUDE_absolute_value_of_expression_l2367_236750

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_value_of_expression : 
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_expression_l2367_236750


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2367_236757

theorem simplify_polynomial (x : ℝ) : 
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 
  4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2367_236757


namespace NUMINAMATH_CALUDE_trail_mix_almonds_l2367_236713

theorem trail_mix_almonds (walnuts : ℝ) (total_nuts : ℝ) (almonds : ℝ) : 
  walnuts = 0.25 → total_nuts = 0.5 → almonds = total_nuts - walnuts → almonds = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_almonds_l2367_236713


namespace NUMINAMATH_CALUDE_P_is_projection_l2367_236779

def P : Matrix (Fin 2) (Fin 2) ℚ := !![20/49, 20/49; 29/49, 29/49]

theorem P_is_projection : P * P = P := by sorry

end NUMINAMATH_CALUDE_P_is_projection_l2367_236779


namespace NUMINAMATH_CALUDE_train_speed_l2367_236763

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (length : Real) (time : Real) (speed_kmh : Real) : 
  length = 140 ∧ time = 7 → speed_kmh = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2367_236763


namespace NUMINAMATH_CALUDE_ratio_problem_l2367_236796

theorem ratio_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y = x * (1 + 14.285714285714285 / 100)) : 
  x / y = 7 / 8 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2367_236796


namespace NUMINAMATH_CALUDE_ellipse_equation_l2367_236754

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  e : ℝ  -- eccentricity

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse C with specific properties, prove its equation -/
theorem ellipse_equation (C : Ellipse) (F₁ F₂ A B : Point) :
  C.center = (0, 0) →  -- center at origin
  F₁.y = 0 →  -- foci on x-axis
  F₂.y = 0 →
  C.e = Real.sqrt 2 / 2 →  -- eccentricity is √2/2
  (A.x - F₁.x)^2 + (A.y - F₁.y)^2 = (B.x - F₁.x)^2 + (B.y - F₁.y)^2 →  -- A and B on line through F₁
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) +
    Real.sqrt ((B.x - F₂.x)^2 + (B.y - F₂.y)^2) +
    Real.sqrt ((A.x - F₂.x)^2 + (A.y - F₂.y)^2) = 16 →  -- perimeter of ABF₂ is 16
  ∀ (x y : ℝ), x^2 / 64 + y^2 / 32 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 / C.a)^2 + (p.2 / C.b)^2 = 1} :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2367_236754


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2367_236774

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 6 ∧ 
  (x + Real.sqrt y) * (x - Real.sqrt y) = 9 → 
  x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2367_236774


namespace NUMINAMATH_CALUDE_fabric_needed_for_coats_l2367_236769

/-- 
Given that:
- 16 meters of fabric can make 4 men's coats and 2 children's coats
- 18 meters of fabric can make 2 men's coats and 6 children's coats

Prove that the fabric needed for one men's coat is 3 meters and for one children's coat is 2 meters.
-/
theorem fabric_needed_for_coats : 
  ∀ (m c : ℝ), 
  (4 * m + 2 * c = 16) → 
  (2 * m + 6 * c = 18) → 
  (m = 3 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_fabric_needed_for_coats_l2367_236769


namespace NUMINAMATH_CALUDE_min_m_value_l2367_236798

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(2+x) = f(2-x) for all x
    and f is monotonically increasing on [m, +∞), then the minimum value of m is 2. -/
theorem min_m_value (f : ℝ → ℝ) (a : ℝ) (m : ℝ) :
  (∀ x, f x = 2^(|x - a|)) →
  (∀ x, f (2 + x) = f (2 - x)) →
  (∀ x y, m ≤ x → x < y → f x ≤ f y) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_m_value_l2367_236798


namespace NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l2367_236782

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 27 = 13 % 27 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 27 = 13 % 27 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_congruence_l2367_236782


namespace NUMINAMATH_CALUDE_intersection_has_one_element_l2367_236722

theorem intersection_has_one_element (a : ℝ) : 
  let A := {x : ℝ | 2^(1+x) + 2^(1-x) = a}
  let B := {y : ℝ | ∃ θ : ℝ, y = Real.sin θ}
  (∃! x : ℝ, x ∈ A ∩ B) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_has_one_element_l2367_236722


namespace NUMINAMATH_CALUDE_alice_cannot_arrive_before_bob_l2367_236709

/-- Proves that Alice cannot arrive before Bob given the conditions --/
theorem alice_cannot_arrive_before_bob :
  let distance : ℝ := 120  -- Distance between cities in miles
  let bob_speed : ℝ := 40  -- Bob's speed in miles per hour
  let alice_speed : ℝ := 48  -- Alice's speed in miles per hour
  let bob_head_start : ℝ := 0.5  -- Bob's head start in hours

  let bob_initial_distance : ℝ := bob_speed * bob_head_start
  let bob_remaining_distance : ℝ := distance - bob_initial_distance
  let bob_remaining_time : ℝ := bob_remaining_distance / bob_speed
  let alice_total_time : ℝ := distance / alice_speed

  alice_total_time ≥ bob_remaining_time :=
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_alice_cannot_arrive_before_bob_l2367_236709


namespace NUMINAMATH_CALUDE_favorite_pet_dog_l2367_236727

theorem favorite_pet_dog (total : ℕ) (cat fish bird other : ℕ) 
  (h_total : total = 90)
  (h_cat : cat = 25)
  (h_fish : fish = 10)
  (h_bird : bird = 15)
  (h_other : other = 5) :
  total - (cat + fish + bird + other) = 35 := by
  sorry

end NUMINAMATH_CALUDE_favorite_pet_dog_l2367_236727


namespace NUMINAMATH_CALUDE_min_sum_abcd_l2367_236737

theorem min_sum_abcd (a b c d : ℕ) 
  (h1 : a + b = 2)
  (h2 : a + c = 3)
  (h3 : a + d = 4)
  (h4 : b + c = 5)
  (h5 : b + d = 6)
  (h6 : c + d = 7) :
  a + b + c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_abcd_l2367_236737


namespace NUMINAMATH_CALUDE_pool_perimeter_is_20_l2367_236746

/-- Represents the dimensions and constraints of a rectangular pool in a garden --/
structure PoolInGarden where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ
  pool_length : ℝ := garden_length - 2 * walkway_width
  pool_width : ℝ := garden_width - 2 * walkway_width

/-- Calculates the perimeter of the pool --/
def pool_perimeter (p : PoolInGarden) : ℝ :=
  2 * (p.pool_length + p.pool_width)

/-- Theorem: The perimeter of the pool is 20 meters --/
theorem pool_perimeter_is_20 (p : PoolInGarden) 
    (h1 : p.garden_length = 8)
    (h2 : p.garden_width = 6)
    (h3 : p.pool_area = 24)
    (h4 : p.pool_length * p.pool_width = p.pool_area) : 
  pool_perimeter p = 20 := by
  sorry

#check pool_perimeter_is_20

end NUMINAMATH_CALUDE_pool_perimeter_is_20_l2367_236746


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l2367_236725

theorem quadratic_root_condition (d : ℚ) : 
  (∀ x : ℚ, 2 * x^2 + 11 * x + d = 0 ↔ x = (-11 + Real.sqrt 15) / 4 ∨ x = (-11 - Real.sqrt 15) / 4) → 
  d = 53 / 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l2367_236725


namespace NUMINAMATH_CALUDE_cube_shortest_distances_l2367_236768

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of the cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length) ∨ 
               (y = 0 ∨ y = c.edge_length) ∨ 
               (z = 0 ∨ z = c.edge_length)

/-- The shortest distance between two points on the cube's surface -/
def shortest_distance (c : Cube) (p1 p2 : CubePoint c) : ℝ :=
  sorry

/-- Two adjacent vertices of the cube -/
def adjacent_vertices (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨0, 0, 0, by simp⟩, ⟨c.edge_length, 0, 0, by simp⟩⟩

/-- Two points on adjacent edges, each 1 unit from their common vertex -/
def adjacent_edge_points (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨c.edge_length, 0, c.edge_length, by simp⟩, ⟨c.edge_length, c.edge_length, 0, by simp⟩⟩

/-- Two non-adjacent vertices of the cube -/
def non_adjacent_vertices (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨0, 0, 0, by simp⟩, ⟨c.edge_length, c.edge_length, 0, by simp⟩⟩

theorem cube_shortest_distances (c : Cube) :
  let (v1, v2) := adjacent_vertices c
  let (p1, p2) := adjacent_edge_points c
  let (u1, u2) := non_adjacent_vertices c
  shortest_distance c v1 v2 = 1 ∧
  shortest_distance c p1 p2 = 2 ∧
  shortest_distance c u1 u2 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_shortest_distances_l2367_236768


namespace NUMINAMATH_CALUDE_circle_locus_line_theorem_l2367_236714

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the locus of M
def locus_M (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x - Real.sqrt 3)

-- Define the length product condition
def length_product_condition (k : ℝ) : Prop :=
  let d := |Real.sqrt 3 * k| / Real.sqrt (1 + k^2)
  let AB := 2 * Real.sqrt (4 - d^2)
  let CD := 4 * (1 + k^2) / (1 + 4 * k^2)
  AB * CD = 8 * Real.sqrt 10 / 5

-- Main theorem
theorem circle_locus_line_theorem :
  ∀ (k : ℝ),
  (∀ (x y : ℝ), circle_O x y → locus_M (x/2) (y/2)) ∧
  (length_product_condition k ↔ (k = 1 ∨ k = -1)) :=
sorry

end NUMINAMATH_CALUDE_circle_locus_line_theorem_l2367_236714


namespace NUMINAMATH_CALUDE_fraction_meaningful_l2367_236749

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l2367_236749


namespace NUMINAMATH_CALUDE_hillarys_descending_rate_l2367_236789

/-- Proves that Hillary's descending rate is 1000 ft/hr given the conditions of the climbing problem -/
theorem hillarys_descending_rate 
  (base_camp_distance : ℝ) 
  (hillary_climbing_rate : ℝ) 
  (eddy_climbing_rate : ℝ) 
  (hillary_stop_distance : ℝ) 
  (start_time : ℝ) 
  (passing_time : ℝ) :
  base_camp_distance = 5000 →
  hillary_climbing_rate = 800 →
  eddy_climbing_rate = 500 →
  hillary_stop_distance = 1000 →
  start_time = 6 →
  passing_time = 12 →
  ∃ (hillary_descending_rate : ℝ), hillary_descending_rate = 1000 := by
  sorry

#check hillarys_descending_rate

end NUMINAMATH_CALUDE_hillarys_descending_rate_l2367_236789


namespace NUMINAMATH_CALUDE_quadratic_solution_square_l2367_236772

theorem quadratic_solution_square (y : ℝ) : 
  6 * y^2 + 2 = 4 * y + 12 → (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_square_l2367_236772


namespace NUMINAMATH_CALUDE_greatest_consecutive_nonneg_integers_sum_120_l2367_236719

theorem greatest_consecutive_nonneg_integers_sum_120 :
  ∀ n : ℕ, (∃ a : ℕ, (n : ℕ) * (2 * a + n - 1) = 240) →
  n ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_nonneg_integers_sum_120_l2367_236719


namespace NUMINAMATH_CALUDE_gcf_lcm_300_125_l2367_236799

theorem gcf_lcm_300_125 :
  (Nat.gcd 300 125 = 25) ∧ (Nat.lcm 300 125 = 1500) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_300_125_l2367_236799


namespace NUMINAMATH_CALUDE_total_hours_is_fifty_l2367_236739

/-- Calculates the total hours needed to make dresses from two types of fabric -/
def total_hours_for_dresses (fabric_a_total : ℕ) (fabric_a_per_dress : ℕ) (fabric_a_hours : ℕ)
                            (fabric_b_total : ℕ) (fabric_b_per_dress : ℕ) (fabric_b_hours : ℕ) : ℕ :=
  let dresses_a := fabric_a_total / fabric_a_per_dress
  let dresses_b := fabric_b_total / fabric_b_per_dress
  dresses_a * fabric_a_hours + dresses_b * fabric_b_hours

/-- Theorem stating that the total hours needed to make dresses from the given fabrics is 50 -/
theorem total_hours_is_fifty :
  total_hours_for_dresses 40 4 3 28 5 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_is_fifty_l2367_236739


namespace NUMINAMATH_CALUDE_delores_initial_money_l2367_236705

def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left : ℕ := 10

theorem delores_initial_money : 
  computer_cost + printer_cost + money_left = 450 := by sorry

end NUMINAMATH_CALUDE_delores_initial_money_l2367_236705


namespace NUMINAMATH_CALUDE_third_term_is_twenty_l2367_236740

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  is_geometric : ∀ n : ℕ, n > 0 → terms (n + 1) * terms (n - 1) = (terms n) ^ 2

/-- Our specific geometric sequence -/
def our_sequence : GeometricSequence where
  terms := sorry
  is_geometric := sorry

theorem third_term_is_twenty 
  (h1 : our_sequence.terms 1 = 5)
  (h5 : our_sequence.terms 5 = 320) : 
  our_sequence.terms 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_twenty_l2367_236740


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2367_236743

theorem cos_150_degrees : Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2367_236743


namespace NUMINAMATH_CALUDE_given_scenario_is_combination_l2367_236788

/-- Represents the type of course selection problem -/
inductive SelectionType
  | Permutation
  | Combination

/-- Represents a course selection scenario -/
structure CourseSelection where
  typeA : ℕ  -- Number of type A courses
  typeB : ℕ  -- Number of type B courses
  total : ℕ  -- Total number of courses to be selected
  atLeastOneEach : Bool  -- Whether at least one of each type is required

/-- Determines the type of selection problem based on the given scenario -/
def selectionProblemType (scenario : CourseSelection) : SelectionType :=
  sorry

/-- The specific scenario from the problem -/
def givenScenario : CourseSelection := {
  typeA := 3
  typeB := 4
  total := 3
  atLeastOneEach := true
}

/-- Theorem stating that the given scenario is a combination problem -/
theorem given_scenario_is_combination :
  selectionProblemType givenScenario = SelectionType.Combination := by
  sorry

end NUMINAMATH_CALUDE_given_scenario_is_combination_l2367_236788


namespace NUMINAMATH_CALUDE_even_square_diff_implies_even_sum_l2367_236755

theorem even_square_diff_implies_even_sum (n m : ℤ) (h : Even (n^2 - m^2)) : Even (n + m) := by
  sorry

end NUMINAMATH_CALUDE_even_square_diff_implies_even_sum_l2367_236755


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l2367_236777

/-- The area of a region bounded by three circular arcs --/
theorem area_of_three_arc_region :
  let r : ℝ := 6  -- radius of each circle
  let θ : ℝ := 90  -- central angle in degrees
  let area_sector : ℝ := (θ / 360) * π * r^2
  let area_triangle : ℝ := (1 / 2) * r^2
  let area_segment : ℝ := area_sector - area_triangle
  let total_area : ℝ := 3 * area_segment
  total_area = 27 * π - 54 :=
by
  sorry

/-- The sum of a, b, and c in the expression a√b + cπ --/
theorem sum_of_coefficients :
  let a : ℝ := 0
  let b : ℝ := 1
  let c : ℝ := 27
  a + b + c = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l2367_236777


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2367_236786

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, (x - y) * x^4 < 0 → x < y) ∧
  (∃ x y : ℝ, x < y ∧ (x - y) * x^4 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2367_236786


namespace NUMINAMATH_CALUDE_cookies_eaten_l2367_236707

theorem cookies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) 
  (h1 : initial = 93)
  (h2 : remaining = 78)
  (h3 : initial = remaining + eaten) :
  eaten = 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_eaten_l2367_236707


namespace NUMINAMATH_CALUDE_company_profit_calculation_l2367_236790

/-- Given a company's total annual profit and the difference between first and second half profits,
    prove that the second half profit is as calculated. -/
theorem company_profit_calculation (total_profit second_half_profit : ℚ) : 
  total_profit = 3635000 →
  second_half_profit + 2750000 + second_half_profit = total_profit →
  second_half_profit = 442500 := by
  sorry

end NUMINAMATH_CALUDE_company_profit_calculation_l2367_236790


namespace NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2367_236700

/-- Given a ship's passenger statistics, calculate the percentage of round-trip ticket holders. -/
theorem round_trip_ticket_percentage
  (total_passengers : ℝ)
  (h1 : total_passengers > 0)
  (h2 : (20 : ℝ) / 100 * total_passengers = (60 : ℝ) / 100 * (round_trip_tickets : ℝ)) :
  (round_trip_tickets : ℝ) / total_passengers = (100 : ℝ) / 3 :=
by sorry

#check round_trip_ticket_percentage

end NUMINAMATH_CALUDE_round_trip_ticket_percentage_l2367_236700


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2367_236732

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (5 * Complex.I) / (4 + 3 * Complex.I)
  Complex.im z = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2367_236732


namespace NUMINAMATH_CALUDE_john_earnings_before_raise_l2367_236718

theorem john_earnings_before_raise (new_earnings : ℝ) (increase_percentage : ℝ) :
  new_earnings = 75 ∧ increase_percentage = 50 →
  ∃ original_earnings : ℝ,
    original_earnings * (1 + increase_percentage / 100) = new_earnings ∧
    original_earnings = 50 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_before_raise_l2367_236718


namespace NUMINAMATH_CALUDE_relationship_abc_l2367_236729

/-- Given the definitions of a, b, and c, prove that a < c < b -/
theorem relationship_abc : 
  let a := (1/2) * Real.cos (80 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (80 * π / 180)
  let b := (2 * Real.tan (13 * π / 180)) / (1 - Real.tan (13 * π / 180)^2)
  let c := Real.sqrt ((1 - Real.cos (52 * π / 180)) / 2)
  a < c ∧ c < b := by
sorry


end NUMINAMATH_CALUDE_relationship_abc_l2367_236729


namespace NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2367_236741

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 2 * Real.sin A + 3 * Real.cos B = 4) 
  (h2 : 3 * Real.sin B + 2 * Real.cos A = Real.sqrt 3) 
  (h3 : A + B + C = Real.pi) : C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_in_triangle_ABC_l2367_236741


namespace NUMINAMATH_CALUDE_red_car_cost_is_three_l2367_236785

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := sorry

/-- Represents the number of red cars -/
def num_red_cars : ℕ := 3

/-- Represents the number of white cars -/
def num_white_cars : ℕ := 2

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the rental duration in minutes -/
def rental_duration : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem red_car_cost_is_three :
  red_car_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_red_car_cost_is_three_l2367_236785


namespace NUMINAMATH_CALUDE_tangent_asymptote_implies_m_value_l2367_236744

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8 = 0

-- Define the hyperbola
def hyperbola_equation (x y m : ℝ) : Prop :=
  y^2 - x^2/m^2 = 1

-- Define the asymptote of the hyperbola
def asymptote_equation (x y m : ℝ) : Prop :=
  y = x/m ∨ y = -x/m

-- Main theorem
theorem tangent_asymptote_implies_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ x y : ℝ, circle_equation x y ∧ 
    asymptote_equation x y m ∧
    hyperbola_equation x y m) →
  m = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_asymptote_implies_m_value_l2367_236744


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l2367_236773

theorem opposite_of_negative_three :
  ∀ x : ℤ, ((-3 : ℤ) + x = 0) → x = 3 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l2367_236773


namespace NUMINAMATH_CALUDE_rectangle_width_l2367_236730

/-- Given a rectangle with length 20 and perimeter 70, prove its width is 15 -/
theorem rectangle_width (length perimeter : ℝ) (h1 : length = 20) (h2 : perimeter = 70) :
  let width := (perimeter - 2 * length) / 2
  width = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l2367_236730


namespace NUMINAMATH_CALUDE_estimate_shadowed_area_l2367_236752

/-- Estimate the area of a shadowed region in a square based on bean distribution --/
theorem estimate_shadowed_area (total_area : ℝ) (total_beans : ℕ) (outside_beans : ℕ) 
  (h1 : total_area = 10) 
  (h2 : total_beans = 200) 
  (h3 : outside_beans = 114) : 
  ∃ (estimated_area : ℝ), abs (estimated_area - (total_area - (outside_beans : ℝ) / (total_beans : ℝ) * total_area)) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_estimate_shadowed_area_l2367_236752


namespace NUMINAMATH_CALUDE_billy_and_sam_money_l2367_236778

/-- The amount of money Sam has -/
def sam_money : ℕ := 75

/-- The amount of money Billy has -/
def billy_money : ℕ := 2 * sam_money - 25

/-- The total amount of money Billy and Sam have together -/
def total_money : ℕ := sam_money + billy_money

theorem billy_and_sam_money : total_money = 200 := by
  sorry

end NUMINAMATH_CALUDE_billy_and_sam_money_l2367_236778


namespace NUMINAMATH_CALUDE_nissa_cat_grooming_time_l2367_236712

/-- Represents the time in seconds for various cat grooming activities -/
structure CatGroomingTime where
  clip_claw : ℕ
  clean_ear : ℕ
  shampoo : ℕ
  brush_fur : ℕ
  give_treat : ℕ
  trim_fur : ℕ

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (t : CatGroomingTime) : ℕ :=
  t.clip_claw * 16 + t.clean_ear * 2 + t.shampoo + t.brush_fur + t.give_treat + t.trim_fur

/-- Theorem stating that the total grooming time for Nissa's cat is 970 seconds -/
theorem nissa_cat_grooming_time :
  ∃ (t : CatGroomingTime),
    t.clip_claw = 10 ∧
    t.clean_ear = 90 ∧
    t.shampoo = 300 ∧
    t.brush_fur = 120 ∧
    t.give_treat = 30 ∧
    t.trim_fur = 180 ∧
    total_grooming_time t = 970 :=
by
  sorry


end NUMINAMATH_CALUDE_nissa_cat_grooming_time_l2367_236712


namespace NUMINAMATH_CALUDE_function_always_positive_l2367_236775

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x + (x - 1) * (deriv f x) > 0) : 
  ∀ x, f x > 0 := by
sorry

end NUMINAMATH_CALUDE_function_always_positive_l2367_236775


namespace NUMINAMATH_CALUDE_rectangle_enclosure_l2367_236761

def rectangle_largest_side (length width : ℝ) : Prop :=
  let perimeter := 2 * (length + width)
  let area := length * width
  perimeter = 240 ∧ 
  area = 12 * perimeter ∧ 
  length ≥ width ∧
  length = 72

theorem rectangle_enclosure :
  ∃ (length width : ℝ), rectangle_largest_side length width :=
sorry

end NUMINAMATH_CALUDE_rectangle_enclosure_l2367_236761


namespace NUMINAMATH_CALUDE_parcel_weight_sum_l2367_236762

/-- Given three parcels with weights x, y, and z, prove that their total weight is 195 pounds
    if the sum of each pair of parcels weighs 112, 146, and 132 pounds respectively. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (pair_xy : x + y = 112)
  (pair_yz : y + z = 146)
  (pair_zx : z + x = 132) :
  x + y + z = 195 := by
  sorry

end NUMINAMATH_CALUDE_parcel_weight_sum_l2367_236762


namespace NUMINAMATH_CALUDE_intersection_chord_length_l2367_236706

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : ℝ → ℝ → ℝ → Prop

/-- Calculates the chord length of intersection between a circle and a line -/
def chordLength (c : PolarCircle) (l : PolarLine) : ℝ := sorry

/-- Main theorem: If a circle ρ = 4cosθ is intersected by a line ρsin(θ - φ) = a 
    with a chord length of 2, then a = 0 or a = -2 -/
theorem intersection_chord_length 
  (c : PolarCircle) 
  (l : PolarLine) 
  (h1 : c.equation = λ ρ θ => ρ = 4 * Real.cos θ)
  (h2 : l.equation = λ ρ θ φ => ρ * Real.sin (θ - φ) = a)
  (h3 : chordLength c l = 2) :
  a = 0 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l2367_236706


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2367_236716

/-- The jumping distances of animals in a contest -/
structure JumpContest where
  frog : ℕ
  grasshopper : ℕ
  grasshopper_frog_diff : grasshopper = frog + 4

/-- Theorem: In a jump contest where the frog jumped 15 inches and the grasshopper
    jumped 4 inches farther than the frog, the grasshopper's jump distance is 19 inches. -/
theorem grasshopper_jump_distance (contest : JumpContest) 
  (h : contest.frog = 15) : contest.grasshopper = 19 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2367_236716


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2367_236733

theorem water_tank_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 5 →
  final_fraction = 2/5 →
  (initial_fraction * added_amount) / (final_fraction - initial_fraction) = 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2367_236733


namespace NUMINAMATH_CALUDE_fourth_column_third_row_position_l2367_236771

-- Define a type for classroom positions
def ClassroomPosition := ℕ × ℕ

-- Define a function that creates a classroom position from column and row numbers
def makePosition (column : ℕ) (row : ℕ) : ClassroomPosition := (column, row)

-- Theorem statement
theorem fourth_column_third_row_position :
  makePosition 4 3 = (4, 3) := by sorry

end NUMINAMATH_CALUDE_fourth_column_third_row_position_l2367_236771


namespace NUMINAMATH_CALUDE_quadratic_roots_l2367_236726

theorem quadratic_roots (b c : ℝ) (h1 : 3 ∈ {x : ℝ | x^2 + b*x + c = 0}) 
  (h2 : 5 ∈ {x : ℝ | x^2 + b*x + c = 0}) :
  {y : ℝ | (y^2 + 4)^2 + b*(y^2 + 4) + c = 0} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2367_236726


namespace NUMINAMATH_CALUDE_discount_calculation_l2367_236738

/-- Given a 25% discount on a purchase where the final price paid is $120, prove that the discount amount is $40. -/
theorem discount_calculation (original_price : ℝ) : 
  (original_price * 0.75 = 120) → (original_price - 120 = 40) := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l2367_236738


namespace NUMINAMATH_CALUDE_unequal_gender_probability_l2367_236781

theorem unequal_gender_probability : 
  let n : ℕ := 12  -- Total number of grandchildren
  let p : ℚ := 1/2 -- Probability of each child being male (or female)
  -- Probability of unequal number of grandsons and granddaughters
  (1 : ℚ) - (n.choose (n/2) : ℚ) / 2^n = 793/1024 :=
by sorry

end NUMINAMATH_CALUDE_unequal_gender_probability_l2367_236781


namespace NUMINAMATH_CALUDE_watch_cost_price_l2367_236767

theorem watch_cost_price (cp : ℝ) : 
  (0.9 * cp = cp - 0.1 * cp) →
  (1.04 * cp = cp + 0.04 * cp) →
  (1.04 * cp = 0.9 * cp + 200) →
  cp = 1428.57 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2367_236767


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2367_236758

theorem simplify_and_evaluate (x : ℝ) (h : x = 5) :
  x^2 * (x + 1) - x * (x^2 - x + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2367_236758


namespace NUMINAMATH_CALUDE_range_of_f_l2367_236731

def f (x : ℕ) : ℤ := 3 * x - 1

def domain : Set ℕ := {x | 1 ≤ x ∧ x ≤ 4}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {2, 5, 8, 11} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2367_236731


namespace NUMINAMATH_CALUDE_triangle_properties_l2367_236715

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively
  (S : Real)      -- Area

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.c * t.a * Real.cos t.C) = 2 * t.S

def condition2 (t : Triangle) : Prop :=
  (Real.sin t.C + Real.sin t.A) * (Real.sin t.C - Real.sin t.A) = 
  Real.sin t.B * (Real.sin t.B - Real.sin t.A)

def condition3 (t : Triangle) : Prop :=
  (2 * t.a - t.b) * Real.cos t.C = t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) →
  t.C = Real.pi / 3 ∧
  (t.c = 2 → t.S ≤ Real.sqrt 3 ∧ 
   ∃ (t' : Triangle), t'.c = 2 ∧ t'.S = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2367_236715


namespace NUMINAMATH_CALUDE_sum_of_roots_l2367_236756

theorem sum_of_roots (N : ℝ) : N * (N - 6) = -7 → ∃ N₁ N₂ : ℝ, N₁ * (N₁ - 6) = -7 ∧ N₂ * (N₂ - 6) = -7 ∧ N₁ + N₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2367_236756


namespace NUMINAMATH_CALUDE_all_expressions_are_identities_l2367_236787

theorem all_expressions_are_identities (x y : ℝ) : 
  ((2*x - 1) * (x - 3) = 2*x^2 - 7*x + 3) ∧
  ((2*x + 1) * (x + 3) = 2*x^2 + 7*x + 3) ∧
  ((2 - x) * (1 - 3*x) = 2 - 7*x + 3*x^2) ∧
  ((2 + x) * (1 + 3*x) = 2 + 7*x + 3*x^2) ∧
  ((2*x - y) * (x - 3*y) = 2*x^2 - 7*x*y + 3*y^2) ∧
  ((2*x + y) * (x + 3*y) = 2*x^2 + 7*x*y + 3*y^2) :=
by
  sorry

#check all_expressions_are_identities

end NUMINAMATH_CALUDE_all_expressions_are_identities_l2367_236787


namespace NUMINAMATH_CALUDE_expression_simplification_l2367_236704

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4*x + 4) / (x - 1)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2367_236704


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_5_7_l2367_236710

/-- Definition of our series of pairs -/
def pair_series : ℕ → ℕ × ℕ
| n => sorry

/-- The sum of the components of the nth pair -/
def pair_sum (n : ℕ) : ℕ := (pair_series n).1 + (pair_series n).2

/-- The 60th pair in the series -/
def sixtieth_pair : ℕ × ℕ := pair_series 60

/-- Theorem stating that the 60th pair is (5,7) -/
theorem sixtieth_pair_is_5_7 : sixtieth_pair = (5, 7) := by sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_5_7_l2367_236710


namespace NUMINAMATH_CALUDE_beef_jerky_ratio_l2367_236765

/-- Proves that the ratio of beef jerky pieces Janette gives to her brother
    to the pieces she keeps for herself is 1:1 --/
theorem beef_jerky_ratio (days : ℕ) (initial_pieces : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ)
  (pieces_left : ℕ) :
  days = 5 →
  initial_pieces = 40 →
  breakfast = 1 →
  lunch = 1 →
  dinner = 2 →
  pieces_left = 10 →
  let daily_consumption := breakfast + lunch + dinner
  let total_consumption := daily_consumption * days
  let remaining_after_trip := initial_pieces - total_consumption
  let given_to_brother := remaining_after_trip - pieces_left
  (given_to_brother : ℚ) / pieces_left = 1 := by
sorry

end NUMINAMATH_CALUDE_beef_jerky_ratio_l2367_236765


namespace NUMINAMATH_CALUDE_geometry_theorem_l2367_236747

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (parallel m α ∧ perpendicular n α → perpendicular_lines m n) ∧
  (perpendicular m α ∧ parallel m β → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2367_236747


namespace NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2367_236728

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents 11101₂ as a list of bits. -/
def binary_11101 : List Bool := [true, true, true, false, true]

/-- Represents 10011₂ as a list of bits. -/
def binary_10011 : List Bool := [true, false, false, true, true]

/-- Represents 101₂ as a list of bits. -/
def binary_101 : List Bool := [true, false, true]

/-- Represents 11101100₂ as a list of bits. -/
def binary_11101100 : List Bool := [true, true, true, false, true, true, false, false]

/-- The main theorem to prove. -/
theorem binary_multiplication_division_equality :
  (binary_to_nat binary_11101 * binary_to_nat binary_10011) / binary_to_nat binary_101 =
  binary_to_nat binary_11101100 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_equality_l2367_236728


namespace NUMINAMATH_CALUDE_largest_angle_measure_l2367_236753

/-- A triangle PQR is obtuse and isosceles with angle P measuring 30 degrees. -/
structure ObtusePQR where
  /-- Triangle PQR is obtuse -/
  obtuse : Bool
  /-- Triangle PQR is isosceles -/
  isosceles : Bool
  /-- Angle P measures 30 degrees -/
  angle_p : ℝ
  /-- Angle P is 30 degrees -/
  h_angle_p : angle_p = 30

/-- The measure of the largest interior angle in triangle PQR is 120 degrees -/
theorem largest_angle_measure (t : ObtusePQR) : ℝ := by
  sorry

#check largest_angle_measure

end NUMINAMATH_CALUDE_largest_angle_measure_l2367_236753


namespace NUMINAMATH_CALUDE_bracket_ratio_eq_neg_199_l2367_236770

/-- Definition of the bracket operation -/
def bracket (a : ℝ) (k : ℕ+) : ℝ := a * (a - k)

/-- The main theorem to prove -/
theorem bracket_ratio_eq_neg_199 :
  (bracket (-1/2) 100) / (bracket (1/2) 100) = -199 := by sorry

end NUMINAMATH_CALUDE_bracket_ratio_eq_neg_199_l2367_236770


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2367_236780

theorem right_triangle_max_ratio :
  ∀ (a b c h : ℝ),
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →
  h * c = a * b →
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2367_236780


namespace NUMINAMATH_CALUDE_function_composition_problem_l2367_236742

/-- Given two functions f and g satisfying certain conditions, prove that [g(9)]^4 = 81 -/
theorem function_composition_problem 
  (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 81 = 81) :
  (g 9)^4 = 81 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_problem_l2367_236742


namespace NUMINAMATH_CALUDE_stone_counting_l2367_236701

theorem stone_counting (n : Nat) (h : n = 99) : n % 16 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stone_counting_l2367_236701


namespace NUMINAMATH_CALUDE_tangent_slope_point_l2367_236724

theorem tangent_slope_point (x₀ : ℝ) :
  let f : ℝ → ℝ := fun x ↦ Real.exp (-x)
  let y₀ : ℝ := f x₀
  (deriv f x₀ = -2) → (x₀ = -Real.log 2 ∧ y₀ = 2) := by
sorry

end NUMINAMATH_CALUDE_tangent_slope_point_l2367_236724


namespace NUMINAMATH_CALUDE_ring_toss_game_l2367_236797

/-- The ring toss game problem -/
theorem ring_toss_game (total_amount : ℕ) (daily_revenue : ℕ) (second_period : ℕ) : 
  total_amount = 186 → 
  daily_revenue = 6 →
  second_period = 16 →
  ∃ (first_period : ℕ), first_period * daily_revenue + second_period * (total_amount - first_period * daily_revenue) / second_period = total_amount ∧ 
                         first_period = 20 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_game_l2367_236797


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2367_236735

theorem complex_arithmetic_equality : 
  (1000 + 15 + 314) * (201 + 360 + 110) + (1000 - 201 - 360 - 110) * (15 + 314) = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2367_236735


namespace NUMINAMATH_CALUDE_log_product_equivalence_l2367_236784

open Real

theorem log_product_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) (hy_neq_1 : y ≠ 1) :
  (log x / log (y^4)) * (log (y^6) / log (x^3)) * (log (x^2) / log (y^3)) *
  (log (y^3) / log (x^2)) * (log (x^3) / log (y^6)) = (3/4) * (log x / log y) := by
  sorry

end NUMINAMATH_CALUDE_log_product_equivalence_l2367_236784


namespace NUMINAMATH_CALUDE_james_total_toys_l2367_236776

/-- The minimum number of toy cars needed to get a discount -/
def discount_threshold : ℕ := 25

/-- The initial number of toy cars James buys -/
def initial_cars : ℕ := 20

/-- The ratio of toy soldiers to toy cars -/
def soldier_to_car_ratio : ℕ := 2

/-- The total number of toys James buys to maximize his discount -/
def total_toys : ℕ := 78

/-- Theorem stating that the total number of toys James buys is 78 -/
theorem james_total_toys :
  let additional_cars := discount_threshold + 1 - initial_cars
  let total_cars := initial_cars + additional_cars
  let total_soldiers := soldier_to_car_ratio * total_cars
  total_cars + total_soldiers = total_toys :=
by sorry

end NUMINAMATH_CALUDE_james_total_toys_l2367_236776


namespace NUMINAMATH_CALUDE_sequence_sum_l2367_236703

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (h : ∀ n, S n = n^3) :
  a 6 + a 7 + a 8 + a 9 = 604 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l2367_236703


namespace NUMINAMATH_CALUDE_mans_speed_mans_speed_specific_l2367_236745

/-- The speed of a man running in the same direction as a train, given the train's length, speed, and time to cross the man. -/
theorem mans_speed (train_length : Real) (train_speed_kmh : Real) (time_to_cross : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / time_to_cross
  let mans_speed_ms := train_speed_ms - relative_speed
  let mans_speed_kmh := mans_speed_ms * 3600 / 1000
  mans_speed_kmh

/-- Given the specific conditions, prove that the man's speed is approximately 8 km/hr. -/
theorem mans_speed_specific : 
  ∃ (ε : Real), ε > 0 ∧ ε < 0.1 ∧ 
  |mans_speed 620 80 30.99752019838413 - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_mans_speed_mans_speed_specific_l2367_236745


namespace NUMINAMATH_CALUDE_average_score_theorem_l2367_236734

/-- The average score of a class on a test with given score distribution -/
theorem average_score_theorem (num_questions : ℕ) (num_students : ℕ) 
  (prop_score_3 : ℚ) (prop_score_2 : ℚ) (prop_score_1 : ℚ) (prop_score_0 : ℚ) :
  num_questions = 3 →
  num_students = 50 →
  prop_score_3 = 30 / 100 →
  prop_score_2 = 50 / 100 →
  prop_score_1 = 10 / 100 →
  prop_score_0 = 10 / 100 →
  prop_score_3 + prop_score_2 + prop_score_1 + prop_score_0 = 1 →
  3 * prop_score_3 + 2 * prop_score_2 + 1 * prop_score_1 + 0 * prop_score_0 = 2 := by
  sorry

#check average_score_theorem

end NUMINAMATH_CALUDE_average_score_theorem_l2367_236734
