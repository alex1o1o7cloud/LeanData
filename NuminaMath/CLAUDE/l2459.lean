import Mathlib

namespace NUMINAMATH_CALUDE_least_side_of_right_triangle_l2459_245995

theorem least_side_of_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a = 8 → b = 15 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_least_side_of_right_triangle_l2459_245995


namespace NUMINAMATH_CALUDE_sin_2phi_value_l2459_245979

theorem sin_2phi_value (φ : ℝ) 
  (h : ∫ x in (0)..(Real.pi / 2), Real.sin (x - φ) = Real.sqrt 7 / 4) : 
  Real.sin (2 * φ) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sin_2phi_value_l2459_245979


namespace NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l2459_245941

theorem unique_solution_for_prime_equation :
  ∃! n : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n^2 = p^2 + 3*p + 9 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_prime_equation_l2459_245941


namespace NUMINAMATH_CALUDE_number_division_problem_l2459_245924

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - 14) / y = 4) : 
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l2459_245924


namespace NUMINAMATH_CALUDE_expression_evaluation_l2459_245965

theorem expression_evaluation :
  let x : ℚ := -1/2
  let expr := 2*x^2 + 6*x - 6 - (-2*x^2 + 4*x + 1)
  expr = -7 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2459_245965


namespace NUMINAMATH_CALUDE_quadratic_expression_rewrite_l2459_245959

theorem quadratic_expression_rewrite (i j : ℂ) : 
  let expression := 8 * j^2 + (6 * i) * j + 16
  ∃ (c p q : ℂ), 
    expression = c * (j + p)^2 + q ∧ 
    q / p = -137 * I / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_rewrite_l2459_245959


namespace NUMINAMATH_CALUDE_exists_tangent_region_l2459_245925

noncomputable section

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := x - x^2
def parabola2 (a : ℝ) (x : ℝ) : ℝ := a * (x - x^2)

-- Define the tangent lines
def tangent1 (b : ℝ) (x : ℝ) : ℝ := (1 - 2*b)*x + b^2
def tangent2 (a : ℝ) (c : ℝ) (x : ℝ) : ℝ := a*(1 - 2*c)*x + a*c^2

-- Define the intersection point of tangent lines
def intersection (a b c : ℝ) : ℝ := (a*c^2 - b^2) / ((1 - 2*b) - a*(1 - 2*c))

-- Define the condition for the third point
def third_point (x b : ℝ) : ℝ := 2*x - b

-- Theorem statement
theorem exists_tangent_region (a : ℝ) (h : a ≥ 2) :
  ∃ (b c : ℝ), 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1 ∧
  let x := intersection a b c
  let d := third_point x b
  0 < d ∧ d < 1 ∧
  (tangent1 b x = tangent2 a c x ∨ tangent1 d x = tangent1 b x ∨ tangent2 a d x = tangent2 a c x) :=
sorry

end NUMINAMATH_CALUDE_exists_tangent_region_l2459_245925


namespace NUMINAMATH_CALUDE_coefficient_x_squared_eq_40_l2459_245909

/-- The coefficient of x^2 in the expansion of (1+2x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 2) * 2^2

/-- Theorem stating that the coefficient of x^2 in (1+2x)^5 is 40 -/
theorem coefficient_x_squared_eq_40 : coefficient_x_squared = 40 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_eq_40_l2459_245909


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2459_245935

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  -- Add any necessary properties for a convex polygon

/-- The number of diagonals in a convex polygon that skip exactly one vertex -/
def diagonals_skipping_one_vertex (n : ℕ) : ℕ := 2 * n

/-- Theorem: In a convex 25-sided polygon, there are 50 diagonals that skip exactly one vertex -/
theorem diagonals_25_sided_polygon :
  diagonals_skipping_one_vertex 25 = 50 := by
  sorry

#eval diagonals_skipping_one_vertex 25

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2459_245935


namespace NUMINAMATH_CALUDE_project_hours_total_l2459_245922

/-- Given the conditions of the project hours charged by Pat, Kate, and Mark, 
    prove that the total number of hours charged is 144. -/
theorem project_hours_total (k p m : ℕ) : 
  p = 2 * k →          -- Pat charged twice as much as Kate
  3 * p = m →          -- Pat charged 1/3 as much as Mark
  m = k + 80 →         -- Mark charged 80 more hours than Kate
  k + p + m = 144 :=   -- Total hours charged
by sorry

end NUMINAMATH_CALUDE_project_hours_total_l2459_245922


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2459_245947

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (right_angle : a^2 + b^2 = c^2) 
  (hypotenuse : c = 25) 
  (known_leg : a = 24) : 
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2459_245947


namespace NUMINAMATH_CALUDE_expand_product_l2459_245900

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2459_245900


namespace NUMINAMATH_CALUDE_gcd_24_36_l2459_245992

theorem gcd_24_36 : Nat.gcd 24 36 = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_24_36_l2459_245992


namespace NUMINAMATH_CALUDE_age_problem_l2459_245926

/-- Represents the ages of Sandy, Molly, and Kim -/
structure Ages where
  sandy : ℝ
  molly : ℝ
  kim : ℝ

/-- The problem statement -/
theorem age_problem (current : Ages) (future : Ages) : 
  -- Current ratio condition
  (current.sandy / current.molly = 4 / 3) ∧
  (current.sandy / current.kim = 4 / 5) ∧
  -- Future age condition
  (future.sandy = current.sandy + 8) ∧
  (future.molly = current.molly + 8) ∧
  (future.kim = current.kim + 8) ∧
  -- Future Sandy's age
  (future.sandy = 74) ∧
  -- Future ratio condition
  (future.sandy / future.molly = 9 / 7) ∧
  (future.sandy / future.kim = 9 / 10) →
  -- Conclusion
  current.molly = 49.5 ∧ current.kim = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l2459_245926


namespace NUMINAMATH_CALUDE_oldest_sibling_age_is_44_l2459_245982

def kay_age : ℕ := 32

def youngest_sibling_age : ℕ := kay_age / 2 - 5

def oldest_sibling_age : ℕ := 4 * youngest_sibling_age

theorem oldest_sibling_age_is_44 : oldest_sibling_age = 44 := by
  sorry

end NUMINAMATH_CALUDE_oldest_sibling_age_is_44_l2459_245982


namespace NUMINAMATH_CALUDE_town_population_growth_l2459_245937

/-- The final population after compound growth --/
def final_population (initial_population : ℕ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_population * (1 + growth_rate) ^ years

/-- Theorem stating the approximate final population after a decade --/
theorem town_population_growth : 
  ∃ (result : ℕ), 
    344251 ≤ result ∧ 
    result ≤ 344252 ∧ 
    result = ⌊final_population 175000 0.07 10⌋ := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l2459_245937


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2459_245966

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ (p : ℝ), (p = 13 ∨ p = 14) ∧ (p = a + 2*b ∨ p = b + 2*a) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2459_245966


namespace NUMINAMATH_CALUDE_point_vector_relations_l2459_245976

/-- Given points A, B, C in ℝ², and points M, N such that CM = 3CA and CN = 2CB,
    prove that M and N have specific coordinates and MN has a specific value. -/
theorem point_vector_relations (A B C M N : ℝ × ℝ) :
  A = (-2, 4) →
  B = (3, -1) →
  C = (-3, -4) →
  M - C = 3 • (A - C) →
  N - C = 2 • (B - C) →
  M = (0, 20) ∧
  N = (9, 2) ∧
  M - N = (9, -18) := by
  sorry

end NUMINAMATH_CALUDE_point_vector_relations_l2459_245976


namespace NUMINAMATH_CALUDE_quadrilateral_area_with_diagonal_and_offsets_l2459_245918

/-- The area of a quadrilateral with a diagonal and its offsets -/
theorem quadrilateral_area_with_diagonal_and_offsets 
  (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) :
  diagonal = 40 → offset1 = 9 → offset2 = 6 →
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 300 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_with_diagonal_and_offsets_l2459_245918


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2459_245998

theorem max_value_of_expression (x y : ℝ) : 
  |x + 1| - |x - 1| - |y - 4| - |y| ≤ -2 := by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2459_245998


namespace NUMINAMATH_CALUDE_households_using_both_brands_l2459_245902

/-- Proves that the number of households using both brands of soap is 25 --/
theorem households_using_both_brands (total : ℕ) (neither : ℕ) (only_a : ℕ) (h1 : total = 240) (h2 : neither = 80) (h3 : only_a = 60) : 
  ∃ (both : ℕ), both = 25 ∧ total = neither + only_a + both + 3 * both := by
  sorry

end NUMINAMATH_CALUDE_households_using_both_brands_l2459_245902


namespace NUMINAMATH_CALUDE_triangle_side_length_l2459_245955

theorem triangle_side_length (b c : ℝ) (C : ℝ) (h1 : b = 6 * Real.sqrt 3) (h2 : c = 6) (h3 : C = 30 * π / 180) :
  ∃ (a : ℝ), (a = 6 ∨ a = 12) ∧ c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2459_245955


namespace NUMINAMATH_CALUDE_train_length_l2459_245951

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length 
  (train_speed : ℝ) 
  (platform_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed = 72) -- speed in kmph
  (h2 : platform_length = 250) -- length in meters
  (h3 : crossing_time = 36) -- time in seconds
  : ∃ (train_length : ℝ), train_length = 470 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l2459_245951


namespace NUMINAMATH_CALUDE_square_plot_with_path_l2459_245990

theorem square_plot_with_path (path_area : ℝ) (edge_diff : ℝ) (total_area : ℝ) :
  path_area = 464 →
  edge_diff = 32 →
  (∃ x y : ℝ,
    x > 0 ∧
    y > 0 ∧
    x^2 - y^2 = path_area ∧
    4 * (x - y) = edge_diff ∧
    total_area = x^2) →
  total_area = 1089 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_with_path_l2459_245990


namespace NUMINAMATH_CALUDE_congruence_problem_l2459_245973

theorem congruence_problem (a b : ℤ) (h1 : a ≡ 16 [ZMOD 44]) (h2 : b ≡ 77 [ZMOD 44]) :
  (a - b ≡ 159 [ZMOD 44]) ∧
  (∀ n : ℤ, 120 ≤ n ∧ n ≤ 161 → (a - b ≡ n [ZMOD 44] ↔ n = 159)) :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l2459_245973


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2459_245905

def P : Set ℝ := {-3, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2459_245905


namespace NUMINAMATH_CALUDE_largest_integer_with_gcd_six_largest_integer_is_138_l2459_245991

theorem largest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ Nat.gcd n 18 = 6 → n ≤ 138 :=
sorry

theorem largest_integer_is_138 : ∃ n : ℕ, n = 138 ∧ n < 150 ∧ Nat.gcd n 18 = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_integer_with_gcd_six_largest_integer_is_138_l2459_245991


namespace NUMINAMATH_CALUDE_min_people_for_valid_arrangement_l2459_245911

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : ℕ
  seated_people : ℕ

/-- Checks if the seating arrangement satisfies the condition that any additional
    person must sit next to someone already seated. -/
def valid_arrangement (table : CircularTable) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ table.total_chairs →
    ∃ i j, i ≠ j ∧
           i ≤ table.seated_people ∧
           j ≤ table.seated_people ∧
           (k = i ∨ k = j ∨ (i < k ∧ k < j) ∨ (j < k ∧ k < i) ∨ (k < i ∧ j < k) ∨ (k < j ∧ i < k))

/-- The main theorem stating that 20 is the minimum number of people required
    for a valid arrangement on a table with 80 chairs. -/
theorem min_people_for_valid_arrangement :
  ∀ n : ℕ, n < 20 →
    ¬(valid_arrangement { total_chairs := 80, seated_people := n }) ∧
    (valid_arrangement { total_chairs := 80, seated_people := 20 }) := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_valid_arrangement_l2459_245911


namespace NUMINAMATH_CALUDE_time_to_paint_one_room_l2459_245957

theorem time_to_paint_one_room 
  (total_rooms : ℕ) 
  (painted_rooms : ℕ) 
  (time_for_remaining : ℕ) : 
  total_rooms = 10 → 
  painted_rooms = 8 → 
  time_for_remaining = 16 → 
  (time_for_remaining / (total_rooms - painted_rooms) : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_time_to_paint_one_room_l2459_245957


namespace NUMINAMATH_CALUDE_blue_pens_count_l2459_245964

/-- Given the prices of red and blue pens, the total amount spent, and the total number of pens,
    prove that the number of blue pens bought is 11. -/
theorem blue_pens_count (red_price blue_price total_spent total_pens : ℕ) 
    (h1 : red_price = 5)
    (h2 : blue_price = 7)
    (h3 : total_spent = 102)
    (h4 : total_pens = 16) : 
  ∃ (red_count blue_count : ℕ),
    red_count + blue_count = total_pens ∧
    red_count * red_price + blue_count * blue_price = total_spent ∧
    blue_count = 11 := by
  sorry

end NUMINAMATH_CALUDE_blue_pens_count_l2459_245964


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2459_245975

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) ∨
  (∀ x y, a < x ∧ x < y ∧ y < b → f x > f y)

theorem quadratic_monotonicity (a : ℝ) :
  monotonic_on (f a) 2 3 ↔ a ≤ 2 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2459_245975


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2459_245984

/-- The slopes of the asymptotes for the hyperbola (y^2/16) - (x^2/9) = 1 are ±4/3 -/
theorem hyperbola_asymptote_slopes :
  let f (x y : ℝ) := y^2 / 16 - x^2 / 9
  ∃ (m : ℝ), m = 4/3 ∧ 
    (∀ ε > 0, ∃ M > 0, ∀ x y, |x| > M → |y| > M → f x y = 1 → 
      (|y - m*x| < ε*|x| ∨ |y + m*x| < ε*|x|)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l2459_245984


namespace NUMINAMATH_CALUDE_son_age_proof_l2459_245907

theorem son_age_proof (son_age man_age : ℕ) : 
  man_age = son_age + 22 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l2459_245907


namespace NUMINAMATH_CALUDE_vector_problem_l2459_245950

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (3, 4)

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  dot_product v w = 0

theorem vector_problem :
  (∃ k : ℝ, parallel (3 • a - b) (a + k • b) ∧ k = -1/3) ∧
  (∃ m : ℝ, perpendicular a (m • a - b) ∧ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2459_245950


namespace NUMINAMATH_CALUDE_study_group_probability_l2459_245945

/-- Represents the gender distribution in the study group -/
def gender_distribution : Fin 2 → ℝ
  | 0 => 0.55  -- women
  | 1 => 0.45  -- men

/-- Represents the age distribution for each gender -/
def age_distribution : Fin 2 → Fin 3 → ℝ
  | 0, 0 => 0.20  -- women below 35
  | 0, 1 => 0.35  -- women 35-50
  | 0, 2 => 0.45  -- women above 50
  | 1, 0 => 0.30  -- men below 35
  | 1, 1 => 0.40  -- men 35-50
  | 1, 2 => 0.30  -- men above 50

/-- Represents the profession distribution for each gender and age group -/
def profession_distribution : Fin 2 → Fin 3 → Fin 3 → ℝ
  | 0, 0, 0 => 0.35  -- women below 35, lawyers
  | 0, 0, 1 => 0.45  -- women below 35, doctors
  | 0, 0, 2 => 0.20  -- women below 35, engineers
  | 0, 1, 0 => 0.25  -- women 35-50, lawyers
  | 0, 1, 1 => 0.50  -- women 35-50, doctors
  | 0, 1, 2 => 0.25  -- women 35-50, engineers
  | 0, 2, 0 => 0.20  -- women above 50, lawyers
  | 0, 2, 1 => 0.30  -- women above 50, doctors
  | 0, 2, 2 => 0.50  -- women above 50, engineers
  | 1, 0, 0 => 0.40  -- men below 35, lawyers
  | 1, 0, 1 => 0.30  -- men below 35, doctors
  | 1, 0, 2 => 0.30  -- men below 35, engineers
  | 1, 1, 0 => 0.45  -- men 35-50, lawyers
  | 1, 1, 1 => 0.25  -- men 35-50, doctors
  | 1, 1, 2 => 0.30  -- men 35-50, engineers
  | 1, 2, 0 => 0.30  -- men above 50, lawyers
  | 1, 2, 1 => 0.40  -- men above 50, doctors
  | 1, 2, 2 => 0.30  -- men above 50, engineers

theorem study_group_probability : 
  gender_distribution 0 * age_distribution 0 0 * profession_distribution 0 0 0 +
  gender_distribution 1 * age_distribution 1 2 * profession_distribution 1 2 2 +
  gender_distribution 0 * age_distribution 0 1 * profession_distribution 0 1 1 +
  gender_distribution 1 * age_distribution 1 1 * profession_distribution 1 1 1 = 0.22025 := by
  sorry

end NUMINAMATH_CALUDE_study_group_probability_l2459_245945


namespace NUMINAMATH_CALUDE_situps_total_l2459_245912

/-- The number of sit-ups Barney can do in one minute -/
def barney_situps : ℕ := 45

/-- The number of sit-ups Carrie can do in one minute -/
def carrie_situps : ℕ := 2 * barney_situps

/-- The number of sit-ups Jerrie can do in one minute -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie does sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three people -/
def total_situps : ℕ := 
  barney_situps * barney_minutes + 
  carrie_situps * carrie_minutes + 
  jerrie_situps * jerrie_minutes

theorem situps_total : total_situps = 510 := by
  sorry

end NUMINAMATH_CALUDE_situps_total_l2459_245912


namespace NUMINAMATH_CALUDE_evenOnesTableCountTheorem_l2459_245946

/-- The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column -/
def evenOnesTableCount (m n : ℕ) : ℕ :=
  2^((m-1)*(n-1))

/-- Theorem: The number of ways to fill an m × n table with zeros and ones,
    such that there is an even number of ones in every row and every column,
    is equal to 2^((m-1)(n-1)) -/
theorem evenOnesTableCountTheorem (m n : ℕ) :
  evenOnesTableCount m n = 2^((m-1)*(n-1)) := by
  sorry


end NUMINAMATH_CALUDE_evenOnesTableCountTheorem_l2459_245946


namespace NUMINAMATH_CALUDE_bobby_pancakes_l2459_245993

theorem bobby_pancakes (total : ℕ) (dog_ate : ℕ) (left : ℕ) (bobby_ate : ℕ) : 
  total = 21 → dog_ate = 7 → left = 9 → bobby_ate = total - dog_ate - left → bobby_ate = 5 := by
  sorry

end NUMINAMATH_CALUDE_bobby_pancakes_l2459_245993


namespace NUMINAMATH_CALUDE_unique_value_2n_plus_m_l2459_245914

theorem unique_value_2n_plus_m :
  ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = 36) :=
by sorry

end NUMINAMATH_CALUDE_unique_value_2n_plus_m_l2459_245914


namespace NUMINAMATH_CALUDE_car_ordering_theorem_l2459_245917

/-- Represents a car with its speeds in different road segments -/
structure Car where
  citySpeed : ℝ
  nonCitySpeed : ℝ

/-- Represents a point on the road -/
structure RoadPoint where
  cityDistance : ℝ
  nonCityDistance : ℝ

/-- The theorem statement -/
theorem car_ordering_theorem 
  (cars : Fin 10 → Car) 
  (points : Fin 2011 → RoadPoint) :
  ∃ i j, i ≠ j ∧ 
    (∀ (c₁ c₂ : Fin 10), 
      (cars c₁).citySpeed / (cars c₂).citySpeed < (cars c₁).nonCitySpeed / (cars c₂).nonCitySpeed →
      ((points i).cityDistance / (cars c₁).citySpeed + (points i).nonCityDistance / (cars c₁).nonCitySpeed <
       (points i).cityDistance / (cars c₂).citySpeed + (points i).nonCityDistance / (cars c₂).nonCitySpeed) ↔
      ((points j).cityDistance / (cars c₁).citySpeed + (points j).nonCityDistance / (cars c₁).nonCitySpeed <
       (points j).cityDistance / (cars c₂).citySpeed + (points j).nonCityDistance / (cars c₂).nonCitySpeed)) :=
by sorry

end NUMINAMATH_CALUDE_car_ordering_theorem_l2459_245917


namespace NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l2459_245939

/-- The number of trailing zeros in n! --/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 310! is 76 --/
theorem trailing_zeros_310_factorial :
  trailingZeros 310 = 76 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_310_factorial_l2459_245939


namespace NUMINAMATH_CALUDE_horners_method_for_f_l2459_245960

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem horners_method_for_f :
  f 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_horners_method_for_f_l2459_245960


namespace NUMINAMATH_CALUDE_thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2459_245977

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sum_decimal_rep (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- Theorem: The 35th digit after the decimal point of the sum of 1/9 and 1/5 is 3 -/
theorem thirty_fifth_digit_of_sum_one_ninth_one_fifth : 
  sum_decimal_rep (1/9) (1/5) 35 = 3 := by sorry

end NUMINAMATH_CALUDE_thirty_fifth_digit_of_sum_one_ninth_one_fifth_l2459_245977


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l2459_245934

theorem quadratic_inequality_integer_solution (z : ℕ) :
  z^2 - 50*z + 550 ≤ 10 ↔ 20 ≤ z ∧ z ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l2459_245934


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2459_245994

theorem unique_solution_quadratic (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) ↔ (j = 0 ∨ j = -36) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2459_245994


namespace NUMINAMATH_CALUDE_odd_function_sum_l2459_245906

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_sum (f : ℝ → ℝ) (h_odd : is_odd f) (h_pos : ∀ x > 0, f x = 2*x - 3) :
  f (-2) + f 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2459_245906


namespace NUMINAMATH_CALUDE_square_area_on_parabola_l2459_245969

theorem square_area_on_parabola : ∃ (x₁ x₂ : ℝ),
  (x₁^2 + 4*x₁ + 3 = 7) ∧
  (x₂^2 + 4*x₂ + 3 = 7) ∧
  (x₂ > x₁) ∧
  ((x₂ - x₁)^2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_l2459_245969


namespace NUMINAMATH_CALUDE_clara_final_stickers_l2459_245940

/-- Calculates the number of stickers Clara has left after a series of operations --/
def clara_stickers : ℕ :=
  let initial := 100
  let after_boy := initial - 10
  let after_teacher := after_boy + 50
  let after_classmates := after_teacher - (after_teacher * 20 / 100)
  let exchange_amount := after_classmates / 3
  let after_exchange := after_classmates - exchange_amount + (2 * exchange_amount)
  let give_to_friends := after_exchange / 4
  let remaining := after_exchange - (give_to_friends / 3 * 3)
  remaining

/-- Theorem stating that Clara ends up with 114 stickers --/
theorem clara_final_stickers : clara_stickers = 114 := by
  sorry


end NUMINAMATH_CALUDE_clara_final_stickers_l2459_245940


namespace NUMINAMATH_CALUDE_negative_one_half_less_than_negative_one_third_l2459_245904

theorem negative_one_half_less_than_negative_one_third :
  -1/2 < -1/3 := by sorry

end NUMINAMATH_CALUDE_negative_one_half_less_than_negative_one_third_l2459_245904


namespace NUMINAMATH_CALUDE_triangle_theorem_l2459_245916

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition for the triangle -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.a * Real.sin t.A = (2 * t.b + t.c) * Real.sin t.B + (2 * t.c + t.b) * Real.sin t.C

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2 * Real.sqrt 3) : 
  t.A = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ Real.sqrt 3) ∧
  (∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l2459_245916


namespace NUMINAMATH_CALUDE_dilation_matrix_determinant_l2459_245936

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 2) (Fin 2) ℝ),
  (∀ (i j : Fin 2), E i j = if i = j then 9 else 0) →
  Matrix.det E = 81 := by
sorry

end NUMINAMATH_CALUDE_dilation_matrix_determinant_l2459_245936


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l2459_245948

theorem simplify_algebraic_expression (a : ℝ) : 
  3*a + 6*a + 9*a + 6 + 12*a + 15 + 18*a = 48*a + 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l2459_245948


namespace NUMINAMATH_CALUDE_initial_insurance_premium_l2459_245923

/-- Proves that the initial insurance premium is $50 given the specified conditions --/
theorem initial_insurance_premium (P : ℝ) : 
  (1.1 * P + 3 * 5 = 70) → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_insurance_premium_l2459_245923


namespace NUMINAMATH_CALUDE_number_of_pupils_l2459_245915

/-- Represents the number of pupils in the class -/
def n : ℕ := sorry

/-- The correct first mark -/
def correct_first_mark : ℕ := 63

/-- The incorrect first mark -/
def incorrect_first_mark : ℕ := 83

/-- The correct second mark -/
def correct_second_mark : ℕ := 85

/-- The incorrect second mark -/
def incorrect_second_mark : ℕ := 75

/-- The weight for the first mark -/
def weight_first : ℕ := 3

/-- The weight for the second mark -/
def weight_second : ℕ := 2

/-- The increase in average marks due to the errors -/
def average_increase : ℚ := 1/2

theorem number_of_pupils : n = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l2459_245915


namespace NUMINAMATH_CALUDE_unique_x_value_l2459_245970

theorem unique_x_value : ∃! x : ℕ, 
  (∃ k : ℕ, x = 9 * k) ∧ 
  (x^2 > 120) ∧ 
  (x < 25) ∧ 
  (x = 18) := by
sorry

end NUMINAMATH_CALUDE_unique_x_value_l2459_245970


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l2459_245985

/-- The number of ways to award medals in the Olympic 100-meter finals --/
def medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medal := Nat.descFactorial non_american_sprinters medals
  let one_american_medal := american_sprinters * medals * (Nat.descFactorial non_american_sprinters (medals - 1))
  no_american_medal + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario --/
theorem olympic_medal_awards : 
  medal_award_ways 10 4 3 = 480 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l2459_245985


namespace NUMINAMATH_CALUDE_triangle_base_length_l2459_245952

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) :
  area = (base * height) / 2 → area = 12 → height = 6 → base = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2459_245952


namespace NUMINAMATH_CALUDE_fruits_in_red_basket_l2459_245919

theorem fruits_in_red_basket :
  let blue_bananas : ℕ := 12
  let blue_apples : ℕ := 4
  let blue_total : ℕ := blue_bananas + blue_apples
  let red_total : ℕ := blue_total / 2
  red_total = 8 := by sorry

end NUMINAMATH_CALUDE_fruits_in_red_basket_l2459_245919


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2459_245954

/-- Given a hyperbola with center at the origin, foci on the y-axis,
    and an asymptote passing through (-2, 4), its eccentricity is √5/2 -/
theorem hyperbola_eccentricity :
  ∀ (a b c : ℝ),
  (∀ x y : ℝ, y = a/b * x → (x = -2 ∧ y = 4)) →  -- asymptote passes through (-2, 4)
  a^2 = c^2 - b^2 →                              -- hyperbola equation
  c^2 / a^2 = (5:ℝ)/4 :=                         -- eccentricity squared
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2459_245954


namespace NUMINAMATH_CALUDE_smallest_c_value_l2459_245938

theorem smallest_c_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * (-π/4) + c)) →
  c ≥ π/4 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2459_245938


namespace NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2459_245989

/-- Calculates the total calories in Jerry's breakfast. -/
theorem jerrys_breakfast_calories :
  let pancake_calories : ℕ := 7 * 120
  let bacon_calories : ℕ := 3 * 100
  let orange_juice_calories : ℕ := 2 * 300
  let cereal_calories : ℕ := 200
  let muffin_calories : ℕ := 350
  pancake_calories + bacon_calories + orange_juice_calories + cereal_calories + muffin_calories = 2290 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_calories_l2459_245989


namespace NUMINAMATH_CALUDE_milk_production_calculation_l2459_245980

/-- Calculates the total milk production for a herd of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * num_days

/-- Theorem stating the total milk production for 120 cows over 15 days -/
theorem milk_production_calculation :
  total_milk_production 120 1362 15 = 2451600 := by
  sorry

#eval total_milk_production 120 1362 15

end NUMINAMATH_CALUDE_milk_production_calculation_l2459_245980


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_max_product_l2459_245931

/-- A cyclic quadrilateral with sides a, b, c, d inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  R : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ R > 0

/-- The product of sums of opposite sides pairs -/
def sideProduct (q : CyclicQuadrilateral) : ℝ :=
  (q.a * q.b + q.c * q.d) * (q.a * q.c + q.b * q.d) * (q.a * q.d + q.b * q.c)

/-- Predicate to check if a cyclic quadrilateral is a square -/
def isSquare (q : CyclicQuadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

theorem cyclic_quadrilateral_max_product (q : CyclicQuadrilateral) :
  ∀ q' : CyclicQuadrilateral, q'.R = q.R → sideProduct q ≤ sideProduct q' ↔ isSquare q' :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_max_product_l2459_245931


namespace NUMINAMATH_CALUDE_vector_problem_l2459_245958

def a : Fin 2 → ℝ := ![- 3, 1]
def b : Fin 2 → ℝ := ![1, -2]
def c : Fin 2 → ℝ := ![1, -1]

def m (k : ℝ) : Fin 2 → ℝ := fun i ↦ a i + k * b i

theorem vector_problem :
  (∃ k : ℝ, (∀ i : Fin 2, m k i * (2 * a i - b i) = 0) ∧ k = 5 / 3) ∧
  (∃ k : ℝ, (∀ i : Fin 2, ∃ t : ℝ, m k i = t * (k * b i + c i)) ∧ k = -1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2459_245958


namespace NUMINAMATH_CALUDE_inequality_solution_l2459_245927

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 (13/2) ∪ {13/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2459_245927


namespace NUMINAMATH_CALUDE_line_through_point_l2459_245949

/-- Given a line ax - y - 1 = 0 passing through the point (1, 3), prove that a = 4 -/
theorem line_through_point (a : ℝ) : (a * 1 - 3 - 1 = 0) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2459_245949


namespace NUMINAMATH_CALUDE_reciprocal_problem_l2459_245953

theorem reciprocal_problem (x : ℚ) : 8 * x = 16 → 200 * (1 / x) = 100 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l2459_245953


namespace NUMINAMATH_CALUDE_ratio_problem_l2459_245928

theorem ratio_problem (x y z w : ℚ) 
  (h1 : x / y = 24)
  (h2 : z / y = 8)
  (h3 : z / w = 1 / 12) :
  x / w = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2459_245928


namespace NUMINAMATH_CALUDE_max_surface_area_parallelepiped_l2459_245961

/-- The maximum surface area of a rectangular parallelepiped with diagonal length 3 is 18 -/
theorem max_surface_area_parallelepiped (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 9 →
  2 * (a * b + b * c + c * a) ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_parallelepiped_l2459_245961


namespace NUMINAMATH_CALUDE_average_weight_of_children_l2459_245920

def ages : List ℝ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

def regression_equation (x : ℝ) : ℝ := 2 * x + 7

theorem average_weight_of_children :
  let avg_age := (ages.sum) / (ages.length : ℝ)
  regression_equation avg_age = 15 := by sorry

end NUMINAMATH_CALUDE_average_weight_of_children_l2459_245920


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2459_245988

/-- Theorem: Ratio of inscribed circle area to square area is π/4 -/
theorem inscribed_circle_area_ratio (a b : ℤ) (h : b ≠ 0) :
  let r : ℚ := a / b
  let circle_area := π * r^2
  let square_side := 2 * r
  let square_area := square_side^2
  circle_area / square_area = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2459_245988


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2459_245999

/-- Given a geometric sequence {aₙ} where a₁ and a₁₃ are the roots of x² - 8x + 1 = 0,
    the product a₅ · a₇ · a₉ equals 1. -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  (a 1)^2 - 8*(a 1) + 1 = 0 →           -- a₁ is a root
  (a 13)^2 - 8*(a 13) + 1 = 0 →         -- a₁₃ is a root
  a 5 * a 7 * a 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2459_245999


namespace NUMINAMATH_CALUDE_fixed_distance_theorem_l2459_245901

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_fixed_distance (p a b : V) : Prop :=
  ∃ (k : ℝ), ∀ (q : V), ‖p - q‖ = k → q = (4/3 : ℝ) • a - (1/3 : ℝ) • b

theorem fixed_distance_theorem (a b p : V) 
  (h : ‖p - b‖ = 2 * ‖p - a‖) : is_fixed_distance p a b := by
  sorry

end NUMINAMATH_CALUDE_fixed_distance_theorem_l2459_245901


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2459_245963

theorem sine_cosine_inequality (x : ℝ) (n : ℕ+) :
  (Real.sin (2 * x))^(n : ℕ) + ((Real.sin x)^(n : ℕ) - (Real.cos x)^(n : ℕ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2459_245963


namespace NUMINAMATH_CALUDE_milk_consumption_ratio_l2459_245910

/-- The ratio of Minyoung's milk consumption to Yuna's milk consumption -/
theorem milk_consumption_ratio (minyoung_milk yuna_milk : ℚ) 
  (h1 : minyoung_milk = 10)
  (h2 : yuna_milk = 2/3) :
  minyoung_milk / yuna_milk = 15 := by
sorry

end NUMINAMATH_CALUDE_milk_consumption_ratio_l2459_245910


namespace NUMINAMATH_CALUDE_num_installments_is_40_l2459_245921

/-- Proves that the number of installments is 40 given the payment conditions --/
theorem num_installments_is_40 
  (n : ℕ) -- Total number of installments
  (h1 : n ≥ 20) -- At least 20 installments
  (first_20_payment : ℕ := 410) -- First 20 payments
  (remaining_payment : ℕ := 475) -- Remaining payments
  (average_payment : ℚ := 442.5) -- Average payment
  (h2 : (20 * first_20_payment + (n - 20) * remaining_payment : ℚ) / n = average_payment) -- Average payment equation
  : n = 40 := by
  sorry

end NUMINAMATH_CALUDE_num_installments_is_40_l2459_245921


namespace NUMINAMATH_CALUDE_complement_determines_set_l2459_245968

def U : Set Nat := {0, 1, 2, 4}

theorem complement_determines_set 
  (h : Set.compl {1, 2} = {0, 4}) : 
  ∃ A : Set Nat, A ⊆ U ∧ Set.compl A = {1, 2} ∧ A = {0, 4} := by
  sorry

#check complement_determines_set

end NUMINAMATH_CALUDE_complement_determines_set_l2459_245968


namespace NUMINAMATH_CALUDE_lattice_point_probability_l2459_245956

theorem lattice_point_probability (d : ℝ) : 
  (d > 0) → 
  (π * d^2 = 3/4) → 
  (d = Real.sqrt (3 / (4 * π))) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_probability_l2459_245956


namespace NUMINAMATH_CALUDE_triangle_theorem_l2459_245929

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sine_rule : a / Real.sin A = b / Real.sin B
  angle_sum : A + B + C = π

/-- The theorem to be proved -/
theorem triangle_theorem (t : AcuteTriangle) 
  (h1 : 2 * t.a * Real.sin t.B = Real.sqrt 3 * t.b)
  (h2 : t.a = 6)
  (h3 : t.b + t.c = 8) :
  t.A = π/3 ∧ 
  1/2 * t.b * t.c * Real.sin t.A = 7 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l2459_245929


namespace NUMINAMATH_CALUDE_curve_point_coordinates_l2459_245943

theorem curve_point_coordinates (θ : Real) (x y : Real) :
  0 ≤ θ ∧ θ ≤ π →
  x = 3 * Real.cos θ →
  y = 4 * Real.sin θ →
  y = x →
  x = 12/5 ∧ y = 12/5 := by
sorry

end NUMINAMATH_CALUDE_curve_point_coordinates_l2459_245943


namespace NUMINAMATH_CALUDE_interior_edges_sum_for_specific_frame_l2459_245987

/-- Represents a rectangular picture frame -/
structure Frame where
  outerLength : ℝ
  outerWidth : ℝ
  frameWidth : ℝ

/-- Calculates the area of the frame -/
def frameArea (f : Frame) : ℝ :=
  f.outerLength * f.outerWidth - (f.outerLength - 2 * f.frameWidth) * (f.outerWidth - 2 * f.frameWidth)

/-- Calculates the sum of the lengths of the four interior edges -/
def interiorEdgesSum (f : Frame) : ℝ :=
  2 * (f.outerLength - 2 * f.frameWidth) + 2 * (f.outerWidth - 2 * f.frameWidth)

/-- Theorem stating the sum of interior edges for a specific frame -/
theorem interior_edges_sum_for_specific_frame :
  ∃ (f : Frame),
    f.outerLength = 7 ∧
    f.frameWidth = 2 ∧
    frameArea f = 30 ∧
    interiorEdgesSum f = 7 := by
  sorry

end NUMINAMATH_CALUDE_interior_edges_sum_for_specific_frame_l2459_245987


namespace NUMINAMATH_CALUDE_cosine_angle_vectors_l2459_245932

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_angle_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : 2 * ‖a‖ = 3 * ‖b‖) (h2 : ‖a - 2•b‖ = ‖a + b‖) :
  inner a b / (‖a‖ * ‖b‖) = 1/3 := by sorry

end NUMINAMATH_CALUDE_cosine_angle_vectors_l2459_245932


namespace NUMINAMATH_CALUDE_shirt_problem_l2459_245908

/-- Given the prices of sarees and shirts, prove the number of shirts that can be bought for $2400 -/
theorem shirt_problem (S T : ℚ) (h1 : 2 * S + 4 * T = 1600) (h2 : S + 6 * T = 1600) :
  ∃ X : ℚ, X * T = 2400 ∧ X = 12 := by
  sorry

end NUMINAMATH_CALUDE_shirt_problem_l2459_245908


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2459_245962

-- Define the sets P and Q
def P : Set ℝ := {x | 2*x^2 - 3*x + 1 ≤ 0}
def Q (a : ℝ) : Set ℝ := {x | (x-a)*(x-a-1) ≤ 0}

-- Theorem 1: P ∩ Q = {1} when a = 1
theorem intersection_when_a_is_one : P ∩ (Q 1) = {1} := by sorry

-- Theorem 2: P ⊆ Q if and only if 0 ≤ a ≤ 1/2
theorem subset_condition (a : ℝ) : P ⊆ Q a ↔ 0 ≤ a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l2459_245962


namespace NUMINAMATH_CALUDE_college_students_count_l2459_245933

theorem college_students_count : ℕ :=
  let students_to_professors_ratio : ℕ := 15
  let total_people : ℕ := 40000
  let students : ℕ := 37500

  have h1 : students = students_to_professors_ratio * (total_people - students) := by sorry
  have h2 : students + (total_people - students) = total_people := by sorry

  students

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_college_students_count_l2459_245933


namespace NUMINAMATH_CALUDE_function_value_at_2012_l2459_245930

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx + β) where f(2001) = 3, 
    prove that f(2012) = -3 -/
theorem function_value_at_2012 
  (a b α β : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β))
  (h2 : f 2001 = 3) :
  f 2012 = -3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_2012_l2459_245930


namespace NUMINAMATH_CALUDE_next_adjacent_natural_number_l2459_245986

theorem next_adjacent_natural_number (n a : ℕ) (h : n = a^2) : 
  n + 1 = a^2 + 1 := by sorry

end NUMINAMATH_CALUDE_next_adjacent_natural_number_l2459_245986


namespace NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l2459_245971

def sum_of_consecutive_integers (n : ℕ) (count : ℕ) : ℕ :=
  count * (2 * n + count - 1) / 2

theorem sum_of_150_consecutive_integers :
  ∃ (n : ℕ), sum_of_consecutive_integers n 150 = 1725225 ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 3410775) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 2245600) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 1257925) ∧
  (∀ (m : ℕ), sum_of_consecutive_integers m 150 ≠ 4146950) :=
by
  sorry

#check sum_of_150_consecutive_integers

end NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l2459_245971


namespace NUMINAMATH_CALUDE_binomial_expectation_five_l2459_245978

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- Probability mass function for a binomial distribution -/
def pmf (ξ : BinomialRV) (k : ℕ) : ℝ :=
  (ξ.n.choose k) * (ξ.p ^ k) * ((1 - ξ.p) ^ (ξ.n - k))

/-- Expected value of a binomial distribution -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

theorem binomial_expectation_five (ξ : BinomialRV) 
    (h_p : ξ.p = 1/2) 
    (h_pmf : pmf ξ 2 = 45 / 2^10) : 
  expectation ξ = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_five_l2459_245978


namespace NUMINAMATH_CALUDE_jade_tower_levels_l2459_245944

/-- Calculates the number of complete levels in a Lego tower. -/
def towerLevels (totalPieces piecesPerLevel unusedPieces : ℕ) : ℕ :=
  (totalPieces - unusedPieces) / piecesPerLevel

/-- Proves that given the specific conditions, the tower has 11 levels. -/
theorem jade_tower_levels :
  towerLevels 100 7 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jade_tower_levels_l2459_245944


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2459_245972

/-- Proves that the actual distance traveled is 100 km given the conditions of the problem -/
theorem actual_distance_traveled (speed_slow speed_fast distance_diff : ℝ) 
  (h1 : speed_slow = 10)
  (h2 : speed_fast = 12)
  (h3 : distance_diff = 20)
  (h4 : speed_slow > 0)
  (h5 : speed_fast > speed_slow) :
  ∃ (actual_distance : ℝ),
    actual_distance / speed_slow = (actual_distance + distance_diff) / speed_fast ∧
    actual_distance = 100 :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2459_245972


namespace NUMINAMATH_CALUDE_east_to_north_ratio_l2459_245903

/-- Represents the number of tents in different areas of the campsite -/
structure Campsite where
  total : ℕ
  north : ℕ
  center : ℕ
  south : ℕ
  east : ℕ

/-- The conditions of the campsite as described in the problem -/
def campsite_conditions (c : Campsite) : Prop :=
  c.total = 900 ∧
  c.north = 100 ∧
  c.center = 4 * c.north ∧
  c.south = 200 ∧
  c.total = c.north + c.center + c.south + c.east

/-- The theorem stating the ratio of tents on the east side to the northernmost part -/
theorem east_to_north_ratio (c : Campsite) 
  (h : campsite_conditions c) : c.east = 2 * c.north :=
sorry

end NUMINAMATH_CALUDE_east_to_north_ratio_l2459_245903


namespace NUMINAMATH_CALUDE_lunchroom_total_people_l2459_245942

def num_tables : ℕ := 34
def first_table_students : ℕ := 6
def teacher_count : ℕ := 5

def arithmetic_sum (n : ℕ) (a1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem lunchroom_total_people :
  arithmetic_sum num_tables first_table_students 1 + teacher_count = 770 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_total_people_l2459_245942


namespace NUMINAMATH_CALUDE_inheritance_calculation_l2459_245983

theorem inheritance_calculation (x : ℝ) : 
  (0.2 * x + 0.1 * (0.8 * x) = 10500) → x = 37500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l2459_245983


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2459_245967

theorem triangle_perimeter (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive side lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  ((a - 6) * (a - 3) = 0 ∨ (b - 6) * (b - 3) = 0 ∨ (c - 6) * (c - 3) = 0) →  -- At least one side satisfies the equation
  (a + b + c = 9 ∨ a + b + c = 15 ∨ a + b + c = 18) :=
by sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2459_245967


namespace NUMINAMATH_CALUDE_football_lineup_count_l2459_245996

/-- The number of ways to choose a starting lineup from a football team. -/
def starting_lineup_count (total_players : ℕ) (offensive_linemen : ℕ) (lineup_size : ℕ) (linemen_in_lineup : ℕ) : ℕ :=
  (Nat.choose offensive_linemen linemen_in_lineup) *
  (Nat.choose (total_players - linemen_in_lineup) (lineup_size - linemen_in_lineup)) *
  (Nat.factorial (lineup_size - linemen_in_lineup))

/-- Theorem stating the number of ways to choose the starting lineup. -/
theorem football_lineup_count :
  starting_lineup_count 15 5 5 2 = 17160 := by
  sorry

end NUMINAMATH_CALUDE_football_lineup_count_l2459_245996


namespace NUMINAMATH_CALUDE_petyas_friends_l2459_245997

theorem petyas_friends (total_stickers : ℕ) : 
  (∃ (x : ℕ), 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) → 
  (∃ (x : ℕ), x = 19 ∧ 5 * x + 8 = total_stickers ∧ 6 * x = total_stickers + 11) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l2459_245997


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2459_245981

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ - 3 = 0) ∧ 
  (x₂^2 + m*x₂ - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2459_245981


namespace NUMINAMATH_CALUDE_smallest_number_with_ten_even_five_or_seven_l2459_245913

def containsDigit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def isEvenWithFiveOrSeven (n : ℕ) : Prop :=
  n % 2 = 0 ∧ (containsDigit n 5 ∨ containsDigit n 7)

theorem smallest_number_with_ten_even_five_or_seven : 
  (∃! m : ℕ, m > 0 ∧ (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S))) ∧
  (∀ m : ℕ, m > 0 → (∃ S : Finset ℕ, Finset.card S = 10 ∧ 
    (∀ n ∈ S, n < m ∧ isEvenWithFiveOrSeven n) ∧
    (∀ n : ℕ, n < m → isEvenWithFiveOrSeven n → n ∈ S)) → m ≥ 160) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_ten_even_five_or_seven_l2459_245913


namespace NUMINAMATH_CALUDE_oliver_final_amount_l2459_245974

-- Define the currencies
structure Currency where
  usd : ℚ
  quarters : ℚ
  dimes : ℚ
  eur : ℚ
  gbp : ℚ
  chf : ℚ
  jpy : ℚ
  cad : ℚ
  aud : ℚ

-- Define the exchange rates
structure ExchangeRates where
  usd_to_gbp : ℚ
  eur_to_gbp : ℚ
  usd_to_chf : ℚ
  eur_to_chf : ℚ
  jpy_to_cad : ℚ
  eur_to_aud : ℚ

-- Define the initial amounts and exchanges
def initial_amount : Currency := {
  usd := 40,
  quarters := 200,
  dimes := 100,
  eur := 15,
  gbp := 0,
  chf := 0,
  jpy := 3000,
  cad := 0,
  aud := 0
}

def exchange_rates : ExchangeRates := {
  usd_to_gbp := 3/4,
  eur_to_gbp := 17/20,
  usd_to_chf := 9/10,
  eur_to_chf := 21/20,
  jpy_to_cad := 3/250,
  eur_to_aud := 3/2
}

def exchanged_amount : Currency := {
  usd := 10,
  quarters := 0,
  dimes := 0,
  eur := 13,
  gbp := 0,
  chf := 0,
  jpy := 2000,
  cad := 0,
  aud := 0
}

def given_to_sister : Currency := {
  usd := 5,
  quarters := 120,
  dimes := 50,
  eur := 0,
  gbp := 7/2,
  chf := 2,
  jpy := 500,
  cad := 0,
  aud := 7
}

-- Theorem to prove
theorem oliver_final_amount (initial : Currency) (rates : ExchangeRates) 
  (exchanged : Currency) (given : Currency) : 
  ∃ (final : Currency),
    final.usd = 20 ∧
    final.quarters = 0 ∧
    final.dimes = 0 ∧
    final.eur = 2 ∧
    final.gbp = 33/4 ∧
    final.chf = 49/4 ∧
    final.jpy = 0 ∧
    final.cad = 24 ∧
    final.aud = 5 :=
by sorry

end NUMINAMATH_CALUDE_oliver_final_amount_l2459_245974
