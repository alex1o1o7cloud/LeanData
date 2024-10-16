import Mathlib

namespace NUMINAMATH_CALUDE_congruence_problem_l46_4648

theorem congruence_problem (x : ℤ) :
  (4 * x + 5) % 20 = 3 → (3 * x + 8) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l46_4648


namespace NUMINAMATH_CALUDE_y_completion_time_l46_4651

/-- The time Y takes to complete the entire work alone, given:
  * X can do the entire work in 40 days
  * X works for 8 days
  * Y finishes the remaining work in 20 days
-/
theorem y_completion_time (x_total_days : ℕ) (x_worked_days : ℕ) (y_completion_days : ℕ) :
  x_total_days = 40 →
  x_worked_days = 8 →
  y_completion_days = 20 →
  (x_worked_days : ℚ) / x_total_days + (y_completion_days : ℚ) * (1 - (x_worked_days : ℚ) / x_total_days) = 1 →
  25 = (1 / (1 / y_completion_days * (1 - (x_worked_days : ℚ) / x_total_days))) := by
  sorry

end NUMINAMATH_CALUDE_y_completion_time_l46_4651


namespace NUMINAMATH_CALUDE_distance_between_trees_l46_4638

/-- Given a yard of length 400 meters with 26 trees planted at equal distances,
    including one tree at each end, the distance between consecutive trees is 16 meters. -/
theorem distance_between_trees (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 400 ∧ num_trees = 26 →
  (yard_length / (num_trees - 1 : ℝ)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l46_4638


namespace NUMINAMATH_CALUDE_quadratic_always_two_roots_l46_4649

theorem quadratic_always_two_roots (k : ℝ) : 
  let a := (1 : ℝ)
  let b := 2 * k
  let c := k - 1
  let discriminant := b^2 - 4*a*c
  0 < discriminant := by sorry

end NUMINAMATH_CALUDE_quadratic_always_two_roots_l46_4649


namespace NUMINAMATH_CALUDE_max_distance_product_l46_4685

/-- Triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_eq : a = 3
  b_eq : b = 4
  c_eq : c = 5

/-- Point inside the triangle -/
structure InteriorPoint (t : RightTriangle) where
  x : ℝ
  y : ℝ
  interior : 0 < x ∧ 0 < y ∧ x + y < 1

/-- Distances from a point to the sides of the triangle -/
def distances (t : RightTriangle) (p : InteriorPoint t) : ℝ × ℝ × ℝ :=
  (p.x, p.y, 1 - p.x - p.y)

/-- Product of distances from a point to the sides of the triangle -/
def distanceProduct (t : RightTriangle) (p : InteriorPoint t) : ℝ :=
  let (d₁, d₂, d₃) := distances t p
  d₁ * d₂ * d₃

theorem max_distance_product (t : RightTriangle) :
  ∀ p : InteriorPoint t, distanceProduct t p ≤ 1/125 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_product_l46_4685


namespace NUMINAMATH_CALUDE_sum_abcd_equals_1986_l46_4696

theorem sum_abcd_equals_1986 
  (h1 : 6 * a + 2 * b = 3848) 
  (h2 : 6 * c + 3 * d = 4410) 
  (h3 : a + 3 * b + 2 * d = 3080) : 
  a + b + c + d = 1986 := by
  sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_1986_l46_4696


namespace NUMINAMATH_CALUDE_bouquet_combinations_l46_4694

theorem bouquet_combinations (total : ℕ) (rose_cost : ℕ) (carnation_cost : ℕ) 
  (h_total : total = 50)
  (h_rose : rose_cost = 3)
  (h_carnation : carnation_cost = 2) :
  (∃ (solutions : Finset (ℕ × ℕ)), 
    solutions.card = 9 ∧ 
    ∀ (r c : ℕ), (r, c) ∈ solutions ↔ rose_cost * r + carnation_cost * c = total) :=
sorry

end NUMINAMATH_CALUDE_bouquet_combinations_l46_4694


namespace NUMINAMATH_CALUDE_inequality_solution_set_l46_4630

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 5 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) ↔ 
  (x < -1 ∨ (2 < x ∧ x < 3) ∨ (5 < x ∧ x < 6)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l46_4630


namespace NUMINAMATH_CALUDE_chessboard_pythagorean_triple_exists_l46_4610

/-- Represents a point on an infinite chessboard --/
structure ChessboardPoint where
  x : Int
  y : Int

/-- Distance function between two ChessboardPoints --/
def distance (p q : ChessboardPoint) : Nat :=
  ((p.x - q.x)^2 + (p.y - q.y)^2).natAbs

/-- Predicate to check if three points are non-collinear --/
def nonCollinear (p q r : ChessboardPoint) : Prop :=
  (q.x - p.x) * (r.y - p.y) ≠ (r.x - p.x) * (q.y - p.y)

/-- Theorem stating the existence of points satisfying the given conditions --/
theorem chessboard_pythagorean_triple_exists : 
  ∃ (A B C : ChessboardPoint), 
    nonCollinear A B C ∧ 
    (distance A C)^2 + (distance B C)^2 = (distance A B)^2 := by
  sorry


end NUMINAMATH_CALUDE_chessboard_pythagorean_triple_exists_l46_4610


namespace NUMINAMATH_CALUDE_bank_layoff_optimization_l46_4645

/-- Represents the economic benefit function for the bank -/
def economic_benefit (x : ℕ) : ℚ :=
  (320 - x) * (20 + 0.2 * x) - 6 * x

/-- Represents the constraint on the number of employees that can be laid off -/
def valid_layoff (x : ℕ) : Prop :=
  x ≤ 80

theorem bank_layoff_optimization :
  ∃ (x : ℕ), valid_layoff x ∧
    (∀ (y : ℕ), valid_layoff y → economic_benefit x ≥ economic_benefit y) ∧
    economic_benefit x = 9160 :=
sorry

end NUMINAMATH_CALUDE_bank_layoff_optimization_l46_4645


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l46_4639

theorem system_of_equations_solutions :
  let solutions : List (ℂ × ℂ × ℂ) := [
    (0, 0, 0),
    (2/3, -1/3, -1/3),
    (1/3, 1/3, 1/3),
    (1, 0, 0),
    (2/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (2/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 + Complex.I * Real.sqrt 3) / 6, (1 - Complex.I * Real.sqrt 3) / 6),
    (1/3, (1 - Complex.I * Real.sqrt 3) / 6, (1 + Complex.I * Real.sqrt 3) / 6)
  ]
  ∀ x y z : ℂ,
    (x^2 + 2*y*z = x ∧ y^2 + 2*z*x = z ∧ z^2 + 2*x*y = y) ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l46_4639


namespace NUMINAMATH_CALUDE_geometric_series_comparison_l46_4600

theorem geometric_series_comparison (a₁ : ℚ) (r₁ r₂ : ℚ) :
  a₁ = 5/12 →
  r₁ = 3/4 →
  r₂ < 1 →
  r₂ > 0 →
  a₁ / (1 - r₁) > a₁ / (1 - r₂) →
  r₂ = 5/6 ∧ a₁ / (1 - r₁) = 5/3 ∧ a₁ / (1 - r₂) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_comparison_l46_4600


namespace NUMINAMATH_CALUDE_questionnaires_from_unit_d_l46_4609

/-- Represents the number of questionnaires drawn from each unit -/
structure SampledQuestionnaires :=
  (a b c d : ℕ)

/-- Represents the total number of questionnaires collected from each unit -/
structure TotalQuestionnaires :=
  (a b c d : ℕ)

/-- The properties of the survey and sampling -/
class SurveyProperties (sampled : SampledQuestionnaires) (total : TotalQuestionnaires) :=
  (total_sum : total.a + total.b + total.c + total.d = 1000)
  (sample_sum : sampled.a + sampled.b + sampled.c + sampled.d = 150)
  (total_arithmetic : ∃ (r : ℕ), total.b = total.a + r ∧ total.c = total.b + r ∧ total.d = total.c + r)
  (sample_arithmetic : ∃ (d : ℤ), sampled.b = sampled.a + d ∧ sampled.c = sampled.b + d ∧ sampled.d = sampled.c + d)
  (unit_b_sample : sampled.b = 30)
  (stratified_sampling : ∀ (x y : Fin 4), 
    (sampled.a * total.b = sampled.b * total.a) ∧
    (sampled.b * total.c = sampled.c * total.b) ∧
    (sampled.c * total.d = sampled.d * total.c))

/-- The main theorem to prove -/
theorem questionnaires_from_unit_d 
  (sampled : SampledQuestionnaires) 
  (total : TotalQuestionnaires) 
  [SurveyProperties sampled total] : 
  sampled.d = 60 := by
  sorry

end NUMINAMATH_CALUDE_questionnaires_from_unit_d_l46_4609


namespace NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l46_4661

/-- An ellipse centered at the origin with axes aligned with the coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem ellipse_equation_from_conditions 
  (e : Ellipse)
  (h_major_twice_minor : e.a = 2 * e.b)
  (h_point_on_ellipse : ellipse_equation e 4 1) :
  ∀ x y, ellipse_equation e x y ↔ x^2 / 20 + y^2 / 5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_from_conditions_l46_4661


namespace NUMINAMATH_CALUDE_inequality_proof_l46_4692

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l46_4692


namespace NUMINAMATH_CALUDE_weight_of_b_l46_4636

/-- Given the weights and heights of four people a, b, c, and d, prove that the weight of b is 1350/31 kg. -/
theorem weight_of_b (Wa Wb Wc Wd Ha Hb Hc Hd : ℝ) : Wb = 1350/31 :=
  sorry
where
  /-- The average weight of a, b, and c is 45 kg -/
  avg_abc : Wa + Wb + Wc = 135 := by sorry

  /-- The average weight of a and b is 40 kg -/
  avg_ab : Wa + Wb = 80 := by sorry

  /-- The average weight of b and c is 43 kg -/
  avg_bc : Wb + Wc = 86 := by sorry

  /-- The average weight of b, c, and d is 47 kg -/
  avg_bcd : Wb + Wc + Wd = 141 := by sorry

  /-- The height proportion of a to b is 3:5 -/
  prop_ab : Wa / Wb = Ha / Hb := by sorry
  height_prop_ab : Ha / Hb = 3 / 5 := by sorry

  /-- The height proportion of b to c is 2:3 -/
  prop_bc : Wb / Wc = Hb / Hc := by sorry
  height_prop_bc : Hb / Hc = 2 / 3 := by sorry

  /-- The height proportion of b:c:d is 4:3:7 -/
  prop_bcd : Wb / Wc = Hb / Hc ∧ Wc / Wd = Hc / Hd := by sorry
  height_prop_bcd : Hb / Hc = 4 / 3 ∧ Hc / Hd = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_weight_of_b_l46_4636


namespace NUMINAMATH_CALUDE_exponent_equality_l46_4688

theorem exponent_equality : 
  (-3^3 = (-3)^3) ∧ 
  ((-2)^3 ≠ (-3)^2) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-3 * 2^3 ≠ (-3 * 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l46_4688


namespace NUMINAMATH_CALUDE_points_subtracted_per_wrong_answer_l46_4656

theorem points_subtracted_per_wrong_answer
  (total_problems : ℕ)
  (total_score : ℕ)
  (points_per_correct : ℕ)
  (wrong_answers : ℕ)
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : points_per_correct = 4)
  (h4 : wrong_answers = 3)
  : (total_problems * points_per_correct - total_score) / wrong_answers = 1 := by
  sorry

end NUMINAMATH_CALUDE_points_subtracted_per_wrong_answer_l46_4656


namespace NUMINAMATH_CALUDE_concentric_circles_theorem_l46_4643

/-- Two concentric circles with radii R and r, where R > r -/
structure ConcentricCircles (R r : ℝ) where
  radius_larger : R > r

/-- Point on a circle -/
structure PointOnCircle (center : ℝ × ℝ) (radius : ℝ) where
  point : ℝ × ℝ
  on_circle : (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

/-- Theorem about the sum of squared distances and the locus of midpoint -/
theorem concentric_circles_theorem
  (R r : ℝ) (h : ConcentricCircles R r)
  (O : ℝ × ℝ) -- Center of the circles
  (P : PointOnCircle O r) -- Fixed point on smaller circle
  (B : PointOnCircle O R) -- Moving point on larger circle
  (A : PointOnCircle O r) -- Point on smaller circle determined by perpendicular line from P to BP
  (C : PointOnCircle O R) -- Intersection of BP with larger circle
  : 
  -- Part 1: Sum of squared distances
  (B.point.1 - C.point.1)^2 + (B.point.2 - C.point.2)^2 +
  (C.point.1 - A.point.1)^2 + (C.point.2 - A.point.2)^2 +
  (A.point.1 - B.point.1)^2 + (A.point.2 - B.point.2)^2 = 6 * R^2 + 2 * r^2
  ∧
  -- Part 2: Locus of midpoint of AB
  ∃ (Q : ℝ × ℝ),
    Q = ((A.point.1 + B.point.1) / 2, (A.point.2 + B.point.2) / 2) ∧
    (Q.1 - (O.1 + P.point.1) / 2)^2 + (Q.2 - (O.2 + P.point.2) / 2)^2 = (R / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_theorem_l46_4643


namespace NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l46_4676

theorem opposite_signs_and_larger_absolute_value (a b : ℚ) 
  (h1 : a * b < 0) (h2 : a + b > 0) : 
  (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0) ∧ 
  ((a < 0 ∧ b > 0 → abs b > abs a) ∧ (a > 0 ∧ b < 0 → abs a > abs b)) :=
sorry

end NUMINAMATH_CALUDE_opposite_signs_and_larger_absolute_value_l46_4676


namespace NUMINAMATH_CALUDE_jen_profit_l46_4681

/-- Calculates the profit in cents for a candy bar business -/
def candy_bar_profit (buy_price sell_price bought_quantity sold_quantity : ℕ) : ℤ :=
  (sell_price * sold_quantity : ℤ) - (buy_price * bought_quantity : ℤ)

/-- Proves that Jen's profit from her candy bar business is 800 cents -/
theorem jen_profit : candy_bar_profit 80 100 50 48 = 800 := by
  sorry

end NUMINAMATH_CALUDE_jen_profit_l46_4681


namespace NUMINAMATH_CALUDE_sum_of_squares_perfect_square_l46_4629

theorem sum_of_squares_perfect_square (n p k : ℤ) :
  ∃ m : ℤ, n^2 + p^2 + k^2 = m^2 ↔ n * k = (p / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_perfect_square_l46_4629


namespace NUMINAMATH_CALUDE_x_to_y_equals_negative_eight_l46_4646

theorem x_to_y_equals_negative_eight (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_equals_negative_eight_l46_4646


namespace NUMINAMATH_CALUDE_marble_distribution_l46_4601

theorem marble_distribution (a : ℕ) : 
  let angela := a
  let brian := 2 * angela
  let caden := 3 * brian
  let daryl := 5 * caden
  angela + brian + caden + daryl = 78 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l46_4601


namespace NUMINAMATH_CALUDE_simplify_expression_l46_4698

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l46_4698


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l46_4642

theorem rectangle_longer_side (a : ℝ) (h1 : a > 0) : 
  (a * (0.8 * a) = 81 / 20) → a = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l46_4642


namespace NUMINAMATH_CALUDE_triangle_properties_l46_4669

/-- Given a triangle ABC with dot product conditions, prove the length of AB and a trigonometric ratio -/
theorem triangle_properties (A B C : ℝ × ℝ) 
  (h1 : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 9)
  (h2 : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = -16) :
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let cosA := ((B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2)) / (AB * CA)
  let cosB := ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / (AB * BC)
  let cosC := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (BC * CA)
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := Real.sqrt (1 - cosB^2)
  let sinC := Real.sqrt (1 - cosC^2)
  AB = 5 ∧ (sinA * cosB - cosA * sinB) / sinC = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l46_4669


namespace NUMINAMATH_CALUDE_factorization_equality_l46_4690

theorem factorization_equality (x y : ℝ) :
  -1/2 * x^3 + 1/8 * x * y^2 = -1/8 * x * (2*x + y) * (2*x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l46_4690


namespace NUMINAMATH_CALUDE_total_baseball_cards_l46_4679

/-- The number of people with baseball cards -/
def num_people : ℕ := 6

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 52

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_baseball_cards : total_cards = 312 := by
  sorry

end NUMINAMATH_CALUDE_total_baseball_cards_l46_4679


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l46_4634

/-- Given two people moving in opposite directions for 4 hours,
    with one person moving at 3 km/hr and ending up 40 km apart,
    prove that the speed of the other person is 7 km/hr. -/
theorem opposite_direction_speed
  (time : ℝ)
  (distance : ℝ)
  (speed_known : ℝ)
  (h1 : time = 4)
  (h2 : distance = 40)
  (h3 : speed_known = 3)
  (h4 : distance = (speed_known + speed_unknown) * time) :
  speed_unknown = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l46_4634


namespace NUMINAMATH_CALUDE_quadratic_properties_l46_4654

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

theorem quadratic_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, quadratic_function m x = 0 ↔ x = 1 ∨ x = 3) ∧
  (m < 0 → 
    (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ 4 ∧ quadratic_function m x₀ = 2 ∧
      ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≤ 2) →
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≥ -6) ∧
    (∃ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ quadratic_function m x₁ = -6)) ∧
  (m ≤ -4/3 ∨ m ≥ 4/5 ↔
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic_function m x = (m + 4) / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l46_4654


namespace NUMINAMATH_CALUDE_fred_onions_l46_4672

theorem fred_onions (sara_onions sally_onions total_onions : ℕ) 
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : total_onions = 18) :
  total_onions - (sara_onions + sally_onions) = 9 := by
sorry

end NUMINAMATH_CALUDE_fred_onions_l46_4672


namespace NUMINAMATH_CALUDE_puppies_bought_l46_4627

/-- The total number of puppies bought by Arven -/
def total_puppies : ℕ := 5

/-- The cost of each puppy on sale -/
def sale_price : ℕ := 150

/-- The cost of each puppy not on sale -/
def regular_price : ℕ := 175

/-- The number of puppies on sale -/
def sale_puppies : ℕ := 3

/-- The total cost of all puppies -/
def total_cost : ℕ := 800

theorem puppies_bought :
  total_puppies = sale_puppies + (total_cost - sale_puppies * sale_price) / regular_price :=
by sorry

end NUMINAMATH_CALUDE_puppies_bought_l46_4627


namespace NUMINAMATH_CALUDE_product_expansion_l46_4631

theorem product_expansion (x : ℝ) : (3 * x + 4) * (2 * x^2 + 3 * x + 6) = 6 * x^3 + 17 * x^2 + 30 * x + 24 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l46_4631


namespace NUMINAMATH_CALUDE_hair_cut_length_l46_4626

def initial_length : ℕ := 14
def current_length : ℕ := 1

theorem hair_cut_length : initial_length - current_length = 13 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l46_4626


namespace NUMINAMATH_CALUDE_right_triangle_enlargement_l46_4620

theorem right_triangle_enlargement (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : c^2 = a^2 + b^2) : 
  (5*c)^2 = (5*a)^2 + (5*b)^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_enlargement_l46_4620


namespace NUMINAMATH_CALUDE_product_equality_l46_4633

theorem product_equality (h : 213 * 16 = 3408) : 1.6 * 213.0 = 340.8 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l46_4633


namespace NUMINAMATH_CALUDE_rectangle_area_l46_4605

/-- The area of a rectangle with length 0.4 meters and width 0.22 meters is 0.088 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 0.4
  let width : ℝ := 0.22
  length * width = 0.088 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l46_4605


namespace NUMINAMATH_CALUDE_smaller_cup_radius_l46_4671

/-- The radius of smaller hemisphere-shaped cups when water from a large hemisphere
    is evenly distributed. -/
theorem smaller_cup_radius (R : ℝ) (n : ℕ) (h1 : R = 2) (h2 : n = 64) :
  ∃ r : ℝ, r > 0 ∧ n * ((2/3) * Real.pi * r^3) = (2/3) * Real.pi * R^3 ∧ r = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cup_radius_l46_4671


namespace NUMINAMATH_CALUDE_long_knight_min_moves_l46_4612

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents a move of the long knight -/
inductive LongKnightMove
  | horizontal : Bool → LongKnightMove  -- True for right, False for left
  | vertical : Bool → LongKnightMove    -- True for up, False for down

/-- The size of the chessboard -/
def boardSize : Nat := 8

/-- Applies a long knight move to a position -/
def applyMove (pos : Position) (move : LongKnightMove) : Position :=
  match move with
  | LongKnightMove.horizontal right =>
      let newX := if right then min (pos.x + 3) (boardSize - 1) else max (pos.x - 3) 0
      let newY := if right then min (pos.y + 1) (boardSize - 1) else max (pos.y - 1) 0
      ⟨newX, newY⟩
  | LongKnightMove.vertical up =>
      let newX := if up then min (pos.x + 1) (boardSize - 1) else max (pos.x - 1) 0
      let newY := if up then min (pos.y + 3) (boardSize - 1) else max (pos.y - 3) 0
      ⟨newX, newY⟩

/-- Checks if a position is at the opposite corner -/
def isOppositeCorner (pos : Position) : Prop :=
  pos.x = boardSize - 1 ∧ pos.y = boardSize - 1

/-- Theorem: The minimum number of moves for a long knight to reach the opposite corner is 5 -/
theorem long_knight_min_moves :
  ∀ (moves : List LongKnightMove),
    let finalPos := moves.foldl applyMove ⟨0, 0⟩
    isOppositeCorner finalPos → moves.length ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_long_knight_min_moves_l46_4612


namespace NUMINAMATH_CALUDE_store_bottles_l46_4668

/-- The total number of bottles in a grocery store, given the number of regular and diet soda bottles. -/
def total_bottles (regular_soda : ℕ) (diet_soda : ℕ) : ℕ :=
  regular_soda + diet_soda

/-- Theorem stating that the total number of bottles in the store is 38. -/
theorem store_bottles : total_bottles 30 8 = 38 := by
  sorry

end NUMINAMATH_CALUDE_store_bottles_l46_4668


namespace NUMINAMATH_CALUDE_annual_percentage_increase_l46_4615

theorem annual_percentage_increase (initial_population final_population : ℕ) 
  (h1 : initial_population = 10000)
  (h2 : final_population = 12000) :
  (((final_population - initial_population) : ℚ) / initial_population) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_annual_percentage_increase_l46_4615


namespace NUMINAMATH_CALUDE_solution_set_correct_l46_4655

def solution_set : Set ℝ := {1, 2, 3, 4, 5}

def equation (x : ℝ) : Prop :=
  (x^2 - 5*x + 5)^(x^2 - 9*x + 20) = 1

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l46_4655


namespace NUMINAMATH_CALUDE_perfect_square_condition_l46_4657

theorem perfect_square_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ z : ℝ, x^2 + 2*x*y + y^2 - a*(x + y) + 25 = z^2) → 
  (a = 10 ∨ a = -10) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l46_4657


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l46_4689

/-- Given a container with alcohol and water, prove the ratio after adding water. -/
theorem alcohol_water_ratio 
  (initial_alcohol : ℚ) 
  (initial_water : ℚ) 
  (added_water : ℚ) 
  (h1 : initial_alcohol = 4) 
  (h2 : initial_water = 4) 
  (h3 : added_water = 2666666666666667 / 1000000000000000) : 
  (initial_alcohol / (initial_water + added_water)) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l46_4689


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l46_4695

noncomputable def triangle_side_length (angleA : Real) (sideBC : Real) : Real :=
  sideBC * Real.tan angleA

theorem right_triangle_side_length :
  let angleA : Real := 30 * π / 180  -- Convert 30° to radians
  let sideBC : Real := 12
  let sideAB : Real := triangle_side_length angleA sideBC
  ∀ ε > 0, |sideAB - 6.9| < ε :=
sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l46_4695


namespace NUMINAMATH_CALUDE_not_always_possible_to_equalize_l46_4666

/-- Represents a board with integers -/
def Board := Matrix (Fin 2018) (Fin 2019) Int

/-- Checks if two positions are neighbors on the board -/
def is_neighbor (i j i' j' : Nat) : Prop :=
  (i = i' ∧ (j = j' + 1 ∨ j + 1 = j')) ∨ 
  (j = j' ∧ (i = i' + 1 ∨ i + 1 = i'))

/-- Represents a single turn of the averaging operation -/
def average_turn (b : Board) (chosen : Set (Fin 2018 × Fin 2019)) : Board :=
  sorry

/-- Represents a sequence of turns -/
def sequence_of_turns (b : Board) (turns : Nat) : Board :=
  sorry

/-- Checks if all numbers on the board are the same -/
def all_same (b : Board) : Prop :=
  ∀ i j i' j', b i j = b i' j'

theorem not_always_possible_to_equalize : ∃ (initial : Board), 
  ∀ (turns : Nat), ¬(all_same (sequence_of_turns initial turns)) :=
sorry

end NUMINAMATH_CALUDE_not_always_possible_to_equalize_l46_4666


namespace NUMINAMATH_CALUDE_sum_of_powers_divisible_by_10_l46_4659

theorem sum_of_powers_divisible_by_10 :
  ∃ k : ℕ, 111^111 + 112^112 + 113^113 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_divisible_by_10_l46_4659


namespace NUMINAMATH_CALUDE_none_of_statements_true_l46_4670

theorem none_of_statements_true (s x y : ℝ) 
  (h_s : s > 1) 
  (h_xy : x^2 * y ≠ 0) 
  (h_ineq : x * s^2 > y * s^2) : 
  ¬(-x > -y) ∧ ¬(-x > y) ∧ ¬(1 > -y/x) ∧ ¬(1 < y/x) := by
sorry

end NUMINAMATH_CALUDE_none_of_statements_true_l46_4670


namespace NUMINAMATH_CALUDE_rachel_age_when_emily_half_l46_4644

theorem rachel_age_when_emily_half (emily_age rachel_age : ℕ) : 
  rachel_age = emily_age + 4 → 
  ∃ (x : ℕ), x = rachel_age ∧ x / 2 = x - 4 → 
  x = 8 := by sorry

end NUMINAMATH_CALUDE_rachel_age_when_emily_half_l46_4644


namespace NUMINAMATH_CALUDE_complement_union_theorem_l46_4682

universe u

def U : Set (Fin 4) := {1, 2, 3, 4}
def S : Set (Fin 4) := {1, 3}
def T : Set (Fin 4) := {4}

theorem complement_union_theorem : 
  (U \ S) ∪ T = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l46_4682


namespace NUMINAMATH_CALUDE_tom_investment_is_3000_l46_4667

/-- Represents the initial investment problem with Tom and Jose --/
structure InvestmentProblem where
  jose_investment : ℕ
  jose_months : ℕ
  total_profit : ℕ
  jose_profit : ℕ

/-- Calculates Tom's initial investment given the problem parameters --/
def calculate_tom_investment (p : InvestmentProblem) : ℕ :=
  sorry

/-- Theorem stating that Tom's initial investment is 3000 --/
theorem tom_investment_is_3000 (p : InvestmentProblem)
  (h1 : p.jose_investment = 45000)
  (h2 : p.jose_months = 10)
  (h3 : p.total_profit = 27000)
  (h4 : p.jose_profit = 15000) :
  calculate_tom_investment p = 3000 :=
sorry

end NUMINAMATH_CALUDE_tom_investment_is_3000_l46_4667


namespace NUMINAMATH_CALUDE_degree_three_polynomial_l46_4640

-- Define the polynomials f and g
def f (x : ℝ) : ℝ := 2 - 15*x + 4*x^2 - 3*x^3 + 6*x^4
def g (x : ℝ) : ℝ := 4 - 3*x + x^2 - 7*x^3 + 10*x^4

-- Define the combined polynomial h
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

-- Theorem statement
theorem degree_three_polynomial :
  ∃ c : ℝ, (∀ x : ℝ, h c x = 2 + (-15 - 3*c)*x + (4 + c)*x^2 + (-3 - 7*c)*x^3) ∧ 
  (-3 - 7*c ≠ 0) ∧ (6 + 10*c = 0) :=
by sorry

end NUMINAMATH_CALUDE_degree_three_polynomial_l46_4640


namespace NUMINAMATH_CALUDE_amount_increases_to_approx_87030_l46_4697

/-- The amount after two years given an initial amount and yearly increase rate. -/
def amount_after_two_years (initial_amount : ℝ) (yearly_increase_rate : ℝ) : ℝ :=
  initial_amount * (1 + yearly_increase_rate)^2

/-- Theorem stating that given an initial amount of 64000 that increases by 1/6th each year,
    the amount after two years is approximately 87030.40. -/
theorem amount_increases_to_approx_87030 :
  let initial_amount := 64000
  let yearly_increase_rate := 1 / 6
  let final_amount := amount_after_two_years initial_amount yearly_increase_rate
  ∃ ε > 0, |final_amount - 87030.40| < ε :=
sorry

end NUMINAMATH_CALUDE_amount_increases_to_approx_87030_l46_4697


namespace NUMINAMATH_CALUDE_youngest_age_proof_l46_4602

theorem youngest_age_proof (n : ℕ) (current_avg : ℝ) (past_avg : ℝ) :
  n = 7 ∧ current_avg = 30 ∧ past_avg = 27 →
  (n : ℝ) * current_avg - (n - 1 : ℝ) * past_avg = 48 := by
  sorry

end NUMINAMATH_CALUDE_youngest_age_proof_l46_4602


namespace NUMINAMATH_CALUDE_equation_solutions_l46_4606

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 5 ∧ x₂ = 1 - Real.sqrt 5 ∧
    x₁^2 - 2*x₁ - 4 = 0 ∧ x₂^2 - 2*x₂ - 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = -1 ∧ y₂ = 2 ∧
    y₁*(y₁-2) + y₁ - 2 = 0 ∧ y₂*(y₂-2) + y₂ - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l46_4606


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l46_4678

/-- Theorem: The area of a square with perimeter 32 feet is 64 square feet. -/
theorem square_area_from_perimeter (perimeter : ℝ) (area : ℝ) : 
  perimeter = 32 → area = (perimeter / 4) ^ 2 → area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l46_4678


namespace NUMINAMATH_CALUDE_longest_piece_length_l46_4684

theorem longest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 75)
  (h2 : rope2 = 90)
  (h3 : rope3 = 135) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 15 := by
  sorry

end NUMINAMATH_CALUDE_longest_piece_length_l46_4684


namespace NUMINAMATH_CALUDE_garrison_provisions_theorem_l46_4693

/-- Represents the number of days provisions last for a garrison -/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  let totalProvisions := initialMen * (daysBeforeReinforcement + daysAfterReinforcement)
  let remainingProvisions := totalProvisions - initialMen * daysBeforeReinforcement
  let totalMenAfterReinforcement := initialMen + reinforcementMen
  (totalProvisions / initialMen : ℕ)

theorem garrison_provisions_theorem (initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement : ℕ) :
  initialMen = 2000 →
  reinforcementMen = 1300 →
  daysBeforeReinforcement = 21 →
  daysAfterReinforcement = 20 →
  provisionDays initialMen reinforcementMen daysBeforeReinforcement daysAfterReinforcement = 54 := by
  sorry

#eval provisionDays 2000 1300 21 20

end NUMINAMATH_CALUDE_garrison_provisions_theorem_l46_4693


namespace NUMINAMATH_CALUDE_calculation_proof_no_solution_proof_l46_4653

-- Part 1
theorem calculation_proof : Real.sqrt 3 ^ 2 - (2023 + Real.pi / 2) ^ 0 - (-1) ^ (-1 : Int) = 3 := by sorry

-- Part 2
theorem no_solution_proof :
  ¬∃ x : ℝ, (5 * x - 4 > 3 * x) ∧ ((2 * x - 1) / 3 < x / 2) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_no_solution_proof_l46_4653


namespace NUMINAMATH_CALUDE_sqrt_transformation_l46_4637

theorem sqrt_transformation (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_transformation_l46_4637


namespace NUMINAMATH_CALUDE_first_discount_percentage_l46_4624

theorem first_discount_percentage (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 510 →
  final_price = 381.48 →
  second_discount = 15 →
  ∃ (first_discount : ℝ),
    first_discount = 12 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l46_4624


namespace NUMINAMATH_CALUDE_least_number_divisible_up_to_28_l46_4647

def is_divisible_up_to (n : ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ m → n % k = 0

theorem least_number_divisible_up_to_28 :
  ∃ n : ℕ, n > 0 ∧ is_divisible_up_to n 28 ∧
  (∀ m : ℕ, 0 < m ∧ m < n → ¬is_divisible_up_to m 28) ∧
  n = 5348882400 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_up_to_28_l46_4647


namespace NUMINAMATH_CALUDE_quadratic_function_j_value_l46_4674

theorem quadratic_function_j_value (a b c : ℤ) (j : ℤ) :
  let f := fun (x : ℤ) => a * x^2 + b * x + c
  (f 1 = 0) →
  (60 < f 7) →
  (f 7 < 70) →
  (80 < f 8) →
  (f 8 < 90) →
  (1000 * j < f 10) →
  (f 10 < 1000 * (j + 1)) →
  j = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_j_value_l46_4674


namespace NUMINAMATH_CALUDE_sector_arc_length_l46_4665

theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 4 → angle = 2 → arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l46_4665


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l46_4603

theorem square_root_fraction_simplification :
  Real.sqrt (8^2 + 6^2) / Real.sqrt (25 + 16) = 10 / Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l46_4603


namespace NUMINAMATH_CALUDE_apple_difference_l46_4621

/-- Proves that the difference between green and red apples after delivery is 140 -/
theorem apple_difference (initial_green : ℕ) (initial_red_difference : ℕ) (delivered_green : ℕ) : 
  initial_green = 32 →
  initial_red_difference = 200 →
  delivered_green = 340 →
  (initial_green + delivered_green) - (initial_green + initial_red_difference) = 140 := by
sorry

end NUMINAMATH_CALUDE_apple_difference_l46_4621


namespace NUMINAMATH_CALUDE_profit_calculation_l46_4607

theorem profit_calculation (P Q R : ℚ) (profit_R : ℚ) :
  4 * P = 6 * Q ∧ 6 * Q = 10 * R ∧ profit_R = 840 →
  (P + Q + R) * (profit_R / R) = 4340 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l46_4607


namespace NUMINAMATH_CALUDE_range_of_a_l46_4619

theorem range_of_a (a : ℝ) : 
  (|a - 1| + |a - 4| = 3) ↔ (1 ≤ a ∧ a ≤ 4) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l46_4619


namespace NUMINAMATH_CALUDE_proposition_equivalence_l46_4617

theorem proposition_equivalence :
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.sqrt x₀ ≤ x₀ + 1) ↔ ¬(∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l46_4617


namespace NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l46_4699

-- Define the necessary structures
structure Parallelogram :=
  (diagonals_bisect : Bool)

structure Rhombus :=
  (is_parallelogram : Bool)
  (diagonals_bisect : Bool)

-- State the theorem
theorem rhombus_diagonals_bisect :
  (∀ p : Parallelogram, p.diagonals_bisect = true) →
  (∀ r : Rhombus, r.is_parallelogram = true) →
  (∀ r : Rhombus, r.diagonals_bisect = true) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_bisect_l46_4699


namespace NUMINAMATH_CALUDE_matrix_identity_proof_l46_4675

variables {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_identity_proof 
  (B : Matrix n n ℝ) 
  (h_inv : IsUnit B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) : 
  B + 10 • B⁻¹ = 8 • 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_identity_proof_l46_4675


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l46_4622

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 
  (2 * π * r = 36 * π) → 
  (π * r^2 = 324 * π) := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l46_4622


namespace NUMINAMATH_CALUDE_function_always_negative_iff_m_in_range_l46_4628

/-- The function f(x) = mx^2 - mx - 1 is negative for all real x
    if and only if m is in the interval (-4, 0]. -/
theorem function_always_negative_iff_m_in_range (m : ℝ) :
  (∀ x, m * x^2 - m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_always_negative_iff_m_in_range_l46_4628


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l46_4658

theorem line_slope_intercept_product (m b : ℚ) 
  (h_m : m = 1 / 3) 
  (h_b : b = -3 / 4) : 
  -1 < m * b ∧ m * b < 0 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l46_4658


namespace NUMINAMATH_CALUDE_acorn_theorem_l46_4683

def acorn_problem (total_acorns : ℕ) 
                  (first_month_allocation : ℚ) 
                  (second_month_allocation : ℚ) 
                  (third_month_allocation : ℚ) 
                  (first_month_consumption : ℚ) 
                  (second_month_consumption : ℚ) 
                  (third_month_consumption : ℚ) : Prop :=
  let first_month := (first_month_allocation * total_acorns : ℚ)
  let second_month := (second_month_allocation * total_acorns : ℚ)
  let third_month := (third_month_allocation * total_acorns : ℚ)
  let remaining_first := first_month * (1 - first_month_consumption)
  let remaining_second := second_month * (1 - second_month_consumption)
  let remaining_third := third_month * (1 - third_month_consumption)
  let total_remaining := remaining_first + remaining_second + remaining_third
  total_acorns = 500 ∧
  first_month_allocation = 2/5 ∧
  second_month_allocation = 3/10 ∧
  third_month_allocation = 3/10 ∧
  first_month_consumption = 1/5 ∧
  second_month_consumption = 1/4 ∧
  third_month_consumption = 3/20 ∧
  total_remaining = 400

theorem acorn_theorem : 
  ∃ (total_acorns : ℕ) 
    (first_month_allocation second_month_allocation third_month_allocation : ℚ)
    (first_month_consumption second_month_consumption third_month_consumption : ℚ),
  acorn_problem total_acorns 
                first_month_allocation 
                second_month_allocation 
                third_month_allocation 
                first_month_consumption 
                second_month_consumption 
                third_month_consumption :=
by
  sorry

end NUMINAMATH_CALUDE_acorn_theorem_l46_4683


namespace NUMINAMATH_CALUDE_rationalize_denominator_l46_4641

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l46_4641


namespace NUMINAMATH_CALUDE_quadratic_sum_l46_4662

/-- Given a quadratic function f(x) = 12x^2 + 144x + 1728, 
    prove that when written in the form a(x+b)^2+c, 
    the sum a+b+c equals 18. -/
theorem quadratic_sum (x : ℝ) : ∃ (a b c : ℝ),
  (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ 
  (a + b + c = 18) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l46_4662


namespace NUMINAMATH_CALUDE_three_dozen_cost_l46_4616

/-- The cost of apples in dollars -/
def apple_cost (dozens : ℚ) : ℚ := 15.60 * dozens / 2

/-- Theorem: The cost of three dozen apples is $23.40 -/
theorem three_dozen_cost : apple_cost 3 = 23.40 := by
  sorry

end NUMINAMATH_CALUDE_three_dozen_cost_l46_4616


namespace NUMINAMATH_CALUDE_circle_equation_and_line_slope_l46_4673

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the form mx + y - 1 = 0 -/
structure Line where
  m : ℝ

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Check if a point lies on a line -/
def onLine (l : Line) (p : ℝ × ℝ) : Prop := sorry

/-- The intersection points of a circle and a line -/
def intersectionPoints (c : Circle) (l : Line) : Set (ℝ × ℝ) := sorry

theorem circle_equation_and_line_slope 
  (c : Circle) 
  (l : Line) 
  (h1 : onCircle c (0, -4))
  (h2 : onCircle c (2, 0))
  (h3 : onCircle c (3, -1))
  (h4 : ∃ (A B : ℝ × ℝ), A ∈ intersectionPoints c l ∧ B ∈ intersectionPoints c l ∧ distance A B = 4) :
  c.center = (1, -2) ∧ c.radius^2 = 5 ∧ l.m = 4/3 := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_line_slope_l46_4673


namespace NUMINAMATH_CALUDE_f_max_implies_k_values_l46_4635

/-- The function f(x) = kx^2 - 2kx --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * k * x

/-- The interval [0,3] --/
def interval : Set ℝ := Set.Icc 0 3

/-- The maximum value of the function is 3 --/
def max_value : ℝ := 3

/-- The set of possible values for k --/
def k_values : Set ℝ := {1, -3}

theorem f_max_implies_k_values :
  ∀ k : ℝ, (∃ x ∈ interval, ∀ y ∈ interval, f k x ≥ f k y) ∧
           (∃ x ∈ interval, f k x = max_value) →
  k ∈ k_values :=
sorry

end NUMINAMATH_CALUDE_f_max_implies_k_values_l46_4635


namespace NUMINAMATH_CALUDE_polynomial_inequality_l46_4613

theorem polynomial_inequality (x : ℝ) : 1 + x^2 + x^6 + x^8 ≥ 4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l46_4613


namespace NUMINAMATH_CALUDE_units_digit_of_product_l46_4608

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_of_product (n : ℕ) :
  (3 * sum_factorials n) % 10 = 9 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l46_4608


namespace NUMINAMATH_CALUDE_simplify_expression_l46_4686

theorem simplify_expression : ((- Real.sqrt 3) ^ 2) ^ (-1/2 : ℝ) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l46_4686


namespace NUMINAMATH_CALUDE_binomial_150_149_l46_4623

theorem binomial_150_149 : Nat.choose 150 149 = 150 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_149_l46_4623


namespace NUMINAMATH_CALUDE_dogs_to_cats_ratio_l46_4660

def total_animals : ℕ := 21
def cats_to_spay : ℕ := 7

theorem dogs_to_cats_ratio :
  let dogs_to_spay := total_animals - cats_to_spay
  (dogs_to_spay : ℚ) / cats_to_spay = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_dogs_to_cats_ratio_l46_4660


namespace NUMINAMATH_CALUDE_equation_solution_l46_4611

theorem equation_solution : ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l46_4611


namespace NUMINAMATH_CALUDE_intersection_tangents_perpendicular_l46_4625

/-- Two circles in the plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

/-- The common chord of the two circles -/
def common_chord (a x y : ℝ) : Prop := 2*(a-1)*x - 2*y + a^2 = 0

/-- The condition for perpendicular tangents at intersection points -/
def perpendicular_tangents (a x y : ℝ) : Prop :=
  (y + 2) / x * (y + 1) / (x - (1 - a)) = -1

/-- The main theorem -/
theorem intersection_tangents_perpendicular (a : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 a x y ∧ common_chord a x y ∧ perpendicular_tangents a x y) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_tangents_perpendicular_l46_4625


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_101010101_base7_l46_4691

def base7_to_decimal (n : ℕ) : ℕ := n

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_101010101_base7 :
  largest_prime_divisor (base7_to_decimal 101010101) = 43 := by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_101010101_base7_l46_4691


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l46_4687

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > b > 0), 
    if a line passing through its right focus intersects the asymptotes at points A and B, 
    with the angle of inclination of the line being twice that of the asymptote, 
    and AF = (5/2) * FB, then the eccentricity of the hyperbola is 2√14/7. -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  a > b ∧ b > 0 ∧  -- Condition: a > b > 0
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ l : Set (ℝ × ℝ), F ∈ l ∧ A ∈ l ∧ B ∈ l) →  -- Line l passes through F, A, and B
  (∃ k : ℝ, k = 2 * b / a) →  -- Angle of inclination of line l is twice that of asymptote
  (A.1 - F.1) = (5/2) * (F.1 - B.1) ∧ (A.2 - F.2) = (5/2) * (F.2 - B.2) →  -- AF = (5/2) * FB
  c^2 = a^2 + b^2 →  -- Definition of c for a hyperbola
  c / a = 2 * Real.sqrt 14 / 7  -- Eccentricity is 2√14/7
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l46_4687


namespace NUMINAMATH_CALUDE_discount_calculation_l46_4652

def calculate_final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2)

theorem discount_calculation (hat_price tie_price : ℝ) 
  (hat_discount1 hat_discount2 tie_discount1 tie_discount2 : ℝ) : 
  hat_price = 20 → tie_price = 15 → 
  hat_discount1 = 0.25 → hat_discount2 = 0.20 → 
  tie_discount1 = 0.10 → tie_discount2 = 0.30 → 
  calculate_final_price hat_price hat_discount1 hat_discount2 = 12 ∧ 
  calculate_final_price tie_price tie_discount1 tie_discount2 = 9.45 := by
  sorry

#check discount_calculation

end NUMINAMATH_CALUDE_discount_calculation_l46_4652


namespace NUMINAMATH_CALUDE_simplify_expression_l46_4604

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - 2*b^2 = 9*b^3 + 4*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l46_4604


namespace NUMINAMATH_CALUDE_cubic_roots_cube_l46_4663

theorem cubic_roots_cube (p q r : ℝ) : 
  let f (x : ℝ) := x^3 + p*x^2 + q*x + r
  let g (x : ℝ) := x^3 + (p^3 - 3*p*q + 3*r)*x^2 + (q^3 - 3*p*q*r + 3*r^2)*x + r^3
  ∀ (x1 x2 x3 : ℝ), 
    (f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) → 
    (g (x1^3) = 0 ∧ g (x2^3) = 0 ∧ g (x3^3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_cube_l46_4663


namespace NUMINAMATH_CALUDE_nail_multiple_l46_4664

theorem nail_multiple (violet_nails : ℕ) (total_nails : ℕ) (M : ℕ) : 
  violet_nails = 27 →
  total_nails = 39 →
  violet_nails = M * (total_nails - violet_nails) + 3 →
  M = 2 := by sorry

end NUMINAMATH_CALUDE_nail_multiple_l46_4664


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l46_4632

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 2 ∧ x₂ = 4 ∧ 
  (x₁^2 - 6*x₁ + 8 = 0) ∧ (x₂^2 - 6*x₂ + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l46_4632


namespace NUMINAMATH_CALUDE_hotel_rooms_count_l46_4614

/-- Calculates the total number of rooms in a hotel with three wings. -/
def total_rooms_in_hotel (
  wing1_floors : ℕ) (wing1_halls_per_floor : ℕ) (wing1_rooms_per_hall : ℕ)
  (wing2_floors : ℕ) (wing2_halls_per_floor : ℕ) (wing2_rooms_per_hall : ℕ)
  (wing3_floors : ℕ) (wing3_halls_per_floor : ℕ) (wing3_rooms_per_hall : ℕ) : ℕ :=
  wing1_floors * wing1_halls_per_floor * wing1_rooms_per_hall +
  wing2_floors * wing2_halls_per_floor * wing2_rooms_per_hall +
  wing3_floors * wing3_halls_per_floor * wing3_rooms_per_hall

/-- Theorem stating that the total number of rooms in the hotel is 6648. -/
theorem hotel_rooms_count : 
  total_rooms_in_hotel 9 6 32 7 9 40 12 4 50 = 6648 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_count_l46_4614


namespace NUMINAMATH_CALUDE_optimal_tire_swap_distance_l46_4618

/-- The lifespan of a front tire in kilometers -/
def front_tire_lifespan : ℕ := 5000

/-- The lifespan of a rear tire in kilometers -/
def rear_tire_lifespan : ℕ := 3000

/-- The total distance traveled before both tires wear out when swapped optimally -/
def total_distance : ℕ := 3750

/-- Theorem stating that given the lifespans of front and rear tires, 
    the total distance traveled before both tires wear out when swapped optimally is 3750 km -/
theorem optimal_tire_swap_distance :
  ∀ (front_lifespan rear_lifespan : ℕ),
    front_lifespan = front_tire_lifespan →
    rear_lifespan = rear_tire_lifespan →
    (∃ (swap_strategy : ℕ → Bool),
      (∀ n : ℕ, swap_strategy n = true → swap_strategy (n + 1) = false) →
      (∃ (wear_front wear_rear : ℕ → ℝ),
        (∀ n : ℕ, wear_front n + wear_rear n = n) ∧
        (∀ n : ℕ, wear_front n ≤ front_lifespan) ∧
        (∀ n : ℕ, wear_rear n ≤ rear_lifespan) ∧
        (∃ m : ℕ, wear_front m = front_lifespan ∧ wear_rear m = rear_lifespan) ∧
        m = total_distance)) :=
by sorry


end NUMINAMATH_CALUDE_optimal_tire_swap_distance_l46_4618


namespace NUMINAMATH_CALUDE_victory_guarantee_l46_4650

/-- Represents the state of the archery tournament -/
structure ArcheryTournament where
  totalShots : ℕ
  halfwayPoint : ℕ
  jessicaLead : ℕ
  bullseyeScore : ℕ
  minJessicaScore : ℕ

/-- Calculates the minimum number of bullseyes Jessica needs to guarantee victory -/
def minBullseyesForVictory (tournament : ArcheryTournament) : ℕ :=
  let remainingShots := tournament.totalShots - tournament.halfwayPoint
  let maxOpponentScore := tournament.bullseyeScore * remainingShots
  let jessicaNeededScore := maxOpponentScore - tournament.jessicaLead + 1
  (jessicaNeededScore + remainingShots * tournament.minJessicaScore - 1) / 
    (tournament.bullseyeScore - tournament.minJessicaScore) + 1

theorem victory_guarantee (tournament : ArcheryTournament) 
  (h1 : tournament.totalShots = 80)
  (h2 : tournament.halfwayPoint = 40)
  (h3 : tournament.jessicaLead = 30)
  (h4 : tournament.bullseyeScore = 10)
  (h5 : tournament.minJessicaScore = 2) :
  minBullseyesForVictory tournament = 37 := by
  sorry

#eval minBullseyesForVictory { 
  totalShots := 80, 
  halfwayPoint := 40, 
  jessicaLead := 30, 
  bullseyeScore := 10, 
  minJessicaScore := 2 
}

end NUMINAMATH_CALUDE_victory_guarantee_l46_4650


namespace NUMINAMATH_CALUDE_count_leftmost_seven_eq_diff_l46_4680

/-- The set of powers of 7 from 0 to 3000 -/
def U : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 3000 ∧ n = 7^k}

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The leftmost digit of a natural number -/
def leftmost_digit (n : ℕ) : ℕ := sorry

/-- The number of elements in U with 7 as the leftmost digit -/
def count_leftmost_seven (U : Set ℕ) : ℕ := sorry

theorem count_leftmost_seven_eq_diff :
  num_digits (7^3000) = 2510 →
  leftmost_digit (7^3000) = 7 →
  count_leftmost_seven U = 3000 - 2509 := by sorry

end NUMINAMATH_CALUDE_count_leftmost_seven_eq_diff_l46_4680


namespace NUMINAMATH_CALUDE_modified_fibonacci_sum_l46_4677

def G : ℕ → ℚ
  | 0 => 2
  | 1 => 1
  | (n + 2) => G (n + 1) + G n

theorem modified_fibonacci_sum :
  (∑' n, G n / 5^n) = 280 / 99 := by
  sorry

end NUMINAMATH_CALUDE_modified_fibonacci_sum_l46_4677
