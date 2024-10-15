import Mathlib

namespace NUMINAMATH_CALUDE_photos_per_page_l3714_371463

theorem photos_per_page (total_photos : ℕ) (total_pages : ℕ) (h1 : total_photos = 736) (h2 : total_pages = 122) :
  total_photos / total_pages = 6 := by
  sorry

end NUMINAMATH_CALUDE_photos_per_page_l3714_371463


namespace NUMINAMATH_CALUDE_susan_menu_fraction_l3714_371429

theorem susan_menu_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (vegan_with_nuts : ℕ) : 
  vegan_dishes = total_dishes / 3 →
  vegan_dishes = 6 →
  vegan_with_nuts = 4 →
  (vegan_dishes - vegan_with_nuts : ℚ) / total_dishes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_susan_menu_fraction_l3714_371429


namespace NUMINAMATH_CALUDE_altitude_bisector_median_inequality_l3714_371419

/-- Triangle structure with altitude, angle bisector, and median from vertex A -/
structure Triangle :=
  (A B C : Point)
  (ha : ℝ) -- altitude from A to BC
  (βa : ℝ) -- angle bisector from A to BC
  (ma : ℝ) -- median from A to BC

/-- Theorem stating the inequality between altitude, angle bisector, and median -/
theorem altitude_bisector_median_inequality (t : Triangle) : t.ha ≤ t.βa ∧ t.βa ≤ t.ma := by
  sorry

end NUMINAMATH_CALUDE_altitude_bisector_median_inequality_l3714_371419


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3714_371402

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3714_371402


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3714_371401

theorem girls_to_boys_ratio :
  ∀ (total girls boys : ℕ),
  total = 36 →
  girls = boys + 6 →
  girls + boys = total →
  (girls : ℚ) / (boys : ℚ) = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3714_371401


namespace NUMINAMATH_CALUDE_square_sum_xy_l3714_371453

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b^2) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = a * b^4 + 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l3714_371453


namespace NUMINAMATH_CALUDE_lunch_spending_difference_l3714_371476

theorem lunch_spending_difference (total_spent friend_spent : ℕ) : 
  total_spent = 17 →
  friend_spent = 10 →
  friend_spent > total_spent - friend_spent →
  friend_spent - (total_spent - friend_spent) = 3 := by
  sorry

end NUMINAMATH_CALUDE_lunch_spending_difference_l3714_371476


namespace NUMINAMATH_CALUDE_hoseok_minyoung_problem_l3714_371439

/-- Given a line of students, calculate the number of students between two specified positions. -/
def students_between (total : ℕ) (right_pos : ℕ) (left_pos : ℕ) : ℕ :=
  left_pos - (total - right_pos + 1) - 1

/-- Theorem: In a line of 13 students, with one student 9th from the right and another 8th from the left, 
    there are 2 students between them. -/
theorem hoseok_minyoung_problem :
  students_between 13 9 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_minyoung_problem_l3714_371439


namespace NUMINAMATH_CALUDE_floor_abs_sum_equals_eleven_l3714_371490

theorem floor_abs_sum_equals_eleven :
  ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_sum_equals_eleven_l3714_371490


namespace NUMINAMATH_CALUDE_price_per_small_bottle_l3714_371447

/-- Calculates the price per small bottle given the number of large and small bottles,
    the price of large bottles, and the average price of all bottles. -/
theorem price_per_small_bottle
  (num_large : ℕ)
  (num_small : ℕ)
  (price_large : ℚ)
  (avg_price : ℚ)
  (h1 : num_large = 1325)
  (h2 : num_small = 750)
  (h3 : price_large = 189/100)
  (h4 : avg_price = 17057/10000) :
  ∃ (price_small : ℚ),
    abs (price_small - 13828/10000) < 1/10000 ∧
    (num_large * price_large + num_small * price_small) / (num_large + num_small) = avg_price :=
sorry

end NUMINAMATH_CALUDE_price_per_small_bottle_l3714_371447


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3714_371400

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 12 = 0 ∧ x ≠ -2 ∧ x^3 - 3*x^2 - 12*x + 9 = -23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3714_371400


namespace NUMINAMATH_CALUDE_simplify_fraction_l3714_371433

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3714_371433


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l3714_371430

/-- Given a function f(x) = ax³ + bx, prove that if f(a) = 8, then f(-a) = -8 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x
  f a = 8 → f (-a) = -8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l3714_371430


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l3714_371488

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x + 1

/-- Theorem corresponding to part (1) of the problem -/
theorem part_one (a : ℝ) : 
  (∀ x, f a x ≥ 0) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

/-- Theorem corresponding to part (2) of the problem -/
theorem part_two (a b : ℝ) :
  (∃ b, ∀ x, f a x < 0 ↔ b < x ∧ x < 2) ↔ a = 3/2 ∧ b = 1/2 := by sorry

/-- Theorem corresponding to part (3) of the problem -/
theorem part_three (a : ℝ) :
  ((∀ x, f a x ≤ 0) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0)) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l3714_371488


namespace NUMINAMATH_CALUDE_solve_for_C_l3714_371452

theorem solve_for_C : ∃ C : ℤ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_C_l3714_371452


namespace NUMINAMATH_CALUDE_class_size_l3714_371432

/-- The number of students in a class with French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french - both) + (german - both) + both + neither

/-- Theorem: The total number of students in the class is 87 -/
theorem class_size : total_students 41 22 9 33 = 87 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3714_371432


namespace NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l3714_371492

/-- A positive integer n cannot be represented as a sum of two or more consecutive integers
    if and only if n is a power of 2. -/
theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (hn : n > 0) :
  (∃ (k m : ℕ), k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ ¬(∃ i : ℕ, n = 2^i) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_sum_iff_not_power_of_two_l3714_371492


namespace NUMINAMATH_CALUDE_equation_solutions_l3714_371465

theorem equation_solutions :
  (∀ x : ℝ, 2 * (2 * x - 1)^2 = 32 ↔ x = 5/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 1 = 0 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) ∧
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3714_371465


namespace NUMINAMATH_CALUDE_irrational_element_exists_l3714_371408

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ (a (n + 1))^2 = a n + 1

theorem irrational_element_exists (a : ℕ → ℝ) (h : sequence_condition a) :
  ∃ i : ℕ, ¬ (∃ q : ℚ, a i = q) :=
sorry

end NUMINAMATH_CALUDE_irrational_element_exists_l3714_371408


namespace NUMINAMATH_CALUDE_total_intersection_points_l3714_371423

/-- Regular polygon inscribed in a circle -/
structure RegularPolygon where
  sides : ℕ
  inscribed : Bool

/-- Represents the configuration of regular polygons in a circle -/
structure PolygonConfiguration where
  square : RegularPolygon
  hexagon : RegularPolygon
  octagon : RegularPolygon
  shared_vertices : ℕ
  no_triple_intersections : Bool

/-- Calculates the number of intersection points between two polygons -/
def intersectionPoints (p1 p2 : RegularPolygon) (shared : Bool) : ℕ :=
  sorry

/-- Theorem stating the total number of intersection points -/
theorem total_intersection_points (config : PolygonConfiguration) : 
  config.square.sides = 4 ∧ 
  config.hexagon.sides = 6 ∧ 
  config.octagon.sides = 8 ∧
  config.square.inscribed ∧
  config.hexagon.inscribed ∧
  config.octagon.inscribed ∧
  config.shared_vertices ≤ 3 ∧
  config.no_triple_intersections →
  intersectionPoints config.square config.hexagon (config.shared_vertices > 0) +
  intersectionPoints config.square config.octagon (config.shared_vertices > 1) +
  intersectionPoints config.hexagon config.octagon (config.shared_vertices > 2) = 164 :=
sorry

end NUMINAMATH_CALUDE_total_intersection_points_l3714_371423


namespace NUMINAMATH_CALUDE_binomial_square_constant_l3714_371404

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 + 27*x + a = (3*x + b)^2) → a = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l3714_371404


namespace NUMINAMATH_CALUDE_symmetric_points_on_ellipse_l3714_371458

/-- Given an ellipse C and a line l, prove the range of m for which there are always two points on C symmetric with respect to l -/
theorem symmetric_points_on_ellipse (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧ 
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧
    (y₁ = 4*x₁ + m) ∧ 
    (y₂ = 4*x₂ + m) ∧ 
    (x₁ ≠ x₂) ∧
    (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2 ∧ y₀ = 4*x₀ + m)) ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_ellipse_l3714_371458


namespace NUMINAMATH_CALUDE_hexagon_perimeter_value_l3714_371409

/-- The perimeter of a hexagon ABCDEF with given side lengths -/
def hexagon_perimeter (AB BC CD DE EF FA : ℝ) : ℝ :=
  AB + BC + CD + DE + EF + FA

/-- Theorem: The perimeter of hexagon ABCDEF is 7.5 + √3 -/
theorem hexagon_perimeter_value :
  hexagon_perimeter 1 1.5 1.5 1.5 (Real.sqrt 3) 2 = 7.5 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_value_l3714_371409


namespace NUMINAMATH_CALUDE_hall_length_is_30_l3714_371422

/-- Represents a rectangular hall with specific properties -/
structure RectangularHall where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_breadth_relation : length = breadth + 5
  area_formula : area = length * breadth

/-- Theorem stating that a rectangular hall with the given properties has a length of 30 meters -/
theorem hall_length_is_30 (hall : RectangularHall) (h : hall.area = 750) : hall.length = 30 := by
  sorry

#check hall_length_is_30

end NUMINAMATH_CALUDE_hall_length_is_30_l3714_371422


namespace NUMINAMATH_CALUDE_water_volume_is_fifty_l3714_371420

/-- A cubical tank partially filled with water -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_fraction : ℝ

/-- The volume of water in the tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.capacity_fraction * tank.side_length^3

theorem water_volume_is_fifty (tank : CubicalTank) 
  (h1 : tank.water_level = 2)
  (h2 : tank.capacity_fraction = 0.4)
  (h3 : tank.water_level = tank.capacity_fraction * tank.side_length) :
  water_volume tank = 50 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_is_fifty_l3714_371420


namespace NUMINAMATH_CALUDE_triangle_side_length_l3714_371407

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  b = Real.sqrt 3 →
  (a / Real.sin A = b / Real.sin B) →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3714_371407


namespace NUMINAMATH_CALUDE_arctan_cos_solution_l3714_371415

theorem arctan_cos_solution (x : Real) :
  -π ≤ x ∧ x ≤ π →
  Real.arctan (Real.cos x) = x / 3 →
  x = Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) ∨
  x = -Real.arccos (Real.sqrt ((Real.sqrt 5 - 1) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_arctan_cos_solution_l3714_371415


namespace NUMINAMATH_CALUDE_A_eq_ge_1989_l3714_371475

/-- The set of functions f: ℕ → ℕ satisfying f(f(x)) - 2f(x) + x = 0 for all x ∈ ℕ -/
def F : Set (ℕ → ℕ) :=
  {f | ∀ x : ℕ, f (f x) - 2 * f x + x = 0}

/-- The set A = {f(1989) | f ∈ F} -/
def A : Set ℕ :=
  {y | ∃ f ∈ F, f 1989 = y}

/-- Theorem stating that A is equal to {k : k ≥ 1989} -/
theorem A_eq_ge_1989 : A = {k : ℕ | k ≥ 1989} := by
  sorry


end NUMINAMATH_CALUDE_A_eq_ge_1989_l3714_371475


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l3714_371484

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 15 * y) / (y + 15 * x) = 3) : 
  x / y = 0.8 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l3714_371484


namespace NUMINAMATH_CALUDE_prime_differences_l3714_371468

theorem prime_differences (x y : ℝ) 
  (h1 : Prime (x - y))
  (h2 : Prime (x^2 - y^2))
  (h3 : Prime (x^3 - y^3)) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_differences_l3714_371468


namespace NUMINAMATH_CALUDE_always_two_real_roots_find_m_value_l3714_371441

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m+1)*x + (3*m-6)

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: Given the condition, m = 3
theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁ = 0) 
  (h₂ : quadratic m x₂ = 0)
  (h₃ : x₁ + x₂ + x₁*x₂ = 7) : 
  m = 3 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_find_m_value_l3714_371441


namespace NUMINAMATH_CALUDE_verify_statement_with_flipped_cards_l3714_371435

/-- Represents a card with a letter on one side and a natural number on the other. -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel. -/
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']

/-- Checks if a natural number is even. -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents the set of cards on the table. -/
def cardsOnTable : List Card := [
  { letter := 'A', number := 0 },
  { letter := 'B', number := 0 },
  { letter := 'C', number := 4 },
  { letter := 'D', number := 5 }
]

/-- The statement to verify for each card. -/
def statementToVerify (c : Card) : Prop :=
  isVowel c.letter → isEven c.number

/-- The set of cards that need to be flipped to verify the statement. -/
def cardsToFlip : List Card :=
  cardsOnTable.filter (fun c => c.letter = 'A' ∨ c.number = 4 ∨ c.number = 5)

/-- Theorem stating that flipping the cards A, 4, and 5 is necessary and sufficient
    to verify the given statement for all cards on the table. -/
theorem verify_statement_with_flipped_cards :
  (∀ c ∈ cardsOnTable, statementToVerify c) ↔
  (∀ c ∈ cardsToFlip, statementToVerify c) :=
sorry

end NUMINAMATH_CALUDE_verify_statement_with_flipped_cards_l3714_371435


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3714_371445

theorem average_speed_calculation (distance1 distance2 speed1 speed2 : ℝ) 
  (h1 : distance1 = 20)
  (h2 : distance2 = 40)
  (h3 : speed1 = 8)
  (h4 : speed2 = 20) :
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2) = 40/3 :=
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3714_371445


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3714_371470

theorem polynomial_expansion (w : ℝ) : 
  (3 * w^3 + 4 * w^2 - 7) * (2 * w^3 - 3 * w^2 + 1) = 
  6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3714_371470


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3714_371462

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
by sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_l3714_371462


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l3714_371485

/-- Given a right triangle with sides 15 and 20, similar to a larger triangle
    where one side is twice a rectangle's shorter side (30), 
    prove the perimeter of the larger triangle is 240. -/
theorem similar_triangle_perimeter : 
  ∀ (small_triangle large_triangle : Set ℝ) 
    (rectangle : Set (ℝ × ℝ)),
  (∃ a b c : ℝ, small_triangle = {a, b, c} ∧ 
    a = 15 ∧ b = 20 ∧ c^2 = a^2 + b^2) →
  (∃ x y : ℝ, rectangle = {(30, 60), (x, y)}) →
  (∃ d e f : ℝ, large_triangle = {d, e, f} ∧
    d = 2 * 30 ∧ 
    (d / 15 = e / 20 ∧ d / 15 = f / (15^2 + 20^2).sqrt)) →
  (∃ p : ℝ, p = d + e + f ∧ p = 240) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l3714_371485


namespace NUMINAMATH_CALUDE_restaurant_ratio_proof_l3714_371450

/-- Proves that the original ratio of cooks to waiters was 1:3 given the conditions -/
theorem restaurant_ratio_proof (cooks : ℕ) (waiters : ℕ) :
  cooks = 9 →
  (cooks : ℚ) / (waiters + 12 : ℚ) = 1 / 5 →
  (cooks : ℚ) / (waiters : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_ratio_proof_l3714_371450


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l3714_371489

theorem fraction_sum_zero (a b : ℝ) (h : a ≠ b) : 
  1 / (a - b) + 1 / (b - a) = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l3714_371489


namespace NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l3714_371448

theorem no_solution_for_diophantine_equation (d : ℤ) (h : d % 4 = 3) :
  ∀ (x y : ℕ), x^2 - d * y^2 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_diophantine_equation_l3714_371448


namespace NUMINAMATH_CALUDE_expression_value_l3714_371487

theorem expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3714_371487


namespace NUMINAMATH_CALUDE_path_length_is_twenty_l3714_371428

/-- A circle with diameter AB and points C, D on AB, and P on the circle. -/
structure CircleWithPoints where
  /-- The diameter of the circle -/
  diameter : ℝ
  /-- The distance from A to C -/
  ac_distance : ℝ
  /-- The distance from B to D -/
  bd_distance : ℝ

/-- The length of the broken-line path CPD when P is at B -/
def path_length (circle : CircleWithPoints) : ℝ :=
  circle.ac_distance + circle.diameter + circle.bd_distance

/-- Theorem stating that the path length is 20 units for the given conditions -/
theorem path_length_is_twenty (circle : CircleWithPoints)
  (h1 : circle.diameter = 12)
  (h2 : circle.ac_distance = 3)
  (h3 : circle.bd_distance = 5) :
  path_length circle = 20 := by
  sorry

end NUMINAMATH_CALUDE_path_length_is_twenty_l3714_371428


namespace NUMINAMATH_CALUDE_sector_central_angle_l3714_371483

theorem sector_central_angle (perimeter : ℝ) (area : ℝ) (angle : ℝ) : 
  perimeter = 4 → area = 1 → angle = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3714_371483


namespace NUMINAMATH_CALUDE_scorpion_millipede_calculation_l3714_371467

/-- Calculates the number of additional millipedes needed to reach a daily segment requirement -/
theorem scorpion_millipede_calculation 
  (daily_requirement : ℕ) 
  (eaten_millipede_segments : ℕ) 
  (eaten_long_millipedes : ℕ) 
  (additional_millipede_segments : ℕ) 
  (h1 : daily_requirement = 800)
  (h2 : eaten_millipede_segments = 60)
  (h3 : eaten_long_millipedes = 2)
  (h4 : additional_millipede_segments = 50) :
  (daily_requirement - (eaten_millipede_segments + eaten_long_millipedes * eaten_millipede_segments * 2)) / additional_millipede_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_millipede_calculation_l3714_371467


namespace NUMINAMATH_CALUDE_log_problem_l3714_371471

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_problem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  a = 2 ∧ 
  f a 1 = 0 ∧ 
  ∀ x > 0, f a x < 1 ↔ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l3714_371471


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3714_371442

theorem trig_identity_proof :
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) +
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3714_371442


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l3714_371497

theorem min_value_quadratic_form :
  (∀ x y z : ℝ, x^2 + x*y + y^2 + z^2 ≥ 0) ∧
  (∃ x y z : ℝ, x^2 + x*y + y^2 + z^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l3714_371497


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3714_371455

theorem no_real_roots_quadratic (a : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + a ≠ 0) ↔ a > 9/4 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3714_371455


namespace NUMINAMATH_CALUDE_seven_digit_nondecreasing_integers_l3714_371416

theorem seven_digit_nondecreasing_integers (n : ℕ) (h : n = 7) :
  (Nat.choose (10 + n - 1) n) % 1000 = 440 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_nondecreasing_integers_l3714_371416


namespace NUMINAMATH_CALUDE_friday_dinner_customers_l3714_371405

/-- The number of customers during breakfast on Friday -/
def breakfast_customers : ℕ := 73

/-- The number of customers during lunch on Friday -/
def lunch_customers : ℕ := 127

/-- The predicted number of customers for Saturday -/
def saturday_prediction : ℕ := 574

/-- The number of customers during dinner on Friday -/
def dinner_customers : ℕ := 87

theorem friday_dinner_customers : 
  dinner_customers = saturday_prediction / 2 - breakfast_customers - lunch_customers :=
by sorry

end NUMINAMATH_CALUDE_friday_dinner_customers_l3714_371405


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l3714_371413

def simplified_terms_count (n : ℕ) : ℕ :=
  (n / 2 + 1)^2

theorem simplified_expression_terms (n : ℕ) (h : n = 2008) :
  simplified_terms_count n = 1010025 :=
by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l3714_371413


namespace NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersections_l3714_371491

/-- Represents a square grid -/
structure SquareGrid :=
  (n : ℕ)

/-- Number of interior vertical or horizontal lines in a square grid -/
def interior_lines (g : SquareGrid) : ℕ := g.n - 1

/-- Number of interior intersection points in a square grid -/
def interior_intersections (g : SquareGrid) : ℕ :=
  (interior_lines g) * (interior_lines g)

/-- Theorem: The number of interior intersection points on a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersections :
  ∃ (g : SquareGrid), g.n = 12 ∧ interior_intersections g = 121 := by
  sorry

end NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersections_l3714_371491


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3714_371457

theorem infinitely_many_solutions (a b : ℤ) (h_coprime : Nat.Coprime a.natAbs b.natAbs) :
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ S →
      a * x^2 + b * y^2 = z^3 ∧ Nat.Coprime x.natAbs y.natAbs :=
by sorry


end NUMINAMATH_CALUDE_infinitely_many_solutions_l3714_371457


namespace NUMINAMATH_CALUDE_road_trip_cost_sharing_l3714_371438

/-- A road trip cost-sharing scenario -/
theorem road_trip_cost_sharing
  (alice_paid bob_paid carlos_paid : ℤ)
  (h_alice : alice_paid = 90)
  (h_bob : bob_paid = 150)
  (h_carlos : carlos_paid = 210)
  (h_split_evenly : alice_paid + bob_paid + carlos_paid = 3 * ((alice_paid + bob_paid + carlos_paid) / 3)) :
  let total := alice_paid + bob_paid + carlos_paid
  let share := total / 3
  let alice_owes := share - alice_paid
  let bob_owes := share - bob_paid
  alice_owes - bob_owes = 60 := by
sorry

end NUMINAMATH_CALUDE_road_trip_cost_sharing_l3714_371438


namespace NUMINAMATH_CALUDE_max_area_inscribed_circle_l3714_371414

/-- A quadrilateral with given angles and perimeter -/
structure Quadrilateral where
  angles : Fin 4 → ℝ
  perimeter : ℝ
  sum_angles : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 2 * Real.pi
  positive_perimeter : perimeter > 0

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- A predicate indicating whether a circle can be inscribed in the quadrilateral -/
def has_inscribed_circle (q : Quadrilateral) : Prop := sorry

/-- The theorem stating that the quadrilateral with an inscribed circle has the largest area -/
theorem max_area_inscribed_circle (q : Quadrilateral) :
  has_inscribed_circle q ↔ ∀ (q' : Quadrilateral), q'.angles = q.angles ∧ q'.perimeter = q.perimeter → area q ≥ area q' :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_circle_l3714_371414


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l3714_371436

theorem min_value_cubic_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 ∧
  (8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) = 4 ↔ 
    a = 1 / Real.rpow 8 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 27 (1/3)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l3714_371436


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l3714_371481

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               M + m = 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l3714_371481


namespace NUMINAMATH_CALUDE_boat_license_combinations_count_l3714_371411

/-- Represents the set of allowed letters for boat licenses -/
def AllowedLetters : Finset Char := {'A', 'M', 'F'}

/-- Represents the set of allowed digits for boat licenses -/
def AllowedDigits : Finset Nat := Finset.range 10

/-- Calculates the number of possible boat license combinations -/
def BoatLicenseCombinations : Nat :=
  (Finset.card AllowedLetters) * (Finset.card AllowedDigits) ^ 5

/-- Theorem stating the number of possible boat license combinations -/
theorem boat_license_combinations_count :
  BoatLicenseCombinations = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_count_l3714_371411


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l3714_371494

-- Part 1
theorem inequality_solution (x : ℝ) :
  (1/3 * x - (3*x + 4)/6 ≤ 2/3) ↔ (x ≥ -8) :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (4*(x+1) ≤ 7*x + 13) ∧ ((x+2)/3 - x/2 > 1) ↔ (-3 ≤ x ∧ x < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l3714_371494


namespace NUMINAMATH_CALUDE_smallest_rational_l3714_371464

theorem smallest_rational (S : Set ℚ) (h : S = {-1, 0, 3, -1/3}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_rational_l3714_371464


namespace NUMINAMATH_CALUDE_six_power_plus_one_all_digits_same_l3714_371426

/-- A number has all digits the same in its decimal representation -/
def all_digits_same (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d ∨ m / 10^k = 0

/-- The set of positive integers n for which 6^n + 1 has all digits the same -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ all_digits_same (6^n + 1)}

theorem six_power_plus_one_all_digits_same :
  S = {1, 5} :=
sorry

end NUMINAMATH_CALUDE_six_power_plus_one_all_digits_same_l3714_371426


namespace NUMINAMATH_CALUDE_cosine_B_one_sixth_area_sqrt_three_halves_l3714_371403

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions of the problem
def triangleConditions (t : Triangle) : Prop :=
  t.b^2 = 3 * t.a * t.c

-- Part I
theorem cosine_B_one_sixth (t : Triangle) 
  (h1 : triangleConditions t) (h2 : t.a = t.b) : 
  Real.cos t.B = 1/6 := sorry

-- Part II
theorem area_sqrt_three_halves (t : Triangle) 
  (h1 : triangleConditions t) (h2 : t.B = 2 * Real.pi / 3) (h3 : t.a = Real.sqrt 2) :
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2 := sorry

end NUMINAMATH_CALUDE_cosine_B_one_sixth_area_sqrt_three_halves_l3714_371403


namespace NUMINAMATH_CALUDE_four_number_puzzle_l3714_371460

theorem four_number_puzzle :
  ∀ (a b c d : ℕ),
    a + b + c + d = 243 →
    ∃ (x : ℚ),
      (a + 8 : ℚ) = x ∧
      (b - 8 : ℚ) = x ∧
      (c * 8 : ℚ) = x ∧
      (d / 8 : ℚ) = x →
    (max (max a b) (max c d)) * (min (min a b) (min c d)) = 576 := by
  sorry

end NUMINAMATH_CALUDE_four_number_puzzle_l3714_371460


namespace NUMINAMATH_CALUDE_deadlift_percentage_increase_l3714_371493

/-- Bobby's initial deadlift at age 13 in pounds -/
def initial_deadlift : ℝ := 300

/-- Bobby's annual deadlift increase in pounds -/
def annual_increase : ℝ := 110

/-- Number of years between age 13 and 18 -/
def years : ℕ := 5

/-- Bobby's deadlift at age 18 in pounds -/
def deadlift_at_18 : ℝ := initial_deadlift + (annual_increase * years)

/-- The percentage increase we're looking for -/
def P : ℝ := sorry

/-- Theorem stating the relationship between Bobby's deadlift at 18 and the percentage increase -/
theorem deadlift_percentage_increase : deadlift_at_18 * (1 + P / 100) = deadlift_at_18 + 100 := by
  sorry

end NUMINAMATH_CALUDE_deadlift_percentage_increase_l3714_371493


namespace NUMINAMATH_CALUDE_proportionality_coefficient_l3714_371425

/-- Given variables x, y, z and a positive integer k, satisfying the following conditions:
    1. z - y = k * x
    2. x - z = k * y
    3. z = 5/3 * (x - y)
    Prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) 
  (h1 : z - y = k * x)
  (h2 : x - z = k * y)
  (h3 : z = 5/3 * (x - y)) :
  k = 3 := by sorry

end NUMINAMATH_CALUDE_proportionality_coefficient_l3714_371425


namespace NUMINAMATH_CALUDE_P_complement_subset_Q_l3714_371446

def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}
def P_complement : Set ℝ := {x | x ≥ 1}

theorem P_complement_subset_Q : P_complement ⊆ Q := by
  sorry

end NUMINAMATH_CALUDE_P_complement_subset_Q_l3714_371446


namespace NUMINAMATH_CALUDE_staff_assignment_arrangements_l3714_371406

/-- The number of staff members --/
def n : ℕ := 7

/-- The number of days --/
def d : ℕ := 7

/-- Staff member A cannot be assigned on the first day --/
def a_constraint : Prop := true

/-- Staff member B cannot be assigned on the last day --/
def b_constraint : Prop := true

/-- The number of different arrangements --/
def num_arrangements : ℕ := 3720

/-- Theorem stating the number of different arrangements --/
theorem staff_assignment_arrangements :
  a_constraint → b_constraint → num_arrangements = 3720 := by
  sorry

end NUMINAMATH_CALUDE_staff_assignment_arrangements_l3714_371406


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3714_371449

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := x ≥ 2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3714_371449


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3714_371412

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p ∣ (36^2 + 49^2) ∧ ∀ q : ℕ, Prime q → q ∣ (36^2 + 49^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3714_371412


namespace NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l3714_371474

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_30_minus_2_minus_pi_to_0_l3714_371474


namespace NUMINAMATH_CALUDE_expand_expression_l3714_371486

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4*x^2 - 45 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3714_371486


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3714_371466

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x

-- Theorem statement
theorem tangent_line_at_negative_one :
  ∃ (m b : ℝ), 
    (f' (-1) = m) ∧ 
    (f (-1) = -2) ∧ 
    (∀ x y : ℝ, y = m * (x + 1) - 2 ↔ m * x - y + b = 0) ∧
    (2 * x - y = 0 ↔ m * x - y + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_l3714_371466


namespace NUMINAMATH_CALUDE_systematic_sample_validity_l3714_371451

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  (∀ i j, i < j → i < sample.length → j < sample.length → 
    sample[i]! < sample[j]! ∧ 
    (sample[j]! - sample[i]!) = (population_size / sample_size) * (j - i)) ∧
  (∀ n, n ∈ sample → n < population_size)

theorem systematic_sample_validity :
  is_valid_systematic_sample [1, 11, 21, 31, 41] 50 5 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_validity_l3714_371451


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3714_371454

/-- Given a hyperbola with equation x²/9 - y²/4 = 1, 
    its asymptotes have the equation y = ±(2/3)x -/
theorem hyperbola_asymptotes : 
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
      (y = f x ∨ y = -f x) ∧ 
      f x = (2/3) * x) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3714_371454


namespace NUMINAMATH_CALUDE_polynomial_difference_l3714_371427

/-- A polynomial of degree 5 with specific properties -/
def f (a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

/-- The theorem statement -/
theorem polynomial_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ m : ℝ, m ∈ ({1, 2, 3, 4} : Set ℝ) → f a₁ a₂ a₃ a₄ a₅ m = 2017 * m) →
  f a₁ a₂ a₃ a₄ a₅ 10 - f a₁ a₂ a₃ a₄ a₅ (-5) = 75615 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_difference_l3714_371427


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_nonnegative_l3714_371473

theorem empty_solution_set_iff_a_nonnegative (a : ℝ) :
  (∀ x : ℝ, ¬(2*x < 5 - 3*x ∧ (x-1)/2 > a)) ↔ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_nonnegative_l3714_371473


namespace NUMINAMATH_CALUDE_bowling_pins_difference_l3714_371456

theorem bowling_pins_difference (patrick_first : ℕ) (richard_first_diff : ℕ) (richard_second_diff : ℕ) : 
  patrick_first = 70 →
  richard_first_diff = 15 →
  richard_second_diff = 3 →
  (patrick_first + richard_first_diff + (2 * (patrick_first + richard_first_diff) - richard_second_diff)) -
  (patrick_first + 2 * (patrick_first + richard_first_diff)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bowling_pins_difference_l3714_371456


namespace NUMINAMATH_CALUDE_line_not_in_plane_necessary_not_sufficient_l3714_371480

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_not_in_plane_necessary_not_sufficient
  (a b : Line) (α : Plane)
  (h : contained_in a α) :
  (¬ contained_in b α ∧ ¬ (∀ b, ¬ contained_in b α → skew a b)) ∧
  (∀ b, skew a b → ¬ contained_in b α) := by
sorry

end NUMINAMATH_CALUDE_line_not_in_plane_necessary_not_sufficient_l3714_371480


namespace NUMINAMATH_CALUDE_transport_probabilities_l3714_371444

structure TransportProbabilities where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  mutually_exclusive : train + ship + car + airplane = 1
  going_probability : ℝ

def prob : TransportProbabilities :=
  { train := 0.3
  , ship := 0.2
  , car := 0.1
  , airplane := 0.4
  , mutually_exclusive := by sorry
  , going_probability := 0.5
  }

theorem transport_probabilities (p : TransportProbabilities) :
  (p.train + p.airplane = 0.7) ∧
  (1 - p.ship = 0.8) ∧
  ((p.train + p.ship = p.going_probability) ∨ (p.car + p.airplane = p.going_probability)) :=
by sorry

end NUMINAMATH_CALUDE_transport_probabilities_l3714_371444


namespace NUMINAMATH_CALUDE_cube_sphere_comparison_l3714_371443

theorem cube_sphere_comparison (a b R : ℝ) 
  (h1 : 6 * a^2 = 4 * Real.pi * R^2) 
  (h2 : b^3 = (4/3) * Real.pi * R^3) :
  a < b :=
by sorry

end NUMINAMATH_CALUDE_cube_sphere_comparison_l3714_371443


namespace NUMINAMATH_CALUDE_find_m_l3714_371421

theorem find_m (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) :
  m = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3714_371421


namespace NUMINAMATH_CALUDE_expression_equality_l3714_371498

theorem expression_equality : (1/2)⁻¹ + (Real.pi + 2023)^0 - 2 * Real.cos (Real.pi / 3) + Real.sqrt 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3714_371498


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l3714_371477

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : 
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l3714_371477


namespace NUMINAMATH_CALUDE_polynomial_value_l3714_371496

theorem polynomial_value : 
  let x : ℚ := 1/2
  2*x^2 - 5*x + x^2 + 4*x - 3*x^2 - 2 = -5/2 := by sorry

end NUMINAMATH_CALUDE_polynomial_value_l3714_371496


namespace NUMINAMATH_CALUDE_apple_distribution_theorem_l3714_371459

/-- Represents the distribution of apples in bags -/
structure AppleDistribution where
  totalApples : Nat
  totalBags : Nat
  xApples : Nat
  threeAppleBags : Nat
  xAppleBags : Nat

/-- Checks if the apple distribution is valid -/
def isValidDistribution (d : AppleDistribution) : Prop :=
  d.totalApples = 109 ∧
  d.totalBags = 20 ∧
  d.threeAppleBags + d.xAppleBags = d.totalBags ∧
  d.xApples * d.xAppleBags + 3 * d.threeAppleBags = d.totalApples

/-- Theorem stating the possible values of x -/
theorem apple_distribution_theorem :
  ∀ d : AppleDistribution,
    isValidDistribution d →
    d.xApples = 10 ∨ d.xApples = 52 :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_theorem_l3714_371459


namespace NUMINAMATH_CALUDE_sum_three_numbers_l3714_371479

theorem sum_three_numbers (a b c M : ℚ) : 
  a + b + c = 120 ∧ 
  a - 9 = M ∧ 
  b + 9 = M ∧ 
  9 * c = M → 
  M = 1080 / 19 := by
sorry

end NUMINAMATH_CALUDE_sum_three_numbers_l3714_371479


namespace NUMINAMATH_CALUDE_find_n_l3714_371431

theorem find_n : ∃ n : ℤ, 3^4 - 13 = 4^3 + n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3714_371431


namespace NUMINAMATH_CALUDE_parabola_c_value_l3714_371437

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a parabola with given properties -/
def parabola : QuadraticFunction :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_c_value :
  (parabola.a * 3^2 + parabola.b * 3 + parabola.c = -5) ∧
  (parabola.a * 5^2 + parabola.b * 5 + parabola.c = -3) →
  parabola.c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3714_371437


namespace NUMINAMATH_CALUDE_greatest_integer_radius_eight_is_greatest_l3714_371469

theorem greatest_integer_radius (r : ℕ) : r ^ 2 < 75 → r ≤ 8 := by
  sorry

theorem eight_is_greatest : ∃ (r : ℕ), r ^ 2 < 75 ∧ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_eight_is_greatest_l3714_371469


namespace NUMINAMATH_CALUDE_browser_windows_l3714_371482

theorem browser_windows (num_browsers : Nat) (tabs_per_window : Nat) (total_tabs : Nat) :
  num_browsers = 2 →
  tabs_per_window = 10 →
  total_tabs = 60 →
  ∃ (windows_per_browser : Nat),
    windows_per_browser * tabs_per_window * num_browsers = total_tabs ∧
    windows_per_browser = 3 := by
  sorry

end NUMINAMATH_CALUDE_browser_windows_l3714_371482


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3714_371434

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3714_371434


namespace NUMINAMATH_CALUDE_dot_product_in_triangle_l3714_371461

/-- Given a triangle ABC where AB = (2, 3) and AC = (3, 4), prove that the dot product of AB and BC is 5. -/
theorem dot_product_in_triangle (A B C : ℝ × ℝ) : 
  B - A = (2, 3) → C - A = (3, 4) → (B - A) • (C - B) = 5 := by sorry

end NUMINAMATH_CALUDE_dot_product_in_triangle_l3714_371461


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3714_371418

theorem complex_number_in_third_quadrant :
  let z : ℂ := ((-1 : ℂ) - 2*I) / (1 - 2*I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3714_371418


namespace NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3714_371478

/-- Represents the probability of Alice having the ball after two turns in the basketball game. -/
def alice_has_ball_after_two_turns (
  alice_toss_prob : ℚ)  -- Probability of Alice tossing the ball to Bob
  (alice_keep_prob : ℚ)  -- Probability of Alice keeping the ball
  (bob_toss_prob : ℚ)    -- Probability of Bob tossing the ball to Alice
  (bob_keep_prob : ℚ) : ℚ :=  -- Probability of Bob keeping the ball
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

/-- Theorem stating the probability of Alice having the ball after two turns -/
theorem alice_probability_after_two_turns :
  alice_has_ball_after_two_turns (2/3) (1/3) (1/4) (3/4) = 5/18 := by
  sorry

end NUMINAMATH_CALUDE_alice_probability_after_two_turns_l3714_371478


namespace NUMINAMATH_CALUDE_oil_usage_l3714_371495

theorem oil_usage (rons_oil sara_usage : ℚ) 
  (h1 : rons_oil = 3/8)
  (h2 : sara_usage = 5/6 * rons_oil) : 
  sara_usage = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_oil_usage_l3714_371495


namespace NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3714_371440

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  planesPerpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_implies_planes_perp_l3714_371440


namespace NUMINAMATH_CALUDE_min_max_values_of_f_l3714_371424

def f (x : ℝ) := -2 * x + 1

theorem min_max_values_of_f :
  ∀ x ∈ Set.Icc 0 5,
    (∃ y ∈ Set.Icc 0 5, f y ≤ f x) ∧
    (∃ z ∈ Set.Icc 0 5, f x ≤ f z) ∧
    f 5 = -9 ∧
    f 0 = 1 ∧
    (∀ w ∈ Set.Icc 0 5, -9 ≤ f w ∧ f w ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_min_max_values_of_f_l3714_371424


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3714_371417

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℚ := (n^2 + 3*n) / 2

-- Define the general term of the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the terms of the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℚ := n / (4*n + 4)

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → a n = n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (4*n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3714_371417


namespace NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l3714_371410

theorem cube_diff_even_iff_sum_even (p q : ℕ) :
  Even (p^3 - q^3) ↔ Even (p + q) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_even_iff_sum_even_l3714_371410


namespace NUMINAMATH_CALUDE_chord_length_l3714_371472

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (t : ℝ) : 
  let line : ℝ → ℝ × ℝ := λ t => (1 + 2*t, 2 + t)
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + p.2^2 = 9
  let chord_length := 
    Real.sqrt (4 * (9 - (3 / Real.sqrt 5)^2))
  chord_length = 12/5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l3714_371472


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3714_371499

/-- A geometric sequence with a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ a 4 = 4

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3714_371499
