import Mathlib

namespace NUMINAMATH_CALUDE_accuracy_of_3_145e8_l655_65543

/-- Represents the level of accuracy for a number -/
inductive Accuracy
  | HundredThousand
  | Million
  | TenMillion
  | HundredMillion

/-- Determines the accuracy of a number in scientific notation -/
def accuracy_of_scientific_notation (mantissa : Float) (exponent : Int) : Accuracy :=
  match exponent with
  | 8 => Accuracy.HundredThousand
  | 9 => Accuracy.Million
  | 10 => Accuracy.TenMillion
  | 11 => Accuracy.HundredMillion
  | _ => Accuracy.HundredThousand  -- Default case

theorem accuracy_of_3_145e8 :
  accuracy_of_scientific_notation 3.145 8 = Accuracy.HundredThousand :=
by sorry

end NUMINAMATH_CALUDE_accuracy_of_3_145e8_l655_65543


namespace NUMINAMATH_CALUDE_convex_hull_perimeter_bounds_l655_65579

/-- A regular polygon inscribed in a unit circle -/
structure RegularPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The convex hull formed by the vertices of two regular polygons inscribed in a unit circle -/
structure ConvexHull where
  p1 : RegularPolygon
  p2 : RegularPolygon

/-- The perimeter of the convex hull -/
noncomputable def perimeter (ch : ConvexHull) : ℝ :=
  sorry

theorem convex_hull_perimeter_bounds (ch : ConvexHull) 
  (h1 : ch.p1.n = 6) 
  (h2 : ch.p2.n = 7) : 
  6.1610929 ≤ perimeter ch ∧ perimeter ch ≤ 6.1647971 :=
sorry

end NUMINAMATH_CALUDE_convex_hull_perimeter_bounds_l655_65579


namespace NUMINAMATH_CALUDE_m_range_for_three_roots_l655_65506

/-- The function f(x) defined in the problem -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 3

/-- The function g(x) defined in the problem -/
def g (x m : ℝ) : ℝ := f x - m

/-- Theorem stating the range of m for which g(x) has exactly 3 real roots -/
theorem m_range_for_three_roots :
  ∀ m : ℝ, (∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x ∈ s, g x m = 0) → m ∈ Set.Ioo (-24) 8 :=
sorry

end NUMINAMATH_CALUDE_m_range_for_three_roots_l655_65506


namespace NUMINAMATH_CALUDE_base5_product_l655_65574

/-- Converts a base-5 number represented as a list of digits to a natural number. -/
def fromBase5 (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a natural number to its base-5 representation as a list of digits. -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The statement of the problem. -/
theorem base5_product : 
  let a := fromBase5 [1, 3, 1]
  let b := fromBase5 [1, 3]
  toBase5 (a * b) = [2, 3, 3, 3] := by sorry

end NUMINAMATH_CALUDE_base5_product_l655_65574


namespace NUMINAMATH_CALUDE_small_kite_area_l655_65560

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its vertices -/
def kiteArea (a b c d : Point) : ℝ :=
  let base := c.x - a.x
  let height := b.y - a.y
  base * height

/-- The grid spacing in inches -/
def gridSpacing : ℝ := 2

theorem small_kite_area :
  let a := Point.mk 0 6
  let b := Point.mk 3 10
  let c := Point.mk 6 6
  let d := Point.mk 3 0
  kiteArea a b c d * gridSpacing^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_small_kite_area_l655_65560


namespace NUMINAMATH_CALUDE_linda_spent_25_dollars_l655_65542

/-- The amount Linda spent on her purchases -/
def linda_total_spent (coloring_book_price : ℚ) (coloring_book_quantity : ℕ)
  (peanut_pack_price : ℚ) (peanut_pack_quantity : ℕ) (stuffed_animal_price : ℚ) : ℚ :=
  coloring_book_price * coloring_book_quantity +
  peanut_pack_price * peanut_pack_quantity +
  stuffed_animal_price

theorem linda_spent_25_dollars :
  linda_total_spent 4 2 (3/2) 4 11 = 25 := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_25_dollars_l655_65542


namespace NUMINAMATH_CALUDE_product_digit_sum_theorem_l655_65575

def is_single_digit (n : ℕ) : Prop := 1 < n ∧ n < 10

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

theorem product_digit_sum_theorem (x y : ℕ) :
  is_single_digit x ∧ is_single_digit y ∧ x ≠ 9 ∧ y ≠ 9 ∧ digit_sum (x * y) = x →
  (x = 3 ∧ y = 4) ∨ (x = 3 ∧ y = 7) ∨ (x = 6 ∧ y = 4) ∨ (x = 6 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_theorem_l655_65575


namespace NUMINAMATH_CALUDE_radical_simplification_l655_65516

theorem radical_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (3 * q^2) * Real.sqrt (8 * q^3) = 6 * q^3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l655_65516


namespace NUMINAMATH_CALUDE_second_train_speed_l655_65566

/-- Calculates the speed of the second train given the parameters of two trains crossing each other. -/
theorem second_train_speed
  (length1 : ℝ)
  (speed1 : ℝ)
  (length2 : ℝ)
  (time_to_cross : ℝ)
  (h1 : length1 = 270)
  (h2 : speed1 = 120)
  (h3 : length2 = 230.04)
  (h4 : time_to_cross = 9)
  : ∃ (speed2 : ℝ), speed2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l655_65566


namespace NUMINAMATH_CALUDE_angle_on_line_l655_65529

theorem angle_on_line (α : Real) : 
  (0 ≤ α) ∧ (α < π) ∧ 
  (∃ (x y : Real), x + 2 * y = 0 ∧ 
    x = Real.cos α ∧ y = Real.sin α) →
  Real.sin (π / 2 - 2 * α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_angle_on_line_l655_65529


namespace NUMINAMATH_CALUDE_cookie_distribution_l655_65513

/-- Given 24 cookies, prove that 6 friends can share them if each friend receives 3 more cookies than the previous friend, with the first friend receiving at least 1 cookie. -/
theorem cookie_distribution (total_cookies : ℕ) (cookie_increment : ℕ) (n : ℕ) : 
  total_cookies = 24 →
  cookie_increment = 3 →
  (n : ℚ) * ((1 : ℚ) + (1 : ℚ) + (cookie_increment : ℚ) * ((n : ℚ) - 1)) / 2 = (total_cookies : ℚ) →
  n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l655_65513


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l655_65550

theorem integral_x_squared_plus_x_minus_one_times_exp_x_over_two :
  ∫ x in (0 : ℝ)..2, (x^2 + x - 1) * Real.exp (x / 2) = 2 * (3 * Real.exp 1 - 5) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_x_minus_one_times_exp_x_over_two_l655_65550


namespace NUMINAMATH_CALUDE_sugar_amount_l655_65551

/-- The number of cups of sugar in Mary's cake recipe -/
def sugar : ℕ := sorry

/-- The total amount of flour needed for the recipe in cups -/
def total_flour : ℕ := 9

/-- The amount of flour already added in cups -/
def flour_added : ℕ := 2

/-- The remaining flour to be added is 1 cup more than the amount of sugar -/
axiom remaining_flour_sugar_relation : total_flour - flour_added = sugar + 1

theorem sugar_amount : sugar = 6 := by sorry

end NUMINAMATH_CALUDE_sugar_amount_l655_65551


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l655_65518

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 ≤ a ∧ a < b ∧ b ≤ 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l655_65518


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l655_65521

/-- Represents the number of male students in the class -/
def male_students : ℕ := 40

/-- Represents the number of female students in the class -/
def female_students : ℕ := 30

/-- Represents the total number of students in the class -/
def total_students : ℕ := male_students + female_students

/-- Represents the size of the stratified sample -/
def sample_size : ℕ := 7

/-- Calculates the number of male students in the stratified sample -/
def male_sample : ℕ := (male_students * sample_size + total_students - 1) / total_students

/-- Calculates the number of female students in the stratified sample -/
def female_sample : ℕ := sample_size - male_sample

/-- Theorem stating that the stratified sample consists of 4 male and 3 female students -/
theorem stratified_sample_composition :
  male_sample = 4 ∧ female_sample = 3 :=
sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l655_65521


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l655_65561

theorem sufficient_not_necessary (a b : ℝ → ℝ) : 
  (∀ x, |a x + b x| + |a x - b x| ≤ 1 → (a x)^2 + (b x)^2 ≤ 1) ∧ 
  (∃ x, (a x)^2 + (b x)^2 ≤ 1 ∧ |a x + b x| + |a x - b x| > 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l655_65561


namespace NUMINAMATH_CALUDE_rectangle_other_vertices_y_sum_l655_65500

/-- A rectangle in a 2D plane --/
structure Rectangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ
  vertex4 : ℝ × ℝ

/-- The property that two points are opposite vertices of a rectangle --/
def areOppositeVertices (p1 p2 : ℝ × ℝ) (r : Rectangle) : Prop :=
  (r.vertex1 = p1 ∧ r.vertex3 = p2) ∨ (r.vertex1 = p2 ∧ r.vertex3 = p1) ∨
  (r.vertex2 = p1 ∧ r.vertex4 = p2) ∨ (r.vertex2 = p2 ∧ r.vertex4 = p1)

/-- The sum of y-coordinates of two points --/
def sumYCoordinates (p1 p2 : ℝ × ℝ) : ℝ :=
  p1.2 + p2.2

theorem rectangle_other_vertices_y_sum 
  (r : Rectangle) 
  (h : areOppositeVertices (3, 17) (9, -4) r) : 
  ∃ (v1 v2 : ℝ × ℝ), 
    ((v1 = r.vertex2 ∧ v2 = r.vertex4) ∨ (v1 = r.vertex1 ∧ v2 = r.vertex3)) ∧
    sumYCoordinates v1 v2 = 13 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_other_vertices_y_sum_l655_65500


namespace NUMINAMATH_CALUDE_largest_five_digit_divisible_by_four_l655_65591

theorem largest_five_digit_divisible_by_four :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≤ 99996 ∧ n % 4 = 0 → n ≤ 99996 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_five_digit_divisible_by_four_l655_65591


namespace NUMINAMATH_CALUDE_min_sum_inequality_min_sum_achievable_l655_65525

theorem min_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) ≥ 4 / (360 ^ (1/4 : ℝ)) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (12 * a)) + ((a + b + c) / (5 * a * b * c)) = 4 / (360 ^ (1/4 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_min_sum_inequality_min_sum_achievable_l655_65525


namespace NUMINAMATH_CALUDE_vector_perpendicular_condition_l655_65532

theorem vector_perpendicular_condition (k : ℝ) : 
  let a : ℝ × ℝ := (-1, k)
  let b : ℝ × ℝ := (3, 1)
  (a.1 + b.1, a.2 + b.2) • a = 0 → k = -2 ∨ k = 1 := by
sorry

end NUMINAMATH_CALUDE_vector_perpendicular_condition_l655_65532


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_40_l655_65576

theorem ones_digit_of_8_to_40 (cycle : List Nat) (h_cycle : cycle = [8, 4, 2, 6]) :
  (8^40 : ℕ) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_40_l655_65576


namespace NUMINAMATH_CALUDE_middle_digit_is_two_l655_65538

theorem middle_digit_is_two (ABCDE : ℕ) : 
  ABCDE ≥ 10000 ∧ ABCDE < 100000 →
  4 * (10 * ABCDE + 4) = 400000 + ABCDE →
  (ABCDE / 100) % 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_middle_digit_is_two_l655_65538


namespace NUMINAMATH_CALUDE_relationship_abc_l655_65508

theorem relationship_abc (a b c : ℝ) :
  (∃ u v : ℝ, u - v = a ∧ u^2 - v^2 = b ∧ u^3 - v^3 = c) →
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l655_65508


namespace NUMINAMATH_CALUDE_combined_weight_leo_kendra_prove_combined_weight_l655_65535

/-- The combined weight of Leo and Kendra given Leo's current weight and the condition of their weight relationship after Leo gains 10 pounds. -/
theorem combined_weight_leo_kendra : ℝ → ℝ → Prop :=
  fun leo_weight kendra_weight =>
    (leo_weight = 104) →
    (leo_weight + 10 = 1.5 * kendra_weight) →
    (leo_weight + kendra_weight = 180)

/-- The theorem statement -/
theorem prove_combined_weight : ∃ (leo_weight kendra_weight : ℝ),
  combined_weight_leo_kendra leo_weight kendra_weight :=
sorry

end NUMINAMATH_CALUDE_combined_weight_leo_kendra_prove_combined_weight_l655_65535


namespace NUMINAMATH_CALUDE_first_five_terms_of_sequence_l655_65584

def a (n : ℕ) : ℤ := (-1)^n + n

theorem first_five_terms_of_sequence :
  (List.range 5).map (fun i => a (i + 1)) = [0, 3, 2, 5, 4] := by
  sorry

end NUMINAMATH_CALUDE_first_five_terms_of_sequence_l655_65584


namespace NUMINAMATH_CALUDE_collinear_vectors_l655_65509

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem collinear_vectors (k : ℝ) :
  (∃ t : ℝ, (a.1 + k * c.1, a.2 + k * c.2) = t • (2 * b.1 - a.1, 2 * b.2 - a.2)) →
  k = -16/13 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_l655_65509


namespace NUMINAMATH_CALUDE_second_month_sale_l655_65541

def sale_month1 : ℕ := 7435
def sale_month3 : ℕ := 7855
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7560
def sale_month6 : ℕ := 6000
def average_sale : ℕ := 7500
def num_months : ℕ := 6

theorem second_month_sale :
  sale_month1 + sale_month3 + sale_month4 + sale_month5 + sale_month6 + 7920 = average_sale * num_months :=
by sorry

end NUMINAMATH_CALUDE_second_month_sale_l655_65541


namespace NUMINAMATH_CALUDE_tree_height_proof_l655_65505

/-- Given a tree that is currently 180 inches tall and 50% taller than its original height,
    prove that its original height was 10 feet. -/
theorem tree_height_proof :
  let current_height_inches : ℝ := 180
  let growth_factor : ℝ := 1.5
  let inches_per_foot : ℝ := 12
  current_height_inches / growth_factor / inches_per_foot = 10
  := by sorry

end NUMINAMATH_CALUDE_tree_height_proof_l655_65505


namespace NUMINAMATH_CALUDE_age_problem_l655_65595

/-- Given ages a, b, c, d, and their sum Y, prove b's age. -/
theorem age_problem (a b c d Y : ℚ) 
  (h1 : a = b + 2)           -- a is two years older than b
  (h2 : b = 2 * c)           -- b is twice as old as c
  (h3 : d = a / 2)           -- d is half the age of a
  (h4 : a + b + c + d = Y)   -- sum of ages is Y
  : b = Y / 3 - 1 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l655_65595


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l655_65581

/-- Given two points A and B in a 2D plane, where:
  - A is at (0, 0)
  - B is on the line y = 5
  - The slope of segment AB is 3/4
  Prove that the sum of the x- and y-coordinates of point B is 35/3 -/
theorem coordinate_sum_of_point_B (B : ℝ × ℝ) : 
  B.2 = 5 ∧ 
  (B.2 - 0) / (B.1 - 0) = 3 / 4 → 
  B.1 + B.2 = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l655_65581


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l655_65507

-- Equation 1: x^2 - 4x + 3 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 ∧ x₁ = 1 ∧ x₂ = 3 := by
  sorry

-- Equation 2: (x + 1)(x - 2) = 4
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)*(x₁ - 2) = 4 ∧ (x₂ + 1)*(x₂ - 2) = 4 ∧ x₁ = -2 ∧ x₂ = 3 := by
  sorry

-- Equation 3: 3x(x - 1) = 2 - 2x
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂ ∧ x₁ = 1 ∧ x₂ = -2/3 := by
  sorry

-- Equation 4: 2x^2 - 4x - 1 = 0
theorem solve_equation_4 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
  x₁ = (2 + Real.sqrt 6) / 2 ∧ x₂ = (2 - Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l655_65507


namespace NUMINAMATH_CALUDE_antonia_emails_l655_65546

theorem antonia_emails :
  ∀ (total : ℕ),
  (1 : ℚ) / 4 * total = total - (3 : ℚ) / 4 * total →
  (2 : ℚ) / 5 * ((3 : ℚ) / 4 * total) = ((3 : ℚ) / 4 * total) - 180 →
  total = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_antonia_emails_l655_65546


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l655_65573

theorem cosine_equation_solution (A ω φ b : ℝ) (h_A : A > 0) :
  (∀ x, 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b) →
  A = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l655_65573


namespace NUMINAMATH_CALUDE_power_product_equality_l655_65544

theorem power_product_equality (x : ℝ) (h : x > 0) : 
  x^x * x^x = x^(2*x) ∧ x^x * x^x = (x^2)^x :=
by sorry

end NUMINAMATH_CALUDE_power_product_equality_l655_65544


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l655_65526

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  nonSpeedsters : ℕ
  speedsterConvertibles : ℕ

/-- Theorem stating the number of Speedster convertibles in the inventory -/
theorem speedster_convertibles_count (inv : Inventory) :
  inv.nonSpeedsters = 30 ∧
  inv.speedsters = 3 * inv.total / 4 ∧
  inv.nonSpeedsters = inv.total - inv.speedsters ∧
  inv.speedsterConvertibles = 3 * inv.speedsters / 5 →
  inv.speedsterConvertibles = 54 := by
  sorry


end NUMINAMATH_CALUDE_speedster_convertibles_count_l655_65526


namespace NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l655_65564

theorem max_consecutive_sum_less_than_1000 :
  ∀ n : ℕ, (n * (n + 1)) / 2 < 1000 ↔ n ≤ 44 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_less_than_1000_l655_65564


namespace NUMINAMATH_CALUDE_line_through_points_l655_65577

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Given distinct vectors p and q, if m*p + (5/6)*q lies on the line through p and q, then m = 1/6 -/
theorem line_through_points (p q : V) (m : ℝ) 
  (h_distinct : p ≠ q)
  (h_on_line : ∃ t : ℝ, m • p + (5/6) • q = p + t • (q - p)) :
  m = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l655_65577


namespace NUMINAMATH_CALUDE_quadratic_vertex_l655_65583

/-- The quadratic function f(x) = -(x+1)^2 - 8 has vertex coordinates (-1, -8) -/
theorem quadratic_vertex (x : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ -(x + 1)^2 - 8
  (∀ x, f x ≤ f (-1)) ∧ f (-1) = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l655_65583


namespace NUMINAMATH_CALUDE_painting_equation_proof_l655_65585

theorem painting_equation_proof (t : ℝ) : 
  let doug_rate : ℝ := 1 / 4
  let dave_rate : ℝ := 1 / 6
  let combined_rate : ℝ := doug_rate + dave_rate
  let break_time : ℝ := 1 / 2
  (combined_rate * (t - break_time) = 1) ↔ 
  ((1 / 4 + 1 / 6) * (t - 1 / 2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_painting_equation_proof_l655_65585


namespace NUMINAMATH_CALUDE_line_bisects_circle_coefficient_product_range_l655_65555

/-- Given a line that always bisects the circumference of a circle, prove the range of the product of its coefficients. -/
theorem line_bisects_circle_coefficient_product_range
  (a b : ℝ)
  (h_bisect : ∀ (x y : ℝ), 4 * a * x - 3 * b * y + 48 = 0 →
    (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
        x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
        4 * a * x₁ - 3 * b * y₁ + 48 = 0 ∧
        4 * a * x₂ - 3 * b * y₂ + 48 = 0 ∧
        (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2))) :
  a * b ≤ 4 ∧ ∀ (k : ℝ), k < 4 → ∃ (a' b' : ℝ), a' * b' = k ∧
    ∀ (x y : ℝ), 4 * a' * x - 3 * b' * y + 48 = 0 →
      (x ^ 2 + y ^ 2 + 6 * x - 8 * y + 1 = 0 →
        ∃ (x₁ y₁ x₂ y₂ : ℝ),
          x₁ ^ 2 + y₁ ^ 2 + 6 * x₁ - 8 * y₁ + 1 = 0 ∧
          x₂ ^ 2 + y₂ ^ 2 + 6 * x₂ - 8 * y₂ + 1 = 0 ∧
          4 * a' * x₁ - 3 * b' * y₁ + 48 = 0 ∧
          4 * a' * x₂ - 3 * b' * y₂ + 48 = 0 ∧
          (x₁ - x₂) ^ 2 + (y₁ - y₂) ^ 2 = 4 * ((x - 3) ^ 2 + (y - 4) ^ 2)) :=
by sorry

end NUMINAMATH_CALUDE_line_bisects_circle_coefficient_product_range_l655_65555


namespace NUMINAMATH_CALUDE_pages_left_to_read_l655_65553

theorem pages_left_to_read (total_pages : ℕ) (saturday_morning : ℕ) (saturday_night : ℕ) : 
  total_pages = 360 →
  saturday_morning = 40 →
  saturday_night = 10 →
  total_pages - (saturday_morning + saturday_night + 2 * (saturday_morning + saturday_night)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_read_l655_65553


namespace NUMINAMATH_CALUDE_max_number_with_30_divisors_l655_65598

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is divisible by m -/
def is_divisible_by (n m : ℕ) : Prop := sorry

theorem max_number_with_30_divisors :
  ∀ n : ℕ, 
    is_divisible_by n 30 → 
    num_divisors n = 30 → 
    n ≤ 11250 ∧ 
    (n = 11250 → is_divisible_by 11250 30 ∧ num_divisors 11250 = 30) :=
by sorry

end NUMINAMATH_CALUDE_max_number_with_30_divisors_l655_65598


namespace NUMINAMATH_CALUDE_tower_difference_l655_65586

/-- The number of blocks Randy used to build different structures -/
structure BlockCounts where
  total : ℕ
  house : ℕ
  tower : ℕ
  bridge : ℕ

/-- The theorem stating the difference in blocks used for the tower versus the house and bridge combined -/
theorem tower_difference (b : BlockCounts) (h1 : b.total = 250) (h2 : b.house = 65) (h3 : b.tower = 120) (h4 : b.bridge = 45) :
  b.tower - (b.house + b.bridge) = 10 := by
  sorry


end NUMINAMATH_CALUDE_tower_difference_l655_65586


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l655_65582

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P(1, 0),
    the line L2 passing through P and perpendicular to L1 has the equation 2x + y - 2 = 0 -/
theorem perpendicular_line_equation (L1 : (ℝ × ℝ) → Prop) (P : ℝ × ℝ) :
  L1 = λ (x, y) => x - 2*y - 2 = 0 →
  P = (1, 0) →
  ∃ (L2 : (ℝ × ℝ) → Prop),
    (∀ (x y : ℝ), L2 (x, y) ↔ 2*x + y - 2 = 0) ∧
    L2 P ∧
    (∀ (v w : ℝ × ℝ), L1 v ∧ L1 w → L2 v ∧ L2 w →
      (v.1 - w.1) * (v.1 - w.1) + (v.2 - w.2) * (v.2 - w.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l655_65582


namespace NUMINAMATH_CALUDE_price_difference_per_can_l655_65503

/-- Proves that the difference in price per can between the local grocery store and the bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price : ℚ) (bulk_cans : ℕ) (grocery_price : ℚ) (grocery_cans : ℕ) 
  (h1 : bulk_price = 12) 
  (h2 : bulk_cans = 48) 
  (h3 : grocery_price = 6) 
  (h4 : grocery_cans = 12) : 
  (grocery_price / grocery_cans - bulk_price / bulk_cans) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_per_can_l655_65503


namespace NUMINAMATH_CALUDE_circular_garden_area_l655_65556

/-- The area of a circular garden with radius 8 units, where the length of the fence
    (circumference) is 1/4 of the area of the garden. -/
theorem circular_garden_area : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference = (1/4) * area →
  area = 64 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circular_garden_area_l655_65556


namespace NUMINAMATH_CALUDE_flower_count_l655_65522

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different candles --/
def num_candles : ℕ := 4

/-- The number of candles to choose --/
def candles_to_choose : ℕ := 2

/-- The number of flowers to choose --/
def flowers_to_choose : ℕ := 8

/-- The total number of candle + flower groupings --/
def total_groupings : ℕ := 54

theorem flower_count :
  ∃ (F : ℕ), 
    F > 0 ∧
    choose num_candles candles_to_choose * choose F flowers_to_choose = total_groupings ∧
    F = 9 :=
by sorry

end NUMINAMATH_CALUDE_flower_count_l655_65522


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l655_65536

theorem cubic_equation_solution :
  ∃ x : ℝ, 2 * x^3 + 24 * x = 3 - 12 * x^2 ∧ x = Real.rpow (19/2) (1/3) - 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l655_65536


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l655_65545

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 10 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 27132 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l655_65545


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l655_65570

-- Define the number of people and price of goods
variable (x y : ℤ)

-- Define the conditions
def condition1 (x y : ℤ) : Prop := 8 * x - 3 = y
def condition2 (x y : ℤ) : Prop := 7 * x + 4 = y

-- Theorem statement
theorem correct_system_of_equations :
  (∀ x y : ℤ, condition1 x y ∧ condition2 x y →
    (8 * x - 3 = y ∧ 7 * x + 4 = y)) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l655_65570


namespace NUMINAMATH_CALUDE_complex_alpha_value_l655_65548

theorem complex_alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).im = 0)
  (h2 : (α - Complex.I * (3*β - α)).im = 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_alpha_value_l655_65548


namespace NUMINAMATH_CALUDE_equation_solutions_l655_65587

theorem equation_solutions (a : ℝ) : 
  ((∃! x : ℝ, 5 + |x - 2| = a) ∧ (∃ x y : ℝ, x ≠ y ∧ 7 - |2*x + 6| = a ∧ 7 - |2*y + 6| = a) ∨
   (∃ x y : ℝ, x ≠ y ∧ 5 + |x - 2| = a ∧ 5 + |y - 2| = a) ∧ (∃! x : ℝ, 7 - |2*x + 6| = a)) ↔
  (a = 5 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l655_65587


namespace NUMINAMATH_CALUDE_mulch_cost_calculation_l655_65530

-- Define the constants
def tons_of_mulch : ℝ := 3
def price_per_pound : ℝ := 2.5
def pounds_per_ton : ℝ := 2000

-- Define the theorem
theorem mulch_cost_calculation :
  tons_of_mulch * pounds_per_ton * price_per_pound = 15000 := by
  sorry

end NUMINAMATH_CALUDE_mulch_cost_calculation_l655_65530


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l655_65514

-- Define the basic types
variable (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (at_least_parallel_to_one : Line → Plane → Prop)

-- Define the given conditions
variable (a : Line) (β : Plane)

-- State the theorem
theorem parallel_sufficient_not_necessary :
  (∀ (l : Line), parallel a l → in_plane l β → at_least_parallel_to_one a β) ∧
  (∃ (a : Line) (β : Plane), at_least_parallel_to_one a β ∧ ¬(∃ (l : Line), in_plane l β ∧ parallel a l)) :=
sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l655_65514


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l655_65558

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2)
  ∃ (m : ℝ), 
    (a^2 / c)^2 + m^2 = c^2 ∧ 
    2 * c * m = 4 * a * b → 
    (a^2 + b^2) / a^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l655_65558


namespace NUMINAMATH_CALUDE_hypotenuse_length_l655_65549

/-- Represents a right triangle with a 45° angle -/
structure RightTriangle45 where
  leg : ℝ
  hypotenuse : ℝ

/-- The hypotenuse of a right triangle with a 45° angle is √2 times the leg -/
axiom hypotenuse_formula (t : RightTriangle45) : t.hypotenuse = t.leg * Real.sqrt 2

/-- Theorem: In a right triangle with one leg of 10 inches and an opposite angle of 45°,
    the length of the hypotenuse is 10√2 inches -/
theorem hypotenuse_length : 
  let t : RightTriangle45 := { leg := 10, hypotenuse := 10 * Real.sqrt 2 }
  t.hypotenuse = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l655_65549


namespace NUMINAMATH_CALUDE_rational_fraction_value_l655_65554

theorem rational_fraction_value (x y : ℝ) :
  3 < (x - y) / (x + y) →
  (x - y) / (x + y) < 4 →
  ∃ (a b : ℤ), x / y = a / b →
  x + y = 10 →
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_rational_fraction_value_l655_65554


namespace NUMINAMATH_CALUDE_bowling_team_size_l655_65547

theorem bowling_team_size (n : ℕ) (original_avg : ℝ) (new_avg : ℝ) 
  (new_player1_weight : ℝ) (new_player2_weight : ℝ) 
  (h1 : original_avg = 112)
  (h2 : new_player1_weight = 110)
  (h3 : new_player2_weight = 60)
  (h4 : new_avg = 106)
  (h5 : n * original_avg + new_player1_weight + new_player2_weight = (n + 2) * new_avg) :
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_size_l655_65547


namespace NUMINAMATH_CALUDE_peanut_cluster_probability_l655_65519

def total_chocolates : ℕ := 50
def caramels : ℕ := 3
def nougats : ℕ := 2 * caramels
def truffles : ℕ := caramels + 6
def peanut_clusters : ℕ := total_chocolates - (caramels + nougats + truffles)

theorem peanut_cluster_probability : 
  (peanut_clusters : ℚ) / total_chocolates = 32 / 50 := by sorry

end NUMINAMATH_CALUDE_peanut_cluster_probability_l655_65519


namespace NUMINAMATH_CALUDE_triangle_theorem_l655_65504

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  c - a = 1 →
  b = Real.sqrt 7 →
  B = π / 3 ∧
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l655_65504


namespace NUMINAMATH_CALUDE_projectile_max_height_l655_65572

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 36

/-- The maximum height reached by the projectile --/
theorem projectile_max_height :
  ∃ (max : ℝ), max = 100 ∧ ∀ (t : ℝ), h t ≤ max :=
by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l655_65572


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l655_65531

theorem cubic_equation_solution (m : ℝ) (h : m^2 + m - 1 = 0) :
  m^3 + 2*m^2 + 2005 = 2006 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l655_65531


namespace NUMINAMATH_CALUDE_original_number_proof_l655_65511

theorem original_number_proof : 
  ∃! x : ℤ, ∃ y : ℤ, x + y = 859560 ∧ x % 456 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l655_65511


namespace NUMINAMATH_CALUDE_soccer_team_starters_l655_65539

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem soccer_team_starters : 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 220 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l655_65539


namespace NUMINAMATH_CALUDE_min_difference_when_sum_maximized_l655_65565

theorem min_difference_when_sum_maximized :
  ∀ x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℕ+,
    x₁ < x₂ → x₂ < x₃ → x₃ < x₄ → x₄ < x₅ → x₅ < x₆ → x₆ < x₇ → x₇ < x₈ → x₈ < x₉ →
    x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 220 →
    (∀ y₁ y₂ y₃ y₄ y₅ : ℕ+,
      y₁ < y₂ → y₂ < y₃ → y₃ < y₄ → y₄ < y₅ →
      y₁ + y₂ + y₃ + y₄ + y₅ ≤ x₁ + x₂ + x₃ + x₄ + x₅) →
    x₉ - x₁ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_difference_when_sum_maximized_l655_65565


namespace NUMINAMATH_CALUDE_area_of_B_l655_65523

-- Define set A
def A : Set ℝ := {a : ℝ | -1 ≤ a ∧ a ≤ 2}

-- Define set B
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 + p.2 ≥ 0}

-- Theorem statement
theorem area_of_B : MeasureTheory.volume B = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_B_l655_65523


namespace NUMINAMATH_CALUDE_initial_strawberry_weight_l655_65517

/-- The weight of strawberries initially collected by Marco and his dad -/
def initial_weight : ℕ := sorry

/-- The additional weight of strawberries found -/
def additional_weight : ℕ := 30

/-- Marco's strawberry weight after finding more -/
def marco_weight : ℕ := 36

/-- Marco's dad's strawberry weight after finding more -/
def dad_weight : ℕ := 16

/-- Theorem stating that the initial weight of strawberries is 22 pounds -/
theorem initial_strawberry_weight : initial_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_strawberry_weight_l655_65517


namespace NUMINAMATH_CALUDE_sqrt_2_plus_sqrt_2_plus_l655_65593

theorem sqrt_2_plus_sqrt_2_plus : ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (2 + x) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_plus_sqrt_2_plus_l655_65593


namespace NUMINAMATH_CALUDE_curve_transformation_l655_65537

/-- Given a 2x2 matrix A and its inverse, prove that if A transforms a curve F to y = 2x, then F is y = -3x -/
theorem curve_transformation (a b : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; 7, 3]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![b, -2; -7, a]
  A * A_inv = 1 →
  (∀ x y : ℝ, (A.mulVec ![x, y] = ![x', y'] ∧ y' = 2*x') → y = -3*x) :=
by sorry

end NUMINAMATH_CALUDE_curve_transformation_l655_65537


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l655_65567

def n : ℕ := 2025000000

-- Define a function to get the kth largest divisor
def kth_largest_divisor (k : ℕ) (n : ℕ) : ℕ :=
  sorry

theorem fifth_largest_divisor :
  kth_largest_divisor 5 n = 126562500 :=
sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l655_65567


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l655_65588

theorem factorial_ratio_squared : (Nat.factorial 45 / Nat.factorial 43)^2 = 3920400 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l655_65588


namespace NUMINAMATH_CALUDE_inequality_proof_l655_65501

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a + b + 1)) + (1 / (b + c + 1)) + (1 / (a + c + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l655_65501


namespace NUMINAMATH_CALUDE_alyssa_picked_25_limes_l655_65533

/-- The number of limes picked by Alyssa -/
def alyssas_limes : ℕ := 57 - 32

/-- The total number of limes picked -/
def total_limes : ℕ := 57

/-- The number of limes picked by Mike -/
def mikes_limes : ℕ := 32

theorem alyssa_picked_25_limes : alyssas_limes = 25 := by sorry

end NUMINAMATH_CALUDE_alyssa_picked_25_limes_l655_65533


namespace NUMINAMATH_CALUDE_proposition_falsity_l655_65510

theorem proposition_falsity (P : ℕ → Prop) 
  (h_induction : ∀ k : ℕ, k > 0 → P k → P (k + 1))
  (h_false_5 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_falsity_l655_65510


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l655_65563

theorem abs_inequality_solution_set (x : ℝ) :
  |x + 1| - |x - 2| > 1 ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l655_65563


namespace NUMINAMATH_CALUDE_double_reflection_of_D_l655_65559

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define reflection over y-axis
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

-- Define reflection over x-axis
def reflectOverXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

-- Define the composition of reflections
def doubleReflection (p : Point) : Point :=
  reflectOverXAxis (reflectOverYAxis p)

-- Theorem statement
theorem double_reflection_of_D :
  let D : Point := { x := 3, y := 3 }
  doubleReflection D = { x := -3, y := -3 } := by
  sorry

end NUMINAMATH_CALUDE_double_reflection_of_D_l655_65559


namespace NUMINAMATH_CALUDE_smallest_bookmark_count_l655_65578

theorem smallest_bookmark_count (b : ℕ) : 
  (b > 0) →
  (b % 5 = 4) →
  (b % 6 = 3) →
  (b % 8 = 7) →
  (∀ x : ℕ, x > 0 ∧ x % 5 = 4 ∧ x % 6 = 3 ∧ x % 8 = 7 → x ≥ b) →
  b = 39 := by
sorry

end NUMINAMATH_CALUDE_smallest_bookmark_count_l655_65578


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l655_65571

theorem sock_pair_combinations (white brown blue : ℕ) 
  (h_white : white = 5) 
  (h_brown : brown = 5) 
  (h_blue : blue = 2) 
  (h_total : white + brown + blue = 12) : 
  (white.choose 2) + (brown.choose 2) + (blue.choose 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l655_65571


namespace NUMINAMATH_CALUDE_annas_cupcakes_l655_65527

theorem annas_cupcakes (C : ℕ) : 
  (C : ℚ) * (1 / 5) - 3 = 9 → C = 60 := by
  sorry

end NUMINAMATH_CALUDE_annas_cupcakes_l655_65527


namespace NUMINAMATH_CALUDE_basketball_score_formula_l655_65592

/-- Represents the total points scored in a basketball game with specific conditions -/
def total_points (x : ℝ) : ℝ :=
  0.2 * x + 50

theorem basketball_score_formula (x y : ℝ) 
  (h1 : x + y = 50) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  0.4 * x * 3 + 0.5 * y * 2 = total_points x :=
sorry

end NUMINAMATH_CALUDE_basketball_score_formula_l655_65592


namespace NUMINAMATH_CALUDE_system_solution_proof_l655_65594

theorem system_solution_proof :
  ∃! (x y : ℝ), 
    (2 * x + Real.sqrt (2 * x + 3 * y) - 3 * y = 5) ∧ 
    (4 * x^2 + 2 * x + 3 * y - 9 * y^2 = 32) ∧
    (x = 17/4) ∧ (y = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l655_65594


namespace NUMINAMATH_CALUDE_grandmother_age_l655_65557

theorem grandmother_age (M : ℕ) (x : ℕ) :
  (2 * M : ℝ) = M →  -- Number of grandfathers is twice that of grandmothers
  (77 : ℝ) < (M * x + 2 * M * (x - 5)) / (3 * M) →  -- Average age of all pensioners > 77
  (M * x + 2 * M * (x - 5)) / (3 * M) < 78 →  -- Average age of all pensioners < 78
  x = 81 :=
by sorry

end NUMINAMATH_CALUDE_grandmother_age_l655_65557


namespace NUMINAMATH_CALUDE_hyperbola_equation_l655_65568

theorem hyperbola_equation (P Q : ℝ × ℝ) : 
  P = (-3, 2 * Real.sqrt 7) → 
  Q = (-6 * Real.sqrt 2, 7) → 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), (y^2 / a^2) - (x^2 / b^2) = 1 ↔ 
      ((x, y) = P ∨ (x, y) = Q)) ∧
    a^2 = 25 ∧ b^2 = 75 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l655_65568


namespace NUMINAMATH_CALUDE_friendly_parabola_symmetric_l655_65597

/-- Represents a parabola of the form y = ax² + bx --/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Defines the "friendly parabola" relationship between two parabolas --/
def is_friendly_parabola (L₁ L₂ : Parabola) : Prop :=
  L₂.a = -L₁.a ∧ 
  L₂.b = -2 * L₁.a * (-L₁.b / (2 * L₁.a)) + L₁.b

/-- Theorem: The "friendly parabola" relationship is symmetric --/
theorem friendly_parabola_symmetric (L₁ L₂ : Parabola) :
  is_friendly_parabola L₁ L₂ → is_friendly_parabola L₂ L₁ := by
  sorry


end NUMINAMATH_CALUDE_friendly_parabola_symmetric_l655_65597


namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l655_65589

theorem binomial_18_choose_6 : Nat.choose 18 6 = 7280 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l655_65589


namespace NUMINAMATH_CALUDE_inequality_proof_l655_65524

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l655_65524


namespace NUMINAMATH_CALUDE_probability_of_valid_selection_l655_65520

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def valid_selection (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  1 ≤ d ∧ d ≤ 12 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  is_multiple a b ∧ is_multiple b c ∧ is_multiple c d

def count_valid_selections : ℕ := sorry

theorem probability_of_valid_selection :
  (count_valid_selections : ℚ) / (12 * 11 * 10 * 9) = 13 / 845 := by sorry

end NUMINAMATH_CALUDE_probability_of_valid_selection_l655_65520


namespace NUMINAMATH_CALUDE_birthday_crayons_l655_65540

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 134

/-- The number of crayons Paul lost or gave away -/
def crayons_lost : ℕ := 345

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost

theorem birthday_crayons : total_crayons = 479 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l655_65540


namespace NUMINAMATH_CALUDE_sally_balloons_l655_65502

/-- Given the number of blue balloons for Alyssa, Sandy, and the total,
    prove that Sally has the correct number of blue balloons. -/
theorem sally_balloons (alyssa_balloons sandy_balloons total_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : total_balloons = 104) :
  total_balloons - (alyssa_balloons + sandy_balloons) = 39 := by
  sorry

#check sally_balloons

end NUMINAMATH_CALUDE_sally_balloons_l655_65502


namespace NUMINAMATH_CALUDE_equation_solution_l655_65569

theorem equation_solution : ∃ x : ℝ, x * 400 = 173 * 2400 + 125 * 480 / 60 ∧ x = 1039.3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l655_65569


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l655_65599

/-- Given a complex number z and a real number m, 
    if z = (2+3i)(1-mi) is a pure imaginary number, then m = -2/3 -/
theorem pure_imaginary_condition (z : ℂ) (m : ℝ) : 
  z = (2 + 3*Complex.I) * (1 - m*Complex.I) ∧ z.re = 0 → m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l655_65599


namespace NUMINAMATH_CALUDE_smallest_x_solution_l655_65528

theorem smallest_x_solution (w x y z : ℝ) 
  (non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (eq1 : y = x - 2003)
  (eq2 : z = 2*y - 2003)
  (eq3 : w = 3*z - 2003) :
  x ≥ 10015/3 ∧ 
  (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_solution_l655_65528


namespace NUMINAMATH_CALUDE_max_intersection_area_is_zero_l655_65512

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a right prism with an equilateral triangle base -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ
  baseVertices : List Point3D

/-- Calculates the area of intersection between a plane and a right prism -/
def intersectionArea (prism : RightPrism) (plane : Plane) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of intersection is 0 -/
theorem max_intersection_area_is_zero (h : ℝ) (s : ℝ) (A B C : Point3D) :
  h = 5 →
  s = 6 →
  A = ⟨3, 0, 0⟩ →
  B = ⟨-3, 0, 0⟩ →
  C = ⟨0, 3 * Real.sqrt 3, 0⟩ →
  let prism : RightPrism := ⟨h, s, [A, B, C]⟩
  let plane : Plane := ⟨2, -3, 6, 30⟩
  intersectionArea prism plane = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersection_area_is_zero_l655_65512


namespace NUMINAMATH_CALUDE_zero_of_f_l655_65552

def f (x : ℝ) := x + 1

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l655_65552


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_l655_65562

/-- Parabola C: y² = 4x -/
def C (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola C -/
def focus : ℝ × ℝ := (1, 0)

/-- Point M -/
def M : ℝ × ℝ := (0, 2)

/-- Line with slope k passing through the focus -/
def line (k x : ℝ) : ℝ := k*(x - focus.1)

/-- Intersection points of the line and the parabola -/
def intersectionPoints (k : ℝ) : Set (ℝ × ℝ) :=
  {p | C p.1 p.2 ∧ p.2 = line k p.1}

/-- Vector from M to a point P -/
def vector_MP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_intersection_slope (k : ℝ) :
  (∃ A B, A ∈ intersectionPoints k ∧ B ∈ intersectionPoints k ∧ A ≠ B ∧
    dot_product (vector_MP A) (vector_MP B) = 0) →
  k = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_l655_65562


namespace NUMINAMATH_CALUDE_yard_area_l655_65534

/-- Calculates the area of a rectangular yard given the length of one side and the total length of the other three sides. -/
theorem yard_area (side_length : ℝ) (other_sides : ℝ) : 
  side_length = 40 → other_sides = 50 → side_length * ((other_sides - side_length) / 2) = 200 :=
by
  sorry

#check yard_area

end NUMINAMATH_CALUDE_yard_area_l655_65534


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l655_65590

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n % 12 = 0 → digit_sum n = 24 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l655_65590


namespace NUMINAMATH_CALUDE_gcd_lcm_problem_l655_65515

theorem gcd_lcm_problem (x y : ℕ+) (u v : ℕ) : 
  (u = Nat.gcd x y ∧ v = Nat.lcm x y) → 
  (x * y * u * v = 3600 ∧ u + v = 32) → 
  ((x = 6 ∧ y = 10) ∨ (x = 10 ∧ y = 6)) ∧ u = 2 ∧ v = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_problem_l655_65515


namespace NUMINAMATH_CALUDE_initial_apps_correct_l655_65596

/-- The initial number of apps Dave had on his phone -/
def initial_apps : ℕ := 10

/-- The number of apps Dave added -/
def added_apps : ℕ := 11

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 17

/-- The number of apps Dave had left after adding and deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps is correct -/
theorem initial_apps_correct : 
  initial_apps + added_apps - deleted_apps = remaining_apps := by
  sorry

#check initial_apps_correct

end NUMINAMATH_CALUDE_initial_apps_correct_l655_65596


namespace NUMINAMATH_CALUDE_apartment_count_l655_65580

theorem apartment_count
  (num_entrances : ℕ)
  (initial_number : ℕ)
  (new_number : ℕ)
  (h1 : num_entrances = 5)
  (h2 : initial_number = 636)
  (h3 : new_number = 242)
  (h4 : initial_number > new_number) :
  (initial_number - new_number) / 2 * num_entrances = 985 :=
by sorry

end NUMINAMATH_CALUDE_apartment_count_l655_65580
