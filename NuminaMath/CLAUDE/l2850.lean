import Mathlib

namespace NUMINAMATH_CALUDE_leopards_count_l2850_285095

def zoo_problem (leopards : ℕ) : Prop :=
  let snakes := 100
  let arctic_foxes := 80
  let bee_eaters := 10 * leopards
  let cheetahs := snakes / 2
  let alligators := 2 * (arctic_foxes + leopards)
  let total_animals := 670
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators = total_animals

theorem leopards_count : ∃ (l : ℕ), zoo_problem l ∧ l = 20 := by
  sorry

end NUMINAMATH_CALUDE_leopards_count_l2850_285095


namespace NUMINAMATH_CALUDE_state_tax_deduction_l2850_285005

theorem state_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) : 
  hourly_wage = 25 → tax_rate = 0.024 → hourly_wage * tax_rate * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_state_tax_deduction_l2850_285005


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2850_285045

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 + x = 0}

-- Theorem statement
theorem intersection_M_complement_N : M ∩ (Set.univ \ N) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2850_285045


namespace NUMINAMATH_CALUDE_tangent_line_at_negative_one_unique_a_for_inequality_l2850_285036

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- Part I
theorem tangent_line_at_negative_one (h : f 1 (-1) = 0) :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x = m * (x + 1) + f 1 (-1) ∧ m = -2 :=
sorry

-- Part II
theorem unique_a_for_inequality (h : ∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f 1 x ∧ f 1 x ≤ 1/4 * x + 1/4)) :
  ∀ a > 0, (∀ x, x ∈ Set.Icc 0 1 → (1/4 * x - 1/4 ≤ f a x ∧ f a x ≤ 1/4 * x + 1/4)) ↔ a = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_negative_one_unique_a_for_inequality_l2850_285036


namespace NUMINAMATH_CALUDE_roses_planted_over_three_days_l2850_285050

/-- Represents the number of roses planted by each person on a given day -/
structure PlantingData where
  susan : ℕ
  maria : ℕ
  john : ℕ

/-- Calculates the total roses planted on a given day -/
def total_roses (data : PlantingData) : ℕ :=
  data.susan + data.maria + data.john

/-- Represents the planting data for all three days -/
structure ThreeDayPlanting where
  day1 : PlantingData
  day2 : PlantingData
  day3 : PlantingData

theorem roses_planted_over_three_days :
  ∀ (planting : ThreeDayPlanting),
    (planting.day1.susan + planting.day1.maria + planting.day1.john = 50) →
    (planting.day1.maria = 2 * planting.day1.susan) →
    (planting.day1.john = planting.day1.susan + 10) →
    (total_roses planting.day2 = total_roses planting.day1 + 20) →
    (planting.day2.susan * 5 = planting.day1.susan * 7) →
    (planting.day2.maria * 5 = planting.day1.maria * 7) →
    (planting.day2.john * 5 = planting.day1.john * 7) →
    (total_roses planting.day3 = 2 * total_roses planting.day1) →
    (planting.day3.susan = planting.day1.susan) →
    (planting.day3.maria = planting.day1.maria + (planting.day1.maria / 4)) →
    (planting.day3.john = planting.day1.john - (planting.day1.john / 10)) →
    (total_roses planting.day1 + total_roses planting.day2 + total_roses planting.day3 = 173) :=
by sorry

end NUMINAMATH_CALUDE_roses_planted_over_three_days_l2850_285050


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l2850_285094

def valid_pairs : List (Int × Int) := [
  (12, 6), (-2, 6), (12, 4), (-2, 4), (10, 10), (0, 10), (10, 0)
]

theorem fraction_equation_solution (x y : Int) :
  x + y ≠ 0 →
  (x^2 + y^2) / (x + y) = 10 ↔ (x, y) ∈ valid_pairs :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l2850_285094


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2850_285025

/-- The area of a triangle with side lengths 9, 40, and 41 is 180 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 9
    let b : ℝ := 40
    let c : ℝ := 41
    (a^2 + b^2 = c^2) ∧ (area = (1/2) * a * b) ∧ (area = 180)

/-- Proof of the theorem -/
theorem triangle_area_proof : ∃ (area : ℝ), triangle_area area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l2850_285025


namespace NUMINAMATH_CALUDE_E_equals_three_iff_x_equals_y_infinite_solutions_exist_l2850_285047

def E (x y : ℕ) : ℚ :=
  x / y + (x + 1) / (y + 1) + (x + 2) / (y + 2)

theorem E_equals_three_iff_x_equals_y (x y : ℕ) :
  E x y = 3 ↔ x = y :=
sorry

theorem infinite_solutions_exist (k : ℕ) :
  ∃ x y : ℕ, E x y = 11 * k + 3 :=
sorry

end NUMINAMATH_CALUDE_E_equals_three_iff_x_equals_y_infinite_solutions_exist_l2850_285047


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l2850_285051

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- Sum function
  h1 : S 6 = 3
  h2 : a 4 = 2

/-- The fifth term of the arithmetic sequence is 5 -/
theorem fifth_term_is_five (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l2850_285051


namespace NUMINAMATH_CALUDE_min_length_MN_l2850_285093

-- Define the circle
def circle_center : ℝ × ℝ := (1, 1)

-- Define the property of being tangent to x and y axes
def tangent_to_axes (c : ℝ × ℝ) : Prop :=
  c.1 = c.2 ∧ c.1 > 0

-- Define the line MN
def line_MN (m n : ℝ × ℝ) : Prop :=
  m.2 = 0 ∧ n.1 = 0 ∧ m.1 > 0 ∧ n.2 > 0

-- Define the property of MN being tangent to the circle
def tangent_to_circle (m n : ℝ × ℝ) (c : ℝ × ℝ) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1 ∧
              (n.2 - m.2) * (p.1 - m.1) = (n.1 - m.1) * (p.2 - m.2)

-- Theorem statement
theorem min_length_MN :
  tangent_to_axes circle_center →
  ∀ m n : ℝ × ℝ, line_MN m n →
  tangent_to_circle m n circle_center →
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 2 - 2 ∧
  ∀ m' n' : ℝ × ℝ, line_MN m' n' → tangent_to_circle m' n' circle_center →
  Real.sqrt ((m'.1 - n'.1)^2 + (m'.2 - n'.2)^2) ≥ min_length :=
sorry

end NUMINAMATH_CALUDE_min_length_MN_l2850_285093


namespace NUMINAMATH_CALUDE_triangle_side_relation_l2850_285013

/-- Given a triangle ABC with side lengths a, b, c satisfying the equation
    a² - 16b² - c² + 6ab + 10bc = 0, prove that a + c = 2b. -/
theorem triangle_side_relation (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b :=
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l2850_285013


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_l2850_285090

theorem r_fourth_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_l2850_285090


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2850_285091

theorem complex_fraction_calculation : 
  (((13.75 + 9 + 1/6) * 1.2) / ((10.3 - 8 - 1/2) * 5/9) + 
   ((6.8 - 3 - 3/5) * (5 + 5/6)) / ((3 + 2/3 - 3 - 1/6) * 56) - 
   (27 + 1/6)) = 29/3 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2850_285091


namespace NUMINAMATH_CALUDE_edward_initial_amount_l2850_285071

def initial_amount (book_price shirt_price shirt_discount meal_price
                    ticket_price ticket_discount amount_left : ℝ) : ℝ :=
  book_price +
  (shirt_price * (1 - shirt_discount)) +
  meal_price +
  (ticket_price - ticket_discount) +
  amount_left

theorem edward_initial_amount :
  initial_amount 9 25 0.2 15 10 2 17 = 69 := by
  sorry

end NUMINAMATH_CALUDE_edward_initial_amount_l2850_285071


namespace NUMINAMATH_CALUDE_crayons_in_box_l2850_285079

def initial_crayons : ℕ := 7
def added_crayons : ℕ := 6

theorem crayons_in_box : initial_crayons + added_crayons = 13 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_box_l2850_285079


namespace NUMINAMATH_CALUDE_max_value_a_l2850_285074

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 2 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 4460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4460 ∧ 
    b' = 1487 ∧ 
    c' = 744 ∧ 
    d' = 149 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l2850_285074


namespace NUMINAMATH_CALUDE_modern_growth_pattern_l2850_285000

/-- Represents the different types of population growth patterns --/
inductive PopulationGrowthPattern
  | Traditional
  | Modern
  | Primitive
  | TransitionPrimitiveToTraditional

/-- Represents the level of a demographic rate --/
inductive RateLevel
  | Low
  | Medium
  | High

/-- Represents a country --/
structure Country where
  birthRate : RateLevel
  deathRate : RateLevel
  naturalGrowthRate : RateLevel

/-- Determines the population growth pattern of a country --/
def determineGrowthPattern (c : Country) : PopulationGrowthPattern :=
  sorry

theorem modern_growth_pattern (ourCountry : Country) 
  (h1 : ourCountry.birthRate = RateLevel.Low)
  (h2 : ourCountry.deathRate = RateLevel.Low)
  (h3 : ourCountry.naturalGrowthRate = RateLevel.Low) :
  determineGrowthPattern ourCountry = PopulationGrowthPattern.Modern :=
sorry

end NUMINAMATH_CALUDE_modern_growth_pattern_l2850_285000


namespace NUMINAMATH_CALUDE_inequality_proof_l2850_285078

theorem inequality_proof (x y a b : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  Real.sqrt (a^2 * x^2 + b^2 * y^2) + Real.sqrt (a^2 * y^2 + b^2 * x^2) ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2850_285078


namespace NUMINAMATH_CALUDE_quadratic_root_implication_l2850_285032

theorem quadratic_root_implication (a b : ℝ) : 
  (∃ x : ℝ, a * x^2 + b * x + 6 = 0 ∧ x = -2) → 
  6 * a - 3 * b + 6 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implication_l2850_285032


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2850_285072

theorem inequality_equivalence (x : ℝ) : 3 - 2 / (3 * x + 2) < 5 ↔ x > -2/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2850_285072


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2850_285089

/-- Given a geometric sequence {a_n}, if a_1 + a_2 = 20 and a_3 + a_4 = 80, then a_5 + a_6 = 320 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 + a 2 = 20 →                           -- first condition
  a 3 + a 4 = 80 →                           -- second condition
  a 5 + a 6 = 320 := by                      -- conclusion
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2850_285089


namespace NUMINAMATH_CALUDE_complex_angle_of_one_plus_i_sqrt_three_l2850_285086

theorem complex_angle_of_one_plus_i_sqrt_three :
  let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_angle_of_one_plus_i_sqrt_three_l2850_285086


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l2850_285019

-- Define constants
def jane_start_age : ℕ := 16
def jane_current_age : ℕ := 32
def years_since_stopped : ℕ := 10

-- Define the theorem
theorem oldest_babysat_age :
  ∀ (oldest_age : ℕ),
  (oldest_age = (jane_current_age - years_since_stopped) / 2 + years_since_stopped) →
  (oldest_age ≤ jane_current_age) →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_start_age ≤ jane_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2 →
    child_age + (jane_current_age - jane_age) ≤ oldest_age) →
  oldest_age = 21 :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_age_l2850_285019


namespace NUMINAMATH_CALUDE_expression_evaluation_l2850_285030

theorem expression_evaluation : 
  68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2850_285030


namespace NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2850_285004

/-- The distance between an endpoint of the major axis and an endpoint of the minor axis of the ellipse 16(x+2)^2 + 4y^2 = 64 is 2√5. -/
theorem ellipse_axis_endpoint_distance :
  ∃ (A' B' : ℝ × ℝ),
    (∀ (x y : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1})) ∧
    A' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ p.2^2 = 16} ∧
    B' ∈ {p : ℝ × ℝ | (p.1 + 2)^2 / 4 + p.2^2 / 16 = 1 ∧ (p.1 + 2)^2 = 4} ∧
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_axis_endpoint_distance_l2850_285004


namespace NUMINAMATH_CALUDE_distance_to_AB_l2850_285060

/-- Triangle ABC with point M inside -/
structure TriangleWithPoint where
  -- Define the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the distances from M to sides
  distMAC : ℝ
  distMBC : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  distMAC_positive : distMAC > 0
  distMBC_positive : distMBC > 0
  -- Triangle inequality
  triangle_inequality : AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB
  -- M is inside the triangle
  M_inside : distMAC < AC ∧ distMBC < BC

/-- The theorem to be proved -/
theorem distance_to_AB (t : TriangleWithPoint) 
  (h1 : t.AB = 10) 
  (h2 : t.BC = 17) 
  (h3 : t.AC = 21) 
  (h4 : t.distMAC = 2) 
  (h5 : t.distMBC = 4) : 
  ∃ (distMAB : ℝ), distMAB = 29 / 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_AB_l2850_285060


namespace NUMINAMATH_CALUDE_centroid_altitude_intersection_ratio_l2850_285088

-- Define an orthocentric tetrahedron
structure OrthocentricTetrahedron where
  -- Add necessary fields

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a sphere in 3D space
structure Sphere where
  center : Point3D
  radius : ℝ

def centroid (t : OrthocentricTetrahedron) : Point3D :=
  sorry

def altitudeFoot (t : OrthocentricTetrahedron) : Point3D :=
  sorry

def circumscribedSphere (t : OrthocentricTetrahedron) : Sphere :=
  sorry

def lineIntersectSphere (l : Line3D) (s : Sphere) : Point3D :=
  sorry

def distance (p1 p2 : Point3D) : ℝ :=
  sorry

def pointBetween (p1 p2 p3 : Point3D) : Prop :=
  sorry

theorem centroid_altitude_intersection_ratio 
  (t : OrthocentricTetrahedron)
  (G : Point3D)
  (F : Point3D)
  (K : Point3D) :
  G = centroid t →
  F = altitudeFoot t →
  K = lineIntersectSphere (Line3D.mk F (centroid t)) (circumscribedSphere t) →
  pointBetween K G F →
  distance K G = 3 * distance F G :=
sorry

end NUMINAMATH_CALUDE_centroid_altitude_intersection_ratio_l2850_285088


namespace NUMINAMATH_CALUDE_adam_apples_solution_l2850_285015

/-- Adam's apple purchases over three days --/
def adam_apples (monday_quantity : ℕ) (tuesday_multiple : ℕ) (wednesday_multiple : ℕ) : Prop :=
  let tuesday_quantity := monday_quantity * tuesday_multiple
  let wednesday_quantity := tuesday_quantity * wednesday_multiple
  monday_quantity + tuesday_quantity + wednesday_quantity = 240

theorem adam_apples_solution :
  ∃ (wednesday_multiple : ℕ),
    adam_apples 15 3 wednesday_multiple ∧ wednesday_multiple = 4 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_solution_l2850_285015


namespace NUMINAMATH_CALUDE_mike_sold_45_books_l2850_285028

/-- The number of books Mike sold at the garage sale -/
def books_sold (initial_books current_books : ℕ) : ℕ :=
  initial_books - current_books

/-- Proof that Mike sold 45 books -/
theorem mike_sold_45_books (h1 : books_sold 51 6 = 45) : books_sold 51 6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_mike_sold_45_books_l2850_285028


namespace NUMINAMATH_CALUDE_gold_alloy_calculation_l2850_285097

theorem gold_alloy_calculation (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
  (target_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 16 →
  initial_gold_percentage = 0.5 →
  target_gold_percentage = 0.8 →
  added_gold = 24 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = target_gold_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_gold_alloy_calculation_l2850_285097


namespace NUMINAMATH_CALUDE_jake_peaches_count_l2850_285042

-- Define the number of apples and peaches for Steven and Jake
def steven_apples : ℕ := 52
def steven_peaches : ℕ := 13
def jake_apples : ℕ := steven_apples + 84
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_count_l2850_285042


namespace NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l2850_285006

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def S (a₁ : ℝ) : Set ℝ := {x | ∃ n : ℕ+, x = Real.cos (arithmetic_sequence a₁ (2 * Real.pi / 3) n)}

theorem cos_arithmetic_sequence_product (a₁ : ℝ) :
  ∃ a b : ℝ, S a₁ = {a, b} → a * b = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_arithmetic_sequence_product_l2850_285006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_quadrant_l2850_285010

def arithmetic_sequence (n : ℕ) : ℚ := 1 - (n - 1) * (1 / 2)

def intersection_x (a_n : ℚ) : ℚ := (a_n + 1) / 3

def intersection_y (a_n : ℚ) : ℚ := (8 * a_n - 1) / 3

theorem arithmetic_sequence_fourth_quadrant :
  ∀ n : ℕ, n > 0 →
  (intersection_x (arithmetic_sequence n) > 0 ∧ 
   intersection_y (arithmetic_sequence n) < 0) →
  (n = 3 ∨ n = 4) ∧ 
  arithmetic_sequence n = -1/2 * n + 3/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_quadrant_l2850_285010


namespace NUMINAMATH_CALUDE_wages_comparison_l2850_285067

theorem wages_comparison (E R C : ℝ) 
  (hC_E : C = E * 1.7)
  (hC_R : C = R * 1.3076923076923077) :
  R = E * 1.3 :=
by sorry

end NUMINAMATH_CALUDE_wages_comparison_l2850_285067


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2850_285073

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle1 B.1 B.2) ∧
  (circle2 A.1 A.2 ∧ circle2 B.1 B.2) ∧
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2850_285073


namespace NUMINAMATH_CALUDE_postage_calculation_l2850_285039

/-- Calculates the postage cost for a letter given its weight and rate structure -/
def calculate_postage (weight : ℚ) (base_rate : ℕ) (additional_rate : ℕ) : ℚ :=
  base_rate + additional_rate * (⌈weight - 1⌉ : ℚ)

/-- The postage for a 5.75 ounce letter is $1.00 given the specified rates -/
theorem postage_calculation :
  let weight : ℚ := 5.75
  let base_rate : ℕ := 25
  let additional_rate : ℕ := 15
  calculate_postage weight base_rate additional_rate = 100 := by
sorry

#eval calculate_postage (5.75 : ℚ) 25 15

end NUMINAMATH_CALUDE_postage_calculation_l2850_285039


namespace NUMINAMATH_CALUDE_probability_no_shaded_square_correct_l2850_285029

/-- The probability of a randomly chosen rectangle not including a shaded square
    in a 2 by 2001 rectangle with the middle unit square of each row shaded. -/
def probability_no_shaded_square : ℚ :=
  1001 / 2001

/-- The number of columns in the rectangle. -/
def num_columns : ℕ := 2001

/-- The number of rows in the rectangle. -/
def num_rows : ℕ := 2

/-- The total number of rectangles that can be formed in a single row. -/
def rectangles_per_row : ℕ := (num_columns + 1).choose 2

/-- The number of rectangles in a single row that include the shaded square. -/
def shaded_rectangles_per_row : ℕ := (num_columns + 1) / 2 * (num_columns / 2)

theorem probability_no_shaded_square_correct :
  probability_no_shaded_square = 1 - (3 * shaded_rectangles_per_row) / (3 * rectangles_per_row) :=
sorry

end NUMINAMATH_CALUDE_probability_no_shaded_square_correct_l2850_285029


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l2850_285061

theorem smallest_fraction_between (p q : ℕ+) : 
  (6 : ℚ) / 11 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (5 : ℚ) / 9 ∧ 
  (∀ (p' q' : ℕ+), (6 : ℚ) / 11 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (5 : ℚ) / 9 → q ≤ q') →
  q.val - p.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l2850_285061


namespace NUMINAMATH_CALUDE_systematic_sampling_validity_l2850_285037

/-- Represents a set of student numbers -/
def StudentSet : Type := List Nat

/-- Checks if a list of natural numbers is arithmetic progression with common difference d -/
def isArithmeticProgression (l : List Nat) (d : Nat) : Prop :=
  l.zipWith (· - ·) (l.tail) = List.replicate (l.length - 1) d

/-- Checks if a set of student numbers is a valid systematic sample -/
def isValidSystematicSample (s : StudentSet) (totalStudents numSelected : Nat) : Prop :=
  s.length = numSelected ∧
  s.all (· ≤ totalStudents) ∧
  isArithmeticProgression s (totalStudents / numSelected)

theorem systematic_sampling_validity :
  let totalStudents : Nat := 50
  let numSelected : Nat := 5
  let sampleSet : StudentSet := [6, 16, 26, 36, 46]
  isValidSystematicSample sampleSet totalStudents numSelected := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_validity_l2850_285037


namespace NUMINAMATH_CALUDE_p_true_q_false_range_l2850_285070

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*x₀ - m - 1 < 0

def q (m : ℝ) : Prop := ∀ x ∈ Set.Icc 1 4, x + 4/x > m

-- Theorem statement
theorem p_true_q_false_range (m : ℝ) :
  p m ∧ ¬(q m) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_p_true_q_false_range_l2850_285070


namespace NUMINAMATH_CALUDE_square_of_powers_l2850_285014

theorem square_of_powers (n : ℕ) : 
  (∃ k : ℕ, 2^10 + 2^13 + 2^14 + 3 * 2^n = k^2) ↔ n = 13 ∨ n = 15 := by
sorry

end NUMINAMATH_CALUDE_square_of_powers_l2850_285014


namespace NUMINAMATH_CALUDE_function_value_at_ln_half_l2850_285087

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (5^x) / (5^x + 1)

theorem function_value_at_ln_half (a : ℝ) :
  (f a (Real.log 2) = 4) → (f a (Real.log (1/2)) = -3) := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_ln_half_l2850_285087


namespace NUMINAMATH_CALUDE_cake_recipe_flour_l2850_285083

/-- The amount of flour required for Mary's cake recipe --/
def flour_required (sugar : ℕ) (flour_sugar_diff : ℕ) (flour_added : ℕ) : ℕ :=
  sugar + flour_sugar_diff

theorem cake_recipe_flour :
  let sugar := 3
  let flour_sugar_diff := 5
  let flour_added := 2
  flour_required sugar flour_sugar_diff flour_added = 8 := by
  sorry

end NUMINAMATH_CALUDE_cake_recipe_flour_l2850_285083


namespace NUMINAMATH_CALUDE_equation_solution_l2850_285034

theorem equation_solution (y : ℚ) : 
  (y ≠ 5) → (y ≠ (3/2 : ℚ)) → 
  ((y^2 - 12*y + 35) / (y - 5) + (2*y^2 + 9*y - 18) / (2*y - 3) = 0) → 
  y = (1/2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2850_285034


namespace NUMINAMATH_CALUDE_stick_division_theorem_l2850_285046

/-- Represents a stick with markings -/
structure MarkedStick where
  divisions : List Nat

/-- Calculates the number of pieces a stick is divided into when cut at all markings -/
def numberOfPieces (stick : MarkedStick) : Nat :=
  sorry

/-- The theorem to be proven -/
theorem stick_division_theorem :
  let stick : MarkedStick := { divisions := [10, 12, 15] }
  numberOfPieces stick = 28 := by
  sorry

end NUMINAMATH_CALUDE_stick_division_theorem_l2850_285046


namespace NUMINAMATH_CALUDE_first_player_wins_l2850_285077

/-- Represents the game state -/
structure GameState where
  m : Nat
  n : Nat

/-- Represents a move in the game -/
structure Move where
  row : Nat
  col : Nat

/-- Determines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  1 ≤ move.row ∧ move.row ≤ state.m ∧ 1 ≤ move.col ∧ move.col ≤ state.n

/-- Applies a move to a game state, returning the new state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { m := move.row - 1, n := move.col - 1 }

/-- Determines if a game state is terminal (i.e., only the losing square remains) -/
def isTerminal (state : GameState) : Prop :=
  state.m = 1 ∧ state.n = 1

/-- Theorem: The first player has a winning strategy in the chocolate bar game -/
theorem first_player_wins (initialState : GameState) : 
  initialState.m ≥ 1 ∧ initialState.n ≥ 1 → 
  ∃ (strategy : GameState → Move), 
    (∀ (state : GameState), isValidMove state (strategy state)) ∧ 
    (∀ (state : GameState), ¬isTerminal state → 
      ¬∃ (counterStrategy : GameState → Move), 
        (∀ (s : GameState), isValidMove s (counterStrategy s)) ∧
        isTerminal (applyMove (applyMove state (strategy state)) (counterStrategy (applyMove state (strategy state))))) :=
by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l2850_285077


namespace NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2850_285069

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 1 < x ∧ x < 9)
  (h2 : 2 < x ∧ x < 15)
  (h3 : -1 < x ∧ x < 7)
  (h4 : 0 < x ∧ x < 4)
  (h5 : x + 1 < 5) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_satisfying_conditions_l2850_285069


namespace NUMINAMATH_CALUDE_expand_binomials_l2850_285009

theorem expand_binomials (x : ℝ) : (2*x - 3) * (4*x + 5) = 8*x^2 - 2*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l2850_285009


namespace NUMINAMATH_CALUDE_trekking_meals_theorem_l2850_285022

/-- Represents the number of meals available for children given the trekking group scenario -/
def meals_for_children (total_adults : ℕ) (total_children : ℕ) (max_adult_meals : ℕ) 
  (adults_eaten : ℕ) (children_fed_with_remainder : ℕ) : ℕ :=
  2 * children_fed_with_remainder

/-- Theorem stating that the number of meals initially available for children is 90 -/
theorem trekking_meals_theorem (total_adults : ℕ) (total_children : ℕ) (max_adult_meals : ℕ) 
  (adults_eaten : ℕ) (children_fed_with_remainder : ℕ) :
  total_adults = 55 →
  total_children = 70 →
  max_adult_meals = 70 →
  adults_eaten = 35 →
  children_fed_with_remainder = 45 →
  meals_for_children total_adults total_children max_adult_meals adults_eaten children_fed_with_remainder = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_trekking_meals_theorem_l2850_285022


namespace NUMINAMATH_CALUDE_minimize_xy_sum_l2850_285076

theorem minimize_xy_sum (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 11) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 11 → x * y ≤ a * b) →
  x * y = 176 ∧ x + y = 30 :=
sorry

end NUMINAMATH_CALUDE_minimize_xy_sum_l2850_285076


namespace NUMINAMATH_CALUDE_divisor_problem_l2850_285023

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 997 →
  quotient = 43 →
  remainder = 8 →
  dividend = divisor * quotient + remainder →
  Nat.Prime divisor →
  divisor = 23 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2850_285023


namespace NUMINAMATH_CALUDE_test_pumping_result_l2850_285024

/-- Calculates the total gallons pumped during a test with two pumps -/
def total_gallons_pumped (pump1_rate : ℝ) (pump2_rate : ℝ) (total_time : ℝ) (pump2_time : ℝ) : ℝ :=
  let pump1_time := total_time - pump2_time
  pump1_rate * pump1_time + pump2_rate * pump2_time

/-- Proves that the total gallons pumped is 1325 given the specified conditions -/
theorem test_pumping_result :
  total_gallons_pumped 180 250 6 3.5 = 1325 := by
  sorry

end NUMINAMATH_CALUDE_test_pumping_result_l2850_285024


namespace NUMINAMATH_CALUDE_polynomial_shift_representation_l2850_285020

theorem polynomial_shift_representation (f : Polynomial ℝ) (x₀ : ℝ) :
  ∃! g : Polynomial ℝ, ∀ x, f.eval x = g.eval (x - x₀) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_shift_representation_l2850_285020


namespace NUMINAMATH_CALUDE_daragh_favorite_bears_l2850_285066

/-- The number of stuffed bears Daragh had initially -/
def initial_bears : ℕ := 20

/-- The number of sisters Daragh divided bears among -/
def num_sisters : ℕ := 3

/-- The number of bears Eden had before receiving more -/
def eden_initial_bears : ℕ := 10

/-- The number of bears Eden has after receiving more -/
def eden_final_bears : ℕ := 14

/-- The number of favorite stuffed bears Daragh took out -/
def favorite_bears : ℕ := initial_bears - (eden_final_bears - eden_initial_bears) * num_sisters

theorem daragh_favorite_bears :
  favorite_bears = 8 :=
by sorry

end NUMINAMATH_CALUDE_daragh_favorite_bears_l2850_285066


namespace NUMINAMATH_CALUDE_expansion_dissimilar_terms_l2850_285054

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^10 -/
def dissimilarTerms : ℕ := 286

/-- The number of variables in the expansion -/
def numVariables : ℕ := 4

/-- The exponent in the expansion -/
def exponent : ℕ := 10

/-- Theorem: The number of dissimilar terms in (a + b + c + d)^10 is 286 -/
theorem expansion_dissimilar_terms :
  dissimilarTerms = (numVariables + exponent - 1).choose (numVariables - 1) :=
sorry

end NUMINAMATH_CALUDE_expansion_dissimilar_terms_l2850_285054


namespace NUMINAMATH_CALUDE_kolya_is_wrong_l2850_285085

/-- Represents the statements made by each boy -/
structure Statements where
  vasya : ℕ → Prop
  kolya : ℕ → Prop
  petya : ℕ → ℕ → Prop
  misha : ℕ → ℕ → Prop

/-- The actual statements made by the boys -/
def boys_statements : Statements where
  vasya := λ b => b ≥ 4
  kolya := λ g => g ≥ 5
  petya := λ b g => b ≥ 3 ∧ g ≥ 4
  misha := λ b g => b ≥ 4 ∧ g ≥ 4

/-- Theorem stating that Kolya's statement is the only one that can be false -/
theorem kolya_is_wrong (s : Statements) (b g : ℕ) :
  s = boys_statements →
  (s.vasya b ∧ s.petya b g ∧ s.misha b g ∧ ¬s.kolya g) ↔
  (b ≥ 4 ∧ g = 4) :=
sorry

end NUMINAMATH_CALUDE_kolya_is_wrong_l2850_285085


namespace NUMINAMATH_CALUDE_abc_relationship_l2850_285099

open Real

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (cos (34 * π / 180) - sin (34 * π / 180))

noncomputable def b : ℝ := cos (50 * π / 180) * cos (128 * π / 180) + cos (40 * π / 180) * cos (38 * π / 180)

noncomputable def c : ℝ := (1 / 2) * (cos (80 * π / 180) - 2 * (cos (50 * π / 180))^2 + 1)

theorem abc_relationship : b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_abc_relationship_l2850_285099


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2850_285056

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^8 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2850_285056


namespace NUMINAMATH_CALUDE_min_value_4a_plus_b_l2850_285001

theorem min_value_4a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  4*a + b ≥ 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀^2 + a₀*b₀ - 3 = 0 ∧ 4*a₀ + b₀ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_4a_plus_b_l2850_285001


namespace NUMINAMATH_CALUDE_twenty_one_numbers_inequality_l2850_285048

theorem twenty_one_numbers_inequality (S : Finset ℕ) : 
  S ⊆ Finset.range 2047 →
  S.card = 21 →
  ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (b * c : ℝ) < 2 * (a^2 : ℝ) ∧ 2 * (a^2 : ℝ) < 4 * (b * c : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_twenty_one_numbers_inequality_l2850_285048


namespace NUMINAMATH_CALUDE_lily_painting_time_l2850_285043

/-- The time it takes to paint a lily -/
def time_for_lily : ℕ := sorry

/-- The time it takes to paint a rose -/
def time_for_rose : ℕ := 7

/-- The time it takes to paint an orchid -/
def time_for_orchid : ℕ := 3

/-- The time it takes to paint a vine -/
def time_for_vine : ℕ := 2

/-- The total time taken to paint all flowers and vines -/
def total_time : ℕ := 213

/-- The number of lilies painted -/
def num_lilies : ℕ := 17

/-- The number of roses painted -/
def num_roses : ℕ := 10

/-- The number of orchids painted -/
def num_orchids : ℕ := 6

/-- The number of vines painted -/
def num_vines : ℕ := 20

theorem lily_painting_time : time_for_lily = 5 := by
  sorry

end NUMINAMATH_CALUDE_lily_painting_time_l2850_285043


namespace NUMINAMATH_CALUDE_joe_marshmallow_fraction_l2850_285011

theorem joe_marshmallow_fraction :
  let dad_marshmallows : ℕ := 21
  let joe_marshmallows : ℕ := 4 * dad_marshmallows
  let dad_roasted : ℕ := dad_marshmallows / 3
  let total_roasted : ℕ := 49
  let joe_roasted : ℕ := total_roasted - dad_roasted
  joe_roasted / joe_marshmallows = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_joe_marshmallow_fraction_l2850_285011


namespace NUMINAMATH_CALUDE_theresas_work_hours_l2850_285038

theorem theresas_work_hours : ∀ (final_week_hours : ℕ),
  final_week_hours ≥ 10 →
  (7 + 10 + 8 + 11 + 9 + 7 + final_week_hours) / 7 = 9 →
  final_week_hours = 11 := by
sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l2850_285038


namespace NUMINAMATH_CALUDE_integer_root_values_l2850_285021

def polynomial (a x : ℤ) : ℤ := x^3 - 2*x^2 + a*x + 8

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial a x = 0

theorem integer_root_values : 
  {a : ℤ | has_integer_root a} = {-49, -47, -22, -10, -7, 4, 9, 16} := by sorry

end NUMINAMATH_CALUDE_integer_root_values_l2850_285021


namespace NUMINAMATH_CALUDE_product_pure_imaginary_implies_magnitude_l2850_285012

open Complex

theorem product_pure_imaginary_implies_magnitude (b : ℝ) :
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).re = 0 ∧
  (((2 : ℂ) + b * I) * ((1 : ℂ) - I)).im ≠ 0 →
  abs ((1 : ℂ) + b * I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_implies_magnitude_l2850_285012


namespace NUMINAMATH_CALUDE_opposite_of_negative_1009_opposite_of_negative_1009_proof_l2850_285096

theorem opposite_of_negative_1009 : Int → Prop :=
  fun x => x + (-1009) = 0 → x = 1009

-- The proof is omitted
theorem opposite_of_negative_1009_proof : opposite_of_negative_1009 1009 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_1009_opposite_of_negative_1009_proof_l2850_285096


namespace NUMINAMATH_CALUDE_billy_experiment_result_l2850_285035

/-- Represents the mouse population dynamics in Billy's experiment --/
structure MousePopulation where
  initial_mice : ℕ
  pups_per_mouse : ℕ
  final_population : ℕ

/-- Calculates the number of pups eaten per adult mouse --/
def pups_eaten_per_adult (pop : MousePopulation) : ℕ :=
  let first_gen_total := pop.initial_mice + pop.initial_mice * pop.pups_per_mouse
  let second_gen_total := first_gen_total + first_gen_total * pop.pups_per_mouse
  let total_eaten := second_gen_total - pop.final_population
  total_eaten / first_gen_total

/-- Theorem stating that in Billy's experiment, each adult mouse ate 2 pups --/
theorem billy_experiment_result :
  let pop : MousePopulation := {
    initial_mice := 8,
    pups_per_mouse := 6,
    final_population := 280
  }
  pups_eaten_per_adult pop = 2 := by
  sorry


end NUMINAMATH_CALUDE_billy_experiment_result_l2850_285035


namespace NUMINAMATH_CALUDE_sqrt_two_division_l2850_285075

theorem sqrt_two_division : 2 * Real.sqrt 2 / Real.sqrt 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_division_l2850_285075


namespace NUMINAMATH_CALUDE_geometric_sum_first_eight_l2850_285063

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_eight :
  let a : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 3280/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_eight_l2850_285063


namespace NUMINAMATH_CALUDE_student_distribution_l2850_285002

theorem student_distribution (total : ℝ) (third_year : ℝ) (second_year : ℝ)
  (h1 : third_year = 0.5 * total)
  (h2 : second_year = 0.3 * total)
  (h3 : total > 0) :
  second_year / (total - third_year) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_student_distribution_l2850_285002


namespace NUMINAMATH_CALUDE_problem_solution_l2850_285068

noncomputable section

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = A ∪ B a → a = 1) ∧
  (∀ a : ℝ, A ∩ B a = B a → a ≤ -1 ∨ a = 1) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2850_285068


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2850_285031

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 1300000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { coefficient := 1.3
    exponent := 6
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2850_285031


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2850_285064

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / (a * b / c) ≤ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2850_285064


namespace NUMINAMATH_CALUDE_lives_gained_l2850_285008

theorem lives_gained (initial_lives lost_lives final_lives : ℕ) :
  initial_lives = 14 →
  lost_lives = 4 →
  final_lives = 46 →
  final_lives - (initial_lives - lost_lives) = 36 := by
sorry

end NUMINAMATH_CALUDE_lives_gained_l2850_285008


namespace NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l2850_285027

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 500! has 124 trailing zeros -/
theorem factorial_500_trailing_zeros :
  trailingZeros 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_factorial_500_trailing_zeros_l2850_285027


namespace NUMINAMATH_CALUDE_rectangle_EF_length_l2850_285082

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  DF : ℝ
  EF : ℝ
  h_AB : AB = 4
  h_BC : BC = 10
  h_DE_DF : DE = DF
  h_area : DE * DF / 2 = AB * BC / 4

/-- The length of EF in the given rectangle -/
theorem rectangle_EF_length (r : Rectangle) : r.EF = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_EF_length_l2850_285082


namespace NUMINAMATH_CALUDE_trig_problem_l2850_285084

theorem trig_problem (α : Real) (h : Real.tan α = 2) : 
  (2 * Real.sin α ^ 2 + 1) / Real.cos (2 * (α - π/4)) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l2850_285084


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2850_285057

theorem smallest_k_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l2850_285057


namespace NUMINAMATH_CALUDE_average_flux_1_to_999_l2850_285055

/-- The flux of a positive integer is the number of times the digits change from increasing to decreasing or vice versa, ignoring consecutive equal digits. -/
def flux (n : ℕ+) : ℕ := sorry

/-- The sum of fluxes for all positive integers from 1 to 999, inclusive. -/
def sum_of_fluxes : ℕ := sorry

theorem average_flux_1_to_999 :
  (sum_of_fluxes : ℚ) / 999 = 175 / 333 := by sorry

end NUMINAMATH_CALUDE_average_flux_1_to_999_l2850_285055


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2850_285053

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 3 / 2 + Real.sqrt 2) ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ 1 / x₀ + 1 / y₀ = 3 / 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2850_285053


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2850_285044

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let downDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundFactor^i)
  let upDistances := List.range bounces |>.map (fun i => initialHeight * reboundFactor^(i+1))
  (downDistances.sum + upDistances.sum)

/-- The total distance traveled by a ball dropped from 150 feet, rebounding 1/3 of its fall distance each time, after 5 bounces is equal to 298.14 feet -/
theorem ball_bounce_distance :
  totalDistance 150 (1/3) 5 = 298.14 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l2850_285044


namespace NUMINAMATH_CALUDE_range_and_minimum_value_l2850_285033

def f (a x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + (a^2 - 2 * a + 2)

theorem range_and_minimum_value (a : ℝ) :
  (a = 1 → Set.range (fun x => f 1 x) ∩ Set.Icc 0 2 = Set.Icc 0 9) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 2 → f a x ≤ f a y) →
  (∀ (x : ℝ), x ∈ Set.Icc 0 2 → f a x ≥ 3) →
  (a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10) :=
sorry

end NUMINAMATH_CALUDE_range_and_minimum_value_l2850_285033


namespace NUMINAMATH_CALUDE_sum_of_squares_is_two_l2850_285080

/-- Given a 2x2 matrix A, if its transpose is equal to its inverse, then the sum of squares of its elements is 2. -/
theorem sum_of_squares_is_two (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.transpose = A⁻¹) → a^2 + b^2 + c^2 + d^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_two_l2850_285080


namespace NUMINAMATH_CALUDE_probability_two_boys_l2850_285065

theorem probability_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 15 →
  boys = 8 →
  girls = 7 →
  boys + girls = total →
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2 : ℚ) = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_two_boys_l2850_285065


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l2850_285081

/-- Represents a repeating decimal where the digits 56 repeat infinitely after the decimal point. -/
def repeating_decimal : ℚ := sorry

theorem repeating_decimal_fraction : repeating_decimal = 56 / 99 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l2850_285081


namespace NUMINAMATH_CALUDE_range_of_2m_plus_n_l2850_285016

noncomputable def f (x : ℝ) := |Real.log x / Real.log 3|

theorem range_of_2m_plus_n (m n : ℝ) (h1 : 0 < m) (h2 : m < n) (h3 : f m = f n) :
  ∃ (lower : ℝ), lower = 2 * Real.sqrt 2 ∧
  (∀ x, x ≥ lower ↔ ∃ (m' n' : ℝ), 0 < m' ∧ m' < n' ∧ f m' = f n' ∧ 2 * m' + n' = x) :=
sorry

end NUMINAMATH_CALUDE_range_of_2m_plus_n_l2850_285016


namespace NUMINAMATH_CALUDE_gcd_455_299_l2850_285062

theorem gcd_455_299 : Nat.gcd 455 299 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_455_299_l2850_285062


namespace NUMINAMATH_CALUDE_correct_sum_after_reversing_tens_digit_l2850_285059

/-- Represents a three-digit number with digits a, b, and c --/
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- Represents the same number with tens digit reversed --/
def reversed_tens_digit (a b c : ℕ) : ℕ := 100 * a + 10 * c + b

theorem correct_sum_after_reversing_tens_digit 
  (m n : ℕ) 
  (a b c : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m = three_digit_number a b c) 
  (h4 : reversed_tens_digit a b c + n = 128) :
  m + n = 128 := by
sorry

end NUMINAMATH_CALUDE_correct_sum_after_reversing_tens_digit_l2850_285059


namespace NUMINAMATH_CALUDE_factor_sum_l2850_285040

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) →
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2850_285040


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l2850_285058

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 9/y = 1) :
  xy ≥ 36 ∧ x + 2*y ≥ 19 + 6*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l2850_285058


namespace NUMINAMATH_CALUDE_y_decreasing_order_l2850_285017

-- Define the linear function
def f (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Define the theorem
theorem y_decreasing_order (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-2) b = y₁)
  (h₂ : f (-1) b = y₂)
  (h₃ : f 1 b = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_decreasing_order_l2850_285017


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2850_285052

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = 64 ∧ d = s * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2850_285052


namespace NUMINAMATH_CALUDE_area_to_paint_is_128_l2850_285003

/-- The area of a rectangle given its height and width -/
def rectangleArea (height width : ℝ) : ℝ := height * width

/-- The area to be painted on a wall with a window and a door -/
def areaToPaint (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) : ℝ :=
  rectangleArea wallHeight wallWidth - rectangleArea windowHeight windowWidth - rectangleArea doorHeight doorWidth

/-- Theorem stating that the area to be painted is 128 square feet -/
theorem area_to_paint_is_128 (wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth : ℝ) :
  wallHeight = 10 ∧ wallWidth = 15 ∧ 
  windowHeight = 3 ∧ windowWidth = 5 ∧ 
  doorHeight = 1 ∧ doorWidth = 7 →
  areaToPaint wallHeight wallWidth windowHeight windowWidth doorHeight doorWidth = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_128_l2850_285003


namespace NUMINAMATH_CALUDE_binomial_1293_2_l2850_285092

theorem binomial_1293_2 : Nat.choose 1293 2 = 835218 := by sorry

end NUMINAMATH_CALUDE_binomial_1293_2_l2850_285092


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l2850_285098

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l2850_285098


namespace NUMINAMATH_CALUDE_line_vector_proof_l2850_285049

def line_vector (t : ℝ) : ℝ × ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 0 = (2, 5, 9)) →
  (line_vector 1 = (3, 3, 5)) →
  (line_vector (-1) = (1, 7, 13)) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l2850_285049


namespace NUMINAMATH_CALUDE_lucky_larry_calculation_l2850_285007

theorem lucky_larry_calculation (a b c d e : ℚ) : 
  a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 6 →
  a * b + c * d - c * e + 1 = a * (b + (c * (d - e))) →
  e = 23/4 := by
sorry

end NUMINAMATH_CALUDE_lucky_larry_calculation_l2850_285007


namespace NUMINAMATH_CALUDE_smallest_number_with_property_l2850_285026

theorem smallest_number_with_property : ∃ n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  n % 10 = 9 ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    m % 2 ≠ 1 ∨
    m % 3 ≠ 2 ∨
    m % 4 ≠ 3 ∨
    m % 5 ≠ 4 ∨
    m % 6 ≠ 5 ∨
    m % 7 ≠ 6 ∨
    m % 8 ≠ 7 ∨
    m % 9 ≠ 8 ∨
    m % 10 ≠ 9) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_property_l2850_285026


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2850_285041

theorem fractional_equation_solution : 
  ∃ x : ℝ, (3 / (x^2 - x) + 1 = x / (x - 1)) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2850_285041


namespace NUMINAMATH_CALUDE_total_honey_production_total_honey_is_1060_l2850_285018

/-- Calculates the total honey production for two bee hives with given characteristics -/
theorem total_honey_production
  (hive1_bees : ℕ)
  (hive1_honey : ℝ)
  (hive2_bee_reduction : ℝ)
  (hive2_honey_increase : ℝ)
  (h1 : hive1_bees = 1000)
  (h2 : hive1_honey = 500)
  (h3 : hive2_bee_reduction = 0.2)
  (h4 : hive2_honey_increase = 0.4)
  : ℝ :=
by
  -- The proof goes here
  sorry

#check total_honey_production

/-- The total honey production is 1060 liters -/
theorem total_honey_is_1060 :
  total_honey_production 1000 500 0.2 0.4 rfl rfl rfl rfl = 1060 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_total_honey_production_total_honey_is_1060_l2850_285018
