import Mathlib

namespace NUMINAMATH_CALUDE_garden_area_l3213_321361

theorem garden_area (width length : ℝ) : 
  length = 3 * width + 30 →
  2 * (width + length) = 780 →
  width * length = 27000 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l3213_321361


namespace NUMINAMATH_CALUDE_product_of_fractions_l3213_321332

theorem product_of_fractions : (1 + 1/3) * (1 + 1/4) = 5/3 := by sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3213_321332


namespace NUMINAMATH_CALUDE_rectangle_side_length_l3213_321362

/-- If a rectangle has area 4a²b³ and one side 2ab³, then the other side is 2a -/
theorem rectangle_side_length (a b : ℝ) (area : ℝ) (side1 : ℝ) :
  area = 4 * a^2 * b^3 → side1 = 2 * a * b^3 → area / side1 = 2 * a :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l3213_321362


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3213_321318

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -3 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, x^2 + 4*x + 3 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3213_321318


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l3213_321337

/-- The intersection points of two parabolas that also lie on a given line -/
theorem parabola_intersection_points (x y : ℝ) :
  (y = 3 * x^2 - 9 * x + 4) ∧ 
  (y = -x^2 + 3 * x + 6) ∧ 
  (y = x + 3) →
  ((x = (3 + Real.sqrt 11) / 2 ∧ y = (9 + Real.sqrt 11) / 2) ∨
   (x = (3 - Real.sqrt 11) / 2 ∧ y = (9 - Real.sqrt 11) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l3213_321337


namespace NUMINAMATH_CALUDE_bert_sandwiches_l3213_321367

/-- The number of sandwiches remaining after two days of eating -/
def sandwiches_remaining (initial : ℕ) : ℕ :=
  initial - (initial / 2) - (initial / 2 - 2)

/-- Theorem stating that given 12 initial sandwiches, 2 remain after two days of eating -/
theorem bert_sandwiches : sandwiches_remaining 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_bert_sandwiches_l3213_321367


namespace NUMINAMATH_CALUDE_triple_percent_40_l3213_321373

/-- The operation % defined on real numbers -/
def percent (M : ℝ) : ℝ := 0.4 * M + 2

/-- Theorem stating that applying the percent operation three times to 40 results in 5.68 -/
theorem triple_percent_40 : percent (percent (percent 40)) = 5.68 := by
  sorry

end NUMINAMATH_CALUDE_triple_percent_40_l3213_321373


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3213_321394

theorem min_value_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x * y = 2 * x + y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 ∧
  (x + y = 3 + 2 * Real.sqrt 2 ↔ x = Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3213_321394


namespace NUMINAMATH_CALUDE_b_not_two_l3213_321355

theorem b_not_two (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → |x + b| ≤ 2) : b ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_b_not_two_l3213_321355


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l3213_321356

/-- Calculates the difference in total fees collected between full capacity and 3/4 capacity for a stadium. -/
def fee_difference (capacity : ℕ) (entry_fee : ℕ) : ℕ :=
  capacity * entry_fee - (capacity * 3 / 4) * entry_fee

/-- Proves that the fee difference for a stadium with 2000 capacity and $20 entry fee is $10,000. -/
theorem stadium_fee_difference :
  fee_difference 2000 20 = 10000 := by
  sorry

#eval fee_difference 2000 20

end NUMINAMATH_CALUDE_stadium_fee_difference_l3213_321356


namespace NUMINAMATH_CALUDE_intensity_for_three_breaks_l3213_321379

/-- Represents the relationship between breaks and intensity -/
def inverse_proportional (breaks intensity : ℝ) (k : ℝ) : Prop :=
  breaks * intensity = k

theorem intensity_for_three_breaks 
  (k : ℝ) 
  (h1 : inverse_proportional 4 6 k) 
  (h2 : inverse_proportional 3 8 k) : 
  True :=
sorry

end NUMINAMATH_CALUDE_intensity_for_three_breaks_l3213_321379


namespace NUMINAMATH_CALUDE_village_population_l3213_321384

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.05) * (1 - 0.15) = 2553 → P = 3162 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l3213_321384


namespace NUMINAMATH_CALUDE_expression_undefined_at_eight_l3213_321369

/-- The expression is undefined when x = 8 -/
theorem expression_undefined_at_eight :
  ∀ x : ℝ, x = 8 → (x^2 - 16*x + 64 = 0) := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_eight_l3213_321369


namespace NUMINAMATH_CALUDE_smallest_a_l3213_321315

/-- The polynomial x³ - ax² + bx - 2010 with three positive integer zeros -/
def polynomial (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 2010

/-- The polynomial has three positive integer zeros -/
def has_three_positive_integer_zeros (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    polynomial a b x = 0 ∧ polynomial a b y = 0 ∧ polynomial a b z = 0

/-- The smallest possible value of a is 78 -/
theorem smallest_a (a b : ℤ) :
  has_three_positive_integer_zeros a b → a ≥ 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_l3213_321315


namespace NUMINAMATH_CALUDE_cid_oil_changes_l3213_321359

/-- Represents the mechanic shop's pricing and services -/
structure MechanicShop where
  oil_change_price : ℕ
  repair_price : ℕ
  car_wash_price : ℕ
  repaired_cars : ℕ
  washed_cars : ℕ
  total_earnings : ℕ

/-- Calculates the number of oil changes given the shop's data -/
def calculate_oil_changes (shop : MechanicShop) : ℕ :=
  (shop.total_earnings - shop.repair_price * shop.repaired_cars - shop.car_wash_price * shop.washed_cars) / shop.oil_change_price

/-- Theorem stating that Cid changed the oil for 5 cars -/
theorem cid_oil_changes :
  let shop : MechanicShop := {
    oil_change_price := 20,
    repair_price := 30,
    car_wash_price := 5,
    repaired_cars := 10,
    washed_cars := 15,
    total_earnings := 475
  }
  calculate_oil_changes shop = 5 := by sorry

end NUMINAMATH_CALUDE_cid_oil_changes_l3213_321359


namespace NUMINAMATH_CALUDE_remaining_books_and_games_l3213_321338

/-- The number of remaining items to experience in a category -/
def remaining (total : ℕ) (experienced : ℕ) : ℕ := total - experienced

/-- The total number of remaining items to experience across categories -/
def total_remaining (remaining1 : ℕ) (remaining2 : ℕ) : ℕ := remaining1 + remaining2

/-- Proof that the number of remaining books and games to experience is 109 -/
theorem remaining_books_and_games :
  let total_books : ℕ := 150
  let total_games : ℕ := 50
  let books_read : ℕ := 74
  let games_played : ℕ := 17
  let remaining_books := remaining total_books books_read
  let remaining_games := remaining total_games games_played
  total_remaining remaining_books remaining_games = 109 := by
  sorry

end NUMINAMATH_CALUDE_remaining_books_and_games_l3213_321338


namespace NUMINAMATH_CALUDE_emily_furniture_puzzle_l3213_321330

/-- The number of tables Emily bought -/
def num_tables : ℕ := 2

/-- The number of chairs Emily bought -/
def num_chairs : ℕ := 4

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent assembling furniture (in minutes) -/
def total_time : ℕ := 48

theorem emily_furniture_puzzle :
  num_tables * time_per_furniture + num_chairs * time_per_furniture = total_time :=
by sorry

end NUMINAMATH_CALUDE_emily_furniture_puzzle_l3213_321330


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3213_321324

theorem rectangle_circle_area_ratio (l w r : ℝ) (h1 : 2 * l + 2 * w = 2 * Real.pi * r) (h2 : l = 2 * w) :
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3213_321324


namespace NUMINAMATH_CALUDE_c_d_not_dine_city_center_l3213_321378

-- Define the participants
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define locations
inductive Location : Type
| CityCenter : Location
| NearAHome : Location

-- Define the dining relation
def dines_together (p1 p2 : Person) (l : Location) : Prop := sorry

-- Define participation in dining
def participates (p : Person) : Prop := sorry

-- Condition 1: Only if A participates, B and C will dine together
axiom cond1 : ∀ (l : Location), dines_together Person.B Person.C l → participates Person.A

-- Condition 2: A only dines at restaurants near their home
axiom cond2 : ∀ (p : Person) (l : Location), 
  dines_together Person.A p l → l = Location.NearAHome

-- Condition 3: Only if B participates, D will go to the restaurant to dine
axiom cond3 : ∀ (p : Person) (l : Location), 
  dines_together Person.D p l → participates Person.B

-- Theorem to prove
theorem c_d_not_dine_city_center : 
  ¬(dines_together Person.C Person.D Location.CityCenter) :=
sorry

end NUMINAMATH_CALUDE_c_d_not_dine_city_center_l3213_321378


namespace NUMINAMATH_CALUDE_expression_evaluation_l3213_321385

theorem expression_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 4/5
  (6*x + 8*y + x^2*y) / (60*x*y^2) = 21/50 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3213_321385


namespace NUMINAMATH_CALUDE_product_prime_factors_l3213_321357

theorem product_prime_factors (m n : ℕ) : 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    m = p₁ * p₂ * p₃ * p₄) →
  (∃ q₁ q₂ q₃ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ 
    q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₂ ≠ q₃ ∧
    n = q₁ * q₂ * q₃) →
  Nat.gcd m n = 15 →
  ∃ r₁ r₂ r₃ r₄ r₅ : ℕ, Prime r₁ ∧ Prime r₂ ∧ Prime r₃ ∧ Prime r₄ ∧ Prime r₅ ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ 
    r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
    r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
    r₄ ≠ r₅ ∧
    m * n = r₁ * r₂ * r₃ * r₄ * r₅ :=
by
  sorry

end NUMINAMATH_CALUDE_product_prime_factors_l3213_321357


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3213_321305

/-- Given points A and B, and a point C on line AB satisfying a vector relationship,
    prove that C has specific coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-1, -1) →
  B = (2, 5) →
  (∃ t : ℝ, C = (1 - t) • A + t • B) →  -- C is on line AB
  (C.1 - A.1, C.2 - A.2) = 5 • (B.1 - C.1, B.2 - C.2) →  -- Vector relationship
  C = (3/2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_c_coordinates_l3213_321305


namespace NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l3213_321390

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces
    (of which 30 are triangular and 14 are quadrilateral) has 335 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by sorry

end NUMINAMATH_CALUDE_specific_polyhedron_space_diagonals_l3213_321390


namespace NUMINAMATH_CALUDE_parabola_equation_l3213_321334

/-- A parabola with vertex at the origin and directrix y = 4 has the standard equation x^2 = -16y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) :
  (∀ x y, p x y ↔ y = -x^2 / 16) →  -- Standard equation of the parabola
  (∀ x, p x 0) →  -- Vertex at the origin
  (∀ x, p x 4 ↔ x = 0) →  -- Directrix equation
  ∀ x y, p x y ↔ x^2 = -16 * y := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3213_321334


namespace NUMINAMATH_CALUDE_toothpicks_per_card_l3213_321391

theorem toothpicks_per_card 
  (total_cards : ℕ) 
  (unused_cards : ℕ) 
  (toothpick_boxes : ℕ) 
  (toothpicks_per_box : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : unused_cards = 16) 
  (h3 : toothpick_boxes = 6) 
  (h4 : toothpicks_per_box = 450) :
  (toothpick_boxes * toothpicks_per_box) / (total_cards - unused_cards) = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_toothpicks_per_card_l3213_321391


namespace NUMINAMATH_CALUDE_square_side_ratio_l3213_321316

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 72 / 98) :
  ∃ (a b c : ℕ), 
    (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) ∧ 
    a = 6 ∧ 
    b = 1 ∧ 
    c = 7 ∧ 
    a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l3213_321316


namespace NUMINAMATH_CALUDE_composition_f_equals_one_over_e_l3213_321319

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

-- State the theorem
theorem composition_f_equals_one_over_e :
  f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_composition_f_equals_one_over_e_l3213_321319


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3213_321306

/-- A right-angled triangle with area 150 cm² and perimeter 60 cm has sides of length 15 cm, 20 cm, and 25 cm. -/
theorem right_triangle_sides (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 150 →
  a + b + c = 60 →
  ((a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15)) ∧ c = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3213_321306


namespace NUMINAMATH_CALUDE_chord_length_l3213_321360

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 16

-- Define the line
def line (x y : ℝ) : Prop := x - 2*y - 4 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2

-- Theorem statement
theorem chord_length : 
  ∀ A B : ℝ × ℝ, intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3213_321360


namespace NUMINAMATH_CALUDE_inequality_proof_l3213_321368

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3213_321368


namespace NUMINAMATH_CALUDE_max_x_value_l3213_321371

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_sum_eq : x*y + x*z + y*z = 8) : 
  x ≤ 7/3 :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l3213_321371


namespace NUMINAMATH_CALUDE_irregular_shape_area_l3213_321340

/-- The area of an irregular shape consisting of a rectangle connected to a semi-circle -/
theorem irregular_shape_area (square_area : ℝ) (rect_length : ℝ) : 
  square_area = 2025 →
  rect_length = 10 →
  let circle_radius := Real.sqrt square_area
  let rect_breadth := (3 / 5) * circle_radius
  let rect_area := rect_length * rect_breadth
  let semicircle_area := (1 / 2) * Real.pi * circle_radius ^ 2
  rect_area + semicircle_area = 270 + 1012.5 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_irregular_shape_area_l3213_321340


namespace NUMINAMATH_CALUDE_nested_radical_simplification_l3213_321350

theorem nested_radical_simplification (a b m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hpos : a + 2 * Real.sqrt b > 0)
  (hm : m > 0) (hn : n > 0) (hmn_sum : m + n = a) (hmn_prod : m * n = b) :
  Real.sqrt (a + 2 * Real.sqrt b) = Real.sqrt m + Real.sqrt n ∧
  Real.sqrt (a - 2 * Real.sqrt b) = |Real.sqrt m - Real.sqrt n| :=
by sorry

end NUMINAMATH_CALUDE_nested_radical_simplification_l3213_321350


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_one_l3213_321326

def curve (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2

def tangent_slope (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2

theorem tangent_slope_at_negative_one (a : ℝ) :
  tangent_slope a (-1) = Real.tan (π/4) → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_one_l3213_321326


namespace NUMINAMATH_CALUDE_m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l3213_321386

theorem m_fourth_plus_twice_m_cubed_minus_m_plus_2007 (m : ℝ) 
  (h : m^2 + m - 1 = 0) : 
  m^4 + 2*m^3 - m + 2007 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l3213_321386


namespace NUMINAMATH_CALUDE_product_mod_eight_l3213_321395

theorem product_mod_eight : (71 * 73) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l3213_321395


namespace NUMINAMATH_CALUDE_system_solution_l3213_321301

theorem system_solution (x y : ℝ) (hx : x = 4) (hy : y = -1) : x - 2*y = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3213_321301


namespace NUMINAMATH_CALUDE_bob_winning_strategy_l3213_321335

/-- A polynomial with natural number coefficients -/
def NatPoly := ℕ → ℕ

/-- The evaluation of a polynomial at a given point -/
def eval (P : NatPoly) (x : ℤ) : ℕ :=
  sorry

/-- The degree of a polynomial -/
def degree (P : NatPoly) : ℕ :=
  sorry

/-- Bob's strategy: choose two integers and receive their polynomial values -/
def bob_strategy (P : NatPoly) : (ℤ × ℤ × ℕ × ℕ) :=
  sorry

theorem bob_winning_strategy :
  ∀ (P Q : NatPoly),
    (∀ (x : ℤ), eval P x = eval Q x) →
    let (a, b, Pa, Pb) := bob_strategy P
    eval P a = Pa ∧ eval P b = Pb ∧ eval Q a = Pa ∧ eval Q b = Pb →
    P = Q :=
  sorry

end NUMINAMATH_CALUDE_bob_winning_strategy_l3213_321335


namespace NUMINAMATH_CALUDE_father_daughter_speed_problem_l3213_321376

theorem father_daughter_speed_problem 
  (total_distance : ℝ) 
  (speed_ratio : ℝ) 
  (speed_increase : ℝ) 
  (time_difference : ℝ) :
  total_distance = 60 ∧ 
  speed_ratio = 2 ∧ 
  speed_increase = 2 ∧ 
  time_difference = 1/12 →
  ∃ (father_speed daughter_speed : ℝ),
    father_speed = 14 ∧ 
    daughter_speed = 28 ∧
    daughter_speed = speed_ratio * father_speed ∧
    (total_distance / (2 * father_speed + speed_increase) - 
     (total_distance / 2) / (father_speed + speed_increase)) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_father_daughter_speed_problem_l3213_321376


namespace NUMINAMATH_CALUDE_circular_bead_arrangements_l3213_321398

/-- The number of red beads -/
def num_red : ℕ := 3

/-- The number of blue beads -/
def num_blue : ℕ := 2

/-- The total number of beads -/
def total_beads : ℕ := num_red + num_blue

/-- The symmetry group of the circular arrangement -/
def symmetry_group : ℕ := 2 * total_beads

/-- The number of fixed arrangements under the identity rotation -/
def fixed_identity : ℕ := (total_beads.choose num_red)

/-- The number of fixed arrangements under each reflection -/
def fixed_reflection : ℕ := 2

/-- The number of reflections in the symmetry group -/
def num_reflections : ℕ := total_beads

/-- The total number of fixed arrangements under all symmetries -/
def total_fixed : ℕ := fixed_identity + num_reflections * fixed_reflection

/-- The number of distinct arrangements of beads on the circular ring -/
def distinct_arrangements : ℕ := total_fixed / symmetry_group

theorem circular_bead_arrangements :
  distinct_arrangements = 2 :=
sorry

end NUMINAMATH_CALUDE_circular_bead_arrangements_l3213_321398


namespace NUMINAMATH_CALUDE_junior_score_junior_score_is_89_l3213_321343

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (overall_avg : ℝ) (senior_avg : ℝ) : ℝ :=
  let junior_count := junior_ratio * n
  let senior_count := senior_ratio * n
  let total_score := overall_avg * n
  let senior_total := senior_avg * senior_count
  let junior_total := total_score - senior_total
  junior_total / junior_count

theorem junior_score_is_89 :
  junior_score 100 0.2 0.8 85 84 = 89 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_junior_score_is_89_l3213_321343


namespace NUMINAMATH_CALUDE_two_out_of_three_probability_l3213_321348

-- Define the success rate of the basketball player
def success_rate : ℚ := 3 / 5

-- Define the number of shots taken
def total_shots : ℕ := 3

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 2

-- Theorem statement
theorem two_out_of_three_probability :
  (Nat.choose total_shots successful_shots : ℚ) * success_rate ^ successful_shots * (1 - success_rate) ^ (total_shots - successful_shots) = 54 / 125 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_probability_l3213_321348


namespace NUMINAMATH_CALUDE_cos_five_pi_fourth_plus_x_l3213_321381

theorem cos_five_pi_fourth_plus_x (x : ℝ) (h : Real.sin (π / 4 - x) = -1 / 5) :
  Real.cos (5 * π / 4 + x) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_fourth_plus_x_l3213_321381


namespace NUMINAMATH_CALUDE_evaluate_expression_l3213_321300

theorem evaluate_expression : (1 / ((-5^3)^4)) * ((-5)^15) * (5^2) = -3125 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3213_321300


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3213_321304

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area : 
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area : ℝ := π * r^2
  let lateral_area : ℝ := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π := by
sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3213_321304


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3213_321345

/-- Represents a square with two internal unshaded squares -/
structure SquareWithInternalSquares where
  /-- Side length of the large square -/
  side : ℝ
  /-- Side length of the bottom-left unshaded square -/
  bottomLeftSide : ℝ
  /-- Side length of the top-right unshaded square -/
  topRightSide : ℝ
  /-- The bottom-left square's side is half of the large square's side -/
  bottomLeftHalf : bottomLeftSide = side / 2
  /-- The top-right square's diagonal is one-third of the large square's diagonal -/
  topRightThird : topRightSide * Real.sqrt 2 = side * Real.sqrt 2 / 3

/-- The fraction of the shaded area in a square with two internal unshaded squares is 19/36 -/
theorem shaded_area_fraction (s : SquareWithInternalSquares) :
  (s.side^2 - s.bottomLeftSide^2 - s.topRightSide^2) / s.side^2 = 19 / 36 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3213_321345


namespace NUMINAMATH_CALUDE_function_relationship_l3213_321302

/-- Given functions f and g, and constants A, B, and C, prove the relationship between A, B, and C. -/
theorem function_relationship (A B C : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 2 * B^2 * x + 3
  let g := fun x => B * x + 1
  f (g 1) = C →
  A = (C + 2 * B^3 + 2 * B^2 - 3) / (B^2 + 2 * B + 1) := by
  sorry

end NUMINAMATH_CALUDE_function_relationship_l3213_321302


namespace NUMINAMATH_CALUDE_largest_b_divisible_by_four_l3213_321377

theorem largest_b_divisible_by_four :
  let n : ℕ → ℕ := λ b => 4000000 + b * 100000 + 508632
  ∃ (b : ℕ), b ≤ 9 ∧ n b % 4 = 0 ∧ ∀ (x : ℕ), x ≤ 9 ∧ n x % 4 = 0 → x ≤ b :=
by sorry

end NUMINAMATH_CALUDE_largest_b_divisible_by_four_l3213_321377


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3213_321307

-- Define the function f(x) = -x^2 + bx + c
def f (b c x : ℝ) : ℝ := -x^2 + b*x + c

-- Theorem statement
theorem quadratic_function_properties :
  ∀ b c : ℝ,
  (f b c 0 = -3) →
  (f b c (-6) = -3) →
  (b = -6 ∧ c = -3) ∧
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ 0 → f b c x ≤ 6) ∧
  (∃ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ f b c x = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3213_321307


namespace NUMINAMATH_CALUDE_molly_sunday_swim_l3213_321325

/-- Represents the distance Molly swam on Sunday -/
def sunday_swim (saturday_swim total_swim : ℕ) : ℕ :=
  total_swim - saturday_swim

/-- Proves that Molly swam 28 meters on Sunday -/
theorem molly_sunday_swim :
  let saturday_swim : ℕ := 45
  let total_swim : ℕ := 73
  let pool_length : ℕ := 25
  sunday_swim saturday_swim total_swim = 28 := by
  sorry

end NUMINAMATH_CALUDE_molly_sunday_swim_l3213_321325


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3213_321347

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3213_321347


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3213_321310

/-- Represents the price and volume of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ  -- Amount of orange juice (same for both days)
  water_day1 : ℝ    -- Amount of water on day 1
  water_day2 : ℝ    -- Amount of water on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  revenue : ℝ        -- Revenue (same for both days)

/-- The price per glass on the second day is $0.20 given the conditions -/
theorem orangeade_price_day2 (o : Orangeade)
    (h1 : o.orange_juice = o.water_day1)
    (h2 : o.water_day2 = 2 * o.water_day1)
    (h3 : o.price_day1 = 0.30)
    (h4 : o.revenue = (o.orange_juice + o.water_day1) * o.price_day1)
    (h5 : o.revenue = (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l3213_321310


namespace NUMINAMATH_CALUDE_parallel_line_equation_l3213_321327

/-- A line passing through (1,1) and parallel to x+2y+2016=0 has equation x+2y-3=0 -/
theorem parallel_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∃ c : ℝ, l = {(x, y) | x + 2*y + c = 0}) →
  ((1, 1) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ (x + 2*y + 2016 = 0 → False)) →
  l = {(x, y) | x + 2*y - 3 = 0} :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l3213_321327


namespace NUMINAMATH_CALUDE_sum_division_l3213_321342

theorem sum_division (x y z : ℝ) (total : ℝ) (y_share : ℝ) : 
  total = 245 →
  y_share = 63 →
  y = 0.45 * x →
  total = x + y + z →
  z / x = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_sum_division_l3213_321342


namespace NUMINAMATH_CALUDE_total_pencils_is_twelve_l3213_321313

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 6

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_is_twelve : total_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_twelve_l3213_321313


namespace NUMINAMATH_CALUDE_intersection_is_empty_l3213_321336

-- Define set A
def A : Set ℝ := {x | x^2 + 4 ≤ 5*x}

-- Define set B
def B : Set (ℝ × ℝ) := {p | p.2 = 3^p.1 + 2}

-- Theorem statement
theorem intersection_is_empty : A ∩ (B.image Prod.fst) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_intersection_is_empty_l3213_321336


namespace NUMINAMATH_CALUDE_book_purchase_problem_l3213_321364

theorem book_purchase_problem (total_volumes : ℕ) (paperback_price hardcover_price : ℚ) 
  (discount : ℚ) (total_cost : ℚ) :
  total_volumes = 12 ∧ 
  paperback_price = 16 ∧ 
  hardcover_price = 27 ∧ 
  discount = 6 ∧ 
  total_cost = 278 →
  ∃ (h : ℕ), 
    h = 8 ∧ 
    h ≤ total_volumes ∧ 
    (h > 5 → hardcover_price * h + paperback_price * (total_volumes - h) - discount = total_cost) ∧
    (h ≤ 5 → hardcover_price * h + paperback_price * (total_volumes - h) = total_cost) :=
by sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l3213_321364


namespace NUMINAMATH_CALUDE_frog_corner_probability_l3213_321363

/-- Represents a position on the 3x3 grid -/
inductive Position
| Center
| Edge
| Corner

/-- Represents the number of hops made -/
def MaxHops : Nat := 4

/-- Probability of reaching a corner from a given position in n hops -/
noncomputable def reachCornerProb (pos : Position) (n : Nat) : Real :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  reachCornerProb Position.Center MaxHops = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l3213_321363


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3213_321380

def repeating_decimal_1 : ℚ := 1/3
def repeating_decimal_2 : ℚ := 4/99
def repeating_decimal_3 : ℚ := 5/999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 42/111 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3213_321380


namespace NUMINAMATH_CALUDE_abc_value_l3213_321389

noncomputable def A (x : ℝ) : ℝ := ∑' k, x^(3*k) / (3*k).factorial
noncomputable def B (x : ℝ) : ℝ := ∑' k, x^(3*k+1) / (3*k+1).factorial
noncomputable def C (x : ℝ) : ℝ := ∑' k, x^(3*k+2) / (3*k+2).factorial

theorem abc_value (x : ℝ) (hx : x > 0) :
  (A x)^3 + (B x)^3 + (C x)^3 + 8*(A x)*(B x)*(C x) = 2014 →
  (A x)*(B x)*(C x) = 183 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l3213_321389


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3213_321309

theorem fraction_to_decimal : (21 : ℚ) / 160 = 0.13125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3213_321309


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3213_321303

/-- Proves that the first interest rate is 10% given the problem conditions -/
theorem first_interest_rate_is_ten_percent
  (total_amount : ℕ)
  (first_part : ℕ)
  (second_part : ℕ)
  (second_rate : ℚ)
  (total_profit : ℕ)
  (h1 : total_amount = 70000)
  (h2 : first_part = 60000)
  (h3 : second_part = 10000)
  (h4 : total_amount = first_part + second_part)
  (h5 : second_rate = 20 / 100)
  (h6 : total_profit = 8000)
  (h7 : total_profit = first_part * (first_rate / 100) + second_part * (second_rate / 100)) :
  first_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3213_321303


namespace NUMINAMATH_CALUDE_light_bulb_probabilities_l3213_321387

/-- Represents a light bulb factory -/
inductive Factory
| A
| B

/-- Properties of the light bulb inventory -/
structure LightBulbInventory where
  total : ℕ
  factoryA_fraction : ℝ
  factoryB_fraction : ℝ
  factoryA_firstclass_rate : ℝ
  factoryB_firstclass_rate : ℝ

/-- The specific light bulb inventory in the problem -/
def problem_inventory : LightBulbInventory :=
  { total := 50
  , factoryA_fraction := 0.6
  , factoryB_fraction := 0.4
  , factoryA_firstclass_rate := 0.9
  , factoryB_firstclass_rate := 0.8
  }

/-- The probability of randomly selecting a first-class product from Factory A -/
def prob_firstclass_A (inv : LightBulbInventory) : ℝ :=
  inv.factoryA_fraction * inv.factoryA_firstclass_rate

/-- The expected value of first-class products from Factory A when selecting two light bulbs -/
def expected_firstclass_A_two_selections (inv : LightBulbInventory) : ℝ :=
  2 * prob_firstclass_A inv

/-- Main theorem stating the probabilities for the given inventory -/
theorem light_bulb_probabilities :
  prob_firstclass_A problem_inventory = 0.54 ∧
  expected_firstclass_A_two_selections problem_inventory = 1.08 := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probabilities_l3213_321387


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l3213_321351

theorem tangent_and_trigonometric_identity (α : ℝ) 
  (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) : 
  (Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3) ∧ 
  (2 * Real.sin α ^ 2 - Real.cos α ^ 2 = -43 / 52) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l3213_321351


namespace NUMINAMATH_CALUDE_real_roots_condition_intersection_condition_l3213_321329

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 2 * x + 1

-- Theorem for the range of m for real roots
theorem real_roots_condition (m : ℝ) :
  (∃ x : ℝ, f m x = 0) ↔ m ≤ 3 :=
sorry

-- Define the condition for intersection at one point
def intersects_once (m : ℝ) : Prop :=
  ∃! x : ℝ, x > 1 ∧ x < 2 ∧ f m x = 0

-- Theorem for the range of m for intersection at one point
theorem intersection_condition (m : ℝ) :
  intersects_once m ↔ -1 ≤ m ∧ m < 3/4 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_intersection_condition_l3213_321329


namespace NUMINAMATH_CALUDE_sum_34_47_in_base5_l3213_321312

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_34_47_in_base5 :
  toBase5 (34 + 47) = [3, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_34_47_in_base5_l3213_321312


namespace NUMINAMATH_CALUDE_smallest_root_of_g_l3213_321396

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 14 * x^2 + 4

-- State the theorem
theorem smallest_root_of_g :
  ∃ (r : ℝ), g r = 0 ∧ r = -1 ∧ ∀ (x : ℝ), g x = 0 → x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_of_g_l3213_321396


namespace NUMINAMATH_CALUDE_smallest_circle_theorem_l3213_321354

/-- Given two circles in the xy-plane, this function returns the equation of the circle 
    with the smallest area that passes through their intersection points. -/
def smallest_circle_through_intersections (c1 c2 : ℝ × ℝ → Prop) : ℝ × ℝ → Prop :=
  sorry

/-- The first given circle -/
def circle1 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 4*p.1 + p.2 = -1

/-- The second given circle -/
def circle2 (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + 2*p.1 + 2*p.2 + 1 = 0

/-- The resulting circle with the smallest area -/
def result_circle (p : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 + (6/5)*p.1 + (12/5)*p.2 + 1 = 0

theorem smallest_circle_theorem :
  smallest_circle_through_intersections circle1 circle2 = result_circle :=
sorry

end NUMINAMATH_CALUDE_smallest_circle_theorem_l3213_321354


namespace NUMINAMATH_CALUDE_value_of_S_l3213_321374

theorem value_of_S : let S : ℝ := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 1 / (Real.sqrt 12 - 3)
  S = 7 := by sorry

end NUMINAMATH_CALUDE_value_of_S_l3213_321374


namespace NUMINAMATH_CALUDE_modulus_of_z_l3213_321382

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3213_321382


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_m_range_l3213_321392

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -4/3 < x ∧ x < 4/3} := by sorry

theorem m_range (m : ℝ) :
  (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
  (m < -3/4 ∨ m > 3/4) := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_m_range_l3213_321392


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l3213_321308

/-- Given two lines that are perpendicular, find the value of 'a' -/
theorem perpendicular_lines_a_value : ℝ → Prop :=
  fun a => 
    let line1 := fun x y : ℝ => 3 * y - x + 4 = 0
    let line2 := fun x y : ℝ => 4 * y + a * x + 5 = 0
    let slope1 := (1 : ℝ) / 3
    let slope2 := -a / 4
    (∀ x y : ℝ, line1 x y ∧ line2 x y → slope1 * slope2 = -1) →
    a = 12

/-- Proof of the theorem -/
theorem perpendicular_lines_a_value_proof : perpendicular_lines_a_value 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_perpendicular_lines_a_value_proof_l3213_321308


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_zero_one_l3213_321321

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_zero_one :
  ∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo (0 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_zero_one_l3213_321321


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3213_321397

theorem fourteenth_root_of_unity (n : ℕ) (hn : n ≤ 13) : 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (5 * Real.pi / 7)) :=
sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3213_321397


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3213_321317

theorem algebraic_expression_value (x : ℝ) :
  12 * x - 8 * x^2 = -1 → 4 * x^2 - 6 * x + 5 = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3213_321317


namespace NUMINAMATH_CALUDE_cubic_factorization_l3213_321365

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3213_321365


namespace NUMINAMATH_CALUDE_cosine_sum_zero_implies_angle_difference_l3213_321331

theorem cosine_sum_zero_implies_angle_difference (α β γ : ℝ) 
  (h1 : 0 < α ∧ α < β ∧ β < γ ∧ γ < 2 * π)
  (h2 : ∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) : 
  γ - α = 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_zero_implies_angle_difference_l3213_321331


namespace NUMINAMATH_CALUDE_triangle_altitude_specific_triangle_altitude_l3213_321388

/-- The altitude of a triangle given its area and base -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (h_area : area > 0) (h_base : base > 0) :
  area = (1/2) * base * (2 * area / base) :=
by sorry

/-- The altitude of a specific triangle with area 800 and base 40 -/
theorem specific_triangle_altitude :
  let area : ℝ := 800
  let base : ℝ := 40
  let altitude : ℝ := 2 * area / base
  altitude = 40 :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_specific_triangle_altitude_l3213_321388


namespace NUMINAMATH_CALUDE_base_6_not_divisible_by_5_others_are_l3213_321328

def base_b_diff (b : ℕ) : ℤ := 2 * b^3 - 2 * b^2 + 1

theorem base_6_not_divisible_by_5_others_are :
  (¬ (base_b_diff 6 % 5 = 0)) ∧
  (base_b_diff 7 % 5 = 0) ∧
  (base_b_diff 8 % 5 = 0) ∧
  (base_b_diff 9 % 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_base_6_not_divisible_by_5_others_are_l3213_321328


namespace NUMINAMATH_CALUDE_smallest_r_value_l3213_321349

theorem smallest_r_value (p q r : ℕ) : 
  0 < p ∧ p < q ∧ q < r ∧                   -- p, q, r are positive integers and p < q < r
  (2 * q = p + r) ∧                         -- arithmetic progression
  (r * r = p * q) →                         -- geometric progression
  r ≥ 5 ∧ ∃ (p' q' r' : ℕ), 
    0 < p' ∧ p' < q' ∧ q' < r' ∧ 
    (2 * q' = p' + r') ∧ 
    (r' * r' = p' * q') ∧ 
    r' = 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_r_value_l3213_321349


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l3213_321346

theorem subset_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x < 5 → x < a) → a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l3213_321346


namespace NUMINAMATH_CALUDE_unique_solution_for_cubic_equations_l3213_321323

/-- Represents the roots of a cubic equation -/
structure CubicRoots (α : Type*) [Field α] where
  r₁ : α
  r₂ : α
  r₃ : α

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  y - x = z - y ∧ y - x ≠ 0

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression {α : Type*} [Field α] (x y z : α) : Prop :=
  ∃ r : α, r ≠ 1 ∧ y = x * r ∧ z = y * r

/-- Represents the coefficients of the first cubic equation -/
structure FirstEquationCoeffs (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Represents the coefficients of the second cubic equation -/
structure SecondEquationCoeffs (α : Type*) [Field α] where
  b : α
  c : α

/-- The main theorem -/
theorem unique_solution_for_cubic_equations 
  (f : FirstEquationCoeffs ℝ) 
  (g : SecondEquationCoeffs ℝ)
  (roots1 : CubicRoots ℝ)
  (roots2 : CubicRoots ℝ)
  (h1 : roots1.r₁^3 - 3*f.a*roots1.r₁^2 + f.b*roots1.r₁ + 18*f.c = 0)
  (h2 : roots1.r₂^3 - 3*f.a*roots1.r₂^2 + f.b*roots1.r₂ + 18*f.c = 0)
  (h3 : roots1.r₃^3 - 3*f.a*roots1.r₃^2 + f.b*roots1.r₃ + 18*f.c = 0)
  (h4 : is_arithmetic_progression roots1.r₁ roots1.r₂ roots1.r₃)
  (h5 : roots2.r₁^3 + g.b*roots2.r₁^2 + roots2.r₁ - g.c^3 = 0)
  (h6 : roots2.r₂^3 + g.b*roots2.r₂^2 + roots2.r₂ - g.c^3 = 0)
  (h7 : roots2.r₃^3 + g.b*roots2.r₃^2 + roots2.r₃ - g.c^3 = 0)
  (h8 : is_geometric_progression roots2.r₁ roots2.r₂ roots2.r₃)
  (h9 : f.b = g.b)
  (h10 : f.c = g.c)
  : f.a = 2 ∧ f.b = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_cubic_equations_l3213_321323


namespace NUMINAMATH_CALUDE_tangent_circles_count_l3213_321311

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the property of two circles being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the property of a circle being tangent to two other circles
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

-- State the theorem
theorem tangent_circles_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), 
    (∀ c ∈ s, c.radius = 4 ∧ is_tangent_to_both c c1 c2) ∧ 
    s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l3213_321311


namespace NUMINAMATH_CALUDE_perpendicular_lines_l3213_321383

def line1 (x y : ℝ) : Prop := 3 * y - 2 * x + 4 = 0
def line2 (x y : ℝ) (b : ℝ) : Prop := 5 * y + b * x - 1 = 0

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, line1 x y → ∃ m1 : ℝ, y = m1 * x + (-4/3)) →
  (∀ x y : ℝ, line2 x y b → ∃ m2 : ℝ, y = m2 * x + (1/5)) →
  perpendicular (2/3) (-b/5) →
  b = 15/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l3213_321383


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3213_321352

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A) ∩ (Set.compl B) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3213_321352


namespace NUMINAMATH_CALUDE_gcd_sum_diff_l3213_321375

theorem gcd_sum_diff (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd a b = 1) :
  (Nat.gcd (a + b) (a - b) = 1) ∨ (Nat.gcd (a + b) (a - b) = 2) :=
sorry

end NUMINAMATH_CALUDE_gcd_sum_diff_l3213_321375


namespace NUMINAMATH_CALUDE_tennis_net_max_cuts_l3213_321344

/-- Represents a grid of squares -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Calculates the total number of edges in a grid -/
def total_edges (g : Grid) : ℕ :=
  (g.rows + 1) * g.cols + (g.cols + 1) * g.rows

/-- Calculates the maximum number of edges that can be cut without disconnecting the grid -/
def max_cuttable_edges (g : Grid) : ℕ :=
  (g.rows - 1) * (g.cols - 1)

/-- Theorem stating that for a 100 × 10 grid, the maximum number of cuttable edges is 891 -/
theorem tennis_net_max_cuts :
  let g : Grid := ⟨10, 100⟩
  max_cuttable_edges g = 891 :=
by sorry

end NUMINAMATH_CALUDE_tennis_net_max_cuts_l3213_321344


namespace NUMINAMATH_CALUDE_a_investment_l3213_321322

/-- A's investment in a partnership business --/
def partners_investment (total_profit partner_a_total_received partner_b_investment : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let partner_a_profit_share := partner_a_total_received - management_fee
  (partner_a_profit_share * partner_b_investment) / (remaining_profit - partner_a_profit_share)

/-- Theorem stating A's investment given the problem conditions --/
theorem a_investment (total_profit partner_a_total_received partner_b_investment : ℚ) 
  (h1 : total_profit = 9600)
  (h2 : partner_a_total_received = 4800)
  (h3 : partner_b_investment = 25000) :
  partners_investment total_profit partner_a_total_received partner_b_investment = 20000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_l3213_321322


namespace NUMINAMATH_CALUDE_sum_of_abc_l3213_321372

theorem sum_of_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l3213_321372


namespace NUMINAMATH_CALUDE_angle_bisector_implies_line_AC_l3213_321366

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (-1, 2)

-- Define the angle bisector equation
def angle_bisector (x y : ℝ) : Prop := y = x + 1

-- Define the equation of line AC
def line_AC (x y : ℝ) : Prop := x - 2*y - 1 = 0

theorem angle_bisector_implies_line_AC :
  ∀ C : ℝ × ℝ,
  angle_bisector C.1 C.2 →
  line_AC C.1 C.2 :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_implies_line_AC_l3213_321366


namespace NUMINAMATH_CALUDE_unique_solution_l3213_321341

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10
  h_nonzero : hundreds ≠ 0

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a number -/
def digitProduct (n : Nat) : Nat :=
  if n < 10 then n
  else if n < 100 then (n / 10) * (n % 10)
  else (n / 100) * ((n / 10) % 10) * (n % 10)

/-- Checks if a three-digit number satisfies the given conditions -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  let firstProduct := digitProduct n.value
  let secondProduct := digitProduct firstProduct
  (10 ≤ firstProduct ∧ firstProduct < 100) ∧
  (0 < secondProduct ∧ secondProduct < 10) ∧
  n.hundreds = 1 ∧
  n.tens = firstProduct / 10 ∧
  n.ones = firstProduct % 10 ∧
  secondProduct = firstProduct % 10

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ n.value = 144 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3213_321341


namespace NUMINAMATH_CALUDE_condition_relationship_l3213_321393

theorem condition_relationship (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3213_321393


namespace NUMINAMATH_CALUDE_sum_of_ages_in_two_years_l3213_321370

def Matt_age (Fem_age : ℕ) : ℕ := 4 * Fem_age

def current_Fem_age : ℕ := 11

def future_age (current_age : ℕ) : ℕ := current_age + 2

theorem sum_of_ages_in_two_years :
  future_age (Matt_age current_Fem_age) + future_age current_Fem_age = 59 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_two_years_l3213_321370


namespace NUMINAMATH_CALUDE_library_visitor_average_l3213_321399

/-- Calculates the average number of visitors per day for a month in a library --/
def average_visitors_per_day (sunday_visitors : ℕ) (weekday_visitors : ℕ) 
  (holiday_increase_percent : ℚ) (total_days : ℕ) (sundays : ℕ) (holidays : ℕ) : ℚ :=
  let weekdays := total_days - sundays
  let regular_weekdays := weekdays - holidays
  let holiday_visitors := weekday_visitors * (1 + holiday_increase_percent)
  let total_visitors := sunday_visitors * sundays + 
                        weekday_visitors * regular_weekdays + 
                        holiday_visitors * holidays
  total_visitors / total_days

/-- Theorem stating that the average number of visitors per day is 256 --/
theorem library_visitor_average : 
  average_visitors_per_day 540 240 (1/4) 30 4 4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_library_visitor_average_l3213_321399


namespace NUMINAMATH_CALUDE_no_solution_exists_l3213_321320

theorem no_solution_exists : ¬∃ (a b c d : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ 
  (a/b + b/c + c/d + d/a = 6) ∧ 
  (b/a + c/b + d/c + a/d = 32) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3213_321320


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l3213_321358

def total_crayons : ℕ := 15
def red_crayons : ℕ := 4
def selection_size : ℕ := 5

theorem crayon_selection_ways : 
  (Nat.choose total_crayons selection_size) -
  (Nat.choose red_crayons 2 * Nat.choose (total_crayons - red_crayons) (selection_size - 2)) +
  (Nat.choose red_crayons 1 * Nat.choose (total_crayons - red_crayons) (selection_size - 1)) +
  (Nat.choose (total_crayons - red_crayons) selection_size) = 1782 :=
by sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l3213_321358


namespace NUMINAMATH_CALUDE_hourly_wage_calculation_l3213_321333

/-- The hourly wage in dollars -/
def hourly_wage : ℝ := 12.5

/-- The number of hours worked per week -/
def hours_worked : ℝ := 40

/-- The pay per widget in dollars -/
def pay_per_widget : ℝ := 0.16

/-- The number of widgets produced per week -/
def widgets_produced : ℝ := 1250

/-- The total earnings for the week in dollars -/
def total_earnings : ℝ := 700

theorem hourly_wage_calculation :
  hourly_wage * hours_worked + pay_per_widget * widgets_produced = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_hourly_wage_calculation_l3213_321333


namespace NUMINAMATH_CALUDE_combined_final_price_theorem_l3213_321353

def calculate_final_price (cost_price repairs discount_rate tax_rate : ℝ) : ℝ :=
  let total_cost := cost_price + repairs
  let discounted_price := total_cost * (1 - discount_rate)
  discounted_price * (1 + tax_rate)

def cycle_a_price := calculate_final_price 1800 200 0.10 0.05
def cycle_b_price := calculate_final_price 2400 300 0.12 0.06
def cycle_c_price := calculate_final_price 3200 400 0.15 0.07

theorem combined_final_price_theorem :
  cycle_a_price + cycle_b_price + cycle_c_price = 7682.76 := by
  sorry

end NUMINAMATH_CALUDE_combined_final_price_theorem_l3213_321353


namespace NUMINAMATH_CALUDE_base_faces_area_sum_l3213_321339

/-- A pentagonal prism with given surface area and lateral area -/
structure PentagonalPrism where
  surfaceArea : ℝ
  lateralArea : ℝ

/-- Theorem: For a pentagonal prism with surface area 30 and lateral area 25,
    the sum of the areas of the two base faces equals 5 -/
theorem base_faces_area_sum (prism : PentagonalPrism)
    (h1 : prism.surfaceArea = 30)
    (h2 : prism.lateralArea = 25) :
    prism.surfaceArea - prism.lateralArea = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_faces_area_sum_l3213_321339


namespace NUMINAMATH_CALUDE_toms_floor_replacement_cost_l3213_321314

/-- The total cost to replace a floor given room dimensions, removal cost, and new floor cost per square foot. -/
def total_floor_replacement_cost (length width removal_cost cost_per_sqft : ℝ) : ℝ :=
  removal_cost + length * width * cost_per_sqft

/-- Theorem stating that the total cost to replace the floor in Tom's room is $120. -/
theorem toms_floor_replacement_cost :
  total_floor_replacement_cost 8 7 50 1.25 = 120 := by
  sorry

#eval total_floor_replacement_cost 8 7 50 1.25

end NUMINAMATH_CALUDE_toms_floor_replacement_cost_l3213_321314
