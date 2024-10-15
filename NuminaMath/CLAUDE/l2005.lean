import Mathlib

namespace NUMINAMATH_CALUDE_games_to_give_away_l2005_200538

def initial_games : ℕ := 50
def desired_games : ℕ := 35

theorem games_to_give_away :
  initial_games - desired_games = 15 :=
by sorry

end NUMINAMATH_CALUDE_games_to_give_away_l2005_200538


namespace NUMINAMATH_CALUDE_parallelepiped_has_twelve_edges_l2005_200556

/-- A parallelepiped is a three-dimensional figure formed by six parallelograms. -/
structure Parallelepiped where
  faces : Fin 6 → Parallelogram
  -- Additional properties ensuring the faces form a valid parallelepiped could be added here

/-- The number of edges in a geometric figure. -/
def numEdges (figure : Type) : ℕ := sorry

/-- Theorem stating that a parallelepiped has 12 edges. -/
theorem parallelepiped_has_twelve_edges (P : Parallelepiped) : numEdges Parallelepiped = 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_has_twelve_edges_l2005_200556


namespace NUMINAMATH_CALUDE_functional_equation_implies_constant_l2005_200582

theorem functional_equation_implies_constant (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_eq : ∀ x y : ℝ, f (x + 2*y) = 2 * f x * f y) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_constant_l2005_200582


namespace NUMINAMATH_CALUDE_pebble_collection_l2005_200553

theorem pebble_collection (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 15 → a = 1 → d = 1 → (n * (2 * a + (n - 1) * d)) / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_l2005_200553


namespace NUMINAMATH_CALUDE_max_distance_from_origin_dog_max_distance_l2005_200575

/-- The maximum distance a point on a circle can be from the origin,
    given the circle's center coordinates and radius. -/
theorem max_distance_from_origin (x y r : ℝ) : 
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  ∀ p : ℝ × ℝ, (p.1 - x)^2 + (p.2 - y)^2 = r^2 → 
    p.1^2 + p.2^2 ≤ max_distance^2 :=
by
  sorry

/-- The specific case for the dog problem -/
theorem dog_max_distance : 
  let x : ℝ := 6
  let y : ℝ := 8
  let r : ℝ := 15
  let center_distance := Real.sqrt (x^2 + y^2)
  let max_distance := center_distance + r
  max_distance = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_dog_max_distance_l2005_200575


namespace NUMINAMATH_CALUDE_minimum_growth_rate_for_doubling_output_l2005_200580

theorem minimum_growth_rate_for_doubling_output :
  let r : ℝ := Real.sqrt 2 - 1
  ∀ x : ℝ, (1 + x)^2 ≥ 2 → x ≥ r :=
by sorry

end NUMINAMATH_CALUDE_minimum_growth_rate_for_doubling_output_l2005_200580


namespace NUMINAMATH_CALUDE_family_travel_info_l2005_200599

structure FamilyMember where
  name : String
  statement : String

structure TravelInfo where
  origin : String
  destination : String
  stopover : Option String

def father : FamilyMember :=
  { name := "Father", statement := "We are going to Spain (we are coming from Newcastle)." }

def mother : FamilyMember :=
  { name := "Mother", statement := "We are not going to Spain but are coming from Newcastle (we stopped in Paris and are not going to Spain)." }

def daughter : FamilyMember :=
  { name := "Daughter", statement := "We are not coming from Newcastle (we stopped in Paris)." }

def family : List FamilyMember := [father, mother, daughter]

def interpretStatements (family : List FamilyMember) : TravelInfo :=
  { origin := "Newcastle", destination := "", stopover := some "Paris" }

theorem family_travel_info (family : List FamilyMember) :
  interpretStatements family = { origin := "Newcastle", destination := "", stopover := some "Paris" } :=
sorry

end NUMINAMATH_CALUDE_family_travel_info_l2005_200599


namespace NUMINAMATH_CALUDE_root_product_sum_l2005_200591

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 5 * p^2 + 20 * p - 10 = 0) →
  (6 * q^3 - 5 * q^2 + 20 * q - 10 = 0) →
  (6 * r^3 - 5 * r^2 + 20 * r - 10 = 0) →
  p * q + p * r + q * r = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l2005_200591


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l2005_200571

theorem max_value_of_a : ∀ a : ℝ, 
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧ 
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two : 
  ∃ a : ℝ, a = -2 ∧
  (∀ x : ℝ, x < a → x^2 - x - 6 > 0) ∧
  (∃ x : ℝ, x^2 - x - 6 > 0 ∧ x ≥ a) ∧
  (∀ b : ℝ, b > a →
    (∃ x : ℝ, x < b ∧ x^2 - x - 6 ≤ 0) ∨
    (∀ x : ℝ, x^2 - x - 6 > 0 → x < b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l2005_200571


namespace NUMINAMATH_CALUDE_like_terms_exponents_l2005_200552

/-- Given that 3x^(2n-1)y^m and -5x^m y^3 are like terms, prove that m = 3 and n = 2 -/
theorem like_terms_exponents (m n : ℤ) : 
  (∀ x y : ℝ, 3 * x^(2*n - 1) * y^m = -5 * x^m * y^3) → 
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l2005_200552


namespace NUMINAMATH_CALUDE_system_solutions_l2005_200510

/-- The system of equations has two solutions with distance 10 between them -/
theorem system_solutions (a : ℝ) : 
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.sin (2 * a) - x₁ * Real.cos (2 * a))) ∧
    (x₁^2 + y₁^2 = 26 * (y₁ * Real.cos (3 * a) - x₁ * Real.sin (3 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.sin (2 * a) - x₂ * Real.cos (2 * a))) ∧
    (x₂^2 + y₂^2 = 26 * (y₂ * Real.cos (3 * a) - x₂ * Real.sin (3 * a))) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 100)) ↔
  (∃ n : ℤ, 
    (a = π / 10 + 2 * π * n / 5) ∨
    (a = π / 10 + (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5) ∨
    (a = π / 10 - (2 / 5) * Real.arctan (12 / 5) + 2 * π * n / 5)) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2005_200510


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2005_200583

theorem other_root_of_quadratic (p : ℝ) : 
  (2 : ℝ)^2 + 4*2 - p = 0 → 
  ∃ (x : ℝ), x^2 + 4*x - p = 0 ∧ x = -6 := by
sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2005_200583


namespace NUMINAMATH_CALUDE_fewest_printers_equal_spend_l2005_200514

def printer_cost_1 : ℕ := 400
def printer_cost_2 : ℕ := 350

theorem fewest_printers_equal_spend (cost1 cost2 : ℕ) (h1 : cost1 = printer_cost_1) (h2 : cost2 = printer_cost_2) :
  ∃ (n1 n2 : ℕ), n1 * cost1 = n2 * cost2 ∧ n1 + n2 = 15 ∧ ∀ (m1 m2 : ℕ), m1 * cost1 = m2 * cost2 → m1 + m2 ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_fewest_printers_equal_spend_l2005_200514


namespace NUMINAMATH_CALUDE_average_difference_l2005_200595

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 40) 
  (hbc : (b + c) / 2 = 60) : 
  c - a = 40 := by
sorry

end NUMINAMATH_CALUDE_average_difference_l2005_200595


namespace NUMINAMATH_CALUDE_distance_less_than_radius_l2005_200506

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point P inside the circle -/
structure PointInside (c : Circle) :=
  (P : ℝ × ℝ)
  (h_inside : dist P c.O < c.radius)

/-- Theorem: The distance between the center and a point inside the circle is less than 3 -/
theorem distance_less_than_radius (c : Circle) (p : PointInside c) :
  dist p.P c.O < 3 := by sorry

end NUMINAMATH_CALUDE_distance_less_than_radius_l2005_200506


namespace NUMINAMATH_CALUDE_consecutive_right_triangle_iff_345_l2005_200543

/-- A right-angled triangle with consecutive integer side lengths -/
structure ConsecutiveRightTriangle where
  n : ℕ
  n_pos : 0 < n
  is_right : (n + 1)^2 + n^2 = (n + 2)^2

/-- The property of having sides 3, 4, and 5 -/
def has_sides_345 (t : ConsecutiveRightTriangle) : Prop :=
  t.n = 3

theorem consecutive_right_triangle_iff_345 :
  ∀ t : ConsecutiveRightTriangle, has_sides_345 t ↔ True :=
sorry

end NUMINAMATH_CALUDE_consecutive_right_triangle_iff_345_l2005_200543


namespace NUMINAMATH_CALUDE_grass_field_width_l2005_200533

/-- The width of a rectangular grass field with specific conditions -/
theorem grass_field_width : ∃ (w : ℝ), w = 40 ∧ w > 0 := by
  -- Define the length of the grass field
  let length : ℝ := 75

  -- Define the width of the path
  let path_width : ℝ := 2.5

  -- Define the cost per square meter of the path
  let cost_per_sqm : ℝ := 2

  -- Define the total cost of the path
  let total_cost : ℝ := 1200

  -- The width w satisfies the equation:
  -- 2 * (80 * (w + 5) - 75 * w) = 1200
  -- where 80 = length + 2 * path_width
  -- and 75 = length

  sorry

end NUMINAMATH_CALUDE_grass_field_width_l2005_200533


namespace NUMINAMATH_CALUDE_sum_of_valid_a_values_l2005_200597

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
              (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1))) ∧
  (∀ a : ℤ, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
             (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1)) → a ∈ S) ∧
  (Finset.sum S (λ a => a) = 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_values_l2005_200597


namespace NUMINAMATH_CALUDE_shopkeeper_mango_profit_l2005_200528

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percent (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: A shopkeeper who buys mangoes at 6 for 1 rupee and sells them at 3 for 1 rupee makes a 100% profit -/
theorem shopkeeper_mango_profit :
  let cost_price : ℚ := 1 / 6  -- Cost price per mango
  let selling_price : ℚ := 1 / 3  -- Selling price per mango
  profit_percent cost_price selling_price = 100 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_mango_profit_l2005_200528


namespace NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l2005_200549

theorem ratio_sum_squares_theorem (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b = 2 * a) (h5 : c = 4 * a) (h6 : a^2 + b^2 + c^2 = 4725) :
  a + b + c = 105 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_squares_theorem_l2005_200549


namespace NUMINAMATH_CALUDE_unique_valid_number_l2005_200585

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 3-digit number -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit numbers with digit sum 25 that are even -/
def validNumbers : Set ℕ := {n : ℕ | isThreeDigit n ∧ digitSum n = 25 ∧ Even n}

theorem unique_valid_number : ∃! n : ℕ, n ∈ validNumbers := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2005_200585


namespace NUMINAMATH_CALUDE_min_questions_100_boxes_l2005_200500

/-- Represents the setup of the box guessing game -/
structure BoxGame where
  num_boxes : ℕ
  num_questions : ℕ

/-- Checks if the number of questions is sufficient to determine the prize box -/
def is_sufficient (game : BoxGame) : Prop :=
  game.num_questions + 1 ≥ game.num_boxes

/-- The minimum number of questions needed for a given number of boxes -/
def min_questions (n : ℕ) : ℕ :=
  n - 1

/-- Theorem stating the minimum number of questions needed for 100 boxes -/
theorem min_questions_100_boxes :
  ∃ (game : BoxGame), game.num_boxes = 100 ∧ game.num_questions = 99 ∧ 
  is_sufficient game ∧ 
  ∀ (g : BoxGame), g.num_boxes = 100 → g.num_questions < 99 → ¬is_sufficient g :=
by sorry


end NUMINAMATH_CALUDE_min_questions_100_boxes_l2005_200500


namespace NUMINAMATH_CALUDE_slope_range_l2005_200566

theorem slope_range (m : ℝ) : ((8 - m) / (m - 5) > 1) → (5 < m ∧ m < 13/2) := by
  sorry

end NUMINAMATH_CALUDE_slope_range_l2005_200566


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2005_200579

theorem simplify_and_evaluate : (Real.sqrt 2 + 1)^2 - 2*(Real.sqrt 2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2005_200579


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l2005_200561

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem: The number of ways to arrange 5 blue, 8 red, and 11 white bulbs in a garland 
    with no consecutive white bulbs is equal to (13 choose 5) * (14 choose 11) -/
theorem bulb_arrangement_count : bulb_arrangements 5 8 11 = 468468 := by
  sorry

#eval bulb_arrangements 5 8 11

end NUMINAMATH_CALUDE_bulb_arrangement_count_l2005_200561


namespace NUMINAMATH_CALUDE_lyssa_incorrect_percentage_is_12_l2005_200502

def exam_items : ℕ := 75
def precious_mistakes : ℕ := 12
def lyssa_additional_correct : ℕ := 3

def lyssa_incorrect_percentage : ℚ :=
  (exam_items - (exam_items - precious_mistakes + lyssa_additional_correct)) / exam_items * 100

theorem lyssa_incorrect_percentage_is_12 :
  lyssa_incorrect_percentage = 12 := by sorry

end NUMINAMATH_CALUDE_lyssa_incorrect_percentage_is_12_l2005_200502


namespace NUMINAMATH_CALUDE_k_value_l2005_200586

theorem k_value : ∃ k : ℝ, (24 / k = 4) ∧ (k = 6) := by
  sorry

end NUMINAMATH_CALUDE_k_value_l2005_200586


namespace NUMINAMATH_CALUDE_wage_increase_l2005_200567

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) 
  (h1 : original_wage = 60)
  (h2 : increase_percentage = 20) : 
  original_wage * (1 + increase_percentage / 100) = 72 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_l2005_200567


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2005_200550

/-- Given a hyperbola with equation x²/a² - y²/9 = 1 where a > 0,
    if one of its asymptotes is y = 3/5 * x, then a = 5 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 9 = 1) →
  (∃ x y : ℝ, y = 3/5 * x) →
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2005_200550


namespace NUMINAMATH_CALUDE_coefficient_x3y3_l2005_200598

/-- The coefficient of x³y³ in the expansion of (x+2y)(x+y)⁵ is 30 -/
theorem coefficient_x3y3 : Int :=
  30

#check coefficient_x3y3

end NUMINAMATH_CALUDE_coefficient_x3y3_l2005_200598


namespace NUMINAMATH_CALUDE_louise_cakes_proof_l2005_200524

/-- The number of cakes Louise needs for the gathering -/
def total_cakes : ℕ := 60

/-- The number of cakes Louise has already baked -/
def baked_cakes : ℕ := total_cakes / 2

/-- The number of cakes Louise bakes on the second day -/
def second_day_bakes : ℕ := (total_cakes - baked_cakes) / 2

/-- The number of cakes Louise bakes on the third day -/
def third_day_bakes : ℕ := (total_cakes - baked_cakes - second_day_bakes) / 3

/-- The number of cakes left to bake after the third day -/
def remaining_cakes : ℕ := total_cakes - baked_cakes - second_day_bakes - third_day_bakes

theorem louise_cakes_proof : remaining_cakes = 10 := by
  sorry

#eval total_cakes
#eval remaining_cakes

end NUMINAMATH_CALUDE_louise_cakes_proof_l2005_200524


namespace NUMINAMATH_CALUDE_linear_function_property_l2005_200507

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) (hcond : g 4 - g 1 = 9) : 
  g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2005_200507


namespace NUMINAMATH_CALUDE_conference_arrangements_l2005_200525

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial / (2^k)

theorem conference_arrangements :
  number_of_arrangements 8 2 = 10080 := by
  sorry

end NUMINAMATH_CALUDE_conference_arrangements_l2005_200525


namespace NUMINAMATH_CALUDE_price_reduction_l2005_200504

theorem price_reduction (x : ℝ) : 
  (1 - x / 100) * (1 - 25 / 100) = 1 - 77.5 / 100 → x = 70 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l2005_200504


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_multiple_of_seven_l2005_200588

theorem sum_seven_consecutive_integers_multiple_of_seven (n : ℕ+) :
  ∃ k : ℕ, n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_multiple_of_seven_l2005_200588


namespace NUMINAMATH_CALUDE_weight_difference_l2005_200587

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 50 →
  (w_a + w_b + w_c + w_d) / 4 = 53 →
  (w_b + w_c + w_d + w_e) / 4 = 51 →
  w_a = 73 →
  w_e > w_d →
  w_e - w_d = 3 := by
sorry

end NUMINAMATH_CALUDE_weight_difference_l2005_200587


namespace NUMINAMATH_CALUDE_big_n_conference_teams_l2005_200572

theorem big_n_conference_teams (n : ℕ) : n * (n - 1) / 2 = 28 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_big_n_conference_teams_l2005_200572


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l2005_200541

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (3 + Complex.I) * m - (2 + Complex.I)

-- Define what it means for a complex number to be in the third quadrant
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- State the theorem
theorem third_quadrant_condition (m : ℝ) :
  in_third_quadrant (z m) ↔ m < 0 := by sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l2005_200541


namespace NUMINAMATH_CALUDE_triangle_rotation_l2005_200539

/-- Triangle OPQ with specific properties -/
structure TriangleOPQ where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_O : O = (0, 0)
  h_Q : Q = (6, 0)
  h_P_first_quadrant : P.1 > 0 ∧ P.2 > 0
  h_right_angle : (P.1 - Q.1) * (Q.1 - O.1) + (P.2 - Q.2) * (Q.2 - O.2) = 0
  h_45_degree : (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) * Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) / Real.sqrt 2

/-- Rotation of a point 90 degrees counterclockwise about the origin -/
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

/-- The main theorem -/
theorem triangle_rotation (t : TriangleOPQ) : rotate90 t.P = (-6, 6) := by
  sorry


end NUMINAMATH_CALUDE_triangle_rotation_l2005_200539


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_84_l2005_200559

theorem gcf_lcm_sum_36_56_84 : 
  let a := 36
  let b := 56
  let c := 84
  Nat.gcd a (Nat.gcd b c) + Nat.lcm a (Nat.lcm b c) = 516 := by
sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_36_56_84_l2005_200559


namespace NUMINAMATH_CALUDE_power_relation_l2005_200577

theorem power_relation (m n : ℤ) : 
  (3 : ℝ) ^ m = (1 : ℝ) / 27 → 
  ((1 : ℝ) / 2) ^ n = 16 → 
  (m : ℝ) ^ n = 1 / 81 := by
sorry

end NUMINAMATH_CALUDE_power_relation_l2005_200577


namespace NUMINAMATH_CALUDE_range_of_a_l2005_200516

open Set Real

theorem range_of_a (p q : Prop) (h : ¬(p ∧ q)) : 
  ∀ a : ℝ, (∀ x ∈ Icc 0 1, a ≥ exp x) = p → 
  (∃ x₀ : ℝ, x₀^2 + 4*x₀ + a = 0) = q → 
  a ∈ Ioi 4 ∪ Iic (exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2005_200516


namespace NUMINAMATH_CALUDE_chair_and_vase_cost_indeterminate_l2005_200531

/-- Represents the cost of items at a garage sale. -/
structure GarageSale where
  total : ℝ
  table : ℝ
  chairs : ℕ
  mirror : ℝ
  lamp : ℝ
  vases : ℕ
  chair_cost : ℝ
  vase_cost : ℝ

/-- Conditions of Nadine's garage sale purchase -/
def nadines_purchase : GarageSale where
  total := 105
  table := 34
  chairs := 2
  mirror := 15
  lamp := 6
  vases := 3
  chair_cost := 0  -- placeholder, actual value unknown
  vase_cost := 0   -- placeholder, actual value unknown

/-- Theorem stating that the sum of one chair and one vase cost cannot be uniquely determined -/
theorem chair_and_vase_cost_indeterminate (g : GarageSale) (h : g = nadines_purchase) :
  ¬ ∃! x : ℝ, x = g.chair_cost + g.vase_cost ∧
    g.total = g.table + g.mirror + g.lamp + g.chairs * g.chair_cost + g.vases * g.vase_cost :=
sorry

end NUMINAMATH_CALUDE_chair_and_vase_cost_indeterminate_l2005_200531


namespace NUMINAMATH_CALUDE_binomial_700_700_l2005_200520

theorem binomial_700_700 : Nat.choose 700 700 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_700_700_l2005_200520


namespace NUMINAMATH_CALUDE_total_pages_in_textbooks_l2005_200554

/-- Represents the number of pages in each textbook and calculates the total --/
def textbook_pages : ℕ → ℕ → ℕ → ℕ → ℕ := fun history geography math science =>
  history + geography + math + science

/-- Theorem stating the total number of pages in Suzanna's textbooks --/
theorem total_pages_in_textbooks : ∃ (history geography math science : ℕ),
  history = 160 ∧
  geography = history + 70 ∧
  math = (history + geography) / 2 ∧
  science = 2 * history ∧
  textbook_pages history geography math science = 905 := by
  sorry

#eval textbook_pages 160 230 195 320

end NUMINAMATH_CALUDE_total_pages_in_textbooks_l2005_200554


namespace NUMINAMATH_CALUDE_trader_profit_loss_percentage_trader_overall_loss_l2005_200518

/-- Calculates the overall profit or loss percentage for a trader selling two cars -/
theorem trader_profit_loss_percentage 
  (selling_price : ℝ) 
  (gain_percentage : ℝ) 
  (loss_percentage : ℝ) : ℝ :=
  let cost_price1 := selling_price / (1 + gain_percentage / 100)
  let cost_price2 := selling_price / (1 - loss_percentage / 100)
  let total_cost := cost_price1 + cost_price2
  let total_selling := 2 * selling_price
  let profit_loss := total_selling - total_cost
  (profit_loss / total_cost) * 100

/-- Proof that the trader's overall loss is approximately 1.44% -/
theorem trader_overall_loss :
  ∃ ε > 0, abs (trader_profit_loss_percentage 325475 12 12 + 1.44) < ε :=
sorry

end NUMINAMATH_CALUDE_trader_profit_loss_percentage_trader_overall_loss_l2005_200518


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_one_l2005_200537

theorem sin_plus_cos_equals_one (x : ℝ) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x + Real.cos x = 1 → x = 0 ∨ x = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_one_l2005_200537


namespace NUMINAMATH_CALUDE_games_this_month_l2005_200547

/-- Represents the number of football games Nancy attended or plans to attend -/
structure FootballGames where
  total : Nat
  lastMonth : Nat
  nextMonth : Nat

/-- Theorem stating that Nancy attended 9 games this month -/
theorem games_this_month (nancy : FootballGames) 
  (h1 : nancy.total = 24) 
  (h2 : nancy.lastMonth = 8) 
  (h3 : nancy.nextMonth = 7) : 
  nancy.total - nancy.lastMonth - nancy.nextMonth = 9 := by
  sorry

#check games_this_month

end NUMINAMATH_CALUDE_games_this_month_l2005_200547


namespace NUMINAMATH_CALUDE_non_working_games_l2005_200584

theorem non_working_games (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) : 
  total_games = 15 → price_per_game = 5 → total_earnings = 30 → 
  total_games - (total_earnings / price_per_game) = 9 := by
sorry

end NUMINAMATH_CALUDE_non_working_games_l2005_200584


namespace NUMINAMATH_CALUDE_smallest_k_remainder_l2005_200522

theorem smallest_k_remainder (k : ℕ) : 
  k > 0 ∧ 
  k % 5 = 2 ∧ 
  k % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 6 = 5 → k ≤ m) → 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_remainder_l2005_200522


namespace NUMINAMATH_CALUDE_extreme_points_range_l2005_200530

/-- The function f(x) = x^2 + a*ln(1+x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (1 + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (2 * x^2 + 2 * x + a) / (1 + x)

theorem extreme_points_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    f_derivative a x = 0 ∧ 
    f_derivative a y = 0 ∧ 
    (∀ z : ℝ, f_derivative a z = 0 → z = x ∨ z = y)) →
  (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_range_l2005_200530


namespace NUMINAMATH_CALUDE_coordinate_axis_angles_characterization_l2005_200503

-- Define the set of angles whose terminal sides lie on the coordinate axes
def CoordinateAxisAngles : Set ℝ :=
  {α | ∃ n : ℤ, α = n * Real.pi / 2}

-- Theorem stating that the set of angles whose terminal sides lie on the coordinate axes
-- is equal to the set {α | α = nπ/2, n ∈ ℤ}
theorem coordinate_axis_angles_characterization :
  CoordinateAxisAngles = {α | ∃ n : ℤ, α = n * Real.pi / 2} := by
  sorry

end NUMINAMATH_CALUDE_coordinate_axis_angles_characterization_l2005_200503


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2005_200542

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_achievable :
  ∃ x : ℝ, x > -1 ∧ x - 4 + 9 / (x + 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l2005_200542


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l2005_200568

/-- The thickness of the paper after n folds -/
def thickness (n : ℕ) : ℚ := (1 / 10) * 2^n

/-- The minimum number of folds required to exceed 12mm -/
def min_folds : ℕ := 7

theorem paper_folding_thickness :
  (∀ k < min_folds, thickness k ≤ 12) ∧ thickness min_folds > 12 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l2005_200568


namespace NUMINAMATH_CALUDE_consecutive_integers_coprime_l2005_200592

theorem consecutive_integers_coprime (n : ℤ) : 
  ∃ k ∈ Finset.range 10, ∀ m ∈ Finset.range 10, m ≠ k → Int.gcd (n + k) (n + m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_coprime_l2005_200592


namespace NUMINAMATH_CALUDE_art_project_markers_l2005_200557

/-- Calculates the total number of markers needed for an art project given the distribution of markers among student groups. -/
theorem art_project_markers (total_students : ℕ) (group1_students : ℕ) (group2_students : ℕ) 
  (group1_markers_per_student : ℕ) (group2_markers_per_student : ℕ) (group3_markers_per_student : ℕ) :
  total_students = 30 →
  group1_students = 10 →
  group2_students = 15 →
  group1_markers_per_student = 2 →
  group2_markers_per_student = 4 →
  group3_markers_per_student = 6 →
  (group1_students * group1_markers_per_student + 
   group2_students * group2_markers_per_student + 
   (total_students - group1_students - group2_students) * group3_markers_per_student) = 110 :=
by sorry


end NUMINAMATH_CALUDE_art_project_markers_l2005_200557


namespace NUMINAMATH_CALUDE_simplify_expression_l2005_200501

theorem simplify_expression (y : ℝ) (h : y^2 ≥ 49) :
  (7 - Real.sqrt (y^2 - 49))^2 = y^2 - 14 * Real.sqrt (y^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2005_200501


namespace NUMINAMATH_CALUDE_james_chore_time_l2005_200526

/-- The total time James spends on all chores -/
def total_chore_time (vacuum_time cleaning_time laundry_time organizing_time : ℝ) : ℝ :=
  vacuum_time + cleaning_time + laundry_time + organizing_time

/-- Theorem stating the total time James spends on chores -/
theorem james_chore_time :
  ∃ (vacuum_time cleaning_time laundry_time organizing_time : ℝ),
    vacuum_time = 3 ∧
    cleaning_time = 3 * vacuum_time ∧
    laundry_time = (1/2) * cleaning_time ∧
    organizing_time = 2 * (vacuum_time + cleaning_time + laundry_time) ∧
    total_chore_time vacuum_time cleaning_time laundry_time organizing_time = 49.5 :=
by
  sorry

end NUMINAMATH_CALUDE_james_chore_time_l2005_200526


namespace NUMINAMATH_CALUDE_scientific_notation_of_2410000_l2005_200513

theorem scientific_notation_of_2410000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2410000 = a * (10 : ℝ) ^ n ∧ a = 2.41 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_2410000_l2005_200513


namespace NUMINAMATH_CALUDE_can_make_all_white_l2005_200540

/-- Represents the color of a number -/
inductive Color
| Black
| White

/-- Represents a move in the repainting process -/
structure Move where
  number : Nat
  deriving Repr

/-- The state of all numbers from 1 to 1,000,000 -/
def State := Fin 1000000 → Color

/-- Apply a move to a state -/
def applyMove (s : State) (m : Move) : State :=
  sorry

/-- Check if all numbers in the state are white -/
def allWhite (s : State) : Prop :=
  sorry

/-- The initial state where all numbers are black -/
def initialState : State :=
  sorry

/-- Theorem stating that it's possible to make all numbers white -/
theorem can_make_all_white : ∃ (moves : List Move), allWhite (moves.foldl applyMove initialState) := by
  sorry

end NUMINAMATH_CALUDE_can_make_all_white_l2005_200540


namespace NUMINAMATH_CALUDE_initial_birds_on_fence_l2005_200545

theorem initial_birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) : 
  (initial_birds + 5 = initial_storks + 4 + 3) → initial_birds = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_on_fence_l2005_200545


namespace NUMINAMATH_CALUDE_multiple_of_seven_proposition_l2005_200517

theorem multiple_of_seven_proposition : 
  (∃ k : ℤ, 47 = 7 * k) ∨ (∃ m : ℤ, 49 = 7 * m) := by sorry

end NUMINAMATH_CALUDE_multiple_of_seven_proposition_l2005_200517


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2005_200534

/-- The value of 'a' for which the line ax - y + 2 = 0 is tangent to the circle
    x = 2 + 2cos(θ), y = 1 + 2sin(θ) -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ θ : ℝ, (a * (2 + 2 * Real.cos θ) - (1 + 2 * Real.sin θ) + 2 = 0) →
   ∃ θ' : ℝ, (a * (2 + 2 * Real.cos θ') - (1 + 2 * Real.sin θ') + 2 = 0 ∧
              ∀ θ'' : ℝ, θ'' ≠ θ' → 
                a * (2 + 2 * Real.cos θ'') - (1 + 2 * Real.sin θ'') + 2 ≠ 0)) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2005_200534


namespace NUMINAMATH_CALUDE_digit_change_sum_inequality_l2005_200512

/-- Changes each digit of a positive integer by 1 (either up or down) -/
def change_digits (n : ℕ) : ℕ :=
  sorry

theorem digit_change_sum_inequality (a b : ℕ) :
  let c := a + b
  let a' := change_digits a
  let b' := change_digits b
  let c' := change_digits c
  a' + b' ≠ c' :=
by sorry

end NUMINAMATH_CALUDE_digit_change_sum_inequality_l2005_200512


namespace NUMINAMATH_CALUDE_georgia_muffins_l2005_200515

/-- Calculates the number of batches of muffins made over a period of months -/
def muffin_batches (students : ℕ) (muffins_per_batch : ℕ) (months : ℕ) : ℕ :=
  (students / muffins_per_batch) * months

theorem georgia_muffins :
  muffin_batches 24 6 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_georgia_muffins_l2005_200515


namespace NUMINAMATH_CALUDE_parabola_b_value_l2005_200521

/-- Prove that for a parabola y = 2x^2 + bx + 3 passing through (1, 2) and (-2, -1), b = 11/2 -/
theorem parabola_b_value (b : ℝ) : 
  (2 * (1 : ℝ)^2 + b * 1 + 3 = 2) ∧ 
  (2 * (-2 : ℝ)^2 + b * (-2) + 3 = -1) → 
  b = 11/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2005_200521


namespace NUMINAMATH_CALUDE_no_square_divisibility_l2005_200511

theorem no_square_divisibility (a b : ℕ) (α : ℕ) (ha : a > 1) (hb : b > 1) 
  (hodd_a : Odd a) (hodd_b : Odd b) (hsum : a + b = 2^α) (hα : α ≥ 1) :
  ¬∃ (k : ℕ), k > 1 ∧ (k^2 ∣ a^k + b^k) := by
  sorry

end NUMINAMATH_CALUDE_no_square_divisibility_l2005_200511


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l2005_200519

theorem sum_remainder_mod_11 : (103104 + 103105 + 103106 + 103107 + 103108 + 103109 + 103110 + 103111 + 103112) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l2005_200519


namespace NUMINAMATH_CALUDE_reinforcement_is_1900_l2005_200594

/-- Calculates the reinforcement size given initial garrison size, provision duration, and remaining provision duration after reinforcement --/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  let provisions_left := initial_garrison * (initial_duration - days_before_reinforcement)
  (provisions_left / remaining_duration) - initial_garrison

/-- The reinforcement size for the given problem --/
def problem_reinforcement : ℕ := calculate_reinforcement 2000 54 15 20

/-- Theorem stating that the reinforcement size for the given problem is 1900 --/
theorem reinforcement_is_1900 : problem_reinforcement = 1900 := by
  sorry

#eval problem_reinforcement

end NUMINAMATH_CALUDE_reinforcement_is_1900_l2005_200594


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2005_200560

theorem function_inequality_implies_a_bound (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, 
    x₁ + 4/x₁ ≥ 2^x₂ + a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l2005_200560


namespace NUMINAMATH_CALUDE_simplify_expression_l2005_200593

theorem simplify_expression (a b : ℝ) : (30*a + 70*b) + (15*a + 45*b) - (12*a + 60*b) = 33*a + 55*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2005_200593


namespace NUMINAMATH_CALUDE_mess_expense_increase_l2005_200573

theorem mess_expense_increase
  (initial_students : ℕ)
  (new_students : ℕ)
  (original_expenditure : ℕ)
  (average_decrease : ℕ)
  (h1 : initial_students = 35)
  (h2 : new_students = 7)
  (h3 : original_expenditure = 420)
  (h4 : average_decrease = 1)
  : (initial_students + new_students) * 
    (original_expenditure / initial_students - average_decrease) - 
    original_expenditure = 42 := by
  sorry

end NUMINAMATH_CALUDE_mess_expense_increase_l2005_200573


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2005_200548

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, 4/5), prove that cos(α) = -3/5 -/
theorem cos_alpha_value (α : Real) (h1 : ∃ (x y : Real), x = -3/5 ∧ y = 4/5 ∧ 
  (Real.cos α = x ∧ Real.sin α = y)) : Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2005_200548


namespace NUMINAMATH_CALUDE_card_distribution_convergence_l2005_200578

/-- Represents a person in the circular arrangement -/
structure Person where
  id : Nat
  cards : Nat

/-- Represents the state of the card distribution -/
structure CardState where
  people : List Person
  total_cards : Nat

/-- Defines a valid move in the card game -/
def valid_move (state : CardState) (giver : Nat) : Prop :=
  ∃ (p : Person), p ∈ state.people ∧ p.id = giver ∧ p.cards ≥ 2

/-- Defines the result of a move -/
def move_result (state : CardState) (giver : Nat) : CardState :=
  sorry

/-- Defines a sequence of moves -/
def move_sequence (initial : CardState) : List Nat → CardState
  | [] => initial
  | (m :: ms) => move_result (move_sequence initial ms) m

/-- The main theorem to be proved -/
theorem card_distribution_convergence 
  (n : Nat) 
  (h : n > 1) :
  ∃ (initial : CardState) (moves : List Nat),
    (initial.people.length = n) ∧ 
    (initial.total_cards = n - 1) ∧
    (∀ (p : Person), p ∈ (move_sequence initial moves).people → p.cards ≤ 1) :=
  sorry

end NUMINAMATH_CALUDE_card_distribution_convergence_l2005_200578


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2005_200581

/-- A quadratic function with specific properties -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^2 + B * x + C

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 3.5 and p(0) = 2, p(20) = 2 -/
theorem quadratic_symmetry (A B C : ℝ) :
  (∀ x : ℝ, p A B C (3.5 + x) = p A B C (3.5 - x)) →  -- Axis of symmetry at x = 3.5
  p A B C 0 = 2 →                                     -- p(0) = 2
  p A B C 20 = 2 :=                                   -- Conclusion: p(20) = 2
by
  sorry


end NUMINAMATH_CALUDE_quadratic_symmetry_l2005_200581


namespace NUMINAMATH_CALUDE_product_of_fractions_l2005_200508

theorem product_of_fractions : (2 : ℚ) / 3 * 3 / 4 * 4 / 5 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2005_200508


namespace NUMINAMATH_CALUDE_min_cans_required_l2005_200574

def can_capacity : ℕ := 10
def tank_capacity : ℕ := 140

theorem min_cans_required : 
  ∃ n : ℕ, n * can_capacity ≥ tank_capacity ∧ 
  ∀ m : ℕ, m * can_capacity ≥ tank_capacity → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_cans_required_l2005_200574


namespace NUMINAMATH_CALUDE_problem_solution_l2005_200532

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10 →
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841 / 100 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2005_200532


namespace NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2005_200565

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_equals_surface_area_l2005_200565


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2005_200551

theorem perfect_square_condition (k : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + k*x + 25 = y^2) → (k = 10 ∨ k = -10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2005_200551


namespace NUMINAMATH_CALUDE_number_problem_l2005_200535

theorem number_problem (N : ℚ) : 
  (4/15 * 5/7 * N) - (4/9 * 2/5 * N) = 24 → N/2 = 945 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2005_200535


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2005_200544

/-- A function f is increasing on ℝ if for all x y, x < y implies f x < f y -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_condition (f : ℝ → ℝ) (h : IncreasingOn f) :
  ∀ a b : ℝ, a + b < 0 ↔ f a + f b < f (-a) + f (-b) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2005_200544


namespace NUMINAMATH_CALUDE_equation_solution_l2005_200509

theorem equation_solution : ∃ x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x + 1) = (729 : ℝ) ^ (x + 2) ∧ x = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2005_200509


namespace NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2005_200589

/-- A line y = 2x + m is tangent to the curve y = x ln x if and only if m = -e -/
theorem tangent_line_to_x_ln_x (m : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (2 * x₀ + m = x₀ * Real.log x₀) ∧ 
    (2 = Real.log x₀ + 1)) ↔ 
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_x_ln_x_l2005_200589


namespace NUMINAMATH_CALUDE_rectangle_length_is_one_point_five_times_width_l2005_200546

/-- Represents the configuration of squares and rectangles in a larger square -/
structure SquareConfiguration where
  /-- Side length of a small square -/
  s : ℝ
  /-- Length of a rectangle -/
  l : ℝ
  /-- The configuration forms a square -/
  is_square : 3 * s = 2 * l
  /-- The width of each rectangle equals the side of a small square -/
  width_eq_side : l > s

/-- Theorem stating that the length of each rectangle is 1.5 times its width -/
theorem rectangle_length_is_one_point_five_times_width (config : SquareConfiguration) :
  config.l = 1.5 * config.s := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_is_one_point_five_times_width_l2005_200546


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2005_200564

theorem factorial_fraction_equality : (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2005_200564


namespace NUMINAMATH_CALUDE_equality_of_exponential_equation_l2005_200555

theorem equality_of_exponential_equation (a b : ℝ) : 
  0 < a → 0 < b → a < 1 → a^b = b^a → a = b := by sorry

end NUMINAMATH_CALUDE_equality_of_exponential_equation_l2005_200555


namespace NUMINAMATH_CALUDE_spinner_points_south_l2005_200563

/-- Represents the four cardinal directions --/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a rotation of the spinner --/
structure Rotation :=
  (revolutions : ℚ)
  (clockwise : Bool)

/-- Calculates the final direction after applying a net rotation --/
def finalDirection (netRotation : ℚ) : Direction :=
  match netRotation.num % 4 with
  | 0 => Direction.North
  | 1 => Direction.East
  | 2 => Direction.South
  | _ => Direction.West

/-- Theorem stating that the given sequence of rotations results in the spinner pointing south --/
theorem spinner_points_south (initialDirection : Direction)
    (rotation1 : Rotation)
    (rotation2 : Rotation)
    (rotation3 : Rotation) :
    initialDirection = Direction.North ∧
    rotation1 = { revolutions := 7/2, clockwise := true } ∧
    rotation2 = { revolutions := 16/3, clockwise := false } ∧
    rotation3 = { revolutions := 13/6, clockwise := true } →
    finalDirection (
      rotation1.revolutions * (if rotation1.clockwise then 1 else -1) +
      rotation2.revolutions * (if rotation2.clockwise then 1 else -1) +
      rotation3.revolutions * (if rotation3.clockwise then 1 else -1)
    ) = Direction.South :=
by sorry

end NUMINAMATH_CALUDE_spinner_points_south_l2005_200563


namespace NUMINAMATH_CALUDE_fishing_line_length_l2005_200527

/-- Given information about fishing line reels and sections, prove the length of each reel. -/
theorem fishing_line_length (num_reels : ℕ) (section_length : ℝ) (num_sections : ℕ) :
  num_reels = 3 →
  section_length = 10 →
  num_sections = 30 →
  (num_sections * section_length) / num_reels = 100 := by
  sorry

#check fishing_line_length

end NUMINAMATH_CALUDE_fishing_line_length_l2005_200527


namespace NUMINAMATH_CALUDE_first_puncture_time_l2005_200570

/-- Given a tyre with two punctures, this theorem proves the time it takes
    for the first puncture alone to flatten the tyre. -/
theorem first_puncture_time
  (second_puncture_time : ℝ)
  (both_punctures_time : ℝ)
  (h1 : second_puncture_time = 6)
  (h2 : both_punctures_time = 336 / 60)
  (h3 : both_punctures_time > 0) :
  ∃ (first_puncture_time : ℝ),
    first_puncture_time > 0 ∧
    1 / first_puncture_time + 1 / second_puncture_time = 1 / both_punctures_time ∧
    first_puncture_time = 84 := by
  sorry

end NUMINAMATH_CALUDE_first_puncture_time_l2005_200570


namespace NUMINAMATH_CALUDE_length_of_A_l2005_200576

def A : ℝ × ℝ := (0, 9)
def B : ℝ × ℝ := (0, 12)
def C : ℝ × ℝ := (2, 8)

def on_line_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

def intersect_at (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (p.1 + t * (q.1 - p.1), p.2 + t * (q.2 - p.2))

theorem length_of_A'B' :
  ∀ A' B' : ℝ × ℝ,
  on_line_y_eq_x A' →
  on_line_y_eq_x B' →
  intersect_at A A' C →
  intersect_at B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l2005_200576


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2005_200596

/-- Given a hyperbola with equation 4x^2 - y^2 + 64 = 0, 
    if a point P on this hyperbola is at distance 1 from one focus,
    then it is at distance 17 from the other focus. -/
theorem hyperbola_foci_distance (x y : ℝ) (P : ℝ × ℝ) :
  P.1 = x ∧ P.2 = y →  -- P is the point (x, y)
  4 * x^2 - y^2 + 64 = 0 →  -- P is on the hyperbola
  (∃ F₁ : ℝ × ℝ, (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = 1) →  -- Distance to one focus is 1
  (∃ F₂ : ℝ × ℝ, (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = 17^2) :=  -- Distance to other focus is 17
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2005_200596


namespace NUMINAMATH_CALUDE_sum_of_a_equals_two_l2005_200523

theorem sum_of_a_equals_two (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (eq1 : 2*a₁ + a₂ + a₃ + a₄ + a₅ = 1 + (1/8)*a₄)
  (eq2 : 2*a₂ + a₃ + a₄ + a₅ = 2 + (1/4)*a₃)
  (eq3 : 2*a₃ + a₄ + a₅ = 4 + (1/2)*a₂)
  (eq4 : 2*a₄ + a₅ = 6 + a₁) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_a_equals_two_l2005_200523


namespace NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2005_200558

theorem exponential_function_passes_through_one (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  (fun x => a^x) 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_passes_through_one_l2005_200558


namespace NUMINAMATH_CALUDE_ruels_usable_stamps_l2005_200529

/-- The number of usable stamps Ruel has -/
def usable_stamps : ℕ :=
  let books_10 := 4
  let stamps_per_book_10 := 10
  let books_15 := 6
  let stamps_per_book_15 := 15
  let books_25 := 3
  let stamps_per_book_25 := 25
  let books_30 := 2
  let stamps_per_book_30 := 30
  let damaged_25 := 5
  let damaged_30 := 3
  let total_stamps := books_10 * stamps_per_book_10 +
                      books_15 * stamps_per_book_15 +
                      books_25 * stamps_per_book_25 +
                      books_30 * stamps_per_book_30
  let total_damaged := damaged_25 + damaged_30
  total_stamps - total_damaged

theorem ruels_usable_stamps :
  usable_stamps = 257 := by
  sorry

end NUMINAMATH_CALUDE_ruels_usable_stamps_l2005_200529


namespace NUMINAMATH_CALUDE_coefficient_x_squared_sum_binomials_l2005_200505

theorem coefficient_x_squared_sum_binomials : 
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let sum := (f 4) + (f 5) + (f 6) + (f 7) + (f 8) + (f 9)
  (sum.coeff 2 : ℚ) = 116 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_sum_binomials_l2005_200505


namespace NUMINAMATH_CALUDE_exponent_equation_solution_l2005_200590

theorem exponent_equation_solution :
  ∃ x : ℝ, (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x * (4 : ℝ)^x = (256 : ℝ)^4 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_solution_l2005_200590


namespace NUMINAMATH_CALUDE_card_distribution_events_l2005_200562

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem card_distribution_events :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (there exists a distribution where neither event occurs)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end NUMINAMATH_CALUDE_card_distribution_events_l2005_200562


namespace NUMINAMATH_CALUDE_square_difference_equality_l2005_200569

theorem square_difference_equality : (45 + 18)^2 - (45^2 + 18^2 + 10) = 1610 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2005_200569


namespace NUMINAMATH_CALUDE_younger_brother_age_l2005_200536

/-- Represents the age of Viggo's younger brother -/
def brother_age : ℕ := sorry

/-- Represents Viggo's age -/
def viggo_age : ℕ := sorry

/-- The age difference between Viggo and his brother remains constant -/
axiom age_difference : viggo_age - brother_age = 12

/-- Viggo's age was 10 years more than twice his younger brother's age when his brother was 2 -/
axiom initial_age_relation : viggo_age - brother_age = 2 * 2 + 10 - 2

/-- The sum of their current ages is 32 -/
axiom current_age_sum : brother_age + viggo_age = 32

theorem younger_brother_age : brother_age = 10 := by sorry

end NUMINAMATH_CALUDE_younger_brother_age_l2005_200536
