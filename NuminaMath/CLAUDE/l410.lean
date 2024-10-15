import Mathlib

namespace NUMINAMATH_CALUDE_ludek_unique_stamps_l410_41075

theorem ludek_unique_stamps 
  (karel_mirek : ℕ) 
  (karel_ludek : ℕ) 
  (mirek_ludek : ℕ) 
  (karel_mirek_shared : ℕ) 
  (karel_ludek_shared : ℕ) 
  (mirek_ludek_shared : ℕ) 
  (h1 : karel_mirek = 101) 
  (h2 : karel_ludek = 115) 
  (h3 : mirek_ludek = 110) 
  (h4 : karel_mirek_shared = 5) 
  (h5 : karel_ludek_shared = 12) 
  (h6 : mirek_ludek_shared = 7) : 
  ∃ (ludek_total : ℕ), 
    ludek_total - karel_ludek_shared - mirek_ludek_shared = 43 :=
by sorry

end NUMINAMATH_CALUDE_ludek_unique_stamps_l410_41075


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l410_41037

/-- The line equation: kx - y + 1 = 0 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x - y + 1 = 0

/-- The curve equation: y² = 4x -/
def curve_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- The tangency condition: the discriminant of the resulting quadratic equation is zero -/
def tangency_condition (k : ℝ) : Prop :=
  (4 * k - 8)^2 - 16 * k^2 = 0

theorem line_tangent_to_curve (k : ℝ) :
  (∀ x y : ℝ, line_equation k x y ∧ curve_equation x y → tangency_condition k) →
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l410_41037


namespace NUMINAMATH_CALUDE_parabola_properties_l410_41026

/-- Parabola represented by y = x^2 + bx - 2 -/
structure Parabola where
  b : ℝ

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about a specific parabola and its properties -/
theorem parabola_properties (p : Parabola) 
  (h1 : p.b = 4) -- Derived from the condition that the parabola passes through (1, 3)
  (A : Point) 
  (hA : A.x = 0 ∧ A.y = -2) -- A is the y-axis intersection point
  (B : Point) 
  (hB : B.x = -2 ∧ B.y = -6) -- B is the vertex of the parabola
  (k : ℝ) 
  (hk : k^2 + p.b * k - 2 = 0) -- k is the x-coordinate of x-axis intersection
  : 
  (1/2 * |A.y| * |B.x| = 2) ∧ 
  ((4*k^4 + 3*k^2 + 12*k - 6) / (k^8 + 2*k^6 + k^5 - 2*k^3 + 8*k^2 + 16) = 1/107) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l410_41026


namespace NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l410_41092

theorem pascal_triangle_15th_row_5th_number : Nat.choose 15 4 = 1365 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_15th_row_5th_number_l410_41092


namespace NUMINAMATH_CALUDE_horses_equal_to_four_oxen_l410_41005

/-- The cost of animals in Rupees --/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions of the problem --/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000 ∧
  costs.camel = 4184.615384615385

/-- The theorem to prove --/
theorem horses_equal_to_four_oxen (costs : AnimalCosts) 
  (h : problem_conditions costs) : 
  costs.horse = 4 * costs.ox := by
  sorry

#check horses_equal_to_four_oxen

end NUMINAMATH_CALUDE_horses_equal_to_four_oxen_l410_41005


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l410_41065

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, x^2 - 6*x + 11 = 23 ↔ x = a ∨ x = b) →
  a ≥ b →
  3*a + 2*b = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l410_41065


namespace NUMINAMATH_CALUDE_arcsin_double_angle_l410_41040

theorem arcsin_double_angle (x : ℝ) (θ : ℝ) 
  (h1 : x ∈ Set.Icc (-1) 1) 
  (h2 : Real.arcsin x = θ) 
  (h3 : θ ∈ Set.Icc (-Real.pi/2) (-Real.pi/4)) :
  Real.arcsin (2 * x * Real.sqrt (1 - x^2)) = -(Real.pi + 2*θ) := by
  sorry

end NUMINAMATH_CALUDE_arcsin_double_angle_l410_41040


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l410_41007

def set_A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def set_B : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt p.1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {(0, 0), (1, 1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l410_41007


namespace NUMINAMATH_CALUDE_average_weight_a_b_l410_41094

/-- Given the weights of three people a, b, and c, prove that the average weight of a and b is 40 kg -/
theorem average_weight_a_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (b + c) / 2 = 43 ∧ 
  b = 31 → 
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_a_b_l410_41094


namespace NUMINAMATH_CALUDE_max_value_theorem_l410_41095

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_equal_roots : a^2 = 4*(b-1)) : 
  (∃ (x : ℝ), (3*a + 2*b) / (a + b) ≤ x) ∧ 
  (∀ (y : ℝ), (3*a + 2*b) / (a + b) ≤ y → y ≥ 5/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l410_41095


namespace NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l410_41097

theorem pen_pricing_gain_percentage :
  ∀ (C S : ℝ),
  C > 0 →
  20 * C = 12 * S →
  (S - C) / C * 100 = 200 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pen_pricing_gain_percentage_l410_41097


namespace NUMINAMATH_CALUDE_solution_property_l410_41036

theorem solution_property (m n : ℝ) (hm : m ≠ 0) 
  (h : m^2 + n*m - m = 0) : m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_property_l410_41036


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l410_41051

theorem sufficient_but_not_necessary (a b : ℝ) :
  (∀ a b, (a + b)/2 < Real.sqrt (a * b) → |a + b| = |a| + |b|) ∧
  (∃ a b, |a + b| = |a| + |b| ∧ (a + b)/2 ≥ Real.sqrt (a * b)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l410_41051


namespace NUMINAMATH_CALUDE_roofing_cost_calculation_l410_41080

theorem roofing_cost_calculation (total_needed : ℕ) (cost_per_foot : ℕ) (free_roofing : ℕ) : 
  total_needed = 300 → 
  cost_per_foot = 8 → 
  free_roofing = 250 → 
  (total_needed - free_roofing) * cost_per_foot = 400 := by
  sorry

end NUMINAMATH_CALUDE_roofing_cost_calculation_l410_41080


namespace NUMINAMATH_CALUDE_marble_remainder_l410_41039

theorem marble_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : 
  (r + p) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l410_41039


namespace NUMINAMATH_CALUDE_sugar_solution_problem_l410_41042

/-- Calculates the final sugar percentage when replacing part of a solution --/
def finalSugarPercentage (initialPercent : ℝ) (replacementPercent : ℝ) (replacementFraction : ℝ) : ℝ :=
  (initialPercent * (1 - replacementFraction) + replacementPercent * replacementFraction) * 100

/-- Theorem stating the final sugar percentage for the given problem --/
theorem sugar_solution_problem :
  finalSugarPercentage 0.1 0.42 0.25 = 18 := by
  sorry

#eval finalSugarPercentage 0.1 0.42 0.25

end NUMINAMATH_CALUDE_sugar_solution_problem_l410_41042


namespace NUMINAMATH_CALUDE_square_difference_169_168_l410_41058

theorem square_difference_169_168 : (169 : ℕ)^2 - (168 : ℕ)^2 = 337 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_169_168_l410_41058


namespace NUMINAMATH_CALUDE_hyperbola_minimum_value_l410_41061

theorem hyperbola_minimum_value (a b : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) 
  (h_eccentricity : (a^2 + b^2) / a^2 = 4) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → 
  (∀ a' b', a' ≥ 1 → b' ≥ 1 → (a'^2 + b'^2) / a'^2 = 4 → 
    (b^2 + 1) / (Real.sqrt 3 * a) ≤ (b'^2 + 1) / (Real.sqrt 3 * a')) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_minimum_value_l410_41061


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l410_41082

theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (∀ x : ℝ, f (f x) = (x - 1) * f x + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l410_41082


namespace NUMINAMATH_CALUDE_lucy_grocery_shopping_l410_41043

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The number of cans of soup Lucy bought -/
def soup : ℕ := 28

/-- The number of boxes of cereals Lucy bought -/
def cereals : ℕ := 5

/-- The number of packs of crackers Lucy bought -/
def crackers : ℕ := 45

/-- The total number of packs and boxes Lucy bought -/
def total_packs_and_boxes : ℕ := cookies + noodles + cereals + crackers

theorem lucy_grocery_shopping :
  total_packs_and_boxes = 78 := by sorry

end NUMINAMATH_CALUDE_lucy_grocery_shopping_l410_41043


namespace NUMINAMATH_CALUDE_dvd_rental_cost_l410_41023

def rental_problem (num_dvds : ℕ) (original_price : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_with_tax := discounted_price * (1 + tax_rate)
  let cost_per_dvd := total_with_tax / num_dvds
  ∃ (rounded_cost : ℚ), 
    rounded_cost = (cost_per_dvd * 100).floor / 100 ∧ 
    rounded_cost = 116 / 100

theorem dvd_rental_cost : 
  rental_problem 4 (480 / 100) (10 / 100) (7 / 100) :=
sorry

end NUMINAMATH_CALUDE_dvd_rental_cost_l410_41023


namespace NUMINAMATH_CALUDE_cosine_sum_eleven_l410_41024

theorem cosine_sum_eleven : 
  Real.cos (π / 11) - Real.cos (2 * π / 11) + Real.cos (3 * π / 11) - 
  Real.cos (4 * π / 11) + Real.cos (5 * π / 11) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_eleven_l410_41024


namespace NUMINAMATH_CALUDE_combination_equality_l410_41045

theorem combination_equality (x : ℕ) : 
  (Nat.choose 18 x = Nat.choose 18 (3*x - 6)) → (x = 3 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_combination_equality_l410_41045


namespace NUMINAMATH_CALUDE_beth_cookie_price_l410_41022

/-- Represents a cookie baker --/
structure Baker where
  name : String
  cookieShape : String
  cookieCount : ℕ
  cookieArea : ℝ
  cookiePrice : ℝ

/-- Given conditions of the problem --/
def alexBaker : Baker := {
  name := "Alex"
  cookieShape := "rectangle"
  cookieCount := 10
  cookieArea := 20
  cookiePrice := 0.50
}

def bethBaker : Baker := {
  name := "Beth"
  cookieShape := "circle"
  cookieCount := 16
  cookieArea := 12.5
  cookiePrice := 0  -- To be calculated
}

/-- The total dough used by each baker --/
def totalDough (b : Baker) : ℝ := b.cookieCount * b.cookieArea

/-- The total earnings of a baker --/
def totalEarnings (b : Baker) : ℝ := b.cookieCount * b.cookiePrice * 100  -- in cents

/-- The main theorem to prove --/
theorem beth_cookie_price :
  totalDough alexBaker = totalDough bethBaker →
  totalEarnings alexBaker = bethBaker.cookieCount * 31.25 := by
  sorry


end NUMINAMATH_CALUDE_beth_cookie_price_l410_41022


namespace NUMINAMATH_CALUDE_triangle_side_length_l410_41025

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, b = 3, and angle C is twice angle A, then the length of side c is √10. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a = 2 ∧ 
  b = 3 ∧ 
  C = 2 * A ∧  -- Angle C is twice angle A
  a / Real.sin A = b / Real.sin B ∧  -- Sine theorem
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C  -- Cosine theorem
  → c = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l410_41025


namespace NUMINAMATH_CALUDE_basketball_team_sales_l410_41012

/-- The number of cupcakes sold by the basketball team -/
def num_cupcakes : ℕ := 50

/-- The price of each cupcake in dollars -/
def cupcake_price : ℚ := 2

/-- The number of cookies sold -/
def num_cookies : ℕ := 40

/-- The price of each cookie in dollars -/
def cookie_price : ℚ := 1/2

/-- The number of basketballs bought -/
def num_basketballs : ℕ := 2

/-- The price of each basketball in dollars -/
def basketball_price : ℚ := 40

/-- The number of energy drinks bought -/
def num_energy_drinks : ℕ := 20

/-- The price of each energy drink in dollars -/
def energy_drink_price : ℚ := 2

theorem basketball_team_sales :
  (num_cupcakes * cupcake_price + num_cookies * cookie_price : ℚ) =
  (num_basketballs * basketball_price + num_energy_drinks * energy_drink_price : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_sales_l410_41012


namespace NUMINAMATH_CALUDE_four_distinct_solutions_range_l410_41062

-- Define the equation
def f (x m : ℝ) : ℝ := x^2 - 4 * |x| + 5 - m

-- State the theorem
theorem four_distinct_solutions_range (m : ℝ) :
  (∃ a b c d : ℝ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a m = 0 ∧ f b m = 0 ∧ f c m = 0 ∧ f d m = 0) →
  m ∈ Set.Ioo 1 5 :=
by sorry

end NUMINAMATH_CALUDE_four_distinct_solutions_range_l410_41062


namespace NUMINAMATH_CALUDE_walk_time_to_school_l410_41078

/-- Represents Maria's travel to school -/
structure SchoolTravel where
  walkSpeed : ℝ
  skateSpeed : ℝ
  distance : ℝ

/-- The conditions of Maria's travel -/
def travelConditions (t : SchoolTravel) : Prop :=
  t.distance = 25 * t.walkSpeed + 13 * t.skateSpeed ∧
  t.distance = 11 * t.walkSpeed + 20 * t.skateSpeed

/-- The theorem to prove -/
theorem walk_time_to_school (t : SchoolTravel) 
  (h : travelConditions t) : t.distance / t.walkSpeed = 51 := by
  sorry

end NUMINAMATH_CALUDE_walk_time_to_school_l410_41078


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l410_41027

def p (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l410_41027


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l410_41063

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l410_41063


namespace NUMINAMATH_CALUDE_average_equation_solution_l410_41093

theorem average_equation_solution (x : ℝ) : 
  (1/3 : ℝ) * ((2*x + 12) + (12*x + 4) + (4*x + 14)) = 8*x - 14 → x = 12 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l410_41093


namespace NUMINAMATH_CALUDE_simplify_expression_l410_41038

theorem simplify_expression (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l410_41038


namespace NUMINAMATH_CALUDE_eggs_donated_to_charity_l410_41049

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs in a week --/
def collection_days : ℕ := 2

/-- Represents the number of dozen eggs Mortdecai collects per collection day --/
def eggs_collected_per_day : ℕ := 8

/-- Represents the number of dozen eggs Mortdecai delivers to the market --/
def eggs_to_market : ℕ := 3

/-- Represents the number of dozen eggs Mortdecai delivers to the mall --/
def eggs_to_mall : ℕ := 5

/-- Represents the number of dozen eggs Mortdecai uses for pie --/
def eggs_for_pie : ℕ := 4

/-- Theorem stating the number of eggs Mortdecai donates to charity --/
theorem eggs_donated_to_charity : 
  (collection_days * eggs_collected_per_day - (eggs_to_market + eggs_to_mall + eggs_for_pie)) * dozen = 48 := by
  sorry

end NUMINAMATH_CALUDE_eggs_donated_to_charity_l410_41049


namespace NUMINAMATH_CALUDE_sector_angle_l410_41016

theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) :
  r = 1 →
  l = 2 →
  l = α * r →
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_angle_l410_41016


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l410_41018

/-- Given a person's walking speeds and additional distance covered at higher speed,
    prove the actual distance traveled. -/
theorem actual_distance_traveled
  (original_speed : ℝ)
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (h1 : original_speed = 10)
  (h2 : faster_speed = 15)
  (h3 : additional_distance = 20)
  (h4 : faster_speed * (additional_distance / (faster_speed - original_speed)) =
        original_speed * (additional_distance / (faster_speed - original_speed)) + additional_distance) :
  original_speed * (additional_distance / (faster_speed - original_speed)) = 40 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l410_41018


namespace NUMINAMATH_CALUDE_debby_candy_count_debby_candy_count_proof_l410_41074

theorem debby_candy_count : ℕ → Prop :=
  fun d : ℕ =>
    (∃ (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ),
      sister_candy = 42 ∧
      eaten_candy = 35 ∧
      remaining_candy = 39 ∧
      d + sister_candy - eaten_candy = remaining_candy) →
    d = 32

-- Proof
theorem debby_candy_count_proof : debby_candy_count 32 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_count_debby_candy_count_proof_l410_41074


namespace NUMINAMATH_CALUDE_circle_equation_proof_l410_41013

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The equation (x - 3)² + (y + 1)² = 25 represents the circle
    with center (3, -1) passing through the point (7, -4) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : Point := ⟨3, -1⟩
  let point : Point := ⟨7, -4⟩
  (x - center.x)^2 + (y - center.y)^2 = squaredDistance center point := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l410_41013


namespace NUMINAMATH_CALUDE_crossroads_four_roads_routes_l410_41050

/-- Represents a crossroads with a given number of roads -/
structure Crossroads :=
  (num_roads : ℕ)

/-- Calculates the number of possible driving routes at a crossroads -/
def driving_routes (c : Crossroads) : ℕ :=
  c.num_roads * (c.num_roads - 1)

/-- Theorem: At a crossroads with 4 roads, where vehicles are not allowed to turn back,
    the total number of possible driving routes is 12 -/
theorem crossroads_four_roads_routes :
  ∃ (c : Crossroads), c.num_roads = 4 ∧ driving_routes c = 12 :=
sorry

end NUMINAMATH_CALUDE_crossroads_four_roads_routes_l410_41050


namespace NUMINAMATH_CALUDE_same_grade_percentage_is_50_l410_41090

/-- Represents the number of students who got the same grade on both tests for each grade -/
structure GradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- Calculates the percentage of students who got the same grade on both tests -/
def sameGradePercentage (totalStudents : ℕ) (gradeCount : GradeCount) : ℚ :=
  (gradeCount.a + gradeCount.b + gradeCount.c + gradeCount.d : ℚ) / totalStudents * 100

/-- The main theorem stating that 50% of students received the same grade on both tests -/
theorem same_grade_percentage_is_50 :
  let totalStudents : ℕ := 40
  let gradeCount : GradeCount := { a := 3, b := 6, c := 7, d := 4 }
  sameGradePercentage totalStudents gradeCount = 50 := by
  sorry


end NUMINAMATH_CALUDE_same_grade_percentage_is_50_l410_41090


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l410_41002

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 ≥ 15/16 ∧ 
  (a^2 + b^2 + c^2 + a^2*b^2 + a^2*c^2 + c^2*b^2 = 15/16 ↔ a = 1/2 ∧ b = 1/2 ∧ c = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l410_41002


namespace NUMINAMATH_CALUDE_vacation_cost_l410_41053

/-- 
If dividing a total cost among 5 people results in a per-person cost that is $120 more than 
dividing the same total cost among 8 people, then the total cost is $1600.
-/
theorem vacation_cost (total_cost : ℝ) : 
  (total_cost / 5 - total_cost / 8 = 120) → total_cost = 1600 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l410_41053


namespace NUMINAMATH_CALUDE_a_squared_greater_than_b_squared_l410_41041

theorem a_squared_greater_than_b_squared (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a = Real.log (1 + b) - Real.log (1 - b)) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_squared_greater_than_b_squared_l410_41041


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l410_41030

/-- Given points A and B, and a point C on the extension of AB such that BC = 2/3 * AB,
    prove that the coordinates of C are (53/3, 17/3). -/
theorem extension_point_coordinates (A B C : ℝ × ℝ) : 
  A = (1, -1) →
  B = (11, 3) →
  C - B = 2/3 • (B - A) →
  C = (53/3, 17/3) := by
sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l410_41030


namespace NUMINAMATH_CALUDE_all_triangles_present_l410_41076

/-- Represents a permissible triangle with angles (i/p)180°, (j/p)180°, (k/p)180° --/
structure PermissibleTriangle (p : ℕ) where
  i : ℕ
  j : ℕ
  k : ℕ
  sum_eq_p : i + j + k = p

/-- The set of all permissible triangles for a given prime p --/
def AllPermissibleTriangles (p : ℕ) : Set (PermissibleTriangle p) :=
  {t : PermissibleTriangle p | True}

/-- The set of triangles obtained after the division process stops --/
def FinalTriangleSet (p : ℕ) : Set (PermissibleTriangle p) :=
  sorry

/-- Theorem stating that the final set of triangles includes all permissible triangles --/
theorem all_triangles_present (p : ℕ) (h : Prime p) :
  FinalTriangleSet p = AllPermissibleTriangles p :=
sorry

end NUMINAMATH_CALUDE_all_triangles_present_l410_41076


namespace NUMINAMATH_CALUDE_square_of_product_pow_two_l410_41085

theorem square_of_product_pow_two (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_pow_two_l410_41085


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l410_41079

theorem incorrect_height_calculation (n : ℕ) (initial_avg actual_avg actual_height : ℝ) :
  n = 20 ∧
  initial_avg = 175 ∧
  actual_avg = 173 ∧
  actual_height = 111 →
  ∃ incorrect_height : ℝ,
    incorrect_height = n * initial_avg - (n - 1) * actual_avg - actual_height :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l410_41079


namespace NUMINAMATH_CALUDE_polynomial_remainder_l410_41067

theorem polynomial_remainder (x : ℝ) : 
  let p : ℝ → ℝ := λ x => x^5 - 2*x^3 + 4*x + 5
  p 2 = 29 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l410_41067


namespace NUMINAMATH_CALUDE_shark_sighting_relationship_l410_41031

/-- The relationship between shark sightings in Cape May and Daytona Beach --/
theorem shark_sighting_relationship (total_sightings cape_may_sightings : ℕ) 
  (h1 : total_sightings = 40)
  (h2 : cape_may_sightings = 24)
  (h3 : ∃ R : ℕ, cape_may_sightings = R - 8) :
  ∃ R : ℕ, R = 32 ∧ cape_may_sightings = R - 8 := by
  sorry

end NUMINAMATH_CALUDE_shark_sighting_relationship_l410_41031


namespace NUMINAMATH_CALUDE_congruent_sufficient_not_necessary_for_equal_area_l410_41070

-- Define a triangle type
structure Triangle where
  -- You might define a triangle using its vertices or side lengths
  -- For simplicity, we'll just assume such a type exists
  mk :: (area : ℝ)

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop :=
  -- In reality, this would involve comparing all sides and angles
  -- For our purposes, we'll leave it as an abstract property
  sorry

-- Theorem statement
theorem congruent_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_congruent_sufficient_not_necessary_for_equal_area_l410_41070


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l410_41057

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l410_41057


namespace NUMINAMATH_CALUDE_lampshade_container_volume_l410_41006

/-- The volume of the smallest cylindrical container that can fit a conical lampshade -/
theorem lampshade_container_volume
  (h : ℝ) -- height of the lampshade
  (d : ℝ) -- diameter of the lampshade base
  (h_pos : h > 0)
  (d_pos : d > 0)
  (h_val : h = 15)
  (d_val : d = 8) :
  let r := d / 2 -- radius of the container
  let v := π * r^2 * h -- volume of the container
  v = 240 * π :=
sorry

end NUMINAMATH_CALUDE_lampshade_container_volume_l410_41006


namespace NUMINAMATH_CALUDE_cat_age_proof_l410_41087

theorem cat_age_proof (cat_age rabbit_age dog_age : ℕ) : 
  rabbit_age = cat_age / 2 →
  dog_age = 3 * rabbit_age →
  dog_age = 12 →
  cat_age = 8 := by
sorry

end NUMINAMATH_CALUDE_cat_age_proof_l410_41087


namespace NUMINAMATH_CALUDE_sin_double_angle_for_line_l410_41019

/-- Given a line with equation 2x-4y+5=0 and angle of inclination α, prove that sin2α = 4/5 -/
theorem sin_double_angle_for_line (x y : ℝ) (α : ℝ) 
  (h : 2 * x - 4 * y + 5 = 0) 
  (h_incline : α = Real.arctan (1 / 2)) : 
  Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_for_line_l410_41019


namespace NUMINAMATH_CALUDE_ac_length_l410_41071

/-- Given a line segment AB of length 4 with a point C on it, prove that if AC is the mean
    proportional between AB and BC, then the length of AC is 2√5 - 2. -/
theorem ac_length (AB : ℝ) (C : ℝ) (hAB : AB = 4) (hC : 0 ≤ C ∧ C ≤ AB) 
  (hMean : C^2 = AB * (AB - C)) : C = 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l410_41071


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l410_41088

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_Pa : P a b c a = 2021 * b * c)
  (h_Pb : P a b c b = 2021 * c * a)
  (h_Pc : P a b c c = 2021 * a * b) :
  a + 2021 * b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l410_41088


namespace NUMINAMATH_CALUDE_jenna_reading_goal_l410_41035

/-- Calculates the number of pages Jenna needs to read per day to meet her reading goal --/
theorem jenna_reading_goal (total_pages : ℕ) (total_days : ℕ) (busy_days : ℕ) (special_day_pages : ℕ) :
  total_pages = 600 →
  total_days = 30 →
  busy_days = 4 →
  special_day_pages = 100 →
  (total_pages - special_day_pages) / (total_days - busy_days - 1) = 20 := by
  sorry

#check jenna_reading_goal

end NUMINAMATH_CALUDE_jenna_reading_goal_l410_41035


namespace NUMINAMATH_CALUDE_face_value_is_75_l410_41033

/-- Given banker's discount (BD) and true discount (TD), calculate the face value (FV) -/
def calculate_face_value (BD TD : ℚ) : ℚ :=
  (TD^2) / (BD - TD)

/-- Theorem stating that given BD = 18 and TD = 15, the face value is 75 -/
theorem face_value_is_75 :
  calculate_face_value 18 15 = 75 := by
  sorry

#eval calculate_face_value 18 15

end NUMINAMATH_CALUDE_face_value_is_75_l410_41033


namespace NUMINAMATH_CALUDE_fraction_problem_l410_41004

theorem fraction_problem (N : ℝ) (f : ℝ) :
  N = 24 →
  N * f - 10 = 0.25 * N →
  f = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l410_41004


namespace NUMINAMATH_CALUDE_tangent_line_theorem_l410_41098

noncomputable def curve (x : ℝ) : ℝ := 2 * x^2 - x^3

def point_P : ℝ × ℝ := (0, -4)

def is_tangent_point (a : ℝ) : Prop :=
  ∃ (m : ℝ), curve a = 2 * a^2 - a^3 ∧
             m * a + (2 * a^2 - a^3) = -4 ∧
             m = 4 * a - 3 * a^2

theorem tangent_line_theorem :
  ∃ (a : ℝ), is_tangent_point a ∧ a = -1 ∧
  ∃ (m : ℝ), m = -7 ∧ 
  (∀ (x y : ℝ), y = m * x - 4 ↔ 7 * x + y + 4 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_theorem_l410_41098


namespace NUMINAMATH_CALUDE_sallys_nickels_l410_41010

theorem sallys_nickels (initial_nickels dad_nickels total_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_nickels = 9)
  (h3 : total_nickels = 18) :
  total_nickels - (initial_nickels + dad_nickels) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_l410_41010


namespace NUMINAMATH_CALUDE_resultant_profit_is_four_percent_l410_41081

/-- Calculates the resultant profit percentage when an item is sold twice -/
def resultantProfitPercentage (firstProfit : Real) (secondLoss : Real) : Real :=
  let firstSalePrice := 1 + firstProfit
  let secondSalePrice := firstSalePrice * (1 - secondLoss)
  (secondSalePrice - 1) * 100

/-- Theorem: The resultant profit percentage when an item is sold with 30% profit
    and then resold with 20% loss is 4% -/
theorem resultant_profit_is_four_percent :
  resultantProfitPercentage 0.3 0.2 = 4 := by sorry

end NUMINAMATH_CALUDE_resultant_profit_is_four_percent_l410_41081


namespace NUMINAMATH_CALUDE_cow_milk_production_l410_41083

/-- Given two groups of cows with different efficiencies, calculate the milk production of the second group based on the first group's rate. -/
theorem cow_milk_production
  (a b c d e f g : ℝ)
  (h₁ : a > 0)
  (h₂ : c > 0)
  (h₃ : f > 0) :
  let rate := b / (a * c * f)
  let second_group_production := d * rate * g * e
  second_group_production = b * d * e * g / (a * c * f) :=
by sorry

end NUMINAMATH_CALUDE_cow_milk_production_l410_41083


namespace NUMINAMATH_CALUDE_cage_cost_calculation_l410_41084

/-- The cost of the cage given the payment and change -/
def cage_cost (payment : ℚ) (change : ℚ) : ℚ :=
  payment - change

theorem cage_cost_calculation (payment : ℚ) (change : ℚ) 
  (h1 : payment = 20) 
  (h2 : change = 0.26) : 
  cage_cost payment change = 19.74 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_calculation_l410_41084


namespace NUMINAMATH_CALUDE_vertical_angles_equal_l410_41056

-- Define a point in a 2D plane
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in a 2D plane using two points
structure Line2D where
  p1 : Point2D
  p2 : Point2D

-- Define an angle
structure Angle where
  vertex : Point2D
  ray1 : Point2D
  ray2 : Point2D

-- Define the intersection of two lines
def intersection (l1 l2 : Line2D) : Point2D :=
  sorry

-- Define vertical angles
def verticalAngles (l1 l2 : Line2D) : (Angle × Angle) :=
  sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line2D) :
  let (a1, a2) := verticalAngles l1 l2
  a1 = a2 :=
sorry

end NUMINAMATH_CALUDE_vertical_angles_equal_l410_41056


namespace NUMINAMATH_CALUDE_alyssa_games_this_year_l410_41001

/-- The number of soccer games Alyssa attended over three years -/
def total_games : ℕ := 39

/-- The number of games Alyssa attended last year -/
def last_year_games : ℕ := 13

/-- The number of games Alyssa plans to attend next year -/
def next_year_games : ℕ := 15

/-- The number of games Alyssa attended this year -/
def this_year_games : ℕ := total_games - last_year_games - next_year_games

theorem alyssa_games_this_year :
  this_year_games = 11 := by sorry

end NUMINAMATH_CALUDE_alyssa_games_this_year_l410_41001


namespace NUMINAMATH_CALUDE_power_of_four_l410_41011

theorem power_of_four (n : ℕ) : 
  (2 * n + 7 + 2 = 31) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l410_41011


namespace NUMINAMATH_CALUDE_all_sums_representable_l410_41003

/-- Available coin denominations -/
def Coins : Set ℕ := {1, 2, 5, 10}

/-- A function that checks if a sum can be represented by both even and odd number of coins -/
def canRepresentEvenAndOdd (S : ℕ) : Prop :=
  ∃ (even_coins odd_coins : List ℕ),
    (∀ c ∈ even_coins, c ∈ Coins) ∧
    (∀ c ∈ odd_coins, c ∈ Coins) ∧
    (even_coins.sum = S) ∧
    (odd_coins.sum = S) ∧
    (even_coins.length % 2 = 0) ∧
    (odd_coins.length % 2 = 1)

/-- Theorem stating that any sum greater than 1 can be represented by both even and odd number of coins -/
theorem all_sums_representable (S : ℕ) (h : S > 1) :
  canRepresentEvenAndOdd S := by
  sorry

end NUMINAMATH_CALUDE_all_sums_representable_l410_41003


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l410_41072

theorem infinite_solutions_condition (b : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 16)) ↔ b = -12 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l410_41072


namespace NUMINAMATH_CALUDE_percentage_seven_plus_years_l410_41028

/-- Represents the number of employees in each employment duration range --/
structure EmployeeDistribution :=
  (less_than_1_year : ℕ)
  (one_to_two_years : ℕ)
  (two_to_three_years : ℕ)
  (three_to_four_years : ℕ)
  (four_to_five_years : ℕ)
  (five_to_six_years : ℕ)
  (six_to_seven_years : ℕ)
  (seven_to_eight_years : ℕ)
  (eight_to_nine_years : ℕ)
  (nine_to_ten_years : ℕ)
  (ten_plus_years : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_two_years + d.two_to_three_years + d.three_to_four_years +
  d.four_to_five_years + d.five_to_six_years + d.six_to_seven_years + d.seven_to_eight_years +
  d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Calculates the number of employees employed for 7 years or more --/
def employees_seven_plus_years (d : EmployeeDistribution) : ℕ :=
  d.seven_to_eight_years + d.eight_to_nine_years + d.nine_to_ten_years + d.ten_plus_years

/-- Theorem stating that the percentage of employees employed for 7 years or more is 21.43% --/
theorem percentage_seven_plus_years (d : EmployeeDistribution) 
  (h : d = {
    less_than_1_year := 4,
    one_to_two_years := 6,
    two_to_three_years := 5,
    three_to_four_years := 2,
    four_to_five_years := 3,
    five_to_six_years := 1,
    six_to_seven_years := 1,
    seven_to_eight_years := 2,
    eight_to_nine_years := 2,
    nine_to_ten_years := 1,
    ten_plus_years := 1
  }) :
  (employees_seven_plus_years d : ℚ) / (total_employees d : ℚ) * 100 = 21.43 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seven_plus_years_l410_41028


namespace NUMINAMATH_CALUDE_inequalities_proof_l410_41068

theorem inequalities_proof (a b : ℝ) (h1 : b > a) (h2 : a * b > 0) :
  (1 / a > 1 / b) ∧ (a + b < 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l410_41068


namespace NUMINAMATH_CALUDE_z2_magnitude_range_l410_41029

theorem z2_magnitude_range (z₁ z₂ : ℂ) 
  (h1 : (z₁ - Complex.I) * (z₂ + Complex.I) = 1)
  (h2 : Complex.abs z₁ = Real.sqrt 2) :
  2 - Real.sqrt 2 ≤ Complex.abs z₂ ∧ Complex.abs z₂ ≤ 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_z2_magnitude_range_l410_41029


namespace NUMINAMATH_CALUDE_raw_materials_cost_l410_41077

def total_amount : ℝ := 93750
def machinery_cost : ℝ := 40000
def cash_percentage : ℝ := 0.20

theorem raw_materials_cost (raw_materials : ℝ) : raw_materials = 35000 :=
  by
    have cash : ℝ := total_amount * cash_percentage
    have total_equation : raw_materials + machinery_cost + cash = total_amount := by sorry
    sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l410_41077


namespace NUMINAMATH_CALUDE_overhead_cost_calculation_l410_41015

/-- The overhead cost for Steve's circus production -/
def overhead_cost : ℕ := sorry

/-- The production cost per performance -/
def production_cost_per_performance : ℕ := 7000

/-- The revenue from a sold-out performance -/
def revenue_per_performance : ℕ := 16000

/-- The number of sold-out performances needed to break even -/
def break_even_performances : ℕ := 9

/-- Theorem stating that the overhead cost is $81,000 -/
theorem overhead_cost_calculation :
  overhead_cost = 81000 :=
by
  sorry

end NUMINAMATH_CALUDE_overhead_cost_calculation_l410_41015


namespace NUMINAMATH_CALUDE_inequality_proof_l410_41017

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l410_41017


namespace NUMINAMATH_CALUDE_soccer_team_games_l410_41054

/-- Calculates the total number of games played by a soccer team given their win:loss:tie ratio and the number of games lost. -/
def total_games (win_ratio : ℕ) (loss_ratio : ℕ) (tie_ratio : ℕ) (games_lost : ℕ) : ℕ :=
  let games_per_part := games_lost / loss_ratio
  let total_parts := win_ratio + loss_ratio + tie_ratio
  total_parts * games_per_part

/-- Theorem stating that for a soccer team with a win:loss:tie ratio of 4:3:1 and 9 losses, the total number of games played is 24. -/
theorem soccer_team_games : 
  total_games 4 3 1 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_games_l410_41054


namespace NUMINAMATH_CALUDE_radical_simplification_l410_41014

theorem radical_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (75 * x) * Real.sqrt (2 * x) * Real.sqrt (14 * x) = 10 * x * Real.sqrt (21 * x) := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l410_41014


namespace NUMINAMATH_CALUDE_kristen_turtles_l410_41096

theorem kristen_turtles (trey kris kristen : ℕ) : 
  trey = 7 * kris →
  kris = kristen / 4 →
  trey = kristen + 9 →
  kristen = 12 := by
sorry

end NUMINAMATH_CALUDE_kristen_turtles_l410_41096


namespace NUMINAMATH_CALUDE_suitable_pairs_solution_l410_41059

def suitable_pair (a b : ℕ+) : Prop := (a + b) ∣ (a * b)

def pairs : List (ℕ+ × ℕ+) := [
  (3, 6), (4, 12), (5, 20), (6, 30), (7, 42), (8, 56),
  (9, 72), (10, 90), (11, 110), (12, 132), (13, 156), (14, 168)
]

theorem suitable_pairs_solution :
  (∀ (p : ℕ+ × ℕ+), p ∈ pairs → suitable_pair p.1 p.2) ∧
  (pairs.length = 12) ∧
  (∀ (n : ℕ+), (n ∈ pairs.map Prod.fst ∨ n ∈ pairs.map Prod.snd) →
    (pairs.map Prod.fst ++ pairs.map Prod.snd).count n = 1) ∧
  (∀ (p : ℕ+ × ℕ+), p ∉ pairs → p.1 ≤ 168 ∧ p.2 ≤ 168 → ¬suitable_pair p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_suitable_pairs_solution_l410_41059


namespace NUMINAMATH_CALUDE_compound_interest_problem_l410_41020

/-- Given a principal amount P, where the simple interest on P for 2 years at 10% per annum is $660,
    prove that the compound interest on P for 2 years at the same rate is $693. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.1 * 2 = 660 → P * (1 + 0.1)^2 - P = 693 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l410_41020


namespace NUMINAMATH_CALUDE_omega_sum_l410_41086

theorem omega_sum (ω : ℂ) (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^15 + ω^20 + ω^25 + ω^30 + ω^35 + ω^40 + ω^45 + ω^50 = 8 := by
  sorry

end NUMINAMATH_CALUDE_omega_sum_l410_41086


namespace NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l410_41099

theorem sin_cos_roots_quadratic (θ : Real) (a : Real) : 
  (4 * Real.sin θ ^ 2 + 2 * a * Real.sin θ + a = 0) ∧ 
  (4 * Real.cos θ ^ 2 + 2 * a * Real.cos θ + a = 0) →
  a = 1 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_roots_quadratic_l410_41099


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_condition_l410_41048

/-- Given a hyperbola mx^2 + y^2 = 1 where m < -1, if its eccentricity is exactly
    the geometric mean of the lengths of the real and imaginary axes, then m = -7 - 4√3 -/
theorem hyperbola_eccentricity_condition (m : ℝ) : 
  m < -1 →
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →
  (∃ e a b : ℝ, e^2 = 4 * a * b ∧ a = 1 ∧ b^2 = -1/m) →
  m = -7 - 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_condition_l410_41048


namespace NUMINAMATH_CALUDE_M_equals_N_l410_41069

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

-- Theorem statement
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l410_41069


namespace NUMINAMATH_CALUDE_extra_tip_amount_l410_41073

/-- The amount of a bill in dollars -/
def bill_amount : ℚ := 26

/-- The percentage of a bad tip -/
def bad_tip_percentage : ℚ := 5 / 100

/-- The percentage of a good tip -/
def good_tip_percentage : ℚ := 20 / 100

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tip amount in cents -/
def tip_amount (bill : ℚ) (percentage : ℚ) : ℚ :=
  dollars_to_cents (bill * percentage)

theorem extra_tip_amount :
  tip_amount bill_amount good_tip_percentage - tip_amount bill_amount bad_tip_percentage = 390 := by
  sorry

end NUMINAMATH_CALUDE_extra_tip_amount_l410_41073


namespace NUMINAMATH_CALUDE_total_ladybugs_l410_41064

theorem total_ladybugs (num_leaves : ℕ) (ladybugs_per_leaf : ℕ) 
  (h1 : num_leaves = 84) 
  (h2 : ladybugs_per_leaf = 139) : 
  num_leaves * ladybugs_per_leaf = 11676 := by
  sorry

end NUMINAMATH_CALUDE_total_ladybugs_l410_41064


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l410_41046

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (-1 + 3*x) * (-3*x - 1) = 1 - 9*x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : 
  (x + 1)^2 - (1 - 3*x) * (1 + 3*x) = 10*x^2 + 2*x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l410_41046


namespace NUMINAMATH_CALUDE_number_of_true_statements_number_of_true_statements_is_correct_l410_41008

/-- A quadratic equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

/-- The number of true propositions among the statement and its variants -/
theorem number_of_true_statements : ℕ :=
  let s1 := ∀ m : ℝ, m > 0 → has_real_roots m
  let s2 := ∀ m : ℝ, has_real_roots m → m > 0
  let s3 := ∀ m : ℝ, m ≤ 0 → ¬has_real_roots m
  let s4 := ∀ m : ℝ, ¬has_real_roots m → m ≤ 0
  2

theorem number_of_true_statements_is_correct :
  number_of_true_statements = 2 :=
by sorry

end NUMINAMATH_CALUDE_number_of_true_statements_number_of_true_statements_is_correct_l410_41008


namespace NUMINAMATH_CALUDE_rearrange_segments_sum_l410_41055

theorem rearrange_segments_sum (a b : ℕ) : ∃ (f g : Fin 1961 → Fin 1961),
  ∀ (i : Fin 1961), ∃ (k : ℕ),
    (a + (f i : ℕ)) + (b + (g i : ℕ)) = k + i ∧ 
    k + 1961 > k + i ∧
    k + i ≥ k :=
sorry

end NUMINAMATH_CALUDE_rearrange_segments_sum_l410_41055


namespace NUMINAMATH_CALUDE_vertex_D_coordinates_l410_41060

/-- A parallelogram with vertices A, B, C, and D in 2D space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The given parallelogram ABCD with specified coordinates for A, B, and C. -/
def givenParallelogram : Parallelogram where
  A := (0, 0)
  B := (1, 2)
  C := (3, 1)
  D := (2, -1)  -- We include D here, but will prove it's correct

/-- Theorem stating that the coordinates of vertex D in the given parallelogram are (2, -1). -/
theorem vertex_D_coordinates (p : Parallelogram) (h : p = givenParallelogram) :
  p.D = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_vertex_D_coordinates_l410_41060


namespace NUMINAMATH_CALUDE_order_of_a_ab2_ab_l410_41000

theorem order_of_a_ab2_ab (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a < a * b^2 ∧ a * b^2 < a * b := by sorry

end NUMINAMATH_CALUDE_order_of_a_ab2_ab_l410_41000


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l410_41052

theorem arithmetic_calculation : 1984 + 180 / 60 - 284 = 1703 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l410_41052


namespace NUMINAMATH_CALUDE_elberta_amount_l410_41044

/-- The amount of money Granny Smith has -/
def granny_smith : ℚ := 75

/-- The amount of money Anjou has -/
def anjou : ℚ := granny_smith / 4

/-- The amount of money Elberta has -/
def elberta : ℚ := anjou + 3

/-- Theorem stating that Elberta has $21.75 -/
theorem elberta_amount : elberta = 21.75 := by
  sorry

end NUMINAMATH_CALUDE_elberta_amount_l410_41044


namespace NUMINAMATH_CALUDE_cubic_roots_reciprocal_sum_squares_l410_41047

theorem cubic_roots_reciprocal_sum_squares (a b c d r s t : ℝ) : 
  a ≠ 0 → d ≠ 0 → 
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = r ∨ x = s ∨ x = t) →
  1/r^2 + 1/s^2 + 1/t^2 = (b^2 - 2*a*c) / d^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_reciprocal_sum_squares_l410_41047


namespace NUMINAMATH_CALUDE_fruit_to_grain_value_fruit_worth_in_grains_l410_41091

-- Define the exchange rates
def fruit_to_vegetable : ℚ := 3 / 4
def vegetable_to_grain : ℚ := 5

-- Theorem statement
theorem fruit_to_grain_value :
  fruit_to_vegetable * vegetable_to_grain = 15 / 4 :=
by sorry

-- Corollary to express the result as a mixed number
theorem fruit_worth_in_grains :
  fruit_to_vegetable * vegetable_to_grain = 3 + 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fruit_to_grain_value_fruit_worth_in_grains_l410_41091


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l410_41021

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) / Real.log 10}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l410_41021


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l410_41034

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The total number of pairs of shoes Helga tried on at the first three stores -/
def first_three_stores : ℕ := first_store + second_store + third_store

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * first_three_stores

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_three_stores + fourth_store

theorem helga_shoe_shopping : total_shoes = 48 := by
  sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l410_41034


namespace NUMINAMATH_CALUDE_optimal_choice_is_104_l410_41066

/-- Counts the number of distinct rectangles with integer sides for a given perimeter --/
def countRectangles (perimeter : ℕ) : ℕ :=
  if perimeter % 2 = 0 then
    (perimeter / 4 : ℕ)
  else
    0

/-- Checks if a number is a valid choice in the game --/
def isValidChoice (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 105

/-- Theorem stating that 104 is the optimal choice for Grisha --/
theorem optimal_choice_is_104 :
  ∀ n, isValidChoice n → countRectangles 104 ≥ countRectangles n :=
by sorry

end NUMINAMATH_CALUDE_optimal_choice_is_104_l410_41066


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l410_41032

theorem trig_expression_simplification :
  (Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (30 * π / 180) =
  2 * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.sin (50 * π / 180) * Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) /
  (Real.sqrt 3 * Real.cos (40 * π / 180) * Real.cos (50 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l410_41032


namespace NUMINAMATH_CALUDE_range_of_z_l410_41089

theorem range_of_z (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  let z := 2 * x - 3 * y
  ∃ (a b : ℝ), a = -5 ∧ b = 4 ∧ ∀ w, w ∈ Set.Icc a b ↔ ∃ (x' y' : ℝ), 
    -1 ≤ x' ∧ x' ≤ 2 ∧ 0 ≤ y' ∧ y' ≤ 1 ∧ w = 2 * x' - 3 * y' :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l410_41089


namespace NUMINAMATH_CALUDE_root_equation_result_l410_41009

theorem root_equation_result (a b m p : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ∃ r, ((a + 1/b)^2 - p*(a + 1/b) + r = 0) ∧ 
       ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
       r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_result_l410_41009
