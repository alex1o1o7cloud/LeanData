import Mathlib

namespace NUMINAMATH_CALUDE_joker_selection_ways_l2590_259063

def total_cards : ℕ := 54
def jokers : ℕ := 2
def standard_cards : ℕ := 52

def ways_to_pick_joker_first (cards : ℕ) (jokers : ℕ) : ℕ :=
  jokers * (cards - 1)

def ways_to_pick_joker_second (cards : ℕ) (standard_cards : ℕ) (jokers : ℕ) : ℕ :=
  standard_cards * jokers

theorem joker_selection_ways :
  ways_to_pick_joker_first total_cards jokers +
  ways_to_pick_joker_second total_cards standard_cards jokers = 210 :=
by sorry

end NUMINAMATH_CALUDE_joker_selection_ways_l2590_259063


namespace NUMINAMATH_CALUDE_g_composition_equals_71_l2590_259077

def g (n : ℤ) : ℤ :=
  if n < 5 then n^2 + 2*n - 1 else 2*n + 5

theorem g_composition_equals_71 : g (g (g 3)) = 71 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_equals_71_l2590_259077


namespace NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l2590_259074

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x_squared_minus_one_l2590_259074


namespace NUMINAMATH_CALUDE_parabola_equation_l2590_259009

theorem parabola_equation (x : ℝ) :
  let f := fun x => -3 * x^2 + 12 * x - 8
  let vertex := (2, 4)
  let point := (1, 1)
  (∀ h, f (vertex.1 + h) = f (vertex.1 - h)) ∧  -- Vertical axis of symmetry
  (f vertex.1 = vertex.2) ∧                     -- Passes through vertex
  (f point.1 = point.2) ∧                       -- Contains the given point
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) -- In quadratic form
  :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l2590_259009


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l2590_259095

theorem imaginary_part_of_2_plus_i_times_i (i : ℂ) : 
  Complex.im ((2 : ℂ) + i * i) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_plus_i_times_i_l2590_259095


namespace NUMINAMATH_CALUDE_cubic_equations_common_root_implies_three_real_roots_l2590_259017

/-- Given distinct nonzero real numbers a, b, c, if the equations ax³ + bx + c = 0, 
    bx³ + cx + a = 0, and cx³ + ax + b = 0 have a common root, then at least one of 
    these equations has three real roots. -/
theorem cubic_equations_common_root_implies_three_real_roots 
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_common_root : ∃ x : ℝ, a * x^3 + b * x + c = 0 ∧ 
                            b * x^3 + c * x + a = 0 ∧ 
                            c * x^3 + a * x + b = 0) :
  (∃ x y z : ℝ, a * x^3 + b * x + c = 0 ∧ 
               a * y^3 + b * y + c = 0 ∧ 
               a * z^3 + b * z + c = 0) ∨
  (∃ x y z : ℝ, b * x^3 + c * x + a = 0 ∧ 
               b * y^3 + c * y + a = 0 ∧ 
               b * z^3 + c * z + a = 0) ∨
  (∃ x y z : ℝ, c * x^3 + a * x + b = 0 ∧ 
               c * y^3 + a * y + b = 0 ∧ 
               c * z^3 + a * z + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equations_common_root_implies_three_real_roots_l2590_259017


namespace NUMINAMATH_CALUDE_no_real_solutions_l2590_259022

theorem no_real_solutions : ¬ ∃ x : ℝ, Real.sqrt ((x^2 - 2*x + 1) + 1) = -x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2590_259022


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l2590_259088

theorem product_from_lcm_gcd (a b : ℕ+) (h1 : Nat.lcm a b = 120) (h2 : Nat.gcd a b = 8) :
  a * b = 960 := by sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l2590_259088


namespace NUMINAMATH_CALUDE_external_tangency_intersection_two_points_l2590_259032

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def C₂ (x y r : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = r^2

-- Define the center and radius of C₁
def center_C₁ : ℝ × ℝ := (1, 1)
def radius_C₁ : ℝ := 1

-- Define the center of C₂
def center_C₂ : ℝ × ℝ := (4, 5)

-- Define the distance between centers
def distance_between_centers : ℝ := 5

-- Theorem for external tangency
theorem external_tangency (r : ℝ) (hr : r > 0) :
  (∀ x y, C₁ x y → C₂ x y r → (x - 1)^2 + (y - 1)^2 = 1 ∧ (x - 4)^2 + (y - 5)^2 = r^2) →
  distance_between_centers = radius_C₁ + r →
  r = 4 :=
sorry

-- Theorem for intersection at two points
theorem intersection_two_points (r : ℝ) (hr : r > 0) :
  (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ r ∧ C₂ x₂ y₂ r) →
  4 < r ∧ r < 6 :=
sorry

end NUMINAMATH_CALUDE_external_tangency_intersection_two_points_l2590_259032


namespace NUMINAMATH_CALUDE_simon_blueberries_l2590_259035

def blueberry_problem (own_bushes nearby_bushes pies_made blueberries_per_pie : ℕ) : Prop :=
  own_bushes + nearby_bushes = pies_made * blueberries_per_pie

theorem simon_blueberries : 
  ∃ (own_bushes : ℕ), 
    blueberry_problem own_bushes 200 3 100 ∧ 
    own_bushes = 100 := by sorry

end NUMINAMATH_CALUDE_simon_blueberries_l2590_259035


namespace NUMINAMATH_CALUDE_factorization_problem_multiplication_problem_l2590_259004

variable (x y : ℝ)

theorem factorization_problem : x^5 - x^3 * y^2 = x^3 * (x - y) * (x + y) := by sorry

theorem multiplication_problem : (-2 * x^3 * y^2) * (3 * x^2 * y) = -6 * x^5 * y^3 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_multiplication_problem_l2590_259004


namespace NUMINAMATH_CALUDE_zoe_app_cost_l2590_259034

/-- Calculates the total cost of an app and its associated expenses -/
def total_app_cost (initial_cost monthly_cost in_game_cost upgrade_cost months : ℕ) : ℕ :=
  initial_cost + (monthly_cost * months) + in_game_cost + upgrade_cost

/-- Theorem stating the total cost for Zoe's app usage -/
theorem zoe_app_cost : total_app_cost 5 8 10 12 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_zoe_app_cost_l2590_259034


namespace NUMINAMATH_CALUDE_digit_59_is_4_l2590_259089

/-- The decimal representation of 1/17 as a list of digits -/
def decimal_rep_1_17 : List Nat := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def cycle_length : Nat := 16

/-- The 59th digit after the decimal point in the decimal representation of 1/17 -/
def digit_59 : Nat := decimal_rep_1_17[(59 - 1) % cycle_length]

theorem digit_59_is_4 : digit_59 = 4 := by sorry

end NUMINAMATH_CALUDE_digit_59_is_4_l2590_259089


namespace NUMINAMATH_CALUDE_square_difference_l2590_259094

theorem square_difference (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  (a - b)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_l2590_259094


namespace NUMINAMATH_CALUDE_vector_sum_equality_l2590_259025

theorem vector_sum_equality (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (-3, 4) →
  (3 : ℝ) • a + (4 : ℝ) • b = (-6, 19) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_equality_l2590_259025


namespace NUMINAMATH_CALUDE_rachel_picked_apples_l2590_259061

/-- Given information about Rachel's apple picking -/
structure ApplePicking where
  num_trees : ℕ
  apples_per_tree : ℕ
  apples_left : ℕ

/-- Theorem stating the total number of apples Rachel picked -/
theorem rachel_picked_apples (ap : ApplePicking)
  (h1 : ap.num_trees = 4)
  (h2 : ap.apples_per_tree = 7)
  (h3 : ap.apples_left = 29) :
  ap.num_trees * ap.apples_per_tree = 28 := by
  sorry

#check rachel_picked_apples

end NUMINAMATH_CALUDE_rachel_picked_apples_l2590_259061


namespace NUMINAMATH_CALUDE_lending_interest_rate_l2590_259071

/-- Calculates the simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lending_interest_rate 
  (principal : ℝ)
  (b_to_c_rate : ℝ)
  (time : ℝ)
  (b_gain : ℝ)
  (h1 : principal = 3200)
  (h2 : b_to_c_rate = 0.145)
  (h3 : time = 5)
  (h4 : b_gain = 400)
  : ∃ (a_to_b_rate : ℝ), 
    simpleInterest principal a_to_b_rate time = 
    simpleInterest principal b_to_c_rate time - b_gain ∧ 
    a_to_b_rate = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_l2590_259071


namespace NUMINAMATH_CALUDE_sum_of_exponents_eight_l2590_259027

/-- Sum of geometric series from 1 to x^n -/
def geometricSum (x n : ℕ) : ℕ := (x^(n+1) - 1) / (x - 1)

/-- Sum of divisors of 2^i * 3^j * 5^k -/
def sumDivisors (i j k : ℕ) : ℕ :=
  (geometricSum 2 i) * (geometricSum 3 j) * (geometricSum 5 k)

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 1800, then i + j + k = 8 -/
theorem sum_of_exponents_eight (i j k : ℕ) :
  sumDivisors i j k = 1800 → i + j + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_eight_l2590_259027


namespace NUMINAMATH_CALUDE_car_mileage_l2590_259042

theorem car_mileage (highway_miles_per_tank : ℕ) (city_mpg : ℕ) (mpg_difference : ℕ) :
  highway_miles_per_tank = 462 →
  city_mpg = 24 →
  mpg_difference = 9 →
  (highway_miles_per_tank / (city_mpg + mpg_difference)) * city_mpg = 336 :=
by sorry

end NUMINAMATH_CALUDE_car_mileage_l2590_259042


namespace NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_1_l2590_259038

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 1 = 0 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 1

theorem discriminant_of_5x2_minus_9x_plus_1 :
  discriminant a b c = 61 := by sorry

end NUMINAMATH_CALUDE_discriminant_of_5x2_minus_9x_plus_1_l2590_259038


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2590_259031

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (9 * bowling_ball_weight = 6 * canoe_weight) →
    (4 * canoe_weight = 120) →
    bowling_ball_weight = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2590_259031


namespace NUMINAMATH_CALUDE_square_value_l2590_259051

theorem square_value : ∃ (square : ℚ), 
  16.2 * ((4 + 1/7 - square * 700) / (1 + 2/7)) = 8.1 ∧ square = 0.005 := by sorry

end NUMINAMATH_CALUDE_square_value_l2590_259051


namespace NUMINAMATH_CALUDE_perpendicular_vectors_collinear_vectors_l2590_259083

def vector_a (x : ℝ) : ℝ × ℝ := (3, x)
def vector_b : ℝ × ℝ := (-2, 2)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2 = 0 → x = 3 := by sorry

theorem collinear_vectors (x : ℝ) :
  ∃ (k : ℝ), k ≠ 0 ∧ 
  (vector_b.1 - (vector_a x).1, vector_b.2 - (vector_a x).2) = 
  k • (3 * (vector_a x).1 + 2 * vector_b.1, 3 * (vector_a x).2 + 2 * vector_b.2) 
  → x = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_collinear_vectors_l2590_259083


namespace NUMINAMATH_CALUDE_sqrt_200_simplification_l2590_259002

theorem sqrt_200_simplification : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_simplification_l2590_259002


namespace NUMINAMATH_CALUDE_volume_of_specific_prism_l2590_259075

/-- Right triangular prism ABC-A₁B₁C₁ -/
structure RightTriangularPrism where
  -- Base triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Circumscribed sphere
  sphereSurfaceArea : ℝ

/-- Volume of a right triangular prism -/
def prismVolume (p : RightTriangularPrism) : ℝ := sorry

theorem volume_of_specific_prism :
  let p : RightTriangularPrism := {
    AB := 2,
    BC := 2,
    AC := 2 * Real.sqrt 3,
    sphereSurfaceArea := 32 * Real.pi
  }
  prismVolume p = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_specific_prism_l2590_259075


namespace NUMINAMATH_CALUDE_fishing_problem_l2590_259024

theorem fishing_problem (jordan_catch : ℕ) (perry_catch : ℕ) (total_catch : ℕ) (fish_lost : ℕ) (fish_remaining : ℕ) : 
  jordan_catch = 4 →
  perry_catch = 2 * jordan_catch →
  total_catch = jordan_catch + perry_catch →
  fish_lost = total_catch / 4 →
  fish_remaining = total_catch - fish_lost →
  fish_remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l2590_259024


namespace NUMINAMATH_CALUDE_power_sum_value_l2590_259041

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 4) (h2 : a^y = 9) : a^(x+y) = 36 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l2590_259041


namespace NUMINAMATH_CALUDE_circle_fixed_point_l2590_259052

/-- A circle with center (a, b) on the parabola y^2 = 4x and tangent to x = -1 passes through (1, 0) -/
theorem circle_fixed_point (a b : ℝ) : 
  b^2 = 4*a →  -- Center (a, b) lies on the parabola y^2 = 4x
  (a + 1)^2 = (1 - a)^2 + b^2 -- Circle is tangent to x = -1
  → (1 - a)^2 + 0^2 = (a + 1)^2 -- Point (1, 0) lies on the circle
  := by sorry

end NUMINAMATH_CALUDE_circle_fixed_point_l2590_259052


namespace NUMINAMATH_CALUDE_min_value_product_l2590_259012

theorem min_value_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (x / y + y) * (y / x + x) ≥ 4 ∧
  ((x / y + y) * (y / x + x) = 4 ↔ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_product_l2590_259012


namespace NUMINAMATH_CALUDE_ellipse_m_value_l2590_259086

/-- Definition of an ellipse with semi-major axis a and semi-minor axis b -/
def is_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1

/-- The sum of distances from any point on an ellipse to its foci is constant -/
axiom ellipse_foci_distance_sum (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_ellipse a b P → (dist P F₁ + dist P F₂ = 2 * a)

/-- Theorem: For an ellipse with equation x²/m + y²/16 = 1, 
    if the distances from any point to the foci are 3 and 7, then m = 25 -/
theorem ellipse_m_value (m : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_ellipse (Real.sqrt m) 4 P →
  dist P F₁ = 3 →
  dist P F₂ = 7 →
  m = 25 := by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l2590_259086


namespace NUMINAMATH_CALUDE_blood_drops_per_liter_l2590_259039

/-- The number of drops of blood sucked by one mosquito in a single feeding. -/
def drops_per_mosquito : ℕ := 20

/-- The number of liters of blood loss that is fatal. -/
def fatal_blood_loss : ℕ := 3

/-- The number of mosquitoes that would cause a fatal blood loss if they all fed. -/
def fatal_mosquito_count : ℕ := 750

/-- The number of drops of blood in one liter. -/
def drops_per_liter : ℕ := 5000

theorem blood_drops_per_liter :
  drops_per_liter = (drops_per_mosquito * fatal_mosquito_count) / fatal_blood_loss := by
  sorry

end NUMINAMATH_CALUDE_blood_drops_per_liter_l2590_259039


namespace NUMINAMATH_CALUDE_derivative_x_minus_sin_l2590_259047

/-- The derivative of x - sin(x) is 1 - cos(x) -/
theorem derivative_x_minus_sin (x : ℝ) : 
  deriv (fun x => x - Real.sin x) x = 1 - Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_minus_sin_l2590_259047


namespace NUMINAMATH_CALUDE_mrs_hilt_candy_distribution_l2590_259085

/-- Mrs. Hilt's candy distribution problem -/
theorem mrs_hilt_candy_distribution 
  (chocolate_per_student : ℕ) 
  (chocolate_students : ℕ) 
  (hard_candy_per_student : ℕ) 
  (hard_candy_students : ℕ) 
  (gummy_per_student : ℕ) 
  (gummy_students : ℕ) 
  (h1 : chocolate_per_student = 2) 
  (h2 : chocolate_students = 3) 
  (h3 : hard_candy_per_student = 4) 
  (h4 : hard_candy_students = 2) 
  (h5 : gummy_per_student = 6) 
  (h6 : gummy_students = 4) : 
  chocolate_per_student * chocolate_students + 
  hard_candy_per_student * hard_candy_students + 
  gummy_per_student * gummy_students = 38 := by
sorry

end NUMINAMATH_CALUDE_mrs_hilt_candy_distribution_l2590_259085


namespace NUMINAMATH_CALUDE_doll_production_time_l2590_259069

/-- Represents the production details of dolls and accessories in a factory --/
structure DollProduction where
  total_dolls : ℕ
  accessories_per_doll : ℕ
  accessory_time : ℕ
  total_operation_time : ℕ

/-- Calculates the time required to make each doll --/
def time_per_doll (prod : DollProduction) : ℕ :=
  (prod.total_operation_time - prod.total_dolls * prod.accessories_per_doll * prod.accessory_time) / prod.total_dolls

/-- Theorem stating that the time to make each doll is 45 seconds --/
theorem doll_production_time (prod : DollProduction) 
  (h1 : prod.total_dolls = 12000)
  (h2 : prod.accessories_per_doll = 11)
  (h3 : prod.accessory_time = 10)
  (h4 : prod.total_operation_time = 1860000) :
  time_per_doll prod = 45 := by
  sorry

#eval time_per_doll { total_dolls := 12000, accessories_per_doll := 11, accessory_time := 10, total_operation_time := 1860000 }

end NUMINAMATH_CALUDE_doll_production_time_l2590_259069


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l2590_259003

/-- Given a circle with equation (x-3)^2+(y+4)^2=2, 
    prove that its symmetric circle with respect to y=0 
    has the equation (x-3)^2+(y-4)^2=2 -/
theorem symmetric_circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), (x - x₀)^2 + (y - y₀)^2 = 2 ∧ x₀ = 3 ∧ y₀ = -4) →
  (∃ (x₁ y₁ : ℝ), (x - x₁)^2 + (y - y₁)^2 = 2 ∧ x₁ = 3 ∧ y₁ = 4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l2590_259003


namespace NUMINAMATH_CALUDE_catherine_friends_l2590_259056

/-- The number of friends Catherine gave pens and pencils to -/
def num_friends : ℕ := sorry

/-- The initial number of pens Catherine had -/
def initial_pens : ℕ := 60

/-- The number of pens given to each friend -/
def pens_per_friend : ℕ := 8

/-- The number of pencils given to each friend -/
def pencils_per_friend : ℕ := 6

/-- The total number of pens and pencils left after giving away -/
def items_left : ℕ := 22

theorem catherine_friends :
  (initial_pens * 2 - items_left) / (pens_per_friend + pencils_per_friend) = num_friends :=
sorry

end NUMINAMATH_CALUDE_catherine_friends_l2590_259056


namespace NUMINAMATH_CALUDE_square_perimeter_32cm_l2590_259006

theorem square_perimeter_32cm (side_length : ℝ) (h : side_length = 8) : 
  4 * side_length = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_32cm_l2590_259006


namespace NUMINAMATH_CALUDE_product_94_106_l2590_259005

theorem product_94_106 : 94 * 106 = 9964 := by
  sorry

end NUMINAMATH_CALUDE_product_94_106_l2590_259005


namespace NUMINAMATH_CALUDE_complement_of_union_l2590_259010

def U : Set ℕ := {x | x ∈ Finset.range 6 \ {0}}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {2, 3}

theorem complement_of_union : (U \ (A ∪ B)) = {1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2590_259010


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2590_259098

theorem complex_on_imaginary_axis (z : ℂ) : 
  Complex.abs (z - 1) = Complex.abs (z + 1) → z.re = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2590_259098


namespace NUMINAMATH_CALUDE_edward_candy_purchase_l2590_259068

def whack_a_mole_tickets : ℕ := 3
def skee_ball_tickets : ℕ := 5
def candy_cost : ℕ := 4

theorem edward_candy_purchase :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_edward_candy_purchase_l2590_259068


namespace NUMINAMATH_CALUDE_construct_remaining_vertices_l2590_259062

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → Point2D

/-- Represents a parallel projection of a regular hexagon onto a plane -/
structure ParallelProjection where
  original : RegularHexagon
  projected : Fin 6 → Point2D

/-- Given three consecutive projected vertices of a regular hexagon, 
    the remaining three vertices can be uniquely determined -/
theorem construct_remaining_vertices 
  (p : ParallelProjection) 
  (h : ∃ (i : Fin 6), 
       (p.projected i).x ≠ (p.projected (i + 1)).x ∨ 
       (p.projected i).y ≠ (p.projected (i + 1)).y) :
  ∃! (q : ParallelProjection), 
    (∃ (i : Fin 6), 
      q.projected i = p.projected i ∧ 
      q.projected (i + 1) = p.projected (i + 1) ∧ 
      q.projected (i + 2) = p.projected (i + 2)) ∧
    (∀ (j : Fin 6), q.projected j = p.projected j) :=
  sorry

end NUMINAMATH_CALUDE_construct_remaining_vertices_l2590_259062


namespace NUMINAMATH_CALUDE_find_number_l2590_259019

theorem find_number : ∃ x : ℝ, 0.3 * ((x / 2.5) - 10.5) = 5.85 ∧ x = 75 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2590_259019


namespace NUMINAMATH_CALUDE_same_color_sock_pairs_l2590_259028

def white_socks : ℕ := 5
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose_pair (n : ℕ) : ℕ := n.choose 2

theorem same_color_sock_pairs :
  choose_pair white_socks + choose_pair brown_socks + choose_pair blue_socks + choose_pair black_socks = 27 := by
  sorry

end NUMINAMATH_CALUDE_same_color_sock_pairs_l2590_259028


namespace NUMINAMATH_CALUDE_dormitory_second_year_fraction_l2590_259015

theorem dormitory_second_year_fraction :
  ∀ (F S : ℚ),
  F + S = 1 →
  (4 : ℚ) / 5 * F = F - (1 : ℚ) / 5 * F →
  (1 : ℚ) / 3 * ((1 : ℚ) / 5 * F) = (1 : ℚ) / 15 * S →
  (14 : ℚ) / 15 * S = (7 : ℚ) / 15 →
  S = (1 : ℚ) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_second_year_fraction_l2590_259015


namespace NUMINAMATH_CALUDE_books_gotten_rid_of_correct_l2590_259048

/-- Calculates the number of coloring books gotten rid of in a sale -/
def books_gotten_rid_of (initial_stock : ℕ) (num_shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  initial_stock - (num_shelves * books_per_shelf)

/-- Proves that the number of coloring books gotten rid of is correct -/
theorem books_gotten_rid_of_correct (initial_stock : ℕ) (num_shelves : ℕ) (books_per_shelf : ℕ) :
  books_gotten_rid_of initial_stock num_shelves books_per_shelf =
  initial_stock - (num_shelves * books_per_shelf) :=
by sorry

#eval books_gotten_rid_of 40 5 4

end NUMINAMATH_CALUDE_books_gotten_rid_of_correct_l2590_259048


namespace NUMINAMATH_CALUDE_candle_flower_groupings_l2590_259036

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem candle_flower_groupings :
  (choose 4 2) * (choose 9 8) = 54 := by
  sorry

end NUMINAMATH_CALUDE_candle_flower_groupings_l2590_259036


namespace NUMINAMATH_CALUDE_equation_solution_l2590_259050

theorem equation_solution :
  let a : ℝ := 9
  let b : ℝ := 4
  let c : ℝ := 3
  ∃ x : ℝ, (x^2 + c + b^2 = (a - x)^2 + c) ∧ (x = 65 / 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2590_259050


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2590_259087

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.log (abs x)

theorem solution_set_of_inequality :
  {x : ℝ | f (x + 1) > f (2 * x - 1)} = {x : ℝ | 0 < x ∧ x < 1/2 ∨ 1/2 < x ∧ x < 2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2590_259087


namespace NUMINAMATH_CALUDE_octal_sum_equality_l2590_259097

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of three octal numbers is equal to another octal number --/
theorem octal_sum_equality : 
  decimal_to_octal (octal_to_decimal 236 + octal_to_decimal 425 + octal_to_decimal 157) = 1042 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_equality_l2590_259097


namespace NUMINAMATH_CALUDE_toilet_paper_squares_per_roll_l2590_259082

theorem toilet_paper_squares_per_roll 
  (daily_visits : ℕ) 
  (squares_per_visit : ℕ) 
  (total_rolls : ℕ) 
  (days_supply_lasts : ℕ) 
  (h1 : daily_visits = 3) 
  (h2 : squares_per_visit = 5) 
  (h3 : total_rolls = 1000) 
  (h4 : days_supply_lasts = 20000) :
  (daily_visits * squares_per_visit * days_supply_lasts) / total_rolls = 300 := by
  sorry

end NUMINAMATH_CALUDE_toilet_paper_squares_per_roll_l2590_259082


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2590_259020

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3)*(x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2590_259020


namespace NUMINAMATH_CALUDE_perpendicular_line_modulus_l2590_259058

/-- Given a line ax + y + 5 = 0 and points P and Q, prove the modulus of z = a + 4i -/
theorem perpendicular_line_modulus (a : ℝ) : 
  let P : ℝ × ℝ := (2, 4)
  let Q : ℝ × ℝ := (4, 3)
  let line (x y : ℝ) := a * x + y + 5 = 0
  let perpendicular (P Q : ℝ × ℝ) (line : ℝ → ℝ → Prop) := 
    (Q.2 - P.2) * a = -(Q.1 - P.1)  -- Perpendicular condition
  let z : ℂ := a + 4 * Complex.I
  perpendicular P Q line → Complex.abs z = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_modulus_l2590_259058


namespace NUMINAMATH_CALUDE_betty_lipstick_count_l2590_259043

/-- Represents an order with different items -/
structure Order where
  total_items : ℕ
  slipper_count : ℕ
  slipper_price : ℚ
  lipstick_price : ℚ
  hair_color_count : ℕ
  hair_color_price : ℚ
  total_paid : ℚ

/-- Calculates the number of lipstick pieces in an order -/
def lipstick_count (o : Order) : ℕ :=
  let slipper_cost := o.slipper_count * o.slipper_price
  let hair_color_cost := o.hair_color_count * o.hair_color_price
  let lipstick_cost := o.total_paid - slipper_cost - hair_color_cost
  (lipstick_cost / o.lipstick_price).num.toNat

/-- Betty's order satisfies the given conditions -/
def betty_order : Order :=
  { total_items := 18
  , slipper_count := 6
  , slipper_price := 5/2
  , lipstick_price := 5/4
  , hair_color_count := 8
  , hair_color_price := 3
  , total_paid := 44 }

theorem betty_lipstick_count : lipstick_count betty_order = 4 := by
  sorry

end NUMINAMATH_CALUDE_betty_lipstick_count_l2590_259043


namespace NUMINAMATH_CALUDE_sqrt_negative_one_squared_l2590_259033

theorem sqrt_negative_one_squared (x : ℝ) : Real.sqrt ((-1) * (-1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_one_squared_l2590_259033


namespace NUMINAMATH_CALUDE_min_c_over_d_l2590_259029

theorem min_c_over_d (x C D : ℝ) (hx : x ≠ 0) (hC : C > 0) (hD : D > 0)
  (hxC : x^4 + 1/x^4 = C) (hxD : x^2 - 1/x^2 = D) :
  ∃ (m : ℝ), (∀ x' C' D', x' ≠ 0 → C' > 0 → D' > 0 → 
    x'^4 + 1/x'^4 = C' → x'^2 - 1/x'^2 = D' → C' / D' ≥ m) ∧ 
  (∃ x₀ C₀ D₀, x₀ ≠ 0 ∧ C₀ > 0 ∧ D₀ > 0 ∧ 
    x₀^4 + 1/x₀^4 = C₀ ∧ x₀^2 - 1/x₀^2 = D₀ ∧ C₀ / D₀ = m) ∧
  m = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_c_over_d_l2590_259029


namespace NUMINAMATH_CALUDE_abc_inequality_l2590_259013

theorem abc_inequality (a b c : ℝ) 
  (ha : a = 2 * Real.sqrt 7)
  (hb : b = 3 * Real.sqrt 5)
  (hc : c = 5 * Real.sqrt 2) : 
  c > b ∧ b > a :=
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2590_259013


namespace NUMINAMATH_CALUDE_time_to_make_one_toy_l2590_259053

/-- Given that a worker makes 40 toys in 80 hours, prove that it takes 2 hours to make one toy. -/
theorem time_to_make_one_toy (total_hours : ℝ) (total_toys : ℝ) 
  (h1 : total_hours = 80) (h2 : total_toys = 40) : 
  total_hours / total_toys = 2 := by
sorry

end NUMINAMATH_CALUDE_time_to_make_one_toy_l2590_259053


namespace NUMINAMATH_CALUDE_sqrt_s6_plus_s3_l2590_259066

theorem sqrt_s6_plus_s3 (s : ℝ) : Real.sqrt (s^6 + s^3) = |s| * Real.sqrt (s * (s^3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_s6_plus_s3_l2590_259066


namespace NUMINAMATH_CALUDE_garden_trees_l2590_259000

/-- The number of trees in a garden with given specifications -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  yard_length / tree_spacing + 1

/-- Theorem stating the number of trees in the garden -/
theorem garden_trees : num_trees 700 28 = 26 := by
  sorry

end NUMINAMATH_CALUDE_garden_trees_l2590_259000


namespace NUMINAMATH_CALUDE_quadratic_sum_abc_l2590_259096

/-- The quadratic function f(x) = -4x^2 + 20x + 196 -/
def f (x : ℝ) : ℝ := -4 * x^2 + 20 * x + 196

/-- The sum of a, b, and c when f(x) is expressed as a(x+b)^2 + c -/
def sum_abc : ℝ := 213.5

theorem quadratic_sum_abc :
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = sum_abc) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_abc_l2590_259096


namespace NUMINAMATH_CALUDE_parallel_condition_l2590_259072

/-- Two lines are parallel if and only if they have the same slope -/
def parallel (m1 a1 b1 : ℝ) (m2 a2 b2 : ℝ) : Prop :=
  m1 = m2

/-- The line l1 with equation ax + 2y - 3 = 0 -/
def l1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y - 3 = 0

/-- The line l2 with equation 2x + y - a = 0 -/
def l2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x + y - a = 0

/-- The statement that a = 4 is a necessary and sufficient condition for l1 to be parallel to l2 -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, parallel (-a/2) 0 0 (-2) 0 0) ↔ a = 4 :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l2590_259072


namespace NUMINAMATH_CALUDE_library_visitors_average_l2590_259001

/-- Calculates the average number of visitors per day in a 30-day month starting with a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays := 5
  let totalOtherDays := 30 - totalSundays
  let totalVisitors := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  totalVisitors / 30

theorem library_visitors_average :
  averageVisitors 1000 700 = 750 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2590_259001


namespace NUMINAMATH_CALUDE_composition_constant_term_l2590_259079

/-- Given two functions f and g, and a condition on their composition,
    prove that the constant term in the composed function is 14. -/
theorem composition_constant_term
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = 5 * x - 1)
  (hg : ∃ c, ∀ x, g x = 2 * c * x + 3)
  (h_comp : ∃ d, ∀ x, f (g x) = 15 * x + d) :
  ∃ d, (∀ x, f (g x) = 15 * x + d) ∧ d = 14 := by sorry

end NUMINAMATH_CALUDE_composition_constant_term_l2590_259079


namespace NUMINAMATH_CALUDE_triangle_side_length_l2590_259049

noncomputable def f (x : ℝ) := Real.sin (7 * Real.pi / 6 - 2 * x) - 2 * Real.sin x ^ 2 + 1

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : f A = 1/2)
  (h2 : b - a = c - b)  -- arithmetic sequence condition
  (h3 : b * c * Real.cos A = 9) : 
  a = 3 * Real.sqrt 2 := by 
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2590_259049


namespace NUMINAMATH_CALUDE_parabola_transformation_l2590_259080

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3) - 2

-- Define the resulting parabola
def result_parabola (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem parabola_transformation :
  ∀ x : ℝ, transform original_parabola x = result_parabola x :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2590_259080


namespace NUMINAMATH_CALUDE_min_equal_number_example_min_equal_number_is_minimum_l2590_259037

/-- Given three initial numbers on a blackboard, this function represents
    the minimum number to which all three can be made equal by repeatedly
    selecting two numbers and adding 1 to each. -/
def min_equal_number (a b c : ℕ) : ℕ :=
  (a + b + c + 2 * ((a + b + c) % 3)) / 3

/-- Theorem stating that 747 is the minimum number to which 20, 201, and 2016
    can be made equal using the described operation. -/
theorem min_equal_number_example : min_equal_number 20 201 2016 = 747 := by
  sorry

/-- Theorem stating that the result of min_equal_number is indeed the minimum
    possible number to which the initial numbers can be made equal. -/
theorem min_equal_number_is_minimum (a b c : ℕ) :
  ∀ n : ℕ, (∃ k : ℕ, a + k ≤ n ∧ b + k ≤ n ∧ c + k ≤ n) →
  min_equal_number a b c ≤ n := by
  sorry

end NUMINAMATH_CALUDE_min_equal_number_example_min_equal_number_is_minimum_l2590_259037


namespace NUMINAMATH_CALUDE_number_expression_not_equal_l2590_259090

theorem number_expression_not_equal (x : ℝ) : 5 * x + 7 ≠ 5 * (x + 7) := by
  sorry

end NUMINAMATH_CALUDE_number_expression_not_equal_l2590_259090


namespace NUMINAMATH_CALUDE_average_playtime_l2590_259084

def wednesday_hours : ℝ := 2
def thursday_hours : ℝ := 2
def friday_additional_hours : ℝ := 3
def total_days : ℕ := 3

theorem average_playtime :
  let total_hours := wednesday_hours + thursday_hours + (wednesday_hours + friday_additional_hours)
  total_hours / total_days = 3 := by
sorry

end NUMINAMATH_CALUDE_average_playtime_l2590_259084


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l2590_259016

/-- The area of a rhombus formed by connecting the midpoints of a square -/
theorem rhombus_area_in_square (side_length : ℝ) (h : side_length = 10) :
  let square_diagonal := side_length * Real.sqrt 2
  let rhombus_side := square_diagonal / 2
  let rhombus_area := (rhombus_side * rhombus_side) / 2
  rhombus_area = 25 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l2590_259016


namespace NUMINAMATH_CALUDE_function_is_identity_l2590_259054

def is_positive (n : ℕ) : Prop := n > 0

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ m n, is_positive m → is_positive n → f (f m + f n) = m + n

theorem function_is_identity 
  (f : ℕ → ℕ) 
  (h : satisfies_functional_equation f) :
  ∀ x, is_positive x → f x = x :=
sorry

end NUMINAMATH_CALUDE_function_is_identity_l2590_259054


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l2590_259060

/-- Given two circles intersecting at points A(1,3) and B(m,-1), with their centers lying on the line x-y+c=0, prove that m+c = 3 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  (∃ (C D : ℝ × ℝ), 
    (C.1 - C.2 + c = 0) ∧ 
    (D.1 - D.2 + c = 0) ∧ 
    ((1 - C.1)^2 + (3 - C.2)^2 = (m - C.1)^2 + (-1 - C.2)^2) ∧
    ((1 - D.1)^2 + (3 - D.2)^2 = (m - D.1)^2 + (-1 - D.2)^2)) →
  m + c = 3 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_circles_sum_l2590_259060


namespace NUMINAMATH_CALUDE_half_sum_negative_l2590_259045

theorem half_sum_negative (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / 2 ∧ y < 0) ↔ (x + 3) / 2 < 0 := by
sorry

end NUMINAMATH_CALUDE_half_sum_negative_l2590_259045


namespace NUMINAMATH_CALUDE_gcd_840_1764_l2590_259008

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l2590_259008


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l2590_259007

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 2

/-- Calculates the number of possible arrangements for the chess team photo -/
def num_arrangements : ℕ := num_girls.factorial * num_boys.factorial

/-- Theorem stating that the number of possible arrangements is 12 -/
theorem chess_team_arrangements :
  num_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l2590_259007


namespace NUMINAMATH_CALUDE_geometric_sequence_identity_l2590_259092

/-- 
Given a geometric sequence and three of its terms L, M, N at positions l, m, n respectively,
prove that L^(m-n) * M^(n-l) * N^(l-m) = 1.
-/
theorem geometric_sequence_identity 
  {α : Type*} [Field α] 
  (a : ℕ → α) 
  (q : α) 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (l m n : ℕ) :
  (a l) ^ (m - n) * (a m) ^ (n - l) * (a n) ^ (l - m) = 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_identity_l2590_259092


namespace NUMINAMATH_CALUDE_line_through_center_chord_length_l2590_259078

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 11/2

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define a line passing through P
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y - point_P.2 = k * (x - point_P.1)

-- Theorem 1: Equation of line passing through P and center of circle
theorem line_through_center : 
  ∃ (x y : ℝ), line_through_P 2 x y ∧ 2*x - y - 2 = 0 := by sorry

-- Theorem 2: Length of chord AB when line slope is 1
theorem chord_length : 
  ∃ (A B : ℝ × ℝ), 
    (circle_C A.1 A.2) ∧ 
    (circle_C B.1 B.2) ∧ 
    (line_through_P 1 A.1 A.2) ∧ 
    (line_through_P 1 B.1 B.2) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 20) := by sorry

end NUMINAMATH_CALUDE_line_through_center_chord_length_l2590_259078


namespace NUMINAMATH_CALUDE_min_layoff_rounds_is_four_l2590_259026

def initial_employees : ℕ := 1000
def layoff_rate : ℝ := 0.1
def total_layoffs : ℕ := 271

def remaining_employees (n : ℕ) : ℝ :=
  initial_employees * (1 - layoff_rate) ^ n

def layoffs_after_rounds (n : ℕ) : ℝ :=
  initial_employees - remaining_employees n

theorem min_layoff_rounds_is_four :
  (∀ k < 4, layoffs_after_rounds k < total_layoffs) ∧
  layoffs_after_rounds 4 ≥ total_layoffs := by sorry

end NUMINAMATH_CALUDE_min_layoff_rounds_is_four_l2590_259026


namespace NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l2590_259014

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_has_three_prime_factors : 
  num_distinct_prime_factors (sum_of_divisors 450) = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_divisors_450_has_three_prime_factors_l2590_259014


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l2590_259065

/-- Proves that the base of an isosceles triangle is 10, given specific conditions about its perimeter and relationship to an equilateral triangle. -/
theorem isosceles_triangle_base : 
  ∀ (s b : ℝ),
  -- Equilateral triangle perimeter condition
  3 * s = 45 →
  -- Isosceles triangle perimeter condition
  2 * s + b = 40 →
  -- Base of isosceles triangle is 10
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l2590_259065


namespace NUMINAMATH_CALUDE_prob_odd_diagonals_eq_1_126_l2590_259011

/-- Represents a 3x3 grid arrangement of numbers 1 to 9 -/
def Grid := Fin 9 → Fin 9

/-- Checks if a given grid has odd sums on both diagonals -/
def has_odd_diagonal_sums (g : Grid) : Prop :=
  (g 0 + g 4 + g 8).val % 2 = 1 ∧ (g 2 + g 4 + g 6).val % 2 = 1

/-- The set of all valid grid arrangements -/
def all_grids : Finset Grid :=
  sorry

/-- The set of grid arrangements with odd diagonal sums -/
def odd_diagonal_grids : Finset Grid :=
  sorry

/-- The probability of a random grid having odd diagonal sums -/
def prob_odd_diagonals : ℚ :=
  (odd_diagonal_grids.card : ℚ) / (all_grids.card : ℚ)

theorem prob_odd_diagonals_eq_1_126 : prob_odd_diagonals = 1 / 126 :=
  sorry

end NUMINAMATH_CALUDE_prob_odd_diagonals_eq_1_126_l2590_259011


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l2590_259030

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_first_three : a 0 + a 1 + a 2 = 168
  diff_2_5 : a 1 - a 4 = 42

/-- The 6th term of the arithmetic progression is 3 -/
theorem sixth_term_is_three (ap : ArithmeticProgression) : ap.a 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l2590_259030


namespace NUMINAMATH_CALUDE_max_value_abc_l2590_259081

theorem max_value_abc (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 ∧
  ∃ a' b' c' : ℝ, (a' ≠ 0 ∨ b' ≠ 0 ∨ c' ≠ 0) ∧
    (a' * b' + 2 * b' * c') / (a'^2 + b'^2 + c'^2) = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2590_259081


namespace NUMINAMATH_CALUDE_complement_of_A_l2590_259044

def U : Set Nat := {2, 4, 6, 8, 10}
def A : Set Nat := {2, 6, 8}

theorem complement_of_A : (Aᶜ : Set Nat) = {4, 10} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l2590_259044


namespace NUMINAMATH_CALUDE_movie_attendance_l2590_259076

/-- The number of people that can ride in each car -/
def people_per_car : ℕ := 6

/-- The number of cars needed -/
def cars_needed : ℕ := 18

/-- The total number of people going to the movie -/
def total_people : ℕ := people_per_car * cars_needed

theorem movie_attendance : total_people = 108 := by
  sorry

end NUMINAMATH_CALUDE_movie_attendance_l2590_259076


namespace NUMINAMATH_CALUDE_sequence_convergence_l2590_259059

/-- Given an integer k > 5, this function represents the operation described in the problem.
    It takes a number in base k, calculates the sum of its digits, multiplies it by (k-1)^2,
    and appends this product to the original number. -/
def baseKOperation (k : ℕ) (n : ℕ) : ℕ :=
  sorry

/-- This function represents the sequence generated by repeatedly applying the baseKOperation -/
def generateSequence (k : ℕ) (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => baseKOperation k (generateSequence k start n)

/-- The theorem states that for any k > 5, the sequence generated by the described process
    will eventually converge to 2(k-1)^3 -/
theorem sequence_convergence (k : ℕ) (start : ℕ) (h : k > 5) :
  ∃ N : ℕ, ∀ n ≥ N, generateSequence k start n = 2 * (k - 1)^3 :=
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l2590_259059


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2590_259064

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m > 0) → m ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2590_259064


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2590_259091

theorem min_value_of_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 2/n = 1) :
  m + n ≥ (Real.sqrt 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2590_259091


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2590_259018

theorem greatest_divisor_with_remainders (a b r1 r2 : ℕ) (h1 : a = 60) (h2 : b = 190) (h3 : r1 = 6) (h4 : r2 = 10) :
  Nat.gcd (a - r1) (b - r2) = 18 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2590_259018


namespace NUMINAMATH_CALUDE_part_1_part_2_l2590_259023

-- Define the inequality
def inequality (a x : ℝ) : Prop := a * x^2 - 2*a + 1 < (1 - a) * x

-- Define the solution set for part (1)
def solution_set_1 (x : ℝ) : Prop := x < -4 ∨ x > 1

-- Define the condition for part (2)
def condition_2 (a : ℝ) : Prop := a > 0

-- Define the property of having exactly 7 prime elements in the solution set
def has_seven_primes (a : ℝ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 p6 p7 : ℕ),
    Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime p5 ∧ Prime p6 ∧ Prime p7 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧ p5 < p6 ∧ p6 < p7 ∧
    (∀ x : ℝ, inequality a x ↔ (x < p1 ∨ x > p7))

-- Theorem for part (1)
theorem part_1 : 
  (∀ x : ℝ, inequality a x ↔ solution_set_1 x) → a = -1/2 :=
sorry

-- Theorem for part (2)
theorem part_2 :
  condition_2 a → has_seven_primes a → 1/21 ≤ a ∧ a < 1/19 :=
sorry

end NUMINAMATH_CALUDE_part_1_part_2_l2590_259023


namespace NUMINAMATH_CALUDE_angela_deliveries_l2590_259093

/-- Calculates the total number of meals and packages delivered -/
def total_deliveries (meals : ℕ) (package_multiplier : ℕ) : ℕ :=
  meals + meals * package_multiplier

/-- Proves that given 3 meals and 8 times as many packages, the total deliveries is 27 -/
theorem angela_deliveries : total_deliveries 3 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_angela_deliveries_l2590_259093


namespace NUMINAMATH_CALUDE_fifteen_ones_sum_multiple_of_30_l2590_259055

theorem fifteen_ones_sum_multiple_of_30 : 
  (Nat.choose 14 9 : ℕ) = 2002 := by sorry

end NUMINAMATH_CALUDE_fifteen_ones_sum_multiple_of_30_l2590_259055


namespace NUMINAMATH_CALUDE_range_of_a_range_is_nonnegative_reals_l2590_259067

-- Define the set A
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 = a}

-- State the theorem
theorem range_of_a (h : ∃ x, x ∈ A a) : a ≥ 0 := by
  sorry

-- Prove that this covers the entire range [0, +∞)
theorem range_is_nonnegative_reals : 
  ∀ a ≥ 0, ∃ x, x ∈ A a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_range_is_nonnegative_reals_l2590_259067


namespace NUMINAMATH_CALUDE_solve_system_l2590_259046

theorem solve_system (x y : ℝ) 
  (eq1 : 2 * x = 3 * x - 25)
  (eq2 : x + y = 50) : 
  x = 25 ∧ y = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2590_259046


namespace NUMINAMATH_CALUDE_complement_A_in_U_intersection_A_B_l2590_259070

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 < x ∧ x ≤ 1}

-- Theorem for the complement of A in U
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -2 ∨ (3 ≤ x ∧ x ≤ 4)} := by sorry

-- Theorem for the intersection of A and B
theorem intersection_A_B : 
  (A ∩ B) = {x | -2 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_intersection_A_B_l2590_259070


namespace NUMINAMATH_CALUDE_probability_at_least_two_same_l2590_259021

def num_dice : ℕ := 8
def num_sides : ℕ := 8

theorem probability_at_least_two_same :
  let total_outcomes := num_sides ^ num_dice
  let all_different := Nat.factorial num_sides
  (1 - (all_different : ℚ) / total_outcomes) = 2043 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_same_l2590_259021


namespace NUMINAMATH_CALUDE_solve_for_x_l2590_259040

theorem solve_for_x (x y : ℝ) (h1 : x - y = 15) (h2 : x + y = 9) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2590_259040


namespace NUMINAMATH_CALUDE_circle_center_coordinates_sum_l2590_259073

theorem circle_center_coordinates_sum (x y : ℝ) : 
  x^2 + y^2 - 12*x + 10*y = 40 → (x - 6)^2 + (y + 5)^2 = 101 ∧ x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_sum_l2590_259073


namespace NUMINAMATH_CALUDE_unique_solution_l2590_259099

def base_6_value (s h e : ℕ) : ℕ := s * 36 + h * 6 + e

theorem unique_solution :
  ∀ (s h e : ℕ),
    s ≠ 0 ∧ h ≠ 0 ∧ e ≠ 0 →
    s < 6 ∧ h < 6 ∧ e < 6 →
    s ≠ h ∧ s ≠ e ∧ h ≠ e →
    base_6_value s h e + base_6_value 0 h e = base_6_value h e s →
    s = 4 ∧ h = 2 ∧ e = 5 ∧ (s + h + e) % 6 = 5 ∧ ((s + h + e) / 6) % 6 = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2590_259099


namespace NUMINAMATH_CALUDE_solution_x_l2590_259057

theorem solution_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 5) (h2 : y + 1 / x = 7 / 4) :
  x = 4 / 7 ∨ x = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_x_l2590_259057
