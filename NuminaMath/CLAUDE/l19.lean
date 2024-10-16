import Mathlib

namespace NUMINAMATH_CALUDE_root_in_interval_l19_1946

def f (x : ℝ) := x^3 + x - 4

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 1 2 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l19_1946


namespace NUMINAMATH_CALUDE_hyperbola_sum_l19_1923

/-- Proves that for a hyperbola with given properties, h + k + a + b = 11 -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  c = Real.sqrt 41 ∧ 
  a = 4 ∧ 
  c^2 = a^2 + b^2 →
  h + k + a + b = 11 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l19_1923


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l19_1951

theorem absolute_value_inequality (x : ℝ) : 
  (3 ≤ |x + 2| ∧ |x + 2| ≤ 8) ↔ ((-10 ≤ x ∧ x ≤ -5) ∨ (1 ≤ x ∧ x ≤ 6)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l19_1951


namespace NUMINAMATH_CALUDE_f_equiv_g_l19_1930

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 1
def g (t : ℝ) : ℝ := t^2 - 2*t + 1

-- Theorem stating that f and g are equivalent for all real numbers
theorem f_equiv_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equiv_g_l19_1930


namespace NUMINAMATH_CALUDE_min_value_of_function_l19_1901

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  (x^2 + x + 25) / x ≥ 11 ∧ ∃ y > 0, (y^2 + y + 25) / y = 11 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l19_1901


namespace NUMINAMATH_CALUDE_sphere_in_cone_l19_1966

theorem sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - b
  sphere_radius = (cone_base_radius * cone_height) / (Real.sqrt (cone_base_radius^2 + cone_height^2) + cone_height) →
  b + d = 12.5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cone_l19_1966


namespace NUMINAMATH_CALUDE_students_in_both_events_l19_1905

/-- The number of students who participated in both volleyball and track and field events -/
def students_in_both (total : ℕ) (volleyball : ℕ) (track : ℕ) (none : ℕ) : ℕ :=
  volleyball + track - (total - none)

/-- Theorem stating the number of students who participated in both events -/
theorem students_in_both_events :
  students_in_both 45 12 20 19 = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_events_l19_1905


namespace NUMINAMATH_CALUDE_total_snacks_weight_l19_1975

-- Define the conversion rate from ounces to pounds
def ounces_to_pounds : ℚ → ℚ := (· / 16)

-- Define the weights of snacks
def peanuts_weight : ℚ := 0.1
def raisins_weight_oz : ℚ := 5
def almonds_weight : ℚ := 0.3

-- Theorem to prove
theorem total_snacks_weight :
  peanuts_weight + ounces_to_pounds raisins_weight_oz + almonds_weight = 0.7125 := by
  sorry

end NUMINAMATH_CALUDE_total_snacks_weight_l19_1975


namespace NUMINAMATH_CALUDE_one_fourth_divided_by_two_l19_1943

theorem one_fourth_divided_by_two : (1 / 4 : ℚ) / 2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_divided_by_two_l19_1943


namespace NUMINAMATH_CALUDE_water_in_bucket_l19_1981

theorem water_in_bucket (initial_water : ℝ) (poured_out : ℝ) (remaining_water : ℝ) :
  initial_water = 0.8 →
  poured_out = 0.2 →
  remaining_water = initial_water - poured_out →
  remaining_water = 0.6 := by
sorry

end NUMINAMATH_CALUDE_water_in_bucket_l19_1981


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l19_1950

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l19_1950


namespace NUMINAMATH_CALUDE_possible_solutions_l19_1967

theorem possible_solutions (a b : ℝ) (h1 : a + 1 > b) (h2 : b > 2/a) (h3 : 2/a > 0) :
  (∃ a₀, a₀ = 2 ∧ a₀ + 1 > 2/a₀ ∧ 2/a₀ > 0) ∧
  (∃ b₀, b₀ = 1 ∧ (∃ a₁, a₁ + 1 > b₀ ∧ b₀ > 2/a₁ ∧ 2/a₁ > 0)) :=
sorry

end NUMINAMATH_CALUDE_possible_solutions_l19_1967


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l19_1918

/-- Given a family with the following properties:
  * The total number of children is 180
  * Boys are given $3900 to share
  * Each boy receives $52
  Prove that the ratio of boys to girls is 5:7 -/
theorem boys_to_girls_ratio (total_children : ℕ) (boys_money : ℕ) (boy_share : ℕ)
  (h_total : total_children = 180)
  (h_money : boys_money = 3900)
  (h_share : boy_share = 52)
  : ∃ (boys girls : ℕ), boys + girls = total_children ∧ 
    boys * boy_share = boys_money ∧
    boys * 7 = girls * 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l19_1918


namespace NUMINAMATH_CALUDE_three_digit_subtraction_problem_l19_1909

theorem three_digit_subtraction_problem :
  ∀ h t u : ℕ,
  h ≤ 9 ∧ t ≤ 9 ∧ u ≤ 9 →  -- Ensure single-digit numbers
  u = h - 5 →
  (100 * h + 10 * t + u) - (100 * h + 10 * u + t) = 96 →
  h = 5 ∧ t = 9 ∧ u = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_subtraction_problem_l19_1909


namespace NUMINAMATH_CALUDE_a6_value_l19_1969

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a6_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 2) ^ 2 - 8 * (a 2) + 4 = 0 →
  (a 10) ^ 2 - 8 * (a 10) + 4 = 0 →
  a 6 = 2 := by
sorry

end NUMINAMATH_CALUDE_a6_value_l19_1969


namespace NUMINAMATH_CALUDE_extreme_value_and_minimum_a_l19_1922

noncomputable def f (a : ℤ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x

theorem extreme_value_and_minimum_a :
  (∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  f 1 1 = (1/2) ∧
  (∀ (a : ℤ), (∀ (x : ℝ), x > 0 → f a x ≥ (1 - a) * x + 1) → a ≥ 2) ∧
  (∀ (x : ℝ), x > 0 → f 2 x ≥ (1 - 2) * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_and_minimum_a_l19_1922


namespace NUMINAMATH_CALUDE_rhombus_existence_condition_l19_1996

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  /-- The perimeter of the rhombus -/
  k : ℝ
  /-- The sum of the diagonals of the rhombus -/
  u : ℝ
  /-- The perimeter is positive -/
  k_pos : k > 0
  /-- The sum of diagonals is positive -/
  u_pos : u > 0

/-- The condition for the existence of a rhombus given its perimeter and sum of diagonals -/
theorem rhombus_existence_condition (r : Rhombus) : 
  Real.sqrt 2 * r.u ≤ r.k ∧ r.k < 2 * r.u :=
by sorry

end NUMINAMATH_CALUDE_rhombus_existence_condition_l19_1996


namespace NUMINAMATH_CALUDE_geometric_proportion_proof_l19_1939

theorem geometric_proportion_proof :
  let a : ℝ := 21
  let b : ℝ := 7
  let c : ℝ := 9
  let d : ℝ := 3
  (a / b = c / d) ∧
  (a + d = 24) ∧
  (b + c = 16) ∧
  (a^2 + b^2 + c^2 + d^2 = 580) := by
  sorry

end NUMINAMATH_CALUDE_geometric_proportion_proof_l19_1939


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l19_1944

/-- Represents the ages of Tom and Jerry -/
structure Ages where
  tom : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  (ages.tom - 3 = 4 * (ages.jerry - 3)) ∧ 
  (ages.tom - 8 = 5 * (ages.jerry - 8))

/-- The future age ratio condition -/
def future_ratio (ages : Ages) (years : ℕ) : Prop :=
  3 * (ages.jerry + years) = ages.tom + years

/-- The main theorem to prove -/
theorem age_ratio_theorem : 
  ∃ (ages : Ages), age_conditions ages → future_ratio ages 7 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l19_1944


namespace NUMINAMATH_CALUDE_gcd_of_36_45_75_l19_1940

/-- The greatest common divisor of 36, 45, and 75 is 3 -/
theorem gcd_of_36_45_75 : Nat.gcd 36 (Nat.gcd 45 75) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_36_45_75_l19_1940


namespace NUMINAMATH_CALUDE_ellipse_equation_l19_1991

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0, and the major axis is √2
    times the minor axis, if the line y = -x + 1 intersects the ellipse at points A and B such that
    the length of chord AB is 4√5/3, then the equation of the ellipse is x²/4 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : a^2 = 2 * b^2) 
    (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
      x1^2/a^2 + y1^2/b^2 = 1 ∧ 
      x2^2/a^2 + y2^2/b^2 = 1 ∧
      y1 = -x1 + 1 ∧ 
      y2 = -x2 + 1 ∧ 
      (x2 - x1)^2 + (y2 - y1)^2 = (4*Real.sqrt 5/3)^2) :
  ∀ x y : ℝ, x^2/4 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l19_1991


namespace NUMINAMATH_CALUDE_subtracted_amount_l19_1926

theorem subtracted_amount (x : ℝ) (h : x = 2.625) : 8 * x - 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l19_1926


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l19_1982

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 48 * Real.sqrt 6 = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l19_1982


namespace NUMINAMATH_CALUDE_sphere_surface_area_l19_1935

theorem sphere_surface_area (v : ℝ) (r : ℝ) (h : v = 72 * Real.pi) :
  4 * Real.pi * r^2 = 36 * Real.pi * (2^(2/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l19_1935


namespace NUMINAMATH_CALUDE_cos_a_minus_pi_fourth_l19_1919

theorem cos_a_minus_pi_fourth (a : ℝ) (ha : a ∈ Set.Ioo 0 2) (h_tan : Real.tan a = 2) :
  Real.cos (a - π / 4) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_a_minus_pi_fourth_l19_1919


namespace NUMINAMATH_CALUDE_candy_problem_l19_1962

theorem candy_problem (given_away eaten remaining : ℕ) 
  (h1 : given_away = 18)
  (h2 : eaten = 7)
  (h3 : remaining = 16) :
  given_away + eaten + remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l19_1962


namespace NUMINAMATH_CALUDE_students_taking_both_courses_l19_1980

theorem students_taking_both_courses 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 69) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 15) :
  french + german - (total - neither) = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_both_courses_l19_1980


namespace NUMINAMATH_CALUDE_number_order_l19_1973

theorem number_order (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (1 / a > Real.sqrt a) ∧ (Real.sqrt a > a) ∧ (a > a^2) := by
  sorry

end NUMINAMATH_CALUDE_number_order_l19_1973


namespace NUMINAMATH_CALUDE_cannot_determine_dracula_state_l19_1993

-- Define the possible states for the Transylvanian and Count Dracula
inductive State : Type
  | Human : State
  | Undead : State
  | Alive : State
  | Dead : State

-- Define the Transylvanian's statement
def transylvanianStatement (transylvanian : State) (dracula : State) : Prop :=
  (transylvanian = State.Human) → (dracula = State.Alive)

-- Define the theorem
theorem cannot_determine_dracula_state :
  ∀ (transylvanian : State) (dracula : State),
    transylvanianStatement transylvanian dracula →
    ¬(∀ (dracula' : State), dracula' = State.Alive ∨ dracula' = State.Dead) :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_dracula_state_l19_1993


namespace NUMINAMATH_CALUDE_cone_volume_l19_1984

/-- Given a cone with base radius 1 and slant height equal to the diameter of the base,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) :
  r = 1 →
  l = 2 * r →
  h ^ 2 + r ^ 2 = l ^ 2 →
  (1 / 3) * π * r ^ 2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l19_1984


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l19_1904

theorem difference_of_reciprocals (p q : ℚ) : 
  3 / p = 6 → 3 / q = 18 → p - q = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l19_1904


namespace NUMINAMATH_CALUDE_bag_production_l19_1942

/-- Given that 15 machines produce 45 bags per minute, 
    prove that 150 machines will produce 3600 bags in 8 minutes. -/
theorem bag_production 
  (machines : ℕ) 
  (bags_per_minute : ℕ) 
  (time : ℕ) 
  (h1 : machines = 15) 
  (h2 : bags_per_minute = 45) 
  (h3 : time = 8) :
  (150 : ℕ) * bags_per_minute * time / machines = 3600 :=
sorry

end NUMINAMATH_CALUDE_bag_production_l19_1942


namespace NUMINAMATH_CALUDE_set_membership_implies_m_values_l19_1988

theorem set_membership_implies_m_values (m : ℝ) : 
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → (m = 3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_values_l19_1988


namespace NUMINAMATH_CALUDE_min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l19_1907

/-- Represents the problem of finding the minimum bailing rate -/
def MinBailingRateProblem (distance : Real) (rowingSpeed : Real) (waterIntakeRate : Real) (maxWaterCapacity : Real) : Prop :=
  ∃ (bailingRate : Real),
    bailingRate ≥ 0 ∧
    (distance / rowingSpeed) * 60 * (waterIntakeRate - bailingRate) ≤ maxWaterCapacity ∧
    ∀ (r : Real), r ≥ 0 ∧ (distance / rowingSpeed) * 60 * (waterIntakeRate - r) ≤ maxWaterCapacity → r ≥ bailingRate

/-- The solution to the minimum bailing rate problem -/
theorem min_bailing_rate_solution :
  MinBailingRateProblem 1 4 10 30 → (∃ (minRate : Real), minRate = 8) :=
by
  sorry

/-- Proof that 8 gallons per minute is the minimum bailing rate required -/
theorem bailing_rate_is_eight_gallons_per_minute :
  MinBailingRateProblem 1 4 10 30 :=
by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_solution_bailing_rate_is_eight_gallons_per_minute_l19_1907


namespace NUMINAMATH_CALUDE_min_value_of_expression_l19_1910

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + a*b - 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + x*y - 3 = 0 → 4*x + y ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l19_1910


namespace NUMINAMATH_CALUDE_fill_with_corners_l19_1986

/-- A corner is represented by a triple of unit cubes -/
def Corner := Fin 3 → Fin 3 → Fin 3 → Bool

/-- A box is represented by its dimensions -/
structure Box where
  m : ℕ
  n : ℕ
  k : ℕ
  m_gt_one : m > 1
  n_gt_one : n > 1
  k_gt_one : k > 1

/-- Predicate to check if a box can be filled with bars and corners -/
def canFillWithBarsAndCorners (b : Box) : Prop := sorry

/-- Predicate to check if a box can be filled with only corners -/
def canFillWithOnlyCorners (b : Box) : Prop := sorry

/-- Main theorem: If a box can be filled with bars and corners, it can be filled with only corners -/
theorem fill_with_corners (b : Box) :
  canFillWithBarsAndCorners b → canFillWithOnlyCorners b := by sorry

end NUMINAMATH_CALUDE_fill_with_corners_l19_1986


namespace NUMINAMATH_CALUDE_y_coordinate_of_C_is_zero_l19_1985

-- Define the parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Define the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

-- Define the conditions
axiom A : Point
axiom B : Point
axiom C : Point

axiom distinct_points : A ≠ B ∧ B ≠ C ∧ A ≠ C

axiom on_parabola : 
  A.y = parabola A.x ∧ 
  B.y = parabola B.x ∧ 
  C.y = parabola C.x

axiom AB_perpendicular_x_axis : A.x = B.x

axiom right_triangle : 
  (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0

axiom triangle_area : 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2 = 2016

-- Theorem to prove
theorem y_coordinate_of_C_is_zero : C.y = 0 := by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_C_is_zero_l19_1985


namespace NUMINAMATH_CALUDE_triangle_area_l19_1994

/-- Given a triangle with perimeter 36 cm and inradius 2.5 cm, its area is 45 cm² -/
theorem triangle_area (p : ℝ) (r : ℝ) (A : ℝ) 
    (h1 : p = 36) -- perimeter is 36 cm
    (h2 : r = 2.5) -- inradius is 2.5 cm
    (h3 : A = r * (p / 2)) -- area formula: A = r * s, where s is semiperimeter (p/2)
    : A = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l19_1994


namespace NUMINAMATH_CALUDE_pants_purchase_l19_1964

theorem pants_purchase (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_paid : ℝ) :
  original_price = 45 →
  discount_rate = 0.20 →
  tax_rate = 0.10 →
  total_paid = 396 →
  ∃ (num_pairs : ℕ), 
    (num_pairs : ℝ) * (original_price * (1 - discount_rate) * (1 + tax_rate)) = total_paid ∧
    num_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_pants_purchase_l19_1964


namespace NUMINAMATH_CALUDE_sample_size_comparison_l19_1968

theorem sample_size_comparison (n m : ℕ+) (x_bar y_bar z a : ℝ) :
  x_bar ≠ y_bar →
  0 < a →
  a < 1/2 →
  z = a * x_bar + (1 - a) * y_bar →
  n > m :=
sorry

end NUMINAMATH_CALUDE_sample_size_comparison_l19_1968


namespace NUMINAMATH_CALUDE_monitor_width_l19_1913

theorem monitor_width (width height diagonal : ℝ) : 
  width / height = 16 / 9 →
  width ^ 2 + height ^ 2 = diagonal ^ 2 →
  diagonal = 24 →
  width = 384 / Real.sqrt 337 :=
by sorry

end NUMINAMATH_CALUDE_monitor_width_l19_1913


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l19_1934

/-- The number of white balls in the box -/
def white_balls : Nat := 5

/-- The number of black balls in the box -/
def black_balls : Nat := 5

/-- The total number of balls in the box -/
def total_balls : Nat := white_balls + black_balls

/-- The number of ways to arrange white_balls white balls and black_balls black balls -/
def total_arrangements : Nat := Nat.choose total_balls white_balls

/-- The number of valid alternating color patterns -/
def valid_patterns : Nat := 2

/-- The probability of drawing all balls in an alternating color pattern -/
def alternating_probability : ℚ := valid_patterns / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l19_1934


namespace NUMINAMATH_CALUDE_rope_division_l19_1952

def rope_length : ℝ := 3
def num_segments : ℕ := 7

theorem rope_division (segment_fraction : ℝ) (segment_length : ℝ) :
  (segment_fraction = 1 / num_segments) ∧
  (segment_length = rope_length / num_segments) ∧
  (segment_fraction = 1 / 7) ∧
  (segment_length = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_rope_division_l19_1952


namespace NUMINAMATH_CALUDE_definite_integral_result_l19_1929

theorem definite_integral_result : 
  ∫ x in -Real.arcsin (2 / Real.sqrt 5)..π/4, (2 - Real.tan x) / (Real.sin x + 3 * Real.cos x)^2 = 15/4 - Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_result_l19_1929


namespace NUMINAMATH_CALUDE_pentagon_area_l19_1977

/-- The area of a pentagon with sides 18, 25, 30, 28, and 25 units is 1020 square units. -/
theorem pentagon_area : ℝ := by
  -- Define the pentagon
  let side1 : ℝ := 18
  let side2 : ℝ := 25
  let side3 : ℝ := 30
  let side4 : ℝ := 28
  let side5 : ℝ := 25

  -- Define the area of the pentagon
  let pentagon_area : ℝ := 1020

  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l19_1977


namespace NUMINAMATH_CALUDE_expression_factorization_l19_1999

theorem expression_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + 2*a*b + 2*b*c + 2*c*a) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l19_1999


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l19_1949

theorem necessary_but_not_sufficient_condition :
  ∃ (x : ℝ), (x < 0 ∨ x > 2) ∧ (2*x^2 - 5*x - 3 < 0) ∧
  ∀ (y : ℝ), (2*y^2 - 5*y - 3 ≥ 0) → (y < 0 ∨ y > 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l19_1949


namespace NUMINAMATH_CALUDE_problem_solution_l19_1902

theorem problem_solution (x : ℝ) (h : 3 * x^2 - x = 1) :
  6 * x^3 + 7 * x^2 - 5 * x + 2010 = 2013 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l19_1902


namespace NUMINAMATH_CALUDE_trig_inequality_l19_1947

theorem trig_inequality : 
  let a := Real.sin (31 * π / 180)
  let b := Real.cos (58 * π / 180)
  let c := Real.tan (32 * π / 180)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_trig_inequality_l19_1947


namespace NUMINAMATH_CALUDE_jose_initial_caps_l19_1900

/-- The number of bottle caps Jose gave to Rebecca -/
def given_caps : ℕ := 2

/-- The number of bottle caps Jose has left -/
def remaining_caps : ℕ := 5

/-- The initial number of bottle caps Jose had -/
def initial_caps : ℕ := given_caps + remaining_caps

theorem jose_initial_caps : initial_caps = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_initial_caps_l19_1900


namespace NUMINAMATH_CALUDE_pages_read_later_l19_1972

/-- Given that Jake initially read some pages of a book and then read more later,
    prove that the number of pages he read later is the difference between
    the total pages read and the initial pages read. -/
theorem pages_read_later (initial_pages total_pages pages_read_later : ℕ) :
  initial_pages + pages_read_later = total_pages →
  pages_read_later = total_pages - initial_pages := by
  sorry

#check pages_read_later

end NUMINAMATH_CALUDE_pages_read_later_l19_1972


namespace NUMINAMATH_CALUDE_machines_count_l19_1917

theorem machines_count (x : ℝ) (N : ℕ) (R : ℝ) : 
  N * R = x / 3 →
  45 * R = 5 * x / 10 →
  N = 30 := by
  sorry

end NUMINAMATH_CALUDE_machines_count_l19_1917


namespace NUMINAMATH_CALUDE_girls_share_l19_1936

theorem girls_share (total_amount : ℕ) (total_children : ℕ) (boys_share : ℕ) (num_boys : ℕ)
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : boys_share = 12)
  (h4 : num_boys = 33) :
  (total_amount - num_boys * boys_share) / (total_children - num_boys) = 8 := by
  sorry

end NUMINAMATH_CALUDE_girls_share_l19_1936


namespace NUMINAMATH_CALUDE_lunch_choices_l19_1995

theorem lunch_choices (chicken_types : ℕ) (drink_types : ℕ) 
  (h1 : chicken_types = 3) (h2 : drink_types = 2) : 
  chicken_types * drink_types = 6 := by
sorry

end NUMINAMATH_CALUDE_lunch_choices_l19_1995


namespace NUMINAMATH_CALUDE_afternoon_sales_l19_1941

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 420) :
  sales.afternoon = 280 := by
  sorry

#check afternoon_sales

end NUMINAMATH_CALUDE_afternoon_sales_l19_1941


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l19_1959

/-- Represents the distribution of students across four years -/
structure StudentDistribution :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students in a stratified sample for a given year -/
def stratified_sample_size (total : ℕ) (year_count : ℕ) (sample_size : ℕ) : ℕ :=
  (year_count * sample_size) / total

theorem stratified_sample_theorem (d : StudentDistribution) 
  (h1 : d.first = 400)
  (h2 : d.second = 300)
  (h3 : d.third = 200)
  (h4 : d.fourth = 100)
  (h5 : total_students d = 1000)
  (sample_size : ℕ)
  (h6 : sample_size = 200) :
  stratified_sample_size (total_students d) d.third sample_size = 40 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l19_1959


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l19_1971

/-- The area of a regular hexagon with vertices A(0,0) and C(6,2) is 20√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let s : ℝ := AC / 2
  let hexagon_area : ℝ := 3 * Real.sqrt 3 * s^2 / 2
  hexagon_area = 20 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l19_1971


namespace NUMINAMATH_CALUDE_unique_fibonacci_partition_l19_1908

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def is_fibonacci (n : ℕ) : Prop := ∃ k, fibonacci k = n

def is_partition (A B : Set ℕ) : Prop :=
  (A ∩ B = ∅) ∧ (A ∪ B = Set.univ)

def is_prohibited (S A B : Set ℕ) : Prop :=
  ∀ k l s, (k ∈ A ∧ l ∈ A ∧ s ∈ S) ∨ (k ∈ B ∧ l ∈ B ∧ s ∈ S) → k + l ≠ s

theorem unique_fibonacci_partition :
  ∃! (A B : Set ℕ), is_partition A B ∧ is_prohibited {n | is_fibonacci n} A B :=
sorry

end NUMINAMATH_CALUDE_unique_fibonacci_partition_l19_1908


namespace NUMINAMATH_CALUDE_cubic_root_sum_l19_1963

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 13*a - 8 = 0 → 
  b^3 - 15*b^2 + 13*b - 8 = 0 → 
  c^3 - 15*c^2 + 13*c - 8 = 0 → 
  a / ((1/a) + b*c) + b / ((1/b) + c*a) + c / ((1/c) + a*b) = 199/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l19_1963


namespace NUMINAMATH_CALUDE_most_suitable_sampling_plan_l19_1960

/-- Represents a production line in the factory -/
structure ProductionLine where
  boxes_per_day : ℕ
  deriving Repr

/-- Represents the factory with its production lines -/
structure Factory where
  production_lines : List ProductionLine
  deriving Repr

/-- Represents a sampling plan -/
inductive SamplingPlan
  | RandomOneFromAll
  | LastFromEach
  | RandomOneFromEach
  | AllFromOne
  deriving Repr

/-- Defines what makes a sampling plan suitable -/
def is_suitable_plan (factory : Factory) (plan : SamplingPlan) : Prop :=
  plan = SamplingPlan.RandomOneFromEach

/-- The theorem stating that randomly selecting one box from each production line is the most suitable sampling plan -/
theorem most_suitable_sampling_plan (factory : Factory) 
  (h1 : factory.production_lines.length = 5)
  (h2 : ∀ line ∈ factory.production_lines, line.boxes_per_day = 20) :
  is_suitable_plan factory SamplingPlan.RandomOneFromEach :=
by
  sorry

#check most_suitable_sampling_plan

end NUMINAMATH_CALUDE_most_suitable_sampling_plan_l19_1960


namespace NUMINAMATH_CALUDE_expand_and_simplify_l19_1997

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l19_1997


namespace NUMINAMATH_CALUDE_special_numbers_characterization_l19_1912

/-- Definition of partial numbers for a natural number -/
def partialNumbers (n : ℕ) : Set ℕ :=
  sorry

/-- Predicate to check if all partial numbers of a natural number are prime -/
def allPartialNumbersPrime (n : ℕ) : Prop :=
  ∀ m ∈ partialNumbers n, Nat.Prime m

/-- The set of natural numbers whose partial numbers are all prime -/
def specialNumbers : Set ℕ :=
  {n : ℕ | allPartialNumbersPrime n}

/-- Theorem stating that the set of natural numbers whose partial numbers
    are all prime is exactly {2, 3, 5, 7, 23, 37, 53, 73} -/
theorem special_numbers_characterization :
  specialNumbers = {2, 3, 5, 7, 23, 37, 53, 73} :=
sorry

end NUMINAMATH_CALUDE_special_numbers_characterization_l19_1912


namespace NUMINAMATH_CALUDE_cost_of_potatoes_l19_1990

/-- Proves that the cost of each bag of potatoes is $6 -/
theorem cost_of_potatoes (chicken_price : ℝ) (celery_price : ℝ) (total_cost : ℝ) :
  chicken_price = 3 →
  celery_price = 2 →
  total_cost = 35 →
  (5 * chicken_price + 4 * celery_price + 2 * ((total_cost - 5 * chicken_price - 4 * celery_price) / 2)) = total_cost →
  (total_cost - 5 * chicken_price - 4 * celery_price) / 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_potatoes_l19_1990


namespace NUMINAMATH_CALUDE_penny_problem_l19_1976

theorem penny_problem (initial_pennies : ℕ) (older_pennies : ℕ) (removal_percentage : ℚ) :
  initial_pennies = 200 →
  older_pennies = 30 →
  removal_percentage = 1/5 →
  initial_pennies - older_pennies - Int.floor ((initial_pennies - older_pennies : ℚ) * removal_percentage) = 136 :=
by sorry

end NUMINAMATH_CALUDE_penny_problem_l19_1976


namespace NUMINAMATH_CALUDE_jockey_riding_time_ratio_l19_1937

def max_riding_time : ℝ := 6

theorem jockey_riding_time_ratio :
  ∀ (fractional_time : ℝ),
  (2 * max_riding_time) +  -- Two days of maximum riding
  (2 * 1.5) +              -- Two days of 1.5 hours riding
  (2 * fractional_time * max_riding_time) = 21 →  -- Two days of fractional riding
  fractional_time = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_jockey_riding_time_ratio_l19_1937


namespace NUMINAMATH_CALUDE_divisibility_of_difference_l19_1911

theorem divisibility_of_difference : 43^43 - 17^17 ≡ 0 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_difference_l19_1911


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l19_1978

/-- Represents the dimensions and volume of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ
  volume : ℝ

/-- Theorem: For a cuboid with volume 315 cm³, width 9 cm, and length 7 cm, the height is 5 cm -/
theorem cuboid_height_calculation (c : Cuboid) 
  (h_volume : c.volume = 315)
  (h_width : c.width = 9)
  (h_length : c.length = 7)
  : c.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l19_1978


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l19_1927

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_margin : ℝ)
  (theft_percentage : ℝ)
  (h_profit : profit_margin = 0.1)
  (h_theft : theft_percentage = 0.6)
  (h_initial_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_margin)
  let remaining_value := initial_value * (1 - theft_percentage)
  let remaining_selling_price := selling_price * (1 - theft_percentage)
  let loss := initial_value - remaining_selling_price
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l19_1927


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l19_1956

theorem polynomial_coefficient_B (A C D : ℤ) : 
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (r₁ * r₂ * r₃ * r₄ * r₅ * r₆ : ℤ) = 64 ∧ 
    (r₁ + r₂ + r₃ + r₄ + r₅ + r₆ : ℤ) = 15 ∧ 
    ∀ (z : ℂ), z^6 - 15*z^5 + A*z^4 + (-244)*z^3 + C*z^2 + D*z + 64 = 
      (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l19_1956


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l19_1979

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- State the theorem
theorem odd_function_symmetry (hf_odd : is_odd f) 
  (hf_decreasing : is_monotone_decreasing_on f 1 2) :
  is_monotone_decreasing_on f (-2) (-1) ∧ 
  (∀ x ∈ Set.Icc (-2) (-1), f x ≤ -f 2) ∧
  f (-2) = -f 2 :=
sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l19_1979


namespace NUMINAMATH_CALUDE_ship_distance_difference_l19_1953

/-- The difference in distance traveled between two ships sailing in opposite directions -/
theorem ship_distance_difference (a : ℝ) : 
  let ship_speed := 50
  let time := 2
  let distance_with_current := time * (ship_speed + a)
  let distance_against_current := time * (ship_speed - a)
  distance_with_current - distance_against_current = 4 * a := by
  sorry

end NUMINAMATH_CALUDE_ship_distance_difference_l19_1953


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l19_1916

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l19_1916


namespace NUMINAMATH_CALUDE_marching_band_ratio_l19_1954

theorem marching_band_ratio (total_students : ℕ) (alto_sax_players : ℕ)
  (h_total : total_students = 600)
  (h_alto : alto_sax_players = 4)
  (h_sax : ∃ sax_players : ℕ, 3 * alto_sax_players = sax_players)
  (h_brass : ∃ brass_players : ℕ, 5 * sax_players = brass_players)
  (h_band : ∃ band_students : ℕ, 2 * brass_players = band_students) :
  band_students / total_students = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_marching_band_ratio_l19_1954


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l19_1957

/-- Given the weights of Antoinette and Rupert, prove their weight ratio -/
theorem weight_ratio_proof (A R : ℚ) (k : ℚ) : 
  A = 63 → 
  A + R = 98 → 
  A = k * R - 7 → 
  A / R = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l19_1957


namespace NUMINAMATH_CALUDE_complex_multiplication_l19_1921

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + 2*i) = -2 + i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l19_1921


namespace NUMINAMATH_CALUDE_change_in_average_weight_l19_1932

/-- The change in average weight when replacing a person in a group -/
theorem change_in_average_weight 
  (n : ℕ) 
  (old_weight new_weight : ℝ) 
  (h1 : n = 6) 
  (h2 : old_weight = 69) 
  (h3 : new_weight = 79.8) : 
  (new_weight - old_weight) / n = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_change_in_average_weight_l19_1932


namespace NUMINAMATH_CALUDE_inscribed_polyhedron_radius_gt_three_l19_1924

/-- A polyhedron inscribed in a sphere -/
structure InscribedPolyhedron where
  radius : ℝ
  volume : ℝ
  surface_area : ℝ
  volume_eq_surface_area : volume = surface_area

/-- Theorem: For any polyhedron inscribed in a sphere, if its volume equals its surface area, then the radius of the sphere is greater than 3 -/
theorem inscribed_polyhedron_radius_gt_three (p : InscribedPolyhedron) : p.radius > 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_polyhedron_radius_gt_three_l19_1924


namespace NUMINAMATH_CALUDE_basketball_game_points_l19_1974

/-- Calculate total points for a player given their shot counts -/
def playerPoints (twoPoints threePoints freeThrows : ℕ) : ℕ :=
  2 * twoPoints + 3 * threePoints + freeThrows

/-- Calculate total points for a team given two players' shot counts -/
def teamPoints (p1TwoPoints p1ThreePoints p1FreeThrows
                p2TwoPoints p2ThreePoints p2FreeThrows : ℕ) : ℕ :=
  playerPoints p1TwoPoints p1ThreePoints p1FreeThrows +
  playerPoints p2TwoPoints p2ThreePoints p2FreeThrows

/-- Theorem: The combined points of both teams is 128 -/
theorem basketball_game_points : 
  teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6 = 128 := by
  sorry

#eval teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6

end NUMINAMATH_CALUDE_basketball_game_points_l19_1974


namespace NUMINAMATH_CALUDE_four_students_two_groups_l19_1983

/-- The number of different ways to assign n students to 2 groups -/
def signUpMethods (n : ℕ) : ℕ := 2^n

/-- The problem statement -/
theorem four_students_two_groups : 
  signUpMethods 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_students_two_groups_l19_1983


namespace NUMINAMATH_CALUDE_acrobats_count_correct_l19_1906

/-- The number of acrobats at the farm. -/
def num_acrobats : ℕ := 13

/-- The number of elephants at the farm. -/
def num_elephants : ℕ := sorry

/-- The number of horses at the farm. -/
def num_horses : ℕ := sorry

/-- The total number of legs at the farm. -/
def total_legs : ℕ := 54

/-- The total number of heads at the farm. -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  2 * num_acrobats + 4 * num_elephants + 4 * num_horses = total_legs ∧
  num_acrobats + num_elephants + num_horses = total_heads ∧
  num_acrobats = 13 := by
  sorry


end NUMINAMATH_CALUDE_acrobats_count_correct_l19_1906


namespace NUMINAMATH_CALUDE_max_k_value_l19_1955

open Real

theorem max_k_value (f : ℝ → ℝ) (k : ℤ) : 
  (∀ x > 2, f x = x + x * log x) →
  (∀ x > 2, ↑k * (x - 2) < f x) →
  k ≤ 4 ∧ ∃ x > 2, 4 * (x - 2) < f x :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l19_1955


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l19_1920

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | (x+4)*(x-2) > 0}

theorem intersection_of_A_and_B : A ∩ B = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l19_1920


namespace NUMINAMATH_CALUDE_abc_inequality_l19_1998

theorem abc_inequality (a b c : ℝ) (sum_zero : a + b + c = 0) (product_one : a * b * c = 1) :
  (a * b + b * c + c * a < 0) ∧ (max a (max b c) ≥ Real.rpow 4 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l19_1998


namespace NUMINAMATH_CALUDE_student_count_l19_1958

theorem student_count (N : ℕ) 
  (h1 : N / 5 + N / 4 + N / 2 + 5 = N) : N = 100 := by
  sorry

#check student_count

end NUMINAMATH_CALUDE_student_count_l19_1958


namespace NUMINAMATH_CALUDE_weights_division_l19_1961

theorem weights_division (n : ℕ) (h : n ≥ 3) :
  (∃ (a b c : Finset ℕ), a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a ∪ b ∪ c = Finset.range (n + 1) \ {0} ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)) ↔
  (∃ k : ℕ, (n = 3 * k + 2 ∨ n = 3 * k + 3) ∧ k > 0) :=
by sorry

end NUMINAMATH_CALUDE_weights_division_l19_1961


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l19_1915

theorem sum_of_square_areas (square1_side : ℝ) (square2_side : ℝ) 
  (h1 : square1_side = 11) (h2 : square2_side = 5) : 
  square1_side ^ 2 + square2_side ^ 2 = 146 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l19_1915


namespace NUMINAMATH_CALUDE_solve_equation_binomial_identity_l19_1928

-- Define A_x as the falling factorial
def A (x : ℕ) (n : ℕ) : ℕ := 
  if n ≤ x then
    (x - n + 1).factorial / (x - n).factorial
  else 0

-- Define binomial coefficient
def C (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then
    n.factorial / (k.factorial * (n - k).factorial)
  else 0

theorem solve_equation : ∃ x : ℕ, x > 3 ∧ 3 * A x 3 = 2 * A (x + 1) 2 + 6 * A x 2 ∧ x = 5 := by
  sorry

theorem binomial_identity (n k : ℕ) (h : k ≤ n) : k * C n k = n * C (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_binomial_identity_l19_1928


namespace NUMINAMATH_CALUDE_range_of_a_l19_1992

theorem range_of_a (a : ℝ) : (∀ x > 0, 4 * a > x^2 - x^3) → a > 1/27 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l19_1992


namespace NUMINAMATH_CALUDE_alternating_odd_sum_equals_21_l19_1948

/-- Calculates the sum of the alternating series of odd numbers from 1 to 41 -/
def alternating_odd_sum : ℤ :=
  let n := 20  -- Number of pairs (1-3), (5-7), etc.
  41 - 2 * n

/-- The sum of the series 1-3+5-7+9-11+13-...-39+41 equals 21 -/
theorem alternating_odd_sum_equals_21 : alternating_odd_sum = 21 := by
  sorry

#eval alternating_odd_sum  -- To check the result

end NUMINAMATH_CALUDE_alternating_odd_sum_equals_21_l19_1948


namespace NUMINAMATH_CALUDE_sin_cos_derivative_l19_1945

theorem sin_cos_derivative (x : ℝ) : 
  deriv (λ x => Real.sin x * Real.cos x) x = Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_derivative_l19_1945


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l19_1933

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 5x + 2 = 0 -/
def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 2 = 0

theorem discriminant_of_specific_quadratic :
  discriminant 1 (-5) 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l19_1933


namespace NUMINAMATH_CALUDE_bretschneiders_theorem_l19_1989

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  m_positive : m > 0
  n_positive : n > 0
  A_range : 0 < A ∧ A < π
  C_range : 0 < C ∧ C < π

-- State Bretschneider's theorem
theorem bretschneiders_theorem (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
sorry

end NUMINAMATH_CALUDE_bretschneiders_theorem_l19_1989


namespace NUMINAMATH_CALUDE_acid_dilution_l19_1925

/-- Proves that adding 15 ounces of pure water to 30 ounces of a 30% acid solution yields a 20% acid solution. -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (water_added : ℝ) (final_concentration : ℝ) :
  initial_volume = 30 →
  initial_concentration = 0.3 →
  water_added = 15 →
  final_concentration = 0.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_dilution_l19_1925


namespace NUMINAMATH_CALUDE_candy_distribution_impossibility_l19_1931

theorem candy_distribution_impossibility :
  ¬ ∃ (n : ℕ), 7 * n = 3 * 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_impossibility_l19_1931


namespace NUMINAMATH_CALUDE_optimal_inequality_values_l19_1987

theorem optimal_inequality_values (x : ℝ) (hx : x ∈ Set.Icc 0 1) :
  let a : ℝ := 2
  let b : ℝ := 1/4
  (∀ (a' : ℝ) (b' : ℝ), a' > 0 → b' > 0 →
    (∀ (y : ℝ), y ∈ Set.Icc 0 1 → Real.sqrt (1 - y) + Real.sqrt (1 + y) ≤ 2 - b' * y ^ a') →
    a' ≥ a) ∧
  (∀ (b' : ℝ), b' > b →
    ∃ (y : ℝ), y ∈ Set.Icc 0 1 ∧ Real.sqrt (1 - y) + Real.sqrt (1 + y) > 2 - b' * y ^ a) ∧
  Real.sqrt (1 - x) + Real.sqrt (1 + x) ≤ 2 - b * x ^ a :=
by sorry

end NUMINAMATH_CALUDE_optimal_inequality_values_l19_1987


namespace NUMINAMATH_CALUDE_defeat_points_is_zero_l19_1970

/-- Represents the point system for a football competition -/
structure PointSystem where
  victory_points : ℕ
  draw_points : ℕ
  defeat_points : ℕ

/-- Represents the state of a team's performance -/
structure TeamPerformance where
  total_matches : ℕ
  matches_played : ℕ
  points : ℕ
  victories : ℕ
  draws : ℕ
  defeats : ℕ

/-- Theorem stating that the number of points for a defeat must be 0 -/
theorem defeat_points_is_zero 
  (ps : PointSystem) 
  (tp : TeamPerformance) 
  (h1 : ps.victory_points = 3)
  (h2 : ps.draw_points = 1)
  (h3 : tp.total_matches = 20)
  (h4 : tp.matches_played = 5)
  (h5 : tp.points = 8)
  (h6 : ∀ (future_victories : ℕ), 
        future_victories ≥ 9 → 
        tp.points + future_victories * ps.victory_points + 
        (tp.total_matches - tp.matches_played - future_victories) * ps.defeat_points ≥ 40) :
  ps.defeat_points = 0 := by
sorry

end NUMINAMATH_CALUDE_defeat_points_is_zero_l19_1970


namespace NUMINAMATH_CALUDE_bookshop_inventory_l19_1965

/-- The initial number of books in John's bookshop -/
def initial_books : ℕ := 900

/-- The number of books sold on Monday -/
def monday_sales : ℕ := 75

/-- The number of books sold on Tuesday -/
def tuesday_sales : ℕ := 50

/-- The number of books sold on Wednesday -/
def wednesday_sales : ℕ := 64

/-- The number of books sold on Thursday -/
def thursday_sales : ℕ := 78

/-- The number of books sold on Friday -/
def friday_sales : ℕ := 135

/-- The percentage of books that were not sold -/
def unsold_percentage : ℚ := 55333333333333336 / 100000000000000000

theorem bookshop_inventory :
  initial_books = 900 ∧
  (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales : ℚ) / initial_books = 1 - unsold_percentage :=
by sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l19_1965


namespace NUMINAMATH_CALUDE_prime_power_constraints_l19_1914

theorem prime_power_constraints (a b m n : ℕ) : 
  a > 1 → b > 1 → m > 1 → n > 1 → 
  Nat.Prime (a^n - 1) → Nat.Prime (b^m + 1) → 
  (∃ k : ℕ, m = 2^k) ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_prime_power_constraints_l19_1914


namespace NUMINAMATH_CALUDE_divisibility_condition_l19_1903

theorem divisibility_condition (n : ℕ) (hn : n ≥ 1) :
  (3^(n-1) + 5^(n-1)) ∣ (3^n + 5^n) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l19_1903


namespace NUMINAMATH_CALUDE_point_A_in_transformed_plane_l19_1938

/-- The similarity transformation coefficient -/
def k : ℝ := -2

/-- The original plane equation -/
def plane_a (x y z : ℝ) : Prop := x - 2*y + z + 1 = 0

/-- The transformed plane equation -/
def plane_a' (x y z : ℝ) : Prop := x - 2*y + z - 2 = 0

/-- Point A -/
def point_A : ℝ × ℝ × ℝ := (2, 1, 2)

/-- Theorem stating that point A belongs to the image of plane a -/
theorem point_A_in_transformed_plane : 
  let (x, y, z) := point_A
  plane_a' x y z := by sorry

end NUMINAMATH_CALUDE_point_A_in_transformed_plane_l19_1938
