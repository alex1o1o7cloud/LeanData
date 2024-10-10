import Mathlib

namespace complement_of_67_is_23_l3784_378468

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 90 - angle

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := by sorry

end complement_of_67_is_23_l3784_378468


namespace pickle_theorem_l3784_378458

def pickle_problem (sammy_slices tammy_slices ron_slices : ℕ) : Prop :=
  tammy_slices = 2 * sammy_slices →
  sammy_slices = 15 →
  ron_slices = 24 →
  (tammy_slices - ron_slices : ℚ) / tammy_slices * 100 = 20

theorem pickle_theorem : pickle_problem 15 30 24 := by sorry

end pickle_theorem_l3784_378458


namespace inequality_and_equality_condition_l3784_378410

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (sum_eq_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end inequality_and_equality_condition_l3784_378410


namespace line_l_general_form_l3784_378414

/-- A line passing through point A(-2, 2) with the same y-intercept as y = x + 6 -/
def line_l (x y : ℝ) : Prop :=
  ∃ (m b : ℝ), y = m * x + b ∧ 2 = m * (-2) + b ∧ b = 6

/-- The general form equation of line l -/
def general_form (x y : ℝ) : Prop := 2 * x - y + 6 = 0

/-- Theorem stating that the general form equation of line l is 2x - y + 6 = 0 -/
theorem line_l_general_form : 
  ∀ x y : ℝ, line_l x y ↔ general_form x y :=
sorry

end line_l_general_form_l3784_378414


namespace mara_janet_ratio_l3784_378497

/-- Represents the number of cards each person has -/
structure Cards where
  brenda : ℕ
  janet : ℕ
  mara : ℕ

/-- The conditions of the card problem -/
def card_problem (c : Cards) : Prop :=
  c.janet = c.brenda + 9 ∧
  ∃ k : ℕ, c.mara = k * c.janet ∧
  c.brenda + c.janet + c.mara = 211 ∧
  c.mara = 150 - 40

/-- The theorem stating the ratio of Mara's cards to Janet's cards -/
theorem mara_janet_ratio (c : Cards) :
  card_problem c → c.mara = 2 * c.janet :=
by
  sorry

#check mara_janet_ratio

end mara_janet_ratio_l3784_378497


namespace simplify_and_evaluate_expression_l3784_378402

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  a / (a^2 + 2*a + 1) / (1 - a / (a + 1)) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expression_l3784_378402


namespace johns_calculation_l3784_378435

theorem johns_calculation (x : ℝ) : 
  Real.sqrt x - 20 = 15 → x^2 + 20 = 1500645 := by
  sorry

end johns_calculation_l3784_378435


namespace tangent_y_intercept_l3784_378495

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 11

-- Define the point of tangency
def P : ℝ × ℝ := (1, 12)

-- Theorem statement
theorem tangent_y_intercept :
  let m := (2 : ℝ) -- Slope of the tangent line
  let b := P.2 - m * P.1 -- y-intercept of the tangent line
  b = 10 := by sorry

end tangent_y_intercept_l3784_378495


namespace car_speed_proof_l3784_378416

/-- Proves that a car traveling at speed v km/h takes 20 seconds longer to travel 1 kilometer 
    than it would at 36 km/h if and only if v = 30 km/h. -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 36) * 3600 = 20 ↔ v = 30 :=
by sorry

end car_speed_proof_l3784_378416


namespace rectangle_y_value_l3784_378486

/-- A rectangle with vertices (-2, y), (6, y), (-2, 2), and (6, 2) -/
structure Rectangle where
  y : ℝ
  h_positive : y > 0

/-- The area of the rectangle -/
def area (r : Rectangle) : ℝ := 8 * (r.y - 2)

/-- The perimeter of the rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (8 + (r.y - 2))

/-- Theorem stating that if the area is 64 and the perimeter is 32, then y = 10 -/
theorem rectangle_y_value (r : Rectangle) 
  (h_area : area r = 64) 
  (h_perimeter : perimeter r = 32) : 
  r.y = 10 := by
  sorry

end rectangle_y_value_l3784_378486


namespace marble_jar_problem_l3784_378460

theorem marble_jar_problem (r g b : ℕ) :
  r + g = 5 →
  r + b = 7 →
  g + b = 9 →
  r + g + b = 12 := by
  sorry

end marble_jar_problem_l3784_378460


namespace gcd_91_49_l3784_378482

theorem gcd_91_49 : Nat.gcd 91 49 = 7 := by
  sorry

end gcd_91_49_l3784_378482


namespace tutor_schedule_lcm_l3784_378452

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 3 4) 6) 7 = 84 := by
  sorry

end tutor_schedule_lcm_l3784_378452


namespace matts_stair_climbing_rate_l3784_378436

theorem matts_stair_climbing_rate 
  (M : ℝ)  -- Matt's rate of climbing stairs in steps per minute
  (h1 : M > 0)  -- Matt's rate is positive
  (h2 : ∃ t : ℝ, t > 0 ∧ M * t = 220 ∧ (M + 5) * t = 275)  -- Condition when Matt reaches 220 steps and Tom reaches 275 steps
  : M = 20 := by
  sorry

end matts_stair_climbing_rate_l3784_378436


namespace triangle_law_of_sines_l3784_378456

theorem triangle_law_of_sines (A B : ℝ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : 0 ≤ A) (h4 : A < π) (h5 : 0 ≤ B) (h6 : B < π) :
  a = 3 → b = 5 → Real.sin A = 1/3 → Real.sin B = 5/9 := by sorry

end triangle_law_of_sines_l3784_378456


namespace negation_equivalence_l3784_378478

theorem negation_equivalence (a : ℝ) :
  (¬ ∀ x > 1, 2^x - a > 0) ↔ (∃ x > 1, 2^x - a ≤ 0) := by
  sorry

end negation_equivalence_l3784_378478


namespace expected_value_is_point_seven_l3784_378433

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ  -- Probability of X=1
  h1 : 0 ≤ p ∧ p ≤ 1  -- Probability is between 0 and 1
  h2 : p - (1 - p) = 0.4  -- Given condition

/-- Expected value of a two-point distribution -/
def expectedValue (X : TwoPointDistribution) : ℝ := X.p

theorem expected_value_is_point_seven (X : TwoPointDistribution) :
  expectedValue X = 0.7 := by
  sorry

end expected_value_is_point_seven_l3784_378433


namespace sandbox_perimeter_l3784_378428

/-- The perimeter of a rectangular sandbox with width 5 feet and length twice the width is 30 feet. -/
theorem sandbox_perimeter : 
  ∀ (width length perimeter : ℝ), 
  width = 5 → 
  length = 2 * width → 
  perimeter = 2 * (length + width) → 
  perimeter = 30 := by sorry

end sandbox_perimeter_l3784_378428


namespace rectangle_area_l3784_378494

/-- Calculates the area of a rectangle given its perimeter and width. -/
theorem rectangle_area (perimeter : ℝ) (width : ℝ) (h1 : perimeter = 42) (h2 : width = 8) :
  width * (perimeter / 2 - width) = 104 := by
  sorry

end rectangle_area_l3784_378494


namespace ellipse_fixed_points_l3784_378432

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  center : Point

/-- Checks if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

/-- Calculates the dot product of two vectors -/
def dotProduct (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Theorem about the existence of fixed points Q on the ellipse -/
theorem ellipse_fixed_points (e : Ellipse) (f a : Point) :
  e.a = 2 ∧ e.b = Real.sqrt 3 ∧ e.center = Point.mk 0 0 ∧
  f = Point.mk 1 0 ∧ a = Point.mk (-2) 0 →
  ∃ q1 q2 : Point,
    q1 = Point.mk 1 0 ∧ q2 = Point.mk 7 0 ∧
    ∀ b c m n : Point,
      isOnEllipse e b ∧ isOnEllipse e c ∧
      b ≠ c ∧
      (∃ t : ℝ, b.x = t * b.y + 1 ∧ c.x = t * c.y + 1) ∧
      m.x = 4 ∧ n.x = 4 ∧
      (m.y - a.y) / (m.x - a.x) = (b.y - a.y) / (b.x - a.x) ∧
      (n.y - a.y) / (n.x - a.x) = (c.y - a.y) / (c.x - a.x) →
      dotProduct (Point.mk (q1.x - m.x) (q1.y - m.y)) (Point.mk (q1.x - n.x) (q1.y - n.y)) = 0 ∧
      dotProduct (Point.mk (q2.x - m.x) (q2.y - m.y)) (Point.mk (q2.x - n.x) (q2.y - n.y)) = 0 :=
sorry

end ellipse_fixed_points_l3784_378432


namespace urn_problem_l3784_378481

theorem urn_problem (N : ℕ) : 
  (6 : ℝ) / 10 * 10 / (10 + N) + (4 : ℝ) / 10 * N / (10 + N) = 1 / 2 → N = 10 := by
  sorry

end urn_problem_l3784_378481


namespace hotel_halls_first_wing_l3784_378459

/-- Represents the number of halls on each floor of the first wing -/
def halls_first_wing : ℕ := sorry

/-- Represents the total number of rooms in the hotel -/
def total_rooms : ℕ := 4248

/-- Represents the number of floors in the first wing -/
def floors_first_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the first wing -/
def rooms_per_hall_first_wing : ℕ := 32

/-- Represents the number of floors in the second wing -/
def floors_second_wing : ℕ := 7

/-- Represents the number of halls on each floor of the second wing -/
def halls_second_wing : ℕ := 9

/-- Represents the number of rooms in each hall of the second wing -/
def rooms_per_hall_second_wing : ℕ := 40

theorem hotel_halls_first_wing : 
  halls_first_wing * floors_first_wing * rooms_per_hall_first_wing + 
  floors_second_wing * halls_second_wing * rooms_per_hall_second_wing = total_rooms ∧ 
  halls_first_wing = 6 := by sorry

end hotel_halls_first_wing_l3784_378459


namespace min_value_geometric_sequence_l3784_378496

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) (s : ℝ) :
  b₁ = 2 →
  b₂ = b₁ * s →
  b₃ = b₂ * s →
  ∃ (min : ℝ), min = -9/8 ∧ ∀ (s : ℝ), 3*b₂ + 4*b₃ ≥ min :=
by sorry

end min_value_geometric_sequence_l3784_378496


namespace distinct_numbers_squared_differences_l3784_378442

theorem distinct_numbers_squared_differences (n : ℕ) (a : Fin n → ℝ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) (h_n : n = 10) : 
  {x | ∃ i j, i < j ∧ x = (a j - a i)^2} ≠ 
  {y | ∃ i j, i < j ∧ y = |a j^2 - a i^2|} :=
sorry

end distinct_numbers_squared_differences_l3784_378442


namespace largest_x_value_l3784_378418

theorem largest_x_value : 
  let f (x : ℝ) := (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x
  ∀ x : ℝ, f x = 8 * x - 2 → x ≤ 5/3 :=
by sorry

end largest_x_value_l3784_378418


namespace phone_number_probability_l3784_378430

/-- The probability of randomly dialing the correct seven-digit number -/
theorem phone_number_probability :
  let first_three_options : ℕ := 2  -- 298 or 299
  let last_four_digits : ℕ := 4  -- 0, 2, 6, 7
  let total_combinations := first_three_options * (Nat.factorial last_four_digits)
  (1 : ℚ) / total_combinations = 1 / 48 :=
by sorry

end phone_number_probability_l3784_378430


namespace convex_quadrilaterals_from_circle_points_l3784_378426

theorem convex_quadrilaterals_from_circle_points (n : ℕ) (k : ℕ) :
  n = 12 → k = 4 → Nat.choose n k = 495 := by
  sorry

end convex_quadrilaterals_from_circle_points_l3784_378426


namespace inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l3784_378425

-- Proposition 1
theorem inverse_implies_negation (P : Prop) : 
  (¬P → P) → (P → ¬P) := by sorry

-- Proposition 2
theorem angle_60_iff_arithmetic_sequence (A B C : ℝ) : 
  (B = 60 ∧ A + B + C = 180) ↔ (∃ d : ℝ, A = B - d ∧ C = B + d) := by sorry

-- Proposition 3 (counterexample)
theorem not_necessary_condition_xy : 
  ∃ x y : ℝ, x + y > 3 ∧ x * y > 2 ∧ ¬(x > 1 ∧ y > 2) := by sorry

-- Proposition 4 (counterexample)
theorem not_necessary_condition_ab : 
  ∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2) := by sorry

end inverse_implies_negation_angle_60_iff_arithmetic_sequence_not_necessary_condition_xy_not_necessary_condition_ab_l3784_378425


namespace average_trees_planted_l3784_378438

theorem average_trees_planted (total_students : ℕ) (trees_3 trees_4 trees_5 trees_6 : ℕ) 
  (h1 : total_students = 50)
  (h2 : trees_3 = 20)
  (h3 : trees_4 = 15)
  (h4 : trees_5 = 10)
  (h5 : trees_6 = 5)
  (h6 : trees_3 + trees_4 + trees_5 + trees_6 = total_students) :
  (3 * trees_3 + 4 * trees_4 + 5 * trees_5 + 6 * trees_6) / total_students = 4 := by
  sorry

end average_trees_planted_l3784_378438


namespace systematic_sampling_interval_l3784_378472

theorem systematic_sampling_interval 
  (total_numbers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_numbers = 2014) 
  (h2 : sample_size = 100) :
  (total_numbers - total_numbers % sample_size) / sample_size = 20 :=
by sorry

end systematic_sampling_interval_l3784_378472


namespace oliver_old_cards_l3784_378443

/-- Calculates the number of old baseball cards Oliver had. -/
def old_cards (cards_per_page : ℕ) (new_cards : ℕ) (pages_used : ℕ) : ℕ :=
  cards_per_page * pages_used - new_cards

/-- Theorem stating that Oliver had 10 old cards. -/
theorem oliver_old_cards : old_cards 3 2 4 = 10 := by
  sorry

end oliver_old_cards_l3784_378443


namespace sqrt_equation_solution_l3784_378411

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 6) = Real.sqrt 6 / 2) → x = -3/8 := by
  sorry

end sqrt_equation_solution_l3784_378411


namespace scale_division_l3784_378445

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 90

/-- Represents the length of each part in inches -/
def part_length : ℕ := 18

/-- Theorem stating that the scale divided into equal parts results in 5 parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end scale_division_l3784_378445


namespace eighty_factorial_zeroes_l3784_378427

/-- Count the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ := sorry

theorem eighty_factorial_zeroes :
  trailingZeroes 73 = 16 → trailingZeroes 80 = 19 := by sorry

end eighty_factorial_zeroes_l3784_378427


namespace function_value_at_50_l3784_378437

theorem function_value_at_50 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9*x^2 - 15*x) :
  f 50 = 146 := by
  sorry

end function_value_at_50_l3784_378437


namespace profit_maximizing_volume_l3784_378492

/-- Annual fixed cost in ten thousand dollars -/
def fixed_cost : ℝ := 10

/-- Variable cost per thousand items in ten thousand dollars -/
def variable_cost : ℝ := 2.7

/-- Revenue function in ten thousand dollars -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    10.8 - x^2 / 30
  else if x > 10 then
    108 / x - 1000 / (3 * x^2)
  else
    0

/-- Profit function in ten thousand dollars -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else if x > 10 then
    x * R x - (fixed_cost + variable_cost * x)
  else
    0

/-- Theorem stating that the profit-maximizing production volume is 9 thousand items -/
theorem profit_maximizing_volume :
  ∃ (max_profit : ℝ), W 9 = max_profit ∧ ∀ x, W x ≤ max_profit :=
by sorry

end profit_maximizing_volume_l3784_378492


namespace place_value_comparison_l3784_378491

def number : ℚ := 43597.2468

theorem place_value_comparison : 
  (100 : ℚ) * (number % 1000 - number % 100) / 100 = (number % 0.1 - number % 0.01) / 0.01 := by
  sorry

end place_value_comparison_l3784_378491


namespace euler_family_mean_age_is_68_over_7_l3784_378404

/-- The mean age of the Euler family's children -/
def euler_family_mean_age : ℚ :=
  let ages : List ℕ := [6, 6, 6, 6, 12, 16, 16]
  (ages.sum : ℚ) / ages.length

/-- Theorem stating the mean age of the Euler family's children -/
theorem euler_family_mean_age_is_68_over_7 :
  euler_family_mean_age = 68 / 7 := by
  sorry

end euler_family_mean_age_is_68_over_7_l3784_378404


namespace hyperbola_vertex_distance_l3784_378474

/-- The distance between the vertices of the hyperbola x²/64 - y²/49 = 1 is 16 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := fun x y => x^2/64 - y^2/49 = 1
  ∃ (x₁ x₂ : ℝ), h x₁ 0 ∧ h x₂ 0 ∧ |x₁ - x₂| = 16 :=
sorry

end hyperbola_vertex_distance_l3784_378474


namespace sqrt_mantissa_equality_l3784_378464

theorem sqrt_mantissa_equality (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0) :
  (∃ (k : ℤ), Real.sqrt m - Real.sqrt n = k) → (∃ (a b : ℕ), m = a^2 ∧ n = b^2) :=
sorry

end sqrt_mantissa_equality_l3784_378464


namespace equation_solution_l3784_378400

theorem equation_solution : 
  ∃! y : ℚ, (4 * y - 5) / (5 * y - 15) = 7 / 10 ∧ y = -11 := by
  sorry

end equation_solution_l3784_378400


namespace cos_two_alpha_plus_pi_third_l3784_378424

theorem cos_two_alpha_plus_pi_third (α : ℝ) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end cos_two_alpha_plus_pi_third_l3784_378424


namespace derivative_exp_cos_l3784_378440

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end derivative_exp_cos_l3784_378440


namespace max_product_digits_l3784_378493

theorem max_product_digits : ∀ a b : ℕ, 
  10000 ≤ a ∧ a < 100000 → 1000 ≤ b ∧ b < 10000 → 
  a * b < 1000000000 := by
  sorry

end max_product_digits_l3784_378493


namespace willy_crayon_count_l3784_378465

/-- The number of crayons Lucy has -/
def lucy_crayons : ℕ := 290

/-- The number of additional crayons Willy has compared to Lucy -/
def additional_crayons : ℕ := 1110

/-- The total number of crayons Willy has -/
def willy_crayons : ℕ := lucy_crayons + additional_crayons

theorem willy_crayon_count : willy_crayons = 1400 := by
  sorry

end willy_crayon_count_l3784_378465


namespace cube_root_inequality_l3784_378488

theorem cube_root_inequality (x : ℤ) : 
  (2 : ℝ) < (2 * (x : ℝ)^2)^(1/3) ∧ (2 * (x : ℝ)^2)^(1/3) < 3 ↔ x = 3 ∨ x = -3 := by
  sorry

end cube_root_inequality_l3784_378488


namespace black_and_white_films_count_l3784_378403

-- Define variables
variable (x y : ℚ)
variable (B : ℚ)

-- Define the theorem
theorem black_and_white_films_count :
  (6 * y) / ((y / x) / 100 * B + 6 * y) = 20 / 21 →
  B = 30 * x := by
sorry

end black_and_white_films_count_l3784_378403


namespace count_rational_roots_l3784_378447

/-- The number of different possible rational roots for a polynomial of the form
    12x^4 + b₃x³ + b₂x² + b₁x + 18 = 0 with integer coefficients -/
def num_rational_roots (b₃ b₂ b₁ : ℤ) : ℕ := 28

/-- Theorem stating that the number of different possible rational roots for the given polynomial is 28 -/
theorem count_rational_roots (b₃ b₂ b₁ : ℤ) : 
  num_rational_roots b₃ b₂ b₁ = 28 := by sorry

end count_rational_roots_l3784_378447


namespace tan_alpha_neg_half_implies_expression_neg_third_l3784_378480

theorem tan_alpha_neg_half_implies_expression_neg_third (α : Real) 
  (h : Real.tan α = -1/2) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end tan_alpha_neg_half_implies_expression_neg_third_l3784_378480


namespace min_values_xy_l3784_378470

theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9*x + y - x*y = 0) :
  (∃ (min_sum : ℝ), min_sum = 16 ∧ 
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a + b ≥ min_sum) ∧
    (x + y = min_sum ↔ x = 4 ∧ y = 12)) ∧
  (∃ (min_prod : ℝ), min_prod = 36 ∧
    (∀ (a b : ℝ), a > 0 → b > 0 → 9*a + b - a*b = 0 → a*b ≥ min_prod) ∧
    (x*y = min_prod ↔ x = 2 ∧ y = 18)) :=
by sorry

end min_values_xy_l3784_378470


namespace cube_root_equation_solution_l3784_378453

theorem cube_root_equation_solution :
  ∃ y : ℝ, (5 - 2 / y)^(1/3 : ℝ) = -3 ∧ y = 1/16 := by sorry

end cube_root_equation_solution_l3784_378453


namespace perfect_squares_between_100_and_400_l3784_378448

theorem perfect_squares_between_100_and_400 : 
  (Finset.filter (fun n => 100 < n^2 ∧ n^2 < 400) (Finset.range 20)).card = 9 := by
  sorry

end perfect_squares_between_100_and_400_l3784_378448


namespace computer_price_increase_l3784_378473

theorem computer_price_increase (b : ℝ) : 
  2 * b = 540 → (351 - b) / b * 100 = 30 := by
  sorry

end computer_price_increase_l3784_378473


namespace cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l3784_378454

/-- Represents the movement of a cat and a mouse on a grid -/
def CatMouseMeet (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n

/-- Theorem stating the conditions for the cat and mouse to meet -/
theorem cat_mouse_meet_iff_both_odd (m n : ℕ) :
  CatMouseMeet m n ↔ (m > 1 ∧ n > 1 ∧ Odd m ∧ Odd n) :=
by sorry

/-- Theorem stating the impossibility of meeting when m + n is odd -/
theorem cat_mouse_not_meet_when_sum_odd (m n : ℕ) :
  m > 1 → n > 1 → Odd (m + n) → ¬(CatMouseMeet m n) :=
by sorry

/-- Theorem stating the impossibility of meeting when both m and n are even -/
theorem cat_mouse_not_meet_when_both_even (m n : ℕ) :
  m > 1 → n > 1 → Even m → Even n → ¬(CatMouseMeet m n) :=
by sorry

end cat_mouse_meet_iff_both_odd_cat_mouse_not_meet_when_sum_odd_cat_mouse_not_meet_when_both_even_l3784_378454


namespace couples_satisfy_handshake_equation_l3784_378467

/-- The number of couples at a gathering where each person shakes hands with everyone
    except themselves and their partner, resulting in a total of 31,000 handshakes. -/
def num_couples : ℕ := 125

/-- The total number of handshakes at the gathering. -/
def total_handshakes : ℕ := 31000

/-- Theorem stating that the number of couples satisfies the equation derived from
    the handshake conditions. -/
theorem couples_satisfy_handshake_equation :
  2 * (num_couples * num_couples) - 2 * num_couples = total_handshakes :=
by sorry

end couples_satisfy_handshake_equation_l3784_378467


namespace nine_more_likely_than_ten_l3784_378490

def roll_combinations (sum : ℕ) : Finset (ℕ × ℕ) :=
  (Finset.range 6 ×ˢ Finset.range 6).filter (fun (a, b) => a + b + 2 = sum)

theorem nine_more_likely_than_ten :
  (roll_combinations 9).card > (roll_combinations 10).card := by
  sorry

end nine_more_likely_than_ten_l3784_378490


namespace coin_array_digit_sum_l3784_378422

/-- The sum of the first n natural numbers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

/-- Theorem: For a triangular array of 3003 coins, where the n-th row has n coins,
    the sum of the digits of the total number of rows is 14 -/
theorem coin_array_digit_sum :
  ∃ (n : ℕ), triangular_sum n = 3003 ∧ digit_sum n = 14 := by
  sorry

end coin_array_digit_sum_l3784_378422


namespace aquarium_fish_count_l3784_378446

/-- The number of stingrays in the aquarium -/
def num_stingrays : ℕ := 28

/-- The number of sharks in the aquarium -/
def num_sharks : ℕ := 2 * num_stingrays

/-- The total number of fish (sharks and stingrays) in the aquarium -/
def total_fish : ℕ := num_sharks + num_stingrays

/-- Theorem stating that the total number of fish in the aquarium is 84 -/
theorem aquarium_fish_count : total_fish = 84 := by
  sorry

end aquarium_fish_count_l3784_378446


namespace families_increase_l3784_378441

theorem families_increase (F : ℝ) (h1 : F > 0) : 
  let families_with_computers_1992 := 0.3 * F
  let families_with_computers_1999 := 1.5 * families_with_computers_1992
  let total_families_1999 := families_with_computers_1999 / (3/7)
  total_families_1999 = 1.05 * F :=
by sorry

end families_increase_l3784_378441


namespace initial_speed_satisfies_conditions_initial_speed_is_unique_l3784_378451

/-- The child's initial walking speed in meters per minute -/
def initial_speed : ℝ := 5

/-- The time it takes for the child to walk to school at the initial speed -/
def initial_time : ℝ := 126

/-- The distance from home to school in meters -/
def distance : ℝ := 630

/-- Theorem stating that the initial speed satisfies the given conditions -/
theorem initial_speed_satisfies_conditions :
  distance = initial_speed * initial_time ∧
  distance = 7 * (initial_time - 36) :=
sorry

/-- Theorem proving that the initial speed is unique -/
theorem initial_speed_is_unique (v : ℝ) :
  (∃ t : ℝ, distance = v * t ∧ distance = 7 * (t - 36)) →
  v = initial_speed :=
sorry

end initial_speed_satisfies_conditions_initial_speed_is_unique_l3784_378451


namespace running_track_dimensions_l3784_378431

/-- Given two concentric circles forming a running track, prove the width and radii -/
theorem running_track_dimensions (r₁ r₂ : ℝ) : 
  (2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) →  -- Difference in circumferences
  (2 * Real.pi * r₂ = 40 * Real.pi) →                     -- Circumference of inner circle
  (r₁ - r₂ = 10) ∧                                        -- Width of the track
  (r₂ = 20) ∧                                             -- Radius of inner circle
  (r₁ = 30) :=                                            -- Radius of outer circle
by
  sorry

end running_track_dimensions_l3784_378431


namespace lisa_caffeine_consumption_l3784_378476

/-- Represents the number of beverages Lisa drinks -/
structure BeverageCount where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Represents the caffeine content of each beverage in milligrams -/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ

/-- Calculates the total caffeine consumed based on beverage count and caffeine content -/
def totalCaffeine (count : BeverageCount) (content : CaffeineContent) : ℕ :=
  count.coffee * content.coffee + count.soda * content.soda + count.tea * content.tea

/-- Lisa's daily caffeine goal in milligrams -/
def caffeineGoal : ℕ := 200

/-- Theorem stating Lisa's caffeine consumption and excess -/
theorem lisa_caffeine_consumption 
  (lisas_beverages : BeverageCount)
  (caffeine_per_beverage : CaffeineContent)
  (h1 : lisas_beverages.coffee = 3)
  (h2 : lisas_beverages.soda = 1)
  (h3 : lisas_beverages.tea = 2)
  (h4 : caffeine_per_beverage.coffee = 80)
  (h5 : caffeine_per_beverage.soda = 40)
  (h6 : caffeine_per_beverage.tea = 50) :
  totalCaffeine lisas_beverages caffeine_per_beverage = 380 ∧
  totalCaffeine lisas_beverages caffeine_per_beverage - caffeineGoal = 180 := by
  sorry

end lisa_caffeine_consumption_l3784_378476


namespace total_people_in_program_l3784_378415

theorem total_people_in_program (parents pupils teachers : ℕ) 
  (h1 : parents = 73)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  parents + pupils + teachers = 1541 := by
  sorry

end total_people_in_program_l3784_378415


namespace total_distance_apart_l3784_378450

/-- Represents the speeds of a skater for three hours -/
structure SkaterSpeeds where
  hour1 : ℝ
  hour2 : ℝ
  hour3 : ℝ

/-- Calculates the total distance traveled by a skater given their speeds -/
def totalDistance (speeds : SkaterSpeeds) : ℝ :=
  speeds.hour1 + speeds.hour2 + speeds.hour3

/-- Ann's skating speeds for each hour -/
def annSpeeds : SkaterSpeeds :=
  { hour1 := 6, hour2 := 8, hour3 := 4 }

/-- Glenda's skating speeds for each hour -/
def glendaSpeeds : SkaterSpeeds :=
  { hour1 := 8, hour2 := 5, hour3 := 9 }

/-- Theorem stating the total distance between Ann and Glenda after three hours -/
theorem total_distance_apart : totalDistance annSpeeds + totalDistance glendaSpeeds = 40 := by
  sorry

end total_distance_apart_l3784_378450


namespace male_outnumber_female_l3784_378449

theorem male_outnumber_female (total : ℕ) (male : ℕ) 
  (h1 : total = 928) 
  (h2 : male = 713) : 
  male - (total - male) = 498 := by
  sorry

end male_outnumber_female_l3784_378449


namespace total_amount_paid_l3784_378444

theorem total_amount_paid (total_work : ℚ) (ac_portion : ℚ) (b_payment : ℚ) : 
  total_work = 1 ∧ 
  ac_portion = 19/23 ∧ 
  b_payment = 12 →
  (1 - ac_portion) * (total_work * b_payment) / (1 - ac_portion) = 69 :=
by
  sorry

end total_amount_paid_l3784_378444


namespace quadratic_form_ratio_l3784_378487

theorem quadratic_form_ratio (k : ℝ) : 
  ∃ (d r s : ℝ), 8 * k^2 - 6 * k + 16 = d * (k + r)^2 + s ∧ s / r = -118 / 3 :=
by sorry

end quadratic_form_ratio_l3784_378487


namespace archer_probability_l3784_378419

def prob_not_both_hit (prob_A prob_B : ℚ) : ℚ :=
  1 - (prob_A * prob_B)

theorem archer_probability :
  let prob_A : ℚ := 1/3
  let prob_B : ℚ := 1/2
  prob_not_both_hit prob_A prob_B = 5/6 := by
  sorry

end archer_probability_l3784_378419


namespace bryans_total_amount_l3784_378413

/-- The total amount received from selling precious stones -/
def total_amount (num_stones : ℕ) (price_per_stone : ℕ) : ℕ :=
  num_stones * price_per_stone

/-- Theorem: Bryan's total amount from selling 8 stones at 1785 dollars each is 14280 dollars -/
theorem bryans_total_amount :
  total_amount 8 1785 = 14280 := by
  sorry

end bryans_total_amount_l3784_378413


namespace probability_two_red_chips_l3784_378499

-- Define the number of red and white chips
def total_red : Nat := 4
def total_white : Nat := 2

-- Define the number of chips in each urn
def chips_per_urn : Nat := 3

-- Define a function to calculate the number of ways to distribute chips
def distribute_chips (r w : Nat) : Nat :=
  Nat.choose total_red r * Nat.choose total_white w

-- Define the probability of drawing a red chip from an urn
def prob_red (red_in_urn total_in_urn : Nat) : Rat :=
  red_in_urn / total_in_urn

-- Theorem statement
theorem probability_two_red_chips :
  -- Calculate the total number of ways to distribute chips
  let total_distributions : Nat :=
    distribute_chips 1 2 + distribute_chips 2 1 + distribute_chips 3 0
  
  -- Calculate the probability for each case
  let prob_case1 : Rat := (distribute_chips 1 2 : Rat) / total_distributions *
    prob_red 1 chips_per_urn * prob_red 3 chips_per_urn
  let prob_case2 : Rat := (distribute_chips 2 1 : Rat) / total_distributions *
    prob_red 2 chips_per_urn * prob_red 2 chips_per_urn
  let prob_case3 : Rat := (distribute_chips 3 0 : Rat) / total_distributions *
    prob_red 3 chips_per_urn * prob_red 1 chips_per_urn

  -- The total probability is the sum of all cases
  prob_case1 + prob_case2 + prob_case3 = 2 / 5 := by
  sorry

end probability_two_red_chips_l3784_378499


namespace option_B_is_inductive_reasoning_l3784_378466

-- Define a sequence
def a : ℕ → ℕ
| 1 => 1
| n => 3 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Define inductive reasoning
def is_inductive_reasoning (process : Prop) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (∀ k ≤ n, ∃ (result : Prop), process → result)

-- Theorem statement
theorem option_B_is_inductive_reasoning :
  is_inductive_reasoning (∃ (formula : ℕ → ℕ), ∀ n, S n = formula n) :=
sorry

end option_B_is_inductive_reasoning_l3784_378466


namespace sunzi_problem_l3784_378484

theorem sunzi_problem : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 := by
  sorry

end sunzi_problem_l3784_378484


namespace square_difference_formula_l3784_378457

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 11/17) 
  (diff_eq : a - b = 1/143) : 
  a^2 - b^2 = 11/2431 := by
  sorry

end square_difference_formula_l3784_378457


namespace volume_tetrahedron_C₁LMN_l3784_378471

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (p₁ p₂ p₃ p₄ : Point3D) : ℝ := sorry

/-- Finds the intersection of a line and a plane -/
def lineIntersectPlane (p₁ p₂ : Point3D) (plane : Plane) : Point3D := sorry

/-- Theorem: Volume of tetrahedron C₁LMN in the given cuboid -/
theorem volume_tetrahedron_C₁LMN (cuboid : Cuboid) 
  (h₁ : cuboid.a₁.z - cuboid.a.z = 2)
  (h₂ : cuboid.d.y - cuboid.a.y = 3)
  (h₃ : cuboid.b.x - cuboid.a.x = 251) :
  ∃ (volume : ℝ),
    let plane_A₁BD : Plane := sorry
    let L : Point3D := lineIntersectPlane cuboid.c cuboid.c₁ plane_A₁BD
    let M : Point3D := lineIntersectPlane cuboid.c₁ cuboid.b₁ plane_A₁BD
    let N : Point3D := lineIntersectPlane cuboid.c₁ cuboid.d₁ plane_A₁BD
    volume = tetrahedronVolume cuboid.c₁ L M N := by sorry

end volume_tetrahedron_C₁LMN_l3784_378471


namespace walmart_cards_sent_eq_two_l3784_378409

/-- Represents the gift card scenario --/
structure GiftCardScenario where
  bestBuyCards : ℕ
  bestBuyValue : ℕ
  walmartCards : ℕ
  walmartValue : ℕ
  sentBestBuy : ℕ
  remainingValue : ℕ

/-- Calculates the number of Walmart gift cards sent --/
def walmartsCardsSent (s : GiftCardScenario) : ℕ :=
  let totalInitialValue := s.bestBuyCards * s.bestBuyValue + s.walmartCards * s.walmartValue
  let sentValue := totalInitialValue - s.remainingValue
  let sentWalmartValue := sentValue - s.sentBestBuy * s.bestBuyValue
  sentWalmartValue / s.walmartValue

/-- Theorem stating the number of Walmart gift cards sent --/
theorem walmart_cards_sent_eq_two (s : GiftCardScenario) 
  (h1 : s.bestBuyCards = 6)
  (h2 : s.bestBuyValue = 500)
  (h3 : s.walmartCards = 9)
  (h4 : s.walmartValue = 200)
  (h5 : s.sentBestBuy = 1)
  (h6 : s.remainingValue = 3900) :
  walmartsCardsSent s = 2 := by
  sorry


end walmart_cards_sent_eq_two_l3784_378409


namespace min_cost_plan_l3784_378434

/-- Represents the production plan for student desks and chairs -/
structure ProductionPlan where
  typeA : ℕ  -- Number of type A sets
  typeB : ℕ  -- Number of type B sets

/-- Calculates the total cost of a production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  102 * plan.typeA + 124 * plan.typeB

/-- Checks if a production plan is valid according to the given constraints -/
def isValidPlan (plan : ProductionPlan) : Prop :=
  plan.typeA + plan.typeB = 500 ∧
  2 * plan.typeA + 3 * plan.typeB ≥ 1250 ∧
  5 * plan.typeA + 7 * plan.typeB ≤ 3020

/-- Theorem stating that the minimum total cost is achieved with 250 sets of each type -/
theorem min_cost_plan :
  ∀ (plan : ProductionPlan),
    isValidPlan plan →
    totalCost plan ≥ 56500 ∧
    (totalCost plan = 56500 ↔ plan.typeA = 250 ∧ plan.typeB = 250) := by
  sorry


end min_cost_plan_l3784_378434


namespace peach_problem_l3784_378412

theorem peach_problem (martine benjy gabrielle : ℕ) : 
  martine = 2 * benjy + 6 →
  benjy = gabrielle / 3 →
  martine = 16 →
  gabrielle = 15 := by
sorry

end peach_problem_l3784_378412


namespace find_d_l3784_378420

theorem find_d (a b c d : ℕ+) 
  (eq1 : a ^ 2 = c * (d + 20))
  (eq2 : b ^ 2 = c * (d - 18)) :
  d = 2 := by
  sorry

end find_d_l3784_378420


namespace min_A_over_C_is_zero_l3784_378485

theorem min_A_over_C_is_zero (x : ℝ) (A C : ℝ) (h1 : x ≠ 0) (h2 : A > 0) (h3 : C > 0)
  (h4 : x^2 + 1/x^2 = A) (h5 : x + 1/x = C) :
  ∃ ε > 0, ∀ δ > 0, ∃ A' C', A' > 0 ∧ C' > 0 ∧ A'/C' < δ ∧
  ∃ x' : ℝ, x' ≠ 0 ∧ x'^2 + 1/x'^2 = A' ∧ x' + 1/x' = C' :=
sorry

end min_A_over_C_is_zero_l3784_378485


namespace inequality_proof_l3784_378401

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_prod : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end inequality_proof_l3784_378401


namespace inequality_range_l3784_378463

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 1| > a^2 - 2*a) → 
  -1 < a ∧ a < 3 := by
sorry

end inequality_range_l3784_378463


namespace not_divisible_by_361_l3784_378455

theorem not_divisible_by_361 (k : ℕ) : ¬(361 ∣ k^2 + 11*k - 22) := by
  sorry

end not_divisible_by_361_l3784_378455


namespace complement_of_A_l3784_378408

-- Define the set A
def A : Set ℝ := {x | x^2 + 3*x ≥ 0} ∪ {x | 2*x > 1}

-- State the theorem
theorem complement_of_A : 
  {x : ℝ | x ∉ A} = {x : ℝ | -3 < x ∧ x < 0} := by sorry

end complement_of_A_l3784_378408


namespace pool_depth_l3784_378483

/-- Represents the dimensions and properties of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (chlorine_coverage : ℝ)
  (chlorine_cost : ℝ)
  (money_spent : ℝ)

/-- Theorem stating the depth of the pool given the conditions -/
theorem pool_depth (p : Pool) 
  (h1 : p.length = 10)
  (h2 : p.width = 8)
  (h3 : p.chlorine_coverage = 120)
  (h4 : p.chlorine_cost = 3)
  (h5 : p.money_spent = 12) :
  p.depth = 6 := by
  sorry

#check pool_depth

end pool_depth_l3784_378483


namespace cube_root_of_negative_one_eighth_l3784_378498

theorem cube_root_of_negative_one_eighth :
  ∃ y : ℝ, y^3 = (-1/8 : ℝ) ∧ y = (-1/2 : ℝ) := by
  sorry

end cube_root_of_negative_one_eighth_l3784_378498


namespace jason_coin_difference_l3784_378477

/-- Given that Jayden received 300 coins, the total coins given to both boys is 660,
    and Jason received more coins than Jayden, prove that Jason received 60 more coins than Jayden. -/
theorem jason_coin_difference (jayden_coins : ℕ) (total_coins : ℕ) (jason_coins : ℕ)
  (h1 : jayden_coins = 300)
  (h2 : total_coins = 660)
  (h3 : jason_coins + jayden_coins = total_coins)
  (h4 : jason_coins > jayden_coins) :
  jason_coins - jayden_coins = 60 := by
  sorry

end jason_coin_difference_l3784_378477


namespace choir_composition_l3784_378407

theorem choir_composition (initial_total : ℕ) (initial_blonde : ℕ) (added_blonde : ℕ) :
  initial_total = 80 →
  initial_blonde = 30 →
  added_blonde = 10 →
  initial_total - initial_blonde + added_blonde = 50 :=
by sorry

end choir_composition_l3784_378407


namespace geometric_series_relation_l3784_378475

/-- Given two infinite geometric series with specified terms, prove that if the sum of the second series
    is three times the sum of the first series, then n = 4. -/
theorem geometric_series_relation (n : ℝ) : 
  let first_series_term1 : ℝ := 15
  let first_series_term2 : ℝ := 9
  let second_series_term1 : ℝ := 15
  let second_series_term2 : ℝ := 9 + n
  let first_series_sum : ℝ := first_series_term1 / (1 - (first_series_term2 / first_series_term1))
  let second_series_sum : ℝ := second_series_term1 / (1 - (second_series_term2 / second_series_term1))
  second_series_sum = 3 * first_series_sum → n = 4 :=
by
  sorry


end geometric_series_relation_l3784_378475


namespace upper_line_formula_l3784_378417

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The sequence of numbers in the lower line -/
def x : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n + 2) => x (n + 1) + x n + 1

/-- The sequence of numbers in the upper line -/
def a (n : ℕ) : ℕ := x (n + 1) - 1

theorem upper_line_formula (n : ℕ) : a n = fib (n + 3) - 2 := by
  sorry

#check upper_line_formula

end upper_line_formula_l3784_378417


namespace g_of_5_l3784_378462

def g (x : ℝ) : ℝ := x^2 - 2*x

theorem g_of_5 : g 5 = 15 := by sorry

end g_of_5_l3784_378462


namespace area_of_right_triangle_with_inscribed_circle_l3784_378423

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of the leg divided by the point of tangency -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The leg a is divided into segments of 6 and 10 by the point of tangency -/
  h_a : a = 16
  /-- The radius of the inscribed circle is 6 -/
  h_r : r = 6
  /-- The semi-perimeter of the triangle -/
  p : ℝ
  /-- Relation between semi-perimeter and leg b -/
  h_p : p = b + 10

/-- The area of the right triangle with an inscribed circle is 240 -/
theorem area_of_right_triangle_with_inscribed_circle 
  (t : RightTriangleWithInscribedCircle) : t.a * t.b / 2 = 240 := by
  sorry

end area_of_right_triangle_with_inscribed_circle_l3784_378423


namespace square_of_binomial_identity_l3784_378421

/-- The square of a binomial formula -/
def square_of_binomial (x y : ℝ) : ℝ := x^2 + 2*x*y + y^2

/-- Expression A -/
def expr_A (a b : ℝ) : ℝ := (a + b) * (a + b)

/-- Expression B -/
def expr_B (x y : ℝ) : ℝ := (x + 2*y) * (x - 2*y)

/-- Expression C -/
def expr_C (a : ℝ) : ℝ := (a - 3) * (3 - a)

/-- Expression D -/
def expr_D (a b : ℝ) : ℝ := (2*a - b) * (-2*a + 3*b)

theorem square_of_binomial_identity (a b : ℝ) :
  expr_A a b = square_of_binomial a b ∧
  ∃ x y, expr_B x y ≠ square_of_binomial x y ∧
  ∃ a, expr_C a ≠ square_of_binomial (a - 3) 3 ∧
  ∃ a b, expr_D a b ≠ square_of_binomial (2*a - b) (-2*a + 3*b) :=
by sorry

end square_of_binomial_identity_l3784_378421


namespace no_such_function_l3784_378469

theorem no_such_function : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end no_such_function_l3784_378469


namespace remainder_problem_l3784_378406

theorem remainder_problem (n : ℕ) : 
  n % 6 = 4 ∧ n / 6 = 124 → (n + 24) % 8 = 4 := by
sorry

end remainder_problem_l3784_378406


namespace scooter_gain_percent_l3784_378439

/-- Calculate the gain percent on a scooter sale -/
theorem scooter_gain_percent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 900)
  (h2 : repair_cost = 300)
  (h3 : selling_price = 1500) :
  (selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost) * 100 = 25 := by
  sorry

end scooter_gain_percent_l3784_378439


namespace total_rainfall_2004_l3784_378479

/-- The average monthly rainfall in Mathborough in 2003 (in mm) -/
def avg_rainfall_2003 : ℝ := 41.5

/-- The increase in average monthly rainfall from 2003 to 2004 (in mm) -/
def rainfall_increase : ℝ := 2

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- Theorem: The total amount of rain that fell in Mathborough in 2004 was 522 mm -/
theorem total_rainfall_2004 : 
  (avg_rainfall_2003 + rainfall_increase) * months_in_year = 522 := by
  sorry

end total_rainfall_2004_l3784_378479


namespace binomial_expansion_max_term_max_term_for_sqrt11_expansion_l3784_378429

theorem binomial_expansion_max_term (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_for_sqrt11_expansion :
  let n : ℕ := 208
  let x : ℝ := Real.sqrt 11
  ∃ k : ℕ, k = 160 ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

end binomial_expansion_max_term_max_term_for_sqrt11_expansion_l3784_378429


namespace hexagon_wire_remainder_l3784_378461

/-- Calculates the remaining wire length after creating a regular hexagon -/
def remaining_wire_length (total_wire : ℝ) (hexagon_side : ℝ) : ℝ :=
  total_wire - 6 * hexagon_side

/-- Theorem: Given a 50 cm wire and a regular hexagon with 8 cm sides, 2 cm of wire remains -/
theorem hexagon_wire_remainder :
  remaining_wire_length 50 8 = 2 := by
  sorry

end hexagon_wire_remainder_l3784_378461


namespace square_area_on_parallel_lines_l3784_378405

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Checks if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line) : Prop := sorry

/-- Calculates the perpendicular distance between two lines -/
def perpendicular_distance (l1 l2 : Line) : ℝ := sorry

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop := sorry

/-- Calculates the area of a square -/
def square_area (s : Square) : ℝ := sorry

/-- The main theorem -/
theorem square_area_on_parallel_lines 
  (l1 l2 l3 : Line) 
  (s : Square) :
  are_parallel l1 l2 l3 →
  perpendicular_distance l1 l2 = 3 →
  perpendicular_distance l2 l3 = 3 →
  point_on_line s.a l1 →
  point_on_line s.b l3 →
  point_on_line s.c l2 →
  square_area s = 45 := by
  sorry

end square_area_on_parallel_lines_l3784_378405


namespace min_value_sum_reciprocals_min_value_achievable_l3784_378489

theorem min_value_sum_reciprocals (x y z : ℕ+) (h : x + y + z = 12) :
  (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) ≥ 9 / 2 :=
by sorry

theorem min_value_achievable :
  ∃ (x y z : ℕ+), x + y + z = 12 ∧
    (x + y + z : ℚ) * (1 / (x + y : ℚ) + 1 / (x + z : ℚ) + 1 / (y + z : ℚ)) = 9 / 2 :=
by sorry

end min_value_sum_reciprocals_min_value_achievable_l3784_378489
