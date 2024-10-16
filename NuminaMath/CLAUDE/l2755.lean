import Mathlib

namespace NUMINAMATH_CALUDE_given_terms_are_like_l2755_275530

/-- Two algebraic terms are like terms if they have the same variables with the same exponents. -/
def are_like_terms (term1 term2 : String) : Prop := sorry

/-- The first term in the pair. -/
def term1 : String := "-m^2n^3"

/-- The second term in the pair. -/
def term2 : String := "-3n^3m^2"

/-- Theorem stating that the given terms are like terms. -/
theorem given_terms_are_like : are_like_terms term1 term2 := by sorry

end NUMINAMATH_CALUDE_given_terms_are_like_l2755_275530


namespace NUMINAMATH_CALUDE_john_quilt_cost_l2755_275509

def quilt_cost (length width cost_per_sqft discount_rate tax_rate : ℝ) : ℝ :=
  let area := length * width
  let initial_cost := area * cost_per_sqft
  let discounted_cost := initial_cost * (1 - discount_rate)
  let total_cost := discounted_cost * (1 + tax_rate)
  total_cost

theorem john_quilt_cost :
  quilt_cost 12 15 70 0.1 0.05 = 11907 := by
  sorry

end NUMINAMATH_CALUDE_john_quilt_cost_l2755_275509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2755_275588

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = (n * (a 1 + a n)) / 2) →  -- Definition of sum for arithmetic sequence
  (∀ n, ∃ d, a (n + 1) - a n = d) →     -- Definition of arithmetic sequence
  a 1 = -2016 →                         -- Given condition
  (S 2015) / 2015 - (S 2012) / 2012 = 3 →  -- Given condition
  S 2016 = -2016 :=                     -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2755_275588


namespace NUMINAMATH_CALUDE_cubic_fraction_equality_l2755_275540

theorem cubic_fraction_equality : 
  let a : ℝ := 5
  let b : ℝ := 4
  (a^3 + b^3) / (a^2 - a*b + b^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equality_l2755_275540


namespace NUMINAMATH_CALUDE_expected_defects_l2755_275545

theorem expected_defects (N D n : ℕ) (h1 : N = 15000) (h2 : D = 1000) (h3 : n = 150) :
  (n : ℚ) * (D : ℚ) / (N : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_defects_l2755_275545


namespace NUMINAMATH_CALUDE_prime_cube_difference_equation_l2755_275501

theorem prime_cube_difference_equation :
  ∃! (p q r : ℕ), 
    Prime p ∧ Prime q ∧ Prime r ∧
    p^3 - q^3 = 11*r ∧
    p = 13 ∧ q = 2 ∧ r = 199 := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_equation_l2755_275501


namespace NUMINAMATH_CALUDE_cost_of_five_basketballs_l2755_275504

/-- The cost of buying multiple basketballs -/
def cost_of_basketballs (price_per_ball : ℝ) (num_balls : ℕ) : ℝ :=
  price_per_ball * num_balls

/-- Theorem: The cost of 5 basketballs is 5a yuan, given that one basketball costs a yuan -/
theorem cost_of_five_basketballs (a : ℝ) :
  cost_of_basketballs a 5 = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_cost_of_five_basketballs_l2755_275504


namespace NUMINAMATH_CALUDE_triangle_angle_b_triangle_sides_l2755_275569

/-- Theorem: In an acute triangle ABC, if b*cos(C) + √3*b*sin(C) = a + c, then B = π/3 -/
theorem triangle_angle_b (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b * Real.cos C + Real.sqrt 3 * b * Real.sin C = a + c →
  B = π/3 := by
  sorry

/-- Corollary: If b = 2 and the area of triangle ABC is √3, then a = 2 and c = 2 -/
theorem triangle_sides (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  b = 2 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →
  a = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_b_triangle_sides_l2755_275569


namespace NUMINAMATH_CALUDE_opposite_signs_and_sum_negative_l2755_275567

theorem opposite_signs_and_sum_negative (a b : ℚ) 
  (h1 : a * b < 0) 
  (h2 : a + b < 0) : 
  a > 0 ∧ b < 0 ∧ |b| > a := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_and_sum_negative_l2755_275567


namespace NUMINAMATH_CALUDE_mung_bean_germination_l2755_275542

theorem mung_bean_germination 
  (germination_rate : ℝ) 
  (total_seeds : ℝ) 
  (h1 : germination_rate = 0.971) 
  (h2 : total_seeds = 1000) : 
  total_seeds * (1 - germination_rate) = 29 := by
sorry

end NUMINAMATH_CALUDE_mung_bean_germination_l2755_275542


namespace NUMINAMATH_CALUDE_joan_apple_count_l2755_275596

/-- Theorem: Given Joan picked 43 apples initially and Melanie gave her 27 more apples, Joan now has 70 apples. -/
theorem joan_apple_count (initial_apples : ℕ) (given_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 43 → given_apples = 27 → total_apples = initial_apples + given_apples → total_apples = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_apple_count_l2755_275596


namespace NUMINAMATH_CALUDE_expression_equality_l2755_275503

theorem expression_equality (x : ℝ) (h : 3 * x^2 - 6 * x + 4 = 7) : 
  x^2 - 2 * x + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2755_275503


namespace NUMINAMATH_CALUDE_find_number_given_hcf_lcm_l2755_275599

/-- Given two positive integers with specific HCF and LCM, prove that one is 24 if the other is 169 -/
theorem find_number_given_hcf_lcm (A B : ℕ+) : 
  (Nat.gcd A B = 13) →
  (Nat.lcm A B = 312) →
  (B = 169) →
  A = 24 := by
sorry

end NUMINAMATH_CALUDE_find_number_given_hcf_lcm_l2755_275599


namespace NUMINAMATH_CALUDE_function_solution_set_l2755_275572

-- Define the function f
def f (x a : ℝ) : ℝ := |2 * x - a| + a

-- State the theorem
theorem function_solution_set (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_solution_set_l2755_275572


namespace NUMINAMATH_CALUDE_compare_squares_and_products_l2755_275535

theorem compare_squares_and_products 
  (x a b : ℝ) 
  (h1 : x < a) 
  (h2 : a < b) 
  (h3 : b < 0) : 
  x^2 > a * x ∧ 
  a * x > b * x ∧ 
  x^2 > a^2 ∧ 
  a^2 > b^2 := by
sorry

end NUMINAMATH_CALUDE_compare_squares_and_products_l2755_275535


namespace NUMINAMATH_CALUDE_initial_men_correct_l2755_275536

/-- The initial number of men working -/
def initial_men : ℕ := 450

/-- The number of hours worked per day initially -/
def initial_hours : ℕ := 8

/-- The depth dug initially in meters -/
def initial_depth : ℕ := 40

/-- The new depth to be dug in meters -/
def new_depth : ℕ := 50

/-- The new number of hours worked per day -/
def new_hours : ℕ := 6

/-- The number of extra men needed for the new task -/
def extra_men : ℕ := 30

/-- Theorem stating that the initial number of men is correct -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l2755_275536


namespace NUMINAMATH_CALUDE_semicircle_perimeter_l2755_275554

/-- The perimeter of a semicircle with radius 6.3 cm is equal to π * 6.3 + 2 * 6.3 cm -/
theorem semicircle_perimeter :
  let r : ℝ := 6.3
  (π * r + 2 * r) = (π * 6.3 + 2 * 6.3) :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_l2755_275554


namespace NUMINAMATH_CALUDE_advertising_customers_l2755_275598

/-- Proves that the number of customers brought to a site by advertising is 100,
    given the cost of advertising, purchase rate, item cost, and profit. -/
theorem advertising_customers (ad_cost profit item_cost : ℝ) (purchase_rate : ℝ) :
  ad_cost = 1000 →
  profit = 1000 →
  item_cost = 25 →
  purchase_rate = 0.8 →
  ∃ (num_customers : ℕ), 
    (↑num_customers : ℝ) * purchase_rate * item_cost = ad_cost + profit ∧
    num_customers = 100 :=
by sorry

end NUMINAMATH_CALUDE_advertising_customers_l2755_275598


namespace NUMINAMATH_CALUDE_probability_at_least_two_succeed_l2755_275548

theorem probability_at_least_two_succeed (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 1/2) (h₂ : p₂ = 1/4) (h₃ : p₃ = 1/5) : 
  p₁ * p₂ * (1 - p₃) + p₁ * (1 - p₂) * p₃ + (1 - p₁) * p₂ * p₃ + p₁ * p₂ * p₃ = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_succeed_l2755_275548


namespace NUMINAMATH_CALUDE_marble_probability_l2755_275595

theorem marble_probability (box1 box2 : Nat) : 
  box1 + box2 = 36 →
  (box1 * box2 : Rat) = 36 →
  (∃ black1 black2 : Nat, 
    black1 ≤ box1 ∧ 
    black2 ≤ box2 ∧ 
    (black1 * black2 : Rat) / (box1 * box2) = 25 / 36) →
  (∃ white1 white2 : Nat,
    white1 = box1 - black1 ∧
    white2 = box2 - black2 ∧
    (white1 * white2 : Rat) / (box1 * box2) = 169 / 324) :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l2755_275595


namespace NUMINAMATH_CALUDE_skittles_taken_away_l2755_275574

def initial_skittles : ℕ := 25
def remaining_skittles : ℕ := 18

theorem skittles_taken_away : initial_skittles - remaining_skittles = 7 := by
  sorry

end NUMINAMATH_CALUDE_skittles_taken_away_l2755_275574


namespace NUMINAMATH_CALUDE_same_floor_prob_is_one_fifth_l2755_275531

/-- A hotel with 6 rooms distributed across 3 floors -/
structure Hotel :=
  (total_rooms : ℕ)
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (h1 : total_rooms = 6)
  (h2 : floors = 3)
  (h3 : rooms_per_floor = 2)
  (h4 : total_rooms = floors * rooms_per_floor)

/-- The probability of two people choosing rooms on the same floor -/
def same_floor_probability (h : Hotel) : ℚ :=
  (h.floors * (h.rooms_per_floor * (h.rooms_per_floor - 1))) / (h.total_rooms * (h.total_rooms - 1))

theorem same_floor_prob_is_one_fifth (h : Hotel) : same_floor_probability h = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_same_floor_prob_is_one_fifth_l2755_275531


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2755_275559

theorem rectangle_perimeter (L B : ℝ) 
  (h1 : L - B = 23) 
  (h2 : L * B = 3650) : 
  2 * L + 2 * B = 338 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2755_275559


namespace NUMINAMATH_CALUDE_unique_rectangle_arrangement_l2755_275519

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.width + r.height)

/-- Checks if two rectangles have equal perimeters -/
def equalPerimeters (r1 r2 : Rectangle) : Prop := r1.perimeter = r2.perimeter

/-- Checks if the total area of two rectangles is 81 -/
def totalAreaIs81 (r1 r2 : Rectangle) : Prop := r1.area + r2.area = 81

/-- The main theorem stating that the only way to arrange 81 unit squares into two rectangles
    with equal perimeters is to form rectangles with dimensions 3 × 11 and 6 × 8 -/
theorem unique_rectangle_arrangement :
  ∀ r1 r2 : Rectangle,
    equalPerimeters r1 r2 → totalAreaIs81 r1 r2 →
    ((r1.width = 3 ∧ r1.height = 11) ∧ (r2.width = 6 ∧ r2.height = 8)) ∨
    ((r1.width = 6 ∧ r1.height = 8) ∧ (r2.width = 3 ∧ r2.height = 11)) := by
  sorry

end NUMINAMATH_CALUDE_unique_rectangle_arrangement_l2755_275519


namespace NUMINAMATH_CALUDE_jakes_weight_l2755_275594

theorem jakes_weight (jake_weight sister_weight : ℕ) : 
  jake_weight - 33 = 2 * sister_weight →
  jake_weight + sister_weight = 153 →
  jake_weight = 113 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l2755_275594


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2755_275564

def is_valid_n (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (10 * (n % 10) + n / 10 - 5 = 2 * n)

theorem smallest_valid_n :
  is_valid_n 13 ∧ ∀ m, is_valid_n m → m ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2755_275564


namespace NUMINAMATH_CALUDE_max_area_is_1406_l2755_275513

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  width : ℕ
  length : ℕ
  perimeter_constraint : width * 2 + length * 2 = 150

/-- The area of a rectangular garden. -/
def garden_area (g : RectangularGarden) : ℕ :=
  g.width * g.length

/-- The maximum area of a rectangular garden with a perimeter of 150 feet. -/
def max_garden_area : ℕ := 1406

/-- Theorem stating that the maximum area of a rectangular garden with
    a perimeter of 150 feet and integer side lengths is 1406 square feet. -/
theorem max_area_is_1406 :
  ∀ g : RectangularGarden, garden_area g ≤ max_garden_area :=
by sorry

end NUMINAMATH_CALUDE_max_area_is_1406_l2755_275513


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l2755_275506

theorem difference_of_squares_special_case : (500 : ℤ) * 500 - 499 * 501 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l2755_275506


namespace NUMINAMATH_CALUDE_points_bound_l2755_275586

/-- A structure representing a set of points on a line with colored circles -/
structure ColoredCircles where
  k : ℕ  -- Number of points
  n : ℕ  -- Number of colors
  points : Fin k → ℝ  -- Function mapping point indices to their positions on the line
  circle_color : Fin k → Fin k → Fin n  -- Function assigning a color to each circle

/-- Predicate to check if two circles are mutually tangent -/
def mutually_tangent (cc : ColoredCircles) (i j m l : Fin cc.k) : Prop :=
  (cc.points i < cc.points m) ∧ (cc.points m < cc.points j) ∧ (cc.points j < cc.points l)

/-- Axiom: Mutually tangent circles have different colors -/
axiom different_colors (cc : ColoredCircles) :
  ∀ (i j m l : Fin cc.k), mutually_tangent cc i j m l →
    cc.circle_color i j ≠ cc.circle_color m l

/-- Theorem: The number of points is at most 2^n -/
theorem points_bound (cc : ColoredCircles) : cc.k ≤ 2^cc.n := by
  sorry

end NUMINAMATH_CALUDE_points_bound_l2755_275586


namespace NUMINAMATH_CALUDE_inequality_proof_l2755_275597

theorem inequality_proof (n : ℕ) (hn : n > 1) : 
  let a : ℚ := 1 / n
  (a^2 : ℚ) < a ∧ a < (1 : ℚ) / a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2755_275597


namespace NUMINAMATH_CALUDE_james_weekly_beats_l2755_275583

/-- The number of beats heard in a week given a music speed and daily listening time -/
def beats_per_week (beats_per_minute : ℕ) (hours_per_day : ℕ) : ℕ :=
  beats_per_minute * (hours_per_day * 60) * 7

/-- Theorem: James hears 168,000 beats per week -/
theorem james_weekly_beats :
  beats_per_week 200 2 = 168000 := by
  sorry

end NUMINAMATH_CALUDE_james_weekly_beats_l2755_275583


namespace NUMINAMATH_CALUDE_percentage_calculation_l2755_275518

theorem percentage_calculation (p : ℝ) : 
  0.25 * 900 = p / 100 * 1600 - 15 → p = 1500 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2755_275518


namespace NUMINAMATH_CALUDE_clock_strike_time_l2755_275593

/-- If a clock takes 42 seconds to strike 7 times, it takes 60 seconds to strike 10 times. -/
theorem clock_strike_time (strike_time : ℕ → ℝ) 
  (h : strike_time 7 = 42) : strike_time 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clock_strike_time_l2755_275593


namespace NUMINAMATH_CALUDE_equal_intercept_line_proof_l2755_275552

/-- A line with equal intercepts on both axes passing through point (3,2) -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem equal_intercept_line_proof :
  -- The line passes through point (3,2)
  equal_intercept_line 3 2 ∧
  -- The line has equal intercepts on both axes
  ∃ a : ℝ, a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a :=
by
  sorry

#check equal_intercept_line_proof

end NUMINAMATH_CALUDE_equal_intercept_line_proof_l2755_275552


namespace NUMINAMATH_CALUDE_triangle_theorem_l2755_275549

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = t.a * t.b

def angleBisectorIntersection (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This is a placeholder for the angle bisector condition
  True

def cdLength (t : Triangle) (D : ℝ × ℝ) : Prop :=
  -- This represents CD = 2
  True

-- Theorem statement
theorem triangle_theorem (t : Triangle) (D : ℝ × ℝ) :
  satisfiesCondition t →
  angleBisectorIntersection t D →
  cdLength t D →
  (t.C = 2 * Real.pi / 3) ∧
  (∃ (min : ℝ), min = 6 + 4 * Real.sqrt 2 ∧
    ∀ (a b : ℝ), a > 0 ∧ b > 0 → 2 * a + b ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2755_275549


namespace NUMINAMATH_CALUDE_impossible_assembly_l2755_275563

theorem impossible_assembly (p q r : ℕ) : ¬∃ (x y z : ℕ),
  (2 * p + 2 * r + 2 = 2 * x) ∧
  (2 * p + q + 1 = 2 * x + y) ∧
  (q + r = y + z) :=
by sorry

end NUMINAMATH_CALUDE_impossible_assembly_l2755_275563


namespace NUMINAMATH_CALUDE_parallel_line_distance_l2755_275561

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- The lengths of the three chords formed by the intersection -/
  chord_lengths : Fin 3 → ℝ
  /-- The chords are formed by the intersection of the circle and parallel lines -/
  chord_formation : chord_lengths 0 = 30 ∧ chord_lengths 1 = 40 ∧ chord_lengths 2 = 30

/-- The theorem stating the distance between adjacent parallel lines -/
theorem parallel_line_distance (c : CircleWithParallelLines) : 
  c.line_distance = 2 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_parallel_line_distance_l2755_275561


namespace NUMINAMATH_CALUDE_smallest_angle_BFE_l2755_275520

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the measure of an angle
def angle_measure (p q r : Point) : ℝ := sorry

-- State the theorem
theorem smallest_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∃ (n : ℕ), 
    (∀ m : ℕ, m < n → ¬(∃ ABC : Triangle, angle_measure ABC.B F E = m)) ∧
    (∃ ABC : Triangle, angle_measure ABC.B F E = n) ∧
    n = 113 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_BFE_l2755_275520


namespace NUMINAMATH_CALUDE_pie_machine_completion_time_l2755_275544

-- Define the start time
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes since midnight

-- Define the time when half the job is completed
def half_job_time : Nat := 12 * 60 + 30  -- 12:30 PM in minutes since midnight

-- Define the duration of the maintenance break
def break_duration : Nat := 45

-- Function to calculate the completion time
def completion_time (start : Nat) (half_job : Nat) (break_dur : Nat) : Nat :=
  let working_time := 2 * (half_job - start)
  start + working_time + break_dur

-- Theorem statement
theorem pie_machine_completion_time :
  completion_time start_time half_job_time break_duration = 16 * 60 + 45 := by
  sorry

end NUMINAMATH_CALUDE_pie_machine_completion_time_l2755_275544


namespace NUMINAMATH_CALUDE_michaels_investment_l2755_275590

theorem michaels_investment (total_investment : ℝ) (thrifty_rate : ℝ) (rich_rate : ℝ) 
  (years : ℕ) (final_amount : ℝ) (thrifty_investment : ℝ) :
  total_investment = 1500 →
  thrifty_rate = 0.04 →
  rich_rate = 0.06 →
  years = 3 →
  final_amount = 1738.84 →
  thrifty_investment * (1 + thrifty_rate) ^ years + 
    (total_investment - thrifty_investment) * (1 + rich_rate) ^ years = final_amount →
  thrifty_investment = 720.84 := by
sorry

end NUMINAMATH_CALUDE_michaels_investment_l2755_275590


namespace NUMINAMATH_CALUDE_system_solution_l2755_275517

theorem system_solution : 
  ∀ x y : ℝ, (x^3 + 3*x*y^2 = 49 ∧ x^2 + 8*x*y + y^2 = 8*y + 17*x) → 
  ((x = 1 ∧ y = 4) ∨ (x = 1 ∧ y = -4)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2755_275517


namespace NUMINAMATH_CALUDE_max_value_inequality_l2755_275565

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x^2 + 1/y^2) * (x^2 + 1/y^2 - 100) + (y^2 + 1/x^2) * (y^2 + 1/x^2 - 100) ≤ -5000 := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2755_275565


namespace NUMINAMATH_CALUDE_special_rectangle_area_l2755_275553

/-- Represents a rectangle divided into 6 squares with specific properties -/
structure SpecialRectangle where
  /-- The side length of the smallest square -/
  smallest_side : ℝ
  /-- The side length of the D square -/
  d_side : ℝ
  /-- Condition: The smallest square has an area of 4 square centimeters -/
  smallest_area : smallest_side ^ 2 = 4
  /-- Condition: The side lengths increase incrementally by 2 centimeters -/
  incremental_increase : d_side = smallest_side + 6

/-- The theorem stating the area of the special rectangle -/
theorem special_rectangle_area (r : SpecialRectangle) : 
  (2 * r.d_side + (r.d_side + 2)) * (r.d_side + 2 + (r.d_side + 4)) = 572 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_l2755_275553


namespace NUMINAMATH_CALUDE_solution_set_of_system_l2755_275591

theorem solution_set_of_system : 
  let S : Set (ℝ × ℝ) := {(x, y) | x - y = 0 ∧ x^2 + y = 2}
  S = {(1, 1), (-2, -2)} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_system_l2755_275591


namespace NUMINAMATH_CALUDE_find_e_l2755_275547

/-- Given two functions f and g, and a composition condition, prove the value of e. -/
theorem find_e (b e : ℝ) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = 3 * x + b)
  (g : ℝ → ℝ) (hg : ∀ x, g x = b * x + 5)
  (h_comp : ∀ x, f (g x) = 15 * x + e) : 
  e = 15 := by sorry

end NUMINAMATH_CALUDE_find_e_l2755_275547


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2755_275522

theorem cos_120_degrees :
  Real.cos (120 * π / 180) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2755_275522


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l2755_275560

/-- Given three items with cost prices 90% of their selling prices,
    prove that the overall profit percentage is 1/9 -/
theorem overall_profit_percentage
  (A_s B_s C_s : ℝ)  -- Selling prices of items A, B, and C
  (A_c B_c C_c : ℝ)  -- Cost prices of items A, B, and C
  (h1 : A_c = 0.9 * A_s)
  (h2 : B_c = 0.9 * B_s)
  (h3 : C_c = 0.9 * C_s)
  (h4 : A_s > 0)
  (h5 : B_s > 0)
  (h6 : C_s > 0) :
  (((A_s + B_s + C_s) - (A_c + B_c + C_c)) / (A_c + B_c + C_c)) = 1/9 := by
  sorry

#eval (1 : ℚ) / 9  -- To show the decimal approximation

end NUMINAMATH_CALUDE_overall_profit_percentage_l2755_275560


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_condition_l2755_275585

/-- A function f(x) = x^2 + 2(a - 1)x + 2 is monotonic on the interval [-1, 2] if and only if a ∈ (-∞, -1] ∪ [2, +∞) -/
theorem monotonic_quadratic_function_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, Monotone (fun x => x^2 + 2*(a - 1)*x + 2)) ↔
  a ∈ Set.Iic (-1 : ℝ) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_condition_l2755_275585


namespace NUMINAMATH_CALUDE_line_contains_point_l2755_275562

/-- The value of k for which the line 1 - 3kx + y = 7y contains the point (-1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (1 - 3 * k * (-1/3) + (-2) = 7 * (-2)) ↔ k = -13 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2755_275562


namespace NUMINAMATH_CALUDE_warehouse_notebooks_l2755_275514

/-- The number of notebooks in a warehouse --/
def total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) : ℕ :=
  num_boxes * parts_per_box * notebooks_per_part

/-- Theorem: The total number of notebooks in the warehouse is 660 --/
theorem warehouse_notebooks : 
  total_notebooks 22 6 5 = 660 := by
  sorry

end NUMINAMATH_CALUDE_warehouse_notebooks_l2755_275514


namespace NUMINAMATH_CALUDE_composite_transformation_matrix_l2755_275525

/-- The dilation matrix with scale factor 2 -/
def dilationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 0],
    ![0, 2]]

/-- The rotation matrix for 90 degrees counterclockwise rotation -/
def rotationMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

/-- The expected result matrix -/
def resultMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -2],
    ![2,  0]]

theorem composite_transformation_matrix :
  rotationMatrix * dilationMatrix = resultMatrix := by
  sorry

end NUMINAMATH_CALUDE_composite_transformation_matrix_l2755_275525


namespace NUMINAMATH_CALUDE_gregorian_calendar_properties_l2755_275579

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the Gregorian calendar system -/
structure GregorianCalendar where
  -- Add necessary fields and methods

/-- Counts occurrences of a specific day for January 1st in a 400-year cycle -/
def countJanuary1Occurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

/-- Counts occurrences of a specific day for the 30th of each month in a 400-year cycle -/
def count30thOccurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

theorem gregorian_calendar_properties (calendar : GregorianCalendar) :
  (countJanuary1Occurrences DayOfWeek.Sunday calendar > countJanuary1Occurrences DayOfWeek.Saturday calendar) ∧
  (∀ d : DayOfWeek, count30thOccurrences DayOfWeek.Friday calendar ≥ count30thOccurrences d calendar) :=
by sorry

end NUMINAMATH_CALUDE_gregorian_calendar_properties_l2755_275579


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2755_275546

theorem no_solution_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (1 / x + 1 / y = 3 / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2755_275546


namespace NUMINAMATH_CALUDE_twelfth_term_is_twelve_l2755_275576

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  -8 + 2 * (n - 2)

/-- Theorem: The 12th term of the arithmetic sequence is 12 -/
theorem twelfth_term_is_twelve : arithmetic_sequence 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_is_twelve_l2755_275576


namespace NUMINAMATH_CALUDE_largest_n_power_inequality_l2755_275573

theorem largest_n_power_inequality : ∃ (n : ℕ), n = 11 ∧ 
  (∀ m : ℕ, m^200 < 5^300 → m ≤ n) ∧ n^200 < 5^300 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_power_inequality_l2755_275573


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l2755_275550

theorem right_triangle_perimeter_equals_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (1/2) * a * b →
  a + b - c = 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l2755_275550


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2755_275571

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  (∀ x y : ℝ, 2 * x + y = 2 → x * y > 0 → 1 / m + 2 / n ≤ 1 / x + 2 / y) ∧
  (∃ x y : ℝ, 2 * x + y = 2 ∧ x * y > 0 ∧ 1 / x + 2 / y = 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2755_275571


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l2755_275575

/-- The rate of mangoes per kg given the purchase details --/
theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) 
  (mango_quantity : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_paid = 965 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 45 :=
by sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l2755_275575


namespace NUMINAMATH_CALUDE_kite_cost_l2755_275570

theorem kite_cost (initial_amount : ℕ) (frisbee_cost : ℕ) (remaining_amount : ℕ) (kite_cost : ℕ) : 
  initial_amount = 78 →
  frisbee_cost = 9 →
  remaining_amount = 61 →
  initial_amount = kite_cost + frisbee_cost + remaining_amount →
  kite_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_kite_cost_l2755_275570


namespace NUMINAMATH_CALUDE_cylinder_height_l2755_275500

/-- Given a cylinder with the following properties:
  * AB is the diameter of the lower base
  * A₁B₁ is a chord of the upper base, parallel to AB
  * The plane passing through AB and A₁B₁ forms an acute angle α with the lower base
  * The line AB₁ forms an angle β with the lower base
  * R is the radius of the base of the cylinder
  * A and A₁ lie on the same side of the line passing through the midpoints of AB and A₁B₁

  Prove that the height of the cylinder is equal to the given expression. -/
theorem cylinder_height (R α β : ℝ) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) :
  ∃ (height : ℝ), height = 2 * R * Real.tan β * (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) / (Real.sin α * Real.cos β) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_l2755_275500


namespace NUMINAMATH_CALUDE_fraction_comparison_l2755_275512

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  a / d > b / c := by
sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2755_275512


namespace NUMINAMATH_CALUDE_sine_special_angle_l2755_275538

theorem sine_special_angle (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (-α - π) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = -(2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sine_special_angle_l2755_275538


namespace NUMINAMATH_CALUDE_ab_is_perfect_cube_l2755_275587

theorem ab_is_perfect_cube (a b : ℕ+) (h1 : b < a) 
  (h2 : ∃ k : ℕ, k * (a * b * (a - b)) = a^3 + b^3 + a*b) : 
  ∃ n : ℕ, (a * b : ℕ) = n^3 := by
sorry

end NUMINAMATH_CALUDE_ab_is_perfect_cube_l2755_275587


namespace NUMINAMATH_CALUDE_min_omega_for_shifted_periodic_function_l2755_275532

/-- The minimum value of ω for a periodic function with a specific shift -/
theorem min_omega_for_shifted_periodic_function (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, 3 * Real.sin (ω * x + π / 6) - 2 = 
            3 * Real.sin (ω * (x - 2 * π / 3) + π / 6) - 2) →
  ω ≥ 3 ∧ ∃ n : ℕ, ω = 3 * n :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_shifted_periodic_function_l2755_275532


namespace NUMINAMATH_CALUDE_billy_crayons_left_l2755_275592

/-- The number of crayons Billy has left after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem stating that Billy has 163 crayons left after starting with 856 and a hippopotamus eating 693. -/
theorem billy_crayons_left : crayons_left 856 693 = 163 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_left_l2755_275592


namespace NUMINAMATH_CALUDE_largest_C_gap_l2755_275505

/-- Represents a square on the chessboard -/
structure Square :=
  (row : Fin 8)
  (col : Fin 8)

/-- The chessboard is an 8x8 grid of squares -/
def Chessboard := Fin 8 → Fin 8 → Square

/-- Two squares are adjacent if they share a side or vertex -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col = s2.col) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col = s2.col) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- A numbering of the chessboard is a function assigning each square a unique number from 1 to 64 -/
def Numbering := Square → Fin 64

/-- A C-gap is a number g such that for every numbering, there exist two adjacent squares whose numbers differ by at least g -/
def is_C_gap (g : ℕ) : Prop :=
  ∀ (n : Numbering), ∃ (s1 s2 : Square), 
    adjacent s1 s2 ∧ |n s1 - n s2| ≥ g

/-- The theorem stating that the largest C-gap for an 8x8 chessboard is 9 -/
theorem largest_C_gap : 
  (is_C_gap 9) ∧ (∀ g : ℕ, g > 9 → ¬(is_C_gap g)) :=
sorry

end NUMINAMATH_CALUDE_largest_C_gap_l2755_275505


namespace NUMINAMATH_CALUDE_divisor_problem_l2755_275581

theorem divisor_problem (d : ℕ) (h : d > 0) :
  (∃ n : ℤ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l2755_275581


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2755_275589

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 6 →
  downstream_distance = 35.2 →
  downstream_time = 44 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 42 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2755_275589


namespace NUMINAMATH_CALUDE_brianna_book_purchase_l2755_275539

theorem brianna_book_purchase (total_money : ℚ) (total_books : ℚ) :
  total_money > 0 ∧ total_books > 0 →
  (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books →
  total_money - 2 * ((1 / 4 : ℚ) * total_money) = (1 / 2 : ℚ) * total_money :=
by sorry

end NUMINAMATH_CALUDE_brianna_book_purchase_l2755_275539


namespace NUMINAMATH_CALUDE_inequality_with_product_condition_l2755_275507

theorem inequality_with_product_condition (x y z : ℝ) (h : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_with_product_condition_l2755_275507


namespace NUMINAMATH_CALUDE_simplify_sqrt_180_l2755_275558

theorem simplify_sqrt_180 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_180_l2755_275558


namespace NUMINAMATH_CALUDE_rectangle_area_l2755_275524

/-- The area of a rectangle with width w, length 3w, and diagonal d is (3/10)d^2 -/
theorem rectangle_area (w d : ℝ) (h1 : w > 0) (h2 : d > 0) (h3 : w^2 + (3*w)^2 = d^2) :
  w * (3*w) = (3/10) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2755_275524


namespace NUMINAMATH_CALUDE_probability_proof_l2755_275529

def total_balls : ℕ := 6
def white_balls : ℕ := 3
def black_balls : ℕ := 3
def drawn_balls : ℕ := 2

def probability_at_most_one_black : ℚ := 4/5

theorem probability_proof :
  (Nat.choose total_balls drawn_balls - Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = probability_at_most_one_black :=
sorry

end NUMINAMATH_CALUDE_probability_proof_l2755_275529


namespace NUMINAMATH_CALUDE_computer_price_increase_l2755_275551

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : 
  (351 - d) / d * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_increase_l2755_275551


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2755_275566

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_sum_magnitude (a b : E) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a - b‖ = 1) : 
  ‖a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2755_275566


namespace NUMINAMATH_CALUDE_debby_bottles_left_l2755_275526

/-- Calculates the number of water bottles left after drinking a certain amount per day for a number of days. -/
def bottles_left (total : ℕ) (per_day : ℕ) (days : ℕ) : ℕ :=
  total - (per_day * days)

/-- Theorem stating that given 264 initial bottles, drinking 15 per day for 11 days leaves 99 bottles. -/
theorem debby_bottles_left : bottles_left 264 15 11 = 99 := by
  sorry

end NUMINAMATH_CALUDE_debby_bottles_left_l2755_275526


namespace NUMINAMATH_CALUDE_probability_sum_five_l2755_275533

def Card : Type := Fin 4

def card_value (c : Card) : ℕ := c.val + 1

def sum_equals_five (c1 c2 : Card) : Prop :=
  card_value c1 + card_value c2 = 5

def total_outcomes : ℕ := 16

def favorable_outcomes : ℕ := 4

theorem probability_sum_five :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l2755_275533


namespace NUMINAMATH_CALUDE_codecracker_combinations_l2755_275515

/-- The number of colors available in the CodeCracker game -/
def num_colors : ℕ := 8

/-- The number of slots in a CodeCracker code -/
def num_slots : ℕ := 4

/-- Theorem stating the total number of possible codes in CodeCracker -/
theorem codecracker_combinations : (num_colors ^ num_slots : ℕ) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l2755_275515


namespace NUMINAMATH_CALUDE_three_solutions_condition_l2755_275582

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  ((x - 5) * Real.sin a - (y - 5) * Real.cos a = 0) ∧
  (((x + 1)^2 + (y + 1)^2 - 4) * ((x + 1)^2 + (y + 1)^2 - 16) = 0)

-- Define the condition for three solutions
def has_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system x₁ y₁ a ∧ system x₂ y₂ a ∧ system x₃ y₃ a ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧
    ∀ (x y : ℝ), system x y a → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃)

-- Theorem statement
theorem three_solutions_condition (a : ℝ) :
  has_three_solutions a ↔ ∃ (n : ℤ), a = π/4 + Real.arcsin (Real.sqrt 2 / 6) + n * π ∨
                                     a = π/4 - Real.arcsin (Real.sqrt 2 / 6) + n * π :=
sorry

end NUMINAMATH_CALUDE_three_solutions_condition_l2755_275582


namespace NUMINAMATH_CALUDE_tank_capacity_l2755_275511

/-- Represents the capacity of a tank and its inlet/outlet properties -/
structure Tank where
  capacity : ℝ
  outlet_time : ℝ
  inlet_rate : ℝ
  combined_time : ℝ

/-- Theorem stating the capacity of the tank given the conditions -/
theorem tank_capacity (t : Tank)
  (h1 : t.outlet_time = 10)
  (h2 : t.inlet_rate = 8 * 60)
  (h3 : t.combined_time = 16)
  : t.capacity = 1280 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l2755_275511


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2755_275508

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) : 
  base = 12 →
  (1/2) * base * height = 30 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2755_275508


namespace NUMINAMATH_CALUDE_derivative_at_one_l2755_275541

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x * Real.exp x) : 
  deriv f 1 = 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2755_275541


namespace NUMINAMATH_CALUDE_prime_triple_product_sum_l2755_275523

theorem prime_triple_product_sum : 
  ∀ x y z : ℕ, 
    Prime x → Prime y → Prime z →
    x * y * z = 5 * (x + y + z) →
    ((x = 2 ∧ y = 5 ∧ z = 7) ∨ (x = 5 ∧ y = 2 ∧ z = 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_triple_product_sum_l2755_275523


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l2755_275543

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_ratio_theorem (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a - b‖ = ‖a + 2 • b‖) 
  (h2 : inner a b / (‖a‖ * ‖b‖) = -1/4) : 
  ‖a‖ / ‖b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l2755_275543


namespace NUMINAMATH_CALUDE_headphones_savings_visits_l2755_275578

/-- The cost of the headphones in rubles -/
def headphones_cost : ℕ := 275

/-- The cost of a combined pool and sauna visit in rubles -/
def combined_cost : ℕ := 250

/-- The difference between pool-only cost and sauna-only cost in rubles -/
def pool_sauna_diff : ℕ := 200

/-- Calculates the cost of a pool-only visit -/
def pool_only_cost : ℕ := combined_cost - (combined_cost - pool_sauna_diff) / 2

/-- Calculates the savings per visit when choosing pool-only instead of combined -/
def savings_per_visit : ℕ := combined_cost - pool_only_cost

/-- The number of pool-only visits needed to save enough for the headphones -/
def visits_needed : ℕ := (headphones_cost + savings_per_visit - 1) / savings_per_visit

theorem headphones_savings_visits : visits_needed = 11 := by
  sorry

#eval visits_needed

end NUMINAMATH_CALUDE_headphones_savings_visits_l2755_275578


namespace NUMINAMATH_CALUDE_Q_subset_P_l2755_275534

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by sorry

end NUMINAMATH_CALUDE_Q_subset_P_l2755_275534


namespace NUMINAMATH_CALUDE_mult_41_equivalence_l2755_275577

theorem mult_41_equivalence (x y : ℤ) :
  (25 * x + 31 * y) % 41 = 0 ↔ (3 * x + 7 * y) % 41 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mult_41_equivalence_l2755_275577


namespace NUMINAMATH_CALUDE_apple_percentage_after_removal_l2755_275537

/-- Calculates the percentage of apples in a bowl of fruit -/
def percentage_apples (apples : ℕ) (oranges : ℕ) : ℚ :=
  (apples : ℚ) / (apples + oranges : ℚ) * 100

/-- Proves that after removing 19 oranges from a bowl with 14 apples and 25 oranges,
    the percentage of apples is 70% -/
theorem apple_percentage_after_removal :
  let initial_apples : ℕ := 14
  let initial_oranges : ℕ := 25
  let removed_oranges : ℕ := 19
  let remaining_oranges : ℕ := initial_oranges - removed_oranges
  percentage_apples initial_apples remaining_oranges = 70 := by
sorry

end NUMINAMATH_CALUDE_apple_percentage_after_removal_l2755_275537


namespace NUMINAMATH_CALUDE_first_number_is_202_l2755_275557

def number_list : List ℕ := [202, 204, 205, 206, 209, 209, 210, 212]

theorem first_number_is_202 (x : ℕ) (h : (number_list.sum + x) / 9 = 207) :
  number_list.head? = some 202 := by
  sorry

end NUMINAMATH_CALUDE_first_number_is_202_l2755_275557


namespace NUMINAMATH_CALUDE_complex_symmetry_quotient_l2755_275568

theorem complex_symmetry_quotient : 
  ∀ (z₁ z₂ : ℂ), 
  (z₁.im = -z₂.im) → 
  (z₁.re = z₂.re) → 
  (z₁ = 2 + I) → 
  (z₁ / z₂ = (3/5 : ℂ) + (4/5 : ℂ) * I) := by
sorry

end NUMINAMATH_CALUDE_complex_symmetry_quotient_l2755_275568


namespace NUMINAMATH_CALUDE_equation_solution_l2755_275521

theorem equation_solution (x : ℤ) (m : ℕ+) : 
  ((3 * x - 1) / 2 + m = 3) →
  ((m = 5 → x = 1) ∧ 
   (x > 0 → m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2755_275521


namespace NUMINAMATH_CALUDE_collinear_points_p_value_l2755_275584

/-- Three points are collinear if they lie on the same straight line -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem collinear_points_p_value :
  ∀ p : ℝ, collinear 1 (-2) 3 4 6 (p/3) → p = 39 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_p_value_l2755_275584


namespace NUMINAMATH_CALUDE_value_set_of_t_l2755_275527

/-- The value set of t given the conditions -/
theorem value_set_of_t (t : ℝ) : 
  (∀ y, y > 2 * 1 - t + 1 → (1, y) = (1, t)) → 
  (∀ x, x^2 + (2*t - 4)*x + 4 > 0) → 
  3 < t ∧ t < 4 := by
  sorry

end NUMINAMATH_CALUDE_value_set_of_t_l2755_275527


namespace NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2755_275502

theorem square_to_rectangle_area_increase : 
  ∀ (a : ℝ), a > 0 →
  let original_area := a * a
  let new_length := a * 1.4
  let new_breadth := a * 1.3
  let new_area := new_length * new_breadth
  (new_area - original_area) / original_area = 0.82 := by
sorry

end NUMINAMATH_CALUDE_square_to_rectangle_area_increase_l2755_275502


namespace NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l2755_275510

/-- The coefficient of x^4 in the binomial expansion of (2x^2 - 1/x)^5 is 80 -/
theorem coefficient_x4_in_binomial_expansion : 
  let n : ℕ := 5
  let a : ℚ → ℚ := λ x => 2 * x^2
  let b : ℚ → ℚ := λ x => -1/x
  let coeff : ℕ → ℚ := λ k => (-1)^k * 2^(n-k) * (n.choose k)
  (coeff 2) = 80 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_binomial_expansion_l2755_275510


namespace NUMINAMATH_CALUDE_least_11_heavy_three_digit_l2755_275516

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_11_heavy_three_digit : 
  (∀ n : ℕ, is_three_digit n → is_11_heavy n → 107 ≤ n) ∧ 
  is_three_digit 107 ∧ 
  is_11_heavy 107 :=
sorry

end NUMINAMATH_CALUDE_least_11_heavy_three_digit_l2755_275516


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2755_275555

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2755_275555


namespace NUMINAMATH_CALUDE_expression_equals_three_l2755_275556

theorem expression_equals_three :
  (-1)^2023 + Real.sqrt 9 - π^0 + Real.sqrt (1/8) * Real.sqrt 32 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_three_l2755_275556


namespace NUMINAMATH_CALUDE_power_difference_zero_l2755_275580

theorem power_difference_zero : (2^3)^2 - 4^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_zero_l2755_275580


namespace NUMINAMATH_CALUDE_travel_time_calculation_l2755_275528

/-- Calculates the time required to travel between two cities given a map scale, distance on the map, and car speed. -/
theorem travel_time_calculation (scale : ℚ) (map_distance : ℚ) (car_speed : ℚ) :
  scale = 1 / 3000000 →
  map_distance = 6 →
  car_speed = 30 →
  (map_distance * scale * 100000) / car_speed = 6000 := by
  sorry

#check travel_time_calculation

end NUMINAMATH_CALUDE_travel_time_calculation_l2755_275528
