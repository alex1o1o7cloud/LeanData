import Mathlib

namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1780_178006

/-- Given a regular pentagon and a square with the same perimeter of 20 inches,
    the ratio of the side length of the pentagon to the side length of the square is 4/5. -/
theorem pentagon_square_side_ratio :
  ∀ (p s : ℝ), 
    p > 0 → s > 0 →
    5 * p = 20 →  -- Perimeter of pentagon
    4 * s = 20 →  -- Perimeter of square
    p / s = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1780_178006


namespace NUMINAMATH_CALUDE_ad_space_width_l1780_178045

def ad_problem (num_spaces : ℕ) (length : ℝ) (cost_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  ∃ w : ℝ,
    w > 0 ∧
    num_spaces * length * w * cost_per_sqft = total_cost ∧
    w = 5

theorem ad_space_width :
  ad_problem 30 12 60 108000 :=
sorry

end NUMINAMATH_CALUDE_ad_space_width_l1780_178045


namespace NUMINAMATH_CALUDE_expression_simplification_l1780_178065

theorem expression_simplification (a b c : ℚ) 
  (ha : a = 1/3) (hb : b = 1/2) (hc : c = 1) :
  (2*a^2 - b) - (a^2 - 4*b) - (b + c) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1780_178065


namespace NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l1780_178016

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  sideLength : n > 0

/-- The total number of faces of all unit cubes when a cube of side length n is cut into n^3 unit cubes -/
def totalFaces (c : Cube n) : ℕ := 6 * n^3

/-- The number of blue faces when a cube of side length n is painted on all sides and cut into n^3 unit cubes -/
def blueFaces (c : Cube n) : ℕ := 6 * n^2

/-- The theorem stating that exactly one-third of the faces are blue if and only if n = 3 -/
theorem one_third_blue_faces_iff_three (c : Cube n) :
  3 * blueFaces c = totalFaces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l1780_178016


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l1780_178052

theorem missing_fraction_sum (sum : ℚ) (f1 f2 f3 f4 f5 f6 : ℚ) :
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/5 →
  f5 = -9/20 →
  f6 = -9/20 →
  ∃ x : ℚ, x = 23/20 ∧ sum = f1 + f2 + f3 + f4 + f5 + f6 + x :=
by sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l1780_178052


namespace NUMINAMATH_CALUDE_polly_tweets_l1780_178049

/-- Represents the tweet rate (tweets per minute) for each of Polly's activities -/
structure TweetRate where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Represents the duration (in minutes) of each of Polly's activities -/
structure ActivityDuration where
  happy : ℕ
  hungry : ℕ
  mirror : ℕ

/-- Calculates the total number of tweets given the tweet rates and activity durations -/
def totalTweets (rate : TweetRate) (duration : ActivityDuration) : ℕ :=
  rate.happy * duration.happy + rate.hungry * duration.hungry + rate.mirror * duration.mirror

/-- Theorem stating that given Polly's specific tweet rates and activity durations, 
    the total number of tweets is 1340 -/
theorem polly_tweets : 
  ∀ (rate : TweetRate) (duration : ActivityDuration),
  rate.happy = 18 ∧ rate.hungry = 4 ∧ rate.mirror = 45 ∧
  duration.happy = 20 ∧ duration.hungry = 20 ∧ duration.mirror = 20 →
  totalTweets rate duration = 1340 := by
sorry

end NUMINAMATH_CALUDE_polly_tweets_l1780_178049


namespace NUMINAMATH_CALUDE_jana_height_l1780_178026

theorem jana_height (jess_height kelly_height jana_height : ℕ) : 
  jess_height = 72 →
  kelly_height = jess_height - 3 →
  jana_height = kelly_height + 5 →
  jana_height = 74 := by
  sorry

end NUMINAMATH_CALUDE_jana_height_l1780_178026


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1780_178036

theorem product_of_polynomials (g h : ℚ) : 
  (∀ d : ℚ, (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 
    21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) → 
  g + h = -107/24 := by sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1780_178036


namespace NUMINAMATH_CALUDE_statues_painted_l1780_178038

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/16 →
  paint_per_statue = 1/16 →
  (total_paint / paint_per_statue : ℚ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_statues_painted_l1780_178038


namespace NUMINAMATH_CALUDE_polygon_product_symmetric_l1780_178095

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  
/-- Calculates the sum of products of side lengths and distances for two polygons -/
def polygonProduct (P Q : ConvexPolygon) : ℝ :=
  sorry

/-- Theorem stating that the polygon product is symmetric -/
theorem polygon_product_symmetric (P Q : ConvexPolygon) :
  polygonProduct P Q = polygonProduct Q P := by
  sorry

end NUMINAMATH_CALUDE_polygon_product_symmetric_l1780_178095


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1780_178035

-- Define an isosceles triangle with sides a, b, and c
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  validTriangle : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def perimeter (t : IsoscelesTriangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ t : IsoscelesTriangle, 
  ((t.a = 8 ∧ t.b = 8 ∧ t.c = 4) ∨ (t.a = 8 ∧ t.b = 4 ∧ t.c = 8) ∨ (t.a = 4 ∧ t.b = 8 ∧ t.c = 8)) →
  perimeter t = 20 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1780_178035


namespace NUMINAMATH_CALUDE_gcf_275_180_l1780_178081

theorem gcf_275_180 : Nat.gcd 275 180 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_275_180_l1780_178081


namespace NUMINAMATH_CALUDE_hyperbola_distance_theorem_l1780_178072

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define the distance function
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- The main theorem
theorem hyperbola_distance_theorem (x y : ℝ) (P : ℝ × ℝ) :
  is_on_hyperbola x y →
  P = (x, y) →
  distance P F₁ = 12 →
  distance P F₂ = 2 ∨ distance P F₂ = 22 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_distance_theorem_l1780_178072


namespace NUMINAMATH_CALUDE_quadratic_radical_problem_l1780_178087

-- Define what it means for two quadratic radicals to be of the same type
def same_type (x y : ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ) (p₁ p₂ : ℕ), c₁ > 0 ∧ c₂ > 0 ∧ 
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧
  Real.sqrt x = c₁ * Real.sqrt (p₁ : ℝ) ∧
  Real.sqrt y = c₂ * Real.sqrt (p₂ : ℝ) ∧
  c₁ = c₂

-- State the theorem
theorem quadratic_radical_problem (a : ℝ) :
  same_type (3*a - 4) 8 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_problem_l1780_178087


namespace NUMINAMATH_CALUDE_golden_ratio_pentagon_l1780_178079

theorem golden_ratio_pentagon (a : ℝ) : 
  a = 2 * Real.cos (72 * π / 180) → 
  (a * Real.cos (18 * π / 180)) / Real.sqrt (2 - a) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_pentagon_l1780_178079


namespace NUMINAMATH_CALUDE_circle_cut_and_reform_l1780_178055

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point inside a circle
def PointInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define the theorem
theorem circle_cut_and_reform (c : Circle) (a : ℝ × ℝ) (h : PointInside c a) :
  ∃ (part1 part2 : Set (ℝ × ℝ)), 
    (part1 ∪ part2 = {p | PointInside c p}) ∧
    (∃ (new_circle : Circle), new_circle.center = a ∧
      part1 ∪ part2 = {p | PointInside new_circle p}) :=
sorry

end NUMINAMATH_CALUDE_circle_cut_and_reform_l1780_178055


namespace NUMINAMATH_CALUDE_course_selection_problem_l1780_178004

theorem course_selection_problem (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 6 → k = 3 → m = 1 → 
  (n.choose m) * ((n - m).choose (k - m)) * ((n - k).choose (k - m)) = 180 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_problem_l1780_178004


namespace NUMINAMATH_CALUDE_philips_bananas_l1780_178063

theorem philips_bananas (num_groups : ℕ) (bananas_per_group : ℕ) 
  (h1 : num_groups = 11) (h2 : bananas_per_group = 37) :
  num_groups * bananas_per_group = 407 := by
  sorry

end NUMINAMATH_CALUDE_philips_bananas_l1780_178063


namespace NUMINAMATH_CALUDE_race_time_difference_l1780_178082

/-- Race parameters and runner speeds -/
def race_distance : ℕ := 12
def malcolm_speed : ℕ := 7
def joshua_speed : ℕ := 8

/-- Theorem stating the time difference between Malcolm and Joshua finishing the race -/
theorem race_time_difference : 
  joshua_speed * race_distance - malcolm_speed * race_distance = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1780_178082


namespace NUMINAMATH_CALUDE_min_max_sum_reciprocals_l1780_178009

open Real

theorem min_max_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) :
  let f := (1 / (x + y) + 1 / (x + z) + 1 / (y + z))
  ∃ (min_val : ℝ), (∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ min_val) ∧
  min_val = (3 / 2) ∧
  ¬∃ (max_val : ℝ), ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_reciprocals_l1780_178009


namespace NUMINAMATH_CALUDE_quadratic_value_at_three_l1780_178076

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  y_at_zero : ℝ
  h : min_value = -4
  h' : min_x = -2
  h'' : y_at_zero = 8

/-- The value of y when x = 3 for the given quadratic function -/
def y_at_three (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- Theorem stating that y = 71 when x = 3 for the given quadratic function -/
theorem quadratic_value_at_three (f : QuadraticFunction) : y_at_three f = 71 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_three_l1780_178076


namespace NUMINAMATH_CALUDE_work_completion_time_l1780_178047

theorem work_completion_time (b_completion_time b_work_days a_remaining_time : ℕ) 
  (hb : b_completion_time = 15)
  (hbw : b_work_days = 10)
  (ha : a_remaining_time = 7) : 
  ∃ (a_completion_time : ℕ), a_completion_time = 21 ∧
  (a_completion_time : ℚ)⁻¹ * a_remaining_time = 1 - (b_work_days : ℚ) / b_completion_time :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1780_178047


namespace NUMINAMATH_CALUDE_total_candies_l1780_178060

/-- Represents the number of candies Lillian has initially -/
def initial_candies : ℕ := 88

/-- Represents the number of candies Lillian receives from her father -/
def additional_candies : ℕ := 5

/-- Theorem stating the total number of candies Lillian has after receiving more -/
theorem total_candies : initial_candies + additional_candies = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_candies_l1780_178060


namespace NUMINAMATH_CALUDE_remainder_negation_l1780_178022

theorem remainder_negation (a : ℤ) : 
  (a % 1999 = 1) → ((-a) % 1999 = 1998) := by
  sorry

end NUMINAMATH_CALUDE_remainder_negation_l1780_178022


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1780_178090

/-- The perimeter of an equilateral triangle with side length 13/12 meters is 3.25 meters. -/
theorem equilateral_triangle_perimeter :
  let side_length : ℚ := 13 / 12
  let perimeter : ℚ := 3 * side_length
  perimeter = 13 / 4 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1780_178090


namespace NUMINAMATH_CALUDE_wades_tips_per_customer_l1780_178061

/-- Wade's tips per customer calculation -/
theorem wades_tips_per_customer :
  ∀ (tips_per_customer : ℚ),
  (28 : ℚ) * tips_per_customer +  -- Friday tips
  (3 * 28 : ℚ) * tips_per_customer +  -- Saturday tips (3 times Friday)
  (36 : ℚ) * tips_per_customer =  -- Sunday tips
  (296 : ℚ) →  -- Total tips
  tips_per_customer = 2 := by
sorry

end NUMINAMATH_CALUDE_wades_tips_per_customer_l1780_178061


namespace NUMINAMATH_CALUDE_sum_in_base5_l1780_178013

-- Define a function to convert from base 10 to base 5
def toBase5 (n : ℕ) : List ℕ := sorry

-- Define a function to interpret a list of digits as a number in base 5
def fromBase5 (digits : List ℕ) : ℕ := sorry

theorem sum_in_base5 : 
  toBase5 (12 + 47) = [2, 1, 4] := by sorry

end NUMINAMATH_CALUDE_sum_in_base5_l1780_178013


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1780_178043

def n : ℕ := 97 * 101 * 104 * 107 * 109

theorem distinct_prime_factors_count : Nat.card (Nat.factors n).toFinset = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1780_178043


namespace NUMINAMATH_CALUDE_share_of_c_l1780_178001

/-- 
Given a total amount to be divided among three people A, B, and C,
where A gets 2/3 of what B gets, and B gets 1/4 of what C gets,
prove that the share of C is 360 when the total amount is 510.
-/
theorem share_of_c (total : ℚ) (share_a share_b share_c : ℚ) : 
  total = 510 →
  share_a = (2/3) * share_b →
  share_b = (1/4) * share_c →
  share_a + share_b + share_c = total →
  share_c = 360 := by
sorry

end NUMINAMATH_CALUDE_share_of_c_l1780_178001


namespace NUMINAMATH_CALUDE_job_completion_time_l1780_178032

/-- Given two people P and Q working on a job, this theorem proves the time
    it takes P to complete the job alone, given the time it takes Q alone
    and the time it takes them working together. -/
theorem job_completion_time
  (time_Q : ℝ)
  (time_PQ : ℝ)
  (h1 : time_Q = 6)
  (h2 : time_PQ = 2.4) :
  ∃ (time_P : ℝ), time_P = 4 ∧ 1 / time_P + 1 / time_Q = 1 / time_PQ :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l1780_178032


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1780_178083

open Set Real

noncomputable def A : Set ℝ := {x | x^2 < 1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B :
  A ∩ (𝒰 \ B) = Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1780_178083


namespace NUMINAMATH_CALUDE_correct_det_calculation_l1780_178003

/-- Definition of 2x2 determinant -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- Polynomial M -/
def M (m n : ℝ) : ℝ := m^2 - 2*m*n

/-- Polynomial N -/
def N (m n : ℝ) : ℝ := 3*m^2 - m*n

/-- Theorem: The correct calculation of |M N; 1 3| equals -5mn -/
theorem correct_det_calculation (m n : ℝ) :
  det2x2 (M m n) (N m n) 1 3 = -5*m*n := by
  sorry

end NUMINAMATH_CALUDE_correct_det_calculation_l1780_178003


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1780_178069

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = -2) 
  (h_y : y = -2 * Real.sqrt 3) 
  (h_z : z = -1) :
  ∃ (r θ : ℝ),
    r > 0 ∧
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 4 ∧
    θ = 4 * Real.pi / 3 ∧
    x = r * Real.cos θ ∧
    y = r * Real.sin θ ∧
    z = -1 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1780_178069


namespace NUMINAMATH_CALUDE_courtyard_length_l1780_178068

/-- Proves that the length of a rectangular courtyard is 18 meters -/
theorem courtyard_length (width : ℝ) (brick_length : ℝ) (brick_width : ℝ) (total_bricks : ℕ) :
  width = 12 →
  brick_length = 0.12 →
  brick_width = 0.06 →
  total_bricks = 30000 →
  (width * (width * total_bricks * brick_length * brick_width)⁻¹) = 18 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l1780_178068


namespace NUMINAMATH_CALUDE_trapezoid_area_is_6_or_10_l1780_178078

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Theorem stating that a trapezoid with side lengths 1, 4, 4, and 5 has an area of either 6 or 10 -/
theorem trapezoid_area_is_6_or_10 (t : Trapezoid) 
    (h1 : t.side1 = 1 ∨ t.side2 = 1 ∨ t.side3 = 1 ∨ t.side4 = 1)
    (h2 : t.side1 = 4 ∨ t.side2 = 4 ∨ t.side3 = 4 ∨ t.side4 = 4)
    (h3 : t.side1 = 4 ∨ t.side2 = 4 ∨ t.side3 = 4 ∨ t.side4 = 4)
    (h4 : t.side1 = 5 ∨ t.side2 = 5 ∨ t.side3 = 5 ∨ t.side4 = 5)
    (h5 : t.side1 ≠ t.side2 ∨ t.side2 ≠ t.side3 ∨ t.side3 ≠ t.side4) : 
  area t = 6 ∨ area t = 10 := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_6_or_10_l1780_178078


namespace NUMINAMATH_CALUDE_like_terms_exponent_difference_l1780_178077

theorem like_terms_exponent_difference (m n : ℕ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * X^m * Y = b * X^3 * Y^n) → m - n = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_difference_l1780_178077


namespace NUMINAMATH_CALUDE_probability_four_students_same_group_l1780_178075

theorem probability_four_students_same_group 
  (total_students : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_students = 800) 
  (h2 : num_groups = 4) 
  (h3 : total_students % num_groups = 0) :
  (1 : ℚ) / (num_groups^3) = 1/64 :=
sorry

end NUMINAMATH_CALUDE_probability_four_students_same_group_l1780_178075


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l1780_178025

/-- Calculates the daily evaporation rate of water in a glass -/
theorem water_evaporation_rate 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) :
  initial_amount = 12 →
  evaporation_period = 22 →
  evaporation_percentage = 5.5 →
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.03 :=
by
  sorry

end NUMINAMATH_CALUDE_water_evaporation_rate_l1780_178025


namespace NUMINAMATH_CALUDE_tree_distance_l1780_178051

/-- Given 6 equally spaced trees along a straight road, where the distance between
    the first and fourth tree is 60 feet, the distance between the first and last
    tree is 100 feet. -/
theorem tree_distance (n : ℕ) (d : ℝ) (h1 : n = 6) (h2 : d = 60) :
  (n - 1) * d / 3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_tree_distance_l1780_178051


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1780_178067

theorem arithmetic_calculations :
  (15 + (-23) - (-10) = 2) ∧
  (-1^2 - (-2)^3 / 4 * (1/4) = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1780_178067


namespace NUMINAMATH_CALUDE_shaded_area_proof_l1780_178028

theorem shaded_area_proof (side_length : ℝ) (circle_radius1 circle_radius2 circle_radius3 : ℝ) :
  side_length = 30 ∧ 
  circle_radius1 = 5 ∧ 
  circle_radius2 = 4 ∧ 
  circle_radius3 = 3 →
  (side_length^2 / 9) * 5 = 500 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l1780_178028


namespace NUMINAMATH_CALUDE_lg_difference_equals_two_l1780_178010

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_difference_equals_two : lg 25 - lg (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_lg_difference_equals_two_l1780_178010


namespace NUMINAMATH_CALUDE_quadratic_solution_l1780_178024

theorem quadratic_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∀ x : ℝ, x^2 + 2*a*x + b = 0 ↔ x = a ∨ x = b) :
  a = 1 ∧ b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1780_178024


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_5pi_12_l1780_178092

theorem cos_2alpha_plus_5pi_12 (α : Real) (h1 : π < α ∧ α < 2*π) 
  (h2 : Real.sin (α + π/3) = -4/5) : 
  Real.cos (2*α + 5*π/12) = 17*Real.sqrt 2/50 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_5pi_12_l1780_178092


namespace NUMINAMATH_CALUDE_circle_intersects_y_axis_l1780_178093

theorem circle_intersects_y_axis (D E F : ℝ) :
  (∃ y₁ y₂ : ℝ, y₁ < 0 ∧ y₂ > 0 ∧ 
    y₁^2 + E*y₁ + F = 0 ∧ 
    y₂^2 + E*y₂ + F = 0) →
  F < 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersects_y_axis_l1780_178093


namespace NUMINAMATH_CALUDE_susan_age_indeterminate_l1780_178033

/-- Represents a person's age at different points in time -/
structure PersonAge where
  current : ℕ
  eightYearsAgo : ℕ
  inFifteenYears : ℕ

/-- The given conditions of the problem -/
axiom james : PersonAge
axiom janet : PersonAge
axiom susan : ℕ → Prop

axiom james_age_condition : james.inFifteenYears = 37
axiom james_janet_age_relation : james.eightYearsAgo = 2 * janet.eightYearsAgo
axiom susan_birth_condition : ∃ (age : ℕ), susan age

/-- The statement that Susan's age in 5 years cannot be determined -/
theorem susan_age_indeterminate : ¬∃ (age : ℕ), ∀ (current_age : ℕ), susan current_age → current_age + 5 = age := by
  sorry

end NUMINAMATH_CALUDE_susan_age_indeterminate_l1780_178033


namespace NUMINAMATH_CALUDE_ratio_problem_l1780_178012

theorem ratio_problem (q r s t u : ℚ) 
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : s / t = 3 / 4)
  (h4 : u / q = 1 / 2) :
  t / u = 16 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1780_178012


namespace NUMINAMATH_CALUDE_mayo_savings_l1780_178086

/-- Proves the savings when buying mayo in bulk -/
theorem mayo_savings (costco_price : ℝ) (store_price : ℝ) (gallon_oz : ℝ) (bottle_oz : ℝ) :
  costco_price = 8 →
  store_price = 3 →
  gallon_oz = 128 →
  bottle_oz = 16 →
  (gallon_oz / bottle_oz) * store_price - costco_price = 16 := by
sorry

end NUMINAMATH_CALUDE_mayo_savings_l1780_178086


namespace NUMINAMATH_CALUDE_min_t_value_fixed_point_BD_l1780_178039

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the triangle area function
def triangle_area (t angle_AOB : ℝ) : Prop := t * Real.tan angle_AOB > 0

-- Theorem for minimum value of t
theorem min_t_value (a : ℝ) : 
  ∃ (t : ℝ), triangle_area t (Real.arctan ((4*a)/(a^2 - 4))) ∧ 
  t ≥ -2 ∧ 
  (t = -2 ↔ a = 2) := 
sorry

-- Theorem for fixed point of line BD when a = -1
theorem fixed_point_BD (x y : ℝ) : 
  parabola_C x y → 
  ∃ (x' y' : ℝ), parabola_C x' (-y') ∧ 
  (y - y' = (4 / (y' + y)) * (x - x'^2/4)) → 
  x = 1 ∧ y = 0 := 
sorry

end NUMINAMATH_CALUDE_min_t_value_fixed_point_BD_l1780_178039


namespace NUMINAMATH_CALUDE_smallest_n_mod_congruence_l1780_178020

theorem smallest_n_mod_congruence :
  ∃ (n : ℕ), n > 0 ∧ (17 * n) % 7 = 1234 % 7 ∧
  ∀ (m : ℕ), m > 0 ∧ (17 * m) % 7 = 1234 % 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_mod_congruence_l1780_178020


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l1780_178030

/-- The number of bouncy balls Maggie kept -/
def total_bouncy_balls : ℝ :=
  let yellow_packs : ℝ := 8.0
  let green_packs_given : ℝ := 4.0
  let green_packs_bought : ℝ := 4.0
  let balls_per_pack : ℝ := 10.0
  yellow_packs * balls_per_pack + (green_packs_bought - green_packs_given) * balls_per_pack

/-- Theorem stating that Maggie kept 80.0 bouncy balls -/
theorem maggie_bouncy_balls : total_bouncy_balls = 80.0 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l1780_178030


namespace NUMINAMATH_CALUDE_congruence_solution_l1780_178007

theorem congruence_solution (x : ℤ) 
  (h1 : (2 + x) % (2^4) = 3^2 % (2^4))
  (h2 : (3 + x) % (3^4) = 2^3 % (3^4))
  (h3 : (4 + x) % (2^3) = 3^3 % (2^3)) :
  x % 24 = 23 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1780_178007


namespace NUMINAMATH_CALUDE_quadratic_distinct_integer_roots_l1780_178073

theorem quadratic_distinct_integer_roots (a : ℤ) : 
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ 
  (a = -2 ∨ a = 18) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_integer_roots_l1780_178073


namespace NUMINAMATH_CALUDE_sum_103_odd_numbers_from_63_l1780_178019

/-- The sum of the first n odd numbers starting from a given odd number -/
def sumOddNumbers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- Theorem: The sum of the first 103 odd numbers starting from 63 is 17015 -/
theorem sum_103_odd_numbers_from_63 :
  sumOddNumbers 63 103 = 17015 := by
  sorry

end NUMINAMATH_CALUDE_sum_103_odd_numbers_from_63_l1780_178019


namespace NUMINAMATH_CALUDE_fifth_largest_divisor_l1780_178084

def n : ℕ := 1936000000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    a > b ∧ b > c ∧ c > e ∧ e > d ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d ∨ x = e ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor :
  is_fifth_largest_divisor 121000000 := by sorry

end NUMINAMATH_CALUDE_fifth_largest_divisor_l1780_178084


namespace NUMINAMATH_CALUDE_area_between_circles_l1780_178015

/-- Given two concentric circles where the outer radius is twice the inner radius
    and the width between circles is 3, prove the area between circles is 27π -/
theorem area_between_circles (r : ℝ) (h1 : r > 0) (h2 : 2 * r - r = 3) :
  π * (2 * r)^2 - π * r^2 = 27 * π := by
  sorry

end NUMINAMATH_CALUDE_area_between_circles_l1780_178015


namespace NUMINAMATH_CALUDE_three_gorges_electricity_production_l1780_178096

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a| 
  h2 : |a| < 10

/-- The number to be represented (798.5 billion) -/
def number : ℝ := 798.5e9

/-- Theorem stating that 798.5 billion can be represented as 7.985 × 10^2 billion in scientific notation -/
theorem three_gorges_electricity_production :
  ∃ (sn : ScientificNotation), sn.a * (10 : ℝ)^sn.n = number ∧ sn.a = 7.985 ∧ sn.n = 2 :=
sorry

end NUMINAMATH_CALUDE_three_gorges_electricity_production_l1780_178096


namespace NUMINAMATH_CALUDE_max_students_for_equal_distribution_l1780_178008

theorem max_students_for_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
sorry

end NUMINAMATH_CALUDE_max_students_for_equal_distribution_l1780_178008


namespace NUMINAMATH_CALUDE_no_valid_solution_l1780_178014

theorem no_valid_solution : ¬∃ (x y z : ℕ+), 
  (x * y * z = 4 * (x + y + z)) ∧ 
  (x * y = z + x) ∧ 
  ∃ (k : ℕ+), x * y * z = k * k :=
by sorry

end NUMINAMATH_CALUDE_no_valid_solution_l1780_178014


namespace NUMINAMATH_CALUDE_total_money_sally_condition_jolly_condition_molly_condition_l1780_178070

/-- The amount of money Sally has -/
def sally_money : ℕ := 100

/-- The amount of money Jolly has -/
def jolly_money : ℕ := 50

/-- The amount of money Molly has -/
def molly_money : ℕ := 70

/-- The theorem stating the total amount of money -/
theorem total_money : sally_money + jolly_money + molly_money = 220 := by
  sorry

/-- Sally would have $80 if she had $20 less -/
theorem sally_condition : sally_money - 20 = 80 := by
  sorry

/-- Jolly would have $70 if she had $20 more -/
theorem jolly_condition : jolly_money + 20 = 70 := by
  sorry

/-- Molly would have $100 if she had $30 more -/
theorem molly_condition : molly_money + 30 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_money_sally_condition_jolly_condition_molly_condition_l1780_178070


namespace NUMINAMATH_CALUDE_red_pens_count_l1780_178050

theorem red_pens_count (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h1 : total = 240)
  (h2 : red + blue = total)
  (h3 : blue = red - 2) : 
  red = 121 := by
  sorry

end NUMINAMATH_CALUDE_red_pens_count_l1780_178050


namespace NUMINAMATH_CALUDE_dan_gave_41_cards_l1780_178098

/-- Given the initial number of cards, the number of cards bought, and the final number of cards,
    calculate the number of cards given by Dan. -/
def cards_given_by_dan (initial_cards : ℕ) (bought_cards : ℕ) (final_cards : ℕ) : ℕ :=
  final_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 cards -/
theorem dan_gave_41_cards :
  cards_given_by_dan 27 20 88 = 41 := by
  sorry

end NUMINAMATH_CALUDE_dan_gave_41_cards_l1780_178098


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l1780_178005

theorem sufficient_not_necessary_implies_necessary_not_sufficient 
  (p q : Prop) (h : (p → q) ∧ ¬(q → p)) : 
  (¬q → ¬p) ∧ ¬(¬p → ¬q) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_implies_necessary_not_sufficient_l1780_178005


namespace NUMINAMATH_CALUDE_fifteen_distinct_configurations_l1780_178091

/-- Represents a 4x4x4 cube configuration with 63 white cubes and 1 black cube -/
def CubeConfiguration := Fin 4 → Fin 4 → Fin 4 → Bool

/-- Counts the number of distinct cube configurations -/
def countDistinctConfigurations : ℕ :=
  let corner_configs := 1
  let edge_configs := 2
  let face_configs := 1
  let inner_configs := 8
  corner_configs + edge_configs + face_configs + inner_configs

/-- Theorem stating that there are 15 distinct cube configurations -/
theorem fifteen_distinct_configurations :
  countDistinctConfigurations = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_distinct_configurations_l1780_178091


namespace NUMINAMATH_CALUDE_target_hit_probability_l1780_178034

theorem target_hit_probability (p1 p2 : ℝ) (h1 : p1 = 1/2) (h2 : p2 = 1/3) :
  1 - (1 - p1) * (1 - p2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1780_178034


namespace NUMINAMATH_CALUDE_translated_segment_endpoint_l1780_178029

/-- Given a segment AB with endpoints A(-4, -1) and B(1, 1), when translated to segment A'B' where A' has coordinates (-2, 2), prove that the coordinates of B' are (3, 4). -/
theorem translated_segment_endpoint (A B A' B' : ℝ × ℝ) : 
  A = (-4, -1) → 
  B = (1, 1) → 
  A' = (-2, 2) → 
  (A'.1 - A.1 = B'.1 - B.1 ∧ A'.2 - A.2 = B'.2 - B.2) → 
  B' = (3, 4) := by
  sorry

end NUMINAMATH_CALUDE_translated_segment_endpoint_l1780_178029


namespace NUMINAMATH_CALUDE_shells_per_friend_l1780_178046

/-- Given the number of shells collected by Jillian, Savannah, and Clayton,
    and the number of friends to distribute the shells to,
    prove that each friend receives 27 shells. -/
theorem shells_per_friend
  (jillian_shells : ℕ)
  (savannah_shells : ℕ)
  (clayton_shells : ℕ)
  (num_friends : ℕ)
  (h1 : jillian_shells = 29)
  (h2 : savannah_shells = 17)
  (h3 : clayton_shells = 8)
  (h4 : num_friends = 2) :
  (jillian_shells + savannah_shells + clayton_shells) / num_friends = 27 :=
by
  sorry

#check shells_per_friend

end NUMINAMATH_CALUDE_shells_per_friend_l1780_178046


namespace NUMINAMATH_CALUDE_same_solutions_quadratic_l1780_178088

theorem same_solutions_quadratic (b c : ℝ) : 
  (∀ x : ℝ, |x - 5| = 2 ↔ x^2 + b*x + c = 0) → 
  b = -10 ∧ c = 21 := by
sorry

end NUMINAMATH_CALUDE_same_solutions_quadratic_l1780_178088


namespace NUMINAMATH_CALUDE_point_movement_l1780_178048

/-- Given three points A, B, and C on a number line, where:
    - B is 4 units to the right of A
    - C is 2 units to the left of B
    - C represents the number -3
    Prove that A represents the number -5 -/
theorem point_movement (A B C : ℝ) 
  (h1 : B = A + 4)
  (h2 : C = B - 2)
  (h3 : C = -3) :
  A = -5 := by sorry

end NUMINAMATH_CALUDE_point_movement_l1780_178048


namespace NUMINAMATH_CALUDE_binomial_sum_l1780_178023

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) : 
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 233 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l1780_178023


namespace NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l1780_178054

/-- Represents the number of socks of each color in the room -/
structure SockCollection where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  black : Nat

/-- The minimum number of socks needed to guarantee a certain number of pairs -/
def minSocksForPairs (socks : SockCollection) (pairs : Nat) : Nat :=
  5 + 5 * 2 * (pairs - 1) + 1

/-- Theorem stating the minimum number of socks needed for 15 pairs -/
theorem min_socks_for_fifteen_pairs (socks : SockCollection)
    (h1 : socks.red = 120)
    (h2 : socks.green = 100)
    (h3 : socks.blue = 70)
    (h4 : socks.yellow = 50)
    (h5 : socks.black = 30) :
    minSocksForPairs socks 15 = 146 := by
  sorry

#eval minSocksForPairs { red := 120, green := 100, blue := 70, yellow := 50, black := 30 } 15

end NUMINAMATH_CALUDE_min_socks_for_fifteen_pairs_l1780_178054


namespace NUMINAMATH_CALUDE_action_figure_fraction_l1780_178002

theorem action_figure_fraction (total_toys dolls : ℕ) : 
  total_toys = 24 → 
  dolls = 18 → 
  (total_toys - dolls : ℚ) / total_toys = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_action_figure_fraction_l1780_178002


namespace NUMINAMATH_CALUDE_volunteer_distribution_l1780_178027

def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem volunteer_distribution :
  distribute_volunteers 7 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_volunteer_distribution_l1780_178027


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l1780_178066

theorem expansion_coefficient_sum (a : ℝ) : 
  ((-a)^4 * (Nat.choose 8 4 : ℝ) = 1120) → 
  ((1 - a)^8 = 1 ∨ (1 - a)^8 = 6561) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l1780_178066


namespace NUMINAMATH_CALUDE_quiz_score_theorem_l1780_178071

theorem quiz_score_theorem :
  ∀ (correct : ℕ),
  correct ≤ 15 →
  6 * correct - 2 * (15 - correct) ≥ 75 →
  correct ≥ 14 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_score_theorem_l1780_178071


namespace NUMINAMATH_CALUDE_prime_square_difference_one_l1780_178085

theorem prime_square_difference_one (p q : ℕ) : 
  Prime p → Prime q → p^2 - 2*q^2 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_difference_one_l1780_178085


namespace NUMINAMATH_CALUDE_nadia_flower_cost_l1780_178099

/-- The total cost of flowers bought by Nadia -/
def total_cost (num_roses : ℕ) (rose_price : ℚ) : ℚ :=
  let num_lilies : ℚ := (3 / 4) * num_roses
  let lily_price : ℚ := 2 * rose_price
  num_roses * rose_price + num_lilies * lily_price

/-- Theorem stating the total cost of flowers for Nadia's purchase -/
theorem nadia_flower_cost : total_cost 20 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_nadia_flower_cost_l1780_178099


namespace NUMINAMATH_CALUDE_unique_total_prices_l1780_178059

def gift_prices : List ℕ := [2, 5, 8, 11, 14]
def box_prices : List ℕ := [3, 5, 7, 9, 11]

def total_prices : List ℕ :=
  List.eraseDups (List.map (λ (p : ℕ × ℕ) => p.1 + p.2) (List.product gift_prices box_prices))

theorem unique_total_prices :
  total_prices.length = 19 := by sorry

end NUMINAMATH_CALUDE_unique_total_prices_l1780_178059


namespace NUMINAMATH_CALUDE_balloon_height_per_ounce_l1780_178040

/-- Calculates the height increase per ounce of helium for a balloon flight --/
theorem balloon_height_per_ounce 
  (total_money : ℚ)
  (sheet_cost : ℚ)
  (rope_cost : ℚ)
  (propane_cost : ℚ)
  (helium_price_per_ounce : ℚ)
  (max_height : ℚ)
  (h1 : total_money = 200)
  (h2 : sheet_cost = 42)
  (h3 : rope_cost = 18)
  (h4 : propane_cost = 14)
  (h5 : helium_price_per_ounce = 3/2)
  (h6 : max_height = 9492) :
  (max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price_per_ounce)) = 113 := by
  sorry

end NUMINAMATH_CALUDE_balloon_height_per_ounce_l1780_178040


namespace NUMINAMATH_CALUDE_square_area_ratio_l1780_178000

theorem square_area_ratio (s t : ℝ) (h : s > 0) (k : t > 0) (h_perimeter : 4 * s = 4 * (4 * t)) :
  s^2 = 16 * t^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1780_178000


namespace NUMINAMATH_CALUDE_complex_sum_problem_l1780_178064

theorem complex_sum_problem (x y u v w z : ℝ) 
  (h1 : y = 2)
  (h2 : w = -x - u)
  (h3 : Complex.mk x y + Complex.mk u v + Complex.mk w z = Complex.I * (-2)) :
  v + z = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l1780_178064


namespace NUMINAMATH_CALUDE_symmetry_of_curves_l1780_178031

/-- The original curve E -/
def E (x y : ℝ) : Prop := x^2 + 2*x*y + y^2 + 3*x + y = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop := 2*x - y - 1 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop := x^2 + 14*x*y + 49*y^2 - 21*x + 103*y + 54 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetry_of_curves :
  ∀ (x y x' y' : ℝ),
    E x y →
    l ((x + x') / 2) ((y + y') / 2) →
    E' x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_curves_l1780_178031


namespace NUMINAMATH_CALUDE_power_of_power_l1780_178044

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1780_178044


namespace NUMINAMATH_CALUDE_largest_possible_b_l1780_178094

theorem largest_possible_b (a b c : ℕ) : 
  (a * b * c = 360) →
  (1 < c) →
  (c < b) →
  (b < a) →
  (∀ a' b' c' : ℕ, (a' * b' * c' = 360) → (1 < c') → (c' < b') → (b' < a') → b' ≤ b) →
  b = 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_b_l1780_178094


namespace NUMINAMATH_CALUDE_max_intersection_points_fifth_degree_polynomials_l1780_178062

/-- A fifth-degree polynomial function with leading coefficient 1 -/
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := 
  λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The difference between two fifth-degree polynomials where one has an additional -x^3 term -/
def PolynomialDifference (p q : ℝ → ℝ) : ℝ → ℝ :=
  λ x => p x - q x

theorem max_intersection_points_fifth_degree_polynomials :
  ∀ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℝ),
  let p := FifthDegreePolynomial a₁ b₁ c₁ d₁ e₁
  let q := FifthDegreePolynomial a₂ (b₂ - 1) c₂ d₂ e₂
  let diff := PolynomialDifference p q
  (∀ x : ℝ, diff x = 0 → x = 0) ∧
  (∃ x : ℝ, diff x = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_points_fifth_degree_polynomials_l1780_178062


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l1780_178021

theorem solve_sqrt_equation (x : ℝ) (h : x > 0) :
  Real.sqrt ((3 / x) + 3) = 2 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l1780_178021


namespace NUMINAMATH_CALUDE_video_game_sales_theorem_l1780_178057

/-- Given a total number of video games, number of non-working games, and a price per working game,
    calculate the total money that can be earned by selling the working games. -/
def total_money_earned (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Theorem stating that given 10 total games, 8 non-working games, and a price of $6 per working game,
    the total money earned is $12. -/
theorem video_game_sales_theorem :
  total_money_earned 10 8 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_video_game_sales_theorem_l1780_178057


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l1780_178041

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (correct_avg : ℚ) (n : ℕ) : 
  initial_avg = 18 →
  incorrect_num = 26 →
  correct_num = 66 →
  correct_avg = 22 →
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg →
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l1780_178041


namespace NUMINAMATH_CALUDE_face_mask_profit_l1780_178056

/-- Calculates the total profit from selling face masks given the specified conditions. -/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let discount_rate : ℚ := 1/5
  let original_price : ℚ := 8
  let masks_per_box : List ℕ := [25, 30, 35]
  let selling_price : ℚ := 3/5

  let discounted_price := original_price * (1 - discount_rate)
  let total_cost := num_boxes * discounted_price
  let total_masks := masks_per_box.sum
  let total_revenue := total_masks * selling_price
  let profit := total_revenue - total_cost

  profit = 348/10 :=
by sorry

end NUMINAMATH_CALUDE_face_mask_profit_l1780_178056


namespace NUMINAMATH_CALUDE_production_days_l1780_178042

/-- Given the average daily production for n days and the effect of adding one more day's production,
    prove the value of n. -/
theorem production_days (n : ℕ) : 
  (∀ (P : ℕ), P / n = 60 → (P + 90) / (n + 1) = 65) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_production_days_l1780_178042


namespace NUMINAMATH_CALUDE_curve_equation_relationship_l1780_178089

-- Define the curve C as a set of points in 2D space
def C : Set (ℝ × ℝ) := sorry

-- Define the function f
def f : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem curve_equation_relationship :
  (∀ x y, f x y = 0 → (x, y) ∈ C) →
  (∀ x y, (x, y) ∉ C → f x y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_relationship_l1780_178089


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l1780_178053

def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 2*x

theorem fixed_points_of_f_composition :
  ∀ x : ℝ, f (f x) = f x ↔ x = 0 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l1780_178053


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1780_178037

theorem perfect_square_binomial (x : ℝ) : 
  ∃ (a b : ℝ), 16 * x^2 - 40 * x + 25 = (a * x + b)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1780_178037


namespace NUMINAMATH_CALUDE_points_on_line_l1780_178017

-- Define the line y = -3x + b
def line (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

-- Define the points
def point1 (y₁ : ℝ) (b : ℝ) : Prop := y₁ = line (-2) b
def point2 (y₂ : ℝ) (b : ℝ) : Prop := y₂ = line (-1) b
def point3 (y₃ : ℝ) (b : ℝ) : Prop := y₃ = line 1 b

-- Theorem statement
theorem points_on_line (y₁ y₂ y₃ b : ℝ) 
  (h1 : point1 y₁ b) (h2 : point2 y₂ b) (h3 : point3 y₃ b) :
  y₁ > y₂ ∧ y₂ > y₃ := by sorry

end NUMINAMATH_CALUDE_points_on_line_l1780_178017


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l1780_178018

theorem bucket_capacity_reduction (current_buckets : ℕ) (reduction_factor : ℚ) : 
  current_buckets = 25 → 
  reduction_factor = 2 / 5 →
  ↑(Nat.ceil ((current_buckets : ℚ) / reduction_factor)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l1780_178018


namespace NUMINAMATH_CALUDE_anne_had_fifteen_sweettarts_l1780_178011

/-- The number of Sweettarts Anne had initially -/
def annes_initial_sweettarts (num_friends : ℕ) (sweettarts_per_friend : ℕ) : ℕ :=
  num_friends * sweettarts_per_friend

/-- Theorem stating that Anne had 15 Sweettarts initially -/
theorem anne_had_fifteen_sweettarts :
  annes_initial_sweettarts 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_had_fifteen_sweettarts_l1780_178011


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1780_178058

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) :
  Complex.abs ((a - Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1780_178058


namespace NUMINAMATH_CALUDE_hexagon_ratio_l1780_178097

/-- A hexagon with specific properties -/
structure Hexagon :=
  (total_area : ℝ)
  (bisector : ℝ → ℝ → Prop)
  (lower_part : ℝ → ℝ → Prop)
  (triangle_base : ℝ)

/-- The theorem statement -/
theorem hexagon_ratio (h : Hexagon) (x y : ℝ) : 
  h.total_area = 7 ∧ 
  h.bisector x y ∧ 
  h.lower_part 1 (5/2) ∧ 
  h.triangle_base = 4 →
  x / y = 1 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_ratio_l1780_178097


namespace NUMINAMATH_CALUDE_calculation_proof_l1780_178074

theorem calculation_proof : (300000 * 200000) / 100000 = 600000 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1780_178074


namespace NUMINAMATH_CALUDE_largest_number_l1780_178080

def a : ℝ := 8.12334
def b : ℝ := 8.123333333 -- Approximation of 8.123̅3
def c : ℝ := 8.123333333 -- Approximation of 8.12̅33
def d : ℝ := 8.123323323 -- Approximation of 8.1̅233
def e : ℝ := 8.123312331 -- Approximation of 8.̅1233

theorem largest_number : 
  (b = c) ∧ (b ≥ a) ∧ (b ≥ d) ∧ (b ≥ e) := by sorry

end NUMINAMATH_CALUDE_largest_number_l1780_178080
