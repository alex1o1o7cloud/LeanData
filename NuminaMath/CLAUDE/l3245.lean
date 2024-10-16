import Mathlib

namespace NUMINAMATH_CALUDE_proposition_a_proposition_d_l3245_324581

-- Proposition A
theorem proposition_a (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) :
  -4 < a - b ∧ a - b < 2 := by sorry

-- Proposition D
theorem proposition_d : ∃ a : ℝ, a + 1 / a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_proposition_a_proposition_d_l3245_324581


namespace NUMINAMATH_CALUDE_junk_mail_count_l3245_324540

/-- The number of houses in a block -/
def houses_per_block : ℕ := 7

/-- The number of junk mails given to each house -/
def junk_mails_per_house : ℕ := 2

/-- The total number of junk mails given to a block -/
def total_junk_mails : ℕ := houses_per_block * junk_mails_per_house

theorem junk_mail_count : total_junk_mails = 14 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_count_l3245_324540


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l3245_324553

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

-- Define the theorem
theorem quadratic_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : f c (-1) = y₁)
  (h2 : f c 2 = y₂)
  (h3 : f c (-3) = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l3245_324553


namespace NUMINAMATH_CALUDE_movie_duration_l3245_324552

theorem movie_duration (tuesday_time : ℕ) (max_movies : ℕ) 
  (h1 : tuesday_time = 270)
  (h2 : max_movies = 9) : 
  ∃ (movie_length : ℕ), movie_length = 90 ∧ 
  ∃ (tuesday_movies : ℕ), 
    tuesday_movies * movie_length = tuesday_time ∧
    3 * tuesday_movies = max_movies :=
by
  sorry

end NUMINAMATH_CALUDE_movie_duration_l3245_324552


namespace NUMINAMATH_CALUDE_hurleys_age_l3245_324561

theorem hurleys_age (hurley_age richard_age : ℕ) : 
  richard_age - hurley_age = 20 →
  (richard_age + 40) + (hurley_age + 40) = 128 →
  hurley_age = 14 := by
sorry

end NUMINAMATH_CALUDE_hurleys_age_l3245_324561


namespace NUMINAMATH_CALUDE_sales_equation_l3245_324563

theorem sales_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : 5000 / (x + 1) > 0) -- sales quantity last year is positive
  (h3 : 5000 / x > 0) -- sales quantity this year is positive
  (h4 : 5000 / (x + 1) = 5000 / x) -- sales quantity remains the same
  : 5000 / (x + 1) = 5000 * (1 - 0.2) / x := by
  sorry

end NUMINAMATH_CALUDE_sales_equation_l3245_324563


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l3245_324541

/-- The midpoint of a line segment in polar coordinates -/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: The midpoint of the line segment with endpoints (5, π/4) and (5, 3π/4) in polar coordinates is (5√2/2, π/2) -/
theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 5 (π/4) 5 (3*π/4)
  r = 5 * Real.sqrt 2 / 2 ∧ θ = π/2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*π :=
by sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l3245_324541


namespace NUMINAMATH_CALUDE_bricks_used_total_bricks_used_l3245_324512

/-- Calculates the total number of bricks used in a construction project -/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) (incomplete_courses : ℕ) : ℕ :=
  let complete_walls := total_walls - 1
  let complete_wall_bricks := courses_per_wall * bricks_per_course
  let incomplete_wall_courses := courses_per_wall - incomplete_courses
  let incomplete_wall_bricks := incomplete_wall_courses * bricks_per_course
  complete_walls * complete_wall_bricks + incomplete_wall_bricks

/-- Proves that the total number of bricks used is 1140 given the specific conditions -/
theorem total_bricks_used :
  bricks_used 10 20 6 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_total_bricks_used_l3245_324512


namespace NUMINAMATH_CALUDE_zero_bounds_l3245_324594

theorem zero_bounds (a : ℝ) (x₀ : ℝ) (h_a : a > 0) 
  (h_zero : Real.exp (2 * x₀) + (a + 2) * Real.exp x₀ + a * x₀ = 0) : 
  Real.log (2 * a / (4 * a + 5)) < x₀ ∧ x₀ < -1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_bounds_l3245_324594


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3245_324513

theorem complex_equation_solution (z : ℂ) :
  (z - 2*I) * (2 - I) = 5 → z = 2 + 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3245_324513


namespace NUMINAMATH_CALUDE_secant_length_l3245_324599

noncomputable section

def Circle (O : ℝ × ℝ) (R : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = R^2}

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def isTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) := 
  ∃! P, P ∈ l ∩ c

def isSecant (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) :=
  ∃ P Q, P ≠ Q ∧ P ∈ l ∩ c ∧ Q ∈ l ∩ c

def isEquidistant (P : ℝ × ℝ) (Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) :=
  ∀ X ∈ l, distance P X = distance Q X

theorem secant_length (O : ℝ × ℝ) (R : ℝ) (A : ℝ × ℝ) 
  (h1 : distance O A = 2 * R)
  (c : Set (ℝ × ℝ)) (h2 : c = Circle O R)
  (t : Set (ℝ × ℝ)) (h3 : isTangent t c)
  (s : Set (ℝ × ℝ)) (h4 : isSecant s c)
  (B : ℝ × ℝ) (h5 : B ∈ t ∩ c)
  (h6 : isEquidistant O B s) :
  ∃ C G : ℝ × ℝ, C ∈ s ∩ c ∧ G ∈ s ∩ c ∧ distance C G = 2 * R * Real.sqrt (10/13) :=
sorry

end NUMINAMATH_CALUDE_secant_length_l3245_324599


namespace NUMINAMATH_CALUDE_perfect_squares_product_sum_l3245_324514

theorem perfect_squares_product_sum (a b : ℕ) : 
  (∃ x : ℕ, a = x^2) → 
  (∃ y : ℕ, b = y^2) → 
  a * b = a + b + 4844 →
  (Real.sqrt a + 1) * (Real.sqrt b + 1) * (Real.sqrt a - 1) * (Real.sqrt b - 1) - 
  (Real.sqrt 68 + 1) * (Real.sqrt 63 + 1) * (Real.sqrt 68 - 1) * (Real.sqrt 63 - 1) = 691 := by
sorry

end NUMINAMATH_CALUDE_perfect_squares_product_sum_l3245_324514


namespace NUMINAMATH_CALUDE_gold_bar_ratio_l3245_324530

theorem gold_bar_ratio (initial_bars : ℕ) (tax_percent : ℚ) (final_bars : ℕ) : 
  initial_bars = 60 →
  tax_percent = 1/10 →
  final_bars = 27 →
  (initial_bars - initial_bars * tax_percent - final_bars) / (initial_bars - initial_bars * tax_percent) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_gold_bar_ratio_l3245_324530


namespace NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l3245_324557

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by sorry

end NUMINAMATH_CALUDE_cos_four_minus_sin_four_equals_cos_double_l3245_324557


namespace NUMINAMATH_CALUDE_cookies_per_person_l3245_324588

/-- The number of cookie batches Beth bakes in a week -/
def batches : ℕ := 8

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 5

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of people sharing the cookies -/
def people : ℕ := 30

/-- Theorem: If 8 batches of 5 dozen cookies are shared equally among 30 people,
    each person will receive 16 cookies -/
theorem cookies_per_person :
  (batches * dozens_per_batch * cookies_per_dozen) / people = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_person_l3245_324588


namespace NUMINAMATH_CALUDE_average_of_numbers_l3245_324595

def numbers : List ℝ := [3, 16, 33, 28]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3245_324595


namespace NUMINAMATH_CALUDE_problem_solution_l3245_324577

theorem problem_solution (x y : ℚ) (hx : x = 3/5) (hy : y = 5/3) :
  (1/3) * x^8 * y^9 = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3245_324577


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3245_324580

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 40 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3245_324580


namespace NUMINAMATH_CALUDE_exists_nonpositive_square_l3245_324536

theorem exists_nonpositive_square : ∃ x : ℝ, -x^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_nonpositive_square_l3245_324536


namespace NUMINAMATH_CALUDE_machine_a_production_time_l3245_324548

/-- The time (in minutes) it takes for Machine A to produce one item -/
def t : ℝ := sorry

/-- The time (in minutes) it takes for Machine B to produce one item -/
def machine_b_time : ℝ := 5

/-- The duration of the production period in minutes -/
def production_period : ℝ := 1440

/-- The ratio of items produced by Machine A compared to Machine B -/
def production_ratio : ℝ := 1.25

theorem machine_a_production_time : 
  (production_period / t = production_ratio * (production_period / machine_b_time)) → t = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_a_production_time_l3245_324548


namespace NUMINAMATH_CALUDE_regular_quadrilateral_pyramid_angle_l3245_324572

/-- A regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The angle between a slant edge and the base plane -/
  slant_base_angle : ℝ
  /-- The angle between a slant edge and the plane of the lateral face that does not contain this edge -/
  slant_lateral_angle : ℝ
  /-- The angles are equal -/
  angle_equality : slant_base_angle = slant_lateral_angle

/-- The theorem stating the angle in a regular quadrilateral pyramid -/
theorem regular_quadrilateral_pyramid_angle (pyramid : RegularQuadrilateralPyramid) :
  pyramid.slant_base_angle = Real.arctan (Real.sqrt (3 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_regular_quadrilateral_pyramid_angle_l3245_324572


namespace NUMINAMATH_CALUDE_gas_cost_equation_l3245_324559

theorem gas_cost_equation (x : ℚ) : x > 0 →
  (∃ (n m : ℕ), n = 4 ∧ m = 7 ∧ x / n - x / m = 10) ↔ x = 280 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gas_cost_equation_l3245_324559


namespace NUMINAMATH_CALUDE_vertex_landing_probability_l3245_324584

/-- Square vertices -/
def square_vertices : List (Int × Int) := [(2, 2), (-2, 2), (-2, -2), (2, -2)]

/-- All boundary points of the square -/
def boundary_points : List (Int × Int) := [
  (2, 2), (-2, 2), (-2, -2), (2, -2),  -- vertices
  (1, 2), (0, 2), (-1, 2),             -- top edge
  (1, -2), (0, -2), (-1, -2),          -- bottom edge
  (2, 1), (2, 0), (2, -1),             -- right edge
  (-2, 1), (-2, 0), (-2, -1)           -- left edge
]

/-- Neighboring points function -/
def neighbors (x y : Int) : List (Int × Int) := [
  (x, y+1), (x+1, y+1), (x+1, y),
  (x+1, y-1), (x, y-1), (x-1, y-1),
  (x-1, y), (x-1, y+1)
]

/-- Theorem: Probability of landing on a vertex is 1/4 -/
theorem vertex_landing_probability :
  let start := (0, 0)
  let p_vertex := (square_vertices.length : ℚ) / (boundary_points.length : ℚ)
  p_vertex = 1/4 := by sorry

end NUMINAMATH_CALUDE_vertex_landing_probability_l3245_324584


namespace NUMINAMATH_CALUDE_ruth_math_class_hours_l3245_324555

/-- Represents Ruth's weekly school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_hours :
  let schedule := RuthSchedule.mk 8 5 0.25
  math_class_hours schedule = 10 := by
  sorry

end NUMINAMATH_CALUDE_ruth_math_class_hours_l3245_324555


namespace NUMINAMATH_CALUDE_total_plums_picked_l3245_324565

-- Define the number of plums picked by each person
def melanie_plums : ℕ := 4
def dan_plums : ℕ := 9
def sally_plums : ℕ := 3

-- Define Ben's plums as twice the sum of Melanie's and Dan's
def ben_plums : ℕ := 2 * (melanie_plums + dan_plums)

-- Define the number of plums Sally ate
def sally_ate : ℕ := 2

-- Theorem: The total number of plums picked is 40
theorem total_plums_picked : 
  melanie_plums + dan_plums + sally_plums + ben_plums - sally_ate = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_picked_l3245_324565


namespace NUMINAMATH_CALUDE_same_solution_implies_a_b_values_l3245_324549

theorem same_solution_implies_a_b_values :
  ∀ (a b x y : ℚ),
  (3 * x - y = 7 ∧ a * x + y = b) ∧
  (x + b * y = a ∧ 2 * x + y = 8) →
  a = -7/5 ∧ b = -11/5 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_b_values_l3245_324549


namespace NUMINAMATH_CALUDE_expression_simplification_l3245_324523

theorem expression_simplification 
  (x y z w : ℝ) 
  (hx : x ≠ 2) 
  (hy : y ≠ 3) 
  (hz : z ≠ 4) 
  (hw : w ≠ 5) 
  (h1 : y ≠ 6) 
  (h2 : w ≠ 4) 
  (h3 : z ≠ 6) :
  (x - 2) / (6 - y) * (y - 3) / (2 - x) * (z - 4) / (3 - y) * 
  (6 - z) / (4 - w) * (w - 5) / (z - 6) * (x + 1) / (5 - w) = 1 := by
  sorry

#check expression_simplification

end NUMINAMATH_CALUDE_expression_simplification_l3245_324523


namespace NUMINAMATH_CALUDE_cost_of_potting_soil_l3245_324522

def cost_of_seeds : ℝ := 2.00
def number_of_plants : ℕ := 20
def price_per_plant : ℝ := 5.00
def net_profit : ℝ := 90.00

theorem cost_of_potting_soil :
  ∃ (cost : ℝ), cost = (number_of_plants : ℝ) * price_per_plant - cost_of_seeds - net_profit :=
by sorry

end NUMINAMATH_CALUDE_cost_of_potting_soil_l3245_324522


namespace NUMINAMATH_CALUDE_min_value_product_l3245_324589

theorem min_value_product (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h_pos : θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₄ > 0) 
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) : 
  (2 * Real.sin θ₁ ^ 2 + 1 / Real.sin θ₁ ^ 2) * 
  (2 * Real.sin θ₂ ^ 2 + 1 / Real.sin θ₂ ^ 2) * 
  (2 * Real.sin θ₃ ^ 2 + 1 / Real.sin θ₃ ^ 2) * 
  (2 * Real.sin θ₄ ^ 2 + 1 / Real.sin θ₄ ^ 2) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l3245_324589


namespace NUMINAMATH_CALUDE_d_value_for_lines_l3245_324504

/-- Two straight lines pass through four points in 3D space -/
def line_through_points (a b c d : ℝ) (k : ℕ) : Prop :=
  ∃ (l₁ l₂ : Set (ℝ × ℝ × ℝ)),
    l₁ ≠ l₂ ∧
    (1, 0, a) ∈ l₁ ∧ (1, 0, a) ∈ l₂ ∧
    (b, 1, 0) ∈ l₁ ∧ (b, 1, 0) ∈ l₂ ∧
    (0, c, 1) ∈ l₁ ∧ (0, c, 1) ∈ l₂ ∧
    (k * d, k * d, -d) ∈ l₁ ∧ (k * d, k * d, -d) ∈ l₂

/-- The theorem stating the possible values of d -/
theorem d_value_for_lines (k : ℕ) (h1 : k ≠ 6) (h2 : k ≠ 1) :
  ∀ a b c d : ℝ, line_through_points a b c d k → d = -k / (k - 1) :=
by sorry

end NUMINAMATH_CALUDE_d_value_for_lines_l3245_324504


namespace NUMINAMATH_CALUDE_vermont_ads_l3245_324562

def ads_problem (first_page_ads : ℕ) (total_pages : ℕ) (click_fraction : ℚ) : Prop :=
  let second_page_ads := 2 * first_page_ads
  let third_page_ads := second_page_ads + 24
  let fourth_page_ads := (3 : ℚ) / 4 * second_page_ads
  let total_ads := first_page_ads + second_page_ads + third_page_ads + fourth_page_ads
  let clicked_ads := click_fraction * total_ads
  
  first_page_ads = 12 ∧
  total_pages = 4 ∧
  click_fraction = 2 / 3 ∧
  clicked_ads = 68

theorem vermont_ads : ads_problem 12 4 (2/3) := by sorry

end NUMINAMATH_CALUDE_vermont_ads_l3245_324562


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3245_324501

theorem smallest_n_square_and_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ (∃ (y : ℕ), 4 * x = y^2) ∧ (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3245_324501


namespace NUMINAMATH_CALUDE_symmetry_about_y_axis_l3245_324587

/-- Given two real numbers a and b such that log(a) + log(b) = 0, a ≠ 1, and b ≠ 1,
    prove that the functions f(x) = ax and g(x) = bx are symmetric about the y-axis. -/
theorem symmetry_about_y_axis 
  (a b : ℝ) 
  (h1 : Real.log a + Real.log b = 0) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  ∀ x : ℝ, ∃ y : ℝ, a * x = b * (-y) ∧ a * (-x) = b * y :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_y_axis_l3245_324587


namespace NUMINAMATH_CALUDE_marbles_given_to_sam_l3245_324535

/-- Given that Mike initially had 8 marbles and now has 4 left, prove that he gave 4 marbles to Sam. -/
theorem marbles_given_to_sam 
  (initial_marbles : Nat) 
  (remaining_marbles : Nat) 
  (h1 : initial_marbles = 8) 
  (h2 : remaining_marbles = 4) : 
  initial_marbles - remaining_marbles = 4 := by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_sam_l3245_324535


namespace NUMINAMATH_CALUDE_friends_pooling_money_l3245_324508

-- Define the friends
inductive Friend
| Emma
| Daya
| Jeff
| Brenda

-- Define a function to get the amount of money each friend has
def money (f : Friend) : ℚ :=
  match f with
  | Friend.Emma => 8
  | Friend.Daya => 8 * (1 + 1/4)
  | Friend.Jeff => (2/5) * (8 * (1 + 1/4))
  | Friend.Brenda => (2/5) * (8 * (1 + 1/4)) + 4

-- Theorem stating that there are 4 friends pooling money for pizza
theorem friends_pooling_money :
  (∃ (s : Finset Friend), s.card = 4 ∧ 
    (∀ f : Friend, f ∈ s) ∧
    (money Friend.Emma = 8) ∧
    (money Friend.Daya = money Friend.Emma * (1 + 1/4)) ∧
    (money Friend.Jeff = (2/5) * money Friend.Daya) ∧
    (money Friend.Brenda = money Friend.Jeff + 4) ∧
    (money Friend.Brenda = 8)) :=
by sorry

end NUMINAMATH_CALUDE_friends_pooling_money_l3245_324508


namespace NUMINAMATH_CALUDE_hyperbola_focal_property_l3245_324590

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_focal_property (P : ℝ × ℝ) :
  is_on_hyperbola P.1 P.2 → distance P F1 = 5 →
  (distance P F2 = 3 ∨ distance P F2 = 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_property_l3245_324590


namespace NUMINAMATH_CALUDE_hotel_occupancy_and_profit_l3245_324529

/-- Represents a hotel with its pricing and occupancy characteristics -/
structure Hotel where
  totalRooms : ℕ
  originalPrice : ℕ
  fullBookingPrice : ℕ
  costPerRoom : ℕ
  vacancyRate : ℚ
  maxPriceMultiplier : ℚ

/-- Calculates the number of occupied rooms given a price increase -/
def occupiedRooms (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  h.totalRooms - priceIncrease * h.vacancyRate

/-- Calculates the profit given a price increase -/
def profit (h : Hotel) (priceIncrease : ℕ) : ℚ :=
  (h.fullBookingPrice + priceIncrease - h.costPerRoom) * occupiedRooms h priceIncrease

/-- The hotel in the problem -/
def problemHotel : Hotel := {
  totalRooms := 50
  originalPrice := 190
  fullBookingPrice := 180
  costPerRoom := 20
  vacancyRate := 1/10
  maxPriceMultiplier := 3/2
}

theorem hotel_occupancy_and_profit :
  (occupiedRooms problemHotel 50 = 45) ∧
  (profit problemHotel 50 = 9450) := by sorry

end NUMINAMATH_CALUDE_hotel_occupancy_and_profit_l3245_324529


namespace NUMINAMATH_CALUDE_smallest_factorization_l3245_324543

def is_valid_factorization (b r s : ℤ) : Prop :=
  r * s = 4032 ∧ r + s = b

def has_integer_factorization (b : ℤ) : Prop :=
  ∃ r s : ℤ, is_valid_factorization b r s

theorem smallest_factorization : 
  (∀ b : ℤ, b > 0 ∧ b < 127 → ¬(has_integer_factorization b)) ∧ 
  has_integer_factorization 127 :=
sorry

end NUMINAMATH_CALUDE_smallest_factorization_l3245_324543


namespace NUMINAMATH_CALUDE_intersection_sum_modulo13_l3245_324516

theorem intersection_sum_modulo13 : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 13 ∧ 
  (∃ (y : ℤ), y ≡ 3*x + 4 [ZMOD 13] ∧ y ≡ 8*x + 9 [ZMOD 13]) ∧
  (∀ (x' : ℤ), 0 ≤ x' ∧ x' < 13 → 
    (∃ (y' : ℤ), y' ≡ 3*x' + 4 [ZMOD 13] ∧ y' ≡ 8*x' + 9 [ZMOD 13]) → 
    x' = x) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_modulo13_l3245_324516


namespace NUMINAMATH_CALUDE_num_valid_schedules_is_336_l3245_324566

/-- Represents the days of the week excluding Saturday -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- Represents the teachers -/
inductive Teacher
| Math
| English
| Other1
| Other2
| Other3
| Other4

/-- A schedule is a function from Teacher to Day -/
def Schedule := Teacher → Day

/-- Predicate to check if a schedule is valid -/
def validSchedule (s : Schedule) : Prop :=
  s Teacher.Math ≠ Day.Monday ∧
  s Teacher.Math ≠ Day.Wednesday ∧
  s Teacher.English ≠ Day.Tuesday ∧
  s Teacher.English ≠ Day.Thursday ∧
  (∀ t1 t2 : Teacher, t1 ≠ t2 → s t1 ≠ s t2)

/-- The number of valid schedules -/
def numValidSchedules : ℕ := sorry

theorem num_valid_schedules_is_336 : numValidSchedules = 336 := by sorry

end NUMINAMATH_CALUDE_num_valid_schedules_is_336_l3245_324566


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l3245_324558

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l3245_324558


namespace NUMINAMATH_CALUDE_combination_equality_l3245_324592

theorem combination_equality (n : ℕ) : 
  (Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8) → n = 14 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l3245_324592


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3245_324526

theorem algebraic_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ -b) :
  (a^2 + a*b + b^2) / (a + b) - (a^2 - a*b + b^2) / (a - b) + (2*b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3245_324526


namespace NUMINAMATH_CALUDE_no_solution_for_quadratic_congruence_l3245_324564

theorem no_solution_for_quadratic_congruence :
  ∀ n : ℕ, ¬(100 ∣ (n^2 + 6*n + 2019)) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_quadratic_congruence_l3245_324564


namespace NUMINAMATH_CALUDE_min_value_of_g_l3245_324596

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Define the property that f is decreasing on (-∞, 1]
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 1 → f a x ≥ f a y

-- Define the function g
def g (a : ℝ) : ℝ := f a (a + 1) - f a 1

-- State the theorem
theorem min_value_of_g :
  ∀ a : ℝ, is_decreasing_on_interval a → g a ≥ 1 ∧ ∃ a₀, g a₀ = 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3245_324596


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3245_324538

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l3245_324538


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l3245_324518

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) :
  Prime p ∧ p^(x : ℕ) = y^3 + 1 ↔ (p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l3245_324518


namespace NUMINAMATH_CALUDE_fermat_mod_large_prime_l3245_324598

theorem fermat_mod_large_prime (n : ℕ) (hn : n > 0) :
  ∃ M : ℕ, ∀ p : ℕ, p > M → Prime p →
    ∃ x y z : ℤ, (x^n + y^n) % p = z^n % p ∧ (x * y * z) % p ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_fermat_mod_large_prime_l3245_324598


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l3245_324510

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 11x + 4 is 41 -/
theorem quadratic_discriminant : discriminant 5 (-11) 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l3245_324510


namespace NUMINAMATH_CALUDE_no_two_digit_number_exists_l3245_324505

theorem no_two_digit_number_exists : ¬∃ (n : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ (d₁ d₂ : ℕ), 
    d₁ < 10 ∧ d₂ < 10 ∧
    n = 10 * d₁ + d₂ ∧
    n = 2 * (d₁^2 + d₂^2) + 6 ∧
    n = 4 * (d₁ * d₂) + 6) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_number_exists_l3245_324505


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l3245_324542

theorem roots_quadratic_equation (a b : ℝ) : 
  (a^2 + 3*a - 2010 = 0) → 
  (b^2 + 3*b - 2010 = 0) → 
  (a^2 - a - 4*b = 2022) := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l3245_324542


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3245_324586

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 27 →
  r - p = 34 →
  (q + r) / 2 = 27 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3245_324586


namespace NUMINAMATH_CALUDE_george_second_day_hours_l3245_324593

/-- Calculates the hours worked on the second day given the hourly rate, 
    hours worked on the first day, and total earnings for two days. -/
def hoursWorkedSecondDay (hourlyRate : ℚ) (hoursFirstDay : ℚ) (totalEarnings : ℚ) : ℚ :=
  (totalEarnings - hourlyRate * hoursFirstDay) / hourlyRate

/-- Proves that given the specific conditions of the problem, 
    the hours worked on the second day is 2. -/
theorem george_second_day_hours : 
  hoursWorkedSecondDay 5 7 45 = 2 := by
  sorry

end NUMINAMATH_CALUDE_george_second_day_hours_l3245_324593


namespace NUMINAMATH_CALUDE_workday_end_time_l3245_324570

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  hLt24 : hours < 24
  mLt60 : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

def workday_duration : Duration := ⟨8, 0⟩
def start_time : Time := ⟨7, 0, by sorry, by sorry⟩
def lunch_start : Time := ⟨11, 30, by sorry, by sorry⟩
def lunch_duration : Duration := ⟨0, 30⟩

/-- Adds a duration to a time -/
def add_duration (t : Time) (d : Duration) : Time :=
  sorry

/-- Subtracts two times to get a duration -/
def time_difference (t1 t2 : Time) : Duration :=
  sorry

theorem workday_end_time : 
  let time_before_lunch := time_difference lunch_start start_time
  let lunch_end := add_duration lunch_start lunch_duration
  let remaining_work := 
    ⟨workday_duration.hours - time_before_lunch.hours, 
     workday_duration.minutes - time_before_lunch.minutes⟩
  let end_time := add_duration lunch_end remaining_work
  end_time = ⟨15, 30, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_workday_end_time_l3245_324570


namespace NUMINAMATH_CALUDE_rohans_age_is_25_l3245_324545

/-- Rohan's current age in years -/
def rohans_current_age : ℕ := 25

/-- Rohan's age 15 years ago -/
def rohans_past_age : ℕ := rohans_current_age - 15

/-- Rohan's age 15 years from now -/
def rohans_future_age : ℕ := rohans_current_age + 15

/-- Theorem stating that Rohan's current age is 25, given the condition -/
theorem rohans_age_is_25 :
  rohans_current_age = 25 ∧
  rohans_future_age = 4 * rohans_past_age :=
by sorry

end NUMINAMATH_CALUDE_rohans_age_is_25_l3245_324545


namespace NUMINAMATH_CALUDE_friend_contribution_is_eleven_l3245_324579

/-- The amount each friend should contribute when splitting the cost of movie tickets, popcorn, and milk tea. -/
def friend_contribution : ℚ :=
  let num_friends : ℕ := 3
  let ticket_price : ℚ := 7
  let num_tickets : ℕ := 3
  let popcorn_price : ℚ := 3/2  -- $1.5 as a rational number
  let num_popcorn : ℕ := 2
  let milk_tea_price : ℚ := 3
  let num_milk_tea : ℕ := 3
  let total_cost : ℚ := ticket_price * num_tickets + popcorn_price * num_popcorn + milk_tea_price * num_milk_tea
  total_cost / num_friends

theorem friend_contribution_is_eleven :
  friend_contribution = 11 := by
  sorry

end NUMINAMATH_CALUDE_friend_contribution_is_eleven_l3245_324579


namespace NUMINAMATH_CALUDE_no_solution_for_system_l3245_324528

theorem no_solution_for_system :
  ¬∃ (x y z : ℝ), 
    (Real.sqrt (2 * x^2 + 2) = y - 1) ∧
    (Real.sqrt (2 * y^2 + 2) = z - 1) ∧
    (Real.sqrt (2 * z^2 + 2) = x - 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l3245_324528


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l3245_324573

/-- Represents a four-digit number ABCD --/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Checks if a four-digit number contains no nines --/
def no_nines (n : FourDigitNumber) : Prop :=
  ∀ d, d ∈ n.value.digits 10 → d ≠ 9

/-- Extracts the first two digits of a four-digit number --/
def first_two_digits (n : FourDigitNumber) : ℕ := n.value / 100

/-- Extracts the last two digits of a four-digit number --/
def last_two_digits (n : FourDigitNumber) : ℕ := n.value % 100

/-- Extracts the first digit of a four-digit number --/
def first_digit (n : FourDigitNumber) : ℕ := n.value / 1000

/-- Extracts the second digit of a four-digit number --/
def second_digit (n : FourDigitNumber) : ℕ := (n.value / 100) % 10

/-- Extracts the third digit of a four-digit number --/
def third_digit (n : FourDigitNumber) : ℕ := (n.value / 10) % 10

/-- Extracts the fourth digit of a four-digit number --/
def fourth_digit (n : FourDigitNumber) : ℕ := n.value % 10

/-- Checks if a quadratic equation ax² + bx + c = 0 has real roots --/
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem four_digit_number_theorem (n : FourDigitNumber) 
  (h_no_nines : no_nines n)
  (h_eq1 : has_real_roots (first_digit n : ℝ) (second_digit n : ℝ) (last_two_digits n : ℝ))
  (h_eq2 : has_real_roots (first_digit n : ℝ) ((n.value / 10) % 100 : ℝ) (fourth_digit n : ℝ))
  (h_eq3 : has_real_roots (first_two_digits n : ℝ) (third_digit n : ℝ) (fourth_digit n : ℝ))
  (h_leading : first_digit n ≠ 0 ∧ second_digit n ≠ 0) :
  n.value = 1710 ∨ n.value = 1810 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l3245_324573


namespace NUMINAMATH_CALUDE_apple_distribution_l3245_324532

theorem apple_distribution (x : ℕ) (h : x > 0) :
  (1430 / x : ℚ) - (1430 / (x + 45) : ℚ) = 9 → 1430 / x = 22 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3245_324532


namespace NUMINAMATH_CALUDE_red_points_on_function_l3245_324534

/-- A point in the plane is a red point if its x-coordinate is a natural number
    and its y-coordinate is a perfect square. -/
def is_red_point (p : ℝ × ℝ) : Prop :=
  ∃ (n m : ℕ), p.1 = n ∧ p.2 = m^2

/-- The function y = (x-36)(x-144) - 1991 -/
def f (x : ℝ) : ℝ := (x - 36) * (x - 144) - 1991

theorem red_points_on_function :
  ∀ p : ℝ × ℝ, is_red_point p ∧ f p.1 = p.2 ↔ p = (2544, 6017209) ∨ p = (444, 120409) := by
  sorry

end NUMINAMATH_CALUDE_red_points_on_function_l3245_324534


namespace NUMINAMATH_CALUDE_jack_bought_55_apples_l3245_324576

def apples_for_father : Nat := 10
def number_of_friends : Nat := 4
def apples_per_person : Nat := 9

theorem jack_bought_55_apples : 
  (apples_for_father + (number_of_friends + 1) * apples_per_person) = 55 := by
  sorry

end NUMINAMATH_CALUDE_jack_bought_55_apples_l3245_324576


namespace NUMINAMATH_CALUDE_paths_from_A_to_B_is_16_l3245_324507

/-- Represents the number of arrows of each color in the hexagonal lattice --/
structure ArrowCounts where
  red : Nat
  blue : Nat
  green : Nat
  purple : Nat
  orange : Nat

/-- Represents the connection rules between arrows of different colors --/
structure ConnectionRules where
  redToBlue : Nat
  blueToGreen : Nat
  greenToPurple : Nat
  purpleToOrange : Nat
  orangeToB : Nat

/-- Calculates the number of paths from A to B in the hexagonal lattice --/
def pathsFromAToB (counts : ArrowCounts) (rules : ConnectionRules) : Nat :=
  counts.red * rules.redToBlue * counts.blue * rules.blueToGreen * counts.green *
  rules.greenToPurple * counts.purple * rules.purpleToOrange * counts.orange * rules.orangeToB

/-- Theorem stating that the number of paths from A to B is 16 --/
theorem paths_from_A_to_B_is_16 (counts : ArrowCounts) (rules : ConnectionRules) :
  counts.red = 2 ∧ counts.blue = 2 ∧ counts.green = 4 ∧ counts.purple = 4 ∧ counts.orange = 4 ∧
  rules.redToBlue = 2 ∧ rules.blueToGreen = 3 ∧ rules.greenToPurple = 2 ∧
  rules.purpleToOrange = 1 ∧ rules.orangeToB = 1 →
  pathsFromAToB counts rules = 16 := by
  sorry

#check paths_from_A_to_B_is_16

end NUMINAMATH_CALUDE_paths_from_A_to_B_is_16_l3245_324507


namespace NUMINAMATH_CALUDE_overlapping_squares_perimeter_is_120_l3245_324585

/-- The perimeter of a shape formed by five overlapping squares -/
def overlapping_squares_perimeter (side_length : ℝ) : ℝ :=
  3 * (4 * side_length)

/-- Theorem: The perimeter of the shape formed by five overlapping squares
    with side length 10 cm is 120 cm -/
theorem overlapping_squares_perimeter_is_120 :
  overlapping_squares_perimeter 10 = 120 := by
  sorry

#check overlapping_squares_perimeter_is_120

end NUMINAMATH_CALUDE_overlapping_squares_perimeter_is_120_l3245_324585


namespace NUMINAMATH_CALUDE_parallel_condition_l3245_324537

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The line ax + y = 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y = 1

/-- The line x + ay = 2a -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + a * y = 2 * a

/-- The condition a = -1 is sufficient but not necessary for the lines to be parallel -/
theorem parallel_condition (a : ℝ) : 
  (a = -1 → are_parallel (-a) (1/a)) ∧ 
  ¬(are_parallel (-a) (1/a) → a = -1) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3245_324537


namespace NUMINAMATH_CALUDE_roden_fish_count_l3245_324567

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := 15

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The total number of fish Roden bought -/
def total_fish : ℕ := gold_fish + blue_fish

theorem roden_fish_count : total_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_roden_fish_count_l3245_324567


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3245_324524

theorem polynomial_divisibility : ∃ z : Polynomial ℤ, 
  X^44 + X^33 + X^22 + X^11 + 1 = (X^4 + X^3 + X^2 + X + 1) * z :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3245_324524


namespace NUMINAMATH_CALUDE_sebastian_grade_size_l3245_324509

/-- The number of students in a grade where a student is ranked both the 70th best and 70th worst -/
def num_students (rank_best : ℕ) (rank_worst : ℕ) : ℕ :=
  (rank_best - 1) + 1 + (rank_worst - 1)

/-- Theorem stating that if a student is ranked both the 70th best and 70th worst, 
    then there are 139 students in total -/
theorem sebastian_grade_size :
  num_students 70 70 = 139 := by
  sorry

end NUMINAMATH_CALUDE_sebastian_grade_size_l3245_324509


namespace NUMINAMATH_CALUDE_quadratic_sum_l3245_324521

/-- A quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  f a b c 0 = 7 ∧ f a b c 1 = 4 → a + b + 2 * c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3245_324521


namespace NUMINAMATH_CALUDE_reinforcement_size_l3245_324525

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (days_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - initial_garrison * days_passed
  let new_total_men := initial_garrison + (provisions_left / remaining_duration - initial_garrison)
  new_total_men - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l3245_324525


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_bound_negation_l3245_324519

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) := by sorry

theorem power_of_two_bound_negation : 
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_bound_negation_l3245_324519


namespace NUMINAMATH_CALUDE_pool_fill_time_ab_l3245_324527

/-- Represents the time it takes for a valve to fill the pool individually -/
structure ValveTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions given in the problem -/
structure PoolFillConditions where
  vt : ValveTime
  all_valves_time : (1 / vt.a + 1 / vt.b + 1 / vt.c) = 1
  ac_time : (1 / vt.a + 1 / vt.c) * 1.5 = 1
  bc_time : (1 / vt.b + 1 / vt.c) * 2 = 1

/-- Theorem stating that given the conditions, the time to fill the pool with valves A and B is 1.2 hours -/
theorem pool_fill_time_ab (conditions : PoolFillConditions) : 
  (1 / conditions.vt.a + 1 / conditions.vt.b) * 1.2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_ab_l3245_324527


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3245_324533

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 3)*x - k + 6 > 0) ↔ -3 < k ∧ k < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3245_324533


namespace NUMINAMATH_CALUDE_correct_international_letters_l3245_324547

/-- The number of international letters in a mailing scenario. -/
def num_international_letters : ℕ :=
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  2

/-- Proof that the number of international letters is correct. -/
theorem correct_international_letters : 
  let total_letters : ℕ := 4
  let standard_postage : ℚ := 108 / 100  -- $1.08
  let international_charge : ℚ := 14 / 100  -- $0.14
  let total_cost : ℚ := 460 / 100  -- $4.60
  num_international_letters = 2 ∧
  (num_international_letters : ℚ) * (standard_postage + international_charge) + 
  (total_letters - num_international_letters : ℚ) * standard_postage = total_cost := by
  sorry

end NUMINAMATH_CALUDE_correct_international_letters_l3245_324547


namespace NUMINAMATH_CALUDE_hyperbola_inequality_l3245_324511

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 = 1

-- Define the line
def line (x : ℝ) : Prop := x = 3

-- Define the intersection points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (3, -1)

-- Define a point P on the hyperbola
def P (x y : ℝ) : Prop := hyperbola x y

-- Define the vector representation of OP
def OP (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  (∀ x y, P x y → OP a b = (x, y)) → |a + b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_inequality_l3245_324511


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l3245_324531

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 560)
  (h2 : floor_width = 240)
  (h3 : tile_length = 60)
  (h4 : tile_width = 56) : 
  (floor_length / tile_length) * (floor_width / tile_width) ≤ 40 ∧ 
  (floor_length / tile_width) * (floor_width / tile_length) ≤ 40 ∧
  ((floor_length / tile_length) * (floor_width / tile_width) = 40 ∨
   (floor_length / tile_width) * (floor_width / tile_length) = 40) :=
by sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l3245_324531


namespace NUMINAMATH_CALUDE_integer_solutions_for_system_l3245_324503

theorem integer_solutions_for_system (x y : ℤ) : 
  x^2 = (y+1)^2 + 1 ∧ 
  x^2 - (y+1)^2 = 1 ∧ 
  (x-y-1) * (x+y+1) = 1 → 
  (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_for_system_l3245_324503


namespace NUMINAMATH_CALUDE_fraction_equality_implies_product_l3245_324569

theorem fraction_equality_implies_product (a b : ℝ) : 
  a / 2 = 3 / b → a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_product_l3245_324569


namespace NUMINAMATH_CALUDE_correct_average_marks_l3245_324515

/-- Calculates the correct average marks for a class given the reported average, 
    number of students, and corrections for three students' marks. -/
def correctAverageMarks (reportedAverage : ℚ) (numStudents : ℕ) 
    (wrongMark1 wrongMark2 wrongMark3 : ℚ) 
    (correctMark1 correctMark2 correctMark3 : ℚ) : ℚ :=
  let incorrectTotal := reportedAverage * numStudents
  let wronglyNotedMarks := wrongMark1 + wrongMark2 + wrongMark3
  let correctMarks := correctMark1 + correctMark2 + correctMark3
  let correctTotal := incorrectTotal - wronglyNotedMarks + correctMarks
  correctTotal / numStudents

/-- The correct average marks for the class are 63.125 -/
theorem correct_average_marks :
  correctAverageMarks 65 40 100 85 15 20 50 55 = 63.125 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l3245_324515


namespace NUMINAMATH_CALUDE_perpendicular_vector_solution_l3245_324551

def direction_vector : ℝ × ℝ := (2, 1)

theorem perpendicular_vector_solution :
  ∃! v : ℝ × ℝ, v.1 + v.2 = 1 ∧ v.1 * direction_vector.1 + v.2 * direction_vector.2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vector_solution_l3245_324551


namespace NUMINAMATH_CALUDE_eliminated_team_size_is_21_l3245_324568

/-- Represents a team in the competition -/
structure Team where
  size : ℕ
  is_girls : Bool

/-- Represents the state of the competition -/
structure Competition where
  teams : List Team
  eliminated_team_size : ℕ

def Competition.remaining_teams (c : Competition) : List Team :=
  c.teams.filter (λ t => t.size ≠ c.eliminated_team_size)

def Competition.total_players (c : Competition) : ℕ :=
  c.teams.map (λ t => t.size) |>.sum

def Competition.remaining_players (c : Competition) : ℕ :=
  c.total_players - c.eliminated_team_size

def Competition.boys_count (c : Competition) : ℕ :=
  c.remaining_teams.filter (λ t => ¬t.is_girls) |>.map (λ t => t.size) |>.sum

def Competition.girls_count (c : Competition) : ℕ :=
  c.remaining_players - c.boys_count

theorem eliminated_team_size_is_21 (c : Competition) : c.eliminated_team_size = 21 :=
  by
  have team_sizes : c.teams.map (λ t => t.size) = [9, 15, 17, 19, 21] := sorry
  have total_five_teams : c.teams.length = 5 := sorry
  have eliminated_is_girls : c.teams.filter (λ t => t.size = c.eliminated_team_size) |>.all (λ t => t.is_girls) := sorry
  have remaining_girls_triple_boys : c.girls_count = 3 * c.boys_count := sorry
  sorry

#check eliminated_team_size_is_21

end NUMINAMATH_CALUDE_eliminated_team_size_is_21_l3245_324568


namespace NUMINAMATH_CALUDE_unique_coin_combination_l3245_324502

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ :=
  14 + coins.ten + 4 * coins.twentyFive

/-- The main theorem -/
theorem unique_coin_combination :
  ∀ (coins : CoinCounts),
    coins.five + coins.ten + coins.twentyFive = 15 →
    differentValues coins = 21 →
    coins.twentyFive = 1 := by
  sorry

#check unique_coin_combination

end NUMINAMATH_CALUDE_unique_coin_combination_l3245_324502


namespace NUMINAMATH_CALUDE_cindys_envelopes_l3245_324575

/-- Cindy's envelope problem -/
theorem cindys_envelopes (initial_envelopes : ℕ) (friends : ℕ) (envelopes_per_friend : ℕ) :
  initial_envelopes = 37 →
  friends = 5 →
  envelopes_per_friend = 3 →
  initial_envelopes - friends * envelopes_per_friend = 22 := by
  sorry

#check cindys_envelopes

end NUMINAMATH_CALUDE_cindys_envelopes_l3245_324575


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3245_324556

theorem complex_power_magnitude : Complex.abs ((3 / 4 : ℂ) + (5 / 4 : ℂ) * Complex.I) ^ 4 = 289 / 64 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3245_324556


namespace NUMINAMATH_CALUDE_remainder_of_M_mod_500_l3245_324578

/-- The number of consecutive 0's at the right end of the decimal representation of n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of factorials from 1 to n -/
def factorialProduct (n : ℕ) : ℕ := sorry

/-- M is the number of consecutive 0's at the right end of the decimal representation of 1!2!3!4!...49!50! -/
def M : ℕ := trailingZeros (factorialProduct 50)

theorem remainder_of_M_mod_500 : M % 500 = 12 := by sorry

end NUMINAMATH_CALUDE_remainder_of_M_mod_500_l3245_324578


namespace NUMINAMATH_CALUDE_restricted_arrangements_l3245_324571

/-- The number of ways to arrange n people in a row. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row with one person fixed at the left end. -/
def permutations_with_left_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with one person fixed at the right end. -/
def permutations_with_right_fixed (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange n people in a row with two people fixed at both ends. -/
def permutations_with_both_ends_fixed (n : ℕ) : ℕ := Nat.factorial (n - 2)

theorem restricted_arrangements (n : ℕ) (h : n = 4) : 
  permutations n - permutations_with_left_fixed n - permutations_with_right_fixed n + permutations_with_both_ends_fixed n = 2 :=
sorry

end NUMINAMATH_CALUDE_restricted_arrangements_l3245_324571


namespace NUMINAMATH_CALUDE_circles_configuration_l3245_324544

-- Define the centers of the circles as points in a metric space
variable (X : Type) [MetricSpace X]
variable (P Q R : X)

-- Define the radii of the circles
variable (p q r : ℝ)

-- Define the distance between P and Q
variable (d : ℝ)

-- State the theorem
theorem circles_configuration (h1 : p > q) (h2 : q > r) 
  (h3 : dist R P < p) (h4 : dist R Q < q) (h5 : d = dist P Q) :
  ¬(p + r = d) := by
  sorry

end NUMINAMATH_CALUDE_circles_configuration_l3245_324544


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3245_324582

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (k : ℕ), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : ℕ), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem problem_solution :
  ∃ (k : ℕ), k = 5 ∧ (3830 - k) % 15 = 0 ∧ ∀ (m : ℕ), m < k → (3830 - m) % 15 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_problem_solution_l3245_324582


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l3245_324591

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The line intersects the ellipse at two distinct points -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ m ∧ line x₂ y₂ m

/-- Main theorem -/
theorem ellipse_line_intersection (m : ℝ) :
  intersects_at_two_points m ↔ m ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l3245_324591


namespace NUMINAMATH_CALUDE_set_conditions_equivalence_l3245_324506

theorem set_conditions_equivalence (m : ℝ) :
  let A := {x : ℝ | 0 < x - m ∧ x - m < 2}
  let B := {x : ℝ | -x^2 + 3*x ≤ 0}
  (A ∩ B = ∅ ∧ A ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_set_conditions_equivalence_l3245_324506


namespace NUMINAMATH_CALUDE_lloyd_house_of_cards_solution_l3245_324574

/-- Represents the number of cards in Lloyd's house of cards problem -/
def lloyd_house_of_cards (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) : ℕ :=
  (decks * cards_per_deck) / layers

/-- Theorem stating the number of cards per layer in Lloyd's house of cards -/
theorem lloyd_house_of_cards_solution :
  lloyd_house_of_cards 24 78 48 = 39 := by
  sorry

#eval lloyd_house_of_cards 24 78 48

end NUMINAMATH_CALUDE_lloyd_house_of_cards_solution_l3245_324574


namespace NUMINAMATH_CALUDE_function_passes_through_point_l3245_324554

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 3)
  f 3 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l3245_324554


namespace NUMINAMATH_CALUDE_cube_sum_product_l3245_324550

theorem cube_sum_product : ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 := by sorry

end NUMINAMATH_CALUDE_cube_sum_product_l3245_324550


namespace NUMINAMATH_CALUDE_function_increasing_l3245_324517

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_l3245_324517


namespace NUMINAMATH_CALUDE_dorothy_found_57_pieces_l3245_324539

/-- The number of sea glass pieces found by Dorothy -/
def dorothy_total (blanche_green blanche_red rose_red rose_blue : ℕ) : ℕ :=
  let dorothy_red := 2 * (blanche_red + rose_red)
  let dorothy_blue := 3 * rose_blue
  dorothy_red + dorothy_blue

/-- Theorem stating that Dorothy found 57 pieces of sea glass -/
theorem dorothy_found_57_pieces :
  dorothy_total 12 3 9 11 = 57 := by
  sorry

#eval dorothy_total 12 3 9 11

end NUMINAMATH_CALUDE_dorothy_found_57_pieces_l3245_324539


namespace NUMINAMATH_CALUDE_distance_on_line_l3245_324546

/-- The distance between two points (5, b) and (10, d) on the line y = 2x + 3 is 5√5. -/
theorem distance_on_line : ∀ b d : ℝ,
  b = 2 * 5 + 3 →
  d = 2 * 10 + 3 →
  Real.sqrt ((10 - 5)^2 + (d - b)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_line_l3245_324546


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3245_324500

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a grid of width w and height h -/
def rectangles_in_grid (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  rectangles_in_grid 5 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3245_324500


namespace NUMINAMATH_CALUDE_concentric_circles_diameter_l3245_324520

theorem concentric_circles_diameter 
  (r₁ r₂ : ℝ) 
  (h₁ : r₁ > 0) 
  (h₂ : r₂ > r₁) 
  (area_small : π * r₁^2 = 4*π) 
  (area_ring : π * r₂^2 - π * r₁^2 = 4*π) : 
  2 * r₂ = 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_diameter_l3245_324520


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3245_324597

/-- Given a point P with coordinates (3a-6, 1-a) that lies on the x-axis, 
    prove that its coordinates are (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = 3*a - 6 ∧ P.2 = 1 - a ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (-3, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3245_324597


namespace NUMINAMATH_CALUDE_open_box_volume_l3245_324583

/-- The volume of an open box constructed from a rectangular sheet --/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume :
  ∀ x : ℝ, 1 ≤ x → x ≤ 3 →
  boxVolume 16 12 x = 4*x^3 - 56*x^2 + 192*x :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3245_324583


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l3245_324560

/-- 
Represents the candy distribution function.
For a given number of students n and a position i,
it returns the number of candies given to the student at position i.
-/
def candy_distribution (n : ℕ) (i : ℕ) : ℕ :=
  sorry

/-- 
Checks if every student receives at least one candy
for a given number of students n.
-/
def every_student_gets_candy (n : ℕ) : Prop :=
  sorry

/-- 
Checks if a given natural number is a power of 2.
-/
def is_power_of_two (n : ℕ) : Prop :=
  sorry

/-- 
Theorem: For n ≥ 2, every student receives at least one candy
if and only if n is a power of 2.
-/
theorem candy_distribution_theorem (n : ℕ) (h : n ≥ 2) :
  every_student_gets_candy n ↔ is_power_of_two n :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l3245_324560
