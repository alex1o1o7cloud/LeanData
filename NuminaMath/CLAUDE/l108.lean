import Mathlib

namespace NUMINAMATH_CALUDE_integral_reciprocal_x_xplus1_l108_10842

theorem integral_reciprocal_x_xplus1 : 
  ∫ x in (1 : ℝ)..2, 1 / (x * (x + 1)) = Real.log (4 / 3) := by sorry

end NUMINAMATH_CALUDE_integral_reciprocal_x_xplus1_l108_10842


namespace NUMINAMATH_CALUDE_cost_per_meat_type_l108_10802

/-- Calculates the cost per type of sliced meat in a 4-pack with rush delivery --/
theorem cost_per_meat_type (base_cost : ℝ) (rush_delivery_rate : ℝ) (num_types : ℕ) :
  base_cost = 40 →
  rush_delivery_rate = 0.3 →
  num_types = 4 →
  (base_cost + base_cost * rush_delivery_rate) / num_types = 13 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_meat_type_l108_10802


namespace NUMINAMATH_CALUDE_third_vertex_y_coord_value_l108_10841

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle with two vertices given -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point
  is_equilateral : True  -- This is a placeholder for the equilateral property
  third_vertex_in_first_quadrant : v3.x > 0 ∧ v3.y > 0

/-- The y-coordinate of the third vertex of an equilateral triangle -/
def third_vertex_y_coord (t : EquilateralTriangle) : ℝ :=
  t.v3.y

/-- The theorem stating the y-coordinate of the third vertex -/
theorem third_vertex_y_coord_value (t : EquilateralTriangle) 
    (h1 : t.v1 = ⟨2, 3⟩) 
    (h2 : t.v2 = ⟨10, 3⟩) : 
  third_vertex_y_coord t = 3 + 4 * Real.sqrt 3 := by
  sorry

#check third_vertex_y_coord_value

end NUMINAMATH_CALUDE_third_vertex_y_coord_value_l108_10841


namespace NUMINAMATH_CALUDE_a_5_value_l108_10863

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a_5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 = -9 →
  a 7 = -1 →
  a 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_a_5_value_l108_10863


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_straight_lines_l108_10893

/-- The equation representing the graph -/
def equation (x y : ℝ) : Prop := 9 * x^2 - y^2 - 6 * x = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The theorem stating that the equation represents a pair of straight lines -/
theorem equation_represents_pair_of_straight_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_straight_lines_l108_10893


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l108_10874

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {1, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l108_10874


namespace NUMINAMATH_CALUDE_raffle_donation_calculation_l108_10889

theorem raffle_donation_calculation (num_tickets : ℕ) (ticket_price : ℚ) 
  (total_raised : ℚ) (fixed_donation : ℚ) :
  num_tickets = 25 →
  ticket_price = 2 →
  total_raised = 100 →
  fixed_donation = 20 →
  ∃ (equal_donation : ℚ),
    equal_donation * 2 + fixed_donation = total_raised - (num_tickets : ℚ) * ticket_price ∧
    equal_donation = 15 := by
  sorry

end NUMINAMATH_CALUDE_raffle_donation_calculation_l108_10889


namespace NUMINAMATH_CALUDE_circle_equation_to_circle_params_l108_10867

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0. -/
def CircleEquation (a b c d e : ℝ) : (ℝ × ℝ) → Prop :=
  fun p => a * p.1^2 + b * p.2^2 + c * p.1 + d * p.2 + e = 0

theorem circle_equation_to_circle_params :
  ∃! (circle : Circle),
    (∀ p, CircleEquation 1 1 (-4) 2 0 p ↔ (p.1 - circle.center.1)^2 + (p.2 - circle.center.2)^2 = circle.radius^2) ∧
    circle.center = (2, -1) ∧
    circle.radius = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_to_circle_params_l108_10867


namespace NUMINAMATH_CALUDE_line_circle_intersection_a_eq_one_l108_10879

/-- A line intersecting a circle forming a right triangle -/
structure LineCircleIntersection where
  a : ℝ
  -- Line equation: ax - y + 6 = 0
  line : ℝ → ℝ → Prop := fun x y ↦ a * x - y + 6 = 0
  -- Circle equation: (x + 1)^2 + (y - a)^2 = 16
  circle : ℝ → ℝ → Prop := fun x y ↦ (x + 1)^2 + (y - a)^2 = 16
  -- Circle center
  center : ℝ × ℝ := (-1, a)
  -- Existence of intersection points A and B
  A : ℝ × ℝ
  B : ℝ × ℝ
  hA : line A.1 A.2 ∧ circle A.1 A.2
  hB : line B.1 B.2 ∧ circle B.1 B.2
  -- Triangle ABC is a right triangle
  hRight : (A.1 - B.1) * (center.1 - B.1) + (A.2 - B.2) * (center.2 - B.2) = 0

/-- The positive value of a in the LineCircleIntersection is 1 -/
theorem line_circle_intersection_a_eq_one (lci : LineCircleIntersection) : 
  lci.a > 0 → lci.a = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_a_eq_one_l108_10879


namespace NUMINAMATH_CALUDE_three_by_three_min_cuts_l108_10806

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a straight-line cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut

/-- Defines the minimum number of cuts required to divide a grid into unit squares -/
def min_cuts (g : Grid) : ℕ := sorry

/-- Theorem stating that a 3x3 grid requires exactly 4 cuts to be divided into unit squares -/
theorem three_by_three_min_cuts :
  ∀ (g : Grid), g.size = 3 → min_cuts g = 4 := by sorry

end NUMINAMATH_CALUDE_three_by_three_min_cuts_l108_10806


namespace NUMINAMATH_CALUDE_min_value_of_f_l108_10830

/-- The quadratic function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x, f x = m) ∧ (m = -44) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l108_10830


namespace NUMINAMATH_CALUDE_unique_divisible_by_1375_l108_10854

theorem unique_divisible_by_1375 : 
  ∃! n : ℕ, 
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = 700000 + 10000 * x + 3600 + 10 * y + 5) ∧ 
    n % 1375 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_1375_l108_10854


namespace NUMINAMATH_CALUDE_inverse_sum_mod_11_l108_10844

theorem inverse_sum_mod_11 : 
  (((2⁻¹ : ZMod 11) + (6⁻¹ : ZMod 11) + (10⁻¹ : ZMod 11))⁻¹ : ZMod 11) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_11_l108_10844


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l108_10876

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that the reflection of (1, 6) across the y-axis is (-1, 6) -/
theorem reflection_across_y_axis :
  let original := Point.mk 1 6
  reflect_y original = Point.mk (-1) 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l108_10876


namespace NUMINAMATH_CALUDE_jill_peaches_l108_10847

theorem jill_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ)
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end NUMINAMATH_CALUDE_jill_peaches_l108_10847


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l108_10871

/-- Given a geometric sequence with first term a₁ = 2, 
    the smallest possible value of 6a₂ + 7a₃ is -18/7 -/
theorem min_value_geometric_sequence (a₁ a₂ a₃ : ℝ) : 
  a₁ = 2 → 
  (∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) → 
  (∀ b₂ b₃ : ℝ, (∃ s : ℝ, b₂ = a₁ * s ∧ b₃ = b₂ * s) → 
    6 * a₂ + 7 * a₃ ≤ 6 * b₂ + 7 * b₃) → 
  6 * a₂ + 7 * a₃ = -18/7 :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l108_10871


namespace NUMINAMATH_CALUDE_product_sequence_sum_l108_10800

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l108_10800


namespace NUMINAMATH_CALUDE_triangle_internal_point_theorem_l108_10846

/-- Triangle with sides a, b, c and internal point P --/
structure TriangleWithInternalPoint where
  a : ℝ
  b : ℝ
  c : ℝ
  P : ℝ × ℝ

/-- Parallel segments through P have equal length d --/
def parallelSegmentsEqual (T : TriangleWithInternalPoint) (d : ℝ) : Prop :=
  ∃ (x y z : ℝ), x + y + z = T.a ∧ x + y + z = T.b ∧ x + y + z = T.c ∧ x = y ∧ y = z ∧ z = d

theorem triangle_internal_point_theorem (T : TriangleWithInternalPoint) 
    (h1 : T.a = 550) (h2 : T.b = 580) (h3 : T.c = 620) :
    ∃ (d : ℝ), parallelSegmentsEqual T d ∧ d = 342 := by
  sorry

end NUMINAMATH_CALUDE_triangle_internal_point_theorem_l108_10846


namespace NUMINAMATH_CALUDE_horner_v1_value_l108_10821

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

def horner_v1 (x : ℝ) : ℝ := x * 1 - 5

theorem horner_v1_value :
  horner_v1 (-2) = -7 :=
by sorry

end NUMINAMATH_CALUDE_horner_v1_value_l108_10821


namespace NUMINAMATH_CALUDE_freelancer_earnings_l108_10898

def calculate_final_amount (initial_amount : ℚ) : ℚ :=
  let first_client_payment := initial_amount / 2
  let second_client_payment := first_client_payment * (1 + 2/5)
  let third_client_payment := 2 * (first_client_payment + second_client_payment)
  let average_first_three := (first_client_payment + second_client_payment + third_client_payment) / 3
  let fourth_client_payment := average_first_three * (1 + 1/10)
  initial_amount + first_client_payment + second_client_payment + third_client_payment + fourth_client_payment

theorem freelancer_earnings (initial_amount : ℚ) :
  initial_amount = 4000 → calculate_final_amount initial_amount = 23680 :=
by sorry

end NUMINAMATH_CALUDE_freelancer_earnings_l108_10898


namespace NUMINAMATH_CALUDE_some_students_not_fraternity_members_l108_10870

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Honest : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the given conditions
axiom some_students_not_honest : ∃ x, Student x ∧ ¬Honest x
axiom all_fraternity_members_honest : ∀ x, FraternityMember x → Honest x

-- Theorem to prove
theorem some_students_not_fraternity_members : 
  ∃ x, Student x ∧ ¬FraternityMember x :=
sorry

end NUMINAMATH_CALUDE_some_students_not_fraternity_members_l108_10870


namespace NUMINAMATH_CALUDE_power_of_product_l108_10883

theorem power_of_product (a b : ℝ) : (-5 * a^3 * b)^2 = 25 * a^6 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l108_10883


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_school_l108_10843

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a sampling strategy -/
structure Sampling :=
  (total_population : ℕ)
  (sample_size : ℕ)
  (schools : Vector School 3)

/-- Checks if the number of students in schools forms an arithmetic sequence -/
def is_arithmetic_sequence (schools : Vector School 3) : Prop :=
  schools[1].students - schools[0].students = schools[2].students - schools[1].students

/-- Theorem: In a stratified sampling of 120 students from 1500 students 
    distributed in an arithmetic sequence across 3 schools, 
    the number of students sampled from the middle school (B) is 40 -/
theorem stratified_sampling_middle_school 
  (sampling : Sampling) 
  (h1 : sampling.total_population = 1500)
  (h2 : sampling.sample_size = 120)
  (h3 : is_arithmetic_sequence sampling.schools)
  : (sampling.sample_size / 3 : ℕ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_school_l108_10843


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l108_10852

theorem consecutive_integers_sum (x : ℤ) : 
  x + 1 < 20 → 
  x * (x + 1) + x + (x + 1) = 156 → 
  x + (x + 1) = 23 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l108_10852


namespace NUMINAMATH_CALUDE_chess_tournament_games_l108_10849

/-- Calculate the number of games in a round-robin tournament stage -/
def gamesInRoundRobin (n : ℕ) : ℕ := n * (n - 1)

/-- Calculate the number of games in a knockout tournament stage -/
def gamesInKnockout (n : ℕ) : ℕ := n - 1

/-- The total number of games in the chess tournament -/
def totalGames : ℕ :=
  gamesInRoundRobin 20 + gamesInRoundRobin 10 + gamesInKnockout 4

theorem chess_tournament_games :
  totalGames = 474 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l108_10849


namespace NUMINAMATH_CALUDE_clock_angle_at_four_l108_10872

/-- The number of degrees in a complete circle -/
def circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour mark on a clock -/
def degrees_per_hour : ℕ := circle_degrees / clock_hours

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- The smaller angle formed by the clock hands at the target hour -/
def smaller_angle (h : ℕ) : ℕ := min (h * degrees_per_hour) (circle_degrees - h * degrees_per_hour)

theorem clock_angle_at_four :
  smaller_angle target_hour = 120 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_four_l108_10872


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l108_10894

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 10) 
  (h2 : a^2 + b^2 = 210) : 
  a * b = 55 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l108_10894


namespace NUMINAMATH_CALUDE_all_statements_false_l108_10875

theorem all_statements_false :
  (¬ ∀ a b : ℝ, a > b → a^2 > b^2) ∧
  (¬ ∀ a b : ℝ, a^2 > b^2 → a > b) ∧
  (¬ ∀ a b c : ℝ, a > b → a*c^2 > b*c^2) ∧
  (¬ ∀ a b : ℝ, (a > b ↔ |a| > |b|)) :=
by sorry

end NUMINAMATH_CALUDE_all_statements_false_l108_10875


namespace NUMINAMATH_CALUDE_susie_score_l108_10813

/-- Calculates the total score in a math contest given the number of correct, incorrect, and unanswered questions. -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * (correct : ℤ) - (incorrect : ℤ)

/-- Theorem stating that Susie's score in the math contest is 20 points. -/
theorem susie_score : calculateScore 15 10 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_susie_score_l108_10813


namespace NUMINAMATH_CALUDE_complex_minimum_value_l108_10820

theorem complex_minimum_value (z : ℂ) (h : Complex.abs (z - (5 + I)) = 5) :
  Complex.abs (z - (1 - 2*I))^2 + Complex.abs (z - (9 + 4*I))^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_complex_minimum_value_l108_10820


namespace NUMINAMATH_CALUDE_symmetric_points_line_intercept_l108_10814

/-- Given two points A and B symmetric with respect to a line y = kx + b,
    prove that the x-intercept of the line is 5/6. -/
theorem symmetric_points_line_intercept 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 3)) 
  (h_B : B = (-2, 1)) 
  (k b : ℝ) 
  (h_symmetric : B = (2 * ((k * A.1 + b) / (1 + k^2) - k * A.2 / (1 + k^2)) - A.1,
                      2 * (k * (k * A.1 + b) / (1 + k^2) + A.2 / (1 + k^2)) - A.2)) :
  (- b / k : ℝ) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_line_intercept_l108_10814


namespace NUMINAMATH_CALUDE_regression_lines_intersect_l108_10801

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passingPoint (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

/-- Theorem: Two regression lines with the same average x and y values intersect at (a, b) -/
theorem regression_lines_intersect (l₁ l₂ : RegressionLine) (a b : ℝ) 
  (h₁ : passingPoint l₁ a = (a, b))
  (h₂ : passingPoint l₂ a = (a, b)) :
  ∃ (x y : ℝ), passingPoint l₁ x = (x, y) ∧ passingPoint l₂ x = (x, y) ∧ x = a ∧ y = b :=
sorry

end NUMINAMATH_CALUDE_regression_lines_intersect_l108_10801


namespace NUMINAMATH_CALUDE_jack_first_half_time_l108_10837

/-- Jack and Jill's hill race problem -/
theorem jack_first_half_time (jill_finish_time jack_second_half_time : ℕ)
  (h1 : jill_finish_time = 32)
  (h2 : jack_second_half_time = 6) :
  let jack_finish_time := jill_finish_time - 7
  jack_finish_time - jack_second_half_time = 19 := by
  sorry

end NUMINAMATH_CALUDE_jack_first_half_time_l108_10837


namespace NUMINAMATH_CALUDE_fourth_sample_is_31_l108_10884

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  known_samples : Finset Nat

/-- Calculates the sample interval for a systematic sampling. -/
def sample_interval (s : SystematicSampling) : Nat :=
  s.total_students / s.sample_size

/-- Theorem: In a systematic sampling of 4 from 56 students, if 3, 17, and 45 are sampled, the fourth sample is 31. -/
theorem fourth_sample_is_31 (s : SystematicSampling) 
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = {3, 17, 45}) :
  ∃ (fourth_sample : Nat), fourth_sample ∈ s.known_samples ∪ {31} ∧ 
  (s.known_samples ∪ {fourth_sample}).card = s.sample_size :=
by
  sorry


end NUMINAMATH_CALUDE_fourth_sample_is_31_l108_10884


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_two_roots_l108_10868

theorem at_least_one_quadratic_has_two_roots (p q₁ q₂ : ℝ) (h : p = q₁ + q₂ + 1) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + x₁ + q₁ = 0 ∧ x₂^2 + x₂ + q₁ = 0) ∨
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + p*y₁ + q₂ = 0 ∧ y₂^2 + p*y₂ + q₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_two_roots_l108_10868


namespace NUMINAMATH_CALUDE_chess_players_lost_to_ai_l108_10873

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 120 →
  never_lost_fraction = 2 / 5 →
  (total_players : ℚ) * (1 - never_lost_fraction) = 72 := by
  sorry

end NUMINAMATH_CALUDE_chess_players_lost_to_ai_l108_10873


namespace NUMINAMATH_CALUDE_circle_standard_equation_l108_10805

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points A and B
def A : ℝ × ℝ := (1, -5)
def B : ℝ × ℝ := (2, -2)

-- Define the line equation
def line_equation (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

-- Define the circle equation
def circle_equation (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_standard_equation :
  ∃ (c : Circle),
    circle_equation c A ∧
    circle_equation c B ∧
    line_equation c.center ∧
    c.center = (-3, -2) ∧
    c.radius = 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_standard_equation_l108_10805


namespace NUMINAMATH_CALUDE_trig_identity_proof_l108_10808

theorem trig_identity_proof : 
  Real.cos (15 * π / 180) * Real.cos (105 * π / 180) - 
  Real.cos (75 * π / 180) * Real.sin (105 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l108_10808


namespace NUMINAMATH_CALUDE_root_implies_k_value_l108_10856

theorem root_implies_k_value (k : ℝ) : 
  (3 : ℝ)^4 + k * (3 : ℝ)^2 + 27 = 0 → k = -12 := by
sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l108_10856


namespace NUMINAMATH_CALUDE_fraction_simplification_l108_10866

theorem fraction_simplification :
  (3/7 + 5/8) / (5/12 + 1/3) = 59/42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l108_10866


namespace NUMINAMATH_CALUDE_expansion_coefficient_implies_k_value_l108_10840

theorem expansion_coefficient_implies_k_value (k : ℕ+) :
  (15 * k ^ 4 : ℕ) < 120 → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_implies_k_value_l108_10840


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l108_10807

theorem arithmetic_progression_sum (a d : ℚ) :
  (let S₂₀ := 20 * (2 * a + 19 * d) / 2
   let S₅₀ := 50 * (2 * a + 49 * d) / 2
   let S₇₀ := 70 * (2 * a + 69 * d) / 2
   S₂₀ = 200 ∧ S₅₀ = 150) →
  S₇₀ = -350 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l108_10807


namespace NUMINAMATH_CALUDE_investment_ratio_l108_10828

/-- Given two partners p and q, their profit ratio, and investment times, 
    prove the ratio of their investments. -/
theorem investment_ratio 
  (profit_ratio_p profit_ratio_q : ℚ) 
  (investment_time_p investment_time_q : ℚ) 
  (profit_ratio_constraint : profit_ratio_p / profit_ratio_q = 7 / 10)
  (time_constraint_p : investment_time_p = 8)
  (time_constraint_q : investment_time_q = 16) :
  ∃ (investment_p investment_q : ℚ),
    investment_p / investment_q = 7 / 5 ∧
    profit_ratio_p / profit_ratio_q = 
      (investment_p * investment_time_p) / (investment_q * investment_time_q) :=
by sorry

end NUMINAMATH_CALUDE_investment_ratio_l108_10828


namespace NUMINAMATH_CALUDE_candy_bar_sales_l108_10825

/-- The number of additional candy bars sold each day -/
def additional_candy_bars : ℕ := sorry

/-- The cost of each candy bar in cents -/
def candy_bar_cost : ℕ := 10

/-- The number of days Sol sells candy bars in a week -/
def selling_days : ℕ := 6

/-- The number of candy bars sold on the first day -/
def first_day_sales : ℕ := 10

/-- The total earnings in cents for the week -/
def total_earnings : ℕ := 1200

theorem candy_bar_sales :
  (first_day_sales * selling_days + 
   additional_candy_bars * (selling_days * (selling_days - 1) / 2)) * 
  candy_bar_cost = total_earnings :=
sorry

end NUMINAMATH_CALUDE_candy_bar_sales_l108_10825


namespace NUMINAMATH_CALUDE_gordon_restaurants_weekly_meals_l108_10888

theorem gordon_restaurants_weekly_meals :
  let first_restaurant_daily_meals : ℕ := 20
  let second_restaurant_daily_meals : ℕ := 40
  let third_restaurant_daily_meals : ℕ := 50
  let days_in_week : ℕ := 7
  (first_restaurant_daily_meals + second_restaurant_daily_meals + third_restaurant_daily_meals) * days_in_week = 770 := by
  sorry

end NUMINAMATH_CALUDE_gordon_restaurants_weekly_meals_l108_10888


namespace NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l108_10881

theorem zinc_copper_mixture_weight 
  (zinc_weight : Real) 
  (zinc_copper_ratio : Real) 
  (h1 : zinc_weight = 28.8) 
  (h2 : zinc_copper_ratio = 9 / 11) : 
  zinc_weight + (zinc_weight * (1 / zinc_copper_ratio)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_zinc_copper_mixture_weight_l108_10881


namespace NUMINAMATH_CALUDE_complex_number_properties_l108_10818

/-- The complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 4*m) (m^2 - m - 6)

/-- Predicate for a complex number being in the third quadrant -/
def in_third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

/-- Predicate for a complex number being on the imaginary axis -/
def on_imaginary_axis (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Predicate for a complex number being on the line x - y + 3 = 0 -/
def on_line (z : ℂ) : Prop := z.re - z.im + 3 = 0

theorem complex_number_properties (m : ℝ) :
  (in_third_quadrant (z m) ↔ 0 < m ∧ m < 3) ∧
  (on_imaginary_axis (z m) ↔ m = 0 ∨ m = 4) ∧
  (on_line (z m) ↔ m = 3) := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l108_10818


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l108_10853

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l108_10853


namespace NUMINAMATH_CALUDE_lakeside_volleyball_club_players_l108_10811

/-- The number of players in the Lakeside Volleyball Club -/
def num_players : ℕ := 80

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 10

/-- The additional cost of a uniform compared to a pair of shoes in dollars -/
def uniform_additional_cost : ℕ := 15

/-- The total expenditure for all gear in dollars -/
def total_expenditure : ℕ := 5600

/-- Theorem stating that the number of players in the Lakeside Volleyball Club is 80 -/
theorem lakeside_volleyball_club_players :
  num_players = (total_expenditure / (2 * (shoe_cost + (shoe_cost + uniform_additional_cost)))) :=
by sorry

end NUMINAMATH_CALUDE_lakeside_volleyball_club_players_l108_10811


namespace NUMINAMATH_CALUDE_walnut_distribution_l108_10839

/-- The total number of walnuts -/
def total_walnuts : ℕ := 55

/-- The number of walnuts in the first pile -/
def first_pile : ℕ := 7

/-- The number of walnuts in each of the other piles -/
def other_piles : ℕ := 12

/-- The number of piles -/
def num_piles : ℕ := 5

theorem walnut_distribution :
  (num_piles - 1) * other_piles + first_pile = total_walnuts ∧
  ∃ (equal_walnuts : ℕ), equal_walnuts * num_piles = total_walnuts :=
by sorry

end NUMINAMATH_CALUDE_walnut_distribution_l108_10839


namespace NUMINAMATH_CALUDE_line_x_eq_1_properties_l108_10833

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-axis in the 2D plane -/
def x_axis : Line := { a := 0, b := 1, c := 0 }

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem line_x_eq_1_properties :
  ∃ (l : Line),
    (∀ (x y : ℝ), l.passes_through (x, y) ↔ x = 1) ∧
    l.passes_through (1, 2) ∧
    l.perpendicular x_axis := by
  sorry

end NUMINAMATH_CALUDE_line_x_eq_1_properties_l108_10833


namespace NUMINAMATH_CALUDE_bag_balls_count_l108_10838

theorem bag_balls_count (red_balls : ℕ) (white_balls : ℕ) (prob_red : ℚ) : 
  red_balls = 4 → 
  prob_red = 1/4 → 
  prob_red = red_balls / (red_balls + white_balls) →
  white_balls = 12 := by
sorry

end NUMINAMATH_CALUDE_bag_balls_count_l108_10838


namespace NUMINAMATH_CALUDE_value_of_a_l108_10836

theorem value_of_a (x a : ℝ) : (x + 1) * (x - 3) = x^2 + a*x - 3 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l108_10836


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l108_10826

/-- Given a quadratic equation 10x^2 + 15x - 25 = 0, 
    the sum of the squares of its roots is equal to 29/4 -/
theorem sum_of_squares_of_roots : 
  let a : ℚ := 10
  let b : ℚ := 15
  let c : ℚ := -25
  let x₁ : ℚ := (-b + (b^2 - 4*a*c).sqrt) / (2*a)
  let x₂ : ℚ := (-b - (b^2 - 4*a*c).sqrt) / (2*a)
  x₁^2 + x₂^2 = 29/4 := by
sorry


end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l108_10826


namespace NUMINAMATH_CALUDE_sum_products_sides_projections_equality_l108_10832

/-- Represents a convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Calculates the sum of products of side lengths and projected widths -/
def sumProductsSidesProjections (P Q : ConvexPolygon) : ℝ :=
  -- Placeholder definition
  0

/-- Theorem stating the equality of sumProductsSidesProjections for two polygons -/
theorem sum_products_sides_projections_equality (P Q : ConvexPolygon) :
  sumProductsSidesProjections P Q = sumProductsSidesProjections Q P :=
by
  sorry

#check sum_products_sides_projections_equality

end NUMINAMATH_CALUDE_sum_products_sides_projections_equality_l108_10832


namespace NUMINAMATH_CALUDE_calculus_class_mean_l108_10827

/-- Calculates the class mean given the number of students and average scores for three groups -/
def class_mean (total_students : ℕ) (group1_students : ℕ) (group1_avg : ℚ) 
               (group2_students : ℕ) (group2_avg : ℚ)
               (group3_students : ℕ) (group3_avg : ℚ) : ℚ :=
  (group1_students * group1_avg + group2_students * group2_avg + group3_students * group3_avg) / total_students

theorem calculus_class_mean :
  let total_students : ℕ := 60
  let group1_students : ℕ := 40
  let group1_avg : ℚ := 68 / 100
  let group2_students : ℕ := 15
  let group2_avg : ℚ := 74 / 100
  let group3_students : ℕ := 5
  let group3_avg : ℚ := 88 / 100
  class_mean total_students group1_students group1_avg group2_students group2_avg group3_students group3_avg = 4270 / 60 :=
by sorry

end NUMINAMATH_CALUDE_calculus_class_mean_l108_10827


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l108_10885

theorem triangle_perimeter_bound : 
  ∀ (s : ℝ), s > 0 → 7 + s > 19 → 19 + s > 7 → s + 7 + 19 < 53 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l108_10885


namespace NUMINAMATH_CALUDE_sin_690_degrees_l108_10829

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l108_10829


namespace NUMINAMATH_CALUDE_kerosene_cost_is_44_cents_l108_10817

/-- The cost of a liter of kerosene in cents -/
def kerosene_cost_cents (rice_cost_dollars : ℚ) : ℚ :=
  let egg_dozen_cost := rice_cost_dollars
  let egg_cost := egg_dozen_cost / 12
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost_dollars := 2 * half_liter_kerosene_cost
  100 * liter_kerosene_cost_dollars

/-- Theorem stating that the cost of a liter of kerosene is 44 cents -/
theorem kerosene_cost_is_44_cents : 
  kerosene_cost_cents (33/100) = 44 := by
sorry

#eval kerosene_cost_cents (33/100)

end NUMINAMATH_CALUDE_kerosene_cost_is_44_cents_l108_10817


namespace NUMINAMATH_CALUDE_franks_initial_money_l108_10803

/-- Frank's lamp purchase problem -/
theorem franks_initial_money (cheapest_lamp : ℕ) (expensive_multiplier : ℕ) (remaining_money : ℕ) : 
  cheapest_lamp = 20 →
  expensive_multiplier = 3 →
  remaining_money = 30 →
  cheapest_lamp * expensive_multiplier + remaining_money = 90 := by
  sorry

end NUMINAMATH_CALUDE_franks_initial_money_l108_10803


namespace NUMINAMATH_CALUDE_mandy_pieces_l108_10857

def chocolate_distribution (total : Nat) (n : Nat) : Nat :=
  if n = 0 then
    total
  else
    chocolate_distribution (total / 2) (n - 1)

theorem mandy_pieces : chocolate_distribution 60 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mandy_pieces_l108_10857


namespace NUMINAMATH_CALUDE_one_nonneg_solution_iff_l108_10822

/-- The quadratic equation with parameter a -/
def quadratic (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2 * (a + 1) * x + 2 * (a + 1)

/-- The condition for having exactly one non-negative solution -/
def has_one_nonneg_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic a x = 0

/-- The theorem stating the condition on parameter a -/
theorem one_nonneg_solution_iff (a : ℝ) :
  has_one_nonneg_solution a ↔ (-1 ≤ a ∧ a ≤ 1) ∨ a = 3 := by sorry

end NUMINAMATH_CALUDE_one_nonneg_solution_iff_l108_10822


namespace NUMINAMATH_CALUDE_trigonometric_problem_l108_10878

theorem trigonometric_problem (α β : Real) 
  (h1 : 3 * Real.sin α - Real.sin β = Real.sqrt 10)
  (h2 : α + β = Real.pi / 2) :
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l108_10878


namespace NUMINAMATH_CALUDE_wrong_number_difference_l108_10864

/-- The number of elements in the set of numbers --/
def n : ℕ := 10

/-- The original average of the numbers --/
def original_average : ℚ := 402/10

/-- The correct average of the numbers --/
def correct_average : ℚ := 403/10

/-- The second wrongly copied number --/
def wrong_second : ℕ := 13

/-- The correct second number --/
def correct_second : ℕ := 31

/-- Theorem stating the difference between the wrongly copied number and the actual number --/
theorem wrong_number_difference (first_wrong : ℚ) (first_actual : ℚ) 
  (h1 : first_wrong > first_actual)
  (h2 : n * original_average = (n - 2) * correct_average + first_wrong + wrong_second)
  (h3 : n * correct_average = (n - 2) * correct_average + first_actual + correct_second) :
  first_wrong - first_actual = 19 := by
  sorry

end NUMINAMATH_CALUDE_wrong_number_difference_l108_10864


namespace NUMINAMATH_CALUDE_intersection_empty_union_real_l108_10809

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 1}

-- Theorem 1
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem 2
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_real_l108_10809


namespace NUMINAMATH_CALUDE_system_solution_l108_10819

theorem system_solution : ∃! (x y : ℚ), 3 * x + 4 * y = 12 ∧ 9 * x - 12 * y = -24 ∧ x = 2/3 ∧ y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l108_10819


namespace NUMINAMATH_CALUDE_polynomial_factor_l108_10891

-- Define the polynomials
def p (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 9
def f (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 3

-- Theorem statement
theorem polynomial_factor (c : ℝ) : 
  (∃ q : ℝ, ∃ r : ℝ → ℝ, ∀ x, p c x = f q x * r x) → c = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_l108_10891


namespace NUMINAMATH_CALUDE_feed_supply_ducks_l108_10892

/-- A batch of feed can supply a certain number of ducks for a given number of days. -/
def FeedSupply (ducks chickens days : ℕ) : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (ducks * x + chickens * y) * days = 210 * y

theorem feed_supply_ducks :
  FeedSupply 10 15 6 →
  FeedSupply 12 6 7 →
  FeedSupply 5 0 21 :=
by sorry

end NUMINAMATH_CALUDE_feed_supply_ducks_l108_10892


namespace NUMINAMATH_CALUDE_alvin_egg_rolls_l108_10899

/-- Given the egg roll consumption of Matthew, Patrick, and Alvin, prove that Alvin ate 4 egg rolls. -/
theorem alvin_egg_rolls (matthew patrick alvin : ℕ) : 
  matthew = 3 * patrick →  -- Matthew eats three times as many egg rolls as Patrick
  patrick = alvin / 2 →    -- Patrick eats half as many egg rolls as Alvin
  matthew = 6 →            -- Matthew ate 6 egg rolls
  alvin = 4 := by           -- Prove that Alvin ate 4 egg rolls
sorry

end NUMINAMATH_CALUDE_alvin_egg_rolls_l108_10899


namespace NUMINAMATH_CALUDE_geometric_series_sum_five_terms_quarter_l108_10882

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_five_terms_quarter :
  geometric_series_sum (1/4) (1/4) 5 = 341/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_five_terms_quarter_l108_10882


namespace NUMINAMATH_CALUDE_chord_relations_l108_10835

theorem chord_relations (d s : ℝ) : 
  0 < s ∧ s < d ∧ d < 2 →  -- Conditions for chords in a unit circle
  (d - s = 1 ∧ d * s = 1 ∧ d^2 - s^2 = Real.sqrt 5) ↔
  (d = (1 + Real.sqrt 5) / 2 ∧ s = (-1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_chord_relations_l108_10835


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l108_10824

/-- The y-intercept of the line 4x + 7y = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → (x = 0 ∧ y = 4) → (0, 4).fst = x ∧ (0, 4).snd = y := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l108_10824


namespace NUMINAMATH_CALUDE_average_age_of_five_students_l108_10861

/-- Given a class of 17 students with an average age of 17 years,
    where 9 students have an average age of 16 years,
    and one student is 75 years old,
    prove that the average age of the remaining 5 students is 14 years. -/
theorem average_age_of_five_students
  (total_students : Nat)
  (total_average : ℝ)
  (nine_students : Nat)
  (nine_average : ℝ)
  (old_student_age : ℝ)
  (h1 : total_students = 17)
  (h2 : total_average = 17)
  (h3 : nine_students = 9)
  (h4 : nine_average = 16)
  (h5 : old_student_age = 75)
  : (total_students * total_average - nine_students * nine_average - old_student_age) / (total_students - nine_students - 1) = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_five_students_l108_10861


namespace NUMINAMATH_CALUDE_bob_cleaning_time_l108_10897

/-- 
Given that Alice takes 32 minutes to clean her room and Bob takes 3/4 of Alice's time,
prove that Bob takes 24 minutes to clean his room.
-/
theorem bob_cleaning_time : 
  let alice_time : ℚ := 32
  let bob_fraction : ℚ := 3/4
  let bob_time : ℚ := alice_time * bob_fraction
  bob_time = 24 := by sorry

end NUMINAMATH_CALUDE_bob_cleaning_time_l108_10897


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l108_10816

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2) :
  (1 + 4 / (a - 1)) / ((a^2 + 6*a + 9) / (a^2 - a)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l108_10816


namespace NUMINAMATH_CALUDE_jackie_daily_distance_l108_10834

/-- Prove that Jackie walks 2 miles per day -/
theorem jackie_daily_distance (jessie_daily : Real) (days : Nat) (extra_distance : Real) :
  jessie_daily = 1.5 →
  days = 6 →
  extra_distance = 3 →
  ∃ (jackie_daily : Real),
    jackie_daily * days = jessie_daily * days + extra_distance ∧
    jackie_daily = 2 := by
  sorry

end NUMINAMATH_CALUDE_jackie_daily_distance_l108_10834


namespace NUMINAMATH_CALUDE_dad_took_90_steps_l108_10886

/-- The number of steps Dad takes for every 5 steps Masha takes -/
def dad_steps : ℕ := 3

/-- The number of steps Masha takes for every 5 steps Yasha takes -/
def masha_steps : ℕ := 3

/-- The total number of steps Masha and Yasha took together -/
def total_steps : ℕ := 400

/-- Theorem stating that Dad took 90 steps -/
theorem dad_took_90_steps : 
  ∃ (d m y : ℕ), 
    d * 5 = m * dad_steps ∧ 
    m * 5 = y * masha_steps ∧ 
    m + y = total_steps ∧ 
    d = 90 := by sorry

end NUMINAMATH_CALUDE_dad_took_90_steps_l108_10886


namespace NUMINAMATH_CALUDE_digit_puzzle_solutions_l108_10855

def is_valid_solution (a b : ℕ) : Prop :=
  a ≠ b ∧
  a < 10 ∧ b < 10 ∧
  10 ≤ 10 * b + a ∧ 10 * b + a < 100 ∧
  10 * b + a ≠ a * b ∧
  a ^ b = 10 * b + a

theorem digit_puzzle_solutions :
  {(a, b) : ℕ × ℕ | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} := by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solutions_l108_10855


namespace NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l108_10895

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 3

theorem extreme_value_and_monotonicity :
  (f 1 = -1 ∧ f' 1 = 0) ∧
  (∀ x, x < -1 → f' x > 0) ∧
  (∀ x, x > 1 → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_monotonicity_l108_10895


namespace NUMINAMATH_CALUDE_hyperbola_equation_l108_10804

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  focal_distance : ℝ
  asymptote_slope : ℝ

-- Define the standard equation of the hyperbola
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, (y^2 / (8/5)) - (x^2 / (72/5)) = 1

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) 
  (h_foci : h.focal_distance = 8)
  (h_asymptote : h.asymptote_slope = 1/3) :
  standard_equation h :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l108_10804


namespace NUMINAMATH_CALUDE_discount_calculation_l108_10890

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 10)
  (h2 : discount_percentage = 10) :
  original_price * (1 - discount_percentage / 100) = 9 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l108_10890


namespace NUMINAMATH_CALUDE_no_real_solutions_l108_10862

theorem no_real_solutions :
  ¬ ∃ y : ℝ, (y - 3*y + 7)^2 + 2 = -2 * |y| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l108_10862


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l108_10845

theorem roots_polynomial_sum (a b c : ℝ) (s : ℝ) : 
  (∀ x, x^3 - 9*x^2 + 11*x - 1 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  s = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  s^4 - 18*s^2 - 8*s = -37 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l108_10845


namespace NUMINAMATH_CALUDE_petrol_expense_l108_10858

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1/10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 6100)
  (h6 : savings = 2400)
  : ∃ (petrol total_salary : ℕ),
    (savings_percentage * total_salary = savings) ∧
    (monthly_expenses rent milk groceries education misc petrol + savings = total_salary) ∧
    (petrol = 2000) := by
  sorry

end NUMINAMATH_CALUDE_petrol_expense_l108_10858


namespace NUMINAMATH_CALUDE_min_n_for_S_gt_1020_l108_10865

def S (n : ℕ) : ℕ := 2^(n+1) - 2 - n

theorem min_n_for_S_gt_1020 : ∀ k : ℕ, k ≥ 10 ↔ S k > 1020 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_S_gt_1020_l108_10865


namespace NUMINAMATH_CALUDE_reflection_theorem_l108_10851

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 1)
  let p'' := (-p'.2, -p'.1)
  (p''.1, p''.2 + 1)

def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (7, 4)

theorem reflection_theorem :
  reflect_line (reflect_x C) = (-5, 8) := by sorry

end NUMINAMATH_CALUDE_reflection_theorem_l108_10851


namespace NUMINAMATH_CALUDE_combination_sequence_implies_value_l108_10869

theorem combination_sequence_implies_value (n : ℕ) : 
  (2 * (n.choose 5) = (n.choose 4) + (n.choose 6)) → 
  (n.choose 12) = 91 := by sorry

end NUMINAMATH_CALUDE_combination_sequence_implies_value_l108_10869


namespace NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l108_10848

/-- The sum of interior angles of a convex polygon with n sides, in degrees -/
def sumInteriorAngles (n : ℕ) : ℝ :=
  180 * (n - 2)

/-- Theorem: The sum of interior angles of a convex n-gon is 180 * (n - 2) degrees -/
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumInteriorAngles n = 180 * (n - 2) := by
  sorry

#check sum_interior_angles_convex_polygon

end NUMINAMATH_CALUDE_sum_interior_angles_convex_polygon_l108_10848


namespace NUMINAMATH_CALUDE_ball_cost_price_l108_10812

theorem ball_cost_price (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) 
  (h1 : selling_price = 720)
  (h2 : num_balls_sold = 17)
  (h3 : num_balls_loss = 5) :
  ∃ (cost_price : ℕ), 
    cost_price * num_balls_sold - cost_price * num_balls_loss = selling_price ∧
    cost_price = 60 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_price_l108_10812


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l108_10850

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l108_10850


namespace NUMINAMATH_CALUDE_fraction_inequality_l108_10877

theorem fraction_inequality (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l108_10877


namespace NUMINAMATH_CALUDE_quadratic_root_value_l108_10896

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 20 * x + c = 0 ↔ x = (-20 + Real.sqrt 16) / 8 ∨ x = (-20 - Real.sqrt 16) / 8) 
  → c = 24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l108_10896


namespace NUMINAMATH_CALUDE_f_k_even_iff_l108_10810

/-- The number of valid coloring schemes for n points on a circle with at least one red point in any k consecutive points. -/
def f_k (k n : ℕ) : ℕ := sorry

/-- Theorem stating the necessary and sufficient conditions for f_k(n) to be even. -/
theorem f_k_even_iff (n k : ℕ) (h1 : n > k) (h2 : k ≥ 2) :
  Even (f_k k n) ↔ Even k ∧ (k + 1 ∣ n) := by sorry

end NUMINAMATH_CALUDE_f_k_even_iff_l108_10810


namespace NUMINAMATH_CALUDE_malcom_remaining_cards_l108_10859

-- Define the number of cards Brandon has
def brandon_cards : ℕ := 20

-- Define the number of additional cards Malcom has compared to Brandon
def malcom_extra_cards : ℕ := 8

-- Define Malcom's initial number of cards
def malcom_initial_cards : ℕ := brandon_cards + malcom_extra_cards

-- Define the number of cards Malcom gives away
def malcom_cards_given : ℕ := malcom_initial_cards / 2

-- Theorem to prove
theorem malcom_remaining_cards :
  malcom_initial_cards - malcom_cards_given = 14 := by
  sorry

end NUMINAMATH_CALUDE_malcom_remaining_cards_l108_10859


namespace NUMINAMATH_CALUDE_square_division_theorem_l108_10887

-- Define a rectangle
structure Rectangle where
  width : ℚ
  height : ℚ

-- Define the problem
theorem square_division_theorem :
  ∃ (r1 r2 r3 r4 r5 : Rectangle),
    -- The sum of areas equals 1
    r1.width * r1.height + r2.width * r2.height + r3.width * r3.height + 
    r4.width * r4.height + r5.width * r5.height = 1 ∧
    -- All widths and heights are distinct
    r1.width ≠ r1.height ∧ r1.width ≠ r2.width ∧ r1.width ≠ r2.height ∧
    r1.width ≠ r3.width ∧ r1.width ≠ r3.height ∧ r1.width ≠ r4.width ∧
    r1.width ≠ r4.height ∧ r1.width ≠ r5.width ∧ r1.width ≠ r5.height ∧
    r1.height ≠ r2.width ∧ r1.height ≠ r2.height ∧ r1.height ≠ r3.width ∧
    r1.height ≠ r3.height ∧ r1.height ≠ r4.width ∧ r1.height ≠ r4.height ∧
    r1.height ≠ r5.width ∧ r1.height ≠ r5.height ∧ r2.width ≠ r2.height ∧
    r2.width ≠ r3.width ∧ r2.width ≠ r3.height ∧ r2.width ≠ r4.width ∧
    r2.width ≠ r4.height ∧ r2.width ≠ r5.width ∧ r2.width ≠ r5.height ∧
    r2.height ≠ r3.width ∧ r2.height ≠ r3.height ∧ r2.height ≠ r4.width ∧
    r2.height ≠ r4.height ∧ r2.height ≠ r5.width ∧ r2.height ≠ r5.height ∧
    r3.width ≠ r3.height ∧ r3.width ≠ r4.width ∧ r3.width ≠ r4.height ∧
    r3.width ≠ r5.width ∧ r3.width ≠ r5.height ∧ r3.height ≠ r4.width ∧
    r3.height ≠ r4.height ∧ r3.height ≠ r5.width ∧ r3.height ≠ r5.height ∧
    r4.width ≠ r4.height ∧ r4.width ≠ r5.width ∧ r4.width ≠ r5.height ∧
    r4.height ≠ r5.width ∧ r4.height ≠ r5.height ∧ r5.width ≠ r5.height := by
  sorry


end NUMINAMATH_CALUDE_square_division_theorem_l108_10887


namespace NUMINAMATH_CALUDE_rectangle_area_l108_10831

/-- The area of a rectangle with width 7 meters and length 2 meters longer than the width is 63 square meters. -/
theorem rectangle_area : ℝ → ℝ → ℝ → Prop :=
  fun width length area =>
    width = 7 ∧ length = width + 2 → area = width * length → area = 63

/-- Proof of the theorem -/
lemma rectangle_area_proof : rectangle_area 7 9 63 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l108_10831


namespace NUMINAMATH_CALUDE_cost_price_calculation_l108_10823

def selling_price : ℝ := 270
def profit_percentage : ℝ := 0.20

theorem cost_price_calculation :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + profit_percentage) = selling_price ∧ 
    cost_price = 225 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l108_10823


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l108_10815

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l108_10815


namespace NUMINAMATH_CALUDE_gcd_of_90_and_450_l108_10880

theorem gcd_of_90_and_450 : Nat.gcd 90 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_450_l108_10880


namespace NUMINAMATH_CALUDE_quartic_two_real_roots_l108_10860

theorem quartic_two_real_roots 
  (a b c d e : ℝ) 
  (ha : a ≠ 0) 
  (h_root : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + (e - d) = 0) :
  ∃ x y : ℝ, x ≠ y ∧ a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 ∧ 
                    a * y^4 + b * y^3 + c * y^2 + d * y + e = 0 :=
by sorry

end NUMINAMATH_CALUDE_quartic_two_real_roots_l108_10860
