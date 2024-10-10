import Mathlib

namespace positive_sum_square_inequality_l2835_283547

theorem positive_sum_square_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) :=
by sorry

end positive_sum_square_inequality_l2835_283547


namespace rhombus_circumcircle_radii_l2835_283548

/-- A rhombus with circumcircles of two triangles formed by its sides -/
structure RhombusWithCircumcircles where
  /-- Side length of the rhombus -/
  side_length : ℝ
  /-- Distance between centers of circumcircles -/
  center_distance : ℝ
  /-- Radius of the circumcircle of triangle ABC -/
  radius_ABC : ℝ
  /-- Radius of the circumcircle of triangle BCD -/
  radius_BCD : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length
  /-- The center distance is positive -/
  center_distance_pos : 0 < center_distance

/-- Theorem about the radii of circumcircles in a specific rhombus configuration -/
theorem rhombus_circumcircle_radii
  (r : RhombusWithCircumcircles)
  (h1 : r.side_length = 6)
  (h2 : r.center_distance = 8) :
  r.radius_ABC = 3 * Real.sqrt 10 ∧ r.radius_BCD = 3 * Real.sqrt 10 := by
  sorry

#check rhombus_circumcircle_radii

end rhombus_circumcircle_radii_l2835_283548


namespace maximize_product_l2835_283564

theorem maximize_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 60) :
  x^3 * y^2 * z^4 ≤ 20^3 * (40/3)^2 * (80/3)^4 ∧
  (x^3 * y^2 * z^4 = 20^3 * (40/3)^2 * (80/3)^4 ↔ x = 20 ∧ y = 40/3 ∧ z = 80/3) :=
by sorry

end maximize_product_l2835_283564


namespace expression_evaluation_l2835_283500

theorem expression_evaluation (a b : ℝ) (ha : a = 2023) (hb : b = 2020) : 
  ((3 / (a - b) + 3 * a / (a^3 - b^3) * (a^2 + a*b + b^2) / (a + b)) * 
   (2*a + b) / (a^2 + 2*a*b + b^2)) * 3 / (a + b) = 3 := by
sorry

end expression_evaluation_l2835_283500


namespace geometric_sum_pebbles_l2835_283540

theorem geometric_sum_pebbles (a : ℕ) (r : ℕ) (n : ℕ) (h1 : a = 1) (h2 : r = 2) (h3 : n = 10) :
  a * (r^n - 1) / (r - 1) = 1023 := by
  sorry

end geometric_sum_pebbles_l2835_283540


namespace cubic_roots_of_unity_l2835_283513

theorem cubic_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
sorry

end cubic_roots_of_unity_l2835_283513


namespace tanya_score_percentage_l2835_283561

/-- Tanya's score on the math quiz -/
def score : ℚ := 20 / 25

/-- The percentage equivalent of Tanya's score -/
def percentage : ℚ := 80 / 100

theorem tanya_score_percentage : score = percentage := by sorry

end tanya_score_percentage_l2835_283561


namespace polynomial_value_for_quadratic_roots_l2835_283502

theorem polynomial_value_for_quadratic_roots : 
  ∀ x : ℝ, x^2 - 4*x + 1 = 0 → 
  (x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3) ∨
  (x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3) := by
  sorry

end polynomial_value_for_quadratic_roots_l2835_283502


namespace complement_intersection_theorem_l2835_283535

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 5} := by
  sorry

end complement_intersection_theorem_l2835_283535


namespace first_quadrant_half_angle_l2835_283514

theorem first_quadrant_half_angle (α : Real) : 0 < α ∧ α < π / 2 → 0 < α / 2 ∧ α / 2 < π / 4 := by
  sorry

end first_quadrant_half_angle_l2835_283514


namespace val_initial_nickels_l2835_283527

/-- The number of nickels Val initially had -/
def initial_nickels : ℕ := 20

/-- The value of a nickel in cents -/
def nickel_value : ℚ := 5/100

/-- The value of a dime in cents -/
def dime_value : ℚ := 10/100

/-- The total value in dollars after tripling the number of nickels -/
def total_value : ℚ := 9

theorem val_initial_nickels :
  ∀ n : ℕ,
  (n : ℚ) * nickel_value * 3 + (3 * n : ℚ) * dime_value = total_value →
  n = initial_nickels :=
by
  sorry

end val_initial_nickels_l2835_283527


namespace tangent_to_ln_curve_l2835_283566

theorem tangent_to_ln_curve (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ a * x = Real.log x ∧ (∀ y : ℝ, y > 0 → a * y ≥ Real.log y)) → 
  a = 1 / Real.exp 1 :=
sorry

end tangent_to_ln_curve_l2835_283566


namespace power_mod_eleven_l2835_283567

theorem power_mod_eleven : (5 : ℤ) ^ 1233 % 11 = 4 := by
  sorry

end power_mod_eleven_l2835_283567


namespace coefficient_of_x_cubed_l2835_283572

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expr := 2*(x^2 - 2*x^3 + x) + 4*(x + 3*x^3 - 2*x^2 + 2*x^5 + x^3) - 6*(2 + x - 5*x^3 - x^2)
  ∃ (a b c d e : ℝ), expr = a*x^5 + b*x^4 + 42*x^3 + c*x^2 + d*x + e :=
by sorry

end coefficient_of_x_cubed_l2835_283572


namespace jessica_purchases_total_cost_l2835_283520

theorem jessica_purchases_total_cost :
  let cat_toy_cost : ℚ := 10.22
  let cage_cost : ℚ := 11.73
  let total_cost : ℚ := cat_toy_cost + cage_cost
  total_cost = 21.95 := by sorry

end jessica_purchases_total_cost_l2835_283520


namespace additional_friends_average_weight_l2835_283581

theorem additional_friends_average_weight
  (initial_count : ℕ)
  (additional_count : ℕ)
  (average_increase : ℝ)
  (final_average : ℝ)
  (h1 : initial_count = 50)
  (h2 : additional_count = 40)
  (h3 : average_increase = 12)
  (h4 : final_average = 46) :
  let total_count := initial_count + additional_count
  let initial_average := final_average - average_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  let additional_average := additional_total_weight / additional_count
  additional_average = 61 := by
sorry

end additional_friends_average_weight_l2835_283581


namespace set_operations_and_intersection_l2835_283531

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem statement
theorem set_operations_and_intersection (a : ℝ) :
  (A ∪ B = {x | 1 ≤ x ∧ x < 10}) ∧
  ((Aᶜ) ∩ B = {x | 7 ≤ x ∧ x < 10}) ∧
  ((A ∩ C a).Nonempty → a > 1) := by
  sorry

end set_operations_and_intersection_l2835_283531


namespace technician_round_trip_completion_l2835_283585

theorem technician_round_trip_completion (D : ℝ) (h : D > 0) : 
  let total_distance : ℝ := 2 * D
  let completed_distance : ℝ := D + 0.3 * D
  (completed_distance / total_distance) * 100 = 65 := by sorry

end technician_round_trip_completion_l2835_283585


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2835_283599

-- 1. 78 × 4 + 488
theorem problem_1 : 78 * 4 + 488 = 800 := by sorry

-- 2. 350 × (12 + 342 ÷ 9)
theorem problem_2 : 350 * (12 + 342 / 9) = 17500 := by sorry

-- 3. (3600 - 18 × 200) ÷ 253
theorem problem_3 : (3600 - 18 * 200) / 253 = 0 := by sorry

-- 4. 1903 - 475 × 4
theorem problem_4 : 1903 - 475 * 4 = 3 := by sorry

-- 5. 480 ÷ (125 - 117)
theorem problem_5 : 480 / (125 - 117) = 60 := by sorry

-- 6. (243 - 162) ÷ 27 × 380
theorem problem_6 : (243 - 162) / 27 * 380 = 1140 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l2835_283599


namespace hindi_books_count_l2835_283546

def number_of_arrangements (n m : ℕ) : ℕ := Nat.choose (n + 1) m

theorem hindi_books_count : ∃ h : ℕ, 
  number_of_arrangements 22 h = 1771 ∧ h = 3 :=
by sorry

end hindi_books_count_l2835_283546


namespace scientific_notation_657000_l2835_283587

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_657000 :
  toScientificNotation 657000 = ScientificNotation.mk 6.57 5 (by norm_num) :=
sorry

end scientific_notation_657000_l2835_283587


namespace intersection_in_second_quadrant_l2835_283589

/-- The slope of line l₁ -/
def m₁ : ℚ := 2/3

/-- The y-intercept of line l₁ in terms of a -/
def b₁ (a : ℚ) : ℚ := (1 - a) / 3

/-- The slope of line l₂ -/
def m₂ : ℚ := -1/2

/-- The y-intercept of line l₂ in terms of a -/
def b₂ (a : ℚ) : ℚ := a

/-- The x-coordinate of the intersection point of l₁ and l₂ -/
def x_intersect (a : ℚ) : ℚ := (b₂ a - b₁ a) / (m₁ - m₂)

/-- The y-coordinate of the intersection point of l₁ and l₂ -/
def y_intersect (a : ℚ) : ℚ := m₁ * x_intersect a + b₁ a

/-- The theorem stating the condition for the intersection point to be in the second quadrant -/
theorem intersection_in_second_quadrant (a : ℚ) :
  (x_intersect a > 0 ∧ y_intersect a > 0) ↔ a > 1/4 := by sorry

end intersection_in_second_quadrant_l2835_283589


namespace weaving_increase_l2835_283503

/-- Represents the daily increase in weaving -/
def daily_increase : ℚ := 16 / 29

/-- Represents the number of days in a month -/
def days_in_month : ℕ := 30

/-- Represents the amount woven on the first day -/
def first_day_weaving : ℚ := 10

/-- Represents the total amount woven in a month -/
def total_weaving : ℚ := 6

theorem weaving_increase :
  first_day_weaving + (days_in_month - 1) * daily_increase / 2 * days_in_month = total_weaving :=
sorry

end weaving_increase_l2835_283503


namespace acute_iff_three_equal_projections_l2835_283526

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A direction in a 2D plane, represented by an angle from the positive x-axis --/
def Direction := ℝ

/-- The length of the projection of a triangle onto a given direction --/
def projectionLength (t : Triangle) (d : Direction) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled --/
def isAcute (t : Triangle) : Prop := sorry

/-- Theorem stating that a triangle is acute if and only if it has three equal projections in distinct directions --/
theorem acute_iff_three_equal_projections (t : Triangle) :
  isAcute t ↔
  ∃ (d₁ d₂ d₃ : Direction),
    d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃ ∧
    ∃ (l : ℝ),
      projectionLength t d₁ = l ∧
      projectionLength t d₂ = l ∧
      projectionLength t d₃ = l :=
sorry

end acute_iff_three_equal_projections_l2835_283526


namespace G_function_iff_strictly_increasing_l2835_283551

open Real

/-- Definition of a "G" function on an open interval (a, b) --/
def is_G_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, a < x₁ ∧ x₁ < b ∧ a < x₂ ∧ x₂ < b ∧ x₁ ≠ x₂ →
    x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing on an open interval (a, b) --/
def strictly_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

/-- Theorem: A function is a "G" function on (a, b) if and only if it is strictly increasing on (a, b) --/
theorem G_function_iff_strictly_increasing (f : ℝ → ℝ) (a b : ℝ) :
  is_G_function f a b ↔ strictly_increasing_on f a b :=
sorry

end G_function_iff_strictly_increasing_l2835_283551


namespace geometric_sequence_common_ratio_l2835_283510

/-- A geometric sequence with first term 1 and sum of first 3 terms equal to 3/4 has common ratio -1/2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
    a 1 = 1 →                     -- first term is 1
    a 1 + a 2 + a 3 = 3/4 →       -- sum of first 3 terms is 3/4
    q = -1/2 := by
  sorry

end geometric_sequence_common_ratio_l2835_283510


namespace smallest_angle_in_special_trapezoid_l2835_283509

/-- Represents the angles of a trapezoid --/
structure TrapezoidAngles where
  a : ℝ  -- First angle
  b : ℝ  -- Second angle
  c : ℝ  -- Third angle
  d : ℝ  -- Fourth angle

/-- Checks if the angles form a valid trapezoid configuration --/
def is_valid_trapezoid (angles : TrapezoidAngles) : Prop :=
  -- Sum of angles in a quadrilateral is 360°
  angles.a + angles.b + angles.c + angles.d = 360 ∧
  -- One pair of opposite angles are supplementary
  (angles.a + angles.c = 180 ∨ angles.b + angles.d = 180) ∧
  -- Consecutive angles on each side form arithmetic sequences
  (∃ x y : ℝ, (angles.a = x ∧ angles.b = x + y) ∨ (angles.c = x ∧ angles.d = x + y)) ∧
  (∃ p q : ℝ, (angles.b = p ∧ angles.c = p + q) ∨ (angles.d = p ∧ angles.a = p + q))

/-- The main theorem --/
theorem smallest_angle_in_special_trapezoid :
  ∀ angles : TrapezoidAngles,
  is_valid_trapezoid angles →
  (angles.a = 140 ∨ angles.b = 140 ∨ angles.c = 140 ∨ angles.d = 140) →
  (angles.a = 20 ∨ angles.b = 20 ∨ angles.c = 20 ∨ angles.d = 20) :=
by sorry

end smallest_angle_in_special_trapezoid_l2835_283509


namespace greatest_q_minus_r_l2835_283570

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  947 = 23 * q + r ∧ 
  q > 0 ∧ 
  r > 0 ∧ 
  ∀ (q' r' : ℕ), (947 = 23 * q' + r' ∧ q' > 0 ∧ r' > 0) → q' - r' ≤ q - r ∧
  q - r = 37 := by
sorry

end greatest_q_minus_r_l2835_283570


namespace cone_surface_area_special_case_l2835_283550

/-- Represents a cone with given slant height and lateral surface property -/
structure Cone where
  slant_height : ℝ
  lateral_surface_semicircle : Prop

/-- Calculates the surface area of a cone -/
def surface_area (c : Cone) : ℝ :=
  sorry -- Definition to be implemented

/-- Theorem: The surface area of a cone with slant height 2 and lateral surface
    that unfolds into a semicircle is 3π -/
theorem cone_surface_area_special_case :
  ∀ (c : Cone), c.slant_height = 2 → c.lateral_surface_semicircle →
  surface_area c = 3 * Real.pi :=
sorry

end cone_surface_area_special_case_l2835_283550


namespace contrapositive_falsehood_l2835_283545

theorem contrapositive_falsehood (p q : Prop) :
  (¬(p → q)) → (¬(¬q → ¬p)) := by
  sorry

end contrapositive_falsehood_l2835_283545


namespace integer_equation_solution_l2835_283559

theorem integer_equation_solution (x y : ℤ) : 
  12 * x^2 + 6 * x * y + 3 * y^2 = 28 * (x + y) ↔ 
  ∃ m n : ℤ, y = 4 * n ∧ x = 3 * m - 4 * n := by
sorry

end integer_equation_solution_l2835_283559


namespace percent_of_x_is_z_l2835_283577

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 0.96 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.6 * x := by
sorry

end percent_of_x_is_z_l2835_283577


namespace square_sum_of_complex_square_l2835_283521

theorem square_sum_of_complex_square (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑a + ↑b * Complex.I)^2 = (3 : ℂ) + 4 * Complex.I →
  a^2 + b^2 = 25 := by sorry

end square_sum_of_complex_square_l2835_283521


namespace nell_cards_remaining_l2835_283525

/-- Given that Nell had 242 cards initially and gave away 136 cards, prove that she has 106 cards left. -/
theorem nell_cards_remaining (initial_cards : ℕ) (cards_given_away : ℕ) (h1 : initial_cards = 242) (h2 : cards_given_away = 136) :
  initial_cards - cards_given_away = 106 := by
  sorry

end nell_cards_remaining_l2835_283525


namespace well_depth_rope_length_l2835_283528

/-- 
Given a well of unknown depth and a rope of unknown length, prove that if:
1) Folding the rope three times results in a length equal to the well's depth plus 4 feet
2) Folding the rope four times results in a length equal to the well's depth plus 1 foot
Then the relationship between the rope length (h) and well depth (x) is:
{ h/3 = x + 4
{ h/4 = x + 1
-/
theorem well_depth_rope_length (h x : ℝ) : 
  (h / 3 = x + 4 ∧ h / 4 = x + 1) ↔ 
  (∃ (depth : ℝ) (rope_length : ℝ), 
    depth = x ∧ 
    rope_length = h ∧
    rope_length / 3 = depth + 4 ∧ 
    rope_length / 4 = depth + 1) := by
sorry

end well_depth_rope_length_l2835_283528


namespace distance_between_complex_points_l2835_283552

theorem distance_between_complex_points :
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -2 + 3*I
  Complex.abs (z₁ - z₂) = Real.sqrt 26 := by sorry

end distance_between_complex_points_l2835_283552


namespace angle_difference_range_l2835_283582

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π) :
  ∃ (x : Real), -3*π/2 < x ∧ x < 0 ∧ ∀ (y : Real), (-3*π/2 < y ∧ y < 0) → ∃ (α' β' : Real),
    -π/2 < α' ∧ α' < β' ∧ β' < π ∧ y = α' - β' :=
by sorry

end angle_difference_range_l2835_283582


namespace notebook_cost_l2835_283534

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buyers notebooks_per_student cost : Nat),
  -- Total number of students
  total_students = 36 ∧
  -- Majority of students bought notebooks
  buyers > total_students / 2 ∧
  -- Each student bought more than one notebook
  notebooks_per_student > 1 ∧
  -- Cost in cents is higher than the number of notebooks bought
  cost > notebooks_per_student ∧
  -- Total cost equation
  buyers * notebooks_per_student * cost = total_cost ∧
  -- Total cost given
  total_cost = 2079 ∧
  -- The cost of each notebook is 11 cents
  cost = 11 := by
sorry

end notebook_cost_l2835_283534


namespace quadruplet_solution_l2835_283591

theorem quadruplet_solution (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_product : a * b * c * d = 1)
  (h_eq1 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h_eq2 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end quadruplet_solution_l2835_283591


namespace cylinder_volume_increase_l2835_283544

theorem cylinder_volume_increase : ∀ (r h : ℝ),
  r > 0 → h > 0 →
  let new_r := r * 2.5
  let new_h := h * 3
  (π * new_r^2 * new_h) / (π * r^2 * h) = 18.75 := by
sorry

end cylinder_volume_increase_l2835_283544


namespace total_volume_of_cubes_l2835_283593

-- Define the number and side length of Carl's cubes
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3

-- Define the number and side length of Kate's cubes
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Function to calculate the volume of a cube
def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

-- Theorem statement
theorem total_volume_of_cubes :
  carl_cubes * cube_volume carl_side_length + kate_cubes * cube_volume kate_side_length = 114 := by
  sorry


end total_volume_of_cubes_l2835_283593


namespace sum_of_cubes_l2835_283556

theorem sum_of_cubes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 2 / 3) (h4 : b - a = 3) :
  a^3 + b^3 = 945 := by
  sorry

end sum_of_cubes_l2835_283556


namespace sum_of_repeating_decimals_l2835_283557

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_04 : ℚ := 4/99
def repeating_decimal_005 : ℚ := 5/999

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_04 + repeating_decimal_005 = 1134/2997 := by sorry

end sum_of_repeating_decimals_l2835_283557


namespace alex_jane_pen_difference_l2835_283578

def pens_after_n_weeks (initial_pens : ℕ) (n : ℕ) : ℕ :=
  initial_pens * (2 ^ n)

theorem alex_jane_pen_difference :
  let alex_initial_pens : ℕ := 4
  let weeks_in_month : ℕ := 4
  let jane_pens : ℕ := 16
  pens_after_n_weeks alex_initial_pens weeks_in_month - jane_pens = 16 :=
by
  sorry

#check alex_jane_pen_difference

end alex_jane_pen_difference_l2835_283578


namespace intersection_A_complement_B_l2835_283590

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1, 2} := by
  sorry

end intersection_A_complement_B_l2835_283590


namespace unique_point_not_on_parabola_l2835_283515

/-- A parabola passing through points (-1,0) and (2,0) -/
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the parabola passes through (-1,0) and (2,0) -/
def parabola_condition (a b c : ℝ) : Prop :=
  parabola a b c (-1) = 0 ∧ parabola a b c 2 = 0

/-- The point P(x_0 + 1, 2x_0^2 - 2) -/
def point_P (x_0 : ℝ) : ℝ × ℝ := (x_0 + 1, 2 * x_0^2 - 2)

/-- The theorem stating that P(-1, 6) is the only point satisfying the conditions -/
theorem unique_point_not_on_parabola :
  ∀ (a b c : ℝ), a ≠ 0 → parabola_condition a b c →
  (∀ (x_0 : ℝ), point_P x_0 ≠ (x_0 + 1, parabola a b c (x_0 + 1))) →
  point_P (-3) = (-1, 6) := by sorry

end unique_point_not_on_parabola_l2835_283515


namespace hyperbola_eccentricity_l2835_283594

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola a b (F₂.1) (F₂.2) ∧ 
  F₁.1 < F₂.1 ∧
  F₁.2 = F₂.2

-- Define the circle with diameter F₁F₂
def circle_diameter (F₁ F₂ P : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (F₂.1 - F₁.1)^2 / 4 + (F₂.2 - F₁.2)^2 / 4

-- Define the intersection of PF₁ and the hyperbola
def intersection (P Q F₁ : ℝ × ℝ) (a b : ℝ) : Prop :=
  ∃ t : ℝ, 
    Q.1 = F₁.1 + t * (P.1 - F₁.1) ∧
    Q.2 = F₁.2 + t * (P.2 - F₁.2) ∧
    hyperbola a b Q.1 Q.2

-- Define the distance condition
def distance_condition (P Q F₁ : ℝ × ℝ) : Prop :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4 * ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2)

-- Theorem statement
theorem hyperbola_eccentricity 
  (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) :
  hyperbola a b P.1 P.2 →
  foci F₁ F₂ a b →
  circle_diameter F₁ F₂ P →
  intersection P Q F₁ a b →
  distance_condition P Q F₁ →
  ∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5 :=
sorry

end hyperbola_eccentricity_l2835_283594


namespace gcd_of_168_and_294_l2835_283537

theorem gcd_of_168_and_294 : Nat.gcd 168 294 = 42 := by
  sorry

end gcd_of_168_and_294_l2835_283537


namespace g_range_l2835_283511

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 4)

theorem g_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -5 ∧ g x = y} = {y : ℝ | y ≠ -27} := by
  sorry

end g_range_l2835_283511


namespace expression_value_l2835_283518

theorem expression_value (x : ℝ) (h : x^2 - 2*x = 3) : 2*x^2 - 4*x + 3 = 9 := by
  sorry

end expression_value_l2835_283518


namespace complex_roots_quadratic_l2835_283586

theorem complex_roots_quadratic (a b : ℝ) : 
  (Complex.mk a 3) ^ 2 - (Complex.mk 12 9) * (Complex.mk a 3) + (Complex.mk 15 65) = 0 ∧
  (Complex.mk b 6) ^ 2 - (Complex.mk 12 9) * (Complex.mk b 6) + (Complex.mk 15 65) = 0 →
  a = 7 / 3 ∧ b = 29 / 3 := by
sorry

end complex_roots_quadratic_l2835_283586


namespace rope_cutting_problem_l2835_283595

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 63) (hb : b = 84) (hc : c = 105) :
  Nat.gcd a (Nat.gcd b c) = 21 := by
  sorry

end rope_cutting_problem_l2835_283595


namespace coin_draw_probability_l2835_283529

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 8
def total_coins : ℕ := pennies + nickels + dimes
def drawn_coins : ℕ := 8
def min_value : ℕ := 75

def successful_outcomes : ℕ := 321
def total_outcomes : ℕ := 12870

theorem coin_draw_probability :
  let prob := successful_outcomes / total_outcomes
  (∀ (outcome : Fin total_outcomes), 
    (outcome.val < successful_outcomes → 
      (∃ (p n d : ℕ), p + n + d = drawn_coins ∧ 
        p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧
        p * 1 + n * 5 + d * 10 ≥ min_value))) ∧
  (∀ (p n d : ℕ), p + n + d = drawn_coins → 
    p ≤ pennies ∧ n ≤ nickels ∧ d ≤ dimes ∧
    p * 1 + n * 5 + d * 10 ≥ min_value →
    (∃ (outcome : Fin total_outcomes), outcome.val < successful_outcomes)) →
  prob = successful_outcomes / total_outcomes :=
by sorry

end coin_draw_probability_l2835_283529


namespace not_adjacent_2010_2011_l2835_283505

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sorted_by_digit_sum_then_value (a b : ℕ) : Prop :=
  (sum_of_digits a < sum_of_digits b) ∨
  (sum_of_digits a = sum_of_digits b ∧ a < b)

theorem not_adjacent_2010_2011 (s : List ℕ) :
  s.length = 100 →
  (∃ k : ℕ, ∀ n ∈ s, k ≤ n ∧ n < k + 100) →
  s.Sorted sorted_by_digit_sum_then_value →
  ¬∃ i : ℕ, i < s.length - 1 ∧
    ((s.get ⟨i, by sorry⟩ = 2010 ∧ s.get ⟨i+1, by sorry⟩ = 2011) ∨
     (s.get ⟨i, by sorry⟩ = 2011 ∧ s.get ⟨i+1, by sorry⟩ = 2010)) :=
by sorry

end not_adjacent_2010_2011_l2835_283505


namespace least_n_satisfying_inequality_l2835_283504

theorem least_n_satisfying_inequality : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(1 / m - 1 / (m + 1) < 1 / 10)) ∧ 
  (1 / n - 1 / (n + 1) < 1 / 10) ∧ n = 3 :=
by sorry

end least_n_satisfying_inequality_l2835_283504


namespace intersection_of_A_and_B_l2835_283592

def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 = 7}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end intersection_of_A_and_B_l2835_283592


namespace polynomial_identity_l2835_283519

theorem polynomial_identity (a b c : ℤ) (h_c_odd : Odd c) :
  let P : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c
  let p : ℕ → ℝ := fun i ↦ P i
  (p 1)^3 + (p 2)^3 + (p 3)^3 = 3*(p 1)*(p 2)*(p 3) →
  (p 2) + 2*(p 1) - 3*(p 0) = 18 := by
sorry

end polynomial_identity_l2835_283519


namespace least_positive_angle_solution_l2835_283549

-- Define the problem
theorem least_positive_angle_solution (θ : Real) : 
  (θ > 0 ∧ 
   Real.cos (10 * π / 180) = Real.sin (35 * π / 180) + Real.sin θ ∧ 
   ∀ φ, φ > 0 ∧ Real.cos (10 * π / 180) = Real.sin (35 * π / 180) + Real.sin φ → θ ≤ φ) → 
  θ = 32.5 * π / 180 := by
sorry

end least_positive_angle_solution_l2835_283549


namespace abs_equation_solution_l2835_283571

theorem abs_equation_solution : 
  {x : ℝ | |2005 * x - 2005| = 2005} = {0, 2} := by
sorry

end abs_equation_solution_l2835_283571


namespace farmer_apples_l2835_283522

/-- The number of apples given away by the farmer -/
def apples_given : ℕ := 88

/-- The number of apples left after giving some away -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given + apples_left

theorem farmer_apples : initial_apples = 127 := by
  sorry

end farmer_apples_l2835_283522


namespace five_toppings_from_eight_equals_fiftysix_l2835_283543

/-- The number of ways to choose 5 items from a set of 8 items -/
def choose_five_from_eight : ℕ := Nat.choose 8 5

/-- The theorem stating that choosing 5 items from 8 results in 56 combinations -/
theorem five_toppings_from_eight_equals_fiftysix : 
  choose_five_from_eight = 56 := by
  sorry

end five_toppings_from_eight_equals_fiftysix_l2835_283543


namespace perpendicular_lines_and_max_product_l2835_283555

-- Define the lines and points
def l1 (m : ℝ) : ℝ × ℝ → Prop := λ p => p.1 + m * p.2 = 0
def l2 (m : ℝ) : ℝ × ℝ → Prop := λ p => m * p.1 - p.2 - m + 3 = 0
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 3)

-- Define the intersection point P
def P (m : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem perpendicular_lines_and_max_product (m : ℝ) :
  (∀ p q : ℝ × ℝ, l1 m p → l2 m q → (p.1 - q.1) * (p.2 - q.2) = 0) ∧ 
  (∀ m : ℝ, ‖P m - A‖ * ‖P m - B‖ ≤ 5) ∧
  (∃ m : ℝ, ‖P m - A‖ * ‖P m - B‖ = 5) :=
sorry

end perpendicular_lines_and_max_product_l2835_283555


namespace sum_of_three_integers_with_product_125_l2835_283565

theorem sum_of_three_integers_with_product_125 :
  ∃ (a b c : ℕ+), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 125 ∧
    (a + b + c : ℕ) = 31 := by
  sorry

end sum_of_three_integers_with_product_125_l2835_283565


namespace apples_given_to_teachers_l2835_283553

theorem apples_given_to_teachers 
  (total_apples : ℕ) 
  (apples_to_friends : ℕ) 
  (apples_eaten : ℕ) 
  (apples_left : ℕ) 
  (h1 : total_apples = 25)
  (h2 : apples_to_friends = 5)
  (h3 : apples_eaten = 1)
  (h4 : apples_left = 3) :
  total_apples - apples_to_friends - apples_eaten - apples_left = 16 := by
sorry

end apples_given_to_teachers_l2835_283553


namespace james_and_david_probability_l2835_283563

def total_workers : ℕ := 14
def workers_to_choose : ℕ := 2

theorem james_and_david_probability :
  (1 : ℚ) / (Nat.choose total_workers workers_to_choose) = 1 / 91 := by
  sorry

end james_and_david_probability_l2835_283563


namespace for_loop_properties_l2835_283573

/-- Represents a for loop in a programming language --/
structure ForLoop where
  init : ℕ
  final : ℕ
  step : ℤ
  body : List (Unit → Unit)

/-- The loop expression of a for loop --/
def loopExpression (loop : ForLoop) : List (Unit → Unit) := loop.body

/-- The loop body of a for loop --/
def loopBody (loop : ForLoop) : List (Unit → Unit) := loop.body

/-- Function to check if step can be omitted --/
def canOmitStep (loop : ForLoop) : Bool := loop.step = 1

/-- Function to check if loop can proceed --/
def canProceed (loop : ForLoop) : Bool := loop.final ≠ 0

/-- Function representing the control of loop termination and new loop start --/
def loopControl (loop : ForLoop) : Unit → Unit :=
  fun _ => ()  -- Placeholder function

theorem for_loop_properties (loop : ForLoop) : 
  (loopExpression loop = loopBody loop) ∧ 
  (canOmitStep loop ↔ loop.step = 1) ∧
  (canProceed loop ↔ loop.final ≠ 0) ∧
  (loopControl loop ≠ fun _ => ()) := by sorry

end for_loop_properties_l2835_283573


namespace overtime_rate_multiple_l2835_283532

/-- Calculates the multiple of the regular rate for excess hours worked --/
theorem overtime_rate_multiple
  (regular_hours : ℝ)
  (regular_rate : ℝ)
  (total_hours : ℝ)
  (total_earnings : ℝ)
  (h1 : regular_hours = 7.5)
  (h2 : regular_rate = 4.5)
  (h3 : total_hours = 10.5)
  (h4 : total_earnings = 60.75)
  : (total_earnings - regular_hours * regular_rate) / ((total_hours - regular_hours) * regular_rate) = 2 := by
  sorry


end overtime_rate_multiple_l2835_283532


namespace distance_between_points_l2835_283541

/-- Prove the distance between two points given rowing speed, stream speed, and round trip time -/
theorem distance_between_points (v : ℝ) (s : ℝ) (t : ℝ) (h1 : v > s) (h2 : v > 0) (h3 : s > 0) (h4 : t > 0) :
  let d := (v * t * (v - s) * (v + s)) / (2 * v)
  d = 24 ∧ (d / (v - s) + d / (v + s) = t) :=
by sorry

end distance_between_points_l2835_283541


namespace puppies_brought_in_solution_l2835_283539

/-- The number of puppies brought to a pet shelter -/
def puppies_brought_in (initial_puppies : ℕ) (adoption_rate : ℕ) (days_to_adopt : ℕ) : ℕ :=
  adoption_rate * days_to_adopt - initial_puppies

/-- Theorem stating that 12 puppies were brought in given the problem conditions -/
theorem puppies_brought_in_solution :
  puppies_brought_in 9 3 7 = 12 := by
  sorry

end puppies_brought_in_solution_l2835_283539


namespace geometric_sequence_minimum_l2835_283568

/-- A positive geometric sequence with common ratio 2 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 2 * a n ∧ a n > 0

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  GeometricSequence a →
  (∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1) →
  (∀ k l : ℕ, Real.sqrt (a k * a l) = 4 * a 1 → 1 / k + 4 / l ≥ 3 / 2) :=
by sorry

end geometric_sequence_minimum_l2835_283568


namespace hyperbola_focal_length_l2835_283507

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 = 1 ∧ m > 0

-- Define the asymptote of C
def asymptote (m : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem hyperbola_focal_length (m : ℝ) :
  (∀ x y, hyperbola m x y → asymptote m x y) →
  (∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4) :=
by sorry

end hyperbola_focal_length_l2835_283507


namespace point_on_curve_iff_function_zero_l2835_283588

variable (f : ℝ × ℝ → ℝ)
variable (x₀ y₀ : ℝ)

theorem point_on_curve_iff_function_zero :
  f (x₀, y₀) = 0 ↔ (x₀, y₀) ∈ {p : ℝ × ℝ | f p = 0} := by sorry

end point_on_curve_iff_function_zero_l2835_283588


namespace min_value_theorem_l2835_283575

theorem min_value_theorem (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + a*b + a*c + b*c = 4) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 → 2*x + y + z ≥ 2*a + b + c :=
by sorry

end min_value_theorem_l2835_283575


namespace students_voting_both_issues_l2835_283517

theorem students_voting_both_issues 
  (total_students : ℕ) 
  (first_issue : ℕ) 
  (second_issue : ℕ) 
  (against_both : ℕ) 
  (h1 : total_students = 250)
  (h2 : first_issue = 171)
  (h3 : second_issue = 141)
  (h4 : against_both = 39) :
  first_issue + second_issue - (total_students - against_both) = 101 := by
sorry

end students_voting_both_issues_l2835_283517


namespace max_largest_integer_l2835_283579

theorem max_largest_integer (a b c d e : ℕ+) : 
  (a + b + c + d + e : ℚ) / 5 = 70 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 340 :=
sorry

end max_largest_integer_l2835_283579


namespace otto_sharpening_cost_l2835_283533

/-- Represents the cost structure for knife sharpening --/
structure Sharpening :=
  (first_knife : ℝ)
  (next_three : ℝ)
  (five_to_ten : ℝ)
  (after_ten : ℝ)

/-- Represents the knife collection --/
structure KnifeCollection :=
  (total : ℕ)
  (chefs : ℕ)
  (paring : ℕ)

/-- Calculates the total cost of sharpening knives --/
def sharpeningCost (s : Sharpening) (k : KnifeCollection) : ℝ :=
  sorry

/-- Theorem stating the total cost of sharpening Otto's knives --/
theorem otto_sharpening_cost :
  let s : Sharpening := {
    first_knife := 6.00,
    next_three := 4.50,
    five_to_ten := 3.75,
    after_ten := 3.25
  }
  let k : KnifeCollection := {
    total := 15,
    chefs := 3,
    paring := 4
  }
  let chefs_discount := 0.15
  let paring_discount := 0.10
  sharpeningCost s k - (chefs_discount * (s.first_knife + 2 * s.next_three)) - 
    (paring_discount * (2 * s.next_three + 2 * s.five_to_ten)) = 54.35 :=
by sorry

end otto_sharpening_cost_l2835_283533


namespace max_parts_three_planes_is_eight_l2835_283516

/-- The maximum number of parts into which three planes can divide space -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts into which three planes can divide space is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by
  sorry

end max_parts_three_planes_is_eight_l2835_283516


namespace clara_hughes_book_purchase_l2835_283597

theorem clara_hughes_book_purchase :
  let total_volumes : ℕ := 12
  let paperback_cost : ℕ := 18
  let hardcover_cost : ℕ := 28
  let total_spent : ℕ := 276
  ∃ (hardcover_count : ℕ) (paperback_count : ℕ),
    hardcover_count + paperback_count = total_volumes ∧
    hardcover_cost * hardcover_count + paperback_cost * paperback_count = total_spent ∧
    hardcover_count = 6 :=
by sorry

end clara_hughes_book_purchase_l2835_283597


namespace endpoint_coordinate_sum_l2835_283542

/-- Given a line segment with midpoint (10, -14) and one endpoint at (12, -6),
    the sum of the coordinates of the other endpoint is -14. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  (x + 12) / 2 = 10 →  -- Midpoint x-coordinate condition
  (y - 6) / 2 = -14 →  -- Midpoint y-coordinate condition
  x + y = -14 :=
by sorry

end endpoint_coordinate_sum_l2835_283542


namespace min_value_2a_plus_b_l2835_283501

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let z : ℂ := (3 * a + 2 * Complex.I) * (b - Complex.I)
  (z.re = 4) →
  (∀ x y : ℝ, x > 0 → y > 0 →
    let w : ℂ := (3 * x + 2 * Complex.I) * (y - Complex.I)
    w.re = 4 → 2 * x + y ≥ 2 * a + b) →
  2 * a + b = 4 * Real.sqrt 3 / 3 :=
by sorry

end min_value_2a_plus_b_l2835_283501


namespace floor_equation_unique_solution_l2835_283569

theorem floor_equation_unique_solution (n : ℤ) :
  (Int.floor (n^2 / 4) - Int.floor (n / 2)^2 = 2) ↔ n = 5 := by
  sorry

end floor_equation_unique_solution_l2835_283569


namespace choose_four_diff_suits_standard_deck_l2835_283576

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    num_suits := 4,
    cards_per_suit := 13,
    h_total := rfl }

/-- The number of ways to choose 4 cards of different suits from a standard deck -/
def choose_four_diff_suits (d : Deck) : Nat :=
  d.cards_per_suit ^ d.num_suits

/-- Theorem stating the number of ways to choose 4 cards of different suits from a standard deck -/
theorem choose_four_diff_suits_standard_deck :
  choose_four_diff_suits standard_deck = 28561 := by
  sorry

end choose_four_diff_suits_standard_deck_l2835_283576


namespace square_side_length_l2835_283596

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 625 →
  side * side = area →
  side = 25 := by
  sorry

end square_side_length_l2835_283596


namespace square_area_specific_vertices_l2835_283508

/-- The area of a square with vertices at (1, 1), (-4, 2), (-3, 7), and (2, 6) is 26 square units. -/
theorem square_area_specific_vertices : 
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-4, 2)
  let R : ℝ × ℝ := (-3, 7)
  let S : ℝ × ℝ := (2, 6)
  let side_length := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  side_length^2 = 26 := by sorry

end square_area_specific_vertices_l2835_283508


namespace sum_of_first_6n_equals_465_l2835_283506

/-- The value of n that satisfies the given condition -/
def n : ℕ := 5

/-- The sum of the first k positive integers -/
def sum_first_k (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The condition that the sum of the first 5n positive integers is 325 more than the sum of the first n positive integers -/
axiom condition : sum_first_k (5 * n) = sum_first_k n + 325

/-- The theorem to be proved -/
theorem sum_of_first_6n_equals_465 : sum_first_k (6 * n) = 465 := by
  sorry

end sum_of_first_6n_equals_465_l2835_283506


namespace triangle_side_length_l2835_283560

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h1 : B = 45 * π / 180)
  (h2 : C = 60 * π / 180)
  (h3 : c = 1)
  (h4 : a + b + c = A + B + C)
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h6 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h7 : A + B + C = π) : 
  b = Real.sqrt 6 / 3 := by
sorry

end triangle_side_length_l2835_283560


namespace simplify_expressions_l2835_283562

theorem simplify_expressions : 
  (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27 = 4 * Real.sqrt 3) ∧
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6) := by
  sorry

end simplify_expressions_l2835_283562


namespace hyperbola_equation_l2835_283598

/-- Given a parabola and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Parabola equation
  (∃ (x₀ y₀ : ℝ), y₀^2 = 8*x₀) →
  -- Hyperbola general form
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
  -- Directrix of parabola (x = -2) passes through a focus of the hyperbola
  (∃ (x₁ y₁ : ℝ), x₁ = -2 ∧ x₁^2/a^2 - y₁^2/b^2 = 1) →
  -- Eccentricity of the hyperbola is 2
  (∃ (c : ℝ), c/a = 2 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  x^2 - y^2/3 = 1 :=
by sorry

end hyperbola_equation_l2835_283598


namespace power_function_comparison_l2835_283584

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_comparison
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_condition : f 8 = 4) :
  f (Real.sqrt 2 / 2) > f (-Real.sqrt 3 / 3) := by
  sorry

end power_function_comparison_l2835_283584


namespace kite_perimeter_l2835_283554

/-- A kite with given side lengths -/
structure Kite where
  short_side : ℝ
  long_side : ℝ

/-- The perimeter of a kite -/
def perimeter (k : Kite) : ℝ :=
  2 * k.short_side + 2 * k.long_side

/-- Theorem: The perimeter of a kite with short sides 10 inches and long sides 15 inches is 50 inches -/
theorem kite_perimeter : 
  let k : Kite := { short_side := 10, long_side := 15 }
  perimeter k = 50 := by sorry

end kite_perimeter_l2835_283554


namespace complex_product_quadrant_l2835_283574

theorem complex_product_quadrant : 
  let z : ℂ := (-1 + 2*I) * (3 - I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end complex_product_quadrant_l2835_283574


namespace arrasta_um_min_moves_l2835_283583

/-- Represents the Arrasta Um game board -/
structure ArrastaUmBoard (n : ℕ) where
  size : n ≥ 2

/-- Represents a move in the Arrasta Um game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Calculates the minimum number of moves required to complete the game -/
def minMoves (board : ArrastaUmBoard n) : ℕ :=
  6 * n - 8

/-- Theorem stating that the minimum number of moves to complete Arrasta Um on an n × n board is 6n - 8 -/
theorem arrasta_um_min_moves (n : ℕ) (board : ArrastaUmBoard n) :
  minMoves board = 6 * n - 8 :=
by sorry

end arrasta_um_min_moves_l2835_283583


namespace remaining_customers_l2835_283523

/-- Proves that the number of customers remaining after some left is 5 -/
theorem remaining_customers (initial : ℕ) (remaining : ℕ) (new : ℕ) (final : ℕ)
  (h1 : initial = 8)
  (h2 : remaining < initial)
  (h3 : new = 99)
  (h4 : final = 104)
  (h5 : remaining + new = final) :
  remaining = 5 := by sorry

end remaining_customers_l2835_283523


namespace triangle_side_integral_difference_l2835_283512

def triangle_side_difference (x : ℤ) : Prop :=
  x > 2 ∧ x < 18

theorem triangle_side_integral_difference :
  (∃ x_max x_min : ℤ, 
    (∀ x : ℤ, triangle_side_difference x → x ≤ x_max ∧ x ≥ x_min) ∧
    triangle_side_difference x_max ∧
    triangle_side_difference x_min ∧
    x_max - x_min = 14) :=
sorry

end triangle_side_integral_difference_l2835_283512


namespace total_legs_l2835_283524

/-- The number of legs for a chicken -/
def chicken_legs : ℕ := 2

/-- The number of legs for a sheep -/
def sheep_legs : ℕ := 4

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := 5

/-- Theorem stating the total number of legs among the animals Farmer Brown fed -/
theorem total_legs : num_chickens * chicken_legs + num_sheep * sheep_legs = 34 := by
  sorry

end total_legs_l2835_283524


namespace rectangle_long_side_l2835_283558

/-- Given a rectangle with perimeter 30 cm and short side 7 cm, prove the long side is 8 cm -/
theorem rectangle_long_side (perimeter : ℝ) (short_side : ℝ) (long_side : ℝ) : 
  perimeter = 30 ∧ short_side = 7 ∧ perimeter = 2 * (short_side + long_side) → long_side = 8 := by
  sorry

end rectangle_long_side_l2835_283558


namespace correct_num_dancers_l2835_283536

/-- The number of dancers on the dance team -/
def num_dancers : ℕ := 8

/-- The number of braids per dancer -/
def braids_per_dancer : ℕ := 5

/-- The time in seconds to make one braid -/
def seconds_per_braid : ℕ := 30

/-- The total time in minutes to braid all dancers' hair -/
def total_time_minutes : ℕ := 20

/-- Theorem stating that the number of dancers is correct given the conditions -/
theorem correct_num_dancers :
  num_dancers * braids_per_dancer * seconds_per_braid = total_time_minutes * 60 :=
by sorry

end correct_num_dancers_l2835_283536


namespace rectangle_areas_sum_l2835_283530

theorem rectangle_areas_sum : 
  let width : ℕ := 3
  let lengths : List ℕ := [1, 3, 5, 7, 9].map (λ x => x^2)
  let areas : List ℕ := lengths.map (λ l => width * l)
  areas.sum = 495 := by
sorry

end rectangle_areas_sum_l2835_283530


namespace card_distribution_l2835_283538

theorem card_distribution (total : ℕ) (ratio_m : ℕ) (ratio_h : ℕ) (ratio_g : ℕ) 
  (h_total : total = 363)
  (h_ratio : ratio_m = 35 ∧ ratio_h = 30 ∧ ratio_g = 56) :
  ∃ (m h g : ℕ), 
    m + h + g = total ∧ 
    m * (ratio_h + ratio_g) = ratio_m * (h + g) ∧
    h * (ratio_m + ratio_g) = ratio_h * (m + g) ∧
    g * (ratio_m + ratio_h) = ratio_g * (m + h) ∧
    m = 105 ∧ h = 90 ∧ g = 168 := by
  sorry

end card_distribution_l2835_283538


namespace remainder_108_112_mod_11_l2835_283580

theorem remainder_108_112_mod_11 : (108 * 112) % 11 = 6 := by
  sorry

end remainder_108_112_mod_11_l2835_283580
