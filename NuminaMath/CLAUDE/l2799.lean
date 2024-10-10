import Mathlib

namespace green_hats_count_l2799_279936

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_price : ℚ) :
  total_hats = 85 →
  blue_cost = 6 →
  green_cost = 7 →
  total_price = 548 →
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_price ∧
    green_hats = 38 := by
  sorry

end green_hats_count_l2799_279936


namespace power_multiplication_l2799_279947

theorem power_multiplication (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end power_multiplication_l2799_279947


namespace company_profit_ratio_l2799_279992

/-- Represents the revenues of a company in a given year -/
structure Revenue where
  amount : ℝ

/-- Calculates the profit given a revenue and a profit percentage -/
def profit (revenue : Revenue) (percentage : ℝ) : ℝ := revenue.amount * percentage

/-- Company N's revenues over three years -/
structure CompanyN where
  revenue2008 : Revenue
  revenue2009 : Revenue
  revenue2010 : Revenue
  revenue2009_eq : revenue2009.amount = 0.8 * revenue2008.amount
  revenue2010_eq : revenue2010.amount = 1.3 * revenue2009.amount

/-- Company M's revenues over three years -/
structure CompanyM where
  revenue : Revenue

theorem company_profit_ratio (n : CompanyN) (m : CompanyM) :
  (profit n.revenue2008 0.08 + profit n.revenue2009 0.15 + profit n.revenue2010 0.10) /
  (profit m.revenue 0.12 + profit m.revenue 0.18 + profit m.revenue 0.14) =
  (0.304 * n.revenue2008.amount) / (0.44 * m.revenue.amount) := by
  sorry

end company_profit_ratio_l2799_279992


namespace line_through_origin_and_negative_one_l2799_279994

/-- The angle of inclination (in degrees) of a line passing through two points -/
def angleOfInclination (x1 y1 x2 y2 : ℝ) : ℝ := sorry

/-- A line passes through the origin (0, 0) and the point (-1, -1) -/
theorem line_through_origin_and_negative_one : 
  angleOfInclination 0 0 (-1) (-1) = 45 := by sorry

end line_through_origin_and_negative_one_l2799_279994


namespace solve_for_y_l2799_279941

theorem solve_for_y (x y : ℝ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 := by
  sorry

end solve_for_y_l2799_279941


namespace line_slope_intercept_sum_l2799_279951

/-- Given two points on a line, prove that the sum of the slope and y-intercept is 3 -/
theorem line_slope_intercept_sum (x₁ y₁ x₂ y₂ m b : ℝ) : 
  x₁ = 1 → y₁ = 3 → x₂ = -3 → y₂ = -1 →
  (y₂ - y₁) = m * (x₂ - x₁) →
  y₁ = m * x₁ + b →
  m + b = 3 := by
sorry

end line_slope_intercept_sum_l2799_279951


namespace square_of_simplified_fraction_l2799_279952

theorem square_of_simplified_fraction : 
  (126 / 882 : ℚ)^2 = 1 / 49 := by sorry

end square_of_simplified_fraction_l2799_279952


namespace fraction_integer_iff_p_equals_three_l2799_279939

theorem fraction_integer_iff_p_equals_three (p : ℕ+) :
  (↑p : ℚ) > 0 →
  (∃ (n : ℕ), n > 0 ∧ (5 * p + 45 : ℚ) / (3 * p - 8 : ℚ) = ↑n) ↔ p = 3 := by
  sorry

end fraction_integer_iff_p_equals_three_l2799_279939


namespace arithmetic_sequence_sum_l2799_279982

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d →
  (a 1 + a 2 + a 3 = 15) →
  (a 1 * a 2 * a 3 = 80) →
  (a 11 + a 12 + a 13 = 105) :=
by
  sorry

end arithmetic_sequence_sum_l2799_279982


namespace trajectory_intersection_properties_l2799_279918

-- Define the trajectory of point M
def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) = |x| + 1

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  y = x + 1

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x - 1)

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem trajectory_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (trajectory A.1 A.2 ∧ line_l2 A.1 A.2) ∧
    (trajectory B.1 B.2 ∧ line_l2 B.1 B.2) ∧
    (A ≠ B) ∧
    (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) ∧
    (Real.sqrt ((A.1 - point_F.1)^2 + (A.2 - point_F.2)^2) *
     Real.sqrt ((B.1 - point_F.1)^2 + (B.2 - point_F.2)^2) = 16) :=
sorry

end trajectory_intersection_properties_l2799_279918


namespace difference_of_squares_example_l2799_279949

theorem difference_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end difference_of_squares_example_l2799_279949


namespace average_of_combined_results_l2799_279985

theorem average_of_combined_results :
  let n₁ : ℕ := 30
  let avg₁ : ℚ := 20
  let n₂ : ℕ := 20
  let avg₂ : ℚ := 30
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  total_sum / total_count = 24 := by
  sorry

end average_of_combined_results_l2799_279985


namespace max_value_sum_of_square_roots_l2799_279979

theorem max_value_sum_of_square_roots (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) 
  (h_sum : a + b + c = 7) : 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 6 ∧
  (∃ (a₀ b₀ c₀ : ℝ), 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀ + b₀ + c₀ = 7 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = 6) :=
by sorry

end max_value_sum_of_square_roots_l2799_279979


namespace share_difference_l2799_279957

/-- Given four shares in the ratio 3:3:7:4, where the second share is 1500
    and the fourth share is 2000, the difference between the largest share
    and the second-largest share is 1500. -/
theorem share_difference (shares : Fin 4 → ℕ) : 
  (∃ x : ℕ, shares 0 = 3*x ∧ shares 1 = 3*x ∧ shares 2 = 7*x ∧ shares 3 = 4*x) →
  shares 1 = 1500 →
  shares 3 = 2000 →
  (shares 2 - (max (shares 0) (shares 3))) = 1500 := by
sorry

end share_difference_l2799_279957


namespace sum_zero_not_all_negative_l2799_279986

theorem sum_zero_not_all_negative (a b c : ℝ) (h : a + b + c = 0) :
  ¬(a < 0 ∧ b < 0 ∧ c < 0) := by
  sorry

end sum_zero_not_all_negative_l2799_279986


namespace circle_graph_parts_sum_to_one_l2799_279967

theorem circle_graph_parts_sum_to_one :
  let white : ℚ := 1/2
  let black : ℚ := 1/4
  let gray : ℚ := 1/8
  let blue : ℚ := 1/8
  white + black + gray + blue = 1 := by
sorry

end circle_graph_parts_sum_to_one_l2799_279967


namespace triangle_properties_l2799_279930

theorem triangle_properties (a b c A B C : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π)
  (h_sides : a = 2)
  (h_equation : (b + 2) * (Real.sin A - Real.sin B) = c * (Real.sin B + Real.sin C)) :
  A = 2 * π / 3 ∧
  ∃ S : ℝ, S > 0 ∧ S ≤ Real.sqrt 3 / 3 ∧
    S = 1 / 2 * a * b * Real.sin C :=
by sorry

end triangle_properties_l2799_279930


namespace nine_point_circle_triangles_l2799_279965

/-- Given 9 points on a circle, this function calculates the number of triangles
    formed by the intersections of chords inside the circle. -/
def triangles_in_circle (n : ℕ) : ℕ :=
  if n = 9 then
    (Nat.choose n 6) * (Nat.choose 6 2) * (Nat.choose 4 2) / 6
  else
    0

/-- Theorem stating that for 9 points on a circle, with chords connecting every pair
    of points and no three chords intersecting at a single point inside the circle,
    the number of triangles formed with all vertices in the interior is 210. -/
theorem nine_point_circle_triangles :
  triangles_in_circle 9 = 210 := by
  sorry

#eval triangles_in_circle 9

end nine_point_circle_triangles_l2799_279965


namespace jennifers_cans_count_l2799_279998

/-- The number of cans Jennifer brought home from the store -/
def jennifers_total_cans (initial_cans : ℕ) (marks_cans : ℕ) : ℕ :=
  initial_cans + (6 * marks_cans) / 5

/-- Theorem stating the total number of cans Jennifer brought home -/
theorem jennifers_cans_count : jennifers_total_cans 40 50 = 100 := by
  sorry

end jennifers_cans_count_l2799_279998


namespace max_sum_constrained_l2799_279945

theorem max_sum_constrained (x y : ℝ) (h : x^2 + y^2 + x*y = 1) :
  x + y ≤ 2 * Real.sqrt 3 / 3 :=
sorry

end max_sum_constrained_l2799_279945


namespace increasing_sufficient_not_necessary_l2799_279907

/-- A function f: ℝ → ℝ is increasing on [1, +∞) -/
def IncreasingOnIntervalOneInf (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y → f x < f y

/-- A sequence a_n = f(n) is increasing -/
def IncreasingSequence (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → f n < f (n + 1)

/-- The main theorem stating that IncreasingOnIntervalOneInf is sufficient but not necessary for IncreasingSequence -/
theorem increasing_sufficient_not_necessary (f : ℝ → ℝ) :
  (IncreasingOnIntervalOneInf f → IncreasingSequence f) ∧
  ∃ g : ℝ → ℝ, IncreasingSequence g ∧ ¬IncreasingOnIntervalOneInf g :=
sorry

end increasing_sufficient_not_necessary_l2799_279907


namespace min_value_w_l2799_279935

theorem min_value_w (x y z : ℝ) :
  x^2 + 4*y^2 + 8*x - 6*y + z - 20 ≥ z - 38.25 := by
  sorry

end min_value_w_l2799_279935


namespace wallpaper_removal_time_l2799_279969

/-- Calculates the time required to remove wallpaper from remaining walls -/
def time_to_remove_wallpaper (time_per_wall : ℕ) (dining_room_walls : ℕ) (living_room_walls : ℕ) (walls_completed : ℕ) : ℕ :=
  time_per_wall * (dining_room_walls + living_room_walls - walls_completed)

/-- Proves that given the conditions, the time required to remove the remaining wallpaper is 14 hours -/
theorem wallpaper_removal_time :
  let time_per_wall : ℕ := 2
  let dining_room_walls : ℕ := 4
  let living_room_walls : ℕ := 4
  let walls_completed : ℕ := 1
  time_to_remove_wallpaper time_per_wall dining_room_walls living_room_walls walls_completed = 14 := by
  sorry

#eval time_to_remove_wallpaper 2 4 4 1

end wallpaper_removal_time_l2799_279969


namespace angle_sequence_convergence_l2799_279934

noncomputable def angle_sequence (α : ℝ) : ℕ → ℝ
  | 0 => 0  -- Initial value doesn't affect the limit
  | n + 1 => (Real.pi - α - angle_sequence α n) / 2

theorem angle_sequence_convergence (α : ℝ) (h : 0 < α ∧ α < Real.pi) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |angle_sequence α n - L| < ε) ∧
             L = (Real.pi - α) / 3 :=
by sorry

end angle_sequence_convergence_l2799_279934


namespace modular_congruence_solution_l2799_279944

theorem modular_congruence_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ (99 * n) % 103 = 73 % 103 ∧ n = 68 := by
  sorry

end modular_congruence_solution_l2799_279944


namespace tangent_circles_constant_l2799_279988

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

/-- The theorem stating the value of 'a' for which the given circles are tangent -/
theorem tangent_circles_constant (a : ℝ) : 
  are_tangent (0, 0) 1 (-4, a) 5 ↔ a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 := by
  sorry

end tangent_circles_constant_l2799_279988


namespace carly_bbq_cooking_time_l2799_279980

/-- Represents the cooking scenario for Carly's BBQ --/
structure BBQScenario where
  cook_time_per_side : ℕ
  burgers_per_batch : ℕ
  total_guests : ℕ
  guests_wanting_two : ℕ
  guests_wanting_one : ℕ

/-- Calculates the total cooking time for all burgers --/
def total_cooking_time (scenario : BBQScenario) : ℕ :=
  let total_burgers := 2 * scenario.guests_wanting_two + scenario.guests_wanting_one
  let num_batches := (total_burgers + scenario.burgers_per_batch - 1) / scenario.burgers_per_batch
  num_batches * (2 * scenario.cook_time_per_side)

/-- Theorem stating that the total cooking time for Carly's scenario is 72 minutes --/
theorem carly_bbq_cooking_time :
  total_cooking_time {
    cook_time_per_side := 4,
    burgers_per_batch := 5,
    total_guests := 30,
    guests_wanting_two := 15,
    guests_wanting_one := 15
  } = 72 := by
  sorry

end carly_bbq_cooking_time_l2799_279980


namespace fifth_inequality_holds_l2799_279995

theorem fifth_inequality_holds : 
  1 + (1 : ℝ) / 2^2 + 1 / 3^2 + 1 / 4^2 + 1 / 5^2 + 1 / 6^2 < (2 * 5 + 1) / (5 + 1) :=
by sorry

end fifth_inequality_holds_l2799_279995


namespace quadratic_equation_from_sum_and_difference_l2799_279971

theorem quadratic_equation_from_sum_and_difference (x y : ℝ) 
  (sum_cond : x + y = 10) 
  (diff_cond : |x - y| = 12) : 
  (∀ z : ℝ, (z - x) * (z - y) = 0 ↔ z^2 - 10*z - 11 = 0) := by
  sorry

end quadratic_equation_from_sum_and_difference_l2799_279971


namespace square_triangle_equal_area_l2799_279964

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) :
  square_perimeter = 80 →
  triangle_height = 40 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * (square_perimeter / 4) →
  (square_perimeter / 4) = 20 :=
by sorry

end square_triangle_equal_area_l2799_279964


namespace parabola_smallest_a_l2799_279929

theorem parabola_smallest_a (a b c : ℝ) : 
  a > 0 ∧ 
  b^2 - 4*a*c = 7 ∧
  (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = a*(x - 1/3)^2) →
  a ≥ 63/20 ∧ ∃ b c : ℝ, b^2 - 4*a*c = 7 ∧ (∀ x y : ℝ, y = a*x^2 + b*x + c ↔ y + 5/9 = 63/20*(x - 1/3)^2) :=
sorry

end parabola_smallest_a_l2799_279929


namespace regular_polygon_sides_l2799_279937

/-- A regular polygon with perimeter 150 and side length 15 has 10 sides -/
theorem regular_polygon_sides (p : ℕ) (perimeter side_length : ℝ) 
  (h_regular : p ≥ 3)
  (h_perimeter : perimeter = 150)
  (h_side : side_length = 15)
  (h_relation : perimeter = p * side_length) : p = 10 := by
  sorry

end regular_polygon_sides_l2799_279937


namespace angela_finished_nine_problems_l2799_279943

/-- The number of math problems Angela and her friends are working on -/
def total_problems : ℕ := 20

/-- The number of problems Martha has finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna has finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Mark has finished -/
def mark_problems : ℕ := jenna_problems / 2

/-- The number of problems Angela has finished on her own -/
def angela_problems : ℕ := total_problems - (martha_problems + jenna_problems + mark_problems)

theorem angela_finished_nine_problems : angela_problems = 9 := by
  sorry

end angela_finished_nine_problems_l2799_279943


namespace max_value_of_function_l2799_279987

theorem max_value_of_function (x : ℝ) : 1 + 1 / (x^2 + 2*x + 2) ≤ 2 := by
  sorry

end max_value_of_function_l2799_279987


namespace geometric_sequence_property_l2799_279919

theorem geometric_sequence_property (x : ℝ) (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence property
  a 1 = Real.sin x →
  a 2 = Real.cos x →
  a 3 = Real.tan x →
  a 8 = 1 + Real.cos x :=
by sorry

end geometric_sequence_property_l2799_279919


namespace equal_projections_implies_a_equals_one_l2799_279915

-- Define the points and vectors
def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 4)

def OA (a : ℝ) : ℝ × ℝ := A a
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem equal_projections_implies_a_equals_one (a : ℝ) :
  dot_product (OA a) OC = dot_product OB OC → a = 1 := by
  sorry

end equal_projections_implies_a_equals_one_l2799_279915


namespace books_in_box_l2799_279960

def box_weight : ℕ := 42
def book_weight : ℕ := 3

theorem books_in_box : 
  box_weight / book_weight = 14 := by sorry

end books_in_box_l2799_279960


namespace second_hand_movement_l2799_279954

/-- Represents the movement of clock hands -/
def ClockMovement : Type :=
  { minutes : ℕ // minutes > 0 }

/-- Converts minutes to seconds -/
def minutesToSeconds (m : ClockMovement) : ℕ :=
  m.val * 60

/-- Calculates the number of circles the second hand moves -/
def secondHandCircles (m : ClockMovement) : ℕ :=
  minutesToSeconds m / 60

/-- The theorem to be proved -/
theorem second_hand_movement (m : ClockMovement) (h : m.val = 2) :
  secondHandCircles m = 2 := by
  sorry

end second_hand_movement_l2799_279954


namespace tree_prob_five_vertices_l2799_279970

/-- The number of vertices in the graph -/
def n : ℕ := 5

/-- The probability of drawing an edge between any two vertices -/
def edge_prob : ℚ := 1/2

/-- The number of labeled trees on n vertices -/
def num_labeled_trees (n : ℕ) : ℕ := n^(n-2)

/-- The total number of possible graphs on n vertices -/
def total_graphs (n : ℕ) : ℕ := 2^(n.choose 2)

/-- The probability that a randomly generated graph is a tree -/
def tree_probability (n : ℕ) : ℚ := (num_labeled_trees n : ℚ) / (total_graphs n : ℚ)

theorem tree_prob_five_vertices :
  tree_probability n = 125 / 1024 :=
sorry

end tree_prob_five_vertices_l2799_279970


namespace system_solution_proof_l2799_279983

theorem system_solution_proof (x y : ℝ) : 
  (4 / (x^2 + y^2) + x^2 * y^2 = 5 ∧ x^4 + y^4 + 3 * x^2 * y^2 = 20) ↔ 
  ((x = Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = Real.sqrt 2 ∧ y = -Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = Real.sqrt 2) ∨ 
   (x = -Real.sqrt 2 ∧ y = -Real.sqrt 2)) :=
by sorry

end system_solution_proof_l2799_279983


namespace distance_acaster_beetown_is_315_l2799_279924

/-- The distance from Acaster to Beetown in kilometers. -/
def distance_acaster_beetown : ℝ := 315

/-- Lewis's speed in km/h. -/
def lewis_speed : ℝ := 70

/-- Geraint's speed in km/h. -/
def geraint_speed : ℝ := 30

/-- The distance from the meeting point to Beetown in kilometers. -/
def distance_meeting_beetown : ℝ := 105

/-- The time Lewis spends in Beetown in hours. -/
def lewis_stop_time : ℝ := 1

theorem distance_acaster_beetown_is_315 :
  let total_time := distance_acaster_beetown / geraint_speed
  let lewis_travel_time := total_time - lewis_stop_time
  lewis_travel_time * lewis_speed = distance_acaster_beetown + distance_meeting_beetown ∧
  total_time * geraint_speed = distance_acaster_beetown - distance_meeting_beetown ∧
  distance_acaster_beetown = 315 := by
  sorry

#check distance_acaster_beetown_is_315

end distance_acaster_beetown_is_315_l2799_279924


namespace fish_per_family_member_l2799_279948

def fish_distribution (family_size : ℕ) (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) : ℕ :=
  let total_eyes := eyes_eaten + eyes_to_dog
  let total_fish := total_eyes / eyes_per_fish
  total_fish / family_size

theorem fish_per_family_member :
  fish_distribution 3 22 2 2 = 4 := by
  sorry

end fish_per_family_member_l2799_279948


namespace total_amount_divided_l2799_279932

/-- Proves that the total amount divided is 3500, given the specified conditions --/
theorem total_amount_divided (first_part : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (total_interest : ℝ) :
  first_part = 1550 →
  interest_rate1 = 0.03 →
  interest_rate2 = 0.05 →
  total_interest = 144 →
  ∃ (total : ℝ), 
    total = 3500 ∧
    first_part * interest_rate1 + (total - first_part) * interest_rate2 = total_interest :=
by
  sorry


end total_amount_divided_l2799_279932


namespace smallest_m_theorem_l2799_279968

def is_multiple_of_100 (n : ℕ) : Prop := ∃ k : ℕ, n = 100 * k

def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def satisfies_conditions (m : ℕ) : Prop :=
  is_multiple_of_100 m ∧ count_divisors m = 100

theorem smallest_m_theorem :
  ∃! m : ℕ, satisfies_conditions m ∧
    ∀ n : ℕ, satisfies_conditions n → m ≤ n ∧
    m / 100 = 2700 := by sorry

end smallest_m_theorem_l2799_279968


namespace tournament_outcomes_l2799_279911

/-- Represents a bowler in the tournament -/
inductive Bowler
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents a match between two bowlers -/
structure Match where
  player1 : Bowler
  player2 : Bowler

/-- Represents the tournament structure -/
structure Tournament where
  initialRound : List Match
  subsequentRounds : List Match

/-- Represents the outcome of the tournament -/
structure Outcome where
  prizeOrder : List Bowler

/-- The number of possible outcomes for the tournament -/
def numberOfOutcomes (t : Tournament) : Nat :=
  2^5

/-- Theorem stating that the number of possible outcomes is 32 -/
theorem tournament_outcomes (t : Tournament) :
  numberOfOutcomes t = 32 := by
  sorry

end tournament_outcomes_l2799_279911


namespace max_cosine_difference_value_l2799_279972

def max_cosine_difference (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  a₃ = a₂ + a₁ ∧ 
  a₄ = a₃ + a₂ ∧ 
  ∃ (a b c : ℝ), ∀ n ∈ ({1, 2, 3, 4} : Set ℕ), 
    a * n^2 + b * n + c = Real.cos (if n = 1 then a₁ 
                                    else if n = 2 then a₂ 
                                    else if n = 3 then a₃ 
                                    else a₄)

theorem max_cosine_difference_value :
  ∀ a₁ a₂ a₃ a₄ : ℝ, max_cosine_difference a₁ a₂ a₃ a₄ →
    Real.cos a₁ - Real.cos a₄ ≤ -9 + 3 * Real.sqrt 13 :=
sorry

end max_cosine_difference_value_l2799_279972


namespace no_equal_perimeter_area_volume_cuboid_l2799_279928

theorem no_equal_perimeter_area_volume_cuboid :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (4 * (a + b + c) = 2 * (a * b + b * c + c * a)) ∧
    (4 * (a + b + c) = a * b * c) :=
by sorry

end no_equal_perimeter_area_volume_cuboid_l2799_279928


namespace land_division_theorem_l2799_279923

/-- Represents a rectangular piece of land --/
structure Land where
  length : ℝ
  width : ℝ

/-- Represents a division of land into three sections --/
structure LandDivision where
  section1 : Land
  section2 : Land
  section3 : Land

def Land.area (l : Land) : ℝ := l.length * l.width

def LandDivision.isValid (ld : LandDivision) (totalLand : Land) : Prop :=
  ld.section1.area + ld.section2.area + ld.section3.area = totalLand.area ∧
  ld.section1.area = ld.section2.area ∧
  ld.section2.area = ld.section3.area

def LandDivision.fenceLength (ld : LandDivision) : ℝ :=
  ld.section1.length + ld.section2.length + ld.section3.length

def countValidDivisions (totalLand : Land) : ℕ :=
  sorry

def minFenceLength (totalLand : Land) : ℝ :=
  sorry

theorem land_division_theorem (totalLand : Land) 
  (h1 : totalLand.length = 25)
  (h2 : totalLand.width = 36) :
  countValidDivisions totalLand = 4 ∧ 
  minFenceLength totalLand = 49 := by
  sorry

end land_division_theorem_l2799_279923


namespace three_heads_in_eight_tosses_l2799_279984

def biased_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem three_heads_in_eight_tosses :
  biased_coin_probability 8 3 (1/3) = 1792/6561 := by
  sorry

end three_heads_in_eight_tosses_l2799_279984


namespace square_sum_product_equality_l2799_279940

theorem square_sum_product_equality : (6 + 10)^2 + (6^2 + 10^2 + 6 * 10) = 452 := by
  sorry

end square_sum_product_equality_l2799_279940


namespace concert_ticket_cost_daria_concert_money_l2799_279922

theorem concert_ticket_cost (ticket_price : ℕ) (current_money : ℕ) : ℕ :=
  let total_tickets : ℕ := 4
  let total_cost : ℕ := total_tickets * ticket_price
  let additional_money_needed : ℕ := total_cost - current_money
  additional_money_needed

theorem daria_concert_money : concert_ticket_cost 90 189 = 171 := by
  sorry

end concert_ticket_cost_daria_concert_money_l2799_279922


namespace cupboard_pricing_l2799_279974

/-- The cost price of a cupboard --/
def C : ℝ := sorry

/-- The selling price of the first cupboard --/
def SP₁ : ℝ := 0.84 * C

/-- The selling price of the second cupboard before tax --/
def SP₂ : ℝ := 0.756 * C

/-- The final selling price of the second cupboard after tax --/
def SP₂' : ℝ := 0.82404 * C

/-- The theorem stating the relationship between the cost price and the selling prices --/
theorem cupboard_pricing :
  2.32 * C - (SP₁ + SP₂') = 1800 :=
sorry

end cupboard_pricing_l2799_279974


namespace conic_is_hyperbola_l2799_279912

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

/-- The discriminant of the conic section -/
def discriminant : ℝ :=
  0^2 - 4 * 4 * (-9)

theorem conic_is_hyperbola :
  discriminant > 0 ∧ 
  (∃ a b c d : ℝ, ∀ x y : ℝ, 
    conic_equation x y ↔ ((x - a)^2 / b^2) - ((y - c)^2 / d^2) = 1) := by
  sorry

end conic_is_hyperbola_l2799_279912


namespace lassis_from_twelve_mangoes_l2799_279917

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  (11 * mangoes) / 2

/-- Theorem stating that Caroline can make 66 lassis from 12 mangoes -/
theorem lassis_from_twelve_mangoes :
  lassis_from_mangoes 12 = 66 := by
  sorry

end lassis_from_twelve_mangoes_l2799_279917


namespace bob_cleaning_time_l2799_279901

/-- Given that Alice takes 25 minutes to clean her room and Bob takes 2/5 of Alice's time,
    prove that Bob takes 10 minutes to clean his room. -/
theorem bob_cleaning_time (alice_time : ℕ) (bob_fraction : ℚ) 
  (h1 : alice_time = 25)
  (h2 : bob_fraction = 2 / 5) :
  bob_fraction * alice_time = 10 := by
  sorry

end bob_cleaning_time_l2799_279901


namespace sequence_property_l2799_279953

/-- Two infinite sequences of rational numbers -/
def Sequence := ℕ → ℚ

/-- Property that a sequence is nonconstant -/
def Nonconstant (s : Sequence) : Prop :=
  ∃ i j, s i ≠ s j

/-- Property that (sᵢ - sⱼ)(tᵢ - tⱼ) is an integer for all i and j -/
def IntegerProduct (s t : Sequence) : Prop :=
  ∀ i j, ∃ k : ℤ, (s i - s j) * (t i - t j) = k

theorem sequence_property (s t : Sequence) 
  (hs : Nonconstant s) (ht : Nonconstant t) (h : IntegerProduct s t) :
  ∃ r : ℚ, (∀ i j : ℕ, ∃ m n : ℤ, (s i - s j) * r = m ∧ (t i - t j) / r = n) :=
sorry

end sequence_property_l2799_279953


namespace f_properties_l2799_279920

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_period : ∀ x, f (x + 6) = f x + f 3
axiom f_increasing_on_0_3 : ∀ x₁ x₂, x₁ ∈ Set.Icc 0 3 → x₂ ∈ Set.Icc 0 3 → x₁ ≠ x₂ → 
  (f x₁ - f x₂) / (x₁ - x₂) > 0

-- Theorem to prove
theorem f_properties :
  (∀ x, f (x - 6) = f (-x)) ∧ 
  (¬ ∀ x₁ x₂, x₁ ∈ Set.Icc (-9) (-6) → x₂ ∈ Set.Icc (-9) (-6) → x₁ < x₂ → f x₁ < f x₂) ∧
  (¬ ∃ x₁ x₂ x₃ x₄ x₅, x₁ ∈ Set.Icc (-9) 9 ∧ x₂ ∈ Set.Icc (-9) 9 ∧ x₃ ∈ Set.Icc (-9) 9 ∧ 
    x₄ ∈ Set.Icc (-9) 9 ∧ x₅ ∈ Set.Icc (-9) 9 ∧ 
    f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
    x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧ 
    x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ 
    x₄ ≠ x₅) :=
by
  sorry


end f_properties_l2799_279920


namespace max_sundays_in_45_days_l2799_279990

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year with a starting day -/
structure Year where
  startDay : DayOfWeek

/-- Counts the number of Sundays in the first n days of a year -/
def countSundays (y : Year) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Sundays in the first 45 days of a year is 7 -/
theorem max_sundays_in_45_days :
  ∀ y : Year, countSundays y 45 ≤ 7 ∧ ∃ y' : Year, countSundays y' 45 = 7 :=
sorry

end max_sundays_in_45_days_l2799_279990


namespace friend_bike_speed_l2799_279973

/-- Proves that given Joann's speed and time, Fran's speed can be calculated for the same distance --/
theorem friend_bike_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) :
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check friend_bike_speed

end friend_bike_speed_l2799_279973


namespace teacher_worksheets_l2799_279993

theorem teacher_worksheets :
  ∀ (total_worksheets : ℕ) 
    (problems_per_worksheet : ℕ) 
    (graded_worksheets : ℕ) 
    (remaining_problems : ℕ),
  problems_per_worksheet = 7 →
  graded_worksheets = 8 →
  remaining_problems = 63 →
  problems_per_worksheet * (total_worksheets - graded_worksheets) = remaining_problems →
  total_worksheets = 17 := by
sorry

end teacher_worksheets_l2799_279993


namespace m_range_when_only_one_proposition_true_l2799_279989

def proposition_p (m : ℝ) : Prop := 0 < m ∧ m < 1/3

def proposition_q (m : ℝ) : Prop := 0 < m ∧ m < 15

theorem m_range_when_only_one_proposition_true :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m) →
  1/3 ≤ m ∧ m < 15 :=
sorry

end m_range_when_only_one_proposition_true_l2799_279989


namespace necessary_but_not_sufficient_condition_l2799_279913

theorem necessary_but_not_sufficient_condition :
  ∃ (x : ℝ), ((-2 < x ∧ x < 3) ∧ ¬(x^2 - 2*x - 3 < 0)) ∧
  ∀ (y : ℝ), (y^2 - 2*y - 3 < 0) → (-2 < y ∧ y < 3) :=
sorry

end necessary_but_not_sufficient_condition_l2799_279913


namespace three_integers_problem_l2799_279991

theorem three_integers_problem :
  ∃ (a b c : ℕ) (k : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    k > 0 ∧
    a + b + c = 93 ∧
    a * b * c = 3375 ∧
    b = k * a ∧
    c = k^2 * a ∧
    a = 3 ∧ b = 15 ∧ c = 75 :=
by sorry

end three_integers_problem_l2799_279991


namespace bug_return_probability_l2799_279996

/-- Probability of the bug being at the starting vertex after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting vertex on the twelfth move -/
theorem bug_return_probability : Q 12 = 44287 / 177147 := by
  sorry

end bug_return_probability_l2799_279996


namespace barycentric_centroid_vector_relation_l2799_279975

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given a triangle ABC, a point X with absolute barycentric coordinates (α:β:γ),
    and M as the centroid of triangle ABC, prove that:
    3 XM⃗ = (α - β)AB⃗ + (β - γ)BC⃗ + (γ - α)CA⃗ -/
theorem barycentric_centroid_vector_relation
  (A B C X M : V) (α β γ : ℝ) :
  X = α • A + β • B + γ • C →
  M = (1/3 : ℝ) • (A + B + C) →
  3 • (X - M) = (α - β) • (B - A) + (β - γ) • (C - B) + (γ - α) • (A - C) := by
  sorry

end barycentric_centroid_vector_relation_l2799_279975


namespace rectangle_dimension_change_l2799_279905

theorem rectangle_dimension_change (x : ℝ) : 
  (1 + x / 100) * (1 - 5 / 100) = 1 + 14.000000000000002 / 100 → x = 20 := by
  sorry

end rectangle_dimension_change_l2799_279905


namespace dog_bones_proof_l2799_279966

/-- The number of bones the dog dug up -/
def bones_dug_up : ℕ := 367

/-- The total number of bones the dog has now -/
def total_bones_now : ℕ := 860

/-- The initial number of bones the dog had -/
def initial_bones : ℕ := total_bones_now - bones_dug_up

theorem dog_bones_proof : initial_bones = 493 := by
  sorry

end dog_bones_proof_l2799_279966


namespace courtyard_breadth_l2799_279959

/-- Proves that the breadth of a rectangular courtyard is 6 meters -/
theorem courtyard_breadth : 
  ∀ (length width stone_length stone_width stone_count : ℝ),
  length = 15 →
  stone_count = 15 →
  stone_length = 3 →
  stone_width = 2 →
  length * width = stone_count * stone_length * stone_width →
  width = 6 := by
sorry

end courtyard_breadth_l2799_279959


namespace max_carlson_jars_l2799_279927

/-- Represents the initial state of jam jars for Carlson and Baby -/
structure JamJars where
  carlsonWeights : List Nat  -- List of weights of Carlson's jars
  babyWeights : List Nat     -- List of weights of Baby's jars

/-- Checks if the given JamJars satisfies the initial condition -/
def satisfiesInitialCondition (jars : JamJars) : Prop :=
  jars.carlsonWeights.sum = 13 * jars.babyWeights.sum

/-- Checks if the given JamJars satisfies the final condition after transfer -/
def satisfiesFinalCondition (jars : JamJars) : Prop :=
  let minWeight := jars.carlsonWeights.minimum?
  match minWeight with
  | some w => (jars.carlsonWeights.sum - w) = 8 * (jars.babyWeights.sum + w)
  | none => False

/-- Theorem stating the maximum number of jars Carlson could have initially had -/
theorem max_carlson_jars :
  ∀ jars : JamJars,
    satisfiesInitialCondition jars →
    satisfiesFinalCondition jars →
    jars.carlsonWeights.length ≤ 23 := by
  sorry

end max_carlson_jars_l2799_279927


namespace polynomial_simplification_l2799_279946

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 + 3 * x^3 - 5 * x^2 + 8 * x - 6) + (-6 * x^5 + x^3 + 4 * x^2 - 8 * x + 7) =
  -4 * x^5 + 4 * x^3 - x^2 + 1 := by sorry

end polynomial_simplification_l2799_279946


namespace largest_digit_sum_l2799_279963

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c y : ℕ) : 
  is_digit a → is_digit b → is_digit c →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  0 < y → y ≤ 10 →
  ∃ (a' b' c' : ℕ), is_digit a' ∧ is_digit b' ∧ is_digit c' ∧
    (a' * 100 + b' * 10 + c' : ℚ) / 1000 = 1 / y ∧
    a' + b' + c' ≤ 8 ∧
    a + b + c ≤ 8 :=
sorry

end largest_digit_sum_l2799_279963


namespace model_fit_relationships_l2799_279938

-- Define the model and its properties
structure Model where
  ssr : ℝ  -- Sum of squared residuals
  r_squared : ℝ  -- Coefficient of determination
  fit_quality : ℝ  -- Model fit quality (higher is better)

-- Define the relationships
axiom ssr_r_squared_inverse (m : Model) : m.ssr < 0 → m.r_squared > 0
axiom r_squared_fit_quality_direct (m : Model) : m.r_squared > 0 → m.fit_quality > 0

-- Theorem statement
theorem model_fit_relationships (m1 m2 : Model) :
  (m1.ssr < m2.ssr → m1.r_squared > m2.r_squared ∧ m1.fit_quality > m2.fit_quality) ∧
  (m1.ssr > m2.ssr → m1.r_squared < m2.r_squared ∧ m1.fit_quality < m2.fit_quality) :=
sorry

end model_fit_relationships_l2799_279938


namespace largest_three_digit_multiple_of_9_with_sum_27_l2799_279958

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ sum_of_digits n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ sum_of_digits m = 27 → m ≤ n :=
by sorry

end largest_three_digit_multiple_of_9_with_sum_27_l2799_279958


namespace ellipse_dot_product_bounds_l2799_279921

/-- The ellipse with equation x²/9 + y²/8 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/9 + y^2/8 = 1

/-- The left focus of the ellipse -/
def F1 : ℝ × ℝ := (-1, 0)

/-- The right focus of the ellipse -/
def F2 : ℝ × ℝ := (1, 0)

/-- The dot product of vectors EF₁ and EF₂ -/
def dotProduct (E : ℝ × ℝ) : ℝ :=
  let (x, y) := E
  (-1-x)*(1-x) + (-y)*(-y)

theorem ellipse_dot_product_bounds :
  ∀ E : ℝ × ℝ, Ellipse E.1 E.2 → 7 ≤ dotProduct E ∧ dotProduct E ≤ 8 :=
sorry

end ellipse_dot_product_bounds_l2799_279921


namespace xiao_ming_exam_probabilities_l2799_279916

/-- Represents the probabilities of scoring in different ranges in a math exam -/
structure ExamProbabilities where
  above90 : ℝ
  between80and89 : ℝ
  between70and79 : ℝ
  between60and69 : ℝ

/-- Calculates the probability of scoring above 80 -/
def probAbove80 (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89

/-- Calculates the probability of passing the exam (scoring above 60) -/
def probPassing (p : ExamProbabilities) : ℝ :=
  p.above90 + p.between80and89 + p.between70and79 + p.between60and69

/-- Theorem stating the probabilities for Xiao Ming's math exam -/
theorem xiao_ming_exam_probabilities (p : ExamProbabilities)
    (h1 : p.above90 = 0.18)
    (h2 : p.between80and89 = 0.51)
    (h3 : p.between70and79 = 0.15)
    (h4 : p.between60and69 = 0.09) :
    probAbove80 p = 0.69 ∧ probPassing p = 0.93 := by
  sorry


end xiao_ming_exam_probabilities_l2799_279916


namespace floor_length_is_twelve_l2799_279910

/-- Represents a rectangular floor with a rug -/
structure FloorWithRug where
  length : ℝ
  width : ℝ
  strip_width : ℝ
  rug_area : ℝ

/-- Theorem: Given the conditions, the floor length is 12 meters -/
theorem floor_length_is_twelve (floor : FloorWithRug) 
  (h1 : floor.width = 10)
  (h2 : floor.strip_width = 3)
  (h3 : floor.rug_area = 24)
  (h4 : floor.rug_area = (floor.length - 2 * floor.strip_width) * (floor.width - 2 * floor.strip_width)) :
  floor.length = 12 := by
  sorry

end floor_length_is_twelve_l2799_279910


namespace subset_sum_property_l2799_279955

theorem subset_sum_property (n : ℕ) (A B C : Finset ℕ) :
  (∀ i ∈ A ∪ B ∪ C, i ≤ 3*n) →
  A.card = n →
  B.card = n →
  C.card = n →
  (A ∩ B ∩ C).card = 0 →
  (A ∪ B ∪ C).card = 3*n →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ a + b = c :=
by sorry

end subset_sum_property_l2799_279955


namespace parallel_sufficient_not_necessary_l2799_279950

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The condition that x = 2 is sufficient but not necessary for parallelism -/
theorem parallel_sufficient_not_necessary (x : ℝ) :
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (4, x)
  (x = 2 → are_parallel a b) ∧
  ¬(are_parallel a b → x = 2) := by
  sorry

end parallel_sufficient_not_necessary_l2799_279950


namespace queens_bounding_rectangle_l2799_279933

theorem queens_bounding_rectangle (a : Fin 2004 → Fin 2004) 
  (h_perm : Function.Bijective a) 
  (h_diag : ∀ i j : Fin 2004, i ≠ j → |a i - a j| ≠ |i - j|) :
  ∃ i j : Fin 2004, |i - j| + |a i - a j| = 2004 := by
  sorry

end queens_bounding_rectangle_l2799_279933


namespace perpendicular_vector_proof_l2799_279976

def line_direction : ℝ × ℝ := (3, 2)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vector_proof (v : ℝ × ℝ) :
  is_perpendicular v line_direction ∧ v.1 + v.2 = 1 → v = (-2, 3) := by
  sorry

end perpendicular_vector_proof_l2799_279976


namespace yoongi_initial_money_l2799_279999

/-- The amount of money Yoongi had initially -/
def initial_money : ℕ := 590

/-- The cost of the candy Yoongi bought -/
def candy_cost : ℕ := 250

/-- The amount of pocket money Yoongi received -/
def pocket_money : ℕ := 500

/-- The amount of money Yoongi had left after all transactions -/
def money_left : ℕ := 420

theorem yoongi_initial_money :
  ∃ (pencil_cost : ℕ),
    initial_money = candy_cost + pencil_cost + money_left ∧
    initial_money + pocket_money - candy_cost = 2 * money_left :=
by
  sorry


end yoongi_initial_money_l2799_279999


namespace harmonious_expressions_l2799_279961

-- Define the concept of a harmonious algebraic expression
def is_harmonious (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f x ∈ Set.Icc a b ∧
  ∃ y ∈ Set.Icc a b, ∀ z ∈ Set.Icc a b, f y ≥ f z

-- Theorem statement
theorem harmonious_expressions :
  let a := -2
  let b := 2
  -- Part 1
  ¬ is_harmonious (fun x => |x - 1|) a b ∧
  -- Part 2
  ¬ is_harmonious (fun x => -x + 1) a b ∧
  is_harmonious (fun x => -x^2 + 2) a b ∧
  ¬ is_harmonious (fun x => x^2 + |x| - 4) a b ∧
  -- Part 3
  ∀ c : ℝ, is_harmonious (fun x => c / (|x| + 1) - 2) a b ↔ (0 ≤ c ∧ c ≤ 4) :=
by
  sorry


end harmonious_expressions_l2799_279961


namespace matrix_power_sum_l2799_279906

/-- Given a matrix B and its mth power, prove that b + m = 381 -/
theorem matrix_power_sum (b m : ℕ) : 
  let B : Matrix (Fin 3) (Fin 3) ℕ := !![1, 3, b; 0, 1, 5; 0, 0, 1]
  let B_pow_m : Matrix (Fin 3) (Fin 3) ℕ := !![1, 33, 4054; 0, 1, 55; 0, 0, 1]
  B^m = B_pow_m → b + m = 381 := by
sorry

end matrix_power_sum_l2799_279906


namespace well_depth_is_30_l2799_279942

/-- The depth of a well that a man climbs out of in 27 days -/
def well_depth (daily_climb : ℕ) (daily_slip : ℕ) (total_days : ℕ) (final_climb : ℕ) : ℕ :=
  (total_days - 1) * (daily_climb - daily_slip) + final_climb

/-- Theorem stating the depth of the well is 30 meters -/
theorem well_depth_is_30 :
  well_depth 4 3 27 4 = 30 := by
  sorry

end well_depth_is_30_l2799_279942


namespace carpet_area_calculation_l2799_279981

/-- The carpet area required for a rectangular room -/
def carpet_area (length width : ℝ) (wastage_factor : ℝ) : ℝ :=
  length * width * (1 + wastage_factor)

/-- Theorem: The carpet area for a 15 ft by 9 ft room with 10% wastage is 148.5 sq ft -/
theorem carpet_area_calculation :
  carpet_area 15 9 0.1 = 148.5 := by
  sorry

end carpet_area_calculation_l2799_279981


namespace x_positive_sufficient_not_necessary_for_abs_x_positive_l2799_279925

theorem x_positive_sufficient_not_necessary_for_abs_x_positive :
  (∀ x : ℝ, x > 0 → |x| > 0) ∧
  (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) :=
by sorry

end x_positive_sufficient_not_necessary_for_abs_x_positive_l2799_279925


namespace correct_answers_l2799_279904

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  totalQuestions : ℕ
  correctScore : ℤ
  wrongScore : ℤ

/-- Represents a student's exam attempt. -/
structure ExamAttempt where
  exam : Exam
  totalScore : ℤ
  attemptedAll : Bool

/-- Theorem stating the number of correctly answered questions given the exam conditions. -/
theorem correct_answers (e : Exam) (a : ExamAttempt) 
    (h1 : e.totalQuestions = 60)
    (h2 : e.correctScore = 4)
    (h3 : e.wrongScore = -1)
    (h4 : a.exam = e)
    (h5 : a.totalScore = 150)
    (h6 : a.attemptedAll = true) :
    ∃ (c : ℕ), c = 42 ∧ 
    c * e.correctScore + (e.totalQuestions - c) * e.wrongScore = a.totalScore :=
  sorry

end correct_answers_l2799_279904


namespace solve_linear_equation_l2799_279931

theorem solve_linear_equation (x : ℤ) : 9823 + x = 13200 → x = 3377 := by
  sorry

end solve_linear_equation_l2799_279931


namespace complex_multiply_i_l2799_279926

theorem complex_multiply_i (i : ℂ) : i * i = -1 → (1 + i) * i = -1 + i := by
  sorry

end complex_multiply_i_l2799_279926


namespace distinct_triangles_in_regular_ngon_l2799_279977

theorem distinct_triangles_in_regular_ngon (n : ℕ) :
  Nat.choose n 3 = (n * (n - 1) * (n - 2)) / 6 :=
sorry

end distinct_triangles_in_regular_ngon_l2799_279977


namespace clothing_tax_rate_l2799_279962

theorem clothing_tax_rate 
  (total : ℝ) 
  (clothing_spend : ℝ) 
  (food_spend : ℝ) 
  (other_spend : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  clothing_spend = 0.6 * total →
  food_spend = 0.1 * total →
  other_spend = 0.3 * total →
  other_tax_rate = 0.08 →
  total_tax_rate = 0.048 →
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_spend + other_tax_rate * other_spend = total_tax_rate * total ∧
    clothing_tax_rate = 0.04 :=
by sorry

end clothing_tax_rate_l2799_279962


namespace ab_positive_necessary_not_sufficient_l2799_279909

-- Define the condition for an ellipse
def is_ellipse (a b : ℝ) : Prop := ∃ (x y : ℝ), a * x^2 + b * y^2 = 1 ∧ a > 0 ∧ b > 0

-- Theorem stating that ab > 0 is necessary but not sufficient for an ellipse
theorem ab_positive_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse a b → a * b > 0) ∧
  ¬(∀ a b : ℝ, a * b > 0 → is_ellipse a b) :=
sorry

end ab_positive_necessary_not_sufficient_l2799_279909


namespace perfect_square_function_characterization_l2799_279914

theorem perfect_square_function_characterization (g : ℕ → ℕ) : 
  (∀ n m : ℕ, ∃ k : ℕ, (g n + m) * (g m + n) = k^2) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c := by
  sorry

end perfect_square_function_characterization_l2799_279914


namespace quadratic_trinomial_factorization_l2799_279997

theorem quadratic_trinomial_factorization 
  (a b c x x₁ x₂ : ℝ) 
  (ha : a ≠ 0) 
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) 
  (hx₂ : a * x₂^2 + b * x₂ + c = 0) : 
  a * x^2 + b * x + c = a * (x - x₁) * (x - x₂) := by
sorry

end quadratic_trinomial_factorization_l2799_279997


namespace monomial_coefficient_degree_product_l2799_279902

/-- 
Given a monomial of the form $-\frac{3}{4}{x^2}{y^2}$, 
this theorem proves that the product of its coefficient and degree is -3.
-/
theorem monomial_coefficient_degree_product : 
  ∃ (m n : ℚ), (m = -3/4) ∧ (n = 4) ∧ (m * n = -3) := by
  sorry

end monomial_coefficient_degree_product_l2799_279902


namespace kevin_cards_l2799_279956

/-- The number of cards Kevin has at the end of the day -/
def final_cards (initial : ℕ) (found : ℕ) (lost1 : ℕ) (lost2 : ℕ) (won : ℕ) : ℕ :=
  initial + found - lost1 - lost2 + won

/-- Theorem stating that Kevin ends up with 63 cards given the problem conditions -/
theorem kevin_cards : final_cards 20 47 7 12 15 = 63 := by
  sorry

end kevin_cards_l2799_279956


namespace gerald_furniture_problem_l2799_279978

/-- Represents the problem of determining the maximum number of chairs Gerald can make --/
theorem gerald_furniture_problem 
  (x t c b : ℕ) 
  (r1 r2 r3 : ℕ) 
  (h_x : x = 2250)
  (h_t : t = 18)
  (h_c : c = 12)
  (h_b : b = 30)
  (h_ratio : r1 = 2 ∧ r2 = 3 ∧ r3 = 1) :
  ∃ (chairs : ℕ), 
    chairs ≤ (x / (t * r1 / r2 + c + b * r3 / r2)) ∧ 
    chairs = 66 := by
  sorry


end gerald_furniture_problem_l2799_279978


namespace absolute_sum_nonzero_iff_either_nonzero_l2799_279908

theorem absolute_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  |x| + |y| ≠ 0 ↔ x ≠ 0 ∨ y ≠ 0 := by sorry

end absolute_sum_nonzero_iff_either_nonzero_l2799_279908


namespace parabola_translation_l2799_279903

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def horizontal_shift (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x - h) }

/-- Shifts a parabola vertically -/
def vertical_shift (p : Parabola) (v : ℝ) : Parabola :=
  { f := fun x => p.f x + v }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := fun x => x^2 }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  vertical_shift (horizontal_shift original_parabola 2) (-3)

theorem parabola_translation :
  resulting_parabola.f = fun x => (x + 2)^2 - 3 := by sorry

end parabola_translation_l2799_279903


namespace parabola_translation_l2799_279900

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * dx + p.b
    c := p.a * dx^2 - p.b * dx + p.c + dy }

/-- The original parabola y = x² -/
def original : Parabola :=
  { a := 1, b := 0, c := 0 }

/-- The resulting parabola after translation -/
def translated : Parabola :=
  translate original 2 1

theorem parabola_translation :
  translated = { a := 1, b := -4, c := 5 } :=
sorry

end parabola_translation_l2799_279900
