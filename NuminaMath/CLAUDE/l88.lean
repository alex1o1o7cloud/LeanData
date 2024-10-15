import Mathlib

namespace NUMINAMATH_CALUDE_mercedes_jonathan_ratio_l88_8828

def jonathan_distance : ℝ := 7.5

def mercedes_davonte_total : ℝ := 32

theorem mercedes_jonathan_ratio : 
  ∃ (mercedes_distance : ℝ),
    mercedes_distance + (mercedes_distance + 2) = mercedes_davonte_total ∧
    mercedes_distance / jonathan_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_mercedes_jonathan_ratio_l88_8828


namespace NUMINAMATH_CALUDE_smallest_n_complex_equation_l88_8816

theorem smallest_n_complex_equation (n : ℕ) (a b : ℝ) : 
  n > 3 ∧ 
  0 < a ∧ 
  0 < b ∧ 
  (∀ k : ℕ, 3 < k ∧ k < n → ¬∃ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y * I) ^ k + x = (x - y * I) ^ k + y) ∧
  (a + b * I) ^ n + a = (a - b * I) ^ n + b →
  b / a = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_n_complex_equation_l88_8816


namespace NUMINAMATH_CALUDE_insufficient_info_for_both_correct_evans_class_test_l88_8812

theorem insufficient_info_for_both_correct (total_students : ℕ) 
  (q1_correct : ℕ) (absent : ℕ) (q2_correct : ℕ) : Prop :=
  total_students = 40 ∧ 
  q1_correct = 30 ∧ 
  absent = 10 ∧
  q2_correct ≥ 0 ∧ q2_correct ≤ (total_students - absent) →
  ∃ (both_correct₁ both_correct₂ : ℕ), 
    both_correct₁ ≠ both_correct₂ ∧
    both_correct₁ ≥ 0 ∧ both_correct₁ ≤ q1_correct ∧
    both_correct₂ ≥ 0 ∧ both_correct₂ ≤ q1_correct ∧
    both_correct₁ ≤ q2_correct ∧ both_correct₂ ≤ q2_correct

theorem evans_class_test : insufficient_info_for_both_correct 40 30 10 q2_correct :=
sorry

end NUMINAMATH_CALUDE_insufficient_info_for_both_correct_evans_class_test_l88_8812


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l88_8854

/-- A line y = 3x + d is tangent to the parabola y^2 = 12x if and only if d = 1 -/
theorem line_tangent_to_parabola (d : ℝ) :
  (∃ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧
    ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≤ 12 * x') ↔ d = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l88_8854


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_160_420_l88_8802

theorem lcm_gcf_ratio_160_420 : 
  (Nat.lcm 160 420) / (Nat.gcd 160 420 - 2) = 187 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_160_420_l88_8802


namespace NUMINAMATH_CALUDE_complex_equation_solution_l88_8875

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 3 - 4 * Complex.I → z = -4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l88_8875


namespace NUMINAMATH_CALUDE_nail_trimming_customers_l88_8884

/-- The number of customers who had their nails trimmed -/
def number_of_customers (total_sounds : ℕ) (nails_per_person : ℕ) : ℕ :=
  total_sounds / nails_per_person

/-- Theorem: Given 60 nail trimming sounds and 20 nails per person, 
    the number of customers who had their nails trimmed is 3 -/
theorem nail_trimming_customers :
  number_of_customers 60 20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l88_8884


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l88_8837

/-- In a Cartesian coordinate system, the coordinates of a point (-1, 2) with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (x y : ℝ) : x = -1 ∧ y = 2 → (x, y) = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l88_8837


namespace NUMINAMATH_CALUDE_f_sum_value_l88_8819

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_value : f Real.pi + (deriv f) (Real.pi / 2) = -3 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_f_sum_value_l88_8819


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l88_8877

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^4

-- State the theorem
theorem f_derivative_at_zero : 
  (deriv f) 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l88_8877


namespace NUMINAMATH_CALUDE_icosahedron_edge_probability_l88_8885

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Nat)
  (edges_per_vertex : Nat)
  (h_vertices : vertices = 12)
  (h_edges_per_vertex : edges_per_vertex = 5)

/-- The probability of selecting two vertices that form an edge in an icosahedron -/
def edge_probability (i : Icosahedron) : ℚ :=
  5 / 11

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular icosahedron is 5/11 -/
theorem icosahedron_edge_probability (i : Icosahedron) : 
  edge_probability i = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_icosahedron_edge_probability_l88_8885


namespace NUMINAMATH_CALUDE_app_cost_is_four_l88_8887

/-- The average cost of an app given the total budget, remaining amount, and number of apps. -/
def average_app_cost (total_budget : ℚ) (remaining : ℚ) (num_apps : ℕ) : ℚ :=
  (total_budget - remaining) / num_apps

/-- Theorem stating that the average cost of an app is $4 given the problem conditions. -/
theorem app_cost_is_four :
  let total_budget : ℚ := 66
  let remaining : ℚ := 6
  let num_apps : ℕ := 15
  average_app_cost total_budget remaining num_apps = 4 := by
  sorry

end NUMINAMATH_CALUDE_app_cost_is_four_l88_8887


namespace NUMINAMATH_CALUDE_twenty_nine_free_travelers_l88_8896

/-- Represents the promotion scenario for a travel agency -/
structure TravelPromotion where
  /-- Number of tourists who came on their own -/
  self_arrivals : ℕ
  /-- Number of tourists who didn't bring anyone -/
  no_referrals : ℕ
  /-- Total number of tourists -/
  total_tourists : ℕ

/-- Calculates the number of tourists who traveled for free -/
def free_travelers (promo : TravelPromotion) : ℕ :=
  (promo.total_tourists - promo.self_arrivals - promo.no_referrals) / 4

/-- Theorem stating that 29 tourists traveled for free -/
theorem twenty_nine_free_travelers (promo : TravelPromotion)
  (h1 : promo.self_arrivals = 13)
  (h2 : promo.no_referrals = 100)
  (h3 : promo.total_tourists = promo.self_arrivals + promo.no_referrals + 4 * (free_travelers promo)) :
  free_travelers promo = 29 := by
  sorry

#eval free_travelers { self_arrivals := 13, no_referrals := 100, total_tourists := 229 }

end NUMINAMATH_CALUDE_twenty_nine_free_travelers_l88_8896


namespace NUMINAMATH_CALUDE_inequality_solution_set_l88_8850

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 1| > a) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l88_8850


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l88_8821

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 16 = (a*x + b)^2) →
  m = 5 ∨ m = -3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l88_8821


namespace NUMINAMATH_CALUDE_yearly_fluid_intake_l88_8889

def weekday_soda : ℕ := 5 * 12
def weekday_water : ℕ := 64
def weekday_juice : ℕ := 3 * 8
def weekday_sports : ℕ := 2 * 16

def weekend_soda : ℕ := 5 * 12
def weekend_water : ℕ := 64
def weekend_juice : ℕ := 3 * 8
def weekend_sports : ℕ := 1 * 16
def weekend_smoothie : ℕ := 32

def weekdays : ℕ := 260
def weekend_days : ℕ := 104
def holidays : ℕ := 1

def weekday_total : ℕ := weekday_soda + weekday_water + weekday_juice + weekday_sports
def weekend_total : ℕ := weekend_soda + weekend_water + weekend_juice + weekend_sports + weekend_smoothie

theorem yearly_fluid_intake :
  weekday_total * weekdays + weekend_total * (weekend_days + holidays) = 67380 := by
  sorry

end NUMINAMATH_CALUDE_yearly_fluid_intake_l88_8889


namespace NUMINAMATH_CALUDE_max_intersection_points_for_arrangement_l88_8862

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the arrangement of two convex polygons in a plane -/
structure PolygonArrangement where
  A₁ : ConvexPolygon
  A₂ : ConvexPolygon
  same_plane : Bool
  can_intersect : Bool
  no_full_overlap : Bool

/-- Calculates the maximum number of intersection points between two polygons -/
def max_intersection_points (arr : PolygonArrangement) : ℕ :=
  arr.A₁.sides * arr.A₂.sides

/-- Theorem stating the maximum number of intersection points for the given arrangement -/
theorem max_intersection_points_for_arrangement 
  (m : ℕ) 
  (arr : PolygonArrangement) 
  (h1 : arr.A₁.sides = m) 
  (h2 : arr.A₂.sides = m + 2) 
  (h3 : arr.same_plane) 
  (h4 : arr.can_intersect) 
  (h5 : arr.no_full_overlap) 
  (h6 : arr.A₁.convex) 
  (h7 : arr.A₂.convex) : 
  max_intersection_points arr = m^2 + 2*m := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_for_arrangement_l88_8862


namespace NUMINAMATH_CALUDE_sector_area_l88_8805

/-- Given a circular sector with circumference 6cm and central angle 1 radian, 
    prove that its area is 2cm². -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) : 
  circumference = 6 → central_angle = 1 → area = 2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l88_8805


namespace NUMINAMATH_CALUDE_cone_volume_with_inscribed_sphere_l88_8846

/-- The volume of a cone with an inscribed sphere -/
theorem cone_volume_with_inscribed_sphere (r α : ℝ) (hr : r > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = -π * r^3 * Real.tan (2 * α) / (24 * Real.cos α ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_with_inscribed_sphere_l88_8846


namespace NUMINAMATH_CALUDE_maze_paths_count_l88_8825

/-- Represents a junction in the maze --/
structure Junction where
  choices : Nat  -- Number of possible directions at this junction

/-- Represents the maze structure --/
structure Maze where
  entrance_choices : Nat  -- Number of choices at the entrance
  x_junctions : Nat       -- Number of x junctions
  dot_junctions : Nat     -- Number of dot junctions per x junction

/-- Calculates the number of paths through the maze --/
def count_paths (m : Maze) : Nat :=
  m.entrance_choices * m.x_junctions * (2 ^ m.dot_junctions)

/-- Theorem stating that the number of paths in the given maze is 16 --/
theorem maze_paths_count :
  ∃ (m : Maze), count_paths m = 16 :=
sorry

end NUMINAMATH_CALUDE_maze_paths_count_l88_8825


namespace NUMINAMATH_CALUDE_regression_slope_l88_8860

-- Define the linear function
def f (x : ℝ) : ℝ := 2 - 3 * x

-- Theorem statement
theorem regression_slope (x : ℝ) :
  f (x + 1) = f x - 3 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_l88_8860


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l88_8830

theorem cylinder_surface_area (r h : ℝ) (base_area : ℝ) : 
  base_area = π * r^2 →
  h = 2 * r →
  2 * base_area + 2 * π * r * h = 384 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l88_8830


namespace NUMINAMATH_CALUDE_sweets_ratio_l88_8820

/-- Proves that the ratio of sweets received by the youngest child to the eldest child is 1:2 --/
theorem sweets_ratio (total : ℕ) (eldest : ℕ) (second : ℕ) : 
  total = 27 →
  eldest = 8 →
  second = 6 →
  (total - (total / 3) - eldest - second) * 2 = eldest := by
  sorry

end NUMINAMATH_CALUDE_sweets_ratio_l88_8820


namespace NUMINAMATH_CALUDE_recurrence_closed_form_l88_8849

def recurrence_sequence (a : ℕ → ℝ) : Prop :=
  (a 0 = 3) ∧ (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2))

theorem recurrence_closed_form (a : ℕ → ℝ) (h : recurrence_sequence a) :
  ∀ n : ℕ, a n = 3^n + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_recurrence_closed_form_l88_8849


namespace NUMINAMATH_CALUDE_expression_simplification_l88_8872

theorem expression_simplification (x y : ℝ) : 
  3 * x + 4 * y^2 + 2 - (5 - 3 * x - 2 * y^2) = 6 * x + 6 * y^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l88_8872


namespace NUMINAMATH_CALUDE_line_point_k_value_l88_8851

/-- A line contains the points (2, 4), (7, k), and (15, 8). The value of k is 72/13. -/
theorem line_point_k_value : ∀ (k : ℚ), 
  (∃ (m b : ℚ), 
    (4 : ℚ) = m * 2 + b ∧ 
    k = m * 7 + b ∧ 
    (8 : ℚ) = m * 15 + b) → 
  k = 72 / 13 := by
sorry

end NUMINAMATH_CALUDE_line_point_k_value_l88_8851


namespace NUMINAMATH_CALUDE_range_of_m_l88_8861

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 5}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l88_8861


namespace NUMINAMATH_CALUDE_point_relation_l88_8822

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -Real.sqrt 2 * x + b

-- Define the theorem
theorem point_relation (m n b : ℝ) 
  (h1 : line_equation (-2) m b)
  (h2 : line_equation 3 n b) : 
  m > n := by sorry

end NUMINAMATH_CALUDE_point_relation_l88_8822


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l88_8873

/-- A parallelogram with side lengths 5, 10y-2, 3x+5, and 12 has x+y equal to 91/30 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3 * x + 5 = 12) → (10 * y - 2 = 5) → x + y = 91 / 30 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l88_8873


namespace NUMINAMATH_CALUDE_percent_commutation_l88_8804

theorem percent_commutation (x : ℝ) (h : 0.3 * 0.4 * x = 36) :
  0.4 * 0.3 * x = 0.3 * 0.4 * x :=
by
  sorry

end NUMINAMATH_CALUDE_percent_commutation_l88_8804


namespace NUMINAMATH_CALUDE_richard_needs_three_touchdowns_per_game_l88_8892

/-- Represents a football player's touchdown record --/
structure TouchdownRecord where
  player : String
  touchdowns : ℕ
  games : ℕ

/-- Calculates the number of touchdowns needed to beat a record --/
def touchdownsNeededToBeat (record : TouchdownRecord) : ℕ :=
  record.touchdowns + 1

/-- Theorem: Richard needs to average 3 touchdowns per game in the final two games to beat Archie's record --/
theorem richard_needs_three_touchdowns_per_game
  (archie : TouchdownRecord)
  (richard_current_touchdowns : ℕ)
  (richard_current_games : ℕ)
  (total_games : ℕ)
  (h1 : archie.player = "Archie")
  (h2 : archie.touchdowns = 89)
  (h3 : archie.games = 16)
  (h4 : richard_current_touchdowns = 6 * richard_current_games)
  (h5 : richard_current_games = 14)
  (h6 : total_games = 16) :
  (touchdownsNeededToBeat archie - richard_current_touchdowns) / (total_games - richard_current_games) = 3 :=
sorry

end NUMINAMATH_CALUDE_richard_needs_three_touchdowns_per_game_l88_8892


namespace NUMINAMATH_CALUDE_parabola_chord_midpoint_l88_8859

/-- Given a parabola y^2 = 2px and a chord with midpoint (3, 1) and slope 2, prove that p = 2 -/
theorem parabola_chord_midpoint (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- Equation of the parabola
  (∃ x₁ y₁ x₂ y₂ : ℝ,        -- Existence of two points on the chord
    y₁^2 = 2*p*x₁ ∧          -- First point satisfies parabola equation
    y₂^2 = 2*p*x₂ ∧          -- Second point satisfies parabola equation
    (x₁ + x₂)/2 = 3 ∧        -- x-coordinate of midpoint is 3
    (y₁ + y₂)/2 = 1 ∧        -- y-coordinate of midpoint is 1
    (y₂ - y₁)/(x₂ - x₁) = 2  -- Slope of the chord is 2
  ) →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_chord_midpoint_l88_8859


namespace NUMINAMATH_CALUDE_minimum_points_to_win_l88_8817

/-- Represents the points earned in a single race -/
inductive RaceResult
| First  : RaceResult
| Second : RaceResult
| Third  : RaceResult
| Other  : RaceResult

/-- Converts a race result to points -/
def pointsForResult (result : RaceResult) : Nat :=
  match result with
  | RaceResult.First  => 4
  | RaceResult.Second => 2
  | RaceResult.Third  => 1
  | RaceResult.Other  => 0

/-- Calculates total points for a series of race results -/
def totalPoints (results : List RaceResult) : Nat :=
  results.map pointsForResult |>.sum

/-- Represents all possible combinations of race results for four races -/
def allPossibleResults : List (List RaceResult) :=
  sorry

theorem minimum_points_to_win (results : List RaceResult) :
  (results.length = 4) →
  (totalPoints results ≥ 15) →
  (∀ other : List RaceResult, other.length = 4 → totalPoints other < totalPoints results) :=
sorry

end NUMINAMATH_CALUDE_minimum_points_to_win_l88_8817


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l88_8811

theorem hyperbola_eccentricity_range (a b : ℝ) (M : ℝ × ℝ) (F P Q : ℝ × ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (x, y) := M
  (x^2 / a^2 - y^2 / b^2 = 1) →  -- M is on the hyperbola
  (F.1 = a * (a^2 + b^2).sqrt / (a^2 + b^2).sqrt ∧ F.2 = 0) →  -- F is a focus on x-axis
  (∃ r : ℝ, (M.1 - F.1)^2 + M.2^2 = r^2 ∧ P.1 = 0 ∧ Q.1 = 0 ∧ (P.2 - M.2)^2 + M.1^2 = r^2 ∧ (Q.2 - M.2)^2 + M.1^2 = r^2) →  -- Circle condition
  (0 < Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) ∧ 
   Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) < π/2) →  -- Acute triangle condition
  let e := ((a^2 + b^2) / a^2).sqrt
  (Real.sqrt 5 + 1) / 2 < e ∧ e < (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l88_8811


namespace NUMINAMATH_CALUDE_age_problem_l88_8831

theorem age_problem (my_age : ℕ) : 
  (∃ (older_brother younger_sister youngest_brother : ℕ),
    -- Ten years ago, my older brother was exactly twice my age
    older_brother = 2 * (my_age - 10) ∧
    -- Ten years ago, my younger sister's age was half of mine
    younger_sister = (my_age - 10) / 2 ∧
    -- Ten years ago, my youngest brother was the same age as my sister
    youngest_brother = younger_sister ∧
    -- In fifteen years, the combined age of the four of us will be 110
    (my_age + 15) + (older_brother + 15) + (younger_sister + 15) + (youngest_brother + 15) = 110) →
  my_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l88_8831


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l88_8838

/-- Given a total profit and a ratio of profit division between two parties,
    calculate the difference between their profit shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 800 and a profit division ratio of 1/2 : 1/3,
    the difference between the profit shares is 160. -/
theorem profit_share_difference_example :
  profit_share_difference 800 (1/2) (1/3) = 160 := by
  sorry


end NUMINAMATH_CALUDE_profit_share_difference_example_l88_8838


namespace NUMINAMATH_CALUDE_probability_in_specific_sequence_l88_8883

/-- Represents an arithmetic sequence with given parameters -/
structure ArithmeticSequence where
  first_term : ℕ
  common_difference : ℕ
  last_term : ℕ

/-- Calculates the number of terms in the arithmetic sequence -/
def number_of_terms (seq : ArithmeticSequence) : ℕ :=
  (seq.last_term - seq.first_term) / seq.common_difference + 1

/-- Calculates the number of terms divisible by 6 in the sequence -/
def divisible_by_six (seq : ArithmeticSequence) : ℕ :=
  (number_of_terms seq) / 3

/-- The probability of selecting a number divisible by 6 from the sequence -/
def probability_divisible_by_six (seq : ArithmeticSequence) : ℚ :=
  (divisible_by_six seq : ℚ) / (number_of_terms seq)

theorem probability_in_specific_sequence :
  let seq := ArithmeticSequence.mk 50 4 998
  probability_divisible_by_six seq = 79 / 238 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_specific_sequence_l88_8883


namespace NUMINAMATH_CALUDE_product_of_fractions_l88_8810

theorem product_of_fractions : 
  (4 : ℚ) / 5 * 9 / 6 * 12 / 4 * 20 / 15 * 14 / 21 * 35 / 28 * 48 / 32 * 24 / 16 = 54 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l88_8810


namespace NUMINAMATH_CALUDE_consecutive_even_odd_squares_divisibility_l88_8800

theorem consecutive_even_odd_squares_divisibility :
  (∀ n : ℕ+, ∃ k : ℕ, (2*n+2)^2 - (2*n)^2 = 4*k) ∧
  (∀ m : ℕ+, ∃ k : ℕ, (2*m+1)^2 - (2*m-1)^2 = 8*k) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_odd_squares_divisibility_l88_8800


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l88_8844

/-- Given a vector a = (2, 1) and another vector b such that a · b = 10 and |a + b| = 5, prove that |b| = 2√10. -/
theorem vector_magnitude_proof (b : ℝ × ℝ) :
  let a : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 10) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 25) →
  (b.1^2 + b.2^2 = 40) := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l88_8844


namespace NUMINAMATH_CALUDE_remainder_sum_l88_8803

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 80 = 75)
  (hb : b % 120 = 115)
  (hc : c % 160 = 155) : 
  (a + b + c) % 40 = 25 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l88_8803


namespace NUMINAMATH_CALUDE_trees_planted_by_fourth_grade_l88_8890

theorem trees_planted_by_fourth_grade :
  ∀ (fifth_grade third_grade fourth_grade : ℕ),
    fifth_grade = 114 →
    fifth_grade = 2 * third_grade →
    fourth_grade = third_grade + 32 →
    fourth_grade = 89 := by sorry

end NUMINAMATH_CALUDE_trees_planted_by_fourth_grade_l88_8890


namespace NUMINAMATH_CALUDE_min_max_inequality_l88_8853

theorem min_max_inequality (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b) 
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b) 
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧ 
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ a/b + b/a - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l88_8853


namespace NUMINAMATH_CALUDE_brownie_division_l88_8865

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a brownie tray with its dimensions -/
def tray : Dimensions := ⟨24, 30⟩

/-- Represents a single brownie piece with its dimensions -/
def piece : Dimensions := ⟨3, 4⟩

/-- Theorem stating that the tray can be divided into exactly 60 brownie pieces -/
theorem brownie_division :
  (area tray) / (area piece) = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_division_l88_8865


namespace NUMINAMATH_CALUDE_number_percentage_equality_l88_8893

theorem number_percentage_equality (x : ℝ) : 
  (25 / 100) * x = (20 / 100) * 30 → x = 24 := by
sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l88_8893


namespace NUMINAMATH_CALUDE_plate_on_square_table_l88_8880

/-- Given a square table with a round plate, if the distances from the plate's edge
    to two adjacent sides of the table are a and b, and the distance from the plate's edge
    to the opposite side of the a measurement is c, then the distance from the plate's edge
    to the opposite side of the b measurement is a + c - b. -/
theorem plate_on_square_table (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + c = b + (a + c - b) :=
by sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l88_8880


namespace NUMINAMATH_CALUDE_annual_increase_fraction_l88_8879

theorem annual_increase_fraction (initial_amount final_amount : ℝ) (f : ℝ) 
  (h1 : initial_amount = 65000)
  (h2 : final_amount = 82265.625)
  (h3 : final_amount = initial_amount * (1 + f)^2) :
  f = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_annual_increase_fraction_l88_8879


namespace NUMINAMATH_CALUDE_thomas_savings_l88_8806

/-- Thomas's savings scenario --/
theorem thomas_savings (
  weekly_allowance : ℝ)
  (weeks_per_year : ℕ)
  (hours_per_week : ℕ)
  (car_cost : ℝ)
  (weekly_spending : ℝ)
  (additional_savings_needed : ℝ)
  (hourly_wage : ℝ)
  (h1 : weekly_allowance = 50)
  (h2 : weeks_per_year = 52)
  (h3 : hours_per_week = 30)
  (h4 : car_cost = 15000)
  (h5 : weekly_spending = 35)
  (h6 : additional_savings_needed = 2000)
  : hourly_wage = 7.83 := by
  sorry

end NUMINAMATH_CALUDE_thomas_savings_l88_8806


namespace NUMINAMATH_CALUDE_profit_rate_equal_with_without_discount_l88_8813

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the profit rate with discount
def profit_rate_with_discount : ℝ := 0.235

-- Theorem statement
theorem profit_rate_equal_with_without_discount :
  profit_rate_with_discount = (1 + profit_rate_with_discount) / (1 - discount_rate) - 1 :=
by sorry

end NUMINAMATH_CALUDE_profit_rate_equal_with_without_discount_l88_8813


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_4_l88_8809

theorem greatest_four_digit_divisible_by_3_and_4 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_3_and_4_l88_8809


namespace NUMINAMATH_CALUDE_abc_greater_than_28_l88_8891

-- Define the polynomials P and Q
def P (a b c x : ℝ) : ℝ := a * x^3 + (b - a) * x^2 - (c + b) * x + c
def Q (a b c x : ℝ) : ℝ := x^4 + (b - 1) * x^3 + (a - b) * x^2 - (c + a) * x + c

-- State the theorem
theorem abc_greater_than_28 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hb_pos : b > 0)
  (hP_roots : ∃ x₀ x₁ x₂ : ℝ, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ 
    P a b c x₀ = 0 ∧ P a b c x₁ = 0 ∧ P a b c x₂ = 0)
  (hQ_roots : ∃ x₀ x₁ x₂ : ℝ, 
    Q a b c x₀ = 0 ∧ Q a b c x₁ = 0 ∧ Q a b c x₂ = 0) :
  a * b * c > 28 :=
sorry

end NUMINAMATH_CALUDE_abc_greater_than_28_l88_8891


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l88_8836

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l88_8836


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l88_8834

theorem quadratic_roots_nature (a b c : ℝ) (h : a ≠ 0) :
  let discriminant := b^2 - 4*a*c
  a * x^2 + b * x + c = 0 ∧ a = 1 ∧ b = -4 * Real.sqrt 3 ∧ c = 12 →
  discriminant = 0 ∧ ∃! x, a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l88_8834


namespace NUMINAMATH_CALUDE_object_length_doubles_on_day_two_l88_8815

/-- Calculates the length multiplier after n days -/
def lengthMultiplier (n : ℕ) : ℚ :=
  (n + 2 : ℚ) / 2

theorem object_length_doubles_on_day_two :
  ∃ n : ℕ, lengthMultiplier n = 2 ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_object_length_doubles_on_day_two_l88_8815


namespace NUMINAMATH_CALUDE_stratified_sampling_total_size_l88_8841

theorem stratified_sampling_total_size 
  (district1_ratio : ℚ) 
  (district2_ratio : ℚ) 
  (district3_ratio : ℚ) 
  (largest_district_sample : ℕ) : 
  district1_ratio + district2_ratio + district3_ratio = 1 →
  district3_ratio > district1_ratio →
  district3_ratio > district2_ratio →
  district3_ratio = 1/2 →
  largest_district_sample = 60 →
  2 * largest_district_sample = 120 :=
by
  sorry

#check stratified_sampling_total_size

end NUMINAMATH_CALUDE_stratified_sampling_total_size_l88_8841


namespace NUMINAMATH_CALUDE_seventh_number_is_177_l88_8881

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 15

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem seventh_number_is_177 : nth_valid_number 7 = 177 := by sorry

end NUMINAMATH_CALUDE_seventh_number_is_177_l88_8881


namespace NUMINAMATH_CALUDE_system_solution_l88_8857

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 8) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l88_8857


namespace NUMINAMATH_CALUDE_angle_expression_simplification_l88_8867

theorem angle_expression_simplification (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) (h3 : Real.cos α = -Real.sqrt 5 / 5) :
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.tan (-α - π) * Real.sin (-π - α)) = 1 / 10 := by
sorry


end NUMINAMATH_CALUDE_angle_expression_simplification_l88_8867


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l88_8807

/-- An arithmetic sequence is a sequence where the difference between 
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 45) 
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l88_8807


namespace NUMINAMATH_CALUDE_min_value_of_expression_l88_8876

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), min = 5 - Real.sqrt 5 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + 12 = 0 → |2*x - y - 2| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l88_8876


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l88_8882

theorem power_of_three_mod_eleven : 3^1234 % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l88_8882


namespace NUMINAMATH_CALUDE_sin_cos_inequality_l88_8842

theorem sin_cos_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π ∧ Real.sin (x - π / 6) > Real.cos x → 
  π / 3 < x ∧ x < 4 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_inequality_l88_8842


namespace NUMINAMATH_CALUDE_books_sold_on_monday_l88_8832

theorem books_sold_on_monday (initial_stock : ℕ) (tuesday_sold : ℕ) (wednesday_sold : ℕ) (thursday_sold : ℕ) (friday_sold : ℕ) (unsold : ℕ) : 
  initial_stock = 800 →
  tuesday_sold = 10 →
  wednesday_sold = 20 →
  thursday_sold = 44 →
  friday_sold = 66 →
  unsold = 600 →
  initial_stock - (tuesday_sold + wednesday_sold + thursday_sold + friday_sold + unsold) = 60 := by
  sorry


end NUMINAMATH_CALUDE_books_sold_on_monday_l88_8832


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l88_8863

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (m p q : ℕ) 
  (h_arithmetic : is_arithmetic_sequence a) (h_positive : m > 0 ∧ p > 0 ∧ q > 0) :
  (p + q = 2 * m → a p + a q = 2 * a m) ∧
  ∃ b : ℕ → ℝ, is_arithmetic_sequence b ∧ ∃ m' p' q' : ℕ, 
    m' > 0 ∧ p' > 0 ∧ q' > 0 ∧ b p' + b q' = 2 * b m' ∧ p' + q' ≠ 2 * m' :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l88_8863


namespace NUMINAMATH_CALUDE_amy_pencil_count_l88_8808

/-- The number of pencils Amy has after buying and giving away some pencils -/
def final_pencil_count (initial : ℕ) (bought_monday : ℕ) (bought_tuesday : ℕ) (given_away : ℕ) : ℕ :=
  initial + bought_monday + bought_tuesday - given_away

/-- Theorem stating that Amy has 12 pencils at the end -/
theorem amy_pencil_count : final_pencil_count 3 7 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_amy_pencil_count_l88_8808


namespace NUMINAMATH_CALUDE_quadratic_real_root_l88_8824

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_l88_8824


namespace NUMINAMATH_CALUDE_product_calculation_l88_8818

theorem product_calculation : 12 * 0.2 * 3 * 0.1 / 0.6 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l88_8818


namespace NUMINAMATH_CALUDE_theater_seats_l88_8897

/-- The number of people watching the movie -/
def people_watching : ℕ := 532

/-- The number of empty seats -/
def empty_seats : ℕ := 218

/-- The total number of seats in the theater -/
def total_seats : ℕ := people_watching + empty_seats

theorem theater_seats : total_seats = 750 := by sorry

end NUMINAMATH_CALUDE_theater_seats_l88_8897


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l88_8868

/-- Rainfall data for a week --/
structure RainfallData where
  monday_morning : ℝ
  monday_afternoon : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ
  daily_average : ℝ
  num_days : ℕ

/-- Theorem stating the ratio of Tuesday's rainfall to Monday's total rainfall --/
theorem tuesday_to_monday_ratio (data : RainfallData) : 
  data.monday_morning = 2 ∧ 
  data.monday_afternoon = 1 ∧ 
  data.wednesday = 0 ∧ 
  data.thursday = 1 ∧ 
  data.friday = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday ∧
  data.daily_average = 4 ∧
  data.num_days = 5 ∧
  data.daily_average * data.num_days = data.monday_morning + data.monday_afternoon + data.tuesday + data.wednesday + data.thursday + data.friday →
  data.tuesday / (data.monday_morning + data.monday_afternoon) = 2 := by
sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l88_8868


namespace NUMINAMATH_CALUDE_negation_equivalence_l88_8866

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → Real.sqrt x > x + 1) ↔ (∃ x : ℝ, x > 0 ∧ Real.sqrt x ≤ x + 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l88_8866


namespace NUMINAMATH_CALUDE_min_value_of_function_l88_8823

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  ∃ (y : ℝ), y = x + 4 / x^2 ∧ ∀ (z : ℝ), z = x + 4 / x^2 → y ≤ z ∧ y = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l88_8823


namespace NUMINAMATH_CALUDE_first_test_score_l88_8898

theorem first_test_score (second_score average : ℝ) (h1 : second_score = 84) (h2 : average = 81) :
  let first_score := 2 * average - second_score
  first_score = 78 := by
sorry

end NUMINAMATH_CALUDE_first_test_score_l88_8898


namespace NUMINAMATH_CALUDE_albert_earnings_increase_l88_8848

theorem albert_earnings_increase (E : ℝ) (P : ℝ) : 
  E * (1 + P / 100) = 598 →
  E * 1.35 = 621 →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_albert_earnings_increase_l88_8848


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l88_8856

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 * Complex.I - 5) / (2 - Complex.I)) = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l88_8856


namespace NUMINAMATH_CALUDE_negative_x_times_three_minus_x_l88_8827

theorem negative_x_times_three_minus_x (x : ℝ) : -x * (3 - x) = -3*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_times_three_minus_x_l88_8827


namespace NUMINAMATH_CALUDE_buffer_water_requirement_l88_8826

/-- Given a buffer solution where water constitutes 1/3 of the total volume,
    prove that 0.72 liters of the buffer solution requires 0.24 liters of water. -/
theorem buffer_water_requirement (total_volume : ℝ) (water_fraction : ℝ) 
    (h1 : total_volume = 0.72)
    (h2 : water_fraction = 1/3) : 
  total_volume * water_fraction = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_buffer_water_requirement_l88_8826


namespace NUMINAMATH_CALUDE_age_ratio_problem_l88_8855

theorem age_ratio_problem (ali_age yusaf_age umar_age : ℕ) : 
  ali_age = 8 →
  ali_age = yusaf_age + 3 →
  ∃ k : ℕ, umar_age = k * yusaf_age →
  umar_age = 10 →
  umar_age / yusaf_age = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l88_8855


namespace NUMINAMATH_CALUDE_second_group_size_l88_8888

/-- Represents a choir split into three groups -/
structure Choir :=
  (total : ℕ)
  (group1 : ℕ)
  (group2 : ℕ)
  (group3 : ℕ)
  (sum_eq_total : group1 + group2 + group3 = total)

/-- Theorem: Given a choir with 70 total members, 25 in the first group,
    and 15 in the third group, the second group must have 30 members -/
theorem second_group_size (c : Choir)
  (h1 : c.total = 70)
  (h2 : c.group1 = 25)
  (h3 : c.group3 = 15) :
  c.group2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_group_size_l88_8888


namespace NUMINAMATH_CALUDE_circular_path_meeting_time_l88_8840

theorem circular_path_meeting_time (c : ℝ) : 
  c > 0 ∧ 
  (6⁻¹ : ℝ) > 0 ∧ 
  c⁻¹ > 0 ∧
  (((6 * c) / (c + 6) + 1) * c⁻¹ = 1) →
  c = 3 :=
by sorry

end NUMINAMATH_CALUDE_circular_path_meeting_time_l88_8840


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l88_8847

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ M) ∩ (U \ N) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l88_8847


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l88_8864

theorem least_positive_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧ 
  n % 2 = 0 ∧ 
  n % 5 = 1 ∧ 
  n % 4 = 2 ∧
  ∀ m : ℕ, m > 0 ∧ m % 2 = 0 ∧ m % 5 = 1 ∧ m % 4 = 2 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l88_8864


namespace NUMINAMATH_CALUDE_lego_set_cost_l88_8839

/-- Represents the sale of toys with given conditions and calculates the cost of a Lego set --/
def toy_sale (total_after_tax : ℚ) (car_price : ℚ) (car_discount : ℚ) (num_cars : ℕ) 
             (num_action_figures : ℕ) (tax_rate : ℚ) : ℚ :=
  let discounted_car_price := car_price * (1 - car_discount)
  let action_figure_price := 2 * discounted_car_price
  let board_game_price := action_figure_price + discounted_car_price
  let known_items_total := num_cars * discounted_car_price + 
                           num_action_figures * action_figure_price + 
                           board_game_price
  let total_before_tax := total_after_tax / (1 + tax_rate)
  total_before_tax - known_items_total

/-- Theorem stating that the Lego set costs $85 before tax --/
theorem lego_set_cost : 
  toy_sale 136.5 5 0.1 3 2 0.05 = 85 := by
  sorry

end NUMINAMATH_CALUDE_lego_set_cost_l88_8839


namespace NUMINAMATH_CALUDE_misread_number_correction_l88_8870

theorem misread_number_correction (n : ℕ) (initial_avg correct_avg wrong_num : ℝ) (correct_num : ℝ) : 
  n = 10 →
  initial_avg = 15 →
  wrong_num = 26 →
  correct_avg = 16 →
  n * initial_avg + (correct_num - wrong_num) = n * correct_avg →
  correct_num = 36 := by
sorry

end NUMINAMATH_CALUDE_misread_number_correction_l88_8870


namespace NUMINAMATH_CALUDE_solution_set_M_range_of_a_l88_8852

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Define the solution set M
def M : Set ℝ := {x | -2/3 ≤ x ∧ x ≤ 6}

-- Define the property for part (2)
def property (a : ℝ) : Prop := ∀ x, x ≥ a → f x ≤ x - a

-- Theorem for part (1)
theorem solution_set_M : {x : ℝ | f x ≥ -2} = M := by sorry

-- Theorem for part (2)
theorem range_of_a : {a : ℝ | property a} = {a | a ≤ -2 ∨ a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_M_range_of_a_l88_8852


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l88_8833

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l88_8833


namespace NUMINAMATH_CALUDE_sum_first_10_odd_integers_l88_8829

theorem sum_first_10_odd_integers : 
  (Finset.range 10).sum (fun i => 2 * i + 1) = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_10_odd_integers_l88_8829


namespace NUMINAMATH_CALUDE_min_value_expression_l88_8869

theorem min_value_expression (a b : ℝ) (h : a^2 ≥ 8*b) :
  ∃ (min : ℝ), min = (9:ℝ)/8 ∧ ∀ (x y : ℝ), x^2 ≥ 8*y →
    (1 - x)^2 + (1 - 2*y)^2 + (x - 2*y)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l88_8869


namespace NUMINAMATH_CALUDE_reduction_equivalence_original_value_proof_l88_8886

theorem reduction_equivalence (original : ℝ) (reduced : ℝ) : 
  reduced = original * (1 / 1000) ↔ reduced = original * 0.001 :=
by sorry

theorem original_value_proof : 
  ∃ (original : ℝ), 16.9 * (1 / 1000) = 0.0169 ∧ original = 16.9 :=
by sorry

end NUMINAMATH_CALUDE_reduction_equivalence_original_value_proof_l88_8886


namespace NUMINAMATH_CALUDE_rice_division_l88_8871

theorem rice_division (total_weight : ℚ) (num_containers : ℕ) (ounces_per_pound : ℕ) :
  total_weight = 25 / 2 →
  num_containers = 4 →
  ounces_per_pound = 16 →
  (total_weight / num_containers) * ounces_per_pound = 50 := by
sorry

end NUMINAMATH_CALUDE_rice_division_l88_8871


namespace NUMINAMATH_CALUDE_number_satisfying_equation_l88_8878

theorem number_satisfying_equation : ∃ x : ℝ, x^2 + 4 = 5*x ∧ (x = 4 ∨ x = 1) := by sorry

end NUMINAMATH_CALUDE_number_satisfying_equation_l88_8878


namespace NUMINAMATH_CALUDE_sum_of_specific_series_l88_8858

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_series :
  let a : ℚ := 1/2
  let r : ℚ := -1/4
  let n : ℕ := 6
  geometric_series a r n = 4095/10240 := by
sorry

end NUMINAMATH_CALUDE_sum_of_specific_series_l88_8858


namespace NUMINAMATH_CALUDE_conference_room_chairs_l88_8899

/-- The number of chairs in the conference room -/
def num_chairs : ℕ := 40

/-- The capacity of each chair -/
def chair_capacity : ℕ := 2

/-- The fraction of unoccupied chairs -/
def unoccupied_fraction : ℚ := 2/5

/-- The number of board members who attended the meeting -/
def attendees : ℕ := 48

theorem conference_room_chairs :
  (num_chairs : ℚ) * chair_capacity * (1 - unoccupied_fraction) = attendees ∧
  num_chairs * chair_capacity = num_chairs * 2 :=
sorry

end NUMINAMATH_CALUDE_conference_room_chairs_l88_8899


namespace NUMINAMATH_CALUDE_odd_function_range_l88_8845

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_neg : ∀ x, x < 0 → f x = 9*x + a^2/x + 7)
    (h_pos : ∀ x, x ≥ 0 → f x ≥ a + 1) :
  a ≤ -8/7 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_range_l88_8845


namespace NUMINAMATH_CALUDE_hyperbola_circle_no_intersection_l88_8843

/-- The range of real values of a for which the asymptotes of the hyperbola x^2/4 - y^2 = 1
    have no common points with the circle x^2 + y^2 - 2ax + 1 = 0 -/
theorem hyperbola_circle_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → x^2 + y^2 - 2*a*x + 1 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-Real.sqrt 5 / 2) (-1) ∪ Set.Ioo 1 (Real.sqrt 5 / 2)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_circle_no_intersection_l88_8843


namespace NUMINAMATH_CALUDE_treaty_of_paris_preliminary_articles_l88_8895

/-- Represents days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculates the day of the week given a number of days before a known day -/
def daysBefore (knownDay : DayOfWeek) (daysBefore : Nat) : DayOfWeek :=
  sorry

theorem treaty_of_paris_preliminary_articles :
  let treatyDay : DayOfWeek := DayOfWeek.Thursday
  let daysBetween : Nat := 621
  daysBefore treatyDay daysBetween = DayOfWeek.Tuesday :=
sorry

end NUMINAMATH_CALUDE_treaty_of_paris_preliminary_articles_l88_8895


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l88_8814

/-- A geometric progression with second term 5 and third term 1 has first term 25. -/
theorem geometric_progression_first_term (a : ℝ) (q : ℝ) : 
  a * q = 5 ∧ a * q^2 = 1 → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l88_8814


namespace NUMINAMATH_CALUDE_rectangle_perimeter_proof_l88_8874

def square_perimeter : ℝ := 24
def rectangle_width : ℝ := 4

theorem rectangle_perimeter_proof :
  let square_side := square_perimeter / 4
  let square_area := square_side ^ 2
  let rectangle_length := square_area / rectangle_width
  2 * (rectangle_length + rectangle_width) = 26 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_proof_l88_8874


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l88_8835

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 5 + 2*I ∨ x = 5 - 2*I) ∧
    (3 : ℝ) * (5 + 2*I)^2 + a * (5 + 2*I) + b = 0 ∧
    a = -30 ∧ b = 87 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l88_8835


namespace NUMINAMATH_CALUDE_three_numbers_sum_l88_8801

theorem three_numbers_sum (x y z : ℝ) : 
  x ≤ y ∧ y ≤ z →  -- Ascending order
  y = 7 →  -- Median is 7
  (x + y + z) / 3 = x + 12 →  -- Mean is 12 more than least
  (x + y + z) / 3 = z - 18 →  -- Mean is 18 less than greatest
  x + y + z = 39 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l88_8801


namespace NUMINAMATH_CALUDE_distance_between_points_l88_8894

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -1 ∧ B = 2020 → |A - B| = 2021 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l88_8894
