import Mathlib

namespace NUMINAMATH_CALUDE_sphere_volume_after_drilling_l2791_279179

/-- The remaining volume of a sphere after drilling cylindrical holes -/
theorem sphere_volume_after_drilling (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (hole3_depth hole3_diameter : ℝ) : 
  sphere_diameter = 24 →
  hole1_depth = 10 → hole1_diameter = 3 →
  hole2_depth = 10 → hole2_diameter = 3 →
  hole3_depth = 5 → hole3_diameter = 4 →
  (4 / 3 * π * (sphere_diameter / 2)^3) - 
  (π * (hole1_diameter / 2)^2 * hole1_depth) - 
  (π * (hole2_diameter / 2)^2 * hole2_depth) - 
  (π * (hole3_diameter / 2)^2 * hole3_depth) = 2239 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_after_drilling_l2791_279179


namespace NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l2791_279130

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph is a set of three distinct vertices -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A triangle is monochromatic if all its edges have the same color -/
def IsMonochromatic (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ 
  c t.val.1 t.val.2.1 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K_17 contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ (c : Coloring 17), ∃ (t : Triangle 17), IsMonochromatic 17 c t :=
sorry


end NUMINAMATH_CALUDE_monochromatic_triangle_in_K17_l2791_279130


namespace NUMINAMATH_CALUDE_union_equals_A_l2791_279121

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - a = 0}

-- State the theorem
theorem union_equals_A (a : ℝ) : (A ∪ B a = A) ↔ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l2791_279121


namespace NUMINAMATH_CALUDE_cousins_money_correct_l2791_279199

/-- The amount of money Jim's cousin brought to the restaurant --/
def cousins_money : ℝ := 10

/-- The total cost of the meal --/
def meal_cost : ℝ := 24

/-- The percentage of their combined money spent on the meal --/
def spent_percentage : ℝ := 0.8

/-- The amount of money Jim brought --/
def jims_money : ℝ := 20

/-- Theorem stating that the calculated amount Jim's cousin brought is correct --/
theorem cousins_money_correct : 
  cousins_money = (meal_cost / spent_percentage) - jims_money := by
  sorry

end NUMINAMATH_CALUDE_cousins_money_correct_l2791_279199


namespace NUMINAMATH_CALUDE_quadratic_factor_sum_l2791_279177

theorem quadratic_factor_sum (a w c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factor_sum_l2791_279177


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2791_279185

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l2791_279185


namespace NUMINAMATH_CALUDE_front_wheel_cost_l2791_279153

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def remaining_amount : ℕ := 20

theorem front_wheel_cost : 
  initial_amount - frame_cost - remaining_amount = 25 := by sorry

end NUMINAMATH_CALUDE_front_wheel_cost_l2791_279153


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2791_279141

-- Define the box contents
def white_balls : ℕ := 1
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2791_279141


namespace NUMINAMATH_CALUDE_skateboard_price_l2791_279127

theorem skateboard_price (upfront_payment : ℝ) (upfront_percentage : ℝ) 
  (h1 : upfront_payment = 60)
  (h2 : upfront_percentage = 20) : 
  let full_price := upfront_payment / (upfront_percentage / 100)
  full_price = 300 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_price_l2791_279127


namespace NUMINAMATH_CALUDE_task_assignment_count_l2791_279175

def select_and_assign (n m : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m 2 * 2

theorem task_assignment_count : select_and_assign 10 4 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_task_assignment_count_l2791_279175


namespace NUMINAMATH_CALUDE_definite_integral_equals_twenty_minus_six_pi_l2791_279158

theorem definite_integral_equals_twenty_minus_six_pi :
  let f : ℝ → ℝ := λ x => x^4 / ((16 - x^2) * Real.sqrt (16 - x^2))
  let a : ℝ := 0
  let b : ℝ := 2 * Real.sqrt 2
  ∫ x in a..b, f x = 20 - 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_definite_integral_equals_twenty_minus_six_pi_l2791_279158


namespace NUMINAMATH_CALUDE_unique_cube_fourth_power_l2791_279104

theorem unique_cube_fourth_power : 
  ∃! (K : ℤ), ∃ (Z : ℤ),
    600 < Z ∧ Z < 2000 ∧ 
    Z = K^4 ∧ 
    ∃ (n : ℤ), Z = n^3 ∧
    K = 8 :=
sorry

end NUMINAMATH_CALUDE_unique_cube_fourth_power_l2791_279104


namespace NUMINAMATH_CALUDE_nonagon_triangle_probability_l2791_279134

/-- The number of vertices in a regular nonagon -/
def nonagon_vertices : ℕ := 9

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 9 vertices -/
def total_triangles : ℕ := Nat.choose nonagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the nonagon -/
def favorable_triangles : ℕ := 54

/-- The probability of forming a triangle with at least one side being a side of the nonagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem nonagon_triangle_probability : probability = 9 / 14 := by sorry

end NUMINAMATH_CALUDE_nonagon_triangle_probability_l2791_279134


namespace NUMINAMATH_CALUDE_two_digit_divisible_by_six_sum_fifteen_l2791_279152

theorem two_digit_divisible_by_six_sum_fifteen (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧                 -- n is a two-digit number
  n % 6 = 0 ∧                        -- n is divisible by 6
  (n / 10 + n % 10 = 15) →           -- sum of digits is 15
  (n / 10) * (n % 10) = 56 ∨ (n / 10) * (n % 10) = 54 := by
sorry

end NUMINAMATH_CALUDE_two_digit_divisible_by_six_sum_fifteen_l2791_279152


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l2791_279170

/-- 
A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis 
if and only if y₁ = y₂.
-/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₁ = y₂

/-- 
The value of k for which the line through the points (3, 2k+1) and (8, 4k-5) 
is parallel to the x-axis.
-/
theorem line_parallel_to_x_axis (k : ℝ) : 
  parallel_to_x_axis 3 (2*k+1) 8 (4*k-5) ↔ k = 3 := by
  sorry

#check line_parallel_to_x_axis

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l2791_279170


namespace NUMINAMATH_CALUDE_true_compound_props_l2791_279118

def p₁ : Prop := True
def p₂ : Prop := False
def p₃ : Prop := False
def p₄ : Prop := True

def compound_prop_1 : Prop := p₁ ∧ p₄
def compound_prop_2 : Prop := p₁ ∧ p₂
def compound_prop_3 : Prop := ¬p₂ ∨ p₃
def compound_prop_4 : Prop := ¬p₃ ∨ ¬p₄

theorem true_compound_props :
  {compound_prop_1, compound_prop_3, compound_prop_4} = 
  {p : Prop | p = compound_prop_1 ∨ p = compound_prop_2 ∨ p = compound_prop_3 ∨ p = compound_prop_4 ∧ p} :=
by sorry

end NUMINAMATH_CALUDE_true_compound_props_l2791_279118


namespace NUMINAMATH_CALUDE_minimum_value_problem_l2791_279117

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (a + 1/b) * (a + 1/b - 2020) + (b + 1/a) * (b + 1/a - 2020) = -2040200 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_l2791_279117


namespace NUMINAMATH_CALUDE_run_difference_is_240_l2791_279186

/-- The width of the street in feet -/
def street_width : ℝ := 30

/-- The side length of the square block in feet -/
def block_side : ℝ := 500

/-- The perimeter of Sarah's run (inner side of the block) -/
def sarah_perimeter : ℝ := 4 * block_side

/-- The perimeter of Sam's run (outer side of the block) -/
def sam_perimeter : ℝ := 4 * (block_side + 2 * street_width)

/-- The difference in distance run by Sam and Sarah -/
def run_difference : ℝ := sam_perimeter - sarah_perimeter

theorem run_difference_is_240 : run_difference = 240 := by
  sorry

end NUMINAMATH_CALUDE_run_difference_is_240_l2791_279186


namespace NUMINAMATH_CALUDE_sports_meet_participation_l2791_279196

/-- The number of students participating in both track and field and ball games -/
def students_in_track_and_ball (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ)
  (swimming_and_track : ℕ) (swimming_and_ball : ℕ) : ℕ :=
  swimming + track + ball - swimming_and_track - swimming_and_ball - total

theorem sports_meet_participation :
  students_in_track_and_ball 28 15 8 14 3 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sports_meet_participation_l2791_279196


namespace NUMINAMATH_CALUDE_system_solution_l2791_279194

theorem system_solution (x y a : ℝ) : 
  x + 2 * y = a ∧ 
  x - 2 * y = 2 ∧ 
  x = 4 → 
  a = 6 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2791_279194


namespace NUMINAMATH_CALUDE_wright_brothers_first_flight_l2791_279119

/-- Represents the different groups of brothers mentioned in the problem -/
inductive Brothers
  | Bell
  | Hale
  | Wright
  | Leon

/-- Represents an aircraft -/
structure Aircraft where
  name : String

/-- Represents a flight achievement -/
structure FlightAchievement where
  date : String
  aircraft : Aircraft
  achievers : Brothers

/-- The first powered human flight -/
def first_powered_flight : FlightAchievement :=
  { date := "December 1903"
  , aircraft := { name := "Flyer 1" }
  , achievers := Brothers.Wright }

/-- Theorem stating that the Wright Brothers achieved the first powered human flight -/
theorem wright_brothers_first_flight :
  first_powered_flight.achievers = Brothers.Wright :=
by sorry

end NUMINAMATH_CALUDE_wright_brothers_first_flight_l2791_279119


namespace NUMINAMATH_CALUDE_locus_of_point_M_l2791_279163

/-- The locus of points M(x,y) forming triangles with fixed points A(-1,0) and B(1,0),
    where the sum of slopes of AM and BM is 2. -/
theorem locus_of_point_M (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (y / (x + 1) + y / (x - 1) = 2) → (x^2 - x*y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_point_M_l2791_279163


namespace NUMINAMATH_CALUDE_combined_average_age_l2791_279135

theorem combined_average_age (people_c people_d : ℕ) (avg_age_c avg_age_d : ℚ) :
  people_c = 8 →
  people_d = 6 →
  avg_age_c = 30 →
  avg_age_d = 35 →
  (people_c * avg_age_c + people_d * avg_age_d) / (people_c + people_d) = 32 := by
  sorry

end NUMINAMATH_CALUDE_combined_average_age_l2791_279135


namespace NUMINAMATH_CALUDE_valid_pairs_l2791_279101

def is_valid_pair (m n : ℕ+) : Prop :=
  (3^m.val + 1) % (m.val * n.val) = 0 ∧ (3^n.val + 1) % (m.val * n.val) = 0

theorem valid_pairs :
  ∀ m n : ℕ+, is_valid_pair m n ↔ 
    ((m = 1 ∧ n = 1) ∨ 
     (m = 1 ∧ n = 2) ∨ 
     (m = 1 ∧ n = 4) ∨ 
     (m = 2 ∧ n = 1) ∨ 
     (m = 4 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2791_279101


namespace NUMINAMATH_CALUDE_total_selection_methods_is_eight_l2791_279142

/-- The number of students who can only use the synthetic method -/
def synthetic_students : Nat := 5

/-- The number of students who can only use the analytical method -/
def analytical_students : Nat := 3

/-- The total number of ways to select a student to prove the problem -/
def total_selection_methods : Nat := synthetic_students + analytical_students

/-- Theorem stating that the total number of selection methods is 8 -/
theorem total_selection_methods_is_eight : total_selection_methods = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_methods_is_eight_l2791_279142


namespace NUMINAMATH_CALUDE_path_length_eq_three_times_PQ_l2791_279193

/-- The length of the segment PQ -/
def PQ_length : ℝ := 73

/-- The length of the path along the squares constructed on the segments of PQ -/
def path_length : ℝ := 3 * PQ_length

theorem path_length_eq_three_times_PQ : path_length = 3 * PQ_length := by
  sorry

end NUMINAMATH_CALUDE_path_length_eq_three_times_PQ_l2791_279193


namespace NUMINAMATH_CALUDE_alan_needs_17_votes_to_win_l2791_279154

/-- Represents the number of votes for each candidate -/
structure VoteCount where
  sally : Nat
  katie : Nat
  alan : Nat

/-- The problem setup -/
def totalVoters : Nat := 130

def currentVotes : VoteCount := {
  sally := 24,
  katie := 29,
  alan := 37
}

/-- Alan needs at least this many more votes to be certain of winning -/
def minVotesNeeded : Nat := 17

theorem alan_needs_17_votes_to_win : 
  ∀ (finalVotes : VoteCount),
  finalVotes.sally ≥ currentVotes.sally ∧ 
  finalVotes.katie ≥ currentVotes.katie ∧
  finalVotes.alan ≥ currentVotes.alan ∧
  finalVotes.sally + finalVotes.katie + finalVotes.alan = totalVoters →
  (finalVotes.alan = currentVotes.alan + minVotesNeeded - 1 → 
   ¬(finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie)) ∧
  (finalVotes.alan ≥ currentVotes.alan + minVotesNeeded → 
   finalVotes.alan > finalVotes.sally ∧ finalVotes.alan > finalVotes.katie) :=
by sorry

#check alan_needs_17_votes_to_win

end NUMINAMATH_CALUDE_alan_needs_17_votes_to_win_l2791_279154


namespace NUMINAMATH_CALUDE_max_value_of_f_l2791_279116

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x - 4

theorem max_value_of_f (a : ℝ) :
  (f' a (-1) = 0) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 4, f a x = 42) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f a x ≤ 42) :=
by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_max_value_of_f_l2791_279116


namespace NUMINAMATH_CALUDE_complex_distance_sum_constant_l2791_279180

theorem complex_distance_sum_constant (w : ℂ) (h : Complex.abs (w - (3 + 2*I)) = 3) :
  Complex.abs (w - (2 - 3*I))^2 + Complex.abs (w - (4 + 5*I))^2 = 71 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_constant_l2791_279180


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l2791_279172

theorem right_triangle_cosine (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : a = 7) (h3 : c = 25) :
  b / c = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l2791_279172


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2791_279160

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2791_279160


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l2791_279168

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (b c : ℝ), (∀ x y : ℝ, y = (a^2 - a) * x + 2 ↔ y = 6 * x + 3)) → 
  (a = -2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l2791_279168


namespace NUMINAMATH_CALUDE_salary_degrees_in_circle_graph_l2791_279100

-- Define the percentages for each category
def transportation_percent : ℝ := 15
def research_dev_percent : ℝ := 9
def utilities_percent : ℝ := 5
def equipment_percent : ℝ := 4
def supplies_percent : ℝ := 2

-- Define the total degrees in a circle
def total_degrees : ℝ := 360

-- Theorem statement
theorem salary_degrees_in_circle_graph :
  let other_categories_percent := transportation_percent + research_dev_percent + 
                                  utilities_percent + equipment_percent + supplies_percent
  let salary_percent := 100 - other_categories_percent
  let salary_degrees := (salary_percent / 100) * total_degrees
  salary_degrees = 234 := by
sorry


end NUMINAMATH_CALUDE_salary_degrees_in_circle_graph_l2791_279100


namespace NUMINAMATH_CALUDE_survival_rate_all_survived_survival_rate_97_trees_l2791_279123

/-- The survival rate of trees given the number of surviving trees and the total number of planted trees. -/
def survival_rate (surviving : ℕ) (total : ℕ) : ℚ :=
  (surviving : ℚ) / (total : ℚ)

/-- Theorem stating that the survival rate is 100% when all planted trees survive. -/
theorem survival_rate_all_survived (n : ℕ) (h : n > 0) :
  survival_rate n n = 1 := by
  sorry

/-- The specific case for 97 trees. -/
theorem survival_rate_97_trees :
  survival_rate 97 97 = 1 := by
  sorry

end NUMINAMATH_CALUDE_survival_rate_all_survived_survival_rate_97_trees_l2791_279123


namespace NUMINAMATH_CALUDE_set_operations_l2791_279140

def A : Set ℝ := {x | x^2 + 3*x - 4 > 0}
def B : Set ℝ := {x | x^2 - x - 6 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ 1 ∨ x ≥ 3}) ∧
  (A ∪ Set.compl B = {x | x ≤ -2 ∨ x > 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2791_279140


namespace NUMINAMATH_CALUDE_min_value_M_l2791_279120

theorem min_value_M (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (4 * Real.sqrt 3) / 3 ≤ max (a + 1/b) (max (b + 1/c) (c + 1/a)) ∧
  ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ a^2 + b^2 + c^2 = 1 ∧
    (4 * Real.sqrt 3) / 3 = max (a + 1/b) (max (b + 1/c) (c + 1/a)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_M_l2791_279120


namespace NUMINAMATH_CALUDE_nested_subtract_201_l2791_279107

/-- Recursive function to represent nested subtractions -/
def nestedSubtract (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtract x n

/-- Theorem stating that the nested subtraction equals 1 iff x = 201 -/
theorem nested_subtract_201 (x : ℝ) :
  nestedSubtract x 199 = 1 ↔ x = 201 := by
  sorry

#check nested_subtract_201

end NUMINAMATH_CALUDE_nested_subtract_201_l2791_279107


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2791_279115

/-- Definition of a circle with center (h, k) and radius r -/
def Circle (h k r : ℝ) := {(x, y) : ℝ × ℝ | (x - h)^2 + (y - k)^2 = r^2}

/-- The intersection line of two circles -/
def IntersectionLine (c1 c2 : ℝ × ℝ × ℝ) : ℝ × ℝ → Prop :=
  let (h1, k1, r1) := c1
  let (h2, k2, r2) := c2
  λ (x, y) => x + y = -59/34

theorem intersection_line_of_circles :
  let c1 : ℝ × ℝ × ℝ := (-12, -6, 15)
  let c2 : ℝ × ℝ × ℝ := (4, 11, 9)
  ∀ (p : ℝ × ℝ), p ∈ Circle c1.1 c1.2.1 c1.2.2 ∩ Circle c2.1 c2.2.1 c2.2.2 →
    IntersectionLine c1 c2 p :=
by
  sorry

#check intersection_line_of_circles

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2791_279115


namespace NUMINAMATH_CALUDE_exactly_three_rainy_days_probability_l2791_279169

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of days in the period -/
def num_days : ℕ := 4

/-- The number of rainy days we're interested in -/
def num_rainy_days : ℕ := 3

/-- The probability of rain on any given day -/
def rain_probability : ℝ := 0.5

theorem exactly_three_rainy_days_probability :
  binomial_probability num_days num_rainy_days rain_probability = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_rainy_days_probability_l2791_279169


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2791_279157

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2791_279157


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l2791_279145

theorem polynomial_equation_solution : 
  ∃ x : ℝ, ((x^3 * 0.76^3 - 0.008) / (x^2 * 0.76^2 + x * 0.76 * 0.2 + 0.04) = 0) ∧ 
  (abs (x - 0.262) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l2791_279145


namespace NUMINAMATH_CALUDE_smallest_in_set_l2791_279146

def S : Set ℚ := {1/2, 2/3, 1/4, 5/6, 7/12}

theorem smallest_in_set : ∀ x ∈ S, 1/4 ≤ x := by sorry

end NUMINAMATH_CALUDE_smallest_in_set_l2791_279146


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2791_279162

theorem complex_square_simplification :
  let z : ℂ := 4 - 3 * I
  z^2 = 7 - 24 * I := by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2791_279162


namespace NUMINAMATH_CALUDE_new_barbell_cost_l2791_279187

theorem new_barbell_cost (old_cost : ℝ) (percentage_increase : ℝ) : 
  old_cost = 250 → percentage_increase = 0.3 → 
  old_cost + old_cost * percentage_increase = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l2791_279187


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2791_279111

theorem sqrt_four_fourth_powers_sum (h : ℝ) : 
  h = Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) → h = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2791_279111


namespace NUMINAMATH_CALUDE_total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2791_279151

/-- The percentage of marbles that are solid-colored -/
def solid_colored_percentage : ℝ := 0.90

/-- The percentage of marbles that have patterns -/
def patterned_percentage : ℝ := 0.10

/-- The percentage of red marbles among solid-colored marbles -/
def red_percentage : ℝ := 0.40

/-- The percentage of blue marbles among solid-colored marbles -/
def blue_percentage : ℝ := 0.30

/-- The percentage of green marbles among solid-colored marbles -/
def green_percentage : ℝ := 0.20

/-- The percentage of yellow marbles among solid-colored marbles -/
def yellow_percentage : ℝ := 0.10

/-- All marbles are either solid-colored or patterned -/
theorem total_percentage : solid_colored_percentage + patterned_percentage = 1 := by sorry

/-- The sum of percentages for all solid-colored marbles is 100% -/
theorem solid_colored_sum :
  red_percentage + blue_percentage + green_percentage + yellow_percentage = 1 := by sorry

/-- The percentage of marbles that are a solid color other than yellow is 81% -/
theorem solid_color_not_yellow_percentage :
  solid_colored_percentage * (red_percentage + blue_percentage + green_percentage) = 0.81 := by sorry

end NUMINAMATH_CALUDE_total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2791_279151


namespace NUMINAMATH_CALUDE_max_pages_copied_l2791_279108

/-- The cost in cents to copy 4 pages -/
def cost_per_4_pages : ℚ := 7

/-- The budget in dollars -/
def budget : ℚ := 15

/-- The number of pages that can be copied with the given budget -/
def pages_copied : ℕ := 857

/-- Theorem stating the maximum number of complete pages that can be copied -/
theorem max_pages_copied : 
  ⌊(budget * 100 / cost_per_4_pages) * 4⌋ = pages_copied :=
sorry

end NUMINAMATH_CALUDE_max_pages_copied_l2791_279108


namespace NUMINAMATH_CALUDE_playground_area_l2791_279161

theorem playground_area (total_posts : ℕ) (post_spacing : ℕ) 
  (h1 : total_posts = 28)
  (h2 : post_spacing = 6)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  ∃ (width length : ℕ), 
    width * length = 1188 ∧ 
    width = post_spacing * short_side ∧ 
    length = post_spacing * long_side :=
sorry

end NUMINAMATH_CALUDE_playground_area_l2791_279161


namespace NUMINAMATH_CALUDE_c_decreases_as_r_increases_l2791_279132

theorem c_decreases_as_r_increases (e n r : ℝ) (h_e : e > 0) (h_n : n > 0) (h_r : r > 0) :
  ∀ (R₁ R₂ : ℝ), R₁ > 0 → R₂ > 0 → R₂ > R₁ →
  (e * n) / (R₁ + n * r) > (e * n) / (R₂ + n * r) := by
sorry

end NUMINAMATH_CALUDE_c_decreases_as_r_increases_l2791_279132


namespace NUMINAMATH_CALUDE_unit_digit_15_power_100_l2791_279122

theorem unit_digit_15_power_100 : ∃ n : ℕ, 15^100 = 10 * n + 5 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_15_power_100_l2791_279122


namespace NUMINAMATH_CALUDE_garden_walkway_area_l2791_279183

/-- Calculates the area of walkways in a garden with vegetable beds -/
theorem garden_walkway_area (rows : Nat) (cols : Nat) (bed_width : Nat) (bed_height : Nat) (walkway_width : Nat) : 
  rows = 4 → cols = 3 → bed_width = 8 → bed_height = 3 → walkway_width = 2 →
  (rows * cols * bed_width * bed_height + 
   (rows + 1) * walkway_width * (cols * bed_width + (cols + 1) * walkway_width) + 
   rows * (cols + 1) * walkway_width * bed_height) - 
  (rows * cols * bed_width * bed_height) = 416 := by
sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l2791_279183


namespace NUMINAMATH_CALUDE_five_numbers_product_1000_l2791_279159

theorem five_numbers_product_1000 :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
                     c ≠ d ∧ c ≠ e ∧
                     d ≠ e ∧
                     a * b * c * d * e = 1000 :=
by sorry

end NUMINAMATH_CALUDE_five_numbers_product_1000_l2791_279159


namespace NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l2791_279182

/-- Represents an irregular pentagonal picture frame with given side lengths -/
structure IrregularPentagon where
  base : ℝ
  leftSide : ℝ
  rightSide : ℝ
  topLeftDiagonal : ℝ
  topRightDiagonal : ℝ

/-- Calculates the perimeter of an irregular pentagonal picture frame -/
def perimeter (p : IrregularPentagon) : ℝ :=
  p.base + p.leftSide + p.rightSide + p.topLeftDiagonal + p.topRightDiagonal

/-- Mrs. Hilt's irregular pentagonal picture frame -/
def mrsHiltFrame : IrregularPentagon :=
  { base := 10
    leftSide := 12
    rightSide := 11
    topLeftDiagonal := 6
    topRightDiagonal := 7 }

/-- Theorem: The perimeter of Mrs. Hilt's irregular pentagonal picture frame is 46 inches -/
theorem mrsHiltFramePerimeter : perimeter mrsHiltFrame = 46 := by
  sorry

end NUMINAMATH_CALUDE_mrsHiltFramePerimeter_l2791_279182


namespace NUMINAMATH_CALUDE_parabola_and_line_theorem_l2791_279128

/-- A parabola with focus F and point A on it -/
structure Parabola where
  p : ℝ
  m : ℝ
  h : p > 0

/-- The distance from point A to the focus F is 5 -/
def distance_condition (par : Parabola) : Prop :=
  4 + par.p / 2 = 5

/-- Point A lies on the parabola -/
def point_on_parabola (par : Parabola) : Prop :=
  par.m^2 = 2 * par.p * 4

/-- m is positive -/
def m_positive (par : Parabola) : Prop :=
  par.m > 0

/-- A line that passes through point A -/
structure Line where
  k : ℝ
  b : ℝ

/-- The line intersects the parabola at exactly one point -/
def line_intersects_once (par : Parabola) (l : Line) : Prop :=
  (∀ x y, y = l.k * x + l.b → y^2 = 4 * x) →
  (∃! x, (l.k * x + l.b)^2 = 4 * x)

theorem parabola_and_line_theorem (par : Parabola) 
  (h1 : distance_condition par)
  (h2 : point_on_parabola par)
  (h3 : m_positive par) :
  (par.p = 2 ∧ par.m = 4) ∧
  (∃ l1 l2 : Line, 
    (l1.k = -2 ∧ l1.b = 4 ∧ line_intersects_once par l1) ∧
    (l2.k = 0 ∧ l2.b = 4 ∧ line_intersects_once par l2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_and_line_theorem_l2791_279128


namespace NUMINAMATH_CALUDE_egg_distribution_l2791_279189

theorem egg_distribution (a : ℚ) : a = 7 ↔
  (a / 2 - 1 / 2) / 2 - 1 / 2 - ((a / 4 - 3 / 4) / 2 + 1 / 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_egg_distribution_l2791_279189


namespace NUMINAMATH_CALUDE_height_comparison_l2791_279138

theorem height_comparison (h_a h_b h_c : ℝ) 
  (h_a_def : h_a = 0.6 * h_b) 
  (h_c_def : h_c = 1.25 * h_a) : 
  (h_b - h_a) / h_a = 2/3 ∧ (h_c - h_a) / h_a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_height_comparison_l2791_279138


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l2791_279125

/-- Given a hexagon with internal angles in the ratio 2:3:3:4:5:6, 
    the measure of the largest angle is 4320°/23. -/
theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 5 / 2 →
  f / a = 3 →
  a + b + c + d + e + f = 720 →
  f = 4320 / 23 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l2791_279125


namespace NUMINAMATH_CALUDE_find_a_l2791_279198

theorem find_a : ∃ a : ℕ, 
  (∀ K : ℤ, K ≠ 27 → ∃ m : ℤ, a - K^3 = m * (27 - K)) → 
  a = 3^9 := by
sorry

end NUMINAMATH_CALUDE_find_a_l2791_279198


namespace NUMINAMATH_CALUDE_marble_remainder_l2791_279165

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 7 → (r + p) % 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l2791_279165


namespace NUMINAMATH_CALUDE_petyas_journey_fraction_l2791_279110

/-- The fraction of the journey Petya completed before remembering his pen -/
def journey_fraction (total_time walking_time early_arrival late_arrival : ℚ) : ℚ :=
  walking_time / total_time

theorem petyas_journey_fraction :
  let total_time : ℚ := 20
  let early_arrival : ℚ := 3
  let late_arrival : ℚ := 7
  ∃ (walking_time : ℚ),
    journey_fraction total_time walking_time early_arrival late_arrival = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_petyas_journey_fraction_l2791_279110


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_m_value_when_intersection_given_l2791_279143

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Part 2
theorem m_value_when_intersection_given :
  A ∩ B m = {x | -1 < x ∧ x < 4} → m = 8 :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_m_value_when_intersection_given_l2791_279143


namespace NUMINAMATH_CALUDE_group_size_calculation_l2791_279131

theorem group_size_calculation (n : ℕ) : 
  (15 * n + 35) / (n + 1) = 17 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l2791_279131


namespace NUMINAMATH_CALUDE_at_op_difference_l2791_279114

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 2 * x

-- State the theorem
theorem at_op_difference : (at_op 5 3) - (at_op 3 5) = -4 := by
  sorry

end NUMINAMATH_CALUDE_at_op_difference_l2791_279114


namespace NUMINAMATH_CALUDE_pages_in_book_l2791_279155

/-- 
Given a person who reads a fixed number of pages per day and finishes a book in a certain number of days,
this theorem proves the total number of pages in the book.
-/
theorem pages_in_book (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_day = 8 → days_to_finish = 12 → pages_per_day * days_to_finish = 96 := by
  sorry

end NUMINAMATH_CALUDE_pages_in_book_l2791_279155


namespace NUMINAMATH_CALUDE_king_midas_gold_l2791_279166

theorem king_midas_gold (x : ℝ) (h : x > 1) : 
  let initial_gold := 1
  let spent_fraction := 1 / x
  let remaining_gold := initial_gold - spent_fraction * initial_gold
  let needed_fraction := (initial_gold - remaining_gold) / remaining_gold
  needed_fraction = 1 / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_king_midas_gold_l2791_279166


namespace NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2791_279102

theorem smallest_factorization_coefficient : 
  ∃ (c : ℕ), c > 0 ∧ 
  (∃ (r s : ℤ), x^2 + c*x + 2016 = (x + r) * (x + s)) ∧ 
  (∀ (c' : ℕ), 0 < c' ∧ c' < c → 
    ¬∃ (r' s' : ℤ), x^2 + c'*x + 2016 = (x + r') * (x + s')) ∧
  c = 108 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorization_coefficient_l2791_279102


namespace NUMINAMATH_CALUDE_books_per_bookshelf_l2791_279148

theorem books_per_bookshelf 
  (total_books : ℕ) 
  (num_bookshelves : ℕ) 
  (h1 : total_books = 38) 
  (h2 : num_bookshelves = 19) 
  (h3 : num_bookshelves > 0) :
  total_books / num_bookshelves = 2 := by
  sorry

end NUMINAMATH_CALUDE_books_per_bookshelf_l2791_279148


namespace NUMINAMATH_CALUDE_arithmetic_mean_inequality_negative_l2791_279184

theorem arithmetic_mean_inequality_negative (m n : ℝ) (h1 : m < n) (h2 : n < 0) : n / m + m / n > 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_inequality_negative_l2791_279184


namespace NUMINAMATH_CALUDE_slope_interpretation_l2791_279176

/-- Regression line equation for poverty and education data -/
def regression_line (x : ℝ) : ℝ := 0.8 * x + 4.6

/-- Theorem stating the relationship between changes in x and y -/
theorem slope_interpretation (x₁ x₂ : ℝ) (h : x₂ = x₁ + 1) :
  regression_line x₂ - regression_line x₁ = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_slope_interpretation_l2791_279176


namespace NUMINAMATH_CALUDE_fourth_term_is_two_l2791_279149

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q
  a6_eq_2 : a 6 = 2
  arithmetic_subseq : a 7 - a 5 = a 9 - a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_two_l2791_279149


namespace NUMINAMATH_CALUDE_equation_solution_l2791_279181

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (x + 2 = 1 / (x - 2)) ↔ (x = Real.sqrt 5 ∨ x = -Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2791_279181


namespace NUMINAMATH_CALUDE_father_age_triple_weiwei_age_l2791_279156

/-- Weiwei's current age in years -/
def weiwei_age : ℕ := 8

/-- Weiwei's father's current age in years -/
def father_age : ℕ := 34

/-- The number of years after which the father's age will be three times Weiwei's age -/
def years_until_triple : ℕ := 5

theorem father_age_triple_weiwei_age :
  father_age + years_until_triple = 3 * (weiwei_age + years_until_triple) :=
sorry

end NUMINAMATH_CALUDE_father_age_triple_weiwei_age_l2791_279156


namespace NUMINAMATH_CALUDE_lucky_5n_is_52000_l2791_279197

/-- A natural number is lucky if the sum of its digits is 7 -/
def isLucky (n : ℕ) : Prop :=
  (n.digits 10).sum = 7

/-- The sequence of lucky numbers in increasing order -/
def luckySeq : ℕ → ℕ := sorry

/-- The nth element of the lucky number sequence is 2005 -/
axiom nth_lucky_is_2005 (n : ℕ) : luckySeq n = 2005

theorem lucky_5n_is_52000 (n : ℕ) : luckySeq (5 * n) = 52000 :=
  sorry

end NUMINAMATH_CALUDE_lucky_5n_is_52000_l2791_279197


namespace NUMINAMATH_CALUDE_distance_point_to_line_polar_l2791_279190

/-- The distance from a point in polar coordinates to a line in polar form -/
theorem distance_point_to_line_polar (ρ_A : ℝ) (θ_A : ℝ) (k : ℝ) :
  let l : ℝ × ℝ → Prop := λ (ρ, θ) ↦ 2 * ρ * Real.sin (θ - π/4) = Real.sqrt 2
  let A : ℝ × ℝ := (ρ_A * Real.cos θ_A, ρ_A * Real.sin θ_A)
  let d := abs (A.1 - A.2 + 1) / Real.sqrt 2
  ρ_A = 2 * Real.sqrt 2 ∧ θ_A = 7 * π / 4 → d = 5 * Real.sqrt 2 / 2 :=
by sorry


end NUMINAMATH_CALUDE_distance_point_to_line_polar_l2791_279190


namespace NUMINAMATH_CALUDE_no_rain_probability_l2791_279192

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2791_279192


namespace NUMINAMATH_CALUDE_x_value_when_s_reaches_15000_l2791_279144

/-- The function that calculates S for a given n -/
def S (n : ℕ) : ℕ := n * (n + 3)

/-- The function that calculates X for a given n -/
def X (n : ℕ) : ℕ := 4 + 2 * (n - 1)

/-- The theorem to prove -/
theorem x_value_when_s_reaches_15000 :
  ∃ n : ℕ, S n ≥ 15000 ∧ ∀ m : ℕ, m < n → S m < 15000 ∧ X n = 244 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_s_reaches_15000_l2791_279144


namespace NUMINAMATH_CALUDE_triangle_area_l2791_279164

theorem triangle_area (A B C : ℝ) (a b c S : ℝ) : 
  A = π/3 → 
  a = Real.sqrt 3 → 
  c = 1 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  S = (1/2) * a * c * Real.sin B →
  S = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2791_279164


namespace NUMINAMATH_CALUDE_division_problem_l2791_279191

theorem division_problem : ∃ A : ℕ, 23 = 6 * A + 5 ∧ A = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2791_279191


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2791_279173

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The point (3, -2) is in the fourth quadrant of the Cartesian coordinate system -/
theorem point_in_fourth_quadrant : is_in_fourth_quadrant 3 (-2) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2791_279173


namespace NUMINAMATH_CALUDE_arithmetic_sequence_forms_straight_line_l2791_279112

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_forms_straight_line
  (a : ℕ → ℝ) (h : isArithmeticSequence a) :
  ∃ m b : ℝ, ∀ n : ℕ, a n = m * n + b :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_forms_straight_line_l2791_279112


namespace NUMINAMATH_CALUDE_teresa_jogging_distance_l2791_279178

def speed : ℝ := 5
def time : ℝ := 5
def distance : ℝ := speed * time

theorem teresa_jogging_distance : distance = 25 := by
  sorry

end NUMINAMATH_CALUDE_teresa_jogging_distance_l2791_279178


namespace NUMINAMATH_CALUDE_diet_soda_bottles_l2791_279124

theorem diet_soda_bottles (total : ℕ) (regular : ℕ) (h1 : total = 30) (h2 : regular = 28) :
  total - regular = 2 := by
  sorry

end NUMINAMATH_CALUDE_diet_soda_bottles_l2791_279124


namespace NUMINAMATH_CALUDE_integral_f_minus_pi_to_zero_l2791_279139

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_f_minus_pi_to_zero :
  ∫ x in Set.Icc (-Real.pi) 0, f x = -2 - (1/2) * Real.pi^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_minus_pi_to_zero_l2791_279139


namespace NUMINAMATH_CALUDE_fish_to_buy_l2791_279106

def current_fish : ℕ := 212
def desired_total : ℕ := 280

theorem fish_to_buy : desired_total - current_fish = 68 := by sorry

end NUMINAMATH_CALUDE_fish_to_buy_l2791_279106


namespace NUMINAMATH_CALUDE_largest_value_l2791_279126

theorem largest_value (a b c d e : ℕ) : 
  a = 3 + 1 + 2 + 4 →
  b = 3 * 1 + 2 + 4 →
  c = 3 + 1 * 2 + 4 →
  d = 3 + 1 + 2 * 4 →
  e = 3 * 1 * 2 * 4 →
  e ≥ a ∧ e ≥ b ∧ e ≥ c ∧ e ≥ d :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l2791_279126


namespace NUMINAMATH_CALUDE_baseball_hits_theorem_l2791_279109

theorem baseball_hits_theorem (total_hits home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 10) :
  let singles := total_hits - (home_runs + triples + doubles)
  let percentage := (singles : ℚ) / total_hits * 100
  singles = 35 ∧ percentage = 70 := by
sorry

end NUMINAMATH_CALUDE_baseball_hits_theorem_l2791_279109


namespace NUMINAMATH_CALUDE_expression_evaluation_l2791_279136

theorem expression_evaluation (x y : ℚ) 
  (hx : x = 2 / 15) (hy : y = 3 / 2) : 
  (2 * x + y)^2 - (3 * x - y)^2 + 5 * x * (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2791_279136


namespace NUMINAMATH_CALUDE_portrait_ratio_l2791_279133

/-- Prove that the ratio of students who had portraits taken before lunch
    to the total number of students is 1:3 -/
theorem portrait_ratio :
  ∀ (before_lunch after_lunch not_taken : ℕ),
  before_lunch + after_lunch + not_taken = 24 →
  after_lunch = 10 →
  not_taken = 6 →
  (before_lunch : ℚ) / 24 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_portrait_ratio_l2791_279133


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_is_correct_l2791_279167

/-- Percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- Percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- Percentage of Type A defects that are reworked and repaired -/
def type_a_rework_rate : ℝ := 0.40

/-- Percentage of Type B defects that are reworked and repaired -/
def type_b_rework_rate : ℝ := 0.30

/-- Percentage of remaining Type A defects that are shipped -/
def type_a_ship_rate : ℝ := 0.03

/-- Percentage of remaining Type B defects that are shipped -/
def type_b_ship_rate : ℝ := 0.06

/-- The percentage of defective units (Type A or B) shipped for sale -/
def defective_shipped_percentage : ℝ :=
  type_a_defect_rate * (1 - type_a_rework_rate) * type_a_ship_rate +
  type_b_defect_rate * (1 - type_b_rework_rate) * type_b_ship_rate

theorem defective_shipped_percentage_is_correct :
  defective_shipped_percentage = 0.00462 := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_is_correct_l2791_279167


namespace NUMINAMATH_CALUDE_imaginary_part_sum_of_complex_fractions_l2791_279147

theorem imaginary_part_sum_of_complex_fractions :
  Complex.im (1 / Complex.ofReal (-2) + Complex.I + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_of_complex_fractions_l2791_279147


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l2791_279171

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l2791_279171


namespace NUMINAMATH_CALUDE_fraction_simplification_l2791_279113

/-- 
For any integer n, the fraction (5n+3)/(7n+8) can be simplified by 5 
if and only if n is divisible by 5 or n is of the form 19k + 7 for some integer k.
-/
theorem fraction_simplification (n : ℤ) : 
  (∃ (m : ℤ), 5 * (7*n + 8) = 7 * (5*n + 3) * m) ↔ 
  (∃ (k : ℤ), n = 5*k ∨ n = 19*k + 7) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2791_279113


namespace NUMINAMATH_CALUDE_two_number_problem_l2791_279174

theorem two_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x < y) 
  (h4 : 4 * y = 6 * x) (h5 : x + y = 36) : y = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_two_number_problem_l2791_279174


namespace NUMINAMATH_CALUDE_three_blocks_selection_count_l2791_279129

-- Define the size of the grid
def grid_size : ℕ := 5

-- Define the number of blocks to select
def blocks_to_select : ℕ := 3

-- Theorem statement
theorem three_blocks_selection_count :
  (grid_size.choose blocks_to_select) * (grid_size.choose blocks_to_select) * (blocks_to_select.factorial) = 600 := by
  sorry

end NUMINAMATH_CALUDE_three_blocks_selection_count_l2791_279129


namespace NUMINAMATH_CALUDE_inequality_proof_l2791_279150

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2 * x * y) / (x + y) + Real.sqrt ((x^2 + y^2) / 2) ≥ (x + y) / 2 + Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2791_279150


namespace NUMINAMATH_CALUDE_product_inequality_find_a_l2791_279105

-- Part I
theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 + 1/a) * (1 + 1/b) ≥ 9 := by sorry

-- Part II
theorem find_a (a : ℝ) (h : ∀ x, |x + 3| - |x - a| ≥ 2 ↔ x ≥ 1) :
  a = 2 := by sorry

end NUMINAMATH_CALUDE_product_inequality_find_a_l2791_279105


namespace NUMINAMATH_CALUDE_b_spend_percent_calculation_l2791_279188

def combined_salary : ℝ := 3000
def a_salary : ℝ := 2250
def a_spend_percent : ℝ := 0.95

theorem b_spend_percent_calculation :
  let b_salary := combined_salary - a_salary
  let a_savings := a_salary * (1 - a_spend_percent)
  let b_spend_percent := 1 - (a_savings / b_salary)
  b_spend_percent = 0.85 := by sorry

end NUMINAMATH_CALUDE_b_spend_percent_calculation_l2791_279188


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_PQR_l2791_279195

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (P Q R : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle PQR is (3/2, 13/2, 5) -/
theorem orthocenter_of_triangle_PQR :
  let P : ℝ × ℝ × ℝ := (2, 3, 4)
  let Q : ℝ × ℝ × ℝ := (6, 4, 2)
  let R : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter P Q R = (3/2, 13/2, 5) :=
by
  sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_PQR_l2791_279195


namespace NUMINAMATH_CALUDE_samantha_bus_time_l2791_279103

def time_to_minutes (hours minutes : ℕ) : ℕ := hours * 60 + minutes

def samantha_schedule : Prop :=
  let leave_home := time_to_minutes 7 0
  let catch_bus := time_to_minutes 7 45
  let arrive_home := time_to_minutes 17 15
  let class_duration := 55
  let num_classes := 8
  let other_activities := time_to_minutes 1 45
  let total_away_time := arrive_home - leave_home
  let total_school_time := num_classes * class_duration + other_activities
  let bus_time := total_away_time - total_school_time
  bus_time = 25

theorem samantha_bus_time : samantha_schedule := by sorry

end NUMINAMATH_CALUDE_samantha_bus_time_l2791_279103


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2791_279137

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the units digit of a two-digit number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A predicate that checks if a number is two-digit -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if the product of digits of a number is 8 -/
def productOfDigitsIs8 (n : ℕ) : Prop := tensDigit n * unitsDigit n = 8

/-- A predicate that checks if adding 18 to a number reverses its digits -/
def adding18ReversesDigits (n : ℕ) : Prop := 
  tensDigit (n + 18) = unitsDigit n ∧ unitsDigit (n + 18) = tensDigit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, isTwoDigit n ∧ productOfDigitsIs8 n ∧ adding18ReversesDigits n ∧ n = 24 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2791_279137
