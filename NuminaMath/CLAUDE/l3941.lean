import Mathlib

namespace median_equation_l3941_394137

/-- The equation of median BD in triangle ABC -/
theorem median_equation (A B C D : ℝ × ℝ) : 
  A = (4, 1) → B = (0, 3) → C = (2, 4) → D = ((A.1 + C.1)/2, (A.2 + C.2)/2) →
  (fun (x y : ℝ) => x + 6*y - 18) = (fun (x y : ℝ) => 0) := by sorry

end median_equation_l3941_394137


namespace class_size_proof_l3941_394170

/-- Represents the number of students who like art -/
def art_students : ℕ := 35

/-- Represents the number of students who like music -/
def music_students : ℕ := 32

/-- Represents the number of students who like both art and music -/
def both_students : ℕ := 19

/-- Represents the total number of students in the class -/
def total_students : ℕ := art_students + music_students - both_students

theorem class_size_proof :
  total_students = 48 :=
sorry

end class_size_proof_l3941_394170


namespace not_prime_base_n_2022_l3941_394101

-- Define the base-n representation of 2022
def base_n_2022 (n : ℕ) : ℕ := 2 * n^3 + 2 * n + 2

-- Theorem statement
theorem not_prime_base_n_2022 (n : ℕ) (h : n ≥ 2) : ¬ Nat.Prime (base_n_2022 n) := by
  sorry

end not_prime_base_n_2022_l3941_394101


namespace no_geometric_sequence_cosines_l3941_394156

theorem no_geometric_sequence_cosines :
  ¬ ∃ a : ℝ, 0 < a ∧ a < 2 * π ∧
    ∃ r : ℝ, (Real.cos (2 * a) = r * Real.cos a) ∧
             (Real.cos (3 * a) = r * Real.cos (2 * a)) :=
by sorry

end no_geometric_sequence_cosines_l3941_394156


namespace arithmetic_geometric_harmonic_inequality_l3941_394173

theorem arithmetic_geometric_harmonic_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a + b) / 2 > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > 2 * a * b / (a + b) := by
  sorry

end arithmetic_geometric_harmonic_inequality_l3941_394173


namespace min_voters_for_tall_to_win_l3941_394179

/-- Represents the voting structure and rules of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat
  (total_voters_eq : total_voters = num_districts * precincts_per_district * voters_per_precinct)

/-- Calculates the minimum number of voters required to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  let min_districts_to_win := contest.num_districts / 2 + 1
  let min_precincts_to_win := contest.precincts_per_district / 2 + 1
  let min_votes_per_precinct := contest.voters_per_precinct / 2 + 1
  min_districts_to_win * min_precincts_to_win * min_votes_per_precinct

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem min_voters_for_tall_to_win (contest : GiraffeContest)
  (h1 : contest.total_voters = 135)
  (h2 : contest.num_districts = 5)
  (h3 : contest.precincts_per_district = 9)
  (h4 : contest.voters_per_precinct = 3) :
  min_voters_to_win contest = 30 := by
  sorry

#eval min_voters_to_win { total_voters := 135, num_districts := 5, precincts_per_district := 9, voters_per_precinct := 3, total_voters_eq := rfl }

end min_voters_for_tall_to_win_l3941_394179


namespace smallest_upper_bound_l3941_394100

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 0 < x ∧ x < 7)
  (h2 : -1 < x ∧ x < 5)
  (h3 : 0 < x ∧ x < 3)
  (h4 : x + 2 < 4) :
  ∀ y : ℤ, (0 < x ∧ x < y) → y ≥ 2 :=
by sorry

end smallest_upper_bound_l3941_394100


namespace complex_sequence_sum_l3941_394109

theorem complex_sequence_sum (a b : ℕ → ℝ) :
  (∀ n : ℕ, (Complex.I + 2) ^ n = Complex.mk (a n) (b n)) →
  (∑' n, (a n * b n) / (7 : ℝ) ^ n) = 7 / 16 := by
sorry

end complex_sequence_sum_l3941_394109


namespace chess_pieces_remaining_l3941_394193

theorem chess_pieces_remaining (initial_pieces : ℕ) (scarlett_lost : ℕ) (hannah_lost : ℕ)
  (h1 : initial_pieces = 32)
  (h2 : scarlett_lost = 6)
  (h3 : hannah_lost = 8) :
  initial_pieces - (scarlett_lost + hannah_lost) = 18 :=
by sorry

end chess_pieces_remaining_l3941_394193


namespace min_value_theorem_min_value_achievable_l3941_394197

theorem min_value_theorem (x : ℝ) (h : x > 0) : 9 * x^2 + 1 / x^6 ≥ 6 * Real.sqrt 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, x > 0 ∧ 9 * x^2 + 1 / x^6 = 6 * Real.sqrt 3 := by
  sorry

end min_value_theorem_min_value_achievable_l3941_394197


namespace bankers_gain_is_nine_l3941_394175

/-- Calculates the banker's gain given the true discount, time period, and interest rate. -/
def bankers_gain (true_discount : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let face_value := (true_discount * (100 + (rate * time))) / (rate * time)
  let bankers_discount := (face_value * rate * time) / 100
  bankers_discount - true_discount

/-- Theorem stating that the banker's gain is 9 given the specified conditions. -/
theorem bankers_gain_is_nine :
  bankers_gain 75 1 12 = 9 := by
  sorry

#eval bankers_gain 75 1 12

end bankers_gain_is_nine_l3941_394175


namespace line_segment_param_sum_l3941_394121

/-- Given a line segment connecting (1, -3) and (-4, 5), parameterized by x = pt + q and y = rt + s
    where 0 ≤ t ≤ 2 and t = 0 corresponds to (1, -3), prove that p^2 + q^2 + r^2 + s^2 = 32.25 -/
theorem line_segment_param_sum (p q r s : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 → p * t + q = 1 - 5 * t / 2 ∧ r * t + s = -3 + 4 * t) →
  p^2 + q^2 + r^2 + s^2 = 129/4 := by
sorry

end line_segment_param_sum_l3941_394121


namespace brocard_angle_inequalities_l3941_394106

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle_angles : α + β + γ = π) 
  (brocard_angle : 0 < φ ∧ φ ≤ π/6)
  (brocard_identity : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end brocard_angle_inequalities_l3941_394106


namespace goose_egg_problem_l3941_394176

-- Define the total number of goose eggs laid
variable (E : ℕ)

-- Define the conditions
axiom hatch_ratio : (1 : ℚ) / 4 * E = (E / 4 : ℕ)
axiom first_month_survival : (4 : ℚ) / 5 * (E / 4 : ℕ) = (4 * E / 20 : ℕ)
axiom first_year_survival : (4 * E / 20 : ℕ) = 120

-- Define the theorem
theorem goose_egg_problem :
  E = 2400 ∧ ((4 * E / 20 : ℕ) - 120 : ℚ) / (4 * E / 20 : ℕ) = 3 / 4 := by
  sorry


end goose_egg_problem_l3941_394176


namespace max_consecutive_integers_l3941_394157

def consecutive_sum (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1) / 2

def is_valid_sequence (start : ℕ) (n : ℕ) : Prop :=
  consecutive_sum start n = 2014 ∧ start > 0

theorem max_consecutive_integers :
  (∃ (start : ℕ), is_valid_sequence start 53) ∧
  (∀ (m : ℕ) (start : ℕ), m > 53 → ¬ is_valid_sequence start m) :=
sorry

end max_consecutive_integers_l3941_394157


namespace unique_rope_triangle_l3941_394161

/-- An isosceles triangle formed from a rope --/
structure RopeTriangle where
  total_length : ℝ
  base_length : ℝ
  side_length : ℝ
  is_isosceles : side_length = (total_length - base_length) / 2
  is_triangle : base_length + 2 * side_length = total_length

/-- The specific rope triangle from the problem --/
def problem_triangle : RopeTriangle where
  total_length := 24
  base_length := 6
  side_length := 9
  is_isosceles := by sorry
  is_triangle := by sorry

/-- Theorem stating that the problem_triangle is the unique solution --/
theorem unique_rope_triangle :
  ∀ (t : RopeTriangle), t.total_length = 24 ∧ t.base_length = 6 → t = problem_triangle :=
by sorry

end unique_rope_triangle_l3941_394161


namespace luke_laundry_problem_l3941_394126

/-- Given a total number of clothing pieces, the number of pieces in the first load,
    and the number of remaining loads, calculate the number of pieces in each small load. -/
def pieces_per_small_load (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) : ℕ :=
  (total - first_load) / num_small_loads

/-- Theorem stating that given the specific conditions of the problem,
    the number of pieces in each small load is 10. -/
theorem luke_laundry_problem :
  pieces_per_small_load 105 34 7 = 10 := by
  sorry

end luke_laundry_problem_l3941_394126


namespace green_apples_count_l3941_394194

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := sorry

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := 6

/-- The number of students who took fruit -/
def students_taking_fruit : ℕ := 5

/-- The number of extra apples left over -/
def extra_apples : ℕ := 16

/-- Theorem stating that the number of green apples ordered is 15 -/
theorem green_apples_count : green_apples = 15 := by
  sorry

end green_apples_count_l3941_394194


namespace trapezoid_segment_length_l3941_394133

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a trapezoid formed after removing segments from a rectangle -/
structure Trapezoid where
  longBase : ℝ
  shortBase : ℝ
  height : ℝ

/-- Calculates the total length of segments in the trapezoid -/
def totalLength (t : Trapezoid) : ℝ :=
  t.longBase + t.shortBase + t.height

/-- Theorem stating that the total length of segments in the resulting trapezoid is 19 units -/
theorem trapezoid_segment_length 
  (r : Rectangle)
  (t : Trapezoid)
  (h1 : r.length = 11)
  (h2 : r.width = 3)
  (h3 : r.height = 12)
  (h4 : t.longBase = 8)
  (h5 : t.shortBase = r.width)
  (h6 : t.height = r.height - 4) :
  totalLength t = 19 := by
    sorry


end trapezoid_segment_length_l3941_394133


namespace unique_solution_l3941_394180

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^2*y + x*y^2 - 2*x - 2*y + 10 = 0
def equation2 (x y : ℝ) : Prop := x^3*y - x*y^3 - 2*x^2 + 2*y^2 - 30 = 0

-- State the theorem
theorem unique_solution :
  ∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2 ∧ p = (-4, -1) := by
  sorry

end unique_solution_l3941_394180


namespace smallest_possible_value_l3941_394130

theorem smallest_possible_value (m n x : ℕ+) : 
  m = 60 →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = 2 * x.val * (x.val + 5) →
  (∀ n' : ℕ+, n'.val < n.val → 
    (Nat.gcd m.val n'.val ≠ x.val + 5 ∨ 
     Nat.lcm m.val n'.val ≠ 2 * x.val * (x.val + 5))) →
  n.val = 75 :=
by sorry

end smallest_possible_value_l3941_394130


namespace two_numbers_lcm_90_gcd_6_l3941_394141

theorem two_numbers_lcm_90_gcd_6 : ∃ (a b : ℕ+), 
  (¬(a ∣ b) ∧ ¬(b ∣ a)) ∧ 
  Nat.lcm a b = 90 ∧ 
  Nat.gcd a b = 6 ∧ 
  ({a, b} : Set ℕ+) = {18, 30} := by
sorry

end two_numbers_lcm_90_gcd_6_l3941_394141


namespace solve_system_l3941_394186

theorem solve_system (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 2 / x) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 :=
by sorry

end solve_system_l3941_394186


namespace rectangle_area_preservation_l3941_394104

theorem rectangle_area_preservation (original_length original_width : ℝ) 
  (h_length : original_length = 280)
  (h_width : original_width = 80)
  (length_increase_percent : ℝ) 
  (h_increase : length_increase_percent = 60) : 
  let new_length := original_length * (1 + length_increase_percent / 100)
  let new_width := (original_length * original_width) / new_length
  let width_decrease_percent := (original_width - new_width) / original_width * 100
  width_decrease_percent = 37.5 := by
sorry

end rectangle_area_preservation_l3941_394104


namespace parents_present_l3941_394150

theorem parents_present (total_people pupils teachers : ℕ) 
  (h1 : total_people = 1541)
  (h2 : pupils = 724)
  (h3 : teachers = 744) :
  total_people - (pupils + teachers) = 73 := by
  sorry

end parents_present_l3941_394150


namespace paco_cookies_theorem_l3941_394124

/-- Calculates the number of sweet cookies Paco ate given the initial quantities and eating conditions -/
def sweet_cookies_eaten (initial_sweet : ℕ) (initial_salty : ℕ) (sweet_salty_difference : ℕ) : ℕ :=
  initial_salty + sweet_salty_difference

theorem paco_cookies_theorem (initial_sweet initial_salty sweet_salty_difference : ℕ) 
  (h1 : initial_sweet = 39)
  (h2 : initial_salty = 6)
  (h3 : sweet_salty_difference = 9) :
  sweet_cookies_eaten initial_sweet initial_salty sweet_salty_difference = 15 := by
  sorry

#eval sweet_cookies_eaten 39 6 9

end paco_cookies_theorem_l3941_394124


namespace rectangle_area_problem_l3941_394131

theorem rectangle_area_problem : ∃ (x y : ℝ), 
  (x + 3) * (y - 1) = x * y ∧ 
  (x - 3) * (y + 1.5) = x * y ∧ 
  x * y = 31.5 := by
  sorry

end rectangle_area_problem_l3941_394131


namespace least_integer_with_given_remainders_l3941_394154

theorem least_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧
  x % 3 = 2 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 419 :=
by sorry

end least_integer_with_given_remainders_l3941_394154


namespace cos_alpha_plus_pi_sixth_l3941_394181

theorem cos_alpha_plus_pi_sixth (α : ℝ) (h : Real.sin (α - π/3) = 1/3) :
  Real.cos (α + π/6) = -1/3 := by
sorry

end cos_alpha_plus_pi_sixth_l3941_394181


namespace valid_paths_count_l3941_394191

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move direction -/
inductive Move
  | Right
  | Up
  | Diagonal

/-- Defines a valid path on the Cartesian plane -/
def ValidPath (start finish : Point) (path : List Move) : Prop :=
  -- Path starts at the start point and ends at the finish point
  -- Each move is valid according to the problem conditions
  -- No right-angle turns in the path
  sorry

/-- Counts the number of valid paths between two points -/
def CountValidPaths (start finish : Point) : ℕ :=
  sorry

theorem valid_paths_count :
  CountValidPaths (Point.mk 0 0) (Point.mk 6 6) = 128 := by
  sorry

end valid_paths_count_l3941_394191


namespace max_value_cos_sin_l3941_394163

theorem max_value_cos_sin (x : ℝ) : 3 * Real.cos x + 4 * Real.sin x ≤ 5 := by
  sorry

end max_value_cos_sin_l3941_394163


namespace remainder_4873_div_29_l3941_394153

theorem remainder_4873_div_29 : 4873 % 29 = 1 := by
  sorry

end remainder_4873_div_29_l3941_394153


namespace jakes_score_l3941_394102

theorem jakes_score (total_students : Nat) (avg_18 : ℚ) (avg_19 : ℚ) (avg_20 : ℚ) 
  (h1 : total_students = 20)
  (h2 : avg_18 = 75)
  (h3 : avg_19 = 76)
  (h4 : avg_20 = 77) :
  (total_students * avg_20 - (total_students - 1) * avg_19 : ℚ) = 96 := by
  sorry

end jakes_score_l3941_394102


namespace friends_in_all_activities_l3941_394110

theorem friends_in_all_activities (movie : ℕ) (picnic : ℕ) (games : ℕ) 
  (movie_and_picnic : ℕ) (movie_and_games : ℕ) (picnic_and_games : ℕ) 
  (total : ℕ) : 
  movie = 10 → 
  picnic = 20 → 
  games = 5 → 
  movie_and_picnic = 4 → 
  movie_and_games = 2 → 
  picnic_and_games = 0 → 
  total = 31 → 
  ∃ (all_three : ℕ), 
    all_three = 2 ∧ 
    total = movie + picnic + games - movie_and_picnic - movie_and_games - picnic_and_games + all_three :=
by sorry

end friends_in_all_activities_l3941_394110


namespace perimeter_of_rectangle_l3941_394147

def rhombus_in_rectangle (WE XF EG FH : ℝ) : Prop :=
  WE = 10 ∧ XF = 25 ∧ EG = 20 ∧ FH = 50

theorem perimeter_of_rectangle (WE XF EG FH : ℝ) 
  (h : rhombus_in_rectangle WE XF EG FH) : 
  ∃ (perimeter : ℝ), perimeter = 53 * Real.sqrt 29 - 73 := by
  sorry

#check perimeter_of_rectangle

end perimeter_of_rectangle_l3941_394147


namespace basketball_free_throws_l3941_394118

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    2 * (2 * two_points) = 3 * three_points →
    free_throws = 2 * two_points →
    2 * two_points + 3 * three_points + free_throws = 74 →
    free_throws = 30 := by
  sorry

end basketball_free_throws_l3941_394118


namespace intersection_point_coordinates_l3941_394171

/-- A line in 2D space represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if two lines are perpendicular --/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The given line l --/
def l : Line := { a := 2, b := 1, c := 10 }

/-- The point through which l' passes --/
def p : Point := { x := -10, y := 0 }

/-- The theorem to prove --/
theorem intersection_point_coordinates :
  ∃ (l' : Line),
    (p.onLine l') ∧
    (l.perpendicular l') ∧
    (∃ (q : Point), q.onLine l ∧ q.onLine l' ∧ q.x = 2 ∧ q.y = 6) :=
by sorry

end intersection_point_coordinates_l3941_394171


namespace platform_length_l3941_394132

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmph = 72 →
  crossing_time = 20 →
  (train_speed_kmph * (1000 / 3600) * crossing_time) - train_length = 150 := by
  sorry

#check platform_length

end platform_length_l3941_394132


namespace imaginary_part_of_z_l3941_394107

theorem imaginary_part_of_z (z : ℂ) : z = (Complex.I - 3) / (Complex.I + 1) → z.im = 2 := by
  sorry

end imaginary_part_of_z_l3941_394107


namespace snow_total_l3941_394165

theorem snow_total (monday_snow tuesday_snow : Real) 
  (h1 : monday_snow = 0.32)
  (h2 : tuesday_snow = 0.21) : 
  monday_snow + tuesday_snow = 0.53 := by
sorry

end snow_total_l3941_394165


namespace eighth_grade_students_l3941_394172

theorem eighth_grade_students (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 68 → 
  girls = 28 → 
  boys < 2 * girls →
  boys = total - girls →
  2 * girls - boys = 16 :=
by
  sorry

end eighth_grade_students_l3941_394172


namespace number_of_children_l3941_394120

/-- Given a group of children born at 2-year intervals with the youngest being 6 years old,
    and the sum of their ages being 50 years, prove that there are 5 children. -/
theorem number_of_children (sum_of_ages : ℕ) (age_difference : ℕ) (youngest_age : ℕ) 
  (h1 : sum_of_ages = 50)
  (h2 : age_difference = 2)
  (h3 : youngest_age = 6) :
  ∃ (n : ℕ), n = 5 ∧ 
  sum_of_ages = n * (youngest_age + (n - 1) * age_difference / 2) := by
  sorry

end number_of_children_l3941_394120


namespace quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l3941_394119

/-- Given two quadratic equations, prove the relationship between their coefficients -/
theorem quadratic_coefficient_relation (a b c d r s : ℝ) : 
  (r + s = -a ∧ r * s = b) →  -- roots of first equation
  (r^2 + s^2 = -c ∧ r^2 * s^2 = d) →  -- roots of second equation
  r * s = 2 * b →  -- additional condition
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

/-- Alternative formulation using polynomial roots -/
theorem quadratic_coefficient_relation_alt (a b c d : ℝ) :
  (∃ r s : ℝ, (r + s = -a ∧ r * s = b) ∧ 
              (r^2 + s^2 = -c ∧ r^2 * s^2 = d) ∧
              r * s = 2 * b) →
  c = -a^2 + 2*b ∧ d = b^2 := by
sorry

end quadratic_coefficient_relation_quadratic_coefficient_relation_alt_l3941_394119


namespace math_team_combinations_l3941_394167

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 5 → boys = 8 → (girls.choose 1) * ((girls - 1).choose 2) * (boys.choose 2) = 840 := by
  sorry

end math_team_combinations_l3941_394167


namespace only_equilateral_forms_triangle_l3941_394122

/-- A function that checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments -/
def segment_sets : List (ℝ × ℝ × ℝ) :=
  [(3, 4, 8), (5, 6, 11), (4, 4, 8), (8, 8, 8)]

/-- Theorem stating that only (8, 8, 8) can form a triangle among the given sets -/
theorem only_equilateral_forms_triangle :
  ∃! set : ℝ × ℝ × ℝ, set ∈ segment_sets ∧ can_form_triangle set.1 set.2.1 set.2.2 :=
by sorry

end only_equilateral_forms_triangle_l3941_394122


namespace square_of_cube_of_third_smallest_prime_l3941_394178

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem square_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end square_of_cube_of_third_smallest_prime_l3941_394178


namespace quadratic_function_value_l3941_394136

/-- Given a quadratic function f(x) = ax^2 + bx + 2 with f(1) = 4 and f(2) = 10, prove that f(3) = 20 -/
theorem quadratic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + 2)
  (h2 : f 1 = 4)
  (h3 : f 2 = 10) :
  f 3 = 20 := by
  sorry

end quadratic_function_value_l3941_394136


namespace inequality_proof_l3941_394127

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  a^2 + b^2 + c^2 + 3 ≥ 1/a + 1/b + 1/c + a + b + c :=
by sorry

end inequality_proof_l3941_394127


namespace real_part_divisible_by_p_l3941_394174

/-- A Gaussian integer is a complex number with integer real and imaginary parts. -/
structure GaussianInteger where
  re : ℤ
  im : ℤ

/-- The real part of a complex number z^p - z is divisible by p for any Gaussian integer z and odd prime p. -/
theorem real_part_divisible_by_p (z : GaussianInteger) (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (k : ℤ), (z.re^p - z.re : ℤ) = p * k := by
  sorry

end real_part_divisible_by_p_l3941_394174


namespace lattice_points_on_hyperbola_l3941_394162

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 - p.2^2 = 65) ∧ 
    s.card = 4 := by
  sorry

end lattice_points_on_hyperbola_l3941_394162


namespace cube_volume_l3941_394149

/-- Represents a rectangular box with given dimensions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Represents the problem setup -/
def problemSetup : Box × ℕ :=
  (⟨7, 18, 3⟩, 42)

/-- Theorem stating the volume of each cube -/
theorem cube_volume (box : Box) (num_cubes : ℕ) 
  (h1 : box = problemSetup.1) 
  (h2 : num_cubes = problemSetup.2) : 
  (boxVolume box) / num_cubes = 9 := by
  sorry

end cube_volume_l3941_394149


namespace sin_difference_simplification_l3941_394113

theorem sin_difference_simplification (x y : ℝ) : 
  Real.sin (x + y) * Real.cos y - Real.cos (x + y) * Real.sin y = Real.sin x := by
  sorry

end sin_difference_simplification_l3941_394113


namespace quadratic_max_value_l3941_394195

def f (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 2) →
  a = -1 ∨ a = 2 := by
sorry

end quadratic_max_value_l3941_394195


namespace square_sum_fifteen_l3941_394144

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end square_sum_fifteen_l3941_394144


namespace base_8_to_base_10_l3941_394105

theorem base_8_to_base_10 : 
  (3 * 8^3 + 5 * 8^2 + 2 * 8^1 + 6 * 8^0 : ℕ) = 1878 := by
  sorry

end base_8_to_base_10_l3941_394105


namespace largest_prime_factor_of_12321_l3941_394185

theorem largest_prime_factor_of_12321 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 12321 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 12321 → q ≤ p :=
by sorry

end largest_prime_factor_of_12321_l3941_394185


namespace jaewoong_ran_most_l3941_394151

-- Define the athletes and their distances
def jaewoong_distance : ℕ := 20  -- in kilometers
def seongmin_distance : ℕ := 2600  -- in meters
def eunseong_distance : ℕ := 5000  -- in meters

-- Define the conversion factor from kilometers to meters
def km_to_m : ℕ := 1000

-- Theorem to prove Jaewoong ran the most
theorem jaewoong_ran_most :
  (jaewoong_distance * km_to_m > seongmin_distance) ∧
  (jaewoong_distance * km_to_m > eunseong_distance) :=
by
  sorry

#check jaewoong_ran_most

end jaewoong_ran_most_l3941_394151


namespace f_monotone_decreasing_l3941_394112

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_monotone_decreasing :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv f (x + 1) < 0 :=
sorry

end f_monotone_decreasing_l3941_394112


namespace f_eq_g_shifted_l3941_394138

/-- Given two functions f and g defined as follows:
    f(x) = sin(x + π/2)
    g(x) = cos(x - π/2)
    Prove that f(x) = g(x + π/2) for all real x. -/
theorem f_eq_g_shifted (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (x + π/2)
  let g : ℝ → ℝ := λ x => Real.cos (x - π/2)
  f x = g (x + π/2) := by
  sorry

end f_eq_g_shifted_l3941_394138


namespace simplify_and_evaluate_l3941_394190

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 + 2*x + 1) / (x + 1) = Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l3941_394190


namespace salt_solution_mixture_l3941_394158

theorem salt_solution_mixture (x : ℝ) : 
  (1 : ℝ) + x > 0 →  -- Ensure total volume is positive
  0.60 * x = 0.10 * ((1 : ℝ) + x) → 
  x = 0.2 := by
sorry

end salt_solution_mixture_l3941_394158


namespace trig_identity_l3941_394177

theorem trig_identity (α : Real) (h : Real.tan α = 1/2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 := by
  sorry

end trig_identity_l3941_394177


namespace money_difference_is_13_96_l3941_394142

def derek_initial : ℚ := 40
def derek_expenses : List ℚ := [14, 11, 5, 8]
def derek_discount_rate : ℚ := 0.1

def dave_initial : ℚ := 50
def dave_expenses : List ℚ := [7, 12, 9]
def dave_tax_rate : ℚ := 0.08

def calculate_remaining_money (initial : ℚ) (expenses : List ℚ) (rate : ℚ) (is_discount : Bool) : ℚ :=
  let total_expenses := expenses.sum
  let adjustment := total_expenses * rate
  if is_discount then
    initial - (total_expenses - adjustment)
  else
    initial - (total_expenses + adjustment)

theorem money_difference_is_13_96 :
  let derek_remaining := calculate_remaining_money derek_initial derek_expenses derek_discount_rate true
  let dave_remaining := calculate_remaining_money dave_initial dave_expenses dave_tax_rate false
  dave_remaining - derek_remaining = 13.96 := by
  sorry

end money_difference_is_13_96_l3941_394142


namespace days_worked_l3941_394145

/-- Given a person works 8 hours each day and a total of 32 hours, prove that the number of days worked is 4. -/
theorem days_worked (hours_per_day : ℕ) (total_hours : ℕ) (h1 : hours_per_day = 8) (h2 : total_hours = 32) :
  total_hours / hours_per_day = 4 := by
  sorry

end days_worked_l3941_394145


namespace soda_volume_difference_is_14_l3941_394159

/-- Calculates the difference in soda volume between Julio and Mateo -/
def soda_volume_difference : ℕ :=
  let julio_orange := 4
  let julio_grape := 7
  let mateo_orange := 1
  let mateo_grape := 3
  let liters_per_bottle := 2
  let julio_total := (julio_orange + julio_grape) * liters_per_bottle
  let mateo_total := (mateo_orange + mateo_grape) * liters_per_bottle
  julio_total - mateo_total

theorem soda_volume_difference_is_14 : soda_volume_difference = 14 := by
  sorry

end soda_volume_difference_is_14_l3941_394159


namespace fifth_employee_speed_is_140_l3941_394166

/-- Calculates the typing speed of the fifth employee given the team size, average speed, and speeds of four employees --/
def fifth_employee_speed (team_size : ℕ) (average_speed : ℕ) (speed1 speed2 speed3 speed4 : ℕ) : ℕ :=
  team_size * average_speed - speed1 - speed2 - speed3 - speed4

/-- Proves that the fifth employee's typing speed is 140 words per minute --/
theorem fifth_employee_speed_is_140 :
  fifth_employee_speed 5 80 64 76 91 89 = 140 := by
  sorry

end fifth_employee_speed_is_140_l3941_394166


namespace fiftieth_term_of_specific_sequence_l3941_394192

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_specific_sequence :
  arithmetic_sequence 3 2 50 = 101 := by
  sorry

end fiftieth_term_of_specific_sequence_l3941_394192


namespace inequality_holds_l3941_394111

theorem inequality_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end inequality_holds_l3941_394111


namespace track_length_l3941_394143

/-- Represents a circular running track with two runners -/
structure CircularTrack where
  length : ℝ
  initial_distance : ℝ
  first_meeting_distance : ℝ
  second_meeting_additional_distance : ℝ

/-- The track satisfies the given conditions -/
def satisfies_conditions (track : CircularTrack) : Prop :=
  track.initial_distance = 120 ∧
  track.first_meeting_distance = 150 ∧
  track.second_meeting_additional_distance = 200

/-- The theorem stating the length of the track -/
theorem track_length (track : CircularTrack) 
  (h : satisfies_conditions track) : track.length = 450 := by
  sorry

end track_length_l3941_394143


namespace counterexample_square_inequality_l3941_394152

theorem counterexample_square_inequality : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end counterexample_square_inequality_l3941_394152


namespace line_parameterization_l3941_394125

/-- Given a line y = 2x - 17 parameterized by (x,y) = (f(t), 20t - 12), prove that f(t) = 10t + 5/2 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20*t - 12 = 2*(f t) - 17) → 
  (∀ t : ℝ, f t = 10*t + 5/2) := by
sorry

end line_parameterization_l3941_394125


namespace number_equation_solution_l3941_394168

theorem number_equation_solution : 
  ∃ x : ℝ, x - (1002 / 20.04) = 3500 ∧ x = 3550 := by
  sorry

end number_equation_solution_l3941_394168


namespace common_difference_from_terms_l3941_394139

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The common difference of an arithmetic sequence -/
def commonDifference (seq : ArithmeticSequence) : ℝ :=
  seq.a 1 - seq.a 0

theorem common_difference_from_terms
  (seq : ArithmeticSequence)
  (h5 : seq.a 5 = 10)
  (h12 : seq.a 12 = 31) :
  commonDifference seq = 3 := by
  sorry


end common_difference_from_terms_l3941_394139


namespace quadratic_root_l3941_394184

theorem quadratic_root (p q r : ℝ) (h : p ≠ 0 ∧ q ≠ r) :
  let f : ℝ → ℝ := λ x ↦ p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)
  (f 1 = 0) →
  ∃ x : ℝ, x ≠ 1 ∧ f x = 0 ∧ x = r * (p - q) / (p * (q - r)) := by
  sorry

end quadratic_root_l3941_394184


namespace sqrt2_not_in_rational_intervals_l3941_394114

theorem sqrt2_not_in_rational_intervals (p q : ℕ) (h_coprime : Nat.Coprime p q) 
  (h_p_lt_q : p < q) (h_q_ne_0 : q ≠ 0) : 
  |Real.sqrt 2 / 2 - p / q| > 1 / (4 * q^2) :=
sorry

end sqrt2_not_in_rational_intervals_l3941_394114


namespace impossibility_of_2005_vectors_l3941_394140

/-- A type representing a vector in a plane -/
def PlaneVector : Type := ℝ × ℝ

/-- A function to check if a vector is non-zero -/
def is_nonzero (v : PlaneVector) : Prop := v ≠ (0, 0)

/-- A function to calculate the sum of three vectors -/
def sum_three (v1 v2 v3 : PlaneVector) : PlaneVector :=
  (v1.1 + v2.1 + v3.1, v1.2 + v2.2 + v3.2)

/-- The main theorem -/
theorem impossibility_of_2005_vectors :
  ¬ ∃ (vectors : Fin 2005 → PlaneVector),
    (∀ i, is_nonzero (vectors i)) ∧
    (∀ (subset : Fin 10 → Fin 2005),
      ∃ (i j k : Fin 10), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
        sum_three (vectors (subset i)) (vectors (subset j)) (vectors (subset k)) = (0, 0)) :=
by sorry

end impossibility_of_2005_vectors_l3941_394140


namespace repeating_decimal_division_l3941_394189

theorem repeating_decimal_division (A B C D : Nat) : 
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) →
  (100 * A + 10 * B + C) / (1000 * B + 100 * B + 10 * B + B) = 
    (1000 * B + 100 * C + 10 * D + B) / 9999 →
  A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7 := by
sorry

end repeating_decimal_division_l3941_394189


namespace square_fence_perimeter_l3941_394123

theorem square_fence_perimeter 
  (total_posts : ℕ) 
  (post_width : ℚ) 
  (gap_width : ℕ) : 
  total_posts = 36 → 
  post_width = 1/3 → 
  gap_width = 6 → 
  (4 * ((total_posts / 4 + 1) * post_width + (total_posts / 4) * gap_width)) = 204 := by
  sorry

end square_fence_perimeter_l3941_394123


namespace function_value_problem_l3941_394135

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (1/2 * x - 1) = 2 * x + 3)
  (h2 : f (m - 1) = 6) : 
  m = 3/4 := by
sorry

end function_value_problem_l3941_394135


namespace OPQRS_shape_l3941_394169

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The figure formed by connecting points O, P, Q, R, and S -/
inductive Figure
  | Parallelepiped
  | Plane
  | StraightLine
  | General3D

/-- The theorem stating that OPQRS can only be a parallelepiped or a plane -/
theorem OPQRS_shape (O P Q R S : Point3D)
  (hO : O = ⟨0, 0, 0⟩)
  (hR : R = ⟨P.x + Q.x, P.y + Q.y, P.z + Q.z⟩)
  (hDistinct : O ≠ P ∧ O ≠ Q ∧ O ≠ R ∧ O ≠ S ∧ P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S) :
  (∃ f : Figure, f = Figure.Parallelepiped ∨ f = Figure.Plane) ∧
  ¬(∃ f : Figure, f = Figure.StraightLine ∨ f = Figure.General3D) :=
sorry

end OPQRS_shape_l3941_394169


namespace problem_solution_l3941_394129

theorem problem_solution :
  let a : ℚ := -1/2
  let x : ℤ := 8
  let y : ℤ := 5
  (a * (a^4 - a + 1) * (a - 2) = 125/64) ∧
  ((x + 2*y) * (x - y) - (2*x - y) * (-x - y) = 87) := by
  sorry

end problem_solution_l3941_394129


namespace largest_four_digit_congruent_to_25_mod_26_l3941_394182

theorem largest_four_digit_congruent_to_25_mod_26 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 26] → n ≤ 9983 :=
by sorry

end largest_four_digit_congruent_to_25_mod_26_l3941_394182


namespace correct_number_of_students_l3941_394103

/-- The number of students in a class preparing for a field trip --/
def number_of_students : ℕ := 30

/-- The amount each student contributes per Friday in dollars --/
def contribution_per_friday : ℕ := 2

/-- The number of Fridays in the collection period --/
def number_of_fridays : ℕ := 8

/-- The total amount collected for the trip in dollars --/
def total_amount : ℕ := 480

/-- Theorem stating that the number of students is correct given the conditions --/
theorem correct_number_of_students :
  number_of_students * contribution_per_friday * number_of_fridays = total_amount :=
sorry

end correct_number_of_students_l3941_394103


namespace fold_square_problem_l3941_394199

/-- Given a square ABCD with side length 8 cm, if corner C is folded to point E
    which is one-third of the way along AD from D, and F is the point where the
    fold intersects CD, then the length of FD is 32/9 cm. -/
theorem fold_square_problem (A B C D E F G : ℝ × ℝ) : 
  (∀ (X Y : ℝ × ℝ), dist X Y = dist A B → dist X Y = 8) →  -- Square side length is 8
  dist A D = 8 →  -- AD is a side of the square
  dist D E = 8/3 →  -- E is one-third along AD from D
  F.1 = D.1 ∧ F.2 ≤ D.2 ∧ F.2 ≥ C.2 →  -- F is on CD
  dist C E = dist C F →  -- C is folded onto E
  dist F D = 32/9 := by
  sorry


end fold_square_problem_l3941_394199


namespace bean_jar_count_bean_jar_count_proof_l3941_394155

theorem bean_jar_count : ℕ → Prop :=
  fun total_beans =>
    let red_beans := total_beans / 4
    let remaining_after_red := total_beans - red_beans
    let white_beans := remaining_after_red / 3
    let remaining_after_white := remaining_after_red - white_beans
    let green_beans := remaining_after_white / 2
    (green_beans = 143) → (total_beans = 572)

theorem bean_jar_count_proof : bean_jar_count 572 := by
  sorry

end bean_jar_count_bean_jar_count_proof_l3941_394155


namespace one_third_of_5_4_l3941_394134

theorem one_third_of_5_4 : (5.4 / 3 : ℚ) = 9 / 5 := by
  sorry

end one_third_of_5_4_l3941_394134


namespace solution_set_inequality_l3941_394146

theorem solution_set_inequality (x : ℝ) : (x - 2) / (x + 1) < 0 ↔ -1 < x ∧ x < 2 := by
  sorry

end solution_set_inequality_l3941_394146


namespace symmetry_implies_periodic_l3941_394196

/-- A function that is symmetric with respect to two distinct points is periodic. -/
theorem symmetry_implies_periodic (f : ℝ → ℝ) (a b : ℝ) 
  (ha : ∀ x, f (a - x) = f (a + x))
  (hb : ∀ x, f (b - x) = f (b + x))
  (hab : a ≠ b) :
  ∀ x, f (x + (2*b - 2*a)) = f x :=
by sorry

end symmetry_implies_periodic_l3941_394196


namespace sum_of_digits_10_95_minus_195_l3941_394183

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem: sum of digits of 10^95 - 195 is 841 -/
theorem sum_of_digits_10_95_minus_195 : sum_of_digits (10^95 - 195) = 841 := by sorry

end sum_of_digits_10_95_minus_195_l3941_394183


namespace shrimp_cost_per_pound_l3941_394116

/-- Calculates the cost per pound of shrimp for Wayne's shrimp cocktail appetizer. -/
theorem shrimp_cost_per_pound 
  (shrimp_per_guest : ℕ) 
  (num_guests : ℕ) 
  (shrimp_per_pound : ℕ) 
  (total_cost : ℚ) : 
  shrimp_per_guest = 5 → 
  num_guests = 40 → 
  shrimp_per_pound = 20 → 
  total_cost = 170 → 
  (total_cost / (shrimp_per_guest * num_guests / shrimp_per_pound : ℚ)) = 17 :=
by sorry

end shrimp_cost_per_pound_l3941_394116


namespace intersection_nonempty_implies_a_range_l3941_394187

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

theorem intersection_nonempty_implies_a_range :
  (∃ a : ℝ, (A ∩ B a).Nonempty) ↔ {a : ℝ | a > -8} = Set.Ioi (-8) := by
  sorry

end intersection_nonempty_implies_a_range_l3941_394187


namespace absolute_value_nonnegative_absolute_value_location_l3941_394117

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

theorem absolute_value_location (a : ℝ) : |a| = 0 ∨ |a| > 0 := by
  sorry

end absolute_value_nonnegative_absolute_value_location_l3941_394117


namespace yoga_to_exercise_ratio_l3941_394115

/-- Proves that the ratio of yoga time to total exercise time is 1:1 -/
theorem yoga_to_exercise_ratio : 
  ∀ (gym_time bicycle_time yoga_time : ℝ),
  gym_time / bicycle_time = 2 / 3 →
  bicycle_time = 12 →
  yoga_time = 20 →
  yoga_time / (gym_time + bicycle_time) = 1 := by
  sorry

end yoga_to_exercise_ratio_l3941_394115


namespace larger_sample_more_accurate_l3941_394188

-- Define a sampling survey
structure SamplingSurvey where
  population : Set ℝ
  sample : Set ℝ
  sample_size : ℕ

-- Define estimation accuracy
def estimation_accuracy (survey : SamplingSurvey) : ℝ := sorry

-- Theorem stating that larger sample size leads to more accurate estimation
theorem larger_sample_more_accurate (survey1 survey2 : SamplingSurvey) 
  (h : survey1.population = survey2.population) 
  (h_size : survey1.sample_size < survey2.sample_size) : 
  estimation_accuracy survey1 < estimation_accuracy survey2 := by
  sorry

end larger_sample_more_accurate_l3941_394188


namespace hyperbola_equation_l3941_394128

/-- A hyperbola with center at the origin, axes of symmetry being coordinate axes,
    one focus coinciding with the focus of y^2 = 8x, and one asymptote being x + y = 0 -/
structure Hyperbola where
  /-- The focus of the parabola y^2 = 8x is (2, 0) -/
  focus : ℝ × ℝ
  /-- One asymptote of the hyperbola is x + y = 0 -/
  asymptote : ℝ → ℝ
  /-- The hyperbola's equation is in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  a : ℝ
  b : ℝ
  focus_eq : focus = (2, 0)
  asymptote_eq : asymptote = fun x => -x
  ab_relation : b / a = 1

/-- The equation of the hyperbola is x^2/2 - y^2/2 = 1 -/
theorem hyperbola_equation (C : Hyperbola) : 
  ∀ x y : ℝ, (x^2 / 2) - (y^2 / 2) = 1 ↔ 
    (x^2 / C.a^2) - (y^2 / C.b^2) = 1 :=
by sorry

end hyperbola_equation_l3941_394128


namespace floor_a_n_l3941_394164

/-- Sequence defined by the recurrence relation -/
def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n)^2 / (a n + 1)

/-- Theorem stating that the floor of a_n is 1994 - n for 0 ≤ n ≤ 998 -/
theorem floor_a_n (n : ℕ) (h : n ≤ 998) : 
  ⌊a n⌋ = 1994 - n := by sorry

end floor_a_n_l3941_394164


namespace sum_of_square_perimeters_l3941_394148

/-- The sum of the perimeters of an infinite sequence of squares, where each subsequent square
    is formed by connecting the midpoints of the sides of the previous square, given that the
    initial square has a side length of s. -/
theorem sum_of_square_perimeters (s : ℝ) (h : s > 0) :
  (∑' n, 4 * s / (2 ^ n)) = 8 * s := by
  sorry

end sum_of_square_perimeters_l3941_394148


namespace furniture_fraction_l3941_394108

def original_savings : ℚ := 960
def tv_cost : ℚ := 240

theorem furniture_fraction : 
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end furniture_fraction_l3941_394108


namespace tangent_point_coordinates_l3941_394198

theorem tangent_point_coordinates : 
  ∀ x y : ℝ, 
    y = Real.exp (-x) →                        -- Point P(x,y) is on the curve y = e^(-x)
    (- Real.exp (-x)) = -2 →                   -- Tangent line is parallel to 2x + y + 1 = 0
    x = -Real.log 2 ∧ y = 2 := by              -- Coordinates of P are (-ln2, 2)
  sorry

end tangent_point_coordinates_l3941_394198


namespace computer_science_marks_l3941_394160

theorem computer_science_marks 
  (geography : ℕ) 
  (history_government : ℕ) 
  (art : ℕ) 
  (modern_literature : ℕ) 
  (average : ℚ) 
  (h1 : geography = 56)
  (h2 : history_government = 60)
  (h3 : art = 72)
  (h4 : modern_literature = 80)
  (h5 : average = 70.6)
  : ∃ (computer_science : ℕ),
    (geography + history_government + art + computer_science + modern_literature) / 5 = average ∧ 
    computer_science = 85 := by
sorry

end computer_science_marks_l3941_394160
