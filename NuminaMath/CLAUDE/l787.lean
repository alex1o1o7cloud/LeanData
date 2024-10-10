import Mathlib

namespace joan_initial_balloons_l787_78737

/-- The number of blue balloons Joan initially had -/
def initial_balloons : ℕ := sorry

/-- The number of balloons Sally gave to Joan -/
def sally_gave : ℕ := 5

/-- The number of balloons Joan gave to Jessica -/
def joan_gave : ℕ := 2

/-- The number of balloons Joan has now -/
def joan_now : ℕ := 12

theorem joan_initial_balloons :
  initial_balloons + sally_gave - joan_gave = joan_now :=
sorry

end joan_initial_balloons_l787_78737


namespace sqrt_equality_implies_inequality_l787_78723

theorem sqrt_equality_implies_inequality (x y α : ℝ) : 
  Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α) → x + y ≥ 2 * α := by
  sorry

end sqrt_equality_implies_inequality_l787_78723


namespace min_value_of_f_l787_78703

/-- The function f(x) = 3x^2 - 18x + 2023 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

theorem min_value_of_f :
  ∃ (m : ℝ), m = 1996 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end min_value_of_f_l787_78703


namespace jake_sausage_cost_l787_78792

-- Define the parameters
def package_weight : ℝ := 2
def num_packages : ℕ := 3
def price_per_pound : ℝ := 4

-- Define the theorem
theorem jake_sausage_cost :
  package_weight * num_packages * price_per_pound = 24 := by
  sorry

end jake_sausage_cost_l787_78792


namespace arithmetic_sequence_property_l787_78798

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (a_val b_val : ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a5 : a 5 = a_val)
  (h_a10 : a 10 = b_val) :
  a 15 = 2 * b_val - a_val :=
sorry

end arithmetic_sequence_property_l787_78798


namespace polynomial_B_value_l787_78709

def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 144

def roots : List ℤ := [3, 3, 2, 2, 1, 1]

theorem polynomial_B_value :
  ∀ (A B C D : ℤ),
  (∀ r ∈ roots, polynomial r A B C D = 0) →
  B = -126 := by
  sorry

end polynomial_B_value_l787_78709


namespace polynomial_factorization_l787_78719

theorem polynomial_factorization (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end polynomial_factorization_l787_78719


namespace polynomial_factorization_l787_78710

theorem polynomial_factorization (a b x : ℝ) : 
  a + (a+b)*x + (a+2*b)*x^2 + (a+3*b)*x^3 + 3*b*x^4 + 2*b*x^5 + b*x^6 = 
  (1 + x)*(1 + x^2)*(a + b*x + b*x^2 + b*x^3) := by
  sorry

end polynomial_factorization_l787_78710


namespace infinite_solutions_iff_d_equals_five_l787_78750

theorem infinite_solutions_iff_d_equals_five :
  ∀ d : ℝ, (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end infinite_solutions_iff_d_equals_five_l787_78750


namespace no_solution_condition_l787_78766

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, 3 * |x + 3*a| + |x + a^2| + 2*x ≠ a) ↔ (a < 0 ∨ a > 10) := by
  sorry

end no_solution_condition_l787_78766


namespace semicircle_radius_l787_78718

/-- The radius of a semi-circle with perimeter 198 cm is 198 / (π + 2) cm. -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 198) : 
  perimeter / (Real.pi + 2) = 198 / (Real.pi + 2) := by
  sorry

end semicircle_radius_l787_78718


namespace total_distance_via_intermediate_point_l787_78784

/-- The total distance traveled from (2, 3) to (-3, 2) via (1, -1) is √17 + 5. -/
theorem total_distance_via_intermediate_point :
  let start : ℝ × ℝ := (2, 3)
  let intermediate : ℝ × ℝ := (1, -1)
  let end_point : ℝ × ℝ := (-3, 2)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance start intermediate + distance intermediate end_point = Real.sqrt 17 + 5 := by
  sorry

end total_distance_via_intermediate_point_l787_78784


namespace adjacent_pair_arrangements_l787_78756

def number_of_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n

theorem adjacent_pair_arrangements :
  let total_people : ℕ := 5
  let adjacent_pair : ℕ := 2
  let remaining_people : ℕ := total_people - adjacent_pair + 1
  number_of_arrangements remaining_people remaining_people = 24 :=
by sorry

end adjacent_pair_arrangements_l787_78756


namespace caspers_candies_l787_78744

theorem caspers_candies (initial_candies : ℕ) : 
  let day1_after_eating := (3 * initial_candies) / 4
  let day1_remaining := day1_after_eating - 3
  let day2_after_eating := (4 * day1_remaining) / 5
  let day2_remaining := day2_after_eating - 5
  let day3_after_giving := day2_remaining - 7
  let final_candies := (5 * day3_after_giving) / 6
  final_candies = 10 → initial_candies = 44 := by
sorry

end caspers_candies_l787_78744


namespace complex_pure_imaginary_l787_78721

theorem complex_pure_imaginary (a : ℝ) : 
  (a^2 - 3*a + 2 : ℂ) + (a - 1 : ℂ) * Complex.I = Complex.I * (b : ℝ) → a = 2 :=
by sorry

end complex_pure_imaginary_l787_78721


namespace fraction_sum_equality_l787_78734

theorem fraction_sum_equality : (3 / 10 : ℚ) + (2 / 100 : ℚ) + (8 / 1000 : ℚ) + (8 / 10000 : ℚ) = 0.3288 := by
  sorry

end fraction_sum_equality_l787_78734


namespace mrs_hilt_reading_l787_78758

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading :
  total_chapters = 68 := by sorry

end mrs_hilt_reading_l787_78758


namespace greatest_x_quadratic_inequality_l787_78762

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 16*x + 63 ≤ 0 → x ≤ 9 :=
by sorry

end greatest_x_quadratic_inequality_l787_78762


namespace transformation_possible_l787_78783

/-- Represents a move that can be applied to a sequence of numbers. -/
inductive Move
  | RotateThree (x y z : ℕ) : Move
  | SwapTwo (x y : ℕ) : Move

/-- Checks if a move is valid according to the rules. -/
def isValidMove (move : Move) : Prop :=
  match move with
  | Move.RotateThree x y z => (x + y + z) % 3 = 0
  | Move.SwapTwo x y => (x - y) % 3 = 0 ∨ (y - x) % 3 = 0

/-- Represents a sequence of numbers. -/
def Sequence := List ℕ

/-- Applies a move to a sequence. -/
def applyMove (seq : Sequence) (move : Move) : Sequence :=
  sorry

/-- Checks if a sequence can be transformed into another sequence using valid moves. -/
def canTransform (initial final : Sequence) : Prop :=
  ∃ (moves : List Move), (∀ move ∈ moves, isValidMove move) ∧
    (moves.foldl applyMove initial = final)

/-- The main theorem to be proved. -/
theorem transformation_possible (n : ℕ) :
  n > 1 →
  (canTransform (List.range n) ((n :: List.range (n-1)))) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
  sorry

end transformation_possible_l787_78783


namespace event_probability_l787_78779

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^3 = 63/64) →
  3 * p * (1 - p)^2 = 9/64 := by
sorry

end event_probability_l787_78779


namespace triangle_semicircle_inequality_l787_78787

-- Define a triangle by its semiperimeter and inradius
structure Triangle where
  s : ℝ  -- semiperimeter
  r : ℝ  -- inradius
  s_pos : 0 < s
  r_pos : 0 < r

-- Define the radius of the circle tangent to the three semicircles
noncomputable def t (tri : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_semicircle_inequality (tri : Triangle) :
  tri.s / 2 < t tri ∧ t tri ≤ tri.s / 2 + (1 - Real.sqrt 3 / 2) * tri.r := by
  sorry

end triangle_semicircle_inequality_l787_78787


namespace polynomial_non_negative_l787_78797

theorem polynomial_non_negative (p q : ℝ) (h : q > p^2) :
  ∀ x : ℝ, x^2 + 2*p*x + q ≥ 0 := by
  sorry

end polynomial_non_negative_l787_78797


namespace single_burger_cost_l787_78794

theorem single_burger_cost 
  (total_spent : ℚ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℚ)
  (h1 : total_spent = 64.5)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 29)
  (h4 : double_burger_cost = 1.5)
  : ∃ (single_burger_cost : ℚ), single_burger_cost = 1 :=
by
  sorry

end single_burger_cost_l787_78794


namespace solve_journey_l787_78726

def journey_problem (total_distance : ℝ) (cycling_speed : ℝ) (walking_speed : ℝ) (total_time : ℝ) : Prop :=
  let cycling_distance : ℝ := (2/3) * total_distance
  let walking_distance : ℝ := total_distance - cycling_distance
  let cycling_time : ℝ := cycling_distance / cycling_speed
  let walking_time : ℝ := walking_distance / walking_speed
  (cycling_time + walking_time = total_time) → (walking_distance = 6)

theorem solve_journey :
  journey_problem 18 20 4 (70/60) := by
  sorry

end solve_journey_l787_78726


namespace original_number_proof_l787_78706

theorem original_number_proof (x : ℝ) : 1 + 1/x = 8/3 → x = 3/5 := by
  sorry

end original_number_proof_l787_78706


namespace pascal_ratio_row_l787_78770

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Checks if three consecutive entries in a row have the ratio 2:3:4 -/
def has_ratio_2_3_4 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 2 / 3 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 3 / 4

theorem pascal_ratio_row : 
  ∃ n : ℕ, has_ratio_2_3_4 n ∧ ∀ m : ℕ, m < n → ¬has_ratio_2_3_4 m :=
by sorry

end pascal_ratio_row_l787_78770


namespace monic_cubic_polynomial_uniqueness_l787_78759

/-- A monic cubic polynomial with real coefficients -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

theorem monic_cubic_polynomial_uniqueness (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  p (1 + 3*I) = 0 ∧ p 0 = -108 →
  a = -12.8 ∧ b = 31 ∧ c = -108 := by
  sorry

end monic_cubic_polynomial_uniqueness_l787_78759


namespace fraction_to_decimal_l787_78740

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.33125 ↔ n = 53 ∧ d = 160 := by
  sorry

end fraction_to_decimal_l787_78740


namespace laundry_time_difference_l787_78782

theorem laundry_time_difference : ∀ (clothes_time towels_time sheets_time : ℕ),
  clothes_time = 30 →
  towels_time = 2 * clothes_time →
  clothes_time + towels_time + sheets_time = 135 →
  towels_time - sheets_time = 15 := by
sorry

end laundry_time_difference_l787_78782


namespace ant_problem_l787_78724

theorem ant_problem (abe_ants cece_ants duke_ants beth_ants : ℕ) 
  (total_ants : ℕ) (beth_percentage : ℚ) :
  abe_ants = 4 →
  cece_ants = 2 * abe_ants →
  duke_ants = abe_ants / 2 →
  beth_ants = abe_ants + (beth_percentage / 100) * abe_ants →
  total_ants = abe_ants + beth_ants + cece_ants + duke_ants →
  total_ants = 20 →
  beth_percentage = 50 := by
sorry

end ant_problem_l787_78724


namespace ab_greater_than_b_squared_l787_78767

theorem ab_greater_than_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end ab_greater_than_b_squared_l787_78767


namespace line_hyperbola_intersection_l787_78713

/-- The line equation kx - y - 2k = 0 -/
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 * k = 0

/-- The hyperbola equation x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- The asymptotes of the hyperbola -/
def asymptotes (x y : ℝ) : Prop := y = x ∨ y = -x

/-- The theorem stating that if the line and hyperbola have only one common point, then k = 1 or k = -1 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, line k p.1 p.2 ∧ hyperbola p.1 p.2) → 
  k = 1 ∨ k = -1 :=
sorry

end line_hyperbola_intersection_l787_78713


namespace triangle_is_equilateral_l787_78749

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- angles
  (a b c : ℝ)  -- side lengths

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.B = 60 ∧ t.b^2 = t.a * t.c

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) (h : is_valid_triangle t) : 
  t.A = 60 ∧ t.B = 60 ∧ t.C = 60 :=
sorry

end triangle_is_equilateral_l787_78749


namespace quadratic_equation_solutions_l787_78785

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by sorry

end quadratic_equation_solutions_l787_78785


namespace acute_triangle_on_perpendicular_lines_l787_78752

-- Define an acute-angled triangle
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute_a : a^2 < b^2 + c^2
  acute_b : b^2 < a^2 + c^2
  acute_c : c^2 < a^2 + b^2

-- Theorem statement
theorem acute_triangle_on_perpendicular_lines (t : AcuteTriangle) :
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
  x^2 + y^2 = t.c^2 ∧
  x^2 + z^2 = t.b^2 ∧
  y^2 + z^2 = t.a^2 :=
sorry

end acute_triangle_on_perpendicular_lines_l787_78752


namespace local_minimum_of_f_l787_78741

-- Define the function f(x) = x³ - 12x
def f (x : ℝ) : ℝ := x^3 - 12*x

-- State the theorem
theorem local_minimum_of_f :
  ∃ (x₀ : ℝ), IsLocalMin f x₀ ∧ x₀ = 2 := by sorry

end local_minimum_of_f_l787_78741


namespace stone_volume_l787_78711

/-- Represents a rectangular cuboid bowl -/
structure Bowl where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the volume of water in the bowl given its height -/
def water_volume (b : Bowl) (water_height : ℝ) : ℝ :=
  b.width * b.length * water_height

theorem stone_volume (b : Bowl) (initial_water_height final_water_height : ℝ) :
  b.width = 16 →
  b.length = 14 →
  b.height = 9 →
  initial_water_height = 4 →
  final_water_height = 9 →
  water_volume b final_water_height - water_volume b initial_water_height = 1120 :=
by sorry

end stone_volume_l787_78711


namespace midpoint_product_and_distance_l787_78736

/-- Given that C is the midpoint of segment AB, prove that xy = -12 and d = 4√5 -/
theorem midpoint_product_and_distance (x y : ℝ) :
  (4 : ℝ) = (2 + x) / 2 →
  (2 : ℝ) = (6 + y) / 2 →
  x * y = -12 ∧ Real.sqrt ((x - 2)^2 + (y - 6)^2) = 4 * Real.sqrt 5 := by
  sorry

#check midpoint_product_and_distance

end midpoint_product_and_distance_l787_78736


namespace fraction_comparisons_and_absolute_value_l787_78769

theorem fraction_comparisons_and_absolute_value :
  (-3 : ℚ) / 7 < (-8 : ℚ) / 21 ∧
  (-5 : ℚ) / 6 > (-6 : ℚ) / 7 ∧
  |3.1 - Real.pi| = Real.pi - 3.1 := by
  sorry

end fraction_comparisons_and_absolute_value_l787_78769


namespace expression_value_l787_78764

theorem expression_value (x y z : ℚ) 
  (eq1 : 3 * x - 2 * y - 4 * z = 0)
  (eq2 : 2 * x + y - 9 * z = 0)
  (z_neq_zero : z ≠ 0) :
  (x^2 - 2*x*y) / (y^2 + 2*z^2) = -24/25 := by
  sorry

end expression_value_l787_78764


namespace sqrt_inequality_l787_78729

theorem sqrt_inequality (a : ℝ) : (0 < a ∧ a < 1) ↔ a < Real.sqrt a := by
  sorry

end sqrt_inequality_l787_78729


namespace two_points_determine_line_l787_78739

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem two_points_determine_line (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃! l : Line, pointOnLine p1 l ∧ pointOnLine p2 l :=
sorry

end two_points_determine_line_l787_78739


namespace first_place_points_l787_78786

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : Nat)
  (games_per_pair : Nat)
  (total_points : Nat)
  (last_place_points : Nat)

/-- Calculates the number of games played in the tournament -/
def num_games (t : Tournament) : Nat :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Theorem stating the first-place team's points in the given tournament conditions -/
theorem first_place_points (t : Tournament)
  (h1 : t.num_teams = 4)
  (h2 : t.games_per_pair = 2)
  (h3 : t.total_points = num_games t * 2)
  (h4 : t.last_place_points = 5) :
  ∃ (first_place_points : Nat),
    first_place_points = 7 ∧
    first_place_points + t.last_place_points ≤ t.total_points :=
by
  sorry


end first_place_points_l787_78786


namespace work_completion_l787_78774

/-- Represents the number of days it takes to complete the work -/
structure WorkDays where
  together : ℝ
  a_alone : ℝ
  initial_together : ℝ
  a_remaining : ℝ

/-- Given work completion rates, proves that 'a' worked alone for 9 days after 'b' left -/
theorem work_completion (w : WorkDays) 
  (h1 : w.together = 40)
  (h2 : w.a_alone = 12)
  (h3 : w.initial_together = 10) : 
  w.a_remaining = 9 := by
sorry

end work_completion_l787_78774


namespace determine_down_speed_man_down_speed_l787_78704

/-- The speed of a man traveling up and down a hill -/
structure TravelSpeed where
  up : ℝ
  down : ℝ
  average : ℝ

/-- Theorem stating that given the up speed and average speed, we can determine the down speed -/
theorem determine_down_speed (s : TravelSpeed) (h1 : s.up = 24) (h2 : s.average = 28.8) :
  s.down = 36 := by
  sorry

/-- Main theorem proving the specific case in the problem -/
theorem man_down_speed :
  ∃ s : TravelSpeed, s.up = 24 ∧ s.average = 28.8 ∧ s.down = 36 := by
  sorry

end determine_down_speed_man_down_speed_l787_78704


namespace cubic_function_extrema_l787_78793

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem cubic_function_extrema (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    IsLocalMax (f m) x₁ ∧ 
    IsLocalMin (f m) x₂) ↔ 
  (m < -3 ∨ m > 6) :=
sorry

end cubic_function_extrema_l787_78793


namespace no_real_solutions_l787_78795

theorem no_real_solutions (k d : ℝ) (hk : k = -1) (hd : d < 0 ∨ d > 2) :
  ¬∃ (x y : ℝ), x^3 + y^3 = 2 ∧ y = k * x + d :=
sorry

end no_real_solutions_l787_78795


namespace cubic_polynomial_root_l787_78751

theorem cubic_polynomial_root (x : ℝ) (h : x = Real.rpow 4 (1/3) + 1) : 
  x^3 - 3*x^2 + 3*x - 5 = 0 := by
sorry

end cubic_polynomial_root_l787_78751


namespace equation_solution_l787_78745

theorem equation_solution : 
  ∀ x y : ℚ, 
  y = 3 * x → 
  (5 * y^2 + 2 * y + 3 = 3 * (9 * x^2 + y + 1)) → 
  (x = 0 ∨ x = 1/6) :=
by sorry

end equation_solution_l787_78745


namespace sum_of_digits_of_k_l787_78715

def k : ℕ := 10^30 - 36

-- Function to calculate the sum of digits
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_k : sum_of_digits k = 262 := by sorry

end sum_of_digits_of_k_l787_78715


namespace equation_four_solutions_l787_78775

theorem equation_four_solutions 
  (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, 
    (x - a) * (x - b) * (x - c) * (x - d) +
    (x - a) * (x - b) * (x - c) * (x - e) +
    (x - a) * (x - b) * (x - d) * (x - e) +
    (x - a) * (x - c) * (x - d) * (x - e) +
    (x - b) * (x - c) * (x - d) * (x - e) = 0 :=
by sorry

end equation_four_solutions_l787_78775


namespace leonardo_nap_duration_l787_78753

def minutes_per_hour : ℕ := 60

def fifth_of_hour (total_minutes : ℕ) : ℕ := total_minutes / 5

theorem leonardo_nap_duration : fifth_of_hour minutes_per_hour = 12 := by
  sorry

end leonardo_nap_duration_l787_78753


namespace parabola_slope_l787_78717

/-- The slope of line MF for a parabola y² = 2px with point M(3, m) at distance 4 from focus -/
theorem parabola_slope (p m : ℝ) : p > 0 → m > 0 → m^2 = 6*p → (3 + p/2)^2 + m^2 = 16 → 
  (m / (3 - p/2) : ℝ) = Real.sqrt 3 := by
  sorry

end parabola_slope_l787_78717


namespace min_value_trig_expression_limit_approaches_min_value_l787_78790

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π/2) :
  3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ ≥ 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3)) :=
by sorry

theorem limit_approaches_min_value :
  ∀ ε > 0, ∃ δ > 0, ∀ θ, 0 < θ ∧ θ < δ →
    abs ((3 * cos θ + 1 / (2 * sin θ) + 2 * sqrt 2 * tan θ) - 3 * (3 ^ (1/3)) * (sqrt 2 ^ (1/3))) < ε :=
by sorry

end min_value_trig_expression_limit_approaches_min_value_l787_78790


namespace john_can_lift_2800_pounds_l787_78735

-- Define the given values
def original_squat : ℝ := 135
def training_increase : ℝ := 265
def bracer_multiplier : ℝ := 7  -- 600% increase means multiplying by 7 (1 + 6)

-- Define the calculation steps
def new_squat : ℝ := original_squat + training_increase
def final_lift : ℝ := new_squat * bracer_multiplier

-- Theorem statement
theorem john_can_lift_2800_pounds : 
  final_lift = 2800 := by sorry

end john_can_lift_2800_pounds_l787_78735


namespace total_colors_over_two_hours_l787_78732

/-- Represents the number of colors the sky changes through in a 10-minute period -/
structure ColorChange where
  quick : ℕ
  slow : ℕ

/-- Calculates the total number of colors for one hour given a ColorChange pattern -/
def colorsPerHour (change : ColorChange) : ℕ :=
  (change.quick + change.slow) * 3

/-- The color change pattern for the first hour -/
def firstHourPattern : ColorChange :=
  { quick := 5, slow := 2 }

/-- The color change pattern for the second hour (doubled rate) -/
def secondHourPattern : ColorChange :=
  { quick := firstHourPattern.quick * 2, slow := firstHourPattern.slow * 2 }

/-- The main theorem stating the total number of colors over two hours -/
theorem total_colors_over_two_hours :
  colorsPerHour firstHourPattern + colorsPerHour secondHourPattern = 63 := by
  sorry


end total_colors_over_two_hours_l787_78732


namespace taxi_fare_equality_l787_78702

/-- Taxi fare calculation and comparison -/
theorem taxi_fare_equality (mike_miles : ℝ) : 
  (2.5 + 0.25 * mike_miles = 2.5 + 5 + 0.25 * 26) → mike_miles = 46 := by
  sorry

end taxi_fare_equality_l787_78702


namespace pascals_triangle_51_numbers_l787_78796

theorem pascals_triangle_51_numbers (n : ℕ) : n + 1 = 51 → Nat.choose n 2 = 1225 := by
  sorry

end pascals_triangle_51_numbers_l787_78796


namespace simple_interest_problem_l787_78757

/-- Given a sum P put at simple interest for 3 years, if increasing the interest rate
    by 1% results in an additional Rs. 75 interest, then P = 2500. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 75 → P = 2500 := by
sorry

end simple_interest_problem_l787_78757


namespace similar_triangles_leg_sum_l787_78742

theorem similar_triangles_leg_sum (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (1/2) * a * b = 10 →
  a^2 + b^2 = 36 →
  (1/2) * c * d = 360 →
  c = 6 * a →
  d = 6 * b →
  c + d = 16 * Real.sqrt 30 := by
sorry

end similar_triangles_leg_sum_l787_78742


namespace binomial_20_4_l787_78780

theorem binomial_20_4 : Nat.choose 20 4 = 4845 := by
  sorry

end binomial_20_4_l787_78780


namespace cube_root_equation_l787_78707

theorem cube_root_equation (x : ℝ) : (9 * x + 8) ^ (1/3 : ℝ) = 4 → x = 56 / 9 := by
  sorry

end cube_root_equation_l787_78707


namespace exists_valid_arrangement_l787_78738

/-- Represents a position on a regular 25-gon -/
inductive Position
| Vertex : Fin 25 → Position
| Midpoint : Fin 25 → Position

/-- Represents an arrangement of numbers on a regular 25-gon -/
def Arrangement := Position → Fin 50

/-- Checks if the sum of numbers at the ends and midpoint of a side is constant -/
def isConstantSum (arr : Arrangement) : Prop :=
  ∃ s : ℕ, ∀ i : Fin 25,
    (arr (Position.Vertex i)).val + 
    (arr (Position.Midpoint i)).val + 
    (arr (Position.Vertex ((i.val + 1) % 25 : Fin 25))).val = s

/-- Theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ arr : Arrangement, isConstantSum arr ∧ 
  (∀ p : Position, (arr p).val ≥ 1 ∧ (arr p).val ≤ 50) ∧
  (∀ p q : Position, p ≠ q → arr p ≠ arr q) :=
sorry

end exists_valid_arrangement_l787_78738


namespace factor_of_polynomial_l787_78772

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (k : ℝ), 5 * x^2 + 17 * x - 12 = (x + 4) * k := by
  sorry

end factor_of_polynomial_l787_78772


namespace birds_in_cage_l787_78755

theorem birds_in_cage (birds_taken_out birds_left : ℕ) 
  (h1 : birds_taken_out = 10)
  (h2 : birds_left = 9) : 
  birds_taken_out + birds_left = 19 := by
  sorry

end birds_in_cage_l787_78755


namespace vector_equation_solution_l787_78746

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) :
  a = (-3, 1) →
  b = (-1, 2) →
  m • a - n • b = (10, 0) →
  m = -4 ∧ n = -2 := by sorry

end vector_equation_solution_l787_78746


namespace delegate_seating_probability_l787_78765

-- Define the number of delegates and countries
def total_delegates : ℕ := 12
def num_countries : ℕ := 3
def delegates_per_country : ℕ := 4

-- Define the probability as a fraction
def probability : ℚ := 106 / 115

-- State the theorem
theorem delegate_seating_probability :
  let total_arrangements := (total_delegates.factorial) / ((delegates_per_country.factorial) ^ num_countries)
  let unwanted_arrangements := 
    (num_countries * total_delegates * ((total_delegates - delegates_per_country).factorial / 
    (delegates_per_country.factorial ^ (num_countries - 1)))) -
    (num_countries * total_delegates * (delegates_per_country + 2)) +
    (total_delegates * 2)
  (total_arrangements - unwanted_arrangements) / total_arrangements = probability := by
  sorry

end delegate_seating_probability_l787_78765


namespace imaginary_unit_power_2015_l787_78754

theorem imaginary_unit_power_2015 (i : ℂ) (h : i^2 = -1) : i^2015 = -i := by
  sorry

end imaginary_unit_power_2015_l787_78754


namespace vitamin_boxes_count_l787_78712

/-- Given the total number of medicine boxes and the number of supplement boxes,
    prove that the number of vitamin boxes is 472. -/
theorem vitamin_boxes_count (total_medicine : ℕ) (supplements : ℕ) 
    (h1 : total_medicine = 760)
    (h2 : supplements = 288)
    (h3 : ∃ vitamins : ℕ, total_medicine = vitamins + supplements) :
  ∃ vitamins : ℕ, vitamins = 472 ∧ total_medicine = vitamins + supplements :=
by
  sorry

end vitamin_boxes_count_l787_78712


namespace leftover_coins_value_l787_78776

/-- The number of nickels in a complete roll -/
def nickels_per_roll : ℕ := 40

/-- The number of pennies in a complete roll -/
def pennies_per_roll : ℕ := 50

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- Michael's nickels -/
def michael_nickels : ℕ := 183

/-- Michael's pennies -/
def michael_pennies : ℕ := 259

/-- Sarah's nickels -/
def sarah_nickels : ℕ := 167

/-- Sarah's pennies -/
def sarah_pennies : ℕ := 342

/-- The value of leftover coins in cents -/
def leftover_value : ℕ :=
  ((michael_nickels + sarah_nickels) % nickels_per_roll) * nickel_value +
  ((michael_pennies + sarah_pennies) % pennies_per_roll) * penny_value

theorem leftover_coins_value : leftover_value = 151 := by
  sorry

end leftover_coins_value_l787_78776


namespace expression_not_simplifiable_to_AD_l787_78722

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (A B C D M : V)

theorem expression_not_simplifiable_to_AD :
  ∃ (BM DA MB : V), -BM - DA + MB ≠ (A - D) :=
by sorry

end expression_not_simplifiable_to_AD_l787_78722


namespace smallest_x_for_equation_l787_78700

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x ≠ 6 ∧ x ≠ -4 ∧
  (x^2 - 3*x - 18) / (x - 6) = 5 / (x + 4) ∧
  ∀ (y : ℝ), y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 5 / (y + 4) → x ≤ y ∧
  x = (-7 - Real.sqrt 21) / 2 :=
sorry

end smallest_x_for_equation_l787_78700


namespace f_injective_l787_78760

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def injective {α β : Type} (f : α → β) : Prop :=
  ∀ a b : α, f a = f b → a = b

theorem f_injective (f : ℕ → ℕ) 
  (h : ∀ (x y : ℕ), is_perfect_square (f x + y) ↔ is_perfect_square (x + f y)) :
  injective f := by
sorry

end f_injective_l787_78760


namespace no_upper_bound_expression_l787_78727

/-- The expression has no upper bound -/
theorem no_upper_bound_expression (a b c d : ℝ) (h : a * d - b * c = 1) :
  ∀ M : ℝ, ∃ a' b' c' d' : ℝ, 
    a' * d' - b' * c' = 1 ∧ 
    a'^2 + b'^2 + c'^2 + d'^2 + a' * b' + c' * d' > M :=
by sorry

end no_upper_bound_expression_l787_78727


namespace cash_realized_approx_103_74_l787_78761

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
def cash_realized (brokerage_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (1 + brokerage_rate)

/-- Theorem stating that the cash realized is approximately 103.74 given the problem conditions -/
theorem cash_realized_approx_103_74 :
  let brokerage_rate : ℚ := 1 / 400  -- 1/4% expressed as a fraction
  let total_amount : ℚ := 104
  |cash_realized brokerage_rate total_amount - 103.74| < 0.01 := by
  sorry

#eval cash_realized (1/400) 104

end cash_realized_approx_103_74_l787_78761


namespace continuity_reciprocal_quadratic_plus_four_l787_78771

theorem continuity_reciprocal_quadratic_plus_four (x : ℝ) :
  Continuous (fun x => 1 / (x^2 + 4)) :=
sorry

end continuity_reciprocal_quadratic_plus_four_l787_78771


namespace complement_of_M_l787_78748

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {4, 5}

theorem complement_of_M :
  (U \ M) = {1, 2, 3} := by sorry

end complement_of_M_l787_78748


namespace ned_garage_sale_games_l787_78728

/-- The number of games Ned bought from a friend -/
def games_from_friend : ℕ := 50

/-- The number of games that didn't work -/
def bad_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- The number of games Ned bought at the garage sale -/
def games_from_garage_sale : ℕ := (good_games + bad_games) - games_from_friend

theorem ned_garage_sale_games :
  games_from_garage_sale = 27 := by sorry

end ned_garage_sale_games_l787_78728


namespace coordinate_axes_characterization_l787_78708

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of all points on the coordinate axes -/
def CoordinateAxesPoints : Set Point :=
  {p : Point | p.x * p.y = 0}

/-- Predicate to check if a point is on a coordinate axis -/
def IsOnAxis (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

theorem coordinate_axes_characterization :
  ∀ p : Point, p ∈ CoordinateAxesPoints ↔ IsOnAxis p :=
by sorry

end coordinate_axes_characterization_l787_78708


namespace num_lists_15_4_l787_78733

/-- The number of elements in the set to draw from -/
def n : ℕ := 15

/-- The number of draws -/
def k : ℕ := 4

/-- The number of possible lists when drawing k times with replacement from a set of n elements -/
def num_lists (n k : ℕ) : ℕ := n^k

/-- Theorem: The number of possible lists when drawing 4 times with replacement from a set of 15 elements is 50625 -/
theorem num_lists_15_4 : num_lists n k = 50625 := by
  sorry

end num_lists_15_4_l787_78733


namespace smallest_winning_k_l787_78701

/-- Represents the game board -/
def Board := Fin 8 → Fin 8 → Option Char

/-- Checks if a sequence "HMM" or "MMH" exists horizontally or vertically -/
def winning_sequence (board : Board) : Prop :=
  ∃ (i j : Fin 8), 
    (board i j = some 'H' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'M') ∨
    (board i j = some 'M' ∧ board i (j+1) = some 'M' ∧ board i (j+2) = some 'H') ∨
    (board i j = some 'H' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'M') ∨
    (board i j = some 'M' ∧ board (i+1) j = some 'M' ∧ board (i+2) j = some 'H')

/-- Mike's strategy for placing 'M's -/
def mike_strategy (k : ℕ) : Board := sorry

/-- Harry's strategy for placing 'H's -/
def harry_strategy (k : ℕ) (mike_board : Board) : Board := sorry

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy -/
theorem smallest_winning_k : 
  (∀ (k : ℕ), k < 16 → ∃ (harry_board : Board), 
    harry_board = harry_strategy k (mike_strategy k) ∧ ¬winning_sequence harry_board) ∧ 
  (∀ (harry_board : Board), 
    harry_board = harry_strategy 16 (mike_strategy 16) → winning_sequence harry_board) :=
sorry

end smallest_winning_k_l787_78701


namespace franks_problems_per_type_l787_78781

/-- The number of math problems composed by Bill. -/
def bills_problems : ℕ := 20

/-- The number of math problems composed by Ryan. -/
def ryans_problems : ℕ := 2 * bills_problems

/-- The number of math problems composed by Frank. -/
def franks_problems : ℕ := 3 * ryans_problems

/-- The number of different types of math problems each person composes. -/
def problem_types : ℕ := 4

/-- Theorem stating that Frank composes 30 problems of each type. -/
theorem franks_problems_per_type :
  franks_problems / problem_types = 30 := by sorry

end franks_problems_per_type_l787_78781


namespace min_value_sum_reciprocals_l787_78763

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x + y - 3 = 0) : 
  2/x + 1/y ≥ 3 ∧ (2/x + 1/y = 3 ↔ x = 1 ∧ y = 1) :=
sorry

end min_value_sum_reciprocals_l787_78763


namespace rank_inequality_l787_78743

variable {n : ℕ}
variable (A B : Matrix (Fin n) (Fin n) ℝ)

theorem rank_inequality (h1 : n ≥ 2) (h2 : B * B = B) :
  Matrix.rank (A * B - B * A) ≤ Matrix.rank (A * B + B * A) := by
  sorry

end rank_inequality_l787_78743


namespace xiao_ming_english_score_l787_78716

/-- Calculates the weighted average score given three component scores and their weights -/
def weighted_average (listening_score language_score written_score : ℚ) 
  (listening_weight language_weight written_weight : ℕ) : ℚ :=
  (listening_score * listening_weight + language_score * language_weight + written_score * written_weight) / 
  (listening_weight + language_weight + written_weight)

/-- Theorem stating that Xiao Ming's English score is 92.6 given his component scores and the weighting ratio -/
theorem xiao_ming_english_score : 
  weighted_average 92 90 95 3 3 4 = 92.6 := by sorry

end xiao_ming_english_score_l787_78716


namespace transportation_cost_l787_78768

/-- Transportation problem theorem -/
theorem transportation_cost
  (city_A_supply : ℕ)
  (city_B_supply : ℕ)
  (market_C_demand : ℕ)
  (market_D_demand : ℕ)
  (cost_A_to_C : ℕ)
  (cost_A_to_D : ℕ)
  (cost_B_to_C : ℕ)
  (cost_B_to_D : ℕ)
  (x : ℕ)
  (h1 : city_A_supply = 240)
  (h2 : city_B_supply = 260)
  (h3 : market_C_demand = 200)
  (h4 : market_D_demand = 300)
  (h5 : cost_A_to_C = 20)
  (h6 : cost_A_to_D = 30)
  (h7 : cost_B_to_C = 24)
  (h8 : cost_B_to_D = 32)
  (h9 : x ≤ city_A_supply)
  (h10 : x ≤ market_C_demand) :
  (cost_A_to_C * x +
   cost_A_to_D * (city_A_supply - x) +
   cost_B_to_C * (market_C_demand - x) +
   cost_B_to_D * (market_D_demand - (city_A_supply - x))) =
  13920 - 2 * x :=
by sorry

end transportation_cost_l787_78768


namespace andrews_game_preparation_time_l787_78799

/-- The time it takes to prepare all games -/
def total_preparation_time (time_per_game : ℕ) (num_games : ℕ) : ℕ :=
  time_per_game * num_games

/-- Theorem: The total preparation time for 5 games, each taking 5 minutes, is 25 minutes -/
theorem andrews_game_preparation_time :
  total_preparation_time 5 5 = 25 := by
sorry

end andrews_game_preparation_time_l787_78799


namespace andrew_final_share_l787_78747

def total_stickers : ℕ := 2800

def ratio_sum : ℚ := 3/5 + 1 + 3/4 + 1/2 + 7/4

def andrew_initial_share : ℚ := (1 : ℚ) * total_stickers / ratio_sum

def sam_initial_share : ℚ := (3/4 : ℚ) * total_stickers / ratio_sum

def sam_to_andrew : ℚ := 0.4 * sam_initial_share

theorem andrew_final_share :
  ⌊andrew_initial_share + sam_to_andrew⌋ = 791 :=
sorry

end andrew_final_share_l787_78747


namespace sum_of_digits_of_B_is_seven_l787_78730

theorem sum_of_digits_of_B_is_seven :
  ∃ (A B : ℕ),
    (A ≡ (16^16 : ℕ) [MOD 9]) →
    (B ≡ A [MOD 9]) →
    (∃ (C : ℕ), C < 10 ∧ C ≡ B [MOD 9] ∧ C = 7) :=
by sorry

end sum_of_digits_of_B_is_seven_l787_78730


namespace intersection_of_A_and_B_l787_78791

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 1, 2} := by
  sorry

end intersection_of_A_and_B_l787_78791


namespace no_integer_solution_l787_78725

theorem no_integer_solution :
  ¬ ∃ (m n k : ℕ+), ∀ (x y : ℝ),
    (x + 1)^2 + y^2 = (m : ℝ)^2 ∧
    (x - 1)^2 + y^2 = (n : ℝ)^2 ∧
    x^2 + (y - Real.sqrt 3)^2 = (k : ℝ)^2 :=
by sorry

end no_integer_solution_l787_78725


namespace square_complex_real_iff_a_or_b_zero_l787_78778

theorem square_complex_real_iff_a_or_b_zero (a b : ℝ) :
  let z : ℂ := Complex.mk a b
  (∃ (r : ℝ), z^2 = (r : ℂ)) ↔ a = 0 ∨ b = 0 := by
  sorry

end square_complex_real_iff_a_or_b_zero_l787_78778


namespace geometric_series_first_term_l787_78788

/-- Represents an infinite geometric series -/
structure GeometricSeries where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum : ℝ -- sum of the series

/-- Condition for the first, third, and fourth terms forming an arithmetic sequence -/
def arithmeticSequenceCondition (s : GeometricSeries) : Prop :=
  2 * s.a * s.r^2 = s.a + s.a * s.r^3

/-- The main theorem statement -/
theorem geometric_series_first_term 
  (s : GeometricSeries) 
  (h_sum : s.sum = 2020)
  (h_arith : arithmeticSequenceCondition s)
  (h_converge : abs s.r < 1) :
  s.a = 1010 * (1 + Real.sqrt 5) :=
sorry

end geometric_series_first_term_l787_78788


namespace square_area_from_corners_l787_78705

/-- The area of a square with adjacent corners at (1, 2) and (-2, 2) is 9 -/
theorem square_area_from_corners : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, 2)
  let side_length := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := side_length^2
  area = 9 := by sorry

end square_area_from_corners_l787_78705


namespace system_solution_l787_78773

theorem system_solution : ∃! (x y : ℝ), x + y = 2 ∧ 3 * x + y = 4 := by
  sorry

end system_solution_l787_78773


namespace min_value_x_plus_y_l787_78789

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end min_value_x_plus_y_l787_78789


namespace value_of_a_l787_78720

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {a, b, 2}
def B (a b : ℝ) : Set ℝ := {2, b^2, 2*a}

-- State the theorem
theorem value_of_a (a b : ℝ) :
  A a b = B a b → a = 0 ∨ a = 1/4 := by
  sorry

end value_of_a_l787_78720


namespace quadratic_equal_roots_l787_78714

theorem quadratic_equal_roots (a b : ℝ) (h : b^2 = 4*a) :
  (a * b^2) / (a^2 - 4*a + b^2) = 4 := by
sorry

end quadratic_equal_roots_l787_78714


namespace town_hall_eggs_l787_78731

/-- The number of eggs Joe found in different locations --/
structure EggCount where
  clubHouse : ℕ
  park : ℕ
  townHall : ℕ
  total : ℕ

/-- Theorem stating that Joe found 3 eggs in the town hall garden --/
theorem town_hall_eggs (e : EggCount) 
  (h1 : e.clubHouse = 12)
  (h2 : e.park = 5)
  (h3 : e.total = 20)
  (h4 : e.total = e.clubHouse + e.park + e.townHall) :
  e.townHall = 3 := by
  sorry

#check town_hall_eggs

end town_hall_eggs_l787_78731


namespace car_overtakes_buses_l787_78777

/-- The time interval between bus departures in minutes -/
def bus_interval : ℕ := 3

/-- The time taken by a bus to reach the city centre in minutes -/
def bus_travel_time : ℕ := 60

/-- The time taken by the car to reach the city centre in minutes -/
def car_travel_time : ℕ := 35

/-- The number of buses overtaken by the car -/
def buses_overtaken : ℕ := (bus_travel_time - car_travel_time) / bus_interval

theorem car_overtakes_buses :
  buses_overtaken = 8 := by
  sorry

end car_overtakes_buses_l787_78777
