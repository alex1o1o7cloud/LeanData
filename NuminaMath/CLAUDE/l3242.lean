import Mathlib

namespace modified_pattern_cannot_form_polyhedron_l3242_324263

/-- Represents a flat pattern of squares -/
structure FlatPattern where
  squares : ℕ
  foldingLines : ℕ

/-- Represents a modified flat pattern with an extra square and a removed folding line -/
def ModifiedPattern (fp : FlatPattern) : FlatPattern :=
  { squares := fp.squares + 1
  , foldingLines := fp.foldingLines - 1 }

/-- Represents whether a pattern can form a simple polyhedron -/
def CanFormPolyhedron (fp : FlatPattern) : Prop := sorry

/-- Theorem stating that a modified pattern cannot form a simple polyhedron -/
theorem modified_pattern_cannot_form_polyhedron (fp : FlatPattern) : 
  ¬(CanFormPolyhedron (ModifiedPattern fp)) := by
  sorry

end modified_pattern_cannot_form_polyhedron_l3242_324263


namespace nine_hash_seven_l3242_324234

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 1

-- State the theorem to be proved
theorem nine_hash_seven : hash 9 7 = 79 := by
  sorry

end nine_hash_seven_l3242_324234


namespace intersection_of_A_and_B_l3242_324243

def A : Set ℝ := {x | x - 6 < 0}
def B : Set ℝ := {-3, 5, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {-3, 5} := by sorry

end intersection_of_A_and_B_l3242_324243


namespace semicircle_area_ratio_l3242_324275

/-- The ratio of the combined areas of two semicircles to the area of their circumscribing circle -/
theorem semicircle_area_ratio (r : ℝ) (h : r > 0) :
  let semicircle_area := π * (r / 2)^2 / 2
  let circle_area := π * r^2
  2 * semicircle_area / circle_area = 1 / 4 := by
  sorry

end semicircle_area_ratio_l3242_324275


namespace select_five_from_eight_l3242_324254

theorem select_five_from_eight : Nat.choose 8 5 = 56 := by sorry

end select_five_from_eight_l3242_324254


namespace linear_system_fraction_sum_l3242_324297

theorem linear_system_fraction_sum (a b c u v w : ℝ) 
  (eq1 : 17 * u + b * v + c * w = 0)
  (eq2 : a * u + 29 * v + c * w = 0)
  (eq3 : a * u + b * v + 56 * w = 0)
  (ha : a ≠ 17)
  (hu : u ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by
  sorry

end linear_system_fraction_sum_l3242_324297


namespace sum_abcd_l3242_324276

theorem sum_abcd (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
  sorry

end sum_abcd_l3242_324276


namespace student_selection_problem_l3242_324253

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem student_selection_problem :
  let total_students : ℕ := 8
  let selected_students : ℕ := 4
  let students_except_AB : ℕ := 6
  
  -- Number of ways to select with exactly one of A or B
  let with_one_AB : ℕ := 2 * (choose students_except_AB (selected_students - 1))
  
  -- Number of ways to select without A and B
  let without_AB : ℕ := choose students_except_AB selected_students
  
  -- Total number of valid selections
  let total_selections : ℕ := with_one_AB + without_AB
  
  total_selections = 55 := by sorry

end student_selection_problem_l3242_324253


namespace num_divisors_3960_l3242_324225

/-- The number of positive divisors of a natural number n -/
def num_positive_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 3960 is 48 -/
theorem num_divisors_3960 : num_positive_divisors 3960 = 48 := by sorry

end num_divisors_3960_l3242_324225


namespace circular_path_length_l3242_324269

/-- The length of a circular path given specific walking conditions -/
theorem circular_path_length
  (step_length_1 : ℝ)
  (step_length_2 : ℝ)
  (total_footprints : ℕ)
  (h1 : step_length_1 = 0.54)  -- 54 cm in meters
  (h2 : step_length_2 = 0.72)  -- 72 cm in meters
  (h3 : total_footprints = 60)
  (h4 : ∃ (n m : ℕ), n * step_length_1 = m * step_length_2) -- Both complete one lap
  : ∃ (path_length : ℝ), path_length = 21.6 :=
by
  sorry

end circular_path_length_l3242_324269


namespace jack_morning_emails_l3242_324283

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating that Jack received 9 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails = evening_emails + 2 → morning_emails = 9 :=
by sorry

end jack_morning_emails_l3242_324283


namespace runners_meet_closer_than_half_diagonal_l3242_324249

/-- A point moving along a diagonal of a square -/
structure DiagonalRunner where
  position : ℝ  -- Position on the diagonal, normalized to [0, 1]
  direction : Bool  -- True if moving towards the endpoint, False if moving towards the start

/-- The state of two runners on diagonals of a square -/
structure SquareState where
  runner1 : DiagonalRunner
  runner2 : DiagonalRunner
  diagonal_length : ℝ

def distance (s : SquareState) : ℝ :=
  sorry

def update_state (s : SquareState) (t : ℝ) : SquareState :=
  sorry

theorem runners_meet_closer_than_half_diagonal
  (initial_state : SquareState)
  (h_positive_length : initial_state.diagonal_length > 0) :
  ∃ t : ℝ, distance (update_state initial_state t) < initial_state.diagonal_length / 2 := by
  sorry

end runners_meet_closer_than_half_diagonal_l3242_324249


namespace prob_two_girls_l3242_324231

/-- The probability of selecting two girls from a group of 15 members, where 6 are girls -/
theorem prob_two_girls (total : ℕ) (girls : ℕ) (h1 : total = 15) (h2 : girls = 6) :
  (girls.choose 2 : ℚ) / (total.choose 2) = 1 / 7 := by
  sorry

end prob_two_girls_l3242_324231


namespace geometry_problem_l3242_324266

/-- Two lines are different if they are not equal -/
def different_lines (a b : Line) : Prop := a ≠ b

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (p1 p2 : Plane) : Prop := sorry

/-- Two lines are parallel -/
def lines_parallel (l1 l2 : Line) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perp (l1 l2 : Line) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def planes_perp (p1 p2 : Plane) : Prop := sorry

theorem geometry_problem (a b : Line) (α β : Plane) 
  (h1 : different_lines a b) (h2 : different_planes α β) : 
  (((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_parallel α β) → lines_parallel a b) ∧
   ((line_perp_plane a α ∧ line_parallel_plane b β ∧ planes_parallel α β) → lines_perp a b) ∧
   (¬((planes_parallel α β ∧ line_in_plane a α ∧ line_in_plane b β) → lines_parallel a b)) ∧
   ((line_perp_plane a α ∧ line_perp_plane b β ∧ planes_perp α β) → lines_perp a b)) := by
  sorry

end geometry_problem_l3242_324266


namespace hillary_climbing_rate_l3242_324255

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  hillary_rate : ℝ
  eddy_rate : ℝ
  total_distance : ℝ
  hillary_stop_distance : ℝ
  hillary_descent_rate : ℝ
  total_time : ℝ

/-- The theorem stating that Hillary's climbing rate is 800 ft/hr -/
theorem hillary_climbing_rate 
  (scenario : ClimbingScenario)
  (h1 : scenario.total_distance = 5000)
  (h2 : scenario.eddy_rate = scenario.hillary_rate - 500)
  (h3 : scenario.hillary_stop_distance = 1000)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.total_time = 6)
  : scenario.hillary_rate = 800 := by
  sorry

end hillary_climbing_rate_l3242_324255


namespace no_integer_solutions_for_3x_minus_12y_eq_7_l3242_324264

theorem no_integer_solutions_for_3x_minus_12y_eq_7 :
  ¬ ∃ (x y : ℤ), 3 * x - 12 * y = 7 := by
  sorry

end no_integer_solutions_for_3x_minus_12y_eq_7_l3242_324264


namespace apple_pricing_l3242_324232

/-- The cost per kilogram for the first 30 kgs of apples -/
def l : ℚ := 200 / 10

/-- The cost per kilogram for each additional kilogram after the first 30 kgs -/
def m : ℚ := 21

theorem apple_pricing :
  (l * 30 + m * 3 = 663) ∧
  (l * 30 + m * 6 = 726) ∧
  (l * 10 = 200) →
  m = 21 := by sorry

end apple_pricing_l3242_324232


namespace fractional_equation_simplification_l3242_324200

theorem fractional_equation_simplification (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) ↔ (x - 4 * (3 - x) = -6) :=
by sorry

end fractional_equation_simplification_l3242_324200


namespace complement_hit_at_least_once_l3242_324284

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the complement of hitting at least once
-- is equivalent to missing both times
theorem complement_hit_at_least_once (ω : Ω) :
  ¬(hit_at_least_once ω) ↔ miss_both_times ω :=
sorry

end complement_hit_at_least_once_l3242_324284


namespace cos_equality_implies_43_l3242_324233

theorem cos_equality_implies_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end cos_equality_implies_43_l3242_324233


namespace find_other_number_l3242_324295

theorem find_other_number (A B : ℕ+) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 := by
  sorry

end find_other_number_l3242_324295


namespace equal_selection_probability_l3242_324236

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a population size, sample size, and sampling method -/
noncomputable def selectionProbability (N n : ℕ) (method : SamplingMethod) : ℝ :=
  sorry

theorem equal_selection_probability (N n : ℕ) (h1 : N > 0) (h2 : n > 0) (h3 : n ≤ N) :
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Systematic ∧
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Stratified :=
by
  sorry

end equal_selection_probability_l3242_324236


namespace complex_number_properties_l3242_324212

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := m^2 - 2*m + m*Complex.I

/-- The line x - y + 2 = 0 -/
def line (z : ℂ) : Prop := z.re - z.im + 2 = 0

/-- z is in the second quadrant -/
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem complex_number_properties :
  ∀ m : ℝ,
  (second_quadrant (z m) ∧ line (z m)) ↔ (m = 1 ∨ m = 2) ∧
  (m = 1 → Complex.abs (z m) = Real.sqrt 2) ∧
  (m = 2 → Complex.abs (z m) = 2) := by sorry

end complex_number_properties_l3242_324212


namespace dog_cord_length_l3242_324239

/-- The maximum radius of the semi-circular path -/
def max_radius : ℝ := 5

/-- The approximate arc length of the semi-circular path -/
def arc_length : ℝ := 30

/-- The length of the nylon cord -/
def cord_length : ℝ := max_radius

theorem dog_cord_length :
  cord_length = max_radius := by sorry

end dog_cord_length_l3242_324239


namespace erased_numbers_l3242_324237

def numbers_with_one : ℕ := 20
def numbers_with_two : ℕ := 19
def numbers_without_one_or_two : ℕ := 30
def total_numbers : ℕ := 100

theorem erased_numbers :
  numbers_with_one + numbers_with_two + numbers_without_one_or_two ≤ total_numbers ∧
  total_numbers - (numbers_with_one + numbers_with_two + numbers_without_one_or_two - 2) = 33 :=
sorry

end erased_numbers_l3242_324237


namespace complex_roots_nature_l3242_324235

theorem complex_roots_nature (k : ℝ) (hk : k > 0) :
  ∃ (z₁ z₂ : ℂ), 
    (10 * z₁^2 + 5 * Complex.I * z₁ - k = 0) ∧
    (10 * z₂^2 + 5 * Complex.I * z₂ - k = 0) ∧
    (z₁.re ≠ 0 ∧ z₁.im ≠ 0) ∧
    (z₂.re ≠ 0 ∧ z₂.im ≠ 0) ∧
    (z₁ ≠ z₂) :=
by sorry

end complex_roots_nature_l3242_324235


namespace problem_1_l3242_324292

theorem problem_1 : Real.sqrt (3/2) * Real.sqrt (21/4) / Real.sqrt (7/2) = 3/2 := by
  sorry

end problem_1_l3242_324292


namespace triangle_properties_l3242_324260

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.B = Real.pi / 6 ∧
  t.b = Real.sqrt 7 ∧
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3) / 4 :=
by sorry

end triangle_properties_l3242_324260


namespace divisible_by_nine_l3242_324271

/-- Sum of digits function -/
def sum_of_digits (n : ℤ) : ℤ := sorry

theorem divisible_by_nine (x : ℤ) 
  (h : sum_of_digits x = sum_of_digits (3 * x)) : 
  9 ∣ x := by sorry

end divisible_by_nine_l3242_324271


namespace solution_to_system_l3242_324230

theorem solution_to_system (x y : ℝ) : 
  (x^2*y - x*y^2 - 5*x + 5*y + 3 = 0 ∧ 
   x^3*y - x*y^3 - 5*x^2 + 5*y^2 + 15 = 0) ↔ 
  (x = 4 ∧ y = 1) := by sorry

end solution_to_system_l3242_324230


namespace weight_loss_probability_is_0_241_l3242_324244

/-- The probability of a person losing weight after taking a drug, given the total number of volunteers and the number of people who lost weight. -/
def probability_of_weight_loss (total_volunteers : ℕ) (weight_loss_count : ℕ) : ℚ :=
  weight_loss_count / total_volunteers

/-- Theorem stating that the probability of weight loss is 0.241 given the provided data. -/
theorem weight_loss_probability_is_0_241 
  (total_volunteers : ℕ) 
  (weight_loss_count : ℕ) 
  (h1 : total_volunteers = 1000) 
  (h2 : weight_loss_count = 241) : 
  probability_of_weight_loss total_volunteers weight_loss_count = 241 / 1000 := by
  sorry

end weight_loss_probability_is_0_241_l3242_324244


namespace monomino_position_l3242_324215

/-- Represents a position on an 8x8 board -/
def Position := Fin 8 × Fin 8

/-- Represents a tromino (3x1 rectangle) -/
def Tromino := List Position

/-- Represents a monomino (1x1 square) -/
def Monomino := Position

/-- Represents a coloring of the board -/
def Coloring := Position → Fin 3

/-- The first coloring pattern -/
def coloring1 : Coloring := sorry

/-- The second coloring pattern -/
def coloring2 : Coloring := sorry

/-- Checks if a tromino is valid (covers exactly one square of each color) -/
def isValidTromino (t : Tromino) (c : Coloring) : Prop := sorry

/-- Checks if a set of trominos and a monomino form a valid covering of the board -/
def isValidCovering (trominos : List Tromino) (monomino : Monomino) : Prop := sorry

theorem monomino_position (trominos : List Tromino) (monomino : Monomino) :
  isValidCovering trominos monomino →
  coloring1 monomino = 1 ∧ coloring2 monomino = 1 := by sorry

end monomino_position_l3242_324215


namespace final_result_l3242_324206

def initial_value : ℕ := 10^8

def operation (n : ℕ) : ℕ := n * 3 / 2

def repeated_operation (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | k + 1 => repeated_operation (operation n) k

theorem final_result :
  repeated_operation initial_value 16 = 3^16 * 5^8 := by
  sorry

end final_result_l3242_324206


namespace skew_lines_distance_and_angle_l3242_324250

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary operations and relations
variable (distance : Line → Line → ℝ)
variable (distancePointToLine : Plane → Point → Line → ℝ)
variable (orthogonalProjection : Line → Plane → Line)
variable (angle : Line → Line → ℝ)
variable (perpendicular : Plane → Line → Prop)
variable (intersect : Plane → Line → Point → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem skew_lines_distance_and_angle 
  (a b : Line) (α : Plane) (A : Point) :
  skew a b →
  perpendicular α a →
  intersect α a A →
  let b' := orthogonalProjection b α
  distance a b = distancePointToLine α A b' ∧
  angle b b' + angle a b = 90 := by
  sorry

end skew_lines_distance_and_angle_l3242_324250


namespace right_triangle_revolution_is_cone_l3242_324296

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Solid of revolution generated by rotating a right-angled triangle about one of its legs -/
def solidOfRevolution (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: The solid of revolution generated by rotating a right-angled triangle
    about one of its legs is a cone -/
theorem right_triangle_revolution_is_cone (t : RightTriangle) :
  ∃ (c : Cone), solidOfRevolution t = c :=
sorry

end right_triangle_revolution_is_cone_l3242_324296


namespace solve_equation_and_evaluate_l3242_324210

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 8 = 15 * x + 4) → (3 * (x + 10) = 26.4) := by sorry

end solve_equation_and_evaluate_l3242_324210


namespace parabola_intersection_theorem_l3242_324238

/-- Represents a parabola with equation y² = px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line with equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the chord length of intersection between a parabola and a line -/
def chordLength (parabola : Parabola) (line : Line) : ℝ := 
  sorry

/-- Theorem stating that if a parabola with equation y² = px intersects 
    the line y = x - 1 with a chord length of √10, then p = 1 -/
theorem parabola_intersection_theorem (parabola : Parabola) (line : Line) :
  line.m = 1 ∧ line.b = -1 → chordLength parabola line = Real.sqrt 10 → parabola.p = 1 := by
  sorry

end parabola_intersection_theorem_l3242_324238


namespace polynomial_product_equality_l3242_324257

theorem polynomial_product_equality (a b c : ℝ) :
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = (a - b) * (b - c) * (c - a) * (a + b + c) := by
  sorry

end polynomial_product_equality_l3242_324257


namespace simplify_trig_expression_l3242_324280

theorem simplify_trig_expression (θ : Real) (h : θ = 160 * π / 180) :
  Real.sqrt (1 - Real.sin θ ^ 2) = -Real.cos θ := by
  sorry

end simplify_trig_expression_l3242_324280


namespace impossible_to_get_50_51_l3242_324240

/-- Represents the operation of replacing consecutive numbers with their count -/
def replace_with_count (s : List ℕ) (start : ℕ) (len : ℕ) : List ℕ := sorry

/-- Checks if a list contains only the numbers 50 and 51 -/
def contains_only_50_51 (s : List ℕ) : Prop := sorry

/-- The initial sequence of numbers from 1 to 100 -/
def initial_sequence : List ℕ := List.range 100

/-- Represents the result of applying the operation multiple times -/
def apply_operations (s : List ℕ) : List ℕ := sorry

theorem impossible_to_get_50_51 :
  ¬∃ (result : List ℕ), (apply_operations initial_sequence = result) ∧ (contains_only_50_51 result) := by
  sorry

end impossible_to_get_50_51_l3242_324240


namespace no_real_solutions_l3242_324205

theorem no_real_solutions (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - 2| ≥ a) ↔ a ≤ 1 := by
  sorry

end no_real_solutions_l3242_324205


namespace garlic_cloves_used_l3242_324291

/-- Proves that the number of garlic cloves used for cooking is the difference between
    the initial number and the remaining number of cloves -/
theorem garlic_cloves_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
    (h1 : initial = 93)
    (h2 : remaining = 7)
    (h3 : used = initial - remaining) :
  used = 86 := by
  sorry

end garlic_cloves_used_l3242_324291


namespace space_filling_crystalline_structure_exists_l3242_324282

/-- A cell in a crystalline structure -/
inductive Cell
| Octahedron : Cell
| Tetrahedron : Cell

/-- A crystalline structure is a periodic arrangement of cells in space -/
structure CrystallineStructure :=
(cells : Set Cell)
(periodic : Bool)
(fillsSpace : Bool)

/-- The existence of a space-filling crystalline structure with octahedrons and tetrahedrons -/
theorem space_filling_crystalline_structure_exists :
  ∃ (c : CrystallineStructure), 
    c.cells = {Cell.Octahedron, Cell.Tetrahedron} ∧ 
    c.periodic = true ∧ 
    c.fillsSpace = true :=
sorry

end space_filling_crystalline_structure_exists_l3242_324282


namespace exponent_division_l3242_324265

theorem exponent_division (a : ℝ) : a^7 / a^3 = a^4 := by
  sorry

end exponent_division_l3242_324265


namespace greatest_three_digit_multiple_of_23_l3242_324270

theorem greatest_three_digit_multiple_of_23 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 23 ∣ n → n ≤ 989 := by sorry

end greatest_three_digit_multiple_of_23_l3242_324270


namespace least_third_side_length_l3242_324288

theorem least_third_side_length (a b c : ℝ) : 
  a = 8 → b = 6 → c > 0 → a^2 + b^2 ≤ c^2 → c ≥ 10 := by
  sorry

end least_third_side_length_l3242_324288


namespace percentage_repeated_approx_l3242_324299

/-- The count of five-digit numbers -/
def total_five_digit_numbers : ℕ := 90000

/-- The count of five-digit numbers without repeated digits -/
def unique_digit_numbers : ℕ := 9 * 9 * 8 * 7 * 6

/-- The count of five-digit numbers with at least one repeated digit -/
def repeated_digit_numbers : ℕ := total_five_digit_numbers - unique_digit_numbers

/-- The percentage of five-digit numbers with at least one repeated digit -/
def percentage_repeated : ℚ := (repeated_digit_numbers : ℚ) / (total_five_digit_numbers : ℚ) * 100

theorem percentage_repeated_approx :
  ∃ ε > 0, ε < 0.1 ∧ |percentage_repeated - 69.8| < ε :=
sorry

end percentage_repeated_approx_l3242_324299


namespace exists_construction_with_1001_free_endpoints_l3242_324293

/-- Represents a point in the construction --/
structure Point :=
  (depth : ℕ)
  (branches : Fin 5)

/-- Represents the construction of line segments --/
def Construction := List Point

/-- Counts the number of free endpoints in a construction --/
def count_free_endpoints (c : Construction) : ℕ := sorry

/-- Theorem: There exists a construction with 1001 free endpoints --/
theorem exists_construction_with_1001_free_endpoints :
  ∃ (c : Construction), count_free_endpoints c = 1001 := by sorry

end exists_construction_with_1001_free_endpoints_l3242_324293


namespace m_range_theorem_l3242_324251

def prop_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

def m_range (m : ℝ) : Prop :=
  (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem m_range_theorem :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) → m_range m :=
by sorry

end m_range_theorem_l3242_324251


namespace hexagon_area_ratio_l3242_324279

theorem hexagon_area_ratio (a b : ℝ) (ha : a > 0) (hb : b = 2 * a) :
  (3 * Real.sqrt 3 / 2 * a^2) / (3 * Real.sqrt 3 / 2 * b^2) = 1 / 4 := by
  sorry

end hexagon_area_ratio_l3242_324279


namespace farmers_market_spending_l3242_324217

theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ) : 
  sandi_initial = 600 →
  gillian_total = 1050 →
  ∃ (multiple : ℕ), 
    gillian_total = (sandi_initial / 2) + (multiple * (sandi_initial / 2)) + 150 ∧
    multiple = 1 :=
by sorry

end farmers_market_spending_l3242_324217


namespace triangle_side_expression_l3242_324278

/-- Given a triangle with sides a, b, and c, 
    prove that |a-b-c| + |b-c-a| + |c+a-b| = 3c + a - b -/
theorem triangle_side_expression 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq1 : a + b > c) 
  (h_ineq2 : b + c > a) 
  (h_ineq3 : c + a > b) : 
  |a - b - c| + |b - c - a| + |c + a - b| = 3 * c + a - b :=
sorry

end triangle_side_expression_l3242_324278


namespace count_perfect_square_factors_l3242_324256

/-- The number of factors of 18000 that are perfect squares -/
def num_perfect_square_factors : ℕ := 8

/-- The prime factorization of 18000 -/
def factorization_18000 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 3)]

/-- Theorem: The number of factors of 18000 that are perfect squares is 8 -/
theorem count_perfect_square_factors :
  (List.prod (factorization_18000.map (fun (p, e) => e + 1)) / 8 : ℚ).num = num_perfect_square_factors := by
  sorry

end count_perfect_square_factors_l3242_324256


namespace car_down_payment_l3242_324261

/-- Given a total down payment to be split equally among a number of people,
    rounding up to the nearest dollar, calculate the amount each person must pay. -/
def splitPayment (total : ℕ) (people : ℕ) : ℕ :=
  (total + people - 1) / people

theorem car_down_payment :
  splitPayment 3500 3 = 1167 := by
  sorry

end car_down_payment_l3242_324261


namespace ship_length_in_emily_steps_l3242_324202

/-- The length of the ship in terms of Emily's steps -/
def ship_length : ℕ := 70

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 210

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 42

/-- Emily's walking speed is faster than the ship's speed -/
axiom emily_faster : ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0

/-- Theorem stating the length of the ship in terms of Emily's steps -/
theorem ship_length_in_emily_steps :
  ∃ (e s : ℝ), e > s ∧ e > 0 ∧ s > 0 →
  (steps_back_to_front : ℝ) * e = ship_length + steps_back_to_front * s ∧
  (steps_front_to_back : ℝ) * e = ship_length - steps_front_to_back * s →
  ship_length = 70 := by
  sorry

end ship_length_in_emily_steps_l3242_324202


namespace inequality_not_true_l3242_324267

theorem inequality_not_true (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ¬((2 * a * b) / (a + b) > Real.sqrt (a * b)) := by
  sorry

end inequality_not_true_l3242_324267


namespace wire_length_problem_l3242_324214

theorem wire_length_problem (total_wires : ℕ) (avg_length : ℝ) (third_avg_length : ℝ) :
  total_wires = 6 →
  avg_length = 80 →
  third_avg_length = 70 →
  let total_length := total_wires * avg_length
  let third_wires := total_wires / 3
  let third_total_length := third_wires * third_avg_length
  let remaining_wires := total_wires - third_wires
  let remaining_length := total_length - third_total_length
  remaining_length / remaining_wires = 85 := by
sorry

end wire_length_problem_l3242_324214


namespace winter_uniform_count_l3242_324222

/-- The number of packages of winter uniforms delivered -/
def num_packages : ℕ := 10

/-- The number of dozens per package -/
def dozens_per_package : ℕ := 10

/-- The number of sets per dozen -/
def sets_per_dozen : ℕ := 12

/-- The total number of winter uniform sets -/
def total_sets : ℕ := num_packages * dozens_per_package * sets_per_dozen

theorem winter_uniform_count : total_sets = 1200 := by
  sorry

end winter_uniform_count_l3242_324222


namespace quadratic_properties_l3242_324289

theorem quadratic_properties (a b c : ℝ) (ha : a ≠ 0) :
  (∃ m n : ℝ, m ≠ n ∧ a * m^2 + b * m + c = a * n^2 + b * n + c) ∧
  (a * c < 0 → ∃ m n : ℝ, m > n ∧ a * m^2 + b * m + c < 0 ∧ 0 < a * n^2 + b * n + c) ∧
  (a * b > 0 → ∃ p q : ℝ, p ≠ q ∧ a * p^2 + b * p + c = a * q^2 + b * q + c ∧ p + q < 0) :=
by sorry

end quadratic_properties_l3242_324289


namespace quadratic_one_solution_l3242_324248

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, (3 * x - 4) * (x + 6) = -53 + k * x) ↔ 
  (k = 14 + 2 * Real.sqrt 87 ∨ k = 14 - 2 * Real.sqrt 87) :=
by sorry

end quadratic_one_solution_l3242_324248


namespace cube_volume_from_total_edge_length_l3242_324207

/-- The volume of a cube with total edge length of 72 feet is 216 cubic feet. -/
theorem cube_volume_from_total_edge_length :
  ∀ (s : ℝ), (12 * s = 72) → s^3 = 216 := by
  sorry

end cube_volume_from_total_edge_length_l3242_324207


namespace batsman_average_after_12th_inning_l3242_324281

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (b : Batsman) (runsInLastInning : ℕ) : ℚ :=
  (b.totalRuns + runsInLastInning : ℚ) / (b.innings + 1)

theorem batsman_average_after_12th_inning 
  (b : Batsman) 
  (h1 : b.innings = 11)
  (h2 : newAverage b 60 = b.average + 4) :
  newAverage b 60 = 16 := by
sorry

end batsman_average_after_12th_inning_l3242_324281


namespace gcd_175_100_65_l3242_324274

theorem gcd_175_100_65 : Nat.gcd 175 (Nat.gcd 100 65) = 5 := by
  sorry

end gcd_175_100_65_l3242_324274


namespace matrix_subtraction_result_l3242_324258

theorem matrix_subtraction_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 8]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 5; -3, 6]
  A - B = !![3, -8; 5, 2] := by sorry

end matrix_subtraction_result_l3242_324258


namespace scholarship_theorem_l3242_324285

def scholarship_problem (wendy_last_year : ℝ) : Prop :=
  let kelly_last_year := 2 * wendy_last_year
  let nina_last_year := kelly_last_year - 8000
  let jason_last_year := 3/4 * kelly_last_year
  let wendy_this_year := wendy_last_year * 1.1
  let kelly_this_year := kelly_last_year * 1.08
  let nina_this_year := nina_last_year * 1.15
  let jason_this_year := jason_last_year * 1.12
  let total_this_year := wendy_this_year + kelly_this_year + nina_this_year + jason_this_year
  wendy_last_year = 20000 → total_this_year = 135600

theorem scholarship_theorem : scholarship_problem 20000 := by
  sorry

end scholarship_theorem_l3242_324285


namespace expected_value_of_coins_l3242_324219

-- Define coin values in cents
def penny : ℚ := 1
def nickel : ℚ := 5
def dime : ℚ := 10
def quarter : ℚ := 25
def half_dollar : ℚ := 50

-- Define probability of heads
def prob_heads : ℚ := 1/2

-- Define function to calculate expected value for a single coin
def expected_value (coin_value : ℚ) : ℚ := prob_heads * coin_value

-- Theorem statement
theorem expected_value_of_coins : 
  expected_value penny + expected_value nickel + expected_value dime + 
  expected_value quarter + expected_value half_dollar = 45.5 := by
  sorry

end expected_value_of_coins_l3242_324219


namespace expression_upper_bound_l3242_324204

theorem expression_upper_bound (a b c d : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (h_prod : a * c = b * d)
  (h_sum : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b) ≤ 4 ∧ 
  ∃ (a' b' c' d' : ℝ), a' / c' + c' / a' + b' / d' + d' / b' = 4 :=
sorry

end expression_upper_bound_l3242_324204


namespace adam_new_books_l3242_324287

theorem adam_new_books (initial_books sold_books final_books : ℕ) 
  (h1 : initial_books = 33) 
  (h2 : sold_books = 11)
  (h3 : final_books = 45) :
  final_books - (initial_books - sold_books) = 23 := by
  sorry

end adam_new_books_l3242_324287


namespace range_of_a_l3242_324242

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - a)^2 < 1}

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : 2 ∈ A a) (h2 : 3 ∉ A a) : 1 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l3242_324242


namespace exists_common_point_l3242_324203

/-- Represents a rectangular map with a scale factor -/
structure Map where
  scale : ℝ
  width : ℝ
  height : ℝ

/-- Represents a point on a map -/
structure MapPoint where
  x : ℝ
  y : ℝ

/-- Theorem stating that there exists a common point on two maps of different scales -/
theorem exists_common_point (map1 map2 : Map) (h_scale : map2.scale = 5 * map1.scale) :
  ∃ (p1 : MapPoint) (p2 : MapPoint),
    p1.x / map1.width = p2.x / map2.width ∧
    p1.y / map1.height = p2.y / map2.height :=
sorry

end exists_common_point_l3242_324203


namespace dividend_rate_calculation_l3242_324259

/-- Given a share worth 48 rupees, with a desired interest rate of 12%,
    and a market value of 36.00000000000001 rupees, the dividend rate is 16%. -/
theorem dividend_rate_calculation (share_value : ℝ) (interest_rate : ℝ) (market_value : ℝ)
    (h1 : share_value = 48)
    (h2 : interest_rate = 0.12)
    (h3 : market_value = 36.00000000000001) :
    (share_value * interest_rate / market_value) * 100 = 16 := by
  sorry

end dividend_rate_calculation_l3242_324259


namespace exists_between_elements_l3242_324227

/-- A sequence of natural numbers where each natural number appears exactly once -/
def UniqueNatSequence : Type := ℕ → ℕ

/-- The property that each natural number appears exactly once in the sequence -/
def isUniqueNatSequence (a : UniqueNatSequence) : Prop :=
  (∀ n : ℕ, ∃ k : ℕ, a k = n) ∧ 
  (∀ m n : ℕ, a m = a n → m = n)

/-- The main theorem -/
theorem exists_between_elements (a : UniqueNatSequence) (h : isUniqueNatSequence a) :
  ∀ n : ℕ, ∃ k : ℕ, k < n ∧ (a (n - k) < a n ∧ a n < a (n + k)) :=
by sorry

end exists_between_elements_l3242_324227


namespace vivienne_phone_count_l3242_324208

theorem vivienne_phone_count : ∀ (v : ℕ), 
  (400 * v + 400 * (v + 10) = 36000) → v = 40 := by
  sorry

end vivienne_phone_count_l3242_324208


namespace min_value_reciprocal_sum_l3242_324213

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m * n > 0) (h2 : m + n = 2) :
  (1 / m + 1 / n) ≥ 2 := by
  sorry

end min_value_reciprocal_sum_l3242_324213


namespace isosceles_right_triangles_are_similar_l3242_324226

/-- An isosceles right triangle is a triangle with two equal sides and a right angle. -/
structure IsoscelesRightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right_angle : side1^2 + side2^2 = hypotenuse^2
  is_isosceles : side1 = side2

/-- Two triangles are similar if their corresponding angles are equal and the ratios of corresponding sides are equal. -/
def are_similar (t1 t2 : IsoscelesRightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.hypotenuse = k * t2.hypotenuse

/-- Theorem: Any two isosceles right triangles are similar. -/
theorem isosceles_right_triangles_are_similar (t1 t2 : IsoscelesRightTriangle) : 
  are_similar t1 t2 := by
  sorry

end isosceles_right_triangles_are_similar_l3242_324226


namespace square_mod_five_not_three_l3242_324272

theorem square_mod_five_not_three (n : ℕ) : (n ^ 2) % 5 ≠ 3 := by
  sorry

end square_mod_five_not_three_l3242_324272


namespace solution_equivalence_l3242_324211

theorem solution_equivalence (x : ℝ) : 
  (3/10 : ℝ) + |x - 7/20| < 4/15 ↔ x ∈ Set.Ioo (19/60 : ℝ) (23/60 : ℝ) :=
by sorry

end solution_equivalence_l3242_324211


namespace circle_equation_k_l3242_324298

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
theorem circle_equation_k (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 10*y - k = 0 ↔ (x + 4)^2 + (y + 5)^2 = 25) → 
  k = -16 := by
sorry

end circle_equation_k_l3242_324298


namespace not_all_positive_l3242_324209

theorem not_all_positive (a b c : ℝ) 
  (sum_eq : a + b + c = 4)
  (sum_sq_eq : a^2 + b^2 + c^2 = 12)
  (prod_eq : a * b * c = 1) :
  ¬(a > 0 ∧ b > 0 ∧ c > 0) := by
  sorry

end not_all_positive_l3242_324209


namespace conference_room_seating_l3242_324223

/-- Represents the seating arrangement in a conference room. -/
structure ConferenceRoom where
  totalPeople : ℕ
  rowCapacities : List ℕ
  allSeatsFilled : totalPeople = rowCapacities.sum

/-- Checks if a conference room arrangement is valid. -/
def isValidArrangement (room : ConferenceRoom) : Prop :=
  ∀ capacity ∈ room.rowCapacities, capacity = 9 ∨ capacity = 10

/-- The main theorem about the conference room seating arrangement. -/
theorem conference_room_seating
  (room : ConferenceRoom)
  (validArrangement : isValidArrangement room)
  (h : room.totalPeople = 54) :
  (room.rowCapacities.filter (· = 10)).length = 0 := by
  sorry


end conference_room_seating_l3242_324223


namespace infinitely_many_non_representable_l3242_324245

theorem infinitely_many_non_representable : 
  Set.Infinite {n : ℤ | ∀ (a b c : ℕ), n ≠ 2^a + 3^b - 5^c} := by
  sorry

end infinitely_many_non_representable_l3242_324245


namespace goldfish_count_l3242_324241

/-- Given that 25% of goldfish are at the surface and 75% are below the surface,
    with 45 goldfish below the surface, prove that there are 15 goldfish at the surface. -/
theorem goldfish_count (surface_percent : ℝ) (below_percent : ℝ) (below_count : ℕ) :
  surface_percent = 25 →
  below_percent = 75 →
  below_count = 45 →
  ↑below_count / below_percent * surface_percent = 15 :=
by sorry

end goldfish_count_l3242_324241


namespace regular_polygon_perimeter_l3242_324220

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (sides : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 2 ∧
  sides = n ∧
  side_length = 7 ∧
  exterior_angle = 90 ∧
  (360 : ℝ) / n = exterior_angle →
  sides * side_length = 28 := by
  sorry

end regular_polygon_perimeter_l3242_324220


namespace bottles_produced_l3242_324246

-- Define the production rate of 4 machines
def production_rate_4 : ℕ := 16

-- Define the number of minutes
def minutes : ℕ := 3

-- Define the number of machines in the first scenario
def machines_1 : ℕ := 4

-- Define the number of machines in the second scenario
def machines_2 : ℕ := 8

-- Theorem to prove
theorem bottles_produced :
  (machines_2 * minutes * (production_rate_4 / machines_1)) = 96 := by
  sorry

end bottles_produced_l3242_324246


namespace diophantine_equation_unique_solution_l3242_324262

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, a^2 = 2*b^2 + 3*c^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end diophantine_equation_unique_solution_l3242_324262


namespace consecutive_square_differences_exist_l3242_324247

theorem consecutive_square_differences_exist : 
  ∃ (a b c : ℝ), 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
    (a > 2022 ∨ b > 2022 ∨ c > 2022) ∧
    (∃ (k : ℤ), 
      (a^2 - b^2 = k) ∧ 
      (b^2 - c^2 = k + 1) ∧ 
      (c^2 - a^2 = k + 2)) :=
by sorry

end consecutive_square_differences_exist_l3242_324247


namespace total_value_correct_l3242_324268

/-- The total value of an imported item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The tax-free threshold -/
def tax_free_threshold : ℝ := 1000

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that the total value is correct given the conditions -/
theorem total_value_correct : 
  tax_rate * (total_value - tax_free_threshold) = tax_paid := by sorry

end total_value_correct_l3242_324268


namespace cubic_roots_sum_l3242_324201

theorem cubic_roots_sum (x y z : ℝ) : 
  (x^3 - 2*x^2 - 9*x - 1 = 0) →
  (y^3 - 2*y^2 - 9*y - 1 = 0) →
  (z^3 - 2*z^2 - 9*z - 1 = 0) →
  (y*z/x + x*z/y + x*y/z = 77) :=
by sorry

end cubic_roots_sum_l3242_324201


namespace range_of_4a_minus_2b_l3242_324273

theorem range_of_4a_minus_2b (a b : ℝ) 
  (h1 : -1 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 2 ≤ a + b ∧ a + b ≤ 4) : 
  ∃ (x : ℝ), -1 ≤ x ∧ x ≤ 7 ∧ x = 4*a - 2*b :=
by sorry

end range_of_4a_minus_2b_l3242_324273


namespace inequality_theorem_l3242_324228

theorem inequality_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end inequality_theorem_l3242_324228


namespace aron_vacuuming_days_l3242_324218

/-- The number of days Aron spends vacuuming each week -/
def vacuuming_days : ℕ := sorry

/-- The time spent vacuuming per day in minutes -/
def vacuuming_time_per_day : ℕ := 30

/-- The time spent dusting per day in minutes -/
def dusting_time_per_day : ℕ := 20

/-- The number of days Aron spends dusting each week -/
def dusting_days : ℕ := 2

/-- The total cleaning time per week in minutes -/
def total_cleaning_time : ℕ := 130

theorem aron_vacuuming_days :
  vacuuming_days * vacuuming_time_per_day +
  dusting_days * dusting_time_per_day =
  total_cleaning_time ∧
  vacuuming_days = 3 :=
by sorry

end aron_vacuuming_days_l3242_324218


namespace nancy_balloons_l3242_324221

theorem nancy_balloons (nancy_balloons : ℕ) (mary_balloons : ℕ) : 
  mary_balloons = 28 → mary_balloons = 4 * nancy_balloons → nancy_balloons = 7 := by
  sorry

end nancy_balloons_l3242_324221


namespace min_value_inequality_l3242_324216

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, |a - x| + |x + b| + c ≥ 1) →
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end min_value_inequality_l3242_324216


namespace parabola_vertex_l3242_324286

/-- Given a quadratic function f(x) = -x^2 + cx + d where the solution to f(x) ≤ 0
    is (-∞, -5] ∪ [1, ∞), the vertex of the parabola defined by f(x) is (3, 4). -/
theorem parabola_vertex (c d : ℝ) :
  let f : ℝ → ℝ := λ x => -x^2 + c*x + d
  (∀ x, f x ≤ 0 ↔ x ≤ -5 ∨ x ≥ 1) →
  (∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = 4 ∧ ∀ x, f x ≤ f p.1) :=
by sorry

end parabola_vertex_l3242_324286


namespace expression_equality_l3242_324290

theorem expression_equality : (45 + 15)^2 - 3 * (45^2 + 15^2 - 2 * 45 * 15) = 900 := by
  sorry

end expression_equality_l3242_324290


namespace triangle_side_length_l3242_324252

theorem triangle_side_length (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2) (h3 : A = π/6) :
  let b := Real.sqrt ((a^2 + c^2 - 2*a*c*(Real.cos A)) / (Real.sin A)^2)
  b = 2 * Real.sqrt 3 := by sorry

end triangle_side_length_l3242_324252


namespace value_of_a_fourth_plus_reciprocal_l3242_324224

theorem value_of_a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + 1/a^4 = 7 := by
sorry

end value_of_a_fourth_plus_reciprocal_l3242_324224


namespace lives_lost_l3242_324277

theorem lives_lost (initial_lives gained_lives final_lives : ℕ) : 
  initial_lives = 43 → gained_lives = 27 → final_lives = 56 → 
  ∃ (lost_lives : ℕ), initial_lives - lost_lives + gained_lives = final_lives ∧ lost_lives = 14 := by
sorry

end lives_lost_l3242_324277


namespace y_minus_x_value_l3242_324294

theorem y_minus_x_value (x y z : ℚ) 
  (eq1 : x + y + z = 12)
  (eq2 : x + y = 8)
  (eq3 : y - 3*x + z = 9) :
  y - x = 13/2 := by
sorry

end y_minus_x_value_l3242_324294


namespace circle_properties_l3242_324229

theorem circle_properties (A : ℝ) (h : A = 4 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * r = 4 ∧ 2 * Real.pi * r = 4 * Real.pi :=
by
  sorry

end circle_properties_l3242_324229
