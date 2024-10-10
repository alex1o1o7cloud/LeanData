import Mathlib

namespace power_tower_mod_500_l1203_120364

theorem power_tower_mod_500 : 4^(4^(4^4)) ≡ 36 [ZMOD 500] := by sorry

end power_tower_mod_500_l1203_120364


namespace roots_can_change_l1203_120341

-- Define the concept of a root being lost
def root_lost (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = g x ∧ ¬(Real.tan (f x) = Real.tan (g x))

-- Define the concept of an extraneous root appearing
def extraneous_root (f g : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x ≠ g x ∧ Real.tan (f x) = Real.tan (g x)

-- Theorem stating that roots can be lost and extraneous roots can appear
theorem roots_can_change (f g : ℝ → ℝ) : 
  (root_lost f g) ∧ (extraneous_root f g) := by
  sorry


end roots_can_change_l1203_120341


namespace min_value_expression_l1203_120388

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 1) :
  ∃ (min_val : ℝ), 
    (∀ c', c' > 2 → (3*a*c'/b + c'/(a*b) + 6/(c'-2) ≥ min_val)) ∧ 
    (∃ c'', c'' > 2 ∧ 3*a*c''/b + c''/(a*b) + 6/(c''-2) = min_val) ∧
    min_val = 1 / (a * (1 - a)) :=
by sorry

end min_value_expression_l1203_120388


namespace perfect_square_trinomial_m_value_l1203_120389

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, a * x^2 + b * x + c = (p * x + q)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, IsPerfectSquareTrinomial 4 m 121 → (m = 44 ∨ m = -44) :=
by
  sorry


end perfect_square_trinomial_m_value_l1203_120389


namespace horizontal_row_different_l1203_120394

/-- Represents the weight of a row of apples -/
def RowWeight : Type := ℝ

/-- Represents the arrangement of apples -/
structure AppleArrangement where
  total_apples : ℕ
  rows : ℕ
  apples_per_row : ℕ
  diagonal_weights : Fin 3 → RowWeight
  vertical_weights : Fin 3 → RowWeight
  horizontal_weight : RowWeight

/-- The given arrangement of apples satisfies the problem conditions -/
def valid_arrangement (a : AppleArrangement) : Prop :=
  a.total_apples = 9 ∧
  a.rows = 10 ∧
  a.apples_per_row = 3 ∧
  ∃ (t : RowWeight),
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧
    (∀ i : Fin 3, a.vertical_weights i = t) ∧
    a.horizontal_weight ≠ t

theorem horizontal_row_different (a : AppleArrangement) 
  (h : valid_arrangement a) : 
  ∃ (t : RowWeight), 
    (∀ i : Fin 3, a.diagonal_weights i = t) ∧ 
    (∀ i : Fin 3, a.vertical_weights i = t) ∧ 
    a.horizontal_weight ≠ t := by
  sorry

#check horizontal_row_different

end horizontal_row_different_l1203_120394


namespace point_position_on_line_l1203_120351

/-- Given five points on a line, prove the position of a point P satisfying a specific ratio condition -/
theorem point_position_on_line (a b c d : ℝ) :
  let O := (0 : ℝ)
  let A := (2*a : ℝ)
  let B := (3*b : ℝ)
  let C := (4*c : ℝ)
  let D := (5*d : ℝ)
  ∃ P : ℝ, B ≤ P ∧ P ≤ C ∧
    (P - A)^2 * (C - P) = (D - P)^2 * (P - B) →
    P = (8*a*c - 15*b*d) / (8*c - 15*d - 6*b + 4*a) :=
by sorry

end point_position_on_line_l1203_120351


namespace biscuits_butter_cookies_difference_l1203_120328

-- Define the number of cookies baked in the morning and afternoon
def morning_butter_cookies : ℕ := 20
def morning_biscuits : ℕ := 40
def afternoon_butter_cookies : ℕ := 10
def afternoon_biscuits : ℕ := 20

-- Define the total number of each type of cookie
def total_butter_cookies : ℕ := morning_butter_cookies + afternoon_butter_cookies
def total_biscuits : ℕ := morning_biscuits + afternoon_biscuits

-- Theorem statement
theorem biscuits_butter_cookies_difference :
  total_biscuits - total_butter_cookies = 30 := by
  sorry

end biscuits_butter_cookies_difference_l1203_120328


namespace point_on_y_axis_l1203_120338

/-- If a point P(a-1, a^2-9) lies on the y-axis, then its coordinates are (0, -8). -/
theorem point_on_y_axis (a : ℝ) :
  (a - 1 = 0) → (a - 1, a^2 - 9) = (0, -8) := by
  sorry

end point_on_y_axis_l1203_120338


namespace exam_pass_rate_l1203_120350

theorem exam_pass_rate (hindi : ℝ) (english : ℝ) (math : ℝ) 
  (hindi_english : ℝ) (hindi_math : ℝ) (english_math : ℝ) (all_three : ℝ)
  (h1 : hindi = 25) (h2 : english = 48) (h3 : math = 35)
  (h4 : hindi_english = 27) (h5 : hindi_math = 20) (h6 : english_math = 15)
  (h7 : all_three = 10) :
  100 - (hindi + english + math - hindi_english - hindi_math - english_math + all_three) = 44 := by
  sorry

end exam_pass_rate_l1203_120350


namespace sqrt_meaningful_range_l1203_120386

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by
sorry

end sqrt_meaningful_range_l1203_120386


namespace triangle_angle_sine_equivalence_l1203_120397

theorem triangle_angle_sine_equivalence (A B C : Real) (h : A > 0 ∧ B > 0 ∧ C > 0) :
  (A > B ↔ Real.sin A > Real.sin B) :=
sorry

end triangle_angle_sine_equivalence_l1203_120397


namespace smallest_angle_satisfying_condition_l1203_120308

theorem smallest_angle_satisfying_condition : 
  ∃ (x : ℝ), x > 0 ∧ x < (π / 180) * 360 ∧ 
  Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
  (∀ (y : ℝ), 0 < y ∧ y < x → 
    Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y)) ∧
  x = (π / 180) * 10 :=
sorry

end smallest_angle_satisfying_condition_l1203_120308


namespace geometric_sequence_a1_l1203_120348

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 = 1 →
  (a 5 + (3/2) * a 4) / 2 = 1/2 →
  a 1 = 4 := by
  sorry


end geometric_sequence_a1_l1203_120348


namespace rectangle_fit_impossibility_l1203_120313

theorem rectangle_fit_impossibility : 
  ∀ (a b c d : ℝ), 
    a = 5 ∧ b = 6 ∧ c = 3 ∧ d = 8 → 
    (c^2 + d^2 : ℝ) > (a^2 + b^2 : ℝ) :=
by
  sorry

end rectangle_fit_impossibility_l1203_120313


namespace quadratic_solution_sum_l1203_120304

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 11 = 25) → 
  (d^2 - 6*d + 11 = 25) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 23 := by
sorry

end quadratic_solution_sum_l1203_120304


namespace complex_fraction_equals_point_l1203_120352

theorem complex_fraction_equals_point : (2 * Complex.I) / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_fraction_equals_point_l1203_120352


namespace largest_inscribed_right_triangle_area_l1203_120315

/-- The area of the largest inscribed right triangle in a circle -/
theorem largest_inscribed_right_triangle_area (r : ℝ) (h : r = 8) :
  let circle_area := π * r^2
  let diameter := 2 * r
  let max_triangle_area := (diameter * r) / 2
  max_triangle_area = 64 := by
  sorry

end largest_inscribed_right_triangle_area_l1203_120315


namespace intersection_points_roots_l1203_120325

theorem intersection_points_roots (x y : ℝ) : 
  (∃ x, x^2 - 3*x = 0 ∧ x ≠ 0 ∧ x ≠ 3) ∨
  (∀ x, x = x - 3 → x^2 - 3*x ≠ 0) :=
by sorry

#check intersection_points_roots

end intersection_points_roots_l1203_120325


namespace roots_of_h_l1203_120312

/-- Given that x = 1 is a root of f(x) = a/x + b and a ≠ 0, 
    prove that the roots of h(x) = ax^2 + bx are 0 and 1. -/
theorem roots_of_h (a b : ℝ) (ha : a ≠ 0) 
  (hf : a / 1 + b = 0) : 
  ∀ x : ℝ, ax^2 + bx = 0 ↔ x = 0 ∨ x = 1 := by
sorry

end roots_of_h_l1203_120312


namespace phone_reps_calculation_l1203_120373

/-- The number of hours each phone rep works per day -/
def hours_per_day : ℕ := 8

/-- The hourly wage of each phone rep in dollars -/
def hourly_wage : ℚ := 14

/-- The number of days worked -/
def days_worked : ℕ := 5

/-- The total payment for all new employees after 5 days in dollars -/
def total_payment : ℚ := 28000

/-- The number of new phone reps the company wants to hire -/
def num_phone_reps : ℕ := 50

theorem phone_reps_calculation :
  (hours_per_day * hourly_wage * days_worked : ℚ) * num_phone_reps = total_payment :=
by sorry

end phone_reps_calculation_l1203_120373


namespace carpooling_arrangements_count_l1203_120330

/-- Represents the last digit of a license plate -/
inductive LicensePlateEnding
| Nine
| Zero
| Two
| One
| Five

/-- Represents a day in the carpooling period -/
inductive Day
| Five
| Six
| Seven
| Eight
| Nine

def is_odd_day (d : Day) : Bool :=
  match d with
  | Day.Five | Day.Seven | Day.Nine => true
  | _ => false

def is_even_ending (e : LicensePlateEnding) : Bool :=
  match e with
  | LicensePlateEnding.Zero | LicensePlateEnding.Two => true
  | _ => false

def is_valid_car (d : Day) (e : LicensePlateEnding) : Bool :=
  (is_odd_day d && !is_even_ending e) || (!is_odd_day d && is_even_ending e)

/-- Represents a carpooling arrangement for the 5-day period -/
def CarpoolingArrangement := Day → LicensePlateEnding

def is_valid_arrangement (arr : CarpoolingArrangement) : Prop :=
  (∀ d, is_valid_car d (arr d)) ∧
  (∃! d, arr d = LicensePlateEnding.Nine)

def number_of_arrangements : ℕ := sorry

theorem carpooling_arrangements_count :
  number_of_arrangements = 80 := by sorry

end carpooling_arrangements_count_l1203_120330


namespace imaginary_part_of_z_l1203_120327

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Real.sqrt 2 + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end imaginary_part_of_z_l1203_120327


namespace line_segment_endpoint_l1203_120316

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((2 - (-6))^2 + (y - 5)^2) = 10 → 
  y = 11 := by
sorry

end line_segment_endpoint_l1203_120316


namespace stanley_tires_l1203_120322

/-- The number of tires Stanley bought -/
def num_tires : ℕ := 240 / 60

/-- The cost of each tire in dollars -/
def cost_per_tire : ℕ := 60

/-- The total amount Stanley spent in dollars -/
def total_spent : ℕ := 240

theorem stanley_tires :
  num_tires = 4 ∧ cost_per_tire * num_tires = total_spent :=
sorry

end stanley_tires_l1203_120322


namespace min_value_sum_min_value_achievable_l1203_120347

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) = 1 / Real.rpow 2 (1/3) :=
sorry

end min_value_sum_min_value_achievable_l1203_120347


namespace no_valid_tetrahedron_labeling_l1203_120375

/-- Represents a labeling of a tetrahedron's vertices -/
def TetrahedronLabeling := Fin 4 → Fin 4

/-- Checks if a labeling uses each number exactly once -/
def is_valid_labeling (l : TetrahedronLabeling) : Prop :=
  ∀ i : Fin 4, ∃! j : Fin 4, l j = i

/-- Represents a face of the tetrahedron -/
def Face := Fin 3 → Fin 4

/-- Gets the sum of labels on a face -/
def face_sum (l : TetrahedronLabeling) (f : Face) : Nat :=
  (f 0).val + (f 1).val + (f 2).val

/-- Checks if all faces have the same sum -/
def all_faces_equal_sum (l : TetrahedronLabeling) : Prop :=
  ∀ f₁ f₂ : Face, face_sum l f₁ = face_sum l f₂

theorem no_valid_tetrahedron_labeling :
  ¬∃ l : TetrahedronLabeling, is_valid_labeling l ∧ all_faces_equal_sum l := by
  sorry

end no_valid_tetrahedron_labeling_l1203_120375


namespace smallest_item_is_a5_l1203_120369

def sequence_a (n : ℕ) : ℚ :=
  2 * n^2 - 21 * n + 40

theorem smallest_item_is_a5 :
  ∀ n : ℕ, n ≥ 1 → sequence_a 5 ≤ sequence_a n :=
sorry

end smallest_item_is_a5_l1203_120369


namespace parallel_vectors_sum_l1203_120374

/-- Given two vectors a and b in ℝ³, where a = (-2, 3, 1) and b = (4, m, n),
    if a is parallel to b, then m + n = -8 -/
theorem parallel_vectors_sum (m n : ℝ) : 
  let a : ℝ × ℝ × ℝ := (-2, 3, 1)
  let b : ℝ × ℝ × ℝ := (4, m, n)
  (∃ (k : ℝ), b = k • a) → m + n = -8 := by
sorry

end parallel_vectors_sum_l1203_120374


namespace not_decreasing_on_interval_l1203_120354

-- Define a real-valued function on the real line
variable (f : ℝ → ℝ)

-- State the theorem
theorem not_decreasing_on_interval (h : f (-1) < f 1) :
  ¬(∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x ≤ y → f x ≥ f y) :=
by sorry

end not_decreasing_on_interval_l1203_120354


namespace experts_win_probability_l1203_120319

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Viewers winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def expert_score : ℕ := 3

/-- The current score of Viewers -/
def viewer_score : ℕ := 4

/-- The number of rounds needed to win the game -/
def winning_score : ℕ := 6

/-- The probability of Experts winning the game from the current state -/
theorem experts_win_probability : 
  p^4 + 4 * p^3 * q = 0.4752 := by sorry

end experts_win_probability_l1203_120319


namespace win_sector_area_l1203_120314

theorem win_sector_area (radius : ℝ) (win_probability : ℝ) (win_area : ℝ) : 
  radius = 8 → win_probability = 1/4 → win_area = 16 * Real.pi → 
  win_area = win_probability * Real.pi * radius^2 := by
  sorry

end win_sector_area_l1203_120314


namespace sin_deg_rad_solutions_l1203_120337

def sin_deg_rad_eq (x : ℝ) : Prop := Real.sin x = Real.sin (x * Real.pi / 180)

theorem sin_deg_rad_solutions :
  ∃ (S : Finset ℝ), S.card = 10 ∧
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 90 ∧ sin_deg_rad_eq x → x ∈ S) :=
by sorry

end sin_deg_rad_solutions_l1203_120337


namespace geometric_sequence_problem_l1203_120353

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  ∃ r : ℝ, r ≠ 0 ∧ b = 10 * r ∧ (3/4) = b * r → 
  b = 5 * Real.sqrt 3 / 3 := by
sorry

end geometric_sequence_problem_l1203_120353


namespace original_polygon_sides_l1203_120362

-- Define the number of sides of the original polygon
def n : ℕ := sorry

-- Define the sum of interior angles of the new polygon
def new_polygon_angle_sum : ℝ := 2520

-- Theorem statement
theorem original_polygon_sides :
  (n + 1 - 2) * 180 = new_polygon_angle_sum → n = 15 := by
  sorry

end original_polygon_sides_l1203_120362


namespace average_height_of_trees_l1203_120329

theorem average_height_of_trees (elm_height oak_height pine_height : ℚ) : 
  elm_height = 35 / 3 →
  oak_height = 107 / 6 →
  pine_height = 31 / 2 →
  (elm_height + oak_height + pine_height) / 3 = 15 := by
  sorry

end average_height_of_trees_l1203_120329


namespace reciprocal_sum_l1203_120391

theorem reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 = b^2 + b*c) (h2 : b^2 = c^2 + a*c) : 
  1/c = 1/a + 1/b := by
sorry

end reciprocal_sum_l1203_120391


namespace age_difference_z_younger_than_x_l1203_120311

-- Define variables for ages
variable (X Y Z : ℕ)

-- Define the condition from the problem
def age_condition (X Y Z : ℕ) : Prop := X + Y = Y + Z + 19

-- Theorem to prove
theorem age_difference (h : age_condition X Y Z) : X - Z = 19 :=
by sorry

-- Convert years to decades
def years_to_decades (years : ℕ) : ℚ := (years : ℚ) / 10

-- Theorem to prove the final result
theorem z_younger_than_x (h : age_condition X Y Z) : 
  years_to_decades (X - Z) = 1.9 :=
by sorry

end age_difference_z_younger_than_x_l1203_120311


namespace integer_solution_less_than_one_l1203_120361

theorem integer_solution_less_than_one :
  ∃ (x : ℤ), x - 1 < 0 :=
by
  use 0
  sorry

end integer_solution_less_than_one_l1203_120361


namespace quadratic_inequality_l1203_120382

theorem quadratic_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ ≠ x₂ →
  y₁ = -x₁^2 →
  y₂ = -x₂^2 →
  x₁ * x₂ > x₂^2 →
  y₁ < y₂ := by sorry

end quadratic_inequality_l1203_120382


namespace print_shop_copies_l1203_120356

theorem print_shop_copies (x_price y_price difference : ℚ) (h1 : x_price = 1.25)
  (h2 : y_price = 2.75) (h3 : difference = 90) :
  ∃ n : ℚ, n * y_price = n * x_price + difference ∧ n = 60 := by
  sorry

end print_shop_copies_l1203_120356


namespace bridge_length_l1203_120381

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  ∃ (bridge_length : ℝ),
    bridge_length = 265 ∧
    bridge_length + train_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by
  sorry

end bridge_length_l1203_120381


namespace condition_neither_sufficient_nor_necessary_l1203_120383

theorem condition_neither_sufficient_nor_necessary
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ)
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) ↔
    (∀ x, a₁ * x^2 + b₁ * x + c₁ > 0 ↔ a₂ * x^2 + b₂ * x + c₂ > 0)) :=
by sorry

end condition_neither_sufficient_nor_necessary_l1203_120383


namespace cricket_average_increase_l1203_120365

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  newInningsRuns : ℕ

/-- Calculates the increase in average runs per innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let oldAverage : ℚ := player.totalRuns / player.innings
  let newTotal : ℕ := player.totalRuns + player.newInningsRuns
  let newAverage : ℚ := newTotal / (player.innings + 1)
  newAverage - oldAverage

/-- Theorem stating the increase in average for the given scenario -/
theorem cricket_average_increase :
  ∀ (player : CricketPlayer),
  player.innings = 10 →
  player.totalRuns = 370 →
  player.newInningsRuns = 81 →
  averageIncrease player = 4 := by
  sorry


end cricket_average_increase_l1203_120365


namespace probability_through_D_l1203_120320

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Calculates the number of paths between two points --/
def numPaths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of choosing a specific path --/
def pathProbability (start finish : Point) : ℚ :=
  (1 / 2) ^ (finish.x - start.x + finish.y - start.y)

theorem probability_through_D (A D B : Point)
  (hA : A = ⟨0, 0⟩)
  (hD : D = ⟨3, 1⟩)
  (hB : B = ⟨6, 3⟩) :
  (numPaths A D * numPaths D B : ℚ) * pathProbability A B / numPaths A B = 20 / 63 :=
sorry

end probability_through_D_l1203_120320


namespace triangle_construction_theorem_l1203_120324

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle2D where
  v1 : Point2D
  v2 : Point2D
  v3 : Point2D

/-- Check if three lines are parallel -/
def are_parallel (l1 l2 l3 : Line2D) : Prop :=
  ∃ (k1 k2 : ℝ), l1.a = k1 * l2.a ∧ l1.b = k1 * l2.b ∧
                 l1.a = k2 * l3.a ∧ l1.b = k2 * l3.b

/-- Check if a point lies on a line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line passes through a point -/
def line_through_point (l : Line2D) (p : Point2D) : Prop :=
  point_on_line p l

/-- Check if a triangle's vertices lie on given lines -/
def triangle_vertices_on_lines (t : Triangle2D) (l1 l2 l3 : Line2D) : Prop :=
  (point_on_line t.v1 l1 ∨ point_on_line t.v1 l2 ∨ point_on_line t.v1 l3) ∧
  (point_on_line t.v2 l1 ∨ point_on_line t.v2 l2 ∨ point_on_line t.v2 l3) ∧
  (point_on_line t.v3 l1 ∨ point_on_line t.v3 l2 ∨ point_on_line t.v3 l3)

/-- Check if a triangle's sides (or extensions) pass through given points -/
def triangle_sides_through_points (t : Triangle2D) (p1 p2 p3 : Point2D) : Prop :=
  ∃ (l1 l2 l3 : Line2D),
    (point_on_line t.v1 l1 ∧ point_on_line t.v2 l1 ∧ line_through_point l1 p1) ∧
    (point_on_line t.v2 l2 ∧ point_on_line t.v3 l2 ∧ line_through_point l2 p2) ∧
    (point_on_line t.v3 l3 ∧ point_on_line t.v1 l3 ∧ line_through_point l3 p3)

theorem triangle_construction_theorem 
  (l1 l2 l3 : Line2D) 
  (p1 p2 p3 : Point2D) 
  (h_parallel : are_parallel l1 l2 l3) :
  ∃ (t : Triangle2D), 
    triangle_vertices_on_lines t l1 l2 l3 ∧ 
    triangle_sides_through_points t p1 p2 p3 := by
  sorry

end triangle_construction_theorem_l1203_120324


namespace equation_solution_l1203_120355

theorem equation_solution : ∃ x : ℝ, 9 - x - 2 * (31 - x) = 27 ∧ x = 80 := by
  sorry

end equation_solution_l1203_120355


namespace waiter_earnings_problem_l1203_120360

/-- Calculates the total earnings of a waiter during a shift --/
def waiter_earnings (customers : ℕ) (no_tip : ℕ) (tip_3 : ℕ) (tip_4 : ℕ) (tip_5 : ℕ) 
  (couple_groups : ℕ) (pool_contribution_rate : ℚ) (meal_cost : ℚ) : ℚ :=
  let total_tips := 3 * tip_3 + 4 * tip_4 + 5 * tip_5
  let net_tips := total_tips * (1 - pool_contribution_rate)
  net_tips - meal_cost

/-- Theorem stating that the waiter's earnings are $64.20 given the problem conditions --/
theorem waiter_earnings_problem : 
  waiter_earnings 25 5 8 6 6 2 (1/10) 6 = 321/5 := by
  sorry

end waiter_earnings_problem_l1203_120360


namespace lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l1203_120390

-- Define the lemniscate curve
def Lemniscate (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; ((x + a)^2 + y^2) * ((x - a)^2 + y^2) = a^4}

-- Statement 1: Symmetry with respect to the origin
theorem lemniscate_symmetric_origin (a : ℝ) (h : a > 0) :
  ∀ (p : ℝ × ℝ), p ∈ Lemniscate a ↔ (-p.1, -p.2) ∈ Lemniscate a :=
sorry

-- Statement 2: Maximum value of |PO|^2 - a^2
theorem lemniscate_max_distance_squared (a : ℝ) (h : a > 0) :
  ∃ (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    ∀ (q : ℝ × ℝ), q ∈ Lemniscate a → (p.1^2 + p.2^2) - a^2 ≥ (q.1^2 + q.2^2) - a^2 ∧
    (p.1^2 + p.2^2) - a^2 = a^2 :=
sorry

-- Statement 3: Unique point equidistant from focal points
theorem lemniscate_unique_equidistant_point (a : ℝ) (h : a > 0) :
  ∃! (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    (p.1 + a)^2 + p.2^2 = (p.1 - a)^2 + p.2^2 :=
sorry

end lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l1203_120390


namespace taxi_service_comparison_l1203_120307

-- Define the taxi services
structure TaxiService where
  initialFee : ℚ
  chargePerUnit : ℚ
  unitDistance : ℚ

def jimTaxi : TaxiService := { initialFee := 2.25, chargePerUnit := 0.35, unitDistance := 2/5 }
def susanTaxi : TaxiService := { initialFee := 3.00, chargePerUnit := 0.40, unitDistance := 1/3 }
def johnTaxi : TaxiService := { initialFee := 1.75, chargePerUnit := 0.30, unitDistance := 1/4 }

-- Function to calculate total charge
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + (distance / service.unitDistance).ceil * service.chargePerUnit

-- Theorem to prove
theorem taxi_service_comparison :
  let tripDistance : ℚ := 3.6
  let jimCharge := totalCharge jimTaxi tripDistance
  let susanCharge := totalCharge susanTaxi tripDistance
  let johnCharge := totalCharge johnTaxi tripDistance
  (jimCharge < johnCharge) ∧ (johnCharge < susanCharge) := by sorry

end taxi_service_comparison_l1203_120307


namespace interval_equivalence_l1203_120335

theorem interval_equivalence (x : ℝ) : 
  (1/4 < x ∧ x < 1/2) ↔ (1 < 5*x ∧ 5*x < 3) ∧ (2 < 8*x ∧ 8*x < 4) := by
  sorry

end interval_equivalence_l1203_120335


namespace room_width_calculation_l1203_120392

/-- Given a room with length 5.5 m and a floor paving cost of 400 Rs per sq metre
    resulting in a total cost of 8250 Rs, prove that the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) :
  length = 5.5 →
  cost_per_sqm = 400 →
  total_cost = 8250 →
  width = total_cost / cost_per_sqm / length →
  width = 3.75 := by
  sorry

#check room_width_calculation

end room_width_calculation_l1203_120392


namespace smallest_n_for_cube_sum_inequality_l1203_120396

theorem smallest_n_for_cube_sum_inequality : 
  ∃ n : ℕ, (∀ x y z : ℝ, (x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) ∧ 
  (∀ m : ℕ, m < n → ∃ x y z : ℝ, (x^3 + y^3 + z^3)^2 > m * (x^6 + y^6 + z^6)) ∧
  n = 3 := by
  sorry

end smallest_n_for_cube_sum_inequality_l1203_120396


namespace square_root_of_a_minus_b_l1203_120303

theorem square_root_of_a_minus_b (a b : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a + 3)^2 = x ∧ (2*a - 6)^2 = x) →
  (b = -8) →
  Real.sqrt (a - b) = 3 := by sorry

end square_root_of_a_minus_b_l1203_120303


namespace polynomial_factors_imply_absolute_value_l1203_120366

theorem polynomial_factors_imply_absolute_value (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (3 * x^3 - h * x + k)) →
  |3 * h - 2 * k| = 15 := by
  sorry

end polynomial_factors_imply_absolute_value_l1203_120366


namespace inequality_solution_l1203_120367

theorem inequality_solution (x : ℝ) : 
  3/20 + |x - 13/60| < 7/30 ↔ 2/15 < x ∧ x < 3/10 := by
sorry

end inequality_solution_l1203_120367


namespace alexanders_paintings_l1203_120387

/-- The number of paintings at each new gallery given Alexander's drawing conditions -/
theorem alexanders_paintings (first_gallery_paintings : ℕ) (new_galleries : ℕ) 
  (pencils_per_painting : ℕ) (signing_pencils_per_gallery : ℕ) (total_pencils : ℕ) :
  first_gallery_paintings = 9 →
  new_galleries = 5 →
  pencils_per_painting = 4 →
  signing_pencils_per_gallery = 2 →
  total_pencils = 88 →
  ∃ (paintings_per_new_gallery : ℕ),
    paintings_per_new_gallery = 2 ∧
    total_pencils = 
      first_gallery_paintings * pencils_per_painting + 
      new_galleries * paintings_per_new_gallery * pencils_per_painting +
      (new_galleries + 1) * signing_pencils_per_gallery :=
by sorry

end alexanders_paintings_l1203_120387


namespace sarah_tuesday_pencils_l1203_120331

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- Theorem: Sarah bought 18 pencils on Tuesday -/
theorem sarah_tuesday_pencils :
  monday_pencils + tuesday_pencils + 3 * tuesday_pencils = total_pencils :=
by sorry

end sarah_tuesday_pencils_l1203_120331


namespace contradiction_proof_l1203_120340

theorem contradiction_proof (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end contradiction_proof_l1203_120340


namespace parallel_lines_a_value_l1203_120378

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂ ∧ b₁ ≠ b₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = -(a/b)x - (c/b) -/
def slope_intercept_form (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -(a/b) * x - (c/b) :=
  sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x + 2 * y - 1 = 0 ↔ 8 * x + a * y + (2 - a) = 0) →
  a = -4 :=
sorry

end parallel_lines_a_value_l1203_120378


namespace base_10_to_base_5_l1203_120393

theorem base_10_to_base_5 : ∃ (a b c d : ℕ), 
  255 = a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0 ∧ 
  a = 2 ∧ b = 1 ∧ c = 0 ∧ d = 0 :=
by sorry

end base_10_to_base_5_l1203_120393


namespace f_property_implies_n_times_s_eq_14_l1203_120339

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the main property of f
axiom f_property (x y z : ℝ) : f (x^2 + y * f z) = x * f x + z * f y + y^2

-- Define n as the number of possible values of f(5)
def n : ℕ := sorry

-- Define s as the sum of all possible values of f(5)
def s : ℝ := sorry

-- State the theorem to be proved
theorem f_property_implies_n_times_s_eq_14 : n * s = 14 := by sorry

end f_property_implies_n_times_s_eq_14_l1203_120339


namespace f_satisfies_all_points_l1203_120310

/-- Function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := 200 - 40*x - 10*x^2

/-- The set of points given in the table -/
def points : List (ℝ × ℝ) := [(0, 200), (1, 160), (2, 80), (3, 0), (4, -120)]

/-- Theorem stating that the function f satisfies all points in the given table -/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end f_satisfies_all_points_l1203_120310


namespace line_plane_perpendicularity_l1203_120376

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β)
  (h3 : perpendicular m α) (h4 : parallel m β) :
  plane_perpendicular α β :=
sorry

end line_plane_perpendicularity_l1203_120376


namespace quadratic_inequality_equivalence_l1203_120370

theorem quadratic_inequality_equivalence : 
  ∀ x : ℝ, x * (2 * x + 3) < -2 ↔ x ∈ Set.Ioo (-2) 1 := by
  sorry

end quadratic_inequality_equivalence_l1203_120370


namespace three_cuts_make_5x5_mat_l1203_120305

/-- Represents a rectangular piece of cloth -/
structure Cloth where
  rows : ℕ
  cols : ℕ
  checkered : Bool

/-- Represents a cut on the cloth -/
inductive Cut
  | Vertical (col : ℕ)
  | Horizontal (row : ℕ)

/-- Represents the result of cutting a cloth -/
def cut_result (c : Cloth) (cut : Cut) : Cloth × Cloth :=
  match cut with
  | Cut.Vertical col => ⟨⟨c.rows, col, c.checkered⟩, ⟨c.rows, c.cols - col, c.checkered⟩⟩
  | Cut.Horizontal row => ⟨⟨row, c.cols, c.checkered⟩, ⟨c.rows - row, c.cols, c.checkered⟩⟩

/-- Checks if a cloth can form a 5x5 mat -/
def is_5x5_mat (c : Cloth) : Bool :=
  c.rows = 5 && c.cols = 5 && c.checkered

/-- The main theorem -/
theorem three_cuts_make_5x5_mat :
  ∃ (cut1 cut2 cut3 : Cut),
    let initial_cloth := Cloth.mk 6 7 true
    let (c1, c2) := cut_result initial_cloth cut1
    let (c3, c4) := cut_result c1 cut2
    let (c5, c6) := cut_result c2 cut3
    ∃ (final_cloth : Cloth),
      is_5x5_mat final_cloth ∧
      (final_cloth.rows * final_cloth.cols =
       c3.rows * c3.cols + c4.rows * c4.cols + c5.rows * c5.cols + c6.rows * c6.cols) :=
by
  sorry


end three_cuts_make_5x5_mat_l1203_120305


namespace silver_beads_count_l1203_120346

/-- Represents the number of beads in a necklace. -/
structure BeadCount where
  total : Nat
  blue : Nat
  red : Nat
  white : Nat
  silver : Nat

/-- Conditions for Michelle's necklace. -/
def michellesNecklace : BeadCount where
  total := 40
  blue := 5
  red := 2 * 5
  white := 5 + (2 * 5)
  silver := 40 - (5 + (2 * 5) + (5 + (2 * 5)))

/-- Theorem stating that the number of silver beads in Michelle's necklace is 10. -/
theorem silver_beads_count : michellesNecklace.silver = 10 := by
  sorry

#eval michellesNecklace.silver

end silver_beads_count_l1203_120346


namespace closest_integer_to_ten_minus_sqrt_thirteen_l1203_120323

theorem closest_integer_to_ten_minus_sqrt_thirteen :
  let sqrt_13 : ℝ := Real.sqrt 13
  ∀ n : ℤ, n ∈ ({4, 5, 7} : Set ℤ) →
    3 < sqrt_13 ∧ sqrt_13 < 4 →
    |10 - sqrt_13 - 6| < |10 - sqrt_13 - ↑n| :=
by sorry

end closest_integer_to_ten_minus_sqrt_thirteen_l1203_120323


namespace collinear_implies_coplanar_not_coplanar_implies_not_collinear_l1203_120372

-- Define a point in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem 1: If three points are collinear, then four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s := by sorry

-- Theorem 2: If four points are not coplanar, then no three points are collinear
theorem not_coplanar_implies_not_collinear (p q r s : Point3D) :
  ¬(coplanar p q r s) →
  ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s) := by sorry

end collinear_implies_coplanar_not_coplanar_implies_not_collinear_l1203_120372


namespace cubic_equation_roots_l1203_120343

theorem cubic_equation_roots (a : ℝ) (h : a > 3) :
  ∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ x^3 - a*x^2 + 1 = 0 :=
by sorry

end cubic_equation_roots_l1203_120343


namespace inverse_inequality_l1203_120302

theorem inverse_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end inverse_inequality_l1203_120302


namespace two_digit_repeating_decimal_l1203_120399

theorem two_digit_repeating_decimal (ab : ℕ) (h1 : ab ≥ 10 ∧ ab < 100) :
  66 * (1 + ab / 100 : ℚ) + 1/2 = 66 * (1 + ab / 99 : ℚ) → ab = 75 := by
  sorry

end two_digit_repeating_decimal_l1203_120399


namespace jake_has_eight_peaches_l1203_120379

-- Define the number of peaches each person has
def steven_peaches : ℕ := 15
def jill_peaches : ℕ := steven_peaches - 14
def jake_peaches : ℕ := steven_peaches - 7

-- Theorem statement
theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end jake_has_eight_peaches_l1203_120379


namespace sum_of_coefficients_l1203_120349

-- Define the polynomial
def p (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + 4*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)

-- Theorem: The sum of coefficients of p is 45
theorem sum_of_coefficients : p 1 = 45 := by
  sorry

end sum_of_coefficients_l1203_120349


namespace coefficient_x3y3_in_x_plus_y_to_6_l1203_120336

theorem coefficient_x3y3_in_x_plus_y_to_6 :
  (Finset.range 7).sum (fun k => (Nat.choose 6 k : ℕ) * 
    (if k = 3 then 1 else 0)) = 20 := by
  sorry

end coefficient_x3y3_in_x_plus_y_to_6_l1203_120336


namespace function_composition_ratio_l1203_120334

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end function_composition_ratio_l1203_120334


namespace square_area_proof_l1203_120358

theorem square_area_proof (x : ℝ) : 
  (5 * x - 18 = 27 - 4 * x) → 
  ((5 * x - 18)^2 : ℝ) = 49 := by
  sorry

end square_area_proof_l1203_120358


namespace water_percentage_in_fresh_grapes_water_percentage_is_90_l1203_120318

/-- The percentage of water in fresh grapes, given the conditions of the drying process -/
theorem water_percentage_in_fresh_grapes : ℝ → Prop :=
  fun p =>
    let fresh_weight : ℝ := 25
    let dried_weight : ℝ := 3.125
    let dried_water_percentage : ℝ := 20
    let fresh_solid_content : ℝ := fresh_weight * (100 - p) / 100
    let dried_solid_content : ℝ := dried_weight * (100 - dried_water_percentage) / 100
    fresh_solid_content = dried_solid_content →
    p = 90

/-- The theorem stating that the water percentage in fresh grapes is 90% -/
theorem water_percentage_is_90 : water_percentage_in_fresh_grapes 90 := by
  sorry

end water_percentage_in_fresh_grapes_water_percentage_is_90_l1203_120318


namespace negation_equivalence_l1203_120333

-- Define the universe of discourse
variable (Teacher : Type)

-- Define the predicates
variable (loves_math : Teacher → Prop)
variable (dislikes_math : Teacher → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ t : Teacher, dislikes_math t) ↔ ¬(∀ t : Teacher, loves_math t) :=
by sorry

end negation_equivalence_l1203_120333


namespace equation_solution_l1203_120398

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = -3 * (x₁ + 2)) ∧ 
  (x₂ * (x₂ + 2) = -3 * (x₂ + 2)) ∧ 
  x₁ = -2 ∧ x₂ = -3 := by
  sorry

end equation_solution_l1203_120398


namespace ceiling_fraction_evaluation_l1203_120357

theorem ceiling_fraction_evaluation : 
  (⌈(23 : ℝ) / 9 - ⌈(35 : ℝ) / 23⌉⌉) / (⌈(35 : ℝ) / 9 + ⌈(9 : ℝ) * 23 / 35⌉⌉) = (1 : ℝ) / 10 := by
  sorry

end ceiling_fraction_evaluation_l1203_120357


namespace cubic_greater_than_quadratic_l1203_120377

theorem cubic_greater_than_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end cubic_greater_than_quadratic_l1203_120377


namespace new_yellow_tint_percentage_l1203_120306

/-- Calculates the new percentage of yellow tint after adding more yellow tint to a mixture -/
theorem new_yellow_tint_percentage
  (initial_volume : ℝ)
  (initial_yellow_percentage : ℝ)
  (added_yellow_volume : ℝ)
  (h1 : initial_volume = 40)
  (h2 : initial_yellow_percentage = 0.25)
  (h3 : added_yellow_volume = 10) :
  let initial_yellow_volume := initial_volume * initial_yellow_percentage
  let new_yellow_volume := initial_yellow_volume + added_yellow_volume
  let new_total_volume := initial_volume + added_yellow_volume
  new_yellow_volume / new_total_volume = 0.4 := by
sorry


end new_yellow_tint_percentage_l1203_120306


namespace sculpture_surface_area_l1203_120385

/-- Represents a step in the staircase sculpture -/
structure Step where
  cubes : ℕ
  exposed_front : ℕ

/-- Represents the staircase sculpture -/
def Sculpture : List Step := [
  { cubes := 6, exposed_front := 6 },
  { cubes := 5, exposed_front := 5 },
  { cubes := 4, exposed_front := 4 },
  { cubes := 2, exposed_front := 2 },
  { cubes := 1, exposed_front := 5 }
]

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area (sculpture : List Step) : ℕ :=
  let top_area := sculpture.map (·.cubes) |>.sum
  let side_area := sculpture.map (·.exposed_front) |>.sum
  top_area + side_area

/-- Theorem: The total exposed surface area of the sculpture is 40 square meters -/
theorem sculpture_surface_area :
  total_exposed_area Sculpture = 40 := by
  sorry

end sculpture_surface_area_l1203_120385


namespace type_T_machine_time_l1203_120326

-- Define the time for a type B machine to complete the job
def time_B : ℝ := 7

-- Define the time for 2 type T machines and 3 type B machines to complete the job together
def time_combined : ℝ := 1.2068965517241381

-- Define the time for a type T machine to complete the job
def time_T : ℝ := 5

-- Theorem statement
theorem type_T_machine_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_T - (1 / ((1 / time_combined) - (3 / (2 * time_B))))| < ε :=
sorry

end type_T_machine_time_l1203_120326


namespace largest_n_for_product_1764_l1203_120300

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem largest_n_for_product_1764 
  (a b : ℕ → ℤ) 
  (ha : is_arithmetic_sequence a) 
  (hb : is_arithmetic_sequence b)
  (h1 : a 1 = 1 ∧ b 1 = 1)
  (h2 : a 2 ≤ b 2)
  (h3 : ∃ n : ℕ, a n * b n = 1764)
  : (∀ m : ℕ, (∃ n : ℕ, a n * b n = 1764) → m ≤ 44) :=
sorry

end largest_n_for_product_1764_l1203_120300


namespace parallel_planes_line_sufficiency_l1203_120395

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relations
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_line_sufficiency 
  (α β : Plane) (m : Line) 
  (h_subset : line_subset_plane m α) 
  (h_distinct : α ≠ β) :
  (∀ α β m, plane_parallel α β → line_parallel_plane m β) ∧ 
  (∃ α β m, line_parallel_plane m β ∧ ¬plane_parallel α β) :=
by sorry

end parallel_planes_line_sufficiency_l1203_120395


namespace pizza_slices_l1203_120309

theorem pizza_slices (buzz_ratio waiter_ratio : ℕ) 
  (h1 : buzz_ratio = 5)
  (h2 : waiter_ratio = 8)
  (h3 : waiter_ratio * x - 20 = 28)
  (x : ℕ) : 
  buzz_ratio * x + waiter_ratio * x = 78 := by
  sorry

end pizza_slices_l1203_120309


namespace triangle_diagram_solutions_l1203_120359

theorem triangle_diagram_solutions : 
  ∃! (solutions : List (ℕ × ℕ × ℕ)), 
    solutions.length = 6 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ solutions ↔ 
      (14 * 4 * a = 14 * 6 * c ∧ 
       14 * 4 * a = a * b * c ∧ 
       a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) := by
  sorry

end triangle_diagram_solutions_l1203_120359


namespace excess_amount_correct_l1203_120342

/-- The amount in excess of which the import tax is applied -/
def excess_amount : ℝ := 1000

/-- The total value of the item -/
def total_value : ℝ := 2560

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The amount of import tax paid -/
def tax_paid : ℝ := 109.20

/-- Theorem stating that the excess amount is correct given the conditions -/
theorem excess_amount_correct : 
  tax_rate * (total_value - excess_amount) = tax_paid :=
by sorry

end excess_amount_correct_l1203_120342


namespace polynomial_divisibility_l1203_120317

/-- A primitive third root of unity -/
noncomputable def α : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

/-- The polynomial x^103 + Cx^2 + Dx + E -/
def P (C D E : ℂ) (x : ℂ) : ℂ := x^103 + C*x^2 + D*x + E

theorem polynomial_divisibility (C D E : ℂ) :
  (∀ x, x^2 - x + 1 = 0 → P C D E x = 0) →
  C + D + E = 0 := by
  sorry

end polynomial_divisibility_l1203_120317


namespace sum_three_not_all_less_than_one_l1203_120368

theorem sum_three_not_all_less_than_one (a b c : ℝ) (h : a + b + c = 3) :
  ¬(a < 1 ∧ b < 1 ∧ c < 1) := by sorry

end sum_three_not_all_less_than_one_l1203_120368


namespace circle_ranges_l1203_120384

/-- The equation of a circle with parameter m -/
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2)*y + 16*m^4 + 9 = 0

/-- The range of m for which the equation represents a circle -/
def m_range (m : ℝ) : Prop :=
  -1/7 < m ∧ m < 1

/-- The range of the radius r of the circle -/
def r_range (r : ℝ) : Prop :=
  0 < r ∧ r ≤ 4/Real.sqrt 7

/-- Theorem stating the ranges of m and r for the given circle equation -/
theorem circle_ranges :
  (∃ x y : ℝ, circle_equation x y m) → m_range m ∧ (∃ r : ℝ, r_range r) :=
by sorry

end circle_ranges_l1203_120384


namespace base_seven_1732_equals_709_l1203_120380

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_1732_equals_709 :
  base_seven_to_ten [2, 3, 7, 1] = 709 := by
  sorry

end base_seven_1732_equals_709_l1203_120380


namespace gasoline_consumption_rate_l1203_120345

/-- Represents the gasoline consumption problem --/
structure GasolineProblem where
  initial_gasoline : ℝ
  supermarket_distance : ℝ
  farm_distance : ℝ
  partial_farm_trip : ℝ
  final_gasoline : ℝ

/-- Calculates the total distance traveled --/
def total_distance (p : GasolineProblem) : ℝ :=
  2 * p.supermarket_distance + 2 * p.partial_farm_trip + p.farm_distance

/-- Calculates the total gasoline consumed --/
def gasoline_consumed (p : GasolineProblem) : ℝ :=
  p.initial_gasoline - p.final_gasoline

/-- Theorem stating the gasoline consumption rate --/
theorem gasoline_consumption_rate (p : GasolineProblem) 
  (h1 : p.initial_gasoline = 12)
  (h2 : p.supermarket_distance = 5)
  (h3 : p.farm_distance = 6)
  (h4 : p.partial_farm_trip = 2)
  (h5 : p.final_gasoline = 2) :
  total_distance p / gasoline_consumed p = 2 := by sorry

end gasoline_consumption_rate_l1203_120345


namespace max_angle_point_is_tangency_point_l1203_120344

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A structure representing a circle in a plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Function to calculate the angle between three points -/
def angle (A B M : Point) : ℝ := sorry

/-- Function to check if a point is on a line -/
def pointOnLine (P : Point) (l : Line) : Prop := sorry

/-- Function to check if a line intersects a segment -/
def lineIntersectsSegment (l : Line) (A B : Point) : Prop := sorry

/-- Function to check if a circle passes through two points -/
def circlePassesThroughPoints (C : Circle) (A B : Point) : Prop := sorry

/-- Function to check if a circle is tangent to a line -/
def circleTangentToLine (C : Circle) (l : Line) : Prop := sorry

/-- Theorem stating that the point M on line (d) that maximizes the angle ∠AMB
    is the point of tangency of the smallest circumcircle passing through A and B
    with the line (d) -/
theorem max_angle_point_is_tangency_point
  (A B : Point) (d : Line) 
  (h : ¬ lineIntersectsSegment d A B) :
  ∃ (M : Point) (C : Circle),
    pointOnLine M d ∧
    circlePassesThroughPoints C A B ∧
    circleTangentToLine C d ∧
    (∀ (M' : Point), pointOnLine M' d → angle A M' B ≤ angle A M B) :=
sorry

end max_angle_point_is_tangency_point_l1203_120344


namespace number_calculation_l1203_120321

theorem number_calculation (n : ℝ) : 
  0.1 * 0.3 * ((Real.sqrt (0.5 * n))^2) = 90 → n = 6000 := by
sorry

end number_calculation_l1203_120321


namespace counterexample_exists_l1203_120363

-- Define the set of numbers to check
def numbers : List Nat := [25, 35, 39, 49, 51]

-- Define what it means for a number to be composite
def isComposite (n : Nat) : Prop := ¬ Nat.Prime n

-- Define the counterexample property
def isCounterexample (n : Nat) : Prop := isComposite n ∧ Nat.Prime (n - 2)

-- Theorem to prove
theorem counterexample_exists : ∃ n ∈ numbers, isCounterexample n := by
  sorry

end counterexample_exists_l1203_120363


namespace arthur_walk_distance_l1203_120371

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance_miles (blocks_east : ℕ) (blocks_north : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * miles_per_block

/-- Theorem stating that walking 6 blocks east and 12 blocks north, with each block being one-third of a mile, results in a total distance of 6 miles. -/
theorem arthur_walk_distance :
  total_distance_miles 6 12 (1/3) = 6 := by sorry

end arthur_walk_distance_l1203_120371


namespace triangle_theorem_l1203_120301

theorem triangle_theorem (a b c A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = π) (h5 : 2 * c * Real.cos C + b * Real.cos A + a * Real.cos B = 0) :
  C = 2 * π / 3 ∧ 
  (c = 3 → A = π / 6 → (1 / 2) * a * b * Real.sin C = 3 * Real.sqrt 3 / 4) :=
by sorry

end triangle_theorem_l1203_120301


namespace line_parameterization_l1203_120332

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y, y = 2*x - 40 ↔ ∃ t, x = g t ∧ y = 20*t - 14) →
  ∀ t, g t = 10*t + 13 := by
sorry

end line_parameterization_l1203_120332
