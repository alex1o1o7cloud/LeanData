import Mathlib

namespace NUMINAMATH_CALUDE_triangle_proof_l605_60563

theorem triangle_proof (A B C : ℝ) (a b c : ℝ) (P : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a = Real.sqrt 5 ∧
  c * Real.sin A = Real.sqrt 2 * Real.sin ((A + B) / 2) ∧
  Real.sqrt 5 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  Real.sqrt 5 = Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) ∧
  1 = Real.sqrt ((P.1 - 0)^2 + (P.2 - 0)^2) ∧
  3 * π / 4 = Real.arccos ((P.1 * 1 + P.2 * 0) / (Real.sqrt (P.1^2 + P.2^2) * Real.sqrt 5)) →
  C = π / 2 ∧
  Real.sqrt ((1 - P.1)^2 + (0 - P.2)^2) = Real.sqrt ((P.1 - 1)^2 + P.2^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l605_60563


namespace NUMINAMATH_CALUDE_no_perfect_square_212_b_l605_60531

theorem no_perfect_square_212_b : ¬ ∃ (b : ℕ), b > 2 ∧ ∃ (n : ℕ), 2 * b^2 + b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_212_b_l605_60531


namespace NUMINAMATH_CALUDE_line_y_coordinate_l605_60529

/-- Given a line passing through points (10, y₁) and (x₂, -8), with an x-intercept at (4, 0), prove that y₁ = -8 -/
theorem line_y_coordinate (y₁ x₂ : ℝ) : 
  (∃ m b : ℝ, 
    (y₁ = m * 10 + b) ∧ 
    (-8 = m * x₂ + b) ∧ 
    (0 = m * 4 + b)) → 
  y₁ = -8 :=
by sorry

end NUMINAMATH_CALUDE_line_y_coordinate_l605_60529


namespace NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l605_60576

theorem simplest_fraction_of_decimal (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.428125 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.428125 → a * d ≤ b * c →
  a = 137 ∧ b = 320 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_of_decimal_l605_60576


namespace NUMINAMATH_CALUDE_descent_time_specific_garage_l605_60533

/-- Represents a parking garage with specified characteristics -/
structure ParkingGarage where
  floors : ℕ
  gateInterval : ℕ
  gateTime : ℕ
  floorDistance : ℕ
  drivingSpeed : ℕ

/-- Calculates the total time to descend the parking garage -/
def descentTime (garage : ParkingGarage) : ℕ :=
  let drivingTime := (garage.floors - 1) * (garage.floorDistance / garage.drivingSpeed)
  let gateCount := (garage.floors - 1) / garage.gateInterval
  let gateTime := gateCount * garage.gateTime
  drivingTime + gateTime

/-- The theorem stating the total descent time for the specific garage -/
theorem descent_time_specific_garage :
  let garage : ParkingGarage := {
    floors := 12,
    gateInterval := 3,
    gateTime := 120,
    floorDistance := 800,
    drivingSpeed := 10
  }
  descentTime garage = 1240 := by sorry

end NUMINAMATH_CALUDE_descent_time_specific_garage_l605_60533


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l605_60597

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 5}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(4, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l605_60597


namespace NUMINAMATH_CALUDE_odd_ceiling_factorial_fraction_l605_60512

theorem odd_ceiling_factorial_fraction (n : ℕ) (h1 : n > 6) (h2 : Nat.Prime (n + 1)) :
  Odd (⌈(Nat.factorial (n - 1) : ℚ) / (n * (n + 1))⌉) := by
  sorry

end NUMINAMATH_CALUDE_odd_ceiling_factorial_fraction_l605_60512


namespace NUMINAMATH_CALUDE_fraction_value_l605_60559

theorem fraction_value : (20 + 24) / (20 - 24) = -11 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l605_60559


namespace NUMINAMATH_CALUDE_goldfish_equality_month_l605_60590

theorem goldfish_equality_month : ∃ n : ℕ, n > 0 ∧ 3^(n+1) = 125 * 5^n ∧ ∀ m : ℕ, 0 < m ∧ m < n → 3^(m+1) ≠ 125 * 5^m :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_month_l605_60590


namespace NUMINAMATH_CALUDE_sector_area_l605_60583

theorem sector_area (r : ℝ) (θ : ℝ) (h : θ = 72 * π / 180) :
  let A := (θ / (2 * π)) * π * r^2
  r = 20 → A = 80 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l605_60583


namespace NUMINAMATH_CALUDE_contrapositive_not_always_false_l605_60522

theorem contrapositive_not_always_false :
  ∃ (p q : Prop), (p → q) ∧ ¬(¬q → ¬p) → False :=
sorry

end NUMINAMATH_CALUDE_contrapositive_not_always_false_l605_60522


namespace NUMINAMATH_CALUDE_percentile_rank_between_90_and_91_l605_60541

/-- Represents a student's rank in a class -/
structure StudentRank where
  total_students : ℕ
  rank : ℕ
  h_rank_valid : rank ≤ total_students

/-- Calculates the percentile rank of a student -/
def percentile_rank (sr : StudentRank) : ℚ :=
  (sr.total_students - sr.rank : ℚ) / sr.total_students * 100

/-- Theorem stating that a student ranking 5th in a class of 48 has a percentile rank between 90 and 91 -/
theorem percentile_rank_between_90_and_91 (sr : StudentRank) 
  (h_total : sr.total_students = 48) 
  (h_rank : sr.rank = 5) : 
  90 < percentile_rank sr ∧ percentile_rank sr < 91 := by
  sorry

#eval percentile_rank ⟨48, 5, by norm_num⟩

end NUMINAMATH_CALUDE_percentile_rank_between_90_and_91_l605_60541


namespace NUMINAMATH_CALUDE_sum_of_inverse_cubes_of_roots_l605_60568

theorem sum_of_inverse_cubes_of_roots (r s : ℝ) : 
  (3 * r^2 + 5 * r + 2 = 0) → 
  (3 * s^2 + 5 * s + 2 = 0) → 
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 25 / 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_inverse_cubes_of_roots_l605_60568


namespace NUMINAMATH_CALUDE_ed_hotel_stay_l605_60519

def hotel_problem (night_rate : ℚ) (morning_rate : ℚ) (initial_money : ℚ) (night_hours : ℚ) (money_left : ℚ) : ℚ :=
  let total_spent := initial_money - money_left
  let night_cost := night_rate * night_hours
  let morning_spent := total_spent - night_cost
  morning_spent / morning_rate

theorem ed_hotel_stay :
  hotel_problem 1.5 2 80 6 63 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ed_hotel_stay_l605_60519


namespace NUMINAMATH_CALUDE_fraction_problem_l605_60542

theorem fraction_problem : 
  let x : ℚ := 2/3
  (3/4 : ℚ) * (4/5 : ℚ) * x = (2/5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l605_60542


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l605_60540

theorem digit_sum_puzzle (a p v m t r : ℕ) 
  (h1 : a + p = v)
  (h2 : v + m = t)
  (h3 : t + a = r)
  (h4 : p + m + r = 18)
  (h5 : a ≠ 0 ∧ p ≠ 0 ∧ v ≠ 0 ∧ m ≠ 0 ∧ t ≠ 0 ∧ r ≠ 0)
  (h6 : a ≠ p ∧ a ≠ v ∧ a ≠ m ∧ a ≠ t ∧ a ≠ r ∧
        p ≠ v ∧ p ≠ m ∧ p ≠ t ∧ p ≠ r ∧
        v ≠ m ∧ v ≠ t ∧ v ≠ r ∧
        m ≠ t ∧ m ≠ r ∧
        t ≠ r) :
  t = 9 := by
  sorry


end NUMINAMATH_CALUDE_digit_sum_puzzle_l605_60540


namespace NUMINAMATH_CALUDE_soccer_most_popular_l605_60514

-- Define the list of sports
inductive Sport
  | Hockey
  | Basketball
  | Soccer
  | Volleyball
  | Badminton

-- Function to get the number of students for each sport
def students_playing (s : Sport) : ℕ :=
  match s with
  | Sport.Hockey => 30
  | Sport.Basketball => 40
  | Sport.Soccer => 50
  | Sport.Volleyball => 35
  | Sport.Badminton => 25

-- Theorem: Soccer has the highest number of students
theorem soccer_most_popular (s : Sport) : 
  students_playing Sport.Soccer ≥ students_playing s :=
sorry

end NUMINAMATH_CALUDE_soccer_most_popular_l605_60514


namespace NUMINAMATH_CALUDE_emilys_cards_l605_60552

theorem emilys_cards (initial_cards : ℕ) (cards_per_apple : ℕ) (bruce_apples : ℕ) :
  initial_cards + cards_per_apple * bruce_apples = 
  initial_cards + cards_per_apple * bruce_apples := by
  sorry

#check emilys_cards 63 7 13

end NUMINAMATH_CALUDE_emilys_cards_l605_60552


namespace NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_six_l605_60595

theorem m_squared_plus_inverse_squared_plus_six (m : ℝ) (h : m + 1/m = 10) : 
  m^2 + 1/m^2 + 6 = 104 := by
  sorry

end NUMINAMATH_CALUDE_m_squared_plus_inverse_squared_plus_six_l605_60595


namespace NUMINAMATH_CALUDE_scientific_notation_of_120000_l605_60545

theorem scientific_notation_of_120000 :
  (120000 : ℝ) = 1.2 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120000_l605_60545


namespace NUMINAMATH_CALUDE_reflect_center_of_circle_l605_60589

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflect_center_of_circle : reflect_about_y_eq_neg_x (3, -7) = (7, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_center_of_circle_l605_60589


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l605_60520

/-- Given a circle with equation x^2 + y^2 - 2x = 0, its center is (1,0) and its radius is 1 -/
theorem circle_center_and_radius :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, 0) ∧ radius = 1 ∧
    ∀ (x y : ℝ), circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l605_60520


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l605_60518

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ x) ↔ (∃ x : ℝ, x^2 + 1 < x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l605_60518


namespace NUMINAMATH_CALUDE_ones_divisibility_l605_60553

theorem ones_divisibility (d : ℕ) (h1 : d > 0) (h2 : ¬ 2 ∣ d) (h3 : ¬ 5 ∣ d) :
  ∃ n : ℕ, d ∣ ((10^n - 1) / 9) :=
sorry

end NUMINAMATH_CALUDE_ones_divisibility_l605_60553


namespace NUMINAMATH_CALUDE_second_largest_is_eleven_l605_60560

def numbers : Finset ℕ := {10, 11, 12}

theorem second_largest_is_eleven :
  ∃ (a b : ℕ), a ∈ numbers ∧ b ∈ numbers ∧ a ≠ b ∧
  (∀ x ∈ numbers, x ≤ a) ∧
  (∃ y ∈ numbers, y > 11) ∧
  (∀ z ∈ numbers, z > 11 → z ≥ a) :=
sorry

end NUMINAMATH_CALUDE_second_largest_is_eleven_l605_60560


namespace NUMINAMATH_CALUDE_square_perimeter_equals_circle_area_l605_60584

theorem square_perimeter_equals_circle_area (r : ℝ) : r = 8 / Real.pi :=
  -- Define the perimeter of the square
  let square_perimeter := 8 * r
  -- Define the area of the circle
  let circle_area := Real.pi * r^2
  -- State that the perimeter of the square equals the area of the circle
  have h : square_perimeter = circle_area := by sorry
  -- Prove that r = 8 / π
  sorry

end NUMINAMATH_CALUDE_square_perimeter_equals_circle_area_l605_60584


namespace NUMINAMATH_CALUDE_union_of_intervals_l605_60526

open Set

theorem union_of_intervals (A B : Set ℝ) : 
  A = {x : ℝ | 3 < x ∧ x ≤ 7} →
  B = {x : ℝ | 4 < x ∧ x ≤ 10} →
  A ∪ B = {x : ℝ | 3 < x ∧ x ≤ 10} :=
by sorry

end NUMINAMATH_CALUDE_union_of_intervals_l605_60526


namespace NUMINAMATH_CALUDE_biff_voting_percentage_l605_60549

theorem biff_voting_percentage (total_polled : ℕ) (marty_votes : ℕ) (undecided_percent : ℚ) :
  total_polled = 200 →
  marty_votes = 94 →
  undecided_percent = 8 / 100 →
  (↑(total_polled - marty_votes - (undecided_percent * ↑total_polled).num) / ↑total_polled : ℚ) = 45 / 100 := by
  sorry

end NUMINAMATH_CALUDE_biff_voting_percentage_l605_60549


namespace NUMINAMATH_CALUDE_inequality_proof_l605_60536

theorem inequality_proof (x : ℝ) : (3 * x - 5) / 2 > 2 * x → x < -5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l605_60536


namespace NUMINAMATH_CALUDE_N_divisible_by_2027_l605_60554

theorem N_divisible_by_2027 : ∃ k : ℤ, (7 * 9 * 13 + 2020 * 2018 * 2014) = 2027 * k := by
  sorry

end NUMINAMATH_CALUDE_N_divisible_by_2027_l605_60554


namespace NUMINAMATH_CALUDE_rational_sum_product_equality_l605_60534

theorem rational_sum_product_equality : ∃ (a b : ℚ), a ≠ b ∧ a + b = a * b ∧ a = 3/2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_product_equality_l605_60534


namespace NUMINAMATH_CALUDE_division_problem_l605_60547

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/3) : 
  c / a = 1/2 := by sorry

end NUMINAMATH_CALUDE_division_problem_l605_60547


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l605_60556

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 + Complex.I) * Complex.I^3 / (1 - Complex.I) = 1 - Complex.I →
  z.im = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l605_60556


namespace NUMINAMATH_CALUDE_teacher_age_l605_60516

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) : 
  num_students = 19 →
  student_avg_age = 20 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l605_60516


namespace NUMINAMATH_CALUDE_unique_solution_l605_60558

/-- A 3x3 matrix with special properties -/
structure SpecialMatrix where
  a : Matrix (Fin 3) (Fin 3) ℝ
  all_positive : ∀ i j, 0 < a i j
  row_sum_one : ∀ i, (Finset.univ.sum (λ j => a i j)) = 1
  col_sum_one : ∀ j, (Finset.univ.sum (λ i => a i j)) = 1
  diagonal_half : ∀ i, a i i = 1/2

/-- The system of equations -/
def system (m : SpecialMatrix) (x y z : ℝ) : Prop :=
  m.a 0 0 * x + m.a 0 1 * y + m.a 0 2 * z = 0 ∧
  m.a 1 0 * x + m.a 1 1 * y + m.a 1 2 * z = 0 ∧
  m.a 2 0 * x + m.a 2 1 * y + m.a 2 2 * z = 0

theorem unique_solution (m : SpecialMatrix) :
  ∀ x y z, system m x y z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l605_60558


namespace NUMINAMATH_CALUDE_russian_in_top_three_l605_60581

structure ChessTournament where
  total_players : Nat
  russian_players : Nat
  foreign_players : Nat
  games_per_pair : Nat
  total_points : Nat
  russian_points : Nat
  foreign_points : Nat

def valid_tournament (t : ChessTournament) : Prop :=
  t.total_players = 11 ∧
  t.russian_players = 4 ∧
  t.foreign_players = 7 ∧
  t.games_per_pair = 2 ∧
  t.total_points = t.total_players * (t.total_players - 1) ∧
  t.russian_points = t.foreign_points ∧
  t.russian_points + t.foreign_points = t.total_points

theorem russian_in_top_three (t : ChessTournament) (h : valid_tournament t) :
  ∃ (top_three : Finset Nat) (russian : Nat),
    top_three.card = 3 ∧
    russian ∈ top_three ∧
    russian ≤ t.russian_players :=
  sorry

end NUMINAMATH_CALUDE_russian_in_top_three_l605_60581


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l605_60598

theorem min_sum_cube_relation (m n : ℕ+) (h : 108 * m = n ^ 3) :
  ∃ (m₀ n₀ : ℕ+), 108 * m₀ = n₀ ^ 3 ∧ m₀ + n₀ = 8 ∧ ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m' + n' ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l605_60598


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l605_60572

theorem sibling_ages_sum (a b c : ℕ+) 
  (h_order : c < b ∧ b < a) 
  (h_product : a * b * c = 72) : 
  a + b + c = 13 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l605_60572


namespace NUMINAMATH_CALUDE_major_preference_stronger_than_gender_l605_60524

/-- Represents the observed K^2 value for gender preference --/
def k1 : ℝ := 1.010

/-- Represents the observed K^2 value for major preference --/
def k2 : ℝ := 9.090

/-- Theorem stating that the observed K^2 value for major preference is greater than the observed K^2 value for gender preference --/
theorem major_preference_stronger_than_gender : k2 > k1 := by sorry

end NUMINAMATH_CALUDE_major_preference_stronger_than_gender_l605_60524


namespace NUMINAMATH_CALUDE_function_is_zero_l605_60591

def is_logarithmic_property (f : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m + f n

def is_non_decreasing (f : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, f (n + 1) ≥ f n

theorem function_is_zero
  (f : ℕ+ → ℝ)
  (h1 : is_logarithmic_property f)
  (h2 : is_non_decreasing f) :
  ∀ n : ℕ+, f n = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_is_zero_l605_60591


namespace NUMINAMATH_CALUDE_line_sequence_stabilizes_l605_60505

/-- Represents a line of 2018 natural numbers -/
def Line := Fin 2018 → ℕ

/-- Creates the next line based on the current line -/
def nextLine (l : Line) : Line := sorry

/-- Checks if two lines are identical -/
def linesEqual (l1 l2 : Line) : Prop := ∀ i, l1 i = l2 i

/-- The sequence of lines generated by repeatedly applying nextLine -/
def lineSequence (initial : Line) : ℕ → Line
  | 0 => initial
  | n + 1 => nextLine (lineSequence initial n)

/-- The main theorem: the sequence of lines eventually stabilizes -/
theorem line_sequence_stabilizes (initial : Line) : 
  ∃ N : ℕ, ∀ n ≥ N, linesEqual (lineSequence initial n) (lineSequence initial (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_line_sequence_stabilizes_l605_60505


namespace NUMINAMATH_CALUDE_exists_dihedral_equal_edge_not_equal_exists_edge_equal_dihedral_not_equal_dihedral_angles_neither_necessary_nor_sufficient_l605_60539

/-- A quadrilateral pyramid with vertex V and base ABCD. -/
structure QuadrilateralPyramid where
  V : Point
  A : Point
  B : Point
  C : Point
  D : Point

/-- The property that all dihedral angles between adjacent faces are equal. -/
def all_dihedral_angles_equal (pyramid : QuadrilateralPyramid) : Prop :=
  sorry

/-- The property that all angles between adjacent edges are equal. -/
def all_edge_angles_equal (pyramid : QuadrilateralPyramid) : Prop :=
  sorry

/-- There exists a pyramid where all dihedral angles are equal but not all edge angles are equal. -/
theorem exists_dihedral_equal_edge_not_equal :
  ∃ (pyramid : QuadrilateralPyramid),
    all_dihedral_angles_equal pyramid ∧ ¬all_edge_angles_equal pyramid :=
  sorry

/-- There exists a pyramid where all edge angles are equal but not all dihedral angles are equal. -/
theorem exists_edge_equal_dihedral_not_equal :
  ∃ (pyramid : QuadrilateralPyramid),
    all_edge_angles_equal pyramid ∧ ¬all_dihedral_angles_equal pyramid :=
  sorry

/-- The main theorem stating that the equality of dihedral angles is neither necessary nor sufficient for the equality of edge angles. -/
theorem dihedral_angles_neither_necessary_nor_sufficient :
  (∃ (pyramid : QuadrilateralPyramid), all_dihedral_angles_equal pyramid ∧ ¬all_edge_angles_equal pyramid) ∧
  (∃ (pyramid : QuadrilateralPyramid), all_edge_angles_equal pyramid ∧ ¬all_dihedral_angles_equal pyramid) :=
  sorry

end NUMINAMATH_CALUDE_exists_dihedral_equal_edge_not_equal_exists_edge_equal_dihedral_not_equal_dihedral_angles_neither_necessary_nor_sufficient_l605_60539


namespace NUMINAMATH_CALUDE_nineteen_ninetyeight_impossible_l605_60575

/-- The type of operations that can be performed on a number -/
inductive Operation
| Square : Operation
| AddOne : Operation

/-- Apply an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Square => n * n
  | Operation.AddOne => n + 1

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Apply a sequence of operations to a number -/
def applySequence (n : ℕ) (seq : OperationSequence) : ℕ :=
  seq.foldl applyOperation n

/-- The theorem stating that 19 and 98 cannot be made equal with the same number of operations -/
theorem nineteen_ninetyeight_impossible :
  ∀ (seq : OperationSequence), applySequence 19 seq ≠ applySequence 98 seq :=
sorry

end NUMINAMATH_CALUDE_nineteen_ninetyeight_impossible_l605_60575


namespace NUMINAMATH_CALUDE_point_b_value_l605_60578

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moving right on a number line -/
def moveRight (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_b_value (a b : Point) (h1 : a.value = -3) (h2 : b = moveRight a 4) :
  b.value = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_b_value_l605_60578


namespace NUMINAMATH_CALUDE_min_value_on_interval_l605_60596

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ 
  (∀ y ∈ Set.Icc 0 3, f y ≥ f x) ∧
  f x = -15 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l605_60596


namespace NUMINAMATH_CALUDE_problem_solution_l605_60501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1 - x) / (a * x)

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∀ x ≥ 1, Monotone (f a) → a ≥ 1) ∧
  (∀ x ∈ Set.Icc 1 2,
    (0 < a ∧ a ≤ 1/2 → f a x ≥ Real.log 2 - 1/(2*a)) ∧
    (1/2 < a ∧ a < 1 → f a x ≥ Real.log (1/a) + 1 - 1/a) ∧
    (a ≥ 1 → f a x ≥ 0)) ∧
  (∀ n : ℕ, n > 1 → Real.log n > (Finset.range (n-1)).sum (λ i => 1 / (i + 2))) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l605_60501


namespace NUMINAMATH_CALUDE_physics_group_size_l605_60551

theorem physics_group_size (total : ℕ) (math_ratio physics_ratio chem_ratio : ℕ) : 
  total = 135 → 
  math_ratio = 6 →
  physics_ratio = 5 →
  chem_ratio = 4 →
  (physics_ratio : ℚ) / (math_ratio + physics_ratio + chem_ratio : ℚ) * total = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_physics_group_size_l605_60551


namespace NUMINAMATH_CALUDE_cos_sin_difference_equals_sqrt3_over_2_l605_60523

theorem cos_sin_difference_equals_sqrt3_over_2 :
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) -
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_difference_equals_sqrt3_over_2_l605_60523


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l605_60594

-- Define a triangle ABC
structure Triangle (α : Type) [Field α] where
  A : α
  B : α
  C : α
  a : α
  b : α
  c : α

-- State the theorem
theorem triangle_angle_proof {α : Type} [Field α] (ABC : Triangle α) :
  ABC.b = 2 * ABC.a →
  ABC.B = ABC.A + 60 →
  ABC.A = 30 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l605_60594


namespace NUMINAMATH_CALUDE_adam_laundry_theorem_l605_60567

/-- The number of loads Adam has already washed -/
def washed_loads : ℕ := 8

/-- The number of loads Adam still needs to wash -/
def remaining_loads : ℕ := 6

/-- The total number of loads Adam has to wash -/
def total_loads : ℕ := washed_loads + remaining_loads

theorem adam_laundry_theorem : total_loads = 14 := by sorry

end NUMINAMATH_CALUDE_adam_laundry_theorem_l605_60567


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l605_60538

theorem cos_alpha_minus_pi_sixth (α : ℝ) (h : Real.sin (α + π/3) = 4/5) : 
  Real.cos (α - π/6) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_l605_60538


namespace NUMINAMATH_CALUDE_max_non_managers_l605_60532

/-- Represents the number of managers in the department -/
def managers : ℕ := 8

/-- Represents the maximum total number of employees allowed in the department -/
def max_total : ℕ := 130

/-- Represents the company-wide ratio of managers to non-managers -/
def company_ratio : ℚ := 5 / 24

/-- Represents the department-specific ratio of managers to non-managers -/
def dept_ratio : ℚ := 3 / 5

/-- Theorem stating the maximum number of non-managers in the department -/
theorem max_non_managers :
  ∃ (n : ℕ), n = 13 ∧ 
  (managers : ℚ) / n > company_ratio ∧
  (managers : ℚ) / n ≤ dept_ratio ∧
  managers + n ≤ max_total ∧
  ∀ (m : ℕ), m > n → 
    ((managers : ℚ) / m ≤ company_ratio ∨
     (managers : ℚ) / m > dept_ratio ∨
     managers + m > max_total) :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l605_60532


namespace NUMINAMATH_CALUDE_f_extrema_l605_60555

def f (x : ℝ) := x^2 - 2*x + 2

def A₁ : Set ℝ := Set.Icc (-2) 0
def A₂ : Set ℝ := Set.Icc 2 3

theorem f_extrema :
  (∀ x ∈ A₁, f x ≤ 10 ∧ f x ≥ 2) ∧
  (∃ x₁ ∈ A₁, f x₁ = 10) ∧
  (∃ x₂ ∈ A₁, f x₂ = 2) ∧
  (∀ x ∈ A₂, f x ≤ 5 ∧ f x ≥ 2) ∧
  (∃ x₃ ∈ A₂, f x₃ = 5) ∧
  (∃ x₄ ∈ A₂, f x₄ = 2) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_l605_60555


namespace NUMINAMATH_CALUDE_fair_coin_probability_l605_60546

/-- A coin toss with two possible outcomes -/
inductive CoinOutcome
  | heads
  | tails

/-- The probability of a coin toss outcome -/
def probability (outcome : CoinOutcome) : Real :=
  0.5

/-- Theorem stating that the probability of getting heads or tails is 0.5 -/
theorem fair_coin_probability :
  ∀ (outcome : CoinOutcome), probability outcome = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_l605_60546


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l605_60502

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l605_60502


namespace NUMINAMATH_CALUDE_x_range_for_negative_f_l605_60570

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

-- Define the theorem
theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0) →
  {x : ℝ | 1 < x ∧ x < 2} = {x : ℝ | ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0} :=
by sorry


end NUMINAMATH_CALUDE_x_range_for_negative_f_l605_60570


namespace NUMINAMATH_CALUDE_simplify_fraction_l605_60548

theorem simplify_fraction (b : ℚ) (h : b = 4) : 18 * b^4 / (27 * b^3) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l605_60548


namespace NUMINAMATH_CALUDE_infinite_sum_equals_two_l605_60511

theorem infinite_sum_equals_two :
  (∑' n : ℕ, (4 * n - 2) / (3 : ℝ)^n) = 2 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_two_l605_60511


namespace NUMINAMATH_CALUDE_two_and_three_digit_number_product_l605_60573

theorem two_and_three_digit_number_product : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 
  100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 9 * x * y ∧
  x + y = 126 := by
sorry

end NUMINAMATH_CALUDE_two_and_three_digit_number_product_l605_60573


namespace NUMINAMATH_CALUDE_overlap_length_l605_60528

/-- Given red line segments with equal lengths and overlaps, prove the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (edge_to_edge : ℝ) (num_overlaps : ℕ) 
  (h1 : total_length = 98)
  (h2 : edge_to_edge = 83)
  (h3 : num_overlaps = 6)
  (h4 : total_length - edge_to_edge = num_overlaps * (total_length - edge_to_edge) / num_overlaps) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_l605_60528


namespace NUMINAMATH_CALUDE_robie_initial_cards_robie_initial_cards_proof_l605_60509

theorem robie_initial_cards (cards_per_box : ℕ) (loose_cards : ℕ) (boxes_given : ℕ) 
  (boxes_returned : ℕ) (current_boxes : ℕ) (cards_bought : ℕ) (cards_traded : ℕ) : ℕ :=
  let initial_boxes := current_boxes - boxes_returned + boxes_given
  let boxed_cards := initial_boxes * cards_per_box
  let total_cards := boxed_cards + loose_cards
  let initial_cards := total_cards - cards_bought
  initial_cards

theorem robie_initial_cards_proof :
  robie_initial_cards 30 18 8 2 15 21 12 = 627 := by
  sorry

end NUMINAMATH_CALUDE_robie_initial_cards_robie_initial_cards_proof_l605_60509


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l605_60580

/-- In a zigzag inside a rectangle, if certain angles are given, prove that angle CDE (θ) equals 11 degrees. -/
theorem zigzag_angle_theorem (ACB FEG DCE DEC : ℝ) (h1 : ACB = 80) (h2 : FEG = 64) (h3 : DCE = 86) (h4 : DEC = 83) :
  180 - DCE - DEC = 11 :=
sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l605_60580


namespace NUMINAMATH_CALUDE_election_majority_l605_60564

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 600 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = 240 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_l605_60564


namespace NUMINAMATH_CALUDE_number_of_children_l605_60504

theorem number_of_children (total : ℕ) (adults : ℕ) (children : ℕ) : 
  total = 42 → 
  children = 2 * adults → 
  total = adults + children →
  children = 28 := by
sorry

end NUMINAMATH_CALUDE_number_of_children_l605_60504


namespace NUMINAMATH_CALUDE_zero_function_equals_derivative_l605_60507

theorem zero_function_equals_derivative : ∃ f : ℝ → ℝ, ∀ x, f x = 0 ∧ (deriv f) x = f x := by
  sorry

end NUMINAMATH_CALUDE_zero_function_equals_derivative_l605_60507


namespace NUMINAMATH_CALUDE_rectangular_prism_width_l605_60515

theorem rectangular_prism_width 
  (length : ℝ) 
  (height : ℝ) 
  (diagonal : ℝ) 
  (width : ℝ) 
  (h1 : length = 5) 
  (h2 : height = 8) 
  (h3 : diagonal = 10) 
  (h4 : diagonal ^ 2 = length ^ 2 + width ^ 2 + height ^ 2) : 
  width = Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_width_l605_60515


namespace NUMINAMATH_CALUDE_jerry_average_additional_hours_l605_60513

def tom_total_hours : ℝ := 10
def jerry_daily_differences : List ℝ := [-2, 1, -2, 2, 2, 1]

theorem jerry_average_additional_hours :
  let jerry_total_hours := tom_total_hours + jerry_daily_differences.sum
  let total_difference := jerry_total_hours - tom_total_hours
  let num_days := jerry_daily_differences.length
  total_difference / num_days = 1/3 := by sorry

end NUMINAMATH_CALUDE_jerry_average_additional_hours_l605_60513


namespace NUMINAMATH_CALUDE_equation_solutions_l605_60510

theorem equation_solutions (n : ℕ) : 
  (∃! (solutions : Finset (ℕ × ℕ × ℕ)), 
    solutions.card = 10 ∧ 
    ∀ (x y z : ℕ), (x, y, z) ∈ solutions ↔ 
      (x > 0 ∧ y > 0 ∧ z > 0 ∧ 4*x + 6*y + 2*z = n)) ↔ 
  (n = 32 ∨ n = 33) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l605_60510


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l605_60577

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l605_60577


namespace NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l605_60525

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 1) (a - 2)

-- Statement of the theorem
theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → z a = Complex.I * Complex.im (z a)) ∧
  (∃ a : ℝ, a ≠ 1 ∧ z a = Complex.I * Complex.im (z a)) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_one_sufficient_not_necessary_l605_60525


namespace NUMINAMATH_CALUDE_negative_sqrt_two_squared_l605_60506

theorem negative_sqrt_two_squared : (-Real.sqrt 2)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_two_squared_l605_60506


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l605_60544

/-- For a geometric sequence with common ratio -1/3, the ratio of the sum of odd-indexed terms
    to the sum of even-indexed terms (up to the 8th term) is -3. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : q = -1/3) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 1 + a 3 + a 5 + a 7) / (a 2 + a 4 + a 6 + a 8) = -3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l605_60544


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l605_60582

-- Problem 1
theorem problem_1 : 18 + (-12) + (-18) = -12 := by sorry

-- Problem 2
theorem problem_2 : (1 + 3 / 7) + (-(2 + 1 / 3)) + (2 + 4 / 7) + (-(1 + 2 / 3)) = 0 := by sorry

-- Problem 3
theorem problem_3 : (-1 / 12 - 1 / 36 + 1 / 6) * (-36) = -2 := by sorry

-- Problem 4
theorem problem_4 : -(1 ^ 2023) - ((-2) ^ 3) - ((-2) * (-3)) = 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l605_60582


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l605_60500

theorem fifth_power_sum_equality : ∃ n : ℕ+, 120^5 + 97^5 + 79^5 + 44^5 = n^5 ∧ n = 144 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l605_60500


namespace NUMINAMATH_CALUDE_max_share_is_18200_l605_60569

/-- Represents the profit share of a partner -/
structure PartnerShare where
  ratio : Nat
  bonus : Bool

/-- Calculates the maximum share given the total profit, bonus amount, and partner shares -/
def maxShare (totalProfit : ℚ) (bonusAmount : ℚ) (shares : List PartnerShare) : ℚ :=
  sorry

/-- The main theorem -/
theorem max_share_is_18200 :
  let shares := [
    ⟨4, false⟩,
    ⟨3, false⟩,
    ⟨2, true⟩,
    ⟨6, false⟩
  ]
  maxShare 45000 500 shares = 18200 := by sorry

end NUMINAMATH_CALUDE_max_share_is_18200_l605_60569


namespace NUMINAMATH_CALUDE_minimum_total_tests_l605_60566

/-- Represents the test data for a student -/
structure StudentData where
  name : String
  numTests : ℕ
  avgScore : ℕ
  totalScore : ℕ

/-- The problem statement -/
theorem minimum_total_tests (k m r : StudentData) : 
  k.name = "Michael K" →
  m.name = "Michael M" →
  r.name = "Michael R" →
  k.avgScore = 90 →
  m.avgScore = 91 →
  r.avgScore = 92 →
  k.numTests > m.numTests →
  m.numTests > r.numTests →
  m.totalScore > r.totalScore →
  r.totalScore > k.totalScore →
  k.totalScore = k.numTests * k.avgScore →
  m.totalScore = m.numTests * m.avgScore →
  r.totalScore = r.numTests * r.avgScore →
  k.numTests + m.numTests + r.numTests ≥ 413 :=
by sorry

end NUMINAMATH_CALUDE_minimum_total_tests_l605_60566


namespace NUMINAMATH_CALUDE_inequality_solution_range_l605_60593

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, abs (x + 2) - abs (x + 3) > m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l605_60593


namespace NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l605_60537

theorem geometric_arithmetic_geometric_sequence :
  ∀ a q : ℝ,
  (∀ x y z : ℝ, x = a ∧ y = a * q ∧ z = a * q^2 →
    (2 * (a * q + 8) = a + a * q^2) ∧
    ((a * q + 8)^2 = a * (a * q^2 + 64))) →
  ((a = 4 ∧ q = 3) ∨ (a = 4/9 ∧ q = -5)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_geometric_sequence_l605_60537


namespace NUMINAMATH_CALUDE_find_other_number_l605_60599

theorem find_other_number (A B : ℕ) (h1 : A = 24) (h2 : Nat.gcd A B = 17) (h3 : Nat.lcm A B = 312) :
  B = 221 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l605_60599


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l605_60503

-- Define the sets A and B
def A : Set ℝ := {x | x + 1 < 0}
def B : Set ℝ := {x | x - 3 < 0}

-- Define the complement of A in ℝ
def complementA : Set ℝ := {x | x ∉ A}

-- State the theorem
theorem complement_A_intersect_B :
  complementA ∩ B = {x : ℝ | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l605_60503


namespace NUMINAMATH_CALUDE_opposite_to_gold_is_yellow_l605_60521

/-- Represents the colors used on the cube faces -/
inductive Color
  | Blue
  | Yellow
  | Orange
  | Black
  | Silver
  | Gold

/-- Represents the positions of faces on the cube -/
inductive Position
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

/-- Represents a view of the cube, showing top, front, and right faces -/
structure CubeView where
  top : Color
  front : Color
  right : Color

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Position → Color

/-- The three views of the cube given in the problem -/
def givenViews : List CubeView := [
  { top := Color.Blue, front := Color.Yellow, right := Color.Orange },
  { top := Color.Blue, front := Color.Black,  right := Color.Orange },
  { top := Color.Blue, front := Color.Silver, right := Color.Orange }
]

/-- Theorem stating that the face opposite to gold is yellow -/
theorem opposite_to_gold_is_yellow (cube : Cube) 
    (h1 : ∀ view ∈ givenViews, 
      cube.faces Position.Top = view.top ∧ 
      cube.faces Position.Right = view.right ∧ 
      (cube.faces Position.Front = view.front ∨ 
       cube.faces Position.Left = view.front ∨ 
       cube.faces Position.Bottom = view.front))
    (h2 : ∃! pos, cube.faces pos = Color.Gold) :
    cube.faces Position.Front = Color.Yellow :=
  sorry

end NUMINAMATH_CALUDE_opposite_to_gold_is_yellow_l605_60521


namespace NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l605_60585

theorem consecutive_four_product_plus_one_is_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_four_product_plus_one_is_square_l605_60585


namespace NUMINAMATH_CALUDE_union_of_sets_l605_60517

theorem union_of_sets : 
  let A : Set Int := {-1, 1, 2, 4}
  let B : Set Int := {-1, 0, 2}
  A ∪ B = {-1, 0, 1, 2, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l605_60517


namespace NUMINAMATH_CALUDE_triangular_intersection_solids_l605_60592

-- Define the types for geometric solids and plane
inductive GeometricSolid
| Cone
| Cylinder
| Pyramid
| Cube

structure Plane

-- Define the intersection of a plane and a geometric solid
def Intersection (p : Plane) (s : GeometricSolid) : Set (ℝ × ℝ × ℝ) := sorry

-- Define what it means for an intersection to be triangular
def IsTriangularIntersection (i : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem triangular_intersection_solids 
  (p : Plane) (s : GeometricSolid) 
  (h : IsTriangularIntersection (Intersection p s)) : 
  s = GeometricSolid.Cone ∨ s = GeometricSolid.Pyramid ∨ s = GeometricSolid.Cube := by
  sorry


end NUMINAMATH_CALUDE_triangular_intersection_solids_l605_60592


namespace NUMINAMATH_CALUDE_two_trains_problem_l605_60557

/-- Two trains problem -/
theorem two_trains_problem (train_length : ℝ) (time_first_train : ℝ) (time_crossing : ℝ) :
  train_length = 120 →
  time_first_train = 12 →
  time_crossing = 16 →
  ∃ time_second_train : ℝ,
    time_second_train = 24 ∧
    train_length / time_first_train + train_length / time_second_train = 2 * train_length / time_crossing :=
by sorry

end NUMINAMATH_CALUDE_two_trains_problem_l605_60557


namespace NUMINAMATH_CALUDE_equation_solution_l605_60571

theorem equation_solution : ∃ x : ℝ, (12 - 2 * x = 6) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l605_60571


namespace NUMINAMATH_CALUDE_michaela_needs_20_oranges_l605_60574

/-- The number of oranges Michaela needs to eat until she gets full. -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to eat until she gets full. -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked. -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after both ate until full. -/
def remaining_oranges : ℕ := 30

/-- Proves that Michaela needs 20 oranges to get full given the conditions. -/
theorem michaela_needs_20_oranges : 
  michaela_oranges = 20 ∧ 
  cassandra_oranges = 2 * michaela_oranges ∧
  total_oranges = 90 ∧
  remaining_oranges = 30 ∧
  total_oranges = michaela_oranges + cassandra_oranges + remaining_oranges :=
by sorry

end NUMINAMATH_CALUDE_michaela_needs_20_oranges_l605_60574


namespace NUMINAMATH_CALUDE_min_value_sum_products_l605_60586

theorem min_value_sum_products (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = a * b * c) (h2 : a + b + c = a^3) :
  ∀ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x + y + z = x * y * z ∧ x + y + z = x^3 →
  x * y + y * z + z * x ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_products_l605_60586


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l605_60579

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌊(-34 : ℝ) / 4⌋⌋ = 2 :=
by sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l605_60579


namespace NUMINAMATH_CALUDE_tan_45_deg_eq_one_l605_60508

/-- Tangent of 45 degrees is 1 -/
theorem tan_45_deg_eq_one :
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_deg_eq_one_l605_60508


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l605_60550

theorem angle_sum_is_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l605_60550


namespace NUMINAMATH_CALUDE_coefficient_of_y_is_one_l605_60535

/-- A line passing through two points with a given equation -/
structure Line where
  m : ℝ
  n : ℝ
  p : ℝ
  equation : ℝ → ℝ → Prop

/-- The line satisfies the given conditions -/
def line_satisfies_conditions (L : Line) : Prop :=
  L.p = 0.6666666666666666 ∧
  ∀ x y, L.equation x y ↔ x = y + 5

/-- The coefficient of y in the line equation is 1 -/
theorem coefficient_of_y_is_one (L : Line) 
  (h : line_satisfies_conditions L) : 
  ∃ b : ℝ, ∀ x y, L.equation x y ↔ y = x + b :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_is_one_l605_60535


namespace NUMINAMATH_CALUDE_sum_of_five_variables_l605_60588

theorem sum_of_five_variables (a b c d e : ℚ) : 
  (a + 1 = b + 2) ∧ 
  (b + 2 = c + 3) ∧ 
  (c + 3 = d + 4) ∧ 
  (d + 4 = e + 5) ∧ 
  (e + 5 = a + b + c + d + e + 10) → 
  a + b + c + d + e = -35/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_variables_l605_60588


namespace NUMINAMATH_CALUDE_data_set_range_is_67_l605_60587

-- Define a structure for our data set
structure DataSet where
  points : List ℝ
  min_value : ℝ
  max_value : ℝ
  h_min : min_value ∈ points
  h_max : max_value ∈ points
  h_lower_bound : ∀ x ∈ points, min_value ≤ x
  h_upper_bound : ∀ x ∈ points, x ≤ max_value

-- Define the range of a data set
def range (d : DataSet) : ℝ := d.max_value - d.min_value

-- Theorem statement
theorem data_set_range_is_67 (d : DataSet) 
  (h_min : d.min_value = 31)
  (h_max : d.max_value = 98) : 
  range d = 67 := by
  sorry

end NUMINAMATH_CALUDE_data_set_range_is_67_l605_60587


namespace NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l605_60530

theorem pencil_eraser_cost_problem :
  ∃ (p e : ℕ), 
    p > 0 ∧ 
    e > 0 ∧ 
    10 * p + 2 * e = 110 ∧ 
    p < e ∧ 
    p + e = 19 :=
by sorry

end NUMINAMATH_CALUDE_pencil_eraser_cost_problem_l605_60530


namespace NUMINAMATH_CALUDE_horner_method_v₃_l605_60565

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

def horner_v₃ (x v v₁ : ℝ) : ℝ :=
  let v₂ := v₁ * x - 4
  v₂ * x + 3

theorem horner_method_v₃ :
  let x : ℝ := 5
  let v : ℝ := 2
  let v₁ : ℝ := 5
  horner_v₃ x v v₁ = 108 := by sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l605_60565


namespace NUMINAMATH_CALUDE_length_XX₁_l605_60562

/-- Configuration of two right triangles with angle bisectors -/
structure TriangleConfig where
  -- Triangle DEF
  DE : ℝ
  DF : ℝ
  hDE : DE = 13
  hDF : DF = 5
  hDEF_right : DE^2 = DF^2 + EF^2
  
  -- D₁ is on EF such that ∠FDD₁ = ∠EDD₁
  D₁F : ℝ
  D₁E : ℝ
  hD₁_on_EF : D₁F + D₁E = EF
  hD₁_bisector : D₁F / D₁E = DF / EF
  
  -- Triangle XYZ
  XY : ℝ
  XZ : ℝ
  hXY : XY = D₁E
  hXZ : XZ = D₁F
  hXYZ_right : XY^2 = XZ^2 + YZ^2
  
  -- X₁ is on YZ such that ∠ZXX₁ = ∠YXX₁
  X₁Z : ℝ
  X₁Y : ℝ
  hX₁_on_YZ : X₁Z + X₁Y = YZ
  hX₁_bisector : X₁Z / X₁Y = XZ / XY

/-- The length of XX₁ in the given configuration is 20/17 -/
theorem length_XX₁ (config : TriangleConfig) : X₁Z = 20/17 := by
  sorry

end NUMINAMATH_CALUDE_length_XX₁_l605_60562


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l605_60527

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l605_60527


namespace NUMINAMATH_CALUDE_afternoon_bundles_burned_eq_three_l605_60561

/-- Given the number of wood bundles burned in the morning, at the start of the day, and at the end of the day, 
    calculate the number of wood bundles burned in the afternoon. -/
def afternoon_bundles_burned (morning_burned start_of_day end_of_day : ℕ) : ℕ :=
  (start_of_day - end_of_day) - morning_burned

theorem afternoon_bundles_burned_eq_three : 
  afternoon_bundles_burned 4 10 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_bundles_burned_eq_three_l605_60561


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l605_60543

theorem modular_arithmetic_problem :
  ∃ (a b : ℕ), a < 65 ∧ b < 65 ∧ (4 * a) % 65 = 1 ∧ (13 * b) % 65 = 1 ∧
  (3 * a + 7 * b) % 65 = 47 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l605_60543
